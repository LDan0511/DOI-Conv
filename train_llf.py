import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.model import SmallDOINet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_INPUT_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise\train\data"
TRAIN_TARGET_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise\train\label"
VAL_INPUT_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise\val\data"
VAL_TARGET_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise\val\label"
TEST_INPUT_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise\test\data"
TEST_TARGET_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise\test\label"

MICH_SHAPE = (155, 78)
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
NUM_WORKERS = 0
SAVE_FREQUENCY = 50

SAVE_DIR = r"C:\Users\hanst\Desktop\DeepLearning\newcode\33conv\checkpoints"
BEST_SAVE_PATH = os.path.join(SAVE_DIR, "small_doi_best.pth")


class MichDataset(Dataset):
    def __init__(self, input_dir, target_dir, shape):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.shape = shape

        input_files = [f for f in os.listdir(input_dir) if f.endswith(".dat")]
        input_bases = sorted([os.path.splitext(f)[0] for f in input_files])

        self.bases = []
        for base in input_bases:
            target_path = os.path.join(target_dir, base + ".img")
            if os.path.exists(target_path):
                self.bases.append(base)

        if len(self.bases) == 0:
            raise ValueError("没有找到可配对的 .dat 和 .img 文件")

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        base = self.bases[idx]

        input_path = os.path.join(self.input_dir, base + ".dat")
        target_path = os.path.join(self.target_dir, base + ".img")

        x = np.fromfile(input_path, dtype=np.float32)
        y = np.fromfile(target_path, dtype=np.float32)

        expected_size = self.shape[0] * self.shape[1]

        if x.size != expected_size:
            raise ValueError(f"{input_path} size error")
        if y.size != expected_size:
            raise ValueError(f"{target_path} size error")

        x = x.reshape(self.shape)
        y = y.reshape(self.shape)

        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0)

        return x, y


# ===== 核心：总和守恒约束 =====
def normalize_sum(pred, target):
    target_sum = target.sum(dim=(-1, -2), keepdim=True)
    pred_sum = pred.sum(dim=(-1, -2), keepdim=True)

    pred = pred * (target_sum / (pred_sum + 1e-12))

    pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
    return pred


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        # 关键修改2：强制总和一致
        pred = normalize_sum(pred, y)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        pred = normalize_sum(pred, y)

        loss = criterion(pred, y)

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_dataset = MichDataset(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, MICH_SHAPE)
    val_dataset = MichDataset(VAL_INPUT_DIR, VAL_TARGET_DIR, MICH_SHAPE)
    test_dataset = MichDataset(TEST_INPUT_DIR, TEST_TARGET_DIR, MICH_SHAPE)

    print("train samples =", len(train_dataset))
    print("val samples   =", len(val_dataset))
    print("test samples  =", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = SmallDOINet().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_SAVE_PATH)
            print(f"save best model to {BEST_SAVE_PATH}")

        if epoch % SAVE_FREQUENCY == 0:
            path = os.path.join(SAVE_DIR, f"small_doi_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), path)
            print(f"save checkpoint to {path}")

    print("Training finished.")



if __name__ == "__main__":
    main()