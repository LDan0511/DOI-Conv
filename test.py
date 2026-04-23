import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.model import SmallDOINet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_INPUT_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise\test\data"
TEST_TARGET_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise\test\label"

MICH_SHAPE = (155, 78)
BATCH_SIZE = 1
NUM_WORKERS = 0

CHECKPOINT_PATH = r"C:\Users\hanst\Desktop\DeepLearning\newcode\33conv\checkpoints\small_doi_best.pth"
OUTPUT_DIR = r"C:\Users\hanst\Desktop\DeepLearning\newcode\33conv\results"

SAVE_PRED = True   # 是否保存预测结果


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
            raise ValueError("没有找到可配对的 .dat 和 .img 文件。")

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
            raise ValueError(f"{input_path} 元素数不对，读到 {x.size}，期望 {expected_size}")
        if y.size != expected_size:
            raise ValueError(f"{target_path} 元素数不对，读到 {y.size}，期望 {expected_size}")

        x = x.reshape(self.shape)
        y = y.reshape(self.shape)

        x = torch.from_numpy(x).float().unsqueeze(0)   # [1, H, W]
        y = torch.from_numpy(y).float().unsqueeze(0)

        return x, y, base


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for x, y, base in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()

        print(f"{base[0]} | loss = {loss.item():.6f}")

        if SAVE_PRED:
            pred_np = pred.squeeze().cpu().numpy().astype(np.float32)
            save_path = os.path.join(OUTPUT_DIR, base[0] + "_pred.mich")
            pred_np.tofile(save_path)

    return total_loss / max(len(loader), 1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_dataset = MichDataset(TEST_INPUT_DIR, TEST_TARGET_DIR, MICH_SHAPE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("test samples =", len(test_dataset))

    model = SmallDOINet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    criterion = nn.L1Loss()

    test_loss = evaluate(model, test_loader, criterion)
    print("===================================")
    print("Final Test Loss =", test_loss)


if __name__ == "__main__":
    main()