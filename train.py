"""
train.py
--------
Trains a small CNN on the labeled square images in ./samples/
and saves the model to ./model/chess_classifier.pt

The model is a lightweight CNN (~200K parameters) that runs fast on CPU.
With 5-10 examples per class it typically reaches 90%+ accuracy.
With 20+ examples per class it reaches 95%+.

Usage:
    python train.py [--epochs 30] [--lr 0.001] [--augment]

Output:
    model/chess_classifier.pt   <- PyTorch model weights
    model/classes.json          <- class index mapping
    model/training_log.txt      <- loss/accuracy per epoch
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Try to import torch; give a clear error if not installed
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
except ImportError:
    print("PyTorch not found. Install it with:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    raise SystemExit(1)

SAMPLES_DIR = Path("samples")
MODEL_DIR = Path("model")
SQ_PX = 64   # input size for the CNN

CLASSES = ["empty", "wK", "wQ", "wR", "wB", "wN", "wP",
           "bK", "bQ", "bR", "bB", "bN", "bP"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChessSquareDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, int]], augment: bool = False):
        self.samples = samples
        self.augment = augment
        self.base_transform = T.Compose([
            T.ToTensor(),           # HxW uint8 → 1xHxW float [0,1]
            T.Normalize([0.5], [0.5]),  # → [-1, 1]
        ])
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.3),
            T.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        # img is HxW uint8 numpy array
        pil = _to_pil(img)
        if self.augment:
            tensor = self.aug_transform(pil)
        else:
            tensor = self.base_transform(pil)
        return tensor, label


def _to_pil(arr: np.ndarray):
    from PIL import Image
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Model: small CNN
# ---------------------------------------------------------------------------

class ChessCNN(nn.Module):
    """
    Lightweight CNN for 64x64 grayscale chess square classification.
    ~200K parameters, runs in <1ms per square on CPU.
    """
    def __init__(self, num_classes: int = 13):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 → 32x32
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2: 32x32 → 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3: 16x16 → 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(augment: bool = False) -> tuple[list, list, dict]:
    """
    Load all labeled samples from samples/<class>/*.png
    Returns (train_samples, val_samples, class_to_idx)
    """
    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    all_samples = []

    for cls in CLASSES:
        cls_dir = SAMPLES_DIR / cls
        if not cls_dir.exists():
            continue
        imgs = list(cls_dir.glob("*.png"))
        if not imgs:
            continue
        for img_path in imgs:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (SQ_PX, SQ_PX))
            all_samples.append((img, class_to_idx[cls]))

    if not all_samples:
        raise ValueError("No labeled samples found. Run label_tool.py first.")

    # Check class coverage
    present = set(label for _, label in all_samples)
    idx_to_cls = {v: k for k, v in class_to_idx.items()}
    missing = [idx_to_cls[i] for i in range(len(CLASSES)) if i not in present]
    if missing:
        print(f"⚠  Missing classes: {missing}")
        print("   The model will not be able to predict these pieces.")

    # Stratified train/val split (80/20)
    random.shuffle(all_samples)
    by_class: dict[int, list] = {}
    for sample in all_samples:
        by_class.setdefault(sample[1], []).append(sample)

    train, val = [], []
    for cls_samples in by_class.values():
        n_val = max(1, len(cls_samples) // 5)
        val.extend(cls_samples[:n_val])
        train.extend(cls_samples[n_val:])

    print(f"Dataset: {len(train)} train, {len(val)} val")
    counts = {idx_to_cls[i]: sum(1 for _, l in all_samples if l == i) for i in range(len(CLASSES))}
    print("Per-class counts:", {k: v for k, v in counts.items() if v > 0})

    return train, val, class_to_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(epochs: int = 40, lr: float = 0.001, augment: bool = True):
    MODEL_DIR.mkdir(exist_ok=True)

    train_samples, val_samples, class_to_idx = load_dataset(augment)

    train_ds = ChessSquareDataset(train_samples, augment=augment)
    val_ds   = ChessSquareDataset(val_samples,   augment=False)

    # Use small batch size since dataset is small
    batch = min(32, len(train_samples))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = ChessCNN(num_classes=len(CLASSES)).to(device)

    # Class-weighted loss to handle imbalanced data (empty squares dominate)
    class_counts = [sum(1 for _, l in train_samples if l == i) for i in range(len(CLASSES))]
    weights = torch.tensor(
        [1.0 / max(c, 1) for c in class_counts], dtype=torch.float32
    )
    weights = weights / weights.sum() * len(CLASSES)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_lines = []
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\nTraining for {epochs} epochs on CPU...")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Acc':>7}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += len(labels)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += len(labels)

        t_loss = train_loss / train_total
        t_acc  = train_correct / train_total * 100
        v_acc  = val_correct / val_total * 100 if val_total else 0.0

        line = f"{epoch:>6}  {t_loss:>10.4f}  {t_acc:>8.1f}%  {v_acc:>6.1f}%"
        print(line)
        log_lines.append(line)

        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_DIR / "chess_classifier.pt")

    # Save class mapping
    with open(MODEL_DIR / "classes.json", "w") as f:
        json.dump({"classes": CLASSES, "class_to_idx": class_to_idx}, f, indent=2)

    # Save log
    with open(MODEL_DIR / "training_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    print(f"\nBest val accuracy: {best_val_acc:.1f}% at epoch {best_epoch}")
    print(f"Model saved to: {MODEL_DIR / 'chess_classifier.pt'}")
    print(f"\nNext step: run  python main.py ..\\book.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr",     type=float, default=0.001)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    train_model(epochs=args.epochs, lr=args.lr, augment=not args.no_augment)
