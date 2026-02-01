import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models.segmentation as segm

# =========================
# CONFIG
# =========================
ROOT = Path("segmentation_data")

IMG_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks_rgb"
SPLIT_DIR = ROOT / "splits"

NUM_CLASSES = 4
BATCH_SIZE = 6          # safe for RTX 4000 Ada
EPOCHS = 40
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = ROOT / "runs/deeplabv3_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CORRECT COLOR MAP
# =========================
# Anything NOT listed here becomes background
COLOR_MAP = {
    (0, 0, 0): 0,        # background
    (0, 255, 0): 1,      # player
    (255, 0, 0): 2,      # table
    (0, 0, 255): 3       # scoreboard
}

# =========================
# MASK CONVERSION
# =========================
def rgb_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB mask to class-id mask.
    Unknown colors -> background (0).
    """
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for rgb, cls in COLOR_MAP.items():
        rgb_arr = np.array(rgb, dtype=np.uint8)
        matches = np.all(mask_rgb == rgb_arr, axis=-1)
        out[matches] = cls

    return out

# =========================
# DATASET
# =========================
class OpenTTSegmentationDataset(Dataset):
    def __init__(self, split_file: Path):
        self.ids = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]

        img_path = IMG_DIR / f"{stem}.jpg"
        mask_path = MASK_DIR / f"{stem}.png"

        # Read image (1920x1080)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask (lower resolution)
        mask = cv2.imread(str(mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mh, mw = mask.shape[:2]

        # 🔑 Resize IMAGE to MASK size
        img = cv2.resize(img, (mw, mh), interpolation=cv2.INTER_LINEAR)

        mask_ids = rgb_to_class(mask)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_ids).long()

        return img_t, mask_t

# =========================
# METRIC
# =========================
def mean_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / max(len(ious), 1)

# =========================
# TRAIN LOOP
# =========================
def main():
    print("Device:", DEVICE)

    train_ds = OpenTTSegmentationDataset(SPLIT_DIR / "train.txt")
    val_ds = OpenTTSegmentationDataset(SPLIT_DIR / "val.txt")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False
    )

    model = segm.deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model = model.to(DEVICE)

    # Scoreboard is small → weight it higher
    class_weights = torch.tensor([1.0, 2.0, 2.0, 4.0], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_miou = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ---- TRAIN ----
        model.train()
        total_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            out = model(imgs)["out"]
            loss = criterion(out, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # ---- VALIDATE ----
        model.eval()
        miou = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                out = model(imgs)["out"]
                pred = out.argmax(1)
                miou += mean_iou(pred, masks, NUM_CLASSES)

        miou /= max(len(val_loader), 1)

        print(f"Epoch {epoch}: loss={avg_loss:.4f}, mIoU={miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(
                {"model": model.state_dict()},
                OUT_DIR / "best.pt"
            )
            print("✅ Saved best.pt")

    torch.save({"model": model.state_dict()}, OUT_DIR / "last.pt")
    print("Training finished. Best mIoU:", best_miou)

if __name__ == "__main__":
    main()
