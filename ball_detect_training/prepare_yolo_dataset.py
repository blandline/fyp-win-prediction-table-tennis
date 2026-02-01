"""
Automatically prepares YOLOv8 dataset:
- Collects images from dataset/training/images/game_*
- Matches YOLO labels from output/game_1
- Splits into train / val / test
"""

import random
import shutil
from pathlib import Path

# Paths
SOURCE_IMAGES_ROOT = Path("dataset/training/images")
SOURCE_LABELS_ROOT = Path("output/game_1")

TARGET_ROOT = Path("yolo_dataset")
IMG_TRAIN = TARGET_ROOT / "images/train"
IMG_VAL = TARGET_ROOT / "images/val"
IMG_TEST = TARGET_ROOT / "images/test"

LBL_TRAIN = TARGET_ROOT / "labels/train"
LBL_VAL = TARGET_ROOT / "labels/val"
LBL_TEST = TARGET_ROOT / "labels/test"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Create directories
for d in [IMG_TRAIN, IMG_VAL, IMG_TEST, LBL_TRAIN, LBL_VAL, LBL_TEST]:
    d.mkdir(parents=True, exist_ok=True)

def main():
    image_label_pairs = []

    # Collect images from all game_* folders
    for game_dir in SOURCE_IMAGES_ROOT.iterdir():
        if not game_dir.is_dir():
            continue

        for img_path in game_dir.glob("*.jpg"):
            label_path = SOURCE_LABELS_ROOT / img_path.with_suffix(".txt").name
            if label_path.exists():
                image_label_pairs.append((img_path, label_path))

    print(f"Found {len(image_label_pairs)} image-label pairs")

    random.shuffle(image_label_pairs)

    n_total = len(image_label_pairs)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_set = image_label_pairs[:n_train]
    val_set = image_label_pairs[n_train:n_train + n_val]
    test_set = image_label_pairs[n_train + n_val:]

    def copy_pairs(pairs, img_dst, lbl_dst):
        for img, lbl in pairs:
            shutil.copy(img, img_dst / img.name)
            shutil.copy(lbl, lbl_dst / lbl.name)

    copy_pairs(train_set, IMG_TRAIN, LBL_TRAIN)
    copy_pairs(val_set, IMG_VAL, LBL_VAL)
    copy_pairs(test_set, IMG_TEST, LBL_TEST)

    print("✅ Dataset prepared successfully")
    print(f"Train: {len(train_set)}")
    print(f"Val:   {len(val_set)}")
    print(f"Test:  {len(test_set)}")

if __name__ == "__main__":
    main()
