"""
Merge broadcast-annotated frames with the original side-view dataset
into a single YOLO dataset for fine-tuning.

Expected input structure:
  broadcast_annotations/
    images/   (annotated broadcast .jpg files)
    labels/   (YOLO .txt label files, same basenames)

  yolo_dataset/           (original side-view dataset)
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/

Output:
  broadcast_yolo_dataset/
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
"""

import random
import shutil
from pathlib import Path

# Input paths
BROADCAST_IMAGES = Path("broadcast_annotations/images")
BROADCAST_LABELS = Path("broadcast_annotations/labels")
ORIGINAL_DATASET = Path("yolo_dataset")

# Output
TARGET_ROOT = Path("broadcast_yolo_dataset")

# Split ratios for broadcast data
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def collect_pairs(images_dir, labels_dir):
    """Collect (image_path, label_path) pairs where both exist."""
    pairs = []
    for img in sorted(images_dir.glob("*.jpg")):
        lbl = labels_dir / img.with_suffix(".txt").name
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def main():
    # Create output directories
    splits = ["train", "val", "test"]
    for split in splits:
        (TARGET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Step 1: Copy original dataset
    copied_original = 0
    for split in splits:
        src_img = ORIGINAL_DATASET / "images" / split
        src_lbl = ORIGINAL_DATASET / "labels" / split
        dst_img = TARGET_ROOT / "images" / split
        dst_lbl = TARGET_ROOT / "labels" / split

        if src_img.exists():
            for f in src_img.glob("*.jpg"):
                shutil.copy(f, dst_img / f.name)
                copied_original += 1
        if src_lbl.exists():
            for f in src_lbl.glob("*.txt"):
                shutil.copy(f, dst_lbl / f.name)

    print(f"Copied {copied_original} original images")

    # Step 2: Split and copy broadcast data
    broadcast_pairs = collect_pairs(BROADCAST_IMAGES, BROADCAST_LABELS)
    if not broadcast_pairs:
        print(f"Warning: No broadcast annotation pairs found in {BROADCAST_IMAGES} + {BROADCAST_LABELS}")
        print("Expected: broadcast_annotations/images/*.jpg + broadcast_annotations/labels/*.txt")
        return

    random.shuffle(broadcast_pairs)
    n = len(broadcast_pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    split_assignments = {
        "train": broadcast_pairs[:n_train],
        "val": broadcast_pairs[n_train:n_train + n_val],
        "test": broadcast_pairs[n_train + n_val:],
    }

    for split, pairs in split_assignments.items():
        dst_img = TARGET_ROOT / "images" / split
        dst_lbl = TARGET_ROOT / "labels" / split
        for img, lbl in pairs:
            shutil.copy(img, dst_img / img.name)
            shutil.copy(lbl, dst_lbl / lbl.name)

    print(f"Added {n} broadcast images:")
    for split, pairs in split_assignments.items():
        print(f"  {split}: {len(pairs)}")

    # Count totals
    for split in splits:
        img_count = len(list((TARGET_ROOT / "images" / split).glob("*.jpg")))
        lbl_count = len(list((TARGET_ROOT / "labels" / split).glob("*.txt")))
        print(f"Total {split}: {img_count} images, {lbl_count} labels")

    print(f"\nMerged dataset ready at: {TARGET_ROOT}")


if __name__ == "__main__":
    main()
