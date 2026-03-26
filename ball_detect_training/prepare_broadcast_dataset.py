"""
Merge broadcast-annotated frames with the original side-view dataset
into a single YOLO dataset for fine-tuning.

Supports two broadcast annotation sources:
  1. Manual annotations in broadcast_annotations/ (your own labeled frames)
  2. Roboflow exports (may use multi-class labels like landing-point,
     negative-slope, etc. — all classes are remapped to 0 = table_tennis_ball)

Expected input structure:
  broadcast_annotations/
    images/   (annotated broadcast .jpg files)
    labels/   (YOLO .txt label files, same basenames)

  roboflow_dataset/       (optional, Roboflow export in YOLOv8 format)
    train/images/  train/labels/
    valid/images/  valid/labels/
    test/images/   test/labels/

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
ROBOFLOW_DATASET = Path("roboflow_dataset")  # optional Roboflow YOLOv8 export
ORIGINAL_DATASET = Path("yolo_dataset")

# Output
TARGET_ROOT = Path("broadcast_yolo_dataset")

# Split ratios for broadcast data (only for manual annotations)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Image extensions to look for
IMG_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def collect_pairs(images_dir, labels_dir):
    """Collect (image_path, label_path) pairs where both exist."""
    pairs = []
    for ext in IMG_EXTENSIONS:
        for img in sorted(images_dir.glob(ext)):
            lbl = labels_dir / img.with_suffix(".txt").name
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def copy_label_remapped(src_label, dst_label):
    """Copy a YOLO label file, remapping all class IDs to 0.

    Roboflow datasets may use multiple classes (e.g. landing-point=0,
    negative-slope=1, positive-slope=2, etc.). Since we only care about
    ball location, we remap everything to class 0 = table_tennis_ball.
    """
    lines = src_label.read_text().strip().splitlines()
    remapped = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Replace class ID with 0, keep bbox coordinates
            parts[0] = "0"
            remapped.append(" ".join(parts))
    dst_label.write_text("\n".join(remapped) + "\n" if remapped else "")


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

    # Step 2: Add Roboflow dataset (if present) — remap all classes to 0
    roboflow_count = 0
    # Roboflow YOLOv8 exports use: train/, valid/, test/ with images/ + labels/ inside
    roboflow_split_map = {"train": "train", "valid": "val", "test": "test"}
    if ROBOFLOW_DATASET.exists():
        for rf_split, our_split in roboflow_split_map.items():
            rf_img_dir = ROBOFLOW_DATASET / rf_split / "images"
            rf_lbl_dir = ROBOFLOW_DATASET / rf_split / "labels"
            if not rf_img_dir.exists():
                continue
            pairs = collect_pairs(rf_img_dir, rf_lbl_dir)
            dst_img = TARGET_ROOT / "images" / our_split
            dst_lbl = TARGET_ROOT / "labels" / our_split
            for img, lbl in pairs:
                # Prefix to avoid name collisions with other sources
                out_name = f"rf_{img.name}"
                shutil.copy(img, dst_img / out_name)
                copy_label_remapped(lbl, dst_lbl / f"rf_{lbl.name}")
                roboflow_count += 1
        print(f"Added {roboflow_count} Roboflow images (all classes remapped to 0)")
    else:
        print(f"No Roboflow dataset found at {ROBOFLOW_DATASET} (skipping)")

    # Step 3: Split and copy manual broadcast annotations
    broadcast_pairs = collect_pairs(BROADCAST_IMAGES, BROADCAST_LABELS)
    broadcast_count = len(broadcast_pairs)
    if broadcast_pairs:
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
                copy_label_remapped(lbl, dst_lbl / lbl.name)

        print(f"Added {n} manual broadcast images:")
        for split, pairs in split_assignments.items():
            print(f"  {split}: {len(pairs)}")
    else:
        print(f"No manual broadcast annotations found in {BROADCAST_IMAGES} (skipping)")

    if roboflow_count == 0 and broadcast_count == 0:
        print("\nWarning: No broadcast data was added — only original dataset was copied.")

    # Count totals
    for split in splits:
        img_count = sum(1 for ext in IMG_EXTENSIONS
                        for _ in (TARGET_ROOT / "images" / split).glob(ext))
        lbl_count = len(list((TARGET_ROOT / "labels" / split).glob("*.txt")))
        print(f"Total {split}: {img_count} images, {lbl_count} labels")

    print(f"\nMerged dataset ready at: {TARGET_ROOT}")


if __name__ == "__main__":
    main()
