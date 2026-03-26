"""
Combine all Roboflow YOLOv8 ball detection exports into a single dataset
for fine-tuning the ball detector from an existing checkpoint.

Why Roboflow-only (no OpenTT data):
  The existing ball_detector checkpoint was trained on ~35K OpenTT side-view
  images. Fine-tuning on only the ~585 Roboflow images lets the model adapt to
  new camera angles without the 60:1 ratio drowning out the new data.

Auto-discovers all Roboflow exports under dataset/ (any subdirectory that
contains a data.yaml file). Pools all images across exports and splits,
then reshuffles and re-splits 80/10/10 -- necessary because 6 of 7 exports
have no val/test splits of their own.

All class IDs are remapped to 0 (table_tennis_ball). The Roboflow exports use
multi-class labels (landing-point, negative-slope, other, positive-slope,
zero-slope) which all represent ball positions.

Output:
  roboflow_combined_dataset/
    images/{train,val,test}/
    labels/{train,val,test}/
"""

import random
import shutil
from pathlib import Path

DATASET_ROOT = Path("dataset")
OUTPUT_ROOT = Path("roboflow_combined_dataset")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1 (remainder)

IMG_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

# Roboflow exports use "valid" not "val"; all three possible split names
RF_SPLITS = ("train", "valid", "test")


def discover_roboflow_exports(dataset_root: Path) -> list[Path]:
    """Return subdirs of dataset_root that look like Roboflow YOLOv8 exports.

    A Roboflow export is identified by having a data.yaml file directly inside
    it. The OpenTT dirs (training/, test/) don't have data.yaml files.
    """
    exports = []
    for entry in sorted(dataset_root.iterdir()):
        if entry.is_dir() and (entry / "data.yaml").exists():
            exports.append(entry)
    return exports


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs where both files exist."""
    pairs = []
    for ext in IMG_EXTENSIONS:
        for img in sorted(images_dir.glob(ext)):
            lbl = labels_dir / img.with_suffix(".txt").name
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def write_label_remapped(src_label: Path, dst_label: Path) -> None:
    """Write a YOLO label file with all class IDs replaced by 0.

    Roboflow exports may use multiple classes (landing-point, negative-slope,
    positive-slope, zero-slope, other). Since we only care about ball location,
    everything is remapped to class 0 = table_tennis_ball.
    """
    lines = src_label.read_text().strip().splitlines()
    remapped = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            parts[0] = "0"
            remapped.append(" ".join(parts))
    dst_label.write_text("\n".join(remapped) + "\n" if remapped else "")


def main() -> None:
    exports = discover_roboflow_exports(DATASET_ROOT)
    if not exports:
        print(f"No Roboflow exports found under {DATASET_ROOT}. Exiting.")
        return

    print(f"Found {len(exports)} Roboflow export(s):")

    all_pairs: list[tuple[Path, Path, str]] = []  # (img, lbl, prefix)

    for idx, export_dir in enumerate(exports):
        prefix = f"rf{idx}_"
        source_count = 0

        for split in RF_SPLITS:
            img_dir = export_dir / split / "images"
            lbl_dir = export_dir / split / "labels"
            if not img_dir.exists():
                continue
            pairs = collect_pairs(img_dir, lbl_dir)
            for img, lbl in pairs:
                all_pairs.append((img, lbl, prefix))
            source_count += len(pairs)

        print(f"  [{idx}] {export_dir.name}: {source_count} images")

    total = len(all_pairs)
    print(f"\nTotal images collected: {total}")

    if total == 0:
        print("No images found. Check that the export directories contain images and labels.")
        return

    # Create output directories
    for split in ("train", "val", "test"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Pool and reshuffle, then split 80/10/10
    random.seed(42)
    random.shuffle(all_pairs)

    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)

    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:],
    }

    for split_name, pairs in splits.items():
        img_out = OUTPUT_ROOT / "images" / split_name
        lbl_out = OUTPUT_ROOT / "labels" / split_name
        for img, lbl, prefix in pairs:
            out_img_name = prefix + img.name
            out_lbl_name = prefix + lbl.name
            shutil.copy(img, img_out / out_img_name)
            write_label_remapped(lbl, lbl_out / out_lbl_name)

    print("\nDataset split:")
    for split_name, pairs in splits.items():
        print(f"  {split_name}: {len(pairs)} images")

    print(f"\nCombined Roboflow dataset ready at: {OUTPUT_ROOT}")
    print("Use ball_roboflow.yaml to fine-tune from your existing checkpoint.")


if __name__ == "__main__":
    main()
