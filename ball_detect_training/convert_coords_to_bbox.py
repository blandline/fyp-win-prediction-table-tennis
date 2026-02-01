"""
Prepare a full YOLOv8 dataset from ALL OpenTTGames.

Steps:
1. Iterate through all game_* folders
2. Convert ball coordinates → YOLO bounding boxes
3. Collect all (image, label) pairs
4. Shuffle dataset
5. Split into train / val / test
"""

import json
import random
import shutil
from pathlib import Path
from PIL import Image

# =========================
# CONFIGURATION
# =========================
DATASET_ROOT = Path("dataset/training")
OUTPUT_ROOT = Path("yolo_dataset")

BALL_BBOX_SIZE = 40
BALL_CLASS_ID = 0

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# =========================
# OUTPUT DIRS
# =========================
IMG_TRAIN = OUTPUT_ROOT / "images/train"
IMG_VAL = OUTPUT_ROOT / "images/val"
IMG_TEST = OUTPUT_ROOT / "images/test"

LBL_TRAIN = OUTPUT_ROOT / "labels/train"
LBL_VAL = OUTPUT_ROOT / "labels/val"
LBL_TEST = OUTPUT_ROOT / "labels/test"

for d in [IMG_TRAIN, IMG_VAL, IMG_TEST, LBL_TRAIN, LBL_VAL, LBL_TEST]:
    d.mkdir(parents=True, exist_ok=True)


# =========================
# HELPERS
# =========================
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # (w, h)


def coord_to_yolo_bbox(x, y, img_width, img_height, bbox_size):
    half = bbox_size / 2.0

    x_min = max(0, x - half)
    y_min = max(0, y - half)
    x_max = min(img_width, x + half)
    y_max = min(img_height, y + half)

    cx = (x_min + x_max) / 2.0 / img_width
    cy = (y_min + y_max) / 2.0 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height

    return cx, cy, w, h


# =========================
# MAIN PIPELINE
# =========================
def main():
    all_samples = []

    images_root = DATASET_ROOT / "images"
    annotations_root = DATASET_ROOT / "annotations"

    for game_dir in sorted(images_root.iterdir()):
        if not game_dir.is_dir():
            continue

        game_name = game_dir.name
        print(f"\nProcessing {game_name}")

        ball_markup_path = annotations_root / game_name / "ball_markup.json"
        if not ball_markup_path.exists():
            print(f"  ❌ Missing annotations for {game_name}, skipping")
            continue

        with open(ball_markup_path, "r") as f:
            ball_coords = json.load(f)

        image_files = sorted(game_dir.glob("*.jpg"))

        for img_path in image_files:
            frame_num = img_path.stem.replace("img_", "").lstrip("0")
            frame_num = frame_num if frame_num else "0"

            if frame_num not in ball_coords:
                continue

            coords = ball_coords[frame_num]
            if "x" not in coords or "y" not in coords:
                continue

            img_w, img_h = get_image_size(img_path)
            bbox = coord_to_yolo_bbox(
                coords["x"],
                coords["y"],
                img_w,
                img_h,
                BALL_BBOX_SIZE
            )

            all_samples.append((img_path, bbox))

    print(f"\n✅ Total labeled samples: {len(all_samples)}")

    # =========================
    # SHUFFLE + SPLIT
    # =========================
    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_set = all_samples[:n_train]
    val_set = all_samples[n_train:n_train + n_val]
    test_set = all_samples[n_train + n_val:]

    def save_split(samples, img_out, lbl_out):
        for img_path, bbox in samples:
            shutil.copy(img_path, img_out / img_path.name)
            label_path = lbl_out / img_path.with_suffix(".txt").name
            with open(label_path, "w") as f:
                f.write(
                    f"{BALL_CLASS_ID} "
                    f"{bbox[0]:.6f} {bbox[1]:.6f} "
                    f"{bbox[2]:.6f} {bbox[3]:.6f}\n"
                )

    save_split(train_set, IMG_TRAIN, LBL_TRAIN)
    save_split(val_set, IMG_VAL, LBL_VAL)
    save_split(test_set, IMG_TEST, LBL_TEST)

    print("\n📊 Dataset split complete:")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")
    print(f"\n📁 YOLO dataset ready at: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
