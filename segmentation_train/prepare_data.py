import shutil
import re
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATASET_ROOT = Path("../dataset")

IMAGES_ROOT = DATASET_ROOT / "training" / "images"
ANNOT_ROOT = DATASET_ROOT / "training" / "annotations"

SEG_FOLDER_NAME = "segmentation_masks"

OUT_ROOT = Path("segmentation_train")
OUT_IMG = OUT_ROOT / "images"
OUT_MASK = OUT_ROOT / "masks_rgb"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

IMG_NUM_RE = re.compile(r"(\d+)")

total = 0
skipped = 0

print("Preparing segmentation dataset (collision-safe)...")

for game_dir in sorted(IMAGES_ROOT.iterdir()):
    if not game_dir.is_dir():
        continue

    game_name = game_dir.name  # e.g. game_1
    img_dir = game_dir
    seg_dir = ANNOT_ROOT / game_name / SEG_FOLDER_NAME

    if not seg_dir.exists():
        print(f"[WARN] Missing segmentation dir for {game_name}, skipping")
        continue

    print(f"Processing {game_name}")

    for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc=game_name):
        match = IMG_NUM_RE.search(img_path.stem)
        if not match:
            skipped += 1
            continue

        frame_id = int(match.group(1))  # removes leading zeros
        mask_path = seg_dir / f"{frame_id}.png"

        if not mask_path.exists():
            skipped += 1
            continue

        # new unique names
        new_stem = f"{game_name}_{frame_id}"

        new_img_path = OUT_IMG / f"{new_stem}.jpg"
        new_mask_path = OUT_MASK / f"{new_stem}.png"

        # copy safely
        shutil.copy2(img_path, new_img_path)
        shutil.copy2(mask_path, new_mask_path)

        total += 1

print("\nDone.")
print(f"Total paired samples: {total}")
print(f"Skipped (missing or unmatched): {skipped}")
print(f"Output written to: {OUT_ROOT.resolve()}")
