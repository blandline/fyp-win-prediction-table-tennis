import random
import shutil
from pathlib import Path

# =====================
# PATHS
# =====================
SRC_ROOT = Path("scoreboard_clean")

SRC_IMG_ROOT = SRC_ROOT / "images"
SRC_LBL_ROOT = SRC_ROOT / "labels"

OUT_ROOT = Path("scoreboard_yolo")

IMG_TRAIN = OUT_ROOT / "images/train"
IMG_VAL   = OUT_ROOT / "images/val"
IMG_TEST  = OUT_ROOT / "images/test"

LBL_TRAIN = OUT_ROOT / "labels/train"
LBL_VAL   = OUT_ROOT / "labels/val"
LBL_TEST  = OUT_ROOT / "labels/test"

for d in [IMG_TRAIN, IMG_VAL, IMG_TEST, LBL_TRAIN, LBL_VAL, LBL_TEST]:
    d.mkdir(parents=True, exist_ok=True)

# =====================
# COLLECT SAMPLES
# =====================
samples = []

for game_dir in SRC_IMG_ROOT.iterdir():
    if not game_dir.is_dir():
        continue

    game = game_dir.name
    label_dir = SRC_LBL_ROOT / game

    for img_path in game_dir.glob("*.jpg"):
        lbl_path = label_dir / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            samples.append((img_path, lbl_path))

print(f"Found {len(samples)} clean samples")

# =====================
# SHUFFLE & SPLIT
# =====================
random.shuffle(samples)

n = len(samples)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)

train = samples[:n_train]
val   = samples[n_train:n_train + n_val]
test  = samples[n_train + n_val:]

def copy_split(split, img_out, lbl_out):
    for img, lbl in split:
        shutil.copy(img, img_out / img.name)
        shutil.copy(lbl, lbl_out / lbl.name)

copy_split(train, IMG_TRAIN, LBL_TRAIN)
copy_split(val, IMG_VAL, LBL_VAL)
copy_split(test, IMG_TEST, LBL_TEST)

print("✅ Scoreboard YOLO dataset prepared")
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
