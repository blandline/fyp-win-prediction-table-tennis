import os, random
from pathlib import Path

IMG_DIR = Path("segmentation_data/images")
MASK_DIR = Path("segmentation_data/masks_rgb")
OUT = Path("splits")
OUT.mkdir(exist_ok=True)

MAX_SAMPLES = 3000      # <<< LIMIT HERE
VAL_RATIO = 0.2
SEED = 42

imgs = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png"]])

pairs = []
for img in imgs:
    mask = MASK_DIR / (img.stem + ".png")
    if mask.exists():
        pairs.append(img.stem)

print("Total available paired samples:", len(pairs))

random.seed(SEED)
random.shuffle(pairs)

# limit total dataset size
if len(pairs) > MAX_SAMPLES:
    pairs = pairs[:MAX_SAMPLES]

n_val = int(len(pairs) * VAL_RATIO)

val = sorted(pairs[:n_val])
train = sorted(pairs[n_val:])

(OUT / "train.txt").write_text("\n".join(train))
(OUT / "val.txt").write_text("\n".join(val))

print("Final dataset size:", len(pairs))
print("Train:", len(train))
print("Val:", len(val))
