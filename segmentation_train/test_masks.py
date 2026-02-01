import cv2
import numpy as np
from pathlib import Path
from collections import Counter

MASK_DIR = Path("segmentation_data/masks_rgb")

counter = Counter()

p = MASK_DIR / "game_1_14.png"
m = cv2.imread(str(p))
m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
colors = np.unique(m.reshape(-1, 3), axis=0)
for c in colors:
    counter[tuple(c.tolist())] += 1

print("Unique colors and how many masks they appear in:\n")
for color, count in counter.most_common():
    print(color, "->", count)
