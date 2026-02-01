"""
Apply a fixed scoreboard bounding box to all frames of a game.
"""

import json
from pathlib import Path
from PIL import Image

# CONFIG
GAME_NAME = "game_4"
IMAGES_DIR = Path(f"dataset/training/images/{GAME_NAME}")
BBOX_JSON = Path("scoreboard_bbox_game_4.json")

OUTPUT_LABEL_DIR = Path(f"output_scoreboard/{GAME_NAME}")
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

SCOREBOARD_CLASS_ID = 0

with open(BBOX_JSON, "r") as f:
    bbox = json.load(f)

x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

def convert_to_yolo(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh

count = 0

for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
    with Image.open(img_path) as img:
        w, h = img.size

    cx, cy, bw, bh = convert_to_yolo(x1, y1, x2, y2, w, h)

    label_path = OUTPUT_LABEL_DIR / img_path.with_suffix(".txt").name
    with open(label_path, "w") as f:
        f.write(f"{SCOREBOARD_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    count += 1

print(f"✅ Generated scoreboard labels for {count} images")
