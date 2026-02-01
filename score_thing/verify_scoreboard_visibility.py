"""
Manually verify scoreboard visibility.
Press:
  y → keep frame
  n → discard frame
  q → quit
"""

import cv2
from pathlib import Path
import shutil

GAME_NAME = "game_4"

IMAGES_DIR = Path(f"dataset/training/images/{GAME_NAME}")
LABELS_DIR = Path(f"output_scoreboard/{GAME_NAME}")

CLEAN_IMG_DIR = Path(f"scoreboard_clean/images/{GAME_NAME}")
CLEAN_LBL_DIR = Path(f"scoreboard_clean/labels/{GAME_NAME}")

CLEAN_IMG_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_LBL_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(IMAGES_DIR.glob("*.jpg"))

for img_path in images:
    label_path = LABELS_DIR / img_path.with_suffix(".txt").name
    if not label_path.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    with open(label_path) as f:
        _, cx, cy, bw, bh = map(float, f.readline().split())

    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Scoreboard Check", img)

    key = cv2.waitKey(0)

    if key == ord("d"):
        shutil.copy(img_path, CLEAN_IMG_DIR / img_path.name)
        shutil.copy(label_path, CLEAN_LBL_DIR / label_path.name)
    elif key == ord("a"):
        pass
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
print("✅ Verification finished")
