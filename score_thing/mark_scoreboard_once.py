"""
Manually draw scoreboard bounding box on ONE image.
The box will be reused for all frames of the same game.
"""

import cv2
from pathlib import Path
import json

# CONFIG
IMAGE_PATH = "dataset/training/images/game_4/img_001270.jpg"
OUTPUT_JSON = "scoreboard_bbox_game_4.json"

print("📦 Instructions:")
print("- Draw a rectangle around the scoreboard")
print("- Press ENTER to confirm")
print("- Press ESC to cancel")

img = cv2.imread(IMAGE_PATH)
clone = img.copy()

bbox = []

def mouse_callback(event, x, y, flags, param):
    global bbox, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        bbox.append((x, y))
        cv2.rectangle(clone, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.imshow("Select Scoreboard", clone)

cv2.namedWindow("Select Scoreboard")
cv2.setMouseCallback("Select Scoreboard", mouse_callback)
cv2.imshow("Select Scoreboard", img)

key = cv2.waitKey(0)
cv2.destroyAllWindows()

if len(bbox) != 2:
    print("❌ No bounding box selected.")
    exit()

(x1, y1), (x2, y2) = bbox

annotation = {
    "x1": min(x1, x2),
    "y1": min(y1, y2),
    "x2": max(x1, x2),
    "y2": max(y1, y2)
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(annotation, f, indent=2)

print(f"✅ Scoreboard bbox saved to {OUTPUT_JSON}")
