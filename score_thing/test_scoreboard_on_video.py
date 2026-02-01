from ultralytics import YOLO
import cv2
from collections import deque
import csv
import numpy as np

# =========================
# CONFIG
# =========================
VIDEO_PATH = "dataset/training/videos/game_3.mp4"

SCOREBOARD_MODEL_PATH = "runs/detect/runs/scoreboard_detector2/weights/best.pt"
DIGIT_MODEL_PATH = "runs/detect/runs/digits/digit_yolov8n/weights/best.pt"

LOG_FILE = "score_log.csv"

DETECT_EVERY_N_FRAMES = 30
READ_EVERY_N_FRAMES = 3
CONF_THRESH = 0.3

PAD_X = 2
PAD_Y = 2

# =========================
# LOAD MODELS
# =========================
scoreboard_model = YOLO(SCOREBOARD_MODEL_PATH)
digit_model = YOLO(DIGIT_MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
last_bbox = None

# =========================
# SCORE STATE (OPTION A)
# =========================
score_history = deque(maxlen=3)
current_score = "-- : --"

# =========================
# LOG FILE
# =========================
with open(LOG_FILE, "w", newline="") as f:
    csv.writer(f).writerow(["timestamp_sec", "frame", "score"])

# =========================
# DIGIT-BASED SCORE READER
# =========================
def read_score_from_crop(scoreboard_crop, debug=False):
    if scoreboard_crop is None or scoreboard_crop.size == 0:
        return None

    h, w = scoreboard_crop.shape[:2]

    results = digit_model(
        scoreboard_crop,
        conf=0.35,
        imgsz=320,
        verbose=False
    )

    if not results or len(results[0].boxes) == 0:
        return None

    digits = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls.item())

        if cls < 0 or cls > 9:
            continue

        cx = (x1 + x2) / 2

        digits.append({
            "digit": str(cls),
            "x1": x1,
            "cx": cx,
            "side": "left" if cx < w / 2 else "right"
        })

        if debug:
            cv2.rectangle(
                scoreboard_crop,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                1
            )
            cv2.putText(
                scoreboard_crop,
                str(cls),
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    left_digits = [d for d in digits if d["side"] == "left"]
    right_digits = [d for d in digits if d["side"] == "right"]

    if not left_digits or not right_digits:
        return None

    left_digits.sort(key=lambda d: d["x1"])
    right_digits.sort(key=lambda d: d["x1"])

    left_str = "".join(d["digit"] for d in left_digits)[:2]
    right_str = "".join(d["digit"] for d in right_digits)[:2]

    try:
        left_score = int(left_str)
        right_score = int(right_str)
    except ValueError:
        return None

    if left_score > 30 or right_score > 30:
        return None

    if debug:
        cv2.imshow("DIGITS DEBUG", scoreboard_crop)
        cv2.waitKey(1)

    return f"{left_score} : {right_score}"

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # =========================
    # SCOREBOARD DETECTION
    # =========================
    if frame_idx % DETECT_EVERY_N_FRAMES == 0:
        results = scoreboard_model(frame, conf=CONF_THRESH, verbose=False)

        if results and len(results[0].boxes) > 0:
            boxes = sorted(
                results[0].boxes,
                key=lambda b: b.conf.item(),
                reverse=True
            )

            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])

            x1 -= PAD_X
            x2 += PAD_X
            y1 -= PAD_Y
            y2 += PAD_Y

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            last_bbox = (x1, y1, x2, y2)

    # =========================
    # SCORE READING (OPTION A)
    # =========================
    if last_bbox:
        x1, y1, x2, y2 = last_bbox
        crop = frame[y1:y2, x1:x2]

        if frame_idx % READ_EVERY_N_FRAMES == 0:
            score = read_score_from_crop(crop, debug=False)

            if score:
                print("RAW SCORE:", score)
                score_history.append(score)

                if len(score_history) == 3:
                    a, b, c = score_history

                    if a == b or a == c:
                        stable_score = a
                    elif b == c:
                        stable_score = b
                    else:
                        stable_score = None

                    if stable_score and stable_score != current_score:
                        current_score = stable_score
                        timestamp = frame_idx / fps

                        with open(LOG_FILE, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [round(timestamp, 2), frame_idx, current_score]
                            )

                        print(f"[{timestamp:.2f}s] SCORE → {current_score}")

                    score_history.clear()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(
        frame,
        f"Score: {current_score}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3
    )

    cv2.imshow("Digit YOLO Score Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

print("✅ Finished")
print(f"📄 Score log saved to {LOG_FILE}")
