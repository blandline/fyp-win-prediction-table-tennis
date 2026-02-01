import cv2
import torch
import numpy as np
import torchvision.models.segmentation as segm
import torch.nn as nn

# =========================
# CONFIG
# =========================
VIDEO_SOURCE = "../dataset/training/videos/game_1.mp4"   # set to 0 for webcam
CKPT_PATH = "segmentation_data/runs/deeplabv3_final/best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4

# Final class colors (match dataset semantics)
PALETTE = {
    0: (0, 0, 0),        # background
    1: (0, 255, 0),      # player
    2: (255, 0, 0),      # table (red)
    3: (0, 0, 255)       # scoreboard (blue)
}

# =========================
# LOAD MODEL
# =========================
print("Loading segmentation model...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

model = segm.deeplabv3_resnet50(weights=None)
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.load_state_dict(ckpt["model"], strict=False)
model.to(DEVICE).eval()

# =========================
# OPEN VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_SOURCE)
assert cap.isOpened(), "Could not open video source"

print("Press 'q' to quit")

# =========================
# DETERMINE MASK RESOLUTION
# =========================
ret, frame = cap.read()
assert ret, "Empty video stream"

h0, w0 = frame.shape[:2]
rgb0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Probe model output size dynamically
probe = cv2.resize(rgb0, (320, 128))  # temporary probe
x = torch.from_numpy(probe).permute(2, 0, 1).float() / 255.0
x = x.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    out = model(x)["out"]

mask_h, mask_w = out.shape[-2:]
print(f"Segmentation resolution: {mask_w} x {mask_h}")

# Reset video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# =========================
# REAL-TIME LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Downscale frame to mask resolution
    small = cv2.resize(frame, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(small_rgb).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        pred = model(x)["out"].argmax(1).squeeze(0).cpu().numpy()

    # Colorize prediction
    color_mask = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
    for cls, color in PALETTE.items():
        color_mask[pred == cls] = color

    # Upscale mask back to original frame size
    color_mask = cv2.resize(color_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Overlay
    overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

    cv2.imshow("Realtime Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
