"""
Fine-tune the ball detector for broadcast camera angles.
Starts from the existing side-view checkpoint and trains on merged dataset.
"""

from ultralytics import YOLO
from pathlib import Path


# Path to existing trained model (starting checkpoint)
CHECKPOINT = "runs/detect/runs/ball_detector/weights/best.pt"


def main():
    if not Path(CHECKPOINT).exists():
        print(f"Error: Checkpoint not found: {CHECKPOINT}")
        print("Train the original ball detector first, or update the CHECKPOINT path.")
        return

    model = YOLO(CHECKPOINT)

    model.train(
        data="ball_broadcast.yaml",
        epochs=60,
        imgsz=960,
        batch=24,
        device=0,
        workers=8,
        optimizer="AdamW",
        lr0=1e-4,           # lower LR for fine-tuning
        patience=15,
        close_mosaic=10,
        pretrained=True,
        resume=False,
        project="runs",
        name="ball_detector_broadcast",
        cache=True,
    )


if __name__ == "__main__":
    main()
