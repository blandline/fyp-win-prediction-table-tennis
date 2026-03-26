from ultralytics import YOLO

VIDEO_PATH = "dataset/training/videos/FULL MATCH  Felix Lebrun vs Truls Moregard  MS QF  WTTFrankfurt 2024 (snipped).mp4"
MODEL_PATH = "runs/detect/runs/ball_detector_broadcast/weights/best.pt"

def main():
    model = YOLO(MODEL_PATH)

    model.predict(
        source=VIDEO_PATH,
        conf=0.45,
        device=0,     # GPU
        # save=True,
        show=True
    )

if __name__ == "__main__":
    main()
