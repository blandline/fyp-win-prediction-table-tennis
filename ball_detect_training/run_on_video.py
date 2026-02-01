from ultralytics import YOLO

VIDEO_PATH = "dataset/training/videos/game_2.mp4"
MODEL_PATH = "runs/detect/runs/ball_detector/weights/best.pt"

def main():
    model = YOLO(MODEL_PATH)

    model.predict(
        source=VIDEO_PATH,
        conf=0.35,
        device=0,     # GPU
        # save=True,
        show=True
    )

if __name__ == "__main__":
    main()
