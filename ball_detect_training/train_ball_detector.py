from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="ball.yaml",
        epochs=120,
        imgsz=960,          # small-object friendly
        batch=24,           # safe for 20GB VRAM
        device=0,           # GPU
        workers=8,
        optimizer="AdamW",
        lr0=5e-4,
        patience=25,
        close_mosaic=15,
        pretrained=True,
        resume=False,
        project="runs",
        name="ball_detector",
        cache=True,
    )

if __name__ == "__main__":
    main()
