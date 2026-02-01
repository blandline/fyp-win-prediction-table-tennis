from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="scoreboard.yaml",
        epochs=30,          # enough for static objects
        imgsz=640,
        batch=32,           # very safe for your GPU
        device=0,           # GPU
        workers=8,
        optimizer="AdamW",
        lr0=1e-3,
        patience=10,
        pretrained=True,
        resume=False,
        project="runs",
        name="scoreboard_detector",
        cache=True,
    )

if __name__ == "__main__":
    main()
