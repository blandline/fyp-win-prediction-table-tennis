from ultralytics import YOLO

def main():
    # Load a small YOLOv8 model (perfect for digits)
    model = YOLO("yolov8n.pt")  # nano is enough for digits

    # Train
    model.train(
        data="numbers.v1i.yolov8/data.yaml",
        imgsz=320,          # digits don't need large images
        epochs=50,          # 30–50 is usually enough
        batch=16,
        device=0,           # GPU (use 'cpu' if needed)
        workers=4,
        project="runs/digits",
        name="digit_yolov8n",
        exist_ok=True
    )

    # Validate after training
    model.val()

if __name__ == "__main__":
    main()
