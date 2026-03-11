from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/ball_detector/weights/best.pt")
    metrics = model.val(plots=True)

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.precision)
    print("Recall:", metrics.box.recall)

if __name__ == "__main__":
    main()
