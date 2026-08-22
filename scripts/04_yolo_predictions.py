from ultralytics import YOLO

model = YOLO("runs/detect/runs/yolo_baseline/weights/best.pt")

model.predict(
    source="dataset/yolo_v8_dataset/valid/images",
    conf=0.25,
    save=True,
    save_txt=True,
    project="runs/predictions",
    name="yolo_baseline"
)