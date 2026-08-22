from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/yolo_data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    project="runs",
    name="yolo_baseline"
)