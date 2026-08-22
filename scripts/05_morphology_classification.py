from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import cv2

model = YOLO("runs/detect/runs/yolo_baseline/weights/best.pt")

results = model.predict(
    source="dataset/yolo_v8_dataset/valid/images",
    conf=0.25
)

counts = Counter()

for result in results:

    image = result.orig_img.copy()

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width = x2 - x1
        height = y2 - y1
        ratio = width / height

        if ratio > 1 and ratio<=1.3:
            label = "Pellet"
            color = (255, 0, 0)

        elif ratio <= 4 and ratio>1.3:
            label = "Fragment"
            color = (0, 255, 0)

        elif ratio>0.8 and ratio<=1:
            label = "Filament"
            color = (0, 0, 255)

        counts[label] += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {ratio:.2f}"

        cv2.putText(
            image,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    cv2.imwrite(
        f"runs/morphology_{Path(result.path).stem}.jpg",
        image
    )

print("\nMorphology Count")
print("----------------")
print("Pellet   :", counts["Pellet"])
print("Fragment :", counts["Fragment"])
print("Filament :", counts["Filament"])
print("Total    :", sum(counts.values()))