from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import cv2
model = YOLO("runs/detect/runs/yolo_baseline/weights/best.pt")

image_dir = "dataset/yolo_v8_dataset/valid/images"
label_dir = "dataset/yolo_v8_dataset/valid/labels"

names = ["Pellet", "Fragment", "Filament"]

matrix = np.zeros((3, 3), dtype=int)


def morphology(ratio):

    if ratio > 1 and ratio <= 1.3:
        return 0

    elif ratio > 1.3 and ratio <= 4:
        return 1

    elif ratio > 0.8 and ratio <= 1:
        return 2

    return -1


def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


results = model.predict(
    source=image_dir,
    conf=0.25,
    verbose=False
)


for result in results:

    image_name = result.path.split("\\")[-1]
    label_name = image_name.rsplit(".", 1)[0]
    label_path = f"{label_dir}/{label_name}.txt"

    image = cv2.imread(result.path)
    h, w = image.shape[:2]

    ground_truth = []

    with open(label_path, "r") as f:

        for line in f:

            _, xc, yc, bw, bh = map(float, line.split())

            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            ground_truth.append([x1, y1, x2, y2])


    predictions = []

    for box in result.boxes.xyxy.cpu().numpy():
        predictions.append(box.tolist())


    matched = set()


    for gt in ground_truth:

        best_iou = 0
        best_index = -1

        for i, pred in enumerate(predictions):

            if i in matched:
                continue

            score = iou(gt, pred)

            if score > best_iou:
                best_iou = score
                best_index = i


        if best_iou >= 0.5:

            matched.add(best_index)

            gt_width = gt[2] - gt[0]
            gt_height = gt[3] - gt[1]

            pred = predictions[best_index]

            pred_width = pred[2] - pred[0]
            pred_height = pred[3] - pred[1]

            actual_ratio = gt_width / gt_height
            predicted_ratio = pred_width / pred_height

            actual = morphology(actual_ratio)
            predicted = morphology(predicted_ratio)

            if actual != -1 and predicted != -1:
                matrix[actual][predicted] += 1


print("\nConfusion Matrix")
print("----------------")
print(matrix)


total = matrix.sum()

print("\nClass-wise Results")
print("------------------")

for i, name in enumerate(names):

    tp = matrix[i, i]
    fp = matrix[:, i].sum() - tp
    fn = matrix[i, :].sum() - tp
    tn = total - tp - fp - fn

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    accuracy = (tp + tn) / total if total else 0

    print(f"\n{name}")
    print(f"TP        : {tp}")
    print(f"FP        : {fp}")
    print(f"FN        : {fn}")
    print(f"TN        : {tn}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")
    print(f"Accuracy  : {accuracy:.3f}")


plt.figure(figsize=(7, 6))

plt.imshow(matrix)

plt.xticks(range(3), names)
plt.yticks(range(3), names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Morphology Confusion Matrix")

for i in range(3):
    for j in range(3):
        plt.text(j, i, matrix[i, j],
                 ha="center",
                 va="center")

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "runs/morphology_confusion_matrix.png",
    dpi=300
)

plt.show()