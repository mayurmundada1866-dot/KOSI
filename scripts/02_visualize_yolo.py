from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

dataset = Path("../dataset/yolo_v8_dataset")
image_dir = dataset / "train" / "images"
label_dir = dataset / "train" / "labels"

images = list(image_dir.glob("*"))
samples = random.sample(images, min(6, len(images)))

for image_path in samples:

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]

    label_path = label_dir / f"{image_path.stem}.txt"

    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(image_path.name)
    plt.axis("off")
    plt.show()