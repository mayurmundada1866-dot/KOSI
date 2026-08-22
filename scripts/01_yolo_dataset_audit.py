from pathlib import Path
from collections import Counter

# Dataset path
dataset = Path("../dataset/yolo_v8_dataset")

for split in ["train", "valid"]:

    image_dir = dataset / split / "images"
    label_dir = dataset / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    class_count = Counter()

    for label in labels:
        with open(label, "r") as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_count[class_id] += 1

    print(f"\n===== {split.upper()} =====")
    print("Images :", len(images))
    print("Labels :", len(labels))
    print("Objects:", sum(class_count.values()))

    print("Classes:")
    for cls, count in sorted(class_count.items()):
        print(f"  Class {cls}: {count}")