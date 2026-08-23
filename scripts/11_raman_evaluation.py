import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("raman_preprocessed.csv")

counts = df["polymer"].value_counts()
df = df[df["polymer"].isin(counts[counts >= 2].index)]

X = df.drop(columns=["ID", "polymer", "condition"])
y = df["polymer"]

labels = sorted(y.unique())

label_map = {label: i for i, label in enumerate(labels)}
y = y.map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced")
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

present = sorted(set(y_test) | set(pred))
present_names = [labels[i] for i in present]

print("\nRaman Evaluation")


print("Accuracy :", round(accuracy_score(y_test, pred), 3))
print("Macro F1 :", round(f1_score(y_test, pred, average="macro"), 3))

print("\nClassification Report")


print(
    classification_report(
        y_test,
        pred,
        labels=present,
        target_names=present_names,
        zero_division=0
    )
)

cm = confusion_matrix(y_test, pred, labels=present)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.xticks(range(len(present_names)), present_names, rotation=45)
plt.yticks(range(len(present_names)), present_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Raman SVM Confusion Matrix")

for i in range(len(present_names)):
    for j in range(len(present_names)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "runs/raman_confusion_matrix.png",
    dpi=300
)

plt.show()