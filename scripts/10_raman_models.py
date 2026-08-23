import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier


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


models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced")
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="mlogloss"
    )
}


print("\nRaman Model Results")
print("-------------------")

print("\nClass counts:")
print(df["polymer"].value_counts())


for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Macro F1 : {f1:.3f}")