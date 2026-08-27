import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


os.makedirs("models", exist_ok=True)


# Raman SVM
df = pd.read_csv("raman_preprocessed.csv")

counts = df["polymer"].value_counts()
df = df[df["polymer"].isin(counts[counts >= 2].index)]

X = df.drop(columns=["ID", "polymer", "condition"])
y = df["polymer"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

raman_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced", probability=True)
)

raman_model.fit(X_train, y_train)

joblib.dump(raman_model, "models/raman_svm.pkl")


# River Random Forest
df = pd.read_csv("dataset/Results_MADE.csv")

X = df.drop(columns=["WQI"])
y = df["WQI"]

river_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

river_model.fit(X, y)

joblib.dump(river_model, "models/river_rf.pkl")


# Digital Twin Random Forest
ganga = pd.read_csv("dataset/river_dataset/ganga.csv")
sangam = pd.read_csv("dataset/river_dataset/sangam.csv")

ganga["Date"] = pd.to_datetime(ganga["Date"])
sangam["Date"] = pd.to_datetime(sangam["Date"])

ganga["site"] = "Ganga"
sangam["site"] = "Sangam"

df = pd.concat([ganga, sangam], ignore_index=True)
df = df.sort_values(["site", "Date"])

for col in ["WQI", "DO", "pH", "ORP", "Cond", "Temp"]:
    df[f"{col}_lag1"] = df.groupby("site")[col].shift(1)
    df[f"{col}_lag2"] = df.groupby("site")[col].shift(2)
    df[f"{col}_lag3"] = df.groupby("site")[col].shift(3)

df = df.dropna()

features = [
    "WQI_lag1", "WQI_lag2", "WQI_lag3",
    "DO_lag1", "DO_lag2", "DO_lag3",
    "pH_lag1", "pH_lag2", "pH_lag3",
    "ORP_lag1", "ORP_lag2", "ORP_lag3",
    "Cond_lag1", "Cond_lag2", "Cond_lag3",
    "Temp_lag1", "Temp_lag2", "Temp_lag3"
]

digital_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

digital_model.fit(df[features], df["WQI"])

joblib.dump(
    digital_model,
    "models/digital_twin_rf.pkl"
)

print("\nModels saved successfully")
print("-------------------------")
print("Raman       : models/raman_svm.pkl")
print("River       : models/river_rf.pkl")
print("Digital Twin: models/digital_twin_rf.pkl")