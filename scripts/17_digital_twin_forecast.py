import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

train = df[df["Date"] < "2020-01-01"]
test = df[df["Date"] >= "2020-01-01"]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(train[features], train["WQI"])

pred = model.predict(test[features])

mae = mean_absolute_error(test["WQI"], pred)
rmse = mean_squared_error(test["WQI"], pred) ** 0.5
r2 = r2_score(test["WQI"], pred)

print("\nTrue Digital Twin Forecasting")
print("-----------------------------")
print("Train samples:", len(train))
print("Test samples :", len(test))
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2  :", round(r2, 3))


