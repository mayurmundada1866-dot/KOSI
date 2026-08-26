import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ganga = pd.read_csv("dataset/river_dataset/ganga.csv")
sangam = pd.read_csv("dataset/river_dataset/sangam.csv")

ganga["Date"] = pd.to_datetime(ganga["Date"])
sangam["Date"] = pd.to_datetime(sangam["Date"])

ganga["site"] = "Ganga"
sangam["site"] = "Sangam"

df = pd.concat([ganga, sangam], ignore_index=True)
df = df.sort_values("Date")

for lag in [1, 2, 3, 7]:
    df[f"WQI_lag_{lag}"] = df.groupby("site")["WQI"].shift(lag)

df["WQI_roll_7"] = (
    df.groupby("site")["WQI"]
    .transform(lambda x: x.rolling(7).mean())
)

df["month"] = df["Date"].dt.month
df["dayofyear"] = df["Date"].dt.dayofyear

df = df.dropna()

features = [
    "DO", "pH", "ORP", "Cond", "Temp",
    "WQI_lag_1", "WQI_lag_2",
    "WQI_lag_3", "WQI_lag_7",
    "WQI_roll_7", "month", "dayofyear"
]

train = df[df["Date"] < "2020-01-01"]
test = df[df["Date"] >= "2020-01-01"]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(train[features], train["WQI"])

test = test.copy()
test["Predicted_WQI"] = model.predict(test[features])

mae = mean_absolute_error(test["WQI"], test["Predicted_WQI"])
rmse = mean_squared_error(test["WQI"], test["Predicted_WQI"]) ** 0.5
r2 = r2_score(test["WQI"], test["Predicted_WQI"])

print("\nDigital Twin Evaluation")
print("-----------------------")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2  :", round(r2, 3))

plt.figure(figsize=(10, 5))

for site in ["Ganga", "Sangam"]:

    data = test[test["site"] == site]

    plt.plot(
        data["Date"],
        data["WQI"],
        label=f"{site} Actual"
    )

    plt.plot(
        data["Date"],
        data["Predicted_WQI"],
        label=f"{site} Predicted"
    )

plt.xlabel("Date")
plt.ylabel("WQI")
plt.title("Digital Twin: Actual vs Predicted WQI")
plt.legend()
plt.tight_layout()
plt.show()