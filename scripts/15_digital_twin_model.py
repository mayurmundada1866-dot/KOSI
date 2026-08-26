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
    "DO",
    "pH",
    "ORP",
    "Cond",
    "Temp",
    "WQI_lag_1",
    "WQI_lag_2",
    "WQI_lag_3",
    "WQI_lag_7",
    "WQI_roll_7",
    "month",
    "dayofyear"
]

train = df[df["Date"] < "2020-01-01"]
test = df[df["Date"] >= "2020-01-01"]

X_train = train[features]
y_train = train["WQI"]

X_test = test[features]
y_test = test["WQI"]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\nDigital Twin Forecasting")
print("------------------------")
print("Train samples:", len(train))
print("Test samples :", len(test))
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2  :", round(r2, 3))