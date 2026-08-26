import pandas as pd
import matplotlib.pyplot as plt

ganga = pd.read_csv("dataset/river_dataset/ganga.csv")
sangam = pd.read_csv("dataset/river_dataset/sangam.csv")

ganga["Date"] = pd.to_datetime(ganga["Date"])
sangam["Date"] = pd.to_datetime(sangam["Date"])

print("\nDigital Twin EDA")
print("----------------")

print("\nGanga")
print("Rows:", len(ganga))
print("Date range:", ganga["Date"].min(), "to", ganga["Date"].max())
print("Missing:", ganga.isna().sum().sum())

print("\nSangam")
print("Rows:", len(sangam))
print("Date range:", sangam["Date"].min(), "to", sangam["Date"].max())
print("Missing:", sangam.isna().sum().sum())

print("\nGanga Status")
print(ganga["Status"].value_counts())

print("\nSangam Status")
print(sangam["Status"].value_counts())

plt.figure(figsize=(10, 5))
plt.plot(ganga["Date"], ganga["WQI"], label="Ganga")
plt.plot(sangam["Date"], sangam["WQI"], label="Sangam")
plt.xlabel("Date")
plt.ylabel("WQI")
plt.title("WQI Trend")
plt.legend()
plt.tight_layout()
plt.show()

print("\nGanga Correlation")
print(ganga[["DO", "pH", "ORP", "Cond", "Temp", "WQI"]].corr()["WQI"].sort_values(ascending=False))

print("\nSangam Correlation")
print(sangam[["DO", "pH", "ORP", "Cond", "Temp", "WQI"]].corr()["WQI"].sort_values(ascending=False))