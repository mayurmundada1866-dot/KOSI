import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/Results_MADE.csv")

print("\nRiver Contamination EDA")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nBasic statistics:")
print(df.describe())

print("\nWQI distribution:")
print(df["WQI"].describe())

print("\nCorrelation with WQI:")
print(df.corr(numeric_only=True)["WQI"].sort_values(ascending=False))


plt.figure(figsize=(8, 5))
plt.hist(df["WQI"], bins=30)
plt.xlabel("WQI")
plt.ylabel("Frequency")
plt.title("WQI Distribution")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(df["pH"], df["WQI"], alpha=0.6)
plt.xlabel("pH")
plt.ylabel("WQI")
plt.title("pH vs WQI")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(df["Dissolved Oxygen"], df["WQI"], alpha=0.6)
plt.xlabel("Dissolved Oxygen")
plt.ylabel("WQI")
plt.title("Dissolved Oxygen vs WQI")
plt.tight_layout()
plt.show()