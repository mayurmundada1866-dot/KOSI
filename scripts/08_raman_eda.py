from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

folder = Path("dataset/A Raman database of microplastics weathered under natural environments")
metadata = folder / "content.xlsx"

df = pd.read_excel(metadata, header=1)
df = df[df["ID"].astype(str).str.strip() != "ID"]

print("\nRaman EDA")
print("Samples:", len(df))

print("\nPolymer distribution:")
print(df["type"].value_counts())

print("\nStandard / Weathered:")
print("Standard :", (df["ID"].str.startswith("sta-")).sum())
print("Weathered:", (df["ID"].str.startswith("wea-")).sum())


samples = ["sta-1", "sta-6", "sta-10", "wea-13", "wea-18", "wea-34"]

plt.figure(figsize=(10, 6))

for sample in samples:

    file = folder / f"{sample}.txt"

    if not file.exists():
        continue

    data = pd.read_csv(
        file,
        sep=r"\s+",
        header=None,
        names=["shift", "intensity"]
    )

    plt.plot(
        data["shift"],
        data["intensity"],
        label=sample
    )

plt.xlabel("Raman Shift")
plt.ylabel("Intensity")
plt.title("Sample Raman Spectra")
plt.legend()
plt.tight_layout()
plt.show()