from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

folder = Path("dataset/A Raman database of microplastics weathered under natural environments")

metadata = pd.read_excel(folder / "content.xlsx", header=1)
metadata = metadata[metadata["ID"].astype(str).str.strip() != "ID"]

spectra = []

for _, row in metadata.iterrows():

    sample = str(row["ID"]).strip()
    polymer = str(row["type"]).strip()

    if polymer == "/":
        continue

    file = folder / f"{sample}.txt"

    if not file.exists():
        continue

    data = pd.read_csv(
        file,
        sep=r"\s+",
        header=None,
        names=["shift", "intensity"]
    )

    data = data.dropna()

    spectra.append((sample, polymer, data))


min_shift = max(s[2]["shift"].min() for s in spectra)
max_shift = min(s[2]["shift"].max() for s in spectra)

grid = np.linspace(min_shift, max_shift, 1400)

processed = []

for sample, polymer, spectrum in spectra:

    x = spectrum["shift"].values
    y = spectrum["intensity"].values

    y = np.interp(grid, x, y)

    baseline = savgol_filter(y, 101, 3)
    y = y - baseline

    y = savgol_filter(y, 11, 3)

    norm = np.linalg.norm(y)

    if norm != 0:
        y = y / norm

    row_data = {
        "ID": sample,
        "polymer": polymer,
        "condition": "Standard" if sample.startswith("sta-") else "Weathered"
    }

    for i, value in enumerate(y):
        row_data[f"x{i}"] = value

    processed.append(row_data)


df = pd.DataFrame(processed)

df.to_csv("raman_preprocessed.csv", index=False)

print("\nPreprocessing completed")
print("-----------------------")
print("Samples:", len(df))
print("Features:", len(df.columns) - 3)
print("Missing values:", df.isna().sum().sum())
print("Saved: raman_preprocessed.csv")