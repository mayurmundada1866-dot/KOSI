from pathlib import Path
import pandas as pd

folder = Path("dataset/A Raman database of microplastics weathered under natural environments")
metadata = folder / "content.xlsx"

df = pd.read_excel(metadata, header=1)
df = df[df["ID"].astype(str).str.strip() != "ID"]

txt_files = list(folder.rglob("*.txt"))
file_ids = {f.stem for f in txt_files}

metadata_ids = set(
    df["ID"].dropna().astype(str).str.strip()
)

matched = metadata_ids & file_ids
missing = metadata_ids - file_ids
extra = file_ids - metadata_ids

print("\nRaman Dataset Audit")
print("-------------------")

print("TXT files          :", len(txt_files))
print("Metadata samples   :", len(metadata_ids))
print("Matched            :", len(matched))
print("Missing TXT files  :", len(missing))
print("Extra TXT files    :", len(extra))

print("\nStandard / Weathered")
print("--------------------")
print("Standard :", df["ID"].astype(str).str.startswith("sta-").sum())
print("Weathered:", df["ID"].astype(str).str.startswith("wea-").sum())

print("\nPolymer Distribution")
print("--------------------")
print(df["type"].value_counts(dropna=False))

print("\nUnknown polymer labels:", (df["type"].astype(str).str.strip() == "/").sum())

if missing:
    print("\nMissing files:")
    for x in sorted(missing):
        print(x)

if extra:
    print("\nExtra files:")
    for x in sorted(extra):
        print(x)