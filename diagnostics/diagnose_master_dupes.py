# import pandas as pd
# import os
# from pathlib import Path
#
# BASE_DIR = Path(__file__).parent.parent
# DATA_DIR = BASE_DIR / "data"
#
# MASTER = DATA_DIR / "master.xlsx"
# # MASTER = "data/master.xlsx"
# PROCESSED_IDS = DATA_DIR/"processed_ids.csv"
#
# print("MASTER =", MASTER)
# print("Exists?", os.path.exists(MASTER))
#
# print("PROCESSED_IDS =", PROCESSED_IDS)
# print("Exists?", os.path.exists(PROCESSED_IDS))
#
# # ── 1. Check master.xlsx for duplicate Respondent IDs ──────────────────────
# if os.path.exists(MASTER):
#     master = pd.read_excel(MASTER, dtype={"Respondent ID": str})
#     master["Respondent ID"] = (
#         master["Respondent ID"]
#         .str.replace(r"\.0$", "", regex=True)
#     )
#     print(f"=== master.xlsx ===")
#     print(f"Total rows: {len(master)}")
#     print(f"Unique Respondent IDs: {master['Respondent ID'].nunique()}")
#     dupes = master[master.duplicated(subset=['Respondent ID'], keep=False)]
#     print(f"Rows with duplicated Respondent ID: {len(dupes)}")
#     if len(dupes):
#         print("\nSample duplicate Respondent IDs and their counts:")
#         counts = master['Respondent ID'].value_counts()
#         print(counts[counts > 1].head(20))
#         print("\nSample duplicate rows (first duplicated ID):")
#         sample_id = counts[counts > 1].index[0]
#         print(master[master['Respondent ID'] == sample_id][['Respondent ID', 'Start Date', 'run_date'] if 'run_date' in master.columns else ['Respondent ID', 'Start Date']])
# else:
#     print("master.xlsx not found")
#
# print()
#
# # ── 2. Check processed_ids.csv for duplicates ───────────────────────────────
# if os.path.exists(PROCESSED_IDS):
#     proc = pd.read_csv(PROCESSED_IDS, dtype={"respondent_id": str})
#     print(f"=== processed_ids.csv ===")
#     print(f"Total rows: {len(proc)}")
#     print(f"Unique respondent_ids: {proc['respondent_id'].nunique()}")
#     dupes = proc[proc.duplicated(subset=['respondent_id'], keep=False)]
#     print(f"Rows with duplicated respondent_id: {len(dupes)}")
#     if len(dupes):
#         print("\nSample duplicate respondent_ids and their processed_dates:")
#         counts = proc['respondent_id'].value_counts()
#         sample_id = counts[counts > 1].index[0]
#         print(proc[proc['respondent_id'] == sample_id])
# else:
#     print("processed_ids.csv not found")
# from pathlib import Path
# import os
#
# print("Current working directory:")
# print(os.getcwd())
#
# print("\nDirectory contents:")
# print(os.listdir())
#
# data_dir = Path(__file__).parent.parent / "data"
#
# print("\nData directory:")
# print(data_dir)
#
# print("Data directory exists:", data_dir.exists())
# print("Data directory is dir:", data_dir.is_dir())
#
# print("\nContents of data:")
# if data_dir.exists():
#     for f in data_dir.iterdir():
#         print(repr(f.name))
#
# master = data_dir / "master.xlsx"
#
# print("\nMaster path:")
# print(master)
# print("Resolved:", master.resolve())
# print("Exists:", master.exists())
# print("Is file:", master.is_file())
#
#
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

PROCESSED_IDS = DATA_DIR / "processed_ids.csv"

print("\n=== processed_ids.csv ===")

processed = pd.read_csv(
    PROCESSED_IDS,
    dtype={"respondent_id": str}
)

# Clean accidental Excel-style decimals
processed["respondent_id"] = (
    processed["respondent_id"]
    .str.replace(r"\.0$", "", regex=True)
    .str.strip()
)

print("Total rows:", len(processed))
print("Unique respondent_ids:", processed["respondent_id"].nunique())

duplicates = processed["respondent_id"].duplicated(keep=False)

print("Rows with duplicated respondent_id:", duplicates.sum())

if duplicates.sum() > 0:
    print("\nDuplicate IDs:")
    print(
        processed.loc[duplicates, "respondent_id"]
        .value_counts()
    )