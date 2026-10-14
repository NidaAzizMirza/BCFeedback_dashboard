import pandas as pd
import pipeline_config as cfg

master = pd.read_excel(cfg.MASTER_FILE, dtype={cfg.COL_RESPONDENT_ID: str})
print(f"Total rows in master.xlsx: {len(master)}")
print(f"Unique respondent IDs: {master[cfg.COL_RESPONDENT_ID].nunique()}")
print(f"Duplicate rows: {len(master) - master[cfg.COL_RESPONDENT_ID].nunique()}")