import sqlite3
import pandas as pd
import pipeline_config as cfg

# ── 1. What's really in master.xlsx, grouped by month ──────────────────────
master = pd.read_excel(cfg.MASTER_FILE, dtype={cfg.COL_RESPONDENT_ID: str})
# master["_month"] = pd.to_datetime(master["Start Date"], errors="coerce").dt.strftime("%Y-%m")
master["_month"] = (
    pd.to_datetime(master["Start Date"], errors="coerce", utc=True)
    .dt.strftime("%Y-%m")
)

print("\nMissing/unreadable Start Dates:")
print(master["_month"].isna().sum())

print("\nMonths found:")

print(master["_month"].value_counts().sort_index())
print("=== master.xlsx — true respondent counts per month ===")
print(master.groupby("_month")[cfg.COL_RESPONDENT_ID].nunique())
print(f"\nTotal unique respondents in master.xlsx: {master[cfg.COL_RESPONDENT_ID].nunique()}")

# ── 2. What's actually stored in monthly_summary ────────────────────────────
conn = sqlite3.connect(cfg.BASE_DIR + "/data/metrics.db")
db_df = pd.read_sql("SELECT * FROM monthly_summary ORDER BY run_date", conn)
print("\n=== monthly_summary table — every row, exactly as stored ===")
print(db_df.to_string())

print(f"\nNumber of rows in monthly_summary: {len(db_df)}")
print(f"Unique 'month' values in monthly_summary: {db_df['month'].nunique() if 'month' in db_df.columns else 'no month column'}")
print(f"Unique 'run_date' (primary key) values: {db_df['run_date'].nunique()}")

if "positive_count" in db_df.columns:
    print(f"\nSum of positive_count across all rows: {db_df['positive_count'].sum()}")
    print(f"Sum of negative_count across all rows: {db_df['negative_count'].sum()}")
    print(f"Sum of neutral_count across all rows: {db_df['neutral_count'].sum() if 'neutral_count' in db_df.columns else 'n/a'}")

conn.close()