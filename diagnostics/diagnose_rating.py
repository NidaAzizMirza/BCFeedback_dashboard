import survey_monkey_fetch as smf
import pandas as pd

df = smf.fetch_survey_as_dataframe()

print("=== RAW value_counts (before any mapping) ===")
print(df["Rate your experience"].value_counts(dropna=False))
print()

rating_col = df["Rate your experience"].copy()
rating_map = {"Excellent": 5, "Poor": 1}
mapped = rating_col.map(rating_map)
final = mapped.fillna(pd.to_numeric(rating_col, errors="coerce"))

print("=== AFTER mapping/coercion ===")
print(final.value_counts(dropna=False).sort_index())
print()
print(f"Total rows: {len(final)}")
print(f"Non-null ratings: {final.notna().sum()}")
print(f"Manually computed average: {final.mean():.4f}")