import sqlite3, pandas as pd

conn = sqlite3.connect("metrics.db")

# 1. monthly_summary — one row per month?
ms = pd.read_sql("SELECT run_date, month, total_respondents, total_with_feedback FROM monthly_summary ORDER BY run_date", conn)
print("=== monthly_summary ===")
print(ms.to_string())
print()

# 2. tag_group_monthly — check for duplicate run_date per month, AND check
#    whether per-month totals already look inflated vs your SVM numbers
tg = pd.read_sql("SELECT run_date, month, tag_group, total_reviews FROM tag_group_monthly ORDER BY tag_group, run_date", conn)
print("=== tag_group_monthly: rows per (month, tag_group) ===")
dup_check = tg.groupby(["month", "tag_group"])["total_reviews"].agg(["count", "sum"]).reset_index()
print(dup_check.to_string())
print()

print("=== tag_group_monthly: per-month total per tag (pivot) ===")
pivot = tg.pivot_table(index="tag_group", columns="month", values="total_reviews", aggfunc="sum", fill_value=0)
pivot["TOTAL_ACROSS_MONTHS"] = pivot.sum(axis=1)
print(pivot.to_string())

conn.close()