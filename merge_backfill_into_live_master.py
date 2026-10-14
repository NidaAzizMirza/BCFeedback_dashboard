"""
merge_backfill_into_live_master.py
───────────────────────────────────
Fills topic_sentiment_pairs / overall_experience_sentiment / user_type for
every row the LIVE pipeline hasn't reached yet — WITHOUT touching the rows
it already has. Output keeps master.xlsx's exact original shape (same 22
columns, same column order) so it can replace the live file directly and
your existing "rebuild from master" step will pick it up.

Rule per row: if the live pipeline already wrote a value (non-null), that
value wins, untouched, byte-for-byte. Only null cells get filled from the
backfill.

Usage:
    python merge_backfill_into_live_master.py --in master.xlsx --out master_merged.xlsx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import backfill_master as bf  # reuses load_master/add_group_columns/.../build_topics


def run(in_path: Path, out_path: Path) -> str:
    original = pd.read_excel(in_path, dtype={bf.COL_ID: str})
    original_cols = list(original.columns)  # preserve exact column order

    already_live = original["topic_sentiment_pairs"].notna()
    n_live, n_total = int(already_live.sum()), len(original)

    # Run the full backfill pipeline (same as backfill_master.py's own run())
    df = bf.load_master(in_path)
    df = bf.add_group_columns(df)
    df = bf.add_quality_flags(df)
    df = bf.add_overall_and_user_type(df)
    df = bf.build_topics(df)

    merged = original.copy()
    fill = ~already_live

    merged.loc[fill, "topic_sentiment_pairs"] = df.loc[fill, "topic_sentiment_pairs_backfilled"]
    merged.loc[fill, "overall_experience_sentiment"] = merged.loc[fill, "overall_experience_sentiment"].where(
        merged.loc[fill, "overall_experience_sentiment"].notna(), df.loc[fill, "overall_experience_sentiment"])
    merged.loc[fill, "user_type"] = merged.loc[fill, "user_type"].where(
        merged.loc[fill, "user_type"].notna(), df.loc[fill, "user_type"])

    merged = merged[original_cols]  # exact original shape, nothing added/reordered
    merged[bf.COL_ID] = merged[bf.COL_ID].astype(str)
    merged.to_excel(out_path, index=False)

    now_filled = merged["topic_sentiment_pairs"].notna().sum()
    empty_after = merged["topic_sentiment_pairs"].apply(
        lambda v: not (isinstance(v, str) and json.loads(v))
    ).sum() if now_filled else n_total

    report = (
        f"# Merge report\n"
        f"Rows: {n_total:,}\n"
        f"Already had real topic_sentiment_pairs from the live pipeline: {n_live:,} "
        f"({n_live/n_total:.1%}) — left untouched\n"
        f"Backfilled this run: {n_total - n_live:,}\n"
        f"Rows with topic_sentiment_pairs populated after merge: {now_filled:,} "
        f"({now_filled/n_total:.1%})\n"
        f"Rows with an empty list (vacuous/test comments, no rating text, etc.): {empty_after:,}\n"
    )
    Path(out_path).with_suffix(".merge_report.md").write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    print(run(a.inp, a.out))
