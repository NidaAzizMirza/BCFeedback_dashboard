"""
quick_tag.py
────────────
One-off tagging for a SMALL file, using the REAL pipeline — actual SVM
classifier + actual ABSA model, not a lexicon approximation. Runs steps
2-5 of run_pipeline.py directly (preprocess -> classify -> ABSA ->
entities), skipping step 1 (SurveyMonkey ingest) and step 6 (save to
master.xlsx / metrics.db / processed_ids.csv). Nothing in your live
pipeline's state is touched — safe to run on any file, any time, with
zero side effects on the real data.

Run this from the SAME folder as run_pipeline.py — it imports it
directly rather than duplicating any classification logic, so it can
never drift out of sync with real pipeline changes.

First run may download/verify the two HuggingFace models
(all-mpnet-base-v2, deberta-v3-base-absa-v1.1) — if you've already run
run_pipeline.py before, they're cached locally already and this is fast.

Usage:
    python quick_tag.py --in some_small_file.xlsx --out tagged.xlsx

Input shape: a raw export — needs at least pipeline_config.py's
COL_FEEDBACK column (the long question text) and ideally Rating.
Respondent ID is optional but recommended for traceability.
"""
import argparse
from pathlib import Path

import pandas as pd

import run_pipeline as rp     # reuses the real step_preprocess/classify/absa/entities
import pipeline_config as cfg


def run(in_path: Path, out_path: Path):
    df = pd.read_excel(in_path) if in_path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(in_path)

    if cfg.COL_FEEDBACK not in df.columns:
        raise SystemExit(
            f"Input is missing the expected feedback column: {cfg.COL_FEEDBACK!r}\n"
            f"Columns found: {list(df.columns)}"
        )

    print(f"Loaded {len(df)} rows from {in_path.name}")
    df = rp.step_preprocess(df)     # real cleaning — also drops <3-word responses, same as live
    df = rp.step_classify(df)       # real SVM
    df = rp.step_absa(df)           # real ABSA
    df = rp.step_entities(df)       # user_type + entity extraction

    df.to_excel(out_path, index=False)
    print(f"\nTagged {len(df)} rows -> {out_path}")
    print(f"Method breakdown:\n{df['prediction_method'].value_counts().to_string()}")
    print(f"Avg topics per row: {df['topic_sentiment_pairs'].apply(len).mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    run(a.inp, a.out)
