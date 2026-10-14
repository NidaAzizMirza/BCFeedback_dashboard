# update_raw_metrics.py
# ── Daily lightweight update ────────────────────────────────────────────────
# Updates ONLY response volume, rating distribution, average rating, and
# NSAT — no classification, sentiment, or entity extraction. Deliberately
# has no dependency on torch/transformers/spacy, so the daily GitHub Action
# installs fast and doesn't spend minutes loading ML models for a ~50-70
# row delta.
#
# The full run_pipeline.py (classify/ABSA/entities/dashboard tag+feature+
# error tables) still runs separately, on its own weekly schedule.
#
# Usage: python update_raw_metrics.py

import os
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

import pipeline_config as cfg
import survey_monkey_fetch as smf
import aggregation_db as adb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# UTC timestamp captured at process start, used as this run's checkpoint
# if it succeeds (see save_checkpoint calls in main()). Separate from
# run_pipeline.py's own FETCH_CHECKPOINT_TS / LAST_FETCH_CHECKPOINT — see
# the comment on LAST_FETCH_CHECKPOINT_RAW in pipeline_config.py for why
# the daily and weekly jobs must not share one checkpoint.
FETCH_CHECKPOINT_TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def load_processed_raw_ids():
    if os.path.exists(cfg.PROCESSED_IDS_RAW):
        return set(
            pd.read_csv(cfg.PROCESSED_IDS_RAW)["respondent_id"].astype(str).tolist()
        )
    return set()


def save_processed_raw_ids(new_ids):
    run_date = datetime.now().strftime("%Y-%m-%d")
    new_rows = pd.DataFrame({
        "respondent_id": [str(i) for i in new_ids],
        "processed_date": run_date,
    })
    if os.path.exists(cfg.PROCESSED_IDS_RAW):
        existing = pd.read_csv(cfg.PROCESSED_IDS_RAW)
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
    new_rows.to_csv(cfg.PROCESSED_IDS_RAW, index=False)


def normalize_id_series(s):
    """
    Normalize respondent IDs to a canonical string form, stripping a
    trailing ".0" if present.

    Root cause this guards against: a single stray blank row anywhere in
    the Respondent ID column forces pandas to read the WHOLE column as
    float64 on read — every clean ID like "119161712636" silently becomes
    "119161712636.0". That one-character difference breaks every
    ID-based dedup check against processed_ids.csv/processed_ids_raw.csv
    (they never match), causing the entire file to look "new" every time
    — exactly what happened on 2026-07-14. This makes ID comparisons
    robust regardless of whether a future export happens to have a blank
    row, rather than relying on the source file always being clean.
    """
    return s.astype(str).str.replace(r'\.0$', '', regex=True)


def normalize_rating_column(df):
    """
    aggregation_db.write_raw_metrics() expects a column literally named
    "Rating", but neither the API nor the manual export produce that —
    both return the raw SurveyMonkey question heading, "Rate your
    experience" (the API returns headings verbatim; the Excel export
    uses the same question text as its column name). This script
    deliberately avoids importing preprocessing.py, which normally does
    this rename, because that module pulls in bs4/spellchecker/demoji —
    heavy-ish deps this lightweight daily script is built to skip. So we
    replicate just the rename + Excellent/Poor mapping needed here,
    nothing else.

    NOTE: this means this exact KeyError would fire on ANY successful
    fetch — API or manual — not just SKIP_API mode. Very likely this
    script has never completed successfully since this rename was
    dropped in a refactor, which explains why total_respondents in
    monthly_summary has been stuck/stale.
    """
    if "Rating" not in df.columns and "Rate your experience" in df.columns:
        df = df.rename(columns={"Rate your experience": "Rating"})
        rating_map = {"Excellent": 5, "Poor": 1}
        df["Rating"] = df["Rating"].map(rating_map).fillna(
            pd.to_numeric(df["Rating"], errors="coerce")
        )
    return df


def main():
    log.info("=" * 60)
    log.info("DAILY RAW METRICS UPDATE (no NLP)")
    log.info("=" * 60)

    # ── Fetch ────────────────────────────────────────────────────────────
    if cfg.USE_LOCAL_CACHE:
        log.info(
            f"USE_LOCAL_CACHE is set — reading from local cache at "
            f"{cfg.SURVEY_CACHE_FILE} instead of the SurveyMonkey API."
        )
        if not os.path.exists(cfg.SURVEY_CACHE_FILE):
            log.error(
                f"USE_LOCAL_CACHE is set but no cache file found at "
                f"{cfg.SURVEY_CACHE_FILE}. Run refresh_survey_cache.py "
                f"at least once first."
            )
            sys.exit(1)

        cache_checkpoint = smf.load_checkpoint(cfg.LAST_FETCH_CHECKPOINT_CACHE)
        age_hours = smf.checkpoint_age_hours(cache_checkpoint)
        if age_hours is not None and age_hours > cfg.STALE_CACHE_WARNING_HOURS:
            log.warning(
                f"Local cache is {age_hours:.1f}h old (threshold: "
                f"{cfg.STALE_CACHE_WARNING_HOURS}h) — refresh_survey_cache.py's "
                f"daily job may have stopped running. Proceeding anyway."
            )

        df = pd.read_csv(cfg.SURVEY_CACHE_FILE, dtype={cfg.COL_RESPONDENT_ID: str})
        log.info(f"Loaded {len(df)} responses from local cache")
    elif cfg.SKIP_API:
        log.info("SKIP_API is set — skipping SurveyMonkey API, reading manual export directly.")
        if not os.path.exists(cfg.INPUT_FILE):
            log.error(f"SKIP_API is set but no manual export file found at {cfg.INPUT_FILE}.")
            sys.exit(1)
        df = pd.read_excel(cfg.INPUT_FILE, sheet_name=cfg.INPUT_SHEET)
        log.info(f"Loaded {len(df)} total responses from manual export ({cfg.INPUT_FILE})")
        # The raw Excel export has a spurious second header row (SurveyMonkey
        # sub-labels) — same as run_pipeline.py's has_extra_header_row logic.
        # The API path doesn't need this; it returns clean data already.
        if len(df) > 0:
            df = df.iloc[1:].reset_index(drop=True)
    else:
        checkpoint = smf.load_checkpoint(cfg.LAST_FETCH_CHECKPOINT_RAW)
        fetch_since = smf.checkpoint_with_overlap(checkpoint)
        try:
            df = smf.fetch_survey_as_dataframe(start_created_at=fetch_since)
            if fetch_since:
                log.info(f"Fetched {len(df)} responses from SurveyMonkey API since {fetch_since}")
            else:
                log.info(f"Fetched {len(df)} total responses from SurveyMonkey API (full history — no checkpoint yet)")
        except Exception as e:
            log.error(f"SurveyMonkey API fetch failed: {e}")
            log.error("Daily raw-metrics update requires the API (or SKIP_API=true with a manual export) — no automatic fallback here.")
            sys.exit(1)

    df = normalize_rating_column(df)

    if len(df) == 0:
        log.warning("API returned zero responses — nothing to do.")
        if not cfg.SKIP_API:
            smf.save_checkpoint(cfg.LAST_FETCH_CHECKPOINT_RAW, FETCH_CHECKPOINT_TS)
        sys.exit(0)

    df[cfg.COL_RESPONDENT_ID] = normalize_id_series(df[cfg.COL_RESPONDENT_ID])

    # ── Dedup against the raw-only processed list ───────────────────────
    processed_ids = load_processed_raw_ids()
    new_df = df[~df[cfg.COL_RESPONDENT_ID].isin(processed_ids)].copy()
    log.info(f"New respondents since last raw update: {len(new_df)}")

    if len(new_df) == 0:
        log.info("No new respondents — nothing to write.")
        if not cfg.SKIP_API:
            smf.save_checkpoint(cfg.LAST_FETCH_CHECKPOINT_RAW, FETCH_CHECKPOINT_TS)
        sys.exit(0)

    # ── Write raw metrics (additive upsert, per month) ──────────────────
    adb.initialise_db()
    adb.write_raw_metrics(new_df)

    # ── Record these IDs as done for raw purposes only ──────────────────
    # (They remain eligible for the weekly full NLP run, which tracks its
    # own separate processed_ids.csv.)
    save_processed_raw_ids(new_df[cfg.COL_RESPONDENT_ID].tolist())

    if not cfg.SKIP_API:
        smf.save_checkpoint(cfg.LAST_FETCH_CHECKPOINT_RAW, FETCH_CHECKPOINT_TS)

    log.info(f"Done — {len(new_df)} new respondents added to raw metrics.")


if __name__ == "__main__":
    main()