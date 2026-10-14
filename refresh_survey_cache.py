# refresh_survey_cache.py
# ── Daily local-cache refresh ───────────────────────────────────────────────
# Pulls new SurveyMonkey responses since this job's OWN checkpoint and
# appends them to a local CSV cache (data/survey_cache.csv). This is
# SEPARATE from both the daily raw-metrics job (update_raw_metrics.py)
# and the weekly full pipeline (run_pipeline.py) — each has its own
# checkpoint (see the comment on LAST_FETCH_CHECKPOINT_CACHE in
# pipeline_config.py for why they must not share one).
#
# The ONLY purpose of this script is to build up a local copy of the raw
# survey data so run_pipeline.py can be run repeatedly — locally, or in
# CI — against real data via USE_LOCAL_CACHE=true, with ZERO
# SurveyMonkey API calls. Useful for previewing the full dashboard
# without burning API quota.
#
# No NLP dependencies — same lightweight footprint as update_raw_metrics.py.
#
# Usage: python refresh_survey_cache.py
# First run will be a full fetch (no checkpoint yet — expect ~128 API
# calls for the current ~12,800+ response history). Every run after
# that is incremental and cheap.

import os
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

import pipeline_config as cfg
import survey_monkey_fetch as smf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# UTC timestamp captured at process start, saved as this run's checkpoint
# only if the run succeeds.
FETCH_CHECKPOINT_TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def normalize_id_series(s):
    """
    Same normalization as update_raw_metrics.py's copy of this function —
    strips a trailing ".0" that appears when a stray blank row forces
    pandas to read the whole Respondent ID column as float64. Duplicated
    here rather than imported, to keep this script's dependencies
    minimal; if you change one copy, change both.
    """
    return s.astype(str).str.replace(r'\.0$', '', regex=True)


def load_cache() -> pd.DataFrame:
    if os.path.exists(cfg.SURVEY_CACHE_FILE):
        return pd.read_csv(cfg.SURVEY_CACHE_FILE, dtype={cfg.COL_RESPONDENT_ID: str})
    return pd.DataFrame()


def save_cache(df: pd.DataFrame):
    os.makedirs(os.path.dirname(cfg.SURVEY_CACHE_FILE), exist_ok=True)
    df.to_csv(cfg.SURVEY_CACHE_FILE, index=False)


def main():
    log.info("=" * 60)
    log.info("SURVEY CACHE REFRESH (no NLP)")
    log.info("=" * 60)

    checkpoint = smf.load_checkpoint(cfg.LAST_FETCH_CHECKPOINT_CACHE)
    fetch_since = smf.checkpoint_with_overlap(checkpoint)

    try:
        new_df = smf.fetch_survey_as_dataframe(start_created_at=fetch_since)
    except Exception as e:
        log.error(f"SurveyMonkey API fetch failed: {e}")
        sys.exit(1)

    if fetch_since:
        log.info(f"Fetched {len(new_df)} responses since {fetch_since}")
    else:
        log.info(f"Fetched {len(new_df)} total responses (full history — first cache build)")

    if len(new_df) == 0:
        log.info("No new responses since last cache checkpoint — cache unchanged.")
        smf.save_checkpoint(cfg.LAST_FETCH_CHECKPOINT_CACHE, FETCH_CHECKPOINT_TS)
        sys.exit(0)

    new_df[cfg.COL_RESPONDENT_ID] = normalize_id_series(new_df[cfg.COL_RESPONDENT_ID])

    existing = load_cache()
    if not existing.empty:
        existing[cfg.COL_RESPONDENT_ID] = normalize_id_series(existing[cfg.COL_RESPONDENT_ID])
        combined = pd.concat([existing, new_df], ignore_index=True)
        # keep="last" so a respondent who appears in both the old cache
        # and this run's fetch (e.g. inside the checkpoint overlap
        # window) ends up with the freshest copy of their response.
        combined = combined.drop_duplicates(subset=cfg.COL_RESPONDENT_ID, keep="last")
    else:
        combined = new_df

    save_cache(combined)
    smf.save_checkpoint(cfg.LAST_FETCH_CHECKPOINT_CACHE, FETCH_CHECKPOINT_TS)
    log.info(
        f"Cache updated: {len(combined)} total responses cached "
        f"({len(new_df)} new/updated this run)."
    )


if __name__ == "__main__":
    main()