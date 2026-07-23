# pipeline_config.py
# ── All pipeline settings in one place ────────────────────────────────────
# Edit this file to configure the pipeline without touching run_pipeline.py

import os
from dotenv import load_dotenv

# Load variables from a local .env file (SURVEYMONKEY_ACCESS_TOKEN, etc.)
# In production (e.g. Streamlit Cloud), set these as secrets/env vars instead
# of shipping a .env file — load_dotenv() is a no-op if no .env is present.
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE        = os.path.join(BASE_DIR, "inputs", "new_export.xlsx")
INPUT_SHEET       = "Sheet"

MASTER_FILE       = os.path.join(BASE_DIR, "data", "master.xlsx")
PROCESSED_IDS     = os.path.join(BASE_DIR, "data", "processed_ids.csv")
# Separate dedup list for the lightweight daily raw-metrics job — kept
# apart from PROCESSED_IDS (which tracks what's gone through full NLP)
# so the daily and weekly jobs never double-count the same respondent.
PROCESSED_IDS_RAW = os.path.join(BASE_DIR, "data", "processed_ids_raw.csv")
TAGS_FILE         = os.path.join(BASE_DIR, "data", "tags.csv")
MANUAL_TAGS_FILE  = os.path.join(BASE_DIR, "data", "manuallytagged_complete.csv")

# 'Last successful API fetch' checkpoint for the weekly full pipeline
# (run_pipeline.py). Lets survey_monkey_fetch.py ask SurveyMonkey for
# only responses created since this time, instead of re-pulling the
# entire ~12,800+ response history (~128 API calls) on every run — see
# _fetch_raw_responses() in run_pipeline.py. Deliberately separate from
# LAST_FETCH_CHECKPOINT_RAW below: the weekly and daily jobs each need
# their own independent checkpoint, since they run on different
# schedules and track different dedup lists (processed_ids.csv vs
# processed_ids_raw.csv) — sharing one checkpoint would let one job's
# fetch silently move the cutoff past responses the other job still
# needs to see.
LAST_FETCH_CHECKPOINT     = os.path.join(BASE_DIR, "data", "last_fetch_checkpoint.txt")
# Same idea, for the daily raw-metrics job (update_raw_metrics.py) —
# NOT YET WIRED UP. update_raw_metrics.py still does a full re-fetch
# every run; it needs the equivalent of the run_pipeline.py change
# applied to it separately, using this path.
LAST_FETCH_CHECKPOINT_RAW = os.path.join(BASE_DIR, "data", "last_fetch_checkpoint_raw.txt")

# ── Local survey cache (refresh_survey_cache.py) ───────────────────────────
# A once-a-day, no-NLP job (its own GitHub Action) that pulls new
# responses since ITS OWN checkpoint and appends them to a local CSV
# cache — separate from LAST_FETCH_CHECKPOINT and
# LAST_FETCH_CHECKPOINT_RAW for the same reason those two are separate
# from each other (see the comment above): each independent consumer of
# the API needs its own checkpoint, or one job's fetch could silently
# move the cutoff past responses another job still needs to see.
#
# Set USE_LOCAL_CACHE=true to make run_pipeline.py read straight from
# this cache instead of calling the SurveyMonkey API (or the manual
# inputs/ file) at all — lets you run the FULL pipeline, repeatedly,
# against real data with zero API calls, e.g. to preview the dashboard
# locally without burning quota. Requires refresh_survey_cache.py to
# have been run at least once first (to build the initial cache).
SURVEY_CACHE_FILE = os.path.join(BASE_DIR, "data", "survey_cache.csv")
LAST_FETCH_CHECKPOINT_CACHE = os.path.join(BASE_DIR, "data", "last_fetch_checkpoint_cache.txt")
USE_LOCAL_CACHE = os.getenv("USE_LOCAL_CACHE", "false").lower() == "true"
# If the cache's checkpoint is older than this, warn (don't fail) —
# signals refresh_survey_cache.py's daily job may have silently stopped
# running, without blocking a run on data that's still perfectly usable.
STALE_CACHE_WARNING_HOURS = 48

MODELS_DIR        = os.path.join(BASE_DIR, "models")
SVM_MODEL         = os.path.join(MODELS_DIR, "svm_final_model.pkl")
SVM_ENCODER       = os.path.join(MODELS_DIR, "svm_final_label_encoder.pkl")

OUTPUTS_LATEST    = os.path.join(BASE_DIR, "outputs", "latest")
OUTPUTS_ARCHIVE   = os.path.join(BASE_DIR, "outputs", "archive")
LOGS_DIR          = os.path.join(BASE_DIR, "logs")

# ── SurveyMonkey API settings ──────────────────────────────────────────────
# Set these in a .env file (local) or as environment variables / secrets
# (Streamlit Cloud). Never commit real values — see .env.example.
SM_ACCESS_TOKEN   = os.getenv("SURVEYMONKEY_ACCESS_TOKEN")
SM_SURVEY_ID      = os.getenv("SURVEYMONKEY_SURVEY_ID")
SM_PER_PAGE       = 100   # responses per page when paginating the API

# ── Testing ─────────────────────────────────────────────────────────────────
# Set SKIP_NLP=true (env var) to bypass classify/ABSA/entity extraction —
# useful for testing ingest → preprocess → save → dashboard plumbing fast,
# without loading the SVM/ABSA/spaCy models. NEVER use this output for
# real analysis — tag/sentiment/entity columns are placeholders.
SKIP_NLP = os.getenv("SKIP_NLP", "false").lower() == "true"

# Set SKIP_API=true (env var) to bypass the SurveyMonkey API entirely and
# read straight from the manual export at INPUT_FILE instead. Useful while
# the API's daily quota is exhausted, or to avoid burning quota during
# heavy local testing — export responses from SurveyMonkey yourself and
# drop the file at INPUT_FILE. NLP/metrics still run normally on whatever
# rows the file contains.
SKIP_API = os.getenv("SKIP_API", "false").lower() == "true"

# ── Column names ───────────────────────────────────────────────────────────
COL_RESPONDENT_ID = "Respondent ID"
COL_FEEDBACK      = "Feedback"
COL_FEEDBACK_CLEAN = "Feedback_clean"
COL_RATING        = "Rating"
COL_DATE          = "Start Date"
COL_FEEDBACK = "Is there anything you would like to tell us about your experience?"
COL_FEEDBACK_CLEAN = "Feedback_clean"

# ── SVM settings ───────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "all-mpnet-base-v2"
MIN_SAMPLES       = 8
SVM_C             = 5
PRIMARY_THRESHOLD   = 0.3
SECONDARY_THRESHOLD = 0.15

# ── ABSA settings ──────────────────────────────────────────────────────────
ABSA_MODEL        = "yangheng/deberta-v3-base-absa-v1.1"
ABSA_MIN_WORDS    = 2       # skip reviews shorter than this
ABSA_MAX_ASPECTS  = 3       # max aspects to check per review

# ── Summarisation settings ─────────────────────────────────────────────────
SUMMARY_MODEL     = "facebook/bart-large-cnn"
SUMMARY_MIN_REVIEWS = 5     # minimum reviews per group to summarise
SUMMARY_MAX_REVIEWS = 80    # sample size for large groups

# ── Entity extraction settings ─────────────────────────────────────────────
SPACY_MODEL       = "en_core_web_sm"
MIN_ENTITY_COUNT  = 3       # minimum mentions to include in aggregations

# ── Output settings ────────────────────────────────────────────────────────
AUDIENCE_OUTPUTS = {
    "research":    ["master", "entities", "absa_detail"],
    "product":     ["summaries", "entity_analysis", "priority_matrix"],
    "leadership":  ["executive_summary"],
}