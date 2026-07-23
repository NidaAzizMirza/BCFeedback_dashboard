# run_pipeline.py
# ── Weekly pipeline runner ─────────────────────────────────────────────────
# Usage: python run_pipeline.py
# Drop new SurveyMonkey export into inputs/new_export.xlsx then run.

import os
import sys
import shutil
import warnings
import logging
import traceback
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib
import re
import torch
import spacy
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import pipeline_config as cfg

import preprocessing as pre

warnings.filterwarnings("ignore")

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# ── Setup ──────────────────────────────────────────────────────────────────
RUN_DATE = datetime.now().strftime("%Y-%m-%d")
RUN_TS   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# UTC timestamp captured at pipeline start, in the DateString format
# SurveyMonkey's API uses (confirmed against date_created in their
# public docs, e.g. "2015-10-06T12:56:55+00:00"). Saved as the fetch
# checkpoint at the end of a successful run (see main()), so the NEXT
# run's start_created_at only asks for responses since this moment —
# instead of re-pulling the entire survey history every time. Captured
# once, at process start (a few seconds before the actual API call),
# which is deliberately conservative: worst case a tiny harmless overlap
# next run, never a gap.
FETCH_CHECKPOINT_TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

# ── RESET (manual only!) ───────────────────────────────────────────────────
# These lines used to run UNCONDITIONALLY on every execution, wiping the
# entire metrics.db every single time the pipeline ran — incompatible with
# any incremental/daily update design. Disabled by default. To actually
# reset the DB, run this from a separate one-off script or a Python shell:
#
#   from aggregation_db import reset_pipeline
#   reset_pipeline()              # preview what will be deleted
#   reset_pipeline(confirm=True)  # actually delete everything
#
# NEVER uncomment this in run_pipeline.py itself while running on a schedule.

# Create directories
for d in [cfg.OUTPUTS_LATEST, cfg.OUTPUTS_ARCHIVE,
          cfg.LOGS_DIR, cfg.MODELS_DIR,
          os.path.join(BASE_DIR := os.path.dirname(os.path.abspath(__file__)), "inputs"),
          os.path.join(BASE_DIR, "data")]:
    os.makedirs(d, exist_ok=True)

ARCHIVE_DIR = os.path.join(cfg.OUTPUTS_ARCHIVE, RUN_DATE)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(cfg.LOGS_DIR, f"pipeline_{RUN_DATE}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Pipeline log entry ─────────────────────────────────────────────────────
pipeline_log = {
    "run_date": RUN_DATE,
    "run_timestamp": RUN_TS,
    "new_reviews": 0,
    "total_master": 0,
    "status": "started",
    "error": "",
    "source": ""
}

def save_output(df_or_writer_fn, filename):
    """Save to both latest/ and archive/."""
    latest_path  = os.path.join(cfg.OUTPUTS_LATEST, filename)
    archive_path = os.path.join(ARCHIVE_DIR, filename)
    df_or_writer_fn(latest_path)
    df_or_writer_fn(archive_path)
    log.info(f"Saved: {filename}")

def append_pipeline_log():
    log_path = os.path.join(cfg.LOGS_DIR, "pipeline_log.csv")
    log_df = pd.DataFrame([pipeline_log])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)

# ══════════════════════════════════════════════════════════════════════════
# STEP 0 — Pre-preprocess
# ══════════════════════════════════════════════════════════════════════════

def step_preprocess_external(df, has_extra_header_row=True):
    log.info("=" * 60)
    log.info("STEP 2 — External preprocessing (preprocessing.py)")
    log.info("=" * 60)

    log.info(f"Columns before preprocessing: {df.columns.tolist()}")
    log.info(f"Rows before preprocessing: {len(df)}")

    # Run your preprocessing
    df = pre.preprocessing_first(df.copy(), has_extra_header_row=has_extra_header_row)

    log.info(f"Columns after preprocessing: {df.columns.tolist()}")
    log.info(f"Rows after preprocessing: {len(df)}")

    # Ensure pipeline column names are consistent
    # preprocessing.py renames to "Feedback" and creates "Feedback_clean"
    if "Feedback_clean" not in df.columns:
        log.error("Feedback_clean column not found after preprocessing!")
        log.error(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Preprocessing did not produce Feedback_clean column")

    # Sync config column name
    cfg.COL_FEEDBACK = "Feedback"
    cfg.COL_FEEDBACK_CLEAN = "Feedback_clean"

    log.info(f"Non-null Feedback_clean: {df['Feedback_clean'].notna().sum()}")
    log.info(f"Sample: {df['Feedback_clean'].head(3).tolist()}")
    log.info("Preprocessing complete ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — INGEST & DEDUPLICATE
# ══════════════════════════════════════════════════════════════════════════
def normalize_id_series(s):
    """
    Normalize respondent IDs to a canonical string form, stripping a
    trailing ".0" if present.

    Root cause this guards against: a single stray blank row anywhere in
    the Respondent ID column forces pandas to read the WHOLE column as
    float64 on read — every clean ID like "119161712636" silently becomes
    "119161712636.0". That one-character difference breaks every
    ID-based dedup check against processed_ids.csv (it never matches),
    causing the entire file to look "new" every time and — worse — real
    duplicate rows to accumulate in master.xlsx once feedback IDs are
    re-classified under the mismatched format. This makes ID comparisons
    robust regardless of whether a future export happens to have a blank
    row, rather than relying on the source file always being clean.
    """
    return s.astype(str).str.replace(r'\.0$', '', regex=True)


def step_ingest():
    log.info("=" * 60)
    log.info("STEP 1 — Ingest & deduplicate")
    log.info("=" * 60)

    df_new, source = _fetch_raw_responses()

    # Load processed IDs
    if os.path.exists(cfg.PROCESSED_IDS):
        processed = pd.read_csv(cfg.PROCESSED_IDS)
        processed_ids = set(normalize_id_series(processed["respondent_id"]).tolist())
        log.info(f"Found {len(processed_ids)} previously processed IDs")
    else:
        processed_ids = set()
        log.info("No processed IDs file found — treating all reviews as new")

    # Filter to new only
    df_new[cfg.COL_RESPONDENT_ID] = normalize_id_series(df_new[cfg.COL_RESPONDENT_ID])
    df_new = df_new[~df_new[cfg.COL_RESPONDENT_ID].isin(processed_ids)].copy()

    log.info(f"New reviews to process: {len(df_new)} (source: {source})")
    pipeline_log["new_reviews"] = len(df_new)
    pipeline_log["source"] = source

    if len(df_new) == 0:
        log.warning("No new reviews found — nothing to process.")
        log.warning("Check that the input file/API contains new Respondent IDs.")
        sys.exit(0)

    return df_new, source


def _fetch_raw_responses():
    """
    Three possible data sources, checked in this order:

    1. USE_LOCAL_CACHE=true — read straight from the local cache file
       built by refresh_survey_cache.py. Zero API calls. Intended for
       repeatedly testing the full pipeline / dashboard locally.
    2. SKIP_API=true — read from the manual export at inputs/new_export.xlsx.
    3. Otherwise — fetch from the SurveyMonkey API directly (see
       INCREMENTAL note below), falling back to the manual export file
       on any non-rate-limit failure.

    API fetches are INCREMENTAL: they only ask for responses created
    since the last saved checkpoint (cfg.LAST_FETCH_CHECKPOINT), instead
    of re-pulling the entire survey history (~12,800+ responses, ~128
    API calls) on every single run — this is what exhausted the daily
    500-request quota mid-run previously. The checkpoint itself is only
    advanced on a fully successful run, in main() — never here.

    Returns (df, source) where source is "cache", "api", or "file" —
    callers need this because "file" data has a spurious extra header
    row that "cache"/"api" data does not, and downstream preprocessing
    has to treat them differently.
    """
    import survey_monkey_fetch as smf

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
        if age_hours is None:
            log.warning(
                "Cache file exists but has no checkpoint — freshness unknown. "
                "Was it built by something other than refresh_survey_cache.py?"
            )
        elif age_hours > cfg.STALE_CACHE_WARNING_HOURS:
            log.warning(
                f"Local cache is {age_hours:.1f}h old (threshold: "
                f"{cfg.STALE_CACHE_WARNING_HOURS}h) — refresh_survey_cache.py's "
                f"daily job may have stopped running. Proceeding anyway; "
                f"data just may not include the most recent responses."
            )
        else:
            log.info(f"Cache is {age_hours:.1f}h old — within freshness threshold.")

        df_cache = pd.read_csv(cfg.SURVEY_CACHE_FILE, dtype={cfg.COL_RESPONDENT_ID: str})
        log.info(f"Loaded {len(df_cache)} responses from local cache")
        # "cache" data is already flattened by fetch_survey_as_dataframe
        # (same shape as "api"), so it correctly gets has_extra_header_row
        # = False and is correctly excluded from the "api"-only checkpoint
        # saves in main() — both checks already key off exact string
        # equality with "api", so no changes needed there.
        return df_cache, "cache"

    if cfg.SKIP_API:
        # Manual mode: you're exporting from SurveyMonkey yourself and
        # dropping the file at cfg.INPUT_FILE. Skip the API call entirely
        # — no wasted quota, no rate-limit noise, and it can't abort.
        log.info("SKIP_API is set — skipping SurveyMonkey API, reading manual export directly.")
        if not os.path.exists(cfg.INPUT_FILE):
            raise FileNotFoundError(
                f"SKIP_API is set but no manual export file found at {cfg.INPUT_FILE}. "
                f"Export your responses from SurveyMonkey and save them there first."
            )
        df_file = pd.read_excel(cfg.INPUT_FILE, sheet_name=cfg.INPUT_SHEET)
        log.info(f"Loaded {len(df_file)} reviews from manual export ({cfg.INPUT_FILE})")
        return df_file, "file"

    try:
        log.info("Attempting to fetch responses directly from SurveyMonkey API...")

        checkpoint = smf.load_checkpoint(cfg.LAST_FETCH_CHECKPOINT)
        fetch_since = smf.checkpoint_with_overlap(checkpoint)
        df_api = smf.fetch_survey_as_dataframe(start_created_at=fetch_since)

        if len(df_api) == 0:
            if fetch_since:
                # Incremental fetch (we have a checkpoint) genuinely
                # returning nothing new is a normal outcome — e.g. a
                # quiet week — NOT a failure. Falling back to the manual
                # export here would be wrong: it would silently
                # reprocess whatever's sitting in inputs/, which has
                # nothing to do with "no new API responses since X".
                log.info("No new responses since last checkpoint — nothing to process.")
                pipeline_log["status"] = "no_new_responses"
                append_pipeline_log()
                sys.exit(0)
            else:
                # No checkpoint yet (first-ever run) AND zero responses
                # is unusual — keep the original behaviour of falling
                # through to the manual-file fallback below.
                raise ValueError("API returned zero responses on a full fetch")

        log.info(f"Loaded {len(df_api)} reviews via SurveyMonkey API")
        return df_api, "api"
    except smf.RateLimitExceeded as e:
        # A 429 is transient and retryable — the next scheduled run will
        # likely succeed once quota resets. Falling back to the stale
        # manual export here would silently process bad/empty data and
        # mark real respondent IDs as "done" before they've actually been
        # captured. Better to abort loudly and let the schedule retry.
        log.error(
            f"SurveyMonkey daily/rate quota exhausted — ABORTING this run "
            f"rather than falling back to a stale local file. Reason: {e}"
        )
        pipeline_log["status"] = "aborted_rate_limit"
        pipeline_log["error"] = str(e)
        append_pipeline_log()
        sys.exit(1)
    except Exception as e:
        # Non-rate-limit failures (auth error, network issue, API not
        # configured, etc.) aren't time-based, so falling back to the
        # manual export is still the right move here.
        log.warning(f"SurveyMonkey API fetch failed or not configured ({e})")
        log.warning("Falling back to manual Excel export...")

    if not os.path.exists(cfg.INPUT_FILE):
        raise FileNotFoundError(
            f"No input file found at {cfg.INPUT_FILE}\n"
            "Either configure the SurveyMonkey API (.env) or drop your "
            "export there and re-run."
        )

    df_file = pd.read_excel(
        cfg.INPUT_FILE,
        sheet_name=cfg.INPUT_SHEET,
        dtype={cfg.COL_RESPONDENT_ID: str}  # ← force string, prevents scientific notation
    )
    log.info(f"Loaded {len(df_file)} reviews from input file")
    return df_file, "file"

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESS
# ══════════════════════════════════════════════════════════════════════════
def step_preprocess(df):
    log.info("=" * 60)
    log.info("STEP 2 — Preprocess")
    log.info("=" * 60)

    df = df.copy()

    # Drop rows with no feedback
    before = len(df)
    df = df.dropna(subset=[cfg.COL_FEEDBACK]).copy()
    df[cfg.COL_FEEDBACK] = df[cfg.COL_FEEDBACK].astype(str)

    # Basic cleaning — lowercase, strip whitespace, remove excess spaces
    df[cfg.COL_FEEDBACK_CLEAN] = (
        df[cfg.COL_FEEDBACK]
        .str.lower()
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace(r'[^\w\s\'\-\.\,\!\?\£]', '', regex=True)
    )

    # Drop very short responses
    df = df[df[cfg.COL_FEEDBACK_CLEAN].str.split().str.len() >= 3].copy()

    log.info(f"After cleaning: {len(df)} reviews (dropped {before - len(df)})")
    return df

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — CLASSIFY (SVM + BOOST + FALLBACK + MULTI-LABEL)
# ══════════════════════════════════════════════════════════════════════════

# ── Keyword dictionaries (imported from your existing pipeline) ────────────
KEYWORD_BOOST = {
    "Fees too high": [
        "fee", "fees", "expensive", "too high", "cost", "overcharge",
        "fee calculator", "planning fee", "service charge", "fees are"
    ],
    "Fees & charges": [
        "fees and charges", "fee structure", "fee breakdown",
        "transparent fees", "fee quote", "additional fee", "extra charge"
    ],
    "Document uploading": [
        "upload", "uploading", "file", "pdf", "attachment",
        "file size", "10mb", "file format", "file limit"
    ],
    "LPI tool": [
        "location plan", "boundary", "lpi", "red line", "blue line",
        "site plan", "requestaplan", "drawing tool", "mapping tool",
        "drawing tools", "which drawings", "drawings required",
    ],
    "Suggestions": [
        "would be good if", "would help if", "it would help",
        "would be useful", "would be helpful", "please add",
        "could you add", "feature request"
    ],
    "Negative experience": [
        "not easy", "not very easy", "not straightforward", "not clear",
        "step backwards", "worse than", "much worse", "far worse",
        "terrible", "awful", "horrible", "useless", "waste of time",
        "frustrating", "poor", "dreadful", "appalling", "worst",
    ],
    "Positive experience/ other praises": [
        "easy to use", "very easy to use", "really easy to use",
        "straightforward process", "excellent service", "brilliant service",
        "great service", "well designed", "works perfectly",
    ],
    "Too complex": [
        "too complex", "over complicated", "overly complicated",
        "too many steps", "too complicated"
    ],
    "Confusing": [
        "confusing", "confused", "not intuitive",
        "hard to understand", "misleading", "baffling",
    ],
    "Bugs/ glitch": [
        "bug", "glitch", "not working", "broken",
        "system issue", "technical issue", "error message",
        "keeps freezing", "freezing", "stalling",
    ],
    "Crash/ data loss/ error message": [
        "crash", "crashed", "crashing", "lost my work", "data lost",
        "session expired", "timed out", "timeout", "lost data",
        "keeps crashing", "lost everything",
    ],
    "Payments": [
        "payment system", "pay instantly", "banking app",
        "manual pay", "manually pay", "payment confirmation",
        "payment method", "card payment", "pay through",
        "new payment system", "nomination fee", "bank transfer",
    ],
}

KEYWORD_BOOST_STRENGTH = {
    "Too complex": 0.15,
    "Confusing": 0.15,
    "Negative experience": 0.15,
    "Document uploading": 0.3,
    "Fees too high": 0.3,
    "Payments": 0.35,
    "Bugs/ glitch": 0.3,
    "Crash/ data loss/ error message": 0.35,
}

HARD_FALLBACK = {
    "Crash/ data loss/ error message": (
        ["crash", "crashed", "crashing", "lost my work", "data lost",
         "error message", "session expired", "timed out", "timeout",
         "lost data", "keeps crashing", "lost everything",
         "repeatedly crashes", "system crashed", "portal crashed"],
        "Challenges and Workarounds"
    ),
    "Bugs/ glitch": (
        ["bug", "glitch", "not working", "broken", "system issue",
         "technical issue", "keeps freezing", "freezing", "stalling"],
        "Challenges and Workarounds"
    ),
    "Payments": (
        ["payment system", "pay instantly", "banking app", "manual pay",
         "payment confirmation", "payment method", "card payment",
         "bank transfer", "payment received", "payment failed",
         "nomination fee", "nominate", "overpayment", "refund"],
        "Payments"
    ),
    "Lack of guidance": (
        ["more guidance", "clearer guidance", "lack of guidance",
         "no guidance", "need guidance", "more explanation",
         "clearer instructions", "not sure", "not sure what",
         "not sure which", "didn't know", "had no idea", "not obvious"],
        "Guidance, clarity & jargon"
    ),
    "Jargon": (
        ["jargon", "technical terms", "technical language", "plain english",
         "layperson", "lay person", "not a professional", "not a builder",
         "tooltips", "tooltip", "abbreviation", "acronym"],
        "Guidance, clarity & jargon"
    ),
    "Guidance": (
        ["example", "examples", "template", "templates",
         "step by step", "high level guide", "simplify process",
         "make it clearer", "make it easier", "faq"],
        "Guidance, clarity & jargon"
    ),
    "Support desk/ Human support": (
        ["had to call", "had to phone", "called helpline", "speak to someone",
         "real person", "can you icer", "phone advice", "phone support",
         "over the phone", "live chat", "help desk", "helpdesk"],
        "Customer support & human assistance"
    ),
    "Tree works": (
        ["tree", "trees", "tree works", "tree surgery", "arborist",
         "tpo", "tree preservation", "woodland", "conservation area"],
        "Planning App/work types"
    ),
    "BNG / Biodiversity metric": (
        ["bng", "biodiversity", "biodiversity net gain",
         "biodiversity metric", "habitat", "small site metric", "lxsm"],
        "Planning App/work types"
    ),
    "Discharge conditions": (
        ["discharge", "discharge conditions", "condition discharge",
         "planning condition", "discharging conditions",
         "approval of conditions"],
        "Planning App/work types"
    ),
    "Missing/ suggested features": (
        ["missing feature", "add a feature", "it would be great if",
         "wish you could", "would be good if", "would be useful",
         "would be helpful", "please add", "could you add"],
        "Missing features & feature requests"
    ),
    "General UX complexity / time": (
        ["takes too long", "time consuming", "lengthy process",
         "long winded", "laborious", "tedious",
         "hours to", "took me hours"],
        "General UX"
    ),
    "Form completion": (
        ["form completion", "filling in", "fill in",
         "completing the form", "tick boxes", "tick box"],
        "Forms & application details"
    ),
    "LPI tool / Drawing tools": (
        ["drawing tool", "drawing tools", "sketch tool", "boundary tool",
         "which drawings", "drawings required", "drawing required",
         "what drawings", "plans required", "lpi tool",
         "red line boundary", "blue line"],
        "Location plans, addresses and mapping"
    ),
}

POSITIVE_TAGS = {
    "Positive experience/ other praises",
    "Easy to use/ Straightforward",
    "Easy to navigate",
    "Easy to understand"
}

NEGATION_SIGNALS = [
    "not easy", "not very easy", "not straightforward", "not clear",
    "not intuitive", "not obvious", "not user friendly", "not great",
    "not good", "not helpful", "not simple", "far from easy",
    "hard to", "difficult to", "struggled", "couldn't", "could not",
    "wasn't easy", "was not easy", "isn't easy", "is not easy",
]

def match_keywords(text, keywords):
    text_lower = text.lower()
    for kw in keywords:
        match = re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower)
        if match:
            preceding = text_lower[:match.start()].split()[-3:]
            negations = ["not", "n't", "never", "no", "hardly", "barely",
                         "wasn't", "weren't", "isn't", "aren't",
                         "didn't", "don't", "doesn't"]
            if any(neg in preceding for neg in negations):
                continue
            return True
    return False

def apply_keyword_boost(text, svm_proba, label_encoder, boost=0.3):
    proba = svm_proba.copy()
    classes = list(label_encoder.classes_)
    for tag, keywords in KEYWORD_BOOST.items():
        if tag in classes and match_keywords(text, keywords):
            idx = classes.index(tag)
            strength = KEYWORD_BOOST_STRENGTH.get(tag, boost)
            proba[idx] = min(1.0, proba[idx] + strength)
    proba = proba / proba.sum()
    return classes[np.argmax(proba)], proba.max()

def apply_hard_fallback(text, svm_tag, svm_conf, tag_to_group,
                        confidence_threshold=cfg.PRIMARY_THRESHOLD):
    if svm_conf >= confidence_threshold:
        return svm_tag, tag_to_group.get(svm_tag, "Miscellaneous"), "svm"
    for granular_tag, (keywords, group) in HARD_FALLBACK.items():
        if match_keywords(text, keywords):
            return granular_tag, group, "fallback"
    return svm_tag, tag_to_group.get(svm_tag, "Miscellaneous"), "svm_low_conf"

def get_multilabel_tags(text, svm_proba, svm_pred, label_encoder,
                        tag_to_group, boost=0.3):
    classes = list(label_encoder.classes_)
    text_lower = text.lower()

    proba = svm_proba.copy()
    for tag, keywords in KEYWORD_BOOST.items():
        if tag in classes and match_keywords(text, keywords):
            idx = classes.index(tag)
            strength = KEYWORD_BOOST_STRENGTH.get(tag, boost)
            proba[idx] = min(1.0, proba[idx] + strength)
    proba = proba / proba.sum()

    candidate_indices = np.where(proba >= cfg.SECONDARY_THRESHOLD)[0]
    candidate_tags = sorted(
        [(classes[i], proba[i]) for i in candidate_indices],
        key=lambda x: x[1], reverse=True
    )
    if not candidate_tags:
        candidate_tags = [(classes[np.argmax(proba)], proba.max())]

    primary_tag, primary_conf = candidate_tags[0]

    if primary_conf < cfg.PRIMARY_THRESHOLD:
        primary_tag, primary_group, method = apply_hard_fallback(
            text, primary_tag, primary_conf, tag_to_group
        )
    else:
        primary_group = tag_to_group.get(primary_tag, "Miscellaneous")
        method = "svm"

    text_has_negation = any(s in text_lower for s in NEGATION_SIGNALS)

    secondary_tags, secondary_groups, seen = [], [], {primary_tag}
    for tag, conf in candidate_tags[1:]:
        if tag in seen:
            continue
        if tag in POSITIVE_TAGS and text_has_negation:
            continue
        secondary_tags.append(tag)
        secondary_groups.append(tag_to_group.get(tag, "Miscellaneous"))
        seen.add(tag)

    if primary_conf < cfg.PRIMARY_THRESHOLD:
        for granular_tag, (keywords, group) in HARD_FALLBACK.items():
            if granular_tag in seen:
                continue
            if match_keywords(text, keywords):
                secondary_tags.append(granular_tag)
                secondary_groups.append(group)
                break

    return (
        primary_tag, primary_group,
        ", ".join(secondary_tags),
        ", ".join(secondary_groups),
        round(float(primary_conf), 3),
        method
    )

def step_classify(df):
    log.info("=" * 60)
    log.info("STEP 3 — Classify")
    log.info("=" * 60)

    # Load tag lookup
    tags_ref = pd.read_csv(cfg.TAGS_FILE)[["Title", "Group"]].copy()
    tags_ref.columns = ["tag", "tag_group"]
    tags_ref["tag"] = tags_ref["tag"].str.strip()
    tag_to_group = dict(zip(tags_ref["tag"], tags_ref["tag_group"]))

    # Load models
    log.info("Loading SVM model...")
    svm = joblib.load(cfg.SVM_MODEL)
    le  = joblib.load(cfg.SVM_ENCODER)
    embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    log.info("Models loaded ✓")

    # Generate embeddings
    log.info("Generating embeddings...")
    embeddings = embedding_model.encode(
        df[cfg.COL_FEEDBACK_CLEAN].tolist(),
        batch_size=32, show_progress_bar=True
    )

    all_proba = svm.predict_proba(embeddings)
    all_pred  = svm.predict(embeddings)

    results = []
    for text, pred, proba in zip(
        df[cfg.COL_FEEDBACK_CLEAN], all_pred, all_proba
    ):
        primary_tag, primary_group, secondary_tags, secondary_groups, conf, method = \
            get_multilabel_tags(text, proba, pred, le, tag_to_group)
        results.append({
            "primary_tag":          primary_tag,
            "primary_tag_group":    primary_group,
            "secondary_tags":       secondary_tags,
            "secondary_tag_groups": secondary_groups,
            "svm_confidence":       conf,
            "prediction_method":    method,
        })

    results_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    log.info(f"Classification complete — method breakdown:")
    log.info(df["prediction_method"].value_counts().to_string())
    return df

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — ABSA
# ══════════════════════════════════════════════════════════════════════════
ABSA_ASPECTS = {
    "Document upload and handling":       ["document upload", "file upload", "uploading documents", "file size", "file format"],
    "Guidance, clarity & jargon":         ["guidance", "instructions", "jargon", "terminology", "clarity"],
    "Fees, charges and quotes":           ["fees", "charges", "cost", "pricing", "fee calculator"],
    "Payments":                           ["payment", "payment system", "card payment", "bank transfer"],
    "Location plans, addresses and mapping": ["location plan", "boundary", "drawing tool", "site plan", "mapping"],
    "Forms & application details":        ["form", "questions", "application form"],
    "General UX":                         ["navigation", "interface", "design", "usability"],
    "Challenges and Workarounds":         ["crash", "bug", "error", "system issue", "glitch"],
    "Planning App/work types":            ["tree works", "biodiversity", "discharge conditions", "application type"],
    "Customer support & human assistance":["support", "helpdesk", "phone support"],
    "Missing features & feature requests":["missing feature", "suggestion", "improvement"],
    "Overall Positive Experience":        ["overall experience", "ease of use", "user experience"],
}

def step_absa(df):
    log.info("=" * 60)
    log.info("STEP 4 — ABSA")
    log.info("=" * 60)

    device = 0 if torch.backends.mps.is_available() else -1
    log.info(f"Using device: {'MPS' if device == 0 else 'CPU'}")

    log.info("Loading ABSA model...")
    absa_pipe = hf_pipeline(
        "text-classification",
        model=cfg.ABSA_MODEL,
        tokenizer=cfg.ABSA_MODEL,
        device=device
    )
    log.info("ABSA model loaded ✓")

    _absa_error_count = [0]   # mutable counter closed over below
    _absa_seen_labels = set()

    def get_absa_sentiment(text, aspect):
        try:
            result = absa_pipe(
                f"{text} [SEP] {aspect}",
                truncation=True, max_length=512
            )
            raw_label = result[0]["label"]
            _absa_seen_labels.add(raw_label)
            return raw_label.lower(), round(result[0]["score"], 3)
        except Exception as e:
            _absa_error_count[0] += 1
            if _absa_error_count[0] <= 5:
                log.error(f"ABSA call failed (error #{_absa_error_count[0]}): {e}")
            return "neutral", 0.0

    def run_absa(text, tag_group):
        results = {}
        sentiment, score = get_absa_sentiment(text, "overall experience")
        results["overall experience"] = {"sentiment": sentiment, "confidence": score}
        for aspect in ABSA_ASPECTS.get(tag_group, [])[:cfg.ABSA_MAX_ASPECTS]:
            sentiment, score = get_absa_sentiment(text, aspect)
            if score >= 0.6:
                results[aspect] = {"sentiment": sentiment, "confidence": score}
        return results

    def summarise_absa(d):
        aspects, sentiments, overall = [], [], ""
        for aspect, data in d.items():
            if aspect == "overall experience":
                overall = data["sentiment"]
                continue
            aspects.append(aspect)
            sentiments.append(f"{aspect}: {data['sentiment']} ({data['confidence']})")
        return ", ".join(aspects), ", ".join(sentiments), overall

    def get_grouping_sentiment(row):
        aspects = str(row.get("absa_aspect_sentiments", ""))
        overall = str(row.get("absa_overall_sentiment", ""))
        if aspects and aspects not in ["nan", ""]:
            neg = aspects.count("negative")
            pos = aspects.count("positive")
            neu = aspects.count("neutral")
            if neg > pos and neg > neu: return "negative"
            if pos > neg and pos > neu: return "positive"
            return "neutral"
        return overall if overall not in ["nan", ""] else "neutral"

    # ── Diagnostic — word count distribution ──────────────────────────────
    word_counts = df[cfg.COL_FEEDBACK_CLEAN].str.split().str.len()
    log.info(f"Word count distribution:\n{word_counts.describe()}")
    log.info(f"Reviews with >= 2 words: {(word_counts >= 2).sum()}")
    log.info(f"Reviews with >= 5 words: {(word_counts >= 5).sum()}")

    # ── Use Feedback column if richer than Feedback_clean ─────────────────
    text_col = cfg.COL_FEEDBACK_CLEAN
    if "Feedback" in df.columns:
        avg_clean = df[cfg.COL_FEEDBACK_CLEAN].str.split().str.len().mean()
        avg_orig  = df["Feedback"].str.split().str.len().mean()
        if avg_orig > avg_clean:
            text_col = "Feedback"
            log.info(f"Using 'Feedback' column for ABSA "
                     f"(avg {avg_orig:.1f} words vs {avg_clean:.1f} clean)")
        else:
            log.info(f"Using 'Feedback_clean' column for ABSA "
                     f"(avg {avg_clean:.1f} words)")

    # ── Filter to reviews with enough words ───────────────────────────────
    df_absa = df[df[text_col].str.split().str.len() >= 2].copy()
    total = len(df_absa)
    log.info(f"Running ABSA on {total} reviews (using column: {text_col})")

    # ── Guard: skip ABSA if nothing to process ────────────────────────────
    if total == 0:
        log.warning("No reviews meet minimum word count for ABSA — skipping")
        df["absa_aspects"]           = ""
        df["absa_aspect_sentiments"] = ""
        df["absa_overall_sentiment"] = ""
        df["grouping_sentiment"]     = "neutral"
        return df

    # ── Run ABSA ───────────────────────────────────────────────────────────
    absa_results = []
    for i, (_, row) in enumerate(df_absa.iterrows()):
        if i % 100 == 0:
            log.info(f"  ABSA progress: {i}/{total}")
        absa_results.append(
            run_absa(row[text_col], row["primary_tag_group"])
        )

    df_absa["absa_results"] = absa_results

    log.info(f"ABSA raw labels seen: {_absa_seen_labels}")
    log.info(f"ABSA call failures: {_absa_error_count[0]} / {total * (1 + cfg.ABSA_MAX_ASPECTS)} approx calls")
    seen_lower = {label.lower() for label in _absa_seen_labels}
    if _absa_seen_labels and not seen_lower & {"positive", "negative", "neutral"}:
        log.error(
            "ABSA model is returning labels that don't match {'positive','negative','neutral'} "
            f"— got {_absa_seen_labels}. Sentiment counts will be wrong until this is fixed."
        )

    # ── Safe column assignment ─────────────────────────────────────────────
    absa_expanded = df_absa["absa_results"].apply(
        lambda x: pd.Series(summarise_absa(x))
    )
    absa_expanded.columns = [
        "absa_aspects", "absa_aspect_sentiments", "absa_overall_sentiment"
    ]
    df_absa = pd.concat([df_absa.reset_index(drop=True),
                         absa_expanded.reset_index(drop=True)], axis=1)

    df_absa["grouping_sentiment"] = df_absa.apply(get_grouping_sentiment, axis=1)

    # ── Merge back to full df ──────────────────────────────────────────────
    merge_cols = [text_col, "absa_aspects", "absa_aspect_sentiments",
                  "absa_overall_sentiment", "grouping_sentiment"]

    # ── Merge back to full df ──────────────────────────────────────────────
    df = df.merge(
        df_absa[[cfg.COL_RESPONDENT_ID, "absa_aspects", "absa_aspect_sentiments",
                 "absa_overall_sentiment", "grouping_sentiment"]],
        on=cfg.COL_RESPONDENT_ID,
        how="left"
    )

    # Drop duplicate text column if merge created one
    if text_col != cfg.COL_FEEDBACK_CLEAN and text_col in df.columns:
        df = df.drop(columns=[text_col])

    df["grouping_sentiment"] = df["grouping_sentiment"].fillna("neutral")

    # Drop duplicate Feedback columns from ABSA merge
    if "Feedback_y" in df.columns:
        df = df.drop(columns=["Feedback_y"])
    if "Feedback_x" in df.columns:
        df = df.rename(columns={"Feedback_x": "Feedback"})
    print(f"printing this {df}")
    log.info("ABSA complete ✓")
    log.info(f"Sentiment distribution: "
             f"{df['grouping_sentiment'].value_counts().to_dict()}")
    return df

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — ENTITY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════
COUNCIL_NAMES = [
    "cornwall", "cornwall council", "st albans", "chelmsford",
    "oxford", "oxford city council", "rushmore", "northampton",
    "bristol", "leeds", "manchester", "birmingham", "london",
]

FEATURE_NAMES = {
    "lpi tool": "LPI tool", "lpi": "LPI tool",
    "location plan tool": "LPI tool", "boundary tool": "LPI tool",
    "drawing tool": "LPI tool", "requestaplan": "RequestAPlan",
    "document upload": "Document upload", "file upload": "Document upload",
    "uploading documents": "Document upload",
    "payment system": "Payment system", "payment portal": "Payment system",
    "nomination": "Nomination/payment", "fee calculator": "Fee calculator",
    "service charge": "Service charge",
    "application form": "Application form", "oil form": "Oil form",
    "cil form": "CIL form", "bng metric": "BNG metric tool",
    "biodiversity metric": "BNG metric tool",
    "dropdown": "Dropdown menu", "drop down": "Dropdown menu",
    "progress indicator": "Progress indicator",
    "helpline": "Helpline", "help desk": "Help desk", "helpdesk": "Help desk",
    "planning portal": "Planning Portal", "portal": "Planning Portal",
    "discharge conditions": "Discharge conditions",
    "listed building": "Listed building consent",
    "householder": "Householder application",
    "tree works": "Tree works application",
    "lawful development": "Lawful development certificate",
}

ERROR_PATTERNS = [
    r"(?:system|portal|page|site)\s+(?:crashed|crash|crashing|froze|freezing|stalled|went down|timed out)",
    r"(?:keeps?|kept)\s+(?:crashing|freezing|stalling|logging (?:me )?out|timing out)",
    r"(?:lost|deleted|removed)\s+(?:my|all|the)?\s*(?:data|work|information|progress|application|documents?)",
    r"(?:unable|couldn't|can't|cannot|could not)\s+(?:upload|submit|save|load|open|access|find|complete|pay)",
    r"(?:file|document|pdf|excel)\s+(?:not accepted|rejected|failed|won't upload|not loading)",
    r"(?:postcode|post code|address)\s+(?:not (?:found|recognised|accepted|working)|rejected|invalid)",
    r"(?:payment|card)\s+(?:failed|declined|not (?:working|going through|processing))",
    r"(?:won't|doesn't|did not|didn't)\s+(?:accept|recognise|recognize|work|load|save|submit)",
    r"(?:10mb|file size|size limit|upload limit)",
    r"(?:square brackets|special characters?|permitted characters?)",
]

def step_entities(df):
    log.info("=" * 60)
    log.info("STEP 5 — Entity extraction")
    log.info("=" * 60)

    try:
        nlp = spacy.load(cfg.SPACY_MODEL)
        log.info("spaCy model loaded ✓")
    except OSError:
        log.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise

    def extract_councils(text):
        tl = text.lower()
        return list(set(
            c.title() for c in COUNCIL_NAMES
            if re.search(r'\b' + re.escape(c) + r'\b', tl)
        ))

    def extract_features(text):
        tl = text.lower()
        return list(set(
            v for k, v in FEATURE_NAMES.items()
            if re.search(r'\b' + re.escape(k) + r'\b', tl)
        ))

    def extract_errors(text):
        tl = text.lower()
        found = []
        for pat in ERROR_PATTERNS:
            for m in re.findall(pat, tl):
                c = m.strip().rstrip(".,;")
                if len(c) > 5:
                    found.append(c)
        return list(set(found))

    def extract_fees(text):
        return list(set(re.findall(r'£[\d,]+(?:\.\d{2})?', text)))

    total = len(df)
    councils_l, features_l, errors_l, fees_l = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            log.info(f"  Entity progress: {i}/{total}")
        t = row[cfg.COL_FEEDBACK_CLEAN]
        councils_l.append(", ".join(extract_councils(t)))
        features_l.append(", ".join(extract_features(t)))
        errors_l.append(", ".join(extract_errors(t)))
        fees_l.append(", ".join(extract_fees(t)))

    df["entities_councils"]  = councils_l
    df["entities_features"]  = features_l
    df["entities_errors"]    = errors_l
    df["entities_fees"]      = fees_l

    log.info("Entity extraction complete ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════
def mark_ids_processed(id_series):
    """
    Append respondent IDs to processed_ids.csv, deduplicated.

    Deliberately takes ALL respondent IDs fetched this run — not just the
    ones that ended up with usable feedback — so that respondents who
    only left a star rating (no free-text comment) are marked "seen" and
    never re-fetched/re-preprocessed on every future run. Previously only
    the post-preprocessing, feedback-bearing subset got marked, so the
    no-comment population silently accumulated forever and was
    re-processed from scratch on every run (see run_pipeline.py history:
    "New reviews to process" staying ~7600+ day after day for a backlog
    that should have been shrinking).
    """
    new_ids = pd.DataFrame({
        "respondent_id": normalize_id_series(id_series),
        "processed_date": RUN_DATE,
    }).drop_duplicates(subset=["respondent_id"])

    if os.path.exists(cfg.PROCESSED_IDS):
        existing_ids = pd.read_csv(cfg.PROCESSED_IDS)
        existing_ids["respondent_id"] = normalize_id_series(existing_ids["respondent_id"])
        combined = pd.concat([existing_ids, new_ids], ignore_index=True)
        combined = combined.drop_duplicates(subset=["respondent_id"], keep="first")
    else:
        combined = new_ids

    combined.to_csv(cfg.PROCESSED_IDS, index=False)
    return len(new_ids)


def step_save(df):
    log.info("=" * 60)
    log.info("STEP 6 — Save outputs")
    log.info("=" * 60)

    # ── Update master file ─────────────────────────────────────────────────
    if os.path.exists(cfg.MASTER_FILE):
        master = pd.read_excel(cfg.MASTER_FILE)
        master = pd.concat([master, df], ignore_index=True)
        before = len(master)
        # Normalize IDs before comparing — otherwise a respondent stored
        # as "119161712636" in one batch and "119161712636.0" in another
        # (see normalize_id_series docstring) looks like two different
        # people and the dedup below silently fails to catch them.
        master[cfg.COL_RESPONDENT_ID] = normalize_id_series(master[cfg.COL_RESPONDENT_ID])
        # Safety net: keep the newest row per respondent in case the same
        # respondent ever gets processed twice (e.g. processed_ids.csv
        # reset during testing) — prevents silent double-counting later.
        master = master.drop_duplicates(subset=[cfg.COL_RESPONDENT_ID], keep="last")
        if len(master) < before:
            log.warning(f"Dropped {before - len(master)} duplicate respondent rows on save")
        log.info(f"Appended to master — total records: {len(master)}")
    else:
        master = df.copy()
        master[cfg.COL_RESPONDENT_ID] = normalize_id_series(master[cfg.COL_RESPONDENT_ID])
        log.info(f"Created new master file — {len(master)} records")

    master.to_excel(cfg.MASTER_FILE, index=False)
    pipeline_log["total_master"] = len(master)

    # NOTE: processed_ids.csv is now updated separately in main(), using
    # the FULL set of respondent IDs fetched this run (df_raw) rather than
    # just this feedback-bearing df — see mark_ids_processed() above.

    # ── Save detailed results ──────────────────────────────────────────────
    def save_results(path):
        df.to_excel(path, index=False)

    save_output(save_results, "results.xlsx")

    # ── Save master copy to archive ────────────────────────────────────────
    master.to_excel(
        os.path.join(ARCHIVE_DIR, "master_snapshot.xlsx"), index=False
    )

    # ── Audience-specific outputs ──────────────────────────────────────────

    # Research — full detail
    def save_research(path):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="New reviews", index=False)
            master.to_excel(writer, sheet_name="All reviews", index=False)

    save_output(save_research, "research_output.xlsx")

    # Product — priority issues
    def save_product(path):
        priority_cols = [
            cfg.COL_FEEDBACK_CLEAN, "primary_tag", "primary_tag_group",
            "secondary_tags", "svm_confidence", "prediction_method",
            "grouping_sentiment", "absa_aspect_sentiments",
            "entities_features", "entities_errors", "entities_fees"
        ]
        available = [c for c in priority_cols if c in df.columns]
        negative = df[df["grouping_sentiment"] == "negative"][available].copy()

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            negative.sort_values(
                "primary_tag_group"
            ).to_excel(writer, sheet_name="Negative reviews", index=False)

            # Tag group summary
            summary = df.groupby(
                ["primary_tag_group", "grouping_sentiment"]
            ).size().unstack(fill_value=0).reset_index()
            summary.to_excel(writer, sheet_name="Tag group summary", index=False)

            # Feature sentiment
            feat_sent = defaultdict(
                lambda: {"positive": 0, "negative": 0, "neutral": 0}
            )
            for _, row in df.iterrows():
                features = str(row.get("entities_features", "")).split(", ")
                sentiment = str(row.get("grouping_sentiment", "neutral")).strip()
                if sentiment not in ["positive", "negative", "neutral"]:
                    sentiment = "neutral"
                for f in features:
                    f = f.strip()
                    if f:
                        feat_sent[f][sentiment] += 1

            feat_rows = []
            for f, c in feat_sent.items():
                total = sum(c.values())
                if total >= cfg.MIN_ENTITY_COUNT and f not in ["Planning Portal", ""]:
                    feat_rows.append({
                        "feature": f,
                        "total": total,
                        "negative": c["negative"],
                        "positive": c["positive"],
                        "neutral": c["neutral"],
                        "negative_pct": round(c["negative"] / total * 100, 1)
                    })

            if feat_rows:
                feat_df = pd.DataFrame(feat_rows).sort_values(
                    "negative_pct", ascending=False
                )
            else:
                feat_df = pd.DataFrame(
                    columns=["feature", "total", "negative", "positive",
                             "neutral", "negative_pct"]
                )

            feat_df.to_excel(writer, sheet_name="Feature sentiment", index=False)

    save_output(save_product, "product_output.xlsx")

    # Leadership — high-level metrics
    def save_leadership(path):
        total = len(df)
        neg_pct = round(
            (df["grouping_sentiment"] == "negative").mean() * 100, 1
        )
        pos_pct = round(
            (df["grouping_sentiment"] == "positive").mean() * 100, 1
        )

        metrics = pd.DataFrame([{
            "run_date": RUN_DATE,
            "new_reviews": total,
            "negative_pct": neg_pct,
            "positive_pct": pos_pct,
            "top_negative_tag": df[
                df["grouping_sentiment"] == "negative"
            ]["primary_tag_group"].value_counts().index[0]
            if (df["grouping_sentiment"] == "negative").any() else "",
        }])

        tag_counts = df.groupby(
            ["primary_tag_group", "grouping_sentiment"]
        ).size().unstack(fill_value=0).reset_index()

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            metrics.to_excel(writer, sheet_name="This week", index=False)
            tag_counts.to_excel(
                writer, sheet_name="By tag group", index=False
            )

    save_output(save_leadership, "leadership_output.xlsx")
    log.info("All outputs saved ✓")

# ══════════════════════════════════════════════════════════════════════════
# TEST MODE — bypass NLP steps
# ══════════════════════════════════════════════════════════════════════════
def _stub_nlp_columns(df):
    """
    Fill in placeholder values for every column that classify/ABSA/entity
    extraction would normally produce, so step_save(), rebuild_from_master(),
    and the dashboard don't break when SKIP_NLP=true. Lets you test
    ingest → preprocess → save → dashboard plumbing fast, without loading
    the SVM/ABSA/spaCy models. NEVER use this output for real analysis.
    """
    df = df.copy()

    # Would normally come from step_classify()
    df["primary_tag"]          = "Untagged (SKIP_NLP)"
    df["primary_tag_group"]    = "Untagged (SKIP_NLP)"
    df["secondary_tags"]       = ""
    df["secondary_tag_groups"] = ""
    df["svm_confidence"]       = 0.0
    df["prediction_method"]    = "skipped"

    # Would normally come from step_absa()
    df["absa_aspects"]           = ""
    df["absa_aspect_sentiments"] = ""
    df["absa_overall_sentiment"] = "neutral"
    df["grouping_sentiment"]     = "neutral"

    # Would normally come from step_entities()
    df["entities_councils"] = ""
    df["entities_features"] = ""
    df["entities_errors"]   = ""
    df["entities_fees"]     = ""

    return df


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info(f"PIPELINE START — {RUN_TS}")
    log.info("=" * 60)

    try:
        # ── Ingest raw data ────────────────────────────────────────────

        df_raw, ingest_source = step_ingest()
        has_extra_header_row = (ingest_source == "file")

        # Full set of respondent IDs fetched this run, BEFORE the feedback
        # dropna in preprocessing — this is what gets marked "processed",
        # regardless of whether a given respondent left free-text feedback.
        all_fetched_ids = df_raw[cfg.COL_RESPONDENT_ID].astype(str).copy()

        # NOTE: raw volume/rating/NSAT metrics are NOT written here anymore.
        # That's handled exclusively by update_raw_metrics.py on its own
        # daily schedule, with its own dedup list (processed_ids_raw.csv).
        # Writing them here too would double-count respondents, since
        # write_raw_metrics() adds to existing totals rather than
        # overwriting them.
        import aggregation_db as adb
        adb.initialise_db()

        # ── Preprocess ─────────────────────────────────────────────────
        df = step_preprocess_external(df_raw.copy(), has_extra_header_row=has_extra_header_row)

        # ── Guard: exit cleanly if no feedback responses ───────────────
        if len(df) == 0:
            log.warning("No feedback responses in this export — skipping NLP steps.")
            # Still mark these respondents as processed — they were rating
            # -only, no comment. Without this, this same batch reappears as
            # "new" on every future run and never gets resolved.
            n_marked = mark_ids_processed(all_fetched_ids)
            log.info(f"Marked {n_marked} rating-only (no-feedback) respondents as processed.")

            # IMPORTANT: still rebuild the DB from master.xlsx even though
            # this run had nothing new to classify. rebuild_from_master()
            # re-reads the ENTIRE master file every time (it's not
            # incremental), so it's always safe to call — and skipping it
            # here means any feedback rows appended by a PRIOR run (e.g.
            # one that crashed after step_save() but before this point)
            # never make it into metrics.db, silently freezing the
            # dashboard even though master.xlsx itself is up to date.
            adb.rebuild_from_master()
            adb.export_for_powerbi()

            # Only advance the checkpoint for API-sourced runs — a
            # manual SKIP_API run isn't governed by this checkpoint and
            # shouldn't move it forward (that would create a gap once
            # the pipeline goes back to fetching from the real API).
            if ingest_source == "api":
                import survey_monkey_fetch as smf
                smf.save_checkpoint(cfg.LAST_FETCH_CHECKPOINT, FETCH_CHECKPOINT_TS)

            pipeline_log["status"] = "skipped"
            return

        # ── NLP pipeline ───────────────────────────────────────────────
        if cfg.SKIP_NLP:
            log.warning("=" * 60)
            log.warning("SKIP_NLP is set — bypassing classify/ABSA/entities")
            log.warning("Tag/sentiment/entity columns are placeholders only.")
            log.warning("Do NOT use this run's output for real analysis.")
            log.warning("=" * 60)
            df = _stub_nlp_columns(df)
        else:
            df = step_classify(df)
            df = step_absa(df)
            df = step_entities(df)
        step_save(df)

        # Mark the FULL fetched set as processed — not just the
        # feedback-bearing rows in df — so rating-only respondents in this
        # same batch aren't left behind to be re-processed again next run.
        n_marked = mark_ids_processed(all_fetched_ids)
        log.info(f"Marked {n_marked} respondents as processed ({len(df)} with feedback, "
                 f"{n_marked - len(df)} rating-only).")

        # ── Write NLP metrics to DB ────────────────────────────────────
        # ── Rebuild DB from full master (groups by actual Start Date) ──
        adb.rebuild_from_master()
        adb.export_for_powerbi()

        # Same reasoning as the no-feedback branch above — only advance
        # the checkpoint for API-sourced runs, and only now that the
        # ENTIRE run (classify/ABSA/entities/save/mark-processed) has
        # actually succeeded. Advancing it any earlier (e.g. right after
        # ingest) would risk silently losing responses if a later step
        # crashed — mark_ids_processed wouldn't have run for them, but
        # the next fetch would already be past their creation date.
        if ingest_source == "api":
            import survey_monkey_fetch as smf
            smf.save_checkpoint(cfg.LAST_FETCH_CHECKPOINT, FETCH_CHECKPOINT_TS)

        # ── Regenerate dashboard from updated DB ───────────────────────
        import generate_dashboard as gd
        gd.generate()
        log.info("Dashboard regenerated ✓")

        pipeline_log["status"] = "success"
        pipeline_log["status"] = "success"
        log.info("=" * 60)
        log.info(f"PIPELINE COMPLETE — {pipeline_log['new_reviews']} new reviews processed")
        log.info(f"Master file now contains {pipeline_log['total_master']} reviews")
        log.info(f"Outputs: outputs/latest/ and outputs/archive/{RUN_DATE}/")
        log.info("=" * 60)

    except Exception as e:
        pipeline_log["status"] = "failed"
        pipeline_log["error"] = str(e)
        log.error(f"PIPELINE FAILED: {e}")
        log.error(traceback.format_exc())
        raise

    finally:
        append_pipeline_log()

if __name__ == "__main__":
    main()