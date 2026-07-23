# aggregation_db.py
# ── Builds and updates the SQLite aggregation database ────────────────────
# Called automatically by run_pipeline.py after each pipeline run.
# Can also be run standalone: python aggregation_db.py

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

import pipeline_config as cfg

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "metrics.db"
)

# ══════════════════════════════════════════════════════════════════════════
# SCHEMA — monthly granularity
# ══════════════════════════════════════════════════════════════════════════
SCHEMA = """
CREATE TABLE IF NOT EXISTS monthly_summary (
    run_date            TEXT,
    month               TEXT,
    total_respondents   INTEGER,
    total_with_feedback INTEGER,
    avg_rating          REAL,
    rating_5            INTEGER,
    rating_4            INTEGER,
    rating_3            INTEGER,
    rating_2            INTEGER,
    rating_1            INTEGER,
    nsat                REAL,
    positive_count      INTEGER,
    negative_count      INTEGER,
    neutral_count       INTEGER,
    positive_pct        REAL,
    negative_pct        REAL,
    PRIMARY KEY (run_date)
);

CREATE TABLE IF NOT EXISTS tag_group_monthly (
    run_date            TEXT,
    month               TEXT,
    tag_group           TEXT,
    total_reviews       INTEGER,
    positive_count      INTEGER,
    negative_count      INTEGER,
    neutral_count       INTEGER,
    avg_sentiment_score REAL,
    PRIMARY KEY (run_date, tag_group)
);

CREATE TABLE IF NOT EXISTS feature_monthly (
    run_date            TEXT,
    month               TEXT,
    feature             TEXT,
    total_mentions      INTEGER,
    negative_mentions   INTEGER,
    positive_mentions   INTEGER,
    negative_pct        REAL,
    PRIMARY KEY (run_date, feature)
);

CREATE TABLE IF NOT EXISTS error_monthly (
    run_date            TEXT,
    month               TEXT,
    error_pattern       TEXT,
    count               INTEGER,
    PRIMARY KEY (run_date, error_pattern)
);

CREATE TABLE IF NOT EXISTS review_detail (
    respondent_id       TEXT,
    run_date            TEXT,
    month               TEXT,
    feedback_clean      TEXT,
    primary_tag         TEXT,
    primary_tag_group   TEXT,
    secondary_tags      TEXT,
    svm_confidence      REAL,
    prediction_method   TEXT,
    grouping_sentiment  TEXT,
    absa_aspects        TEXT,
    entities_features   TEXT,
    entities_errors     TEXT,
    entities_fees       TEXT,
    rating              REAL,
    PRIMARY KEY (respondent_id)
);
"""

# ══════════════════════════════════════════════════════════════════════════
# CONNECTION
# ══════════════════════════════════════════════════════════════════════════
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def initialise_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print("DB initialised ✓")

def get_month(run_date):
    """Convert run_date string to YYYY-MM month label."""
    return pd.to_datetime(run_date).strftime("%Y-%m")

def sentiment_to_score(s):
    return {"positive": 1, "neutral": 0, "negative": -1}.get(str(s).lower(), 0)

def count_entity_col(series):
    counts = defaultdict(int)
    for cell in series:
        if pd.notna(cell) and str(cell).strip():
            for item in str(cell).split(", "):
                item = item.strip()
                if item:
                    counts[item] += 1
    return counts

# ══════════════════════════════════════════════════════════════════════════
# WRITE RAW METRICS (from full export before preprocessing)
# ══════════════════════════════════════════════════════════════════════════
def _write_raw_metrics_single_month(df_raw, run_date):
    """Write raw volume/rating metrics for a single month's slice of the raw export."""
    month = get_month(run_date)

    # ── Rating ─────────────────────────────────────────────────────────────
    rating_col = df_raw["Rating"].copy()

    # First map text values, then coerce remaining to numeric
    rating_map = {"Excellent": 5, "Poor": 1}
    ratings = rating_col.map(rating_map)  # maps Excellent/Poor
    numeric_mask = ratings.isna() & rating_col.notna()
    ratings[numeric_mask] = pd.to_numeric(  # fills in 2/3/4
        rating_col[numeric_mask], errors="coerce"
    )
    ratings = ratings.astype(float)

    total_respondents = len(df_raw)
    total_rated = int(ratings.notna().sum())
    avg_rating = round(float(ratings.mean()), 2) if total_rated > 0 else None

    # ── Feedback text count ────────────────────────────────────────────────
    # detect whichever feedback column is present
    feedback_col_options = [
        "Is there anything you would like to tell us about your experience?",
        "Feedback", "Feedback_x"
    ]
    feedback_col = next((c for c in feedback_col_options if c in df_raw.columns), None)

    if feedback_col:
        non_blank = df_raw[feedback_col].notna() & (df_raw[feedback_col].astype(str).str.strip() != "")
        total_with_feedback = int(non_blank.sum())
    else:
        total_with_feedback = 0

    conn = get_connection()

    # Ensure raw columns exist in monthly_summary
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monthly_summary (
            run_date            TEXT,
            month               TEXT,
            total_respondents   INTEGER,
            total_with_feedback INTEGER,
            avg_rating          REAL,
            rating_5            INTEGER,
            rating_4            INTEGER,
            rating_3            INTEGER,
            rating_2            INTEGER,
            rating_1            INTEGER,
            nsat                REAL,
            positive_count      INTEGER,
            negative_count      INTEGER,
            neutral_count       INTEGER,
            positive_pct        REAL,
            negative_pct        REAL,
            PRIMARY KEY (run_date)
        )
    """)

    # Migration safety: add nsat column if this DB pre-dates it
    cols = [r[1] for r in conn.execute("PRAGMA table_info(monthly_summary)").fetchall()]
    if "nsat" not in cols:
        conn.execute("ALTER TABLE monthly_summary ADD COLUMN nsat REAL")

    # Upsert — adds to existing raw totals for this month rather than
    # overwriting, so multiple exports landing in the same month accumulate
    # correctly. NLP columns are left untouched (NULL until
    # write_monthly_metrics runs).
    conn.execute("""
        INSERT INTO monthly_summary
            (run_date, month, total_respondents, total_with_feedback,
             avg_rating, rating_5, rating_4, rating_3, rating_2, rating_1, nsat)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_date) DO UPDATE SET
            total_respondents   = monthly_summary.total_respondents   + excluded.total_respondents,
            total_with_feedback = monthly_summary.total_with_feedback + excluded.total_with_feedback,
            rating_5            = monthly_summary.rating_5 + excluded.rating_5,
            rating_4            = monthly_summary.rating_4 + excluded.rating_4,
            rating_3            = monthly_summary.rating_3 + excluded.rating_3,
            rating_2            = monthly_summary.rating_2 + excluded.rating_2,
            rating_1            = monthly_summary.rating_1 + excluded.rating_1,
            avg_rating          = CASE
                WHEN (monthly_summary.rating_5 + monthly_summary.rating_4 + monthly_summary.rating_3
                      + monthly_summary.rating_2 + monthly_summary.rating_1
                      + excluded.rating_5 + excluded.rating_4 + excluded.rating_3
                      + excluded.rating_2 + excluded.rating_1) > 0
                THEN ROUND(
                    CAST(
                        (monthly_summary.rating_5 + excluded.rating_5) * 5
                      + (monthly_summary.rating_4 + excluded.rating_4) * 4
                      + (monthly_summary.rating_3 + excluded.rating_3) * 3
                      + (monthly_summary.rating_2 + excluded.rating_2) * 2
                      + (monthly_summary.rating_1 + excluded.rating_1) * 1
                    AS REAL) /
                    (monthly_summary.rating_5 + monthly_summary.rating_4 + monthly_summary.rating_3
                     + monthly_summary.rating_2 + monthly_summary.rating_1
                     + excluded.rating_5 + excluded.rating_4 + excluded.rating_3
                     + excluded.rating_2 + excluded.rating_1)
                , 2)
                ELSE NULL
            END,
            nsat                = CASE
                WHEN (monthly_summary.rating_5 + monthly_summary.rating_4 + monthly_summary.rating_3
                      + monthly_summary.rating_2 + monthly_summary.rating_1
                      + excluded.rating_5 + excluded.rating_4 + excluded.rating_3
                      + excluded.rating_2 + excluded.rating_1) > 0
                THEN ROUND(
                    CAST(
                        (monthly_summary.rating_5 + excluded.rating_5 + monthly_summary.rating_4 + excluded.rating_4)
                      - (monthly_summary.rating_1 + excluded.rating_1 + monthly_summary.rating_2 + excluded.rating_2)
                    AS REAL) /
                    (monthly_summary.rating_5 + monthly_summary.rating_4 + monthly_summary.rating_3
                     + monthly_summary.rating_2 + monthly_summary.rating_1
                     + excluded.rating_5 + excluded.rating_4 + excluded.rating_3
                     + excluded.rating_2 + excluded.rating_1)
                    * 100
                , 1)
                ELSE NULL
            END
    """, (
        run_date, month,
        total_respondents, total_with_feedback,
        avg_rating,
        int((ratings == 5).sum()),
        int((ratings == 4).sum()),
        int((ratings == 3).sum()),
        int((ratings == 2).sum()),
        int((ratings == 1).sum()),
        # initial nsat for a brand-new row (no existing counts to add to)
        # NSAT = (% 4-5 star) - (% 1-2 star)
        round(
            ((ratings >= 4).sum() - (ratings <= 2).sum()) / total_rated * 100, 1
        ) if total_rated > 0 else None,
    ))

    conn.commit()
    conn.close()
    print(f"Raw metrics written for {month} — {total_respondents} respondents, "
          f"avg rating: {avg_rating}, {total_with_feedback} with feedback")


def write_raw_metrics(df_raw, run_date=None):
    """
    Write volume and rating metrics from the raw SurveyMonkey export,
    BEFORE preprocessing drops any rows (so rating-only respondents are
    counted in total_respondents and avg_rating).

    The export is grouped by its actual "Start Date" so that totals land
    in the correct calendar month, regardless of when the pipeline is run.
    Falls back to `run_date` (or today) if "Start Date" is missing/unparseable.
    """
    date_col = cfg.COL_DATE if cfg.COL_DATE in df_raw.columns else "Start Date"

    if date_col in df_raw.columns:
        # utc=True handles a mix of tz-aware dates (API responses come back
        # as ISO 8601 with a UTC offset, e.g. "...+00:00") and tz-naive dates
        # (older rows already in master.xlsx) in the same column — without
        # it, pandas raises "Mixed timezones detected" instead of parsing.
        # We only need the calendar date for monthly grouping, so drop the
        # tz info immediately after parsing to keep everything downstream
        # (string formatting, comparisons) working with plain naive dates
        # exactly as before.
        dates = pd.to_datetime(
            df_raw[date_col], errors="coerce", utc=True, format="mixed"
        ).dt.tz_localize(None)
        # NOTE: format="mixed" is required, not optional — with utc=True
        # alone, pandas infers ONE date format from the first value in the
        # column and silently coerces every row that doesn't match that
        # exact format to NaT. Since this column mixes tz-aware API dates
        # ("...+00:00") with tz-naive older dates, that would have quietly
        # dropped real historic rows as "unparseable" rather than raising
        # an error — worse than the crash it was meant to fix.
        if dates.notna().any():
            df_tmp = df_raw.copy()
            df_tmp["_month_key"] = dates.dt.strftime("%Y-%m-01")
            # rows with unparseable dates fall back to run_date/today
            fallback = run_date or datetime.now().strftime("%Y-%m-%d")
            df_tmp["_month_key"] = df_tmp["_month_key"].fillna(get_month(fallback) + "-01")

            for month_key, batch in df_tmp.groupby("_month_key"):
                _write_raw_metrics_single_month(batch.drop(columns=["_month_key"]), month_key)
            return

    # No usable date column — treat the whole export as one run_date
    if run_date is None:
        run_date = datetime.now().strftime("%Y-%m-%d")
    _write_raw_metrics_single_month(df_raw, run_date)

# ══════════════════════════════════════════════════════════════════════════
# WRITE NLP METRICS (from processed pipeline output)
# ══════════════════════════════════════════════════════════════════════════
def write_monthly_metrics(df, run_date=None):
    """Write NLP-derived metrics for a pipeline run."""
    if run_date is None:
        run_date = datetime.now().strftime("%Y-%m-%d")

    month = get_month(run_date)
    conn  = get_connection()

    # Migration safety: add nsat column if this DB pre-dates it
    cols = [r[1] for r in conn.execute("PRAGMA table_info(monthly_summary)").fetchall()]
    if "nsat" not in cols:
        conn.execute("ALTER TABLE monthly_summary ADD COLUMN nsat REAL")

    # ── 1. Update monthly_summary with NLP sentiment ──────────────────────
    total = len(df)
    pos   = int((df["grouping_sentiment"] == "positive").sum())
    neg   = int((df["grouping_sentiment"] == "negative").sum())
    neu   = int((df["grouping_sentiment"] == "neutral").sum())

    conn.execute("""
        INSERT INTO monthly_summary
            (run_date, month, total_respondents, total_with_feedback,
             avg_rating, rating_5, rating_4, rating_3, rating_2, rating_1, nsat,
             positive_count, negative_count, neutral_count, positive_pct, negative_pct)
        VALUES (?, ?, COALESCE((SELECT total_respondents FROM monthly_summary WHERE run_date=?), 0),
                      COALESCE((SELECT total_with_feedback FROM monthly_summary WHERE run_date=?), 0),
                      (SELECT avg_rating FROM monthly_summary WHERE run_date=?),
                      COALESCE((SELECT rating_5 FROM monthly_summary WHERE run_date=?), 0),
                      COALESCE((SELECT rating_4 FROM monthly_summary WHERE run_date=?), 0),
                      COALESCE((SELECT rating_3 FROM monthly_summary WHERE run_date=?), 0),
                      COALESCE((SELECT rating_2 FROM monthly_summary WHERE run_date=?), 0),
                      COALESCE((SELECT rating_1 FROM monthly_summary WHERE run_date=?), 0),
                      (SELECT nsat FROM monthly_summary WHERE run_date=?),
                      ?, ?, ?, ?, ?)
        ON CONFLICT(run_date) DO UPDATE SET
            positive_count = excluded.positive_count,
            negative_count = excluded.negative_count,
            neutral_count  = excluded.neutral_count,
            positive_pct   = excluded.positive_pct,
            negative_pct   = excluded.negative_pct
    """, (
        run_date, month,
        run_date, run_date, run_date, run_date, run_date, run_date, run_date, run_date, run_date,
        pos, neg, neu,
        round(pos / total * 100, 1) if total > 0 else 0,
        round(neg / total * 100, 1) if total > 0 else 0,
    ))

    # ── 2. Tag group monthly ──────────────────────────────────────────────
    df["sentiment_score"] = df["grouping_sentiment"].apply(sentiment_to_score)

    for tag_group, grp in df.groupby("primary_tag_group"):
        g_total = len(grp)
        g_pos   = int((grp["grouping_sentiment"] == "positive").sum())
        g_neg   = int((grp["grouping_sentiment"] == "negative").sum())
        g_neu   = int((grp["grouping_sentiment"] == "neutral").sum())
        g_score = round(float(grp["sentiment_score"].mean()), 3)

        conn.execute("""
            INSERT OR REPLACE INTO tag_group_monthly
            (run_date, month, tag_group, total_reviews,
             positive_count, negative_count, neutral_count, avg_sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_date, month, str(tag_group).strip(),
            g_total, g_pos, g_neg, g_neu, g_score
        ))

    # ── 3. Feature monthly ────────────────────────────────────────────────
    feat_sentiment = defaultdict(lambda: {"pos": 0, "neg": 0, "neu": 0})
    for _, row in df.iterrows():
        sentiment = str(row.get("grouping_sentiment", "neutral"))
        for f in str(row.get("entities_features", "")).split(", "):
            f = f.strip()
            if f and f != "Planning Portal":
                if sentiment == "positive":   feat_sentiment[f]["pos"] += 1
                elif sentiment == "negative": feat_sentiment[f]["neg"] += 1
                else:                         feat_sentiment[f]["neu"] += 1

    for feature, counts in feat_sentiment.items():
        total_f = counts["pos"] + counts["neg"] + counts["neu"]
        if total_f < 2:
            continue
        conn.execute("""
            INSERT OR REPLACE INTO feature_monthly
            (run_date, month, feature, total_mentions,
             negative_mentions, positive_mentions, negative_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_date, month, feature, total_f,
            counts["neg"], counts["pos"],
            round(counts["neg"] / total_f * 100, 1) if total_f > 0 else 0
        ))

    # ── 4. Error monthly ──────────────────────────────────────────────────
    if "entities_errors" in df.columns:
        error_counts = count_entity_col(df["entities_errors"])
        for pattern, count in error_counts.items():
            conn.execute("""
                INSERT OR REPLACE INTO error_monthly
                (run_date, month, error_pattern, count)
                VALUES (?, ?, ?, ?)
            """, (run_date, month, pattern, int(count)))

    # ── 5. Review detail ──────────────────────────────────────────────────
    for _, row in df.iterrows():
        respondent_id = str(row.get(cfg.COL_RESPONDENT_ID, ""))
        if not respondent_id or respondent_id == "nan":
            continue
        conn.execute("""
            INSERT OR REPLACE INTO review_detail
            (respondent_id, run_date, month, feedback_clean,
             primary_tag, primary_tag_group, secondary_tags,
             svm_confidence, prediction_method, grouping_sentiment,
             absa_aspects, entities_features, entities_errors,
             entities_fees, rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            respondent_id, run_date, month,
            str(row.get("Feedback_clean", "")),
            str(row.get("primary_tag", "")),
            str(row.get("primary_tag_group", "")),
            str(row.get("secondary_tags", "")),
            float(row.get("svm_confidence", 0) or 0),
            str(row.get("prediction_method", "")),
            str(row.get("grouping_sentiment", "neutral")),
            str(row.get("absa_aspects", "")),
            str(row.get("entities_features", "")),
            str(row.get("entities_errors", "")),
            str(row.get("entities_fees", "")),
            float(row.get("Rating")) if pd.notna(row.get("Rating")) else None
        ))

    conn.commit()
    conn.close()
    print(f"Monthly metrics written — {total} NLP reviews for {run_date} ({month})")

# ══════════════════════════════════════════════════════════════════════════
# RESET — fresh start
# ══════════════════════════════════════════════════════════════════════════
def reset_pipeline(confirm=False):
    """
    Full reset — clears DB, master file, processed IDs, outputs and logs.
    Pass confirm=True to actually execute (safety guard).
    """
    if not confirm:
        print("Safety check: call reset_pipeline(confirm=True) to proceed.")
        print("This will delete:")
        print(f"  - {DB_PATH}")
        print(f"  - {cfg.MASTER_FILE}")
        print(f"  - {cfg.PROCESSED_IDS}")
        print(f"  - All files in outputs/latest/")
        print(f"  - All files in logs/")
        return

    deleted = []

    # DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        deleted.append(DB_PATH)

    # Master file
    if os.path.exists(cfg.MASTER_FILE):
        os.remove(cfg.MASTER_FILE)
        deleted.append(cfg.MASTER_FILE)

    # Processed IDs
    if os.path.exists(cfg.PROCESSED_IDS):
        os.remove(cfg.PROCESSED_IDS)
        deleted.append(cfg.PROCESSED_IDS)

    # Outputs/latest
    latest_dir = cfg.OUTPUTS_LATEST
    if os.path.exists(latest_dir):
        for f in os.listdir(latest_dir):
            fp = os.path.join(latest_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
                deleted.append(fp)

    # Logs
    logs_dir = cfg.LOGS_DIR
    if os.path.exists(logs_dir):
        for f in os.listdir(logs_dir):
            fp = os.path.join(logs_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
                deleted.append(fp)

    # Reinitialise clean DB
    initialise_db()

    print(f"Reset complete — {len(deleted)} files deleted")
    print("Fresh DB created ✓")
    print("\nDeleted:")
    for f in deleted:
        print(f"  - {f}")

# ══════════════════════════════════════════════════════════════════════════
# RECOMPUTE NSAT — for existing rows, using current formula
# ══════════════════════════════════════════════════════════════════════════
def recompute_nsat():
    """
    Recalculate nsat for all existing monthly_summary rows using the
    current formula: NSAT = (% 4-5 star) - (% 1-2 star).
    Run this once after changing the NSAT definition so historical
    months reflect the new formula without needing a full rebuild.
    """
    conn = get_connection()

    cols = [r[1] for r in conn.execute("PRAGMA table_info(monthly_summary)").fetchall()]
    if "nsat" not in cols:
        conn.execute("ALTER TABLE monthly_summary ADD COLUMN nsat REAL")

    conn.execute("""
        UPDATE monthly_summary
        SET nsat = CASE
            WHEN (rating_5 + rating_4 + rating_3 + rating_2 + rating_1) > 0
            THEN ROUND(
                CAST((rating_5 + rating_4) - (rating_1 + rating_2) AS REAL)
                / (rating_5 + rating_4 + rating_3 + rating_2 + rating_1)
                * 100
            , 1)
            ELSE NULL
        END
    """)

    conn.commit()
    df = pd.read_sql("SELECT run_date, month, nsat FROM monthly_summary ORDER BY run_date", conn)
    conn.close()

    print("NSAT recomputed for all months:")
    print(df.to_string())


# ══════════════════════════════════════════════════════════════════════════
# QUERY HELPERS
# ══════════════════════════════════════════════════════════════════════════
def get_monthly_summary():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM monthly_summary ORDER BY run_date", conn)
    conn.close()
    return df

def get_tag_group_trends():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM tag_group_monthly ORDER BY run_date, tag_group", conn
    )
    conn.close()
    return df

def get_feature_trends():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM feature_monthly ORDER BY run_date, feature", conn
    )
    conn.close()
    return df

def get_error_trends():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM error_monthly ORDER BY run_date, count DESC", conn
    )
    conn.close()
    return df

def get_latest_run_date():
    conn = get_connection()
    result = conn.execute(
        "SELECT MAX(run_date) FROM monthly_summary"
    ).fetchone()
    conn.close()
    return result[0] if result else None

def export_for_powerbi():
    """Export all DB tables to Excel for Power BI consumption."""
    output_path = os.path.join(cfg.OUTPUTS_LATEST, "powerbi_data.xlsx")
    conn = get_connection()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.read_sql(
            "SELECT * FROM monthly_summary ORDER BY run_date", conn
        ).to_excel(writer, sheet_name="Monthly Summary", index=False)
        pd.read_sql(
            "SELECT * FROM tag_group_monthly ORDER BY run_date", conn
        ).to_excel(writer, sheet_name="Tag Group Monthly", index=False)
        pd.read_sql(
            "SELECT * FROM feature_monthly ORDER BY run_date", conn
        ).to_excel(writer, sheet_name="Feature Monthly", index=False)
        pd.read_sql(
            "SELECT * FROM error_monthly ORDER BY run_date", conn
        ).to_excel(writer, sheet_name="Error Monthly", index=False)
        pd.read_sql(
            "SELECT * FROM review_detail ORDER BY run_date", conn
        ).to_excel(writer, sheet_name="Review Detail", index=False)
    conn.close()
    print(f"Power BI data exported: {output_path}")
    return output_path


def rebuild_from_master():
    """One-off: repopulate metrics.db from data/master.xlsx"""
    master_path = cfg.MASTER_FILE

    if not os.path.exists(master_path):
        print(f"Master file not found at {master_path}")
        return

    df = pd.read_excel(master_path, dtype={cfg.COL_RESPONDENT_ID: str})
    print(f"Loaded {len(df)} reviews from master")
    print(f"Columns: {df.columns.tolist()}")

    # Defensive dedup: if the same respondent ever got appended to master
    # more than once (e.g. from re-running the pipeline during testing
    # with processed_ids.csv reset), every duplicate would otherwise be
    # double-counted in monthly sentiment/tag/feature aggregates. Keep the
    # last occurrence (most recently processed).
    before = len(df)
    df = df.drop_duplicates(subset=[cfg.COL_RESPONDENT_ID], keep="last")
    if len(df) < before:
        print(f"Dropped {before - len(df)} duplicate respondent rows before rebuild")

    if "Start Date" not in df.columns:
        print("No 'Start Date' column found — cannot group by month")
        return

    # Same mixed tz-aware/tz-naive issue as write_raw_metrics() above —
    # master.xlsx now accumulates both API-sourced rows (tz-aware ISO
    # dates) and older rows (tz-naive). utc=True + tz_localize(None)
    # normalizes everything to plain naive dates before grouping by month.
    # format="mixed" is required here too — utc=True alone infers a single
    # date format from the first row and silently NaTs every row that
    # doesn't match it exactly, which would masquerade as "unparseable
    # dates" below and quietly drop real historic rows from the rebuild.
    df["Start Date"] = pd.to_datetime(
        df["Start Date"], errors="coerce", utc=True, format="mixed"
    ).dt.tz_localize(None)
    dropped = df["Start Date"].isna().sum()
    if dropped:
        print(f"Dropping {dropped} rows with unparseable dates")
    df = df.dropna(subset=["Start Date"])

    df["_month_key"] = df["Start Date"].dt.strftime("%Y-%m-01")

    groups = df.groupby("_month_key")
    print(f"Found {len(groups)} months: {sorted(df['_month_key'].unique())}")

    for month_key, batch in groups:
        batch = batch.copy()
        print(f"\n  Month: {month_key} — {len(batch)} reviews")

        # NOTE: raw totals (total_respondents, avg_rating, rating_1..5)
        # are written by write_raw_metrics() directly from the true raw
        # export in run_pipeline.py — NOT from master, which only contains
        # rows that had written feedback (rating-only respondents were
        # dropped before master was built). Do not call write_raw_metrics
        # here, or it will overwrite the correct totals with the
        # feedback-only subset.

        # ── Write NLP metrics ─────────────────────────────────────────────
        # Only call if NLP columns exist (they won't if pipeline never ran)
        if "grouping_sentiment" in batch.columns:
            write_monthly_metrics(batch, run_date=month_key)
        else:
            print(f"  No NLP columns found for {month_key} — skipping sentiment/tags")

    print("\nRebuild complete ✓")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        initialise_db()
        rebuild_from_master()
    elif len(sys.argv) > 1 and sys.argv[1] == "recompute-nsat":
        recompute_nsat()