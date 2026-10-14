# store.py
# ── Dashboard data-access layer ────────────────────────────────────────────
# Thin read-only wrappers around metrics.db (SQLite). The dashboard never
# touches raw survey data or runs any NLP — it only reads what
# run_pipeline.py / aggregation_db.py has already written.

# Needed for `str | None` / `list[str]` type hints below to work on Python
# < 3.10 (this file is imported both by run_pipeline.py's venv and the
# dashboard's — they're not guaranteed to be the same Python version).
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "metrics.db"
)

# ── Add to store.py ─────────────────────────────────────────────────────

def _ensure_visitor_log_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visitor_log (
            visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            visit_timestamp TEXT NOT NULL
        )
    """)
    conn.commit()


def log_visit():
    """
    Records one visit, timestamped in UTC. Called once per browser
    session (see app.py) — never on every rerun/interaction within
    that session.
    """
    conn = _connect()
    if conn is None:
        return
    try:
        _ensure_visitor_log_table(conn)
        conn.execute(
            "INSERT INTO visitor_log (visit_timestamp) VALUES (?)",
            (datetime.now(timezone.utc).isoformat(),)
        )
        conn.commit()
    except Exception:
        pass  # never let logging break the dashboard itself
    finally:
        conn.close()


def load_visitor_summary() -> dict:
    """Total visits, visits in the last 7/30 days, and the most recent visit."""
    df = _read_sql("SELECT visit_timestamp FROM visitor_log ORDER BY visit_timestamp DESC")
    if df.empty:
        return {"total": 0, "last_7_days": 0, "last_30_days": 0, "last_visit": None}

    df["visit_timestamp"] = pd.to_datetime(df["visit_timestamp"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    return {
        "total": len(df),
        "last_7_days": int((df["visit_timestamp"] >= now - pd.Timedelta(days=7)).sum()),
        "last_30_days": int((df["visit_timestamp"] >= now - pd.Timedelta(days=30)).sum()),
        "last_visit": df["visit_timestamp"].iloc[0],
    }


def load_visits_by_day() -> pd.DataFrame:
    """One row per day with a visit, for a simple trend chart."""
    df = _read_sql("SELECT visit_timestamp FROM visitor_log")
    if df.empty:
        return pd.DataFrame(columns=["date", "visits"])
    df["date"] = pd.to_datetime(df["visit_timestamp"], utc=True).dt.date
    return df.groupby("date").size().reset_index(name="visits").sort_values("date")


def _connect():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)


def _read_sql(query, params=None):
    conn = _connect()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql(query, conn, params=params)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def db_exists() -> bool:
    return os.path.exists(DB_PATH)


def load_available_months() -> list[str]:
    """All months with data, sorted chronologically (e.g. '2026-06')."""
    df = _read_sql("SELECT DISTINCT month FROM monthly_summary ORDER BY month")
    return df["month"].dropna().tolist() if not df.empty else []

DB_PATH = Path(__file__).resolve().parent / "data" / "metrics.db"
CHECKPOINT_PATH = DB_PATH.parent / "last_fetch_checkpoint_cache.txt"

def load_latest_run_date() -> str | None:
    """When SurveyMonkey data was last pulled (UTC), not the newest month."""
    try:
        ts = pd.to_datetime(CHECKPOINT_PATH.read_text().strip(), utc=True)
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return None


def load_monthly_summary() -> pd.DataFrame:
    """One row per month: volumes, ratings, NSAT, sentiment split."""
    return _read_sql("SELECT * FROM monthly_summary ORDER BY month")


def load_topic_trends() -> pd.DataFrame:
    """One row per (month, topic): volume + sentiment breakdown.
    Replaces load_tag_group_trends() and load_feature_trends() — topics
    already fold the old tag_group *and* the near-duplicate ABSA aspects
    ("document upload" / "file upload" / "uploading documents") into one
    number per topic, and never include "User type" (that's its own
    field now, not written into topic_monthly at all).

    Confirmed against aggregation_db.py: topic_monthly columns are
    run_date, month, topic, total_mentions, positive_count, negative_count,
    neutral_count, mixed_count, avg_sentiment_score — total_mentions
    (not total_reviews, unlike tag_group_monthly) and an extra mixed_count
    (a comment whose mentions of this topic were both positive and
    negative). SELECT * picks up all of these; app.py reads total_mentions
    explicitly, mixed_count is available if you want to surface it later.
    """
    return _read_sql(
        "SELECT * FROM topic_monthly ORDER BY month, topic"
    )


def load_error_trends() -> pd.DataFrame:
    """One row per (month, error_pattern): occurrence count."""
    return _read_sql(
        "SELECT * FROM error_monthly ORDER BY month, count DESC"
    )


def load_review_detail(months: list[str] | None = None) -> pd.DataFrame:
    """
    Individual tagged reviews, optionally filtered to a list of months.

    months=None            -> all months (no filter)
    months=[] (empty list)  -> explicitly zero months selected -> empty result
    months=[...]            -> only those months

    (Previously an empty list was treated the same as None, which is the
    bug behind "select months isn't working" — deselecting everything
    silently fell back to showing all-time data instead of nothing.)
    """
    if months is not None and len(months) == 0:
        return pd.DataFrame()
    if months:
        placeholders = ",".join(["?"] * len(months))
        query = f"SELECT * FROM review_detail WHERE month IN ({placeholders}) ORDER BY run_date DESC"
        return _read_sql(query, params=months)
    return _read_sql("SELECT * FROM review_detail ORDER BY run_date DESC")


def filter_by_months(df: pd.DataFrame, months: list[str] | None) -> pd.DataFrame:
    """
    Helper: filter any month-keyed table down to a list of selected months.

    months=None            -> return df unfiltered (all-time)
    months=[] (empty list)  -> return an empty frame (explicitly nothing selected)
    months=[...]            -> filter to those months
    """
    if months is None:
        return df
    if "month" not in df.columns:
        return df
    if len(months) == 0:
        return df.iloc[0:0].copy()
    return df[df["month"].isin(months)].copy()