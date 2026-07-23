# store.py
# ── Dashboard data-access layer ────────────────────────────────────────────
# Thin read-only wrappers around metrics.db (SQLite). The dashboard never
# touches raw survey data or runs any NLP — it only reads what
# run_pipeline.py / aggregation_db.py has already written.

import os
import sqlite3

import pandas as pd

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "metrics.db"
)


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


def load_latest_run_date() -> str | None:
    df = _read_sql("SELECT MAX(run_date) as latest FROM monthly_summary")
    if df.empty or pd.isna(df.iloc[0]["latest"]):
        return None
    return df.iloc[0]["latest"]


def load_monthly_summary() -> pd.DataFrame:
    """One row per month: volumes, ratings, NSAT, sentiment split."""
    return _read_sql("SELECT * FROM monthly_summary ORDER BY month")


def load_tag_group_trends() -> pd.DataFrame:
    """One row per (month, tag_group): volume + sentiment breakdown."""
    return _read_sql(
        "SELECT * FROM tag_group_monthly ORDER BY month, tag_group"
    )


def load_feature_trends() -> pd.DataFrame:
    """One row per (month, feature): mentions + sentiment breakdown."""
    return _read_sql(
        "SELECT * FROM feature_monthly ORDER BY month, feature"
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