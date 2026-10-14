"""
backfill_feedback_original.py
Fills review_detail.feedback_original for rows already in metrics.db, using the
original comments in data/survey_cache.csv. Touches ONLY that one column.

    python backfill_feedback_original.py            # dry run
    python backfill_feedback_original.py --apply    # writes to metrics.db
"""
import sys
import sqlite3

from aggregation_db import DB_PATH, CACHE_PATH, load_original_comments

APPLY = "--apply" in sys.argv


def main():
    comments = load_original_comments()
    if not comments:
        sys.exit(f"No original comments loaded from {CACHE_PATH}")

    # Scientific-notation IDs (e.g. 1.19E+11) mean the cache was saved through Excel
    sci = sum(("E+" in k.upper()) or ("." in k) for k in comments)
    print(f"cache comments loaded: {len(comments)} | IDs that look corrupted: {sci}")
    if sci:
        sys.exit("Some cache IDs look like scientific notation — matching would be unreliable.")

    conn = sqlite3.connect(DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(review_detail)")]
    if "feedback_original" not in cols:
        if APPLY:
            conn.execute("ALTER TABLE review_detail ADD COLUMN feedback_original TEXT")
        print("review_detail has no feedback_original column yet"
              + (" (added)" if APPLY else " (would be added)"))

    db_ids = [r[0] for r in conn.execute("SELECT respondent_id FROM review_detail")]
    matched = [(comments[i], i) for i in db_ids if i in comments]
    print(f"review_detail rows: {len(db_ids)} | matched: {len(matched)} | "
          f"no original comment: {len(db_ids) - len(matched)}")

    if APPLY:
        conn.executemany("UPDATE review_detail SET feedback_original = ? WHERE respondent_id = ?", matched)
        conn.commit()
        print("Done.")
    else:
        print("Dry run only. Re-run with --apply to write.")
    conn.close()


if __name__ == "__main__":
    main()