"""
eval_tagged_examples.py
Runs your hand-tagged comments through the pipeline's real classification step
and compares the topics it assigns with the topics you tagged.

    python eval_tagged_examples.py                       # uses tagged_examples.csv
    python eval_tagged_examples.py path/to/examples.csv

Reads nothing from the API and writes nothing to metrics.db / master.xlsx.
Detail file: outputs/eval_tagged_examples.csv (contains comments, keep out of git).
"""
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

import pipeline_config as cfg
import run_pipeline as rp

EXAMPLES = sys.argv[1] if len(sys.argv) > 1 else "tagged_examples.csv"
NOT_TOPICS = {"Overall Positive Experience", "Negative experience", "User type"}  # sentiment / user type, never topics
OUT_FILE = os.path.join("outputs", "eval_tagged_examples.csv")


def _clean(s):
    return str(s).strip()


def load_gold(path):
    ex = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    ex["Text"] = ex["Text"].map(_clean)
    ex["Tag"] = ex["Tag"].map(_clean)
    ex["Group"] = ex["Tag - Group"].map(_clean)
    ex = ex[ex["Text"] != ""]
    gold = ex.groupby("Text").agg(
        gold_groups=("Group", lambda s: set(s)),
        gold_tags=("Tag", lambda s: {x.lower() for x in s}),
    ).reset_index()
    return gold


def run_pipeline_on(gold):
    """Push the comments through the pipeline's own preprocess + classify steps."""
    raw = pd.DataFrame({
        cfg.COL_RESPONDENT_ID: [str(900000000000 + i) for i in range(len(gold))],
        cfg.COL_DATE: "2026-01-15",
        cfg.COL_RATING: 3,
        cfg.COL_FEEDBACK: gold["Text"].values,
    })
    try:
        pre = rp.step_preprocess(raw.copy())
        print(f"step_preprocess kept {len(pre)} of {len(raw)} comments")
    except Exception as e:                                   # keep going with a simple clean
        print(f"!! step_preprocess failed ({type(e).__name__}: {e}); using a simple lowercase clean instead")
        pre = raw.copy()
        pre[cfg.COL_FEEDBACK_CLEAN] = pre[cfg.COL_FEEDBACK].str.lower()
    pre[cfg.COL_RESPONDENT_ID] = pre[cfg.COL_RESPONDENT_ID].astype(str)
    out = rp.step_classify(pre.copy())
    out[cfg.COL_RESPONDENT_ID] = out[cfg.COL_RESPONDENT_ID].astype(str)
    gold = gold.assign(**{cfg.COL_RESPONDENT_ID: raw[cfg.COL_RESPONDENT_ID].values})
    return gold.merge(out, on=cfg.COL_RESPONDENT_ID, how="inner")


def main():
    gold = load_gold(EXAMPLES)
    print(f"{len(gold)} unique tagged comments loaded from {EXAMPLES}")
    tags = pd.read_csv(cfg.TAGS_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    known_groups = {g.strip() for g in tags["Group"] if g.strip()}

    df = run_pipeline_on(gold)
    needed = ["primary_tag", "primary_tag_group", "secondary_tag_groups", "svm_confidence", "prediction_method"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"step_classify output is missing {missing}. Columns present: {df.columns.tolist()}")
    print(f"{len(df)} comments scored (the rest were dropped by preprocessing)\n")

    def pred_groups(row):
        groups = {_clean(row["primary_tag_group"])}
        sec = row["secondary_tag_groups"]
        if isinstance(sec, str) and sec.strip():
            groups |= {g.strip() for g in rp.split_group_names(sec, known_groups)}
        return {g for g in groups if g and g.lower() != "nan"}

    df["pred_groups"] = df.apply(pred_groups, axis=1)
    df["gold_topics"] = df["gold_groups"].map(lambda s: s - NOT_TOPICS)
    df["pred_topics"] = df["pred_groups"].map(lambda s: s - NOT_TOPICS)
    df["conf"] = pd.to_numeric(df["svm_confidence"], errors="coerce")

    # ---- 1. Tag / group agreement (like your QA round) ---------------------
    tag_hit = df.apply(lambda r: _clean(r["primary_tag"]).lower() in r["gold_tags"], axis=1)
    grp_hit = df.apply(lambda r: _clean(r["primary_tag_group"]) in r["gold_groups"], axis=1)
    print("AGREEMENT WITH YOUR TAGS")
    print(f"  Primary tag is one of your tags:        {tag_hit.mean():.1%}")
    print(f"  Primary group is one of your groups:    {grp_hit.mean():.1%}")

    # ---- 2. Topics only (what the dashboard shows) -------------------------
    has_topic = df["gold_topics"].map(len) > 0
    sub = df[has_topic]
    overlap = sub.apply(lambda r: len(r["gold_topics"] & r["pred_topics"]) > 0, axis=1)
    exact = sub.apply(lambda r: r["gold_topics"] == r["pred_topics"], axis=1)
    none_gold = df[~has_topic]
    quiet = none_gold["pred_topics"].map(len) == 0
    print(f"\nTOPICS (excluding sentiment-only groups and user type), {len(sub)} comments with a topic")
    print(f"  Model finds at least one of your topics: {overlap.mean():.1%}")
    print(f"  Model's topic set matches yours exactly: {exact.mean():.1%}")
    print(f"  Comments you gave no topic, where the model also gave none: "
          f"{quiet.mean():.1%} of {len(none_gold)}" if len(none_gold) else "")

    # ---- 3. Per topic -----------------------------------------------------
    rows = []
    for g in sorted(set().union(*df["gold_topics"]) | set().union(*df["pred_topics"])):
        gold_n = df["gold_topics"].map(lambda s: g in s)
        pred_n = df["pred_topics"].map(lambda s: g in s)
        tp = int((gold_n & pred_n).sum())
        rows.append({"topic": g, "yours": int(gold_n.sum()), "model": int(pred_n.sum()), "both": tp,
                     "precision": round(tp / pred_n.sum(), 2) if pred_n.sum() else None,
                     "recall": round(tp / gold_n.sum(), 2) if gold_n.sum() else None})
    print("\nPER TOPIC (precision = of the model's, how many you also tagged; recall = of yours, how many it found)")
    print(pd.DataFrame(rows).sort_values("yours", ascending=False).to_string(index=False))

    # ---- 4. Confidence bands (compare with your Planning Portal QA) --------
    band = pd.cut(df["conf"], [-1, 0.40, 0.70, 2], labels=["<0.40 review", "0.40-0.70", ">=0.70 auto"], right=False)
    bands = df.assign(band=band, hit=grp_hit).groupby("band", observed=False).agg(
        comments=("hit", "size"), group_agreement=("hit", "mean"))
    bands["share"] = (bands["comments"] / len(df)).round(3)
    bands["group_agreement"] = bands["group_agreement"].round(3)
    print("\nBY CONFIDENCE BAND (Planning Portal: 62.4% of rows needed review)")
    print(bands.to_string())
    print("\nPREDICTION METHOD"); print(df["prediction_method"].value_counts().to_string())

    os.makedirs("outputs", exist_ok=True)
    detail = df[["Text", "gold_groups", "pred_groups", "primary_tag", "conf", "prediction_method"]].copy()
    for c in ("gold_groups", "pred_groups"):
        detail[c] = detail[c].map(lambda s: " | ".join(sorted(s)))
    detail.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nDetail per comment written to {OUT_FILE}")


if __name__ == "__main__":
    main()