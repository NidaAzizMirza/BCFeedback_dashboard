"""
train_group_model.py
Builds the two files group_topics.py uses, from your hand-tagged comments:
  models/group_model.npz   a small topic model (numpy arrays only, safe to commit)
  data/manual_topics.csv   your tags per comment, keyed by a hash of the cleaned text (no comment text)

    python train_group_model.py                         # reads tagged_examples.csv
    python train_group_model.py --threshold 0.5         # probability needed to add a topic (default 0.5)
    python train_group_model.py --topics "Payments;Fees, charges and quotes"   # choose topics yourself (; between names)

Topics are kept only if, in 5-fold cross-validation, precision >= 0.35 and recall >= 0.40.
Do not commit tagged_examples.csv (it contains the comments).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
import pipeline_config as cfg
import run_pipeline as rp
from group_topics import MODEL_FILE, MANUAL_FILE, normalise, text_key

EXAMPLES = next((a for a in sys.argv[1:] if a.endswith(".csv")), "tagged_examples.csv")
THRESHOLD = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 0.5
CHOSEN = ([t.strip() for t in sys.argv[sys.argv.index("--topics") + 1].split(";") if t.strip()]
          if "--topics" in sys.argv else None)
NOT_TOPICS = {"Overall Positive Experience", "Negative experience", "User type"}
MIN_COMMENTS, MIN_PRECISION, MIN_RECALL, FOLDS = 10, 0.35, 0.40, 5


def load_examples(path):
    ex = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    ex["Text"] = ex["Text"].str.strip()
    ex["Tag"] = ex["Tag"].str.strip()
    ex["Group"] = ex["Tag - Group"].str.strip()
    from group_topics import regroup
    ex["Group"] = regroup(ex["Tag"], ex["Group"])
    ex = ex[(ex["Text"] != "") & (ex["Tag"] != "") & (ex["Group"] != "")]
    pairs = ex.groupby("Text", sort=False).apply(lambda d: list(dict.fromkeys(zip(d["Tag"], d["Group"]))))
    return pairs.reset_index(name="pairs")


def clean(ex):
    raw = pd.DataFrame({
        cfg.COL_RESPONDENT_ID: [str(900000000000 + i) for i in range(len(ex))],
        cfg.COL_DATE: "2026-01-15", cfg.COL_RATING: 3, cfg.COL_FEEDBACK: ex["Text"].values})
    pre = rp.step_preprocess(raw.copy())
    pre[cfg.COL_RESPONDENT_ID] = pre[cfg.COL_RESPONDENT_ID].astype(str)
    ex = ex.assign(**{cfg.COL_RESPONDENT_ID: raw[cfg.COL_RESPONDENT_ID].values})
    return ex.merge(pre[[cfg.COL_RESPONDENT_ID, cfg.COL_FEEDBACK_CLEAN]], on=cfg.COL_RESPONDENT_ID)


def main():
    ex = clean(load_examples(EXAMPLES))
    print(f"{len(ex)} tagged comments kept after the pipeline's own cleaning")

    # ---- manual overrides: tags per cleaned-text hash ------------------------
    merged = {}
    for text, pairs in zip(ex[cfg.COL_FEEDBACK_CLEAN], ex["pairs"]):
        merged.setdefault(text_key(text), [])
        merged[text_key(text)] += [p for p in pairs if p not in merged[text_key(text)]]
    os.makedirs(os.path.dirname(MANUAL_FILE), exist_ok=True)
    pd.DataFrame([{"text_hash": k, "tags": "|".join(t for t, _ in v), "groups": "|".join(g for _, g in v)}
                  for k, v in merged.items()]).to_csv(MANUAL_FILE, index=False)
    print(f"wrote {MANUAL_FILE} ({len(merged)} comments, hashes only)")

    # ---- topic model ---------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    X = normalise(SentenceTransformer(cfg.EMBEDDING_MODEL).encode(
        ex[cfg.COL_FEEDBACK_CLEAN].tolist(), batch_size=32, show_progress_bar=False))
    gold = ex["pairs"].map(lambda p: {g for _, g in p} - NOT_TOPICS)
    counts = pd.Series([g for s in gold for g in s]).value_counts()
    candidates = sorted(counts[counts >= MIN_COMMENTS].index)
    Y = np.array([[t in s for t in candidates] for s in gold])

    def fit(j, idx):
        return LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000).fit(X[idx], Y[idx, j])

    proba = np.zeros(Y.shape)
    for train, test in KFold(FOLDS, shuffle=True, random_state=0).split(X):
        for j in range(len(candidates)):
            proba[test, j] = fit(j, train).predict_proba(X[test])[:, 1]
    keep = []
    print(f"\ncross-validated scores at threshold {THRESHOLD} (comments the model did not train on):")
    for j, t in enumerate(candidates):
        pred = proba[:, j] >= THRESHOLD
        tp = int((pred & Y[:, j]).sum())
        p = tp / pred.sum() if pred.sum() else 0.0
        r = tp / Y[:, j].sum()
        ok = (t in CHOSEN) if CHOSEN else (p >= MIN_PRECISION and r >= MIN_RECALL)
        keep.append(ok)
        print(f"  {'USE ' if ok else 'skip'}  {t:42s} comments {int(Y[:, j].sum()):3d}  precision {p:.2f}  recall {r:.2f}")
    chosen = [t for t, k in zip(candidates, keep) if k]
    if not chosen:
        sys.exit("No topic cleared the bar, so no model was written.")

    coefs, intercepts = [], []
    for j, t in enumerate(candidates):
        if t in chosen:
            m = fit(j, np.arange(len(X)))
            coefs.append(m.coef_[0]); intercepts.append(m.intercept_[0])
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    np.savez(MODEL_FILE, topics=np.array(chosen), coef=np.array(coefs), intercept=np.array(intercepts),
             threshold=np.array(THRESHOLD), n_train=np.array(len(X)))
    print(f"\nwrote {MODEL_FILE}: {len(chosen)} topics, trained on {len(X)} comments")


if __name__ == "__main__":
    main()