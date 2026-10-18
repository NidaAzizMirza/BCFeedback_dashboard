"""
cv_group_classifier.py
Quick experiment: could a small classifier trained on YOUR tagged comments predict topics
(tag groups) better than the current Planning Portal model?

For each topic with enough examples it does 5-fold cross-validation: the model is trained on
80% of your comments and scored on the other 20%, so every score is on comments it never saw.
If outputs/eval_tagged_examples.csv exists (from eval_tagged_examples.py), the current model's
precision/recall on the same comments is shown next to it.

    python cv_group_classifier.py                    # sentence embeddings (same model as the pipeline)
    python cv_group_classifier.py --tfidf            # word-based baseline, no model download
    python cv_group_classifier.py --threshold 0.4    # lower = finds more, less precise (default 0.5)

Read-only: does not change the pipeline, the models, or any data.
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

EXAMPLES = next((a for a in sys.argv[1:] if a.endswith(".csv")), "tagged_examples.csv")
USE_TFIDF = "--tfidf" in sys.argv
THRESHOLD = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 0.5
NOT_TOPICS = {"Overall Positive Experience", "Negative experience", "User type"}
MIN_COMMENTS = 10          # topics with fewer tagged comments are left out
FOLDS = 5
EVAL_FILE = os.path.join("outputs", "eval_tagged_examples.csv")


def load_gold(path):
    ex = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    ex["Text"] = ex["Text"].str.strip()
    ex["Group"] = ex["Tag - Group"].str.strip()
    ex = ex[(ex["Text"] != "") & ~ex["Group"].isin(NOT_TOPICS)]
    return ex.groupby("Text")["Group"].agg(set)          # comment -> set of topics


def features(texts):
    if USE_TFIDF:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)


def current_model_topics(texts):
    """Current model's topic sets for the same comments, if the evaluation file exists."""
    if not os.path.exists(EVAL_FILE):
        return None
    ev = pd.read_csv(EVAL_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sets = {t: {g.strip() for g in p.split("|") if g.strip()} - NOT_TOPICS
            for t, p in zip(ev["Text"].str.strip(), ev["pred_groups"])}
    return pd.Series(texts).map(sets)                    # NaN where the comment was dropped by preprocessing


def main():
    all_gold = load_gold(EXAMPLES)
    all_texts = pd.read_csv(EXAMPLES, dtype=str, keep_default_na=False, encoding="utf-8-sig")["Text"].str.strip()
    texts = sorted(set(t for t in all_texts if t))                       # every tagged comment, incl. sentiment-only ones
    gold = pd.Series({t: all_gold.get(t, set()) for t in texts})
    counts = pd.Series([g for s in gold for g in s]).value_counts()
    topics = sorted(counts[counts >= MIN_COMMENTS].index)
    print(f"{len(texts)} tagged comments | topics with >= {MIN_COMMENTS} comments: {len(topics)}")
    print(f"features: {'TF-IDF words' if USE_TFIDF else cfg.EMBEDDING_MODEL} | threshold {THRESHOLD} | {FOLDS}-fold CV\n")

    cur = current_model_topics(texts)
    if cur is not None:                                                  # compare on comments the current model scored
        keep = cur.notna().values
        texts = [t for t, k in zip(texts, keep) if k]
        gold = gold[texts]
        cur = cur[keep].reset_index(drop=True)
        print(f"comparing on the {len(texts)} comments the current model scored\n")

    gold = gold.map(lambda s: s & set(topics)).reset_index(drop=True)
    X = features(texts)
    Y = np.array([[t in s for t in topics] for s in gold])
    proba = np.zeros(Y.shape)
    for train, test in KFold(FOLDS, shuffle=True, random_state=0).split(np.arange(len(texts))):
        for j in range(len(topics)):
            if Y[train, j].sum() == 0 or Y[train, j].sum() == len(train):
                continue
            clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000).fit(X[train], Y[train, j])
            proba[test, j] = clf.predict_proba(X[test])[:, 1]
    pred = proba >= THRESHOLD

    rows = []
    for j, t in enumerate(topics):
        tp = int((pred[:, j] & Y[:, j]).sum())
        r = {"topic": t, "yours": int(Y[:, j].sum()), "new_precision": round(tp / pred[:, j].sum(), 2) if pred[:, j].sum() else None,
             "new_recall": round(tp / Y[:, j].sum(), 2)}
        if cur is not None:
            cp = cur.map(lambda s: t in s); ctp = int((cp & Y[:, j]).sum())
            r["current_precision"] = round(ctp / cp.sum(), 2) if cp.sum() else None
            r["current_recall"] = round(ctp / Y[:, j].sum(), 2)
        rows.append(r)
    print("PER TOPIC (scores on comments the model did not train on)")
    print(pd.DataFrame(rows).sort_values("yours", ascending=False).to_string(index=False))

    has = Y.any(axis=1)
    pred_sets = [{t for t, p in zip(topics, row) if p} for row in pred]
    overlap = np.mean([len(gold[i] & pred_sets[i]) > 0 for i in np.where(has)[0]])
    exact = np.mean([gold[i] == pred_sets[i] for i in np.where(has)[0]])
    print(f"\nAMONG {int(has.sum())} COMMENTS WITH ONE OF THESE TOPICS")
    print(f"  new model finds at least one of your topics: {overlap:.1%}   exact set: {exact:.1%}")
    if cur is not None:
        cs = [cur[i] & set(topics) for i in range(len(texts))]
        print(f"  current model:                               {np.mean([len(gold[i] & cs[i]) > 0 for i in np.where(has)[0]]):.1%}   exact set: "
              f"{np.mean([gold[i] == cs[i] for i in np.where(has)[0]]):.1%}")


if __name__ == "__main__":
    main()