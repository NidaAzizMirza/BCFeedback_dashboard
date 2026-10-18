"""
group_topics.py
Two optional additions to step_classify. Each is skipped if its file is missing.

1. A small topic model (models/group_model.npz) trained on your hand-tagged comments. It ADDS
   topics the existing classifier missed. It never removes a topic the existing logic assigned.
2. Manual overrides (data/manual_topics.csv). A comment you tagged by hand keeps your tags
   exactly. The file stores a hash of each comment's cleaned text, not the comment itself.

Numpy only (no pickled scikit-learn objects), so the model file loads the same on any machine
or in GitHub Actions. Build both files with train_group_model.py.
"""
import hashlib
import logging
import os

import numpy as np
import pandas as pd

import pipeline_config as cfg

log = logging.getLogger(__name__)

MODEL_FILE = os.path.join(cfg.MODELS_DIR, "group_model.npz")
MANUAL_FILE = os.path.join(cfg.BASE_DIR, "data", "manual_topics.csv")
SENTIMENT_ONLY = {"Overall Positive Experience", "Negative experience"}
IGNORE_GROUPS = {"User type"}          # user type is its own field, never a topic
OUTPUT_COLUMNS = ("primary_tag", "primary_tag_group", "secondary_tags",
                  "secondary_tag_groups", "svm_confidence", "prediction_method")


def text_key(text) -> str:
    """Stable id for a comment: hash of its cleaned, lower-cased, whitespace-normalised text."""
    norm = " ".join(str(text).lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def normalise(embeddings):
    e = np.asarray(embeddings, dtype=float)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


def load_model(path=MODEL_FILE):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {"topics": [str(t) for t in z["topics"]], "coef": z["coef"], "intercept": z["intercept"],
            "threshold": float(z["threshold"])}


def predict_topics(embeddings, model):
    """Topics per comment, most likely first, whose probability is at or above the model's threshold."""
    z = normalise(embeddings) @ model["coef"].T + model["intercept"]
    proba = 1.0 / (1.0 + np.exp(-z))
    return [[model["topics"][j] for j in np.argsort(-row) if row[j] >= model["threshold"]] for row in proba]


def load_manual(path=MANUAL_FILE):
    """{text_hash: [(tag, group), ...]} from data/manual_topics.csv."""
    if not os.path.exists(path):
        return {}
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for key, tags, groups in zip(m["text_hash"], m["tags"], m["groups"]):
        pairs = [(t, g) for t, g in zip(tags.split("|"), groups.split("|"))
                 if t and g and g not in IGNORE_GROUPS]
        if pairs:
            out[key] = pairs
    return out


def apply_group_topics(df, embeddings, known_groups, text_col=cfg.COL_FEEDBACK_CLEAN):
    """Add model topics and apply manual overrides to step_classify's output (same row order as embeddings)."""
    df = df.copy()
    model, manual = load_model(), load_manual()
    if model is None and not manual:
        log.info("Group topics: no models/group_model.npz or data/manual_topics.csv found, skipping")
        return df
    col = {c: df.columns.get_loc(c) for c in OUTPUT_COLUMNS}
    topics = predict_topics(embeddings, model) if model is not None else [[] for _ in range(len(df))]
    keys = [text_key(t) for t in df[text_col]]

    n_manual, n_added, per_topic = 0, 0, {}
    for i, key in enumerate(keys):
        if key in manual:                                   # your own tags win, exactly as tagged
            pairs = manual[key]
            real = [p for p in pairs if p[1] not in SENTIMENT_ONLY]
            primary = (real or pairs)[0]
            rest, seen = [], {primary[0]}
            for tag, group in pairs:
                if tag not in seen:
                    rest.append((tag, group)); seen.add(tag)
            df.iat[i, col["primary_tag"]] = primary[0]
            df.iat[i, col["primary_tag_group"]] = primary[1]
            df.iat[i, col["secondary_tags"]] = ", ".join(t for t, _ in rest)
            df.iat[i, col["secondary_tag_groups"]] = ", ".join(g for _, g in rest)
            df.iat[i, col["svm_confidence"]] = 1.0
            df.iat[i, col["prediction_method"]] = "override"
            n_manual += 1
            continue
        primary_group = str(df.iat[i, col["primary_tag_group"]])
        sec = df.iat[i, col["secondary_tag_groups"]]
        sec = "" if pd.isna(sec) else str(sec)
        added = False
        for g in topics[i]:
            if g in known_groups and g != primary_group and g not in sec:
                sec = f"{sec}, {g}" if sec else g
                per_topic[g] = per_topic.get(g, 0) + 1
                added = True
        if added:
            df.iat[i, col["secondary_tag_groups"]] = sec
            n_added += 1
    log.info(f"Group topics: {n_manual} comments kept your manual tags; the topic model added topics to "
             f"{n_added} others {per_topic if per_topic else ''}")
    return df