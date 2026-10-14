"""
Migrates the existing flat tag taxonomy (primary_tag + secondary_tags) onto the
new schema: independent (topic, sentiment) pairs + a standalone user_type field.

This is a RULE-BASED, INTERIM migration — it works entirely from tags that already
exist in `Data`, it does not re-run any model. It closes the gap for topics whose
sentiment is baked into the tag itself (e.g. `Too complex` -> negative), and uses
row-level heuristics where the old data genuinely doesn't have per-topic sentiment.

KNOWN LIMITATION (read before trusting this for anything but a first pass):
When a row has 2+ topics that DON'T carry their own sentiment (e.g. `Document
uploading` + `Payments`, both from secondary_tags) and only one row-level
`Positive/Negative experience` tag, this script applies that SAME sentiment to
both topics. It cannot invent the "document upload great, payment confusing"
split from data that was never captured per-topic in the first place — that
still needs the ABSA aspect-extraction retrain. Rows affected by this specific
limitation are flagged via `multi_topic_same_sentiment_heuristic=True` in the
output so they're easy to find and reprocess later.

RESOLVED: earlier versions of this script mapped Easy to use/navigate/
understand, Too complex, and Confusing onto "Overall Positive Experience" /
"Negative experience" as if those were real topics — they aren't, they're
production groups that bake sentiment into the group name. Those tags are
now treated as pure sentiment signals (SENTIMENT_ONLY_TAGS): they inform
sentiment on whatever real topic is present in the row, and produce no
topic pair of their own when a row has no other topic.
"""

import re
import pandas as pd

# old_tag -> (new_topic, inherent_sentiment_or_None)
#
# Topic names below are RECONCILED against the actual tag_group values this
# pipeline produces (verified against Data sheet primary_tag/primary_tag_group
# pairs), not the fresh names from the original schema proposal — so rows
# backfilled from history and rows classified live use the same vocabulary.
# step_absa looks up ABSA_ASPECTS by these exact strings.
#
# RESOLVED (previously a "known compromise"): Easy to use/navigate/understand,
# Too complex, and Confusing are NOT mapped to a topic below. They were only
# ever sitting in "Overall Positive Experience" / "Negative experience" —
# production groups that bake sentiment into the group name — which is
# exactly the problem this redesign exists to remove. They're handled as
# pure sentiment signals instead (see SENTIMENT_ONLY_TAGS below): they feed
# sentiment into whatever REAL topic is present in the row, and produce no
# topic pair of their own if the row has no other topic.
#
# KNOWN GRANULARITY CHANGE: Tree works / BNG / Discharge conditions collapse
# into one topic ("Planning App/work types") because that's the group
# tag_to_group actually assigns them to today — step_absa can only ever
# produce this one combined group for these three tags, so keeping them
# separate here would make historical data more granular than any live row
# can be.
TAG_MAP = {
    # "General UX" — a real (if loosely-bounded) topic group: learning curve,
    # application-type selection, login, email comms, etc. — not a sentiment
    # catch-all like the two removed above, so this stays.
    "General UX complexity / time": ("General UX", "negative"),
    # Guidance, clarity & jargon — already a clean topic-only group
    "Guidance": ("Guidance, clarity & jargon", "positive"),
    "Lack of guidance": ("Guidance, clarity & jargon", "negative"),
    "Jargon": ("Guidance, clarity & jargon", "negative"),
    # Straight topics, no inherent sentiment — renamed to production groups
    "Document uploading": ("Document upload and handling", None),
    "Payments": ("Payments", None),
    "Form completion": ("Forms & application details", None),
    "Support desk/ Human support": ("Customer support & human assistance", None),
    "LPI tool": ("Location plans, addresses and mapping", None),
    "LPI tool / Drawing tools": ("Location plans, addresses and mapping", None),
    # Tree works/BNG/Discharge collapse to one topic (see granularity note)
    "Tree works": ("Planning App/work types", None),
    "BNG / Biodiversity metric": ("Planning App/work types", None),
    "Discharge conditions": ("Planning App/work types", None),
    # Fees carries inherent negative sentiment when it's specifically "too high";
    # bare "Fees & charges" mentions don't imply a direction on their own
    "Fees too high": ("Fees, charges and quotes", "negative"),
    "Fees & charges": ("Fees, charges and quotes", None),
    # Bugs cluster (Crash merged in per sign-off)
    "Bugs/ glitch": ("Challenges and Workarounds", "negative"),
    "Crash/ data loss/ error message": ("Challenges and Workarounds", "negative"),
    # Forward-looking, not a complaint about something existing
    "Suggestions": ("Missing features & feature requests", "neutral"),
    "Missing/ suggested features": ("Missing features & feature requests", "neutral"),
}

SENTIMENT_ONLY_TAGS = {
    "Positive experience/ other praises": "positive",
    "Negative experience": "negative",
    # Moved here from TAG_MAP — these only ever sat in "Overall Positive
    # Experience" / "Negative experience", production groups that bake
    # sentiment into the group name. They're not real topics; they only
    # inform sentiment on whatever REAL topic is present in the row.
    "Easy to use/ Straightforward": "positive",
    "Easy to navigate": "positive",
    "Easy to understand": "positive",
    "Too complex": "negative",
    "Confusing": "negative",
}
USER_TYPE_TAG = "Beginner/ One-time/ Non-professional users"
RETIRED_TAGS = {"Painpoints"}  # used once in 4,447 rows — dropped, not mapped
NO_COMMENT_TAG = "No comment"

# --- QA-correction overrides -------------------------------------------------
# Where a human reviewer already corrected a row (total data sheet), that's
# ground truth and should win over the raw SVM tag. Only unambiguous,
# single-concept corrections are auto-applied here; compound corrections like
# "confusing, too complex" or "negative experience, unclear guidance" are left
# alone — those need a human to split them, not a guessed parse.
CORRECTION_TOPIC_MAP = {
    "suggestion": ("Missing features & feature requests", "neutral"),
    "missing feature": ("Missing features & feature requests", "neutral"),
    "missing features & feature requests": ("Missing features & feature requests", "neutral"),
    "bugs/ glitch": ("Challenges and Workarounds", "negative"),
    "payment": "Payments",
    "lack of guidance": ("Guidance, clarity & jargon", "negative"),
    "jargons": ("Guidance, clarity & jargon", "negative"),
    "guidance, clarity & jargon": "Guidance, clarity & jargon",
}
CORRECTION_SENTIMENT_MAP = {
    "positive": "positive",
    "positive experinace": "positive",
    "negative": "negative",
    "negative experience": "negative",
    "neutral": "neutral",
    "counfsing": "negative",  # "confusing" (typo) — sentiment-only, not a topic anymore
}
CORRECTION_USER_TYPE_MAP = {
    "missing - beginner user": "beginner_or_one_time",
    # "frequent user" is a *new* user-type value not in the current taxonomy —
    # flagged, not silently mapped onto "beginner".
}


def load_qa_corrections(xls_path):
    """Respondent ID -> {"verdict": ..., "correction_norm": ...} for reviewed rows."""
    td = pd.read_excel(xls_path, sheet_name="total data")
    td = td[td["Agree with SVM tag?"].isin(["Yes", "Yes ", "No", "Partially"])].copy()
    td = td.drop_duplicates(subset="Respondent ID")
    td["Agree with SVM tag?"] = td["Agree with SVM tag?"].str.strip()
    corr_col = "If No/Partially — correct tag (Free text), ideally from your existing tag list"
    out = {}
    for _, r in td.iterrows():
        if pd.isna(r["Respondent ID"]) or pd.isna(r[corr_col]):
            continue
        out[float(r["Respondent ID"])] = {
            "verdict": r["Agree with SVM tag?"],
            "correction_norm": str(r[corr_col]).strip().lower(),
        }
    return out


USER_TYPE_KEYWORDS = re.compile(
    r"first time|new user|never used|first use|novice|beginner", re.IGNORECASE
)


def split_tags(primary_tag, secondary_tags):
    tags = []
    if pd.notna(primary_tag):
        tags.append(primary_tag)
    if pd.notna(secondary_tags):
        tags += [t.strip() for t in secondary_tags.split(",")]
    return tags


def migrate_row(primary_tag, secondary_tags, grouping_sentiment, feedback_text=None,
                 qa_correction=None):
    tags = split_tags(primary_tag, secondary_tags)

    if primary_tag == NO_COMMENT_TAG:
        return [], None, False, None, None, None
    tags = [t for t in tags if t != NO_COMMENT_TAG]  # drop a stray secondary "No comment"

    user_type = "beginner_or_one_time" if USER_TYPE_TAG in tags else None
    pos_hits = [t for t in tags if SENTIMENT_ONLY_TAGS.get(t) == "positive"]
    neg_hits = [t for t in tags if SENTIMENT_ONLY_TAGS.get(t) == "negative"]
    has_pos = bool(pos_hits)
    has_neg = bool(neg_hits)

    topic_tags = [t for t in tags if t not in SENTIMENT_ONLY_TAGS
                  and t != USER_TYPE_TAG and t not in RETIRED_TAGS]

    # Map to new topics, merging duplicates and catching contradictions
    # (e.g. both "Easy to use" and "Too complex" landing on the same topic).
    topics = {}  # new_topic -> {"sentiment": ..., "source": ...}
    for old_tag in topic_tags:
        if old_tag not in TAG_MAP:
            continue  # unrecognised tag — shouldn't happen against this taxonomy
        new_topic, inherent = TAG_MAP[old_tag]
        if new_topic not in topics:
            topics[new_topic] = {"sentiment": inherent, "source": "tag_inherent" if inherent else None}
        elif inherent is not None:
            existing = topics[new_topic]["sentiment"]
            if existing is not None and existing != inherent:
                topics[new_topic] = {"sentiment": "mixed", "source": "contradictory_tags"}
            else:
                topics[new_topic] = {"sentiment": inherent, "source": "tag_inherent"}

    # --- QA-correction override (ground truth beats the raw model tag) ------
    qa_flag = None
    if qa_correction is not None:
        corr = qa_correction["correction_norm"]
        verdict = qa_correction["verdict"]
        if corr in CORRECTION_TOPIC_MAP:
            # Reviewer gave an unambiguous single-concept correction — trust it
            # over the raw model tag regardless of whether they marked the row
            # "No" or "Partially" (e.g. "Partially" + "Suggestion" still means
            # the reviewer knows the right topic, just wasn't 100% on wording).
            mapped = CORRECTION_TOPIC_MAP[corr]
            new_topic, sent = mapped if isinstance(mapped, tuple) else (mapped, None)
            topics = {new_topic: {"sentiment": sent, "source": "qa_correction"}}
            qa_flag = "topic_overridden_by_qa"
        if corr in CORRECTION_SENTIMENT_MAP:
            # Pure sentiment corrections ("positive"/"negative"/"neutral") — apply
            # as the row's best-known sentiment regardless of verdict, since these
            # aren't ambiguous about what they mean.
            has_pos = CORRECTION_SENTIMENT_MAP[corr] == "positive"
            has_neg = CORRECTION_SENTIMENT_MAP[corr] == "negative"
            grouping_sentiment = CORRECTION_SENTIMENT_MAP[corr]
            qa_flag = (qa_flag or "") + "+sentiment_overridden_by_qa"
        if corr in CORRECTION_USER_TYPE_MAP:
            user_type = CORRECTION_USER_TYPE_MAP[corr]
            qa_flag = (qa_flag or "") + "+user_type_added_by_qa"

    # --- Keyword-assisted user_type recall -----------------------------------
    # 79% of rows mentioning first-time/new-user language in this dataset had no
    # user-type tag applied at all (67/85). Model recall on this is weak enough
    # that a keyword assist is warranted here, same rationale as the Fees
    # high-recall requirement. Flagged with its own source so it's auditable —
    # this is NOT treated as equivalent confidence to an explicit tag.
    user_type_source = None
    if user_type == "beginner_or_one_time":
        user_type_source = "tag_inherent" if qa_flag is None else "qa_correction"
    elif feedback_text and USER_TYPE_KEYWORDS.search(str(feedback_text)):
        user_type = "beginner_or_one_time"
        user_type_source = "keyword_assist"


    if has_pos and has_neg:
        row_level_sentiment = "mixed"
    elif has_pos:
        row_level_sentiment = "positive"
    elif has_neg:
        row_level_sentiment = "negative"
    elif pd.notna(grouping_sentiment):
        row_level_sentiment = grouping_sentiment
    else:
        row_level_sentiment = None

    multi_topic_same_sentiment_flag = False
    unresolved_topics = [t for t, v in topics.items() if v["sentiment"] is None]
    if len(unresolved_topics) >= 2 and row_level_sentiment is not None:
        multi_topic_same_sentiment_flag = True

    for t in unresolved_topics:
        if row_level_sentiment is not None:
            source = "row_level_heuristic" if (has_pos or has_neg) else "absa_fallback"
            topics[t] = {"sentiment": row_level_sentiment, "source": source}
        else:
            topics[t] = {"sentiment": "unknown", "source": "unresolved"}

    # `topic_sentiment_pairs` no longer fabricates a topic for bare
    # Positive/Negative experience (or Easy to use/Too complex/Confusing with
    # no other topic) — those tags were never a real topic. But "how did the
    # user feel about the process overall" is still a real, wanted signal —
    # it's just not a topic, so it's returned separately as
    # `overall_experience_sentiment` instead of living inside the topic list.
    overall_experience_sentiment = row_level_sentiment

    pairs = [
        {"topic": t, "sentiment": v["sentiment"], "sentiment_source": v["source"]}
        for t, v in topics.items()
    ]
    return (pairs, user_type, multi_topic_same_sentiment_flag, qa_flag,
            user_type_source, overall_experience_sentiment)


def migrate_dataframe(df, xls_path=None):
    qa_lookup = load_qa_corrections(xls_path) if xls_path else {}

    def _run(r):
        rid = r["Respondent ID"]
        qa = qa_lookup.get(float(rid)) if pd.notna(rid) else None
        return migrate_row(
            r["primary_tag"], r["secondary_tags"], r["grouping_sentiment"],
            feedback_text=r["Feedback_clean"], qa_correction=qa,
        )

    results = df.apply(_run, axis=1)
    df = df.copy()
    df["topic_sentiment_pairs"] = results.apply(lambda r: r[0])
    df["user_type"] = results.apply(lambda r: r[1])
    df["multi_topic_same_sentiment_heuristic"] = results.apply(lambda r: r[2])
    df["qa_override_applied"] = results.apply(lambda r: r[3])
    df["user_type_source"] = results.apply(lambda r: r[4])
    df["overall_experience_sentiment"] = results.apply(lambda r: r[5])
    df["n_topics"] = df["topic_sentiment_pairs"].apply(len)
    return df


if __name__ == "__main__":
    xls_path = "/mnt/user-data/uploads/testing_data.xlsx"
    data = pd.read_excel(xls_path, sheet_name="Data")
    migrated = migrate_dataframe(data, xls_path=xls_path)

    print(f"Rows migrated: {len(migrated)}")
    print(f"Rows with 0 topics (no comment / unmapped): {(migrated['n_topics'] == 0).sum()}")
    print(f"Rows with 1 topic: {(migrated['n_topics'] == 1).sum()}")
    print(f"Rows with 2+ topics: {(migrated['n_topics'] >= 2).sum()}")
    print(f"Rows flagged as multi-topic-same-sentiment heuristic (needs ABSA retrain to truly resolve): "
          f"{migrated['multi_topic_same_sentiment_heuristic'].sum()}")
    print(f"Rows with a standalone user_type: {migrated['user_type'].notna().sum()}")
    print(f"Rows where a QA correction overrode the raw model tag: {migrated['qa_override_applied'].notna().sum()}")
    print(f"Rows with an overall_experience_sentiment: {migrated['overall_experience_sentiment'].notna().sum()}")

    print()
    print("overall_experience_sentiment distribution:")
    print(migrated["overall_experience_sentiment"].value_counts(dropna=False))

    print()
    print("user_type source breakdown:")
    print(migrated["user_type_source"].value_counts(dropna=False))

    all_sources = migrated["topic_sentiment_pairs"].explode().dropna().apply(
        lambda p: p["sentiment_source"] if isinstance(p, dict) else None
    )
    print()
    print("Sentiment source breakdown across all topic pairs:")
    print(all_sources.value_counts())

    out = migrated[[
        "Respondent ID", "Feedback_clean", "Rating", "user_type", "user_type_source",
        "overall_experience_sentiment", "topic_sentiment_pairs",
        "multi_topic_same_sentiment_heuristic", "qa_override_applied",
    ]].copy()
    out["topic_sentiment_pairs"] = out["topic_sentiment_pairs"].apply(str)
    out.to_csv("/mnt/user-data/outputs/migrated_topic_sentiment_data.csv", index=False)
    print()
    print("Saved: migrated_topic_sentiment_data.csv")
