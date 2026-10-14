"""
backfill_master.py
──────────────────
Rebuilds the new topic/sentiment schema for EVERY row in master.xlsx, not just
the rows the live pipeline has processed since the redesign.

What it does (each step is a small function, so you can switch any of them off):

  1. Clean group names (strip the "General UX " trailing space) and parse the
     comma-separated `secondary_tag_groups` cell WITHOUT splitting group names
     that contain commas ("Fees, charges and quotes", etc.).
  2. Flag junk: test submissions, vacuous comments ("no", "nothing").
  3. Redact PII (NI numbers, dates of birth, phone numbers, emails, postcodes)
     into a new `feedback_redacted` column. Use this column in the dashboard.
  4. Topic recall: a keyword lexicon finds topics the SVM/fallback missed.
     It ONLY fills gaps (rows with no real topic) plus an always-on check for
     Fees (agreed high-recall topic). Each hit is stored with its evidence.
  5. Build `topic_sentiments`: [{topic, sentiment, confidence}] per respondent. The
     overall experience is ONE entry inside this list ("Overall Positive Experience" /
     "Negative experience", named by polarity); real topics follow it.
  6. Fix one-word comments ("good", "great") that ABSA skips and that were
     defaulting to neutral.
  7. Fill `overall_experience_sentiment` (falls back to old
     `absa_overall_sentiment`) and `user_type` (negation-aware keywords).
  8. Add review tiers so the human queue is smaller and better targeted.

It never overwrites the input file. It writes a new workbook + a short report.

Usage:
    python backfill_master.py --in data/master.xlsx --out data/master_backfilled.xlsx --jsonl data/tags.jsonl
    python backfill_master.py --in data/master.xlsx --dry-run        # report only

Logic notes
  • Column names mirror pipeline_config.py (COL_RESPONDENT_ID etc.).
  • Respondent ID is read AS TEXT. Reading it as a number is what caused the
    earlier scientific-notation dedup bug.
  • Lexicon precision is NOT validated against your QA sample (that file was
    not available). Treat `source == "lexicon"` pairs as lower certainty.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Columns (mirror pipeline_config.py) ────────────────────────────────────
COL_ID, COL_TEXT, COL_RATING, COL_DATE = "Respondent ID", "Feedback_clean", "Rating", "Start Date"

SENTIMENT_ONLY_GROUPS = {"Overall Positive Experience", "Negative experience"}
NON_TOPIC_GROUPS = SENTIMENT_ONLY_GROUPS | {"Miscellaneous", "User type"}
POLAR = {"positive", "negative"}

# Same tiers as the confidence-routing decision (QA agreement 98% / 86% / 61%)
CONF_HIGH, CONF_REVIEW_BELOW = 0.70, 0.40
MAX_TOPICS_PER_ROW = 3
BACKFILL_VERSION = "2026-09-v1"

# ══════════════════════════════════════════════════════════════════════════
# TOPIC LEXICON  — one place to edit. Keys are the REAL production group names.
# Text in Feedback_clean is lower-case with apostrophes removed ("dont").
# Broad single words that caused past false positives ("tree", "refund") are
# deliberately NOT used on their own.
# Listed in priority order: specific topics before broad ones.
# ══════════════════════════════════════════════════════════════════════════
TOPIC_LEXICON: dict[str, list[str]] = {
    "Payments": [
        r"payments?", r"pay(?:ing)? (?:online|by|with|through|via)", r"paid online",
        r"cards?", r"debit cards?", r"credit cards?", r"fee calc\w*", r"check ?out",
        r"invoices?", r"receipts?", r"worldpay", r"gov ?pay", r"bacs",
    ],
    "Fees, charges and quotes": [
        r"fees?", r"costs?", r"costly", r"expensive", r"pric(?:e|es|ed|ing)",
        r"charg(?:e|es|ed|ing)", r"overpriced", r"afford\w*", r"rip ?off",
        r"waste of money", r"value for money", r"£\s?\d+", r"cheap\w*",
    ],
    "Document upload and handling": [
        r"upload\w*", r"attach\w*", r"pdfs?",
        r"file (?:size|sizes|name|names|type|types|format|formats|limit)", r"filenames?",
        r"documents? (?:submission|section|types?|handling|list)",
        r"(?:submit|submitting|add|adding|delete|deleting|remove|removing|replace|replacing)"
        r" (?:all )?(?:the )?(?:correct |required |supporting |multiple )?(?:documents|docs|files|drawings)",
        r"scan(?:ned|ning)?",
    ],
    "Location plans, addresses and mapping": [
        r"maps?", r"mapping", r"location plans?", r"site plans?", r"block plans?",
        r"site location", r"boundar(?:y|ies)", r"red ?lines?", r"post ?codes?",
        r"drawing tools?", r"draw(?:ing)? (?:the |a |my )?(?:line|lines|boundary|outline|shape|polygon)",
        r"uprn", r"grid ref\w*", r"address(?:es)? (?:lookup|look up|search|list|drop ?down|finder|not found)",
        r"(?:full|site|property|correct) address(?:es)?", r"ordnance survey", r"satellite",
    ],
    "Planning App/work types": [
        r"biodiversity", r"bng", r"net gain", r"discharge (?:of )?(?:planning )?conditions?",
        r"condition discharge", r"tree (?:works?|preservation|surgery|application|order|officer|report)", r"tpo",
        r"trees? protection order", r"protected trees?", r"arboricultur\w*", r"felling",
        r"crown (?:reduction|lift\w*)", r"root protection", r"woodland",
        r"listed building", r"lawful (?:development|use)", r"lawfulness", r"prior approval",
        r"householder", r"retrospective", r"change of use", r"conservation area",
        r"permitted development", r"reserved matters", r"outline (?:application|permission)",
        r"variation of condition", r"non ?material", r"advertis\w+ consent", r"hedgerow",
        r"agricultur\w+", r"type of (?:application|work|project|development)",
    ],
    "Challenges and Workarounds": [
        r"bugs?", r"buggy", r"glitch\w*", r"crash\w*", r"froze\w*", r"freez\w*", r"(?<!trial and )errors?",
        r"delet(?:e|es|ed|ing) (?:my|all|the|everything)", r"erased", r"wiped",
        r"erased", r"disappear\w*", r"vanish\w*",
        r"not (?:working|loading|saving|responding|displaying|showing)",
        r"(?:does ?nt|does not|did ?nt|did not|wo ?nt|will not|would ?nt|could ?nt|can ?not|cant)"
        r" (?:work|load|save|open|submit|display|show|upload)",
        r"time[ds]? out", r"logged out", r"logs? me out",
        r"start(?:ed)? (?:again|over|from scratch)",
        r"lost (?:all |my |the |everything|data|work|progress|information)",
        r"stuck", r"work ?arounds?", r"slow(?:ly)?", r"technical (?:issues?|problems?|difficult\w*|hitch)",
        r"hitch(?:es)?", r"unable to (?:open|save|submit|load|proceed|continue)",
    ],
    "Forms & application details": [
        r"questions?", r"fields?", r"mandatory", r"required (?:fields?|information|questions?)",
        r"ownership certificates?", r"certificate [a-e]",
        r"(?:same|repeat\w*) (?:information|details|questions?|data)",
        r"(?:applicant|agent) details", r"pre ?populat\w*", r"auto ?(?:fill|populate)\w*",
        r"drop ?downs?", r"tick ?box\w*", r"cil", r"sections? (?:did ?nt|do ?nt|does ?nt|not) apply",
    ],
    "Customer support & human assistance": [
        r"help ?desk", r"support (?:team|desk|line|staff|centre|center)",
        r"customer (?:service|services|support|care)", r"telephone", r"phone (?:call|number|line|support)",
        r"live ?chat", r"chat ?bot", r"rang", r"called (?:the |them |support)",
        r"speak(?:ing)? (?:to|with) (?:a |someone|somebody|an? (?:real|actual)|person|human)",
        r"no (?:one|reply|response|answer)", r"(?:never|not) (?:replied|responded|answered|got back)",
    ],
    "Guidance, clarity & jargon": [
        r"jargon", r"terminology", r"technical (?:terms|language|wording)", r"wording",
        r"plain (?:english|language)", r"simple english", r"lay ?(?:man|person|people)",
        r"(?:not|isn ?t|wasn ?t|aren ?t|weren ?t|un)\s?clear\w*", r"clearer", r"clarity",
        r"guidance", r"instructions?", r"explain\w*", r"explanation\w*",
        r"(?:difficult|hard) to follow",
        r"help (?:text|video|videos|notes|page|pages|section|guide)", r"tool ?tips?",
        r"guidelines?", r"tutorial",
    ],
    "General UX": [
        r"layout", r"interface", r"clunky", r"cumbersome", r"fiddly",
        r"time ?consuming", r"takes? (?:too |far too |so )?long",
        r"took (?:ages|hours|days|forever|too long|so long|a long time|far too long)",
        r"long ?winded", r"lengthy", r"laborious", r"tedious",
        r"too many (?:steps|clicks|screens|pages|stages)",
        # "complicated"/"complex" were missing entirely.
        r"complicated", r"(?:too|very|so|overly|extremely|quite|really) complex",
        # Sentiment-agnostic on purpose: positive navigation mentions
        # ("easy to navigate through all the options") are now a real
        # topic too, not sentiment-only — confirmed decision, reversing
        # an earlier fix that scoped this to negative context only.
        # Sentiment is still scored separately, same as every other topic.
        r"navigat\w*",
        r"(?:difficult|hard|impossible) to use",
        r"tricky", r"(?:impossible|hard|difficult) to work out",
    ],
    "Missing features & feature requests": [
        r"would be (?:nice|useful|helpful|great|good|better|easier|handy)", r"would love", r"would like to see",
        r"suggest\w*", r"please (?:add|allow|include|make|provide|consider)", r"option (?:to|for)",
        r"(?:should|could) (?:be able|allow|let)", r"needs? to (?:allow|let)",
        r"(?:no|lack of|missing) (?:option|way|facility|ability|function\w*|feature\w*|button)",
        r"features?", r"wish\w*", r"why (?:not|cant|can t|isnt)", r"clon(?:e|ing)",
        r"add (?:a |an )?(?:function|feature|button|option)",
    ],
}
PRIORITY = list(TOPIC_LEXICON)                 # tie-break order when capping topics
ALWAYS_CHECK = {"Fees, charges and quotes"}    # high-recall topic: check even if row already has topics

# ── One-word comment sentiment (ABSA_MIN_WORDS=2 skips these, so they defaulted to neutral) ──
SHORT_POS = {"good", "great", "excellent", "easy", "perfect", "awesome", "brilliant", "straightforward",
             "simple", "fantastic", "amazing", "wonderful", "nice", "superb", "fab", "quick", "smooth", "clear"}
SHORT_NEG = {"bad", "poor", "terrible", "awful", "difficult", "rubbish", "useless", "horrible", "hard",
             "confusing", "slow", "expensive", "frustrating", "complicated", "appalling", "disappointing"}
SHORT_NEUTRAL = {"ok", "okay", "fine", "alright", "average", "adequate", "no", "none", "nothing",
                 "na", "nope", "nil", "test", "testing"}

# ── PII patterns (text is lower-case, punctuation stripped, so digits are often space-separated) ──
PII_PATTERNS = {
    "[NI_NUMBER]": r"\b[a-z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[a-d]\b",
    "[DOB]":       r"\b(?:0?[1-9]|[12]\d|3[01])[\s/.\-]?(?:0?[1-9]|1[0-2])[\s/.\-]?(?:19|20)\d{2}\b",
    "[PHONE]":     r"(?<!\d)(?:\+?44\s?|0)\d{3,4}\s?\d{3}\s?\d{3,4}(?!\d)",
    "[EMAIL]":     r"[\w.\-]+@[\w.\-]+\.\w+",
    "[POSTCODE]":  r"\b[a-z]{1,2}\d[a-z\d]?\s?\d[a-z]{2}\b",
    "[NUMBER]":    r"\b\d{9,}\b",
}

# ── user_type ─────────────────────────────────────────────────────────────
# Imported directly from run_pipeline.py rather than maintained as a
# separate copy here — USER_TYPE_KEYWORDS and extract_user_type() are the
# single source of truth for user_type detection, used identically by the
# live pipeline and this one-off/historical tooling. No more drift risk:
# a keyword added in one place is automatically used in both.
from run_pipeline import USER_TYPE_KEYWORDS, extract_user_type  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# 1. Load + normalise
# ══════════════════════════════════════════════════════════════════════════
def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={COL_ID: str})
    df[COL_TEXT] = df[COL_TEXT].fillna("").astype(str)
    df["primary_tag_group"] = df["primary_tag_group"].map(_norm)   # fixes "General UX "
    return df


def parse_groups(cell, known_longest_first: list[str]) -> list[str]:
    """Split `secondary_tag_groups` using the KNOWN group names.
    A naive split(',') breaks 'Fees, charges and quotes' into two fake groups."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    s, i, out = _norm(cell), 0, []
    while i < len(s):
        if s[i] in ", ":
            i += 1
            continue
        for g in known_longest_first:
            if s.startswith(g, i):
                out.append(g)
                i += len(g)
                break
        else:                                   # unknown name: read to the next ", "
            j = s.find(", ", i)
            j = len(s) if j == -1 else j
            out.append(s[i:j])
            i = j
    return out


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    known = sorted({g for g in df["primary_tag_group"].unique() if g}, key=len, reverse=True)
    sec = df["secondary_tag_groups"].map(lambda c: parse_groups(c, known))
    df["_all_groups"] = [[p] + [g for g in s if g != p] for p, s in zip(df["primary_tag_group"], sec)]
    df["_existing_topics"] = df["_all_groups"].map(lambda gs: [g for g in gs if g not in NON_TOPIC_GROUPS])
    tags = df["primary_tag"].fillna("").astype(str) + "," + df["secondary_tags"].fillna("").astype(str)
    df["_has_beginner_tag"] = tags.str.contains("Beginner", case=False)
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. Flags: junk, PII
# ══════════════════════════════════════════════════════════════════════════
def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    t = df[COL_TEXT].str.strip().str.lower()
    df["words"] = t.str.split().str.len().fillna(0).astype(int)
    df["is_test"] = t.str.fullmatch(r"(?:qa\s+)?test(?:ing)?(?:\s+test(?:ing)?)*") | t.str.startswith("qa testing")
    df["is_vacuous"] = t.str.fullmatch(r"|no|none|nothing|na|n a|nope|nil|no comments?|not really|no thanks")

    red = df[COL_TEXT]
    pii_hit = pd.Series(False, index=df.index)
    for label, pat in PII_PATTERNS.items():
        pii_hit |= red.str.contains(pat, flags=re.I, regex=True)
        red = red.str.replace(pat, label, flags=re.I, regex=True)
    df["pii_flag"] = pii_hit
    df["feedback_redacted"] = red
    return df


# ══════════════════════════════════════════════════════════════════════════
# 3. Topic lexicon (vectorised: one pass per topic over the whole column)
# ══════════════════════════════════════════════════════════════════════════
def lexicon_hits(text: pd.Series) -> dict[str, pd.Series]:
    """topic -> Series of matched-term lists (empty list = no hit)."""
    out = {}
    for topic, terms in TOPIC_LEXICON.items():
        rx = r"(?<!\w)(?:" + "|".join(terms) + r")(?!\w)"
        out[topic] = text.str.lower().str.findall(rx, flags=re.I)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 4. Sentiment helpers
# ══════════════════════════════════════════════════════════════════════════
ASPECT_RE = re.compile(r"([^:,]+):\s*(positive|negative|neutral)\s*\(([\d.]+)\)")


def aspect_names(cell) -> list[str]:
    """The specific ABSA aspect names behind a topic's sentiment (e.g.
    ["tree works", "biodiversity"]) — the sub-topic-level detail a bare
    group name doesn't carry."""
    if not isinstance(cell, str):
        return []
    return [name.strip() for name, _, _ in ASPECT_RE.findall(cell)]


def aspect_sentiment(cell) -> tuple[str, float] | None:
    """The 2-3 aspects per group are synonyms (fees/charges/cost) = ONE question
    asked several times. Average them (an ensemble) instead of treating them
    as separate topics.

    Signed-mean cancellation is deliberate when aspects DISAGREE (e.g.
    fees:negative(0.99) + cost:positive(0.62) -> low-confidence neutral,
    correctly flagging a genuinely mixed/uncertain read) — but it has a
    real bug when every aspect agrees on "neutral": neutral contributes 0
    to the signed sum regardless of how confidently neutral the model was,
    so 3 aspects at neutral(0.7-0.9) averaged to confidence 0.0, which
    reads as "no data" rather than "confidently neutral". Special-cased
    below: only average signed values when there's an actual polarity
    signal to weigh; an all-neutral cell reports the model's own
    neutral-class confidence instead.
    """
    if not isinstance(cell, str):
        return None
    triples = ASPECT_RE.findall(cell)
    if not triples:
        return None
    if all(lab == "neutral" for _, lab, _ in triples):
        return "neutral", round(float(np.mean([float(sc) for _, _, sc in triples])), 3)
    vals = [(1 if lab == "positive" else -1 if lab == "negative" else 0) * float(sc)
            for _, lab, sc in triples]
    m = float(np.mean(vals))
    return ("positive" if m > 0.25 else "negative" if m < -0.25 else "neutral"), round(abs(m), 3)


def short_comment_sentiment(text: str, rating: float) -> tuple[str, str]:
    w = text.strip().lower()
    if w in SHORT_POS:
        return "positive", "short_lexicon"
    if w in SHORT_NEG:
        return "negative", "short_lexicon"
    if w in SHORT_NEUTRAL or not w:
        return "neutral", "short_lexicon"
    return ("positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"), "rating_fallback"


def row_sentiment(row) -> tuple[str, str]:
    """Row-level sentiment used for topics that have no per-topic ABSA score.
    Evidence from the data: grouping_sentiment is better on complaint/request
    wording ('why does your fee keep rising'); absa_overall is better when
    grouping_sentiment says neutral. So: grouping first, then ABSA if it is polar."""
    if row.words <= 1:
        return short_comment_sentiment(row[COL_TEXT], row[COL_RATING])
    if row.grouping_sentiment in POLAR:
        return row.grouping_sentiment, "grouping_sentiment"
    if row.absa_overall_sentiment in POLAR:
        return row.absa_overall_sentiment, "absa_overall"
    return (row.grouping_sentiment if isinstance(row.grouping_sentiment, str) else "neutral"), "grouping_sentiment"


def existing_new_pipeline_pairs(cell) -> list[dict]:
    """Keep real-topic pairs the live pipeline already produced (they have true per-topic ABSA)."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        pairs = json.loads(cell)
    except json.JSONDecodeError:
        return []
    return [p for p in pairs if p.get("topic") not in NON_TOPIC_GROUPS]



# ══════════════════════════════════════════════════════════════════════════
# 5. Overall sentiment + user type
#    (runs BEFORE topics because the "overall experience" entry needs it)
# ══════════════════════════════════════════════════════════════════════════
def add_overall_and_user_type(df: pd.DataFrame) -> pd.DataFrame:
    short = df["words"] <= 1
    short_vals = [short_comment_sentiment(t, r)[0] for t, r in zip(df[COL_TEXT], df[COL_RATING])]
    fallback = df["absa_overall_sentiment"].where(df["absa_overall_sentiment"].notna(), df["grouping_sentiment"])
    overall = pd.Series(np.where(short, short_vals, fallback), index=df.index)
    df["overall_experience_sentiment"] = df["overall_experience_sentiment"].where(
        df["overall_experience_sentiment"].notna(), overall)

    # user_type: detect ALL matching categories via the real pipeline's own
    # extract_user_type() (imported above), then union with whatever's
    # already in the cell (a value the live pipeline already wrote, or an
    # earlier run of this script) — never drop an existing tag, only add
    # to it. Using the live function directly means the negation handling,
    # keyword list, and matching behavior are identical to what
    # run_pipeline.py does on new rows — no separate logic to keep in sync.
    text = df[COL_TEXT].str.lower()

    def detect_types(t: str) -> set[str]:
        hits = extract_user_type(t)
        return set(hits.split(", ")) if hits else set()

    detected = text.map(detect_types)

    def combine(existing, det, has_beginner_tag):
        types = set(str(existing).split(", ")) if isinstance(existing, str) and existing.strip() else set()
        types |= det
        if has_beginner_tag:
            types.add("beginner_or_one_time")
        return ", ".join(sorted(types)) if types else None

    df["user_type"] = [combine(e, d, h) for e, d, h in zip(df["user_type"], detected, df["_has_beginner_tag"])]

    # Rating says one thing, sentiment says the opposite -> likely a missed negation. Flag for review.
    # (neutral + 1-2 stars is included: e.g. "not that easy" was scored neutral. We flag it, we do
    #  NOT assign a sentiment from the rating — 4-5 star comments are often genuine complaints.)
    ov = df["overall_experience_sentiment"]
    df["rating_sentiment_conflict"] = (
        ((df[COL_RATING] <= 2) & ov.isin(["positive", "neutral"])) |
        ((df[COL_RATING] == 5) & (ov == "negative")))
    return df


# ══════════════════════════════════════════════════════════════════════════
# 6. Build topic_sentiments  (target schema — see to_record())
# ══════════════════════════════════════════════════════════════════════════
# The overall-experience entry lives INSIDE topic_sentiments, named by polarity so the
# topic name and the sentiment can never contradict each other.
OVERALL_TOPIC = {"positive": "Overall Positive Experience", "negative": "Negative experience"}


def overall_entry(r) -> dict | None:
    """One overall-experience entry per row. Neutral / empty / test rows get none."""
    if r["is_test"] or r["is_vacuous"]:
        return None
    sent = r["overall_experience_sentiment"]
    topic = OVERALL_TOPIC.get(sent)
    if topic is None:
        return None
    conf = None                       # only use a real model score; never invent one
    if r["primary_tag_group"] in SENTIMENT_ONLY_GROUPS:
        asp = aspect_sentiment(r["absa_aspect_sentiments"])
        if asp and asp[0] == sent:
            conf = asp[1]
    return {"topic": topic, "detail": None, "sentiment": sent, "confidence": conf, "source": "overall"}


def build_topics(df: pd.DataFrame) -> pd.DataFrame:
    hit_lists = {t: s.tolist() for t, s in lexicon_hits(df[COL_TEXT]).items()}
    out_final, out_prov, out_real = [], [], []
    # Pair assembly is inherently per-row (variable-length lists); the expensive
    # text matching above is already vectorised.
    for idx, (_, r) in enumerate(df.iterrows()):
        sent, sent_src = row_sentiment(r)
        lex = {t: sorted(set(hit_lists[t][idx])) for t in PRIORITY if hit_lists[t][idx]}

        pairs = existing_new_pipeline_pairs(r["topic_sentiment_pairs"])   # real per-topic ABSA, keep

        have = {p["topic"] for p in pairs}

        # a) topics already assigned by SVM / fallback / secondary tags
        for k, topic in enumerate(r["_existing_topics"]):
            if topic in have:
                continue
            asp = aspect_sentiment(r["absa_aspect_sentiments"]) if k == 0 and r["primary_tag_group"] == topic else None
            if asp:
                # aspect_names(): which specific ABSA aspects (e.g. "tree
                # works", "biodiversity") drove this topic's sentiment —
                # the detail a bare group name like "Planning App/work
                # types" doesn't carry on its own.
                names = aspect_names(r["absa_aspect_sentiments"])
                pairs.append({"topic": topic, "detail": ", ".join(names) or None,
                              "sentiment": asp[0], "confidence": asp[1], "source": "absa_aspect"})
            else:
                # Nothing more specific than "the whole comment leans this
                # way" — no lexicon or aspect evidence pins it to a phrase.
                pairs.append({"topic": topic, "detail": None, "sentiment": sent, "confidence": None,
                              "source": f"row_level_inherited:{sent_src}"})
            have.add(topic)

        # b) lexicon: fill gaps (no real topic yet) and always check Fees
        need_gap_fill = not have and not r["is_vacuous"] and not r["is_test"] and r["words"] >= 3
        candidates = [t for t in lex if t not in have and (need_gap_fill or t in ALWAYS_CHECK)]
        candidates.sort(key=lambda t: (-len(lex[t]), PRIORITY.index(t)))
        for topic in candidates[: max(0, MAX_TOPICS_PER_ROW - len(have))]:
            # detail = the actual matched word(s)/phrase(s) for this topic,
            # e.g. "trees", "tpo" — the sub-topic-level evidence.
            pairs.append({"topic": topic, "detail": ", ".join(lex[topic]), "sentiment": sent,
                          "confidence": None, "source": f"lexicon:{sent_src}"})
            have.add(topic)

        # Hard override, not a lexicon boost: any form of "nomination" must
        # tag Payments, never Fees, charges and quotes — "nomination fee"
        # contains the bare word "fee", which independently matches the
        # Fees lexicon too, so both could otherwise end up in the list
        # together or Fees alone if the gap-fill cap trims Payments first.
        if re.search(r"\bnominat\w*\b", r[COL_TEXT].lower()):
            pairs = [p for p in pairs if p["topic"] != "Fees, charges and quotes"]
            have = {p["topic"] for p in pairs}
            if "Payments" not in have:
                pairs.append({"topic": "Payments", "detail": "nomination", "sentiment": sent,
                              "confidence": None, "source": "nomination_override"})

        ov = overall_entry(r)
        final = ([ov] if ov else []) + pairs
        out_final.append(final)
        out_real.append(pairs)
        out_prov.append([{"topic": p["topic"], "source": p["source"],
                          "evidence": lex.get(p["topic"], [])} for p in final])

    clean = lambda ps: [{"topic": p["topic"], "detail": p.get("detail"), "sentiment": p["sentiment"],
                         "confidence": p["confidence"]} for p in ps]
    df["topic_sentiments"] = [json.dumps(clean(p), ensure_ascii=False) for p in out_final]
    # Legacy shape (topic/sentiment/confidence/source, overall entry included) — matches
    # exactly what the live pipeline's own step_absa already writes into
    # master.xlsx's topic_sentiment_pairs column. Used only for merging backfilled
    # rows back into the live file; topic_sentiments above is the clean schema.
    df["topic_sentiment_pairs_backfilled"] = [json.dumps(p, ensure_ascii=False) for p in out_final]
    df["topic_provenance"] = [json.dumps(p, ensure_ascii=False) for p in out_prov]
    df["topic_names"] = ["; ".join(p["topic"] for p in ps) for ps in out_real]
    df["n_topics"] = [len(p) for p in out_real]                 # REAL topics only (overall entry excluded)
    df["topic_source"] = ["+".join(sorted({p["source"].split(":")[0] for p in ps})) or "none" for ps in out_real]
    prim = df["primary_tag_group"]
    df["lexicon_agrees_primary"] = [(g not in NON_TOPIC_GROUPS) and (g in {t for t in PRIORITY if hit_lists[t][i]})
                                    for i, g in enumerate(prim)]
    return df


def add_review_tier(df: pd.DataFrame) -> pd.DataFrame:
    c = df["svm_confidence"]
    tier = np.select(
        [c >= CONF_HIGH, c >= CONF_REVIEW_BELOW,
         (df["prediction_method"] == "fallback"),      # fallback is itself keyword-based: not independent evidence
         (c < CONF_REVIEW_BELOW) & df["lexicon_agrees_primary"]],
        ["high", "medium", "keyword_only", "low_but_lexicon_agrees"], default="low_unconfirmed")
    df["review_tier"] = tier
    df.loc[df["is_vacuous"] | df["is_test"], "review_tier"] = "n/a"
    df["needs_review"] = df["review_tier"].isin(["low_unconfirmed", "keyword_only"]) | df["rating_sentiment_conflict"]
    df["backfill_version"] = BACKFILL_VERSION
    return df


# ══════════════════════════════════════════════════════════════════════════
# 7. Target record schema + validation
# ══════════════════════════════════════════════════════════════════════════
def to_record(r) -> dict:
    """{ respondent_id:int, rating:int, feedback_clean:str, user_type:str|null,
         topic_sentiments:[{topic, detail, sentiment, confidence}] }
    feedback_clean here is the PII-REDACTED text. detail is the specific
    word/phrase/aspect behind the topic match (e.g. "trees", "too
    complex") — null when nothing more specific than the topic itself
    was pinned down."""
    ut = r["user_type"]
    return {
        "respondent_id": int(r[COL_ID]),
        "rating": int(r[COL_RATING]),
        "feedback_clean": r["feedback_redacted"],
        "user_type": ut if isinstance(ut, str) else None,
        "topic_sentiments": json.loads(r["topic_sentiments"]),
    }


def validate_records(records: list[dict]) -> list[str]:
    errs, ok_sent = [], {"positive", "negative", "neutral"}
    keys = ["respondent_id", "rating", "feedback_clean", "user_type", "topic_sentiments"]
    for rec in records:
        rid = rec.get("respondent_id")
        if list(rec) != keys:
            errs.append(f"{rid}: wrong keys/order")
        if not isinstance(rid, int) or not isinstance(rec["rating"], int) or not 1 <= rec["rating"] <= 5:
            errs.append(f"{rid}: bad id/rating")
        for t in rec["topic_sentiments"]:
            c = t.get("confidence")
            if list(t) != ["topic", "detail", "sentiment", "confidence"] or t["sentiment"] not in ok_sent \
                    or (c is not None and not 0 <= c <= 1) \
                    or (t.get("detail") is not None and not isinstance(t["detail"], str)):
                errs.append(f"{rid}: bad topic entry {t}")
    return errs


# ══════════════════════════════════════════════════════════════════════════
# 8. Report
# ══════════════════════════════════════════════════════════════════════════
def report(before: pd.DataFrame, df: pd.DataFrame, n_schema_errors: int) -> str:
    n = len(df)
    old_topic = before["_existing_topics"].map(len)
    entries = df["topic_sentiments"].map(json.loads)
    flat = pd.DataFrame([e for es in entries for e in es])
    lines = [f"# Backfill report ({BACKFILL_VERSION})", f"Rows: {n:,}", ""]
    lines += ["## Topic coverage (real topics; the overall-experience entry is not counted)",
              "| Metric | Before | After |", "|---|---|---|",
              f"| Rows with ≥1 real topic | {(old_topic > 0).mean():.1%} | {(df.n_topics > 0).mean():.1%} |",
              f"| Rows with ≥2 topics | {(old_topic > 1).mean():.1%} | {(df.n_topics > 1).mean():.1%} |", ""]
    substantive = df[(df.words >= 5) & ~df.is_test & ~df.is_vacuous]
    lines += [f"Substantive comments (≥5 words): {len(substantive):,}. "
              f"With no real topic after backfill: {(substantive.n_topics == 0).mean():.1%}", ""]
    src = df["topic_source"].value_counts()
    lines += ["## Where real topics came from", "| Source | Rows |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in src.items()] + [""]
    per_topic = flat["topic"].value_counts()
    lines += ["## Entries per topic (includes the two overall-experience entries)", "| Topic | Entries |", "|---|---|"] + \
             [f"| {k} | {v:,} |" for k, v in per_topic.items()] + [""]
    lines += ["## Schema checks",
              f"- Records: {n:,}   Entries: {len(flat):,}   Schema errors: {n_schema_errors}",
              f"- Entries with confidence = null (no model score exists): {flat['confidence'].isna().mean():.1%}",
              f"- Rows with an empty topic_sentiments list: {(entries.map(len) == 0).sum():,}", ""]
    tiers = df["review_tier"].value_counts()
    lines += ["## Review queue", "| Tier | Rows | Share |", "|---|---|---|"] + \
             [f"| {k} | {v:,} | {v / n:.1%} |" for k, v in tiers.items()]
    lines += ["", f"needs_review = {df.needs_review.sum():,} ({df.needs_review.mean():.1%}) "
                  f"(old rule `<0.40` = {(df.svm_confidence < CONF_REVIEW_BELOW).mean():.1%})", ""]
    lines += ["## Data quality", f"- PII-flagged rows: {int(df.pii_flag.sum())} (redacted in feedback_clean of the JSONL)",
              f"- Test rows: {int(df.is_test.sum())}   Vacuous rows: {int(df.is_vacuous.sum())}",
              f"- One-word comments given a sentiment: {int((df.words <= 1).sum())}",
              f"- Rating/sentiment conflicts flagged: {int(df.rating_sentiment_conflict.sum())}",
              f"- user_type populated (any of the 19 categories): {int(df.user_type.notna().sum())} rows"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
def run(in_path: Path, out_path: Path | None, jsonl_path: Path | None) -> str:
    df = load_master(in_path)
    df = add_group_columns(df)
    before = df[["_existing_topics"]].copy()
    df = add_quality_flags(df)
    df = add_overall_and_user_type(df)
    df = build_topics(df)
    df = add_review_tier(df)

    records = [to_record(r) for _, r in df.iterrows()]
    errors = validate_records(records)
    rep = report(before, df, len(errors))
    if errors:
        rep += "\n\n## Schema errors (first 10)\n" + "\n".join(errors[:10])

    if out_path:
        out = df.drop(columns=[c for c in df.columns if c.startswith("_")] + ["topic_sentiment_pairs"])
        out.to_excel(out_path, index=False)              # Respondent ID stays text in the workbook
        out_path.with_suffix(".report.md").write_text(rep, encoding="utf-8")
    if jsonl_path:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="workbook with all helper columns")
    ap.add_argument("--jsonl", type=Path, default=None, help="one record per line in the target schema")
    ap.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    a = ap.parse_args()
    if a.dry_run:
        print(run(a.inp, None, None))
    else:
        print(run(a.inp, a.out, a.jsonl))