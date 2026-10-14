"""
error_patterns.py — one canonical label per kind of error found in comments.

Every wording of the same problem ("keeps crashing", "kept crashing",
"portal crashed", "site crash") maps to ONE label ("Portal crashed"), so the
Errors page counts a problem once instead of once per spelling.

To add a new wording: add a regex to the matching label's list.
To add a new kind of error: add a new label with its regex list.

IMPORTANT: labels must NOT contain commas. entities_errors is saved as one
", "-joined string and split on ", " later, so a comma inside a label would
cut it in two (the same bug as the topic names).
"""
import re

# Things that can fail. Add your own words as you see new cases.
ERROR_SUBJECTS = (
    "system|portal|page|site|website|form|upload|uploader|payment|button|link|map|tool|"
    "postcode|address|file|document|login|password|plan|drawing|app|application|service|"
    "screen|field|boundary"
)
SYSTEM_WORDS = "system|portal|page|site|website|app"

# NOTE: the cleaned comment text has apostrophes removed ("cant", "didnt"), so every
# contraction below accepts the apostrophe or not.
# "I can't / couldn't / was unable to ..." — the person says they could not do something
_CANT = r"\b(?:cannot|can['’]?t|couldn['’]?t|could not|unable to|not able to)"
# "The <thing> didn't / won't ..." — needs a named thing, so "I did not work out where to
# click" or "it did not work" (no subject) don't count
_FAILS = (r"\b(?:" + ERROR_SUBJECTS + r")\s+(?:\w+\s+)?"
          r"(?:won['’]?t|will not|wouldn['’]?t|would not|doesn['’]?t|does not|didn['’]?t|did not)")
# "The system/portal/page/site ..." with an optional "has/kept/keeps"
_SYS = r"\b(?:" + SYSTEM_WORDS + r")\s+(?:has\s+|kept\s+|keeps\s+)?"


def _action(verbs):
    """Both ways of saying an action failed: the person couldn't, or the thing didn't."""
    v = r"(?:" + verbs + r")\b"
    return [_CANT + r"\s+" + v, _FAILS + r"\s+" + v]


ERROR_LABELS = {
    # ── System problems ────────────────────────────────────────────────────
    "Portal crashed": [
        _SYS + r"(?:crash(?:ed|es|ing)?|went down)\b",
        r"\b(?:keeps?|kept)\s+crashing\b",
    ],
    "Freezing or timing out": [
        _SYS + r"(?:froze|freez(?:es|ing)|stall(?:ed|s|ing)|timed out|times out)\b",
        r"\b(?:keeps?|kept)\s+(?:freezing|stalling|timing out)\b",
    ],
    "Logged out unexpectedly": [
        r"\b(?:keeps?|kept)\s+logging (?:me )?out\b",
    ],
    "Lost or deleted work": [
        r"\b(?:lost|deleted|removed)\s+(?:my|all|the)?\s*"
        r"(?:data|work|information|progress|application|documents?)\b",
    ],

    # ── Something could not be done ────────────────────────────────────────
    "Could not upload": _action("upload") + [
        r"\b(?:file|document|pdf|excel)\s+(?:not accepted|rejected|failed|not loading)\b",
    ],
    "Could not submit": _action("submit"),
    "Could not complete": [_CANT + r"\s+complete\b"],
    "Could not save": _action("save"),
    "Could not load or open": _action("load|open|access"),
    "Payment failed or could not pay": [
        _CANT + r"\s+pay\b",
        r"\b(?:payment|card)\s+(?:failed|declined|not (?:working|going through|processing))\b",
    ],

    # ── Something was refused ──────────────────────────────────────────────
    "Not accepted or recognised": [_FAILS + r"\s+(?:accept|recogni[sz]e)\b"],
    "Address or postcode not recognised": [
        r"\b(?:postcode|post code|address)\s+"
        r"(?:not (?:found|recognised|accepted|working)|rejected|invalid)\b",
    ],
    "Feature did not work": [
        _FAILS + r"\s+work\b(?!\s+(?:anywhere|nearly|out\b|as\s+(?:well|good|easily|smoothly)))",
    ],

    # ── Limits and rules ───────────────────────────────────────────────────
    "File size limit": [
        r"\b(?:10\s?mb|size limit|upload limit|file limit|size restriction)\b",
        r"\bfile size\b.{0,40}\b(?:limit|restriction|too (?:large|big)|exceed\w*|allowance)\b",
        r"\b(?:limit|restriction)s?\b.{0,20}\bfile size\b",
    ],
    "Character restrictions": [
        r"\b(?:square brackets|special characters?|permitted characters?)\b",
    ],
}

assert not any("," in label for label in ERROR_LABELS), "Error labels must not contain commas"

_COMPILED = {label: [re.compile(p) for p in pats] for label, pats in ERROR_LABELS.items()}


def extract_errors(text):
    """Return the error labels found in one comment (each label at most once)."""
    tl = str(text).lower()
    return [label for label, pats in _COMPILED.items() if any(p.search(tl) for p in pats)]