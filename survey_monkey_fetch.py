# survey_monkey_fetch.py
# ── Direct SurveyMonkey API ingestion ──────────────────────────────────────
# Pulls responses straight from SurveyMonkey instead of a manual Excel
# export. Requires SURVEYMONKEY_ACCESS_TOKEN and SURVEYMONKEY_SURVEY_ID to
# be set (via .env locally, or environment/secrets in production).
#
# Usage:
#   import survey_monkey_fetch as smf
#   df = smf.fetch_survey_as_dataframe()

import logging
import os
import time

import pandas as pd
import requests

import pipeline_config as cfg

log = logging.getLogger(__name__)

API_BASE = "https://api.surveymonkey.com/v3"

# How much earlier than the last checkpoint to actually ask the API for,
# as a safety margin against clock skew / API write latency near the
# boundary. Re-fetching a small overlap window is cheap (a handful of
# responses, not the whole history) and any duplicates are caught by the
# existing processed_ids dedup in run_pipeline.py — so it's safe to be
# generous here rather than risk silently missing a response.
CHECKPOINT_OVERLAP_MINUTES = 60


def load_checkpoint(path):
    """
    Read the ISO-8601 'last successful fetch' timestamp from a checkpoint
    file. Returns None if the file doesn't exist yet (e.g. first-ever
    run) — callers should treat None as "do a full fetch".
    """
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        ts = f.read().strip()
    return ts or None


def save_checkpoint(path, timestamp_iso):
    """
    Write the 'last successful fetch' timestamp. Callers must only call
    this AFTER a run has fully succeeded — advancing the checkpoint on a
    failed run would mean any responses in that window are never
    fetched again.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(timestamp_iso)
    log.info(f"Checkpoint saved: {timestamp_iso}")


def checkpoint_with_overlap(checkpoint_iso):
    """
    Given a checkpoint timestamp (or None), return the timestamp to
    actually pass as start_created_at — pulled back by
    CHECKPOINT_OVERLAP_MINUTES as a safety margin. Returns None
    unchanged (meaning: no checkpoint yet, do a full fetch).

    Expects the SurveyMonkey DateString format used elsewhere in this
    module, e.g. "2026-07-15T08:31:15+00:00".
    """
    if not checkpoint_iso:
        return None
    from datetime import datetime, timedelta
    dt = datetime.strptime(checkpoint_iso, "%Y-%m-%dT%H:%M:%S+00:00")
    dt -= timedelta(minutes=CHECKPOINT_OVERLAP_MINUTES)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def checkpoint_age_hours(checkpoint_iso):
    """
    How many hours old a checkpoint timestamp is. Returns None if
    checkpoint_iso is falsy (no checkpoint yet). Used by consumers of
    the local survey cache to warn if refresh_survey_cache.py appears
    to have silently stopped running, rather than failing outright —
    stale data is still usable, just worth flagging.
    """
    if not checkpoint_iso:
        return None
    from datetime import datetime, timezone
    dt = datetime.strptime(checkpoint_iso, "%Y-%m-%dT%H:%M:%S+00:00")
    dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 3600


class RateLimitExceeded(Exception):
    """
    Raised when SurveyMonkey's daily or per-minute rate limit is hit
    (HTTP 429). Deliberately a distinct exception type — callers (e.g.
    run_pipeline.py) need to tell "transient, retryable rate limit" apart
    from other failures like auth errors or malformed requests, so they
    can abort cleanly instead of silently falling back to stale data.
    """
    pass


def log_rate_limit_status(response, warn_threshold=50):
    """
    Logs SurveyMonkey's rate-limit headers after every API call, so we
    find out quota is running low from the logs — not from a failed
    scheduled run. SurveyMonkey enforces both a per-minute limit and a
    shared daily limit (500/day at time of writing) across the whole
    app, so the daily one is usually the one that bites.
    """
    day_remaining = response.headers.get("X-Ratelimit-App-Global-Day-Remaining")
    day_limit = response.headers.get("X-Ratelimit-App-Global-Day-Limit")
    day_reset = response.headers.get("X-Ratelimit-App-Global-Day-Reset")

    if day_remaining is None:
        return  # headers not present on this response, nothing to log

    remaining = int(day_remaining)
    limit = int(day_limit) if day_limit else "?"

    if remaining == 0:
        reset_hrs = round(int(day_reset) / 3600, 1) if day_reset else "?"
        log.error(
            f"SurveyMonkey daily quota EXHAUSTED (0/{limit}). "
            f"Resets in ~{reset_hrs}h."
        )
    elif remaining < warn_threshold:
        log.warning(
            f"SurveyMonkey daily quota running low: {remaining}/{limit} requests left today."
        )
    else:
        log.info(f"SurveyMonkey daily quota: {remaining}/{limit} remaining.")


def _get(url, headers, params=None, timeout=30):
    """
    Thin wrapper around requests.get that logs rate-limit headers on
    every call and raises RateLimitExceeded (instead of a generic
    HTTPError) on a 429, so callers can handle that case specifically.
    """
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    log_rate_limit_status(resp)

    if resp.status_code == 429:
        try:
            message = resp.json().get("error", {}).get("message", "Rate limit reached")
        except ValueError:
            message = "Rate limit reached"
        log.warning(f"SurveyMonkey rate limit hit: {message}")
        raise RateLimitExceeded(message)

    resp.raise_for_status()
    return resp


def _headers():
    if not cfg.SM_ACCESS_TOKEN:
        raise EnvironmentError(
            "SURVEYMONKEY_ACCESS_TOKEN is not set. Check your .env file "
            "(or environment/secrets in production)."
        )
    return {"Authorization": f"Bearer {cfg.SM_ACCESS_TOKEN}"}


def _survey_id(survey_id=None):
    survey_id = survey_id or cfg.SM_SURVEY_ID
    if not survey_id:
        raise EnvironmentError(
            "SURVEYMONKEY_SURVEY_ID is not set. Check your .env file "
            "(or environment/secrets in production)."
        )
    return survey_id


def get_question_metadata(survey_id=None):
    """
    For each question, map its id to:
      - "heading": the question text (e.g. "Rate your experience")
      - "choices": {choice_id: choice_label} for choice-based questions
                   (e.g. {"123": "Excellent", "456": "Poor"})

    Needed because the bulk responses endpoint often returns only a raw
    choice_id for single/multi-choice questions, not the human-readable
    label — without this map, "Rate your experience" answers come back
    as huge internal SurveyMonkey IDs instead of "Excellent"/"Poor".
    """
    survey_id = _survey_id(survey_id)
    url = f"{API_BASE}/surveys/{survey_id}/details"

    resp = _get(url, headers=_headers())
    details = resp.json()

    question_meta = {}
    for page in details.get("pages", []):
        for question in page.get("questions", []):
            qid = question["id"]
            headings = question.get("headings", [])
            heading = headings[0].get("heading", qid) if headings else qid

            choices = {}
            for choice in question.get("answers", {}).get("choices", []):
                choices[str(choice["id"])] = {
                    "text": choice.get("text", ""),
                    "weight": choice.get("weight"),
                }

            question_meta[qid] = {"heading": heading, "choices": choices}

    log.info(f"Mapped {len(question_meta)} questions for survey {survey_id}")
    return question_meta


def get_question_map(survey_id=None):
    """Backwards-compatible: heading-only map (qid -> heading text)."""
    meta = get_question_metadata(survey_id)
    return {qid: m["heading"] for qid, m in meta.items()}


def fetch_all_responses(survey_id=None, per_page=None, start_created_at=None):
    """
    Pull completed responses for a survey, paginating through the bulk
    responses endpoint until there are no more pages.

    If start_created_at is given (an ISO-8601 timestamp string), only
    responses created at or after that time are returned — this is what
    makes incremental fetching possible instead of re-pulling the entire
    survey history (currently ~12,800+ responses, ~128 API calls) on
    every single run. Pass None for a full fetch (e.g. first-ever run,
    or no checkpoint file found yet).
    """
    survey_id = _survey_id(survey_id)
    per_page = per_page or cfg.SM_PER_PAGE

    url = f"{API_BASE}/surveys/{survey_id}/responses/bulk"
    params = {"per_page": per_page, "page": 1, "status": "completed"}
    if start_created_at:
        params["start_created_at"] = start_created_at
        log.info(f"Incremental fetch: responses since {start_created_at}")
    else:
        log.info("Full fetch: no checkpoint set, pulling entire survey history")

    all_responses = []
    while url:
        resp = _get(url, headers=_headers(), params=params)
        data = resp.json()

        all_responses.extend(data.get("data", []))
        log.info(f"  Fetched {len(all_responses)} responses so far...")

        # SurveyMonkey returns a "next" link with query params already
        # baked in — once we follow it, don't pass params again.
        url = data.get("links", {}).get("next")
        params = None

        if url:
            time.sleep(0.3)  # be polite to the API rate limit

    log.info(f"Retrieved {len(all_responses)} total responses for survey {survey_id}")
    return all_responses


def responses_to_dataframe(raw_responses, question_meta):
    """
    Flatten SurveyMonkey's nested response JSON into one row per
    respondent, with columns named after the question headings —
    matching the shape of the existing manual Excel export.
    """
    rows = []
    for r in raw_responses:
        row = {
            "Respondent ID": str(r.get("id")),
            "Start Date": r.get("date_created"),
        }
        for page in r.get("pages", []):
            for q in page.get("questions", []):
                meta = question_meta.get(q["id"], {"heading": q["id"], "choices": {}})
                heading = meta["heading"]
                choices = meta["choices"]
                answers = q.get("answers", [])
                if not answers:
                    continue

                first = answers[0]
                if "text" in first:
                    # Open text (essay) answers
                    row[heading] = first["text"]
                elif "choice_id" in first:
                    # Single/multi-choice answers (e.g. rating) — resolve
                    # the choice_id to its real label via the survey's
                    # choice map. Some scales (e.g. star/smiley ratings)
                    # only label the endpoints ("Excellent"/"Poor") and
                    # leave middle choices with blank text — for those,
                    # fall back to the choice's numeric "weight" instead
                    # of losing the value entirely.
                    cid = str(first["choice_id"])
                    choice_info = choices.get(cid, {})
                    text = choice_info.get("text", "")
                    weight = choice_info.get("weight")
                    if text:
                        row[heading] = text
                    elif weight is not None:
                        row[heading] = weight
                    else:
                        row[heading] = first.get("text") or cid

        rows.append(row)

    df = pd.DataFrame(rows)
    log.info(f"Flattened to DataFrame: {df.shape[0]} rows, columns: {df.columns.tolist()}")
    return df


def fetch_survey_as_dataframe(survey_id=None, start_created_at=None):
    """
    Main entry point. Returns a DataFrame in the same shape
    run_pipeline.py expects from the manual Excel export — NOTE: unlike
    the Excel export, this data has no spurious extra header row, so
    downstream code must not apply the usual .iloc[1:] drop to
    API-sourced data.

    start_created_at: pass an ISO-8601 timestamp (typically read from a
    checkpoint file via load_checkpoint()) to fetch only responses since
    then. Pass None for a full fetch.
    """
    survey_id = _survey_id(survey_id)
    log.info(f"Fetching question metadata for survey {survey_id}...")
    question_meta = get_question_metadata(survey_id)

    log.info(f"Fetching responses for survey {survey_id}...")
    raw = fetch_all_responses(survey_id, start_created_at=start_created_at)

    return responses_to_dataframe(raw, question_meta)