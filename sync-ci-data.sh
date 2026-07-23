#!/bin/bash
# sync-ci-data.sh
# ── Pull the latest CI-produced data files, safely ─────────────────────────
# data/metrics.db, data/master.xlsx, data/processed_ids.csv, and
# outputs/latest/dashboard.html are marked --skip-worktree locally, so
# local pipeline runs never create merge conflicts with the CI jobs that
# actually own these files. That also means a normal `git pull` silently
# does NOT update your local copies of them.
#
# Run this whenever you want your local copies to match what's actually
# on GitHub (e.g. before comparing local vs. deployed dashboard output).
#
# Usage: bash sync-ci-data.sh

set -e

CI_FILES="data/metrics.db data/master.xlsx data/processed_ids.csv outputs/latest/dashboard.html"

echo "=== Temporarily un-skipping CI-owned files ==="
git update-index --no-skip-worktree $CI_FILES

echo ""
echo "=== Pulling latest from origin/main ==="
git pull --rebase origin main

echo ""
echo "=== Re-applying skip-worktree ==="
git update-index --skip-worktree $CI_FILES

echo ""
echo "=== Done — local copies now match GitHub, and are protected again ==="
echo "Note: if you had uncommitted local pipeline-run output in these files"
echo "before running this, it's now been overwritten by the pulled version."