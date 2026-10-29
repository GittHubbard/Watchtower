#!/usr/bin/env bash
# Mirrors .github/workflows/eod.yml for manual runs from a Codespace.
set -euo pipefail

BRANCH=main

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree is dirty. Commit or stash changes before running." >&2
    git status --short >&2
    exit 1
fi

git fetch origin
git pull --rebase origin "$BRANCH"

if [ ! -x .venv/bin/python ]; then
    echo "No usable venv interpreter — rebuilding."
    rm -rf .venv
    python3 -m venv .venv
fi

if .venv/bin/python -c 'import numpy, pandas, scipy, pyarrow, matplotlib, yfinance, cloudscraper, bs4, yaml, requests' 2>/dev/null; then
    echo "Cached venv is usable — skipping install."
else
    echo "Dependency import check failed — installing."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip install -e .
fi

.venv/bin/python -m gamma_workflow.run_eod

git add data/

if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

DATE=$(date -u +%Y-%m-%d)
MESSAGE_SUFFIX="${1:-EOD pipeline}"
git commit -m "Manual run ${DATE}: ${MESSAGE_SUFFIX}"

git pull --rebase origin "$BRANCH"
git push
