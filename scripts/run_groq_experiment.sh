#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt >/dev/null

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  if [[ -f .env ]] && grep -q '^GROQ_API_KEY=' .env; then
    echo "Using GROQ_API_KEY from .env (key not printed)."
  else
    echo "GROQ_API_KEY is not set. Add GROQ_API_KEY=... to .env or export it first." >&2
    exit 1
  fi
fi

if [[ "$#" -eq 0 ]]; then
  set -- --provider groq --scale small --ablation --out results_live_rerun.json
fi

echo "Running: evaluate.py $*"
.venv/bin/python evaluate.py "$@"
