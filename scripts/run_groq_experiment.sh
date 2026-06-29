#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt >/dev/null

if [[ -z "${GROQ_API_KEY:-}" && -z "${GROQ_API_KEYS:-}" ]]; then
  if [[ -f .env ]] && grep -Eq '^GROQ_API_KEYS?=' .env; then
    echo "Using Groq API key(s) from .env (keys not printed)."
  else
    echo "Groq API key is not set. Add GROQ_API_KEY=... or GROQ_API_KEYS=key1,key2 to .env, or export it first." >&2
    exit 1
  fi
fi

if [[ "$#" -eq 0 ]]; then
  set -- --provider groq --scale small --ablation --out results_live_rerun.json
fi

echo "Running: evaluate.py $*"
.venv/bin/python evaluate.py "$@"
