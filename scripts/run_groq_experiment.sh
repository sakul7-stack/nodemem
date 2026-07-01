#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt >/dev/null

if [[ -z "${NARA_API_KEY:-}" && -z "${NARA_API_KEYS:-}" && -z "${BYNARA_API_KEY:-}" ]]; then
  if [[ -f .env ]] && grep -Eq '^(NARA_API_KEYS?|BYNARA_API_KEY)=' .env; then
    echo "Using Nara Router API key(s) from .env (keys not printed)."
  else
    echo "Nara Router API key is not set. Add NARA_API_KEY=... or NARA_API_KEYS=key1,key2 to .env, or export it first." >&2
    exit 1
  fi
fi

if [[ "$#" -eq 0 ]]; then
  set -- --provider nara --scale small --ablation --out results_nara_rerun.json
fi

echo "Running: evaluate.py $*"
.venv/bin/python evaluate.py "$@"
