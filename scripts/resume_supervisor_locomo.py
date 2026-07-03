#!/usr/bin/env python3
"""Unattended auto-resume supervisor for the full LoCoMo sweep (n~450).

Same design as resume_supervisor.py (see that file for rationale), but
targeting the LoCoMo scale-up run that powers the retrieval-vs-full-context
paired test (power analysis: ~360-430 questions needed).

Launch detached so it outlives the shell:
  setsid nohup .venv/bin/python scripts/resume_supervisor_locomo.py \
      >> logs/resume_supervisor_locomo.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_locomo_n450.json"
DONE_MARKER = ROOT / "logs" / "resume_supervisor_locomo.DONE"
RUN_LOG = ROOT / "logs" / "locomo_n450_run.log"
TARGET_CASES = 10            # ten LoCoMo conversations, 45 QA each (category-balanced)
POLL_S = 1200
SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"

RESUME_CMD = [
    str(ROOT / ".venv/bin/python"), str(ROOT / "evaluate.py"),
    "--provider", "groq",
    "--dataset", "locomo",
    "--data-path", "data/locomo10.json",
    "--max-qa-per-conv", "45",
    "--synth-model", SCOUT,
    "--judge", "llm", "--judge-model", "llama-3.3-70b-versatile",
    "--segment-size", "12", "--slot-memory", "--rag",
    "--seed", "7",
    "--resume", "--out", str(OUT.name),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}", flush=True)


def evaluate_running() -> bool:
    try:
        me = os.getpid()
        out = subprocess.run(["pgrep", "-af", "evaluate.py"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            pid = int(line.split(None, 1)[0])
            if pid != me and "evaluate.py" in line and OUT.name in line:
                return True
    except Exception:
        pass
    return False


def any_evaluate_running() -> bool:
    """Defer to ANY evaluate.py run (they share the same Groq key pool)."""
    try:
        me = os.getpid()
        out = subprocess.run(["pgrep", "-af", "evaluate.py"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            pid = int(line.split(None, 1)[0])
            if pid != me and "evaluate.py" in line:
                return True
    except Exception:
        pass
    return False


def n_cases_done() -> int:
    try:
        d = json.loads(OUT.read_text())
        return len(d.get("case_metrics", []))
    except Exception:
        return 0


MIN_READY_KEYS = 6


def budget_available() -> int:
    try:
        import evaluate
        import openai
        evaluate.load_dotenv_if_present()
        keys = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
        big = "User: " + ("I attended the Holiday Market and bought an iPhone 13 Pro. " * 400)
        ok = 0
        for i, key in enumerate(keys):
            remaining = len(keys) - i
            if ok + remaining < MIN_READY_KEYS:
                break
            cli = openai.OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1",
                                max_retries=0, timeout=30)
            try:
                cli.chat.completions.create(model=SCOUT, max_tokens=8,
                                            messages=[{"role": "user", "content": big}])
                ok += 1
                if ok >= MIN_READY_KEYS:
                    break
            except openai.RateLimitError:
                continue
            except Exception:
                continue
        return ok
    except Exception as e:
        log(f"budget probe error: {type(e).__name__}: {e}")
        return 0


LOCK = ROOT / "logs" / "resume_supervisor_locomo.lock"


def acquire_lock() -> bool:
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            os.kill(old, 0)
            return False
        except (ValueError, ProcessLookupError):
            pass
        except PermissionError:
            return False
    LOCK.write_text(str(os.getpid()))
    return True


def main() -> None:
    if not acquire_lock():
        log(f"another supervisor is already running (see {LOCK}); exiting")
        return
    log(f"supervisor starting; target={TARGET_CASES} cases; poll={POLL_S}s")
    os.environ.setdefault("GROQ_CALL_DEADLINE_S", "300")
    os.environ.setdefault("GROQ_KEY_COOLDOWN_S", "12")
    # Unlike the LME supervisor, we attempt an immediate run first (the user
    # believes today's budget has reset); if it makes no progress we fall back
    # to waiting for the next 00:00 UTC boundary.
    next_reset = datetime.now(timezone.utc)   # attempt immediately
    forced = os.environ.get("SUPERVISOR_FORCE_RESET_UTC")
    if forced:
        next_reset = datetime.fromisoformat(forced)
    while True:
        done = n_cases_done()
        log(f"progress: {done}/{TARGET_CASES} cases complete")
        if done >= TARGET_CASES:
            DONE_MARKER.write_text(f"completed {done} cases at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
            log("TARGET reached -- sweep complete. Exiting.")
            LOCK.unlink(missing_ok=True)
            return
        if any_evaluate_running():
            log(f"another evaluate.py run is active on the shared key pool -- "
                f"deferring (recheck in 120s)")
            time.sleep(120)
            continue
        now = datetime.now(timezone.utc)
        if now < next_reset:
            wait = min(POLL_S, max(30, (next_reset - now).total_seconds()))
            log(f"before TPD reset boundary {next_reset.isoformat()} "
                f"({(next_reset - now).total_seconds()/3600:.1f}h away) -- idle {wait:.0f}s")
            time.sleep(wait)
            continue
        ready = budget_available()
        if ready >= MIN_READY_KEYS:
            log(f"budget confirmed ({ready}/{MIN_READY_KEYS} keys) -- launching resume run")
            with open(RUN_LOG, "a") as lf:
                lf.write(f"\n===== supervised resume {time.strftime('%Y-%m-%d %H:%M:%S %Z')} =====\n")
                subprocess.run(RESUME_CMD, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
            after = n_cases_done()
            gained = after - done
            log(f"resume run returned: {gained} new cases ({after}/{TARGET_CASES} total)")
            if gained > 0:
                time.sleep(30)
            else:
                next_reset = (datetime.now(timezone.utc)
                              .replace(hour=0, minute=1, second=0, microsecond=0)
                              + timedelta(days=1))
                log(f"no progress; next attempt after {next_reset.isoformat()}")
        else:
            log(f"only {ready}/{MIN_READY_KEYS} keys ready -- rechecking in {POLL_S}s")
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()
