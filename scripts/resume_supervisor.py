#!/usr/bin/env python3
"""Unattended auto-resume supervisor for the remaining LongMemEval sweep.

Groq's free tier caps the scout synthesizer at 500K tokens/day/key. Today's
budget is spent, so the sweep must continue after the daily reset (00:00 UTC).
This supervisor runs independently of any Claude/interactive session:

  * every POLL_S seconds it checks whether the scout daily budget is back by
    sending one realistically-sized request (a rejected 429 costs no tokens,
    so polling while exhausted is nearly free);
  * when budget is available it invokes the exact --resume command, which
    processes as many cases as fit, checkpoints each to the results JSON, and
    stops cleanly on BudgetExhaustedError;
  * it exits once the results file holds TARGET_CASES cases, writing a DONE
    marker so a later step can detect completion.

Launch detached so it outlives the shell:
  setsid nohup .venv/bin/python scripts/resume_supervisor.py \
      >> logs/resume_supervisor.log 2>&1 &
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
sys.path.insert(0, str(ROOT))   # so `import evaluate` works when run from scripts/
OUT = ROOT / "results_lme_remaining.json"
DONE_MARKER = ROOT / "logs" / "resume_supervisor.DONE"
RUN_LOG = ROOT / "logs" / "lme_remaining_run.log"
TARGET_CASES = 278           # temporal + multi_session + abstention (oracle)
POLL_S = 1200                # 20 min between checks
SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"

RESUME_CMD = [
    str(ROOT / ".venv/bin/python"), str(ROOT / "evaluate.py"),
    "--provider", "groq",
    "--dataset", "longmemeval",
    "--data-path", "data/longmemeval_oracle.json",
    "--lme-categories", "temporal", "multi_session", "abstention",
    "--synth-model", SCOUT,
    "--judge", "llm", "--judge-model", "llama-3.3-70b-versatile",
    "--segment-size", "12", "--slot-memory", "--rag",
    "--resume", "--out", "results_lme_remaining.json",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}", flush=True)


def evaluate_running() -> bool:
    """True if some other evaluate.py writer is already touching the shared
    checkpoint. The per-case save is NOT atomic (it truncates then rewrites),
    so two concurrent --resume writers would corrupt results_lme_remaining.json.
    The supervisor must defer to any run already in flight."""
    try:
        me = os.getpid()
        out = subprocess.run(["pgrep", "-af", "evaluate.py"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            pid = int(line.split(None, 1)[0])
            if pid != me and "evaluate.py" in line and "results_lme_remaining.json" in line:
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


MIN_READY_KEYS = 6   # majority of 12 -> a genuine daily reset, not a few crumbs


def budget_available() -> int:
    """Return how many keys accept a ~6K-token scout request. A handful of keys
    passing just means leftover crumbs (which wall a real run almost instantly);
    a majority passing means the daily TPD actually reset. Early-exits once the
    threshold is reached, or once too many keys have failed to reach it, to
    avoid burning fresh budget on the probe."""
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
                break                # can't reach threshold even if all remaining pass
            cli = openai.OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1",
                                max_retries=0, timeout=30)
            try:
                cli.chat.completions.create(model=SCOUT, max_tokens=8,
                                            messages=[{"role": "user", "content": big}])
                ok += 1
                if ok >= MIN_READY_KEYS:
                    break            # enough signal; stop consuming budget
            except openai.RateLimitError:
                continue
            except Exception:
                continue
        return ok
    except Exception as e:
        log(f"budget probe error: {type(e).__name__}: {e}")
        return 0


LOCK = ROOT / "logs" / "resume_supervisor.lock"


def acquire_lock() -> bool:
    """Refuse to start a second supervisor (concurrent runs would corrupt the
    shared results checkpoint). Stale locks from dead PIDs are reclaimed."""
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            os.kill(old, 0)          # raises if the PID is gone
            return False             # a live supervisor already owns the lock
        except (ValueError, ProcessLookupError):
            pass                     # stale lock -- fall through and reclaim
        except PermissionError:
            return False
    LOCK.write_text(str(os.getpid()))
    return True


def main() -> None:
    if not acquire_lock():
        log(f"another supervisor is already running (see {LOCK}); exiting")
        return
    log(f"supervisor starting; target={TARGET_CASES} cases; poll={POLL_S}s")
    # Ensure the child run stops itself on a hard wall rather than hanging.
    os.environ.setdefault("GROQ_CALL_DEADLINE_S", "300")
    os.environ.setdefault("GROQ_KEY_COOLDOWN_S", "12")
    # Groq's free-tier TPD resets at 00:00 UTC. Probing/attempting runs before
    # that only burns tokens (a successful ~6K probe counts against TPD, and on
    # a rolling window that would *delay* the reset). So we sit idle until the
    # next reset boundary, then attempt runs. If a post-reset run still walls
    # (0 progress -- e.g. that day's fresh budget got used up), we wait for the
    # following boundary. An early boundary can be forced for testing.
    next_reset = datetime.now(timezone.utc).replace(hour=0, minute=1, second=0, microsecond=0) + timedelta(days=1)
    forced = os.environ.get("SUPERVISOR_FORCE_RESET_UTC")  # ISO string, optional
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
        if evaluate_running():
            log(f"another evaluate.py writer is active on the checkpoint -- "
                f"deferring to it (recheck in {POLL_S}s)")
            time.sleep(POLL_S)
            continue
        now = datetime.now(timezone.utc)
        if now < next_reset:
            wait = min(POLL_S, max(30, (next_reset - now).total_seconds()))
            log(f"before TPD reset boundary {next_reset.isoformat()} "
                f"({(next_reset - now).total_seconds()/3600:.1f}h away) -- idle {wait:.0f}s")
            time.sleep(wait)
            continue
        # Past the reset boundary: confirm real budget, then run.
        ready = budget_available()
        if ready >= MIN_READY_KEYS:
            log(f"post-reset budget confirmed ({ready}/{MIN_READY_KEYS} keys) -- launching resume run")
            with open(RUN_LOG, "a") as lf:
                lf.write(f"\n===== supervised resume {time.strftime('%Y-%m-%d %H:%M:%S %Z')} =====\n")
                subprocess.run(RESUME_CMD, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
            after = n_cases_done()
            gained = after - done
            log(f"resume run returned: {gained} new cases ({after}/{TARGET_CASES} total)")
            if gained > 0:
                time.sleep(30)                 # keep going while budget lasts
            else:
                next_reset = next_reset + timedelta(days=1)   # today's budget spent; wait for next reset
                log(f"no progress post-reset; next attempt after {next_reset.isoformat()}")
        else:
            log(f"past boundary but only {ready}/{MIN_READY_KEYS} keys ready -- rechecking in {POLL_S}s")
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()
