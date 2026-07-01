#!/usr/bin/env python3
"""Paired analysis for evaluator outputs.

Two granularities are supported:

* ``--level case`` (default): paired differences across per-conversation
  ``case_metrics`` rows (retention/cost/token-reduction). Best for cost, which
  is naturally a per-conversation quantity.
* ``--level question``: paired correctness differences across per-question
  ``question_metrics`` rows. This yields far more pairs (one per QA, not one per
  conversation), so retention comparisons are properly powered, and it supports
  a per-category breakdown plus an exact sign test.

Negative cost differences are better; positive retention/token-reduction
differences are better.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


CASE_METRICS = [
    "retention_pct",
    "fact_update_retention_pct",
    "cost_pct_of_baseline",
    "token_reduction_real_pct",
    "token_reduction_cumulative_pct",
]

DEFAULT_BASELINES = ["full_context", "truncation", "running_summary"]
DEFAULT_METHOD = "slot_memory"


def bootstrap_ci(values: list[float], iters: int = 5000, seed: int = 17) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def sign_test_p(values: list[float]) -> float | None:
    nonzero = [v for v in values if abs(v) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return None
    positives = sum(1 for v in nonzero if v > 0)
    k = min(positives, n - positives)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def _summarize_diffs(diffs: list[float]) -> dict:
    mean = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs)
    return {
        "paired_diff_mean": round(mean, 4),
        "ci95_bootstrap": [round(lo, 4), round(hi, 4)],
        "n_pairs": len(diffs),
        "sign_test_p_two_sided": None if sign_test_p(diffs) is None else round(sign_test_p(diffs), 6),
    }


# --------------------------------------------------------------------------- #
# Case-level (per conversation)
# --------------------------------------------------------------------------- #

def case_diffs(data: dict, method: str, baseline: str, metric: str) -> list[float]:
    diffs = []
    for row in data.get("case_metrics", []):
        strategies = row.get("strategies", {})
        if method not in strategies or baseline not in strategies:
            continue
        a = strategies[method].get(metric)
        b = strategies[baseline].get(metric)
        if a is None or b is None:
            continue
        diffs.append(float(a) - float(b))
    return diffs


def analyze_case(data: dict, method: str, baselines: list[str]) -> dict:
    out = {"comparisons": {}}
    for baseline in baselines:
        out["comparisons"][baseline] = {}
        for metric in CASE_METRICS:
            diffs = case_diffs(data, method, baseline, metric)
            if diffs:
                out["comparisons"][baseline][metric] = _summarize_diffs(diffs)
    return out


# --------------------------------------------------------------------------- #
# Question-level (per QA), with category breakdown
# --------------------------------------------------------------------------- #

def question_diffs(rows: list[dict], method: str, baseline: str) -> list[float]:
    diffs = []
    for row in rows:
        bs = row.get("by_strategy", {})
        if method not in bs or baseline not in bs:
            continue
        diffs.append(float(bs[method]["correct"]) - float(bs[baseline]["correct"]))
    return diffs


def analyze_question(data: dict, method: str, baselines: list[str]) -> dict:
    rows = data.get("question_metrics") or []
    if not rows:
        raise SystemExit(
            "This result file has no question_metrics. Re-run evaluate.py with the "
            "updated harness on a real dataset to store per-question rows."
        )
    categories = sorted({r.get("category", "default") for r in rows})
    out = {"n_questions": len(rows), "comparisons": {}}
    for baseline in baselines:
        entry = {"overall": {}, "by_category": {}}
        overall = question_diffs(rows, method, baseline)
        if overall:
            entry["overall"]["retention_diff"] = _summarize_diffs(overall)
        for cat in categories:
            cat_rows = [r for r in rows if r.get("category") == cat]
            d = question_diffs(cat_rows, method, baseline)
            if d:
                entry["by_category"][cat] = _summarize_diffs(d)
        out["comparisons"][baseline] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired method-vs-baseline analysis.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--level", choices=["case", "question"], default="question")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.result.read_text(encoding="utf-8"))
    body = (analyze_question(data, args.method, args.baselines)
            if args.level == "question"
            else analyze_case(data, args.method, args.baselines))
    payload = {
        "config": data.get("config", {}),
        "method": args.method,
        "level": args.level,
        "interpretation": (
            f"paired_diff = {args.method} - baseline; negative cost is better, "
            "positive retention/token-reduction is better"),
        **body,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
