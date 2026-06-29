#!/usr/bin/env python3
"""Paired analysis for result files with raw per-case metrics.

Run this on outputs produced by the updated evaluator. Older result JSON files
may only contain run-level summaries; those cannot support paired tests.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


METRICS = [
    "retention_pct",
    "fact_update_retention_pct",
    "cost_pct_of_baseline",
    "token_reduction_real_pct",
    "token_reduction_cumulative_pct",
]

DEFAULT_BASELINES = ["full_context", "truncation", "running_summary"]
DEFAULT_METHOD = "hierarchical"


def bootstrap_ci(values: list[float], iters: int = 5000, seed: int = 17) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def sign_test_p(values: list[float]) -> float | None:
    nonzero = [value for value in values if abs(value) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return None
    positives = sum(1 for value in nonzero if value > 0)
    k = min(positives, n - positives)
    # Two-sided exact binomial sign test under p=0.5.
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def paired_rows(data: dict, method: str, baseline: str, metric: str) -> list[float]:
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


def analyze(data: dict, method: str, baselines: list[str]) -> dict:
    case_metrics = data.get("case_metrics") or []
    if not case_metrics:
        raise SystemExit(
            "This result file has no case_metrics. Re-run evaluate.py with the updated "
            "harness to store raw per-case rows before paired tests can be computed."
        )

    output = {
        "config": data.get("config", {}),
        "method": method,
        "interpretation": f"paired_diff = {method} - baseline; negative cost is better, positive retention/token-reduction is better",
        "comparisons": {},
    }
    for baseline in baselines:
        output["comparisons"][baseline] = {}
        for metric in METRICS:
            diffs = paired_rows(data, method, baseline, metric)
            if not diffs:
                continue
            mean = sum(diffs) / len(diffs)
            lo, hi = bootstrap_ci(diffs)
            output["comparisons"][baseline][metric] = {
                "paired_diff_mean": round(mean, 4),
                "ci95_bootstrap": [round(lo, 4), round(hi, 4)],
                "n_pairs": len(diffs),
                "sign_test_p_two_sided": None if sign_test_p(diffs) is None else round(sign_test_p(diffs), 6),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired method-vs-baseline analysis from raw case metrics.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--method", default=DEFAULT_METHOD,
                        help="strategy to compare against baselines")
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES,
                        help="baseline strategies to compare against")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.result.read_text(encoding="utf-8"))
    payload = analyze(data, method=args.method, baselines=args.baselines)
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
