#!/usr/bin/env python3
"""Aggregate independent ``evaluate.py`` result JSON files.

The current evaluator writes per-run summary statistics rather than every raw
case score.  This script pools means and population standard deviations using
standard summary-statistic formulas.  Because raw case observations are not
available, the aggregated 95% intervals are normal approximations rather than
bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


SUMMARY_KEYS = {"mean", "std", "ci95", "n"}


def is_summary_stat(value: object) -> bool:
    return isinstance(value, dict) and {"mean", "std", "n"}.issubset(value)


def combine_stats(rows: Iterable[dict]) -> dict:
    stats = [row for row in rows if row and int(row.get("n", 0)) > 0]
    total_n = sum(int(row["n"]) for row in stats)
    if total_n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0], "n": 0}

    mean = sum(int(row["n"]) * float(row["mean"]) for row in stats) / total_n
    second_moment = sum(
        int(row["n"]) * (float(row.get("std", 0.0)) ** 2 + float(row["mean"]) ** 2)
        for row in stats
    ) / total_n
    variance = max(0.0, second_moment - mean**2)
    std = math.sqrt(variance) if total_n > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(total_n) if total_n > 1 else 0.0
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "ci95": [round(mean - half_width, 2), round(mean + half_width, 2)],
        "n": total_n,
    }


def combine_nested_dicts(dicts: list[dict]) -> dict:
    keys = sorted({key for d in dicts for key in d})
    combined: dict = {}
    for key in keys:
        values = [d[key] for d in dicts if key in d]
        if all(is_summary_stat(value) for value in values):
            combined[key] = combine_stats(values)
        elif all(isinstance(value, dict) for value in values):
            combined[key] = combine_nested_dicts(values)
        elif all(isinstance(value, (int, float)) for value in values):
            combined[key] = round(sum(float(value) for value in values), 6)
        else:
            combined[key] = values[-1]
    return combined


def aggregate(paths: list[Path]) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("at least one result file is required")

    first_config = runs[0].get("config", {})
    source_files = [path.name for path in paths]
    config = dict(first_config)
    config.update({
        "n_cases_requested": sum(int(run.get("config", {}).get("n_cases_requested", 0)) for run in runs),
        "n_cases_completed": sum(int(run.get("config", {}).get("n_cases_completed", 0)) for run in runs),
        "seeds": [run.get("config", {}).get("seed") for run in runs],
        "seed": None,
        "source_files": source_files,
        "elapsed_s": round(sum(float(run.get("config", {}).get("elapsed_s", 0.0)) for run in runs), 1),
        "aggregation_method": "pooled from run-level summary means/stds; ci95 is normal approximation for summary tables",
    })

    latency = combine_nested_dicts([run.get("latency_s", {}) for run in runs])
    if "full_response" in latency and "hier_response" in latency:
        latency["overhead_mean"] = round(
            float(latency["hier_response"].get("mean", 0.0))
            - float(latency["full_response"].get("mean", 0.0)),
            3,
        )

    payload = {
        "config": config,
        "strategies": combine_nested_dicts([run.get("strategies", {}) for run in runs]),
        "cost_breakdown_usd": combine_nested_dicts([run.get("cost_breakdown_usd", {}) for run in runs]),
        "latency_s": latency,
        "aggregation_note": (
            "Run-level summaries were pooled from independent seeds. Means and standard "
            "deviations are reconstructed from rounded summary statistics, so summary-table "
            "values may differ slightly from an aggregation over raw per-case observations."
        ),
    }
    case_metrics = []
    for path, run in zip(paths, runs):
        seed = run.get("config", {}).get("seed")
        for row in run.get("case_metrics", []) or []:
            tagged = dict(row)
            tagged["source_file"] = path.name
            tagged["seed"] = seed
            case_metrics.append(tagged)
    if case_metrics:
        payload["case_metrics"] = case_metrics
        payload["case_metrics_note"] = "Raw per-case rows were preserved from source files; use scripts/paired_analysis.py for paired tests."

    ablations = [run["ablation_cheap_vs_expensive_extractor"] for run in runs
                 if "ablation_cheap_vs_expensive_extractor" in run]
    if ablations:
        payload["ablation_cheap_vs_expensive_extractor"] = combine_nested_dicts(ablations)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent evaluation result JSON files.")
    parser.add_argument("results", nargs="+", type=Path, help="result JSON files to aggregate")
    parser.add_argument("--out", type=Path, default=Path("results_live_aggregate.json"))
    args = parser.parse_args()

    payload = aggregate(args.results)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out} from {len(args.results)} run(s).")


if __name__ == "__main__":
    main()
