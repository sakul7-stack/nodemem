#!/usr/bin/env python3
"""Summarize an evaluator result file into paper-ready tables.

Prints an overall strategy table (token reduction / retention / cost), a
per-category retention matrix, and (if question_metrics are present) paired
retention deltas vs. full context with an exact sign test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ORDER = ["full_context", "truncation", "running_summary", "hierarchical",
         "slot_memory", "rag", "slot_rag_hybrid"]
LABELS = {
    "full_context": "Full context", "truncation": "Truncation (k=10)",
    "running_summary": "Running summary", "hierarchical": "Append-only notes",
    "slot_memory": "Fact slots", "rag": "Retrieval (BM25)",
    "slot_rag_hybrid": "Slots + retrieval",
}


def sign_test_p(diffs):
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = min(sum(d > 0 for d in nz), sum(d < 0 for d in nz))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--baseline", default="full_context")
    args = ap.parse_args()
    d = json.loads(args.result.read_text())
    cfg = d["config"]

    print(f"# {args.result.name}")
    print(f"dataset={cfg.get('dataset')} synth={cfg.get('synth_model')} "
          f"judge={cfg.get('judge')}/{cfg.get('judge_model')} "
          f"n_conv={cfg.get('n_cases_completed')} "
          f"n_q={len(d.get('question_metrics', []))}")
    print()
    # Overall table
    hdr = f"{'strategy':20s} {'tokRed%':>8s} {'retention%':>11s} {'cost%':>8s}"
    print(hdr); print("-" * len(hdr))
    for s in strat_list(d):
        st = d["strategies"][s]
        print(f"{LABELS.get(s, s):20s} "
              f"{st['token_reduction_real_pct']['mean']:8.1f} "
              f"{st['retention_pct']['mean']:11.1f} "
              f"{st['cost_pct_of_baseline']['mean']:8.1f}")
    print()
    # Per-category retention
    cats = []
    for s in strat_list(d):
        for c in d["strategies"][s].get("retention_by_category_pct", {}):
            if c not in cats:
                cats.append(c)
    cats.sort()
    if cats:
        print("Per-category retention (%)")
        print(f"{'strategy':20s} " + " ".join(f"{c[:10]:>11s}" for c in cats))
        for s in strat_list(d):
            bc = d["strategies"][s].get("retention_by_category_pct", {})
            print(f"{LABELS.get(s, s):20s} " +
                  " ".join(f"{bc.get(c, 0):11.1f}" for c in cats))
        print()
    # Paired vs baseline (question level)
    qm = d.get("question_metrics", [])
    if qm:
        base = args.baseline
        print(f"Paired retention delta vs {base} (question level, n={len(qm)})")
        print(f"{'strategy':20s} {'delta_pp':>9s} {'sign_p':>9s}")
        for s in strat_list(d):
            if s == base:
                continue
            diffs = [r["by_strategy"][s]["correct"] - r["by_strategy"][base]["correct"]
                     for r in qm if base in r["by_strategy"] and s in r["by_strategy"]]
            if not diffs:
                continue
            delta = 100.0 * sum(diffs) / len(diffs)
            print(f"{LABELS.get(s, s):20s} {delta:9.1f} {sign_test_p(diffs):9.4f}")


def strat_list(d):
    return [s for s in ORDER if s in d["strategies"]]


if __name__ == "__main__":
    main()
