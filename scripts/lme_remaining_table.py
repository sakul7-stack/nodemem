#!/usr/bin/env python3
"""Emit paper-ready LaTeX for the remaining LongMemEval categories
(temporal / multi-session / abstention) from a results JSON.

Produces, matching the style of Table~\\ref{tab:lme_ku} in main.tex:
  1. an overall retention/cost table across the blended question set;
  2. a per-category retention matrix (strategy x category);
  3. headline stats (paired sign test vs full context) for the prose.

Usage: python scripts/lme_remaining_table.py results_lme_remaining.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ORDER = ["full_context", "truncation", "running_summary", "hierarchical",
         "slot_memory", "rag", "slot_rag_hybrid"]
LABELS = {
    "full_context": "Full Context", "truncation": "Truncation",
    "running_summary": "Running Summary", "hierarchical": "Hierarchical Notes",
    "slot_memory": "Fact-Slot Memory", "rag": "Retrieval (BM25)",
    "slot_rag_hybrid": "Slots + Retrieval",
}
CATS = ["temporal", "multi_session", "abstention"]
CAT_LABEL = {"temporal": "Temporal", "multi_session": "Multi-Sess.",
             "abstention": "Abstention"}


def sign_test_p(diffs):
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = min(sum(d > 0 for d in nz), sum(d < 0 for d in nz))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def present(d):
    return [s for s in ORDER if s in d["strategies"]]


def overall_table(d):
    strat = d["strategies"]
    n = strat["full_context"]["retention_pct"]["n"]
    lines = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{LongMemEval temporal + multi-session + abstention retention "
        rf"($n={n}$, LLM-judge)}}",
        r"\label{tab:lme_remaining}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"\textbf{Method} & \textbf{Retention} & \textbf{Cost} \\", r"\midrule",
    ]
    for s in present(d):
        r = strat[s]["retention_pct"]["mean"]
        c = strat[s]["cost_pct_of_baseline"]["mean"]
        lines.append(rf"{LABELS[s]} & {r:.1f}\% & {c:.1f}\% \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def category_table(d):
    strat = d["strategies"]
    cats = [c for c in CATS
            if any(c in strat[s].get("retention_by_category_pct", {}) for s in present(d))]
    counts = {}
    for s in present(d):
        for c, v in strat[s].get("category_counts", {}).items():
            counts[c] = v
    header = " & ".join(rf"\textbf{{{CAT_LABEL.get(c, c)}}}" for c in cats)
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Per-category retention on the remaining LongMemEval "
        r"categories (\%, LLM-judge). Abstention credits correct refusal.}",
        r"\label{tab:lme_remaining_cat}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{l{'c' * len(cats)}}}", r"\toprule",
        rf"\textbf{{Method}} & {header} \\",
    ]
    lines.append("$n$ " + "".join(rf"& {counts.get(c, 0)} " for c in cats) + r"\\")
    lines.append(r"\midrule")
    for s in present(d):
        rbc = strat[s].get("retention_by_category_pct", {})
        row = " & ".join(f"{rbc.get(c, float('nan')):.1f}" if c in rbc else "--" for c in cats)
        lines.append(rf"{LABELS[s]} & {row} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def headline(d):
    qm = d.get("question_metrics", [])
    base = "full_context"
    out = []
    for s in present(d):
        if s == base:
            continue
        diffs = []
        for q in qm:
            bs = q.get("by_strategy", {})
            if base in bs and s in bs:
                diffs.append(int(bs[s]["correct"]) - int(bs[base]["correct"]))
        if not diffs:
            continue
        delta = 100.0 * sum(diffs) / len(diffs)
        out.append(f"  {LABELS[s]:20s} delta={delta:+5.1f}pp  sign_p={sign_test_p(diffs):.4f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    args = ap.parse_args()
    d = json.loads(args.result.read_text())
    cfg = d["config"]
    n = len(d.get("question_metrics", []))
    print(f"% source: {args.result.name}  n_cases={len(d['case_metrics'])}  n_q={n}")
    print(f"% dataset={cfg.get('dataset')} synth={cfg.get('synth_model')} "
          f"judge={cfg.get('judge_model')}")
    print("\n% ===== Overall table =====\n")
    print(overall_table(d))
    print("\n% ===== Per-category table =====\n")
    print(category_table(d))
    print("\n% ===== Headline: paired retention delta vs full context =====")
    print(headline(d))


if __name__ == "__main__":
    main()
