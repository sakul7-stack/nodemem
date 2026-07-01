#!/usr/bin/env python3
"""error_analysis.py -- why do retrieval/slots fail on LoCoMo multi-hop questions?

For each multi-hop (category 1) question we know the gold answer and the gold
*evidence* turns (dia ids). We run the same BM25 retriever used by the harness
and measure how much of the evidence it actually surfaces in the top-k. This
isolates retrieval misses / fragmentation (the hypothesised multi-hop failure
mode) from downstream reasoning failures, and dumps examples for qualitative
reading.

Usage: python scripts/error_analysis.py [top_k] [n_examples]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hierarchical_memory import BM25Retriever  # noqa: E402

RAW = Path("data/locomo10.json")
TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_EX = int(sys.argv[2]) if len(sys.argv) > 2 else 18


def dia_map(conv: dict) -> dict[str, str]:
    """dia_id -> 'speaker: text [+caption]' for every turn in a conversation."""
    out = {}
    for k in conv:
        if re.match(r"session_\d+$", k):
            for t in conv[k] or []:
                text = (t.get("text") or "").strip()
                if t.get("blip_caption"):
                    text += f" [shares a photo: {t['blip_caption'].strip()}]"
                out[t.get("dia_id", "")] = f"{t.get('speaker','')}: {text}".strip()
    return out


def turn_docs(conv: dict) -> list[str]:
    docs = []
    for k in conv:
        if re.match(r"session_\d+$", k):
            for t in conv[k] or []:
                text = (t.get("text") or "").strip()
                if t.get("blip_caption"):
                    text += f" [shares a photo: {t['blip_caption'].strip()}]"
                if text:
                    docs.append(f"{t.get('speaker','')}: {text}".strip())
    return docs


def main() -> None:
    data = json.loads(RAW.read_text())
    full = part = none = 0
    n = 0
    frac_sum = 0.0
    examples = []
    for sample in data:
        conv = sample["conversation"]
        dmap = dia_map(conv)
        docs = turn_docs(conv)
        retr = BM25Retriever(docs)
        for q in sample.get("qa", []):
            if q.get("category") != 1:   # multi-hop only
                continue
            evidence = [e for e in (q.get("evidence") or []) if e in dmap]
            if not evidence:
                continue
            n += 1
            hits = retr.retrieve(q["question"], top_k=TOP_K)
            hitset = set(hits)
            covered = sum(1 for e in evidence if dmap[e] in hitset)
            frac = covered / len(evidence)
            frac_sum += frac
            if covered == len(evidence):
                full += 1
            elif covered == 0:
                none += 1
            else:
                part += 1
            if len(examples) < N_EX:
                examples.append({
                    "q": q["question"], "answer": q.get("answer"),
                    "n_evidence": len(evidence), "covered": covered,
                    "evidence_texts": [dmap[e] for e in evidence],
                })
    print(f"LoCoMo multi-hop questions: {n}")
    print(f"BM25 top-{TOP_K} evidence coverage:")
    print(f"  mean fraction of evidence turns retrieved: {frac_sum/n:.2%}")
    print(f"  all evidence retrieved:     {full} ({full/n:.0%})")
    print(f"  partial (some hops missed): {part} ({part/n:.0%})")
    print(f"  zero evidence retrieved:    {none} ({none/n:.0%})")
    print(f"  -> multi-hop questions with at least one hop missing: {(part+none)/n:.0%}")
    print(f"\nAvg evidence turns per multi-hop question: "
          f"{(sum(e['n_evidence'] for e in examples)/len(examples) if examples else 0):.1f} (sample)")
    print(f"\n--- {len(examples)} example multi-hop questions (for qualitative read) ---")
    for i, ex in enumerate(examples, 1):
        print(f"\n[{i}] Q: {ex['q']}")
        print(f"    gold: {ex['answer']} | evidence turns: {ex['n_evidence']}, "
              f"BM25-covered: {ex['covered']}")
        for t in ex["evidence_texts"]:
            print(f"      - {t[:130]}")


if __name__ == "__main__":
    main()
