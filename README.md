# Cheap Structured Memory for LLM Agents: LoCoMo Cost/Retention Study

Reference implementation for the paper
*"Cheap Structured Memory for LLM Agents: A Cost–Retention Study on Real
Long-Conversation Benchmarks."*

A cheap model (the **extractor**) reads conversation segments and writes
structured memory (latest-value **fact slots** and/or append-only **notes**); a
larger model (the **synthesizer**) answers each question from that compact memory
— optionally fused with **BM25 retrieval** over past turns — plus a short
recent-message window, instead of the full transcript.

Evaluated on **LoCoMo** (real, human-annotated long conversations) with an
**LLM-as-judge** scorer using Nara Router cascade (mimo-v2.5-free extractor,
mistral-large synthesizer). On $n=50$ questions from 10 conversations (369–689
turns each), the slots+retrieval hybrid achieves 52.0% retention at 8.4% of
full-context cost — within 8 points of full context (60.0%) while being
$>10\times$ cheaper.

## What's here

| File | Purpose |
|------|---------|
| `hierarchical_memory.py` | Memory stores + strategies: `NotesFile`, `SlotMemoryStore`, BM25 retriever, `FullContext`, `Truncation`, `RunningSummary`, `Hierarchical`, `SlotMemory`, `RetrievalRAG`, `SlotRAGHybrid`, and provider clients (Anthropic + OpenAI-compatible + offline mock). |
| `real_datasets.py` | Loaders for **LoCoMo** into the common `Case`/`Fact` shape, with category labels and abstention flags. |
| `judge.py` | LLM-as-judge scorer using LongMemEval grading prompts (containment fallback offline). |
| `datasets.py` | Retained deterministic synthetic suite (offline sanity check only). |
| `evaluate.py` | Runs all strategies, real or synthetic datasets, LLM-judge scoring, per-category + per-question metrics, `--resume`, and per-call failure resilience. |
| `token_utils.py` | Provider-usage and local `tiktoken` token accounting plus the cost model. |
| `make_figures.py` | Rebuilds figures (frontier, per-category, cost-vs-length simulation). |
| `scripts/` | `paired_analysis.py` (question-level paired tests), `summarize_results.py` (paper tables), aggregation, audit. |
| `EVALUATION_REPORT.md` | Full results, methodology, and reproduction steps. |
| `figures/` | Generated PDF/PNG figures for the paper. |

## Baselines

1. **Full context** — entire history (cost baseline).
2. **Truncation** — keep only last $k=10$ messages.
3. **Running summary** — single rolling summary via same model.
4. **Hierarchical Notes** — append-only structured JSON extracted by small model.
5. **Fact-Slot Memory** — latest-value slot table extracted by small model.
6. **Retrieval (BM25)** — sparse top-$k$ retrieval over turns.
7. **Slots + Retrieval (hybrid)** — slots for durable facts + retrieved turns.

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Deterministic offline smoke test (no API key, no cost):
.venv/bin/python evaluate.py --offline --scale small --out results_offline.json

# Real LoCoMo run (Nara Router):
export NARA_API_KEY=sk-nry-...
.venv/bin/python evaluate.py --provider nara --dataset locomo \
  --seed 7 --segment-size 24 --max-qa-per-conv 5 \
  --out results_nara_rerun.json

# Aggregate + paired analysis:
.venv/bin/python scripts/aggregate_results.py results_nara_rerun.json \
  --out results_nara_aggregate.json
.venv/bin/python scripts/paired_analysis.py results_nara_rerun.json \
  --level question --out results_nara_paired.json

# Publication figures:
.venv/bin/python make_figures.py --results results_nara_aggregate.json
```

## Key results (LoCoMo, n=50, seed 7)

| Method | Token↓ (real) | Retention | Cost (% of full) |
|---|---|---|---|
| Full context | 0.0% | 60.0% | 100.0% |
| Truncation (k=10) | 98.1% | 26.0% | 4.2% |
| Running summary | 95.6% | 26.0% | 9.8% |
| Hierarchical Notes | 80.3% | 48.0% | 19.8% |
| Fact-Slot Memory | 96.8% | 38.0% | 6.4% |
| Retrieval (BM25) | 98.0% | 44.0% | 4.9% |
| **Slots + Retrieval** | **95.9%** | **52.0%** | **8.4%** |

All compression strategies achieve 80–96% cost reduction. The slots+retrieval
hybrid is the Pareto-optimal strategy.

## Current artifacts

- `results_nara_rerun.json`: seed 7 LoCoMo results (10/10 cases, 50 questions).
- `results_nara_aggregate.json`: pooled aggregate.
- `results_nara_paired.json`: question-level paired analysis.
- `results_nara_seed8.json`: seed 8 (3 cases, interrupted by free-tier credit exhaustion HTTP 402).
- `results_growth_simulation.json`: deterministic token-accounting simulation.
- `figures/`: architecture, measured results, per-category, frontier, cost simulation.

## Paper

Build with `pdflatex main.tex`. Requires IEEEtran class and standard LaTeX packages.
