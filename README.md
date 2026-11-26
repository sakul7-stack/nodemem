# Retrieval Beats Cheap Structured Memory

Reference implementation for the paper
*"Retrieval Beats Cheap Structured Memory: A Cost–Retention Study of LLM Agent Memory on Real Long-Conversation Benchmarks"*
(Ishan Gautam, Kushal K.C., Pulchowk Campus, IOE, Tribhuvan University).

## Paper

The final submission package is in `submission/`:
- `submission/main.tex` — LaTeX source (IEEEtran, bibliography embedded)
- `submission/main.pdf` — Compiled PDF
- `submission/figures/` — 4 figures

## What's here

| File | Purpose |
|------|---------|
| `hierarchical_memory.py` | Memory stores + strategies: `NotesFile`, `SlotMemoryStore`, BM25 retriever, `FullContext`, `Truncation`, `RunningSummary`, `Hierarchical`, `SlotMemory`, `RetrievalRAG`, `SlotRAGHybrid` |
| `real_datasets.py` | Loaders for **LoCoMo** and **LongMemEval** |
| `judge.py` | LLM-as-judge scorer |
| `evaluate.py` | Run all strategies with LLM-judge scoring |
| `token_utils.py` | Token accounting and cost model |
| `make_figures.py` | Rebuild figures |
| `scripts/` | Paired analysis, summarization, audit scripts |

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Offline smoke test (no API key):
.venv/bin/python evaluate.py --offline --scale small --out results_offline.json

# Real LoCoMo run:
export GROQ_API_KEY=gsk_...
.venv/bin/python evaluate.py --provider groq --dataset locomo --out results.json
```

## Key results

See `submission/main.pdf` for full tables. Headline (LoCoMo n=160):

| Method | Token reduction | Retention | Cost |
|---|---|---|---|
| Full context | 0% | 30.6% | 100% |
| **BM25 retrieval** | **98%** | **39.4%** | **4.8%** |
| Slots + Retrieval | 95.9% | 38.8% | 8.4% |

BM25 retrieval holds retention level with full context at 4.8% cost.

## Citation

```bibtex
@misc{gautam2026retrieval,
  title={Retrieval Beats Cheap Structured Memory: A Cost--Retention Study of LLM Agent Memory on Real Long-Conversation Benchmarks},
  author={Gautam, Ishan and K.C., Kushal},
  year={2026}
}
```
