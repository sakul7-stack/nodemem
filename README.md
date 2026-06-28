# Persistent Note-Based Memory for Token-Efficient LLM Agents

Reference implementation and evaluation harness for the paper
*"Persistent Note-Based Memory for Token-Efficient LLM Agents: A Hierarchical,
Model-Cascading Alternative to Full-Context Conversations."*

A cheap model (the **extractor**) reads conversation segments and writes
structured, append-only **notes**; an expensive model (the **synthesizer**)
answers each turn from those notes plus a short recent-message window instead of
the full transcript. This trades a small, cheap extraction cost for a large
reduction in expensive-model input tokens.

## What's here

| File | Purpose |
|------|---------|
| `hierarchical_memory.py` | `HierarchicalMemory`, the append-only `NotesFile`, the provider abstraction (Anthropic + an offline mock), and the four comparison strategies. |
| `token_utils.py` | Token counting via the **real Anthropic tokenizer** (`count_tokens`) and a cost model that includes the extractor and notes-growth terms. |
| `datasets.py` | Deterministic synthetic conversation suite with planted facts and adversarial **fact-update** probes. |
| `evaluate.py` | Runs all baselines, measures token reduction / cost / retention / latency, runs the cheap-vs-expensive-extractor **ablation**, and writes `results.json`. |

## Baselines

1. **Full context** — resend the entire history (the cost baseline).
2. **Truncation** — keep only the last *k* messages.
3. **Running summary** — the large model maintains a single running summary.
4. **Hierarchical (ours)** — cheap extractor writes notes; synth sees notes + recent.

## Quick start

```bash
pip install -r requirements.txt

# Deterministic offline smoke test (no API key, no cost):
python evaluate.py --offline --scale small --out results_offline.json

# Real measured run used for the paper (Groq: llama-3.1-8b-instant extractor
# + llama-3.3-70b-versatile synthesizer):
export GROQ_API_KEY=gsk_...
python evaluate.py --provider groq --scale small --ablation --out results_live.json

# Anthropic backend (Haiku extractor + Sonnet synthesizer):
export ANTHROPIC_API_KEY=sk-ant-...
python evaluate.py --provider anthropic --scale small --ablation --out results.json

# Full paper-scale suite (20 conversations):
python evaluate.py --provider groq --scale paper --ablation --out results_paper.json
```

## Metrics

- **Token reduction** — cumulative *input* tokens sent to the large model over a
  conversation, relative to full context (exact `count_tokens`).
- **Cost** — USD as a percentage of the full-context baseline, **including** the
  cheap extractor and running-summary maintenance.
- **Retention** — QA accuracy on planted facts; reported separately for the
  fact-update probes.
- **Latency** — mean per-response wall-clock; hierarchical = extractor + synth.

## A note on token counting

Two counts are reported, and neither is a char heuristic:

- **Real (per-query)** — taken from each completion's `usage` field, exactly as
  billed by the provider. These are the headline cost/token numbers.
- **Cumulative (growth curve)** — `tiktoken`/`cl100k` over the full history, used
  only for the *shape* of the cost-vs-length curve. `cl100k` is OpenAI's
  tokenizer and its absolute counts differ from Llama/Claude by ~15–20%, so it is
  never used for billed figures.

The paper run (`results_live.json`) used the Groq OpenAI-compatible backend, so
per-query counts come from Groq's `usage`. On the Anthropic backend, real counts
come from `messages.count_tokens` / `usage` instead. Pure char heuristics appear
only in `--offline` mode and are never reported as measured results.

## Reproducibility

The conversation suite is seeded (`--seed`, default 7) and built without any API
call, so the dataset is identical on every machine. Results in the paper were
produced with the models named in `results_live.json["config"]` (Groq:
`llama-3.1-8b-instant` extractor + `llama-3.3-70b-versatile` synthesizer, n=7).

## TODO — making this stronger

The current evaluation establishes a favorable retention/cost trade-off at short
horizons. To turn that into a strong, publishable result, in rough priority
order:

**Experiments (highest impact)**
- [ ] **Longer conversations (>100 messages).** This is where note-based memory
  should actually *win* on tokens — the notes file stays bounded while raw
  history grows. The current ≤56-message runs only show the trade-off, not the
  asymptotic payoff. Run `--scale paper` and add a `--scale long` tier.
- [ ] **Larger n + significance.** n=7 gives very wide CIs (truncation retention
  CI is [14.3, 85.7]). Bump to ≥30 conversations per length bucket so the
  headline numbers are tight enough to defend.
- [ ] **Plot the cost-vs-length curve.** Show the crossover point where
  hierarchical cost drops below full-context. A single figure makes the whole
  argument.
- [ ] **Second model family.** Reproduce the 8B-beats-70B ablation on a non-Llama
  cascade (e.g. Haiku→Sonnet) to show the cascade result isn't Llama-specific.

**Method**
- [ ] **Notes compaction / dedup.** Append-only notes grow unbounded; add
  periodic summarization of stale notes and conflict resolution by timestamp.
- [ ] **Retrieval instead of full-notes injection.** For long runs, fetch only
  relevant notes (embeddings / vector store) rather than concatenating all of
  them — this is also what unlocks the token win.
- [ ] **Tune `recent_k` / `segment_size`.** Sweep these and report sensitivity;
  right now they're fixed at 6.

**Rigor / repro**
- [ ] **Real tokenizer for cumulative counts.** Replace `tiktoken/cl100k` on
  Llama outputs with the provider's tokenizer so cumulative ≈ real.
- [ ] **CI smoke test.** Wire `--offline` run into a GitHub Action so the repo
  stays green.
- [ ] **Pin dependency versions** in `requirements.txt`.
- [ ] **Replace `REPO_URL_PLACEHOLDER`** in `main.tex` with the public repo URL.

**Paper**
- [ ] Add the cost-vs-length figure and a qualitative example of an extracted
  notes file.
- [ ] Reference Table~II / Table~III explicitly from the Results prose.
