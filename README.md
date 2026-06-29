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
| `evaluate.py` | Runs all baselines, measures token reduction / cost / retention / latency, supports the cheap-vs-expensive-extractor **ablation**, long-horizon stress scale, and optional notes compaction. |
| `make_figures.py` | Rebuilds publication figures and the deterministic long-horizon cost simulation. |
| `FINDINGS.md` | Concise measured findings, long-horizon simulation interpretation, and live-rerun status. |
| `scripts/` | Reproducibility helpers for live Groq reruns and publication-readiness checks. |
| `figures/` | Generated PDF/PNG figures for the paper and report. |

## Baselines

1. **Full context** — resend the entire history (the cost baseline).
2. **Truncation** — keep only the last *k* messages.
3. **Running summary** — the large model maintains a single running summary.
4. **Hierarchical (ours)** — cheap extractor writes notes; synth sees notes + recent.

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Deterministic offline smoke test (no API key, no cost):
.venv/bin/python evaluate.py --offline --scale small --out results_offline.json

# Long-horizon offline stress test with bounded notes:
.venv/bin/python evaluate.py --offline --scale long --compact-notes 8 --out results_long_offline.json

# Rebuild the paper figures and deterministic cost-vs-length simulation:
.venv/bin/python make_figures.py

# Real measured run used for the paper (Groq: llama-3.1-8b-instant extractor
# + llama-3.3-70b-versatile synthesizer):
export GROQ_API_KEY=gsk_...
.venv/bin/python evaluate.py --provider groq --scale small --ablation --out results_live.json

# Or use the helper script; it also reads GROQ_API_KEY from .env:
scripts/run_groq_experiment.sh

# Anthropic backend (Haiku extractor + Sonnet synthesizer):
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python evaluate.py --provider anthropic --scale small --ablation --out results.json

# Full paper-scale suite (20 conversations):
.venv/bin/python evaluate.py --provider groq --scale paper --ablation --out results_paper.json

# Pre-submission source/artifact audit:
.venv/bin/python scripts/publication_audit.py
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

## Current artifacts

- `results_live.json`: live Groq measured small-scale run used for the headline retention/cost results.
- `FINDINGS.md`: compact statement of the measured findings and safe claims.
- `results_growth_simulation.json`: deterministic token-accounting simulation over 14, 32, 60, 120, 240, and 480-message conversations.
- `results_long_offline.json`: offline long-scale plumbing run for the new `--scale long` tier (not a scientific retention result because mock answers are heuristic).
- `figures/architecture_flowchart.{pdf,png}`: method overview.
- `figures/measured_results_bars.{pdf,png}`: measured Groq comparison.
- `figures/retention_cost_frontier.{pdf,png}`: retention/cost frontier with bootstrap CIs.
- `figures/cascade_ablation.{pdf,png}`: cheap-vs-expensive extractor ablation.
- `figures/cost_vs_length_simulation.{pdf,png}`: deterministic long-horizon cost curve.
- `examples/sample_notes_product_development.json`: qualitative notes-file example showing timestamped fact updates.

## Research-integrity notes

- The paper avoids copied passages and long quotations; related work is cited rather than reproduced.
- `scripts/publication_audit.py` checks unresolved placeholders, missing bibliography entries, required artifacts, and accidental API-key leakage.
- No tool can guarantee acceptance or guarantee that an automated AI-detector will classify the prose a certain way; use the audit as a reproducibility/originality hygiene check, not as an evasion tool.

## TODO — making this stronger

The current evaluation establishes a favorable retention/cost trade-off at short
horizons. To turn that into a strong, publishable result, in rough priority
order:

**Experiments (highest impact)**
- [x] **Longer conversations (>100 messages).** Added `--scale long` with 120,
  240, and 480-message stress cases plus a deterministic cost-vs-length
  simulation. A live long-horizon retention run is still needed before making
  stronger quality claims.
- [ ] **Larger n + significance.** n=7 gives very wide CIs (truncation retention
  CI is [14.3, 85.7]). Bump to ≥30 conversations per length bucket so the
  headline numbers are tight enough to defend.
- [x] **Plot the cost-vs-length curve.** `make_figures.py` writes the curve to
  `figures/cost_vs_length_simulation.{pdf,png}`.
- [ ] **Second model family.** Reproduce the 8B-beats-70B ablation on a non-Llama
  cascade (e.g. Haiku→Sonnet) to show the cascade result isn't Llama-specific.

**Method**
- [x] **Notes compaction / dedup.** Added deterministic notes compaction and an
  `--compact-notes` evaluation flag; future work should replace this heuristic
  with learned or retrieval-aware compaction.
- [ ] **Retrieval instead of full-notes injection.** For long runs, fetch only
  relevant notes (embeddings / vector store) rather than concatenating all of
  them — this is also what unlocks the token win.
- [ ] **Tune `recent_k` / `segment_size`.** Sweep these and report sensitivity;
  right now they're fixed at 6.

**Rigor / repro**
- [ ] **Real tokenizer for cumulative counts.** Replace `tiktoken/cl100k` on
  Llama outputs with the provider's tokenizer so cumulative ≈ real.
- [x] **CI smoke test.** Wire `--offline` run into a GitHub Action so the repo
  stays green.
- [x] **Pin dependency versions** in `requirements.txt`.
- [x] **Replace `REPO_URL_PLACEHOLDER`** in `main.tex` with the public repo URL.

**Paper**
- [x] Add the cost-vs-length figure.
- [x] Add a qualitative example of an extracted notes file.
- [x] Reference the results and ablation tables explicitly from the Results prose.
