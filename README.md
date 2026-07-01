# Persistent Note-Based Memory: Real-Benchmark Cost/Retention Study

Reference implementation and evaluation harness for the paper
*"Cheap Structured Memory for LLM Agents: A Cost–Retention Study on Real
Long-Conversation Benchmarks."*

A cheap model (the **extractor**) reads conversation segments and writes
structured memory (latest-value **fact slots** and/or append-only **notes**); a
larger model (the **synthesizer**) answers each question from that compact memory
— optionally fused with **BM25 retrieval** over past turns — plus a short
recent-message window, instead of the full transcript.

The evaluation now runs on **real, human-annotated long-conversation benchmarks**
(**LoCoMo**, **LongMemEval**) with an **LLM-as-judge** scorer, replacing the
earlier synthetic planted-fact pilot. On LoCoMo (conversations of several hundred
turns), the slots+retrieval hybrid and BM25 retrieval **match or exceed
full-context retention at 5–10% of the cost** (real token reduction > 95%). See
`REAL_DATA_EXPERIMENTS.md` for the live status, exact results, rate-limit notes,
and the one-command **resume** instructions.

## What's here

| File | Purpose |
|------|---------|
| `hierarchical_memory.py` | Memory stores + strategies: append-only `NotesFile`, `SlotMemoryStore`, BM25/dense retrievers, `FullContext`, `Truncation`, `RunningSummary`, `Hierarchical`, `SlotMemory`, `RetrievalRAG`, `SlotRAGHybrid`, and the provider clients (Anthropic + Groq/OpenAI-compatible + offline mock). |
| `real_datasets.py` | Loaders for **LoCoMo** and **LongMemEval** into the common `Case`/`Fact` shape, with category labels and abstention flags. |
| `judge.py` | LLM-as-judge scorer using the official LongMemEval grading prompts (containment fallback offline). |
| `datasets.py` | Retained deterministic synthetic suite (offline sanity check only). |
| `evaluate.py` | Runs all strategies, real or synthetic datasets, LLM-judge scoring, per-category + per-question metrics, `--resume`, and per-call failure resilience. |
| `token_utils.py` | Provider-usage and local `tiktoken` token accounting plus the cost model. |
| `make_figures.py` | Rebuilds figures (frontier, per-category, cost-vs-length simulation). |
| `scripts/` | `paired_analysis.py` (question-level paired tests), `summarize_results.py` (paper tables), aggregation, audit, Groq runner. |
| `REAL_DATA_EXPERIMENTS.md` | **Live status, results, rate-limit findings, and resume guide.** |
| `figures/` | Generated PDF/PNG figures for the paper. |

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
.venv/bin/python evaluate.py --provider groq --scale small --ablation --out results_live_rerun.json

# Multiple Groq keys can also be supplied in .env as:
# GROQ_API_KEYS=key1,key2

# Or use the helper script; it reads GROQ_API_KEY or GROQ_API_KEYS from .env:
scripts/run_groq_experiment.sh

# Optional replication seed and n=16 aggregate used by the current paper draft:
scripts/run_groq_experiment.sh --provider groq --scale small --seed 8 --ablation --out results_live_seed8.json
.venv/bin/python scripts/aggregate_results.py results_live_rerun.json results_live_seed8.json --out results_live_aggregate.json
.venv/bin/python make_figures.py --results results_live_aggregate.json

# Future runs now include raw per-case metrics, enabling paired analysis:
.venv/bin/python scripts/paired_analysis.py results_paper.json --out paired_results.json

# New experimental approach: latest-value fact-slot memory
.venv/bin/python evaluate.py --offline --scale tiny --slot-memory --out /tmp/nodemem_slot_tiny.json
scripts/run_groq_experiment.sh --provider groq --scale small --seed 9 --slot-memory --out results_live_slot_seed9.json
scripts/run_groq_experiment.sh --provider groq --scale small --seed 10 --slot-memory --out results_live_slot_seed10.json
.venv/bin/python scripts/aggregate_results.py results_live_slot_seed9.json results_live_slot_seed10.json --out results_live_slot_aggregate.json
.venv/bin/python scripts/paired_analysis.py results_live_slot_aggregate.json --method slot_memory --baselines full_context hierarchical truncation running_summary --out results_live_slot_paired.json

# Anthropic backend (Haiku extractor + Sonnet synthesizer):
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python evaluate.py --provider anthropic --scale small --ablation --out results.json

# Full paper-scale suite (20 conversations):
.venv/bin/python evaluate.py --provider groq --scale paper --ablation --out results_paper.json

# Pre-submission source/artifact audit:
.venv/bin/python scripts/publication_audit.py
```

## Metrics

- **Token reduction** — real per-query prompt tokens from provider `usage`, plus
  cumulative local-token counts for cost-growth shape, both relative to full context.
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

The paper reruns (`results_live_rerun.json` and `results_live_seed8.json`) used the Groq OpenAI-compatible backend, so
per-query counts come from Groq's `usage`. On the Anthropic backend, real counts
come from `messages.count_tokens` / `usage` instead. Pure char heuristics appear
only in `--offline` mode and are never reported as measured results.

## Reproducibility

The conversation suite is seeded (`--seed`, default 7) and built without any API
call, so the dataset is identical on every machine. Results in the paper were
produced with the models named in `results_live_aggregate.json["config"]` (Groq:
`llama-3.1-8b-instant` extractor + `llama-3.3-70b-versatile` synthesizer, n=16
across seeds 7 and 8). The aggregate pools run-level summaries, so its 95%
intervals are normal approximations; the per-seed files retain their original
bootstrap summaries. New evaluator outputs include `case_metrics`, and
`scripts/paired_analysis.py` can compute paired method-vs-baseline differences;
the existing n=16 live artifacts were generated before raw case rows were added.

## Current artifacts

- `results_live_aggregate.json`: current live Groq aggregate for the paper (16/16 cases across seeds 7 and 8); mixed/negative versus full context at short horizon.
- `results_live_rerun.json`: fresh seed-7 live Groq measured small-scale run (8/8 cases completed).
- `results_live_seed8.json`: seed-8 replication live Groq measured small-scale run (8/8 cases completed).
- `results_live.json`: earlier live Groq run retained for comparison.
- `results_live_slot_seed9.json`, `results_live_slot_seed10.json`: live fact-slot memory pilots (16/16 total cases completed).
- `results_live_slot_aggregate.json`: fact-slot memory aggregate; slot memory reached 85.2% retention at 89.1% of full-context cost.
- `results_live_slot_paired.json`: paired comparisons for the fact-slot strategy.
- `FINDINGS.md`: compact statement of the measured findings and safe claims.
- `NEXT_EXPERIMENT.md`: fact-slot memory hypothesis, commands, and success criteria.
- `results_growth_simulation.json`: deterministic token-accounting simulation over 14, 32, 60, 120, 240, and 480-message conversations.
- `results_long_offline.json`: offline long-scale plumbing run for the new `--scale long` tier (not a scientific retention result because mock answers are heuristic).
- `figures/architecture_flowchart.{pdf,png}`: method overview.
- `figures/measured_results_bars.{pdf,png}`: measured Groq comparison.
- `figures/retention_cost_frontier.{pdf,png}`: retention/cost frontier with 95% intervals.
- `figures/cascade_ablation.{pdf,png}`: cheap-vs-expensive extractor ablation.
- `figures/cost_vs_length_simulation.{pdf,png}`: deterministic long-horizon cost curve.
- `examples/sample_notes_product_development.json`: qualitative notes-file example showing timestamped fact updates.

## Research-integrity notes

- The paper avoids copied passages and long quotations; related work is cited rather than reproduced.
- `scripts/publication_audit.py` checks unresolved placeholders, missing bibliography entries, required artifacts, and accidental API-key leakage.
- No tool can guarantee acceptance or guarantee that an automated AI-detector will classify the prose a certain way; use the audit as a reproducibility/originality hygiene check, not as an evasion tool.

## TODO — making this stronger

The current evaluation is a pilot and does not establish a cost-saving win over
full context. To turn this into a stronger publishable result, in rough priority
order:

**Experiments (highest impact)**
- [x] **Longer conversations (>100 messages).** Added `--scale long` with 120,
  240, and 480-message stress cases plus a deterministic cost-vs-length
  simulation. A live long-horizon retention run is still needed before making
  stronger quality claims.
- [ ] **Larger n + significance.** n=16 still gives wide CIs (hierarchical
  retention CI is [59.6, 79.2]). Bump to ≥30 conversations per length bucket so the
  headline numbers are tight enough to defend.
- [ ] **Retention under compaction.** The 480-message 13.0% cost result is
  retention-blind. Run live QA with `--compact-notes 8` before claiming quality
  at long horizons.
- [x] **New representation to try.** Added optional `--slot-memory`, a latest-
  value fact-slot strategy designed to reduce prompt size and handle updates.
- [x] **Live fact-slot pilot.** Seeds 9 and 10 completed. Fact-slot memory was
  cheaper and higher-retention than append-only notes, but update retention still
  needs work.
- [x] **Raw per-case metrics.** Future evaluator outputs include `case_metrics`
  so paired tests can be run.
- [x] **Paired-analysis helper.** `scripts/paired_analysis.py` computes paired
  method-vs-baseline differences for outputs with `case_metrics`.
- [x] **Plot the cost-vs-length curve.** `make_figures.py` writes the curve to
  `figures/cost_vs_length_simulation.{pdf,png}`.
- [ ] **Second model family.** Reproduce the extractor ablation on a non-Llama
  cascade (e.g. Haiku→Sonnet) to test whether the inconclusive Llama result generalizes.

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
