# Findings: Persistent Note-Based Memory Pilot Study

> **2026-07-01 UPDATE — LongMemEval knowledge-update result added.** On
> LongMemEval's knowledge-update category (n=72), BM25 retrieval matches full
> context (73.6% vs. 69.4%) at 88% cost. Fact-slots alone are the worst method
> (37.5%) — they fail even on their intended use-case, because the cheap 8B
> extractor misses most fact updates. A judge cross-validation (30 Qs, mistral-large
> vs gpt-oss-120b) shows 83.3% agreement, κ=0.468. Full details in
> **`REAL_DATA_EXPERIMENTS.md`**.
>
> **2026-06-30 UPDATE — migrated to real benchmarks.** The headline evaluation
> runs on the **real, human-annotated LoCoMo benchmark** with an **LLM-as-judge**
> scorer. On LoCoMo (n=160 questions, 8/10 conversations of 369–680 turns), BM25
> retrieval and a slots+retrieval hybrid exceed full-context retention (39.4% and
> 38.8% vs. 30.6%) at 5–9% of full-context cost. A question-level paired sign
> test makes retrieval/hybrid non-inferior to full context (p=0.09 / 0.12) and
> significantly better than truncation (p<0.001). This reverses the negative
> synthetic-pilot finding below. Full status, per-category tables, rate-limit
> notes are in **`REAL_DATA_EXPERIMENTS.md`**.
>
> Current LoCoMo table (n=160):
>
> | Strategy | Real token↓ | Retention | Cost |
> |---|---:|---:|---:|
> | Full context | 0.0% | 30.6% | 100.0% |
> | Truncation (k=10) | 98.1% | 21.9% | 4.2% |
> | Running summary | 96.9% | 26.9% | 7.7% |
> | Append-only notes | 80.0% | 30.0% | 22.8% |
> | Fact slots | 97.1% | 26.9% | 7.0% |
> | Retrieval (BM25) | 98.0% | 39.4% | 4.8% |
> | **Slots + retrieval** | **96.2%** | **38.8%** | **9.0%** |
>
> LongMemEval knowledge-update (n=72):
>
> | Strategy | Retention | Cost |
> |---|---:|---:|
> | Full context | 69.4% | 100.0% |
> | Truncation | 66.7% | 70.1% |
> | Running summary | 47.2% | 73.6% |
> | Fact slots | 37.5% | 59.5% |
> | **Retrieval (BM25)** | **73.6%** | **88.2%** |
> | **Slots + retrieval** | **73.6%** | **99.5%** |
>
> The sections below document the earlier **synthetic** pilot and are retained for
> history only.

---

## (Archived) Synthetic pilot findings

Date: 2026-06-29

## Fresh live Groq reruns on this device

The current headline file is `results_live_aggregate.json`, pooled from two
completed live reruns: `results_live_rerun.json` (seed 7) and
`results_live_seed8.json` (seed 8). Both were produced with:

- Extractor: `llama-3.1-8b-instant`
- Synthesizer: `llama-3.3-70b-versatile`
- Provider: Groq OpenAI-compatible API
- Scale: `small`, seeds `7` and `8`
- Completed cases: 16 of 16 requested cases
- Elapsed time: 903.6 seconds total

The aggregate was reconstructed from run-level summary statistics, so its 95%
intervals are normal approximations rather than bootstrap intervals. The
per-seed files retain their original bootstrap summaries.

| Strategy | Real token reduction | Retention | Update retention | Cost of full context |
|---|---|---:|---:|---:|---:|
| Full context | 0.0% | 100.0% | 100.0% | 100.0% |
| Truncation (`k=10`) | 50.5% | 37.5% | 37.5% | 70.2% |
| Running summary | 18.6% | 100.0% | 100.0% | 162.7% |
| Hierarchical notes | 20.6% | 69.4% | 68.8% | 102.7% |

Main finding: at <=56 messages, hierarchical notes are **not** a demonstrated
cost-saving replacement for full context. They cost slightly more than full
context (102.7%) while retaining substantially fewer facts (69.4% vs. 100.0%).
They are better than truncation on retention, but this is a mixed/negative pilot
result, not validation of token efficiency.

## Cascade ablation

| Extractor | Downstream retention |
|---|---:|
| `llama-3.1-8b-instant` | 69.4% |
| `llama-3.3-70b-versatile` | 68.2% |

Finding: inconclusive. The cheap 8B extractor is only 1.2 percentage points
higher than the 70B extractor and the CIs almost entirely overlap. This should
not be reported as evidence that the cheap extractor is better; it only shows no
obvious degradation in this small pilot.

## New approach: latest-value fact-slot memory

After the append-only note result proved weak, we tested a new representation:
`slot_memory`. Instead of injecting all note text, it stores compact current
facts as `subject | predicate | value | timestamp` and retrieves compact slots
for each question.

Fresh live Groq runs completed for seeds 9 and 10 using `--slot-memory`:

- `results_live_slot_seed9.json`: 8/8 cases
- `results_live_slot_seed10.json`: 8/8 cases
- `results_live_slot_aggregate.json`: 16/16 pooled cases
- `results_live_slot_paired.json`: paired comparisons against baselines

| Strategy | Real token reduction | Retention | Update retention | Cost of full context |
|---|---|---:|---:|---:|---:|
| Full context | 0.0% | 95.6% | 93.8% | 100.0% |
| Truncation (`k=10`) | 50.8% | 35.4% | 37.5% | 70.2% |
| Running summary | 20.1% | 100.0% | 100.0% | 159.8% |
| Append-only hierarchical notes | 19.2% | 75.7% | 87.5% | 103.9% |
| **Fact-slot memory** | **40.9%** | **85.2%** | **71.9%** | **89.1%** |

Interpretation: this is the first live result that is not dominated by the old
baselines. Fact-slot memory is still below full context on retention, especially
for updated facts, but it is cheaper than full context and dominates append-only
hierarchical notes on both retention and cost in the two-seed aggregate.

Paired result highlights for `slot_memory`:

- vs. full context: −10.9 cost points, −10.4 retention points.
- vs. append-only hierarchical notes: −14.9 cost points, +9.5 retention points.
- vs. truncation: +49.8 retention points, +18.9 cost points.
- vs. running summary: −70.7 cost points, −14.8 retention points.

This is promising but still small-n. The next test should run more seeds and
focus on improving update retention.

## Long-horizon deterministic cost simulation

The deterministic simulation in `results_growth_simulation.json` is not a live
retention result. It uses seeded conversations, the same context assembly code,
local `tiktoken` counts, Groq prices, deterministic note extraction, and optional
compaction. It shows the expected cost-growth shape:

| Messages | Hierarchical notes | Hierarchical + compaction |
|---:|---:|---:|
| 14 | 121.7% | 121.7% |
| 32 | 91.1% | 91.1% |
| 60 | 73.3% | 71.9% |
| 120 | 51.9% | 42.3% |
| 240 | 42.0% | 24.0% |
| 480 | 36.3% | 13.0% |

Finding: bounded notes/compaction are necessary for strong long-horizon cost
savings. At 480 messages, compacted notes fall to 13.0% of full-context cost.
However, this is retention-blind: it does not test whether the compacted notes
still answer fact questions correctly.

## Live rerun status on this device

Both live reruns were executed successfully using comma-separated `GROQ_API_KEYS`
from `.env` without printing the keys. To reproduce the default seed-7 run, run:

```bash
scripts/run_groq_experiment.sh
```

The script defaults to:

```bash
--provider groq --scale small --ablation --out results_live_rerun.json
```

To reproduce the added seed-8 replication and aggregate the two runs:

```bash
scripts/run_groq_experiment.sh --provider groq --scale small --seed 8 --ablation --out results_live_seed8.json
.venv/bin/python scripts/aggregate_results.py results_live_rerun.json results_live_seed8.json --out results_live_aggregate.json
```

The evaluator loads `.env` automatically and accepts either `GROQ_API_KEY` or
comma-separated `GROQ_API_KEYS`.

To reproduce the fact-slot memory runs:

```bash
scripts/run_groq_experiment.sh --provider groq --scale small --seed 9 --slot-memory --out results_live_slot_seed9.json
scripts/run_groq_experiment.sh --provider groq --scale small --seed 10 --slot-memory --out results_live_slot_seed10.json
.venv/bin/python scripts/aggregate_results.py results_live_slot_seed9.json results_live_slot_seed10.json --out results_live_slot_aggregate.json
.venv/bin/python scripts/paired_analysis.py results_live_slot_aggregate.json --method slot_memory --baselines full_context hierarchical truncation running_summary --out results_live_slot_paired.json
```

## Publication-safe interpretation

- Do not claim 60--80% live token reduction at short horizons.
- Do not claim the method beats full context in the live n=16 result.
- Describe the live result as a mixed/negative pilot: similar cost to full context, lower retention.
- Describe the cheap-vs-expensive extractor ablation as inconclusive.
- Treat fact-slot memory as a promising new variant, not yet as a final result.
- Treat the 120--480 message curve as deterministic cost-shape evidence only, not quality evidence.
- State that live long-horizon retention and retention under compaction remain future work.
