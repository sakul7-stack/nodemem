# Evaluation Report: Persistent Note-Based Memory Pilot Study

**Author:** Ishan Gautam
**Date:** 2026-06-29
**Status:** Measured Groq results plus deterministic long-horizon cost simulation.

---

## 1. Summary

This report documents a *real* experimental evaluation of the hierarchical,
model-cascading memory architecture proposed in the paper. Earlier drafts
reported a projected Table 2 ("60–80% token reduction, 15–25% cost"). Those
numbers were never measured; they were extrapolated from related work. This
report replaces them with numbers produced by the released harness against
live LLM inference.

**Headline finding.** At the conversation lengths tested (≤56 messages), the
hierarchical method is **not a demonstrated cost-saving replacement for full
context**. It holds **69.4% retention at 102.7% of full-context cost**, while
full context holds 100.0% retention at 100.0% cost. It improves retention over
truncation (37.5% retention, 70.2% cost), but the short-horizon result is mixed
and partly negative rather than validating the architecture. The extractor-size
ablation is inconclusive: cheap 8B extraction and expensive 70B extraction are
69.4% vs. 68.2% retention with strongly overlapping intervals.

The original projected gains do not hold at these lengths and have been removed
from the paper. The honest, defensible claim is a pilot characterization:
persistent notes may help at longer horizons, but the current live result does
not prove token efficiency.

New artifacts added on 2026-06-29 include a long-horizon deterministic
token-accounting simulation, a `--scale long` stress-test tier, bounded-note
compaction, a reproducible figure-generation script, and publication-ready
figures under `figures/`.

On this device, two live Groq reruns completed successfully using comma-separated
`GROQ_API_KEYS` from `.env` without printing secrets. The fresh outputs are
`results_live_rerun.json` (seed 7, 8/8 cases, 330.7 seconds),
`results_live_seed8.json` (seed 8, 8/8 cases, 572.9 seconds), and the pooled
`results_live_aggregate.json` (16/16 cases, 903.6 seconds total).

---

## 2. Methodology

### 2.1 System under test

- **Extractor** ($\phi_{\text{small}}$): `llama-3.1-8b-instant` (Groq)
- **Synthesizer** ($\phi_{\text{large}}$): `llama-3.3-70b-versatile` (Groq)
- **Provider:** Groq OpenAI-compatible inference API
- Notes are written by the cheap extractor; only synthesis uses the large model.

### 2.2 Workload

- Deterministic synthetic conversation generator (`datasets.py`), seeds = `7`
  and `8`.
- Scale = `small`: 8 fact-tracking conversations per seed, up to 56 messages
  each, 16 completed live cases total.
- Each conversation plants facts, then asks questions whose answers depend on
  those facts. A subset of facts are **updated** mid-conversation (a later
  message overrides an earlier value); the `fact_update_retention` metric
  isolates whether each strategy returns the *current* value rather than a
  stale one.
- **16 of 16 cases completed** across the two fresh reruns, including the
  expensive-extractor ablation arm.

### 2.3 Baselines

1. **Full context** — entire history every turn (reference; retention = 100%).
2. **Truncation** — last `k = 10` messages only.
3. **Running summary** — single-model rolling summary regenerated each segment.
4. **Hierarchical (ours)** — append-only structured notes + recent `k = 6`.

### 2.4 Metrics and accounting

- **Token reduction (real):** from each provider response's own `usage` field —
  the per-query prompt tokens actually billed, vs. full context.
- **Token reduction (cumulative):** `tiktoken` / `cl100k` count of the whole
  context assembled over the conversation. Reported separately because
  `tiktoken` is an OpenAI tokenizer and only approximates Llama token counts;
  it is used for the *shape* of the growth curve, not for billing.
- **Retention:** fraction of planted-fact questions answered correctly
  (containment match).
- **Fact-update retention:** retention restricted to facts that were overridden.
- **Cost (% of baseline):** synthesis tokens × price + maintenance term
  (extractor calls for hierarchical; summary regeneration for running summary),
  using Groq's published per-token prices.
- **Latency overhead:** wall-clock hierarchical response time − full-context
  response time.
- Individual run dispersion is reported as mean ± std with **95% bootstrap CIs**
  (2000 resamples). The pooled `results_live_aggregate.json` artifact combines
  run-level summaries, so its intervals are normal approximations reconstructed
  from the two runs' rounded means, population standard deviations, and `n`.
- The evaluator now stores `case_metrics` for future runs, and
  `scripts/paired_analysis.py` computes paired method-vs-baseline differences.
  The current n=16 live artifacts were generated before raw per-case rows were
  stored, so paired tests cannot be applied retroactively.

---

## 3. Results

### 3.1 Main comparison (n = 16, seeds = 7 and 8)

| Strategy | Token↓ (real) | Token↓ (cumulative) | Retention | Update retention | Cost (% of full) |
|---|---|---|---|---|---|
| Full context | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| Truncation (k=10) | 50.5% | 29.8% | 37.5% | 37.5% | 70.2% |
| Running summary | 18.6% | −31.3% | 100.0% | 100.0% | 162.7% |
| Hierarchical (ours) | 20.6% | 1.7% | 69.4% | 68.8% | 102.7% |

95% CIs (mean):

- Truncation retention: [13.8, 61.2] — wide; truncation is unreliable.
- Hierarchical retention: [59.6, 79.2].
- Running summary cost: [138.5, 187.0] — reliably *above* 100%.
- Hierarchical cost: [94.7, 110.6].

### 3.2 Cascade ablation — cheap vs. expensive extractor (n = 16)

| Extractor | Downstream retention |
|---|---|
| `llama-3.1-8b-instant` (cheap) | 69.4% (CI [59.6, 79.2]) |
| `llama-3.3-70b-versatile` (expensive) | 68.2% (CI [59.1, 77.4]) |

The ablation is statistically inconclusive. The cheap extractor is slightly
higher in the pooled point estimate while being far cheaper, but the gap is only
1.2 percentage points and the confidence intervals almost entirely overlap. This
should not be reported as evidence that the cheap extractor is better.

### 3.3 Latency and absolute cost

- **Latency overhead:** 0.570 s in the pooled runs. Extraction is still
  amortized across segments, but provider-side variance made hierarchical
  responses slower than full context in these runs.
- Extractor spend over the two runs: **$0.00116**.
- Summary-maintenance spend over the two runs: **$0.00724**.

### 3.4 Deterministic long-horizon cost simulation

To address the paper's asymptotic claim without incurring a large live-API run,
we added a deterministic token-accounting simulation in `make_figures.py`. It
uses the same seeded conversation generator, prompt assembly code, `tiktoken` /
`cl100k` local token counting, and Groq price model as the evaluation harness.
Notes are produced by the deterministic offline extractor, so these results are
**cost-shape evidence only**; they are not live retention measurements.

The simulation combines the existing `paper` scale with a new `long` scale
(120, 240, and 480 messages) and compares full context, truncation,
append-only hierarchical notes, and hierarchical notes with deterministic
compaction to at most 8 note entries.

| Messages | Full context | Truncation | Hierarchical notes | Hierarchical + compaction |
|---:|---:|---:|---:|---:|
| 14 | 100.0% | 96.3% | 121.7% | 121.7% |
| 32 | 100.0% | 61.4% | 91.1% | 91.1% |
| 60 | 100.0% | 37.7% | 73.3% | 71.9% |
| 120 | 100.0% | 20.8% | 51.9% | 42.3% |
| 240 | 100.0% | 11.0% | 42.0% | 24.0% |
| 480 | 100.0% | 5.7% | 36.3% | 13.0% |

Interpretation: append-only notes already cross below full-context cost by the
32-message bucket in the deterministic simulation, but note growth keeps them
well above truncation. Compaction matters strongly at long horizons: at 480
messages, bounded hierarchical memory is **13.0%** of full-context cost versus
**36.3%** for append-only notes. This is cost-shape evidence only. It does not
show that compacted notes preserve facts, and it must not be merged with the
live n=16 retention result as if they were the same experiment.

---

## 4. Interpretation

1. **Raw token reduction is small at these lengths.** Real per-query reduction
  for hierarchical memory is ~21%, and cumulative token reduction is ~1.7%.
  The projected "60–80%" does not appear at ≤56 messages.
   Such reductions require longer horizons, where bounded notes diverge from a
   linearly growing transcript. We state this as a limitation rather than
   claiming it.

2. **The live result is dominated by full context at this scale.** Hierarchical
  memory keeps more information than truncation (69.4% vs. 37.5% retention),
  but it costs slightly more than full context while retaining much less. This
  is a negative short-horizon result.

3. **The cascade ablation is inconclusive.** The cheap extractor and expensive
  extractor are statistically indistinguishable in this pilot (69.4% vs. 68.2%).
  The cheap extractor is cheaper, but the overlapping intervals mean this should
  not be treated as evidence of a quality win.

4. **Fact updates are partially handled.** Hierarchical update-retention (68.8%)
  tracks its overall retention and outperforms truncation (37.5%), but still
  leaves room for better conflict resolution. Running summary reaches 100% at
  much higher cost.

---

## 5. Limitations / Threats to Validity

- **Conversation length.** Live retention results are at ≤56 messages. The
  deterministic long-horizon simulation shows the expected cost-growth curve,
  but a live >100-message run is still needed to measure cost and retention
  together.
- **Single workload family.** Synthetic fact-tracking conversations, two seeds
  reported. Broader topic and seed variation remain future work.
- **Tokenizer mismatch.** Cumulative token counts use `tiktoken` (OpenAI) on
  Llama outputs; treat them as approximate. Real per-query figures use the
  provider's `usage` and are exact.
- **n = 16.** The fresh reruns completed all small-scale cases, but CIs remain
  wide. Larger n is required before making stronger statistical claims.
- **Aggregate CIs are approximate.** The evaluator stores summary statistics,
  not raw per-case observations, so `results_live_aggregate.json` reconstructs
  pooled means/stds from rounded run-level summaries and uses normal-approximate
  intervals. Future runs should store per-case metrics for exact bootstrap CIs.
- **Compaction retention unmeasured.** The 13.0% at 480 messages is a
  retention-blind simulation. It assumes bounded notes but does not test whether
  bounded notes still answer correctly.
- **Containment-based QA.** Retention is scored by answer containment, which can
  under-credit correct paraphrases — this penalizes the expensive extractor more
  than the cheap one and should be read alongside the ablation caveat.
- **Live rerun dependency.** Re-running the Groq experiment requires valid Groq
  credentials. The evaluator accepts either `GROQ_API_KEY` or comma-separated
  `GROQ_API_KEYS` and the repository includes the exact helper script.

---

## 6. Research Integrity and Citation Hygiene

- The manuscript is written as original prose and avoids long copied passages or
  copyrighted excerpts.
- Claims about prior work are attributed through in-text citations and matching
  bibliography entries in `main.tex`.
- A local publication audit script checks unresolved placeholders, missing
  bibliography entries, required artifacts, and accidental API-key leakage:

```bash
.venv/bin/python scripts/publication_audit.py
```

This audit is a reproducibility and source-hygiene check. It does not guarantee
acceptance by a venue and cannot guarantee any particular automated AI-detector
classification.

---

## 7. Reproduction

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export GROQ_API_KEY=...                   # your own key
# Or put comma-separated keys in .env as GROQ_API_KEYS=key1,key2
.venv/bin/python evaluate.py \
  --provider groq \
  --scale small \
  --ablation \
  --seed 7 \
  --out results_live_rerun.json

.venv/bin/python evaluate.py \
  --provider groq \
  --scale small \
  --ablation \
  --seed 8 \
  --out results_live_seed8.json

.venv/bin/python scripts/aggregate_results.py \
  results_live_rerun.json results_live_seed8.json \
  --out results_live_aggregate.json

# Long-horizon deterministic cost simulation + figures:
.venv/bin/python make_figures.py --results results_live_aggregate.json

# Offline long-scale plumbing test with bounded notes:
.venv/bin/python evaluate.py --offline --scale long --compact-notes 8 --out results_long_offline.json

# Same live Groq setup via helper script:
scripts/run_groq_experiment.sh
```

Raw measured outputs: `results_live_rerun.json` and `results_live_seed8.json`;
pooled summary: `results_live_aggregate.json`. The harness writes results
incrementally per case, so a late rate-limit failure does not discard completed
cases.

Generated publication artifacts:

- `figures/architecture_flowchart.{pdf,png}`
- `figures/measured_results_bars.{pdf,png}`
- `figures/retention_cost_frontier.{pdf,png}`
- `figures/cascade_ablation.{pdf,png}`
- `figures/cost_vs_length_simulation.{pdf,png}`
- `examples/sample_notes_product_development.json`
- `results_live_rerun.json`
- `results_live_seed8.json`
- `results_live_aggregate.json`
- `results_growth_simulation.json`
- `results_long_offline.json`
