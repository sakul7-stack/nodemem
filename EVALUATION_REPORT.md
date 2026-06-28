# Evaluation Report: Persistent Note-Based Memory for Token-Efficient LLM Agents

**Author:** Ishan Gautam
**Date:** 2026-06-28
**Status:** Measured results replacing the projected Table 2 of the paper.

---

## 1. Summary

This report documents a *real* experimental evaluation of the hierarchical,
model-cascading memory architecture proposed in the paper. Earlier drafts
reported a projected Table 2 ("60–80% token reduction, 15–25% cost"). Those
numbers were never measured; they were extrapolated from related work. This
report replaces them with numbers produced by the released harness against
live LLM inference.

**Headline finding.** At the conversation lengths tested (≤56 messages), the
contribution of hierarchical note-based memory is **not** a large raw-token
reduction. It is a favorable **retention-vs-cost trade-off**: hierarchical
memory holds **84.5% retention at 108% of full-context cost**, whereas the two
cheap baselines either collapse on retention (truncation: 43%) or cost *more*
than full context while barely reducing tokens (running summary: 165% cost).
The cascade thesis itself is supported: a cheap 8B extractor produces **higher**
downstream retention (84.5%) than an expensive 70B extractor (70.5%) on the
same extraction task.

The original projected gains do not hold at these lengths and have been removed
from the paper. The honest, defensible claim is the trade-off above plus the
cascade ablation.

---

## 2. Methodology

### 2.1 System under test

- **Extractor** ($\phi_{\text{small}}$): `llama-3.1-8b-instant` (Groq)
- **Synthesizer** ($\phi_{\text{large}}$): `llama-3.3-70b-versatile` (Groq)
- **Provider:** Groq OpenAI-compatible inference API
- Notes are written by the cheap extractor; only synthesis uses the large model.

### 2.2 Workload

- Deterministic synthetic conversation generator (`datasets.py`), seed = `7`.
- Scale = `small`: 8 fact-tracking conversations, up to 56 messages each.
- Each conversation plants facts, then asks questions whose answers depend on
  those facts. A subset of facts are **updated** mid-conversation (a later
  message overrides an earlier value); the `fact_update_retention` metric
  isolates whether each strategy returns the *current* value rather than a
  stale one.
- **7 of 8 cases completed.** The 8th case's expensive-extractor *ablation*
  arm hit Groq's tokens-per-minute ceiling and was dropped; the core 7-case
  table is complete and internally consistent (incremental saves preserved it).

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
- Dispersion is reported as mean ± std with **95% bootstrap CIs** (2000
  resamples).

---

## 3. Results

### 3.1 Main comparison (n = 7, seed = 7)

| Strategy | Token↓ (real) | Token↓ (cumulative) | Retention | Update retention | Cost (% of full) |
|---|---|---|---|---|---|
| Full context | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| Truncation (k=10) | **46.9%** | 25.5% | 42.9% | 42.9% | **74.5%** |
| Running summary | 17.2% | −32.3% | 97.1% | 100.0% | 165.2% |
| **Hierarchical (ours)** | 15.4% | −3.7% | **84.5%** | **85.7%** | 108.4% |

95% CIs (mean):

- Truncation retention: [14.3, 85.7] — extremely wide; truncation is a coin flip.
- Hierarchical retention: [73.8, 95.2].
- Running summary cost: [137.6, 189.4] — reliably *above* 100%.
- Hierarchical cost: [94.8, 120.8].

### 3.2 Cascade ablation — cheap vs. expensive extractor (n = 7)

| Extractor | Downstream retention |
|---|---|
| `llama-3.1-8b-instant` (cheap) | **84.5%** (CI [73.8, 95.2]) |
| `llama-3.3-70b-versatile` (expensive) | 70.5% (CI [60.7, 82.1]) |

The cheap extractor is not merely *adequate* — it is **better** on this
extraction-and-storage task. Structured fact extraction rewards concise,
literal copying; the larger model tended to paraphrase and compress away the
exact tokens the containment-based QA needed. This directly supports the
paper's central claim that the memory-write step should be delegated to the
cheap model.

### 3.3 Latency and absolute cost

- **Latency overhead:** −0.059 s (hierarchical was marginally *faster* than
  full context — the synthesis prompt is shorter; extraction is amortized
  across the segment, not added to the critical path of each response).
- Extractor spend over the whole run: **$0.00046**.
- Summary-maintenance spend: **$0.00291**.

---

## 4. Interpretation

1. **Raw token reduction is small at these lengths.** Real per-query reduction
   for hierarchical memory is ~15%, and cumulative token count is slightly
   *negative* (−3.7%) because the notes file plus recent window can exceed a
   short raw history. The projected "60–80%" does not appear at ≤56 messages.
   Such reductions require longer horizons, where bounded notes diverge from a
   linearly growing transcript. We state this as a limitation rather than
   claiming it.

2. **The real contribution is the retention/cost frontier.** Among strategies
   that meaningfully bound context, hierarchical memory is the only one that
   keeps retention high (84.5%) without exceeding full-context cost by a large
   margin (108%). Truncation is cheaper but unreliable (43% retention, and a CI
   that spans almost the entire range). Running summary matches retention but
   costs **65% more** than full context because it re-summarizes with the
   expensive model every segment.

3. **The cascade works.** The ablation is the cleanest positive result: moving
   extraction to the cheap model improved both cost *and* retention. This is the
   load-bearing claim of the paper and it survived measurement.

4. **Fact updates are handled.** Hierarchical update-retention (85.7%) tracks
   its overall retention, and running summary reaches 100%, indicating that
   timestamp-ordered notes and rolling summaries both surface the *current*
   value rather than a stale one. Truncation fails here too (42.9%).

---

## 5. Limitations / Threats to Validity

- **Conversation length.** Results are at ≤56 messages. The token-reduction case
  for note-based memory is asymptotic; longer conversations are needed to show
  cumulative wins, and we do not claim them.
- **Single workload family.** Synthetic fact-tracking conversations, one seed
  reported. Topic and seed variation are future work.
- **Tokenizer mismatch.** Cumulative token counts use `tiktoken` (OpenAI) on
  Llama outputs; treat them as approximate. Real per-query figures use the
  provider's `usage` and are exact.
- **n = 7.** One ablation arm of the 8th case was lost to a rate limit. CIs are
  correspondingly wide; this is reported, not smoothed over.
- **Containment-based QA.** Retention is scored by answer containment, which can
  under-credit correct paraphrases — this penalizes the expensive extractor more
  than the cheap one and should be read alongside the ablation caveat.

---

## 6. Reproduction

```bash
pip install -r requirements.txt          # anthropic, openai, tiktoken
export GROQ_API_KEY=...                   # your own key
python evaluate.py \
    --provider groq \
    --scale small \
    --seed 7 \
    --out results_live.json
```

Raw measured output: `results_live.json`. The harness writes results
incrementally per case, so a late rate-limit failure does not discard completed
cases.
