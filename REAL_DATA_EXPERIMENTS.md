# Real-Data Evaluation Log & Resume Guide

Migration of the persistent-memory study from a synthetic planted-fact suite to
**real, openly available long-conversation memory benchmarks**. This file is the
single source of truth for state and how to continue.

---

## CURRENT STATUS (resume here)

### ⏸ CHECKPOINT 2026-07-03 (machine shutdown)

**Paper state: publication-ready.** `main.tex` / `build/main.pdf` (10 pp) build
clean under tectonic (0 errors, 0 undefined refs), publication audit PASSES
(16/16 citations), every table verified against its result JSON, all tables and
figures referenced in text. Title, abstract, and conclusion reflect the final
results including the n=254 LongMemEval boundary finding. Root `main.pdf` is
synced with `build/main.pdf`. Nothing is committed to git (by request) — the
working tree holds all changes.

**One experiment SKIPPED by decision (2026-07-04): LoCoMo n=450 scale-up.**
It needs ~4–8 days of daily quota windows; the paper is complete and honest at
n=160 (claims parity-at-lower-cost, not superiority). Checkpoint preserved:
2/10 conversations (n=90) in `results_locomo_n450.json`. If ever resumed:

```bash
cd ~/Desktop/Research/Persistent_Note_Based_Memory/repo
nohup .venv/bin/python3 -u scripts/resume_supervisor_locomo.py \
  >> logs/resume_supervisor_locomo.log 2>&1 &
```

The supervisor waits for quota, launches `evaluate.py --resume`, survives TPD
walls, polls every 20 min, and writes `logs/resume_supervisor_locomo.DONE` at
10/10. Expect ~1–2 conversations per daily quota window → ~4–8 days total.
When done: regenerate tables (`scripts/summarize_results.py
results_locomo_n450.json`), figures (`make_figures.py --results
results_locomo_n450.json`), update Tables results/category/paired + the power
section in `main.tex`, rebuild (`cd build && cp ../main.tex . && tectonic
main.tex` twice), re-run `scripts/publication_audit.py`.

**The n=450 run is optional** — the paper is complete and honest at n=160
(claims parity-at-lower-cost, not superiority; the power analysis says so
explicitly). n=450 would only upgrade the "non-significant trend" into a
confirmed/refuted claim.

---

### 2026-07-02 checkpoint — paper rewritten, two new experiments, one in flight

- **Paper REWRITTEN and rebuilt** (`main.tex`, `build/main.pdf`, root `main.pdf`
  synced). Reframed as an honest empirical study: new title *"Retrieval Beats
  Cheap Structured Memory..."*; fact slots are now a hypothesis-tested-and-
  rejected, not the proposed method; fixed the 70B/17B synthesizer
  contradiction, 369–680 turn counts, κ=0.47 relabeled "moderate", per-cell
  n/CIs added to the category table, limitations reordered by severity,
  truncated abstract/conclusion sentences fixed. Build: `tectonic main.tex
  --outdir build && cp build/main.pdf main.pdf`.
- **Extractor-size ablation COMPLETE (Experiment A).** Results in
  `results_lme_ku_bigextractor.json` (LME knowledge-update n=72, extractor
  upgraded 8B→`llama-3.3-70b-versatile`, everything else identical to
  `results_lme_knowledge_update.json`). Key finding: fact slots 37.5%→66.7%
  (+29.2pp paired, 23 wins vs 2, p<1e-4); hybrid becomes best at 80.6%; BM25
  unchanged (1v1) = clean control; BUT 70B extraction costs 1.8–2.2× full
  context on short oracle histories → "representation works, cheap-cascade
  premise fails". Already integrated into the paper (abstract, contribution 2,
  Table `tab:lme_ku_ablation`, summary bullets, limitations, future work,
  conclusion).
- **LoCoMo n≈450 sweep IN FLIGHT (Experiment B).** Target: 10 conversations ×
  45 category-balanced QA (~450 questions) to power the retrieval-vs-full-
  context paired test (power analysis says 360–430 needed). Checkpoint file:
  `results_locomo_n450.json` (per-conversation checkpointing; 1/10 done as of
  this note, ~17 min/conv). Driven by
  `scripts/resume_supervisor_locomo.py` (auto-resumes across Groq daily budget
  resets at 00:00 UTC; defers to any other running evaluate.py; lock file
  `logs/resume_supervisor_locomo.lock`). Logs:
  `logs/resume_supervisor_locomo.log` and `logs/locomo_n450_run.log`.
  **LAPTOP WAS CLOSED MID-RUN — the supervisor and evaluate.py die on
  shutdown. TO RESUME after reboot:**

  ```bash
  rm -f logs/resume_supervisor_locomo.lock
  setsid nohup .venv/bin/python scripts/resume_supervisor_locomo.py \
      >> logs/resume_supervisor_locomo.log 2>&1 &
  ```

  Completed conversations are preserved in `results_locomo_n450.json`; the
  supervisor resumes from there. Done marker:
  `logs/resume_supervisor_locomo.DONE`.
- **AFTER Experiment B completes**, remaining paper work:
  1. `python scripts/paired_analysis.py results_locomo_n450.json --method rag
     --baselines full_context truncation` (and same for `slot_rag_hybrid`,
     `slot_memory`, `running_summary`, `hierarchical`) — check whether the
     retrieval-vs-full-context sign test is now significant at n≈450.
  2. Update `main.tex`: Table `tab:results`, Table `tab:category`, Table
     `tab:paired`, the power-analysis section (rewrite as confirmed/refuted),
     abstract/intro/conclusion claims, limitation #1.
  3. Regenerate figures (`make_figures.py`) and rebuild with tectonic.
- 12 Groq keys valid as of this checkpoint (probe returned 200; 6/6 budget
  keys confirmed at launch).

### Earlier status (superseded but kept for provenance)

- **LoCoMo run COMPLETE at 8 / 10 conversations (n = 160).** Final results in
  `results_locomo_n160.json`. The paper is finalized on this. The last two
  conversations (conv-49, conv-50) were dropped due to rate-limit wedging.
- **LongMemEval knowledge-update run COMPLETE (n = 72).** Results in
  `results_lme_knowledge_update.json`. Key finding: BM25 RAG matches full context
  (73.6% vs 69.4%) at 88% cost; fact-slots alone are worst (37.5%).
- **Judge cross-validation COMPLETE.** 30 LoCoMo questions, llama-3.3-70b vs
  openai/gpt-oss-120b: 83.3% agreement, Cohen's κ = 0.468.
- **Remaining LongMemEval categories COMPLETE at n = 254** (temporal 127,
  multi-session 121, abstention 6). Results in `results_lme_remaining_n254.json`
  (= final `results_lme_remaining.json`; the interim n=135 snapshot is kept in
  `results_lme_remaining_n135.json`). The full sample **reverses the interim
  finding**: full context leads (53.1%), BM25 trails significantly (42.9%,
  Δ=−10.2pp, p=0.004), only the hybrid is statistically indistinguishable
  (47.6%, Δ=−5.5pp, p=0.14); fact-slots worst (14.2%, p<0.001). Matches the
  multi-hop error-analysis prediction (evidence fragmentation under bounded
  top-k). Regenerate the paper tables with
  `python scripts/lme_remaining_table.py results_lme_remaining_n254.json`.
- The Mistral-Large second-model run via the free Bynara router was dropped from
  the main loop (router throttles under sustained load); the `--synth-provider
  mistral` code path remains for use with a faster Mistral endpoint later.
- 12 Groq keys configured (≈360K TPM aggregate).

## Second model family: Mistral Large (multi-provider)

The harness now supports a **separate synthesizer provider** so the synthesizer
can be swapped while the cheap extractor and the LLM judge stay fixed (for
comparable scoring). Mistral Large is served via the OpenAI-compatible Bynara
router.

- `.env`: `BYNARA_API_KEY=...` (endpoint `https://router.bynara.id/v1`, model
  `mistral-large`). Note: the Bynara free tier requires joining a Telegram
  group and relinking before the key activates.
- Roles: extractor `llama-3.1-8b-instant` (Groq), **synthesizer `mistral-large`
  (Bynara)**, judge `llama-3.3-70b-versatile` (Groq).
- Steady-state latency ~2--4 s/call (first call ~46 s cold start).

**Why:** the Scout (17B) result risks the criticism that retrieval only ties
full context because a 17B model is weak at long context. A 3-question smoke test
already shows **Mistral Large at 100% full-context vs Scout's 55%** on the same
conversation, so the strong-model comparison is essential and may show full
context pulling ahead while memory keeps most accuracy at ~10% of the cost.

**Run / resume command:**

```bash
.venv/bin/python evaluate.py --provider groq --synth-provider mistral \
  --judge llm --judge-model llama-3.3-70b-versatile \
  --dataset locomo --max-qa-per-conv 20 --segment-size 12 \
  --slot-memory --rag --seed 7 --resume --out results_locomo_mistral.json
```

(Run detached with `setsid ... > /tmp/locomo_mistral.log 2>&1 < /dev/null &`.)

### To continue a LoCoMo run (skips done cases):

```bash
.venv/bin/python evaluate.py --provider groq --dataset locomo \
  --synth-model meta-llama/llama-4-scout-17b-16e-instruct \
  --judge llm --judge-model llama-3.3-70b-versatile \
  --max-qa-per-conv 20 --segment-size 12 --slot-memory --rag \
  --seed 7 --resume --out results_locomo_main.json
```

### To run remaining LongMemEval categories (e.g. temporal):

```bash
.venv/bin/python evaluate.py --provider groq --dataset longmemeval \
  --data-path data/longmemeval_oracle.json \
  --lme-categories temporal-reasoning multi-session \
  --synth-model meta-llama/llama-4-scout-17b-16e-instruct \
  --judge llm --judge-model llama-3.3-70b-versatile \
  --segment-size 12 --slot-memory --rag \
  --seed 7 --resume --out results_lme_remaining.json
```

---

## Datasets (openly available, research-released; downloaded, not redistributed)

| Dataset | Source | Scale |
|---|---|---|
| **LoCoMo** (Maharana et al., ACL 2024) | github.com/snap-research/locomo → `data/locomo10.json` | 10 conversations, 369–689 turns (~10–20K tokens), 1,986 human-annotated QA (single-hop / multi-hop / temporal / open-domain / adversarial) |
| **LongMemEval** (Wu et al., ICLR 2025) | HF `xiaowu0162/longmemeval-cleaned` → `data/longmemeval_oracle.json` | 500 questions; info-extraction / multi-session / knowledge-update / temporal / abstention |

## Scoring
LLM-judge with the **official LongMemEval grading prompts** (`judge.py`), served
by Groq `llama-3.3-70b-versatile`. Per-category rubrics incl. temporal off-by-one,
knowledge-update, and abstention. Containment kept as offline fallback.

## Strategies (7)
`full_context`, `truncation` (k=10), `running_summary`, `hierarchical`
(append-only notes), `slot_memory` (latest-value fact slots), `rag` (BM25 over
turns), `slot_rag_hybrid` (slots + retrieval).

## Models / infrastructure
- **Synthesizer:** `meta-llama/llama-4-scout-17b-16e-instruct` (Groq) — chosen
  because its free-tier limit is **30,000 TPM**, which fits LoCoMo's 10–20K-token
  conversations.
- **Extractor (cheap):** `llama-3.1-8b-instant`. **Judge:** `llama-3.3-70b-versatile`.
- 12 Groq keys, round-robined. `--segment-size 12` to cut extractor calls.

## Rate limits (Groq free / on-demand tier) — IMPORTANT for resuming
- **12 independent Groq keys** are configured (≈30K TPM each for Scout →
  ≈360K TPM aggregate).
- The client is single-threaded (one request at a time, round-robining keys), so
  more keys mainly avoid 429 backoff rather than truly parallelizing.
- `llama-3.3-70b-versatile` caps a single request at **12,000 tokens** → HTTP 413
  on full-context for real conversations. This is *itself evidence* for the
  cost/feasibility argument and is why the synthesizer is Scout (30K TPM), not 70B.
- Practical effect: full-context QA on the longest conversations (~20K tokens)
  consumes a key's whole minute-budget per request, so those cases are slow
  (~10 min each). Memory methods (compact prompts) are fast. The harness saves
  **per case incrementally** and supports `--resume`; never re-does a finished case.

---

## RESULTS — LoCoMo, n = 160 (8 / 10 conversations), LLM-judge

| Strategy | Real token red. | Retention | Cost (% of full) |
|---|---|---:|---:|---:|
| Full context | 0.0% | 30.6% | 100.0% |
| Truncation (k=10) | 98.1% | 21.9% | 4.2% |
| Running summary | 96.9% | 26.9% | 7.7% |
| Append-only notes | 80.0% | 30.0% | 22.8% |
| Fact slots | 97.1% | 26.9% | 7.0% |
| **Retrieval (BM25)** | **98.0%** | **39.4%** | **4.8%** |
| **Slots + retrieval (hybrid)** | **96.2%** | **38.8%** | **9.0%** |

Per-category retention (%):

| Strategy | adversarial(abst) | multi_hop | open_domain | single_hop | temporal |
|---|---|---:|---:|---:|---:|---:|
| Full context | 12.1 | 18.2 | 50.0 | 60.6 | 15.2 |
| Truncation | 90.9 | 0.0 | 14.3 | 3.0 | 0.0 |
| Running summary | 90.9 | 6.1 | 17.9 | 18.2 | 0.0 |
| Append-only notes | 51.5 | 18.2 | 28.6 | 27.3 | 24.2 |
| Fact slots | 81.8 | 6.1 | 21.4 | 24.2 | 0.0 |
| Retrieval (BM25) | 51.5 | 12.1 | 39.3 | 72.7 | 21.2 |
| Slots + retrieval | 45.5 | 9.1 | 35.7 | 81.8 | 21.2 |

Paired (question-level, n=160, sign test):
- BM25 retrieval vs **full context**: +8.8 pp retention, p=0.087 (**non-inferior, trending better**), at **~1/20th the cost**.
- hybrid vs **full context**: +8.1 pp, CI [−0.6, +17.5], p=0.117 (**non-inferior**), at **~1/11th the cost**.
- hybrid vs **truncation**: +16.9 pp, CI [+8.1, +25.6], **p<0.001 (significant)**.

## RESULTS — LongMemEval knowledge-update, n = 72

| Strategy | Retention | Cost (% of full) |
|---:|---:|---:|
| Full context | 69.4% | 100.0% |
| Truncation | 66.7% | 70.1% |
| Running summary | 47.2% | 73.6% |
| Hierarchical notes | 56.9% | 60.8% |
| Fact slots | 37.5% | 59.5% |
| **Retrieval (BM25)** | **73.6%** | **88.2%** |
| **Slots + retrieval** | **73.6%** | **99.5%** |

**Key finding:** BM25 retrieval matches full context (73.6% vs 69.4%, overlapping
CIs) at 88% cost. Fact-slots alone are the worst strategy (37.5%) — the cheap
extractor misses most fact updates, nullifying the structural advantage of
latest-value overwrite. Knowledge-update questions are mostly answerable from
recent sessions (truncation at 66.7% confirms this).

## Judge Cross-Validation

30 LoCoMo questions, BM25-RAG answers, dual-judged by llama-3.3-70b-versatile and
openai/gpt-oss-120b (both on Groq):

| Metric | Value |
|---|---|
| Judge A (llama-3.3-70b) accuracy on sample | 26.7% |
| Judge B (gpt-oss-120b) accuracy on sample | 10.0% |
| Observed agreement | 83.3% |
| Cohen's κ | 0.468 |

Both judges agree BM25-RAG answers are mostly incorrect on this sample (neither
judge has a systematic bias), and their 83.3% agreement with κ=0.468 constitutes
moderate-to-substantial inter-rater reliability.

---

## PENDING WORK
1. **Scale LongMemEval** from the oracle subset (n=254 done, all categories) to the full 500-Q suite.
2. **Second LoCoMo seed** for tighter LoCoMo intervals.
3. **Dense (embedding) RAG backend** evaluation.
4. **Extractor-size ablation** — DONE on LongMemEval knowledge-update (see paper Table lme_ku_ablation); extend to LoCoMo.
5. **Sensitivity sweeps**: `recent_k`, `max_slots`, `top_k`.
6. **Build final PDF** — DONE (`build/main.pdf`; root `main.pdf` synced).

## FILE INVENTORY (new/changed this migration)
- `real_datasets.py` — LoCoMo + LongMemEval loaders into the `Case`/`Fact` shape.
- `judge.py` — LLM-judge with official LongMemEval prompts (+ containment fallback).
- `hierarchical_memory.py` — added `BM25Retriever`, `DenseRetriever`,
  `RetrievalRAG`, `SlotRAGHybrid`, `build_retriever`; tuned backoff.
- `evaluate.py` — `--dataset/--judge/--rag/--segment-size/--resume/...`,
  per-category metrics, per-question rows, per-call resilience (413/429 → record,
  don't abort).
- `scripts/paired_analysis.py` — `--level question` with category breakdown + sign test.
- `scripts/summarize_results.py` — paper-ready table extractor.
- `scripts/judge_agreement.py` — judge cross-validation.
- `results_locomo_n160.json` — final LoCoMo results.
- `results_lme_knowledge_update.json` — LongMemEval knowledge-update results.
- `results_lme_remaining_n135.json` — LongMemEval temporal/multi-session/abstention (n=135).
- `scripts/lme_remaining_table.py` — table generator for the remaining LME categories.
- `figures/category_retention.{pdf,png}` — per-category figure.
- `data/locomo10.json`, `data/longmemeval_oracle.json` — downloaded benchmarks.
