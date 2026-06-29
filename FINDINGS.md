# Findings: Persistent Note-Based Memory

Date: 2026-06-29

## Live Groq result already measured in this repository

The current live result file is `results_live.json`, produced with:

- Extractor: `llama-3.1-8b-instant`
- Synthesizer: `llama-3.3-70b-versatile`
- Provider: Groq OpenAI-compatible API
- Scale: `small`, seed `7`
- Completed cases: 7 of 8 requested cases

| Strategy | Real token reduction | Retention | Update retention | Cost of full context |
|---|---:|---:|---:|---:|
| Full context | 0.0% | 100.0% | 100.0% | 100.0% |
| Truncation (`k=10`) | 46.9% | 42.9% | 42.9% | 74.5% |
| Running summary | 17.2% | 97.1% | 100.0% | 165.2% |
| Hierarchical notes | 15.4% | 84.5% | 85.7% | 108.4% |

Main finding: at <=56 messages, hierarchical notes are not yet a large raw-token
win. The publishable claim is a retention/cost trade-off: much higher retention
than truncation while avoiding the high maintenance cost of running summary.

## Cascade ablation

| Extractor | Downstream retention |
|---|---:|
| `llama-3.1-8b-instant` | 84.5% |
| `llama-3.3-70b-versatile` | 70.5% |

Finding: the cheap 8B extractor produced better downstream retention than the
70B extractor under the same notes schema and scoring protocol. This supports
the model-cascading thesis for the memory-write step.

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

## Live rerun status on this device

I checked the local terminal environment and `.env`; `GROQ_API_KEY` is not set,
so a new live Groq run cannot be executed on this device yet. Once a key is
added, run:

```bash
scripts/run_groq_experiment.sh
```

The script defaults to:

```bash
--provider groq --scale small --ablation --out results_live_rerun.json
```

The evaluator now loads `.env` automatically and fails early with a clear message
instead of producing a confusing SDK error.

## Publication-safe interpretation

- Do not claim 60--80% live token reduction at short horizons.
- Claim measured retention/cost trade-off at <=56 messages.
- Claim cheap-extractor ablation support for memory-write model cascading.
- Treat the 120--480 message curve as deterministic cost-shape evidence only.
- State that live long-horizon retention validation remains future work.
