# Next Experiment: Fact-Slot Memory

## Why try this

The current append-only note memory is too verbose and still loses facts. At <=56
messages it costs about the same as full context while retaining less. The next
approach should change the memory representation, not just the model.

## Hypothesis

A latest-value fact table can beat append-only notes because it stores only:

```text
subject | predicate | value | timestamp
```

Examples:

```text
Project Atlas | budget | 120,000 | t=2
Project Atlas | owner | Priya | t=1
```

This should help because:

1. updates overwrite stale values by key;
2. prompts contain compact current facts instead of verbose segment notes;
3. retrieval can select query-relevant slots;
4. compaction is natural because the table is already bounded by unique facts.

## What was implemented

- `SlotFact` and `SlotMemoryStore` in `hierarchical_memory.py`.
- `SlotMemory` strategy in `hierarchical_memory.py`.
- `SLOT_EXTRACT_SYSTEM` for live LLM slot extraction.
- `--slot-memory` option in `evaluate.py`.
- Raw per-case metrics in future `evaluate.py` outputs.
- Configurable paired tests in `scripts/paired_analysis.py`.

## Offline smoke test

```bash
.venv/bin/python evaluate.py --offline --scale tiny --slot-memory --out /tmp/nodemem_slot_tiny.json
.venv/bin/python scripts/paired_analysis.py /tmp/nodemem_slot_tiny.json \
  --method slot_memory \
  --baselines full_context hierarchical truncation \
  --out /tmp/nodemem_slot_paired.json
```

Offline retention is not scientific because the mock answerer is heuristic. Use
it only to check plumbing and token accounting.

## Recommended live pilot

Status: completed for seeds 9 and 10 without ablation.

```bash
scripts/run_groq_experiment.sh \
  --provider groq \
  --scale small \
  --seed 9 \
  --slot-memory \
  --out results_live_slot_seed9.json

.venv/bin/python scripts/paired_analysis.py results_live_slot_seed9.json \
  --method slot_memory \
  --baselines full_context hierarchical truncation running_summary \
  --out results_live_slot_seed9_paired.json
```

Aggregate command used:

```bash
.venv/bin/python scripts/aggregate_results.py \
  results_live_slot_seed9.json results_live_slot_seed10.json \
  --out results_live_slot_aggregate.json

.venv/bin/python scripts/paired_analysis.py results_live_slot_aggregate.json \
  --method slot_memory \
  --baselines full_context hierarchical truncation running_summary \
  --out results_live_slot_paired.json
```

## Live result so far

Two live Groq seeds completed: `results_live_slot_seed9.json` and
`results_live_slot_seed10.json` (16/16 cases total). The aggregate is
`results_live_slot_aggregate.json`; paired comparisons are in
`results_live_slot_paired.json`.

| Strategy | Real token reduction | Retention | Update retention | Cost of full context |
|---|---:|---:|---:|---:|
| Full context | 0.0% | 95.6% | 93.8% | 100.0% |
| Truncation | 50.8% | 35.4% | 37.5% | 70.2% |
| Running summary | 20.1% | 100.0% | 100.0% | 159.8% |
| Append-only hierarchical notes | 19.2% | 75.7% | 87.5% | 103.9% |
| Fact-slot memory | 40.9% | 85.2% | 71.9% | 89.1% |

This is promising because fact-slot memory dominates append-only hierarchical
notes on the two key axes: higher retention and lower cost. It also creates a
real trade-off against full context: lower cost but lower retention. The weakest
point is update retention, where append-only notes did better in this run.

## Success criteria

The approach is worth writing up only if live paired analysis shows:

1. `slot_memory` has lower cost than full context with a CI mostly below 100%;
2. retention is much closer to full context than truncation;
3. `slot_memory` dominates append-only `hierarchical` on retention or cost;
4. update retention improves, showing latest-value overwrites are working.

The first two live seeds satisfy criteria 2 and 3, partially satisfy criterion 1
(mean cost is lower, but uncertainty remains), and fail criterion 4. The next
engineering target should be better update extraction / overwrite handling.

If it still costs about the same as full context or loses many facts, the paper
should remain a negative/cautionary result.
