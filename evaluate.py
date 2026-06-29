"""
evaluate.py -- evaluation harness for persistent note-based memory.

Measured outputs
----------------
* Token reduction (PRIMARY, REAL): per-query reduction in the input tokens the
  large model actually receives, from the Anthropic ``usage`` returned by real
  completions.  Conservative -- a single query, not the cumulative growth.
* Token reduction (SECONDARY): cumulative large-model input tokens over the whole
  conversation, counted with a consistent local tokenizer (tiktoken/cl100k).
  The same tokenizer is applied to every strategy so the *ratio* is faithful;
  absolute counts are tokenizer-estimated and labelled as such.
* Retention: QA accuracy on planted facts, with a fact-UPDATE breakdown.
* Cost (% of full-context baseline): synth token flow x price + maintenance.
  Reported for the INTENDED cascade (cheap extractor priced at ``--price-extract``)
  using measured token flows.
* Latency: mean per-response wall-clock; hierarchical = extractor + synth.

Notes on the endpoint used for the paper run: the available router served only
``claude-opus-4-8`` reliably and did not implement ``count_tokens``.  The harness
therefore runs both roles on the synth model and prices the extractor at the
cheap-model rate to show the intended cascade economics; with a second model
available, pass distinct --extract-model / --synth-model and --ablation.

Usage:
  python evaluate.py --offline --scale tiny                 # no API
    python evaluate.py --offline --scale long --compact-notes 8 \
                --out results_long_offline.json                     # long stress smoke test
  python evaluate.py --scale tiny --synth-model claude-opus-4-8 \
        --extract-model claude-opus-4-8 --throttle 3.4 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time

from datasets import build_suite, Case
from hierarchical_memory import (
    LLMClient, OpenAICompatClient, MockLLMClient, NotesFile, Note,
    FullContext, Truncation, RunningSummary, Hierarchical,
    EXTRACT_SYSTEM, _safe_json,
)
from token_utils import (
    cost_usd, price_for, reduction_pct, fraction_of_baseline_pct,
    count_tokens_local, assemble_text,
)

RECENT_K = 6
SEGMENT_SIZE = 6
TRUNCATE_K = 10


def load_dotenv_if_present(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from ``.env`` without printing secrets."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# --------------------------------------------------------------------------- #
# Retention judging
# --------------------------------------------------------------------------- #

def normalize(s: str) -> str:
    s = s.lower().replace(",", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def answer_correct(model_output: str, gold: str) -> bool:
    out = normalize(model_output)
    g = normalize(gold)
    return bool(g) and all(tok in out for tok in g.split())


# --------------------------------------------------------------------------- #
# Conversation processing
# --------------------------------------------------------------------------- #

def segment_messages(messages, size):
    for i in range(0, len(messages), size):
        yield i // size, messages[i:i + size]


def build_notes(client, case: Case, extract_model: str):
    """Cheap-extractor pass: one note per segment. Returns (NotesFile, ex_in, ex_out, latencies)."""
    notes = NotesFile()
    ex_in = ex_out = 0
    lat = []
    for seg_id, seg in segment_messages(case.messages, SEGMENT_SIZE):
        convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in seg)
        c = client.complete(extract_model, EXTRACT_SYSTEM,
                            [{"role": "user", "content": convo}], max_tokens=512)
        data = _safe_json(c.text)
        notes.append(Note(timestamp=seg_id, segment_id=seg_id,
                          entities=data.get("entities", []) or [],
                          decisions=data.get("decisions", []) or [],
                          facts=data.get("facts", []) or [],
                          goals=data.get("goals", []) or [],
                          action_items=data.get("action_items", []) or []))
        ex_in += c.input_tokens; ex_out += c.output_tokens; lat.append(c.latency_s)
    return notes, ex_in, ex_out, lat


def build_summary(client, case: Case, synth_model: str):
    """Running-summary baseline: a single faithful summary over the whole history
    (one call -- conserves rate-limited calls). Returns (summary, in, out)."""
    convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in case.messages)
    prompt = ("Summarize this conversation, preserving every entity, decision, "
              "fact (numbers/dates/IDs verbatim), goal, and action item. If a "
              "fact was later corrected, keep only the corrected value.\n\n" + convo)
    c = client.complete(synth_model, "You write faithful, complete summaries.",
                        [{"role": "user", "content": prompt}], max_tokens=700)
    return c.text, c.input_tokens, c.output_tokens


def compacted_notes_copy(notes: NotesFile, max_notes: int) -> NotesFile:
    """Replay notes into a new store while applying deterministic compaction."""
    out = NotesFile()
    for note in notes.notes:
        out.notes.append(note)
        out.compact(max_notes=max_notes)
    return out


def token_curve(case: Case, notes: NotesFile, summary: str, compact_notes: int = 0):
    """Cumulative large-model INPUT tokens per strategy (local tiktoken).

    Notes/summary become progressively available as segments complete.
    If ``compact_notes`` is non-zero, the hierarchical notes store is compacted
    after each completed segment to model bounded long-horizon memory.
    """
    strategies = {
        "full_context": FullContext(),
        "truncation": Truncation(TRUNCATE_K),
        "running_summary": RunningSummary(RECENT_K),
        "hierarchical": Hierarchical(RECENT_K),
    }
    inc_notes = NotesFile()
    seg_index = 0
    n_segments = (len(case.messages) + SEGMENT_SIZE - 1) // SEGMENT_SIZE

    totals = {k: 0 for k in strategies}
    history = []
    for turn_i, m in enumerate(case.messages):
        if m["role"] == "user":
            q = m["content"]
            sysf, msgsf = strategies["full_context"].build_context(history, q)
            totals["full_context"] += count_tokens_local(assemble_text(sysf, msgsf))
            syst, msgst = strategies["truncation"].build_context(history, q)
            totals["truncation"] += count_tokens_local(assemble_text(syst, msgst))
            strategies["running_summary"].summary = summary if seg_index > 0 else ""
            syss, msgss = strategies["running_summary"].build_context(history, q)
            totals["running_summary"] += count_tokens_local(assemble_text(syss, msgss))
            strategies["hierarchical"].notes = inc_notes
            sysh, msgsh = strategies["hierarchical"].build_context(history, q)
            totals["hierarchical"] += count_tokens_local(assemble_text(sysh, msgsh))
        history.append(m)
        while seg_index < n_segments and (turn_i + 1) >= (seg_index + 1) * SEGMENT_SIZE:
            if seg_index < len(notes.notes):
                inc_notes.notes.append(notes.notes[seg_index])
                if compact_notes:
                    inc_notes.compact(max_notes=compact_notes)
            seg_index += 1
    return totals


def run_qa(client, case: Case, synth_model: str, notes: NotesFile, summary: str):
    """Answer each planted question under each strategy with the real synth model."""
    strategies = {
        "full_context": FullContext(),
        "truncation": Truncation(TRUNCATE_K),
        "running_summary": RunningSummary(RECENT_K),
        "hierarchical": Hierarchical(RECENT_K),
    }
    strategies["running_summary"].summary = summary
    strategies["hierarchical"].notes = notes
    out = {k: {"correct": 0, "total": 0, "upd_correct": 0, "upd_total": 0,
               "in": 0, "out": 0, "lat": []} for k in strategies}
    for fact in case.facts:
        for name, strat in strategies.items():
            system, msgs = strat.build_context(case.messages, fact.question)
            c = client.complete(synth_model, system, msgs, max_tokens=120)
            ok = answer_correct(c.text, fact.answer)
            st = out[name]
            st["total"] += 1; st["correct"] += int(ok)
            st["in"] += c.input_tokens; st["out"] += c.output_tokens; st["lat"].append(c.latency_s)
            if fact.updated:
                st["upd_total"] += 1; st["upd_correct"] += int(ok)
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def bootstrap_ci(values, iters=2000, seed=13):
    if not values:
        return (0.0, 0.0)
    import random
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def summarize(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0], "n": 0}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    lo, hi = bootstrap_ci(values)
    return {"mean": round(mean, 2), "std": round(std, 2),
            "ci95": [round(lo, 2), round(hi, 2)], "n": len(values)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--provider", choices=["anthropic", "groq"], default="anthropic")
    ap.add_argument("--scale", choices=["tiny", "small", "paper", "long"], default="tiny")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--synth-model", default=None)
    ap.add_argument("--extract-model", default=None,
                    help="model actually CALLED for extraction")
    ap.add_argument("--price-extract", default=None,
                    help="price tier applied to the extractor in the cascade cost projection")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="min seconds between API calls (for rate-limited proxies)")
    ap.add_argument("--ablation", action="store_true",
                    help="also build notes with the synth model (expensive extractor) and compare retention")
    ap.add_argument("--compact-notes", type=int, default=0,
                    help="compact hierarchical notes to at most this many entries after each segment")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    load_dotenv_if_present()

    # Provider defaults
    if args.provider == "groq":
        args.synth_model = args.synth_model or "llama-3.3-70b-versatile"
        args.extract_model = args.extract_model or "llama-3.1-8b-instant"
        args.price_extract = args.price_extract or "llama-3.1-8b-instant"
    else:
        args.synth_model = args.synth_model or "claude-opus-4-8"
        args.extract_model = args.extract_model or "claude-opus-4-8"
        args.price_extract = args.price_extract or "claude-haiku-4-5"

    if args.offline:
        client = MockLLMClient()
    elif args.provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            raise SystemExit(
                "GROQ_API_KEY is required for live Groq experiments. "
                "Set it in the environment or add GROQ_API_KEY=... to .env."
            )
        client = OpenAICompatClient(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            min_interval_s=args.throttle)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is required for live Anthropic experiments. "
                "Set it in the environment or add ANTHROPIC_API_KEY=... to .env."
            )
        client = LLMClient(min_interval_s=args.throttle)
    suite = build_suite(seed=args.seed, scale=args.scale)
    if args.max_cases:
        suite = suite[: args.max_cases]

    strat_names = ["full_context", "truncation", "running_summary", "hierarchical"]
    acc = {s: {"tok_red_real": [], "tok_red_cum": [], "cost_pct": [],
               "retention": [], "upd_retention": [], "synth_lat": []} for s in strat_names}
    ablation = {"cheap": [], "expensive": []}
    extract_cost_actual = extract_cost_cascade = summary_cost = 0.0
    ex_lat_all, hier_lat, full_lat = [], [], []

    opus_in, opus_out = price_for(args.synth_model)
    px_in, px_out = price_for(args.price_extract)        # cascade extractor price
    px_in_a, px_out_a = price_for(args.extract_model)    # actual extractor price

    t0 = time.time()
    done_cases = 0

    def build_results():
        res = {
            "config": {
                "offline": args.offline, "provider": args.provider, "scale": args.scale,
                "n_cases_requested": len(suite), "n_cases_completed": done_cases,
                "synth_model": args.synth_model, "extract_model_called": args.extract_model,
                "extract_price_tier": args.price_extract,
                "recent_k": RECENT_K, "truncate_k": TRUNCATE_K, "segment_size": SEGMENT_SIZE,
                "compact_notes": args.compact_notes,
                "token_counter": "tiktoken/cl100k (cumulative); provider usage (per-query real)",
                "seed": args.seed, "elapsed_s": round(time.time() - t0, 1),
            },
            "strategies": {s: {
                "token_reduction_real_pct": summarize(acc[s]["tok_red_real"]),
                "token_reduction_cumulative_pct": summarize(acc[s]["tok_red_cum"]),
                "cost_pct_of_baseline": summarize(acc[s]["cost_pct"]),
                "retention_pct": summarize(acc[s]["retention"]),
                "fact_update_retention_pct": summarize(acc[s]["upd_retention"]),
            } for s in strat_names},
            "cost_breakdown_usd": {
                "extractor_total_cascade_price": round(extract_cost_cascade, 6),
                "extractor_total_actual_price": round(extract_cost_actual, 6),
                "summary_maintenance_total": round(summary_cost, 6),
            },
            "latency_s": {
                "full_response": summarize(full_lat),
                "hier_response": summarize(hier_lat),
                "extractor_call": summarize(ex_lat_all),
                "overhead_mean": round((statistics.fmean(hier_lat) if hier_lat else 0)
                                       - (statistics.fmean(full_lat) if full_lat else 0), 3),
            },
        }
        if ablation["cheap"]:
            res["ablation_cheap_vs_expensive_extractor"] = {
                "cheap_retention_pct": summarize(ablation["cheap"]),
                "expensive_retention_pct": summarize(ablation["expensive"]),
            }
        return res

    for ci, case in enumerate(suite):
        print(f"[{ci+1}/{len(suite)}] {case.id} ({len(case.messages)} msgs, "
              f"{len(case.facts)} facts)  t={time.time()-t0:.0f}s", file=sys.stderr)
        try:
            raw_notes, ex_in, ex_out, exlat = build_notes(client, case, args.extract_model)
            notes = compacted_notes_copy(raw_notes, args.compact_notes) if args.compact_notes else raw_notes
            summary, sm_in, sm_out = build_summary(client, case, args.synth_model)
            curve = token_curve(case, raw_notes, summary, compact_notes=args.compact_notes)
            qa = run_qa(client, case, args.synth_model, notes, summary)
        except Exception as e:
            print(f"  !! case {case.id} failed ({type(e).__name__}: {str(e)[:80]}); skipping",
                  file=sys.stderr)
            continue

        ex_lat_all += exlat
        extract_cost_cascade += cost_usd(args.price_extract, ex_in, ex_out)
        extract_cost_actual += cost_usd(args.extract_model, ex_in, ex_out)
        summary_cost += cost_usd(args.synth_model, sm_in, sm_out)

        base_cum = curve["full_context"]
        base_real = qa["full_context"]["in"]
        base_cost = base_cum * opus_in / 1000.0

        for s in strat_names:
            # cost (% of baseline): synth token flow (cum, tiktoken) x price + maintenance
            maint = 0.0
            if s == "hierarchical":
                maint = (ex_in * px_in + ex_out * px_out) / 1000.0
            elif s == "running_summary":
                maint = (sm_in * opus_in + sm_out * opus_out) / 1000.0
            cost_s = curve[s] * opus_in / 1000.0 + maint
            acc[s]["cost_pct"].append(fraction_of_baseline_pct(base_cost, cost_s))
            acc[s]["tok_red_cum"].append(reduction_pct(base_cum, curve[s]))
            acc[s]["tok_red_real"].append(reduction_pct(base_real, qa[s]["in"]))
            acc[s]["retention"].append(
                qa[s]["correct"] / qa[s]["total"] * 100 if qa[s]["total"] else 0.0)
            if qa[s]["upd_total"]:
                acc[s]["upd_retention"].append(qa[s]["upd_correct"] / qa[s]["upd_total"] * 100)
            if qa[s]["lat"]:
                acc[s]["synth_lat"].append(statistics.fmean(qa[s]["lat"]))

        if qa["hierarchical"]["lat"]:
            hier_lat.append(statistics.fmean(qa["hierarchical"]["lat"]))
        if qa["full_context"]["lat"]:
            full_lat.append(statistics.fmean(qa["full_context"]["lat"]))

        if args.ablation and not args.offline and args.extract_model != args.synth_model:
            try:
                ablation["cheap"].append(acc["hierarchical"]["retention"][-1])
                raw_notes_x, *_ = build_notes(client, case, args.synth_model)
                notes_x = (compacted_notes_copy(raw_notes_x, args.compact_notes)
                           if args.compact_notes else raw_notes_x)
                qa_x = run_qa(client, case, args.synth_model, notes_x, summary)
                ablation["expensive"].append(
                    qa_x["hierarchical"]["correct"] / qa_x["hierarchical"]["total"] * 100
                    if qa_x["hierarchical"]["total"] else 0.0)
            except Exception as e:
                ablation["cheap"].pop()  # keep cheap/expensive aligned
                print(f"  !! ablation for {case.id} failed ({type(e).__name__}); skipping",
                      file=sys.stderr)

        done_cases += 1
        # Incremental save so a later failure never wipes completed work.
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(build_results(), f, indent=2)
            f.write("\n")

    results = build_results()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out} ({done_cases}/{len(suite)} cases, "
          f"{time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
