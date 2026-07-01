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
from real_datasets import load_real_suite
from judge import judge_answer, containment_correct
from hierarchical_memory import (
    LLMClient, OpenAICompatClient, MockLLMClient, NotesFile, Note,
    FullContext, Truncation, RunningSummary, Hierarchical, SlotMemory, SlotMemoryStore,
    RetrievalRAG, SlotRAGHybrid, build_retriever,
    EXTRACT_SYSTEM, SLOT_EXTRACT_SYSTEM, _safe_json,
)
from token_utils import (
    cost_usd, price_for, reduction_pct, fraction_of_baseline_pct,
    count_tokens_local, assemble_text,
)

RECENT_K = 6
SEGMENT_SIZE = 6
TRUNCATE_K = 10
RAG_TOP_K = 5


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


def env_keys(*names: str) -> list[str]:
    """Return unique API keys from environment variables without logging them."""
    keys: list[str] = []
    seen = set()
    for name in names:
        raw = os.environ.get(name, "")
        for candidate in raw.split(","):
            key = candidate.strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


# Provider registry: maps a provider name to its OpenAI-compatible base URL and
# the environment variables that hold its key(s). Anthropic is handled separately.
_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", ("GROQ_API_KEY", "GROQ_API_KEYS")),
    "mistral": ("https://router.bynara.id/v1",
                ("BYNARA_API_KEY", "BYNARA_API_KEYS", "MISTRAL_API_KEY", "MISTRAL_API_KEYS")),
    "nara": ("https://router.bynara.id/v1",
             ("NARA_API_KEY", "NARA_API_KEYS", "BYNARA_API_KEY", "BYNARA_API_KEYS")),
}


def build_provider_client(provider: str, throttle: float):
    """Build a client for one provider role (extractor / synthesizer / judge)."""
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is required for the anthropic provider.")
        return LLMClient(min_interval_s=throttle)
    if provider not in _PROVIDERS:
        raise SystemExit(f"unknown provider: {provider!r}")
    base_url, key_names = _PROVIDERS[provider]
    keys = env_keys(*key_names)
    if not keys:
        raise SystemExit(
            f"{' or '.join(key_names)} is required for the {provider} provider. "
            f"Add it (comma-separated for multiple) to .env.")
    return OpenAICompatClient(api_keys=keys, base_url=base_url, min_interval_s=throttle)


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


def score_answer(client, judge_model, use_judge, fact, text):
    """Score one model answer against a fact, via LLM judge or containment."""
    if use_judge:
        return judge_answer(client, judge_model, fact.category, fact.question,
                            fact.answer, text, fact.is_abstention)
    return containment_correct(text, fact.answer, fact.is_abstention)


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


def build_slot_memory(client, case: Case, extract_model: str):
    """Fact-slot pass: latest-value table keyed by subject/predicate."""
    store = SlotMemoryStore()
    ex_in = ex_out = 0
    lat = []
    for seg_id, seg in segment_messages(case.messages, SEGMENT_SIZE):
        convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in seg)
        c = client.complete(extract_model, SLOT_EXTRACT_SYSTEM,
                            [{"role": "user", "content": convo}], max_tokens=512)
        data = _safe_json(c.text)
        before = len(store.events)
        store.ingest_data(data, timestamp=seg_id, segment_id=seg_id)
        if len(store.events) == before:
            # Deterministic fallback for the released synthetic grammar; useful
            # when an extractor emits strings or malformed JSON.
            store.ingest_text(convo, timestamp=seg_id, segment_id=seg_id)
        ex_in += c.input_tokens; ex_out += c.output_tokens; lat.append(c.latency_s)
    return store, ex_in, ex_out, lat


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


def token_curve(case: Case, notes: NotesFile, summary: str, compact_notes: int = 0,
                slot_store: SlotMemoryStore | None = None, retriever=None):
    """Cumulative large-model INPUT tokens per strategy (local tiktoken).

    Notes/summary become progressively available as segments complete.
    If ``compact_notes`` is non-zero, the hierarchical notes store is compacted
    after each completed segment to model bounded long-horizon memory.

    Retrieval-augmented strategies retrieve a bounded ``top_k`` set each turn, so
    their per-query token count is approximately constant; we use the
    conversation-level retriever as a stand-in for memory-so-far retrieval.
    """
    strategies = {
        "full_context": FullContext(),
        "truncation": Truncation(TRUNCATE_K),
        "running_summary": RunningSummary(RECENT_K),
        "hierarchical": Hierarchical(RECENT_K),
    }
    if slot_store is not None:
        strategies["slot_memory"] = SlotMemory(RECENT_K)
    if retriever is not None:
        strategies["rag"] = RetrievalRAG(RECENT_K, retriever, top_k=RAG_TOP_K)
        if slot_store is not None:
            strategies["slot_rag_hybrid"] = SlotRAGHybrid(
                RECENT_K, slot_store, retriever, top_k=RAG_TOP_K)
    inc_notes = NotesFile()
    inc_slots = SlotMemoryStore()
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
            if slot_store is not None:
                strategies["slot_memory"].store = inc_slots
                sysslot, msgsslot = strategies["slot_memory"].build_context(history, q)
                totals["slot_memory"] += count_tokens_local(assemble_text(sysslot, msgsslot))
            if retriever is not None:
                sysr, msgsr = strategies["rag"].build_context(history, q)
                totals["rag"] += count_tokens_local(assemble_text(sysr, msgsr))
                if slot_store is not None:
                    strategies["slot_rag_hybrid"].store = inc_slots
                    syshy, msgshy = strategies["slot_rag_hybrid"].build_context(history, q)
                    totals["slot_rag_hybrid"] += count_tokens_local(assemble_text(syshy, msgshy))
        history.append(m)
        while seg_index < n_segments and (turn_i + 1) >= (seg_index + 1) * SEGMENT_SIZE:
            if seg_index < len(notes.notes):
                inc_notes.notes.append(notes.notes[seg_index])
                if compact_notes:
                    inc_notes.compact(max_notes=compact_notes)
            if slot_store is not None:
                for fact in slot_store.facts_for_segment(seg_index):
                    inc_slots.upsert(fact)
            seg_index += 1
    return totals


def run_qa(synth_client, case: Case, synth_model: str, notes: NotesFile, summary: str,
           slot_store: SlotMemoryStore | None = None, retriever=None,
           judge_client=None, judge_model: str | None = None, use_judge: bool = False):
    """Answer each planted question under each strategy with the real synth model.

    ``synth_client`` generates answers; ``judge_client`` (defaults to the synth
    client) scores them, so the synthesizer can be a different provider/model
    than the judge while keeping scoring fixed across runs.
    """
    judge_client = judge_client or synth_client
    strategies = {
        "full_context": FullContext(),
        "truncation": Truncation(TRUNCATE_K),
        "running_summary": RunningSummary(RECENT_K),
        "hierarchical": Hierarchical(RECENT_K),
    }
    if slot_store is not None:
        strategies["slot_memory"] = SlotMemory(RECENT_K, slot_store)
    if retriever is not None:
        strategies["rag"] = RetrievalRAG(RECENT_K, retriever, top_k=RAG_TOP_K)
        if slot_store is not None:
            strategies["slot_rag_hybrid"] = SlotRAGHybrid(
                RECENT_K, slot_store, retriever, top_k=RAG_TOP_K)
    strategies["running_summary"].summary = summary
    strategies["hierarchical"].notes = notes
    out = {k: {"correct": 0, "total": 0, "upd_correct": 0, "upd_total": 0,
               "in": 0, "out": 0, "lat": [], "cat_correct": {}, "cat_total": {},
               "errors": 0} for k in strategies}
    qa_rows = []
    for fact in case.facts:
        row = {"key": fact.key, "category": fact.category or "default",
               "is_abstention": fact.is_abstention, "updated": fact.updated,
               "by_strategy": {}}
        for name, strat in strategies.items():
            system, msgs = strat.build_context(case.messages, fact.question)
            try:
                c = synth_client.complete(synth_model, system, msgs, max_tokens=160)
                errored = False
            except Exception as e:
                # A hard failure (e.g., 413 request-too-large for full context on a
                # rate-limited tier) is recorded as an incorrect answer for THIS
                # strategy/question rather than aborting the whole case.
                print(f"     ~ {name} call failed on {fact.key} "
                      f"({type(e).__name__}: {str(e)[:60]})", file=sys.stderr)
                from hierarchical_memory import Completion
                c = Completion(text="", input_tokens=0, output_tokens=0, latency_s=0.0)
                errored = True
            ok = False if errored else score_answer(
                judge_client, judge_model or synth_model, use_judge, fact, c.text)
            st = out[name]
            st["total"] += 1; st["correct"] += int(ok)
            st["in"] += c.input_tokens; st["out"] += c.output_tokens
            if not errored:
                st["lat"].append(c.latency_s)
            else:
                st["errors"] += 1
            cat = fact.category or "default"
            st["cat_total"][cat] = st["cat_total"].get(cat, 0) + 1
            st["cat_correct"][cat] = st["cat_correct"].get(cat, 0) + int(ok)
            if fact.updated:
                st["upd_total"] += 1; st["upd_correct"] += int(ok)
            row["by_strategy"][name] = {"correct": int(ok),
                                        "real_input_tokens": c.input_tokens,
                                        "errored": errored}
        qa_rows.append(row)
    return out, qa_rows


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
    ap.add_argument("--provider", choices=["anthropic", "groq", "nara"], default="nara")
    ap.add_argument("--synth-provider", choices=["groq", "mistral", "nara", "anthropic"], default=None,
                    help="provider for the SYNTHESIZER only (default: same as --provider). "
                         "Use 'mistral' to test a stronger second model family; the extractor "
                         "and judge stay on --provider for comparable scoring.")
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
    ap.add_argument("--slot-memory", action="store_true",
                    help="also evaluate a latest-value fact-slot memory strategy")
    ap.add_argument("--dataset", choices=["synthetic", "locomo", "longmemeval"],
                    default="synthetic", help="evaluation substrate")
    ap.add_argument("--data-path", default=None,
                    help="path to the real dataset json (LoCoMo/LongMemEval)")
    ap.add_argument("--max-qa-per-conv", type=int, default=0,
                    help="LoCoMo: cap QA per conversation (category-balanced)")
    ap.add_argument("--max-instances", type=int, default=0,
                    help="LongMemEval: cap number of question/history instances")
    ap.add_argument("--lme-categories", nargs="+", default=None,
                    help="LongMemEval: restrict to these categories (e.g. knowledge_update)")
    ap.add_argument("--judge", choices=["llm", "containment"], default="containment",
                    help="answer scoring: LLM-as-judge (real datasets) or token containment")
    ap.add_argument("--judge-model", default=None,
                    help="model used for LLM-as-judge scoring (defaults to synth model)")
    ap.add_argument("--rag", action="store_true",
                    help="also evaluate a retrieval-augmented (BM25/dense) baseline")
    ap.add_argument("--rag-backend", choices=["bm25", "dense"], default="bm25",
                    help="retriever backend for the RAG strategies")
    ap.add_argument("--segment-size", type=int, default=0,
                    help="override extractor segment size (default 6; larger = fewer extractor calls)")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases already present in --out and continue (rate-limit friendly)")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    load_dotenv_if_present()
    if args.segment_size:
        global SEGMENT_SIZE
        SEGMENT_SIZE = args.segment_size

    # Provider defaults
    synth_provider = args.synth_provider or args.provider
    if args.provider == "groq":
        args.extract_model = args.extract_model or "llama-3.1-8b-instant"
        args.price_extract = args.price_extract or "llama-3.1-8b-instant"
    elif args.provider == "nara":
        args.extract_model = args.extract_model or "mimo-v2.5-free"
        args.price_extract = args.price_extract or "mimo-v2.5-free"
    else:
        args.extract_model = args.extract_model or "claude-opus-4-8"
        args.price_extract = args.price_extract or "claude-haiku-4-5"
    if not args.synth_model:
        args.synth_model = {
            "groq": "llama-3.3-70b-versatile",
            "mistral": "mistral-large",
            "nara": "mistral-large",
            "anthropic": "claude-opus-4-8",
        }[synth_provider]

    if args.offline:
        client = MockLLMClient()
        synth_client = client
    else:
        client = build_provider_client(args.provider, args.throttle)
        synth_client = (client if synth_provider == args.provider
                        else build_provider_client(synth_provider, args.throttle))
    # The extractor and the judge run on the main `client`; only the synthesizer
    # (answer generation + running-summary baseline) uses `synth_client`.
    judge_client = client

    if args.dataset == "locomo":
        path = args.data_path or "data/locomo10.json"
        suite = load_real_suite("locomo", path=path,
                                max_qa_per_conv=args.max_qa_per_conv, seed=args.seed)
    elif args.dataset == "longmemeval":
        path = args.data_path or "data/longmemeval_s.json"
        suite = load_real_suite("longmemeval", path=path,
                                max_instances=args.max_instances, seed=args.seed,
                                categories=args.lme_categories)
    else:
        suite = build_suite(seed=args.seed, scale=args.scale)
    if args.max_cases:
        suite = suite[: args.max_cases]

    use_judge = (args.judge == "llm") and not args.offline
    judge_model = args.judge_model or args.synth_model

    strat_names = ["full_context", "truncation", "running_summary", "hierarchical"]
    if args.slot_memory:
        strat_names.append("slot_memory")
    if args.rag:
        strat_names.append("rag")
        if args.slot_memory:
            strat_names.append("slot_rag_hybrid")
    acc = {s: {"tok_red_real": [], "tok_red_cum": [], "cost_pct": [],
               "retention": [], "upd_retention": [], "synth_lat": []} for s in strat_names}
    cat_acc = {s: {} for s in strat_names}   # per-category {cat: {correct,total}}
    question_rows = []                        # flat per-question records for paired stats
    ablation = {"cheap": [], "expensive": []}
    case_rows = []
    extract_cost_actual = extract_cost_cascade = summary_cost = 0.0
    ex_lat_all, hier_lat, full_lat = [], [], []

    # --resume: reload completed cases from --out and pre-seed accumulators so a
    # rate-limited run can be continued with one command without redoing work.
    completed_ids = set()
    if args.resume and os.path.exists(args.out):
        try:
            prev = json.load(open(args.out, encoding="utf-8"))
        except Exception:
            prev = None
        if prev and prev.get("case_metrics"):
            case_rows = list(prev["case_metrics"])
            question_rows = list(prev.get("question_metrics", []))
            completed_ids = {row["case_id"] for row in case_rows}
            for row in case_rows:
                for s in strat_names:
                    st = row.get("strategies", {}).get(s)
                    if not st:
                        continue
                    acc[s]["cost_pct"].append(st["cost_pct_of_baseline"])
                    acc[s]["tok_red_cum"].append(st["token_reduction_cumulative_pct"])
                    acc[s]["tok_red_real"].append(st["token_reduction_real_pct"])
                    acc[s]["retention"].append(st["retention_pct"])
                    if st.get("fact_update_retention_pct") is not None:
                        acc[s]["upd_retention"].append(st["fact_update_retention_pct"])
                    if st.get("latency_s") is not None:
                        acc[s]["synth_lat"].append(st["latency_s"])
                h = row.get("strategies", {}).get("hierarchical", {}).get("latency_s")
                if h is not None:
                    hier_lat.append(h)
                f = row.get("strategies", {}).get("full_context", {}).get("latency_s")
                if f is not None:
                    full_lat.append(f)
            for qrow in question_rows:
                cat = qrow.get("category", "default")
                for s in strat_names:
                    bs = qrow.get("by_strategy", {}).get(s)
                    if not bs:
                        continue
                    bucket = cat_acc[s].setdefault(cat, {"correct": 0, "total": 0})
                    bucket["total"] += 1
                    bucket["correct"] += int(bs.get("correct", 0))
            cb = prev.get("cost_breakdown_usd", {})
            extract_cost_cascade = cb.get("extractor_total_cascade_price", 0.0)
            extract_cost_actual = cb.get("extractor_total_actual_price", 0.0)
            summary_cost = cb.get("summary_maintenance_total", 0.0)
            print(f"Resuming: {len(completed_ids)} cases already done, "
                  f"{len(question_rows)} questions preserved.", file=sys.stderr)

    opus_in, opus_out = price_for(args.synth_model)
    px_in, px_out = price_for(args.price_extract)        # cascade extractor price
    px_in_a, px_out_a = price_for(args.extract_model)    # actual extractor price

    t0 = time.time()
    done_cases = len(completed_ids)

    def build_results():
        res = {
            "config": {
                "offline": args.offline, "provider": args.provider, "scale": args.scale,
                "synth_provider": synth_provider,
                "dataset": args.dataset, "judge": args.judge if not args.offline else "containment",
                "judge_model": judge_model if use_judge else None,
                "rag": args.rag, "rag_backend": args.rag_backend if args.rag else None,
                "n_cases_requested": len(suite), "n_cases_completed": done_cases,
                "synth_model": args.synth_model, "extract_model_called": args.extract_model,
                "extract_price_tier": args.price_extract,
                "recent_k": RECENT_K, "truncate_k": TRUNCATE_K, "segment_size": SEGMENT_SIZE,
                "rag_top_k": RAG_TOP_K,
                "compact_notes": args.compact_notes, "slot_memory": args.slot_memory,
                "token_counter": "tiktoken/cl100k (cumulative); provider usage (per-query real)",
                "seed": args.seed, "elapsed_s": round(time.time() - t0, 1),
            },
            "strategies": {s: {
                "token_reduction_real_pct": summarize(acc[s]["tok_red_real"]),
                "token_reduction_cumulative_pct": summarize(acc[s]["tok_red_cum"]),
                "cost_pct_of_baseline": summarize(acc[s]["cost_pct"]),
                "retention_pct": summarize(acc[s]["retention"]),
                "fact_update_retention_pct": summarize(acc[s]["upd_retention"]),
                "retention_by_category_pct": {
                    cat: round(b["correct"] / b["total"] * 100, 2)
                    for cat, b in sorted(cat_acc[s].items()) if b["total"]
                },
                "category_counts": {
                    cat: b["total"] for cat, b in sorted(cat_acc[s].items())
                },
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
            "case_metrics": case_rows,
            "question_metrics": question_rows,
        }
        if ablation["cheap"]:
            res["ablation_cheap_vs_expensive_extractor"] = {
                "cheap_retention_pct": summarize(ablation["cheap"]),
                "expensive_retention_pct": summarize(ablation["expensive"]),
            }
        return res

    for ci, case in enumerate(suite):
        if case.id in completed_ids:
            print(f"[{ci+1}/{len(suite)}] {case.id} already done; skipping (resume)",
                  file=sys.stderr)
            continue
        print(f"[{ci+1}/{len(suite)}] {case.id} ({len(case.messages)} msgs, "
              f"{len(case.facts)} facts)  t={time.time()-t0:.0f}s", file=sys.stderr)
        try:
            raw_notes, ex_in, ex_out, exlat = build_notes(client, case, args.extract_model)
            slot_store = None
            slot_ex_in = slot_ex_out = 0
            slot_exlat = []
            if args.slot_memory:
                slot_store, slot_ex_in, slot_ex_out, slot_exlat = build_slot_memory(
                    client, case, args.extract_model)
            retriever = build_retriever(case, backend=args.rag_backend) if args.rag else None
            notes = compacted_notes_copy(raw_notes, args.compact_notes) if args.compact_notes else raw_notes
            summary, sm_in, sm_out = build_summary(synth_client, case, args.synth_model)
            curve = token_curve(case, raw_notes, summary, compact_notes=args.compact_notes,
                                slot_store=slot_store, retriever=retriever)
            qa, qa_rows = run_qa(synth_client, case, args.synth_model, notes, summary,
                                 slot_store=slot_store, retriever=retriever,
                                 judge_client=judge_client, judge_model=judge_model,
                                 use_judge=use_judge)
        except Exception as e:
            print(f"  !! case {case.id} failed ({type(e).__name__}: {str(e)[:80]}); skipping",
                  file=sys.stderr)
            continue

        ex_lat_all += exlat
        ex_lat_all += slot_exlat
        extract_cost_cascade += cost_usd(args.price_extract, ex_in, ex_out)
        extract_cost_actual += cost_usd(args.extract_model, ex_in, ex_out)
        extract_cost_cascade += cost_usd(args.price_extract, slot_ex_in, slot_ex_out)
        extract_cost_actual += cost_usd(args.extract_model, slot_ex_in, slot_ex_out)
        summary_cost += cost_usd(args.synth_model, sm_in, sm_out)

        base_cum = curve["full_context"]
        base_real = qa["full_context"]["in"]
        base_cost = base_cum * opus_in / 1000.0
        case_row = {
            "case_id": case.id,
            "domain": case.domain,
            "length_class": case.length_class,
            "n_messages": len(case.messages),
            "n_facts": len(case.facts),
            "n_updated_facts": sum(1 for fact in case.facts if fact.updated),
            "strategies": {},
        }

        for s in strat_names:
            # cost (% of baseline): synth token flow (cum, tiktoken) x price + maintenance
            maint = 0.0
            if s == "hierarchical":
                maint = (ex_in * px_in + ex_out * px_out) / 1000.0
            elif s in ("slot_memory", "slot_rag_hybrid"):
                maint = (slot_ex_in * px_in + slot_ex_out * px_out) / 1000.0
            elif s == "running_summary":
                maint = (sm_in * opus_in + sm_out * opus_out) / 1000.0
            # "rag" maintenance is local indexing (BM25/embeddings) -> no API cost.
            cost_s = curve[s] * opus_in / 1000.0 + maint
            cost_pct = fraction_of_baseline_pct(base_cost, cost_s)
            tok_red_cum = reduction_pct(base_cum, curve[s])
            tok_red_real = reduction_pct(base_real, qa[s]["in"])
            retention = qa[s]["correct"] / qa[s]["total"] * 100 if qa[s]["total"] else 0.0
            upd_retention = (
                qa[s]["upd_correct"] / qa[s]["upd_total"] * 100 if qa[s]["upd_total"] else None
            )
            mean_lat = statistics.fmean(qa[s]["lat"]) if qa[s]["lat"] else None
            acc[s]["cost_pct"].append(cost_pct)
            acc[s]["tok_red_cum"].append(tok_red_cum)
            acc[s]["tok_red_real"].append(tok_red_real)
            acc[s]["retention"].append(retention)
            if qa[s]["upd_total"]:
                acc[s]["upd_retention"].append(upd_retention)
            if qa[s]["lat"]:
                acc[s]["synth_lat"].append(mean_lat)
            # per-category accumulation (pooled over questions, not cases)
            for cat, tot in qa[s]["cat_total"].items():
                bucket = cat_acc[s].setdefault(cat, {"correct": 0, "total": 0})
                bucket["total"] += tot
                bucket["correct"] += qa[s]["cat_correct"].get(cat, 0)
            case_row["strategies"][s] = {
                "token_reduction_real_pct": round(tok_red_real, 6),
                "token_reduction_cumulative_pct": round(tok_red_cum, 6),
                "cost_pct_of_baseline": round(cost_pct, 6),
                "retention_pct": round(retention, 6),
                "fact_update_retention_pct": round(upd_retention, 6) if upd_retention is not None else None,
                "correct": qa[s]["correct"],
                "total": qa[s]["total"],
                "updated_correct": qa[s]["upd_correct"],
                "updated_total": qa[s]["upd_total"],
                "real_input_tokens": qa[s]["in"],
                "cumulative_input_tokens": curve[s],
                "cost_usd": round(cost_s, 8),
                "latency_s": round(mean_lat, 6) if mean_lat is not None else None,
            }

        if qa["hierarchical"]["lat"]:
            hier_lat.append(statistics.fmean(qa["hierarchical"]["lat"]))
        if qa["full_context"]["lat"]:
            full_lat.append(statistics.fmean(qa["full_context"]["lat"]))

        for row in qa_rows:
            row["case_id"] = case.id
            question_rows.append(row)

        if args.ablation and not args.offline and args.extract_model != args.synth_model:
            try:
                ablation["cheap"].append(acc["hierarchical"]["retention"][-1])
                raw_notes_x, *_ = build_notes(synth_client, case, args.synth_model)
                notes_x = (compacted_notes_copy(raw_notes_x, args.compact_notes)
                           if args.compact_notes else raw_notes_x)
                qa_x, _ = run_qa(synth_client, case, args.synth_model, notes_x, summary,
                                 judge_client=judge_client, judge_model=judge_model,
                                 use_judge=use_judge)
                expensive_retention = (
                    qa_x["hierarchical"]["correct"] / qa_x["hierarchical"]["total"] * 100
                    if qa_x["hierarchical"]["total"] else 0.0)
                ablation["expensive"].append(expensive_retention)
                case_row["ablation_expensive_extractor_retention_pct"] = round(expensive_retention, 6)
            except Exception as e:
                ablation["cheap"].pop()  # keep cheap/expensive aligned
                print(f"  !! ablation for {case.id} failed ({type(e).__name__}); skipping",
                      file=sys.stderr)

        case_rows.append(case_row)
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
