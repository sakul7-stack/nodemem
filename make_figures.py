"""Generate publication artifacts for the persistent note-memory paper.

Outputs are deterministic and safe to run without API keys:

* measured-result figures from the latest live Groq result JSON;
* an architecture flowchart;
* a long-horizon token/cost simulation using local token counts and the seeded
  synthetic suite.

The simulation is not a replacement for live retention measurements.  It exists
to visualize the asymptotic cost argument under the same prompt assembly and
price model used by the evaluation harness.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from datasets import Case, build_suite
from hierarchical_memory import (
    EXTRACT_SYSTEM,
    SLOT_EXTRACT_SYSTEM,
    FullContext,
    Hierarchical,
    Note,
    NotesFile,
    SlotMemory,
    SlotMemoryStore,
    Truncation,
    _heuristic_extract,
    parse_slot_facts,
)
from token_utils import assemble_text, count_tokens_local, fraction_of_baseline_pct, price_for


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results_live_slot_aggregate.json"
DEFAULT_FIG_DIR = ROOT / "figures"
DEFAULT_GROWTH_OUT = ROOT / "results_growth_simulation.json"

RECENT_K = 6
SEGMENT_SIZE = 6
TRUNCATE_K = 10
COMPACT_MAX_NOTES = 8
SYNTH_MODEL = "llama-3.3-70b-versatile"
EXTRACT_MODEL = "llama-3.1-8b-instant"

STRATEGIES = [
    ("full_context", "Full context"),
    ("truncation", "Truncation"),
    ("running_summary", "Running summary"),
    ("hierarchical", "Append-only notes"),
    ("slot_memory", "Fact slots"),
]

COLORS = {
    "full_context": "#4c78a8",
    "truncation": "#f58518",
    "running_summary": "#54a24b",
    "hierarchical": "#b279a2",
    "slot_memory": "#e45756",
}


def available_strategies(results: dict) -> list[tuple[str, str]]:
    strategies = results.get("strategies", {})
    return [(key, label) for key, label in STRATEGIES if key in strategies]


def save_figure(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def stat(results: dict, strategy: str, metric: str) -> tuple[float, float, float]:
    s = results["strategies"][strategy][metric]
    mean = float(s["mean"])
    lo, hi = s.get("ci95", [mean, mean])
    return mean, float(lo), float(hi)


def plot_measured_bars(results: dict, out_dir: Path) -> None:
    metrics = [
        ("token_reduction_real_pct", "Real token reduction (%)"),
        ("retention_pct", "Retention (%)"),
        ("cost_pct_of_baseline", "Cost (% of full)"),
    ]
    strategies = available_strategies(results)
    colors = [COLORS[key] for key, _ in strategies]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    labels = [label for _, label in strategies]
    for ax, (metric, title) in zip(axes, metrics):
        means = [stat(results, key, metric)[0] for key, _ in strategies]
        ax.bar(range(len(labels)), means, color=colors, edgecolor="black", linewidth=0.5)
        if metric == "cost_pct_of_baseline":
            ax.axhline(100, color="#444444", linestyle="--", linewidth=1, label="Full-context cost")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    n = results.get("config", {}).get("n_cases_completed", "?")
    fig.suptitle(f"Measured Groq small-scale evaluation (n = {n})", y=1.05, fontsize=12)
    save_figure(fig, out_dir, "measured_results_bars")


def plot_retention_cost_frontier(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label in available_strategies(results):
        cost, cost_lo, cost_hi = stat(results, key, "cost_pct_of_baseline")
        retention, ret_lo, ret_hi = stat(results, key, "retention_pct")
        ax.errorbar(
            cost,
            retention,
            xerr=[[cost - cost_lo], [cost_hi - cost]],
            yerr=[[retention - ret_lo], [ret_hi - retention]],
            fmt="o",
            color=COLORS[key],
            capsize=4,
            markersize=7,
            label=label,
        )
        ax.annotate(label, (cost, retention), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axvline(100, color="#444444", linestyle="--", linewidth=1)
    ax.axhline(90, color="#777777", linestyle=":", linewidth=1)
    ax.set_xlabel("Cost (% of full-context baseline)")
    ax.set_ylabel("Retention (%)")
    ax.set_title("Retention/cost frontier with 95% CIs")
    ax.set_xlim(55, 190)
    ax.set_ylim(0, 108)
    ax.grid(alpha=0.25)
    save_figure(fig, out_dir, "retention_cost_frontier")


def plot_ablation(results: dict, out_dir: Path) -> None:
    ablation = results.get("ablation_cheap_vs_expensive_extractor")
    if not ablation:
        return
    labels = ["Cheap 8B extractor", "Expensive 70B extractor"]
    keys = ["cheap_retention_pct", "expensive_retention_pct"]
    means = [float(ablation[k]["mean"]) for k in keys]
    errors = []
    for k, mean in zip(keys, means):
        lo, hi = ablation[k].get("ci95", [mean, mean])
        errors.append([mean - float(lo), float(hi) - mean])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(labels, means, yerr=list(zip(*errors)), capsize=5,
           color=["#4c78a8", "#e45756"], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Downstream retention (%)")
    ax.set_title("Cascade ablation: extractor size")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir, "cascade_ablation")


def draw_box(ax, xy: tuple[float, float], text: str, width: float = 2.2, height: float = 0.72,
             color: str = "#e8f1fb") -> FancyBboxPatch:
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#263238",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
    return box


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14,
                                 linewidth=1.2, color="#263238"))


def plot_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.set_axis_off()
    boxes = {
        "segment": draw_box(ax, (0.2, 2.25), "Conversation\nsegment", color="#e8f1fb"),
        "extractor": draw_box(ax, (3.0, 2.25), "Cheap extractor\n$\\phi_{small}$", color="#e8f7ee"),
        "notes": draw_box(ax, (5.8, 2.25), "Fact-slot table\n(latest value per key)", color="#fff4df"),
        "recent": draw_box(ax, (3.0, 0.9), "Recent window\n(last $k$ messages)", color="#f4e8fb"),
        "synth": draw_box(ax, (5.8, 0.9), "Large synthesizer\n$\\phi_{large}$", color="#fdebea"),
        "answer": draw_box(ax, (8.3, 0.9), "Answer", width=1.4, color="#eeeeee"),
    }
    draw_arrow(ax, (2.4, 2.61), (3.0, 2.61))
    draw_arrow(ax, (5.2, 2.61), (5.8, 2.61))
    draw_arrow(ax, (6.9, 2.25), (6.9, 1.62))
    draw_arrow(ax, (5.2, 1.26), (5.8, 1.26))
    draw_arrow(ax, (8.0, 1.26), (8.3, 1.26))
    ax.text(0.3, 3.35, "Write path: extract slots once per segment, overwrite by key", fontsize=9, weight="bold")
    ax.text(3.0, 0.35, "Read path: latest-value slots + recent messages replace full-history prompts", fontsize=9, weight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.8)
    save_figure(fig, out_dir, "architecture_flowchart")


def segment_messages(messages: list[dict], size: int):
    for i in range(0, len(messages), size):
        yield i // size, messages[i:i + size]


def build_segment_notes(case: Case) -> tuple[list[Note], int, int]:
    notes = []
    extract_in = 0
    extract_out = 0
    for seg_id, seg in segment_messages(case.messages, SEGMENT_SIZE):
        convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in seg)
        data = _heuristic_extract(convo)
        notes.append(Note(
            timestamp=seg_id,
            segment_id=seg_id,
            entities=data.get("entities", []) or [],
            decisions=data.get("decisions", []) or [],
            facts=data.get("facts", []) or [],
            goals=data.get("goals", []) or [],
            action_items=data.get("action_items", []) or [],
        ))
        extract_in += count_tokens_local(assemble_text(EXTRACT_SYSTEM, [{"role": "user", "content": convo}]))
        extract_out += count_tokens_local(json.dumps(data, sort_keys=True))
    return notes, extract_in, extract_out


def build_segment_slots(case: Case) -> tuple[list[list], int, int]:
    """Deterministic per-segment slot facts and extractor token accounting."""
    per_segment = []
    extract_in = 0
    extract_out = 0
    for seg_id, seg in segment_messages(case.messages, SEGMENT_SIZE):
        convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in seg)
        facts = parse_slot_facts(convo, seg_id, seg_id)
        per_segment.append(facts)
        extract_in += count_tokens_local(assemble_text(SLOT_EXTRACT_SYSTEM, [{"role": "user", "content": convo}]))
        extract_out += count_tokens_local(json.dumps([f.value for f in facts], sort_keys=True))
    return per_segment, extract_in, extract_out


def simulate_case_costs(case: Case) -> dict:
    segment_notes, extract_in, extract_out = build_segment_notes(case)
    slot_segments, slot_extract_in, slot_extract_out = build_segment_slots(case)
    n_segments = (len(case.messages) + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    totals = {
        "full_context": 0,
        "truncation": 0,
        "hierarchical_append_only": 0,
        "hierarchical_compacted": 0,
        "slot_memory": 0,
    }
    raw_notes = NotesFile()
    compact_notes = NotesFile()
    slot_store = SlotMemoryStore()
    seg_index = 0
    history = []
    full = FullContext()
    trunc = Truncation(TRUNCATE_K)

    for turn_i, message in enumerate(case.messages):
        if message["role"] == "user":
            query = message["content"]
            system, msgs = full.build_context(history, query)
            totals["full_context"] += count_tokens_local(assemble_text(system, msgs))
            system, msgs = trunc.build_context(history, query)
            totals["truncation"] += count_tokens_local(assemble_text(system, msgs))
            system, msgs = Hierarchical(RECENT_K, raw_notes).build_context(history, query)
            totals["hierarchical_append_only"] += count_tokens_local(assemble_text(system, msgs))
            system, msgs = Hierarchical(RECENT_K, compact_notes).build_context(history, query)
            totals["hierarchical_compacted"] += count_tokens_local(assemble_text(system, msgs))
            system, msgs = SlotMemory(RECENT_K, slot_store).build_context(history, query)
            totals["slot_memory"] += count_tokens_local(assemble_text(system, msgs))

        history.append(message)
        while seg_index < n_segments and (turn_i + 1) >= (seg_index + 1) * SEGMENT_SIZE:
            if seg_index < len(segment_notes):
                raw_notes.notes.append(segment_notes[seg_index])
                compact_notes.notes.append(segment_notes[seg_index])
                compact_notes.compact(max_notes=COMPACT_MAX_NOTES)
            if seg_index < len(slot_segments):
                for fact in slot_segments[seg_index]:
                    slot_store.upsert(fact)
            seg_index += 1

    synth_in_price, _ = price_for(SYNTH_MODEL)
    extract_in_price, extract_out_price = price_for(EXTRACT_MODEL)
    extract_cost = (extract_in * extract_in_price + extract_out * extract_out_price) / 1000.0
    slot_extract_cost = (slot_extract_in * extract_in_price + slot_extract_out * extract_out_price) / 1000.0
    full_cost = totals["full_context"] * synth_in_price / 1000.0
    costs = {
        "full_context": full_cost,
        "truncation": totals["truncation"] * synth_in_price / 1000.0,
        "hierarchical_append_only": totals["hierarchical_append_only"] * synth_in_price / 1000.0 + extract_cost,
        "hierarchical_compacted": totals["hierarchical_compacted"] * synth_in_price / 1000.0 + extract_cost,
        "slot_memory": totals["slot_memory"] * synth_in_price / 1000.0 + slot_extract_cost,
    }
    return {
        "messages": len(case.messages),
        "input_tokens": totals,
        "cost_usd": costs,
        "cost_pct_of_full": {k: fraction_of_baseline_pct(full_cost, v) for k, v in costs.items()},
        "extract_tokens": {"input": extract_in, "output": extract_out},
    }


def mean_dict(rows: list[dict], key: str, subkey: str) -> float:
    return round(statistics.fmean(row[key][subkey] for row in rows), 3)


def run_growth_simulation(out_path: Path) -> dict:
    cases = build_suite(seed=7, scale="paper") + build_suite(seed=7, scale="long")
    grouped = defaultdict(list)
    for case in cases:
        grouped[len(case.messages)].append(simulate_case_costs(case))

    strategy_keys = ["full_context", "truncation", "hierarchical_append_only", "hierarchical_compacted", "slot_memory"]
    points = []
    for n_messages in sorted(grouped):
        rows = grouped[n_messages]
        point = {"messages": n_messages, "n_cases": len(rows)}
        for key in strategy_keys:
            point[key] = {
                "input_tokens": mean_dict(rows, "input_tokens", key),
                "cost_pct_of_full": mean_dict(rows, "cost_pct_of_full", key),
                "cost_usd": mean_dict(rows, "cost_usd", key),
            }
        points.append(point)

    payload = {
        "config": {
            "seed": 7,
            "source_scales": ["paper", "long"],
            "synth_model": SYNTH_MODEL,
            "extract_model": EXTRACT_MODEL,
            "recent_k": RECENT_K,
            "truncate_k": TRUNCATE_K,
            "segment_size": SEGMENT_SIZE,
            "compact_max_notes": COMPACT_MAX_NOTES,
            "token_counter": "tiktoken/cl100k local simulation",
            "note_extraction": "deterministic heuristic notes; no live LLM calls",
        },
        "points": points,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def plot_growth_simulation(simulation: dict, out_dir: Path) -> None:
    points = simulation["points"]
    xs = [p["messages"] for p in points]
    series = [
        ("full_context", "Full context", "#4c78a8", "-"),
        ("truncation", "Truncation", "#f58518", "--"),
        ("hierarchical_append_only", "Hierarchical notes", "#b279a2", "-"),
        ("hierarchical_compacted", "Hierarchical + compaction", "#54a24b", "-"),
        ("slot_memory", "Fact slots", "#e45756", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for key, label, color, linestyle in series:
        ys = [p[key]["cost_pct_of_full"] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, linestyle=linestyle, label=label)
    ax.axhline(100, color="#444444", linestyle=":", linewidth=1)
    ax.set_xlabel("Conversation length (messages)")
    ax.set_ylabel("Input + memory-maintenance cost (% of full context)")
    ax.set_title("Deterministic long-horizon cost simulation")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, out_dir, "cost_vs_length_simulation")


def load_results(path: Path) -> dict | None:
    if not path.exists():
        print(f"Measured results file not found: {path}; skipping measured-result plots.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures and growth simulation output.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--growth-out", type=Path, default=DEFAULT_GROWTH_OUT)
    args = parser.parse_args()

    plot_architecture(args.out_dir)
    simulation = run_growth_simulation(args.growth_out)
    plot_growth_simulation(simulation, args.out_dir)

    results = load_results(args.results)
    if results is not None:
        plot_measured_bars(results, args.out_dir)
        plot_retention_cost_frontier(results, args.out_dir)

    print(f"Wrote figures to {args.out_dir}")
    print(f"Wrote growth simulation to {args.growth_out}")


if __name__ == "__main__":
    main()
