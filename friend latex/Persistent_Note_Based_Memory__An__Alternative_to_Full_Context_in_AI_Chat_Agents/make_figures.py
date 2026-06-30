"""
make_figures.py -- deterministic, publication-quality figure generator.

Regenerates every figure used in the paper from the published numbers (sourced
from results_live.json and the deterministic cost-growth simulation). Produces
both PDF (for LaTeX) and PNG (for preview) into ./figures/.

Run:  python make_figures.py
Deps: matplotlib (>=3.5).

All numbers are hard-coded from the committed results files so the figures are
reproducible without an API call and always match the paper's tables.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ---- consistent publication style -------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# Strategy colours (shared across all figures).
C = {
    "full":    "#1f77b4",   # blue
    "trunc":   "#ff7f0e",   # orange
    "summary": "#2ca02c",   # green
    "hier":    "#9467bd",   # purple
    "comp":    "#2ca02c",   # green (compaction line)
}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


# ---- 1. architecture flowchart ---------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    linewidth=1.2, edgecolor="#333", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.2, color="#333"))

    # write path (top), read path (bottom)
    box(0.2, 3.0, 2.2, 1.1, "Conversation\nsegment", "#e8f0fb")
    box(2.9, 3.0, 2.2, 1.1, r"Cheap extractor" + "\n$\\phi_{small}$", "#e6f5ea")
    box(5.6, 3.0, 2.4, 1.1, "Persistent notes\n(JSON ledger)", "#fdf2dc")
    box(2.9, 0.6, 2.2, 1.1, "Recent window\n(last $k$ msgs)", "#f3e8fb")
    box(5.6, 0.6, 2.4, 1.1, r"Large synthesizer" + "\n$\\phi_{large}$", "#fcebeb")
    box(8.4, 0.6, 1.4, 1.1, "Answer", "#eeeeee")

    arrow(2.4, 3.55, 2.9, 3.55)
    arrow(5.1, 3.55, 5.6, 3.55)
    arrow(6.8, 3.0, 6.8, 1.7)      # notes -> synth
    arrow(5.1, 1.15, 5.6, 1.15)    # recent -> synth
    arrow(8.0, 1.15, 8.4, 1.15)    # synth -> answer

    ax.text(0.2, 4.5, "Write path: extract once per segment", fontsize=10, fontweight="bold")
    ax.text(2.9, 0.05, "Read path: notes + recent messages replace full-history prompts",
            fontsize=10, fontweight="bold")
    _save(fig, "architecture_flowchart")


# ---- 2. retention / cost frontier (measured, n=7) ---------------------------
# (cost%, cost_lo, cost_hi, ret%, ret_lo, ret_hi)  from results_live.json
FRONTIER = {
    "Full context":    (C["full"],    100.00, 100.00, 100.00, 100.00, 100.00, 100.00),
    "Truncation":      (C["trunc"],    74.52,  58.37,  88.23,  42.86,  14.29,  85.71),
    "Running summary": (C["summary"], 165.18, 137.63, 189.44,  97.14,  91.43, 100.00),
    "Hierarchical":    (C["hier"],    108.37,  94.83, 120.83,  84.52,  73.81,  95.24),
}


def fig_frontier():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, (col, c, clo, chi, r, rlo, rhi) in FRONTIER.items():
        ax.errorbar(c, r, xerr=[[c - clo], [chi - c]], yerr=[[r - rlo], [rhi - r]],
                    fmt="o", color=col, capsize=4, markersize=8, elinewidth=1.5)
        dx, dy = (6, 6) if name != "Truncation" else (8, -2)
        ax.annotate(name, (c, r), textcoords="offset points", xytext=(dx, dy), fontsize=10)
    ax.axvline(100, ls="--", color="#555", lw=1)
    ax.axhline(90, ls=":", color="#777", lw=1)
    ax.set_xlabel("Cost (% of full-context baseline)")
    ax.set_ylabel("Retention (%)")
    ax.set_title("Retention/cost frontier with 95% bootstrap CIs")
    ax.set_ylim(0, 108)
    _save(fig, "retention_cost_frontier")


# ---- 3. cascade ablation ----------------------------------------------------
def fig_ablation():
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    labels = ["Cheap 8B extractor", "Expensive 70B extractor"]
    means = [84.52, 70.48]
    err = [[84.52 - 73.81, 70.48 - 60.71], [95.24 - 84.52, 82.14 - 70.48]]
    ax.bar(labels, means, yerr=err, capsize=6, color=[C["full"], "#d62728"],
           edgecolor="#222", linewidth=0.8, width=0.6)
    ax.set_ylabel("Downstream retention (%)")
    ax.set_title("Cascade ablation: extractor size")
    ax.set_ylim(0, 105)
    _save(fig, "cascade_ablation")


# ---- 4. measured results, three panels --------------------------------------
def fig_measured_bars():
    methods = ["Full context", "Truncation", "Running summary", "Hierarchical"]
    cols = [C["full"], C["trunc"], C["summary"], C["hier"]]
    tok = [0.0, 46.9, 17.2, 15.4]
    ret = [100.0, 42.86, 97.14, 84.52]
    cost = [100.0, 74.52, 165.18, 108.37]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, vals, title in zip(axes, [tok, ret, cost],
                               ["Real token reduction (%)", "Retention (%)", "Cost (% of full)"]):
        ax.bar(range(4), vals, color=cols, edgecolor="#222", linewidth=0.7)
        ax.set_xticks(range(4)); ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
        ax.set_title(title)
    axes[2].axhline(100, ls="--", color="#333", lw=1)
    fig.suptitle("Measured Groq small-scale evaluation (n = 7)", fontsize=12)
    _save(fig, "measured_results_bars")


# ---- 5. deterministic long-horizon cost simulation (THE FIX) ----------------
# % of full-context cost (input + maintenance) at each length.  From Table III.
LEN = [14, 32, 60, 120, 240, 480]
SIM = {
    "Full context":           (C["full"],  "-",  [100, 100, 100, 100, 100, 100]),
    "Truncation":             (C["trunc"], "--", [96.3, 61.4, 37.7, 20.8, 11.0, 5.7]),
    "Hierarchical notes":     (C["hier"],  "-",  [121.7, 91.1, 73.3, 51.9, 42.0, 36.3]),
    "Hierarchical + compaction": (C["comp"], "-", [121.7, 91.1, 71.9, 42.3, 24.0, 13.0]),
}


def fig_cost_growth():
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name, (col, ls, ys) in SIM.items():
        ax.plot(LEN, ys, ls, color=col, marker="o", markersize=6, linewidth=2,
                label=name, zorder=3)
    ax.axhline(100, ls=":", color="#555", lw=1, zorder=1)
    # Fixed, readable log x-axis: explicit ticks at the simulated lengths only,
    # no overlapping minor decade labels (the bug in the previous figure).
    ax.set_xscale("log")
    ax.set_xticks(LEN)
    ax.set_xticklabels([str(n) for n in LEN])
    ax.minorticks_off()
    ax.set_xlim(12, 560)
    ax.set_ylim(0, 130)
    ax.set_xlabel("Conversation length (messages, log scale)")
    ax.set_ylabel("Input + maintenance cost (% of full context)")
    ax.set_title("Deterministic long-horizon cost simulation")
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    # annotate the crossover and the endpoint
    ax.annotate("crossover below\nfull context", xy=(32, 91), xytext=(40, 112),
                fontsize=8.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.9))
    ax.annotate("13.0%", xy=(480, 13.0), xytext=(300, 22), fontsize=9, color=C["comp"],
                arrowprops=dict(arrowstyle="->", color=C["comp"], lw=0.9))
    _save(fig, "cost_vs_length_simulation")


if __name__ == "__main__":
    fig_architecture()
    fig_frontier()
    fig_ablation()
    fig_measured_bars()
    fig_cost_growth()
    print("all figures written to", OUT)
