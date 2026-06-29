"""Lightweight publication-readiness audit for the paper source.

This does not certify novelty or guarantee acceptance. It checks practical
failure modes before submission: unresolved placeholders, citation coverage,
required reproducibility artifacts, and accidental secret tracking.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "main.tex",
    "README.md",
    "EVALUATION_REPORT.md",
    "results_live_rerun.json",
    "results_live_seed8.json",
    "results_live_aggregate.json",
    "results_growth_simulation.json",
    "scripts/paired_analysis.py",
    "figures/architecture_flowchart.pdf",
    "figures/retention_cost_frontier.pdf",
    "figures/cascade_ablation.pdf",
    "figures/cost_vs_length_simulation.pdf",
    "examples/sample_notes_product_development.json",
]
PLACEHOLDERS = ["REPO_URL_PLACEHOLDER", "TODO", "???", "FIXME"]
SECRET_PATTERNS = [r"gsk_[A-Za-z0-9_-]{20,}", r"sk-ant-[A-Za-z0-9_-]{20,}"]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> None:
    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            fail(f"missing required artifact: {rel}")

    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    for marker in PLACEHOLDERS:
        if marker in tex:
            fail(f"unresolved placeholder in main.tex: {marker}")

    cites = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        cites.update(item.strip() for item in match.group(1).split(","))
    bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    missing = cites - bibs
    unused = bibs - cites
    if missing:
        fail(f"citation keys missing bibliography entries: {sorted(missing)}")
    if unused:
        fail(f"bibliography entries not cited: {sorted(unused)}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or ".venv" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".md", ".tex", ".json", ".yml", ".yaml", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text):
                fail(f"possible API key leaked in tracked text file: {path.relative_to(ROOT)}")

    print("PASS: publication audit checks completed.")
    print(f"Citations: {len(cites)} cited keys, {len(bibs)} bibliography entries, no missing entries.")


if __name__ == "__main__":
    main()
