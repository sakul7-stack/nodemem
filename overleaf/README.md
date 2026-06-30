# Overleaf folder

This folder contains the LaTeX source and high-resolution figures intended for submission and for upload to Overleaf.

Files of interest:
- `main.tex` — the manuscript (IEEEtran conference style).
- `references.bib` — BibTeX file with canonical and domain references.
- `figures/` — high-resolution PNG figures (primary) and `backup_pdfs/` (PDF backups).
- `backup_sources/` — backups of original `.tex` sources moved into this folder.

To compile locally (requires LaTeX distribution with `pdflatex` and `bibtex` or `latexmk`):

```bash
cd overleaf
latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" main.tex
```

If you prefer Overleaf: upload the entire `overleaf/` folder as a new project; Overleaf will compile automatically.

Notes:
- I performed figure regeneration and a targeted copyedit. I could not run `pdflatex` in this environment — please compile locally or on Overleaf and share logs if you want me to iterate on layout warnings.

Live experiment (Groq) instructions
----------------------------------

To run the real Fact-Slot experiment (NOT the deterministic simulation) you need Groq API keys and network access. Set `GROQ_API_KEY` or `GROQ_API_KEYS` in your environment or add them to a `.env` file in the repo root in the form:

```
GROQ_API_KEY=sk-...
# or
GROQ_API_KEYS=key1,key2,key3
```

Then run from the repository root:

```powershell
.\run_live_groq.bat
```

Notes and safety:
- Live runs cost money; please confirm you want to run them and monitor usage.
- The harness will call the extractor model for every segment and the synthesizer for every user query. Use `--throttle` to rate-limit calls.
- Results will be written to `results_live_run.json`. Upload that file with the Overleaf project or share it for further analysis.
