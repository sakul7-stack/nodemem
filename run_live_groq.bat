@echo off
REM Run a live Groq experiment for Fact-Slot Memory. Requires GROQ_API_KEY or GROQ_API_KEYS in environment or .env
REM Usage: set GROQ_API_KEY=your_key_here && run_live_groq.bat

cd %~dp0
python evaluate.py --provider groq --scale paper --slot-memory --compact-notes 8 --out results_live_run.json

echo Live run complete. Results written to results_live_run.json
