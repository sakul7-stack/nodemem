@echo off
REM Simple build script for Windows (requires MikTeX/TeX Live on PATH)
cd %~dp0
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo Build finished. Check main.pdf
