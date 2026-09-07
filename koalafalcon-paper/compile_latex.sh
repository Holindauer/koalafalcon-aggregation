#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode koalafalcon.tex
bibtex koalafalcon
pdflatex -interaction=nonstopmode koalafalcon.tex
pdflatex -interaction=nonstopmode koalafalcon.tex

rm -f koalafalcon.aux koalafalcon.bbl koalafalcon.blg \
      koalafalcon.log koalafalcon.out koalafalcon.toc \
      koalafalcon.lof koalafalcon.lot koalafalcon.fls \
      koalafalcon.fdb_latexmk koalafalcon.synctex.gz

echo "Wrote PDF to $(pwd)/koalafalcon.pdf"
