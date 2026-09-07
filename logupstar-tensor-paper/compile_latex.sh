#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode logupstar_tensor.tex
bibtex logupstar_tensor
pdflatex -interaction=nonstopmode logupstar_tensor.tex
pdflatex -interaction=nonstopmode logupstar_tensor.tex

rm -f logupstar_tensor.aux logupstar_tensor.bbl logupstar_tensor.blg \
      logupstar_tensor.log logupstar_tensor.out logupstar_tensor.toc \
      logupstar_tensor.lof logupstar_tensor.lot logupstar_tensor.fls \
      logupstar_tensor.fdb_latexmk logupstar_tensor.synctex.gz

echo "Wrote PDF to $(pwd)/logupstar_tensor.pdf"