#!/bin/sh
# Uso:
#   ./build_submission.sh   compila el informe y arma submission/ con los entregables:
#                           PDF, fuentes LaTeX (informe/), notebook y pruebas (tests/).
# Después basta comprimir el contenido de submission/ en un .zip.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
submission="$repo_root/submission"
report="$repo_root/docs/report"

"$repo_root/build_pdf.sh"

rm -rf "$submission/informe" "$submission/notebooks" "$submission/tests"
mkdir -p "$submission/informe/sections" "$submission/informe/figuras" \
  "$submission/notebooks" "$submission/tests"

# Fuentes LaTeX del informe; review.tex es interno y no se entrega.
cp "$report/main.tex" "$report/references.bib" "$submission/informe/"
cp "$report"/sections/*.tex "$submission/informe/sections/"

# Solo las figuras que el informe incluye.
sed -n 's/.*\\includegraphics\(\[[^]]*\]\)\{0,1\}{\([^}]*\)}.*/\2/p' "$report"/sections/*.tex |
  sort -u | while read -r figure; do
    case "$figure" in
      *.png|*.pdf|*.jpg) source="$repo_root/results/figures/$figure" ;;
      *) source="$repo_root/results/figures/$figure.png" ;;
    esac
    cp "$source" "$submission/informe/figuras/"
  done

# Notebook y pruebas; las pruebas del ejecutor auxiliar del proyecto no se entregan.
cp "$repo_root/notebooks/tp2_detector_spam.ipynb" "$submission/notebooks/"
for test_file in "$repo_root"/tests/*.py; do
  case "$(basename -- "$test_file")" in
    test_partial_runner.py) ;;
    *) cp "$test_file" "$submission/tests/" ;;
  esac
done

printf 'Entrega lista en %s\n' "$submission"
