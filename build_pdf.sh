#!/bin/sh
# Uso:
#   ./build_pdf.sh          informe (docs/report/main.tex)   -> submission/tp2_detector_spam.pdf
#   ./build_pdf.sh review   revisión interna (review.tex)    -> docs/report/review.pdf
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

case "${1:-informe}" in
  informe)
    document=main
    output="$repo_root/submission/tp2_detector_spam.pdf"
    ;;
  review)
    document=review
    output="$repo_root/docs/report/review.pdf"
    ;;
  *)
    printf 'Uso: %s [review]\n' "$0" >&2
    exit 2
    ;;
esac

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/tp2-pdf.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT

# En Git Bash (Windows), latexmk le pasa a BibTeX rutas como /c/Users/..., que
# MiKTeX no entiende: termina usando otro references.bib de sus propios paquetes.
# Se agrega la carpeta del informe en formato de Windows para que use el nuestro.
if command -v cygpath >/dev/null 2>&1; then
  BIBINPUTS="$(cygpath -w "$repo_root/docs/report");${BIBINPUTS:-}"
  export BIBINPUTS
fi

latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  -outdir="$build_dir" "$repo_root/docs/report/$document.tex"

mkdir -p "$(dirname -- "$output")"
cp "$build_dir/$document.pdf" "$output"
printf 'Created %s\n' "$output"
