#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/tp2-pdf.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT

latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  -outdir="$build_dir" "$repo_root/docs/report/main.tex"

mkdir -p "$repo_root/submission"
cp "$build_dir/main.pdf" "$repo_root/submission/tp2_detector_spam.pdf"
printf 'Created %s\n' "$repo_root/submission/tp2_detector_spam.pdf"
