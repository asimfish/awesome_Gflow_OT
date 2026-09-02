#!/usr/bin/env bash
# Build the consolidated reports (zh/en) into PDF: pandoc -> .tex, then xelatex x2 (TOC).
# Two-step on purpose: pandoc's built-in engine runs with -halt-on-error and dies on a
# recoverable longtable glue warning; standalone xelatex recovers and produces the full PDF.
# Usage: bash scripts/build_pdf.sh [zh|en|all]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/reports"
build() {
  lang=$1; src=$2; base="awesome_gflow_ot_report_${lang}"; work="pdf/build_${lang}"; mkdir -p "$work"
  if [ "$lang" = zh ]; then fonts=(-V mainfont="Helvetica Neue" -V sansfont="Helvetica Neue" -V monofont="Menlo" -H pdf/tex/header_zh.tex)
  else fonts=(-V mainfont="Palatino" -V sansfont="Helvetica Neue" -V monofont="Menlo" -H pdf/tex/header_en.tex); fi
  pandoc "$src" -f markdown+tex_math_single_backslash+pipe_tables+raw_tex -t latex -s -o "$work/$base.tex" \
    --toc --toc-depth=2 \
    -V geometry:"a4paper,margin=2.2cm" -V fontsize=10.5pt -V colorlinks=true "${fonts[@]}" || return 1
  ( cd "$work" && for pass in 1 2; do xelatex -interaction=nonstopmode "$base.tex" > "$base.pass$pass.log" 2>&1; done )
  errs=$(rg -c '^!' "$work/$base.pass2.log" 2>/dev/null || echo 0)
  if [ -f "$work/$base.pdf" ]; then cp "$work/$base.pdf" "pdf/$base.pdf"; echo "OK $lang: pdf/$base.pdf ($(rg -o 'Output written on .*\(([0-9]+) pages' -r '$1' "$work/$base.pass2.log" | tail -1) pages, $errs latex errors)"
  else echo "FAIL $lang"; rg -v MiKTeX "$work/$base.pass2.log" | rg -A3 '^!' | head -20; return 1; fi
}
case "${1:-all}" in
  zh) build zh AWESOME_GFLOW_OT_REPORT_zh.md ;;
  en) build en AWESOME_GFLOW_OT_REPORT_en.md ;;
  all) build zh AWESOME_GFLOW_OT_REPORT_zh.md; build en AWESOME_GFLOW_OT_REPORT_en.md ;;
esac
