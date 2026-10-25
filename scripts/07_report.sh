#!/usr/bin/env bash
# 07_report.sh -- tables, figures, numbers: $RUNS/report/{report.md, numbers.json, numbers.tex, tables/, figures/}; then copies the LaTeX tables, the numbers and the figures to paper/generated/ and builds
# paper/main.pdf when latexmk is installed.  Nothing in the report is typed in: see docs/EXPERIMENTS.md.
. "$(dirname "$0")/env.sh"
banner "7. report"
TARGET= exp_run --stage report --force report
OUT="$RUNS/report"
if [ "$PROFILE" = full ] || [ "${PUBLISH:-0}" = 1 ]; then
  rm -rf paper/generated; mkdir -p paper/generated
  cp -r "$OUT/tables" paper/generated/tables; cp -r "$OUT/figures" paper/generated/figures; cp "$OUT/numbers.tex" paper/generated/numbers.tex
  cp "$OUT/report.md" paper/generated/report.md
  echo "copied the tables, figures and numbers to paper/generated/"
  if command -v latexmk >/dev/null 2>&1; then make -C paper || echo "the paper did not build: see paper/main.log" >&2; else echo "latexmk is not installed: build the paper with  make -C paper"; fi
else
  echo "profile $PROFILE: the report is in $OUT (paper/generated/ is only updated for the full profile; PUBLISH=1 forces it)"
fi
echo "report: $OUT/report.md"
