#!/usr/bin/env bash
# 07_report.sh -- tables, figures, numbers and the statistics: $RUNS/report/ (report.md, numbers.json, tables/, figures/, per_problem/, stats/, provenance/, README.md, MANIFEST.sha256), then a COPY of that
# folder in $RESULTS (default results/$PROFILE): the results folder, kept apart from the paper -- everything the paper is written from, and the files it is checked against.  The folder is verified after it
# is copied.  Nothing in it is typed in: see docs/EXPERIMENTS.md.  PUBLISH=1 also copies the LaTeX tables, the numbers and the figures to paper/generated/ and builds paper/main.pdf (when latexmk is installed):
# the paper skeleton is written by hand, and this is only for a paper that cites the generated macros.
. "$(dirname "$0")/env.sh"
banner "7. report"
TARGET= exp_run --stage report --force report
OUT="$RUNS/report"
mkdir -p "$(dirname "$RESULTS")"
if [ "$(cd "$OUT" && pwd)" != "$(mkdir -p "$RESULTS" && cd "$RESULTS" && pwd)" ]; then rm -rf "$RESULTS"; cp -r "$OUT" "$RESULTS"; fi
"$PY" src/python/experiments/exp_bundle.py --verify "$RESULTS" || { echo "the results folder $RESULTS does not match its manifest" >&2; exit 1; }
if [ "${PUBLISH:-0}" = 1 ]; then
  rm -rf paper/generated; mkdir -p paper/generated
  cp -r "$OUT/tables" paper/generated/tables; cp -r "$OUT/figures" paper/generated/figures; cp "$OUT/numbers.tex" paper/generated/numbers.tex
  cp "$OUT/report.md" paper/generated/report.md
  echo "copied the tables, figures and numbers to paper/generated/"
  if command -v latexmk >/dev/null 2>&1; then make -C paper || echo "the paper did not build: see paper/main.log" >&2; else echo "latexmk is not installed: build the paper with  make -C paper"; fi
fi
echo "results folder: $RESULTS  (start with $RESULTS/README.md; the whole report is $RESULTS/report.md)"
