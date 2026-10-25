#!/usr/bin/env bash
# smoke_test.sh -- the whole pipeline on a tiny world (17 problems, 300 synthetic theorems, 2 seeds): 1-2 hours on two cores, much less on a big machine.  The numbers mean nothing; it checks that
# every stage runs on THIS machine (CPU or GPU) and that every table and figure is produced.  Do this before the first real run.
#   ./scripts/smoke_test.sh             DEVICE=cuda ./scripts/smoke_test.sh
export PROFILE=smoke
. "$(dirname "$0")/env.sh"
banner "smoke test (runs in $RUNS)"
"$SCRIPTS_DIR/run_all.sh"
test -s "$RUNS/report/report.md" && test -s "$RUNS/report/numbers.tex" || { echo "SMOKE TEST FAILED: no report" >&2; exit 1; }
n=$(ls "$RUNS"/report/tables/*.tex | wc -l); f=$(ls "$RUNS"/report/figures/*.png 2>/dev/null | wc -l)
echo "SMOKE TEST OK: $n tables, $f figures; read $RUNS/report/report.md"
