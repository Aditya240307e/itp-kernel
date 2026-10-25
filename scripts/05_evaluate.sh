#!/usr/bin/env bash
# 05_evaluate.sh -- every model on the dev and test splits with the solvers of the study, at the largest budget (the smaller budgets are read off the same runs); each proof is replayed by the
# engine, checked by the kernel and re-checked by the independently written checker.  Output: rows.jsonl per evaluation under $RUNS/eval/.
. "$(dirname "$0")/env.sh"
stage_script 05_evaluate eval "5. evaluation"
