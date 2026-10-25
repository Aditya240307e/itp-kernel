#!/usr/bin/env bash
# run_all.sh -- everything, in dependency order, with ONE scheduler (a stage starts for a seed as soon as its inputs exist, so the stages overlap):
#     check -> dataset -> synthetic theorems -> pre-train -> fine-tune -> reinforcement learning -> evaluation -> report
# then the benchmarks alone at the end (they need an idle machine).  Resumable: run it again after any interruption.
#   PROFILE=small SEEDS=0-2 ./scripts/run_all.sh        PROFILE=full SEEDS=0-6 DEVICE=cuda JOBS=4 ./scripts/run_all.sh
. "$(dirname "$0")/env.sh"
banner "run_all: profile $PROFILE, seeds $SEEDS, runs in $RUNS"
ls build/itp_env_lnl*.so >/dev/null 2>&1 || make kernel PY="$PY"
"$SCRIPTS_DIR/00_check_env.sh"
t0=$SECONDS
exp_run --stage data,pre,ft,rl,eval
"$SCRIPTS_DIR/06_benchmarks.sh"
"$SCRIPTS_DIR/07_report.sh"
printf '\nall done in %dh%02dm\n' $(((SECONDS - t0) / 3600)) $(((SECONDS - t0) % 3600 / 60))
