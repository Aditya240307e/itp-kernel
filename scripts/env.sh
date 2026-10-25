#!/usr/bin/env bash
# scripts/env.sh -- the settings every script of this directory shares (they `source` it; running it prints them).
#
# Change a setting by exporting it before you run a script, e.g.   PROFILE=small SEEDS=0-2 ./scripts/run_all.sh
# or once and for all in scripts/local.env (not tracked by git; copy local.env.example).
#
#   PROFILE   smoke | small | full           how big the experiments are (docs/EXPERIMENTS.md, "Profiles")                  default: full
#   SEEDS     "0-4" (five seeds), "0-6" (seven), "0,1,2"   the training seeds = the replicates                              default: 0-1 smoke, 0-2 small, 0-4 full
#   RUNS      the runs directory: every checkpoint, log, evaluation row and the report live below it                        default: runs/$PROFILE
#   JOBS      how many commands may run at once; the scheduler also keeps within CORES and MEM_GB                          default: the number of cores
#   CORES     cores the scheduler may use                                                                                   default: all
#   MEM_GB    memory (GB) the scheduler may use                                                                             default: 85% of the machine's
#   DEVICE    cpu | cuda | cuda:1   where the networks are TRAINED (proof search and evaluation always run on the CPU)      default: cpu
#   WORKERS   forked worker processes inside one training / evaluation command (they share the world of the bank)           default: the profile's (full: 4)
#   SET       profile fields to change, whitespace separated, no spaces inside a value:  "rl_iters=20 synth_sizes=(0,500,2000)"
#   TARGET    only these runs and what they need (comma separated; a trailing * is a prefix):  "rl/ei_her/*,rl/ppo/s0"
#   PY        the Python interpreter                                                                                         default: python3
#
# Every command of the matrix is resumable: run a script again after an interruption and it continues where it stopped.
# A runs directory belongs to ONE profile / seed list / SET: the runner refuses to mix results made with different settings
# (DEVICE, WORKERS, JOBS, CORES and MEM_GB are about the machine, not the results, and may change between runs).

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
cd "$ROOT"
# scripts/local.env holds your usual settings; anything exported in the environment of the call wins over it
if [ -f "$SCRIPTS_DIR/local.env" ]; then
  _kept=""
  for _v in PROFILE SEEDS RUNS JOBS CORES MEM_GB DEVICE WORKERS SET TARGET PY; do
    if [ -n "${!_v+x}" ]; then _kept="$_kept $_v"; eval "_keep_$_v=\${$_v}"; fi
  done
  # shellcheck disable=SC1091
  . "$SCRIPTS_DIR/local.env"
  for _v in $_kept; do eval "$_v=\$_keep_$_v"; done
  unset _kept _v
fi

: "${PROFILE:=full}"
case "$PROFILE" in smoke|small|full) ;; *) echo "PROFILE must be smoke, small or full (got '$PROFILE')" >&2; exit 2;; esac
if [ -z "${SEEDS:-}" ]; then case "$PROFILE" in smoke) SEEDS=0-1;; small) SEEDS=0-2;; *) SEEDS=0-4;; esac; fi
: "${RUNS:=runs/$PROFILE}"
: "${PY:=python3}"
: "${DEVICE:=cpu}"
: "${WORKERS:=}"
: "${SET:=}"
: "${TARGET:=}"
: "${CORES:=}"
: "${MEM_GB:=}"
if [ -z "${JOBS:-}" ]; then JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

banner() { printf '\n==== %s   [%s]\n' "$*" "$(date '+%Y-%m-%d %H:%M:%S')"; }

# exp_run [more exp_run.py arguments]: the experiment runner with the settings above
exp_run() {
  local args=("$PROFILE" --runs "$RUNS" --seeds "$SEEDS" --jobs "$JOBS" --set "device=$DEVICE")
  [ -n "$CORES" ]   && args+=(--cores "$CORES")
  [ -n "$MEM_GB" ]  && args+=(--mem-gb "$MEM_GB")
  [ -n "$WORKERS" ] && args+=(--set "workers=$WORKERS")
  [ -n "$TARGET" ]  && args+=(--target "$TARGET")
  local s; for s in $SET; do args+=(--set "$s"); done
  "$PY" src/python/experiments/exp_run.py "${args[@]}" "$@"
}

# stage_script NAME STAGES TITLE: the body of the per-stage scripts
stage_script() {
  local name="$1" stages="$2" title="$3"
  banner "$title (profile $PROFILE, seeds $SEEDS, runs in $RUNS)"
  local t0=$SECONDS
  if exp_run --stage "$stages"; then
    printf '%s finished in %dm%02ds\n' "$name" $(((SECONDS - t0) / 60)) $(((SECONDS - t0) % 60))
  else
    echo "$name: some runs failed or were skipped; the logs are under $RUNS/logs/ (fix the cause, then run this script again: finished runs are not repeated)" >&2
    return 1
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "PROFILE=$PROFILE SEEDS=$SEEDS RUNS=$RUNS JOBS=$JOBS CORES=${CORES:-all} MEM_GB=${MEM_GB:-85%} DEVICE=$DEVICE WORKERS=${WORKERS:-profile} SET='$SET' TARGET='$TARGET' PY=$PY"
fi
