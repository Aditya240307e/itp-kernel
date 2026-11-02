#!/usr/bin/env bash
# scripts/env.sh -- the settings every script of this directory shares (they `source` it; running it prints them).
#
# Change a setting by exporting it before you run a script, e.g.   PROFILE=small SEEDS=0-2 ./scripts/run_all.sh
# or once and for all in scripts/local.env (not tracked by git; copy local.env.example).
#
#   PROFILE   smoke | small | full           how big the experiments are (docs/EXPERIMENTS.md, "Profiles")                  default: full
#   SEEDS     "0-4" (five seeds), "0-6" (seven), "0,1,2"   the training seeds = the replicates                              default: 0-1 smoke, 0-2 small, 0-4 full
#   RUNS      the runs directory: every checkpoint, log, evaluation row and the report live below it                        default: runs/$PROFILE
#   RESULTS   the results folder: a copy of the report (tables, figures, numbers, per-problem outcomes, statistics, provenance) kept apart from the paper   default: results/$PROFILE
#   JOBS      how many commands may run at once; the scheduler also keeps within CORES and MEM_GB                          default: the number of cores
#   CORES     cores the scheduler may use                                                                                   default: all
#   MEM_GB    memory (GB) the scheduler may use                                                                             default: 85% of the machine's
#   DEVICE    cpu | cuda | cuda:1   where the networks are TRAINED (proof search and evaluation always run on the CPU)      default: cpu
#   WORKERS   forked worker processes inside one training / evaluation command (they share the world of the bank)           default: the profile's (full: 4)
#   SET       profile fields to change, whitespace separated, no spaces inside a value:  "rl_iters=20 synth_sizes=(0,500,2000)"
#   TARGET    only these runs and what they need (comma separated; a trailing * is a prefix):  "rl/ei_her/*,rl/ppo/s0"
#   PY        the Python interpreter                                                                                         default: python3
#
# Every setting is also an option of every script (they win over scripts/local.env and the environment):
#     ./scripts/run_all.sh --profile full --seeds 0-6 --device cuda --jobs 4 --set rl_iters=20 --set eval_probes=100
#   --profile --seeds --runs --results --jobs --cores --mem-gb --device --workers --target --py, and --set (repeatable; added to SET); `--seeds=0-6` works too; --help prints this text.
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
  for _v in PROFILE SEEDS RUNS RESULTS JOBS CORES MEM_GB DEVICE WORKERS SET TARGET PY; do
    if [ -n "${!_v+x}" ]; then _kept="$_kept $_v"; eval "_keep_$_v=\${$_v}"; fi
  done
  # shellcheck disable=SC1091
  . "$SCRIPTS_DIR/local.env"
  for _v in $_kept; do eval "$_v=\$_keep_$_v"; done
  unset _kept _v
fi

# command-line options (the same settings; see the header)
while [ $# -gt 0 ]; do
  _opt="$1"; _val=""; _have=0; shift
  case "$_opt" in --*=*) _val="${_opt#*=}"; _opt="${_opt%%=*}"; _have=1;; esac
  case "$_opt" in
    -h|--help) sed -n '2,/^$/p' "$SCRIPTS_DIR/env.sh" | sed 's/^# \{0,1\}//'; exit 0;;
    --profile|--seeds|--runs|--results|--jobs|--cores|--mem-gb|--device|--workers|--target|--py|--set)
      if [ "$_have" = 0 ]; then
        [ $# -ge 1 ] || { echo "$_opt needs a value (see --help)" >&2; exit 2; }
        _val="$1"; shift
      fi
      case "$_opt" in
        --profile) PROFILE="$_val";;  --seeds) SEEDS="$_val";;  --runs) RUNS="$_val";;  --results) RESULTS="$_val";;  --jobs) JOBS="$_val";;  --cores) CORES="$_val";;  --mem-gb) MEM_GB="$_val";;
        --device) DEVICE="$_val";;  --workers) WORKERS="$_val";;  --target) TARGET="$_val";;  --py) PY="$_val";;  --set) SET="${SET:+$SET }$_val";;
      esac;;
    *) echo "unknown option '$_opt' (see --help)" >&2; exit 2;;
  esac
done
unset _opt _val _have

: "${PROFILE:=full}"
case "$PROFILE" in smoke|small|full) ;; *) echo "PROFILE must be smoke, small or full (got '$PROFILE')" >&2; exit 2;; esac
if [ -z "${SEEDS:-}" ]; then case "$PROFILE" in smoke) SEEDS=0-1;; small) SEEDS=0-2;; *) SEEDS=0-4;; esac; fi
: "${RUNS:=runs/$PROFILE}"
: "${RESULTS:=results/$PROFILE}"
: "${PY:=python3}"
: "${DEVICE:=cpu}"
: "${WORKERS:=}"
: "${SET:=}"
: "${TARGET:=}"
: "${CORES:=}"
: "${MEM_GB:=}"
if [ -z "${JOBS:-}" ]; then JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"; fi
export PROFILE SEEDS RUNS RESULTS PY DEVICE WORKERS SET TARGET CORES MEM_GB JOBS            # (the scripts run each other: what an option set is what the next script reads from its environment)
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
  echo "PROFILE=$PROFILE SEEDS=$SEEDS RUNS=$RUNS RESULTS=$RESULTS JOBS=$JOBS CORES=${CORES:-all} MEM_GB=${MEM_GB:-85%} DEVICE=$DEVICE WORKERS=${WORKERS:-profile} SET='$SET' TARGET='$TARGET' PY=$PY"
fi
