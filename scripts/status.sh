#!/usr/bin/env bash
# status.sh -- what is done, what is running, what failed (per stage), and the last lines of the newest log.
. "$(dirname "$0")/env.sh"
exp_run --status
newest=$(ls -t "$RUNS"/logs/*.log 2>/dev/null | head -n 1 || true)
[ -n "$newest" ] && { echo; echo "newest log: $newest"; tail -n 5 "$newest"; }
exit 0
