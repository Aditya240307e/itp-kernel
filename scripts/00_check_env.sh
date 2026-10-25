#!/usr/bin/env bash
# 00_check_env.sh -- is this machine ready?  Python and libraries, the kernel, cores / memory / disk / GPU, the git state, the dataset.
. "$(dirname "$0")/env.sh"
banner "0. environment"
"$PY" tools/check_env.py --profile "$PROFILE" --runs "$RUNS" --device "$DEVICE"
