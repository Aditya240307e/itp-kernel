#!/usr/bin/env bash
# 01_build_dataset.sh -- the kernel and the problem dataset.
#   * builds the kernel (build/itp_env_lnl*.so) if it is missing;
#   * full / small: replays EVERY recorded proof of data/dataset_v2 in a fresh engine and checks the digests (about a minute): the dataset on disk is what the paper's benchmark section describes;
#   * smoke: builds the small dataset of the smoke profile (a few minutes).
. "$(dirname "$0")/env.sh"
banner "1. kernel and dataset"
ls build/itp_env_lnl*.so >/dev/null 2>&1 || make kernel PY="$PY"
TARGET=data/dataset exp_run --stage data
if [ "$PROFILE" != smoke ]; then
  echo
  sed -n '1,40p' data/dataset_v2/DATASET_CARD.md
fi
