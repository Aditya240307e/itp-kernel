#!/usr/bin/env bash
# 02_generate_synthetic.sh -- the synthetic theorem set: generate (parallel, resumable by shard), merge with the exclusions, verify from the data alone, write the card and the saturation statistics.
#
# full: 400 000 attempts -> about 85 000 raw theorems -> 60 000 kept (about 8 core-hours; synth_jobs=4 processes).  The bank's dev / test statements are excluded by digest and by shape
# (--exclude-dataset): the synthetic set cannot contain an olympiad problem of the benchmark.
. "$(dirname "$0")/env.sh"
stage_script 02_generate_synthetic data "2. synthetic theorems"
echo; sed -n '1,60p' "$RUNS/synth/DATASET_CARD.md" 2>/dev/null || true
