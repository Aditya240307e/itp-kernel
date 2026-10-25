#!/usr/bin/env bash
# 02_generate_synthetic.sh -- the synthetic theorem set: generate (parallel, resumable by shard), merge with the exclusions, verify from the data alone, write the card and the saturation statistics.
#
# full: 450 000 attempts to start with (the pilot's yield of 23.5% means about 105 000 raw theorems) -> 60 000 TRAIN theorems, and about 8 000 more held out in dev / test_iid / test_comp; `grow` generates more, up to three
# times as many attempts, when the merge keeps fewer.  About 10 core-hours at 12 attempts per second and core (F82), synth_jobs=4 processes.  No statement of the bank (train, dev or test), no lemma of the library and no sub-theorem
# enters the set: all are excluded by digest and by shape (--exclude-dataset), so it cannot contain an olympiad problem of the benchmark.
. "$(dirname "$0")/env.sh"
stage_script 02_generate_synthetic data "2. synthetic theorems"
echo; sed -n '1,60p' "$RUNS/synth/DATASET_CARD.md" 2>/dev/null || true
