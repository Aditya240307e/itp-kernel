#!/usr/bin/env bash
# 06_benchmarks.sh -- throughput and memory: the world build, the Gymnasium interface, every solver, worker scaling, training steps per second, proof-extraction rate (bench/throughput), and the kernel
# micro-benchmarks (bench/kernel).  Run it ALONE on an otherwise idle machine (JOBS=1), or the numbers measure the other jobs too.
. "$(dirname "$0")/env.sh"
JOBS=1 stage_script 06_benchmarks bench "6. throughput and memory benchmarks"
