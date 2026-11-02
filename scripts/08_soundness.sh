#!/usr/bin/env bash
# 08_soundness.sh -- the soundness evidence of the paper in ONE stamped folder, $SOUNDNESS (default results/soundness): every gate, every mutant of the catalogue, the falsification campaigns, the kernel against the
# independent checker, the inventory of the trusted code and the classification of the findings (docs/findings_tags.tsv); then the tables, the numbers (numbers.tex), the README and the manifest.  Nothing in it
# is typed in, and a record counts only for the code it ran on (the tree digest): a record of other code is listed as stale.  Resumable: a gate, a mutant, a campaign that has a result for THIS tree is not run again.
# It takes hours -- on two cores the gates about an hour and the mutation catalogue many hours; JOBS=n runs n mutants (or gates, or campaigns) at once, so use the big machine.  Not part of run_all.sh: it does not
# depend on the experiments, and it must be run on the final code.  Exit status: 0 CLEAN, 1 a claim is contradicted (FAILED), 2 evidence is missing or stale (INCOMPLETE).
#   ./scripts/08_soundness.sh                                    everything, with the fuzz seeds of SEEDS (default 0-4)
#   STAGES=gates,fuzz ./scripts/08_soundness.sh                  some stages (gates, mutants, fuzz, differential, inventory, findings)
#   MUTANTS="GE1 GQ" ./scripts/08_soundness.sh --jobs 8          only the mutants whose names begin with these
#   SOUNDNESS     the output folder                                                        default results/soundness
#   STAGES        comma separated                                                           default all
#   MUTANTS       name prefixes, whitespace separated                                       default all
#   MUT_GATES     run the mutants against these gates only (quick check; the folder says it is incomplete)
#   MUT_FULL=1    run EVERY gate against each mutant (how many gates catch it: slower)
#   FUZZ_SEEDS    seeds of the falsification campaigns                                      default SEEDS (0-4);  --seeds 0-6 for seven
#   FUZZ_N        statements per campaign (300)          FUZZ_SECONDS  seconds the search may take per statement (5)
#   DIFF_PER_PROOF  mutated proofs per tactic proof in the comparison with the independent checker (1000)
#   REDO=1        run again what has a result already
# The other settings (JOBS, PY, SEEDS) are those of scripts/env.sh.
. "$(dirname "$0")/env.sh"
banner "8. soundness"
: "${SOUNDNESS:=results/soundness}"
: "${FUZZ_SEEDS:=$SEEDS}"
: "${FUZZ_N:=300}"
: "${FUZZ_SECONDS:=5}"
: "${DIFF_PER_PROOF:=1000}"
ls build/itp_env_lnl*.so >/dev/null 2>&1 || make kernel PY="$PY"
args=(run --out "$SOUNDNESS" --jobs "$JOBS" --fuzz-seeds "$FUZZ_SEEDS" --fuzz-n "$FUZZ_N" --fuzz-seconds "$FUZZ_SECONDS" --diff-per-proof "$DIFF_PER_PROOF")
[ -n "${STAGES:-}" ]     && args+=(--stages "$STAGES")
[ -n "${MUT_GATES:-}" ]  && args+=(--mut-gates "$MUT_GATES")
[ -n "${MUT_FULL:-}" ]   && args+=(--mut-full)
[ "${REDO:-0}" = 1 ]     && args+=(--redo)
# shellcheck disable=SC2086
[ -n "${MUTANTS:-}" ]    && args+=(--mutants $MUTANTS)
t0=$SECONDS
rc=0
"$PY" src/python/experiments/exp_soundness.py "${args[@]}" || rc=$?
"$PY" src/python/experiments/exp_soundness.py --verify "$SOUNDNESS" || { echo "the folder $SOUNDNESS does not match its manifest" >&2; rc=1; }
printf '08_soundness finished in %dh%02dm\n' $(((SECONDS - t0) / 3600)) $(((SECONDS - t0) % 3600 / 60))
echo "soundness folder: $SOUNDNESS  (start with $SOUNDNESS/README.md; the tables are in $SOUNDNESS/tables/)"
exit $rc
