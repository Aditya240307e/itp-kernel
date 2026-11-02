#!/usr/bin/env bash
# 09_audit.sh -- the STATEMENT AUDIT of the problem bank in ONE stamped folder, $AUDIT (default results/audit): for every problem the checks of its formal statement against the intended meaning (the box, the slips,
# non-vacuity, sayability), its provenance, the cautions (where the automatic evidence is weaker), where it stands with the ledger of second readers (docs/statement_reviews.tsv), and the review sheet a second reader
# fills in; then the tables, the numbers (numbers.tex), the README and the manifest.  A record counts only for the code it ran on (the tree digest) and a reading only for the statement it was written for (its digest).
# It builds the world of the whole bank (about two minutes) and checks 268 statements (about four minutes on a laptop); resumable.  Not part of run_all.sh.  Exit status: 0 CLEAN, 1 a check fails or a second reader
# found a statement unfaithful (FAILED), 2 records are missing or stale (INCOMPLETE).
#   ./scripts/09_audit.sh                                       the whole bank
#   ONLY="imo1959_p1 imo1964_p1a" ./scripts/09_audit.sh         some problems (a quick look: the folder then says it is incomplete)
#   AUDIT         the output folder                                                        default results/audit
#   ONLY          problem names, whitespace separated                                        default all
#   REVIEWS       the ledger of second readers                                               default docs/statement_reviews.tsv
#   REDO=1        check again what has a record already
# After a second reader has filled in $AUDIT/review_sheet.csv:  $PY src/python/experiments/exp_audit.py import-reviews $AUDIT/review_sheet.csv --reviewer NAME   and then   $PY src/python/experiments/exp_audit.py report
# The other settings (PY) are those of scripts/env.sh.
. "$(dirname "$0")/env.sh"
banner "9. statement audit"
: "${AUDIT:=results/audit}"
: "${REVIEWS:=docs/statement_reviews.tsv}"
ls build/itp_env_lnl*.so >/dev/null 2>&1 || make kernel PY="$PY"
args=(run --out "$AUDIT" --reviews "$REVIEWS")
[ "${REDO:-0}" = 1 ] && args+=(--redo)
# shellcheck disable=SC2086
[ -n "${ONLY:-}" ] && args+=(--only $ONLY)
t0=$SECONDS
rc=0
"$PY" src/python/experiments/exp_audit.py "${args[@]}" || rc=$?
"$PY" src/python/experiments/exp_audit.py --verify "$AUDIT" || { echo "the folder $AUDIT does not match its manifest" >&2; rc=1; }
printf '09_audit finished in %dm%02ds\n' $(((SECONDS - t0) / 60)) $(((SECONDS - t0) % 60))
echo "audit folder: $AUDIT  (start with $AUDIT/README.md; the statements to read are in $AUDIT/review_sheet.md)"
exit $rc
