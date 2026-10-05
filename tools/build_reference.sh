#!/bin/sh
# Rebuilds the FROZEN reference kernel (module itp_env_ref) from the refactor baseline commit, isolated (own nanobind domain,
# no gnu-unique symbols) so it can be imported next to the hybrid kernel in one process.
#   usage: tools/build_reference.sh [commit] [outdir]
set -e
COMMIT=${1:-015c7a7}; OUT=${2:-/home/claude/itp_ref}
mkdir -p "$OUT"
git show "$COMMIT:itp_kernel.hpp" > "$OUT/itp_kernel.hpp"
git show "$COMMIT:itp_bindings.cpp" | sed 's/NB_MODULE(itp_env, m)/NB_MODULE(itp_env_ref, m)/' > "$OUT/itp_bindings.cpp"
sed 's/itp_env_lnl/itp_env_ref/g' lnl/build_kernel.py > "$OUT/build_kernel.py"
( cd "$OUT" && ITP_NB_DOMAIN=ref python3 build_kernel.py )
