"""Gate wrapper for the self-consistency oracle: the strict fragment must be perfectly clean, and the full distribution may only fail where
attributed to F12 (concrete-index induction)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, oracle_stats as o
ok = True
for label, kw, seed in (("strict fragment", dict(mode2_only=True), 1), ("full distribution", dict(), 2), ("parameters ON, full distribution, NO exemptions", dict(params=True), 4), ("parameters ON + index generalisation by equations", dict(params=True, eq=True), 5), ("legacy + index generalisation by equations", dict(eq=True), 6)):
    tot, c, v, fb = o.run(2500, seed, verbose=False, **kw)
    bad = sum(x[1] + x[2] for x in tot.values()) + (c - v) + (sum(x[4] for x in tot.values()) if (kw.get("params") or kw.get("eq")) else 0)
    print("oracle %-18s unattributed failures: %d (completed %d, verified %d)" % (label, bad, c, v)); ok = ok and bad == 0
print("ORACLE GATE:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
