"""Gate wrapper for the self-consistency oracle: the strict fragment must be perfectly clean, and the full distribution may only fail where
attributed to F12 (concrete-index induction)."""
import sys, oracle_stats as o
ok = True
for label, kw, seed in (("strict fragment", dict(mode2_only=True), 1), ("full distribution", dict(), 2)):
    tot, c, v, fb = o.run(2500, seed, verbose=False, **kw)
    bad = sum(x[1] + x[2] for x in tot.values()) + (c - v)
    print("oracle %-18s unattributed failures: %d (completed %d, verified %d)" % (label, bad, c, v)); ok = ok and bad == 0
print("ORACLE GATE:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
