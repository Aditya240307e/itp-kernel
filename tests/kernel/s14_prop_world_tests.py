"""S14 gate: the Prop/Type world (prop_elim + parameters, data in Type1). The library is built by tactics in this mode; every global is re-checked by the independent checker (finding F20 resolved: nothing here is
a term written for the legacy Match rule); injectivity lemmas for Nat are generated; deterministic proofs (equation-based induction, refutation, injectivity); and the zero-tolerance random oracle."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, functools, time
sys.setrecursionlimit(30000)
import harness as H
from harness import *
from s3_tests import V, app
import prop_world as pw
import oracle_prop as op_
import s8_indep_tests as t8
import s11_eqgen_tests as t11
import indep_checker as ic
import noconf

if __name__ == '__main__':
    FAIL = []; t0 = time.time()
    k, g = pw.build_prop_world()
    gids = {n: g[n] for n in ('True', 'Eq', 'transport', 'refl')}
    fac = lambda: ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=4_000_000)
    bad = fac().check_definitions(); print("world: %d globals; independent re-check of every global: %d rejected" % (len(g), len(bad)))
    if bad: FAIL.append('library rejected: ' + str(bad[:3]))
    gen = noconf.generate_injectivity(k, build_new, parse_raw, fac, g['Nat'], gids); g = {name: gid for gid, name, _, _ in k.list_globals()}
    print("generated injectivity for Nat:", gen)
    C = lambda n: ('const', g[n]); NAT, LE = C('Nat'), C('Le'); stmts = op_.statements(g)
    transp = app(C('transport'), NAT, ('lam', NAT, app(LE, V(1), V(8))), V(4), V(0), app(C('Succ_inj_0'), V(4), V(0), V(7)), V(5))
    scripts = {
        'add_zero_r':      [('intro',), ('induct', 0), ('apply', C('refl')), ('apply', C('cong')), ('apply', V(2))],
        'add_succ_r':      [('intro',), ('intro',), ('induct', 0), ('apply', C('refl')), ('apply', C('cong')), ('apply', V(3))],
        'no_succ_le_zero': [('intro',), ('intro',), ('cases', 1)],
        'zero_ne_succ':    [('intro',), ('intro',), ('cases', 1)],
        'le_zero_eq':      [('intro',), ('intro',), ('induct_eq', 1), ('apply', V(3)), ('cases', 6)],
        'le_inv':          [('intro',)] * 3 + [('induct_eq', 2), ('apply', C('inr')), ('apply', V(4)), ('apply', C('inl')), ('apply', transp)],
        'and_swap':        [('intro',)] * 3 + [('cases', 2), ('apply', C('conj')), ('apply', V(6)), ('apply', V(5))],
        'or_swap':         [('intro',)] * 3 + [('cases', 2), ('apply', C('inr')), ('apply', V(5)), ('apply', C('inl')), ('apply', V(5))],
    }
    found = {}
    for name, ops in scripts.items():
        ps, log = t11.run_ops(k, g, stmts[name], ops); failed = [l for l in log if not l[1]]
        proof = parse_raw(k.dump_raw(ps.extract()), True) if (not failed and ps.num_goals() == 0 and ps.is_complete()) else None
        found[name] = (proof, failed)
    chk = fac()          # built AFTER the scripts: their proofs mention derived globals (the barrier, same_ctor, ...) created while they ran
    for name, (proof, failed) in found.items():
        kv = proof is not None and t8.kernel_verdict(k, proof, l2h(stmts[name], 0)); iv = proof is not None and chk.check_closed(proof, l2h(stmts[name], 0))[0]
        print("   %-18s kernel %-5s independent %-5s %s" % (name, kv, iv, '' if (kv and iv) else '  <-- ' + str(failed[:1])[:100]))
        if not (kv and iv): FAIL.append('proof ' + name)
    for seed in (51, 52):
        r = op_.run(int(sys.argv[1]) if len(sys.argv) > 1 else 2500, seed)
        badn = sum(x[1] + x[2] for x in r['tot'].values()) + (r['completed'] - r['verified']) + (r['indep_total'] - r['indep_verified'])
        print("   oracle seed %d: %d tactic applications, completed %d, kernel-verified %d, independently verified %d/%d, failures %d" % (seed, sum(x[0] for x in r['tot'].values()), r['completed'], r['verified'], r['indep_verified'], r['indep_total'], badn))
        if badn: FAIL.append('oracle seed %d' % seed)
    print("elapsed %.0fs" % (time.time() - t0)); print("S14 PROP-WORLD GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
