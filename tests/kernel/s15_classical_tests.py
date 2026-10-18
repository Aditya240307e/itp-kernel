"""S15 gate: classical logic in the Prop/Type world. The four axioms (em, propext, funext, choice) are the ONLY additions to the trusted base; the derived principles are proved by tactics; every global is re-checked by
the independent checker; and a canary tries, with the axioms available, to prove false statements (a verified proof of one would be a soundness bug)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, random, functools, time
sys.setrecursionlimit(30000)
import harness as H
from harness import *
from s3_tests import V, app
import prop_world as pw
import oracle_prop as op_
import s8_indep_tests as t8
import s11_eqgen_tests as t11
import indep_checker as ic
import canary_lib as canlib

def canary(N, seed, which='false'):
    k, g = pw.build_prop_world(classical=True)
    C = lambda n: ('const', g[n]); NAT, ZERO, S, EQ, FALSE, OR = C('Nat'), C('Zero'), C('Succ'), C('Eq'), C('False'), C('Or')
    T0, T1 = pw.T0, pw.T1; eq = lambda A, a, b: app(EQ, A, a, b)
    false_stmts = {'False': FALSE, '0 = 1': eq(NAT, ZERO, app(S, ZERO)), 'forall P : Prop, P': ('pi', T0, V(0)), 'forall P Q, Or P Q -> P': ('pi', T0, ('pi', T0, ('pi', app(OR, V(0), V(1)), V(0)))),
                   'forall (A : Type1) (a b : A), a = b': ('pi', T1, ('pi', V(0), ('pi', V(0), eq(V(0), V(1), V(2))))), 'Prop = False': eq(T0, FALSE, C('True')),
                   'forall n, S n = n': ('pi', NAT, eq(NAT, app(S, V(0)), V(0)))}
    NOT, IFF = C('Not'), C('Iff')
    true_stmts = {'forall P, Or P (Not P)': ('pi', T0, app(OR, V(0), app(NOT, V(0)))), 'forall P, ~~P -> P': ('pi', T0, ('pi', app(NOT, app(NOT, V(0))), V(0))),
                  'forall P Q, Or P Q -> Or Q P': ('pi', T0, ('pi', T0, ('pi', app(OR, V(0), V(1)), app(OR, V(1), V(0))))), 'forall P, False -> P': ('pi', T0, ('pi', FALSE, V(0)))}
    if which == 'true': false_stmts = true_stmts
    consts = [g[n] for n in ('refl', 'cong', 'symm', 'trans', 'transport', 'le_n', 'le_S', 'le_trans', 'Succ', 'add', 'conj', 'inl', 'inr', 'ex_intro', 'tt', 'em', 'propext', 'funext', 'choice', 'by_contra', 'or_elim', 'false_elim')]
    ind = {g[n] for n in ('Nat', 'Eq', 'Le', 'And', 'Or', 'Ex', 'False', 'True')}
    tried, completed, verified, first = canlib.run(k, g, false_stmts, consts, ind, N, seed, stop_on_verified=(which == 'false'))
    return (tried, completed, first) if which == 'false' else (tried, completed, verified)

if __name__ == '__main__':
    FAIL = []; t0 = time.time()
    k, g = pw.build_prop_world(classical=True)
    fac = lambda: ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=4_000_000)
    bad = fac().check_definitions(); print("classical world: %d globals; trusted axioms %s; independent re-check of every global (axiom types included): %d rejected" % (len(g), pw.CLASSICAL_AXIOMS, len(bad)))
    if bad: FAIL.append('rejected: ' + str(bad[:3]))
    axioms = [nm for gid, nm, _, _ in k.list_globals() if k.global_info(gid)[2] is None and k.global_info(gid)[3] < 0 and nm in pw.CLASSICAL_AXIOMS]
    print("   axioms present:", axioms)
    if sorted(axioms) != sorted(pw.CLASSICAL_AXIOMS): FAIL.append('axiom set')
    C = lambda n: ('const', g[n]); T0, T1 = pw.T0, pw.T1; NOT = C('Not'); EQ, OR, NAT, ZERO = C('Eq'), C('Or'), C('Nat'), C('Zero')
    chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)
    tests = {
      'contrapositive: (~Q -> ~P) -> P -> Q (by_contra)': (chain([T0, T0, ('pi', app(NOT, V(1)), app(NOT, V(0))), V(0)], V(1)),
            [('intro',)] * 4 + [('apply', app(C('by_contra'), V(1))), ('intro',), ('apply', app(V(2), V(4), V(3)))]),
      'double negation introduction (constructive)': (chain([T0, V(0)], app(NOT, app(NOT, V(0)))), [('intro',)] * 3 + [('apply', V(2)), ('apply', V(1))]),
      'excluded middle for a proposition about numbers': (chain([NAT], app(OR, app(EQ, NAT, V(0), ZERO), app(NOT, app(EQ, NAT, V(0), ZERO)))),
            [('intro',), ('apply', app(C('em'), app(EQ, NAT, V(0), ZERO)))]),
    }
    found = {}
    for name, (stmt, ops) in tests.items():
        ps, log = t11.run_ops(k, g, stmt, ops); failed = [l for l in log if not l[1]]
        found[name] = (stmt, parse_raw(k.dump_raw(ps.extract()), True) if (not failed and ps.num_goals() == 0 and ps.is_complete()) else None, failed)
    chk = fac()
    for name, (stmt, proof, failed) in found.items():
        kv = proof is not None and t8.kernel_verdict(k, proof, l2h(stmt, 0)); iv = proof is not None and chk.check_closed(proof, l2h(stmt, 0))[0]
        print("   %-58s kernel %-5s independent %-5s %s" % (name, kv, iv, '' if (kv and iv) else '  <-- ' + str(failed[:1])[:100]))
        if not (kv and iv): FAIL.append(name)
    # F25: apply must REFUSE a solution of the wrong type: choice concludes with a bare Type1 variable, but the goal P lives in Type0
    ps, log = t11.run_ops(k, g, ('pi', T0, ('pi', C('False'), V(0))), [('intro',), ('intro',), ('apply', C('choice'))])
    refused = (not log[-1][1]) and "does not have that argument's type" in log[-1][2]
    print("   apply choice on a Prop goal (a universe mismatch) is refused with a clear message: %s" % refused)
    if not refused: FAIL.append('apply accepted a universe-mismatched solution')
    tried, completed, badn = canary(int(sys.argv[1]) if len(sys.argv) > 1 else 600, 61)
    print("   soundness canary (classical axioms available): %d random proof attempts on 7 FALSE statements, %d completed, verified proof of a false statement: %s" % (tried, completed, badn))
    if badn is not None: FAIL.append('canary proved ' + badn)
    tr2, co2, ver2 = canary(int(sys.argv[1]) if len(sys.argv) > 1 else 600, 62, 'true')
    print("   canary POWER (the same generator on 4 TRUE classical statements): %d attempts, %d completed, %d verified" % (tr2, co2, ver2))
    if ver2 < 5: FAIL.append('canary has no power: only %d true statements proved' % ver2)
    if co2 != ver2: FAIL.append('%d completed proof states did not verify (completed %d, verified %d)' % (co2 - ver2, co2, ver2))
    print("elapsed %.0fs" % (time.time() - t0)); print("S15 CLASSICAL GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
