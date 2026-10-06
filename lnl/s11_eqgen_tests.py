"""F12 feature gate: index generalisation with equations (induct on a hypothesis whose index is a concrete term)."""
import sys, functools
sys.setrecursionlimit(30000)
import harness as hn
from harness import *
from s3_tests import V, app
import s6_logic_tests as t6
import s8_indep_tests as t8
import indep_checker as ic
import noconf

def run_ops(k, g, stmt, ops):
    ps = k.start_proof(build_new(k, l2h(stmt, 0))); log = []
    for op in ops:
        nh = ps.goal_num_hyps(0) if ps.num_goals() else 0
        try:
            if op[0] == 'intro': ps.intro('x')
            elif op[0] == 'cases': ps.cases(op[1])
            elif op[0] == 'induct': ps.induct(op[1])
            elif op[0] == 'induct_eq': ps.induct(op[1], eq_gid=g['Eq'], refl_gid=g['refl'])
            else: ps.apply(build_new(k, l2h(op[1], nh, lenient=True)))
            log.append((op[0], True, ''))
        except Exception as e: log.append((op[0], False, str(e)))
    return ps, log

def show(k, ps, nm, which):
    h = lambda t: ('(' + ' '.join([h(t[1])] + [h(t[2])]) + ')') if False else None
    def pp(t):
        kd = t[0]
        if kd == 'const': return nm.get(t[1], 'c%d' % t[1])
        if kd == 'fvar': return 'v%d' % t[1]
        if kd == 'app':
            a = []
            while t[0] == 'app': a.append(t[2]); t = t[1]
            return '(' + ' '.join([pp(t)] + [pp(x) for x in reversed(a)]) + ')'
        if kd == 'pi': return '(Pi ' + pp(t[1]) + '. ' + pp(t[2]) + ')'
        return kd
    nh = ps.goal_num_hyps(which)
    return '[' + ', '.join('%d:%s' % (j, pp(parse_raw(k.dump_raw(ps.goal_hyp_type(which, j)), True))) for j in range(nh)) + '] |- ' + pp(parse_raw(k.dump_raw(ps.goal_target(which)), True))

def run(params, verbose=False):
    kref, knew, g = t6.build_logic()
    if params: knew.set_option('parameters', True)
    gids = {n: g[n] for n in ('True', 'Eq', 'transport', 'refl')}
    fac = lambda: ic.Checker(t8.build_env(knew), params=params, fuel=4_000_000)
    noconf.generate_injectivity(knew, build_new, parse_raw, fac, g['Nat'], gids)
    g = {name: gid for gid, name, _, _ in knew.list_globals()}; nm = {gid: name for name, gid in g.items()}
    C = lambda n: ('const', g[n]); NAT, ZERO, S, LE, EQ, OR = C('Nat'), C('Zero'), C('Succ'), C('Le'), C('Eq'), C('Or')
    eq = lambda A, a, b: app(EQ, A, a, b); le = lambda a, b: app(LE, a, b)
    chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)
    cases = {}
    t1 = chain([NAT, le(V(0), ZERO)], eq(NAT, V(0), ZERO))                                                         # forall n, n <= 0 -> n = 0
    cases['n <= 0 -> n = 0   (induct; base: the equation IS the goal; step: S m = 0 is absurd)'] = (t1, [('intro',), ('intro',), ('induct_eq', 1), ('apply', V(3)), ('cases', 6)])
    t2 = chain([NAT, NAT, le(V(1), app(S, V(0)))], app(OR, le(V(1), V(0)), eq(NAT, V(1), app(S, V(0)))))          # Le inversion: b <= S a -> b <= a \/ b = S a
    transp = app(C('transport'), NAT, ('lam', NAT, app(LE, V(1), V(8))), V(4), V(0), app(C('Succ_inj_0'), V(4), V(0), V(7)), V(5))
    cases['b <= S a -> b <= a \\/ b = S a   (equation + injectivity of S + transport)'] = (t2, [('intro',)] * 3 + [('induct_eq', 2), ('apply', C('inr')), ('apply', V(4)), ('apply', C('inl')), ('apply', transp)])
    out = {}; proofs = []
    for name, (stmt, ops) in cases.items():
        ps, log = run_ops(knew, g, stmt, ops)
        failed = [l for l in log if not l[1]]
        proof = parse_raw(knew.dump_raw(ps.extract()), True) if (not failed and ps.num_goals() == 0 and ps.is_complete()) else None
        if verbose and proof is None: print("   ", name, "log:", failed[:2], "| goals left:", ps.num_goals())
        out[name] = (proof is not None and t8.kernel_verdict(knew, proof, l2h(stmt, 0)), False)
        if proof is not None:
            proofs.append((name, proof, l2h(stmt, 0)))
    chk = fac()
    for name, proof, st in proofs: out[name] = (out[name][0], chk.check_closed(proof, st)[0])
    # what the user sees after induct on  n <= 0
    ps, _ = run_ops(knew, g, t1, [('intro',), ('intro',), ('induct_eq', 1)])
    if verbose and ps.num_goals() == 2: print("   goal 1:", show(knew, ps, nm, 0), "\n   goal 2:", show(knew, ps, nm, 1))
    return knew, out, proofs

def legacy_partial(verbose=False):
    """legacy mode cannot discriminate S m = 0 (F2b), so the proofs cannot complete there: what must hold is that every partial proof after induct is well typed"""
    kref, knew, g = t6.build_logic()
    gids = {n: g[n] for n in ('True', 'Eq', 'transport', 'refl')}
    noconf.generate_injectivity(knew, build_new, parse_raw, lambda: ic.Checker(t8.build_env(knew)), g['Nat'], gids)
    g = {name: gid for gid, name, _, _ in knew.list_globals()}
    C = lambda n: ('const', g[n]); NAT, ZERO, S, LE, EQ, OR = C('Nat'), C('Zero'), C('Succ'), C('Le'), C('Eq'), C('Or')
    eq = lambda A, a, b: app(EQ, A, a, b); le = lambda a, b: app(LE, a, b)
    chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)
    stmts = {'n <= 0 -> n = 0': (chain([NAT, le(V(0), ZERO)], eq(NAT, V(0), ZERO)), 2, 1),
             'b <= S a -> Or (b <= a) (b = S a)': (chain([NAT, NAT, le(V(1), app(S, V(0)))], app(OR, le(V(1), V(0)), eq(NAT, V(1), app(S, V(0))))), 3, 2)}
    out = {}
    for name, (stmt, nintro, hyp) in stmts.items():
        st = build_new(knew, l2h(stmt, 0)); ps = knew.start_proof(st)
        for _ in range(nintro): ps.intro('x')
        ps.induct(hyp, eq_gid=g['Eq'], refl_gid=g['refl'])
        out[name] = (knew.check_open(ps.extract(), st, [])[0], ps.num_goals() == 2)
    return out

if __name__ == '__main__':
    FAIL = []
    knew, out, proofs = run(True, verbose=True)
    print("parameter mode (completed proofs; kernel and independent checker):")
    for k, (kv, iv) in out.items():
        print("   %-86s kernel %-5s independent %s" % (k, kv, iv))
        if not (kv and iv): FAIL.append('params ' + k)
    _, st = t8.differential('eqgen', knew, proofs, True, 300, 5)
    print("   mutated proofs: %d mutants, verdicts agree %d, kernel accepts / checker rejects %d, checker accepts / kernel rejects %d" % (st['mutants'], st['agree'], len(st['kernel_accepts_indep_rejects']), len(st['indep_accepts_kernel_rejects'])))
    if st['kernel_accepts_indep_rejects'] or st['indep_accepts_kernel_rejects'] or st['mutants'] < 100: FAIL.append('differential on the equation proofs')
    # the barrier exists now: the library index must not offer it as an apply candidate (its conclusion is a bare variable, so it would match every goal)
    names = {gid: name for gid, name, _, _ in knew.list_globals()}
    barrier_gids = {gid for gid, nm_ in names.items() if nm_.startswith('__barrier_')}
    lib = knew.index_library()
    nat_gid = [g_ for g_, n_ in names.items() if n_ == 'Nat'][0]; zero_gid = [g_ for g_, n_ in names.items() if n_ == 'Zero'][0]
    probe = ('pi', ('pi', ('const', nat_gid), ('type', 0)), app(V(0), ('const', zero_gid)))      # forall (P : Nat -> Type0), P Zero : the goal's head is the context variable at level 0
    ps0 = knew.start_proof(build_new(knew, l2h(probe, 0))); ps0.intro('x')
    cands = [parse_raw(knew.dump_raw(c), True) for c in knew.apply_candidates(ps0, lib)]
    leaked = [c for c in cands if c[0] == 'const' and c[1] in barrier_gids]
    print("   a barrier global exists: %s; offered by the library index: %s" % (bool(barrier_gids), bool(leaked)))
    if not barrier_gids or leaked: FAIL.append('barrier missing or leaked into the library index')
    print("legacy mode (partial proofs after induct with equations must be well typed; two goals expected):")
    for k, (wt, two) in legacy_partial().items():
        print("   %-50s well-typed %-5s two branch goals %s" % (k, wt, two))
        if not (wt and two): FAIL.append('legacy partial ' + k)
    print("S11 EQGEN GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
