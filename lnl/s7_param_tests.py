"""S7 gate: the parameter / index distinction (kernel option 'parameters'). Default mode is untouched (the lockstep gates cover it); here the option is ON."""
import sys, random, time, functools
import harness as hn
from harness import *
import s2_tests
import s3_tests as s3
from s3_tests import V, app, Lockstep
import s4_tests as t4
import s6_logic_tests as t6
import oracle_stats as o

T0 = ('type', 0)
chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)

def declare_family_new(knew, name, sort, ctors):
    tid = knew.reserve_inductive_type(name, build_new(knew, l2h(sort, 0))); specs = []
    for cn, ats, ris in ctors:
        ats2 = [s2_tests.subst_self(a, tid) for a in ats]; ris2 = [s2_tests.subst_self(r, tid) for r in ris]
        specs.append(H.CtorSpec(cn, [build_new(knew, l2h(a, j, lenient=True)) for j, a in enumerate(ats2)], [build_new(knew, l2h(r, len(ats2), lenient=True)) for r in ris2]))
    knew.finish_inductive(tid, specs); return tid

def test_analysis():
    kref, knew, g = t6.build_logic(); NAT, Z, S = ('const', g['Nat']), ('const', g['Zero']), ('const', g['Succ']); SELF = ('self',)
    fam = {
      'List':       (('pi', T0, T0), [('nil', [T0], [('var', 0)]), ('cons', [T0, ('var', 0), ('app', SELF, ('var', 0))], [('var', 0)])]),
      'Vec':        (('pi', T0, ('pi', NAT, T0)), [('vnil', [T0], [('var', 0), Z]), ('vcons', [T0, NAT, ('var', 0), app(SELF, ('var', 0), ('var', 1))], [('var', 0), app(S, ('var', 1))])]),
      'NonUniform': (('pi', T0, T0), [('bar', [T0, ('app', SELF, NAT)], [('var', 0)])]),          # the recursive field passes Nat, not its own A: not parametric
      'Rep':        (('pi', T0, ('pi', T0, T0)), [('mk', [T0], [('var', 0), ('var', 0)])]),        # slot 1 repeats slot 0's variable: slot 1 is an index
      'Late':       (('pi', NAT, ('pi', T0, T0)), [('q0', [T0], [Z, ('var', 0)]), ('q1', [NAT, T0], [app(S, ('var', 0)), ('var', 1)])]),   # slot 0 is not a variable: the prefix stops
    }
    for name, (sort, ctors) in fam.items(): declare_family_new(knew, name, sort, ctors)
    g2 = {name: gid for gid, name, _, _ in knew.list_globals()}
    expect = {'Nat': [], 'True': [], 'False': [], 'Eq': [1, 1, 0], 'Le': [1, 0], 'And': [1, 1], 'Or': [1, 1], 'Ex': [1, 1], 'List': [1], 'Vec': [1, 0], 'NonUniform': [0], 'Rep': [1, 0], 'Late': [0, 0]}
    return {n: (knew.param_slots(g2[n]), e) for n, e in expect.items()}

def run_new(knew, stmt, ops):
    ps = knew.start_proof(build_new(knew, l2h(stmt, 0))); log = []
    for op in ops:
        nh = ps.goal_num_hyps(0) if ps.num_goals() else 0
        try:
            if op[0] == 'intro': ps.intro('x')
            elif op[0] == 'cases': ps.cases(op[1])
            elif op[0] == 'induct': ps.induct(op[1])
            elif op[0] == 'apply': ps.apply(build_new(knew, l2h(op[1], nh, lenient=True)))
            log.append((op[0], True, ''))
        except Exception as e: log.append((op[0], False, str(e)[:90]))
    return ps, log

def verifies(knew, ps, stmt):
    return ps.is_complete() and knew.check_open(ps.extract(), build_new(knew, l2h(stmt, 0)), [])[0]

def kit_on(kref, knew, g):
    C = lambda n: ('const', g[n]); NAT, EQ, Z, S, LE = C('Nat'), C('Eq'), C('Zero'), C('Succ'), C('Le')
    eq = lambda a, b: app(EQ, NAT, a, b); le = lambda a, b: app(LE, a, b); FALSE = C('False')
    stmts = {
      'ex_lambda (F5)':        (t6.kit(g, True)[0]['ex_lambda'], t6.kit(g, True)[1]['ex_lambda']),
      'ex_elim (only f, after h, is reverted)':  (t6.kit(g, True)[0]['ex_elim'], t6.kit(g, True)[1]['ex_elim']),
      '0 != S n via Eq':       (chain([NAT, eq(app(S, V(0)), Z)], FALSE), [('intro',), ('intro',), ('cases', 1)]),
      'S n != 0 via Eq (flip)': (chain([NAT, eq(Z, app(S, V(0)))], FALSE), [('intro',), ('intro',), ('cases', 1)]),
      'Le (S n) 0 -> False':   (chain([NAT, le(app(S, V(0)), Z)], FALSE), [('intro',), ('intro',), ('cases', 1)]),
      'Le: goal about b becomes a': (chain([NAT, NAT, le(V(0), V(1))], eq(V(1), V(1))), [('intro',)] * 3 + [('cases', 2), ('apply', C('refl')), ('apply', C('refl'))]),
      'induct on Le a b':      (chain([NAT, NAT, le(V(0), V(1))], le(V(0), app(S, V(1)))), [('intro',)] * 3 + [('induct', 2), ('apply', C('le_S')), ('apply', C('le_n')), ('apply', C('le_S')), ('apply', V(6))]),
      'induct on Le 0 x (concrete PARAMETER, F12 for parameter slots)': (chain([NAT, le(Z, V(0))], le(Z, app(S, V(0)))), [('intro',), ('intro',), ('induct', 1), ('apply', C('le_S')), ('apply', C('le_n')), ('apply', C('le_S')), ('apply', V(5))]),
    }
    return stmts

def soundness_canary(N, seed):
    """random tactic search on FALSE statements, parameters ON: a verified completion would be a soundness bug"""
    kref, knew, g = t6.build_logic(); knew.set_option('parameters', True); rng = random.Random(seed)
    C = lambda n: ('const', g[n]); NAT, EQ, Z, S, LE, FALSE = C('Nat'), C('Eq'), C('Zero'), C('Succ'), C('Le'), C('False')
    eq = lambda a, b: app(EQ, NAT, a, b); le = lambda a, b: app(LE, a, b)
    false_stmts = {'False': FALSE, '0 = 1': eq(Z, app(S, Z)), '1 <= 0': le(app(S, Z), Z), 'forall n, S n <= n': chain([NAT], le(app(S, V(0)), V(0))),
                   'forall n, S n = n': chain([NAT], eq(app(S, V(0)), V(0))), 'forall a b, a <= b': chain([NAT, NAT], le(V(0), V(1))), 'forall a b, a = b': chain([NAT, NAT], eq(V(0), V(1))),
                   # CONDITIONALLY false: the new substitution rule is exercised on a hypothesis (le_n identifies b with a, le_S needs S m <= a)
                   'a <= b -> b <= a': chain([NAT, NAT, le(V(0), V(1))], le(V(1), V(0))), 'a <= b -> a = b': chain([NAT, NAT, le(V(0), V(1))], eq(V(0), V(1))),
                   'a = b -> S a = b': chain([NAT, NAT, eq(V(0), V(1))], eq(app(S, V(0)), V(1))), 'S a = S b -> a = S b': chain([NAT, NAT, eq(app(S, V(0)), app(S, V(1)))], eq(V(0), app(S, V(1)))),
                   'S n <= m -> m <= n': chain([NAT, NAT, le(app(S, V(0)), V(1))], le(V(1), V(0))), 'Or A B -> A': chain([T0, T0, app(C('Or'), V(0), V(1))], V(0)),
                   'Ex A P -> P a (no witness)': chain([T0, ('pi', V(0), T0), V(0), app(C('Ex'), V(0), V(1))], app(V(1), V(2)))}
    consts = [g[n] for n in ('refl', 'cong', 'symm', 'trans', 'transport', 'le_n', 'le_S', 'le_trans', 'Succ', 'add', 'tt', 'conj', 'inl', 'inr', 'ex_intro')]
    ind = {g['Nat'], g['Eq'], g['Le'], g['And'], g['Or'], g['Ex'], g['False'], g['True']}; head = lambda x: head(x[1]) if x[0] == 'app' else x
    tried = completed = 0
    for name, stmt_tree in false_stmts.items():
        stmt = build_new(knew, l2h(stmt_tree, 0))
        for _ in range(N):
            ps = knew.start_proof(stmt); tried += 1
            for _step in range(8):
                if ps.num_goals() == 0: break
                nh = ps.goal_num_hyps(0); tgt = parse_raw(knew.dump_raw(ps.goal_target(0)), True)
                hyps = [parse_raw(knew.dump_raw(ps.goal_hyp_type(0, j)), True) for j in range(nh)]
                indh = [j for j, h in enumerate(hyps) if head(h)[0] == 'const' and head(h)[1] in ind]; r = rng.random()
                try:
                    if tgt[0] == 'pi' and r < 0.4: ps.intro('x')
                    elif indh and r < 0.7: (ps.cases if rng.random() < 0.7 else ps.induct)(rng.choice(indh))
                    elif nh and r < 0.85: ps.apply(build_new(knew, ('fvar', rng.randrange(nh))))
                    else: ps.apply(build_new(knew, ('const', rng.choice(consts))))
                except Exception: ps = None; break
                if ps is None: break
            if ps is not None and ps.num_goals() == 0 and ps.is_complete():
                completed += 1
                if knew.check_open(ps.extract(), stmt, [])[0]: return tried, completed, name           # a VERIFIED proof of a false statement
    return tried, completed, None

if __name__ == '__main__':
    t0 = time.time(); FAIL = []
    a = test_analysis()
    print("parameter slots (found / expected):", {k: v for k, v in a.items()})
    for k, (found, exp) in a.items():
        if found != exp: FAIL.append('analysis %s: %s != %s' % (k, found, exp))
    kref, knew, g = t6.build_logic(); knew.set_option('parameters', True)
    print("option parameters:", knew.get_option('parameters'))
    for name, (stmt, ops) in kit_on(kref, knew, g).items():
        ps, log = run_new(knew, stmt, ops); ok = verifies(knew, ps, stmt)
        extra = ''
        if name.startswith('ex_elim'):
            ps2, _ = run_new(knew, stmt, ops[:6]); extra = ' | hypotheses after cases: %d (expected 10: A P Q h f, the 4 fields, and the reverted copy f\')' % ps2.goal_num_hyps(0); ok = ok and ps2.goal_num_hyps(0) == 10
        if name.startswith('Le: goal'):
            ps2, _ = run_new(knew, stmt, ops[:4]); tgt = parse_raw(knew.dump_raw(ps2.goal_target(0)), True)
            extra = ' | le_n branch goal: %s' % (tgt,); ok = ok and tgt == ('app', ('app', ('app', ('const', g['Eq']), ('const', g['Nat'])), ('fvar', 0)), ('fvar', 0))
        print("   %-64s complete+verifies: %s%s%s" % (name, ok, extra, '' if all(l[1] for l in log) else ' | FAILED STEP %s' % [l for l in log if not l[1]][:1]))
        if not ok: FAIL.append(name)
    # injectivity is NOT provided by cases: the open goal documents the remaining F2b limitation
    inj = chain([('const', g['Nat']), ('const', g['Nat']), app(('const', g['Eq']), ('const', g['Nat']), app(('const', g['Succ']), V(0)), app(('const', g['Succ']), V(1)))], app(('const', g['Eq']), ('const', g['Nat']), V(0), V(1)))
    ps, log = run_new(knew, inj, [('intro',)] * 3 + [('cases', 2)])
    print("   injectivity  S a = S b -> a = b  by cases alone: goals left %d (expected 1: needs injectivity/no-confusion, the rest of F2b)" % ps.num_goals())
    if ps.num_goals() != 1: FAIL.append('injectivity probe changed shape')
    # parameter-free scripts must be IDENTICAL to the reference with the option on
    stmts4 = t4.statements4(g); same_all = True
    for name in ('add_zero_r', 'zero_le', 'add_succ_r'):
        L = Lockstep(kref, knew, g, stmts4[name])
        for op in t4.SCRIPTS4[name](g):
            ok, same, st = L.step(op); same_all = same_all and same
    print("   Nat-only scripts (add_zero_r, zero_le, add_succ_r) with the option ON: states identical to the reference at every step: %s" % same_all)
    if not same_all: FAIL.append('option changed parameter-free behaviour')
    tried, completed, bad = soundness_canary(int(sys.argv[1]) if len(sys.argv) > 1 else 400, 3)
    if bad is None: tried2, completed2, bad = soundness_canary(int(sys.argv[1]) if len(sys.argv) > 1 else 400, 4); tried += tried2; completed += completed2
    print("   soundness canary (parameters ON): %d random proof attempts on 14 FALSE statements, %d completed, verified proof of a false statement: %s" % (tried, completed, bad))
    if bad: FAIL.append('CANARY: verified proof of ' + bad)
    tot, c, v, fb = o.run(int(sys.argv[2]) if len(sys.argv) > 2 else 2500, 11, verbose=False, mode2_only=True, params=True)
    unatt = sum(x[1] + x[2] for x in tot.values()) + (c - v)
    print("   self-consistency oracle (parameters ON, index slots must be variables): scripts %d, unattributed failures %d, completed %d verified %d" % (int(sys.argv[2]) if len(sys.argv) > 2 else 2500, unatt, c, v))
    if unatt: FAIL.append('oracle in parameter mode: %s' % list(fb.items())[:1])
    print("elapsed %.1fs" % (time.time() - t0)); print("S7 PARAM GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
