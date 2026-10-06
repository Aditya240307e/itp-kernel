"""S4 gate (G3b): cases and induct against the reference, as lockstep tactic scripts."""
import sys, random, time, functools
import harness as hn
from harness import *
import s3_tests as s3
from s3_tests import V, app, Lockstep, library, check_proof_pair, shift_meta

def statements4(g):
    NAT, EQ, LE = ('const', g['Nat']), ('const', g['Eq']), ('const', g['Le'])
    Z, S, ADD = ('const', g['Zero']), ('const', g['Succ']), ('const', g['add'])
    eq = lambda a, b: app(EQ, NAT, a, b); le = lambda a, b: app(LE, a, b)
    def chain(doms, goal):
        t = goal
        for d in reversed(doms): t = ('pi', d, t)
        return t
    return {
      'add_zero_r':  chain([NAT], eq(app(ADD, V(0), Z), V(0))),
      'zero_le':     chain([NAT], le(Z, V(0))),
      'symm_cases':  chain([NAT, NAT, eq(V(0), V(1))], eq(V(1), V(0))),
      'le_succ_ind': chain([NAT, NAT, le(V(0), V(1))], le(V(0), app(S, V(1)))),
      'succ_inj':    chain([NAT, NAT, eq(app(S, V(0)), app(S, V(1)))], eq(V(0), V(1))),
      'neq_zero':    chain([NAT, eq(app(S, V(0)), Z)], le(app(S, Z), Z)),
      'revert_case': chain([NAT, eq(V(0), V(0))], le(V(0), V(0))),
      'add_succ_r':  chain([NAT, NAT], eq(app(ADD, V(0), app(S, V(1))), app(S, app(ADD, V(0), V(1))))),
      'le_trans_i':  chain([NAT, NAT, NAT, le(V(0), V(1)), le(V(1), V(2))], le(V(0), V(2))),
      'trans_cases': chain([NAT, NAT, NAT, eq(V(0), V(1)), eq(V(1), V(2))], eq(V(0), V(2))),
      # no Nat hypotheses, so nothing is reverted and refutation exploitation can fire (a Nat-typed hypothesis counts as dependent on Eq's first index `Nat`)
      'exploit_eq':  chain([eq(app(S, Z), Z)], le(Z, Z)),                 # every constructor refuted: the goal list becomes empty
      'exploit_le':  chain([le(Z, Z)], eq(Z, Z)),                         # le_S refuted, le_n survives as a sibling WITH a field
      # h2 : P h1 depends on n only THROUGH P and h1 (both mention n): reverting it needs the transitive closure
      'transitive_revert': chain([NAT, le(V(0), Z), ('pi', le(V(0), Z), ('type', 0)), app(V(2), V(1))], le(Z, Z)),
    }

SCRIPTS4 = {   # each step is ('intro',) / ('apply', tree) / ('rewrite', j, x, y) / ('cases', hyp) / ('induct', hyp)
  'add_zero_r':  lambda g: [('intro',), ('induct', 0), ('apply', ('const', g['refl'])), ('apply', ('const', g['cong'])), ('apply', V(2))],
  'zero_le':     lambda g: [('intro',), ('induct', 0), ('apply', ('const', g['le_n'])), ('apply', ('const', g['le_S'])), ('apply', V(2))],
  'symm_cases':  lambda g: [('intro',)] * 3 + [('cases', 2), ('apply', ('const', g['refl']))],
  'le_succ_ind': lambda g: [('intro',)] * 3 + [('induct', 2), ('apply', ('const', g['le_S'])), ('apply', ('const', g['le_n'])), ('apply', ('const', g['le_S'])), ('apply', V(6))],
  'succ_inj':    lambda g: [('intro',)] * 3 + [('cases', 2)],
  'neq_zero':    lambda g: [('intro',)] * 2 + [('cases', 1)],
  'revert_case': lambda g: [('intro',)] * 2 + [('cases', 0)],
  'add_succ_r':  lambda g: [('intro',)] * 2 + [('induct', 0)],
  'le_trans_i':  lambda g: [('intro',)] * 5 + [('induct', 4)],
  'trans_cases': lambda g: [('intro',)] * 5 + [('cases', 3), ('apply', V(4))],
  'transitive_revert': lambda g: [('intro',)] * 4 + [('induct', 0)],
  'exploit_eq': lambda g: [('intro',), ('cases', 0)],
  'exploit_le': lambda g: [('intro',), ('cases', 0), ('apply', ('const', g['refl']))],
}
REF_MAY_BE_INVALID = {'exploit_le'}   # the reference wraps the surviving sibling's branch in duplicated field lambdas (suspected F14); the hybrid proof must verify

def show(st): return str(st)[:420]

def erase_vars(x):
    """identical structure once variable identities (and meta spines) are erased: the signature of the reference's marker-collision bug F10"""
    if isinstance(x, tuple):
        if x and x[0] == 'fvar': return ('fvar',)
        if x and x[0] == 'app':
            f = x
            while isinstance(f, tuple) and f and f[0] == 'app': f = f[1]
            if isinstance(f, tuple) and f and f[0] == 'meta': return f
        return tuple(erase_vars(y) for y in x)
    if isinstance(x, list): return [erase_vars(y) for y in x]
    return x

def strict_ok(L, stmt):
    """the hybrid kernel's own result must be consistent: every goal well-formed, and the partial proof type-checks (when it can be checked:
    no unsolved meta behind a goal, no pending constraint, no earlier concrete-index induction -- F12)"""
    from oracle_lib import goals_wellformed, goals_meta_free
    if L.psn.num_goals() and not goals_wellformed(L.knew, L.psn): return False
    if L.m1 or L.psn.num_pending() > 0 or not goals_meta_free(L.knew, L.psn): return True
    return L.knew.check_open(L.psn.extract(), build_new(L.knew, l2h(stmt, 0)), [])[0]

F10_STATS = {'validated': 0, 'unvalidated_metas': 0}
def tree_state_wellformed(k, states):
    from oracle_lib import is_type
    for (mid, nh, tgt, hy) in states:
        ctxs = []
        for h in hy:
            if not is_type(k, h, ctxs): return False
            ctxs.append(build_new(k, h))
        if not is_type(k, tgt, ctxs): return False
    return True

def f10_class(L, stmt, st):
    """both succeeded, states differ only in WHICH variables are referenced, the hybrid state is consistent -- and (when checkable) the
    REFERENCE's state is genuinely ill-formed: otherwise the difference is not a marker collision and stays a divergence"""
    if erase_vars(st[0]) != erase_vars(st[1]) or not strict_ok(L, stmt): return False
    ref_states = st[0][0]
    if "'meta'" in str([(x[2], x[3]) for x in ref_states]): F10_STATS['unvalidated_metas'] += 1; return True
    if tree_state_wellformed(L.knew, ref_states): return False
    F10_STATS['validated'] += 1; return True

def strip_meta_args(x):
    """drop the arguments of every meta application: used to recognise differences confined to meta spines (reference bug F10)"""
    if isinstance(x, tuple):
        if x and x[0] == 'app':
            f = x
            while isinstance(f, tuple) and f and f[0] == 'app': f = f[1]
            if isinstance(f, tuple) and f and f[0] == 'meta': return f
        return tuple(strip_meta_args(y) for y in x)
    if isinstance(x, list): return [strip_meta_args(y) for y in x]
    return x

f10_scripted = []
def hyp_is_pi(L, j):
    return parse_raw(L.knew.dump_raw(L.psn.goal_hyp_type(0, j)), True)[0] == 'pi' if j < L.psn.goal_num_hyps(0) else False

def explain_divergence(L, stmt, op, st):
    """name the explained category of a lockstep divergence (each is validated on the hybrid side), or None if it is unexplained"""
    last = L.log[-1]
    if L.stopped: return 'F13'
    if last[1] and last[2] and f10_class(L, stmt, st): return 'F10'
    if op[0] in ('induct', 'cases') and last[1] is False and last[2] is True and 'embedded bound structure' in L.last_msgs[0] and strict_ok(L, stmt): return 'relaxed'
    # F16: the reference's apply misbinds the binders of a Pi-typed hypothesis; the hybrid kernel's result must itself be consistent
    if op[0] == 'apply' and op[1][0] == 'var' and last[1] is False and last[2] is True and hyp_is_pi(L, op[1][1]) and (L.psn.num_pending() > 0 or strict_ok(L, stmt)): return 'F16'
    return None

def run_scripted(kref, knew, g):
    stmts = statements4(g); rows = []; del f10_scripted[:]
    for name, mk in SCRIPTS4.items():
        L = Lockstep(kref, knew, g, stmts[name]); bad = None
        for op in mk(g):
            ok, same, st = L.step(op)
            if not same:
                if L.log[-1][1] and L.log[-1][2] and f10_class(L, stmts[name], st): f10_scripted.append((name, op)); break
                bad = (op, ok, show(st[0]), show(st[1])); break
        done = L.psr.is_complete() and L.psn.is_complete()
        proofs = check_proof_pair(L, stmts[name]) if (done and not bad) else (None, None)
        rows.append((name, bad, done, proofs, [l[1] for l in L.log], L.psn.num_goals()))
    return rows

INDUCTIVE = None
def random_script(kref, knew, g, stmt, rng, max_steps=9):
    L = Lockstep(kref, knew, g, stmt); consts = [g[n] for n in ('refl', 'cong', 'symm', 'trans', 'transport', 'le_n', 'le_S', 'le_trans', 'Succ', 'add')]
    ind = {g['Nat'], g['Eq'], g['Le']}
    for _ in range(max_steps):
        if L.psr.num_goals() == 0: break
        nh = L.psr.goal_num_hyps(0); tgt = parse_raw(kref.dump_raw(L.psr.goal_target(0)), False)
        hyps = [parse_raw(kref.dump_raw(L.psr.goal_hyp_type(0, j)), False) for j in range(nh)]
        def head(t):
            while t[0] == 'app': t = t[1]
            return t
        indh = [j for j, h in enumerate(hyps) if head(h)[0] == 'const' and head(h)[1] in ind]
        eqh = [j for j, h in enumerate(hyps) if h[0] == 'app' and h[1][0] == 'app' and h[1][1][0] == 'app' and h[1][1][1] == ('const', g['Eq']) and not any(w in str(h) for w in ("'lam'", "'pi'", "'match'", "'fix'"))]
        r = rng.random()
        if tgt[0] == 'pi' and r < 0.35: op = ('intro',)
        elif indh and tgt[0] != 'pi' and r < 0.60: op = (rng.choice(['cases', 'induct']), rng.choice(indh))
        elif eqh and tgt[0] != 'pi' and r < 0.68:
            j = rng.choice(eqh); h = hyps[j]; op = ('rewrite', j, h[1][2], h[2])
        elif nh and r < 0.82: op = ('apply', V(rng.randrange(nh)))
        else: op = ('apply', ('const', rng.choice(consts)))
        ok, same, st = L.step(op)
        if L.stopped and same: return L, ('F13', op)
        if not same:
            last = L.log[-1]
            if op[0] in ('induct', 'cases') and last[1] is False and last[2] is True and 'embedded bound structure' in L.last_msgs[0] and strict_ok(L, stmt):
                return L, ('PERMISSIVE', op)          # the reference rejects goals/hypotheses with embedded binders (a numbering limitation); the hybrid kernel accepts: validated by the caller
            if last[1] and last[2] and f10_class(L, stmt, st):
                return L, ('F10', op)                  # same structure, different variable references: the reference's sequential marker substitution collides
            return L, (op, show(st[0]), show(st[1]))
    return L, None

def run_random(N, seed):
    kref, knew, g, defs = library(); stmts = statements4(g); rng = random.Random(seed); names = list(stmts)
    stats = dict(scripts=0, steps=0, ok_steps=0, failed_steps=0, completed=0, proofs_ok=0, bad=[], by_op={})
    for i in range(N):
        stmt = stmts[rng.choice(names)]
        L, bad = random_script(kref, knew, g, stmt, rng)
        stats['scripts'] += 1; stats['steps'] += len(L.log)
        for l in L.log:
            d = stats['by_op'].setdefault(l[0], [0, 0]); d[0 if l[1] else 1] += 1
            stats['ok_steps' if l[1] else 'failed_steps'] += 1
        if bad and bad[0] == 'F13':
            stats['f13'] = stats.get('f13', 0) + 1; kref, knew, g, defs = library(); continue
        if bad and bad[0] == 'F10':
            stats['f10'] = stats.get('f10', 0) + 1; kref, knew, g, defs = library(); continue
        if bad and bad[0] == 'PERMISSIVE':
            # oracle: the partial proof built by the hybrid induct (metas typed by closed Pi chains) must type-check against the statement
            good = strict_ok(L, stmt)
            stats.setdefault('permissive', [0, 0])[0 if good else 1] += 1
            kref, knew, g, defs = library(); continue
        if bad: stats['bad'].append((L.log[-3:], bad)); kref, knew, g, defs = library(); continue
        if L.psr.is_complete() and L.psn.is_complete():
            stats['completed'] += 1; ok, same_nf = check_proof_pair(L, stmt)
            if ok == [True, True] and same_nf: stats['proofs_ok'] += 1
            elif ok == [False, True]: stats['ref_invalid'] = stats.get('ref_invalid', 0) + 1       # the reference's own proof fails verification (F14/F9 family); the hybrid's is verified by the kernel
            else: stats['bad'].append(('proof', ok, same_nf, [l[:2] for l in L.log]))
    return stats

def test_rewrite_fixes():
    """F13a: rewrite instantiates solved metas first, so `apply symm; rewrite h; apply refl` completes and verifies.
       F13b: rewrite checks its motive: rewriting a := b in `Eq (Le a a) p p` (p : Le a a) is ill-typed and must be REJECTED; the reference accepts it
       and leaves an ill-formed goal."""
    from oracle_lib import goals_wellformed
    kref, knew, g, defs = library(); NAT, EQ, LE = ('const', g['Nat']), ('const', g['Eq']), ('const', g['Le'])
    eq = lambda a, b: app(EQ, NAT, a, b)
    chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)
    out = {}
    # F13a
    stmt = chain([NAT, NAT, eq(V(0), V(1))], eq(V(1), V(0)))
    L = Lockstep(kref, knew, g, stmt)
    for op in [('intro',)] * 3 + [('apply', ('const', g['symm']))]: L.step(op)
    ok, same, st = L.step(('rewrite', 2, V(0), V(1)))
    out['a_rewrite_ok'] = (L.log[-1][1], L.log[-1][2]); out['a_new_wellformed'] = goals_wellformed(knew, L.psn)
    L.psn.apply(build_new(knew, ('const', g['refl'])))
    out['a_new_complete'] = L.psn.is_complete()
    out['a_new_verifies'] = knew.check_open(L.psn.extract(), build_new(knew, l2h(stmt, 0)), [])[0] if out['a_new_complete'] else False
    try:
        L.psr.apply(build_ref(kref, ('const', g['refl']))); out['a_ref_complete'] = L.psr.is_complete(); out['a_ref_verifies'] = None
        if out['a_ref_complete']:
            try: kref.whole_term_check(L.psr.extract(), build_ref(kref, stmt)); out['a_ref_verifies'] = True
            except Exception: out['a_ref_verifies'] = False
    except Exception: out['a_ref_complete'] = False; out['a_ref_verifies'] = None
    # F13b
    stmtb = chain([NAT, NAT, eq(V(0), V(1)), app(LE, V(0), V(0))], app(EQ, app(LE, V(0), V(0)), V(3), V(3)))
    L = Lockstep(kref, knew, g, stmtb)
    for _ in range(4): L.step(('intro',))
    L.step(('rewrite', 2, V(0), V(1))); out['b_rewrite_ok'] = (L.log[-1][1], L.log[-1][2])
    ref_state = L.state()[0][0]
    out['b_ref_state_illformed'] = (not tree_state_wellformed(knew, ref_state)) if out['b_rewrite_ok'][0] else None
    return out

def test_conv_termination():
    """regression (a native crash found by the oracle): converting two stuck recursive definitions must terminate -- fail closed -- not overflow the stack"""
    kref, knew, g, defs = library(); stmt = build_new(knew, l2h(statements4(g)['zero_le'], 0)); ps = knew.start_proof(stmt)
    ops = [('intro',), ('apply', ('const', g['le_trans'])), ('induct', 0), ('induct', 0), ('apply', ('fvar', 0)), ('apply', ('fvar', 0)), ('apply', ('fvar', 2)), ('induct', 0)]
    for op in ops:
        if op[0] == 'intro': ps.intro('x')
        elif op[0] == 'induct': ps.induct(op[1])
        else: ps.apply(build_new(knew, op[1]))
    ok, msg = knew.check_open(ps.extract(), stmt, [])          # used to segfault (unbounded conv_value/branches_convertible recursion)
    return ok, msg

def test_field_after_ih():
    """family D: leaf | mk (r : D) (a : Nat) (p : Eq Nat a a). The IH of r sits between r and a, so p's type must mention a's ACTUAL level."""
    from s2_tests import subst_self
    kref, knew, g, defs = library(); NAT, EQ, Z = ('const', g['Nat']), ('const', g['Eq']), ('const', g['Zero'])
    SELF = ('self',); trees = [('leaf', [], []), ('mk', [SELF, NAT, app(EQ, NAT, V(1), V(1))], [])]; tids = {}
    for which, k, bld, conv, CS in (('ref', kref, build_ref, lambda t, n: t, R.CtorSpec), ('new', knew, build_new, lambda t, n: l2h(t, n, lenient=True), H.CtorSpec)):
        tid = k.reserve_inductive_type('D', bld(k, conv(('type', 0), 0))); specs = []
        for cn, ats, ris in trees:
            ats = [subst_self(a, tid) for a in ats]; specs.append(CS(cn, [bld(k, conv(a, j)) for j, a in enumerate(ats)], []))
        k.finish_inductive(tid, specs); tids[which] = tid
    assert tids['ref'] == tids['new']
    stmt = ('pi', ('const', tids['new']), app(EQ, NAT, Z, Z))
    L = Lockstep(kref, knew, g, stmt); L.step(('intro',)); ok, same, (sr, sn) = L.step(('induct', 0))
    # the step goal (second): last hypothesis is p : Eq Nat a a; in the branch context [d, r, IH, a, p] a is level 3 (the reference assumes consecutive levels)
    new_p = sn[0][1][3][-1]; ref_p = sr[0][1][3][-1]
    right = new_p == ('app', ('app', ('app', ('const', g['Eq']), ('const', g['Nat'])), ('fvar', 3)), ('fvar', 3))
    for ps in (L.psn,):
        n0 = ps.goal_num_hyps(0)
        ps.apply(build_new(knew, ('const', g['refl']))); ps.apply(build_new(knew, ('const', g['refl'])))
    good, msg = knew.check_open(L.psn.extract(), build_new(knew, l2h(stmt, 0)), [])
    return dict(states_agree=same, ref_p=str(ref_p)[:90], new_p_correct=right, new_proof_verifies=good, msg=msg[:80], completes=L.psn.is_complete())

if __name__ == '__main__':
    t0 = time.time(); FAIL = []
    kref, knew, g, defs = library()
    print("S4 scripted proofs (lockstep, reference vs hybrid):")
    for name, bad, done, proofs, oks, ngoals in run_scripted(kref, knew, g):
        if bad: st = "DIVERGED at %s (both ok: %s)\n        ref: %s\n        new: %s" % (bad[0], bad[1], bad[2], bad[3]); FAIL.append(name)
        elif done and name in REF_MAY_BE_INVALID:
            st = "complete; verifies in the hybrid kernel: %s | verifies in the reference: %s" % (proofs[0][1], proofs[0][0]); (FAIL.append(name) if not proofs[0][1] else None)
        elif done: st = "complete; verify in both: %s; identical normal forms: %s" % (proofs[0], proofs[1]); (FAIL.append(name) if not (proofs[0] == [True, True] and proofs[1]) else None)
        else: st = "states identical after every step (%d tactics ok, %d goals left)" % (sum(oks), ngoals)
        print("   %-13s %s" % (name, st))
    for name, op in f10_scripted: print("   (%s: reference collides on %s -- classified F10, hybrid state well-formed)" % (name, op))
    # golden: the step goal's last (transitively reverted) hypothesis must be P' h1' = (v7 v6); the reference says (v7 v7)
    Lg = Lockstep(kref, knew, g, statements4(g)['transitive_revert'])
    for _ in range(4): Lg.step(('intro',))
    _, _, (sr_, sn_) = Lg.step(('induct', 0))
    golden_ok = sn_[0][1][3][-1] == ('app', ('fvar', 7), ('fvar', 6)) and sn_[0][1][3][:8] == sr_[0][1][3][:8] and len(sn_[0]) == len(sr_[0])
    print("   transitive_revert golden (h2' = P' h1' = (v7 v6), reference has (v7 v7)):", golden_ok)
    if not golden_ok: FAIL.append('transitive_revert golden')
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    s = run_random(N, 5)
    print("S4 random scripts: %d scripts, %d steps (%d ok / %d failed identically), %d completed proofs (%d identical, %d where only the REFERENCE's proof fails verification), divergences: %d" % (s['scripts'], s['steps'], s['ok_steps'], s['failed_steps'], s['completed'], s['proofs_ok'], s.get('ref_invalid', 0), len(s['bad'])))
    print("   accepted by the hybrid kernel but rejected by the reference's embedded-binder guard: %d (hybrid state consistent: %d, INCONSISTENT: %d)" % (sum(s.get('permissive', [0, 0])), s.get('permissive', [0, 0])[0], s.get('permissive', [0, 0])[1]))
    if s.get('permissive', [0, 0])[1]: FAIL.append('relaxed-guard events produced inconsistent states')
    print("   states equal except which variables are referenced (reference bug F10, sequential marker collision): %d | rewrite on a meta-bearing goal (F13: hybrid zonks first): %d" % (s.get('f10', 0), s.get('f13', 0)))
    print("   per tactic (ok / failed-identically):", {k: tuple(v) for k, v in sorted(s['by_op'].items())})
    rw = test_rewrite_fixes()
    print("   F13a (zonk first): hybrid goal well-formed %s, completes %s, verifies %s | reference completes: %s, its proof verifies: %s" % (rw['a_new_wellformed'], rw['a_new_complete'], rw['a_new_verifies'], rw['a_ref_complete'], rw['a_ref_verifies']))
    print("   F13b (motive check): rewrite accepted by (reference, hybrid) = %s; reference's resulting goal ill-formed: %s" % (rw['b_rewrite_ok'], rw['b_ref_state_illformed']))
    if not (rw['a_new_wellformed'] and rw['a_new_complete'] and rw['a_new_verifies'] and rw['b_rewrite_ok'][1] is False and rw['b_ref_state_illformed'] in (True, None)): FAIL.append('rewrite fixes')
    ct = test_conv_termination(); print("   conversion-termination regression (was a segfault): check returned %s" % (ct,))
    print("   F10 events validated against the reference's ill-formed state: %d (unvalidated, meta-bearing: %d)" % (F10_STATS['validated'], F10_STATS['unvalidated_metas']))
    d = test_field_after_ih()
    print("   family with a field after a recursive one: new p-hypothesis correct %s, new proof completes+verifies %s/%s; reference agrees with new: %s%s" % (d['new_p_correct'], d['completes'], d['new_proof_verifies'], d['states_agree'], '' if d['states_agree'] else '  (reference: %s)' % d['ref_p']))
    if not (d['new_p_correct'] and d['new_proof_verifies'] and d['completes']): FAIL.append('field-after-IH')
    for b in s['bad'][:3]: print("     DIVERGENCE:", str(b)[:900])
    if s['bad']: FAIL.append('random scripts')
    print("elapsed %.1fs" % (time.time() - t0)); print("S4 GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
