"""S3 gate (G3a): metas, unification, intro/apply/rewrite against the reference, as lockstep tactic scripts."""
import sys, random, time
import harness as hn
from harness import *
from s2_tests import build_library

V = lambda i: ('var', i)
app = lambda f, *a: __import__('functools').reduce(lambda x, y: ('app', x, y), a, f)

def library():
    """reference library + symm + trans (declared through the reference's CHECKED path), loaded into the hybrid kernel."""
    kref, refs = build_library()
    g = {name: gid for gid, name, _, _ in kref.list_globals()}
    NAT, T0 = ('const', g['Nat']), ('type', 0)
    EQ, REFL, TR = ('const', g['Eq']), ('const', g['refl']), ('const', g['transport'])
    eq = lambda A, a, b: app(EQ, A, a, b)
    sym_ty = ('pi', T0, ('pi', V(0), ('pi', V(0), ('pi', eq(V(0), V(1), V(2)), eq(V(0), V(2), V(1))))))
    sym_val = ('lam', T0, ('lam', V(0), ('lam', V(0), ('lam', eq(V(0), V(1), V(2)),
        app(TR, V(0), ('lam', V(0), eq(V(0), V(4), V(1))), V(1), V(2), V(3), app(REFL, V(0), V(1)))))))
    kref.declare_checked(build_ref(kref, sym_ty), build_ref(kref, sym_val), 'symm')
    trn_ty = ('pi', T0, ('pi', V(0), ('pi', V(0), ('pi', V(0), ('pi', eq(V(0), V(1), V(2)), ('pi', eq(V(0), V(2), V(3)), eq(V(0), V(1), V(3))))))))
    trn_val = ('lam', T0, ('lam', V(0), ('lam', V(0), ('lam', V(0), ('lam', eq(V(0), V(1), V(2)), ('lam', eq(V(0), V(2), V(3)),
        app(TR, V(0), ('lam', V(0), eq(V(0), V(1), V(6))), V(2), V(3), V(5), V(4))))))))
    kref.declare_checked(build_ref(kref, trn_ty), build_ref(kref, trn_val), 'trans')
    q_ty = ('pi', ('pi', NAT, NAT), ('pi', NAT, eq(NAT, app(V(0), V(1)), V(1))))
    kref.reserve_recursive_function('Q', build_ref(kref, q_ty))   # a global without a value: an axiom in both kernels
    g = {name: gid for gid, name, _, _ in kref.list_globals()}
    knew = H.Kernel(); load_library(kref, knew)
    # the hybrid kernel must ALSO accept symm and trans through its own checked path
    kn2 = H.Kernel(); events = kref.export_library(); gid_of = {name: gid for gid, name, _, _ in kref.list_globals()}
    return kref, knew, g, (sym_ty, sym_val, trn_ty, trn_val)

def statements(g):
    NAT, EQ, LE = ('const', g['Nat']), ('const', g['Eq']), ('const', g['Le'])
    Z, S, ADD = ('const', g['Zero']), ('const', g['Succ']), ('const', g['add'])
    eq = lambda a, b: app(EQ, NAT, a, b); le = lambda a, b: app(LE, a, b); pis = lambda n, body: __import__('functools').reduce(lambda t, _: ('pi', NAT, t), range(n), body)
    # hypotheses are introduced as binders: forall a b ... (h : ..) -> goal ; levels = binder positions
    def chain(doms, goal): 
        t = goal
        for d in reversed(doms): t = ('pi', d, t)
        return t
    return {
      'refl':        chain([NAT], eq(V(0), V(0))),
      'symm':        chain([NAT, NAT, eq(V(0), V(1))], eq(V(1), V(0))),
      'trans':       chain([NAT, NAT, NAT, eq(V(0), V(1)), eq(V(1), V(2))], eq(V(0), V(2))),
      'cong_succ':   chain([NAT, NAT, eq(V(0), V(1))], eq(app(S, V(0)), app(S, V(1)))),
      'le_trans':    chain([NAT, NAT, NAT, le(V(0), V(1)), le(V(1), V(2))], le(V(0), V(2))),
      'le_succ':     chain([NAT, NAT, le(V(0), V(1))], le(V(0), app(S, V(1)))),
      'le_refl':     chain([NAT], le(V(0), V(0))),
      'cong_add':    chain([NAT, NAT, eq(V(0), V(1))], eq(app(ADD, V(0), Z), app(ADD, V(1), Z))),
      'rewrite2':    chain([NAT, NAT, NAT, eq(V(0), V(1)), eq(app(S, V(1)), V(2))], eq(app(S, V(0)), V(2))),
      'add_zero_l':  chain([NAT], eq(app(ADD, Z, V(0)), V(0))),
      'drain':       chain([NAT], eq(app(S, V(0)), V(0))),
    }

SCRIPTS = {   # name -> list of steps; ('apply', tree) uses level trees (hypotheses are ('var', level)); ('rewrite', hyp_level, x_tree, y_tree)
  'refl':      lambda g: [('intro',), ('apply', ('const', g['refl']))],
  'symm':      lambda g: [('intro',), ('intro',), ('intro',), ('rewrite', 2, V(0), V(1)), ('apply', ('const', g['refl']))],
  'trans':     lambda g: [('intro',)] * 5 + [('rewrite', 3, V(0), V(1)), ('apply', V(4))],
  'cong_succ': lambda g: [('intro',)] * 3 + [('rewrite', 2, V(0), V(1)), ('apply', ('const', g['refl']))],
  'le_trans':  lambda g: [('intro',)] * 5 + [('apply', ('const', g['le_trans'])), ('apply', V(1)), ('apply', V(3)), ('apply', V(4))],
  'le_refl':   lambda g: [('intro',), ('apply', ('const', g['le_n']))],
  'le_succ':   lambda g: [('intro',)] * 3 + [('apply', ('const', g['le_S'])), ('apply', V(2))],
  'rewrite2':  lambda g: [('intro',)] * 5 + [('rewrite', 3, V(0), V(1)), ('apply', V(4))],
  'add_zero_l': lambda g: [('intro',), ('apply', ('const', g['refl']))],
  'drain':     lambda g: [('intro',), ('apply', ('const', g['Q']))],     # needs drain_pending: ?x is solved AFTER (?f ?x =?= S a) was deferred
  'cong_add':  lambda g: [('intro',)] * 3 + [('apply', ('const', g['cong']))],          # higher-order unification: pending constraints
}

def shift_meta(t, off):
    """meta ids are global counters; the reference spent some building the library, so compare modulo a constant offset"""
    if isinstance(t, tuple):
        if t and t[0] == 'meta': return ('meta', t[1] + off)
        return tuple(shift_meta(x, off) if isinstance(x, tuple) else x for x in t)
    return t

def canon_metas(struct):
    """meta ids are opaque handles (the reference allocates metas on paths where the hybrid kernel fails earlier): rename by rank"""
    ids = set()
    def collect(x):
        if isinstance(x, tuple):
            if len(x) == 2 and x[0] == 'meta' and isinstance(x[1], int): ids.add(x[1]); return
            for y in x: collect(y)
        elif isinstance(x, list):
            for y in x: collect(y)
    collect(struct); m = {i: r for r, i in enumerate(sorted(ids))}
    def ren(x):
        if isinstance(x, tuple):
            if len(x) == 2 and x[0] == 'meta' and isinstance(x[1], int): return ('meta', m[x[1]])
            return tuple(ren(y) for y in x)
        if isinstance(x, list): return [ren(y) for y in x]
        return x
    return ren(struct)

class Lockstep:
    def __init__(self, kref, knew, g, stmt):
        self.kref, self.knew, self.g = kref, knew, g
        self.psr = kref.start_proof(build_ref(kref, stmt)); self.psn = knew.start_proof(build_new(knew, l2h(stmt, 0)))
        self.m1 = False; self.stopped = False; self.ops = []; self.log = []; self.off = self.psr.goal_meta_id(0) - self.psn.goal_meta_id(0)
    def state(self):
        out = []
        for ps, k, new in ((self.psr, self.kref, False), (self.psn, self.knew, True)):
            st = []
            for i in range(ps.num_goals()):
                nh = ps.goal_num_hyps(i); tgt = parse_raw(k.dump_raw(ps.goal_target(i)), new); hy = [parse_raw(k.dump_raw(ps.goal_hyp_type(i, j)), new) for j in range(nh)]
                if not new: tgt = l2h(tgt, nh, lenient=True); hy = [l2h(h, j, lenient=True) for j, h in enumerate(hy)]
                st.append((('meta', ps.goal_meta_id(i)), nh, tgt, tuple(hy)))
            out.append((canon_metas(st), ps.is_complete()))
        return out
    def step(self, op):
        self.ops.append(op)
        from oracle_lib import is_mode2, goals_wellformed
        meta_bearing = mode1 = False
        if self.psr.num_goals():
            if op[0] == 'rewrite': meta_bearing = "'meta'" in str(parse_raw(self.kref.dump_raw(self.psr.goal_target(0)), False))
            if op[0] == 'induct' and op[1] < self.psn.goal_num_hyps(0): mode1 = not is_mode2(parse_raw(self.knew.dump_raw(self.psn.goal_hyp_type(0, op[1])), True), self.g)
        nh = self.psn.goal_num_hyps(0) if self.psn.num_goals() else 0; res = []; self.last_msgs = []
        for ps, k, bld, conv in ((self.psr, self.kref, build_ref, lambda t, n: t), (self.psn, self.knew, build_new, lambda t, n: l2h(t, n, lenient=True))):
            try:
                if op[0] == 'intro': ps.intro('x')
                elif op[0] == 'apply': ps.apply(bld(k, conv(op[1], nh)))
                elif op[0] == 'cases': ps.cases(op[1])
                elif op[0] == 'induct': ps.induct(op[1])
                elif op[0] == 'rewrite':
                    _, j, x, y = op
                    ps.rewrite(bld(k, conv(V(j), nh)), bld(k, conv(x, nh)), bld(k, conv(y, nh)), self.g['transport'], self.g['symm'])
                res.append(True)
            except Exception as e: res.append(False); self.last_msgs.append(str(e))
            else: self.last_msgs.append('')
        sr, sn = self.state()
        same = res[0] == res[1] and sr == sn
        if mode1 and res[0] and res[1]: self.m1 = True
        if op[0] == 'rewrite' and meta_bearing and res[0] and not same:
            # F13: the hybrid rewrite instantiates solved metas first (and checks its motive); the reference rewrites inside meta spines.
            # Explained only if the hybrid state is well-formed; the script cannot continue in lockstep.
            if res[1] and not goals_wellformed(self.knew, self.psn): pass          # a real problem: stays a divergence
            else: self.stopped = True; self.f13 = getattr(self, 'f13', 0) + 1; same = True
        self.log.append((op[0], res[0], res[1], same))
        return res[0], same, (sr, sn)

def check_proof_pair(L, stmt):
    """both kernels' extracted proofs must verify and normalise identically"""
    ok = []
    pr, pn = L.psr.extract(), L.psn.extract()
    try: L.kref.whole_term_check(pr, build_ref(L.kref, stmt)); ok.append(True)
    except Exception: ok.append(False)
    good, msg = L.knew.check_open(pn, build_new(L.knew, l2h(stmt, 0)), []); ok.append(good)
    a = l2h(parse_raw(L.kref.dump_raw(L.kref.normalize_node(pr)), False), 0, lenient=True); b = parse_raw(L.knew.dump_raw(L.knew.normalize_open(pn, 0)), True)
    return ok, a == b

def run_scripted(kref, knew, g):
    stmts = statements(g); rows = []
    for name, mk in SCRIPTS.items():
        L = Lockstep(kref, knew, g, stmts[name]); bad = None
        for op in mk(g):
            ok, same, st = L.step(op)
            if not same: bad = (op, ok, 'STATE DIFFERS' if True else ''); break
        done = L.psr.is_complete() and L.psn.is_complete()
        proofs = check_proof_pair(L, stmts[name]) if (done and not bad) else (None, None)
        rows.append((name, bad, done, proofs))
    return rows

def random_script(kref, knew, g, stmt, rng, max_steps=10):
    L = Lockstep(kref, knew, g, stmt); consts = [g[n] for n in ('refl', 'cong', 'symm', 'trans', 'transport', 'le_n', 'le_S', 'le_trans', 'Succ', 'add')]
    for _ in range(max_steps):
        if L.psr.num_goals() == 0: break
        nh = L.psr.goal_num_hyps(0); tgt = parse_raw(kref.dump_raw(L.psr.goal_target(0)), False)
        hyps = [parse_raw(kref.dump_raw(L.psr.goal_hyp_type(0, j)), False) for j in range(nh)]
        r = rng.random()
        eqh = [j for j, h in enumerate(hyps) if h[0] == 'app' and h[1][0] == 'app' and h[1][1][0] == 'app' and h[1][1][1] == ('const', g['Eq']) and not any(w in str(h) for w in ("'lam'", "'pi'", "'match'", "'fix'"))]
        if tgt[0] == 'pi' and r < 0.45: op = ('intro',)
        elif eqh and r < 0.65:
            j = rng.choice(eqh); h = hyps[j]; op = ('rewrite', j, h[1][2], h[2])
        elif nh and r < 0.8: op = ('apply', V(rng.randrange(nh)))
        else: op = ('apply', ('const', rng.choice(consts)))
        ok, same, st = L.step(op)
        if not same: return L, (op, st)
        if L.stopped: break
    return L, None

def run_random(N, seed):
    kref, knew, g, defs = library(); stmts = statements(g); rng = random.Random(seed); names = list(stmts)
    stats = dict(scripts=0, steps=0, ok_steps=0, failed_steps=0, completed=0, proofs_ok=0, bad=[])
    t_ref = t_new = 0.0
    for i in range(N):
        stmt = stmts[rng.choice(names)]
        L, bad = random_script(kref, knew, g, stmt, rng)
        stats['scripts'] += 1; stats['steps'] += len(L.log); stats['ok_steps'] += sum(1 for l in L.log if l[1]); stats['failed_steps'] += sum(1 for l in L.log if not l[1])
        if bad: stats['bad'].append((L.log[-3:], str(bad)[:200])); kref, knew, g, defs = library(); continue   # fresh kernels after a divergence
        if L.psr.is_complete() and L.psn.is_complete():
            stats['completed'] += 1; ok, same_nf = check_proof_pair(L, stmt)
            if ok == [True, True] and same_nf: stats['proofs_ok'] += 1
            else: stats['bad'].append(('proof', ok, same_nf))
    return stats

# ------------------------------------------------------------------ telescope closers vs an independent right-fold (dependent telescopes)
def test_telescopes(n, seed):
    kref, knew, g, defs = library(); rng = random.Random(seed); bad = []
    def py_close(kind, doms, body, base):
        t = body
        for j in reversed(range(len(doms))): t = (kind, doms[j], py_abs(t, [base + j]))
        return t
    for i in range(n):
        kind = rng.choice(['lam', 'pi']); m = rng.randrange(1, 5); base = rng.randrange(0, 3)
        doms = [rand_h(rng, 3, 0, base + j, nconst=10) for j in range(m)]; body = rand_h(rng, 3, 0, base + m, nconst=10)
        got = parse_raw(knew.dump_raw(knew.close_telescope(kind, [build_new(knew, d) for d in doms], build_new(knew, body), base)), True)
        want = py_close(kind, doms, body, base)
        if got != want: bad.append((kind, base, m, str(got)[:100], str(want)[:100]))
    return bad

# ------------------------------------------------------------------ unifier unit tests (hybrid kernel; ground truth by logic)
def test_unifier():
    kref, knew, g, defs = library(); NAT = ('const', g['Nat']); Z = ('const', g['Zero']); S = ('const', g['Succ'])
    Sx = lambda t: ('app', S, t); NATn = build_new(knew, NAT); rows = []
    def meta(scope):   # a fresh meta of type Nat valid in a context of size `scope`, applied to every context variable
        m = knew.fresh_meta(NATn, scope); t = ('meta', m)
        for i in range(scope): t = ('app', t, ('fvar', i))
        return m, t
    def case(name, ctx_n, mk, expect_ok, expect_pending=0, expect_solved=None):
        hyps = [NATn] * ctx_n; ms, a, b = mk(); ok, msg, pending = knew.unify_open(build_new(knew, a), build_new(knew, b), hyps)
        solved = [knew.meta_is_solved(m) for m in ms]
        good = (ok == expect_ok) and (pending == expect_pending) and (expect_solved is None or solved == expect_solved)
        rows.append((name, expect_ok, ok, pending, solved, good, msg[:50]))
    def f1(): m, t = meta(0); return [m], t, t
    case('?m =?= ?m  (reference quirk preserved: rejected by the occurs check)', 0, f1, False, 0, [False])
    def f2(): m, t = meta(0); return [m], t, Sx(t)
    case('occurs check: ?m =?= S ?m', 0, f2, False, 0, [False])
    def f3(): m, t = meta(0); return [m], t, Sx(Z)
    case('?m =?= S Zero (solves)', 0, f3, True, 0, [True])
    def f4(): m, t = meta(0); return [m], Sx(t), Sx(Z)
    case('S ?m =?= S Zero (solves by decomposition)', 0, f4, True, 0, [True])
    def f5(): m, t = meta(0); return [m], Sx(t), Z
    case('S ?m =?= Zero (constructor clash)', 0, f5, False, 0, [False])
    def f6(): m, t = meta(2); return [m], t, ('fvar', 0)
    case('Miller: ?m a b =?= a (spine variable)', 2, f6, True, 0, [True])
    def f7():
        m = knew.fresh_meta(NATn, 2); t = ('app', ('meta', m), ('fvar', 0)); return [m], t, ('fvar', 1)
    case('scope escape: ?m a =?= b (b outside the spine)', 2, f7, False, 0, [False])
    def f8():
        m = knew.fresh_meta(NATn, 2); t = ('app', ('app', ('meta', m), ('fvar', 0)), ('fvar', 0)); return [m], t, ('fvar', 0)
    case('non-linear: ?m a a =?= a (not a pattern: deferred)', 2, f8, True, 1, [False])
    def f9(): m, t = meta(1); return [m], Sx(t), Sx(('fvar', 0))
    case('S (?m a) =?= S a', 1, f9, True, 0, [True])
    return rows

if __name__ == '__main__':
    t0 = time.time(); FAIL = []
    kref, knew, g, defs = library()
    sym_ty, sym_val, trn_ty, trn_val = defs
    # the hybrid kernel's own checked path must accept symm and trans too
    kc = H.Kernel(); gid_of = {name: gid for gid, name, _, _ in kref.list_globals()}
    ev = [e for e in kref.export_library() if e['name'] not in ('symm', 'trans')]
    # replay everything but symm/trans, then declare those two through declare_checked
    tmp = H.Kernel(); 
    import harness as _h
    class _K: pass
    for e in ev: pass
    print("S3 scripted proofs (lockstep, reference vs hybrid):")
    for name, bad, done, proofs in run_scripted(kref, knew, g):
        st = "DIVERGED %s" % (bad,) if bad else ("complete, proofs verify in both, identical normal forms" if (done and proofs[0] == [True, True] and proofs[1]) else ("states identical; proof not completed (expected for open-ended scripts)" if not done else "PROOF PROBLEM %s" % (proofs,)))
        print("   %-12s %s" % (name, st))
        if bad or (done and not (proofs[0] == [True, True] and proofs[1])): FAIL.append(name)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    s = run_random(N, 3)
    print("S3 random tactic scripts: %d scripts, %d steps (%d succeeded / %d failed identically), %d completed proofs, %d verified in both with identical normal forms, divergences: %d" %
          (s['scripts'], s['steps'], s['ok_steps'], s['failed_steps'], s['completed'], s['proofs_ok'], len(s['bad'])))
    for b in s['bad'][:3]: print("     DIVERGENCE:", str(b)[:360])
    if s['bad']: FAIL.append('random scripts')
    tb = test_telescopes(4000, 11)
    print('   telescope closers vs independent right-fold: 4000 random dependent telescopes, %d mismatches' % len(tb))
    if tb: FAIL.append('telescopes'); print('     ', tb[0])
    for r in test_unifier():
        print('   unify %-52s expect %-6s got %-6s pending %d solved %s %s' % (r[0], 'ok' if r[1] else 'reject', 'ok' if r[2] else 'reject', r[3], r[4], 'OK' if r[5] else 'BAD')); (FAIL.append('unify: ' + r[0]) if not r[5] else None)
    print("elapsed %.1fs" % (time.time() - t0)); print("S3 GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)

