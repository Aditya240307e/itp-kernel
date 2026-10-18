"""S6 gate: the logical vocabulary (F7) with a shape probe per connective, and the F5 shape (elimination over a lambda predicate).
Every family is declared in BOTH kernels through each one's own checked admission path. Proofs are tactic scripts run in lockstep; wrong proofs must be rejected;
valid proofs are MUTATED and the two kernels' accept/reject verdicts compared."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, random, time, functools
import harness as hn
from harness import *
import s2_tests
import s3_tests as s3
from s3_tests import V, app, Lockstep, library, check_proof_pair

PARAMS = False
T0 = ('type', 0)
chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)

def declare_family(kref, knew, name, sort, ctors):
    tids = []
    for k, bld, conv, CS in ((kref, build_ref, lambda t, n: t, R.CtorSpec), (knew, build_new, lambda t, n: l2h(t, n, lenient=True), H.CtorSpec)):
        tid = k.reserve_inductive_type(name, bld(k, conv(sort, 0))); specs = []
        for cn, ats, ris in ctors:
            ats2 = [s2_tests.subst_self(a, tid) for a in ats]; ris2 = [s2_tests.subst_self(r, tid) for r in ris]
            specs.append(CS(cn, [bld(k, conv(a, j)) for j, a in enumerate(ats2)], [bld(k, conv(r, len(ats2))) for r in ris2]))
        k.finish_inductive(tid, specs); tids.append(tid)
    assert tids[0] == tids[1], (name, tids)
    return tids[0]

def declare_def(kref, knew, name, ty, val):
    for k, bld, conv in ((kref, build_ref, lambda t, n: t), (knew, build_new, lambda t, n: l2h(t, n, lenient=True))):
        k.declare_checked(bld(k, conv(ty, 0)), bld(k, conv(val, 0)), name)

def build_logic():
    kref, knew, g, defs = library()
    declare_family(kref, knew, 'True', T0, [('tt', [], [])])
    declare_family(kref, knew, 'False', T0, [])
    declare_family(kref, knew, 'And', ('pi', T0, ('pi', T0, T0)), [('conj', [T0, T0, ('var', 0), ('var', 1)], [('var', 0), ('var', 1)])])
    declare_family(kref, knew, 'Or', ('pi', T0, ('pi', T0, T0)), [('inl', [T0, T0, ('var', 0)], [('var', 0), ('var', 1)]), ('inr', [T0, T0, ('var', 1)], [('var', 0), ('var', 1)])])
    declare_family(kref, knew, 'Ex', ('pi', T0, ('pi', ('pi', ('var', 0), T0), T0)),
                   [('ex_intro', [T0, ('pi', ('var', 0), T0), ('var', 0), ('app', ('var', 1), ('var', 2))], [('var', 0), ('var', 1)])])
    gg = {name: gid for gid, name, _, _ in kref.list_globals()}
    declare_def(kref, knew, 'Not', ('pi', T0, T0), ('lam', T0, ('pi', ('var', 0), ('const', gg['False']))))
    declare_def(kref, knew, 'Iff', ('pi', T0, ('pi', T0, T0)),
                ('lam', T0, ('lam', T0, app(('const', gg['And']), ('pi', ('var', 0), ('var', 1)), ('pi', ('var', 1), ('var', 0))))))
    g = {name: gid for gid, name, _, _ in kref.list_globals()}
    assert g == {name: gid for gid, name, _, _ in knew.list_globals()}, "the two kernels disagree about the global table"
    return kref, knew, g

def kit(g, params=False):
    C = lambda n: ('const', g[n]); NAT, EQ, Z = C('Nat'), C('Eq'), C('Zero')
    AND, OR, EX, NOT, IFF, FALSE, TRUE = C('And'), C('Or'), C('Ex'), C('Not'), C('Iff'), C('False'), C('True')
    stmts = {
      'true_intro':    TRUE,
      'false_elim':    chain([T0, FALSE], V(0)),
      'and_comm':      chain([T0, T0, app(AND, V(0), V(1))], app(AND, V(1), V(0))),
      'or_comm':       chain([T0, T0, app(OR, V(0), V(1))], app(OR, V(1), V(0))),
      'not_not_intro': chain([T0, V(0)], app(NOT, app(NOT, V(0)))),
      'iff_refl':      chain([T0], app(IFF, V(0), V(0))),
      'ex_elim':       chain([T0, ('pi', V(0), T0), T0, app(EX, V(0), V(1)), ('pi', V(0), ('pi', app(V(1), V(4)), V(2)))], V(2)),
    }
    scripts = {
      'true_intro':    [('apply', C('tt'))],
      'false_elim':    [('intro',), ('intro',), ('cases', 1)],
      'and_comm':      [('intro',)] * 3 + [('cases', 2), ('apply', C('conj')), ('apply', V(6)), ('apply', V(5))],
      'or_comm':       [('intro',)] * 3 + [('cases', 2), ('apply', C('inr')), ('apply', V(5)), ('apply', C('inl')), ('apply', V(5))],
      'not_not_intro': [('intro',)] * 3 + [('apply', V(2)), ('apply', V(1))],
      'iff_refl':      [('intro',), ('apply', C('conj')), ('intro',), ('apply', V(1)), ('intro',), ('apply', V(1))],
      # legacy behaviour: the indices of h are the VARIABLES A and P, so cases reverts P, Q, f and the goal becomes Q'; the reverted copy of f is hypothesis 11
      'ex_elim':       [('intro',)] * 5 + [('cases', 3), ('apply', app(V(11), V(7), V(8)))],
    }
    if params:
        # with parameters only the hypotheses AFTER the scrutinee are reverted: [A, P, Q, h, f, A', P', x, p, f'], and the usable copy is f' (hypothesis 9)
        scripts['ex_elim'] = [('intro',)] * 5 + [('cases', 3), ('apply', app(V(9), V(7), V(8)))]
        scripts['ex_lambda'] = [('intro',), ('cases', 0), ('apply', C('ex_intro')), ('apply', V(3)), ('apply', C('symm')), ('apply', V(4))]
        stmts['ex_lambda'] = ('pi', app(EX, NAT, ('lam', NAT, app(EQ, NAT, V(0), Z))), app(EX, NAT, ('lam', NAT, app(EQ, NAT, Z, V(1)))))
    wrong = {   # (type, term): must be REJECTED
      'and_comm: identity':         (stmts['and_comm'], ('lam', T0, ('lam', T0, ('lam', app(AND, V(0), V(1)), V(2))))),
      'false_elim: returns the proof of False as P': (stmts['false_elim'], ('lam', T0, ('lam', FALSE, V(1)))),
      'not_not_intro: returns p for False': (stmts['not_not_intro'], ('lam', T0, ('lam', V(0), ('lam', app(NOT, V(0)), V(1))))),
      'Ex from a True': (chain([T0, ('pi', V(0), T0)], app(EX, V(0), V(1))), ('lam', T0, ('lam', ('pi', V(0), T0), C('tt')))),
      'Or: inr given an A':         (chain([T0, T0, V(0)], app(OR, V(0), V(1))), ('lam', T0, ('lam', T0, ('lam', V(0), app(C('inr'), V(0), V(1), V(2)))))),
      'And: both components a':     (chain([T0, T0, V(0)], app(AND, V(0), V(1))), ('lam', T0, ('lam', T0, ('lam', V(0), app(C('conj'), V(0), V(1), V(2), V(2)))))),
    }
    return stmts, scripts, wrong

def run_scripts(kref, knew, g):
    stmts, scripts, wrong = kit(g, PARAMS); rows = []
    for name, ops in scripts.items():
        L = Lockstep(kref, knew, g, stmts[name]); first_div = None
        for i, op in enumerate(ops):
            ok, same, st = L.step(op)
            if first_div is None and not same: first_div = (i, op[0], L.log[-1][1], L.log[-1][2], L.last_msgs[0][:50], L.last_msgs[1][:50])
        done_n = L.psn.is_complete(); done_r = L.psr.is_complete()
        ver_n = knew.check_open(L.psn.extract(), build_new(knew, l2h(stmts[name], 0)), [])[0] if done_n else False
        ver_r = None
        if done_r:
            try: kref.whole_term_check(L.psr.extract(), build_ref(kref, stmts[name])); ver_r = True
            except Exception: ver_r = False
        rows.append((name, done_n, ver_n, done_r, ver_r, first_div, L))
    return rows

def run_wrong(kref, knew, g):
    stmts, scripts, wrong = kit(g); rows = []
    for name, (ty, term) in wrong.items():
        ok_n = knew.check_open(build_new(knew, l2h(term, 0)), build_new(knew, l2h(ty, 0)), [])[0]
        try: kref.whole_term_check(build_ref(kref, term), build_ref(kref, ty)); ok_r = True
        except Exception: ok_r = False
        rows.append((name, ok_n, ok_r))
    return rows

def subterm_paths(t, path=()):
    out = [path]
    if isinstance(t, tuple) and t and isinstance(t[0], str):
        for i, c in enumerate(t[1:], 1):
            if isinstance(c, tuple): out += subterm_paths(c, path + (i,))
            elif isinstance(c, list): pass
    return out
def get(t, path):
    for i in path: t = t[i]
    return t
def put(t, path, new):
    if not path: return new
    l = list(t); l[path[0]] = put(t[path[0]], path[1:], new); return tuple(l)

def mutate(tree, rng, consts):
    paths = subterm_paths(tree); path = rng.choice(paths); node = get(tree, path)
    if not (isinstance(node, tuple) and node and isinstance(node[0], str)): return None
    k = node[0]
    if k == 'var': new = ('var', max(0, node[1] + rng.choice([-2, -1, 1, 2])))
    elif k == 'const': new = ('const', rng.choice(consts))
    elif k == 'app': new = ('app', node[2], node[1]) if rng.random() < 0.5 else ('app', node[1], rng.choice([('var', 0), ('var', 1), ('const', rng.choice(consts))]))
    elif k in ('lam', 'pi'): new = (k, rng.choice([('const', consts[0]), T0, ('var', 0)]), node[2])
    else: return None
    return put(tree, path, new) if new != node else None

def f19_minimal(kref, knew, g):
    C = lambda n: ('const', g[n]); FALSE, OR = C('False'), C('Or')
    ty = chain([T0, FALSE], V(0))
    mk = lambda dom: ('lam', T0, ('lam', FALSE, ('match', V(1), ('lam', dom, V(0)), [])))     # lam P h. match h as (lam (_ : dom). P) {}
    res = []
    for dom in (OR, FALSE):
        t_r, t_n = build_ref(kref, mk(dom)), build_new(knew, l2h(mk(dom), 0))
        try: kref.whole_term_check(t_r, build_ref(kref, ty)); vr = True
        except Exception: vr = False
        vn = knew.check_open(t_n, build_new(knew, l2h(ty, 0)), [])[0]
        res.append((vr, vn))
    return res[0][0], res[0][1], res[1][0], res[1][1]

def scoped(t, d):
    """are all variable levels of this level-tree bound (level < depth)?"""
    if not (isinstance(t, tuple) and t and isinstance(t[0], str)): return True
    k = t[0]
    if k == 'var': return t[1] < d
    if k in ('lam', 'pi'): return scoped(t[1], d) and scoped(t[2], d + 1)
    if k == 'fix': return scoped(t[2], d) and scoped(t[3], d + 1)
    return all(scoped(c, d) for c in t[1:] if isinstance(c, tuple))

def mutant_differential(kref, knew, g, rows, per_proof=80, seed=1):
    rng = random.Random(seed); stmts, _, _ = kit(g); consts = [v for k, v in g.items()]
    agree = acc_both = rej_both = 0; dis = []; f19 = []
    for (name, done_n, ver_n, done_r, ver_r, first_div, L) in rows:
        if not (done_n and ver_n and done_r and ver_r and first_div is None): continue
        proof = parse_raw(kref.dump_raw(L.psr.extract()), False)
        for _ in range(per_proof):
            mt = mutate(proof, rng, consts)
            if mt is None or not scoped(mt, 0): continue          # the reference ABORTS on unbound variables (F18): only well-scoped mutants go to it
            try: kref.whole_term_check(build_ref(kref, mt), build_ref(kref, stmts[name])); vr = True
            except Exception: vr = False
            try: vn, vmsg = knew.check_open(build_new(knew, l2h(mt, 0, lenient=True)), build_new(knew, l2h(stmts[name], 0)), [])
            except Exception as e: vn, vmsg = False, str(e)
            if vr == vn: agree += 1; acc_both += vr; rej_both += (not vr)
            elif vr and not vn and vmsg == 'explicit motive_id is not well-typed': f19.append((name, vmsg))      # F19: the reference does not validate a motive's binder domains
            else: dis.append((name, vr, vn, vmsg))
    return agree, acc_both, rej_both, dis, f19

def ill_scoped_robustness(n=400, seed=5):
    """the HYBRID kernel must reject a term with unbound variables, never abort (run in a subprocess by the gate)"""
    kref, knew, g = build_logic(); stmts, scripts, wrong = kit(g); rng = random.Random(seed); consts = list(g.values()); rows = run_scripts(kref, knew, g)
    tried = rejected = accepted = 0
    for (name, done_n, ver_n, done_r, ver_r, fd, L) in rows:
        if not (done_n and ver_n): continue
        proof = parse_raw(kref.dump_raw(L.psr.extract()), False)
        for _ in range(n):
            mt = mutate(proof, rng, consts)
            if mt is None or scoped(mt, 0): continue                      # only ILL-scoped mutants
            tried += 1
            try: ok = knew.check_open(build_new(knew, l2h(mt, 0, lenient=True)), build_new(knew, l2h(stmts[name], 0)), [])[0]
            except Exception: ok = False
            accepted += ok; rejected += (not ok)
    # loose BOUND variables cannot arise from level mutation: build them directly
    NAT = ('const', g['Nat']); target = build_new(knew, T0)
    for tree in (('bvar', 0), ('lam', T0, ('bvar', 1)), ('app', ('bvar', 0), NAT), ('pi', ('bvar', 0), T0), ('lam', ('bvar', 2), ('bvar', 0))):
        tried += 1
        try: ok = knew.check_open(build_new(knew, tree), target, [])[0]
        except Exception: ok = False
        accepted += ok; rejected += (not ok)
    print("ill-scoped terms offered to the hybrid kernel: %d, rejected %d, ACCEPTED %d" % (tried, rejected, accepted)); sys.exit(1 if accepted else 0)

if __name__ == '__main__' and '--ill-scoped' in sys.argv:
    ill_scoped_robustness()

if __name__ == '__main__':
    t0 = time.time(); FAIL = []
    kref, knew, g = build_logic()
    print("logic kit declared in both kernels (each through its own checked admission): %s" % [n for n in ('True', 'False', 'And', 'Or', 'Ex', 'Not', 'Iff') if n in g])
    rows = run_scripts(kref, knew, g)
    for (name, done_n, ver_n, done_r, ver_r, fd, L) in rows:
        print("   %-14s hybrid: complete %s verifies %s | reference: complete %s verifies %s | first lockstep divergence: %s" % (name, done_n, ver_n, done_r, ver_r, fd))
        if not (done_n and ver_n): FAIL.append('hybrid cannot prove ' + name)
    for name, ok_n, ok_r in run_wrong(kref, knew, g):
        print("   wrong proof %-44s accepted by hybrid: %s, by reference: %s" % (name, ok_n, ok_r))
        if ok_n: FAIL.append('hybrid ACCEPTED a wrong proof: ' + name)
        if ok_r: FAIL.append('reference accepted a wrong proof: ' + name)
    agree, acc, rej, dis, f19 = mutant_differential(kref, knew, g, rows)
    print("   mutated valid proofs: %d verdicts compared, agree %d (both accept %d, both reject %d), F19 (reference accepts an ill-formed motive, hybrid rejects) %d, UNEXPLAINED disagreements %d %s" % (agree + len(dis) + len(f19), agree, acc, rej, len(f19), len(dis), dis[:3]))
    if dis: FAIL.append('unexplained accept/reject disagreement between the kernels')
    f19t = f19_minimal(kref, knew, g)
    print("   F19 minimal case (Match on False with motive lam (_ : Or). P): reference accepts %s, hybrid accepts %s | well-formed motive: reference %s, hybrid %s" % f19t)
    if f19t[1] or not (f19t[2] and f19t[3]): FAIL.append('F19 minimal case')
    import subprocess
    r = subprocess.run([sys.executable, __file__, '--ill-scoped'], capture_output=True, text=True, timeout=200)
    print("   robustness (subprocess): " + (r.stdout.strip().split('\n')[-1] if r.stdout.strip() else "no output") + (" | EXIT %d: %s" % (r.returncode, (r.stdout + r.stderr)[-160:].replace('\n', ' ')) if r.returncode else ""))
    if r.returncode: FAIL.append('the hybrid kernel crashed or accepted an ill-scoped term')
    print("elapsed %.1fs" % (time.time() - t0)); print("S6 LOGIC GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
