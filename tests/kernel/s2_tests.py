"""S2 gate (G2): the full typing core of the hybrid kernel against the reference kernel."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, random, time
import harness as hn
from harness import *

def build_library():
    kref, refs = cur.build_base_kernel()
    for nm in ('setup_nat_order', 'setup_transport', 'setup_le_trans'):
        r = getattr(cur, nm)(kref, refs)
        if isinstance(r, dict): refs.update(r)
    return kref, refs

def subst_self(t, tid):
    if isinstance(t, tuple):
        if t == ('self',): return ('const', tid)
        return tuple(subst_self(x, tid) if isinstance(x, (tuple, list)) else x for x in t)
    return t

# ------------------------------------------------------------------ A. the whole reference library, re-declared through the CHECKED path
def test_library_checked():
    kref, refs = build_library(); knew = H.Kernel()
    gid_of = {name: gid for gid, name, _, _ in kref.list_globals()}; out = []
    for e in kref.export_library():
        kind = e['kind']; Hh = lambda d, n0: build_new(knew, l2h(dict_to_tree(d, gid_of), n0))
        try:
            if kind == 'family':
                tid = knew.reserve_inductive_type(e['name'], Hh(e['sort'], 0)); specs = []
                for c in e['ctors']:
                    specs.append(H.CtorSpec(c['name'], [Hh(a, j) for j, a in enumerate(c['arg_types'])], [Hh(r, len(c['arg_types'])) for r in c['result_indices']]))
                knew.finish_inductive(tid, specs)
            elif kind == 'recursive_function':
                gid = knew.reserve_recursive_function(e['name'], Hh(e['type'], 0))
                knew.finish_recursive_function_checked(gid, gid_of[e['ind_type_name']], e['decreasing_arg_level'], Hh(e['body'], 0))
            elif kind == 'definition':
                knew.declare_checked(Hh(e['type'], 0), Hh(e['value'], 0), e['name'])
            out.append((kind, e['name'], 'OK'))
        except Exception as ex: out.append((kind, e['name'], 'FAILED: ' + str(ex)[:160])); break
    bad = 0
    for (g1, n1, t1, _), (g2, n2, t2, _) in zip(kref.list_globals(), knew.list_globals()):
        if (g1, n1) != (g2, n2) or l2h(parse_raw(kref.dump_raw(t1), False), 0) != parse_raw(knew.dump_raw(t2), True): bad += 1
    return out, bad, len(kref.list_globals()), knew.num_globals()

# ------------------------------------------------------------------ B. inductive admission corpus (both kernels must agree)
def admission_cases(g):
    nat, zero, succ, T0 = ('const', g['Nat']), ('const', g['Zero']), ('const', g['Succ']), ('type', 0)
    S = ('self',); neg = lambda x: ('pi', ('app', S, x), nat)         # neg x  :=  (self x -> Nat): a proposition (the codomain is in Type0), and the family to the left of an arrow
    return [
      ('U_paradox: mk : Type0 -> U', T0, [('mk', [T0], [])], False, "universe"),
      ('field Nat -> Type0 in a Type0 inductive', T0, [('mk', [('pi', nat, T0)], [])], False, "universe"),
      ('field whose type is a value', T0, [('mk', [zero], [])], False, "not a type"),
      ('unbound variable in a field type', T0, [('mk', [('var', 5)], [])], False, "ill-"),
      ('constructor omits a required index', ('pi', nat, T0), [('mk', [], [])], False, "fully apply"),
      ('too many indices', ('pi', nat, T0), [('mk', [], [zero, zero])], False, "ill-typed"),
      ('index of the wrong type', ('pi', nat, T0), [('mk', [], [T0])], False, "ill-typed"),
      ('declared type is a value', zero, [('mk', [], [])], False, "declared type"),
      ('strict positivity violated', T0, [('mk', [('pi', S, nat)], [])], False, "positivity"),
      ('small impredicative field', T0, [('mk', [('pi', T0, ('var', 0))], [])], True, ""),
      ('large parameter-like field', ('pi', T0, T0), [('mk', [T0], [('var', 0)])], True, ""),
      ('Vec-style indexed family', ('pi', T0, ('pi', nat, T0)),
         [('vnil', [T0], [('var', 0), zero]), ('vcons', [T0, nat, ('var', 0), ('app', ('app', S, ('var', 0)), ('var', 1))], [('var', 0), ('app', succ, ('var', 1))])], True, ""),
      # F92: the family in an INDEX. The frozen reference kernel looks at the head of a recursive field only (a sixth element says what it does); the hybrid kernel refuses all three.
      ('F92: the family in an index of a recursive field', ('pi', T0, ('type', 1)), [('mkq', [T0, ('app', S, neg(('var', 0)))], [('var', 0)])], False, "positivity", True),
      ('F92: ... under a product', ('pi', T0, ('type', 1)), [('mkq', [T0, ('pi', nat, ('app', S, neg(('var', 0))))], [('var', 0)])], False, "positivity", True),
      ('F92: the family in a result index', ('pi', T0, ('type', 1)), [('mkr', [T0], [neg(('var', 0))])], False, "result index", True),
      ('infinitely branching with an index (control)', ('pi', T0, T0), [('sup', [T0, ('pi', nat, ('app', S, ('var', 0)))], [('var', 0)])], True, ""),
    ]
def test_admission():
    rows = []
    kref0, _ = build_library()
    for nm_idx in range(len(admission_cases({name: gid for gid, name, _, _ in kref0.list_globals()}))):
        kref, refs = build_library(); knew = H.Kernel(); load_library(kref, knew)
        g = {name: gid for gid, name, _, _ in kref.list_globals()}
        name, sort, ctors, expect, needle, *ref_says = admission_cases(g)[nm_idx]
        ref_expect = ref_says[0] if ref_says else expect                    # what the frozen reference does: the same, except for the cases of finding F92
        res = {}
        for which, k, bld, conv, CS in (('ref', kref, build_ref, lambda t, n: t, R.CtorSpec), ('new', knew, build_new, lambda t, n: l2h(t, n, lenient=True), H.CtorSpec)):
            try:
                tid = k.reserve_inductive_type('Ty', bld(k, conv(subst_self(sort, 10**6), 0)))   # sort never mentions self
                specs = []
                for cn, ats, ris in ctors:
                    ats = [subst_self(a, tid) for a in ats]; ris = [subst_self(r, tid) for r in ris]
                    specs.append(CS(cn, [bld(k, conv(a, j)) for j, a in enumerate(ats)], [bld(k, conv(r, len(ats))) for r in ris]))
                k.finish_inductive(tid, specs); res[which] = (True, '')
            except Exception as ex: res[which] = (False, str(ex))
        agree = res['ref'][0] == ref_expect and res['new'][0] == expect
        reason_ok = expect or (needle in res['new'][1]) or (needle == 'positivity' and 'positivity' in res['new'][1])
        rows.append((name, expect, res['ref'][0], res['new'][0], agree and reason_ok, res['new'][1][:70]))
    return rows

# ------------------------------------------------------------------ C. structural-recursion guard corpus
def guard_cases(g):
    nat, zero, succ = ('const', g['Nat']), ('const', g['Zero']), ('const', g['Succ']); S = ('self',)
    mot = ('lam', nat, nat)
    lam = lambda body: ('lam', nat, body)
    mt = lambda scrut, zb, sb: ('match', scrut, mot, (zb, ('lam', nat, sb)))
    n, k = ('var', 0), ('var', 1)
    return [
      ('double: S (S (f k))', lam(mt(n, zero, ('app', succ, ('app', succ, ('app', S, k))))), True),
      ('successor-copy: S (f k)', lam(mt(n, zero, ('app', succ, ('app', S, k)))), True),
      ('loop: f n = f n', lam(('app', S, n)), False),
      ('recurse on the scrutinee: f (S k) = f n', lam(mt(n, zero, ('app', S, n))), False),
      ('growing: f n = f (S n)', lam(('app', S, ('app', succ, n))), False),
      ('nested self call as the argument: f (f k)', lam(mt(n, zero, ('app', S, ('app', S, k)))), False),
    ]
def test_guards():
    rows = []
    for idx in range(6):
        kref, refs = build_library(); knew = H.Kernel(); load_library(kref, knew)
        g = {name: gid for gid, name, _, _ in kref.list_globals()}
        name, body, expect = guard_cases(g)[idx]; fty = ('pi', ('const', g['Nat']), ('const', g['Nat'])); res = {}
        for which, k, bld, conv in (('ref', kref, build_ref, lambda t, n: t), ('new', knew, build_new, lambda t, n: l2h(t, n))):
            try:
                gid = k.reserve_recursive_function('f', bld(k, conv(fty, 0)))
                k.finish_recursive_function_checked(gid, g['Nat'], 0, bld(k, conv(subst_self(body, gid), 0))); res[which] = (True, '')
            except Exception as ex: res[which] = (False, str(ex))
        rows.append((name, expect, res['ref'][0], res['new'][0], res['ref'][0] == res['new'][0] == expect, res['new'][1][:60]))
    return rows

# ------------------------------------------------------------------ D. Match typing differential (with Match, no Fix)
def test_match_typing(N, seed):
    kref, knew, g, zf = setup(); rng = random.Random(seed); G = Gen(g, rng, zf); res = dict(ok=[0, []], reject=[0, []]); res['t'] = [0.0, 0.0]
    NAT = ('const', g['Nat']); ZERO = ('const', g['Zero']); BOOLT = ('const', g['Bool'])
    mot_n = ('lam', NAT, NAT); mot_b = ('lam', NAT, BOOLT)
    for i in range(N):
        T = G.rtype(); term = G.gen(T, [], 4)
        try:
            ref_ok = True
            tr_, ty_r = build_ref(kref, term), build_ref(kref, G.ty(T)); tn_, ty_n = build_new(knew, l2h(term, 0)), build_new(knew, l2h(G.ty(T), 0)); a_ = time.perf_counter()
            try: kref.whole_term_check(tr_, ty_r)
            except Exception: ref_ok = False
            b_ = time.perf_counter(); ok, msg = knew.check_open(tn_, ty_n, []); c_ = time.perf_counter(); res['t'][0] += b_ - a_; res['t'][1] += c_ - b_
            res['ok'][0] += 1
            if not (ref_ok and ok): res['ok'][1].append((term, T, ref_ok, ok, msg if not ok else ''))
        except Exception as e: res['ok'][1].append((term, T, 'EXC', repr(e)[:100]))
        if i % 2 == 0:   # ill-typed terms: wrong branch count, wrong branch type, non-inductive scrutinee, WRONG LAMBDA DOMAIN
            c = rng.randrange(5); sc = G.gen('nat', [], 2); exp = NAT
            if c == 0: bad = ('match', sc, mot_n, (G.gen('nat', [], 2),))
            elif c == 1: bad = ('match', sc, mot_n, (G.gen('bool', [], 2), ('lam', NAT, ZERO)))
            elif c == 2: bad = ('match', G.gen(('fn', 'nat', 'nat'), [], 2), mot_n, (ZERO, ('lam', NAT, ZERO)))
            elif c == 4: bad = ('match', sc, mot_n, (ZERO, ('lam', NAT, ZERO), ZERO))      # over-long match: more branches than constructors
            else:   # lam x:A'. body  checked against  A -> B  with A' /= A, body otherwise well-typed
                A = rng.choice(['nat', 'bool']); Ap = 'bool' if A == 'nat' else 'nat'; B = rng.choice(['nat', 'bool'])
                bad = ('lam', G.ty(Ap), G.gen(B, [Ap], 2)); exp = G.ty(('fn', A, B))
            try:
                ref_rej = False
                try: kref.whole_term_check(build_ref(kref, bad), build_ref(kref, exp))
                except Exception: ref_rej = True
                ok, msg = knew.check_open(build_new(knew, l2h(bad, 0)), build_new(knew, l2h(exp, 0)), [])
                res['reject'][0] += 1
                if not (ref_rej and not ok): res['reject'][1].append((bad, c, ref_rej, ok))
            except Exception as e: res['reject'][1].append((bad, c, 'EXC', repr(e)[:100]))
    return res

# ------------------------------------------------------------------ E. proof corpus (Fix + Match, nested) and the guard on hand-built Fix terms
def test_proof_corpus():
    kref, knew, g, zf = setup(); kk, refs = cur.build_base_kernel(); nat = refs['nat_ty']; V = kk.mk_var
    NAT = ('const', g['Nat']); EQ = ('const', g['Eq']); SUCC = ('const', g['Succ'])
    eq = lambda a, b: ('app', ('app', ('app', EQ, NAT), a), b)
    specs = [(['I:0'], 1), (['C:0'], 1), (['I:0', 'I:1'], 2), (['I:1', 'I:0'], 2), (['I:0', 'I:last'], 2), (['C:0', 'I:last'], 2),
             (['I:0', 'I:1', 'I:2'], 3), (['I:2', 'I:1', 'I:0'], 3), (['I:1', 'I:2', 'I:0'], 3), (['C:0', 'C:1', 'I:2'], 3),
             (['I:0', 'I:1', 'I:2', 'I:3'], 4), (['I:3', 'I:2', 'I:1', 'I:0'], 4)]
    out = dict(accepted=0, wrong_stmt_rejected=0, n=0, bad=[], fixes=0)
    for spec, nv in specs:
        ty = cur.eq_app(kk, refs, nat, V(0), V(0))
        for _ in range(nv): ty = kk.mk_pi(nat, ty)
        ps = kk.start_proof(ty)
        for i in range(nv): ps.intro('v%d' % i, target_meta_id=ps.goal_meta_id(0))
        cm = ps.goal_meta_id(0); gi = lambda m: next(i for i in range(ps.num_goals()) if ps.goal_meta_id(i) == m)
        def last_nat(gg): return max(j for j in range(ps.goal_num_hyps(gg)) if kk.print_node(ps.goal_hyp_type(gg, j)) == 'Nat')
        for step in spec:
            kind, which = step.split(':'); gg = gi(cm); h = last_nat(gg) if which == 'last' else int(which)
            before = {ps.goal_meta_id(i) for i in range(ps.num_goals())}
            (ps.induct if kind == 'I' else ps.cases)(h, target_meta_id=cm)
            new = [ps.goal_meta_id(i) for i in range(ps.num_goals()) if ps.goal_meta_id(i) not in before]
            for m in new[:-1]: ps.apply(refs['refl_c'], target_meta_id=m)
            cm = new[-1]
        ps.apply(refs['refl_c'], target_meta_id=cm); proof = ps.extract(); kk.whole_term_check(proof, ty)
        ptree = parse_raw(kk.dump_raw(proof), False); out['fixes'] += str(ptree).count("'fix'")
        def stmt(last_arg):  # forall v0..v_{nv-1}. v0 = last_arg
            t = eq(('var', 0), last_arg)
            for _ in range(nv): t = ('pi', NAT, t)
            return t
        p_h = build_new(knew, l2h(ptree, 0)); right = build_new(knew, l2h(stmt(('var', 0)), 0))
        wrong = build_new(knew, l2h(stmt(('app', SUCC, ('var', 0))), 0))
        ok, msg = knew.check_open(p_h, right, []); out['n'] += 1
        if ok: out['accepted'] += 1
        else: out['bad'].append((spec, 'REJECTED a valid proof', msg))
        ok2, msg2 = knew.check_open(p_h, wrong, [])
        if not ok2: out['wrong_stmt_rejected'] += 1
        else: out['bad'].append((spec, 'ACCEPTED the proof of a FALSE statement', ''))
    # hand-built Fix terms (new kernel only): structural recursion is checked by the kernel itself
    NATn = build_new(knew, ('const', g['Nat'])); fn_t = build_new(knew, ('pi', ('const', g['Nat']), ('const', g['Nat'])))
    nat_h, zero_h, succ_h = ('const', g['Nat']), ('const', g['Zero']), ('const', g['Succ'])
    mot = ('lam', nat_h, nat_h)
    fixes = {
      'valid: f n = match n | Zero | S k -> S (f k)': (True, ('fix', 0, ('pi', nat_h, nat_h), ('lam', nat_h, ('match', ('bvar', 0), mot, (zero_h, ('lam', nat_h, ('app', succ_h, ('app', ('bvar', 2), ('bvar', 0))))))))),
      'invalid: f n = f n': (False, ('fix', 0, ('pi', nat_h, nat_h), ('lam', nat_h, ('app', ('bvar', 1), ('bvar', 0))))),
      'invalid: f n = match n | Zero | S k -> f n': (False, ('fix', 0, ('pi', nat_h, nat_h), ('lam', nat_h, ('match', ('bvar', 0), mot, (zero_h, ('lam', nat_h, ('app', ('bvar', 2), ('bvar', 1)))))))),
      'invalid: f n = f (S n)': (False, ('fix', 0, ('pi', nat_h, nat_h), ('lam', nat_h, ('app', ('bvar', 1), ('app', succ_h, ('bvar', 0)))))),
    }
    out['fix_guard'] = []
    for nm, (expect, tree) in fixes.items():
        ok, ty = knew.infer_open(build_new(knew, tree), [])
        out['fix_guard'].append((nm, expect, ok, ok == expect))
    return out


# ------------------------------------------------------------------ F. dependent typing: refl proofs whose statement needs delta/iota/beta in the TYPE
def test_dependent(N, seed):
    kref, knew, g, zf = setup(); rng = random.Random(seed); G = Gen(g, rng, zf); G.allow_fix = False
    NAT = ('const', g['Nat']); EQ = ('const', g['Eq']); REFL = ('const', g['refl']); out = [0, 0, []]
    eq = lambda a, b: ('app', ('app', ('app', EQ, NAT), a), b)
    refl = lambda a: ('app', ('app', REFL, NAT), a)
    def ref_accepts(proof, stmt):
        try: kref.whole_term_check(build_ref(kref, proof), build_ref(kref, stmt)); return True
        except Exception: return False
    cases = []
    for i in range(N):
        a = G.gen('nat', [], 3)
        b = parse_raw(kref.dump_raw(kref.normalize_node(build_ref(kref, a))), False) if rng.random() < 0.5 else G.gen('nat', [], 3)
        cases.append((refl(a), eq(a, b)))
    # open statements under a binder: Pi x. Eq (add Zero x) x  (provable by reduction)  vs  Pi x. Eq (add x Zero) x  (needs induction)
    add = ('const', g['add']); zero = ('const', g['Zero'])
    cases.append((('lam', NAT, refl(('var', 0))), ('pi', NAT, eq(('app', ('app', add, zero), ('var', 0)), ('var', 0)))))
    cases.append((('lam', NAT, refl(('var', 0))), ('pi', NAT, eq(('app', ('app', add, ('var', 0)), zero), ('var', 0)))))
    cases.append((('lam', NAT, refl(('var', 0))), ('pi', NAT, eq(('var', 0), ('var', 0)))))
    for proof, stmt in cases:
        want = ref_accepts(proof, stmt); ok, msg = knew.check_open(build_new(knew, l2h(proof, 0)), build_new(knew, l2h(stmt, 0)), [])
        out[0] += 1; out[1] += want
        if want != ok: out[2].append((proof, stmt, want, ok, msg[:60]))
    return out

if __name__ == '__main__':
    t0 = time.time(); FAIL = []
    lib, bad, n1, n2 = test_library_checked()
    print("A. library re-declared through the CHECKED path (positivity, admission, guard, full typing of every definition):")
    for kind, name, st in lib: print("     %-18s %-10s %s" % (kind, name, st))
    if any(st != 'OK' for _, _, st in lib) or bad or n1 != n2: FAIL.append('A library')
    print("   global types equal after conversion: %s (%d vs %d globals)" % ("yes" if bad == 0 and n1 == n2 else "NO (%d mismatches)" % bad, n1, n2))
    print("B. admission corpus (ref accepts/rejects == new == expected, rejection reason matches; the F92 rows: the reference accepts, the hybrid kernel rejects):")
    for r in test_admission():
        if not r[4]: FAIL.append('B ' + r[0])
        print("     %-46s expect %-6s ref %-6s new %-6s %s  %s" % (r[0], "accept" if r[1] else "reject", "accept" if r[2] else "reject", "accept" if r[3] else "reject", "OK " if r[4] else "BAD", r[5]))
    print("C. structural-recursion guard corpus:")
    for r in test_guards():
        if not r[4]: FAIL.append('C ' + r[0])
        print("     %-46s expect %-6s ref %-6s new %-6s %s  %s" % (r[0], "accept" if r[1] else "reject", "accept" if r[2] else "reject", "accept" if r[3] else "reject", "OK " if r[4] else "BAD", r[5]))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    m = test_match_typing(N, 5)
    if m['ok'][1] or m['reject'][1]: FAIL.append('D Match typing / rejections: %d + %d' % (len(m['ok'][1]), len(m['reject'][1])))
    print("D. Match typing, %d random typed terms with Match: new accepts what the reference accepts: %d checked, %d disagreements | ill-typed matches rejected by both: %d checked, %d disagreements" % (N, m['ok'][0], len(m['ok'][1]), m['reject'][0], len(m['reject'][1])))
    for k_ in ('ok', 'reject'):
        for ex in m[k_][1][:2]: print("     FIRST FAILURE [%s]: %s" % (k_, str(ex)[:300]))
    df = test_dependent(N, 9)
    if df[2]: FAIL.append('F dependent typing: %d' % len(df[2]))
    print("F. dependent typing (refl a : Eq a b, with delta/iota/beta needed in the type): %d checked (%d accepted by the reference) | disagreements %d" % (df[0], df[1], len(df[2])))
    for ex in df[2][:2]: print('     FIRST FAILURE:', str(ex)[:300])
    print("   timing, Match typing: reference whole_term_check %.0f ms vs hybrid check %.0f ms (ratio %.2fx)" % (m['t'][0] * 1e3, m['t'][1] * 1e3, m['t'][1] / max(m['t'][0], 1e-9)))
    pc = test_proof_corpus()
    if pc['bad'] or any(not good for _, _, _, good in pc['fix_guard']): FAIL.append('E proof corpus / Fix guard')
    print("E. proof corpus: %d real Fix+Match proofs (%d Fix nodes): accepted at their statement: %d | proofs of FALSE statements rejected: %d | problems: %d" % (pc['n'], pc['fixes'], pc['accepted'], pc['wrong_stmt_rejected'], len(pc['bad'])))
    for b in pc['bad'][:3]: print("     PROBLEM:", b)
    for nm, expect, ok, good in pc['fix_guard']: print("     hand-built Fix, %-52s expect %-6s got %-6s %s" % (nm, "accept" if expect else "reject", "accept" if ok else "reject", "OK" if good else "BAD"))
    print("elapsed %.1fs" % (time.time() - t0))
    print("S2 GATE:", "PASS" if not FAIL else "FAIL -- " + '; '.join(FAIL))
    sys.exit(1 if FAIL else 0)
