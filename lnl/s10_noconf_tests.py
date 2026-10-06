"""F2b gate (injectivity): generated lemmas are accepted by the kernel's CHECKED path and by the independent checker, and they close the goals discrimination cannot."""
import sys, functools
sys.setrecursionlimit(30000)
import harness as hn
from harness import *
from s3_tests import V, app
import s6_logic_tests as t6
import s7_param_tests as t7
import s8_indep_tests as t8
import indep_checker as ic
import noconf

T0 = ('type', 0)
def run(params):
    kref, knew, g = t6.build_logic()
    if params: knew.set_option('parameters', True)
    res = {}
    gids = {n: g[n] for n in ('True', 'Eq', 'transport', 'refl')}
    fac = lambda: ic.Checker(t8.build_env(knew), params=params, fuel=2_000_000)
    res['Nat: lemmas declared through the checked path'] = noconf.generate_injectivity(knew, build_new, parse_raw, fac, g['Nat'], gids)
    # a binary family: leaf | node l r
    tid = t7.declare_family_new(knew, 'Tree', T0, [('leaf', [], []), ('node', [('self',), ('self',)], [])])
    g = {name: gid for gid, name, _, _ in knew.list_globals()}
    res['Tree'] = noconf.generate_injectivity(knew, build_new, parse_raw, fac, tid, gids)
    g = {name: gid for gid, name, _, _ in knew.list_globals()}          # refresh: the lemmas now exist
    NAT, EQ, S = ('const', g['Nat']), ('const', g['Eq']), ('const', g['Succ']); TREE, NODE = ('const', g['Tree']), ('const', g['node'])
    eq = lambda A, a, b: app(EQ, A, a, b)
    checks = {}
    # S a = S b -> a = b : the goal cases could not close
    stmt = functools.reduce(lambda t, d: ('pi', d, t), reversed([NAT, NAT, eq(NAT, app(S, V(0)), app(S, V(1)))]), eq(NAT, V(0), V(1)))
    ps, log = t7.run_new(knew, stmt, [('intro',)] * 3 + [('apply', app(('const', g['Succ_inj_0']), V(0), V(1), V(2)))])
    checks['S a = S b -> a = b'] = (ps, stmt)
    # node l r = node l' r' -> r = r'   (the SECOND payload field)
    doms = [TREE, TREE, TREE, TREE, eq(TREE, app(NODE, V(0), V(1)), app(NODE, V(2), V(3)))]
    stmt2 = functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), eq(TREE, V(1), V(3)))
    ps2, _ = t7.run_new(knew, stmt2, [('intro',)] * 5 + [('apply', app(('const', g['node_inj_1']), V(0), V(1), V(2), V(3), V(4)))])
    checks['node l r = node l2 r2 -> r = r2'] = (ps2, stmt2)
    stmt3 = functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), eq(TREE, V(0), V(2)))
    ps3, _ = t7.run_new(knew, stmt3, [('intro',)] * 5 + [('apply', app(('const', g['node_inj_0']), V(0), V(1), V(2), V(3), V(4)))])
    checks['node l r = node l2 r2 -> l = l2'] = (ps3, stmt3)
    chk = fac()
    out = {}
    for name, (ps, st) in checks.items():
        proof = parse_raw(knew.dump_raw(ps.extract()), True) if ps.is_complete() else None
        out[name] = (proof is not None and t8.kernel_verdict(knew, proof, l2h(st, 0)), proof is not None and chk.check_closed(proof, l2h(st, 0))[0])
    bad = [b for b in chk.check_definitions() if b[1] not in ('cong', 'le_trans', 'transport')] if params else chk.check_definitions()
    out['independent checker re-checks every global (generated Inj and lemmas included)'] = (not bad, not bad)
    # the converse direction must NOT be provable this way: a wrong lemma use is rejected
    wrong = functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), eq(TREE, V(0), V(3)))        # l = r2: not implied
    psw, logw = t7.run_new(knew, wrong, [('intro',)] * 5 + [('apply', app(('const', g['node_inj_0']), V(0), V(1), V(2), V(3), V(4)))])
    out['l = r2 is NOT provable by node_inj_0 (the apply must fail)'] = (not psw.is_complete(), not psw.is_complete())
    return res, out

def run_param_families(params):
    """injectivity for families whose slots are all parameters: List (declared here), And, Or, Ex. Usage proofs by tactics; the independent checker re-checks every global."""
    kref, knew, g = t6.build_logic()
    if params: knew.set_option('parameters', True)
    gids = {n: g[n] for n in ('True', 'Eq', 'transport', 'refl')}; fac = lambda: ic.Checker(t8.build_env(knew), params=params, fuel=4_000_000)
    lid = t7.declare_family_new(knew, 'List', ('pi', T0, T0), [('nil', [T0], [('var', 0)]), ('cons', [T0, ('var', 0), ('app', ('self',), ('var', 0))], [('var', 0)])])
    gen = {}
    for fam in ('List', 'And', 'Or', 'Ex'):
        gid = lid if fam == 'List' else g[fam]; gen[fam] = noconf.generate_injectivity(knew, build_new, parse_raw, fac, gid, gids)
    g = {name: gid for gid, name, _, _ in knew.list_globals()}
    expected = {'List': {'nil': [None], 'cons': [None, 'cons_inj_1', 'cons_inj_2']}, 'And': {'conj': [None, None, 'conj_inj_2', 'conj_inj_3']},
                'Or': {'inl': [None, None, 'inl_inj_2'], 'inr': [None, None, 'inr_inj_2']}, 'Ex': {'ex_intro': [None, None, 'ex_intro_inj_2', None]}}
    out = {'generated lemma names (Ex: the proof component is dependent, so None)': (gen == expected, gen == expected)}
    C = lambda n: ('const', g[n]); EQ = C('Eq'); eq = lambda A, a, b: app(EQ, A, a, b); LIST, CONS, AND, CONJ, OR, INL, EX, EXI = [C(n) for n in ('List', 'cons', 'And', 'conj', 'Or', 'inl', 'Ex', 'ex_intro')]
    chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)
    lt = lambda a: app(LIST, a)
    cases = {}
    d_list = [T0, V(0), V(0), lt(V(0)), lt(V(0)), eq(lt(V(0)), app(CONS, V(0), V(1), V(3)), app(CONS, V(0), V(2), V(4)))]            # A x y xs ys h
    cases['List: cons x xs = cons y ys -> x = y'] = (chain(d_list, eq(V(0), V(1), V(2))), 'cons_inj_1', [0, 1, 3, 2, 4, 5])
    cases['List: cons x xs = cons y ys -> xs = ys'] = (chain(d_list, eq(lt(V(0)), V(3), V(4))), 'cons_inj_2', [0, 1, 3, 2, 4, 5])
    d_and = [T0, T0, V(0), V(1), V(0), V(1), eq(app(AND, V(0), V(1)), app(CONJ, V(0), V(1), V(2), V(3)), app(CONJ, V(0), V(1), V(4), V(5)))]  # A B a b a2 b2 h
    cases['And: conj a b = conj a2 b2 -> a = a2'] = (chain(d_and, eq(V(0), V(2), V(4))), 'conj_inj_2', [0, 1, 2, 3, 4, 5, 6])
    cases['And: conj a b = conj a2 b2 -> b = b2'] = (chain(d_and, eq(V(1), V(3), V(5))), 'conj_inj_3', [0, 1, 2, 3, 4, 5, 6])
    d_or = [T0, T0, V(0), V(0), eq(app(OR, V(0), V(1)), app(INL, V(0), V(1), V(2)), app(INL, V(0), V(1), V(3)))]                          # A B a a2 h
    cases['Or: inl a = inl a2 -> a = a2'] = (chain(d_or, eq(V(0), V(2), V(3))), 'inl_inj_2', [0, 1, 2, 3, 4])
    d_ex = [T0, ('pi', V(0), T0), V(0), V(0), app(V(1), V(2)), app(V(1), V(3)), eq(app(EX, V(0), V(1)), app(EXI, V(0), V(1), V(2), V(4)), app(EXI, V(0), V(1), V(3), V(5)))]   # A P w w2 p p2 h
    cases['Ex: ex_intro w p = ex_intro w2 p2 -> w = w2'] = (chain(d_ex, eq(V(0), V(2), V(3))), 'ex_intro_inj_2', [0, 1, 2, 4, 3, 5, 6])
    chk = fac()
    for name, (stmt, lemma, order) in cases.items():
        ps, log = t7.run_new(knew, stmt, [('intro',)] * (len(order)) + [('apply', app(C(lemma), *[V(i) for i in order]))])
        proof = parse_raw(knew.dump_raw(ps.extract()), True) if (ps.num_goals() == 0 and ps.is_complete()) else None
        out[name] = (proof is not None and t8.kernel_verdict(knew, proof, l2h(stmt, 0)), proof is not None and chk.check_closed(proof, l2h(stmt, 0))[0])
    bad = [b for b in chk.check_definitions() if b[1] not in ('cong', 'le_trans', 'transport')] if params else chk.check_definitions()
    out['independent checker re-checks every global (generated Inj and lemmas included)'] = (not bad, not bad)
    return out

if __name__ == '__main__':
    FAIL = []
    for params in (False, True):
        res, out = run(params); print("%s mode: generated %s" % ("parameter" if params else "legacy", {k: v for k, v in res['Tree'].items()}))
        for k, (kv, iv) in out.items():
            print("   %-72s kernel %-5s independent %s" % (k, kv, iv))
            if not (kv and iv): FAIL.append(('params ' if params else 'legacy ') + k)
    for params in (False, True):
        print("%s mode, families whose slots are all parameters:" % ("parameter" if params else "legacy"))
        for k, (kv, iv) in run_param_families(params).items():
            print("   %-76s kernel %-5s independent %s" % (k, kv, iv))
            if not (kv and iv): FAIL.append(('params ' if params else 'legacy ') + 'param family: ' + k)
    print("S10 NOCONF GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
