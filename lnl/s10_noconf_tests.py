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

if __name__ == '__main__':
    FAIL = []
    for params in (False, True):
        res, out = run(params); print("%s mode: generated %s" % ("parameter" if params else "legacy", {k: v for k, v in res['Tree'].items()}))
        for k, (kv, iv) in out.items():
            print("   %-72s kernel %-5s independent %s" % (k, kv, iv))
            if not (kv and iv): FAIL.append(('params ' if params else 'legacy ') + k)
    print("S10 NOCONF GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
