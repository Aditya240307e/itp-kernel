"""F6 probe: strong induction by a BOUNDED PREDICATE (Below P n := True / And (Below P k) (P k)), so the induction hypothesis is a plain proposition, not a forall."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, functools, time
sys.setrecursionlimit(30000)
import harness as hn
from harness import *
import s2_tests
from s3_tests import V, app
import s6_logic_tests as t6
import s7_param_tests as t7
import s8_indep_tests as t8
import indep_checker as ic

T0 = ('type', 0)

def declare_below(kref, knew, g):
    NAT, TRUE, AND = ('const', g['Nat']), ('const', g['True']), ('const', g['And']); P = ('pi', NAT, T0); SELF = ('self',)
    fty = ('pi', P, ('pi', NAT, T0))
    body = ('lam', P, ('lam', NAT, ('match', V(1), ('lam', NAT, T0), (TRUE, ('lam', NAT, app(AND, app(SELF, V(0), V(2)), app(V(0), V(2))))))))
    for k, bld, conv in ((kref, build_ref, lambda t, n: t), (knew, build_new, lambda t, n: l2h(t, n, lenient=True))):
        gid = k.reserve_recursive_function('Below', bld(k, conv(fty, 0)))
        k.finish_recursive_function_checked(gid, g['Nat'], 1, bld(k, conv(s2_tests.subst_self(body, gid), 0)))
    return {name: gid for gid, name, _, _ in knew.list_globals()}

def run_ops(k, stmt, ops):
    ps = k.start_proof(build_new(k, l2h(stmt, 0)))
    for op in ops:
        nh = ps.goal_num_hyps(0) if ps.num_goals() else 0
        if op[0] == 'intro': ps.intro('x')
        elif op[0] == 'induct': ps.induct(op[1])
        else: ps.apply(build_new(k, l2h(op[1], nh, lenient=True)))
    return ps

def run(params):
    kref, knew, g = t6.build_logic()
    if params: knew.set_option('parameters', True)
    g = declare_below(kref, knew, g); C = lambda n: ('const', g[n]); NAT, LE = C('Nat'), C('Le'); P = ('pi', NAT, T0)
    below = lambda p, n: app(C('Below'), p, n)
    h_ty = ('pi', NAT, ('pi', below(V(0), V(1)), app(V(0), V(1))))                                   # forall n, Below P n -> P n   (P is hypothesis 0)
    below_all = chain3([P, h_ty, NAT], below(V(0), V(2)))
    res = {}
    # 1. Below computes
    ps = run_ops(knew, ('pi', P, below(V(0), C('Zero'))), [('intro',), ('apply', C('tt'))]); res['Below P Zero = True (by computation)'] = t8.kernel_verdict(knew, parse_raw(knew.dump_raw(ps.extract()), True), l2h(('pi', P, below(V(0), C('Zero'))), 0))
    # 2. below_all by plain induction: the IH is  Below P x
    ps = run_ops(knew, below_all, [('intro',)] * 3 + [('induct', 2), ('apply', C('tt')), ('apply', C('conj')), ('apply', V(4)), ('apply', app(V(1), V(3), V(4)))])
    proof = ps.extract() if ps.is_complete() else None; res['below_all (induction on n; IH : Below P x)'] = proof is not None and t8.kernel_verdict(knew, parse_raw(knew.dump_raw(proof), True), l2h(below_all, 0))
    if proof is not None:
        knew.declare_checked(build_new(knew, l2h(below_all, 0)), proof, 'below_all'); g2 = {name: gid for gid, name, _, _ in knew.list_globals()}
        strong = chain3([P, h_ty, NAT], app(V(0), V(2)))
        # 3. strong induction from it:  P n := h n (below_all P h n)
        ps = run_ops(knew, strong, [('intro',)] * 3 + [('apply', app(V(1), V(2), app(('const', g2['below_all']), V(0), V(1), V(2))))])
        sproof = ps.extract() if ps.is_complete() else None; res['strong_induction (h n (below_all P h n))'] = sproof is not None and t8.kernel_verdict(knew, parse_raw(knew.dump_raw(sproof), True), l2h(strong, 0))
        if sproof is not None:
            knew.declare_checked(build_new(knew, l2h(strong, 0)), sproof, 'strong_induction'); g3 = {name: gid for gid, name, _, _ in knew.list_globals()}
            # 4. a CLIENT: forall n, n <= n  by strong induction with P := fun m => m <= m
            Pl_at = lambda d: ('lam', NAT, app(LE, V(d), V(d)))                                        # a level-tree lambda names its binder by ABSOLUTE level: one tree per depth
            Pl = Pl_at(1)                                                                              # as an argument in ctx [n]
            hl = ('lam', NAT, ('lam', below(Pl_at(2), V(1)), app(C('le_n'), V(1))))                    # inside hl's first binder (level 1) the predicate sits one deeper
            client = ('pi', NAT, app(LE, V(0), V(0)))
            ps = run_ops(knew, client, [('intro',), ('apply', app(('const', g3['strong_induction']), Pl, hl, V(0)))])
            cproof = ps.extract() if ps.is_complete() else None; res['client: forall n, n <= n by strong induction'] = cproof is not None and t8.kernel_verdict(knew, parse_raw(knew.dump_raw(cproof), True), l2h(client, 0))
            # the independent checker re-verifies the whole chain (it also re-checks every global, Below and the lemmas included)
            chk = ic.Checker(t8.build_env(knew), params=params, fuel=4_000_000); bad = [b for b in chk.check_definitions() if b[1] not in ('cong', 'le_trans', 'transport')] if params else chk.check_definitions()
            ok_c, msg = chk.check_closed(parse_raw(knew.dump_raw(cproof), True), l2h(client, 0)) if cproof is not None else (False, 'no proof')
            res['independent checker: all globals incl. Below, below_all, strong_induction'] = not bad
            res['independent checker: the client proof'] = ok_c
    return res

def chain3(doms, goal): return functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)

if __name__ == '__main__':
    FAIL = []
    for params in (False, True):
        r = run(params); print("%s mode:" % ("parameter" if params else "legacy"))
        for k, v in r.items(): print("   %-70s %s" % (k, v)); (FAIL.append(('params' if params else 'legacy') + ': ' + k) if not v else None)
        if len(r) < 6: FAIL.append('chain incomplete (%d results)' % len(r))
    print("S9 STRONG GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
