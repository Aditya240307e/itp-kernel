"""prop_elim gate. With the option, level 0 is Prop: an inductive family in Type0 that is not a SYNTACTIC SUBSINGLETON (no constructor, or one whose every field is a proof or occurs in the result indices)
can only be eliminated into Type0, and large fields are allowed in such families; data lives at level >= 1. This is the soundness prerequisite for classical axioms (Berardi, Reynolds). Fresh kernels, no legacy
library. Every case is judged by the kernel AND the independent checker (written from the same specification), with the option on and, as a control, off."""
import sys, functools
sys.setrecursionlimit(30000)
import harness as H
from harness import *
from s3_tests import V, app
import s7_param_tests as t7
import s8_indep_tests as t8
import indep_checker as ic

T0, T1, SELF = ('type', 0), ('type', 1), ('self',)

def world(prop, parameters=False):
    k = H.Kernel()
    if prop: k.set_option('prop_elim', True)
    if parameters: k.set_option('parameters', True)
    adm = {}
    def decl(name, sort, ctors):
        try: t7.declare_family_new(k, name, sort, ctors); adm[name] = True
        except Exception as e: adm[name] = False
    decl('Nat', T1, [('Zero', [], []), ('Succ', [SELF], [])])
    decl('True', T0, [('tt', [], [])]); decl('False', T0, [])
    decl('Eq', ('pi', T1, ('pi', V(0), ('pi', V(0), T0))), [('refl', [T1, ('var', 0)], [('var', 0), ('var', 1), ('var', 1)])])
    decl('And', ('pi', T0, ('pi', T0, T0)), [('conj', [T0, T0, ('var', 0), ('var', 1)], [('var', 0), ('var', 1)])])
    decl('Or', ('pi', T0, ('pi', T0, T0)), [('inl', [T0, T0, ('var', 0)], [('var', 0), ('var', 1)]), ('inr', [T0, T0, ('var', 1)], [('var', 0), ('var', 1)])])
    decl('Ex', ('pi', T1, ('pi', ('pi', V(0), T0), T0)), [('ex_intro', [T1, ('pi', ('var', 0), T0), ('var', 0), ('app', ('var', 1), ('var', 2))], [('var', 0), ('var', 1)])])
    decl('Box', T0, [('box', [T0], [])])
    return k, adm

def verdicts(k, term, stmt, prop):
    try: kv = k.check_open(build_new(k, l2h(term, 0)), build_new(k, l2h(stmt, 0)), []); kok, kmsg = bool(kv[0]), (kv[1] if len(kv) > 1 else '')
    except Exception as e: kok, kmsg = False, str(e)
    iok, imsg = ic.Checker(t8.build_env(k), prop_elim=prop, fuel=2_000_000).check_closed(l2h(term, 0), l2h(stmt, 0))
    return kok, str(kmsg), iok, imsg

def cases(g):
    C = lambda n: ('const', g[n]); NAT, ZERO, S, OR, AND, EX, EQ, FALSE, TRUE, BOX = [C(n) for n in ('Nat', 'Zero', 'Succ', 'Or', 'And', 'Ex', 'Eq', 'False', 'True', 'Box')]
    INL, INR = C('inl'), C('inr'); mat = lambda s, m, bs: ('match', s, m, tuple(bs)); lam = lambda d, b: ('lam', d, b); pi = lambda d, b: ('pi', d, b)
    out = []
    # (name, statement, term, expected acceptance with the option ON)
    out.append(('Or -> Nat (large elimination of a non-subsingleton proposition)', pi(T0, pi(T0, pi(app(OR, V(0), V(1)), NAT))),
        lam(T0, lam(T0, lam(app(OR, V(0), V(1)), mat(V(2), lam(T0, lam(T0, lam(app(OR, V(3), V(4)), NAT))),
            [lam(T0, lam(T0, lam(V(0), ZERO))), lam(T0, lam(T0, lam(V(1), app(S, ZERO))))])))), False))
    out.append(('Or -> Or (elimination into Type0 is fine)', pi(T0, pi(T0, pi(app(OR, V(0), V(1)), app(OR, V(1), V(0))))),
        lam(T0, lam(T0, lam(app(OR, V(0), V(1)), mat(V(2), lam(T0, lam(T0, lam(app(OR, V(3), V(4)), app(OR, V(4), V(3))))),
            [lam(T0, lam(T0, lam(V(0), app(INR, V(1), V(0), V(5))))), lam(T0, lam(T0, lam(V(1), app(INL, V(1), V(0), V(5)))))])))), True))
    out.append(('And -> Nat (a subsingleton: large elimination allowed)', pi(T0, pi(T0, pi(app(AND, V(0), V(1)), NAT))),
        lam(T0, lam(T0, lam(app(AND, V(0), V(1)), mat(V(2), lam(T0, lam(T0, lam(app(AND, V(3), V(4)), NAT))), [lam(T0, lam(T0, lam(V(0), lam(V(1), ZERO))))])))), True))
    out.append(('Ex A P -> A (witness extraction into data)', pi(T1, pi(pi(V(0), T0), pi(app(EX, V(0), V(1)), V(0)))),
        lam(T1, lam(pi(V(0), T0), lam(app(EX, V(0), V(1)), mat(V(2), lam(T1, lam(pi(V(3), T0), lam(app(EX, V(3), V(4)), V(3)))),
            [lam(T1, lam(pi(V(3), T0), lam(V(0), lam(app(V(1), V(5)), V(5)))))])))), False))
    out.append(('Eq Nat a b -> Nat (J into data: Eq is a subsingleton)', pi(NAT, pi(NAT, pi(app(EQ, NAT, V(0), V(1)), NAT))),
        lam(NAT, lam(NAT, lam(app(EQ, NAT, V(0), V(1)), mat(V(2), lam(T1, lam(V(3), lam(V(3), lam(app(EQ, V(3), V(4), V(5)), NAT)))), [lam(T1, lam(NAT, ZERO))])))), True))
    out.append(('False -> Nat', pi(FALSE, NAT), lam(FALSE, mat(V(0), lam(FALSE, NAT), [])), True))
    out.append(('Nat -> Type0 (data into types: Nat lives in Type1, unrestricted)', pi(NAT, T0), lam(NAT, mat(V(0), lam(NAT, T0), [TRUE, lam(NAT, FALSE)])), True))
    out.append(('Box -> Type0 (the Hurkens route: unbox)', pi(BOX, T0), lam(BOX, mat(V(0), lam(BOX, T0), [lam(T0, V(1))])), False))
    return out

def tactic_cases(k, g):
    C = lambda n: ('const', g[n]); NAT, OR, AND = C('Nat'), C('Or'), C('And')
    to_nat = lambda F: ('pi', T0, ('pi', T0, ('pi', app(F, V(0), V(1)), NAT)))
    to_or = ('pi', T0, ('pi', T0, ('pi', app(OR, V(0), V(1)), app(OR, V(1), V(0)))))
    res = {}
    for name, stmt in (('cases on Or with a data goal', to_nat(OR)), ('cases on Or with a Prop goal', to_or), ('cases on And with a data goal', to_nat(AND))):
        ps, log = t7.run_new(k, stmt, [('intro',), ('intro',), ('intro',), ('cases', 2)])
        res[name] = (log[-1][1], log[-1][2]) if len(log) == 4 else (False, 'setup failed')
    return res

if __name__ == '__main__':
    FAIL = []; EXPECT_ADM = {'on': {'Nat': 1, 'True': 1, 'False': 1, 'Eq': 1, 'And': 1, 'Or': 1, 'Ex': 1, 'Box': 1}, 'off': {'Nat': 1, 'True': 1, 'False': 1, 'Eq': 1, 'And': 1, 'Or': 1, 'Ex': 0, 'Box': 0}}
    for label, prop, params in (('option OFF (control)', False, False), ('option ON', True, False), ('option ON + parameters ON', True, True)):
        k, adm = world(prop, params); g = {name: gid for gid, name, _, _ in k.list_globals()}
        exp = EXPECT_ADM['on' if prop else 'off']
        print("%s: admitted %s" % (label, {n: int(v) for n, v in adm.items()}))
        if {n: int(v) for n, v in adm.items()} != exp: FAIL.append(label + ': admission differs from the expectation ' + str(exp))
        for name, stmt, term, accepted_on in cases(g):
            if not all(adm.get(f, True) for f in ('Ex',) if 'Ex A P' in name) or (not adm['Box'] and 'Box' in name): print("   %-72s (family not admitted in this mode)" % name); continue
            kok, kmsg, iok, imsg = verdicts(k, term, stmt, prop)
            expected = accepted_on if prop else True                    # with the option off every one of these terms is well typed
            ok = (kok == expected) and (iok == expected) and (prop is False or accepted_on or ('large elimination' in kmsg and 'large elimination' in imsg))
            print("   %-72s kernel %-5s independent %-5s expected %-5s %s" % (name, kok, iok, expected, '' if ok else '  <-- MISMATCH ' + kmsg[:70] + ' | ' + imsg[:50]))
            if not ok: FAIL.append(label + ': ' + name)
        for name, (okk, msg) in tactic_cases(k, g).items():
            expected = True if (not prop or 'Prop goal' in name or 'And' in name) else False
            ok = (okk == expected) and (expected or 'not a syntactic subsingleton' in msg)
            print("   tactic: %-62s accepted %-5s expected %-5s %s" % (name, okk, expected, '' if ok else '  <-- ' + msg[:90]))
            if not ok: FAIL.append(label + ': tactic ' + name)
    print("S13 PROP-ELIM GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
