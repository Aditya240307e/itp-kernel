"""prop_world.py -- the library for the Prop/Type world: kernel options prop_elim and parameters ON, data in Type1, propositions in Type0.
Built from scratch in the hybrid kernel (no reference kernel, no legacy library). Families are admitted through the CHECKED path; every theorem is PROVED BY TACTICS in this mode, extracted and admitted through
the checked path, so no term written for the legacy Match rule is reused (finding F20: those do not check under the parameter rule)."""
import sys, functools
sys.setrecursionlimit(30000)
import harness as H
from harness import *
from s3_tests import V, app
import s2_tests
import s7_param_tests as t7

T0, T1, SELF = ('type', 0), ('type', 1), ('self',)

def build_prop_world(classical=False):
    k = H.Kernel(); k.set_option('prop_elim', True); k.set_option('parameters', True)
    G = lambda: {name: gid for gid, name, _, _ in k.list_globals()}
    fam = lambda name, sort, ctors: t7.declare_family_new(k, name, sort, ctors)
    fam('Nat', T1, [('Zero', [], []), ('Succ', [SELF], [])]); g = G(); NAT, ZERO, S = ('const', g['Nat']), ('const', g['Zero']), ('const', g['Succ'])
    fam('True', T0, [('tt', [], [])]); fam('False', T0, [])
    fam('Eq', ('pi', T1, ('pi', V(0), ('pi', V(0), T0))), [('refl', [T1, ('var', 0)], [('var', 0), ('var', 1), ('var', 1)])])
    fam('And', ('pi', T0, ('pi', T0, T0)), [('conj', [T0, T0, ('var', 0), ('var', 1)], [('var', 0), ('var', 1)])])
    fam('Or', ('pi', T0, ('pi', T0, T0)), [('inl', [T0, T0, ('var', 0)], [('var', 0), ('var', 1)]), ('inr', [T0, T0, ('var', 1)], [('var', 0), ('var', 1)])])
    fam('Ex', ('pi', T1, ('pi', ('pi', V(0), T0), T0)), [('ex_intro', [T1, ('pi', ('var', 0), T0), ('var', 0), ('app', ('var', 1), ('var', 2))], [('var', 0), ('var', 1)])])
    fam('Le', ('pi', NAT, ('pi', NAT, T0)), [('le_n', [NAT], [('var', 0), ('var', 0)]), ('le_S', [NAT, NAT, app(SELF, ('var', 0), ('var', 1))], [('var', 0), app(S, ('var', 1))])])
    g = G(); C = lambda n: ('const', g[n]); FALSE, AND = C('False'), C('And')
    def defn(name, ty, val): k.declare_checked(build_new(k, l2h(ty, 0)), build_new(k, l2h(val, 0)), name)
    defn('Not', ('pi', T0, T0), ('lam', T0, ('pi', V(0), FALSE)))
    defn('Iff', ('pi', T0, ('pi', T0, T0)), ('lam', T0, ('lam', T0, app(AND, ('pi', V(0), V(1)), ('pi', V(1), V(0))))))
    # add : Nat -> Nat -> Nat, structural on the first argument
    fty = ('pi', NAT, ('pi', NAT, NAT)); body = ('lam', NAT, ('lam', NAT, ('match', V(0), ('lam', NAT, NAT), (V(1), ('lam', NAT, app(S, app(SELF, V(2), V(1))))))))
    gid = k.reserve_recursive_function('add', build_new(k, l2h(fty, 0)))
    k.finish_recursive_function_checked(gid, g['Nat'], 0, build_new(k, l2h(s2_tests.subst_self(body, gid), 0, lenient=True)))
    def prove(name, stmt, ops):
        g_ = G(); C_ = lambda n: ('const', g_[n])
        ps = k.start_proof(build_new(k, l2h(stmt, 0)))
        for op in ops:
            nh = ps.goal_num_hyps(0) if ps.num_goals() else 0
            if op[0] == 'intro': ps.intro('x')
            elif op[0] == 'cases': ps.cases(op[1])
            elif op[0] == 'induct': ps.induct(op[1])
            else: ps.apply(build_new(k, l2h(op[1] if not isinstance(op[1], str) else C_(op[1]), nh, lenient=True)))
        assert ps.num_goals() == 0 and ps.is_complete(), "the tactic script for %s did not complete" % name
        k.declare_checked(build_new(k, l2h(stmt, 0)), ps.extract(), name)
    eq = lambda A, a, b: app(('const', G()['Eq']), A, a, b); le = lambda a, b: app(('const', G()['Le']), a, b)
    chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)
    # A x y : the equation is eliminated by cases on its index
    prove('transport', chain([T1, ('pi', V(0), T0), V(0), V(0), eq(V(0), V(2), V(3)), app(V(1), V(2))], app(V(1), V(3))), [('intro',)] * 6 + [('cases', 4), ('apply', V(5))])
    prove('symm', chain([T1, V(0), V(0), eq(V(0), V(1), V(2))], eq(V(0), V(2), V(1))), [('intro',)] * 4 + [('cases', 3), ('apply', 'refl')])
    prove('trans', chain([T1, V(0), V(0), V(0), eq(V(0), V(1), V(2)), eq(V(0), V(2), V(3))], eq(V(0), V(1), V(3))), [('intro',)] * 6 + [('cases', 5), ('apply', V(4))])
    prove('cong', chain([NAT, NAT, eq(NAT, V(0), V(1))], eq(NAT, app(S, V(0)), app(S, V(1)))), [('intro',)] * 3 + [('cases', 2), ('apply', 'refl')])
    prove('le_trans', chain([NAT, NAT, NAT, le(V(0), V(1)), le(V(1), V(2))], le(V(0), V(2))), [('intro',)] * 5 + [('induct', 4), ('apply', V(3)), ('apply', 'le_S'), ('apply', V(8))])
    prove('zero_le', chain([NAT], le(ZERO, V(0))), [('intro',), ('induct', 0), ('apply', 'le_n'), ('apply', 'le_S'), ('apply', V(2))])
    if classical: add_classical(k, G, defn, prove)
    return k, G()

CLASSICAL_AXIOMS = ('em', 'propext', 'funext', 'choice')      # the trusted base added on top of the kernel

def add_classical(k, G, defn, prove):
    """The four standard classical axioms, in the Prop/Type world only (the elimination restriction is what keeps them consistent with an impredicative Prop), and the derived principles, proved by tactics."""
    g = G(); C = lambda n: ('const', g[n]); NOT, OR, EQ, IFF, EX, TRUE, FALSE = [C(n) for n in ('Not', 'Or', 'Eq', 'Iff', 'Ex', 'True', 'False')]
    defn('Nonempty', ('pi', T1, T0), ('lam', T1, app(EX, V(0), ('lam', V(0), TRUE))))
    g = G(); NONEMPTY = ('const', g['Nonempty'])
    axiom = lambda name, ty: k.declare_unchecked(build_new(k, l2h(ty, 0)), None, name)
    axiom('em', ('pi', T0, app(OR, V(0), app(NOT, V(0)))))
    axiom('propext', ('pi', T0, ('pi', T0, ('pi', app(IFF, V(0), V(1)), app(EQ, T0, V(0), V(1))))))
    axiom('funext', ('pi', T1, ('pi', ('pi', V(0), T1), ('pi', ('pi', V(0), app(V(1), V(2))), ('pi', ('pi', V(0), app(V(1), V(3))),
                     ('pi', ('pi', V(0), app(EQ, app(V(1), V(4)), app(V(2), V(4)), app(V(3), V(4)))), app(EQ, ('pi', V(0), app(V(1), V(5))), V(2), V(3))))))))
    axiom('choice', ('pi', T1, ('pi', app(NONEMPTY, V(0)), V(0))))
    prove('false_elim', ('pi', T0, ('pi', FALSE, V(0))), [('intro',), ('intro',), ('cases', 1)])
    prove('or_elim', ('pi', T0, ('pi', T0, ('pi', T0, ('pi', app(OR, V(0), V(1)), ('pi', ('pi', V(0), V(2)), ('pi', ('pi', V(1), V(2)), V(2))))))),
          [('intro',)] * 6 + [('cases', 3), ('apply', V(9)), ('apply', V(8)), ('apply', V(10)), ('apply', V(8))])
    g = G(); C = lambda n: ('const', g[n])
    by_contra_term = app(C('or_elim'), V(0), app(NOT, V(0)), V(0), app(C('em'), V(0)), ('lam', V(0), V(2)), ('lam', app(NOT, V(0)), app(C('false_elim'), V(0), app(V(1), V(2)))))
    prove('by_contra', ('pi', T0, ('pi', app(NOT, app(NOT, V(0))), V(0))), [('intro',), ('intro',), ('apply', by_contra_term)])

if __name__ == '__main__':
    import s8_indep_tests as t8, indep_checker as ic
    k, g = build_prop_world()
    print("globals:", len(g), "| options: prop_elim %s, parameters %s" % (k.get_option('prop_elim'), k.get_option('parameters')))
    chk = ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=4_000_000)
    bad = chk.check_definitions()
    print("independent re-check of EVERY global under the parameter rule and the elimination restriction: %d rejected %s" % (len(bad), bad[:4]))
    kv = {n: [r for r in [k.get_option('prop_elim')]][0] for n in ()}
    sys.exit(1 if bad else 0)
