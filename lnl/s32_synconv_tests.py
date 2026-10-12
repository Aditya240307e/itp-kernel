"""S32 gate: the quick conversion check (SynConv) in front of the evaluator.

`unify` decides definitional equality by evaluating both sides with every definition unfolded. Most pairs a checker meets are equal after a definition or two (`Z.lt x y` against `Z.le (x+1) y`), and
evaluating them costs thousands of steps; the kernel now first tries to settle such a pair ON SYNTAX: identical nodes, one head applied to pairwise equal arguments, binders with equal parts, and after
that the head definition of one side unfolded (or a lambda head applied). It answers only YES, and every yes is a congruence, delta or beta step, so it can change how fast the kernel decides, never what
it accepts. The option `syn_conv` switches it off (the evaluator alone then decides), `syn_stats()` counts (pairs looked at, pairs settled).

Nothing here widens the trusted base unchecked: a build with -DITP_SYN_SHADOW re-decides every quick yes with the evaluator and aborts on a disagreement (the whole gate suite is run on that build too,
see README_S1.md). This gate pins the behaviour down from the outside:

  A. crafted pairs: for each, the verdict with the shortcut ON equals the verdict OFF equals the expected one, and the counters say whether the shortcut settled it (a pair equal after delta/beta/congruence),
     looked at it and gave up (an iota step, an eta step, a different head, a different argument) or never looked (identical nodes);
  B. the shortcut leaves metavariables alone: the same verdict, the same solutions, the same deferred constraints on and off;
  C. the limits (rounds of unfolding, nesting depth, steps in all; lowered with `set_syn_limits` so that the pairs stay small): a pair beyond a limit is handed to the evaluator (counted as not settled),
     and the evaluator still gets the verdict right; a wide pair that differs only at its end is NOT accepted when the shortcut runs out of budget; the setter refuses nonsense;
  D. random pairs generated from one term with different folding of its definitions (equal by construction) and with one place changed (usually different): ON and OFF agree on every one, and the shortcut
     settles a good share of the equal ones;
  E. a Pi hidden behind a definition (`AllLe a` is `forall b, a <= b`; one behind two definitions) in the expected type of a lambda and in the type of the function of an application: the definition is
     unfolded on syntax, and the binders and arguments consumed so far are substituted into what is exposed (a lambda `fun a b => hyp b a` must be refused);
  F. a whole library built with the shortcut ON and OFF (the unary and the integer libraries, with their tactic proofs): every global has the same type and value; and the final check of a large
     `nlinarith` proof does its work in at most 400000 steps (the kernel's own fuel meter; it needs about 88000 with the shortcut and 2.0 million without), which pins the speed-up down deterministically.
"""
import os, sys, random, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(100000)
import itp_env_lnl as E
import author
import lnl_base
from author import World

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


# ---------------------------------------------------------------------------------------------------------------------------------------- the world
def build():
    w = World()
    lnl_base.declare_types(w); lnl_base.declare_add(w)
    Nat, S, Z0, add, PROP, Le = w['Nat'], w['Succ'], w['Zero'], w['add'], w.PROP, w['Le']
    NN = w.arrow(Nat, Nat)
    w.define('two', Nat, S(S(Z0)))
    two = w['two']
    w.define('idn', NN, w.lam(Nat, lambda x: x))
    w.define('plus2', NN, w.lam(Nat, lambda n: add(n, two)))
    w.define('dbl', NN, w.lam(Nat, lambda n: add(n, n)))
    w.define('dbl2', NN, w.lam(Nat, lambda n: w['dbl'](n)))                                  # a definition of a definition
    w.define('konst', w.forall_([Nat, Nat], lambda a, b: Nat), w.fun([Nat, Nat], lambda a, b: a))
    w.define('adder', w.arrow(Nat, NN), w.lam(Nat, lambda n: add(n)))                       # ONE lambda, and the body is a function: `adder x y` has an argument more than the lambdas
    w.define('comp', w.forall_([NN, NN, Nat], lambda f, g, x: Nat), w.fun([NN, NN, Nat], lambda f, g, x: f(g(x))))
    w.define('twice', w.forall_([NN, Nat], lambda f, x: Nat), w.fun([NN, Nat], lambda f, x: f(f(x))))
    w.define('P0', w.arrow(Nat, PROP), w.lam(Nat, lambda n: Le(n, n)))
    w.define('Q0', w.arrow(Nat, PROP), w.lam(Nat, lambda n: w['P0'](w['idn'](n))))
    # definitions that hide a Pi:  AllLe a := forall b, a <= b   (dependent: the codomain mentions a),   Bnd := fun a => forall b, AllLe b -> a <= b ...
    w.define('AllLe', w.arrow(Nat, PROP), w.lam(Nat, lambda a: w.pi(Nat, lambda b: Le(a, b))))
    w.define('Imp2', w.forall_([PROP, PROP], lambda p, q: PROP), w.fun([PROP, PROP], lambda p, q: w.arrow(p, q)))
    w.define('AllLe2', w.arrow(Nat, w.arrow(Nat, PROP)), w.lam(Nat, lambda a: w.lam(Nat, lambda b: w['Imp2'](Le(a, b), Le(b, a)))))     # AllLe2 a b := (a <= b) -> (b <= a)   (a Pi hidden behind TWO definitions)
    # a chain of definitions e1 := idn, e_{k+1} := fun x => e_k x   (to count unfolding steps)
    w.define('e1', NN, w['idn'])
    for k in range(2, 41):
        w.define('e%d' % k, NN, w.lam(Nat, lambda x, k=k: w['e%d' % (k - 1)](x)))
    return w


def num(w, n):
    t = w['Zero']
    for _ in range(n): t = w['Succ'](t)
    return t


class Ctx:
    """a context of four natural numbers x y z u and one function f : Nat -> Nat, entered for the duration of a `with`"""
    def __init__(self, w): self.w = w
    def __enter__(self):
        w = self.w; Nat = w['Nat']; NN = w.arrow(Nat, Nat)
        self.vars = [w._bind(Nat, nm)[1] for nm in ('x', 'y', 'z', 'u')] + [w._bind(NN, 'f')[1]]
        return self.vars
    def __exit__(self, *exc): self.w._unbind(5)


def unify_both(w, a, b):
    """the kernel's unifier on (a, b) in the current context, with the quick check OFF and then ON; returns (off, on, tries, yes) where off/on = (ok, message, pending) and (tries, yes) are the counts of the ON run"""
    k = w.k; na, nb = w._n(a), w._n(b)
    k.set_option('syn_conv', False); s0 = k.syn_stats(); off = k.unify_open(na, nb, w.ctx); s1 = k.syn_stats()
    expect(s1 == s0, 'the counters moved with the option off: %s -> %s' % (s0, s1))
    k.set_option('syn_conv', True); on = k.unify_open(na, nb, w.ctx); s2 = k.syn_stats()
    return off, on, s2[0] - s1[0], s2[1] - s1[1]


def top(tries, yes):
    """how the quick check fared on the pair itself: 'same' (identical nodes, not looked at), 'settled' (it said yes, so the evaluator never ran: exactly one attempt), 'unsettled' (it gave up, and the
    evaluator that follows may look at more pairs -- the rule for a refused pair re-infers types -- some of which the quick check settles: so only `yes < tries` is certain)"""
    if tries == 0: return 'same'
    if tries == yes: return 'settled' if tries == 1 else 'confused'
    return 'unsettled' if yes < tries else 'confused'


def well_typed(w, t):
    ok, _ = w.k.infer_open(w._n(t), w.ctx)
    return ok


# ---------------------------------------------------------------------------------------------------------------------------------------- A: crafted pairs
# expected syn: 'same' (identical nodes: not looked at), 'yes' (settled by the quick check), 'no' (looked at, handed to the evaluator)
def pairs_A(w):
    Nat, S, Z0, add, Le = w['Nat'], w['Succ'], w['Zero'], w['add'], w['Le']
    two, idn, plus2, dbl, dbl2, konst, adder, comp, twice, P0, Q0 = (w[n] for n in ('two', 'idn', 'plus2', 'dbl', 'dbl2', 'konst', 'adder', 'comp', 'twice', 'P0', 'Q0'))
    lam_xx = lambda: w.lam(Nat, lambda n: add(n, n))
    return [
        # identical nodes
        ('identical', lambda x, y, z, u, f: (plus2(x), plus2(x)), True, 'same'),
        # delta
        ('idn x = x', lambda x, y, z, u, f: (idn(x), x), True, 'yes'),
        ('x = idn x', lambda x, y, z, u, f: (x, idn(x)), True, 'yes'),
        ('plus2 x = add x two', lambda x, y, z, u, f: (plus2(x), add(x, two)), True, 'yes'),
        ('add x two = plus2 x', lambda x, y, z, u, f: (add(x, two), plus2(x)), True, 'yes'),
        ('dbl x = add x x', lambda x, y, z, u, f: (dbl(x), add(x, x)), True, 'yes'),
        ('dbl2 x = add x x (two steps)', lambda x, y, z, u, f: (dbl2(x), add(x, x)), True, 'yes'),
        ('dbl2 x = dbl x (the later definition unfolds)', lambda x, y, z, u, f: (dbl2(x), dbl(x)), True, 'yes'),
        ('dbl x = dbl2 x (the later definition on the right)', lambda x, y, z, u, f: (dbl(x), dbl2(x)), True, 'yes'),
        # congruence
        ('dbl (idn x) = dbl x', lambda x, y, z, u, f: (dbl(idn(x)), dbl(x)), True, 'yes'),
        ('add (idn x) (plus2 y) = add x (add y two)', lambda x, y, z, u, f: (add(idn(x), plus2(y)), add(x, add(y, two))), True, 'yes'),
        ('S (idn x) = S x', lambda x, y, z, u, f: (S(idn(x)), S(x)), True, 'yes'),
        ('f (idn x) = f x   (a variable head)', lambda x, y, z, u, f: (f(idn(x)), f(x)), True, 'yes'),
        ('add x (idn y) = add x y   (second argument)', lambda x, y, z, u, f: (add(x, idn(y)), add(x, y)), True, 'yes'),
        # beta
        ('(fun n => add n n) x = dbl x', lambda x, y, z, u, f: (w.app(lam_xx(), x), dbl(x)), True, 'yes'),
        ('(fun n => add n) x y = add x y   (beta with a spare argument)', lambda x, y, z, u, f: (w.app(w.lam(Nat, lambda n: add(n)), x, y), add(x, y)), True, 'yes'),
        # a definition applied to more arguments than it has lambdas
        ('adder x y = add x y', lambda x, y, z, u, f: (adder(x, y), add(x, y)), True, 'yes'),
        ('add x y = adder x y', lambda x, y, z, u, f: (add(x, y), adder(x, y)), True, 'yes'),
        # a definition that ignores an argument
        ('konst x y = konst x z', lambda x, y, z, u, f: (konst(x, y), konst(x, z)), True, 'yes'),
        ('konst x y = x', lambda x, y, z, u, f: (konst(x, y), x), True, 'yes'),
        # definitions that take functions
        ('comp idn idn x = x', lambda x, y, z, u, f: (comp(idn, idn, x), x), True, 'yes'),
        ('comp plus2 plus2 x = add (add x two) two', lambda x, y, z, u, f: (comp(plus2, plus2, x), add(add(x, two), two)), True, 'yes'),
        ('twice plus2 x = plus2 (plus2 x)', lambda x, y, z, u, f: (twice(plus2, x), plus2(plus2(x))), True, 'yes'),
        ('twice f x = f (f x)   (f a variable)', lambda x, y, z, u, f: (twice(f, x), f(f(x))), True, 'yes'),
        # binders
        ('(forall n, P0 (idn n)) = (forall n, Le n n)', lambda x, y, z, u, f: (w.pi(Nat, lambda n: P0(idn(n))), w.pi(Nat, lambda n: Le(n, n))), True, 'yes'),
        ('Q0 x = Le x x', lambda x, y, z, u, f: (Q0(x), Le(x, x)), True, 'yes'),
        ('(fun n => dbl n) = (fun n => add n n)', lambda x, y, z, u, f: (w.lam(Nat, lambda n: dbl(n)), w.lam(Nat, lambda n: add(n, n))), True, 'yes'),
        ('(fun n => idn n) = idn   (a lambda against a constant)', lambda x, y, z, u, f: (w.lam(Nat, lambda n: idn(n)), idn), True, 'yes'),
        ('idn = (fun n => idn n)', lambda x, y, z, u, f: (idn, w.lam(Nat, lambda n: idn(n))), True, 'yes'),
        ('(fun n => plus2 (dbl n)) = (fun n => add (add n n) two)', lambda x, y, z, u, f: (w.lam(Nat, lambda n: plus2(dbl(n))), w.lam(Nat, lambda n: add(add(n, n), two))), True, 'yes'),
        # the chain of definitions: 20 steps are fine
        ('e20 x = x   (19 unfoldings)', lambda x, y, z, u, f: (w['e20'](x), x), True, 'yes'),
        ('e20 x = e3 x', lambda x, y, z, u, f: (w['e20'](x), w['e3'](x)), True, 'yes'),
        # different pairs: looked at, not settled, and the evaluator says no
        ('plus2 x = plus2 y', lambda x, y, z, u, f: (plus2(x), plus2(y)), False, 'no'),
        ('dbl x = add x y', lambda x, y, z, u, f: (dbl(x), add(x, y)), False, 'no'),
        ('add x y = add x z   (only the second argument differs)', lambda x, y, z, u, f: (add(x, y), add(x, z)), False, 'no'),
        ('add x (plus2 y) = add x y', lambda x, y, z, u, f: (add(x, plus2(y)), add(x, y)), False, 'no'),
        ('plus2 x = dbl x   (the same arguments, different heads)', lambda x, y, z, u, f: (plus2(x), dbl(x)), False, 'no'),
        ('dbl x = plus2 x', lambda x, y, z, u, f: (dbl(x), plus2(x)), False, 'no'),
        ('add x y = add y x', lambda x, y, z, u, f: (add(x, y), add(y, x)), False, 'no'),
        ('konst x y = y', lambda x, y, z, u, f: (konst(x, y), y), False, 'no'),
        ('S x = Zero', lambda x, y, z, u, f: (S(x), Z0), False, 'no'),
        ('S x = x', lambda x, y, z, u, f: (S(x), x), False, 'no'),
        ('f x = f y', lambda x, y, z, u, f: (f(x), f(y)), False, 'no'),
        ('f x = idn y', lambda x, y, z, u, f: (f(x), idn(y)), False, 'no'),
        ('(forall n, Le n x) = (forall n, Le n y)', lambda x, y, z, u, f: (w.pi(Nat, lambda n: Le(n, x)), w.pi(Nat, lambda n: Le(n, y))), False, 'no'),
        ('(fun n => add n x) = (fun n => add n y)', lambda x, y, z, u, f: (w.lam(Nat, lambda n: add(n, x)), w.lam(Nat, lambda n: add(n, y))), False, 'no'),
        # binder types: two lambdas with different binder types are different even when the bodies agree (each side is well typed; the pair is not of one type, which is exactly what the check must refuse)
        ('(fun n:Nat => Zero) = (fun h:True => Zero)', lambda x, y, z, u, f: (w.lam(Nat, lambda n: Z0), w.lam(w['True'], lambda h: Z0)), False, 'no'),
        ('(fun h:True => Zero) = (fun n:Nat => Zero)', lambda x, y, z, u, f: (w.lam(w['True'], lambda h: Z0), w.lam(Nat, lambda n: Z0)), False, 'no'),
        ('(fun n:Nat => Zero) = (fun n:Nat => Zero)   (the same binder type)', lambda x, y, z, u, f: (w.lam(Nat, lambda n: Z0), w.lam(Nat, lambda n: Z0)), True, 'same'),
        # the same argument list in the two sides of a repeated pair: the second occurrence must not reuse the verdict of the first (memo keyed by the whole pair)
        ('add (plus2 x) (plus2 x) = add (add x two) (add y two)', lambda x, y, z, u, f: (add(plus2(x), plus2(x)), add(add(x, two), add(y, two))), False, 'no'),
        ('add (add x two) (add y two) = add (plus2 x) (plus2 x)', lambda x, y, z, u, f: (add(add(x, two), add(y, two)), add(plus2(x), plus2(x))), False, 'no'),
        ('add (plus2 x) (plus2 y) = add (add x two) (add x two)', lambda x, y, z, u, f: (add(plus2(x), plus2(y)), add(add(x, two), add(x, two))), False, 'no'),
        # domains of a Pi: different binder types are different even when the bodies agree
        ('(Nat -> Le x x) = (True -> Le x x)', lambda x, y, z, u, f: (w.arrow(Nat, Le(x, x)), w.arrow(w['True'], Le(x, x))), False, 'no'),
        ('(True -> Le x x) = (Nat -> Le x x)', lambda x, y, z, u, f: (w.arrow(w['True'], Le(x, x)), w.arrow(Nat, Le(x, x))), False, 'no'),
        # equal, but not by congruence / delta / beta: the quick check gives up and the evaluator says yes
        ('add (S x) y = S (add x y)   (iota)', lambda x, y, z, u, f: (add(S(x), y), S(add(x, y))), True, 'no'),
        ('add two x = S (S x)   (iota)', lambda x, y, z, u, f: (add(two, x), S(S(x))), True, 'no'),
        ('add Zero x = x', lambda x, y, z, u, f: (add(Z0, x), x), True, 'no'),
        ('plus2 (S Zero) = S (S (S Zero))   (a recursive function computes)', lambda x, y, z, u, f: (plus2(S(Z0)), S(S(S(Z0)))), True, 'no'),
        ('dbl two = 4', lambda x, y, z, u, f: (dbl(two), S(S(S(S(Z0))))), True, 'no'),
        ('(fun n => f n) = f   (eta)', lambda x, y, z, u, f: (w.lam(Nat, lambda n: f(n)), f), True, 'no'),
        ('f = (fun n => f n)', lambda x, y, z, u, f: (f, w.lam(Nat, lambda n: f(n))), True, 'no'),
        ('e41-like: a chain of 40 definitions needs more than 32 unfolding rounds', lambda x, y, z, u, f: (w['e40'](x), x), True, 'no'),
        # the DEFINITION DECLARED LATER is unfolded first (it can mention the earlier one, never the reverse): e30 x = e20 x takes 10 rounds that way and about 50 the other way round (e20 down to x, then e30 down to x),
        # and 32 rounds are all the check has
        ('e30 x = e20 x   (the later definition on the left unfolds first)', lambda x, y, z, u, f: (w['e30'](x), w['e20'](x)), True, 'yes'),
        ('e20 x = e30 x   (the later definition on the right unfolds first)', lambda x, y, z, u, f: (w['e20'](x), w['e30'](x)), True, 'yes'),
        ('e30 x = e3 x', lambda x, y, z, u, f: (w['e30'](x), w['e3'](x)), True, 'yes'),
        # the same head with a different number of arguments is never a congruence (each side is well typed, but the two are not of one type: the check must not rely on that)
        ('adder x = adder x y   (one argument more on the right)', lambda x, y, z, u, f: (adder(x), adder(x, y)), False, 'no'),
        ('adder x y = adder x   (one argument more on the left)', lambda x, y, z, u, f: (adder(x, y), adder(x)), False, 'no'),
        ('f = f x   (a variable head)', lambda x, y, z, u, f: (f, f(x)), False, 'no'),
        ('f x = f', lambda x, y, z, u, f: (f(x), f), False, 'no'),
        ('add x = add x y', lambda x, y, z, u, f: (add(x), add(x, y)), False, 'no'),
        ('add x y = add x', lambda x, y, z, u, f: (add(x, y), add(x)), False, 'no'),
    ]


def part_A(w):
    n_yes = n_no = n_same = 0
    with Ctx(w) as vs:
        for name, build_pair, conv, syn in pairs_A(w):
            a, b = build_pair(*vs)
            if not (well_typed(w, a) and well_typed(w, b)): expect(False, 'A: the test pair is ill-typed (a bug in the gate): %s' % name); continue
            off, on, tries, yes = unify_both(w, a, b)
            expect(off[0] == conv, 'A: %s: the evaluator alone says %s, expected %s' % (name, off[0], conv))
            expect(on[0] == off[0], 'A: %s: the verdict changed with the quick check: off %s, on %s' % (name, off[0], on[0]))
            expect(on[1] == off[1] or on[0], 'A: %s: the refusal text changed with the quick check:\n  off: %s\n  on:  %s' % (name, off[1], on[1]))
            expect(on[2] == off[2], 'A: %s: the deferred constraints changed with the quick check: off %d, on %d' % (name, off[2], on[2]))
            want = {'same': 'same', 'yes': 'settled', 'no': 'unsettled'}[syn]
            expect(top(tries, yes) == want, 'A: %s: (looked at, settled) = (%d, %d) is "%s", expected "%s"' % (name, tries, yes, top(tries, yes), want))
            if syn == 'no' and conv: expect((tries, yes) == (1, 0), 'A: %s: an equal pair the evaluator accepts directly: (looked at, settled) = (%d, %d), expected (1, 0)' % (name, tries, yes))
            n_yes += syn == 'yes'; n_no += syn == 'no'; n_same += syn == 'same'
    expect(n_yes >= 30 and n_no >= 20, 'A: the table lost its balance (%d settled, %d not, %d identical)' % (n_yes, n_no, n_same))
    # the option itself
    k = w.k
    k.set_option('syn_conv', False); expect(k.get_option('syn_conv') is False, 'A: syn_conv could not be switched off')
    k.set_option('syn_conv', True); expect(k.get_option('syn_conv') is True, 'A: syn_conv could not be switched on')


# ---------------------------------------------------------------------------------------------------------------------------------------- B: metavariables
def part_B(w):
    Nat, S, add = w['Nat'], w['Succ'], w['add']
    two, idn, plus2, dbl = w['two'], w['idn'], w['plus2'], w['dbl']
    k = w.k
    with Ctx(w) as (x, y, z, u, f):
        scope = len(w.ctx)
        cases = [
            ('plus2 ?m = plus2 x', lambda m: (plus2(m), plus2(x))),
            ('plus2 ?m = add y two', lambda m: (plus2(m), add(y, two))),
            ('?m = plus2 x', lambda m: (m, plus2(x))),
            ('dbl ?m = add x x', lambda m: (dbl(m), add(x, x))),
            ('add ?m two = plus2 ?m   (the same unsolved meta on both sides)', lambda m: (add(m, two), plus2(m))),
            ('idn ?m = x', lambda m: (idn(m), x)),
            ('add (idn ?m) y = add x y', lambda m: (add(idn(m), y), add(x, y))),
            ('plus2 ?m = plus2 ?m2   (two unsolved metas)', None),
            ('S ?m = S x', lambda m: (S(m), S(x))),
            ('add ?m x = add y y   (no solution: x and y differ)', lambda m: (add(m, x), add(y, y))),
        ]
        for name, mk in cases:
            outs = []
            for mode in (False, True):
                k.set_option('syn_conv', mode)
                mid = k.fresh_meta(w._n(Nat), scope); m = w.T(k.mk_meta(mid))
                if mk is None:
                    mid2 = k.fresh_meta(w._n(Nat), scope); m2 = w.T(k.mk_meta(mid2)); a, b = plus2(m), plus2(m2)
                else:
                    a, b = mk(m)
                s0 = k.syn_stats()
                r = k.unify_open(w._n(a), w._n(b), w.ctx)
                s1 = k.syn_stats()
                sol = k.meta_solution_raw(mid) if k.meta_is_solved(mid) else None
                outs.append((r[0], r[2], sol is not None, top(s1[0] - s0[0], s1[1] - s0[1]) if mode else None))
            expect(outs[0][:3] == outs[1][:3], 'B: %s: ON and OFF disagree (verdict, deferred, solved): off %s, on %s' % (name, outs[0][:3], outs[1][:3]))
            expect(outs[1][3] == 'unsettled', 'B: %s: the quick check did not give up on a pair that mentions a metavariable (it is "%s")' % (name, outs[1][3]))
    k.set_option('syn_conv', True)
    # the same, solving: the solution a meta gets is the same (as syntax) with the check on and off
    sols = []
    for mode in (False, True):
        with Ctx(w) as (x, y, z, u, f):
            k.set_option('syn_conv', mode)
            mid = k.fresh_meta(w._n(Nat), len(w.ctx)); m = w.T(k.mk_meta(mid))
            r = k.unify_open(w._n(plus2(m)), w._n(plus2(add(x, y))), w.ctx)
            sols.append((r[0], k.meta_solution_raw(mid)))
    expect(sols[0][0] and sols[0] == sols[1], 'B: the solution of ?m in  plus2 ?m = plus2 (x + y)  differs: off %r, on %r' % (sols[0], sols[1]))
    k.set_option('syn_conv', True)


# ---------------------------------------------------------------------------------------------------------------------------------------- C: limits
def tower(w, f, base, k):
    t = base
    for _ in range(k): t = f(t)
    return t


def part_C(w):
    """The quick check has three limits: steps in all (budget), nesting of comparisons (depth), unfoldings of one pair in a row (rounds). A pair beyond a limit is not settled and goes to the evaluator, which
    must give the same verdict. (The evaluator alone is exponential on a tower of nested stuck matches -- f^n x against f^n (idn x) costs 2^n steps -- so the limits are lowered with set_syn_limits to keep the
    pairs small enough for it; only the round limit is tested at its shipped value.)"""
    Nat, S, add = w['Nat'], w['Succ'], w['add']
    two, idn, plus2 = w['two'], w['idn'], w['plus2']
    k = w.k
    expect(k.get_syn_limits() == (4000, 256, 32), 'C: the shipped limits are %s, expected (4000, 256, 32)' % (k.get_syn_limits(),))
    for bad in ((0, 256, 32), (4000, 0, 32), (4000, 256, 0), (10 ** 8, 256, 32), (4000, 256, 10 ** 6)):
        try: k.set_syn_limits(*bad); expect(False, 'C: set_syn_limits%s was accepted' % (bad,))
        except E.ITPError: pass
    expect(k.get_syn_limits() == (4000, 256, 32), 'C: a refused set_syn_limits changed the limits')
    try:
        with Ctx(w) as (x, y, z, u, f):
            # rounds: the chain e_k x -> e_(k-1) x -> ... -> e1 x -> idn x -> x takes k+1 unfoldings; 32 rounds are allowed in one go
            for name, want in (('e30', (1, 1)), ('e31', (1, 0)), ('e5', (1, 1))):
                off, on, tries, yes = unify_both(w, w[name](x), x)
                expect(off[0] and on[0], 'C: %s x = x: off %s, on %s' % (name, off[0], on[0]))
                expect((tries, yes) == want, 'C: %s x = x: (looked at, settled) = (%d, %d), expected %s' % (name, tries, yes, want))
            k.set_syn_limits(4000, 256, 4)
            for name, want in (('e2', (1, 1)), ('e3', (1, 0))):                 # e2 x -> e1 x -> idn x -> x : round 3 sees x;   e3 x needs round 4
                off, on, tries, yes = unify_both(w, w[name](x), x)
                expect(off[0] and on[0] and (tries, yes) == want, 'C: with 4 rounds, %s x = x: off %s on %s (looked at, settled) = (%d, %d), expected %s' % (name, off[0], on[0], tries, yes, want))
            # depth: a congruence nested 5 levels deep is settled with a depth limit of 8, one 12 levels deep is not (and both are still decided right)
            k.set_syn_limits(4000, 8, 32)
            for n_levels, want in ((5, (1, 1)), (12, (1, 0))):
                a = tower(w, plus2, x, n_levels); b = tower(w, plus2, idn(x), n_levels)
                off, on, tries, yes = unify_both(w, a, b)
                expect(off[0] and on[0], 'C: plus2^%d x = plus2^%d (idn x): off %s, on %s' % (n_levels, n_levels, off[:2], on[:2]))
                expect((tries, yes) == want, 'C: plus2^%d x = plus2^%d (idn x) with depth limit 8: (looked at, settled) = (%d, %d), expected %s' % (n_levels, n_levels, tries, yes, want))
                a = tower(w, plus2, x, n_levels); b = tower(w, plus2, idn(y), n_levels)
                off, on, tries, yes = unify_both(w, a, b)
                expect(not off[0] and not on[0], 'C: plus2^%d x = plus2^%d (idn y) was accepted: off %s, on %s' % (n_levels, n_levels, off[0], on[0]))
            # budget: a balanced tree of leaves  plus2 (S^i v)  against the same with every leaf unfolded.  With a budget of 100 steps 4 leaves are settled, 64 leaves are not (and are still equal)
            k.set_syn_limits(100, 256, 32)
            rnd = random.Random(32)
            vs = [x, y, z, u]
            spec = [(rnd.choice(vs), rnd.randrange(3)) for _ in range(64)]
            def leaf(i, folded, extra=0):
                v, nsucc = spec[i]; t = v
                for _ in range(nsucc + extra): t = S(t)
                return plus2(t) if folded else add(t, two)
            def tree(lo, hi, folded, changed_leaf=-1):
                if hi - lo == 1: return leaf(lo, folded, 1 if lo == changed_leaf else 0)
                mid = (lo + hi) // 2
                return add(tree(lo, mid, folded, changed_leaf), tree(mid, hi, folded, changed_leaf))
            off, on, tries, yes = unify_both(w, tree(0, 4, True), tree(0, 4, False))
            expect(off[0] and on[0] and (tries, yes) == (1, 1), 'C: 4 leaves, budget 100: off %s on %s (looked at, settled) = (%d, %d)' % (off[0], on[0], tries, yes))
            off, on, tries, yes = unify_both(w, tree(0, 64, True), tree(0, 64, False))
            expect(off[0] and on[0] and (tries, yes) == (1, 0), 'C: 64 leaves, budget 100: off %s on %s (looked at, settled) = (%d, %d), expected (1, 0)' % (off[0], on[0], tries, yes))
            # the same pair with a difference in its LAST leaf: the budget runs out before the difference is reached, and the pair must still be refused
            off, on, tries, yes = unify_both(w, tree(0, 64, True), tree(0, 64, False, changed_leaf=63))
            expect(not off[0] and not on[0], 'C: 64 leaves differing in the last: off %s, on %s (a pair beyond the budget was accepted)' % (off[0], on[0]))
            off, on, tries, yes = unify_both(w, tree(0, 64, True), tree(0, 64, False, changed_leaf=0))
            expect(not off[0] and not on[0], 'C: 64 leaves differing in the first: off %s, on %s' % (off[0], on[0]))
            k.set_syn_limits(4000, 256, 32)
            off, on, tries, yes = unify_both(w, tree(0, 64, True), tree(0, 64, False))
            expect(off[0] and on[0] and (tries, yes) == (1, 1), 'C: 64 leaves, shipped budget: off %s on %s (looked at, settled) = (%d, %d)' % (off[0], on[0], tries, yes))
            off, on, tries, yes = unify_both(w, tree(0, 64, True), tree(0, 64, False, changed_leaf=63))
            expect(not off[0] and not on[0], 'C: 64 leaves differing in the last, shipped budget: off %s, on %s' % (off[0], on[0]))
    finally:
        k.set_syn_limits(4000, 256, 32); k.set_option('syn_conv', True)


# ---------------------------------------------------------------------------------------------------------------------------------------- D: random pairs
class Gen:
    """terms of type Nat as a syntax tree, rendered with the definitions folded or unfolded at random"""
    def __init__(self, w, rnd, vars_):
        self.w, self.rnd, self.vars = w, rnd, vars_

    def tree(self, depth):
        r = self.rnd
        if depth == 0 or r.random() < 0.15: return ('var', r.randrange(4)) if r.random() < 0.8 else ('zero',)
        c = r.choice(['succ', 'add', 'plus2', 'dbl', 'dbl2', 'idn', 'konst', 'comp', 'twice', 'beta', 'adder', 'fapp', 'e'])
        if c == 'succ': return ('succ', self.tree(depth - 1))
        if c in ('add', 'konst', 'adder'): return (c, self.tree(depth - 1), self.tree(depth - 1))
        if c in ('plus2', 'dbl', 'dbl2', 'idn', 'fapp'): return (c, self.tree(depth - 1))
        if c == 'e': return ('e', r.randrange(1, 12), self.tree(depth - 1))
        if c in ('comp', 'twice'): return (c, self.fn(depth - 1), self.fn(depth - 1), self.tree(depth - 1)) if c == 'comp' else (c, self.fn(depth - 1), self.tree(depth - 1))
        if c == 'beta': return ('beta', self.tree(depth - 1), self.tree(depth - 1))          # (fun n => add n <t>) <t2>
        raise AssertionError(c)

    def fn(self, depth):
        c = self.rnd.choice(['plus2', 'dbl', 'idn', 'lam', 'f'])
        if c == 'lam': return ('lam', self.tree(max(depth, 0)))
        return (c,)

    def render_fn(self, fn, fold, mut):
        w = self.w; add, two = w['add'], w['two']
        if fn[0] == 'f': return self.vars[4]
        if fn[0] == 'lam':
            body = fn[1]
            return w.lam(w['Nat'], lambda n: add(n, self.render(body, fold, mut)))
        use = fold()
        if fn[0] == 'idn': return w['idn'] if use else w.lam(w['Nat'], lambda n: n)
        if fn[0] == 'plus2': return w['plus2'] if use else w.lam(w['Nat'], lambda n: add(n, two))
        if fn[0] == 'dbl': return w['dbl'] if use else w.lam(w['Nat'], lambda n: add(n, n))
        raise AssertionError(fn)

    def render(self, t, fold, mut):
        """fold: a function (tag) -> bool deciding at each node whether the definition is used or its body written out; mut: a one-element list holding the number of the node to change (or -1)"""
        w, r = self.w, self.rnd
        add, S, Z0, two = w['add'], w['Succ'], w['Zero'], w['two']
        mut[1] += 1
        here = mut[1]
        use = fold()
        tag = t[0]
        R = lambda sub: self.render(sub, fold, mut)
        if tag == 'var':
            v = self.vars[t[1]]
            return S(v) if mut[0] == here else v
        if tag == 'zero': return S(Z0) if mut[0] == here else Z0
        if tag == 'succ': return S(R(t[1]))
        if tag == 'add': return add(R(t[1]), R(t[2]))
        if tag == 'plus2': a = R(t[1]); return w['plus2'](a) if use else add(a, two)
        if tag == 'dbl':
            a = R(t[1])
            return w['dbl'](a) if use else add(a, a)
        if tag == 'dbl2':
            a = R(t[1])
            return w['dbl2'](a) if use else (w['dbl'](a) if fold() else add(a, a))
        if tag == 'idn': a = R(t[1]); return w['idn'](a) if use else a
        if tag == 'konst':
            a = R(t[1]); b = R(t[2])
            return w['konst'](a, b) if use else a
        if tag == 'adder': a = R(t[1]); b = R(t[2]); return w['adder'](a, b) if use else add(a, b)
        if tag == 'fapp': return self.vars[4](R(t[1]))
        if tag == 'e': a = R(t[2]); return w['e%d' % t[1]](a) if use else (w['e%d' % (t[1] - 1)](a) if t[1] > 1 else a)
        if tag == 'comp':
            f, g, a = self.render_fn(t[1], fold, mut), self.render_fn(t[2], fold, mut), R(t[3])
            return w['comp'](f, g, a) if use else w.app(f, w.app(g, a))
        if tag == 'twice':
            f, a = self.render_fn(t[1], fold, mut), R(t[2])
            return w['twice'](f, a) if use else w.app(f, w.app(f, a))
        if tag == 'beta':
            a, b = R(t[1]), R(t[2])
            return w.app(w.lam(w['Nat'], lambda n: add(n, a)), b) if use else add(b, a)
        raise AssertionError(tag)


def part_D(w, n_pairs=700):
    rnd = random.Random(20261012)
    stats = {'equal': [0, 0, 0], 'mut': [0, 0, 0]}           # pairs, quick check settled, accepted
    with Ctx(w) as vs:
        gen = Gen(w, rnd, vs)
        for i in range(n_pairs):
            t = gen.tree(rnd.choice([2, 3, 3, 4]))
            kind = 'equal' if i % 3 else 'mut'
            p1 = rnd.random(); p2 = rnd.random()
            mut = [-1, 0]
            a = gen.render(t, lambda: rnd.random() < p1, mut)
            n_nodes = mut[1]
            mut = [rnd.randrange(1, n_nodes + 1) if kind == 'mut' else -1, 0]
            b = gen.render(t, lambda: rnd.random() < p2, mut)
            if not (well_typed(w, a) and well_typed(w, b)):
                expect(False, 'D: the random pair is ill-typed (a bug in the gate): %s  ~  %s' % (w.show(a), w.show(b))); continue
            off, on, tries, yes = unify_both(w, a, b)
            expect(on[0] == off[0], 'D: ON and OFF disagree on\n   %s\n   %s\n  off %s on %s' % (w.show(a), w.show(b), off[0], on[0]))
            expect(on[2] == off[2], 'D: the deferred constraints differ on\n   %s\n   %s' % (w.show(a), w.show(b)))
            if kind == 'equal': expect(off[0], 'D: a pair that is equal by construction was refused by the evaluator:\n   %s\n   %s\n  %s' % (w.show(a), w.show(b), off[1]))
            s = stats[kind]; s[0] += 1; s[1] += top(tries, yes) == 'settled'; s[2] += bool(off[0])
    print('   D: %d equal-by-construction pairs (%d settled by the quick check), %d changed pairs (%d accepted, %d settled)' % (stats['equal'][0], stats['equal'][1], stats['mut'][0], stats['mut'][2], stats['mut'][1]))
    expect(stats['equal'][1] >= stats['equal'][0] * 0.45, 'D: the quick check settles only %d of %d equal pairs' % (stats['equal'][1], stats['equal'][0]))
    expect(stats['mut'][2] < stats['mut'][0], 'D: every changed pair was accepted (the generator is broken)')
    expect(stats['mut'][1] <= stats['mut'][2], 'D: the quick check settled more changed pairs than are equal (%d > %d)' % (stats['mut'][1], stats['mut'][2]))


# ---------------------------------------------------------------------------------------------------------------------------------------- E: a Pi hidden behind a definition
def part_E(w):
    """A binder hidden behind a definition (`AllLe a` is `forall b, a <= b`) is exposed by unfolding the definition ON SYNTAX when a lambda is checked against it and when an application is inferred. The
    codomain of the exposed Pi mentions the earlier binder / argument, so a wrong substitution gives a wrong type: `hyp b a` must not pass for `forall a b, a <= b`."""
    Nat, Le = w['Nat'], w['Le']
    AllLe, AllLe2 = w['AllLe'], w['AllLe2']
    k = w.k
    for mode in (True, False):
        k.set_option('syn_conv', mode)
        h_ty = w.forall_([Nat], lambda a: AllLe(a))                              # forall a, AllLe a      (the same as forall a b, a <= b)
        w._bind(h_ty, 'hyp')
        try:
            hyp = w.T(k.mk_fvar(len(w.ctx) - 1))
            stmt = w.forall_([Nat], lambda a: AllLe(a))
            good = w.lam(Nat, lambda a: w.lam(Nat, lambda b: hyp(a, b)))
            swapped = w.lam(Nat, lambda a: w.lam(Nat, lambda b: hyp(b, a)))
            doubled = w.lam(Nat, lambda a: w.lam(Nat, lambda b: hyp(b, b)))
            first = w.lam(Nat, lambda a: w.lam(Nat, lambda b: hyp(a, a)))
            for name, t, ok in (('fun a b => hyp a b', good, True), ('fun a b => hyp b a', swapped, False), ('fun a b => hyp b b', doubled, False), ('fun a b => hyp a a', first, False)):
                got, msg = k.check_open(w._n(t), w._n(stmt), w.ctx)
                expect(got == ok, 'E[%s]: %s : forall a, AllLe a  is %s (expected %s): %s' % (mode, name, got, ok, msg[:150]))
            # inference of an application whose function type hides the Pi
            x = w._bind(Nat, 'x')[1]; y = w._bind(Nat, 'y')[1]
            try:
                ty = w.infer(hyp(x, y))
                expect(w.conv(ty, Le(x, y)), 'E[%s]: hyp x y : %s, expected x <= y' % (mode, w.show(ty)))
                expect(not w.conv(ty, Le(y, x)), 'E[%s]: hyp x y was typed as y <= x' % mode)
                expect(not w.conv(ty, Le(y, y)), 'E[%s]: hyp x y was typed as y <= y' % mode)
                expect(not w.conv(ty, Le(x, x)), 'E[%s]: hyp x y was typed as x <= x' % mode)
                z = w._bind(Nat, 'z')[1]
                try:
                    ok_, msg = k.check_open(w._n(hyp(z, x)), w._n(Le(z, x)), w.ctx); expect(ok_, 'E[%s]: hyp z x : z <= x was refused: %s' % (mode, msg[:150]))
                    ok_, msg = k.check_open(w._n(hyp(z, x)), w._n(Le(x, z)), w.ctx); expect(not ok_, 'E[%s]: hyp z x : x <= z was accepted' % mode)
                finally: w._unbind(1)
            finally: w._unbind(2)
        finally: w._unbind(1)
        # a Pi behind two definitions:  AllLe2 a b = Imp2 (a <= b) (b <= a) = ((a <= b) -> (b <= a)),  checked against a lambda over a, b and the premise
        sym_ty = w.forall_([Nat, Nat], lambda a, b: AllLe2(a, b))
        w._bind(sym_ty, 'sym')
        try:
            sym = w.T(k.mk_fvar(len(w.ctx) - 1))
            goal = w.forall_([Nat, Nat], lambda a, b: AllLe2(a, b))
            good = w.fun([Nat, Nat, lambda a, b: Le(a, b)], lambda a, b, h: sym(a, b, h))
            bad = w.fun([Nat, Nat, lambda a, b: Le(b, a)], lambda a, b, h: sym(a, b, h))              # the premise has the wrong type
            for name, t, ok in (('fun a b h => sym a b h', good, True), ('the premise of the wrong type', bad, False)):
                got, msg = k.check_open(w._n(t), w._n(goal), w.ctx)
                expect(got == ok, 'E[%s]: %s against AllLe2: %s (expected %s): %s' % (mode, name, got, ok, msg[:150]))
        finally: w._unbind(1)
    k.set_option('syn_conv', True)


# ---------------------------------------------------------------------------------------------------------------------------------------- F: whole libraries
def build_with(mode, builder):
    """run `builder()` with every World created in it having the quick check set to `mode`"""
    orig = author.World.__init__
    def init(self, *a, **kw):
        orig(self, *a, **kw); self.k.set_option('syn_conv', mode)
    author.World.__init__ = init
    try: return builder()
    finally: author.World.__init__ = orig


def library_dump(w):
    k = w.k; out = []
    for gid in range(k.num_globals()):
        info = k.global_info(gid)
        out.append((info[0], k.dump_raw(info[1]), k.dump_raw(info[2]) if info[2] is not None else None))
    return out


def same_library(name, wa, wb):
    da, db = library_dump(wa), library_dump(wb)
    expect(len(da) == len(db), 'F: %s: %d globals with the check off, %d with it on' % (name, len(da), len(db)))
    bad = [a[0] for a, b in zip(da, db) if a != b]
    expect(not bad, 'F: %s: %d globals differ between the check off and on, e.g. %s' % (name, len(bad), bad[:5]))


def part_F():
    import nat_lib, int_lib
    from tactics import Proof
    t0 = time.time()
    n_off = build_with(False, nat_lib.build_nat); n_on = build_with(True, nat_lib.build_nat)
    same_library('the unary library', n_off, n_on)
    expect(n_off.k.syn_stats() == (0, 0), 'F: the counters moved in the world built with the check off: %s' % (n_off.k.syn_stats(),))
    expect(n_on.k.syn_stats()[1] > 0, 'F: the quick check never settled a pair while the unary library was built')
    z_off = build_with(False, int_lib.build_int); z_on = build_with(True, int_lib.build_int)
    same_library('the integer library', z_off, z_on)
    tries, yes = z_on.k.syn_stats()
    print('   E: the integer library: %d globals, quick check settled %d of %d pairs; built in %.1f s (both ways)' % (z_on.k.num_globals(), yes, tries, time.time() - t0))
    expect(yes > 0.4 * tries, 'F: the quick check settled only %d of %d pairs while the integer library was built' % (yes, tries))
    # a proof by nlinarith: the same term either way, and a final check that does its work in few steps
    terms = []
    for w in (z_off, z_on):
        Z = w['Z']; F = w.forall_; zsub = w['Z.sub']
        sqn = lambda x, y: ('Z.sq_nonneg', (zsub, x, y))
        with w.nota.using(Z):
            stmt = F([Z, Z, Z], lambda a, b, c: a * b + b * c + c * a <= a * a + b * b + c * c)
            p = Proof(w, stmt, 'nl_example'); p.intro('a', 'b', 'c'); p.nlinarith(sqn('a', 'b'), sqn('b', 'c'), sqn('c', 'a'))
            expect(p.done, 'F: the nlinarith example was not finished')
            term = p.term(); sn = w._n(stmt)
        terms.append(w.k.dump_raw(term.n))
        ok, msg = w.k.check_open(term.n, sn, [])
        expect(ok, 'F: the kernel refused the nlinarith proof (%s syntax check): %s' % ('with' if w is z_on else 'without', msg[:200]))
        if w is z_on:
            used = w.k.fuel_used()
            print('   E: the final check of the nlinarith proof took %d steps' % used)
            expect(used <= 400000, 'F: the final check of the nlinarith proof took %d steps (the bound is 400000; about 88000 expected, 2.0 million without the quick check)' % used)
    expect(terms[0] == terms[1], 'F: the nlinarith proof term differs between the check off and on')


def main():
    w = build()
    part_A(w); part_B(w); part_C(w); part_D(w); part_E(w)
    if '--quick' not in sys.argv: part_F()
    print('%d checks' % CHECKS[0])
    if FAIL:
        print('S32 SYNCONV GATE: FAIL (%d)' % len(FAIL))
        for f in FAIL[:25]: print('   ', f)
        sys.exit(1)
    print('S32 SYNCONV GATE: PASS')


if __name__ == '__main__':
    main()
