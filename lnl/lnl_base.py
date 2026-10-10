"""lnl_base.py -- the logical base of the library, written with the authoring layer: the natural numbers as a type, the connectives, equality, and the first lemmas about them.

The world is the Prop/Type world (kernel options prop_elim and parameters on): propositions live in Type0, data in Type1. Every declaration goes through the kernel's checked path and every
theorem is proved by a tactic script, extracted and re-checked (World.prove), so nothing here is trusted."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from author import World, AuthorError, Term, Notation


def thm(w, name, doms, body):
    """decorator: `@thm(w, 'name', [binder types], lambda vars...: proposition)` over a function of the Proof that proves it"""
    def deco(script):
        w.prove(name, w.forall_(doms, body), script); return script
    return deco


def declare_types(w):
    TYPE, PROP = w.TYPE, w.PROP
    w.inductive('Nat', TYPE, [('Zero', []), ('Succ', [lambda *_: w['Nat']])])
    w.inductive('True', PROP, [('tt', [])])
    w.inductive('False', PROP, [])
    w.inductive('Eq', w.forall_([TYPE, lambda A: A, lambda A, a: A], lambda A, a, b: PROP), [('refl', [TYPE, lambda A: A], lambda A, a: [A, a, a])])
    w.inductive('And', w.forall_([PROP, PROP], lambda a, b: PROP), [('conj', [PROP, PROP, lambda a, b: a, lambda a, b, x: b], lambda a, b, x, y: [a, b])])
    w.inductive('Or', w.forall_([PROP, PROP], lambda a, b: PROP),
                [('inl', [PROP, PROP, lambda a, b: a], lambda a, b, x: [a, b]), ('inr', [PROP, PROP, lambda a, b: b], lambda a, b, x: [a, b])])
    w.inductive('Ex', w.forall_([TYPE, lambda A: w.arrow(A, PROP)], lambda A, P: PROP),
                [('ex_intro', [TYPE, lambda A: w.arrow(A, PROP), lambda A, P: A, lambda A, P, a: P(a)], lambda A, P, a, h: [A, P])])
    Nat = w['Nat']
    w.inductive('Le', w.forall_([Nat, Nat], lambda n, m: PROP),
                [('le_n', [Nat], lambda n: [n, n]), ('le_S', [Nat, Nat, lambda n, m: w['Le'](n, m)], lambda n, m, h: [n, w['Succ'](m)])])
    w.define('Not', w.arrow(PROP, PROP), w.lam(PROP, lambda P: w.arrow(P, w['False'])))
    w.define('Iff', w.forall_([PROP, PROP], lambda a, b: PROP), w.fun([PROP, PROP], lambda a, b: w['And'](w.arrow(a, b), w.arrow(b, a))))
    w.infix.update({'add': ('+', 6), 'mul': ('*', 7), 'Le': ('<=', 4), 'And': ('/\\', 3), 'Or': ('\\/', 2), 'Iff': ('<->', 1)})


def declare_add(w):
    Nat, S = w['Nat'], w['Succ']
    w.recursive('add', w.forall_([Nat, Nat], lambda n, m: Nat),
                lambda add: w.fun([Nat, Nat], lambda n, m: w.match(n, lambda _: Nat, [m, lambda n1: S(add(n1, m))])))


def equality_lemmas(w):
    TYPE, PROP, Nat = w.TYPE, w.PROP, w['Nat']
    Eq = w.Eq

    @thm(w, 'transport', [TYPE, lambda A: w.arrow(A, PROP), lambda A, P: A, lambda A, P, x: A, lambda A, P, x, y: Eq(x, y), lambda A, P, x, y, h: P(x)], lambda A, P, x, y, h, px: P(y))
    def _(p): p.intro('A', 'P', 'x', 'y', 'h', 'px'); p.cases('h'); p.exact('px')

    @thm(w, 'symm', [TYPE, lambda A: A, lambda A, x: A, lambda A, x, y: Eq(x, y)], lambda A, x, y, h: Eq(y, x))
    def _(p): p.intro('A', 'x', 'y', 'h'); p.cases('h'); p.apply('refl')

    @thm(w, 'trans', [TYPE, lambda A: A, lambda A, x: A, lambda A, x, y: A, lambda A, x, y, z: Eq(x, y), lambda A, x, y, z, h1: Eq(y, z)], lambda A, x, y, z, h1, h2: Eq(x, z))
    def _(p): p.intro('A', 'x', 'y', 'z', 'h1', 'h2'); p.cases('h2'); p.exact('h1')

    @thm(w, 'f_equal', [TYPE, TYPE, lambda A, B: w.arrow(A, B), lambda A, B, f: A, lambda A, B, f, x: A, lambda A, B, f, x, y: Eq(x, y)], lambda A, B, f, x, y, h: Eq(f(x), f(y)))
    def _(p): p.intro('A', 'B', 'f', 'x', 'y', 'h'); p.cases('h'); p.refl()

    @thm(w, 'cong_succ', [Nat, Nat, lambda n, m: Eq(n, m)], lambda n, m, h: Eq(w['Succ'](n), w['Succ'](m)))
    def _(p): p.intro('n', 'm', 'h'); p.cases('h'); p.refl()


def logic_lemmas(w):
    PROP, TYPE = w.PROP, w.TYPE
    T, F = w['True'], w['False']

    @thm(w, 'true_intro', [], lambda: T)
    def _(p): p.apply('tt')

    @thm(w, 'false_elim', [PROP, F], lambda P, h: P)
    def _(p): p.intro('P', 'h'); p.cases('h')

    @thm(w, 'and_left', [PROP, PROP, lambda a, b: a & b], lambda a, b, h: a)
    def _(p): p.intro('a', 'b', 'h'); p.cases('h', names=['x', 'y']); p.exact('x')

    @thm(w, 'and_right', [PROP, PROP, lambda a, b: a & b], lambda a, b, h: b)
    def _(p): p.intro('a', 'b', 'h'); p.cases('h', names=['x', 'y']); p.exact('y')

    @thm(w, 'and_comm', [PROP, PROP, lambda a, b: a & b], lambda a, b, h: b & a)
    def _(p): p.intro('a', 'b', 'h'); p.cases('h', names=['x', 'y']); p.apply('conj'); p.exact('y'); p.exact('x')

    @thm(w, 'or_comm', [PROP, PROP, lambda a, b: a | b], lambda a, b, h: b | a)
    def _(p):
        p.intro('a', 'b', 'h'); p.cases('h', names=['x', 'x'])
        p.apply('inr'); p.exact('x')
        p.apply('inl'); p.exact('x')

    @thm(w, 'not_not_intro', [PROP, lambda P: P, lambda P, h: ~P], lambda P, h, n: F)
    def _(p): p.intro('P', 'h', 'n'); p.apply('n'); p.exact('h')

    @thm(w, 'absurd', [PROP, PROP, lambda a, b: a, lambda a, b, h: ~a], lambda a, b, h, n: b)
    def _(p): p.intro('a', 'b', 'h', 'n'); p.apply('false_elim'); p.apply('n'); p.exact('h')

    @thm(w, 'iff_refl', [PROP], lambda P: w.Iff(P, P))
    def _(p):
        p.intro('P'); p.apply('conj'); p.intro('x'); p.exact('x'); p.intro('x'); p.exact('x')

    @thm(w, 'iff_symm', [PROP, PROP, lambda a, b: w.Iff(a, b)], lambda a, b, h: w.Iff(b, a))
    def _(p): p.intro('a', 'b', 'h'); p.cases('h', names=['f', 'g']); p.apply('conj'); p.exact('g'); p.exact('f')

    @thm(w, 'cut', [PROP, PROP, lambda P, Q: P, lambda P, Q, p: w.arrow(P, Q)], lambda P, Q, p, f: Q)
    def _(p): p.intro('P', 'Q', 'x', 'f'); p.apply('f'); p.exact('x')

    @thm(w, 'iff_mp', [PROP, PROP, lambda a, b: w.Iff(a, b)], lambda a, b, h: w.arrow(a, b))
    def _(p): p.intro('a', 'b', 'h'); p.cases('h', names=['f', 'g']); p.exact('f')

    @thm(w, 'iff_mpr', [PROP, PROP, lambda a, b: w.Iff(a, b)], lambda a, b, h: w.arrow(b, a))
    def _(p): p.intro('a', 'b', 'h'); p.cases('h', names=['f', 'g']); p.exact('g')

    @thm(w, 'ex_elim', [TYPE, lambda A: w.arrow(A, PROP), lambda A, P: PROP, lambda A, P, Q: w['Ex'](A, P)], lambda A, P, Q, h: w.arrow(w.forall_([A, lambda a: P(a)], lambda a, pa: Q), Q))
    def _(p): p.intro('A', 'P', 'Q', 'h', 'f'); p.cases('h', names=['a', 'pa']); p.apply('f'); p.exact('a'); p.exact('pa')


def build_base():
    w = World()
    declare_types(w); declare_add(w); equality_lemmas(w); logic_lemmas(w)
    return w


if __name__ == '__main__':
    import time
    t0 = time.time(); w = build_base()
    print('base: %d globals in %.2f s' % (w.k.num_globals(), time.time() - t0))
