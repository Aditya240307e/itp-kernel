"""quot_lib.py -- quotients in the authoring world: the five quotient constants, function extensionality DERIVED from `Quot.sound`, pairs, and the lifting of two-argument functions.

Nothing here is trusted. `declare_quot` is the kernel's own entry point (it builds the five types itself and checks that the equality family it is given is the right one); `funext`, `Quot.lift2` and its
computation rule are theorems and definitions admitted through the kernel's checked path like everything else. The only things this adds to the trusted base are the five quotient constants, exactly the ones
audited by the S16 gate -- no axiom at all.

    funext   : forall (A : Type) (B : A -> Type) (f g : forall x, B x), (forall x, f x = g x) -> f = g        (from Quot.sound and the eta rule, as in S16)
    Prod     : Type -> Type -> Type, with `pair`, and the projections `fst`, `snd` (which compute on `pair`)
    Quot.lift2 A r B f hr h : Quot A r -> Quot A r -> B     for  f : A -> A -> B,  hr : forall a, r a a,  h : forall a a' b b', r a a' -> r b b' -> f a b = f a' b'
    Quot.lift2_mk           : Quot.lift2 A r B f hr h (Quot.mk A r a) (Quot.mk A r b) = f a b            (by computation)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys
from author import AuthorError
from lnl_base import thm


def declare_quotients(w):
    """register Quot, Quot.mk, Quot.lift, Quot.ind, Quot.sound (the kernel builds their types from the world's equality family) and derive `funext`"""
    handle = w.k.declare_quot(w.gid['Eq'])
    for name, gid in zip(('Quot', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound'), handle): w._register(name, gid)
    TYPE = w.TYPE; Eq = w.Eq
    Quot, MK, LIFT, SOUND = w['Quot'], w['Quot.mk'], w['Quot.lift'], w['Quot.sound']
    stmt = w.forall_([TYPE, lambda A: w.arrow(A, TYPE), lambda A, B: w.pi(A, lambda x: B(x)), lambda A, B, f: w.pi(A, lambda x: B(x)),
                      lambda A, B, f, g: w.forall_([A], lambda x: Eq(f(x), g(x)))],
                     lambda A, B, f, g, h: Eq(f, g))

    def proof(A, B, f, g, h):
        Fn = w.pi(A, lambda x: B(x))
        rel = w.fun([Fn, Fn], lambda u, v: w.forall_([A], lambda x: Eq(u(x), v(x))))
        QF = Quot(Fn, rel)
        ext = w.fun([QF, A], lambda z, x: LIFT(Fn, rel, B(x), w.lam(Fn, lambda u: u(x)), w.fun([Fn, Fn, lambda u, v: rel(u, v)], lambda u, v, pp: pp(x)), z))       # z |-> (x |-> the value of z at x)
        mkf = MK(Fn, rel, f)
        motive = w.lam(QF, lambda z: Eq(ext(mkf), ext(z)))
        return w['transport'](QF, motive, mkf, MK(Fn, rel, g), SOUND(Fn, rel, f, g, h), w['refl'](Fn, ext(mkf)))
    w.define('funext', stmt, w.fun([TYPE, lambda A: w.arrow(A, TYPE), lambda A, B: w.pi(A, lambda x: B(x)), lambda A, B, f: w.pi(A, lambda x: B(x)),
                                    lambda A, B, f, g: w.forall_([A], lambda x: Eq(f(x), g(x)))], proof))


def declare_prod(w):
    """pairs, and the projections (declared as recursive functions without a recursive call, so that the display normal form computes `fst (pair a b)`)"""
    TYPE = w.TYPE
    w.inductive('Prod', w.forall_([TYPE, TYPE], lambda A, B: TYPE), [('pair', [TYPE, TYPE, lambda A, B: A, lambda A, B, a: B], lambda A, B, a, b: [A, B])])
    Prod = w['Prod']
    w.recursive('fst', w.forall_([TYPE, TYPE, lambda A, B: Prod(A, B)], lambda A, B, p: A),
                lambda f: w.fun([TYPE, TYPE, lambda A, B: Prod(A, B)], lambda A, B, p: w.match(p, lambda q: A, [lambda a, b: a])), on=2)
    w.recursive('snd', w.forall_([TYPE, TYPE, lambda A, B: Prod(A, B)], lambda A, B, p: B),
                lambda f: w.fun([TYPE, TYPE, lambda A, B: Prod(A, B)], lambda A, B, p: w.match(p, lambda q: B, [lambda a, b: b])), on=2)


def quot_lift2(w):
    """a function of two arguments out of a quotient by ONE relation (reflexive, so that the inner function is well defined for every first argument), and its computation rule"""
    TYPE, PROP = w.TYPE, w.PROP; Eq = w.Eq
    Quot, MK, LIFT, IND = w['Quot'], w['Quot.mk'], w['Quot.lift'], w['Quot.ind']; funext = w['funext']
    rel_ty = lambda A: w.arrow(A, w.arrow(A, PROP))
    doms = [TYPE, lambda A: rel_ty(A), lambda A, r: TYPE, lambda A, r, B: w.arrow(A, w.arrow(A, B)),
            lambda A, r, B, f: w.pi(A, lambda a: r(a, a)),
            lambda A, r, B, f, hr: w.forall_([A, A, A, A, lambda a, a2, b, b2: r(a, a2), lambda a, a2, b, b2, h1: r(b, b2)], lambda a, a2, b, b2, h1, h2: Eq(f(a, b), f(a2, b2)))]
    stmt_ty = w.forall_(doms, lambda A, r, B, f, hr, h: w.arrow(Quot(A, r), w.arrow(Quot(A, r), B)))

    def body(A, r, B, f, hr, h):
        Q = Quot(A, r)
        inner = lambda a: LIFT(A, r, B, f(a), w.fun([A, A, lambda b, b2: r(b, b2)], lambda b, b2, hb: h(a, a, b, b2, hr(a), hb)))          # the function of the second argument, for a fixed first argument
        outer = w.fun([A, A, lambda a, a2: r(a, a2)], lambda a, a2, ha: funext(Q, w.lam(Q, lambda _q: B), inner(a), inner(a2),
                                                                                  w.lam(Q, lambda q: IND(A, r, w.lam(Q, lambda q1: Eq(inner(a)(q1), inner(a2)(q1))),
                                                                                                          w.lam(A, lambda b: h(a, a2, b, b, ha, hr(b))), q))))
        return LIFT(A, r, w.arrow(Q, B), w.lam(A, inner), outer)
    w.define('Quot.lift2', stmt_ty, w.fun(doms, body))

    @thm(w, 'Quot.lift2_mk', doms + [lambda A, r, B, f, hr, h: A, lambda A, r, B, f, hr, h, a: A],
         lambda A, r, B, f, hr, h, a, b: Eq(w['Quot.lift2'](A, r, B, f, hr, h, MK(A, r, a), MK(A, r, b)), f(a, b)))
    def _(p):
        p.intro('A', 'r', 'B', 'f', 'hr', 'h', 'a', 'b'); p.refl()
