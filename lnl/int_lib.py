"""int_lib.py -- the integers as a QUOTIENT of pairs of binary naturals, with every law proved (no axiom): the pair (a, b) stands for a - b, and  (a, b) ~ (c, d)  iff  a + d = c + b.

    Z            := Quot (Prod N N) Zrel                        Z.mk a b := the class of (a, b)
    Z.add, Z.mul, Z.neg  by `Quot.lift` / `Quot.lift2`, computing on `Z.mk`:  mk a b + mk c d = mk (a + c) (b + d)    (by `refl`)
    laws         by quotient induction (`Z.induction_on`) down to equations of N, which `ring` proves
    effectiveness  mk a b = mk c d  ->  a + d = c + b    (`Z.mk_inj`): from a Bool-valued test lifted out of the quotient -- Quot.lift into Prop would need propext, which this library does not assume

Nothing here is trusted: every definition and theorem goes through the kernel's checked path.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from author import World, AuthorError, Term, Notation, DispatchNotation
import lnl_base, quot_lib, ring_lib
from lnl_base import thm



def calc_scope(p, names, steps_fn):
    """p.calc(*steps_fn(*vars)) where vars are the hypotheses called `names` (terms are built with the notation context of the first goal)"""
    with p.focus():
        vars_ = [p.h(n) for n in names]
        steps = steps_fn(*vars_)
    return p.calc(*steps)


def _no_instance(e): return 'no instance of the pattern' in str(e)


def rw_all(p, lemmas):
    """rewrite the first goal by the given lemmas, each as long as it has an instance, until none has"""
    changed = True
    while changed and not p.done:
        changed = False
        for lem in lemmas:
            try: p.rw(lem, close=False)
            except AuthorError as e:
                if not _no_instance(e): raise
                continue
            changed = True
    return p


def rw_all_at(p, h, lemmas):
    """rewrite the hypothesis `h` by the given lemmas, each as long as it has an instance, until none has"""
    changed = True
    while changed:
        changed = False
        for lem in lemmas:
            try: p.rw_at(h, lem)
            except AuthorError as e:
                if not _no_instance(e): raise
                continue
            changed = True
    return p


class ZNotation(Notation):
    """operators and numerals of Z: a numeral is `Z.of_N` of a numeral of N, a negative one its negation"""
    def __init__(self, w): self.w = w
    def _t(self, x): return self.lit(x) if isinstance(x, int) and not isinstance(x, bool) else x
    def add(self, a, b): return self.w['Z.add'](self._t(a), self._t(b))
    def mul(self, a, b): return self.w['Z.mul'](self._t(a), self._t(b))
    def sub(self, a, b): return self.w['Z.sub'](self._t(a), self._t(b))
    def neg(self, a): return self.w['Z.neg'](self._t(a))
    def pow(self, a, e):
        if isinstance(e, int) and not isinstance(e, bool): e = self.w.nota.by_type[self.w['N'].n].lit(e)
        return self.w['Z.pow'](self._t(a), e)
    def le(self, a, b): return self.w['Z.le'](self._t(a), self._t(b))
    def lt(self, a, b): return self.w['Z.lt'](self._t(a), self._t(b))
    def lit(self, n):
        t = self.w['Z.of_N'](self.w.nota.by_type[self.w['N'].n].lit(abs(n)))
        return t if n >= 0 else self.w['Z.neg'](t)


# ---------------------------------------------------------------------------------------------------------------------------------------------- Bool and equality tests on N
def bool_lib(w):
    """Bool, and the Bool-valued equality test on Pos and N with its specification (equal iff the test says true)"""
    TYPE, PROP = w.TYPE, w.PROP; Eq = w.Eq
    w.inductive('Bool', TYPE, [('Bool.false', []), ('Bool.true', [])])
    Bool = w['Bool']; BF, BT = w['Bool.false'], w['Bool.true']
    Pos, N = w['Pos'], w['N']; XH, XO, XI, N0, Npos = w['XH'], w['XO'], w['XI'], w['N0'], w['Npos']
    F = w['False']
    w.recursive('Pos.beq', w.forall_([Pos, Pos], lambda x, y: Bool),
                lambda beq: w.fun([Pos, Pos], lambda x, y: w.match(x, lambda _: Bool, [
                    w.match(y, lambda _: Bool, [BT, lambda q: BF, lambda q: BF]),
                    lambda p: w.match(y, lambda _: Bool, [BF, lambda q: beq(p, q), lambda q: BF]),
                    lambda p: w.match(y, lambda _: Bool, [BF, lambda q: BF, lambda q: beq(p, q)])])))
    pbeq = w['Pos.beq']
    w.recursive('N.beq', w.forall_([N, N], lambda x, y: Bool),
                lambda f: w.fun([N, N], lambda x, y: w.match(x, lambda _: Bool, [
                    w.match(y, lambda _: Bool, [BT, lambda q: BF]),
                    lambda p: w.match(y, lambda _: Bool, [BF, lambda q: pbeq(p, q)])])))
    nbeq = w['N.beq']

    def absurd(p, h):
        """the hypothesis `h` computes to  false = true : close the goal"""
        with p.focus(): stmt = Eq(BF, BT)
        p.have('_bad', stmt, lambda q: q.exact(h)); p.cases('_bad')

    @thm(w, 'Bool.false_ne_true', [Eq(BF, BT)], lambda h: F)
    def _(p): p.intro('h'); p.cases('h')

    @thm(w, 'Pos.beq_refl', [Pos], lambda x: Eq(pbeq(x, x), BT))
    def _(p):
        p.intro('x'); p.induct('x', names=['q', 'ih', 'q', 'ih'])
        p.refl()
        p.exact('ih')
        p.exact('ih')

    @thm(w, 'Pos.beq_eq', [Pos, Pos, lambda x, y: Eq(pbeq(x, y), BT)], lambda x, y, h: Eq(x, y))
    def _(p):
        p.intro('x'); p.induct('x', names=['q', 'ih', 'q', 'ih'])
        p.intro('y', 'h'); p.cases('y', names=['r', 'r'])
        p.refl()
        absurd(p, 'h')
        absurd(p, 'h')
        p.intro('y', 'h'); p.cases('y', names=['r', 'r'])
        absurd(p, 'h')
        p.apply('f_equal', Pos, Pos, XO); p.apply('ih'); p.exact('h')
        absurd(p, 'h')
        p.intro('y', 'h'); p.cases('y', names=['r', 'r'])
        absurd(p, 'h')
        absurd(p, 'h')
        p.apply('f_equal', Pos, Pos, XI); p.apply('ih'); p.exact('h')

    @thm(w, 'N.beq_refl', [N], lambda x: Eq(nbeq(x, x), BT))
    def _(p):
        p.intro('x'); p.cases('x', names=['q'])
        p.refl()
        p.apply('Pos.beq_refl')

    @thm(w, 'N.beq_eq', [N, N, lambda x, y: Eq(nbeq(x, y), BT)], lambda x, y, h: Eq(x, y))
    def _(p):
        p.intro('x', 'y', 'h'); p.cases('x', names=['q'])
        p.cases('y', names=['r'])
        p.refl()
        absurd(p, 'h')
        p.cases('y', names=['r'])
        absurd(p, 'h')
        p.apply('f_equal', Pos, N, Npos); p.apply('Pos.beq_eq'); p.exact('h')

    @thm(w, 'N.eq_beq', [N, N, lambda x, y: Eq(x, y)], lambda x, y, h: Eq(nbeq(x, y), BT))
    def _(p): p.intro('x', 'y', 'h'); p.cases('h'); p.apply('N.beq_refl')

    @thm(w, 'Bool.ext', [Bool, Bool, lambda a, b: w.arrow(Eq(a, BT), Eq(b, BT)), lambda a, b, h1: w.arrow(Eq(b, BT), Eq(a, BT))], lambda a, b, h1, h2: Eq(a, b))
    def _(p):
        p.intro('a', 'b', 'h1', 'h2'); p.cases('a'); p.cases('b')
        p.refl()
        p.exfalso(); p.apply('Bool.false_ne_true'); p.apply('h2'); p.refl()
        p.cases('b')
        p.exfalso(); p.apply('Bool.false_ne_true'); p.apply('h1'); p.refl()
        p.refl()


# ---------------------------------------------------------------------------------------------------------------------------------------------- the quotient
def int_core(w):
    """Z, Z.mk, soundness (`Z.mk_eq`), induction (`Z.induction_on`), effectiveness (`Z.mk_inj`)"""
    TYPE, PROP = w.TYPE, w.PROP; Eq = w.Eq
    N, Prod = w['N'], w['Prod']; pair, fst, snd = w['pair'], w['fst'], w['snd']
    Quot, MK, LIFT, SOUND, IND = w['Quot'], w['Quot.mk'], w['Quot.lift'], w['Quot.sound'], w['Quot.ind']
    Bool, BT, BF = w['Bool'], w['Bool.true'], w['Bool.false']
    nbeq = w['N.beq']
    NN = Prod(N, N)
    F = lambda p: fst(N, N, p)
    S = lambda p: snd(N, N, p)
    PR = lambda a, b: pair(N, N, a, b)
    with w.nota.using(N):
        Zrel = w.define('Zrel', w.arrow(NN, w.arrow(NN, PROP)), w.fun([NN, NN], lambda p, q: Eq(F(p) + S(q), F(q) + S(p))))
    Z = w.define('Z', TYPE, Quot(NN, Zrel))
    Zmk = w.define('Z.mk', w.forall_([N, N], lambda a, b: Z), w.fun([N, N], lambda a, b: MK(NN, Zrel, PR(a, b))))

    @thm(w, 'Z.mk_eq', [N, N, N, N, lambda a, b, c, d: Eq(a + d, c + b)], lambda a, b, c, d, h: Eq(Zmk(a, b), Zmk(c, d)))
    def _(p):
        p.intro('a', 'b', 'c', 'd', 'h')
        p.exact(lambda p: SOUND(NN, Zrel, PR(p.h('a'), p.h('b')), PR(p.h('c'), p.h('d')), p.h('h')))

    @thm(w, 'Z.induction_on', [w.arrow(Z, PROP), Z, lambda P, z: w.forall_([N, N], lambda a, b: P(Zmk(a, b)))], lambda P, z, h: P(z))
    def _(p):
        p.intro('P', 'z', 'h')
        def term(p):
            P, z, h = p.h('P'), p.h('z'), p.h('h')
            h2 = w.lam(NN, lambda q: w.match(q, lambda q2: P(MK(NN, Zrel, q2)), [lambda a, b: h(a, b)]))
            return IND(NN, Zrel, w.lam(Z, lambda q: P(q)), h2, z)
        p.exact(term)

    # ---- effectiveness: a Bool-valued test lifted out of the quotient
    with w.nota.using(N):
        @thm(w, 'Z.shift', [N] * 6 + [lambda a, b, a1, b1, c, d: Eq(a + b1, a1 + b), lambda a, b, a1, b1, c, d, h1: Eq(a + d, c + b)], lambda a, b, a1, b1, c, d, h1, h2: Eq(a1 + d, c + b1))
        def _(p):
            p.intro('a', 'b', 'a1', 'b1', 'c', 'd', 'h1', 'h2')
            with p.focus():
                a, b, a1, b1, c, d = (p.h(x) for x in ('a', 'b', 'a1', 'b1', 'c', 'd'))
                l, r = a1 + d, c + b1
                s0, s1, s2, s3, s4 = (a1 + d) + b, (a1 + b) + d, (a + b1) + d, (a + d) + b1, (c + b) + b1
                s5 = (c + b1) + b
            p.apply('N.add_right_cancel', l, r, b)
            p.calc(s0, ('ring', s1), (lambda q: q.rw('h1', rev=True), s2), ('ring', s3), (lambda q: q.rw('h2'), s4), ('ring', s5))

        @thm(w, 'Z.test_wd', [N, N, NN, NN, lambda c, d, p, q: Zrel(p, q)],
             lambda c, d, p, q, h: Eq(nbeq(F(p) + d, c + S(p)), nbeq(F(q) + d, c + S(q))))
        def _(p):
            p.intro('c', 'd', 'p', 'q', 'h'); p.cases('p', names=['a', 'b']); p.cases('q', names=['a1', 'b1'])
            with p.focus():
                a, b, a1, b1 = (p.h(x) for x in ('a', 'b', 'a1', 'b1'))
                e_stmt = Eq(a + b1, a1 + b)
            p.have('e', e_stmt, lambda q: q.exact('h'))
            p.apply('Bool.ext')
            p.intro('t'); p.apply('N.eq_beq')
            p.exact(lambda q: w['Z.shift'](q.h('a'), q.h('b'), q.h('a1'), q.h('b1'), q.h('c'), q.h('d'), q.h('e'), w['N.beq_eq'](q.h('a') + q.h('d'), q.h('c') + q.h('b'), q.h('t'))))
            p.intro('t'); p.apply('N.eq_beq')
            p.exact(lambda q: w['Z.shift'](q.h('a1'), q.h('b1'), q.h('a'), q.h('b'), q.h('c'), q.h('d'), w['symm'](N, q.h('a') + q.h('b1'), q.h('a1') + q.h('b'), q.h('e')),
                                           w['N.beq_eq'](q.h('a1') + q.h('d'), q.h('c') + q.h('b1'), q.h('t'))))

        Ztest = w.define('Z.test', w.forall_([N, N], lambda c, d: w.arrow(Z, Bool)),
                         w.fun([N, N], lambda c, d: LIFT(NN, Zrel, Bool, w.lam(NN, lambda p: nbeq(F(p) + d, c + S(p))),
                                                           w.fun([NN, NN, lambda p, q: Zrel(p, q)], lambda p, q, h: w['Z.test_wd'](c, d, p, q, h)))))

        @thm(w, 'Z.mk_inj', [N, N, N, N, lambda a, b, c, d: Eq(Zmk(a, b), Zmk(c, d))], lambda a, b, c, d, h: Eq(a + d, c + b))
        def _(p):
            p.intro('a', 'b', 'c', 'd', 'h'); p.apply('N.beq_eq')
            def term(q):
                a, b, c, d, h = (q.h(x) for x in ('a', 'b', 'c', 'd', 'h'))
                e = w['f_equal'](Z, Bool, Ztest(c, d), Zmk(a, b), Zmk(c, d), h)
                return w['trans'](Bool, Ztest(c, d, Zmk(a, b)), Ztest(c, d, Zmk(c, d)), BT, e, w['N.beq_refl'](c + d))
            p.exact(term)


def int_ops(w):
    """addition, negation, multiplication of Z, their computation rules, and the laws of a commutative ring"""
    TYPE, PROP = w.TYPE, w.PROP; Eq = w.Eq
    N, Prod, Z = w['N'], w['Prod'], w['Z']; pair, fst, snd = w['pair'], w['fst'], w['snd']
    Zrel, Zmk = w['Zrel'], w['Z.mk']
    Quot, LIFT, LIFT2 = w['Quot'], w['Quot.lift'], w['Quot.lift2']
    NN = Prod(N, N)
    F = lambda p: fst(N, N, p)
    S = lambda p: snd(N, N, p)
    N0, one = w['N0'], w['Npos'](w['XH'])
    mkeq = w['Z.mk_eq']

    with w.nota.using(N):
        # ---- well-definedness
        @thm(w, 'Z.neg_wd', [NN, NN, lambda p, q: Zrel(p, q)], lambda p, q, h: Eq(Zmk(S(p), F(p)), Zmk(S(q), F(q))))
        def _(p):
            p.intro('p', 'q', 'h'); p.cases('p', names=['a', 'b']); p.cases('q', names=['a1', 'b1'])
            with p.focus():
                a, b, a1, b1 = (p.h(x) for x in ('a', 'b', 'a1', 'b1')); e_stmt = Eq(a + b1, a1 + b)
            p.have('e', e_stmt, lambda q: q.exact('h'))
            p.apply('Z.mk_eq')
            calc_scope(p, ['a', 'b', 'a1', 'b1'], lambda a, b, a1, b1: (b + a1, ('ring', a1 + b), ('e', a + b1)  , ('ring', b1 + a)))
            # the middle step is the hypothesis read backwards: a1 + b = a + b1

        @thm(w, 'Z.add_wd', [NN, NN, NN, NN, lambda p, p1, q, q1: Zrel(p, p1), lambda p, p1, q, q1, h1: Zrel(q, q1)],
             lambda p, p1, q, q1, h1, h2: Eq(Zmk(F(p) + F(q), S(p) + S(q)), Zmk(F(p1) + F(q1), S(p1) + S(q1))))
        def _(p):
            p.intro('p', 'p1', 'q', 'q1', 'h1', 'h2')
            p.cases('p', names=['a', 'b']); p.cases('p1', names=['a1', 'b1']); p.cases('q', names=['c', 'd']); p.cases('q1', names=['c1', 'd1'])
            with p.focus():
                a, b, a1, b1, c, d, c1, d1 = (p.h(x) for x in ('a', 'b', 'a1', 'b1', 'c', 'd', 'c1', 'd1'))
                e1_stmt, e2_stmt = Eq(a + b1, a1 + b), Eq(c + d1, c1 + d)
            p.have('e1', e1_stmt, lambda q: q.exact('h1')); p.have('e2', e2_stmt, lambda q: q.exact('h2'))
            p.apply('Z.mk_eq')
            calc_scope(p, ['a', 'b', 'a1', 'b1', 'c', 'd', 'c1', 'd1'], lambda a, b, a1, b1, c, d, c1, d1: (
                (a + c) + (b1 + d1), ('ring', (a + b1) + (c + d1)), (lambda q: (q.rw('e1'), q.rw('e2')), (a1 + b) + (c1 + d)), ('ring', (a1 + c1) + (b + d))))

        @thm(w, 'Z.mul_wd1', [N] * 6 + [lambda a, b, a1, b1, c, d: Eq(a + b1, a1 + b)],
             lambda a, b, a1, b1, c, d, h: Eq((a * c + b * d) + (a1 * d + b1 * c), (a1 * c + b1 * d) + (a * d + b * c)))
        def _(p):
            p.intro('a', 'b', 'a1', 'b1', 'c', 'd', 'h')
            names = ['a', 'b', 'a1', 'b1', 'c', 'd']
            with p.focus():
                a, b, a1, b1, c, d = (p.h(x) for x in names)
                st0, st1 = Eq(a * c + b1 * c, a1 * c + b * c), Eq(a * d + b1 * d, a1 * d + b * d)
            p.have('s0', st0, lambda q: calc_scope(q, names, lambda a, b, a1, b1, c, d: (a * c + b1 * c, ('ring', (a + b1) * c), (lambda r: r.rw('h'), (a1 + b) * c), ('ring', a1 * c + b * c))))
            p.have('s1', st1, lambda q: calc_scope(q, names, lambda a, b, a1, b1, c, d: (a * d + b1 * d, ('ring', (a + b1) * d), (lambda r: r.rw('h'), (a1 + b) * d), ('ring', a1 * d + b * d))))
            calc_scope(p, names, lambda a, b, a1, b1, c, d: (
                (a * c + b * d) + (a1 * d + b1 * c), ('ring', (a * c + b1 * c) + (a1 * d + b * d)), (lambda q: q.rw('s0'), (a1 * c + b * c) + (a1 * d + b * d)),
                (lambda q: q.rw('s1', rev=True), (a1 * c + b * c) + (a * d + b1 * d)), ('ring', (a1 * c + b1 * d) + (a * d + b * c))))

        @thm(w, 'Z.mul_wd2', [N] * 6 + [lambda a, b, c, d, c1, d1: Eq(c + d1, c1 + d)],
             lambda a, b, c, d, c1, d1, h: Eq((a * c + b * d) + (a * d1 + b * c1), (a * c1 + b * d1) + (a * d + b * c)))
        def _(p):
            p.intro('a', 'b', 'c', 'd', 'c1', 'd1', 'h')
            names = ['a', 'b', 'c', 'd', 'c1', 'd1']
            with p.focus():
                a, b, c, d, c1, d1 = (p.h(x) for x in names)
                st0, st1 = Eq(a * (c + d1), a * (c1 + d)), Eq(b * (c1 + d), b * (c + d1))
            p.have('t0', st0, lambda q: q.rw('h'))
            p.have('t1', st1, lambda q: q.rw('h', rev=True))
            calc_scope(p, names, lambda a, b, c, d, c1, d1: (
                (a * c + b * d) + (a * d1 + b * c1), ('ring', a * (c + d1) + b * (c1 + d)), (lambda q: q.rw('t0'), a * (c1 + d) + b * (c1 + d)),
                (lambda q: q.rw('t1'), a * (c1 + d) + b * (c + d1)), ('ring', (a * c1 + b * d1) + (a * d + b * c))))

        @thm(w, 'Z.mul_wd', [NN, NN, NN, NN, lambda p, p1, q, q1: Zrel(p, p1), lambda p, p1, q, q1, h1: Zrel(q, q1)],
             lambda p, p1, q, q1, h1, h2: Eq(Zmk(F(p) * F(q) + S(p) * S(q), F(p) * S(q) + S(p) * F(q)), Zmk(F(p1) * F(q1) + S(p1) * S(q1), F(p1) * S(q1) + S(p1) * F(q1))))
        def _(p):
            p.intro('p', 'p1', 'q', 'q1', 'h1', 'h2')
            p.cases('p', names=['a', 'b']); p.cases('p1', names=['a1', 'b1']); p.cases('q', names=['c', 'd']); p.cases('q1', names=['c1', 'd1'])
            with p.focus():
                a, b, a1, b1, c, d, c1, d1 = (p.h(x) for x in ('a', 'b', 'a1', 'b1', 'c', 'd', 'c1', 'd1'))
                e1_stmt, e2_stmt = Eq(a + b1, a1 + b), Eq(c + d1, c1 + d)
                x, y, z = Zmk(a * c + b * d, a * d + b * c), Zmk(a1 * c + b1 * d, a1 * d + b1 * c), Zmk(a1 * c1 + b1 * d1, a1 * d1 + b1 * c1)
            p.have('e1', e1_stmt, lambda q: q.exact('h1')); p.have('e2', e2_stmt, lambda q: q.exact('h2'))
            p.apply('trans', Z, x, y, z)
            p.apply('Z.mk_eq'); p.apply('Z.mul_wd1'); p.exact('e1')
            p.apply('Z.mk_eq'); p.apply('Z.mul_wd2'); p.exact('e2')

        # ---- the operations
        refl_rel = w.fun([NN], lambda p: w['refl'](N, F(p) + S(p)))                         # Zrel p p
        w.define('Z.neg', w.arrow(Z, Z), LIFT(NN, Zrel, Z, w.lam(NN, lambda p: Zmk(S(p), F(p))), w.fun([NN, NN, lambda p, q: Zrel(p, q)], lambda p, q, h: w['Z.neg_wd'](p, q, h))))
        w.define('Z.add', w.arrow(Z, w.arrow(Z, Z)),
                 LIFT2(NN, Zrel, Z, w.fun([NN, NN], lambda p, q: Zmk(F(p) + F(q), S(p) + S(q))), refl_rel,
                       w.fun([NN, NN, NN, NN, lambda p, p1, q, q1: Zrel(p, p1), lambda p, p1, q, q1, h1: Zrel(q, q1)], lambda p, p1, q, q1, h1, h2: w['Z.add_wd'](p, p1, q, q1, h1, h2))))
        w.define('Z.mul', w.arrow(Z, w.arrow(Z, Z)),
                 LIFT2(NN, Zrel, Z, w.fun([NN, NN], lambda p, q: Zmk(F(p) * F(q) + S(p) * S(q), F(p) * S(q) + S(p) * F(q))), refl_rel,
                       w.fun([NN, NN, NN, NN, lambda p, p1, q, q1: Zrel(p, p1), lambda p, p1, q, q1, h1: Zrel(q, q1)], lambda p, p1, q, q1, h1, h2: w['Z.mul_wd'](p, p1, q, q1, h1, h2))))
        w.define('Z.zero', Z, Zmk(N0, N0))
        w.define('Z.one', Z, Zmk(one, N0))
        zneg, zadd, zmul, zzero, zone = w['Z.neg'], w['Z.add'], w['Z.mul'], w['Z.zero'], w['Z.one']
        w.define('Z.sub', w.arrow(Z, w.arrow(Z, Z)), w.fun([Z, Z], lambda x, y: zadd(x, zneg(y))))
        w.infix.update({'Z.add': ('+', 6), 'Z.mul': ('*', 7), 'Z.sub': ('-', 6)})
        w.define('Z.of_N', w.arrow(N, Z), w.fun([N], lambda n: Zmk(n, N0)))
        w.nota.register(Z, ZNotation(w))

        @thm(w, 'Z.neg_mk', [N, N], lambda a, b: Eq(zneg(Zmk(a, b)), Zmk(b, a)))
        def _(p): p.intro('a', 'b'); p.refl()

        @thm(w, 'Z.add_mk', [N, N, N, N], lambda a, b, c, d: Eq(zadd(Zmk(a, b), Zmk(c, d)), Zmk(a + c, b + d)))
        def _(p): p.intro('a', 'b', 'c', 'd'); p.refl()

        @thm(w, 'Z.mul_mk', [N, N, N, N], lambda a, b, c, d: Eq(zmul(Zmk(a, b), Zmk(c, d)), Zmk(a * c + b * d, a * d + b * c)))
        def _(p): p.intro('a', 'b', 'c', 'd'); p.refl()

        @thm(w, 'Z.sub_def', [Z, Z], lambda x, y: Eq(w['Z.sub'](x, y), zadd(x, zneg(y))))
        def _(p): p.intro('x', 'y'); p.refl()

    MKS = ['Z.neg_mk', 'Z.add_mk', 'Z.mul_mk']

    def zlaw(name, nvars, body, via_mk=True):
        """a law of Z: induction on every variable, the operations computed on the representatives, and the equation of N that remains by `ring`"""
        names = 'xyzuvw'[:nvars]
        @thm(w, name, [Z] * nvars, body)
        def _(p):
            p.intro(*names)
            for i, x in enumerate(names): p.induction_on('Z.induction_on', x, names=[chr(ord('a') + 2 * i), chr(ord('a') + 2 * i + 1)])
            rw_all(p, MKS)
            p.apply('Z.mk_eq'); p.ring()

    with w.nota.using(Z):
        zlaw('Z.add_comm', 2, lambda x, y: Eq(x + y, y + x))
        zlaw('Z.add_assoc', 3, lambda x, y, z: Eq((x + y) + z, x + (y + z)))
        zlaw('Z.add_zero', 1, lambda x: Eq(x + zzero, x))
        zlaw('Z.zero_add', 1, lambda x: Eq(zzero + x, x))
        zlaw('Z.add_left_comm', 3, lambda x, y, z: Eq(x + (y + z), y + (x + z)))
        zlaw('Z.add_right_comm', 3, lambda x, y, z: Eq((x + y) + z, (x + z) + y))
        zlaw('Z.add_neg_cancel', 1, lambda x: Eq(x + zneg(x), zzero))
        zlaw('Z.neg_add_cancel', 1, lambda x: Eq(zneg(x) + x, zzero))
        zlaw('Z.neg_neg', 1, lambda x: Eq(zneg(zneg(x)), x))
        zlaw('Z.neg_add', 2, lambda x, y: Eq(zneg(x + y), zneg(x) + zneg(y)))
        zlaw('Z.mul_comm', 2, lambda x, y: Eq(x * y, y * x))
        zlaw('Z.mul_assoc', 3, lambda x, y, z: Eq((x * y) * z, x * (y * z)))
        zlaw('Z.mul_zero', 1, lambda x: Eq(x * zzero, zzero))
        zlaw('Z.zero_mul', 1, lambda x: Eq(zzero * x, zzero))
        zlaw('Z.mul_one', 1, lambda x: Eq(x * zone, x))
        zlaw('Z.one_mul', 1, lambda x: Eq(zone * x, x))
        zlaw('Z.mul_left_comm', 3, lambda x, y, z: Eq(x * (y * z), y * (x * z)))
        zlaw('Z.mul_right_comm', 3, lambda x, y, z: Eq((x * y) * z, (x * z) * y))
        zlaw('Z.mul_add', 3, lambda x, y, z: Eq(x * (y + z), x * y + x * z))
        zlaw('Z.add_mul', 3, lambda x, y, z: Eq((x + y) * z, x * z + y * z))
        zlaw('Z.mul_neg', 2, lambda x, y: Eq(x * zneg(y), zneg(x * y)))
        zlaw('Z.neg_mul', 2, lambda x, y: Eq(zneg(x) * y, zneg(x * y)))


def int_pow(w):
    """powers of Z by exponents in N (computed by repeated squaring on the binary digits), the embedding of N, and the first facts about both"""
    PROP = w.PROP; Eq = w.Eq
    N, Pos, Z = w['N'], w['Pos'], w['Z']; XH, XO, XI, N0, Npos = w['XH'], w['XO'], w['XI'], w['N0'], w['Npos']
    zadd, zmul, zneg, zzero, zone, zof = w['Z.add'], w['Z.mul'], w['Z.neg'], w['Z.zero'], w['Z.one'], w['Z.of_N']
    Zmk = w['Z.mk']; nsucc = w['N.succ']; psucc = w['Pos.succ']
    one = Npos(XH)

    w.define('Z.sqr', w.arrow(Z, Z), w.fun([Z], lambda z: zmul(z, z)))
    sqr = w['Z.sqr']
    w.recursive('Z.ppow', w.forall_([Z, Pos], lambda x, e: Z),
                lambda pw: w.fun([Z, Pos], lambda x, e: w.match(e, lambda _: Z, [x, lambda q: sqr(pw(x, q)), lambda q: zmul(x, sqr(pw(x, q)))])), on=1)
    ppow = w['Z.ppow']
    w.recursive('Z.pow', w.forall_([Z, N], lambda x, e: Z),
                lambda f: w.fun([Z, N], lambda x, e: w.match(e, lambda _: Z, [zone, lambda q: ppow(x, q)])), on=1)
    zpow = w['Z.pow']
    w.infix.update({'Z.pow': ('^', 8)})

    # N.induction_on: the principle in the shape `induction_on` wants
    @thm(w, 'N.induction_on', [w.arrow(N, PROP), N, lambda P, n: P(N0), lambda P, n, h0: w.forall_([N, lambda m: P(m)], lambda m, hm: P(nsucc(m)))], lambda P, n, h0, hs: P(n))
    def _(p): p.intro('P', 'n', 'h0', 'hs'); p.exact(lambda q: w['N.ind'](q.h('P'), q.h('h0'), q.h('hs'), q.h('n')))

    with w.nota.using(Z):
        @thm(w, 'Z.sqr_def', [Z], lambda x: Eq(sqr(x), x * x))
        def _(p): p.intro('x'); p.refl()

        @thm(w, 'Z.pow_zero', [Z], lambda x: Eq(zpow(x, N0), zone))
        def _(p): p.intro('x'); p.refl()

        @thm(w, 'Z.pow_one', [Z], lambda x: Eq(zpow(x, one), x))
        def _(p): p.intro('x'); p.refl()

        @thm(w, 'Z.ppow_succ', [Z, Pos], lambda x, q: Eq(ppow(x, psucc(q)), x * ppow(x, q)))
        def _(p):
            p.intro('x', 'q'); p.induct('q', names=['r', 'ih', 'r', 'ih'])
            p.refl()
            p.refl()
            # ppow x (XO (succ r)) = sqr (ppow x (succ r)) ;  x * ppow x (XI r) = x * (x * sqr (ppow x r))
            p.rw('ih'); p.rw('Z.sqr_def'); p.rw('Z.sqr_def'); p.rw('Z.mul_assoc')
            with p.focus(): x, r = p.h('x'), p.h('r'); y = ppow(x, r)
            p.rw('Z.mul_left_comm', y, x, y)

        @thm(w, 'Z.pow_succ', [Z, N], lambda x, n: Eq(zpow(x, nsucc(n)), x * zpow(x, n)))
        def _(p):
            p.intro('x', 'n'); p.cases('n', names=['q'])
            p.apply('symm', Z); p.apply('Z.mul_one')
            p.apply('Z.ppow_succ')

        @thm(w, 'Z.pow_add', [Z, N, N], lambda x, m, n: Eq(zpow(x, w['N.add'](m, n)), zpow(x, m) * zpow(x, n)))
        def _(p):
            p.intro('x', 'm', 'n'); p.induction_on('N.induction_on', 'n')
            p.rw('N.add_zero'); p.apply('symm', Z); p.apply('Z.mul_one')
            p.intro('k', 'ih')
            p.rw('N.add_succ'); p.rw('Z.pow_succ'); p.rw('Z.pow_succ'); p.rw('ih')
            p.apply('Z.mul_left_comm')

    return None



def int_nat(w):
    """N inside Z (`Z.of_N`): it commutes with the operations, is injective; cancellation in Z; 0 and 1"""
    PROP = w.PROP; Eq = w.Eq
    N, Z = w['N'], w['Z']; N0 = w['N0']; Nsucc = w['N.succ']; zpow = w['Z.pow']
    zadd, zmul, zneg, zzero, zone, zof, Zmk = w['Z.add'], w['Z.mul'], w['Z.neg'], w['Z.zero'], w['Z.one'], w['Z.of_N'], w['Z.mk']
    nadd, nmul = w['N.add'], w['N.mul']
    F = w['False']
    MKS = ['Z.of_N_def', 'Z.neg_mk', 'Z.add_mk', 'Z.mul_mk']

    @thm(w, 'Z.of_N_def', [N], lambda n: Eq(zof(n), Zmk(n, N0)))
    def _(p): p.intro('n'); p.refl()

    @thm(w, 'Z.of_N_zero', [], lambda: Eq(zof(N0), zzero))
    def _(p): p.refl()

    @thm(w, 'Z.of_N_one', [], lambda: Eq(zof(w['Npos'](w['XH'])), zone))
    def _(p): p.refl()

    @thm(w, 'Z.of_N_add', [N, N], lambda m, n: Eq(zof(nadd(m, n)), zadd(zof(m), zof(n))))
    def _(p): p.intro('m', 'n'); p.refl()

    with w.nota.using(N):
        @thm(w, 'Z.of_N_mul', [N, N], lambda m, n: Eq(zof(m * n), zmul(zof(m), zof(n))))
        def _(p):
            p.intro('m', 'n'); rw_all(p, MKS); p.apply('Z.mk_eq'); p.ring()

    @thm(w, 'Z.of_N_succ', [N], lambda n: Eq(zof(Nsucc(n)), zadd(zof(n), zone)))
    def _(p): p.intro('n'); p.rw('N.succ_eq_add_one')           # the goal then closes by computation

    @thm(w, 'Z.of_N_inj', [N, N, lambda a, b: Eq(zof(a), zof(b))], lambda a, b, h: Eq(a, b))
    def _(p):
        p.intro('a', 'b', 'h')
        with p.focus(): a, b = p.h('a'), p.h('b'); st = Eq(nadd(a, N0), nadd(b, N0))
        p.have('e', st, lambda q: q.exact(lambda r: w['Z.mk_inj'](r.h('a'), N0, r.h('b'), N0, r.h('h'))))
        p.rw_at('e', 'N.add_zero'); p.rw_at('e', 'N.add_zero'); p.exact('e')

    with w.nota.using(Z):
        @thm(w, 'Z.add_left_cancel', [Z, Z, Z, lambda x, y, z: Eq(x + y, x + z)], lambda x, y, z, h: Eq(y, z))
        def _(p):
            p.intro('x', 'y', 'z', 'h')
            calc_scope(p, ['x', 'y', 'z'], lambda x, y, z: (
                y, (('Z.zero_add', y), zzero + y), (lambda q: q.rw('Z.neg_add_cancel'), (zneg(x) + x) + y), (('Z.add_assoc', zneg(x), x, y), zneg(x) + (x + y)),
                (lambda q: q.rw('h'), zneg(x) + (x + z)), (('Z.add_assoc', zneg(x), x, z), (zneg(x) + x) + z), (lambda q: q.rw('Z.neg_add_cancel'), zzero + z), (('Z.zero_add', z), z)))

        @thm(w, 'Z.add_right_cancel', [Z, Z, Z, lambda x, y, z: Eq(x + z, y + z)], lambda x, y, z, h: Eq(x, y))
        def _(p):
            p.intro('x', 'y', 'z', 'h'); p.apply('Z.add_left_cancel', 'z')
            p.rw('Z.add_comm', 'z', 'x'); p.rw('Z.add_comm', 'z', 'y'); p.exact('h')

    @thm(w, 'Z.zero_ne_one', [], lambda: ~Eq(zzero, zone))
    def _(p):
        p.intro('h')
        p.have('e', Eq(nadd(N0, N0), nadd(w['Npos'](w['XH']), N0)), lambda q: q.exact(lambda r: w['Z.mk_inj'](N0, N0, w['Npos'](w['XH']), N0, r.h('h'))))
        p.cases('e')


def int_ring(w):
    """the `ring_*` lemmas of Z (ring_lib.py), which is all `ring` needs to prove equations between integer expressions"""
    ring_lib.ring_lemma_set(w, ring_lib.z_desc(w))


def int_order(w):
    """the order of Z:  x <= y  iff  some natural k has  x + k = y  (the same certificate as for N); its laws, and the signs of an integer"""
    PROP = w.PROP; Eq = w.Eq; Ex = w['Ex']
    N, Z = w['N'], w['Z']; N0 = w['N0']
    zadd, zmul, zneg, zzero, zone, zof, Zmk = w['Z.add'], w['Z.mul'], w['Z.neg'], w['Z.zero'], w['Z.one'], w['Z.of_N'], w['Z.mk']
    nadd, nmul = w['N.add'], w['N.mul']; nle = w['N.le']
    MKS = ['Z.of_N_def', 'Z.neg_mk', 'Z.add_mk', 'Z.mul_mk']
    zle = w.define('Z.le', w.forall_([Z, Z], lambda x, y: PROP), w.fun([Z, Z], lambda x, y: Ex(N, w.lam(N, lambda k: Eq(zadd(x, zof(k)), y)))))
    zlt = w.define('Z.lt', w.forall_([Z, Z], lambda x, y: PROP), w.fun([Z, Z], lambda x, y: zle(zadd(x, zone), y)))
    w.infix.update({'Z.le': ('<=', 4), 'Z.lt': ('<', 4)})

    @thm(w, 'Z.le_refl', [Z], lambda x: zle(x, x))
    def _(p): p.intro('x'); p.exists(N0); p.exact(lambda q: w['Z.add_zero'](q.h('x')))

    @thm(w, 'Z.le_trans', [Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h1: zle(y, z)], lambda x, y, z, h1, h2: zle(x, z))
    def _(p):
        p.intro('x', 'y', 'z', 'h1', 'h2'); p.cases('h1', names=['a', 'ha']); p.cases('h2', names=['b', 'hb'])
        p.exists((nadd, 'a', 'b'))
        p.rw('Z.of_N_add'); p.rw('Z.add_assoc', rev=True); p.rw('ha'); p.exact('hb')

    @thm(w, 'Z.le_antisymm', [Z, Z, lambda x, y: zle(x, y), lambda x, y, h1: zle(y, x)], lambda x, y, h1, h2: Eq(x, y))
    def _(p):
        p.intro('x', 'y', 'h1', 'h2'); p.cases('h1', names=['a', 'ha']); p.cases('h2', names=['b', 'hb'])
        with p.focus():
            x, a, b = p.h('x'), p.h('a'), p.h('b')
            st1, st2, st3 = Eq(zadd(x, zof(nadd(a, b))), zadd(x, zzero)), Eq(zof(nadd(a, b)), zzero), Eq(nadd(a, b), N0)
        p.have('e1', st1, lambda q: calc_scope(q, ['x', 'a', 'b', 'y'], lambda x, a, b, y: (
            zadd(x, zof(nadd(a, b))), (lambda r: r.rw('Z.of_N_add'), zadd(x, zadd(zof(a), zof(b)))), (('Z.add_assoc', x, zof(a), zof(b)), zadd(zadd(x, zof(a)), zof(b))),
            (lambda r: r.rw('ha'), zadd(y, zof(b))), ('hb', x), (('Z.add_zero', x), zadd(x, zzero)))))
        p.have('e2', st2, lambda q: q.exact(lambda r: w['Z.add_left_cancel'](r.h('x'), zof(nadd(r.h('a'), r.h('b'))), zzero, r.h('e1'))))
        p.have('e3', st3, lambda q: q.exact(lambda r: w['Z.of_N_inj'](nadd(r.h('a'), r.h('b')), N0, r.h('e2'))))
        p.obtain(['e4', 'e5'], ('N.add_eq_zero', 'a', 'b', 'e3'))
        p.rw_at('ha', 'e4')
        calc_scope(p, ['x', 'y'], lambda x, y: (x, (('Z.add_zero', x), zadd(x, zzero)), ('ha', y)))

    # ---- the order on representatives
    with w.nota.using(N):
        @thm(w, 'Z.le_of_mk', [N, N, N, N, lambda a, b, c, d: nle(a + d, c + b)], lambda a, b, c, d, h: zle(Zmk(a, b), Zmk(c, d)))
        def _(p):
            p.intro('a', 'b', 'c', 'd', 'h'); p.cases('h', names=['k', 'hk']); p.exists('k')
            rw_all(p, MKS); p.apply('Z.mk_eq')
            calc_scope(p, ['a', 'b', 'c', 'd', 'k'], lambda a, b, c, d, k: ((a + k) + d, ('ring', (a + d) + k), ('hk', c + b), ('ring', c + (b + N0))))

        @thm(w, 'Z.mk_le', [N, N, N, N, lambda a, b, c, d: zle(Zmk(a, b), Zmk(c, d))], lambda a, b, c, d, h: nle(a + d, c + b))
        def _(p):
            p.intro('a', 'b', 'c', 'd', 'h'); p.cases('h', names=['k', 'hk']); p.exists('k')
            rw_all_at(p, 'hk', MKS)
            with p.focus(): a, b, c, d, k = (p.h(x) for x in 'abcdk'); st = Eq((a + k) + d, c + (b + N0))
            p.have('e', st, lambda q: q.exact(lambda r: w['Z.mk_inj'](r.h('a') + r.h('k'), r.h('b') + N0, r.h('c'), r.h('d'), r.h('hk'))))
            calc_scope(p, ['a', 'b', 'c', 'd', 'k'], lambda a, b, c, d, k: ((a + d) + k, ('ring', (a + k) + d), ('e', c + (b + N0)), ('ring', c + b)))

    @thm(w, 'Z.le_total', [Z, Z], lambda x, y: zle(x, y) | zle(y, x))
    def _(p):
        p.intro('x', 'y'); p.induction_on('Z.induction_on', 'x', names=['a', 'b']); p.induction_on('Z.induction_on', 'y', names=['c', 'd'])
        p.obtain(['h', 'h'], ('N.le_total', (nadd, 'a', 'd'), (nadd, 'c', 'b')))
        p.left(); p.apply('Z.le_of_mk'); p.exact('h')
        p.right(); p.apply('Z.le_of_mk'); p.exact('h')

    @thm(w, 'Z.nonneg_of_N', [N], lambda k: zle(zzero, zof(k)))
    def _(p): p.intro('k'); p.exists('k'); p.exact(lambda q: w['Z.zero_add'](zof(q.h('k'))))

    with w.nota.using(Z):
        @thm(w, 'Z.add_le_add_right', [Z, Z, Z, lambda x, y, z: zle(x, y)], lambda x, y, z, h: zle(x + z, y + z))
        def _(p):
            p.intro('x', 'y', 'z', 'h'); p.cases('h', names=['k', 'hk']); p.exists('k')
            calc_scope(p, ['x', 'y', 'z', 'k'], lambda x, y, z, k: (
                (x + z) + zof(k), (('Z.add_right_comm', x, z, zof(k)), (x + zof(k)) + z), (lambda q: q.rw('hk'), y + z)))

        @thm(w, 'Z.add_le_add_left', [Z, Z, Z, lambda x, y, z: zle(x, y)], lambda x, y, z, h: zle(z + x, z + y))
        def _(p):
            p.intro('x', 'y', 'z', 'h'); p.cases('h', names=['k', 'hk']); p.exists('k')
            calc_scope(p, ['x', 'y', 'z', 'k'], lambda x, y, z, k: (
                (z + x) + zof(k), (('Z.add_assoc', z, x, zof(k)), z + (x + zof(k))), (lambda q: q.rw('hk'), z + y)))

        @thm(w, 'Z.add_le_add', [Z, Z, Z, Z, lambda x, y, u, v: zle(x, y), lambda x, y, u, v, h1: zle(u, v)], lambda x, y, u, v, h1, h2: zle(x + u, y + v))
        def _(p):
            p.intro('x', 'y', 'u', 'v', 'h1', 'h2')
            p.apply('Z.le_trans', (zadd, 'x', 'u'), (zadd, 'y', 'u'), (zadd, 'y', 'v'))
            p.apply('Z.add_le_add_right'); p.exact('h1')
            p.apply('Z.add_le_add_left'); p.exact('h2')

        @thm(w, 'Z.nonneg_exists', [Z, lambda x: zle(zzero, x)], lambda x, h: Ex(N, w.lam(N, lambda k: Eq(zof(k), x))))
        def _(p):
            p.intro('x', 'h'); p.cases('h', names=['k', 'hk']); p.exists('k')
            calc_scope(p, ['x', 'k'], lambda x, k: (zof(k), (('Z.zero_add', zof(k)), zzero + zof(k)), ('hk', x)))

        @thm(w, 'Z.mul_nonneg', [Z, Z, lambda x, y: zle(zzero, x), lambda x, y, h1: zle(zzero, y)], lambda x, y, h1, h2: zle(zzero, x * y))
        def _(p):
            p.intro('x', 'y', 'hx', 'hy'); p.obtain(['a', 'ha'], ('Z.nonneg_exists', 'x', 'hx')); p.obtain(['b', 'hb'], ('Z.nonneg_exists', 'y', 'hy'))
            p.exists((nmul, 'a', 'b'))
            calc_scope(p, ['x', 'y', 'a', 'b'], lambda x, y, a, b: (
                zzero + zof(nmul(a, b)), (('Z.zero_add', zof(nmul(a, b))), zof(nmul(a, b))), (('Z.of_N_mul', a, b), zof(a) * zof(b)),
                (lambda q: (q.rw('ha'), q.rw('hb')), x * y)))

def int_order2(w):
    """differences, negation and squares in the order of Z; the strict order (x < y is x + 1 <= y); decidable equality, trichotomy; products of positives"""
    PROP = w.PROP; Eq = w.Eq; Ex = w['Ex']; Or = w['Or'] if 'Or' in w else None
    N, Z = w['N'], w['Z']; N0 = w['N0']; none_ = w['Npos'](w['XH'])
    zadd, zmul, zneg, zzero, zone, zof, Zmk = w['Z.add'], w['Z.mul'], w['Z.neg'], w['Z.zero'], w['Z.one'], w['Z.of_N'], w['Z.mk']
    nadd, nmul = w['N.add'], w['N.mul']; zle, zlt = w['Z.le'], w['Z.lt']; F = w['False']; nsucc = w['N.succ']

    with w.nota.using(Z):
        @thm(w, 'Z.le_of_le_eq', [Z, Z, Z, lambda a, b, c: zle(a, b), lambda a, b, c, h1: Eq(b, c)], lambda a, b, c, h1, h2: zle(a, c))
        def _(p): p.intro('a', 'b', 'c', 'h1', 'h2'); p.cases('h2'); p.exact('h1')

        @thm(w, 'Z.le_of_eq_le', [Z, Z, Z, lambda a, b, c: Eq(a, b), lambda a, b, c, h1: zle(b, c)], lambda a, b, c, h1, h2: zle(a, c))
        def _(p): p.intro('a', 'b', 'c', 'h1', 'h2'); p.cases('h1'); p.exact('h2')

        @thm(w, 'Z.sub_nonneg_of_le', [Z, Z, lambda x, y: zle(x, y)], lambda x, y, h: zle(zzero, y - x))
        def _(p):
            p.intro('x', 'y', 'h'); p.cases('h', names=['k', 'hk']); p.exists('k')
            calc_scope(p, ['x', 'y', 'k'], lambda x, y, k: (zzero + zof(k), ('ring', (x + zof(k)) - x), (lambda q: q.rw('hk'), y - x)))

        @thm(w, 'Z.le_of_sub_nonneg', [Z, Z, lambda x, y: zle(zzero, y - x)], lambda x, y, h: zle(x, y))
        def _(p):
            p.intro('x', 'y', 'h'); p.cases('h', names=['k', 'hk']); p.exists('k')
            calc_scope(p, ['x', 'y', 'k'], lambda x, y, k: (x + zof(k), ('ring', x + (zzero + zof(k))), (lambda q: q.rw('hk'), x + (y - x)), ('ring', y)))

        @thm(w, 'Z.neg_le_neg', [Z, Z, lambda x, y: zle(x, y)], lambda x, y, h: zle(zneg(y), zneg(x)))
        def _(p):
            p.intro('x', 'y', 'h'); p.apply('Z.le_of_sub_nonneg')
            with p.focus(): x, y = p.h('x'), p.h('y'); st = Eq(zneg(x) - zneg(y), y - x)
            p.have('e', st, lambda q: q.ring())
            p.rw('e'); p.apply('Z.sub_nonneg_of_le'); p.exact('h')

        @thm(w, 'Z.add_nonneg', [Z, Z, lambda x, y: zle(zzero, x), lambda x, y, h: zle(zzero, y)], lambda x, y, hx, hy: zle(zzero, x + y))
        def _(p):
            p.intro('x', 'y', 'hx', 'hy'); p.exact(lambda q: w['Z.add_le_add'](zzero, q.h('x'), zzero, q.h('y'), q.h('hx'), q.h('hy')))

        @thm(w, 'Z.sq_nonneg', [Z], lambda x: zle(zzero, x * x))
        def _(p):
            p.intro('x'); p.obtain(['h', 'h'], ('Z.le_total', zzero, 'x'))
            p.exact(lambda q: w['Z.mul_nonneg'](q.h('x'), q.h('x'), q.h('h'), q.h('h')))
            with p.focus(): x = p.h('x'); st1 = zle(zzero, zneg(x)); st2 = Eq(zneg(x) * zneg(x), x * x)
            p.have('n', st1, lambda q: q.exact(lambda r: w['Z.neg_le_neg'](r.h('x'), zzero, r.h('h'))))
            p.have('e', st2, lambda q: q.ring())
            p.exact(lambda r: w['Z.le_of_le_eq'](zzero, zneg(r.h('x')) * zneg(r.h('x')), r.h('x') * r.h('x'),
                                                  w['Z.mul_nonneg'](zneg(r.h('x')), zneg(r.h('x')), r.h('n'), r.h('n')), r.h('e')))

        @thm(w, 'Z.mul_le_mul_of_nonneg_left', [Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h: zle(zzero, z)], lambda x, y, z, h, hz: zle(z * x, z * y))
        def _(p):
            p.intro('x', 'y', 'z', 'h', 'hz'); p.apply('Z.le_of_sub_nonneg')
            with p.focus(): x, y, z = p.h('x'), p.h('y'), p.h('z'); st = Eq(z * y - z * x, z * (y - x))
            p.have('e', st, lambda q: q.ring())
            p.rw('e'); p.apply('Z.mul_nonneg'); p.exact('hz'); p.apply('Z.sub_nonneg_of_le'); p.exact('h')

        @thm(w, 'Z.mul_le_mul_of_nonneg_right', [Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h: zle(zzero, z)], lambda x, y, z, h, hz: zle(x * z, y * z))
        def _(p):
            p.intro('x', 'y', 'z', 'h', 'hz'); p.rw('Z.mul_comm', 'x', 'z'); p.rw('Z.mul_comm', 'y', 'z'); p.apply('Z.mul_le_mul_of_nonneg_left'); p.exact('h'); p.exact('hz')

        @thm(w, 'Z.le_succ', [Z], lambda x: zle(x, x + zone))
        def _(p): p.intro('x'); p.exists(none_); p.ring()

        # ---- the strict order:  x < y  is  x + 1 <= y
        @thm(w, 'Z.le_of_lt', [Z, Z, lambda x, y: zlt(x, y)], lambda x, y, h: zle(x, y))
        def _(p):
            p.intro('x', 'y', 'h'); p.apply('Z.le_trans', 'x', (zadd, 'x', zone)); p.apply('Z.le_succ'); p.exact('h')

        @thm(w, 'Z.lt_irrefl', [Z, lambda x: zlt(x, x)], lambda x, h: F)
        def _(p):
            p.intro('x', 'h')
            with p.focus(): x = p.h('x'); st = Eq(x + zone, x + zzero)
            p.have('e', st, lambda q: q.exact(lambda r: w['trans'](Z, r.h('x') + zone, r.h('x'), r.h('x') + zzero,
                                                                    w['Z.le_antisymm'](r.h('x') + zone, r.h('x'), r.h('h'), w['Z.le_succ'](r.h('x'))), w['symm'](Z, r.h('x') + zzero, r.h('x'), w['Z.add_zero'](r.h('x'))))))
            p.exact(lambda q: w['Z.zero_ne_one'](w['symm'](Z, zone, zzero, w['Z.add_left_cancel'](q.h('x'), zone, zzero, q.h('e')))))

        @thm(w, 'Z.lt_of_lt_of_le', [Z, Z, Z, lambda x, y, z: zlt(x, y), lambda x, y, z, h1: zle(y, z)], lambda x, y, z, h1, h2: zlt(x, z))
        def _(p): p.intro('x', 'y', 'z', 'h1', 'h2'); p.apply('Z.le_trans', (zadd, 'x', zone), 'y'); p.exact('h1'); p.exact('h2')

        @thm(w, 'Z.lt_of_le_of_lt', [Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h1: zlt(y, z)], lambda x, y, z, h1, h2: zlt(x, z))
        def _(p):
            p.intro('x', 'y', 'z', 'h1', 'h2'); p.apply('Z.le_trans', (zadd, 'x', zone), (zadd, 'y', zone)); p.apply('Z.add_le_add_right'); p.exact('h1'); p.exact('h2')

        @thm(w, 'Z.lt_trans', [Z, Z, Z, lambda x, y, z: zlt(x, y), lambda x, y, z, h1: zlt(y, z)], lambda x, y, z, h1, h2: zlt(x, z))
        def _(p):
            p.intro('x', 'y', 'z', 'h1', 'h2'); p.apply('Z.lt_of_lt_of_le', 'x', 'y'); p.exact('h1'); p.apply('Z.le_of_lt'); p.exact('h2')


def int_order3(w):
    """discreteness (an integer below another is at least one step below it), trichotomy, the decision of equality; the signs of an integer; the integral domain; products of positives"""
    PROP = w.PROP; Eq = w.Eq; Ex = w['Ex']
    N, Z = w['N'], w['Z']; N0 = w['N0']
    zadd, zmul, zneg, zzero, zone, zof = w['Z.add'], w['Z.mul'], w['Z.neg'], w['Z.zero'], w['Z.one'], w['Z.of_N']
    nsucc, nmul = w['N.succ'], w['N.mul']; zle, zlt = w['Z.le'], w['Z.lt']; F = w['False']

    @thm(w, 'N.zero_or_succ', [N], lambda k: Eq(k, N0) | Ex(N, w.lam(N, lambda j: Eq(k, nsucc(j)))))
    def _(p):
        p.intro('k'); p.induction_on('N.induction_on', 'k')
        p.apply('inl'); p.refl()
        p.intro('m', 'hm'); p.apply('inr'); p.exists('m'); p.refl()

    with w.nota.using(Z):
        @thm(w, 'Z.lt_or_eq_of_le', [Z, Z, lambda x, y: zle(x, y)], lambda x, y, h: zlt(x, y) | Eq(x, y))
        def _(p):
            p.intro('x', 'y', 'h'); p.cases('h', names=['k', 'hk'])
            p.obtain(['e', 'e'], ('N.zero_or_succ', 'k'))
            p.apply('inr'); p.rw_at('hk', 'e', new='hk0')
            calc_scope(p, ['x', 'y'], lambda x, y: (x, ('ring', x + zof(N0)), ('hk0', y)))
            p.cases('e', names=['j', 'ej'])
            p.apply('inl'); p.rw_at('hk', 'ej', new='hk1'); p.rw_at('hk1', 'Z.of_N_succ', new='hk2')
            p.exists('j')
            calc_scope(p, ['x', 'y', 'j'], lambda x, y, j: ((x + zone) + zof(j), ('ring', x + (zof(j) + zone)), ('hk2', y)))

        @thm(w, 'Z.le_or_lt', [Z, Z], lambda x, y: zle(x, y) | zlt(y, x))
        def _(p):
            p.intro('x', 'y'); p.obtain(['h', 'h'], ('Z.le_total', 'x', 'y'))
            p.apply('inl'); p.exact('h')
            p.obtain(['h2', 'h2'], ('Z.lt_or_eq_of_le', 'y', 'x', 'h'))
            p.apply('inr'); p.exact('h2')
            p.apply('inl'); p.rw('h2'); p.apply('Z.le_refl')

        @thm(w, 'Z.not_le_of_lt', [Z, Z, lambda x, y: zlt(x, y)], lambda x, y, h: ~zle(y, x))
        def _(p):
            p.intro('x', 'y', 'h', 'h2'); p.apply('Z.lt_irrefl', 'x')
            p.exact(lambda q: w['Z.lt_of_lt_of_le'](q.h('x'), q.h('y'), q.h('x'), q.h('h'), q.h('h2')))

        @thm(w, 'Z.not_lt_of_le', [Z, Z, lambda x, y: zle(x, y)], lambda x, y, h: ~zlt(y, x))
        def _(p):
            p.intro('x', 'y', 'h', 'h2'); p.apply('Z.lt_irrefl', 'x')
            p.exact(lambda q: w['Z.lt_of_le_of_lt'](q.h('x'), q.h('y'), q.h('x'), q.h('h'), q.h('h2')))

        @thm(w, 'Z.le_of_not_lt', [Z, Z, lambda x, y: ~zlt(x, y)], lambda x, y, h: zle(y, x))
        def _(p):
            p.intro('x', 'y', 'h'); p.obtain(['h1', 'h1'], ('Z.le_or_lt', 'y', 'x'))
            p.exact('h1')
            p.apply('false_elim'); p.apply('h'); p.exact('h1')

        @thm(w, 'Z.lt_of_not_le', [Z, Z, lambda x, y: ~zle(x, y)], lambda x, y, h: zlt(y, x))
        def _(p):
            p.intro('x', 'y', 'h'); p.obtain(['h1', 'h1'], ('Z.le_or_lt', 'x', 'y'))
            p.apply('false_elim'); p.apply('h'); p.exact('h1')
            p.exact('h1')

        @thm(w, 'Z.ne_of_lt', [Z, Z, lambda x, y: zlt(x, y)], lambda x, y, h: ~Eq(x, y))
        def _(p):
            p.intro('x', 'y', 'h', 'e'); p.apply('Z.lt_irrefl', 'x'); p.rw_at('h', 'e', rev=True, new='h3'); p.exact('h3')

        @thm(w, 'Z.eq_dec', [Z, Z], lambda x, y: Eq(x, y) | ~Eq(x, y))
        def _(p):
            p.intro('x', 'y'); p.obtain(['h', 'h'], ('Z.le_or_lt', 'x', 'y'))
            p.obtain(['h2', 'h2'], ('Z.lt_or_eq_of_le', 'x', 'y', 'h'))
            p.apply('inr'); p.apply('Z.ne_of_lt'); p.exact('h2')
            p.apply('inl'); p.exact('h2')
            p.apply('inr'); p.intro('e'); p.apply('Z.ne_of_lt', 'y', 'x'); p.exact('h'); p.rw('e')

        @thm(w, 'Z.lt_trichotomy', [Z, Z], lambda x, y: zlt(x, y) | (Eq(x, y) | zlt(y, x)))
        def _(p):
            p.intro('x', 'y'); p.obtain(['h', 'h'], ('Z.le_or_lt', 'x', 'y'))
            p.obtain(['h2', 'h2'], ('Z.lt_or_eq_of_le', 'x', 'y', 'h'))
            p.apply('inl'); p.exact('h2')
            p.apply('inr'); p.apply('inl'); p.exact('h2')
            p.apply('inr'); p.apply('inr'); p.exact('h')

        @thm(w, 'Z.zero_lt_one', [], lambda: zlt(zzero, zone))
        def _(p): p.exists(N0); p.ring()


if __name__ == '__main__':
    import time, bin_lib
    t0 = time.time(); w = bin_lib.build_bin_defs()
    quot_lib.declare_quotients(w); quot_lib.declare_prod(w); quot_lib.quot_lift2(w)
    bool_lib(w); int_core(w); int_ops(w); int_pow(w); int_nat(w); int_ring(w); int_order(w); int_order2(w); int_order3(w)
    print('int world so far: %d globals in %.2f s' % (w.k.num_globals(), time.time() - t0))
    print('%d globals in %.2f s' % (w.k.num_globals(), time.time() - t0))
