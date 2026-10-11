"""nat_lib.py -- arithmetic on the unary natural numbers: addition, multiplication, powers, order, and the laws of a commutative semiring, all proved by tactic scripts
and admitted through the kernel's checked path (see author.py / tactics.py). Statements are written with Python operators on Terms (`n + m`, `n * m`, `n ** k`, `n <= m`)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from author import World, AuthorError, Term, Notation, DispatchNotation
import lnl_base
from lnl_base import thm


class NatNotation(Notation):
    def __init__(self, w): self.w = w
    def add(self, a, b): return self.w['add'](a, b)
    def mul(self, a, b): return self.w['mul'](a, b)
    def pow(self, a, b): return self.w['pow'](a, b)
    def le(self, a, b): return self.w['Le'](a, b)
    def lt(self, a, b): return self.w['Lt'](a, b)
    def lit(self, n):
        t = self.w['Zero']; S = self.w['Succ']
        for _ in range(n): t = S(t)
        return t


def declare_arith(w):
    Nat, S, Z = w['Nat'], w['Succ'], w['Zero']
    w.recursive('mul', w.forall_([Nat, Nat], lambda n, m: Nat),
                lambda mul: w.fun([Nat, Nat], lambda n, m: w.match(n, lambda _: Nat, [Z, lambda n1: w['add'](m, mul(n1, m))])))
    w.recursive('pow', w.forall_([Nat, Nat], lambda b, e: Nat),
                lambda pw: w.fun([Nat, Nat], lambda b, e: w.match(e, lambda _: Nat, [S(Z), lambda e1: w['mul'](b, pw(b, e1))])), on=1)
    w.recursive('pred', w.arrow(Nat, Nat), lambda pred: w.fun([Nat], lambda n: w.match(n, lambda _: Nat, [Z, lambda n1: n1])))
    w.define('Lt', w.forall_([Nat, Nat], lambda n, m: w.PROP), w.fun([Nat, Nat], lambda n, m: w['Le'](S(n), m)))
    w.infix.update({'pow': ('^', 8), 'Lt': ('<', 4)})


def succ_lemmas(w):
    Nat, S, Z, F = w['Nat'], w['Succ'], w['Zero'], w['False']
    Eq = w.Eq

    @thm(w, 'succ_inj', [Nat, Nat, lambda n, m: Eq(S(n), S(m))], lambda n, m, h: Eq(n, m))
    def _(p): p.intro('n', 'm', 'h'); p.exact(lambda p: w['f_equal'](Nat, Nat, w['pred'], S(p.h('n')), S(p.h('m')), p.h('h')))      # pred (Succ n) computes to n

    @thm(w, 'succ_ne_zero', [Nat, lambda n: Eq(S(n), Z)], lambda n, h: F)
    def _(p): p.intro('n', 'h'); p.cases('h')

    @thm(w, 'zero_ne_succ', [Nat, lambda n: Eq(Z, S(n))], lambda n, h: F)
    def _(p): p.intro('n', 'h'); p.cases('h')


def add_lemmas(w):
    Nat, S, Z = w['Nat'], w['Succ'], w['Zero']
    Eq = w.Eq

    @thm(w, 'add_zero', [Nat], lambda n: Eq(n + Z, n))
    def _(p): p.intro('n'); p.induct('n', names=['k', 'ih']); p.refl(); p.rw('ih')

    @thm(w, 'zero_add', [Nat], lambda n: Eq(Z + n, n))
    def _(p): p.intro('n'); p.refl()

    @thm(w, 'add_succ', [Nat, Nat], lambda n, m: Eq(n + S(m), S(n + m)))
    def _(p): p.intro('n', 'm'); p.induct('n', names=['k', 'ih']); p.refl(); p.rw('ih')

    @thm(w, 'succ_add', [Nat, Nat], lambda n, m: Eq(S(n) + m, S(n + m)))
    def _(p): p.intro('n', 'm'); p.refl()

    @thm(w, 'add_assoc', [Nat, Nat, Nat], lambda n, m, k: Eq((n + m) + k, n + (m + k)))
    def _(p): p.intro('n', 'm', 'k'); p.induct('n', names=['i', 'ih']); p.refl(); p.rw('ih')

    @thm(w, 'add_comm', [Nat, Nat], lambda n, m: Eq(n + m, m + n))
    def _(p):
        p.intro('n', 'm'); p.induct('n', names=['k', 'ih'])
        p.rw('add_zero')
        p.rw('add_succ'); p.rw('ih')

    @thm(w, 'add_left_comm', [Nat, Nat, Nat], lambda n, m, k: Eq(n + (m + k), m + (n + k)))
    def _(p):
        p.intro('n', 'm', 'k')
        p.rw('add_assoc', rev=True); p.rw('add_comm', 'n', 'm'); p.rw('add_assoc')

    @thm(w, 'add_right_comm', [Nat, Nat, Nat], lambda n, m, k: Eq((n + m) + k, (n + k) + m))
    def _(p):
        p.intro('n', 'm', 'k')
        p.rw('add_assoc'); p.rw('add_comm', 'm', 'k'); p.rw('add_assoc', rev=True)

    @thm(w, 'add_left_cancel', [Nat, Nat, Nat, lambda n, m, k: Eq(n + m, n + k)], lambda n, m, k, h: Eq(m, k))
    def _(p):
        p.intro('n', 'm', 'k', 'h'); p.induct('n', names=['i', 'ih'], generalizing=['h'])
        p.exact('h')
        p.apply('ih'); p.apply('succ_inj'); p.exact('h')

    @thm(w, 'add_right_cancel', [Nat, Nat, Nat, lambda n, m, k: Eq(n + m, k + m)], lambda n, m, k, h: Eq(n, k))
    def _(p):
        p.intro('n', 'm', 'k', 'h'); p.apply('add_left_cancel', 'm')
        p.rw('add_comm', 'm', 'n'); p.rw('add_comm', 'm', 'k'); p.exact('h')


def mul_lemmas(w):
    Nat, S, Z, F = w['Nat'], w['Succ'], w['Zero'], w['False']
    Eq = w.Eq

    @thm(w, 'mul_zero', [Nat], lambda n: Eq(n * Z, Z))
    def _(p): p.intro('n'); p.induct('n', names=['k', 'ih']); p.refl(); p.exact('ih')

    @thm(w, 'zero_mul', [Nat], lambda n: Eq(Z * n, Z))
    def _(p): p.intro('n'); p.refl()

    @thm(w, 'succ_mul', [Nat, Nat], lambda n, m: Eq(S(n) * m, m + n * m))
    def _(p): p.intro('n', 'm'); p.refl()

    @thm(w, 'mul_succ', [Nat, Nat], lambda n, m: Eq(n * S(m), n + n * m))
    def _(p):
        p.intro('n', 'm'); p.induct('n', names=['k', 'ih']); p.refl()
        p.rw('ih'); p.rw('add_left_comm')

    @thm(w, 'one_mul', [Nat], lambda n: Eq(S(Z) * n, n))
    def _(p): p.intro('n'); p.rw('add_zero')

    @thm(w, 'mul_one', [Nat], lambda n: Eq(n * S(Z), n))
    def _(p): p.intro('n'); p.induct('n', names=['k', 'ih']); p.refl(); p.rw('ih')

    @thm(w, 'mul_comm', [Nat, Nat], lambda n, m: Eq(n * m, m * n))
    def _(p):
        p.intro('n', 'm'); p.induct('n', names=['k', 'ih'])
        p.rw('mul_zero')
        p.rw('mul_succ'); p.rw('ih')

    @thm(w, 'add_mul', [Nat, Nat, Nat], lambda n, m, k: Eq((n + m) * k, n * k + m * k))
    def _(p):
        p.intro('n', 'm', 'k'); p.induct('n', names=['i', 'ih']); p.refl()
        p.rw('ih'); p.rw('add_assoc')

    @thm(w, 'mul_add', [Nat, Nat, Nat], lambda n, m, k: Eq(n * (m + k), n * m + n * k))
    def _(p):
        p.intro('n', 'm', 'k'); p.rw('mul_comm', 'n', 'm + k') if False else None
        p.induct('n', names=['i', 'ih']); p.refl()
        p.rw('ih'); p.rw('add_assoc'); p.rw('add_assoc'); p.rw('add_left_comm', 'k')

    @thm(w, 'mul_assoc', [Nat, Nat, Nat], lambda n, m, k: Eq((n * m) * k, n * (m * k)))
    def _(p):
        p.intro('n', 'm', 'k'); p.induct('n', names=['i', 'ih']); p.refl()
        p.rw('add_mul'); p.rw('ih')

    @thm(w, 'mul_left_comm', [Nat, Nat, Nat], lambda n, m, k: Eq(n * (m * k), m * (n * k)))
    def _(p):
        p.intro('n', 'm', 'k'); p.rw('mul_assoc', rev=True); p.rw('mul_comm', 'n', 'm'); p.rw('mul_assoc')

    @thm(w, 'mul_right_comm', [Nat, Nat, Nat], lambda n, m, k: Eq((n * m) * k, (n * k) * m))
    def _(p):
        p.intro('n', 'm', 'k'); p.rw('mul_assoc'); p.rw('mul_comm', 'm', 'k'); p.rw('mul_assoc', rev=True)


def pow_lemmas(w):
    Nat, S, Z = w['Nat'], w['Succ'], w['Zero']
    Eq = w.Eq

    @thm(w, 'pow_zero', [Nat], lambda b: Eq(b ** Z, S(Z)))
    def _(p): p.intro('b'); p.refl()

    @thm(w, 'pow_succ', [Nat, Nat], lambda b, e: Eq(b ** S(e), b * b ** e))
    def _(p): p.intro('b', 'e'); p.refl()

    @thm(w, 'pow_one', [Nat], lambda b: Eq(b ** S(Z), b))
    def _(p): p.intro('b'); p.rw('mul_one')

    @thm(w, 'one_pow', [Nat], lambda e: Eq(S(Z) ** e, S(Z)))
    def _(p): p.intro('e'); p.induct('e', names=['k', 'ih']); p.refl(); p.rw('ih')

    @thm(w, 'zero_pow', [Nat], lambda e: Eq(Z ** S(e), Z))
    def _(p): p.intro('e'); p.refl()

    @thm(w, 'pow_add', [Nat, Nat, Nat], lambda b, m, n: Eq(b ** (m + n), b ** m * b ** n))
    def _(p):
        p.intro('b', 'm', 'n'); p.induct('m', names=['k', 'ih'])
        p.rw('add_zero')
        p.rw('ih'); p.rw('mul_assoc')

    @thm(w, 'mul_pow', [Nat, Nat, Nat], lambda a, b, n: Eq((a * b) ** n, a ** n * b ** n))
    def _(p):
        p.intro('a', 'b', 'n'); p.induct('n', names=['k', 'ih']); p.refl()
        p.rw('ih'); p.rw('mul_assoc'); p.rw('mul_assoc'); p.rw('mul_left_comm', 'b')

    @thm(w, 'pow_mul', [Nat, Nat, Nat], lambda b, m, n: Eq(b ** (m * n), (b ** m) ** n))
    def _(p):
        p.intro('b', 'm', 'n'); p.induct('n', names=['k', 'ih'])
        p.rw('mul_zero')
        p.rw('mul_succ'); p.rw('pow_add'); p.rw('ih')


def arith_lemmas(w):
    Nat, S, Z, F = w['Nat'], w['Succ'], w['Zero'], w['False']
    Eq, pred = w.Eq, w['pred']

    @thm(w, 'add_eq_zero', [Nat, Nat, lambda n, m: Eq(n + m, Z)], lambda n, m, h: Eq(n, Z) & Eq(m, Z))
    def _(p):
        p.intro('n', 'm'); p.cases('n', names=['k'])
        p.intro('h'); p.split(); p.refl(); p.exact('h')
        p.intro('h'); p.exfalso(); p.apply('succ_ne_zero', (w['add'], 'k', 'm')); p.exact('h')

    @thm(w, 'eq_dec', [Nat, Nat], lambda n, m: Eq(n, m) | ~Eq(n, m))
    def _(p):
        p.intro('n'); p.induct('n', names=['k', 'ih']); p.intro('m'); p.cases('m', names=['j'])
        p.left(); p.refl()
        p.right(); p.intro('h'); p.apply('zero_ne_succ', 'j'); p.exact('h')
        p.intro('m'); p.cases('m', names=['j'])
        p.right(); p.intro('h'); p.apply('succ_ne_zero', 'k'); p.exact('h')
        p.obtain(['e', 'e'], ('ih', 'j'))
        p.left(); p.apply('cong_succ'); p.exact('e')
        p.right(); p.intro('h'); p.apply('e'); p.apply('succ_inj'); p.exact('h')


def order_lemmas(w):
    Nat, S, Z, F = w['Nat'], w['Succ'], w['Zero'], w['False']
    Eq, Ex, pred = w.Eq, w['Ex'], w['pred']

    @thm(w, 'le_refl', [Nat], lambda n: n <= n)
    def _(p): p.intro('n'); p.apply('le_n')

    @thm(w, 'le_succ', [Nat], lambda n: n <= S(n))
    def _(p): p.intro('n'); p.apply('le_S'); p.apply('le_n')

    @thm(w, 'zero_le', [Nat], lambda n: Z <= n)
    def _(p): p.intro('n'); p.induct('n', names=['k', 'ih']); p.apply('le_n'); p.apply('le_S'); p.exact('ih')

    @thm(w, 'le_trans', [Nat, Nat, Nat, lambda a, b, c: a <= b, lambda a, b, c, h1: b <= c], lambda a, b, c, h1, h2: a <= c)
    def _(p):
        p.intro('a', 'b', 'c', 'h1', 'h2'); p.induct('h2', names=['x', 'y', 'hxy'], indexed=True)
        p.exact('h1'); p.apply('le_S'); p.exact('hxy')

    @thm(w, 'succ_le_succ', [Nat, Nat, lambda n, m: n <= m], lambda n, m, h: S(n) <= S(m))
    def _(p):
        p.intro('n', 'm', 'h'); p.induct('h', names=['x', 'y', 'ih'], indexed=True)
        p.apply('le_n'); p.apply('le_S'); p.exact('ih')

    @thm(w, 'pred_le', [Nat], lambda n: pred(n) <= n)
    def _(p): p.intro('n'); p.cases('n', names=['k']); p.apply('le_n'); p.apply('le_succ')

    @thm(w, 'le_pred_mono', [Nat, Nat, lambda a, b: a <= b], lambda a, b, h: pred(a) <= pred(b))
    def _(p):
        p.intro('a', 'b', 'h'); p.induct('h', names=['x', 'y', 'ih'], indexed=True)
        p.apply('le_n'); p.apply('le_trans', (pred, 'a'), (pred, 'x'), 'x'); p.exact('ih'); p.apply('pred_le')

    @thm(w, 'le_of_succ_le_succ', [Nat, Nat, lambda n, m: S(n) <= S(m)], lambda n, m, h: n <= m)
    def _(p): p.intro('n', 'm', 'h'); p.apply('le_pred_mono', (S, 'n'), (S, 'm')); p.exact('h')

    @thm(w, 'le_add_right', [Nat, Nat], lambda n, k: n <= n + k)
    def _(p):
        p.intro('n', 'k'); p.induct('k', names=['j', 'ih'])
        p.rw('add_zero'); p.apply('le_n')
        p.rw('add_succ'); p.apply('le_S'); p.exact('ih')

    @thm(w, 'exists_le', [Nat, Nat, lambda n, m: w.Ex(Nat, lambda k: Eq(n + k, m))], lambda n, m, h: n <= m)
    def _(p):
        p.intro('n', 'm', 'h'); p.cases('h', names=['k', 'hk'])
        p.rw('hk', rev=True); p.apply('le_add_right')

    @thm(w, 'le_exists', [Nat, Nat, lambda n, m: n <= m], lambda n, m, h: w.Ex(Nat, lambda k: Eq(n + k, m)))
    def _(p):
        p.intro('n', 'm', 'h'); p.induct('h', names=['x', 'y', 'ih'], indexed=True)
        p.exists(0); p.rw('add_zero')
        p.cases('ih', names=['k', 'hk']); p.exists((S, 'k')); p.rw('add_succ'); p.rw('hk')

    @thm(w, 'le_iff_exists', [Nat, Nat], lambda n, m: w.Iff(n <= m, w.Ex(Nat, lambda k: Eq(n + k, m))))
    def _(p): p.intro('n', 'm'); p.split(); p.exact(('le_exists', 'n', 'm')); p.exact(('exists_le', 'n', 'm'))

    @thm(w, 'le_antisymm', [Nat, Nat, lambda n, m: n <= m, lambda n, m, h1: m <= n], lambda n, m, h1, h2: Eq(n, m))
    def _(p):
        p.intro('n', 'm', 'h1', 'h2')
        p.obtain(['a', 'ha'], ('le_exists', 'n', 'm', 'h1')); p.obtain(['b', 'hb'], ('le_exists', 'm', 'n', 'h2'))
        p.have('hab', lambda p: Eq(p.h('a') + p.h('b'), Z), lambda p: (p.apply('add_left_cancel', 'n'), p.rw('add_assoc', rev=True), p.rw('ha'), p.rw('hb'), p.rw('add_zero')))
        p.obtain(['x', 'y'], ('add_eq_zero', 'a', 'b', 'hab'))
        p.rw('ha', rev=True); p.rw('x'); p.rw('add_zero')

    @thm(w, 'le_total', [Nat, Nat], lambda n, m: (n <= m) | (m <= n))
    def _(p):
        p.intro('n'); p.induct('n', names=['k', 'ih'])
        p.intro('m'); p.left(); p.apply('zero_le')
        p.intro('m'); p.cases('m', names=['j'])
        p.right(); p.apply('zero_le')
        p.obtain(['e', 'e'], ('ih', 'j'))
        p.left(); p.apply('succ_le_succ'); p.exact('e')
        p.right(); p.apply('succ_le_succ'); p.exact('e')

    @thm(w, 'succ_ne_self', [Nat], lambda n: ~Eq(S(n), n))
    def _(p):
        p.intro('n'); p.induct('n', names=['k', 'ih']); p.intro('h'); p.apply('succ_ne_zero', Z); p.exact('h')
        p.intro('h'); p.apply('ih'); p.apply('succ_inj'); p.exact('h')

    @thm(w, 'lt_irrefl', [Nat], lambda n: ~(n < n))
    def _(p):
        p.intro('n', 'h'); p.apply('succ_ne_self', 'n'); p.apply('le_antisymm'); p.exact('h'); p.apply('le_succ')

    @thm(w, 'add_le_add_left', [Nat, Nat, Nat, lambda n, m, k: m <= k], lambda n, m, k, h: n + m <= n + k)
    def _(p):
        p.intro('n', 'm', 'k', 'h'); p.obtain(['a', 'ha'], ('le_exists', 'm', 'k', 'h'))
        p.apply('exists_le'); p.exists('a'); p.rw('add_assoc'); p.rw('ha')

    @thm(w, 'add_le_add_right', [Nat, Nat, Nat, lambda n, m, k: n <= m], lambda n, m, k, h: n + k <= m + k)
    def _(p):
        p.intro('n', 'm', 'k', 'h'); p.rw('add_comm', 'n'); p.rw('add_comm', 'm'); p.apply('add_le_add_left'); p.exact('h')

    @thm(w, 'add_le_add', [Nat, Nat, Nat, Nat, lambda a, b, c, d: a <= b, lambda a, b, c, d, h1: c <= d], lambda a, b, c, d, h1, h2: a + c <= b + d)
    def _(p):
        p.intro('a', 'b', 'c', 'd', 'h1', 'h2')
        p.apply('le_trans', (w['add'], 'a', 'c'), (w['add'], 'b', 'c'), (w['add'], 'b', 'd')); p.apply('add_le_add_right'); p.exact('h1'); p.apply('add_le_add_left'); p.exact('h2')

    @thm(w, 'mul_le_mul_left', [Nat, Nat, Nat, lambda n, m, k: m <= k], lambda n, m, k, h: n * m <= n * k)
    def _(p):
        p.intro('n', 'm', 'k', 'h'); p.obtain(['a', 'ha'], ('le_exists', 'm', 'k', 'h'))
        p.apply('exists_le'); p.exists((w['mul'], 'n', 'a')); p.rw('mul_add', rev=True); p.rw('ha')

    @thm(w, 'mul_le_mul_right', [Nat, Nat, Nat, lambda n, m, k: n <= m], lambda n, m, k, h: n * k <= m * k)
    def _(p):
        p.intro('n', 'm', 'k', 'h'); p.rw('mul_comm', 'n'); p.rw('mul_comm', 'm'); p.apply('mul_le_mul_left'); p.exact('h')

    @thm(w, 'mul_le_mul', [Nat, Nat, Nat, Nat, lambda a, b, c, d: a <= b, lambda a, b, c, d, h1: c <= d], lambda a, b, c, d, h1, h2: a * c <= b * d)
    def _(p):
        p.intro('a', 'b', 'c', 'd', 'h1', 'h2')
        p.apply('le_trans', (w['mul'], 'a', 'c'), (w['mul'], 'b', 'c'), (w['mul'], 'b', 'd')); p.apply('mul_le_mul_right'); p.exact('h1'); p.apply('mul_le_mul_left'); p.exact('h2')


def ring_lemmas(w):
    """the lemmas the `ring` normaliser (ring.py) instantiates: congruence, and one lemma per merge step of sums and products of normal forms. Each is a theorem proved from the commutative-semiring laws above."""
    Nat, S, Z = w['Nat'], w['Succ'], w['Zero']
    Eq = w.Eq
    one = S(Z)

    @thm(w, 'ring_add_congr', [Nat, Nat, Nat, Nat, lambda a, b, c, d: Eq(a, c), lambda a, b, c, d, h1: Eq(b, d)], lambda a, b, c, d, h1, h2: Eq(a + b, c + d))
    def _(p): p.intro('a', 'b', 'c', 'd', 'h1', 'h2'); p.cases('h1'); p.cases('h2'); p.refl()

    @thm(w, 'ring_mul_congr', [Nat, Nat, Nat, Nat, lambda a, b, c, d: Eq(a, c), lambda a, b, c, d, h1: Eq(b, d)], lambda a, b, c, d, h1, h2: Eq(a * b, c * d))
    def _(p): p.intro('a', 'b', 'c', 'd', 'h1', 'h2'); p.cases('h1'); p.cases('h2'); p.refl()

    @thm(w, 'ring_pow_congr', [Nat, Nat, Nat, lambda a, c, e: Eq(a, c)], lambda a, c, e, h: Eq(a ** e, c ** e))
    def _(p): p.intro('a', 'c', 'e', 'h'); p.cases('h'); p.refl()

    @thm(w, 'ring_atom', [Nat], lambda x: Eq(x, x ** one * one + Z))
    def _(p): p.intro('x'); p.rw('pow_one'); p.rw('mul_one'); p.rw('add_zero')

    @thm(w, 'ring_succ', [Nat], lambda x: Eq(S(x), x + one))
    def _(p): p.intro('x'); p.rw('add_succ'); p.rw('add_zero')

    # sums of normal forms (monomials sorted by key)
    @thm(w, 'ring_add_lt', [Nat, Nat, Nat, Nat, lambda a, A, B, C: Eq(A + B, C)], lambda a, A, B, C, h: Eq((a + A) + B, a + C))
    def _(p): p.intro('a', 'A', 'B', 'C', 'h'); p.rw('add_assoc'); p.rw('h')

    @thm(w, 'ring_add_gt', [Nat, Nat, Nat, Nat, lambda b, A, B, C: Eq(A + B, C)], lambda b, A, B, C, h: Eq(A + (b + B), b + C))
    def _(p): p.intro('b', 'A', 'B', 'C', 'h'); p.rw('add_left_comm'); p.rw('h')

    @thm(w, 'ring_add_eq', [Nat, Nat, Nat, Nat, Nat, Nat, lambda a, b, c, A, B, C: Eq(a + b, c), lambda a, b, c, A, B, C, h1: Eq(A + B, C)],
         lambda a, b, c, A, B, C, h1, h2: Eq((a + A) + (b + B), c + C))
    def _(p):
        p.intro('a', 'b', 'c', 'A', 'B', 'C', 'h1', 'h2')
        p.rw('add_assoc'); p.rw('add_left_comm', 'A'); p.rw('add_assoc', rev=True); p.rw('h1'); p.rw('h2')

    @thm(w, 'ring_mono_add', [Nat, Nat, Nat, Nat, Nat, lambda x, e, a, b, s: Eq(a + b, s)], lambda x, e, a, b, s, h: Eq(x ** e * a + x ** e * b, x ** e * s))
    def _(p): p.intro('x', 'e', 'a', 'b', 's', 'h'); p.rw('mul_add', rev=True); p.rw('h')

    # products of monomials  x^e * (y^f * ... * c)
    @thm(w, 'ring_mono_lt', [Nat, Nat, Nat, Nat, Nat, lambda x, e, A, B, R: Eq(A * B, R)], lambda x, e, A, B, R, h: Eq((x ** e * A) * B, x ** e * R))
    def _(p): p.intro('x', 'e', 'A', 'B', 'R', 'h'); p.rw('mul_assoc'); p.rw('h')

    @thm(w, 'ring_mono_gt', [Nat, Nat, Nat, Nat, Nat, lambda y, f, A, B, R: Eq(A * B, R)], lambda y, f, A, B, R, h: Eq(A * (y ** f * B), y ** f * R))
    def _(p): p.intro('y', 'f', 'A', 'B', 'R', 'h'); p.rw('mul_left_comm'); p.rw('h')

    @thm(w, 'ring_mul_mul_mul', [Nat, Nat, Nat, Nat], lambda a, b, c, d: Eq((a * b) * (c * d), (a * c) * (b * d)))
    def _(p): p.intro('a', 'b', 'c', 'd'); p.rw('mul_assoc'); p.rw('mul_left_comm', 'b'); p.rw('mul_assoc', rev=True)

    @thm(w, 'ring_mono_eq', [Nat, Nat, Nat, Nat, Nat, Nat, Nat, lambda x, e, f, g, A, B, R: Eq(e + f, g), lambda x, e, f, g, A, B, R, h1: Eq(A * B, R)],
         lambda x, e, f, g, A, B, R, h1, h2: Eq((x ** e * A) * (x ** f * B), x ** g * R))
    def _(p):
        p.intro('x', 'e', 'f', 'g', 'A', 'B', 'R', 'h1', 'h2')
        p.rw('ring_mul_mul_mul'); p.rw('pow_add', rev=True); p.rw('h1'); p.rw('h2')

    # products of polynomials
    @thm(w, 'ring_scale_nil', [Nat], lambda M: Eq(M * Z, Z))
    def _(p): p.intro('M'); p.rw('mul_zero')

    @thm(w, 'ring_scale_cons', [Nat, Nat, Nat, Nat, Nat, lambda M, N, Q, MN, R: Eq(M * N, MN), lambda M, N, Q, MN, R, h1: Eq(M * Q, R)],
         lambda M, N, Q, MN, R, h1, h2: Eq(M * (N + Q), MN + R))
    def _(p): p.intro('M', 'N', 'Q', 'MN', 'R', 'h1', 'h2'); p.rw('mul_add'); p.rw('h1'); p.rw('h2')

    @thm(w, 'ring_mulp_nil', [Nat], lambda Q: Eq(Z * Q, Z))
    def _(p): p.intro('Q'); p.refl()

    @thm(w, 'ring_mulp_cons', [Nat, Nat, Nat, Nat, Nat, Nat, lambda M, P, Q, A, B, C: Eq(M * Q, A), lambda M, P, Q, A, B, C, h1: Eq(P * Q, B), lambda M, P, Q, A, B, C, h1, h2: Eq(A + B, C)],
         lambda M, P, Q, A, B, C, h1, h2, h3: Eq((M + P) * Q, C))
    def _(p): p.intro('M', 'P', 'Q', 'A', 'B', 'C', 'h1', 'h2', 'h3'); p.rw('add_mul'); p.rw('h1'); p.rw('h2'); p.rw('h3')

    # powers by a numeral
    @thm(w, 'ring_pow_zero', [Nat], lambda P: Eq(P ** Z, one + Z))
    def _(p): p.intro('P'); p.refl()

    @thm(w, 'ring_pow_succ', [Nat, Nat, Nat, Nat, lambda P, k, A, B: Eq(P ** k, A), lambda P, k, A, B, h1: Eq(P * A, B)], lambda P, k, A, B, h1, h2: Eq(P ** S(k), B))
    def _(p): p.intro('P', 'k', 'A', 'B', 'h1', 'h2'); p.rw('pow_succ'); p.rw('h1'); p.rw('h2')


def build_nat():
    w = lnl_base.build_base()
    w.nota = DispatchNotation(w); w.nota.register(w['Nat'], NatNotation(w), default=True)
    declare_arith(w); succ_lemmas(w); add_lemmas(w); mul_lemmas(w); pow_lemmas(w); arith_lemmas(w); order_lemmas(w); ring_lemmas(w)
    return w


if __name__ == '__main__':
    import time
    t0 = time.time(); w = build_nat()
    print('nat library: %d globals in %.2f s' % (w.k.num_globals(), time.time() - t0))
