"""bank_lib.py -- the problems of the bank (see bank.py): olympiad-style statements over the integers, each with its text, source, intended meaning (a Python spec, see semantic.py) and a reference proof.

Conventions.  A statement is written with the authoring layer on top of `imo_lib.build_imo()` (Z, N, divisibility, gcd, primes, induction, ring / linarith / nlinarith); the formal statement says what the text says
and nothing more -- the dataset's answer is part of the statement for "determine all" problems (an equivalence with the solution set).  Every problem has a `spec`: the same statement as Python, written
independently of the formal one; `bank.check` reads the formal statement back (semantic.py) and compares the two on a box of small values, with mutated statements as negative controls.

    1  a single arithmetic step               identities (ring), inequalities (nlinarith with the square that makes them true)
    2  a short proof with one idea            divisibility and parity lemmas: a witness, a case split, one induction
    3  an olympiad lemma                       several ideas / lemmas
    4  an olympiad problem                     the solution is short
    5  a hard olympiad problem
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imo_lib
from nt_lib import _syms
from lnl_base import thm
from semantic import Spec
import bank as B
from bank import BANK

W_IMO = 'IMO'


def base(w=None): return imo_lib.build_imo(w)


def load(only=None):
    return B.load(base=base, bank=BANK, only=only)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ problems the library already proves
BANK.declare('imo1959_p1', text='Prove that the fraction (21n+4)/(14n+3) is irreducible for every natural number n.', source='IMO 1959 P1', year=1959, topics=('number-theory', 'gcd'), level=2,
             spec=Spec(('Z',), (lambda n: n >= 0,), lambda n: math.gcd(21 * n + 4, 14 * n + 3) == 1, atom=(lambda n: 21 * n + 4, lambda n: 14 * n + 3, lambda n: 1)),
             notes='irreducible = the greatest common divisor is 1; natural number = integer >= 0 (the statement is true for every integer)')

BANK.declare('sqrt2_irrational', text='Prove that sqrt(2) is irrational: there are no integers p, q with q /= 0 and p^2 = 2 q^2.', source='Euclid', topics=('number-theory', 'descent', 'parity'), level=3,
             helpers=('Z.sqrt2_aux',), strict=False,
             spec=Spec(('Z', 'Z'), (lambda p, q: q != 0,), lambda p, q: p * p != 2 * q * q, atom=(lambda p, q: p * p, lambda p, q: 2 * q * q)),
             notes='irrational = not a quotient of two integers with a non-zero denominator; the reference proof is a descent by strong induction on q')


def _imo2019_cond(f, W=range(-8, 9)): return all(f(2 * a) + 2 * f(b) == f(f(a + b)) for a in W for b in W)


def _imo2019_sol(f, W=range(-8, 9)): return all(f(n) == 0 for n in W) or any(all(f(n) == 2 * n + c for n in W) for c in W)


def _imo2019_partial(n):
    """satisfies  f(2a) + 2 f(b) = f(f(a+b))  for a = 0 and every b, but not for every a: it tells `for all a` from `for some a`"""
    if n == 0: return 0
    if n == 1: return 2
    return 2 * n if n > 1 and (n & (n - 1)) == 0 else 0


BANK.declare('imo2019_p1', text='Let Z be the set of integers. Determine all functions f : Z -> Z such that f(2a) + 2f(b) = f(f(a+b)) for all integers a and b.', source='IMO 2019 P1', year=2019,
             topics=('functional-equation', 'induction'), level=4, kind='answer', answer='f(n) = 0 for all n, and f(n) = 2n + c for all n, for any integer c',
             helpers=('imo2019_p1_fwd', 'imo2019_p1_bwd'), strict=False,
             spec=Spec(('fn',), (), (_imo2019_cond, _imo2019_sol), extra_fns=(_imo2019_partial,),
                       skip_mutants={'Z.add -> Z.sub in ((x0 x2) = (Z.add (Z.mul 2 x2) x1))': 'f(n) = 2n - c is f(n) = 2n + c with c replaced by -c: the existential ranges over a symmetric window'}),
             notes='formalised as an equivalence: f satisfies the equation iff f is the zero function or f(n) = 2n + c for some c')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ level 1: identities
@BANK.problem('ident_diff_squares', text='For all integers a and b: (a + b)(a - b) = a^2 - b^2.', source='folklore', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z', 'Z'), (), lambda a, b: (a + b) * (a - b) == a * a - b * b, atom=(lambda a, b: (a + b) * (a - b), lambda a, b: a ** 2 - b ** 2)))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_diff_squares', [Z, Z], lambda a, c: Eq((a + c) * (a - c), a ** 2 - c ** 2))
        def _(p): p.intro('a', 'c'); p.ring()


@BANK.problem('ident_sophie_germain', text='For all integers a and b: a^4 + 4 b^4 = (a^2 + 2ab + 2b^2)(a^2 - 2ab + 2b^2).', source='folklore (Sophie Germain)', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z', 'Z'), (), lambda a, b: a ** 4 + 4 * b ** 4 == (a * a + 2 * a * b + 2 * b * b) * (a * a - 2 * a * b + 2 * b * b),
                        atom=(lambda a, b: a ** 4 + 4 * b ** 4, lambda a, b: (a * a + 2 * a * b + 2 * b * b) * (a * a - 2 * a * b + 2 * b * b))))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_sophie_germain', [Z, Z], lambda a, c: Eq(a ** 4 + 4 * c ** 4, (a ** 2 + 2 * a * c + 2 * c ** 2) * (a ** 2 - 2 * a * c + 2 * c ** 2)))
        def _(p): p.intro('a', 'c'); p.ring()


@BANK.problem('ident_brahmagupta', text='For all integers a, b, c, d: (a^2 + b^2)(c^2 + d^2) = (ac - bd)^2 + (ad + bc)^2.', source='folklore (Brahmagupta-Fibonacci)', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z', 'Z', 'Z', 'Z'), (), lambda a, b, c, d: (a * a + b * b) * (c * c + d * d) == (a * c - b * d) ** 2 + (a * d + b * c) ** 2,
                        atom=(lambda a, b, c, d: (a * a + b * b) * (c * c + d * d), lambda a, b, c, d: (a * c - b * d) ** 2 + (a * d + b * c) ** 2), windows={'z': 3},
                        skip_mutants={'swap the arguments of Z.sub in': 'a square does not see the sign of its base: (ac - bd)^2 = (bd - ac)^2'}))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_brahmagupta', [Z, Z, Z, Z], lambda a, c, d, e: Eq((a ** 2 + c ** 2) * (d ** 2 + e ** 2), (a * d - c * e) ** 2 + (a * e + c * d) ** 2))
        def _(p): p.intro('a', 'c', 'd', 'e'); p.ring()


@BANK.problem('ident_consecutive_four', text='For every integer n: n(n+1)(n+2)(n+3) + 1 = (n^2 + 3n + 1)^2.', source='folklore', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z',), (), lambda n: n * (n + 1) * (n + 2) * (n + 3) + 1 == (n * n + 3 * n + 1) ** 2,
                        atom=(lambda n: n * (n + 1) * (n + 2) * (n + 3) + 1, lambda n: (n * n + 3 * n + 1) ** 2)))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_consecutive_four', [Z], lambda n: Eq(n * (n + 1) * (n + 2) * (n + 3) + 1, (n ** 2 + 3 * n + 1) ** 2))
        def _(p): p.intro('n'); p.ring()


@BANK.problem('ident_fourth_powers', text='For all integers a and b: a^4 + b^4 + (a + b)^4 = 2 (a^2 + ab + b^2)^2.', source='folklore', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z', 'Z'), (), lambda a, b: a ** 4 + b ** 4 + (a + b) ** 4 == 2 * (a * a + a * b + b * b) ** 2,
                        atom=(lambda a, b: a ** 4 + b ** 4 + (a + b) ** 4, lambda a, b: 2 * (a * a + a * b + b * b) ** 2)))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_fourth_powers', [Z, Z], lambda a, c: Eq(a ** 4 + c ** 4 + (a + c) ** 4, 2 * (a ** 2 + a * c + c ** 2) ** 2))
        def _(p): p.intro('a', 'c'); p.ring()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ level 1: inequalities
@BANK.problem('ineq_am_gm_two', text='For all integers a and b: 2ab <= a^2 + b^2.', source='folklore (AM-GM)', topics=('inequality',), level=1,
              spec=Spec(('Z', 'Z'), (), lambda a, b: 2 * a * b <= a * a + b * b, atom=(lambda a, b: 2 * a * b, lambda a, b: a * a + b * b)))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_am_gm_two', [Z, Z], lambda a, c: zle(2 * a * c, a ** 2 + c ** 2))
        def _(p): p.intro('a', 'c'); p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - q.h('c')))


@BANK.problem('ineq_square_sum_three', text='For all integers a, b, c: ab + bc + ca <= a^2 + b^2 + c^2.', source='folklore', topics=('inequality',), level=1,
              spec=Spec(('Z', 'Z', 'Z'), (), lambda a, b, c: a * b + b * c + c * a <= a * a + b * b + c * c, atom=(lambda a, b, c: a * b + b * c + c * a, lambda a, b, c: a * a + b * b + c * c)))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_square_sum_three', [Z, Z, Z], lambda a, c, d: zle(a * c + c * d + d * a, a ** 2 + c ** 2 + d ** 2))
        def _(p):
            p.intro('a', 'c', 'd')
            p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - q.h('c')), ('Z.sq_nonneg', lambda q: q.h('c') - q.h('d')), ('Z.sq_nonneg', lambda q: q.h('d') - q.h('a')))


@BANK.problem('ineq_four_ab', text='For all integers a and b: 4ab <= (a + b)^2.', source='folklore', topics=('inequality',), level=1,
              spec=Spec(('Z', 'Z'), (), lambda a, b: 4 * a * b <= (a + b) ** 2, atom=(lambda a, b: 4 * a * b, lambda a, b: (a + b) ** 2)))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_four_ab', [Z, Z], lambda a, c: zle(4 * a * c, (a + c) ** 2))
        def _(p): p.intro('a', 'c'); p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - q.h('c')))


@BANK.problem('ineq_cauchy_two', text='For all integers a, b, c, d: (ac + bd)^2 <= (a^2 + b^2)(c^2 + d^2).', source='folklore (Cauchy-Schwarz)', topics=('inequality',), level=1,
              spec=Spec(('Z', 'Z', 'Z', 'Z'), (), lambda a, b, c, d: (a * c + b * d) ** 2 <= (a * a + b * b) * (c * c + d * d),
                        atom=(lambda a, b, c, d: (a * c + b * d) ** 2, lambda a, b, c, d: (a * a + b * b) * (c * c + d * d)), windows={'z': 3}))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_cauchy_two', [Z, Z, Z, Z], lambda a, c, d, e: zle((a * d + c * e) ** 2, (a ** 2 + c ** 2) * (d ** 2 + e ** 2)))
        def _(p): p.intro('a', 'c', 'd', 'e'); p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') * q.h('e') - q.h('c') * q.h('d')))


@BANK.problem('sum_squares_zero', text='If a and b are integers with a^2 + b^2 = 0, then a = 0 and b = 0.', source='folklore', topics=('algebra', 'inequality'), level=1,
              spec=Spec(('Z', 'Z'), (lambda a, b: a * a + b * b == 0,), lambda a, b: a == 0 and b == 0))
def _(w, b):
    s = _syms(w); Z, Eq, zzero = s.Z, s.Eq, s.zzero
    with w.nota.using(Z):
        @thm(w, 'sum_squares_zero', [Z, Z, lambda a, c: Eq(a ** 2 + c ** 2, zzero)], lambda a, c, h: Eq(a, zzero) & Eq(c, zzero))
        def _(p):
            p.intro('a', 'c', 'h')
            p.have('ha', p.T('a', lambda a_: Eq(a_ * a_, zzero)), lambda q: q.nlinarith(('Z.sq_nonneg', q.h('a')), ('Z.sq_nonneg', q.h('c'))))
            p.have('hc', p.T('c', lambda c_: Eq(c_ * c_, zzero)), lambda q: q.nlinarith(('Z.sq_nonneg', q.h('a')), ('Z.sq_nonneg', q.h('c'))))
            p.split()
            p.obtain(['u', 'u'], ('Z.mul_eq_zero', 'a', 'a', 'ha')); p.exact('u'); p.exact('u')
            p.obtain(['v', 'v'], ('Z.mul_eq_zero', 'c', 'c', 'hc')); p.exact('v'); p.exact('v')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ level 2: divisibility and parity
def destruct(p, fact, names):
    r"""`obtain` for a fact that is a chain  exists x, exists y, A /\ B /\ C ..  (nested to the right): one name per component, in order"""
    if len(names) == 2: return p.obtain(names, fact)
    p.obtain([names[0], '_d1'], fact)
    for i in range(1, len(names) - 2): p.cases('_d%d' % i, names=[names[i], '_d%d' % (i + 1)])
    p.cases('_d%d' % (len(names) - 2), names=[names[-2], names[-1]])


def _dvd(a, b): return (a == 0 and b == 0) or (a != 0 and b % a == 0)


def AND(*ps):
    """p1 /\\ (p2 /\\ (.. /\\ pn))"""
    r = ps[-1]
    for q in reversed(ps[:-1]): r = q & r
    return r


@BANK.problem('not_dvd_between', text='Let d > 0 and k be integers. No multiple of d lies strictly between dk and d(k+1): if dk < x < d(k+1) then d does not divide x.', source='folklore', topics=('number-theory', 'divisibility'), level=2,
              spec=Spec(('Z', 'Z', 'Z'), (lambda d, k, x: d > 0, lambda d, k, x: d * k < x, lambda d, k, x: x < d * (k + 1)), lambda d, k, x: not _dvd(d, x), windows={'z': 6},
                        atom=(lambda d, k, x: d, lambda d, k, x: x)))
def _(w, b):
    s = _syms(w); Z, zlt, zzero, D = s.Z, s.zlt, s.zzero, s.D
    with w.nota.using(Z):
        @thm(w, 'not_dvd_between', [Z, Z, Z, lambda d, k, x: zlt(zzero, d), lambda d, k, x, h: zlt(d * k, x), lambda d, k, x, h, h1: zlt(x, d * (k + 1))], lambda d, k, x, h, h1, h2: ~D(d, x))
        def _(p):
            p.intro('d', 'k', 'x', 'hd', 'h1', 'h2', 'hx')
            p.cases('hx', names=['m', 'hm'])
            p.obtain(['c', 'c'], ('Z.le_or_lt', 'm', 'k'))
            p.nlinarith(); p.nlinarith()


@BANK.problem('square_mod_four', text='The square of every integer is of the form 4k or 4k + 1.', source='folklore', topics=('number-theory', 'modular', 'parity'), level=2,
              spec=Spec(('Z',), (), lambda n: any(n * n == 4 * k or n * n == 4 * k + 1 for k in range(0, 40)), windows={'z': 6, 'ex': 40},
                        body=(('Z',), lambda n, k: n * n == 4 * k or n * n == 4 * k + 1),
                        atoms=[(lambda n, k: n * n, lambda n, k: 4 * k), (lambda n, k: n * n, lambda n, k: 4 * k + 1)]))
def _(w, b):
    s = _syms(w); Z, Eq, Ex, zzero, zlt = s.Z, s.Eq, s.Ex, s.zzero, s.zlt
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'square_mod_four', [Z], lambda n: Ex(Z, w.lam(Z, lambda k: Eq(n ** 2, 4 * k) | Eq(n ** 2, 4 * k + 1))))
        def _(p):
            p.intro('n')
            p.have('h2', p.T('', lambda: zlt(zzero, lit(2))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(2), 'h2', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 1)
            p.rw('r_eq'); p.exists(p.T('q', lambda q: q ** 2)); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: q ** 2 + q)); p.right(); p.ring()


@BANK.problem('sum_two_squares_mod4', text='No integer of the form 4k + 3 is a sum of two squares: a^2 + b^2 /= 4k + 3 for all integers a, b, k.', source='folklore', topics=('number-theory', 'modular', 'diophantine'), level=2,
              spec=Spec(('Z', 'Z', 'Z'), (), lambda a, b, k: a * a + b * b != 4 * k + 3, windows={'z': 6}, atom=(lambda a, b, k: a * a + b * b, lambda a, b, k: 4 * k + 3)))
def _(w, b):
    s = _syms(w); Z, Eq, zzero = s.Z, s.Eq, s.zzero
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'sum_two_squares_mod4', [Z, Z, Z], lambda a, c, k: ~Eq(a ** 2 + c ** 2, 4 * k + 3))
        def _(p):
            p.intro('a', 'c', 'k', 'h')
            p.obtain(['ka', 'ha'], ('square_mod_four', 'a')); p.obtain(['kc', 'hc'], ('square_mod_four', 'c'))
            def close(x):                       # a^2 + c^2 = 4 (ka + kc) + x  and  = 4 k + 3: 4 (ka + kc - k) = 3 - x with 0 < 3 - x < 4
                p.apply('not_dvd_between', lit(4), lit(0), lit(x))
                p.linarith(); p.linarith(); p.linarith()
                p.exists(p.T('ka kc k', lambda a_, c_, k_: a_ + c_ - k_)); p.linarith()
            p.cases('ha', names=['ha', 'ha'])
            p.cases('hc', names=['hc', 'hc']); close(3); close(2)
            p.cases('hc', names=['hc', 'hc']); close(2); close(1)


@BANK.problem('odd_square_mod_eight', text='The square of an odd integer leaves remainder 1 upon division by 8.', source='folklore', topics=('number-theory', 'modular', 'parity'), level=2,
              spec=Spec(('Z',), (lambda n: n % 2 == 1,), lambda n: any(n * n == 8 * m + 1 for m in range(-5, 40)), windows={'z': 8, 'ex': 40},
                        body=(('Z',), lambda n, m: n * n == 8 * m + 1),
                        atoms=[(lambda n, m: n * n, lambda n, m: 8 * m + 1)],
                        skip_mutants={'in (x0 = (Z.add (Z.mul 2 x1) 1))': 'the hypothesis n = 2k + 1, n = -2k + 1, n = 2k - 1, n = -2k - 1 all say that n is odd (k -> -k, k -> k + 1); the hypothesis is compared as a truth table'}))
def _(w, b):
    s = _syms(w); Z, Eq, Ex, zzero, zlt = s.Z, s.Eq, s.Ex, s.zzero, s.zlt
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'odd_square_mod_eight', [Z, lambda n: Ex(Z, w.lam(Z, lambda k: Eq(n, 2 * k + 1)))], lambda n, h: Ex(Z, w.lam(Z, lambda m: Eq(n ** 2, 8 * m + 1))))
        def _(p):
            p.intro('n', 'h'); p.cases('h', names=['k', 'hk']); p.rw('hk')
            p.have('h2', p.T('', lambda: zlt(zzero, lit(2))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(2), 'h2', 'k'), ['j', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 1)
            p.rw('r_eq'); p.exists(p.T('j', lambda j: 2 * j ** 2 + j)); p.ring()
            p.rw('r_eq'); p.exists(p.T('j', lambda j: 2 * j ** 2 + 3 * j + 1)); p.ring()


@BANK.problem('cube_mod_nine', text='The cube of every integer is of the form 9k, 9k + 1 or 9k + 8.', source='folklore', topics=('number-theory', 'modular'), level=2,
              spec=Spec(('Z',), (), lambda n: any(n ** 3 in (9 * k, 9 * k + 1, 9 * k + 8) for k in range(-80, 81)), windows={'z': 8, 'ex': 80},
                        body=(('Z',), lambda n, k: n ** 3 in (9 * k, 9 * k + 1, 9 * k + 8)),
                        atoms=[(lambda n, k: n ** 3, lambda n, k: 9 * k), (lambda n, k: n ** 3, lambda n, k: 9 * k + 1), (lambda n, k: n ** 3, lambda n, k: 9 * k + 8)]))
def _(w, b):
    s = _syms(w); Z, Eq, Ex, zzero, zlt = s.Z, s.Eq, s.Ex, s.zzero, s.zlt
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'cube_mod_nine', [Z], lambda n: Ex(Z, w.lam(Z, lambda k: Eq(n ** 3, 9 * k) | (Eq(n ** 3, 9 * k + 1) | Eq(n ** 3, 9 * k + 8)))))
        def _(p):
            p.intro('n')
            p.have('h3', p.T('', lambda: zlt(zzero, lit(3))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 2)
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 3)); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 3 + 3 * q ** 2 + q)); p.right(); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 3 + 6 * q ** 2 + 4 * q)); p.right(); p.right(); p.ring()


@BANK.problem('cube_minus_n_div6', text='For every integer n, 6 divides n^3 - n.', source='folklore', topics=('number-theory', 'divisibility', 'modular'), level=2,
              spec=Spec(('Z',), (), lambda n: _dvd(6, n ** 3 - n), atom=(lambda n: 6, lambda n: n ** 3 - n)))
def _(w, b):
    s = _syms(w); Z, zzero, zlt, D = s.Z, s.zzero, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'cube_minus_n_div6', [Z], lambda n: D(lit(6), n ** 3 - n))
        def _(p):
            p.intro('n')
            p.have('h6', p.T('', lambda: zlt(zzero, lit(6))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(6), 'h6', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 5)
            for k in range(6):                      # n = 6q + k:  n^3 - n = 6 (36 q^3 + 18 k q^2 + 3 k^2 q - q + (k^3 - k) / 6)
                p.rw('r_eq')
                p.exists(p.T('q', lambda q, k=k: lit(36) * q ** 3 + lit(18 * k) * q ** 2 + lit(3 * k * k) * q - q + lit((k ** 3 - k) // 6)))
                p.ring()


@BANK.problem('pow_sub_dvd', text='For integers a, b and every natural number n, a - b divides a^n - b^n.', source='folklore', topics=('number-theory', 'divisibility', 'induction'), level=2,
              spec=Spec(('Z', 'Z', 'N'), (), lambda a, b, n: _dvd(a - b, a ** n - b ** n), windows={'z': 6, 'n': 6}, atom=(lambda a, b, n: a - b, lambda a, b, n: a ** n - b ** n)))
def _(w, b):
    s = _syms(w); Z, N, zzero, D = s.Z, s.N, s.zzero, s.D
    with w.nota.using(Z):
        @thm(w, 'pow_sub_dvd', [Z, Z, N], lambda a, c, n: D(a - c, a ** n - c ** n))
        def _(p):
            p.intro('a', 'c', 'n')
            p.induction_on('N.induction_on', 'n')
            p.exists(zzero); p.ring()
            p.intro('m', 'hm'); p.cases('hm', names=['k', 'hk'])
            p.exists(p.T('a c m k', lambda a_, c_, m_, k_: a_ * k_ + c_ ** m_))             # a^(m+1) - c^(m+1) = a (a^m - c^m) + c^m (a - c)
            p.rw('Z.pow_succ'); p.rw('Z.pow_succ')
            p.linear_combination((p.T('a', lambda a_: a_), 'hk'))


@BANK.problem('coprime_consecutive', text='Consecutive integers are coprime: gcd(n, n + 1) = 1 for every integer n.', source='folklore', topics=('number-theory', 'gcd'), level=2,
              spec=Spec(('Z',), (), lambda n: math.gcd(n, n + 1) == 1, atom=(lambda n: n, lambda n: n + 1, lambda n: 1)))
def _(w, b):
    s = _syms(w); Z, zone = s.Z, s.zone
    zgcd = w['Z.is_gcd']
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'coprime_consecutive', [Z], lambda n: zgcd(n, n + 1, zone))
        def _(p):
            p.intro('n'); p.apply('Z.is_gcd_one_of_coprime')
            p.exists(-lit(1)); p.exists(lit(1)); p.ring()


@BANK.problem('no_square_between', text='There is no perfect square strictly between n^2 and (n + 1)^2: for integers n >= 0 and m, it is not the case that n^2 < m^2 < (n + 1)^2.', source='folklore', topics=('number-theory', 'bounding'), level=2,
              spec=Spec(('Z', 'Z'), (lambda n, m: n >= 0,), lambda n, m: not (n * n < m * m < (n + 1) ** 2),
                        atoms=[(lambda n, m: n * n, lambda n, m: m * m), (lambda n, m: m * m, lambda n, m: (n + 1) ** 2)]))
def _(w, b):
    s = _syms(w); Z, zle, zlt, zzero = s.Z, s.zle, s.zlt, s.zzero
    with w.nota.using(Z):
        @thm(w, 'no_square_between', [Z, Z, lambda n, m: zle(zzero, n)], lambda n, m, h: ~(zlt(n ** 2, m ** 2) & zlt(m ** 2, (n + 1) ** 2)))
        def _(p):
            p.intro('n', 'm', 'hn', 'h'); p.cases('h', names=['h1', 'h2'])
            p.obtain(['c', 'c'], ('Z.le_or_lt', zzero, 'm'))
            p.obtain(['d', 'd'], ('Z.le_or_lt', 'm', 'n')); p.nlinarith(); p.nlinarith()
            p.obtain(['d', 'd'], ('Z.le_or_lt', (s.zneg, 'm'), 'n')); p.nlinarith(); p.nlinarith()


@BANK.problem('three_dvd_sum_squares', text='If 3 divides a^2 + b^2 for integers a and b, then 3 divides both a and b.', source='folklore', topics=('number-theory', 'divisibility', 'modular'), level=3,
              spec=Spec(('Z', 'Z'), (lambda a, b: _dvd(3, a * a + b * b),), lambda a, b: _dvd(3, a) and _dvd(3, b), atoms=[(lambda a, b: 3, lambda a, b: a), (lambda a, b: 3, lambda a, b: b)],
                        skip_mutants={'constant 3 -> -3 in (Z.dvd 3': 'a divisor and its negative divide the same integers'}))
def _(w, b):
    s = _syms(w); Z, Eq, zzero, zlt, D = s.Z, s.Eq, s.zzero, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'three_dvd_sum_squares', [Z, Z, lambda a, c: D(lit(3), a ** 2 + c ** 2)], lambda a, c, h: D(lit(3), a) & D(lit(3), c))
        def _(p):
            p.intro('a', 'c', 'h'); p.cases('h', names=['k', 'hk'])
            p.have('h3', p.T('', lambda: zlt(zzero, lit(3))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'a'), ['q', 'r', 'e1', 'h10', 'h11'])
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'c'), ['t', 'u', 'e2', 'h20', 'h21'])
            p.rw_at('hk', 'e1'); p.rw_at('hk', 'e2')
            p.interval_cases('r', 0, 2)
            for rv in range(3):
                p.interval_cases('u', 0, 2)
                for uv in range(3):
                    p.rw_at('hk', 'r_eq'); p.rw_at('hk', 'u_eq')
                    if rv == 0 and uv == 0:
                        p.split()
                        p.exists('q'); p.rw('e1'); p.rw('r_eq'); p.ring()
                        p.exists('t'); p.rw('e2'); p.rw('u_eq'); p.ring()
                    else:
                        x = rv * rv + uv * uv; k0 = x // 3
                        p.exfalso()
                        p.apply('not_dvd_between', lit(3), lit(k0), lit(x))
                        p.linarith(); p.linarith(); p.linarith()
                        p.exists(p.T('k q t', lambda k_, q_, t_: k_ - 3 * q_ ** 2 - 2 * rv * q_ - 3 * t_ ** 2 - 2 * uv * t_))
                        p.linear_combination((1, 'hk'))


@BANK.problem('N.induction3', text='Induction in steps of three: if P(0), P(1) and P(2) hold, and P(m) implies P(m + 3) for every natural number m, then P(n) holds for every natural number n.', source='folklore',
              topics=('induction', 'logic'), level=2)
def _(w, b):
    s = _syms(w); N, PROP, F = s.N, s.PROP, s.F
    N0, nsucc = w['N0'], w['N.succ']
    PN = w.arrow(N, PROP)
    def S3(m): return nsucc(nsucc(nsucc(m)))
    @thm(w, 'N.induction3', [PN, N, lambda P, n: P(N0), lambda P, n, h0: P(nsucc(N0)), lambda P, n, h0, h1: P(nsucc(nsucc(N0))),
                              lambda P, n, h0, h1, h2: F([N, lambda m: P(m)], lambda m, hm: P(S3(m)))],
         lambda P, n, h0, h1, h2, hs: P(n))
    def _(p):
        p.intro('P', 'n', 'h0', 'h1', 'h2', 'hs')
        st = p.T('P', lambda P: F([N], lambda k: P(k) & (P(nsucc(k)) & P(nsucc(nsucc(k))))))
        def main(q):
            q.intro('k'); q.induction_on('N.induction_on', 'k')
            q.split(); q.exact('h0'); q.split(); q.exact('h1'); q.exact('h2')
            q.intro('m', 'hm'); q.cases('hm', names=['a', 'bc']); q.cases('bc', names=['b', 'c'])
            q.split(); q.exact('b'); q.split(); q.exact('c'); q.apply('hs'); q.exact('a')
        p.have('hq', st, main)
        p.obtain(['a', 'bc'], ('hq', 'n')); p.exact('a')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ level 3: olympiad problems
@BANK.problem('imo1984_p2', text='Find two positive integers a and b such that none of the numbers a, b, a + b is divisible by 7 and (a + b)^7 - a^7 - b^7 is divisible by 7^7.', source='IMO 1984 P2', year=1984,
              topics=('number-theory', 'divisibility', 'construction'), level=3, kind='exists', answer='a = 18, b = 1 (and the pairs equivalent to it modulo 7^3)',
              spec=Spec((), (), lambda: any(a > 0 and b > 0 and a % 7 != 0 and b % 7 != 0 and (a + b) % 7 != 0 and ((a + b) ** 7 - a ** 7 - b ** 7) % 7 ** 7 == 0 for a in range(-22, 23) for b in range(-22, 23)),
                        windows={'ex': 22, 'atom': 3},
                        body=(('Z', 'Z'), lambda a, b: a > 0 and b > 0 and a % 7 != 0 and b % 7 != 0 and (a + b) % 7 != 0 and ((a + b) ** 7 - a ** 7 - b ** 7) % 7 ** 7 == 0),
                        atoms=[(lambda a, b: 0, lambda a, b: a), (lambda a, b: 0, lambda a, b: b), (lambda a, b: 7, lambda a, b: a), (lambda a, b: 7, lambda a, b: b), (lambda a, b: 7, lambda a, b: a + b),
                               (lambda a, b: 7 ** 7, lambda a, b: (a + b) ** 7 - a ** 7 - b ** 7)],
                        skip_mutants={'constant 7 -> -7 in (Z.dvd 7': 'a divisor and its negative divide the same integers'}))
def _(w, b):
    s = _syms(w); Z, Ex, zzero, zlt, D = s.Z, s.Ex, s.zzero, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        def body(a, c): return AND(zlt(zzero, a), zlt(zzero, c), ~D(lit(7), a), ~D(lit(7), c), ~D(lit(7), a + c), D(lit(7) ** 7, (a + c) ** 7 - a ** 7 - c ** 7))
        @thm(w, 'imo1984_p2', [], lambda: Ex(Z, w.lam(Z, lambda a: Ex(Z, w.lam(Z, lambda c: body(a, c))))))
        def _(p):
            p.exists(lit(18)); p.exists(lit(1))
            p.split(); p.linarith()
            p.split(); p.linarith()
            for x, k0 in ((18, 2), (1, 0), (19, 2)):
                p.split()
                p.intro('h'); p.apply('not_dvd_between', lit(7), lit(k0), lit(x)); p.linarith(); p.linarith(); p.linarith(); p.exact('h')
            p.exists(lit(342)); p.ring()


def _prime(n):
    if n < 2: return False
    d = 2
    while d * d <= n:
        if n % d == 0: return False
        d += 1
    return True


@BANK.problem('imo1969_p1', text='Prove that there exist infinitely many positive integers a such that n^4 + a is not a prime number for any positive integer n.', source='IMO 1969 P1', year=1969,
              topics=('number-theory', 'prime', 'construction', 'identity'), level=3, helpers=('not_prime_mul', 'germain_not_prime'),
              spec=Spec(('Z',), (lambda m: m >= 0,), lambda m: any(a > m and all(not _prime(n ** 4 + a) for n in range(1, 9)) for a in range(-70, 71)), windows={'z': 8, 'ex': 70, 'atom': 3},
                        body=(('Z',), lambda m, a: a > m and all(not _prime(n ** 4 + a) for n in range(1, 9))),
                        atoms=[(lambda m, a: m, lambda m, a: a), (lambda m, a, n: 0, lambda m, a, n: n), (lambda m, a, n: n ** 4 + a,)]))
def _(w, b):
    s = _syms(w); Z, Ex, F, zzero, zone, zle, zlt, D = s.Z, s.Ex, s.F, s.zzero, s.zone, s.zle, s.zlt, s.D
    prime = w['Z.prime']
    with w.nota.using(Z):
        lit = w.nota.lit

        @thm(w, 'not_prime_mul', [Z, Z, lambda a, c: zlt(zone, a), lambda a, c, h: zlt(zone, c)], lambda a, c, h1, h2: ~prime(a * c))
        def not_prime_mul_script(p):
            p.intro('a', 'c', 'h1', 'h2', 'hp'); p.cases('hp', names=['g1', 'g2'])
            p.have('ha', p.T('a', lambda a_: zlt(zzero, a_)), lambda q: q.linarith())
            p.obtain(['e', 'e'], ('g2', 'a', 'ha', ('Z.dvd_mul_right', 'a', 'c')))
            p.linarith()
            p.nlinarith()

        @thm(w, 'germain_not_prime', [Z, Z, lambda n, k: zlt(zzero, n), lambda n, k, h: zle(lit(2), k)], lambda n, k, h1, h2: ~prime(n ** 4 + 4 * k ** 4))
        def germain_script(p):
            p.intro('n', 'k', 'hn', 'hk')
            p.have('e', p.T('n k', lambda n_, k_: w.Eq(n_ ** 4 + 4 * k_ ** 4, (n_ ** 2 + 2 * n_ * k_ + 2 * k_ ** 2) * (n_ ** 2 - 2 * n_ * k_ + 2 * k_ ** 2))), lambda q: q.ring())
            p.rw('e')
            p.apply('not_prime_mul')
            p.nlinarith()
            p.nlinarith(('Z.sq_nonneg', lambda q: q.h('n') - q.h('k')))

        @thm(w, 'imo1969_p1', [Z, lambda m: zle(zzero, m)], lambda m, h: Ex(Z, w.lam(Z, lambda a: zlt(m, a) & F([Z, lambda n: zlt(zzero, n)], lambda n, hn: ~prime(n ** 4 + a)))))
        def _(p):
            p.have('not_prime_mul', w.type_of(w['not_prime_mul']), not_prime_mul_script)
            p.have('germain_not_prime', w.type_of(w['germain_not_prime']), germain_script)
            p.intro('m', 'hm')
            p.exists(p.T('m', lambda m_: 4 * (m_ + 2) ** 4))
            p.split(); p.nlinarith(('Z.sq_nonneg', 'm'), ('Z.sq_nonneg', p.T('m', lambda m_: m_ * m_)), ('Z.mul_nonneg', 'm', p.T('m', lambda m_: m_ * m_), 'hm', ('Z.sq_nonneg', 'm')))
            p.intro('n', 'hn')
            p.apply('germain_not_prime'); p.exact('hn'); p.linarith()


@BANK.problem('imo1964_p1b', text='Prove that there is no positive integer n for which 2^n + 1 is divisible by 7.', source='IMO 1964 P1 (b)', year=1964,
              topics=('number-theory', 'divisibility', 'modular', 'induction'), level=3, helpers=('pow2_mod7',),
              spec=Spec(('N',), (lambda n: n > 0,), lambda n: (2 ** n + 1) % 7 != 0, windows={'n': 16}, atom=(lambda n: 7, lambda n: 2 ** n + 1)),
              notes='the powers of 2 leave the remainders 1, 2, 4 upon division by 7 (the lemma pow2_mod7); positivity of n is not needed')
def _(w, b):
    s = _syms(w); Z, N, Eq, Ex, zzero, D = s.Z, s.N, s.Eq, s.Ex, s.zzero, s.D
    N0 = w['N0']; nlt = w['N.lt']
    with w.nota.using(Z):
        lit = w.nota.lit

        @thm(w, 'pow2_mod7', [N], lambda n: Ex(Z, w.lam(Z, lambda k: Eq(lit(2) ** n, 7 * k + 1) | (Eq(lit(2) ** n, 7 * k + 2) | Eq(lit(2) ** n, 7 * k + 4)))))
        def pow2_mod7_script(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            p.exists(zzero); p.left(); p.ring()
            p.intro('m', 'hm'); p.cases('hm', names=['k', 'hk']); p.rw('Z.pow_succ')
            p.cases('hk', names=['c1', 'c2'])
            p.exists(p.T('k', lambda k: 2 * k)); p.right(); p.left(); p.linear_combination((2, 'c1'))
            p.cases('c2', names=['c3', 'c4'])
            p.exists(p.T('k', lambda k: 2 * k)); p.right(); p.right(); p.linear_combination((2, 'c3'))
            p.exists(p.T('k', lambda k: 2 * k + 1)); p.left(); p.linear_combination((2, 'c4'))

        @thm(w, 'imo1964_p1b', [N, lambda n: nlt(N0, n)], lambda n, hn: ~D(lit(7), lit(2) ** n + 1))
        def _(p):
            p.have('pow2_mod7', w.type_of(w['pow2_mod7']), pow2_mod7_script)                    # (the lemma again, inline: the proof does not need the library to offer it)
            p.intro('n', 'hn', 'h')
            p.obtain(['k', 'c'], ('pow2_mod7', 'n'))
            def refute(q):
                q.apply('not_dvd_between', lit(7), 'k', q.T('n', lambda n_: lit(2) ** n_ + 1)); q.linarith(); q.linarith(); q.linarith(); q.exact('h')
            p.cases('c', names=['c1', 'c2']); refute(p)
            p.cases('c2', names=['c3', 'c4']); refute(p); refute(p)


@BANK.problem('imo1964_p1a', text='Find all positive integers n for which 2^n - 1 is divisible by 7.', source='IMO 1964 P1 (a)', year=1964,
              topics=('number-theory', 'divisibility', 'modular', 'induction'), level=3, kind='answer', answer='exactly the multiples of 3', helpers=('pow2_iff',),
              spec=Spec(('N',), (lambda n: n > 0,), (lambda n: (2 ** n - 1) % 7 == 0, lambda n: n % 3 == 0), windows={'n': 16}, atoms=[(lambda n: 7, lambda n: 2 ** n - 1), (lambda n: 3, lambda n: n)]),
              notes='formalised as an equivalence of the condition with the answer: 7 divides 2^n - 1 iff 3 divides n; the proof is an induction in steps of three')
def _(w, b):
    s = _syms(w); Z, N, Ex, zzero, D, zof = s.Z, s.N, s.Ex, s.zzero, s.D, s.zof
    N0 = w['N0']; nlt = w['N.lt']
    with w.nota.using(Z):
        lit = w.nota.lit

        @thm(w, 'pow2_iff', [N], lambda n: w.Iff(D(lit(7), lit(2) ** n - 1), D(lit(3), zof(n))))
        def pow2_iff_script(p):
            p.intro('n'); p.induction_on('N.induction3', 'n')
            p.split()
            p.intro('h'); p.exists(zzero); p.ring()
            p.intro('h'); p.exists(zzero); p.ring()
            p.split()
            p.intro('h'); p.exfalso()
            p.apply('not_dvd_between', lit(7), zzero, lit(2) - lit(1)); p.linarith(); p.linarith(); p.linarith(); p.exact('h')
            p.intro('h'); p.exfalso()
            p.apply('not_dvd_between', lit(3), zzero, lit(1)); p.linarith(); p.linarith(); p.linarith(); p.exact('h')
            p.split()
            p.intro('h'); p.exfalso()
            p.apply('not_dvd_between', lit(7), zzero, lit(2) ** 2 - lit(1)); p.linarith(); p.linarith(); p.linarith(); p.exact('h')
            p.intro('h'); p.exfalso()
            p.apply('not_dvd_between', lit(3), zzero, lit(2)); p.linarith(); p.linarith(); p.linarith(); p.exact('h')
            p.intro('m', 'hm')                                                              # 2^(m+3) - 1 = 7 * 2^m + (2^m - 1) and 3 | m + 3 iff 3 | m
            p.rw('Z.pow_succ'); p.rw('Z.pow_succ'); p.rw('Z.pow_succ')
            p.rw('Z.of_N_succ'); p.rw('Z.of_N_succ'); p.rw('Z.of_N_succ')
            p.cases('hm', names=['hf', 'hb'])
            p.split()
            p.intro('h'); p.cases('h', names=['u', 'hu'])
            p.have('h7', p.T('m', lambda m_: D(lit(7), lit(2) ** m_ - 1)), lambda r: (r.exists(r.T('m u', lambda m_, u_: u_ - lit(2) ** m_)), r.linear_combination((1, 'hu'))))
            p.obtain(['t', 'ht'], ('hf', 'h7'))
            p.exists(p.T('t', lambda t_: t_ + 1)); p.linear_combination((1, 'ht'))
            p.intro('h'); p.cases('h', names=['t', 'ht'])
            p.have('h3', p.T('m', lambda m_: D(lit(3), zof(m_))), lambda r: (r.exists(r.T('t', lambda t_: t_ - 1)), r.linear_combination((1, 'ht'))))
            p.obtain(['u', 'hu'], ('hb', 'h3'))
            p.exists(p.T('m u', lambda m_, u_: lit(2) ** m_ + u_)); p.linear_combination((1, 'hu'))

        @thm(w, 'imo1964_p1a', [N, lambda n: nlt(N0, n)], lambda n, hn: w.Iff(D(lit(7), lit(2) ** n - 1), D(lit(3), zof(n))))
        def _(p):
            p.have('pow2_iff', w.type_of(w['pow2_iff']), pow2_iff_script)
            p.intro('n', 'hn'); p.exact(('pow2_iff', 'n'))
