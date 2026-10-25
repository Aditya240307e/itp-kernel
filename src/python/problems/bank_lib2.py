"""bank_lib2.py -- more problems of the bank (see bank.py and bank_lib.py, whose conventions these follow): a statement, its text, its source, its intended meaning as Python (`Spec`) and a reference proof.

The problems are registered after those of bank_lib.py, so the world below them (and every digest in s41_golden.json) is the one of the first 28.  Levels:

    1  a single arithmetic step (an identity by `ring`, an inequality by `nlinarith` with the square that makes it true)
    2  a short proof with one idea: a case split on a remainder, a witness, a hint that is a product
    3  an olympiad lemma: several ideas, a descent, an induction in steps
    4  an olympiad problem with a short solution
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, math
from nt_lib import _syms
from lnl_base import thm
from semantic import Spec
from bank import BANK
from bank_lib import destruct, AND, _dvd, _prime

_N0 = len(BANK.problems)              # the problems of bank_lib.py come first: the world below this file's problems is the one of S41


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ level 1: identities
@BANK.problem('ident_cube_sum', text='For all integers a and b: a^3 + b^3 = (a + b)(a^2 - ab + b^2).', source='folklore', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z', 'Z'), (), lambda a, b: a ** 3 + b ** 3 == (a + b) * (a * a - a * b + b * b), atom=(lambda a, b: a ** 3 + b ** 3, lambda a, b: (a + b) * (a * a - a * b + b * b))))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_cube_sum', [Z, Z], lambda a, c: Eq(a ** 3 + c ** 3, (a + c) * (a ** 2 - a * c + c ** 2)))
        def _(p): p.intro('a', 'c'); p.ring()


@BANK.problem('ident_cube_diff', text='For all integers a and b: a^3 - b^3 = (a - b)(a^2 + ab + b^2).', source='folklore', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z', 'Z'), (), lambda a, b: a ** 3 - b ** 3 == (a - b) * (a * a + a * b + b * b), atom=(lambda a, b: a ** 3 - b ** 3, lambda a, b: (a - b) * (a * a + a * b + b * b))))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_cube_diff', [Z, Z], lambda a, c: Eq(a ** 3 - c ** 3, (a - c) * (a ** 2 + a * c + c ** 2)))
        def _(p): p.intro('a', 'c'); p.ring()


@BANK.problem('ident_three_cubes', text='For all integers a, b, c: a^3 + b^3 + c^3 - 3abc = (a + b + c)(a^2 + b^2 + c^2 - ab - bc - ca).', source='folklore', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z', 'Z', 'Z'), (), lambda a, b, c: a ** 3 + b ** 3 + c ** 3 - 3 * a * b * c == (a + b + c) * (a * a + b * b + c * c - a * b - b * c - c * a),
                        atom=(lambda a, b, c: a ** 3 + b ** 3 + c ** 3 - 3 * a * b * c, lambda a, b, c: (a + b + c) * (a * a + b * b + c * c - a * b - b * c - c * a))))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_three_cubes', [Z, Z, Z], lambda a, c, d: Eq(a ** 3 + c ** 3 + d ** 3 - 3 * a * c * d, (a + c + d) * (a ** 2 + c ** 2 + d ** 2 - a * c - c * d - d * a)))
        def _(p): p.intro('a', 'c', 'd'); p.ring()


@BANK.problem('ident_pell_product', text='For all integers a, b, c, d: (a^2 - 2b^2)(c^2 - 2d^2) = (ac + 2bd)^2 - 2(ad + bc)^2.', source='folklore (composition of Pell forms)', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z', 'Z', 'Z', 'Z'), (), lambda a, b, c, d: (a * a - 2 * b * b) * (c * c - 2 * d * d) == (a * c + 2 * b * d) ** 2 - 2 * (a * d + b * c) ** 2,
                        atom=(lambda a, b, c, d: (a * a - 2 * b * b) * (c * c - 2 * d * d), lambda a, b, c, d: (a * c + 2 * b * d) ** 2 - 2 * (a * d + b * c) ** 2), windows={'z': 3}))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_pell_product', [Z, Z, Z, Z], lambda a, c, d, e: Eq((a ** 2 - 2 * c ** 2) * (d ** 2 - 2 * e ** 2), (a * d + 2 * c * e) ** 2 - 2 * (a * e + c * d) ** 2))
        def _(p): p.intro('a', 'c', 'd', 'e'); p.ring()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ level 1-2: inequalities
@BANK.problem('ineq_three_squares', text='For all integers a, b, c: (a + b + c)^2 <= 3(a^2 + b^2 + c^2).', source='folklore (QM-AM)', topics=('inequality',), level=1,
              spec=Spec(('Z', 'Z', 'Z'), (), lambda a, b, c: (a + b + c) ** 2 <= 3 * (a * a + b * b + c * c), atom=(lambda a, b, c: (a + b + c) ** 2, lambda a, b, c: 3 * (a * a + b * b + c * c))))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_three_squares', [Z, Z, Z], lambda a, c, d: zle((a + c + d) ** 2, 3 * (a ** 2 + c ** 2 + d ** 2)))
        def _(p):
            p.intro('a', 'c', 'd')
            p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - q.h('c')), ('Z.sq_nonneg', lambda q: q.h('c') - q.h('d')), ('Z.sq_nonneg', lambda q: q.h('d') - q.h('a')))


@BANK.problem('ineq_shifted_squares', text='For all integers a, b, c: a^2 + b^2 + c^2 + 3 >= 2(a + b + c).', source='folklore', topics=('inequality',), level=1,
              spec=Spec(('Z', 'Z', 'Z'), (), lambda a, b, c: 2 * (a + b + c) <= a * a + b * b + c * c + 3, atom=(lambda a, b, c: 2 * (a + b + c), lambda a, b, c: a * a + b * b + c * c + 3)))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'ineq_shifted_squares', [Z, Z, Z], lambda a, c, d: zle(2 * (a + c + d), a ** 2 + c ** 2 + d ** 2 + 3))
        def _(p):
            p.intro('a', 'c', 'd')
            p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - 1), ('Z.sq_nonneg', lambda q: q.h('c') - 1), ('Z.sq_nonneg', lambda q: q.h('d') - 1))


@BANK.problem('ineq_cross_terms', text='For all integers x and y: x^2 + y^2 + 1 >= xy + x + y.', source='folklore', topics=('inequality',), level=1,
              spec=Spec(('Z', 'Z'), (), lambda x, y: x * y + x + y <= x * x + y * y + 1, atom=(lambda x, y: x * y + x + y, lambda x, y: x * x + y * y + 1)))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_cross_terms', [Z, Z], lambda x, y: zle(x * y + x + y, x ** 2 + y ** 2 + 1))
        def _(p):
            p.intro('x', 'y')
            p.nlinarith(('Z.sq_nonneg', lambda q: q.h('x') - q.h('y')), ('Z.sq_nonneg', lambda q: q.h('x') - 1), ('Z.sq_nonneg', lambda q: q.h('y') - 1))


@BANK.problem('ineq_cubes_two', text='For all integers a, b >= 0: a^3 + b^3 >= a^2 b + a b^2.', source='folklore (rearrangement)', topics=('inequality',), level=2,
              spec=Spec(('Z', 'Z'), (lambda a, b: a >= 0, lambda a, b: b >= 0), lambda a, b: a * a * b + a * b * b <= a ** 3 + b ** 3, atom=(lambda a, b: a * a * b + a * b * b, lambda a, b: a ** 3 + b ** 3)))
def _(w, b):
    s = _syms(w); Z, zle, zzero = s.Z, s.zle, s.zzero
    with w.nota.using(Z):
        @thm(w, 'ineq_cubes_two', [Z, Z, lambda a, c: zle(zzero, a), lambda a, c, h: zle(zzero, c)], lambda a, c, ha, hc: zle(a ** 2 * c + a * c ** 2, a ** 3 + c ** 3))
        def _(p):                                                                         # a^3 + b^3 - a^2 b - a b^2 = (a + b)(a - b)^2
            p.intro('a', 'c', 'ha', 'hc')
            p.nlinarith(('Z.mul_nonneg', lambda q: q.h('a') + q.h('c'), lambda q: (q.h('a') - q.h('c')) * (q.h('a') - q.h('c')),
                         ('Z.add_nonneg', 'a', 'c', 'ha', 'hc'), ('Z.sq_nonneg', lambda q: q.h('a') - q.h('c'))))


@BANK.problem('ineq_cubes_three', text='For all integers a, b, c >= 0: a^3 + b^3 + c^3 >= 3abc.', source='folklore (AM-GM for three terms)', topics=('inequality',), level=2,
              spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, c: a >= 0, lambda a, b, c: b >= 0, lambda a, b, c: c >= 0), lambda a, b, c: 3 * a * b * c <= a ** 3 + b ** 3 + c ** 3,
                        atom=(lambda a, b, c: 3 * a * b * c, lambda a, b, c: a ** 3 + b ** 3 + c ** 3)))
def _(w, b):
    s = _syms(w); Z, zle, zzero = s.Z, s.zle, s.zzero
    with w.nota.using(Z):
        @thm(w, 'ineq_cubes_three', [Z, Z, Z, lambda a, c, d: zle(zzero, a), lambda a, c, d, h: zle(zzero, c), lambda a, c, d, h, h1: zle(zzero, d)],
             lambda a, c, d, ha, hc, hd: zle(3 * a * c * d, a ** 3 + c ** 3 + d ** 3))
        def _(p):                                                              # a^3 + b^3 + c^3 - 3abc = (a + b + c)((a - b)^2 + (b - c)^2 + (c - a)^2) / 2
            p.intro('a', 'c', 'd', 'ha', 'hc', 'hd')
            p.have('hs', p.T('a c d', lambda a_, c_, d_: zle(zzero, a_ + c_ + d_)), lambda q: q.linarith())
            def hint(x, y): return ('Z.mul_nonneg', lambda q: q.h('a') + q.h('c') + q.h('d'), lambda q: (q.h(x) - q.h(y)) * (q.h(x) - q.h(y)), 'hs', ('Z.sq_nonneg', lambda q: q.h(x) - q.h(y)))
            p.nlinarith(hint('a', 'c'), hint('c', 'd'), hint('d', 'a'))


@BANK.problem('ineq_pairs_eight', text='For all integers a, b, c >= 0: (a + b)(b + c)(c + a) >= 8abc.', source='folklore', topics=('inequality',), level=2,
              spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, c: a >= 0, lambda a, b, c: b >= 0, lambda a, b, c: c >= 0), lambda a, b, c: 8 * a * b * c <= (a + b) * (b + c) * (c + a),
                        atom=(lambda a, b, c: 8 * a * b * c, lambda a, b, c: (a + b) * (b + c) * (c + a))))
def _(w, b):
    s = _syms(w); Z, zle, zzero = s.Z, s.zle, s.zzero
    with w.nota.using(Z):
        @thm(w, 'ineq_pairs_eight', [Z, Z, Z, lambda a, c, d: zle(zzero, a), lambda a, c, d, h: zle(zzero, c), lambda a, c, d, h, h1: zle(zzero, d)],
             lambda a, c, d, ha, hc, hd: zle(8 * a * c * d, (a + c) * (c + d) * (d + a)))
        def _(p):                                                              # (a + b)(b + c)(c + a) - 8abc = a (b - c)^2 + b (c - a)^2 + c (a - b)^2
            p.intro('a', 'c', 'd', 'ha', 'hc', 'hd')
            def hint(x, y, z, hz): return ('Z.mul_nonneg', lambda q: q.h(x), lambda q: (q.h(y) - q.h(z)) * (q.h(y) - q.h(z)), hz, ('Z.sq_nonneg', lambda q: q.h(y) - q.h(z)))
            p.nlinarith(hint('a', 'c', 'd', 'ha'), hint('c', 'd', 'a', 'hc'), hint('d', 'a', 'c', 'hd'))


@BANK.problem('ineq_fourth_powers_three', text='For all integers a, b, c: a^4 + b^4 + c^4 >= abc(a + b + c).', source='folklore', topics=('inequality',), level=2,
              spec=Spec(('Z', 'Z', 'Z'), (), lambda a, b, c: a * b * c * (a + b + c) <= a ** 4 + b ** 4 + c ** 4, atom=(lambda a, b, c: a * b * c * (a + b + c), lambda a, b, c: a ** 4 + b ** 4 + c ** 4)))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_fourth_powers_three', [Z, Z, Z], lambda a, c, d: zle(a * c * d * (a + c + d), a ** 4 + c ** 4 + d ** 4))
        def _(p):                                                              # a^4 + b^4 + c^4 >= a^2 b^2 + b^2 c^2 + c^2 a^2 >= abc (a + b + c)
            p.intro('a', 'c', 'd')
            sq = lambda f: ('Z.sq_nonneg', f)
            p.nlinarith(sq(lambda q: q.h('a') ** 2 - q.h('c') ** 2), sq(lambda q: q.h('c') ** 2 - q.h('d') ** 2), sq(lambda q: q.h('d') ** 2 - q.h('a') ** 2),
                        sq(lambda q: q.h('a') * q.h('c') - q.h('c') * q.h('d')), sq(lambda q: q.h('c') * q.h('d') - q.h('d') * q.h('a')), sq(lambda q: q.h('d') * q.h('a') - q.h('a') * q.h('c')))


@BANK.problem('ineq_four_fourth_powers', text='For all integers a, b, c, d: a^4 + b^4 + c^4 + d^4 >= 4abcd.', source='folklore (AM-GM for four terms)', topics=('inequality',), level=2,
              spec=Spec(('Z', 'Z', 'Z', 'Z'), (), lambda a, b, c, d: 4 * a * b * c * d <= a ** 4 + b ** 4 + c ** 4 + d ** 4, atom=(lambda a, b, c, d: 4 * a * b * c * d, lambda a, b, c, d: a ** 4 + b ** 4 + c ** 4 + d ** 4),
                        windows={'z': 3}))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_four_fourth_powers', [Z, Z, Z, Z], lambda a, c, d, e: zle(4 * a * c * d * e, a ** 4 + c ** 4 + d ** 4 + e ** 4))
        def _(p):                                                              # a^4 + b^4 >= 2 a^2 b^2, c^4 + d^4 >= 2 c^2 d^2, a^2 b^2 + c^2 d^2 >= 2 abcd
            p.intro('a', 'c', 'd', 'e')
            sq = lambda f: ('Z.sq_nonneg', f)
            p.nlinarith(sq(lambda q: q.h('a') ** 2 - q.h('c') ** 2), sq(lambda q: q.h('d') ** 2 - q.h('e') ** 2), sq(lambda q: q.h('a') * q.h('c') - q.h('d') * q.h('e')))


@BANK.problem('quad_form_zero', text='If a and b are integers with a^2 + ab + b^2 = 0, then a = 0 and b = 0.', source='folklore', topics=('algebra', 'inequality'), level=2,
              spec=Spec(('Z', 'Z'), (lambda a, b: a * a + a * b + b * b == 0,), lambda a, b: a == 0 and b == 0,
                        skip_mutants={'Z.add -> Z.sub in ((Z.add (Z.add (Z.pow x0 2) (Z.mul x0 x1)) (Z.pow x1 2)) = 0)': 'a^2 + ab - b^2 = 0 and a^2 - ab + b^2 = 0 also have no integer solution but (0, 0) (5 and -3 are not squares): the hypothesis is compared as a truth table'}))
def _(w, b):
    s = _syms(w); Z, Eq, zzero = s.Z, s.Eq, s.zzero
    with w.nota.using(Z):
        @thm(w, 'quad_form_zero', [Z, Z, lambda a, c: Eq(a ** 2 + a * c + c ** 2, zzero)], lambda a, c, h: Eq(a, zzero) & Eq(c, zzero))
        def _(p):                                                              # 4 (a^2 + ab + b^2) = (2a + b)^2 + 3 b^2 = (2b + a)^2 + 3 a^2
            p.intro('a', 'c', 'h')
            p.have('ha', p.T('a', lambda a_: Eq(a_ * a_, zzero)), lambda q: q.nlinarith(('Z.sq_nonneg', lambda r: 2 * r.h('c') + r.h('a')), ('Z.sq_nonneg', q.h('a'))))
            p.have('hc', p.T('c', lambda c_: Eq(c_ * c_, zzero)), lambda q: q.nlinarith(('Z.sq_nonneg', lambda r: 2 * r.h('a') + r.h('c')), ('Z.sq_nonneg', q.h('c'))))
            p.split()
            p.obtain(['u', 'u'], ('Z.mul_eq_zero', 'a', 'a', 'ha')); p.exact('u'); p.exact('u')
            p.obtain(['v', 'v'], ('Z.mul_eq_zero', 'c', 'c', 'hc')); p.exact('v'); p.exact('v')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ level 2: remainders
@BANK.problem('consecutive_product_even', text='The product of two consecutive integers is even: 2 divides n(n + 1) for every integer n.', source='folklore', topics=('number-theory', 'divisibility', 'parity'), level=2,
              spec=Spec(('Z',), (), lambda n: _dvd(2, n * (n + 1)), atom=(lambda n: 2, lambda n: n * (n + 1)), skip_mutants={'constant 2 -> -2 in (Z.dvd 2': 'a divisor and its negative divide the same integers'}))
def _(w, b):
    s = _syms(w); Z, zzero, zlt, D = s.Z, s.zzero, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'consecutive_product_even', [Z], lambda n: D(lit(2), n * (n + 1)))
        def _(p):
            p.intro('n')
            p.have('h2', p.T('', lambda: zlt(zzero, lit(2))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(2), 'h2', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 1)
            p.rw('r_eq'); p.exists(p.T('q', lambda q: q * (2 * q + 1))); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: (2 * q + 1) * (q + 1))); p.ring()


@BANK.problem('three_consecutive_div3', text='The product of three consecutive integers is divisible by 3: 3 divides n(n + 1)(n + 2) for every integer n.', source='folklore', topics=('number-theory', 'divisibility', 'modular'), level=2,
              spec=Spec(('Z',), (), lambda n: _dvd(3, n * (n + 1) * (n + 2)), atom=(lambda n: 3, lambda n: n * (n + 1) * (n + 2)), skip_mutants={'constant 3 -> -3 in (Z.dvd 3': 'a divisor and its negative divide the same integers'}))
def _(w, b):
    s = _syms(w); Z, zzero, zlt, D = s.Z, s.zzero, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'three_consecutive_div3', [Z], lambda n: D(lit(3), n * (n + 1) * (n + 2)))
        def _(p):
            p.intro('n')
            p.have('h3', p.T('', lambda: zlt(zzero, lit(3))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 2)
            p.rw('r_eq'); p.exists(p.T('q', lambda q: q * (3 * q + 1) * (3 * q + 2))); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: (3 * q + 1) * (3 * q + 2) * (q + 1))); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: (3 * q + 2) * (q + 1) * (3 * q + 4))); p.ring()


@BANK.problem('fourth_power_odd_16', text='The fourth power of an odd integer leaves remainder 1 upon division by 16: if n is odd then 16 divides n^4 - 1.', source='folklore', topics=('number-theory', 'divisibility', 'modular', 'parity'), level=2,
              spec=Spec(('Z',), (lambda n: n % 2 == 1,), lambda n: _dvd(16, n ** 4 - 1), windows={'z': 8}, atom=(lambda n: 16, lambda n: n ** 4 - 1),
                        skip_mutants={'constant 16 -> -16 in (Z.dvd 16': 'a divisor and its negative divide the same integers',
                                      'in (x0 = (Z.add (Z.mul 2 x1) 1))': 'the hypothesis n = 2k + 1, n = -2k + 1, n = 2k - 1, n = -2k - 1 all say that n is odd (k -> -k, k -> k + 1); the hypothesis is compared as a truth table'}))
def _(w, b):
    s = _syms(w); Z, Eq, Ex, zzero, zlt, D = s.Z, s.Eq, s.Ex, s.zzero, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'fourth_power_odd_16', [Z, lambda n: Ex(Z, w.lam(Z, lambda k: Eq(n, 2 * k + 1)))], lambda n, h: D(lit(16), n ** 4 - 1))
        def _(p):
            p.intro('n', 'h'); p.cases('h', names=['k', 'hk']); p.rw('hk')
            p.have('h2', p.T('', lambda: zlt(zzero, lit(2))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(2), 'h2', 'k'), ['j', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 1)
            p.rw('r_eq'); p.exists(p.T('j', lambda j: 16 * j ** 4 + 16 * j ** 3 + 6 * j ** 2 + j)); p.ring()
            p.rw('r_eq'); p.exists(p.T('j', lambda j: 16 * j ** 4 + 48 * j ** 3 + 54 * j ** 2 + 27 * j + 5)); p.ring()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ level 3: remainders and Diophantine equations
@BANK.problem('diff_squares_not_2_mod_4', text='No integer of the form 4k + 2 is a difference of two squares: x^2 - y^2 /= 4k + 2 for all integers x, y, k.', source='folklore', topics=('number-theory', 'modular', 'diophantine'), level=3,
              spec=Spec(('Z', 'Z', 'Z'), (), lambda x, y, k: x * x - y * y != 4 * k + 2, windows={'z': 6}, atom=(lambda x, y, k: x * x - y * y, lambda x, y, k: 4 * k + 2)))
def _(w, b):
    s = _syms(w); Z, Eq, zzero = s.Z, s.Eq, s.zzero
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'diff_squares_not_2_mod_4', [Z, Z, Z], lambda x, y, k: ~Eq(x ** 2 - y ** 2, 4 * k + 2))
        def _(p):
            p.intro('x', 'y', 'k', 'h')
            p.obtain(['a', 'ha'], ('square_mod_four', 'x')); p.obtain(['c', 'hc'], ('square_mod_four', 'y'))
            def close(r):                       # x^2 = 4a + ex, y^2 = 4c + ey, x^2 - y^2 = 4k + 2:  4 (a - c - k) = 2 - ex + ey = r  with 0 < r < 4
                p.apply('not_dvd_between', lit(4), lit(0), lit(r)); p.linarith(); p.linarith(); p.linarith()
                p.exists(p.T('a c k', lambda a_, c_, k_: a_ - c_ - k_)); p.linarith()
            p.cases('ha', names=['ha', 'ha'])
            for ex in (0, 1):
                p.cases('hc', names=['hc', 'hc'])
                for ey in (0, 1): close(2 - ex + ey)


@BANK.problem('pythag_three', text='In a Pythagorean triple a^2 + b^2 = c^2 of integers, one of a, b is divisible by 3.', source='folklore', topics=('number-theory', 'divisibility', 'modular', 'case-analysis'), level=3,
              spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, c: a * a + b * b == c * c,), lambda a, b, c: _dvd(3, a) or _dvd(3, b), windows={'z': 8}, atoms=[(lambda a, b, c: 3, lambda a, b, c: a), (lambda a, b, c: 3, lambda a, b, c: b)],
                        skip_mutants={'constant 3 -> -3 in (Z.dvd 3': 'a divisor and its negative divide the same integers'}))
def _(w, b):
    s = _syms(w); Z, Eq, zzero, zlt, D = s.Z, s.Eq, s.zzero, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'pythag_three', [Z, Z, Z, lambda a, c, d: Eq(a ** 2 + c ** 2, d ** 2)], lambda a, c, d, h: D(lit(3), a) | D(lit(3), c))
        def _(p):
            p.intro('a', 'c', 'd', 'h')
            p.have('h3', p.T('', lambda: zlt(zzero, lit(3))), lambda q: q.linarith())
            p.obtain(['k', 'hk'], ('square_mod_three', 'd'))
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'a'), ['q', 'r', 'e1', 'h10', 'h11'])
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'c'), ['t', 'u', 'e2', 'h20', 'h21'])
            p.interval_cases('r', 0, 2)
            for rv in range(3):
                p.interval_cases('u', 0, 2)
                for uv in range(3):
                    if rv == 0:
                        p.left(); p.exists('q'); p.rw('e1'); p.rw('r_eq'); p.ring()
                    elif uv == 0:
                        p.right(); p.exists('t'); p.rw('e2'); p.rw('u_eq'); p.ring()
                    else:                                                      # a^2 + c^2 = 3 S + rv^2 + uv^2 and d^2 = 3k + j: 3 (k - S) = rv^2 + uv^2 - j, which 3 does not divide
                        p.exfalso()
                        p.rw_at('h', 'e1'); p.rw_at('h', 'e2'); p.rw_at('h', 'r_eq'); p.rw_at('h', 'u_eq')
                        p.cases('hk', names=['hk', 'hk'])
                        for j in (0, 1):
                            x = rv * rv + uv * uv - j
                            p.apply('not_dvd_between', lit(3), lit(x // 3), lit(x)); p.linarith(); p.linarith(); p.linarith()
                            p.exists(p.T('k q t', lambda k_, q_, t_: k_ - 3 * q_ ** 2 - 2 * rv * q_ - 3 * t_ ** 2 - 2 * uv * t_)); p.linarith()


@BANK.problem('cube_sum_not_4_mod_9', text='No integer of the form 9k + 4 is a sum of three cubes: x^3 + y^3 + z^3 /= 9k + 4 for all integers x, y, z, k.', source='folklore', topics=('number-theory', 'modular', 'diophantine', 'case-analysis'), level=3,
              spec=Spec(('Z', 'Z', 'Z', 'Z'), (), lambda x, y, z, k: x ** 3 + y ** 3 + z ** 3 != 9 * k + 4, windows={'z': 4}, atom=(lambda x, y, z, k: x ** 3 + y ** 3 + z ** 3, lambda x, y, z, k: 9 * k + 4)))
def _(w, b):
    s = _syms(w); Z, Eq, zzero = s.Z, s.Eq, s.zzero
    with w.nota.using(Z):
        lit = w.nota.lit
        def L(n): return lit(n) if n >= 0 else -lit(-n)
        @thm(w, 'cube_sum_not_4_mod_9', [Z, Z, Z, Z], lambda x, y, z, k: ~Eq(x ** 3 + y ** 3 + z ** 3, 9 * k + 4))
        def _(p):
            p.intro('x', 'y', 'z', 'k', 'h')
            p.obtain(['a', 'ha'], ('cube_mod_nine', 'x')); p.obtain(['c', 'hc'], ('cube_mod_nine', 'y')); p.obtain(['d', 'hd'], ('cube_mod_nine', 'z'))
            def close(e):                       # the cubes are 9a + e1, 9c + e2, 9d + e3: 9 (a + c + d - k) = 4 - e, which 9 does not divide
                x_ = 4 - e
                p.apply('not_dvd_between', lit(9), L(x_ // 9), L(x_)); p.linarith(); p.linarith(); p.linarith()
                p.exists(p.T('a c d k', lambda a_, c_, d_, k_: a_ + c_ + d_ - k_)); p.linarith()
            def go(rest, acc):
                if not rest: close(acc); return
                hn = rest[0]
                p.cases(hn, names=[hn, hn + '_'])
                go(rest[1:], acc)
                p.cases(hn + '_', names=[hn, hn + '_'])
                go(rest[1:], acc + 1)
                go(rest[1:], acc + 8)
            go(['ha', 'hc', 'hd'], 0)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ gcd, coprimality, primes
@BANK.problem('gcd_shift', text='If gcd(a, b) = 1 then gcd(a, b + ka) = 1 for every integer k.', source='folklore', topics=('number-theory', 'gcd'), level=2,
              spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, k: math.gcd(a, b) == 1,), lambda a, b, k: math.gcd(a, b + k * a) == 1, windows={'z': 6}, atom=(lambda a, b, k: a, lambda a, b, k: b + k * a, lambda a, b, k: 1)))
def _(w, b):
    s = _syms(w); Z, zone = s.Z, s.zone
    zgcd = w['Z.is_gcd']
    with w.nota.using(Z):
        @thm(w, 'gcd_shift', [Z, Z, Z, lambda a, c, k: zgcd(a, c, zone)], lambda a, c, k, h: zgcd(a, c + k * a, zone))
        def _(p):                                                              # a x + b y = 1  gives  a (x - k y) + (b + k a) y = 1
            p.intro('a', 'c', 'k', 'h')
            p.obtain(['x', 'hx'], ('Z.coprime_of_is_gcd_one', 'a', 'c', 'h')); p.cases('hx', names=['y', 'hy'])
            p.apply('Z.is_gcd_one_of_coprime')
            p.exists(p.T('x y k', lambda x, y, k: x - k * y)); p.exists('y'); p.linear_combination((1, 'hy'))


@BANK.problem('gcd_sum_product', text='If gcd(a, b) = 1 then gcd(a + b, ab) = 1.', source='folklore', topics=('number-theory', 'gcd'), level=3,
              spec=Spec(('Z', 'Z'), (lambda a, b: math.gcd(a, b) == 1,), lambda a, b: math.gcd(a + b, a * b) == 1, windows={'z': 8}, atom=(lambda a, b: a + b, lambda a, b: a * b, lambda a, b: 1)))
def _(w, b):
    s = _syms(w); Z, zone = s.Z, s.zone
    zgcd = w['Z.is_gcd']; zcop = w['Z.coprime']
    with w.nota.using(Z):
        @thm(w, 'gcd_sum_product', [Z, Z, lambda a, c: zgcd(a, c, zone)], lambda a, c, h: zgcd(a + c, a * c, zone))
        def _(p):                                                              # (a + b) y + a (x - y) = 1 and (a + b) x + b (y - x) = 1, so a + b is coprime to a and to b, hence to ab
            p.intro('a', 'c', 'h')
            p.obtain(['x', 'hx'], ('Z.coprime_of_is_gcd_one', 'a', 'c', 'h')); p.cases('hx', names=['y', 'hy'])
            p.have('h1', p.T('a c', lambda a_, c_: zcop(a_ + c_, a_)), lambda q: (q.exists(q.h('y')), q.exists(q.T('x y', lambda x, y: x - y)), q.linear_combination((1, 'hy'))))
            p.have('h2', p.T('a c', lambda a_, c_: zcop(a_ + c_, c_)), lambda q: (q.exists(q.h('x')), q.exists(q.T('x y', lambda x, y: y - x)), q.linear_combination((1, 'hy'))))
            p.apply('Z.is_gcd_one_of_coprime'); p.exact(('Z.coprime_mul_right', p.T('a c', lambda a_, c_: a_ + c_), 'a', 'c', 'h1', 'h2'))


@BANK.problem('square_not_prime', text='A perfect square is never a prime: for every integer n, n^2 is not prime.', source='folklore', topics=('number-theory', 'prime'), level=3,
              spec=Spec(('Z',), (), lambda n: not _prime(n * n), windows={'z': 8}, atom=(lambda n: n * n,)))
def _(w, b):
    s = _syms(w); Z, Eq, zzero, zone, zlt, zle, D = s.Z, s.Eq, s.zzero, s.zone, s.zlt, s.zle, s.D
    prime = w['Z.prime']
    with w.nota.using(Z):
        @thm(w, 'square_not_prime', [Z], lambda n: ~prime(n ** 2))
        def _(p):                                                              # d = |n| is a positive divisor of n^2, so d = 1 or d = n^2; both are impossible when 1 < n^2
            p.intro('n', 'hp'); p.cases('hp', names=['g1', 'g2'])
            p.obtain(['c', 'c'], ('Z.le_or_lt', zzero, 'n'))
            p.have('dpos', p.T('n', lambda n_: zlt(zzero, n_)), lambda q: q.nlinarith())
            p.have('dd', p.T('n', lambda n_: D(n_, n_ ** 2)), lambda q: (q.exists(q.h('n')), q.ring()))
            p.obtain(['e', 'e'], ('g2', 'n', 'dpos', 'dd'))
            p.nlinarith(); p.nlinarith()
            p.have('dpos', p.T('n', lambda n_: zlt(zzero, -n_)), lambda q: q.nlinarith())
            p.have('dd', p.T('n', lambda n_: D(-n_, n_ ** 2)), lambda q: (q.exists(q.T('n', lambda n_: -n_)), q.ring()))
            p.obtain(['e', 'e'], ('g2', p.T('n', lambda n_: -n_), 'dpos', 'dd'))
            p.nlinarith(); p.nlinarith()


@BANK.problem('no_square_pronic', text='The product of two consecutive positive integers is never a perfect square: n(n + 1) /= m^2 for integers n > 0 and m.', source='folklore', topics=('number-theory', 'bounding'), level=2,
              spec=Spec(('Z', 'Z'), (lambda n, m: n > 0,), lambda n, m: n * (n + 1) != m * m, atom=(lambda n, m: n * (n + 1), lambda n, m: m * m)))
def _(w, b):
    s = _syms(w); Z, Eq, zzero, zlt, zle = s.Z, s.Eq, s.zzero, s.zlt, s.zle
    with w.nota.using(Z):
        @thm(w, 'no_square_pronic', [Z, Z, lambda n, m: zlt(zzero, n)], lambda n, m, h: ~Eq(n * (n + 1), m ** 2))
        def _(p):                                                              # n^2 < n (n + 1) < (n + 1)^2: m^2 would lie strictly between two consecutive squares
            p.intro('n', 'm', 'hn', 'h')
            p.obtain(['c', 'c'], ('Z.le_or_lt', zzero, 'm'))
            p.obtain(['d', 'd'], ('Z.le_or_lt', 'm', 'n')); p.nlinarith(); p.nlinarith()
            p.obtain(['d', 'd'], ('Z.le_or_lt', (s.zneg, 'm'), 'n')); p.nlinarith(); p.nlinarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ powers: induction on a natural exponent
@BANK.problem('five_dvd_six_pow_minus_one', text='For every natural number n, 5 divides 6^n - 1.', source='folklore', topics=('number-theory', 'divisibility', 'induction'), level=2,
              spec=Spec(('N',), (), lambda n: _dvd(5, 6 ** n - 1), windows={'n': 8}, atom=(lambda n: 5, lambda n: 6 ** n - 1)))
def _(w, b):
    s = _syms(w); Z, N, zzero, D = s.Z, s.N, s.zzero, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'five_dvd_six_pow_minus_one', [N], lambda n: D(lit(5), lit(6) ** n - 1))
        def _(p):                                                              # 6^(m+1) - 1 = 6 (6^m - 1) + 5
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            p.exists(zzero); p.ring()
            p.intro('m', 'hm'); p.cases('hm', names=['k', 'hk'])
            p.exists(p.T('k', lambda k_: 6 * k_ + 1)); p.rw('Z.pow_succ'); p.linear_combination((6, 'hk'))


@BANK.problem('three_dvd_four_pow_plus_five', text='For every natural number n, 3 divides 4^n + 5.', source='folklore', topics=('number-theory', 'divisibility', 'induction'), level=2,
              spec=Spec(('N',), (), lambda n: _dvd(3, 4 ** n + 5), windows={'n': 8}, atom=(lambda n: 3, lambda n: 4 ** n + 5)))
def _(w, b):
    s = _syms(w); Z, N, D = s.Z, s.N, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'three_dvd_four_pow_plus_five', [N], lambda n: D(lit(3), lit(4) ** n + 5))
        def _(p):                                                              # 4^(m+1) + 5 = 4 (4^m + 5) - 15
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            p.exists(lit(2)); p.ring()
            p.intro('m', 'hm'); p.cases('hm', names=['k', 'hk'])
            p.exists(p.T('k', lambda k_: 4 * k_ - 5)); p.rw('Z.pow_succ'); p.linear_combination((4, 'hk'))


@BANK.problem('two_pow_gt_n', text='For every natural number n, n < 2^n.', source='folklore', topics=('inequality', 'induction'), level=2,
              spec=Spec(('N',), (), lambda n: n < 2 ** n, windows={'n': 10}, atom=(lambda n: n, lambda n: 2 ** n)))
def _(w, b):
    s = _syms(w); Z, N, zlt, zof = s.Z, s.N, s.zlt, s.zof
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'two_pow_gt_n', [N], lambda n: zlt(zof(n), lit(2) ** n))
        def _(p):                                                              # n + 1 <= 2^n, and 2^n >= 1, so n + 1 < 2 * 2^n
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            p.rw('Z.of_N_zero'); p.exact('Z.zero_lt_one')
            p.intro('m', 'hm'); p.rw('Z.of_N_succ'); p.rw('Z.pow_succ')
            p.linarith(('Z.nonneg_of_N', 'm'))


@BANK.problem('bernoulli_int', text='Bernoulli\'s inequality: for every integer x >= -1 and every natural number n, (1 + x)^n >= 1 + n x.', source='folklore (Bernoulli)', topics=('inequality', 'induction'), level=3,
              spec=Spec(('Z', 'N'), (lambda x, n: x >= -1,), lambda x, n: 1 + n * x <= (1 + x) ** n, windows={'z': 6, 'n': 7}, atom=(lambda x, n: 1 + n * x, lambda x, n: (1 + x) ** n)))
def _(w, b):
    s = _syms(w); Z, N, zle, zzero, zof = s.Z, s.N, s.zle, s.zzero, s.zof
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'bernoulli_int', [Z, N, lambda x, n: zle(-lit(1), x)], lambda x, n, h: zle(1 + zof(n) * x, (1 + x) ** n))
        def _(p):                                                              # (1 + x)^(m+1) = (1 + x)(1 + x)^m >= (1 + x)(1 + m x) = 1 + (m + 1) x + m x^2
            p.intro('x', 'n', 'hx'); p.induction_on('N.induction_on', 'n')
            p.rw('Z.of_N_zero'); p.linarith()
            p.intro('m', 'hm'); p.rw('Z.of_N_succ'); p.rw('Z.pow_succ')
            p.have('h1', p.T('x', lambda x_: zle(zzero, 1 + x_)), lambda q: q.linarith())
            p.have('h2', p.T('x m', lambda x_, m_: zle(zzero, (1 + x_) ** m_ - (1 + zof(m_) * x_))), lambda q: q.linarith())
            p.nlinarith(('Z.mul_nonneg', p.T('x', lambda x_: 1 + x_), p.T('x m', lambda x_, m_: (1 + x_) ** m_ - (1 + zof(m_) * x_)), 'h1', 'h2'),
                        ('Z.mul_nonneg', p.T('m', lambda m_: zof(m_)), p.T('x', lambda x_: x_ * x_), ('Z.nonneg_of_N', 'm'), ('Z.sq_nonneg', 'x')))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ construction and "find all"
def _square_factor(n):
    """n has a divisor a^2 with a > 1"""
    return n > 1 and any(n % (a * a) == 0 for a in range(2, int(n ** 0.5) + 1))


@BANK.problem('three_consecutive_not_squarefree', text='Prove that there are three consecutive positive integers, each of which is divisible by the square of an integer greater than 1.', source='folklore', topics=('number-theory', 'divisibility', 'construction'), level=2, kind='exists',
              answer='548, 549, 550 (divisible by 2^2, 3^2, 5^2)',
              spec=Spec((), (), lambda: any(n > 0 and _square_factor(n) and _square_factor(n + 1) and _square_factor(n + 2) for n in range(-560, 561)), windows={'ex': 560, 'atom': 6},
                        body=(('Z',), lambda n: n > 0 and _square_factor(n) and _square_factor(n + 1) and _square_factor(n + 2)),
                        atoms=[(lambda n: 0, lambda n: n), (lambda n, a: 1, lambda n, a: a), (lambda n, a: a * a, lambda n, a: n), (lambda n, a: 1, lambda n, a: a), (lambda n, a: a * a, lambda n, a: n + 1),
                               (lambda n, a: 1, lambda n, a: a), (lambda n, a: a * a, lambda n, a: n + 2)]))
def _(w, b):
    s = _syms(w); Z, Ex, zzero, zone, zlt, D = s.Z, s.Ex, s.zzero, s.zone, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        def sq_div(m): return Ex(Z, w.lam(Z, lambda a: zlt(zone, a) & D(a * a, m)))
        @thm(w, 'three_consecutive_not_squarefree', [], lambda: Ex(Z, w.lam(Z, lambda n: AND(zlt(zzero, n), sq_div(n), sq_div(n + 1), sq_div(n + 2)))))
        def _(p):
            p.exists(lit(548))
            p.split(); p.linarith()
            p.split(); p.exists(lit(2)); p.split(); p.linarith(); p.exists(lit(137)); p.ring()
            p.split(); p.exists(lit(3)); p.split(); p.linarith(); p.exists(lit(61)); p.ring()
            p.exists(lit(5)); p.split(); p.linarith(); p.exists(lit(22)); p.ring()


@BANK.problem('find_all_x2_eq_x', text='Find all integers x with x^2 = x.', source='folklore', topics=('algebra', 'case-analysis'), level=1, kind='answer', answer='x = 0 or x = 1',
              spec=Spec(('Z',), (), (lambda x: x * x == x, lambda x: x == 0 or x == 1), atoms=[(lambda x: x * x, lambda x: x), (lambda x: x, lambda x: 0), (lambda x: x, lambda x: 1)]),
              notes='formalised as an equivalence of the equation with the answer')
def _(w, b):
    s = _syms(w); Z, Eq, zzero, zone = s.Z, s.Eq, s.zzero, s.zone
    with w.nota.using(Z):
        @thm(w, 'find_all_x2_eq_x', [Z], lambda x: w.Iff(Eq(x * x, x), Eq(x, zzero) | Eq(x, zone)))
        def _(p):
            p.intro('x'); p.split()
            p.intro('h')
            p.have('h0', p.T('x', lambda x_: Eq(x_ * (x_ - 1), zzero)), lambda q: q.linear_combination((1, 'h')))
            p.obtain(['u', 'u'], ('Z.mul_eq_zero', 'x', p.T('x', lambda x_: x_ - 1), 'h0'))
            p.left(); p.exact('u')
            p.right(); p.linarith()
            p.intro('h'); p.cases('h', names=['u', 'u'])
            p.linear_combination((p.T('x', lambda x_: x_ - 1), 'u'))
            p.linear_combination((p.T('x', lambda x_: x_), 'u'))


@BANK.problem('find_all_xy_eq_x_plus_y', text='Find all pairs of integers (x, y) with xy = x + y.', source='folklore', topics=('number-theory', 'diophantine'), level=3, kind='answer', answer='(0, 0) and (2, 2)',
              spec=Spec(('Z', 'Z'), (), (lambda x, y: x * y == x + y, lambda x, y: (x == 0 and y == 0) or (x == 2 and y == 2)), windows={'z': 8},
                        atoms=[(lambda x, y: x * y, lambda x, y: x + y), (lambda x, y: x, lambda x, y: 0), (lambda x, y: y, lambda x, y: 0), (lambda x, y: x, lambda x, y: 2), (lambda x, y: y, lambda x, y: 2)]),
              notes='formalised as an equivalence of the equation with the answer; (x - 1)(y - 1) = 1 makes x - 1 a unit')
def _(w, b):
    s = _syms(w); Z, Eq, zzero, D = s.Z, s.Eq, s.zzero, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'find_all_xy_eq_x_plus_y', [Z, Z], lambda x, y: w.Iff(Eq(x * y, x + y), (Eq(x, zzero) & Eq(y, zzero)) | (Eq(x, lit(2)) & Eq(y, lit(2)))))
        def _(p):
            p.intro('x', 'y'); p.split()
            p.intro('h')
            p.have('hd', p.T('x', lambda x_: D(x_ - 1, lit(1))), lambda q: (q.exists(q.T('y', lambda y_: y_ - 1)), q.linear_combination((-1, 'h'))))
            p.obtain(['u', 'u'], ('Z.dvd_one', p.T('x', lambda x_: x_ - 1), 'hd'))
            p.right(); p.split(); p.linarith()                                                  # x - 1 = 1: x = 2, and then y = 2
            p.linear_combination((1, 'h'), (p.T('y', lambda y_: -(y_ - 1)), 'u'))
            p.left(); p.split(); p.linarith()                                                   # x - 1 = -1: x = 0, and then y = 0
            p.linear_combination((-1, 'h'), (p.T('y', lambda y_: y_ - 1), 'u'))
            p.intro('h'); p.cases('h', names=['h1', 'h2'])
            p.cases('h1', names=['a', 'c'])
            p.linear_combination((p.T('y', lambda y_: y_ - 1), 'a'), (-1, 'c'))
            p.cases('h2', names=['a', 'c'])
            p.linear_combination((p.T('y', lambda y_: y_ - 1), 'a'), (1, 'c'))


@BANK.problem('find_all_n_plus_1_dvd', text='Find all positive integers n such that n + 1 divides n^2 + 1.', source='folklore', topics=('number-theory', 'divisibility'), level=3, kind='answer', answer='n = 1',
              spec=Spec(('Z',), (lambda n: n > 0,), (lambda n: _dvd(n + 1, n * n + 1), lambda n: n == 1), windows={'z': 10}, atoms=[(lambda n: n + 1, lambda n: n * n + 1), (lambda n: n, lambda n: 1)]),
              notes='formalised as an equivalence of the divisibility with the answer; n + 1 divides n^2 + 1 - (n + 1)(n - 1) = 2')
def _(w, b):
    s = _syms(w); Z, Eq, zzero, zlt, zle, D = s.Z, s.Eq, s.zzero, s.zlt, s.zle, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'find_all_n_plus_1_dvd', [Z, lambda n: zlt(zzero, n)], lambda n, hn: w.Iff(D(n + 1, n ** 2 + 1), Eq(n, lit(1))))
        def _(p):
            p.intro('n', 'hn'); p.split()
            p.intro('h'); p.cases('h', names=['k', 'hk'])
            p.have('h2', p.T('n', lambda n_: D(n_ + 1, lit(2))), lambda q: (q.exists(q.T('n k', lambda n_, k_: k_ - n_ + 1)), q.linear_combination((1, 'hk'))))
            p.have('h3', p.T('n', lambda n_: zle(n_ + 1, lit(2))), lambda q: (q.apply('Z.le_of_dvd'), q.linarith(), q.exact('h2')))
            p.linarith()
            p.intro('h'); p.exists(lit(1)); p.linear_combination((p.T('n', lambda n_: n_), 'h'))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ induction from a starting point, in both directions
def _frob_rep(n): return any(a >= 0 and b >= 0 and n == 3 * a + 5 * b for a in range(-2, 12) for b in range(-2, 12))


@BANK.problem('frobenius_3_5', text='Prove that every integer n >= 8 can be written as 3a + 5b with integers a, b >= 0.', source='folklore (Frobenius coin problem)', topics=('number-theory', 'induction', 'diophantine'), level=3,
              spec=Spec(('Z',), (lambda n: n >= 8,), lambda n: _frob_rep(n), windows={'z': 24, 'ex': 12},
                        body=(('Z', 'Z'), lambda n, a, b: a >= 0 and b >= 0 and n == 3 * a + 5 * b),
                        atoms=[(lambda n, a, b: 0, lambda n, a, b: a), (lambda n, a, b: 0, lambda n, a, b: b), (lambda n, a, b: n, lambda n, a, b: 3 * a + 5 * b)]),
              notes='strong induction from 8: the cases 8, 9, 10 by hand, and n - 3 >= 8 otherwise')
def _(w, b):
    s = _syms(w); Z, Eq, Ex, zzero, zle, zlt = s.Z, s.Eq, s.Ex, s.zzero, s.zle, s.zlt
    with w.nota.using(Z):
        lit = w.nota.lit
        def rep(n): return Ex(Z, w.lam(Z, lambda a: Ex(Z, w.lam(Z, lambda c: AND(zle(zzero, a), zle(zzero, c), Eq(n, 3 * a + 5 * c))))))
        @thm(w, 'frobenius_3_5', [Z, lambda n: zle(lit(8), n)], lambda n, h: rep(n))
        def _(p):
            p.intro('n', 'hn')
            p.induction_on('Z.strong_induction', 'n', extra=(lit(8),))
            p.intro('m', 'hm', 'ih')
            p.obtain(['c', 'c'], ('Z.le_or_lt', lit(11), 'm'))
            p.have('h8', p.T('m', lambda m_: zle(lit(8), m_ - 3)), lambda q: q.linarith())
            p.have('hlt', p.T('m', lambda m_: zlt(m_ - 3, m_)), lambda q: q.linarith())
            destruct(p, ('ih', p.T('m', lambda m_: m_ - 3), 'h8', 'hlt'), ['a', 'd', 'ha', 'hb', 'e'])
            p.exists(p.T('a', lambda a_: a_ + 1)); p.exists('d')
            p.split(); p.linarith(); p.split(); p.exact('hb'); p.linear_combination((1, 'e'))
            p.interval_cases('m', 8, 10)
            for av, bv in ((1, 1), (3, 0), (0, 2)):
                p.exists(lit(av)); p.exists(lit(bv)); p.split(); p.linarith(); p.split(); p.linarith(); p.linarith()
            p.assumption()


def _cauchy_cond(f, W=range(-8, 9)): return all(f(x + y) == f(x) + f(y) for x in W for y in W)
def _cauchy_sol(f, W=range(-8, 9)): return any(all(f(n) == c * n for n in W) for c in W)


def fe_additive_parts(w):
    """(the type of the functions, the statement of fe_additive as a function of f, its proof script): shared with the problems that prove the Cauchy equation inline"""
    s = _syms(w); Z, Eq, F, Ex, zzero, zone = s.Z, s.Eq, s.F, s.Ex, s.zzero, s.zone
    fT = w.arrow(Z, Z)
    with w.nota.using(Z):
        def cond(f): return F([Z, Z], lambda x, y: Eq(f(x + y), f(x) + f(y)))
        def sol(f): return Ex(Z, w.lam(Z, lambda c: F([Z], lambda n: Eq(f(n), c * n))))
        def statement(f): return w.Iff(cond(f), sol(f))
        def script(p):
            p.intro('f'); p.split()
            p.intro('H')
            p.exists(p.T('f', lambda f_: f_(zone)))
            p.intro('n'); p.induction_on('Z.int_induction', 'n')
            p.linarith(('H', zzero, zzero))
            p.intro('m', 'hm'); p.linarith(('H', 'm', zone))
            p.intro('m', 'hm'); p.linarith(('H', p.T('m', lambda m_: m_ - 1), zone))
            p.intro('h'); p.cases('h', names=['c', 'hc'])
            p.intro('x', 'y')
            p.have('e1', ('hc', p.T('x y', lambda x_, y_: x_ + y_))); p.have('e2', ('hc', 'x')); p.have('e3', ('hc', 'y'))
            p.linear_combination((1, 'e1'), (-1, 'e2'), (-1, 'e3'))
    return fT, statement, script


@BANK.problem('fe_additive', text='Determine all functions f : Z -> Z such that f(x + y) = f(x) + f(y) for all integers x and y.', source='folklore (Cauchy)', topics=('functional-equation', 'induction'), level=3, kind='answer',
              answer='f(n) = c n for an integer c', spec=Spec(('fn',), (), (_cauchy_cond, _cauchy_sol)),
              notes='formalised as an equivalence: f is additive iff f(n) = c n for some integer c; the proof is an induction on n in both directions, with c = f(1)')
def _(w, b):
    fT, statement, script = fe_additive_parts(w)
    thm(w, 'fe_additive', [fT], statement)(script)


@BANK.problem('no_f_f_succ', text='Prove that there is no function f : Z -> Z with f(f(n)) = n + 1 for all integers n.', source='folklore', topics=('functional-equation', 'induction', 'parity'), level=3,
              spec=Spec(('fn',), (), lambda f: not all(f(f(n)) == n + 1 for n in range(-8, 9)), atom=(lambda f, n: f(f(n)), lambda f, n: n + 1)),
              notes='applying f to the equation gives f(n + 1) = f(n) + 1, so f(n) = n + f(0), and then f(f(0)) = 2 f(0) = 1')
def _(w, b):
    s = _syms(w); Z, Eq, F, zzero, zone, zle = s.Z, s.Eq, s.F, s.zzero, s.zone, s.zle
    fT = w.arrow(Z, Z)
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'no_f_f_succ', [fT], lambda f: ~F([Z], lambda n: Eq(f(f(n)), n + 1)))
        def _(p):
            p.intro('f', 'H')
            def step(q):
                q.intro('n')
                q.have('e1', ('H', q.T('f n', lambda f_, n_: f_(n_))))
                q.rw_at('e1', 'H', 'n'); q.exact('e1')
            p.have('hs', p.T('f', lambda f_: F([Z], lambda n: Eq(f_(n + 1), f_(n) + 1))), step)
            def lin(q):
                q.intro('n'); q.induction_on('Z.int_induction', 'n')
                q.linarith()
                q.intro('m', 'hm'); q.linarith(('hs', 'm'))
                q.intro('m', 'hm'); q.linarith(('hs', q.T('m', lambda m_: m_ - 1)))
            p.have('hl', p.T('f', lambda f_: F([Z], lambda n: Eq(f_(n), n + f_(zzero)))), lin)
            p.have('e2', ('hl', p.T('f', lambda f_: f_(zzero)))); p.have('e3', ('H', zzero))
            p.obtain(['c', 'c'], ('Z.le_or_lt', lit(1), p.T('f', lambda f_: f_(zzero))))
            p.linarith(); p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ descent
@BANK.problem('sqrt3_irrational', text='Prove that sqrt(3) is irrational: there are no integers p, q with q /= 0 and p^2 = 3 q^2.', source='Euclid', origin='folklore', topics=('number-theory', 'descent', 'divisibility'), level=3,
              helpers=('Z.sqrt3_aux',),
              spec=Spec(('Z', 'Z'), (lambda p, q: q != 0,), lambda p, q: p * p != 3 * q * q, atom=(lambda p, q: p * p, lambda p, q: 3 * q * q)),
              notes='descent by strong induction on q >= 0: p^2 = 3 q^2 makes 3 divide p, then q, and p/3, q/3 solve the same equation')
def _(w, b):
    s = _syms(w); Z, Eq, F, zle, zlt, zzero, D = s.Z, s.Eq, s.F, s.zle, s.zlt, s.zzero, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        def Psq(q): return zlt(zzero, q) >> F([Z], lambda p: ~Eq(p * p, 3 * (q * q)))

        @thm(w, 'Z.sqrt3_aux', [Z, lambda q: zle(zzero, q)], lambda q, h: Psq(q))
        def aux_script(p):
            p.intro('q0', 'h0')
            with p.focus():
                motive = w.lam(Z, Psq)
                Hst = F([Z, lambda n: zle(zzero, n), lambda n, hn: F([Z, lambda m: zle(zzero, m), lambda m, h1: zlt(m, n)], lambda m, h1, h2: Psq(m))], lambda n, hn, ih: Psq(n))
            def H(q):
                q.intro('q', 'hq', 'IH', 'hpos', 'p', 'hpp')
                q.have('h3p', q.T('p', lambda p_: D(lit(3), p_ ** 2 + zzero ** 2)), lambda r: (r.exists(r.T('q', lambda q_: q_ * q_)), r.linear_combination((1, 'hpp'))))
                q.obtain(['h3', 'h30'], ('three_dvd_sum_squares', 'p', zzero, 'h3p'))
                q.cases('h3', names=['r', 'hr'])
                q.rw_at('hpp', 'hr')
                q.have('hqq', q.T('q r', lambda q_, r_: Eq(q_ * q_, 3 * (r_ * r_))), lambda t: t.linarith())
                q.have('h3q', q.T('q', lambda q_: D(lit(3), q_ ** 2 + zzero ** 2)), lambda t: (t.exists(t.T('r', lambda r_: r_ * r_)), t.linear_combination((1, 'hqq'))))
                q.obtain(['h4', 'h40'], ('three_dvd_sum_squares', 'q', zzero, 'h3q'))
                q.cases('h4', names=['s', 'hs'])
                q.have('spos', q.T('s', lambda s_: zlt(zzero, s_)), lambda t: t.linarith())
                q.have('slt', q.T('s q', lambda s_, q_: zlt(s_, q_)), lambda t: t.linarith())
                q.have('s0', q.T('s', lambda s_: zle(zzero, s_)), lambda t: t.linarith())
                q.rw_at('hqq', 'hs')
                q.have('hrs', q.T('r s', lambda r_, s_: Eq(r_ * r_, 3 * (s_ * s_))), lambda t: t.linarith())
                q.exact(('IH', 's', 's0', 'slt', 'spos', 'r', 'hrs'))
            p.have('H', Hst, H)
            p.exact(lambda q: w['Z.strong_induction'](motive, zzero, q.h('H'), q.h('q0'), q.h('h0')))

        @thm(w, 'sqrt3_irrational', [Z, Z, lambda p, q: ~Eq(q, zzero)], lambda p, q, h: ~Eq(p * p, 3 * (q * q)))
        def _(p):
            p.have('Z.sqrt3_aux', w.type_of(w['Z.sqrt3_aux']), aux_script)
            p.intro('p', 'q', 'h', 'hpp')
            p.obtain(['c', 'c'], ('Z.le_or_lt', zzero, 'q'))
            p.obtain(['c2', 'c2'], ('Z.lt_or_eq_of_le', zzero, 'q', 'c'))
            p.exact(('Z.sqrt3_aux', 'q', 'c', 'c2', 'p', 'hpp'))
            p.apply('h'); p.linarith()
            p.have('c3', p.T('q', lambda q_: zlt(zzero, -q_)), lambda r: r.linarith())
            p.have('c4', p.T('q', lambda q_: zle(zzero, -q_)), lambda r: r.linarith())
            p.have('hpp2', p.T('p q', lambda p_, q_: Eq(p_ * p_, 3 * ((-q_) * (-q_)))), lambda r: r.linarith())
            p.exact(('Z.sqrt3_aux', p.T('q', lambda q_: -q_), 'c4', 'c3', 'p', 'hpp2'))


@BANK.problem('x2_y2_3z2', text='Prove that the only integers x, y, z with x^2 + y^2 = 3 z^2 are x = y = z = 0.', source='folklore (infinite descent)', topics=('number-theory', 'descent', 'diophantine', 'modular'), level=4,
              helpers=('Z.x2_y2_3z2_aux',),
              spec=Spec(('Z', 'Z', 'Z'), (lambda x, y, z: x * x + y * y == 3 * z * z,), lambda x, y, z: x == 0 and y == 0 and z == 0,
                        atoms=[(lambda x, y, z: x, lambda x, y, z: 0), (lambda x, y, z: y, lambda x, y, z: 0), (lambda x, y, z: z, lambda x, y, z: 0)], hatoms=[(lambda x, y, z: x * x + y * y, lambda x, y, z: 3 * z * z)], points=5000),
              notes='descent by strong induction on z >= 0: 3 divides x^2 + y^2, so x and y; then z^2 = 3 (a^2 + b^2) makes 3 divide z, and (a, b, z/3) solve the same equation')
def _(w, b):
    s = _syms(w); Z, Eq, F, zle, zlt, zzero, D = s.Z, s.Eq, s.F, s.zle, s.zlt, s.zzero, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        def Pz(z): return F([Z, Z], lambda x, y: Eq(x * x + y * y, 3 * (z * z)) >> Eq(z, zzero))

        @thm(w, 'Z.x2_y2_3z2_aux', [Z, lambda z: zle(zzero, z)], lambda z, h: Pz(z))
        def aux_script(p):
            p.intro('z0', 'h0')
            with p.focus():
                motive = w.lam(Z, Pz)
                Hst = F([Z, lambda n: zle(zzero, n), lambda n, hn: F([Z, lambda m: zle(zzero, m), lambda m, h1: zlt(m, n)], lambda m, h1, h2: Pz(m))], lambda n, hn, ih: Pz(n))
            def H(q):
                # 3 | x^2 + y^2 gives x = 3a and y = 3d; then z^2 = 3 (a^2 + d^2) gives z = 3e and a^2 + d^2 = 3 e^2 with 0 <= e < z
                q.intro('z', 'hz', 'IH', 'x', 'y', 'hxy')
                q.obtain(['c', 'c'], ('Z.lt_or_eq_of_le', zzero, 'z', 'hz'))
                q.have('h3s', q.T('x y', lambda x_, y_: D(lit(3), x_ ** 2 + y_ ** 2)), lambda r: (r.exists(r.T('z', lambda z_: z_ * z_)), r.linear_combination((1, 'hxy'))))
                q.obtain(['h3x', 'h3y'], ('three_dvd_sum_squares', 'x', 'y', 'h3s'))
                q.cases('h3x', names=['a', 'ha']); q.cases('h3y', names=['d', 'hd'])
                q.rw_at('hxy', 'ha'); q.rw_at('hxy', 'hd')
                q.have('hzz', q.T('z a d', lambda z_, a_, d_: Eq(z_ * z_, 3 * (a_ * a_ + d_ * d_))), lambda t: t.linarith())
                q.have('h3z', q.T('z', lambda z_: D(lit(3), z_ ** 2 + zzero ** 2)), lambda t: (t.exists(t.T('a d', lambda a_, d_: a_ * a_ + d_ * d_)), t.linear_combination((1, 'hzz'))))
                q.obtain(['h4', 'h40'], ('three_dvd_sum_squares', 'z', zzero, 'h3z'))
                q.cases('h4', names=['e', 'he'])
                q.have('slt', q.T('e z', lambda e_, z_: zlt(e_, z_)), lambda t: t.linarith())
                q.have('s0', q.T('e', lambda e_: zle(zzero, e_)), lambda t: t.linarith())
                q.rw_at('hzz', 'he')
                q.have('had', q.T('a d e', lambda a_, d_, e_: Eq(a_ * a_ + d_ * d_, 3 * (e_ * e_))), lambda t: t.linarith())
                q.have('h5', ('IH', 'e', 's0', 'slt', 'a', 'd', 'had'))
                q.linarith()
                q.linarith()
            p.have('H', Hst, H)
            p.exact(lambda q: w['Z.strong_induction'](motive, zzero, q.h('H'), q.h('z0'), q.h('h0')))

        @thm(w, 'x2_y2_3z2', [Z, Z, Z, lambda x, y, z: Eq(x * x + y * y, 3 * (z * z))], lambda x, y, z, h: Eq(x, zzero) & (Eq(y, zzero) & Eq(z, zzero)))
        def _(p):
            p.have('Z.x2_y2_3z2_aux', w.type_of(w['Z.x2_y2_3z2_aux']), aux_script)
            p.intro('x', 'y', 'z', 'h')

            def hz_script(r):
                r.obtain(['c', 'c'], ('Z.le_or_lt', zzero, 'z'))
                r.exact(('Z.x2_y2_3z2_aux', 'z', 'c', 'x', 'y', 'h'))
                r.have('c4', r.T('z', lambda z_: zle(zzero, -z_)), lambda t: t.linarith())
                r.have('h2', r.T('x y z', lambda x_, y_, z_: Eq(x_ * x_ + y_ * y_, 3 * ((-z_) * (-z_)))), lambda t: t.linarith())
                r.have('h3', ('Z.x2_y2_3z2_aux', r.T('z', lambda z_: -z_), 'c4', 'x', 'y', 'h2'))
                r.linarith()
            p.have('hz', p.T('z', lambda z_: Eq(z_, zzero)), hz_script)
            p.have('h0', p.T('x y', lambda x_, y_: Eq(x_ ** 2 + y_ ** 2, zzero)), lambda t: t.linear_combination((1, 'h'), (t.T('z', lambda z_: 3 * z_), 'hz')))
            p.obtain(['hx', 'hy'], ('sum_squares_zero', 'x', 'y', 'h0'))
            p.split(); p.exact('hx')
            p.split(); p.exact('hy'); p.exact('hz')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ Pell: infinitely many solutions
@BANK.problem('pell_infinitely_many', text='Prove that the equation x^2 - 2y^2 = 1 has solutions in integers x, y that are larger than any given bound m >= 0.', source='folklore (Pell)', topics=('number-theory', 'diophantine', 'induction', 'construction'), level=3,
              spec=Spec(('Z',), (lambda m: m >= 0,), lambda m: any(x > m and y > m and x * x - 2 * y * y == 1 for x in range(-20, 21) for y in range(-20, 21)), windows={'z': 8, 'ex': 20},
                        body=(('Z', 'Z'), lambda m, x, y: m < x and m < y and x * x - 2 * y * y == 1),
                        atoms=[(lambda m, x, y: m, lambda m, x, y: x), (lambda m, x, y: m, lambda m, x, y: y), (lambda m, x, y: x * x - 2 * y * y, lambda m, x, y: 1)]),
              notes='from a solution (x, y) the pair (3x + 4y, 2x + 3y) is a larger one: an induction on m from 0, starting at (3, 2)')
def _(w, b):
    s = _syms(w); Z, Eq, Ex, zzero, zone, zle, zlt = s.Z, s.Eq, s.Ex, s.zzero, s.zone, s.zle, s.zlt
    with w.nota.using(Z):
        lit = w.nota.lit
        def sol(m): return Ex(Z, w.lam(Z, lambda x: Ex(Z, w.lam(Z, lambda y: AND(zlt(m, x), zlt(m, y), Eq(x * x - 2 * (y * y), zone))))))
        @thm(w, 'pell_infinitely_many', [Z, lambda m: zle(zzero, m)], lambda m, h: sol(m))
        def _(p):
            p.intro('m', 'hm')
            p.induction_on('Z.induction_from', 'm', extra=(zzero,))
            p.exists(lit(3)); p.exists(lit(2)); p.split(); p.linarith(); p.split(); p.linarith(); p.linarith()
            p.intro('k', 'hk', 'ih')
            destruct(p, 'ih', ['x', 'y', 'h1', 'h2', 'h3'])
            p.exists(p.T('x y', lambda x_, y_: 3 * x_ + 4 * y_)); p.exists(p.T('x y', lambda x_, y_: 2 * x_ + 3 * y_))
            p.split(); p.linarith(); p.split(); p.linarith(); p.linear_combination((1, 'h3'))
            p.assumption()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ x^2 - 2y^2 = 3
@BANK.problem('no_x2_minus_2y2_eq_3', text='Prove that the equation x^2 - 2y^2 = 3 has no solution in integers.', source='folklore', topics=('number-theory', 'diophantine', 'modular', 'divisibility'), level=3,
              spec=Spec(('Z', 'Z'), (), lambda x, y: x * x - 2 * y * y != 3, atom=(lambda x, y: x * x - 2 * y * y, lambda x, y: 3)),
              notes='x^2 + y^2 = 3 (y^2 + 1), so 3 divides x and y, and then 9 divides x^2 - 2y^2 = 3')
def _(w, b):
    s = _syms(w); Z, Eq, zzero, zle, zlt, D = s.Z, s.Eq, s.zzero, s.zle, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'no_x2_minus_2y2_eq_3', [Z, Z], lambda x, y: ~Eq(x * x - 2 * (y * y), lit(3)))
        def _(p):
            p.intro('x', 'y', 'h')
            p.have('h3', p.T('x y', lambda x_, y_: D(lit(3), x_ ** 2 + y_ ** 2)), lambda r: (r.exists(r.T('y', lambda y_: y_ * y_ + 1)), r.linear_combination((1, 'h'))))
            p.obtain(['hx', 'hy'], ('three_dvd_sum_squares', 'x', 'y', 'h3'))
            p.cases('hx', names=['a', 'ha']); p.cases('hy', names=['c', 'hc'])
            p.rw_at('h', 'ha'); p.rw_at('h', 'hc')
            p.have('hk', p.T('a c', lambda a_, c_: Eq(3 * (a_ * a_ - 2 * (c_ * c_)), lit(1))), lambda r: r.linarith())
            p.obtain(['c1', 'c1'], ('Z.le_or_lt', p.T('a c', lambda a_, c_: a_ * a_ - 2 * (c_ * c_)), zzero))
            p.linarith(); p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ three squares mod 8
@BANK.problem('three_squares_mod_eight', text='Prove that no integer of the form 8n + 7 is a sum of three squares: x^2 + y^2 + z^2 /= 8n + 7 for all integers x, y, z, n.', source='folklore (Legendre)', topics=('number-theory', 'modular', 'diophantine', 'case-analysis'), level=3,
              helpers=('sq_mod_eight',),
              spec=Spec(('Z', 'Z', 'Z', 'Z'), (), lambda x, y, z, n: x * x + y * y + z * z != 8 * n + 7, windows={'z': 4}, points=7000, atom=(lambda x, y, z, n: x * x + y * y + z * z, lambda x, y, z, n: 8 * n + 7)),
              notes='a square is 8k, 8k + 1 or 8k + 4; the 27 combinations leave 8 (a + b + c - n) = 7 - r, which no integer solves: the three cases K < 0, K = 0, K > 0 of K = a + b + c - n close them by linear arithmetic')
def _(w, b):
    s = _syms(w); Z, Eq, Ex, zzero, zlt = s.Z, s.Eq, s.Ex, s.zzero, s.zlt
    with w.nota.using(Z):
        lit = w.nota.lit

        @thm(w, 'sq_mod_eight', [Z], lambda x: Ex(Z, w.lam(Z, lambda k: Eq(x ** 2, 8 * k) | (Eq(x ** 2, 8 * k + 1) | Eq(x ** 2, 8 * k + 4)))))
        def sq8_script(p):
            p.intro('x')
            p.have('h4', p.T('', lambda: zlt(zzero, lit(4))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(4), 'h4', 'x'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 3)
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 2 * q ** 2)); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 2 * q ** 2 + q)); p.right(); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 2 * q ** 2 + 2 * q)); p.right(); p.right(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 2 * q ** 2 + 3 * q + 1)); p.right(); p.left(); p.ring()

        @thm(w, 'three_squares_mod_eight', [Z, Z, Z, Z], lambda x, y, z, n: ~Eq(x ** 2 + y ** 2 + z ** 2, 8 * n + 7))
        def _(p):
            p.have('sq_mod_eight', w.type_of(w['sq_mod_eight']), sq8_script)
            p.intro('x', 'y', 'z', 'n', 'h')
            p.obtain(['a', 'ha'], ('sq_mod_eight', 'x')); p.obtain(['c', 'hc'], ('sq_mod_eight', 'y')); p.obtain(['d', 'hd'], ('sq_mod_eight', 'z'))
            def leaves(q, hyps):
                if not hyps: q.linarith(); return
                hh = hyps[0]
                q.cases(hh, names=[hh + '1', hh + '23']); leaves(q, hyps[1:])
                q.cases(hh + '23', names=[hh + '2', hh + '3']); leaves(q, hyps[1:]); leaves(q, hyps[1:])
            p.obtain(['t1', 't23'], ('Z.lt_trichotomy', p.T('a c d n', lambda a_, c_, d_, n_: a_ + c_ + d_ - n_), zzero))
            leaves(p, ['ha', 'hc', 'hd'])
            p.cases('t23', names=['t2', 't3'])
            leaves(p, ['ha', 'hc', 'hd']); leaves(p, ['ha', 'hc', 'hd'])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ Schur, Cauchy, Chebyshev
def _schur(a, b, c): return a * (a - b) * (a - c) + b * (b - a) * (b - c) + c * (c - a) * (c - b)


def schur_parts(w):
    """(the types of the binders of schur_ordered and of schur_nonneg, their statements, their proof scripts): shared with the problems that prove Schur's inequality inline"""
    s = _syms(w); Z, zle, zlt, zzero = s.Z, s.zle, s.zlt, s.zzero
    with w.nota.using(Z):
        def schur(a, b, c): return a * (a - b) * (a - c) + b * (b - a) * (b - c) + c * (c - a) * (c - b)
        ordered_doms = [Z, Z, Z, lambda a, b, c: zle(zzero, c), lambda a, b, c, h: zle(c, b), lambda a, b, c, h, h2: zle(b, a)]
        ordered_stmt = lambda a, b, c, h1, h2, h3: zle(zzero, schur(a, b, c))
        main_doms = [Z, Z, Z, lambda a, b, c: zle(zzero, a), lambda a, b, c, h: zle(zzero, b), lambda a, b, c, h, h2: zle(zzero, c)]
        main_stmt = lambda a, b, c, h1, h2, h3: zle(zzero, schur(a, b, c))

        def ordered_script(p):
            p.intro('a', 'b', 'c', 'hc', 'hcb', 'hba')
            p.have('f1', p.T('a b c', lambda a_, b_, c_: zle(zzero, a_ + b_ - c_)), lambda q: q.linarith())
            p.have('f2', p.T('a c', lambda a_, c_: zle(zzero, a_ - c_)), lambda q: q.linarith())
            p.have('f3', p.T('b c', lambda b_, c_: zle(zzero, b_ - c_)), lambda q: q.linarith())
            p.have('g1', ('Z.mul_nonneg', p.T('a b', lambda a_, b_: (a_ - b_) * (a_ - b_)), p.T('a b c', lambda a_, b_, c_: a_ + b_ - c_), ('Z.sq_nonneg', p.T('a b', lambda a_, b_: a_ - b_)), 'f1'))
            p.have('g2', ('Z.mul_nonneg', p.T('a c', lambda a_, c_: a_ - c_), p.T('b c', lambda b_, c_: b_ - c_), 'f2', 'f3'))
            p.have('g3', ('Z.mul_nonneg', 'c', p.T('a b c', lambda a_, b_, c_: (a_ - c_) * (b_ - c_)), 'hc', 'g2'))
            p.linarith()

        def main_script(p):
            p.have('schur_ordered', w.type_of(w['schur_ordered']) if 'schur_ordered' in w.gid else w.forall_(ordered_doms, ordered_stmt), ordered_script)
            p.intro('a', 'b', 'c', 'ha', 'hb', 'hc')
            def leaf(q, x, y, z, hz):
                q.have('o2', q.T(z + ' ' + y, lambda z_, y_: zle(z_, y_)), lambda r: r.linarith())
                q.have('o3', q.T(y + ' ' + x, lambda y_, x_: zle(y_, x_)), lambda r: r.linarith())
                q.linarith(('schur_ordered', x, y, z, hz, 'o2', 'o3'))
            p.obtain(['o', 'o'], ('Z.le_or_lt', 'a', 'b'))
            p.obtain(['o', 'o'], ('Z.le_or_lt', 'b', 'c'))
            leaf(p, 'c', 'b', 'a', 'ha')
            p.obtain(['o', 'o'], ('Z.le_or_lt', 'a', 'c'))
            leaf(p, 'b', 'c', 'a', 'ha')
            leaf(p, 'b', 'a', 'c', 'hc')
            p.obtain(['o', 'o'], ('Z.le_or_lt', 'b', 'c'))
            p.obtain(['o', 'o'], ('Z.le_or_lt', 'a', 'c'))
            leaf(p, 'c', 'a', 'b', 'hb')
            leaf(p, 'a', 'c', 'b', 'hb')
            leaf(p, 'a', 'b', 'c', 'hc')
    return (ordered_doms, ordered_stmt, ordered_script), (main_doms, main_stmt, main_script)


@BANK.problem('schur_nonneg', text='Schur\'s inequality: for all non-negative integers a, b, c: a(a - b)(a - c) + b(b - a)(b - c) + c(c - a)(c - b) >= 0.', source='folklore (Schur)', topics=('inequality', 'case-analysis'), level=3,
              helpers=('schur_ordered',),
              spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, c: a >= 0, lambda a, b, c: b >= 0, lambda a, b, c: c >= 0), lambda a, b, c: _schur(a, b, c) >= 0, windows={'z': 8},
                        atom=(lambda a, b, c: 0, lambda a, b, c: _schur(a, b, c)), hatoms=[(lambda a, b, c: 0, lambda a, b, c: a), (lambda a, b, c: 0, lambda a, b, c: b), (lambda a, b, c: 0, lambda a, b, c: c)]),
              notes='for a >= b >= c the expression is (a - b)^2 (a + b - c) + c (a - c)(b - c); the six orders of a, b, c reduce to it because the expression is symmetric')
def _(w, b):
    (od, ost, osc), (md, mst, msc) = schur_parts(w)
    thm(w, 'schur_ordered', od, ost)(osc)
    thm(w, 'schur_nonneg', md, mst)(msc)


@BANK.problem('ineq_cauchy_three', text='For all integers a, b, c, x, y, z: (ax + by + cz)^2 <= (a^2 + b^2 + c^2)(x^2 + y^2 + z^2).', source='folklore (Cauchy-Schwarz)', topics=('inequality',), level=2,
              spec=Spec(('Z', 'Z', 'Z', 'Z', 'Z', 'Z'), (), lambda a, b, c, x, y, z: (a * x + b * y + c * z) ** 2 <= (a * a + b * b + c * c) * (x * x + y * y + z * z), windows={'z': 2}, points=16000,
                        atom=(lambda a, b, c, x, y, z: (a * x + b * y + c * z) ** 2, lambda a, b, c, x, y, z: (a * a + b * b + c * c) * (x * x + y * y + z * z))))
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_cauchy_three', [Z, Z, Z, Z, Z, Z], lambda a, c, d, x, y, z: zle((a * x + c * y + d * z) ** 2, (a ** 2 + c ** 2 + d ** 2) * (x ** 2 + y ** 2 + z ** 2)))
        def _(p):
            p.intro('a', 'c', 'd', 'x', 'y', 'z')
            p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') * q.h('y') - q.h('c') * q.h('x')), ('Z.sq_nonneg', lambda q: q.h('c') * q.h('z') - q.h('d') * q.h('y')), ('Z.sq_nonneg', lambda q: q.h('a') * q.h('z') - q.h('d') * q.h('x')))


@BANK.problem('ineq_chebyshev_cubes', text='For all non-negative integers a, b, c: (a + b + c)(a^2 + b^2 + c^2) <= 3 (a^3 + b^3 + c^3).', source='folklore (Chebyshev)', topics=('inequality',), level=2,
              spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, c: a >= 0, lambda a, b, c: b >= 0, lambda a, b, c: c >= 0), lambda a, b, c: (a + b + c) * (a * a + b * b + c * c) <= 3 * (a ** 3 + b ** 3 + c ** 3), windows={'z': 8},
                        atom=(lambda a, b, c: (a + b + c) * (a * a + b * b + c * c), lambda a, b, c: 3 * (a ** 3 + b ** 3 + c ** 3)), hatoms=[(lambda a, b, c: 0, lambda a, b, c: a), (lambda a, b, c: 0, lambda a, b, c: b), (lambda a, b, c: 0, lambda a, b, c: c)]),
              notes='3 (a^3 + b^3 + c^3) - (a + b + c)(a^2 + b^2 + c^2) = (a + b)(a - b)^2 + (b + c)(b - c)^2 + (c + a)(c - a)^2')
def _(w, b):
    s = _syms(w); Z, zle, zzero = s.Z, s.zle, s.zzero
    with w.nota.using(Z):
        @thm(w, 'ineq_chebyshev_cubes', [Z, Z, Z, lambda a, b, c: zle(zzero, a), lambda a, b, c, h: zle(zzero, b), lambda a, b, c, h, h2: zle(zzero, c)], lambda a, b, c, h1, h2, h3: zle((a + b + c) * (a ** 2 + b ** 2 + c ** 2), 3 * (a ** 3 + b ** 3 + c ** 3)))
        def _(p):
            p.intro('a', 'b', 'c', 'ha', 'hb', 'hc')
            def pair(q, nm, x, y, hx, hy):
                q.have(nm + 's', q.T(x + ' ' + y, lambda x_, y_: zle(zzero, x_ + y_)), lambda r: r.linarith())
                q.have(nm, ('Z.mul_nonneg', q.T(x + ' ' + y, lambda x_, y_: x_ + y_), q.T(x + ' ' + y, lambda x_, y_: (x_ - y_) * (x_ - y_)), nm + 's', ('Z.sq_nonneg', q.T(x + ' ' + y, lambda x_, y_: x_ - y_))))
            pair(p, 'g1', 'a', 'b', 'ha', 'hb'); pair(p, 'g2', 'b', 'c', 'hb', 'hc'); pair(p, 'g3', 'c', 'a', 'hc', 'ha')
            p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ f(x + f(y)) = f(x) + y
def _fx_cond(f, W=range(-8, 9)): return all(f(x + f(y)) == f(x) + y for x in W for y in W)
def _fx_sol(f, W=range(-8, 9)): return all(f(n) == n for n in W) or all(f(n) == -n for n in W)


@BANK.problem('fe_f_x_plus_f_y', text='Determine all functions f : Z -> Z such that f(x + f(y)) = f(x) + y for all integers x and y.', source='folklore', topics=('functional-equation', 'induction'), level=4, kind='answer',
              answer='f(n) = n for all n, and f(n) = -n for all n', spec=Spec(('fn',), (), (_fx_cond, _fx_sol), extra_fns=(lambda n: {1: 2, 2: 1}.get(n, n),)),
              notes='f(f(y)) = f(0) + y makes f injective, so f(0) = 0 and f(f(y)) = y; replacing y by f(y) makes f additive (fe_additive), so f(n) = c n with c^2 = 1')
def _(w, b):
    s = _syms(w); Z, Eq, F, Ex, zzero, zone = s.Z, s.Eq, s.F, s.Ex, s.zzero, s.zone
    fT = w.arrow(Z, Z)
    with w.nota.using(Z):
        def cond(f): return F([Z, Z], lambda x, y: Eq(f(x + f(y)), f(x) + y))
        def sol(f): return F([Z], lambda n: Eq(f(n), n)) | F([Z], lambda n: Eq(f(n), -n))
        @thm(w, 'fe_f_x_plus_f_y', [fT], lambda f: w.Iff(cond(f), sol(f)))
        def _(p):
            p.intro('f'); p.split()
            p.intro('H')
            def s1(q):
                q.intro('y'); q.have('e', ('H', zzero, 'y')); q.rw_at('e', 'Z.zero_add'); q.exact('e')
            p.have('h1', p.T('f', lambda f_: F([Z], lambda y: Eq(f_(f_(y)), f_(zzero) + y))), s1)
            def inj(q):
                q.intro('a', 'c', 'hac')
                q.have('e1', ('h1', 'a')); q.have('e2', ('h1', 'c'))
                q.rw_at('e1', 'hac'); q.linarith()
            p.have('inj', p.T('f', lambda f_: F([Z, Z], lambda a, c: Eq(f_(a), f_(c)) >> Eq(a, c))), inj)
            def s0(q):
                q.have('e2', q.T('f', lambda f_: Eq(f_(f_(zzero)), f_(zzero))), lambda r: r.linarith(('h1', zzero)))
                q.exact(('inj', q.T('f', lambda f_: f_(zzero)), zzero, 'e2'))
            p.have('f0', p.T('f', lambda f_: Eq(f_(zzero), zzero)), s0)
            p.have('h2', p.T('f', lambda f_: F([Z], lambda y: Eq(f_(f_(y)), y))), lambda q: (q.intro('y'), q.linarith(('h1', 'y'), 'f0')))
            def sadd(q):
                q.intro('x', 'y'); q.have('e', ('H', 'x', p.T('f y', lambda f_, y_: f_(y_)))); q.rw_at('e', 'h2'); q.exact('e')
            p.have('hadd', p.T('f', lambda f_: F([Z, Z], lambda x, y: Eq(f_(x + y), f_(x) + f_(y)))), sadd)
            p.have('fe_additive', w.type_of(w['fe_additive']), fe_additive_parts(w)[2])
            p.obtain(['fw', 'bw'], ('fe_additive', 'f'))
            p.obtain(['c', 'hc'], ('fw', 'hadd'))
            p.have('e1', ('h2', zone)); p.have('e2', ('hc', zone)); p.have('e3', ('hc', p.T('f', lambda f_: f_(zone))))
            p.have('E', p.T('f c', lambda f_, c_: Eq(c_ * f_(zone), zone)), lambda q: q.linarith())
            p.have('e4', p.T('c', lambda c_: Eq(c_ * c_, zone)), lambda q: q.linear_combination((1, 'E'), (q.T('c', lambda c_: -c_), 'e2')))
            p.have('e5', p.T('c', lambda c_: Eq((c_ - 1) * (c_ + 1), zzero)), lambda q: q.linarith())
            p.obtain(['u', 'u'], ('Z.mul_eq_zero', p.T('c', lambda c_: c_ - 1), p.T('c', lambda c_: c_ + 1), 'e5'))
            p.left(); p.intro('n'); p.have('hn', ('hc', 'n')); p.linear_combination((1, 'hn'), (p.T('n', lambda n_: n_), 'u'))
            p.right(); p.intro('n'); p.have('hn', ('hc', 'n')); p.linear_combination((1, 'hn'), (p.T('n', lambda n_: n_), 'u'))
            p.intro('h'); p.cases('h', names=['hid', 'hneg'])
            p.intro('x', 'y'); p.linarith(('hid', p.T('f x y', lambda f_, x_, y_: x_ + f_(y_))), ('hid', 'y'), ('hid', 'x'))
            p.intro('x', 'y'); p.linarith(('hneg', p.T('f x y', lambda f_, x_, y_: x_ + f_(y_))), ('hneg', 'y'), ('hneg', 'x'))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ IMO 1982 P1
def _i82_h1(f): return all(f(n) >= 0 for n in range(1, 9))
def _i82_h2(f): return all(f(m + n) - f(m) - f(n) in (0, 1) for m in range(1, 9) for n in range(1, 9))
def _i82_e1(n): return n // 3 if n > 0 else 0                                  # a solution
def _i82_e2(n): return 1 if n == 0 else _i82_e1(n)                             # f(0) = 1: only the instances with m = 0 or n = 0 notice
def _i82_e3(n): return -1 if n < 0 else _i82_e1(n)                             # negative values: only a non-negativity for every n notices
def _i82_e4(n): return 3333 if n == 9999 else (n // 4 if n > 0 else 0)         # f(3) = 0: only f(3) >= 0 for f(3) > 0 notices
def _i82_e5(n): return n if n >= 2 else 0                                     # f(m + n) - f(m) - f(n) is 0 or 1 for every pair but (1, 1): only the guard 1 < m instead of 0 < m (which leaves out that pair) notices


@BANK.problem('imo1982_p1', text='The function f(n) is defined for all positive integers n and takes non-negative integer values. Also, for all m, n: f(m + n) - f(m) - f(n) = 0 or 1, and f(2) = 0, f(3) > 0, f(9999) = 3333. Determine f(1982).',
              source='IMO 1982 P1', verified='answer-checked', year=1982, topics=('functional-equation', 'inequality', 'induction', 'bounding'), level=4, kind='answer', answer='f(1982) = 660',
              spec=Spec(('fn',), (_i82_h1, _i82_h2, lambda f: f(2) == 0, lambda f: f(3) > 0, lambda f: f(9999) == 3333), lambda f: f(1982) == 660, extra_fns=(_i82_e1, _i82_e2, _i82_e3, _i82_e4, _i82_e5),
                        atom=(lambda f: f(1982), lambda f: 660)),
              notes='f is superadditive, so f(3k) >= k by induction; f(9999) = 3333 forces f(1980) = 660; and f(1982) >= 661 would give f(9999) >= 5 f(1982) + f(89) >= 3334')
def _(w, b):
    s = _syms(w); Z, Eq, F, zle, zlt, zzero = s.Z, s.Eq, s.F, s.zle, s.zlt, s.zzero
    fT = w.arrow(Z, Z)
    with w.nota.using(Z):
        lit = w.nota.lit
        def H1(f): return F([Z, lambda n: zlt(zzero, n)], lambda n, hn: zle(zzero, f(n)))
        def H2(f): return F([Z, Z, lambda m, n: zlt(zzero, m), lambda m, n, h: zlt(zzero, n)], lambda m, n, h1, h2: Eq(f(m + n), f(m) + f(n)) | Eq(f(m + n), f(m) + f(n) + 1))
        @thm(w, 'imo1982_p1', [fT, lambda f: H1(f), lambda f, h1: H2(f), lambda f, h1, h2: Eq(f(lit(2)), zzero), lambda f, h1, h2, h3: zlt(zzero, f(lit(3))), lambda f, h1, h2, h3, h4: Eq(f(lit(9999)), lit(3333))],
             lambda f, h1, h2, h3, h4, h5: Eq(f(lit(1982)), lit(660)))
        def _(p):
            p.intro('f', 'hnn', 'hadd', 'h2', 'h3', 'h9999')
            for m in (1, 2, 3, 87, 89, 1980, 1982, 3964, 5946, 7928, 8019, 9910):
                p.have('pos%d' % m, p.T('', lambda m=m: zlt(zzero, lit(m))), lambda q: q.linarith())
            def sup(q):
                q.intro('m', 'n', 'hm', 'hn'); q.obtain(['e', 'e'], ('hadd', 'm', 'n', 'hm', 'hn')); q.linarith(); q.linarith()
            p.have('hs', p.T('f', lambda f_: F([Z, Z, lambda m, n: zlt(zzero, m), lambda m, n, h: zlt(zzero, n)], lambda m, n, h1, h2: zle(f_(m) + f_(n), f_(m + n)))), sup)
            def inst(nm, m, n):                                                    # f m + f n <= f (m + n)
                p.have(nm, p.T('f', lambda f_: zle(f_(lit(m)) + f_(lit(n)), f_(lit(m + n)))), lambda q: q.exact(('hs', lit(m), lit(n), 'pos%d' % m, 'pos%d' % n)))
            inst('s11', 1, 1)
            p.have('f1', p.T('f', lambda f_: Eq(f_(lit(1)), zzero)), lambda q: q.linarith(('hnn', lit(1), 'pos1')))
            p.have('i3', p.T('f', lambda f_: Eq(f_(lit(3)), f_(lit(2)) + f_(lit(1))) | Eq(f_(lit(3)), f_(lit(2)) + f_(lit(1)) + 1)), lambda q: q.exact(('hadd', lit(2), lit(1), 'pos2', 'pos1')))
            p.have('f3', p.T('f', lambda f_: Eq(f_(lit(3)), lit(1))), lambda q: (q.obtain(['e', 'e'], 'i3'), q.linarith(), q.linarith()))
            def l3(q):
                q.intro('k', 'hk')
                q.induction_on('Z.induction_from', 'k', extra=(lit(1),))
                q.have('b1', q.T('f', lambda f_: zle(lit(1), f_(lit(3)))), lambda r: r.linarith())
                q.exact('b1')
                q.intro('j', 'hj', 'ih')
                q.have('e', q.T('j', lambda j_: Eq(3 * (j_ + s.zone), 3 * j_ + 3)), lambda r: r.ring())
                q.rw('e')
                q.have('p3j', q.T('j', lambda j_: zlt(zzero, 3 * j_)), lambda r: r.linarith())
                q.have('hs3', ('hs', q.T('j', lambda j_: 3 * j_), lit(3), 'p3j', 'pos3'))
                q.linarith()
                q.assumption()
            p.have('l3', p.T('f', lambda f_: F([Z, lambda k: zle(lit(1), k)], lambda k, hk: zle(k, f_(3 * k)))), l3)
            for k in (29, 660, 2673):
                p.have('ge1_%d' % k, p.T('', lambda k=k: zle(lit(1), lit(k))), lambda q: q.linarith())
            p.have('a29', p.T('f', lambda f_: zle(lit(29), f_(lit(87)))), lambda q: q.exact(('l3', lit(29), 'ge1_29')))
            p.have('a660', p.T('f', lambda f_: zle(lit(660), f_(lit(1980)))), lambda q: q.exact(('l3', lit(660), 'ge1_660')))
            p.have('a2673', p.T('f', lambda f_: zle(lit(2673), f_(lit(8019)))), lambda q: q.exact(('l3', lit(2673), 'ge1_2673')))
            inst('s1', 1980, 8019); inst('s2', 1980, 2)
            inst('s3', 1982, 1982); inst('s4', 3964, 1982); inst('s5', 5946, 1982); inst('s6', 7928, 1982)
            inst('s7', 9910, 89); inst('s8', 87, 2)
            p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ IMO 1988 P6 (Vieta jumping)
@BANK.problem('imo1988_p6', text='Let a and b be positive integers such that ab + 1 divides a^2 + b^2. Show that (a^2 + b^2) / (ab + 1) is the square of an integer.', source='IMO 1988 P6', verified='answer-checked', year=1988,
              topics=('number-theory', 'divisibility', 'descent', 'diophantine', 'induction'), level=5, helpers=('imo1988_jump', 'imo1988_descent'),
              spec=Spec(('Z', 'Z'), (lambda a, b: a > 0, lambda a, b: b > 0, lambda a, b: _dvd(a * b + 1, a * a + b * b)), lambda a, b: any(a * a + b * b == m * m * (a * b + 1) for m in range(-12, 13)), windows={'z': 8, 'ex': 12},
                        body=(('Z',), lambda a, b, m: a * a + b * b == m * m * (a * b + 1)),
                        atoms=[(lambda a, b, m: a * a + b * b, lambda a, b, m: m * m * (a * b + 1))],
                        hatoms=[(lambda a, b: 0, lambda a, b: a), (lambda a, b: 0, lambda a, b: b), (lambda a, b: a * b + 1, lambda a, b: a * a + b * b)]),
              notes='Vieta jumping: for a >= b > 0 with a^2 + b^2 = k (ab + 1) the other root c = kb - a of the quadratic is 0 <= c < a; a strong induction on a + b ends at a zero, where k = a^2 or b^2')
def _(w, b):
    s = _syms(w); Z, Eq, F, Ex, zle, zlt, zzero, D = s.Z, s.Eq, s.F, s.Ex, s.zle, s.zlt, s.zzero, s.D
    with w.nota.using(Z):
        lit = w.nota.lit

        @thm(w, 'imo1988_jump', [Z, Z, Z, lambda a, b, k: zlt(zzero, b), lambda a, b, k, h: zle(b, a), lambda a, b, k, h, h2: zlt(zzero, k), lambda a, b, k, h, h2, h3: Eq(a * a + b * b, k * (a * b + 1))],
             lambda a, b, k, h1, h2, h3, h4: Ex(Z, w.lam(Z, lambda c: AND(zle(zzero, c), zlt(c, a), Eq(c * c + b * b, k * (c * b + 1))))))
        def jump_script(p):
            p.intro('a', 'b', 'k', 'hb', 'hba', 'hk', 'h')
            p.exists(p.T('k b a', lambda k_, b_, a_: k_ * b_ - a_))
            p.split()
            p.obtain(['c', 'c'], ('Z.le_or_lt', zzero, p.T('k b a', lambda k_, b_, a_: k_ * b_ - a_)))
            p.exact('c')
            p.exfalso()
            p.have('n1', p.T('k b a', lambda k_, b_, a_: zle(zzero, -((k_ * b_ - a_) + 1))), lambda q: q.linarith())
            p.have('n2', p.T('b', lambda b_: zle(zzero, b_)), lambda q: q.linarith())
            p.have('t1', ('Z.mul_nonneg', p.T('k b a', lambda k_, b_, a_: -((k_ * b_ - a_) + 1)), 'b', 'n1', 'n2'))
            p.have('n3', p.T('k b a', lambda k_, b_, a_: zle(zzero, -((k_ * b_ - a_) * b_ + 1))), lambda q: q.linarith())
            p.have('n4', p.T('k', lambda k_: zle(zzero, k_)), lambda q: q.linarith())
            p.have('t3', ('Z.mul_nonneg', 'k', p.T('k b a', lambda k_, b_, a_: -((k_ * b_ - a_) * b_ + 1)), 'n4', 'n3'))
            p.have('t4', ('Z.sq_nonneg', p.T('b', lambda b_: b_ - 1)))
            p.have('t5', ('Z.sq_nonneg', p.T('k b a', lambda k_, b_, a_: k_ * b_ - a_)))
            p.linarith()
            p.split()
            p.obtain(['d', 'd'], ('Z.le_or_lt', 'a', p.T('k b a', lambda k_, b_, a_: k_ * b_ - a_)))
            p.exfalso()
            p.have('m1', p.T('a', lambda a_: zle(zzero, a_)), lambda q: q.linarith())
            p.have('m2', p.T('k b a', lambda k_, b_, a_: zle(zzero, (k_ * b_ - a_) - a_)), lambda q: q.linarith())
            p.have('t6', ('Z.mul_nonneg', 'a', p.T('k b a', lambda k_, b_, a_: (k_ * b_ - a_) - a_), 'm1', 'm2'))
            p.have('m3', p.T('a b', lambda a_, b_: zle(zzero, a_ - b_)), lambda q: q.linarith())
            p.have('m4', p.T('a b', lambda a_, b_: zle(zzero, a_ + b_)), lambda q: q.linarith())
            p.have('t7', ('Z.mul_nonneg', p.T('a b', lambda a_, b_: a_ - b_), p.T('a b', lambda a_, b_: a_ + b_), 'm3', 'm4'))
            p.linarith()
            p.exact('d')
            p.linear_combination((1, 'h'))

        def P(s_): return F([Z, Z, Z], lambda a, b, k: Eq(a + b, s_) >> (zle(zzero, a) >> (zle(zzero, b) >> (zlt(zzero, k) >> (Eq(a * a + b * b, k * (a * b + 1)) >> Ex(Z, w.lam(Z, lambda m: Eq(k, m * m))))))))

        @thm(w, 'imo1988_descent', [Z, lambda s_: zle(zzero, s_)], lambda s_, h: P(s_))
        def descent_script(p):
            p.intro('s0', 'h0')
            with p.focus():
                motive = w.lam(Z, P)
                Hst = F([Z, lambda n: zle(zzero, n), lambda n, hn: F([Z, lambda m: zle(zzero, m), lambda m, h1: zlt(m, n)], lambda m, h1, h2: P(m))], lambda n, hn, ih: P(n))
            def H(q):
                q.intro('s', 'hs', 'IH', 'a', 'b', 'k', 'hab', 'ha', 'hb', 'hk', 'h')
                q.obtain(['cb', 'cb'], ('Z.lt_or_eq_of_le', zzero, 'b', 'hb'))
                q.obtain(['ca', 'ca'], ('Z.lt_or_eq_of_le', zzero, 'a', 'ha'))
                q.obtain(['o', 'o'], ('Z.le_or_lt', 'b', 'a'))
                destruct(q, ('imo1988_jump', 'a', 'b', 'k', 'cb', 'o', 'hk', 'h'), ['c', 'hc0', 'hca', 'hc'])
                q.have('hlt', q.T('c b s', lambda c_, b_, s_: zlt(c_ + b_, s_)), lambda r: r.linarith())
                q.have('hge', q.T('c b', lambda c_, b_: zle(zzero, c_ + b_)), lambda r: r.linarith())
                q.have('rr', q.T('c b', lambda c_, b_: Eq(c_ + b_, c_ + b_)), lambda r: r.ring())
                q.exact(('IH', q.T('c b', lambda c_, b_: c_ + b_), 'hge', 'hlt', 'c', 'b', 'k', 'rr', 'hc0', 'hb', 'hk', 'hc'))
                q.have('h2', q.T('a b k', lambda a_, b_, k_: Eq(b_ * b_ + a_ * a_, k_ * (b_ * a_ + 1))), lambda r: r.linear_combination((1, 'h')))
                q.have('o2', q.T('a b', lambda a_, b_: zle(a_, b_)), lambda r: r.linarith())
                destruct(q, ('imo1988_jump', 'b', 'a', 'k', 'ca', 'o2', 'hk', 'h2'), ['c', 'hc0', 'hca', 'hc'])
                q.have('hlt', q.T('c a s', lambda c_, a_, s_: zlt(c_ + a_, s_)), lambda r: r.linarith())
                q.have('hge', q.T('c a', lambda c_, a_: zle(zzero, c_ + a_)), lambda r: r.linarith())
                q.have('rr', q.T('c a', lambda c_, a_: Eq(c_ + a_, c_ + a_)), lambda r: r.ring())
                q.exact(('IH', q.T('c a', lambda c_, a_: c_ + a_), 'hge', 'hlt', 'c', 'a', 'k', 'rr', 'hc0', 'ha', 'hk', 'hc'))
                q.exists('b'); q.linear_combination((-1, 'h'), (q.T('k b a', lambda k_, b_, a_: k_ * b_ - a_), 'ca'))
                q.exists('a'); q.linear_combination((-1, 'h'), (q.T('k b a', lambda k_, b_, a_: k_ * a_ - b_), 'cb'))
            p.have('H', Hst, H)
            p.exact(lambda q: w['Z.strong_induction'](motive, zzero, q.h('H'), q.h('s0'), q.h('h0')))

        @thm(w, 'imo1988_p6', [Z, Z, lambda a, b: zlt(zzero, a), lambda a, b, h: zlt(zzero, b), lambda a, b, h1, h2: D(a * b + 1, a ** 2 + b ** 2)],
             lambda a, b, h1, h2, h3: Ex(Z, w.lam(Z, lambda m: Eq(a ** 2 + b ** 2, m * m * (a * b + 1)))))
        def _(p):
            p.have('imo1988_jump', w.type_of(w['imo1988_jump']), jump_script)
            p.have('imo1988_descent', w.type_of(w['imo1988_descent']), descent_script)
            p.intro('a', 'b', 'ha', 'hb', 'hd')
            p.cases('hd', names=['k', 'hk'])
            p.have('ha0', p.T('a', lambda a_: zle(zzero, a_)), lambda q: q.linarith())
            p.have('hb0', p.T('b', lambda b_: zle(zzero, b_)), lambda q: q.linarith())
            p.have('hab0', ('Z.mul_nonneg', 'a', 'b', 'ha0', 'hb0'))
            def kpos(q):
                q.obtain(['c', 'c'], ('Z.le_or_lt', 'k', zzero))
                q.exfalso()
                q.have('n1', q.T('k', lambda k_: zle(zzero, -k_)), lambda r: r.linarith())
                q.have('n2', q.T('a b', lambda a_, b_: zle(zzero, a_ * b_ + 1)), lambda r: r.linarith())
                q.have('t', ('Z.mul_nonneg', q.T('a b', lambda a_, b_: a_ * b_ + 1), q.T('k', lambda k_: -k_), 'n2', 'n1'))
                q.have('u', ('Z.mul_pos', 'a', 'a', 'ha', 'ha'))
                q.have('v', ('Z.sq_nonneg', 'b'))
                q.linarith()
                q.exact('c')
            p.have('kpos', p.T('k', lambda k_: zlt(zzero, k_)), kpos)
            p.have('hs0', p.T('a b', lambda a_, b_: zle(zzero, a_ + b_)), lambda q: q.linarith())
            p.have('rr', p.T('a b', lambda a_, b_: Eq(a_ + b_, a_ + b_)), lambda q: q.ring())
            p.have('hk2', p.T('a b k', lambda a_, b_, k_: Eq(a_ * a_ + b_ * b_, k_ * (a_ * b_ + 1))), lambda q: q.linear_combination((1, 'hk')))
            p.obtain(['m', 'hm'], ('imo1988_descent', p.T('a b', lambda a_, b_: a_ + b_), 'hs0', 'a', 'b', 'k', 'rr', 'ha0', 'hb0', 'kpos', 'hk2'))
            p.exists('m'); p.linear_combination((1, 'hk'), (p.T('a b', lambda a_, b_: a_ * b_ + 1), 'hm'))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ a batch of classics
@BANK.problem('n5_minus_n_div30', text='For every integer n, 30 divides n^5 - n.', source='folklore', topics=('number-theory', 'divisibility', 'modular', 'gcd'), level=3,
              spec=Spec(('Z',), (), lambda n: _dvd(30, n ** 5 - n), atom=(lambda n: 30, lambda n: n ** 5 - n), skip_mutants={'constant 30 -> -30 in (Z.dvd 30': 'a divisor and its negative divide the same integers'}),
              notes='6 divides n^3 - n, so n^5 - n = (n^3 - n)(n^2 + 1); 5 divides n^5 - n (Fermat); 6 and 5 are coprime')
def _(w, b):
    s = _syms(w); Z, Eq, zone, D = s.Z, s.Eq, s.zone, s.D
    zgcd = w['Z.is_gcd']
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'n5_minus_n_div30', [Z], lambda n: D(lit(30), n ** 5 - n))
        def _(p):
            p.intro('n')
            p.obtain(['k', 'hk'], ('cube_minus_n_div6', 'n'))
            p.have('d6', p.T('n', lambda n_: D(lit(6), n_ ** 5 - n_)), lambda q: (q.exists(q.T('n k', lambda n_, k_: k_ * (n_ ** 2 + 1))), q.linear_combination((q.T('n', lambda n_: n_ ** 2 + 1), 'hk'))))
            p.have('d5', ('five_dvd_fifth_power', 'n'))
            p.have('cop', zgcd(lit(6), lit(5), zone), lambda q: (q.apply('Z.is_gcd_one_of_coprime'), q.exists(lit(1)), q.exists(-lit(1)), q.ring()))
            p.exact(('dvd_coprime_product', lit(6), lit(5), p.T('n', lambda n_: n_ ** 5 - n_), 'cop', 'd6', 'd5'))


@BANK.problem('imo1964_p2', text='Suppose a, b, c are the sides of a triangle. Prove that a^2 (b + c - a) + b^2 (c + a - b) + c^2 (a + b - c) <= 3abc.', source='IMO 1964 P2', verified='answer-checked', year=1964, topics=('inequality',), level=3,
              spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, c: a > 0, lambda a, b, c: b > 0, lambda a, b, c: c > 0, lambda a, b, c: a < b + c, lambda a, b, c: b < c + a, lambda a, b, c: c < a + b),
                        lambda a, b, c: a * a * (b + c - a) + b * b * (c + a - b) + c * c * (a + b - c) <= 3 * a * b * c, windows={'z': 8}, points=5000,
                        atom=(lambda a, b, c: a * a * (b + c - a) + b * b * (c + a - b) + c * c * (a + b - c), lambda a, b, c: 3 * a * b * c),
                        hatoms=[(lambda a, b, c: 0, lambda a, b, c: a), (lambda a, b, c: 0, lambda a, b, c: b), (lambda a, b, c: 0, lambda a, b, c: c), (lambda a, b, c: a, lambda a, b, c: b + c), (lambda a, b, c: b, lambda a, b, c: c + a), (lambda a, b, c: c, lambda a, b, c: a + b)]),
              notes='over the integers (a special case of the real inequality); the difference 3abc - (left side) is exactly the expression of Schur\'s inequality (schur_nonneg), which needs only a, b, c >= 0')
def _(w, b):
    s = _syms(w); Z, zle, zlt, zzero = s.Z, s.zle, s.zlt, s.zzero
    with w.nota.using(Z):
        @thm(w, 'imo1964_p2', [Z, Z, Z, lambda a, b, c: zlt(zzero, a), lambda a, b, c, h: zlt(zzero, b), lambda a, b, c, h, h2: zlt(zzero, c), lambda a, b, c, h, h2, h3: zlt(a, b + c), lambda a, b, c, h, h2, h3, h4: zlt(b, c + a),
                       lambda a, b, c, h, h2, h3, h4, h5: zlt(c, a + b)], lambda a, b, c, h1, h2, h3, h4, h5, h6: zle(a ** 2 * (b + c - a) + b ** 2 * (c + a - b) + c ** 2 * (a + b - c), 3 * a * b * c))
        def _(p):
            p.intro('a', 'b', 'c', 'ha', 'hb', 'hc', 'h1', 'h2', 'h3')
            p.have('ha0', p.T('a', lambda a_: zle(zzero, a_)), lambda q: q.linarith())
            p.have('hb0', p.T('b', lambda b_: zle(zzero, b_)), lambda q: q.linarith())
            p.have('hc0', p.T('c', lambda c_: zle(zzero, c_)), lambda q: q.linarith())
            p.have('schur_nonneg', w.type_of(w['schur_nonneg']), schur_parts(w)[1][2])
            p.linarith(('schur_nonneg', 'a', 'b', 'c', 'ha0', 'hb0', 'hc0'))


def _fq_cond(f, W=range(-8, 9)): return all(f(x + y) + f(x - y) == 2 * f(x) + 2 * f(y) for x in W for y in W)
def _fq_sol(f, W=range(-8, 9)): return any(all(f(n) == c * n * n for n in W) for c in W)


@BANK.problem('fe_quadratic', text='Determine all functions f : Z -> Z such that f(x + y) + f(x - y) = 2 f(x) + 2 f(y) for all integers x and y.', source='folklore (Jordan-von Neumann)', topics=('functional-equation', 'induction'), level=4, kind='answer',
              answer='f(n) = c n^2 for an integer c', spec=Spec(('fn',), (), (_fq_cond, _fq_sol), skip_mutants={'swap the arguments of Z.sub in ((Z.add (x0 (Z.add x1 x2)) (x0 (Z.sub x1 x2)))': 'f(x + y) + f(y - x) = 2 f(x) + 2 f(y) at the pair (y, x) is the original equation: the roles of x and y are interchangeable under the quantifiers'}),
              notes='f(0) = 0 and f is even; with c = f(1) the pair f(n) = c n^2, f(n + 1) = c (n + 1)^2 goes up by f(n + 2) = 2 f(n + 1) + 2 f(1) - f(n); negative n by evenness')
def _(w, b):
    s = _syms(w); Z, Eq, F, Ex, zzero, zone, zle = s.Z, s.Eq, s.F, s.Ex, s.zzero, s.zone, s.zle
    fT = w.arrow(Z, Z)
    with w.nota.using(Z):
        def cond(f): return F([Z, Z], lambda x, y: Eq(f(x + y) + f(x - y), 2 * f(x) + 2 * f(y)))
        def sol(f): return Ex(Z, w.lam(Z, lambda c: F([Z], lambda n: Eq(f(n), c * (n * n)))))
        @thm(w, 'fe_quadratic', [fT], lambda f: w.Iff(cond(f), sol(f)))
        def _(p):
            p.intro('f'); p.split()
            p.intro('H')
            p.have('e00', p.T('f', lambda f_: Eq(f_(zzero) + f_(zzero), 2 * f_(zzero) + 2 * f_(zzero))), lambda q: q.exact(('H', zzero, zzero)))
            p.have('f0', p.T('f', lambda f_: Eq(f_(zzero), zzero)), lambda q: q.linarith())
            def ev(q):
                q.intro('y'); q.have('e', ('H', zzero, 'y')); q.rw_at('e', 'Z.zero_add'); q.rw_at('e', 'Z.sub_def'); q.rw_at('e', 'Z.zero_add'); q.linarith()
            p.have('hev', p.T('f', lambda f_: F([Z], lambda y: Eq(f_(-y), f_(y)))), ev)
            def up(q):
                q.intro('n', 'hn'); q.induction_on('Z.induction_from', 'n', extra=(zzero,))
                q.split(); q.linarith(); q.linarith()
                q.intro('m', 'hm', 'ih'); q.cases('ih', names=['i1', 'i2'])
                q.have('e1', ('H', q.T('m', lambda m_: m_ + zone), zone))
                q.have('e2', q.T('m', lambda m_: Eq(m_ + zone - zone, m_)), lambda r: r.ring())
                q.rw_at('e1', 'e2')
                q.split(); q.linarith(); q.linarith()
                q.assumption()
            p.have('hup', p.T('f', lambda f_: F([Z, lambda n: zle(zzero, n)], lambda n, hn: Eq(f_(n), f_(zone) * (n * n)) & Eq(f_(n + zone), f_(zone) * ((n + zone) * (n + zone))))), up)
            p.exists(p.T('f', lambda f_: f_(zone)))
            p.intro('n')
            p.obtain(['c', 'c'], ('Z.le_or_lt', zzero, 'n'))
            p.obtain(['u1', 'u2'], ('hup', 'n', 'c')); p.exact('u1')
            p.have('c2', p.T('n', lambda n_: zle(zzero, -n_)), lambda q: q.linarith())
            p.obtain(['u1', 'u2'], ('hup', p.T('n', lambda n_: -n_), 'c2'))
            p.have('e', ('hev', 'n')); p.linear_combination((-1, 'e'), (1, 'u1'))
            p.intro('h'); p.cases('h', names=['c', 'hc'])
            p.intro('x', 'y')
            p.have('e1', ('hc', p.T('x y', lambda x_, y_: x_ + y_))); p.have('e2', ('hc', p.T('x y', lambda x_, y_: x_ - y_))); p.have('e3', ('hc', 'x')); p.have('e4', ('hc', 'y'))
            p.linear_combination((1, 'e1'), (1, 'e2'), (-2, 'e3'), (-2, 'e4'))


@BANK.problem('n4_n2_1_composite', text='Prove that n^4 + n^2 + 1 is not a prime for any integer n > 1.', source='folklore', topics=('number-theory', 'prime', 'identity'), level=3,
              spec=Spec(('Z',), (lambda n: n > 1,), lambda n: not _prime(n ** 4 + n * n + 1), windows={'z': 8}, atom=(lambda n: n ** 4 + n * n + 1,), hatoms=[(lambda n: 1, lambda n: n)]),
              notes='n^4 + n^2 + 1 = (n^2 + n + 1)(n^2 - n + 1) with both factors larger than 1')
def _(w, b):
    s = _syms(w); Z, Eq, zone, zlt = s.Z, s.Eq, s.zone, s.zlt
    prime = w['Z.prime']
    with w.nota.using(Z):
        @thm(w, 'n4_n2_1_composite', [Z, lambda n: zlt(zone, n)], lambda n, h: ~prime(n ** 4 + n ** 2 + 1))
        def _(p):
            p.intro('n', 'hn')
            p.have('e', p.T('n', lambda n_: Eq(n_ ** 4 + n_ ** 2 + 1, (n_ ** 2 + n_ + 1) * (n_ ** 2 - n_ + 1))), lambda q: q.ring())
            p.rw('e')
            p.apply('not_prime_mul')
            p.nlinarith()
            p.nlinarith(('Z.mul_pos', 'n', p.T('n', lambda n_: n_ - 1)))


@BANK.problem('ineq_a_plus_b_cubed', text='For all non-negative integers a and b: (a + b)^3 <= 4 (a^3 + b^3).', source='folklore (power mean)', topics=('inequality',), level=2,
              spec=Spec(('Z', 'Z'), (lambda a, b: a >= 0, lambda a, b: b >= 0), lambda a, b: (a + b) ** 3 <= 4 * (a ** 3 + b ** 3), windows={'z': 8},
                        atom=(lambda a, b: (a + b) ** 3, lambda a, b: 4 * (a ** 3 + b ** 3)), hatoms=[(lambda a, b: 0, lambda a, b: a), (lambda a, b: 0, lambda a, b: b)]),
              notes='4 (a^3 + b^3) - (a + b)^3 = 3 (a + b)(a - b)^2')
def _(w, b):
    s = _syms(w); Z, zle, zzero = s.Z, s.zle, s.zzero
    with w.nota.using(Z):
        @thm(w, 'ineq_a_plus_b_cubed', [Z, Z, lambda a, c: zle(zzero, a), lambda a, c, h: zle(zzero, c)], lambda a, c, h1, h2: zle((a + c) ** 3, 4 * (a ** 3 + c ** 3)))
        def _(p):
            p.intro('a', 'c', 'ha', 'hc')
            p.have('s', p.T('a c', lambda a_, c_: zle(zzero, a_ + c_)), lambda q: q.linarith())
            p.have('t', ('Z.mul_nonneg', p.T('a c', lambda a_, c_: a_ + c_), p.T('a c', lambda a_, c_: (a_ - c_) * (a_ - c_)), 's', ('Z.sq_nonneg', p.T('a c', lambda a_, c_: a_ - c_))))
            p.linarith()


@BANK.problem('ineq_square_products', text='For all integers a, b, c: a^2 b^2 + b^2 c^2 + c^2 a^2 >= abc (a + b + c).', source='folklore', topics=('inequality',), level=2,
              spec=Spec(('Z', 'Z', 'Z'), (), lambda a, b, c: a * a * b * b + b * b * c * c + c * c * a * a >= a * b * c * (a + b + c), windows={'z': 8},
                        atom=(lambda a, b, c: a * b * c * (a + b + c), lambda a, b, c: a * a * b * b + b * b * c * c + c * c * a * a)),
              notes='x^2 + y^2 + z^2 >= xy + yz + zx for x = ab, y = bc, z = ca: the three squares (ab - bc)^2, (bc - ca)^2, (ca - ab)^2')
def _(w, b):
    s = _syms(w); Z, zle = s.Z, s.zle
    with w.nota.using(Z):
        @thm(w, 'ineq_square_products', [Z, Z, Z], lambda a, c, d: zle(a * c * d * (a + c + d), a ** 2 * c ** 2 + c ** 2 * d ** 2 + d ** 2 * a ** 2))
        def _(p):
            p.intro('a', 'c', 'd')
            p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') * q.h('c') - q.h('c') * q.h('d')), ('Z.sq_nonneg', lambda q: q.h('c') * q.h('d') - q.h('d') * q.h('a')), ('Z.sq_nonneg', lambda q: q.h('d') * q.h('a') - q.h('a') * q.h('c')))


NAMES = tuple(list(BANK.problems)[_N0:])          # the problems this file registers, in order (S43 checks them)
