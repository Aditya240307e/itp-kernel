"""bank_lib2.py -- more problems of the bank (see bank.py and bank_lib.py, whose conventions these follow): a statement, its text, its source, its intended meaning as Python (`Spec`) and a reference proof.

The problems are registered after those of bank_lib.py, so the world below them (and every digest in s41_golden.json) is the one of the first 28.  Levels:

    1  a single arithmetic step (an identity by `ring`, an inequality by `nlinarith` with the square that makes it true)
    2  a short proof with one idea: a case split on a remainder, a witness, a hint that is a product
    3  an olympiad lemma: several ideas, a descent, an induction in steps
    4  an olympiad problem with a short solution
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nt_lib import _syms
from lnl_base import thm
from semantic import Spec
from bank import BANK
from bank_lib import destruct, AND, _dvd, _prime


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
@BANK.problem('square_mod_three', text='The square of every integer is of the form 3k or 3k + 1.', source='folklore', topics=('number-theory', 'modular'), level=2,
              spec=Spec(('Z',), (), lambda n: any(n * n == 3 * k or n * n == 3 * k + 1 for k in range(0, 40)), windows={'z': 6, 'ex': 40},
                        body=(('Z',), lambda n, k: n * n == 3 * k or n * n == 3 * k + 1),
                        atoms=[(lambda n, k: n * n, lambda n, k: 3 * k), (lambda n, k: n * n, lambda n, k: 3 * k + 1)]))
def _(w, b):
    s = _syms(w); Z, Eq, Ex, zzero, zlt = s.Z, s.Eq, s.Ex, s.zzero, s.zlt
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'square_mod_three', [Z], lambda n: Ex(Z, w.lam(Z, lambda k: Eq(n ** 2, 3 * k) | Eq(n ** 2, 3 * k + 1))))
        def _(p):
            p.intro('n')
            p.have('h3', p.T('', lambda: zlt(zzero, lit(3))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 2)
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 2)); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 2 + 2 * q)); p.right(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 2 + 4 * q + 1)); p.right(); p.ring()


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


@BANK.problem('five_dvd_fifth_power', text='For every integer n, 5 divides n^5 - n.', source='folklore (Fermat)', topics=('number-theory', 'divisibility', 'modular'), level=2,
              spec=Spec(('Z',), (), lambda n: _dvd(5, n ** 5 - n), atom=(lambda n: 5, lambda n: n ** 5 - n), skip_mutants={'constant 5 -> -5 in (Z.dvd 5': 'a divisor and its negative divide the same integers'}))
def _(w, b):
    s = _syms(w); Z, zzero, zlt, D = s.Z, s.zzero, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'five_dvd_fifth_power', [Z], lambda n: D(lit(5), n ** 5 - n))
        def _(p):
            p.intro('n')
            p.have('h5', p.T('', lambda: zlt(zzero, lit(5))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(5), 'h5', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 4)
            for k in range(5):                      # (5q + k)^5 - (5q + k) = 5 (625 q^5 + 625 k q^4 + 250 k^2 q^3 + 50 k^3 q^2 + (5 k^4 - 1) q + (k^5 - k) / 5)
                p.rw('r_eq')
                p.exists(p.T('q', lambda q, k=k: lit(625) * q ** 5 + lit(625 * k) * q ** 4 + lit(250 * k * k) * q ** 3 + lit(50 * k ** 3) * q ** 2 + lit(5 * k ** 4 - 1) * q + lit((k ** 5 - k) // 5)))
                p.ring()


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


@BANK.problem('dvd_coprime_product', text='If a divides c, b divides c and gcd(a, b) = 1, then ab divides c.', source='folklore', topics=('number-theory', 'divisibility', 'gcd'), level=3,
              spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, c: math.gcd(a, b) == 1, lambda a, b, c: _dvd(a, c), lambda a, b, c: _dvd(b, c)), lambda a, b, c: _dvd(a * b, c), windows={'z': 6}, atom=(lambda a, b, c: a * b, lambda a, b, c: c)))
def _(w, b):
    s = _syms(w); Z, Eq, zone, D = s.Z, s.Eq, s.zone, s.D
    zgcd = w['Z.is_gcd']
    with w.nota.using(Z):
        @thm(w, 'dvd_coprime_product', [Z, Z, Z, lambda a, c, d: zgcd(a, c, zone), lambda a, c, d, h: D(a, d), lambda a, c, d, h, h1: D(c, d)], lambda a, c, d, h, h1, h2: D(a * c, d))
        def _(p):                                                              # c = a m and b | a m with gcd(b, a) = 1 give b | m
            p.intro('a', 'c', 'd', 'h', 'ha', 'hb')
            p.cases('ha', names=['m', 'hm'])
            p.have('hc', p.T('a c', lambda a_, c_: w['Z.coprime'](c_, a_)), lambda q: q.exact(('Z.coprime_symm', 'a', 'c', ('Z.coprime_of_is_gcd_one', 'a', 'c', 'h'))))
            p.have('hbm', p.T('a c m', lambda a_, c_, m_: D(c_, a_ * m_)), lambda q: (q.rw_at('hb', 'hm'), q.exact('hb')))
            p.obtain(['n', 'hn'], ('Z.coprime_dvd_mul', 'c', 'a', 'm', 'hc', 'hbm'))
            p.exists('n'); p.rw('hm'); p.rw('hn'); p.ring()


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
