"""bank_amc2.py -- problems in the manner of the American contests (AMC 8 / 10 / 12), written for the bank.  Integer vocabulary, a kernel-checked reference proof for each.

Provenance.  These problems are `origin='style'`: written for this bank in the manner of a contest (no contest's text, no contest's number); `verified='none'` (there is no published answer to check against;
the faithfulness of the formal statement to the words is checked by semantic.py like every other problem's).  They are the easy end of the curriculum -- levels 1 and 2 are a few lines, level 3 a lemma.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, math
from semantic import Spec
from bank import BANK
from bank_spec import spec, EQ, LT, LE, DVD, COP, GCD, PRIME, V as X, dvd_
from bank_amc import AND, OR, sq
from ext_lib import destruct
import bank_kit as K

NAMES = []


def problem(name, **kw):
    NAMES.append(name)
    kw.setdefault('origin', 'style'); kw.setdefault('verified', 'none'); kw.setdefault('source', 'original, in the manner of the AMC')
    return BANK.problem(name, **kw)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ algebra and word problems
@problem('amc_alg_01', text='If 3x + 7 = 2x + 19, what is x?', topics=('algebra',), level=1, kind='answer', answer='x = 12',
         spec=spec(('Z',), [EQ(lambda x: 3 * x + 7, lambda x: 2 * x + 19)], EQ(X(0), 12), domains=(range(-4, 20),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_01', [v.Z], [lambda x: Eq(3 * x + 7, 2 * x + 19)], lambda x: Eq(x, lit(12)))
        def _(p):
            p.intro('x', 'h'); p.linarith()


@problem('amc_alg_02', text='The sum of two numbers is 50 and their difference is 14. What is the larger number?', topics=('algebra',), level=1, kind='answer', answer='the larger number is 32',
         spec=spec(('Z', 'Z'), [EQ(lambda a, b: a + b, 50), EQ(lambda a, b: a - b, 14)], EQ(X(0), 32), domains=(range(28, 38), range(14, 24))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_02', [v.Z, v.Z], [lambda a, b_: Eq(a + b_, lit(50)), lambda a, b_: Eq(a - b_, lit(14))], lambda a, b_: Eq(a, lit(32)))
        def _(p):
            p.intro('a', 'b', 'h1', 'h2'); p.linarith()


@problem('amc_alg_03', text='Five students have a mean score of 78 on a test. Four of the scores are 70, 75, 80 and 85. What is the fifth score?', topics=('algebra', 'arithmetic'), level=1, kind='answer',
         answer='the fifth score is 80', spec=spec(('Z',), [EQ(lambda x: 70 + 75 + 80 + 85 + x, 5 * 78)], EQ(X(0), 80), domains=(range(70, 90),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_03', [v.Z], [lambda x: Eq(lit(70) + lit(75) + lit(80) + lit(85) + x, lit(5) * lit(78))], lambda x: Eq(x, lit(80)))
        def _(p):
            p.intro('x', 'h'); p.linarith()


@problem('amc_alg_04', text='A rectangle has perimeter 30 and its length is 3 more than twice its width. What is its area?', topics=('algebra',), level=2, kind='answer', answer='the area is 44',
         spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), EQ(lambda l, w: 2 * (l + w), 30), EQ(X(0), lambda l, w: 2 * w + 3)], EQ(lambda l, w: l * w, 44), domains=(range(-2, 14), range(-2, 14))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_04', [v.Z, v.Z], [lambda l, w_: v.pos(l), lambda l, w_: v.pos(w_), lambda l, w_: Eq(2 * (l + w_), lit(30)), lambda l, w_: Eq(l, 2 * w_ + 3)], lambda l, w_: Eq(l * w_, lit(44)))
        def _(p):
            p.intro('l', 'w', 'hl', 'hw', 'h1', 'h2')
            p.have('ew', p.T('w', lambda w_: Eq(w_, lit(4))), lambda q: q.linarith())
            p.have('el', p.T('l', lambda l: Eq(l, lit(11))), lambda q: q.linarith())
            p.linear_combination((p.h('w'), 'el'), (11, 'ew'))


@problem('amc_alg_05', text='The sum of three consecutive integers is 72. What is the product of the smallest and the largest?', topics=('algebra', 'arithmetic'), level=2, kind='answer', answer='575',
         spec=spec(('Z',), [EQ(lambda n: n + (n + 1) + (n + 2), 72)], EQ(lambda n: n * (n + 2), 575), domains=(range(20, 28),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_05', [v.Z], [lambda n: Eq(n + (n + 1) + (n + 2), lit(72))], lambda n: Eq(n * (n + 2), lit(575)))
        def _(p):
            p.intro('n', 'h')
            p.have('en', p.T('n', lambda n: Eq(n, lit(23))), lambda q: q.linarith())
            p.linear_combination((p.T('n', lambda n: n + 25), 'en'))


@problem('amc_alg_06', text='The sum of the squares of two consecutive positive integers is 481. What is their sum?', topics=('algebra', 'diophantine'), level=2, kind='answer', answer='31',
         spec=spec(('Z',), [LT(0, X(0)), EQ(lambda n: n * n + (n + 1) * (n + 1), 481)], EQ(lambda n: n + (n + 1), 31), domains=(range(-18, 20),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_06', [v.Z], [lambda n: v.pos(n), lambda n: Eq(n ** 2 + (n + 1) ** 2, lit(481))], lambda n: Eq(n + (n + 1), lit(31)))
        def _(p):
            p.intro('n', 'hn', 'h')
            p.have('e', p.T('n', lambda n: Eq((n - 15) * (n + 16), v.zero)), lambda q: q.linarith())
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('n', lambda n: n - 15), p.T('n', lambda n: n + 16), 'e'))
            p.linarith()
            p.exfalso(); p.linarith()


@problem('amc_alg_07', text='The numbers x and y are in the ratio 3 : 5 (that is, 5x = 3y), and x + y = 56. What is x?', topics=('algebra',), level=1, kind='answer', answer='x = 21',
         spec=spec(('Z', 'Z'), [EQ(lambda x, y: 5 * x, lambda x, y: 3 * y), EQ(lambda x, y: x + y, 56)], EQ(X(0), 21), domains=(range(17, 26), range(30, 40))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_07', [v.Z, v.Z], [lambda x, y: Eq(5 * x, 3 * y), lambda x, y: Eq(x + y, lit(56))], lambda x, y: Eq(x, lit(21)))
        def _(p):
            p.intro('x', 'y', 'h1', 'h2'); p.linarith()


@problem('amc_alg_08', text='The digits of a two-digit number add up to 11, and interchanging them makes the number 27 smaller. What is the number?', topics=('algebra', 'digits'), level=2, kind='answer', answer='74',
         spec=spec(('Z', 'Z'), [LE(1, X(0)), LE(X(0), 9), LE(0, X(1)), LE(X(1), 9), EQ(lambda a, b: a + b, 11), EQ(lambda a, b: (10 * a + b) - (10 * b + a), 27)], EQ(lambda a, b: 10 * a + b, 74),
                   domains=(range(0, 12), range(-1, 12))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_08', [v.Z, v.Z], [lambda a, b_: v.le(lit(1), a), lambda a, b_: v.le(a, lit(9)), lambda a, b_: v.nonneg(b_), lambda a, b_: v.le(b_, lit(9)), lambda a, b_: Eq(a + b_, lit(11)),
                                                  lambda a, b_: Eq((10 * a + b_) - (10 * b_ + a), lit(27))], lambda a, b_: Eq(10 * a + b_, lit(74)))
        def _(p):
            p.intro('a', 'b', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'); p.linarith()


@problem('amc_alg_09', text='Ann is three times as old as Bob. In six years she will be twice as old as he will be. How old is Ann now?', topics=('algebra',), level=1, kind='answer', answer='18',
         spec=spec(('Z', 'Z'), [LT(0, X(1)), EQ(X(0), lambda a, b: 3 * b), EQ(lambda a, b: a + 6, lambda a, b: 2 * (b + 6))], EQ(X(0), 18), domains=(range(14, 24), range(2, 10))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_09', [v.Z, v.Z], [lambda a, b_: v.pos(b_), lambda a, b_: Eq(a, 3 * b_), lambda a, b_: Eq(a + 6, 2 * (b_ + 6))], lambda a, b_: Eq(a, lit(18)))
        def _(p):
            p.intro('a', 'b', 'hb', 'h1', 'h2'); p.linarith()


@problem('amc_alg_10', text='Two positive numbers x > y have sum 7 and product 12. What is x^2 - y^2?', topics=('algebra',), level=2, kind='answer', answer='7',
         spec=spec(('Z', 'Z'), [EQ(lambda x, y: x + y, 7), EQ(lambda x, y: x * y, 12), LT(X(1), X(0))], EQ(lambda x, y: x * x - y * y, 7), domains=(range(-9, 11), range(-9, 11))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_10', [v.Z, v.Z], [lambda x, y: Eq(x + y, lit(7)), lambda x, y: Eq(x * y, lit(12)), lambda x, y: v.lt(y, x)], lambda x, y: Eq(x ** 2 - y ** 2, lit(7)))
        def _(p):
            p.intro('x', 'y', 'h1', 'h2', 'h3')
            p.have('e', p.T('x y', lambda x, y: Eq((x - y - 1) * (x - y + 1), v.zero)), lambda q: q.linear_combination((q.T('x y', lambda x, y: x + y + 7), 'h1'), (-4, 'h2')))
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('x y', lambda x, y: x - y - 1), p.T('x y', lambda x, y: x - y + 1), 'e'))
            p.linear_combination((p.T('x y', lambda x, y: x + y), 'c'), (1, 'h1'))
            p.exfalso(); p.linarith()


@problem('amc_alg_11', text='Show that there are integers x and y with 7x + 11y = 1.', topics=('algebra', 'diophantine'), level=1, kind='exists',
         spec=spec((), (), lambda: any(7 * x + 11 * y == 1 for x in range(-6, 7) for y in range(-6, 7)), windows={'ex': 7},
                   body=(('Z', 'Z'), lambda x, y: 7 * x + 11 * y == 1), atoms=[(lambda x, y: 7 * x + 11 * y, lambda x, y: 1)]))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_11', [], [], lambda: v.exists(lambda x: v.exists(lambda y: Eq(7 * x + 11 * y, lit(1)))))
        def _(p):
            p.exists(-lit(3)); p.exists(lit(2)); p.ring()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ divisibility, remainders, parity
@problem('amc_div_01', text='Show that n^3 + 5n is divisible by 6 for every integer n.', topics=('number-theory', 'divisibility'), level=2,
         spec=spec(('Z',), [], DVD(6, lambda n: n ** 3 + 5 * n), domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_div_01', [v.Z], [], lambda n: D(lit(6), n ** 3 + 5 * n))
        def _(p):
            p.intro('n')
            p.have('e', p.T('n', lambda n: Eq(n ** 3 + 5 * n, (n ** 3 - n) + lit(6) * n)), lambda q: q.ring())
            p.rw('e')
            p.apply('Z.dvd_add'); p.exact(('cube_minus_n_div6', 'n')); p.apply('Z.dvd_mul_right')


@problem('amc_div_02', text='Show that the product of four consecutive integers is divisible by 24.', topics=('number-theory', 'divisibility', 'case-analysis'), level=3,
         spec=spec(('Z',), [], DVD(24, lambda n: n * (n + 1) * (n + 2) * (n + 3)), domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_div_02', [v.Z], [], lambda n: D(lit(24), n * (n + 1) * (n + 2) * (n + 3)))
        def _(p):
            p.intro('n')
            p.obtain(['k', 'hk'], ('cube_minus_n_div6', p.T('n', lambda n: n + 1)))                # (n + 1)^3 - (n + 1) = 6 k  is  n (n + 1)(n + 2)
            p.have('h4', p.T('', lambda: v.pos(lit(4))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(4), 'h4', 'n'), ['s', 'r', 'e', 'h0', 'h1'])
            p.interval_cases('r', 0, 3)
            # n = 4 s + j: the product is 8 a_j(s) and 6 k (n + 3); 24 (k (n + 3) - a_j) is the product
            a = [lambda s: s * (4 * s + 1) * (2 * s + 1) * (4 * s + 3), lambda s: (4 * s + 1) * (2 * s + 1) * (4 * s + 3) * (s + 1),
                 lambda s: (2 * s + 1) * (4 * s + 3) * (s + 1) * (4 * s + 5), lambda s: (4 * s + 3) * (s + 1) * (4 * s + 5) * (2 * s + 3)]
            for j in range(4):
                p.rw_at('e', 'r_eq'); p.rw('e'); p.rw_at('hk', 'e')
                p.exists(p.T('s k', lambda s, k, j=j: k * (4 * s + j + 3) - a[j](s)))
                p.linear_combination((p.T('s', lambda s, j=j: 4 * (4 * s + j + 3)), 'hk'))


@problem('amc_div_03', text='Show that n(n + 1)(2n + 1) is divisible by 6 for every integer n.', topics=('number-theory', 'divisibility', 'case-analysis'), level=3,
         spec=spec(('Z',), [], DVD(6, lambda n: n * (n + 1) * (2 * n + 1)), domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_div_03', [v.Z], [], lambda n: D(lit(6), n * (n + 1) * (2 * n + 1)))
        def _(p):
            p.intro('n')
            p.have('h6', p.T('', lambda: v.pos(lit(6))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(6), 'h6', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.interval_cases('r', 0, 5)
            for r in range(6):                                                   # n = 6 q + r:  n (n + 1)(2n + 1) = 6 W
                p.rw_at('e', 'r_eq'); p.rw('e')
                p.exists(p.T('q', lambda q, r=r: 72 * q ** 3 + 36 * r * q ** 2 + 6 * r * r * q + 18 * q ** 2 + 6 * r * q + q + r * (r + 1) * (2 * r + 1) // 6))
                p.ring()


@problem('amc_div_04', text='Show that n and n^5 have the same last digit: 10 divides n^5 - n for every integer n.', topics=('number-theory', 'divisibility', 'parity'), level=3,
         spec=spec(('Z',), [], DVD(10, lambda n: n ** 5 - n), domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_div_04', [v.Z], [], lambda n: D(lit(10), n ** 5 - n))
        def _(p):
            p.intro('n')
            p.obtain(['b', 'hb'], ('five_dvd_fifth_power', 'n'))                                  # n^5 - n = 5 b
            p.obtain(['m', 'c'], ('Z.even_or_odd', 'n'))                                          # n = 2 m  or  n = 2 m + 1: then n^5 - n = 2 a, 5 b = 2 a, b = 2 (3 b - a)
            p.cases('c', names=['c', 'c'])
            p.rw_at('hb', 'c'); p.rw('c')
            p.exists(p.T('b m', lambda b_, m: 3 * b_ - (16 * m ** 5 - m)))
            p.linear_combination((6, 'hb'))
            p.rw_at('hb', 'c'); p.rw('c')
            p.exists(p.T('b m', lambda b_, m: 3 * b_ - (16 * m ** 5 + 40 * m ** 4 + 40 * m ** 3 + 20 * m ** 2 + 4 * m)))
            p.linear_combination((6, 'hb'))


@problem('amc_div_05', text='A three-digit number 100a + 10b + c (digits a, b, c) is divisible by 9 exactly when the sum of its digits is.', topics=('number-theory', 'divisibility', 'digits'), level=2,
         spec=spec(('Z', 'Z', 'Z'), [LE(1, X(0)), LE(X(0), 9), LE(0, X(1)), LE(X(1), 9), LE(0, X(2)), LE(X(2), 9)],
                   (lambda a, b, c: (100 * a + 10 * b + c) % 9 == 0, lambda a, b, c: (a + b + c) % 9 == 0),
                   atoms=[(lambda a, b, c: 9, lambda a, b, c: 100 * a + 10 * b + c), (lambda a, b, c: 9, lambda a, b, c: a + b + c)], domains=(range(0, 11), range(-1, 11), range(-1, 11))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_div_05', [v.Z, v.Z, v.Z], [lambda a, b_, c: v.le(lit(1), a), lambda a, b_, c: v.le(a, lit(9)), lambda a, b_, c: v.nonneg(b_), lambda a, b_, c: v.le(b_, lit(9)),
                                                      lambda a, b_, c: v.nonneg(c), lambda a, b_, c: v.le(c, lit(9))],
                   lambda a, b_, c: v.Iff(D(lit(9), 100 * a + 10 * b_ + c), D(lit(9), a + b_ + c)))
        def _(p):
            p.intro('a', 'b', 'c', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6')
            p.split()
            p.intro('h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('a b k', lambda a, b_, k: k - 11 * a - b_)); p.linear_combination((1, 'hk'))
            p.intro('h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('a b k', lambda a, b_, k: k + 11 * a + b_)); p.linear_combination((1, 'hk'))


@problem('amc_div_06', text='Show that the sum of any five consecutive integers is divisible by 5.', topics=('number-theory', 'divisibility'), level=1,
         spec=spec(('Z',), [], DVD(5, lambda n: n + (n + 1) + (n + 2) + (n + 3) + (n + 4)), domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_div_06', [v.Z], [], lambda n: D(lit(5), n + (n + 1) + (n + 2) + (n + 3) + (n + 4)))
        def _(p):
            p.intro('n'); p.exists(p.T('n', lambda n: n + 2)); p.ring()


@problem('amc_mod_01', text='Show that n^2 + n + 2 is never divisible by 5.', topics=('number-theory', 'modular', 'case-analysis'), level=3,
         spec=spec(('Z',), [], lambda n: (n * n + n + 2) % 5 != 0, atoms=[(lambda n: 5, lambda n: n * n + n + 2)], domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_mod_01', [v.Z], [], lambda n: v.Not(D(lit(5), n ** 2 + n + 2)))
        def _(p):
            p.intro('n', 'h'); p.cases('h', names=['k', 'hk'])
            p.have('h5', p.T('', lambda: v.pos(lit(5))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(5), 'h5', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.interval_cases('r', 0, 4)
            for r in range(5):                                                   # n = 5 q + r:  5 (k - 5 q^2 - 2 r q - q) = r^2 + r + 2
                p.rw_at('e', 'r_eq'); p.rw_at('hk', 'e')
                K.no_multiple(p, p.T('k q', lambda k, q, r=r: k - 5 * q * q - 2 * r * q - q), 5, r * r + r + 2)


@problem('amc_mod_02', text='What are the last two digits of 7^100?  (Show that 7^100 leaves remainder 1 upon division by 100.)', topics=('number-theory', 'modular', 'arithmetic'), level=3,
         spec=spec((), [], DVD(100, 7 ** 100 - 1), skip_mutants={'constant 7 -> -7 in': 'an even power does not see the sign of its base: (-7)^100 = 7^100'}))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_mod_02', [], [], lambda: D(lit(100), lit(7) ** 100 - 1))
        def _(p):
            p.have('h1', ('Z.sub_dvd_pow_sub', lit(2401), lit(1), v.nlit(25)))                       # 2401 - 1 divides 2401^25 - 1^25
            p.have('e1', p.T('', lambda: Eq(lit(2401) ** 25 - lit(1) ** 25, lit(7) ** 100 - 1)), lambda q: q.ring())
            p.have('e2', p.T('', lambda: Eq(lit(2401) - 1, lit(2400))), lambda q: q.ring())
            p.rw_at('h1', 'e1'); p.rw_at('h1', 'e2')
            p.have('h2', p.T('', lambda: D(lit(100), lit(2400))), lambda q: (q.exists(lit(24)), q.ring()))
            p.apply('Z.dvd_trans', lit(100), lit(2400)); p.exact('h2'); p.exact('h1')


@problem('amc_mod_03', text='What is the remainder when 2^2024 is divided by 7?  (Show that 7 divides 2^2024 - 4.)', topics=('number-theory', 'modular', 'arithmetic'), level=3,
         spec=spec((), [], DVD(7, 2 ** 2024 - 4), skip_mutants={'constant 2 -> -2 in': 'an even power does not see the sign of its base: (-2)^2024 = 2^2024'}))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_mod_03', [], [], lambda: D(lit(7), lit(2) ** 2024 - 4))
        def _(p):
            p.obtain(['m', 'hm'], ('Z.sub_dvd_pow_sub', lit(8), lit(1), v.nlit(674)))                # 8^674 - 1^674 = (8 - 1) m
            p.exists(p.T('m', lambda m: 4 * m))
            p.linear_combination((4, 'hm'))


@problem('amc_mod_04', text='The integer a leaves remainder 3 and b leaves remainder 5 upon division by 7. Show that ab leaves remainder 1.', topics=('number-theory', 'modular'), level=2,
         spec=spec(('Z', 'Z'), [DVD(7, lambda a, b: a - 3), DVD(7, lambda a, b: b - 5)], DVD(7, lambda a, b: a * b - 1)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_mod_04', [v.Z, v.Z], [lambda a, b_: D(lit(7), a - 3), lambda a, b_: D(lit(7), b_ - 5)], lambda a, b_: D(lit(7), a * b_ - 1))
        def _(p):
            p.intro('a', 'b', 'ha', 'hb'); p.cases('ha', names=['s', 'hs']); p.cases('hb', names=['t', 'ht'])
            p.exists(p.T('s t', lambda s, t: 2 + 3 * t + 5 * s + 7 * s * t))
            p.linear_combination((p.h('b'), 'hs'), (p.T('s', lambda s: 7 * s + 3), 'ht'))


@problem('amc_par_01', text='If a^2 + b^2 is even, show that a + b is even.', topics=('number-theory', 'parity', 'case-analysis'), level=2,
         spec=spec(('Z', 'Z'), [DVD(2, lambda a, b: a * a + b * b)], DVD(2, lambda a, b: a + b)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_par_01', [v.Z, v.Z], [lambda a, b_: D(lit(2), a ** 2 + b_ ** 2)], lambda a, b_: D(lit(2), a + b_))
        def _(p):
            p.intro('a', 'b', 'h'); p.cases('h', names=['k', 'hk'])
            p.obtain(['s', 'ca'], ('Z.even_or_odd', 'a')); p.obtain(['t', 'cb'], ('Z.even_or_odd', 'b'))
            p.cases('ca', names=['ca', 'ca'])
            p.cases('cb', names=['cb', 'cb'])
            p.exists(p.T('s t', lambda s, t: s + t)); p.linarith()                                        # even, even
            p.rw_at('hk', 'ca'); p.rw_at('hk', 'cb')                                                      # even, odd: 2 (k - 2 s^2 - 2 t^2 - 2 t) = 1
            K.no_multiple(p, p.T('k s t', lambda k, s, t: k - 2 * s * s - 2 * t * t - 2 * t), 2, 1)
            p.cases('cb', names=['cb', 'cb'])
            p.rw_at('hk', 'ca'); p.rw_at('hk', 'cb')                                                      # odd, even
            K.no_multiple(p, p.T('k s t', lambda k, s, t: k - 2 * s * s - 2 * s - 2 * t * t), 2, 1)
            p.exists(p.T('s t', lambda s, t: s + t + 1)); p.linarith()                                    # odd, odd


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ primes and greatest common divisors
@problem('amc_prime_01', text='Show that if p is a prime greater than 3, then 24 divides p^2 - 1.', topics=('number-theory', 'prime', 'divisibility', 'case-analysis'), level=3,
         spec=spec(('Z',), [PRIME(X(0)), LT(3, X(0))], DVD(24, lambda p: p * p - 1), domains=(range(-2, 60),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd; P = w['Z.prime']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_prime_01', [v.Z], [lambda p: P(p), lambda p: v.lt(lit(3), p)], lambda p: D(lit(24), p * p - 1))
        def _(p):
            p.intro('p', 'hp', 'h3'); p.cases('hp', names=['h1', 'hd'])
            p.have('pos2', p.T('', lambda: v.pos(lit(2))), lambda q: q.linarith())
            p.have('pos3', p.T('', lambda: v.pos(lit(3))), lambda q: q.linarith())
            p.have('h24', p.T('', lambda: v.pos(lit(24))), lambda q: q.linarith())
            for d in (2, 3):                                                       # a prime greater than 3 has neither 2 nor 3 as a divisor
                p.have('n%d' % d, p.T('p', lambda p_, d=d: v.Not(D(lit(d), p_))),
                       lambda q, d=d: (q.intro('h'), q.obtain(['c', 'c'], ('hd', lit(d), 'pos%d' % d, 'h')), q.linarith(), q.linarith()))
            destruct(p, ('Z.exists_divmod', lit(24), 'h24', 'p'), ['q', 'r', 'e', 'h0', 'h1'])
            p.interval_cases('r', 0, 23)
            for r in range(24):                                                    # p = 24 q + r
                if r % 2 == 0:
                    p.exfalso(); p.apply('n2'); p.exists(p.T('q', lambda q, r=r: 12 * q + r // 2)); p.linarith()
                elif r % 3 == 0:
                    p.exfalso(); p.apply('n3'); p.exists(p.T('q', lambda q, r=r: 8 * q + r // 3)); p.linarith()
                else:
                    p.rw_at('e', 'r_eq'); p.rw('e')
                    p.exists(p.T('q', lambda q, r=r: 24 * q * q + 2 * r * q + (r * r - 1) // 24)); p.ring()


@problem('amc_prime_02', text='Primes p and p + 2 are called twin primes. Show that if p > 3 and p, p + 2 are both prime, then 6 divides p + 1.', topics=('number-theory', 'prime', 'divisibility', 'case-analysis'), level=3,
         spec=spec(('Z',), [PRIME(X(0)), PRIME(lambda p: p + 2), LT(3, X(0))], DVD(6, lambda p: p + 1), domains=(range(-2, 90),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd; P = w['Z.prime']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_prime_02', [v.Z], [lambda p: P(p), lambda p: P(p + 2), lambda p: v.lt(lit(3), p)], lambda p: D(lit(6), p + 1))
        def _(p):
            p.intro('p', 'hp', 'hq', 'h3'); p.cases('hp', names=['h1', 'hd']); p.cases('hq', names=['h1q', 'hdq'])
            p.have('pos2', p.T('', lambda: v.pos(lit(2))), lambda q: q.linarith())
            p.have('pos3', p.T('', lambda: v.pos(lit(3))), lambda q: q.linarith())
            p.have('h6', p.T('', lambda: v.pos(lit(6))), lambda q: q.linarith())
            for d in (2, 3):
                p.have('n%d' % d, p.T('p', lambda p_, d=d: v.Not(D(lit(d), p_))),
                       lambda q, d=d: (q.intro('h'), q.obtain(['c', 'c'], ('hd', lit(d), 'pos%d' % d, 'h')), q.linarith(), q.linarith()))
            p.have('m3', p.T('p', lambda p_: v.Not(D(lit(3), p_ + 2))),                       # 3 does not divide p + 2 either: p + 2 is a prime larger than 3
                   lambda q: (q.intro('h'), q.obtain(['c', 'c'], ('hdq', lit(3), 'pos3', 'h')), q.linarith(), q.linarith()))
            destruct(p, ('Z.exists_divmod', lit(6), 'h6', 'p'), ['q', 'r', 'e', 'h0', 'h1'])
            p.interval_cases('r', 0, 5)
            p.exfalso(); p.apply('n2'); p.exists(p.T('q', lambda q: 3 * q)); p.linarith()          # r = 0
            p.exfalso(); p.apply('m3'); p.exists(p.T('q', lambda q: 2 * q + 1)); p.linarith()      # r = 1: p + 2 = 6 q + 3
            p.exfalso(); p.apply('n2'); p.exists(p.T('q', lambda q: 3 * q + 1)); p.linarith()      # r = 2
            p.exfalso(); p.apply('n3'); p.exists(p.T('q', lambda q: 2 * q + 1)); p.linarith()      # r = 3
            p.exfalso(); p.apply('n2'); p.exists(p.T('q', lambda q: 3 * q + 2)); p.linarith()      # r = 4
            p.exists(p.T('q', lambda q: q + 1)); p.linarith()                                      # r = 5: p + 1 = 6 (q + 1)


@problem('amc_prime_03', text='The sum of two primes is 99. What is their product?', topics=('number-theory', 'prime', 'parity'), level=2, kind='answer', answer='194',
         spec=spec(('Z', 'Z'), [PRIME(X(0)), PRIME(X(1)), EQ(lambda p, q: p + q, 99)], EQ(lambda p, q: p * q, 194), domains=(range(0, 100), range(0, 100)), points=10000))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd; P = w['Z.prime']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_prime_03', [v.Z, v.Z], [lambda p, q: P(p), lambda p, q: P(q), lambda p, q: Eq(p + q, lit(99))], lambda p, q: Eq(p * q, lit(194)))
        def _(p):
            p.intro('p', 'q', 'hp', 'hq', 'h'); p.cases('hp', names=['h1', 'hd']); p.cases('hq', names=['h1q', 'hdq'])
            p.have('pos2', p.T('', lambda: v.pos(lit(2))), lambda q_: q_.linarith())
            p.obtain(['k', 'c'], ('Z.even_or_odd', 'p')); p.obtain(['m', 'd'], ('Z.even_or_odd', 'q'))
            p.cases('c', names=['c', 'c'])
            # p = 2 k: 2 divides p, so p = 2 and q = 97
            p.have('d2', p.T('p', lambda p_: D(lit(2), p_)), lambda q_: (q_.exists(q_.h('k')), q_.linarith()))
            p.obtain(['e', 'e'], ('hd', lit(2), 'pos2', 'd2'))
            p.exfalso(); p.linarith()                                                          # 2 = 1
            p.have('ep', p.T('p', lambda p_: Eq(p_, lit(2))), lambda q_: q_.linarith())         # 2 = p
            p.have('eq', p.T('q', lambda q_: Eq(q_, lit(97))), lambda q_: q_.linarith())
            p.linear_combination((p.h('q'), 'ep'), (2, 'eq'))
            # p = 2 k + 1 and q = 2 m: q = 2 and p = 97;  p = 2 k + 1 and q = 2 m + 1: p + q is even
            p.cases('d', names=['d', 'd'])
            p.have('d2', p.T('q', lambda q_: D(lit(2), q_)), lambda q_: (q_.exists(q_.h('m')), q_.linarith()))
            p.obtain(['e', 'e'], ('hdq', lit(2), 'pos2', 'd2'))
            p.exfalso(); p.linarith()
            p.have('eq', p.T('q', lambda q_: Eq(q_, lit(2))), lambda q_: q_.linarith())
            p.have('ep', p.T('p', lambda p_: Eq(p_, lit(97))), lambda q_: q_.linarith())
            p.linear_combination((p.h('p'), 'eq'), (2, 'ep'))
            K.no_multiple(p, p.T('k m', lambda k, m: k + m + 1), 2, 99)                            # 2 (k + m + 1) = 99


@problem('amc_prime_04', text='Find all primes p for which p^2 + 2 is also a prime.', topics=('number-theory', 'prime', 'modular', 'case-analysis'), level=3, kind='answer', answer='p = 3',
         spec=spec(('Z',), [PRIME(X(0)), PRIME(lambda p: p * p + 2)], EQ(X(0), 3), domains=(range(-2, 40),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd; P = w['Z.prime']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_prime_04', [v.Z], [lambda p: P(p), lambda p: P(p * p + 2)], lambda p: Eq(p, lit(3)))
        def _(p):
            p.intro('p', 'hp', 'hq'); p.cases('hp', names=['h1', 'hd']); p.cases('hq', names=['h1q', 'hdq'])
            p.have('pos3', p.T('', lambda: v.pos(lit(3))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(3), 'pos3', 'p'), ['q', 'r', 'e', 'h0', 'h1'])
            p.interval_cases('r', 0, 2)
            # p = 3 q: 3 divides p, so p = 3
            p.have('d3', p.T('p', lambda p_: D(lit(3), p_)), lambda q: (q.exists(q.h('q')), q.linarith()))
            p.obtain(['c', 'c'], ('hd', lit(3), 'pos3', 'd3')); p.exfalso(); p.linarith(); p.linarith()
            # p = 3 q + 1 and p = 3 q + 2: 3 divides p^2 + 2, which would be 3
            for r, wit in ((1, lambda q: 3 * q * q + 2 * q + 1), (2, lambda q: 3 * q * q + 4 * q + 2)):
                p.rw_at('e', 'r_eq')
                p.have('d3', p.T('p', lambda p_: D(lit(3), p_ * p_ + 2)), lambda q_, wit=wit: (q_.rw('e'), q_.exists(q_.T('q', wit)), q_.ring()))
                p.obtain(['c', 'c'], ('hdq', lit(3), 'pos3', 'd3')); p.exfalso(); p.linarith(); p.exfalso(); p.nlinarith()


@problem('amc_prime_05', text='Show that n^4 + 4 is not a prime for any integer n > 1.', topics=('number-theory', 'prime', 'identity'), level=3,
         spec=spec(('Z',), [LT(1, X(0))], lambda n: not PRIME(lambda n: n ** 4 + 4)(n), atoms=[(lambda n: n ** 4 + 4,)], domains=(range(-3, 12),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd; P = w['Z.prime']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_prime_05', [v.Z], [lambda n: v.lt(lit(1), n)], lambda n: v.Not(P(n ** 4 + 4)))
        def _(p):
            p.intro('n', 'hn')
            p.have('e', p.T('n', lambda n: Eq(n ** 4 + 4, (n * n + 2 * n + 2) * (n * n - 2 * n + 2))), lambda q: q.ring())
            p.rw('e'); p.apply('not_prime_mul')
            p.nlinarith(); p.nlinarith()


@problem('amc_gcd_01', text='Show that n + 1 and n^2 + n + 1 have greatest common divisor 1 for every integer n.', topics=('number-theory', 'gcd', 'identity'), level=2,
         spec=spec(('Z',), [], GCD(lambda n: n + 1, lambda n: n * n + n + 1, 1), domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; G = w['Z.is_gcd']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_gcd_01', [v.Z], [], lambda n: G(n + 1, n * n + n + 1, v.one))
        def _(p):                                                                             # (n + 1)(-n) + (n^2 + n + 1) 1 = 1
            p.intro('n'); p.apply('Z.is_gcd_one_of_coprime')
            p.exists(p.T('n', lambda n: -n)); p.exists(lit(1)); p.ring()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ inequalities and bounds
@problem('amc_ineq_01', text='If a and b are integers with a + b = 10, show that ab <= 25.', topics=('inequality', 'algebra'), level=1,
         spec=spec(('Z', 'Z'), [EQ(lambda a, b: a + b, 10)], LE(lambda a, b: a * b, 25)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_ineq_01', [v.Z, v.Z], [lambda a, b_: Eq(a + b_, lit(10))], lambda a, b_: v.le(a * b_, lit(25)))
        def _(p):                                                                  # 25 - ab = (a - 5)^2 once b = 10 - a
            p.intro('a', 'b', 'h')
            p.have('e', p.T('a b', lambda a, b_: Eq(a * b_, 10 * a - a * a)), lambda q: q.linear_combination((q.h('a'), 'h')))
            p.have('s', ('Z.sq_nonneg', p.T('a', lambda a: a - 5)))
            p.linarith()


@problem('amc_ineq_02', text='Show that x^2 + y^2 + z^2 >= xy + yz + zx for all integers x, y, z.', topics=('inequality',), level=2,
         spec=spec(('Z', 'Z', 'Z'), [], LE(lambda x, y, z: x * y + y * z + z * x, lambda x, y, z: x * x + y * y + z * z), domains=(range(-4, 5),) * 3))
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_ineq_02', [v.Z, v.Z, v.Z], [], lambda x, y, z: v.le(x * y + y * z + z * x, x ** 2 + y ** 2 + z ** 2))
        def _(p):                                                                  # twice the difference is (x - y)^2 + (y - z)^2 + (z - x)^2
            p.intro('x', 'y', 'z')
            p.have('s1', ('Z.sq_nonneg', p.T('x y', lambda x, y: x - y)))
            p.have('s2', ('Z.sq_nonneg', p.T('y z', lambda y, z: y - z)))
            p.have('s3', ('Z.sq_nonneg', p.T('x z', lambda x, z: z - x)))
            p.linarith()


@problem('amc_ineq_03', text='Show that a^3 + b^3 + c^3 >= 3abc for all non-negative integers a, b, c.', topics=('inequality', 'identity'), level=3,
         spec=spec(('Z', 'Z', 'Z'), [LE(0, X(0)), LE(0, X(1)), LE(0, X(2))], LE(lambda a, b, c: 3 * a * b * c, lambda a, b, c: a ** 3 + b ** 3 + c ** 3), domains=(range(-4, 5),) * 3))
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_ineq_03', [v.Z, v.Z, v.Z], [lambda a, b_, c: v.nonneg(a), lambda a, b_, c: v.nonneg(b_), lambda a, b_, c: v.nonneg(c)], lambda a, b_, c: v.le(3 * a * b_ * c, a ** 3 + b_ ** 3 + c ** 3))
        def _(p):                                                                  # a^3 + b^3 + c^3 - 3abc = (a + b + c)((a - b)^2 + (b - c)^2 + (c - a)^2) / 2
            p.intro('a', 'b', 'c', 'ha', 'hb', 'hc')
            p.have('s', p.T('a b c', lambda a, b_, c: v.nonneg(a + b_ + c)), lambda q: q.linarith())
            p.have('s1', ('Z.sq_nonneg', p.T('a b', lambda a, b_: a - b_)))
            p.have('s2', ('Z.sq_nonneg', p.T('b c', lambda b_, c: b_ - c)))
            p.have('s3', ('Z.sq_nonneg', p.T('a c', lambda a, c: c - a)))
            p.have('u', p.T('a b c', lambda a, b_, c: v.nonneg((a - b_) * (a - b_) + (b_ - c) * (b_ - c) + (c - a) * (c - a))), lambda q: q.linarith())
            p.have('t', ('Z.mul_nonneg', p.T('a b c', lambda a, b_, c: a + b_ + c), p.T('a b c', lambda a, b_, c: (a - b_) * (a - b_) + (b_ - c) * (b_ - c) + (c - a) * (c - a)), 's', 'u'))
            p.linarith()


@problem('amc_ineq_04', text='Show that n^2 >= n for every integer n.', topics=('inequality', 'case-analysis'), level=1,
         spec=spec(('Z',), [], LE(X(0), lambda n: n * n)))
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_ineq_04', [v.Z], [], lambda n: v.le(n, n ** 2))
        def _(p):
            p.intro('n'); p.obtain(['c', 'c'], ('Z.le_or_lt', lit(1), 'n'))
            p.nlinarith()                                                          # 1 <= n
            p.have('s', ('Z.sq_nonneg', 'n')); p.linarith()                        # n < 1, so n <= 0 <= n^2


@problem('amc_ineq_05', text='Show that a^3 + b^3 >= a^2 b + a b^2 for all non-negative integers a and b.', topics=('inequality', 'identity'), level=2,
         spec=spec(('Z', 'Z'), [LE(0, X(0)), LE(0, X(1))], LE(lambda a, b: a * a * b + a * b * b, lambda a, b: a ** 3 + b ** 3), domains=(range(-5, 6),) * 2))
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_ineq_05', [v.Z, v.Z], [lambda a, b_: v.nonneg(a), lambda a, b_: v.nonneg(b_)], lambda a, b_: v.le(a * a * b_ + a * b_ * b_, a ** 3 + b_ ** 3))
        def _(p):                                                                  # a^3 + b^3 - a^2 b - a b^2 = (a + b)(a - b)^2
            p.intro('a', 'b', 'ha', 'hb')
            p.have('s', p.T('a b', lambda a, b_: v.nonneg(a + b_)), lambda q: q.linarith())
            p.have('t', ('Z.mul_nonneg', p.T('a b', lambda a, b_: a + b_), p.T('a b', lambda a, b_: (a - b_) * (a - b_)), 's', ('Z.sq_nonneg', p.T('a b', lambda a, b_: a - b_))))
            p.linarith()


@problem('amc_bound_01', text='What is the largest integer n for which n^2 < 200?', topics=('arithmetic', 'bounding'), level=1, kind='answer', answer='14',
         spec=spec(('Z',), [LT(lambda n: n * n, 200)], [LE(X(0), 14), LT(14 * 14, 200)], domains=(range(-20, 21),)))
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_bound_01', [v.Z], [lambda n: v.lt(n * n, lit(200))], lambda n: v.le(n, lit(14)) & v.lt(lit(14) * lit(14), lit(200)))
        def _(p):                                                                  # 14 works, and n >= 15 would give n^2 >= 225
            p.intro('n', 'h'); p.split()
            p.obtain(['c', 'c'], ('Z.le_or_lt', 'n', lit(14)))
            p.exact('c')
            p.exfalso(); p.nlinarith()
            p.linarith()


@problem('amc_bound_02', text='What is the smallest positive integer n for which n^3 > 1000?', topics=('arithmetic', 'bounding', 'case-analysis'), level=2, kind='answer', answer='11',
         spec=spec(('Z',), [LT(0, X(0)), LT(1000, lambda n: n ** 3)], [LE(11, X(0)), LT(1000, 11 ** 3)], domains=(range(-3, 16),)))
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_bound_02', [v.Z], [lambda n: v.pos(n), lambda n: v.lt(lit(1000), n ** 3)], lambda n: v.le(lit(11), n) & v.lt(lit(1000), lit(11) ** 3))
        def _(p):                                                                  # n <= 10 would give n^3 <= 1000
            p.intro('n', 'h0', 'h'); p.split()
            p.obtain(['c', 'c'], ('Z.le_or_lt', lit(11), 'n'))
            p.exact('c')
            p.exfalso(); p.interval_cases('n', 1, 10)
            for k in range(1, 11): p.rw_at('h', 'n_eq'); p.linarith()
            p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ sums and induction
@problem('amc_sum_01', text='What is 1 + 2 + 3 + ... + 100?', topics=('sum', 'arithmetic'), level=1, kind='answer', answer='5050',
         spec=spec((), [], EQ(lambda: sum(i + 1 for i in range(100)), 5050)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; zsum = w['Z.sum']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_sum_01', [], [], lambda: Eq(zsum(w.lam(v.Z, lambda i: i + 1), v.nlit(100)), lit(5050)))
        def _(p):
            p.have('hs', ('Z.sum_first', v.nlit(100))); p.linarith()


@problem('amc_sum_02', text='What is the sum of the odd numbers 1 + 3 + 5 + ... + 99?', topics=('sum', 'arithmetic'), level=1, kind='answer', answer='2500',
         spec=spec((), [], EQ(lambda: sum(2 * i + 1 for i in range(50)), 2500)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; zsum = w['Z.sum']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_sum_02', [], [], lambda: Eq(zsum(w.lam(v.Z, lambda i: 2 * i + 1), v.nlit(50)), lit(2500)))
        def _(p):
            p.have('hs', ('Z.sum_odd', v.nlit(50))); p.linarith()


@problem('amc_sum_03', text='What is the sum of the squares of the integers from 0 to 30: 0^2 + 1^2 + 2^2 + ... + 30^2?', topics=('sum', 'arithmetic'), level=2, kind='answer', answer='9455',
         spec=spec((), [], EQ(lambda: sum(i ** 2 for i in range(31)), 9455)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; zsum = w['Z.sum']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_sum_03', [], [], lambda: Eq(zsum(w.lam(v.Z, lambda i: i ** 2), v.nlit(31)), lit(9455)))
        def _(p):
            p.have('hs', ('Z.sum_sq', v.nlit(31))); p.linarith()


@problem('amc_sum_04', text='What is the sum of the cubes of the integers from 0 to 20: 0^3 + 1^3 + 2^3 + ... + 20^3?', topics=('sum', 'arithmetic'), level=2, kind='answer', answer='44100',
         spec=spec((), [], EQ(lambda: sum(i ** 3 for i in range(21)), 44100)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; zsum = w['Z.sum']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_sum_04', [], [], lambda: Eq(zsum(w.lam(v.Z, lambda i: i ** 3), v.nlit(21)), lit(44100)))
        def _(p):
            p.have('hs', ('Z.sum_cube', v.nlit(21))); p.linarith()


@problem('amc_ind_01', text='Show that 3 divides 4^n - 1 for every natural number n.', topics=('number-theory', 'divisibility', 'induction'), level=2,
         spec=spec(('N',), [], DVD(3, lambda n: 4 ** n - 1), domains=(range(0, 12),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_ind_01', [v.N], [], lambda n: D(lit(3), lit(4) ** n - 1))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            p.exists(lit(0)); p.ring()
            p.intro('m', 'ih'); p.cases('ih', names=['k', 'hk'])
            p.rw('Z.pow_succ'); p.exists(p.T('k', lambda k: 4 * k + 1)); p.linear_combination((4, 'hk'))


@problem('amc_ind_02', text='Show that 2^n >= n + 1 for every natural number n.', topics=('inequality', 'induction'), level=2,
         spec=spec(('N',), [], LE(lambda n: n + 1, lambda n: 2 ** n), domains=(range(0, 12),)))
def _(w, b):
    v = K.Vocab(w); zof = w['Z.of_N']; Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_ind_02', [v.N], [], lambda n: v.le(zof(n) + 1, lit(2) ** n))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            p.have('e0', p.T('', lambda: Eq(lit(2) ** v.nlit(0), lit(1))), lambda q: q.ring()); p.linarith()
            p.intro('m', 'ih'); p.rw('Z.pow_succ'); p.rw('Z.of_N_succ')
            p.have('hm', ('Z.nonneg_of_N', 'm')); p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ equations and Diophantine problems
def atom(*args):
    """the numbers of one relation atom as functions of the variables (a number stands for the constant function)"""
    return tuple(a if callable(a) else (lambda *v, c=a: c) for a in args)


@problem('amc_eq_01', text='Find all integers x with x^2 - 5x + 6 = 0.', topics=('algebra',), level=1, kind='answer', answer='x = 2 or x = 3',
         spec=spec(('Z',), [EQ(lambda x: x * x - 5 * x + 6, 0)], lambda x: x in (2, 3), atoms=[atom(X(0), 2), atom(X(0), 3)], domains=(range(-6, 10),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_eq_01', [v.Z], [lambda x: Eq(x * x - 5 * x + 6, v.zero)], lambda x: Eq(x, lit(2)) | Eq(x, lit(3)))
        def _(p):
            p.intro('x', 'h')
            p.have('e', p.T('x', lambda x: Eq((x - 2) * (x - 3), v.zero)), lambda q: q.linear_combination((1, 'h')))
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('x', lambda x: x - 2), p.T('x', lambda x: x - 3), 'e'))
            p.left(); p.linarith()
            p.right(); p.linarith()


@problem('amc_eq_02', text='Find all integers x with (x - 3)^2 = 25.', topics=('algebra',), level=2, kind='answer', answer='x = 8 or x = -2',
         spec=spec(('Z',), [EQ(lambda x: (x - 3) * (x - 3), 25)], lambda x: x in (8, -2), atoms=[atom(X(0), 8), atom(X(0), -2)], domains=(range(-6, 12),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_eq_02', [v.Z], [lambda x: Eq((x - 3) * (x - 3), lit(25))], lambda x: Eq(x, lit(8)) | Eq(x, -lit(2)))
        def _(p):
            p.intro('x', 'h')
            p.have('e', p.T('x', lambda x: Eq((x - 8) * (x + 2), v.zero)), lambda q: q.linear_combination((1, 'h')))
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('x', lambda x: x - 8), p.T('x', lambda x: x + 2), 'e'))
            p.left(); p.linarith()
            p.right(); p.linarith()


@problem('amc_eq_03', text='Find all integers x with x^3 - 6x^2 + 11x - 6 = 0.', topics=('algebra', 'polynomial'), level=2, kind='answer', answer='x = 1, 2 or 3',
         spec=spec(('Z',), [EQ(lambda x: x ** 3 - 6 * x * x + 11 * x - 6, 0)], lambda x: x in (1, 2, 3), atoms=[atom(X(0), 1), atom(X(0), 2), atom(X(0), 3)], domains=(range(-4, 8),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_eq_03', [v.Z], [lambda x: Eq(x ** 3 - 6 * x ** 2 + 11 * x - 6, v.zero)], lambda x: Eq(x, lit(1)) | (Eq(x, lit(2)) | Eq(x, lit(3))))
        def _(p):
            p.intro('x', 'h')
            p.have('e', p.T('x', lambda x: Eq((x - 1) * ((x - 2) * (x - 3)), v.zero)), lambda q: q.linear_combination((1, 'h')))
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('x', lambda x: x - 1), p.T('x', lambda x: (x - 2) * (x - 3)), 'e'))
            p.left(); p.linarith()
            p.obtain(['d', 'd'], ('Z.mul_eq_zero', p.T('x', lambda x: x - 2), p.T('x', lambda x: x - 3), 'c'))
            p.right(); p.left(); p.linarith()
            p.right(); p.right(); p.linarith()


@problem('amc_dio_01', text='Find all pairs of positive integers (x, y) with xy + x + y = 20.', topics=('diophantine', 'algebra'), level=3, kind='answer', answer='(2, 6) and (6, 2)',
         spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), EQ(lambda x, y: x * y + x + y, 20)], lambda x, y: (x, y) in ((2, 6), (6, 2)),
                   atoms=[atom(X(0), 2), atom(X(1), 6), atom(X(0), 6), atom(X(1), 2)], domains=(range(-2, 23), range(-2, 23))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_dio_01', [v.Z, v.Z], [lambda x, y: v.pos(x), lambda x, y: v.pos(y), lambda x, y: Eq(x * y + x + y, lit(20))],
                   lambda x, y: (Eq(x, lit(2)) & Eq(y, lit(6))) | (Eq(x, lit(6)) & Eq(y, lit(2))))
        def _(p):                                                                  # (x + 1)(y + 1) = 21
            p.intro('x', 'y', 'hx', 'hy', 'h')
            K.let(p, 'u', 'x', lambda x: x + 1); K.let(p, 'v', 'y', lambda y: y + 1)
            p.have('hu', p.T('u', v.pos), lambda q: q.linarith())
            p.have('hv', p.T('v', v.pos), lambda q: q.linarith())
            p.have('e', p.T('u v', lambda u, v_: Eq(u * v_, lit(21))), lambda q: q.linear_combination((q.h('v'), 'u_def'), (q.T('x', lambda x: x + 1), 'v_def'), (1, 'h')))

            def pair(p, d, c):
                xv, yv = d - 1, c - 1
                if xv <= 0 or yv <= 0: p.exfalso(); p.linarith(); return
                p.have('ex', p.T('x', lambda x: Eq(x, lit(xv))), lambda q: q.linarith())
                p.have('ey', p.T('y', lambda y: Eq(y, lit(yv))), lambda q: q.linarith())
                if (xv, yv) == (2, 6): p.left()
                else: p.right()
                p.split(); p.linarith(); p.linarith()
            K.divisor_cases(p, 'u', 'v', 'e', 21, pair)


@problem('amc_dio_02', text='Find all pairs of positive integers (x, y) with x^2 - y^2 = 15.', topics=('diophantine', 'algebra'), level=3, kind='answer', answer='(4, 1) and (8, 7)',
         spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), EQ(lambda x, y: x * x - y * y, 15)], lambda x, y: (x, y) in ((4, 1), (8, 7)),
                   atoms=[atom(X(0), 4), atom(X(1), 1), atom(X(0), 8), atom(X(1), 7)], domains=(range(-2, 12), range(-2, 12))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_dio_02', [v.Z, v.Z], [lambda x, y: v.pos(x), lambda x, y: v.pos(y), lambda x, y: Eq(x * x - y * y, lit(15))],
                   lambda x, y: (Eq(x, lit(4)) & Eq(y, lit(1))) | (Eq(x, lit(8)) & Eq(y, lit(7))))
        def _(p):                                                                  # (x - y)(x + y) = 15
            p.intro('x', 'y', 'hx', 'hy', 'h')
            p.have('xy', p.T('x y', lambda x, y: v.lt(y, x)), lambda q: (q.obtain(['c', 'c'], ('Z.le_or_lt', q.h('x'), q.h('y'))), q.exfalso(), q.nlinarith(), q.exact('c')))
            K.let(p, 'u', 'x y', lambda x, y: x - y); K.let(p, 'v', 'x y', lambda x, y: x + y)
            p.have('hu', p.T('u', v.pos), lambda q: q.linarith())
            p.have('hv', p.T('v', v.pos), lambda q: q.linarith())
            p.have('e', p.T('u v', lambda u, v_: Eq(u * v_, lit(15))), lambda q: q.linear_combination((q.h('v'), 'u_def'), (q.T('x y', lambda x, y: x - y), 'v_def'), (1, 'h')))

            def pair(p, d, c):
                if c <= d: p.exfalso(); p.linarith(); return
                xv, yv = (c + d) // 2, (c - d) // 2
                p.have('ex', p.T('x', lambda x: Eq(x, lit(xv))), lambda q: q.linarith())
                p.have('ey', p.T('y', lambda y: Eq(y, lit(yv))), lambda q: q.linarith())
                if (xv, yv) == (4, 1): p.left()
                else: p.right()
                p.split(); p.linarith(); p.linarith()
            K.divisor_cases(p, 'u', 'v', 'e', 15, pair)


@problem('amc_dio_03', text='Find all pairs of positive integers (x, y) with 3x + 5y = 31.', topics=('diophantine', 'case-analysis'), level=2, kind='answer', answer='(2, 5) and (7, 2)',
         spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), EQ(lambda x, y: 3 * x + 5 * y, 31)], lambda x, y: (x, y) in ((2, 5), (7, 2)),
                   atoms=[atom(X(0), 2), atom(X(1), 5), atom(X(0), 7), atom(X(1), 2)], domains=(range(-2, 12), range(-2, 8))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_dio_03', [v.Z, v.Z], [lambda x, y: v.pos(x), lambda x, y: v.pos(y), lambda x, y: Eq(3 * x + 5 * y, lit(31))],
                   lambda x, y: (Eq(x, lit(2)) & Eq(y, lit(5))) | (Eq(x, lit(7)) & Eq(y, lit(2))))
        def _(p):                                                                  # x <= 8, and 5 y = 31 - 3 x
            p.intro('x', 'y', 'hx', 'hy', 'h')
            p.have('hb', p.T('x', lambda x: v.le(x, lit(8))), lambda q: q.linarith())
            p.interval_cases('x', 1, 8)
            for k in range(1, 9):
                if (31 - 3 * k) % 5 != 0: K.no_multiple(p, p.h('y'), 5, 31 - 3 * k)
                else:
                    yv = (31 - 3 * k) // 5
                    p.have('ey', p.T('y', lambda y, yv=yv: Eq(y, lit(yv))), lambda q: q.linarith())
                    if k == 2: p.left()
                    else: p.right()
                    p.split(); p.linarith(); p.linarith()


@problem('amc_dio_04', text='The sum of two numbers is 13 and their product is 36. What are the numbers?', topics=('algebra', 'diophantine'), level=2, kind='answer', answer='4 and 9',
         spec=spec(('Z', 'Z'), [EQ(lambda a, b: a + b, 13), EQ(lambda a, b: a * b, 36)], lambda a, b: (a, b) in ((4, 9), (9, 4)),
                   atoms=[atom(X(0), 4), atom(X(1), 9), atom(X(0), 9), atom(X(1), 4)], domains=(range(-3, 16), range(-3, 16))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_dio_04', [v.Z, v.Z], [lambda a, b_: Eq(a + b_, lit(13)), lambda a, b_: Eq(a * b_, lit(36))],
                   lambda a, b_: (Eq(a, lit(4)) & Eq(b_, lit(9))) | (Eq(a, lit(9)) & Eq(b_, lit(4))))
        def _(p):                                                                  # (a - 4)(a - 9) = a (a + b - 13) - (ab - 36)
            p.intro('a', 'b', 'h1', 'h2')
            p.have('e', p.T('a', lambda a: Eq((a - 4) * (a - 9), v.zero)), lambda q: q.linear_combination((q.h('a'), 'h1'), (-1, 'h2')))
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('a', lambda a: a - 4), p.T('a', lambda a: a - 9), 'e'))
            p.left(); p.split(); p.linarith(); p.linarith()
            p.right(); p.split(); p.linarith(); p.linarith()


@problem('amc_dvs_01', text='The product of two positive integers is 12. Which values can their sum take?', topics=('number-theory', 'divisibility', 'case-analysis'), level=2, kind='answer', answer='7, 8 or 13',
         spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), EQ(lambda x, y: x * y, 12)], lambda x, y: x + y in (7, 8, 13),
                   atoms=[atom(lambda x, y: x + y, 7), atom(lambda x, y: x + y, 8), atom(lambda x, y: x + y, 13)], domains=(range(-2, 15), range(-2, 15))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_dvs_01', [v.Z, v.Z], [lambda x, y: v.pos(x), lambda x, y: v.pos(y), lambda x, y: Eq(x * y, lit(12))],
                   lambda x, y: Eq(x + y, lit(7)) | (Eq(x + y, lit(8)) | Eq(x + y, lit(13))))
        def _(p):
            p.intro('x', 'y', 'hx', 'hy', 'h')

            def pair(p, d, c):
                s = d + c
                if s == 7: p.left()
                elif s == 8: p.right(); p.left()
                else: p.right(); p.right()
                p.linarith()
            K.divisor_cases(p, 'x', 'y', 'h', 12, pair)


@problem('amc_dvs_02', text='Find all positive integers n greater than 3 that divide 12.', topics=('number-theory', 'divisibility', 'case-analysis'), level=3, kind='answer', answer='n = 4, 6 or 12',
         spec=spec(('Z',), [LT(0, X(0))], (lambda n: dvd_(n, 12) and 3 < n, lambda n: n in (4, 6, 12)),
                   atoms=[atom(X(0), 12), atom(3, X(0)), atom(X(0), 4), atom(X(0), 6), atom(X(0), 12)], domains=(range(-2, 20),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_dvs_02', [v.Z], [lambda n: v.pos(n)], lambda n: v.Iff(D(n, lit(12)) & v.lt(lit(3), n), Eq(n, lit(4)) | (Eq(n, lit(6)) | Eq(n, lit(12)))))
        def _(p):                                                                  # n <= 12 (Z.le_of_dvd), then the cases 4 .. 12
            p.intro('n', 'hn'); p.split()
            p.intro('h'); p.cases('h', names=['h', 'h3'])
            p.have('p12', p.T('', lambda: v.pos(lit(12))), lambda q: q.linarith())
            p.have('hb', ('Z.le_of_dvd', 'n', lit(12), 'p12', 'h'))
            p.cases('h', names=['k', 'hk'])
            p.interval_cases('n', 4, 12)
            for k in range(4, 13):
                if 12 % k != 0: p.rw_at('hk', 'n_eq'); K.no_multiple(p, p.h('k'), k, 12)
                elif k == 4: p.left(); p.linarith()
                elif k == 6: p.right(); p.left(); p.linarith()
                else: p.right(); p.right(); p.linarith()
            p.intro('h'); p.cases('h', names=['c', 'c'])
            p.split(); p.exists(lit(3)); p.linarith(); p.linarith()
            p.cases('c', names=['c', 'c'])
            p.split(); p.exists(lit(2)); p.linarith(); p.linarith()
            p.split(); p.exists(lit(1)); p.linarith(); p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ more divisibility, digits and algebra
@problem('amc_div_07', text='Show that the product of two consecutive integers, n(n + 1), is even.', topics=('number-theory', 'parity', 'divisibility'), level=1,
         spec=spec(('Z',), [], DVD(2, lambda n: n * (n + 1)), domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_div_07', [v.Z], [], lambda n: D(lit(2), n * (n + 1)))
        def _(p):
            p.intro('n'); p.obtain(['k', 'c'], ('Z.even_or_odd', 'n')); p.cases('c', names=['c', 'c'])
            p.rw('c'); p.exists(p.T('k', lambda k: k * (2 * k + 1))); p.ring()
            p.rw('c'); p.exists(p.T('k', lambda k: (2 * k + 1) * (k + 1))); p.ring()


@problem('amc_div_08', text='Show that a - b divides a^2 - b^2 for all integers a and b.', topics=('number-theory', 'divisibility', 'identity'), level=1,
         spec=spec(('Z', 'Z'), [], DVD(lambda a, b: a - b, lambda a, b: a * a - b * b)))
def _(w, b):
    v = K.Vocab(w); D = v.dvd
    with v.ints():
        @K.theorem(w, 'amc_div_08', [v.Z, v.Z], [], lambda a, b_: D(a - b_, a * a - b_ * b_))
        def _(p):
            p.intro('a', 'b'); p.exists(p.T('a b', lambda a, b_: a + b_)); p.ring()


@problem('amc_div_09', text='A divisibility test for 7: for all integers a and b, 7 divides 10a + b exactly when 7 divides a - 2b.', topics=('number-theory', 'divisibility'), level=2,
         spec=spec(('Z', 'Z'), [], (lambda a, b: (10 * a + b) % 7 == 0, lambda a, b: (a - 2 * b) % 7 == 0), atoms=[atom(7, lambda a, b: 10 * a + b), atom(7, lambda a, b: a - 2 * b)]))
def _(w, b):
    v = K.Vocab(w); D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_div_09', [v.Z, v.Z], [], lambda a, b_: v.Iff(D(lit(7), 10 * a + b_), D(lit(7), a - 2 * b_)))
        def _(p):
            p.intro('a', 'b'); p.split()
            p.intro('h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('a b k', lambda a, b_, k: 5 * k - 7 * a - b_)); p.linear_combination((5, 'hk'))
            p.intro('h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('a b k', lambda a, b_, k: 3 * k + a + b_)); p.linear_combination((3, 'hk'))


@problem('amc_div_10', text='Show that if n is an odd integer, then 8 divides n^2 - 1.', topics=('number-theory', 'divisibility', 'parity', 'case-analysis'), level=3,
         spec=spec(('Z',), [lambda n: n % 2 != 0], DVD(8, lambda n: n * n - 1), hatoms=[atom(2, X(0))], domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_div_10', [v.Z], [lambda n: v.Not(D(lit(2), n))], lambda n: D(lit(8), n * n - 1))
        def _(p):                                                                  # n = 2k + 1, k = 2m or 2m + 1: n^2 - 1 = 8 (2m^2 + m) or 8 (2m^2 + 3m + 1)
            p.intro('n', 'h'); p.obtain(['k', 'c'], ('Z.even_or_odd', 'n')); p.cases('c', names=['c', 'c'])
            p.exfalso(); p.apply('h'); p.exists('k'); p.linarith()
            p.obtain(['m', 'd'], ('Z.even_or_odd', 'k')); p.cases('d', names=['d', 'd'])
            p.rw('c'); p.rw('d'); p.exists(p.T('m', lambda m: 2 * m * m + m)); p.ring()
            p.rw('c'); p.rw('d'); p.exists(p.T('m', lambda m: 2 * m * m + 3 * m + 1)); p.ring()


@problem('amc_dig_01', text='A two-digit number has digits a and b (it is 10a + b). Show that the difference between it and the number with the digits reversed is divisible by 9.', topics=('number-theory', 'digits', 'divisibility'), level=1,
         spec=spec(('Z', 'Z'), [LE(1, X(0)), LE(X(0), 9), LE(0, X(1)), LE(X(1), 9)], DVD(9, lambda a, b: (10 * a + b) - (10 * b + a)), domains=(range(-1, 12),) * 2))
def _(w, b):
    v = K.Vocab(w); D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_dig_01', [v.Z, v.Z], [lambda a, b_: v.le(lit(1), a), lambda a, b_: v.le(a, lit(9)), lambda a, b_: v.nonneg(b_), lambda a, b_: v.le(b_, lit(9))],
                   lambda a, b_: D(lit(9), (10 * a + b_) - (10 * b_ + a)))
        def _(p):
            p.intro('a', 'b', 'h1', 'h2', 'h3', 'h4'); p.exists(p.T('a b', lambda a, b_: a - b_)); p.ring()


@problem('amc_dig_02', text='A three-digit number 100a + 10b + c (digits a, b, c) is divisible by 11 exactly when a - b + c is.', topics=('number-theory', 'divisibility', 'digits'), level=2,
         spec=spec(('Z', 'Z', 'Z'), [LE(1, X(0)), LE(X(0), 9), LE(0, X(1)), LE(X(1), 9), LE(0, X(2)), LE(X(2), 9)],
                   (lambda a, b, c: (100 * a + 10 * b + c) % 11 == 0, lambda a, b, c: (a - b + c) % 11 == 0),
                   atoms=[atom(11, lambda a, b, c: 100 * a + 10 * b + c), atom(11, lambda a, b, c: a - b + c)], domains=(range(0, 11), range(-1, 11), range(-1, 11))))
def _(w, b):
    v = K.Vocab(w); D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_dig_02', [v.Z, v.Z, v.Z], [lambda a, b_, c: v.le(lit(1), a), lambda a, b_, c: v.le(a, lit(9)), lambda a, b_, c: v.nonneg(b_), lambda a, b_, c: v.le(b_, lit(9)),
                                                      lambda a, b_, c: v.nonneg(c), lambda a, b_, c: v.le(c, lit(9))],
                   lambda a, b_, c: v.Iff(D(lit(11), 100 * a + 10 * b_ + c), D(lit(11), a - b_ + c)))
        def _(p):                                                                  # 100a + 10b + c = 11 (9a + b) + (a - b + c)
            p.intro('a', 'b', 'c', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'); p.split()
            p.intro('h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('a b k', lambda a, b_, k: k - 9 * a - b_)); p.linear_combination((1, 'hk'))
            p.intro('h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('a b k', lambda a, b_, k: k + 9 * a + b_)); p.linear_combination((1, 'hk'))


@problem('amc_dig_03', text='A four-digit number written abab (digits a, b, a, b) is divisible by 101. Show this.', topics=('number-theory', 'digits', 'divisibility'), level=1,
         spec=spec(('Z', 'Z'), [LE(1, X(0)), LE(X(0), 9), LE(0, X(1)), LE(X(1), 9)], DVD(101, lambda a, b: 1000 * a + 100 * b + 10 * a + b), domains=(range(-1, 12),) * 2))
def _(w, b):
    v = K.Vocab(w); D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_dig_03', [v.Z, v.Z], [lambda a, b_: v.le(lit(1), a), lambda a, b_: v.le(a, lit(9)), lambda a, b_: v.nonneg(b_), lambda a, b_: v.le(b_, lit(9))],
                   lambda a, b_: D(lit(101), 1000 * a + 100 * b_ + 10 * a + b_))
        def _(p):
            p.intro('a', 'b', 'h1', 'h2', 'h3', 'h4'); p.exists(p.T('a b', lambda a, b_: 10 * a + b_)); p.ring()


@problem('amc_mod_05', text='Show that n^2 + (n + 1)^2 is never divisible by 3.', topics=('number-theory', 'modular', 'case-analysis'), level=3,
         spec=spec(('Z',), [], lambda n: (n * n + (n + 1) * (n + 1)) % 3 != 0, atoms=[atom(3, lambda n: n * n + (n + 1) * (n + 1))], domains=(range(-9, 10),)))
def _(w, b):
    v = K.Vocab(w); D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_mod_05', [v.Z], [], lambda n: v.Not(D(lit(3), n * n + (n + 1) * (n + 1))))
        def _(p):
            p.intro('n', 'h'); p.cases('h', names=['k', 'hk'])
            p.have('h3', p.T('', lambda: v.pos(lit(3))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.interval_cases('r', 0, 2)
            for r in range(3):                                                   # n = 3 q + r:  3 (k - 6 q^2 - (4 r + 2) q) = r^2 + (r + 1)^2
                p.rw_at('e', 'r_eq'); p.rw_at('hk', 'e')
                K.no_multiple(p, p.T('k q', lambda k, q, r=r: k - 6 * q * q - (4 * r + 2) * q), 3, r * r + (r + 1) * (r + 1))


@problem('amc_alg_12', text='A jar holds 25 coins, all of them nickels (5 cents) or dimes (10 cents), worth 200 cents altogether. How many dimes are there?', topics=('algebra',), level=1, kind='answer', answer='15',
         spec=spec(('Z', 'Z'), [EQ(lambda n, d: n + d, 25), EQ(lambda n, d: 5 * n + 10 * d, 200)], EQ(X(1), 15), domains=(range(-2, 30), range(-2, 30))))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_12', [v.Z, v.Z], [lambda n, d: Eq(n + d, lit(25)), lambda n, d: Eq(5 * n + 10 * d, lit(200))], lambda n, d: Eq(d, lit(15)))
        def _(p):
            p.intro('n', 'd', 'h1', 'h2'); p.linarith()


@problem('amc_alg_13', text='If x + y = 10 and x^2 + y^2 = 58, what is xy?', topics=('algebra', 'identity'), level=2, kind='answer', answer='21',
         spec=spec(('Z', 'Z'), [EQ(lambda x, y: x + y, 10), EQ(lambda x, y: x * x + y * y, 58)], EQ(lambda x, y: x * y, 21), domains=(range(-2, 13),) * 2))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_13', [v.Z, v.Z], [lambda x, y: Eq(x + y, lit(10)), lambda x, y: Eq(x * x + y * y, lit(58))], lambda x, y: Eq(x * y, lit(21)))
        def _(p):                                                                  # (x + y)^2 = 100
            p.intro('x', 'y', 'h1', 'h2')
            p.have('e', p.T('x y', lambda x, y: Eq((x + y) * (x + y), lit(100))), lambda q: q.linear_combination((q.T('x y', lambda x, y: x + y + 10), 'h1')))
            p.linarith()


@problem('amc_alg_14', text='The numbers x > y satisfy x + y = 10 and (x - y)^2 = 4. What is x?', topics=('algebra',), level=2, kind='answer', answer='x = 6',
         spec=spec(('Z', 'Z'), [EQ(lambda x, y: (x - y) * (x - y), 4), EQ(lambda x, y: x + y, 10), LT(X(1), X(0))], EQ(X(0), 6), domains=(range(-2, 13),) * 2))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_14', [v.Z, v.Z], [lambda x, y: Eq((x - y) * (x - y), lit(4)), lambda x, y: Eq(x + y, lit(10)), lambda x, y: v.lt(y, x)], lambda x, y: Eq(x, lit(6)))
        def _(p):
            p.intro('x', 'y', 'h1', 'h2', 'h3')
            p.have('e', p.T('x y', lambda x, y: Eq((x - y - 2) * (x - y + 2), v.zero)), lambda q: q.linear_combination((1, 'h1')))
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('x y', lambda x, y: x - y - 2), p.T('x y', lambda x, y: x - y + 2), 'e'))
            p.linarith()
            p.exfalso(); p.linarith()


@problem('amc_alg_15', text='The product of two consecutive even positive integers is 168. What is their sum?', topics=('algebra',), level=2, kind='answer', answer='26',
         spec=spec(('Z',), [LT(0, X(0)), EQ(lambda n: n * (n + 2), 168)], EQ(lambda n: n + (n + 2), 26), domains=(range(-16, 16),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_15', [v.Z], [lambda n: v.pos(n), lambda n: Eq(n * (n + 2), lit(168))], lambda n: Eq(n + (n + 2), lit(26)))
        def _(p):
            p.intro('n', 'hn', 'h')
            p.have('e', p.T('n', lambda n: Eq((n - 12) * (n + 14), v.zero)), lambda q: q.linear_combination((1, 'h')))
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('n', lambda n: n - 12), p.T('n', lambda n: n + 14), 'e'))
            p.linarith()
            p.exfalso(); p.linarith()


@problem('amc_alg_16', text='Three consecutive integers have product 504. What is the smallest of them?', topics=('algebra', 'polynomial'), level=3, kind='answer', answer='7',
         spec=spec(('Z',), [EQ(lambda n: n * (n + 1) * (n + 2), 504)], EQ(X(0), 7), domains=(range(-12, 13),)))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_alg_16', [v.Z], [lambda n: Eq(n * (n + 1) * (n + 2), lit(504))], lambda n: Eq(n, lit(7)))
        def _(p):                                                                  # (n - 7)(n^2 + 10 n + 72) = 0, and n^2 + 10 n + 72 = (n + 5)^2 + 47 > 0
            p.intro('n', 'h')
            p.have('e', p.T('n', lambda n: Eq((n - 7) * (n * n + 10 * n + 72), v.zero)), lambda q: q.linear_combination((1, 'h')))
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('n', lambda n: n - 7), p.T('n', lambda n: n * n + 10 * n + 72), 'e'))
            p.linarith()
            p.have('s', ('Z.sq_nonneg', p.T('n', lambda n: n + 5)))
            p.exfalso(); p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ functional equations and sequences (the function is a variable)
_W = range(-8, 9)                                         # the window the quantifiers of these specs run over (as in semantic.py)


def _fe1(x): return 2 - 3 * x


_F_FE1 = [_fe1, lambda x: 0, lambda x: x, lambda x: 3 * x, lambda x: 1 - x, lambda x: x * x, lambda x: -x,
          lambda x: _fe1(x) + (1 if x == 3 else 0), lambda x: _fe1(x) + (1 if x == -2 else 0), lambda x: _fe1(x) + (1 if x == 5 else 0), lambda x: 5 if x == 2 else _fe1(x)]


@problem('amc_fe_01', text='A function f on the integers satisfies f(x) + 2 f(1 - x) = 3x for every integer x. What is f(2)?', topics=('functional-equation', 'algebra'), level=2, kind='answer', answer='f(2) = -4',
         spec=spec(('fn',), [lambda f: all(f(x) + 2 * f(1 - x) == 3 * x for x in _W)], EQ(lambda f: f(2), -4), hatoms=False, fns=_F_FE1))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_fe_01', [fZ], [lambda f: v.forall(lambda x: Eq(f(x) + 2 * f(lit(1) - x), 3 * x))], lambda f: Eq(f(lit(2)), -lit(4)))
        def _(p):                                                                  # x = 2:  f(2) + 2 f(-1) = 6;  x = -1:  f(-1) + 2 f(2) = -3
            p.intro('f', 'h')
            p.have('a', ('h', lit(2))); p.have('b', ('h', -lit(1)))
            p.linarith()


def _f2(x): return x + 1


_F_FE2 = [_f2, lambda x: x + 2, lambda x: x, lambda x: 2 * x, lambda x: 0, lambda x: x + 1 if x != 3 else 100, lambda x: x + 1 if x != 0 else 7, lambda x: x + 1 if x != -4 else 7,
          lambda x: x + 3, lambda x: x - 1, lambda x: x + 1 if x != 6 else 7, lambda x: x + 1 if x != 8 else 7]


@problem('amc_fe_02', text='A function f on the integers satisfies f(f(x)) = x + 2 for every integer x. Show that f(x + 2) = f(x) + 2 for every integer x.', topics=('functional-equation',), level=2,
         spec=spec(('fn',), [lambda f: all(f(f(x)) == x + 2 for x in _W)], lambda f: all(f(x + 2) == f(x) + 2 for x in _W), atoms=[(lambda f, x: f(x + 2), lambda f, x: f(x) + 2)], hatoms=False, fns=_F_FE2))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_fe_02', [fZ], [lambda f: v.forall(lambda x: Eq(f(f(x)), x + 2))], lambda f: v.forall(lambda x: Eq(f(x + 2), f(x) + 2)))
        def _(p):                                                                  # f(f(f(x))) = f(x) + 2, and f(f(x)) = x + 2
            p.intro('f', 'h', 'x')
            p.have('a', ('h', p.T('f x', lambda f, x: f(x))))
            p.have('b', ('h', 'x'))
            p.rw_at('a', 'b')
            p.exact('a')


def _mul2(x):                                              # completely multiplicative, f(2) = 3: 3 to the power of the exponent of 2
    if x == 0: return 0
    x = abs(x); e = 0
    while x % 2 == 0: x //= 2; e += 1
    return 3 ** e


_F_FE3 = [_mul2, lambda x: x, lambda x: x * x, lambda x: 1, lambda x: 0, lambda x: 3 * x // 2 if x % 2 == 0 else x, lambda x: _mul2(x) + (1 if x == 8 else 0), lambda x: _mul2(x) + (1 if x == 4 else 0),
          lambda x: _mul2(x) + (1 if x == 2 else 0), lambda x: _mul2(x) if x != 3 else 5, lambda x: abs(x) ** 2 if x != 8 else 27, lambda x: 3 ** (abs(x) // 2) if x else 0]


@problem('amc_fe_03', text='A function f on the integers satisfies f(xy) = f(x) f(y) for all integers x and y, and f(2) = 3. What is f(8)?', topics=('functional-equation',), level=2, kind='answer', answer='f(8) = 27',
         spec=spec(('fn',), [lambda f: all(f(x * y) == f(x) * f(y) for x in _W for y in _W), EQ(lambda f: f(2), 3)], EQ(lambda f: f(8), 27), hatoms=False, fns=_F_FE3))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_fe_03', [fZ], [lambda f: v.forall(lambda x: v.forall(lambda y: Eq(f(x * y), f(x) * f(y)))), lambda f: Eq(f(lit(2)), lit(3))], lambda f: Eq(f(lit(8)), lit(27)))
        def _(p):                                                                  # f(4) = f(2)^2 = 9, f(8) = f(2) f(4) = 27
            p.intro('f', 'h', 'h2')
            p.have('a', ('h', lit(2), lit(2))); p.rw_at('a', 'h2')
            p.have('e', p.T('f', lambda f: Eq(f(lit(4)), lit(9))), lambda q: q.linarith())
            p.have('b', ('h', lit(2), lit(4))); p.rw_at('b', 'h2'); p.rw_at('b', 'e')
            p.linarith()


_F_FE4 = [lambda x: 3 * x, lambda x: x, lambda x: 0, lambda x: x * x, lambda x: x + 1, lambda x: 3 * x if x != 0 else 7, lambda x: 3 * x if x != -1 else 100, lambda x: 3 * x if x != 4 else 1,
          lambda x: 2 * x, lambda x: -x, lambda x: 3 * x + (1 if x == 3 else 0), lambda x: 3 * x if x != 2 else 5, lambda x: 3 * x if x != 1 else 4, lambda x: 3 * x if x != 8 else 25]


@problem('amc_fe_04', text='A function f on the integers satisfies f(x + y) = f(x) + f(y) for all integers x and y. Show that f(n) = n f(1) for every integer n >= 0.', topics=('functional-equation', 'induction'), level=3,
         spec=spec(('fn',), [lambda f: all(f(x + y) == f(x) + f(y) for x in _W for y in _W)], lambda f: all(f(n) == n * f(1) for n in _W if n >= 0),
                   atoms=[(lambda f, n: 0, lambda f, n: n), (lambda f, n: f(n), lambda f, n: n * f(1))], hatoms=False, fns=_F_FE4))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_fe_04', [fZ], [lambda f: v.forall(lambda x: v.forall(lambda y: Eq(f(x + y), f(x) + f(y))))], lambda f: v.forall(lambda n: v.nonneg(n) >> Eq(f(n), n * f(lit(1)))))
        def _(p):
            p.intro('f', 'h', 'n', 'hn'); p.induction_on('Z.induction_from', 'n', extra=(lit(0),))
            p.have('z', ('h', lit(0), lit(0))); p.linarith()                      # f(0) = 2 f(0)
            p.intro('t', 'ht', 'ih')
            p.have('s', ('h', 't', lit(1)))
            p.linear_combination((1, 's'), (1, 'ih'))
            p.assumption()


_F_FE5 = [lambda x: x * x, lambda x: x * x + x, lambda x: 0, lambda x: x, lambda x: x * x + 2 * x, lambda x: x * x if x != 4 else 17, lambda x: x * x if x != 0 else 1, lambda x: x * x if x != -3 else 0,
          lambda x: x * x if x != 1 else 2, lambda x: x * x if x != 8 else 7, lambda x: 2 * x * x, lambda x: x * x if x != 2 else 5, lambda x: x * x if x != 3 else 5]


@problem('amc_fe_05', text='A function g on the integers satisfies g(x + y) = g(x) + g(y) + 2xy for all integers x and y, and g(1) = 1. Show that g(n) = n^2 for every integer n >= 0.', topics=('functional-equation', 'induction'), level=3,
         spec=spec(('fn',), [lambda g: all(g(x + y) == g(x) + g(y) + 2 * x * y for x in _W for y in _W), EQ(lambda g: g(1), 1)], lambda g: all(g(n) == n * n for n in _W if n >= 0),
                   atoms=[(lambda g, n: 0, lambda g, n: n), (lambda g, n: g(n), lambda g, n: n * n)], hatoms=False, fns=_F_FE5))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_fe_05', [fZ], [lambda g: v.forall(lambda x: v.forall(lambda y: Eq(g(x + y), g(x) + g(y) + 2 * x * y))), lambda g: Eq(g(lit(1)), lit(1))],
                   lambda g: v.forall(lambda n: v.nonneg(n) >> Eq(g(n), n * n)))
        def _(p):
            p.intro('g', 'h', 'h1', 'n', 'hn'); p.induction_on('Z.induction_from', 'n', extra=(lit(0),))
            p.have('z', ('h', lit(0), lit(0))); p.linarith()                      # g(0) = 2 g(0)
            p.intro('t', 'ht', 'ih')
            p.have('s', ('h', 't', lit(1)))
            p.linear_combination((1, 's'), (1, 'ih'), (1, 'h1'))
            p.assumption()


def _arith(n): return 5 * n - 2


_F_SEQ1 = [_arith, lambda n: 5 * n - 2 if n != 0 else 100, lambda n: 5 * n - 2 if n >= 1 else -7, lambda n: 100 + 5 * (n - 2) if n >= 2 else (3 if n == 1 else 0), lambda n: 4 * n - 1, lambda n: 5 * n - 1,
           lambda n: 5 * n - 2 if n != 20 else 99, lambda n: 5 * n - 2 if n != 19 else 7, lambda n: 5 * n - 2 if n != 5 else 0, lambda n: 0, lambda n: n, lambda n: 5 * n - 2 if n != 8 else 3, lambda n: 5 * n - 2 if n != -1 else 1]


@problem('amc_seq_01', text='A sequence a_1, a_2, a_3, ... of integers has a_1 = 3 and a_(n+1) = a_n + 5 for every n >= 1. What is a_20?', topics=('sequence', 'induction', 'arithmetic'), level=2, kind='answer', answer='a_20 = 98',
         spec=spec(('fn',), [EQ(lambda a: a(1), 3), lambda a: all(a(n + 1) == a(n) + 5 for n in _W if n >= 1)], EQ(lambda a: a(20), 98), hatoms=False, fns=_F_SEQ1))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_seq_01', [fZ], [lambda a: Eq(a(lit(1)), lit(3)), lambda a: v.forall(lambda n: v.le(lit(1), n) >> Eq(a(n + 1), a(n) + 5))], lambda a: Eq(a(lit(20)), lit(98)))
        def _(p):
            p.intro('a', 'h1', 'h')

            def lem(q):                                                            # a_n = 5 n - 2 for n >= 1
                q.intro('n', 'hn'); q.induction_on('Z.induction_from', 'n', extra=(lit(1),))
                q.linarith()
                q.intro('t', 'ht', 'ih')
                q.have('r', q.T('a t', lambda a, t: Eq(a(t + 1), a(t) + 5)), lambda r: (r.apply('h', r.h('t')), r.linarith()))
                q.linarith()
                q.assumption()
            p.have('L', p.T('a', lambda a: v.forall(lambda n: v.le(lit(1), n) >> Eq(a(n), 5 * n - 2))), lem)
            p.have('o', p.T('', lambda: v.le(lit(1), lit(20))), lambda q: q.linarith())
            p.have('l20', ('L', lit(20), 'o'))
            p.linarith()


_F_SEQ2 = [lambda n: 2 ** n - 1 if n >= 0 else 0, lambda n: 2 ** n - 1 if n >= 1 else 5, lambda n: 2 ** n - 1 if n >= 0 else -1, lambda n: 2 ** n - 1 if n >= 0 else (-1 if n == -1 else 5),
           lambda n: 2 ** n if n >= 0 else 0, lambda n: 0, lambda n: 3 ** n - 1 if n >= 0 else 0, lambda n: 2 ** n - 1 if n != 10 else 1022, lambda n: 2 ** n - 1 if n != 9 else 510, lambda n: 2 ** n - 1 if n not in (3, -2) else 0,
           lambda n: 2 ** n - 1 if 0 <= n != 4 else 0, lambda n: 2 ** n - 1 if 0 <= n != 7 else 0, lambda n: 2 ** n - 1 if n != 8 else 0, lambda n: (1 if n >= 1 else 0) if n else 0]


@problem('amc_seq_02', text='A sequence of integers a_0, a_1, a_2, ... has a_0 = 0 and a_(n+1) = 2 a_n + 1 for every n >= 0. What is a_10?', topics=('sequence', 'arithmetic'), level=2, kind='answer', answer='a_10 = 1023',
         spec=spec(('fn',), [EQ(lambda a: a(0), 0), lambda a: all(a(n + 1) == 2 * a(n) + 1 for n in _W if n >= 0)], EQ(lambda a: a(10), 1023), hatoms=False, fns=_F_SEQ2))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_seq_02', [fZ], [lambda a: Eq(a(lit(0)), lit(0)), lambda a: v.forall(lambda n: v.nonneg(n) >> Eq(a(n + 1), 2 * a(n) + 1))], lambda a: Eq(a(lit(10)), lit(1023)))
        def _(p):
            p.intro('a', 'h0', 'h')
            for k in range(10):
                p.have('c%d' % k, p.T('a', lambda a, k=k: Eq(a(lit(k) + 1), 2 * a(lit(k)) + 1)), lambda q, k=k: (q.apply('h', lit(k)), q.linarith()))
            p.linarith()


def _tri(n): return n * (n - 1) // 2 + 1


_F_SEQ3 = [_tri, lambda n: _tri(n) if n >= 1 else 5, lambda n: _tri(n) if n != 0 else 9, lambda n: _tri(n) if n >= 2 else (1 if n == 1 else 0), lambda n: _tri(n) if n != 10 else 47, lambda n: _tri(n) if n != 9 else 0,
           lambda n: n, lambda n: n * n, lambda n: 0, lambda n: _tri(n) + 1, lambda n: _tri(n) if n != 4 else 0, lambda n: _tri(n) if n != 1 else 2, lambda n: 2 * n - 1, lambda n: _tri(n) if n != 5 else 3]


@problem('amc_seq_03', text='A sequence of integers has a_1 = 1 and a_(n+1) = a_n + n for every n >= 1. What is a_10?', topics=('sequence', 'induction', 'sum'), level=3, kind='answer', answer='a_10 = 46',
         spec=spec(('fn',), [EQ(lambda a: a(1), 1), lambda a: all(a(n + 1) == a(n) + n for n in _W if n >= 1)], EQ(lambda a: a(10), 46), hatoms=False, fns=_F_SEQ3))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_seq_03', [fZ], [lambda a: Eq(a(lit(1)), lit(1)), lambda a: v.forall(lambda n: v.le(lit(1), n) >> Eq(a(n + 1), a(n) + n))], lambda a: Eq(a(lit(10)), lit(46)))
        def _(p):
            p.intro('a', 'h1', 'h')

            def lem(q):                                                            # 2 a_n = n^2 - n + 2 for n >= 1
                q.intro('n', 'hn'); q.induction_on('Z.induction_from', 'n', extra=(lit(1),))
                q.linarith()
                q.intro('t', 'ht', 'ih')
                q.have('r', q.T('a t', lambda a, t: Eq(a(t + 1), a(t) + t)), lambda r: (r.apply('h', r.h('t')), r.linarith()))
                q.linear_combination((2, 'r'), (1, 'ih'))
                q.assumption()
            p.have('L', p.T('a', lambda a: v.forall(lambda n: v.le(lit(1), n) >> Eq(2 * a(n), n * n - n + 2))), lem)
            p.have('o', p.T('', lambda: v.le(lit(1), lit(10))), lambda q: q.linarith())
            p.have('l10', ('L', lit(10), 'o'))
            p.linarith()


def _fib(n):                                                # Fibonacci numbers, continued to the negative integers by the recurrence
    if n < 0: return (-1) ** (-n + 1) * _fib(-n)
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a


_F_SEQ4 = [_fib, lambda n: n + 5 if n >= 1 else 4, lambda n: n + 2 if n >= 2 else (2 if n == 1 else 9), lambda n: 1, lambda n: 2 ** n if n >= 0 else 1, lambda n: n, lambda n: _fib(n) if n != 5 else 6, lambda n: _fib(n) if n != 6 else 9,
           lambda n: _fib(n) if n != 3 else 0, lambda n: 1 if n in (1, 2) else 3 * _fib(n), lambda n: _fib(n) if n >= 1 else 2 * _fib(n), lambda n: _fib(n) if n >= 2 else (1 if n == 1 else 0),
           lambda n: 2 if n == 1 else (_fib(n) if n != 2 else 1), lambda n: _fib(n) if n != 0 else 4, lambda n: _fib(n) if n != -1 else 4, lambda n: _fib(n) if n != 7 else 12, lambda n: _fib(n) if n != 8 else 21, lambda n: _fib(n) if n != 2 else 2]


@problem('amc_seq_04', text='The Fibonacci-type sequence a_1 = a_2 = 1, a_(n+2) = a_(n+1) + a_n (n >= 1) has the property that consecutive terms are coprime. Show that gcd(a_n, a_(n+1)) = 1 for every n >= 1.', topics=('sequence', 'induction', 'gcd'), level=4,
         spec=spec(('fn',), [EQ(lambda a: a(1), 1), EQ(lambda a: a(2), 1), lambda a: all(a(n + 2) == a(n + 1) + a(n) for n in _W if n >= 1)],
                   lambda a: all(math.gcd(a(n), a(n + 1)) == 1 for n in _W if n >= 1), atoms=[(lambda a, n: 1, lambda a, n: n), (lambda a, n: a(n), lambda a, n: a(n + 1), lambda a, n: 1)], hatoms=False, fns=_F_SEQ4))
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z); G = w['Z.is_gcd']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'amc_seq_04', [fZ], [lambda a: Eq(a(lit(1)), lit(1)), lambda a: Eq(a(lit(2)), lit(1)), lambda a: v.forall(lambda n: v.le(lit(1), n) >> Eq(a(n + 2), a(n + 1) + a(n)))],
                   lambda a: v.forall(lambda n: v.le(lit(1), n) >> G(a(n), a(n + 1), v.one)))
        def _(p):                                                                  # a_n x + a_(n+1) y = 1  gives  a_(n+1) (y - x) + a_(n+2) x = 1
            p.intro('a', 'h1', 'h2', 'h', 'n', 'hn'); p.induction_on('Z.induction_from', 'n', extra=(lit(1),))
            p.apply('Z.is_gcd_one_of_coprime'); p.exists(lit(1)); p.exists(lit(0)); p.linear_combination((1, 'h1'))
            p.intro('t', 'ht', 'ih')
            p.obtain(['x', 'hx'], ('Z.coprime_of_is_gcd_one', p.T('a t', lambda a, t: a(t)), p.T('a t', lambda a, t: a(t + 1)), 'ih')); p.cases('hx', names=['y', 'hy'])
            p.have('r0', ('h', 't', 'ht'))
            p.have('r', p.T('a t', lambda a, t: Eq(a(t + 1 + 1), a(t + 1) + a(t))), lambda q: q.linarith())
            p.apply('Z.is_gcd_one_of_coprime'); p.exists(p.T('x y', lambda x, y: y - x)); p.exists('x')
            p.linear_combination((1, 'hy'), (p.h('x'), 'r'))
            p.assumption()
