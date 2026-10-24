"""bank_aime2.py -- more AIME problems (paraphrased, as bank_amc.py), among them the first problem whose reference proof is not in the bank (`statement`: the formal statement only).

Provenance as in bank_amc.py: `origin='contest'` the contest's problem as recalled and paraphrased (its published answer agrees with the formal statement: `verified='answer-checked'`); `adapted` a contest problem
changed to fit the integer vocabulary (a logarithm replaced by the product it says, a real unknown by an integer one, the decimal expansion by the fraction it denotes), the change is in each problem's notes.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import math
from semantic import Spec
from bank import BANK
from bank_spec import spec, EQ, LT, LE, DVD, COP, GCD, PRIME, V as X
from bank_amc import AND, OR, sq
from ext_lib import destruct
import bank_kit as K

NAMES = []


def problem(name, **kw):
    NAMES.append(name)
    return BANK.problem(name, **kw)


def statement(name, **kw):
    NAMES.append(name)
    return BANK.statement(name, **kw)


def L(v, n):
    """the integer numeral n (negative ones too)"""
    return v.lit(n) if n >= 0 else -v.lit(-n)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 1986
@problem('aime1986_p10', text='A magician asks a volunteer to think of a three-digit number abc (digits a, b, c; a is not 0), to form the five numbers acb, bca, bac, cab and cba and to tell her their sum. The sum is 3194. Find abc.',
         source='AIME 1986 P10', year=1986, topics=('number-theory', 'digits', 'diophantine'), level=3, kind='answer', answer='abc = 358', origin='contest', verified='answer-checked',
         spec=spec(('Z', 'Z', 'Z'), [LT(0, X(0)), LE(X(0), 9), LE(0, X(1)), LE(X(1), 9), LE(0, X(2)), LE(X(2), 9),
                                     EQ(lambda a, b, c: (100 * a + 10 * c + b) + (100 * b + 10 * c + a) + (100 * b + 10 * a + c) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a), 3194)],
                   [EQ(lambda a, b, c: 100 * a + 10 * b + c, 358)], domains=(range(-1, 11), range(-1, 11), range(-1, 11)), points=3000),
         notes='the five numbers add up to 222 (a + b + c) - abc, so abc + 3194 = 222 s for the digit sum s; 15 <= s <= 18 leaves four candidates, and only s = 16 (abc = 358) has digits adding up to s')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        perms = lambda a, b_, c: (100 * a + 10 * c + b_) + (100 * b_ + 10 * c + a) + (100 * b_ + 10 * a + c) + (100 * c + 10 * a + b_) + (100 * c + 10 * b_ + a)
        @K.theorem(w, 'aime1986_p10', [v.Z, v.Z, v.Z], [lambda a, b_, c: v.pos(a), lambda a, b_, c: v.le(a, lit(9)), lambda a, b_, c: v.nonneg(b_), lambda a, b_, c: v.le(b_, lit(9)),
                                                         lambda a, b_, c: v.nonneg(c), lambda a, b_, c: v.le(c, lit(9)), lambda a, b_, c: Eq(perms(a, b_, c), lit(3194))],
                   lambda a, b_, c: Eq(100 * a + 10 * b_ + c, lit(358)))
        def _(p):
            p.intro('a', 'b', 'c', 'ha', 'ha9', 'hb', 'hb9', 'hc', 'hc9', 'h')
            K.let(p, 's', 'a b c', lambda a, b_, c: a + b_ + c)
            p.have('e', p.T('a b c s', lambda a, b_, c, s: Eq(222 * s, 3194 + (100 * a + 10 * b_ + c))), lambda q: q.linear_combination((1, 'h'), (222, 's_def')))
            p.have('s_lo', p.T('s', lambda s: v.le(lit(15), s)), lambda q: q.linarith())
            p.have('s_hi', p.T('s', lambda s: v.le(s, lit(18))), lambda q: q.linarith())
            p.interval_cases('s', 15, 18, name='hs')
            for sv in range(15, 19):
                nv = 222 * sv - 3194                                                   # the number abc, if the digits add up to sv
                a0 = nv // 100
                p.have('ea', p.T('a', lambda a, a0=a0: Eq(a, lit(a0))), lambda q: q.linarith())
                r, t = nv - 100 * a0, sv - a0                                           # 10 b + c = r and b + c = t, so 9 b = r - t
                p.have('e1', p.T('b c', lambda b_, c, r=r: Eq(10 * b_ + c, lit(r))), lambda q: q.linarith())
                p.have('e2', p.T('b c', lambda b_, c, t=t: Eq(b_ + c, lit(t))), lambda q: q.linarith())
                if r - t < 0: p.exfalso(); p.linarith(); continue
                if (r - t) % 9 != 0: K.no_multiple(p, 'b', 9, r - t); continue
                b0 = (r - t) // 9; c0 = t - b0
                if not (0 <= b0 <= 9 and 0 <= c0 <= 9): p.exfalso(); p.linarith(); continue
                assert 100 * a0 + 10 * b0 + c0 == 358
                p.have('eb', p.T('b', lambda b_, b0=b0: Eq(b_, lit(b0))), lambda q: q.linarith())
                p.have('ec', p.T('c', lambda c, c0=c0: Eq(c, lit(c0))), lambda q: q.linarith())
                p.linarith()


@problem('aime1986_p1', text='The positive real number x satisfies  x^(1/4) = 12 / (7 - x^(1/4)).  Find the sum of all such x. (Here: the fourth root is an integer y, so y (7 - y) = 12 and x = y^4.)', source='AIME 1986 P1', year=1986,
         topics=('algebra', 'number-theory'), level=2, kind='answer', answer='x = 81 or x = 256 (sum 337)', origin='adapted', verified='answer-checked',
         spec=Spec(('Z',), (), (lambda x: any(y ** 4 == x and y * (7 - y) == 12 for y in range(0, 12)), lambda x: x in (81, 256)), windows={'z': 12, 'ex': 12},
                   domains=([-1, 0, 1, 16, 80, 81, 82, 255, 256, 257, 625],), hatoms=[],
                   atoms=[(lambda x, y: 0, lambda x, y: y), (lambda x, y: y ** 4, lambda x, y: x), (lambda x, y: y * (7 - y), lambda x, y: 12), (lambda x: x, lambda x: 81), (lambda x: x, lambda x: 256)]),
         notes='adapted: y = x^(1/4) is taken to be an integer; the equation y^2 - 7y + 12 = 0 has only the integer roots 3 and 4, so no solution is lost, and x = 81 or 256 (sum 337)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1986_p1', [v.Z], [], lambda x: v.Iff(v.exists(lambda y: v.nonneg(y) & (Eq(y ** 4, x) & Eq(y * (lit(7) - y), lit(12)))), OR(Eq(x, lit(81)), Eq(x, lit(256)))))
        def _(p):
            p.intro('x'); p.split()
            p.intro('h'); destruct(p, 'h', ['y', 'hy0', 'h4', 'h12'])
            p.have('hy6', p.T('y', lambda y: v.le(y, lit(6))), lambda q: (q.obtain(['c', 'c'], ('Z.le_or_lt', 'y', lit(6))), q.exact('c'), q.exfalso(), q.nlinarith()))
            p.interval_cases('y', 0, 6, name='hy')
            for yv in range(0, 7):
                p.rw_at('h12', 'hy'); p.rw_at('h4', 'hy')
                if yv * (7 - yv) != 12: p.exfalso(); p.linarith(); continue
                if yv == 3: p.left(); p.linarith()
                else: assert yv == 4; p.right(); p.linarith()
            p.intro('h'); p.cases('h', names=['h1', 'h2'])
            p.exists(lit(3)); p.split(); p.linarith(); p.split(); p.linarith(); p.linarith()
            p.exists(lit(4)); p.split(); p.linarith(); p.split(); p.linarith(); p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 1989, 1990, 1991
@problem('aime1989_p3', text='The positive integer n and the digit d satisfy  n / 810 = 0.d25 d25 d25 ...  (the block d25 repeats for ever). Find n. (Here: the repeating decimal is the fraction (100 d + 25) / 999.)', source='AIME 1989 P3',
         year=1989, topics=('number-theory', 'digits', 'diophantine'), level=2, kind='answer', answer='n = 750 (d = 9)', origin='adapted', verified='answer-checked',
         spec=spec(('Z', 'Z'), [LT(0, X(0)), LE(0, X(1)), LE(X(1), 9), EQ(lambda n, d: 999 * n, lambda n, d: 810 * (100 * d + 25))], [EQ(X(0), 750)],
                   domains=([-1, 0, 1, 2, 20, 21, 749, 750, 751], range(-1, 11))),
         notes='adapted: the repeating decimal 0.d25d25... is the fraction (100 d + 25) / 999, so the statement is 999 n = 810 (100 d + 25); mod 37 only d = 9 works')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1989_p3', [v.Z, v.Z], [lambda n, d: v.pos(n), lambda n, d: v.nonneg(d), lambda n, d: v.le(d, lit(9)), lambda n, d: Eq(999 * n, 810 * (100 * d + 25))], lambda n, d: Eq(n, lit(750)))
        def _(p):
            p.intro('n', 'd', 'hn', 'hd0', 'hd9', 'h')
            p.interval_cases('d', 0, 9, name='hd')
            for dv in range(0, 10):
                p.rw_at('h', 'hd')
                c = 810 * (100 * dv + 25)
                if c % 999 != 0: K.no_multiple(p, 'n', 999, c)
                else: assert c // 999 == 750; p.linarith()


@problem('aime1990_p15', text='The real numbers a, b, x, y satisfy ax + by = 3, ax^2 + by^2 = 7, ax^3 + by^3 = 16 and ax^4 + by^4 = 42. Find ax^5 + by^5. (Here the power sums s_n are given with the recurrence they obey, s_(n+2) = p s_(n+1) - q s_n, p = x + y, q = xy.)',
         source='AIME 1990 P15', year=1990, topics=('algebra', 'sequence'), level=2, kind='answer', answer='s_5 = 20 (p = -14, q = -38)', origin='adapted', verified='answer-checked',
         spec=spec(('Z',) * 7, [EQ(X(2), 3), EQ(X(3), 7), EQ(X(4), 16), EQ(X(5), 42), EQ(X(4), lambda p, q, s1, s2, s3, s4, s5: p * s2 - q * s1), EQ(X(5), lambda p, q, s1, s2, s3, s4, s5: p * s3 - q * s2),
                                  EQ(X(6), lambda p, q, s1, s2, s3, s4, s5: p * s4 - q * s3)], [EQ(X(6), 20)],
                   domains=([-14, 0, 1], [-38, 0, 1], [3, 0], [7, 0], [16, 0], [42, 0], [20, 19, 21, 0]), points=3000),
         notes='adapted: the contest has real a, b, x, y (no integers satisfy its equations); the formal statement is the recurrence the power sums obey. 16 = 7p - 3q and 42 = 16p - 7q give p = -14, q = -38 and s_5 = 42 p - 16 q = 20')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1990_p15', [v.Z] * 7, [lambda p_, q, s1, s2, s3, s4, s5: Eq(s1, lit(3)), lambda p_, q, s1, s2, s3, s4, s5: Eq(s2, lit(7)), lambda p_, q, s1, s2, s3, s4, s5: Eq(s3, lit(16)),
                                                  lambda p_, q, s1, s2, s3, s4, s5: Eq(s4, lit(42)), lambda p_, q, s1, s2, s3, s4, s5: Eq(s3, p_ * s2 - q * s1), lambda p_, q, s1, s2, s3, s4, s5: Eq(s4, p_ * s3 - q * s2),
                                                  lambda p_, q, s1, s2, s3, s4, s5: Eq(s5, p_ * s4 - q * s3)], lambda p_, q, s1, s2, s3, s4, s5: Eq(s5, lit(20)))
        def _(p):
            p.intro('p', 'q', 's1', 's2', 's3', 's4', 's5', 'h1', 'h2', 'h3', 'h4', 'r1', 'r2', 'r3')
            p.rw_at('r1', 'h1'); p.rw_at('r1', 'h2'); p.rw_at('r1', 'h3')
            p.rw_at('r2', 'h2'); p.rw_at('r2', 'h3'); p.rw_at('r2', 'h4')
            p.rw_at('r3', 'h3'); p.rw_at('r3', 'h4')
            p.have('hp', p.T('p', lambda p_: Eq(p_, -lit(14))), lambda q_: q_.linarith())
            p.have('hq', p.T('q', lambda q_: Eq(q_, -lit(38))), lambda q_: q_.linarith())
            p.linarith()


@problem('aime1991_p8', text='For how many real numbers a does the quadratic equation x^2 + a x + 6a = 0 have only integer roots? (Here: the integers a for which both roots are integers.)', source='AIME 1991 P8', year=1991,
         topics=('number-theory', 'diophantine', 'algebra'), level=3, kind='answer', answer='ten values: a = -25, -8, -3, -1, 0, 24, 25, 27, 32, 49', origin='adapted', verified='answer-checked',
         spec=Spec(('Z',), (), (lambda a: (lambda d: d >= 0 and sq(d))(a * a - 24 * a), lambda a: a in (-25, -8, -3, -1, 0, 24, 25, 27, 32, 49)), windows={'z': 50, 'ex': 45},
                   domains=([-26, -25, -24, -9, -8, -7, -4, -3, -2, -1, 0, 1, 23, 24, 25, 26, 27, 28, 31, 32, 33, 48, 49, 50],), hatoms=[],
                   atoms=[(lambda a, r, s: r + s, lambda a, r, s: -a), (lambda a, r, s: r * s, lambda a, r, s: 6 * a)] + [(lambda a, k=k: a, lambda a, k=k: k) for k in (-25, -8, -3, -1, 0, 24, 25, 27, 32, 49)]),
         notes='adapted: a is an integer (it is minus the sum of the roots); with roots r, s: r + s = -a and rs = 6a, so (r + 6)(s + 6) = 36, and each of the ten unordered factorisations of 36 (five with positive and five with negative factors) gives one value of a')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    vals = (-25, -8, -3, -1, 0, 24, 25, 27, 32, 49)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1991_p8', [v.Z], [], lambda a: v.Iff(v.exists(lambda r: v.exists(lambda s: Eq(r + s, -a) & Eq(r * s, 6 * a))), OR(*[Eq(a, L(v, k)) for k in vals])))
        def _(p):
            def pick(p, val):                                                           # the goal is the disjunction: choose the one that says a = val
                p.have('ea', p.T('a', lambda a, val=val: Eq(a, L(v, val))), lambda q: q.linarith())
                i = vals.index(val)
                for _ in range(i): p.right()
                if i < len(vals) - 1: p.left()
                p.exact('ea')
            p.intro('a'); p.split()
            p.intro('h'); destruct(p, 'h', ['r', 's', 'h1', 'h2'])
            K.let(p, 'u', 'r', lambda r: r + 6); K.let(p, 'vv', 's', lambda s: s + 6)
            p.have('e', p.T('u vv', lambda u, v_: Eq(u * v_, lit(36))), lambda q: q.linear_combination((q.h('vv'), 'u_def'), (q.T('r', lambda r: r + 6), 'vv_def'), (1, 'h2'), (6, 'h1')))
            p.obtain(['c', 'c'], ('Z.le_or_lt', 'u', lit(0)))
            # u <= 0: then vv < 0 and u < 0; the positive numbers -u, -vv multiply to 36
            p.have('nv', p.T('vv', lambda v_: v.lt(v_, lit(0))), lambda q: (q.obtain(['d', 'd'], ('Z.le_or_lt', lit(0), 'vv')), q.exfalso(), q.nlinarith(), q.exact('d')))
            p.have('nu', p.T('u', lambda u: v.lt(u, lit(0))), lambda q: (q.obtain(['d', 'd'], ('Z.le_or_lt', lit(0), 'u')), q.exfalso(), q.nlinarith(), q.exact('d')))
            K.let(p, 'u2', 'u', lambda u: -u); K.let(p, 'v2', 'vv', lambda v_: -v_)
            p.have('hu2', p.T('u2', v.pos), lambda q: q.linarith())
            p.have('hv2', p.T('v2', v.pos), lambda q: q.linarith())
            p.have('e2', p.T('u2 v2', lambda u, v_: Eq(u * v_, lit(36))), lambda q: q.linear_combination((q.h('v2'), 'u2_def'), (q.T('u', lambda u: -u), 'v2_def'), (1, 'e')))
            K.divisor_cases(p, 'u2', 'v2', 'e2', 36, lambda p, d, c: pick(p, 12 + d + c), xn='dx2', yn='dy2')
            # u > 0: then vv > 0
            p.have('pv', p.T('vv', v.pos), lambda q: (q.obtain(['d', 'd'], ('Z.le_or_lt', 'vv', lit(0))), q.exfalso(), q.nlinarith(), q.exact('d')))
            K.divisor_cases(p, 'u', 'vv', 'e', 36, lambda p, d, c: pick(p, 12 - d - c))
            # the other direction: for each of the ten values the roots
            p.intro('h')
            for i, val in enumerate(vals):
                if i < len(vals) - 1: p.cases('h', names=['h1', 'h'])
                r_, s_ = [(x, y) for x in range(-60, 61) for y in range(-60, 61) if x + y == -val and x * y == 6 * val][0]
                p.exists(L(v, r_)); p.exists(L(v, s_)); p.split(); p.linarith(); p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 2002, 2009 and a statement without a proof
@problem('aime2002ii_p3', text='The positive integers a < b < c form a geometric sequence, their product is 6^6 (the contest says: log_6 a + log_6 b + log_6 c = 6), and b - a is the square of an integer. Find a + b + c.',
         source='AIME 2002 II P3', year=2002, topics=('number-theory', 'diophantine', 'algebra'), level=3, kind='answer', answer='a + b + c = 111 (27, 36, 48)', origin='adapted', verified='answer-checked',
         spec=spec(('Z',) * 4, [LT(0, X(0)), LT(X(0), X(1)), LT(X(1), X(2)), EQ(lambda a, b, c, m: b * b, lambda a, b, c, m: a * c), EQ(lambda a, b, c, m: a * b * c, 6 ** 6), EQ(lambda a, b, c, m: b - a, lambda a, b, c, m: m * m)],
                   [EQ(lambda a, b, c, m: a + b + c, 111)], domains=([0, 1, 26, 27, 28, 36], [0, 1, 35, 36, 37], [1, 36, 47, 48, 49], range(-4, 5)), points=3000,
                   skip_mutants={'constant 6 -> -6 in ((Z.mul (Z.mul x0 x1) x2) = (Z.pow 6 6))': 'an even power: (-6)^6 = 6^6'}),
         notes='adapted: the logarithms say  a b c = 6^6; the middle term of the geometric sequence is b = 36 (b^3 = 6^6), a c = 36^2, and 36 - a is a square only for a = 27 among the divisors of 1296 below 36')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime2002ii_p3', [v.Z] * 4, [lambda a, b_, c, m: v.pos(a), lambda a, b_, c, m: v.lt(a, b_), lambda a, b_, c, m: v.lt(b_, c), lambda a, b_, c, m: Eq(b_ * b_, a * c),
                                                   lambda a, b_, c, m: Eq(a * b_ * c, lit(6) ** 6), lambda a, b_, c, m: Eq(b_ - a, m ** 2)], lambda a, b_, c, m: Eq(a + b_ + c, lit(111)))
        def _(p):
            p.intro('a', 'b', 'c', 'm', 'ha', 'hab', 'hbc', 'hsq', 'hprod', 'hm')
            p.have('h3', p.T('b', lambda b_: Eq(b_ ** 3, lit(36) ** 3)), lambda q: q.linear_combination((1, 'hprod'), (q.h('b'), 'hsq')))
            def cube(q):                                                               # b = 36 : the cube is strictly increasing
                q.have('hb0', q.T('b', v.nonneg), lambda r: r.linarith())
                q.obtain(['c1', 'c1'], ('Z.le_or_lt', 'b', lit(35)))
                q.have('m1', ('Z.pow_le_pow_left', 'b', lit(35), v.nlit(3), 'hb0', 'c1'))
                q.exfalso(); q.linarith()
                q.obtain(['d1', 'd1'], ('Z.le_or_lt', 'b', lit(36)))
                q.linarith()
                q.have('h37', q.T('b', lambda b_: v.le(lit(37), b_)), lambda r: r.linarith())
                q.have('z37', q.T('', lambda: v.nonneg(lit(37))), lambda r: r.linarith())
                q.have('m2', ('Z.pow_le_pow_left', lit(37), 'b', v.nlit(3), 'z37', 'h37'))
                q.exfalso(); q.linarith()
            p.have('hb', p.T('b', lambda b_: Eq(b_, lit(36))), cube)
            p.have('hc0', p.T('c', v.pos), lambda q: q.linarith())
            p.have('e', p.T('a c', lambda a_, c_: Eq(a_ * c_, lit(1296))), lambda q: (q.rw_at('hsq', 'hb'), q.linarith()))
            def pair(p, d, q_):
                if d >= 36: p.exfalso(); p.linarith(); return                          # a < b = 36
                if d == 27: p.linarith(); return
                k = math.isqrt(36 - d); assert k * k != 36 - d
                p.exfalso(); p.apply('not_square_between', p.h('m'), lit(k)); p.linarith(); p.linarith(); p.linarith()
            K.divisor_cases(p, 'a', 'c', 'e', 1296, pair)


def _geo(v, a, b, c):
    """a, b, c are the digits of a geometric number: a is not 0, the digits are different, b^2 = a c (as a conjunction, for the instances 1 2 4 and 9 6 4)"""
    lit = v.lit
    return AND(v.pos(a), v.le(a, lit(9)), v.nonneg(b), v.le(b, lit(9)), v.nonneg(c), v.le(c, lit(9)), v.Not(v.Eq(a, b)), v.Not(v.Eq(b, c)), v.Not(v.Eq(a, c)), v.Eq(b * b, a * c))


def _geo_atoms(a, b, c):
    z = lambda *t: 0; nine = lambda *t: 9
    return [(z, a), (a, nine), (z, b), (b, nine), (z, c), (c, nine), (a, b), (b, c), (a, c), (lambda *t: b(*t) * b(*t), lambda *t: a(*t) * c(*t))]


def _geo_ok(a, b, c): return 0 < a <= 9 and 0 <= b <= 9 and 0 <= c <= 9 and a != b and b != c and a != c and b * b == a * c


def _const(k): return lambda *t: k


_A, _B, _C = (lambda a, b, c: a), (lambda a, b, c: b), (lambda a, b, c: c)


@problem('aime2009i_p1', text='Call a three-digit number geometric if it has three different digits which, read from left to right, form a geometric sequence. Find the difference between the largest and the smallest geometric numbers. (Here: every geometric number lies between 124 and 964, and these two are geometric.)',
         source='AIME 2009 I P1', year=2009, topics=('number-theory', 'digits', 'case-analysis'), level=3, kind='answer', answer='964 - 124 = 840', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, c: a > 0, lambda a, b, c: a <= 9, lambda a, b, c: b >= 0, lambda a, b, c: b <= 9, lambda a, b, c: c >= 0, lambda a, b, c: c <= 9, lambda a, b, c: a != b, lambda a, b, c: b != c, lambda a, b, c: a != c,
                                       lambda a, b, c: b * b == a * c),
                   lambda a, b, c: 124 <= 100 * a + 10 * b + c <= 964 and _geo_ok(1, 2, 4) and _geo_ok(9, 6, 4), domains=(range(-1, 11), range(-1, 11), range(-1, 11)), points=2000,
                   hatoms=_geo_atoms(_A, _B, _C), atoms=[(_const(124), lambda a, b, c: 100 * a + 10 * b + c), (lambda a, b, c: 100 * a + 10 * b + c, _const(964))] + _geo_atoms(_const(1), _const(2), _const(4)) + _geo_atoms(_const(9), _const(6), _const(4)),
                   skip_mutants={'Z.mul -> Z.add in ((Z.mul 2 2) = (Z.mul 1 4))': '2 + 2 = 2 * 2'}),
         notes='formalised as: every geometric number lies between 124 and 964, and 124 = 1 2 4, 964 = 9 6 4 are geometric. For a = 1: b^2 = c leaves b = 2, 3; for a = 9: b^2 = 9 c leaves b = 3, 6')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        num = lambda a, b_, c: 100 * a + 10 * b_ + c
        hyps = [lambda a, b_, c: v.pos(a), lambda a, b_, c: v.le(a, lit(9)), lambda a, b_, c: v.nonneg(b_), lambda a, b_, c: v.le(b_, lit(9)), lambda a, b_, c: v.nonneg(c), lambda a, b_, c: v.le(c, lit(9)),
                lambda a, b_, c: v.Not(Eq(a, b_)), lambda a, b_, c: v.Not(Eq(b_, c)), lambda a, b_, c: v.Not(Eq(a, c)), lambda a, b_, c: Eq(b_ * b_, a * c)]
        @K.theorem(w, 'aime2009i_p1', [v.Z] * 3, hyps, lambda a, b_, c: AND(v.le(lit(124), num(a, b_, c)), v.le(num(a, b_, c), lit(964)), _geo(v, lit(1), lit(2), lit(4)), _geo(v, lit(9), lit(6), lit(4))))
        def _(p):
            p.intro('a', 'b', 'c', 'ha', 'ha9', 'hb', 'hb9', 'hc', 'hc9', 'hab', 'hbc', 'hac', 'hsq')
            p.split()
            # 124 <= abc: for a = 1 the middle digit b has b^2 = c
            p.obtain(['c1', 'c1'], ('Z.le_or_lt', 'a', lit(1)))
            p.have('ea', p.T('a', lambda a: Eq(a, lit(1))), lambda q: q.linarith())
            p.interval_cases('b', 0, 9, name='eb')
            for k in range(10):
                p.rw_at('hsq', 'eb'); p.rw_at('hsq', 'ea')
                if k == 0: p.exfalso(); p.apply('hbc'); p.linarith()
                elif k == 1: p.exfalso(); p.apply('hab'); p.linarith()
                elif k in (2, 3): p.linarith()
                else: p.exfalso(); p.linarith()
            p.linarith()
            p.split()
            # abc <= 964: for a <= 8 it is at most 899, for a = 9: b^2 = 9 c
            p.obtain(['c2', 'c2'], ('Z.le_or_lt', 'a', lit(8)))
            p.linarith()
            p.have('ea', p.T('a', lambda a: Eq(a, lit(9))), lambda q: q.linarith())
            p.interval_cases('b', 0, 9, name='eb')
            for k in range(10):
                p.rw_at('hsq', 'eb'); p.rw_at('hsq', 'ea')
                if k == 0: p.exfalso(); p.apply('hbc'); p.linarith()
                elif k == 9: p.exfalso(); p.apply('hab'); p.linarith()
                elif (k * k) % 9 != 0: K.no_multiple(p, 'c', 9, k * k)
                else: p.linarith()
            # the two geometric numbers
            p.split()
            for t in range(2):
                for i in range(10):
                    if i < 9: p.split()
                    if i in (6, 7, 8): p.intro('h0')
                    p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ statement only
@statement('aime1985_p7', text='The positive integers a, b, c, d satisfy a^5 = b^4, c^3 = d^2 and c - a = 19. Determine d - b.', source='AIME 1985 P7', year=1985, topics=('number-theory', 'diophantine'), level=3, kind='answer',
           answer='d - b = 757 (a = 81, b = 243, c = 100, d = 1000)', origin='contest', verified='answer-checked',
           spec=spec(('Z',) * 4, [LT(0, X(0)), LT(0, X(1)), LT(0, X(2)), LT(0, X(3)), EQ(lambda a, b, c, d: a ** 5, lambda a, b, c, d: b ** 4), EQ(lambda a, b, c, d: c ** 3, lambda a, b, c, d: d ** 2),
                                       EQ(lambda a, b, c, d: c - a, 19)], [EQ(lambda a, b, c, d: d - b, 757)],
                     domains=([0, 1, 80, 81, 82], [0, 1, 242, 243, 244], [0, 1, 99, 100, 101], [0, 1, 999, 1000, 1001])),
           notes='a = t^4, b = t^5, c = s^2, d = s^3 and (s - t^2)(s + t^2) = 19 give s = 10, t = 3. No reference proof: the step a^5 = b^4 implies a = t^4 needs unique factorisation, which the library does not have')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 4, [lambda a, b_, c, d: v.pos(a), lambda a, b_, c, d: v.pos(b_), lambda a, b_, c, d: v.pos(c), lambda a, b_, c, d: v.pos(d), lambda a, b_, c, d: Eq(a ** 5, b_ ** 4),
                                          lambda a, b_, c, d: Eq(c ** 3, d ** 2), lambda a, b_, c, d: Eq(c - a, lit(19))], lambda a, b_, c, d: Eq(d - b_, lit(757)))
