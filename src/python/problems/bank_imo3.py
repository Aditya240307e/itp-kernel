"""bank_imo3.py -- olympiad problems WITH a kernel-checked reference proof, for the IMO tier: classical lemmas of olympiad number theory and algebra (composite values of polynomials, runs of composite numbers,
Mersenne-type numbers, Vasc's inequality, ..), two real contest problems (IMO 1983 P6 for integer sides, IMO 1977 P6) and a functional equation.

Provenance as in bank_imo.py: `origin='contest'` the contest's problem as recalled and paraphrased; `adapted` a contest problem changed to fit the integer vocabulary (the change is in the notes); `folklore` a classical fact or exercise with
no contest behind it (so there is nothing to check a source label against: `verified='none'`).  The reference proofs are kernel-checked; the statements are checked against Python by the semantic check (semantic.py).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import math
from semantic import Spec
from bank import BANK
from bank_spec import spec, EQ, LT, LE, DVD, COP, GCD, PRIME, V as X, is_prime_
from bank_amc import AND, OR, sq
from ext_lib import destruct
import bank_kit as K

NAMES = []
_W = range(-8, 9)


def problem(name, **kw):
    NAMES.append(name)
    return BANK.problem(name, **kw)


def statement(name, **kw):
    NAMES.append(name)
    return BANK.statement(name, **kw)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ composite numbers
@problem('n5_n4_1_composite', text='Prove that n^5 + n^4 + 1 is not a prime for any integer n > 1.', source='folklore', topics=('number-theory', 'prime', 'identity'), level=3, origin='folklore',
         spec=spec(('Z',), [LT(1, X(0))], lambda n: not is_prime_(n ** 5 + n ** 4 + 1), atoms=[(lambda n: n ** 5 + n ** 4 + 1,)], domains=(range(-3, 12),)),
         notes='n^5 + n^4 + 1 = (n^2 + n + 1)(n^3 - n + 1), and both factors are larger than 1 once n > 1')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; P = w['Z.prime']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'n5_n4_1_composite', [v.Z], [lambda n: v.lt(lit(1), n)], lambda n: v.Not(P(n ** 5 + n ** 4 + 1)))
        def _(p):
            p.intro('n', 'hn')
            p.have('e', p.T('n', lambda n: Eq(n ** 5 + n ** 4 + 1, (n * n + n + 1) * (n ** 3 - n + 1))), lambda q: q.ring())
            p.have('h0', p.T('n', lambda n: v.pos(n)), lambda q: q.linarith())
            p.have('h1', p.T('n', lambda n: v.pos(n - 1)), lambda q: q.linarith())
            p.have('h2', p.T('n', lambda n: v.pos(n + 1)), lambda q: q.linarith())
            p.have('h3', ('Z.mul_pos', 'n', p.T('n', lambda n: n - 1), 'h0', 'h1'))
            p.have('h4', ('Z.mul_pos', p.T('n', lambda n: n * (n - 1)), p.T('n', lambda n: n + 1), 'h3', 'h2'))
            p.rw('e'); p.apply('not_prime_mul')
            p.nlinarith(); p.linarith()


@problem('composite_run', text='Prove that for every natural number n there are n consecutive composite numbers: there is an integer a > 0 such that none of a + 1, a + 2, ..., a + n is a prime.', source='folklore', origin='folklore',
         topics=('number-theory', 'prime', 'construction'), level=3,
         spec=spec(('N',), [], lambda n: any(a > 0 and all(not is_prime_(a + i) for i in range(1, n + 1)) for a in range(-3, 101)),
                   body=(('Z',), lambda n, a: a > 0 and all(not is_prime_(a + i) for i in range(1, n + 1))),
                   atoms=[(lambda n, a: 0, lambda n, a: a), (lambda n, a, i: 0, lambda n, a, i: i), (lambda n, a, i: i, lambda n, a, i: n), (lambda n, a, i: a + i,)],
                   domains=(range(0, 7),), windows={'ex': 100}),
         notes='take a = (n + 1)! + 1: for 1 <= i <= n the number a + i = (n + 1)! + (i + 1) is divisible by i + 1, which lies between 2 and n + 1')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; P = w['Z.prime']; zN = w['Z.of_N']; zfact = w['Z.fact']; nsucc = w['N.succ']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'composite_run', [v.N], [], lambda n: v.exists(lambda a: v.pos(a) & v.forall(lambda i: v.pos(i) >> (v.le(i, zN(n)) >> v.Not(P(a + i))))))
        def _(p):
            p.intro('n')
            p.have('fp', ('Z.fact_pos', p.T('n', lambda n: nsucc(n))))
            p.exists(p.T('n', lambda n: zfact(nsucc(n)) + 1))
            p.split(); p.linarith()
            p.intro('i', 'hi', 'hin')
            p.have('c1', p.T('i', lambda i: v.pos(i + 1)), lambda q: q.linarith())
            p.have('c2', p.T('n i', lambda n, i: v.le(i + 1, zN(nsucc(n)))), lambda q: q.linarith(('Z.of_N_succ', 'n')))
            p.obtain(['m', 'hm'], ('Z.dvd_fact', p.T('n', lambda n: nsucc(n)), p.T('i', lambda i: i + 1), 'c1', 'c2'))
            p.have('m0', p.T('m', lambda m: v.pos(m)), lambda q: q.nlinarith())
            p.have('e', p.T('n i m', lambda n, i, m: Eq(zfact(nsucc(n)) + 1 + i, (i + 1) * (m + 1))), lambda q: q.linear_combination((1, 'hm')))
            p.rw('e'); p.apply('not_prime_mul'); p.linarith(); p.linarith()


@problem('n2_3n_5_not_121', text='Prove that n^2 + 3n + 5 is never divisible by 121 (for an integer n).', source='folklore (olympiad longlist)', origin='folklore', topics=('number-theory', 'divisibility', 'modular', 'case-analysis'), level=3,
         spec=spec(('Z',), [], lambda n: (n * n + 3 * n + 5) % 121 != 0, atoms=[(lambda n: 121, lambda n: n * n + 3 * n + 5)], windows={'z': 12}),
         notes='11 divides n^2 + 3n + 5 only for n = 4 (mod 11), and then n^2 + 3n + 5 = 33 (mod 121)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'n2_3n_5_not_121', [v.Z], [], lambda n: v.Not(v.dvd(lit(121), n * n + 3 * n + 5)))
        def _(p):
            p.intro('n', 'h'); p.cases('h', names=['k', 'hk'])
            p.have('p11', p.T('', lambda: v.pos(lit(11))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(11), 'p11', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.interval_cases('r', 0, 10, name='hr')
            for rv in range(11):
                p.rw_at('hk', 'e'); p.rw_at('hk', 'hr')
                c = rv * rv + 3 * rv + 5
                if rv != 4:
                    K.let(p, 't', 'k q', lambda k, q, rv=rv: 11 * (k - q * q) - (2 * rv + 3) * q)
                    p.have('e11', p.T('t', lambda t, c=c: Eq(11 * t, lit(c))), lambda q_: q_.linear_combination((-1, 'hk'), (11, 't_def')))
                    K.no_multiple(p, 't', 11, c)
                else:
                    K.let(p, 'u', 'k q', lambda k, q: k - q * q - q)
                    p.have('e121', p.T('u', lambda u: Eq(121 * u, lit(33))), lambda q_: q_.linear_combination((-1, 'hk'), (121, 'u_def')))
                    K.no_multiple(p, 'u', 121, 33)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ Mersenne-type numbers
def _mersenne_lemma(v, w):
    """the lemma  forall x m, 2 < x -> not prime (x^(m+2) - 1)  as (type, script) -- to be used inside a `with v.ints()` block: x - 1 divides x^(m+2) - 1, with a cofactor larger than 1"""
    lit = v.lit; P = w['Z.prime']; nadd = w['N.add']; zpow = w['Z.pow']
    ty = v.F([v.Z, v.N, lambda x, m: v.lt(lit(2), x)], lambda x, m, h: v.Not(P(zpow(x, nadd(m, v.nlit(2))) - 1)))

    def script(q):
        q.intro('x', 'm', 'hx', 'hp')
        m2 = q.T('m', lambda m: nadd(m, v.nlit(2)))
        q.have('hx1', q.T('x', lambda x: v.le(lit(1), x)), lambda r: r.linarith())
        q.have('hx0', q.T('x', lambda x: v.lt(lit(1), x - 1)), lambda r: r.linarith())
        q.obtain(['k', 'hk'], ('Z.sub_dvd_pow_sub', 'x', v.one, m2))                     # x^(m+2) - 1^(m+2) = (x - 1) k
        q.have('o', ('Z.one_pow', m2)); q.rw_at('hk', 'o')
        q.have('pw', ('Z.pow_add', 'x', 'm', v.nlit(2)))                                   # x^(m+2) = x^m x^2
        q.have('p1', ('Z.one_le_pow', 'x', 'm', 'hx1'))
        q.have('p1b', q.T('x m', lambda x, m: v.nonneg(zpow(x, m) - 1)), lambda r: r.linarith())
        q.have('p2', ('Z.mul_nonneg', q.T('x m', lambda x, m: zpow(x, m) - 1), q.T('x', lambda x: x * x), 'p1b', ('Z.sq_nonneg', 'x')))
        q.have('hk1', q.T('k', lambda k: v.lt(lit(1), k)), lambda r: r.nlinarith())
        q.have('e', q.T('x m k', lambda x, m, k: v.Eq(zpow(x, nadd(m, v.nlit(2))) - 1, (x - 1) * k)), lambda r: r.linarith())
        q.rw_at('hp', 'e')
        q.exact(('not_prime_mul', q.T('x', lambda x: x - 1), 'k', 'hx0', 'hk1', 'hp'))
    return ty, script


@problem('mersenne_base_2', text='Let a > 1 and n >= 2 be integers. Prove that if a^n - 1 is a prime, then a = 2.  (Here n = m + 2 for a natural number m.)', source='folklore', origin='folklore',
         topics=('number-theory', 'prime', 'divisibility'), level=3,
         spec=spec(('Z', 'N'), [LT(1, X(0)), lambda a, m: is_prime_(a ** (m + 2) - 1)], EQ(X(0), 2), hatoms=[(lambda a, m: 1, lambda a, m: a), (lambda a, m: a ** (m + 2) - 1,)], domains=(range(-2, 10), range(0, 6))),
         notes='a - 1 divides a^n - 1 and, for a >= 3, 1 < a - 1 < a^n - 1')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; P = w['Z.prime']; nadd = w['N.add']; zpow = w['Z.pow']
    with v.ints():
        lit = v.lit
        ty, script = _mersenne_lemma(v, w)
        @K.theorem(w, 'mersenne_base_2', [v.Z, v.N], [lambda a, m: v.lt(lit(1), a), lambda a, m: P(zpow(a, nadd(m, v.nlit(2))) - 1)], lambda a, m: Eq(a, lit(2)))
        def _(p):
            p.intro('a', 'm', 'ha', 'hp')
            p.have('L', ty, script)
            p.obtain(['c', 'c'], ('Z.le_or_lt', 'a', lit(2)))
            p.linarith()
            p.exfalso(); p.exact(('L', 'a', 'm', 'c', 'hp'))


@problem('mersenne_composite_exponent', text='Let a > 1 be an integer and let the exponent n = (d + 2)(e + 2) be a product of two integers larger than 1. Prove that a^n - 1 is not a prime.', source='folklore', origin='folklore',
         topics=('number-theory', 'prime', 'divisibility'), level=3,
         spec=spec(('Z', 'N', 'N'), [LT(1, X(0))], lambda a, d, e: not is_prime_(a ** ((d + 2) * (e + 2)) - 1), atoms=[(lambda a, d, e: a ** ((d + 2) * (e + 2)) - 1,)], domains=(range(-2, 9), range(0, 4), range(0, 4))),
         notes='x = a^(d+2) satisfies x - 1 | x^(e+2) - 1 = a^n - 1, and 1 < x - 1 < a^n - 1')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; P = w['Z.prime']; nadd = w['N.add']; nmul = w['N.mul']; zpow = w['Z.pow']
    with v.ints():
        lit = v.lit
        ty, script = _mersenne_lemma(v, w)
        @K.theorem(w, 'mersenne_composite_exponent', [v.Z, v.N, v.N], [lambda a, d, e: v.lt(lit(1), a)], lambda a, d, e: v.Not(P(zpow(a, nmul(nadd(d, v.nlit(2)), nadd(e, v.nlit(2)))) - 1)))
        def _(p):
            p.intro('a', 'd', 'e', 'ha')
            p.have('L', ty, script)
            p.rw('Z.pow_mul')
            p.have('pw', ('Z.pow_add', 'a', 'd', v.nlit(2)))                                 # a^(d+2) = a^d a^2
            p.have('ha1', p.T('a', lambda a: v.le(lit(1), a)), lambda q: q.linarith())
            p.have('p1', ('Z.one_le_pow', 'a', 'd', 'ha1'))
            p.have('p1b', p.T('a d', lambda a, d: v.nonneg(zpow(a, d) - 1)), lambda q: q.linarith())
            p.have('p2', ('Z.mul_nonneg', p.T('a d', lambda a, d: zpow(a, d) - 1), p.T('a', lambda a: a * a), 'p1b', ('Z.sq_nonneg', 'a')))
            p.have('big', p.T('a d', lambda a, d: v.lt(lit(2), zpow(a, nadd(d, v.nlit(2))))), lambda q: q.nlinarith())
            p.exact(('L', p.T('a d', lambda a, d: zpow(a, nadd(d, v.nlit(2)))), 'e', 'big'))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ inequalities
R5 = range(-5, 6)


@problem('ineq_vasc', text="Vasc's inequality: for all integers a, b, c:  3 (a^3 b + b^3 c + c^3 a) <= (a^2 + b^2 + c^2)^2.", source='folklore (Vasile Cirtoaje)', origin='folklore', topics=('inequality', 'algebra', 'identity'), level=4,
         spec=spec(('Z', 'Z', 'Z'), [], LE(lambda a, b, c: 3 * (a ** 3 * b + b ** 3 * c + c ** 3 * a), lambda a, b, c: (a * a + b * b + c * c) ** 2), domains=(R5, R5, R5)),
         notes='twice the difference is the sum of the squares (a^2 - 2ab + bc - c^2 + ca)^2 + (b^2 - 2bc + ca - a^2 + ab)^2 + (c^2 - 2ca + ab - b^2 + bc)^2')
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'ineq_vasc', [v.Z, v.Z, v.Z], [], lambda a, b_, c: v.le(3 * (a ** 3 * b_ + b_ ** 3 * c + c ** 3 * a), (a * a + b_ * b_ + c * c) ** 2))
        def _(p):
            p.intro('a', 'b', 'c')
            p.nlinarith(('Z.sq_nonneg', p.T('a b c', lambda a, b_, c: a * a - 2 * a * b_ + b_ * c - c * c + c * a)),
                        ('Z.sq_nonneg', p.T('a b c', lambda a, b_, c: b_ * b_ - 2 * b_ * c + c * a - a * a + a * b_)),
                        ('Z.sq_nonneg', p.T('a b c', lambda a, b_, c: c * c - 2 * c * a + a * b_ - b_ * b_ + b_ * c)))


@problem('ineq_amgm_three', text='For all non-negative integers a, b, c:  27 abc <= (a + b + c)^3  (the inequality between the arithmetic and the geometric mean of three numbers).', source='folklore', origin='folklore',
         topics=('inequality', 'algebra'), level=3,
         spec=spec(('Z', 'Z', 'Z'), [LE(0, X(0)), LE(0, X(1)), LE(0, X(2))], LE(lambda a, b, c: 27 * a * b * c, lambda a, b, c: (a + b + c) ** 3), domains=(range(-3, 7), range(-3, 7), range(-3, 7))),
         notes='(a + b + c)^3 - 27 abc = (a + b + c) ((a - b)^2 + (b - c)^2 + (c - a)^2) / 2 + 3 (a (b - c)^2 + b (c - a)^2 + c (a - b)^2)')
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'ineq_amgm_three', [v.Z, v.Z, v.Z], [lambda a, b_, c: v.nonneg(a), lambda a, b_, c: v.nonneg(b_), lambda a, b_, c: v.nonneg(c)], lambda a, b_, c: v.le(27 * a * b_ * c, (a + b_ + c) ** 3))
        def _(p):
            p.intro('a', 'b', 'c', 'ha', 'hb', 'hc')
            p.have('s', p.T('a b c', lambda a, b_, c: v.nonneg(a + b_ + c)), lambda q: q.linarith())
            S = lambda a, b_, c: a + b_ + c
            for nm, d in (('t1', lambda a, b_, c: a - b_), ('t2', lambda a, b_, c: b_ - c), ('t3', lambda a, b_, c: c - a)):
                p.have(nm, ('Z.mul_nonneg', p.T('a b c', S), p.T('a b c', lambda a, b_, c, d=d: d(a, b_, c) * d(a, b_, c)), 's', ('Z.sq_nonneg', p.T('a b c', d))))
            for nm, x, d in (('u1', 'a', lambda a, b_, c: b_ - c), ('u2', 'b', lambda a, b_, c: c - a), ('u3', 'c', lambda a, b_, c: a - b_)):
                p.have(nm, ('Z.mul_nonneg', x, p.T('a b c', lambda a, b_, c, d=d: d(a, b_, c) * d(a, b_, c)), 'h' + x, ('Z.sq_nonneg', p.T('a b c', d))))
            p.linarith()


@problem('ineq_abc_2abc_1', text='For all non-negative integers a, b, c:  a^2 + b^2 + c^2 + 2abc + 1 >= 2 (ab + bc + ca).', source='folklore', origin='folklore', topics=('inequality', 'algebra', 'case-analysis'), level=4,
         spec=spec(('Z', 'Z', 'Z'), [LE(0, X(0)), LE(0, X(1)), LE(0, X(2))], LE(lambda a, b, c: 2 * (a * b + b * c + c * a), lambda a, b, c: a * a + b * b + c * c + 2 * a * b * c + 1), domains=(range(-3, 7), range(-3, 7), range(-3, 7))),
         notes='two of a - 1, b - 1, c - 1 have the same sign, say (a - 1)(b - 1) >= 0; then the difference is (a - b)^2 + (c - 1)^2 + 2c (a - 1)(b - 1)')
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'ineq_abc_2abc_1', [v.Z, v.Z, v.Z], [lambda a, b_, c: v.nonneg(a), lambda a, b_, c: v.nonneg(b_), lambda a, b_, c: v.nonneg(c)],
                   lambda a, b_, c: v.le(2 * (a * b_ + b_ * c + c * a), a * a + b_ * b_ + c * c + 2 * a * b_ * c + 1))
        def _(p):
            p.intro('a', 'b', 'c', 'ha', 'hb', 'hc')
            def close(x, y, z):                                                     # (x - 1)(y - 1) >= 0 by nlinarith, then the identity
                p.have('hxy', p.T(x + ' ' + y, lambda x_, y_: v.nonneg((x_ - 1) * (y_ - 1))), lambda q: q.nlinarith())
                p.nlinarith(('Z.mul_nonneg', z, p.T(x + ' ' + y, lambda x_, y_: (x_ - 1) * (y_ - 1)), 'h' + z, 'hxy'),
                            ('Z.sq_nonneg', p.T(x + ' ' + y, lambda x_, y_: x_ - y_)), ('Z.sq_nonneg', p.T(z, lambda z_: z_ - 1)))
            def go(rest, signs):
                if rest:
                    p.obtain(['s', 's'], ('Z.le_or_lt', lit(1), rest[0]))
                    go(rest[1:], signs + [True]); go(rest[1:], signs + [False])
                    return
                sa, sb, sc = signs
                if sa == sb: close('a', 'b', 'c')
                elif sb == sc: close('b', 'c', 'a')
                else: close('a', 'c', 'b')
            go(['a', 'b', 'c'], [])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ IMO 1983 P6, integer sides
def _E83(a, b, c): return a * a * b * (a - b) + b * b * c * (b - c) + c * c * a * (c - a)


@problem('imo1983_p6_integer', text='Let a, b and c be the lengths of the sides of a triangle (here: positive integers with a < b + c, b < c + a, c < a + b). Prove that  a^2 b (a - b) + b^2 c (b - c) + c^2 a (c - a) >= 0.  Determine when equality holds.',
         source='IMO 1983 P6', year=1983, origin='adapted', verified='recalled', topics=('inequality', 'algebra', 'identity'), level=4, kind='answer', answer='equality holds exactly when a = b = c',
         spec=spec(('Z', 'Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), LT(0, X(2)), LT(X(0), lambda a, b, c: b + c), LT(X(1), lambda a, b, c: c + a), LT(X(2), lambda a, b, c: a + b)],
                   lambda a, b, c: _E83(a, b, c) >= 0 and (_E83(a, b, c) != 0 or (a == b and b == c)),
                   atoms=[(lambda a, b, c: 0, _E83), (_E83, lambda a, b, c: 0), (lambda a, b, c: a, lambda a, b, c: b), (lambda a, b, c: b, lambda a, b, c: c)], domains=(range(-1, 9),) * 3),
         notes='adapted: the sides are integers (the original has real sides). With x = b + c - a, y = c + a - b, z = a + b - c (all positive) one has 8 (a^2 b (a - b) + b^2 c (b - c) + c^2 a (c - a)) = zx (x - y)^2 + xy (y - z)^2 + yz (z - x)^2; '
               'equality forces x = y = z because the weights zx, xy, yz are at least 1')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        X_ = lambda a, b_, c: b_ + c - a
        Y_ = lambda a, b_, c: c + a - b_
        Z_ = lambda a, b_, c: a + b_ - c
        D1 = lambda a, b_, c: X_(a, b_, c) - Y_(a, b_, c)
        D2 = lambda a, b_, c: Y_(a, b_, c) - Z_(a, b_, c)
        D3 = lambda a, b_, c: Z_(a, b_, c) - X_(a, b_, c)
        @K.theorem(w, 'imo1983_p6_integer', [v.Z, v.Z, v.Z], [lambda a, b_, c: v.pos(a), lambda a, b_, c: v.pos(b_), lambda a, b_, c: v.pos(c), lambda a, b_, c: v.lt(a, b_ + c), lambda a, b_, c: v.lt(b_, c + a), lambda a, b_, c: v.lt(c, a + b_)],
                   lambda a, b_, c: v.nonneg(_E83(a, b_, c)) & (Eq(_E83(a, b_, c), lit(0)) >> (Eq(a, b_) & Eq(b_, c))))
        def _(p):
            p.intro('a', 'b', 'c', 'ha', 'hb', 'hc', 'h1', 'h2', 'h3')
            T3 = lambda f: p.T('a b c', f)
            p.have('hx', T3(lambda a, b_, c: v.pos(X_(a, b_, c))), lambda q: q.linarith())
            p.have('hy', T3(lambda a, b_, c: v.pos(Y_(a, b_, c))), lambda q: q.linarith())
            p.have('hz', T3(lambda a, b_, c: v.pos(Z_(a, b_, c))), lambda q: q.linarith())
            W = {}
            for nm, f, g, hf, hg in (('zx', Z_, X_, 'hz', 'hx'), ('xy', X_, Y_, 'hx', 'hy'), ('yz', Y_, Z_, 'hy', 'hz')):
                p.have(nm, ('Z.mul_pos', T3(f), T3(g), hf, hg))                                                                       # 0 < zx, ..
                p.have(nm + '0', T3(lambda a, b_, c, f=f, g=g: v.nonneg(f(a, b_, c) * g(a, b_, c))), lambda q: q.linarith())
                p.have(nm + '1', T3(lambda a, b_, c, f=f, g=g: v.nonneg(f(a, b_, c) * g(a, b_, c) - 1)), lambda q: q.linarith())           # integers: at least 1
            for nm, wt, wnm, dd in (('t1', (Z_, X_), 'zx', D1), ('t2', (X_, Y_), 'xy', D2), ('t3', (Y_, Z_), 'yz', D3)):
                f, g = wt
                sqd = T3(lambda a, b_, c, dd=dd: dd(a, b_, c) * dd(a, b_, c))
                p.have(nm, ('Z.mul_nonneg', T3(lambda a, b_, c, f=f, g=g: f(a, b_, c) * g(a, b_, c)), sqd, wnm + '0', ('Z.sq_nonneg', T3(dd))))
                p.have('s' + nm, ('Z.mul_nonneg', T3(lambda a, b_, c, f=f, g=g: f(a, b_, c) * g(a, b_, c) - 1), sqd, wnm + '1', ('Z.sq_nonneg', T3(dd))))
            p.have('key', T3(lambda a, b_, c: Eq(8 * _E83(a, b_, c), Z_(a, b_, c) * X_(a, b_, c) * (D1(a, b_, c) * D1(a, b_, c)) + X_(a, b_, c) * Y_(a, b_, c) * (D2(a, b_, c) * D2(a, b_, c))
                                                  + Y_(a, b_, c) * Z_(a, b_, c) * (D3(a, b_, c) * D3(a, b_, c)))), lambda q: q.ring())
            p.split(); p.linarith()
            p.intro('h0')
            # (x - y)^2 <= zx (x - y)^2 = t1 <= 8 E = 0 and (y - z)^2 <= xy (y - z)^2 = t2 <= 8 E = 0
            for nm, dd, wnm in (('d1', D1, 'st1'), ('d2', D2, 'st2')):
                p.have(nm + 'sq', T3(lambda a, b_, c, dd=dd: Eq(dd(a, b_, c) * dd(a, b_, c), lit(0))), lambda q: q.nlinarith(('Z.sq_nonneg', T3(dd))))
                p.have(nm, T3(lambda a, b_, c, dd=dd: Eq(dd(a, b_, c), lit(0))),
                       lambda q, dd=dd, nm=nm: (q.obtain(['u', 'u'], ('Z.mul_eq_zero', q.T('a b c', dd), q.T('a b c', dd), nm + 'sq')), q.exact('u'), q.exact('u')))
            p.split(); p.linarith(); p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ functional equations
_F_RING_HOM = [lambda x: 0, lambda x: x, lambda x: 2 * x, lambda x: x * x, lambda x: 1, lambda x: -x, lambda x: abs(x), lambda x: x if x != 3 else 0, lambda x: x if x != 0 else 1, lambda x: 0 if x != 5 else 1,
               lambda x: x if abs(x) <= 12 else 0, lambda x: 0 if abs(x) <= 12 else x, lambda x: x if x % 2 == 0 else 0, lambda x: x % 2, lambda x: x if x != 1 else 2, lambda x: 3 * x, lambda x: x * x * x,
               lambda x: x if x >= 0 else 0, lambda x: x if x <= 0 else 0, lambda x: 0 if x % 3 else x]


@problem('fe_ring_hom', text='Determine all functions f : Z -> Z such that f(x + y) = f(x) + f(y) and f(x y) = f(x) f(y) for all integers x and y.', source='folklore', origin='folklore', topics=('functional-equation', 'induction'), level=3,
         kind='answer', answer='f = 0 or f(x) = x for all x',
         spec=spec(('fn',), [lambda f: all(f(x + y) == f(x) + f(y) for x in _W for y in _W), lambda f: all(f(x * y) == f(x) * f(y) for x in _W for y in _W)],
                   lambda f: all(f(x) == 0 for x in _W) or all(f(x) == x for x in _W), atoms=[(lambda f, x: f(x), lambda f, x: 0), (lambda f, x: f(x), lambda f, x: x)], hatoms=False, fns=_F_RING_HOM),
         notes='additivity gives f(n) = n f(1) for all integers n (induction in both directions), and f(1) = f(1)^2 leaves f(1) = 0 or 1')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'fe_ring_hom', [fZ], [lambda f: v.forall(lambda x: v.forall(lambda y: Eq(f(x + y), f(x) + f(y)))), lambda f: v.forall(lambda x: v.forall(lambda y: Eq(f(x * y), f(x) * f(y))))],
                   lambda f: OR(v.forall(lambda x: Eq(f(x), lit(0))), v.forall(lambda x: Eq(f(x), x))))
        def _(p):
            p.intro('f', 'ha', 'hm')
            def lin(q):
                q.intro('n'); q.induction_on('Z.int_induction', 'n')
                q.linarith(('ha', lit(0), lit(0)))
                q.intro('m', 'hm1'); q.linarith(('ha', 'm', lit(1)))
                q.intro('m', 'hm1'); q.linarith(('ha', q.T('m', lambda m: m - 1), lit(1)))
            p.have('lin', p.T('f', lambda f: v.forall(lambda n: Eq(f(n), n * f(lit(1))))), lin)
            p.have('o', ('hm', lit(1), lit(1)))
            p.have('e', p.T('f', lambda f: Eq(f(lit(1)) * (f(lit(1)) - 1), lit(0))), lambda q: q.linarith())
            p.obtain(['c', 'c'], ('Z.mul_eq_zero', p.T('f', lambda f: f(lit(1))), p.T('f', lambda f: f(lit(1)) - 1), 'e'))
            p.left(); p.intro('x'); p.have('t', ('lin', 'x')); p.linear_combination((1, 't'), (p.T('x', lambda x: x), 'c'))
            p.right(); p.intro('x'); p.have('t', ('lin', 'x')); p.linear_combination((1, 't'), (p.T('x', lambda x: x), 'c'))


def _fix_beyond(g, k, v):
    """g on the positive integers up to k and the value v beyond"""
    return lambda n: g(n) if n <= k else v


_F_77 = [lambda n: n, lambda n: n + 1, lambda n: 1, lambda n: 2, lambda n: 2 * n, lambda n: n - 1, lambda n: n + 2, lambda n: max(n - 1, 1), lambda n: abs(n) if n else 5, lambda n: n if n >= 1 else 5,
         lambda n: n if n >= 1 else -3, lambda n: n if n != 3 else 4, lambda n: n if n != 3 else 2, lambda n: n if n != 1 else 2, lambda n: n if n != 8 else 9, lambda n: n if n != 8 else 7, lambda n: n if n != 5 else 6,
         lambda n: n if n != 5 else 4, lambda n: n if n <= 9 else 1, lambda n: n if n <= 9 else 2 * n, lambda n: n if n <= 9 else n + 1, lambda n: n if n != 2 else 3, lambda n: n if n != 4 else 1, lambda n: n if n != 6 else 0,
         lambda n: n if n != 7 else 1, lambda n: n if n >= 2 else 0, lambda n: n if n >= 3 else 1, lambda n: n if n % 2 else n + 1, lambda n: n if n % 2 == 0 else n + 1, lambda n: (n + 1) // 2, lambda n: 2 * n - 1]


@problem('imo1977_p6', text='Let f : N -> N be a function from the positive integers into the positive integers such that f(n + 1) > f(f(n)) for every positive integer n. Prove that f(n) = n for every positive integer n.',
         source='IMO 1977 P6', year=1977, origin='contest', verified='recalled', topics=('functional-equation', 'induction', 'bounding'), level=4,
         spec=spec(('fn',), [lambda f: all(f(n) > 0 for n in _W if n > 0), lambda f: all(f(f(n)) < f(n + 1) for n in _W if n > 0)], lambda f: all(f(n) == n for n in _W if n > 0),
                   atoms=[(lambda f, n: 0, lambda f, n: n), (lambda f, n: f(n), lambda f, n: n)], hatoms=False, fns=_F_77),
         notes='the function is formalised on the integers: only its values at positive integers are constrained. Induction on n shows that f(k) >= n for all k >= n; hence f(n) >= n and f(f(n)) >= f(n), so f is strictly increasing; '
               'then f(n) >= n + 1 would give f(f(n)) >= f(n + 1)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'imo1977_p6', [fZ], [lambda f: v.forall(lambda n: v.pos(n) >> v.pos(f(n))), lambda f: v.forall(lambda n: v.pos(n) >> v.lt(f(f(n)), f(n + 1)))],
                   lambda f: v.forall(lambda n: v.pos(n) >> Eq(f(n), n)))
        def _(p):
            p.intro('f', 'h1', 'h2', 'n', 'hn')
            Pn = lambda f, n: v.forall(lambda k: v.le(n, k) >> v.le(n, f(k)))                        # f(k) >= n for all k >= n
            def base(q):
                q.intro('k', 'hk')
                q.have('k0', q.T('k', lambda k: v.pos(k)), lambda r: r.linarith())
                q.have('x', ('h1', 'k', 'k0')); q.linarith()
            def step(q):
                q.intro('t', 'ht', 'ih', 'k', 'hk')
                q.have('j', q.T('t k', lambda t, k: v.le(t, k - 1)), lambda r: r.linarith())
                q.have('i1', ('ih', q.T('k', lambda k: k - 1), 'j'))
                q.have('i2', ('ih', q.T('f k', lambda f, k: f(k - 1)), 'i1'))
                q.have('k0', q.T('t k', lambda t, k: v.pos(k - 1)), lambda r: r.linarith())
                q.have('s', ('h2', q.T('k', lambda k: k - 1), 'k0'))
                q.linarith()
            def A(q):
                q.intro('m', 'hm', 'kk', 'hkk')
                q.have('base', q.T('f', lambda f: Pn(f, lit(1))), base)
                q.have('step', q.T('f', lambda f: v.forall(lambda t: v.le(lit(1), t) >> (Pn(f, t) >> Pn(f, t + v.one)))), step)
                q.exact(lambda r: w['Z.induction_from'](r.T('f', lambda f: w.lam(v.Z, lambda n_: Pn(f, n_))), lit(1), r.h('base'), r.h('step'), r.h('m'), r.h('hm'), r.h('kk'), r.h('hkk')))
            p.have('A', p.T('f', lambda f: v.forall(lambda m: v.le(lit(1), m) >> Pn(f, m))), A)
            def B(q):                                                                 # f(m) >= m
                q.intro('m', 'hm'); q.have('r', q.T('m', lambda m: v.le(m, m)), lambda r: r.linarith())
                q.exact(('A', 'm', 'hm', 'm', 'r'))
            p.have('B', p.T('f', lambda f: v.forall(lambda m: v.le(lit(1), m) >> v.le(m, f(m)))), B)
            def C(q):                                                                 # f(m) < f(m + 1)
                q.intro('m', 'hm')
                q.have('m0', q.T('m', lambda m: v.pos(m)), lambda r: r.linarith())
                q.have('b1', ('B', 'm', 'hm'))
                q.have('b1x', q.T('f m', lambda f, m: v.le(lit(1), f(m))), lambda r: r.linarith())
                q.have('b2', ('B', q.T('f m', lambda f, m: f(m)), 'b1x'))
                q.have('s', ('h2', 'm', 'm0'))
                q.linarith()
            p.have('C', p.T('f', lambda f: v.forall(lambda m: v.le(lit(1), m) >> v.lt(f(m), f(m + 1)))), C)
            def D(q):                                                                 # f is monotone on the positive integers
                q.intro('a', 'ha', 'b', 'hb'); q.induction_on('Z.induction_from', 'b', extra=(q.T('a', lambda a: a),))
                q.linarith()
                q.intro('t', 'ht', 'ih')
                q.have('c1', q.T('t', lambda t: v.le(lit(1), t)), lambda r: r.linarith())
                q.have('c', ('C', 't', 'c1')); q.linarith()
                q.assumption()
            p.have('D', p.T('f', lambda f: v.forall(lambda a: v.le(lit(1), a) >> v.forall(lambda b_: v.le(a, b_) >> v.le(f(a), f(b_))))), D)
            p.have('hn1', p.T('n', lambda n: v.le(lit(1), n)), lambda q: q.linarith())
            p.have('b1', ('B', 'n', 'hn1'))
            p.obtain(['c', 'c'], ('Z.le_or_lt', p.T('f n', lambda f, n: f(n)), 'n'))
            p.linarith()
            p.have('m1', p.T('n', lambda n: v.le(lit(1), n + 1)), lambda q: q.linarith())
            p.have('m2', p.T('f n', lambda f, n: v.le(n + 1, f(n))), lambda q: q.linarith())
            p.have('d', ('D', p.T('n', lambda n: n + 1), 'm1', p.T('f n', lambda f, n: f(n)), 'm2'))
            p.have('s', ('h2', 'n', 'hn'))
            p.exfalso(); p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ISL 2005 N3 and a cyclic inequality
_D05 = (0, 1, 2, 3, 4)
_S05 = lambda a, b, c, d, e, f: a + b + c + d + e + f


@problem('isl2005_n3', text='Let a, b, c, d, e, f be positive integers and let S = a + b + c + d + e + f. Suppose that S divides abc + def and ab + bc + ca - de - ef - fd. Prove that S is composite.', source='ISL 2005 N3', year=2005,
         topics=('number-theory', 'divisibility', 'prime', 'identity'), level=4, origin='contest', verified='recalled',
         spec=spec(('Z',) * 6, [LT(0, X(i)) for i in range(6)] + [DVD(_S05, lambda a, b, c, d, e, f: a * b * c + d * e * f), DVD(_S05, lambda a, b, c, d, e, f: a * b + b * c + c * a - d * e - e * f - f * d)],
                   lambda a, b, c, d, e, f: not is_prime_(_S05(a, b, c, d, e, f)), atoms=[(_S05,)], domains=(_D05,) * 6, points=16000),
         notes='S divides P(x) = (x + a)(x + b)(x + c) - (x - d)(x - e)(x - f) = S x^2 + (ab + bc + ca - de - ef - fd) x + (abc + def) for every integer x; at x = d the second product vanishes, so a prime S would divide (a + d)(b + d)(c + d) and hence one of the '
               'factors, but each of them lies strictly between 0 and S. "Composite" is "not a prime" (S >= 6). The source label (Mongolia, ISL 2005) is from the shortlist; the exact wording of the second divisibility condition is from memory '
               '(it is the one for which the polynomial identity above works)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; P = w['Z.prime']
    with v.ints():
        S6 = _S05
        @K.theorem(w, 'isl2005_n3', [v.Z] * 6, [lambda a, b_, c, d, e, f: v.pos(a), lambda a, b_, c, d, e, f: v.pos(b_), lambda a, b_, c, d, e, f: v.pos(c), lambda a, b_, c, d, e, f: v.pos(d), lambda a, b_, c, d, e, f: v.pos(e), lambda a, b_, c, d, e, f: v.pos(f),
                                                lambda a, b_, c, d, e, f: v.dvd(S6(a, b_, c, d, e, f), a * b_ * c + d * e * f),
                                                lambda a, b_, c, d, e, f: v.dvd(S6(a, b_, c, d, e, f), a * b_ + b_ * c + c * a - d * e - e * f - f * d)],
                   lambda a, b_, c, d, e, f: v.Not(P(S6(a, b_, c, d, e, f))))
        def _(p):
            p.intro('a', 'b', 'c', 'd', 'e', 'f', 'ha', 'hb', 'hc', 'hd', 'he', 'hf', 'h1', 'h2', 'hp')
            p.obtain(['k1', 'hk1'], 'h1')
            p.obtain(['k2', 'hk2'], 'h2')
            p.have('key', p.T('a b c d e f k1 k2', lambda a, b_, c, d, e, f, k1, k2: Eq((a + d) * (b_ + d) * (c + d), S6(a, b_, c, d, e, f) * (d * d + k2 * d + k1))),
                   lambda q: q.linear_combination((q.h('d'), 'hk2'), (1, 'hk1')))
            p.have('dv', p.T('a b c d e f', lambda a, b_, c, d, e, f: v.dvd(S6(a, b_, c, d, e, f), (a + d) * (b_ + d) * (c + d))), lambda q: (q.exists(q.T('d k1 k2', lambda d, k1, k2: d * d + k2 * d + k1)), q.exact('key')))
            Sx = lambda a, b_, c, d, e, f: S6(a, b_, c, d, e, f)

            def close(hyp, X):                                                          # hyp : S | X  with 0 < X < S: False
                p.have('px', p.T('a b c d', lambda a, b_, c, d: v.pos(X(a, b_, c, d))), lambda q: q.linarith())
                p.have('lx', p.T('a b c d e f', lambda a, b_, c, d, e, f: v.le(S6(a, b_, c, d, e, f), X(a, b_, c, d))), lambda q: q.exact(('Z.le_of_dvd', p.T('a b c d e f', Sx), p.T('a b c d', X), 'px', hyp)))
                p.linarith()
            p.obtain(['u', 'u'], ('Z.prime_dvd_mul', p.T('a b c d e f', Sx), 'hp', p.T('a b d', lambda a, b_, d: (a + d) * (b_ + d)), p.T('c d', lambda c, d: c + d), 'dv'))
            p.obtain(['t', 't'], ('Z.prime_dvd_mul', p.T('a b c d e f', Sx), 'hp', p.T('a d', lambda a, d: a + d), p.T('b d', lambda b_, d: b_ + d), 'u'))
            close('t', lambda a, b_, c, d: a + d)
            close('t', lambda a, b_, c, d: b_ + d)
            close('u', lambda a, b_, c, d: c + d)


@problem('ineq_cyclic_27_4', text='For all non-negative integers a, b, c:  27 (a^2 b + b^2 c + c^2 a + a b c) <= 4 (a + b + c)^3.', source='folklore', origin='folklore', topics=('inequality', 'algebra', 'case-analysis'), level=4,
         spec=spec(('Z', 'Z', 'Z'), [LE(0, X(0)), LE(0, X(1)), LE(0, X(2))], LE(lambda a, b, c: 27 * (a * a * b + b * b * c + c * c * a + a * b * c), lambda a, b, c: 4 * (a + b + c) ** 3), domains=(range(-3, 7), range(-3, 7), range(-3, 7))),
         notes='the expression is cyclic, so one may rotate the variables until y is the middle one of x, y, z; then 4 (x + y + z)^3 - 27 (x^2 y + y^2 z + z^2 x + x y z) = (x + z - 2y)^2 (4 (x + z) + y) + 27 z (x - y)(y - z), and both terms are non-negative. '
               'Equality holds for a = b = c and for (a, b, c) = (2, 1, 0) and its rotations')
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        @K.theorem(w, 'ineq_cyclic_27_4', [v.Z, v.Z, v.Z], [lambda a, b_, c: v.nonneg(a), lambda a, b_, c: v.nonneg(b_), lambda a, b_, c: v.nonneg(c)],
                   lambda a, b_, c: v.le(27 * (a * a * b_ + b_ * b_ * c + c * c * a + a * b_ * c), 4 * (a + b_ + c) ** 3))
        def _(p):
            p.intro('a', 'b', 'c', 'ha', 'hb', 'hc')

            def leaf(x, y, z, asc):                                                    # y is the middle one of x, y, z (x <= y <= z if asc, else x >= y >= z), the triple is a rotation of (a, b, c)
                d1 = (lambda X_, Y_: Y_ - X_) if asc else (lambda X_, Y_: X_ - Y_)
                d2 = (lambda Y_, Z_: Z_ - Y_) if asc else (lambda Y_, Z_: Y_ - Z_)
                p.have('d1', p.T(x + ' ' + y, lambda X_, Y_: v.nonneg(d1(X_, Y_))), lambda q: q.linarith())
                p.have('d2', p.T(y + ' ' + z, lambda Y_, Z_: v.nonneg(d2(Y_, Z_))), lambda q: q.linarith())
                p.have('m', ('Z.mul_nonneg', p.T(x + ' ' + y, d1), p.T(y + ' ' + z, d2), 'd1', 'd2'))
                p.have('f2', ('Z.mul_nonneg', z, p.T(x + ' ' + y + ' ' + z, lambda X_, Y_, Z_: d1(X_, Y_) * d2(Y_, Z_)), 'h' + z, 'm'))
                p.have('lin', p.T(x + ' ' + y + ' ' + z, lambda X_, Y_, Z_: v.nonneg(4 * (X_ + Z_) + Y_)), lambda q: q.linarith())
                p.have('f1', ('Z.mul_nonneg', p.T(x + ' ' + y + ' ' + z, lambda X_, Y_, Z_: (X_ + Z_ - 2 * Y_) * (X_ + Z_ - 2 * Y_)), p.T(x + ' ' + y + ' ' + z, lambda X_, Y_, Z_: 4 * (X_ + Z_) + Y_),
                              ('Z.sq_nonneg', p.T(x + ' ' + y + ' ' + z, lambda X_, Y_, Z_: X_ + Z_ - 2 * Y_)), 'lin'))
                p.linarith()
            p.obtain(['s1', 's1'], ('Z.le_or_lt', 'a', 'b'))
            p.obtain(['s2', 's2'], ('Z.le_or_lt', 'b', 'c'))
            leaf('a', 'b', 'c', True)                                                  # a <= b <= c
            p.obtain(['s3', 's3'], ('Z.le_or_lt', 'a', 'c'))
            leaf('b', 'c', 'a', False)                                                 # a <= c < b: b >= c >= a
            leaf('c', 'a', 'b', True)                                                  # c < a <= b: c <= a <= b
            p.obtain(['s2', 's2'], ('Z.le_or_lt', 'b', 'c'))
            p.obtain(['s3', 's3'], ('Z.le_or_lt', 'a', 'c'))
            leaf('c', 'a', 'b', False)                                                 # b < a <= c: c >= a >= b
            leaf('b', 'c', 'a', True)                                                  # b <= c < a: b <= c <= a
            leaf('a', 'b', 'c', False)                                                 # c < b < a
