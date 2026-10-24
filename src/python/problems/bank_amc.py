"""bank_amc.py -- problems in the manner of (and some from) the American contests: AMC 8 / 10 / 12 and AIME.  Integer vocabulary, a kernel-checked reference proof for each.

Provenance.  A problem of kind `origin='contest'` is a problem of the named contest as the authors recall it, PARAPHRASED (the texts are not the contest's); its formal statement is checked by brute force
against the contest's published answer where that is an integer (`verified='answer-checked'`: the statement's answer agrees; `recalled`: it was not checked).  `adapted`: a contest problem changed to fit the
integer vocabulary.  `style`: written for the bank in the manner of a contest.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, math
from nt_lib import _syms
from lnl_base import thm
from ext_lib import destruct
from semantic import Spec
from bank import BANK
from bank_spec import spec, EQ, LT, LE, DVD, COP, GCD, PRIME, V as X
import bank_kit as K

NAMES = []


def problem(name, **kw):
    NAMES.append(name)
    return BANK.problem(name, **kw)


R = range(-8, 9)


def V(i):
    """the atom `0 < v_i` (or `0 <= v_i`) of a hypothesis or conclusion: the variable number i"""
    return (lambda *a: 0, lambda *a, i=i: a[i])


def _dvd(a, b): return b == 0 if a == 0 else b % a == 0


def AND(*xs):
    """a right-nested conjunction"""
    r = xs[-1]
    for x in reversed(xs[:-1]): r = x & r
    return r


def OR(*xs):
    """a right-nested disjunction"""
    r = xs[-1]
    for x in reversed(xs[:-1]): r = x | r
    return r


def sq(n):
    r = math.isqrt(n)
    return r * r == n


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ AIME
@problem('aime1985_p3', text='Positive integers a, b, c satisfy c = (a + bi)^3 - 107i, where i^2 = -1. Find c.', source='AIME 1985 P3', year=1985, topics=('number-theory', 'diophantine', 'algebra'), level=3,
         kind='answer', answer='c = 198', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z', 'Z'), (lambda a, b, c: a > 0, lambda a, b, c: b > 0, lambda a, b, c: c > 0, lambda a, b, c: c == a ** 3 - 3 * a * b * b, lambda a, b, c: 3 * a * a * b - b ** 3 == 107),
                   lambda a, b, c: c == 198, domains=(R, R, [-1, 0, 1, 197, 198, 199]), points=2000,
                   hatoms=[(lambda a, b, c: 0, lambda a, b, c: a), (lambda a, b, c: 0, lambda a, b, c: b), (lambda a, b, c: 0, lambda a, b, c: c), (lambda a, b, c: c, lambda a, b, c: a ** 3 - 3 * a * b * b),
                           (lambda a, b, c: 3 * a * a * b - b ** 3, lambda a, b, c: 107)],
                   atoms=[(lambda a, b, c: c, lambda a, b, c: 198)]),
         notes='real part c = a^3 - 3ab^2, imaginary part 3a^2 b - b^3 = 107 = b (3a^2 - b^2): b divides the prime 107')
def _(w, b):
    s = _syms(w); Z, Eq, zzero, zlt, zle = s.Z, s.Eq, s.zzero, s.zlt, s.zle
    with w.nota.using(Z):
        lit = w.nota.lit
        @thm(w, 'aime1985_p3', [Z, Z, Z, lambda a, b_, c: zlt(zzero, a), lambda a, b_, c, h: zlt(zzero, b_), lambda a, b_, c, h, h1: zlt(zzero, c), lambda a, b_, c, h, h1, h2: Eq(c, a ** 3 - 3 * a * b_ ** 2),
                                lambda a, b_, c, h, h1, h2, h3: Eq(3 * a ** 2 * b_ - b_ ** 3, lit(107))], lambda a, b_, c, h, h1, h2, h3, h4: Eq(c, lit(198)))
        def _(p):
            p.intro('a', 'b', 'c', 'ha', 'hb', 'hc', 'h1', 'h2')
            K.let(p, 't', 'a b', lambda a_, b_: 3 * a_ ** 2 - b_ ** 2)
            p.have('e', p.T('b t', lambda b_, t_: Eq(b_ * t_, lit(107))), lambda q: q.linear_combination((q.h('b'), 't_def'), (1, 'h2')))
            p.have('ht', p.T('t', lambda t_: zlt(zzero, t_)), lambda q: q.nlinarith())
            def pair(p, d, qv):
                if d == 1:                                                              # b = 1, t = 107: 3a^2 = 108
                    p.rw_at('t_def', 'dx')
                    p.have('a2', p.T('a', lambda a_: Eq(a_ ** 2, lit(36))), lambda q: q.linarith())
                    p.have('a6', p.T('a', lambda a_: Eq(a_, lit(6))), lambda q: q.nlinarith())
                    p.rw_at('h1', 'a6'); p.rw_at('h1', 'dx'); p.linarith()
                else:                                                                   # b = 107, t = 1: 3a^2 = 11450, and 3 does not divide 11450
                    p.exfalso(); p.apply('not_dvd_between', lit(3), lit(3816), lit(11450)); p.linarith(); p.linarith(); p.linarith()
                    p.exists(p.T('a', lambda a_: a_ ** 2))
                    p.linear_combination((1, 't_def'), (-1, 'dy'), (p.T('b', lambda b_: -(b_ + 107)), 'dx'))
            K.divisor_cases(p, 'b', 't', 'e', 107, pair)


@problem('aime1991_p1', text='Positive integers x and y satisfy xy + x + y = 71 and x^2 y + x y^2 = 880. Find x^2 + y^2.', source='AIME 1991 P1', year=1991, topics=('algebra', 'diophantine'), level=2, kind='answer',
         answer='x^2 + y^2 = 146', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z'), (lambda x, y: x > 0, lambda x, y: y > 0, lambda x, y: x * y + x + y == 71, lambda x, y: x * x * y + x * y * y == 880), lambda x, y: x * x + y * y == 146,
                   domains=(range(-4, 14), range(-4, 14)), hatoms=[V(0), V(1), (lambda x, y: x * y + x + y, lambda x, y: 71), (lambda x, y: x * x * y + x * y * y, lambda x, y: 880)],
                   atoms=[(lambda x, y: x * x + y * y, lambda x, y: 146)]),
         notes='(x + 1)(y + 1) = 72: the positive factorisations of 72 leave the pairs (5, 11) and (11, 5)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1991_p1', [v.Z, v.Z], [lambda x, y: v.pos(x), lambda x, y: v.pos(y), lambda x, y: Eq(x * y + x + y, lit(71)), lambda x, y: Eq(x ** 2 * y + x * y ** 2, lit(880))], lambda x, y: Eq(x ** 2 + y ** 2, lit(146)))
        def _(p):
            p.intro('x', 'y', 'hx', 'hy', 'h1', 'h2')
            K.let(p, 'u', 'x', lambda x: x + 1); K.let(p, 'v', 'y', lambda y: y + 1)
            p.have('hu', p.T('u', v.pos), lambda q: q.linarith())
            p.have('hv', p.T('v', v.pos), lambda q: q.linarith())
            p.have('e', p.T('u v', lambda u, v_: Eq(u * v_, lit(72))), lambda q: q.linear_combination((q.h('v'), 'u_def'), (q.T('x', lambda x: x + 1), 'v_def'), (1, 'h1')))
            def pair(p, d, c):
                xv, yv = d - 1, c - 1
                if xv <= 0 or yv <= 0: p.exfalso(); p.linarith(); return
                p.have('ex', p.T('x', lambda x: Eq(x, lit(xv))), lambda q: q.linarith())
                p.have('ey', p.T('y', lambda y: Eq(y, lit(yv))), lambda q: q.linarith())
                if xv * xv * yv + xv * yv * yv != 880: p.exfalso(); p.rw_at('h2', 'ex'); p.rw_at('h2', 'ey'); p.linarith()
                else: p.rw('ex'); p.rw('ey')
            K.divisor_cases(p, 'u', 'v', 'e', 72, pair)


@problem('aime1987_p5', text='Integers x and y satisfy y^2 + 3 x^2 y^2 = 30 x^2 + 517. Find 3 x^2 y^2.', source='AIME 1987 P5', year=1987, topics=('algebra', 'diophantine'), level=3, kind='answer',
         answer='3 x^2 y^2 = 588', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z'), (lambda x, y: y * y + 3 * x * x * y * y == 30 * x * x + 517,), lambda x, y: 3 * x * x * y * y == 588, domains=(range(-6, 7), range(-9, 10)),
                   hatoms=[(lambda x, y: y * y + 3 * x * x * y * y, lambda x, y: 30 * x * x + 517)], atoms=[(lambda x, y: 3 * x * x * y * y, lambda x, y: 588)]),
         notes='(3x^2 + 1)(y^2 - 10) = 507 = 3 . 13^2, and 3x^2 + 1 is 1 modulo 3: only 3x^2 + 1 = 13 gives squares')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1987_p5', [v.Z, v.Z], [lambda x, y: Eq(y ** 2 + 3 * x ** 2 * y ** 2, 30 * x ** 2 + lit(517))], lambda x, y: Eq(3 * x ** 2 * y ** 2, lit(588)))
        def _(p):
            p.intro('x', 'y', 'h')
            K.let(p, 'k', 'x', lambda x: 3 * x ** 2 + 1); K.let(p, 't', 'y', lambda y: y ** 2 - 10)
            p.have('hk', p.T('k', v.pos), lambda q: q.nlinarith())
            p.have('e', p.T('k t', lambda k, t: Eq(k * t, lit(507))), lambda q: q.linear_combination((q.h('t'), 'k_def'), (q.T('x', lambda x: 3 * x ** 2 + 1), 't_def'), (1, 'h')))
            p.have('ht', p.T('t', v.pos), lambda q: q.nlinarith())
            def pair(p, d, c):
                x2 = (d - 1) // 3 if (d - 1) % 3 == 0 else None; y2 = c + 10
                if x2 is None:                                                                   # 3 x^2 = d - 1 is not a multiple of 3
                    p.exfalso(); p.obtain(['_c', '_c'], ('Z.le_or_lt', p.T('x', lambda x: x * x), lit((d - 1) // 3))); p.linarith(); p.linarith(); return
                if not sq(x2):
                    p.exfalso(); p.apply('not_square_between', p.h('x'), lit(math.isqrt(x2))); p.linarith(); p.linarith(); p.linarith(); return
                if not sq(y2):
                    p.exfalso(); p.apply('not_square_between', p.h('y'), lit(math.isqrt(y2))); p.linarith(); p.linarith(); p.linarith(); return
                p.have('hx2', p.T('x', lambda x: Eq(x ** 2, lit(x2))), lambda q: q.linarith())
                p.have('hy2', p.T('y', lambda y: Eq(y ** 2, lit(y2))), lambda q: q.linarith())
                p.linear_combination((p.T('y', lambda y: 3 * y ** 2), 'hx2'), (lit(3 * x2), 'hy2'))
            K.divisor_cases(p, 'k', 't', 'e', 507, pair)


@problem('aime1986_p5', text='Find the largest positive integer n for which n^3 + 100 is divisible by n + 10.', source='AIME 1986 P5', year=1986, topics=('number-theory', 'divisibility'), level=2, kind='answer',
         answer='n = 890', origin='contest', verified='answer-checked',
         spec=Spec(('Z',), (lambda n: n > 0, lambda n: _dvd(n + 10, n ** 3 + 100)), lambda n: n <= 890, domains=(list(range(-3, 30)) + list(range(885, 900)),),
                   hatoms=[V(0), (lambda n: n + 10, lambda n: n ** 3 + 100)], atoms=[(lambda n: n, lambda n: 890), (lambda n: 890 + 10, lambda n: 890 ** 3 + 100)]),
         notes='formalised as: no positive n above 890 has the property, and 890 has it; n^3 + 100 = (n + 10)(n^2 - 10 n + 100) - 900, so n + 10 divides 900')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1986_p5', [v.Z], [lambda n: v.pos(n), lambda n: D(n + 10, n ** 3 + 100)], lambda n: v.le(n, lit(890)) & D(lit(890) + 10, lit(890) ** 3 + 100))
        def _(p):
            p.intro('n', 'hn', 'hd')
            p.split()
            p.have('h1', p.T('n', lambda n: D(n + 10, (n + 10) * (n ** 2 - 10 * n + 100))), lambda q: q.apply('Z.dvd_mul_right'))
            p.have('h2', p.T('n', lambda n: D(n + 10, (n + 10) * (n ** 2 - 10 * n + 100) - (n ** 3 + 100))), lambda q: (q.apply('Z.dvd_sub'), q.exact('h1'), q.exact('hd')))
            p.have('e', p.T('n', lambda n: Eq((n + 10) * (n ** 2 - 10 * n + 100) - (n ** 3 + 100), lit(900))), lambda q: q.ring())
            p.rw_at('h2', 'e')
            p.have('h3', p.T('n', lambda n: v.le(n + 10, lit(900))), lambda q: (q.apply('Z.le_of_dvd'), q.linarith(), q.exact('h2')))
            p.linarith()
            p.exists(lit(783299)); p.refl()


@problem('aime1999_p3', text='Find the sum of all positive integers n for which n^2 - 19 n + 99 is a perfect square.', source='AIME 1999 P3', year=1999, topics=('number-theory', 'diophantine'), level=3, kind='answer',
         answer='n = 1, 9, 10, 18 (sum 38)', origin='contest', verified='answer-checked',
         spec=Spec(('Z',), (lambda n: n > 0,), (lambda n: sq(n * n - 19 * n + 99), lambda n: n in (1, 9, 10, 18)), windows={'z': 25, 'ex': 20}, domains=(range(-2, 26),),
                   hatoms=[V(0)], atoms=[(lambda n, m: m * m, lambda n, m: n * n - 19 * n + 99), (lambda n: n, lambda n: 1), (lambda n: n, lambda n: 9), (lambda n: n, lambda n: 10), (lambda n: n, lambda n: 18)]),
         notes='formalised as an equivalence with the list of the solutions (whose sum is 38); 4 (n^2 - 19 n + 99) = (2n - 19)^2 + 35, so (2m - 2n + 19)(2m + 2n - 19) = 35')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1999_p3', [v.Z], [lambda n: v.pos(n)], lambda n: v.Iff(v.exists(lambda m: Eq(m ** 2, n ** 2 - 19 * n + 99)), OR(Eq(n, lit(1)), Eq(n, lit(9)), Eq(n, lit(10)), Eq(n, lit(18)))))
        def _(p):
            p.intro('n', 'hn'); p.split()
            p.intro('h'); p.cases('h', names=['m', 'hm'])
            def key(q):
                q.obtain(['c', 'c'], ('Z.le_or_lt', lit(0), 'm'))
                q.exists('m'); q.split(); q.exact('c'); q.linear_combination((1, 'hm'))
                q.exists(q.T('m', lambda m: -m)); q.split(); q.linarith(); q.linear_combination((1, 'hm'))
            p.have('key', p.T('n', lambda n: v.exists(lambda a: v.nonneg(a) & Eq(a * a, n ** 2 - 19 * n + 99))), key)
            p.cases('key', names=['a', 'hk']); p.cases('hk', names=['ha', 'hsq'])
            K.let(p, 'u', 'a n', lambda a, n: 2 * a + 2 * n - 19); K.let(p, 'v', 'a n', lambda a, n: 2 * a - 2 * n + 19)
            p.have('e', p.T('u v', lambda u, v_: Eq(u * v_, lit(35))), lambda q: q.linear_combination((q.h('v'), 'u_def'), (q.T('a n', lambda a, n: 2 * a + 2 * n - 19), 'v_def'), (4, 'hsq')))
            p.have('hs', p.T('u v', lambda u, v_: v.nonneg(u + v_)), lambda q: q.linarith())
            p.have('hu', p.T('u', v.pos), lambda q: (q.obtain(['c', 'c'], ('Z.le_or_lt', q.h('u'), lit(0))), q.exfalso(), q.nlinarith(('Z.sq_nonneg', q.h('u'))), q.exact('c')))
            p.have('hv', p.T('v', v.pos), lambda q: (q.obtain(['c', 'c'], ('Z.le_or_lt', q.h('v'), lit(0))), q.exfalso(), q.nlinarith(('Z.sq_nonneg', q.h('v'))), q.exact('c')))
            def pair(p, d, c):
                nv = (d - c + 38) // 4
                p.have('en', p.T('n', lambda n: Eq(n, lit(nv))), lambda q: q.linarith())
                for i, t in enumerate((1, 9, 10, 18)):
                    if t == nv:
                        for _ in range(i): p.right()
                        if i < 3: p.left()
                        p.exact('en'); return
            K.divisor_cases(p, 'u', 'v', 'e', 35, pair)
            p.intro('h'); p.cases('h', names=['h1', 'h2'])
            p.exists(lit(9)); K.rw_ring(p, 'h1')
            p.cases('h2', names=['h2', 'h3'])
            p.exists(lit(3)); K.rw_ring(p, 'h2')
            p.cases('h3', names=['h3', 'h4'])
            p.exists(lit(3)); K.rw_ring(p, 'h3')
            p.exists(lit(9)); K.rw_ring(p, 'h4')


@problem('aime1985_p13', text='For n >= 1 let a_n = 100 + n^2 and let d_n be the greatest common divisor of a_n and a_(n+1). Find the maximum value of d_n.', source='AIME 1985 P13', year=1985, topics=('number-theory', 'gcd'), level=3,
         kind='answer', answer='the maximum is 401 (n = 200)', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z'), (lambda n, d: n > 0, lambda n, d: d == math.gcd(100 + n * n, 100 + (n + 1) ** 2)), lambda n, d: d <= 401,
                   domains=(list(range(-2, 13)) + [199, 200, 201, 202], list(range(0, 7)) + [400, 401, 402]),
                   hatoms=[V(0), (lambda n, d: 100 + n * n, lambda n, d: 100 + (n + 1) ** 2, lambda n, d: d)],
                   atoms=[(lambda n, d: d, lambda n, d: 401), (lambda n, d: 100 + 200 ** 2, lambda n, d: 100 + 201 ** 2, lambda n, d: 401)]),
         notes='formalised as: no gcd exceeds 401, and gcd(a_200, a_201) = 401; d divides a_(n+1) - a_n = 2n + 1 and 4 a_n = (2n - 1)(2n + 1) + 401')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd; G = w['Z.is_gcd']; sqr = w['Z.sqr']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1985_p13', [v.Z, v.Z], [lambda n, d: v.pos(n), lambda n, d: G(100 + n ** 2, 100 + (n + 1) ** 2, d)], lambda n, d: v.le(d, lit(401)) & G(100 + lit(200) * lit(200), 100 + lit(201) * lit(201), lit(401)))
        def _(p):
            p.intro('n', 'd', 'hn', 'hg')
            p.cases('hg', names=['g0', 'g']); p.cases('g', names=['da', 'g2']); p.cases('g2', names=['db', 'gm'])
            p.split()
            p.have('s1', p.T('n d', lambda n, d: D(d, 4 * (100 + n ** 2))), lambda q: q.exact(('Z.dvd_mul_of_dvd_right', 'd', lit(4), q.T('n', lambda n: 100 + sqr(n)), 'da')))
            p.have('s2', p.T('n d', lambda n, d: D(d, (100 + (n + 1) ** 2) - (100 + n ** 2))), lambda q: q.exact(('Z.dvd_sub', 'd', q.T('n', lambda n: 100 + sqr(n + 1)), q.T('n', lambda n: 100 + sqr(n)), 'db', 'da')))
            p.have('s3', p.T('n d', lambda n, d: D(d, (2 * n - 1) * ((100 + (n + 1) ** 2) - (100 + n ** 2)))),
                   lambda q: q.exact(('Z.dvd_mul_of_dvd_right', 'd', q.T('n', lambda n: 2 * n - 1), q.T('n', lambda n: (100 + sqr(n + 1)) - (100 + sqr(n))), 's2')))
            p.have('s4', p.T('n d', lambda n, d: D(d, 4 * (100 + n ** 2) - (2 * n - 1) * ((100 + (n + 1) ** 2) - (100 + n ** 2)))),
                   lambda q: q.exact(('Z.dvd_sub', 'd', q.T('n', lambda n: 4 * (100 + sqr(n))), q.T('n', lambda n: (2 * n - 1) * ((100 + sqr(n + 1)) - (100 + sqr(n)))), 's1', 's3')))
            p.have('e', p.T('n', lambda n: Eq(4 * (100 + n ** 2) - (2 * n - 1) * ((100 + (n + 1) ** 2) - (100 + n ** 2)), lit(401))), lambda q: q.ring())
            p.rw_at('s4', 'e')
            p.have('h401', p.T('', lambda: v.pos(lit(401))), lambda q: q.linarith())
            p.exact(('Z.le_of_dvd', 'd', lit(401), 'h401', 's4'))
            p.split(); p.linarith()
            p.split(); p.exists(lit(100)); p.ring()
            p.split(); p.exists(lit(101)); p.ring()
            p.intro('e', 'h1', 'h2')
            p.have('s', p.T('e', lambda e: D(e, (100 + lit(201) * lit(201)) - (100 + lit(200) * lit(200)))), lambda q: q.exact(('Z.dvd_sub', 'e', 100 + lit(201) * lit(201), 100 + lit(200) * lit(200), 'h2', 'h1')))
            p.have('z', p.T('', lambda: Eq((100 + lit(201) * lit(201)) - (100 + lit(200) * lit(200)), lit(401))), lambda q: q.ring())
            p.rw_at('s', 'z'); p.exact('s')


@problem('aime1989_p9', text='A positive integer n satisfies 133^5 + 110^5 + 84^5 + 27^5 = n^5. Find n.', source='AIME 1989 P9', year=1989, topics=('number-theory', 'inequality'), level=2, kind='answer', answer='n = 144',
         origin='contest', verified='answer-checked',
         spec=Spec(('Z',), (lambda n: n > 0, lambda n: n ** 5 == 133 ** 5 + 110 ** 5 + 84 ** 5 + 27 ** 5), lambda n: n == 144, domains=(list(range(-3, 8)) + list(range(140, 150)),),
                   hatoms=[V(0), (lambda n: n ** 5, lambda n: 133 ** 5 + 110 ** 5 + 84 ** 5 + 27 ** 5)], atoms=[(lambda n: n, lambda n: 144)]),
         notes="Euler's counterexample to his conjecture: 144^5 is the sum, and the fifth power is strictly increasing on the non-negative integers")
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1989_p9', [v.Z], [lambda n: v.pos(n), lambda n: Eq(n ** 5, lit(133) ** 5 + lit(110) ** 5 + lit(84) ** 5 + lit(27) ** 5)], lambda n: Eq(n, lit(144)))
        def _(p):
            p.intro('n', 'hn', 'h')
            p.have('h5', p.T('n', lambda n: Eq(n ** 5, lit(144) ** 5)), lambda q: q.linarith())
            p.have('hn0', p.T('n', v.nonneg), lambda q: q.linarith())
            p.obtain(['c', 'c'], ('Z.le_or_lt', 'n', lit(143)))
            p.have('m1', ('Z.pow_le_pow_left', 'n', lit(143), v.nlit(5), 'hn0', 'c'))
            p.exfalso(); p.linarith()
            p.obtain(['d', 'd'], ('Z.le_or_lt', 'n', lit(144)))
            p.linarith()
            p.have('h145', p.T('n', lambda n: v.le(lit(145), n)), lambda q: q.linarith())
            p.have('z145', p.T('', lambda: v.nonneg(lit(145))), lambda q: q.linarith())
            p.have('m2', ('Z.pow_le_pow_left', lit(145), 'n', v.nlit(5), 'z145', 'h145'))
            p.exfalso(); p.linarith()


@problem('aime1997_p3', text='Sarah meant to multiply a two-digit number x by a three-digit number y, but left out the multiplication sign and wrote x to the left of y, forming a five-digit number. This number is exactly nine times the product she should have obtained. What is x + y?',
         source='AIME 1997 P3', year=1997, topics=('number-theory', 'diophantine'), level=3, kind='answer', answer='x + y = 126 (x = 14, y = 112)', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z'), (lambda x, y: 10 <= x, lambda x, y: x <= 99, lambda x, y: 100 <= y, lambda x, y: y <= 999, lambda x, y: 1000 * x + y == 9 * x * y), lambda x, y: x + y == 126,
                   domains=(list(range(8, 20)) + [50, 99, 100], list(range(98, 103)) + list(range(110, 115)) + [999, 1000]),
                   hatoms=[(lambda x, y: 10, lambda x, y: x), (lambda x, y: x, lambda x, y: 99), (lambda x, y: 100, lambda x, y: y), (lambda x, y: y, lambda x, y: 999), (lambda x, y: 1000 * x + y, lambda x, y: 9 * x * y)],
                   atoms=[(lambda x, y: x + y, lambda x, y: 126)]),
         notes='the five-digit number is 1000 x + y; (9x - 1)(9y - 1000) = 1000 and 9x - 1 = 125 is the only divisor of 1000 that is 8 modulo 9 and fits')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1997_p3', [v.Z, v.Z], [lambda x, y: v.le(lit(10), x), lambda x, y: v.le(x, lit(99)), lambda x, y: v.le(lit(100), y), lambda x, y: v.le(y, lit(999)), lambda x, y: Eq(1000 * x + y, 9 * x * y)],
                   lambda x, y: Eq(x + y, lit(126)))
        def _(p):
            p.intro('x', 'y', 'hx1', 'hx2', 'hy1', 'hy2', 'h')
            K.let(p, 'u', 'x', lambda x: 9 * x - 1); K.let(p, 't', 'y', lambda y: 9 * y - 1000)
            p.have('hu', p.T('u', v.pos), lambda q: q.linarith())
            p.have('e', p.T('u t', lambda u, t: Eq(u * t, lit(1000))), lambda q: q.linear_combination((q.h('t'), 'u_def'), (q.T('x', lambda x: 9 * x - 1), 't_def'), (lit(-9), 'h')))
            p.have('ht', p.T('t', v.pos), lambda q: q.nlinarith())
            def pair(p, d, c):
                if (d + 1) % 9 != 0: K.no_multiple(p, p.h('x'), 9, d + 1); return
                xv = (d + 1) // 9
                if not 10 <= xv <= 99: p.exfalso(); p.linarith(); return
                if (c + 1000) % 9 != 0: K.no_multiple(p, p.h('y'), 9, c + 1000); return
                yv = (c + 1000) // 9
                p.linarith()
            K.divisor_cases(p, 'u', 't', 'e', 1000, pair)


@problem('aime1990_p3', text='Let P_1 be a regular r-gon and P_2 a regular s-gon with r >= s >= 3 such that each interior angle of P_1 is 59/58 as large as each interior angle of P_2. What is the largest possible value of s?',
         source='AIME 1990 P3', year=1990, topics=('number-theory', 'diophantine'), level=3, kind='answer', answer='s = 117 (with r = 13572)', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z'), (lambda r, s: 3 <= s, lambda r, s: s <= r, lambda r, s: 58 * s * (r - 2) == 59 * r * (s - 2)), lambda r, s: s <= 117,
                   domains=([-1, 3, 4, 5, 116, 120, 356, 1595, 3306, 6728, 13572, 13573], [-1, 2, 3, 4, 59, 60, 89, 110, 114, 116, 117, 118]),
                   hatoms=[(lambda r, s: 3, lambda r, s: s), (lambda r, s: s, lambda r, s: r), (lambda r, s: 58 * s * (r - 2), lambda r, s: 59 * r * (s - 2))],
                   atoms=[(lambda r, s: s, lambda r, s: 117), (lambda r, s: 58 * 117 * (13572 - 2), lambda r, s: 59 * 13572 * (117 - 2))]),
         notes="the interior angle of a regular n-gon is 180 (n - 2) / n, which gives (r + 116)(s - 118) = -13688; the formalisation also records that s = 117 occurs, with r = 13572")
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1990_p3', [v.Z, v.Z], [lambda r, s: v.le(lit(3), s), lambda r, s: v.le(s, r), lambda r, s: Eq(58 * s * (r - 2), 59 * r * (s - 2))],
                   lambda r, s: v.le(s, lit(117)) & Eq(58 * lit(117) * (lit(13572) - 2), 59 * lit(13572) * (lit(117) - 2)))
        def _(p):
            p.intro('r', 's', 'h3', 'hsr', 'h')
            p.split()
            p.have('key', p.T('r s', lambda r, s: Eq((r + 116) * (s - 118), lit(-13688))), lambda q: q.linear_combination((-1, 'h')))
            p.obtain(['c', 'c'], ('Z.le_or_lt', 's', lit(117)))
            p.exact('c')
            p.exfalso(); p.nlinarith()
            p.ring()


@problem('aime1989_p1', text='Compute the positive integer n whose square is 31 * 30 * 29 * 28 + 1.', source='AIME 1989 P1', year=1989, topics=('arithmetic', 'algebra'), level=1, kind='answer', answer='n = 869',
         origin='contest', verified='answer-checked',
         spec=Spec(('Z',), (lambda n: n > 0, lambda n: n * n == 31 * 30 * 29 * 28 + 1), lambda n: n == 869, domains=(list(range(-2, 6)) + list(range(866, 873)),),
                   hatoms=[V(0), (lambda n: n * n, lambda n: 31 * 30 * 29 * 28 + 1)], atoms=[(lambda n: n, lambda n: 869)]),
         notes='the original asks for the square root of 31 * 30 * 29 * 28 + 1; x (x + 1)(x + 2)(x + 3) + 1 = (x^2 + 3x + 1)^2 with x = 28')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1989_p1', [v.Z], [lambda n: v.pos(n), lambda n: Eq(n ** 2, lit(31) * 30 * 29 * 28 + 1)], lambda n: Eq(n, lit(869)))
        def _(p):
            p.intro('n', 'hn', 'h')
            p.have('e', p.T('n', lambda n: Eq(n ** 2, lit(869) ** 2)), lambda q: q.linarith())
            p.obtain(['c', 'c'], ('Z.sq_eq_sq', 'n', lit(869), 'e'))
            p.exact('c')
            p.exfalso(); p.linarith()


@problem('aime1989_p4', text='Five positive integers a < b < c < d < e are consecutive. The sum b + c + d is a perfect square and the sum a + b + c + d + e is a perfect cube. What is the smallest possible value of c?',
         source='AIME 1989 P4', year=1989, topics=('number-theory', 'prime', 'divisibility'), level=4, kind='answer', answer='c = 675', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z', 'Z'), (lambda c, m, k: 2 < c, lambda c, m, k: 3 * c == m * m, lambda c, m, k: 5 * c == k ** 3), lambda c, m, k: 675 <= c,
                   domains=([1, 2, 3, 5, 12, 675, 676], [-45, -6, -3, 0, 3, 6, 45, 46], [-5, 0, 5, 10, 15, 16]),
                   hatoms=[(lambda c, m, k: 2, lambda c, m, k: c), (lambda c, m, k: 3 * c, lambda c, m, k: m * m), (lambda c, m, k: 5 * c, lambda c, m, k: k ** 3)],
                   atoms=[(lambda c, m, k: 675, lambda c, m, k: c), (lambda c, m, k: 3 * 675, lambda c, m, k: 45 ** 2), (lambda c, m, k: 5 * 675, lambda c, m, k: 15 ** 3)],
                   skip_mutants={'constant 45 -> -45 in ((Z.mul 3 675)': 'a square does not see the sign of its base: (-45)^2 = 45^2'}),
         notes='the middle one c; b + c + d = 3c = m^2 and a + .. + e = 5c = k^3; 3 | m, 5 | k, then 3 | s where k = 5s, and c = 675 u^3. The positivity of a is c > 2. The formalisation also records that c = 675 works')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1989_p4', [v.Z, v.Z, v.Z], [lambda c, m, k: v.lt(lit(2), c), lambda c, m, k: Eq(3 * c, m ** 2), lambda c, m, k: Eq(5 * c, k ** 3)],
                   lambda c, m, k: v.le(lit(675), c) & (Eq(3 * lit(675), lit(45) ** 2) & Eq(5 * lit(675), lit(15) ** 3)))
        def _(p):
            p.intro('c', 'm', 'k', 'hc', 'hm', 'hk')
            p.split()
            p.have('d3', p.T('c m', lambda c, m: D(lit(3), m * m)), lambda q: (q.exists(q.h('c')), q.linear_combination((-1, 'hm'))))
            p.have('d3m', p.T('m', lambda m: D(lit(3), m)), lambda q: q.exact(('Z.prime_dvd_sq', lit(3), 'Z.prime_three', 'm', 'd3')))
            p.cases('d3m', names=['t', 'ht'])
            p.have('ct', p.T('c t', lambda c, t: Eq(c, 3 * t * t)), lambda q: (q.have('c9', q.T('c t', lambda c, t: Eq(3 * c, 9 * t * t)), lambda r: r.linear_combination((1, 'hm'), (r.T('m t', lambda m, t: m + 3 * t), 'ht'))), q.linarith()))
            p.have('d5', p.T('c k', lambda c, k: D(lit(5), k ** 3)), lambda q: (q.exists(q.h('c')), q.linear_combination((-1, 'hk'))))
            p.have('d5k', p.T('k', lambda k: D(lit(5), k)), lambda q: q.exact(('Z.prime_dvd_pow', lit(5), 'Z.prime_five', 'k', v.nlit(3), 'd5')))
            p.cases('d5k', names=['s', 'hs'])
            p.have('cs', p.T('c s', lambda c, s: Eq(c, 25 * s ** 3)),
                   lambda q: (q.have('c5', q.T('c s', lambda c, s: Eq(5 * c, 125 * s ** 3)), lambda r: r.linear_combination((1, 'hk'), (r.T('k s', lambda k, s: k ** 2 + 5 * k * s + 25 * s ** 2), 'hs'))), q.linarith()))
            p.have('d3s', p.T('s t', lambda s, t: D(lit(3), 25 * s ** 3)), lambda q: (q.exists(q.T('t', lambda t: t * t)), q.linarith()))
            p.obtain(['c3', 'c3'], ('Z.prime_dvd_mul', lit(3), 'Z.prime_three', lit(25), p.T('s', lambda s: s * w['Z.sqr'](s)), 'd3s'))
            p.exfalso(); p.apply('not_dvd_between', lit(3), lit(8), lit(25)); p.linarith(); p.linarith(); p.linarith(); p.exact('c3')
            p.have('d3s2', p.T('s', lambda s: D(lit(3), s)), lambda q: q.exact(('Z.prime_dvd_pow', lit(3), 'Z.prime_three', 's', v.nlit(3), 'c3')))
            p.cases('d3s2', names=['u', 'hu'])
            p.have('cu', p.T('c u', lambda c, u: Eq(c, 675 * u ** 3)), lambda q: q.linear_combination((1, 'cs'), (q.T('s u', lambda s, u: 25 * (s ** 2 + 3 * s * u + 9 * u ** 2)), 'hu')))
            p.obtain(['u0', 'u0'], ('Z.le_or_lt', 'u', lit(0)))
            p.exfalso(); p.nlinarith(('Z.sq_nonneg', p.h('u')))
            p.have('u1', p.T('u', lambda u: v.le(lit(1), u)), lambda q: q.linarith())
            p.have('u3', ('Z.one_le_pow', 'u', v.nlit(3), 'u1'))
            p.linarith()
            p.split(); p.ring(); p.ring()


@problem('aime1989_p7', text='Adding the same integer k to each of 36, 300 and 596 gives the squares of three consecutive terms of an arithmetic progression of integers. Find k.', source='AIME 1989 P7', year=1989,
         topics=('algebra', 'diophantine'), level=3, kind='answer', answer='k = 925', origin='adapted', verified='answer-checked',
         spec=Spec(('Z', 'Z', 'Z'), (lambda a, d, k: a ** 2 == k + 36, lambda a, d, k: (a + d) ** 2 == k + 300, lambda a, d, k: (a + 2 * d) ** 2 == k + 596), lambda a, d, k: k == 925,
                   domains=([-31, -30, -1, 0, 1, 30, 31, 32], list(range(-5, 6)), [0, 1, 924, 925, 926]),
                   hatoms=[(lambda a, d, k: a ** 2, lambda a, d, k: k + 36), (lambda a, d, k: (a + d) ** 2, lambda a, d, k: k + 300), (lambda a, d, k: (a + 2 * d) ** 2, lambda a, d, k: k + 596)],
                   atoms=[(lambda a, d, k: k, lambda a, d, k: 925)]),
         notes='adapted: the contest allows real terms; here the terms a, a + d, a + 2d are integers. Subtracting the equations gives d^2 = 16 and a d = 124, so 16 a^2 = (a d)^2 = 15376, a^2 = 961 and k = 925')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1989_p7', [v.Z, v.Z, v.Z], [lambda a, d, k: Eq(a ** 2, k + 36), lambda a, d, k: Eq((a + d) ** 2, k + 300), lambda a, d, k: Eq((a + 2 * d) ** 2, k + 596)], lambda a, d, k: Eq(k, lit(925)))
        def _(p):
            p.intro('a', 'd', 'k', 'h1', 'h2', 'h3')
            p.have('e1', p.T('d', lambda d: Eq(2 * d ** 2, lit(32))), lambda q: q.linear_combination((1, 'h3'), (-2, 'h2'), (1, 'h1')))
            p.have('hd', p.T('d', lambda d: Eq(d ** 2, lit(16))), lambda q: q.linarith())
            p.have('e2', p.T('a d', lambda a, d: Eq(2 * (a * d), lit(248))), lambda q: q.linear_combination((1, 'h2'), (-1, 'h1'), (-1, 'hd')))
            p.have('had', p.T('a d', lambda a, d: Eq(a * d, lit(124))), lambda q: q.linarith())
            p.have('e3', p.T('a', lambda a: Eq(16 * a ** 2, lit(15376))), lambda q: q.linear_combination((q.T('a', lambda a: -(a ** 2)), 'hd'), (q.T('a d', lambda a, d: a * d + 124), 'had')))
            p.linarith()


@problem('aime1992_p3', text='A tennis player computes her win ratio by dividing the number of matches she has won by the total number of matches she has played. At the start of a weekend her win ratio is exactly 1/2. During the weekend she plays four matches, winning three and losing one. At the end of the weekend her win ratio is greater than 0.503. What is the largest number of matches she could have won before the weekend began?',
         source='AIME 1992 P3', year=1992, topics=('algebra', 'bounding', 'arithmetic'), level=2, kind='answer', answer='164', origin='contest', verified='answer-checked',
         spec=Spec(('Z',), (lambda w: 0 <= w, lambda w: 503 * (2 * w + 4) < 1000 * (w + 3)), lambda w: w <= 164, domains=(list(range(-2, 8)) + list(range(160, 170)),),
                   hatoms=[V(0), (lambda w: 503 * (2 * w + 4), lambda w: 1000 * (w + 3))], atoms=[(lambda w: w, lambda w: 164), (lambda w: 503 * (2 * 164 + 4), lambda w: 1000 * (164 + 3))]),
         notes='w wins in 2w matches before the weekend; the ratio afterwards (w + 3) / (2w + 4) > 503/1000 gives 6 w < 988; the formalisation records that 164 satisfies it')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1992_p3', [v.Z], [lambda w_: v.nonneg(w_), lambda w_: v.lt(503 * (2 * w_ + 4), 1000 * (w_ + 3))], lambda w_: v.le(w_, lit(164)) & v.lt(503 * (2 * lit(164) + 4), 1000 * (lit(164) + 3)))
        def _(p):
            p.intro('w', 'h0', 'h')
            p.split()
            p.obtain(['c', 'c'], ('Z.le_or_lt', 'w', lit(164)))
            p.exact('c')
            p.exfalso(); p.linarith()
            p.linarith()


@problem('aime1991_p13', text='A drawer contains a mixture of red socks and blue socks, at most 1991 in all. When two socks are selected randomly without replacement, the probability that both are red or both are blue is exactly 1/2. What is the largest possible number of red socks in the drawer?',
         source='AIME 1991 P13', year=1991, topics=('algebra', 'bounding', 'diophantine'), level=3, kind='answer', answer='990', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z'), (lambda r, b: 0 <= r, lambda r, b: 0 <= b, lambda r, b: 2 <= r + b, lambda r, b: r + b <= 1991, lambda r, b: 2 * (r * (r - 1) + b * (b - 1)) == (r + b) * (r + b - 1)), lambda r, b: r <= 990,
                   domains=([0, 1, 2, 3, 6, 10, 990, 991], [0, 1, 3, 6, 10, 946, 947]),
                   hatoms=[(lambda r, b: 0, lambda r, b: r), (lambda r, b: 0, lambda r, b: b), (lambda r, b: 2, lambda r, b: r + b), (lambda r, b: r + b, lambda r, b: 1991),
                           (lambda r, b: 2 * (r * (r - 1) + b * (b - 1)), lambda r, b: (r + b) * (r + b - 1))],
                   atoms=[(lambda r, b: r, lambda r, b: 990), (lambda r, b: 2 * (990 * 989 + 946 * 945), lambda r, b: (990 + 946) * (990 + 946 - 1)), (lambda r, b: 990 + 946, lambda r, b: 1991)]),
         notes='with r red and b blue socks the probability is (r(r-1) + b(b-1)) / ((r+b)(r+b-1)); it equals 1/2 exactly when (r - b)^2 = r + b. The formalisation also records that r = 990, b = 946 works')
def _(w, b_):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1991_p13', [v.Z, v.Z], [lambda r, b: v.nonneg(r), lambda r, b: v.nonneg(b), lambda r, b: v.le(lit(2), r + b), lambda r, b: v.le(r + b, lit(1991)), lambda r, b: Eq(2 * (r * (r - 1) + b * (b - 1)), (r + b) * (r + b - 1))],
                   lambda r, b: v.le(r, lit(990)) & (Eq(2 * (lit(990) * 989 + lit(946) * 945), (lit(990) + 946) * (lit(990) + 946 - 1)) & v.le(lit(990) + 946, lit(1991))))
        def _(p):
            p.intro('r', 'b', 'hr', 'hb', 'hs', 'hs2', 'h')
            p.split()
            p.have('key', p.T('r b', lambda r, b: Eq((r - b) ** 2, r + b)), lambda q: q.linear_combination((1, 'h')))
            K.let(p, 't', 'r b', lambda r, b: r - b)
            p.have('kt', p.T('t r b', lambda t, r, b: Eq(t ** 2, r + b)), lambda q: (q.rw('t_def'), q.exact('key')))
            p.have('t1', p.T('t', lambda t: v.le(t, lit(44))), lambda q: (q.obtain(['c', 'c'], ('Z.le_or_lt', q.h('t'), lit(44))), q.exact('c'), q.exfalso(), q.nlinarith()))
            p.have('t2', p.T('t', lambda t: v.le(lit(-44), t)), lambda q: (q.obtain(['c', 'c'], ('Z.le_or_lt', lit(-44), q.h('t'))), q.exact('c'), q.exfalso(), q.nlinarith()))
            p.nlinarith()
            p.split(); p.ring(); p.linarith()


@problem('aime1993_p6', text='What is the smallest positive integer that can be written as a sum of nine consecutive integers, as a sum of ten consecutive integers, and as a sum of eleven consecutive integers?', source='AIME 1993 P6',
         year=1993, topics=('number-theory', 'divisibility', 'modular'), level=3, kind='answer', answer='495', origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z', 'Z', 'Z'), (lambda n, m, k, j: n > 0, lambda n, m, k, j: n == 9 * m + 36, lambda n, m, k, j: n == 10 * k + 45, lambda n, m, k, j: n == 11 * j + 55), lambda n, m, k, j: 495 <= n,
                   domains=([-1, 45, 99, 396, 495, 496], [-4, 0, 50, 51, 52], [-5, 0, 44, 45, 46], [-5, 0, 39, 40, 41]),
                   hatoms=[(lambda n, m, k, j: 0, lambda n, m, k, j: n), (lambda n, m, k, j: n, lambda n, m, k, j: 9 * m + 36), (lambda n, m, k, j: n, lambda n, m, k, j: 10 * k + 45), (lambda n, m, k, j: n, lambda n, m, k, j: 11 * j + 55)],
                   atoms=[(lambda n, m, k, j: 495, lambda n, m, k, j: n), (lambda n, m, k, j: 495, lambda n, m, k, j: 9 * 51 + 36), (lambda n, m, k, j: 495, lambda n, m, k, j: 10 * 45 + 45), (lambda n, m, k, j: 495, lambda n, m, k, j: 11 * 40 + 55)]),
         notes='m, k, j are the first terms of the three runs: the sums are 9m + 36, 10k + 45 and 11 j + 55. 9 and 11 divide n, so 99 divides n; n = 99 t = 10 k + 45 needs t >= 5. The formalisation also records the three runs for 495')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; D = v.dvd
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1993_p6', [v.Z, v.Z, v.Z, v.Z], [lambda n, m, k, j: v.pos(n), lambda n, m, k, j: Eq(n, 9 * m + 36), lambda n, m, k, j: Eq(n, 10 * k + 45), lambda n, m, k, j: Eq(n, 11 * j + 55)],
                   lambda n, m, k, j: v.le(lit(495), n) & AND(Eq(lit(495), 9 * lit(51) + 36), Eq(lit(495), 10 * lit(45) + 45), Eq(lit(495), 11 * lit(40) + 55)))
        def _(p):
            p.intro('n', 'm', 'k', 'j', 'hn', 'h1', 'h2', 'h3')
            p.split()
            p.have('d9', p.T('n m', lambda n, m: D(lit(9), n)), lambda q: (q.exists(q.T('m', lambda m: m + 4)), q.linear_combination((1, 'h1'))))
            p.have('d11', p.T('n j', lambda n, j: D(lit(11), n)), lambda q: (q.exists(q.T('j', lambda j: j + 5)), q.linear_combination((1, 'h3'))))
            p.have('cop', p.T('', lambda: w['Z.coprime'](lit(9), lit(11))), lambda q: (q.exists(lit(5)), q.exists(lit(-4)), q.ring()))
            p.have('g', p.T('', lambda: w['Z.is_gcd'](lit(9), lit(11), v.one)), lambda q: q.exact(('Z.is_gcd_one_of_coprime', lit(9), lit(11), 'cop')))
            p.have('d99', p.T('n', lambda n: D(lit(9) * lit(11), n)), lambda q: q.exact(('dvd_coprime_product', lit(9), lit(11), q.h('n'), 'g', 'd9', 'd11')))
            p.cases('d99', names=['t', 'ht'])
            p.obtain(['c', 'c'], ('Z.le_or_lt', 't', lit(4)))
            p.have('t1', p.T('t', lambda t: v.le(lit(1), t)), lambda q: q.linarith())
            p.interval_cases('t', 1, 4)
            for tv in range(1, 5): K.no_multiple(p, p.h('k'), 10, 99 * tv - 45)
            p.linarith()
            p.split(); p.ring(); p.split(); p.ring(); p.ring()


@problem('aime1986_p4', text='Determine 3 x4 + 2 x5 if x1, x2, x3, x4 and x5 satisfy the system 2 x1 + x2 + x3 + x4 + x5 = 6, x1 + 2 x2 + x3 + x4 + x5 = 12, x1 + x2 + 2 x3 + x4 + x5 = 24, x1 + x2 + x3 + 2 x4 + x5 = 48 and x1 + x2 + x3 + x4 + 2 x5 = 96.',
         source='AIME 1986 P4', year=1986, topics=('algebra',), level=1, kind='answer', answer='3 x4 + 2 x5 = 181', origin='contest', verified='recalled',
         spec=Spec(('Z',) * 5, tuple((lambda x1, x2, x3, x4, x5, i=i, r=r: x1 + x2 + x3 + x4 + x5 + [x1, x2, x3, x4, x5][i] == r) for i, r in enumerate((6, 12, 24, 48, 96))), lambda x1, x2, x3, x4, x5: 3 * x4 + 2 * x5 == 181,
                   domains=([-25, 0, 1], [-19, 0, 1], [-7, 0, 1], [17, 0, 1], [65, 0, 1]),
                   hatoms=[(lambda x1, x2, x3, x4, x5, i=i: x1 + x2 + x3 + x4 + x5 + [x1, x2, x3, x4, x5][i], lambda x1, x2, x3, x4, x5, r=r: r) for i, r in enumerate((6, 12, 24, 48, 96))],
                   atoms=[(lambda x1, x2, x3, x4, x5: 3 * x4 + 2 * x5, lambda x1, x2, x3, x4, x5: 181)]),
         notes='adding the five equations gives x1 + .. + x5 = 31, then each x_i follows')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        def eq_i(i, r): return lambda x1, x2, x3, x4, x5: Eq(x1 + x2 + x3 + x4 + x5 + [x1, x2, x3, x4, x5][i], lit(r))
        @K.theorem(w, 'aime1986_p4', [v.Z] * 5, [eq_i(i, r) for i, r in enumerate((6, 12, 24, 48, 96))], lambda x1, x2, x3, x4, x5: Eq(3 * x4 + 2 * x5, lit(181)))
        def _(p):
            p.intro('x1', 'x2', 'x3', 'x4', 'x5', 'h1', 'h2', 'h3', 'h4', 'h5')
            p.linarith()


@problem('aime1983_p6', text='Let a_n = 6^n + 8^n. Determine the remainder on dividing a_83 by 49.', source='AIME 1983 P6', year=1983, topics=('number-theory', 'modular', 'arithmetic'), level=2, kind='answer', answer='35',
         origin='contest', verified='answer-checked',
         spec=Spec(('Z', 'Z'), (lambda q, r: 6 ** 83 + 8 ** 83 == 49 * q + r, lambda q, r: 0 <= r, lambda q, r: r < 49), lambda q, r: r == 35,
                   domains=([0, 1, (6 ** 83 + 8 ** 83 - 35) // 49, (6 ** 83 + 8 ** 83 - 35) // 49 + 1], list(range(-2, 50))),
                   hatoms=[(lambda q, r: 6 ** 83 + 8 ** 83, lambda q, r: 49 * q + r), (lambda q, r: 0, lambda q, r: r), (lambda q, r: r, lambda q, r: 49)], atoms=[(lambda q, r: r, lambda q, r: 35)]),
         notes='with (1 + x)^n = 1 + n x + x^2 k for x = 7 and x = -7: 8^83 = 582 + 49 k1 and -6^83 = -580 + 49 k2, so the sum is 1162 + 49 (k1 - k2) and the remainder of 1162 by 49 is 35')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1983_p6', [v.Z, v.Z], [lambda q, r: Eq(lit(6) ** 83 + lit(8) ** 83, 49 * q + r), lambda q, r: v.nonneg(r), lambda q, r: v.lt(r, lit(49))], lambda q, r: Eq(r, lit(35)))
        def _(p):
            p.intro('q', 'r', 'h', 'h0', 'h1')
            p.obtain(['k1', 'hk1'], ('Z.pow_one_add', lit(7), v.nlit(83)))
            p.obtain(['k2', 'hk2'], ('Z.pow_one_add', -lit(7), v.nlit(83)))
            K.let(p, 't', 'q k1 k2', lambda q, k1, k2: 23 + k1 - k2 - q)
            p.have('et', p.T('r t', lambda r, t: Eq(r, 35 + 49 * t)), lambda q: q.linarith())
            p.obtain(['c', 'c'], ('Z.le_or_lt', 't', -lit(1)))
            p.exfalso(); p.linarith()
            p.obtain(['c2', 'c2'], ('Z.le_or_lt', lit(1), 't'))
            p.exfalso(); p.linarith()
            p.linarith()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ AIME: sums, sequences and functions
@problem('aime1986_p6', text='The pages of a book are numbered 1 through n. When the page numbers were added, one of them was mistakenly added twice, and the (incorrect) sum was 1986. What was the number of the page that was added twice?',
         source='AIME 1986 P6', year=1986, topics=('sum', 'arithmetic', 'bounding'), level=2, kind='answer', answer='33', origin='contest', verified='answer-checked',
         spec=spec(('N', 'Z'), [LE(1, X(1)), LE(X(1), X(0)), EQ(lambda n, g: sum(i + 1 for i in range(n)) + g, 1986)], EQ(X(1), 33), domains=([0, 1, 61, 62, 63], [0, 1, 2, 32, 33, 34, 61, 62, 63])),
         notes='1 + 2 + .. + n = n (n + 1) / 2 and the doubled page g is one of 1..n: n (n + 1) / 2 < 1986 <= n (n + 1) / 2 + n gives n = 62, and 1953 + g = 1986')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; zsum = w['Z.sum']; zof = w['Z.of_N']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1986_p6', [v.N, v.Z], [lambda n, g: v.le(lit(1), g), lambda n, g: v.le(g, zof(n)), lambda n, g: Eq(zsum(w.lam(v.Z, lambda i: i + 1), n) + g, lit(1986))], lambda n, g: Eq(g, lit(33)))
        def _(p):
            p.intro('n', 'g', 'h1', 'h2', 'h')
            p.have('hs', ('Z.sum_first', 'n'))
            p.have('hn', ('Z.nonneg_of_N', 'n'))
            p.obtain(['c', 'c'], ('Z.le_or_lt', p.T('n', lambda n: zof(n)), lit(61)))
            p.exfalso(); p.nlinarith()
            p.obtain(['c2', 'c2'], ('Z.le_or_lt', p.T('n', lambda n: zof(n)), lit(62)))
            p.have('hm', p.T('n', lambda n: Eq(zof(n), lit(62))), lambda q: q.linarith())
            p.rw_at('hs', 'hm')
            p.linarith()
            p.exfalso(); p.nlinarith()


def _sol1984(n): return n - 3 if n >= 1000 else (997 if n % 2 == 0 else 998)


_W1010 = range(-1010, 1011)
_F1984 = [_sol1984, lambda n: n - 3 if n >= 1000 else 997, lambda n: n - 3 if n >= 1000 else 998, lambda n: n - 3 if n >= 1000 else (998 if n % 2 == 0 else 997), lambda n: n - 3, lambda n: 997,
          lambda n: 5 if n == 1000 else _sol1984(n), lambda n: _sol1984(n) + (1 if n == 999 else 0)]


@problem('aime1984_p7', text='The function f is defined on the integers and satisfies f(n) = n - 3 if n >= 1000, and f(n) = f(f(n + 5)) if n < 1000. Find f(84).', source='AIME 1984 P7', year=1984,
         topics=('functional-equation', 'induction', 'case-analysis'), level=3, kind='answer', answer='f(84) = 997', origin='contest', verified='answer-checked',
         spec=spec(('fn',), [lambda f: all(f(n) == n - 3 for n in _W1010 if n >= 1000), lambda f: all(f(n) == f(f(n + 5)) for n in _W1010 if n < 1000)], EQ(lambda f: f(84), 997), hatoms=False,
                   windows={'z': 1010}, fns=_F1984),
         notes='f(999) = f(f(1004)) = f(1001) = 998, f(998) = 997, f(997) = 998; then f(n) = f(f(n + 5)) shows f(n) + f(n + 5) = 1995 below 995, and f(84) = f(94) = .. = f(994) = 997 in steps of ten')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z); zof = w['Z.of_N']
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1984_p7', [fZ], [lambda f: v.forall(lambda n: v.le(lit(1000), n) >> Eq(f(n), n - 3)), lambda f: v.forall(lambda n: v.lt(n, lit(1000)) >> Eq(f(n), f(f(n + 5))))], lambda f: Eq(f(lit(84)), lit(997)))
        def _(p):
            p.intro('f', 'h1', 'h2')
            known = {}                                                      # k -> the value of f k, for the facts  f<k> : f k = value

            def top(k):                                                     # k >= 1000: f k = k - 3
                p.have('f%d' % k, p.T('f', lambda f: Eq(f(lit(k)), lit(k - 3))),
                       lambda q: (q.have('t', q.T('f', lambda f: Eq(f(lit(k)), lit(k) - 3)), lambda r: (r.apply('h1', lit(k)), r.linarith())), q.linarith()))
                known[k] = k - 3

            def below(n):                                                   # n < 1000 with the values of f (n + 5) and of f at that value known: f n = f (f (n + 5)) = f u
                m = n + 5; u = known[m]
                p.have('b%d' % n, p.T('f', lambda f: Eq(f(lit(n)), f(f(lit(m))))), lambda q: (q.apply('h2', lit(n)), q.linarith()))
                p.rw_at('b%d' % n, 'f%d' % m)
                known[n] = known[u]
                p.have('f%d' % n, p.T('f', lambda f: Eq(f(lit(n)), lit(known[n]))), lambda q: q.linarith())

            for k in range(1000, 1005): top(k)
            for n in (999, 998, 997, 996, 995, 994): below(n)

            def val_script(q):                                              # every value below 1000 is 997 or 998: strong induction on m = 999 - n
                q.intro('mm', 'hmm'); q.induction_on('Z.strong_induction', 'mm', extra=(lit(0),)); q.intro('k', 'hk', 'ih')
                q.obtain(['c', 'c'], ('Z.le_or_lt', 'k', lit(4)))
                q.interval_cases('k', 0, 4)
                for kv in range(5):
                    q.have('nv', q.T('k', lambda k, kv=kv: Eq(lit(999) - k, lit(999 - kv))), lambda r: r.linarith())
                    q.rw('nv')
                    (q.right if known[999 - kv] == 998 else q.left)()
                    q.exact('f%d' % (999 - kv))
                q.have('hk5', q.T('k', lambda k: v.le(lit(0), k - 5)), lambda r: r.linarith())
                q.have('hk6', q.T('k', lambda k: v.lt(k - 5, k)), lambda r: r.linarith())
                q.have('hv', ('ih', q.T('k', lambda k: k - 5), 'hk5', 'hk6'))
                q.have('b0', q.T('f k', lambda f, k: Eq(f(lit(999) - k), f(f(lit(999) - k + 5)))), lambda r: (r.apply('h2', r.T('k', lambda k: lit(999) - k)), r.linarith()))
                q.have('e', q.T('k', lambda k: Eq(lit(999) - k + 5, lit(999) - (k - 5))), lambda r: r.ring())
                q.rw_at('b0', 'e')
                q.cases('hv', names=['d1', 'd2'])
                q.rw_at('b0', 'd1'); q.right(); q.linarith()
                q.rw_at('b0', 'd2'); q.left(); q.linarith()
                q.assumption()
            p.have('val', p.T('f', lambda f: v.forall(lambda m: v.le(lit(0), m) >> (Eq(f(lit(999) - m), lit(997)) | Eq(f(lit(999) - m), lit(998))))), val_script)

            def alt_script(q):                                              # f n + f (n + 5) = 1995 for n <= 994
                q.intro('n', 'hn')
                q.have('b', q.T('f n', lambda f, n: Eq(f(n), f(f(n + 5)))), lambda r: (r.apply('h2', r.h('n')), r.linarith()))
                q.have('hm', q.T('n', lambda n: v.le(lit(0), lit(994) - n)), lambda r: r.linarith())
                q.have('hv', ('val', q.T('n', lambda n: lit(994) - n), 'hm'))
                q.have('e', q.T('n', lambda n: Eq(lit(999) - (lit(994) - n), n + 5)), lambda r: r.ring())
                q.rw_at('hv', 'e')
                q.cases('hv', names=['d1', 'd2'])
                q.rw_at('b', 'd1'); q.linarith()
                q.rw_at('b', 'd2'); q.linarith()
            p.have('alt', p.T('f', lambda f: v.forall(lambda n: v.le(n, lit(994)) >> Eq(f(n) + f(n + 5), lit(1995)))), alt_script)

            def u_script(q):                                                # f (994 - 10 i) = 997 for every i
                q.intro('i'); q.induction_on('N.induction_on', 'i')
                q.rw('Z.of_N_zero'); q.exact('f994')
                q.intro('j', 'ih'); q.rw('Z.of_N_succ')
                q.have('hj', ('Z.nonneg_of_N', 'j'))
                q.have('l1', q.T('j', lambda j: v.le(lit(994) - 10 * (zof(j) + 1), lit(994))), lambda r: r.linarith())
                q.have('l2', q.T('j', lambda j: v.le(lit(994) - 10 * (zof(j) + 1) + 5, lit(994))), lambda r: r.linarith())
                q.have('a1', ('alt', q.T('j', lambda j: lit(994) - 10 * (zof(j) + 1)), 'l1'))
                q.have('a2', ('alt', q.T('j', lambda j: lit(994) - 10 * (zof(j) + 1) + 5), 'l2'))
                q.linarith()
            p.have('U', p.T('f', lambda f: v.F([v.N], lambda i: Eq(f(lit(994) - 10 * zof(i)), lit(997)))), u_script)
            p.have('u91', ('U', v.nlit(91)))
            p.have('e91', p.T('', lambda: Eq(lit(994) - 10 * zof(v.nlit(91)), lit(84))), lambda q: q.ring())
            p.rw_at('u91', 'e91')
            p.exact('u91')


def _per6(x, y): return lambda n: [x, y, y - x, -x, -y, x - y][(n - 1) % 6]


def _S(a, k): return sum(a(i + 1) for i in range(k))


def _bumped(delta):
    """the sequence 1: -999, 2: 493, whose recurrence fails by `delta` at n = 3 only (and below the start: 5)"""
    v = {1: -999, 2: 493}; v[3] = v[2] - v[1] + delta
    for i in range(4, 3010): v[i] = v[i - 1] - v[i - 2]
    return lambda n: v[n] if n >= 1 else 5


_F1985 = [_per6(-999, 493), _per6(0, 0), _per6(1, 0), _per6(0, 1), _per6(-999, 494), _per6(-998, 493), _per6(5, -7), lambda n: n, lambda n: 1,
          lambda n: _per6(-999, 493)(n) if n >= 1 else 5,           # the recurrence holds from n = 3 on, not below it
          _bumped(1)]                                                # fails at n = 3 only


@problem('aime1985_p5', text='A sequence of integers a_1, a_2, a_3, ... satisfies a_n = a_(n-1) - a_(n-2) for each n >= 3. What is the sum of the first 2001 terms if the sum of the first 1492 terms is 1985 and the sum of the first 1985 terms is 1492?',
         source='AIME 1985 P5', year=1985, topics=('sequence', 'sum', 'induction'), level=3, kind='answer', answer='986', origin='contest', verified='answer-checked',
         spec=spec(('fn', 'N', 'N', 'N'), [lambda a, n1, n2, n3: all(a(n) == a(n - 1) - a(n - 2) for n in R if n >= 3), EQ(X(1), 1492), EQ(X(2), 1985), EQ(X(3), 2001),
                                          EQ(lambda a, n1, n2, n3: _S(a, n1), 1985), EQ(lambda a, n1, n2, n3: _S(a, n2), 1492)],
                   EQ(lambda a, n1, n2, n3: _S(a, n3), 986), hatoms=False, fns=_F1985, domains=(None, [1492], [1985], [2001])),
         notes='the three lengths are named (n1 = 1492, n2 = 1985, n3 = 2001): a sum of a literal length would be unfolded by the kernel\'s conversion, whenever the engine tries to close a goal by a hypothesis. '
               'The sequence has period 6 and S(n) = a_2 - a_(n+2) (telescoping a_k = a_(k+1) - a_(k+2)); with a_1494 = a_6, a_1987 = a_1, a_2003 = a_5 the two given sums fix a_2 = 493 and S(2001) = 2 a_2')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z); zsum = w['Z.sum']; zof = w['Z.of_N']
    with v.ints():
        lit = v.lit
        def S(a, k): return zsum(w.lam(v.Z, lambda i: a(i + 1)), k)
        @K.theorem(w, 'aime1985_p5', [fZ, v.N, v.N, v.N], [lambda a, n1, n2, n3: v.forall(lambda n: v.le(lit(3), n) >> Eq(a(n), a(n - 1) - a(n - 2))), lambda a, n1, n2, n3: Eq(n1, v.nlit(1492)),
                                                       lambda a, n1, n2, n3: Eq(n2, v.nlit(1985)), lambda a, n1, n2, n3: Eq(n3, v.nlit(2001)),
                                                       lambda a, n1, n2, n3: Eq(S(a, n1), lit(1985)), lambda a, n1, n2, n3: Eq(S(a, n2), lit(1492))],
                   lambda a, n1, n2, n3: Eq(S(a, n3), lit(986)))
        def _(p):
            p.intro('a', 'n1', 'n2', 'n3', 'h', 'e1', 'e2', 'e3', 'h1', 'h2')

            def form_script(q):                                             # S(n) = a_2 - a_(n+2)
                q.intro('n'); q.induction_on('N.induction_on', 'n')
                q.rw('Z.sum_zero'); q.rw('Z.of_N_zero'); q.linarith()
                q.intro('m', 'ih'); q.rw('Z.sum_succ'); q.rw('Z.of_N_succ')
                q.have('hm', ('Z.nonneg_of_N', 'm'))
                q.have('r', q.T('a m', lambda a, m: Eq(a(zof(m) + 3), a(zof(m) + 3 - 1) - a(zof(m) + 3 - 2))), lambda r: (r.apply('h', r.T('m', lambda m: zof(m) + 3)), r.linarith()))
                q.linarith()
            p.have('form', p.T('a', lambda a: v.F([v.N], lambda n: Eq(S(a, n), a(lit(2)) - a(zof(n) + 2)))), form_script)

            def per6_script(q):                                             # a_(n+6) = a_n for n >= 1
                q.intro('n', 'hn')
                for d in range(2, 7):
                    q.have('r%d' % d, q.T('a n', lambda a, n, d=d: Eq(a(n + d), a(n + d - 1) - a(n + d - 2))), lambda r, d=d: (r.apply('h', r.T('n', lambda n: n + d)), r.linarith()))
                q.linarith()
            p.have('per6', p.T('a', lambda a: v.forall(lambda n: v.le(lit(1), n) >> Eq(a(n + 6), a(n)))), per6_script)

            def per_script(q):                                              # a_(n + 6 j) = a_n
                q.intro('j'); q.induction_on('N.induction_on', 'j')
                q.intro('n', 'hn'); q.rw('Z.of_N_zero'); q.linarith()
                q.intro('i', 'ih', 'n', 'hn'); q.rw('Z.of_N_succ')
                q.have('hi', ('Z.nonneg_of_N', 'i'))
                q.have('l', q.T('n i', lambda n, i: v.le(lit(1), n + 6 * zof(i))), lambda r: r.linarith())
                q.have('s', ('per6', q.T('n i', lambda n, i: n + 6 * zof(i)), 'l'))
                q.have('t', ('ih', 'n', 'hn'))
                q.linarith()
            p.have('per', p.T('a', lambda a: v.F([v.N], lambda j: v.forall(lambda n: v.le(lit(1), n) >> Eq(a(n + 6 * zof(j)), a(n))))), per_script)

            for i in (1, 2, 3):                                             # the lengths as integers: Z.of_N n_i = the numeral
                p.have('z%d' % i, p.T('n%d' % i, lambda x, i=i: Eq(zof(x), lit((1492, 1985, 2001)[i - 1]))), lambda q, i=i: K.rw_ring(q, 'e%d' % i))
                p.have('s%d' % i, ('form', 'n%d' % i))
                p.rw_at('s%d' % i, 'z%d' % i)
            for nm, j, n0 in (('p1494', 248, 6), ('p1987', 331, 1), ('p2003', 333, 5)):
                p.have('l' + nm, p.T('', lambda n0=n0: v.le(lit(1), lit(n0))), lambda q: q.linarith())
                p.have(nm, ('per', v.nlit(j), lit(n0), 'l' + nm))
            for d in range(3, 7):
                p.have('c%d' % d, p.T('a', lambda a, d=d: Eq(a(lit(d)), a(lit(d) - 1) - a(lit(d) - 2))), lambda q, d=d: (q.apply('h', lit(d)), q.linarith()))
            p.linarith()


_W8 = range(-8, 9)
def _q92(n): return (n - 19) * (n - 92) // 2


_F1992 = [_q92, lambda n: _q92(n) + 1, lambda n: (n - 18) * (n - 91) // 2, lambda n: n * (n - 1) // 2, lambda n: n, lambda n: n * n, lambda n: 0,
          lambda n: 0 if n in (19, 92) else (n * (n - 1) // 2 if n >= 1 else 5),               # the recurrence holds from n = 1 on, not below it; the zeros are right, a_1 = 0
          lambda n: 0 if n in (19, 92) else n * (n - 1) // 2 + (7 if n == 1 else 0),           # holds from n = 2 on; a_1 = 7
          lambda n: 0 if n in (19, 92) else (n * (n - 1) // 2 if n >= 0 else 5)]              # holds from n = 0 on


@problem('aime1992_p8', text='A sequence of integers a_1, a_2, a_3, ... has the property that for every n >= 1, (a_(n+2) - a_(n+1)) - (a_(n+1) - a_n) = 1. If a_19 = a_92 = 0, find a_1.',
         source='AIME 1992 P8', year=1992, topics=('sequence', 'induction', 'algebra'), level=3, kind='answer', answer='a_1 = 819', origin='adapted', verified='answer-checked',
         spec=spec(('fn',), [lambda a: all((a(n + 2) - a(n + 1)) - (a(n + 1) - a(n)) == 1 for n in _W8 if n >= 1), EQ(lambda a: a(19), 0), EQ(lambda a: a(92), 0)], EQ(lambda a: a(1), 819), hatoms=False, fns=_F1992),
         notes='adapted: the contest has real terms; here they are integers (the solution a_n = (n - 19)(n - 92)/2 is integral). The first differences grow by 1, so 2 a_n = 2 a_1 + 2 (n - 1) d + (n - 1)(n - 2) with d = a_2 - a_1; the two zeros fix d = -54 and a_1 = 819')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1992_p8', [fZ], [lambda a: v.forall(lambda n: v.le(lit(1), n) >> Eq((a(n + 2) - a(n + 1)) - (a(n + 1) - a(n)), lit(1))), lambda a: Eq(a(lit(19)), lit(0)), lambda a: Eq(a(lit(92)), lit(0))],
                   lambda a: Eq(a(lit(1)), lit(819)))
        def _(p):
            p.intro('a', 'h', 'h19', 'h92')

            def d_script(q):                                                # a_(n+1) - a_n = (a_2 - a_1) + (n - 1)
                q.intro('n', 'hn'); q.induction_on('Z.induction_from', 'n', extra=(lit(1),))
                q.linarith()
                q.intro('t', 'ht', 'ih')
                q.have('r', q.T('a t', lambda a, t: Eq((a(t + 2) - a(t + 1)) - (a(t + 1) - a(t)), lit(1))), lambda r: (r.apply('h', r.h('t')), r.linarith()))
                q.linarith()
                q.assumption()
            p.have('D', p.T('a', lambda a: v.forall(lambda n: v.le(lit(1), n) >> Eq(a(n + 1) - a(n), (a(lit(2)) - a(lit(1))) + (n - 1)))), d_script)

            def q_script(q):                                                # 2 a_n = 2 a_1 + 2 (n - 1) d + (n - 1)(n - 2)
                q.intro('n', 'hn'); q.induction_on('Z.induction_from', 'n', extra=(lit(1),))
                q.ring()
                q.intro('t', 'ht', 'ih')
                q.have('r', ('D', 't', 'ht'))
                q.linear_combination((2, 'r'), (1, 'ih'))
                q.assumption()
            p.have('Q', p.T('a', lambda a: v.forall(lambda n: v.le(lit(1), n) >> Eq(2 * a(n), 2 * a(lit(1)) + 2 * (n - 1) * (a(lit(2)) - a(lit(1))) + (n - 1) * (n - 2)))), q_script)
            for k in (19, 92):
                p.have('l%d' % k, p.T('', lambda k=k: v.le(lit(1), lit(k))), lambda q: q.linarith())
                p.have('q%d' % k, ('Q', lit(k), 'l%d' % k))
            p.linarith()


def _lcm2(x, y): return abs(x * y) // math.gcd(x, y) if x and y else 0


def _pos(x, y): return x > 0 and y > 0


_F1988 = [_lcm2, lambda x, y: math.gcd(x, y), lambda x, y: x * y, lambda x, y: max(x, y), lambda x, y: x, lambda x, y: 0, lambda x, y: x + y, lambda x, y: _lcm2(x, y) + 1,
          lambda x, y: 0 if (x, y) == (14, 52) else _lcm2(x, y),                                                       # all three properties hold on the window, f(14, 52) is wrong
          lambda x, y: 0 if (x, y) == (14, 52) else (_lcm2(x, y) if _pos(x, y) else 7),                                # ... and the properties fail off the positive quadrant (f(0, 0) = 7)
          lambda x, y: 0 if (x, y) == (14, 52) else (_lcm2(x, y) if _pos(x, y) else x),                                # ... asymmetrically
          lambda x, y: 0 if (x, y) == (14, 52) else _lcm2(x, y) * (5 if math.gcd(x, y) == 1 else 1),                  # f(1, 1) = 5: only the first property fails, at x = 1
          lambda x, y: _lcm2(x, y) if x >= 2 else y * y,                                                               # the third property fails at x = 1 only
          lambda x, y: 0 if y == 1 else _lcm2(x, y)]                                                                   # the third property fails at y = 1 only


@problem('aime1988_p8', text='The function f is defined on ordered pairs of positive integers and satisfies f(x, x) = x, f(x, y) = f(y, x) and (x + y) f(x, y) = y f(x, x + y). Calculate f(14, 52).', source='AIME 1988 P8', year=1988,
         topics=('functional-equation', 'arithmetic', 'case-analysis'), level=3, kind='answer', answer='f(14, 52) = 364', origin='adapted', verified='answer-checked',
         spec=spec(('fn',), [lambda f: all(f(x, x) == x for x in _W8 if x > 0), lambda f: all(f(x, y) == f(y, x) for x in _W8 for y in _W8 if x > 0 and y > 0),
                             lambda f: all((x + y) * f(x, y) == y * f(x, x + y) for x in _W8 for y in _W8 if x > 0 and y > 0)], EQ(lambda f: f(14, 52), 364), hatoms=False, fns=_F1988,
                   skip_mutants=['constant 0 -> 1 in (Z.lt 0 x2)', 'constant 0 -> 1 in (Z.lt 0 x3)', 'Z.lt -> Z.le in (Z.lt 0 x3)', 'constant 0 -> -1 in (Z.lt 0 x3)']),
         notes='adapted: f is a function Z x Z -> Z whose three properties are asked for positive arguments only (the guards x > 1 / y > 1 of the symmetry property, and x >= 0 of the third, change nothing: the instance (1, y) of the symmetry is the instance (y, 1), and at x = 0 the third property reads (0 + y) f(0, y) = y f(0, 0 + y), a tautology). The function is lcm; the chain (14, 52) <- (14, 38) <- (14, 24) <- (14, 10) = (10, 14) <- (10, 4) = (4, 10) <- (4, 6) <- (4, 2) = (2, 4) <- (2, 2) computes it')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; f2 = w.arrow(v.Z, w.arrow(v.Z, v.Z))
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1988_p8', [f2], [lambda f: v.forall(lambda x: v.pos(x) >> Eq(f(x, x), x)),
                                            lambda f: v.F([v.Z, v.Z], lambda x, y: v.pos(x) >> (v.pos(y) >> Eq(f(x, y), f(y, x)))),
                                            lambda f: v.F([v.Z, v.Z], lambda x, y: v.pos(x) >> (v.pos(y) >> Eq((x + y) * f(x, y), y * f(x, x + y))))], lambda f: Eq(f(lit(14), lit(52)), lit(364)))
        def _(p):
            p.intro('f', 'h1', 'h2', 'h3')

            def e3(x, y):                                                   # (x + y) f(x, y) = y f(x, x + y)
                p.have('e%d_%d' % (x, y), p.T('f', lambda f: Eq((lit(x) + lit(y)) * f(lit(x), lit(y)), lit(y) * f(lit(x), lit(x) + lit(y)))), lambda q: (q.apply('h3', lit(x), lit(y)), q.linarith(), q.linarith()))

            def s2(x, y):                                                   # f(x, y) = f(y, x)
                p.have('s%d_%d' % (x, y), p.T('f', lambda f: Eq(f(lit(x), lit(y)), f(lit(y), lit(x)))), lambda q: (q.apply('h2', lit(x), lit(y)), q.linarith(), q.linarith()))
            p.have('d22', p.T('f', lambda f: Eq(f(lit(2), lit(2)), lit(2))), lambda q: (q.apply('h1', lit(2)), q.linarith()))
            e3(2, 2); s2(4, 2); e3(4, 2); e3(4, 6); s2(10, 4); e3(10, 4); s2(14, 10); e3(14, 10); e3(14, 24); e3(14, 38)
            p.linarith()


@problem('aime1998_p14', text='An m x n x p rectangular box has half the volume of an (m + 2) x (n + 2) x (p + 2) rectangular box, where m, n and p are positive integers with m <= n <= p. What is the largest possible value of p?',
         source='AIME 1998 P14', year=1998, topics=('number-theory', 'diophantine', 'bounding', 'case-analysis'), level=4, kind='answer', answer='p = 130', origin='contest', verified='answer-checked',
         spec=spec(('Z', 'Z', 'Z'), [LT(0, X(0)), LE(X(0), X(1)), LE(X(1), X(2)), EQ(lambda m, n, p: 2 * m * n * p, lambda m, n, p: (m + 2) * (n + 2) * (p + 2))],
                   [LE(X(2), 130), EQ(2 * 3 * 11 * 130, (3 + 2) * (11 + 2) * (130 + 2)), LE(3, 11), LE(11, 130)], domains=([0, 1, 2, 3, 4, 7, 8], [3, 5, 10, 11, 12], [10, 11, 98, 130, 131])),
         notes='formalised as: no solution has p above 130, and (3, 11, 130) is a solution. 2mnp = (m + 2)(n + 2)(p + 2) gives p (mn - 2m - 2n - 4) = 2mn + 4m + 4n + 8; m >= 8 is impossible, m = 1, 2 too, and m = 3..7 give p <= 130')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        @K.theorem(w, 'aime1998_p14', [v.Z, v.Z, v.Z], [lambda m, n, k: v.pos(m), lambda m, n, k: v.le(m, n), lambda m, n, k: v.le(n, k), lambda m, n, k: Eq(2 * m * n * k, (m + 2) * (n + 2) * (k + 2))],
                   lambda m, n, k: v.le(k, lit(130)) & AND(Eq(2 * lit(3) * lit(11) * lit(130), (lit(3) + 2) * (lit(11) + 2) * (lit(130) + 2)), v.le(lit(3), lit(11)), v.le(lit(11), lit(130))))
        def _(p):
            p.intro('m', 'n', 'k', 'h0', 'h1', 'h2', 'h')
            p.split()
            p.obtain(['c', 'c'], ('Z.le_or_lt', 'k', lit(130)))
            p.exact('c')
            p.exfalso()
            p.have('key', p.T('m n k', lambda m, n, k: Eq(k * (m * n - 2 * m - 2 * n - 4), 2 * m * n + 4 * m + 4 * n + 8)), lambda q: q.linear_combination((1, 'h')))
            p.have('hk0', p.T('k', lambda k: v.nonneg(k)), lambda q: q.linarith())
            p.have('hn0', p.T('n', lambda n: v.nonneg(n)), lambda q: q.linarith())
            p.obtain(['c1', 'c1'], ('Z.le_or_lt', 'm', lit(7)))
            p.interval_cases('m', 1, 7)
            for mv in range(1, 8):
                p.rw_at('key', 'm_eq')
                if mv == 1:                                                 # k (-n - 6) = 6 n + 12 with k n >= 0
                    p.have('kn', ('Z.mul_nonneg', 'k', 'n', 'hk0', 'hn0'))
                    p.linarith()
                    continue
                if mv == 2:                                                 # -8 k = 8 n + 16
                    p.linarith()
                    continue
                X_ = lambda n, mv=mv: (mv - 2) * n - 2 * mv - 4             # key:  k X = 2 (mv + 2)(n + 2)

                def pos_script(q, X_=X_):
                    q.obtain(['d', 'd'], ('Z.le_or_lt', q.T('n', X_), lit(0)))
                    q.exfalso()
                    q.have('hnx', q.T('n', lambda n: v.nonneg(-X_(n))), lambda r: r.linarith())
                    q.have('pr', ('Z.mul_nonneg', q.T('n', lambda n: -X_(n)), q.h('k'), 'hnx', 'hk0'))
                    q.linarith()
                    q.exact('d')
                p.have('hX', p.T('n', lambda n: v.pos(X_(n))), pos_script)
                p.have('hX0', p.T('n', lambda n: v.nonneg(X_(n))), lambda q: q.linarith())
                p.have('l131', p.T('k', lambda k: v.le(lit(131), k)), lambda q: q.linarith())
                p.have('hp', ('Z.mul_le_mul_of_nonneg_right', lit(131), p.h('k'), p.T('n', X_), 'l131', 'hX0'))
                p.linarith()
            p.have('hm8', p.T('m', lambda m: v.nonneg(m - 8)), lambda q: q.linarith())
            p.have('hn8', p.T('n', lambda n: v.nonneg(n - 8)), lambda q: q.linarith())
            p.have('hk2', p.T('k', lambda k: v.nonneg(k + 2)), lambda q: q.linarith())
            p.have('hk8', p.T('k', lambda k: v.nonneg(k - 8)), lambda q: q.linarith())
            p.have('hn1', p.T('n', lambda n: v.pos(n)), lambda q: q.linarith())
            p.have('mn8', ('Z.mul_nonneg', p.T('m', lambda m: m - 8), p.T('n', lambda n: n - 8), 'hm8', 'hn8'))
            p.have('b1', p.T('m n', lambda m, n: v.nonneg(25 * m * n - 16 * (m + 2) * (n + 2))), lambda q: q.linarith())
            p.have('c2', ('Z.mul_nonneg', p.T('m n', lambda m, n: 25 * m * n - 16 * (m + 2) * (n + 2)), p.T('k', lambda k: k + 2), 'b1', 'hk2'))
            p.have('mnp', ('Z.mul_pos', 'm', 'n', 'h0', 'hn1'))
            p.have('mnp0', p.T('m n', lambda m, n: v.nonneg(m * n)), lambda q: q.linarith())
            p.have('c3', ('Z.mul_nonneg', p.T('k', lambda k: k - 8), p.T('m n', lambda m, n: m * n), 'hk8', 'mnp0'))
            p.linarith()
            p.split()
            p.ring()
            p.split(); p.linarith(); p.linarith()
