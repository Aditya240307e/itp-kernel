"""bank_classic.py -- the classical results and exercises of olympiad number theory, integer vocabulary, STATEMENT-ONLY (see bank_imo.py for what that means and how the statements are checked).

These are not problems of one contest: they are the lemmas and "find all" exercises that every olympiad training meets (Fermat's little theorem, Wilson's theorem, the two- and four-squares theorems, the gcd of 2^m - 1 and
2^n - 1, the equation x^y = y^x, the exponential Diophantine equations of Catalan type, the functional equations over the integers).  All have `origin='folklore'`; `source` says where the result is usually met.
`verified='answer-checked'`: the claim (the answer) was re-derived by an exhaustive search on a range far beyond the box (imo_truth.py); `recalled`: the claim is a well-known theorem and the search covers only part of it.
Several of them (Fermat's little theorem, Wilson's theorem, Lagrange's theorem) are well beyond what the agent can prove today: they are the top of the benchmark, not a training target.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import math
from bank import BANK
from bank_spec import spec, EQ, LT, LE, DVD, COP, GCD, PRIME, V as X, dvd_
from bank_amc import AND, OR
import bank_kit as K
from bank_imo import L, C, sol_rhs, sol_atoms, sq, is_prime_, exN, fact

NAMES = []


def problem(name, **kw):
    NAMES.append(name)
    return BANK.problem(name, **kw)


def statement(name, **kw):
    NAMES.append(name)
    return BANK.statement(name, **kw)


@statement('classic_fermat_little', text="Fermat's little theorem: if p is a prime and a is an integer, then p divides a^p - a.", source="Fermat's little theorem", topics=('number-theory', 'prime', 'divisibility'), level=4, kind='prove',
           origin='folklore', verified='answer-checked',
           spec=spec(('N', 'Z'), [PRIME(X(0))], DVD(X(0), lambda p, a: a ** p - a), domains=(range(0, 14), range(-6, 7))),
           notes='the prime is a natural number p (it is the exponent), a is any integer')
def _(w, b):
    v = K.Vocab(w); ofN = w['Z.of_N']
    with v.ints():
        return K.statement(w, [v.N, v.Z], [lambda p, a: w['Z.prime'](ofN(p))], lambda p, a: v.dvd(ofN(p), a ** p - a))


@statement('classic_wilson', text="Wilson's theorem: if p is a prime, then p divides (p - 1)! + 1.", source="Wilson's theorem", topics=('number-theory', 'prime', 'product'), level=5, kind='prove', origin='folklore', verified='answer-checked',
           spec=spec(('N',), [PRIME(lambda q: q + 1)], DVD(lambda q: q + 1, lambda q: fact(q) + 1), domains=(range(0, 16),)),
           notes='the prime is p = q + 1 (q a natural number), so that (p - 1)! is q!')
def _(w, b):
    v = K.Vocab(w); ofN = w['Z.of_N']; f = w['Z.fact']
    with v.ints():
        return K.statement(w, [v.N], [lambda q: w['Z.prime'](ofN(q) + 1)], lambda q: v.dvd(ofN(q) + 1, f(q) + 1))


@statement('classic_wilson_converse', text='If n > 1 is an integer such that n divides (n - 1)! + 1, then n is a prime.', source="converse of Wilson's theorem", topics=('number-theory', 'prime', 'product'), level=5, kind='prove', origin='folklore',
           verified='answer-checked',
           spec=spec(('N',), [LT(0, X(0)), DVD(lambda q: q + 1, lambda q: fact(q) + 1)], PRIME(lambda q: q + 1), domains=(range(0, 16),)),
           notes='n = q + 1 with q >= 1')
def _(w, b):
    v = K.Vocab(w); ofN = w['Z.of_N']; f = w['Z.fact']
    with v.ints():
        return K.statement(w, [v.N], [lambda q: w['N.lt'](v.nlit(0), q), lambda q: v.dvd(ofN(q) + 1, f(q) + 1)], lambda q: w['Z.prime'](ofN(q) + 1))


@statement('classic_two_squares', text='Every prime of the form 4k + 1 is a sum of two squares of integers (Fermat).', source="Fermat's two-squares theorem", topics=('number-theory', 'prime', 'diophantine'), level=5, kind='prove', origin='folklore',
           verified='answer-checked',
           spec=spec(('Z', 'Z'), [PRIME(X(0)), EQ(X(0), lambda p, k: 4 * k + 1)], lambda p, k: any(p == x * x + y * y for x in range(-10, 11) for y in range(-10, 11)), atoms=[(X(0), lambda p, k, x, y: x * x + y * y)],
                     domains=([2, 3, 4, 5, 7, 9, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101], range(0, 26)), windows={'ex': 10}, body=(('Z', 'Z'), lambda p, k, x, y: p == x * x + y * y)),
           notes='"of the form 4k + 1" is the hypothesis p = 4k + 1 on the integer k')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z, v.Z], [lambda p, k: w['Z.prime'](p), lambda p, k: Eq(p, 4 * k + 1)], lambda p, k: v.exists(lambda x: v.exists(lambda y: Eq(p, x ** 2 + y ** 2))))


@statement('classic_four_squares', text='Every non-negative integer is a sum of four squares of integers (Lagrange).', source="Lagrange's four-squares theorem", topics=('number-theory', 'diophantine'), level=5, kind='prove', origin='folklore',
           verified='answer-checked',
           spec=spec(('Z',), [LE(0, X(0))], lambda n: any(n == a * a + b * b + c * c + d * d for a in range(-6, 7) for b in range(-6, 7) for c in range(-6, 7) for d in range(-6, 7)),
                     atoms=[(X(0), lambda n, a, b, c, d: a * a + b * b + c * c + d * d)], domains=(range(-1, 26),), windows={'ex': 6}, body=(('Z', 'Z', 'Z', 'Z'), lambda n, a, b, c, d: n == a * a + b * b + c * c + d * d)),
           notes='')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z], [lambda n: v.nonneg(n)], lambda n: v.exists(lambda a: v.exists(lambda b_: v.exists(lambda c: v.exists(lambda d: Eq(n, a ** 2 + b_ ** 2 + c ** 2 + d ** 2))))))


@statement('classic_gcd_pow2', text='If d is the greatest common divisor of the positive integers m and n, then 2^d - 1 is the greatest common divisor of 2^m - 1 and 2^n - 1.', source='classical (gcd of 2^m - 1 and 2^n - 1)',
           topics=('number-theory', 'gcd'), level=4, kind='prove', origin='folklore', verified='answer-checked',
           spec=spec(('N', 'N', 'N'), [LT(0, X(0)), LT(0, X(1)), GCD(X(0), X(1), X(2))], GCD(lambda m, n, d: 2 ** m - 1, lambda m, n, d: 2 ** n - 1, lambda m, n, d: 2 ** d - 1), domains=(range(0, 10), range(0, 10), range(0, 10))),
           notes='')
def _(w, b):
    v = K.Vocab(w); ofN = w['Z.of_N']; g = w['Z.is_gcd']
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N] * 3, [lambda m, n, d: w['N.lt'](v.nlit(0), m), lambda m, n, d: w['N.lt'](v.nlit(0), n), lambda m, n, d: g(ofN(m), ofN(n), ofN(d))],
                           lambda m, n, d: g(lit(2) ** m - 1, lit(2) ** n - 1, lit(2) ** d - 1))


@statement('classic_pythagorean', text='Primitive Pythagorean triples: if a, b, c are positive integers with a^2 + b^2 = c^2, gcd(a, b) = 1 and b even, then there are coprime integers m > n > 0 with a = m^2 - n^2, b = 2 m n, c = m^2 + n^2.',
           source='parametrisation of Pythagorean triples', topics=('number-theory', 'diophantine', 'gcd'), level=4, kind='prove', origin='folklore', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), LT(0, X(2)), EQ(lambda a, b, c: a * a + b * b, lambda a, b, c: c * c), COP(X(0), X(1)), DVD(2, X(1))],
                     lambda a, b, c: any(0 < n < m and math.gcd(m, n) == 1 and a == m * m - n * n and b == 2 * m * n and c == m * m + n * n for m in range(-8, 9) for n in range(-8, 9)),
                     atoms=[(C(0), lambda a, b, c, m, n: n), (lambda a, b, c, m, n: n, lambda a, b, c, m, n: m), (lambda a, b, c, m, n: m, lambda a, b, c, m, n: n), (X(0), lambda a, b, c, m, n: m * m - n * n),
                            (X(1), lambda a, b, c, m, n: 2 * m * n), (X(2), lambda a, b, c, m, n: m * m + n * n)],
                     domains=([1, 3, 5, 7, 15, 21, 35, 9, 4], [0, 2, 4, 8, 12, 20, 24, 40, 3], [5, 13, 17, 25, 29, 37, 41, 1, 6]), windows={'ex': 8},
                     body=(('Z', 'Z'), lambda a, b, c, m, n: 0 < n < m and math.gcd(m, n) == 1 and a == m * m - n * n and b == 2 * m * n and c == m * m + n * n)),
           notes='"b even" is part of the hypothesis (in a primitive triple exactly one of a, b is even)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 3, [lambda a, b_, c: v.pos(a), lambda a, b_, c: v.pos(b_), lambda a, b_, c: v.pos(c), lambda a, b_, c: Eq(a ** 2 + b_ ** 2, c ** 2), lambda a, b_, c: w['Z.coprime'](a, b_), lambda a, b_, c: v.dvd(v.lit(2), b_)],
                           lambda a, b_, c: v.exists(lambda m: v.exists(lambda n: AND(v.pos(n), v.lt(n, m), w['Z.coprime'](m, n), Eq(a, m ** 2 - n ** 2), Eq(b_, 2 * m * n), Eq(c, m ** 2 + n ** 2)))))


@statement('classic_2n_not_div', text='Prove that no integer n > 1 divides 2^n - 1.', source='classical (n does not divide 2^n - 1)', topics=('number-theory', 'divisibility', 'descent'), level=4, kind='prove', origin='folklore', verified='answer-checked',
           spec=spec(('N',), [LT(1, X(0))], lambda n: not dvd_(n, 2 ** n - 1), atoms=[(X(0), lambda n: 2 ** n - 1)], domains=(range(0, 14),)),
           notes='the proof looks at the least prime divisor p of n: the order of 2 modulo p divides n and p - 1, hence is 1')
def _(w, b):
    v = K.Vocab(w); ofN = w['Z.of_N']
    with v.ints():
        return K.statement(w, [v.N], [lambda n: w['N.lt'](v.nlit(1), n)], lambda n: v.Not(v.dvd(ofN(n), v.lit(2) ** n - 1)))


_SXY = [(2, 4), (4, 2)]


@statement('classic_xy_yx', text='Find all pairs of positive integers (x, y) with x^y = y^x.', source='classical (x^y = y^x)', topics=('number-theory', 'diophantine', 'bounding'), level=4, kind='answer', answer='x = y, or (2, 4), (4, 2)',
           origin='folklore', verified='answer-checked',
           spec=spec(('N', 'N'), [LT(0, X(0)), LT(0, X(1))], (lambda x, y: x ** y == y ** x, lambda x, y: x == y or (x, y) in _SXY),
                     atoms=[(lambda x, y: x ** y, lambda x, y: y ** x), (X(0), X(1))] + sol_atoms(_SXY, (0, 1)), domains=(range(0, 8), range(0, 8))),
           notes='')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        return K.statement(w, [v.N, v.N], [lambda x, y: w['N.lt'](v.nlit(0), x), lambda x, y: w['N.lt'](v.nlit(0), y)],
                           lambda x, y: v.Iff(Eq(ofN(x) ** y, ofN(y) ** x), OR(Eq(x, y), sol_rhs(v, [x, y], _SXY, ['N', 'N']))))


_SPQ = [(2, 3), (3, 2)]


@statement('classic_pq_qp', text='Find all primes p and q such that p^q + q^p is a prime.', source='classical (p^q + q^p)', topics=('number-theory', 'prime', 'case-analysis'), level=4, kind='answer', answer='(p, q) = (2, 3), (3, 2)', origin='folklore',
           verified='answer-checked',
           spec=spec(('N', 'N'), [PRIME(X(0)), PRIME(X(1))], (PRIME(lambda p, q: p ** q + q ** p), lambda p, q: (p, q) in _SPQ), atoms=[(lambda p, q: p ** q + q ** p,)] + sol_atoms(_SPQ, (0, 1)), domains=([0, 1, 2, 3, 4, 5, 7, 11], [0, 1, 2, 3, 4, 5, 7, 11])),
           notes='if p, q are both odd the sum is even; if p = 2 and q > 3 then 2^q + q^2 is a multiple of 3')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        return K.statement(w, [v.N, v.N], [lambda p, q: w['Z.prime'](ofN(p)), lambda p, q: w['Z.prime'](ofN(q))],
                           lambda p, q: v.Iff(w['Z.prime'](ofN(p) ** q + ofN(q) ** p), sol_rhs(v, [p, q], _SPQ, ['N', 'N'])))


@statement('classic_fact_composite', text='If n > 4 is a composite integer, then n divides (n - 1)!.', source='classical ((n - 1)! and composite numbers)', topics=('number-theory', 'divisibility', 'product'), level=4, kind='prove', origin='folklore',
           verified='answer-checked',
           spec=spec(('N',), [LT(3, X(0)), lambda q: not is_prime_(q + 1)], DVD(lambda q: q + 1, lambda q: fact(q)), hatoms=[(C(3), X(0)), (lambda q: q + 1,)], domains=(range(0, 18),)),
           notes='n = q + 1, so that (n - 1)! is q!; composite means "not prime" for n > 4')
def _(w, b):
    v = K.Vocab(w); ofN = w['Z.of_N']; f = w['Z.fact']
    with v.ints():
        return K.statement(w, [v.N], [lambda q: w['N.lt'](v.nlit(3), q), lambda q: v.Not(w['Z.prime'](ofN(q) + 1))], lambda q: v.dvd(ofN(q) + 1, f(q)))


@statement('classic_bertrand', text="Bertrand's postulate: for every positive integer n there is a prime p with n < p <= 2n.", source="Bertrand's postulate (Chebyshev)", topics=('number-theory', 'prime', 'bounding'), level=5, kind='prove', origin='folklore',
           verified='answer-checked',
           spec=spec(('Z',), [LT(0, X(0))], lambda n: any(is_prime_(p) and n < p <= 2 * n for p in range(-90, 91)), atoms=[(lambda n, p: p,), (X(0), lambda n, p: p), (lambda n, p: p, lambda n, p: 2 * n)], domains=(range(0, 41),),
                     windows={'ex': 90}, body=(('Z',), lambda n, p: is_prime_(p) and n < p <= 2 * n)),
           notes='checked here for n <= 40 and by imo_truth.py for n < 100000')
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        return K.statement(w, [v.Z], [lambda n: v.pos(n)], lambda n: v.exists(lambda p: AND(w['Z.prime'](p), v.lt(n, p), v.le(p, 2 * n))))


@statement('classic_primes_3mod4', text='There are infinitely many primes of the form 4k + 3: for every n there is a prime p > n with p = 4k + 3.', source='classical (primes 3 mod 4)', topics=('number-theory', 'prime', 'construction'), level=4, kind='prove',
           origin='folklore', verified='answer-checked',
           spec=spec(('Z',), [LE(0, X(0))], lambda n: any(is_prime_(p) and n < p and p == 4 * k + 3 for p in range(-60, 61) for k in range(-60, 61)),
                     atoms=[(lambda n, p, k: p,), (X(0), lambda n, p, k: p), (lambda n, p, k: p, lambda n, p, k: 4 * k + 3)], domains=(range(0, 25),), windows={'ex': 60}, body=(('Z', 'Z'), lambda n, p, k: is_prime_(p) and n < p and p == 4 * k + 3)),
           notes='the proof: if p_1 .. p_r were all of them, 4 p_1 .. p_r - 1 has a prime factor of the form 4k + 3')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z], [lambda n: v.nonneg(n)], lambda n: v.exists(lambda p: v.exists(lambda k: AND(w['Z.prime'](p), v.lt(n, p), Eq(p, 4 * k + 3)))))


@statement('classic_pyth_60', text='If a, b, c are integers with a^2 + b^2 = c^2, then 60 divides abc.', source='classical (the product of a Pythagorean triple)', topics=('number-theory', 'divisibility', 'modular'), level=4, kind='prove', origin='folklore',
           verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z'), [EQ(lambda a, b, c: a * a + b * b, lambda a, b, c: c * c)], DVD(60, lambda a, b, c: a * b * c), domains=(range(0, 21), range(0, 21), range(0, 30)), points=14000),
           notes='the argument looks at the three primes: 3 divides one of a, b; 4 one of a, b (or both are even); 5 one of a, b, c')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 3, [lambda a, b_, c: Eq(a ** 2 + b_ ** 2, c ** 2)], lambda a, b_, c: v.dvd(v.lit(60), a * b_ * c))


@statement('classic_consecutive_product', text='The product of n consecutive non-negative integers m (m + 1) .. (m + n - 1) is divisible by n!.', source='classical (n! divides a product of n consecutive integers)', topics=('number-theory', 'divisibility', 'product'), level=4,
           kind='prove', origin='folklore', verified='answer-checked',
           spec=spec(('Z', 'N'), [LE(0, X(0))], DVD(lambda m, n: fact(n), lambda m, n: math.prod(m + i for i in range(n))), domains=(range(-1, 13), range(0, 10))),
           notes='the product is Z.prod over i < n of m + i (the empty product is 1)')
def _(w, b):
    v = K.Vocab(w); f = w['Z.fact']
    with v.ints():
        return K.statement(w, [v.Z, v.N], [lambda m, n: v.nonneg(m)], lambda m, n: v.dvd(f(n), w['Z.prod'](w.lam(v.Z, lambda i: m + i), n)))


@statement('classic_fermat_coprime', text='The Fermat numbers F_k = 2^(2^k) + 1 are pairwise coprime.', source='classical (Fermat numbers)', topics=('number-theory', 'gcd'), level=4, kind='prove', origin='folklore', verified='answer-checked',
           spec=spec(('N', 'N'), [lambda m, n: m != n], COP(lambda m, n: 2 ** (2 ** m) + 1, lambda m, n: 2 ** (2 ** n) + 1), hatoms=[(X(0), X(1))], domains=(range(0, 7), range(0, 7))),
           notes='F_n - 2 = F_0 F_1 .. F_(n-1)')
def _(w, b):
    v = K.Vocab(w); pw = w['N.pow']
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N, v.N], [lambda m, n: v.Not(v.Eq(m, n))], lambda m, n: w['Z.coprime'](lit(2) ** pw(v.nlit(2), m) + 1, lit(2) ** pw(v.nlit(2), n) + 1))


_S3X2Y = [(1, 1), (2, 3)]


@statement('classic_3x_2y', text='Find all pairs of positive integers (x, y) with 3^x - 2^y = 1.', source='classical (exponential Diophantine equations)', topics=('number-theory', 'diophantine', 'modular'), level=4, kind='answer', answer='(x, y) = (1, 1), (2, 3)',
           origin='folklore', verified='answer-checked',
           spec=spec(('N', 'N'), [LT(0, X(0)), LT(0, X(1))], (lambda x, y: 3 ** x - 2 ** y == 1, lambda x, y: (x, y) in _S3X2Y), atoms=[(lambda x, y: 3 ** x - 2 ** y, C(1))] + sol_atoms(_S3X2Y, (0, 1)), domains=(range(0, 6), range(0, 9))),
           notes='')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N, v.N], [lambda x, y: w['N.lt'](v.nlit(0), x), lambda x, y: w['N.lt'](v.nlit(0), y)], lambda x, y: v.Iff(Eq(lit(3) ** x - lit(2) ** y, v.one), sol_rhs(v, [x, y], _S3X2Y, ['N', 'N'])))


_S2X1 = [(1, 1), (3, 2)]


@statement('classic_2x1_3y', text='Find all pairs of positive integers (x, y) with 2^x + 1 = 3^y.', source='classical (exponential Diophantine equations)', topics=('number-theory', 'diophantine', 'modular'), level=4, kind='answer', answer='(x, y) = (1, 1), (3, 2)',
           origin='folklore', verified='answer-checked',
           spec=spec(('N', 'N'), [LT(0, X(0)), LT(0, X(1))], (lambda x, y: 2 ** x + 1 == 3 ** y, lambda x, y: (x, y) in _S2X1), atoms=[(lambda x, y: 2 ** x + 1, lambda x, y: 3 ** y)] + sol_atoms(_S2X1, (0, 1)), domains=(range(0, 7), range(0, 6))),
           notes='')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N, v.N], [lambda x, y: w['N.lt'](v.nlit(0), x), lambda x, y: w['N.lt'](v.nlit(0), y)], lambda x, y: v.Iff(Eq(lit(2) ** x + 1, lit(3) ** y), sol_rhs(v, [x, y], _S2X1, ['N', 'N'])))


_S23Z = [(0, 1, 2), (3, 0, 3), (4, 2, 5)]


@statement('classic_2x_3y_square', text='Find all non-negative integers x, y, z with 2^x + 3^y = z^2.', source='classical (exponential Diophantine equations)', topics=('number-theory', 'diophantine', 'modular'), level=5, kind='answer',
           answer='(x, y, z) = (0, 1, 2), (3, 0, 3), (4, 2, 5)', origin='folklore', verified='answer-checked',
           spec=spec(('N', 'N', 'N'), [], (lambda x, y, z: 2 ** x + 3 ** y == z * z, lambda x, y, z: (x, y, z) in _S23Z), atoms=[(lambda x, y, z: 2 ** x + 3 ** y, lambda x, y, z: z * z)] + sol_atoms(_S23Z, (0, 1, 2)),
                     domains=(range(0, 7), range(0, 5), range(0, 8))),
           notes='z is taken non-negative (z and -z give the same equation)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N] * 3, [], lambda x, y, z: v.Iff(Eq(lit(2) ** x + lit(3) ** y, ofN(z) ** 2), sol_rhs(v, [x, y, z], _S23Z, ['N', 'N', 'N'])))


@statement('classic_fact_product_sum', text='Find all positive integers a, b, c with a! b! = a! + b! + c!.', source='classical (factorial Diophantine equations)', topics=('number-theory', 'diophantine', 'bounding'), level=4, kind='answer',
           answer='(a, b, c) = (3, 3, 4)', origin='folklore', verified='answer-checked',
           spec=spec(('N', 'N', 'N'), [LT(0, X(0)), LT(0, X(1)), LT(0, X(2))], (lambda a, b, c: fact(a) * fact(b) == fact(a) + fact(b) + fact(c), lambda a, b, c: (a, b, c) == (3, 3, 4)),
                     atoms=[(lambda a, b, c: fact(a) * fact(b), lambda a, b, c: fact(a) + fact(b) + fact(c))] + sol_atoms([(3, 3, 4)], (0, 1, 2)), domains=(range(0, 6), range(0, 6), range(0, 7))),
           notes='')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; f = w['Z.fact']
    with v.ints():
        return K.statement(w, [v.N] * 3, [lambda a, b_, c: w['N.lt'](v.nlit(0), a), lambda a, b_, c: w['N.lt'](v.nlit(0), b_), lambda a, b_, c: w['N.lt'](v.nlit(0), c)],
                           lambda a, b_, c: v.Iff(Eq(f(a) * f(b_), f(a) + f(b_) + f(c)), sol_rhs(v, [a, b_, c], [(3, 3, 4)], ['N', 'N', 'N'])))


@statement('classic_fact_sum_fact', text='Find all positive integers a, b, c with a! + b! = c!.', source='classical (factorial Diophantine equations)', topics=('number-theory', 'diophantine', 'bounding'), level=3, kind='answer', answer='(a, b, c) = (1, 1, 2)',
           origin='folklore', verified='answer-checked',
           spec=spec(('N', 'N', 'N'), [LT(0, X(0)), LT(0, X(1)), LT(0, X(2))], (lambda a, b, c: fact(a) + fact(b) == fact(c), lambda a, b, c: (a, b, c) == (1, 1, 2)),
                     atoms=[(lambda a, b, c: fact(a) + fact(b), lambda a, b, c: fact(c))] + sol_atoms([(1, 1, 2)], (0, 1, 2)), domains=(range(0, 6), range(0, 6), range(0, 7))),
           notes='')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; f = w['Z.fact']
    with v.ints():
        return K.statement(w, [v.N] * 3, [lambda a, b_, c: w['N.lt'](v.nlit(0), a), lambda a, b_, c: w['N.lt'](v.nlit(0), b_), lambda a, b_, c: w['N.lt'](v.nlit(0), c)],
                           lambda a, b_, c: v.Iff(Eq(f(a) + f(b_), f(c)), sol_rhs(v, [a, b_, c], [(1, 1, 2)], ['N', 'N', 'N'])))


@statement('classic_x2_y3_1', text='Find all pairs of positive integers (x, y) with x^2 = y^3 + 1.', source='classical (Euler)', topics=('number-theory', 'diophantine', 'case-analysis'), level=4, kind='answer', answer='(x, y) = (3, 2)',
           origin='folklore', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1))], (lambda x, y: x * x == y ** 3 + 1, lambda x, y: (x, y) == (3, 2)), atoms=[(lambda x, y: x * x, lambda x, y: y ** 3 + 1)] + sol_atoms([(3, 2)], (0, 1)),
                     domains=(range(0, 11), range(0, 11))),
           notes='y^3 = (x - 1)(x + 1): the factors are coprime or have gcd 2')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 2, [lambda x, y: v.pos(x), lambda x, y: v.pos(y)], lambda x, y: v.Iff(Eq(x ** 2, y ** 3 + 1), sol_rhs(v, [x, y], [(3, 2)])))


@statement('classic_y2_x3_2', text='Find all pairs of positive integers (x, y) with y^2 = x^3 - 2.', source="classical (Fermat's problem y^2 = x^3 - 2)", topics=('number-theory', 'diophantine'), level=5, kind='answer', answer='(x, y) = (3, 5)',
           origin='folklore', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1))], (lambda x, y: y * y == x ** 3 - 2, lambda x, y: (x, y) == (3, 5)), atoms=[(lambda x, y: y * y, lambda x, y: x ** 3 - 2)] + sol_atoms([(3, 5)], (0, 1)),
                     domains=(range(0, 7), range(0, 8))),
           notes='the standard proof factorises y^2 + 2 in Z[sqrt(-2)], a unique factorisation domain: this is far outside what the agent can do and is in the bank as a statement to be recognised, not as a goal')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 2, [lambda x, y: v.pos(x), lambda x, y: v.pos(y)], lambda x, y: v.Iff(Eq(y ** 2, x ** 3 - 2), sol_rhs(v, [x, y], [(3, 5)])))


@statement('classic_2p1_cube', text='Find all primes p for which 2p + 1 is a perfect cube of an integer.', source='classical (primes and cubes)', topics=('number-theory', 'prime', 'divisibility'), level=3, kind='answer', answer='p = 13',
           origin='folklore', verified='answer-checked',
           spec=spec(('Z', 'Z'), [PRIME(X(0))], (lambda p, n: 2 * p + 1 == n ** 3, lambda p, n: (p, n) == (13, 3)), atoms=[(lambda p, n: 2 * p + 1, lambda p, n: n ** 3)] + sol_atoms([(13, 3)], (0, 1)),
                     domains=([0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41], range(-3, 6))),
           notes='the cube is n^3 for an integer n: 2p + 1 = n^3 means p = (n - 1)(n^2 + n + 1)/2')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 2, [lambda p, n: w['Z.prime'](p)], lambda p, n: v.Iff(Eq(2 * p + 1, n ** 3), sol_rhs(v, [p, n], [(13, 3)])))


def _std1(k): return lambda x: k * x * x


@statement('classic_fe_n2_1', text='Find all functions f from the integers to the integers such that f(m + n) + f(mn - 1) = f(m) f(n) + 2 for all integers m and n.', source='classical functional equation over the integers', topics=('functional-equation', 'algebra'),
           level=4, kind='answer', answer='f(n) = n^2 + 1', origin='folklore', verified='answer-checked',
           spec=spec(('fn',), [], (lambda f: all(f(m + n) + f(m * n - 1) == f(m) * f(n) + 2 for m in range(-8, 9) for n in range(-8, 9)), lambda f: all(f(n) == n * n + 1 for n in range(-8, 9))),
                     atoms=[(lambda f, m, n: f(m + n) + f(m * n - 1), lambda f, m, n: f(m) * f(n) + 2), (lambda f, n: f(n), lambda f, n: n * n + 1)]),
           notes='m = 0 gives f(0) = 1 and f(-1) = 2 (no constant solves the equation); then m = 1 and m = -1 determine f from f(1) = t, and t = 2 is the only value that works; imo_truth.py solves it exactly on |n| <= 12')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        return K.statement(w, [fZ], [], lambda f: v.Iff(v.F([v.Z, v.Z], lambda m, n: Eq(f(m + n) + f(m * n - 1), f(m) * f(n) + 2)), v.forall(lambda n: Eq(f(n), n ** 2 + 1))))


@statement('classic_fe_sum_prod', text='Find all functions f from the integers to the integers such that f(x + y) + f(x y) = f(x) f(y) + 1 for all integers x and y.', source='classical functional equation over the integers', topics=('functional-equation', 'algebra', 'parity'),
           level=4, kind='answer', answer='f(n) = 1; f(n) = n + 1; f(n) = 1 for even n and 0 for odd n', origin='folklore', verified='answer-checked',
           spec=spec(('fn',), [], (lambda f: all(f(x + y) + f(x * y) == f(x) * f(y) + 1 for x in range(-8, 9) for y in range(-8, 9)),
                                   lambda f: all(f(n) == 1 for n in range(-8, 9)) or all(f(n) == n + 1 for n in range(-8, 9)) or all(f(n) == (1 if n % 2 == 0 else 0) for n in range(-8, 9))),
                     atoms=[(lambda f, x, y: f(x + y) + f(x * y), lambda f, x, y: f(x) * f(y) + 1), (lambda f, n: f(n), C(1)), (lambda f, n: f(n), lambda f, n: n + 1),
                            (C(2), lambda f, n: n), (lambda f, n: f(n), C(1)), (C(2), lambda f, n: n), (lambda f, n: f(n), C(0))],
                     extra_fns=[lambda n: 1 if n % 2 == 0 else 0, lambda n: 1 if n % 2 else 0]),
           notes='f(0) = 1 follows from x = 0, then x = 1 gives f(y + 1) = (f(1) - 1) f(y) + 1, so f is determined by f(1) = t, and t = 0, 1, 2 are the only values that work')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        one = v.one; z0 = v.zero; two = v.lit(2)
        return K.statement(w, [fZ], [], lambda f: v.Iff(v.F([v.Z, v.Z], lambda x, y: Eq(f(x + y) + f(x * y), f(x) * f(y) + 1)),
                                                      OR(v.forall(lambda n: Eq(f(n), one)), v.forall(lambda n: Eq(f(n), n + 1)),
                                                         v.forall(lambda n: AND(v.dvd(two, n) >> Eq(f(n), one), v.Not(v.dvd(two, n)) >> Eq(f(n), z0))))))


@statement('classic_fe_sum_squares', text='Let f be a function from the non-negative integers to the non-negative integers with f(1) > 0 and f(m^2 + n^2) = f(m)^2 + f(n)^2 for all m, n. Prove that f(n) = n for all n.', source='classical functional equation over the integers',
           topics=('functional-equation', 'induction'), level=5, kind='prove', origin='folklore', verified='answer-checked',
           spec=spec(('fn',), [lambda f: all(f(n) >= 0 for n in range(0, 9)), lambda f: f(1) > 0, lambda f: all(f(m * m + n * n) == f(m) ** 2 + f(n) ** 2 for m in range(0, 9) for n in range(0, 9))], lambda f: all(f(n) == n for n in range(0, 9)),
                     hatoms=False, atoms=[(C(0), lambda f, n: n), (lambda f, n: f(n), lambda f, n: n)], extra_fns=[lambda n: n if n >= 0 else 5, lambda n: n if n >= 0 else -5, lambda n: n if n <= 3 else 0, lambda n: n if n != 0 else 1],
                     skip_mutants={'Z.le -> Z.lt in (Z.le 0 x3)': 'equivalent: the equation is symmetric in m and n, so a guard 0 < m only loses the pair (0, 0), and the pair (1, 0) already gives f(1) = f(1)^2 + f(0)^2, hence f(0) = 0 and f(0) = 2 f(0)^2',
                                   'constant 0 -> 1 in (Z.le 0 x3)': 'the same', 'Z.le -> Z.lt in (Z.le 0 x4)': 'the same', 'constant 0 -> 1 in (Z.le 0 x4)': 'the same'}),
           notes='f is a function on the integers whose values at the non-negative integers are non-negative and satisfy the equation there; f(0) = 0, f(1) = 1, f(2) = 2, f(4) = 4, f(5) = 5 and then every f(n) follows from identities like 3^2 + 4^2 = 5^2 + 0^2, '
                 'in general from (2k+1)^2 + (k-2)^2 = (2k-1)^2 + (k+2)^2')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        return K.statement(w, [fZ], [lambda f: v.forall(lambda n: v.nonneg(n) >> v.nonneg(f(n))), lambda f: v.pos(f(v.one)),
                                    lambda f: v.F([v.Z, v.Z], lambda m, n: v.nonneg(m) >> (v.nonneg(n) >> Eq(f(m ** 2 + n ** 2), f(m) ** 2 + f(n) ** 2)))],
                           lambda f: v.forall(lambda n: v.nonneg(n) >> Eq(f(n), n)))
