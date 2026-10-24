"""bank_imo2.py -- olympiad problems beyond the IMO proper: the IMO SHORTLIST and classical results of olympiad number theory; integer vocabulary; STATEMENT-ONLY (see bank_imo.py for what that means, how the statements
are checked and what the provenance tags say).

`origin='contest'`: the problem of the named contest as the authors recall it, paraphrased; `adapted`: changed to fit the vocabulary (the change is in `notes`); `folklore`: a classical fact or exercise of olympiad number
theory with no single contest behind it (the `source` names where it is usually met).  `verified='answer-checked'`: the claim agrees with an exhaustive search on a range far beyond the box (imo_truth.py);
`recalled`: the statement or its source label is from memory and the search covers only part of the claim.
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


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the IMO Shortlist
@statement('isl2006_n5', text='Find all integer solutions of the equation (x^7 - 1)/(x - 1) = y^5 - 1.', source='ISL 2006 N5', year=2006, topics=('number-theory', 'diophantine', 'modular'), level=5, kind='prove', origin='adapted', verified='recalled',
           spec=spec(('Z', 'Z'), [EQ(lambda x, y: x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1, lambda x, y: y ** 5 - 1)], lambda x, y: False, atoms=[], vacuous=True, domains=(range(-9, 10), range(-9, 10))),
           notes='there are no solutions; the quotient is written as the polynomial x^6 + x^5 + .. + 1 (which also covers x = 1, where it is 7 and 7 = y^5 - 1 has no integer solution either). The source label is from memory; '
                 'imo_truth.py finds no solution with |x| <= 3000')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 2, [lambda x, y: Eq(x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1, y ** 5 - 1)], lambda x, y: v.False_)


@statement('isl2007_n1', text='Find all pairs (k, n) of positive integers such that 7^k - 3^n divides k^4 + n^2.', source='ISL 2007 N1', year=2007, topics=('number-theory', 'divisibility', 'bounding'), level=5, kind='answer', answer='(k, n) = (2, 4)',
           origin='contest', verified='answer-checked',
           spec=spec(('N', 'N'), [LT(0, X(0)), LT(0, X(1))], (lambda k, n: dvd_(7 ** k - 3 ** n, k ** 4 + n * n), lambda k, n: (k, n) == (2, 4)),
                     atoms=[(lambda k, n: 7 ** k - 3 ** n, lambda k, n: k ** 4 + n * n)] + sol_atoms([(2, 4)], (0, 1)), domains=(range(0, 8), range(0, 11))),
           notes='7^k - 3^n is even and never zero, so "divides" is divisibility by a possibly negative integer')
def _(w, b):
    v = K.Vocab(w); ofN = w['Z.of_N']
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N, v.N], [lambda k, n: w['N.lt'](v.nlit(0), k), lambda k, n: w['N.lt'](v.nlit(0), n)],
                           lambda k, n: v.Iff(v.dvd(lit(7) ** k - lit(3) ** n, ofN(k) ** 4 + ofN(n) ** 2), sol_rhs(v, [k, n], [(2, 4)], ['N', 'N'])))


@statement('isl2007_n6', text='Let k be a positive integer. Prove that the number (4k^2 - 1)^2 has a positive divisor of the form 8kn - 1 if and only if k is even.', source='ISL 2007 N6', year=2007, topics=('number-theory', 'divisibility', 'parity'), level=5,
           kind='prove', origin='contest', verified='recalled',
           spec=spec(('Z',), [LT(0, X(0))], (lambda k: any(dvd_(8 * k * n - 1, (4 * k * k - 1) ** 2) for n in range(1, 451)), lambda k: k % 2 == 0),
                     atoms=[(C(0), lambda k, n: n), (lambda k, n: 8 * k * n - 1, lambda k, n: (4 * k * k - 1) ** 2), (C(2), lambda k: k)], domains=(range(0, 7),), windows={'ex': 450}),
           notes='n ranges over the positive integers (a divisor 8kn - 1 of (4k^2 - 1)^2 has n <= (4k^2 - 1)^2 / 8k, so for k <= 6 the witness window 450 is exhaustive). The source label is from memory (the problem is the key lemma of '
                 'IMO 2007 P5, see imo2007_p5)')
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z], [lambda k: v.pos(k)], lambda k: v.Iff(v.exists(lambda n: v.pos(n) & v.dvd(8 * k * n - 1, (4 * k ** 2 - 1) ** 2)), v.dvd(lit(2), k)))


@statement('isl2008_n1', text='Let n be a positive integer and p a prime number. Prove that if a, b, c are integers (not necessarily positive) satisfying a^n + p b = b^n + p c = c^n + p a, then a = b = c.', source='ISL 2008 N1', year=2008,
           topics=('number-theory', 'prime', 'descent'), level=5, kind='prove', origin='contest', verified='answer-checked',
           spec=spec(('N', 'Z', 'Z', 'Z', 'Z'), [LT(0, X(0)), PRIME(X(1)), EQ(lambda n, p, a, b, c: a ** n + p * b, lambda n, p, a, b, c: b ** n + p * c), EQ(lambda n, p, a, b, c: b ** n + p * c, lambda n, p, a, b, c: c ** n + p * a)],
                     [EQ(X(2), X(3)), EQ(X(3), X(4))], domains=(range(0, 6), [-2, 0, 1, 2, 3, 5, 7], range(-3, 4), range(-3, 4), range(-3, 4)), points=20000),
           notes='the chain of equalities a^n + p b = b^n + p c = c^n + p a is the two equations a^n + p b = b^n + p c and b^n + p c = c^n + p a')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.N, v.Z, v.Z, v.Z, v.Z],
                           [lambda n, p, a, b_, c: w['N.lt'](v.nlit(0), n), lambda n, p, a, b_, c: w['Z.prime'](p),
                            lambda n, p, a, b_, c: Eq(a ** n + p * b_, b_ ** n + p * c), lambda n, p, a, b_, c: Eq(b_ ** n + p * c, c ** n + p * a)],
                           lambda n, p, a, b_, c: AND(Eq(a, b_), Eq(b_, c)))


_S2010 = [(6, 3), (9, 3), (9, 5), (54, 5)]


@statement('isl2010_n2', text='Find all pairs (m, n) of non-negative integers for which m^2 + 2 * 3^n = m (2^(n+1) - 1).', source='ISL 2010 N2', year=2010, topics=('number-theory', 'diophantine'), level=5, kind='answer',
           answer='(m, n) = (6, 3), (9, 3), (9, 5), (54, 5)', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'N'), [LE(0, X(0))], (lambda m, n: m * m + 2 * 3 ** n == m * (2 ** (n + 1) - 1), lambda m, n: (m, n) in _S2010),
                     atoms=[(lambda m, n: m * m + 2 * 3 ** n, lambda m, n: m * (2 ** (n + 1) - 1))] + sol_atoms(_S2010, (0, 1)), domains=(range(-1, 61), range(0, 9))),
           notes='the equation is a quadratic in m, so for every n there are at most two values of m; imo_truth.py solves it for n < 60')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z, v.N], [lambda m, n: v.nonneg(m)],
                           lambda m, n: v.Iff(Eq(m ** 2 + 2 * lit(3) ** n, m * (lit(2) ** w['N.succ'](n) - 1)), sol_rhs(v, [m, n], _S2010, ['Z', 'N'])))


@statement('isl2012_n2', text='Find all triples (x, y, z) of positive integers with x <= y <= z and x^3 (y^3 + z^3) = 2012 (x y z + 2).', source='ISL 2012 N2', year=2012, topics=('number-theory', 'diophantine', 'bounding'), level=5, kind='answer',
           answer='(x, y, z) = (2, 251, 252)', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z'), [LT(0, X(0)), LE(X(0), X(1)), LE(X(1), X(2))], (lambda x, y, z: x ** 3 * (y ** 3 + z ** 3) == 2012 * (x * y * z + 2), lambda x, y, z: (x, y, z) == (2, 251, 252)),
                     atoms=[(lambda x, y, z: x ** 3 * (y ** 3 + z ** 3), lambda x, y, z: 2012 * (x * y * z + 2))] + sol_atoms([(2, 251, 252)], (0, 1, 2)),
                     domains=([0, 1, 2, 3, 4], [1, 2, 250, 251, 252, 253], [1, 2, 250, 251, 252, 253, 254])),
           notes='')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 3, [lambda x, y, z: v.pos(x), lambda x, y, z: v.le(x, y), lambda x, y, z: v.le(y, z)],
                           lambda x, y, z: v.Iff(Eq(x ** 3 * (y ** 3 + z ** 3), 2012 * (x * y * z + 2)), sol_rhs(v, [x, y, z], [(2, 251, 252)])))


@statement('isl2015_n2', text='Let a and b be positive integers such that a! + b! divides a! b!. Prove that 3a >= 2b + 2.', source='ISL 2015 N2', year=2015, topics=('number-theory', 'divisibility', 'bounding'), level=5, kind='prove',
           origin='contest', verified='answer-checked',
           spec=spec(('N', 'N'), [LT(0, X(0)), LT(0, X(1)), DVD(lambda a, b: fact(a) + fact(b), lambda a, b: fact(a) * fact(b))], LE(lambda a, b: 2 * b + 2, lambda a, b: 3 * a), domains=(range(0, 11), range(0, 11))),
           notes='equality holds for a = b = 2')
def _(w, b):
    v = K.Vocab(w); ofN = w['Z.of_N']; f = w['Z.fact']
    with v.ints():
        return K.statement(w, [v.N, v.N], [lambda a, b_: w['N.lt'](v.nlit(0), a), lambda a, b_: w['N.lt'](v.nlit(0), b_), lambda a, b_: v.dvd(f(a) + f(b_), f(a) * f(b_))],
                           lambda a, b_: v.le(2 * ofN(b_) + 2, 3 * ofN(a)))


@statement('isl2016_n4', text='Let n, m, k and l be positive integers with n != 1 such that n^k + m n^l + 1 divides n^(k+l) - 1. Prove that m = 1 and l = 2k; or l divides k and m = (n^(k-l) - 1)/(n^l - 1).', source='ISL 2016 N4', year=2016,
           topics=('number-theory', 'divisibility', 'case-analysis'), level=5, kind='prove', origin='adapted', verified='answer-checked',
           spec=spec(('Z', 'Z', 'N', 'N'), [LE(2, X(0)), LT(0, X(1)), LT(0, X(2)), LT(0, X(3)), DVD(lambda n, m, k, l: n ** k + m * n ** l + 1, lambda n, m, k, l: n ** (k + l) - 1)],
                     lambda n, m, k, l: (m == 1 and l == 2 * k) or any(k == l * (i + 1) and m * (n ** l - 1) == n ** (l * i) - 1 for i in range(0, 13)),
                     atoms=[(X(1), C(1)), (X(3), lambda n, m, k, l: 2 * k), (X(2), lambda n, m, k, l, i: l * (i + 1)), (lambda n, m, k, l, i: m * (n ** l - 1), lambda n, m, k, l, i: n ** (l * i) - 1)],
                     domains=(range(1, 5), range(0, 6), range(0, 6), range(0, 6))),
           notes='the second case "l divides k and m = (n^(k-l) - 1)/(n^l - 1)" is written without subtraction in the exponent: k = l (i + 1) for some i >= 0 and m (n^l - 1) = n^(l i) - 1. n >= 2 stands for "n positive, n != 1"')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']; dbl = w['N.double']; mul = w['N.mul']; suc = w['N.succ']
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z, v.Z, v.N, v.N],
                           [lambda n, m, k, l: v.le(lit(2), n), lambda n, m, k, l: v.pos(m), lambda n, m, k, l: w['N.lt'](v.nlit(0), k), lambda n, m, k, l: w['N.lt'](v.nlit(0), l),
                            lambda n, m, k, l: v.dvd(n ** k + m * n ** l + 1, n ** w['N.add'](k, l) - 1)],
                           lambda n, m, k, l: OR(AND(Eq(m, v.one), Eq(l, dbl(k))), exN(w, lambda i: AND(Eq(k, mul(l, suc(i))), Eq(m * (n ** l - 1), n ** mul(l, i) - 1)))))


_S2019 = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]


@statement('isl2019_n2', text='Find all triples (a, b, c) of positive integers such that a^3 + b^3 + c^3 = (a b c)^2.', source='ISL 2019 N2', year=2019, topics=('number-theory', 'diophantine', 'bounding'), level=5, kind='answer',
           answer='(a, b, c) = (1, 2, 3) and its permutations', origin='contest', verified='answer-checked',
           spec=spec(('N', 'N', 'N'), [LT(0, X(0)), LT(0, X(1)), LT(0, X(2))], (lambda a, b, c: a ** 3 + b ** 3 + c ** 3 == (a * b * c) ** 2, lambda a, b, c: (a, b, c) in _S2019),
                     atoms=[(lambda a, b, c: a ** 3 + b ** 3 + c ** 3, lambda a, b, c: (a * b * c) ** 2)] + sol_atoms(_S2019, (0, 1, 2)), domains=(range(0, 5), range(0, 5), range(0, 5))),
           notes='')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        return K.statement(w, [v.N] * 3, [lambda a, b_, c: w['N.lt'](v.nlit(0), a), lambda a, b_, c: w['N.lt'](v.nlit(0), b_), lambda a, b_, c: w['N.lt'](v.nlit(0), c)],
                           lambda a, b_, c: v.Iff(Eq(ofN(a) ** 3 + ofN(b_) ** 3 + ofN(c) ** 3, (ofN(a) * ofN(b_) * ofN(c)) ** 2), sol_rhs(v, [a, b_, c], _S2019, ['N', 'N', 'N'])))


@statement('isl2022_n1', text='A number is called Norwegian if it has three distinct positive divisors whose sum is equal to 2022. Determine the smallest Norwegian number.', source='ISL 2022 N1', year=2022, topics=('number-theory', 'divisibility', 'bounding'),
           level=4, kind='answer', answer='1344', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), LT(X(1), X(2)), LT(X(2), X(3)), DVD(X(1), X(0)), DVD(X(2), X(0)), DVD(X(3), X(0)), EQ(lambda n, a, b, c: a + b + c, 2022)],
                     [LE(1344, X(0)), LT(6, 672), LT(672, 1344), DVD(6, 1344), DVD(672, 1344), DVD(1344, 1344), EQ(6 + 672 + 1344, 2022)],
                     domains=([0, 1343, 1344, 1345, 2688], [0, 1, 6, 7], [5, 6, 672, 673], [672, 1343, 1344, 1345]), points=2000,
                     skip_mutants={'swap the arguments of Z.dvd in (Z.dvd 1344 1344)': 'both arguments are the same number'}),
           notes='the statement says that every number with such three divisors is at least 1344, and that 1344 has them: 6 + 672 + 1344 = 2022')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 4, [lambda n, a, b_, c: v.pos(n), lambda n, a, b_, c: v.pos(a), lambda n, a, b_, c: v.lt(a, b_), lambda n, a, b_, c: v.lt(b_, c), lambda n, a, b_, c: v.dvd(a, n),
                                        lambda n, a, b_, c: v.dvd(b_, n), lambda n, a, b_, c: v.dvd(c, n), lambda n, a, b_, c: Eq(a + b_ + c, lit(2022))],
                           lambda n, a, b_, c: AND(v.le(lit(1344), n), v.lt(lit(6), lit(672)), v.lt(lit(672), lit(1344)), v.dvd(lit(6), lit(1344)), v.dvd(lit(672), lit(1344)), v.dvd(lit(1344), lit(1344)), Eq(lit(6) + lit(672) + lit(1344), lit(2022))))
