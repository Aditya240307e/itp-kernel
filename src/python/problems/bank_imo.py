"""bank_imo.py -- olympiad problems at the level of the IMO and its shortlist, integer vocabulary.  Mostly STATEMENT-ONLY (`Bank.statement`): the formal statement is in the bank, the proof is not (the tier below the reference
proofs: a problem an agent may try, and a benchmark that is not saturated); a statement-only problem is checked like any other for FAITHFULNESS (semantic.py) and, since nothing proves it, SCREENED for truth: it must hold at every
point of its box, and `imo_truth.py` re-checks every claim by brute force on a larger range.

Provenance.  `origin='contest'`: the problem of the named contest as the authors recall it, PARAPHRASED; its published answer agrees with the formal statement (`verified='answer-checked'`: checked by brute force by
`imo_truth.py`; `recalled`: the answer is from memory and the brute force covers only part of the claim).  `adapted`: a contest problem changed to fit the vocabulary (a "find all" as an equivalence with the list of
solutions, an exponent in N, a quotient as the existence of a quotient, ...); the change is in `notes`.  `style`: written for the bank in the manner of the contest.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import math
from semantic import Spec
from bank import BANK
from bank_spec import spec, EQ, LT, LE, DVD, COP, GCD, PRIME, V as X, dvd_
from bank_amc import AND, OR
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


def C(k):
    """the constant k as a function of any number of variables (the numbers of an atom)"""
    return lambda *t: k


def sol_rhs(v, xs, S, kinds=None):
    """the Prop  (x1 = s1 /\\ x2 = s2 ..) \\/ ..  for the tuples s of S; `kinds[i]` is 'N' for a variable of N"""
    kinds = kinds or ['Z'] * len(xs)
    num = lambda i, s: v.nlit(s) if kinds[i] == 'N' else L(v, s)
    return OR(*[AND(*[v.Eq(x, num(i, s)) for i, (x, s) in enumerate(zip(xs, t))]) for t in S])


def sol_atoms(S, idx):
    """the atoms of `sol_rhs` for the variables number idx[0], idx[1], .. of the Spec"""
    return [(X(i), C(t[j])) for t in S for j, i in enumerate(idx)]


def sq(n):
    """n is the square of an integer (negative numbers are not)"""
    return n >= 0 and math.isqrt(n) ** 2 == n


def is_prime_(n):
    if n < 2: return False
    d = 2
    while d * d <= n:
        if n % d == 0: return False
        d += 1
    return True


DIG = range(-1, 11)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 1960 - 1986
_S1960 = [(5, 5, 0), (8, 0, 3)]


@statement('imo1960_p1', text='Find all three-digit numbers that are divisible by 11 and whose quotient by 11 is equal to the sum of the squares of its digits.', source='IMO 1960 P1', year=1960, topics=('number-theory', 'digits', 'diophantine'),
           level=3, kind='answer', answer='550 and 803', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z'), [LT(0, X(0)), LE(X(0), 9), LE(0, X(1)), LE(X(1), 9), LE(0, X(2)), LE(X(2), 9)],
                     (lambda a, b, c: 100 * a + 10 * b + c == 11 * (a * a + b * b + c * c), lambda a, b, c: (a, b, c) in _S1960),
                     atoms=[(lambda a, b, c: 100 * a + 10 * b + c, lambda a, b, c: 11 * (a * a + b * b + c * c))] + sol_atoms(_S1960, (0, 1, 2)), domains=(DIG, DIG, DIG), points=2000),
           notes='formalised with the digits a, b, c of the number 100 a + 10 b + c: it is 11 times the sum of the squares of its digits exactly for 550 and 803')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 3, [lambda a, b_, c: v.pos(a), lambda a, b_, c: v.le(a, lit(9)), lambda a, b_, c: v.nonneg(b_), lambda a, b_, c: v.le(b_, lit(9)), lambda a, b_, c: v.nonneg(c), lambda a, b_, c: v.le(c, lit(9))],
                           lambda a, b_, c: v.Iff(Eq(100 * a + 10 * b_ + c, 11 * (a ** 2 + b_ ** 2 + c ** 2)), sol_rhs(v, [a, b_, c], _S1960)))


_S1977 = [(50, 37), (50, 7), (37, 50), (7, 50)]


@statement('imo1977_p5', text='Let a and b be positive integers. When a^2 + b^2 is divided by a + b, the quotient is q and the remainder is r. Find all pairs (a, b) for which q^2 + r = 1977.', source='IMO 1977 P5', year=1977,
           topics=('number-theory', 'diophantine'), level=5, kind='answer', answer='(a, b) = (50, 37), (50, 7), (37, 50), (7, 50)', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1))],
                     (lambda a, b: any(0 <= a * a + b * b - q * (a + b) < a + b and q * q + (a * a + b * b - q * (a + b)) == 1977 for q in range(-60, 61)), lambda a, b: (a, b) in _S1977),
                     atoms=[(C(0), lambda a, b, q: a * a + b * b - q * (a + b)), (lambda a, b, q: a * a + b * b - q * (a + b), lambda a, b, q: a + b),
                            (lambda a, b, q: q * q + (a * a + b * b - q * (a + b)), C(1977))] + sol_atoms(_S1977, (0, 1)),
                     domains=([0, 1, 2, 6, 7, 8, 36, 37, 38, 49, 50, 51], [0, 1, 2, 6, 7, 8, 36, 37, 38, 49, 50, 51]), windows={'ex': 60}),
           notes='the remainder is r = a^2 + b^2 - q (a + b) with 0 <= r < a + b, so the statement says that some q makes q^2 + r = 1977. (2a - q)^2 + (2b - q)^2 = 7908 - 2 q^2, and q = 44 is the only quotient that works')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        r = lambda a, b_, q: a ** 2 + b_ ** 2 - q * (a + b_)
        return K.statement(w, [v.Z] * 2, [lambda a, b_: v.pos(a), lambda a, b_: v.pos(b_)],
                           lambda a, b_: v.Iff(v.exists(lambda q: v.nonneg(r(a, b_, q)) & (v.lt(r(a, b_, q), a + b_) & Eq(q ** 2 + r(a, b_, q), lit(1977)))), sol_rhs(v, [a, b_], _S1977)))


_FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


@statement('imo1981_p3', text='Determine the maximum value of m^2 + n^2, where m and n are integers from 1 to 1981 satisfying (n^2 - m n - m^2)^2 = 1.', source='IMO 1981 P3', year=1981, topics=('number-theory', 'diophantine', 'bounding'),
           level=4, kind='answer', answer='987^2 + 1597^2 = 3524578', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LE(1, X(0)), LE(X(0), 1981), LE(1, X(1)), LE(X(1), 1981), EQ(lambda m, n: (n * n - m * n - m * m) ** 2, 1)],
                     lambda m, n: m * m + n * n <= 3524578 and 1597 * 1597 - 987 * 1597 - 987 * 987 == 1,
                     atoms=[(lambda m, n: m * m + n * n, C(3524578)), (C(1), C(987)), (C(987), C(1981)), (C(1), C(1597)), (C(1597), C(1981)), (C(1597 * 1597 - 987 * 1597 - 987 * 987), C(1))],
                     domains=(_FIB + [0, 4, 1981, 1982], _FIB + [0, 4, 1981, 1982]),
                     skip_mutants={'swap the arguments of Z.sub in ((Z.pow (Z.sub (Z.sub (Z.pow x1 2) (Z.mul x0 x1)) (Z.pow x0 2)) 2) = 1)': 'the hypothesis is a square: (-t)^2 = t^2'}),
           notes='formalised as: every solution has m^2 + n^2 at most 987^2 + 1597^2, and (987, 1597) is a solution. The solutions are the pairs of consecutive Fibonacci numbers, in either order')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 2, [lambda m, n: v.le(lit(1), m), lambda m, n: v.le(m, lit(1981)), lambda m, n: v.le(lit(1), n), lambda m, n: v.le(n, lit(1981)), lambda m, n: Eq((n ** 2 - m * n - m ** 2) ** 2, lit(1))],
                           lambda m, n: AND(v.le(m ** 2 + n ** 2, lit(3524578)), v.le(lit(1), lit(987)), v.le(lit(987), lit(1981)), v.le(lit(1), lit(1597)), v.le(lit(1597), lit(1981)),
                                            Eq(lit(1597) * lit(1597) - lit(987) * lit(1597) - lit(987) * lit(987), lit(1))))


@statement('imo1986_p1', text='Let d be a positive integer different from 2, 5 and 13. Show that one can find two distinct elements a, b of the set {2, 5, 13, d} such that ab - 1 is not a perfect square.', source='IMO 1986 P1', year=1986,
           topics=('number-theory', 'modular'), level=4, kind='prove', origin='adapted', verified='answer-checked',
           spec=spec(('Z',), [LT(0, X(0)), lambda d: d != 2, lambda d: d != 5, lambda d: d != 13, lambda d: sq(2 * d - 1), lambda d: sq(5 * d - 1), lambda d: sq(13 * d - 1)], lambda d: False, hatoms=False,
                     atoms=[], domains=(list(range(0, 31)),), windows={'ex': 25}, vacuous=True),
           notes='adapted: the contrapositive. The pairs 2 5, 2 13, 5 13 give the squares 9, 25, 64, so the claim is that 2d - 1, 5d - 1 and 13d - 1 are not all squares: no such d exists (the hypotheses d != 2, 5, 13 only say that the set has four elements). Checked by brute force for d up to 200000')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        issq = lambda t: v.exists(lambda k: Eq(k ** 2, t))
        return K.statement(w, [v.Z], [lambda d: v.pos(d), lambda d: v.Not(Eq(d, lit(2))), lambda d: v.Not(Eq(d, lit(5))), lambda d: v.Not(Eq(d, lit(13))), lambda d: issq(2 * d - 1), lambda d: issq(5 * d - 1), lambda d: issq(13 * d - 1)],
                           lambda d: v.False_)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 1990 - 2022
def pow2(t, top=8):
    """t is 2^x for some x in 0 .. top"""
    return any(t == 2 ** x for x in range(top + 1))


@statement('imo1990_p3', text='Determine all integers n > 1 such that n^2 divides 2^n + 1.', source='IMO 1990 P3', year=1990, topics=('number-theory', 'divisibility'), level=5, kind='answer', answer='n = 3', origin='contest',
           verified='answer-checked',
           spec=spec(('N',), [LT(1, X(0))], (lambda n: dvd_(n * n, 2 ** n + 1), lambda n: n == 3), atoms=[(lambda n: n * n, lambda n: 2 ** n + 1), (X(0), C(3))], domains=(range(0, 42),)),
           notes='n is a natural number (the exponent). The only solution is n = 3 (brute force up to 6000; the proof takes the least prime divisor p of n and the order of 2 modulo p)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N], [lambda n: w['N.lt'](v.nlit(1), n)], lambda n: v.Iff(v.dvd(ofN(n) ** 2, lit(2) ** n + 1), Eq(n, v.nlit(3))))


_S1992 = [(2, 4, 8), (3, 5, 15)]


@statement('imo1992_p1', text='Find all integer triples (a, b, c) with 1 < a < b < c such that (a - 1)(b - 1)(c - 1) divides abc - 1.', source='IMO 1992 P1', year=1992, topics=('number-theory', 'divisibility', 'diophantine'),
           level=4, kind='answer', answer='(2, 4, 8) and (3, 5, 15)', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z'), [LT(1, X(0)), LT(X(0), X(1)), LT(X(1), X(2))],
                     (lambda a, b, c: dvd_((a - 1) * (b - 1) * (c - 1), a * b * c - 1), lambda a, b, c: (a, b, c) in _S1992),
                     atoms=[(lambda a, b, c: (a - 1) * (b - 1) * (c - 1), lambda a, b, c: a * b * c - 1)] + sol_atoms(_S1992, (0, 1, 2)),
                     domains=([0, 1, 2, 3, 4], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 8, 9, 15, 16])),
           notes='a > 1 and (a - 1)(b - 1)(c - 1) < abc - 1 force the quotient k to be 2 or 3 (brute force: a < 60, b < 120, c < 400)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 3, [lambda a, b_, c: v.lt(lit(1), a), lambda a, b_, c: v.lt(a, b_), lambda a, b_, c: v.lt(b_, c)],
                           lambda a, b_, c: v.Iff(v.dvd((a - 1) * (b_ - 1) * (c - 1), a * b_ * c - 1), sol_rhs(v, [a, b_, c], _S1992)))


_S1994 = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 5), (3, 1), (3, 5), (5, 2), (5, 3)]


@statement('imo1994_p4', text='Determine all ordered pairs (m, n) of positive integers such that (n^3 + 1) / (mn - 1) is an integer.', source='IMO 1994 P4', year=1994, topics=('number-theory', 'divisibility', 'diophantine'), level=4,
           kind='answer', answer='(1, 2), (1, 3), (2, 1), (2, 2), (2, 5), (3, 1), (3, 5), (5, 2), (5, 3)', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1))], (lambda m, n: dvd_(m * n - 1, n ** 3 + 1), lambda m, n: (m, n) in _S1994),
                     atoms=[(lambda m, n: m * n - 1, lambda m, n: n ** 3 + 1)] + sol_atoms(_S1994, (0, 1)), domains=(range(-1, 8), range(-1, 8))),
           notes='mn - 1 = 0 (m = n = 1) divides nothing but 0, so (1, 1) is not a solution; brute force for m, n < 200')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 2, [lambda m, n: v.pos(m), lambda m, n: v.pos(n)], lambda m, n: v.Iff(v.dvd(m * n - 1, n ** 3 + 1), sol_rhs(v, [m, n], _S1994)))


_S1997 = [(1, 1), (16, 2), (27, 3)]


@statement('imo1997_p5', text='Find all pairs (a, b) of integers a, b >= 1 that satisfy the equation a^(b^2) = b^a.', source='IMO 1997 P5', year=1997, topics=('number-theory', 'diophantine'), level=5, kind='answer',
           answer='(1, 1), (16, 2), (27, 3)', origin='contest', verified='answer-checked',
           spec=spec(('N', 'N'), [LT(0, X(0)), LT(0, X(1))], (lambda a, b: a ** (b * b) == b ** a, lambda a, b: (a, b) in _S1997),
                     atoms=[(lambda a, b: a ** (b * b), lambda a, b: b ** a)] + sol_atoms(_S1997, (0, 1)), domains=([0, 1, 2, 3, 4, 5, 8, 16, 17, 26, 27, 28], [0, 1, 2, 3, 4, 5, 6])),
           notes='a and b are natural numbers (they are exponents); brute force for a < 80, b < 12')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        return K.statement(w, [v.N, v.N], [lambda a, b_: w['N.lt'](v.nlit(0), a), lambda a, b_: w['N.lt'](v.nlit(0), b_)],
                           lambda a, b_: v.Iff(Eq(ofN(a) ** w['N.mul'](b_, b_), ofN(b_) ** a), sol_rhs(v, [a, b_], _S1997, ['N', 'N'])))


@statement('imo1998_p4', text='Determine all pairs (a, b) of positive integers such that ab^2 + b + 7 divides a^2 b + a + b.', source='IMO 1998 P4', year=1998, topics=('number-theory', 'divisibility', 'diophantine'), level=5,
           kind='answer', answer='(11, 1), (49, 1) and (7k^2, 7k) for k >= 1', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1))],
                     (lambda a, b: dvd_(a * b * b + b + 7, a * a * b + a + b), lambda a, b: (a, b) in ((11, 1), (49, 1)) or any(a == 7 * k * k and b == 7 * k for k in range(1, 13))),
                     atoms=[(lambda a, b: a * b * b + b + 7, lambda a, b: a * a * b + a + b), (X(0), C(11)), (X(1), C(1)), (X(0), C(49)), (X(1), C(1)), (C(0), lambda a, b, k: k), (X(0), lambda a, b, k: 7 * k * k), (X(1), lambda a, b, k: 7 * k)],
                     domains=([0, 1, 2, 7, 8, 11, 12, 28, 29, 49, 50, 63, 112], [0, 1, 2, 7, 8, 14, 21, 22, 28]), windows={'ex': 12}),
           notes='the solutions with b = 1 are a = 11 and a = 49; with b > 1: b = 7k, a = 7k^2 (brute force a < 800, b < 120)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 2, [lambda a, b_: v.pos(a), lambda a, b_: v.pos(b_)],
                           lambda a, b_: v.Iff(v.dvd(a * b_ ** 2 + b_ + 7, a ** 2 * b_ + a + b_),
                                               OR(AND(Eq(a, lit(11)), Eq(b_, lit(1))), AND(Eq(a, lit(49)), Eq(b_, lit(1))), v.exists(lambda k: v.pos(k) & (Eq(a, 7 * k ** 2) & Eq(b_, 7 * k))))))


@statement('imo1999_p4', text='Determine all pairs (n, p) of positive integers such that p is a prime, n <= 2p, and (p - 1)^n + 1 is divisible by n^(p - 1).', source='IMO 1999 P4', year=1999,
           topics=('number-theory', 'divisibility', 'prime'), level=5, kind='answer', answer='(1, p) for every prime p, (2, 2) and (3, 3)', origin='adapted', verified='answer-checked',
           spec=spec(('N', 'N'), [PRIME(lambda e, n: e + 1), LT(0, X(1)), LE(X(1), lambda e, n: 2 * (e + 1))],
                     (lambda e, n: dvd_(n ** e, e ** n + 1), lambda e, n: n == 1 or (e, n) in ((1, 2), (2, 3))),
                     atoms=[(lambda e, n: n ** e, lambda e, n: e ** n + 1), (X(1), C(1)), (X(0), C(1)), (X(1), C(2)), (X(0), C(2)), (X(1), C(3))], domains=(range(0, 15), range(0, 31))),
           notes='adapted: the exponent p - 1 is a natural number e, so the prime is p = e + 1 and the pairs are (n, e + 1); checked for the primes p <= 23')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N, v.N], [lambda e, n: w['Z.prime'](ofN(e) + 1), lambda e, n: w['N.lt'](v.nlit(0), n), lambda e, n: v.le(ofN(n), 2 * (ofN(e) + 1))],
                           lambda e, n: v.Iff(v.dvd(ofN(n) ** e, ofN(e) ** n + 1), OR(Eq(n, v.nlit(1)), AND(Eq(e, v.nlit(1)), Eq(n, v.nlit(2))), AND(Eq(e, v.nlit(2)), Eq(n, v.nlit(3))))))


@statement('imo2003_p2', text='Find all pairs (a, b) of positive integers such that a^2 / (2ab^2 - b^3 + 1) is a positive integer.', source='IMO 2003 P2', year=2003, topics=('number-theory', 'divisibility', 'diophantine'), level=5,
           kind='answer', answer='(2l, 1), (l, 2l), (8l^4 - l, 2l) for l >= 1', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1))],
                     (lambda a, b: 2 * a * b * b - b ** 3 + 1 > 0 and dvd_(2 * a * b * b - b ** 3 + 1, a * a),
                      lambda a, b: any((a == 2 * l and b == 1) or (a == l and b == 2 * l) or (a == 8 * l ** 4 - l and b == 2 * l) for l in range(1, 131))),
                     atoms=[(C(0), lambda a, b: 2 * a * b * b - b ** 3 + 1), (lambda a, b: 2 * a * b * b - b ** 3 + 1, lambda a, b: a * a), (C(0), lambda a, b, l: l), (X(0), lambda a, b, l: 2 * l), (X(1), C(1)),
                            (X(0), lambda a, b, l: l), (X(1), lambda a, b, l: 2 * l), (X(0), lambda a, b, l: 8 * l ** 4 - l), (X(1), lambda a, b, l: 2 * l)],
                     domains=([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 124, 125, 126, 127], [0, 1, 2, 3, 4, 5, 6, 7, 8]), windows={'ex': 130}),
           notes='a quotient a^2 / d is a positive integer exactly when d > 0 and d divides a^2. For b = 1: d = 2a; for b = 2l: d = a (a + ...) gives the other two families (brute force a < 400, b < 100)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        den = lambda a, b_: 2 * a * b_ ** 2 - b_ ** 3 + 1
        return K.statement(w, [v.Z] * 2, [lambda a, b_: v.pos(a), lambda a, b_: v.pos(b_)],
                           lambda a, b_: v.Iff(v.pos(den(a, b_)) & v.dvd(den(a, b_), a ** 2),
                                               v.exists(lambda l: v.pos(l) & OR(AND(Eq(a, 2 * l), Eq(b_, lit(1))), AND(Eq(a, l), Eq(b_, 2 * l)), AND(Eq(a, 8 * l ** 4 - l), Eq(b_, 2 * l))))))


def exN(w, f):
    """exists x : N, f(x)"""
    N = w['N']
    return w['Ex'](N, w.lam(N, f))


@statement('imo2005_p4', text='Consider the sequence a_n = 2^n + 3^n + 6^n - 1 (n = 1, 2, ...). Determine all positive integers that are relatively prime to every term of the sequence.', source='IMO 2005 P4', year=2005,
           topics=('number-theory', 'gcd', 'prime'), level=5, kind='answer', answer='only 1', origin='contest', verified='answer-checked',
           spec=spec(('Z',), [LT(0, X(0))], (lambda k: all(math.gcd(k, 2 ** n + 3 ** n + 6 ** n - 1) == 1 for n in range(1, 15)), lambda k: k == 1),
                     atoms=[(C(1), lambda k, n: n), (X(0), lambda k, n: 2 ** n + 3 ** n + 6 ** n - 1), (X(0), C(1))], domains=(range(0, 14),), windows={'n': 14}),
           notes='for a prime p >= 5 the term a_(p-2) is divisible by p (Fermat); for 2 and 3 use a_1 = 10 and a_2 = 48. The brute force covers k < 400 and n < 400. The quantifier over n is evaluated up to n = 14')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z], [lambda k: v.pos(k)],
                           lambda k: v.Iff(v.F([v.N], lambda n: w['N.le'](v.nlit(1), n) >> w['Z.coprime'](k, lit(2) ** n + lit(3) ** n + lit(6) ** n - 1)), Eq(k, lit(1))))


_S2006 = [(0, 2), (0, -2), (4, 23), (4, -23)]


@statement('imo2006_p4', text='Determine all pairs (x, y) of integers such that 1 + 2^x + 2^(2x + 1) = y^2.', source='IMO 2006 P4', year=2006, topics=('number-theory', 'diophantine'), level=5, kind='answer',
           answer='(0, 2), (0, -2), (4, 23), (4, -23)', origin='adapted', verified='answer-checked',
           spec=spec(('N', 'Z'), [], (lambda x, y: 1 + 2 ** x + 2 ** (2 * x + 1) == y * y, lambda x, y: (x, y) in _S2006),
                     atoms=[(lambda x, y: 1 + 2 ** x + 2 ** (2 * x + 1), lambda x, y: y * y)] + sol_atoms(_S2006, (0, 1)), domains=(range(0, 13), range(-30, 31))),
           notes='adapted: x is a natural number (a negative x makes 2^x a fraction); brute force for x < 60')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N, v.Z], [], lambda x, y: v.Iff(Eq(1 + lit(2) ** x + lit(2) ** w['N.succ'](w['N.double'](x)), y ** 2), sol_rhs(v, [x, y], _S2006, ['N', 'Z'])))


@statement('imo2007_p5', text='Let a and b be positive integers such that 4ab - 1 divides (4a^2 - 1)^2. Prove that a = b.', source='IMO 2007 P5', year=2007, topics=('number-theory', 'divisibility', 'descent'), level=5, kind='prove',
           origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), DVD(lambda a, b: 4 * a * b - 1, lambda a, b: (4 * a * a - 1) ** 2)], EQ(X(0), X(1)), domains=(range(0, 14), range(0, 14))),
           notes='the proof is by Vieta jumping (descent); brute force for a, b < 300')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 2, [lambda a, b_: v.pos(a), lambda a, b_: v.pos(b_), lambda a, b_: v.dvd(4 * a * b_ - 1, (4 * a ** 2 - 1) ** 2)], lambda a, b_: Eq(a, b_))


_S2015 = sorted({p for t in [(2, 2, 2), (2, 2, 3), (2, 6, 11), (3, 5, 7)] for p in __import__('itertools').permutations(t)})


@statement('imo2015_p2', text='Determine all triples (a, b, c) of positive integers such that each of the numbers ab - c, bc - a, ca - b is a power of 2. (A power of 2 is 2^n for an integer n >= 0.)', source='IMO 2015 P2', year=2015,
           topics=('number-theory', 'diophantine'), level=5, kind='answer', answer='(2, 2, 2), (2, 2, 3), (2, 6, 11), (3, 5, 7) and their permutations (16 triples)', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), LT(0, X(2))],
                     (lambda a, b, c: pow2(a * b - c) and pow2(b * c - a) and pow2(c * a - b), lambda a, b, c: (a, b, c) in _S2015),
                     atoms=[(lambda a, b, c, x: a * b - c, lambda a, b, c, x: 2 ** x), (lambda a, b, c, y: b * c - a, lambda a, b, c, y: 2 ** y), (lambda a, b, c, z: c * a - b, lambda a, b, c, z: 2 ** z)] + sol_atoms(_S2015, (0, 1, 2)),
                     domains=([0, 1, 2, 3, 4, 5, 6, 7, 11], [0, 1, 2, 3, 4, 5, 6, 7, 11], [0, 1, 2, 3, 4, 5, 6, 7, 11]), windows={'n': 8}, points=800),
           notes='the 16 ordered triples are the permutations of the four listed; brute force for a, b, c < 80. The exponents are evaluated up to 8')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 3, [lambda a, b_, c: v.pos(a), lambda a, b_, c: v.pos(b_), lambda a, b_, c: v.pos(c)],
                           lambda a, b_, c: v.Iff(AND(exN(w, lambda x: Eq(a * b_ - c, lit(2) ** x)), exN(w, lambda y: Eq(b_ * c - a, lit(2) ** y)), exN(w, lambda z: Eq(c * a - b_, lit(2) ** z))),
                                                  sol_rhs(v, [a, b_, c], _S2015)))


@statement('imo2022_p5', text='Find all triples (a, b, p) of positive integers with p prime and a^p = b! + p.', source='IMO 2022 P5', year=2022, topics=('number-theory', 'diophantine', 'prime'), level=5, kind='answer',
           answer='(2, 2, 2) and (3, 4, 3)', origin='contest', verified='answer-checked',
           spec=spec(('N', 'N', 'N'), [LT(0, X(0)), LT(0, X(1)), PRIME(X(2))], (lambda a, b, p: a ** p == math.factorial(b) + p, lambda a, b, p: (a, b, p) in [(2, 2, 2), (3, 4, 3)]),
                     atoms=[(lambda a, b, p: a ** p, lambda a, b, p: math.factorial(b) + p)] + sol_atoms([(2, 2, 2), (3, 4, 3)], (0, 1, 2)), domains=(range(0, 6), range(0, 7), range(0, 8))),
           notes='brute force for b < 25 and the primes p < 32')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        return K.statement(w, [v.N] * 3, [lambda a, b_, p: w['N.lt'](v.nlit(0), a), lambda a, b_, p: w['N.lt'](v.nlit(0), b_), lambda a, b_, p: w['Z.prime'](ofN(p))],
                           lambda a, b_, p: v.Iff(Eq(ofN(a) ** p, w['Z.fact'](b_) + ofN(p)), sol_rhs(v, [a, b_, p], [(2, 2, 2), (3, 4, 3)], ['N', 'N', 'N'])))


@statement('imo2001_p6', text='Let a > b > c > d be positive integers and suppose that ac + bd = (b + d + a - c)(b + d - a + c). Prove that ab + cd is not prime.', source='IMO 2001 P6', year=2001, topics=('number-theory', 'prime', 'diophantine'),
           level=5, kind='prove', origin='contest', verified='answer-checked',
           spec=spec(('Z',) * 4, [LT(0, X(3)), LT(X(3), X(2)), LT(X(2), X(1)), LT(X(1), X(0)), EQ(lambda a, b, c, d: a * c + b * d, lambda a, b, c, d: (b + d + a - c) * (b + d - a + c))],
                     lambda a, b, c, d: not is_prime_(a * b + c * d), atoms=[(lambda a, b, c, d: a * b + c * d,)],
                     domains=([9, 10, 11, 12, 13, 14], [8, 9, 10, 11, 12], [3, 4, 5, 6, 7, 9], [0, 1, 2, 3])),
           notes='the hypothesis says a^2 - ac + c^2 = b^2 + bd + d^2; brute force: all 341 solutions with a < 60 have ab + cd composite')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 4, [lambda a, b_, c, d: v.pos(d), lambda a, b_, c, d: v.lt(d, c), lambda a, b_, c, d: v.lt(c, b_), lambda a, b_, c, d: v.lt(b_, a),
                                          lambda a, b_, c, d: Eq(a * c + b_ * d, (b_ + d + a - c) * (b_ + d - a + c))], lambda a, b_, c, d: v.Not(w['Z.prime'](a * b_ + c * d)))


def _g1987_zero(n): return -1 if n == 0 else n + 993                                  # not >= 0 at 0 only: tells `0 <= n` from `0 < n` in the first hypothesis
def _g1987_partial(n): return n + 1987 if 0 <= n <= 8 else (n if n >= 1987 else 0)       # f(f(n)) = f(n + 1987) = n + 1987 for 0 <= n <= 8 (a solution on the window of the box, and on nothing larger)


def _g1987_no0(n): return n + 1987 if 1 <= n <= 9 else (n if n >= 1988 else 0)              # the same without n = 0: tells `0 <= n` from `0 < n` in the second hypothesis


_F1987 = [lambda n: n + 993, lambda n: n + 994, lambda n: 0, lambda n: -1, lambda n: n, lambda n: n + 1987, _g1987_zero, _g1987_partial, _g1987_no0, lambda n: n + 993 if n >= 0 else -5, lambda n: n + 993 if n != 0 else 7,
          lambda n: n + 993 if n != 3 else 100, lambda n: abs(n) + 993, lambda n: 1987 - n]


@statement('imo1987_p4', text='Prove that there is no function f from the non-negative integers to the non-negative integers such that f(f(n)) = n + 1987 for all n.', source='IMO 1987 P4', year=1987, topics=('functional-equation', 'number-theory'),
           level=5, kind='prove', origin='contest', verified='recalled',
           spec=spec(('fn',), [lambda f: all(f(n) >= 0 for n in range(0, 9)), lambda f: all(f(f(n)) == n + 1987 for n in range(0, 9))], lambda f: False, hatoms=False, atoms=[], fns=_F1987, vacuous=True),
           notes='an impossibility: f is a function on the integers whose values on the non-negative integers are non-negative and satisfy the equation there (a function on the non-negative integers is such a function, with any values on the negative ones); '
                 'no such function exists: f(n + 1987) = f(n) + 1987, so f permutes the residues modulo 1987 as an involution, which has a fixed point because 1987 is odd, and there f(r) = r + 1987 q gives 2q = 1. '
                 'The box sees only a window of n, so test functions that solve the equation on the window exist (they are what tells the quantifiers from their slips); imo_truth.py checks the finite reduction for odd and even shifts.')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        return K.statement(w, [fZ], [lambda f: v.forall(lambda n: v.nonneg(n) >> v.nonneg(f(n))), lambda f: v.forall(lambda n: v.nonneg(n) >> Eq(f(f(n)), n + lit(1987)))], lambda f: v.False_)
