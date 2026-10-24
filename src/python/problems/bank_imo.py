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


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the second batch (1962 - 1996)
fact = math.factorial


@statement('imo1962_p1', text='Find the smallest positive integer n with these properties: (a) its decimal representation ends in the digit 6; (b) if this last digit 6 is erased and placed in front of the remaining digits, the resulting number is four times n.',
           source='IMO 1962 P1', year=1962, topics=('number-theory', 'digits'), level=3, kind='answer', answer='153846', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'N'), [LT(0, X(0)), LE(lambda m, k: 10 ** k, lambda m, k: 10 * m), LT(lambda m, k: 10 * m, lambda m, k: 10 ** (k + 1)), EQ(lambda m, k: 4 * (10 * m + 6), lambda m, k: 6 * 10 ** k + m)],
                     [LE(153846, lambda m, k: 10 * m + 6), LE(10 ** 5, 153840), LT(153840, 10 ** 6), EQ(4 * 153846, 6 * 10 ** 5 + 15384)], domains=([0, 1, 2, 20, 15383, 15384, 15385], range(0, 8))),
           notes='n = 10 m + 6, where m (with k digits, 10^k <= 10 m < 10^(k+1)) is the number left when the 6 is erased; moving the 6 to the front gives 6 * 10^k + m. The statement says that every such n is at least 153846, and that 153846 is one '
                 '(m = 15384, k = 5: 4 * 153846 = 615384 = 6 * 10^5 + 15384)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit; p = lambda k: lit(10) ** k
        return K.statement(w, [v.Z, v.N], [lambda m, k: v.pos(m), lambda m, k: v.le(p(k), 10 * m), lambda m, k: v.lt(10 * m, p(w['N.succ'](k))), lambda m, k: Eq(4 * (10 * m + 6), 6 * p(k) + m)],
                           lambda m, k: AND(v.le(lit(153846), 10 * m + 6), v.le(p(v.nlit(5)), lit(153840)), v.lt(lit(153840), p(v.nlit(6))), Eq(4 * lit(153846), 6 * p(v.nlit(5)) + lit(15384))))


@statement('imo1972_p3', text='Let m and n be arbitrary non-negative integers. Prove that (2m)! (2n)! / (m! n! (m + n)!) is an integer.', source='IMO 1972 P3', year=1972, topics=('number-theory', 'product', 'divisibility'), level=4, kind='prove',
           origin='contest', verified='answer-checked',
           spec=spec(('N', 'N'), [], DVD(lambda m, n: fact(m) * fact(n) * fact(m + n), lambda m, n: fact(2 * m) * fact(2 * n)), domains=(range(0, 13), range(0, 13))),
           notes='the quotient is an integer exactly when the denominator divides the numerator; checked for m, n <= 12 here and by imo_truth.py for m, n <= 60')
def _(w, b):
    v = K.Vocab(w); f = w['Z.fact']; add = w['N.add']; dbl = w['N.double']
    with v.ints():
        return K.statement(w, [v.N, v.N], [], lambda m, n: v.dvd(f(m) * f(n) * f(add(m, n)), f(dbl(m)) * f(dbl(n))))


@statement('imo1978_p1', text='Let m and n be natural numbers with 1 <= m < n. The last three digits of 1978^m are the same as the last three digits of 1978^n. Find m and n for which m + n is as small as possible.', source='IMO 1978 P1', year=1978,
           topics=('number-theory', 'modular', 'bounding'), level=4, kind='answer', answer='m = 3, n = 103', origin='contest', verified='answer-checked',
           spec=spec(('N', 'N'), [LE(1, X(0)), LT(X(0), X(1)), DVD(1000, lambda m, n: 1978 ** n - 1978 ** m)], [LE(106, lambda m, n: m + n), DVD(1000, 1978 ** 103 - 1978 ** 3)],
                     domains=(range(0, 6), [0, 1, 2, 3, 4, 50, 100, 101, 102, 103, 104, 105, 106])),
           notes='"the last three digits agree" is 1000 | 1978^n - 1978^m. The statement says that every such pair has m + n >= 106, and that (3, 103) is a pair (m + n = 106)')
def _(w, b):
    v = K.Vocab(w); ofN = w['Z.of_N']
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.N, v.N], [lambda m, n: w['N.le'](v.nlit(1), m), lambda m, n: w['N.lt'](m, n), lambda m, n: v.dvd(lit(1000), lit(1978) ** n - lit(1978) ** m)],
                           lambda m, n: AND(v.le(lit(106), ofN(m) + ofN(n)), v.dvd(lit(1000), lit(1978) ** v.nlit(103) - lit(1978) ** v.nlit(3))))


def _F82(x, y): return x ** 3 - 3 * x * y * y + y ** 3


@statement('imo1982_p4a', text='Prove that if a positive integer n can be written as n = x^3 - 3 x y^2 + y^3 with integers x and y, then it has at least three such representations (three different pairs (x, y)).', source='IMO 1982 P4 (a)', year=1982,
           topics=('number-theory', 'diophantine', 'identity'), level=4, kind='prove', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z'), [LT(0, X(2)), EQ(lambda x, y, n: _F82(x, y), X(2))],
                     lambda x, y, n: (_F82(y - x, -x) == n and _F82(-y, x - y) == n and not (x == y - x and y == -x) and not (x == -y and y == x - y) and not (y - x == -y and -x == x - y)),
                     atoms=[(lambda x, y, n: _F82(y - x, -x), X(2)), (lambda x, y, n: _F82(-y, x - y), X(2)), (X(0), lambda x, y, n: y - x), (X(1), lambda x, y, n: -x), (X(0), lambda x, y, n: -y), (X(1), lambda x, y, n: x - y),
                            (lambda x, y, n: y - x, lambda x, y, n: -y), (lambda x, y, n: -x, lambda x, y, n: x - y)],
                     domains=(range(-4, 5), range(-4, 5), [-9, -2, -1, 0, 1, 2, 3, 5, 9, 25]),
                     skip_mutants={'and -> or in (~((x0 = (Z.sub x1 x0)) /\\ (x1 = (Z.neg x0))) /\\ (~(': 'each of the three pairs equals another one only at the origin: the three non-equalities are equivalent',
                                   'and -> or in (~((x0 = (Z.neg x1)) /\\ (x1 = (Z.sub x0 x1))) /\\ ~(': 'the same: the last two non-equalities are equivalent'}),
           notes='the pairs (x, y), (y - x, -x), (-y, x - y) all represent n, and they are pairwise different when n > 0 (a repeated pair would be a fixed point of the map (x, y) -> (y - x, -x), whose only fixed point is the origin)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        F = lambda x, y: x ** 3 - 3 * x * y ** 2 + y ** 3
        ne = lambda a1, b1, a2, b2: v.Not(AND(Eq(a1, a2), Eq(b1, b2)))
        return K.statement(w, [v.Z] * 3, [lambda x, y, n: v.pos(n), lambda x, y, n: Eq(F(x, y), n)],
                           lambda x, y, n: AND(Eq(F(y - x, -x), n), Eq(F(-y, x - y), n), ne(x, y, y - x, -x), ne(x, y, -y, x - y), ne(y - x, -x, -y, x - y)))


@statement('imo1982_p4b', text='Show that the equation x^3 - 3 x y^2 + y^3 = 2891 has no solution in integers x, y.', source='IMO 1982 P4 (b)', year=1982, topics=('number-theory', 'diophantine', 'modular'), level=4, kind='prove',
           origin='contest', verified='recalled',
           spec=spec(('Z', 'Z'), [], lambda x, y: _F82(x, y) != 2891, atoms=[(lambda x, y: _F82(x, y), C(2891))], domains=(range(-9, 10), range(-9, 10))),
           notes='no integer pair gives 2891 in the box and in the range checked by imo_truth.py (|x|, |y| <= 400); the proof is a computation modulo 9')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 2, [], lambda x, y: v.Not(Eq(x ** 3 - 3 * x * y ** 2 + y ** 3, v.lit(2891))))


@statement('imo1984_p6', text='Let a, b, c, d be odd positive integers with a < b < c < d and ad = bc. Suppose that a + d = 2^k and b + c = 2^m for some integers k and m. Prove that a = 1.', source='IMO 1984 P6', year=1984,
           topics=('number-theory', 'diophantine', 'bounding'), level=5, kind='prove', origin='adapted', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z', 'Z', 'N', 'N'), [LE(0, X(0)), LT(X(0), X(1)), LT(X(1), X(2)), LT(X(2), X(3)), EQ(lambda x, y, z, t, k, m: (2 * x + 1) * (2 * t + 1), lambda x, y, z, t, k, m: (2 * y + 1) * (2 * z + 1)),
                                                 EQ(lambda x, y, z, t, k, m: 2 * x + 2 * t + 2, lambda x, y, z, t, k, m: 2 ** k), EQ(lambda x, y, z, t, k, m: 2 * y + 2 * z + 2, lambda x, y, z, t, k, m: 2 ** m)],
                     EQ(X(0), 0), domains=([-1, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 7, 8], [3, 4, 5], [2, 3, 4]), points=3000),
           notes='the odd numbers are written a = 2x + 1, b = 2y + 1, c = 2z + 1, d = 2t + 1 with x, y, z, t >= 0 (so a < b is x < y, and a = 1 is x = 0); a + d = 2x + 2t + 2 and b + c = 2y + 2z + 2')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 4 + [v.N] * 2,
                           [lambda x, y, z, t, k, m: v.nonneg(x), lambda x, y, z, t, k, m: v.lt(x, y), lambda x, y, z, t, k, m: v.lt(y, z), lambda x, y, z, t, k, m: v.lt(z, t),
                            lambda x, y, z, t, k, m: Eq((2 * x + 1) * (2 * t + 1), (2 * y + 1) * (2 * z + 1)), lambda x, y, z, t, k, m: Eq(2 * x + 2 * t + 2, lit(2) ** k),
                            lambda x, y, z, t, k, m: Eq(2 * y + 2 * z + 2, lit(2) ** m)], lambda x, y, z, t, k, m: Eq(x, v.zero))


@statement('imo1996_p4', text='The positive integers a and b are such that the numbers 15a + 16b and 16a - 15b are both squares of positive integers. What is the least possible value of the smaller of these two squares?', source='IMO 1996 P4', year=1996,
           topics=('number-theory', 'diophantine', 'bounding'), level=5, kind='answer', answer='481^2 = 231361', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), LT(0, X(2)), LT(0, X(3)), EQ(lambda a, b, r, s: 15 * a + 16 * b, lambda a, b, r, s: r * r), EQ(lambda a, b, r, s: 16 * a - 15 * b, lambda a, b, r, s: s * s)],
                     [LE(481, X(2)), LE(481, X(3)), EQ(15 * 14911 + 16 * 481, 481 * 481), EQ(16 * 14911 - 15 * 481, 481 * 481)],
                     domains=([0, 1, 3, 14910, 14911, 14912], [0, 1, 3, 480, 481, 482], [0, 1, 480, 481, 482], [0, 1, 480, 481, 482]), points=2000),
           notes='the two squares are r^2 and s^2; the statement says that r and s are both at least 481 (so the smaller square is at least 481^2), and that a = 14911, b = 481 make both squares equal to 481^2')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z] * 4, [lambda a, b_, r, s: v.pos(a), lambda a, b_, r, s: v.pos(b_), lambda a, b_, r, s: v.pos(r), lambda a, b_, r, s: v.pos(s),
                                        lambda a, b_, r, s: Eq(15 * a + 16 * b_, r * r), lambda a, b_, r, s: Eq(16 * a - 15 * b_, s * s)],
                           lambda a, b_, r, s: AND(v.le(lit(481), r), v.le(lit(481), s), Eq(15 * lit(14911) + 16 * lit(481), lit(481) * lit(481)), Eq(16 * lit(14911) - 15 * lit(481), lit(481) * lit(481))))


@statement('imo1989_p5', text='Prove that for each positive integer n there exist n consecutive positive integers none of which is a prime or a prime power.', source='IMO 1989 P5', year=1989, topics=('number-theory', 'prime', 'construction'), level=4,
           kind='prove', origin='adapted', verified='answer-checked',
           spec=spec(('Z',), [LT(0, X(0))], lambda n: any(m > 0 and all(any(a > 1 and b > 1 and math.gcd(a, b) == 1 and a * b == m + i for a in range(-40, 41) for b in range(-40, 41)) for i in range(1, n + 1)) for m in range(-40, 41)),
                     atoms=[(C(0), lambda n, m: m), (C(1), lambda n, m, i: i), (lambda n, m, i: i, lambda n, m, i: n), (C(1), lambda n, m, i, a, b: a), (C(1), lambda n, m, i, a, b: b),
                            (lambda n, m, i, a, b: a, lambda n, m, i, a, b: b), (lambda n, m, i, a, b: a * b, lambda n, m, i, a, b: m + i)], domains=([0, 1, 2, 3],), windows={'ex': 40},
                     body=(('Z',), lambda n, m: m > 0 and all(any(a > 1 and b > 1 and math.gcd(a, b) == 1 and a * b == m + i for a in range(-40, 41) for b in range(-40, 41)) for i in range(1, n + 1)))),
           notes='"neither a prime nor a prime power" for a number m + i >= 2 means that it has two different prime divisors, i.e. it is a product a b of two coprime numbers a, b > 1. The test range is n <= 3 (the first three consecutive numbers '
                 'of this kind are 20, 21, 22); imo_truth.py finds them for every n <= 7')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z], [lambda n: v.pos(n)],
                           lambda n: v.exists(lambda m: v.pos(m) & v.forall(lambda i: (v.le(lit(1), i) & v.le(i, n)) >> v.exists(lambda a: v.exists(lambda b_: AND(v.lt(lit(1), a), v.lt(lit(1), b_), w['Z.coprime'](a, b_), Eq(a * b_, m + i)))))))


def _fam1(k): return lambda x: k * x * x
def _fam2(k): return lambda x: 0 if x % 2 == 0 else k
def _fam3(k): return lambda x: 0 if x % 4 == 0 else (k if x % 2 else 4 * k)


def _lhs2012(f): return all(f(a) ** 2 + f(b) ** 2 + f(-a - b) ** 2 == 2 * f(a) * f(b) + 2 * f(b) * f(-a - b) + 2 * f(-a - b) * f(a) for a in range(-8, 9) for b in range(-8, 9) if abs(a + b) <= 8)


def _rhs2012(f):
    k = f(1)
    return all(f(x) == k * x * x for x in range(-8, 9)) or all(f(x) == (0 if x % 2 == 0 else k) for x in range(-8, 9)) or all(f(x) == (0 if x % 4 == 0 else (k if x % 2 else 4 * k)) for x in range(-8, 9))


@statement('imo2012_p4', text='Find all functions f from the integers to the integers such that, for all integers a, b, c with a + b + c = 0, f(a)^2 + f(b)^2 + f(c)^2 = 2 f(a) f(b) + 2 f(b) f(c) + 2 f(c) f(a).', source='IMO 2012 P4', year=2012,
           topics=('functional-equation', 'number-theory'), level=5, kind='answer',
           answer='f(x) = k x^2; f(x) = k for odd x and 0 for even x; f(x) = 0 for x = 0 mod 4, k for odd x, 4k for x = 2 mod 4 (k any integer)', origin='contest', verified='answer-checked',
           spec=spec(('fn',), [], (_lhs2012, _rhs2012),
                     atoms=[(lambda f, a, b, c: a + b + c, C(0)), (lambda f, a, b, c: f(a) ** 2 + f(b) ** 2 + f(c) ** 2, lambda f, a, b, c: 2 * f(a) * f(b) + 2 * f(b) * f(c) + 2 * f(c) * f(a)),
                            (lambda f, x: f(x), lambda f, x: f(1) * x * x),
                            (C(2), lambda f, x: x), (lambda f, x: f(x), C(0)), (C(2), lambda f, x: x), (lambda f, x: f(x), lambda f, x: f(1)),
                            (C(4), lambda f, x: x), (lambda f, x: f(x), C(0)), (C(2), lambda f, x: x), (lambda f, x: f(x), lambda f, x: f(1)), (C(2), lambda f, x: x), (C(4), lambda f, x: x), (lambda f, x: f(x), lambda f, x: 4 * f(1))],
                     extra_fns=[g(k) for k in (-3, -2, -1, 1, 2, 3) for g in (_fam1, _fam2, _fam3)] + [lambda x: x % 4]),
           notes='the three families of solutions, written with k = f(1) (in every family f(1) = k; k = 0 is the zero function in all three); the test functions of the box include members of each family. imo_truth.py solves the equation on |x| <= 16 by exhaustive search and finds exactly these')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit; two = lit(2); four = lit(4); z0 = v.zero
        def lhs(f): return v.F([v.Z, v.Z, v.Z], lambda a, b_, c: Eq(a + b_ + c, z0) >> Eq(f(a) ** 2 + f(b_) ** 2 + f(c) ** 2, 2 * f(a) * f(b_) + 2 * f(b_) * f(c) + 2 * f(c) * f(a)))
        def fam1(f, k): return v.forall(lambda x: Eq(f(x), k * x ** 2))
        def fam2(f, k): return v.forall(lambda x: AND(v.dvd(two, x) >> Eq(f(x), z0), v.Not(v.dvd(two, x)) >> Eq(f(x), k)))
        def fam3(f, k): return v.forall(lambda x: AND(v.dvd(four, x) >> Eq(f(x), z0), v.Not(v.dvd(two, x)) >> Eq(f(x), k), AND(v.dvd(two, x), v.Not(v.dvd(four, x))) >> Eq(f(x), 4 * k)))
        return K.statement(w, [fZ], [], lambda f: v.Iff(lhs(f), OR(fam1(f, f(lit(1))), fam2(f, f(lit(1))), fam3(f, f(lit(1))))))


def _tri2009(f): return all(a < f(b) + f(b + f(a) - 1) and f(b) < a + f(b + f(a) - 1) and f(b + f(a) - 1) < a + f(b) for a in range(1, 9) for b in range(1, 9))


@statement('imo2009_p5', text='Determine all functions f from the positive integers to the positive integers such that, for all positive integers a and b, there is a non-degenerate triangle with sides of lengths a, f(b) and f(b + f(a) - 1).', source='IMO 2009 P5',
           year=2009, topics=('functional-equation', 'inequality'), level=5, kind='answer', answer='f(n) = n', origin='contest', verified='recalled',
           spec=spec(('fn',), [lambda f: all(f(n) > 0 for n in range(1, 9))], (_tri2009, lambda f: all(f(n) == n for n in range(1, 9))), hatoms=False,
                     atoms=[(C(0), lambda f, a, b: a), (C(0), lambda f, a, b: b), (lambda f, a, b: a, lambda f, a, b: f(b) + f(b + f(a) - 1)), (lambda f, a, b: f(b), lambda f, a, b: a + f(b + f(a) - 1)),
                            (lambda f, a, b: f(b + f(a) - 1), lambda f, a, b: a + f(b)), (C(0), lambda f, n: n), (lambda f, n: f(n), lambda f, n: n)]),
           notes='f is a function on the integers whose values at the positive integers are positive; "non-degenerate triangle" means the three strict triangle inequalities. The answer is the identity (the published answer; the box only sees f(n) = n for n <= 8)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        lit = v.lit
        def tri(f): return v.F([v.Z, v.Z], lambda a, b_: v.pos(a) >> (v.pos(b_) >> AND(v.lt(a, f(b_) + f(b_ + f(a) - 1)), v.lt(f(b_), a + f(b_ + f(a) - 1)), v.lt(f(b_ + f(a) - 1), a + f(b_)))))
        return K.statement(w, [fZ], [lambda f: v.forall(lambda n: v.pos(n) >> v.pos(f(n)))], lambda f: v.Iff(tri(f), v.forall(lambda n: v.pos(n) >> Eq(f(n), n))))
