"""bank_imo4.py -- olympiad problems of the IMO Shortlist, of the Asia-Pacific olympiad and of the classical repertoire, integer vocabulary; STATEMENT-ONLY (see bank_imo.py for what that means, how the statements are checked
and what the provenance tags say).  The third batch of the statement tier: the shortlist problems whose texts could be compared with the published ones, and the classical exercises that every olympiad training meets.

`origin='contest'`: the problem of the named contest as the authors recall it, paraphrased (the statement agrees with the published text, which was compared with the official shortlist or the contest page); `adapted`:
changed to fit the vocabulary (the change is in `notes`); `folklore`: a classical fact or exercise with no single contest behind it.  `verified='answer-checked'`: the claim agrees with an exhaustive search on a range far
beyond the box (imo_truth.py).
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


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ IMO 2016 P4
def _P(x): return x * x + x + 1


def _frag_py(a, b):
    """the set {P(a+1), .., P(a+b)} is fragrant: at least two elements, and each has a common prime factor with another"""
    return all(any(j != i and math.gcd(_P(a + i), _P(a + j)) > 1 for j in range(1, b + 1)) for i in range(1, b + 1)) and b >= 2


def _frag(v, w, a, b):
    """the Prop `{P(a+1), .., P(a+b)} is fragrant` (b a term): at least two elements (2 <= b), and every P(a+i), 1 <= i <= b, has a prime factor in common with some other P(a+j), 1 <= j <= b, j != i (the two are not coprime)"""
    return AND(v.le(v.lit(2), b), v.forall(lambda i: v.le(v.one, i) >> (v.le(i, b) >> v.exists(lambda j: AND(v.le(v.one, j), v.le(j, b), v.Not(v.Eq(j, i)), v.Not(w['Z.coprime'](_P(a + i), _P(a + j))))))))


def _fatoms(a_f, b_f, k):
    """the atoms of `_frag` for a = a_f(*vars), b = b_f(*vars), where `vars` are the first k variables (the quantifier variables above the formula); the atoms are under i, and then j"""
    return [(C(2), lambda *t: b_f(*t[:k])), (C(1), lambda *t: t[k]), (lambda *t: t[k], lambda *t: b_f(*t[:k])),
            (C(1), lambda *t: t[k + 1]), (lambda *t: t[k + 1], lambda *t: b_f(*t[:k])), (lambda *t: t[k + 1], lambda *t: t[k]),
            (lambda *t: _P(a_f(*t[:k]) + t[k]), lambda *t: _P(a_f(*t[:k]) + t[k + 1]))]


_DOM_P4 = ([-2, 0, 1, 2, 3, 10, 194, 195, 196, 197, 200], [0, 1, 2, 3, 4, 5, 6, 7])
_TXT_P4 = ('Let P(n) = n^2 + n + 1. A set of positive integers is called fragrant if it contains at least two elements and each of its elements has a prime factor in common with at least one of the other elements. ')


@statement('imo2016_p4a', text=_TXT_P4 + 'Show that the set {P(196), P(197), P(198), P(199), P(200), P(201)} is fragrant. (This is the half of IMO 2016 P4 that shows that b = 6 is attained: the problem asks for the least positive integer b for which some non-negative integer a '
                                        'makes the set {P(a + 1), ..., P(a + b)} fragrant, and the answer is 6.)',
           source='IMO 2016 P4 (a)', year=2016, topics=('number-theory', 'gcd', 'construction'), level=4, kind='prove', origin='contest', verified='answer-checked',
           spec=spec(('Z',), [EQ(X(0), 195)], lambda a: _frag_py(a, 6), atoms=_fatoms(lambda a: a, lambda a: 6, 1), domains=([-2, 0, 1, 2, 3, 10, 194, 195, 196, 197, 200],), windows={'z': 8, 'ex': 8}),
           notes='the witness a = 195 (the set {P(196), .., P(201)}) is a variable with the hypothesis a = 195, so that the box sees the fragrant condition at other values of a as well; a = 196 is the next witness. Fragrant: each of the six values shares a prime factor '
                 'with another one: P(196) = 3 * 61 * 211 and P(199) = 3 * 13267; P(197) = 19 * 2053 and P(201) = 19 * 2137; P(198) = 7 * 13 * 433 and P(200) = 7 * 5743')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z], [lambda a: Eq(a, lit(195))], lambda a: _frag(v, w, a, lit(6)))


@statement('imo2016_p4b', text=_TXT_P4 + 'Show that for every non-negative integer a and every b in {1, 2, 3, 4, 5} the set {P(a + 1), ..., P(a + b)} is not fragrant. (This is the half of IMO 2016 P4 that shows that no b < 6 works: the problem asks for the least '
                                        'positive integer b for which some non-negative integer a makes the set {P(a + 1), ..., P(a + b)} fragrant, and the answer is 6.)',
           source='IMO 2016 P4 (b)', year=2016, topics=('number-theory', 'gcd', 'case-analysis'), level=5, kind='prove', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LE(0, X(0)), LE(1, X(1)), LE(X(1), 5)], lambda a, b: not _frag_py(a, b), atoms=_fatoms(lambda a, b: a, lambda a, b: b, 2), domains=_DOM_P4, windows={'z': 8, 'ex': 8}),
           notes='the fragrant condition is written with bounded quantifiers over the positions i, j in 1..b; the clause "at least two elements" is the conjunct 2 <= b. '
                 'gcd(P(m), P(m + 1)) = 1, and gcd(P(m), P(m + d)) divides 7, 3, 19, 7 for d = 2, 3, 4, 5 (the case analysis of the official solution); imo_truth.py checks that no a < 10^6 works for b <= 5')
def _(w, b):
    v = K.Vocab(w)
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z, v.Z], [lambda a, b_: v.nonneg(a), lambda a, b_: v.le(v.one, b_), lambda a, b_: v.le(b_, lit(5))], lambda a, b_: v.Not(_frag(v, w, a, b_)))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the IMO Shortlist
@statement('isl2000_n4', text='Find all triples (a, m, n) of positive integers such that a^m + 1 divides (a + 1)^n.', source='ISL 2000 N4', year=2000, topics=('number-theory', 'divisibility', 'case-analysis'), level=5, kind='answer',
           answer='a = 1 (any m, n); m = 1 (any a, n); and (a, m) = (2, 3) with n >= 2', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'N', 'N'), [LT(0, X(0)), LT(0, X(1)), LT(0, X(2))],
                     (lambda a, m, n: dvd_(a ** m + 1, (a + 1) ** n), lambda a, m, n: a == 1 or m == 1 or (a == 2 and m == 3 and n >= 2)),
                     atoms=[(lambda a, m, n: a ** m + 1, lambda a, m, n: (a + 1) ** n), (X(0), C(1)), (X(1), C(1)), (X(0), C(2)), (X(1), C(3)), (C(2), X(2))], domains=(range(0, 7), range(0, 6), range(0, 6))),
           notes='"find all" as an equivalence with the answer: a = 1 (then a^m + 1 = 2 divides 2^n), m = 1 (a + 1 divides (a + 1)^n), or a = 2, m = 3 and n >= 2 (9 divides 3^n); there are no other triples')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit; nl = v.nlit
        return K.statement(w, [v.Z, v.N, v.N], [lambda a, m, n: v.pos(a), lambda a, m, n: w['N.lt'](nl(0), m), lambda a, m, n: w['N.lt'](nl(0), n)],
                           lambda a, m, n: v.Iff(v.dvd(a ** m + 1, (a + 1) ** n), OR(Eq(a, v.one), Eq(m, nl(1)), AND(Eq(a, lit(2)), Eq(m, nl(3)), w['N.le'](nl(2), n)))))


@statement('isl2007_n2', text='Let b and n be integers greater than 1. Suppose that for each integer k > 1 there exists an integer a_k such that b - a_k^n is divisible by k. Prove that b = A^n for some integer A.', source='ISL 2007 N2', year=2007,
           topics=('number-theory', 'modular', 'divisibility'), level=5, kind='prove', origin='adapted', verified='answer-checked',
           spec=spec(('Z', 'N'), [LE(2, X(0)), LE(2, X(1))],
                     (lambda b, n: all(any(dvd_(k, b - a ** n) for a in range(-20, 21)) for k in range(2, 37)), lambda b, n: any(b == A ** n for A in range(-20, 21))),
                     atoms=[(C(2), lambda b, n, k: k), (lambda b, n, k, a: k, lambda b, n, k, a: b - a ** n), (X(0), lambda b, n, A: A ** n)], windows={'z': 36, 'ex': 20},
                     domains=([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 18, 25, 27, 32, 36, 48, 49, 50, 64], [0, 1, 2, 3, 4])),
           notes='adapted: "if b is an n-th power residue modulo every k > 1 then b is an n-th power" is written as an equivalence (the converse is trivial: for b = A^n take a_k = A). The condition is a statement about all k > 1; on the box it is tested for k <= 36, '
                 'which refutes it for every b that is not an n-th power in the box (the first k for which b is not an n-th power residue is at most 31 for all b, n tested; imo_truth.py)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z, v.N], [lambda b_, n: v.le(lit(2), b_), lambda b_, n: w['N.le'](v.nlit(2), n)],
                           lambda b_, n: v.Iff(v.forall(lambda k: v.le(lit(2), k) >> v.exists(lambda a: v.dvd(k, b_ - a ** n))), v.exists(lambda A: Eq(b_, A ** n))))


@statement('isl2012_n4b', text='An integer a is called friendly if the equation (m^2 + n)(n^2 + m) = a (m - n)^3 has a solution over the positive integers. Decide whether a = 2 is friendly.', source='ISL 2012 N4 (b)', year=2012,
           topics=('number-theory', 'diophantine', 'bounding'), level=5, kind='prove', origin='adapted', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1))], lambda m, n: (m * m + n) * (n * n + m) != 2 * (m - n) ** 3, atoms=[(lambda m, n: (m * m + n) * (n * n + m), lambda m, n: 2 * (m - n) ** 3)],
                     domains=(range(0, 14), range(0, 14))),
           notes='adapted: the answer of part (b) is "a = 2 is not friendly", written as the impossibility: no positive integers m, n satisfy (m^2 + n)(n^2 + m) = 2 (m - n)^3. Part (a) (at least 500 friendly integers up to 2012) is a statement about a count and is not in the bank. '
                 'imo_truth.py finds no solution with m, n <= 2000')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z, v.Z], [lambda m, n: v.pos(m), lambda m, n: v.pos(n)], lambda m, n: v.Not(Eq((m ** 2 + n) * (n ** 2 + m), 2 * (m - n) ** 3)))


@statement('isl2012_n6', text='Let x and y be positive integers. If x^(2^n) - 1 is divisible by 2^n y + 1 for every positive integer n, prove that x = 1.', source='ISL 2012 N6', year=2012, topics=('number-theory', 'divisibility', 'prime'), level=5,
           kind='prove', origin='adapted', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1))], (lambda x, y: all(dvd_(2 ** n * y + 1, x ** (2 ** n) - 1) for n in range(1, 13)), lambda x, y: x == 1),
                     atoms=[(C(0), lambda x, y, n: n), (lambda x, y, n: 2 ** n * y + 1, lambda x, y, n: x ** (2 ** n) - 1), (X(0), C(1))], domains=(range(-1, 10), range(-1, 10))),
           notes='adapted: the claim "the divisibility for every n >= 1 forces x = 1" is written as an equivalence (the converse is trivial: for x = 1 the number x^(2^n) - 1 is 0). The condition is a statement about all n >= 1; on the box it is tested for n <= 12, '
                 'which refutes it for every x >= 2, y >= 1 in the box (the first failing n is at most 5 there)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; pw = w['N.pow']
    with v.ints():
        lit = v.lit; nl = v.nlit
        return K.statement(w, [v.Z, v.Z], [lambda x, y: v.pos(x), lambda x, y: v.pos(y)],
                           lambda x, y: v.Iff(v.F([v.N], lambda n: w['N.lt'](nl(0), n) >> v.dvd(lit(2) ** n * y + 1, x ** pw(nl(2), n) - 1)), Eq(x, v.one)))


@statement('isl2013_n1', text='Find all functions f from the positive integers to the positive integers such that m^2 + f(n) divides m f(m) + n for all positive integers m and n.', source='ISL 2013 N1', year=2013,
           topics=('functional-equation', 'number-theory', 'divisibility'), level=5, kind='answer', answer='f(n) = n', origin='contest', verified='answer-checked',
           spec=spec(('fn',), [lambda f: all(f(n) > 0 for n in range(1, 9))],
                     (lambda f: all(dvd_(m * m + f(n), m * f(m) + n) for m in range(1, 9) for n in range(1, 9)), lambda f: all(f(n) == n for n in range(1, 9))), hatoms=False,
                     atoms=[(C(0), lambda f, m, n: m), (C(0), lambda f, m, n: n), (lambda f, m, n: m * m + f(n), lambda f, m, n: m * f(m) + n), (C(0), lambda f, n: n), (lambda f, n: f(n), lambda f, n: n)],
                     extra_fns=[lambda n: n if n != 3 else 4, lambda n: n if n <= 5 else n + 1, lambda n: n if n > 0 else 7, lambda n: n if n != 1 else 2, lambda n: 1]),
           notes='f is a function on the integers whose values at the positive integers are positive; the answer is the identity (the published answer; the box only sees f(n) = n for n <= 8)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        return K.statement(w, [fZ], [lambda f: v.forall(lambda n: v.pos(n) >> v.pos(f(n)))],
                           lambda f: v.Iff(v.F([v.Z, v.Z], lambda m, n: v.pos(m) >> (v.pos(n) >> v.dvd(m ** 2 + f(n), m * f(m) + n))), v.forall(lambda n: v.pos(n) >> Eq(f(n), n))))


def _isl14(k): return (k ** 3 + k ** 2 - 2 * k - 1, k ** 3 + 2 * k ** 2 - k - 1)


@statement('isl2014_n2', text='Determine all pairs (x, y) of positive integers such that the cube root of (7 x^2 - 13 x y + 7 y^2) equals |x - y| + 1.', source='ISL 2014 N2', year=2014, topics=('number-theory', 'diophantine', 'bounding'), level=5,
           kind='answer', answer='(x, y) = (1, 1); (k^3 + k^2 - 2k - 1, k^3 + 2k^2 - k - 1) and the same pairs reversed, for k >= 2', origin='contest', verified='answer-checked',
           spec=spec(('Z', 'Z'), [LT(0, X(0)), LT(0, X(1))],
                     (lambda x, y: (abs(x - y) + 1) ** 3 == 7 * x * x - 13 * x * y + 7 * y * y, lambda x, y: (x, y) == (1, 1) or any((x, y) == _isl14(k) or (y, x) == _isl14(k) for k in range(2, 6))),
                     atoms=[(X(1), X(0)), (lambda x, y: (x - y + 1) ** 3, lambda x, y: 7 * x * x - 13 * x * y + 7 * y * y), (X(0), X(1)), (lambda x, y: (y - x + 1) ** 3, lambda x, y: 7 * x * x - 13 * x * y + 7 * y * y),
                            (X(0), C(1)), (X(1), C(1)), (C(2), lambda x, y, k: k), (X(0), lambda x, y, k: _isl14(k)[0]), (X(1), lambda x, y, k: _isl14(k)[1]), (X(0), lambda x, y, k: _isl14(k)[1]), (X(1), lambda x, y, k: _isl14(k)[0])],
                     domains=([0, 1, 2, 3, 7, 8, 13, 14, 29, 30, 41, 42, 71, 72, 91, 92], [0, 1, 2, 3, 7, 8, 13, 14, 29, 30, 41, 42, 71, 72, 91, 92]), windows={'ex': 5}),
           notes='the cube root is removed by cubing (the radicand 7x^2 - 13xy + 7y^2 is positive and |x - y| + 1 is positive, so the two equations are equivalent), and the absolute value by the two cases x >= y and x < y. '
                 'Writing d = |x - y| and taking y = min(x, y) the equation becomes y^2 + d y + 7 d^2 = (d + 1)^3; imo_truth.py solves it exactly for d <= 2 * 10^6')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        E = lambda x, y: 7 * x ** 2 - 13 * x * y + 7 * y ** 2
        f1 = lambda k: k ** 3 + k ** 2 - 2 * k - 1
        f2 = lambda k: k ** 3 + 2 * k ** 2 - k - 1
        return K.statement(w, [v.Z, v.Z], [lambda x, y: v.pos(x), lambda x, y: v.pos(y)],
                           lambda x, y: v.Iff(OR(AND(v.le(y, x), Eq((x - y + 1) ** 3, E(x, y))), AND(v.lt(x, y), Eq((y - x + 1) ** 3, E(x, y)))),
                                              OR(AND(Eq(x, v.one), Eq(y, v.one)), v.exists(lambda k: AND(v.le(lit(2), k), OR(AND(Eq(x, f1(k)), Eq(y, f2(k))), AND(Eq(x, f2(k)), Eq(y, f1(k)))))))))


def _cubefree_py(X_, primes=(2, 3, 5, 7, 11)): return all(X_ % (p ** 3) != 0 for p in primes)


@statement('isl2021_n1', text='Determine all integers n >= 1 for which there exists a pair of positive integers (a, b) such that no cube of a prime divides a^2 + b + 3 and (a b + 3 b + 8) / (a^2 + b + 3) = n.', source='ISL 2021 N1', year=2021,
           topics=('number-theory', 'divisibility', 'prime'), level=5, kind='answer', answer='n = 2', origin='contest', verified='answer-checked',
           spec=spec(('Z',), [LT(0, X(0))],
                     (lambda n: any(_cubefree_py(a * a + b + 3) and (a * a + b + 3) * n == a * b + 3 * b + 8 for a in range(1, 13) for b in range(1, 13)), lambda n: n == 2),
                     atoms=[(C(0), lambda n, a, b: a), (C(0), lambda n, a, b: b), (lambda n, a, b, p: p,), (lambda n, a, b, p: p ** 3, lambda n, a, b, p: a * a + b + 3), (lambda n, a, b: (a * a + b + 3) * n, lambda n, a, b: a * b + 3 * b + 8), (X(0), C(2))],
                     domains=(range(0, 9),), windows={'ex': 12, 'z': 12, 'atom': 2},
                     skip_mutants={'negate Z.lt in (Z.lt 0 x2)': 'not the same statement but the same truth table in n: with a <= 0 the pairs (a, b) = (-1, b) give the quotient 2 for every b > 0, and no other n arises from a <= 0 in the window '
                                   '(a <= -3 gives n <= 0, and n = 0 needs a cube divisor: (a, b) = (-4, 8), (-5, 4), (-7, 2), (-11, 1) give 27, 32, 54 and 125); the box varies n only, so it cannot see the sign condition on a; the atom check covers the numbers of the guard'}),
           notes='"find all" as an equivalence with the answer n = 2; the quotient is an integer exactly when n (a^2 + b + 3) = a b + 3 b + 8, and the witnesses are (a, b) = (2, 2) (9 and 18) and (4, 6). The condition "no cube of a prime divides a^2 + b + 3" is the statement '
                 'that p^3 does not divide it for every prime p. imo_truth.py finds no other n with a < 400, b < 8000')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z], [lambda n: v.pos(n)],
                           lambda n: v.Iff(v.exists(lambda a: v.exists(lambda b_: AND(v.pos(a), v.pos(b_), v.forall(lambda p: w['Z.prime'](p) >> v.Not(v.dvd(p ** 3, a ** 2 + b_ + 3))), Eq((a ** 2 + b_ + 3) * n, a * b_ + 3 * b_ + 8)))),
                                           Eq(n, lit(2))))


@statement('apmo2019_p1', text='Let Z+ be the set of positive integers. Determine all functions f from Z+ to Z+ such that a^2 + f(a) f(b) is divisible by f(a) + b for all positive integers a and b.', source='APMO 2019 P1', year=2019,
           topics=('functional-equation', 'number-theory', 'divisibility'), level=5, kind='answer', answer='f(n) = n', origin='contest', verified='answer-checked',
           spec=spec(('fn',), [lambda f: all(f(n) > 0 for n in range(1, 9))],
                     (lambda f: all(dvd_(f(a) + b, a * a + f(a) * f(b)) for a in range(1, 9) for b in range(1, 9)), lambda f: all(f(n) == n for n in range(1, 9))), hatoms=False,
                     atoms=[(C(0), lambda f, a, b: a), (C(0), lambda f, a, b: b), (lambda f, a, b: f(a) + b, lambda f, a, b: a * a + f(a) * f(b)), (C(0), lambda f, n: n), (lambda f, n: f(n), lambda f, n: n)],
                     extra_fns=[lambda n: n if n != 3 else 4, lambda n: n if n <= 5 else n + 1, lambda n: n if n > 0 else 7, lambda n: n if n != 1 else 2, lambda n: 1]),
           notes='f is a function on the integers whose values at the positive integers are positive; the answer is the identity (the published answer; the box only sees f(n) = n for n <= 8)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; fZ = w.arrow(v.Z, v.Z)
    with v.ints():
        return K.statement(w, [fZ], [lambda f: v.forall(lambda n: v.pos(n) >> v.pos(f(n)))],
                           lambda f: v.Iff(v.F([v.Z, v.Z], lambda a, b_: v.pos(a) >> (v.pos(b_) >> v.dvd(f(a) + b_, a ** 2 + f(a) * f(b_)))), v.forall(lambda n: v.pos(n) >> Eq(f(n), n))))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ classical exercises
@statement('classic_ab_cd_sum', text='If a, b, c, d are positive integers with a b = c d, then a + b + c + d is not a prime.', source='classical (a classical olympiad exercise on composite numbers)', topics=('number-theory', 'prime', 'divisibility'), level=3,
           kind='prove', origin='folklore', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z', 'Z'), [LT(0, X(0)), LT(0, X(1)), LT(0, X(2)), LT(0, X(3)), EQ(lambda a, b, c, d: a * b, lambda a, b, c, d: c * d)], lambda a, b, c, d: not is_prime_(a + b + c + d),
                     atoms=[(lambda a, b, c, d: a + b + c + d,)], domains=(range(0, 10), range(0, 10), range(0, 10), range(0, 10)), points=2500),
           notes='a + b + c + d >= 4; the classical proof: with g = gcd(a, c), a = g a1, c = g c1, a1 b = c1 d forces c1 | b, b = c1 t, d = a1 t, and the sum is (a1 + c1)(g + t)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 4, [lambda a, b_, c, d: v.pos(a), lambda a, b_, c, d: v.pos(b_), lambda a, b_, c, d: v.pos(c), lambda a, b_, c, d: v.pos(d), lambda a, b_, c, d: Eq(a * b_, c * d)],
                           lambda a, b_, c, d: v.Not(w['Z.prime'](a + b_ + c + d)))


@statement('classic_odd_prime_pow_sum', text='Let p be an odd prime and a, b integers such that p divides a^p + b^p. Prove that p^2 divides a^p + b^p.', source='classical (a lemma of lifting the exponent)', topics=('number-theory', 'prime', 'divisibility', 'modular'),
           level=4, kind='prove', origin='folklore', verified='answer-checked',
           spec=spec(('N', 'Z', 'Z'), [PRIME(X(0)), lambda p, a, b: p != 2, DVD(X(0), lambda p, a, b: a ** p + b ** p)], DVD(lambda p, a, b: p * p, lambda p, a, b: a ** p + b ** p),
                     hatoms=[(X(0),), (X(0), C(2)), (X(0), lambda p, a, b: a ** p + b ** p)], domains=(range(0, 14), range(-5, 6), range(-5, 6)), points=2500),
           notes='a^p + b^p is congruent to a + b modulo p (Fermat), so p divides a + b, and then a^p + b^p = (a + b)(a^(p-1) - a^(p-2) b + .. + b^(p-1)) with the second factor congruent to p a^(p-1) modulo p. The claim is false for p = 2 (a = b = 1)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        return K.statement(w, [v.N, v.Z, v.Z], [lambda p, a, b_: w['Z.prime'](ofN(p)), lambda p, a, b_: v.Not(Eq(p, v.nlit(2))), lambda p, a, b_: v.dvd(ofN(p), a ** p + b_ ** p)],
                           lambda p, a, b_: v.dvd(ofN(p) ** 2, a ** p + b_ ** p))


@statement('classic_descent_2xyz', text='Find all integer solutions of the equation x^2 + y^2 + z^2 = 2 x y z.', source='classical (infinite descent)', topics=('number-theory', 'diophantine', 'descent', 'parity'), level=4, kind='answer', answer='(x, y, z) = (0, 0, 0)',
           origin='folklore', verified='answer-checked',
           spec=spec(('Z', 'Z', 'Z'), [EQ(lambda x, y, z: x * x + y * y + z * z, lambda x, y, z: 2 * x * y * z)], [EQ(X(0), 0), EQ(X(1), 0), EQ(X(2), 0)], domains=(range(-4, 5), range(-4, 5), range(-4, 5))),
           notes='the sum of three squares on the left is even, so x, y, z are all even or exactly one is; modulo 4 the second case is impossible and the first gives (x/2, y/2, z/2) as a solution of x^2 + y^2 + z^2 = 4 x y z, which descends the same way: every solution is divisible by every power of 2')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        return K.statement(w, [v.Z] * 3, [lambda x, y, z: Eq(x ** 2 + y ** 2 + z ** 2, 2 * x * y * z)], lambda x, y, z: AND(Eq(x, v.zero), Eq(y, v.zero), Eq(z, v.zero)))


@statement('classic_two_coprime', text='Every integer n >= 7 is the sum of two coprime integers, each greater than 1.', source='classical (decomposition into coprime summands)', topics=('number-theory', 'gcd', 'construction', 'parity'), level=3, kind='prove',
           origin='folklore', verified='answer-checked',
           spec=spec(('Z',), [LE(7, X(0))], lambda n: any(a > 1 and b > 1 and math.gcd(a, b) == 1 and a + b == n for a in range(-16, 17) for b in range(-16, 17)),
                     atoms=[(C(1), lambda n, a, b: a), (C(1), lambda n, a, b: b), (lambda n, a, b: a, lambda n, a, b: b), (lambda n, a, b: a + b, X(0))], body=(('Z', 'Z'), lambda n, a, b: a > 1 and b > 1 and math.gcd(a, b) == 1 and a + b == n),
                     domains=(range(5, 17),), windows={'ex': 16}),
           notes='for odd n take 2 and n - 2; for n = 4k take 2k - 1 and 2k + 1; for n = 4k + 2 with k >= 2 take 2k - 1 and 2k + 3 (both odd, and their gcd divides their difference 4). The values 6 = 1 + 5 = 3 + 3 = 2 + 4 and 4 have no such decomposition, so 7 is where the claim starts '
                 '(n = 5 = 2 + 3 works)')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq
    with v.ints():
        lit = v.lit
        return K.statement(w, [v.Z], [lambda n: v.le(lit(7), n)],
                           lambda n: v.exists(lambda a: v.exists(lambda b_: AND(v.lt(v.one, a), v.lt(v.one, b_), w['Z.coprime'](a, b_), Eq(a + b_, n)))))


@statement('classic_n4_4n', text='Prove that n^4 + 4^n is not a prime for any positive integer n > 1.', source='classical (the identity of Sophie Germain)', topics=('number-theory', 'prime', 'identity', 'parity'), level=4, kind='prove', origin='folklore',
           verified='answer-checked',
           spec=spec(('N',), [LT(0, X(0)), PRIME(lambda n: n ** 4 + 4 ** n)], EQ(X(0), 1), domains=(range(0, 13),)),
           notes='formalised as: if n >= 1 and n^4 + 4^n is a prime then n = 1 (for n = 1 it is 5). For even n the number is even and larger than 2; for odd n = 2k + 1 it is n^4 + 4 (2^k)^4 = (n^2 + 2 n 2^k + 2 (2^k)^2)(n^2 - 2 n 2^k + 2 (2^k)^2), and the second factor is (n - 2^k)^2 + 2^(2k) > 1')
def _(w, b):
    v = K.Vocab(w); Eq = v.Eq; ofN = w['Z.of_N']
    with v.ints():
        return K.statement(w, [v.N], [lambda n: w['N.lt'](v.nlit(0), n), lambda n: w['Z.prime'](ofN(n) ** 4 + v.lit(4) ** n)], lambda n: Eq(n, v.nlit(1)))
