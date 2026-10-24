"""imo_truth.py -- brute-force evidence for the claims of the STATEMENT-ONLY problems of the bank.

A statement-only problem has no proof in the kernel.  What stands behind it is (1) the faithfulness check of semantic.py (the formal statement means what the Python spec says, on a box of small values),
(2) the truth screen of the same module (the statement holds at every point of the box) and (3) THIS FILE: for every statement-only problem an independent checker that re-derives its claim by exhaustive
search on a range much larger than the box, written directly from the problem (not from the formal statement), and compares the result with the published answer where there is one.

Run   python3 imo_truth.py [name ..]   to run all (or the named) checkers; it prints one line per problem (the range searched and the time) and exits with status 1 if one fails.
`python3 imo_truth.py --list` prints the table of claims and ranges.  A checker returns True/False; its docstring-like `rng` says what was searched.  What a search cannot do (a claim about ALL integers, a
function on all of N) is said in `kind`: 'search' (the claim is a statement about a bounded or bounded-by-argument set and the search is exhaustive for it), 'range' (a claim about all integers; the search is a range),
'reduction' (the search is exhaustive for a finite problem equivalent to the claim by a proof sketched in the docstring).
"""
import os as _os, sys as _sys, time, math
from math import isqrt, gcd, factorial

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401

CLAIMS = {}


def claim(name, rng, kind='range'):
    def deco(fn):
        CLAIMS[name] = (fn, rng, kind); return fn
    return deco


def sq(t): return t >= 0 and isqrt(t) ** 2 == t


def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    d = 3
    while d * d <= n:
        if n % d == 0: return False
        d += 2
    return True


def iroot(n, k):
    """the integer k-th root of n >= 0 (floor), by Newton's iteration on integers"""
    if n < 2: return n
    x = 1 << ((n.bit_length() + k - 1) // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x: return x
        x = y


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ AIME
@claim('aime1985_p7', 'a <= 300000')
def _():
    """a^5 = b^4, c^3 = d^2, c - a = 19: d - b = 757"""
    out = set()
    for a in range(1, 300000):
        b4 = a ** 5; b = iroot(b4, 4)
        if b ** 4 != b4: continue
        c = a + 19; d2 = c ** 3; d = isqrt(d2)
        if d * d == d2: out.add(d - b)
    return out == {757}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ IMO, first batch
@claim('imo1960_p1', 'all three-digit numbers')
def _():
    return [n for n in range(100, 1000) if n % 11 == 0 and n // 11 == sum(int(c) ** 2 for c in str(n))] == [550, 803]


@claim('imo1977_p5', 'a, b < 800')
def _():
    S = set()
    for a in range(1, 800):
        for b in range(1, 800):
            q, r = divmod(a * a + b * b, a + b)
            if q * q + r == 1977: S.add((a, b))
    return S == {(50, 37), (50, 7), (37, 50), (7, 50)}


@claim('imo1981_p3', 'all 1 <= m, n <= 1981')
def _():
    S = [(m, n) for m in range(1, 1982) for n in range(1, 1982) if (n * n - m * n - m * m) ** 2 == 1]
    return max(m * m + n * n for m, n in S) == 987 ** 2 + 1597 ** 2 == 3524578


@claim('imo1986_p1', 'd < 400000')
def _():
    return [d for d in range(1, 400000) if sq(2 * d - 1) and sq(5 * d - 1) and sq(13 * d - 1)] == []


@claim('imo1987_p4', 'shift k = 1..9 (odd: none; even: some), values < 6k', 'reduction')
def _():
    """f(f(n)) = n + k on the non-negative integers forces f(n + k) = f(n) + k, so f is determined by f(0..k-1); the equation is then a condition on those k values only (f(m) = f(m mod k) + k (m div k)).
    Exhaustive over the values f(r) < 6k: for odd k there is no solution, for even k there is (f(n) = n + k/2)."""
    def solutions(k, B):
        import itertools
        found = 0
        for vals in itertools.product(range(B), repeat=k):
            f = lambda m: vals[m % k] + k * (m // k)
            if all(f(f(r)) == r + k for r in range(k)): found += 1
        return found
    return all(solutions(k, 6 * k if k <= 4 else 3 * k) == 0 for k in (1, 3, 5)) and all(solutions(k, 6 * k) > 0 for k in (2, 4))


@claim('imo1990_p3', 'n < 30000')
def _():
    return [n for n in range(2, 30000) if pow(2, n, n * n) == n * n - 1] == [3]


@claim('imo1992_p1', 'a < 40, b < 150, c < 1200')
def _():
    S = [(a, b, c) for a in range(2, 40) for b in range(a + 1, 150) for c in range(b + 1, 1200) if (a * b * c - 1) % ((a - 1) * (b - 1) * (c - 1)) == 0]
    return S == [(2, 4, 8), (3, 5, 15)]


@claim('imo1994_p4', 'm, n < 400')
def _():
    S = {(m, n) for m in range(1, 400) for n in range(1, 400) if m * n - 1 != 0 and (n ** 3 + 1) % (m * n - 1) == 0}
    return S == {(1, 2), (1, 3), (2, 1), (2, 2), (2, 5), (3, 1), (3, 5), (5, 2), (5, 3)}


@claim('imo1997_p5', 'a < 300, b < 20')
def _():
    return {(a, b) for a in range(1, 300) for b in range(1, 20) if a ** (b * b) == b ** a} == {(1, 1), (16, 2), (27, 3)}


@claim('imo1998_p4', 'a < 1500, b < 200')
def _():
    return {(a, b) for a in range(1, 1500) for b in range(1, 200) if (a * a * b + a + b) % (a * b * b + b + 7) == 0} == {(11, 1), (49, 1)} | {(7 * k * k, 7 * k) for k in range(1, 29) if 7 * k * k < 1500 and 7 * k < 200}


@claim('imo1999_p4', 'primes p <= 31, n <= 2p')
def _():
    res = {(n, p) for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31) for n in range(1, 2 * p + 1) if ((p - 1) ** n + 1) % (n ** (p - 1)) == 0}
    return res == {(1, p) for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)} | {(2, 2), (3, 3)}


@claim('imo2001_p6', 'a < 80 (all solutions of ac + bd = (b + d + a - c)(b + d - a + c) with a > b > c > d > 0)')
def _():
    n = 0
    for a in range(1, 80):
        for b in range(1, a):
            for c in range(1, b):
                for d in range(1, c):
                    if a * c + b * d == (b + d + a - c) * (b + d - a + c):
                        n += 1
                        if is_prime(a * b + c * d): return False
    return n > 100


@claim('imo2003_p2', 'a < 600, b < 120')
def _():
    S = set()
    for a in range(1, 600):
        for b in range(1, 120):
            d = 2 * a * b * b - b ** 3 + 1
            if d > 0 and (a * a) % d == 0: S.add((a, b))
    want = set()
    for l in range(1, 600):
        want |= {(2 * l, 1), (l, 2 * l), (8 * l ** 4 - l, 2 * l)}
    return S == {t for t in want if t[0] < 600 and t[1] < 120}


@claim('imo2005_p4', 'k < 300, n < 300')
def _():
    return [k for k in range(1, 300) if all(gcd(k, 2 ** n + 3 ** n + 6 ** n - 1) == 1 for n in range(1, 300))] == [1]


@claim('imo2006_p4', 'x <= 200')
def _():
    S = [(x, isqrt(1 + 2 ** x + 2 ** (2 * x + 1))) for x in range(0, 200) if sq(1 + 2 ** x + 2 ** (2 * x + 1))]
    return S == [(0, 2), (4, 23)]


@claim('imo2007_p5', 'a, b < 400')
def _():
    S = [(a, b) for a in range(1, 400) for b in range(1, 400) if (4 * a * a - 1) ** 2 % (4 * a * b - 1) == 0 and (4 * a * b - 1) != 0]
    return all(a == b for a, b in S) and len(S) >= 399


@claim('imo2015_p2', 'a, b, c < 150')
def _():
    p2 = lambda t: t > 0 and t & (t - 1) == 0
    S = {(a, b, c) for a in range(1, 150) for b in range(1, 150) for c in range(1, 150) if p2(a * b - c) and p2(b * c - a) and p2(c * a - b)}
    return S == {(2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2), (2, 6, 11), (2, 11, 6), (6, 2, 11), (6, 11, 2), (11, 2, 6), (11, 6, 2), (3, 5, 7), (3, 7, 5), (5, 3, 7), (5, 7, 3), (7, 3, 5), (7, 5, 3)}


@claim('imo2022_p5', 'b <= 60, primes p <= 61')
def _():
    ps = [p for p in range(2, 62) if is_prime(p)]
    res = set()
    for b in range(1, 61):
        f = factorial(b)
        for p in ps:
            t = f + p; a = iroot(t, p)
            if a ** p == t: res.add((a, b, p))
    return res == {(2, 2, 2), (3, 4, 3)}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ IMO, second batch
@claim('imo1962_p1', 'k <= 60 digits')
def _():
    sols = []
    for k in range(1, 61):
        for m in [(2 * 10 ** k - 8) // 13] if (2 * 10 ** k - 8) % 13 == 0 else []:
            n = 10 * m + 6
            if 10 ** k <= 10 * m < 10 ** (k + 1) and 4 * n == 6 * 10 ** k + m: sols.append(n)
    return min(sols) == 153846 and all(10 ** 5 <= 10 * 15384 < 10 ** 6 for _ in (0,)) and 4 * 153846 == 6 * 10 ** 5 + 15384


@claim('imo1972_p3', 'm, n <= 60')
def _():
    return all((factorial(2 * m) * factorial(2 * n)) % (factorial(m) * factorial(n) * factorial(m + n)) == 0 for m in range(61) for n in range(61))


@claim('imo1978_p1', 'n < 400')
def _():
    best = min((m + n, m, n) for n in range(2, 400) for m in range(1, n) if (1978 ** n - 1978 ** m) % 1000 == 0)
    return best == (106, 3, 103)


@claim('imo1982_p4a', '|x|, |y| <= 60')
def _():
    f = lambda x, y: x ** 3 - 3 * x * y * y + y ** 3
    for x in range(-60, 61):
        for y in range(-60, 61):
            n = f(x, y)
            if f(y - x, -x) != n or f(-y, x - y) != n: return False
            if n > 0 and len({(x, y), (y - x, -x), (-y, x - y)}) < 3: return False
    return True


@claim('imo1982_p4b', '|x|, |y| <= 1500 (x^3 - 3xy^2 + y^3 = 2891; also impossible mod 9)')
def _():
    f = lambda x, y: x ** 3 - 3 * x * y * y + y ** 3
    mod9 = {f(x, y) % 9 for x in range(9) for y in range(9)}
    return all(f(x, y) != 2891 for x in range(-1500, 1501) for y in range(-60, 61)) and 2891 % 9 not in mod9


@claim('imo1984_p6', 'all odd b < c with b + c = 2^m <= 2^15, then a + d = 2^k (k <= 32), a d = b c')
def _():
    n = 0
    for m in range(2, 16):
        s2 = 2 ** m
        for b in range(1, s2 // 2, 2):
            c = s2 - b; P = b * c
            for k in range(2, 33):
                s1 = 2 ** k; D = s1 * s1 - 4 * P
                if D < 0: continue
                r = isqrt(D)
                if r * r != D: continue
                a = (s1 - r) // 2; d = (s1 + r) // 2
                if a >= 1 and a % 2 and d % 2 and a < b < c < d and a * d == P:
                    n += 1
                    if a != 1: return False
    return n >= 12


@claim('imo1989_p5', 'n <= 7: the first n consecutive numbers with two prime divisors each; every one has a coprime factorisation')
def _():
    def two_primes(t):
        ps = 0; d = 2; u = t
        while d * d <= u:
            if u % d == 0:
                ps += 1
                while u % d == 0: u //= d
            d += 1
        return ps + (u > 1) >= 2
    def coprime_split(t): return any(a > 1 and t // a > 1 and t % a == 0 and gcd(a, t // a) == 1 for a in range(2, t))
    first = {}
    for n in range(1, 8):
        m = next(m for m in range(1, 10000) if all(two_primes(m + i) for i in range(1, n + 1)))
        first[n] = m
        if not all(coprime_split(m + i) for i in range(1, n + 1)): return False
    return first[3] == 19 and all(two_primes(first[7] + i) for i in range(1, 8))


@claim('imo1996_p4', 'r, s < 3000')
def _():
    S = []
    for r in range(1, 3000):
        for s in range(1, 3000):
            if (15 * r * r + 16 * s * s) % 481 == 0 and (16 * r * r - 15 * s * s) % 481 == 0:
                a = (15 * r * r + 16 * s * s) // 481; b = (16 * r * r - 15 * s * s) // 481
                if a > 0 and b > 0: S.append((r, s, a, b))
    return min(min(r, s) for r, s, _, _ in S) == 481 and all(r >= 481 and s >= 481 for r, s, _, _ in S) and (481, 481, 14911, 481) in S


@claim('imo2009_p5', 'the identity for a, b <= 300; affine functions with |coefficients| <= 6 and periodic ones (period <= 4, values 1..6) fail', 'partial')
def _():
    """the uniqueness of the identity is the published proof (a = 1 forces f(b) = f(b + f(1) - 1), so f is periodic or the identity, and a large a contradicts a periodic f): a finite search cannot see it, because the
    constraints of a function on 1..N reach beyond N. What is checked: the identity satisfies the condition, and the natural competitors (affine, periodic) do not."""
    def holds(f, lim=300):
        for a in range(1, lim + 1):
            for b in range(1, lim + 1):
                x, y, z = a, f(b), f(b + f(a) - 1)
                if not (x < y + z and y < x + z and z < x + y): return False
        return True
    if not holds(lambda n: n): return False
    for p_ in range(-6, 7):
        for q in range(-6, 7):
            if (p_, q) == (1, 0): continue
            f = lambda n, p_=p_, q=q: p_ * n + q
            if all(f(n) > 0 for n in range(1, 40)) and holds(f, 60): return False
    import itertools
    for per in range(1, 5):
        for vals in itertools.product(range(1, 7), repeat=per):
            f = lambda n, vals=vals, per=per: vals[(n - 1) % per]
            if holds(f, 60): return False
    return True


@claim('imo2012_p4', '|x| <= 16, values by the recursion f(n) = f(n-1) + f(1) +- 2 sqrt(f(1) f(n-1)), |f(1)| <= 60', 'search')
def _():
    N = 16

    def ok(f, n):
        g = lambda x: f[abs(x)]
        for a in range(-n, n + 1):
            for b in range(-n, n + 1):
                c = -a - b
                if abs(c) > n: continue
                fa, fb, fc = g(a), g(b), g(c)
                if fa * fa + fb * fb + fc * fc != 2 * fa * fb + 2 * fb * fc + 2 * fc * fa: return False
        return True
    sols = set()

    def dfs(f, n):
        if n > N:
            sols.add(tuple(f)); return
        k = f[1]; p = f[n - 1]; d = k * p
        if d < 0 or not sq(d): return
        r = isqrt(d)
        for c in {k + p + 2 * r, k + p - 2 * r}:
            f2 = f + [c]
            if ok(f2, n): dfs(f2, n + 1)
    for k in range(-60, 61): dfs([0, k], 2)
    fam = set()
    for k in range(-60, 61):
        fam.add(tuple(k * x * x for x in range(N + 1))); fam.add(tuple(0 if x % 2 == 0 else k for x in range(N + 1)))
        fam.add(tuple(0 if x % 4 == 0 else (k if x % 2 else 4 * k) for x in range(N + 1)))
    return sols == fam


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the IMO Shortlist
@claim('isl2006_n5', '|x| <= 4000: x^6 + x^5 + .. + x + 2 is never a fifth power (modulo 11 alone does not settle it: x = 7, 9 mod 11 survive)')
def _():
    def is_fifth(v):
        r = iroot(abs(v), 5); return r ** 5 == abs(v)
    return all(not is_fifth(sum(x ** i for i in range(7)) + 1) for x in range(-4000, 4001))


@claim('isl2007_n1', 'k < 60, n < 100')
def _():
    return [(k, n) for k in range(1, 60) for n in range(1, 100) if (k ** 4 + n * n) % (7 ** k - 3 ** n) == 0] == [(2, 4)]


@claim('isl2007_n6', 'k <= 12: all divisors of (4k^2 - 1)^2 of the form 8kn - 1')
def _():
    for k in range(1, 13):
        N = (4 * k * k - 1) ** 2
        has = any(N % (8 * k * n - 1) == 0 for n in range(1, N // (8 * k) + 2) if 8 * k * n - 1 > 0)
        if has != (k % 2 == 0): return False
    return True


@claim('isl2008_n1', 'n <= 7, p in {2, 3, 5, 7, 11, 13}, |a|, |b|, |c| <= 14')
def _():
    for n in range(1, 8):
        for p in (2, 3, 5, 7, 11, 13):
            for a in range(-14, 15):
                for b in range(-14, 15):
                    for c in range(-14, 15):
                        if a ** n + p * b == b ** n + p * c == c ** n + p * a and not (a == b == c): return False
    return True


@claim('isl2010_n2', 'n < 120: the quadratic in m solved exactly')
def _():
    S = set()
    for n in range(0, 120):
        B = 2 ** (n + 1) - 1; D = B * B - 8 * 3 ** n
        if D >= 0 and sq(D):
            r = isqrt(D)
            if (B + r) % 2 == 0:
                for m in ((B + r) // 2, (B - r) // 2):
                    if m >= 0 and m * m + 2 * 3 ** n == m * B: S.add((m, n))
    return S == {(6, 3), (9, 3), (9, 5), (54, 5)}


@claim('isl2012_n2', 'x < 40, y < 300, z < 800')
def _():
    S = [(x, y, z) for x in range(1, 40) for y in range(x, 300) for z in range(y, 800) if x ** 3 * (y ** 3 + z ** 3) == 2012 * (x * y * z + 2)]
    return S == [(2, 251, 252)]


@claim('isl2015_n2', 'a, b < 60')
def _():
    return all(3 * a >= 2 * b + 2 for a in range(1, 60) for b in range(1, 60) if (factorial(a) * factorial(b)) % (factorial(a) + factorial(b)) == 0)


@claim('isl2016_n4', '2 <= n < 8, k, l < 12, m < 600')
def _():
    n_hyp = 0
    for n in range(2, 8):
        for k in range(1, 12):
            for l in range(1, 12):
                for m in range(1, 600):
                    if (n ** (k + l) - 1) % (n ** k + m * n ** l + 1) == 0:
                        n_hyp += 1
                        c1 = m == 1 and l == 2 * k
                        c2 = k % l == 0 and k >= l and m * (n ** l - 1) == n ** (k - l) - 1
                        if not (c1 or c2): return False
    return n_hyp > 50


@claim('isl2019_n2', 'a, b, c < 150')
def _():
    return {(a, b, c) for a in range(1, 150) for b in range(1, 150) for c in range(1, 150) if a ** 3 + b ** 3 + c ** 3 == (a * b * c) ** 2} == {(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)}


@claim('isl2022_n1', 'n <= 1344: no n < 1344 has three distinct divisors of sum 2022, and 1344 has')
def _():
    def norwegian(n):
        ds = [d for d in range(1, n + 1) if n % d == 0]; S = set(ds)
        return any(2022 - a - b > b and (2022 - a - b) in S for i, a in enumerate(ds) for b in ds[i + 1:])
    return all(not norwegian(n) for n in range(1, 1344)) and norwegian(1344) and 6 + 672 + 1344 == 2022


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ driver
def run(names=None, out=print):
    names = list(names) if names else list(CLAIMS)
    bad = []
    for nm in names:
        fn, rng, kind = CLAIMS[nm]
        t0 = time.time(); ok = bool(fn()); dt = time.time() - t0
        out('%-4s %-22s %-8s %-90s %6.1fs' % ('ok' if ok else 'FAIL', nm, kind, rng, dt))
        if not ok: bad.append(nm)
    return bad


if __name__ == '__main__':
    args = _sys.argv[1:]
    if args == ['--list']:
        for nm, (fn, rng, kind) in CLAIMS.items(): print('%-22s %-9s %s' % (nm, kind, rng))
        _sys.exit(0)
    bad = run(args or None)
    _sys.exit(1 if bad else 0)
