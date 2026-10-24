"""bank_kit.py -- proof patterns the problems of the bank share: a fresh variable for an expression, the cases of a positive factorisation, ...

Each function takes the Proof `p` and the NAMES of hypotheses; it works inside a `with w.nota.using(Z)` block (the numerals it writes are integers).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import math
from lnl_base import thm
from nt_lib import _syms


class Vocab:
    """the symbols a statement is written with: `with v.ints():` makes + - * ** and the numerals integer arithmetic; `v.lit(n)` an integer numeral; `v.pos(x)` is 0 < x, `v.nonneg(x)` 0 <= x; `v.exists(f)` an integer witness"""
    def __init__(self, w):
        s = _syms(w); self.w = w; self.Z, self.N, self.Eq, self.Ex, self.F = s.Z, s.N, s.Eq, s.Ex, s.F
        self.zero, self.one, self.lt, self.le, self.dvd = s.zzero, s.zone, s.zlt, s.zle, s.D
        self.And, self.Or, self.Not, self.Iff, self.False_ = w.And, w.Or, w.Not, w.Iff, w['False']
    def ints(self): return self.w.nota.using(self.Z)
    def lit(self, n): return self.zero if n == 0 else self.w.nota.lit(n)
    def nlit(self, n):
        """a numeral of N (the exponents of powers, the lengths of sums)"""
        with self.w.nota.using(self.N): return self.w.nota.lit(n)
    def pos(self, x): return self.lt(self.zero, x)
    def nonneg(self, x): return self.le(self.zero, x)
    def exists(self, f): return self.Ex(self.Z, self.w.lam(self.Z, f))
    def forall(self, f): return self.F([self.Z], f)


def theorem(w, name, kinds, hyps, concl):
    """the decorator `thm` for a statement  forall (v1 : k1) .. (vn : kn), H1 -> .. -> Hm -> C : `kinds` the types of the variables, `hyps` and `concl` functions of the VARIABLES (not of the earlier hypotheses)"""
    n = len(kinds)
    return thm(w, name, list(kinds) + [(lambda *a, h=h: h(*a[:n])) for h in hyps], lambda *a: concl(*a[:n]))


def statement(w, kinds, hyps, concl):
    """the proposition  forall (v1 : k1) .. (vn : kn), H1 -> .. -> Hm -> C  of a STATEMENT-ONLY problem (`Bank.statement`): the same arguments as `theorem`, without a proof"""
    n = len(kinds)
    return w.forall_(list(kinds) + [(lambda *a, h=h: h(*a[:n])) for h in hyps], lambda *a: concl(*a[:n]))


def rw_ring(p, h):
    """rewrite the goal with the equation `h` and close it by `ring` (when the rewriting leaves an equation of numerals; unless the rewriting has closed it by computation already)"""
    n0 = p.n_goals; p.rw(h)
    if p.n_goals >= n0: p.ring()


def no_multiple(p, term, m, c):
    """close the goal when the (linear) facts of the context give  m * term = c  and m does not divide c: term <= c // m would make m * term < c, and c // m < term would make m * term > c"""
    lit = p.w.nota.lit
    q = c // m
    assert c % m != 0
    if p.goal().target.n != p.w['False'].n: p.exfalso()                     # (exfalso on the goal False would be a step that changes nothing)
    p.obtain(['_c', '_c'], ('Z.le_or_lt', term, lit(q) if q != 0 else p.w['Z.zero'])); p.linarith(); p.linarith()


def let(p, name, names, f):
    """introduce an integer variable `name` with the hypothesis `name_def : name = f(*terms of the hypotheses `names`)`"""
    w = p.w; Z = w['Z']
    ex = p.T(names, lambda *ts: w['Ex'](Z, w.lam(Z, lambda t: w.Eq(t, f(*ts)))))
    p.have('_let', ex, lambda q: (q.exists(q.T(names, f)), q.refl()))
    p.cases('_let', names=[name, name + '_def'])


def divisor_cases(p, x, y, e, n, on_pair, xn='dx', yn='dy'):
    """`e : x * y = n` (n a positive Python int) with `0 < x` and `0 < y` among the hypotheses (x, y are variables): one case for every positive divisor d of n, with `xn : x = d` and `yn : y = n / d`
    in the context, which `on_pair(p, d, n // d)` closes; every other value of x is refuted."""
    w = p.w; lit = w.nota.lit; Eq = w.Eq
    s = math.isqrt(n)
    p.obtain(['_c', '_c'], ('Z.le_or_lt', x, lit(s)))
    p.interval_cases(x, 1, s, name=xn)                                       # x <= s: the cases of x
    for d in range(1, s + 1):
        p.rw_at(e, xn)
        if n % d == 0:
            p.have(yn, p.T(y, lambda y_, q=n // d: Eq(y_, lit(q))), lambda q_: q_.linarith())
            on_pair(p, d, n // d)
        else:
            p.exfalso(); p.obtain(['_c', '_c'], ('Z.le_or_lt', y, lit(n // d)))
            p.linarith(); p.linarith()
    p.have('_yb', p.T(y, lambda y_: w['Z.le'](y_, lit(s))), lambda q_: q_.nlinarith())      # s < x: then y <= s
    p.interval_cases(y, 1, s, name=yn)
    for d in range(1, s + 1):
        p.rw_at(e, yn)
        if n % d == 0 and n // d > s:
            p.have(xn, p.T(x, lambda x_, q=n // d: Eq(x_, lit(q))), lambda q_: q_.linarith())
            on_pair(p, n // d, d)
        elif n % d == 0:
            p.exfalso(); p.linarith()                                         # the pair (n / d, d) has n / d <= s: already done, and contradicts s < x
        else:
            p.exfalso(); p.obtain(['_c', '_c'], ('Z.le_or_lt', x, lit(n // d)))
            p.linarith(); p.linarith()
