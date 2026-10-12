"""linarith.py -- linear arithmetic over the integers, by CERTIFICATE.

`linarith` closes a goal that follows from the hypotheses by adding them up. NOTHING HERE IS TRUSTED: the search (below) only has to FIND the multipliers; what it hands to the kernel is an
ordinary proof term, so a wrong search can only make the tactic fail, never make a false theorem appear.

    hypotheses  -->  facts  0 <= p_i   (a hypothesis  x <= y  is  0 <= y - x,  x < y  is  0 <= y - (x + 1),  x = y  is two facts, ~(x <= y) and ~(x < y) are read as  y < x  and  y <= x)
    search      -->  integers  m_i >= 0  and  kappa >= 1  with   sum_i m_i * p_i  ==  -kappa   as POLYNOMIALS  (every non-constant monomial cancels; a monomial that is not linear is an atom)
    proof       -->  0 <= sum_i m_i * p_i        by Z.mul_nonneg (Z.nonneg_of_N m_i) and Z.add_nonneg
                     sum_i m_i * p_i = -kappa    by `ring`   (the kernel checks the whole normalisation)
                     0 <= -kappa  is absurd      by Z.not_nonneg_neg

A goal  x <= y  is proved by contradiction from  y < x  (Z.le_of_not_lt), x < y from  y <= x  (Z.lt_of_not_le), x = y by both inequalities (Z.le_antisymm), False directly, anything else after
`false_elim`. The whole proof is ONE `exact`, so a search that fails leaves the proof state as it was.

`nlinarith` adds, before the search, the squares of the sub-terms  t * t  and  t ^ 2  that occur in the hypotheses (0 <= t * t, Z.sq_nonneg) and the products of every pair of facts
(0 <= p_i * p_j, Z.mul_nonneg), then searches as above: a certificate of degree two.

The search is a linear program (feasibility of  sum_i m_i P_i = -1,  m >= 0, with a row per non-constant monomial): an integer-preserving simplex with Bland's rule in exact arithmetic, preceded for
large problems by SciPy's HiGHS to guess the support, which the exact simplex then re-solves -- so floating point never decides anything.
"""
import os, sys, itertools
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from author import AuthorError, Term
import ring as R

MAX_MONOS = 20000            # a polynomial with more monomials than this is refused (the expansion of a huge power is not a thing to attempt)
EXACT_LIMIT = 40000          # rows * columns up to which the exact simplex is used without a first guess from SciPy
MAX_PRODUCTS = 600           # nlinarith: the most products of two facts it forms
STATS = {'float_guess': 0}  # how often the support guessed by SciPy was the one the exact simplex went on to solve (tests read it: a guess that is never used would be invisible otherwise)


# ---------------------------------------------------------------------------------------------------------------------------------------------- polynomials: {monomial: coefficient}
# a monomial is a tuple of (atom index, exponent) in increasing atom index; the empty tuple is the constant monomial
def mono_mul(a, b):
    if not a: return b
    if not b: return a
    out = []; i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x[0] == y[0]: out.append((x[0], x[1] + y[1])); i += 1; j += 1
        elif x[0] < y[0]: out.append(x); i += 1
        else: out.append(y); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return tuple(out)


def p_add(p, q, sign=1):
    r = dict(p)
    for m, c in q.items():
        v = r.get(m, 0) + sign * c
        if v: r[m] = v
        else: r.pop(m, None)
    return r


def p_scale(p, c): return {m: c * v for m, v in p.items()} if c else {}


def p_mul(p, q):
    if not p or not q: return {}
    r = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = mono_mul(m1, m2); v = r.get(m, 0) + c1 * c2
            if v: r[m] = v
            else: r.pop(m, None)
    if len(r) > MAX_MONOS: raise AuthorError('linarith: a polynomial with more than %d monomials' % MAX_MONOS)
    return r


def p_pow(p, e):
    if e == 0: return {(): 1}
    if len(p) > 1 and e > 64: raise AuthorError('linarith: the power %d of a sum is not expanded' % e)
    r = {(): 1}; b = p
    while e:
        if e & 1: r = p_mul(r, b)
        e >>= 1
        if e: b = p_mul(b, b)
    return r


class Reader:
    """a term of Z read as a polynomial with integer coefficients, WITHOUT building any proof; the numerals, the operations, the exponents and the atoms are those `ring` reads (it is the same `ring.Ring`
    that numbers the atoms), so what the search finds is what `ring` will check"""
    def __init__(self, rg):
        self.rg = rg; self.R = rg.R; self.w = rg.w; self.memo = {}

    def poly(self, t):
        hit = self.memo.get(t)
        if hit is not None: return hit
        R_, rg, w = self.R, self.rg, self.w
        k = R_.numeral(t)
        if k is not None: res = {(): k} if k else {}
        else:
            head, args = w.spine(t); hv = w.k.view(head.n)
            kind = R_.ops.get(hv[1]) if hv[0] == 'const' else None
            if kind == 'add' and len(args) == 2: res = p_add(self.poly(args[0]), self.poly(args[1]))
            elif kind == 'mul' and len(args) == 2: res = p_mul(self.poly(args[0]), self.poly(args[1]))
            elif kind == 'pow' and len(args) == 2 and (e := rg.const_exp(args[1])[0]) is not None: res = p_pow(self.poly(args[0]), e)
            elif kind == 'neg' and len(args) == 1: res = p_scale(self.poly(args[0]), -1)
            elif kind == 'sub' and len(args) == 2: res = p_add(self.poly(args[0]), self.poly(args[1]), -1)
            elif kind == 'sqr' and len(args) == 1: res = p_mul(self.poly(args[0]), self.poly(args[0]))         # `Z.sqr t` is what the display normal form makes of `t ^ 2`
            else: res = {((rg.atom(t), 1),): 1}
        self.memo[t] = res
        return res


# ---------------------------------------------------------------------------------------------------------------------------------------------- the search: a linear program
def prune(cols, nrows):
    """the columns that can have a positive multiplier at all: a column with a monomial that no other live column cancels is dead, and so on until nothing changes. cols[i] is {row: coefficient}; the last row is
    the constant row, which is never a reason to prune"""
    alive = set(range(len(cols))); crow = nrows - 1
    while True:
        pos = {}; neg = {}
        for i in alive:
            for r, c in cols[i].items():
                if r == crow: continue
                (pos if c > 0 else neg).setdefault(r, 0); (pos if c > 0 else neg)[r] += 1
        dead = [i for i in alive if any((c > 0 and r not in neg) or (c < 0 and r not in pos) for r, c in cols[i].items() if r != crow)]
        if not dead: return sorted(alive)
        alive.difference_update(dead)


def simplex(cols, nrows, crow):
    """integers m_i >= 0 and D >= 1 with  sum_i m_i * cols[i] = D * e_crow,  or None when there is no solution. Phase one of the simplex method with Bland's rule, in integer-preserving arithmetic:
    the tableau holds the integers  det(B) * (B^-1 [A | I | b]), so every entry is an exact integer and no fraction is ever formed"""
    m = len(cols); width = m + nrows + 1; rhs = width - 1
    T = [[0] * width for _ in range(nrows)]
    for i, col in enumerate(cols):
        for r, c in col.items(): T[r][i] = c
    for r in range(nrows): T[r][m + r] = 1
    T[crow][rhs] = 1
    z = [0] * width                                   # reduced costs of the sum of the artificial variables; the last entry is minus its value
    for r in range(nrows):
        for j in range(m): z[j] -= T[r][j]
    z[rhs] = -1
    basis = [m + r for r in range(nrows)]; D = 1
    while True:
        c = next((j for j in range(width - 1) if z[j] < 0), -1)            # Bland: the smallest index with a negative reduced cost
        if c < 0: break
        best = -1
        for r in range(nrows):
            a = T[r][c]
            if a <= 0: continue
            if best < 0: best = r; continue
            lhs, rgt = T[r][rhs] * T[best][c], T[best][rhs] * a             # compare the ratios rhs/a without dividing
            if lhs < rgt or (lhs == rgt and basis[r] < basis[best]): best = r
        if best < 0: return None                                           # unbounded: cannot happen for a sum of non-negative artificial variables
        pr = T[best]; p = pr[c]
        for i in range(nrows):
            if i == best: continue
            row = T[i]; f = row[c]
            if f == 0:
                if p != D: T[i] = [(x * p) // D for x in row]
            else: T[i] = [(x * p - f * y) // D for x, y in zip(row, pr)]
        f = z[c]
        z = [(x * p - f * y) // D for x, y in zip(z, pr)]
        basis[best] = c; D = p
    if z[rhs] != 0: return None
    sol = {basis[r]: T[r][rhs] for r in range(nrows) if basis[r] < m and T[r][rhs] > 0}
    return sol, D


def float_support(cols, nrows, crow):
    """the columns SciPy's HiGHS gives a positive multiplier (a guess: the exact simplex decides), or None"""
    try:
        import numpy as np
        from scipy.optimize import linprog
        A = np.zeros((nrows, len(cols)))
        for i, col in enumerate(cols):
            for r, c in col.items(): A[r, i] = float(c)
        b = np.zeros(nrows); b[crow] = 1.0
        res = linprog(np.zeros(len(cols)), A_eq=A, b_eq=b, bounds=(0, None), method='highs-ds')
        if res.status != 0: return None
        return [i for i in range(len(cols)) if res.x[i] > 1e-9]
    except Exception:
        return None


def combination(polys):
    """integers m_i >= 0 (a dict index -> m_i) and kappa >= 1 with  sum_i m_i * polys[i] == -kappa  as polynomials, or None"""
    rows = {}
    for P in polys:
        for m in P:
            if m and m not in rows: rows[m] = len(rows)
    crow = len(rows); nrows = crow + 1
    cols = []
    for P in polys:
        col = {rows[m]: c for m, c in P.items() if m}
        c0 = P.get((), 0)
        if c0: col[crow] = -c0
        cols.append(col)
    if not any(c0 < 0 for c0 in (P.get((), 0) for P in polys)): return None            # the constant must be driven negative
    alive = prune(cols, nrows)
    live_cols = [cols[i] for i in alive]
    if not any(crow in c for c in live_cols): return None
    res = None
    if len(live_cols) * nrows > EXACT_LIMIT:
        sup = float_support(live_cols, nrows, crow)
        if sup:
            r = simplex([live_cols[i] for i in sup], nrows, crow)
            if r is not None: res = ({alive[sup[i]]: v for i, v in r[0].items()}, r[1]); STATS['float_guess'] += 1
    if res is None:
        r = simplex(live_cols, nrows, crow)
        if r is not None: res = ({alive[i]: v for i, v in r[0].items()}, r[1])
    if res is None: return None
    sol, D = res
    g = D
    for v in sol.values(): g = gcd(g, v)
    return {i: v // g for i, v in sol.items()}, D // g


# ---------------------------------------------------------------------------------------------------------------------------------------------- facts
class Fact:
    """0 <= term, with the polynomial of `term` and a function  (the hypothesis for the negated goal) -> a proof of it"""
    __slots__ = ('poly', 'term', 'proof', 'label')
    def __init__(self, poly, term, proof, label): self.poly = poly; self.term = term; self.proof = proof; self.label = label


NEG = object()              # stands for the hypothesis of the negated goal in a fact that is only proved once that hypothesis exists


class Problem:
    """what the tactic needs to know about the integers: the terms and lemmas it builds proofs from"""
    def __init__(self, w):
        self.w = w; self.Z = w['Z']; self.inst = R.instance_for(w, self.Z); self.N = self.inst.N
        if not isinstance(self.inst, R.SemiringZ): raise AuthorError('linarith: the integers of this world have no ring instance')
        self.rg = R.Ring(w, self.inst); self.reader = Reader(self.rg)
        self.zadd, self.zmul, self.zsub, self.zof = w['Z.add'], w['Z.mul'], w['Z.sub'], w['Z.of_N']
        self.zero, self.one = w['Z.zero'], w['Z.one']; self.facts = []; self.seen = {}

    def classify(self, ty):
        """('le' | 'lt' | 'eq' | 'not_le' | 'not_lt', a, b) for a statement about integers, else None"""
        w = self.w; name = w.head_name(ty)
        if name is None: return None
        _, args = w.spine(ty)
        if name == 'Z.le' and len(args) == 2: return ('le', args[0], args[1])
        if name == 'Z.lt' and len(args) == 2: return ('lt', args[0], args[1])
        if name == 'Eq' and len(args) == 3 and args[0] == self.Z: return ('eq', args[1], args[2])
        if name == 'Not' and len(args) == 1:
            inner = self.classify(args[0])
            if inner is not None and inner[0] in ('le', 'lt'): return ('not_' + inner[0], inner[1], inner[2])
        return None

    def add_fact(self, term, proof, label):
        """a fact  0 <= term  (skipped when the polynomial is one we already have)"""
        poly = self.reader.poly(term)
        if not poly: return None                                    # 0 <= 0 says nothing
        key = tuple(sorted(poly.items()))
        if key in self.seen: return None
        f = Fact(poly, term, proof, label); self.seen[key] = len(self.facts); self.facts.append(f); return f

    def add_hypothesis(self, ty, hyp, label):
        """read the proposition `ty` (proved by the term `hyp`, or by a function of the negated goal's hypothesis) as facts"""
        w = self.w; sub, add, one = self.zsub, self.zadd, self.one
        c = self.classify(ty)
        if c is None: return 0
        kind, a, b = c
        const = (lambda neg: neg) if hyp is NEG else (lambda neg: hyp)
        if kind == 'le': self.add_fact(sub(b, a), lambda neg: w['Z.sub_nonneg_of_le'](a, b, const(neg)), label); return 1
        if kind == 'lt': self.add_fact(sub(b, add(a, one)), lambda neg: w['Z.sub_nonneg_of_le'](add(a, one), b, const(neg)), label); return 1
        if kind == 'not_le':                       # ~(a <= b): b < a
            self.add_fact(sub(a, add(b, one)), lambda neg: w['Z.sub_nonneg_of_le'](add(b, one), a, w['Z.lt_of_not_le'](a, b, const(neg))), label); return 1
        if kind == 'not_lt':                       # ~(a < b): b <= a
            self.add_fact(sub(a, b), lambda neg: w['Z.sub_nonneg_of_le'](b, a, w['Z.le_of_not_lt'](a, b, const(neg))), label); return 1
        # an equation: two inequalities
        self.add_fact(sub(b, a), lambda neg: w['Z.sub_nonneg_of_le'](a, b, w['Z.le_of_eq'](a, b, const(neg))), label + ' (<=)')
        self.add_fact(sub(a, b), lambda neg: w['Z.sub_nonneg_of_le'](b, a, w['Z.le_of_eq'](b, a, w['symm'](self.Z, a, b, const(neg)))), label + ' (>=)')
        return 2

    def add_casts(self):
        """every atom  Z.of_N t  is non-negative"""
        w = self.w; of_n = w.gid['Z.of_N']
        for t in list(self.rg.atoms):
            head, args = w.spine(t); hv = w.k.view(head.n)
            if hv[0] == 'const' and hv[1] == of_n and len(args) == 1:
                self.add_fact(t, (lambda tt: lambda neg: w['Z.nonneg_of_N'](tt))(args[0]), 'cast')

    def squares(self, terms):
        """the t of the sub-terms  t * t  and  t ^ 2  of the given terms"""
        w = self.w; out = []; seen = set(); done = set()
        stack = list(terms)
        while stack:
            t = stack.pop()
            if t in done: continue
            done.add(t)
            head, args = w.spine(t); hv = w.k.view(head.n)
            if hv[0] != 'const': continue
            kind = self.inst.ops.get(hv[1])
            if kind == 'mul' and len(args) == 2 and args[0] == args[1] and args[0] not in seen: seen.add(args[0]); out.append(args[0])
            if kind == 'pow' and len(args) == 2 and self.inst.enumeral(args[1]) == 2 and args[0] not in seen: seen.add(args[0]); out.append(args[0])
            if kind == 'sqr' and len(args) == 1 and args[0] not in seen: seen.add(args[0]); out.append(args[0])
            stack.extend(args)
        return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- the proof of False from a certificate
def build_false(pb, facts, mult, kappa, neg):
    """a proof of False from  sum_i mult[i] * facts[i].term == -kappa"""
    w, inst, rg = pb.w, pb.inst, pb.rg
    S = None; PS = None
    for i in sorted(mult):
        f = facts[i]; m = mult[i]; pf = f.proof(neg)
        if m == 1: t, pt = f.term, pf
        else:
            c = inst.lit(m); t = pb.zmul(c, f.term); pt = w['Z.mul_nonneg'](c, f.term, w['Z.nonneg_of_N'](inst.N.lit(m)), pf)
        if S is None: S, PS = t, pt
        else: PS = w['Z.add_nonneg'](S, t, PS, pt); S = pb.zadd(S, t)
    target = inst.lit(-kappa)
    eq, nfs = rg.prove_eq(S, target)
    if eq is None: raise AuthorError('linarith: internal error: the certificate does not add up to -%d (it normalises to %s)' % (kappa, rg.show_nf(nfs[0])))
    le = w['Z.le_of_le_eq'](pb.zero, S, target, PS, eq.proof)
    v = w.k.view(inst.N.lit(kappa).n)                                         # N.lit kappa = Npos q for kappa >= 1
    return w['Z.not_nonneg_neg'](w.T(v[2]), le)


def search(pb, nonlinear):
    """(multipliers, kappa) for the facts collected in `pb`, after adding what `nonlinear` asks for; None when no certificate exists"""
    w = pb.w
    pb.add_casts()
    facts = pb.facts
    if nonlinear:
        base = list(facts)
        for t in pb.squares([f.term for f in base]):
            sq = pb.zmul(t, t)
            pb.add_fact(sq, (lambda tt: lambda neg: w['Z.sq_nonneg'](tt))(t), 'square')
        base = list(pb.facts)
        n = 0
        for i in range(len(base)):
            for j in range(i, len(base)):
                if n >= MAX_PRODUCTS: break
                a, b = base[i], base[j]
                pb.add_fact(pb.zmul(a.term, b.term), (lambda a, b: lambda neg: w['Z.mul_nonneg'](a.term, b.term, a.proof(neg), b.proof(neg)))(a, b), 'product'); n += 1
        pb.add_casts()
    facts = pb.facts
    res = combination([f.poly for f in facts])
    return None if res is None else (res[0], res[1], facts)


def prove_false(pb, nonlinear):
    """a proof of False (a function of the negated goal's hypothesis `neg`, if there is one) -- or None when no certificate exists"""
    r = search(pb, nonlinear)
    if r is None: return None
    mult, kappa, facts = r
    return lambda neg: build_false(pb, facts, mult, kappa, neg)


# ---------------------------------------------------------------------------------------------------------------------------------------------- the tactic
def linarith(p, extra=(), nonlinear=False):
    """close the first goal by linear arithmetic over the integers from the hypotheses (and the proofs in `extra`): see the module documentation"""
    w = p.w; k = p.k; g = p.goal(); name = 'nlinarith' if nonlinear else 'linarith'
    with p.focus():
        pb0 = Problem(w); zlt, zle = w['Z.lt'], w['Z.le']
        tgt = pb0.classify(g.target)
        if tgt is not None and tgt[0] in ('le', 'lt', 'eq'): kind, x, y = tgt
        elif w.head_name(g.target) == 'False': kind, x, y = 'false', None, None
        else: kind, x, y = 'other', None, None
        extras = [p.t(e) for e in extra]

        def refute(gk, a, b):
            """a proof of the goal  a <= b  (gk 'le'),  a < b  (gk 'lt')  or False (gk 'false', 'other')"""
            pb = Problem(w)
            for lvl, (nm, ty) in enumerate(g.hyps): pb.add_hypothesis(ty, w.T(k.mk_fvar(lvl)), 'hypothesis ' + nm)
            for t in extras: pb.add_hypothesis(w.type_of(t), t, 'extra')
            neg_ty = None
            if gk == 'le': neg_ty = zlt(b, a)
            elif gk == 'lt': neg_ty = zle(b, a)
            if neg_ty is not None: pb.add_hypothesis(neg_ty, NEG, 'the negated goal')
            pf = prove_false(pb, nonlinear)
            if pf is None:
                raise AuthorError('%s failed: no combination of the %d facts read from the hypotheses%s refutes the negated goal' % (name, len(pb.facts), ' and their products' if nonlinear else ''))
            if neg_ty is None: return pf(None)
            body = w.lam(neg_ty, lambda h: pf(h))
            return w['Z.le_of_not_lt'](b, a, body) if gk == 'le' else w['Z.lt_of_not_le'](b, a, body)

        if kind == 'eq': proof = w['Z.le_antisymm'](x, y, refute('le', x, y), refute('le', y, x))
        elif kind in ('le', 'lt'): proof = refute(kind, x, y)
        elif kind == 'false': proof = refute('false', None, None)
        else: proof = w['false_elim'](g.target, refute('other', None, None))
    p.exact(proof)
    p.trace.append((name,))
    return p
