"""S31 gate: `linarith` and `nlinarith` over the integers (linarith.py).

Nothing in the tactic is trusted: it searches for a certificate and hands the kernel an ordinary proof term (Z.mul_nonneg / Z.add_nonneg, `ring` for the sum, Z.not_nonneg_neg), so a wrong search can only make
the tactic FAIL. What this gate pins down is that the search is right (it finds a certificate exactly when one exists), that the facts are read from the hypotheses with the right meaning, and that nothing
unsound reaches the kernel:

  A. the polynomial reader against Python polynomial arithmetic (random expressions with + - * neg ^, numerals in all spellings, `Z.sqr` as an atom): the monomials, the exponents, the signed coefficients;
  B. the search against an independent decision procedure: ~600 random systems of linear inequalities, a certificate exists exactly when exact Fourier-Motzkin elimination says the system has no rational solution;
     every certificate is checked in exact arithmetic (sum m_i P_i = -kappa); the SciPy-guided path and the exact simplex agree on larger systems; pruning changes no verdict;
  C. end to end, planted: systems built to be contradictory (a hidden non-negative combination of the hypotheses is a negative constant) are refuted in every presentation of the hypotheses (<=, <, =, ~<=, ~<,
     shifted by one) and every goal form (False, <=, <, =); consequences of the hypotheses are proved, claims false at a point that satisfies the hypotheses are refused (and the proof state is left untouched);
     fixed examples for strictness over Z, casts of naturals, hypotheses named alike, extra facts;
  D. `nlinarith`: classic inequalities with hints, products of hypotheses, squares of sub-terms, a degree-three certificate; false claims are refused;
  E. nothing unsound can come out: a wrong certificate from the search is caught by `ring` (an error, never a theorem), corrupted linarith proofs are rejected by the kernel and the independent checker alike;
  F. the independent checker re-checks the theorems proved here;
  G. speed: a chain of 40 hypotheses, many variables, a larger system through the SciPy-guided search.
"""
import os, random, sys, time
from fractions import Fraction
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(200000)
import int_lib, ring as R, linarith as LA
from author import AuthorError
from tactics import Proof
import indep_checker as ic
from harness import parse_raw
from s23_ring_tests import rebuild, dag_nodes
from s30_zlib_tests import Syntax, rand_expr, zpoly, small_num, terms_bound, zclean

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S31_TRACE'))
def trace(msg):
    if TRACE: print('   [%s]' % msg, flush=True)
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


# ---------------------------------------------------------------------------------------------------------------------- Python side: linear forms and an exact decision procedure
# a linear form is (dict variable -> coefficient, constant); the system {0 <= f_i}
def lf_add(a, b):
    r = dict(a[0])
    for j, c in b[0].items():
        v = r.get(j, 0) + c
        if v: r[j] = v
        else: r.pop(j, None)
    return (r, a[1] + b[1])

def lf_scale(a, c): return ({j: c * v for j, v in a[0].items()} if c else {}, c * a[1])
def lf_eval(a, x): return sum(c * x[j] for j, c in a[0].items()) + a[1]

def fm_infeasible(polys, nvars, cap=400):
    """exact Fourier-Motzkin elimination: True when the system  0 <= P_i  (the monomials ((j, 1),) as the variables) has no real solution; None when it grew past `cap` inequalities"""
    cons = [([Fraction(P.get(((j, 1),), 0)) for j in range(nvars)], Fraction(P.get((), 0))) for P in polys]
    for j in range(nvars):
        pos = [c for c in cons if c[0][j] > 0]; neg = [c for c in cons if c[0][j] < 0]; new = [c for c in cons if c[0][j] == 0]
        for a in pos:
            for b in neg:
                ca, cb = -b[0][j], a[0][j]
                new.append(([ca * x + cb * y for x, y in zip(a[0], b[0])], ca * a[1] + cb * b[1]))
        cons = list(dict.fromkeys((tuple(c[0]), c[1]) for c in new)); cons = [(list(a), b) for a, b in cons]
        if len(cons) > cap: return None
    return any(c[1] < 0 for c in cons)

def primal_infeasible(polys, nvars):
    """the same question put to HiGHS as the PRIMAL problem (find real numbers y with  c_i + sum_j a_ij y_j >= 0): True when there are none. Floating point, but small integers: a second, independent opinion"""
    import numpy as np
    from scipy.optimize import linprog
    A = np.array([[float(P.get(((j, 1),), 0)) for j in range(nvars)] for P in polys]); c = np.array([float(P.get((), 0)) for P in polys])
    res = linprog(np.zeros(nvars), A_ub=-A, b_ub=c, bounds=[(None, None)] * nvars, method='highs')
    return res.status == 2

def certificate_ok(polys, res):
    """the multipliers are non-negative integers, kappa >= 1, and  sum m_i P_i  is exactly the constant -kappa"""
    if res is None: return False
    mult, kappa = res
    if kappa < 1 or not mult or any(not isinstance(m, int) or m < 1 for m in mult.values()): return False
    g = kappa
    for m in mult.values(): g = gcd(g, m)
    if g != 1: return False                                          # the multipliers and kappa are reduced
    tot = {}
    for i, m in mult.items():
        for mono, c in polys[i].items():
            v = tot.get(mono, 0) + m * c
            if v: tot[mono] = v
            else: tot.pop(mono, None)
    return tot == {(): -kappa}

def random_system(rng, nvars, nfacts, density=0.6, cmax=3):
    polys = []
    for _ in range(nfacts):
        P = {}
        for j in range(nvars):
            if rng.random() < density: P[((j, 1),)] = rng.choice([c for c in range(-cmax, cmax + 1) if c])
        c = rng.randrange(-4, 5)
        if c: P[()] = c
        polys.append(P)
    return polys


# ---------------------------------------------------------------------------------------------------------------------- building statements
class Build:
    """terms for linear forms over the context variables xs, in several equivalent spellings"""
    def __init__(self, w, rng):
        self.w = w; self.rng = rng; self.inst = R.instance_for(w, w['Z']); self.sy = Syntax(w, rng)
        self.add, self.mul, self.sub, self.neg = w['Z.add'], w['Z.mul'], w['Z.sub'], w['Z.neg']; self.zero, self.one = w['Z.zero'], w['Z.one']
        self.le, self.lt, self.Eq, self.Not = w['Z.le'], w['Z.lt'], w.Eq, w.Not
    def lit(self, c): return self.sy.numeral(c)
    def term(self, lf, xs):
        rng = self.rng; parts = []
        for j, c in lf[0].items():
            r = rng.random()
            if c == 1 and r < 0.8: t = xs[j]
            elif r < 0.45: t = self.mul(self.lit(c), xs[j])
            elif r < 0.65: t = self.mul(xs[j], self.lit(c))
            elif r < 0.8 and c < 0: t = self.neg(self.mul(self.lit(-c), xs[j]))
            else: t = self.mul(self.lit(c), xs[j])
            parts.append(t)
        if lf[1] or not parts: parts.append(self.lit(lf[1]))
        rng.shuffle(parts); acc = parts[0]
        for t in parts[1:]: acc = self.add(acc, t)
        return acc
    def nonneg(self, lf, xs, form):
        """a proposition equivalent to  0 <= lf  over the integers, in one of 8 spellings"""
        P = self.term(lf, xs); Pn = self.term(lf_scale(lf, -1), xs); add, one, zero = self.add, self.one, self.zero
        if form == 0: return self.le(zero, P)
        if form == 1: return self.le(Pn, zero)
        if form == 2: return self.le(one, add(P, one))
        if form == 3: return self.Not(self.lt(P, zero))
        if form == 4: return self.lt(zero, add(P, one))
        if form == 5: return self.lt(self.lit(-1), P)
        if form == 6: return self.Not(self.le(add(P, one), zero))
        return self.lt(Pn, one)
    def zero_eq(self, lf, xs, form):
        """a proposition equivalent to  lf = 0"""
        P = self.term(lf, xs)
        if form == 0: return self.Eq(P, self.zero)
        if form == 1: return self.Eq(self.zero, P)
        # P = A - B with A, B arbitrary forms: Eq A B
        shift = ({j: self.rng.choice([-2, -1, 1, 2]) for j in list(lf[0])[:2]}, self.rng.randrange(-3, 4))
        A = lf_add(lf, shift); B = shift
        return self.Eq(self.term(A, xs), self.term(B, xs))


class Runner:
    def __init__(self, w):
        self.w = w; self.k = w.k; self.F = w.forall_; self.Z = w['Z']; self.counter = 0; self.proved = []; self.gid = {}
    def statement(self, n, hyps, goal):
        """hyps: functions xs -> proposition; goal: xs -> proposition"""
        doms = [self.Z] * n + [(lambda *vs, f=f: f(list(vs[:n]))) for f in hyps]
        return self.F(doms, lambda *vs: goal(list(vs[:n])))
    def prove(self, stmt, n, nh, tactic='linarith', hints=(), declare=True):
        """(ok, message): run the tactic on the statement; a finished proof is admitted by the kernel (and remembered)"""
        w, k = self.w, self.k
        p = Proof(w, stmt, 's31'); p.intro(*(['x%d' % j for j in range(n)] + ['h%d' % j for j in range(nh)]))
        before = str(p)
        try: getattr(p, tactic)(*hints)
        except AuthorError as e:
            expect(str(p) == before and p.n_goals == 1, 'C  a refused %s leaves the proof state as it was' % tactic)
            return False, str(e)
        if not p.done: return False, 'not finished'
        term = p.term(); self.counter += 1; nm = 's31_%d' % self.counter
        if declare:
            self.gid[nm] = k.declare_checked(w._n(stmt), term.n, nm); self.proved.append((nm, term, stmt))
        return True, ''


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    t_start = time.time(); rng = random.Random(31)
    t0 = time.time(); w = int_lib.build_int(); k = w.k; tb = time.time() - t0
    Z = w['Z']; Eq = w.Eq; F = w.forall_; zle, zlt = w['Z.le'], w['Z.lt']; Z0, Z1 = w['Z.zero'], w['Z.one']
    inst = R.instance_for(w, Z); sy0 = Syntax(w)
    zadd, zmul, zsub, zneg, zsqr, zof = (w[n] for n in ('Z.add', 'Z.mul', 'Z.sub', 'Z.neg', 'Z.sqr', 'Z.of_N'))
    run = Runner(w); B = Build(w, rng)
    lib_end = k.num_globals()

    # ------------------------------------------------------------------------------------------------------------------ A. the reader
    natoms = 3
    pv = Proof(w, F([Z, Z, Z], lambda a, b, c: Eq(a, a)), 'rd'); pv.intro('a', 'b', 'c'); n_read = 0
    with pv.focus():
        xs = [pv.h('a'), pv.h('b'), pv.h('c')]; sy = Syntax(w, rng); sqrs = [sy.sqr(x) for x in xs]
        for case in range(170):
            e = rand_expr(rng, natoms, 4, numeral=small_num)
            if terms_bound(e) > 300: continue
            t = sy.build(e, xs); rg = R.Ring(w, inst); got = LA.Reader(rg).poly(t); want = zclean(zpoly(e, natoms)); mine = {}
            for mono, c in got.items():
                v = [0] * (2 * natoms)
                for idx, ex in mono:
                    atom = rg.atoms[idx]
                    pos = xs.index(atom) if atom in xs else natoms + sqrs.index(atom)
                    v[pos] += ex
                mine[tuple(v)] = c
            expect(mine == want, 'A  the reader gets the polynomial of %s right' % (e,)); n_read += 1
        expect(n_read >= 120, 'A  enough random expressions were read (%d)' % n_read)
        # exponents: numerals, closed composite expressions, variables
        nl = inst.N.lit; nadd, nmul = w['N.add'], w['N.mul']
        rg = R.Ring(w, inst); rd = LA.Reader(rg); a = xs[0]; pw = w['Z.pow']
        expect(rd.poly(pw(zadd(a, Z1), nl(3))) == {(((0, 3)),): 1, ((0, 2),): 3, ((0, 1),): 3, (): 1}, 'A  (a + 1)^3')
        expect(rd.poly(pw(a, nadd(nl(2), nl(3)))) == {((0, 5),): 1}, 'A  a^(2 + 3) is a^5, as `ring` reads it')
        expect(rd.poly(pw(a, nmul(nl(2), nl(3)))) == {((0, 6),): 1}, 'A  a^(2 * 3) is a^6')
        expect(rd.poly(zsub(a, a)) == {}, 'A  a - a is zero')
        expect(rd.poly(zmul(inst.lit(-3), zneg(a))) == {((0, 1),): 3}, 'A  (-3) * (-a)')
        expect(rd.poly(zof(nl(7))) == {(): 7} and rd.poly(zneg(zof(nl(7)))) == {(): -7} and rd.poly(Z0) == {} and rd.poly(Z1) == {(): 1}, 'A  numerals')
        n_atom = zof(w['N0'])
        expect(rd.poly(zsqr(a)) == {((rg.atom(zsqr(a)), 1),): 1} and rd.poly(zmul(zsqr(a), zsqr(a))) == {((rg.atom(zsqr(a)), 2),): 1}, 'A  Z.sqr is an atom')
        big = {(): 1}
        pb_test = LA.Problem(w)
        with pv.focus():
            f1 = pb_test.add_hypothesis(zle(xs[0], xs[1]), xs[0], 'h'); f2 = pb_test.add_hypothesis(zle(xs[0], xs[1]), xs[0], 'h again'); f3 = pb_test.add_hypothesis(zle(Z0, zsub(xs[1], xs[0])), xs[0], 'the same fact spelled differently')
            expect(len(pb_test.facts) == 1, 'A  facts with the same polynomial are kept once (%d facts)' % len(pb_test.facts))
            pb_test.add_hypothesis(zle(xs[0], xs[0]), xs[0], 'trivial'); expect(len(pb_test.facts) == 1, 'A  0 <= 0 is not a fact')
            expect(pb_test.classify(w['Eq'](w['N'], w['N0'], w['N0'])) is None and pb_test.classify(w.Eq(xs[0], xs[1])) == ('eq', xs[0], xs[1]), 'A  an equation of naturals is no fact about integers')
            expect(pb_test.classify(zlt(xs[0], xs[1])) == ('lt', xs[0], xs[1]) and pb_test.classify(~zle(xs[0], xs[1])) == ('not_le', xs[0], xs[1]) and pb_test.classify(~zlt(xs[0], xs[1])) == ('not_lt', xs[0], xs[1])
                   and pb_test.classify(~Eq(xs[0], xs[1])) is None and pb_test.classify(w['False']) is None, 'A  the shapes of the hypotheses that are read')
        try: LA.p_pow({((0, 1),): 1, (): 1}, 65); got_err = False
        except AuthorError: got_err = True
        expect(got_err, 'A  the 65th power of a sum is not expanded')
    # a product that would have more than MAX_MONOS monomials is refused
    pbig = Proof(w, F([Z] * 201, lambda *v: Eq(v[0], v[0])), 'big'); pbig.intro(*['v%d' % i for i in range(201)])
    with pbig.focus():
        vs = [pbig.h('v%d' % i) for i in range(201)]; s_ = vs[0]
        for v_ in vs[1:200]: s_ = zadd(s_, v_)
        try: LA.Reader(R.Ring(w, inst)).poly(zmul(s_, s_)); got_err = False
        except AuthorError: got_err = True
        expect(got_err, 'A  a product with more than %d monomials is refused' % LA.MAX_MONOS)
        s_ = zadd(s_, vs[200])
        rd_ok = LA.Reader(R.Ring(w, inst)); expect(len(rd_ok.poly(s_)) == 201, 'A  a sum of 201 atoms is read')
    ta = time.time() - t0
    trace('section done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ B. the search against Fourier-Motzkin
    n_feas = n_infeas = n_skip = 0
    for case in range(600):
        nvars = rng.randrange(1, 5); nfacts = rng.randrange(1, 9)
        polys = random_system(rng, nvars, nfacts, density=rng.choice([0.4, 0.7, 1.0]))
        want = fm_infeasible(polys, nvars)
        if want is None: n_skip += 1; continue
        res = LA.combination(polys)
        expect((res is not None) == want, 'B  certificate exists = system infeasible (%s): %s' % (want, polys))
        if res is not None:
            expect(certificate_ok(polys, res), 'B  the certificate is exact: %s -> %s' % (polys, res)); n_infeas += 1
        else: n_feas += 1
    expect(n_infeas >= 120 and n_feas >= 120, 'B  both kinds of system occur (%d contradictory, %d satisfiable, %d skipped)' % (n_infeas, n_feas, n_skip))
    # the SciPy-guided path and the exact simplex agree; pruning does not matter
    saved = LA.EXACT_LIMIT; n_big = 0; agree_infeas = 0
    try:
        for case in range(120):
            nvars = rng.randrange(3, 7); nfacts = rng.randrange(6, 16)
            polys = random_system(rng, nvars, nfacts, density=rng.choice([0.3, 0.5, 0.8]))
            want = fm_infeasible(polys, nvars)
            if want is None: want = primal_infeasible(polys, nvars)
            LA.EXACT_LIMIT = 10 ** 9; r_exact = LA.combination(polys)
            LA.EXACT_LIMIT = 0; r_float = LA.combination(polys)
            expect((r_exact is not None) == want and (r_float is not None) == want, 'B  exact and SciPy-guided searches agree with the independent decision (%s): %s' % (want, polys))
            for r in (r_exact, r_float):
                if r is not None: expect(certificate_ok(polys, r), 'B  a certificate of the larger system is exact: %s' % polys)
            n_big += 1; agree_infeas += r_exact is not None
    finally: LA.EXACT_LIMIT = saved
    expect(agree_infeas >= 20, 'B  the larger systems are not all satisfiable (%d of %d are contradictory)' % (agree_infeas, n_big))
    expect(LA.STATS['float_guess'] >= agree_infeas // 2, 'B  the support guessed by SciPy is what the exact simplex went on to solve (%d times for %d contradictory systems)' % (LA.STATS['float_guess'], agree_infeas))
    # the guess of the support is a small part of a large contradictory system (a vertex has at most as many positive multipliers as there are rows)
    big_sys = random_system(rng, 4, 40, density=0.8)
    for _ in range(40):
        if fm_infeasible(big_sys, 4, 300): break
        big_sys = random_system(rng, 4, 40, density=0.8)
    rows_ = {}
    for P in big_sys:
        for m_ in P:
            if m_ and m_ not in rows_: rows_[m_] = len(rows_)
    cols_ = []
    for P in big_sys:
        col = {rows_[m_]: c for m_, c in P.items() if m_}
        if P.get((), 0): col[len(rows_)] = -P[()]
        cols_.append(col)
    sup_ = LA.float_support(cols_, len(rows_) + 1, len(rows_))
    expect(sup_ is not None and 1 <= len(sup_) <= len(rows_) + 1 and len(sup_) < len(cols_), 'B  the SciPy guess of the support is a vertex (%s of %d columns, %d rows)' % (None if sup_ is None else len(sup_), len(cols_), len(rows_) + 1))
    # a system whose simplex solution has a common factor: the result is reduced
    gcd_case = [{((1, 1),): -3, ((2, 1),): -1, (): 3}, {((1, 1),): 2, ((2, 1),): 4, (): 1}, {((0, 1),): -2, ((2, 1),): 1, (): 4}, {((0, 1),): 2, ((2, 1),): -3, (): -3}, {((0, 1),): -2, ((2, 1),): 4, (): 2}]
    expect(LA.combination(gcd_case) == ({2: 1, 3: 3, 4: 2}, 1), 'B  the multipliers and kappa are divided by their common factor: %s' % (LA.combination(gcd_case),))
    expect(LA.combination([{((0, 1),): 1}, {((0, 1),): -1}]) is None and LA.combination([{(): 1}]) is None and LA.combination([]) is None, 'B  nothing to refute: no certificate')
    expect(LA.combination([{(): -1}]) == ({0: 1}, 1) and LA.combination([{(): -5}, {(): 3}]) == ({0: 1}, 5), 'B  a negative constant alone is the certificate')
    # the pruning keeps exactly the columns that can matter
    cols = [{0: 1, 3: -1}, {0: -1, 3: 1}, {1: 5, 3: -2}, {2: -1, 3: 1}]
    expect(LA.prune(cols, 4) == [0, 1], 'B  prune: the columns with an uncancelled monomial are dropped, then the ones that depended on them')
    expect(LA.prune([{0: 1, 1: -1}, {0: -1, 1: 1}], 2) == [0, 1], 'B  prune keeps a balanced pair')
    tb2 = time.time() - t0
    trace('section done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ C. end to end, planted
    def rand_form(n, cmax=3): return ({j: rng.choice([c for c in range(-cmax, cmax + 1) if c]) for j in range(n) if rng.random() < 0.7}, rng.randrange(-4, 5))
    n_ok = n_refused = 0; forms_seen = set()
    for case in range(36):
        n = rng.randrange(2, 4); k_h = rng.randrange(2, 4)
        ps = [rand_form(n) for _ in range(k_h)]
        mus = [rng.randrange(1, 4) for _ in ps]
        total = ({}, 0)
        for mu, p in zip(mus, ps): total = lf_add(total, lf_scale(p, mu))
        kappa = rng.randrange(1, 4)
        q = lf_add(lf_scale(({j: c for j, c in total[0].items()}, 0), -1), ({}, -kappa - total[1]))          # sum mu_i p_i + q = -kappa
        facts = ps + [q]; forms = [rng.randrange(8) for _ in facts]; forms_seen.update(forms)
        # one fact may be an equation instead: use  q = 0  shifted so that it is not contradictory by itself -- keep it simple: replace the last by an equation when its negation is also allowed
        mode = rng.choice(['false', 'false', 'goal_le', 'goal_lt', 'goal_eq', 'refuse'])
        if mode == 'false':
            stmt = run.statement(n, [(lambda xs, f=f, fo=fo: B.nonneg(f, xs, fo)) for f, fo in zip(facts, forms)], lambda xs: w['False'])
            ok, msg = run.prove(stmt, n, len(facts))
            expect(ok, 'C  planted contradiction refuted (%s): %s' % (facts, msg)); n_ok += ok
        elif mode == 'goal_le':
            # hypotheses p_i; goal  0 <= sum mu_i p_i + t   with t >= 0  presented as  -(that) <= 0  or  0 <= that
            t = rng.randrange(0, 3); g = lf_add(total, ({}, t))
            stmt = run.statement(n, [(lambda xs, f=f, fo=fo: B.nonneg(f, xs, fo)) for f, fo in zip(ps, forms)], lambda xs: B.le(B.zero, B.term(g, xs)) if rng.random() < 0.5 else B.le(B.term(lf_scale(g, -1), xs), B.zero))
            ok, msg = run.prove(stmt, n, len(ps)); expect(ok, 'C  a consequence is proved (<=): %s' % msg); n_ok += ok
        elif mode == 'goal_lt':
            t = rng.randrange(1, 3); g = lf_add(total, ({}, t))
            stmt = run.statement(n, [(lambda xs, f=f, fo=fo: B.nonneg(f, xs, fo)) for f, fo in zip(ps, forms)], lambda xs: B.lt(B.term(lf_add(lf_scale(g, -1), ({}, 0)), xs), B.zero))
            ok, msg = run.prove(stmt, n, len(ps)); expect(ok, 'C  a consequence is proved (<): %s' % msg); n_ok += ok
        elif mode == 'goal_eq':
            # hypotheses  p = 0  and  r = 0  (equations in several spellings); goal  a p + b r = 0
            p1, p2 = ps[0], ps[1]; a, b = rng.randrange(-2, 3), rng.randrange(-2, 3)
            e1, e2 = ({j: c for j, c in p1[0].items()}, 0), ({j: c for j, c in p2[0].items()}, 0)
            g = lf_add(lf_scale(e1, a), lf_scale(e2, b))
            stmt = run.statement(n, [lambda xs: B.zero_eq(e1, xs, forms[0] % 3), lambda xs: B.zero_eq(e2, xs, forms[1] % 3)], lambda xs: B.Eq(B.term(g, xs), B.zero))
            ok, msg = run.prove(stmt, n, 2); expect(ok, 'C  a consequence of two equations is proved (=): %s' % msg); n_ok += ok
        else:
            # a point that satisfies every hypothesis, and a claim that fails there: nothing can prove it
            x0 = [rng.randrange(-3, 4) for _ in range(n)]; hy = []
            for p in ps:
                v = lf_eval(p, x0); hy.append((p[0], p[1] + max(0, -v) + rng.randrange(0, 2)))
            gl = rand_form(n); v = lf_eval(gl, x0); gl = (gl[0], gl[1] - max(0, v) - 1)               # gl(x0) < 0:  0 <= gl is false at x0
            stmt = run.statement(n, [(lambda xs, f=f, fo=fo: B.nonneg(f, xs, fo)) for f, fo in zip(hy, forms)], lambda xs: B.le(B.zero, B.term(gl, xs)))
            ok, msg = run.prove(stmt, n, len(hy))
            expect(not ok and 'linarith failed' in msg, 'C  a claim that is false at a point satisfying the hypotheses is refused: %s' % msg); n_refused += not ok
    expect(len(forms_seen) >= 7, 'C  the hypotheses came in %d of the 8 spellings' % len(forms_seen))
    expect(n_ok >= 20 and n_refused >= 3, 'C  the planted cases are not vacuous (%d proved, %d refused)' % (n_ok, n_refused))
    # fixed examples
    x, y, z = ('x0', 'x1', 'x2')
    def fixed(label, n, hyps, goal, want_ok=True, tactic='linarith', hints=()):
        stmt = run.statement(n, hyps, goal); ok, msg = run.prove(stmt, n, len(hyps), tactic, hints)
        expect(ok == want_ok, 'C  %s (%s)' % (label, msg[:200])); return ok
    le, lt, add, mul = B.le, B.lt, B.add, B.mul; lit = inst.lit; one = Z1; zero = Z0
    fixed('x <= y, y <= z |- x <= z', 3, [lambda v: le(v[0], v[1]), lambda v: le(v[1], v[2])], lambda v: le(v[0], v[2]))
    fixed('x < y, y < z |- x + 2 <= z (strictness over Z)', 3, [lambda v: lt(v[0], v[1]), lambda v: lt(v[1], v[2])], lambda v: le(add(v[0], lit(2)), v[2]))
    fixed('x < y, y < x + 1 |- False (no integer between)', 2, [lambda v: lt(v[0], v[1]), lambda v: lt(v[1], add(v[0], one))], lambda v: w['False'])
    fixed('2x + 1 <= y, y <= 2x |- False', 2, [lambda v: le(add(mul(lit(2), v[0]), one), v[1]), lambda v: le(v[1], mul(lit(2), v[0]))], lambda v: w['False'])
    fixed('x <= y |- y <= x is refused', 2, [lambda v: le(v[0], v[1])], lambda v: le(v[1], v[0]), want_ok=False)
    fixed('x <= y |- x < y is refused', 2, [lambda v: le(v[0], v[1])], lambda v: lt(v[0], v[1]), want_ok=False)
    fixed('x <= y, y <= z, z <= x |- x = z', 3, [lambda v: le(v[0], v[1]), lambda v: le(v[1], v[2]), lambda v: le(v[2], v[0])], lambda v: Eq(v[0], v[2]))
    fixed('x = y |- y <= x', 2, [lambda v: Eq(v[0], v[1])], lambda v: le(v[1], v[0]))
    fixed('x = y, y = z |- x = z', 3, [lambda v: Eq(v[0], v[1]), lambda v: Eq(v[1], v[2])], lambda v: Eq(v[0], v[2]))
    fixed('3x + 2y = 7, x = 1 |- y = 2', 2, [lambda v: Eq(add(mul(lit(3), v[0]), mul(lit(2), v[1])), lit(7)), lambda v: Eq(v[0], one)], lambda v: Eq(v[1], lit(2)))
    fixed('~(x <= y) |- y < x', 2, [lambda v: B.Not(le(v[0], v[1]))], lambda v: lt(v[1], v[0]))
    fixed('~(x < y) |- y <= x', 2, [lambda v: B.Not(lt(v[0], v[1]))], lambda v: le(v[1], v[0]))
    fixed('hypotheses that contradict each other prove anything', 2, [lambda v: lt(v[0], v[1]), lambda v: lt(v[1], v[0])], lambda v: Eq(v[0], lit(12345)))
    fixed('a goal that is no inequality, from contradictory hypotheses', 1, [lambda v: lt(v[0], v[0])], lambda v: w['Or'](Eq(v[0], zero), Eq(v[0], one)))
    fixed('a goal that is no inequality, hypotheses consistent: refused', 1, [lambda v: le(zero, v[0])], lambda v: w['Or'](Eq(v[0], zero), Eq(v[0], one)), want_ok=False)
    fixed('products are atoms: x*y <= z, z <= x*y - 1 |- False', 3, [lambda v: le(mul(v[0], v[1]), v[2]), lambda v: le(v[2], zsub(mul(v[0], v[1]), one))], lambda v: w['False'])
    fixed('products are atoms up to ring: x*y <= z, z + 1 <= y*x |- False', 3, [lambda v: le(mul(v[0], v[1]), v[2]), lambda v: le(add(v[2], one), mul(v[1], v[0]))], lambda v: w['False'])
    fixed('squares are atoms: x*x <= 3, 4 <= x^2 |- False', 1, [lambda v: le(mul(v[0], v[0]), lit(3)), lambda v: le(lit(4), w['Z.pow'](v[0], inst.N.lit(2)))], lambda v: w['False'])
    fixed('64-bit coefficients: x <= 2^64, 2^64 + 1 <= x |- False', 1, [lambda v: le(v[0], lit(2 ** 64)), lambda v: le(lit(2 ** 64 + 1), v[0])], lambda v: w['False'])
    fixed('the same hypothesis twice', 2, [lambda v: le(v[0], v[1]), lambda v: le(v[0], v[1])], lambda v: le(v[0], v[1]))
    # casts of naturals: Z.of_N t is non-negative
    stmt = F([w['N'], w['N']], lambda a, b: le(zero, add(zof(a), zof(b)))); ok, msg = run.prove(stmt, 2, 0); expect(ok, 'C  of_N a + of_N b >= 0 (%s)' % msg)
    stmt = F([w['N'], lambda a: le(zof(a), lit(-1))], lambda a, h: w['False']); ok, msg = run.prove(stmt, 1, 1); expect(ok, 'C  of_N a <= -1 is absurd (%s)' % msg)
    stmt = F([w['N'], w['N'], lambda a, b: lt(zof(a), zof(b))], lambda a, b, h: lt(zero, zof(b))); ok, msg = run.prove(stmt, 2, 1); expect(ok, 'C  of_N a < of_N b |- 0 < of_N b (%s)' % msg)
    stmt = F([w['N'], w['N'], lambda a, b: Eq(a, b)], lambda a, b, h: w['False'])
    pn = Proof(w, stmt, 'neq'); pn.intro('a', 'b', 'h')
    try: pn.linarith(); msg = 'proved'
    except AuthorError as e: msg = str(e)
    expect('linarith failed' in msg, 'C  an equation of naturals is not read as a fact about integers (%s)' % msg[:100])
    # extra facts (proofs given to the tactic)
    stmt = F([Z, Z], lambda a, b: le(mul(lit(2), mul(a, b)), add(mul(a, a), mul(b, b)))); ok, msg = run.prove(stmt, 2, 0, 'linarith', (('Z.sq_nonneg', (zsub, 'x0', 'x1')),))
    expect(ok, 'C  an extra fact: (a - b)^2 >= 0 gives 2ab <= a^2 + b^2 (%s)' % msg)
    ok, msg = run.prove(stmt, 2, 0, 'linarith'); expect(not ok, 'C  the same without the extra fact is refused')
    # names: two hypotheses with the same name are both used
    stmt = F([Z, lambda a: le(zero, a), lambda a, h: le(a, lit(-1))], lambda a, h1, h2: w['False'])
    p = Proof(w, stmt, 'names'); p.intro('a', 'h', 'h'); p.linarith(); expect(p.done, 'C  hypotheses with the same name are both read')
    tc = time.time() - t0
    trace('section done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ D. nlinarith
    sq = lambda a, b: ('Z.sq_nonneg', (zsub, a, b))
    def nl(label, n, hyps, goal, want_ok=True, hints=()): return fixed('nlinarith: ' + label, n, hyps, goal, want_ok, 'nlinarith', hints)
    pw = w['Z.pow']
    nl('0 <= a, 0 <= b |- 0 <= a*b', 2, [lambda v: le(zero, v[0]), lambda v: le(zero, v[1])], lambda v: le(zero, mul(v[0], v[1])))
    nl('a <= b, 0 <= c |- a*c <= b*c', 3, [lambda v: le(v[0], v[1]), lambda v: le(zero, v[2])], lambda v: le(mul(v[0], v[2]), mul(v[1], v[2])))
    nl('1 <= a, 1 <= b |- a + b <= a*b + 1', 2, [lambda v: le(one, v[0]), lambda v: le(one, v[1])], lambda v: le(add(v[0], v[1]), add(mul(v[0], v[1]), one)))
    nl('0 <= a |- 0 <= a*a*a (a square, then a product)', 1, [lambda v: le(zero, v[0])], lambda v: le(zero, mul(mul(v[0], v[0]), v[0])))
    nl('a <= b, b <= c |- 0 <= (b - a)*(c - b)', 3, [lambda v: le(v[0], v[1]), lambda v: le(v[1], v[2])], lambda v: le(zero, mul(zsub(v[1], v[0]), zsub(v[2], v[1]))))
    nl('3 <= a |- 6a <= a*a + 9 (the square of a hypothesis, no hint)', 1, [lambda v: le(lit(3), v[0])], lambda v: le(mul(lit(6), v[0]), add(mul(v[0], v[0]), lit(9))))
    nl('0 <= a*a (a square found in the goal)', 1, [], lambda v: le(zero, mul(v[0], v[0])))
    nl('0 <= a^2 (a power found in the goal)', 1, [], lambda v: le(zero, pw(v[0], inst.N.lit(2))))
    nl('2ab <= a^2 + b^2 with the hint (a-b)^2', 2, [], lambda v: le(mul(lit(2), mul(v[0], v[1])), add(mul(v[0], v[0]), mul(v[1], v[1]))), hints=(sq('x0', 'x1'),))
    nl('4ab <= (a+b)^2 with the hint (a-b)^2', 2, [], lambda v: le(mul(lit(4), mul(v[0], v[1])), mul(add(v[0], v[1]), add(v[0], v[1]))), hints=(sq('x0', 'x1'),))
    nl('ab + bc + ca <= a^2 + b^2 + c^2 with three hints', 3, [], lambda v: le(add(add(mul(v[0], v[1]), mul(v[1], v[2])), mul(v[2], v[0])), add(add(mul(v[0], v[0]), mul(v[1], v[1])), mul(v[2], v[2]))),
       hints=(sq('x0', 'x1'), sq('x1', 'x2'), sq('x2', 'x0')))
    nl('Cauchy-Schwarz in the plane with the hint (ad - bc)^2', 4, [], lambda v: le(mul(add(mul(v[0], v[2]), mul(v[1], v[3])), add(mul(v[0], v[2]), mul(v[1], v[3]))),
                                                                                    mul(add(mul(v[0], v[0]), mul(v[1], v[1])), add(mul(v[2], v[2]), mul(v[3], v[3])))),
       hints=(('Z.sq_nonneg', (zsub, (zmul, 'x0', 'x3'), (zmul, 'x1', 'x2'))),))
    nl('a, b >= 0 |- a*b*(a+b) <= a^3 + b^3 (degree three: hypotheses times a hint)', 2, [lambda v: le(zero, v[0]), lambda v: le(zero, v[1])],
       lambda v: le(mul(mul(v[0], v[1]), add(v[0], v[1])), add(mul(mul(v[0], v[0]), v[0]), mul(mul(v[1], v[1]), v[1]))), hints=(sq('x0', 'x1'),))
    nl('refused: a*b <= a*a', 2, [], lambda v: le(mul(v[0], v[1]), mul(v[0], v[0])), want_ok=False)
    nl('refused: 0 <= a*b without hypotheses', 2, [], lambda v: le(zero, mul(v[0], v[1])), want_ok=False)
    nl('refused: a <= b |- a*a <= b*b', 2, [lambda v: le(v[0], v[1])], lambda v: le(mul(v[0], v[0]), mul(v[1], v[1])), want_ok=False)
    # random: products of hypotheses.  0 <= p, 0 <= q  |-  0 <= p*q + t  (t >= 0)
    n_nl = 0
    for case in range(8):
        n = 2; p, q = rand_form(n, 2), rand_form(n, 2); t = rng.randrange(0, 3)
        g = ({}, 0)          # the goal polynomial p*q + t is not linear: write it as a term
        forms = [rng.randrange(8), rng.randrange(8)]
        stmt = run.statement(n, [lambda xs: B.nonneg(p, xs, forms[0]), lambda xs: B.nonneg(q, xs, forms[1])], lambda xs: le(zero, add(mul(B.term(p, xs), B.term(q, xs)), lit(t))))
        ok, msg = run.prove(stmt, n, 2, 'nlinarith'); expect(ok, 'D  a product of hypotheses: 0 <= p, 0 <= q |- 0 <= p*q + %d (%s)' % (t, msg[:150])); n_nl += ok
    # a failing run mentions the products
    ok, msg = run.prove(run.statement(2, [], lambda v: le(mul(v[0], v[1]), mul(v[0], v[0]))), 2, 0, 'nlinarith'); expect(not ok and 'products' in msg, 'D  the failure message of nlinarith mentions the products')
    td = time.time() - t0
    trace('section done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ E. nothing unsound can come out
    real = LA.combination
    stmt = run.statement(2, [lambda v: le(v[0], v[1]), lambda v: le(v[1], v[0])], lambda v: lt(v[0], v[1]))        # not provable; hypotheses are consistent
    try:
        LA.combination = lambda polys: ({i: 1 for i in range(len(polys))}, 1)         # a certificate that is wrong (the facts do not add up to -1)
        ok, msg = run.prove(stmt, 2, 2)
        expect(not ok and 'internal error' in msg, 'E  a wrong certificate is caught by ring, not turned into a proof (%s)' % msg[:120])
        stmt2 = run.statement(2, [lambda v: le(v[0], v[1])], lambda v: le(v[1], v[0]))
        LA.combination = lambda polys: ({0: 2}, 1)
        ok, msg = run.prove(stmt2, 2, 1); expect(not ok, 'E  another wrong certificate is not a proof either')
    finally: LA.combination = real
    # corrupted proofs: the kernel and the independent checker agree on every one, and most are rejected
    stmt_c = run.statement(3, [lambda v: le(v[0], v[1]), lambda v: lt(v[1], v[2]), lambda v: le(add(v[2], one), v[0])], lambda v: w['False'])
    pc = Proof(w, stmt_c, 'corrupt'); pc.intro('x0', 'x1', 'x2', 'h0', 'h1', 'h2'); pc.linarith(); proof_c = pc.term()
    env = ic.Env()
    for gid in range(lib_end):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid), recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    chk_small = ic.Checker(env, params=True, fuel=1_500_000, prop_elim=True)
    stmt_h = parse_raw(k.dump_raw(stmt_c.n), True)
    pool = [n for n in dag_nodes(k, proof_c.n) if k.view(n)[0] in ('app', 'const', 'lam')]
    lemma_consts = [k.mk_const(w.gid[nm]) for nm in ('Z.add_nonneg', 'Z.mul_nonneg', 'Z.nonneg_of_N', 'Z.sub_nonneg_of_le', 'Z.le_of_le_eq', 'Z.not_nonneg_neg', 'Z.le_of_eq', 'Z.lt_of_not_le', 'Z.le_of_not_lt')]
    kinds = {'kernel_rej': 0, 'kernel_acc': 0, 'disagree': 0, 'undecided': 0}
    for case in range(36):
        target = rng.choice(pool); v = k.view(target); r = rng.random()
        if r < 0.3: repl = rng.choice(pool)
        elif r < 0.5 and v[0] == 'app' and k.view(v[1])[0] == 'app':
            inner = k.view(v[1]); repl = k.mk_app(k.mk_app(inner[1], v[2]), inner[2])
        elif r < 0.8: repl = rng.choice(lemma_consts)
        else: repl = k.mk_const(w.gid['Z.zero'])
        if repl == target: continue
        try: mutant = rebuild(k, proof_c.n, {target: repl})
        except ValueError: continue
        if mutant == proof_c.n: continue
        try: kok = bool(k.check_open(mutant, stmt_c.n, [])[0])
        except Exception: kok = False
        try: iok, imsg = chk_small.check_closed(parse_raw(k.dump_raw(mutant), True), stmt_h)
        except ic.Unsupported: continue
        if not iok and 'fuel' in imsg: kinds['undecided'] += 1; continue
        if kok != iok: kinds['disagree'] += 1; FAIL.append('E  the kernel (%s) and the independent checker (%s: %s) disagree about a corrupted linarith proof' % (kok, iok, imsg))
        elif kok: kinds['kernel_acc'] += 1
        else: kinds['kernel_rej'] += 1
    expect(kinds['kernel_rej'] >= 15, 'E  the corruption test is not vacuous (%s)' % kinds)
    expect(k.check_open(proof_c.n, stmt_c.n, [])[0] and chk_small.__class__ is not None, 'E  the uncorrupted proof is accepted')
    te = time.time() - t0
    trace('section done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ G. speed (before the independent re-check)
    t1 = time.time(); m = 40
    stmt = run.statement(m + 1, [(lambda v, i=i: le(v[i], v[i + 1])) for i in range(m)], lambda v: le(v[0], v[m]))
    ok, msg = run.prove(stmt, m + 1, m); dt_chain = time.time() - t1
    expect(ok and dt_chain < 20, 'G  a chain of %d hypotheses (%.2f s, %s)' % (m, dt_chain, msg[:100]))
    t1 = time.time(); n = 12
    stmt = run.statement(n, [(lambda v, i=i: le(add(v[i], one), v[(i + 1) % n])) for i in range(n)], lambda v: w['False'])
    ok, msg = run.prove(stmt, n, n); dt_cycle = time.time() - t1
    expect(ok and dt_cycle < 20, 'G  a cycle of %d strict inequalities (%.2f s, %s)' % (n, dt_cycle, msg[:100]))
    # a larger planted system through the SciPy-guided search
    t1 = time.time(); n = 8; ps = [rand_form(n) for _ in range(14)]; mus = [rng.randrange(0, 3) for _ in ps]; mus[0] = 1
    total = ({}, 0)
    for mu, p in zip(mus, ps): total = lf_add(total, lf_scale(p, mu))
    qq = lf_add(lf_scale(({j: c for j, c in total[0].items()}, 0), -1), ({}, -1 - total[1])); facts = ps + [qq]
    saved = LA.EXACT_LIMIT
    try:
        LA.EXACT_LIMIT = 0
        stmt = run.statement(n, [(lambda xs, f=f: B.nonneg(f, xs, 0)) for f in facts], lambda xs: w['False'])
        ok, msg = run.prove(stmt, n, len(facts)); expect(ok, 'G  a planted system of %d facts in %d variables through the SciPy-guided search (%s)' % (len(facts), n, msg[:100]))
    finally: LA.EXACT_LIMIT = saved
    dt_big = time.time() - t1
    tg = time.time() - t0
    trace('section done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ F. the independent checker over the theorems proved here
    t2 = time.time()
    for gid in range(lib_end, k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid), recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    chk = ic.Checker(env, params=True, fuel=200_000_000, prop_elim=True)
    n_checked = 0; bad = []; tree_limit = int(os.environ.get('S31_INDEP_TREE', '30000'))
    for gid, gl in chk.g.items():
        if gid < lib_end: continue
        if gl.value is None or k.node_info(k.global_info(gid)[2])[2] > tree_limit: continue
        ok, msg = chk.run(lambda: (chk.sort_of({}, gl.type), None if gl.value is None else chk.check({}, gl.value, gl.type)))
        n_checked += 1
        if not ok: bad.append((gl.name, msg))
    expect(n_checked >= 25 and not bad, 'F  the independent checker re-checks %d of the %d theorems proved here without a rejection: %s' % (n_checked, len(run.proved), bad[:3]))
    tf = time.time() - t2

    print("   %d checks; world built in %.1f s; reader: %d expressions; search: %d contradictory and %d satisfiable systems, %d larger ones in both modes; %d theorems proved and admitted; "
          "corrupted proofs %s; chain of %d: %.2f s, cycle of %d: %.2f s, SciPy-guided system: %.2f s; independent re-check of %d theorems in %.1f s (%.1f s total)" %
          (CHECKS[0], tb, n_read, n_infeas, n_feas, n_big, len(run.proved), kinds, m, dt_chain, 12, dt_cycle, dt_big, n_checked, tf, time.time() - t_start))
    print("   sections (cumulative s): A %.1f, B %.1f, C %.1f, D %.1f, E %.1f, G %.1f" % (ta, tb2, tc, td, te, tg))
    for f in FAIL[:40]: print("   FAIL " + f)
    if len(FAIL) > 40: print("   ... and %d more" % (len(FAIL) - 40))
    print("S31 LINARITH GATE: " + ("PASS" if not FAIL else "FAIL"))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
