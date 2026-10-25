"""S34 gate: elementary number theory over the integers (nt_lib.py) and the first olympiad problems (imo_lib.py), `linear_combination` (lincomb.py).

Nothing in these libraries is trusted: definitions and statements are plain terms, every theorem is the kernel's re-check of a tactic script's proof term. What this gate pins down is WHAT the library says (a wrong
statement is still a theorem, just not the one the library claims), that the definitions mean divisibility / gcd / coprimality / primality, which results rest on the one axiom, and that the tactic added here is sound:

  A. the statements: ~60 theorems written out here with explicit operations and compared as terms with what the libraries declared (and: nothing added by the number-theory part is missing from the table);
     the definitions have the intended types and the intended unfolding (conversion checks, with negative controls: `a | b` is `b = a k`, not `a = b k`); the world has exactly ONE axiom, `em`, and the set of
     theorems that rest on it is exactly the expected one -- everything else (Bezout, Euclid's lemma, the problems) is constructive;
  B. the definitions on numerals, against Python: for every pair in a range, the library PROVES `a | b` (witness) and REFUTES `a | b` (a remainder) exactly when Python says so; coprimality with Bezout
     coefficients from the extended Euclidean algorithm and refutation for gcd > 1 through the library's lemmas; `is_gcd` of the gcd Python computes, with a proof, and uniqueness; primes up to 40 proved and
     composites refuted -- all by the library's lemmas, checked by the kernel;
  C. `linear_combination`: hundreds of random identities (a planted combination of random equations is accepted, a perturbed coefficient or a perturbed hypothesis is refused with both normal forms), over Z, N
     and Nat; wrong-type hypotheses; the proof is the ring/congruence/cancellation chain and nothing else;
  D. the problems: IMO 1959 P1, sqrt 2 and IMO 2019 P1 -- statements written out here; non-vacuity (the equation of IMO 2019 P1 holds for the solutions and is refuted for other functions by the kernel; the
     irrationality statement is refuted when `2` is replaced by `4`, the gcd statement fails for the pair (21n+4, 14n+6)); the proof scripts of the problems REFUSE a perturbed statement;
  E. the independent checker re-checks every global of the number-theory part and the problems; corrupted proofs of library theorems: the kernel and the independent checker agree on every one.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, sys, time
from math import gcd

sys.setrecursionlimit(200000)
import itp_env_lnl as E
import int_lib, nt_lib, imo_lib, ext_lib, ring as R, audit
from author import AuthorError
from tactics import Proof
import indep_checker as ic
from harness import parse_raw
from s23_ring_tests import rebuild, dag_nodes

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S34_TRACE'))
def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


# ---------------------------------------------------------------------------------------------------------------------- A. the statements
def ntspec(w):
    """the statements of the number-theory libraries, written out with explicit operations (no notation, nothing taken from nt_lib / imo_lib)"""
    TYPE, PROP = w.TYPE, w.PROP; Eq = w.Eq; F = w.forall_; Ex = w['Ex']; Fls = w['False']
    N, Z = w['N'], w['Z']; N0 = w['N0']
    zadd, zmul, zneg, zsub, zof = (w[n] for n in ('Z.add', 'Z.mul', 'Z.neg', 'Z.sub', 'Z.of_N'))
    Z0, Z1 = w['Z.zero'], w['Z.one']; zle, zlt = w['Z.le'], w['Z.lt']; nsucc = w['N.succ']
    dvd, cop, gcdr, prime = w['Z.dvd'], w['Z.coprime'], w['Z.is_gcd'], w['Z.prime']
    def numeral(v):
        """the canonical numeral of a non-negative value: Z.of_N of the binary natural"""
        if v == 0: return Z0
        p = w['XH']
        for bit in bin(v)[3:]: p = (w['XI'] if bit == '1' else w['XO'])(p)
        return zof(w['Npos'](p))
    Z2, Z3, Z4, Z14, Z21 = numeral(2), numeral(3), numeral(4), numeral(14), numeral(21)
    PZ = w.arrow(Z, PROP)
    S = {}
    # ---- induction
    S['Z.int_induction'] = F([PZ, lambda P: P(Z0), lambda P, h0: F([Z, lambda n: P(n)], lambda n, hn: P(zadd(n, Z1))),
                              lambda P, h0, hs: F([Z, lambda n: P(n)], lambda n, hn: P(zsub(n, Z1))), Z], lambda P, h0, hs, hp, z: P(z))
    S['Z.induction_from'] = F([PZ, Z, lambda P, a: P(a), lambda P, a, h0: F([Z, lambda n: zle(a, n), lambda n, hn: P(n)], lambda n, hn, hp: P(zadd(n, Z1))), Z, lambda P, a, h0, hs, n: zle(a, n)],
                              lambda P, a, h0, hs, n, hn: P(n))
    S['Z.strong_induction'] = F([PZ, Z, lambda P, a: F([Z, lambda n: zle(a, n), lambda n, hn: F([Z, lambda m: zle(a, m), lambda m, h1: zlt(m, n)], lambda m, h1, h2: P(m))], lambda n, hn, ih: P(n)),
                                 Z, lambda P, a, H, n: zle(a, n)], lambda P, a, H, n, hn: P(n))
    # ---- divisibility:  a | b  is  exists k, b = a * k
    S['Z.dvd_trans'] = F([Z, Z, Z, lambda a, b, c: dvd(a, b), lambda a, b, c, h: dvd(b, c)], lambda a, b, c, h1, h2: dvd(a, c))
    S['Z.dvd_refl'] = F([Z], lambda a: dvd(a, a))
    S['Z.dvd_zero'] = F([Z], lambda a: dvd(a, Z0))
    S['Z.one_dvd'] = F([Z], lambda a: dvd(Z1, a))
    S['Z.dvd_mul_right'] = F([Z, Z], lambda a, b: dvd(a, zmul(a, b)))
    S['Z.dvd_mul_left'] = F([Z, Z], lambda a, b: dvd(a, zmul(b, a)))
    S['Z.dvd_add'] = F([Z, Z, Z, lambda a, b, c: dvd(a, b), lambda a, b, c, h: dvd(a, c)], lambda a, b, c, h1, h2: dvd(a, zadd(b, c)))
    S['Z.dvd_neg'] = F([Z, Z, lambda a, b: dvd(a, b)], lambda a, b, h: dvd(a, zneg(b)))
    S['Z.dvd_of_dvd_neg'] = F([Z, Z, lambda a, b: dvd(a, zneg(b))], lambda a, b, h: dvd(a, b))
    S['Z.dvd_sub'] = F([Z, Z, Z, lambda a, b, c: dvd(a, b), lambda a, b, c, h: dvd(a, c)], lambda a, b, c, h1, h2: dvd(a, zsub(b, c)))
    S['Z.neg_dvd'] = F([Z, Z, lambda a, b: dvd(a, b)], lambda a, b, h: dvd(zneg(a), b))
    S['Z.dvd_of_neg_dvd'] = F([Z, Z, lambda a, b: dvd(zneg(a), b)], lambda a, b, h: dvd(a, b))
    S['Z.dvd_mul_of_dvd_left'] = F([Z, Z, Z, lambda a, b, c: dvd(a, b)], lambda a, b, c, h: dvd(a, zmul(b, c)))
    S['Z.dvd_mul_of_dvd_right'] = F([Z, Z, Z, lambda a, b, c: dvd(a, c)], lambda a, b, c, h: dvd(a, zmul(b, c)))
    S['Z.mul_dvd_mul'] = F([Z, Z, Z, Z, lambda a, b, c, d: dvd(a, b), lambda a, b, c, d, h: dvd(c, d)], lambda a, b, c, d, h1, h2: dvd(zmul(a, c), zmul(b, d)))
    S['Z.dvd_of_dvd_add_left'] = F([Z, Z, Z, lambda a, b, c: dvd(a, b), lambda a, b, c, h: dvd(a, zadd(b, c))], lambda a, b, c, h1, h2: dvd(a, c))
    S['Z.dvd_of_dvd_add_right'] = F([Z, Z, Z, lambda a, b, c: dvd(a, c), lambda a, b, c, h: dvd(a, zadd(b, c))], lambda a, b, c, h1, h2: dvd(a, b))
    S['Z.eq_zero_of_zero_dvd'] = F([Z, lambda a: dvd(Z0, a)], lambda a, h: Eq(a, Z0))
    S['Z.eq_one_of_mul_eq_one_pos'] = F([Z, Z, lambda a, k: zlt(Z0, a), lambda a, k, h: Eq(Z1, zmul(a, k))], lambda a, k, h, e: Eq(a, Z1))
    S['Z.dvd_one'] = F([Z, lambda a: dvd(a, Z1)], lambda a, h: Eq(a, Z1) | Eq(a, zneg(Z1)))
    S['Z.le_of_dvd'] = F([Z, Z, lambda a, b: zlt(Z0, b), lambda a, b, h: dvd(a, b)], lambda a, b, h, d: zle(a, b))
    S['Z.dvd_antisymm'] = F([Z, Z, lambda a, b: zle(Z0, a), lambda a, b, h: zle(Z0, b), lambda a, b, h, h2: dvd(a, b), lambda a, b, h, h2, h3: dvd(b, a)], lambda a, b, h1, h2, h3, h4: Eq(a, b))
    # ---- division with remainder
    S['Z.exists_divmod'] = F([Z, lambda b: zlt(Z0, b), Z], lambda b, hb, a: Ex(Z, w.lam(Z, lambda q: Ex(Z, w.lam(Z, lambda r: Eq(a, zadd(zmul(b, q), r)) & (zle(Z0, r) & zlt(r, b)))))))
    S['Z.divmod_unique'] = F([Z, Z, Z, Z, Z, Z, lambda a, b, q1, r1, q2, r2: zlt(Z0, b),
                              lambda a, b, q1, r1, q2, r2, hb: Eq(a, zadd(zmul(b, q1), r1)) & (zle(Z0, r1) & zlt(r1, b)),
                              lambda a, b, q1, r1, q2, r2, hb, h1: Eq(a, zadd(zmul(b, q2), r2)) & (zle(Z0, r2) & zlt(r2, b))],
                             lambda a, b, q1, r1, q2, r2, hb, h1, h2: Eq(q1, q2) & Eq(r1, r2))
    # ---- classical logic
    S['by_contra'] = F([PROP, lambda P: ~~P], lambda P, h: P)
    S['not_forall'] = F([TYPE, lambda A: w.arrow(A, PROP), lambda A, P: ~F([A], lambda x: P(x))], lambda A, P, h: Ex(A, w.lam(A, lambda x: ~P(x))))
    S['not_exists'] = F([TYPE, lambda A: w.arrow(A, PROP), lambda A, P: ~Ex(A, w.lam(A, lambda x: P(x)))], lambda A, P, h: F([A], lambda x: ~P(x)))
    S['not_imp'] = F([PROP, PROP, lambda P, Q: ~(P >> Q)], lambda P, Q, h: P & ~Q)
    S['not_or'] = F([PROP, PROP, lambda P, Q: ~(P | Q)], lambda P, Q, h: ~P & ~Q)
    # ---- Bezout, gcd, coprime
    def bez(a, b):
        return Ex(Z, w.lam(Z, lambda d: Ex(Z, w.lam(Z, lambda x: Ex(Z, w.lam(Z, lambda y: zle(Z0, d) & (dvd(d, a) & (dvd(d, b) & Eq(d, zadd(zmul(a, x), zmul(b, y)))))))))))
    S['Z.bezout_aux'] = F([Z, lambda b: zle(Z0, b), Z], lambda b, hb, a: bez(a, b))
    S['Z.bezout'] = F([Z, Z], lambda a, b: bez(a, b))
    S['Z.exists_gcd'] = F([Z, Z], lambda a, b: Ex(Z, w.lam(Z, lambda d: gcdr(a, b, d))))
    S['Z.is_gcd_unique'] = F([Z, Z, Z, Z, lambda a, b, d, d2: gcdr(a, b, d), lambda a, b, d, d2, h1: gcdr(a, b, d2)], lambda a, b, d, d2, h1, h2: Eq(d, d2))
    S['Z.coprime_symm'] = F([Z, Z, lambda a, b: cop(a, b)], lambda a, b, h: cop(b, a))
    S['Z.coprime_dvd_mul'] = F([Z, Z, Z, lambda a, b, c: cop(a, b), lambda a, b, c, h: dvd(a, zmul(b, c))], lambda a, b, c, h1, h2: dvd(a, c))
    S['Z.coprime_mul_right'] = F([Z, Z, Z, lambda a, b, c: cop(a, b), lambda a, b, c, h: cop(a, c)], lambda a, b, c, h1, h2: cop(a, zmul(b, c)))
    S['Z.is_gcd_one_of_coprime'] = F([Z, Z, lambda a, b: cop(a, b)], lambda a, b, h: gcdr(a, b, Z1))
    S['Z.coprime_of_is_gcd_one'] = F([Z, Z, lambda a, b: gcdr(a, b, Z1)], lambda a, b, h: cop(a, b))
    # ---- primes
    S['Z.prime_dvd_or_coprime'] = F([Z, lambda p: prime(p), Z], lambda p, hp, a: dvd(p, a) | cop(p, a))
    S['Z.prime_dvd_mul'] = F([Z, lambda p: prime(p), Z, Z, lambda p, hp, a, b: dvd(p, zmul(a, b))], lambda p, hp, a, b, h: dvd(p, a) | dvd(p, b))
    S['Z.prime_dvd_sq'] = F([Z, lambda p: prime(p), Z, lambda p, hp, a: dvd(p, zmul(a, a))], lambda p, hp, a, h: dvd(p, a))
    S['Z.prime_two'] = prime(Z2)
    S['Z.exists_prime_dvd'] = F([Z, lambda n: zlt(Z1, n)], lambda n, h: Ex(Z, w.lam(Z, lambda p: prime(p) & dvd(p, n))))
    # ---- the problems
    S['imo1959_p1'] = F([Z, lambda n: zle(Z0, n)], lambda n, h: gcdr(zadd(zmul(Z21, n), Z4), zadd(zmul(Z14, n), Z3), Z1))
    S['Z.sqrt2_aux'] = F([Z, lambda q: zle(Z0, q)], lambda q, h: zlt(Z0, q) >> F([Z], lambda p: ~Eq(zmul(p, p), zmul(Z2, zmul(q, q)))))
    S['sqrt2_irrational'] = F([Z, Z, lambda p, q: ~Eq(q, Z0)], lambda p, q, h: ~Eq(zmul(p, p), zmul(Z2, zmul(q, q))))
    fT = w.arrow(Z, Z)
    cond = lambda f: F([Z, Z], lambda a, b: Eq(zadd(f(zmul(Z2, a)), zmul(Z2, f(b))), f(f(zadd(a, b)))))
    sol = lambda f: F([Z], lambda n: Eq(f(n), Z0)) | Ex(Z, w.lam(Z, lambda c: F([Z], lambda n: Eq(f(n), zadd(zmul(Z2, n), c)))))
    S['imo2019_p1_fwd'] = F([fT, lambda f: cond(f)], lambda f, H: sol(f))
    S['imo2019_p1_bwd'] = F([fT, lambda f: sol(f)], lambda f, h: cond(f))
    S['imo2019_p1'] = F([fT], lambda f: w['Iff'](cond(f), sol(f)))
    return S, cond, sol


def definition_types(w):
    PROP = w.PROP; F = w.forall_; Z = w['Z']
    return {'Z.dvd': F([Z, Z], lambda a, b: PROP), 'Z.coprime': F([Z, Z], lambda a, b: PROP), 'Z.is_gcd': F([Z, Z, Z], lambda a, b, d: PROP), 'Z.prime': F([Z], lambda p: PROP)}


# theorems that rest on the axiom `em` (and the lemmas that use them): everything else must be constructive
EM_USERS = {'by_contra', 'not_forall', 'not_imp', 'Z.exists_prime_dvd'}
# the theorems that are NOT part of the statement tables because the tables cannot state them (the axiom itself)
AXIOMS = {'em'}


# ---------------------------------------------------------------------------------------------------------------------- polynomial oracle and expression builder (section C)
NV = 3                           # the variables x, y, z of the random identities
def gen_expr(rng, depth, zring):
    """a random expression: ('v', i) | ('c', n) | ('+', a, b) | ('*', a, b) | ('^', a, e) and, over the integers only, ('-', a, b) | ('~', a)"""
    if depth == 0 or rng.random() < 0.25:
        if rng.random() < 0.65: return ('v', rng.randrange(NV))
        return ('c', rng.randrange(0, 5))
    r = rng.random()
    if r < 0.40: return ('+', gen_expr(rng, depth - 1, zring), gen_expr(rng, depth - 1, zring))
    if r < 0.75: return ('*', gen_expr(rng, depth - 1, zring), gen_expr(rng, depth - 1, zring))
    if r < 0.85: return ('^', gen_expr(rng, depth - 1, zring), rng.randrange(0, 3))
    if zring:
        if r < 0.94: return ('-', gen_expr(rng, depth - 1, zring), gen_expr(rng, depth - 1, zring))
        return ('~', gen_expr(rng, depth - 1, zring))
    return ('+', gen_expr(rng, depth - 1, zring), gen_expr(rng, depth - 1, zring))
def p_add(a, b):
    r = dict(a)
    for m, c in b.items(): r[m] = r.get(m, 0) + c
    return {m: c for m, c in r.items() if c}
def p_mul(a, b):
    r = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            m = tuple(x + y for x, y in zip(m1, m2)); r[m] = r.get(m, 0) + c1 * c2
    return {m: c for m, c in r.items() if c}
def p_neg(a): return {m: -c for m, c in a.items()}
def p_pow(a, n):
    r = {tuple([0] * NV): 1}
    for _ in range(n): r = p_mul(r, a)
    return r
def poly_of(e):
    """the oracle: the expression as a polynomial {exponent vector: integer coefficient} (written here, independently of ring.py)"""
    k = e[0]
    if k == 'v': v = [0] * NV; v[e[1]] = 1; return {tuple(v): 1}
    if k == 'c': return {tuple([0] * NV): e[1]} if e[1] else {}
    if k == '+': return p_add(poly_of(e[1]), poly_of(e[2]))
    if k == '*': return p_mul(poly_of(e[1]), poly_of(e[2]))
    if k == '^': return p_pow(poly_of(e[1]), e[2])
    if k == '-': return p_add(poly_of(e[1]), p_neg(poly_of(e[2])))
    if k == '~': return p_neg(poly_of(e[1]))
    raise ValueError(k)
def build_expr(w, e, xs):
    k = e[0]
    if k == 'v': return xs[e[1]]
    if k == 'c': return w.nota.lit(e[1])
    if k == '^': return build_expr(w, e[1], xs) ** e[2]
    if k == '~': return -build_expr(w, e[1], xs)
    a, b = build_expr(w, e[1], xs), build_expr(w, e[2], xs)
    return a + b if k == '+' else a * b if k == '*' else a - b
def lc_statement(w, T, hyps, goal):
    """forall x y z : T, (l1 = r1) -> ... -> lhs = rhs"""
    Eq = w.Eq
    binders = [T, T, T] + [(lambda l, r: lambda *a: Eq(build_expr(w, l, a[:NV]), build_expr(w, r, a[:NV])))(l, r) for l, r in hyps]
    return w.forall_(binders, lambda *a: Eq(build_expr(w, goal[0], a[:NV]), build_expr(w, goal[1], a[:NV])))
def lc_planted(rng, zring, nh):
    """equations l_i = r_i, coefficients c_i, and a goal lhs = rhs with lhs - rhs = sum c_i (l_i - r_i): lhs = G + sum c_i l_i, rhs = G + sum c_i r_i (no subtraction, so it is a statement about semirings too)"""
    hyps = [(gen_expr(rng, 2, zring), gen_expr(rng, 2, zring)) for _ in range(nh)]
    cs = [gen_expr(rng, 2, zring) for _ in range(nh)]
    G = gen_expr(rng, 2, zring); lhs, rhs = G, G
    for (l, r), c in zip(hyps, cs): lhs = ('+', lhs, ('*', c, l)); rhs = ('+', rhs, ('*', c, r))
    return hyps, cs, (lhs, rhs)
def lc_holds(hyps, cs, uses, goal):
    """the oracle: goal.lhs - goal.rhs = sum c_i (l_{u_i} - r_{u_i}) as polynomials (uses[i] is the hypothesis the i-th coefficient multiplies)"""
    d = p_add(poly_of(goal[0]), p_neg(poly_of(goal[1]))); acc = {}
    for c, u in zip(cs, uses):
        l, r = hyps[u]; acc = p_add(acc, p_mul(poly_of(c), p_add(poly_of(l), p_neg(poly_of(r)))))
    return d == acc

# ---------------------------------------------------------------------------------------------------------------------- numeral tools (sections B and D)
class Num:
    """integer numerals and the proof scripts of sections B and D: every script only uses the library's lemmas, and `run` hands the finished term to the kernel"""
    def __init__(self, w):
        self.w = w; self.k = w.k
        self.Z = w['Z']; self.zle, self.zlt = w['Z.le'], w['Z.lt']; self.Z0, self.Z1 = w['Z.zero'], w['Z.one']; self.Eq = w.Eq; self.F = w.forall_
        self.zadd, self.zmul, self.zneg, self.zof = w['Z.add'], w['Z.mul'], w['Z.neg'], w['Z.of_N']
        self.dvd, self.cop, self.gcdr, self.prime = w['Z.dvd'], w['Z.coprime'], w['Z.is_gcd'], w['Z.prime']
    def numeral(self, v):
        w = self.w
        if v == 0: return self.Z0
        p = w['XH']
        for bit in bin(v)[3:]: p = (w['XI'] if bit == '1' else w['XO'])(p)
        return self.zof(w['Npos'](p))
    def zint(self, v): return self.numeral(v) if v >= 0 else self.zneg(self.numeral(-v))
    def run(self, stmt, script):
        """True iff the script proves `stmt` and the kernel accepts the term"""
        pr = Proof(self.w, stmt, 'tmp'); script(pr)
        if not pr.done: return False
        return bool(self.k.check_open(pr.term().n, stmt.n, [])[0])
    def refused(self, stmt, script):
        """True iff the script does NOT prove `stmt`: it raises, leaves a goal open, or the kernel rejects what it built"""
        try: return not self.run(stmt, script)
        except (AuthorError, RuntimeError, ValueError, E.ITPError): return True
    @staticmethod
    def xgcd(a, b):
        if b == 0: return (a, 1, 0) if a >= 0 else (-a, -1, 0)
        g, x, y = Num.xgcd(b, a % b); return g, y, x - (a // b) * y
    # --- divisibility
    def dvd_script(self, a, b):
        """a | b by the witness b / a"""
        z = self.zint; k0 = 0 if a == 0 else b // a
        def s(p): p.exists(z(k0)); p.ring()
        return s
    def refute_dvd_script(self, a, b):
        """~ (a | b): a = 0 forces b = 0; otherwise b = |a| k contradicts the remainder b mod |a| through the uniqueness of division with remainder"""
        z, Z0, Eq, zle, zlt, zadd, zmul, dvd = self.zint, self.Z0, self.Eq, self.zle, self.zlt, self.zadd, self.zmul, self.dvd
        def s(p):
            p.intro('h')
            if a == 0:
                p.have('e', Eq(z(b), Z0), lambda q: q.exact(('Z.eq_zero_of_zero_dvd', z(b), 'h')))
                p.linarith(); return
            aa = abs(a); hname = 'h'
            if a < 0:
                p.have('h2', dvd(z(aa), z(b)), lambda q: q.exact(('Z.dvd_of_neg_dvd', z(aa), z(b), 'h'))); hname = 'h2'
            q_, r_ = divmod(b, aa)
            p.have('hpos', zlt(Z0, z(aa)), lambda q: q.linarith())
            p.cases(hname, names=['k', 'hk'])
            p.have('c1', p.T('k', lambda k_: Eq(z(b), zadd(zmul(z(aa), k_), Z0)) & (zle(Z0, Z0) & zlt(Z0, z(aa)))),
                   lambda q: (q.split(), q.linear_combination((1, 'hk')), q.split(), q.linarith(), q.linarith()))
            p.have('c2', Eq(z(b), zadd(zmul(z(aa), z(q_)), z(r_))) & (zle(Z0, z(r_)) & zlt(z(r_), z(aa))),
                   lambda q: (q.split(), q.ring(), q.split(), q.linarith(), q.linarith()))
            p.obtain(['e1', 'e2'], ('Z.divmod_unique', z(b), z(aa), p.h('k'), Z0, z(q_), z(r_), 'hpos', 'c1', 'c2'))
            p.linarith()
        return s
    # --- coprime and gcd
    def cop_script(self, a, b):
        z = self.zint; g, x, y = self.xgcd(a, b); assert g == 1
        def s(p): p.exists(z(x)); p.exists(z(y)); p.ring()
        return s
    def refute_cop_script(self, a, b):
        """~ coprime a b for gcd a b = g /= 1: the gcd-one form of coprimality makes g divide 1"""
        z, Z0, Z1, Eq, zle, dvd, gcdr = self.zint, self.Z0, self.Z1, self.Eq, self.zle, self.dvd, self.gcdr
        g = gcd(abs(a), abs(b)); assert g != 1
        def s(p):
            p.intro('h')
            p.have('hg', gcdr(z(a), z(b), Z1), lambda q: q.exact(('Z.is_gcd_one_of_coprime', z(a), z(b), 'h')))
            p.cases('hg', names=['n0', 'h1']); p.cases('h1', names=['d1', 'h2']); p.cases('h2', names=['d2', 'm'])
            p.have('ga', dvd(z(g), z(a)), self.dvd_script(g, a) if g else lambda q: (q.exists(Z0), q.ring()))
            p.have('gb', dvd(z(g), z(b)), self.dvd_script(g, b) if g else lambda q: (q.exists(Z0), q.ring()))
            p.have('g1', dvd(z(g), Z1), lambda q: q.exact(('m', z(g), 'ga', 'gb')))
            if g == 0:
                p.have('e', Eq(Z1, Z0), lambda q: q.exact(('Z.eq_zero_of_zero_dvd', Z1, 'g1')))
                p.linarith()
            else:
                p.have('l', zle(z(g), Z1), lambda q: q.exact(('Z.le_of_dvd', z(g), Z1, self.w['Z.zero_lt_one'], 'g1')))
                p.linarith()
        return s
    def is_gcd_script(self, a, b):
        """is_gcd a b g (g the gcd): the Bezout combination a x + b y = g makes every common divisor divide g"""
        z, Eq, dvd, zadd, zmul = self.zint, self.Eq, self.dvd, self.zadd, self.zmul
        g, x, y = self.xgcd(a, b)
        def s(p):
            p.split(); p.linarith()
            p.split(); p.exists(z(a // g if g else 0)); p.ring()
            p.split(); p.exists(z(b // g if g else 0)); p.ring()
            p.intro('e', 'h1', 'h2')
            p.have('hd', p.T('e', lambda e: dvd(e, zadd(zmul(z(a), z(x)), zmul(z(b), z(y))))),
                   lambda q: q.exact(('Z.dvd_add', 'e', zmul(z(a), z(x)), zmul(z(b), z(y)),
                                      ('Z.dvd_mul_of_dvd_left', 'e', z(a), z(x), 'h1'), ('Z.dvd_mul_of_dvd_left', 'e', z(b), z(y), 'h2'))))
            p.have('eg', Eq(zadd(zmul(z(a), z(x)), zmul(z(b), z(y))), z(g)), lambda q: q.ring())
            p.rw_at('hd', 'eg'); p.exact('hd')
        return s
    def refute_is_gcd_script(self, a, b, d2):
        """~ is_gcd a b d2 for d2 /= gcd a b: by uniqueness against the proved  is_gcd a b g"""
        z, Eq, gcdr = self.zint, self.Eq, self.gcdr
        g = gcd(abs(a), abs(b)); assert d2 != g
        def s(p):
            p.intro('h')
            p.have('hg', gcdr(z(a), z(b), z(g)), self.is_gcd_script(a, b))
            p.have('e', Eq(z(g), z(d2)), lambda q: q.exact(('Z.is_gcd_unique', z(a), z(b), z(g), z(d2), 'hg', 'h')))
            p.linarith()
        return s
    # --- primes
    def prime_script(self, pp):
        """prime pp: 1 < pp, and every positive divisor d <= pp is 1, pp, or (case by case, by `interval_cases`) a number that does not divide pp"""
        z, Z0, zle, zlt, dvd = self.zint, self.Z0, self.zle, self.zlt, self.dvd
        def s(p):
            p.split(); p.linarith()
            p.intro('d', 'hd', 'hdv')
            p.have('hpp', zlt(Z0, z(pp)), lambda q: q.linarith())
            p.have('hle', p.T('d', lambda d: zle(d, z(pp))), lambda q: q.exact(('Z.le_of_dvd', 'd', z(pp), 'hpp', 'hdv')))
            p.interval_cases('d', 1, pp)
            for kk in range(1, pp + 1):
                if kk == 1: p.left(); p.linarith()
                elif kk == pp: p.right(); p.linarith()
                else:
                    p.rw_at('hdv', 'd_eq')
                    p.exfalso()
                    p.have('hn', ~dvd(z(kk), z(pp)), self.refute_dvd_script(kk, pp))
                    p.exact(('hn', 'hdv'))
        return s
    def composite_script(self, c, a):
        """~ prime c for c = a * (c / a) with 1 < a < c: a is a positive divisor that is neither 1 nor c"""
        z, zlt, Z0, dvd = self.zint, self.zlt, self.Z0, self.dvd
        def s(p):
            p.intro('h'); p.cases('h', names=['h1', 'h2'])
            p.have('hd', dvd(z(a), z(c)), self.dvd_script(a, c))
            p.have('hpos', zlt(Z0, z(a)), lambda q: q.linarith())
            p.obtain(['e', 'e'], ('h2', z(a), 'hpos', 'hd'))
            p.linarith()
            p.linarith()
        return s
    def small_nonprime_script(self):
        """~ prime v for v <= 1: the first conjunct 1 < v is false"""
        def s(p):
            p.intro('h'); p.cases('h', names=['h1', 'h2'])
            p.linarith()
        return s


def isprime(n): return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (5 << 30, 5 << 30))
    t_start = time.time(); rng = random.Random(34)
    t0 = time.time(); w_int = int_lib.build_int(); k = w_int.k; n_int = k.num_globals(); t_int = time.time() - t0
    t0 = time.time(); w = nt_lib.build_nt(w_int); n_nt = k.num_globals(); t_nt = time.time() - t0
    t0 = time.time(); ext_lib.build_ext(w); n_ext = k.num_globals(); t_ext = time.time() - t0                  # (the library extension: its lemmas have no row in this gate's table; S40, S41, S43, S50 replay them)
    t0 = time.time(); imo_lib.build_imo(w, ext=False); n_all = k.num_globals(); t_imo = time.time() - t0
    trace('built: int %.1f s (%d globals), nt %.1f s (+%d), ext %.1f s (+%d), imo %.1f s (+%d)' % (t_int, n_int, t_nt, n_nt - n_int, t_ext, n_ext - n_nt, t_imo, n_all - n_ext))
    TYPE, PROP = w.TYPE, w.PROP; Eq = w.Eq; F = w.forall_; Ex = w['Ex']
    N, Z = w['N'], w['Z']
    zadd, zmul, zneg, zsub, zof = (w[n] for n in ('Z.add', 'Z.mul', 'Z.neg', 'Z.sub', 'Z.of_N'))
    Z0, Z1 = w['Z.zero'], w['Z.one']; zle, zlt = w['Z.le'], w['Z.lt']
    dvd, cop, gcdr, prime = w['Z.dvd'], w['Z.coprime'], w['Z.is_gcd'], w['Z.prime']

    # ------------------------------------------------------------------------------------------------------------------ A. statements
    spec, cond, sol = ntspec(w)
    for name, stmt in spec.items():
        if expect(name in w, 'A  the library declares %s' % name):
            expect(w.global_type(name) == stmt, 'A  %s has the intended statement' % name)
            expect(k.global_info(w.gid[name])[2] is not None, 'A  %s is a theorem (it has a value)' % name)
    for name, ty in definition_types(w).items():
        expect(w.global_type(name) == ty, 'A  %s has the intended type' % name)
    # nothing added by the number-theory part escapes the table
    known = set(spec) | set(definition_types(w)) | AXIOMS
    unknown = [k.global_info(g)[0] for g in list(range(n_int, n_nt)) + list(range(n_ext, n_all)) if k.global_info(g)[0] not in known and not k.global_info(g)[0].startswith('__same_ctor_')]       # (not the library extension in between)
    expect(not unknown, 'A  the number-theory part declares things the table does not cover: %s' % unknown[:8])
    # the definitions unfold as they should
    def numeral(v):
        if v == 0: return Z0
        p = w['XH']
        for bit in bin(v)[3:]: p = (w['XI'] if bit == '1' else w['XO'])(p)
        r = zof(w['Npos'](p))
        return r
    def zint(v): return numeral(v) if v >= 0 else zneg(numeral(-v))
    k3, k6 = zint(3), zint(6)
    conv_yes = [
        ('3 | 6 is  exists k, 6 = 3 * k', dvd(k3, k6), Ex(Z, w.lam(Z, lambda k_: Eq(k6, zmul(k3, k_))))),
        ('coprime 3 4 is  exists x y, 3 x + 4 y = 1', cop(k3, zint(4)), Ex(Z, w.lam(Z, lambda x: Ex(Z, w.lam(Z, lambda y: Eq(zadd(zmul(k3, x), zmul(zint(4), y)), Z1)))))),
        ('is_gcd 6 9 3 is  0 <= 3 /\\ 3 | 6 /\\ 3 | 9 /\\ every common divisor divides 3', gcdr(k6, zint(9), k3),
         zle(Z0, k3) & (dvd(k3, k6) & (dvd(k3, zint(9)) & F([Z, lambda e: dvd(e, k6), lambda e, h1: dvd(e, zint(9))], lambda e, h1, h2: dvd(e, k3))))),
        ('prime 7 is  1 < 7 /\\ every positive divisor of 7 is 1 or 7', prime(zint(7)),
         zlt(Z1, zint(7)) & F([Z, lambda d: zlt(Z0, d), lambda d, h: dvd(d, zint(7))], lambda d, h1, h2: Eq(d, Z1) | Eq(d, zint(7)))),
    ]
    conv_no = [
        ('3 | 6 is not  exists k, 3 = 6 * k  (the quotient is on the other side)', dvd(k3, k6), Ex(Z, w.lam(Z, lambda k_: Eq(k3, zmul(k6, k_))))),
        ('coprime is not  exists x y, 3 x + 4 y = 2', cop(k3, zint(4)), Ex(Z, w.lam(Z, lambda x: Ex(Z, w.lam(Z, lambda y: Eq(zadd(zmul(k3, x), zmul(zint(4), y)), zint(2))))))),
        ('is_gcd does not mention positivity of the divisor (0 <= d, not 0 < d)', gcdr(k6, zint(9), k3),
         zlt(Z0, k3) & (dvd(k3, k6) & (dvd(k3, zint(9)) & F([Z, lambda e: dvd(e, k6), lambda e, h1: dvd(e, zint(9))], lambda e, h1, h2: dvd(e, k3))))),
        ('prime does not let the divisor be 1 only', prime(zint(7)),
         zlt(Z1, zint(7)) & F([Z, lambda d: zlt(Z0, d), lambda d, h: dvd(d, zint(7))], lambda d, h1, h2: Eq(d, Z1))),
        ('prime asks 1 < p, not 0 < p', prime(zint(7)), zlt(Z0, zint(7)) & F([Z, lambda d: zlt(Z0, d), lambda d, h: dvd(d, zint(7))], lambda d, h1, h2: Eq(d, Z1) | Eq(d, zint(7)))),
    ]
    for label, a, b in conv_yes: expect(w.conv(a, b), 'A  conversion: ' + label)
    for label, a, b in conv_no: expect(not w.conv(a, b), 'A  no conversion: ' + label)
    # the axioms of the whole world: exactly `em`; and exactly the expected theorems rest on it
    axioms = sorted(k.global_info(g)[0] for g in range(k.num_globals()) if audit.kind(k, g) == 'axiom')
    expect(axioms == ['em'], 'A  the number-theory world has exactly one axiom, em (found %s)' % axioms)
    expect(w.global_type('em') == F([PROP], lambda P: P | ~P), 'A  em is excluded middle: forall P : Prop, P \\/ ~P')
    memo = {}; users = set(); quot_bad = []
    for name in spec:
        ax, qs = audit.dependencies_of_global(k, w.gid[name], memo)
        if ax: users.add(name)
        if not ax <= {'em'}: quot_bad.append((name, sorted(ax)))
        if not qs <= {'Quot', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound'}: quot_bad.append((name, sorted(qs)))
    expect(not quot_bad, 'A  nothing rests on an axiom other than em: %s' % quot_bad[:3])
    expect(users == EM_USERS, 'A  the theorems that rest on em are exactly %s (found %s)' % (sorted(EM_USERS), sorted(users)))
    for name in ('imo1959_p1', 'sqrt2_irrational', 'imo2019_p1', 'Z.bezout', 'Z.prime_dvd_mul', 'Z.coprime_dvd_mul'):
        ax, _ = audit.dependencies_of_global(k, w.gid[name], memo)
        expect(not ax, 'A  %s is constructive (axioms: %s)' % (name, sorted(ax)))
    trace('section A done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ B. the definitions on numerals, against Python
    t_b = time.time(); nm = Num(w); z = nm.zint
    bad = []; n_dvd = 0
    for a in range(-5, 6):
        for b in range(-12, 13):
            truth = (b == 0) if a == 0 else (b % a == 0)
            n_dvd += 1
            if truth:
                if not nm.run(nm.dvd(z(a), z(b)), nm.dvd_script(a, b)): bad.append(('a true divisibility is not proved', a, b))
                if not nm.refused(~nm.dvd(z(a), z(b)), nm.refute_dvd_script(a, b)): bad.append(('a true divisibility is refuted', a, b))
            else:
                if not nm.run(~nm.dvd(z(a), z(b)), nm.refute_dvd_script(a, b)): bad.append(('a false divisibility is not refuted', a, b))
                if not nm.refused(nm.dvd(z(a), z(b)), nm.dvd_script(a, b)): bad.append(('a false divisibility is proved', a, b))
    expect(not bad, 'B  divisibility of %d pairs of numerals is proved or refuted exactly as Python says, and the script for the other verdict is refused every time: %s' % (n_dvd, bad[:4]))
    trace('B dvd done at %.1f s' % (time.time() - t_b))
    bad = []; n_cg = 0; n_cop = n_notcop = 0
    pairs = [(a, b) for a in range(-6, 7) for b in range(-6, 7)]
    for a, b in pairs:
        g = gcd(abs(a), abs(b)); n_cg += 1
        if g == 1:
            n_cop += 1
            if not nm.run(nm.cop(z(a), z(b)), nm.cop_script(a, b)): bad.append(('coprime pair not proved', a, b))
        else:
            n_notcop += 1
            if not nm.run(~nm.cop(z(a), z(b)), nm.refute_cop_script(a, b)): bad.append(('a pair with a common factor is not refuted', a, b))
            gg, x, y = nm.xgcd(a, b)           # the Bezout coefficients of the extended Euclid combine to gcd /= 1: they cannot witness coprimality
            if not nm.refused(nm.cop(z(a), z(b)), lambda p, x=x, y=y: (p.exists(z(x)), p.exists(z(y)), p.ring())): bad.append(('a pair with a common factor is proved coprime', a, b))
        if not nm.run(nm.gcdr(z(a), z(b), z(g)), nm.is_gcd_script(a, b)): bad.append(('the gcd is not proved', a, b, g))
    expect(not bad, 'B  coprimality (%d pairs coprime, %d not) and the gcd of all %d pairs in [-6, 6]^2: proved/refuted as Python says, through Bezout coefficients and the library lemmas: %s' % (n_cop, n_notcop, n_cg, bad[:4]))
    bad = []; n_wrong = 0
    for a, b in rng.sample(pairs, 50):
        g = gcd(abs(a), abs(b)); d2 = rng.choice([d for d in range(0, 9) if d != g]); n_wrong += 1
        if not nm.run(~nm.gcdr(z(a), z(b), z(d2)), nm.refute_is_gcd_script(a, b, d2)): bad.append(('a wrong gcd is not refuted', a, b, d2))
        if not nm.refused(nm.gcdr(z(a), z(b), z(d2)), nm.is_gcd_script(a, b)): bad.append(('the script for the true gcd proves a wrong one', a, b, d2))      # (is_gcd_script ignores d2: it proves is_gcd a b g, the kernel must reject it against d2)
    expect(not bad, 'B  a wrong gcd is refuted through the uniqueness of the gcd (%d pairs), and the proof of the true one is rejected for it: %s' % (n_wrong, bad[:4]))
    trace('B coprime/gcd done at %.1f s' % (time.time() - t_b))
    bad = []; n_pr = n_comp = 0
    for v in range(2, 41):
        if isprime(v):
            if v > 23: continue                                 # (the proofs for the larger primes are longer than this gate needs; the composites below reach 40)
            n_pr += 1
            if not nm.run(nm.prime(z(v)), nm.prime_script(v)): bad.append(('a prime is not proved', v))
        else:
            n_comp += 1; a = min(d for d in range(2, v) if v % d == 0)
            if not nm.run(~nm.prime(z(v)), nm.composite_script(v, a)): bad.append(('a composite is not refuted', v))
    for v in (0, 1, -2, -3, -7):
        if not nm.run(~nm.prime(z(v)), nm.small_nonprime_script()): bad.append(('a number <= 1 is not refuted', v))
    for v in (4, 9, 15, 21):                                    # the proof of primality is refused for a composite, the refutation for a prime
        if not nm.refused(nm.prime(z(v)), nm.prime_script(v)): bad.append(('a composite is proved prime', v))
    for v in (2, 3, 5, 7, 11, 13):
        if not nm.refused(~nm.prime(z(v)), nm.composite_script(v, 2 if v > 2 else 1)): bad.append(('a prime is refuted', v))
    expect(not bad, 'B  the primes up to 23 (%d) are proved by interval_cases and the library lemmas, the composites up to 40 (%d) and the numbers <= 1 are refuted; the wrong scripts are refused: %s' % (n_pr, n_comp, bad[:4]))
    tb_end = time.time() - t_b
    trace('section B done at %.1f s (B itself %.1f s)' % (time.time() - t_start, tb_end))

    # ------------------------------------------------------------------------------------------------------------------ C. linear_combination, interval_cases, by_contra, `ring` after the display normal form
    t_c = time.time(); Nn, Nat = w['N'], w['Nat']
    def lc_attempt(T, hyps, cs, uses, goal):
        """'ok' when linear_combination closes the goal and the kernel accepts the term, 'refused' when the tactic gives up (with its report of the two normal forms), else a description of what went wrong"""
        stmt = lc_statement(w, T, hyps, goal)
        try:
            pr = Proof(w, stmt, 'tmp')
            pr.intro('x', 'y', 'z', *['h%d' % (i + 1) for i in range(len(hyps))])
            pr.linear_combination(*[((lambda c: lambda q: build_expr(w, c, [q.h('x'), q.h('y'), q.h('z')]))(c), 'h%d' % (u + 1)) for c, u in zip(cs, uses)])
            if not pr.done: return 'goal left open'
            return 'ok' if bool(k.check_open(pr.term().n, stmt.n, [])[0]) else 'the kernel rejects the term'
        except AuthorError as e:
            msg = str(e)
            return 'refused' if msg.startswith('linear_combination failed') and 'left:' in msg and 'right:' in msg else 'error: ' + msg.split('\n')[0][:100]
    n_cases = int(os.environ.get('S34_LC_CASES', '45'))
    lc_stats = {}
    for tname, zring in (('Z', True), ('N', False), ('Nat', False)):
        T = w[tname]; st = {'accepted': 0, 'refused': 0, 'bad': []}
        with w.nota.using(T):
            for case in range(n_cases):
                nh = 1 + case % 3
                hyps, cs, goal = lc_planted(rng, zring, nh); uses = list(range(nh))
                variants = [(cs, uses, goal)]
                cs2 = list(cs); i = rng.randrange(nh); cs2[i] = ('+', cs2[i], ('c', 1)) if rng.random() < 0.7 else ('*', cs2[i], ('v', rng.randrange(NV)))
                variants.append((cs2, uses, goal))                                          # a perturbed coefficient
                if nh >= 2: variants.append((cs, uses[1:] + uses[:1], goal))                  # the hypotheses permuted
                variants.append((cs, uses, (goal[0], ('+', goal[1], ('c', 1)))))             # a goal that is off by one
                for cs_, uses_, goal_ in variants:
                    truth = lc_holds(hyps, cs_, uses_, goal_)
                    r = lc_attempt(T, hyps, cs_, uses_, goal_)
                    if r == 'ok' and truth: st['accepted'] += 1
                    elif r == 'refused' and not truth: st['refused'] += 1
                    else: st['bad'].append((case, r, truth, [poly_of(c) for c in cs_][:1]))
        lc_stats[tname] = st
        expect(not st['bad'], 'C  linear_combination over %s: accepted exactly when the independent polynomial oracle says lhs - rhs = sum c_i (l_i - r_i) (and every accepted proof passes the kernel): %s' % (tname, st['bad'][:3]))
        expect(st['accepted'] >= n_cases and st['refused'] >= 2 * n_cases, 'C  the %s test is not vacuous (accepted %d, refused %d)' % (tname, st['accepted'], st['refused']))
    trace('C random linear_combination done at %.1f s: %s' % (time.time() - t_c, {a: (b['accepted'], b['refused']) for a, b in lc_stats.items()}))

    def err_of(stmt, script):
        """the AuthorError message the script raises, or None when it does not raise"""
        try:
            pr = Proof(w, stmt, 'tmp'); script(pr)
        except AuthorError as e: return str(e)
        return None
    Z_ = w['Z']; zle = w['Z.le']
    with w.nota.using(Z_):
        st_rev = F([Z_, Z_, lambda a, b: Eq(a, b)], lambda a, b, h: Eq(b, a))
        expect(nm.run(st_rev, lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((-1, 'h')))), 'C  linear_combination: an integer coefficient of either sign (b = a from a = b with coefficient -1)')
        expect(nm.refused(st_rev, lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((1, 'h')))), 'C  linear_combination: the wrong sign is refused')
        st_sq = F([Z_, Z_], lambda a, b: Eq((a + b) ** 2, a * a + 2 * a * b + b * b))
        expect(nm.run(st_sq, lambda p: (p.intro('a', 'b'), p.linear_combination())), 'C  linear_combination with no hypotheses is `ring`')
        st_sq_bad = F([Z_, Z_], lambda a, b: Eq((a + b) ** 2, a * a + a * b + b * b))
        expect(nm.refused(st_sq_bad, lambda p: (p.intro('a', 'b'), p.linear_combination())), 'C  ... and refuses a false identity')
        m = err_of(F([Z_, Nn, lambda a, n: Eq(n, n)], lambda a, n, h: Eq(a, a)), lambda p: (p.intro('a', 'n', 'h'), p.linear_combination((1, 'h'))))
        expect(m is not None and 'is an equation in' in m, 'C  linear_combination: a hypothesis about another type is refused with a report (%s)' % (m or '')[:60])
        m = err_of(F([Z_, Z_, lambda a, b: zle(a, b)], lambda a, b, h: Eq(a, a)), lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((1, 'h'))))
        expect(m is not None and 'not an equation' in m, 'C  linear_combination: a hypothesis that is not an equation is refused (%s)' % (m or '')[:60])
        m = err_of(F([Z_, Z_, lambda a, b: Eq(a, b)], lambda a, b, h: zle(a, b)), lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((1, 'h'))))
        expect(m is not None and 'not an equation' in m, 'C  linear_combination: a goal that is not an equation is refused (%s)' % (m or '')[:60])
        st_same = F([Z_, Z_, lambda a, b: Eq(a, b)], lambda a, b, h: Eq(a, b))
        expect(nm.run(st_same, lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((1, 'h')))), 'C  linear_combination: a = b from h : a = b with the coefficient 1')
        m = err_of(st_same, lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((True, 'h'))))
        expect(m is not None, 'C  linear_combination: a Python bool is not a coefficient (True would be the numeral 1, which proves this statement)')
        m = err_of(st_rev, lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((1, 'nope'))))
        expect(m is not None, 'C  linear_combination: an unknown hypothesis is refused')
        # a proof term as the hypothesis, and a coefficient that is a term
        st_pf = F([Z_, Z_, lambda a, b: Eq(a, b + 1)], lambda a, b, h: Eq(a + a, 2 * b + 2))
        expect(nm.run(st_pf, lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((p.T('a', lambda a: w.nota.lit(2)), 'h')))), 'C  linear_combination: a coefficient given as a term')
    for tname in ('N', 'Nat'):
        T = w[tname]
        with w.nota.using(T):
            stn = F([T, T, lambda a, b: Eq(a, b)], lambda a, b, h: Eq(a + a, b + b))
            expect(nm.run(stn, lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((2, 'h')))), 'C  linear_combination over %s: a numeral coefficient' % tname)
            m = err_of(stn, lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((-2, 'h'))))
            expect(m is not None and 'negative' in m, 'C  linear_combination over %s: a negative coefficient is an error, not a numeral (%s)' % (tname, (m or '')[:50]))
    import ring as R
    for tname in ('N', 'Nat'):
        inst = R.instance_for(w, w[tname])
        try: inst.lit(-3); neg_ok = False
        except AuthorError: neg_ok = True
        expect(neg_ok, 'C  the numerals of %s refuse a negative value (they used to build the numeral of 7 / recurse for ever)' % tname)

    # `ring`, `linarith` and `nlinarith` still read a goal after the kernel's display normal form (every tactic applies it): x^2 became `Z.sqr x`, and `1 ^ 2` of N a numeral `Npos (Pos.sqr XH)`
    for tname, mark in (('Z', 'Z.sqr'), ('N', 'Pos.sqr'), ('Nat', None)):
        T = w[tname]
        with w.nota.using(T):
            one, two, three, four, eight = (w.nota.lit(i) for i in (1, 2, 3, 4, 8))
            ident = lambda c: F([T, T], lambda x, y: Eq((x + one) ** 3 + x ** 2 + one ** 2 + (x * y) ** 2 + two ** 3 * x ** 6 + (one + one) ** 2 + c,
                                                      x ** 3 + three * x ** 2 + three * x + one + x * x + one + x * x * y * y + eight * (x ** 3) ** 2 + four))
            script = lambda p: (p.intro('x', 'y'), p.norm(), p.ring())
            pr = Proof(w, ident(w.nota.lit(0)), 'tmp'); pr.intro('x', 'y'); pr.norm()
            shown = w.show(pr.goal().target)
            expect(mark is None or mark in shown, 'C  the display normal form really produces %s in the %s goal (the test below is not vacuous)' % (mark, tname))
            expect(nm.run(ident(w.nota.lit(0)), script), 'C  ring closes the %s identity after the display normal form' % tname)
            expect(nm.refused(ident(one), script), 'C  ring still refuses the false %s identity after the display normal form' % tname)
    Z_ = w['Z']; zlt = w['Z.lt']; Z0 = w['Z.zero']
    with w.nota.using(Z_):
        one, two = w.nota.lit(1), w.nota.lit(2)
        st1 = F([Z_], lambda x: zlt(Z0, x ** 2 + one))
        expect(nm.run(st1, lambda p: (p.intro('x'), p.norm(), p.nlinarith())), 'C  nlinarith: 0 < x^2 + 1 after the display normal form (the square Z.sqr x is a square)')
        expect(nm.refused(st1, lambda p: (p.intro('x'), p.norm(), p.linarith())), 'C  ... and linarith alone cannot')
        st2 = F([Z_, lambda x: zle(Z0, x)], lambda x, h: zle(x ** 2 + two * x + one, (x + one) ** 2))
        expect(nm.run(st2, lambda p: (p.intro('x', 'h'), p.norm(), p.linarith())), 'C  linarith: x^2 + 2x + 1 <= (x+1)^2 after the display normal form (both sides read as the same polynomial)')
        st3 = F([Z_, Z_], lambda a, b: zle(two * a * b, a ** 2 + b ** 2))
        expect(nm.run(st3, lambda p: (p.intro('a', 'b'), p.norm(), p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - q.h('b'))))), 'C  nlinarith: 2ab <= a^2 + b^2 from the square of a - b, after the display normal form')
        st3_bad = F([Z_, Z_], lambda a, b: zle(w.nota.lit(3) * a * b, a ** 2 + b ** 2))
        expect(nm.refused(st3_bad, lambda p: (p.intro('a', 'b'), p.norm(), p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - q.h('b'))))), 'C  ... and 3ab <= a^2 + b^2 (false for a = b = 1) is refused')

    # interval_cases: one goal per value, in increasing order, each with the hypothesis  d = value
    with w.nota.using(Z_):
        lit = w.nota.lit
        st_ic = F([Z_, lambda d: zle(lit(2), d), lambda d, h1: zle(d, lit(5))], lambda d, h1, h2: Eq(d, lit(2)) | (Eq(d, lit(3)) | (Eq(d, lit(4)) | Eq(d, lit(5)))))
        pr = Proof(w, st_ic, 'ic'); pr.intro('d', 'h1', 'h2'); pr.interval_cases('d', 2, 5, name='hd')
        expect(pr.n_goals == 4, 'C  interval_cases 2..5 makes four goals (got %d)' % pr.n_goals)
        for i in range(4):
            for _ in range(i): pr.right()
            if i < 3: pr.left()
            pr.exact('hd')
        expect(pr.done and bool(k.check_open(pr.term().n, st_ic.n, [])[0]), 'C  the four goals come out in increasing order, each with its own  d = value  (each is closed by the matching disjunct)')
        m = err_of(st_ic, lambda p: (p.intro('d', 'h1', 'h2'), p.interval_cases('d', 2, 4)))
        expect(m is not None and 'linarith' in m, 'C  interval_cases: a bound that does not follow from the hypotheses is refused (%s)' % (m or '')[:60])
        m = err_of(st_ic, lambda p: (p.intro('d', 'h1', 'h2'), p.interval_cases('d', 3, 2)))
        expect(m is not None and 'empty range' in m, 'C  interval_cases: an empty range is refused')
        m = err_of(st_ic, lambda p: (p.intro('d', 'h1', 'h2'), p.interval_cases('d', 2, 1000)))
        expect(m is not None and 'too many' in m, 'C  interval_cases: a thousand cases are refused')
        m = err_of(st_ic, lambda p: (p.intro('d', 'h1', 'h2'), p.interval_cases('d', 2.0, 5)))
        expect(m is not None and 'Python ints' in m, 'C  interval_cases: bounds must be ints')
        m = err_of(st_ic, lambda p: (p.intro('d', 'h1', 'h2'), p.interval_cases('d', True, 5)))
        expect(m is not None and 'Python ints' in m, 'C  interval_cases: a bool is not a bound')
        m = err_of(st_ic, lambda p: (p.intro('d', 'h1', 'h2'), p.interval_cases('d', 2, False)))
        expect(m is not None and 'Python ints' in m, 'C  interval_cases: a bool is not a bound (the upper one)')
        st_n = F([Nn], lambda n: Eq(n, n))
        m = err_of(st_n, lambda p: (p.intro('n'), p.interval_cases('n', 0, 3)))
        expect(m is not None and 'not an integer' in m, 'C  interval_cases: a variable that is not an integer is refused (%s)' % (m or '')[:50])
    # by_contra: classical, from the axiom
    P0 = w.PROP
    st_bc = F([P0, lambda P: ~~P], lambda P, h: P)
    expect(nm.run(st_bc, lambda p: (p.intro('P', 'h'), p.by_contra('hn'), p.exact(('h', 'hn')))), 'C  by_contra: assume the negation of the goal and refute it')
    import lnl_base
    wb = lnl_base.build_base()
    prb = Proof(wb, wb.forall_([wb.PROP], lambda P: P), 'tmp_bc'); prb.intro('P')
    try: prb.by_contra(); m = None
    except AuthorError as e: m = str(e)
    expect(m is not None and 'no classical logic' in m, 'C  by_contra: a world without the lemma by_contra says so (%s)' % (m or '')[:60])
    # linear_combination: the failures are reported by name, the step is recorded, and the order of the cancellation lemma's variables is read off its statement
    with w.nota.using(Z_):
        m = err_of(st_rev, lambda p: (p.intro('a', 'b', 'h'), p.linear_combination((1, 'h'))))
        expect(m is not None and 'does not close the goal' in m, 'C  linear_combination: a combination that does not close the goal is reported as such (%s)' % (m or '')[:60])
        ZZ_ = w.arrow(Z_, Z_)
        m = err_of(F([ZZ_], lambda f: Eq(f, f)), lambda p: (p.intro('f'), p.linear_combination((1, 'f'))))
        expect(m is not None and 'no ring structure' in m, 'C  linear_combination: an equation between functions has no ring structure (%s)' % (m or '')[:60])
        prt = Proof(w, st_rev, 'tmp_trace'); prt.intro('a', 'b', 'h'); prt.linear_combination((-1, 'h'))
        expect(prt.done and prt.trace[-1] == ('linear_combination',), 'C  linear_combination: the step is the last entry of the trace')
        import lincomb as LCm
        pr0 = Proof(w, st_rev, 'tmp_order')
        for ty, want in ((F([Z_, Z_, Z_, lambda a, b, c: Eq(a + c, b + c)], lambda a, b, c, h: Eq(a, b)), (0, 1, 2)),
                         (F([Z_, Z_, Z_, lambda a, b, c: Eq(a + b, c + b)], lambda a, b, c, h: Eq(a, c)), (0, 2, 1)),
                         (F([Z_, Z_, Z_, lambda a, b, c: Eq(c + a, c + b)], lambda a, b, c, h: Eq(b, a)), (1, 0, 2))):
            got = LCm._cancel_order(pr0, ty, 'fake')
            expect(got == want, 'C  _cancel_order reads the positions of the two sides and of the common summand off the lemma: %s, wanted %s' % (got, want))
        for ty, why in ((F([Z_, Z_, Z_, lambda a, b, c: Eq(a + c, b + c)], lambda a, b, c, h: Eq(a, a)), 'the same variable on both sides'),
                        (F([Z_, Z_, Z_, lambda a, b, c: Eq(a + c, b + c)], lambda a, b, c, h: Eq(a + b, c)), 'a side that is not one of the variables'),
                        (F([Z_, Z_, lambda a, b: Eq(a, b)], lambda a, b, h: Eq(a, b)), 'a hypothesis where a third variable is expected')):
            try: LCm._cancel_order(pr0, ty, 'fake'); raised = False
            except AuthorError as e: raised = 'expected shape' in str(e)
            expect(raised, 'C  _cancel_order refuses a lemma with %s' % why)
    # unfold_head: one syntactic delta step
    a3, a6 = nm.zint(3), nm.zint(6)
    u = k.unfold_head(nm.dvd(a3, a6).n)
    expect(u is not None and w.head_name(w.T(u)) == 'Ex' and w.conv(w.T(u), nm.dvd(a3, a6)), 'C  unfold_head: `3 | 6` unfolds one step to an existential that is convertible with it')
    expect(k.unfold_head(Eq(a3, a6).n) is None, 'C  unfold_head: an application of an inductive family is not unfolded')
    expect(k.unfold_head(a3.n) is not None and k.unfold_head(w['Nat'].n) is None, 'C  unfold_head: a defined constant unfolds, a type constant does not')
    tc_end = time.time() - t_c
    trace('section C done at %.1f s (C itself %.1f s)' % (time.time() - t_start, tc_end))

    # ------------------------------------------------------------------------------------------------------------------ D. the problems: the statements say something, and nearby statements are refuted
    t_d = time.time(); Zt = w['Z']; zle = w['Z.le']
    with w.nota.using(Zt):
        lit = w.nota.lit
        # -- IMO 2019 P1:  f(2a) + 2 f(b) = f(f(a+b)).  The kernel refutes the equation for functions that are not solutions (at a = b = 1) and proves it for the solutions; `sol` is refuted for the same functions
        wrong_fs = {'the identity': w.lam(Zt, lambda n: n), 'the constant 1': w.lam(Zt, lambda n: lit(1)), 'n -> 3n': w.lam(Zt, lambda n: 3 * n), 'n -> n^2': w.lam(Zt, lambda n: n * n), 'n -> n + 1': w.lam(Zt, lambda n: n + 1)}
        def refute_cond(p):
            p.intro('H'); p.have('e', ('H', lit(1), lit(1))); p.linarith()
        def refute_sol(p):
            p.intro('h'); p.cases('h', names=['h1', 'h2'])
            p.have('e0', ('h1', lit(0))); p.have('e1', ('h1', lit(1))); p.linarith()
            p.cases('h2', names=['c', 'hc'])
            p.have('e0', ('hc', lit(0))); p.have('e1', ('hc', lit(1))); p.linarith()
        for label, f in wrong_fs.items():
            expect(nm.run(~cond(f), refute_cond), 'D  IMO 2019 P1: the functional equation is false for %s (refuted by the kernel at a = b = 1)' % label)
            expect(nm.run(~sol(f), refute_sol), 'D  IMO 2019 P1: %s is not in the claimed solution set' % label)
        good_fs = {'f = 0': (w.lam(Zt, lambda n: lit(0)), 'left'), 'f(n) = 2n - 3': (w.lam(Zt, lambda n: 2 * n - 3), 'right'), 'f(n) = 2n': (w.lam(Zt, lambda n: 2 * n), 'right'), 'f(n) = 2n + 5': (w.lam(Zt, lambda n: 2 * n + 5), 'right')}
        for label, (f, side) in good_fs.items():
            expect(nm.run(cond(f), lambda p: (p.intro('a', 'b'), p.norm(), p.ring())), 'D  IMO 2019 P1: the functional equation holds for %s (a ring identity)' % label)
            def in_sol(p, f=f, side=side):
                if side == 'left': p.left(); p.intro('n'); p.norm(); p.refl()
                else:
                    p.right(); c = {'f(n) = 2n - 3': -3, 'f(n) = 2n': 0, 'f(n) = 2n + 5': 5}[label]
                    p.exists(lit(c) if c >= 0 else -lit(-c)); p.intro('n'); p.norm(); p.ring()
            expect(nm.run(sol(f), in_sol), 'D  IMO 2019 P1: %s is in the claimed solution set' % label)
        # the two directions of the theorem, applied to a function whose equation and solution form were each proved directly above: the library's proof of the other one is then accepted
        f2 = w.lam(Zt, lambda n: 2 * n + 1)
        pr_sol = Proof(w, sol(f2), 'sol'); pr_sol.right(); pr_sol.exists(lit(1)); pr_sol.intro('n'); pr_sol.norm(); pr_sol.ring()
        pr_cond = Proof(w, cond(f2), 'cond'); pr_cond.intro('a', 'b'); pr_cond.norm(); pr_cond.ring()
        expect(pr_sol.done and pr_cond.done, 'D  f(n) = 2n + 1: both the equation and the solution form are proved directly')
        expect(nm.run(cond(f2), lambda p: p.exact(('imo2019_p1_bwd', f2, pr_sol.term()))), 'D  imo2019_p1_bwd turns the proof that f(n) = 2n + 1 is of the solution form into the functional equation')
        expect(nm.run(sol(f2), lambda p: p.exact(('imo2019_p1_fwd', f2, pr_cond.term()))), 'D  imo2019_p1_fwd turns the proof of the functional equation for f(n) = 2n + 1 into the solution form')
        expect(nm.refused(~cond(f2), lambda p: p.exact(('imo2019_p1_bwd', f2, pr_sol.term()))), 'D  ... and the same theorem does not refute the equation for it')
        # -- sqrt 2 irrational: p^2 = 2 q^2.  With the 2 replaced by a square the equation HAS solutions (refuted: the statement would be false), without q /= 0 it has (0, 0)
        for coef, witness in ((4, 2), (1, 1), (9, 3)):
            stmt = ~F([Zt, Zt, lambda p, q: ~Eq(q, Z0)], lambda p, q, h: ~Eq(p * p, coef * (q * q)))
            def s(p, coef=coef, witness=witness):
                p.intro('H'); p.have('hne', ~Eq(lit(1), Z0), lambda q: (q.intro('e'), q.linarith()))
                p.have('hq', ('H', lit(witness), lit(1), 'hne'))
                p.have('e', Eq(lit(witness) * lit(witness), coef * (lit(1) * lit(1))), lambda q: q.ring())
                p.exact(('hq', 'e'))
            expect(nm.run(stmt, s), 'D  p^2 = %d q^2 has the solution (%d, 1): the irrationality statement with %d for 2 is refuted' % (coef, witness, coef))
            if coef == 4:                                                    # the same script cannot refute the real statement (the witness equation is false): a ring failure
                stmt2 = ~F([Zt, Zt, lambda p, q: ~Eq(q, Z0)], lambda p, q, h: ~Eq(p * p, 2 * (q * q)))
                expect(nm.refused(stmt2, s), 'D  the same refutation does not go through for the real statement p^2 = 2 q^2')
        stmt = ~F([Zt, Zt], lambda p, q: ~Eq(p * p, 2 * (q * q)))
        def s0(p):
            p.intro('H'); p.have('hq', ('H', lit(0), lit(0))); p.have('e', Eq(lit(0) * lit(0), 2 * (lit(0) * lit(0))), lambda q: q.ring()); p.exact(('hq', 'e'))
        expect(nm.run(stmt, s0), 'D  without q /= 0 the statement is false: (p, q) = (0, 0) solves p^2 = 2 q^2')
    # -- IMO 1959 P1: gcd (21n+4, 14n+3) = 1.  The theorem gives coprimality for n = 0..12 (compared with Python's gcd), the pair (21n+4, 14n+6) is not coprime, and the library's own script (Bezout coefficients -2, 3) fails on it
    z = nm.zint; Z0 = nm.Z0
    bad = []
    for n in range(0, 13):
        a_, b_ = 21 * n + 4, 14 * n + 3
        if gcd(a_, b_) != 1: bad.append(('python', n))
        stmt = nm.cop(nm.zadd(nm.zmul(z(21), z(n)), z(4)), nm.zadd(nm.zmul(z(14), z(n)), z(3)))
        def s(p, n=n):
            p.have('hn', zle(Z0, z(n)), lambda q: q.linarith())
            p.exact(('Z.coprime_of_is_gcd_one', nm.zadd(nm.zmul(z(21), z(n)), z(4)), nm.zadd(nm.zmul(z(14), z(n)), z(3)), ('imo1959_p1', z(n), 'hn')))
        if not nm.run(stmt, s): bad.append(('theorem', n))
    expect(not bad, 'D  IMO 1959 P1: the theorem, specialised to n = 0..12, proves coprimality of (21n+4, 14n+3) -- as Python says: %s' % bad[:3])
    expect(nm.run(~nm.gcdr(z(25), z(20), z(1)), nm.refute_is_gcd_script(25, 20, 1)), 'D  IMO 1959 P1: (21n+4, 14n+6) at n = 1 is (25, 20), whose gcd is 5, not 1')
    def imo59_script(p):
        p.intro('n', 'h'); p.apply('Z.is_gcd_one_of_coprime'); p.exists(nm.zneg(z(2))); p.exists(z(3)); p.ring()
    with w.nota.using(Zt):
        st_true = F([Zt, lambda n: zle(Z0, n)], lambda n, h: nm.gcdr(21 * n + 4, 14 * n + 3, nm.Z1))
        st_pert = F([Zt, lambda n: zle(Z0, n)], lambda n, h: nm.gcdr(21 * n + 4, 14 * n + 6, nm.Z1))
        expect(nm.run(st_true, imo59_script), 'D  the library\'s proof script for IMO 1959 P1 proves the statement')
        expect(nm.refused(st_pert, imo59_script), 'D  ... and is refused (by `ring`) on the perturbed statement with 14n + 6')
    trace('section D done at %.1f s (D itself %.1f s)' % (time.time() - t_start, time.time() - t_d))

    # ------------------------------------------------------------------------------------------------------------------ E. the independent checker re-checks the number-theory part; corrupted proofs: agreement
    t_e = time.time()
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid),
                recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    chk = ic.Checker(env, params=True, fuel=200_000_000, prop_elim=True)
    bad = []
    if chk.quot_problem is not None: bad.append(('quotients: ' + chk.quot_problem,))
    bad += chk.well_founded_problems() + chk.recursion_problems()
    n_checked = 0; times_e = []
    for gid, gl in chk.g.items():
        if gid < n_int: continue                                       # (the integer part is re-checked by S30)
        t_g = time.time()
        ok, msg = chk.run(lambda: (chk.sort_of({}, gl.type), None if gl.value is None else chk.check({}, gl.value, gl.type)))
        n_checked += 1; times_e.append((time.time() - t_g, gl.name))
        if not ok: bad.append((gid, gl.name, msg))
    expect(n_checked >= 50 and not bad, 'E  the independent checker re-checks the %d globals of the number-theory part and the problems without a rejection: %s' % (n_checked, bad[:3]))
    trace('E independent re-check: %d globals in %.1f s, slowest %s' % (n_checked, time.time() - t_e, [(round(a, 1), b) for a, b in sorted(times_e, reverse=True)[:5]]))
    kinds = {'kernel_rej': 0, 'kernel_acc': 0, 'disagree': 0, 'undecided': 0}
    chk_small = ic.Checker(env, params=True, fuel=1_500_000, prop_elim=True)
    consts = [k.mk_const(w.gid[n_]) for n_ in ('Z.dvd', 'Z.dvd_trans', 'Z.dvd_add', 'Z.mul_dvd_mul', 'Z.coprime', 'Z.is_gcd', 'Z.one_dvd', 'Z.dvd_refl', 'Z.add', 'Z.mul', 'Z.neg', 'Z.zero', 'Z.one')]
    for name in ('Z.dvd_trans', 'Z.neg_dvd', 'imo1959_p1', 'Z.coprime_symm'):
        proof = w.global_value(name); stmt = w.global_type(name); stmt_h = parse_raw(k.dump_raw(stmt.n), True)
        expect(bool(k.check_open(proof.n, stmt.n, [])[0]) and chk.check_closed(parse_raw(k.dump_raw(proof.n), True), stmt_h)[0], 'E  the uncorrupted proof of %s is accepted by both checkers' % name)
        pool = [n_ for n_ in dag_nodes(k, proof.n) if k.view(n_)[0] in ('app', 'const', 'lam')]
        for case in range(14):
            target = rng.choice(pool); v = k.view(target); r = rng.random()
            if r < 0.3: repl = rng.choice(pool)
            elif r < 0.5 and v[0] == 'app' and k.view(v[1])[0] == 'app':
                inner = k.view(v[1]); repl = k.mk_app(k.mk_app(inner[1], v[2]), inner[2])          # swap the last two arguments
            elif r < 0.7: repl = rng.choice(consts)
            elif r < 0.8: repl = k.mk_const(w.gid['Z.zero'])
            elif r < 0.9: repl = k.mk_const(w.gid['Z.neg'])
            else: repl = k.mk_app(k.mk_const(w.gid['Z.neg']), target) if v[0] != 'lam' else target
            if repl == target: continue
            try: mutant = rebuild(k, proof.n, {target: repl})
            except ValueError: continue
            if mutant == proof.n: continue
            try: kok = bool(k.check_open(mutant, stmt.n, [])[0])
            except Exception: kok = False
            try: iok, imsg = chk_small.check_closed(parse_raw(k.dump_raw(mutant), True), stmt_h)
            except ic.Unsupported: continue
            if not iok and 'fuel' in imsg: kinds['undecided'] += 1; continue
            if kok != iok: kinds['disagree'] += 1; FAIL.append('E  the kernel (%s) and the independent checker (%s: %s) disagree about a corrupted proof of %s' % (kok, iok, imsg, name))
            elif kok: kinds['kernel_acc'] += 1
            else: kinds['kernel_rej'] += 1
    expect(kinds['kernel_rej'] >= 25 and kinds['disagree'] == 0, 'E  the corruption test is not vacuous and the two checkers never disagree (%s)' % kinds)
    trace('section E done at %.1f s (E itself %.1f s)' % (time.time() - t_start, time.time() - t_e))

    print('S34: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:30]: print('  FAIL:', f)
    print('S34 NT GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
