"""S39 gate: a comparison that FAILS is decided by what it found, not by normalising both sides.

Two stuck terms (a variable, a constant or an axiom with arguments and pending case splits on it) are compared frame by frame. When the walk fails there are two kinds of failure:
  * CERTAIN: it failed at an ARGUMENT -- data, compared by the very walk that decides every success -- and nothing inside it was incomplete. The terms are different; the answer is no.
  * UNCERTAIN: it failed at a case split (two derivations of one computation may capture different environments), or somewhere inside a fixpoint or past the nesting bound of case-split comparisons
    (the walk fails CLOSED there). The terms may still be equal, so both sides are written out in full (normalised) and compared as syntax: the RETRY.
Before this gate the retry ran after every failure. Over a quotient-defined number type, whose every operation on a variable unfolds into a page of lifted functions and proofs, that cost 200,000 to
1,600,000 steps for `Z.add a b` against `Z.add a a`, and a failed `apply` / `exact` / `rw` probe paid it each time: it was the whole cost of a failed probe of the search. The kernel counts both outcomes
(`neutral_stats()`: retries, refusals without a retry), so this gate asserts on what the kernel DID, and bounds the steps of what it did not have to do:

  A. refusals at data over the integers (`Z`, a quotient): `Z.add a b` against `Z.add a a`, `f (2a) + 2 f b` against `f (2a) + 0`, products, nested sums: unify_open and conv_open refuse in a few
     hundred steps (the old kernel needed 200,000 and more), with NO retry and at least one refusal; the same pairs through a chain of wrappers keep the cost linear;
  B. failures that stay UNCERTAIN keep the retry: a differing case split (`add x 2` against `add x 3`), a differing fixpoint argument (`h add` against `h mul`), two different-but-equivalent recursive
     definitions (the descent past the nesting bound, which fails closed: not convertible, a bounded number of steps); an answer found by a retry is DECIDED -- the enclosing comparisons see a
     certain failure -- so a chain of k wrappers around a case-split mismatch retries once, not k times;
  C. the retry DECIDES: with the nesting bound lowered to 0 every comparison that needs the branches of a case split fails closed and is answered by the retry alone. `add x y` against the same
     case split written inline is accepted (by the retry, once); against one with a different branch it is refused; the bound is a per-context hook that can be set, read back and restored; and
     ~500 random pairs of terms (sums, products, wrappers, opaque functions, normal forms of themselves) are decided by `unify_open` and `conv_open` exactly as full normalisation decides them --
     at the shipped bound, at 2 and at 0, in the last two with hundreds of answers coming from the retry;
  D. nothing is shared that should not be: closures with the same code and different captured values (`K x` against `K y`), types that are products over different values, equal values met through
     different environments;
  E. proofs: a stuck proof (a case split whose motive is a proposition) and an axiom's proof of the same proposition are equal for the unifier (proof irrelevance) but not for plain conversion; a
     stuck case split on data (the motive is a type) is NOT offered to proof irrelevance (cheap refusal); an argument rescued by proof irrelevance does not make the enclosing refusal uncertain;
  F. function types that reach their codomain through a beta redex (`Z.strong_induction P ...` has the codomain `P n`) are exposed on syntax: the application is inferred and checked in a few
     hundred steps (1,160,866 before, by evaluating and writing back the whole type), and a wrong motive is refused as cheaply;
  G. the counters are statistics: they never decrease, and a comparison re-derived by the self-check option (verify_memo) leaves them as a plain run does.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, sys, time

sys.setrecursionlimit(100000)
import itp_env_lnl as E
import lnl_base, int_lib, nt_lib
from author import World

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


FUEL = 200_000_000


class Res:
    """what comparing (a, b) did: the verdicts of unify_open (`ver`) and conv_open (`cver`), the steps each took, and for each the (retries, refusals) it made"""
    def __init__(self): self.ver = self.cver = None; self.usteps = self.csteps = 0; self.uret = self.uref = self.cret = self.cref = 0
    def __repr__(self): return 'Res(ver=%s cver=%s usteps=%d csteps=%d u(retries=%d refusals=%d) c(retries=%d refusals=%d))' % (self.ver, self.cver, self.usteps, self.csteps, self.uret, self.uref, self.cret, self.cref)
    @property
    def steps(self): return max(self.usteps, self.csteps)
    @property
    def retries(self): return max(self.uret, self.cret)
    @property
    def refusals(self): return min(self.uref, self.cref)


def decide(w, a, b, syn=False):
    """compare (a, b) by unify_open and by conv_open, each from the same counters; `syn`: the quick syntactic check (it only ever answers yes) on or off"""
    k = w.k; k.set_option('syn_conv', syn); r = Res()
    s0 = k.neutral_stats(); k.set_fuel(FUEL)
    r.ver = k.unify_open(w._n(a), w._n(b), w.ctx)[0]; r.usteps = k.fuel_used()
    s1 = k.neutral_stats(); r.uret, r.uref = s1[0] - s0[0], s1[1] - s0[1]
    k.set_fuel(FUEL)
    r.cver = k.conv_open(w._n(a), w._n(b), len(w.ctx)); r.csteps = k.fuel_used()
    s2 = k.neutral_stats(); r.cret, r.cref = s2[0] - s1[0], s2[1] - s1[1]
    k.set_option('syn_conv', True)
    return r


class Ctx:
    """variables of a world's context, unbound again on exit"""
    def __init__(self, w, types_and_names): self.w = w; self.tn = types_and_names
    def __enter__(self):
        self.vars = [self.w._bind(t, nm)[1] for t, nm in self.tn]; return self.vars
    def __exit__(self, *exc): self.w._unbind(len(self.tn))


def nat_vars(w): return Ctx(w, [(w['Nat'], 'x'), (w['Nat'], 'y'), (w['Nat'], 'z')])


# ---------------------------------------------------------------------------------------------------------------------------------------- the worlds
def build_nat():
    """the naturals with addition and multiplication, opaque functions, two definitions of addition that agree on every value but not by computation, a proposition by cases, and a family of axioms"""
    w = World()
    lnl_base.declare_types(w); lnl_base.declare_add(w)
    TYPE, PROP = w.TYPE, w.PROP
    Nat, S, Z0, add = w['Nat'], w['Succ'], w['Zero'], w['add']
    NN = w.arrow(Nat, Nat); NNN = w.arrow(Nat, NN)
    w.recursive('mul', w.forall_([Nat, Nat], lambda n, m: Nat), lambda mul: w.fun([Nat, Nat], lambda n, m: w.match(n, lambda _: Nat, [Z0, lambda n1: add(m, mul(n1, m))])))
    w.define('two', Nat, S(S(Z0))); w.define('three', Nat, S(S(S(Z0))))
    w.define('idn', NN, w.lam(Nat, lambda n: n))
    w.define('plus2', NN, w.lam(Nat, lambda n: add(n, w['two'])))
    w.define('K', NNN, w.fun([Nat, Nat], lambda a, b: a))                                      # constant function: `K x` is a closure that holds x
    w.recursive('addb', w.forall_([Nat, Nat], lambda n, m: Nat), lambda addb: w.fun([Nat, Nat], lambda n, m: w.match(n, lambda _: Nat, [w['idn'](m), lambda n1: S(addb(n1, m))])))
    w.axiom('f', NN); w.axiom('g', NNN)
    w.axiom('h1', w.arrow(w.arrow(Nat, NN), Nat))                                              # takes a function: its argument is a fixpoint value
    # a proposition by cases, a stuck proof of it, an axiom proving the same thing, and functions that take such a proof
    w.axiom('P', w.arrow(Nat, PROP)); P = w['P']
    w.axiom('p0', P(Z0)); w.axiom('ps', w.forall_([Nat], lambda n: P(S(n))))
    w.define('pc', w.forall_([Nat], lambda n: P(n)), w.fun([Nat], lambda n: w.match(n, lambda m: P(m), [w['p0'], lambda n1: w['ps'](n1)])))
    w.axiom('pax', w.forall_([Nat], lambda n: P(n)))
    w.axiom('qf', w.arrow(w.arrow(Nat, NN), P(Z0)))                                            # a proof of  P 0  that takes a function: two proofs of one proposition that differ in a fixpoint argument
    w.axiom('hz', w.arrow(P(Z0), NN))
    F2 = w.arrow(Nat, NN)
    w.axiom('Q', w.arrow(F2, PROP)); w.axiom('qa', w['Q'](w['add'])); w.axiom('qm', w['Q'](w['mul']))        # two proofs of `exists f, Q f`, built from two different recursive functions
    w.axiom('hx', w.arrow(w['Ex'](F2, w['Q']), NN))
    w.axiom('hq', w.forall_([Nat, lambda n: P(n)], lambda n, p: Nat))
    w.axiom('hr', w.forall_([Nat, lambda n: P(n)], lambda n, p: NN))
    # a family of types over a value:  Pk a = forall b, a = b
    w.define('Pk', w.arrow(Nat, PROP), w.lam(Nat, lambda a: w.forall_([Nat], lambda b: w.Eq(a, b))))
    return w


def build_int():
    w = int_lib.build_int(); nt_lib.build_nt(w)
    w.axiom('fz', w.arrow(w['Z'], w['Z']))
    return w


# ---------------------------------------------------------------------------------------------------------------------------------------- A
Z_BOUND = 1000          # the old kernel needed 200,000 to 1,600,000 steps for each of these (and the new one needs 20 to 2,100)


def part_A(wz):
    Zt, zadd, zmul, fz, z0, z1 = wz['Z'], wz['Z.add'], wz['Z.mul'], wz['fz'], wz['Z.zero'], wz['Z.one']
    two = zadd(z1, z1)
    with Ctx(wz, [(Zt, 'a'), (Zt, 'b'), (Zt, 'c')]) as (a, b, c):
        cases = [
            ('Z.add a b = Z.add a a', zadd(a, b), zadd(a, a), Z_BOUND),
            ('Z.add a b = Z.add b a', zadd(a, b), zadd(b, a), Z_BOUND),
            ('Z.mul a b = Z.mul a c', zmul(a, b), zmul(a, c), Z_BOUND),
            ('Z.add (Z.mul a b) c = Z.add (Z.mul a b) a', zadd(zmul(a, b), c), zadd(zmul(a, b), a), Z_BOUND),
            ('f (2a) = f (f (a + b))', fz(zmul(two, a)), fz(fz(zadd(a, b))), Z_BOUND),
            ('f (a + b) = f (a + c)', fz(zadd(a, b)), fz(zadd(a, c)), Z_BOUND),
            ('f (a + b) + c = f (a + b) + b', zadd(fz(zadd(a, b)), c), zadd(fz(zadd(a, b)), b), Z_BOUND),
            ('a * (b + c) = a * (c + b)', zmul(a, zadd(b, c)), zmul(a, zadd(c, b)), Z_BOUND),
            ('f (2a) + 2 f b = f (2a) + 0', zadd(fz(zmul(two, a)), zmul(two, fz(b))), zadd(fz(zmul(two, a)), z0), 4 * Z_BOUND),
        ]
        for name, x, y, bound in cases:
            r = decide(wz, x, y)
            expect(r.ver is False and r.cver is False, 'A: %s was accepted: %r' % (name, r))
            expect(r.steps <= bound, 'A: %s took %d steps (bound %d; 200,000 to 1,600,000 when both sides were normalised)' % (name, r.steps, bound))
            expect(r.uret == 0 and r.cret == 0, 'A: %s: a failure at data was retried on the normalised terms: %r' % (name, r))
            expect(r.uref >= 1 and r.cref >= 1, 'A: %s: no refusal was counted: %r' % (name, r))
        # through wrappers the cost stays linear, the refusal is counted at every level, and there is never a retry
        for k_ in (5, 20, 60):
            x = zadd(a, b); y = zadd(a, a)
            for _ in range(k_): x = fz(x); y = fz(y)
            r = decide(wz, x, y)
            expect(r.ver is False and r.cver is False and r.steps <= 300 + 60 * k_ and r.retries == 0 and r.refusals >= k_, 'A: f^%d (a + b) = f^%d (a + a): %r' % (k_, k_, r))
        # the same pairs are equal when they are: a negative control for the verdicts (and the quick check on, which only ever answers yes)
        for name, x, y in (('Z.add a b = Z.add a b', zadd(a, b), zadd(a, b)), ('f (a + b) = f (a + b)', fz(zadd(a, b)), fz(zadd(a, b))),
                           ('(fun u v => Z.add u v) a b = Z.add a b', wz.app(wz.fun([Zt, Zt], lambda u, v: zadd(u, v)), a, b), zadd(a, b))):
            for syn in (False, True):
                r = decide(wz, x, y, syn)
                expect(r.ver is True and r.cver is True and r.retries == 0 and r.steps <= Z_BOUND, 'A[syn %s]: %s: %r' % (syn, name, r))


# ---------------------------------------------------------------------------------------------------------------------------------------- B
def part_B(w):
    add, addb, mul, idn, f, g, h1, S, two, three = (w[n] for n in ('add', 'addb', 'mul', 'idn', 'f', 'g', 'h1', 'Succ', 'two', 'three'))
    with nat_vars(w) as (x, y, z):
        # B1: a differing case split: the environments the two splits captured differ (two against three): written out in full, and different
        r = decide(w, add(x, two), add(x, three))
        expect(r.ver is False and r.cver is False and r.uret >= 1 and r.cret >= 1, 'B1: add x 2 = add x 3 was not retried: %r' % r)
        r = decide(w, add(mul(x, y), z), add(mul(x, y), x))
        expect(r.ver is False and r.cver is False and r.uret >= 1 and r.cret >= 1, 'B1: add (mul x y) z = add (mul x y) x was not retried: %r' % r)
        # B2: an answer found by the retry is decided: k wrappers around the mismatch retry ONCE (each wrapper's refusal is certain), in linear steps
        for k_ in (3, 10, 30):
            a = add(x, two); b = add(x, three)
            for _ in range(k_): a = f(a); b = f(b)
            r = decide(w, a, b)
            expect(r.ver is False and r.cver is False, 'B2: f^%d (add x 2) = f^%d (add x 3) was accepted: %r' % (k_, k_, r))
            expect(r.uret == 1 and r.cret == 1, 'B2: f^%d (add x 2) = f^%d (add x 3): %d / %d retries, expected one (a retry is not decided: every wrapper retried)' % (k_, k_, r.uret, r.cret))
            expect(r.uref >= k_ and r.cref >= k_, 'B2: f^%d: only %d / %d refusals counted' % (k_, r.uref, r.cref))
            expect(r.steps <= 500 + 40 * k_, 'B2: f^%d (add x 2) = f^%d (add x 3) took %d steps' % (k_, k_, r.steps))
        # B3: a differing fixpoint argument:  h1 add = h1 mul  (two recursive functions as values): the failure is at a fixpoint, which is incomplete; the retry finds them different
        r = decide(w, h1(add), h1(mul))
        expect(r.ver is False and r.cver is False and r.uret >= 1 and r.cret >= 1 and r.steps <= 600, 'B3: h1 add = h1 mul: %r' % r)
        r = decide(w, h1(add), h1(add))
        expect(r.ver is True and r.cver is True and r.retries == 0 and r.steps <= 100, 'B3: h1 add = h1 add: %r' % r)
        eta = w.fun([w['Nat'], w['Nat']], lambda a, b: add(a, b))
        r = decide(w, h1(add), h1(eta))
        expect(r.ver is True and r.cver is True, 'B3: h1 add = h1 (fun a b => add a b) (a function and its eta expansion): %r' % r)
        # B4: two definitions of addition that agree on every value, not by computation: the descent into their case splits never ends and is cut at the bound (fails closed: not convertible)
        r = decide(w, add(x, y), addb(x, y))
        expect(r.ver is False and r.cver is False, 'B4: add x y = addb x y was accepted (they are not convertible by computation): %r' % r)
        expect(r.retries >= 1, 'B4: the descent past the bound was not retried: %r' % r)
        expect(r.steps <= 40_000, 'B4: add x y = addb x y took %d steps (about 12,000: 129 levels, each retried once)' % r.steps)
        r = decide(w, h1(add), h1(addb))
        expect(r.ver is False and r.cver is False and r.steps <= 40_000, 'B4: h1 add = h1 addb: %r' % r)
        # B5: equal and different arguments of a stuck constant, in every position
        for name, a, b, want in (('g (plus2 x) y = g (plus2 x) y', g(w['plus2'](x), y), g(w['plus2'](x), y), True),
                                 ('g (plus2 x) y = g (plus2 x) z', g(w['plus2'](x), y), g(w['plus2'](x), z), False),
                                 ('g (plus2 x) y = g (plus2 y) y', g(w['plus2'](x), y), g(w['plus2'](y), y), False),
                                 ('g x (idn y) = g x y', g(x, idn(y)), g(x, y), True)):
            r = decide(w, a, b)
            expect(r.ver is want and r.cver is want and r.steps <= 300, 'B5: %s: %r' % (name, r))
            if not want: expect(r.uret == 0 and r.cret == 0 and r.refusals >= 1, 'B5: %s: a failure at data was retried: %r' % (name, r))


# ---------------------------------------------------------------------------------------------------------------------------------------- C
def inline_add(w, x, y, zero_branch=None):
    """`match x with Zero => y | Succ p => Succ (add p y)` written out: the computation `add x y` has, with the case split a syntax node of the ambient context instead of the function's own"""
    Nat, S, add = w['Nat'], w['Succ'], w['add']
    return w.match(x, lambda _: Nat, [y if zero_branch is None else zero_branch, lambda p: S(add(p, y))])


def gen_term(w, rnd, vs, depth):
    Nat, S, Z0, add = w['Nat'], w['Succ'], w['Zero'], w['add']
    f, g, mul, plus2, idn, two, three = (w[n] for n in ('f', 'g', 'mul', 'plus2', 'idn', 'two', 'three'))
    if depth == 0 or rnd.random() < 0.15: return rnd.choice(vs + [Z0, two, three])
    c = rnd.random(); sub = lambda: gen_term(w, rnd, vs, depth - 1)
    if c < 0.18: return S(sub())
    if c < 0.40: return add(sub(), sub())
    if c < 0.58: return mul(sub(), sub())
    if c < 0.68: return f(sub())
    if c < 0.78: return g(sub(), sub())
    if c < 0.90: return plus2(sub())
    return idn(sub())


def part_C(w, pairs=170):
    k = w.k
    expect(k.get_conv_nesting_limit() == 256, 'C: the shipped nesting bound is %s, expected 256' % k.get_conv_nesting_limit())
    for bad in (-1, 4097, 10**6):
        try: k.set_conv_nesting_limit(bad); expect(False, 'C: set_conv_nesting_limit(%d) was accepted' % bad)
        except E.ITPError: pass
    expect(k.get_conv_nesting_limit() == 256, 'C: a refused limit changed the bound')
    add, S = w['add'], w['Succ']
    with nat_vars(w) as (x, y, z):
        good = inline_add(w, x, y); bad = inline_add(w, x, y, z)
        r = decide(w, add(x, y), good)
        expect(r.ver is True and r.cver is True and r.retries == 0, 'C1: add x y = its case split written inline, at the shipped bound: %r' % r)
        r = decide(w, add(x, y), bad)
        expect(r.ver is False and r.cver is False, 'C1: add x y = a case split with another zero branch was accepted: %r' % r)
        for lim in (0, 1):
            k.set_conv_nesting_limit(lim)
            try:
                expect(k.get_conv_nesting_limit() == lim, 'C2: the bound did not take the value %d' % lim)
                r = decide(w, add(x, y), good)
                expect(r.ver is True and r.cver is True, 'C2[%d]: add x y = its case split written inline was refused with the bound lowered (the retry must find them equal): %r' % (lim, r))
                expect(r.uret >= 1 and r.cret >= 1, 'C2[%d]: ... and it must be the retry that finds them equal: %r' % (lim, r))
                r = decide(w, add(x, y), bad)
                expect(r.ver is False and r.cver is False and r.uret >= 1 and r.cret >= 1, 'C2[%d]: add x y = a case split with another zero branch: %r' % (lim, r))
                nf = w.normalize(add(w['mul'](x, y), z))
                r = decide(w, add(w['mul'](x, y), z), nf)
                expect(r.ver is True and r.cver is True, 'C2[%d]: add (mul x y) z = its normal form with the bound lowered: %r' % (lim, r))
            finally: k.set_conv_nesting_limit(256)
        expect(k.get_conv_nesting_limit() == 256, 'C2: the bound was not restored')
        # a refusal at data is still a refusal with the bound lowered (an argument that differs is different data, whatever the case splits inside it did)
        k.set_conv_nesting_limit(0)
        try:
            r = decide(w, w['g'](add(x, y), z), w['g'](add(x, y), x))
            expect(r.ver is False and r.cver is False and r.retries == 0, 'C2: g (add x y) z = g (add x y) x with the bound at 0: %r' % r)
            # ... but a difference at data that has an argument failing at the bound is decided by the retry: the arguments are equal after all
            r = decide(w, w['g'](add(x, y), z), w['g'](inline_add(w, x, y), z))
            expect(r.ver is True and r.cver is True and r.retries >= 1, 'C2: g (add x y) z = g (inline) z with the bound at 0 (an argument fails at the bound, the retry on the whole decides): %r' % r)
            r = decide(w, w['g'](add(x, y), z), w['g'](inline_add(w, x, y), x))
            expect(r.ver is False and r.cver is False and r.retries >= 1, 'C2: g (add x y) z = g (inline) x with the bound at 0: %r' % r)
        finally: k.set_conv_nesting_limit(256)
        # C3: random pairs against full normalisation, at the shipped bound, at 2 and at 0
        rnd = random.Random(39); vs = [x, y, z]; stats = {}
        data = []
        for it in range(pairs):
            t = gen_term(w, rnd, vs, rnd.choice([2, 3, 4])); nf = w.normalize(t); kind = rnd.random()
            if kind < 0.30: u = nf
            elif kind < 0.55: u = w.simp(t)
            elif kind < 0.75: t2 = gen_term(w, rnd, vs, 2); t = add(t, t2); u = add(nf, t2)
            else: u = gen_term(w, rnd, vs, rnd.choice([2, 3]))
            data.append((t, u, w.normalize(t) == w.normalize(u)))
        for lim in (256, 2, 0):
            k.set_conv_nesting_limit(lim); retried_yes = 0; wrong = 0
            try:
                for t, u, want in data:
                    r = decide(w, t, u)
                    if r.ver is not want or r.cver is not want:
                        wrong += 1
                        if wrong <= 3: expect(False, 'C3[%d]: %s = %s: %r, full normalisation says %s' % (lim, w.show(t), w.show(u), r, want))
                    if want and (r.uret > 0 or r.cret > 0): retried_yes += 1
            finally: k.set_conv_nesting_limit(256)
            expect(wrong == 0, 'C3[%d]: %d of %d pairs decided otherwise than by full normalisation' % (lim, wrong, len(data)))
            stats[lim] = retried_yes
        expect(sum(1 for d in data if d[2]) >= pairs // 2 and sum(1 for d in data if not d[2]) >= 10, 'C3: the sample is lopsided (%d equal of %d)' % (sum(1 for d in data if d[2]), len(data)))
        expect(stats[0] >= pairs // 4, 'C3: with the bound at 0 only %d of %d pairs were answered by the retry (the hook does not bite?)' % (stats[0], len(data)))
        expect(stats[256] == 0, 'C3: at the shipped bound %d equal pairs needed the retry' % stats[256])
        expect(stats[0] > stats[2] > 0 or stats[0] == stats[2] > 0, 'C3: retry counts by bound: %s' % stats)


# ---------------------------------------------------------------------------------------------------------------------------------------- D
def part_D(w):
    Nat, add, K, Pk, S, Z0, two, f = (w[n] for n in ('Nat', 'add', 'K', 'Pk', 'Succ', 'Zero', 'two', 'f'))
    with nat_vars(w) as (x, y, z):
        # closures with the same code: K x holds x, K y holds y.  Equal values reached through different environments are equal
        cases = [
            ('K x = K y', K(x), K(y), False),
            ('K x = K x', K(x), K(x), True),
            ('K (add Zero x) = K x', K(add(Z0, x)), K(x), True),
            ('K (add x two) = K (add x (Succ (Succ Zero)))', K(add(x, two)), K(add(x, S(S(Z0)))), True),
            ('K (add x two) = K (add y two)', K(add(x, two)), K(add(y, two)), False),
            ('K (f x) = K (f y)', K(f(x)), K(f(y)), False),
            ('K x = fun b => x', K(x), w.lam(Nat, lambda b: x), True),
            ('K x = fun b => y', K(x), w.lam(Nat, lambda b: y), False),
            # types: Pk a = forall b, a = b.  Equal values reached through different environments, and different ones
            ('Pk x = Pk y', Pk(x), Pk(y), False),
            ('Pk x = Pk x', Pk(x), Pk(x), True),
            ('Pk (add Zero x) = Pk x', Pk(add(Z0, x)), Pk(x), True),
            ('Pk (add x two) = Pk (add x three)', Pk(add(x, two)), Pk(add(x, w['three'])), False),
            ('Pk (f x) = Pk (f x)', Pk(f(x)), Pk(f(x)), True),
        ]
        for name, a, b, want in cases:
            for syn in (False, True):
                r = decide(w, a, b, syn)
                expect(r.ver is want and r.cver is want, 'D[syn %s]: %s: %r, expected %s' % (syn, name, r, want))
                expect(r.steps <= 600, 'D[syn %s]: %s took %d steps' % (syn, name, r.steps))


# ---------------------------------------------------------------------------------------------------------------------------------------- E
def part_E(w):
    add, f, g, mul, two, pc, pax, hq, hr, P, S = (w[n] for n in ('add', 'f', 'g', 'mul', 'two', 'pc', 'pax', 'hq', 'hr', 'P', 'Succ'))
    with nat_vars(w) as (x, y, z):
        # E1: proofs of one proposition: equal for the unifier, not for plain conversion (which has no types)
        r = decide(w, pc(x), pax(x))
        expect(r.ver is True and r.cver is False, 'E1: pc x = pax x (a stuck proof against an axiom\'s proof of the same proposition): %r' % r)
        r = decide(w, hq(x, pc(x)), hq(x, pax(x)))
        expect(r.ver is True and r.cver is False, 'E1: hq x (pc x) = hq x (pax x): %r' % r)
        r = decide(w, hr(x, pc(x), y), hr(x, pax(x), y))
        expect(r.ver is True and r.cver is False, 'E1: hr x (pc x) y = hr x (pax x) y: %r' % r)
        # E2: an argument rescued by proof irrelevance does not make the refusal at the next argument uncertain
        r = decide(w, hr(x, pc(x), y), hr(x, pax(x), z))
        expect(r.ver is False and r.cver is False and r.uret == 0 and r.uref >= 1, 'E2: hr x (pc x) y = hr x (pax x) z: %r' % r)
        r = decide(w, hr(S(x), pc(S(x)), y), hr(S(x), pax(S(x)), z))
        expect(r.ver is False and r.uret == 0 and r.uref >= 1, 'E2: hr (S x) (pc (S x)) y = hr (S x) (pax (S x)) z: %r' % r)
        # ... also when the proofs differ in a FIXPOINT, whose failure is incomplete: the fixpoints are compared in a stuck term (retried there, once, and decided), the proofs are equal by proof irrelevance
        qf, hz = w['qf'], w['hz']
        r = decide(w, hz(qf(add), y), hz(qf(w['mul']), z))
        expect(r.ver is False and r.cver is False and r.uret == 1 and r.uref >= 1, 'E2: hz (qf add) y = hz (qf mul) z (the proofs are rescued, the data differs): %r' % r)
        r = decide(w, hz(qf(add), y), hz(qf(w['mul']), y))
        expect(r.ver is True and r.cver is False and r.uret == 1 and r.cret >= 1, 'E2: hz (qf add) y = hz (qf mul) y (equal by proof irrelevance; plain conversion retries and refuses): %r' % r)
        # a failure at a fixpoint that is NOT inside a stuck term (the fixpoints are fields of a constructor of a proposition: `exists f, Q f`, built from add and from mul) stays counted when the structural
        # walk gives up -- and is taken back when proof irrelevance then proves the two proofs equal, so the refusal at the next argument is certain
        F2 = w.arrow(w['Nat'], w.arrow(w['Nat'], w['Nat'])); Q, ex_intro, hx = w['Q'], w['ex_intro'], w['hx']
        pa_, pm_ = ex_intro(F2, Q, add, w['qa']), ex_intro(F2, Q, w['mul'], w['qm'])
        r = decide(w, hx(pa_, y), hx(pm_, z))
        expect(r.ver is False and r.uret == 1 and r.uref >= 1, 'E2: hx (ex_intro add qa) y = hx (ex_intro mul qm) z (the proofs are rescued, the refusal at the data is certain; the one retry is the comparison of the types `Q add` and `Q mul` of their last fields, decided where it was made): %r' % r)
        r = decide(w, hx(pa_, y), hx(pm_, y))
        expect(r.ver is True, 'E2: hx (ex_intro add qa) y = hx (ex_intro mul qm) y (equal by proof irrelevance): %r' % r)
        # E3: a stuck case split on DATA is not a proof: refused without trying proof irrelevance (which infers and compares the types of both sides: 30 to 120 steps more for each of these)
        for name, a, b, bound in (('add x two = add y two', add(x, two), add(y, two), 45), ('f (mul x y) = f (mul y x)', f(mul(x, y)), f(mul(y, x)), 45), ('mul x y = mul x z', mul(x, y), mul(x, z), 115),
                                  ('add (mul x y) z = add (mul x y) x', add(mul(x, y), z), add(mul(x, y), x), 210), ('g (mul x y) z = g (mul x y) x', g(mul(x, y), z), g(mul(x, y), x), 60)):
            r = decide(w, a, b)
            expect(r.ver is False and r.cver is False, 'E3: %s was accepted: %r' % (name, r))
            expect(r.usteps <= bound, 'E3: %s took %d steps in unify_open (bound %d; 111 to 280 when a stuck case split on data was offered to proof irrelevance)' % (name, r.usteps, bound))


# ---------------------------------------------------------------------------------------------------------------------------------------- F
def part_F(wz):
    """`Z.strong_induction P a step n h` has the type `P n` -- a beta redex of the family P applied to n at the end of a Pi chain.  Exposing the Pi on syntax keeps the type folded"""
    k = wz.k; Zt, zadd, zmul, z0, z1, zle, zlt = wz['Z'], wz['Z.add'], wz['Z.mul'], wz['Z.zero'], wz['Z.one'], wz['Z.le'], wz['Z.lt']
    Eq = wz.Eq; strong = wz['Z.strong_induction']
    motive = wz.lam(Zt, lambda u: Eq(zadd(zmul(u, u), u), zmul(u, zadd(u, z1))))
    wrong_motive = wz.lam(Zt, lambda u: Eq(zadd(zmul(u, u), u), zmul(u, zadd(u, z0))))
    step_ty = lambda mot: wz.forall_([Zt, lambda u: zle(z0, u), lambda u, hu: wz.forall_([Zt, lambda v: zle(z0, v), lambda v, hv: zlt(v, u)], lambda v, hv, hl: mot(v))], lambda u, hu, ih: mot(u))
    base = len(wz.ctx)
    step = wz._bind(step_ty(motive), 'step')[1]; n = wz._bind(Zt, 'n')[1]; h = wz._bind(zle(z0, n), 'h')[1]
    try:
        app = wz.app(strong, motive, z0, step, n, h)
        target = motive(n)
        k.set_fuel(FUEL); ok, ty = k.infer_open(wz._n(app), wz.ctx); isteps = k.fuel_used()
        expect(ok, 'F: the application of Z.strong_induction is ill-typed: %s' % (ty,))
        expect(isteps <= 400, 'F: inferring Z.strong_induction P 0 step n h took %d steps (1,160,866 when the type was evaluated and written back)' % isteps)
        k.set_fuel(FUEL); res = k.check_open(wz._n(app), wz._n(target), wz.ctx); csteps = k.fuel_used()
        expect(res[0], 'F: Z.strong_induction P 0 step n h is not accepted at P n: %s' % (res,))
        expect(csteps <= 1000, 'F: checking Z.strong_induction P 0 step n h against P n took %d steps' % csteps)
        k.set_fuel(FUEL); u = k.unify_open(ty, wz._n(target), wz.ctx); usteps = k.fuel_used()
        expect(u[0] is True and usteps <= 300, 'F: the inferred type against P n: %s in %d steps' % (u[:1], usteps))
        # the statement is an equation of integers -- it is also accepted at the unfolded motive, and refused at a wrong one, cheaply
        wrong = wrong_motive(n)
        k.set_fuel(FUEL); res = k.check_open(wz._n(app), wz._n(wrong), wz.ctx); wsteps = k.fuel_used()
        expect(not res[0], 'F: Z.strong_induction P 0 step n h was accepted at the wrong motive')
        expect(wsteps <= 6000, 'F: checking against the wrong motive took %d steps' % wsteps)
        # a term whose motive is not the one the step proves is ill-typed (and found so without normalising the whole type)
        step2 = wz._bind(step_ty(wrong_motive), 'step2')[1]
        try:
            bad = wz.app(strong, motive, z0, step2, n, h)
            k.set_fuel(FUEL); ok2, msg2 = k.infer_open(wz._n(bad), wz.ctx); bsteps = k.fuel_used()
            expect(not ok2, 'F: Z.strong_induction with a step for another motive was accepted')
            expect(bsteps <= 8000, 'F: refusing a step for another motive took %d steps' % bsteps)
        finally: wz._unbind(1)
    finally: wz._unbind(len(wz.ctx) - base)


# ---------------------------------------------------------------------------------------------------------------------------------------- H
def part_H(wz):
    """the unifier opens a binder to compare two functions or two Pi types: the variable it adds to its context has the binder's type AS WRITTEN. (It used to write the binder's VALUE back, which normalises it: a hypothesis
    such as `u < u` over the integers is a page of lifted functions, ~400,000 steps, spent on every comparison under it that failed -- and on every one that succeeded but was not syntactically equal.)"""
    Zt, zlt, z0, z1 = wz['Z'], wz['Z.lt'], wz['Z.zero'], wz['Z.one']; Eq = wz.Eq
    with Ctx(wz, [(Zt, 'u')]) as (u,):
        lt = zlt(u, u)
        kk = wz._bind(wz.arrow(lt, Zt), 'kk')[1]
        try:
            eta = wz.lam(lt, lambda hh: kk(hh)); const0 = wz.lam(lt, lambda hh: z0); const1 = wz.lam(lt, lambda hh: z1)
            pi_a = wz.forall_([lt], lambda hh: Eq(u, u)); pi_b = wz.forall_([lt], lambda hh: Eq(u, z0))
            cases = [('kk = fun h => kk h', kk, eta, True), ('fun h => kk h = kk', eta, kk, True), ('kk = fun h => 0', kk, const0, False), ('fun h => 0 = kk', const0, kk, False),
                     ('fun h => 0 = fun h => 1', const0, const1, False), ('fun h => 0 = fun h => 0', const0, wz.lam(lt, lambda hh: z0), True),
                     ('(u < u) -> u = u   against   (u < u) -> u = 0', pi_a, pi_b, False)]
            for name, a, b, want in cases:
                r = decide(wz, a, b)
                expect(r.ver is want and r.cver is want, 'H: %s: %r, expected %s' % (name, r, want))
                expect(r.usteps <= 2500 and r.csteps <= 2500, 'H: %s took %d (unify) and %d (conv) steps (~400,000 when the binder\'s type was normalised)' % (name, r.usteps, r.csteps))
        finally: wz._unbind(1)


# ---------------------------------------------------------------------------------------------------------------------------------------- G
def part_G(w):
    k = w.k
    add, addb, mul, f, h1, two, three, S = (w[n] for n in ('add', 'addb', 'mul', 'f', 'h1', 'two', 'three', 'Succ'))
    with nat_vars(w) as (x, y, z):
        pairs = [(add(x, two), add(x, three)), (f(add(x, two)), f(add(x, three))), (h1(add), h1(mul)), (add(x, y), inline_add(w, x, y)), (add(x, y), addb(x, y)), (f(x), f(y)), (add(mul(x, y), z), w.normalize(add(mul(x, y), z)))]
        before = k.neutral_stats()
        deltas = {}
        for flag in (False, True):
            k.set_option('verify_memo', flag)
            try:
                got = []
                for lim in (256, 0):
                    k.set_conv_nesting_limit(lim)
                    try:
                        for a, b in pairs:
                            s0 = k.neutral_stats(); r = decide(w, a, b); s1 = k.neutral_stats()
                            expect(s1[0] >= s0[0] and s1[1] >= s0[1], 'G: a counter went down')
                            got.append((r.ver, r.cver, r.uret, r.uref, r.cret, r.cref))
                    finally: k.set_conv_nesting_limit(256)
                deltas[flag] = got
            finally: k.set_option('verify_memo', False)
        expect(deltas[False] == deltas[True], 'G: the counters and verdicts differ with the self-check option on:\n  off %s\n  on  %s' % (deltas[False], deltas[True]))
        after = k.neutral_stats()
        expect(after[0] > before[0] and after[1] > before[1], 'G: nothing was counted: %s -> %s' % (before, after))


def main():
    t0 = time.time()
    w = build_nat()
    wz = build_int()
    for name, part, world in (('A', part_A, wz), ('B', part_B, w), ('C', part_C, w), ('D', part_D, w), ('E', part_E, w), ('F', part_F, wz), ('H', part_H, wz), ('G', part_G, w)):
        t1 = time.time(); n0 = len(FAIL)
        part(world)
        print('  part %s: %d checks so far, %d failures, %.1f s' % (name, CHECKS[0], len(FAIL) - n0, time.time() - t1), flush=True)
    print('%d checks, %.1f s' % (CHECKS[0], time.time() - t0))
    if FAIL:
        for f_ in FAIL[:60]: print('  FAIL:', f_)
        print('S39 FAILFAST GATE: FAIL (%d of %d)' % (len(FAIL), CHECKS[0])); sys.exit(1)
    print('S39 FAILFAST GATE: PASS')


if __name__ == '__main__':
    main()
