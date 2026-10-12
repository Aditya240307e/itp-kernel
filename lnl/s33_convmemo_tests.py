"""S33 gate: conversion and unification over SHARED structure (the tables ConvMemo and UnifyMemo).

The values the evaluator compares are DAGs. `add n m` on a stuck `n` leaves a pending `match` whose captured environment holds `n` itself, the stuck value one level down; so the value of k nested
applications `plus2 (plus2 (... x))` holds the value of k-1 of them TWICE (as the prefix of its spine and inside its last frame), and comparing two such values frame by frame compared the level
below twice at every level: 2^k steps (4000 at k = 12, 134 million at k = 25). The same happens for constructors (`mk t = node t t` makes `mk^k leaf` a DAG of k nodes and a tree of 2^k) and for any
stuck function applied to one value twice (`dup n = hh n n` with an opaque `hh`).

While one comparison runs, `conv_value` (the pure conversion check) and `unify_structural` (the unifier's walk, which may solve metavariables, defer constraints and take attempts back) remember the pairs
of sub-values they found equal, keyed by the identity of the shared blocks, the head, the depth, the evaluation switches (and, in the unifier, the elaboration context), so each distinct pair is decided
once. The tables only hold answers already found, within one comparison; nothing here changes what is accepted. This gate pins down (the quick check `syn_conv` is OFF unless said, so that the evaluator decides):

  A. nested stuck matches: `plus2^k x` against `plus2^k (idn x)` for k up to 100 is decided in a number of steps polynomial in k (the kernel's own fuel meter; 2^k before), and the table is used;
  B. the same for constructors, `mk^k leaf` against the same tree written out, k up to 60, by the unifier and by conversion;
  C. the tables never confuse pairs: the same value compared with two different partners in ONE comparison (the second must be refused), differences in the last, a middle and the first frame, a different
     constant in a frame, unequal heads; every verdict equals the expected one, the verdicts of `conv_open` and `unify_open` agree, with the quick check off and on;
  D. the tables live for one comparison: both are empty between calls, and a pair found equal in one call does not make a different pair equal in the next;
  E. stuck function applications that share an argument (`dup^k x`, `hh3 A p A`): polynomial, the unifier's table is used, a pair is not confused with a different partner;
  F. metavariables and the unifier's table: a solution recorded in between keeps the table, a solution TAKEN BACK (an attempt that failed and was rolled back by proof irrelevance) drops it -- a pair
     remembered under that solution must be decided again and solves the metavariable again; a rolled-back DEFERRED constraint is deferred again by a later visit of the same pair;
  G. the option verify_memo: every answer from a table (and from the quick check) is decided again the slow way; the verdicts do not change, answers were verified, and nothing is thrown;
  H. quotation (ConvMemo and UnifyMemo fall back to writing both sides back as syntax when a comparison is refused, and normalisation does it always): the syntax of a shared value is found without
     walking every path (`normalize` of `dup^k x`, of `mk^k leaf` and of a lambda that holds a big shared value twice), the result is the small DAG the construction gives (compared as hash-consed
     nodes), the table is used and is empty between calls;
  I. the tables are GIVEN BACK when a comparison ends (a cleared hash table keeps its bucket array, and clearing writes all of it: after one big comparison every later small one paid for the biggest table
     ever built): after comparisons that fill each table with hundreds of entries none holds more than 256 buckets, the typing memo included.
"""
import os, sys, random, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(100000)
import itp_env_lnl as E
import lnl_base
from author import World

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


def build():
    w = World()
    lnl_base.declare_types(w); lnl_base.declare_add(w)
    Nat, S, Z0, add, TYPE, True_ = w['Nat'], w['Succ'], w['Zero'], w['add'], w.TYPE, w['True']
    NN = w.arrow(Nat, Nat)
    w.define('two', Nat, S(S(Z0)))
    two = w['two']
    w.define('idn', NN, w.lam(Nat, lambda x: x))
    w.define('plus2', NN, w.lam(Nat, lambda n: add(n, two)))
    w.define('plus3', NN, w.lam(Nat, lambda n: add(n, S(two))))
    # binary trees (data), and a definition that shares: mk t = node t t
    w.inductive('Tree', TYPE, [('leaf', []), ('node', [lambda *_: w['Tree'], lambda *_: w['Tree']])])
    Tree = w['Tree']
    w.define('mk', w.arrow(Tree, Tree), w.lam(Tree, lambda t: w['node'](t, t)))
    w.define('mk3', w.forall_([Tree, Tree, Tree], lambda a, b, c: Tree), w.fun([Tree, Tree, Tree], lambda a, b, c: w['node'](w['node'](a, b), c)))
    w.define('mk2', w.arrow(Tree, Tree), w.lam(Tree, lambda t: w['node'](t, t)))              # the same function under another name: its value is built separately
    # opaque functions (axioms) and definitions that hold their argument more than once:  dup n = hh n n,  share3 n p = hh3 n p n
    w.axiom('hh', w.arrow(Nat, w.arrow(Nat, Nat)))
    w.axiom('hh3', w.arrow(Nat, w.arrow(Nat, w.arrow(Nat, Nat))))
    hh, hh3 = w['hh'], w['hh3']
    w.define('dup', NN, w.lam(Nat, lambda n: hh(n, n)))
    w.define('dup2', NN, w.lam(Nat, lambda n: hh(n, n)))
    w.define('share3', w.arrow(Nat, w.arrow(Nat, Nat)), w.lam(Nat, lambda n: w.lam(Nat, lambda p: hh3(n, p, n))))
    # proofs of True: two unrelated ones, and proofs built from data.   hq u n : Nat  takes a proof;   enc n z = hq (ax4 n n z) n  holds n three times
    w.axiom('pa', True_); w.axiom('pb', True_)
    w.axiom('ax4', w.arrow(Nat, w.arrow(Nat, w.arrow(Nat, True_))))
    w.axiom('hq', w.arrow(True_, w.arrow(Nat, Nat)))
    w.define('enc', w.arrow(Nat, w.arrow(Nat, Nat)), w.lam(Nat, lambda n: w.lam(Nat, lambda z: w['hq'](w['ax4'](n, n, z), n))))
    return w


class Ctx:
    def __init__(self, w): self.w = w
    def __enter__(self):
        w = self.w; Nat = w['Nat']; Tree = w['Tree']
        self.vars = [w._bind(Nat, nm)[1] for nm in ('x', 'y', 'z')] + [w._bind(Tree, nm)[1] for nm in ('s', 't')]
        return self.vars
    def __exit__(self, *exc): self.w._unbind(5)


def tower(f, base, k):
    t = base
    for _ in range(k): t = f(t)
    return t


FUEL = 20_000_000


class Res:
    """what one comparison did: verdicts of unify_open (`ver`) and conv_open (`cver`), the steps each took, the sub-comparisons each answered from its table (None: the budget ran out)"""
    def __init__(self, ver, cver, usteps, csteps, uhits, chits, pending=0):
        self.ver, self.cver, self.usteps, self.csteps, self.uhits, self.chits, self.pending = ver, cver, usteps, csteps, uhits, chits, pending
    def __repr__(self): return 'Res(ver=%s cver=%s usteps=%s csteps=%s uhits=%s chits=%s pending=%s)' % (self.ver, self.cver, self.usteps, self.csteps, self.uhits, self.chits, self.pending)


def decide(w, a, b, mode=False, unify=True, conv=True):
    """compare (a, b) with the quick check set to `mode` by unify_open and by conv_open"""
    k = w.k; k.set_option('syn_conv', mode)
    ver = cver = None; usteps = csteps = FUEL; uh = ch = 0; pending = 0
    if unify:
        k.set_fuel(FUEL); h0 = k.unify_memo_hits()
        try:
            r = k.unify_open(w._n(a), w._n(b), w.ctx); ver = r[0]; pending = r[2]; usteps = k.fuel_used()
        except E.FuelExhausted: pass
        uh = k.unify_memo_hits() - h0
    if conv:
        k.set_fuel(FUEL); h0 = k.conv_memo_hits()
        try:
            cver = k.conv_open(w._n(a), w._n(b), len(w.ctx)); csteps = k.fuel_used()
        except E.FuelExhausted: pass
        ch = k.conv_memo_hits() - h0
    return Res(ver, cver, usteps, csteps, uh, ch, pending)


# ---------------------------------------------------------------------------------------------------------------------------------------- A
def part_A(w):
    plus2, idn = w['plus2'], w['idn']
    with Ctx(w) as (x, y, z, s, t):
        for mode in (False, True):
            for k_ in (4, 10, 20, 40, 70, 100):
                a = tower(plus2, x, k_); b = tower(plus2, idn(x), k_)
                r = decide(w, a, b, mode)
                bound = 5000 + 2 * k_ ** 3
                expect(r.ver is True and r.cver is True, 'A[%s]: plus2^%d x = plus2^%d (idn x): %r' % (mode, k_, k_, r))
                if not mode:
                    expect(r.usteps <= bound and r.csteps <= bound, 'A: plus2^%d x = plus2^%d (idn x) took %d (unify) and %d (conv) steps with the quick check off (bound %d; 2^k before the table)' % (k_, k_, r.usteps, r.csteps, bound))
                    if k_ >= 10: expect(r.chits > 0, 'A: the conversion table was never used for k = %d' % k_)
            # the work grows like a polynomial, not like 2^k (the ratio of the steps at 100 and at 50 levels is far below 2^50)
            if not mode:
                a50 = tower(plus2, x, 50); b50 = tower(plus2, idn(x), 50); a100 = tower(plus2, x, 100); b100 = tower(plus2, idn(x), 100)
                r50 = decide(w, a50, b50, mode); r100 = decide(w, a100, b100, mode)
                expect(r100.usteps <= 12 * r50.usteps and r100.csteps <= 12 * r50.csteps, 'A: doubling the depth from 50 to 100 multiplied the steps by %.1f (unify) and %.1f (conv) (a polynomial of degree 3 gives 8)' % (r100.usteps / max(r50.usteps, 1), r100.csteps / max(r50.csteps, 1)))


# ---------------------------------------------------------------------------------------------------------------------------------------- B
def part_B(w):
    """constructors: a DAG of k nodes that is a tree of 2^k"""
    mk, mk2, node, leaf = w['mk'], w['mk2'], w['node'], w['leaf']
    with Ctx(w) as (x, y, z, s, t):
        for k_ in (3, 10, 25, 40, 60):
            a = tower(mk, leaf, k_)                                  # mk (mk (... leaf)) : a DAG of k nodes
            b = tower(mk2, leaf, k_)                                 # the same, built by another definition: its own DAG
            c = node(tower(mk, leaf, k_ - 1), tower(mk2, leaf, k_ - 1))      # node A A2 : two separate towers below the top
            r = decide(w, a, b, False); r2 = decide(w, a, c, False)
            bound = 20000 + 300 * k_ ** 2
            expect(r.ver is True and r.cver is True, 'B: mk^%d leaf = mk2^%d leaf: %r' % (k_, k_, r))
            expect(r2.ver is True and r2.cver is True, 'B: mk^%d leaf = node (mk^%d leaf) (mk2^%d leaf): %r' % (k_, k_ - 1, k_ - 1, r2))
            expect(max(r.usteps, r.csteps, r2.usteps, r2.csteps) <= bound, 'B: k = %d took %r and %r (bound %d; the tree has 2^k nodes)' % (k_, r, r2, bound))
            if k_ >= 10: expect(r.uhits > 0 and r.chits > 0 and r2.uhits > 0 and r2.chits > 0, 'B: a table was never used for k = %d: %r %r' % (k_, r, r2))
            # with the quick check on, the unifier settles these pairs on syntax
            r3 = decide(w, a, b, True)
            expect(r3.ver is True and r3.cver is True, 'B: mk^%d leaf = mk2^%d leaf with the quick check on: %r' % (k_, k_, r3))
        for k_ in (10, 40):
            a = tower(mk, leaf, k_)
            d = tower(mk2, s, k_)                                    # a variable at the bottom: not convertible
            r = decide(w, a, d, False)
            expect(r.ver is False and r.cver is False, 'B: mk^%d leaf = mk2^%d s was accepted: %r' % (k_, k_, r))
            # a difference only in the SECOND child of the top node, after the first child (a deep DAG) has been found equal
            e = node(tower(mk2, leaf, k_ - 1), node(leaf, leaf))
            r = decide(w, a, e, False)
            expect(r.ver is False and r.cver is False and max(r.usteps, r.csteps) <= 20000 + 300 * k_ ** 2, 'B: mk^%d leaf = node (mk2^%d leaf) (node leaf leaf): %r' % (k_, k_ - 1, r))


# ---------------------------------------------------------------------------------------------------------------------------------------- C
def part_C(w):
    plus2, plus3, idn, add, two, S, node, leaf, mk3, mk = w['plus2'], w['plus3'], w['idn'], w['add'], w['two'], w['Succ'], w['node'], w['leaf'], w['mk3'], w['mk']
    with Ctx(w) as (x, y, z, s, t):
        K = 14
        N = tower(plus2, x, K); N2 = tower(plus2, idn(x), K)                # equal towers
        M = tower(plus3, x, K)                                              # a different tower (+3 at every level)
        topdiff = add(tower(plus2, x, K - 1), S(two)); topsame = add(tower(plus2, x, K - 1), two)
        mid = tower(plus2, x, 6); mid = plus3(mid); mid = tower(plus2, mid, K - 7)       # a different constant in ONE middle frame
        first = plus3(x); first = tower(plus2, first, K - 1)                # a different constant in the first frame
        cases = [
            ('N = N2', N, N2, True),
            ('N = M (every frame differs)', N, M, False),
            ('add (plus2^(K-1) x) 3 = add (plus2^(K-1) x) 2   (the last frame)', topdiff, topsame, False),
            ('plus2^K x = the middle frame changed', N, mid, False),
            ('plus2^K x = the first frame changed', N, first, False),
            ('plus2^K y = plus2^K x   (unequal heads)', tower(plus2, y, K), N, False),
            ('add N N = add N2 N2', add(N, N), add(N2, N2), True),
            # the SAME value met twice, once with an equal partner and once with a different one: the second pair must not be taken for the first
            ('add N N = add N2 M', add(N, N), add(N2, M), False),
            ('add N N = add M N2', add(N, N), add(M, N2), False),
            ('add N M = add N2 M', add(N, M), add(N2, M), True),
            ('add N M = add N2 N2', add(N, M), add(N2, N2), False),
            ('add N M = add M N2', add(N, M), add(M, N2), False),
            # constructors: the same child twice with different partners
            ('node (mk^6 leaf) (mk^6 leaf) = the same', node(tower(mk, leaf, 6), tower(mk, leaf, 6)), node(tower(mk, leaf, 6), tower(mk, leaf, 6)), True),
            ('node T T = node T T2', node(tower(mk, leaf, 12), tower(mk, leaf, 12)), node(tower(mk, leaf, 12), node(tower(mk, leaf, 11), s)), False),
            ('mk3 T T T = node (node T T) T', mk3(tower(mk, leaf, 9), tower(mk, leaf, 9), tower(mk, leaf, 9)), node(node(tower(mk, leaf, 9), tower(mk, leaf, 9)), tower(mk, leaf, 9)), True),
            ('mk3 T T T = node (node T T) T2', mk3(tower(mk, leaf, 9), tower(mk, leaf, 9), tower(mk, leaf, 9)), node(node(tower(mk, leaf, 9), tower(mk, leaf, 9)), node(tower(mk, leaf, 8), s)), False),
            ('node T T = node T2 T   (the first child differs)', node(tower(mk, leaf, 12), tower(mk, leaf, 12)), node(node(tower(mk, leaf, 11), s), tower(mk, leaf, 12)), False),
        ]
        for name, a, b, want in cases:
            for m in (False, True):
                r = decide(w, a, b, m)
                expect(r.ver is want, 'C[syn %s]: %s: unify says %s, expected %s' % (m, name, r.ver, want))
                expect(r.cver is want, 'C[syn %s]: %s: conv says %s, expected %s' % (m, name, r.cver, want))


# ---------------------------------------------------------------------------------------------------------------------------------------- D
def part_D(w):
    plus2, idn, mk, leaf, dup, hh = w['plus2'], w['idn'], w['mk'], w['leaf'], w['dup'], w['hh']
    k = w.k
    def sizes(): return k.conv_memo_size(), k.unify_memo_size(), k.quote_memo_size()
    with Ctx(w) as (x, y, z, s, t):
        N = tower(plus2, x, 12); N2 = tower(plus2, idn(x), 12); M = tower(plus2, y, 12)
        expect(sizes() == (0, 0, 0), 'D: a table is not empty between calls: %s' % (sizes(),))
        for name, a, b in (('plus2 towers', N, N2), ('trees', tower(mk, leaf, 20), tower(w['mk2'], leaf, 20)), ('dup towers', tower(dup, x, 20), tower(w['dup2'], idn(x), 20))):
            r = decide(w, a, b, False)
            expect(r.ver is True and r.cver is True, 'D: %s: %r' % (name, r))
            expect(sizes() == (0, 0, 0), 'D: %s: a table is not empty after a comparison: %s' % (name, sizes()))
        # a failing comparison leaves nothing behind either, and an equal pair found in one call decides nothing in the next
        r = decide(w, N, M, False)
        expect(r.ver is False and r.cver is False, 'D: plus2^12 x = plus2^12 y was accepted: %r' % r)
        expect(sizes() == (0, 0, 0), 'D: a table is not empty after a refused comparison: %s' % (sizes(),))
        r = decide(w, tower(dup, x, 20), tower(w['dup2'], y, 20), False)
        expect(r.ver is False and r.cver is False, 'D: dup^20 x = dup2^20 y was accepted: %r' % r)
        expect(sizes() == (0, 0, 0), 'D: a table is not empty after a refused comparison of dup towers: %s' % (sizes(),))
        a = k.conv_open(w._n(N), w._n(N2), len(w.ctx)); b = k.conv_open(w._n(N), w._n(M), len(w.ctx))
        expect(a is True and b is False, 'D: conv_open after unify_open: %s, %s' % (a, b))
        # a call that ends by an exception (the budget) leaves nothing behind
        k.set_option('syn_conv', False); k.set_fuel(2000)
        try:
            k.unify_open(w._n(tower(w['dup'], x, 60)), w._n(tower(w['dup2'], idn(x), 60)), w.ctx)
        except E.FuelExhausted: pass
        k.set_fuel(FUEL)
        expect(sizes() == (0, 0, 0), 'D: a table is not empty after a call that ran out of budget: %s' % (sizes(),))


# ---------------------------------------------------------------------------------------------------------------------------------------- E
def chain(w, base, n, head):
    """hh head (hh head (... base)), n levels: a stuck value without sharing of its own, whose comparison takes real work"""
    t = base
    for _ in range(n): t = w['hh'](head, t)
    return t


def part_E(w):
    dup, dup2, idn, hh, hh3, share3, S = w['dup'], w['dup2'], w['idn'], w['hh'], w['hh3'], w['share3'], w['Succ']
    with Ctx(w) as (x, y, z, s, t):
        for k_ in (3, 10, 20, 40, 70):
            a = tower(dup, x, k_); b = tower(dup2, idn(x), k_)
            r = decide(w, a, b, False)
            bound = 3000 + 150 * k_
            expect(r.ver is True and r.cver is True, 'E: dup^%d x = dup2^%d (idn x): %r' % (k_, k_, r))
            expect(r.usteps <= bound and r.csteps <= bound, 'E: dup^%d x = dup2^%d (idn x) took %d (unify) and %d (conv) steps (bound %d; the tree has 2^k leaves)' % (k_, k_, r.usteps, r.csteps, bound))
            if k_ >= 10: expect(r.uhits >= k_ // 2 and r.chits >= k_ // 2, 'E: the tables were barely used for k = %d: %r' % (k_, r))
            r = decide(w, a, b, True)
            expect(r.ver is True and r.cver is True, 'E: dup^%d x = dup2^%d (idn x) with the quick check on: %r' % (k_, k_, r))
        # differences
        for k_ in (10, 40):
            r = decide(w, tower(dup, x, k_), tower(dup2, y, k_), False)
            expect(r.ver is False and r.cver is False and max(r.usteps, r.csteps) <= 3000 + 150 * k_, 'E: dup^%d x = dup2^%d y (the base differs): %r' % (k_, k_, r))
            left = tower(dup, x, k_); right = hh(tower(dup2, x, k_ - 1), tower(dup2, S(x), k_ - 1))        # the first child is equal, the second differs at its base
            r = decide(w, left, right, False)
            expect(r.ver is False and r.cver is False and max(r.usteps, r.csteps) <= 3000 + 150 * k_, 'E: dup^%d x = hh (dup2^%d x) (dup2^%d (S x)): %r' % (k_, k_ - 1, k_ - 1, r))
        # one argument held twice, met with an equal partner and then a different one (the second must not reuse the first answer), in both orders and in both positions
        n = 30
        A = chain(w, y, n, x); B = chain(w, idn(y), n, x); C = chain(w, z, n, x)
        cases = [
            ('share3 A p = hh3 B p B2', share3(A, x), hh3(B, x, chain(w, idn(y), n, idn(x))), True),
            ('share3 A p = hh3 B p C', share3(A, x), hh3(B, x, C), False),
            ('share3 A p = hh3 C p B', share3(A, x), hh3(C, x, B), False),
            ('hh3 A p C = share3 B p', hh3(A, x, C), share3(B, x), False),
            ('hh3 C p A = share3 B p', hh3(C, x, A), share3(B, x), False),
            ('share3 A p = share3 B p', share3(A, x), share3(B, x), True),
            ('share3 A p = share3 C p', share3(A, x), share3(C, x), False),
        ]
        for name, a, b, want in cases:
            for m in (False, True):
                r = decide(w, a, b, m)
                expect(r.ver is want and r.cver is want, 'E[syn %s]: %s: %r, expected %s' % (m, name, r, want))
        r = decide(w, share3(A, x), share3(B, x), False)
        expect(r.uhits >= 1 and r.chits >= 1, 'E: share3 A p = share3 B p: the second visit of (A, B) was not answered from the tables: %r' % r)


# ---------------------------------------------------------------------------------------------------------------------------------------- F
def new_meta(w, ty, apply_to=()):
    """a fresh metavariable of the context, applied to the given variables (so that its solution may mention them: the elaborator makes every metavariable `?m v0 v1 ..` over its context)"""
    k = w.k; mid = k.fresh_meta(w._n(ty), len(w.ctx)); m = w.T(k.mk_meta(mid))
    return mid, (m(*apply_to) if apply_to else m)


def unify_with_metas(w, a, b, mode=False):
    """unify_open (a, b): (verdict, deferred constraints, steps, hits of the unifier's table)"""
    k = w.k; k.set_option('syn_conv', mode); k.set_fuel(FUEL); h0 = k.unify_memo_hits()
    try: r = k.unify_open(w._n(a), w._n(b), w.ctx); return r[0], r[2], k.fuel_used(), k.unify_memo_hits() - h0
    except E.FuelExhausted: return None, 0, FUEL, k.unify_memo_hits() - h0


def part_F(w):
    Nat, S, hh, hh3, dup, dup2, idn, share3, enc = w['Nat'], w['Succ'], w['hh'], w['hh3'], w['dup'], w['dup2'], w['idn'], w['share3'], w['enc']
    k = w.k
    n = 30
    with Ctx(w) as (x, y, z, s, t):
        def M():
            return new_meta(w, w.arrow(Nat, w.arrow(Nat, w.arrow(Nat, Nat))), (x, y, z))
        def solution_of_var(v):
            """the raw dump of the solution a trivial unification gives a fresh metavariable: ?m x y z := v"""
            mid, m = M(); r = k.unify_open(w._n(m), w._n(v), w.ctx)
            assert r[0] and k.meta_is_solved(mid), r
            return k.meta_solution_raw(mid)
        sol_y = solution_of_var(y)
        # F1: a metavariable at the bottom of a shared tower: solved on the way down, the rest of the tower is polynomial
        for k_ in (5, 20, 50):
            mid, m = M()
            ver, pend, steps, hits = unify_with_metas(w, tower(dup, m, k_), tower(dup2, y, k_))
            expect(ver is True and pend == 0, 'F1: dup^%d ?m = dup2^%d y: verdict %s, %d deferred' % (k_, k_, ver, pend))
            expect(k.meta_is_solved(mid) and k.meta_solution_raw(mid) == sol_y, 'F1: dup^%d ?m = dup2^%d y: ?m is %s' % (k_, k_, k.meta_solution_raw(mid) if k.meta_is_solved(mid) else 'unsolved'))
            expect(steps <= 4000 + 200 * k_, 'F1: dup^%d ?m = dup2^%d y took %d steps (the tree has 2^k leaves)' % (k_, k_, steps))
            if k_ >= 20: expect(hits > 0, 'F1: the unifier\'s table was never used for k = %d' % k_)
            # the same with a base that does not fit the second use of ?m:  hh ?m ?m = hh y z
            mid, m = M()
            ver, pend, steps, hits = unify_with_metas(w, tower(dup, m, k_), hh(tower(dup2, y, k_ - 1), tower(dup2, z, k_ - 1)))
            expect(ver is False, 'F1: dup^%d ?m = hh (dup2^%d y) (dup2^%d z) was accepted (?m would have to be y and z): verdict %s' % (k_, k_ - 1, k_ - 1, ver))
        # F2: a solution recorded between two visits of one pair keeps the table: share3 A ?p = share3 B y   visits (A, B) before and after ?p := y
        A = chain(w, y, n, x); B = chain(w, idn(y), n, x)
        mid, m = M()
        ver, pend, steps, hits = unify_with_metas(w, share3(A, m), share3(B, y))
        expect(ver is True and pend == 0 and k.meta_is_solved(mid) and k.meta_solution_raw(mid) == sol_y, 'F2: share3 A ?p = share3 B y: verdict %s, %d deferred, ?p %s' % (ver, pend, k.meta_solution_raw(mid) if k.meta_is_solved(mid) else 'unsolved'))
        expect(hits == 1, 'F2: share3 A ?p = share3 B y: the second visit of (A, B) after the solution of ?p was answered from the table %d times, expected once' % hits)
        # F3: an attempt that solved ?m, remembered pairs under that solution, and failed; proof irrelevance then takes the attempt back.  The pair visited again afterwards must be decided again
        # (and ?m solved again): enc n z = hq (ax4 n n z) n.  The proofs  ax4 A A x  and  ax4 B B z  are compared first (A, B twice, then x against z: different), the second argument of hq then meets (A, B) again.
        for failing in (False, True):
            mid, m = M()
            A_m = chain(w, m, n, x); B_y = chain(w, y, n, x)
            ver, pend, steps, hits = unify_with_metas(w, enc(A_m, x), enc(B_y, z if failing else x))
            expect(ver is True, 'F3[failing=%s]: enc A(?m) x = enc B(y) %s: verdict %s' % (failing, 'z' if failing else 'x', ver))
            expect(k.meta_is_solved(mid) and k.meta_solution_raw(mid) == sol_y,
                   'F3[failing=%s]: ?m is %s after the comparison, expected y (a pair remembered under a solution that was taken back must not stand for it)' % (failing, k.meta_solution_raw(mid) if k.meta_is_solved(mid) else 'unsolved'))
            expect(pend == 0, 'F3[failing=%s]: %d deferred constraints' % (failing, pend))
            expect(hits == (1 if failing else 2), 'F3[failing=%s]: the table answered %d sub-comparisons, expected %d' % (failing, hits, 1 if failing else 2))
        # F4: the same with a DEFERRED constraint instead of a solution: at the bottom of A  ?f (S y)  against  S (S y)  (not a pattern: deferred).  The failed attempt defers it, the rollback takes it back,
        # the second argument of hq defers it again: exactly one constraint is pending at the end
        mid, f = new_meta(w, w.arrow(Nat, Nat))
        A_f = chain(w, f(S(y)), n, x); B_s = chain(w, S(S(y)), n, x)
        ver, pend, steps, hits = unify_with_metas(w, enc(A_f, x), enc(B_s, z))
        expect(ver is True and pend == 1, 'F4: enc A(?f (S y)) x = enc B(S (S y)) z: verdict %s, %d deferred constraints, expected 1 (a constraint deferred in an attempt that was taken back must be deferred again)' % (ver, pend))
        ver, pend, steps, hits = unify_with_metas(w, enc(A_f, x), enc(B_s, x))
        expect(ver is True and pend >= 1, 'F4: enc A(?f (S y)) x = enc B(S (S y)) x: verdict %s, %d deferred constraints' % (ver, pend))
        # F5: ?m would have to be y in one use and z in the other: refused
        mid, m = M()
        ver, pend, steps, hits = unify_with_metas(w, hh3(chain(w, m, n, x), x, chain(w, m, n, x)), hh3(chain(w, y, n, x), x, chain(w, z, n, x)))
        expect(ver is False, 'F5: hh3 A(?m) x A(?m) = hh3 A(y) x A(z) was accepted (?m would have to be y and z): verdict %s' % ver)


# ---------------------------------------------------------------------------------------------------------------------------------------- G
def part_G(w):
    plus2, idn, mk, mk2, leaf, dup, dup2, share3 = w['plus2'], w['idn'], w['mk'], w['mk2'], w['leaf'], w['dup'], w['dup2'], w['share3']
    k = w.k
    k.set_option('verify_memo', True)
    try:
        with Ctx(w) as (x, y, z, s, t):
            v0, s0 = k.memo_verify_stats()
            cases = [
                (tower(plus2, x, 8), tower(plus2, idn(x), 8), True), (tower(plus2, x, 8), tower(plus2, y, 8), False),
                (tower(mk, leaf, 12), tower(mk2, leaf, 12), True), (tower(mk, leaf, 12), tower(mk2, s, 12), False),
                (tower(dup, x, 12), tower(dup2, idn(x), 12), True), (tower(dup, x, 12), tower(dup2, y, 12), False),
                (share3(chain(w, y, 30, x), x), share3(chain(w, idn(y), 30, x), x), True),
                (tower(plus2, x, 60), tower(plus2, idn(x), 60), True), (tower(dup, x, 60), tower(dup2, idn(x), 60), True),         # too big to be verified: counted as skipped
            ]
            for a, b, want in cases:
                for m in (False, True):
                    r = decide(w, a, b, m)
                    expect(r.ver is want and r.cver is want, 'G[syn %s]: %r, expected %s' % (m, r, want))
            v1, s1 = k.memo_verify_stats()
            expect(v1 > v0, 'G: no answer from a table or from the quick check was verified (%d -> %d)' % (v0, v1))
            expect(s1 > s0, 'G: no answer was too big to be verified (%d -> %d)' % (s0, s1))
            # the metavariable scenarios, verified
            Nat, S = w['Nat'], w['Succ']
            mid, m = new_meta(w, w.arrow(Nat, w.arrow(Nat, w.arrow(Nat, Nat))), (x, y, z))
            n = 30
            ver, pend, steps, hits = unify_with_metas(w, w['enc'](chain(w, m, n, x), x), w['enc'](chain(w, y, n, x), z))
            expect(ver is True and k.meta_is_solved(mid), 'G: enc A(?m) x = enc B(y) z with verification: verdict %s, solved %s' % (ver, k.meta_is_solved(mid)))
            v2, s2 = k.memo_verify_stats()
            expect(v2 > v1, 'G: the metavariable scenario verified nothing')
    finally:
        k.set_option('verify_memo', False)
        k.set_option('syn_conv', True)


# ---------------------------------------------------------------------------------------------------------------------------------------- H
def part_H(w):
    Nat, hh, hh3, dup, dup2, mk, mk2, node, leaf, idn, share3 = w['Nat'], w['hh'], w['hh3'], w['dup'], w['dup2'], w['mk'], w['mk2'], w['node'], w['leaf'], w['idn'], w['share3']
    k = w.k
    def norm(t):
        k.set_fuel(FUEL); h0 = k.quote_memo_hits()
        try: r = w.normalize(t); return r, k.fuel_used(), k.quote_memo_hits() - h0
        except E.FuelExhausted: return None, FUEL, k.quote_memo_hits() - h0
    def built(f, base, n):                  # the DAG the construction gives: f (f (... base)) with `f` applied as syntax (hash-consed: n nodes)
        t = base
        for _ in range(n): t = f(t)
        return t
    with Ctx(w) as (x, y, z, s, t):
        expect(k.quote_memo_size() == 0, 'H: the quotation table is not empty between calls: %d' % k.quote_memo_size())
        for n in (3, 10, 30, 60):
            r, steps, hits = norm(tower(dup, x, n))
            want = built(lambda u: hh(u, u), x, n)
            expect(r is not None and r == want, 'H: normalize (dup^%d x) is not hh (..) (..) built %d deep' % (n, n))
            expect(steps <= 2000 + 100 * n, 'H: normalize (dup^%d x) took %d steps (the tree has 2^n nodes)' % (n, steps))
            if n >= 10: expect(hits > 0, 'H: normalize (dup^%d x) did not use the table' % n)
            r, steps, hits = norm(tower(mk, leaf, n))
            want = built(lambda u: node(u, u), leaf, n)
            expect(r is not None and r == want, 'H: normalize (mk^%d leaf) is not node (..) (..) built %d deep' % (n, n))
            expect(steps <= 2000 + 100 * n, 'H: normalize (mk^%d leaf) took %d steps' % (n, steps))
            if n >= 10: expect(hits > 0, 'H: normalize (mk^%d leaf) did not use the table' % n)
        # two different heads / constructors with the same arguments must not share an answer:  hh V V  next to  hh3 V y V  and  node V V  next to  node V s
        V = tower(dup, y, 20); Vn = built(lambda u: hh(u, u), y, 20)
        for name, term, want in (
                ('hh3 V x V', hh3(V, x, V), hh3(Vn, x, Vn)), ('hh V (hh V V)', hh(V, hh(V, V)), hh(Vn, hh(Vn, Vn))),
                ('share3 V x', share3(V, x), hh3(Vn, x, Vn)),
                ('node (mk^15 leaf) (node (mk^15 leaf) s)', node(tower(mk, leaf, 15), node(tower(mk, leaf, 15), s)), node(built(lambda u: node(u, u), leaf, 15), node(built(lambda u: node(u, u), leaf, 15), s)))):
            r, steps, hits = norm(term)
            expect(r is not None and r == want, 'H: normalize (%s) is wrong' % name)
        # a lambda that holds a big shared value twice: the value is quoted under a binder (one deeper than where it was made)
        body = w.lam(Nat, lambda n: hh3(V, n, V))
        r, steps, hits = norm(body)
        want = w.lam(Nat, lambda n: hh3(Vn, n, Vn))
        expect(r is not None and r == want, 'H: normalize (fun n => hh3 V n V) is wrong')
        expect(steps <= 4000 + 200 * 20, 'H: normalize (fun n => hh3 V n V) took %d steps' % steps)
        shared = w.app(w.lam(Nat, lambda v: w.lam(Nat, lambda n: hh3(v, n, v))), V)         # the same, with V captured by the closure of an application: ONE value held twice
        r, steps, hits = norm(shared)
        expect(r is not None and r == want, 'H: normalize ((fun v n => hh3 v n v) V) is wrong')
        expect(hits > 0 and steps <= 4000 + 200 * 20, 'H: normalize ((fun v n => hh3 v n v) V): %d steps, %d table hits' % (steps, hits))
        # a stuck-match tower: the normal form is convertible with the term
        for n in (6, 20, 40):
            tm = tower(w['plus2'], x, n)
            r, steps, hits = norm(tm)
            expect(r is not None and w.conv(r, tm), 'H: normalize (plus2^%d x) is not convertible with it' % n)
            expect(steps <= 20000 + 200 * n ** 2, 'H: normalize (plus2^%d x) took %d steps' % (n, steps))
        expect(k.quote_memo_size() == 0, 'H: the quotation table is not empty after the calls: %d' % k.quote_memo_size())


# ---------------------------------------------------------------------------------------------------------------------------------------- I
def part_I(w):
    """I. A table of one comparison is GIVEN BACK when the comparison ends. `unordered_map::clear()` writes every bucket, empty or not, and keeps the bucket array, so a table that was once big would make every later
    comparison (and every later typing-memo flush) pay for its size. With the option verify_memo everything is remembered, so a comparison of two towers of 300 definitions fills the conversion, unifier and quotation
    tables with hundreds of entries; once the calls have returned no table may hold more than a few buckets (and some table really was that big: the kernel records the largest bucket count it released)."""
    plus2, idn, dup, dup2 = w['plus2'], w['idn'], w['dup'], w['dup2']
    k = w.k
    names = ('conversion/values', 'conversion/constructors', 'unifier/values', 'unifier/constructors', 'quotation', 'typing memo')
    verify_env = k.get_option('verify_memo'); k.set_option('verify_memo', True)
    try:
        with Ctx(w) as (x, y, z, s, t):
            r = decide(w, tower(plus2, x, 300), tower(plus2, idn(x), 300), False)
            expect(r.ver is True and r.cver is True, 'I: plus2^300 x = plus2^300 (idn x): %r' % r)
            r = decide(w, tower(dup, x, 150), tower(dup2, y, 150), False)           # refused: the quotation table is used
            expect(r.ver is False and r.cver is False, 'I: dup^150 x = dup^150 y was accepted: %r' % r)
            r = decide(w, tower(w['mk'], w['leaf'], 200), tower(w['mk2'], w['leaf'], 200), False)       # constructor values
            expect(r.ver is True and r.cver is True, 'I: mk^200 = mk2^200: %r' % r)
            b = k.memo_table_buckets()
            expect(b[6] > 256, 'I: no table ever got big (largest released %d buckets): the test would prove nothing' % b[6])
            for nm, n in zip(names[:5], b[:5]): expect(n <= 256, 'I: the %s table keeps %d buckets after the comparisons ended (largest released: %d)' % (nm, n, b[6]))
            # the typing memo: fill it with hundreds of types, install a function body (which makes every remembered type stale), type ONE term; the flush must give the buckets back
            for j in range(12, 400, 1): k.infer_open(w._n(tower(plus2, x, j)), w.ctx)
            filled = k.memo_stats()[2]; before = k.memo_table_buckets()[5]
            expect(filled > 256 and before > 256, 'I: the typing memo did not fill up (%d entries, %d buckets)' % (filled, before))
        Nat = w['Nat']; Z0 = w['Zero']
        w.recursive('probe_rec', w.arrow(Nat, Nat), lambda f: w.fun([Nat], lambda n: w.match(n, lambda _: Nat, [Z0, lambda m: f(m)])))
        with Ctx(w) as (x, y, z, s, t):
            k.infer_open(w._n(tower(plus2, x, 14)), w.ctx)
            expect(k.memo_table_buckets()[5] <= 256, 'I: the typing memo keeps %d buckets after its flush' % k.memo_table_buckets()[5])
    finally:
        k.set_option('verify_memo', verify_env)


def main():
    w = build()
    t0 = time.time()
    # the gate is also run with ITP_VERIFY_MEMO=1 (every answer from a table decided again): numbers that depend on which comparisons are remembered are then different, and parts A-F are run with it off
    verify_env = w.k.get_option('verify_memo')
    w.k.set_option('verify_memo', False)
    part_A(w); part_B(w); part_C(w); part_D(w); part_E(w); part_F(w); part_H(w); part_G(w); part_I(w)
    w.k.set_option('verify_memo', verify_env)
    print('%d checks (%.1f s)' % (CHECKS[0], time.time() - t0))
    if FAIL:
        print('S33 CONVMEMO GATE: FAIL (%d)' % len(FAIL))
        for f in FAIL[:25]: print('   ', f)
        sys.exit(1)
    print('S33 CONVMEMO GATE: PASS')


if __name__ == '__main__':
    main()
