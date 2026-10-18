"""S26 gate: the facts every term node carries, the shared-term (DAG) traversals, the typing memo, and the fast proof extraction.

A proof produced by `ring` (or any tactic that builds the same normal form into dozens of lemma applications) is a DAG whose TREE is polynomially (or exponentially) larger. The kernel therefore

  * keeps, for every compound node, facts computed once when the node is created (`NodeInfo`: the loose bound-variable range, the context-variable range, the size as a tree, whether a metavariable is
    mentioned) and uses them instead of walking terms: every substitution is guarded by the loose range, the scope check at the API boundary is two comparisons, closing and renaming skip
    subterms that cannot contain what they look for;
  * traverses terms with a per-call memo keyed by (node, binder depth) once a traversal has grown big (`lift`, `instantiate`, `abstract_fvars`, `replace_fvars`, `resolve_metas`);
  * remembers the type found for a large metavariable-free term in a given context (`infer`), so a shared subterm is typed once per context, not once per occurrence;
  * extracts a proof term in time linear in the DAG when nothing in it can reduce (`extract_proof_term`).

All four are TRUSTED code (a wrong loose range makes a substitution skip a variable; a wrong memo key accepts an ill-typed term), so this gate pins them down by what they must do, against an oracle that shares
no code with them:

  A. the facts of every node, on hand-made terms (with the expected numbers) and on random shared terms (against a Python mirror), and the kernel's own audit of them (`check_node_info`);
     the facts survive a rewind (the tables are cut back with the pools) and are right for the nodes made afterwards;
  B. `lift`, `instantiate`, `abstract_fvars`, `replace_fvars` on random big shared terms (larger than the memo threshold, with the same node under different binder depths) against a Python reference;
  C. `resolve_metas` against a reference on terms whose metavariable solutions are applied under binders;
  D. the typing memo: the same term in two contexts of the same size with different hypotheses; a term whose tree is 2^60 nodes is typed at once and the memo is hit; a term that mentions a
     metavariable is never remembered; the answer follows a change of the `parameters` option; verification mode raises nothing on the whole corpus;
  E. extraction: a proof with nothing to reduce takes the linear path and equals evaluate-and-quote; a proof with a beta redex, a let, a match on a constructor, an applied fix or a Quot.lift on Quot.mk
     takes the slow path and comes out reduced; both paths give the same term wherever both apply;
  F. deep terms: a term 300000 nodes deep is a clean error from every traversal, never a crash.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, sys, time

sys.setrecursionlimit(100000)

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond

SIZE_MAX = 2**31 - 1


class Mirror:
    """The kernel's term constructors, with a Python mirror of what each node IS (its kind and children) and of its facts, computed here from the definition and not read from the kernel."""
    def __init__(self, k, nglobals=1):
        self.k = k; self.rec = {}; self.info = {}; self.nglobals = nglobals

    def fork(self):
        m = Mirror(self.k, self.nglobals); m.rec = dict(self.rec); m.info = dict(self.info); return m

    @staticmethod
    def join(parts):
        """parts: (info, binders the child sits under) -> info of the parent"""
        loose = fvars = 0; size = 1; meta = False
        for (l, f, s, m), under in parts:
            loose = max(loose, max(l - under, 0)); fvars = max(fvars, f); size += s; meta = meta or m
        return (loose, fvars, min(size, SIZE_MAX), meta)

    def _add(self, n, rec, info):
        self.rec[n] = rec; self.info[n] = info; return n
    def fvar(self, l):   return self._add(self.k.mk_fvar(l), ('fvar', l), (0, l + 1, 1, False))
    def bvar(self, i):   return self._add(self.k.mk_bvar(i), ('bvar', i), (i + 1, 0, 1, False))
    def const(self, g):  return self._add(self.k.mk_const(g), ('const', g), (0, 0, 1, False))
    def type_(self, l):  return self._add(self.k.mk_type(l), ('type', l), (0, 0, 1, False))
    def meta(self, m):   return self._add(self.k.mk_meta(m), ('meta', m), (0, 0, 1, True))
    def app(self, f, a): return self._add(self.k.mk_app(f, a), ('app', f, a), self.join([(self.info[f], 0), (self.info[a], 0)]))
    def lam(self, t, b): return self._add(self.k.mk_lambda(t, b), ('lam', t, b), self.join([(self.info[t], 0), (self.info[b], 1)]))
    def pi(self, d, c):  return self._add(self.k.mk_pi(d, c), ('pi', d, c), self.join([(self.info[d], 0), (self.info[c], 1)]))
    def let(self, t, v, b): return self._add(self.k.mk_let(t, v, b), ('let', t, v, b), self.join([(self.info[t], 0), (self.info[v], 0), (self.info[b], 1)]))
    def match(self, s, m, bs): return self._add(self.k.mk_match(s, m, list(bs)), ('match', s, m, tuple(bs)), self.join([(self.info[s], 0), (self.info[m], 0)] + [(self.info[b], 0) for b in bs]))
    def fix(self, ft, dec, body): return self._add(self.k.mk_fix(ft, dec, body), ('fix', ft, dec, body), self.join([(self.info[ft], 0), (self.info[body], 1)]))

    def children(self, n):
        r = self.rec[n]
        if r[0] in ('app', 'lam', 'pi'): return [(r[1], 0), (r[2], 0 if r[0] == 'app' else 0)]
        return []

    # ---- reference implementations of the substitution functions (declarative, memoised on (node, parameter); they build their results with the kernel's constructors, so equal terms are equal nodes)
    def rebuild(self, n, f):
        """the node n with f(child, binders crossed) applied to each child"""
        r = self.rec[n]
        if r[0] == 'app':   return self.app(f(r[1], 0), f(r[2], 0))
        if r[0] == 'lam':   return self.lam(f(r[1], 0), f(r[2], 1))
        if r[0] == 'pi':    return self.pi(f(r[1], 0), f(r[2], 1))
        if r[0] == 'let':   return self.let(f(r[1], 0), f(r[2], 0), f(r[3], 1))
        if r[0] == 'match': return self.match(f(r[1], 0), f(r[2], 0), [f(b, 0) for b in r[3]])
        if r[0] == 'fix':   return self.fix(f(r[1], 0), r[2], f(r[3], 1))
        return n

    def ref_lift(self, n, d, cutoff, memo=None):
        memo = {} if memo is None else memo
        key = (n, cutoff)
        if key in memo: return memo[key]
        r = self.rec[n]
        if r[0] == 'bvar': out = self.bvar(r[1] + d) if r[1] >= cutoff else n
        else: out = self.rebuild(n, lambda c, extra: self.ref_lift(c, d, cutoff + extra, memo))
        memo[key] = out; return out

    def ref_instantiate(self, n, vals, offset=0, memo=None):
        memo = {} if memo is None else memo
        key = (n, offset)
        if key in memo: return memo[key]
        r = self.rec[n]; cnt = len(vals)
        if r[0] == 'bvar':
            i = r[1]
            if i < offset: out = n
            elif i - offset < cnt: out = self.ref_lift(vals[i - offset], offset, 0)
            else: out = self.bvar(i - cnt)
        else: out = self.rebuild(n, lambda c, extra: self.ref_instantiate(c, vals, offset + extra, memo))
        memo[key] = out; return out

    def ref_abstract(self, n, levels, depth=0, memo=None):
        memo = {} if memo is None else memo
        key = (n, depth)
        if key in memo: return memo[key]
        r = self.rec[n]; cnt = len(levels)
        if r[0] == 'bvar': out = self.bvar(r[1] + cnt) if r[1] >= depth else n
        elif r[0] == 'fvar':
            out = n
            for i, l in enumerate(levels):
                if l == r[1]: out = self.bvar(depth + (cnt - 1 - i)); break
        else: out = self.rebuild(n, lambda c, extra: self.ref_abstract(c, levels, depth + extra, memo))
        memo[key] = out; return out

    def ref_replace(self, n, repl, depth=0, memo=None):
        memo = {} if memo is None else memo
        key = (n, depth)
        if key in memo: return memo[key]
        r = self.rec[n]
        if r[0] == 'fvar': out = self.ref_lift(repl[r[1]], depth, 0) if r[1] in repl else n
        else: out = self.rebuild(n, lambda c, extra: self.ref_replace(c, repl, depth + extra, memo))
        memo[key] = out; return out


def random_terms(mi, rng, steps, nconst, nmeta=0, leaf_bias=0.15, max_bvar=4, max_fvar=6):
    """a pool of `steps` nodes made from earlier pool members (so subterms are shared, and the same node sits under different numbers of binders)"""
    pool = []
    def leaf():
        c = rng.randrange(5 if nmeta else 4)
        if c == 0: return mi.fvar(rng.randrange(max_fvar))
        if c == 1: return mi.bvar(rng.randrange(max_bvar))
        if c == 2: return mi.const(rng.randrange(nconst))
        if c == 3: return mi.type_(rng.randrange(3))
        return mi.meta(rng.randrange(nmeta))
    for _ in range(4): pool.append(leaf())
    for _ in range(steps):
        pick = lambda: pool[max(0, len(pool) - 1 - int(rng.expovariate(0.08)))] if rng.random() < 0.7 else rng.choice(pool)
        r = rng.random()
        if r < leaf_bias: n = leaf()
        elif r < 0.50: n = mi.app(pick(), pick())
        elif r < 0.62: n = mi.lam(pick(), pick())
        elif r < 0.72: n = mi.pi(pick(), pick())
        elif r < 0.80: n = mi.let(pick(), pick(), pick())
        elif r < 0.90: n = mi.match(pick(), pick(), [pick() for _ in range(rng.randrange(1, 4))])
        else: n = mi.fix(pick(), rng.randrange(3), pick())
        pool.append(n)
    return pool


# ---------------------------------------------------------------------------------------------------------------------- A. node facts
def part_A(k, mi, rng, nconst):
    # hand-made terms with the numbers worked out by hand
    Z = mi.const(0); T = mi.type_(0)
    cases = [
        ('fvar 3',                           mi.fvar(3),                                                (0, 4, 1, False)),
        ('bvar 2',                           mi.bvar(2),                                                (3, 0, 1, False)),
        ('const',                            Z,                                                         (0, 0, 1, False)),
        ('app fvar5 bvar2',                  mi.app(mi.fvar(5), mi.bvar(2)),                           (3, 6, 3, False)),
        ('lam Z. bvar 0',                    mi.lam(Z, mi.bvar(0)),                                     (0, 0, 3, False)),
        ('lam Z. bvar 1 (one loose)',        mi.lam(Z, mi.bvar(1)),                                     (1, 0, 3, False)),
        ('lam bvar0. Z (loose in the domain)', mi.lam(mi.bvar(0), Z),                                   (1, 0, 3, False)),
        ('pi bvar0. bvar1',                  mi.pi(mi.bvar(0), mi.bvar(1)),                             (1, 0, 3, False)),
        ('pi bvar0. bvar0',                  mi.pi(mi.bvar(0), mi.bvar(0)),                             (1, 0, 3, False)),
        ('lam T. lam T. bvar 3',             mi.lam(T, mi.lam(T, mi.bvar(3))),                          (2, 0, 5, False)),
        ('let T := bvar0 in bvar0',          mi.let(T, mi.bvar(0), mi.bvar(0)),                         (1, 0, 4, False)),
        ('let T := Z in bvar 2',             mi.let(T, Z, mi.bvar(2)),                                  (2, 0, 4, False)),
        ('match fvar2 with [bvar1; Z]',      mi.match(mi.fvar(2), T, [mi.bvar(1), Z]),                 (2, 3, 5, False)),
        ('fix T. body bvar 1',               mi.fix(T, 0, mi.bvar(1)),                                  (1, 0, 3, False)),
        ('fix T. body bvar 0',               mi.fix(T, 0, mi.bvar(0)),                                  (0, 0, 3, False)),
        ('fix fn_type bvar 0',               mi.fix(mi.bvar(0), 0, Z),                                  (1, 0, 3, False)),
        ('meta',                             mi.meta(0),                                                (0, 0, 1, True)),
        ('app meta fvar',                    mi.app(mi.meta(0), mi.fvar(1)),                            (0, 2, 3, True)),
        ('lam under a meta',                 mi.lam(Z, mi.app(mi.bvar(0), mi.meta(1))),                 (0, 0, 5, True)),
    ]
    for name, n, want in cases:
        expect(k.node_info(n) == want, 'A node facts: %s: %s, expected %s' % (name, k.node_info(n), want))
        expect(k.check_node_info(n) == '', 'A audit: %s: %r' % (name, k.check_node_info(n)))
        expect(k.loose_range(n) == want[0], 'A loose_range: %s' % name)
    # the size of a tree saturates: t_{i+1} = f t_i t_i has 2^i leaves
    t = Z
    for i in range(45): t = mi.app(mi.app(mi.fvar(0), t), t)
    expect(k.node_info(t)[2] == SIZE_MAX, 'A size saturates at 2^31-1 (got %d)' % k.node_info(t)[2])
    expect(k.check_node_info(t) == '', 'A audit of a saturated size')
    t = Z
    for i in range(20): t = mi.app(mi.app(mi.fvar(0), t), t)
    expect(k.node_info(t)[2] == mi.info[t][2], 'A size of a shared term counts every occurrence (2^21 + ...) got %d want %d' % (k.node_info(t)[2], mi.info[t][2]))

    # random shared terms against the mirror
    bad = 0
    for trial in range(30):
        mj = mi.fork(); pool = random_terms(mj, rng, 160, nconst, nmeta=2)
        for n in pool[-60:]:
            if k.node_info(n) != mj.info[n]: bad += 1; FAIL.append('A random node facts: kernel %s mirror %s (%s)' % (k.node_info(n), mj.info[n], mj.rec[n][0]))
            if bad > 3: break
        audit = k.check_node_info(pool[-1])
        expect(audit == '', 'A random audit: %r' % audit)
        if bad > 3: break

    # rewind: the tables are cut back with the pools, and the nodes made afterwards get their own facts
    mj = mi.fork(); cp = k.checkpoint()
    first = random_terms(mj, rng, 120, nconst)
    k.rewind(cp)
    mk = mi.fork()                                                    # the mirror as it was at the checkpoint; the nodes of `first` are gone and their numbers are handed out again
    second = random_terms(mk, rng, 150, nconst, nmeta=1)
    bad = [n for n in second if k.node_info(n) != mk.info[n]]
    expect(not bad, 'A facts after a rewind: %d nodes differ, first: %s' % (len(bad), (k.node_info(bad[0]), mk.info[bad[0]], mk.rec[bad[0]][0]) if bad else ''))
    kinds = {mk.rec[n][0] for n in second}
    expect({'app', 'lam', 'pi', 'let', 'match', 'fix'} <= kinds, 'A the post-rewind corpus covers every compound kind (%s)' % sorted(kinds))
    expect(k.check_node_info(second[-1]) == '', 'A audit after a rewind')


# ---------------------------------------------------------------------------------------------------------------------- B. substitution functions against references
def part_B(k, mi, rng, nconst):
    n_checked = 0; deepest_shared = 0
    for trial in range(60):
        mj = mi.fork(); pool = random_terms(mj, rng, rng.choice([120, 200, 300]), nconst, max_bvar=5, max_fvar=6)
        t = pool[-1]
        # lift
        d, cutoff = rng.randrange(1, 4), rng.randrange(0, 3)
        got = k.lift(t, d, cutoff); want = mj.ref_lift(t, d, cutoff)
        n_checked += 1
        if not expect(got == want, 'B lift d=%d cutoff=%d trial %d' % (d, cutoff, trial)): break
        # instantiate
        nvals = rng.randrange(1, 4); vals = [rng.choice(pool[:40]) for _ in range(nvals)]
        got = k.instantiate(t, vals); want = mj.ref_instantiate(t, vals)
        n_checked += 1
        if not expect(got == want, 'B instantiate %d values trial %d' % (nvals, trial)): break
        # abstract_fvars
        levels = rng.sample(range(6), rng.randrange(1, 4))
        got = k.abstract_fvars(t, levels); want = mj.ref_abstract(t, levels)
        n_checked += 1
        if not expect(got == want, 'B abstract_fvars %s trial %d' % (levels, trial)): break
        # replace_fvars (replacements are closed terms)
        closed = [n for n in pool if mj.info[n][0] == 0]
        repl_levels = rng.sample(range(6), rng.randrange(1, 4)); repl = {l: rng.choice(closed) for l in repl_levels}
        got = k.replace_fvars(t, sorted(repl.items())); want = mj.ref_replace(t, repl)
        n_checked += 1
        if not expect(got == want, 'B replace_fvars %s trial %d' % (repl_levels, trial)): break
        # a second, larger variation: a shared subterm under many binders
        big = t
        for _ in range(3): big = mj.lam(mj.const(0), mj.app(big, big))
        got = k.instantiate(big, vals[:1]); want = mj.ref_instantiate(big, vals[:1])
        n_checked += 1
        if not expect(got == want, 'B instantiate under shared binders trial %d' % trial): break
        got = k.abstract_fvars(big, levels); want = mj.ref_abstract(big, levels)
        n_checked += 1
        if not expect(got == want, 'B abstract_fvars under shared binders trial %d' % trial): break
        expect(k.check_node_info(got) == '', 'B audit of an abstracted term')
    expect(n_checked > 300, 'B checked %d substitutions' % n_checked)

    # a shared subterm under 50 applications: the TREE has 2^50 nodes, the DAG has 50; every function must work on the DAG
    mj = mi.fork(); Z = mj.const(0)
    big = mj.app(mj.app(mj.fvar(0), mj.bvar(0)), mj.fvar(2))             # one loose bound variable, two context variables
    for _ in range(50): big = mj.app(big, big)
    for _ in range(3): big = mj.lam(Z, mj.app(big, mj.bvar(1)))
    jobs = [('lift', lambda: k.lift(big, 2, 0), lambda: mj.ref_lift(big, 2, 0)),
            ('lift with a cutoff', lambda: k.lift(big, 3, 2), lambda: mj.ref_lift(big, 3, 2)),
            ('instantiate', lambda: k.instantiate(big, [Z]), lambda: mj.ref_instantiate(big, [Z])),
            ('abstract_fvars', lambda: k.abstract_fvars(big, [0, 2]), lambda: mj.ref_abstract(big, [0, 2])),
            ('replace_fvars', lambda: k.replace_fvars(big, [(0, Z), (2, Z)]), lambda: mj.ref_replace(big, {0: Z, 2: Z})),
            ('check_node_info', lambda: k.check_node_info(big) == '', lambda: True)]
    for name, run, ref in jobs:
        t0 = time.time(); got = run(); dt = time.time() - t0
        expect(got == ref(), 'B %s on a shared term with 2^50 nodes is right' % name)
        expect(dt < 3.0, 'B %s on a shared term with 2^50 nodes takes %.2fs (it must not walk the tree)' % (name, dt))


# ---------------------------------------------------------------------------------------------------------------------- C. resolve_metas
def part_C(k, mi, rng, nconst, g):
    NAT = mi.const(g['Nat']); ZERO = mi.const(g['Zero']); AND = mi.const(g['And'])
    # ?f x0 x1 := And x1 (And x0 x0)    (a pattern: a metavariable applied to distinct context variables; built from a type constructor so that the unifier stores the solution exactly as written),
    # then used with arguments that are open and sit under binders
    mf = k.fresh_meta(k.close_telescope('pi', [NAT, NAT], NAT, 0), 0)
    mfm = mi.meta(mf)
    body_open = mi.app(mi.app(AND, mi.fvar(1)), mi.app(mi.app(AND, mi.fvar(0)), mi.fvar(0)))
    sol = k.close_telescope('lam', [NAT, NAT], body_open, 0)
    ok, msg, pend = k.unify_open(mfm, sol, [])
    expect(ok, 'C could not solve the metavariable by unification: %s' % msg)
    sol_body = mi.app(mi.app(AND, mi.bvar(0)), mi.app(mi.app(AND, mi.bvar(1)), mi.bvar(1)))     # the body of that lambda telescope: x1 is bvar 0, x0 is bvar 1
    expect(k.meta_solution_raw(mf) == k.dump_raw(sol), 'C the solution is stored as written (%s)' % k.meta_solution_raw(mf))

    def make_ref(mj):
        """the reference: every  ?f a b  becomes the body with its variables instantiated by b (the innermost, bvar 0) and a; everything else is rebuilt unchanged"""
        memo = {}
        def ref(n):
            if n in memo: return memo[n]
            r = mj.rec[n]
            if r[0] == 'app' and mj.rec[r[1]][0] == 'app' and mj.rec[r[1]][1] == mfm:
                out = mj.ref_instantiate(sol_body, [ref(r[2]), ref(mj.rec[r[1]][2])])
            else: out = mj.rebuild(n, lambda c, extra: ref(c))
            memo[n] = out; return out
        return ref

    # use sites: ?f a b with a, b big shared terms that mention loose bound variables, under lambdas
    for trial in range(25):
        mj = mi.fork()
        pool = random_terms(mj, rng, 140, nconst, max_bvar=3)
        a, b = pool[-1], pool[-2]
        wrapped = mj.app(mj.app(mfm, a), b)
        for _ in range(rng.randrange(0, 3)): wrapped = mj.lam(mj.const(g['Nat']), mj.app(wrapped, mj.bvar(0)))
        got = k.resolve_metas(wrapped); want = make_ref(mj)(wrapped)
        if not expect(got == want, 'C resolve_metas trial %d' % trial): break
        expect(k.check_node_info(got) == '', 'C audit of a resolved term')
    # a term with a metavariable that is not solved is left alone
    mu = k.fresh_meta(k.mk_const(g['Nat']), 0)
    t = mi.app(mi.app(AND, mi.meta(mu)), ZERO)
    expect(k.resolve_metas(t) == t, 'C an unsolved metavariable is left alone')
    # a solved metavariable inside the arguments of another use, and a use whose arguments are themselves uses (nested)
    mj = mi.fork(); inner = mj.app(mj.app(mfm, mj.fvar(0)), mj.bvar(0)); outer = mj.app(mj.app(mfm, inner), inner)
    got = k.resolve_metas(mj.lam(NAT, outer)); want = make_ref(mj)(mj.lam(NAT, outer))
    expect(got == want, 'C nested uses of a solved metavariable')
    # a term whose tree has 2^50 nodes: resolved in time linear in the DAG
    mj = mi.fork(); big = mj.app(mj.app(mfm, mj.fvar(0)), mj.bvar(0))
    for _ in range(50): big = mj.app(big, big)
    t0 = time.time(); got = k.resolve_metas(big); dt = time.time() - t0
    expect(got == make_ref(mj)(big), 'C resolve_metas on a shared term with 2^50 nodes')
    expect(dt < 3.0, 'C resolve_metas on a shared term takes %.2fs' % dt)


# ---------------------------------------------------------------------------------------------------------------------- D. the typing memo
def part_D(k, g):
    import prop_world
    mk = lambda n: k.mk_const(g[n])
    NAT, ZERO, SUCC, ADD, EQ, REFL = (mk(n) for n in ('Nat', 'Zero', 'Succ', 'add', 'Eq', 'refl'))
    app = k.mk_app
    T0, T1 = k.mk_type(0), k.mk_type(1)

    # D1. the same large term in two contexts of the same size: well typed in the first (x : Nat), ill typed in the second (x : Type 1)
    x = k.mk_fvar(0); t = x
    for _ in range(12): t = app(SUCC, t)
    ok1 = k.check_open(t, NAT, [NAT])[0]
    ok2 = k.check_open(t, NAT, [T1])[0]
    ok3 = k.check_open(t, NAT, [NAT])[0]
    expect(ok1 and not ok2 and ok3, 'D1 a term is typed in the context it is in (got %s %s %s)' % (ok1, ok2, ok3))
    # ... also through infer_open, which answers with the type
    r1 = k.infer_open(t, [NAT]); r2 = k.infer_open(t, [T1])
    expect(r1[0] and not r2[0], 'D1 infer_open follows the context')
    # a hypothesis that is a function in one context and a number in the other
    f = k.mk_fvar(0); u = app(f, ZERO)
    for _ in range(14): u = app(f, u)
    fnty = k.mk_pi(NAT, NAT)
    a1 = k.infer_open(u, [fnty]); a2 = k.infer_open(u, [NAT]); a3 = k.infer_open(u, [fnty])
    expect(a1[0] and not a2[0] and a3[0], 'D1b a large application depends on the type of the variable (got %s %s %s)' % (a1[0], a2[0], a3[0]))

    # D2. a term whose TREE has 2^60 nodes is typed at once, and the memo is what makes it possible
    d = k.mk_fvar(0)
    for _ in range(60): d = app(app(ADD, d), d)
    h0, m0, e0 = k.memo_stats(); t0 = time.time()
    r = k.check_open(d, NAT, [NAT]); dt = time.time() - t0
    h1, m1, e1 = k.memo_stats()
    expect(r[0], 'D2 the doubling term is well typed (%s)' % (r,))
    expect(dt < 5.0, 'D2 the doubling term is typed in %.1fs (it must not walk the tree)' % dt)
    expect(h1 - h0 >= 50, 'D2 the memo is hit on shared subterms (%d hits)' % (h1 - h0))
    expect(e1 - e0 >= 50 and e1 - e0 <= 300, 'D2 about one entry per distinct subterm (%d)' % (e1 - e0))
    # ... and an ill-typed shared term is still refused (the root is `add d (Succ-less ...)`: a Type where a Nat is expected)
    bad = k.mk_fvar(0)
    for _ in range(40): bad = app(app(ADD, bad), bad)
    bad = app(app(ADD, bad), T0)
    expect(not k.check_open(bad, NAT, [NAT])[0], 'D2 an ill-typed shared term is refused')
    # ... and a failure inside a shared subterm is not remembered as a success
    inner = app(app(ADD, ZERO), T0)
    shared = app(app(ADD, inner), inner)
    for _ in range(10): shared = app(app(ADD, shared), shared)
    expect(not k.check_open(shared, NAT, []) [0] and not k.check_open(shared, NAT, [])[0], 'D2 a failure is not remembered as a success')

    # D3. a term that mentions a metavariable is never remembered
    m = k.fresh_meta(NAT, 0); mm = k.mk_meta(m); d = mm
    for _ in range(11): d = app(app(ADD, d), d)
    h0, m0, e0 = k.memo_stats()
    r = k.check_partial(d, NAT, [])
    h1, m1, e1 = k.memo_stats()
    expect(r[0], 'D3 the term with a metavariable is well typed')
    expect(e1 == e0 and h1 == h0, 'D3 nothing about a term with a metavariable is remembered (entries %d -> %d, hits %d -> %d)' % (e0, e1, h0, h1))
    # a hypothesis that mentions a metavariable makes the context unfit for the memo, too: the (large) term below is well typed only with the metavariable's solution, so nothing about it may be remembered
    m2 = k.fresh_meta(k.mk_type(1), 0); hyp = k.mk_meta(m2)                      # x : ?m2   (a type, unsolved)
    t = k.mk_fvar(0)
    for _ in range(20): t = app(app(ADD, t), t)
    h0, m0, e0 = k.memo_stats()
    k.check_partial(t, NAT, [hyp])
    h1, m1, e1 = k.memo_stats()
    expect(e1 == e0 and h1 == h0, 'D3b nothing is remembered in a context whose hypothesis mentions a metavariable (entries %d -> %d)' % (e0, e1))

    # D4. the memo follows the options: theorems of the prop world are well typed with `parameters` on and refused with it off, again and again
    flips = 0
    for name in ('transport', 'symm', 'trans', 'cong', 'le_trans'):
        info = k.global_info(g[name]); ty, val = info[1], info[2]
        seq = []
        for params in (True, False, True, False):
            k.set_option('parameters', params)
            seq.append((k.infer_open(val, [])[0], k.check_open(val, ty, [])[0]))
        k.set_option('parameters', True)
        expect(seq == [(True, True), (False, False), (True, True), (False, False)], 'D4 %s follows the option: %s' % (name, seq))
        flips += 1
    expect(flips == 5, 'D4 ran')


# ---------------------------------------------------------------------------------------------------------------------- E. extraction
def part_E(k, g):
    mk = lambda n: k.mk_const(g[n])
    NAT, ZERO, SUCC, ADD, EQ, REFL = (mk(n) for n in ('Nat', 'Zero', 'Succ', 'add', 'Eq', 'refl'))
    app = k.mk_app; one = app(SUCC, ZERO); two = app(SUCC, one)
    eq = lambda a, b: app(app(app(EQ, NAT), a), b)

    def solution(stmt, term):
        ps = k.start_proof(stmt); ps.exact(term)
        assert ps.is_complete(); return ps.extract()
    f0, s0 = k.extract_stats()
    # E1. nothing to reduce: the linear path; the proof is returned as it is
    plain = app(app(REFL, NAT), two)
    r = solution(eq(two, two), plain)
    f1, s1 = k.extract_stats()
    expect(r == plain, 'E1 a proof with nothing to reduce is returned as it is')
    expect(f1 == f0 + 1 and s1 == s0, 'E1 it took the linear path (%d fast, %d slow)' % (f1 - f0, s1 - s0))
    # E2. a beta redex: the slow path, and it comes out reduced
    redex = app(k.mk_lambda(NAT, app(app(REFL, NAT), k.mk_bvar(0))), two)
    r = solution(eq(two, two), redex)
    f2, s2 = k.extract_stats()
    expect(r == plain, 'E2 a beta redex is contracted by extraction (got %s)' % k.dump_raw(r))
    expect(s2 == s1 + 1 and f2 == f1, 'E2 it took the slow path')
    # E3. a let
    let = k.mk_let(NAT, two, app(app(REFL, NAT), k.mk_bvar(0)))
    r = solution(eq(two, two), let)
    f3, s3 = k.extract_stats()
    expect(r == plain, 'E3 a let is contracted by extraction (got %s)' % k.dump_raw(r))
    expect(s3 == s2 + 1, 'E3 it took the slow path')
    # E4. a match on a constructor: iota
    motive = k.mk_lambda(NAT, eq(two, two))
    mt = k.mk_match(one, motive, [plain, k.mk_lambda(NAT, plain)])
    r = solution(eq(two, two), mt)
    f4, s4 = k.extract_stats()
    expect(r == plain, 'E4 a match on a constructor is contracted by extraction (got %s)' % k.dump_raw(r))
    expect(s4 == s3 + 1, 'E4 it took the slow path')
    # E5. a fixpoint applied to a constructor (unfolded when applied)
    fix_ty = k.mk_pi(NAT, eq(two, two))
    fx = k.mk_fix(fix_ty, 0, k.mk_lambda(NAT, plain))                      # fix f : Nat -> (2 = 2) := fun n => refl 2
    r = solution(eq(two, two), app(fx, one))
    f5, s5 = k.extract_stats()
    expect(r == plain, 'E5 a fix applied to a constructor is reduced by extraction (got %s)' % k.dump_raw(r))
    expect(s5 == s4 + 1, 'E5 it took the slow path')
    # E6. a big proof with sharing and nothing to reduce is extracted on the linear path in no time (its tree has 2^50 nodes): p_{i+1} = trans Nat x x x p_i p_i  proves  x = x
    TRANS = mk('trans'); x = k.mk_fvar(0)
    big = app(app(REFL, NAT), x)
    for _ in range(50): big = app(app(app(app(app(app(TRANS, NAT), x), x), x), big), big)
    stmt = k.mk_pi(NAT, app(app(app(EQ, NAT), k.mk_bvar(0)), k.mk_bvar(0))); t0 = time.time()
    ps = k.start_proof(stmt); ps.intro('x'); ps.exact(big); assert ps.is_complete()
    t_in = time.time() - t0; r = ps.extract(); t_out = time.time() - t0
    f6, s6 = k.extract_stats()
    expect(f6 == f5 + 1 and s6 == s5, 'E6 a shared proof takes the linear path (%d fast, %d slow)' % (f6 - f5, s6 - s5))
    expect(t_out < 5.0, 'E6 checking and extracting a proof with a 2^50-node tree takes %.2fs' % t_out)
    ok, msg = k.check_open(r, stmt, [])
    expect(ok, 'E6 the extracted proof is accepted by the final check: %s' % msg)
    # E7. the unfinished proof: a hole is left as a hole
    ps = k.start_proof(eq(two, two)); e = ps.extract()
    expect(k.dump_raw(e).startswith('?') or 'M' in k.dump_raw(e), 'E7 an unfinished proof extracts as its hole (%s)' % k.dump_raw(e))
    # E8. with the verification on, both paths are compared on everything extracted
    k.set_option('verify_memo', True)
    try:
        r = solution(eq(two, two), plain); r = solution(eq(two, two), redex)
        small = app(app(REFL, NAT), x)                                    # verification re-derives every hit without the memo: keep the tree small (2^9 nodes)
        for _ in range(9): small = app(app(app(app(app(app(TRANS, NAT), x), x), x), small), small)
        ps = k.start_proof(stmt); ps.intro('x'); ps.exact(small); r = ps.extract()
        expect(k.check_open(r, stmt, [])[0], 'E8 the extracted proof is accepted')
    except Exception as ex:
        expect(False, 'E8 verification raised on a correct extraction: %s' % ex)
    finally:
        k.set_option('verify_memo', False)


def part_E9():
    """E9. Quot.lift applied to Quot.mk is a redex that no other rule looks like: it is reduced by extraction (slow path); the same lift applied to a hypothesis is stuck and comes out as it went in"""
    import s16_quot_tests as Q16
    from harness import build_new
    from named import HY, ap
    k, g, _ = Q16.std_world(); T = Q16.Terms(k, g)
    B = lambda t: build_new(k, HY(t))
    f0, s0 = k.extract_stats()
    # the redex  K0 (mk Nat total Zero)  :  Nat,  with K0 := Quot.lift Nat total Nat (fun _ => Zero) h
    redex = B(ap(T.K0, T.mk(T.NAT, T.total, T.ZERO)))
    ps = k.start_proof(B(T.NAT)); ps.exact(redex)
    expect(ps.is_complete(), 'E9 the redex solves its goal')
    r = ps.extract(); f1, s1 = k.extract_stats()
    expect(r == B(T.ZERO), 'E9 Quot.lift on Quot.mk is contracted by extraction (got %s)' % k.dump_raw(r))
    expect(s1 == s0 + 1 and f1 == f0, 'E9 it took the slow path (%d fast, %d slow)' % (f1 - f0, s1 - s0))
    # the same redex nested inside a proof of  Eq Nat Zero Zero  (the redex is an argument of refl): contracted in place
    stmt = B(T.eq(T.NAT, T.ZERO, T.ZERO)); inner = k.mk_app(k.mk_app(B(T.REFL), B(T.NAT)), redex)
    ps = k.start_proof(stmt); ps.exact(inner); r = ps.extract(); f2, s2 = k.extract_stats()
    plain = k.mk_app(k.mk_app(B(T.REFL), B(T.NAT)), B(T.ZERO))
    expect(r == plain, 'E9 a Quot.lift redex inside a proof is contracted in place (got %s)' % k.dump_raw(r))
    expect(s2 == s1 + 1 and f2 == f1, 'E9 nested: slow path')
    # stuck: the lift applied to a hypothesis stays as it is (and is still found by the scan, so the slow path runs and gives the same term back)
    qt = B(T.QT); stuck_stmt = k.mk_pi(qt, B(T.NAT))
    ps = k.start_proof(stuck_stmt); ps.intro('q'); ps.exact(k.mk_app(B(T.K0), k.mk_fvar(0))); r = ps.extract(); f3, s3 = k.extract_stats()
    want = k.mk_lambda(qt, k.mk_app(B(T.K0), k.mk_bvar(0)))
    expect(r == want, 'E9 a stuck Quot.lift is returned unchanged (got %s)' % k.dump_raw(r))
    expect(s3 == s2 + 1 and f3 == f2, 'E9 stuck: slow path')
    expect(k.check_open(r, stuck_stmt, [])[0], 'E9 the stuck proof is accepted')
    # with the verification on, both paths are compared on these as well
    k.set_option('verify_memo', True)
    try:
        ps = k.start_proof(B(T.NAT)); ps.exact(redex); r = ps.extract()
        expect(r == B(T.ZERO), 'E9 verified extraction of the redex')
    except Exception as ex:
        expect(False, 'E9 verification raised: %s' % ex)
    finally:
        k.set_option('verify_memo', False)


# ---------------------------------------------------------------------------------------------------------------------- the memo audit over a corpus
def part_G(k, g, seed):
    """verification mode: every hit of the memo is recomputed from scratch and compared; the library's theorems are checked again with it on"""
    k.set_option('verify_memo', True)
    try:
        n = 0
        for name in ('transport', 'symm', 'trans', 'cong', 'le_trans', 'zero_le', 'add'):
            info = k.global_info(g[name]); ty, val = info[1], info[2]
            r = k.check_open(val, ty, []); n += 1
            expect(r[0], 'G %s checks with the memo verification on: %s' % (name, r[1]))
    except Exception as ex:
        expect(False, 'G the memo verification raised: %s' % ex)
    finally:
        k.set_option('verify_memo', False)


# ---------------------------------------------------------------------------------------------------------------------- F. deep terms: a clean error, never a crash (run in a child process)
def part_F_child():
    import prop_world as PW, itp_env_lnl as E
    k, g = PW.build_prop_world()
    mi = Mirror(k); Z = mi.const(g['Zero']); S = mi.const(g['Succ'])
    depth = 300000
    results = []
    def op(name, f):
        try: f(); results.append((name, 'ok'))
        except (RuntimeError, E.ITPError) as ex: results.append((name, 'error:' + str(ex)[:40]))
    for shape in ('app', 'lam'):
        t = Z
        for _ in range(depth): t = mi.app(S, t) if shape == 'app' else mi.lam(Z, t)
        # the facts are constant time and need no recursion at all
        op(shape + ' node_info', lambda: k.node_info(t))
        op(shape + ' loose_range', lambda: k.loose_range(t))
        op(shape + ' check_node_info', lambda: k.check_node_info(t))
        op(shape + ' lift', lambda: k.lift(t, 1))
        op(shape + ' replace_fvars', lambda: k.replace_fvars(t, [(0, Z)]))
        op(shape + ' resolve_metas', lambda: k.resolve_metas(t))
        op(shape + ' abstract_fvars', lambda: k.abstract_fvars(t, [0]))
        ps = k.start_proof(k.mk_const(g['True']))
        op(shape + ' extract', lambda: (ps.exact(k.mk_const(g['tt'])), ps.extract()))
    print('RESULTS', results)


def main():
    # The verification mode re-derives every memo hit without the memo, which is exponential on exactly the terms this gate is about (2^60-node trees): it must not be switched on for the whole process
    # from outside (ITP_VERIFY_MEMO=1 is how the OTHER gates are run with it). This gate turns it on itself, where the terms are small (E8, E9, G).
    os.environ.pop('ITP_VERIFY_MEMO', None)
    if '--child-deep' in sys.argv:
        part_F_child(); return
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t0 = time.time()
    import prop_world as PW
    k, g = PW.build_prop_world(); rng = random.Random(seed)
    mi = Mirror(k); nconst = k.num_globals()
    for name, fn in (('A', lambda: part_A(k, mi, rng, nconst)), ('B', lambda: part_B(k, mi, rng, nconst)), ('C', lambda: part_C(k, mi, rng, nconst, g)),
                     ('D', lambda: part_D(k, g)), ('E', lambda: part_E(k, g)), ('E9', part_E9), ('G', lambda: part_G(k, g, seed))):
        t1 = time.time()
        try: fn()
        except Exception as ex:
            import traceback; traceback.print_exc()
            expect(False, '%s raised %s: %s' % (name, type(ex).__name__, str(ex)[:200]))
        print('  part %s done in %.1fs (%d checks so far)' % (name, time.time() - t1, CHECKS[0]), flush=True)
    # F: deep terms in a child process
    import subprocess
    r = subprocess.run([sys.executable, os.path.abspath(__file__), '--child-deep'], capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or 'RESULTS' not in r.stdout:
        expect(False, 'F the child process died (exit %s): %s' % (r.returncode, (r.stderr or r.stdout)[-300:]))
    else:
        res = eval(r.stdout.split('RESULTS', 1)[1].strip())
        for name, outcome in res:
            expect(outcome == 'ok' or outcome.startswith('error:'), 'F %s: %s' % (name, outcome))
        print('  part F: %d operations on a 300000-deep term, none crashed (%d refused cleanly)' % (len(res), sum(1 for _, o in res if o != 'ok')), flush=True)
    print('%d checks in %.1fs, seed %d' % (CHECKS[0], time.time() - t0, seed))
    if FAIL:
        print('S26 MEMO GATE: FAIL (%d)' % len(FAIL))
        for f in FAIL[:25]: print('   ', f)
        sys.exit(1)
    print('S26 MEMO GATE: PASS')


if __name__ == '__main__':
    main()
