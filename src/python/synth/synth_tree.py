"""synth_tree.py -- the synthetic theorem generator, first family: proof trees grown BACKWARDS from a lemma-shaped goal.

    gen = Grower(w, pool, limit)                      # w: the library world;  pool: synth_pool.Pool;  limit: the number of globals the proofs may use
    th = gen.theorem(rng, Plan(...))                  # a Theorem, or raises Reject(reason)

THE IDEA.  A proof of a theorem is a tree: the root is the goal, an inner node is a goal and the step that reduces it to its children, a leaf is a goal that is not reduced any further. Growing such a tree
at random, step by step, and then DECLARING THE UNPROVED LEAVES TO BE THE HYPOTHESES gives a theorem together with a proof, by construction:

    root goal      a lemma's conclusion with random terms for its variables           x | y * z + w
    inner node     the engine applies a library lemma to the goal (the same unifier an agent uses), the lemma's premises become the children
    leaf           closed by a decision procedure (ring, linarith, nlinarith) or by a premise-free lemma, or LIFTED: it becomes a hypothesis of the theorem
    theorem        forall (variables), leaf_1 -> ... -> leaf_m -> root

Every step is validated by the engine (`Engine.probe`), the finished proof is replayed from DATA in a fresh proof state (portable actions, the order in which an agent meets the goals) and its term
is checked by the kernel. Nothing here is trusted: a bug makes a theorem disappear (a `Reject`), never a false theorem appear.

The tree is what gives the theorem its DEPTH (the longest chain of steps), its WIDTH (leaves), the lemmas it needs and the directions of its steps; these are the labels the dataset carries.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import collections, itertools, math, random
import gym_env as G
import gym_acts as A
import gym_build as GB
from gym_env import Kind, Action, GoalData
import synth_terms as ST
import semantic as SEM
import synth_check as SC
import synth_poly as SPY


class Reject(Exception):
    """a growth attempt that produced no usable theorem; `reason` is a short stable string (the generator counts them)"""
    def __init__(self, reason): Exception.__init__(self, reason); self.reason = reason


class Disagreement(Exception):
    """the evaluator (semantic.py) and the kernel disagree about a theorem: never expected, always worth a look"""
    def __init__(self, text, prof): Exception.__init__(self, 'semantic disagreement on %s: %s' % (text[:120], prof)); self.text = text; self.prof = prof


class Plan:
    """what one theorem should be like: the lemma its root is made of, the depth to aim for, the number of operators of its terms, ..."""
    def __init__(self, main=None, depth=3, budget=None, nvars=3, ops=None, min_vars=2, min_depth=None, max_hyps=5, max_size=22, max_total=None, p_branch=0.6, min_sat=5, root_trials=60, max_kids=4, derive_rounds=3, root_candidates=8, root_combos=250):
        self.main = main; self.depth = depth; self.budget = budget if budget is not None else depth + 2
        self.nvars = nvars; self.ops = min(depth, 4) if ops is None else ops    # operators in the arguments of the root lemma (spread at random)
        self.min_vars = min_vars; self.min_depth = (max(2, depth - 1) if depth <= 5 else depth - 2) if min_depth is None else min_depth
        self.max_hyps = max_hyps; self.max_size = max_size; self.p_branch = p_branch
        self.area_weights = None; self.min_sat = min_sat; self.root_trials = root_trials; self.max_kids = max_kids; self.derive_rounds = derive_rounds; self.root_candidates = root_candidates; self.root_combos = root_combos
        self.max_total = max_total if max_total is not None else 40 + 9 * depth        # the size of the finished statement (in tokens: operators, variables, binders; a numeral is one)


class TNode:
    __slots__ = ('goal', 'depth', 'step', 'portable', 'kids', 'lemma', 'kind', 'spine', 'parent')
    def __init__(self, goal, depth, parent=None):
        self.goal = goal; self.depth = depth; self.step = None; self.portable = None; self.kids = []; self.lemma = None; self.kind = 'open'; self.spine = False; self.parent = parent
    # kind: 'open' (not decided yet) | 'step' (reduced by `step`) | 'closer' (closed by a decision procedure) | 'lift' (a hypothesis of the theorem)


class Theorem:
    """the product of a growth: the closed statement (a kernel handle), the portable proof, and what the tree says about it"""
    def __init__(self, stmt, steps, root, leaves, variables, info):
        self.stmt = stmt; self.steps = steps; self.root = root; self.leaves = leaves; self.variables = variables; self.info = info


def tsize(k, node, numeral=None, memo=None):
    """the size of a term as a tree (operators, binders and leaves), so that a shared subterm counts every time it is written; a numeral (`Z.of_N <binary literal>`, when its constant `numeral` is given) counts as one"""
    memo = {} if memo is None else memo
    def go(n):
        r = memo.get(n)
        if r is not None: return r
        v = k.view(n); t = v[0]
        if t == 'app':
            if numeral is not None and k.view(v[1]) == ('const', numeral): r = 1
            else: r = go(v[1]) + go(v[2])
        elif t in ('lam', 'pi'): r = go(v[1]) + go(v[2]) + 1
        else: r = 1
        memo[n] = min(r, 10_000); return memo[n]
    return go(node)


class Grower:
    def __init__(self, w, pool, limit, domain='Z', closers=(Kind.RING, Kind.LINARITH, Kind.NLINARITH)):
        self.w = w; self.k = w.k; self.pool = pool; self.limit = limit; self.domain = domain
        self.eng = G.Engine(w, kinds={Kind.APPLY} | set(closers), strict=True, premises=limit)
        self.vocab = ST.Vocab(w, **ST.VOCABS[domain]); self.tg = ST.TermGen(self.eng, self.vocab)
        self.closers = tuple(closers)
        self.lemmas = [l for l in pool.lemmas if l.domain == domain and not any(j in l.data for j in ())]
        self._open = {}
        self.stats = collections.Counter(); self.sigma = None
        self.var_names = 'abcdefgh'
        self.sem = SEM.Semantics(w, window_ex=60)                # (witnesses of an existential: a wide window, for the goals; the profile of a finished theorem uses a small one)
        self._ast = {}; self._specs = {}
        self.of_n = w.gid.get('Z.of_N') if domain == 'Z' else None
        self._roots = None

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ root
    def lemma_roots(self):
        """the lemmas a root can be made of: data binders all of the domain's number type (no proposition variables), a conclusion with an area"""
        if self._roots is None:
            out = []
            for l in self.lemmas:
                if l.domain != self.domain or l.negated or not l.premises or not l.data: continue
                doms, concl = G.open_binders(self.k, self.k.global_info(l.gid)[1], 0)
                if all(doms[j] == self.vocab.ty for j in l.data) and (self.domain != 'Z' or self.root_spec(l) is not None): out.append(l)
            self._roots = out
        return self._roots

    def size(self, node): return tsize(self.k, node, self.of_n)

    def pick_root(self, rng, area_weights=None):
        """a root lemma: an area is drawn first (by `area_weights`, default: every area that has a root lemma equally), then a lemma of the area uniformly, so that the large areas (order) do not crowd out the others"""
        roots = self.lemma_roots(); by = collections.defaultdict(list)
        for l in roots: by[l.area].append(l)
        areas = sorted(by)                                       # (sorted: the draw must not depend on the order of the library)
        wts = [(area_weights or {}).get(a, 1.0) for a in areas]
        return rng.choice(by[rng.choices(areas, weights=wts)[0]])

    def root_spec(self, lemma):
        """the premises of a root lemma as Python predicates: (the binders in order, 'd' data / 'p' proof, [(predicate, number of binders before the premise)]); None if the evaluator cannot read the lemma
        or a binder is not an integer. The binders may be interleaved (`forall p, prime p -> forall a, ...`): a premise sees the binders before it, a proof as None."""
        if lemma.gid in self._specs: return self._specs[lemma.gid]
        spec = None
        try:
            a = self.sem.decode(self.k.global_info(lemma.gid)[1]); entries = []; prem = []
            while a[0] in ('all', 'imp'):
                if a[0] == 'all':
                    if a[1] != 'Z': raise SEM.Unsupported('a binder that is not an integer')
                    entries.append('d'); a = a[2]
                else:
                    if a[1][0] in ('not', 'imp', 'all'): raise SEM.Unsupported('a premise that is a negation or has binders (its goal would live in a larger context)')
                    prem.append((self.sem.compile(a[1]), len(entries))); entries.append('p'); a = a[2]
            if a[0] != 'false' and entries.count('d') == len(lemma.data) and entries.count('p') == len(lemma.premises): spec = (tuple(entries), prem)
        except SEM.Unsupported:
            pass
        self._specs[lemma.gid] = spec
        return spec

    @staticmethod
    def premises_hold(spec, vals):
        entries, prem = spec; env = []; it = iter(vals)
        for e in entries: env.append(next(it) if e == 'd' else None)
        return all(f(tuple(env[:m])) for f, m in prem)

    def make_root(self, rng, plan, lemma):
        """(the statement `forall variables, root`, the variables' types and names, the codes the root lemma's data arguments were drawn with, the witness model).

        The WITNESS MODEL comes first: integers for the variables, and the lemma's arguments are drawn (as Python trees, so that hundreds of draws are cheap) until the lemma's premises hold at the model. The root
        is then true there, and so are the goals the first step leaves. Variables that no argument uses are dropped from the context."""
        w = self.w; k = self.k; ty = self.vocab.ty; vocab = self.vocab
        spec = self.root_spec(lemma)
        if spec is None: raise Reject('unsupported root lemma')
        nv = plan.nvars; data = list(lemma.data)
        total_ops = plan.ops + rng.randint(0, 2)
        for _ in range(plan.root_trials):
            sigma = tuple(rng.randint(-4, 5) for _ in range(nv))
            shares = [0] * len(data)
            for _ in range(total_ops): shares[rng.randrange(len(data))] += 1
            used = collections.Counter(); cands = []
            for share in shares:                                  # a few candidate terms for every argument; then the combinations are searched for one that makes the premises true at the model
                weights = {i: 1.0 / (1 + 3 * used[i]) for i in range(nv)}
                ts = [vocab.random_tree(rng, nv, share, p_op=0.85 if share else 0.0, p_num=0.12, weights=weights, no_unit=True) for _ in range(plan.root_candidates)]
                ts = [t for t in ts if t is not None]
                if not ts: break
                cands.append([(t, vocab.value(t, sigma)) for t in ts])
                for t in ts[:1]:
                    for i in SPY.tree_vars(t)[0]: used[i] += 1
            else:
                combos = list(itertools.product(*cands)) if math.prod(len(c) for c in cands) <= plan.root_combos else [tuple(rng.choice(c) for c in cands) for _ in range(plan.root_combos)]
                rng.shuffle(combos)
                for combo in combos:
                    trees = [t for t, _ in combo]
                    vs = set().union(*[SPY.tree_vars(t)[0] for t in trees])
                    if len(vs) < plan.min_vars: continue
                    if self.premises_hold(spec, tuple(v for _, v in combo)): break
                else:
                    continue
                break
        else:
            raise Reject('the root has no model')
        levels = sorted(vs); mapping = {old: new for new, old in enumerate(levels)}
        def remap(t): return ('v', mapping[t[1]]) if t[0] == 'v' else t if t[0] == 'n' else (t[0],) + tuple(remap(c) for c in t[1:])
        trees = [remap(t) for t in trees]; sigma = tuple(sigma[l] for l in levels)
        nv = len(levels); types = [ty] * nv; names = list(self.var_names[:nv])
        dummy = GoalData(w['True'].n, types, names)
        doms, concl = GB.open_pis(k, k.global_info(lemma.gid)[1])
        binding = {}; codes = {}
        for j, t in zip(data, trees):
            code, node = self.tg.build(dummy, ty, vocab.code(t))
            binding[GB.HOLE + j] = node; codes[j] = code
        root = GB.inst_beta(k, concl, binding)
        if not self.statement_tidy(root, types): raise Reject('untidy root')
        stmt = k.close_telescope('pi', types, root, 0)
        return stmt, types, names, codes, sigma

    def statement_tidy(self, node, types):
        """does a proposition (over the context `types`) read the way a person writes it?  No x + 0, x - 0, 0 - x, x * 1, x * 0, x - x, double negation, negated zero or arithmetic on two numerals (2 + 1)
        anywhere in it (lemma statements with a constant in them make such terms when an argument is a numeral or a negation), and no term in it that cancels or is a polynomial with fewer variables
        than it is written with"""
        k = self.k; w = self.w; seen = set(); roles = self.vocab.role
        zero = w.gid.get('Z.zero'); one = w.gid.get('Z.one')
        arithmetic_predicates = {w.gid[n] for n in ('Z.dvd', 'Z.coprime', 'Z.is_gcd', 'Z.prime') if n in w.gid}         # divisibility by zero, gcd(0, 0), "zero is prime": degenerate
        of_n = self.of_n; neg = w.gid.get('Z.neg')
        def is_c(n, gid): return gid is not None and k.view(n) == ('const', gid)
        def is_num(n):                                           # a numeral, or the negation of one
            v = k.view(n)
            if v[0] == 'const': return v[1] in (zero, one)
            if v[0] == 'app':
                if of_n is not None and k.view(v[1]) == ('const', of_n): return True
                if neg is not None and k.view(v[1]) == ('const', neg): return is_num(v[2])
            return False
        stack = [node]
        while stack:
            n = stack.pop()
            if n in seen: continue
            seen.add(n)
            v = k.view(n)
            if v[0] in ('app',):
                head, args = GB.spine(k, n); hv = k.view(head)
                if hv[0] == 'const' and hv[1] in arithmetic_predicates and any(is_c(a, zero) for a in args): return False
                if hv[0] == 'const' and hv[1] in roles:
                    role = roles[hv[1]]
                    if role in ('add', 'mul', 'sub') and all(is_num(a) for a in args): return False             # arithmetic on numerals (2 + 1): a person writes the number
                    if role == 'add' and any(is_c(a, zero) for a in args): return False
                    if role == 'mul' and any(is_c(a, zero) or is_c(a, one) for a in args): return False
                    if role == 'sub' and (is_c(args[0], zero) or is_c(args[1], zero) or args[0] == args[1]): return False
                    if role == 'neg':
                        if is_c(args[0], zero): return False
                        ah, aa = GB.spine(k, args[0])
                        if k.view(ah) == ('const', hv[1]): return False
                stack.extend(args); stack.append(head) if hv[0] != 'const' else None
            elif v[0] in ('lam', 'pi'): stack.append(v[1]); stack.append(v[2])
        if self.domain != 'Z': return True
        try: kinds, hyps, concl = self.parts_of(node, types)
        except SEM.Unsupported: return False
        return not self._cancels(concl, len(types))

    def _cancels(self, a, depth):
        """does a term of the proposition AST `a` (under `depth` binders) cancel or simplify?  (synth_poly.reducible)"""
        t = a[0]
        def term(x):
            if x[0] in ('int', 'bv'): return False
            p = SPY.ast_poly(x, depth)
            if p is None: return False
            vs, occ = SPY.ast_vars(x, depth)
            return SPY.reducible(p, vs, occ)
        if t == 'prim' and a[1] not in SPY.ARITH: return any(term(x) for x in a[2])
        if t == 'eq': return term(a[1]) or term(a[2])
        if t in ('and', 'or', 'iff'): return self._cancels(a[1], depth) or self._cancels(a[2], depth)
        if t == 'not': return self._cancels(a[1], depth)
        if t in ('imp', 'andd'): return self._cancels(a[1], depth) or self._cancels(a[2], depth + 1)
        if t in ('all', 'ex'): return self._cancels(a[2], depth + 1)
        return False

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ meaning
    def parts_of(self, target, types):
        """(binder kinds, hypotheses, conclusion) of the closed statement `forall types, target` as ASTs (cached for the theorem being grown); raises SEM.Unsupported"""
        key = (target, tuple(types))
        r = self._ast.get(key)
        if r is None:
            stmt = self.k.close_telescope('pi', list(types), target, 0)
            r = self._ast[key] = self.sem.parts(self.sem.decode(stmt))
        return r

    def truth_fn(self, target, types):
        """a Python predicate on tuples of integers for the proposition `target` over the context variables (the reference semantics of semantic.py); raises SEM.Unsupported"""
        kinds, hyps, concl = self.parts_of(target, types)
        if hyps: raise SEM.Unsupported('a goal with hypotheses')
        return self.sem.compile(concl)

    def true_under(self, target, types, sigma):
        try: return bool(self.truth_fn(target, types)(tuple(sigma)))
        except SEM.Unsupported: return None

    def looks_valid(self, target, types, rng, samples=80):
        """does the proposition hold at every one of `samples` random assignments of its variables (and at the witness model)?  False is certain (a counterexample), True is a strong hint that it is valid, None:
        the evaluator cannot say. Only a valid statement can be closed without hypotheses -- by a decision procedure or a premise-free lemma -- so an invalid one is never worth the attempt."""
        try: f = self.truth_fn(target, types)
        except SEM.Unsupported: return None
        n = len(types); special = (-2, -1, 0, 1, 2, 3)
        if self.sigma is not None and not f(tuple(self.sigma)): return False
        for _ in range(samples):
            env = tuple(rng.choice(special) if rng.random() < 0.4 else rng.randint(-6, 6) for _ in range(n))
            if not f(env): return False
        return True

    def draw_model(self, rng, targets, types, tries=400):
        """integers for the variables that make all the propositions true (None if none is found): the WITNESS MODEL of a theorem, which the growth keeps every goal true in"""
        fs = [self.truth_fn(t, types) for t in targets]
        for _ in range(tries):
            sigma = tuple(rng.choice((-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)) for _ in types)
            if all(f(sigma) for f in fs): return sigma
        return None

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ expansions
    def lemma_candidates(self, g, menu):
        """[(PoolLemma, action, portable)] for the pool lemmas that can be applied to the goal: plain APPLYs from the menu, and APPLY_WITH for the lemmas whose conclusion leaves data arguments open
        (their arguments are drawn at random by the caller)"""
        eng = self.eng; k = self.k
        out = []
        listed = {}
        for a in menu.actions:
            if a.kind == Kind.APPLY:
                c = menu.cands[a.arg][0]; v = k.view(c)
                if v[0] == 'const' and v[1] in self.pool.by_gid: listed[v[1]] = a
        for gid, a in listed.items():
            out.append((self.pool.by_gid[gid], a, A.to_portable(eng, menu, a)))
        return out

    def with_candidates(self, g):
        """the pool lemmas whose conclusion fits the goal's shape but leaves data arguments undetermined: [(PoolLemma, undetermined binder positions)]"""
        eng = self.eng; k = self.k; nh = g.n_hyps
        tnorm = k.simp_normalize(g.target, nh); seen = set(); out = []
        for c in k.library_candidates(eng.lib, tnorm, nh):
            v = k.view(c)
            if v[0] != 'const' or v[1] >= self.limit or v[1] in seen or v[1] not in self.pool.by_gid: continue
            seen.add(v[1])
            pr = eng.profile(v[1])
            if not pr.undetermined or pr.undetermined == (-1,): continue
            lem = self.pool.by_gid[v[1]]
            if any(j in lem.premises for j in pr.undetermined): continue
            out.append((lem, pr.undetermined))
        return out

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ subterms of a goal
    def goal_terms(self, g, ty):
        """the subterms of the goal's target that have the domain's number type (variables, numerals, results of the operators): where a meaningful intermediate term comes from"""
        k = self.k; vocab = self.vocab; out = []; seen = set()
        def go(n):
            if n in seen: return
            seen.add(n)
            v = k.view(n)
            if v[0] == 'fvar':
                if v[1] < g.n_hyps and g.hyp_types[v[1]] == ty: out.append(n)
                return
            if v[0] == 'app':
                head, args = GB.spine(k, n); hv = k.view(head)
                if hv[0] == 'const' and hv[1] in vocab.ops and len(args) == vocab.ops[hv[1]]: out.append(n)
                for a in args: go(a)
                go(head) if hv[0] == 'app' else None
        go(g.target)
        return out

    def draw_argument(self, g, ty, rng, plan):
        """the code of a term for an argument that the goal does not determine: a third of the time a subterm of the goal, a quarter a combination of two, else a small random term (a Python tree: the kernel
        sees the term only when the engine builds the action)"""
        r = rng.random()
        pool = self.goal_terms(g, ty) if r < 0.6 else []
        try:
            if pool and r < 0.35:
                return tuple(self.eng.space(g).encode(rng.choice(pool), ty))
            if len(pool) >= 2:
                a, b = rng.sample(pool, 2); space = self.eng.space(g)
                ops = [gid for gid, ar in sorted(self.vocab.ops.items()) if ar == 2]
                if ops: return (('c', rng.choice(ops), 2),) + tuple(space.encode(a, ty)) + tuple(space.encode(b, ty))
        except (GB.BuildError, A.ActionError, G.AuthorError, RuntimeError):
            pass
        t = self.vocab.random_tree(rng, g.n_hyps, rng.choice([0, 1, 1, 2]), no_unit=True)
        if t is None: t = ('v', rng.randrange(g.n_hyps))
        return tuple(self.vocab.code(t))

    def with_action(self, g, lem, undetermined, rng, plan, fixed=None):
        """APPLY_WITH for a lemma with undetermined data arguments: random arguments (their codes in the order of the binders), except those in `fixed` (binder position -> code)"""
        k = self.k
        doms, _ = G.open_binders(k, k.global_info(lem.gid)[1], 0)
        codes = []
        for j in undetermined:
            if doms[j] != self.vocab.ty: return None
            if fixed and j in fixed: codes.append(fixed[j]); continue
            codes.append(self.draw_argument(g, doms[j], rng, plan))
        return Action(Kind.APPLY_WITH, 0, (), ((('c', lem.gid, 0),),) + tuple(codes))

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the growth
    def try_expand(self, node, rng, plan, st, forced=None, fixed=None):
        """reduce a node by a pool lemma: the first candidate (in a random order, weighted) that the engine accepts and that passes the filters. Returns True if the node was reduced."""
        eng = self.eng; g = node.goal
        menu = eng.menu(g)
        cands = []                                              # (weight, lemma, action or None, undetermined)
        path = set(); p = node
        while p is not None:
            if p.lemma is not None: path.add(p.lemma.gid)
            p = p.parent
        for lem, a, port in self.lemma_candidates(g, menu):
            if lem.domain != self.domain: continue
            wgt = 1.0
            if lem.gid in path: wgt *= 0.25
            wgt *= (1.6 if lem.premises else 0.6) if node.spine else 1.0
            cands.append((wgt, lem, a, None))
        for lem, und in self.with_candidates(g):
            if lem.domain != self.domain: continue
            wgt = 0.7 * (0.25 if lem.gid in path else 1.0)
            cands.append((wgt, lem, None, und))
        if forced is not None:
            cands = [c for c in cands if c[1].gid == forced.gid]
            if not cands: self.stats['root_not_listed'] += 1
        tries = 0
        while cands and tries < 14:
            tries += 1
            i = rng.choices(range(len(cands)), weights=[c[0] for c in cands])[0]
            wgt, lem, a, und = cands.pop(i)
            for attempt in range(1 if a is not None else 12):
                if a is None:
                    act = self.with_action(g, lem, und, rng, plan, fixed if forced is not None else None)
                    if act is None: break
                else: act = a
                kids = eng.probe(g, act, menu)
                if kids is None: self.stats['probe_none'] += 1; continue
                if not self.accept(node, kids, plan): self.stats['filtered'] += 1; continue
                node.kind = 'step'; node.step = act; node.portable = A.to_portable(eng, menu, act); node.lemma = lem
                node.kids = [TNode(c, node.depth + 1, node) for c in kids]
                return True
        return False

    def accept(self, node, kids, plan):
        k = self.k; nv = node.goal.n_hyps
        if len(kids) > plan.max_kids: return False
        anc = set(); p = node
        while p is not None: anc.add(p.goal.target); p = p.parent
        for c in kids:
            if c.n_hyps != nv: return False
            if c.target in anc: return False
            if self.size(c.target) > plan.max_size: return False
            if not self.statement_tidy(c.target, c.hyp_types): self.stats['untidy_goal'] += 1; return False
            if self.sigma is not None:                           # every goal of the tree is true in the witness model (the proof is sound, so a false goal could never be proved from true hypotheses)
                if self.true_under(c.target, c.hyp_types, self.sigma) is not True: self.stats['false_in_model'] += 1; return False
        return True

    def finalize(self, node, rng):
        """a node that is not reduced further: closed by a premise-free lemma or a decision procedure, else lifted (it becomes a hypothesis of the theorem). A goal that fails at some assignment cannot be closed
        by either (they prove statements without hypotheses), so it is lifted at once; a goal that holds everywhere and that nothing closes would be a hypothesis that is a truth: the growth is abandoned."""
        eng = self.eng; g = node.goal
        valid = self.looks_valid(g.target, g.hyp_types, rng)
        if valid is False: node.kind = 'lift'; return
        menu = eng.menu(g)
        free = [(lem, a, port) for lem, a, port in self.lemma_candidates(g, menu) if not lem.premises]
        order = ['lemma', 'closer'] if rng.random() < 0.5 else ['closer', 'lemma']
        for what in order:
            if what == 'lemma':
                rng.shuffle(free)
                for lem, a, port in free:
                    kids = eng.probe(g, a, menu)
                    if kids is not None and not kids:
                        node.kind = 'step'; node.step = a; node.portable = port; node.lemma = lem; return
            else:
                for kind in self.closers:
                    acts = [a for a in menu.actions if a.kind == kind]
                    if not acts: continue
                    kids, status = eng.probe_status(g, acts[0], menu, fuel=eng.fuel_levels(kind)[1])
                    if kids is not None and not kids:
                        node.kind = 'closer'; node.step = acts[0]; node.portable = {'kind': Kind(kind).name}; return
        if valid is True: raise Reject('a hypothesis that is a truth')
        node.kind = 'lift'

    def grow(self, rng, plan, lemma):
        eng = self.eng; k = self.k
        stmt0, types, names, root_codes, sigma = self.make_root(rng, plan, lemma)
        ep = G.Episode(eng, self.w.T(stmt0), premises=self.limit)
        if ep.n_goals == 0: raise Reject('the root settles at once')
        g0 = ep.goal()
        if g0.n_hyps != len(types): raise Reject('root context')
        self.sigma = sigma
        if self.true_under(g0.target, g0.hyp_types, sigma) is not True: raise Reject('the root is false in its model')
        root = TNode(g0, 0); root.spine = True
        if not self.try_expand(root, rng, plan, None, forced=lemma, fixed=root_codes): raise Reject('root step')
        frontier = list(root.kids)
        for kid in root.kids: kid.spine = False
        if root.kids: rng.choice(root.kids).spine = True
        budget = plan.budget - 1; deepest = 1; lifted = []; root_size = self.size(g0.target)
        while frontier:
            spine = [n for n in frontier if n.spine]
            node = spine[0] if spine else rng.choice(frontier)
            frontier.remove(node)
            want = node.spine and node.depth < plan.depth
            if not want and node.depth < plan.depth + 1 and budget > 0:
                want = rng.random() < plan.p_branch * (0.8 ** (node.depth - 1))
            projected = root_size + sum(self.size(n.goal.target) for n in frontier) + sum(lifted)          # (what the statement would weigh if every open goal became a hypothesis)
            if want and projected > 1.3 * plan.max_total: want = False; self.stats['size_gate'] += 1
            if want and budget > 0 and self.try_expand(node, rng, plan, None):
                budget -= 1; deepest = max(deepest, node.depth + 1)
                if node.spine and node.kids:
                    for kid in node.kids: kid.spine = False
                    rng.choice(node.kids).spine = True
                frontier.extend(node.kids)
            else:
                was_spine = node.spine; node.spine = False
                self.finalize(node, rng)
                if node.kind == 'lift': lifted.append(self.size(node.goal.target))
                if was_spine and node.depth < plan.depth and frontier and not any(n.spine for n in frontier):      # the main line ended early: continue along the deepest other line
                    max(frontier, key=lambda n: n.depth).spine = True
        return root, types, names, stmt0, ep

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the theorem
    def preorder(self, node):
        out = []; stack = [node]
        while stack:
            n = stack.pop(); out.append(n); stack.extend(reversed(n.kids))
        return out

    def closed_by(self, hyp, target, n):
        """would the engine's settling close a goal `target` by the hypothesis `hyp` -- they are equal, or convertible within the settling budget? (the engine closes a goal by a hypothesis "up to computation":
        `0 < x` closes `1 <= x`, because `Z.lt` unfolds to it)"""
        if hyp == target: return True
        eng = self.eng; hs = eng.shape_fp(hyp); ts = eng.shape_fp(target)
        if hs is not None and ts is not None and hs != ts: return False
        k = self.k; saved = k.get_fuel(); k.set_fuel(G.SETTLE_FUEL)
        try: return eng._conv(hyp, target, n)
        finally: k.set_fuel(saved)

    def tidy(self, root):
        """make the tree consistent with the engine's settling: a goal that a lifted leaf closes (it is equal or convertible to it) is closed by assumption when the proof is replayed, so it can have no step of its own"""
        while True:
            nodes = self.preorder(root)
            lifts = [n for n in nodes if n.kind == 'lift']
            n = root.goal.n_hyps + len(lifts)
            changed = False
            for node in nodes:
                if node.kind in ('step', 'closer') and any(self.closed_by(l.goal.target, node.goal.target, n) for l in lifts):
                    if node.kind == 'closer': raise Reject('a closed goal is also a hypothesis')
                    node.kind = 'lift'; node.kids = []; node.step = None; node.portable = None; node.lemma = None; changed = True; break
            if not changed: return nodes

    def leaves_of(self, nodes, n_vars):
        """the distinct lifted leaves, in the order of the tree (leaves that the engine would close by another one -- equal or convertible -- are one)"""
        leaves = []; nlift = sum(1 for n in nodes if n.kind == 'lift')
        for n in nodes:
            if n.kind == 'lift' and not any(self.closed_by(l, n.goal.target, n_vars + nlift) for l in leaves): leaves.append(n.goal.target)
        return leaves

    def assemble(self, root, nodes, types, plan, rng):
        """the theorem of a tree whose leaves are decided: (statement, leaves, steps, info, semantic profile); raises Reject. The proof is replayed in a fresh proof state and checked by the kernel."""
        k = self.k; w = self.w; eng = self.eng
        leaves = self.leaves_of(nodes, len(types))
        if len(leaves) > plan.max_hyps: raise Reject('too many hypotheses')
        used = set(G.atoms_of(k, root.goal.target)[1])
        for l in leaves: used |= G.atoms_of(k, l)[1]
        if len(used) != len(types): raise Reject('an unused variable')
        if max([n.depth + 1 for n in nodes if n.kind == 'step'] or [0]) < plan.min_depth: raise Reject('too shallow')
        if not self.statement_tidy(root.goal.target, types) or not all(self.statement_tidy(l, types) for l in leaves): raise Reject('untidy statement')
        stmt = k.close_telescope('pi', types + leaves, root.goal.target, 0)
        if self.size(stmt) > plan.max_total: raise Reject('too big')
        steps = [n.portable for n in nodes if n.kind in ('step', 'closer')]
        try: A.replay_portable(eng, w.T(stmt), steps, premises=self.limit)
        except (A.ActionError, G.AuthorError, RuntimeError) as e: raise Reject('replay: ' + str(e)[:60])
        info = self.info(nodes, leaves, types, stmt); info['model'] = list(self.sigma)
        info['text'] = w.show(w.T(stmt)); info['digest'] = SC.digest(info['text']); info['pretty'] = SC.pretty(info['text'])
        old_wex = self.sem.wex; self.sem.wex = 8
        try: prof = SC.semantic_profile(self.sem, stmt, rng, model=self.sigma)
        finally: self.sem.wex = old_wex
        if prof is None: raise Reject('unsupported by the evaluator')
        if prof['disagreements']: e = Disagreement(info['text'], prof); e.stmt = stmt; e.model = self.sigma; raise e
        return stmt, leaves, steps, info, prof

    def derive(self, root, nodes, types, names, leaves, i):
        """make the hypothesis `leaves[i]` a consequence of the others: the goals that were lifted to it are closed by a decision procedure that reads the remaining hypotheses (linarith, nlinarith, ring).
        True if every such goal was closed (the tree is changed); else the tree is left as it was."""
        remaining = leaves[:i] + leaves[i + 1:]; target = leaves[i]
        hyp_types = list(root.goal.hyp_types[:len(types)]) + remaining
        hyp_names = list(names) + ['h%d' % j for j in range(len(remaining))]
        todo = [n for n in nodes if n.kind == 'lift' and self.closed_by(target, n.goal.target, len(hyp_types) + 1)]
        changes = []
        for n in todo:
            g = GoalData(n.goal.target, hyp_types, hyp_names)
            try: menu = self.eng.menu(g)
            except (G.AuthorError, RuntimeError): return False
            for kind in self.closers:
                acts = [a for a in menu.actions if a.kind == kind]
                if not acts: continue
                kids, status = self.eng.probe_status(g, acts[0], menu, fuel=self.eng.fuel_levels(kind)[1])
                if kids is not None and not kids: changes.append((n, acts[0], kind)); break
            else:
                return False
        for n, act, kind in changes:
            n.kind = 'closer'; n.step = act; n.portable = {'kind': Kind(kind).name}
        self.stats['derived'] += len(changes)
        return True

    def theorem(self, rng, plan, lemma=None):
        eng = self.eng; k = self.k; w = self.w
        self._ast.clear()
        if lemma is None: lemma = plan.main if plan.main is not None else self.pick_root(rng, plan.area_weights)
        root, types, names, stmt0, ep = self.grow(rng, plan, lemma)
        nodes = self.tidy(root)
        if root.kind == 'lift': raise Reject('the root is a hypothesis')
        derived = 0; before = None                                            # `before`: the tree and its result as they were before the last derivation
        for rounds in range(plan.derive_rounds + 1):
            try:
                stmt, leaves, steps, info, prof = self.assemble(root, nodes, types, plan, rng)
            except Reject as e:
                # a closer placed by an earlier derivation may have read a hypothesis that a later derivation made a consequence: the replay says so (finding F74); the tree before the last derivation is a theorem
                if before is None or not e.reason.startswith('replay'): raise
                for n, kind, step, portable in before[0]: n.kind = kind; n.step = step; n.portable = portable
                stmt, leaves, steps, info, prof = before[1]; derived -= 1; self.stats['derive_undone'] += 1
                break
            redundant = [i for i, need in enumerate(prof['needed']) if not need] if prof['exact'] and prof['needed'] else []
            if not redundant or rounds == plan.derive_rounds: break
            snapshot = [(n, n.kind, n.step, n.portable) for n in nodes]
            if not any(self.derive(root, nodes, types, names, leaves, i) for i in redundant): break
            before = (snapshot, (stmt, leaves, steps, info, prof)); derived += 1
        if prof['valid'] and leaves: raise Reject('the hypotheses are decoration')
        if any(prof['hyp_always']): raise Reject('a hypothesis that is a truth')
        if prof['sat_distinct'] < plan.min_sat: raise Reject('hardly satisfiable')
        if prof['exact'] and leaves and prof['n_needed'] < (len(leaves) + 1) // 2: raise Reject('hypotheses not needed')
        info['semantic'] = prof; info['derived'] = derived
        info['solved_by'] = SC.solver_profile(self.eng, w.T(stmt), self.limit)
        info['code'] = SC.encode_statement(self.eng, w.T(stmt), self.limit)
        return Theorem(stmt, steps, root, leaves, names, info)

    def info(self, nodes, leaves, types, stmt):
        k = self.k
        lem = [n.lemma.gid for n in nodes if n.kind == 'step' and n.lemma is not None]
        depth = max([n.depth + 1 for n in nodes if n.kind == 'step'] or [0])
        return dict(depth=depth, n_steps=sum(1 for n in nodes if n.kind in ('step', 'closer')), n_lemmas=len(set(lem)), lemmas=lem, n_hyps=len(leaves), n_vars=len(types),
                    closers=dict(collections.Counter(Kind(n.step.kind).name for n in nodes if n.kind == 'closer')), size=self.size(stmt))
