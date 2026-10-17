"""gym_env.py -- the learner's environment on the hybrid kernel: goals as DATA, a menu of legal actions, atomic probes and replay.

Nothing here is trusted. A step of a proof is an action applied to the first open goal by the tactic layer (tactics.py), which builds ordinary proof terms; whatever the search believes, the proof it
finally commits is re-checked by the kernel (`Engine.verify`). So a bug in this file can waste work or lose a proof, never make a false theorem appear.

THE MODEL.  A search works on GOALS, not on proof states: a `GoalData` is a target and the types and names of the hypotheses (kernel handles), nothing else, so two searches that reach the same goal
by different routes reach the same object (`key`). Trying an action is a PROBE:

    checkpoint -> make a one-goal proof state from the goal data -> run the action (tactics.py) -> read the goals it left -> rewind

`Kernel.finish_probe` reads the goals, rewinds EVERYTHING since the checkpoint (the proof-term scaffolding, metavariables, the scratch goal) and rebuilds the surviving goals from a dump, so a probe
leaves the kernel exactly as it was plus the nodes of the new goals. A failed action leaves it exactly as it was. Nothing is kept from a probe but those goals.

WHAT A PROBE MAY NOT DO.  A rewind reclaims every kernel node made since the checkpoint, so no Python object may keep a handle made during a probe: the caches of the authoring layer are
`NodeCache`s, which ignore writes while `World.probing` is set, and the engine warms them before it starts. Handles the engine keeps (the menu's candidates, the goals) are made outside probes.

THE ACTIONS (`Kind`).  Every action acts on the first open goal and is a finite choice: a kind and an argument that indexes either the hypotheses (cases, induction) or the menu's candidate table
(apply, rewrite, interval_cases):
    INTRO            introduce the first quantifier or hypothesis of the goal (offered only when the engine does not introduce them by itself: `auto_intro`)
    CASES, INDUCT    case analysis / induction on a hypothesis (argument: its position)
    APPLY            apply a lemma or a hypothesis (candidates from the library index; only lemmas all of whose data arguments the goal determines: the goals it leaves are then independent)
    REWRITE          rewrite by an equation, left to right or right to left, at the first instance of its pattern (lemmas whose pattern determines their variables, and equations among the hypotheses)
    RING, LINARITH, NLINARITH   close the goal by the decision procedures of ring.py / linarith.py (kernel-checked certificates)
    BY_CONTRA        classical: assume the negation of the goal and prove False
    INTERVAL_CASES   split an integer variable that the hypotheses bound on both sides into its finitely many values
After every action the goals it left are SETTLED, identically in probes and in replay, so a search never spends a step on them: a goal that a hypothesis (assumption) or reflexivity closes is closed,
and (with `auto_intro`) the quantifiers and implications at the head of a goal that is left are introduced.
"""
import os, sys, enum, collections, contextlib, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import itp_env_lnl as E
from author import AuthorError, Term, NodeCache
from tactics import Proof, HOLE_BASE
import ring as R


OBS_TAG_NAMES = E.OBS_TAG_NAMES
TAG = {n: i for i, n in enumerate(OBS_TAG_NAMES)}


class Kind(enum.IntEnum):
    """the kinds of action; the numbers are part of the data format (training examples store them)"""
    INTRO = 0
    CASES = 1
    INDUCT = 2
    APPLY = 3
    REWRITE = 4
    RING = 5
    LINARITH = 6
    NLINARITH = 7
    BY_CONTRA = 8
    INTERVAL_CASES = 9


KIND_NAMES = tuple(k.name for k in Kind)
NEEDS_CANDIDATE = frozenset({Kind.APPLY, Kind.REWRITE, Kind.INTERVAL_CASES})
NEEDS_HYPOTHESIS = frozenset({Kind.CASES, Kind.INDUCT})


class Action(collections.namedtuple('Action', 'kind arg')):
    """an action: its kind and its argument (a hypothesis position for CASES / INDUCT, an index into the menu's candidates for APPLY / REWRITE / INTERVAL_CASES, else 0)"""
    __slots__ = ()
    def __repr__(self): return '%s(%d)' % (KIND_NAMES[self.kind], self.arg)


class ActionError(Exception):
    """an action that could not be applied (the tactic refused it, or ran out of budget)"""


STRICT_DEFAULT = False       # engines made without `strict` take this (the tests and the fuzzer set it)
REFUSALS = (ActionError, AuthorError, E.ITPError, RuntimeError)      # what a probe may legitimately see go wrong: the tactic layer's and the kernel's own refusals (nanobind turns the kernel's logic errors into RuntimeError)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ goals
def goal_key(target, hyp_types):
    """the identity of a goal: the kernel's terms are hash-consed, so equal terms are the same node and this key is exact (the hypotheses' names are display data and are not part of it)"""
    return (target,) + tuple(hyp_types)


class GoalData:
    """one goal: target and hypothesis types (kernel handles), the hypotheses' names, an exact `key`, and the observation (computed on first use)"""
    __slots__ = ('target', 'hyp_types', 'hyp_names', 'key', 'obs')
    def __init__(self, target, hyp_types, hyp_names):
        self.target = target; self.hyp_types = list(hyp_types); self.hyp_names = list(hyp_names)
        self.key = goal_key(target, self.hyp_types); self.obs = None
    @property
    def n_hyps(self): return len(self.hyp_types)


class Menu:
    """the legal actions of a goal and the candidates they refer to. `cands[i]` is (term, rev, aux): the kernel handle of a lemma or hypothesis (None for INTERVAL_CASES), whether a REWRITE goes right
    to left, and for INTERVAL_CASES the triple (hypothesis position, lowest value, highest value). `cand_tags` / `cand_scalars` are the (tag, scalar) features of the terms (as `Kernel.term_features`),
    `cand_flags` is 1 for a right-to-left rewrite."""
    __slots__ = ('actions', 'cands', 'cand_tags', 'cand_scalars', 'cand_flags')
    def __init__(self, actions, cands, cand_tags, cand_scalars, cand_flags):
        self.actions = actions; self.cands = cands; self.cand_tags = cand_tags; self.cand_scalars = cand_scalars; self.cand_flags = cand_flags
    def signature(self, action):
        """what an action means independently of the menu it came from: (kind, the candidate's node or hypothesis position, direction, range)"""
        if action.kind in NEEDS_CANDIDATE:
            term, rev, aux = self.cands[action.arg]; return (int(action.kind), term, rev, aux)
        return (int(action.kind), action.arg, False, None)
    def find(self, signature):
        """the action with this signature in this menu, or None"""
        for a in self.actions:
            if self.signature(a) == signature: return a
        return None


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ reading terms
def spine_raw(k, n):
    """(head, arguments) of an application, on kernel handles"""
    args = []
    while True:
        v = k.view(n)
        if v[0] != 'app': break
        args.append(v[2]); n = v[1]
    args.reverse()
    return n, args


def unfolds_to_pi(k, ty, steps=8):
    """is the type (looking through definitions at its head) a quantifier or implication?"""
    cur = ty
    for _ in range(steps):
        if k.view(cur)[0] == 'pi': return True
        cur = k.unfold_head(cur)
        if cur is None: return False
    return False


def conclusion_head(k, ty):
    """what the conclusion of a type is headed by, after peeling its quantifiers and implications (a definition at the head that hides one -- `Not P` is `P -> False` -- is unfolded):
    ('c', global id), ('v', context level), ('b', bound index: a variable of the type itself), or None when it cannot be told"""
    cur = ty
    for _ in range(256):
        v = k.view(cur)
        if v[0] == 'pi': cur = v[2]; continue
        h, _ = spine_raw(k, cur); hv = k.view(h)
        if hv[0] == 'const':
            nxt = k.unfold_head(cur)
            if nxt is not None and k.view(nxt)[0] == 'pi': cur = nxt; continue
            return ('c', hv[1])
        if hv[0] == 'fvar': return ('v', hv[1])
        if hv[0] == 'bvar': return ('b', hv[1])
        return None
    return None


def heads_compatible(a, b):
    """can two conclusion heads (as `conclusion_head`) belong to a lemma and a goal that `apply` could match? Variables of the lemma itself (kind 'b') match anything"""
    if a is None or b is None or a[0] == 'b' or b[0] == 'b': return True
    return a == b


def atoms_of(k, node):
    """(the set of global ids, the set of context levels) that occur in a term"""
    tags, scalar = k.observe_term(node)[:2]
    return set(scalar[tags == TAG['CONST']].tolist()), set(scalar[tags == TAG['FVAR']].tolist())


def open_binders(k, ty, base, limit=64):
    """the leading quantifiers of `ty` opened as context variables base, base + 1, ...: (the domains, in terms of the variables before them, and the conclusion). A definition at the head that
    hides a quantifier is unfolded."""
    doms = []; cur = ty
    while len(doms) < limit:
        v = k.view(cur)
        if v[0] != 'pi':
            nxt = k.unfold_head(cur)
            if nxt is None or k.view(nxt)[0] != 'pi': break
            v = k.view(nxt)
        doms.append(v[1]); cur = k.instantiate(v[2], [k.mk_fvar(base + len(doms) - 1)])
    return doms, cur


class _MatchShim:
    """`Proof._match` needs only `self.k`: the matching `rw` does is used here as it is, so that the menu offers exactly the rewrites that can find their instance"""
    def __init__(self, k): self.k = k
    match = Proof._match


def subterms_by_head(k, root):
    """the closed subterms of a term (no loose bound variable, the ones `rw` looks at), grouped by the global id at their head; {} entries only for heads that are constants"""
    out = {}; seen = set(); stack = [root]
    while stack:
        t = stack.pop()
        if t in seen: continue
        seen.add(t)
        if k.loose_range(t) == 0:
            h, _ = spine_raw(k, t); hv = k.view(h)
            if hv[0] == 'const': out.setdefault(hv[1], []).append(t)
        v = k.view(t); kind = v[0]
        if kind == 'app': stack.append(v[2]); stack.append(v[1])
        elif kind in ('lam', 'pi'): stack.append(v[2]); stack.append(v[1])
        elif kind == 'let': stack += [v[3], v[2], v[1]]
        elif kind == 'match': stack += list(reversed(v[3])) + [v[2], v[1]]
        elif kind == 'fix': stack += [v[3], v[1]]
    return out


class Profile:
    """what an `apply` or a `rw` of a statement needs to know, read off its type once. `undetermined`: the DATA quantifiers that the conclusion does not determine (they would stay as unsolved
    metavariables shared by the goals that are left: the goals would not be independent); `rw`: the directions in which an equation can be used by `rw` (its pattern contains every quantifier),
    as (reverse?, head symbol of the pattern as a global id, the pattern with its quantifiers as holes: context variables from HOLE_BASE up)"""
    __slots__ = ('n', 'undetermined', 'rw')
    def __init__(self, n, undetermined, rw):
        self.n = n; self.undetermined = undetermined; self.rw = rw


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the engine
PROBE_FUEL = {Kind.RING: 50_000_000, Kind.LINARITH: 50_000_000, Kind.NLINARITH: 100_000_000, Kind.INTERVAL_CASES: 50_000_000}
DEFAULT_PROBE_FUEL = 5_000_000
SETTLE_FUEL = 200_000          # the budget of each conversion check that settles a goal (is it an assumption? is it `x = x`?)
MAX_INTROS = 64


class Engine:
    """goals, menus, probes and replay for ONE world. `premises`: the number of globals (by id) that actions may use -- a lemma declared later is not offered, so a theorem is attacked with
    the library that precedes it; None means all. `auto_intro`: introduce the quantifiers and implications at the head of every goal (so INTRO is not an action); `autoclose`: close goals that a
    hypothesis or reflexivity proves; `clean`: after `cases` / `induct` remove the dead copies they leave behind (the variable that was split, the hypotheses they reverted and brought back, the
    copies of the family's parameters), so that goals that differ only by such copies are the same goal. `kinds`: the kinds of action the menu offers (all by default). `strict`: an exception inside a
    probe that is not a refusal of the tactic layer or of the kernel (a TypeError, a stale checkpoint, ...) is a BUG of this code: it is always kept in `bugs`, and a strict engine raises it (after the
    kernel has been rewound); a lenient one counts it as a failed action, so a long search survives it. A refusal that is mistaken for a bug costs a search nothing and hides nothing; a bug that is mistaken
    for a refusal hides every action that hits it, silently."""
    def __init__(self, w, premises=None, auto_intro=True, autoclose=True, clean=True, probe_fuel=None, kinds=None, strict=None):
        if w.ctx: raise AuthorError('an engine must be made at the top level')
        self.w = w; self.k = w.k; self.auto_intro = auto_intro; self.autoclose = autoclose; self.clean = clean
        self.kinds = frozenset(Kind) if kinds is None else frozenset(kinds)       # the kinds of action the menu offers
        self.premises = premises
        self.probe_fuel = dict(PROBE_FUEL); self.default_fuel = DEFAULT_PROBE_FUEL if probe_fuel is None else int(probe_fuel)
        self.strict = STRICT_DEFAULT if strict is None else strict
        self.stats = collections.Counter()
        self.errors = collections.Counter()              # exception types seen while probing, by name (a type other than the tactic layer's own refusals is a bug: see `bugs`)
        self.bugs = []                                   # (type name, message, traceback) of the first few exceptions that were not refusals
        self._profiles = {}; self._scanned = 0; self._rw_index = {}      # global id -> Profile;  head global id -> [(lemma global id, reverse?)]
        self._hyp_profiles = NodeCache(w); self._families = NodeCache(w); self._recursive = {}
        self.refresh()

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- set-up
    @property
    def limit(self):
        n = self.k.num_globals()
        return n if self.premises is None else min(self.premises, n)

    @contextlib.contextmanager
    def in_context(self, types, names=None):
        """terms built (and tactics of the world run) inside the `with` see `types` as the context"""
        w = self.w; saved = (w.ctx, w.ctx_names)
        w.ctx = list(types); w.ctx_names = list(names) if names is not None else ['x%d' % i for i in range(len(types))]
        try: yield
        finally: w.ctx, w.ctx_names = saved

    def refresh(self):
        """(re)build everything that depends on the library: the index of lemmas, the profiles, the equation index, and the caches that probes only read (handles made inside a probe are reclaimed with
        it, so they are made here, once). Call it again after the world has grown."""
        w = self.w; k = self.k
        for name in list(w.gid): w.const(name)
        self.lib = k.index_library()
        for tyname in ('Nat', 'N', 'Z'):
            if tyname in w.gid:
                inst = R.instance_for(w, w[tyname])
                if inst is not None:
                    for v in range(-2, 17):
                        try: inst.lit(v)
                        except AuthorError: pass
        self._scan()

    def reset_caches(self):
        """forget everything keyed by kernel handles (call it after anything has rewound the kernel below nodes the engine has seen)"""
        self._hyp_profiles.clear(); self._families.clear()

    def _scan(self):
        k = self.k
        for gid in range(self._scanned, k.num_globals()):
            pr = self.profile(gid)
            for rev, head, pat in pr.rw: self._rw_index.setdefault(head, []).append((gid, rev, pat))
        self._scanned = k.num_globals()

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- profiles
    def profile(self, gid):
        pr = self._profiles.get(gid)
        if pr is None: pr = self._profiles[gid] = self._profile_of(self.k.global_info(gid)[1], [], 0, rewriting=True)
        return pr

    def _profile_of(self, ty, ctx_types, base, rewriting=False):
        """the profile of a statement `ty` that lives in the context `ctx_types` (`base` variables)"""
        k = self.k; w = self.w; PROP = w.PROP.n
        try:
            doms, concl = open_binders(k, ty, base)
            n = len(doms)
            occ = [atoms_of(k, d)[1] for d in doms]
            concl_fv = atoms_of(k, concl)[1]
            determined = {j for j in range(n) if base + j in concl_fv}
            changed = True
            while changed:                          # a quantifier that occurs in the type of a determined one is determined with it (f_equal: A occurs in the type of f)
                changed = False
                for j in list(determined):
                    for lv in occ[j]:
                        i = lv - base
                        if 0 <= i < n and i not in determined: determined.add(i); changed = True
            undetermined = []
            for j in range(n):
                if j in determined: continue
                ok, sort = k.infer_open(doms[j], list(ctx_types) + doms[:j])
                if ok and sort == PROP: continue                  # a hypothesis of the lemma: a goal of its own
                undetermined.append(j)
            rw = []
            if rewriting:
                with self.in_context(list(ctx_types) + doms):
                    try: A, lhs, rhs = w.eq_parts(w.T(concl))
                    except AuthorError: A = None
                if A is not None:
                    for rev, pat in ((False, lhs), (True, rhs)):
                        head, _ = spine_raw(k, pat.n); hv = k.view(head)
                        if hv[0] != 'const': continue             # a bare variable (or a lambda, a match) as the pattern matches everything: not offered
                        fv = atoms_of(k, pat.n)[1]
                        if all(base + j in fv for j in range(n)):
                            holes = [(base + j, k.mk_fvar(HOLE_BASE + j)) for j in range(n)]
                            rw.append((rev, hv[1], k.replace_fvars(pat.n, holes) if holes else pat.n))
            return Profile(n, tuple(undetermined), tuple(rw))
        except (E.ITPError, AuthorError, RuntimeError):
            return Profile(0, (-1,), ())                    # unreadable: never offered

    def hyp_profile(self, g, h):
        key = (g.hyp_types[h], h)
        pr = self._hyp_profiles.get(key)
        if pr is None:
            pr = self._profile_of(g.hyp_types[h], g.hyp_types[:h], h, rewriting=True)
            self._hyp_profiles[key] = pr
        return pr

    def family_of(self, ty):
        """the global id of the inductive family a type is an instance of -- looking through definitions at its head -- or None"""
        k = self.k; hit = self._families.get(ty)
        if hit is not None: return hit if hit >= 0 else None
        cur = ty; res = -1
        for _ in range(32):
            h, _ = spine_raw(k, cur); hv = k.view(h)
            if hv[0] == 'const' and k.is_inductive(hv[1]): res = hv[1]; break
            cur = k.unfold_head(cur)
            if cur is None: break
        self._families[ty] = res
        return res if res >= 0 else None

    def split_consumes(self, ty):
        """does splitting a hypothesis of this type use it up? Yes when the family has no indices, or its indices are distinct variables (the split then substitutes for them); with an index that is a
        term the kernel only rewrites that term in the goal, and the hypothesis stays valuable"""
        k = self.k; cur = ty
        for _ in range(32):
            h, args = spine_raw(k, cur); hv = k.view(h)
            if hv[0] == 'const' and k.is_inductive(hv[1]):
                slots = k.param_slots(hv[1])
                if len(slots) != len(args): return False
                seen = set(); params = set()
                for a, is_param in zip(args, slots):
                    if is_param: params |= atoms_of(k, a)[1]
                for a, is_param in zip(args, slots):
                    if is_param: continue
                    v = k.view(a)
                    if v[0] != 'fvar' or v[1] in seen or v[1] in params: return False
                    seen.add(v[1])
                return True
            cur = k.unfold_head(cur)
            if cur is None: return False
        return False

    def is_recursive_family(self, fam):
        r = self._recursive.get(fam)
        if r is None:
            k = self.k; r = False
            for c in k.global_info(fam)[5]:
                doms, _ = open_binders(k, k.global_info(c)[1], 0)
                if any(fam in atoms_of(k, d)[0] for d in doms): r = True; break
            self._recursive[fam] = r
        return r

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- goals
    def goal_of(self, p, i=0):
        """goal i of a live proof state as data"""
        ps = p.ps; n = ps.goal_num_hyps(i)
        return GoalData(ps.goal_target(i), [ps.goal_hyp_type(i, h) for h in range(n)], list(p._names[ps.goal_meta_id(i)]))

    def observe(self, g):
        """the observation of a goal (computed once): the DAG arrays tags, scalar, child_a, child_b, roots, heights -- see itp_bindings.cpp"""
        if g.obs is None: g.obs = self.k.observe_goal(g.target, g.hyp_types)
        return g.obs

    def render(self, g):
        w = self.w; lines = []
        for i, (nm, ty) in enumerate(zip(g.hyp_names, g.hyp_types)): lines.append('%s : %s' % (nm or '_', w.show(w.T(ty), names=g.hyp_names[:i])))
        lines.append('|- ' + w.show(w.T(g.target), names=g.hyp_names))
        return '\n'.join(lines)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- the menu
    def menu(self, g):
        """the legal actions of a goal, with their candidate table. Nothing here is tried: an action may still fail when probed."""
        w = self.w; k = self.k; limit = self.limit; nh = g.n_hyps
        actions = []; cands = []
        if not self.auto_intro and unfolds_to_pi(k, g.target): actions.append(Action(Kind.INTRO, 0))
        for h in range(nh):
            fam = self.family_of(g.hyp_types[h])
            if fam is None: continue
            actions.append(Action(Kind.CASES, h))
            if self.is_recursive_family(fam): actions.append(Action(Kind.INDUCT, h))
        # --- apply: the library through its index (looked up with the display normal form of the target, which is how lemmas are stated), the hypotheses by the head of their conclusion
        tnorm = k.simp_normalize(g.target, nh)
        gh = conclusion_head(k, g.target)
        seen = set()
        for c in k.library_candidates(self.lib, tnorm, nh):
            v = k.view(c)
            if v[0] != 'const' or v[1] >= limit or v[1] in seen: continue
            seen.add(v[1])
            if self.profile(v[1]).undetermined: continue
            cands.append((c, False, None)); actions.append(Action(Kind.APPLY, len(cands) - 1))
        for h in range(nh):
            if not heads_compatible(conclusion_head(k, g.hyp_types[h]), gh): continue
            if self.hyp_profile(g, h).undetermined: continue
            cands.append((k.mk_fvar(h), False, None)); actions.append(Action(Kind.APPLY, len(cands) - 1))
        # --- rewrite: the equations whose pattern occurs in the goal (the very test `rw` makes: first-order matching against the closed subterms; the probe still has the last word)
        t_consts, t_fvars = atoms_of(k, g.target)
        heads = subterms_by_head(k, g.target); shim = _MatchShim(k)
        def occurs(head, pat): return any(shim.match(pat, t, {}) for t in heads.get(head, ()))
        for head in sorted(t_consts):
            for gid, rev, pat in self._rw_index.get(head, ()):
                if gid >= limit or not occurs(head, pat): continue
                cands.append((k.mk_const(gid), rev, None)); actions.append(Action(Kind.REWRITE, len(cands) - 1))
        for h in range(nh):
            pr = self.hyp_profile(g, h)
            for rev, head, pat in pr.rw:
                if occurs(head, pat): cands.append((k.mk_fvar(h), rev, None)); actions.append(Action(Kind.REWRITE, len(cands) - 1))
            if pr.n == 0:                                          # a plain equation with a context variable as a side (x = f y): the pattern is that variable
                with self.in_context(g.hyp_types[:h], g.hyp_names[:h]):
                    try: A, lhs, rhs = w.eq_parts(w.T(g.hyp_types[h]))
                    except AuthorError: continue
                for rev, pat in ((False, lhs), (True, rhs)):
                    pv = k.view(pat.n)
                    if pv[0] == 'fvar' and pv[1] in t_fvars: cands.append((k.mk_fvar(h), rev, None)); actions.append(Action(Kind.REWRITE, len(cands) - 1))
        # --- decision procedures and classical reasoning
        with self.in_context(g.hyp_types, g.hyp_names):
            if self._is_ring_goal(g): actions.append(Action(Kind.RING, 0))
            if self._arith_ok(g):
                actions.append(Action(Kind.LINARITH, 0)); actions.append(Action(Kind.NLINARITH, 0))
            if 'by_contra' in w.gid and self._classical_ok(g): actions.append(Action(Kind.BY_CONTRA, 0))
            intervals = self._interval_candidates(g)
        for (h, lo, hi) in intervals:
            cands.append((None, False, (h, lo, hi))); actions.append(Action(Kind.INTERVAL_CASES, len(cands) - 1))
        if len(self.kinds) < len(Kind):                     # drop the kinds that are not offered, and the candidates only they used (renumbering the others)
            keep = [a for a in actions if a.kind in self.kinds]
            used = sorted({a.arg for a in keep if a.kind in NEEDS_CANDIDATE}); remap = {old: new for new, old in enumerate(used)}
            actions = [Action(a.kind, remap[a.arg]) if a.kind in NEEDS_CANDIDATE else a for a in keep]; cands = [cands[i] for i in used]
        nodes = [c[0] for c in cands if c[0] is not None]
        feats = iter(zip(*[a.tolist() for a in k.term_features(nodes)])) if nodes else iter(())
        tags = []; scal = []
        for c in cands:
            if c[0] is None: tags.append(TAG['FVAR']); scal.append(c[2][0])     # an interval split is described by the variable it splits
            else: t, s = next(feats); tags.append(t); scal.append(s)
        flags = [1 if c[1] else 0 for c in cands]
        return Menu(actions, cands, np.array(tags, dtype=np.uint32), np.array(scal, dtype=np.uint32), np.array(flags, dtype=np.uint8))

    def _is_ring_goal(self, g):
        w = self.w
        try: A, l, r = w.eq_parts(w.T(g.target))
        except AuthorError: return False
        return R.instance_for(w, A) is not None

    def _arith_ok(self, g):
        """is the goal, or a hypothesis, a statement about the integers that `linarith` reads (order, equation, negation of an order)?"""
        w = self.w; k = self.k
        if 'Z.lt' not in w.gid or 'Z.le' not in w.gid or 'Eq' not in w.gid: return False
        lt, le, eq, zty = w.gid['Z.lt'], w.gid['Z.le'], w.gid['Eq'], w['Z'].n
        not_ = w.gid.get('Not')
        def is_arith(ty):
            h, args = spine_raw(k, ty); hv = k.view(h)
            if hv[0] != 'const': return False
            if hv[1] in (lt, le): return len(args) == 2
            if hv[1] == eq: return len(args) == 3 and args[0] == zty
            if hv[1] == not_ and len(args) == 1:
                h2, a2 = spine_raw(k, args[0]); v2 = k.view(h2)
                return v2[0] == 'const' and v2[1] in (lt, le) and len(a2) == 2
            return False
        if is_arith(g.target): return True
        h, _ = spine_raw(k, g.target); hv = k.view(h)
        if hv[0] == 'const' and w.gname.get(hv[1]) == 'False': return True
        return any(is_arith(t) for t in g.hyp_types)

    def _classical_ok(self, g):
        k = self.k
        h, _ = spine_raw(k, g.target); hv = k.view(h)
        if hv[0] == 'const' and self.w.gname.get(hv[1]) in ('False', 'Not'): return False
        return not unfolds_to_pi(k, g.target)

    def _interval_candidates(self, g):
        """integer variables that the hypotheses bound on both sides by numerals: (position, lowest value, highest value) -- at most 40 values"""
        w = self.w; k = self.k
        if 'Z' not in w.gid or 'Z.le' not in w.gid or 'Z.lt' not in w.gid: return []
        inst = R.instance_for(w, w['Z'])
        if inst is None: return []
        zty = w['Z'].n; le, lt = w.gid['Z.le'], w.gid['Z.lt']
        lo = {}; hi = {}
        def var_of(n):
            v = k.view(n); return v[1] if v[0] == 'fvar' else None
        for ty in g.hyp_types:
            h, args = spine_raw(k, ty); hv = k.view(h)
            if hv[0] != 'const' or hv[1] not in (le, lt) or len(args) != 2: continue
            strict = hv[1] == lt
            a, b = args
            xa, xb = var_of(a), var_of(b)
            if xb is not None and xa is None:                       # c <= x  /  c < x
                c = inst.numeral(w.T(a))
                if c is not None:
                    c += 1 if strict else 0; lo[xb] = max(lo.get(xb, c), c)
            elif xa is not None and xb is None:                     # x <= c  /  x < c
                c = inst.numeral(w.T(b))
                if c is not None:
                    c -= 1 if strict else 0; hi[xa] = min(hi.get(xa, c), c)
        out = []
        for x in sorted(lo):
            if x in hi and 0 <= hi[x] - lo[x] < 40 and x < g.n_hyps and g.hyp_types[x] == zty:
                if max(i for i, nm in enumerate(g.hyp_names) if nm == g.hyp_names[x]) == x: out.append((x, lo[x], hi[x]))     # `interval_cases` finds the variable by its (latest) name
        return out

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- applying actions
    def _run(self, p, action, menu):
        """apply `action` to the first goal of the proof state `p` (tactic layer); raises what the tactic raises"""
        w = self.w; kind = action.kind
        if kind == Kind.INTRO: p.intro()
        elif kind == Kind.CASES: p.cases(action.arg)
        elif kind == Kind.INDUCT: p.induct(action.arg)
        elif kind == Kind.APPLY: p.apply(w.T(menu.cands[action.arg][0]))
        elif kind == Kind.REWRITE:
            term, rev, _ = menu.cands[action.arg]; p.rw(w.T(term), rev=rev)
        elif kind == Kind.RING: p.ring()
        elif kind == Kind.LINARITH: p.linarith()
        elif kind == Kind.NLINARITH: p.nlinarith()
        elif kind == Kind.BY_CONTRA: p.by_contra()
        elif kind == Kind.INTERVAL_CASES:
            h, lo, hi = menu.cands[action.arg][2]; p.interval_cases(p._names[p.ps.goal_meta_id(0)][h], lo, hi)
        else: raise ActionError('unknown action kind %r' % (kind,))

    def _close_goal(self, p, i):
        """close goal i if a hypothesis (up to computation) or reflexivity does; True if it did"""
        k = self.k; ps = p.ps; w = self.w
        n = ps.goal_num_hyps(i); target = ps.goal_target(i); meta = ps.goal_meta_id(i)
        types = [ps.goal_hyp_type(i, h) for h in range(n)]
        saved = k.get_fuel(); k.set_fuel(SETTLE_FUEL)
        try:
            for lvl in range(n - 1, -1, -1):
                if types[lvl] == target or self._conv(types[lvl], target, n):
                    ps.exact(k.mk_fvar(lvl), meta); p.trace.append(('exact', k.mk_fvar(lvl))); return True
            with self.in_context(types, p._names[meta]):
                try: A, a, b = w.eq_parts(w.T(target))
                except AuthorError: return False
                if a == b or self._conv(a.n, b.n, n):
                    r = w.app(w['refl'], A, a); ps.exact(r.n, meta); p.trace.append(('exact', r.n)); return True
        finally: k.set_fuel(saved)
        return False

    def _conv(self, a, b, n):
        try: return self.k.conv_open(a, b, n)
        except E.ITPError: return False                  # out of budget: not shown to be convertible

    def _intro_at(self, p, i):
        """introduce the first quantifier of goal i; False if the goal has none"""
        ps = p.ps; lvl = ps.goal_num_hyps(i); nm = 'a%d' % lvl
        while nm in p._names[ps.goal_meta_id(i)]: nm += '_'
        if not unfolds_to_pi(self.k, ps.goal_target(i)): return False
        try: ps.intro(nm, ps.goal_meta_id(i), True)
        except (E.ITPError, RuntimeError): return False
        p._sync_names(); p.trace.append(('intro', nm)); return True

    def settle(self, p, count=None):
        """settle the first `count` goals of the proof state (all by default): close those that a hypothesis or reflexivity closes and (with auto_intro) introduce the leading quantifiers of the others.
        Returns how many of them are left."""
        if count is None: count = p.n_goals
        i = 0
        while i < count:
            closed = False
            for _ in range(MAX_INTROS + 1):
                if self.autoclose and self._close_goal(p, i): closed = True; break
                if not self.auto_intro or not self._intro_at(p, i): break
            if closed: count -= 1
            else: i += 1
        return count

    def step_in(self, p, action, menu):
        """apply an action to the first goal of a live proof state, clean up after it and settle the goals it made: returns the number of goals left in its place"""
        before = p.n_goals; n_before = p.ps.goal_num_hyps(0)
        consumes = self.split_consumes(p.ps.goal_hyp_type(0, action.arg)) if action.kind in (Kind.CASES, Kind.INDUCT) else False
        self._run(p, action, menu)
        made = p.n_goals - before + 1
        if self.clean:
            if action.kind in (Kind.CASES, Kind.INDUCT): self._clean_split(p, action.arg, n_before, made, consumes)
            elif action.kind == Kind.INTERVAL_CASES: self._clean_interval(p, made, n_before)
        made = self.settle(p, made)
        if p.ps.num_pending(): raise ActionError('a unification constraint is left pending: the goals are not independent')
        return made

    def _clear_dead(self, p, i, dead):
        """remove from goal i the hypotheses in `dead`, except those that the target or a hypothesis that stays mentions (the set shrinks until nothing that stays refers to it)"""
        if not dead: return
        k = self.k; g = self.goal_of(p, i); dead = set(dead)
        mention = [atoms_of(k, t)[1] for t in g.hyp_types]
        tgt = atoms_of(k, g.target)[1]
        changed = True
        while changed and dead:
            changed = False
            for l in sorted(dead):
                if l in tgt or any(l in mention[j] for j in range(g.n_hyps) if j not in dead): dead.discard(l); changed = True
        if dead: p.clear(sorted(dead), i)

    def _clean_split(self, p, scrutinee, n_before, made, consumes):
        """after `cases` / `induct` on hypothesis `scrutinee` of a goal with `n_before` hypotheses: remove, from each goal it made, the variable that was split (when the split used it up), the hypotheses it
        reverted (the new goals bring them back), and the copies of the family's parameters that the constructor patterns introduce (named p<digits> by the tactic layer)"""
        reverted = set(p.ps.last_reverted())
        for i in range(made):
            names = p._names[p.ps.goal_meta_id(i)]
            dead = ({scrutinee} if consumes else set()) | reverted | {l for l in range(n_before, len(names)) if names[l][:1] == 'p' and names[l][1:].isdigit()}
            self._clear_dead(p, i, dead)

    def _clean_interval(self, p, made, n_before):
        """after `interval_cases` on a goal with `n_before` hypotheses: remove from each goal it made the helpers it leaves behind -- the bounds and cases it proved on the way (_ic_*), the disjunctions it split
        (_obtain) and the copies of the parameters of `Or` that the split introduces (p<digits>) -- so that what is left is the original context and the new equation"""
        for i in range(made):
            names = p._names[p.ps.goal_meta_id(i)]
            self._clear_dead(p, i, {l for l in range(n_before, len(names)) if names[l].startswith('_ic_') or names[l] == '_obtain' or (names[l][:1] == 'p' and names[l][1:].isdigit())})

    def probe(self, g, action, menu):
        """try an action on a goal, leaving the kernel as it was: the list of goals it leaves (GoalData), or None when the action fails. See the module documentation."""
        k = self.k; w = self.w
        cp = k.checkpoint(); old_fuel = k.get_fuel(); w.probing += 1
        refused = None
        try:
            k.set_fuel(self.fuel_of(action))
            p = Proof.from_goal(w, g.target, g.hyp_types, g.hyp_names)
            try: self.step_in(p, action, menu)
            except REFUSALS as e: refused = e
            else: children = k.finish_probe(cp, p.ps)
        except Exception as e:                       # not a refusal: something in this code or in the kernel's boundary is wrong
            k.rewind(cp)
            self.stats['probe_failed'] += 1; self.stats['probe_bug'] += 1
            self._bug(e)
            return None
        finally:
            w.probing -= 1; k.set_fuel(old_fuel)
        if refused is not None:
            self.errors[type(refused).__name__] += 1
            k.rewind(cp)
            self.stats['probe_failed'] += 1
            return None
        if children is None: self.stats['probe_dependent'] += 1; return None
        out = [GoalData(t, hs, ns) for (t, hs, ns) in children]
        if any(c.key == g.key for c in out): self.stats['probe_noprogress'] += 1; return None        # the goal came back unchanged: nothing was done
        self.stats['probe_ok'] += 1
        return out

    def fuel_of(self, action):
        """the work budget of an action (steps of the kernel's counter: an integer)"""
        return int(self.probe_fuel.get(action.kind, self.default_fuel))

    def _bug(self, e):
        """an exception that was not a refusal: counted, kept (the first few, with the traceback) and, in a strict engine, raised again (call it inside the `except` block)"""
        self.errors[type(e).__name__] += 1
        if len(self.bugs) < 20: self.bugs.append((type(e).__name__, str(e)[:300], traceback.format_exc(limit=8)))
        if self.strict: raise e

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- live proofs
    def start(self, stmt, name=None):
        """a live proof state for a closed statement (the real thing: its extracted term is a proof), with its first goal settled"""
        p = Proof(self.w, self.w.term(stmt, 'statement'), name)
        p.norm()
        self.settle(p)
        return p

    def replay(self, p, action, menu):
        """apply an action to the first goal of a live proof; raises ActionError if the tactic refuses it. Returns the number of goals left in its place. It runs under the same budget as `probe`."""
        k = self.k; old_fuel = k.get_fuel()
        try:
            k.set_fuel(self.fuel_of(action))
            return self.step_in(p, action, menu)
        except (AuthorError, E.ITPError, RuntimeError) as e: raise ActionError(str(e)) from None
        finally: k.set_fuel(old_fuel)

    def verify(self, stmt, p):
        """the proof state must be finished; the term is checked by the kernel against the statement (empty context). Returns the term."""
        term = p.term(); self.w.check(term, self.w.term(stmt, 'statement'))
        return term


class Episode:
    """one proof attempt as a game for an agent that acts step by step: the state is a proof, the agent acts on its first open goal.
    `step` is atomic: an action is probed first, so one that fails changes nothing."""
    def __init__(self, engine, stmt, name=None):
        self.eng = engine; self.stmt = engine.w.term(stmt, 'statement'); self.p = engine.start(self.stmt, name)
        self.steps = []; self._menu = None; self._goal = None
    @property
    def done(self): return self.p.done and self.p.ps.is_complete()
    @property
    def n_goals(self): return self.p.n_goals
    def goal(self):
        if self._goal is None: self._goal = self.eng.goal_of(self.p, 0)
        return self._goal
    def menu(self):
        if self._menu is None: self._menu = self.eng.menu(self.goal())
        return self._menu
    def observe(self): return self.eng.observe(self.goal())
    def step(self, action):
        menu = self.menu(); g = self.goal()
        if self.eng.probe(g, action, menu) is None: raise ActionError('the action %r does not apply' % (action,))
        self.eng.replay(self.p, action, menu)
        self.steps.append(action); self._menu = None; self._goal = None
        return self.p.n_goals
    def finish(self):
        if not self.done: raise ActionError('the proof is not finished')
        return self.eng.verify(self.stmt, self.p)
