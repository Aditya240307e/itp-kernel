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
    INSTANTIATE      a new hypothesis: a hypothesis (or an allowed lemma) applied to chosen terms -- `have h := H 0 b`; the terms come from the goal's TERM POOL (gym_terms.py), named by position
    USE              give the witness of an existential goal, a term of the pool (or a term said by a code)
THE SECOND LANGUAGE (gym_acts.py, opt-in with `kinds=`).  Actions whose arguments are SAID -- a statement, a witness, a lemma's arguments, a source -- as codes of the typed term builder (gym_build.py):
    CLAIM            `have h : S`: two goals, S and the goal with h     INDUCT_WITH   induction on a variable by a library principle (the starting point said)
    FORWARD          `have h := source slots`                           INJECT        injectivity of a constructor equation
    REWRITE_AT       rewrite inside a hypothesis                         REFINE        a term with propositions left as goals (exact, witnesses, splits, apply with arguments)
    APPLY_WITH       apply a lemma to the arguments the goal does not determine      REWRITE_WITH  rewrite by a lemma given its leading arguments      LINCOMB   linear_combination
After every action the goals it left are SETTLED, identically in probes and in replay, so a search never spends a step on them: a goal that a hypothesis (assumption) or reflexivity closes is closed,
and (with `auto_intro`) the quantifiers and implications at the head of a goal that is left are introduced.
"""
import os, sys, enum, collections, contextlib, traceback, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import itp_env_lnl as E
from author import AuthorError, Term, NodeCache
from tactics import Proof, HOLE_BASE
import ring as R
import gym_terms as GT
import gym_build as GB


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
    INSTANTIATE = 10
    USE = 11
    CLAIM = 12
    INDUCT_WITH = 13
    FORWARD = 14
    INJECT = 15
    REWRITE_AT = 16
    REFINE = 17
    APPLY_WITH = 18
    REWRITE_WITH = 19
    LINCOMB = 20


KIND_NAMES = tuple(k.name for k in Kind)
V1_KINDS = frozenset(k for k in Kind if k <= Kind.USE)          # the kinds of the first language: what an engine offers unless told otherwise
NEEDS_CANDIDATE = frozenset({Kind.APPLY, Kind.REWRITE, Kind.INTERVAL_CASES, Kind.INSTANTIATE})
NEEDS_HYPOTHESIS = frozenset({Kind.CASES, Kind.INDUCT, Kind.INJECT})
NEEDS_TERMS = frozenset({Kind.INSTANTIATE, Kind.USE})
TERM_KINDS = NEEDS_TERMS
CODE_KINDS = frozenset({Kind.CLAIM, Kind.INDUCT_WITH, Kind.FORWARD, Kind.REWRITE_AT, Kind.REFINE, Kind.APPLY_WITH, Kind.REWRITE_WITH, Kind.LINCOMB})     # the kinds whose whole meaning is in `codes` / `hyps`
CODE_MARK = 'code'                                              # the first element of the signature's terms of a USE whose witness is said by a code
GEN_MARK = 'gen'                                                # the same for a CASES / INDUCT that generalises hypotheses


class Action(collections.namedtuple('Action', 'kind arg args codes hyps', defaults=((), (), ()))):
    """an action: its kind, its argument (a hypothesis position for CASES / INDUCT / INJECT, an index into the menu's candidates for APPLY / REWRITE / INTERVAL_CASES / INSTANTIATE, a direction for
    REWRITE_AT / REWRITE_WITH, else 0) and, for the kinds that carry terms (INSTANTIATE, USE), the positions in the menu's term pool of the terms (INSTANTIATE: the first quantifiers of the candidate, in
    order; USE: the witness). The kinds of the second language (gym_acts.py) say their arguments: `codes` is a tuple of codes of the typed term builder, `hyps` a tuple of hypothesis positions."""
    __slots__ = ()
    def __repr__(self):
        # (an action that makes no sense -- a kind that does not exist, an argument that is no number -- is printed all the same: this is what error messages show)
        name = KIND_NAMES[self.kind] if isinstance(self.kind, (int, np.integer)) and not isinstance(self.kind, bool) and 0 <= self.kind < len(KIND_NAMES) else 'kind %r' % (self.kind,)
        try: terms = ', '.join(str(i) for i in self.args)
        except TypeError: terms = repr(self.args)
        extra = ''
        if self.hyps: extra += ' hyps=%s' % (self.hyps,)
        if self.codes: extra += ' codes=%s' % (self.codes,)
        if len(extra) > 160: extra = extra[:157] + '...'
        return '%s(%s%s%s)' % (name, self.arg, (', ' + terms) if terms else '', extra)


class ActionError(Exception):
    """an action that could not be applied (the tactic refused it, or ran out of budget)"""


STRICT_DEFAULT = False       # engines made without `strict` take this (the tests and the fuzzer set it)
TERMS_DEFAULT = False        # engines made without `terms` take this: do they list INSTANTIATE and USE in their menus?
MAX_USE = 40                 # USE: the witnesses listed per goal (the first of the pool's terms of the right type)
INST_ONE = 40                # INSTANTIATE with one term: the first this many terms of the type are tried
INST_TWO = 10                # INSTANTIATE with two terms: the first this many for each
MAX_INST = 400               # INSTANTIATE actions listed per goal
ST_OK, ST_REFUSED, ST_FUEL, ST_BUG = 'ok', 'refused', 'fuel', 'bug'       # the outcomes of `Engine.probe_status`
REFUSALS = (ActionError, AuthorError, E.ITPError, RuntimeError)      # what a probe may legitimately see go wrong: the tactic layer's and the kernel's own refusals (nanobind turns the kernel's logic errors into RuntimeError)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ goals
def goal_key(target, hyp_types):
    """the identity of a goal: the kernel's terms are hash-consed, so equal terms are the same node and this key is exact (the hypotheses' names are display data and are not part of it)"""
    return (target,) + tuple(hyp_types)


class GoalData:
    """one goal: target and hypothesis types (kernel handles), the hypotheses' names, an exact `key`, and the observation (computed on first use)"""
    __slots__ = ('target', 'hyp_types', 'hyp_names', 'key', 'obs', 'ckey', 'space')
    def __init__(self, target, hyp_types, hyp_names):
        self.target = target; self.hyp_types = list(hyp_types); self.hyp_names = list(hyp_names)
        self.key = goal_key(target, self.hyp_types); self.obs = None; self.ckey = None; self.space = None
    @property
    def n_hyps(self): return len(self.hyp_types)


class Menu:
    """the legal actions of a goal and the candidates they refer to. `cands[i]` is (term, rev, aux): the kernel handle of a lemma or hypothesis (None for INTERVAL_CASES), whether a REWRITE goes right
    to left, and for INTERVAL_CASES the triple (hypothesis position, lowest value, highest value). `cand_tags` / `cand_scalars` are the (tag, scalar) features of the terms (as `Kernel.term_features`),
    `cand_flags`: bit 0 is set for a right-to-left rewrite, bit 1 for a rewrite by a permutative equation (see `Profile`). `pool` is the goal's term pool (gym_terms.Pool: the terms that INSTANTIATE and USE name by position) with the (tag, scalar) features of its terms."""
    __slots__ = ('actions', 'cands', 'cand_tags', 'cand_scalars', 'cand_flags', 'pool', 'pool_tags', 'pool_scalars')
    def __init__(self, actions, cands, cand_tags, cand_scalars, cand_flags, pool=GT.EMPTY, pool_tags=None, pool_scalars=None):
        self.actions = actions; self.cands = cands; self.cand_tags = cand_tags; self.cand_scalars = cand_scalars; self.cand_flags = cand_flags
        self.pool = pool; self.pool_tags = np.zeros(0, dtype=np.uint32) if pool_tags is None else pool_tags; self.pool_scalars = np.zeros(0, dtype=np.uint32) if pool_scalars is None else pool_scalars
    def signature(self, action):
        """what an action means independently of the menu it came from: (kind, the candidate's node or hypothesis position, direction, range) -- and for the kinds that carry terms, the nodes of the terms.
        The kinds of the second language say their arguments themselves: (kind, argument, False, (hypotheses, codes)); a USE whose witness is said by a code: (kind, 0, False, ('code', codes))"""
        if action.kind in CODE_KINDS: return (int(action.kind), action.arg, False, (tuple(action.hyps), tuple(action.codes)))
        if action.kind == Kind.USE and action.codes: return (int(action.kind), action.arg, False, (CODE_MARK, tuple(action.codes)))
        if action.kind in (Kind.CASES, Kind.INDUCT) and action.hyps: return (int(action.kind), action.arg, False, (GEN_MARK, tuple(action.hyps)))
        if action.kind in NEEDS_CANDIDATE:
            term, rev, aux = self.cands[action.arg]
            if action.kind in NEEDS_TERMS: aux = tuple(self.pool.terms[i] for i in action.args)
            return (int(action.kind), term, rev, aux)
        if action.kind in NEEDS_TERMS: return (int(action.kind), action.arg, False, tuple(self.pool.terms[i] for i in action.args))
        return (int(action.kind), action.arg, False, None)
    def find(self, signature):
        """the action with this signature in this menu, or None. The actions of the second language mean what they say, whether or not the menu lists them: they are made from the signature."""
        kind = signature[0]
        if kind in CODE_KINDS:
            hyps, codes = signature[3]
            return Action(Kind(kind), signature[1], (), tuple(codes), tuple(hyps))
        aux = signature[3]
        if kind == Kind.USE and isinstance(aux, tuple) and len(aux) == 2 and isinstance(aux[0], str) and aux[0] == CODE_MARK:       # (never compare a kernel handle with a string: the handles' __eq__ refuses)
            return Action(Kind.USE, signature[1], (), tuple(aux[1]))
        if kind in (Kind.CASES, Kind.INDUCT) and isinstance(aux, tuple) and len(aux) == 2 and isinstance(aux[0], str) and aux[0] == GEN_MARK:
            return Action(Kind(kind), signature[1], (), (), tuple(aux[1]))
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


def conclusion_args(k, ty):
    """the arguments of the conclusion of a type, its quantifiers and implications peeled (the variables of the type itself are still bound variables)"""
    cur = ty
    while True:
        v = k.view(cur)
        if v[0] != 'pi': break
        cur = v[2]
    return spine_raw(k, cur)[1]


def fps_compatible(a, b):
    """can two lists of argument fingerprints (`Engine.arg_fps`) belong to a lemma and a goal that `apply` could match? Where both name a type, it must be the same one"""
    return all(x is None or y is None or x == y for x, y in zip(a, b))


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


def rw_head_key(hv, args=True):
    """the key under which a rewrite pattern, and the subterms it can match, are indexed by their head: the global id of a constant, or -1 - level for a variable of the context that is applied to something (`f` in
    `f x`); None for anything else (a bare variable matches everything)"""
    if hv[0] == 'const': return hv[1]
    if hv[0] == 'fvar' and args: return -1 - hv[1]
    return None


def subterms_by_head(k, root):
    """the closed subterms of a term (no loose bound variable, the ones `rw` looks at), grouped by the key of their head (`rw_head_key`)"""
    out = {}; seen = set(); stack = [root]
    while stack:
        t = stack.pop()
        if t in seen: continue
        seen.add(t)
        if k.loose_range(t) == 0:
            h, args = spine_raw(k, t); key = rw_head_key(k.view(h), bool(args))
            if key is not None: out.setdefault(key, []).append(t)
        v = k.view(t); kind = v[0]
        if kind == 'app': stack.append(v[2]); stack.append(v[1])
        elif kind in ('lam', 'pi'): stack.append(v[2]); stack.append(v[1])
        elif kind == 'let': stack += [v[3], v[2], v[1]]
        elif kind == 'match': stack += list(reversed(v[3])) + [v[2], v[1]]
        elif kind == 'fix': stack += [v[3], v[1]]
    return out


def leaf_counts(k, root, limit=600):
    """the multiset of constants and variables at the leaves of a term, each counted as often as it occurs in the term written as a tree ({} entries for what is not looked into); None for a term of more than
    `limit` nodes"""
    out = collections.Counter(); stack = [root]; seen = 0
    while stack:
        t = stack.pop(); seen += 1
        if seen > limit: return None
        v = k.view(t); kind = v[0]
        if kind in ('const', 'fvar', 'bvar'): out[(kind, v[1])] += 1
        elif kind in ('app', 'lam', 'pi'): stack.append(v[1]); stack.append(v[2])
        else: out[(kind,)] += 1
    return out


class Profile:
    """what an `apply` or a `rw` of a statement needs to know, read off its type once. `undetermined`: the DATA quantifiers that the conclusion does not determine (they would stay as unsolved
    metavariables shared by the goals that are left: the goals would not be independent); `rw`: the directions in which an equation can be used by `rw` (its pattern contains every quantifier),
    as (reverse?, head symbol of the pattern as a global id, the pattern with its quantifiers as holes: context variables from HOLE_BASE up); `permutative`: the equation only reorders the parts of
    a term (both sides have the same constants and variables, as often: `a + b = b + a`, `(a * b) * c = a * (b * c)`) -- rewriting with it can walk through all the orders of a term without getting anywhere"""
    __slots__ = ('n', 'undetermined', 'rw', 'permutative')
    def __init__(self, n, undetermined, rw, permutative=False):
        self.n = n; self.undetermined = undetermined; self.rw = rw; self.permutative = permutative


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the engine
PROBE_FUEL = {Kind.RING: 50_000_000, Kind.LINARITH: 50_000_000, Kind.NLINARITH: 100_000_000, Kind.INTERVAL_CASES: 50_000_000, Kind.LINCOMB: 50_000_000}
DEFAULT_PROBE_FUEL = 5_000_000
SETTLE_FUEL = 20_000          # the budget of each conversion check that settles a goal (is it an assumption? is it `x = x`?)
MAX_INTROS = 64
INTRO_HEADS = ('And', 'Or', 'Iff', 'Ex', 'True')       # the connectives whose introduction rules only split a goal (see Engine.introduces)


class Engine:
    """goals, menus, probes and replay for ONE world. `premises`: the number of globals (by id) that actions may use -- a lemma declared later is not offered, so a theorem is attacked with
    the library that precedes it; None means all. `auto_intro`: introduce the quantifiers and implications at the head of every goal (so INTRO is not an action); `autoclose`: close goals that a
    hypothesis or reflexivity proves; `clean`: after `cases` / `induct` remove the dead copies they leave behind (the variable that was split, the hypotheses they reverted and brought back, the
    copies of the family's parameters), so that goals that differ only by such copies are the same goal. `kinds`: the kinds of action the menu offers (all by default). `strict`: an exception inside a
    probe that is not a refusal of the tactic layer or of the kernel (a TypeError, a stale checkpoint, ...) is a BUG of this code: it is always kept in `bugs`, and a strict engine raises it (after the
    kernel has been rewound); a lenient one counts it as a failed action, so a long search survives it. A refusal that is mistaken for a bug costs a search nothing and hides nothing; a bug that is mistaken
    for a refusal hides every action that hits it, silently."""
    def __init__(self, w, premises=None, auto_intro=True, autoclose=True, clean=True, probe_fuel=None, kinds=None, strict=None, terms=None, inst_lemmas=(), claims=()):
        if w.ctx: raise AuthorError('an engine must be made at the top level')
        self.w = w; self.k = w.k; self.auto_intro = auto_intro; self.autoclose = autoclose; self.clean = clean
        self.kinds = V1_KINDS if kinds is None else frozenset(kinds)              # the kinds of action the menu offers (by default those of the first language; every kind is legal either way)
        self.claim_hints = tuple(claims)                                          # closed statements (kernel handles) that the menu offers as CLAIMs, when it lists that kind
        self.terms = (TERMS_DEFAULT or (kinds is not None and bool(self.kinds & TERM_KINDS))) if terms is None else bool(terms)       # does the menu list the actions that carry terms (INSTANTIATE, USE)? They are legal either way
        self.inst_lemmas = tuple(inst_lemmas)                                     # the library lemmas (names) that INSTANTIATE is offered for, besides the hypotheses
        self.premises = premises
        self.probe_fuel = dict(PROBE_FUEL); self.default_fuel = DEFAULT_PROBE_FUEL if probe_fuel is None else int(probe_fuel)
        self.strict = STRICT_DEFAULT if strict is None else strict
        self.stats = collections.Counter()
        self.errors = collections.Counter()              # exception types seen while probing, by name (a type other than the tactic layer's own refusals is a bug: see `bugs`)
        self.bugs = []                                   # (type name, message, traceback) of the first few exceptions that were not refusals
        self._profiles = {}; self._scanned = 0; self._rw_index = {}      # global id -> Profile;  head global id -> [(lemma global id, reverse?)]
        self._lemma_fps = {}; self._intros = {}; self._intro_heads = frozenset()   # global id -> fingerprints of the arguments of its conclusion;  global id -> does it introduce a connective?
        self.type_filter = True                          # offer no lemma about another type than the goal's (switched off by the tests that check the filter loses nothing)
        self._hyp_profiles = NodeCache(w); self._families = NodeCache(w); self._recursive = {}
        self.name_tag = ''                               # prefixed to the names the engine gives to the hypotheses it introduces (a1, i3, r5): the recorder sets it so that they never clash with an author's names
        self.principles = {}; self.inj_index = {}                 # global id -> gym_acts.Principle;  head constant -> [injectivity lemmas]  (read off the library by `refresh`)
        self._acts_scanned = 0; self._hint_codes = {}; self.table = None
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
        self._intro_heads = frozenset(w.gid[n] for n in INTRO_HEADS if n in w.gid)
        for tyname in ('Nat', 'N', 'Z'):
            if tyname in w.gid:
                inst = R.instance_for(w, w[tyname])
                if inst is not None:
                    for v in range(-2, 17):
                        try: inst.lit(v)
                        except AuthorError: pass
        self.table = GB.head_table(w)                          # the heads a term can start with (the typed term builder)
        self._scan()
        ACTS.scan(self, self._acts_scanned, k.num_globals()); self._acts_scanned = k.num_globals()

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
            rw = []; permutative = False
            if rewriting:
                with self.in_context(list(ctx_types) + doms):
                    try: A, lhs, rhs = w.eq_parts(w.T(concl))
                    except AuthorError: A = None
                if A is not None:
                    if lhs.n != rhs.n:
                        cl = leaf_counts(k, lhs.n); permutative = cl is not None and cl == leaf_counts(k, rhs.n)
                    for rev, pat in ((False, lhs), (True, rhs)):
                        head, pargs = spine_raw(k, pat.n); hv = k.view(head)
                        if hv[0] == 'fvar' and hv[1] >= base: continue                       # one of the statement's own quantifiers at the head: the pattern matches everything
                        key = rw_head_key(hv, bool(pargs))
                        if key is None: continue                  # a bare variable (or a lambda, a match) as the pattern: not offered
                        fv = atoms_of(k, pat.n)[1]
                        if all(base + j in fv for j in range(n)):
                            holes = [(base + j, k.mk_fvar(HOLE_BASE + j)) for j in range(n)]
                            rw.append((rev, key, k.replace_fvars(pat.n, holes) if holes else pat.n))
            return Profile(n, tuple(undetermined), tuple(rw), permutative)
        except (E.ITPError, AuthorError, RuntimeError):
            return Profile(0, (-1,), ())                    # unreadable: never offered

    def shape_fp(self, t):
        """what a type is headed by once its definitions are unfolded: the global id of an inductive family or of the quotient former (`Z` is a definition, of a quotient), 'pi' for a quantifier or an
        implication, ('type', level) for a sort -- or None where that cannot be told (a variable, a match, ...). Types of different shapes are not convertible, and telling so is cheap, while
        checking that two types are convertible evaluates both."""
        k = self.k; cur = t
        try:
            for _ in range(16):
                v = k.view(cur)
                if v[0] == 'pi': return 'pi'
                if v[0] == 'type': return v
                h, _ = spine_raw(k, cur); hv = k.view(h)
                if hv[0] != 'const': return None
                if k.is_inductive(hv[1]) or k.quot_role(hv[1]) == 1: return hv[1]
                cur = k.unfold_head(cur)
                if cur is None: return None
        except (E.ITPError, RuntimeError): return None
        return None

    def type_fp(self, t):
        """which type a term is, as far as can be told without computing: the global id of the inductive family or of the quotient former that it is an instance of, looking through definitions at its
        head -- or None for anything else (a variable, a quantifier, a term that is no type, ...). Two different ones are different types."""
        r = self.shape_fp(t)
        return r if isinstance(r, int) else None

    def arg_fps(self, ty):
        """for each argument of the conclusion of a statement: which type it is (`type_fp`) or None -- `Eq Z a b` and `Eq Nat a b` differ in it, and a lemma about one cannot be applied to a goal about the other"""
        k = self.k
        return tuple(self.type_fp(a) for a in conclusion_args(k, ty))

    def lemma_fps(self, gid):
        r = self._lemma_fps.get(gid)
        if r is None: r = self._lemma_fps[gid] = self.arg_fps(self.k.global_info(gid)[1])
        return r

    def introduces(self, gid):
        """is the lemma an introduction rule of a connective -- its conclusion is a conjunction, a disjunction, an equivalence, an existential or `True`, and none of its premises is one of those
        too (`and_comm` only turns a conjunction round)? Applying one splits the goal into its parts"""
        r = self._intros.get(gid)
        if r is None:
            k = self.k; ty = k.global_info(gid)[1]; ch = conclusion_head(k, ty)
            r = ch is not None and ch[0] == 'c' and ch[1] in self._intro_heads
            if r:
                try: doms, _ = open_binders(k, ty, 0)
                except (E.ITPError, RuntimeError): doms = []
                r = not any(conclusion_head(k, d) == ch for d in doms)
            self._intros[gid] = r
        return r

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

    def space(self, g, pool=None):
        """the term space of a goal (gym_build.TermSpace): its builders say any well-typed term of the goal's context. Made on first use and kept with the goal -- except inside a probe, whose nodes are
        reclaimed, where a fresh one is made and not kept -- and with a `pool` (gym_terms.Pool, whose terms the tokens ('p', i) name) a fresh one, not kept."""
        if pool is not None or self.w.probing: return GB.TermSpace(self, g.hyp_types, g.hyp_names, pool=pool, roots=[g.target])
        if g.space is None: g.space = GB.TermSpace(self, g.hyp_types, g.hyp_names, roots=[g.target])
        return g.space

    def builder(self, g, expected=None, pool=None, goals=False):
        """a builder (gym_build.Builder) for a term of the type `expected` (the goal's target by default) in the context of goal g: `tokens()` lists the choices at the first hole, `push` takes one. With
        `goals` a proposition may be left as a goal (`('g',)`): the term is then the argument of a REFINE."""
        b = self.space(g, pool).builder(g.target if expected is None else expected)
        b.allow_goals = bool(goals)
        return b

    def claim_codes(self, space):
        """the codes of the engine's claim hints (closed statements) in the context of `space`. A code names the variables of its own binders by their levels, which come after those of the context: so
        the codes are made for each size of context, once, and outside probes (whose nodes are reclaimed)."""
        n = len(space.ctx); codes = self._hint_codes.get(n)
        if codes is None:
            if self.w.probing: return []
            codes = []
            for h in self.claim_hints:
                try: code = space.encode(h, space.prop)
                except GB.BuildError: continue
                if code not in codes: codes.append(code)                                  # (hints that are the same statement are one claim)
            self._hint_codes[n] = codes
        return codes

    def render(self, g):
        w = self.w; lines = []
        for i, (nm, ty) in enumerate(zip(g.hyp_names, g.hyp_types)): lines.append('%s : %s' % (nm or '_', w.show(w.T(ty), names=g.hyp_names[:i])))
        lines.append('|- ' + w.show(w.T(g.target), names=g.hyp_names))
        return '\n'.join(lines)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- the menu
    def menu(self, g):
        """the legal actions of a goal, with their candidate table. Nothing here is tried: an action may still fail when probed."""
        w = self.w; k = self.k; limit = self.limit; nh = g.n_hyps
        actions = []; cands = []; cperm = []                 # (cperm: for each candidate, 1 if it is a rewrite by a permutative equation)
        if not self.auto_intro and unfolds_to_pi(k, g.target): actions.append(Action(Kind.INTRO, 0))
        for h in range(nh):
            fam = self.family_of(g.hyp_types[h])
            if fam is None: continue
            actions.append(Action(Kind.CASES, h))
            if self.is_recursive_family(fam): actions.append(Action(Kind.INDUCT, h))
        # --- apply: the library through its index (looked up with the display normal form of the target, which is how lemmas are stated), the hypotheses by the head of their conclusion
        tnorm = k.simp_normalize(g.target, nh)
        gh = conclusion_head(k, g.target); goal_fps = self.arg_fps(tnorm)
        seen = set()
        for c in k.library_candidates(self.lib, tnorm, nh):
            v = k.view(c)
            if v[0] != 'const' or v[1] >= limit or v[1] in seen: continue
            seen.add(v[1])
            if self.profile(v[1]).undetermined: continue
            if self.type_filter and not fps_compatible(self.lemma_fps(v[1]), goal_fps): continue       # a lemma about another type
            cands.append((c, False, None)); cperm.append(0); actions.append(Action(Kind.APPLY, len(cands) - 1))
        for h in range(nh):
            if not heads_compatible(conclusion_head(k, g.hyp_types[h]), gh): continue
            if self.hyp_profile(g, h).undetermined: continue
            cands.append((k.mk_fvar(h), False, None)); cperm.append(0); actions.append(Action(Kind.APPLY, len(cands) - 1))
        # --- rewrite: the equations whose pattern occurs in the goal (the very test `rw` makes: first-order matching against the closed subterms; the probe still has the last word)
        t_consts, t_fvars = atoms_of(k, g.target)
        heads = subterms_by_head(k, g.target); shim = _MatchShim(k)
        def occurs(head, pat): return any(shim.match(pat, t, {}) for t in heads.get(head, ()))
        for head in sorted(t_consts):
            for gid, rev, pat in self._rw_index.get(head, ()):
                if gid >= limit or not occurs(head, pat): continue
                cands.append((k.mk_const(gid), rev, None)); cperm.append(int(self.profile(gid).permutative)); actions.append(Action(Kind.REWRITE, len(cands) - 1))
        for h in range(nh):
            pr = self.hyp_profile(g, h)
            for rev, head, pat in pr.rw:
                if occurs(head, pat): cands.append((k.mk_fvar(h), rev, None)); cperm.append(int(pr.permutative)); actions.append(Action(Kind.REWRITE, len(cands) - 1))
            if pr.n == 0:                                          # a plain equation with a context variable as a side (x = f y): the pattern is that variable
                with self.in_context(g.hyp_types[:h], g.hyp_names[:h]):
                    try: A, lhs, rhs = w.eq_parts(w.T(g.hyp_types[h]))
                    except AuthorError: continue
                for rev, pat in ((False, lhs), (True, rhs)):
                    pv = k.view(pat.n)
                    if pv[0] == 'fvar' and pv[1] in t_fvars: cands.append((k.mk_fvar(h), rev, None)); cperm.append(0); actions.append(Action(Kind.REWRITE, len(cands) - 1))
        # --- decision procedures and classical reasoning
        with self.in_context(g.hyp_types, g.hyp_names):
            if self._is_ring_goal(g): actions.append(Action(Kind.RING, 0))
            if self._arith_ok(g):
                actions.append(Action(Kind.LINARITH, 0)); actions.append(Action(Kind.NLINARITH, 0))
            if 'by_contra' in w.gid and self._classical_ok(g): actions.append(Action(Kind.BY_CONTRA, 0))
            intervals = self._interval_candidates(g)
        for (h, lo, hi) in intervals:
            cands.append((None, False, (h, lo, hi))); cperm.append(0); actions.append(Action(Kind.INTERVAL_CASES, len(cands) - 1))
        # --- terms: witnesses of existential goals, hypotheses (and allowed lemmas) applied to terms of the pool
        pool = GT.EMPTY
        if self.terms and self.kinds & TERM_KINDS:
            with self.in_context(g.hyp_types, g.hyp_names):
                use_type = self._exists_type(g.target) if Kind.USE in self.kinds else None
                sources = self._inst_sources(g) if Kind.INSTANTIATE in self.kinds else []
                if use_type is not None or sources:
                    want = ({use_type} if use_type is not None else set()).union(*[set(src[3][:2]) for src in sources])
                    pool = GT.build_pool(w, g.hyp_types, g.hyp_names, [g.target] + list(g.hyp_types), want)
                    if use_type is not None: actions.extend(self._use_actions(use_type, pool))
                    if sources: self._inst_actions(g, pool, cands, actions, sources)
                    cperm.extend([0] * (len(cands) - len(cperm)))
        if any(a.kind not in self.kinds for a in actions):     # drop the kinds that are not offered, and the candidates only they used (renumbering the others)
            keep = [a for a in actions if a.kind in self.kinds]
            used = sorted({a.arg for a in keep if a.kind in NEEDS_CANDIDATE}); remap = {old: new for new, old in enumerate(used)}
            actions = [Action(a.kind, remap[a.arg], a.args) if a.kind in NEEDS_CANDIDATE else a for a in keep]; cands = [cands[i] for i in used]; cperm = [cperm[i] for i in used]
        ACTS.extend_menu(self, g, actions)                  # the actions of the second language that can be listed (they refer to no candidate)
        nodes = [c[0] for c in cands if c[0] is not None]
        feats = iter(zip(*[a.tolist() for a in k.term_features(nodes)])) if nodes else iter(())
        tags = []; scal = []
        for c in cands:
            if c[0] is None: tags.append(TAG['FVAR']); scal.append(c[2][0])     # an interval split is described by the variable it splits
            else: t, s = next(feats); tags.append(t); scal.append(s)
        flags = [(1 if c[1] else 0) | (2 if cperm[i] else 0) for i, c in enumerate(cands)]
        ptags = pscal = None
        if len(pool):
            pf = k.term_features(list(pool.terms)); ptags = np.asarray(pf[0], dtype=np.uint32); pscal = np.asarray(pf[1], dtype=np.uint32)
        return Menu(actions, cands, np.array(tags, dtype=np.uint32), np.array(scal, dtype=np.uint32), np.array(flags, dtype=np.uint8), pool, ptags, pscal)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- actions that carry terms
    def _exists_type(self, target):
        """the type A when the goal is `Ex A P` (looking through definitions at its head: `a | b` is one), else None"""
        k = self.k; ex = self.w.gid.get('Ex'); cur = target
        if ex is None: return None
        for _ in range(8):
            h, args = spine_raw(k, cur); hv = k.view(h)
            if hv[0] == 'const' and hv[1] == ex and len(args) == 2: return args[0]
            if hv[0] != 'const': return None
            cur = k.unfold_head(cur)
            if cur is None: return None
        return None

    def _use_actions(self, use_type, pool):
        return [Action(Kind.USE, 0, (i,)) for i in pool.of_type.get(use_type, ())[:MAX_USE]]

    def _inst_binders(self, ty, ctx_types):
        """the types (kernel handles) of the leading DATA quantifiers of a proposition -- a hypothesis in the context `ctx_types`, or a lemma -- up to the first premise or the first quantifier whose type
        mentions an earlier one (at most four)"""
        k = self.k; PROP = self.w.PROP.n; doms = []; cur = ty
        if k.view(ty)[0] != 'pi': return doms
        ok, sort = k.infer_open(ty, list(ctx_types))
        if not ok or sort != PROP: return doms
        while len(doms) < 4:
            v = k.view(cur)
            if v[0] != 'pi' or k.loose_range(v[1]) != 0: break
            ok, sort = k.infer_open(v[1], list(ctx_types))
            if not ok or sort == PROP: break
            doms.append(v[1]); cur = v[2]
        return doms

    def _inst_sources(self, g):
        """what INSTANTIATE is offered for: the hypotheses that are propositions with leading data quantifiers, and the allowed lemmas -- as (term, statement, context types, types of the quantifiers)"""
        k = self.k; w = self.w; out = []
        cands = [(k.mk_fvar(h), g.hyp_types[h], g.hyp_types[:h]) for h in range(g.n_hyps)]
        for name in self.inst_lemmas:
            gid = w.gid.get(name)
            if gid is not None and gid < self.limit: cands.append((k.mk_const(gid), k.global_info(gid)[1], []))
        for term, ty, ctx in cands:
            doms = self._inst_binders(ty, ctx)
            if doms: out.append((term, ty, ctx, doms))
        return out

    def _inst_actions(self, g, pool, cands, actions, sources):
        """INSTANTIATE for every source whose first quantifier's type the pool has terms of: with one term, or with two when there are two quantifiers; the facts that are already hypotheses (up to the display
        normal form) are left out. The terms are the first of the pool's, simplest first."""
        k = self.k; nh = g.n_hyps
        existing = set(g.hyp_types) | {k.simp_normalize(t, nh) for t in g.hyp_types}
        count = 0
        for term, ty, ctx, doms in sources:
            if not pool.of_type.get(doms[0]): continue
            two = len(doms) >= 2 and bool(pool.of_type.get(doms[1]))
            lists = [pool.of_type[doms[0]][:INST_TWO if two else INST_ONE]] + ([pool.of_type[doms[1]][:INST_TWO]] if two else [])
            ci = None
            for combo in itertools.product(*lists):
                if count >= MAX_INST: return
                inst = ty
                for i in combo: inst = k.instantiate(k.view(inst)[2], [pool.terms[i]])
                if k.simp_normalize(inst, nh) in existing: continue
                if ci is None: cands.append((term, False, None)); ci = len(cands) - 1
                actions.append(Action(Kind.INSTANTIATE, ci, tuple(combo))); count += 1

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
        """apply `action` to the first goal of the proof state `p` (tactic layer); raises what the tactic raises. Returns what the clean-up after the action needs (None for most kinds)."""
        w = self.w; kind = action.kind
        if kind in CODE_KINDS or kind == Kind.INJECT or (kind == Kind.USE and action.codes): return ACTS.run(self, p, action)
        if kind == Kind.INTRO: p.intro()
        elif kind == Kind.CASES: p.cases(action.arg, generalizing=list(action.hyps))
        elif kind == Kind.INDUCT: p.induct(action.arg, generalizing=list(action.hyps))
        elif kind == Kind.APPLY: p.apply(w.T(menu.cands[action.arg][0]))
        elif kind == Kind.REWRITE:
            term, rev, _ = menu.cands[action.arg]; p.rw(w.T(term), rev=rev)
        elif kind == Kind.RING: p.ring()
        elif kind == Kind.LINARITH: p.linarith()
        elif kind == Kind.NLINARITH: p.nlinarith()
        elif kind == Kind.BY_CONTRA: p.by_contra()
        elif kind == Kind.INTERVAL_CASES:
            h, lo, hi = menu.cands[action.arg][2]; p.interval_cases(p._names[p.ps.goal_meta_id(0)][h], lo, hi)
        elif kind == Kind.INSTANTIATE: self._instantiate(p, menu.cands[action.arg][0], self._pool_terms(menu, action, at_least=1))
        elif kind == Kind.USE: p.exists(w.T(self._pool_terms(menu, action, exactly=1)[0]))
        else: raise ActionError('unknown action kind %r' % (kind,))

    def _pool_terms(self, menu, action, at_least=0, exactly=None):
        """the kernel handles of the pool terms an action names"""
        n = len(action.args)
        if (exactly is not None and n != exactly) or n < at_least: raise ActionError('%r: wrong number of terms' % (action,))
        pool = menu.pool
        for i in action.args:
            if not (isinstance(i, int) and 0 <= i < len(pool)): raise ActionError('%r: no term %r in the pool' % (action, i))
        return [pool.terms[i] for i in action.args]

    def _instantiate(self, p, source, terms):
        """add the fact  source t1 .. tn  to the hypotheses of the first goal (`have`): the proof is the application, the statement is read off it. A fact that is already a hypothesis is refused."""
        w = self.w; ps = p.ps; n = ps.goal_num_hyps(0)
        if source is None: raise ActionError('INSTANTIATE: no candidate')
        with p.focus():
            proof = w.app(w.T(source), *[w.T(t) for t in terms])
        nm = '%si%d' % (self.name_tag, n)
        while nm in p._names[ps.goal_meta_id(0)]: nm += '_'
        before = {ps.goal_hyp_type(0, h) for h in range(n)}
        p.have(nm, proof)
        if ps.goal_hyp_type(0, ps.goal_num_hyps(0) - 1) in before: raise ActionError('INSTANTIATE: the fact is already a hypothesis')

    def _close_goal(self, p, i):
        """close goal i if a hypothesis (up to computation) or reflexivity does; True if it did"""
        k = self.k; ps = p.ps; w = self.w
        n = ps.goal_num_hyps(i); target = ps.goal_target(i); meta = ps.goal_meta_id(i)
        types = [ps.goal_hyp_type(i, h) for h in range(n)]
        saved = k.get_fuel(); k.set_fuel(SETTLE_FUEL)
        try:
            tshape = self.shape_fp(target)
            for lvl in range(n - 1, -1, -1):
                if types[lvl] != target:
                    hshape = self.shape_fp(types[lvl])
                    if tshape is not None and hshape is not None and hshape != tshape: continue            # not even the same kind of statement: no need to compute
                    if not self._conv(types[lvl], target, n): continue
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
        ps = p.ps; lvl = ps.goal_num_hyps(i); nm = '%sa%d' % (self.name_tag, lvl)
        while nm in p._names[ps.goal_meta_id(i)]: nm += '_'
        if not unfolds_to_pi(self.k, ps.goal_target(i)): return False
        try: ps.intro(nm, ps.goal_meta_id(i), True)
        except E.FuelExhausted: raise                                       # (running out of work is not "no quantifier left": the goals would differ with the budget)
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

    def check_action(self, action, menu, n_hyps):
        """the arguments of an action must mean something in this menu; raises ActionError if they do not -- an unknown kind, an index out of range, a candidate of the wrong sort (an interval where a lemma
        is wanted), a hypothesis that does not exist, terms where the kind takes none. (The menu need not LIST the action: what it lists is a suggestion, what it can carry out is the tactic layer's.) A
        policy that emits an arbitrary number gets a refusal, never an exception from the inside of the engine."""
        try: kind = Kind(action.kind)
        except ValueError: raise ActionError('%r: unknown kind of action' % (action,)) from None
        arg = action.arg
        if isinstance(arg, bool) or not isinstance(arg, (int, np.integer)): raise ActionError('%r: the argument is not an integer' % (action,))
        if kind in NEEDS_CANDIDATE:
            if not 0 <= arg < len(menu.cands): raise ActionError('%r: no candidate %d (the menu has %d)' % (action, arg, len(menu.cands)))
            term, _, aux = menu.cands[arg]
            if (aux is not None and term is None) != (kind == Kind.INTERVAL_CASES): raise ActionError('%r: candidate %d is not one this kind of action takes' % (action, arg))
        elif kind in NEEDS_HYPOTHESIS:
            if not 0 <= arg < n_hyps: raise ActionError('%r: no hypothesis %d (the goal has %d)' % (action, arg, n_hyps))
        if action.args and kind not in NEEDS_TERMS: raise ActionError('%r: this kind of action takes no terms' % (action,))
        ACTS.check_shape(self, action, n_hyps)

    def step_in(self, p, action, menu):
        """apply an action to the first goal of a live proof state, clean up after it and settle the goals it made: returns the number of goals left in its place"""
        before = p.n_goals; n_before = p.ps.goal_num_hyps(0)
        self.check_action(action, menu, n_before)
        consumes = self.split_consumes(p.ps.goal_hyp_type(0, action.arg)) if action.kind in (Kind.CASES, Kind.INDUCT) else False
        info = self._run(p, action, menu)
        made = p.n_goals - before + 1
        if self.clean:
            if action.kind in (Kind.CASES, Kind.INDUCT): self._clean_split(p, action.arg, n_before, made, consumes)
            elif action.kind == Kind.INTERVAL_CASES: self._clean_interval(p, made, n_before)
            elif action.kind == Kind.REWRITE_AT: self._clear_dead(p, 0, {action.hyps[0]})          # the hypothesis that was rewritten is replaced by the rewritten one
        made = self.settle(p, made)
        if self.clean and action.kind == Kind.INDUCT_WITH: ACTS.clean_induct(self, p, info, made, n_before)       # (after settling: the side premises of the principle are closed by the hypotheses first)
        self._unique_names(p, made)
        if p.ps.num_pending(): raise ActionError('a unification constraint is left pending: the goals are not independent')
        return made

    def _unique_names(self, p, count):
        """give the first `count` goals hypotheses with different names. (Names are for display; but a goal that shows two `h6` cannot be read. The tactic layer names the variables that cases and
        induction bring in by their position -- a name that an older hypothesis, revived by a revert, may have had; and the layer allows names to be reused, as the author's scripts do.) The later of two
        equal names gets an underscore more; the kernel is told, as it keeps the names of reverted hypotheses."""
        ps = p.ps
        for i in range(min(count, p.n_goals)):
            names = p._names[ps.goal_meta_id(i)]
            if len(set(names)) == len(names): continue
            seen = set()
            for h, nm in enumerate(names):
                if nm in seen:
                    while nm in seen or nm in names: nm += '_'
                    names[h] = nm; ps.set_goal_hyp_name(i, h, nm)
                seen.add(nm)

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
        return self.probe_status(g, action, menu)[0]

    def probe_status(self, g, action, menu, fuel=None):
        """`probe` with the reason and a chosen work budget: (the goals it leaves, ST_OK) or (None, why) -- ST_REFUSED: the tactic or the kernel said no, the action changed nothing or its goals are not
        independent; ST_FUEL: the work budget ran out (a bigger one might succeed: results are monotone in the budget); ST_BUG: something other than a refusal went wrong (see `bugs`)"""
        k = self.k; w = self.w
        cp = k.checkpoint(); old_fuel = k.get_fuel(); w.probing += 1
        refused = None
        try:
            k.set_fuel(self.fuel_of(action) if fuel is None else int(fuel))
            try:
                p = Proof.from_goal(w, g.target, g.hyp_types, g.hyp_names)             # (also this takes work: a budget that is too small for the goal itself is "out of work", not a bug)
                self.step_in(p, action, menu)
            except REFUSALS as e: refused = e
            else: children = k.finish_probe(cp, p.ps)
        except Exception as e:                       # not a refusal: something in this code or in the kernel's boundary is wrong
            k.rewind(cp)
            self.stats['probe_failed'] += 1; self.stats['probe_bug'] += 1
            self._bug(e)
            return None, ST_BUG
        finally:
            w.probing -= 1; k.set_fuel(old_fuel)
        if refused is not None:
            self.errors[type(refused).__name__] += 1
            k.rewind(cp)
            self.stats['probe_failed'] += 1
            if isinstance(refused, E.FuelExhausted) or getattr(refused, 'fuel', False): self.stats['probe_fuel'] += 1; return None, ST_FUEL
            return None, ST_REFUSED
        if children is None: self.stats['probe_dependent'] += 1; return None, ST_REFUSED
        out = [GoalData(t, hs, ns) for (t, hs, ns) in children]
        if any(c.key == g.key for c in out): self.stats['probe_noprogress'] += 1; return None, ST_REFUSED        # the goal came back unchanged: nothing was done
        self.stats['probe_ok'] += 1
        return out, ST_OK

    def fuel_of(self, action):
        """the work budget of an action (steps of the kernel's counter: an integer)"""
        return int(self.probe_fuel.get(action.kind, self.default_fuel))

    def fuel_levels(self, kind):
        """the work budgets a probe of this kind is tried with, smallest first, the last being `fuel_of`: most actions that fail do so at once or not at all, so a search tries the small budget on every
        action first and a larger one only where the small one ran out (a probe's outcome never depends on the budget except by running out of it)"""
        top = int(self.probe_fuel.get(kind, self.default_fuel))
        return tuple(sorted({max(1, top // 256), max(1, top // 64), max(1, top // 16), max(1, top // 4), top}))

    def canon_key(self, g):
        """the identity of a goal up to the order of the hypotheses that nothing refers to (neither the target nor another hypothesis mentions them): two goals that differ only by such an order
        are the same problem. Exact `key`s stay the identity for everything that refers to positions (menus, replay)."""
        ck = g.ckey
        if ck is None:
            k = self.k; n = g.n_hyps
            mentioned = set(atoms_of(k, g.target)[1])
            for t in g.hyp_types: mentioned |= atoms_of(k, t)[1]
            types = list(g.hyp_types); i = 0
            while i < n:
                if i in mentioned: i += 1; continue
                j = i
                while j < n and j not in mentioned: j += 1
                types[i:j] = sorted(types[i:j], key=lambda t: (t.tag, t.index))
                i = j
            ck = g.ckey = (g.target,) + tuple(types)
        return ck

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
    _KEEP = object()
    def __init__(self, engine, stmt, name=None, premises=_KEEP):
        """`premises`: the number of globals the agent may use in this episode (None: all); it is set on the engine before every operation of the episode, so that episodes with different premises can be
        interleaved on one engine. Default: whatever the engine has now, and it stays."""
        self.eng = engine
        self.premises = engine.premises if premises is Episode._KEEP else premises
        self._sync()
        self.stmt = engine.w.term(stmt, 'statement'); self.p = engine.start(self.stmt, name)
        self.steps = []; self._menu = None; self._goal = None
    def _sync(self): self.eng.premises = self.premises
    @property
    def done(self): return self.p.done and self.p.ps.is_complete()
    @property
    def n_goals(self): return self.p.n_goals
    def goal(self):
        if self._goal is None: self._sync(); self._goal = self.eng.goal_of(self.p, 0)
        return self._goal
    def menu(self):
        if self._menu is None: self._sync(); self._menu = self.eng.menu(self.goal())
        return self._menu
    def observe(self): self._sync(); return self.eng.observe(self.goal())
    def step(self, action):
        self._sync(); menu = self.menu(); g = self.goal()
        if self.eng.probe(g, action, menu) is None: raise ActionError('the action %r does not apply' % (action,))
        self.eng.replay(self.p, action, menu)
        self.steps.append(action); self._menu = None; self._goal = None
        return self.p.n_goals
    def finish(self):
        if not self.done: raise ActionError('the proof is not finished')
        self._sync()
        return self.eng.verify(self.stmt, self.p)


import gym_acts as ACTS          # (at the end: gym_acts needs the names above)
