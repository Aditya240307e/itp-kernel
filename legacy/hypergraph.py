"""
hypergraph.py -- Layer 1b: the policy-agnostic AND-OR hypergraph search
tree, built entirely on top of the Layer 1a nanobind surface (itp_env).

THE CORE MECHANICAL PROBLEM THIS FILE EXISTS TO SOLVE, AND HOW THE
DESIGN GOT HERE (kept deliberately, not just for history's sake -- the
two dead ends below are exactly the mistakes a future edit could easily
reintroduce)
---------------------------------------------------------------------
The kernel's own tactics are DESTRUCTIVE: cases(ps, hyp, target_meta_id=X)
REMOVES goal X from ps.goals and replaces it with its own subgoals. HTPS-
style Expansion needs to try several different candidate tactics against
the SAME goal, keeping every one that succeeds as its own HyperEdge --
so a first design (checkpoint/probe against the SAME, shared ProofState,
then rewind) was tried and, by direct test, found broken: rewind() only
ever undoes ExecutionContext's own state (NodePools, meta_ctx) -- it has
no visibility into ProofState at all, which is a separate object. A
probed intro() mutates ps.goals[idx] IN PLACE (replacing it with a
freshly-minted meta at the same position), and that mutation survives
any later rewind() untouched, permanently corrupting the "probed" goal.

The first fix -- forking a single goal out of the shared ProofState
(Kernel.fork_single_goal) before every probe, so probing never touches
the original -- solved that, but exposed a second, deeper problem when
wired into real search: a node can have several different candidate
tactics (several HyperEdges), and PUCT needs to be able to explore more
than one over time, exactly the way ordinary MCTS can revisit a parent
and try a different child. But re-applying a tactic to a parent's own,
single "live_meta_id" CONSUMES that goal -- so once one edge is
genuinely applied, the underlying kernel goal is gone, and a second
edge from the very same node can never be materialized again. That is
a real loss of search quality, not an acceptable simplification.

The actual fix: a node's own (target, hyp_types, hyp_names) are already
fully, immutably captured the moment it is first discovered. Nothing
about exploring it -- either for further discovery, or for genuinely
walking into it -- needs to trace back through a chain of parent live
state at all. Kernel.materialize_from_node rebuilds a FRESH goal (fresh
ElabCtx, fresh meta) directly from that stored data, independent of
whatever has happened to any parent or sibling. Every node can
therefore be independently, repeatedly rebuilt, as many times as the
search wants, from nothing but its own already-known shape.

One consequence worth being explicit about: this means NOTHING during
the search loop itself (selection, expansion, backup) ever touches the
caller's own, real ProofState. The search operates entirely on
ephemeral, per-call forks and the Python-side hypergraph's own
bookkeeping (status, N, W, priors). Only once the root is solved does
anything commit back to the real ProofState -- commit_solution, below,
walks the winning path ONE time and genuinely applies each tactic, in
order, producing a real, kernel-checkable proof term. This also happens
to remove a risk flagged in this file's own earlier draft (find_goal_index's
linear scan degrading as ps.goals grows with total search size) rather
than just deferring it: ps.goals now only ever grows along the one,
final, winning path, never with how much of the search tree was
explored.

A DIFFERENT KIND OF GROWTH THAT IS CORRECT, NOT A BUG: the underlying
kernel-level NodePools genuinely, permanently grow with every
SUCCESSFUL discovery, for the whole duration of the search -- expand()
deliberately does NOT rewind after a successful probe (see its own
comment). This was, at first, assumed to be avoidable the same way a
failed probe's own state is reclaimed, and checkpoint()/rewind() was
tried around every probe unconditionally -- but a discovered child's
own (target, hyp_types) are read off AFTER the tactic succeeds and then
stored, long-term, on its own Node object. Rewinding past them reclaims
their own backing pool slots for reuse, leaving those stored NodeIds
dangling; a later, unrelated allocation can silently reoccupy the same
slot, and the corruption this produces (confirmed directly, once) is
not a clean failure -- it is a garbled term read back somewhere
completely unrelated. So a successful probe's own state must be kept,
permanently, for as long as the hypergraph might still reference it --
which, in practice, means for the rest of the search. This is real,
necessary memory for what the search has actually found, not wasted
work -- unlike a failed probe's own state (still safely reclaimed) or
the earlier hash-consing-growth problem this session already solved
elsewhere (which was specifically about DISCARDED, backtracked
attempts, not about remembering genuine discoveries).
"""

from __future__ import annotations
from enum import IntEnum
from typing import Optional
import random
import numpy as np
import itp_env
from policy_api import GoalRequest


# ---------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------

class ActionKind(IntEnum):
    INTRO = 0
    CASES = 1
    INDUCT = 2
    APPLY = 3
    REWRITE = 4


class Action:
    """A small, hashable (kind, arg) pair -- used as HyperEdge dict
    keys. arg is unused for INTRO, a hypothesis index for CASES/INDUCT,
    and an index into a Node's own cached_candidates for APPLY (resolved
    there rather than re-querying apply_candidates() -- its own
    ordering is deterministic within one episode, since the library
    never grows mid-search, but re-running it on every use would still
    be wasteful)."""
    __slots__ = ('kind', 'arg')

    def __init__(self, kind: ActionKind, arg: int = 0):
        self.kind = kind
        self.arg = arg

    def __eq__(self, other):
        return isinstance(other, Action) and self.kind == other.kind and self.arg == other.arg

    def __hash__(self):
        return hash((int(self.kind), self.arg))

    def __repr__(self):
        return f"Action({self.kind.name}, {self.arg})"


# ---------------------------------------------------------------------
# Node / HyperEdge / Status
# ---------------------------------------------------------------------

class Status(IntEnum):
    UNSOLVED = 0
    SOLVED = 1
    INVALID = 2


class HyperEdge:
    """One (goal, tactic) pair. children is empty exactly when the
    tactic solves the goal outright (e.g. exact-style closure via
    apply(refl)).

    children CAN include the very Node this edge hangs off of, or an
    ancestor further up the tree: get_or_create_node_from_data dedups
    every goal by structure, so a tactic that makes no real progress
    (e.g. inducting on a hypothesis the target doesn't depend on) or
    one whose subgoal happens to structurally coincide with something
    already seen elsewhere legitimately produces a self-loop or a
    longer cycle in this object graph -- confirmed directly: a
    real curriculum run left a Node whose own INDUCT edge's children
    list contained that same Node. This is correct hypergraph
    structure, not a bug, and Python's ordinary reference counting
    cannot free a cycle like that on its own -- it needs the cyclic
    garbage collector (the `gc` module), which runs automatically and
    was confirmed, directly, to reclaim exactly this shape of cycle
    completely (RSS stays flat across many rounds with zero manual
    gc.collect() calls; see PHASE4_HYGIENE_NOTES.md). No weakref or
    other restructuring is used here to avoid the cycle -- doing so
    would trade a already-solved non-problem for real weakref-lifetime
    risk (a node dropped while something else still needed it)."""
    __slots__ = ('prior', 'N', 'W', 'virtual_count', 'children')

    def __init__(self, prior: float, children: list["Node"]):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.virtual_count = 0
        self.children = children


class Node:
    """One goal, identified entirely by its own (target, hyp_types,
    hyp_names) -- deliberately carries NO live kernel handle of any
    kind. Every kernel-touching operation on a node (expand, or the
    final commit) rebuilds a fresh goal directly from this stored data
    via Kernel.materialize_from_node; see the module docstring for why
    a persistent "live_meta_id" field was tried and abandoned."""
    __slots__ = ('dedup_key', 'target', 'hyp_types', 'hyp_names', 'observation',
                 'status', 'tactics', 'critic_value', 'cached_candidates', 'cached_value',
                 'cached_rewrite_extra')

    def __init__(self, dedup_key: bytes, target, hyp_types: list, hyp_names: list[str], observation):
        self.dedup_key = dedup_key
        self.target = target
        self.hyp_types = hyp_types
        self.hyp_names = hyp_names
        self.observation = observation
        self.status = Status.UNSOLVED
        self.tactics: dict[Action, HyperEdge] = {}
        self.critic_value: Optional[float] = None
        self.cached_candidates: list = []
        # REWRITE-specific: candidates[i] is always a plain NodeId (the eq_term itself) so
        # term_features(candidates) keeps working unchanged for every existing (APPLY-only)
        # caller; the (x, y, transport_gid, symm_gid) a REWRITE candidate also needs live here
        # instead, keyed by the SAME index into candidates -- x/y specifically must be exactly the
        # flat, never-eval()'d terms enumerate_legal_actions already constructed (see rewrite_eq's
        # own extensive comment on why re-deriving them later, e.g. via infer() on eq_term, would
        # silently reconstruct a DIFFERENT, eagerly-unfolded NodeId instead).
        self.cached_rewrite_extra: dict = {}
        # Starts at the "no edges yet" fallback (0.5); expand() updates
        # it once critic_value is known, and update_status recomputes
        # it from direct children's OWN cached_value whenever this
        # node's own edges change -- see update_status's own comment
        # for why this replaces the earlier, uncached, full-subtree
        # recursive version.
        self.cached_value: float = 0.5


def dedup_key_for(target, hyp_types: list) -> bytes:
    """A cheap, exact key: NodeId.tag/index pairs concatenated. Two
    goals are the SAME node iff their own target and hyp types are
    identical NodeIds -- exact, not approximate, given hash-consing's
    own guarantee that structurally-equal terms already share identical
    NodeIds."""
    parts = [f"{target.tag}:{target.index}"]
    for h in hyp_types:
        parts.append(f"{h.tag}:{h.index}")
    return "|".join(parts).encode()


def get_or_create_node(kernel: "itp_env.Kernel", ps, idx: int, dedup_map: dict) -> Node:
    """Reads goal at position idx of some live ProofState (ps) into a
    (possibly deduplicated) Node, caching its own observation() right
    now -- the underlying kernel goal this was read from may not exist
    by the time this Node is looked at again (e.g. an ephemeral,
    about-to-be-rewound probe).

    Stores target/hyp_types EXACTLY as the kernel produced them, with
    NO normalization -- normalizing here was tried first, and found,
    by direct test, to be unsound: it unfolds global Fix definitions
    (e.g. add) along with reducing the stuck beta-redex cases()/
    induct() leave behind, and since a search applies cases()/induct()
    repeatedly, each subsequent normalization unfolds the SAME
    definition one layer further than the last, permanently baking an
    ever-deeper stuck-match reconstruction into the node's own,
    long-lived target -- exactly the failure mode intro()'s own
    comment describes having chased across earlier sessions. See
    enumerate_legal_actions for where a normalized form is still
    needed (apply_candidates' own discrimination-tree lookup), computed
    there as a transient, throwaway value instead."""
    target = ps.goal_target(idx)
    hyp_types = [ps.goal_hyp_type(idx, h) for h in range(ps.goal_num_hyps(idx))]
    hyp_names = [ps.goal_hyp_name(idx, h) for h in range(ps.goal_num_hyps(idx))]
    key = dedup_key_for(target, hyp_types)
    if key in dedup_map:
        return dedup_map[key]
    obs = kernel.observation(ps, target_meta_id=ps.goal_meta_id(idx))
    node = Node(key, target, hyp_types, hyp_names, obs)
    dedup_map[key] = node
    return node


def get_or_create_node_from_data(kernel: "itp_env.Kernel", target, hyp_types: list, hyp_names: list,
                                  dedup_map: dict) -> Node:
    """Like get_or_create_node, for a goal already read off a (now
    possibly discarded) probe as plain (target, hyp_types, hyp_names) --
    what Kernel.finish_probe returns. The observation is computed only on
    a dedup miss, directly from the node ids (Kernel.observe_goal), so no
    live ProofState is needed and nothing is computed for revisited goals."""
    key = dedup_key_for(target, hyp_types)
    node = dedup_map.get(key)
    if node is None:
        node = Node(key, target, hyp_types, hyp_names, kernel.observe_goal(target, hyp_types))
        dedup_map[key] = node
    return node


# Whether expand() reclaims the proof-term scaffolding a successful probe
# leaves behind (Kernel.finish_probe -> itp::compact_since). On by default;
# switchable so behavior can be A/B-compared and bisected. `compaction_stats`
# counts finish_probe outcomes by status (0 compacted, 1 kept: older meta
# solved, 2 kept: live term mentions a post-checkpoint meta, 3 disabled).
COMPACT_PROBES = True
compaction_stats = {0: 0, 1: 0, 2: 0, 3: 0}

# ---------------------------------------------------------------------
# Rewrite tactic configuration: set once, before search, by whichever script builds the curriculum
# (curriculum.py knows each lemma's own structure best, having just built it) -- mirrors
# COMPACT_PROBES's own "module-level, set-once" convention, rather than threading new parameters
# through enumerate_legal_actions/_apply_tactic's entire call chain (prepare_expansion -> expand ->
# search_steps -> search), which every existing caller would otherwise need updating for.
#
# REWRITE_TRANSPORT_GID / REWRITE_SYMM_GID: the two library facts rewrite_eq itself needs (see
# itp_kernel.hpp's own rewrite_eq, and curriculum.py's setup_transport / CURRICULUM's own symm).
# REWRITE_EQ_CONST: refs['eq_ty'] itself (mk_const(eq_id)) -- enumerate_legal_actions needs this to
# recognize an Eq-typed hypothesis by comparing decompose_app's own head against it directly, and
# has no other access to refs (its own signature takes only kernel/ps/meta_id/library).
REWRITE_TRANSPORT_GID: int = None
REWRITE_SYMM_GID: int = None
REWRITE_EQ_CONST = None

# REWRITE_CANDIDATE_FNS: each entry is a callable(kernel, ps, idx, hyp_types, hyp_levels) ->
# list[(eq_term, x, y)], called once per node expansion. hyp_types/hyp_levels are the CURRENT
# goal's own hypotheses (raw, never-evaluated types -- exactly as needed, see rewrite_eq's own
# extensive comment on why x/y must come from flat, never-eval()'d terms, not infer()'s own
# reconstruction). Each registered function decides for itself which combinations of in-scope
# hypotheses to try instantiating its own lemma's leading arguments with, and constructs x/y
# directly the same, safe way curriculum.py's own hand-built proofs already do.
REWRITE_CANDIDATE_FNS: list = []


# ---------------------------------------------------------------------
# Policy interface -- deliberately minimal, matching the policy-
# agnostic goal: anything satisfying this one method works.
# ---------------------------------------------------------------------

class PolicyInterface:
    """The sole integration point between the search (this module) and
    any policy/value model -- anything implementing this one method
    works, regardless of what's behind it (a from-scratch trained
    model, a hand-written heuristic, a call to an external service).
    This is the actual "policy agnostic" contract: nothing else in
    this module inspects what concrete type `policy` is."""
    def predict(self, observation, actions: list[Action], candidates: list) -> tuple[dict[Action, float], float]:
        """observation: this node's own kernel DAG dump -- opaque to
        this module, whatever kernel.observation() returned (as of
        this module's own current use, a 5-tuple of numpy arrays;
        callers should not hardcode that shape, since it is the
        kernel's own contract to define, not this module's).

        actions: the full, enumerated candidate action menu for this
        node (enumerate_legal_actions' own return, before any top_k
        truncation) -- what the returned priors dict assigns
        probabilities over.

        candidates: apply_candidates()'s own NodeId list. Required,
        not optional: an Action(APPLY, i)'s own identity IS index i
        into this list (which term gets applied) -- without it, two
        different APPLY actions are indistinguishable to a policy.

        Returns (priors, value): priors maps actions in `actions` to a
        score (need not be normalized -- expand() only ever compares
        them via .get(action, 0.0) for ranking); value is a scalar
        critic estimate in [0, 1]."""
        raise NotImplementedError

    def predict_batch(self, requests) -> list:
        """The Layer 3 contract (see policy_api.py): score many goals at once.
        Returns one (priors, value) per request; priors is a dict {action: score}
        or a sequence aligned with request.actions.

        The default simply loops the one-goal predict() above, so every policy
        written against predict() keeps working unchanged. A policy that can
        score a batch more cheaply than one at a time (any model) should
        override THIS instead -- and then need not implement predict() at all."""
        if type(self).predict is PolicyInterface.predict:
            raise NotImplementedError(f"{type(self).__name__} must implement predict_batch() or predict()")
        return [self.predict(r.observation, list(r.actions), list(r.candidates)) for r in requests]


class UniformPolicy(PolicyInterface):
    """Trivial, fully-supported baseline -- matches HTPS's own "no
    critic" ablation (works, just worse), a legitimate configuration,
    not a degenerate one."""
    def predict(self, observation, actions: list[Action], candidates: list) -> tuple[dict[Action, float], float]:
        if not actions:
            return {}, 0.5
        p = 1.0 / len(actions)
        return {a: p for a in actions}, 0.5


# ---------------------------------------------------------------------
# Kernel-facing helpers
# ---------------------------------------------------------------------

def enumerate_legal_actions(kernel: "itp_env.Kernel", ps, meta_id: int,
                             library: "itp_env.Library") -> tuple[list[Action], list, dict]:
    """ps here is always a single-goal ProofState built by
    materialize_from_node, so meta_id is always at index 0 -- but this
    still looks it up properly rather than assuming, since ps is an
    ordinary parameter, not guaranteed-fresh by this function's own
    type. Returns (actions, candidates, rewrite_extra) -- candidates is
    a single, uniform NodeId list (apply_candidates()'s own entries,
    plus one eq_term per REWRITE candidate), needed by expand() to
    populate Node.cached_candidates; rewrite_extra is a separate dict,
    keyed by the SAME index into candidates, carrying each REWRITE
    candidate's own (x, y, transport_gid, symm_gid), for
    Node.cached_rewrite_extra -- see that slot's own comment for why
    this lives separately rather than folded into candidates itself."""
    idx = _goal_index(ps, meta_id)
    actions = [Action(ActionKind.INTRO, 0)]  # always attempted; may fail (goal not a Pi type), caught during probing
    raw_target = ps.goal_target(idx)
    raw_hyp_types = [ps.goal_hyp_type(idx, h) for h in range(ps.goal_num_hyps(idx))]
    for h in range(ps.goal_num_hyps(idx)):
        if not kernel.is_inductive_family_type(raw_hyp_types[h]):
            continue
        # No references_level_in_target pre-filter here: an earlier
        # version of this guard skipped offering CASES/INDUCT whenever
        # a hypothesis didn't syntactically appear in the target,
        # reasoning that induct()'s own branch_expected/ih_ty
        # construction produces an ill-scoped subgoal in exactly that
        # situation (confirmed by a real crash, at the time). That
        # guard was ALSO confirmed, directly, to incorrectly block a
        # completely legitimate, necessary case-split: casing on
        # h:Eq A x y to prove "Eq A y x" (symm's own proof) -- h never
        # appears in the target at all, yet forcing x=y via refl's own
        # index pattern is exactly what the proof needs. "A hypothesis
        # absent from the target is never useful" is false in general;
        # it only happened to hold for the specific scenario the
        # now-removed guard was built around. is_target_well_scoped
        # (below, in expand()) is a general safety net that catches the
        # actual malformed-subgoal condition directly, regardless of
        # why it arose, and was already independently confirmed
        # sufficient on its own -- so nothing is lost by removing this
        # narrower, over-eager pre-filter and relying on that instead.
        actions.append(Action(ActionKind.CASES, h))
        actions.append(Action(ActionKind.INDUCT, h))

    # apply_candidates' own discrimination-tree lookup matches purely
    # on syntactic head, so it finds nothing against a goal left as an
    # un-reduced beta-redex by a prior cases()/induct() (confirmed
    # directly). Querying it against the REAL goal would need
    # permanently normalizing that goal's own stored target -- tried,
    # and found unsound (see get_or_create_node's own comment: it
    # compounds into an ever-deeper baked-in reconstruction of global
    # Fix definitions across repeated cases()/induct() calls). Instead,
    # build one throwaway fork with a NORMALIZED target purely for this
    # lookup, then discard it entirely (checkpoint/rewind): the
    # returned candidates (library lemma references, or mk_var(level)
    # for an in-scope hypothesis) are meaningful independent of which
    # presentation of the goal found them, since normalization changes
    # neither a hypothesis's own level nor a lemma's own identity --
    # apply()'s own real unification, later, still runs against the
    # REAL, unnormalized goal, and already handles that fine (confirmed
    # directly: apply(refl) succeeds against an un-reduced target).
    normalized_target = kernel.normalize_in_context(raw_target, raw_hyp_types)
    # No checkpoint/rewind here: this lookup always succeeds (it
    # returns a list, possibly empty, never raises), and its own
    # resulting NodeIds -- in particular, apply_candidates' own
    # hypothesis-matching loop freshly constructs mk_var(level) for
    # each in-scope hypothesis it proposes -- get stored, long-term, on
    # node.cached_candidates. Rewinding here would risk invalidating
    # whichever of those NodeIds happened to be newly constructed by
    # THIS call (rather than already existing from an earlier one) --
    # the same class of bug already found and fixed for a successful
    # discovery probe (see expand()'s own comment).
    lookup_fork = kernel.materialize_from_node(normalized_target, raw_hyp_types,
                                                [ps.goal_hyp_name(idx, h) for h in range(ps.goal_num_hyps(idx))])
    candidates = kernel.apply_candidates(lookup_fork, library, target_meta_id=lookup_fork.root_meta_id)

    for i in range(len(candidates)):
        actions.append(Action(ActionKind.APPLY, i))

    # REWRITE candidates, appended to the SAME candidates list (continuing its own indexing), but
    # as a plain NodeId each time (eq_term alone) -- NOT a tuple -- so term_features(candidates),
    # just below in prepare_expansion, keeps working completely unchanged for every existing,
    # APPLY-only caller: it needs a uniform list of NodeId. The (x, y, transport_gid, symm_gid)
    # each REWRITE candidate ALSO needs lives in rewrite_extra instead, keyed by the same index --
    # see Node.cached_rewrite_extra's own comment for why x/y specifically cannot be re-derived
    # later from eq_term (e.g. via infer()) without silently reconstructing a different NodeId.
    # A no-op (contributes zero actions, empty rewrite_extra) whenever REWRITE_TRANSPORT_GID is
    # unset, exactly as every existing curriculum goal (none of which use rewrite) already needs.
    rewrite_extra: dict = {}
    # Restricted to goals with no remaining leading Pi: detail::generalize's own fresh motive
    # variable (inside rewrite_eq) is assigned level g.ec.size(), which COLLIDES with a
    # still-un-intro'd Pi's own, pre-assigned bound-variable level at that exact value (confirmed
    # directly: offering rewrite on a Pi-headed goal silently replaced that Pi's own bound
    # variable, everywhere it appeared, with whatever "y" happened to be -- the classic signature
    # of two logically distinct binders sharing one level number). cases()/induct() generalize
    # g.target the identical way and so carry the identical risk -- both now reject outright,
    # kernel-side, exactly this situation (goal's own target still a Pi), which is the more
    # fundamental fix: offering REWRITE here is this file's own, analogous guard so that search
    # never has to pay for a round trip into the kernel to discover the same thing. INTRO itself
    # remains offered regardless (see the top of this function) so search can still peel the Pi
    # away first and reach rewrite (or cases/induct) on the very next node.
    if REWRITE_TRANSPORT_GID is not None and not kernel.is_pi(raw_target):
        for h in range(ps.goal_num_hyps(idx)):
            parts = kernel.decompose_app(raw_hyp_types[h])
            if len(parts) != 4 or parts[0] != REWRITE_EQ_CONST:
                continue
            hyp_ref = kernel.mk_var(h)
            x_actual, y_actual = parts[2], parts[3]
            # forward: replace x_actual with y_actual, using the hypothesis directly
            candidates.append(hyp_ref)
            rewrite_extra[len(candidates) - 1] = (x_actual, y_actual, REWRITE_TRANSPORT_GID, REWRITE_SYMM_GID)
            actions.append(Action(ActionKind.REWRITE, len(candidates) - 1))
            # backward: replace y_actual with x_actual -- needs symm(A,x,y,h), not h itself (h's
            # own type is "Eq A x_actual y_actual", not "Eq A y_actual x_actual")
            symm_term = kernel.mk_app(kernel.mk_app(kernel.mk_app(kernel.mk_app(
                kernel.mk_const(REWRITE_SYMM_GID), parts[1]), x_actual), y_actual), hyp_ref)
            candidates.append(symm_term)
            rewrite_extra[len(candidates) - 1] = (y_actual, x_actual, REWRITE_TRANSPORT_GID, REWRITE_SYMM_GID)
            actions.append(Action(ActionKind.REWRITE, len(candidates) - 1))

        for fn in REWRITE_CANDIDATE_FNS:
            for (eq_term, x, y) in fn(kernel, ps, idx, raw_hyp_types, list(range(ps.goal_num_hyps(idx)))):
                candidates.append(eq_term)
                rewrite_extra[len(candidates) - 1] = (x, y, REWRITE_TRANSPORT_GID, REWRITE_SYMM_GID)
                actions.append(Action(ActionKind.REWRITE, len(candidates) - 1))

    return actions, candidates, rewrite_extra


def _goal_index(ps, meta_id: int) -> int:
    """Linear scan -- fine now that ps is always either a freshly-built,
    single-goal fork (used throughout search) or the real ProofState
    during the one-time final commit (where it only ever grows along
    the single winning path, not with total search size -- the risk
    this function's own earlier version was flagged for no longer
    applies, see the module docstring)."""
    for i in range(ps.num_goals()):
        if ps.goal_meta_id(i) == meta_id:
            return i
    raise ValueError(f"_goal_index: meta_id {meta_id} not found in ps.goals")


def _apply_tactic(kernel: "itp_env.Kernel", ps, meta_id: int, action: Action, candidates: list,
                   rewrite_extra: dict = None):
    if action.kind == ActionKind.INTRO:
        ps.intro('', target_meta_id=meta_id)
    elif action.kind == ActionKind.CASES:
        ps.cases(action.arg, target_meta_id=meta_id)
    elif action.kind == ActionKind.INDUCT:
        ps.induct(action.arg, target_meta_id=meta_id)
    elif action.kind == ActionKind.APPLY:
        ps.apply(candidates[action.arg], target_meta_id=meta_id)
    elif action.kind == ActionKind.REWRITE:
        eq_term = candidates[action.arg]
        x, y, transport_gid, symm_gid = rewrite_extra[action.arg]
        ps.rewrite(eq_term, x, y, transport_gid, symm_gid, target_meta_id=meta_id)
    else:
        raise ValueError(f"_apply_tactic: unrecognized action kind {action.kind}")


# ---------------------------------------------------------------------
# Discovery: the ONLY kernel-touching operation during search itself.
# Rebuilds a fresh goal from the node's own stored data, tries K
# candidate tactics against fresh, independent forks of it (checkpoint/
# apply/read-off/rewind each time), and records what it finds. Happens
# at most once per node -- never re-run, never needs a "live" anything.
# ---------------------------------------------------------------------

def goal_text(kernel: "itp_env.Kernel", node: Node) -> str:
    """Readable rendering of a goal, for GoalRequest.text() (LLM-style policies, debugging)."""
    lines = []
    for name, ty in zip(node.hyp_names, node.hyp_types):
        lines.append(f"{name or '_'} : {kernel.print_node(ty)}")
    lines.append("|- " + kernel.print_node(node.target))
    return "\n".join(lines)


class PreparedExpansion:
    """Everything expand() needs to remember between asking the policy for scores
    and acting on them: the node, its action menu, and the request that was sent."""
    __slots__ = ('node', 'actions', 'candidates', 'request')

    def __init__(self, node, actions, candidates, request):
        self.node = node
        self.actions = actions
        self.candidates = candidates
        self.request = request


_EMPTY_U32 = np.zeros(0, dtype=np.uint32)


def prepare_expansion(kernel: "itp_env.Kernel", node: Node, library: "itp_env.Library") -> PreparedExpansion:
    """The kernel-side half of expanding a node, up to (not including) asking the
    policy: rebuild the goal, enumerate its legal actions, and package a
    self-contained GoalRequest. APPLY candidates' (tag, scalar) features are
    resolved here, once, in a single kernel call, so no policy ever needs a
    kernel handle to embed them."""
    base = kernel.materialize_from_node(node.target, node.hyp_types, node.hyp_names)
    actions, candidates, rewrite_extra = enumerate_legal_actions(kernel, base, base.root_meta_id, library)
    node.cached_candidates = candidates
    node.cached_rewrite_extra = rewrite_extra
    if candidates:
        cand_tags, cand_scalars = kernel.term_features(candidates)
    else:
        cand_tags = cand_scalars = _EMPTY_U32
    request = GoalRequest(observation=node.observation, actions=actions, candidates=candidates,
                          candidate_tags=cand_tags, candidate_scalars=cand_scalars,
                          _render=lambda: goal_text(kernel, node))
    return PreparedExpansion(node, actions, candidates, request)


def _normalize_response(actions: list, response) -> tuple[dict, float]:
    """Accepts (priors, value) with priors either a dict {action: score} or a sequence/array
    aligned with `actions` (the array form is what array-native and out-of-process policies
    return); always yields a dict and a plain float."""
    priors, value = response
    if not isinstance(priors, dict):
        vals = np.asarray(priors, dtype=np.float64).reshape(-1)
        if len(vals) != len(actions):
            raise ValueError(f"policy returned {len(vals)} priors for {len(actions)} actions")
        priors = {a: float(v) for a, v in zip(actions, vals)}
    return priors, float(value)


def finish_expansion(kernel: "itp_env.Kernel", prepared: PreparedExpansion, response, dedup_map: dict,
                     top_k: int = 16, dirichlet_epsilon: float = 0.0, dirichlet_alpha: float = 0.3):
    """The other half: given the policy's (priors, value) for `prepared`, rank the actions,
    probe the top_k of them against fresh forks of the goal, and record the resulting
    hyperedges."""
    node, actions, candidates = prepared.node, prepared.actions, prepared.candidates
    priors, value = _normalize_response(actions, response)
    node.critic_value = value

    if dirichlet_epsilon > 0.0 and actions:
        # Standard AlphaZero-style root exploration noise: mixes the
        # model's own prior with a Dirichlet sample over the SAME
        # action set, so every action retains a genuine minimum
        # exploration probability regardless of how confident (or
        # wrong) the raw model score is. Mixing the WHOLE distribution
        # here -- before ranking, not just before storing edge.prior
        # afterward -- matters concretely: an action the model scores
        # so low it would never make the top_k cut during discovery
        # could never benefit from noise added only to an edge that
        # never gets created in the first place. alpha=0.3, epsilon=
        # 0.25 are the published AlphaZero defaults (tuned for a much
        # larger, ~250-move action space) -- a reasonable starting
        # point here, not something rigorously retuned for this
        # project's own, much smaller action spaces; a genuinely
        # different scale might warrant revisiting this.
        #
        # Sampled via the standard Gamma-normalization construction
        # (draw independent Gamma(alpha, 1) variates, normalize to sum
        # to 1) using Python's own stdlib random module, deliberately
        # NOT torch/numpy -- this file has no dependency on either (see
        # its own module docstring / import list) and must stay usable
        # with UniformPolicy alone, without Layer 2 installed at all.
        gamma_samples = [random.gammavariate(dirichlet_alpha, 1.0) for _ in actions]
        gamma_total = sum(gamma_samples)
        noise = [g / gamma_total for g in gamma_samples]
        priors = {a: (1 - dirichlet_epsilon) * priors.get(a, 0.0) + dirichlet_epsilon * noise[i]
                  for i, a in enumerate(actions)}

    ranked = sorted(actions, key=lambda a: -priors.get(a, 0.0))[:top_k]

    for action in ranked:
        # The checkpoint comes FIRST, so the probe's own scratch goal (its fresh root
        # metavariable) is reclaimed along with everything else the probe builds.
        cp = kernel.checkpoint()
        probe = kernel.materialize_from_node(node.target, node.hyp_types, node.hyp_names)
        try:
            _apply_tactic(kernel, probe, probe.root_meta_id, action, candidates, node.cached_rewrite_extra)
        except itp_env.ITPError:
            kernel.rewind(cp)  # nothing was kept from this attempt -- safe to reclaim
            continue

        # finish_probe verifies every resulting subgoal is genuinely well-scoped before
        # trusting it (confirmed necessary directly: induct(), even restricted to a
        # hypothesis that DOES occur in the target, can still produce a subgoal whose
        # target references one level beyond what its own hyp count provides, crashing
        # once evaluated -- a general safety net for any tactic, not tied to induct()),
        # treating a violation exactly like a failed application (rewind, skip). On
        # success it also reclaims, when safe, the proof-term scaffolding the probe left
        # behind: the search only ever needs the resulting GOALS (targets and hypothesis
        # types), never the proof terms built to justify them -- measured directly, >=73%
        # of what a search left committed was such garbage. The probe state must not be
        # used after this call; the children come back as plain (target, hyps, names).
        finished = kernel.finish_probe(cp, probe, COMPACT_PROBES)
        if finished is None:
            continue
        status, child_data = finished
        compaction_stats[status] += 1
        children = [get_or_create_node_from_data(kernel, t, hyps, names, dedup_map) for (t, hyps, names) in child_data]
        node.tactics[action] = HyperEdge(prior=priors.get(action, 0.0), children=children)

    update_status(node)




def expand(kernel: "itp_env.Kernel", node: Node, library: "itp_env.Library",
           policy: PolicyInterface, dedup_map: dict, top_k: int = 16,
           dirichlet_epsilon: float = 0.0, dirichlet_alpha: float = 0.3):
    """Synchronous expansion of one node: prepare, ask the policy (a batch of one), finish."""
    prepared = prepare_expansion(kernel, node, library)
    (response,) = policy.predict_batch([prepared.request])
    finish_expansion(kernel, prepared, response, dedup_map, top_k, dirichlet_epsilon, dirichlet_alpha)


def edge_value(edge: HyperEdge) -> float:
    """Product of the edge's own DIRECT children's current cached_value
    -- O(1) per child, no recursion: cached_value already reflects each
    child's own status (1.0 solved, 0.0 invalid, else its own best
    edge's value), kept current incrementally by update_status. 1.0
    for an empty children list (a tactic that closed the goal
    outright: vacuously, nothing further needs to succeed)."""
    v = 1.0
    for c in edge.children:
        v *= c.cached_value
    return v


def update_status(node: Node):
    """Recomputes both node.status and node.cached_value from its own,
    CURRENT, DIRECT children only -- no recursion at all. Correct
    because this is always called bottom-up (deepest node first) along
    whichever nodes changed this round (see run_simulation's own
    backup loop, and expand()'s own call for a freshly-discovered
    leaf), so by the time a node's own update_status runs here, every
    one of its direct children already has an up-to-date status/
    cached_value -- either from THIS round (if it's also on the
    visited path, processed earlier in the same reversed loop) or an
    already-correct one from an earlier round (if untouched this
    round).

    This replaces an earlier version (node_value/edge_value, computing
    a node's value by recursing through its ENTIRE subtree on every
    call) confirmed, by direct measurement, to degrade severely: 20-
    simulation batches went from 0.076s to 21s as the hypergraph grew
    from 40 to 360 nodes -- clearly super-linear, not a constant-factor
    cost. Caching here, updated incrementally in the same single
    bottom-up pass backup already performs, makes each update O(this
    node's own direct edge/child count) instead of O(subtree size). It
    also structurally removes the separate cycle-recursion concern the
    old version needed an _in_progress guard for: reading a cached
    value is not a recursive call, so there is nothing left to loop.

    An edge is solved iff every one of its children is solved
    (vacuously true for an empty children list -- a tactic that closed
    the goal outright); an edge is invalid iff any of its children is
    invalid (AND-semantics: one impossible subgoal makes the whole
    tactic choice useless). A node is solved if ANY edge is solved;
    invalid if EVERY edge is invalid AND at least one edge exists (an
    unexpanded node -- no edges at all yet -- stays unsolved, not
    invalid, and keeps whatever cached_value expand() already gave it
    from the critic)."""
    if not node.tactics:
        return
    any_solved = False
    all_invalid = True
    best_value = 0.0
    for edge in node.tactics.values():
        edge_solved = all(c.status == Status.SOLVED for c in edge.children)
        edge_invalid = any(c.status == Status.INVALID for c in edge.children)
        if edge_solved:
            any_solved = True
        if not edge_invalid:
            all_invalid = False
        best_value = max(best_value, edge_value(edge))
    if any_solved:
        node.status = Status.SOLVED
        node.cached_value = 1.0
    elif all_invalid:
        node.status = Status.INVALID
        node.cached_value = 0.0
    else:
        node.cached_value = best_value


# ---------------------------------------------------------------------
# Selection: PUCT with first-play urgency (HTPS's own formula, grounded
# several turns ago against the actual paper). Pure Python-side
# bookkeeping -- touches the hypergraph only, never the kernel.
# ---------------------------------------------------------------------

def puct_score(edge: HyperEdge, sum_siblings_n: int, c_puct: float = 1.4) -> float:
    C = edge.N + edge.virtual_count
    if edge.N == 0:
        q = 0.5 / max(1, C)
    else:
        q = edge.W / max(1, C)
    return q + c_puct * edge.prior * (sum_siblings_n ** 0.5) / (1 + C)


def select_expandable_subtree(root: Node) -> tuple[list[Node], list[tuple[Node, Action, HyperEdge]]]:
    """Walks down from root by PUCT, collecting: the never-yet-expanded
    frontier leaves reached this round, and every (node, action, edge)
    visited along the way (for virtual-loss and backup bookkeeping).

    Cycle-safe: a structural cycle in the hypergraph (see node_value's
    own comment) could otherwise push the same node back onto the
    stack indefinitely within one call. visited_this_call ensures each
    node is processed at most once per walk; a cycle-back edge simply
    isn't descended into again this round (its own contribution is
    already reflected via whatever value it had from being visited
    earlier in the SAME walk, or from an earlier simulation's backup)."""
    frontier = []
    visited_edges = []
    visited_this_call = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.status != Status.UNSOLVED or id(node) in visited_this_call:
            continue
        visited_this_call.add(id(node))
        if not node.tactics:
            frontier.append(node)
            continue
        sum_n = sum(e.N for e in node.tactics.values())
        candidates = [(a, e) for a, e in node.tactics.items()
                      if any(c.status == Status.UNSOLVED for c in e.children)]
        if not candidates:
            continue  # every edge already fully resolved; update_status should have caught this, but defensive
        action, edge = max(candidates, key=lambda ae: puct_score(ae[1], sum_n))
        visited_edges.append((node, action, edge))
        edge.virtual_count += 1
        for child in edge.children:
            if child.status == Status.UNSOLVED:
                stack.append(child)
    return frontier, visited_edges


def drive(gen, policy):
    """Runs a search generator (see search_steps) to completion against a policy: each yielded
    list of GoalRequests is answered with one predict_batch call. Returns the generator's result."""
    try:
        requests = next(gen)
        while True:
            requests = gen.send(policy.predict_batch(requests))
    except StopIteration as stop:
        return stop.value


def _simulation_steps(kernel, root: Node, library, dedup_map: dict, top_k: int, batch_frontier: bool):
    """One selection -> expansion -> backup round, as a generator: it yields a list of
    GoalRequests whenever it needs the policy, and is resumed with the list of responses."""
    frontier, visited_edges = select_expandable_subtree(root)

    if batch_frontier:
        # Score every frontier leaf in ONE request batch before expanding any of them. Cheaper
        # (one model call for the whole frontier), but NOT guaranteed identical to the sequential
        # default: expanding one leaf can change kernel metavariable state that a later leaf's
        # preparation would have seen.
        preps, seen = [], set()
        for leaf in frontier:
            if not leaf.tactics and id(leaf) not in seen:
                seen.add(id(leaf))
                preps.append(prepare_expansion(kernel, leaf, library))
        if preps:
            responses = yield [p.request for p in preps]
            for prep, resp in zip(preps, responses):
                finish_expansion(kernel, prep, resp, dedup_map, top_k)
    else:
        for leaf in frontier:
            if not leaf.tactics:  # guards against double-expansion if the same node appears twice in one round
                prep = prepare_expansion(kernel, leaf, library)
                (resp,) = yield [prep.request]
                finish_expansion(kernel, prep, resp, dedup_map, top_k)

    for node, action, edge in reversed(visited_edges):
        edge.virtual_count -= 1
        edge.N += 1
        edge.W += edge_value(edge)
        update_status(node)
    update_status(root)


def run_simulation(kernel: "itp_env.Kernel", root: Node, library: "itp_env.Library",
                    policy: PolicyInterface, dedup_map: dict, top_k: int = 16):
    """One full selection -> expansion -> backup round. Never touches
    the caller's own ProofState at all -- see the module docstring."""
    drive(_simulation_steps(kernel, root, library, dedup_map, top_k, False), policy)


def search_steps(kernel: "itp_env.Kernel", root: Node, library: "itp_env.Library", dedup_map: dict,
                 max_simulations: int = 200, top_k: int = 16, dirichlet_epsilon: float = 0.0,
                 dirichlet_alpha: float = 0.3, batch_frontier: bool = False):
    """The search as a resumable generator. It yields a list of GoalRequests whenever it needs a
    policy's scores; send it back the list of (priors, value) responses; it returns the root's
    final Status. Driving many of these together (driver.py) is what lets ONE batched model
    call serve many independent searches at once -- with every individual search producing
    exactly the result it would have produced running alone.

    dirichlet_epsilon > 0 mixes Dirichlet root exploration noise into ONLY the root's own action
    priors (standard AlphaZero practice) -- see finish_expansion. Defaults to 0 (off), so
    existing callers (and every test written before this parameter existed) see byte-identical,
    deterministic behavior unless they explicitly opt in."""
    if not root.tactics:
        prep = prepare_expansion(kernel, root, library)
        (resp,) = yield [prep.request]
        finish_expansion(kernel, prep, resp, dedup_map, top_k, dirichlet_epsilon, dirichlet_alpha)
        update_status(root)
    for _ in range(max_simulations):
        if root.status != Status.UNSOLVED:
            break
        yield from _simulation_steps(kernel, root, library, dedup_map, top_k, batch_frontier)
    return root.status


def search(kernel: "itp_env.Kernel", root: Node, library: "itp_env.Library",
           policy: PolicyInterface, dedup_map: dict, max_simulations: int = 200,
           top_k: int = 16, dirichlet_epsilon: float = 0.0, dirichlet_alpha: float = 0.3,
           batch_frontier: bool = False) -> Status:
    """Runs simulations until root is solved/invalid or the budget runs
    out. Operates entirely on ephemeral kernel state -- call
    commit_solution afterward to produce a real, checkable proof.
    Synchronous convenience over search_steps(): see there for the
    dirichlet_* and batch_frontier parameters."""
    return drive(search_steps(kernel, root, library, dedup_map, max_simulations, top_k,
                              dirichlet_epsilon, dirichlet_alpha, batch_frontier), policy)


# ---------------------------------------------------------------------
# The one place search results ever touch a real ProofState: walks the
# winning (highest-N, all-children-solved) path from a solved node and
# genuinely commits it, in order, into the caller's own ps.
# ---------------------------------------------------------------------

def commit_solution(kernel: "itp_env.Kernel", ps, node: Node, target_meta_id: int,
                     _visiting: frozenset = frozenset()):
    """Requires node.status == Status.SOLVED. target_meta_id must name
    a currently-open goal in ps matching node's own (target, hyp_types)
    -- ordinarily the ProofState's own root, or a goal produced by a
    prior commit_solution call for a sibling.

    _visiting tracks node ids currently being committed along the
    CURRENT recursion path. Among node's own solved edges (highest N
    first), the first one whose own children include NONE already in
    _visiting (including node itself) is committed; an edge that would
    close a cycle is skipped in favor of the next-best one instead.

    This matters, and is not merely defensive: an edge can retroactively
    look "solved" purely because one of its own children IS the parent
    node itself (a genuine self-reference via structural
    deduplication), once that parent became solved via a DIFFERENT,
    legitimate edge first -- update_status only ever guarantees a
    node's FIRST route to SOLVED status is non-circular; a LATER
    update_status call, triggered by something else entirely, can still
    mark an edge "solved" whose own apparent solution depends
    circularly on the parent's own, already-established status.
    Confirmed directly: INDUCT on a hypothesis absent from the target
    (e.g. induct()'s own vestigial-hypothesis convention) can produce a
    Zero-branch child structurally IDENTICAL to its own parent goal
    (nothing about the goal actually changed), which deduplicates to
    the exact same Node -- a genuine self-loop, observed directly as a
    RecursionError before this function had any cycle awareness at all.
    Checking before applying any tactic (not after, by catching the
    recursive call's own cycle error) avoids committing a tactic whose
    effects would then need to be undone."""
    if node.status != Status.SOLVED:
        raise ValueError("commit_solution: node is not SOLVED")
    if not node.tactics:
        raise ValueError("commit_solution: SOLVED node has no recorded tactic -- inconsistent hypergraph")
    next_visiting = _visiting | {id(node)}
    solved_edges = [(a, e) for a, e in node.tactics.items()
                     if all(c.status == Status.SOLVED for c in e.children)]
    solved_edges.sort(key=lambda ae: -ae[1].N)
    chosen = None
    for action, edge in solved_edges:
        if any(id(c) in next_visiting for c in edge.children):
            continue  # this edge would close a cycle; try the next-best one instead
        chosen = (action, edge)
        break
    if chosen is None:
        raise RuntimeError(
            f"commit_solution: every solved edge of node (target={kernel.print_node(node.target)!r}, "
            f"hyp count={len(node.hyp_types)}) would close a cycle back onto an ancestor already "
            "being committed along this path -- no genuinely acyclic route to a proof exists among "
            "this node's own recorded edges."
        )
    action, edge = chosen
    old_index = _goal_index(ps, target_meta_id)
    _apply_tactic(kernel, ps, target_meta_id, action, node.cached_candidates, node.cached_rewrite_extra)
    # Every child lands at old_index, in sequence -- NOT old_index + k.
    # A fully-committed, SOLVED child always closes to zero net open
    # goals (by induction: its own goal is replaced by its own
    # children, each of which recursively closes to zero the same
    # way), so each earlier sibling's own completion vacates old_index
    # before the next sibling is committed there in turn. Using
    # old_index + k here was tried first and confirmed wrong directly:
    # it works for a goal's FIRST child but throws an out-of-range
    # index on the second, since the first child's own completion had
    # already shifted it down to old_index.
    for child in edge.children:
        child_meta_id = ps.goal_meta_id(old_index)
        commit_solution(kernel, ps, child, child_meta_id, next_visiting)


def commit_and_verify(kernel: "itp_env.Kernel", ps, root: Node, root_meta_id: int, goal_type):
    """The recommended entry point for turning a SOLVED search result
    into a genuinely trustworthy proof: commits the winning path (via
    commit_solution) and THEN performs the same independent, from-
    scratch whole_term_check() every existing caller previously had to
    remember to run manually -- commit_solution's own success (an
    empty ps.goals) means the TACTIC LAYER accepted the proof, which
    this project's own history shows is necessary but not sufficient
    (confirmed directly, earlier this session: a tactic-level success
    that an independent check() then correctly rejected). This makes
    that final check the default, not an opt-in every caller has to
    remember -- exactly the "only the final, independent check
    actually matters" discipline this kernel is built around.

    Returns the extracted, VERIFIED proof term. Raises itp_env.ITPError
    if the independent check fails, propagating the kernel's own
    reason rather than silently returning an unverified term."""
    commit_solution(kernel, ps, root, root_meta_id)
    term = ps.extract()
    kernel.whole_term_check(term, goal_type)
    return term


# ---------------------------------------------------------------------
# Layer 2: training data extraction. Called once, right after search()
# returns (whether it converged or exhausted its budget), while
# dedup_map is still in memory -- Layer 1b nodes carry no live kernel
# handle and cannot be revisited afterward.
# ---------------------------------------------------------------------

def extract_training_examples(kernel: "itp_env.Kernel", dedup_map: dict) -> tuple[list, list]:
    """Extracts (policy_examples, value_examples) from a completed
    search's own dedup_map.

    A policy example is emitted for every node with node.tactics
    non-empty AND a nonzero total visit count:
    (observation, target, candidate_tag_scalars) where target is
    {action: edge.N / total_N for each of its own edges} -- the
    search's own visit-count-improved distribution, exactly the
    AlphaZero-style target this design has followed throughout.
    candidate_tag_scalars is node.cached_candidates, resolved to a
    list of (tag, scalar) pairs via ONE kernel.observe_term() call per
    candidate, RIGHT NOW, while this node's own kernel is still alive
    -- not deferred to training time. This matters concretely, not
    just stylistically: a candidate is a NodeId, meaningful only
    relative to the specific kernel instance it was created in, and an
    expert-iteration round's own kernel is discarded once that round's
    search phase ends. Resolving here, to plain (tag, scalar) tuples,
    means the training loop that later scores these examples never
    needs kernel access at all -- it only ever sees portable, already-
    resolved data, regardless of which (possibly since-discarded)
    kernel produced it. Zero-total-visit nodes are excluded
    deliberately: such a node was expanded (discovery ran, edges
    exist) but never revisited by select_expandable_subtree's own PUCT
    walk, so there is no search-informed preference to distill -- only
    a degenerate, uniform-looking target that would be pure noise to
    train against.

    A value example is emitted for every node with node.tactics
    non-empty: (observation, node.cached_value). Restricted to
    expanded nodes, not every leaf, deliberately: a never-expanded
    leaf's own cached_value is still exactly the raw critic estimate
    the model itself just produced for it (see Node.__init__'s own
    default and update_status's own fallback) -- training on it would
    be circular self-distillation with zero new information. An
    expanded node's cached_value, by contrast, reflects real backed-up
    information from its own children (1.0/0.0 for a genuinely
    solved/invalid node, or a real product-rule estimate for a node
    still unsolved at budget exhaustion) -- that is the actual,
    search-derived signal worth distilling into the value head."""
    policy_examples = []
    value_examples = []
    for node in dedup_map.values():
        if not node.tactics:
            continue
        value_examples.append((node.observation, node.cached_value))
        total_n = sum(edge.N for edge in node.tactics.values())
        if total_n == 0:
            continue
        target = {action: edge.N / total_n for action, edge in node.tactics.items()}
        candidate_tag_scalars = []
        for c in node.cached_candidates:
            tags, scalar, _, _ = kernel.observe_term(c)
            candidate_tag_scalars.append((int(tags[0]), int(scalar[0])))
        policy_examples.append((node.observation, target, candidate_tag_scalars))
    return policy_examples, value_examples
