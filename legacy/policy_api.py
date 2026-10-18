"""policy_api.py -- Layer 3: the policy-agnostic contract.

Everything a policy author needs to know lives in this file, and it depends on
NOTHING but numpy: no kernel handle, no torch, no search internals.

A policy is any object with

    predict_batch(requests: Sequence[GoalRequest]) -> Sequence[(priors, value)]

* `requests` -- one GoalRequest per goal the search needs scored right now.
  When many searches run together (see driver.py) a single call can carry
  requests from all of them, so a model can score them in one batched pass.
* `priors`   -- either a dict {action: score} or a sequence/array aligned with
  `request.actions` (array-native policies and out-of-process policies should
  prefer this: no per-action dict to build). Scores need not be normalized;
  the search only ever RANKS them and uses them as PUCT priors, so they should
  be non-negative, and bounded in [0, 1] if you want the usual PUCT scaling.
  A policy need not score every action: unscored actions count as 0.
* `value`    -- a scalar critic estimate in [0, 1] (0.5 = "no opinion").

The older one-goal-at-a-time `predict(observation, actions, candidates)` method
still works: hypergraph.PolicyInterface.predict_batch defaults to looping it.

The observation format (`GoalRequest.observation`) is versioned; see
OBSERVATION_SCHEMA below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Observation schema (version 2)
# ---------------------------------------------------------------------------
# `observation` is a 6-tuple of 1-D numpy arrays describing ONE goal as a DAG:
#     (tags, scalar, child_a, child_b, roots, heights)
# * tags[i]    -- node kind, an index into TAG_NAMES.
# * scalar[i]  -- Var: de Bruijn LEVEL; Const: global id; otherwise 0.
# * child_a[i], child_b[i] -- indices of the (up to two) children, -1 if absent.
#   Nodes with more than two fields are packed into a right-folded chain of
#   SYNTHETIC_CONS nodes, so the DAG is strictly binary.
# * roots      -- indices of the goal's hypothesis types (in order) followed by
#   its target as the LAST entry.
# * heights[i] -- 0 for a leaf, else 1 + max(heights of its children). Every node's
#   children are strictly lower, so encoding level by level (all height-0 nodes,
#   then all height-1 nodes, ...) is valid and needs only (max height + 1)
#   sequential steps however many nodes there are. (Added in v2; a v1 5-tuple
#   without it is still accepted by collate_observations, which computes it.)
# Index order is NOT topological (shared subterms are re-referenced by later
# parents): follow the child pointers, or use the heights.
OBSERVATION_SCHEMA_VERSION = 2
TAG_NAMES = ("VAR", "APP", "LAMBDA", "PI", "TYPE", "CONST", "MATCH", "META", "TYPE_UNIV",
             "QUOT_TYPE", "QUOT_MK", "QUOT_LIFT", "LET", "FIX", "SYNTHETIC_CONS")
TAG = {name: i for i, name in enumerate(TAG_NAMES)}

# Action kinds, as ints (Action.kind is an IntEnum with these values).
ACTION_INTRO, ACTION_CASES, ACTION_INDUCT, ACTION_APPLY = 0, 1, 2, 3
ACTION_KIND_NAMES = ("INTRO", "CASES", "INDUCT", "APPLY")


@dataclass(frozen=True, eq=False)
class GoalRequest:
    """Everything a policy needs to score ONE goal. Self-contained: scoring never
    requires a kernel handle.

    observation      -- the goal's DAG (see OBSERVATION_SCHEMA above).
    actions          -- the legal action menu, as opaque Action objects with integer
                        `.kind` (ACTION_*) and `.arg` (INTRO: unused; CASES/INDUCT: a
                        hypothesis index, an index into observation roots; APPLY: an
                        index into `candidates`). Priors you return are matched to
                        these actions, by position or by identity.
    candidate_tags / candidate_scalars
                     -- for each APPLY candidate (an atom: a library constant or a
                        hypothesis variable), its node kind (TAG_NAMES index) and
                        scalar (const global id or var level). Aligned with `candidates`;
                        this is all an embedding of an APPLY action needs.
    candidates       -- the raw candidate handles (kernel NodeIds): opaque; do not
                        rely on them. Kept so legacy predict() policies keep working.
    """
    observation: tuple
    actions: Sequence[Any]
    candidates: Sequence[Any]
    candidate_tags: np.ndarray
    candidate_scalars: np.ndarray
    _render: Optional[Callable[[], str]] = None
    # Optional int32 arrays aligned with `actions` (the framework fills them in once, when it builds
    # the request); collation falls back to computing them from `actions` if absent.
    action_kinds: Optional[np.ndarray] = None
    action_args: Optional[np.ndarray] = None

    def kinds_and_args(self) -> tuple:
        if self.action_kinds is not None and self.action_args is not None:
            return self.action_kinds, self.action_args
        n = len(self.actions)
        return (np.fromiter((int(a.kind) for a in self.actions), dtype=np.int32, count=n),
                np.fromiter((int(a.arg) for a in self.actions), dtype=np.int32, count=n))

    def text(self) -> str:
        """Human/LLM-readable rendering of the goal ("h1 : T1 ... |- target"), computed
        lazily (never on the hot path). Empty string if the request was built without one."""
        return self._render() if self._render is not None else ""

    def to_portable(self) -> dict:
        """A plain, picklable form (numpy arrays and ints only) for sending a request to another
        process or machine. Inverse of nothing: responses come back as aligned arrays (see module
        docstring), so no Action objects need to cross the boundary."""
        tags, scalar, child_a, child_b, roots, *rest = self.observation
        kinds, args = self.kinds_and_args()
        heights = rest[0] if rest else compute_heights(child_a, child_b)
        return dict(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            tags=np.asarray(tags), scalar=np.asarray(scalar),
            child_a=np.asarray(child_a), child_b=np.asarray(child_b), roots=np.asarray(roots),
            heights=np.asarray(heights),
            action_kinds=np.asarray(kinds), action_args=np.asarray(args),
            candidate_tags=np.asarray(self.candidate_tags), candidate_scalars=np.asarray(self.candidate_scalars),
        )


class BatchPolicy(Protocol):
    """The whole Layer 3 contract."""
    def predict_batch(self, requests: Sequence[GoalRequest]) -> Sequence[tuple[Any, float]]: ...


# ---------------------------------------------------------------------------
# Collation: many goals -> one set of concatenated arrays (framework-agnostic numpy)
# ---------------------------------------------------------------------------

def compute_heights(child_a, child_b) -> np.ndarray:
    """Heights for a dump that doesn't carry them (schema v1, or a hand-built DAG): leaf 0, else
    1 + max child height, by relaxation. O(height * n); the kernel computes them for free at dump
    time, so this is only a fallback."""
    ca = np.asarray(child_a, dtype=np.int64); cb = np.asarray(child_b, dtype=np.int64)
    n = len(ca)
    h = np.zeros(n, dtype=np.int64)
    ha = ca >= 0; hb = cb >= 0
    for _ in range(n + 1):
        new = np.maximum(np.where(ha, h[np.where(ha, ca, 0)] + 1, 0), np.where(hb, h[np.where(hb, cb, 0)] + 1, 0))
        if np.array_equal(new, h):
            return h.astype(np.uint32)
        h = new
    raise ValueError("compute_heights: the child pointers contain a cycle")


@dataclass(frozen=True, eq=False)
class Batch:
    """G goals' DAGs concatenated. Node arrays are length N (all goals' nodes end to end); every
    child index is already offset into this global numbering (-1 preserved).

    dag_offsets[g]:dag_offsets[g+1]  -- goal g's nodes.
    roots / root_offsets             -- global node indices of each goal's hypothesis types then target.
    target_root[g]                   -- global index of goal g's target.
    hyp_roots / hyp_goal             -- every hypothesis root, flat, and the goal that owns it.
    n_hyps[g]                        -- number of hypotheses of goal g."""
    n_goals: int
    tags: np.ndarray
    scalar: np.ndarray
    child_a: np.ndarray
    child_b: np.ndarray
    heights: np.ndarray
    dag_offsets: np.ndarray
    roots: np.ndarray
    root_offsets: np.ndarray
    target_root: np.ndarray
    hyp_roots: np.ndarray
    hyp_goal: np.ndarray
    n_hyps: np.ndarray


@dataclass(frozen=True, eq=False)
class ActionBatch:
    """A ragged set of A actions over a Batch. `goal[a]` is the goal whose embedding action a is
    scored against; `segment[a]` groups actions that compete in one softmax (== goal when scoring
    requests; the example index when training, where several examples may share a goal).
    hyp_node[a]  -- for CASES/INDUCT, the global node index of the hypothesis it targets (else 0).
    cand_tag/cand_scalar[a] -- for APPLY, the candidate's (tag, scalar) (else 0).
    offsets      -- present when actions are grouped contiguously per segment: segment s owns
                    [offsets[s], offsets[s+1])."""
    n_actions: int
    kinds: np.ndarray
    args: np.ndarray
    goal: np.ndarray
    segment: np.ndarray
    hyp_node: np.ndarray
    cand_tag: np.ndarray
    cand_scalar: np.ndarray
    n_segments: int
    offsets: Optional[np.ndarray] = None


def collate_observations(observations: Sequence[tuple]) -> Batch:
    G = len(observations)
    sizes = np.fromiter((len(o[0]) for o in observations), dtype=np.int64, count=G)
    dag_offsets = np.zeros(G + 1, dtype=np.int64)
    np.cumsum(sizes, out=dag_offsets[1:])
    node_off = np.repeat(dag_offsets[:-1], sizes)
    tags = np.concatenate([np.asarray(o[0]) for o in observations]).astype(np.int64)
    scalar = np.concatenate([np.asarray(o[1]) for o in observations]).astype(np.int64)
    ca = np.concatenate([np.asarray(o[2]) for o in observations]).astype(np.int64)
    cb = np.concatenate([np.asarray(o[3]) for o in observations]).astype(np.int64)
    ca = np.where(ca >= 0, ca + node_off, -1)
    cb = np.where(cb >= 0, cb + node_off, -1)
    heights = np.concatenate([np.asarray(o[5]) if len(o) > 5 else compute_heights(o[2], o[3])
                              for o in observations]).astype(np.int64)
    root_lists = [np.asarray(o[4]).astype(np.int64) for o in observations]
    root_counts = np.fromiter((len(r) for r in root_lists), dtype=np.int64, count=G)
    root_offsets = np.zeros(G + 1, dtype=np.int64)
    np.cumsum(root_counts, out=root_offsets[1:])
    roots = np.concatenate(root_lists) + np.repeat(dag_offsets[:-1], root_counts)
    target_root = roots[root_offsets[1:] - 1]
    n_hyps = root_counts - 1
    is_target = np.zeros(len(roots), dtype=bool)
    is_target[root_offsets[1:] - 1] = True
    hyp_roots = roots[~is_target]
    hyp_goal = np.repeat(np.arange(G, dtype=np.int64), n_hyps)
    return Batch(G, tags, scalar, ca, cb, heights, dag_offsets, roots, root_offsets, target_root,
                 hyp_roots, hyp_goal, n_hyps)


def collate_actions(batch: Batch, kinds: Sequence[np.ndarray], args: Sequence[np.ndarray],
                    cand_tags: Sequence[np.ndarray], cand_scalars: Sequence[np.ndarray],
                    goals: Sequence[int], segments: Optional[Sequence[int]] = None) -> ActionBatch:
    """One entry per example (a goal's action list): kinds/args are that example's int arrays,
    cand_tags/cand_scalars its candidate features, goals[j] the index into `batch` it is scored
    against, segments[j] (default j) its softmax group."""
    n_ex = len(kinds)
    lens = np.fromiter((len(k) for k in kinds), dtype=np.int64, count=n_ex)
    A = int(lens.sum())
    offsets = np.zeros(n_ex + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    k_all = np.concatenate(kinds).astype(np.int64) if A else np.zeros(0, dtype=np.int64)
    a_all = np.concatenate(args).astype(np.int64) if A else np.zeros(0, dtype=np.int64)
    goals_a = np.asarray(goals, dtype=np.int64)
    seg_ex = np.arange(n_ex, dtype=np.int64) if segments is None else np.asarray(segments, dtype=np.int64)
    goal = np.repeat(goals_a, lens)
    segment = np.repeat(seg_ex, lens)
    # hypothesis actions: the root index of the hypothesis they target
    is_hyp = (k_all == ACTION_CASES) | (k_all == ACTION_INDUCT)
    root_base = batch.root_offsets[goal]
    hyp_idx = np.where(is_hyp, root_base + np.minimum(np.maximum(a_all, 0), batch.n_hyps[goal]), 0)
    hyp_node = np.where(is_hyp, batch.roots[hyp_idx] if len(batch.roots) else 0, 0)
    # APPLY actions: candidate features, gathered from all examples' candidate arrays end to end
    is_apply = k_all == ACTION_APPLY
    cand_lens = np.fromiter((len(c) for c in cand_tags), dtype=np.int64, count=n_ex)
    cand_off = np.zeros(n_ex + 1, dtype=np.int64)
    np.cumsum(cand_lens, out=cand_off[1:])
    ct_all = np.concatenate([np.asarray(c) for c in cand_tags]).astype(np.int64) if cand_off[-1] else np.zeros(1, dtype=np.int64)
    cs_all = np.concatenate([np.asarray(c) for c in cand_scalars]).astype(np.int64) if cand_off[-1] else np.zeros(1, dtype=np.int64)
    g_idx = np.where(is_apply, np.repeat(cand_off[:-1], lens) + a_all, 0)
    cand_tag = np.where(is_apply, ct_all[g_idx], 0)
    cand_scalar = np.where(is_apply, cs_all[g_idx], 0)
    return ActionBatch(A, k_all, a_all, goal, segment, hyp_node, cand_tag, cand_scalar, n_segments=int(seg_ex.max()) + 1 if n_ex else 0,
                       offsets=offsets)


def collate_requests(requests: Sequence[GoalRequest]) -> tuple:
    """Requests -> (Batch, ActionBatch): every request's goal is a goal of the batch, its actions
    are scored against it, and softmax groups are per request."""
    batch = collate_observations([r.observation for r in requests])
    ka = [r.kinds_and_args() for r in requests]
    ab = collate_actions(batch, [k for k, _ in ka], [a for _, a in ka],
                         [r.candidate_tags for r in requests], [r.candidate_scalars for r in requests],
                         goals=range(len(requests)))
    return batch, ab


def collate_portable(portables: Sequence[dict]) -> tuple:
    """The process-boundary analog of collate_requests: takes GoalRequest.to_portable() dicts (numpy
    arrays and ints only -- exactly what survives a pickle across a multiprocessing Queue or a
    socket) instead of GoalRequest objects, and produces the identical (Batch, ActionBatch).
    collate_observations/collate_actions already take plain arrays, not GoalRequest/Action objects,
    so this needs no new collation logic -- just unpacking the dict fields into the same calls
    collate_requests makes."""
    batch = collate_observations([(p["tags"], p["scalar"], p["child_a"], p["child_b"], p["roots"], p["heights"])
                                  for p in portables])
    ab = collate_actions(batch, [p["action_kinds"] for p in portables], [p["action_args"] for p in portables],
                         [p["candidate_tags"] for p in portables], [p["candidate_scalars"] for p in portables],
                         goals=range(len(portables)))
    return batch, ab
