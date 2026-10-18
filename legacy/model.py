"""
model.py -- Layer 2: the term/goal encoder and policy/value model.

Built entirely on top of the existing, unmodified Layer 1b surface
(hypergraph.PolicyInterface) and the Layer 1a observation format
(Kernel.observation()/observe_term()). Nothing in this file touches the
kernel or the search directly.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class TermEncoder(nn.Module):
    """Encodes a kernel DAG dump into a fixed-size embedding per array
    entry, via genuine memoized recursion driven by the actual
    child_a/child_b pointers (see forward's own docstring for why a
    single linear pass, in either direction, over the storage index is
    NOT valid here -- confirmed directly, against real kernel output,
    during this same implementation session).

    Tag values 0 (Var), 4 (Type), and 5 (Const) were confirmed directly
    against the kernel's own output (NOT assumed from the Term
    variant's own declaration order, which does not match tag
    numbering) and get their own scalar featurization. Every other tag
    -- including tag 14, SyntheticCons, which exists only in this dump
    format and has no corresponding kernel Term at all -- is handled
    uniformly: the tag embedding table alone gives each one a
    distinct, learnable representation, and the SAME combine step
    processes child_a/child_b for all of them regardless of what
    structure those children encode. This is deliberate: a Match
    node's own cons-chained branches, a Let's packed value/body, and
    an ordinary App's own two arguments are all just "a tag plus up to
    two children" from this class's own point of view -- no special-
    casing for >2-ary node kinds is needed here, since build_dag_dump
    already linearized them into this same binary shape.

    Const's own scalar (a global id) is mapped to an embedding slot via
    a multiplicative hash into a fixed-size table (const_table_size),
    NOT a pre-registered vocabulary -- an earlier version of this class
    used a fixed dict with an explicit "unknown constant" fallback,
    which meant a lemma declared after model construction (exactly what
    happens every round, via declare_checked, for any curriculum lemma
    beyond a hand-picked headroom) collapsed to the same shared
    "unknown" embedding as every other unregistered constant, making
    them indistinguishable to the model. Hashing needs no advance
    registration at all: ANY global id, including ones that don't exist
    yet when this class is constructed, gets its own deterministic
    table slot immediately. The real, honest cost is a chance of
    collision (two different constants mapping to the same slot) --
    quantified, not silently assumed away: for n constants hashed into
    an m-slot table, P(>=1 collision) = 1 - exp(-n(n-1)/(2m)) (the
    standard birthday-paradox approximation). At the default table size
    (16384) this is ~1.2% for 20 constants and ~7.2% for 50 -- a small,
    known, bounded risk at the scale this project's own curricula
    actually reach, not zero, and worth re-checking against this same
    formula if the constant count grows much further. The hash itself
    uses Knuth's multiplicative constant (2654435761, chosen for its
    good bit-dispersion properties on sequential or clustered inputs --
    plain modulo would happen to be collision-free for THIS kernel's
    own sequential global-id assignment specifically, but would not
    generalize to any other id-assignment scheme, which is precisely
    why a real dispersing hash is used instead of relying on that
    coincidence).
    """

    TAG_VAR = 0
    TAG_TYPE = 4
    TAG_CONST = 5
    N_TAGS = 15  # 14 real kernel Term kinds + 1 SyntheticCons (dump-format-only)
    _HASH_MULTIPLIER = 2654435761  # Knuth's multiplicative hash constant

    def __init__(self, embed_dim: int = 64, const_table_size: int = 16384,
                 max_var_level: int = 63, max_type_level: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.tag_embed = nn.Embedding(self.N_TAGS, embed_dim)

        self.max_var_level = max_var_level
        self.var_level_embed = nn.Embedding(max_var_level + 2, embed_dim)  # +1 overflow bucket, +1 headroom

        self.max_type_level = max_type_level
        self.type_level_embed = nn.Embedding(max_type_level + 2, embed_dim)

        self.const_table_size = const_table_size
        self.const_embed = nn.Embedding(const_table_size, embed_dim)

        # Used in place of a real child embedding when child_a/child_b
        # is -1 (no such field -- e.g. Var/Const/Type are leaves, or
        # QuotType has no child_b).
        self.no_child = nn.Parameter(torch.zeros(embed_dim))

        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # When True, forward() runs the original per-node memoized-recursion implementation
        # (forward_reference) instead of the vectorized one. Exists so tests can compute an
        # INDEPENDENT oracle from the very same weights; never needed in normal use.
        self.use_reference = False

    def _hash_const(self, global_id: int) -> int:
        """Deterministic, dispersing hash of a global id into
        [0, const_table_size) -- see this class's own docstring for the
        collision-probability analysis this table size is chosen from."""
        return ((global_id * self._HASH_MULTIPLIER) & 0xFFFFFFFF) % self.const_table_size

    def _scalar_feature(self, tag: int, scalar: int) -> torch.Tensor:
        if tag == self.TAG_VAR:
            level = min(scalar, self.max_var_level + 1)
            return self.var_level_embed(torch.tensor(level))
        if tag == self.TAG_TYPE:
            level = min(scalar, self.max_type_level + 1)
            return self.type_level_embed(torch.tensor(level))
        if tag == self.TAG_CONST:
            return self.const_embed(torch.tensor(self._hash_const(scalar)))
        return torch.zeros(self.embed_dim)

    def forward_reference(self, tags, scalar, child_a, child_b) -> torch.Tensor:
        """THE ORIGINAL, SLOW, OBVIOUSLY-CORRECT implementation -- kept verbatim as the oracle the
        vectorized forward() is tested against. tags/scalar/child_a/child_b: the four parallel arrays from
        observation() or observe_term() (numpy arrays or plain
        sequences; any indexable, len()-able type works). Returns a
        (N, embed_dim) tensor of per-entry embeddings, in the SAME
        index order as the input arrays -- so embeds[i] is entry i's
        own embedding, directly usable with any root index from
        observation()'s own `roots` array.

        Uses genuine memoized recursion driven by the actual child_a/
        child_b pointers, NOT a single linear pass over the storage
        index in either direction. An earlier version of this method
        assumed storage index order was a valid topological order
        (children always at strictly larger indices than their own
        parent) -- confirmed directly, against a real kernel dump, to
        be false: a shared subterm (e.g. a hypothesis's own type,
        first dumped as an early child of one parent) can be RE-
        referenced later by a DEEPER, separately-recursed parent whose
        own index is LARGER than the shared child's -- "first-visit-
        wins" storage order is neither a forward nor a backward
        topological sort once cross-parent sharing is involved, so
        only genuine dependency-driven recursion (not a fixed
        direction) is correct here.

        Deliberately a plain recursive implementation, not vectorized
        across nodes within one term, and not converted to an
        explicit-stack iterative form to avoid Python's own recursion
        limit -- correct first; both are known, flagged, not-yet-
        measured/needed costs for this curriculum's own term sizes,
        matching this project's own established discipline of
        deferring optimization until it is actually shown to matter."""
        n = len(tags)
        embeds: list[torch.Tensor | None] = [None] * n

        def compute(i: int) -> torch.Tensor:
            if embeds[i] is not None:
                return embeds[i]  # type: ignore[return-value]
            tag = int(tags[i])
            t_embed = self.tag_embed(torch.tensor(tag))
            s_embed = self._scalar_feature(tag, int(scalar[i]))
            ca = int(child_a[i])
            cb = int(child_b[i])
            a_embed = compute(ca) if ca >= 0 else self.no_child
            b_embed = compute(cb) if cb >= 0 else self.no_child
            result = self.combine(torch.cat([t_embed + s_embed, a_embed, b_embed]))
            embeds[i] = result
            return result

        for i in range(n):
            compute(i)
        return torch.stack(embeds)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Vectorized implementation
    # ------------------------------------------------------------------

    def node_features(self, tags: np.ndarray, scalar: np.ndarray) -> torch.Tensor:
        """(N, embed_dim): tag embedding + scalar featurization, for N nodes at once. Row i equals
        exactly `tag_embed(tags[i]) + _scalar_feature(tags[i], scalar[i])` of the reference code.
        All three scalar tables are looked up for every row and masked (multiplying by exactly 0.0
        or 1.0), which is exact and avoids boolean-indexed scatter."""
        tags = np.asarray(tags, dtype=np.int64)
        scalar = np.asarray(scalar, dtype=np.int64)
        var_idx = np.minimum(scalar, self.max_var_level + 1)
        type_idx = np.minimum(scalar, self.max_type_level + 1)
        const_idx = ((scalar * self._HASH_MULTIPLIER) & 0xFFFFFFFF) % self.const_table_size
        m_var = torch.from_numpy((tags == self.TAG_VAR).astype(np.float32)).unsqueeze(1)
        m_type = torch.from_numpy((tags == self.TAG_TYPE).astype(np.float32)).unsqueeze(1)
        m_const = torch.from_numpy((tags == self.TAG_CONST).astype(np.float32)).unsqueeze(1)
        s = (self.var_level_embed(torch.from_numpy(var_idx)) * m_var
             + self.type_level_embed(torch.from_numpy(type_idx)) * m_type
             + self.const_embed(torch.from_numpy(const_idx)) * m_const)
        return self.tag_embed(torch.from_numpy(tags)) + s

    def encode_nodes(self, tags, scalar, child_a, child_b, heights) -> torch.Tensor:
        """Embeddings for every node of one or MANY concatenated DAGs (children already offset into
        the same global numbering), processed LEVEL BY LEVEL: all height-0 nodes in one batched
        step, then all height-1 nodes, and so on. Every node's children have strictly lower height,
        so each level only reads rows already finalized -- the number of sequential steps is
        (max height + 1), independent of how many nodes or goals there are. Returns (N, embed_dim)
        in the input node order, matching forward_reference row for row."""
        tags = np.asarray(tags, dtype=np.int64)
        N = len(tags)
        D = self.embed_dim
        if N == 0:
            return torch.zeros(0, D)
        heights = np.asarray(heights, dtype=np.int64)
        ca = np.asarray(child_a, dtype=np.int64)
        cb = np.asarray(child_b, dtype=np.int64)
        order = np.argsort(heights, kind="stable")                       # nodes sorted by height
        bounds = np.concatenate(([0], np.cumsum(np.bincount(heights))))  # level l = order[bounds[l]:bounds[l+1]]
        pos = np.empty(N, dtype=np.int64)
        pos[order] = np.arange(N)
        # Child pointers in SORTED coordinates; a missing child (-1) points at the extra row N,
        # which holds the learned `no_child` embedding.
        ca_s = np.where(ca >= 0, pos[np.where(ca >= 0, ca, 0)], N)[order]
        cb_s = np.where(cb >= 0, pos[np.where(cb >= 0, cb, 0)], N)[order]
        base = self.node_features(tags[order], np.asarray(scalar, dtype=np.int64)[order])
        ca_t, cb_t = torch.from_numpy(ca_s), torch.from_numpy(cb_s)
        rows = torch.arange(N)
        buf = torch.cat([torch.zeros(N, D), self.no_child.unsqueeze(0)], dim=0)   # (N+1, D)
        for lvl in range(len(bounds) - 1):
            b0, b1 = int(bounds[lvl]), int(bounds[lvl + 1])
            x = torch.cat([base[b0:b1], buf.index_select(0, ca_t[b0:b1]), buf.index_select(0, cb_t[b0:b1])], dim=1)
            buf = buf.index_copy(0, rows[b0:b1], self.combine(x))   # out-of-place: autograd-safe
        return buf.index_select(0, torch.from_numpy(pos))

    def forward(self, tags, scalar, child_a, child_b, heights=None) -> torch.Tensor:
        """Per-entry embeddings for ONE dump (numpy arrays or sequences), (N, embed_dim), in input
        order -- same contract as always. `heights` (from the kernel's observation) is computed if
        omitted. Uses the vectorized level-wise implementation unless `use_reference` is set."""
        if self.use_reference:
            return self.forward_reference(tags, scalar, child_a, child_b)
        if heights is None:
            from policy_api import compute_heights
            heights = compute_heights(child_a, child_b)
        return self.encode_nodes(tags, scalar, child_a, child_b, heights)


class PolicyValueModel(nn.Module):
    """Wraps TermEncoder with action scoring and a value head. This is
    the reference, first-instantiation model behind TrainablePolicy --
    NOT the only possible one; anyone implementing hypergraph.
    PolicyInterface can swap in something else entirely, per this
    whole design's own policy-agnostic goal (see hypergraph.py's own
    PolicyInterface docstring).

    An APPLY candidate that is a library constant is encoded via its
    own bare reference alone (tag Const, scalar its own global id,
    hashed through TermEncoder's own const_embed table) -- not via
    encoding its own declared type's full structure through a second
    observe_term() call, which was the original plan. This is simpler
    and avoids needing any kernel access from within this class at all.
    An earlier version of this hashing scheme used a small, fixed,
    pre-registered vocabulary instead, with an explicit "unknown
    constant" fallback -- meaning a lemma declared mid-curriculum via
    declare_checked (exactly what the "solved theorems become new
    library lemmas" loop does every round) collapsed to the same
    shared embedding as every other unregistered constant, genuinely
    indistinguishable to the model. TermEncoder's own hashing (see its
    docstring) removes that limitation: any global id gets its own
    deterministic slot immediately, with no advance registration
    needed, at the cost of a small, quantified collision probability
    rather than an unbounded "everything unknown looks identical"
    failure mode.
    """

    def __init__(self, embed_dim: int = 64, const_table_size: int = 16384,
                 max_var_level: int = 63, max_type_level: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.term_encoder = TermEncoder(embed_dim=embed_dim, const_table_size=const_table_size,
                                         max_var_level=max_var_level, max_type_level=max_type_level)
        # INTRO / CASES / INDUCT each get their own fixed, learned type
        # embedding; APPLY has no fixed embedding of its own at all --
        # its own identity comes entirely from the candidate term being
        # applied (encoded via self.term_encoder directly, same as
        # everything else), which is why this table has only 3 rows,
        # not 4.
        self.ACTION_INTRO, self.ACTION_CASES, self.ACTION_INDUCT = 0, 1, 2
        self.action_type_embed = nn.Embedding(3, embed_dim)

        self.score_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def encode_goal(self, observation) -> tuple[torch.Tensor, torch.Tensor, list]:
        """observation: the 5-tuple from Kernel.observation() (tags,
        scalar, child_a, child_b, roots). Returns (goal_embed,
        node_embeds, roots): goal_embed is the fixed-size embedding for
        the WHOLE goal (target pooled with its own hypotheses);
        node_embeds/roots are returned too since action_embed's own
        CASES/INDUCT handling needs to look up a specific hypothesis's
        own embedding by its own root index."""
        tags, scalar, child_a, child_b, roots, *rest = observation
        node_embeds = self.term_encoder(tags, scalar, child_a, child_b, rest[0] if rest else None)
        roots = [int(r) for r in roots]
        target_embed = node_embeds[roots[-1]]
        hyp_roots = roots[:-1]
        if hyp_roots:
            hyp_pool = torch.stack([node_embeds[r] for r in hyp_roots]).mean(dim=0)
            goal_embed = target_embed + hyp_pool
        else:
            goal_embed = target_embed
        return goal_embed, node_embeds, roots

    def action_embed(self, action, node_embeds: torch.Tensor, roots: list,
                      candidate_tag_scalar: tuple[int, int] | None) -> torch.Tensor:
        """Returns an embed_dim embedding for ONE action.
        candidate_tag_scalar is (tag, scalar) for the candidate term an
        APPLY action refers to -- required (and only meaningful) for
        ActionKind.APPLY; ignored otherwise. Callers (TrainablePolicy)
        are responsible for resolving a candidate NodeId to its own
        (tag, scalar) pair, since that resolution needs kernel access
        this class deliberately does not have (see this class's own
        docstring)."""
        from hypergraph import ActionKind  # local import: model.py must not require hypergraph.py to import (keeps this module usable standalone), only the reverse

        if action.kind == ActionKind.INTRO:
            return self.action_type_embed(torch.tensor(self.ACTION_INTRO))
        if action.kind == ActionKind.CASES:
            level_embed = self.term_encoder.var_level_embed(torch.tensor(min(action.arg, self.term_encoder.max_var_level + 1)))
            return self.action_type_embed(torch.tensor(self.ACTION_CASES)) + node_embeds[roots[action.arg]] + level_embed
        if action.kind == ActionKind.INDUCT:
            level_embed = self.term_encoder.var_level_embed(torch.tensor(min(action.arg, self.term_encoder.max_var_level + 1)))
            return self.action_type_embed(torch.tensor(self.ACTION_INDUCT)) + node_embeds[roots[action.arg]] + level_embed
        if action.kind == ActionKind.APPLY:
            assert candidate_tag_scalar is not None, "APPLY action requires candidate_tag_scalar"
            tag, scalar = candidate_tag_scalar
            t_embed = self.term_encoder.tag_embed(torch.tensor(tag))
            s_embed = self.term_encoder._scalar_feature(tag, scalar)
            return t_embed + s_embed
        raise ValueError(f"action_embed: unrecognized action kind {action.kind}")

    # ------------------------------------------------------------------
    # Batched implementation: many goals, one pass (see policy_api.Batch)
    # ------------------------------------------------------------------

    def encode_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """(goal_embed (G, D), node_embeds (N, D)) for every goal of a policy_api.Batch. Matches
        encode_goal per goal: goal_embed = target embedding + mean of hypothesis embeddings (just
        the target embedding for a goal with no hypotheses)."""
        E = self.term_encoder.encode_nodes(batch.tags, batch.scalar, batch.child_a, batch.child_b, batch.heights)
        G = batch.n_goals
        target = E.index_select(0, torch.from_numpy(batch.target_root))
        if len(batch.hyp_roots):
            sums = torch.zeros(G, self.embed_dim).index_add(0, torch.from_numpy(batch.hyp_goal),
                                                             E.index_select(0, torch.from_numpy(batch.hyp_roots)))
            counts = torch.from_numpy(np.maximum(batch.n_hyps, 1).astype(np.float32)).unsqueeze(1)
            goal_embed = target + sums / counts     # goals with no hypotheses add an exact 0
        else:
            goal_embed = target
        return goal_embed, E

    def action_embed_batch(self, node_embeds: torch.Tensor, ab) -> torch.Tensor:
        """(A, D) embeddings for a policy_api.ActionBatch, row for row equal to action_embed():
        INTRO/CASES/INDUCT = type embedding (+ hypothesis embedding + level embedding for the two
        that target a hypothesis); APPLY = its candidate's tag + scalar features."""
        te = self.term_encoder
        kinds = torch.from_numpy(ab.kinds)
        f = lambda m: torch.from_numpy(m.astype(np.float32)).unsqueeze(1)
        is_hyp = f((ab.kinds == 1) | (ab.kinds == 2))
        is_type = f(ab.kinds < 3)
        is_apply = f(ab.kinds == 3)
        type_part = self.action_type_embed(torch.clamp(kinds, max=2)) * is_type
        hyp_part = node_embeds.index_select(0, torch.from_numpy(ab.hyp_node)) * is_hyp
        lvl_part = te.var_level_embed(torch.from_numpy(np.minimum(ab.args, te.max_var_level + 1))) * is_hyp
        apply_part = te.node_features(ab.cand_tag, ab.cand_scalar) * is_apply
        return type_part + hyp_part + lvl_part + apply_part

    def score_actions(self, goal_embed: torch.Tensor, node_embeds: torch.Tensor, ab) -> torch.Tensor:
        """(A,) raw scores of every action against its goal."""
        a = self.action_embed_batch(node_embeds, ab)
        g = goal_embed.index_select(0, torch.from_numpy(ab.goal))
        return self.score_head(torch.cat([g, a], dim=1)).squeeze(-1)

    def forward_batch(self, batch, ab) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a whole policy_api.Batch/ActionBatch pair in one pass. Returns (priors (A,), values
        (G,)): priors are per-goal softmax probabilities over that goal's actions, laid out like
        `ab` (contiguous per goal, `ab.offsets`), values sigmoid critic estimates. Row for row the
        same numbers forward() produces one goal at a time (to float rounding)."""
        goal_embed, node_embeds = self.encode_batch(batch)
        scores = self.score_actions(goal_embed, node_embeds, ab)
        priors = segment_log_softmax(scores, torch.from_numpy(ab.segment), ab.n_segments).exp()
        values = torch.sigmoid(self.value_head(goal_embed)).squeeze(-1)
        return priors, values

    def forward(self, observation, actions: list, candidate_tag_scalars: list) -> tuple[dict, float]:
        """The full predict() computation: encode the goal, encode and
        score every action, and produce a value estimate. actions and
        candidate_tag_scalars are as in PolicyInterface.predict (the
        latter pre-resolved to (tag, scalar) pairs by the caller, one
        per entry in `candidates`, indexed the same way action.arg
        indexes into `candidates` for an APPLY action).

        priors are genuine, softmax-normalized probabilities over
        `actions` (summing to 1), NOT raw score_head logits. This
        matters for correctness, not just presentation: puct_score's
        own formula multiplies prior directly into the exploration
        bonus, expecting a bounded [0, 1] probability the way
        UniformPolicy already, correctly provides one -- confirmed
        directly that returning raw, unbounded logits instead breaks
        this badly: a large positive logit (e.g. ~12) inflates the
        exploration bonus far beyond anything a real probability could
        produce, while a negative logit actively subtracts from the
        PUCT score, permanently locking that action out of ever being
        selected regardless of simulation budget -- observed directly
        as unbounded, runaway re-induction with APPLY (a legitimately
        necessary action) never once tried (N=0) after hundreds of
        simulations."""
        from hypergraph import ActionKind

        goal_embed, node_embeds, roots = self.encode_goal(observation)
        scores = []
        for action in actions:
            cts = candidate_tag_scalars[action.arg] if action.kind == ActionKind.APPLY else None
            a_embed = self.action_embed(action, node_embeds, roots, cts)
            scores.append(self.score_head(torch.cat([goal_embed, a_embed])))
        scores_t = torch.stack(scores).squeeze(-1)
        probs = torch.softmax(scores_t, dim=0)
        priors = {action: probs[i] for i, action in enumerate(actions)}
        value = torch.sigmoid(self.value_head(goal_embed))
        return priors, value


def segment_log_softmax(scores: torch.Tensor, segment: torch.Tensor, n_segments: int) -> torch.Tensor:
    """log-softmax of `scores` within each segment (segment[i] = the group i competes in), for
    ragged groups with no padding. The per-group max used for stability is detached: it is a
    constant that cancels exactly, so gradients are unaffected."""
    neg_inf = torch.full((n_segments,), float("-inf"))
    m = neg_inf.scatter_reduce(0, segment, scores.detach(), reduce="amax", include_self=True)
    shifted = scores - m.index_select(0, segment)
    z = torch.zeros(n_segments).index_add(0, segment, shifted.exp())
    return shifted - z.log().index_select(0, segment)
