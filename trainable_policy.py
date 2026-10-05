"""
trainable_policy.py -- the PolicyInterface implementation wrapping
PolicyValueModel. Imports both hypergraph.py and model.py; neither of
those imports this file or each other at module-load time, keeping the
dependency direction one-way and model.py usable standalone.
"""

from __future__ import annotations
import torch
import itp_env
import policy_api as pa
from hypergraph import PolicyInterface, Action, ActionKind
from model import PolicyValueModel, TermEncoder


class TrainablePolicy(PolicyInterface):
    """Wraps a PolicyValueModel to implement hypergraph.PolicyInterface.

    Holds its own itp_env.Kernel reference, used ONLY to resolve an
    APPLY action's own candidate NodeId to the (tag, scalar) pair
    PolicyValueModel.action_embed needs -- a bare NodeId exposes only
    (tag, index), where index is an opaque pool position, not the
    semantic level/global-id a candidate's own identity actually is;
    recovering that requires a kernel call. This is a LOCAL,
    implementation-specific choice by this one policy, not something
    the PolicyInterface contract itself requires -- UniformPolicy
    needs no kernel access at all, and nothing about hypergraph.py's
    own search mechanics is aware this policy happens to hold one.

    predict() runs the model under torch.no_grad() and returns plain
    Python floats (via .item()), not tensors -- inference/search time
    never needs gradients, and expand()'s own sorted()/max() calls
    over the returned priors dict should not be silently carrying a
    live autograd graph around for the rest of the search.
    """

    def __init__(self, model: PolicyValueModel, kernel: "itp_env.Kernel | None" = None):
        # `kernel` is needed ONLY by the legacy one-goal predict() below, which must resolve APPLY
        # candidates itself. predict_batch() (what the search actually calls) works purely from the
        # GoalRequest, which already carries those features, so it needs no kernel at all.
        self.model = model
        self.kernel = kernel

    def predict_batch(self, requests) -> list:
        """Layer 3 entry point. Scores every request in ONE model.forward_batch call: one level-wise
        encoder pass over all requests' DAGs concatenated, one segment-softmax over all their action
        menus, regardless of how many requests there are. Needs no kernel handle -- everything an
        action needs (including APPLY candidate features) is already on the GoalRequest.
        Row-for-row identical (to float32 tolerance) to looping the legacy predict() below; see
        test_phase3_vectorized.py's forward_batch-vs-looped-forward checks.

        Requests with zero legal actions are scored for value only (mirroring predict()'s own
        `if not actions: return {}, 0.5` short-circuit) and are not sent through the action-scoring
        path at all, since collate_actions has nothing to batch for them anyway."""
        if not requests:
            return []
        scoreable_idx = [i for i, r in enumerate(requests) if r.actions]
        out = [({}, 0.5)] * len(requests)   # placeholder for the zero-action requests
        with torch.no_grad():
            if len(scoreable_idx) == 1:
                # skip collation for the overwhelmingly common case (search asks one goal at a time
                # outside of an explicit multi-goal batch): identical result, less overhead.
                r = requests[scoreable_idx[0]]
                cts = list(zip(r.candidate_tags.tolist(), r.candidate_scalars.tolist()))
                priors_t, value_t = self.model(r.observation, list(r.actions), cts)
                out[scoreable_idx[0]] = ({a: float(v.item()) for a, v in priors_t.items()}, float(value_t.item()))
            elif scoreable_idx:
                scoreable = [requests[i] for i in scoreable_idx]
                batch, ab = pa.collate_requests(scoreable)
                priors, values = self.model.forward_batch(batch, ab)
                for j, i in enumerate(scoreable_idx):
                    r = requests[i]
                    lo = int(ab.offsets[j])
                    p = {a: float(priors[lo + k].item()) for k, a in enumerate(r.actions)}
                    out[i] = (p, float(values[j].item()))
        return out

    def _resolve_candidate(self, candidate) -> tuple[int, int]:
        """Returns (tag, scalar) for a single candidate NodeId, via the
        cheapest possible kernel call: observing just that one, bare
        node (a Var or Const reference has no children at all, so this
        never recurses)."""
        tags, scalar, child_a, child_b = self.kernel.observe_term(candidate)
        return int(tags[0]), int(scalar[0])

    def predict(self, observation, actions: list[Action], candidates: list) -> tuple[dict[Action, float], float]:
        if not actions:
            return {}, 0.5
        candidate_tag_scalars = [self._resolve_candidate(c) for c in candidates]
        with torch.no_grad():
            priors_t, value_t = self.model(observation, actions, candidate_tag_scalars)
        priors = {a: float(v.item()) for a, v in priors_t.items()}
        return priors, float(value_t.item())
