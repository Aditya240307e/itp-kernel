"""
training.py -- loss computation and the supervised training step for
PolicyValueModel, operating entirely on extract_training_examples' own
output (already-resolved, portable data -- no kernel access needed
here at all, by design; see extract_training_examples' own docstring).
"""

from __future__ import annotations
import numpy as np
import torch
import policy_api as pa
from model import PolicyValueModel, segment_log_softmax


def policy_loss_for_example(model: PolicyValueModel, observation, target: dict, candidate_tag_scalars: list) -> torch.Tensor:
    """Cross-entropy between the model's own softmax over target's own
    action set (NOT the full, original candidate menu -- only the
    actions the search actually tried, matching the AlphaZero/HTPS-
    style positive-only supervision this design has followed
    throughout; see extract_training_examples' own docstring) and the
    search's own visit-count target distribution."""
    from hypergraph import ActionKind
    goal_embed, node_embeds, roots = model.encode_goal(observation)
    actions = list(target.keys())
    scores = []
    for action in actions:
        cts = candidate_tag_scalars[action.arg] if action.kind == ActionKind.APPLY else None
        a_embed = model.action_embed(action, node_embeds, roots, cts)
        scores.append(model.score_head(torch.cat([goal_embed, a_embed])))
    scores_t = torch.stack(scores).squeeze(-1)
    log_probs = torch.log_softmax(scores_t, dim=0)
    target_t = torch.tensor([target[a] for a in actions], dtype=torch.float32)
    return -(target_t * log_probs).sum()


def value_loss_for_example(model: PolicyValueModel, observation, target_value: float) -> torch.Tensor:
    """MSE between the model's own predicted value and the search's
    own backed-up target (see extract_training_examples' own docstring
    for why MSE, not BCE: targets are genuinely continuous in [0, 1],
    not strictly binary)."""
    goal_embed, _, _ = model.encode_goal(observation)
    pred = torch.sigmoid(model.value_head(goal_embed)).squeeze(-1)
    return (pred - target_value) ** 2


def train_step(model: PolicyValueModel, optimizer: torch.optim.Optimizer,
               policy_examples: list, value_examples: list,
               policy_weight: float = 1.0, value_weight: float = 1.0) -> dict:
    """One full pass over the given examples: accumulates policy loss
    over every policy example and value loss over every value example,
    backprops the combined loss ONCE, and takes one optimizer step.
    Returns the (unweighted) mean policy and value loss for logging."""
    optimizer.zero_grad()
    policy_losses = [policy_loss_for_example(model, obs, target, cts) for obs, target, cts in policy_examples]
    value_losses = [value_loss_for_example(model, obs, v) for obs, v in value_examples]

    mean_policy_loss = torch.stack(policy_losses).mean() if policy_losses else torch.tensor(0.0)
    mean_value_loss = torch.stack(value_losses).mean() if value_losses else torch.tensor(0.0)
    total_loss = policy_weight * mean_policy_loss + value_weight * mean_value_loss
    total_loss.backward()
    optimizer.step()

    return {
        "policy_loss": float(mean_policy_loss.item()),
        "value_loss": float(mean_value_loss.item()),
        "total_loss": float(total_loss.item()),
        "n_policy_examples": len(policy_examples),
        "n_value_examples": len(value_examples),
    }


def batched_loss(model: PolicyValueModel, policy_examples: list, value_examples: list,
                 policy_weight: float = 1.0, value_weight: float = 1.0):
    """The Phase 3 batched equivalent of accumulating policy_loss_for_example/value_loss_for_example
    over every example: ONE model.forward_batch call for all policy examples' goals and ONE
    model.encode_batch/value_head pass for all value examples' goals, instead of a Python loop that
    re-runs the encoder once per example. Matches the per-example loop to float32 gradient tolerance
    (~1e-8 absolute; see test_phase3_vectorized.py) -- NOT bit-identical, since a batched reduction
    sums in a different order than a sequential Python loop, which is a real (tiny) source of
    numerical drift, not a bug. Returns (total, mean_policy_loss, mean_value_loss) as un-backward'd
    tensors, so a caller can inspect them (or add more terms) before calling .backward() itself --
    see train_step_batched below for the normal use.

    Each policy example gets its OWN softmax group even if two examples share the same goal
    (e.g. re-sampled at different rounds): the ActionBatch's `segment` is the example index, and the
    goal collation is per-example rather than deduplicated, matching what the per-example loop does
    (each example computes its own encode_goal call, so it also never merges two examples' goals)."""
    from hypergraph import ActionKind  # local import, as elsewhere in this file's style: keeps training.py -> hypergraph.py one-directional
    mean_policy_loss = torch.tensor(0.0)
    mean_value_loss = torch.tensor(0.0)
    if policy_examples:
        batch = pa.collate_observations([obs for obs, target, cts in policy_examples])
        kinds_list, args_list, cand_t_list, cand_s_list, target_list = [], [], [], [], []
        for obs, target, cts in policy_examples:
            actions = list(target.keys())
            kinds_list.append(np.fromiter((int(a.kind) for a in actions), dtype=np.int32, count=len(actions)))
            args_list.append(np.fromiter((int(a.arg) for a in actions), dtype=np.int32, count=len(actions)))
            cand_t_list.append(np.asarray(cts and [c[0] for c in cts], dtype=np.int64) if cts else np.zeros(0, dtype=np.int64))
            cand_s_list.append(np.asarray(cts and [c[1] for c in cts], dtype=np.int64) if cts else np.zeros(0, dtype=np.int64))
            target_list.append(np.fromiter((target[a] for a in actions), dtype=np.float64, count=len(actions)))
        ab = pa.collate_actions(batch, kinds_list, args_list, cand_t_list, cand_s_list, goals=range(len(policy_examples)))
        target_t = torch.from_numpy(np.concatenate(target_list)).float()
        goal_embed, node_embeds = model.encode_batch(batch)
        scores = model.score_actions(goal_embed, node_embeds, ab)
        log_probs = segment_log_softmax(scores, torch.from_numpy(ab.segment), ab.n_segments)
        per_example = torch.zeros(len(policy_examples)).index_add(0, torch.from_numpy(ab.segment), -target_t * log_probs)
        mean_policy_loss = per_example.mean()
    if value_examples:
        vbatch = pa.collate_observations([obs for obs, v in value_examples])
        vgoal_embed, _ = model.encode_batch(vbatch)
        pred = torch.sigmoid(model.value_head(vgoal_embed)).squeeze(-1)
        target_v = torch.tensor([v for _, v in value_examples], dtype=torch.float32)
        mean_value_loss = ((pred - target_v) ** 2).mean()
    total_loss = policy_weight * mean_policy_loss + value_weight * mean_value_loss
    return total_loss, mean_policy_loss, mean_value_loss


def train_step_batched(model: PolicyValueModel, optimizer: torch.optim.Optimizer,
                       policy_examples: list, value_examples: list,
                       policy_weight: float = 1.0, value_weight: float = 1.0) -> dict:
    """Drop-in replacement for train_step: same signature, same returned dict, one optimizer step --
    but the encoder runs ONCE over all examples' goals concatenated (see batched_loss) instead of once
    per example. Prefer this for any round with more than a handful of examples; train_step remains
    for exact-loop-order needs and as batched_loss's own test oracle."""
    optimizer.zero_grad()
    total_loss, mean_policy_loss, mean_value_loss = batched_loss(model, policy_examples, value_examples, policy_weight, value_weight)
    total_loss.backward()
    optimizer.step()
    return {
        "policy_loss": float(mean_policy_loss.item()),
        "value_loss": float(mean_value_loss.item()),
        "total_loss": float(total_loss.item()),
        "n_policy_examples": len(policy_examples),
        "n_value_examples": len(value_examples),
    }
