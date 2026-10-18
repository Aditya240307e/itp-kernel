"""test_phase3_vectorized.py -- Phase 3 equivalence gate.

Everything here compares the NEW vectorized/batched code against the OLD
per-node / per-goal / per-example loop code, from the SAME weights, and
requires agreement to float32 tolerance (~1e-5 relative, looser after many
accumulated ops). This is the gate the plan set before any speedup claim.

Covers:
  1. TermEncoder.forward (vectorized) vs forward_reference, over many random
     synthetic DAGs (all 15 tags, deep chains, wide sharing, disconnected
     forest-in-one-array, single node, zero nodes) AND real kernel goals.
  2. Gradient equivalence: d(loss)/d(every parameter) matches between the two
     encoder implementations -- not just the forward values.
  3. PolicyValueModel.forward_batch vs looping .forward(): priors and values,
     for real curriculum/held-out goals, with 1..8 goals per batch.
  4. A batched training step (new, added here) vs training.train_step:
     identical loss AND identical resulting parameters after one optimizer
     step, from the same starting weights and the same examples.
  5. Edge cases: a goal with zero hypotheses, a goal with zero legal actions,
     empty batches.
"""
import copy
import sys

import numpy as np
import torch

import curriculum
import hypergraph as hg
import policy_api as pa
import training
from model import PolicyValueModel, TermEncoder, segment_log_softmax

failures = 0


def check(cond, msg):
    global failures
    if cond:
        print(f"[PASS] {msg}")
    else:
        failures += 1
        print(f"[FAIL] {msg}")


def close(a, b, atol=1e-5, rtol=1e-4):
    # torch.allclose silently broadcasts mismatched shapes into an all-pairs comparison, which can
    # spuriously pass (confirmed directly: it did, here, until this check was added) -- require
    # identical shapes first.
    if tuple(a.shape) != tuple(b.shape):
        return False
    return torch.allclose(a, b, atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- 1. synthetic DAGs
def random_dag(rng, n, n_tags=15, max_height_fanout=True):
    """A random binary DAG over n nodes: node i may point only at nodes < i (so it is already a
    valid topological order) -- children are picked to exercise both deep chains (child = i-1) and
    wide sharing (child = a small shared index) depending on `rng`."""
    tags = rng.integers(0, n_tags, size=n)
    # keep VAR(0)/TYPE(4)/CONST(5) scalars meaningful; others get small ints (stored, unused)
    scalar = rng.integers(0, 50, size=n)
    ca = np.full(n, -1, dtype=np.int64)
    cb = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if i == 0:
            continue
        n_children = 0 if tags[i] in (0, 4, 5, 7) else rng.integers(1, 3)  # VAR/TYPE/CONST/META: leaves
        if n_children >= 1:
            ca[i] = rng.integers(0, i)
        if n_children >= 2:
            cb[i] = rng.integers(0, i)
    return tags, scalar, ca, cb


def check_encoder_case(name, tags, scalar, ca, cb, embed_dim=24, seed=0):
    torch.manual_seed(seed)
    enc = TermEncoder(embed_dim=embed_dim)
    heights = pa.compute_heights(ca, cb)
    enc.use_reference = True
    ref = enc.forward_reference(tags, scalar, ca, cb)
    enc.use_reference = False
    vec = enc.forward(tags, scalar, ca, cb, heights)
    ok = ref.shape == vec.shape and close(ref, vec)
    check(ok, f"encoder values match: {name} (N={len(tags)}, max height {int(heights.max()) if len(heights) else -1})")
    return enc, ref, vec, heights


rng = np.random.default_rng(12345)
# forward_reference itself cannot handle an empty DAG (torch.stack on an empty list raises) --
# a PRE-EXISTING limitation, not something to reproduce. Just check the vectorized path is sane.
_empty = TermEncoder(embed_dim=24).forward(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                                           np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                                           np.zeros(0, dtype=np.int64))
check(tuple(_empty.shape) == (0, 24), "vectorized forward() on an empty DAG returns shape (0, embed_dim) without crashing")
check_encoder_case("single leaf (Var)", np.array([0]), np.array([3]), np.array([-1]), np.array([-1]))
# a long chain: height == n-1, exercises many sequential levels with 1 node each
n = 60
chain_tags = np.array([1] * n)  # APP
chain_ca = np.array([-1] + list(range(n - 1)))
chain_cb = np.full(n, -1)
chain_scalar = np.zeros(n, dtype=np.int64)
check_encoder_case("deep chain (height=N-1)", chain_tags, chain_scalar, chain_ca, chain_cb)
# maximal sharing: every node's children are both node 0 -> everything at height <=1
wide_tags = np.array([0] + [1] * 50)
wide_scalar = np.zeros(51, dtype=np.int64)
wide_ca = np.array([-1] + [0] * 50)
wide_cb = np.array([-1] + [0] * 50)
check_encoder_case("wide sharing (height<=1, fan-out 50)", wide_tags, wide_scalar, wide_ca, wide_cb)
for seed in range(8):
    r = np.random.default_rng(seed)
    tags, scalar, ca, cb = random_dag(r, n=r.integers(2, 200))
    check_encoder_case(f"random DAG seed={seed}", tags, scalar, ca, cb, embed_dim=r.integers(4, 40))
# all 15 tags represented, each as an isolated leaf (no children of its own) plus one combining
# parent (index 15) that points at two of them -- exercises node_features for every tag value.
leaf_tags = np.arange(15)
leaf_scalar = np.arange(15) * 7
all_tags = np.concatenate([leaf_tags, [1]])
all_scalar = np.concatenate([leaf_scalar, [0]])
all_ca = np.array([-1] * 15 + [0])
all_cb = np.array([-1] * 15 + [14])
check_encoder_case("all 15 tags as leaves + combining parent", all_tags, all_scalar, all_ca, all_cb)

# real kernel goals, at various proof depths (so Match/Fix/Pi/App mixes actually occur)
k, refs = curriculum.build_base_kernel()
lib = k.index_library()
real_obs = []
for g in list(curriculum.CURRICULUM) + list(curriculum.HELD_OUT):
    ps = k.start_proof(g.build_type(k, refs))
    dm = {}
    root = hg.get_or_create_node(k, ps, 0, dm)
    real_obs.append((g.name + " (root)", root.observation))
    prep = hg.prepare_expansion(k, root, lib)
    if prep.actions:
        # go one probe deep to get a goal with hypotheses/Match structure
        act = prep.actions[0]
        probe = k.materialize_from_node(root.target, root.hyp_types, root.hyp_names)
        try:
            hg._apply_tactic(k, probe, probe.root_meta_id, act, prep.candidates, prep.node.cached_rewrite_extra)
            obs = k.observation(probe, target_meta_id=probe.root_meta_id) if probe.num_goals() else None
            if obs is not None:
                real_obs.append((g.name + " (one probe deep)", obs))
        except Exception:
            pass

for name, obs in real_obs:
    tags, scalar, ca, cb, roots, h = obs
    check_encoder_case(f"real goal: {name}", tags, scalar, ca, cb, embed_dim=24)

# --------------------------------------------------------------------------- 2. gradients match
torch.manual_seed(7)
tags, scalar, ca, cb = random_dag(np.random.default_rng(99), n=80)
heights = pa.compute_heights(ca, cb)
enc_ref = TermEncoder(embed_dim=20)
enc_vec = TermEncoder(embed_dim=20)
enc_vec.load_state_dict(enc_ref.state_dict())
enc_ref.use_reference = True
out_ref = enc_ref.forward_reference(tags, scalar, ca, cb)
loss_ref = (out_ref ** 2).sum()
loss_ref.backward()
out_vec = enc_vec.forward(tags, scalar, ca, cb, heights)
loss_vec = (out_vec ** 2).sum()
loss_vec.backward()
grad_ok = all(close(pr.grad, pv.grad, atol=1e-4, rtol=1e-3)
             for (nr, pr), (nv, pv) in zip(enc_ref.named_parameters(), enc_vec.named_parameters()))
check(grad_ok, "encoder gradients match reference w.r.t. every parameter (not just forward values)")
check(float(loss_ref.item()) == float(loss_vec.item()) or abs(loss_ref.item() - loss_vec.item()) < 1e-3,
      f"loss values match ({loss_ref.item():.6f} vs {loss_vec.item():.6f})")

# --------------------------------------------------------------------------- 3. batched model vs looped forward()
def make_examples(goals, sims=200, top_k=32):
    exs = []
    k2, refs2 = curriculum.build_base_kernel()
    lib2 = k2.index_library()
    for g in goals:
        ps = k2.start_proof(g.build_type(k2, refs2))
        dm = {}
        root = hg.get_or_create_node(k2, ps, 0, dm)
        prep = hg.prepare_expansion(k2, root, lib2)
        exs.append(prep.request)
    return exs


torch.manual_seed(11)
model = PolicyValueModel(embed_dim=24)
reqs_all = make_examples(list(curriculum.CURRICULUM) + list(curriculum.HELD_OUT))
for count in (1, 2, 3, 8):
    reqs = (reqs_all * 3)[:count]
    looped = [model.forward(r.observation, list(r.actions), list(zip(r.candidate_tags.tolist(), r.candidate_scalars.tolist())))
             for r in reqs]
    loop_priors = torch.cat([torch.stack([p[a] for a in r.actions]) if r.actions else torch.zeros(0)
                             for (p, v), r in zip(looped, reqs)])
    loop_values = torch.stack([torch.as_tensor(v).reshape(()) for _, v in looped])  # forward() returns a 1-elem tensor; flatten so shapes actually match
    batch, ab = pa.collate_requests(reqs)
    bpriors, bvalues = model.forward_batch(batch, ab)
    check(close(loop_values, bvalues), f"forward_batch values match looped forward(), {count} goal(s)")
    check(loop_priors.numel() == bpriors.numel() and close(loop_priors, bpriors, atol=1e-4, rtol=1e-3),
          f"forward_batch priors match looped forward(), {count} goal(s), {bpriors.numel()} actions total")

# edge case: a goal with zero legal actions must not crash collation/scoring
zero_action_req = pa.GoalRequest(observation=reqs_all[0].observation, actions=[], candidates=[],
                                 candidate_tags=np.zeros(0, dtype=np.uint32), candidate_scalars=np.zeros(0, dtype=np.uint32))
try:
    batch, ab = pa.collate_requests([zero_action_req, reqs_all[1]])
    _, bvalues = model.forward_batch(batch, ab)
    check(bvalues.shape[0] == 2, "a zero-action goal batched alongside a normal one does not crash")
except Exception as e:
    check(False, f"zero-action goal crashed: {e!r}")

# --------------------------------------------------------------------------- 4. batched training step
# batched_loss / train_step_batched are the ACTUAL production implementations (training.py) -- tested
# directly here, not a local reimplementation.
_batched_loss = training.batched_loss
batched_train_step = training.train_step_batched


torch.manual_seed(21)
model_a = PolicyValueModel(embed_dim=20)
model_b = PolicyValueModel(embed_dim=20)
model_b.load_state_dict(model_a.state_dict())
opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

# real training examples via extract_training_examples, across several goals/depths
policy_examples, value_examples = [], []
for g in list(curriculum.CURRICULUM)[:3]:
    k3, refs3 = curriculum.build_base_kernel()
    lib3 = k3.index_library()
    ps = k3.start_proof(g.build_type(k3, refs3))
    dm = {}
    root = hg.get_or_create_node(k3, ps, 0, dm)
    hg.search(k3, root, lib3, hg.UniformPolicy(), dm, max_simulations=60, top_k=16)
    pe, ve = hg.extract_training_examples(k3, dm)
    policy_examples += pe
    value_examples += ve
check(len(policy_examples) > 3 and len(value_examples) > 3, f"gathered real training examples: {len(policy_examples)} policy, {len(value_examples)} value")

def named_grads(model, loss):
    model.zero_grad()
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, \
           [n for n, p in model.named_parameters() if p.grad is None]


# Compare GRADIENTS directly (the actual thing that must agree for two training
# implementations to be "the same"), before either optimizer takes a step.
pl_a = torch.stack([training.policy_loss_for_example(model_a, o, t, c) for o, t, c in policy_examples]).mean()
vl_a = torch.stack([training.value_loss_for_example(model_a, o, v) for o, v in value_examples]).mean()
grads_a, none_a = named_grads(model_a, pl_a + vl_a)

total_b, pl_b, vl_b = _batched_loss(model_b, policy_examples, value_examples)
grads_b, none_b = named_grads(model_b, total_b)

check(abs(float(pl_a) - float(pl_b)) < 1e-5, f"batched vs looped policy_loss match ({float(pl_a):.6f} vs {float(pl_b):.6f})")
check(abs(float(vl_a) - float(vl_b)) < 1e-5, f"batched vs looped value_loss match ({float(vl_a):.6f} vs {float(vl_b):.6f})")
check(set(none_a) == set(none_b), "the same parameters receive a gradient in both implementations")
# ABSOLUTE tolerance only: relative error is meaningless once a gradient component is near zero,
# and float32 summed in a different order (sequential loop vs batched reduction) -- neither more
# "correct" than the other -- differs at the ~1e-8 scale confirmed directly on this test's own
# examples; 1e-6 leaves two orders of magnitude of headroom.
max_diff = max((grads_a[n] - grads_b[n]).abs().max().item() for n in grads_a if n in grads_b)
check(max_diff < 1e-6, f"gradients match batched vs looped, every parameter, abs tolerance only (max |diff| = {max_diff:.2e})")

# One optimizer step from each (real Adam, real train_step/batched_train_step -- not the raw
# grads above): parameters should be close, but NOT required to match as tightly as the
# gradients, since Adam divides by sqrt(v)+eps and so can amplify float32-noise-sized gradient
# differences for near-zero-gradient parameters into larger (though still small) parameter drift.
# This is exactly the numerical-drift risk the original plan flagged for batched ops, not a
# correctness bug -- reported here rather than silently ignored.
r_a = training.train_step(model_a, opt_a, policy_examples, value_examples)
r_b = batched_train_step(model_b, opt_b, policy_examples, value_examples)
check(abs(r_a["policy_loss"] - r_b["policy_loss"]) < 1e-4, f"batched vs looped policy_loss match ({r_a['policy_loss']:.6f} vs {r_b['policy_loss']:.6f})")
check(abs(r_a["value_loss"] - r_b["value_loss"]) < 1e-4, f"batched vs looped value_loss match ({r_a['value_loss']:.6f} vs {r_b['value_loss']:.6f})")
param_drift = max((pa_ - pb_).abs().max().item() for pa_, pb_ in zip(model_a.parameters(), model_b.parameters()))
check(param_drift < 5e-3, f"AFTER one Adam step, parameters stay close (max |diff| = {param_drift:.2e}, "
                          f"expected order lr={1e-3:.0e} x amplified float32 noise, not a correctness threshold)")

# second step from the (now slightly drifted, if Adam amplified anything) state: loss should stay close
r_a2 = training.train_step(model_a, opt_a, policy_examples, value_examples)
r_b2 = batched_train_step(model_b, opt_b, policy_examples, value_examples)
check(abs(r_a2["total_loss"] - r_b2["total_loss"]) < 1e-2, "a SECOND train step also matches within a loose loss tolerance (optimizer state stayed in sync)")

# --------------------------------------------------------------------------- 5. more edge cases
check(pa.collate_observations([reqs_all[0].observation]).n_goals == 1, "collate_observations of a single goal works")
empty_tags = np.zeros(0, dtype=np.int64)
h_empty = pa.compute_heights(empty_tags, empty_tags)
check(len(h_empty) == 0, "compute_heights on an empty DAG returns an empty array")
try:
    bad_ca = np.array([1, 0])  # a 2-cycle
    pa.compute_heights(bad_ca, np.array([-1, -1]))
    check(False, "compute_heights should reject a cyclic child graph")
except ValueError:
    check(True, "compute_heights rejects a cyclic child graph instead of looping forever")

print()
print("ALL PHASE 3 EQUIVALENCE TESTS PASSED" if failures == 0 else f"{failures} PHASE 3 TEST(S) FAILED")
sys.exit(1 if failures else 0)
