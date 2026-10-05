"""Phase 2 (Layer 3): the policy contract and the lockstep driver.

Key properties checked:
  * predict_batch of length 1 == predict; a policy implementing ONLY predict_batch works.
  * priors may be a dict OR an array aligned with request.actions -- identical search either way.
  * GoalRequest is self-contained (features present, text() works) and to_portable() is picklable; a
    policy that sees ONLY the portable form (no kernel, no Action objects) reproduces the search exactly.
  * THE driver property: N searches run in lockstep, sharing batched policy calls, each produce
    EXACTLY the tree they produce running alone -- for a stateless policy and for the model.
  * max_batch_size chunking never changes results; the batches actually formed are > 1.
"""
import pickle
import sys

import numpy as np
import torch

import curriculum
import hypergraph as hg
import policy_api as pa
from driver import DriverStats, SearchTask, run_lockstep
from equivalence_snapshot import fingerprint
from model import PolicyValueModel
from trainable_policy import TrainablePolicy

failures = 0


def check(cond, msg):
    global failures
    if cond:
        print(f"[PASS] {msg}")
    else:
        failures += 1
        print(f"[FAIL] {msg}")


GOALS = [curriculum.CURRICULUM[0], curriculum.CURRICULUM[1], curriculum.CURRICULUM[2], curriculum.CURRICULUM[3]] + list(curriculum.HELD_OUT)


def make_task(goal, sims=200, top_k=32, **kw):
    k, refs = curriculum.build_base_kernel()
    lib = k.index_library()
    ps = k.start_proof(goal.build_type(k, refs))
    dm = {}
    root = hg.get_or_create_node(k, ps, 0, dm)
    return SearchTask(kernel=k, root=root, library=lib, dedup_map=dm, max_simulations=sims, top_k=top_k, **kw)


def run_alone(goal, policy_factory, **kw):
    t = make_task(goal, **kw)
    status = hg.search(t.kernel, t.root, t.library, policy_factory(t.kernel), t.dedup_map, max_simulations=t.max_simulations, top_k=t.top_k)
    return fingerprint(t.dedup_map, status)


# --------------------------------------------------------------------------- policies
torch.manual_seed(0)
MODEL = PolicyValueModel(embed_dim=32)


class BatchOnlyUniform:
    """Implements ONLY predict_batch -- no predict(), no PolicyInterface base class."""
    def __init__(self):
        self.calls = []
    def predict_batch(self, requests):
        self.calls.append(len(requests))
        return [({a: 1.0 / len(r.actions) for a in r.actions}, 0.5) if r.actions else ({}, 0.5) for r in requests]


class ArrayUniform:
    """Returns priors as an ARRAY aligned with request.actions, not a dict."""
    def predict_batch(self, requests):
        return [(np.full(len(r.actions), 1.0 / max(1, len(r.actions))), 0.5) for r in requests]


class PortableOnly:
    """Sees ONLY r.to_portable() (numpy arrays + ints, after a pickle round trip) -- the shape of a policy that
    lives in another process. Returns aligned arrays."""
    def __init__(self):
        self.saw_keys = set()
    def predict_batch(self, requests):
        out = []
        for r in requests:
            blob = pickle.loads(pickle.dumps(r.to_portable()))
            self.saw_keys |= set(blob)
            n = len(blob['action_kinds'])
            out.append((np.full(n, 1.0 / max(1, n)), 0.5))
        return out


uni = lambda k: hg.UniformPolicy()
mdl = lambda k: TrainablePolicy(MODEL, k)

# --------------------------------------------------------------------------- 1. contract
t = make_task(GOALS[2])
prep = hg.prepare_expansion(t.kernel, t.root, t.library)
req = prep.request
check(len(req.candidate_tags) == len(req.candidates) == len(req.candidate_scalars), "request carries one (tag, scalar) feature per APPLY candidate")
check("|-" in req.text() and len(req.text()) > 3, f"request.text() renders the goal ({req.text()!r})")
pol = TrainablePolicy(MODEL, t.kernel)
p_legacy, v_legacy = pol.predict(req.observation, req.actions, req.candidates)
(p_batch, v_batch), = pol.predict_batch([req])
check(p_legacy == p_batch and v_legacy == v_batch, "predict_batch([r]) == predict(...) exactly (model policy)")
check(TrainablePolicy(MODEL).predict_batch([req])[0][0] == p_batch, "TrainablePolicy.predict_batch needs no kernel handle")
blob = pickle.loads(pickle.dumps(req.to_portable()))
check(blob['schema_version'] == pa.OBSERVATION_SCHEMA_VERSION and len(blob['action_kinds']) == len(req.actions), "to_portable() survives a pickle round trip")

ref = {g.name: run_alone(g, uni) for g in GOALS}
bo = BatchOnlyUniform()
check(all(run_alone(g, lambda k: bo) == ref[g.name] for g in GOALS), "a policy implementing ONLY predict_batch reproduces the UniformPolicy search exactly")
check(all(run_alone(g, lambda k: ArrayUniform()) == ref[g.name] for g in GOALS), "array-aligned priors give an identical search to dict priors")
po = PortableOnly()
check(all(run_alone(g, lambda k: po) == ref[g.name] for g in GOALS), "a policy that sees ONLY the pickled portable request reproduces the search exactly")
check({'tags', 'action_kinds', 'candidate_tags'} <= po.saw_keys, "portable form exposes DAG arrays, action kinds/args, and candidate features")
try:
    hg.PolicyInterface().predict_batch([req]); ok = False
except NotImplementedError:
    ok = True
check(ok, "a policy implementing neither predict nor predict_batch fails loudly")

# --------------------------------------------------------------------------- 2. lockstep == alone
def lockstep(policy, goals, copies=1, max_batch_size=None):
    tasks = [make_task(g) for _ in range(copies) for g in goals]
    st = DriverStats()
    statuses = run_lockstep(tasks, policy, max_batch_size=max_batch_size, stats=st)
    return [fingerprint(t.dedup_map, s) for t, s in zip(tasks, statuses)], st

for name, factory, policy in (("UniformPolicy", uni, hg.UniformPolicy()), ("seeded model", mdl, TrainablePolicy(MODEL))):
    alone = {g.name: run_alone(g, factory) for g in GOALS}
    got, st = lockstep(policy, GOALS, copies=3)
    exp = [alone[g.name] for _ in range(3) for g in GOALS]
    check(got == exp, f"[{name}] 21 searches in lockstep each equal running alone (nodes/edges/visits/status)")
    check(st.mean_batch > 1.5, f"[{name}] driver formed real batches: mean batch {st.mean_batch:.1f}, max {max(st.batch_sizes)}, {st.policy_calls} calls for {st.requests} goals")
    got2, st2 = lockstep(policy, GOALS, copies=3, max_batch_size=4)
    check(got2 == exp and max(st2.batch_sizes) <= 4, f"[{name}] max_batch_size=4 chunking leaves results unchanged (max batch {max(st2.batch_sizes)})")

# --------------------------------------------------------------------------- 3. batch_frontier (opt-in, not guaranteed identical)
same = 0
for g in GOALS:
    t = make_task(g)
    status = hg.search(t.kernel, t.root, t.library, hg.UniformPolicy(), t.dedup_map, max_simulations=200, top_k=32, batch_frontier=True)
    same += (fingerprint(t.dedup_map, status) == ref[g.name])
    if g.lemma_name and status == hg.Status.SOLVED:
        pass
check(True, f"batch_frontier=True (opt-in): {same}/{len(GOALS)} goals produce the identical tree (not guaranteed; documented)")
solved_ok = True
for g in GOALS:
    k, refs = curriculum.build_base_kernel(); lib = k.index_library(); gty = g.build_type(k, refs)
    ps = k.start_proof(gty); rm = ps.goal_meta_id(0); dm = {}; root = hg.get_or_create_node(k, ps, 0, dm)
    if hg.search(k, root, lib, hg.UniformPolicy(), dm, max_simulations=200, top_k=32, batch_frontier=True) != hg.Status.SOLVED:
        solved_ok = False; continue
    try:
        hg.commit_and_verify(k, ps, root, rm, gty)
    except Exception:
        solved_ok = False
check(solved_ok, "batch_frontier=True still solves every goal AND every proof passes independent verification")

print()
print("ALL PHASE 2 TESTS PASSED" if failures == 0 else f"{failures} PHASE 2 TEST(S) FAILED")
sys.exit(1 if failures else 0)
