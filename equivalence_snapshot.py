"""Behavioral fingerprint of the search, for equivalence gating.

Search is deterministic given a policy, so any change that is supposed to be
behavior-preserving (compacting commit, driver refactor, ...) must reproduce
these numbers EXACTLY: per goal, the solve status, tree size, total edges,
and total visit count. Run with `python equivalence_snapshot.py out.json`.
"""
import json, sys
import torch
import curriculum, hypergraph as hg
from model import PolicyValueModel
from trainable_policy import TrainablePolicy


def fingerprint(dm, status):
    edges = sum(len(n.tactics) for n in dm.values())
    visits = sum(e.N for n in dm.values() for e in n.tactics.values())
    return dict(status=status.name, nodes=len(dm), edges=edges, visits=visits)


def run(policy_factory, goals, kcls=None, sims=200, top_k=32):
    out = {}
    k, refs = curriculum.build_base_kernel()
    for goal in goals:
        lib = k.index_library(); gty = goal.build_type(k, refs)
        ps = k.start_proof(gty); rm = ps.goal_meta_id(0); dm = {}
        root = hg.get_or_create_node(k, ps, 0, dm)
        status = hg.search(k, root, lib, policy_factory(k), dm, max_simulations=sims, top_k=top_k)
        out[goal.name] = fingerprint(dm, status)
        if status == hg.Status.SOLVED and goal.lemma_name:
            term = hg.commit_and_verify(k, ps, root, rm, gty)
            refs[goal.lemma_name + '_c'] = k.mk_const(k.declare_checked(gty, term, goal.lemma_name))
    return out


def add_comm_fingerprint(sims=300):
    k, refs = curriculum.build_base_kernel()
    for goal in curriculum.CURRICULUM:
        if goal.lemma_name is None: continue
        lib = k.index_library(); gty = goal.build_type(k, refs)
        ps = k.start_proof(gty); rm = ps.goal_meta_id(0); dm = {}
        root = hg.get_or_create_node(k, ps, 0, dm)
        hg.search(k, root, lib, hg.UniformPolicy(), dm, max_simulations=300, top_k=16)
        term = hg.commit_and_verify(k, ps, root, rm, gty)
        refs[goal.lemma_name + '_c'] = k.mk_const(k.declare_checked(gty, term, goal.lemma_name))
    eq = lambda a, x, y: k.mk_app(k.mk_app(k.mk_app(refs['eq_ty'], a), x), y)
    n = refs['nat_ty']; add = lambda x, y: k.mk_app(k.mk_app(refs['add_c'], x), y)
    goal = k.mk_pi(n, k.mk_pi(n, eq(n, add(k.mk_var(1), k.mk_var(0)), add(k.mk_var(0), k.mk_var(1)))))
    lib = k.index_library(); ps = k.start_proof(goal); dm = {}
    root = hg.get_or_create_node(k, ps, 0, dm)
    status = hg.search(k, root, lib, hg.UniformPolicy(), dm, max_simulations=sims, top_k=16)
    return fingerprint(dm, status)


def snapshot():
    snap = {}
    snap['uniform_curriculum'] = run(lambda k: hg.UniformPolicy(), curriculum.CURRICULUM)
    snap['uniform_heldout'] = {}
    for g in curriculum.HELD_OUT:
        snap['uniform_heldout'].update(run(lambda k: hg.UniformPolicy(), [g]))
    torch.manual_seed(0)
    model = PolicyValueModel(embed_dim=32)
    snap['model_seed0_curriculum'] = run(lambda k: TrainablePolicy(model, k), curriculum.CURRICULUM)
    snap['add_comm_uniform_300'] = add_comm_fingerprint()
    return snap


if __name__ == '__main__':
    import os
    hg.COMPACT_PROBES = os.environ.get('ITP_COMPACT', '1') == '1'
    snap = snapshot()
    print('compaction_stats (0=compacted 1=old-meta-solved 2=new-meta-reachable 3=disabled):', hg.compaction_stats, file=sys.stderr)
    json.dump(snap, open(sys.argv[1], 'w'), indent=1, sort_keys=True)
    print(json.dumps(snap, indent=1, sort_keys=True)[:1800])
