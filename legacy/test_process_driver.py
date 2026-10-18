"""test_process_driver.py -- Phase 4C correctness gate.

Everything here is a CORRECTNESS test, not a performance test: this sandbox has one vCPU, so a
process pool cannot show a real speedup here regardless of whether the design is sound. What can be
checked on one core is exactly what matters most before trusting this on real hardware: do N worker
processes, each with its own kernel, talking to a shared batching server process over queues,
produce results BYTE-IDENTICAL to a single process running the same goals sequentially?
"""
import sys

import torch

import curriculum
import hypergraph as hg
from model import PolicyValueModel
from process_driver import run_multiprocess

failures = 0


def check(cond, msg):
    global failures
    if cond:
        print(f"[PASS] {msg}")
    else:
        failures += 1
        print(f"[FAIL] {msg}")


def reference_fingerprint(name, model, sims=150, top_k=16):
    from trainable_policy import TrainablePolicy
    all_goals = {g.name: g for g in list(curriculum.CURRICULUM) + list(curriculum.HELD_OUT)}
    g = all_goals[name]
    k, refs = curriculum.build_base_kernel()
    lib = k.index_library()
    ps = k.start_proof(g.build_type(k, refs))
    dm = {}
    root = hg.get_or_create_node(k, ps, 0, dm)
    status = hg.search(k, root, lib, TrainablePolicy(model), dm, max_simulations=sims, top_k=top_k)
    edges = sum(len(n.tactics) for n in dm.values())
    visits = sum(e.N for n in dm.values() for e in n.tactics.values())
    return dict(status=status.name, nodes=len(dm), edges=edges, visits=visits)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PolicyValueModel(embed_dim=24)  # small, so the correctness run finishes quickly
    GOAL_NAMES = [g.name for g in list(curriculum.CURRICULUM) + list(curriculum.HELD_OUT)]
    SIMS, TOP_K = 150, 16

    reference = {name: reference_fingerprint(name, model, SIMS, TOP_K) for name in GOAL_NAMES}
    check(all(r["status"] == "SOLVED" for r in reference.values()), "single-process reference: every goal solves")

    for n_workers in (1, 3, 8):
        # more goals than workers, and NOT evenly divisible, to exercise real round-robin bucketing
        names = (GOAL_NAMES * 2)[: len(GOAL_NAMES) + 3]
        results, stats = run_multiprocess(names, model, n_workers=n_workers, sims=SIMS, top_k=TOP_K,
                                          model_kwargs=dict(embed_dim=24))
        check(len(results) == len(GOAL_NAMES),
              f"n_workers={n_workers}: got a result for every distinct goal name ({len(results)}/{len(GOAL_NAMES)})")
        all_match = all(results.get(name) == reference[name] for name in GOAL_NAMES)
        if not all_match:
            for name in GOAL_NAMES:
                if results.get(name) != reference[name]:
                    print(f"   MISMATCH {name}: got {results.get(name)} vs reference {reference[name]}")
        check(all_match, f"n_workers={n_workers}: every result (nodes/edges/visits/status) matches the "
                         f"single-process reference exactly")
        check(stats.wall_seconds > 0 and stats.n_workers == min(n_workers, len(names)),
              f"n_workers={n_workers}: driver stats sane (wall={stats.wall_seconds:.1f}s, "
              f"{stats.server_calls} server calls, mean batch {stats.mean_server_batch:.1f})")

    # a worker with ZERO assigned goals (more workers than goals) must not hang or crash
    results, stats = run_multiprocess(GOAL_NAMES[:2], model, n_workers=8, sims=SIMS, top_k=TOP_K,
                                      model_kwargs=dict(embed_dim=24))
    check(len(results) == 2 and all(results[n] == reference[n] for n in GOAL_NAMES[:2]),
          "more workers than goals (some workers get nothing to do): no hang, correct results")

    print()
    print("ALL PROCESS DRIVER TESTS PASSED" if failures == 0 else f"{failures} PROCESS DRIVER TEST(S) FAILED")
    sys.exit(1 if failures else 0)
