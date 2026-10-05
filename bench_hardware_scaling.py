"""bench_hardware_scaling.py -- the numbers the development sandbox (1 vCPU) could not produce.

Two independent strong-scaling sweeps, both against a FIXED total amount of work, varying worker
count from 1 up to os.cpu_count():
  1. GIL-released threads (Phase 4B). On the 1-vCPU sandbox this measured 8x SLOWER at 8 threads
     than sequential, root-caused to Python-bound search bookkeeping dominating wall time (60-70%)
     and threads only being able to parallelize the released-GIL C++ portion -- a real limitation of
     THIS workload's Python/C++ split, not of GIL release itself (proven genuinely released:
     test_gil_release.py). Whether real cores change this conclusion is exactly what this answers.
  2. The process pool (Phase 4C, process_driver.py), which sidesteps the GIL question entirely --
     each worker is a separate interpreter. Measured separately: spawn/import overhead in isolation
     (dominated total time in the short sandbox runs; on real hardware this may be proportionally
     much smaller) and strong scaling at two work sizes, so a genuinely fixed cost is easy to see
     against a workload large enough to amortize it.

Run with: python3 bench_hardware_scaling.py [--goals N] [--sims N] [--skip-threads] [--skip-process]
Writes a summary to hardware_scaling_report.md in addition to stdout.
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # robust to being invoked from elsewhere

import curriculum
import hypergraph as hg
from model import PolicyValueModel
from trainable_policy import TrainablePolicy

REPORT_LINES: list[str] = []


def log(line: str = ""):
    print(line)
    REPORT_LINES.append(line)


def env_summary() -> str:
    import torch
    return (f"platform: {platform.system()} {platform.machine()}, "
           f"cpu_count: {os.cpu_count()}, "
           f"python: {platform.python_version()}, torch: {torch.__version__}")


ALL_GOAL_NAMES = None  # populated in main()


def solve_one(kernel_cls, name, model, sims, top_k):
    import curriculum as _c
    all_goals = {g.name: g for g in list(_c.CURRICULUM) + list(_c.HELD_OUT)}
    g = all_goals[name]
    k, refs = _c.build_base_kernel(kernel_cls)
    lib = k.index_library()
    ps = k.start_proof(g.build_type(k, refs))
    dm = {}
    root = hg.get_or_create_node(k, ps, 0, dm)
    hg.search(k, root, lib, TrainablePolicy(model, k), dm, max_simulations=sims, top_k=top_k)


# --------------------------------------------------------------------------- 1. GIL-released threads
def bench_threads(goal_names: list, model, sims: int, top_k: int, max_workers: int):
    import itp_env
    log("\n## 1. GIL-released threads: strong scaling\n")
    log(f"Fixed total work: {len(goal_names)} goal-solves (sims={sims}, top_k={top_k}). "
       f"Each thread owns its own Kernel (see itp_kernel.hpp's own thread_local design) -- "
       f"never shared -- and calls into the 17 GIL-released bindings plus finish_probe's own "
       f"scoped release; everything else stays serialized by the GIL as usual.")
    counts = sorted(set([1, 2, 4, min(max_workers, 8), max_workers] + ([max_workers // 2] if max_workers > 2 else [])))
    counts = [c for c in counts if 1 <= c <= max_workers]
    results = {}
    for n in counts:
        buckets = [goal_names[i::n] for i in range(n)]

        def run():
            threads = [threading.Thread(target=lambda names=b: [solve_one(itp_env.Kernel, nm, model, sims, top_k) for nm in names])
                      for b in buckets if b]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return time.perf_counter() - t0

        dt = statistics.median(run() for _ in range(3))
        results[n] = dt
    base = results[1]
    log(f"{'threads':>8}{'wall time':>14}{'speedup':>10}")
    for n in counts:
        log(f"{n:>8}{results[n]:>11.2f}s{base/results[n]:>9.2f}x")
    best_n = max(results, key=lambda n: base / results[n])
    log(f"\nbest observed speedup: {base/results[best_n]:.2f}x at {best_n} threads (1.0x = no benefit; "
       f"< 1.0x = threading made it SLOWER, as the 1-vCPU sandbox measured -- see this file's own header).")
    return results


# --------------------------------------------------------------------------- 2. process pool
def _spawn_noop_full(q):
    """Module-level (not nested): 'spawn' pickles a function by qualified name and re-imports it in
    the child, which fails for a locally-defined function -- learned the hard way earlier this
    project (process_driver.py's own worker functions are module-level for the same reason)."""
    import torch, curriculum, hypergraph, itp_env, process_driver  # noqa: F401
    q.put("ready")


def bench_process_spawn_overhead():
    import multiprocessing as mp
    log("\n## 2a. Process-spawn overhead in isolation\n")
    log("Importing torch+curriculum+itp_env in each freshly spawned worker, before any real work --"
       " on the 1-vCPU sandbox this alone cost ~1.6s per process and dominated every short-lived "
       "process-pool measurement there.")
    ctx = mp.get_context("spawn")

    log(f"{'workers':>8}{'total spawn+import time':>28}{'per-worker':>14}")
    for n in sorted(set([1, 2, 4, os.cpu_count() or 4])):
        q = ctx.Queue()
        t0 = time.perf_counter()
        procs = [ctx.Process(target=_spawn_noop_full, args=(q,)) for _ in range(n)]
        for p in procs:
            p.start()
        for _ in range(n):
            q.get()
        for p in procs:
            p.join()
        dt = time.perf_counter() - t0
        log(f"{n:>8}{dt:>25.2f}s{dt/n:>13.2f}s")


def bench_process_pool(goal_names: list, model, sims: int, top_k: int, max_workers: int):
    from process_driver import run_multiprocess
    log("\n## 2b. Process pool: strong scaling\n")
    log(f"Fixed total work: {len(goal_names)} goal-solves (sims={sims}, top_k={top_k}) per data "
       f"point, ONE run_multiprocess call each (so spawn cost is paid once per point, same as any "
       f"other real invocation would pay it) -- distinguishing this from a warm, long-lived pool "
       f"(where spawn cost is paid ONCE across a whole training run, not per round) matters: a "
       f"single-call number here is a LOWER bound on what a warm pool would achieve.")
    model_kwargs = dict(embed_dim=model.embed_dim, const_table_size=model.term_encoder.const_table_size,
                        max_var_level=model.term_encoder.max_var_level, max_type_level=model.term_encoder.max_type_level)
    counts = sorted(set([1, 2, 4, min(max_workers, 8), max_workers]))
    counts = [c for c in counts if 1 <= c <= max_workers]
    results = {}
    server_stats = {}
    for n in counts:
        _, stats = run_multiprocess(goal_names, model, n_workers=n, sims=sims, top_k=top_k, model_kwargs=model_kwargs)
        results[n] = stats.wall_seconds
        server_stats[n] = (stats.server_calls, stats.mean_server_batch)
    base = results[1]
    log(f"{'workers':>8}{'wall time':>14}{'speedup':>10}{'server calls':>15}{'mean batch':>13}")
    for n in counts:
        calls, mean_batch = server_stats[n]
        log(f"{n:>8}{results[n]:>11.2f}s{base/results[n]:>9.2f}x{calls:>15}{mean_batch:>13.1f}")
    best_n = max(results, key=lambda n: base / results[n])
    log(f"\nbest observed speedup (single call, spawn cost included): {base/results[best_n]:.2f}x at {best_n} workers.")
    log("If 'mean batch' stays small even at high worker counts, the shared server isn't actually "
       "pooling much across workers -- each round-trip pays IPC cost for little batching benefit. "
       "See 2d for the direct test: workers scoring with a LOCAL model copy, no shared server at all.")
    return results


def _local_model_worker(goal_names: list, model_state: dict, model_kwargs: dict, sims: int, top_k: int, results_queue):
    """No RemotePolicy, no shared server, no cross-worker batching at all: this worker loads its OWN
    full model copy and scores every goal itself, exactly as if it were running alone. Isolates
    whether the shared-server design's IPC/coordination cost is what's limiting 2b's scaling, by
    removing it entirely and seeing what's left.

    torch.set_num_threads(1) BEFORE importing torch: by default torch's own BLAS/OMP pool tries to
    use every core for a single forward call, so N of these worker processes running concurrently
    means N processes each independently competing for all cores at once -- up to N x cores threads
    contending for `cores` cores. That fights the inter-process parallelism this benchmark is
    actually trying to measure. Confirmed worth trying directly: 2a's own spawn+import time GREW
    with worker count (0.59s -> 1.29s total, 1 -> 12 workers) rather than staying flat, which is
    what contention during those concurrent imports looks like."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    import curriculum as _c
    import hypergraph as hg
    from model import PolicyValueModel
    from trainable_policy import TrainablePolicy

    m = PolicyValueModel(**model_kwargs)
    m.load_state_dict(model_state)
    policy = TrainablePolicy(m)
    all_goals = {g.name: g for g in list(_c.CURRICULUM) + list(_c.HELD_OUT)}
    for name in goal_names:
        g = all_goals[name]
        k, refs = _c.build_base_kernel()
        lib = k.index_library()
        ps = k.start_proof(g.build_type(k, refs))
        dm = {}
        root = hg.get_or_create_node(k, ps, 0, dm)
        hg.search(k, root, lib, policy, dm, max_simulations=sims, top_k=top_k)
    results_queue.put("done")


def bench_process_pool_local_model(goal_names: list, model, sims: int, top_k: int, max_workers: int):
    import multiprocessing as mp
    log("\n## 2d. Process pool, LOCAL model per worker: no shared server, no cross-worker batching\n")
    log(f"Same {len(goal_names)}-goal fixed workload as 2b, round-robin split across workers, but "
       f"each worker loads its own full model copy and scores entirely on its own -- zero IPC for "
       f"scoring, zero coordination between workers. If THIS scales meaningfully better than 2b's "
       f"shared-server numbers, that confirms the server is the bottleneck there. If it scales "
       f"similarly poorly, the bottleneck is elsewhere (not enough real compute per goal at this "
       f"scale, multiprocessing's own overhead, or memory-bandwidth contention across cores).")
    model_kwargs = dict(embed_dim=model.embed_dim, const_table_size=model.term_encoder.const_table_size,
                        max_var_level=model.term_encoder.max_var_level, max_type_level=model.term_encoder.max_type_level)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ctx = mp.get_context("spawn")
    counts = sorted(set([1, 2, 4, min(max_workers, 8), max_workers]))
    counts = [c for c in counts if 1 <= c <= max_workers]
    results = {}
    for n in counts:
        buckets = [goal_names[i::n] for i in range(n)]
        q = ctx.Queue()
        procs = [ctx.Process(target=_local_model_worker, args=(b, state, model_kwargs, sims, top_k, q)) for b in buckets if b]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        for _ in procs:
            q.get()
        for p in procs:
            p.join()
        results[n] = time.perf_counter() - t0
    base = results[1]
    log(f"{'workers':>8}{'wall time':>14}{'speedup':>10}")
    for n in counts:
        log(f"{n:>8}{results[n]:>11.2f}s{base/results[n]:>9.2f}x")
    best_n = max(results, key=lambda n: base / results[n])
    log(f"\nbest observed speedup (local model, no server): {base/results[best_n]:.2f}x at {best_n} workers.")
    return results


def _sustained_local_model_worker(model_state: dict, model_kwargs: dict, sims: int, top_k: int,
                                  duration_s: float, results_queue):
    """Runs for a FIXED WALL-CLOCK DURATION, not a fixed goal count: repeatedly solves goals
    (cycling through the curriculum) until time is up, then reports how many it completed. This is
    the sustained-throughput counterpart to _local_model_worker's fixed-work version -- see
    bench_process_pool_sustained's own docstring for why this is the more representative measure."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import time as _time
    import torch
    torch.set_num_threads(1)
    import curriculum as _c
    import hypergraph as hg
    from model import PolicyValueModel
    from trainable_policy import TrainablePolicy

    m = PolicyValueModel(**model_kwargs)
    m.load_state_dict(model_state)
    policy = TrainablePolicy(m)
    goals = list(_c.CURRICULUM) + list(_c.HELD_OUT)
    n_solved = 0
    end = _time.perf_counter() + duration_s
    i = 0
    while _time.perf_counter() < end:
        g = goals[i % len(goals)]
        i += 1
        k, refs = _c.build_base_kernel()
        lib = k.index_library()
        ps = k.start_proof(g.build_type(k, refs))
        dm = {}
        root = hg.get_or_create_node(k, ps, 0, dm)
        hg.search(k, root, lib, policy, dm, max_simulations=sims, top_k=top_k)
        n_solved += 1
    results_queue.put(n_solved)


def bench_process_pool_sustained(model, sims: int, top_k: int, max_workers: int, duration_s: float = 20.0):
    """The number that actually answers 'are we fast enough for real training': every prior process-
    pool measurement in this file (2b, 2d) times a single short burst of fixed work, which means a
    good chunk of what gets measured is ONE-TIME spawn+import cost that a real, long-running worker
    pool pays once and then amortizes away over hours -- not something that should count against
    steady-state throughput. This instead spawns N workers ONCE and lets them run continuously for
    `duration_s`, each independently cycling through the curriculum for as long as they're given,
    then compares AGGREGATE THROUGHPUT (total goals solved across all workers, divided by wall time)
    at N=1 vs N=max_workers. This is the throughput a real sustained training run would actually see."""
    import multiprocessing as mp
    log("\n## 3. Sustained throughput: a long-running pool, not a short burst\n")
    log(f"Each worker runs continuously for {duration_s:.0f}s (not a fixed goal count), solving as "
       f"many goals as it can, local model per worker (no shared server -- see 2b vs 2d above for "
       f"why that's the right default). Spawn/import cost is paid ONCE per worker regardless of how "
       f"long it then runs, so a longer duration amortizes it the way a real training run naturally "
       f"would; this is deliberately the number the earlier short-burst benchmarks (2b/2d) were NOT "
       f"measuring, and is the more representative one for actual usage.")
    model_kwargs = dict(embed_dim=model.embed_dim, const_table_size=model.term_encoder.const_table_size,
                        max_var_level=model.term_encoder.max_var_level, max_type_level=model.term_encoder.max_type_level)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ctx = mp.get_context("spawn")
    counts = sorted(set([1, 2, 4, min(max_workers, 8), max_workers]))
    counts = [c for c in counts if 1 <= c <= max_workers]
    throughput = {}
    for n in counts:
        q = ctx.Queue()
        procs = [ctx.Process(target=_sustained_local_model_worker, args=(state, model_kwargs, sims, top_k, duration_s, q))
                for _ in range(n)]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        total_solved = sum(q.get() for _ in procs)
        for p in procs:
            p.join()
        wall = time.perf_counter() - t0
        throughput[n] = total_solved / wall
    base = throughput[1]
    log(f"{'workers':>8}{'goals/sec':>13}{'speedup':>10}")
    for n in counts:
        log(f"{n:>8}{throughput[n]:>12.2f}{throughput[n]/base:>9.2f}x")
    best_n = max(throughput, key=lambda n: throughput[n] / base)
    log(f"\nbest sustained speedup: {throughput[best_n]/base:.2f}x at {best_n} workers "
       f"(vs. {os.cpu_count()} cores available). This is the number that actually answers whether "
       f"the process pool is worth using for real training, not the short-burst numbers above.")
    return throughput


def bench_process_pool_warm(goal_names: list, model, sims: int, top_k: int, n_workers: int, rounds: int = 4):
    from process_driver import run_multiprocess
    log("\n## 2c. Process pool: repeated calls at a fixed worker count (approximating a warm pool)\n")
    log(f"The SAME {n_workers}-worker call repeated {rounds} times back to back: the first call's "
       f"time is dominated by spawn+import (see 2a); later calls still pay SOME spawn cost each "
       f"(this driver spawns fresh workers per call, by design -- see process_driver.py's own "
       f"header), but comparing call 1 to the median of the rest isolates roughly how much of the "
       f"total is one-time vs how much would remain even in a genuinely persistent pool.")
    model_kwargs = dict(embed_dim=model.embed_dim, const_table_size=model.term_encoder.const_table_size,
                        max_var_level=model.term_encoder.max_var_level, max_type_level=model.term_encoder.max_type_level)
    times = []
    for r in range(rounds):
        _, stats = run_multiprocess(goal_names, model, n_workers=n_workers, sims=sims, top_k=top_k, model_kwargs=model_kwargs)
        times.append(stats.wall_seconds)
        log(f"  call {r}: {stats.wall_seconds:.2f}s")
    if rounds > 1:
        log(f"\ncall 1 (cold): {times[0]:.2f}s   |   median of calls 2..{rounds}: {statistics.median(times[1:]):.2f}s"
           f"   |   ratio: {times[0]/statistics.median(times[1:]):.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", type=int, default=64, help="total goal-solves per data point (round-robin over the curriculum)")
    ap.add_argument("--large-goals", type=int, default=256, help="goal count for the large-workload process-pool point")
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--duration", type=float, default=20.0, help="seconds each worker runs for in the sustained-throughput test (section 3)")
    ap.add_argument("--skip-threads", action="store_true")
    ap.add_argument("--skip-process", action="store_true")
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)  # see _local_model_worker's own comment: keeps this benchmark's numbers
                              # comparable to 2d's (both single-threaded torch), isolating inter-
                              # process/inter-thread parallelism as the only variable being measured
    model = PolicyValueModel(embed_dim=32)

    all_names = [g.name for g in list(curriculum.CURRICULUM) + list(curriculum.HELD_OUT)]
    goal_names = (all_names * ((args.goals // len(all_names)) + 1))[: args.goals]
    large_goal_names = (all_names * ((args.large_goals // len(all_names)) + 1))[: args.large_goals]

    log(f"# Hardware scaling report\n")
    log(env_summary())
    log(f"work per data point: {args.goals} goal-solves (large: {args.large_goals}), sims={args.sims}, top_k={args.top_k}\n")

    max_workers = os.cpu_count() or 4

    if not args.skip_threads:
        bench_threads(goal_names, model, args.sims, args.top_k, max_workers)
    if not args.skip_process:
        bench_process_spawn_overhead()
        bench_process_pool(goal_names, model, args.sims, args.top_k, max_workers)
        bench_process_pool(large_goal_names, model, args.sims, args.top_k, max_workers)
        bench_process_pool_local_model(goal_names, model, args.sims, args.top_k, max_workers)
        bench_process_pool_local_model(large_goal_names, model, args.sims, args.top_k, max_workers)
        bench_process_pool_sustained(model, args.sims, args.top_k, max_workers, duration_s=args.duration)
        bench_process_pool_warm(goal_names, model, args.sims, args.top_k, max_workers)

    with open("hardware_scaling_report.md", "w") as f:
        f.write("\n".join(REPORT_LINES) + "\n")
    log("\nReport written to hardware_scaling_report.md")


if __name__ == "__main__":
    main()
