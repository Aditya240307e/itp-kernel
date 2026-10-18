"""process_driver.py -- Phase 4C: process-based scale-out.

Each worker process owns its own itp_env.Kernel (its own C++ ExecutionContext) and drives its own
slice of SearchTasks in local lockstep (see driver.run_lockstep) exactly as a single process would.
The one thing that crosses a process boundary is policy scoring: instead of calling a local model,
each worker's RemotePolicy pickles its pending GoalRequests (via GoalRequest.to_portable(), numpy
arrays and ints only) onto a shared queue; a single dedicated server process holds ONE copy of the
model, pools whatever requests are currently waiting from ALL workers into one batched
model.forward_batch call, and routes each response back to the worker that asked for it.

This exists because Phase 4B found that Python threads do not help this workload: releasing the GIL
around the C++ tactic calls is correct (proven: test_gil_release.py) and cheap in isolation
(~100ns/call, flat under contention -- see gil_overhead.py's own numbers), but 60-70% of a search's
wall time is Python bookkeeping that the GIL still fully serializes across threads regardless, and 8
threads measured 8x SLOWER than one thread on this session's 1-core sandbox. Separate OS processes
sidestep that entirely: each has its own interpreter and its own GIL, so Python-bound work in one
process never contends with Python-bound work in another.

What this module answers, and what it cannot: everything here is testable for CORRECTNESS on a
single core (do N worker processes, run concurrently, produce byte-identical results to a single
process's sequential/lockstep run?). Whether it is actually FASTER needs real multiple cores --
this sandbox has one vCPU, so no speedup number from here should be trusted; see
bench_process_driver.py's own header for what it can and cannot tell you.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from typing import Optional

import policy_api as pa

# 'spawn' rather than the POSIX default 'fork': a forked child inherits the WHOLE parent address
# space, including whatever itp_env.ExecutionContext the parent's thread_local active_context
# happened to point at when fork() was called -- fine as long as the child only ever builds and
# uses brand-new Kernel() objects afterward (a fresh Kernel's own activate() overwrites that
# inherited pointer before doing anything else), but 'spawn' sidesteps the question entirely by
# giving every worker a genuinely fresh interpreter with nothing inherited, which is worth the
# slower startup for a correctness-first spike.
_CTX = mp.get_context("spawn")


class RemotePolicy:
    """A policy_api.BatchPolicy that sends requests to the shared server process over a queue
    instead of calling a local model. One instance per worker process; the server pools whatever
    is currently pending from every worker's RemotePolicy into one batched call."""

    def __init__(self, worker_id: int, req_queue, resp_queue):
        self.worker_id = worker_id
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self._next_id = 0

    def predict_batch(self, requests):
        if not requests:
            return []
        req_id = self._next_id
        self._next_id += 1
        portables = [r.to_portable() for r in requests]
        self.req_queue.put((self.worker_id, req_id, portables))
        while True:
            got_id, responses = self.resp_queue.get()
            if got_id == req_id:
                return responses
            # Cannot happen with the current one-outstanding-request-per-worker protocol (a worker
            # always waits for its own response before issuing another), but a stray/duplicate
            # message should not silently corrupt a different request's results.
            raise RuntimeError(f"RemotePolicy worker {self.worker_id}: got response for req {got_id}, expected {req_id}")


def _worker_main(worker_id: int, goal_names: list, sims: int, top_k: int, req_queue, resp_queue, results_queue):
    """Runs in a child process. Builds its OWN kernel per goal (never touches anything from before
    the spawn), drives its assigned goals in local lockstep, and reports one fingerprint per goal
    plus a final DONE sentinel."""
    import curriculum
    import hypergraph as hg
    from driver import SearchTask, run_lockstep

    all_goals = {g.name: g for g in list(curriculum.CURRICULUM) + list(curriculum.HELD_OUT)}
    named_tasks = []
    for name in goal_names:
        g = all_goals[name]
        k, refs = curriculum.build_base_kernel()
        lib = k.index_library()
        ps = k.start_proof(g.build_type(k, refs))
        dm = {}
        root = hg.get_or_create_node(k, ps, 0, dm)
        named_tasks.append((name, ps, k, root, dm,
                            SearchTask(kernel=k, root=root, library=lib, dedup_map=dm,
                                       max_simulations=sims, top_k=top_k)))

    policy = RemotePolicy(worker_id, req_queue, resp_queue)
    statuses = run_lockstep([t for _, _, _, _, _, t in named_tasks], policy)

    for (name, ps, k, root, dm, t), status in zip(named_tasks, statuses):
        edges = sum(len(n.tactics) for n in dm.values())
        visits = sum(e.N for n in dm.values() for e in n.tactics.values())
        result = dict(status=status.name, nodes=len(dm), edges=edges, visits=visits)
        results_queue.put((worker_id, name, result))
    results_queue.put((worker_id, None, "DONE"))


def _server_main(n_workers: int, req_queue, resp_queues: dict, model_class, model_kwargs: dict,
                 state_dict, stop_event, batch_window_s: float, stats_queue):
    """Runs in its own process, holds ONE copy of the model, and pools pending requests from every
    worker into batched model.forward_batch calls until told to stop.

    Stopping protocol: the main process sets `stop_event` only after it has received a DONE
    message (via results_queue) from every worker -- and a worker only sends DONE after its
    search_steps generator has fully returned, which can only happen after it has already
    synchronously received every response it was waiting on. So by the time stop_event is set,
    no worker will ever enqueue another request; this loop just drains anything still sitting in
    req_queue (there normally is none) and exits.

    Also caps its own torch thread count to 1 (see bench_hardware_scaling.py's _local_model_worker
    for the fuller reasoning): a single small forward_batch call spinning up torch's default
    all-core BLAS/OMP pool adds per-call thread-spawn overhead that compounds badly under this
    server's high-frequency, small-batch request pattern."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    model = model_class(**model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()

    n_calls = 0
    batch_sizes = []
    with torch.no_grad():
        while not stop_event.is_set() or not req_queue.empty():
            items = []
            try:
                items.append(req_queue.get(timeout=0.5))
            except queue.Empty:
                continue
            deadline = time.perf_counter() + batch_window_s
            while time.perf_counter() < deadline:
                try:
                    items.append(req_queue.get_nowait())
                except queue.Empty:
                    break

            flat_portables, route = [], []   # route[i] = (worker_id, req_id) owning flat_portables[i]
            for worker_id, req_id, portables in items:
                for p in portables:
                    flat_portables.append(p)
                    route.append((worker_id, req_id))

            if not flat_portables:
                continue

            batch, ab = pa.collate_portable(flat_portables)
            priors, values = model.forward_batch(batch, ab)
            n_calls += 1
            batch_sizes.append(len(flat_portables))

            # Slice priors/values back into one response list PER (worker_id, req_id), in the
            # order that worker's own request list had them (== the order they appear in `route`,
            # since collate_portable/collate_actions preserve input order and ab.offsets are
            # aligned to that same flat order).
            grouped: dict = {}
            for i, (worker_id, req_id) in enumerate(route):
                lo, hi = int(ab.offsets[i]), int(ab.offsets[i + 1])
                grouped.setdefault((worker_id, req_id), []).append((priors[lo:hi].numpy(), float(values[i].item())))
            for (worker_id, req_id), responses in grouped.items():
                resp_queues[worker_id].put((req_id, responses))

    if stats_queue is not None:
        stats_queue.put(dict(server_calls=n_calls, server_batch_sizes=batch_sizes))


@dataclass
class ProcessDriverStats:
    n_workers: int = 0
    wall_seconds: float = 0.0
    server_calls: int = 0
    server_batch_sizes: list = field(default_factory=list)

    @property
    def mean_server_batch(self) -> float:
        return sum(self.server_batch_sizes) / len(self.server_batch_sizes) if self.server_batch_sizes else 0.0


def run_multiprocess(goal_names: list, model, n_workers: int, sims: int = 150, top_k: int = 16,
                     batch_window_s: float = 0.003, model_kwargs: Optional[dict] = None) -> tuple[dict, ProcessDriverStats]:
    """Runs `goal_names` (round-robin split across `n_workers` worker processes) against `model`,
    scored by a single shared server process. Returns ({goal_name: fingerprint}, stats).

    model_kwargs must reconstruct an equivalent model via type(model)(**model_kwargs) in the
    server's own process (a state_dict crosses process boundaries fine; the live nn.Module object
    itself does not need to, and reconstructing avoids ever pickling the model object graph)."""
    from model import PolicyValueModel

    if model_kwargs is None:
        te = model.term_encoder
        model_kwargs = dict(embed_dim=model.embed_dim, const_table_size=te.const_table_size,
                            max_var_level=te.max_var_level, max_type_level=te.max_type_level)
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    req_queue = _CTX.Queue()
    resp_queues = {i: _CTX.Queue() for i in range(n_workers)}
    results_queue = _CTX.Queue()
    stats_queue = _CTX.Queue()
    stop_event = _CTX.Event()

    buckets = [[] for _ in range(n_workers)]
    for i, name in enumerate(goal_names):
        buckets[i % n_workers].append(name)

    server = _CTX.Process(target=_server_main, args=(n_workers, req_queue, resp_queues, type(model),
                                                      model_kwargs, state_dict, stop_event, batch_window_s, stats_queue))
    workers = [_CTX.Process(target=_worker_main, args=(i, buckets[i], sims, top_k, req_queue, resp_queues[i], results_queue))
              for i in range(n_workers) if buckets[i]]

    t0 = time.perf_counter()
    server.start()
    for w in workers:
        w.start()

    results = {}
    done = 0
    n_live_workers = len(workers)
    while done < n_live_workers:
        worker_id, name, payload = results_queue.get()
        if name is None:
            done += 1
        else:
            results[name] = payload
    stop_event.set()

    for w in workers:
        w.join()
    server.join(timeout=5)
    if server.is_alive():
        server.terminate()
    wall = time.perf_counter() - t0

    stats = ProcessDriverStats(n_workers=n_live_workers, wall_seconds=wall)
    try:
        s = stats_queue.get(timeout=2)
        stats.server_calls = s["server_calls"]
        stats.server_batch_sizes = s["server_batch_sizes"]
    except queue.Empty:
        pass
    return results, stats
