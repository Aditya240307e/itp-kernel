"""driver.py -- run many searches in lockstep so ONE policy call scores goals from ALL of them.

Each SearchTask is an independent search (its own root, dedup map, and -- for exactness -- its
own Kernel). The driver advances every unfinished task until it needs the policy, pools all the
pending GoalRequests into one batch (split by `max_batch_size` if set), makes one predict_batch
call per chunk, and hands each task back exactly the responses to ITS requests. Each individual
search therefore takes exactly the steps it would have taken alone: batching changes when the
policy is called and with how many goals at once, never what any search does.

Requires nothing of a policy beyond policy_api.BatchPolicy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import hypergraph as hg


@dataclass
class SearchTask:
    """One independent search. `kernel` should not be shared between tasks if you need the
    result to be identical to running each alone (tasks on separate kernels cannot interact)."""
    kernel: object
    root: hg.Node
    library: object
    dedup_map: dict
    max_simulations: int = 200
    top_k: int = 16
    dirichlet_epsilon: float = 0.0
    dirichlet_alpha: float = 0.3
    batch_frontier: bool = False
    status: Optional[hg.Status] = None       # set when the search finishes
    _gen: object = field(default=None, repr=False)

    def start(self):
        self._gen = hg.search_steps(self.kernel, self.root, self.library, self.dedup_map, self.max_simulations,
                                    self.top_k, self.dirichlet_epsilon, self.dirichlet_alpha, self.batch_frontier)


@dataclass
class DriverStats:
    rounds: int = 0                 # lockstep rounds (each advances every live task once)
    policy_calls: int = 0           # predict_batch invocations
    requests: int = 0               # goals scored in total
    batch_sizes: list = field(default_factory=list)
    policy_seconds: float = 0.0     # wall time inside predict_batch
    total_seconds: float = 0.0      # wall time of the whole run

    @property
    def mean_batch(self) -> float:
        return self.requests / self.policy_calls if self.policy_calls else 0.0

    @property
    def framework_seconds(self) -> float:
        """Everything that was NOT the policy: kernel work, search bookkeeping, driver overhead."""
        return self.total_seconds - self.policy_seconds


def run_lockstep(tasks: list, policy, max_batch_size: Optional[int] = None,
                 stats: Optional[DriverStats] = None) -> list:
    """Runs every task to completion, batching policy calls across them. Returns each task's
    final Status (also stored on task.status). Pass a DriverStats to collect batching metrics."""
    st = stats if stats is not None else DriverStats()
    t_start = time.perf_counter()
    for t in tasks:
        t.start()
    pending = {}          # task index -> list of GoalRequests it is waiting on
    live = list(range(len(tasks)))
    first = True
    inbox = {}            # task index -> responses to deliver on the next advance
    while live:
        # 1. advance every live task to its next policy request (or to completion)
        still = []
        for i in live:
            task = tasks[i]
            try:
                pending[i] = next(task._gen) if first else task._gen.send(inbox.pop(i))
                still.append(i)
            except StopIteration as stop:
                task.status = stop.value
                pending.pop(i, None)
        first = False
        live = still
        if not live:
            break
        st.rounds += 1
        # 2. pool all pending requests, score them in (chunked) batches
        flat, owners = [], []
        for i in live:
            for r in pending[i]:
                flat.append(r); owners.append(i)
        responses = []
        step = max_batch_size or len(flat)
        for a in range(0, len(flat), step):
            chunk = flat[a:a + step]
            t0 = time.perf_counter()
            out = policy.predict_batch(chunk)
            st.policy_seconds += time.perf_counter() - t0
            if len(out) != len(chunk):
                raise ValueError(f"policy returned {len(out)} responses for {len(chunk)} requests")
            responses.extend(out)
            st.policy_calls += 1
            st.requests += len(chunk)
            st.batch_sizes.append(len(chunk))
        # 3. hand each task the responses to its own requests, in order
        cursor = 0
        for i in live:
            n = len(pending[i])
            inbox[i] = responses[cursor:cursor + n]
            cursor += n
    st.total_seconds = time.perf_counter() - t_start
    return [t.status for t in tasks]
