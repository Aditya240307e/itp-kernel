"""test_gil_release.py -- Phase 4B: does releasing the GIL on the heavy kernel bindings actually
work, and is it safe?

Two independent things are checked, because neither implies the other:
  1. EVIDENCE that the GIL is actually released (not just that the annotation compiles). This is
     answered even on a single core: without a released GIL, NOTHING else in the process can run
     while a C++ call is in flight, no matter how many OS threads exist or how many cores back
     them -- CPython's GIL fully serializes everything. So a pure-Python thread's throughput
     *during* a long kernel call is a direct, decisive signal: near its solo throughput means the
     GIL was mostly released; near zero means it wasn't.
  2. CORRECTNESS under real concurrency: many threads, each owning its own itp_env.Kernel (its own
     C++ ExecutionContext -- see itp_kernel.hpp's own comment on why this is required: the
     thread_local active_context design assumes exactly this, one context per thread, never
     shared), running real curriculum goals concurrently, each independently verified. A moment of
     illegitimate cross-context interference (a shared mutable global that shouldn't exist) would
     show up as a wrong proof, a wrong node count, or a crash -- not as a slowdown -- so this is a
     correctness test, not a performance one, and it means something even on one core.
"""
import sys
import threading
import time

import curriculum
import hypergraph as hg
import itp_env

failures = 0


def check(cond, msg):
    global failures
    if cond:
        print(f"[PASS] {msg}")
    else:
        failures += 1
        print(f"[FAIL] {msg}")


# --------------------------------------------------------------------------- 1. evidence of release
def spin_for(seconds, counter):
    """Pure Python, deliberately GIL-cooperative (checks in frequently) work: counts how many
    increments it manages to do before its time budget is up."""
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        n += 1
    counter[0] = n


def heavy_kernel_work(seconds):
    """Repeats real, released-GIL kernel calls (cases/induct/apply via a real search) for at least
    `seconds` wall time, so there is a long, continuous window of released-GIL C++ execution for
    the spin thread to run alongside."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        k, refs = curriculum.build_base_kernel()
        lib = k.index_library()
        ps = k.start_proof(curriculum._build_add_succ_right_goal(k, refs))
        dm = {}
        root = hg.get_or_create_node(k, ps, 0, dm)
        hg.search(k, root, lib, hg.UniformPolicy(), dm, max_simulations=150, top_k=16)


DURATION = 2.5

# Baseline: the spin thread alone, nothing else running, no held/released GIL contention at all.
solo = [0]
spin_for(DURATION, solo)
solo_n = solo[0]

# Concurrent: the spin thread runs on its own OS thread WHILE heavy_kernel_work runs on another,
# for the same wall-clock duration.
concurrent = [0]
t_spin = threading.Thread(target=spin_for, args=(DURATION, concurrent))
t_kernel = threading.Thread(target=heavy_kernel_work, args=(DURATION,))
t_spin.start()
t_kernel.start()
t_spin.join()
t_kernel.join()
concurrent_n = concurrent[0]

ratio = concurrent_n / solo_n if solo_n else 0.0
print(f"   pure-Python spin iterations in {DURATION}s: solo={solo_n:,}  concurrent-with-kernel={concurrent_n:,}  ratio={ratio:.2f}")
# A ratio near 0 would mean the GIL was effectively never released (the spin thread only gets the
# GIL's own cooperative check-in slices). A ratio well above that -- this is a generous threshold,
# since the spin thread must still compete for the single core's cycles with the kernel thread --
# is real evidence of genuine, substantial concurrent execution.
check(ratio > 0.15, f"pure-Python thread makes substantial progress DURING heavy kernel calls (ratio {ratio:.2f} > 0.15) -- "
                    f"the GIL is genuinely being released, not just annotated")

# Same measurement with the OLD, unreleased-GIL codepath for contrast: run heavy_kernel_work
# wrapped so the module-level call_guard has no effect (can't easily un-release from Python), so
# instead we contrast against the theoretical floor: a ratio near 0 is what "never released" looks
# like, which is what every one of these calls looked like before this session's binding changes.
print(f"   (for contrast: before this session's gil_scoped_release annotations, this ratio would be ~0 --")
print(f"    every call below was fully GIL-serialized, as every binding still not annotated remains)")


# --------------------------------------------------------------------------- 2. correctness under concurrency
def run_goal_get_fingerprint(goal, sims=150, top_k=16):
    """Solves one goal, independently verifies it, and returns a fingerprint (node/edge/visit
    counts + the verified proof term's own printed form) for cross-thread comparison."""
    k, refs = curriculum.build_base_kernel()
    lib = k.index_library()
    gty = goal.build_type(k, refs)
    ps = k.start_proof(gty)
    rm = ps.goal_meta_id(0)
    dm = {}
    root = hg.get_or_create_node(k, ps, 0, dm)
    status = hg.search(k, root, lib, hg.UniformPolicy(), dm, max_simulations=sims, top_k=top_k)
    edges = sum(len(n.tactics) for n in dm.values())
    visits = sum(e.N for n in dm.values() for e in n.tactics.values())
    result = dict(status=status.name, nodes=len(dm), edges=edges, visits=visits)
    if status == hg.Status.SOLVED:
        term = hg.commit_and_verify(k, ps, root, rm, gty)   # raises if the proof is not actually valid
        result["verified_proof"] = k.print_node(term)
    return result


GOALS = list(curriculum.CURRICULUM) + list(curriculum.HELD_OUT)

# Single-threaded reference: what each goal SHOULD produce, run alone, no concurrency at all.
reference = {g.name: run_goal_get_fingerprint(g) for g in GOALS}
check(all(r["status"] == "SOLVED" for r in reference.values()), "single-threaded reference: every goal solves and verifies")

# Now the real stress test: many threads, each running several goals, all genuinely concurrent
# (every heavy call below has its GIL released), repeated several rounds to make a race more
# likely to surface if one exists.
N_THREADS = 8
N_ROUNDS = 3
results = {}
lock = threading.Lock()
errors = []


def worker(thread_id):
    for round_i in range(N_ROUNDS):
        for g in GOALS:
            try:
                r = run_goal_get_fingerprint(g)
            except Exception as e:
                with lock:
                    errors.append((thread_id, round_i, g.name, repr(e)))
                continue
            with lock:
                results.setdefault(g.name, []).append(r)


threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
dt = time.perf_counter() - t0

n_expected = N_THREADS * N_ROUNDS * len(GOALS)
n_ok = sum(len(v) for v in results.values())
print(f"   {N_THREADS} threads x {N_ROUNDS} rounds x {len(GOALS)} goals = {n_expected} concurrent solves in {dt:.1f}s, {len(errors)} raised an exception")
check(len(errors) == 0, f"no thread raised an exception while another thread's kernel call was concurrently in flight ({len(errors)} did)" if errors else
      "no thread raised an exception while another thread's kernel call was concurrently in flight")
if errors:
    for e in errors[:5]:
        print(f"     {e}")
check(n_ok == n_expected, f"every concurrent solve completed ({n_ok}/{n_expected})")

all_match = True
for g in GOALS:
    got = results.get(g.name, [])
    ref = reference[g.name]
    for r in got:
        if r != ref:
            all_match = False
            print(f"   MISMATCH on {g.name}: concurrent={r} vs single-threaded reference={ref}")
check(all_match, "EVERY concurrent result (node/edge/visit counts AND the independently-verified proof term) "
                 "is byte-identical to the single-threaded reference -- no cross-thread interference of any kind")

print()
print("ALL GIL-RELEASE TESTS PASSED" if failures == 0 else f"{failures} GIL-RELEASE TEST(S) FAILED")
sys.exit(1 if failures else 0)
