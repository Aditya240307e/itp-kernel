#!/usr/bin/env python3
"""run_hardware_suite.py -- the final step: build, verify correctness, then measure real scaling.

Runs, in order, stopping at the first failure (correctness must pass before any performance number
is trusted):
  1. Environment summary (platform, core count, Python/torch versions).
  2. Portable build (build_kernel.py) -- no hardcoded paths; see that file's own header for why.
  3. The FULL correctness regression suite: the 15-file baseline suite, Phase 2/3/process-driver test
     suites, both C++ pool fuzzers (16.5M + 19,860 checks), binary_nat's exhaustive tests, and the
     GIL-release concurrency-correctness test -- everything this whole project has accumulated.
     A behavioral-snapshot check confirms the port to this machine is bit-for-bit identical to the
     reference recorded during development, not just "the tests happen to also pass here."
  4. bench_hardware_scaling.py -- the actual point of this script: real multi-core scaling numbers
     for GIL-released threads and the process pool, which the single-vCPU development sandbox could
     only ever prove correct, never measure.

Writes a full report to RUN_REPORT.md alongside stdout. Safe to re-run; nothing here is destructive.

Usage: python3 run_hardware_suite.py [--skip-build] [--quick] [--goals N] [--sims N]
  --skip-build   reuse an already-built itp_env extension (skip step 2)
  --quick        smaller workload for a fast sanity run before committing to the full one
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT: list[str] = []


def log(line: str = ""):
    print(line, flush=True)
    REPORT.append(line)


def section(title: str):
    log(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run(cmd: list[str], timeout: int, cwd: Path = HERE) -> tuple[bool, str]:
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        ok = r.returncode == 0
        out = r.stdout + r.stderr
        return ok, f"({dt:.1f}s, rc={r.returncode})\n" + out
    except subprocess.TimeoutExpired:
        dt = time.perf_counter() - t0
        return False, f"({dt:.1f}s) TIMED OUT after {timeout}s"


PY = sys.executable

REGRESSION_PY_FILES = [
    "test_commit_and_verify.py", "test_const_hashing.py", "test_dag_dump_match.py",
    "test_dirichlet_noise.py", "test_extract_training_examples.py", "test_guardedness_reverify.py",
    "test_hypergraph_discovery.py", "test_hypergraph_search.py", "test_library_corruption.py",
    "test_library_persistence.py", "test_observation_roots.py", "test_term_encoder_sanity.py",
    "test_term_encoder_sharing.py", "test_train_step.py", "test_trainable_policy_plumbing.py",
]
SUITE_SCRIPTS = [
    "test_phase2_batching.py", "test_phase3_vectorized.py", "test_process_driver.py",
    "test_gil_release.py", "test_binary_nat.py",
]
CPP_FUZZERS = ["test_nodepool_rewind_fuzz.cpp", "test_compact_since.cpp"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--quick", action="store_true", help="small workload: fast sanity run, not real scaling numbers")
    ap.add_argument("--goals", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None)
    args = ap.parse_args()

    all_ok = True

    section("0. Environment")
    import os
    try:
        import torch
        torch_v = torch.__version__
    except ImportError:
        torch_v = "NOT INSTALLED"
    log(f"platform:  {platform.system()} {platform.machine()} ({platform.platform()})")
    log(f"cpu_count: {os.cpu_count()}")
    log(f"python:    {platform.python_version()} ({PY})")
    log(f"torch:     {torch_v}")
    if torch_v == "NOT INSTALLED":
        log("\nERROR: torch is required. Install with: pip install torch")
        return _finish(False)
    try:
        import nanobind
        log(f"nanobind:  {nanobind.include_dir()}")
    except ImportError:
        log("\nERROR: nanobind is required. Install with: pip install nanobind")
        return _finish(False)

    if not args.skip_build:
        section("1. Building itp_env (portable build_kernel.py)")
        ok, out = run([PY, "build_kernel.py"], timeout=300)
        log(out)
        if not ok:
            log("\nBUILD FAILED. Nothing further will run -- fix the build first (see the error "
               "above; build_kernel.py's own error messages are meant to point at the likely cause,"
               " e.g. a missing/too-old compiler).")
            return _finish(False)
        all_ok &= ok
    else:
        section("1. Build skipped (--skip-build)")
        so_files = list(HERE.glob("itp_env*.so")) + list(HERE.glob("itp_env*.pyd"))
        if not so_files:
            log("ERROR: --skip-build given but no itp_env extension found. Run without --skip-build first.")
            return _finish(False)
        log(f"using existing: {so_files[0].name}")

    section("2. Correctness regression: 15-file baseline suite")
    for f in REGRESSION_PY_FILES:
        ok, out = run([PY, f], timeout=120)
        n_fail = out.count("[FAIL]")
        status = "PASS" if ok and n_fail == 0 else "FAIL"
        log(f"  [{status}] {f}  {out.splitlines()[0] if out.splitlines() else ''}")
        if status == "FAIL":
            log("    " + "\n    ".join(out.splitlines()[-15:]))
        all_ok &= (ok and n_fail == 0)

    section("3. Correctness regression: phase test suites")
    for f in SUITE_SCRIPTS:
        timeout = 600 if f == "test_gil_release.py" else 300
        ok, out = run([PY, f], timeout=timeout)
        tail = out.splitlines()[-3:] if out.splitlines() else []
        status = "PASS" if ok else "FAIL"
        log(f"  [{status}] {f}")
        log("    " + "\n    ".join(tail))
        if not ok:
            log("    " + "\n    ".join(out.splitlines()[-20:]))
        all_ok &= ok

    section("4. Correctness regression: C++ pool fuzzers")
    sys.path.insert(0, str(HERE))
    from build_kernel import find_compiler, pick_std_flag
    cxx = find_compiler()
    std_flag = pick_std_flag(cxx)
    log(f"using {cxx} {std_flag} for the C++ fuzzers (same detection build_kernel.py uses)")
    for f in CPP_FUZZERS:
        bin_path = HERE / f"_{f.replace('.cpp', '')}_bin"
        cxx_ok, cxx_out = run([cxx, std_flag, "-O1", "-I.", f, "-o", str(bin_path)], timeout=120)
        if not cxx_ok:
            log(f"  [FAIL] {f} (compile)\n    " + "\n    ".join(cxx_out.splitlines()[-15:]))
            all_ok = False
            continue
        ok, out = run([str(bin_path)], timeout=300)
        tail = out.splitlines()[-1:] if out.splitlines() else []
        log(f"  [{'PASS' if ok else 'FAIL'}] {f}  " + " ".join(tail))
        if not ok:
            log("    " + "\n    ".join(out.splitlines()[-15:]))
        all_ok &= ok
        bin_path.unlink(missing_ok=True)

    section("5. Behavioral snapshot: this machine vs. the recorded development-time baseline")
    ok, out = run([PY, "equivalence_snapshot.py", "_this_machine_snapshot.json"], timeout=180)
    log(out)
    baseline = HERE / "baseline_snapshot.json"
    this_machine = HERE / "_this_machine_snapshot.json"
    if baseline.exists() and this_machine.exists():
        same = baseline.read_text() == this_machine.read_text()
        log(f"  [{'PASS' if same else 'FAIL'}] identical to baseline_snapshot.json: {same}")
        all_ok &= same
    else:
        log("  [SKIP] baseline_snapshot.json not present alongside this script -- nothing to compare against "
           "(the snapshot just produced is still saved to _this_machine_snapshot.json).")

    if not all_ok:
        section("STOPPING: correctness regression failed")
        log("Not running performance benchmarks -- a performance number from code that doesn't "
           "reproduce the verified reference behavior on this machine isn't trustworthy. Fix the "
           "failure(s) above (a compiler/library version difference is the most likely cause on a "
           "machine this project has never run on before) and re-run.")
        return _finish(False)

    section("6. Hardware scaling benchmarks (the actual point of this script)")
    bench_args = [PY, "bench_hardware_scaling.py"]
    if args.quick:
        bench_args += ["--goals", "8", "--large-goals", "16", "--sims", "30", "--top-k", "8", "--duration", "5"]
    else:
        if args.goals:
            bench_args += ["--goals", str(args.goals)]
        if args.sims:
            bench_args += ["--sims", str(args.sims)]
    ok, out = run(bench_args, timeout=3600)
    log(out)
    all_ok &= ok

    return _finish(all_ok)


def _finish(ok: bool) -> int:
    (HERE / "RUN_REPORT.md").write_text("\n".join(REPORT) + "\n")
    print(f"\nFull report written to {HERE / 'RUN_REPORT.md'}")
    print("\n" + ("ALL CORRECTNESS CHECKS PASSED" if ok else "SOME CHECKS FAILED -- see above / RUN_REPORT.md"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
