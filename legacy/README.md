# legacy/ -- the previous generation

Everything here predates the hybrid kernel's layout and is kept because something still depends on it or because it documents how the project got here. Nothing in `src/` imports it, except through one door:
`tests/support/harness.py` appends this directory to `sys.path` and imports `curriculum` and `hypergraph`, the previous generation's RL modules, to drive the *differential* gates (the same random tactic scripts run on
the frozen reference kernel and on the hybrid kernel).

| | |
|---|---|
| `curriculum.py`, `hypergraph.py`, `model.py`, `driver.py`, `policy_api.py`, `process_driver.py`, `binary_nat.py` | the previous generation's environment, curriculum and learner (they drive the reference kernel through the module name `itp_env`) |
| `test_*.py`, `test_compact_since.cpp` | its tests; run from this directory, after building the reference kernel under the name `itp_env`: `python3 ../src/python/build_kernel.py reference --module itp_env --out .` (the C++ test needs `-I ../src/c++/reference`) |
| `bench_*.py`, `run_hardware_suite.py`, `equivalence_snapshot.py`, `build_kernel.py` | benchmarks and the previous build script (superseded by `src/python/build_kernel.py`; the old name is kept here because `run_hardware_suite.py` imports it from this directory) |
| `*.json` | baselines and snapshots those scripts compare against |
| `notes/`, `patches/` | design notes of earlier stages and the patches that were applied between them |

Module names in this directory are NOT part of the flat namespace of `src/python` and `tests` (S45 checks that those are unique); `build_kernel.py` exists in both places by design, and a script run from here finds its own sibling first.
