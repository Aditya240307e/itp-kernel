# src/python -- the Python layer

The kernel is C++ ([`src/c++`](../c++/hybrid/README.md)); everything around it is Python. The packages below are organised by *role*, and each README describes its own.

```
        problems  ◄── bank of formal problems, semantic checks, dataset
            │
   environment    ◄── goals as data, action language, probes, search, recorder
            │
         library  ◄── logic, N, binary N, Z, number theory, contest theorems
            │
       authoring  ◄── World, Term, Proof, tactics, ring / linarith / nlinarith
            │
   itp_env_lnl    ◄── the C++ kernel (nanobind module);  checker ◄── the independent second checker
```

| package | what | trusted? |
|---|---|---|
| [`authoring/`](authoring/README.md) | the layer you write mathematics in: worlds, terms, notation, tactics, decision procedures | no |
| [`library/`](library/README.md) | the standard library, built from scratch by tactic scripts: logic, unary and binary naturals, the integers as a quotient, number theory, finite sums and products | no |
| [`checker/`](checker/README.md) | an independently written type checker, the glue that hands a world to it, and the dependency auditor | no; a second opinion |
| [`environment/`](environment/README.md) | proving as a sequential decision problem without a text interface: engine, menus, probes with rollback, the action language, the typed term builder, search, the recorder | no |
| [`problems/`](problems/README.md) | the bank of 268 formal problems with their provenance and faithfulness checks, the splits, the dataset | no |
| `learner/`, `synth/`, `experiments/` | optional, see below | no |

**Only the kernel is trusted.** Everything in this directory drives it through its checked entry points, so a bug here can make a proof fail or a library fail to build, never make a false theorem appear. The argument and the evidence are in the [main README](../../README.md#why-believe-it).

## Importing

Every module is imported by its plain name, whichever subdirectory it lives in (`import gym_env`, `import nat_lib`), and module names are unique across the tree (gate S45 checks it). [`itp_paths.py`](itp_paths.py) makes that work: importing it puts the build directory (where the compiled kernels live) and each package directory on `sys.path`. Scripts in the repository begin with one line that does this; from your own code, either

```python
import sys; sys.path.insert(0, 'src/python'); import itp_paths          # from the repository root
```

or `PYTHONPATH=src/python python3 your_script.py` with `import itp_paths` first. `itp_paths` derives the repository root from its own location, never from the working directory, and `ITP_BUILD_DIR` selects another build directory.

## Building the kernel

```bash
python3 src/python/build_kernel.py            # both kernels into build/ (about 75 s)
python3 src/python/build_kernel.py hybrid     # only the kernel this project is about
```

`build_kernel.py` finds everything by introspection (nanobind's include directory, `sysconfig` for Python's headers and extension suffix) and needs a C++23 compiler: g++ 13 or later on Linux, a recent Apple clang (Xcode 16 or later) on macOS. `CXX` picks the compiler, `--force` rebuilds, `--out DIR` writes elsewhere; `python3 tools/preflight.py --quick` checks the toolchain of a new machine in seconds. Two kernels are built: the hybrid kernel, and a frozen earlier generation (`src/c++/reference`) that is the oracle of the lockstep gates.

## Optional packages

These are not needed to prove theorems, build libraries or use the bank. Nothing in the rest of this directory imports them.

* **`synth/`** is a generator of synthetic theorems: a reproducible process (a theorem is a pure function of a configuration and an index), sharded output that is merged into splits, and a verifier that replays every record in a fresh engine. `python3 src/python/synth/synth_cli.py {gen,merge,grow,verify}`; see [`docs/GENERATOR.md`](../../docs/GENERATOR.md).
* **`learner/`** is a machine-learning layer on top of the environment: the featuriser, a policy network, behaviour cloning, reinforcement-learning and search agents, and the Gymnasium wrapper `ITPBank-v0` ([`bank_env.py`](learner/bank_env.py)). It needs `torch` and `gymnasium`; every proof an agent reports is replayed and checked by the kernel.
* **`experiments/`** is the tooling around runs and evidence: the soundness folder (`make soundness`), the statement audit (`make audit`), and the experiment matrix with its reports; the shell drivers are in [`scripts/`](../../scripts/README.md).
