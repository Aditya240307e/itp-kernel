# ITP -- a trustworthy interactive theorem prover, and a gym for learning olympiad mathematics

A small, heavily tested **trusted kernel** in C++23, an **untrusted** Python layer on top of it that does all the convenient things (tactics, decision procedures, libraries from Peano arithmetic to elementary
number theory, a problem bank of formal olympiad problems), and a **gym environment** with no text interface in which a learner proves them. The point of the split is that nothing above the kernel has to be
believed: a tactic, a decision procedure, a search or a neural network can be wrong in any way it likes, and a finished proof is still checked by the kernel (and, independently, by a second checker written from the
typing rules in Python), so a wrong tool wastes time and cannot make a false statement provable.

Targets: IMO-level algebra and number theory first (the problem bank is the beginning of that dataset), then undergraduate mathematics.

| | |
|---|---|
| Kernel | `src/c++/hybrid`: dependent types, inductive families (parameters, indices), structural recursion, a Prop/Type world with definitional proof irrelevance, quotients, a deterministic work budget; 7.5 k lines of C++23 plus 0.8 k of Python bindings |
| Reference kernel | `src/c++/reference`: the frozen previous generation, byte-identical (checksums), kept as the oracle of the lockstep gates |
| Checkers | the kernel; `indep_checker.py`, a second implementation of the typing rules written separately; both must accept a proof |
| Libraries | unary and binary naturals, the integers as a quotient of `N x N`, ordered ring laws, number theory (divisibility, Bezout, gcd, primes); built by tactic scripts, no axioms (four classical axioms are an optional layer) |
| Decision procedures | `ring` (naturals, binary naturals, integers), `linarith`, `nlinarith`, `linear_combination`, `interval_cases` -- each produces a certificate the kernel checks |
| Problem bank | 82 formal problems (9 IMO), levels 1-5, each with its text, a checked-faithful formal statement, and kernel-checked reference proofs as sequences of actions; dataset v1 in `data/dataset_v1`, with a label for every action (direction, depth, size of the subtree) and the 1 800-odd theorems that the reference proofs contain, each with its own proof |
| Environment | `gym_env` / `gym_acts`: goals as data, menus of legal actions, actions that *say* their term arguments, atomic probes; `gym_tree`: the proof tree of an episode and the theorems it proved (hindsight); `learner/bank_env.py` is a Gymnasium environment (`ITPBank-v0`) that starts on any goal given as data, has a step budget and reports what each action achieved |
| Learner | features, a transformer + GRU policy network over *options*, behaviour cloning, policy-guided IDA\* search, REINFORCE / filtered-BC loops. A baseline, not a result: see `docs/TRAINING.md` |
| Tests | 47 kernel / tactic / environment / bank gates, the learner's gate S44, the repository gates S45-S46; 1615 mutants of the code to prove the gates bite |

## Quick start

Requirements: a C++23 compiler (g++ 13 or recent clang), Python 3.13 with `pip install -r requirements.txt` (numpy, scipy, nanobind; torch and gymnasium only for the learner). Developed and tested on Linux; the build script also has the macOS flags (untested here).

```
make kernel                              # builds build/itp_env_lnl*.so (the kernel) and build/itp_env_ref*.so (the oracle)   [python3 src/python/build_kernel.py]
python3 examples/01_prove_with_tactics.py
python3 tests/run_gates.py -j 2          # the 47 kernel gates, about 13 minutes on two cores
```

Every script finds the repository through its own location (`src/python/itp_paths.py`), so it works from any directory; `ITP_BUILD_DIR` selects another build directory.

## Examples (`examples/`, each ends with `EXAMPLE OK`; the gate S46 runs them)

| | |
|---|---|
| `01_prove_with_tactics.py` | a recursive function, induction, `ring`; a false identity refused with both normal forms; a forged proof refused by the kernel; the independent re-check and the axiom audit |
| `02_ring_linarith.py` | the integers: `ring`, `linarith`, `nlinarith` (with a hint), a refused claim |
| `03_bank_problem.py` | an IMO problem of the bank: statement, faithfulness check, recorded proof replayed, second opinion |
| `04_environment_episode.py` | one proof attempt as a game: observation, menu, atomic steps, a refused action |
| `05_blind_search.py` | the blind search baseline and what it cannot do |
| `06_gymnasium_env.py` | the Gymnasium environment: a recorded proof played decision by decision |
| `07_read_dataset.py` | the dataset on disk: shape, splits, digests |
| `08_hindsight.py` | an attempt that stops half-way: what each action achieved, the theorems it proved, one of them proved again from its data and judged by the independent checker |

## Layout

```
src/c++/hybrid/            the kernel (itp_kernel.hpp, nanobind bindings itp_bindings.cpp)
src/c++/reference/         the frozen reference kernel (SHA256SUMS)
src/python/itp_paths.py    where everything lives; every script's first line imports it
src/python/build_kernel.py builds either kernel
src/python/authoring/      the untrusted authoring layer: World, tactics, ring, linarith, linear_combination, injectivity lemmas
src/python/checker/        the independent checker, recheck (a second opinion on a whole world), audit (what a theorem rests on)
src/python/library/        the libraries, built by tactic scripts: naturals, binary naturals, integers, number theory, IMO statements
src/python/environment/    the gym: engine, action language, typed term builder, policy assembler, search, recorder
src/python/problems/       the problem bank, faithfulness checks, splits, the dataset builder
src/python/learner/        features, network, training, guided search, RL, the Gymnasium environment
tests/                     the gates by role (kernel, library, environment, problems, learner, repo), shared modules (support), fuzzers, benchmarks, golden files
mutants/                   the mutation catalogue (1615 mutants) and its driver
examples/                  runnable examples; examples/repros holds scripts that reproduce fixed defects
data/dataset_v1/           the dataset (JSON + card + manifest);  data/results/ the result files of the baseline runs
docs/                      FINDINGS (the tracker), ROADMAP, KERNEL_DESIGN, ACTIONS, BANK, ARCHITECTURE_RL, TRAINING
legacy/                    the previous generation's modules, tests and notes (some gates import curriculum / hypergraph from here)
tools/                     sanitizer run, release archive
```

Modules are imported by their plain names (`import gym_env`); the names are unique across the tree (S45 checks it).

## Where to read next

* `docs/KERNEL_DESIGN.md` -- the kernel: terms, evaluation, conversion, inductive types, the options and their reasons.
* `docs/FINDINGS.md` -- every defect and design decision found so far, with the test that pins it.
* `docs/BANK.md`, `docs/ACTIONS.md` -- the problem bank (what makes a formal statement *the* problem) and the action language.
* `docs/ARCHITECTURE_RL.md` -- how a learner should be structured to climb from imitation to a manager/worker design; `docs/TRAINING.md` -- what exists and what it achieves (little, honestly).
* `docs/ROADMAP.md` -- what is missing for IMO-level and undergraduate mathematics, and a calibrated plan.
* `tests/README.md` (the gates), `mutants/README.md` (mutation testing), `legacy/README.md`.

## Trust

What must be believed for "this proof is correct": the C++ kernel (`src/c++/hybrid`, about 8 k lines with its bindings), the C++ compiler and the machine -- and, as a cross-check that does not share code with the kernel, the independent
checker (`src/python/checker/indep_checker.py`). `audit.dependencies` lists the axioms a theorem uses (the libraries use none; the integers use the kernel's five quotient constants). Everything else -- tactics,
libraries' scripts, the bank, the search, the networks -- is outside the trust boundary by construction, and the gates and mutants exist to make that boundary real rather than hoped for.
