# ITP -- a trustworthy interactive theorem prover, and a gym for learning olympiad mathematics

A small, heavily tested **trusted kernel** in C++23, an **untrusted** Python layer on top of it that does all the convenient things (tactics, decision procedures, libraries from Peano arithmetic to elementary
number theory, a problem bank of formal olympiad problems), and a **gym environment** with no text interface in which a learner proves them. The point of the split is that nothing above the kernel has to be
believed: a tactic, a decision procedure, a search or a neural network can be wrong in any way it likes, and a finished proof is still checked by the kernel (and, independently, by a second checker written from the
typing rules in Python), so a wrong tool wastes time and cannot make a false statement provable.

Targets: IMO-level algebra and number theory first (the problem bank is the beginning of that dataset), then undergraduate mathematics. **Status, honestly:** the kernel, the libraries, the environment, the data
and the experiment pipeline are built and heavily tested; the learners are a baseline. No large training run has been made and nothing has been measured on the current dataset: the paper's experiments are for
`scripts/` to run (`docs/EXPERIMENTS.md`), and `docs/TRAINING.md` says what the first, small measurements were and how little they show.

| | |
|---|---|
| Kernel | `src/c++/hybrid`: dependent types, inductive families (parameters, indices), structural recursion, a Prop/Type world with definitional proof irrelevance, quotients, a deterministic work budget; 7.5 k lines of C++23 plus 0.8 k of Python bindings |
| Reference kernel | `src/c++/reference`: the frozen previous generation, byte-identical (checksums), kept as the oracle of the lockstep gates |
| Checkers | the kernel; `indep_checker.py`, a second implementation of the typing rules written separately, and `recheck.py`, which re-checks a whole world (and the dependencies of a proof) with it; both must accept a proof |
| Libraries | unary and binary naturals, the integers as a quotient of `N x N`, ordered ring laws, number theory (divisibility, Bezout, gcd, primes); built by tactic scripts, no axioms (four classical axioms are an optional layer) |
| Decision procedures | `ring` (naturals, binary naturals, integers), `linarith`, `nlinarith`, `linear_combination`, `interval_cases` -- each produces a certificate the kernel checks |
| Problem bank | **268 formal problems** (levels 1-5): 41 IMO, 18 ISL and 1 APMO problems (only 12 of them with proofs), 31 AIME, 72 AMC-style problems written for the bank, 105 graded problems, classical results and exercises; **187 have a kernel-checked reference proof** (a sequence of actions), 81 are statement-only. Every problem says where its text comes from (`origin`) and whether it was compared with a source (`verified`) -- the contest texts are paraphrases, some written from memory (`docs/BANK.md`). Dataset v2 in `data/dataset_v2`: the problems and proofs, 453 library lemmas with proofs, a label for every action (direction, depth, size of the subtree), and the theorems that the reference proofs contain, each with its own proof; splits by problem family |
| Environment | `gym_env` / `gym_acts`: goals as data, menus of legal actions, actions that *say* their term arguments, atomic probes; `gym_tree`: the proof tree of an episode and the theorems it proved (hindsight); `learner/bank_env.py` is a Gymnasium environment (`ITPBank-v0`) that starts on any goal given as data, has a step budget and reports what each action achieved |
| Learner | features, a transformer + GRU policy network over *options*; behaviour cloning (also as pre-training on synthetic theorems); provers: sampling, policy-guided IDA\*, MCTS, a manager-worker design, with blind and random baselines; expert iteration and PPO with **hindsight** (every goal an attempt closes on the way is a theorem with a proof) and a **bootstrapped library** (kernel-checked hindsight theorems become premises); all measured by one protocol (kernel-checked proofs within a budget of probes). A baseline, not a result: see `docs/TRAINING.md` |
| Synthetic data | `src/python/synth`: a generator of kernel-checked theorems (proof trees grown backwards from the library's lemmas, labelled by depth, lemmas and hypotheses; reproducible from a seed and an index; verified from the data alone), shards, splits, a dataset card and saturation statistics; no statement of the bank enters it (excluded by digest and by shape); see `docs/GENERATOR.md` |
| Experiments | `scripts/` + `src/python/experiments`: one resumable pipeline that generates the data, pre-trains, fine-tunes, runs the RL variants and the ablations over several seeds, evaluates every model with the same provers on the same problems, measures throughput and memory, and writes the tables, figures and numbers of the paper (`paper/`, a TMLR skeleton); see `docs/EXPERIMENTS.md` |
| Tests | 54 gates: 47 kernel / tactic / environment / bank gates, the problem-set gates S49-S50, the learner's gates S44 and S51, the repository gates S45-S46, the generator's gate S48; 1616 mutants of the kernel, libraries, checker and environment (none yet for the problem sets, the learner, `synth` and the experiment pipeline) to show the gates bite |

## Quick start

Requirements: a C++23 compiler (g++ 13 or recent clang), Python 3.13 with `pip install -r requirements.txt` (numpy, scipy, nanobind; torch, gymnasium and matplotlib for the learner and the experiments). Developed and
tested on Linux; the build script also has the macOS flags (untested here).

```
make kernel                              # builds build/itp_env_lnl*.so (the kernel) and build/itp_env_ref*.so (the oracle)   [python3 src/python/build_kernel.py]
python3 examples/01_prove_with_tactics.py
python3 tests/run_gates.py -j 2          # the 47 kernel gates, about 13 minutes on two cores;  make gates-all runs all 54
make dataset-verify                      # replay every recorded proof of data/dataset_v2 in a fresh engine
```

The experiments (they need torch; a GPU is optional and makes the networks faster):

```
make check-env                           # is this machine ready? python, torch, CUDA, memory, disk
make smoke                               # the whole pipeline on a tiny world: do this before a real run (an hour or two on two cores)
PROFILE=small SEEDS=0-2 make experiments        # the shape of every result: the whole bank, 5 000 synthetic train theorems, 3 seeds
DEVICE=cuda PROFILE=full SEEDS=0-6 make experiments      # the paper: 60 000 synthetic train theorems, 7 seeds (default 5); resumable, `make status` says where it is
./scripts/run_all.sh --profile full --seeds 0-6 --device cuda      # the same, as options of the script (every setting is one: `--help`)
make report ; make paper                 # tables, figures, numbers (paper/generated/); paper/main.pdf (needs latexmk)
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
src/python/environment/    the gym: engine, action language, typed term builder, policy assembler, search, recorder, proof trees (hindsight)
src/python/problems/       the problem bank (bank_*.py), faithfulness checks, duplicate screen, splits, the dataset builder, sub-theorems
src/python/learner/        features, network, training (ml_bc, ml_ei, ml_ppo), provers (ml_agents, ml_search, ml_mcts, ml_hier), hindsight and library (ml_her, ml_lemmas), evaluation, the Gymnasium environment
src/python/synth/          the synthetic theorem generator: lemma pool, proof trees grown backwards, records, shards, merge and splits, verification from the data alone (docs/GENERATOR.md)
src/python/experiments/    the experiment graph, scheduler, collector, statistics, tables, figures, throughput benchmarks, report (docs/EXPERIMENTS.md)
scripts/                   the shell scripts that run the experiments (00_check_env ... 07_report, run_all, smoke_test, status; env.sh and local.env.example hold the settings)
paper/                     the TMLR paper skeleton (main.tex, bibliography, style) and generated/ (tables and figures from `make report`)
tests/                     the gates by role (kernel, library, environment, problems, learner, synth, repo), shared modules (support), fuzzers, benchmarks, golden files
mutants/                   the mutation catalogue (1616 mutants) and its driver
examples/                  runnable examples; examples/repros holds scripts that reproduce fixed defects
data/dataset_v2/           the dataset (JSON + card + manifest);  data/results/v1/ the result files of the first runs on the earlier dataset (historical)
docs/                      FINDINGS (the tracker), ROADMAP, KERNEL_DESIGN, ACTIONS, BANK, GENERATOR, ARCHITECTURE_RL, TRAINING, EXPERIMENTS
legacy/                    the previous generation's modules, tests and notes (some gates import curriculum / hypergraph from here)
tools/                     sanitizer run, environment check, release archive (make_dist.py)
```

Modules are imported by their plain names (`import gym_env`); the names are unique across the tree (S45 checks it).

## Where to read next

* `docs/KERNEL_DESIGN.md` -- the kernel: terms, evaluation, conversion, inductive types, the options and their reasons.
* `docs/FINDINGS.md` -- every defect and design decision found so far, with the test that pins it.
* `docs/BANK.md`, `docs/ACTIONS.md` -- the problem bank (what makes a formal statement *the* problem; where each text comes from; how duplicates and splits are handled) and the action language.
* `docs/GENERATOR.md` -- the synthetic theorem generator: how a theorem and its proof are grown, what is rejected and why, how many theorems there can be, how the data are reproduced and verified, what they do not show.
* `docs/ARCHITECTURE_RL.md` -- how a learner should be structured to climb from imitation to a manager/worker design; `docs/TRAINING.md` -- what exists and what it achieves (little, honestly).
* `docs/EXPERIMENTS.md` -- the experiments of the paper: questions, protocol, statistics, the run graph, the scripts, what the results can and cannot show.
* `docs/ROADMAP.md` -- what is missing for IMO-level and undergraduate mathematics, and a calibrated plan.
* `tests/README.md` (the gates), `mutants/README.md` (mutation testing), `legacy/README.md`.

## Trust

What must be believed for "this proof is correct": the C++ kernel (`src/c++/hybrid`, about 8 k lines with its bindings), the C++ compiler and the machine -- and, as a cross-check that does not share code with the kernel, the independent
checker (`src/python/checker/indep_checker.py`). `audit.dependencies` lists the axioms a theorem uses (the libraries use none; the integers use the kernel's five quotient constants). Everything else -- tactics,
libraries' scripts, the bank, the search, the networks, the experiment pipeline -- is outside the trust boundary by construction, and the gates and mutants exist to make that boundary real rather than hoped for.
