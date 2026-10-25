# EXPERIMENTS.md -- the experiments of the paper: protocol, runs, scripts, outputs

Everything the paper reports is produced by one pipeline (`scripts/`, which wraps `src/python/experiments/`): generate the data, train every method for several seeds, evaluate every model with the same
provers on the same problems, measure throughput, and turn the logs into tables, figures and the numbers quoted in the text. **Nothing in a table is typed in.** This file says what is run, how it is
measured, how to run it, and what the results can and cannot show.

```
scripts/smoke_test.sh              # the whole pipeline on a tiny world: does it run on this machine?   (an hour or two on 2 cores; much less on a big machine)
PROFILE=small scripts/run_all.sh   # the first real numbers: the whole bank, 5 000 synthetic train theorems, 3 seeds
PROFILE=full  scripts/run_all.sh   # the paper: 60 000 synthetic train theorems, 5 seeds (SEEDS=0-6 for seven)
```

## 1. Questions, experiments, outputs

| question | experiment (runs) | table / figure |
|---|---|---|
| E0 Is the benchmark what it says it is? | `data/dataset` (replays every recorded proof, checks digests) | `bank` |
| E1 How do the methods compare on held-out problems? | `eval/main/*`, `eval/baselines` | `main`, `strict`, `fig_budget`, `levels`, `fig_levels` |
| E2a How does accuracy grow with inference compute? | the same rows read at budgets 10 ... 2000 probes | `fig_budget` (log x-axis) |
| E2b How does olympiad accuracy grow with the amount of synthetic pre-training? | `pre/n*`, `ft/bc_n*`, `ft/bc`, `ft/bc_scratch`, `eval/scale/*`, `eval/main/bc` | `scaling`, `fig_scaling` |
| E2c How does the learner improve with training iterations / compute, and what happens to proof length? | `rl/*` logs | `fig_learning`, `fig_bc`, `cost` |
| E3 Which ingredient matters? (ablations) | pre-training: `bc` vs `bc_scratch`; fine-tuning: `bc` vs `pre`; hindsight: `ei_her` vs `ei`, `ppo` vs `ppo_nosub`; bootstrapped library: `ei_her_boot` vs `ei_her`; algorithm: `ei_her` vs `ppo`; actor: `ei_mw`, `ei_mcts`, `ei_ida` vs `ei_her` | `ablations`, `fig_ablations` |
| E4 Which prover is best for a given network? | `eval/solvers/*`, `eval/mw/*` | `solvers`, `mw`, `fig_solvers` |
| E5 Is the synthetic data diverse, and when does the generator saturate? | `data/synth_*` | `generator`, `fig_generator` |
| E6 Does the library grow, and are the lemmas used? | `rl/ei_her_boot/*`, `eval/main/ei_her_boot/*` | `library`, `fig_library` |
| E7 What does it cost? throughput, memory, worker scaling, kernel micro-benchmarks | `bench/throughput`, `bench/kernel` | `bench_solvers`, `bench_workers`, `bench_world`, `bench_kernel`, `fig_throughput`, `cost` |
| E8 Is the evaluation sound? | every solved proof is replayed, kernel-checked and re-checked by the independent checker | `soundness` |

## 2. Definitions (they are the definitions of the paper)

* **Task**: a problem of the bank in a *regime*. The experiments use the *prefix* regime (every global before the problem is on the menu, the problem's own helper lemmas included) on all problems, and
  additionally the *strict* regime (helper lemmas hidden) for the problems where that is a different task (`ml_evaluate --regimes prefix,strict`; reported in the table `strict`). Only 10 of the 268 problems
  (4%) have helper lemmas of their own, so for the other 258 the regimes coincide and nothing is given away; the table `main` has a column for them (`free`) and the table `strict` reports the ten.
* **Probe**: one try of an action on a goal by the engine, whoever asks (a search, a policy rollout, a worker). Refused tries cost as much as accepted ones. All budgets are in probes.
* **Solved within B probes**: the prover returned a proof that the engine replayed and the kernel checked, after at most B probes. A prover is run **once at the largest budget** and the curve
  "solved within B" is read off that run (exact, because a budget only truncates a run; a run stopped by the per-problem time limit counts as unsolved at every larger budget).
* **Replicate**: a whole run of the pipeline from its seed (pre-training, fine-tuning, RL, evaluation; the seed also seeds the sampling of the prover). The synthetic data set is **one** set shared by all
  seeds (the seed varies initialisation, data order and sampling; it does not vary the generator). The dev/test split is fixed.
* **Subsets**: `all`, `proved` (a reference proof exists: solvable with the library), `unproved` (statement-only: may or may not be), `free` (no helper lemma of its own).
* **Headline**: the **test** split at the profile's `headline` budget (full: 1000 probes). Dev is for development; the starting checkpoint of every RL variant was selected on dev (teacher-forced loss),
  so dev numbers are optimistic and are shown only next to test.

## 3. Statistics

* A percentage over several replicates is the mean over seeds with the 95% Student-t interval (half-width shown). With one replicate (the model-free baselines) the interval is Wilson's, over problems.
* A difference between two methods is paired by seed (learners that continue from `bc` start from its checkpoint of the same seed): the mean difference, its t-interval and a paired t-test, with Holm's
  correction over the family of ablations in one table; next to it a paired bootstrap over **problems** (the same resampled problems in both methods), which ignores training noise but shows how much the result
  depends on which problems the benchmark holds. With 3 seeds the first interval is wide by construction; with 5-7 it is the number to read.
* Generator saturation: Good-Turing coverage and the Chao1 estimate of the number of distinct proof skeletons / statement shapes / recipes, with rarefaction curves.
* The report never drops a missing result silently: `report.md` starts with a Warnings section (missing rows, solver errors, mixed commits).

## 4. Profiles

| | smoke | small | full |
|---|---|---|---|
| purpose | does the pipeline run? (numbers mean nothing) | the shape of every result | the paper |
| problems | 17 (+ what they need) | all | all |
| seeds (default) | 2 | 3 | 5 (`SEEDS=0-6` for 7) |
| synthetic TRAIN theorems (pre-training sizes) | 300 (0, 100, 300) | 5 000 (0, 500, 1500, 5000) | 60 000 (0, 1000, 4000, 15000, 60000) |
| generator attempts, first round (`grow` adds rounds up to 3x) | 1 800 | 40 000 | 450 000 |
| epochs: pre-train / fine-tune / scratch | 2 / 3 / 3 | 6 / 12 / 20 | 6 / 20 / 30 |
| expert iteration: iterations x tasks x probes | 2 x 8 x 40 | 10 x 48 x 100 | 30 x 128 x 200 |
| PPO: iterations x episodes | 3 x 8 | 60 x 32 | 250 x 64 |
| evaluation: largest budget (headline), time limit | 60 (60), 5 s | 500 (200), 30 s | 2000 (1000), 60 s |
| workers per run | 0 | 2 | 4 |

The authoritative values are in `src/python/experiments/exp_spec.py` (`PROFILES`). Any field can be changed per invocation: `SET="rl_iters=20 eval_probes=1000" scripts/run_all.sh`
(no spaces inside a value). The effective profile is written to `RUNS/PROFILE.json`, shown in the report (table `hyper`) and refused to be mixed: one runs directory = one profile / seed list / `SET`
(machine settings -- `DEVICE`, `WORKERS`, `JOBS`, `CORES`, `MEM_GB` -- may change between invocations).

## 5. The run graph (`exp_spec.py`)

A *run* is a node: an id (its directory under the runs directory), a command, the runs it needs, the files that exist when it succeeded. `exp_run.py` executes the graph, resumably.

| stage | run id | what | needs |
|---|---|---|---|
| data | `data/dataset` | verify the dataset (smoke: build it) | -- |
| data | `data/synth_gen`, `synth_merge`, `synth_verify`, `synth_card`, `synth_stats` | `grow`: generate until the merge would keep the target number of TRAIN theorems (the held-out 12% come on top; a shortfall at three times the first round's attempts is an error, not a smaller set), merge (excluding every bank statement), verify from the data alone, card, saturation statistics | dataset |
| pre | `pre/n<N>/s<seed>` | behaviour cloning on N synthetic theorems only (library-only world: no olympiad data) | synth_merge |
| ft | `ft/bc/s<seed>` | pre-trained (N = max) then fine-tuned on the bank's train split and the library lemmas | `pre/n<max>` |
| ft | `ft/bc_scratch`, `ft/bc_mw`, `ft/bc_n<N>` | no pre-training; manager-worker labels; smaller pre-training | |
| rl | `rl/<variant>/s<seed>`, variant in `ppo ppo_nosub ei ei_her ei_her_boot ei_mw ei_mcts ei_ida` | RL from the seed's `ft/bc` (`ei_mw` from `ft/bc_mw`) | `ft/*` |
| eval | `eval/main/<variant>` (`ei_her_boot`: one run per seed, with its own lemma file) | every model, solver `sample`, dev + test, prefix + strict regimes, independent re-check | checkpoints |
| eval | `eval/solvers/{bc,ei_her}`, `eval/mw/{bc_mw,ei_mw}`, `eval/scale/bc_n<N>`, `eval/baselines` | the prover study, the manager-worker study, the scaling study, the model-free baselines | |
| bench | `bench/throughput`, `bench/kernel` | throughput / memory / worker scaling; kernel micro-benchmarks | `ft/bc/s<first>` |
| report | `report` | tables, figures, numbers | -- |

`python3 src/python/experiments/exp_spec.py full` prints the graph; `scripts/status.sh` says what is done. Checkpoints that are evaluated: behaviour-cloned models use **`best.pt`** (the epoch with the
lowest dev loss); RL models use **`last.pt`** (no selection: nothing about dev or test decides which iteration is reported).

### The methods compared

| variant | what it is |
|---|---|
| `uniform`, `blind` | no network: random legal actions; iterative deepening on the engine's own action costs |
| `pre` | behaviour cloning on the synthetic theorems only (never saw an olympiad problem): zero-shot |
| `bc_scratch` | behaviour cloning on the bank's train split + library lemmas from random weights |
| `bc` | pre-trained on all the synthetic theorems, then `bc_scratch`'s data: the starting point of every RL variant |
| `bc_n<N>` | the same with N synthetic theorems (scaling) |
| `bc_mw` | `bc` with manager-worker masks and labels |
| `ppo`, `ppo_nosub` | PPO from `bc`, with / without the hindsight sub-theorem goals |
| `ei` | expert iteration from `bc` (keeps the kernel-checked proofs of whole tasks) |
| `ei_her` | + hindsight: theorems an attempt proved on the way are goals and training data |
| `ei_her_boot` | + bootstrapping: kernel-checked hindsight theorems become library lemmas (train side only; never one that says what a dev/test problem says) |
| `ei_mw`, `ei_mcts`, `ei_ida` | `ei_her` with the manager-worker prover / MCTS / IDA* as the actor |

Provers (`ml_agents.py`): `blind`, `uniform`, `sample`, `greedy`, `ida`, `mcts` (`mcts:value=critic`), `mw` (`mw:gate=never|fixed`).

## 6. Leakage protocol (what keeps the test split clean)

1. The split is by problem **family** (frozen salt); variants of one problem are never separated.
2. The synthetic generator excludes every statement of the bank (all splits), of the library and of the sub-theorem set, by digest and by shape (`synth_cli merge --exclude-dataset`); the statistics in
   `data/synth_stats` count what was excluded.
3. Nothing from dev or test is a goal, a demonstration or a lemma during training: sub-theorems are taken from train proofs only; the bootstrapped library admits a lemma only from a train task and refuses
   one that says what a dev/test problem says (a comparison on a grid of values, any naming and order of the parameters), or that is a statement of the bank or the library.
4. Models that were trained with a bootstrapped library are evaluated with it (`--lemmas auto` reads the file next to the checkpoint and checks its digest); the library is per seed.
5. The test split is not used to choose anything: no epoch, no hyper-parameter, no checkpoint.

## 7. Outputs (`RUNS/`)

```
RUNS/PROFILE.json  GRAPH.json  .status/<id>.json        the effective profile, the graph, the return code / times / host / commit of every run
RUNS/logs/<id>.log                                       stdout+stderr of every command
RUNS/data/  RUNS/synth/                                  (smoke) dataset; the synthetic theorems: shards, merged splits, manifest, verification.json, DATASET_CARD.md, stats.json
RUNS/pre|ft|rl/<variant>/s<k>/                           log.jsonl  final.json  best.pt  last.pt  state.pt  (lemmas.json)
RUNS/eval/<group>/<variant>[/s<k>]/                      rows.jsonl  meta.json  recheck.json
RUNS/bench/                                              bench.json  kernel.json
RUNS/report/                                             report.md  numbers.json  numbers.tex  problems.csv  tables/*.md|tex  figures/*.pdf|png
```

`scripts/07_report.sh` copies `tables/`, `figures/` and `numbers.tex` to `paper/generated/` (full profile) and builds `paper/main.pdf`.

## 8. Running it

```
scripts/00_check_env.sh            # python, libraries, the built kernel, cores / memory / disk / GPU, git state
scripts/01_build_dataset.sh        # kernel (make kernel) + replay every recorded proof of data/dataset_v2
scripts/02_generate_synthetic.sh   # the synthetic set (full: 60 000 train theorems; about 450 000 attempts, more if the caps need it)
scripts/03_pretrain_finetune.sh    # pre/*, ft/*
scripts/04_reinforcement_learning.sh
scripts/05_evaluate.sh
scripts/06_benchmarks.sh           # alone on an idle machine
scripts/07_report.sh
scripts/run_all.sh                 # all of it, one scheduler (stages overlap), resumable
scripts/status.sh
```

Settings are options of every script (`./scripts/run_all.sh --profile full --seeds 0-6 --device cuda --jobs 4 --set rl_iters=20 --set eval_probes=100`; `--help` lists them), or environment variables
(`PROFILE=full SEEDS=0-6 ./scripts/run_all.sh`), or written once in `scripts/local.env`; an option wins over the environment and that over `local.env`.  The settings: `PROFILE`, `SEEDS` (the default is five
seeds, `0-4`; `0-6` is seven; `0,2,5`), `RUNS`, `JOBS`, `CORES`, `MEM_GB`, `DEVICE` (`cpu`, `cuda`), `WORKERS`, `SET` (`--set` is repeatable), `TARGET` (`'rl/ei_her/*'`). A script that calls another one hands its
settings on. The scheduler packs runs within `CORES` and `MEM_GB` using the runs' estimates (`MEM_GB` table in `exp_spec.py`; a run's cores are its `--workers`); every command is resumable
(a training run continues from its last epoch/iteration, an evaluation from its rows, the generator from its shards), so `Ctrl-C` and a rerun lose at most one checkpoint interval. `exp_run.py --force ID
[--cascade]` redoes a run (and what depends on it); a failed run blocks only what needs it.

**GPU.** `DEVICE=cuda` trains the networks on the GPU; proof search, the engine and evaluation stay on the CPU (a decision of the network on one goal is faster there than the transfer). Worker processes are
forked before CUDA is touched and use CPU copies of the weights. The code path for CUDA has been written to be correct by construction but **was not run on a GPU by its author**; run
`DEVICE=cuda scripts/smoke_test.sh` first and `DEVICE=cuda scripts/06_benchmarks.sh` to see the training rate before committing to the full profile.

## 9. What the results can and cannot show

* The benchmark is small (`bank` table) and the test split smaller; a few problems move a percentage point. The paired bootstrap over problems is there to say so.
* Statement-only problems are lower bounds on solvability; read `proved` and `unproved` separately. A statement-only problem that some run solves (each replayed and kernel-checked) is listed in
  `report.md`, "Statement-only problems solved".
* Contest texts are paraphrases, some written from memory; the labels `origin` and `verified` of every problem are in `problems.csv` and the dataset card.
* The prefix regime can be easy for problems with helper lemmas: use the helper-free column and the strict table.
* The synthetic proofs are backward chains: they can teach premise selection and term building, not forward reasoning, cases or induction. The scaling experiment measures whether they help; it does not assume it.
* Models are small and trained from scratch; nothing here says how the methods behave at the scale of language-model provers.
* RL variants share hyper-parameters; none was tuned per method. PPO and expert iteration are compared at equal iterations and by cumulative probes (logs), not at equal wall-clock.
* The mutation catalogue (1 616 mutants) is the evidence that the gates bite; the full mutation run takes many hours and is not part of this pipeline (`python3 mutants/mutate.py PREFIX`).

## 10. Compute

`python3 src/python/experiments/exp_estimate.py PROFILE [--seeds 0-6] [--cores N] [--set k=v]` prints the estimate of a profile stage by stage. It multiplies the work of every run (training steps, probes) by
rates: those of the development machine (2 cores; `exp_estimate.RATES`: 55 training steps/s for the default network, 6 to 60 probes/s depending on the solver) or, once `scripts/06_benchmarks.sh` has run, the
ones it measured on YOUR machine (`RUNS/bench/bench.json`; on a GPU pass `--train-rate` with the rate the benchmark reports for `DEVICE=cuda`).
**These are estimates. The author has not run the `full` profile** (the pipeline was exercised on worlds of `smoke` size and smaller, and the rates come from those runs); run `scripts/smoke_test.sh` and
`scripts/06_benchmarks.sh` first and read the estimate again with your own rates.

| profile | seeds | runs | run-hours | core-hours | wall-clock on 8 cores, perfect packing |
|---|---|---|---|---|---|
| smoke | 2 | 55 | 2.2 | 2.2 | 0.7 h |
| small | 3 | 77 | 57 | 102 | 13 h |
| full | 5 | 122 | 344 | 1 111 | 139 h (about six days) |

`full`, in core-hours: reinforcement learning 797 (40 runs: PPO and the expert-iteration variants), evaluation 215, pre-training 64, fine-tuning 23, data 11, benchmarks 1 -- the RL runs and the evaluation of
the search solvers are 91 % of it. The cost is linear in the number of seeds (`--seeds 0-6` costs 1.4 times the five-seed training and evaluation, the data and the benchmarks once). To spend less: run the
`small` profile; fewer seeds; one method (`--target 'rl/ei_her/*'`); fewer iterations or probes (`--set rl_iters=100 --set eval_probes=500`, section 4); a GPU speeds up the network updates only (proof search stays on
the CPU).

**Memory.** The world of the whole bank takes about 1.2 to 1.4 GB per process when built and up to about 2.8 GB in the runs that were measured (`smoke`); forked workers share it. Pre-training on 60 000
theorems keeps about 3 GB of steps in memory. (The gate S50, which replays the whole bank's proofs, reaches 4.5 GB: on an 8 GB machine do not run it next to another heavy gate -- `tests/run_gates.py` runs the heavy gates one at a time.) Set `MEM_GB` to what the machine has; the scheduler runs fewer jobs at once rather than more.
**Disk.** Measured on `smoke`: about 30 MB per training run (checkpoints) and about 10 KB per generator attempt; extrapolated to `full` that is about 3 GB for its 90 training runs and several GB for the
450 000 generator attempts. Keep 20 GB free.

## 11. Troubleshooting

* `FAILED ...` in the runner's output: read `RUNS/logs/<id>.log`; fix; run the same script again (finished runs are skipped).
* Out of memory: lower `JOBS`, or `MEM_GB`; the world of the whole bank is about 2.5 GB per process and forked workers share it; pre-training on 60 000 theorems keeps about 3 GB of steps in memory.
* "holds runs made with another profile": the runs directory was made with other `SEEDS`/`SET`; use the same, or another `RUNS`.
* "is being run by process N": a previous invocation is alive; if not, remove `RUNS/.lock`.
* A table or figure is missing from the report: its inputs were not produced; `report.md` > Warnings says which.
