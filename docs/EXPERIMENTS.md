# EXPERIMENTS.md -- the experiments of the paper: protocol, runs, scripts, outputs

Everything the paper reports is produced by one pipeline (`scripts/`, which wraps `src/python/experiments/`): generate the data, train every method for several seeds, evaluate every model with the same
provers on the same problems, measure throughput, and turn the logs into tables, figures and the numbers quoted in the text. **Nothing in a table is typed in.** This file says what is run, how it is
measured, how to run it, and what the results can and cannot show.

```
scripts/smoke_test.sh              # the whole pipeline on a tiny world: does it run on this machine?   (about 70 minutes of run time on 2 cores, two runs at a time; much less on a big machine)
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
| E2d Does the encoder of the goal matter, and does the answer depend on the learner? | `pre/<arch>_n<N>`, `ft/bc_<arch>`, `rl/<learner>_<arch>`, `eval/arch/*` (arch in `bigru cnn bow`; the transformer's runs are the main ones) | `arch`, `fig_arch`, `bench_archs` |
| E2e How do imitation quality and olympiad accuracy scale with the amount of synthetic data and with the size of the network? | `pre/n<N>` (data), `pre/w<d>_n<N>`, `ft/bc_w<d>`, `eval/size/*` (other widths beside the default 128) | `laws`, `size`, `fig_laws` |
| E2f Can a learner start from nothing? | `init/<arch>/s<k>` (an untrained network), `rl/scratch_<learner>[_<arch>]`, `eval/scratch/*` | `scratch`, `fig_scratch` |
| E3 Which ingredient matters? (ablations) | pre-training: `bc` vs `bc_scratch`; fine-tuning: `bc` vs `pre`; hindsight: `ei_her` vs `ei`, `ppo` vs `ppo_nosub`; bootstrapped library: `ei_her_boot` vs `ei_her`; algorithm: `ei_her` vs `ppo`; actor: `ei_mw`, `ei_mcts`, `ei_ida` vs `ei_her` | `ablations`, `fig_ablations` |
| E4 Which prover is best for a given network? | `eval/solvers/*`, `eval/mw/*` | `solvers`, `mw`, `fig_solvers` |
| E5 Is the synthetic data diverse, and when does the generator saturate? | `data/synth_*` | `generator`, `fig_generator` |
| E6 Does the library grow, and are the lemmas used? | `rl/ei_her_boot/*`, `eval/main/ei_her_boot/*` | `library`, `fig_library` |
| E7 What does it cost? throughput, memory, worker scaling, the cost of each encoder, kernel micro-benchmarks | `bench/throughput`, `bench/kernel` | `bench_machine`, `bench_solvers`, `bench_breakdown`, `bench_latency`, `bench_workers`, `bench_archs`, `bench_world`, `bench_kernel`, `fig_throughput`, `fig_breakdown`, `fig_latency`, `cost` |
| E8 Is the evaluation sound? | every solved proof is replayed, kernel-checked and re-checked by the independent checker | `soundness` |
| E9 Is the verifier sound, and how do we know? | every gate; every mutant of the catalogue; falsification fuzzing over seeds and worlds; the kernel against the independent checker on thousands of mutated proofs; the trusted code counted; the findings classified (`scripts/08_soundness.sh`) | `results/soundness`: `sound_summary`, `sound_gates`, `sound_mutants`, `sound_survivors`, `sound_fuzz`, `sound_differential`, `sound_trusted`, `sound_findings`, `sound_open` |
| E10 Are the statements what they claim to be? | for each problem of the bank: the formal statement read back and compared with its intended meaning on a box of points, the slips of a formaliser tried on it, its provenance, the cautions, and where it stands with the ledger of second readers (`scripts/09_audit.sh`) | `results/audit`: `audit_summary`, `audit_origins`, `audit_cautions`, `audit_reviews`, `audit_failures` |

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
* The report never drops a missing result silently: `report.md` starts with a Warnings section (missing rows, solver errors, runs made with different versions of the code).

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

The **studies** (E2d-E2f) are fields of the profile, so that a part can be changed or dropped like any setting -- `--set archs=() --set widths=() --set scratch=()` is the core experiment alone:

| field | meaning | smoke | small | full |
|---|---|---|---|---|
| `archs` | encoders of the goal compared with the transformer (`bigru`, `cnn`, `bow`) | bigru | bigru cnn bow | bigru cnn bow |
| `arch_learners` | learners run with each of them (`bc` is always there; `ei_her`, `ppo`) | bc ei_her | bc ei_her | bc ei_her ppo |
| `widths` | other widths of the transformer, multiples of 32 (layers and heads grow with the width: 64 has 2 layers, 128 has 3, 256 has 4, 512 has 6) | 64 | 64 256 | 64 256 |
| `width_learners` | learners run at each width | bc | bc | bc |
| `scratch` | learners that start from random weights (`learner`, or `learner@encoder`) | ei_her | ei_her ppo | ei_her ppo ei_her@bigru ei_her@cnn ei_her@bow |
| `scratch_mult` | a learner from random weights runs this many times the iterations of the same learner continued from behaviour cloning | 1 | 3 | 3 |

A study changes the network or the start and nothing else: the synthetic data, the recipe (epochs, learning rates, batch, iterations, probes, time limits), the seeds and the evaluation are those of the core experiment. A learner from random weights draws
70 % of its goals from the synthetic theorems (the bank's problems are far out of reach of a random policy) and the rest from the bank's train problems and the sub-theorems proved on the way.

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
| pre | `pre/<arch>_n<max>/s<seed>`, `pre/w<d>_n<max>/s<seed>` | the pre-training of the networks of the studies: all the synthetic theorems, another encoder or another width | synth_merge |
| ft | `ft/bc_<arch>`, `ft/bc_w<d>` | the same fine-tuning of those networks | their `pre` run |
| rl | `rl/<learner>_<arch>`, `rl/<learner>_w<d>` | a learner continued from the study network's `ft/bc_...` of the same seed | `ft/bc_...` |
| pre | `init/<arch>/s<seed>` | an untrained network (`ml_init.py`): the start of a learner from random weights (it is a `pre`-stage run, written to `RUNS/init/`) | data |
| rl | `rl/scratch_<learner>[_<arch>]/s<seed>` | the learner from `init/<arch>/s<seed>`: no pre-training, no demonstrations, `scratch_mult` times the iterations | `init/...` |
| rl | `rl/<variant>/s<seed>`, variant in `ppo ppo_nosub ei ei_her ei_her_boot ei_mw ei_mcts ei_ida` | RL from the seed's `ft/bc` (`ei_mw` from `ft/bc_mw`) | `ft/*` |
| eval | `eval/main/<variant>` (`ei_her_boot`: one run per seed, with its own lemma file) | every model, solver `sample`, dev + test, prefix + strict regimes, independent re-check | checkpoints |
| eval | `eval/solvers/{bc,ei_her}`, `eval/mw/{bc_mw,ei_mw}`, `eval/scale/bc_n<N>`, `eval/baselines` | the prover study, the manager-worker study, the scaling study, the model-free baselines | |
| eval | `eval/arch/<variant>`, `eval/size/<variant>`, `eval/scratch/<variant>` | every variant of a study, solver `sample`, the same splits, regimes and budget as the main results | checkpoints |
| bench | `bench/throughput`, `bench/kernel` | throughput / memory / worker scaling; the cost of each encoder and width (`--archs`, `--widths`); kernel micro-benchmarks | `ft/bc/s<first>` |
| report | `report` | tables, figures, numbers | -- |

`python3 src/python/experiments/exp_spec.py full` prints the graph (`exp_estimate.py` prints what it costs, in total and by part: core, arch, size, scratch); `scripts/status.sh` says what is done. Checkpoints that are evaluated: behaviour-cloned models use **`best.pt`** (the epoch with the
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
| `bc_<arch>`, `<learner>_<arch>` | the whole chain with another encoder of the goal: pre-trained on all the synthetic theorems, fine-tuned, then the learner continues from it |
| `bc_w<d>`, `<learner>_w<d>` | the same with the transformer at width d |
| `scratch_<learner>[_<arch>]` | the learner from random weights: no pre-training, no demonstration |

### The encoders of the goal

`ml_model.ARCHS` is `('transformer', 'bigru', 'cnn', 'bow')`. The policy network is an *encoder* of the tokens of the goal followed by the decoder and the heads that every encoder shares (the scorers of the menu, the value head, the auxiliary
heads). `PolicyNet.encode(tokens, seg, mask, hyp_of, n_hyps) -> (cls, hyp)` is all an encoder has to provide, so another architecture is one class and one line of `ENCODERS`. `transformer` (the default) pools at the [CLS] token; `bigru` is a
bidirectional GRU over packed sequences (the state at the first token has read the whole goal; padding never changes a goal's states); `cnn` is residual dilated 1-d convolutions (kernel 5, dilations 1, 2, 4 ...; the padding is zeroed before every
convolution; the mean of the tokens' states is the goal); `bow` is a bag of tokens -- every token transformed by itself, no positions, no neighbours, the mean is the goal -- the control that says how much of the task a multiset of symbols carries.
`ml_bc`, `ml_ei` and `ml_ppo` train the architecture of their `--init` checkpoint (`ml_bc --arch` only chooses it for a new network; naming another one than the checkpoint's is an error), `ml_init.py` makes an untrained checkpoint of any
architecture and size, and a checkpoint made before the encoders were swappable has no `arch` and is a transformer. The default path is bit-identical to the network before the change (gate S44, section G). Parameter counts differ little
between encoders (the embeddings and heads are most of the parameters at this vocabulary); `bench_archs` and the table `arch` give them.

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
RUNS/PROFILE.json  GRAPH.json  .status/<id>.json        the effective profile, the graph, the return code / times / host / commit of every run and the digest of the code at its start and at its end
RUNS/logs/<id>.log                                       stdout+stderr of every command
RUNS/data/  RUNS/synth/                                  (smoke) dataset; the synthetic theorems: shards, merged splits, manifest, verification.json, DATASET_CARD.md, stats.json
RUNS/pre|ft|rl/<variant>/s<k>/                           log.jsonl  final.json  best.pt  last.pt  state.pt  (lemmas.json)
RUNS/init/<arch>/s<k>/                                   best.pt  final.json                  (an untrained network, ml_init.py)
RUNS/eval/<group>/<variant>[/s<k>]/                      rows.jsonl  meta.json  recheck.json
RUNS/bench/                                              bench.json  kernel.json
RUNS/report/                                             the results folder (below)
results/<profile>/                                       a copy of RUNS/report/, kept apart from the paper (scripts/07_report.sh, `make report`)
results/soundness/                                       the soundness evidence (scripts/08_soundness.sh, `make soundness`): not a run of the experiments, a record of the code (7.3)
results/audit/                                           the statement audit (scripts/09_audit.sh, `make audit`): what was checked about each formal statement and who has read it (7.4)
```

### 7.1 The results folder

Everything a paper is written from is saved **apart from the paper**, in one folder per profile (`results/smoke`, `results/small`, `results/full`; the same files are in `RUNS/report/`). It is made again from the runs
directory whenever `scripts/07_report.sh` runs, nothing in it is typed in, and `python3 src/python/experiments/exp_bundle.py --verify results/<profile>` checks it against its `MANIFEST.sha256`.

```
README.md                  what each file is and how to read the numbers
report.md                  the whole report, readable: what was run, warnings, every table, the figures, the statement-only problems the agents solved
numbers.json  numbers.tex  every number the report states, by name (the macros a paper can cite)
tables/<key>.md|tex|csv    one file per table
figures/<name>.pdf|png|csv the csv holds the values each figure draws (panel, series, kind, label, x, y, intervals), so a plot of one's own can be made from the numbers and not from the picture
per_problem/               outcomes.csv (every attempt), by_problem.csv, by_family.csv
stats/                     rates.csv, comparisons.csv (paired differences with intervals over seeds and over problems, Holm's correction, the smallest detectable difference), benchmark.csv
provenance/                the profile, the run graph, every run (commit, the digest of the code at the start and at the end, host, seconds), the machine, the repository facts, the dataset digests, bench.json and kernel_bench.json (the raw measurements)
MANIFEST.sha256            SHA-256 of every other file
```

**Which code made the numbers.** A result is bound to the CODE it was made with, not to a commit. The digest (`mutants/mutate.py tree_digest`; `python3 tools/check_numbers.py --tree` prints it) covers every file below `src/`, `tests/`
and `mutants/` (not caches, built files, or `.md` documents such as `tests/README.md`) and the manifest of the dataset (`itp_paths.CODE_PATHS`), and nothing else: editing a document, the paper, a tool or a script, or typing the paper, does not change it, so the commits that do that are not another
version of the code. Every run records the digest when it starts (`tree`) and when it ends (`tree_finish`) in `.status/<id>.json` and `provenance/runs.csv`; `report.md` names the versions (`code trees: ...`) and warns when the
runs were made with more than one, or when the code changed while a run was going (its result then belongs to neither version). A runs directory made before the digest was recorded is judged by its commits, which is stricter than it
should be. `dirty` (in the stamps of the soundness and audit folders and in `provenance/`) means that the CODE has changes that are not committed (a changed file of `CODE_PATHS`, or a new file there); then the commit does not
identify the code and the digest does. `tools/check_numbers.py` compares the digests of the folders with those of the code that is there (7.5).

The folders are not tracked by git (`.gitignore`); copy the folder of the run a paper cites. `PUBLISH=1 scripts/07_report.sh` also copies the LaTeX tables, the figures and the macros to `paper/generated/` and builds `paper/main.pdf`
for a paper that cites them; a paper typed by hand needs only the folder.

### 7.2 What `bench/throughput` measures (`exp_bench.py`, `exp_profile.py`)

Run it alone on an idle machine (`scripts/06_benchmarks.sh`); every number is that machine's, and the machine is recorded (`bench_machine`: processor, usable cores, memory, versions, a digest of the kernel's sources, the load average before and
after; a busy machine at the start is a warning of the report).

* **Throughput** (`bench_solvers`, `fig_throughput` a). Probes (one try of an action by the engine), decisions of the policy and actions per second, per prover, on one core, over a fixed set of train problems picked deterministically.
  The figure is the median of `--repeats` passes (default 3) after an untimed warm-up call (the first pass is cold: the caches of the kernel and of the engine fill) with the range of the passes beside it. The passes must find the same
  proofs with the same number of probes, or the report warns.
* **Where the time goes** (`bench_breakdown`, `fig_breakdown`). One more pass with a ledger around the engine (the menu of a goal, the normal forms, a probe, the tactic and the kernel under it, a replay, the start of a proof, the
  verification), the decision language of the Gymnasium interface (the Python assembler), the features and the network. The categories are **exclusive** (the time of a call does not include the calls under it), so they add up to the wall time; what
  no category accounts for is shown as search and glue. The instrumented pass must find what the plain pass finds (`same_outcome`), and its cost is in `bench.json` (`ratio_to_plain`; it is what a profiler costs, not what the prover costs).
* **Latency** (`bench_latency`, `fig_latency`). Inclusive wall-clock latencies with the median, 90th and 99th percentile and the maximum: a probe, by kind of action and by outcome (`ok`, `refused`, `fuel`, `bug`, `raised`), a menu (with the
  number of options), a replay, the start of a proof, a decision and a reset of the Gymnasium interface, and the cost of checking a reference proof (the engine's replay, the kernel's check of the finished term, the independent checker). The percentiles
  are linear interpolations between observed values; with fewer than 100 samples the 99th percentile is little more than the maximum.
* **Workers and memory** (`bench_workers`, `fig_throughput` b and c). The same tasks through 1 to N forked workers: speed-up and efficiency against one worker; memory as the proportional set size of the whole process tree (a page the
  workers share with the parent, the world, is counted once; a separate process samples it every 0.25 s) next to the sum of the resident sets (what a naive measurement says, and what overstates it); and **determinism**: the rows of the in-process run
  and of every worker count are compared field by field but for the clock, and an `outcome_digest` that does not depend on the machine is recorded, so two machines can be checked against each other.
* **The encoders and widths** (`bench_archs`, `fig_arch` c). Untrained networks of the sizes of the experiments (every encoder at the default width, the transformer at each other width): parameters, training steps per second (forward, backward
  and AdamW on batches of 64 real steps) and the latency of encoding one goal without gradient (what a decision costs first). The cost does not depend on the weights. The default transformer is always measured, as the reference the others are compared with
  (`exp_estimate` reads the speed ratios from it).
* **Fixed costs** (`bench_world`): the build of the world, its memory, the parameters of the network, training steps and inference decisions per second. **Kernel micro-benchmarks** (`bench_kernel`, `tests/bench/bench_kernel.py`): the evaluator
  and the type checker on terms of growing size.

What it cannot show: the numbers are those of this implementation on that machine (a decision of the Gymnasium interface includes the Python assembler, which is most of its time; a faster interface is an engineering task, not a claim of the
paper); a comparison with another prover's speed needs that prover on the same machine and the same statements, and is not made here.

### 7.3 The soundness folder (`exp_soundness.py`, `scripts/08_soundness.sh`)

The claims that the verifier can be trusted are evidence about the CODE, not about a run of the experiments, so they have a folder of their own, `results/soundness/`, made by one command (`make soundness`; hours: on two cores
the gates take about an hour and the mutation catalogue many hours, so use the big machine with `J=8`). It is resumable and every record carries the digest of the tree it ran on (`mutants/mutate.py tree_digest`: the code -- every file below
`src/`, `tests/` and `mutants/` but caches, built files and `.md` documents, and the manifest of the dataset; 7.1): a record of other code is listed as stale and does not count, so the folder cannot quote a number the present code did not produce.

| stage | what runs | what the folder says |
|---|---|---|
| `gates` | `tests/run_gates.py --all --json` | every gate and whether it passed, by part of the system |
| `mutants` | `mutants/mutate.py --jobs N --json` | per component of the system: mutants, caught (by assertion, crash or timeout), documented equivalent, surviving; the survivors with their reasons |
| `fuzz` | `tests/fuzz/gym_fuzz.py --json` over the seeds, naturals (with and without the arithmetic tactics) and integers | statements, true and false, proved; a false statement proved (`UNSOUND`), a proof the kernel refused (`REJECTED`), an exception (`ERROR`) must all be zero, and a campaign in which nothing true is proved is flagged: it would prove nothing |
| `differential` | `tests/kernel/s8_indep_tests.py N` (`S8_JSON`) | the kernel and the independently written checker on every library definition, the tactic proofs and thousands of mutated proofs, in both modes; they must agree on all |
| `inventory` | counted from the files | lines of the trusted code (kernel, bindings), of the second opinion, and of what is not trusted; the digests |
| `findings` | `docs/FINDINGS.md` + `docs/findings_tags.tsv` | the findings by layer and effect: how many concerned the trusted code and how many of those accepted what they must refuse; the open ones |

The folder (`README.md` first) holds `report.md`, `soundness.json`, `numbers.json|tex` (`\numSoundMutantsCaught`, `\numSoundFuzzUnsound`, ...), `tables/sound_*.md|tex|csv`, `stats/*.csv` (one row per mutant, series, gate, campaign class and
finding), `evidence/` (the raw records and the logs: the only files that are not computed) and `MANIFEST.sha256`. The status is **CLEAN** when every claim is backed by evidence for this tree, **FAILED** when the evidence
contradicts one (a gate fails, a mutant survives without a documented reason, a false statement is proved, the checker disagrees, a finding is not tagged) and **INCOMPLETE** when evidence is missing, partial or stale (a restricted
set of gates, uncommitted changes to the code (documents, the paper, tools and scripts do not count), a kernel built before its source changed); the exit status is 0, 1, 2. `exp_soundness.py report` makes the tables again from the evidence, `exp_soundness.py --verify DIR` checks the manifest.

What it does not show: mutation testing measures the gates against the faults of a hand-written catalogue; falsification is evidence, not proof; the checker is independent in code, not in authorship; and the statement of a theorem is outside
the verified chain (the statement audit says what was checked about it). The folder's README states this.

### 7.4 The audit folder (`exp_audit.py`, `scripts/09_audit.sh`)

A proof of a statement that is not the problem proves nothing about the problem, and the kernel cannot tell: it checks proofs, not statements. The bank therefore checks each formal statement against a description of its meaning
written separately in Python (`semantic.py`, `bank_spec.py`; `docs/BANK.md`), and the audit collects those checks for the whole bank in a folder of its own, `results/audit/` (`make audit`: about two minutes to build the world and four to
check the 268 statements on a laptop). It is resumable, and a record counts only for the tree it ran on, like the soundness folder.

| what | where it comes from | what the folder says |
|---|---|---|
| the checks | `problem_record`: `LoadedBank.check_semantics`, sayability in the typed term builder | per problem: points of the box, how many satisfy the hypotheses, slips tried and caught, slips declared equivalent (with the reason), failures |
| provenance | the problem's own fields | origin (contest, adapted, style, folklore), whether the published wording was compared (`verified`), kind, level, split, proved or statement-only |
| cautions | derived from a record (`CAUTIONS`) | where the evidence is weaker: adapted from a contest problem, wording not compared or recalled, predecessor on the naturals, an existential or a function quantifier, an equation in a hypothesis whose numbers are not compared, slips declared equivalent, a box with few points that only compares truth values |
| second readers | `docs/statement_reviews.tsv`, one line per reading, bound to the digest of the statement that was read | confirmed, ambiguous, author only, stale (the statement changed since), unreviewed, disputed, unfaithful; by origin and for the test split |

Verdict: **FAILED** when a check fails, a problem could not be checked, a second reader found a statement unfaithful, readers disagree or the ledger is wrong; **INCOMPLETE** when records are missing or were made for other code;
**CLEAN** otherwise. How many statements a second reader confirmed is a number of the report (`\numAuditConfirmed`, `\numAuditConfirmedPct`, `\numAuditConfirmedTest`), not a condition: the paper says what the folder says. The folder holds
`README.md` (its limits stated first-hand), `report.md`, `audit.json`, `numbers.json|tex`, `tables/audit_*.md|tex|csv`, `stats/*.csv` (one row per problem, caution, exempted slip and line of the ledger), `evidence/` (the raw records) and
`MANIFEST.sha256`, and the **review sheet** (`review_sheet.md` to read, `review_sheet.csv` to fill in) that lists the statements still to be read, the hidden test split first, each with the problem text, the statement as a formula
and the digest to put in the ledger.

`tools/review_packs.py AUDIT_DIR --readers 4 --copies 2` splits the sheet into one pack per reader (the hidden test split by default, each statement read by two people, the handout `docs/SECOND_READER.md` inside each pack); it changes nothing in the repository.

A second reader reads the sheet, fills in `verdict` (`faithful`, `adapted`, `ambiguous`, `unfaithful`) and `reviewer` in the CSV, and
`python3 src/python/experiments/exp_audit.py import-reviews review_sheet.csv --reviewer NAME` appends the lines to the ledger (all or nothing; a wrong row is named and nothing is written); `exp_audit.py report` then counts them. A reading by the
person who wrote the statement is recorded with the role `author` and does not count.

What it does not show: the spec is written by the people who wrote the statement, so a misreading of the English that both share is invisible to the checks (that is what the second readers are for); the box is finite (integers up to 8 in
absolute value, naturals up to 12 unless the spec says otherwise); `verified` is declared by whoever added the problem; the synthetic theorems are kernel-checked by construction and are not part of this audit.

### 7.5 Checking a hand-typed paper against the folders (`tools/check_numbers.py`)

The paper is typed by hand, so nothing keeps its numbers, tables and figures in step with the folders except the author. `tools/check_numbers.py paper/main.tex` reads the LaTeX source (a file, or every `.tex` below a
directory) and the results folders (`results/<profile>`, `results/soundness`, `results/audit`; the ones that exist) and says what is wrong, in three grades:

| grade | what |
|---|---|
| ERROR (exit 1) | a number macro (`\N{Name}` or `\numName`) that no folder defines (the nearest names are suggested); a table or figure that no folder holds; a folder that is not what its manifest says (a file changed after it was made, or added); a soundness or audit folder whose status is FAILED; one macro defined with two values |
| WARNING (exit 2 with `--strict`) | a soundness or audit folder that is INCOMPLETE, was made for other code than the code here (the tree digest, not the commit: edits to documents, the paper and the tools do not change it), or from a working tree with uncommitted changes; a report whose runs ran with other code than the code here, with two versions of it, or recorded none; warnings the report lists itself (missing rows, runs of different versions of the code); a fact two folders count differently (kernel lines, problems, mutants); `\todo` left in the text |
| INFO (`--literals`, `--unused`) | a number typed in the text that a macro already holds (use the macro, so that it follows the runs); a macro that nothing cites |

`--list PATTERN` prints the macros whose name matches, with value and folder (for finding the name of the number you want); `--tree` prints the digest of the code that the soundness and audit evidence is bound to. Nothing is written.
`docs/PAPER_PLAN.md` is the plan this serves: which claim each sentence may make, which folder holds its number, what is run in what order on the big machine, and what is decided before the data exist. Run
`python3 tools/check_numbers.py paper/main.tex --strict` before every release of a draft and again after the last run: a paper that passes it cites only numbers that exist, from evidence made for the code that is being released.

### 7.6 Do the tests of these tools bite? (`mutants/check_tools.py`)

The soundness folder, the statement audit, the number checker and the record of the code are what a reader trusts when the paper says "0 survivors" or "268 statements checked", so their tests are themselves tested by fault
injection, like the kernel's gates are (finding F88: of the first 172 faults, 21 got past the tests written for the tool, among them a stage that reused a result made for other code). `mutants/tool_catalogue.py` holds 252 single-line
faults of six tools (the soundness folder, the number checker, the audit, the identity of the code, the digest of the code, and the runner itself), and `make tool-mutants J=8` (or `python3 mutants/check_tools.py [TOOL ...]
--jobs N --json FILE`) injects each into its own copy of the tree and runs the gate section of its tool, which must fail. The unmutated copy has to be green and to run a check first; the exit status is 0 (all caught), 1 (a fault
survived: a check is missing, or the line is not needed) or 2 (the catalogue is stale or the baseline is red). `make mutants-check` verifies in seconds that every fault still applies; S45 pins it, so an edit of a tool that moves
the text a fault replaces fails the layout gate until the catalogue is repaired. All 252 are caught (measured on two cores: 46 minutes 50 seconds with `--jobs 2`, of which about 32 minutes are the 70 faults of the audit, each of which runs the audit on the small bank; a few minutes on the big machine).

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
scripts/08_soundness.sh          # the soundness evidence (gates, every mutant, fuzzing, the checker): hours, not part of run_all.sh; run it on the final code
scripts/09_audit.sh              # the statement audit (every statement against its intended meaning, the cautions, the ledger of second readers): minutes, not part of run_all.sh
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
* Whether a formal statement says what the contest problem says is checked against a description written by the same people, and read by a second person only where the ledger says so: `scripts/09_audit.sh` reports, for every problem, which checks hold and
  which statements a second reader has confirmed, and the paper may claim no more than its numbers (7.4).
* The prefix regime can be easy for problems with helper lemmas: use the helper-free column and the strict table.
* The synthetic proofs are backward chains: they can teach premise selection and term building, not forward reasoning, cases or induction. The scaling experiment measures whether they help; it does not assume it.
* Models are small and trained from scratch; nothing here says how the methods behave at the scale of language-model provers.
* The studies of the encoders and the sizes hold the recipe fixed: an encoder that needs another learning rate, or a width that needs more epochs, is disadvantaged; the tables say "with this recipe". A scaling exponent is an empirical
  slope of a log-log line over the range that was measured (a few points per seed, fitted per seed, the interval over seeds); it assumes no asymptote and says nothing outside that range.
* A learner from random weights is compared with the model-free baselines and with the same learner continued from behaviour cloning; it is not a claim about reinforcement learning in general, only about this environment, these goals and this budget.
* RL variants share hyper-parameters; none was tuned per method. PPO and expert iteration are compared at equal iterations and by cumulative probes (logs), not at equal wall-clock.
* The mutation catalogue (1 647 mutants) is the evidence that the gates bite; the full mutation run takes many hours and is not part of this pipeline: `scripts/08_soundness.sh` runs it with the gates, the fuzzers and the checker comparison and states what it shows (7.3); `python3 mutants/mutate.py PREFIX` runs a part.

## 10. Compute

`python3 src/python/experiments/exp_estimate.py PROFILE [--seeds 0-6] [--cores N] [--set k=v]` prints the estimate of a profile stage by stage. It multiplies the work of every run (training steps, probes) by
rates: those of the development machine (2 cores; `exp_estimate.RATES`: 55 training steps/s for the default network, 6 to 60 probes/s depending on the solver) or, once `scripts/06_benchmarks.sh` has run, the
ones it measured on YOUR machine (`RUNS/bench/bench.json`; on a GPU pass `--train-rate` with the rate the benchmark reports for `DEVICE=cuda`).
**These are estimates. The author has not run the `full` profile** (the pipeline was exercised on worlds of `smoke` size and smaller, and the rates come from those runs); run `scripts/smoke_test.sh` and
`scripts/06_benchmarks.sh` first and read the estimate again with your own rates.

| profile | seeds | runs | run-hours | core-hours | wall-clock on 8 cores, perfect packing |
|---|---|---|---|---|---|
| smoke | 2 | 73 (core experiment alone: 55) | 2.4 (2.2) | 2.4 (2.2) | 0.7 h |
| small | 3 | 135 (77) | 107 (57) | 179 (102) | 23 h (13 h) |
| full | 5 | 263 (122) | 1 044 (344) | 2 982 (1 111) | 373 h, about 16 days (139 h, six days) |
| full | 7 | 351 | 1 459 | 4 169 | 522 h |

`exp_estimate.py` prints the same by part of the experiments (core, arch, size, scratch). The full profile's core experiment is 1 111 core-hours; the encoders add 650, the widths 233 and
learning from random weights 988 (three times the iterations, five learners). A machine with many cores shortens the wall-clock, not the core-hours; `--set archs=() --set widths=() --set scratch=()`, or a shorter list in each field, drops what the machine cannot carry.

**One measurement.** The `smoke` profile has been run once from the first stage to the report on the 2-core reference machine (two runs at a time, other jobs competing for the cores): 52 training and evaluation runs
(data 7 min 25 s, pre-training 3 min 34 s, fine-tuning 3 min 4 s, reinforcement learning 19 min 58 s, evaluation 29 min 45 s; the run times of the runs added up), the two benchmarks (6 min 25 s) and the report
(10 s): 70 minutes of run time. The estimator with the rates `06_benchmarks.sh` measured there says 0.8 run-hours for the stages before the benchmarks (measured: 1.06) and 1.0 for the benchmarks (measured: 0.11); with its built-in
rates it said 2.2 for everything. So on this one profile it is within about a third on the training and evaluation stages and pessimistic on the benchmarks. It has not been checked on a larger profile.

`full`, in core-hours: reinforcement learning 797 (40 runs: PPO and the expert-iteration variants), evaluation 215, pre-training 64, fine-tuning 23, data 11, benchmarks 1 -- the RL runs and the evaluation of
the search solvers are 91 % of it. The cost is linear in the number of seeds (`--seeds 0-6` costs 1.4 times the five-seed training and evaluation, the data and the benchmarks once). To spend less: run the
`small` profile; fewer seeds; one method (`--target 'rl/ei_her/*'`); fewer iterations or probes (`--set rl_iters=100 --set eval_probes=500`, section 4); a GPU speeds up the network updates only (proof search stays on
the CPU).

**Memory.** The world of the whole bank takes about 1.2 to 1.4 GB per process when built and up to about 2.8 GB in the runs that were measured (`smoke`); forked workers share it. Pre-training on 60 000
theorems keeps about 3 GB of steps in memory. (The gate S50, which replays the whole bank's proofs, holds 1.5 to 2.8 GB itself and peaks at 4.6 GB while an independent-checker process runs -- measured here, 20 minutes in all; on an 8 GB machine do not run it next to another heavy gate -- `tests/run_gates.py` runs the heavy gates one at a time -- and keep `S50_E_MEM`, the memory one independent check may use (1 500 MB), below what is left.) Set `MEM_GB` to what the machine has; the scheduler runs fewer jobs at once rather than more.
**Disk.** Measured on `smoke`: about 30 MB per training run (checkpoints) and about 10 KB per generator attempt; extrapolated to `full` that is about 3 GB for its 90 training runs and several GB for the
450 000 generator attempts. Keep 20 GB free.

## 11. Troubleshooting

* `FAILED ...` in the runner's output: read `RUNS/logs/<id>.log`; fix; run the same script again (finished runs are skipped).
* Out of memory: lower `JOBS`, or `MEM_GB`; the world of the whole bank is about 2.5 GB per process and forked workers share it; pre-training on 60 000 theorems keeps about 3 GB of steps in memory.
* "holds runs made with another profile": the runs directory was made with other `SEEDS`/`SET`; use the same, or another `RUNS`.
* "is being run by process N": a previous invocation is alive; if not, remove `RUNS/.lock`.
* A table or figure is missing from the report: its inputs were not produced; `report.md` > Warnings says which.
