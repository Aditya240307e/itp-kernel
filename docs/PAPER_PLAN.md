# PAPER_PLAN.md -- from the repository to the paper

Status: 2026-11-02. The plan for the TMLR paper, a **platform / resource paper**: a small trusted prover with a Gymnasium environment on which anyone can train their own models from scratch, two data sets, and the evidence
that the platform is fast, deterministic and sound. The paper is **typed by hand**; this file says which claim each sentence may make, where the number comes from, what is run in what order on the big machine, and what is
decided before the data exist. It is a working document: the parts marked *proposed* are mine and are yours to change **before the freeze (section 4, step R6)**; after the freeze they change only through the deviations log (section 9).

```
code (src/ tests/ mutants/) -- scripts/08_soundness.sh --> results/soundness   evidence about the VERIFIER      (make soundness; hours)
the bank (data/dataset_v2)  -- scripts/09_audit.sh ------> results/audit       evidence about the STATEMENTS    (make audit; minutes; second readers add to docs/statement_reviews.tsv)
code + data + seeds         -- scripts/run_all.sh -------> RUNS/ -> results/<profile>   speed, memory, and what LEARNERS do on it   (make smoke / the full profile; days)
the three folders + your .tex --> tools/check_numbers.py   every number, table and figure you cite exists, is current, and agrees across folders
```

Nothing in a results folder is typed in. A number in the paper is a macro (`\N{Name}` or `\numName`, defined in a folder's `numbers.tex`), a cell copied from a `tables/*.md|tex|csv` file, or a value read off a
`figures/*.csv`; `tools/check_numbers.py` (section 8) says which of your sentences disagree with the folders.

## 1. The paper in one paragraph

**Thesis (proposed wording).** Training a theorem prover with reinforcement learning needs an environment that is *trusted* (a wrong proof cannot be accepted), *fast and deterministic* (a run can be repeated and its cost
predicted), and *open to your own model* (goals are data, actions are a finite menu, rewards come from the kernel). We present one whose trusted part is small enough to read (`\numRepoKernelLines` lines of C++23 kernel,
`\numRepoBindingsLines` of bindings, `\numRepoCheckerLines` of independently written second checker), with a Gymnasium interface (`ITPBank-v0`), a benchmark of `\numBankProblems` olympiad-style formal problems
(`\numBankProved` with kernel-checked reference proofs) and a generator of kernel-checked synthetic theorems (`\numGenTotal` in the full set). We report how fast it is (probes per second per core, latency, memory,
worker scaling, determinism), how far the verifier has been tested (`\numSoundGates` gates, `\numSoundMutants` mutants, falsification fuzzing, differential testing against the second checker), what has been checked about
every formal statement, and what several kinds of learner (behaviour cloning, PPO, expert iteration with hindsight, three actors) and four encoders of the goal do on it from scratch over `\numNSeeds` seeds.

**What the reader should be able to do afterwards.** Install it and run an episode in minutes (`examples/`, `make smoke`); train a model of their own on `ITPBank-v0`; trust the verdict of the kernel *for the
statement as written* and know exactly where that trust stops; predict what an experiment will cost (`exp_estimate.py`); and compare against numbers made with five or more seeds and honest intervals.

**What the paper does not claim** (put these in section 1 and in the limitations; each is a way to be rejected for over-claiming):

* No state-of-the-art, no comparison with language-model provers, no claim that any learner or encoder is *the* best. The comparisons are demonstrations on one environment with a fixed recipe.
* No speed comparison with other platforms (Lean, Coq, Isabelle, Peano): none was run on the same statements and machine. Say "the numbers are those of this implementation on this machine".
* No claim that the kernel is *proved* correct. Say "tested": gates, mutants, falsification, differential testing; the kernel is small so that it can be audited, and the statement of a theorem is outside the verified chain.
* No claim that the benchmark reproduces official contest problems: the texts are paraphrases, some written from memory (`origin`, `verified`); the `\numBankAmc` AMC-style problems are written *in the manner of* that contest.
* No claim about scale (language-model-size networks, millions of theorems) or about real numbers, geometry, combinatorics (not in the library).
* No claim that a learner solves *olympiad* problems. Of the 40 provable problems of the test split 35 are of levels 1-3 (`bank.LEVELS`: from one arithmetic step to an olympiad *lemma*), 5 are of level 4 and none of level 5; the IMO tier (41 IMO, 18 ISL, 1 APMO) has proofs for 12 problems. What the experiments measure is learning on elementary to lemma-level problems; the upper tiers are an open challenge, and a statement-only problem that some run solves is a lower bound on solvability. Choose the title and the abstract accordingly ("towards olympiad problems", not "olympiad algebra and number theory").
* No claim from the size of the olympiad evaluation: dev has 20 problems and test 58, so a single run's 95% Wilson interval has a half-width of 6 points at 5% solved, 10 at 20% and 12.5 at 50%. Claim large effects, say the counts, use paired comparisons over seeds.
* No determinism claim for training on a GPU: determinism is claimed for the engine and the provers (the `outcome_digest` of `bench_workers`), not for CUDA kernels.

## 2. Claims, evidence, artifacts

Each claim is worded as far as the evidence goes and no further. *Weakens* says what result would force a narrower wording; *Tier* says whether the paper stands on it (1), is better with it (2) or can do without (3).
`results/full` stands for `results/<profile>` of the run the paper cites.

| ID | Claim (the strongest wording the evidence supports) | Evidence (what is run or counted) | Where the numbers are | Weakens it | Tier |
|---|---|---|---|---|---|
| C1 | A third party can install the platform and train a model of their own on the Gymnasium environment: the numbered examples run end to end, a smoke pipeline trains and evaluates every learner on a tiny world | `make examples` (gate S46); `scripts/smoke_test.sh`; a clean extract of `make dist` on a fresh machine: `scripts/00_check_env.sh`, `make kernel`, one example, timed | `bench_world` (build time, memory, parameters); wall-clock of the clean install (measure and quote it) | the clean install fails or needs a step the README does not list | 1 |
| C2 | The trusted base is small: the kernel and its bindings are `\numSoundKernelLines` + `\numSoundBindingsLines` lines; a second checker of `\numSoundCheckerLines` lines shares no code with it; everything else is untrusted and produces certificates | counted from the files (`exp_repo`, `exp_soundness` inventory); `docs/KERNEL_DESIGN.md` for what is trusted | `results/soundness`: `sound_trusted`; report table `system` | something in the Python layer decides acceptance (it must not: check the "trusted" list is complete) | 1 |
| C3 | The verifier has been tested hard: all `\numSoundGates` gates pass; `\numSoundMutantsCaught` of `\numSoundMutants` catalogue mutants are caught and `\numSoundMutantsEquivalent` are documented equivalent, none unexplained; falsification fuzzing found no false statement proved (`\numSoundFuzzUnsound` of `\numSoundFuzzStatements`); the kernel and the second checker agree on `\numSoundDiffProofs` proofs and `\numSoundDiffMutants` mutated proofs | `scripts/08_soundness.sh` at the final commit (status CLEAN) | `results/soundness`: `sound_summary`, `sound_gates`, `sound_mutants`, `sound_survivors`, `sound_fuzz`, `sound_differential`, `sound_findings`, `sound_open` | status FAILED or INCOMPLETE; a survivor without a reason; a finding that accepted what it must refuse and is open | 1 |
| C4 | Every proof an agent solved was replayed by the engine, kernel-checked, and re-checked by the second checker (`\numRecheckAccepted` of `\numRecheckChecked`) | `eval/*/recheck.json` | report table `soundness`; `\numRecheckAccepted`, `\numRecheckChecked` | any refusal (it is a bug report, not a number to hide) | 1 |
| C5 | Speed and cost: probes per second per core for each prover, decisions per second of the Gymnasium interface, latency percentiles of the operations, the split of the time (interface vs engine vs kernel), speed-up and memory with worker processes, cost of each encoder | `scripts/06_benchmarks.sh` alone on an idle machine | report tables `bench_machine`, `bench_solvers`, `bench_breakdown`, `bench_latency`, `bench_workers`, `bench_archs`, `bench_world`, `bench_kernel`; figures `fig_throughput`, `fig_breakdown`, `fig_latency`; `\numBenchProbesPerSSample`, `\numBenchGymDecisionsPerS`, `\numBenchProbeMedianMs`, `\numBenchSpeedupFour`, `\numBenchPssMbFour`, `\numBenchCores` | a busy machine at the start (the report warns); one machine only | 1 |
| C6 | The engine is deterministic: the rows of the in-process run and of every worker count agree field by field; the `outcome_digest` is machine-independent, so two machines can be checked against each other | `bench/throughput` determinism section; (best) the same digest from a second machine | `\numBenchDeterministic`, `\numBenchDeterministicTasks`; `bench_workers` | `NO`: then the claim is withdrawn and the cause is the finding | 1 |
| C7 | The benchmark: `\numBankProblems` formal problems (`\numBankProved` with a reference proof, `\numBankStatementOnly` statement-only), levels 1-5, split by problem family into `\numBankTrain`/`\numBankDev`/`\numBankTest`, with provenance labels, a near-duplicate screen and executable specifications | E0 (`data/dataset`); gates S49, S50 | report table `bank`; `docs/BANK.md`; `\numBankProblems` ...; `levels` | the splits are small (test: 58, dev: 20), so a problem is 1.7 points: say it | 1 |
| C8 | What was checked about each statement: `\numAuditChecked` of `\numAuditProblems` statements were read back from the kernel's term and compared with an independent description on `\numAuditPoints` points; `\numAuditMutantsCaught` plausible formalisation slips were all detected (`\numAuditMutantsEquivalent` declared equivalent, with reasons); `\numAuditConfirmed` (`\numAuditConfirmedPct`%) were read by a second person | `scripts/09_audit.sh`; the ledger `docs/statement_reviews.tsv` | `results/audit`: `audit_summary`, `audit_origins`, `audit_cautions`, `audit_reviews`, `audit_failures` | any failure; a reader finds a statement unfaithful or ambiguous; coverage of second readers is low (the paper then says so, with the number) | 1 |
| C9 | The synthetic set: `\numGenTotal` kernel-checked theorems, none equal to a bank statement (by digest or shape), with `\numGenSkeletons` distinct proof skeletons; a Good-Turing/Chao1 estimate of how many remain unseen (`\numGenChao`) | E5 (`data/synth_*`), the verification from the data alone | report tables `generator`, `fig_generator`; the data card | the skeleton space saturates early (say it); the proofs are backward chains (narrow) | 1 |
| C10 | Learning from the data: held-out synthetic loss falls with the number of training theorems with exponent `\numLawDataIid` (iid) and `\numLawDataComp` (compositional) over the measured range | E2e, `pre/n<N>` | report tables `laws`, `scaling`; figures `fig_laws`, `fig_scaling` | an exponent whose interval contains 0; fewer than 4 points; R squared low | 2 |
| C11 | Different encoders of the goal train on the same environment with the same recipe (`transformer`, `bigru`, `cnn`, `bow`): their imitation of held-out synthetic proofs, the olympiad problems solved by the learners that continue from each, their cost | E2d | report tables `arch`, `bench_archs`; figure `fig_arch`; `\numTestBcBigru` ... | the recipe is tuned for one encoder (say "with this recipe"); the bag of tokens matches the ordered encoders (then the data carry little order: report it) | 1 |
| C12 | Learners start from nothing: PPO and expert iteration from random weights, no demonstration, no pre-training, goals from the synthetic set (no proofs), reach `\numTestScratchEiHer`% on test, against `\numTestUniform`% for random actions | E2f | report table `scratch`; figure `fig_scratch` | they stay at the level of random actions (then the claim is "does not yet", and the training-side success curve is what is shown) | 1 |
| C13 | Several learners and actors run on the same environment: BC, PPO, expert iteration, hindsight, a bootstrapped library, MCTS, IDA\*, manager-worker; each is one command | E1, E3, E4 | report tables `main`, `strict`, `ablations`, `solvers`, `mw`, `library`; figures `fig_budget`, `fig_ablations`, `fig_solvers` | none: it is a statement that they run; their *ranking* is a separate, hedged result | 1 |
| C14 | Does synthetic pre-training help on olympiad problems? Does hindsight help? Does a bootstrapped library help? | E3 ablations (paired by seed, Holm over the family) | `ablations`; `\numAblPretrain`, `\numAblHer`, `\numAblBoot` with their `...P` | the test split is 58 problems: expect "not distinguishable" for effects below about 10 points; report the interval, never "no effect" | 3 |
| C15 | What an experiment costs, and the estimator's accuracy: predicted vs measured run-hours per stage | `exp_estimate.py` vs `RUNS/.status` | report table `cost`; `docs/EXPERIMENTS.md` section 10 | the estimator is off by more than 2x (say it) | 2 |

Priorities, if time or compute run short: **Tier 1 is the paper** (platform, trust, speed, data, one demonstration per kind of learner and encoder). C10 and C15 make it stronger. C14 is a result that may well be negative or
unresolved at this benchmark size; the paper must be written so that it stands without it.

## 3. Pre-registration (proposed; edit, then freeze with the tag of step R6)

**3.1 Frozen before the full run.** Profile `full`; seeds `0-4` (`\numNSeeds` = 5; `0-6` only if the core experiment finishes early); headline budget `\numHeadlineBudget` probes; the dev/test split of the dataset (salted hash of the
family; test is 58 problems); one recipe for all methods (epochs, learning rates, batch, iterations, probes, time limits: `RUNS/PROFILE.json`); checkpoint rule (behaviour cloning `best.pt` by dev loss, reinforcement learning `last.pt`,
nothing chosen by test); the list of studies that are run (`archs`, `widths`, `scratch` and their learners); the set of analyses below. The effective profile and the commit are recorded by the runner.

**3.2 Sealed test.** The olympiad test split is used only by the `eval/*` runs of the full profile. Pilots (step R5) run with `--set eval_splits=dev`. No recipe, checkpoint, hyper-parameter, prover or stopping rule is chosen by looking
at test; dev is for development and is shown only next to test. The seal is *procedural*, not cryptographic: the statements are public, and the leakage protocol (`docs/EXPERIMENTS.md` section 6) is what keeps our pipeline from training on them.

**3.3 Three kinds of result, named in advance** (so the text cannot drift to fit the data):

* **Confirmatory**: few, well-powered (thousands of held-out synthetic theorems), with a fixed direction. Tested; the decision rule is below.
* **Descriptive**: headline numbers with intervals (every method on the olympiad test and dev). No test; the interval is the result.
* **Exploratory**: everything else (all ablations on the olympiad test, widths, provers, per-level breakdowns). Reported with the Holm-adjusted p-values of `stats/comparisons.csv`, worded with the vocabulary of 3.5, never as findings.

**3.4 Hypotheses.**

| ID | Statement (direction is a prediction, not a hope) | Data | Test | Kind |
|---|---|---|---|---|
| H1 | The held-out synthetic loss falls with the number of synthetic training theorems N: the exponent of loss ~ N^e is negative, for the iid and for the compositional split | `pre/n<N>`, sizes of the profile | per-seed log-log line, 95% Student-t interval of the exponents over seeds (`exponent_ci`); needs at least 4 sizes | confirmatory |
| H2 | An encoder that sees the order of the symbols imitates held-out compositional proofs better than the bag of tokens: loss(bow) - loss(transformer), loss(bow) - loss(bigru), loss(bow) - loss(cnn) are positive | `pre/<arch>_n<max>` | paired by seed, two-sided t-test, Holm over the 3 contrasts | confirmatory |
| H3 | Every learner is above model-free random actions on the olympiad test at the headline budget: `bc`, `ppo`, `ei`, `ei_her` against `uniform` and `blind` | `eval/main`, `eval/baselines` | difference with the t-interval over seeds against the Wilson interval of the baseline; "supported" only if the intervals do not overlap | descriptive |
| H4 | Learners that start from random weights (`scratch_ei_her`, `scratch_ppo`) rise above random actions during training: the training-side success rate at the last iteration is higher than at the first by more than the interval over seeds; dev/test reported next to it | `rl/scratch_*` logs, `eval/scratch` | paired by seed, t-test on (last - first) | confirmatory for the log, descriptive for dev/test |
| H5 | The ablations of E3 (11 comparisons: pretrain, finetune, ei, ppo, her, boot, sub, algo, mw, mcts, ida) | `eval/main/*` | paired by seed (t-test, Holm over the 11) and paired bootstrap over problems | exploratory |
| H6 | The ranking of the encoders on held-out synthetic loss is the same for the iid and the compositional split | `arch` | Kendall rank agreement over the 4 encoders, per seed | exploratory |
| S1 | The in-process run and every worker count produce identical rows (but for the clock) | `bench/throughput` | `identical == yes` | must hold |
| S2 | Parallel evaluation with W workers has speed-up S(W) and efficiency S(W)/W as measured; the memory (PSS) of W workers is below W times that of one | `bench_workers` | reported | descriptive |
| K1 | No false statement is proved in any fuzz campaign; no proof the kernel refused is counted as solved; no exception escapes | `results/soundness` | counts are 0 | must hold |
| K2 | Every mutant of the catalogue is caught or documented equivalent | `results/soundness` | `unexplained == 0` | must hold |
| K3 | The kernel and the second checker agree on every proof of the comparison | `results/soundness` | `disagreements == 0` | must hold |
| K4 | No statement of the bank fails the audit; every slip is caught or declared equivalent with a reason | `results/audit` | `failed == 0` | must hold |

"Must hold" is not a hypothesis that the data may refute quietly: a failure is the first finding of the paper's soundness section, it is fixed in the code (which restarts the soundness and audit evidence, and the
experiments if the engine changed), and the history of the failure is told, not omitted.

**3.5 Words for results** (fixed now; the text uses these and no others for olympiad comparisons).

* **supported**: the 95% interval over seeds excludes 0 *after* Holm's correction **and** the paired bootstrap over problems excludes 0.
* **suggestive**: exactly one of the two intervals excludes 0, or the seed interval excludes 0 only before correction.
* **not distinguishable**: neither does. This is *not* "no effect": the paper quotes the interval and the smallest detectable difference (`stats/comparisons.csv`), and makes no equivalence claim (no equivalence margin was set).

**3.6 Power, stated honestly.** The test split has 58 problems; at a solve rate near 30% a single method's Wilson half-width is about 12 points, and a paired difference of two methods that disagree on 9 problems has a half-width
near 10. Differences below about 10 points on the olympiad test are therefore not resolvable by the bootstrap over problems, and the paper says so wherever it compares methods. That is why H1 and H2 (thousands of held-out synthetic
theorems) carry the confirmatory weight, and why the olympiad test is a *descriptive* result with its intervals.

**3.7 Analysis rules.** Replicates are whole runs from a seed; a crashed run is repeated with the same seed, never replaced by another; the number of repetitions is reported. The synthetic set is one set shared by all seeds. A
missing result is a warning at the top of `report.md`, never a silent gap. Every table and figure of the paper is produced by the pipeline from the runs; a number that is not is marked as such in the text.

## 4. Run order on the big machine

Each step has a **go** criterion. If it is not met, stop and read the log; do not continue to the next step with a failure in hand. Times are estimates at the development machine's rates (2 slow cores) unless marked
*measure*; `exp_estimate.py` re-reads them with your machine's rates after step R3 (`exp_estimate.py full --bench runs/smoke/bench/bench.json --cores N --seeds 0-4`; add `--train-rate X` once you know the rate of `DEVICE=cuda`).

| Step | Command | Time | Go if | If not |
|---|---|---|---|---|
| R0 | `scripts/00_check_env.sh`; note CPU, cores, RAM, GPU, disk | minutes | Python, the libraries and 20 GB of disk are there; `nproc` and memory match what you meant to use | fix the environment; the report records the machine |
| R1 | `make kernel && make gates` (all 54: `make gates-all`) | default gates about 45 min on 2 cores; all about 1 h; S50 alone 20 min and 4.6 GB | every gate passes, also the second time with `ITP_VERIFY_MEMO=1` if you run it | a failure is a bug on this machine (compiler, library versions): report it, do not run experiments |
| R2 | `scripts/01_build_dataset.sh` | about 25 min | every recorded proof replays; digests match | the dataset or the kernel differs from the commit |
| R3 | `scripts/smoke_test.sh` (and with `DEVICE=cuda` if the GPU will be used) | 70 min on 2 cores; *measure* | the report builds; every run of the graph finished (`scripts/status.sh`); no warning except those of a smoke profile; `\numBenchDeterministic` is `yes` | the pipeline or the GPU path has a bug on this machine; this is the cheap place to find it |
| R4 | `scripts/06_benchmarks.sh` on the idle machine (the smoke run already did one), then `exp_estimate.py full ...` | minutes; the estimate | the load average before the run is low; the estimate fits the time you have, with 30% slack; choose the studies (`--set archs=() widths=() scratch=()` drops them) | cut studies in this order: `widths`, then fewer `scratch` learners, then `archs` learners; never fewer seeds than 3 |
| R5 | (optional, **your decision**) a pilot on `small` with `--set eval_splits=dev` and one seed | the core experiment of `small` is about 34 core-hours per seed (estimate: 102 for three seeds), so a few hours on 8 cores; *measure* | the training-side success rate of expert iteration rises; `bc` is above `uniform` on dev; nothing crashes | adjust the **recipe** using train and dev only, write each change in the deviations log (section 9) |
| R6 | **Freeze**: commit; `git tag -a prereg-v1`; keep `RUNS/PROFILE.json`; no change of `src/`, `tests/`, `mutants/` or the dataset afterwards | -- | `git status` is clean; this file's sections 3 and 4 are as you want them; the second readers' verdicts (R10) are in, or you decided to freeze without them and the paper says so | -- |
| R7 | `scripts/02_generate_synthetic.sh` (part of `run_all.sh`) | about 10 core-hours for 450 000 attempts at 12 per second and core | exactly the target number of train theorems; the verification from the data alone passes; `fig_generator` has a saturation curve | a shortfall at three times the attempts is an error by design: look at the exclusions |
| R8 | `PROFILE=full SEEDS=0-4 JOBS=... DEVICE=... scripts/run_all.sh`, studies in this order of value per cost: core, `archs`, `scratch` (start with `ei_her` and `ppo` on the transformer), `widths` | core about 1 100 core-hours, `archs` 650, `widths` 233, `scratch` 988 (estimates); resumable | `scripts/status.sh`: no failed run; the report's Warnings section is empty | a failed run blocks only what needs it; fix, rerun the same command |
| R9 | `make soundness J=8` **at the frozen commit, not while R4 or R8 run** | hours (the mutation catalogue dominates) | status CLEAN | FAILED is a finding (section 3.4, K1-K3); INCOMPLETE says what is missing, run again (it resumes) |
| R9b | `make tool-mutants J=8` at the frozen commit (it injects faults into the tools that make R9's and R10's evidence, so it must run on the same tree) | 47 min on 2 cores (measured, 252 faults), a few on the big machine | exit 0: every fault caught | a survivor is a check missing in a tool that makes evidence (finding F88): add the check (this changes `tests/`, so redo R6 and R9), or remove the line the fault re-breaks |
| R10 | **start it before R6, the readers need days:** `make audit`; `python3 tools/review_packs.py results/audit --readers 4 --copies 2 --out review_packs` (the 58 test statements, each read by two people, about 2.4 hours per reader; `docs/SECOND_READER.md` is the handout and is copied into each pack); the filled `reader_K.csv` come back and go through `exp_audit.py import-reviews` | 6 min to build; readers: about 5 min per statement | status CLEAN; the coverage you reached is a number of the report | a reader's "unfaithful" is a bug in the bank: fix the statement (this changes the dataset: redo R2, and the runs that used it, which is why this step comes before the freeze) |
| R11 | `scripts/07_report.sh`; `exp_bundle.py --verify` on each folder | minutes | the manifests verify; the report shows one code tree | mixed code versions are a warning: do not publish such a folder |
| R12 | `python3 tools/check_numbers.py YOUR.tex --results results/full --results results/soundness --results results/audit --strict` | seconds | exit 0 | it names each macro, table or figure that is missing, stale or inconsistent |
| R13 | Archive: `make dist` (GIT=1 for the history), copy the three folders, the digests | minutes | the archive extracts and passes `make examples` | -- |

Two operational rules. **Docs and paper edits are free**; edits of `src/`, `tests/`, `mutants/` or the dataset manifest change the tree digest, which makes the soundness and audit evidence stale and the runs inconsistent.
**Do not run anything else on the machine during R4 and R9's timing-sensitive parts**: throughput numbers measure the other jobs too.

## 5. The paper, section by section (what to type, where the numbers are)

A reordering of the skeleton in `paper/main.tex` (the generated tables do not depend on the order): platform first, evidence next, learners last. Page budget is advisory; TMLR sets no limit but asks that length be justified.

| § | Content | Numbers and artifacts | Pages |
|---|---|---|---|
| Abstract | the platform, the two data sets, the evidence of speed and soundness, one sentence on what learners do (including the clearest negative result) | `\numRepoKernelLines`, `\numBankProblems`, `\numGenTotal`, `\numSoundMutants`, `\numNSeeds`, `\numTestBc`, `\numTestUniform` | 0.3 |
| 1 Introduction | the problem (RL for provers needs an environment you can trust, afford and plug a model into); contributions C1-C13 as five bullets; scope and non-claims (section 1 above) | claims table | 1.0 |
| 2 Related work | environments (HOList, CoqGym, GamePad, LeanDojo, Peano, INT), benchmarks (miniF2F), neural provers and RL at scale, synthetic data and hindsight; a **feature matrix** if you want one: only entries you verified from the system's own paper or README, with the date (start from `docs/PRIOR_WORK.md`) | `paper/main.bib` (verified entries); no speed comparison | 0.8 |
| 3 The platform | kernel and second checker, what is trusted (C2); the untrusted layer with certificates (`ring`, `linarith`, `nlinarith` emit kernel-checkable terms); the fuel meter (a wrong action fails after a bounded work, deterministically); the Gymnasium interface: goal as tokens, menu of legal options, typed term building, probes as the unit of cost, hindsight goals; a 10-line training loop (`examples/06_gymnasium_env.py`) | table `system`; `sound_trusted`; Figure 1 (draw it yourself: the stack and the trust boundary); Listing 1 | 2.5 |
| 4 Data | the bank (C7): composition, levels, provenance, splits and leakage, statement-only tier; the statement audit (C8); the synthetic set (C9): construction, exclusions, diversity and saturation | tables `bank`, `levels`, `generator`, `audit_summary`, `audit_origins`, `audit_cautions`; figure `fig_generator`; data cards | 2.0 |
| 5 Speed and cost | throughput, latency, where the time goes, workers and memory, determinism, cost of encoders and widths (C5, C6, C15) | `bench_*`, `cost`; `fig_throughput`, `fig_breakdown`, `fig_latency` | 1.5 |
| 6 Soundness | gates, mutants, falsification, differential testing, findings (C3); every solved proof re-checked (C4); what none of it shows | `sound_summary`, `sound_mutants`, `sound_survivors`, `sound_fuzz`, `sound_differential`, `sound_findings`; table `soundness` | 1.5 |
| 7 Learning on the platform | the protocol (tasks, probes, budgets, seeds, statistics, vocabulary 3.5); the methods; results in the order: baselines, one learner per kind, encoders (C11), from scratch (C12), scaling (C10), then ablations as exploratory (C14) | `main`, `strict`, `arch`, `scratch`, `laws`, `ablations`; `fig_budget`, `fig_arch`, `fig_scratch`, `fig_laws`, `fig_ablations` | 3.0 |
| 8 Limitations and threats | section 7 of this file, condensed | -- | 0.8 |
| 9 Conclusion; reproducibility; broader impact | what a reader can run; the archive and its digests; TMLR asks for a broader-impact statement only when the work carries a significant risk of harm | `MANIFEST.sha256` of the folders | 0.5 |
| Appendix | the configuration (`hyper`), the benchmark list (`problems.csv`), the cautions per problem, the mutation catalogue summary, the review procedure, additional figures (`fig_bc`, `fig_levels`, `fig_library`, `fig_solvers`, `fig_learning`) | `hyper`, `per_problem/`, `stats/` | free |

How to type a number: use the macro (`\N{TestBc}`), or copy a cell from `tables/*.md` and run `tools/check_numbers.py --literals`, which reports a typed number that equals a macro's value (it should be the macro) so that
it cannot drift when the runs are redone. The macro names are in each folder's `numbers.tex`; `tools/check_numbers.py --list PATTERN` prints them with their values.

## 6. Where each figure and table lives

| Needed for | File (under `results/full/` unless stated) |
|---|---|
| Every table | `tables/<key>.md` (read), `.tex` (booktabs, `\input` it), `.csv` (plot your own) |
| Every figure | `figures/<name>.pdf` (vector), `.png`, `.csv` (the exact values drawn: panel, series, kind, label, x, y, intervals) |
| Every outcome | `per_problem/outcomes.csv`, `by_problem.csv`, `by_family.csv` |
| Every comparison | `stats/comparisons.csv` (paired differences, intervals over seeds and over problems, Holm, detectable difference), `stats/rates.csv`, `stats/benchmark.csv` |
| Provenance | `provenance/` (profile, graph, every run: commit, host, seconds; machine; dataset digests; raw `bench.json`) |
| Soundness | `results/soundness/` (`README.md` first; `tables/sound_*`, `stats/*.csv`, `evidence/`) |
| Statement audit | `results/audit/` (`README.md` first; `tables/audit_*`, `review_sheet.md`, `stats/*.csv`) |
| Figures to draw by hand | the stack and trust boundary; one worked episode (goal, menu, chosen option, closed goal); the audit procedure |

## 7. Threats to validity (condense into the paper's limitations)

| Kind | Threat | What the repository does about it | What remains |
|---|---|---|---|
| Internal | few seeds; training noise | `\numNSeeds` seeds, Student-t intervals, paired tests; the number of repetitions of a crashed run is reported | five seeds give wide intervals by construction |
| Internal | bugs in the learners | gates S44 and S51 (tests of every learner, protocol and statistic); the independent re-check of every solved proof | a bug that only shows at scale |
| Internal | bugs in the tools that make the evidence (the soundness folder, the audit, the number checker, the record of the code) | their tests are tested by fault injection: `make tool-mutants` (finding F88: of the first 172 faults 21 got past the tests written for the tool; all are now caught; `mutants/README.md`) | the faults are written by the author, as the kernel's mutants are: they measure the tests against a hand-written list |
| Internal | the recipe favours one method | one recipe for all; no per-method tuning (stated); the studies hold it fixed | "with this recipe" must stay in the wording |
| Internal | hidden dependence on the machine | determinism check across worker counts; the machine and the load are recorded | one machine; GPU training is not deterministic |
| Construct | "solved" is a proof within a probe budget | every solved proof is replayed, kernel-checked, re-checked; probes are counted whoever asks (refused tries cost as much) | probes are not seconds: both are reported |
| Construct | a proof of the wrong statement | `results/audit` (specification, slips, provenance, second readers) | the specification is written by the authors; only second readers cover a shared misreading |
| Construct | the prefix regime gives helper lemmas away | `free` column and the `strict` table | ten problems only |
| External | small benchmark, narrow domain | stated; the synthetic set and the held-out synthetic splits give well-powered measurements | olympiad-level claims are descriptive |
| External | small models, no language pre-training | stated | nothing about scale |
| Conclusion | multiple comparisons | Holm over the family; exploratory results named as such | many comparisons on 58 problems |
| Conclusion | the test split informed a choice | sealed by procedure (3.2); pilots run on dev | trust in the procedure |
| Reproducibility | cost and time | the estimator; resumable runs; digests | the full profile has not been run by the author of the code: the first run on the big machine is the test of the estimate |

## 8. Submission checklist

TMLR facts below were read from the journal's pages on 2026-11-02 (author guide, acceptance criteria, editorial policies: jmlr.org/tmlr); read them again before submitting.

* **Authorship.** The authors answer for every claim, and in September 2026 TMLR's editors reported a flood of submissions and stricter desk rejection (reported to include asking authors to explain their papers). Read `docs/AUTHOR_GUIDE.md` and pass its self-test (section 12) before you submit; state the assistance of an AI in the acknowledgments as the skeleton does.
* **Evidence over novelty.** TMLR's first question is whether the claims are supported by accurate and convincing evidence; it does not require a new state-of-the-art result or a novel method, and asks whether some of its audience would be interested. Where a claim outruns the evidence, narrow the claim or add the experiment. This file's claims table is that discipline.
* **Format.** The PDF is made with the TMLR style file (`paper/tmlr.sty`, unchanged), double-blind: no author names, no link that names the authors, anonymised code and data (supplementary material up to 100 MB, PDF or ZIP, is allowed; reviewers may not read it). No page limit, but the length must be justified. The appendix follows the references.
* **Statements.** A broader-impact statement is required only when the work carries a significant risk of harm (this work does not, say why in two sentences). LLMs may be used as general-purpose assistive tools; the authors are fully responsible for the content and an LLM is not an author: state the assistance in the acknowledgments (after acceptance) as the skeleton does. No expanded version of a conference paper.
* **Numbers.** `tools/check_numbers.py YOUR.tex --results results/full --results results/soundness --results results/audit --strict` exits 0: no undefined macro, no missing table or figure, no stale or failed folder, no disagreement between folders, no TODO left.
* **Claims audit.** Every sentence in the abstract, introduction and conclusion that states a result has a row in section 2 and cites a macro or a table. Read the abstract against the section 1 non-claims.
* **Status of the evidence.** `results/soundness` and `results/audit` are CLEAN for the *submitted* tree (the folders print the tree digest; `check_numbers.py` compares it with the code you ship). If either is INCOMPLETE, the paper says what is missing.
* **Release.** Decide the licences (code and data are not licensed yet in this repository; the contest texts are paraphrases and the dataset card says so), tag the commit, archive with `make dist`, publish an anonymised copy for review and the real one after acceptance. Include the data cards, the digests, `docs/statement_reviews.tsv` and the three results folders.
* **Reproducibility statement.** One command per stage (`scripts/*.sh`), the seeds, the profile, the machine, the digests, the time of the runs, what was and was not run on a GPU.
* **Decisions only you can make** (listed so they are made on purpose): the licences; the authors, title and the name of the platform; whether to run a Lean or other comparison (none is planned; if you do it, run both systems on the same statements on the same machine and say what it covers); how many second readers and who; seeds 5 or 7; which studies the budget allows; whether to implement more generator families (the generator's proofs are backward chains: the paper says so) before the full run.

## 9. Deviations log and freeze record

The freeze record is filled at step R6; the log is append-only (never edit a past row). A deviation is anything that differs from sections 3 and 4 after the freeze, and every change of the recipe before it.

| Field | Value |
|---|---|
| Tag | `prereg-v1` |
| Commit | |
| Tree digest (`python3 tools/check_numbers.py --tree`) | |
| Effective profile (`RUNS/PROFILE.json`, sha256) | |
| Date, who froze it | |

| Date | Step | What changed | Why | Effect on the claims (which ID) |
|---|---|---|---|---|
| | | | | |

## 10. Reviewer-risk register (what a careful reviewer will ask, and the answer in the repository)

1. *"The benchmark is tiny."* A platform paper: the benchmark is the part that grows with the user; the synthetic set (`\numGenTotal`) and its held-out splits are large. The olympiad test is descriptive and its size is stated.
2. *"No comparison with Lean/Peano environments."* Stated as a non-claim; the difference is in what is trusted, in the action interface and in the cost accounting, not in a speed race. A feature matrix from verified sources, if included. `docs/PRIOR_WORK.md` records what was read (Peano, Minimo, INT, LeanDojo, Pantograph, Kimina, PVSGym), with `?` where it could not be verified; two consequences: hindsight is not a novelty (Minimo), and INT is the nearest prior art for a fast lightweight assistant with a synthetic generator.
3. *"Learners barely solve anything."* The claims are that they run, learn above random actions on a measured scale, and that the cost is known; the clearest negative result is in the abstract. Levels and the proved/statement-only split show where the difficulty is.
4. *"Soundness is tested, not proved."* Yes, and the paper says "tested"; the trusted base is counted; the checker is independent in code, not in authorship; mutation testing measures the gates against a hand-written catalogue (and the tools that make the evidence are tested the same way, F88).
5. *"Are the formal statements right?"* The audit says what was checked and who read it, with the coverage as a number, not a promise.
6. *"Is the synthetic data leaking the benchmark?"* Exclusion by digest and by shape, the statistics of what was excluded, the near-duplicate screen with manual review, split by family.
7. *"Can I reproduce it?"* One command per stage, digests, determinism evidence, the estimate of the cost and the measurement against it.
8. *"Hyper-parameters were not tuned."* Stated in the limitations; the recipe is the same for every method; the result is "with this recipe".
