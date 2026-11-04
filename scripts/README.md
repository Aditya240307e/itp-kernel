# scripts -- shell drivers

Thin wrappers around the Python tools in [`src/python/experiments`](../src/python/README.md#optional-packages). Each one reads its settings from the environment, from `scripts/local.env` (yours, not tracked by git; start from [`local.env.example`](local.env.example)) and from command-line options, builds the kernel when it needs it and `build/` has none, and can be run again after an interruption: finished work is not repeated. They change to the repository root first, so they run from any directory. Most of them are reached through `make`.

## Evidence about the prover

| script | `make` | what it does |
|---|---|---|
| [`08_soundness.sh`](08_soundness.sh) | `make soundness` | the **soundness folder** (`results/soundness`): every gate, every mutant of the catalogue, the falsification campaigns, the kernel against the independent checker, the inventory of the trusted code, the classification of the findings; then the tables, the numbers, a README and a manifest |
| [`09_audit.sh`](09_audit.sh) | `make audit` | the **statement audit** (`results/audit`): for every problem of the bank, the checks of its formal statement against the intended meaning, its provenance, the cautions, the state of the second readers' ledger, and the review sheet a person fills in |

Both write a stamped folder ([`results/README.md`](../results/README.md)): a record counts only for the code it ran on (the digest of the tree), so a record of other code is listed as stale and run again. Exit status 0 is `CLEAN`, 1 `FAILED` (a claim is contradicted or a check fails), 2 `INCOMPLETE` (evidence is missing or stale).

```bash
make soundness                                          # everything: hours (the gates alone about an hour on two cores; J=8 for a bigger machine)
STAGES=gates,fuzz ./scripts/08_soundness.sh             # some stages: gates, mutants, fuzz, differential, inventory, findings
MUTANTS="GE1 GQ" ./scripts/08_soundness.sh --jobs 8    # only the mutants whose names begin with these

make audit                                              # the whole bank: minutes
ONLY="imo1959_p1 imo1964_p1a" ./scripts/09_audit.sh     # some problems (the folder then says it is incomplete)
```

| setting | meaning | default |
|---|---|---|
| `SOUNDNESS`, `AUDIT` | the output folder | `results/soundness`, `results/audit` |
| `STAGES`, `MUTANTS` | stages to run; name prefixes of the mutants to run | all |
| `MUT_GATES` | run the mutants against these gates only (quick; the folder says it is incomplete) | all gates |
| `MUT_FULL=1` | run *every* gate against each mutant, to count how many catch it (slower) | off |
| `FUZZ_SEEDS`, `FUZZ_N`, `FUZZ_SECONDS` | seeds of the falsification campaigns; statements per campaign; seconds the search may take per statement | `SEEDS` (0-4), 300, 5 |
| `DIFF_PER_PROOF` | mutated proofs per tactic proof in the comparison with the independent checker | 1000 |
| `ONLY`, `REVIEWS` | problems to audit; the ledger of second readers | all, `docs/statement_reviews.tsv` |
| `REDO=1` | run again what already has a record | off |

After a second reader has filled in `review_sheet.csv`: `python3 src/python/experiments/exp_audit.py import-reviews results/audit/review_sheet.csv --reviewer NAME`, then `python3 src/python/experiments/exp_audit.py report`.

## The learning experiments (optional)

These drive the learner and the synthetic-data pipeline; they need `torch` (and `gymnasium`), and a GPU is worth having for the large profiles. The prover, the libraries and the bank do not depend on them.

| script | stage |
|---|---|
| [`00_check_env.sh`](00_check_env.sh) | is this machine ready (`tools/check_env.py`) |
| [`01_build_dataset.sh`](01_build_dataset.sh) | the kernel and the problem dataset (replays every recorded proof of `data/dataset_v2`: about 25 minutes, 3 GB) |
| [`02_generate_synthetic.sh`](02_generate_synthetic.sh) | the synthetic theorem set: generate (parallel, resumable by shard), merge with the exclusions, verify from the data alone |
| [`03_pretrain_finetune.sh`](03_pretrain_finetune.sh) | behaviour cloning: pre-training on synthetic theorems, fine-tuning on the bank's train split |
| [`04_reinforcement_learning.sh`](04_reinforcement_learning.sh) | PPO and expert-iteration learners that continue from the cloned network of the same seed |
| [`05_evaluate.sh`](05_evaluate.sh) | every model on the dev and test splits; each proof is replayed by the engine, checked by the kernel and re-checked by the independent checker |
| [`06_benchmarks.sh`](06_benchmarks.sh) | throughput and memory: world build, the Gymnasium interface, every solver, worker scaling, kernel micro-benchmarks (run it alone on an idle machine) |
| [`07_report.sh`](07_report.sh) | tables, figures, numbers and statistics, in `runs/<profile>/report` and a copy in `results/<profile>` |
| [`run_all.sh`](run_all.sh) | everything above in dependency order with one scheduler, then the benchmarks and the report |
| [`smoke_test.sh`](smoke_test.sh) | the whole pipeline on a tiny world (17 problems, 300 synthetic theorems, 2 seeds; about 70 minutes on two cores): it checks that every stage runs on this machine, and the numbers mean nothing |
| [`status.sh`](status.sh) | what is done, running or failed, and the tail of the newest log |

```bash
PROFILE=small SEEDS=0-2 ./scripts/run_all.sh
./scripts/run_all.sh --profile full --seeds 0-4 --device cuda --jobs 4 --set rl_iters=20
TARGET='rl/ei_her/*' ./scripts/04_reinforcement_learning.sh        # one variant
```

The settings are those of [`env.sh`](env.sh) (`./scripts/env.sh` prints them; `--help` on any script lists the options): `PROFILE` (`smoke`, `small`, `full`: how big the experiments are), `SEEDS` (`0-4`, `0,1,2`), `RUNS` and `RESULTS` (where the runs and the report go; default `runs/$PROFILE`, `results/$PROFILE`), `JOBS`, `CORES` and `MEM_GB` (what the scheduler may use; default the cores, 85% of the memory), `DEVICE` (`cpu`, `cuda`: where networks are trained; proof search always runs on the CPU), `WORKERS`, `SET` (profile fields, e.g. `"rl_iters=20"`), `TARGET` (only these runs and what they need) and `PY`. A runs directory belongs to one profile, seed list and `SET`: the runner refuses to mix results made with different settings. The experiment matrix and its outputs are described in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).
