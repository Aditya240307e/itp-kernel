# docs -- reference documentation

Start with the [main README](../README.md); these documents go deeper. The first group describes the system as it is; the second group holds working notes on the optional learning experiments, which are not part of using the prover.

## The system

| document | what it covers |
|---|---|
| [`KERNEL_DESIGN.md`](KERNEL_DESIGN.md) | the design of the kernel and a dated log of how each feature was added and tested: hybrid variables (de Bruijn indices and context levels), parameters versus indices, the `Prop`/`Type` world and proof irrelevance, quotients, the work budget, each with the gate that covers it |
| [`ACTIONS.md`](ACTIONS.md) | the proof environment's action language: goals as data, the 21 kinds of action, the typed term builder, the portable form, the recorder |
| [`BANK.md`](BANK.md) | the problem bank: what a problem is, the regimes, how a formal statement is checked against the problem's words (`semantic.py`), reference proofs, how to add a problem |
| [`GENERATOR.md`](GENERATOR.md) | the synthetic theorem generator: what it makes, how, how the data are checked |
| [`FINDINGS.md`](FINDINGS.md) | the tracker of findings F1 to F91 -- bugs, weaknesses and design questions found by testing -- each with its status and the test that covers it ([`findings_tags.tsv`](findings_tags.tsv) is the machine-readable index) |
| [`ROADMAP.md`](ROADMAP.md) | what is missing between this system and olympiad-level and undergraduate mathematics, in the order the problems need it; a judgment, not a measurement |

Other places to look: [`tests/README.md`](../tests/README.md) lists every gate and how to run it, [`mutants/README.md`](../mutants/README.md) describes the mutation testing, [`src/c++/hybrid/README.md`](../src/c++/hybrid/README.md) the kernel's implementation, and [`examples/repros`](../examples/repros/README.md) reproduces three of the findings.

## Notes on the optional learning experiments

Working documents for the machine-learning side of the repository (the learner, the experiment pipeline, the synthetic-data runs). The prover, the libraries, the bank and the environment do not depend on any of it.

| document | what it covers |
|---|---|
| [`ARCHITECTURE_RL.md`](ARCHITECTURE_RL.md) | a design note: how a learner for this environment could be structured (imitation, search, expert iteration, hindsight, a manager/worker split) |
| [`TRAINING.md`](TRAINING.md) | the training pipeline and measurement protocol |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | the experiment matrix, the scripts that run it and the outputs |
| [`PAPER_PLAN.md`](PAPER_PLAN.md), [`PRIOR_WORK.md`](PRIOR_WORK.md) | notes towards writing about the project |
| [`SECOND_READER.md`](SECOND_READER.md) and [`statement_reviews.tsv`](statement_reviews.tsv) | the handout for a person who reads formal statements against their problems, and the ledger such readings are recorded in (empty so far) |
