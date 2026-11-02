# AUTHOR_GUIDE.md -- what the person who signs the paper must be able to explain

**Why this file exists.** The authors answer for every claim of a submission, whoever or whatever helped to build the system or to write the text; TMLR lets LLMs be used as assistive tools and says that
submissions that look largely LLM-generated are examined closely (editorial policies, read 2026-11-02). In September 2026 TMLR's editors also reported a flood of submissions and a stricter desk rejection, and
are reported to have asked the authors of ten papers slated for desk rejection to explain them (one team withdrew at once; a headline elsewhere says three could not, which I could not verify). Most of this repository was written with an AI assistant. That is fine, and the
acknowledgments of the skeleton say it; what is not fine is a signature under something nobody can defend. This guide is the shortest honest path from "I have the repository" to "I can defend each claim".

**How to read it.** The map (sections 1-3) takes an hour. The rest takes a weekend with the code open. Run the commands instead of believing the text: they are quick, and a claim you have seen fail and pass
is a claim you can explain. Every section ends with *Read* (where to look) and *Ask yourself* (what you should be able to say without notes). Section 11 is a self-test.
`docs/PAPER_PLAN.md` says WHAT the paper claims and where its numbers come from; this file says WHAT THE SYSTEM IS, so that those claims mean something to you.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1. The system on one page

| layer | what it is | trusted? | where | what makes it believable |
|---|---|---|---|---|
| **kernel** | a typed lambda calculus with inductive families: it checks that a term has a type, admits definitions and inductive types, computes with a work meter | **yes** (the only trusted code, with its Python bindings) | `src/c++/hybrid/itp_kernel.hpp` (7.5 k lines), `itp_bindings.cpp` (0.8 k) | the four kinds of evidence of section 3 |
| **independent checker** | a second implementation of the typing rules, in Python, on terms with named variables; it shares nothing with the kernel | no: it is an alarm, not a judge (a disagreement is a bug in one of the two) | `src/python/checker/` (0.85 k lines) | it agrees with the kernel on every library definition, on tactic proofs and on thousands of mutated proofs |
| **tactics, libraries** | `intro`, `apply`, `rewrite`, `cases`, `induct`, `ring`, `linarith`, ...; Peano arithmetic to elementary number theory | **no**: a tactic only builds a term, and the kernel checks the term | `src/python/authoring/`, `src/python/library/` | whatever they build is re-checked, so a wrong tactic wastes time and cannot make a false statement provable |
| **environment** | the learner's world: goals as data, menus of legal actions, actions that *say* their term arguments, probes that try an action and rewind | no (but "proved" only ever comes from the kernel) | `src/python/environment/` (`gym_env`, `gym_acts`, `gym_build`), `src/python/learner/bank_env.py` (`ITPBank-v0`) | gates S40 and friends, 1647 mutants, fuzzing |
| **data** | the problem bank (268 formal problems), the synthetic theorem generator, the splits | no, but a statement is the one thing a kernel cannot check: it checks proofs OF a statement, not that the statement says what the contest said | `src/python/problems/`, `src/python/synth/`, `data/dataset_v2/` | the statement audit (`make audit`), second readers (none yet) |
| **learner** | networks and provers: behaviour cloning, expert iteration with hindsight, PPO, search | no | `src/python/learner/ml_*.py` | gates S44, S47, S51; the kernel checks every proof |
| **experiments and evidence tools** | the pipeline that makes the numbers, and the tools that count them | no, but a wrong number is a wrong claim | `src/python/experiments/`, `tools/check_numbers.py`, `scripts/` | S51 and S45 gates; faults injected into the tools (`make tool-mutants`, finding F88) |

The sentence the design rests on: **nothing above the kernel has to be believed.** A tactic, a search or a neural network can be wrong in any way it likes; a finished proof is a term, and the kernel decides
whether the term has the stated type.

*Read:* `README.md`, then the table above against `src/` and `tests/`. *Ask yourself:* which lines are trusted, and what happens if a tactic is wrong? What is the one thing the kernel cannot check?

## 2. The kernel

**What it checks.** Dependent types; inductive families with parameters and indices; structural recursion; a `Prop`/`Type` world with definitional proof irrelevance and restricted elimination out of `Prop`;
quotients (five kernel constants and one reduction rule; `funext` is then a theorem); four classical axioms as an optional layer (`docs/ROADMAP.md`, "Where we are").

**How terms are represented.** Bound variables are de Bruijn *indices*, variables of the context (hypotheses) are *levels*: the "hybrid" of the name (`docs/KERNEL_DESIGN.md`, first page). Terms are
*hash-consed*: equal subterms are one node of a DAG, so memory is proportional to the number of distinct subterms and equality of nodes is pointer equality. Evaluation is *normalisation by evaluation*
(`eval`, `quote_at`, `normalize`, `conv`): terms are evaluated to values and read back.

**The work meter.** Every call into the kernel runs under a budget of *steps* (`fuel`, `ITP_STACK_CHECK`); when it is spent the kernel throws `FuelExhausted`. Three properties matter and you should be able to
say why each is wanted: (1) **fail-closed**: a spent budget is never read as "the answer is no", it is an error, and the environment never turns it into "no goal here" or an empty menu (finding F51 was exactly
that mistake); (2) **deterministic**: steps are counted, not seconds, so a result is the same on every machine and every run, which is what makes the cost of a learning episode reproducible; (3) **per call**:
each call into the kernel gets its own budget (the engine's budget for an action is far above what any legitimate call needs: the largest call over 934 library proof terms used 1/1500 of it).

**What it is not.** No universe polymorphism or cumulativity (a checked instantiation of universe levels is the nearest thing), no structures with eta, no mutual or nested inductives, no well-founded
recursion, no real numbers, no Mathlib; `Nat` is unary, so concrete large numbers are constructors (the binary `N` and the integers `Z` are libraries on top). `docs/ROADMAP.md` ("What is missing") lists them with an effort. A reviewer who asks "can it do X from Lean" is usually asking about one of these; the honest answer is no, and why that is not the point.

*Read:* `docs/KERNEL_DESIGN.md` (skim: it is a log of spikes; the first page and the "Contracts and known limitations" of S4 are the ones to read), the header comment of `itp_kernel.hpp`,
`examples/01_prove_with_tactics.py` (run it: it proves, refuses a false identity and a forged proof). *Ask yourself:* what is hash-consing for? What does a spent budget turn into, and why not "false"?

## 3. Why believe the kernel: four kinds of evidence, each with its limit

1. **A frozen reference kernel and lockstep gates.** The previous generation of the kernel is kept byte-identical (checksums) and the new kernel is compared with it on random terms and on the library:
   zero-tolerance oracles (S1-S4, `tests/support/harness.py`). *Limit:* it shows the two agree, and the new one was written by looking at the old one.
2. **The independent checker** (`indep_checker.py`, `recheck.py`): every library definition, tactic proofs, thousands of mutated proofs, in both modes (S8), and the bank's reference proofs (S50 judges every
   eighth by default, all of them with `S50_E=all`, which takes more than an hour). *Limit:* independent in
   code, not in authorship: the same person wrote both, so a misunderstanding of a typing rule could be shared.
3. **Mutation testing** (`mutants/`): 1647 single-line faults, each re-breaking one rule in an isolated copy of the tree; the gates must fail; 22 are documented as equivalent (no input can tell them from the
   original). *Limit:* the catalogue is hand-written, so it measures the gates against the faults somebody thought of; and the kernel, tactic and environment series were run to completion while they were
   written, the newest series only partly; `make soundness` (hours) produces the run for the tree you submit. There are no kernel-style mutants for the bank, the learner, the generator or the pipeline; the
   tools that make the paper's evidence have their own catalogue (252 faults, finding F88).
4. **Falsification through the environment** (`tests/fuzz/gym_fuzz.py`): statements that are FALSE at some point are searched for proofs. A proof of one would be a kernel bug (UNSOUND); a proof the kernel
   refuses is a tactic bug (REJECTED). *Limit:* finite search; the campaigns are listed with their seeds in the soundness folder.

Behind these there are 54 gates and an 89-row findings log (`docs/FINDINGS.md`): the log is itself evidence, because it shows that the process finds bugs (F51, F86, F87 and F88 are good ones to read).

**What the paper may say:** "tested", "no unsoundness known", "the trusted base is 7.5 k lines of C++ plus 0.8 k of bindings", "the independent checker agrees on N proofs", "every seeded fault is caught
except 22 documented equivalents". **What it must not say:** "proved sound" or "verified". Nothing here is a mechanised proof; the statement "a proof accepted by the kernel proves its statement" rests on the
evidence above.

*Read:* `mutants/README.md`, three rows of `mutants/catalogue.py` (what a mutant looks like), `docs/FINDINGS.md` F51/F87/F88, `docs/EXPERIMENTS.md` section 7 (the soundness folder).
*Ask yourself:* why is 0 survivors not a proof? What would a mutant that survives mean? What does "independent" mean here, and what does it not mean?

## 4. The environment: what an agent sees and does

* **Goals are data**, not text. An observation is the goal as a term tokenised into symbols (`features`), plus the menu of legal options; "there is no text interface anywhere" (`bank_env.py`).
* **A menu of options** lists the legal actions at a goal (apply this lemma, rewrite with that hypothesis, case-split, induct with this principle, ...). Actions that need a term (a witness, a lemma
  instantiation, a claim) SAY it: the term is built by a typed builder, so it is well typed by construction and the network chooses among constructions rather than writing characters.
* **A probe** tries an action and rewinds (the kernel's probe/rewind interface). Probes are the unit of cost: a budget is "B probes", not seconds, so results are comparable across machines.
* **Reward**: `solve_reward` (1.0) when a proof that closes every goal has been checked by the kernel; nothing else is rewarded by default (`bank_env.py`).
* **Hindsight**: every goal an attempt closed on the way is a theorem with a proof; it is replayed in a fresh engine and kernel-checked, then becomes a training example (`ml_her`) and, if it is a
  train-side theorem, may enter the library as a lemma (`ml_lemmas`; never one that says what a dev or test problem says).
* **Premises**: a problem is attacked inside the world that precedes it: the library's globals, and in the `prefix` regime its own helper lemmas; in the `strict` regime the helpers are hidden and the
  agent has to find the ideas. The blocks of the other problems of the bank are masked (`docs/BANK.md` section 2; S50 tests the mask).

*Read:* `docs/ACTIONS.md`, the header of `src/python/learner/bank_env.py`, `examples/03_bank_problem.py`. *Ask yourself:* why probes instead of seconds? What stops hindsight from leaking test problems?

## 5. The data, and what its numbers mean

**The bank** (`data/dataset_v2/bank.json`): 268 formal problems, levels 1-5, 144 `prove`, 121 `answer` ("determine all ..."), 3 `exists`. 187 are PROVED (a kernel-checked reference proof, an action sequence
replayed in a fresh engine by `dataset.py --verify`, and judged by the independent checker in S50) and 81 are STATEMENT-ONLY (the checked formal statement and brute-force evidence that it is true, no proof). Splits: **train 190, dev 20,
test 58**, cut by FAMILY (the parts of one problem, or a statement in other clothes, stay together) as a hash of the family under a frozen salt, 70/10/20 in expectation, so adding problems never moves an
existing one; the salt was searched once, for the first 82 problems, until the coverage requirements held, and is part of the dataset, not a tuning knob (`bank_split.py`). Read the table below before
writing the word "olympiad":

| level | meaning (`bank.LEVELS`) | problems | in test (proved / statement-only) |
|---|---|---|---|
| 1 | one arithmetic step after the introductions (`ring`, `linarith`, `nlinarith`) | 37 | 5 / 0 |
| 2 | a short proof with one idea: a case split, a witness, one induction | 67 | 11 / 0 |
| 3 | an olympiad lemma: several ideas or lemmas (15-60 actions) | 75 | 19 / 4 |
| 4 | an olympiad problem with a short solution (60-200 actions) | 45 | 5 / 11 |
| 5 | a hard olympiad problem | 44 | 0 / 3 |

So **35 of the 40 provable test problems are of levels 1-3**, and the IMO tier (41 IMO, 18 ISL, 1 APMO problems) has proofs for 12. What the experiments can measure is learning on elementary to lemma-level
problems; the upper tiers are an open challenge and a statement-only problem solved by an agent is a lower bound on solvability. **With 58 test problems the 95% Wilson interval of a single run's rate has a
half-width of 6 points at 5% solved, 10 at 20% and 12.5 at 50%**, so claim large effects, report counts with intervals, and use paired comparisons over seeds.

**Where the texts come from.** The contest texts are paraphrases written for the bank (`origin`, `verified`, `source`); 114 of 268 have had their wording compared with a published source, and the audit counts
that and does not compute it. The AMC-style problems (72) were written for the bank "in the manner of the AMC": none is an AMC problem. Licences are NOT decided (PAPER_PLAN section 8).

**The statement audit** (`make audit`, `results/audit`): every statement is read back from the kernel's term into a small language, evaluated on a box of points and compared part by part with a Python spec
written separately; every plausible one-slip variant of it (10 854 of them) must differ somewhere; cautions say where the evidence is weaker (88 of 268 carry one). It checks that statement and spec agree; **it
cannot catch a misreading of the English that both share**. Only a second reader can, and the ledger (`docs/statement_reviews.tsv`) is empty: 0 of 268 confirmed.

**The synthetic generator** (`src/python/synth/`, `docs/GENERATOR.md`): proof trees are grown BACKWARDS from lemma-shaped goals by the gym's own engine; the unproved leaves become hypotheses; the result is a
theorem with a kernel-checked proof and labels (depth, lemmas used, hypotheses, size). A record is a pure function of (configuration, index), is replayed from its own data and is cut into splits by statement
digest and by proof *recipe* (`test_comp` holds out whole recipes: the compositional split). Anything that says what a dev or test problem says is excluded (by digest and by shape). The yield is about 21%
and the rate 12 attempts per second per core, so 60 000 theorems are about 450 000 attempts, about 10 core-hours. **There is ONE family of generation**; the paper says the family is narrow, and
the saturation statistics (Chao1, rarefaction) say how much of it a larger set adds.

*Read:* `data/dataset_v2/DATASET_CARD.md`, `docs/BANK.md` sections 1-6, `docs/GENERATOR.md`, `results/audit/README.md` after `make audit`. *Ask yourself:* what does "test" mean in this paper, and
which levels does it really measure? What can the audit not see? How do you know the synthetic set does not contain the test problems?

## 6. The learner and the methods

* **One network over options** (`ml_model.py`): an encoder of the goal (transformer by default; bidirectional GRU, dilated CNN and bag of tokens as the studies' alternatives, behind one `encode` method), the
  heads that score the options (the decisions of the assembler), a value head, and the heads the reinforcement learners and the manager-worker agent use (a critic V(goal, budget left, open goals), a manager
  head, a size estimate). `docs/ARCHITECTURE_RL.md` section 3 is the design; the code is what runs.
* **Behaviour cloning** on synthetic theorems (pre-training) then on the bank's train proofs; **expert iteration with hindsight** (sample, keep the kernel-checked proofs, train on them; hindsight theorems are
  goals and data; `ei_her_boot` also adds them to the library); **PPO** from the cloned network; **search provers** (sample, greedy, IDA* on the policy's costs, MCTS, manager-worker).
* **Baselines without a model**: `blind` (iterative deepening over the menus) and `uniform` (a uniformly random legal action). They say how much of the benchmark is search and how much is a policy.
* **"Solved within B probes"** is the headline measure: the fraction of problems for which some attempt within B probes ended in a kernel-checked proof. An unsolved search spends B + 1 probes by
  construction, so the curve read off the largest budget is exactly what a run with a smaller budget finds (S51 G).
* **The studies**: other encoders (`bc_<arch>`), other widths (`bc_w64`, `bc_w256`), learning from random weights (`scratch_*`), the number of synthetic theorems (`bc_n500`, ...).

*Read:* `docs/ARCHITECTURE_RL.md` sections 1-3, `docs/TRAINING.md` sections 1-4, `src/python/experiments/exp_spec.py` (the whole matrix as data; its docstring lists every method).
*Ask yourself:* what is the difference between `pre`, `bc_scratch` and `bc`? Why is "solved within B probes" a curve and not a number? What does `uniform` tell a reader that `blind` does not?

## 7. The experiments, and what each result may claim

| question | evidence | a positive result may say | a null or negative result may say |
|---|---|---|---|
| (a) what does it cost? throughput, memory, convergence | `bench/*` (profiled benchmark: exclusive time ledger, latency percentiles, workers, memory as proportional set size), training curves | probes per second per solver, where the time goes, how it scales with workers | the same: this is description; it needs no comparison to be true, only a clean machine |
| (b) does synthetic pre-training help on the bank's problems? (hidden test) | `eval/main/*` on dev and test: `pre`, `bc_scratch`, `bc`, `bc_n*`, RL variants | "above random and above the same learner without pre-training, by X points (interval over seeds)" | "no measurable effect at this scale"; the benchmark is then a description of difficulty |
| (c) scaling | held-out synthetic loss against data and parameters, olympiad accuracy against data | exponents with intervals over seeds, over the range measured (no asymptote assumed) | "no clear power law in this range" |
| (d) ablations | hindsight, bootstrapped library, sub-theorem goals, encoder, manager-worker | differences with paired intervals | "no effect we could detect with 58 problems" |

**The statistics** (`exp_stats.py`, checked against hand-worked values in S51 A): per-seed values, Student-t intervals over seeds, Wilson intervals for a single run's rate, paired tests with Holm
correction, bootstrap, Chao1 and rarefaction for the generator. Default 5 seeds (`--seeds 0-6` for 7).

**The sealed test.** The test split is read only by the `eval/*` runs of the `full` profile; pilots run with `--set eval_splits=dev`; no recipe, checkpoint, hyper-parameter, prover or stopping rule is chosen
by looking at test (PAPER_PLAN 3.2). Tuning was not done: the recipe is one for every method ("with this recipe"). **Pre-registration** is the tag of step R6 and the deviations log of section 9 of the plan.

*Read:* `docs/PAPER_PLAN.md` sections 2-4, `docs/EXPERIMENTS.md` sections 1-5. *Ask yourself:* which claim of the abstract does each table carry? What would you write if every learner solved nothing on test?

## 8. How the numbers get into the paper (the evidence tools)

* **The identity of the code.** Evidence is bound to the digest of `src/`, `tests/`, `mutants/` and the manifest of the dataset (documents, the paper, tools and scripts are not code: `python3 tools/check_numbers.py
  --tree`). A record of another tree is stale and is not quoted. `dirty` means that CODE is uncommitted; a run records the digest when it starts and when it ends.
* **`make soundness`** (`results/soundness`): every gate, the mutation catalogue, fuzz campaigns, the independent checker against the kernel, the inventory of the trusted code, the findings by layer and
  effect. **`make audit`** (`results/audit`): section 5. **`make tool-mutants`**: faults injected into those tools (252; all caught on this tree). **`tools/check_numbers.py YOUR.tex --strict`**: every macro,
  table and figure the text cites exists, in a folder that is what its manifest says, made for the code you submit.
* Why the tools are tested: a test written together with the code tends to check what the code does, not what it must do; of the first 172 faults injected into the tools 21 got past their tests, among them
  stages that reused a result made for other code (F88).

*Read:* `docs/EXPERIMENTS.md` section 7, `mutants/README.md` (the second half). *Ask yourself:* how do you know that a number in the PDF comes from the code you submit?

## 9. The limitations, as a reviewer will list them (condense into the paper)

1. Soundness is tested, not proved; the checker is independent in code, not in authorship; the mutation catalogue is hand-written and does not cover the bank, learner, generator or pipeline.
2. Statement faithfulness is checked against a spec written by the same people; no second reader has confirmed any statement yet (0 of 268). `docs/SECOND_READER.md` is the handout and `tools/review_packs.py` makes the packs: do this before the freeze, because an `unfaithful` verdict changes the dataset.
3. The olympiad part is small: 20 dev and 58 test problems, 35 of the 40 provable test problems of levels 1-3; the IMO tier is mostly statements without proofs. Rates have wide intervals.
4. One family of synthetic theorems; the library is elementary (unary `Nat`, binary `N`, `Z`, divisibility, gcd, primes); no real analysis, no geometry, no combinatorics beyond what is encoded.
5. No tuning: the same recipe for every method; the ranking of methods may change with tuning.
6. The learners are baselines: no claim of competitiveness with LLM provers, and none with Lean-based systems, which by late 2026 are reported to have all but saturated miniF2F and PutnamBench.
7. The experiments run on CPU (or the GPU you have); the timing numbers describe that machine.

## 10. Prior work: what to compare, and what NOT to claim

TMLR should not accept a paper that "incorrectly claims novelty over existing published work". Before writing "first", "novel" or "unlike", read the systems below and say what you checked. The paper's related
work is yours to write; `paper/main.bib` holds the references of the skeleton, to be verified (title, venue, year) before use.

* **Lean-based**: LeanDojo (Yang et al.), Lean Gym / Pantograph-style machine interfaces, miniF2F, ProofNet, PutnamBench. Ask: what do they trust (Lean's kernel is also small; its tactic layer is not), how fast is a step,
  how is cost accounted?
* **Other proof assistants as learning environments**: CoqGym, HOList / DeepHOL, GamePad, Metamath-based systems (Holophrasm, GPT-f), INT (synthetic inequalities).
* **The closest: Peano** (Poesia and Goodman): a small dependently typed kernel built as an environment for learning formal mathematics from a curriculum. Read it first and say precisely how this system differs.
* `docs/PRIOR_WORK.md` has what was read on 2026-11-02 (Peano, Minimo, INT, LeanDojo, Pantograph, Kimina, PVSGym) and what was not. Two results: hindsight on proof trees is already in Minimo (do not claim it), and INT is the nearest prior art for a fast lightweight assistant with a synthetic generator.
* **Candidate differences to verify, not to assume:** a deterministic, fail-closed step budget inside the kernel (so episode cost is reproducible); a mutation-tested kernel with an independent checker at
  this scale; actions that say their term arguments through a typed builder; kernel-replayed hindsight with a library that grows (Minimo's library is fixed, by its own account); evidence tooling bound to the identity of the code. For each, find out
  whether another system already has it.

## 11. Questions an editor or reviewer may ask, with the short answer and where to look

1. *Why a new kernel and not Lean?* The use case is RL from scratch: a 7.5 k-line trusted base you can audit, a deterministic cost model, in-process speed. It is not a claim that this beats Lean at proving. (Section 10: do the comparison or say you did not.)
2. *What is trusted?* The kernel and its bindings, the compiler and the machine. Section 1.
3. *How do you know it is sound?* You do not know; you have tested it four ways (section 3), and the paper says "tested".
4. *What does 0 survivors mean?* Every seeded fault is caught by some gate (22 documented equivalents). It bounds how much the tests miss of what was thought of, not what was not.
5. *What is a probe, and why is it the cost?* One attempted action with rewind; seconds depend on the machine, probes do not.
6. *Are the 268 statements the problems they claim to be?* Checked against a spec by the authors, 114 compared with a source, 0 confirmed by a second reader.
7. *Is the test set contaminated by the synthetic data?* Exclusion by digest and by shape, statistics of what was excluded, near-duplicate screen (`docs/GENERATOR.md`).
8. *Is 58 test problems enough?* For large effects, with intervals; the paper states the sizes and claims accordingly.
9. *Why is the olympiad part mostly lemma-level?* Because proofs of IMO problems need a library this system does not have; the IMO tier is statements and an open challenge.
10. *Do the learners beat the baselines?* Whatever `eval/main/*` says; report it with intervals, including if the answer is no.
11. *Were hyper-parameters tuned on test?* No: pilots run on dev, the test split is read once by the full profile (PAPER_PLAN 3.2).
12. *Why one synthetic family?* It was the one that gave kernel-checked proofs with labels at the required scale; the paper lists it as a limitation and gives saturation statistics.
13. *Can I reproduce a number?* One command per stage, digests, determinism evidence across worker counts (`bench`), the estimate against the measurement.
14. *What did the AI assistant do?* Say it plainly in the acknowledgments; and be ready to explain any part of the code or text yourself.
15. *Which claims are in the abstract and which table carries each?* PAPER_PLAN section 2.
16. *What would falsify your main claim?* Section 7, last column.
17. *What did you find wrong in your own system?* The findings log: 89 findings with the tests that pin them (F51, F86, F87, F88, F89 are the instructive ones).
18. *What does the benchmark cost to run?* `exp_estimate.py`; the full profile is about 1 100 core-hours for the core experiment (estimate), the studies extra.
19. *Is the benchmark saturated or too hard?* Look at the per-level solved fractions of the baselines and of the best learner: a benchmark is useful when they spread out.
20. *Why should anyone use it?* Because training a policy from scratch on a proof assistant needs a fast, deterministic, auditable environment and a graded task set; give the one-paragraph answer in your own words.

## 12. Self-test (answer without notes before you submit)

1. Name the trusted components and say why a wrong tactic cannot prove a false statement.
2. What does hash-consing buy, and what does the work meter turn into when it is spent?
3. List the four kinds of soundness evidence and the limit of each.
4. What is a mutant, an equivalent mutant, and what would a survivor tell you?
5. Why is the audit not enough to say that the 268 statements are faithful?
6. What is a probe? What is "solved within B probes"? Why does an unsolved search spend B + 1?
7. Which levels do the 40 provable test problems belong to? What does that mean for the word "olympiad"?
8. How is the synthetic set kept from containing dev or test problems? What does `test_comp` hold out?
9. Which learners and baselines are compared, and which of them uses no model?
10. What is sealed, by what procedure, and what did the pilots touch?
11. What is the tree digest, what is bound to it, and what does `dirty` mean?
12. What did F88 find, and why does it matter for a number in the paper?
13. Which three limitations would you put first, and which claim does each one restrict?
14. What would you write in the abstract if every learner solved nothing on test?
15. What is the closest prior system, and what is the difference you can defend?

## 13. A reading and doing order

1. `README.md`; this file sections 1-3. 2. Run `examples/01` to `08` (each ends with `EXAMPLE OK`) and `python3 tests/run_gates.py s14_prop_world_tests.py s16_quot_tests.py` (seconds).
3. `docs/PAPER_PLAN.md` sections 1-3 and 10. 4. `docs/BANK.md` sections 1-6 and `docs/GENERATOR.md`. 5. `docs/TRAINING.md` sections 1-4 and `exp_spec.py` (read the docstring, then `exp_run.py small --list`).
6. Twenty rows of `docs/FINDINGS.md`, chosen at random. 7. Break something on purpose: change one line of `itp_kernel.hpp` (or one line of a mutant's target), run the gates, read the failure.
8. `make audit` and read three statements of `review_sheet.md` against their contest text. 9. Write the abstract and section 1 of the paper before looking at any table, then compare with what the tables say.
