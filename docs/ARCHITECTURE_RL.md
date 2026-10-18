# ARCHITECTURE_RL.md -- from imitation to a manager/worker learner (design note)

Status: **design, not a result**. Nothing in this file has been trained at scale; the numbers quoted are measurements of the dataset and of the first small experiments (TRAINING.md). The note answers one
question: *how should the learner be structured so that it can eventually prove olympiad algebra and number theory, given what the kernel, the environment and the data are today?*

## 0. Summary

1. **The kernel makes every learner safe, so the architecture can be bold.** A finished proof is checked by the kernel; the learner can be wrong in any way it likes and cannot make a false statement provable.
2. **The bottleneck today is data and generalisation, not the RL algorithm.** 82 problems, 1 816 demonstration actions (56 train problems), 398 library lemma proofs (1 978 actions). Behaviour cloning (BC) overfits after
   four epochs; blind search solves 1 of 10 dev problems. A hierarchy that needs more data than a flat policy will not help before the data exists.
3. **Recommended shape: one network, three roles, trained in stages** -- (a) a *worker* that closes a goal in a short, budgeted sequence of actions (the current assembler network), (b) a *critic* that says how likely
   a goal is to be closed within a budget (the value head, re-targeted), (c) a *manager* whose only extra move is **CLAIM** (say a statement, split the goal into "prove the claim" and "use the claim").
   The manager/worker split is justified by a data argument: the worker's task distribution can be **synthesised at scale** (decision procedures, mutated lemmas) while the manager's must come from real mathematics.
4. **Hindsight (HER) is natural here, and exact.** A "goal" is a statement in a context. Whatever a failed attempt *derived* is a statement the agent *did* prove; relabelling it as the goal gives a kernel-checkable
   positive example, never a guess.
5. **Search is part of the learner, not an add-on.** The alternative to learned hierarchy -- AND-OR proof search guided by the policy and value, with expert iteration -- is the safer baseline and the source of the
   hindsight data. Build it first (stages S0-S2); add the learned manager (S3) only if it beats the baseline at equal probe budget.
6. Environment and dataset work that every variant needs is listed in sections 5 and 6 and can start now.

## 1. What exists (measured)

| | |
|---|---|
| Kernel | hybrid kernel (`src/c++/hybrid`), 46 gates + the learner's gate S44 + the repository gates S45-S46; every proof the agent finishes is re-checked by the kernel (and can be re-checked by the independent checker: `recheck.SecondOpinion`) |
| Environment | `gym_env` (menu actions) and `gym_acts` (actions that *say* terms: CLAIM, FORWARD, REFINE, USE, ...); `gym_policy.Assembler` turns an action into a sequence of discrete **decisions**; `bank_env.BankEnv` is a Gymnasium env at the decision level (no text anywhere) |
| Bank | 82 problems (9 IMO), levels 1-5, splits by family: train 56 / dev 10 / test 16; 8 problems are strict-regime tasks (self-contained reference proofs) |
| Demonstrations | 1 816 actions in 82 proofs: **mean 22.1, median 11, max 197 actions per proof**; library: 398 lemma proofs, 1 978 actions, mean 5.0, max 56 |
| Decomposition | 4 223 of 4 319 actions decompose into assembler decisions; about 4 real choices per action |
| Model | transformer over the goal's tokens (window 512) + GRU over the decisions of one action + dot-product scoring of the options; value head = log(1 + actions still needed) |
| BC | best dev loss at epoch 4, later epochs reach ~0.31 action accuracy but are miscalibrated: **severe overfitting** |
| Search | blind IDA* solves dev 1/10, test 2/19 rows; the policy-guided search is ~40x slower per probe and has not been evaluated yet |

Action mix of the bank demonstrations (what a proof is made of):

| kind | share | proofs using it | | kind | share | proofs using it |
|---|---|---|---|---|---|---|
| LINARITH | 24.8 % | 42 | | APPLY | 4.5 % | 29 |
| CASES | 16.4 % | 48 | | REWRITE | 4.3 % | 21 |
| FORWARD | 10.9 % | 62 | | RING | 3.4 % | 34 |
| **CLAIM** | **9.7 %** | **42** | | REWRITE_AT / LINCOMB | 3.0 / 2.8 % | 11 / 21 |
| USE | 8.8 % | 40 | | NLINARITH / REFINE | 2.3 / 2.2 % | 23 / 15 |
| APPLY_WITH | 5.2 % | 18 | | INTERVAL_CASES / INDUCT_WITH | 1.0 / 0.8 % | 13 / 14 |

Half of the proofs (42 of 82) contain claims, and those use **4.2 claims each** on average: the proofs that have structure have a lot of it. The other half are straight-line (arithmetic closing, case analysis, witnesses).

## 2. What the agent decides

An action is assembled by decisions (`gym_policy.CHOICE_NAMES`): `kind` -> `cand | hyp | source` (which lemma / hypothesis) -> `token`* (the next token of a term it *says*, among the tokens that keep the term
well typed) -> `more | slot | gen | dir | lin | range`. So the current policy is already **factored into a macro choice (the kind) and micro choices (arguments, terms)**; what is missing is any *temporal*
abstraction above a single action and any *explicit goal* below it.

Kinds fall into five **directions**, which is the vocabulary I propose for "direction of proof":

| direction | kinds | what it does to the goal |
|---|---|---|
| CLOSE | RING, LINARITH, NLINARITH, LINCOMB | ends the goal with a kernel-checked certificate |
| BACKWARD | APPLY, APPLY_WITH, REFINE, REWRITE, REWRITE_WITH, USE, BY_CONTRA | replaces the goal by (usually simpler) goals |
| FORWARD | FORWARD, INSTANTIATE, REWRITE_AT, INJECT | adds a hypothesis (derives a fact), keeps the goal |
| SPLIT | CASES, INDUCT, INDUCT_WITH, INTERVAL_CASES | the goal becomes several goals (an AND node) |
| CLAIM | CLAIM | invents a statement: two goals, "prove S" and "goal with S assumed" |

(INTRO is done by the engine itself.)

CLAIM is the **only place where the agent invents mathematics** (a statement that is not already in the goal, a hypothesis or a lemma); USE/INSTANTIATE invent *terms* but within a typed pool. That makes it the natural
boundary between a manager and a worker.

## 3. The learner

### 3.1 One network, three heads

```
goal tokens ──► transformer trunk ──► goal vector ──┬─► worker head: decisions of the assembler (existing)
(+ budget, direction hint, claim history)           ├─► critic head: P(goal closed within b more actions), b = 1..B, ">B"
                                                    ├─► manager head: claim tokens (typed-builder codes), or "hand over"
                                                    └─► auxiliary heads: next-3 kinds, which hypotheses will be used, steps to go
```

* **Worker** = the existing policy. New inputs: the *remaining budget* and an optional *direction hint* (one of the five directions, dropped at random during training so the worker also runs without a manager).
* **Critic** = the value head, re-targeted from a regression on log(1 + steps) to a **classification over "closed within b actions"** (b = 1..B and ">B"). A distribution over remaining effort is what a budgeted
  search and a manager need; it is trained from demonstrations (steps-to-go is known exactly) and later from search outcomes (closed / not closed within the budget).
* **Manager** = the same decoder restricted to the CLAIM action: the statement is *said* as tokens of the typed builder, so every claim is well typed by construction; plus a "hand over" move (give the goal to the worker).
* **Auxiliary heads** are cheap multi-task regularisers for the BC regime (see S0).

The trunk is shared at first (so BC trains everything at once) and can be split later if the roles interfere.

### 3.2 The manager/worker interface (options)

The interface is a **goal in a context plus a budget**, nothing else:

* The manager sees the open goal `G` (hypotheses + target) and the claims made so far. It either **hands `G` over** (worker budget `b`), or **claims** `S`: the engine splits `G` into `G_S` (prove `S`) and `G'` (`G` with `S` assumed).
* The worker gets `G_S` with budget `b` and acts until that goal -- and every goal its own SPLIT moves create, which share the budget -- is closed (success) or the budget is spent (failure).
  CLAIM is **reserved to the manager**; inside a worker episode the allowed kinds are CLOSE, BACKWARD, FORWARD and SPLIT.
* Control returns to the manager at the next open goal. The manager's episode ends when the main goal is closed (reward 1, kernel-checked) or its own claim/step budget runs out.

For the half of the proofs without claims the manager's only move is "hand over", and the system is the flat worker -- the hierarchy costs nothing where there is no structure.

### 3.3 Hindsight: what is the goal in a theorem prover?

A goal is a statement in a context (`GoalData`: target + hypothesis types and names). Because states are *data* (hash-consed, with canonical keys, probes rewind the kernel), any intermediate goal can be re-created
and used as the start of a new episode. This gives four relabelling rules; each produces a *kernel-checkable* example, because the relabelled "proof" is a sub-sequence of actions the kernel already accepted.

1. **Derived-fact relabelling (worker).** A FORWARD / INSTANTIATE / REWRITE_AT / proved-CLAIM step creates a hypothesis `h : T`. The (dependency-sliced) actions that produced `h` are a proof of `T` in the context they
   started from. Relabel: goal = `T`, reward 1, actions = the slice. A failed attempt at `S` that derived five facts yields five positive examples.
2. **Subgoal relabelling (worker, from search).** Every goal that a search closed -- also inside a failed top-level search -- is a theorem with a proof tree. Its (goal, action) pairs enter the training set. This is what
   expert iteration already gives for the root; the change is to keep the *subtrees* of failed searches.
3. **Hindsight action transitions (manager).** If the manager claimed `S`, the worker failed on `S`, but the worker's episode *closed* some `T` (or derived `T`), the manager's transition is relabelled with `T` as its action
   (as in HAC -- Levy et al. 2019 -- "hindsight action transitions"). The manager learns what *is* provable from that context.
4. **Subgoal testing (manager).** Claims whose proof attempts fail at full budget are kept as negatives for the **claim-feasibility critic** `Q(context, S) = P(worker proves S within b)`, trained on all of the above. The
   manager proposes claims, `Q` ranks or filters them; this is the learned analogue of "is this subgoal reachable?" and it is what keeps the manager from proposing claims nobody can prove.

Reward hacking to guard against: trivial or unused claims. The manager's reward is the final kernel-checked proof; the *shaping* terms (claim proved: small +; claim used in the final term: +; claim failed: -) are
checked against usage in the final proof term, and any shaping is potential-based (a function of the critic's estimate) so that the optimal policy does not change.

### 3.4 Training stages (and when to move on)

| stage | what | losses / data | advance when |
|---|---|---|---|
| **S0** BC done right | regularise, augment, multi-task | CE on decisions + critic + next-3-kinds + hypothesis relevance; dropout, weight decay, label smoothing, hypothesis-order and name-invariant augmentation; train on bank + library; early-stop on **dev probes-to-solve**, not dev loss | policy-guided search beats blind search on dev at equal probes (blind: 1/10) |
| **S1** search + expert iteration (flat) | `PolicySearcher` (IDA* on -log p + step cost) on train problems and library lemmas, retrain on what was found | filtered BC on kernel-verified proofs; **keep subtrees** (rule 2); policy distillation of the search | dev solved rises over iterations; new proofs of train problems differ from the references (diversity) |
| **S2** worker at scale | synthetic goals + hindsight | rules 1-2 data, goals generated by decision procedures and lemma mutation (section 6), critic trained on search outcomes | worker closes the held-out synthetic split, and the dev gain from S1 does not drop (no forgetting) |
| **S3** manager | claim proposals | BC on the claim segmentation of demonstrations (177 claims), then rules 3-4; `Q` filters; RL (REINFORCE with baseline, KL to the BC manager) with the final proof as reward | beats S1/S2 at **equal probe budget** on dev; otherwise stay flat |
| **S4** self-play (optional) | the manager as *conjecturer* | claims that the worker proves and that are *used* later become library lemmas | new lemmas improve held-out problems |

Policy-gradient RL (the current `ml_rl.py`, REINFORCE with per-problem baseline) is a fine-tuning tool here, always with a KL penalty to the BC policy; with this reward density the main gains are expected from
search + expert iteration + hindsight data, not from the gradient estimator.

### 3.5 Alternatives, and when to prefer them

* **Flat policy + value + search (AlphaZero / HTPS style).** The baseline to beat. Hierarchy is implicit in the AND-OR search; macro structure comes from CLAIM in the action language. Prefer it if S3 does not pay for itself.
* **Kind-skeleton plans.** The manager predicts a short sequence of kinds (a plan) and the worker is conditioned on it. Cheap to test *inside BC* as the next-3-kinds auxiliary loss before any hierarchy is built; if
  it does not improve dev accuracy or search, drop it. (Without an extra information source, a plan of kinds carries little beyond the state itself.)
* **Option discovery.** Mine frequent action n-grams from demonstrations and searches (compression, library learning) and give the manager *macro-tactics* to choose from. Complementary: it grows the worker's
  vocabulary. Needs enough proofs to find repeated structure -- after S1/S2.
* **Informal sketches** (Draft-Sketch-Prove). A manager conditioned on a natural-language solution would be the strongest prior for olympiad problems but contradicts the "no text interface" constraint of the agent; keep the
  dataset fields (sources, ideas) so that it remains possible as an *offline* teacher.

## 4. Why not just feudal / option-critic RL end to end?

Learned hierarchies trained by RL alone are brittle: the manager's problem is non-stationary while the worker learns, and a proof environment gives one bit of reward per attempt. The design above avoids the usual
failure modes by making the interface *symbolic* (a statement and a budget, both checkable), by supervising both levels from demonstrations first, by deriving the manager's training signal from kernel-checked
hindsight data, and by keeping a flat baseline at every stage.

## 5. Environment features (priority P0 first)

| id | feature | why | sketch | stage |
|---|---|---|---|---|
| E1 P0 | **goal-conditioned reset**: start an episode from an arbitrary (context, statement) | worker training on claims, relabelled goals, synthetic goals | `reset(options={'goal': GoalSpec(hyps=[(name, code)], target=code)})`; `tactics.Proof.from_goal` already builds the state | S2 |
| E2 P0 | **achieved goals in `info`**: statements (typed-builder codes + context id) of hypotheses derived and goals closed at each step | hindsight rules 1-4 | `info['achieved'] = [(kind, code, ctx_id), ...]` | S1 |
| E3 P0 | **budget** in the observation and in `info` | worker conditioning, options | `obs['budget']`, `info['remaining']`; truncation by the budget | S2 |
| E4 P0 | **synthetic goal generators** (ring identities, linear-arithmetic implications, divisibility, one-step lemma applications, mutated library lemmas) with the decision procedure's own proof as reference | worker data at scale | `synth.py` producing `(statement, portable proof)` pairs, kernel-checked at generation time | S2 |
| E5 P1 | **option wrapper**: manager actions = (claim, hand over); the worker runs inside as a callback | manager/worker training and evaluation | `OptionEnv(bank_env, worker=policy, budget=b)` | S3 |
| E6 P1 | **curriculum sampler**: prioritised by recent success rate in [0.1, 0.9], stratified by level and family | efficient use of 56 train problems | `CurriculumSampler(names, window)` | S1 |
| E7 P1 | **episode records**: every episode as portable actions + decisions + outcome (JSONL), appendable to the dataset | expert iteration, reproducibility | `dataset.append_proofs` | S1 |
| E8 P1 | **vector env**: one world per process, shared dataset directory, deterministic seeds | throughput on bigger machines | `make_vector(n)` over `AsyncVectorEnv` | S1 |
| E9 P2 | **richer observations**: term-DAG structure (tree depth / binder positions as features), history of kinds in this branch, summary of sibling goals | long goals, manager context | extra arrays, behind flags | S2 |
| E10 P2 | **lookahead features**: per option, the result of its probe (closes / opens n goals / fails) | much better ranking, at probe cost (~40x) | `lookahead=True`, off by default | S2 |
| E11 P2 | **standard telemetry**: probes, fuel and wall time per phase in `info` | budgets by probes, not seconds | | S1 |

Existing safeguards to keep: refused actions are free of side effects (atomic probes), `illegal='penalize'`, truncation by actions and decisions, the kernel check at the end of every episode.

## 6. Dataset roadmap

1. **More problems, bridging the gap** to IMOSL N1-N3 / A1-A3: levels 2-4 number theory and algebra (descent, parity, bounding, functional equations over Z, polynomial identities). Each problem keeps the faithfulness
   checks (a reference semantics validates the statement; the proof replays).
2. **Several proofs per problem.** Found by search and expert iteration, kept when they differ in structure (kind sequence) from the references; stored next to the reference proof.
3. **Problem mutants** (not kernel mutants): statement variations that preserve provability by construction -- constants changed, hypotheses reordered, variables renamed, a lemma generalised -- each re-proved by the
   reference proof's *script* or by search; marks the family in the split so that no mutant of a test problem is trained on.
4. **Lemma corpus growth.** Every problem adds helper lemmas; keep them all with their proofs (398 now) and split by name; flag duplicates of bank statements (done: `bank_duplicate`).
5. **Annotations for the manager**: claim segmentation of every demonstration (claim, its proof, the continuation), the direction of each action, technique tags (induction, descent, parity, bounding, case split, Vieta-style
   substitution). Derived automatically where possible; used for analysis and for conditioning, never required by the worker.
6. **Negatives for the critic**: goals searched to budget without a proof, with the budget; statements that are *false* (found by the reference semantics) so that the critic learns to give up early.
7. **Difficulty calibration**: blind and guided probes-to-solve per problem, proof length, number of claims, stored in `bank.json`; used by the curriculum sampler and for stratified reporting.
8. **Hygiene**: splits by family with a frozen salt (done), a leakage check between the library corpus, mutants and the held-out splits, digests of every file (`manifest.json`).

## 7. Evaluation and ablations

* Metric: **kernel-verified solves within a probe budget** (pass@probes curves at 250 ... 16 000 probes), per split, per level, with the regime (prefix / strict) stated. Dev for decisions, test frozen and reported once.
* Equal-budget comparisons only. A probe is the unit; wall time is reported separately (the policy-guided search is slower per probe).
* Ablations in order: blind search -> BC policy -> + critic in the search -> + expert iteration -> + subtree hindsight -> + synthetic worker data -> + manager (claims) -> + claim critic.
  Each arrow must pay for itself on dev before the next is built.
* Seeds: at least three for every learned result; report the spread (56 train problems make the variance large).

## 8. Risks and open questions

* **Tiny data.** Everything learned from 56 train problems overfits; the synthetic worker data and the lemma corpus are the main hope, not the algorithm.
* **Claim space is unbounded.** The manager must be a ranker over claims generated from the typed builder's pool before it is a free generator (S3 starts with ranking).
* **Non-stationarity** of the manager while the worker changes: mitigated by relabelling (rule 3) and by freezing the worker between manager phases.
* **Evaluation leakage** through the lemma corpus: lemmas whose statement equals a bank statement are flagged; keep it that way for mutants.
* **Search cost.** The policy-guided search is ~40x slower per probe than blind search; batching decisions across the nodes of a frontier and caching goal encodings are required before large budgets.
* **Whether any of the hierarchy beats flat search + expert iteration is an empirical question**; section 7 says how to find out.

## 9. Next steps that need no large run

1. E1-E3 in `bank_env` with tests in S44 (goal-conditioned reset, `achieved`, budget).
2. Claim segmentation and direction labels as fields of `bank.json` / `library.json` (analysis only).
3. `synth.py`: generators for ring / linarith / divisibility goals with their kernel-checked proofs.
4. S0 regularisation sweep on small models (minutes each), early stopping on dev probes-to-solve; evaluate `bc1.pt` guided vs blind on dev at 1 000 probes.
5. Re-target the value head (closed within b actions) and test it as a pruning signal in `PolicySearcher`.

## References (for orientation, not a survey)

Andrychowicz et al. 2017, *Hindsight Experience Replay*. Levy, Konidaris, Platt, Saenko 2019, *Learning Multi-Level Hierarchies with Hindsight* (HAC). Nachum et al. 2018, *Data-Efficient Hierarchical RL* (HIRO).
Vezhnevets et al. 2017, *FeUdal Networks*. Sutton, Precup, Singh 1999, *Between MDPs and semi-MDPs* (options). Polu & Sutskever 2020, *Generative Language Modeling for Automated Theorem Proving*.
Polu et al. 2022, *Formal Mathematics Statement Curriculum Learning* (expert iteration). Lample et al. 2022, *HyperTree Proof Search*. Jiang et al. 2022, *Draft, Sketch, and Prove*.
Wang & Deng 2020, *Learning to Prove Theorems by Learning to Generate Theorems*. Trinh et al. 2024, *Solving olympiad geometry without human demonstrations* (synthetic data from a symbolic engine).
