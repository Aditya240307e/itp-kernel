# TRAINING.md -- learning to prove on the bank: the pipeline, the protocol, the results

This file says how the learner is built, how it is trained and measured, and what it achieves -- including what it does NOT. Every number below comes from a run whose command is given in section 8 (the
result files are in `data/results/`); every proof that counts as "solved" is a **kernel-checked proof** (the learner can be wrong in any way it likes; it cannot make a false statement provable). The gate
S44 (`tests/learner/s44_ml_tests.py`) protects the bookkeeping that the results depend on (section 7).

**Status in one paragraph.** The pipeline is complete and tested (features, policy network, behaviour cloning, policy-guided search, REINFORCE / filtered-BC loops, a Gymnasium environment). What has been *run*
is small: one behaviour-cloning run (30 epochs, 1.56 M parameters, about 73 minutes on one CPU thread), the blind baselines on all three splits, and smoke tests of the policy-guided search and of the RL loop. The
policy-guided search has **not** been evaluated on the dev or test split, and the RL loop has not been run beyond two iterations of a toy network. Nothing here is evidence that the learner can prove olympiad
problems; it is evidence that the pipeline works and a baseline to beat. The next steps are in `ARCHITECTURE_RL.md`.

## 1. The task

An **episode** is one proof attempt of one statement of the bank (`bank_env.BankEnv`). The state is a proof; the agent works on its first open goal. The agent has **no text interface**: it sees the goal as integer
tokens and acts by answering the engine's questions -- *which kind of action? which lemma? which hypothesis? what is the next token of the term I am saying?* -- each among the **options** the engine says are legal at
that point (`gym_policy.Assembler`). The assembled action is probed; if it applies it is taken. The episode ends when every goal is closed (the term is then checked by the kernel: reward `solve_reward`) or when a
limit of actions or decisions is reached.

Two **regimes** (BANK.md section 3): *prefix* -- every global before the theorem is on the menu (the lemmas written for the problem included); *strict* -- the problem's own helper lemmas are hidden. A problem is a
strict-regime task only if its reference proof proves its lemmas inline (8 problems). **The prefix regime is much easier than the real task**: for several olympiad problems the reference proof in the prefix regime
is three actions long because the hard work sits in helper lemmas that are on the menu. Report the strict regime (or the prefix regime with the problem's helpers withheld) whenever a claim is about solving a
problem.

## 2. Data (dataset v1, `src/python/problems/dataset.py`; `data/dataset_v1/DATASET_CARD.md`)

| | problems | demonstration actions | split rule |
|---|---|---|---|
| bank (82 problems, 9 IMO, levels 1-5) | train 56 / dev 10 / test 16 | 1816 (+ 546 strict-regime) | by FAMILY, frozen salt 25 |
| library lemmas (the proofs the recorder replays as actions) | 398 (train 282 / dev 38 / test 78; 3 duplicate a bank problem) | 1978 (train 1420) | by lemma name |

A demonstration is decomposed into the **decisions** of the assembler: 4223 of the 4319 actions decompose (96 do not: REFINE 49, USE 18, CLAIM 17, APPLY_WITH 11, FORWARD 1 -- terms whose heads fit only up to
computation, F61; they are skipped, the proof continues). About 4 real choices per action; 182 tokens per bank goal on average (the longest goals are cut to fit the 512-token window by dropping the OLDEST
hypotheses, never the target).

Proof lengths: bank mean 22.1 / median 11 / max 197 actions; library mean 5.0 / max 56. The action mix and the argument for the manager/worker split are in `ARCHITECTURE_RL.md` section 1.

## 3. The model (`ml_model.py`, `ml_features.py`)

`PolicyNet(bags)`, 1 562 335 parameters at the default size (d = 128, 3 layers, 4 heads, feed-forward 384, dropout 0.1).

* **Goal features** (`ml_features`): the goal is a token sequence `[CLS] hyp_1 [SEP] ... [TGT] target`, every term in prefix order with applications flattened (`APP_n head arg_1 .. arg_n`), constants by global id,
  context variables by level, bound variables by de Bruijn index, binders `PI` / `LAM`. Window 512; `GoalFeat` remembers the span of each hypothesis.
* **Symbol bags** (`GlobalBags`): a global (lemma, constructor, definition) is also described by the symbols of its type and of its conclusion, so that a lemma never used in a training proof can still be scored by
  what it talks about; `overlap` (the fraction of those symbols the goal mentions) is an input feature of every option.
* **Encoder**: a transformer over the goal's tokens; the `[CLS]` vector is the goal, the mean over a hypothesis' tokens is the hypothesis.
* **Decoder**: an action is assembled by a sequence of decisions; a recurrent cell (GRU) carries what has been decided so far (the kind, the lemma, the tokens of the term being said). At each decision a query is
  formed from the goal, the recurrent state, the decision's name, the kind of action and the **hole** (the type of the term being said: the mean of its token embeddings), and every **option** is scored by a dot
  product with its embedding. Options are rows `(otype, a, b, flags)`: a global (symbol embedding, symbols of its type, overlap with the goal), a hypothesis (embedding of its type in the goal), a number (its
  value), a kind of action, a structural token, a range, stop / yes / direction. Nothing about the options is learned per problem.
* **Value head**: predicts `log(1 + actions still needed)` from the goal (trained on demonstrations; the search may use it).
* The **incremental** decoder (one decision at a time; used by search and rollouts) and the **batched** forward pass (used by training) are tested to give the same log-probabilities (S44 B, E).

## 4. Protocol

**Splits.** Train on the train split of the bank (prefix regime, plus the strict-regime proofs where there are two) and of the library lemmas. Dev picks the epoch (and nothing else); test is never read by
training code. Problems of one family are in one split; there is no leakage through variants of a problem. (The library lemma splits are by name; a lemma used inside a dev problem's proof may be a train lemma --
that is the intended use of a library.)

**Behaviour cloning** (`ml_train.py`): loss = cross-entropy over the options of every non-forced decision + 0.2 x the value-head regression; AdamW with a one-cycle schedule (10 % warm-up) to the peak learning rate
1e-3, weight decay 0.01, gradient clipping at 1.0, batch 32, 30 epochs, seed 0, one thread. Metrics are teacher-forced: decision accuracy, **action accuracy** (all decisions of an action right), value error, reported for dev-bank, dev-library and both.

**Search evaluation** (`ml_eval.py`): the unit of cost is the **probe** (one tried action), not seconds. A search is deterministic and its budget only truncates it, so one run at the largest budget gives the whole
curve: a problem is solved at budget B iff its proof was found after at most B probes. Every proof is replayed on a live proof and checked by the kernel. Rows are (problem, regime); a problem whose strict
regime equals its prefix regime is one proof. The blind baseline is `gym_search.CostSearcher` (IDA* on action costs) with its default costs; the policy-guided search is `ml_search.PolicySearcher` (cost of an action =
its negative log-probability under the policy + 0.2 per step; candidate actions enumerated best first; at most the 24 best options of a decision are expanded, and an option less probable than e^-9 is not).

**RL** (`ml_rl.py`): the policy plays `BankEnv` on TRAIN problems only (bank problems in both regimes; library lemmas, a fraction `--lemma-frac`); reward 1 for a kernel-checked proof minus `--len-penalty` per action;
decisions that assembled a refused action are pushed down (`--refused-weight`); the policy is kept close to the demonstrations by a behaviour-cloning term (`--bc-coef`). Modes: `reinforce` (per-problem moving-average
baseline) and `filter` (behaviour cloning on the episodes that finished the proof: expert iteration by rollouts). Dev is only used to watch progress, by sampled rollouts at low temperature.

## 5. Results

### 5.1 Blind search (no learning): `data/results/blind_{train,dev,test}.json`

8000 probes, 400 s per problem, both regimes (rows are problems per regime):

| split | problems | solved, prefix regime | solved, strict regime | what was solved |
|---|---|---|---|---|
| train | 56 | 9 | 9 (of 55 rows) | the polynomial identities (`ident_*`, 1 probe each), `sum_squares_zero` (175 probes, 3 actions), `quad_form_zero` (165 probes, 2 actions) |
| dev | 10 | 1 | 1 | `ident_diff_squares` (1 probe) |
| test | 16 | 2 | 1 (of 15 rows) | `ident_brahmagupta` (1 probe), `imo1964_p1a` (1 probe, prefix regime only) |

Everything the blind search proves is closed by `ring`, `nlinarith` or one lemma plus `nlinarith`. It never finds a proof that needs a witness, an induction, a case split with different branches, or a CLAIM. Several
unsolved rows stop long before the budget (e.g. `sum_two_squares_mod4` after 496 probes, `imo2019_p1` after 3490): the search exhausted the space it can reach with its cost cap, which says that these proofs need
actions the plain ranking never offers.

### 5.2 Behaviour cloning: `data/results/bc1_history.json`

30 epochs, 4396 s (about 73 minutes, one thread). Teacher-forced metrics on dev:

| epoch | train loss | train acc | dev loss | dev decision acc | dev action acc | dev-bank action acc | dev-library action acc |
|---|---|---|---|---|---|---|---|
| 0 | 2.551 | 0.246 | 2.599 | 0.274 | 0.117 | 0.065 | 0.216 |
| 2 | 1.761 | 0.409 | 2.037 | 0.384 | 0.222 | 0.225 | 0.216 |
| **4** | 1.444 | 0.499 | **2.008** | 0.416 | 0.217 | 0.177 | 0.294 |
| 8 | 1.004 | 0.640 | 2.323 | 0.427 | 0.276 | 0.259 | 0.307 |
| 15 | 0.536 | 0.803 | 3.075 | 0.446 | 0.285 | 0.253 | 0.346 |
| 29 | 0.255 | 0.906 | 3.931 | 0.452 | 0.314 | 0.290 | 0.359 |

The dev loss is lowest at epoch 4 and rises steadily afterwards (the model memorises the 56 training proofs: train accuracy 0.91 against dev 0.45), while dev *accuracy* keeps creeping up -- the model becomes
confident and often wrong, i.e. **miscalibrated**. For a search that uses the probabilities as costs, calibration is what matters, so `bc1.pt` is the epoch-4 checkpoint (best dev loss), `bc1_early.pt` the
epoch-2 checkpoint. Chance level for a decision is far below these numbers (a menu has about 14 kinds, a term token has tens of options), so the model has learned something; with 56 training proofs it cannot be
expected to generalise to new mathematics.

### 5.3 Policy-guided search: a smoke test only (`data/results/smoke_policy.json`)

`bc1_early.pt`, prefix regime, 500 probes, four hand-picked dev/test problems: solved `ident_diff_squares` (3 probes) and `imo2019_p1` (36 probes; blind search exhausted its space on it after 3490 probes); did
not solve `not_dvd_between` or `coprime_consecutive`. This is **one data point chosen by hand, in the easy regime**: the reference proof of `imo2019_p1` in the prefix regime is 3 actions because its helper lemmas are
on the menu. It shows that the policy can steer the search; it does not measure how much. The full dev/test evaluation (`ml_eval.py` with `--model`) costs about 40x more time per probe than the blind search and was
not run.

### 5.4 RL: a plumbing test only (`data/results/rl_smoke_history.json`)

Two iterations of REINFORCE with a randomly initialised network of 190 k parameters (6 episodes each): the loop runs, the buffers fill, nothing is learned or claimed.

### 5.5 What we do not know yet

* How much the policy-guided search improves on the blind baseline at equal probes, on dev and test, in the strict regime.
* Whether regularisation (dropout/weight decay sweeps, smaller models, early stopping on action accuracy, data augmentation by synthetic problems) removes the overfitting, or whether the bottleneck is the amount
  of data (it probably is: see `ARCHITECTURE_RL.md` section 6).
* Whether RL from the BC policy (REINFORCE, filtered BC) improves the solve rate on held-out problems; the plan and its stopping rules are in `ARCHITECTURE_RL.md` section 3.4.

## 6. The Gymnasium environment (`src/python/learner/bank_env.py`; example 06)

```python
import bank_all, bank_env as BE
bank = bank_all.load()                                              # the world (~20 s)
env = BE.BankEnv(bank, names=['imo1959_p1'], regime='prefix', max_actions=40, max_decisions=400)
obs, info = env.reset(options={'problem': 'imo1959_p1'})
obs, reward, terminated, truncated, info = env.step(int(option_index))         # info['action_mask'], info['choice'] describe the next question
```

* **Observation** (a Dict of padded arrays): `tokens, seg, hyp_of` (the goal), `n_hyps, goals_left`, `overlap (n_globals, 2)`, `choice (3,)` (the kind of decision, the kind of action it belongs to, the position inside a
  term), `hole` (tokens of the type being said), `rows, n_options, mask` (the options as `(otype, a, b, flags)` rows). **Action**: `Discrete(max_options)`, the index of an option; `action_masks()` marks the legal ones; an
  illegal index raises `ValueError` (or loses the decision with `illegal='penalize'`).
* **Reward**: `solve_reward` (default 1) only for a proof the kernel has checked, minus `action_penalty` per action taken and `fail_penalty` per refused action; **truncation** after `max_actions` actions (taken or
  refused) or `max_decisions` decisions. `rich=True` puts the raw `Choice` / `GoalFeat` objects in `info` (what the learners of this repository use to recompute log-probabilities).
* `gym.make('ITPBank-v0', split='train')` builds the whole bank (about 20 s) and restricts it to a split. A library lemma can be played like a problem (`lemmas=` argument).
* Features that a stronger agent will want and the environment does not have yet (CLAIM hints from the goal, a hindsight relabeller, a cheap `solved?` oracle for sub-goals) are listed in `ARCHITECTURE_RL.md` section 5.

## 7. What is tested

S44 (`tests/learner/s44_ml_tests.py`, `run_gates.py --ml`): **A** features and data (a goal's tokens are determined, framed and typed; the symbol overlap of a global with its own type is 1; six bank problems'
demonstrations decompose without a gap and batch into the documented shapes); **B** the network (incremental decoder = batched forward pass, padding included; weights multiply the loss; a positive weight raises the
probability of the decisions it is given to, a negative one lowers it; a saved model loads to the same function); **C** the search (the stream of candidate actions is in nondecreasing cost, without repeats; an
action's cost is the negative log-probability of its decisions; the search proves one-step problems with an untrained network and returns a kernel-checked term whose portable actions replay); **D** the environment
(Gymnasium's API, mask, rewards only for kernel-checked proofs, truncation, reproducible seeds, scripted runs that solve six problems decision by decision, library lemmas); **E** the RL loop (the log-probabilities
recorded when a rollout samples are those the batched network gives to the same decisions; a greedy rollout is deterministic).

S45 and S46 (`tests/repo/`, `run_gates.py --repo`) test the repository: the layout, the registry, imports from a fresh interpreter, the frozen kernel's checksums, the documents' links, the dataset's digests, and that
every example runs. The kernel, tactic, environment and bank gates (46) protect everything below the learner.

## 8. Reproduce

All commands run from the repository root after `make kernel` (or `python3 src/python/build_kernel.py`). The dataset is `data/dataset_v1`.

```
python3 src/python/problems/dataset.py --verify data/dataset_v1                  # replay all 90 demonstrations in a fresh engine over a freshly built world (about 1 min)
python3 src/python/learner/ml_eval.py data/dataset_v1 --split dev --probes 8000 --time 400 --blind --out data/results/blind_dev.json      # also: --split train / test
python3 src/python/learner/ml_train.py data/dataset_v1 --out artifacts/models/bc1.pt --epochs 30                                       # about 73 min, one thread
python3 src/python/learner/ml_eval.py data/dataset_v1 --split dev --probes 4000 --model artifacts/models/bc1.pt --out data/results/bc1_dev.json
python3 src/python/learner/ml_report.py blind=data/results/blind_dev.json bc1=data/results/bc1_dev.json --markdown
python3 src/python/learner/ml_rl.py data/dataset_v1 --init artifacts/models/bc1.pt --out artifacts/models/rl.pt --iters 60 --episodes 24 --mode reinforce
```

Seeds are fixed (`--seed 0`); training and search use one thread, so runs are reproducible on the same library versions. The trained models of these runs (`bc1.pt` etc.) are in the release archive under
`artifacts/models/` (they are not in the git history); the result files in `data/results/` are.

## 9. Limits

* 82 problems is a tiny training set; a transformer with 1.5 M parameters overfits it within four epochs. Do not tune the architecture before the data grows (`ARCHITECTURE_RL.md` section 6).
* The splits are small (dev 10, test 16): a difference of one or two solved problems is noise. Use the library lemmas' dev and test splits (38 and 78 proofs) as the larger, easier held-out sets.
* The prefix regime overstates ability (section 1). The strict regime has only 8 problems.
* Teacher-forced accuracy is not search success; the only number that counts is kernel-checked proofs per probe budget on held-out problems.
* The reference proofs are *one* proof per problem. A learner that finds another proof is right; the BC loss punishes it. Expert iteration and search-based evaluation do not have this problem; teacher-forced
  metrics do.
* Probes are not seconds: a policy-guided probe costs about 40x a blind probe (network evaluation on one CPU thread). Compare methods at equal probes AND report seconds.
