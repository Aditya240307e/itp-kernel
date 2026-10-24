# The synthetic theorem generator

`src/python/synth/` makes tens of thousands of theorems, each with a kernel-checked proof, from the lemma library. They are **synthetic**: true by construction, varied in depth and in the lemmas they need,
and not meant to be interesting mathematics. They are the practice ground for a prover that is later trained and tested on olympiad problems (the problem bank, `docs/BANK.md`). This note says what the
generator makes, how, why that way, how the data are checked, and what they do **not** show. Gate: S48 (`tests/synth/s48_synth_tests.py`). Numbers that describe a particular dataset are in its card
(`DATASET_CARD.md`, written from the manifest by `synth_cli.py card`); the numbers below that are marked *pilot* come from a 3 000-attempt run and are there to give a sense of scale, not to be quoted.

## 1. What a theorem is

A theorem is a closed statement over the integers, `forall (variables), hypothesis_1 -> ... -> hypothesis_m -> conclusion`, and a proof that the gym's action language replays and the kernel checks. Two examples
from the pilot (the printer's numerals are binary `Z.of_N (Npos ...)`; `pretty` is for reading):

```
forall (x0 x1 x2 x3 : Z), Z.is_gcd (x3 + x0) (-x2 - 4) 1 -> Z.is_gcd (-x1) (-x2 - 4) 1 -> Z.coprime (-x2 - 4) ((x3 + x0) * -x1)
    APPLY Z.coprime_mul_right; APPLY Z.coprime_symm; APPLY Z.coprime_of_is_gcd_one; APPLY Z.coprime_symm; APPLY Z.coprime_of_is_gcd_one
    depth 3, 5 steps, 3 lemmas, 2 hypotheses (both used as they stand), area gcd

forall (x0 x1 : Z), Z.dvd (x1 + x0) (x0 + 3) -> Z.dvd (-(x1 + x0)) (-((x0 + 3) + 1)) -> x1 + x0 = 1 \/ x1 + x0 = -1
    APPLY Z.dvd_one; APPLY Z.dvd_of_dvd_neg; APPLY Z.dvd_of_neg_dvd; APPLY Z.dvd_neg; APPLY_WITH Z.dvd_of_dvd_add_left; APPLY Z.neg_dvd; APPLY Z.dvd_of_dvd_neg
    depth 6, 7 steps, 6 lemmas, 2 hypotheses; a divisor of x0+3 and of x0+4 is a unit
```

The first is a bookkeeping exercise (a short chain of gcd lemmas, with terms that make it look like more); the second is a small real argument. Both are in the data and nothing in the generator tells them apart
except the labels it computes: depth, number of steps, lemmas used, hypotheses, size of the statement, whether a decision procedure closes the statement at once. See the limitations (section 9) before drawing
conclusions from either.

A **record** (JSON, one per line in `train/dev/test_iid/test_comp.jsonl.gz`) holds: `id` (`s<seed>-<index>`), `statement` (printed by the library's printer; `digest` is its hash), `pretty`, `code` (the statement
in the tokens of the typed term builder, `gym_build`: what a policy starts an episode from), `steps` (the proof as portable actions, `gym_acts.replay_portable`), and the labels `root_lemma`, `area`, `lemmas` (in
proof order), `depth`, `n_steps`, `n_lemmas`, `n_hyps`, `n_vars`, `closers`, `size`, `solved_by`, `semantic`, `model` (a witness: integers for which the hypotheses hold and the conclusion with them), `recipe`
(a hash of the set of lemmas and closers), and, added by the merge, `shape`, `skeleton` and `split`.

## 2. How a theorem is made (family F1, `backward_tree`)

A proof is a tree: the root is the goal, an inner node is a goal together with the step that reduces it to its children, a leaf is a goal that is not reduced further. Grown at random, backwards from a
lemma-shaped goal, and declaring the unproved leaves to be hypotheses, such a tree is a theorem *and* its proof, by construction.

1. **Plan.** From `random.Random("<seed>/<index>")`: the depth to aim for (2 to 8), the number of variables (3 to 5), the family. The index is the identity: nothing carries over from one theorem to the next.
2. **Root.** An *area* is drawn first (order, equality, divisibility, gcd, prime, existence, logic: with weights, so that the large areas do not crowd out the others), then a lemma of the area. A **witness
   model** (integers for the variables) is drawn, and arguments for the lemma are drawn (random small terms, polynomially tidy, see 4) until the lemma's *premises* hold at the model. The root goal is the lemma's
   conclusion at these arguments: true at the model by the lemma.
3. **Growth.** Along a main line of the planned depth and a few side branches, the engine applies library lemmas to open goals (`APPLY`, or `APPLY_WITH` when the lemma needs a term the goal does not give:
   the same unifier an agent uses, every step validated by a probe). A step is accepted only if every new goal is **still true at the model**; so every goal of the tree is true somewhere.
4. **Leaves.** A goal that holds at every sample point is closed (by a premise-free lemma, or by ring / linarith / nlinarith). A goal that fails at some point cannot be proved from nothing: it is **lifted**,
   it becomes a hypothesis of the theorem. A lifted goal that a decision procedure can prove from the other hypotheses is not lifted but closed (*derived*, up to three rounds), which removes redundant hypotheses.
5. **Consistency with the engine.** The engine *settles* goals (it closes a goal by a hypothesis up to conversion, not only syntactically: `0 < -x` closes `1 <= -x`), so the tree is made to agree with what
   the engine will do when the proof is replayed (finding F74); the finished proof is replayed in a fresh proof state and its term is checked by the kernel before the record exists.
6. **Theorem.** `forall vars, leaves -> root`, the proof is the tree's steps in the order the engine meets the goals.

Nothing here is trusted: a bug in the generator makes a theorem disappear (a `Reject` with a counted reason), never a false theorem appear. Every record is replayed from its own data by the generator
(`Generator.self_check`) and again by `synth_verify` (section 7).

## 3. Why this construction

* **The proofs are in the agent's own language.** The generator uses the engine and the portable action format of the gym, so everything generated can also be *said* by a policy and every step is one that
  the unifier, the menus and the probes support. A learner pre-trained on these records has seen the actions it will use on problems.
* **The statements are not templates.** Hypotheses come out of the proof (what could not be proved), so the same lemma gets different hypotheses depending on how it was reached; terms are random but tidy.
* **Labels come for free and are exact.** Depth, width, the lemmas used, the steps in order, and for every hypothesis whether it is needed (on the sample) are read off the tree, not estimated.
* **Verification does not need the generator.** A record carries its statement as code and its proof as data; the checker rebuilds both.

What this construction is *not*: a model of how people find proofs. Real proofs have a *forward* part (claims, derived facts, case splits, induction) that F1 does not produce. F1 teaches backward chaining,
unification and the decision procedures; further families (section 10) add case splits, claims and rewriting.

## 4. Quality: what is rejected, and why

A growth attempt ends in a record or in a counted rejection. About one attempt in five becomes a record (*pilot*: 21%). The rejections are the quality control; the card of a dataset carries the ledger.

| rule | why |
|---|---|
| **tidy terms**: no `x + 0`, `x * 1`, `x - x`, `--x`, `2 + 1`; nothing that cancels (`(a + b) - a`) or is a polynomial with fewer variables than it is written with (`synth_poly`: polynomial normal form, not patterns) | a theorem about `(a+b)-a` is a theorem about `b` in disguise; a training set of such statements teaches the wrong things |
| **at least the planned depth** (depth 2 to 5: plan-1; deeper: plan-2), at most 5 hypotheses, size at most `40 + 9 * depth` tokens | "varying depth" is only worth anything if the depth is real; unbounded statements are noise |
| **satisfiable**: at least 5 distinct assignments of a box of small integers satisfy the hypotheses (the witness model is one) | a vacuous theorem proves nothing |
| **not decoration**: the conclusion is not true for all sampled assignments when there are hypotheses; no hypothesis is true at every sample; at least half of the hypotheses are *needed* (an assignment where only this one fails and the conclusion fails) | a hypothesis that is always true, or never used, makes the statement look harder than it is |
| **every variable is used** | |
| **closable quota**: a statement that ring / linarith / nlinarith closes at once is kept with probability 0.2 (by its digest) | such a theorem is a calculation, not a search; the dataset keeps some, labelled `solved_by`, but is not dominated by them |
| **no root surprise**: the root does not settle at once, and is not a hypothesis | |

The semantic checks use the evaluator of `semantic.py` (the meaning of the library's symbols, written once) on a box of small integers. **They are a sanity layer, never a proof**: they can reject a
good theorem (the box is small), they cannot make a bad one pass the kernel. An existential conclusion is not evaluated at all (its witnesses may lie outside the box): such records have `semantic.exact =
false`. The evaluator and the kernel have never disagreed on a generated theorem (counted in every shard's meta file; a disagreement would be a bug in one of them and is not silently dropped).

## 5. The pool of lemmas

`synth_pool.Pool` selects the lemmas the generator may build on by the **structure** of their statements, not by name: a theorem of the library below the first bank problem, whose statement mentions only the
constants of one domain (`logic`, `Z`, `N`, `Nat`), whose binders are numbers or propositions (no motive, function or type), and which is not on the short list of the library's own machinery
(`NOT_A_BUILDING_BLOCK`: the quotient's well-definedness lemmas, representation lemmas, induction principles). A library that grows (sums, remainders, ...) feeds the pool by extending a vocabulary.
Roots additionally need premises the term builder can satisfy: lemmas with `Not` premises or nested binders and lemmas that conclude `False` are not roots (they can still be *applied*).
Today's Z pool has 123 lemmas in six areas; the pilot applied 71 of them. Lemmas that never appear are, mostly, the ring laws (`ring` closes those goals first), the case-split lemmas (`Z.le_total`,
`Z.lt_trichotomy`, `Z.eq_dec`: they are for family F2) and lemmas with negated premises.

## 6. Identity, shards, merge

* **The index is the identity.** Record *i* of seed *s* is a pure function of the configuration and *i*. A shard can be remade alone, a record regenerated to check it, the index space cut into ranges
  and handed to processes in any order. Gate S48 compares records made in order, in reverse, by a new generator, and by other processes with other `PYTHONHASHSEED`s, and pins the digests of a fixed range.
* **Three digests say what made the data**: the configuration (`Config.digest()`: everything that decides the data and nothing else), the generator (a hash of the source of `synth_*.py` that makes theorems),
  the world (the library below the limit, `bank.world_digest`). Each shard's `.meta.json` names the first two; the manifest names all three.
* **Merge** (`synth_merge.merge`, in index order, so the result does not depend on how the index space was cut): (1) *exclusion*: a theorem whose statement is that of a library lemma, a bank problem
  or a sub-theorem of one is dropped; (2) *duplicates*; (3) *shape cap*: a shape is a statement with numerals and variable names forgotten, at most 2 per shape; (3b) *skeleton cap*: a skeleton is a
  proof with its terms forgotten (kinds and lemmas of the steps), at most 25 per skeleton; (4) **splits**, functions of the record: `test_comp` = the record's *recipe* (the set of lemmas and
  decision procedures its proof uses) hashes into the held-out 6% of recipes, so no training record has that recipe (a test of putting known lemmas together in a way not seen in training); `test_iid` 3% and
  `dev` 3% by the statement's digest; the rest is `train`.

## 7. Verification

`synth_verify.verify_files` checks a dataset **from its data alone**, in a fresh process, in a freshly built library, with none of the generator's code. For every record it says the statement again from
its code and compares it with the printed statement, the digest and the readable text; replays the proof from the steps (each action refused or accepted by the engine, the proof term checked by the
kernel against that statement); reads the labels back from the data (id, steps, hypotheses, variables, lemmas in order, root lemma, closers, recipe, size of the statement) and checks that the witness model
satisfies the statement; every 20th record also finds the label `solved_by` again by running the decision procedures; and it hands the proof and the statement to the independent checker
(`recheck.SecondOpinion`, which shares no code with the kernel and fails closed). Tamper tests in S48 flip, one at a time, about thirty fields and steps and see that each is refused with a reason.
*Pilot*: 631 of 631 records pass, the second checker accepts all 631 proofs, in about 0.1 s per record.

## 8. Commands

```
python3 src/python/synth/synth_cli.py gen    --out DIR --seed 1 --attempts 160000 --shard 2000 --jobs 2     # resumable; the result does not depend on --jobs or --shard
python3 src/python/synth/synth_cli.py merge  --out DIR [--target 60000] [--shape-cap 2] [--skeleton-cap 25] [--subproblems data/dataset_v1/subproblems.json]
python3 src/python/synth/synth_cli.py verify --out DIR [--sample K] [--no-second-opinion]
python3 src/python/synth/synth_cli.py card   --out DIR
python3 tests/run_gates.py s48_synth_tests.py                                                               # the gate (about 70 s)
```

Cost (*pilot*, one core): about 0.1 s per attempt, about one attempt in five yields a record; the verification costs about 0.1 s per record. A dataset of 60 000 records is therefore about 300 000 attempts
(8 core-hours) and 2 core-hours of verification.

## 9. Limitations (read before using the data, or before quoting a result)

* **Synthetic.** The theorems are random combinations of library lemmas. They are true and proved, but often not what a person would ask, and the distribution of hypotheses is whatever the growth left over.
  A prover that does well on them has learned lemma application, unification and the decision procedures; it has not necessarily learned mathematics.
* **Depth is not difficulty.** It is the depth of the tree that produced the proof; a deep tree of order lemmas is a transitivity chain, and the `solved_by` label shows how many statements a decision
  procedure settles at once. For difficulty use a solver's behaviour (the baseline results of `docs/TRAINING.md`), not the label.
* **Vocabulary.** Integers only; order, equality, divisibility, gcd, primes, existence. Nothing here has the structure of a real olympiad problem (no sums, recursion, or case analysis beyond what lemmas give).
  That is what the problem bank is for; a pre-training effect on it is a hypothesis to be measured, not a property of this dataset.
* **Action kinds.** F1 proofs use `APPLY`, `APPLY_WITH`, `RING`, `LINARITH` and `NLINARITH`; a policy pre-trained on F1 alone has never seen `CLAIM`, `CASES`, `INDUCT`, `FORWARD` or `REWRITE`.
* **The evaluator is not the kernel.** The semantic labels (`sat_rate`, `needed`) are evidence about the *statement*, from a window of small integers. The kernel's acceptance of the proof is the only claim.
* **A pool that is small in places.** The gcd area has four lemmas, so its statements have few proof skeletons (hence the skeleton cap); the order and divisibility areas are rich.

## 10. Extending it

* **A new family** is a function `(rng, plan) -> Theorem` in `synth_tree.py` or a new module, registered in `Generator.record` and in `Config.families`; it must produce `steps` in the portable format, so that
  `self_check`, the merge and `synth_verify` need no change. Planned: F2 *case split* (claim a dichotomy such as `x <= y \/ y < x` with a premise-free lemma, then `CASES` with two grown branches, each with its
  own witness model); *claims and forward steps*; *rewriting* chains; *existential* statements proved by `USE`; Nat / N versions for the curriculum; induction once the library has finite sums.
* **A bigger library** needs no change: the pool reads the structure of statements. The digest of the world changes, so a dataset has to be regenerated (the manifest and S48's pins say when).
* **A different mix** is a configuration: `area_weights`, `depth_weights`, `nvars_weights`, `closer_keep`. Whatever decides the data is in `Config.fields`, and the configuration digest changes with it.
