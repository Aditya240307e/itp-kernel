# The problem bank

A dataset of formal olympiad problems for the gym environment (`ACTIONS.md`). A **problem** is a theorem of one
deterministic world (the library, then the problems in the order they were registered) plus what a dataset needs to
know about it: where it comes from, what it says in words, **a check that the formal statement says what the words
say**, a reference proof, and the regimes it is attacked in. Everything is data: the agent sees a goal, never text.

Files: `bank.py` (design, building, checks, export), the problem modules (`bank_lib` 21 and `bank_lib2` 51 graded problems; `bank_amc`, `bank_amc2`, `bank_aime2` the AMC/AIME tier; `bank_imo`, `bank_imo2`, `bank_imo3`,
`bank_imo4`, `bank_classic` the IMO tier; helpers `bank_kit.py`, `bank_spec.py`), `bank_all.py` (the whole bank, 268 problems, as one world), `semantic.py` (reference semantics and faithfulness checks), `imo_truth.py`
(brute-force evidence for statements without a proof), `bank_dups.py` (the duplicate screen), `bank_baseline.py` (blind-search difficulty), `bank_split.py` (train/dev/test), `lib_corpus.py` (the proofs of the
library lemmas as action sequences), `subproblems.py` (the labels of every action and the theorems a reference proof contains), `dataset.py` (build, verify and describe the dataset). Gates: `s41_bank_tests.py`
and `s43_bank2_tests.py` (the graded problems and the machinery: `s41_golden.json`, `s43_golden.json`), `s49_contest_tests.py` (the contest problems: provenance, statements, the statement-only tier, duplicates;
`s49_golden.json`), `s50_proofs_tests.py` (the premises mask and the proofs of the contest tier, with the independent checker), `s47_hindsight_tests.py` (the sub-theorems).

```python
import bank_all
bank = bank_all.load()                         # the 268 problems: builds the world once (about a minute and 2.5 GB)
bank.names(level=2, topic='divisibility')      # the index
ep = bank.episode('imo1959_p1')                # a gym Episode: the statement as a goal, with the premises of the prefix regime
ep = bank.episode('imo1969_p1', 'strict')      # ... with the lemmas written for the problem hidden
rep = bank.check('imo1959_p1', demo=True)      # faithfulness + sayability + the reference proof through the action language
bank.export('bank.json', demos, strict_demos)  # the dataset (section 7)
```

## 1. What a problem is

| field | meaning |
|---|---|
| `name` | the unique name of the problem **and** of the theorem that proves it |
| `text` | the problem in words (ASCII) — what a human checks the formal statement against |
| `source`, `year` | `IMO 1959 P1`, `ISL 2019 N1`, `folklore (Sophie Germain)`, … |
| `topics` ⊂ `TOPICS`, `level` 1–5 | the classification (levels below) |
| `kind` | `prove` · `answer` ("determine all …": the answer is **part of the statement**, an equivalence with the solution set; `answer` holds it in words) · `exists` (an example is asked for) |
| `spec` | the intended meaning as Python (`semantic.Spec`, section 3) |
| `helpers` | the lemmas of the library written for this problem (hidden in the strict regime) |
| `strict` | the reference proof is **self-contained**: it proves its lemmas inline and replays when the helpers are hidden. Default: true for a problem with a build function |
| `build` | `build(w, b)` declares the helpers and the theorem; `b.statement_ready()` marks the end of the definitions the *statement* needs |

Levels: **1** one arithmetic step after the introductions (`ring` / `linarith` / `nlinarith`) · **2** a short proof with
one idea (a case split, a witness, one induction) · **3** an olympiad lemma (15–60 actions) · **4** an olympiad
problem with a short solution (60–200 actions) · **5** a hard olympiad problem.

A problem is registered in one of two ways:

```python
BANK.declare('imo1959_p1', text=..., source='IMO 1959 P1', year=1959, topics=(...), level=2, spec=Spec(...))   # a library stage already proves it

@BANK.problem('ident_diff_squares', text=..., source='folklore', topics=('algebra', 'identity'), level=1, spec=Spec(...))
def _(w, b):                                   # declares the theorem in the world
    s = _syms(w); Z, Eq = s.Z, s.Eq
    with w.nota.using(Z):
        @thm(w, 'ident_diff_squares', [Z, Z], lambda a, c: Eq((a + c) * (a - c), a ** 2 - c ** 2))
        def _(p): p.intro('a', 'c'); p.ring()
```

Statement conventions: natural numbers in a problem about integers are `Z` with the hypothesis `0 ≤ n` (as the problem
says it); "determine all" problems are an equivalence with the solution set (`kind='answer'`); the theorem has the
problem's name; nothing the text does not say is assumed. A problem with lemmas declares them first (`p.have` or a
`thm`) and, to be strict, **proves them inline again in the main proof** (`imo1969_p1`, `imo1964_p1a/b`).

## 2. Regimes and premises

The agent attacks a problem inside the world that precedes it; the engine's `premises` is a number of globals (by id).

* **prefix** — every global before the theorem, the problem's own helper lemmas included. The helpers are on the
  menu: this regime measures how well the agent *uses* a good lemma (and a helper that *is* the statement makes the
  problem trivial — that is what it is for: `imo1964_p1a` prefix is one `APPLY`).
* **strict** — every global before the problem's block, its definitions aside. The helpers are not offered: the
  agent has to find the ideas.

A problem without helpers has the same two regimes (one number). An `Episode` carries its premises, so episodes of
different regimes interleave on one engine. A global above the limit cannot be said by the term builder
(`BuildError('no global n')`), so the hidden lemmas are hidden from *every* way of naming them, not only from the menu.
The world is deterministic: a problem is identified by its name and the digest of the world below its block.

## 3. Is the formal statement the problem? (`semantic.py`)

The kernel proves that a statement follows from the library. It cannot say that the statement **means** what the
problem says: a wrong constant, a `<` for a `≤`, a forgotten hypothesis gives a theorem that is true, proved, and not
the problem. For a dataset that is the main risk, and `semantic.py` is the independent check of it:

1. **Reader.** The statement (a kernel term) is read back into a small AST (`Semantics.decode`): quantifiers over `Z`,
   `N` and functions `Z → Z`, the connectives, equations, `≤`/`<`, `∣`, `gcd`, primality, numerals, `+ − × ^`. Anything
   else is `Unsupported` — the problem cannot be checked and the gate says so.
2. **Evaluator.** The AST is evaluated on Python integers. The meaning of each library symbol is written **once**
   (`VALUE_PRIMS`, `PROP_PRIMS`: a few dozen lines to audit). `audit_prims` compares each entry with the library's
   *definition* over a window, and `audit_theorems` runs **every supported theorem of the library (383)** on its window
   — a table entry that contradicted a proved theorem fails (the gate also corrupts the tables and demands that the
   audits notice).
3. **Spec.** The person who formalises writes the same statement **again, as Python**, independently:

   ```python
   Spec(('Z',), (lambda n: n >= 0,), lambda n: math.gcd(21*n+4, 14*n+3) == 1,        # params, hypotheses, conclusion
        atom=(lambda n: 21*n+4, lambda n: 14*n+3, lambda n: 1))                         # the numbers the atom is made of
   ```

   * `params`: the kinds of the leading quantified variables (`'Z'`, `'N'`, `'fn'` for a function); `hyps`: one predicate per
     hypothesis; `concl`: the conclusion, or `(lhs, rhs)` for an equivalence.
   * Windows: `Z` is −8..8, `N` 0..12 by default; `windows={'z':…, 'n':…, 'ex':…, 'atom':…}` overrides them (`ex`: the
     window of the *witnesses* of existentials; `atom`: of the variables bound inside the conclusion).
   * Functions: tried on a table of test functions (`standard_functions`, plus `extra_fns` — functions that satisfy the
     equation only partly are exactly what tells "for all" from "for some").
4. **Comparison.** Every part (each hypothesis; the conclusion, or each side of the equivalence) must agree with the
   Python predicate at every point of the box; the statement must be **non-vacuous** (some point satisfies all
   hypotheses; each side of an equivalence is true somewhere and false somewhere).
5. **Atoms.** A true statement has a true conclusion wherever it matters, so a truth table cannot tell `2ab ≤ a²+b²`
   from `ab ≤ a²+b²`, or `gcd(21n+4, 14n+3)=1` from `gcd(20n+4, 14n+3)=1`. `atoms` gives the intended *numbers* of every
   relation atom of the conclusion (both sides of `=`, `≤`, `<`; `(a, b)` of `∣`; `(a, b, d)` of `is_gcd`), and they must
   match at every row.
5b. **Hypothesis atoms.** The same for the relation atoms of the *hypotheses* (`hatoms=[(lhs, rhs), ...]`, in the order they
   are written, no quantifier inside): `x² + y² = 3 z²` has the same integer solutions as `x² + y² = -3 z²` (both only the
   origin), so no truth table sees a slip in that hypothesis — the numbers do (`x2_y2_3z2`).
6. **Body.** A closed existential is true as soon as one witness works, so truth tables cannot see whether a connective or
   an inequality *inside* the body is the right way round. `body=(kinds, pred)` compares the body of the leading
   existentials at every combination of a point and the witnesses of the window (`imo1984_p2`, `imo1969_p1`,
   `square_mod_four`, `cube_mod_nine`, `odd_square_mod_eight`).
7. **Negative controls.** `mutants(ast)` makes every plausible slip of the statement, one at a time — a changed
   numeral, `<`↔`≤`, `+`↔`−`, a swapped argument, `∧`↔`∨`, a dropped or extra negation, a changed quantifier, a changed
   hypothesis, … (687 over the first 28 problems, 1415 over the 54 added). Each one **must differ** from the statement in some part or in the numbers of an atom.
   A mutant that survives is an error: the box would not notice that slip when the formalisation makes it. A mutant
   that genuinely means the same thing (a divisor and its negative, the sign of a base inside a square, `c` for `-c` under a symmetric window) is listed in
   `skip_mutants={'description substring': 'reason'}` — each with its reason (5 problems of the first 28, 9 of the 54; the divisor-and-its-negative slip is the usual one).

**What this is not.** It is a check on small values, never a proof of faithfulness: it cannot tell a statement that
differs only outside the window, and the text → spec step is still a human step (that is why the spec is written
independently and with the text next to it). What it removes is the slip that is invisible when one reads a printed
statement. A statement with something the reader does not know cannot be checked; add the symbol to the tables
(section 8) instead of leaving a problem unchecked.

## 4. The reference proof and the action language

Every problem has a proof written in the authoring DSL. `bank.demonstrate(name, regime)` replays it **through the action
language** (`recorder.py`, `ACTIONS.md` §6): the author's calls become `Action`s (claims, rewrites with said terms, an
induction with a chosen principle, `nlinarith`, …), the kernel checks the resulting term, and the portable steps are
the dataset's demonstration. The proof is replayed again with the problem's helpers hidden when `strict` — that is what
"self-contained" means and it is checked, not assumed. The gate also runs every demonstration through an independent
proof checker and replays the exported JSON in a fresh engine over a rebuilt world.

## 5. Difficulty: blind search (`bank_baseline.py`)

```
python3 src/python/problems/bank_baseline.py --probes 4000 --json out.json [names…]     # --regime prefix|strict  --cost  --time S
```

The plain searcher (iterative deepening over the menus, no learning) runs on each statement in each regime with a
budget of **probes** (not seconds: reproducible). A "solved" is a kernel-checked proof. At 3000 probes only the
identities, `sum_squares_zero` and the trivial prefix regime of `imo1964_p1a` fall (13 of 56 rows): inequalities need the
right square as a hint term, the number-theory lemmas need the case split — a good signal that the bank asks for
ideas and not only for search.

## 6. The problems (268)

**Two tiers.** *Proved* (187): the formal statement, its checked meaning and a kernel-checked reference proof (an action sequence, replayed in a fresh engine and re-checked by the independent checker: S50).
*Statement-only* (81): the formal statement, its checked meaning (section 3), and brute-force evidence over ranges far beyond the box that it is true (`imo_truth.py`), but no proof. A statement-only problem is a
task all the same -- whatever an agent proves for it is kernel-checked -- and one that some run solves is listed in the experiment report (`Statement-only problems solved`). They are lower bounds on solvability:
read the `proved` and `unproved` subsets separately.

| module (problems) | proved | statement-only | what |
|---|---|---|---|
| `bank_lib` (21), `bank_lib2` (51) | 72 | 0 | the graded problems of S41/S43: identities, inequalities, divisibility, Diophantine and functional equations, nine IMO problems (1959 P1, 1964 P1 a/b and P2, 1969 P1, 1982 P1, 1984 P2, 1988 P6, 2019 P1) |
| `bank_amc` (23), `bank_aime2` (8) | 30 | 1 | AIME problems (texts paraphrased; 23 `contest`, 8 `adapted`), answers compared with a public record |
| `bank_amc2` (72) | 72 | 0 | AMC-style problems written for the bank (`origin = style`): **not** AMC problems, no claim of provenance |
| `bank_imo` (28), `bank_imo2` (10) | 0 | 38 | IMO and ISL problems, statement only |
| `bank_imo3` (13) | 13 | 0 | olympiad lemmas and a few contest problems with proofs (IMO 1977 P6, IMO 1983 P6, ISL 2005 N3, Vasc's inequality, ...) |
| `bank_imo4` (15), `bank_classic` (27) | 0 | 42 | more shortlist problems, APMO, and classical results and exercises of olympiad number theory (Euclid, Wilson, Bertrand, Lagrange, ...), statement only |

Levels (how much *thinking* the problem asks for, not how long the proof is): 1: 37, 2: 67, 3: 75, 4: 45, 5: 44; kinds: 144 `prove`, 121 `answer` ("determine all ..."), 3 `exists`. Of the 81 statement-only problems 43 are
level 5 and 31 level 4. **The IMO tier is mostly statements**: of 41 IMO, 18 ISL and 1 APMO problems, 12 have a proof; what an agent can be asked to do on the rest is to find one.

**Provenance.** Every problem carries `origin` and `verified` (gate S49 pins the composition, because a paper has to report it):

| `origin` | meaning | problems |
|---|---|---|
| `contest` | a problem of that contest; the text is a paraphrase written for this bank (some from memory) | 72 |
| `adapted` | a contest problem changed to fit the formal vocabulary (a part of it, integers instead of reals, a fixed parameter) | 19 |
| `style` | written for the bank in the manner of a contest; no claim to be a contest problem | 72 |
| `folklore` | classical identities, inequalities and named results | 105 |

| `verified` | meaning | problems |
|---|---|---|
| `answer-checked` | the answer or the claim was recomputed by brute force (`imo_truth.py`, or the brute-force evidence of the problem) and, for contest problems, compared with a public record of it (AoPS threads, a public dataset of AIME answers) | 114 |
| `recalled` | the text or the numbers were written from memory and not compared with a source; marked so | 9 |
| `none` | nothing to compare (style problems, folklore) | 145 |

The texts of AMC/AIME/IMO/ISL problems here are **paraphrases**, and the source label (`IMO 1982 P4 (a)`, `ISL 2012 N6`) is a pointer, not a claim of verbatim text; the formal statement is checked against the text
by section 3, which is the part that matters for a prover. A few ISL labels were assigned from memory (`verified = recalled`); the text of ISL 2000 N4, 2007 N2, 2012 N4/N6, 2013 N1, 2014 N2, 2016 N4, 2021 N1, APMO 2019 P1,
IMO 2016 P4 and the nine oldest IMO problems was compared with the source on the web.

**Duplicates.** Two problems that say the same thing are a leak when a split separates them. `bank_dups.py` screens in three layers (gate S49, section E): (1) identical formal statements; (2) the same meaning under any
renaming of the variables (the statements are decoded and compared by the *numbers* their relation atoms are made of, so `a^3 + b^3 >= a^2 b + a b^2` and `a^3 + b^3 >= a b (a + b)` are told apart and `x + y = 2` /
`a + b = 2` are not); (3) alike texts (a pair above a threshold is reviewed by a person: one family, or entered in `REVIEWED_DIFFERENT` with the reason). The audit replaced four AMC-style problems that were identical in
meaning to graded ones and grouped near-duplicates into families (`bank_split.FAMILIES`); a family is never separated by a split.

**Splits** (by family, frozen salt, section 7b): 190 train / 20 dev / 58 test. Proved: 132 / 15 / 40. The test split holds 11 IMO (3 proved), 2 ISL (1 proved), 7 AIME, 17 folklore, 12 AMC-style and 9 classical
problems: 40 proved and 18 statement-only. Dataset v1 had 82 problems (56 / 10 / 16); ten of them were library lemmas in disguise (`not_dvd_between`, `square_mod_four`, ...) and are not problems any more, and
`ident_pell_product` moved from train to test when the audit found it to be a variant of `ident_brahmagupta`; every other problem of v1 has the same statement and the same split.

### 6a. The first graded problems

The table below lists the problems of the first modules by level, as they were when the bank had 82 problems. Ten of them are library lemmas now and no longer problems of the bank (`N.induction3`, `cube_minus_n_div6`,
`cube_mod_nine`, `dvd_coprime_product`, `five_dvd_fifth_power`, `not_dvd_between`, `square_mod_four`, `square_mod_three`, `sum_squares_zero`, `three_dvd_sum_squares`); the other 72 are unchanged.

| level | problems |
|---|---|
| 1 | `ident_diff_squares`, `ident_sophie_germain`, `ident_brahmagupta`, `ident_consecutive_four`, `ident_fourth_powers`, `ineq_am_gm_two`, `ineq_square_sum_three`, `ineq_four_ab`, `ineq_cauchy_two`, `sum_squares_zero`, `ident_cube_sum`, `ident_cube_diff`, `ident_three_cubes`, `ident_pell_product`, `ineq_three_squares`, `ineq_shifted_squares`, `ineq_cross_terms`, `find_all_x2_eq_x` (answer) |
| 2 | `imo1959_p1` (IMO 1959 P1), `not_dvd_between`, `square_mod_four`, `sum_two_squares_mod4`, `odd_square_mod_eight`, `cube_mod_nine`, `cube_minus_n_div6`, `pow_sub_dvd`, `coprime_consecutive`, `no_square_between`, `N.induction3` (a library principle, no spec), `ineq_cubes_two`, `ineq_cubes_three`, `ineq_pairs_eight`, `ineq_fourth_powers_three`, `ineq_four_fourth_powers`, `quad_form_zero`, `square_mod_three`, `consecutive_product_even`, `three_consecutive_div3`, `five_dvd_fifth_power`, `fourth_power_odd_16`, `gcd_shift`, `no_square_pronic`, `five_dvd_six_pow_minus_one`, `three_dvd_four_pow_plus_five`, `two_pow_gt_n`, `three_consecutive_not_squarefree` (exists), `ineq_cauchy_three`, `ineq_chebyshev_cubes`, `ineq_a_plus_b_cubed`, `ineq_square_products` |
| 3 | `sqrt2_irrational`, `three_dvd_sum_squares`, `imo1984_p2` (IMO 1984 P2, exists), `imo1969_p1` (IMO 1969 P1), `imo1964_p1b` (IMO 1964 P1b), `imo1964_p1a` (IMO 1964 P1a, answer), `diff_squares_not_2_mod_4`, `pythag_three`, `cube_sum_not_4_mod_9`, `gcd_sum_product`, `dvd_coprime_product`, `square_not_prime`, `bernoulli_int`, `find_all_xy_eq_x_plus_y` (answer), `find_all_n_plus_1_dvd` (answer), `frobenius_3_5`, `fe_additive` (answer), `no_f_f_succ`, `sqrt3_irrational`, `pell_infinitely_many`, `no_x2_minus_2y2_eq_3`, `three_squares_mod_eight`, `schur_nonneg`, `n5_minus_n_div30`, `imo1964_p2` (IMO 1964 P2), `n4_n2_1_composite` |
| 4 | `imo2019_p1` (IMO 2019 P1, answer: all functions with `f(2a)+2f(b)=f(f(a+b))`), `x2_y2_3z2` (infinite descent), `fe_f_x_plus_f_y` (answer), `imo1982_p1` (IMO 1982 P1, answer: `f(1982) = 660`), `fe_quadratic` (answer) |
| 5 | `imo1988_p6` (IMO 1988 P6, Vieta jumping) |

Nine of them are IMO problems (1959 P1, 1964 P1 a/b and P2, 1969 P1, 1982 P1, 1984 P2, 1988 P6, 2019 P1). The level is how much *thinking* the
problem asks for, not how long the proof is: the case enumerations (`pythag_three`, `cube_sum_not_4_mod_9`,
`three_squares_mod_eight`: 107–197 actions) are level 3, `imo1964_p1a` is one `APPLY` in the prefix regime. Inequalities
that are over the reals in the original (`imo1964_p2`, `schur_nonneg`, the AM-GM family) are stated for integers (the
`notes` of the problem say so) until the ordered-field layer exists.

`sqrt2_irrational` and `imo2019_p1` are declared (the library proves them) and are not self-contained: their strict
regime is not available (`strict=False`). Eight problems have two reference proofs (prefix and strict).

## 7. Export

`bank.export(path, demos, strict_demos)` writes one JSON file:

```
{"format": 1, "globals": [names in id order], "world_digest": …,
 "problems": [{"name", "text", "source", "year", "topics", "level", "kind", "answer", "notes",
               "statement" (printed), "statement_digest", "statement_ast", "statement_code" (the typed-builder tokens, global ids),
               "gid", "premises": {"prefix": n, "strict": m}, "helpers", "world_digest" (below the block), "has_spec",
               "demonstration": {"regime", "steps" (portable actions), "length"}, "demonstration_strict": {…}}]}
```

The ids in `statement_code` and in the steps are those of *this* world: a consumer names the world by
`world_digest` and rebuilds it (`bank_lib.load()`); the digest is deterministic (gated) and every statement digest is
pinned in `s41_golden.json` — a change to a printed statement fails the gate until it is accepted consciously
(`S41_UPDATE_GOLDEN=1`).

## 7b. The dataset (`dataset.py`)

```
python3 src/python/problems/dataset.py OUT_DIR           # bank.json, splits.json, library.json, DATASET_CARD.md, manifest.json (digests)
python3 src/python/problems/dataset.py --verify OUT_DIR  # replay every demonstration in a fresh engine over a rebuilt world
```

The released copy of dataset v2 is `data/dataset_v2/` (its digests are checked by gate S45, section J; `make dataset-verify` replays every recorded proof and sub-theorem in a fresh engine over a rebuilt world, a few
minutes; example 07 reads it without the kernel). Dataset v1 (82 problems) is gone from the tree (it is in the git history, in the commits before the one that introduced dataset v2); the result files of the first baseline runs, made on it, are in
`data/results/v1/` and are a historical record (`data/results/README.md`, `docs/TRAINING.md` section 5).

* `bank.json` — section 7's file for all 268 problems, each with its `split`, `family`, `origin` and `verified` (and `proved`: a statement-only problem has no `demonstration`). 187 demonstrations (6 033 actions; mean 32, median 11,
  longest 466; 4 154 actions in the train split); 8 problems have a second proof for the strict regime (538 actions).
* `splits.json` — the split policy and the splits. **By family, as a pure function of the family key** (`bank_split.py`): the two parts of
  one olympiad problem, a statement in other clothes (`4ab ≤ (a+b)²` and `2ab ≤ a²+b²`), an argument with another constant
  (`sqrt2`/`sqrt3`) are in one split; the split of a family is a hash of its key under a frozen salt (found once, the smallest
  that satisfies the coverage requirements: every split has problems of levels 1–3, test has IMO problems and six topics), so
  **adding problems never moves an old one** (the exception is a problem that the duplicate audit puts into the family of another one). Version 2: 190 train / 20 dev / 58 test (test has `imo2019_p1`, `imo1964_p1a/b`,
  `x2_y2_3z2`, `frobenius_3_5`, and eight statement-only IMO problems, …). Dev picks the epoch and the search parameters; test is read once.
* `library.json` — the proofs of the library's 453 lemmas (2 677 actions) as action sequences, with a split by lemma name (train 319 / dev 40 / test 94; the
  imitation data besides the bank); a lemma whose statement is a bank problem's is marked `bank_duplicate` (4) and left out of training. Six lemmas of the library are NOT in the corpus because the recorder cannot turn their authored proofs into actions (`unreplayed` in the file, with the reason: `Z.induction_on` and `Z.mk_inj` (a `match` term, a code that builds a different term), `Z.dvd_fact` and `Z.one_le_pow` (the author's script uses an induction hypothesis the engine does not have), `Z.pow_nonneg` and `Z.pow_pos` (the author's `apply` leaves one goal where the engine's action leaves none: the engine settles the goal itself)); they are premises like the rest, they just have no demonstration.
* `DATASET_CARD.md` — generated from the files (counts, splits, kinds of action), pinned by the manifest.

## 8. Adding to the bank

1. Write the statement and its **text**, then the **spec** (independently — do not copy the formal statement).
2. Build it (`bank_lib.load(only=[name])`), run `bank.check(name, demo=True)` and read the errors: *disagrees with the
   intended meaning* (the formalisation or the spec is wrong), *go unnoticed by the box* (sharpen the spec with `atoms`
   or `body`, or explain the mutant in `skip_mutants` **only** if it truly means the same), *not sayable* (the term
   builder cannot say the statement — a `match`/`let`, say), *does not replay* (the proof uses something the recorder
   has no translation for, or is not self-contained).
3. New vocabulary: add the symbol to `VALUE_PRIMS`/`PROP_PRIMS` (and the reader if it is a new shape), and let
   `audit_prims` / `audit_theorems` confirm the entry against the definitions and the proved theorems.
4. Run the gate (`S41_ONLY=B python3 tests/problems/s41_bank_tests.py` for the bank content), update the golden digests on purpose.

## 9. Gates S41, S43, S49 and S50

**S41** (126 checks, about 100 s) — the first 28 problems and the machinery:
A. the meaning tables against the library definitions and 383 proved theorems, and corrupted tables are noticed ·
A2. the reader and the evaluator, wrong specs are rejected, atoms, body, vacuity, equivalences · B. every problem:
metadata, faithfulness, all mutants caught, skip reasons, golden digests · C. replay in both regimes, the independent
checker on 26 proofs, JSON round-trip in a fresh engine, truncated/extended/foreign demonstrations refused ·
D. premises, helper visibility through the builder, menus, `Episode` playthroughs · E. export, determinism, digests,
bookkeeping errors · F. banks of deliberately wrong formalisations are rejected, right ones accepted, declared
non-theorem / lazy build / unsayable statement.
Environment: `S41_ONLY=A,A2,B,…`, `S41_FAILFAST`, `S41_TRACE`, `S41_UPDATE_GOLDEN`, `S41_HEAVY`.

**S43** (about 7 min) — the 54 added problems and the dataset:
A. `hatoms` (the `-3 z²` mutant survives without them, wrong numbers and counts are rejected), the recorder's claims under
the hypotheses of the outer goal (`sqrt3_irrational`, `x2_y2_3z2`) and a `linarith` whose hint is the goal (`schur_nonneg`),
`induction_on` with the starting point as an extra argument and what it refuses · B. every new problem: faithfulness with 1415
mutants all caught, statements pinned in `s43_golden.json`, the old 28 still those of `s41_golden.json`, no two problems with the
same statement · C. replay in both regimes (1384 actions), the independent checker on all but the seven proofs it takes longest on
(15 s to minutes; `S43_HEAVY=1` includes them), JSON round-trip in a fresh engine · D. helper lemmas hidden in the strict regime,
`Episode` playthroughs · E. splits (partition, no family straddles, coverage, purity: the same split with fewer problems),
the card, the library corpus replays from its JSON.
Environment: `S43_ONLY=A,B,…`, `S43_FAILFAST`, `S43_TRACE`, `S43_UPDATE_GOLDEN`, `S43_HEAVY`.

**S49** (the contest problems: the AMC/AIME tier and the IMO tier, 196 problems on top of the 72 graded ones) — A. the registry: 268 problems in the order of the modules, the two tiers, the *composition* (contest class x origin x
verified x tier), levels, kinds and the numbers per split are pinned, because they are what a paper has to report · B. the evidence: every statement-only problem has a brute-force claim over a range far beyond the box,
and every claim holds · C. the statements, in the built world: each contest problem is read back and agrees with its spec, its mutants are told from it or explained, it is sayable in the typed term builder, a statement
without a proof is true on the box; the statements are pinned by digest (`s49_golden.json`) and pairwise different · D. the tier: a false statement-only problem is rejected, a true one accepted, and a statement-only
problem that is no proposition, declares a theorem, has helpers, an unknown `origin` or an unknown `verified` is refused · E. duplicates: no two problems say the same thing (three layers, `bank_dups.py`), with negative controls.

**S50** (the premises mask and the proofs of the contest tier) — A. `gym_premises.Premises` as values (normal form, `sees`, JSON round trip) · B. the leak: in a small bank of three problems, with a plain prefix the theorem of an
*earlier* problem is on the menu of a later one and one action finishes it; with the mask it is not on the menu, the term builder refuses to say it and the action is refused · C. the mask in the whole bank: the premises of every
problem are exactly the library and its own block, no theorem or helper of another problem is visible, a proved problem does not see its own theorem · D. the 115 proved contest problems: each reference proof replays as a
kernel-checked action sequence, survives a JSON round trip, mentions only globals its premises offer, and depends on no axiom (quotient primitives only) · E. the second opinion: the independent checker
(`recheck.SecondOpinion`, restricted to the globals a proof needs) accepts every one of those proofs and rejects a proof for another problem's statement.

## 10. Toward ISL N1–N3, A1–A3

What the bank still needs, in the order the problems need it: `Z.modeq` and a congruence calculus; factorial and
finite sums/products (an `N`-indexed recursion) for Euclid, Wilson, telescoping; floor and digits; the ordered-field
layer (ℚ then ℝ as one auditable structure) for the A problems — each new symbol needs its entry in the meaning tables
and its mutants in `mutants()`; more of the level-3 lemmas that the ISL solutions are built from (Vieta jumping,
infinite descent, bounding arguments, the standard inequality toolbox).

Limits of the reference proofs that the new problems exposed: `INDUCT_WITH` cannot generalise variables that were introduced
before the induction (a descent is proved by `Z.strong_induction` with an explicit statement, not by a bare induction);
`linarith` has no integer reasoning (no gcd, no parity: `8a + 1 = 8n + 2` is refuted by an explicit case split on
`K ≤ -1 / K = 0 / K ≥ 1`); `x**2` is `Z.sqr x` after normalisation and not definitionally `Z.pow x 2`, so a lemma used inside a
motive is stated with `x*x`.
