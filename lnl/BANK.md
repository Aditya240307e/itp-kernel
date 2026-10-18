# The problem bank

A dataset of formal olympiad problems for the gym environment (`ACTIONS.md`). A **problem** is a theorem of one
deterministic world (the library, then the problems in the order they were registered) plus what a dataset needs to
know about it: where it comes from, what it says in words, **a check that the formal statement says what the words
say**, a reference proof, and the regimes it is attacked in. Everything is data: the agent sees a goal, never text.

Files: `bank.py` (design, building, checks, export), `bank_lib.py` (the problems), `semantic.py` (reference semantics
and faithfulness checks), `bank_baseline.py` (blind-search difficulty), gate `s41_bank_tests.py` (127 checks),
`s41_golden.json` (digest of every statement).

```python
import bank_lib
bank = bank_lib.load()                         # builds the world once (a couple of seconds)
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
6. **Body.** A closed existential is true as soon as one witness works, so truth tables cannot see whether a connective or
   an inequality *inside* the body is the right way round. `body=(kinds, pred)` compares the body of the leading
   existentials at every combination of a point and the witnesses of the window (`imo1984_p2`, `imo1969_p1`,
   `square_mod_four`, `cube_mod_nine`, `odd_square_mod_eight`).
7. **Negative controls.** `mutants(ast)` makes every plausible slip of the statement, one at a time — a changed
   numeral, `<`↔`≤`, `+`↔`−`, a swapped argument, `∧`↔`∨`, a dropped or extra negation, a changed quantifier, a changed
   hypothesis, … (687 over the bank). Each one **must differ** from the statement in some part or in the numbers of an atom.
   A mutant that survives is an error: the box would not notice that slip when the formalisation makes it. A mutant
   that genuinely means the same thing (a divisor and its negative, the sign of a base inside a square, `c` for `-c` under a symmetric window) is listed in
   `skip_mutants={'description substring': 'reason'}` — 17 are, each with its reason, the gate prints them.

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
python3 bank_baseline.py --probes 4000 --json out.json [names…]     # --regime prefix|strict  --cost  --time S
```

The plain searcher (iterative deepening over the menus, no learning) runs on each statement in each regime with a
budget of **probes** (not seconds: reproducible). A "solved" is a kernel-checked proof. At 3000 probes only the
identities, `sum_squares_zero` and the trivial prefix regime of `imo1964_p1a` fall (13 of 56 rows): inequalities need the
right square as a hint term, the number-theory lemmas need the case split — a good signal that the bank asks for
ideas and not only for search.

## 6. The problems (28)

| level | problems |
|---|---|
| 1 | `ident_diff_squares`, `ident_sophie_germain`, `ident_brahmagupta`, `ident_consecutive_four`, `ident_fourth_powers`, `ineq_am_gm_two`, `ineq_square_sum_three`, `ineq_four_ab`, `ineq_cauchy_two`, `sum_squares_zero` |
| 2 | `imo1959_p1` (IMO 1959 P1), `not_dvd_between`, `square_mod_four`, `sum_two_squares_mod4`, `odd_square_mod_eight`, `cube_mod_nine`, `cube_minus_n_div6`, `pow_sub_dvd`, `coprime_consecutive`, `no_square_between`, `N.induction3` (a library principle, no spec) |
| 3 | `sqrt2_irrational`, `three_dvd_sum_squares`, `imo1984_p2` (IMO 1984 P2, exists), `imo1969_p1` (IMO 1969 P1), `imo1964_p1b` (IMO 1964 P1b), `imo1964_p1a` (IMO 1964 P1a, answer) |
| 4 | `imo2019_p1` (IMO 2019 P1, answer: all functions with `f(2a)+2f(b)=f(f(a+b))`) |

`sqrt2_irrational` and `imo2019_p1` are declared (the library proves them) and are not self-contained: their strict
regime is not available (`strict=False`).

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

## 8. Adding to the bank

1. Write the statement and its **text**, then the **spec** (independently — do not copy the formal statement).
2. Build it (`bank_lib.load(only=[name])`), run `bank.check(name, demo=True)` and read the errors: *disagrees with the
   intended meaning* (the formalisation or the spec is wrong), *go unnoticed by the box* (sharpen the spec with `atoms`
   or `body`, or explain the mutant in `skip_mutants` **only** if it truly means the same), *not sayable* (the term
   builder cannot say the statement — a `match`/`let`, say), *does not replay* (the proof uses something the recorder
   has no translation for, or is not self-contained).
3. New vocabulary: add the symbol to `VALUE_PRIMS`/`PROP_PRIMS` (and the reader if it is a new shape), and let
   `audit_prims` / `audit_theorems` confirm the entry against the definitions and the proved theorems.
4. Run the gate (`S41_ONLY=B python3 s41_bank_tests.py` for the bank content), update the golden digests on purpose.

## 9. Gate S41 (127 checks, ≈150 s)

A. the meaning tables against the library definitions and 383 proved theorems, and corrupted tables are noticed ·
A2. the reader and the evaluator, wrong specs are rejected, atoms, body, vacuity, equivalences · B. every problem:
metadata, faithfulness, all mutants caught, skip reasons, golden digests · C. replay in both regimes, the independent
checker on 26 proofs, JSON round-trip in a fresh engine, truncated/extended/foreign demonstrations refused ·
D. premises, helper visibility through the builder, menus, `Episode` playthroughs · E. export, determinism, digests,
bookkeeping errors · F. banks of deliberately wrong formalisations are rejected, right ones accepted, declared
non-theorem / lazy build / unsayable statement.
Environment: `S41_ONLY=A,A2,B,…`, `S41_FAILFAST`, `S41_TRACE`, `S41_UPDATE_GOLDEN`, `S41_HEAVY`.

## 10. Toward ISL N1–N3, A1–A3

What the bank still needs, in the order the problems need it: `Z.modeq` and a congruence calculus; factorial and
finite sums/products (an `N`-indexed recursion) for Euclid, Wilson, telescoping; floor and digits; the ordered-field
layer (ℚ then ℝ as one auditable structure) for the A problems — each new symbol needs its entry in the meaning tables
and its mutants in `mutants()`; more of the level-3 lemmas that the ISL solutions are built from (Vieta jumping,
infinite descent, bounding arguments, the standard inequality toolbox).
