# src/python/problems -- the problem bank

**268 formal problems** over the integers and naturals, stated in the kernel: identities and inequalities, divisibility and congruences, Diophantine equations, functional equations, AMC- and AIME-style problems, olympiad lemmas, and statements of IMO, Shortlist and APMO problems. **187 have a kernel-checked reference proof**; 81 are statement-only. Each problem carries its text, source, level, topics, and an independent description of what its statement is supposed to mean.

Nothing here is trusted. A reference proof is a library-style tactic script that the kernel checks like any other theorem; whether a formal *statement* says what the words say is a separate question, which is what most of this directory is about.

```python
import itp_paths, bank_lib

bank = bank_lib.load()                          # the first 21 problems: about 5 s       (bank_all.load(): all 268, about 80 s and 1.3 GB)
bank.names()[:3]                                # ['imo1959_p1', 'sqrt2_irrational', 'imo2019_p1']
print(bank.placed['imo1959_p1'].problem.text)   # Prove that the fraction (21n+4)/(14n+3) is irreducible for every natural number n.
print(bank.statement_text('imo1959_p1'))        # forall (x0 : Z), Z.zero <= x0 -> Z.is_gcd (Z.of_N (Npos (XI (XO (XI (XO XH))))) * x0 + ...) (...) Z.one
                                                #   (the formal statement, with the numerals 21, 4, 14, 3 printed in binary)

report = bank.check('sqrt2_irrational', demo=True)       # faithfulness of the statement, sayability as a term, and the reference proof replayed through the action language
report.ok                                                # True
ep = bank.episode('imo1959_p1', regime='prefix')         # an environment Episode: the statement as a goal, with the premises of the regime
```

| module | what |
|---|---|
| `bank.py` | `Problem`, `Bank` (the registry: `@BANK.problem`, `@BANK.statement`, `BANK.declare`), `LoadedBank` (a built world with the index of its problems), the regimes, digests, export |
| `bank_lib.py`, `bank_lib2.py` | the first graded problems, levels 1 to 4, with reference proofs; `bank_lib.load()` builds the first 21 |
| `bank_amc.py`, `bank_amc2.py`, `bank_aime2.py` | AMC- and AIME-style problems, with reference proofs |
| `bank_imo.py`, `bank_imo2.py`, `bank_imo4.py` | IMO, Shortlist and APMO problems and classical exercises, statement-only |
| `bank_imo3.py` | olympiad lemmas and a few contest problems with reference proofs (IMO 1977 P6, IMO 1983 P6 for integer sides, ISL 2005 N3, Vasc's inequality, ...) |
| `bank_classic.py` | the classical results and "find all" exercises of olympiad number theory, statement-only (Fermat's little theorem, Wilson's theorem, the four-squares theorem, ...) |
| `bank_all.py` | the whole bank as one world, in the order the modules register their problems |
| `bank_kit.py`, `bank_spec.py` | proof patterns the problems share; a short way to write the `Spec` of a problem |
| `semantic.py` | the reference semantics: reads a formal statement back into an AST, evaluates it, and compares it with the intended meaning |
| `imo_truth.py` | brute-force evidence for the claims of the statement-only problems |
| `bank_dups.py`, `bank_split.py` | the duplicate screen; the splits by problem family |
| `bank_baseline.py` | how hard each problem is for a blind proof search with a given budget of probes |
| `dataset.py`, `lib_corpus.py`, `subproblems.py` | the dataset: the bank with its proofs as action sequences, the library's proofs, and the theorems the reference proofs contain |

## What a problem is

```python
@BANK.problem('ident_diff_squares', text='For all integers a and b: (a + b)(a - b) = a^2 - b^2.', source='folklore', topics=('algebra', 'identity'), level=1,
              spec=Spec(('Z', 'Z'), (), lambda a, b: (a + b) * (a - b) == a * a - b * b, atom=(lambda a, b: (a + b) * (a - b), lambda a, b: a ** 2 - b ** 2)))
def _(w, b):
    s = _syms(w); Z, Eq = s.Z, s.Eq                    # (_syms: a small helper of the module for the library's names)
    with w.nota.using(Z):
        @thm(w, 'ident_diff_squares', [Z, Z], lambda a, c: Eq((a + c) * (a - c), a ** 2 - c ** 2))
        def _(p): p.intro('a', 'c'); p.ring()
```

* **`name`** names the problem and the theorem; **`text`** is the problem in words (ASCII), what a human checks the formal statement against; **`source`**, `year`, **`topics`**, **`level`** (1 a single arithmetic step, 2 a short proof with one idea, 3 an olympiad lemma, 4 an olympiad problem with a short solution, 5 a hard olympiad problem), **`kind`** (`prove`; `answer`, where the statement contains the answer to "determine all ..."; `exists`).
* **`spec`** is the intended meaning as Python, written independently of the formal statement (below).
* **Provenance.** `origin` is `contest` (a named contest's problem, paraphrased: the texts are not the contests'), `adapted` (a contest problem changed to fit the integer vocabulary; the change is in `notes`), `style` (written for the bank in the manner of a contest, no claim to be a contest problem) or `folklore` (a classical fact or exercise). `verified` is `answer-checked` (the formal claim agrees with a brute-force search and the published answer), `recalled` (the source or answer is from memory and the search covers only part of the claim) or `none`.
* **Tiers.** *Proved*: the bank holds a build function that declares helper lemmas and the theorem (`@BANK.problem`). *Statement-only*: it holds the formal statement and no proof (`@BANK.statement`). A statement-only problem is a task all the same: anything that proves it is checked by the kernel; the bank only does not know how.
* **Regimes.** A problem is attacked in a world made of the library (every global before `imo1959_p1`) and *its own block*, and never sees another problem's theorems or helpers. In the `prefix` regime the problem's own helper lemmas are available; in the `strict` regime only the library and the problem's definitions are, so the ideas must be found. The mask is [`gym_premises.Premises`](../environment/README.md); the world is deterministic, so a problem is identified by its name and the digest of the world below its block.

## Does the statement mean what the problem says?

A proof shows that a statement follows from the library. A wrong constant, a `<` for a `<=`, a missing hypothesis gives a theorem that is true, proved, and not the problem. [`semantic.py`](semantic.py) is the independent check:

1. **Read** the formal statement back into a small AST (a statement outside the fragment it reads is reported, not skipped).
2. **Evaluate** it on Python integers, with the meaning of the library's symbols written down once (`VALUE_PRIMS`, `PROP_PRIMS`: a few dozen lines to audit, and themselves checked against the definitions and the proved theorems of the library by `audit_prims` and `audit_theorems`).
3. **Compare** its parts -- hypotheses, conclusion, and the *numbers* that each relation atom is made of, so that a slip in a constant or the direction of an inequality cannot hide behind a truth table -- with the `Spec` the author wrote, on a box of small values.
4. **Mutate**: statements with one plausible slip of the formaliser injected (a constant off by one or negated, a comparison weakened or strengthened, `+` for `-`, the arguments of a subtraction or divisibility swapped, a relation negated, a hypothesis dropped, `and` for `or`, `forall` for `exists`, a negation dropped) must be told apart from the original by the box, or the mutant must be explained in `skip_mutants` as meaning the same.
5. **Screen**: a statement-only problem must hold at every point of its box, and [`imo_truth.py`](imo_truth.py) re-derives each claim by exhaustive search on a range far beyond the box, written from the problem and not from the formal statement, and compares it with the published answer where there is one.

Quantifiers over `Z` and `N` are evaluated over windows, so these checks are evidence of faithfulness, not a proof of it. `make audit` collects the checks, the provenance and the cautions (where the automatic evidence is weaker) for every statement into one stamped folder with a review sheet for a human second reader.

Two statements that say the same thing are a leak when they land in different splits, so [`bank_dups.py`](bank_dups.py) screens in three layers: identical statements (by digest), the same meaning (the decoded statements evaluated on a canonical grid, comparing truth tables *and* atoms), and similar texts (reviewed by a person).

## The dataset

`python3 src/python/problems/dataset.py OUT_DIR` builds the bank with its reference proofs as sequences of portable actions, the splits, the library corpus (the proofs of the 453 lemmas of the base library as action sequences), the annotations and sub-theorems that the reference proofs contain, a dataset card and a manifest. Every file is deterministic: two builds of the same sources give the same digests. `--verify OUT_DIR` reads the files back and replays every demonstration and every sub-theorem in a fresh engine over a freshly built world (`make dataset-verify`, about 25 minutes and 3 GB). The shipped dataset is [`data/dataset_v2`](../../../data/README.md); its [card](../../../data/dataset_v2/DATASET_CARD.md) lists the composition, the regimes, the splits and the statistics of the proofs.

Splits are by *family* (the two parts of one olympiad problem, a statement in other clothes, an argument with another constant stay together) and are a pure function of the family key, so adding problems never moves an existing one from a split to another: 190 problems in the train split, 20 in dev, 58 in test.

## Adding a problem

1. Write the statement, its **text**, and the **spec** independently (do not copy the formal statement).
2. Build it (`bank_lib.load(only=[name])`), run `bank.check(name, demo=True)` and read the errors: *disagrees with the intended meaning* (the formalisation or the spec is wrong), *go unnoticed by the box* (sharpen the spec with `atoms` or `body`), *not sayable* (the term builder cannot say a statement with `match` or `let`), *does not replay* (the proof uses something the recorder cannot yet translate, or is not self-contained).
3. New vocabulary goes into `VALUE_PRIMS` / `PROP_PRIMS` (and the reader, for a new shape); `audit_prims` and `audit_theorems` confirm the entry against the definitions.
4. Run the bank gates and update the pinned digests on purpose (`S41_UPDATE_GOLDEN=1 python3 tests/problems/s41_bank_tests.py`).

The details, the regimes, the blind-search difficulty of each problem and the list of what the bank still needs are in [`docs/BANK.md`](../../../docs/BANK.md). The bank gates are S41 and S43 (the graded problems), S49 (the contest tier: provenance, faithfulness, splits, duplicates) and S50 (the premises mask and every reference proof replayed and kernel-checked).
