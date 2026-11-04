# examples -- eight programs, in the order to read them

Each example is a self-contained script with a docstring that says what it shows and what it needs. They run from any directory (`python3 examples/01_prove_with_tactics.py`), print what they do, and **end with `EXAMPLE OK`**; gate S46 runs all of them and checks that. The kernel must be built first (`make kernel`). `make examples` runs them in order.

| | example | what it shows | needs | time (two cores) |
|---|---|---|---|---|
| 01 | [`01_prove_with_tactics.py`](01_prove_with_tactics.py) | A function by recursion and a theorem about it by induction; a polynomial identity closed by `ring`; a false identity refused with both normal forms; a **forged proof refused by the kernel**; every global of the world re-checked by the independent checker; what a theorem rests on | kernel | under a second |
| 02 | [`02_ring_linarith.py`](02_ring_linarith.py) | The integers as a quotient of pairs of binary naturals: `ring`, `linarith` (non-negative multipliers found by a simplex) and `nlinarith` (products and squares, with hints), each a certificate that the kernel checks | kernel | 6 seconds |
| 03 | [`03_bank_problem.py`](03_bank_problem.py) | One problem of the bank end to end: the statement, whether the formalisation says what the words say, its recorded proof replayed by the kernel, the verdict of the independent checker | kernel | 90 seconds |
| 04 | [`04_environment_episode.py`](04_environment_episode.py) | One proof attempt as a game: the goal as data, the menu of legal actions, an action that does not apply and changes nothing, a recorded proof played action by action | kernel | 80 seconds |
| 05 | [`05_blind_search.py`](05_blind_search.py) | The built-in proof search (iterative deepening over the menus): the tree it finds for one problem, and how many probes five problems need when the lemmas written for them are hidden | kernel | 90 seconds |
| 06 | [`06_gymnasium_env.py`](06_gymnasium_env.py) | The bank as a Gymnasium environment (`ITPBank-v0`): observations as arrays of token ids, an action as the index of one option, a recorded proof played through it decision by decision | kernel, `gymnasium` | 85 seconds |
| 07 | [`07_read_dataset.py`](07_read_dataset.py) | The dataset on disk (`data/dataset_v2`): what is in each file, how it is split, and the check of its manifest | none | 2 seconds |
| 08 | [`08_hindsight.py`](08_hindsight.py) | Hindsight: an attempt stopped halfway still proved things (a case that closed, a fact a lemma derived); they come back as new problems with proofs, each replayed in a fresh engine and checked by the kernel and by the independent checker | kernel | 80 seconds |

Examples 01 and 02 are the ones to read first; they show the whole path from a statement to a proof checked twice. 03 to 06 and 08 build the world of the problem bank (about 80 seconds and 1.3 GB for the whole bank), so they take about a minute and a half each (the times above were measured on a two-core Linux machine). Examples 03 to 05 take an optional problem name (`python3 examples/03_bank_problem.py imo2019_p1`).

## `repros/`

Small scripts that reproduce findings about the kernel recorded in [`docs/FINDINGS.md`](../docs/FINDINGS.md) (conversion that is incomplete but never unsound, an exponential case in conversion over shared terms, a wrongly instantiated `apply` against a large goal that ends in a spent work budget). Each prints what it shows; see [`repros/README.md`](repros/README.md).
