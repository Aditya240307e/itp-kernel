# src/python/checker -- the second opinion

A type checker for the kernel's fragment, **written separately in Python from the specification** rather than from the C++ source, plus the glue that hands a finished world to it and a tool that reports what a theorem rests on. A proof is believed when both implementations accept it.

| module | what |
|---|---|
| `indep_checker.py` | the independent checker (735 lines): terms, substitution, conversion, the typing rules, the inductive and quotient analyses, the structural-recursion guard |
| `recheck.py` | the glue: read the kernel's exported terms, build an environment, run the checker over it (`recheck_world`, `SecondOpinion`) |
| `audit.py` | `dependencies_of_global`: the axioms and quotient primitives a theorem reaches, like Lean's `#print axioms` |

## Why a second checker

The kernel is about 8,300 lines of C++ with its bindings, and it is tested hard (mutation testing, fuzzing, lockstep comparison with a frozen older kernel; see [`tests/`](../../../tests/README.md) and [`mutants/`](../../../mutants/README.md)). A second implementation turns a silent bug in either one into a visible disagreement, and it is built so that the two are unlikely to be wrong in the same way:

| | C++ kernel (`src/c++/hybrid`) | independent checker |
|---|---|---|
| variables | de Bruijn indices for bound variables, levels for context variables | names, with capture-avoiding substitution |
| conversion | normalisation by evaluation: closures, persistent environments, shared caches, a syntactic pre-check | weak-head normalisation; no closures, no caches |
| universe rule, structural-recursion guard, parameter analysis, `match` typing | in the kernel | re-implemented here |
| quotients | five constants whose types the kernel builds, plus one reduction rule | does **not** trust the kernel's labels: it derives the universe level and the equality family from the declared type of `Quot.sound`, demands that each of the five constants has exactly the specified type (up to alpha-equivalence), and only then admits the one reduction rule |
| language | C++23 | Python |

The two share the *specification* and the *input format* (term trees as printed by the kernel's `dump_raw`), and nothing else. A flaw in the design itself, as opposed to in one implementation of it, would be shared; that is the limit of this kind of evidence.

## Using it

```python
import itp_paths, nat_lib, recheck, audit

w = nat_lib.build_nat()
recheck.recheck_world(w)            # (129, []): every global re-checked from scratch; the list holds (gid, name, message) per problem

# a restricted second opinion on one theorem: only the globals it needs (here 12 of 129)
so = recheck.SecondOpinion(w.k, needed_by=[w['add_comm']])
so.accepts(w['add_comm'], w.global_type('add_comm'))    # (True, 'OK')
so.accepts(w['add_comm'], w.global_type('add_zero'))    # (False, 'type mismatch')

audit.dependencies_of_global(w.k, w.gid['add_comm'])    # (set(), set()): no axioms, no quotient primitives
```

* **`recheck_world(w, fuel=200_000_000)`** takes an `author.World` or a bare kernel and returns `(number of globals, problems)`. The problem list is empty iff the independent checker accepts every global.
* **`SecondOpinion(k, fuel, needed_by=None)`** builds the checker's environment from the kernel as it is *now* (make it after the proofs it will judge). With `needed_by`, a list of closed terms, the environment holds only the globals they need, transitively, which costs seconds where a world holding the proofs of the whole problem bank costs a minute; `accepts` then refuses terms that mention anything else. `accepts(proof, statement)` answers `(ok, message)`; `problems()` re-checks every global in the environment.
* **`audit.dependencies_of_global(k, gid)`** returns `(axiom names, quotient-constant names)`. Definitions are unfolded, both their types and their values, transitively; inductive families and constructors are part of the kernel's checked declaration path and are not reported. An empty result is a theorem of the bare kernel (the natural numbers). The integers are a quotient of pairs of naturals, so their theorems report the five quotient constants; `by_contra` and whatever is built on it report `em`, the one axiom of the libraries.

## What it checks

`Checker.check_definitions()` (what `recheck_world` runs) establishes:

1. the quotient constants are exactly the specified five (above);
2. the environment is **well founded**: no global mentions an unknown one or one that is declared but unfinished, the graph of mentions has no cycle, and the only self-reference is the call of a structurally recursive function to itself;
3. every global the kernel recorded as structurally recursive **passes the checker's own guard**;
4. the type of every global is a type, and every definition's value has its type;
5. every inductive family is **admitted** under the checker's own reading of the stated conditions, from the types of its constructors alone: the family occurs strictly positively in each field (not at all, or in a field that is a product whose domain does not mention it, or as the family applied to arguments that do not mention it) and not at all in a result index (F92), and a field that is not parameter-like lives in a universe no larger than the family's, unless `Type 0` is `Prop`.

The typing rules it implements are the kernel's: `Type k : Type (k+1)` without cumulativity; `Pi` impredicative into `Type 0`; delta, iota, the unfolding of `fix` on a constructor, function eta; proof irrelevance in the `Prop`/`Type` world (decided by inference, never by the shape of a term); the elimination restriction for a `Prop` family that is not a subsingleton; parameters as a prefix of a family's arguments; and `match` typing with index unification, including the rule for the motive of a `match`: it is well typed, its chain of products ends in a sort, and it is applicable, that is, its applications to the indices and the value of the scrutinee, and to the indices and the value of each constructor under the constructor's fields, are well typed (F93). The header of `indep_checker.py` states each rule in a line.

## What it does not do

* It checks terms that are **closed and metavariable-free**: `let` and metavariable nodes raise `Unsupported`, and a term containing one is not accepted. The authoring layer never produces them in a finished proof.
* It checks, it does not elaborate: there is no tactic engine, no unification, no inference of implicit arguments.
* It is bounded by a work budget like the kernel (`fuel`); running out is a refusal, never an acceptance.

## Where it is used

The gates S8 (the kernel and the checker must give the same verdict on the library's proofs and on mutants of them), S10 (generated lemmas), S14 (the `Prop`/`Type` library, every global re-checked) and S47 (the restricted second opinion on hindsight theorems); gate S50 judges a sample of the contest proofs, each in a process of its own; `make soundness` includes a run of the kernel against the checker. Examples [`01`](../../../examples/01_prove_with_tactics.py) and [`02`](../../../examples/02_ring_linarith.py) re-check a whole world, [`03`](../../../examples/03_bank_problem.py) and [`08`](../../../examples/08_hindsight.py) ask for a second opinion on one theorem.
