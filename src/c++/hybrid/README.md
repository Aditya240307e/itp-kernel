# src/c++/hybrid -- the kernel

The trusted core of ITP: a dependently typed kernel in C++23 that admits declarations, type-checks terms, decides definitional equality and runs the goal-directed tactic engine. Two files:

| file | lines | what |
|---|---|---|
| `itp_kernel.hpp` | 7,546 | the kernel: terms, evaluation, conversion, type checker, admission of inductives, recursion and quotients, the tactic engine |
| `itp_bindings.cpp` | 788 | the [nanobind](https://github.com/wjakob/nanobind) module `itp_env_lnl`: the Python class `Kernel` and its helpers |

Build: `make kernel` (or `python3 src/python/build_kernel.py hybrid`) from the root of the repository. It needs a compiler that accepts `-std=c++23` with `std::expected` (g++ 13 or newer; Xcode 16 or newer on macOS) and needs nothing but `nanobind` and the Python headers. About 40 s on two cores. The output is `build/itp_env_lnl*.so`.

## What is trusted, and what is not

What must be believed for "this proof is correct" is this directory, the C++ compiler and the machine. The Python layer (`src/python`) builds terms and calls the kernel; it can offer a wrong term, never make the kernel accept one:

* A theorem enters the global environment only through a **checked** entry point: `declare_checked(type, value, name)` type-checks `value` against `type`; `finish_inductive` runs the admission checks (below); `finish_recursive_function_checked` runs the structural-decrease guard; `declare_quot` builds the five quotient types itself and checks the equality family it is given. The `*_unchecked` entry points exist for axioms and for tests and are what `audit` looks for.
* The tactic engine is in the same header, but its output is a term like any other: the proof state is extracted and re-checked by `declare_checked`.
* Every call from Python validates the node ids it receives (a malformed or foreign id is an error, not a memory access), runs under a work budget and a stack guard, and turns every C++ exception into a Python exception.

The soundness evidence is in [`tests/`](../../../tests/README.md) and [`mutants/`](../../../mutants/README.md): it is *tested*, not proved.

## The term language

Eleven node kinds, each a plain struct of indices and `NodeId`s (a type tag and a pool index):

| node | meaning |
|---|---|
| `Var` | a variable: bound (de Bruijn **index**, 0 = innermost) or free (a context **level**) |
| `App`, `Lambda`, `Pi` | application, abstraction (with its domain type), dependent function type |
| `Type k` | the universe at level `k`; `Type k : Type (k+1)`, no cumulativity |
| `Const` | a global: an axiom, a definition, an inductive type or a constructor |
| `Match` | case analysis on an inductive value: scrutinee, motive, one branch per constructor (a curried function of the constructor's fields) |
| `Fix` | a local fixpoint with a declared decreasing argument; the structural-decrease guard runs when it is typed |
| `Let` | `let x : T := v in b` (typed with zeta) |
| `Meta` | a metavariable of the tactic engine; an unsolved one is an opaque stuck term, a solved one unfolds |
| `TypeUniv` | the placeholder for a universe parameter in a declaration's template (instantiated before use) |

There are no separate constructor, recursor or projection nodes. Constructors are constants; an eliminator is a `Match`, a recursive function is a `Fix` or a global definition whose recursion the guard has accepted.

## Variables: indices below, levels in the context

Bound variables are de Bruijn indices, so a closed term has exactly one representation and equal terms hash equal. Hypotheses of the context (what a proof state's goals talk about) are **levels**: stable names counted from the outside, so extending the context does not renumber anything. `instantiate`, `abstract_fvars`, `lift`, `replace_fvars` and `resolve_metas` are all built on one binder-aware traversal, `map_children`, which is total over the node kinds. This hybrid is what the previous kernel (absolute levels everywhere, kept frozen in [`../reference`](../reference/README.md)) was replaced by.

## Hash-consing and memory

Every node is interned in a per-kind pool with an arena allocator: structurally equal terms are the same `NodeId`, so equality is an integer comparison and proofs that share subterms stay small as DAGs. Each node carries precomputed facts (the range of loose bound variables, among others). A `checkpoint` / `rewind` pair discards everything allocated since the checkpoint, which is how the environment probes an action and undoes it; `compact_since` reclaims the scaffolding of a successful probe while keeping its subgoals. The traversals of shared terms are memoised per DAG node, never per path.

## Evaluation and conversion

* **Normalisation by evaluation.** `eval` maps terms to semantic values (closures over persistent, structurally shared environments, constructor applications, stuck terms), `quote_at` maps values back; `normalize` is `quote(eval(t))`. Values are O(1) to copy through shared immutable sequences. `Let` is a beta-redex (zeta).
* **Definitional equality.** Beta, delta (a constant with a value unfolds at the head of a term), iota (`match` on a constructor application), `Fix` unfolding when its decreasing argument is a constructor application, function eta, the quotient computation rule, and -- in the `Prop`/`Type` world -- proof irrelevance.
* **Speed without losing soundness.** A quick conversion check on *syntax* runs in front of the evaluator (`SynConv`, bounded in steps, depth and unfoldings and falling back to the evaluator when it gives up); sharing tables (`ConvMemo`, `UnifyMemo`, `QuoteMemo`) remember comparisons of shared structure; a failed comparison is decided by what it found rather than by normalising both sides. With `ITP_VERIFY_MEMO=1` (or `set_option('verify_memo', True)`) every table hit and every quick "yes" is re-derived with the tables bypassed and a disagreement throws. The kernel gates are run both ways.
* **Bounds fail closed.** Nesting of conversions is limited to 256 and of proof-irrelevance checks to 8; beyond the bound the answer is "not equal" / "not irrelevant", never "equal".

## Type checking and unification

`infer` / `check` are a bidirectional elaborator over `ElabCtx` (a typing scope that parallels the evaluation environment) and `ElabState` (metavariables, deferred constraints). Unification is first-order with **Miller patterns** for metavariables: every metavariable the tactic engine creates is applied to all the context variables in scope, so its solutions are closed lambdas. The motive of a `match` is inferred when the scrutinee's family is known; for indexed families the checker forces the index equalities each constructor determines, and parameters are distinguished from indices (kernel option `parameters`). Failed unifications roll back every assignment they made (`TacticTxn`).

## Declarations

* **Inductive families** (`reserve_inductive_type`, `finish_inductive`): one non-mutual family at a time, with parameters and indices. Admission checks the family's declared type, every constructor argument type and every result index, applies the **universe rule** (a constructor of an impredicative family may not take a large field, which would admit Girard's paradox), and **strict positivity**: the family may occur in a constructor argument only directly or in the codomain of an arrow, never in a domain; an occurrence nested inside another family's application is rejected.
* **Structural recursion** (`reserve_recursive_function`, `finish_recursive_function_checked`): accepted when every self-call passes, at a fixed argument position, a value obtained by destructuring something already known to be structurally smaller. There is no well-founded recursion; strong induction is built from a bounded predicate (`Below P n`) in the libraries.
* **Quotients** (`declare_quot`): not a node kind. Five ordinary constants -- `Quot`, `Quot.mk`, `Quot.lift`, `Quot.ind`, `Quot.sound` -- whose types the kernel builds itself (so nobody can hand it different ones), and one reduction rule, `Quot.lift A r B f h (Quot.mk A r a) ~> f a`. It requires the `Prop`/`Type` world and an equality family of exactly the expected shape, and is declared at most once. Function extensionality is then a *theorem* (library `quot_lib`).
* **Axioms** (`declare_unchecked` with no value) are ordinary constants with no definition. They are reported by `audit`; the libraries declare one (excluded middle, in the number-theory stage), and the gates also exercise a four-axiom classical layer (excluded middle, propositional extensionality, dependent function extensionality, choice).

## The `Prop` / `Type` world

With `set_option('prop_elim', True)`, `Type 0` is `Prop`: impredicative (a `Pi` whose codomain is a proposition is a proposition), data lives in `Type 1` and above, proofs of one proposition are **definitionally equal** (proof irrelevance), and an inductive family in `Prop` that is not a syntactic subsingleton eliminates only into `Prop`. Proof irrelevance is decided by the type checker itself -- quote the value, infer its type, demand that the type is a proposition, do the same for the other side, unify -- behind a cheap syntactic prefilter that only ever says "certainly not a proof". The authoring layer's `World` turns both this option and `parameters` on.

## The tactic engine

`start_proof(goal_type)` returns a `ProofState`: a list of goals, each an unsolved metavariable with its own hypothesis telescope. The primitives are `intro`, `apply` (candidates from a discrimination tree, then real unification), `cases`, `induct` (with a `generalizing` list and equations for concrete indices), `exact`, `rewrite` (by an equation and a transport lemma), `change` and `clear`; `extract()` returns the finished proof term. Each primitive either does its whole job or leaves the state exactly as it was. The conveniences (named hypotheses, rewriting by pattern, `ring`, `linarith`) are in the Python layer and end in calls to these.

## Work budget and stack guard

Recursion on the depth of a term and work that is exponential in sharing are the two ways a checker hangs or dies. Every function that can recurse starts with `ITP_STACK_CHECK()`, which compares the stack pointer with the thread's floor (a violation throws `StackExhausted`, a `RuntimeError` in Python) and spends **one unit of a work budget** (a violation throws `FuelExhausted`, a subclass of `ITPError`). The budget is per call into the kernel (`set_fuel`; the authoring layer's default is 2 x 10^9 steps, `ITP_FUEL` overrides it), counts steps rather than seconds, so the same input spends the same on every machine and run, and once spent stays spent for the rest of the call: a handler that swallows the exception cannot carry on. A refusal for lack of budget is therefore distinguishable from a refusal on logical grounds, and neither is ever an acceptance.

## The Python API (`itp_env_lnl.Kernel`)

| group | methods |
|---|---|
| build terms | `mk_type`, `mk_bvar`, `mk_fvar`, `mk_app`, `mk_lambda`, `mk_pi`, `mk_let`, `mk_const`, `mk_match`, `mk_fix`, `mk_meta`; `lift`, `instantiate`, `abstract_fvars`, `replace_fvars`, `close_telescope` |
| declare | `declare_checked`, `declare_unchecked`, `reserve_inductive_type` / `finish_inductive`, `reserve_recursive_function` / `finish_recursive_function_checked`, `declare_quot` |
| check and compare open terms | `infer_open`, `check_open`, `check_partial`, `conv_open`, `normalize_open`, `simp_normalize`, `unfold_head`, `unify_open`, `fresh_meta`, `resolve_metas` |
| prove | `start_proof` -> `ProofState` (`intro`, `apply`, `cases`, `induct`, `exact`, `rewrite`, `change`, `clear`, `extract`, `goal_target`, `goal_hyp_type`, ...) |
| search support | `checkpoint`, `rewind`, `fork_single_goal`, `materialize_from_node`, `finish_probe`, `index_library`, `apply_candidates`, `library_candidates` |
| read back | `num_globals`, `list_globals`, `global_info`, `view`, `dump_raw`, `param_slots`, `recursive_info`, `quot_role`, `is_inductive` |
| budget and options | `set_fuel`, `get_fuel`, `fuel_used`, `fuel_peak`, `set_option` / `get_option` (`parameters`, `prop_elim`, `verify_memo`, `syn_conv`), `set_syn_limits`, `set_conv_nesting_limit` |
| statistics | `memo_stats`, `conv_memo_hits`, `unify_memo_hits`, `quote_memo_hits`, `syn_stats`, `extract_stats`, `pool_sizes` |

Errors: `itp_env_lnl.ITPError` (a failed or rejected action), `FuelExhausted` (its subclass: the budget ran out), and `RuntimeError` for a stack guard that fired. Every `Kernel` owns its own context; nodes from one kernel are not valid in another (the authoring layer's `Term` carries its world for that reason).

## Reading the header

Search for the banner comment of each part, in file order: *Stack safety*, *Work budget*, *Term representation*, *Hashing*, *Arena allocation*, *Hash consing*, *Execution context and global environment*, *Term construction and structural dispatch*, *Normalization by evaluation*, *Env* and *LocalCtx*, *Elaborator: ElabCtx, Miller-pattern unification, infer/check*, *Match motive inference*, *Definitional proof irrelevance*, *Quick conversion check on syntax*, *Compacting commit*, *Phase 2: ProofState*, *Apply*, *Cases*, *Inductive types*, *Structural recursion checking*, *Quotients*, and the entry points at the end. [`docs/KERNEL_DESIGN.md`](../../../docs/KERNEL_DESIGN.md) is the chronological design log: what was decided at each step and why, with the experiments behind it.

## Two kernels in one process

The frozen reference kernel (`../reference`) and this one are built as different modules (`itp_env_ref`, `itp_env_lnl`) that can be imported side by side, which is what the lockstep gates do. Two nanobind modules only coexist with their own type registries (`NB_DOMAIN`), hidden symbol visibility, and on GCC `-fno-gnu-unique` (otherwise inline variables such as the thread-local active context are merged across modules); `build_kernel.py` applies all three.
