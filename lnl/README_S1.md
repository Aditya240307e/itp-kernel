# Hybrid-variable kernel: spike S0 + S1 (status 2026-10-05) -- GO

Bound variables are de Bruijn **indices**, context variables (hypotheses) stay **levels**. See the refactor plan for the design
rationale. This directory is a separate build (`itp_env_lnl`); the reference kernel (`itp_env_ref`) is the frozen baseline commit.

## Build / run
    tools/build_reference.sh                      # reference, from commit 015c7a7  -> /home/claude/itp_ref
    cd lnl && ITP_NB_DOMAIN=lnl python3 build_kernel.py
    PYTHONPATH=/home/claude/itp_ref:/home/claude/itp_py:/home/claude/itp_py/lnl python3 lnl/harness.py <N> <seed> [depth]
Two nanobind modules only coexist in one process with `ITP_NB_DOMAIN` (own type registry), `-fvisibility=hidden` and, on GCC,
`-fno-gnu-unique` (otherwise inline variables such as the thread_local `active_context` are merged across modules).

## What is implemented
* `Var{n, bound}` in the existing pool (no new node kind). `mk_var`/`Var::level` removed so the compiler flagged every site.
* `map_children`: ONE binder-aware traversal that is total over node kinds; `lift`, `instantiate`, `abstract_fvars`,
  `replace_fvars`, `resolve_metas`, `requote` are built on it (the old hand-written traversals all missed `Fix`).
* `eval` (one-line change), `quote_at(v, n0, depth)`, `requote` (no re-anchoring, no second depth counter), `normalize`, `conv`.
* `Fix`: `fn_type` closed under the ambient context, `body` under the self binder (old "anchored at self+1" convention gone).
* Declaration helpers (unchecked) and a typing fragment (Type, Pi, Lambda, App, Let, Const, context variables).
* NOT yet: Match/Fix typing, admission, positivity, guard checker (S2); unification, metas, apply/rewrite (S3); cases/induct (S4).

## S1 evidence (differential against the reference, all via the converter in harness.py)
| check | volume | disagreements |
|---|---|---|
| core-op property tests vs independent Python implementations (lift, instantiate, abstract, loose_range, round-trip laws) | 3,000 terms x ~6 laws | 0 |
| closed normalisation | 220,000 terms (depths 4-6) | 0 |
| open normalisation (1-3 context vars, stuck Match in 25-30% of normal forms) | 220,000 terms | 0 |
| conversion, oracle = reference `refl t1 : Eq t1 t2` (eta included) | 440,000 pairs | 0 |
| typing (Match-free) closed / open | 220,000 / 220,000 | 0 / 0 |
| ill-typed terms rejected by both | ~73,000 | 0 |
| proof corpus: real nested induct/cases proofs (38 Fix nodes, fn_types mention outer binders), closed / numerals / free vars | 39 normalisations | 0 |
| invariant: every normalisation output has no loose bound variable | all of the above | 0 violations |
| library: all 14 globals loaded unchecked, every global type equal after conversion | 14 | 0 |

## Performance (single thread, same terms both kernels)
* normalisation: hybrid 0.74-0.84x of reference (faster).
* typing, random terms depth 4-9: 0.93-1.19x.
* typing, adversarial chain of n nested binders (`lambda x1..xn. x1`):
  - first version (one `instantiate` per binder): QUADRATIC, 1.6x at n=10, 3.6x at 50, 8.3x at 200, 16x at 400.
  - after telescope opening (peel the whole Lambda/Pi chain, ONE instantiate, type the body once, close with one abstract per
    domain): 0.79x at n=10, 0.75x at 100, 0.72x at 400, 0.60x at 800 -- linear, faster than the reference.
  The differential suite was re-run after this change (90,000 terms + proof corpus + 5,000 property cases): 0 disagreements.
  S2 must keep this discipline for the full typing core (Match branches, Fix bodies, constructor telescopes).

## Harness limitations (so the evidence is not over-read)
Reference has no `mk_fix`, so Fix is covered only through extracted proofs, not synthetic terms. Generated types are
non-dependent (Nat, Bool, arrows); dependent typing is exercised only by library types. Typing excludes Match/Fix.
The converter was validated by comparing every library global's type, and semantically by the differential runs.


---
# S2: the typing core (status 2026-10-05) -- gate G2 PASSED

## What is ported
`infer`/`check` for every node kind (Match and Fix included); `check_match_branches` with index forcing as a simultaneous
`replace_fvars` (forced values may carry binders -- the case that broke the old syntactic substitution, F5); checked inductive
declaration (strict positivity + admission: typed fields/indices, universe rule); checked recursive-function declaration;
`declare_checked`. Lambda/Pi chains are opened as telescopes (one instantiate per chain), constructor branch types are closed with
one abstract per field. The legacy structural-recursion analyses are reused through an all-levels bridge (`to_levels`).

## Deliberate differences from the reference
1. **`infer(Fix)` runs the structural-recursion guard.** The reference only checked it inside the `induct` tactic, so
   `whole_term_check` trusted the tactic for every local Fix. The hybrid kernel does not.
2. `Let` is typed with the value substituted (zeta); the reference typed the body with an opaque variable.
3. `unify` is pure conversion for now; motive *inference* for a Match with an unresolved motive, universe polymorphism, mutual
   recursion guards and the derived Unit/Empty/same_ctor types are not ported yet (S3/S4).

## Evidence (all against the frozen reference)
| check | volume | result |
|---|---|---|
| library re-declared through the CHECKED path (Nat, Bool, Eq, Le, add, cong, lt, transport, le_trans) | 9 events | all accepted, 16/16 global types equal |
| inductive admission corpus (paradox, large fields, bad indices, positivity, impredicative/param-like/Vec accepted) | 12 | ref == new == expected, reasons match |
| structural-recursion guard corpus | 6 + 4 hand-built Fix | ref == new == expected |
| Match typing, random typed terms / ill-typed matches (too few, too many, wrong type, non-inductive scrutinee, wrong lambda domain) | 40,000 / 20,000 | 0 disagreements |
| dependent typing: `refl a : Eq a b` needing delta/iota/beta in the type | 40,003 | 0 disagreements |
| proof corpus: real nested induct/cases proofs (31 Fix nodes) accepted at their statements | 12 | 12 accepted |
| the same proofs checked against FALSE statements | 12 | 12 rejected |
| main differential (normalise/open/conv/typing/reject) with the full typing core | 60,000 | 0 disagreements |
| typing speed vs reference | | 0.98-1.03x |

## Mutation testing of the typing core (each mutant re-breaks one rule in an isolated build; the suite must fail)
M1 guard disabled, M2 index forcing no-op, M3 lambda domain not unified, M4 Pi-telescope closing off by one, M5 application argument
unchecked, M6 admission universe rule off, M7 strict positivity off, M8 Match branch count unchecked: **8/8 caught.**
Two lessons: (a) M3 first looked like a survivor because my detector missed one output format -- the suite now exits non-zero on
any failure instead of being scraped; (b) M8 exposed a real gap in the ill-typed generator (it never produced an over-long match);
fixed. Run: `python3 lnl/mutate.py [M1 M2 ...]`.
