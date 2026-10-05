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


---
# S3: metavariables, unification, proof states, intro / apply / rewrite / exact (status 2026-10-05) -- gate G3a PASSED

## What is ported
`unify_values` (verbatim: it works on Values), `resolve_solved_meta_head`, `is_miller_pattern`, `drain_pending`, a new `solve_pattern`
(spine variables are renamed to a fresh telescope and closed with one abstract), `start_proof`, `intro`, `exact`, `apply`,
`rewrite_eq`, Match-motive inference in `check` (indices and scrutinee abstracted with `replace_terms`), and the Python proof-state API
(`start_proof`, `intro`, `apply`, `exact`, `rewrite`, goal accessors, `extract`).
Meta occurrences are `?m x0 .. x_{n-1}` (applied to every context variable); solutions are closed lambdas over the goal context.

## Deliberate differences / preserved quirks
1. `intro` on a literal Pi stays syntactic (`instantiate1` on the codomain, goals keep their folded shape). The reference took
   `pi->dom_id/cod_id` even when the Pi only appeared after unfolding a definition, i.e. with ids belonging to another
   environment; the hybrid kernel routes that case through NbE.
2. `solve_pattern` now CHECKS that the spine is in dependency order (the reference silently assumed it).
3. PRESERVED quirk: `?m =?= ?m` is rejected (an unapplied meta is a Miller pattern, so the occurs check fires on itself). Reflexive
   unification should succeed; this will be fixed deliberately later, not silently during a behaviour-preserving port.
4. NOT YET PORTED: the library index (`index_library`, `apply_candidates`, the discrimination tree) -- moved to the start of S5 together
   with the search integration; `cases`, `induct` (S4).

## Evidence (all against the frozen reference)
| check | volume | result |
|---|---|---|
| scripted proofs in lockstep (refl, symm, trans, cong_succ, le_trans, le_refl, le_succ, rewrite2, add_zero_l) | 9 | states identical after every step; all proofs verify in both; identical normal forms |
| higher-order script (`apply cong`: pending constraints) | 1 | states identical, open in both |
| random tactic scripts (intro / apply const or hypothesis / rewrite) | 16,000 scripts, ~157,000 steps | 0 divergences (58k steps succeeded, 99k failed identically), 828 completed proofs verified identically |
| unifier unit tests (occurs check, Miller spine, scope escape, constructor clash, non-linear deferral, decomposition) | 9 | all as expected |
| telescope closers vs an independent right-fold on random DEPENDENT telescopes | 4,000 | 0 mismatches |
| S2 gate and the main differential re-run after S3 | | still pass |

## Mutation testing of the S3 rules
M9 occurs check off, M10 intro does not instantiate, M11 rewrite substitutes x for y, M12 apply skips drain_pending, M13 telescope closer
off by one, M14 meta applied to all but the last context variable.
**5/6 caught.** Two survivors needed work: M9 (nothing produced `?m =?= S ?m`) -> direct unifier unit tests; M13 (wrong lambda domains are
invisible in extracted proofs because contextual lambdas are beta-reduced away) -> property test of the closer on dependent telescopes.
**M12 SURVIVES and is documented rather than hidden:** `drain_pending` can only make progress when a deferred constraint becomes a Miller
pattern after a sibling meta is solved, but every meta `apply` creates is applied to ALL context variables, so the re-unified spine repeats
a variable. An axiom `Q : forall f x, f x = x` built to force this still stays open in both kernels. The path is effectively dead at the
tactic level; its unifier-level behaviour is unverified.


---
# S4: cases and induct, plus the hardening the oracle forced (status 2026-10-05) -- gates G3b and the self-consistency oracle PASS

## What is ported
`cases` and `induct` as a clean rewrite: ONE construction (the generalised goal G over context variables n..n+arity; the motive is its lambda closure and the Fix
type its Pi closure) instead of the reference's "agent" and "runtime" numberings; reverted hypotheses are Pi binders of G peeled with the next context variable;
the derived types `__Unit`, `__Empty`, `__same_ctor_*` for refutation exploitation (`same_ctor` rebuilt for the new representation).
Python API: `ProofState.cases(hyp, target_meta_id)`, `.induct(hyp, target_meta_id)`, plus `.exact`, `.num_pending`.

## Deliberate differences from the reference (each has a test that fails without it)
| id | difference | why |
|---|---|---|
| D1 | every goal meta gets the closed Pi-over-context type it is applied at | makes every partial proof type-checkable; no observable effect on states |
| D2 | `induct` requires a recursive constructor field | the reference never accepted induct on Eq (1,869 rejections, 0 acceptances in ~8,300 attempts; Nat and Le agreed 6,475/6,475); its rejection was an accident of a marker/self-level collision |
| D3 | no `has_internal_binder_ref` guards | embedded binders cannot collide with context variables any more |
| D4 (F11) | constructor field types are mapped to their ACTUAL context levels | `induct` interleaves an IH after each recursive field; the reference assumed consecutive levels |
| D5 (F9) | the scrutinee's constructor application uses the FORCED field values (le_S a m h', le_n n, refl Nat x) -- in the kernel's Match typing AND in both tactics | the reference applied the constructor to the field binders: ill-typed against the forced field types |
| D6 (F13) | `rewrite` instantiates solved metas first and type-checks its motive | the reference rewrote inside meta spines (ill-formed goals) and accepted ill-typed motives |
| D7 | conversion of stuck Matches has a nesting bound (256) that FAILS CLOSED | found by the oracle: a native segfault (unbounded recursion between conv_value and branches_convertible on two stuck recursive definitions) |
| D8 (F14) | the surviving sibling of an exploited refutation is wrapped well-typed | the reference duplicated the field lambdas |

## Findings in the reference (F9-F14)
* F9 forced fields applied as dummy binders (see D5). Before the fix 84 of 1,427 induct partial proofs with terms IDENTICAL to the reference's failed to type-check.
* F10 sequential marker substitution collides when two reverted hypotheses are re-introduced (`(v7 v7)` where `(v7 v6)` is right). Seen in meta spines and in
  hypotheses (golden test `transitive_revert`). Every F10 event is validated: the reference's state is checked to be genuinely ill-formed when it is meta-free.
* F11 field after a recursive field (see D4); direct test with a custom family.
* F13 rewrite (see D6). F13b is verified directly: the reference accepts `a := b` in `Eq (Le a a) p p` and leaves an ill-formed goal. F13a (apply symm; rewrite h) leaves
  an ill-formed INTERMEDIATE goal in the pre-fix port, but the reference's final proof there verifies, so it is not a visible failure.
* F14 exploited sibling (see D8): `exploit_le` verifies in the hybrid kernel and FAILS verification in the reference.
* Refutation exploitation never fires for Eq (a repeated index is not compared against its forced value): the known F2 limitation, unchanged.
* Quirk preserved: any Nat-typed hypothesis counts as dependent on Eq's first index (`Nat`), so it is reverted and exploitation is disabled.

## Evidence (all on the final tree)
| check | volume | result |
|---|---|---|
| S2 gate | 4,000 | PASS |
| main differential | 60,000 terms, seed 77 | 0 disagreements (norm, open norm, 120k conversions, typing, open typing, 20k rejections, proof corpus) |
| S3 lockstep | 2,000 scripts, 19,629 steps | 0 divergences; 90 completed proofs verified identically |
| S4 lockstep | 5 seeds x 3,000 scripts (~134k steps) | 0 hard divergences |
| S4 completed proofs | 184 | 174 identical; 10 where ONLY the reference's proof fails verification (never the reverse) |
| S4 explained categories | per seed ~32-41 F10, 1-4 F13, 0-2 relaxed-guard | F10 validated against the reference's ill-formed state (seed 5: 32 validated, 1 meta-bearing so unvalidated); relaxed-guard events: hybrid state consistent in every case |
| oracle, strict fragment (variable indices only) | 4 seeds x 6,000 scripts | 0 failures of any kind; 92 completed proofs, all verify in the kernel |
| oracle, full distribution | seeds 1,2 x 6,000 | 0 unattributed failures; ~190 per run attributed to F12 |
| scripted: add_zero_r, zero_le, symm_cases, le_succ_ind, exploit_le | 5 complete Fix-based proofs | verify in the hybrid kernel; the first four also in the reference with identical normal forms |
| mutants M15-M30 | 16, against a baseline the driver now checks is green | all caught (M19-M21 re-validated after the baseline was fixed; M20 needed the exploit scripts; M22-M26 are F9 regressions; M27/M28 F13; M29 the segfault; M30 the induct rule) |

The oracle: after every successful tactic, in states with no unsolved meta behind a goal and no pending constraint, (a) the partial proof type-checks against the statement
and (b) every goal's hypotheses and target are well-formed types. It runs on the hybrid kernel alone, so reference divergences cannot truncate its coverage.

## Contracts and known limitations (read these)
1. **F12, NOT fixed: `induct` on a hypothesis with a concrete or repeated index** (e.g. `IH : Le Zero x`). The runtime Fix is generic over its indices but the user-facing branch goals
   are specialised, so `?sub` is applied to arguments whose types do not match its binder types. Completed proofs are valid only when the proof does not use the specialised
   hypotheses. I decided NOT to reject these: the kernel does not distinguish parameters from indices, so `induct` on `l : List Nat` has a "concrete index" `Nat` too (my reading of the
   code, not tested). The proper fix is a parameter/index distinction plus index generalisation with equations; it changes the branch goals, so it belongs with F2b.
   The oracle attributes later failures to F12 coarsely ("any earlier concrete-index induct in the script"); the strict-fragment runs are the real evidence.
2. A failed tactic may leave side effects (solved metas, queued constraints). Same as the reference; the search already works on fresh forks (hypergraph.py header). Not made atomic.
3. The lockstep no longer compares partial-proof TERMS with the reference (F9 changes them); the evidence is states + the oracle + completed proofs.
4. The conversion bound (256) is arbitrary and only ever rejects: a legitimate comparison nested deeper would be refused.
5. M12 (drain_pending) still survives, see the S3 section. M1-M14 re-runs: see the next section if present.


## S4 addendum (2026-10-05, later): deferred-constraint context, full mutant inventory, soak test
* **D9 / F15 (reference algorithm, fixed): a pending unification constraint is re-unified in the context it was deferred IN.** `unify_values` quotes a deferred constraint at the
  current binder depth, but `drain_pending` re-evaluated it in the OUTER context: a constraint met under a binder mentioned that binder's variable and the process ABORTED
  (`Env::operator[] OUT OF BOUNDS: level=1 count=1`). Constraints are also shared across the whole proof attempt, so one deferred in an earlier goal was re-unified in whichever goal
  context the next `apply` happened to use. Now `PendingConstraint` records its context. Regression case in the unifier tests (`unify_drain_open`); mutant M31 reintroduces it and is
  caught. Reachability from the tactics is rare (it needs a non-pattern meta constraint under a binder), which is why no earlier test hit it; the reference has the identical logic.
* **Mutant inventory on the final tree: 30 of 31 caught** (M1-M11, M13-M31), each against a baseline the driver verifies is green first. The one survivor is M12 (`apply` skipping
  `drain_pending`), documented in the S3 section: at the tactic level a deferred constraint essentially cannot become a Miller pattern. The M4 mutant was re-pinned to the Pi
  closer after S3 introduced a second, textually identical closer.
* **Soak:** 96,000 random scripts over 16 seeds (alternating variable-index-only and the full mix), fresh kernel per seed, one CPU: no crash, no hang, 0 unattributed oracle
  failures, 1,152 completed proofs all verified by the kernel, memory flat at ~630 MB. A first attempt with six parallel processes was OOM-killed: the hash-consing tables grow without
  bound inside one kernel (by design; the search reclaims with rewind), so long runs must recreate the kernel.
* The statement mix of the oracle changed when `exploit_eq` and `exploit_le` were added (all their hypotheses have concrete indices), so the full-distribution runs now attribute
  ~1,300 steps per 6,000 scripts to F12 instead of ~180. The variable-index runs are the strict evidence and are unaffected.
* Full-size gates on the final build: S2 (4,000), main differential (40,000 terms), S3 (2,000 scripts, 19,629 steps, 90 completed proofs verified identically), S4 (3,000 scripts,
  26,847 steps, 0 divergences), oracle gate: all PASS.
