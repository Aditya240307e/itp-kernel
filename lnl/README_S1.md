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


---
# S5 part 1: the library index (status 2026-10-05) -- gate PASS
Ported `index_library`, `apply_candidates`, the discrimination tree and its keys (Python: `Kernel.index_library()`, `Kernel.apply_candidates(ps, library, target_meta_id=None)`, class `Library`).
The only non-mechanical point: the reference keyed a variable-headed conclusion on the variable's absolute LEVEL. A bound variable of a lemma type had level = its binder's position, so with
de Bruijn indices the same key is `base + depth - 1 - index` (bound variable `depth` binders below `base` context variables: base 0 for a lemma type, the context size for a goal, the
hypothesis's own position for a hypothesis); context variables key on their own level. Note the reference's key coincidences between bound and context variables are preserved on purpose
(e.g. a lemma whose conclusion head is its second binder is offered for goals headed by context variable 1); they are an accident of absolute numbering, not a design, and are now
reproduced rather than "fixed", so that the search sees the same candidates.

Evidence: `apply_candidates` compared for EVERY goal after EVERY step of random lockstep scripts and at the initial state, entries and order. 5 seeds, 1,500-3,000 scripts each, ~145,000 goal
states, 0 mismatches. Coverage (per 3,000 scripts): ~19,000 non-empty lists, ~6,700 with hypothesis candidates, ~8,900 Pi-targets, ~280 distinct lists, ~1,300 bound-variable-headed and
~3,900 context-variable-headed targets (new statements with predicate variables: `pred_transport`, `pred_hyp`, `pred_two`, `pred_pi_goal`, `pi_bound_head`, `pi_bound_pair`, `pred_pi_hyp`).
Mutants M32-M36 (bound-head base dropped, per-hypothesis base dropped, argument-headed lemmas not generic, generic lemmas never offered, hypothesis offered at the wrong level): see below.
Not yet ported (S5 part 2): checkpoint/rewind/compact (the search's probing), and the search-level fingerprint itself.


## S5 part 2a: the search's probing protocol (2026-10-05) -- gate PASS
Ported/bound: `Kernel.checkpoint()`, `Kernel.rewind(cp)`, `Kernel.fork_single_goal(ps, meta_id)`, `Kernel.materialize_from_node(target, hyp_types, hyp_names)` (the goal meta gets the closed Pi type, D1),
class `Checkpoint`. The arena rollback itself already lived in ExecutionContext; the work was the Python layer and the hazards specific to the new representation:
* HAZARD, checked and tested: the S1 core operations memoise `loose_range` per node id, and `rewind` REUSES node indices for different nodes. `rewind` clears the memo
  (test: a node nobody built before, memoised with loose range 4243, then a different new node reusing its index must report 0; mutant M37 removes the clearing and is caught).
* rewind restores the global count, drops derived globals (`__Unit`, `__tt`, `__Empty`, `__same_ctor_*`) and their cache entries, and rolls the metas back (M39, M40 caught);
  replaying the same script after a rewind is identical, in both kernels (`exploit_le`: 4 derived globals created, dropped, recreated identically).
* the search's protocol (checkpoint, fork the goal, apply, read off, rewind) in lockstep inside random scripts: 2 seeds x 1,200 scripts, ~14,300 steps, ~57,000 probes
  (24,600 succeeded in both kernels): verdicts and resulting states agree (exact for intro/apply, variable-erased shape for cases/induct/rewrite), probing never disturbed the live state or
  left the live goal meta solved, and `materialize_from_node` reproduced the live goal in both kernels (14,316 checks).
Mutants M32-M36 (library index) and M37, M39, M40 (rewind): all caught. Inventory: 38 of the 39 defined mutants caught (the survivor is M12).

## F16 (reference bug, found by the S5 tests): `apply` misbinds the binders of a Pi-typed hypothesis
The reference REJECTS `apply h` for EVERY Pi-typed hypothesis I tried, including `h : forall n, Le n n |- Le 0 0` (and with 0, 1, 2 dummy hypotheses before it, and a first-order
`h : forall n, P n |- P 0`). The hybrid kernel closes the goal each time and the completed proofs verify. The cause has the signature of the F3/F5/F10 class (a binder referenced by absolute level, peeled in an environment where
that level now belongs to a context variable); I have NOT read the reference's `apply`/`infer(Var)` to pin the exact line. Curriculum proofs are unaffected because they apply an explicitly instantiated IH term, not the bare hypothesis.
`h : forall Q, Q 0 |- P 0` additionally needs higher-order unification: the hybrid kernel defers `?m P h 0 =?= P 0` as a pending constraint (cannot complete, but not wrong); the reference errors.
CONSEQUENCE FOR THE FINGERPRINT: any search step that applies a bare Pi-typed hypothesis behaves differently (the hybrid kernel succeeds where the reference fails). The behavioural fingerprint can therefore equal the reference's only if the
search never tries those applications; first MEASURE how often it does (instrument candidate generation on the baseline runs); if it does, compare against a reference with F16 patched, or explain the deltas.

Test-discipline fix: the S5 candidate test used to END a script silently on any lockstep divergence, which hid 112 F16 divergences per 3,000 scripts. Divergences now must fall in an explained, validated category
(F10, F13, relaxed guard, F16) or they are mismatches. Candidate lists are identical wherever they can be compared (5 seeds, ~145,000 goal states).


---
# S6: logical vocabulary, scope validation, F6 groundwork (2026-10-06)

## Fixed / added
* **`?m =?= ?m` now succeeds** (the S3 section's preserved quirk is gone). Mutant M41.
* **F6 (strong induction groundwork).** The Pi-headed guard is removed from `cases` and `induct` (with de Bruijn indices there is nothing for it to protect) and both take an explicit
  `generalizing=[...]` list. Probes, all verified by the kernel: `Pi n acc. add (S n) acc = add n (S acc)` by `intro n; induct n` on the Pi-headed goal, and by `induct n generalizing acc`; for `cases`,
  `generalizing` matters for hypotheses BEFORE the scrutinee (everything after it is already reverted): hypothesis counts [2,3] vs [3,4]. The reference rejects the Pi-headed `induct`.
  Mutants M47, M48. NOT done: the bounded-predicate strong-induction probe (`Below P n`).
* **F7 (logical vocabulary).** `True`, `False`, `And`, `Or`, `Ex` (inductive families) and `Not`, `Iff` (definitions) declared in BOTH kernels through each one's own checked admission.
  Tactic proofs verified by the hybrid kernel: `true_intro`, `false_elim`, `and_comm`, `or_comm`, `not_not_intro`, `iff_refl`, `ex_elim`. Six hand-made wrong proofs are rejected by both kernels.
  Differential on MUTATED valid proofs (only well-scoped mutants are sent to the reference): 244 verdicts, 223 agree, 21 are F19, 0 unexplained.
* **Scope validation at every API entry that accepts a user-built term** (no loose bound variable, no context variable beyond the context). Before it, the hybrid kernel ABORTED on such terms
  (`check_lambda` evaluates a domain before inferring it). 288 ill-scoped terms (including hand-built loose bound variables) are now rejected, 0 accepted, no abort. Mutants M42-M44.

## New findings in the reference
* **F17** `intro` on a goal that is a definition unfolding to a Pi (`Not (Not P)`): the reference cannot complete `not_not_intro`; lockstep diverges at the `intro` step. The hybrid routes the
  non-literal case through NbE (mutant M46 restores the raw-id behaviour and is caught). Cause taken from my reading of the reference's `intro` in S3, not re-verified now.
* **F18** the reference ABORTS (not rejects) on an ill-scoped term in `whole_term_check` (backtrace: `whole_term_check -> check -> unify -> eval -> Env::operator[]`).
* **F19** the reference accepts a `Match` whose motive has a binder domain that is not a type: `match h as (lam (_ : Or). P) {}` on `h : False` is accepted by the reference and rejected by the hybrid
  ("Lambda domain must be a type", mutant M45); a well-formed motive is accepted by both. I did not try to exploit this in the reference.
* F16 again: `ex_elim` needs `apply (f' x p)` on a reverted Pi-typed hypothesis, which the reference cannot do.

## Still open (and why)
* **F5 is NOT verified.** `Ex Nat (lam n. n = 0) -> Ex Nat (lam n. 0 = n)` cannot even reach the elimination: `cases` treats `Ex`'s parameters `A` and `P` as indices, generalises EVERY occurrence of `Nat` in the goal
  (so `Zero : A` stops type-checking) and reverts unrelated hypotheses. This is the missing PARAMETER / INDEX distinction, which is also the core of F12 and of the useful half of F2b.
  "Fixed by construction in the new representation" is still only a claim for F5 until that is done.
* F2b (clash/injectivity through repeated indices), F12 (induct on a concrete index), F8 (an independent checker; scaling benchmarks later on a larger machine), M12.
* Inventory: 47 mutants defined (M1-M37, M39-M48), 46 caught; the survivor is M12.


---
# S7: parameters vs indices (kernel option "parameters", default OFF) -- gate PASS (2026-10-06)

`Kernel.set_option("parameters", True)`; `Kernel.param_slots(family_gid)` shows the analysis. With the option OFF every earlier gate is unchanged (re-run at full size).

## The rule
A family's leading result-index positions are PARAMETERS when, in EVERY constructor, the position holds a bare field variable that differs from the earlier parameters' variables, and EVERY occurrence of
the family inside any field type passes exactly that variable there (they form a prefix). Found: Eq [A, x], Le [n] (as in Coq), And/Or/Ex [all], List [A], Vec [A]; NOT parameters: a recursive field that passes
a different argument (NonUniform), a repeated variable (Rep slot 1), a slot after a non-variable one (Late).
* kernel (trusted): forcing records which field variables a PARAMETER slot determined; those are substituted EVERYWHERE, repeated occurrences included (`refl Nat a`, `le_n a`), in the result indices and in
  the constructor application. This is the standard eliminator's minor premise. Argument for soundness: parameters are fixed by the scrutinee's type, so by subject reduction the field equals the parameter.
* tactics: parameters are not generalised or abstracted; the motive's binder TYPES are specialised to the actual parameters (y : Nat, not y : A) so a premise like `same_ctor y _` type-checks; the induction's
  Fix is generic over the INDEX slots only; every constructor refuted at the same slot shares one same_ctor premise.

## Evidence
* F5: `Ex Nat (lam n. n = 0) -> Ex Nat (lam n. 0 = n)` completes and verifies. `ex_elim` too (only `f`, after the scrutinee, is reverted: 10 hypotheses).
* F2b-lite: `0 != S n` and `S n != 0` via Eq, and `Le (S n) 0 -> False`, close and verify. Branch goals are stated with `a`, not a dummy field (le_n: `Eq Nat a a`).
* F12 for parameter slots: induct on `Le 0 x` and on `Le a b` completes and verifies.
* parameter-free scripts are identical to the reference with the option ON.
* soundness canary: random tactic search on 14 FALSE statements (7 conditionally false, to exercise the new substitution): 16,800 attempts, 0 verified proofs. The canary CAN fail: with a deliberately unsound
  unifier (M59) it finds a verified proof of `0 = 1`.
* oracle (partial proofs type-check, goals well-formed) with the option ON: 0 unattributed failures; on the full random mix completed proofs rise from 73 to 454, all verified.
* mutants M49-M58 (analysis: uniformity, distinctness, prefix; kernel and tactic substitution; generalisation; binder specialisation; multi-constructor exploit; Fix genericity; kernel ignoring parameters) all
  caught; M59 caught by the canary. Inventory: 58 mutants defined (M1-M37, M39-M59), 57 caught; the survivor is M12.

## Limits
* F12 is only fixed for PARAMETER slots. The F12-attributed share of the full random mix barely moved (1,243 -> 1,207 per 6,000 scripts): concrete indices on true index slots (`Le n 0`) dominate it.
* injectivity (`S a = S b -> a = b`) is NOT derivable by `cases` (1 goal is left): the rest of F2b.
* motive-hole inference ignores parameters (untested). The soundness argument is an argument plus a canary, not a proof; the independent checker (F8) is meant to test the implementation.
See FINDINGS.md for the status of every finding.


---
# S8: the independent checker (F8) -- gate PASS (2026-10-06)
`indep_checker.py` (~330 lines) was written from the SPECIFICATION (its docstring states the rules) and built differently from the kernel: named variables with explicit capture-avoiding substitution, conversion by
weak-head normalisation, its own universe rule, structural-recursion guard, parameter analysis and Match rule. It reads the kernel's global table through the read-only `Kernel.global_info(gid)` (type, value, constructor
family/index, a family's constructors) and shares only the term-tree format with the code under test. `s8_indep_tests.py` is the gate.

## Evidence
* re-checks all 35 globals (library, logic kit, derived `__same_ctor`/`__Unit`/`__Empty`): 0 rejected in legacy mode; accepts the 7 kit proofs; rejects the 6 wrong proofs.
* differential on MUTATED proofs (scope-aware mutations of named terms: variable, constant, sort, argument swap, annotation, branch swap, scrutinee, Fix index): legacy 4,296 of 4,296 verdicts agree (138 accept, 4,158 reject);
  parameter mode 7,676 of 7,676 (389 / 7,287). Zero disagreements in either direction.
* guard corpus: 7 Fix terms valid or invalid only because of the structural-recursion guard agree with my labels in the kernel AND the checker.
* re-declaring the library through the kernel's checked path in parameter mode: the kernel and the checker reject exactly {cong, le_trans, transport} (see F20 below).
* the checker alone catches kernel mutants M1 (guard off; needs the guard corpus), M2, M3, M5, M8, M60.

## Findings it produced
* **F21 (fixed):** `check_lambda` only unified a binder annotation with the expected domain and never checked that the annotation is a type, so an ill-typed annotation convertible to a well-typed one was accepted
  (`__same_ctor_0 (Succ Q) Zero` and `__same_ctor_0 (Succ m) Zero` both reduce to `__Empty`). It is probably harmless computationally (the annotation equals the expected type) but it broke the invariant that every
  subterm of an accepted term is well typed. Now validated unless the annotation is literally the expected domain. 15 disagreements in the first run (all from exploit_le) disappeared; M60 restores it and is caught.
* **F20 (documented):** Match terms are mode-dependent. Library definitions written under the legacy rule (a branch uses its field binder for a parameter position) do not check under the parameter rule, where the branch must
  use the parameter's value. A library therefore has to be checked in ONE mode; the tactics produce terms consistent with the active mode.

## Limits
The checker implements the same SPECIFICATION as the kernel (forcing, the parameter substitution rule): it validates the implementation, not the design. No eta and no cumulativity (as in both kernels); Let/Meta/Quot are
rejected as unsupported; the guard is a conservative syntactic rule (a valid Fix it cannot recognise would show up as a disagreement); work is bounded by a fuel counter (exhaustion counts as rejection). Not done: the checker over the
S2-style random typed terms. Mutant inventory: 59 defined (M1-M37, M39-M60), 58 caught; the survivor is M12.


---
# S9, S10, F12 policy (2026-10-06)
* **S9 (F6 done):** `Below P n` (a recursive function returning a type: True / And (Below P k) (P k)), `below_all` by plain induction (the IH is `Below P x`, not a forall), `strong_induction`, and a client proof
  (`forall n, n <= n`), all verified by the kernel AND the independent checker, in both modes.
* **S10 (F2b injectivity):** `noconf.py` generates, per constructor and payload field of a non-indexed, non-parametric family, `Inj_c_q` (a type-valued Match) and `c_inj_q : c a = c b -> a_q = b_q` (transport along the
  equation), declared through the kernel's CHECKED path and re-checked by the independent checker. `S a = S b -> a = b` (which `cases` cannot close) and both fields of a binary `node` are proved; a wrong use fails.
  Not covered: indexed/parametric families, dependent fields (HEq), automatic use inside `cases`.
* **F12 policy:** in parameter mode `induct` refuses a hypothesis whose index slot is not a distinct variable (clear message); `cases` handles those through its motive. With that, the parameter-mode oracle shows NO failure
  of any kind over the full random distribution (12,000 scripts, 980 completed proofs all verified) and the gate runs it with zero tolerance. Mutant M61 removes the guard and is caught by it. Index generalisation with
  equations is NOT implemented (design note in FINDINGS.md).
* Inventory: 60 mutants defined (M1-M37, M39-M61), 59 caught; the survivor is M12. All gates (S2-S10, oracle, main differential) pass at full size.


---
# S11: index generalisation with equations (F12 done), F22, F23 (2026-10-06)
* **Feature:** `induct(hyp, eq_gid, refl_gid)` generalises concrete (or repeated) indices through equations (details in FINDINGS.md). `n <= 0 -> n = 0` and the capstone `b <= S a -> b <= a \/ b = S a` (equation, injectivity of
  S, transport) complete and verify in the kernel AND the independent checker; 583 mutants of those proofs get identical verdicts from both.
* **F22 (fixed, parameter mode):** `cases` did not exploit a refutation when any hypothesis was reverted. Found because `induct` with equations produces exactly such states.
* **F23 (mitigated):** the evaluator unfolds a Fix on a neutral argument, so extraction peels the Fix off an induct proof; this is the root cause of F12. The equation path wraps its Fix in `__barrier_k` (an identity that
  extraction keeps folded; excluded from the library index). I first tried passing the constructor application for the scrutinee and then mixed good/bad slots; both were wrong, which is why every slot gets an equation and the
  scrutinee is the original `h`. The kernel and the checker caught each wrong turn immediately.
* **Evidence:** zero-tolerance random oracle with equations: 16,000 scripts over both modes, ~10,800 induct applications, 0 ill-typed partial proofs, 0 ill-formed goals, 783 completed proofs all verified. Soundness canary with
  equations: 50,400 random attempts on 14 false statements, 0 verified proofs (both modes). All 13 gates plus the 40,000-term main differential pass.
* **Mutants:** M62 (cases exploit with reverted hypotheses), M63 (no barrier), M64 (barrier offered by the index) are caught by S11. Inventory: 63 defined (M1-M37, M39-M64); the 62 caught include M60-M64 individually; the survivor is M12. The full
  63-mutant re-run was NOT repeated after S11 (about 90 minutes); M62-M64 were run against S11 only.
* **Limits:** an index whose type depends on an earlier index needs heterogeneous equality (refused). F23's root cause is not fixed. Injectivity lemmas are still generated only for arity-0 families.


---
# S12-S15: probing x equations, the Prop/Type world, classical logic (2026-10-06)
* **Rewind hazard (found by review, fixed):** `__barrier_k` was not a derived global, so a probe that was the first to create it could not be rewound (the kernel's own guard threw). Fixed; S12 (6,600 probes per mode, 0 exceptions,
  0 non-determinism, 0 residue against a probe-free replay) and a deterministic first-creation test in S11; M65. Every parameter-mode `induct` now keeps its `Fix` through extraction (F23 mitigation complete for tactic proofs); M66.
* **F2b:** injectivity lemmas now cover every family whose slots are all parameters (List, And, Or, Ex witness) besides arity 0; dependent payload fields (Ex's proof) are skipped (they need HEq).
* **prop_elim (S13):** see F24. Kernel + independent checker + 3 configurations (off as control, on, on with parameters); M67-M70.
* **Prop/Type world (S14):** `prop_world.py`: Nat : Type1, Eq : Pi (A : Type1), ... : Type0, Ex over Type1, Le, add, Not, Iff; transport, symm, trans, cong, le_trans, zero_le proved by tactics under prop_elim+parameters; every global re-checked by the
  independent checker (0 rejected); Succ injectivity generated; equation-based induction, refutations, the Le-inversion capstone verified; zero-tolerance oracle (2 x 2,500 scripts, ~19,000 tactic applications, 159 completed proofs, all verified by
  both).
* **Classical logic (S15):** em, propext, dependent funext, choice (via Nonempty) as the only added axioms; false_elim, or_elim, by_contra proved by tactics; contrapositive verified; canary: 5,600 random attempts on 7 false statements, 0 verified,
  and POWER: the same generator completes and verifies 45 of 3,200 attempts on true classical statements (completed = verified).
* **Soundness status, stated plainly:** the consistency of these axioms with this kernel rests on (a) the standard set-theoretic-model argument for CIC with an impredicative Prop and restricted elimination, (b) our rules matching Lean/Coq's
  on the points that matter (large-field admission, syntactic-subsingleton elimination), and (c) tests. There is no mechanical proof. The kernel has no definitional proof irrelevance (fewer equalities than Lean), which only makes it more conservative.
* **Not done, and now the real bottlenecks:** no universe polymorphism or cumulativity (Eq cannot even be stated on proofs: Eq P p q needs A : Type1 but P : Type0), no proof irrelevance, no structures/records, no implicit arguments or instances, unary Nat (2024 is 2024 constructors),
  no automation beyond the basic tactics, no numbers beyond Nat. See ROADMAP.md.
* Inventory: 71 mutants defined (M1-M37, M39-M71), 70 caught (the survivor is M12). New ones were each run against their own gate only; the 90-minute full re-run has NOT been repeated.


---
# Gate index (2026-10-11): 32 gates, 601 mutants defined

`python3 run_gates.py` runs every gate (the list is `GATE_CMDS` in mutate.py, the one place to add a gate); `python3 mutate.py --check` verifies every mutant's anchor; `python3 mutate.py PREFIX...` runs the mutants
whose names start with a prefix (each is built in isolation and must make some gate fail). Each gate's docstring says what it pins down and why.

| gate | what it pins down |
|---|---|
| `s2_tests.py` | the typing core against the reference kernel |
| `s3_tests.py`, `s4_tests.py`, `s5_tests.py`, `s5_probe_tests.py`, `harness.py` | metas / unification / intro / apply / rewrite, cases / induct, the library index and the probing protocol, lockstep against the reference (raw `apply`, no `simp`) |
| `s6_logic_tests.py`, `s6_f6_tests.py`, `s9_strong_tests.py` | logical vocabulary, induction with Pi goals and `generalizing`, strong induction by a bounded predicate |
| `s7_param_tests.py`, `s13_prop_tests.py`, `s14_prop_world_tests.py`, `s15_classical_tests.py` | parameters vs indices, Prop with restricted elimination, the Prop/Type world, the classical axioms |
| `s8_indep_tests.py` | the independently written checker against the kernels |
| `s10_noconf_tests.py`, `s11_eqgen_tests.py`, `s12_probe_eq_tests.py` | injectivity lemmas, index generalisation with equations, probing x equations |
| `s16_quot_tests.py`, `s17_irrelevance_tests.py` | quotients; definitional proof irrelevance |
| `s18_deep_tests.py`, `s19_boundary_tests.py`, `s20_reservation_tests.py`, `s21_view_tests.py` | terms of any depth, the Python API boundary, well-founded global environment, reading terms back |
| `s22_simp_tests.py` | display normal form, `change`, `apply` with folded premises (which reading of the lemma meets the goal) |
| `s23_ring_tests.py`, `s24_bin_tests.py`, `s25_nring_tests.py` | Nat library and `ring`; binary naturals and their bridge; `ring` over N and the shared lemma template |
| `s26_memo_tests.py` | node facts, shared-term traversals, typing memo, linear proof extraction |
| `s27_fuel_tests.py` | the work budget and all-or-nothing tactics |
| `s28_headunfold_tests.py` | definitions at the head of a goal or hypothesis are unfolded on syntax |
| `s29_apply_pi_goal_tests.py` | `apply` on implication / negation / universal goals (arity rule) |
| `oracle_gate.py` | the self-consistency oracle (random scripts, zero tolerance) |

Libraries: `lnl_base.py` (logic, equality, Nat), `nat_lib.py`, `bin_lib.py` (Pos / N), `quot_lib.py` (quotients, `funext`, Prod, `Quot.lift2`), `int_lib.py` (Z), `ring.py` + `ring_lib.py` (the `ring` tactic and its lemma
template), `tactics.py` and `author.py` (the authoring layer). `repros/` holds small scripts that reproduce findings F28-F30.
