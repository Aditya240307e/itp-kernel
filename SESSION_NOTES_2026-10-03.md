# Session notes -- 2026-10-03

## What changed, in order

1. **`rewrite` tactic wired into search** (`hypergraph.py`): `ActionKind.REWRITE` added;
   `enumerate_legal_actions` now offers rewrite candidates (generic Eq-typed-hypothesis rewrites,
   forward and backward via `symm`, plus a registry of library-fact rewrites configured via
   `REWRITE_TRANSPORT_GID` / `REWRITE_SYMM_GID` / `REWRITE_EQ_CONST` / `REWRITE_CANDIDATE_FNS`).
   `Node.cached_rewrite_extra` carries each candidate's `(x, y, transport_gid, symm_gid)`
   separately from `Node.cached_candidates` (kept as a uniform `NodeId` list so
   `term_features` keeps working unchanged). `curriculum.build_rewrite_registry` supplies the
   registry entries for `mul_succ_right`, `mul_zero_right`, `add_succ_right`, `add_zero_right`.

2. **New kernel bindings** (`itp_bindings.cpp`): `decompose_app` (purely syntactic
   head/argument-spine decomposition, no `eval()`) and `is_pi` (zero-unfolding `Pi` check).

3. **Soundness fix in `rewrite_eq`'s caller** (`hypergraph.py`): `enumerate_legal_actions` no
   longer offers `REWRITE` when the current goal's target is still a `Pi` (`kernel.is_pi` guard).
   Root cause: `detail::generalize`'s fresh motive-variable level (`g.ec.size()`) collides with a
   still-un-intro'd `Pi`'s own, pre-assigned bound-variable level at that exact value -- confirmed
   directly (a `rewrite` offered on a `Pi`-headed goal silently replaced that `Pi`'s own bound
   variable, everywhere it appeared, with whatever the replacement value was). `cases()`/`induct()`
   never hit this since `detail::generalize`'s only other caller is always already past every intro.

4. **Completeness fix in `conv_value`/`unify_values`** (`itp_kernel.hpp`): comparing two
   independently-reconstructed stuck `Match` values that are semantically equal but structurally
   different (one via a genuine `Lambda`-application chain, one via `infer()`'s own quoted-back
   value substitution) could hard-fail even though the two sides are genuinely, definitionally
   equal. Fixed by falling back to quoting both full `VNeutral` sides once and comparing the
   resulting `NodeId`, when the frame-by-frame structural spine walk fails -- a single,
   non-mutually-recursive reconstruction, not the (tried first, found unsafe via a direct A/B test
   that reproduced a real segfault) `branches_convertible` fallback the pre-existing
   captured-env-size-mismatch case already uses. See `test_stuck_match_convertibility.py` for full
   derivation and regression coverage (positive, depth-stress, and soundness-negative cases).

5. **`mul_succ_left_unfold` added to `curriculum.py`** (`setup_mul`): `(n m:Nat) -> Eq Nat
   (mul (Succ n) m) (add (mul n m) m)`, a direct restatement of `mul`'s own computation rule, built
   via `refl`. This is the lemma whose own verification first exposed the bug in (4).

6. **`induct`/`intro` ordering discipline documented and regression-tested**
   (`test_induct_pi_ordering.py`): calling `induct()` while the goal still has a remaining `Pi`
   (i.e. not every quantified variable has been `intro`'d yet) produces a `Goal.hyps` listing whose
   index no longer matches the hypothesis's actual de Bruijn level for any hypothesis introduced
   after that point -- `induct()`'s own `branch_hyps`/`branch_ec` construction is correct; the
   divergence comes from the remaining `Pi`'s own body still referencing the not-yet-introduced
   variable at the level it would have had *before* `induct()` inserted the predecessor and IH
   ahead of it. **Not a soundness gap** (verified directly, twice, in
   `test_induct_pi_ordering.py`): the position the goal's own target actually uses is
   type-incompatible with naive use (it is really the IH), and a reference built from the naive,
   listing-index assumption is a safe no-op against the real goal, never a silent corruption. The
   established curriculum (`mul_succ_right`, `mul_comm`, `add_comm`) already follows the correct
   discipline -- `intro` every quantified variable first, then `induct` on the one wanted, by its
   own hyp index -- which is also just the standard, idiomatic way such proofs are written. This
   matters for the search/RL setting specifically: `hypergraph.py`'s own `REWRITE_CANDIDATE_FNS`
   and generic Eq-hypothesis path assume listing-index == level, and nothing currently stops a
   policy from calling `INDUCT` before every variable is introduced. Worth a defensive guard in
   `induct()` itself later (mirroring the `is_pi` guard in point 3) if this shows up in practice --
   not done yet, flagged here as a known, low-priority, non-blocking follow-up.

7. **`mul_comm` re-derived via `rewrite`** (demonstration/validation only, not added to the
   permanent library as a second declaration): confirms `rewrite` + the registry can produce a
   genuinely simpler proof than the original hand-built `cong_add_right`/`trans` chain, once the
   `intro`-before-`induct` discipline from (6) is followed. The original, independently-verified
   hand-built `mul_comm` remains the one declared in the library; nothing about it changed.

## New permanent tests
- `test_stuck_match_convertibility.py` -- regression coverage for (4): single-level control,
  the original doubly-nested bug, triple/quadruple-nesting depth stress (must stay fast, not just
  terminate), and three soundness-negative cases confirming genuinely false statements are still
  rejected.
- `test_induct_pi_ordering.py` -- regression coverage for (6): the good-discipline alignment
  invariant, the bad-discipline divergence, and both safety properties backing the "not a soundness
  gap" claim.

Both pass, along with the full pre-existing suite (23 files total), with no timeouts, no segfaults,
and no timing regressions versus before today's changes.

## Addendum (same day, continued): defensive guard in cases()/induct(), and add_comm resolved

8. **`cases()`/`induct()` now reject outright, kernel-side, when the goal's own target is still a
   `Pi`** (new `itp::detail::is_pi_headed` helper in `itp_kernel.hpp`, shared with the Python-facing
   `is_pi` binding, which now delegates to it instead of duplicating the check). Confirmed directly
   by a careful, temporary A/B disable-and-recheck: the pre-existing `shift_levels_from` mechanism
   (from a still-earlier session, intended to handle exactly this "remaining Pi" situation) *does*
   shift the remaining Pi's own stale body-reference, but under-shifts by exactly one level for any
   branch that also inserts an IH (the recursive case) -- its own shift amount (`arity + 1`) only
   accounts for the scrutinee, not the additional IH hypothesis `induct()`'s own `Succ`-style branch
   also adds. This is a genuine, narrower, previously-unidentified gap in an existing partial fix,
   not a duplicate of it. The guard closes the gap by rejecting the vulnerable call outright rather
   than attempting to correct the shift arithmetic (lower risk, and the correct discipline --
   `intro` everything before `induct`ing -- remains the standard, idiomatic approach regardless of
   whether the arithmetic gets fixed). `test_induct_pi_ordering.py` updated accordingly: its own
   "bad discipline" case now verifies the guard fires immediately and cleanly for both `cases()` and
   `induct()`, with a working-again control confirming the identical call succeeds once the missing
   `intro` is added.

9. **The `add_comm` search-extraction bug (`setup_add_comm`'s own docstring, previously "a distinct,
   separately-discovered bug... not yet isolated") is the SAME root cause as (8), confirmed
   directly, not inferred:** its own documented symptom, `"different variable/constant heads"`, is
   the exact symptom this session's own `shift_levels_from` investigation (point 8) traces to this
   same gap, and is extensively pre-documented elsewhere in `itp_kernel.hpp` (`cases()`'s own
   comment around its `shift_levels_from` call) as arising from precisely "casing/inducting on a
   hypothesis before introducing a further variable the goal still quantifies over" -- exactly what
   an exploring UniformPolicy search is free to do. The guard in (8) prevents this path from ever
   producing a goal that can report solved-but-fail-`whole_term_check` again. Additionally, and
   independently of the guard: a full `add_comm` proof via `rewrite` (not the existing
   `cases`/`induct`/`apply`-with-congruence-lemmas style) now exists and verifies end-to-end, using
   a new `add_succ_left_unfold` lemma (`(n m:Nat) -> Eq Nat (add (Succ n) m) (Succ (add n m))`,
   mirroring `mul_succ_left_unfold`'s own construction exactly) added to `setup_add_comm`. Both
   `add_succ_left_unfold` and `mul_succ_left_unfold` are now also in `build_rewrite_registry` (6
   candidate functions total), so search itself, not just hand-driven proof construction, can use
   them going forward. The original hand-built `add_comm` (via `symm`/`trans`/`cong`) is kept as-is
   in `setup_add_comm` regardless -- not because anything about it was ever broken, but because it
   predates `rewrite` and remains a perfectly good, independent proof of the same lemma, the same
   reason `mul_comm`'s own hand-built proof is kept in `setup_mul` alongside the newer rewrite path.

## Known open items
- `le_trans`: **resolved 2026-10-04** -- see SESSION_NOTES_2026-10-04.md and
  INDUCT_REVERT_SWEEP_NOTES.md. (Originally: kernel limitation with indexed-relation induction when an
  earlier hypothesis depends on what's being generalized.)
- `add_comm` search-extraction bug: **resolved** (see point 9, above) -- no longer open.
- Anti-self-induction: deferred to the learned-policy phase, as before.
- `shift_levels_from`'s own off-by-one (point 8, above) is guarded against, not corrected at the
  source. Fixing the arithmetic itself (so `cases()`/`induct()` could safely accept a remaining Pi
  rather than rejecting it) was deliberately not attempted: low value (the correct, standard
  discipline works regardless and is what the whole existing curriculum already follows) for real
  risk (three separate `shift_levels_from` call sites within `induct()` alone, each handling a
  different sub-case, with no isolated test harness for the arithmetic itself). Worth a look only if
  a concrete future need for calling `induct()`/`cases()` before every variable is introduced
  actually arises.
