# Session notes — 2026-10-04: `le_trans` and `induct()` with generalized hypotheses

Full technical write-up: **INDUCT_REVERT_SWEEP_NOTES.md**. This file is the short version.

## Outcome
* `le_trans : (a b c) → Le a b → Le b c → Le a c` is proven (scripted; `curriculum.setup_le_trans`) and
  independently verified by `whole_term_check`; usable as a library lemma (`le_trans4` derived from two uses
  in the test).
* Root cause was a chain of nine separate bugs in `induct()` and shared kernel machinery (revert mechanism,
  over-coarse sweep, peeling mechanism, unwrapped zero-field branch, unforced result indices, a wrongly
  pre-applied IH, an unspecialized reverted position in the generic Fix body, a sequential-substitution
  collision in solved-meta beta-reduction, and — pre-existing, unrelated — bound structure embedded in the
  target). All fixed, or (the last) rejected cleanly.
* 25/25 regression tests pass (24 existing + `test_indexed_induction.py`). Search-tree sizes are back within a
  few percent of the true pre-session baseline.

## Files changed
* `itp_kernel.hpp` — `induct()`: fixpoint revert sweep, reverted-hyp peeling/Lambda wrapping, forced result
  indices, reverted-position specialization, embedded-binder guard (`detail::has_internal_binder_ref`);
  `requote_captured_syntax` and `resolve_metas`: simultaneous substitution (`substitute_vars` forward-declared).
* `curriculum.py` — `setup_le_trans`; ORDER_CURRICULUM comment corrected; a stray duplicate comment removed.
* `hypergraph.py` — unchanged net of this session's debug print (added and removed).
* New: `test_indexed_induction.py`, `INDUCT_REVERT_SWEEP_NOTES.md`, this file.
* `CURRICULUM_PLAN.md` — le_trans status, Stage B list, §7 item 2.

## Corrections to earlier claims (kept for the record)
* A source comment claimed `le_trans` "confirmed" the motive-scheme shift change (`arity+3 → 2*arity+3`)
  was necessary. False; replaced. The change is derived from binder counting and is an equivalent mutant
  under the new guard (INDUCT_REVERT_SWEEP_NOTES.md §3).
* Several intermediate diagnoses this session were wrong and corrected (a `print_node` mislabeling red
  herring; test-script errors). Method notes in §4 of the write-up.

## Open items
* Embedded bound structure in targets / reverted hyp types: rejected cleanly, not supported. `cases()`'s
  reverted-hyp handling has probably the same issue — unverified.
* Concrete-index (Mode 1) `induct()` gap (pre-existing, untouched).
* `le_trans` is not search-discoverable with plain `UniformPolicy` at ≤1200 simulations; I did not try other
  policies, lower `top_k`, or action restriction.
* Not yet done: Stage B continues (`le_antisymm`, trichotomy, monotonicity, …).
