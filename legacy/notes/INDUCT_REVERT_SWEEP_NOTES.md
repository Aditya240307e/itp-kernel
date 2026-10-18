# `induct()` with generalized hypotheses — root causes, fixes, evidence (2026-10-04)

**Status.** `le_trans` is proven by a scripted tactic proof (`curriculum.setup_le_trans`) and
independently verified by `whole_term_check`. All 25 regression tests pass, including the new
`test_indexed_induction.py`, whose individual checks were mutation-tested (§3). One pre-existing,
unrelated limitation was found along the way and is now rejected cleanly instead of aborting the
process (§1, cause 9). Free-search results for `le_trans`: §6.

The extracted proof is the textbook one — induction on the *second* hypothesis, generalizing the first:

```
λ a b c h1 h2. (match h2 with
   | le_n n'       => λ h1':(Le a n'). h1'
   | le_S n' m' p  => λ h1':(Le a b).
        le_S a m' ( (fix self := λ i0 i1 s. match s with
                        | le_n nf        => λ h1'':(Le a nf). h1''
                        | le_S nf mf pf  => λ h1'':(Le a i0). le_S a mf (self i0 mf pf h1'')
                     ) b m' p h1' ) ) h1
```

## 1. What was wrong — a chain of nine causes

`le_trans` failed at the first of these; each fix exposed the next. Every diagnosis below was
confirmed with a raw term dump, a backtrace, or a direct comparison against the unpatched baseline —
not inferred from `print_node` output (see §4).

| # | Cause | Symptom | Fix |
|---|-------|---------|-----|
| 1 | `induct()` had no revert mechanism (`cases()` does): an earlier hypothesis depending on a generalized index (`h1 : Le a b` while inducting on `h2 : Le b c`) was never generalized. | `constructed motive failed to type-check` | port the mechanism |
| 2 | The port used `cases()`'s *contiguous* sweep ("everything from the earliest dependent hyp to the end"). Harmless for `cases()`'s small groups; for `induct()`'s double/nested induction it sweeps genuinely unrelated hyps every call. | `add_succ_right` search tree **749 → 4744 nodes** and unsolved at 200 sims; SIGABRT in `test_gil_release` / `test_phase2_batching` / `test_process_driver`; process abort at 3000 sims | selective, **fixpoint** dependency sweep (a hyp is reverted iff its type transitively mentions the scrutinee/indices/another reverted hyp) |
| 3 | Copied `cases()`'s `eval()`/`quote()` peeling of reverted Pi-layers, but `induct()`'s `branch_expected` is built by *syntactic* `substitute_vars` (not a reduced application), so the numbering schemes don't line up; also used `arg_types_agent.size()` as the base level, missing the IH's slot. | `Env::operator[]` out of bounds (gdb: inside the peel) | peel syntactically; base level = `branch_ec.size()` |
| 4 | True-runtime branches were wrapped only for constructor fields. `le_n` has **zero** fields, so its branch was a bare value where the motive demands a function of the reverted hyp. | `mismatched value shapes` (`Le a n` vs `Π (Le a n). Le a n`) | wrap each branch with a Lambda per reverted hyp |
| 5 | Result indices were **unforced** on both the agent-facing and true-runtime sides. `le_S`'s first result index is a bare field variable `n'`, forced to the concrete `b` in the field types, so branches mentioned `Le a n'` where `check_match_branches` expects `Le a b`. (`cases()` and `check_match_branches` already had this fix, with a comment describing exactly this bug class; it was never ported.) | `spine comparison differs` (`Le a n'` vs `Le a b`) | apply `apply_ctor_index_forcing_to_result_indices` on both sides |
| 6 | An earlier attempt ("Gap D") pre-applied the recursive call to the reverted hyps. The IH is a *function* of them (`IH : Π h1:(Le a b). Le a c'`); pre-applying handed the agent a non-function. | `applying a term that is not a function` | removed |
| 7 | Inside the **generic** Fix body, a reverted position must receive the branch-local specialization (the Fix must stay valid for every index, including the smaller one a recursive call passes). The agent's valid `IH(h1_orig)` became `self i0 mf pf h1_orig` with `h1_orig : Le a b` where `Le a i0` is required. | `spine comparison differs` | supply the branch-local parameter at reverted positions (same principle as the index/scrutinee positions already handled) |
| 8 | **Latent bug**: solved-meta beta-reduction (`requote_captured_syntax`, `resolve_metas`) substituted arguments *sequentially*, so an argument whose free level equalled a later, still-pending position was rewritten. `self` (re-embedded at level 9) was overwritten by the last slot's value (position 9). It never surfaced before because the IH was always the *last* argument position. | a proof variable (`h1''`) in function position | simultaneous substitution (`substitute_vars`) |
| 9 | **Pre-existing, not part of `le_trans`**: `induct()` builds child goals by syntactic substitution, which does not re-level bound structure *embedded in the target* (e.g. a lambda-redex whose body refers to its own binder). Reproduced on the **unpatched baseline** with a plain non-indexed goal `(n:Nat) → Eq ((λx. x) n) n`. | process abort (`Env::operator[]` out of bounds) at the next tactic that evaluates the goal | precise guard (`has_internal_binder_ref`): reject cleanly. Soundness was never at stake (`whole_term_check` re-verifies everything). |

## 2. Verification

* `le_trans` (reverted hyp *before* the scrutinee) and the same lemma with the hypotheses reordered
  (reverted hyp *after* it — the other direction of the fixpoint) both pass `whole_term_check`.
* `succ_le_succ` (the goal mentions **both** indices) passes; it was not provable at all before the
  forced-index fix, since the step target mentioned `n'` while the IH mentioned `n`.
* Bare `induct` on `Le` with no reverted hyp passes; a lambda whose binder is *never referenced* is
  correctly **not** rejected by the guard, and that proof verifies (guard precision).
* Three soundness negatives are rejected (stale `h1` where the base goal needs `Le a n'`; `le_S a c' h1`
  with `h1 : Le a b`; the verified proof checked against `Le c a`).
* Search-tree parity with the true pre-session baseline (`UniformPolicy`, 200 sims):
  `symm 7/7`, `trans 14/14`, `add_zero_right 382 vs 376`, `add_succ_right 769 vs 749`,
  `n_eq_add_n_zero 52/52`, `le_refl 6/6`, `zero_le 262 vs 228` (patched vs baseline).
* The 3000-simulation exploration of `add_succ_right` that used to abort now completes (tree 315).

## 3. Mutation testing (each fix re-broken in an isolated copy; `test_indexed_induction.py` must fail)

| Mutant | Caught by |
|--------|-----------|
| M1 sequential substitution restored | le_trans → `applying a term that is not a function` |
| M2 unforced agent-facing indices | forced-index consistency assertion |
| M3 unforced true-runtime reverted types | le_trans → `spine comparison differs` |
| M3b unforced index positions in `subgoal_applied` | `succ_le_succ` (the only goal whose solution mentions an index var) |
| M4 no specialization at reverted positions | le_trans → `spine comparison differs` |
| M5 contiguous (over-)sweep | structural over-sweep guard (it still *proves* everything — exactly why the guard is needed) |
| M6 no Lambda for reverted hyps in branches | le_trans → `unbound local variable` |
| M7 "Gap D" reintroduced | le_trans → `applying a term that is not a function` |
| M8 motive-scheme shift `2*arity+3 → arity+3` | **not caught — an equivalent mutant** (see below) |

**M8, honestly.** An earlier source comment claimed `le_trans` "confirmed" the `arity+3 → 2*arity+3`
change was necessary. That was false: `le_trans` verifies with either value, and the claim was made while
the real causes were still unfound. The corrected amount is derived from binder counting (self, the Fix's
index params and scrutinee, then the motive's own index lambdas and scrutinee = `2*arity+3`; it equals the
old literal at `arity=0`, so it is a no-op for every non-indexed family). It only ever moves `Var`s ≥
`g.ec.size()`, i.e. exactly the embedded bound structure that the cause-9 guard now rejects up front — so
for every accepted input it is a no-op and M8 is unobservable by construction. It is kept (and now derived
from the actual binder levels instead of a closed-form guess) so that relaxing the guard cannot silently
reintroduce an undercount.

## 4. Method notes (mistakes worth recording)

* `print_node` assumes a closed term at depth 0, so a **free** low-level variable inside a binder prints as
  the *bound* `x_k`. It sent me after a non-existent `fn_type` bug. Raw dumps of each `Var`'s literal level
  are the reliable view.
* A gdb breakpoint without a hit count catches the *first* of N occurrences (here 69), which was unrelated
  search exploration. Count occurrences first, then use `ignore`.
* Several "kernel bugs" in this session were bugs in my own test scripts (wrong de Bruijn-style level for
  `m`, a missing constructor argument, a wrong index after an extra forced field). Check the input first.

## 5. Known limitations / open items

* **Embedded bound structure** (cause 9) is rejected, not supported. A proper fix needs a per-branch
  re-leveling pass: build the branch-facing generalized goal with embedded structure in a collision-free band,
  substitute the constructor's values, then shift down by *that branch's* binder count (differs per
  constructor, and for IH vs. target). `cases()` re-levels the *target* via `eval()+quote()` and is not
  affected there, but `cases()`'s `reverted_target` construction (`mk_pi(g.hyps[j].type, …)`) places a hyp
  type that is well-formed in context `j` at context `S`, so a reverted hyp with a *referenced* embedded
  binder is probably mis-leveled there too — **unverified; not tested**.
* `induct()`'s documented "KNOWN, OPEN GAP" for concrete-index (Mode 1) induction (e.g. inducting on a hyp of
  type `Le n Zero`) is untouched; `le_trans`'s `h2 : Le b c` has two bare-variable indices (Mode 2).
* `induct()` still rejects a remaining Pi in the target (the earlier `is_pi_headed` guard); intro first.

## 6. Free search on `le_trans` (follow-up to the earlier session's "Problem C")

The earlier session reported that unguided `UniformPolicy` search on `Le`-shaped goals could abort the
process (`Env::operator[]` out of bounds) even on the unpatched kernel ("Problem C"). Re-tested on the final
kernel, each run in an isolated subprocess so an abort, a timeout and a solve are distinguishable
(sandbox: 4 GB RAM, 1 core):

| goal | sims / top_k | outcome | tree | peak RSS |
|------|--------------|---------|------|----------|
| `le_trans` | 600 / 16 | clean exit, **unsolved** | 120,083 | 1.65 GB |
| `le_trans`, hypotheses reordered | 600 / 16 | clean exit, **unsolved** | 120,087 | 1.65 GB |
| `le_trans` | 1200 / 16 | clean exit, **unsolved** | 262,276 | 3.65 GB |
| `le_trans`, hypotheses reordered | 1200 / 16 | clean exit, **unsolved** | 262,228 | 3.64 GB |
| `le_trans` | 3000 / 16 | killed, **rc=137 (SIGKILL — out of memory)**, *not* 134 (SIGABRT) | — | > 4 GB |

What this does and does not show:

* **No kernel abort in any run** (≈ 2,400 simulations over both orderings, plus the 3000-simulation
  `add_succ_right` run, which used to abort and now completes). Problem C was **not reproduced** — but it
  was found at unspecified budgets, so this is absence of evidence at these budgets, not a proof.
* **`le_trans` is not search-discoverable at ≤ 1200 simulations** with plain `UniformPolicy`: the proof is
  ~10 tactics deep with five hypotheses to branch over, and the tree reaches 262K nodes (~14 KB/node)
  without a solve. I did not try lower `top_k`, other policies, or action restriction, so "unfindable" is
  *not* established. 3000 simulations exhausts the sandbox's memory — a search-tree memory characteristic
  (Python side), not a kernel fault.
* Consequently `le_trans` is a **scripted library lemma** (`curriculum.setup_le_trans`, same convention as
  `mul_zero_right` in `setup_mul`), not an `ORDER_CURRICULUM` goal. It is a natural early benchmark for a
  learned policy: a short, verified, known-depth proof that uniform search cannot reach.

## 7. Reproduce

```
python3 build_kernel.py
python3 test_indexed_induction.py          # the whole chain above, ~1 min
for f in test_*.py; do python3 $f || echo FAILED $f; done
python3 -c "import curriculum as c; k,r=c.build_base_kernel(); r.update(c.setup_nat_order(k,r)); \
            r.update(c.setup_le_trans(k,r)); print('le_trans gid', r['le_trans_id'])"
```
