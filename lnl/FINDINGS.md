# Findings tracker (F1-F19) -- status as of the parameter work

Legend: **fixed** = fixed and covered by a test that fails without it; **by construction** = the hybrid representation removes the cause (covered by differential/oracle tests);
**partial** = part is done, the rest is stated; **open**.  "Reference" = the frozen baseline kernel; "hybrid" = this kernel.

| id | finding | status | evidence / where |
|---|---|---|---|
| F1 | inductive admission (universe rule, large fields) | fixed (baseline b777dc8); ported | S2 admission corpus agrees with the reference |
| F2a | extraction inlined constants | fixed (baseline) | probes; `extract_proof_term` uses NoDeltaScope |
| F2b | no clash/injectivity machinery | **fixed for non-indexed, non-parametric families** | discrimination: `0 != S n` via Eq and `Le (S n) 0 -> False` close in parameter mode (S7). Injectivity: `noconf.py` generates `Inj_c_q` and `c_inj_q` lemmas through the checked path (S10: `S a = S b -> a = b`, both fields of `node`). OPEN: indexed/parametric families, dependent fields (HEq), automatic use inside `cases` |
| F3 | nested induction | fixed (baseline); by construction | S3/S4 lockstep, oracle, soak |
| F4 | induct after cases | fixed (baseline); by construction | S4 |
| F5 | Ex over a lambda cannot be eliminated | fixed in parameter mode | S7 `ex_lambda`; needs `parameters` ON (the kernel otherwise generalises Ex's parameters) |
| F6 | strong induction needs a design | fixed | Pi-headed induct/cases + `generalizing` (S6 F6); the bounded-predicate probe (S9): `Below`, `below_all`, `strong_induction`, and a client proof, verified by the kernel and the independent checker in both modes |
| F7 | no logical vocabulary | fixed | True, False, And, Or, Ex, Not, Iff declared; 7 proofs verified; 6 wrong proofs rejected (S6) |
| F8 | the checker is not independent; scaling unmeasured | **partial** | independent checker DONE (indep_checker.py, S8): written from the spec with named variables; exact agreement with the kernel on 11,972 mutated proofs, 7 guard cases, all library definitions. OPEN: scaling benchmarks (deferred to the large machine) |
| F9 | forced constructor application used dummy binders | fixed | kernel Match typing + both tactics; M22-M26 |
| F10 | sequential marker substitution collides (reverted hypotheses) | by construction | 184+ events validated against the reference's ill-formed state |
| F11 | field after a recursive field used the IH's level | by construction | custom family test; M17 |
| F12 | induct on a concrete index (the generic Fix cannot supply the specialised binder types) | **fixed** | index generalisation with EQUATIONS (S11): `induct(hyp, eq_gid=.., refl_gid=..)`. Parameter slots need nothing (S7). Parameter mode refuses a concrete index when the ids are not supplied; legacy mode keeps the reference behaviour unless they are. Capstone: `b <= S a -> b <= a \/ b = S a` (equation + injectivity + transport) verifies in the kernel and the independent checker. Zero-tolerance oracle: 0 failures over 16,000 random scripts in both modes; M61-M64 |
| F13 | rewrite acts inside meta spines / accepts ill-typed motives | fixed | zonk + motive check; M27, M28 |
| F14 | exploited sibling branch wrapped in duplicated lambdas | fixed | `exploit_le` verifies in the hybrid kernel only |
| F15 | deferred constraint re-unified in the wrong context (abort under binders) | fixed | M31 |
| F16 | reference `apply` misbinds Pi-typed hypotheses | by construction | 7/7 variants; ex_elim |
| F17 | reference `intro` on a definition that unfolds to a Pi | fixed | NbE path; `not_not_intro`; M46 |
| F18 | reference aborts on ill-scoped terms (hybrid did too; now validated) | fixed in the hybrid | 288 ill-scoped terms rejected; M42-M44 |
| F19 | reference accepts a Match motive whose binder domain is not a type | hybrid is stricter | minimal case + 21 mutant disagreements; M45 |
| F22 | `cases` never exploited a refutation when any hypothesis was reverted (reference behaviour) | fixed in parameter mode | `S m = 0` was not refuted as soon as another hypothesis mentioned `0`; the same_ctor premise is the outermost motive layer, so the reverted binders move one level up; M62. Legacy mode keeps the reference behaviour (lockstep) |
| F23 | the evaluator's `apply(VFix)` unfolds one layer EVEN WHEN the decreasing argument is a neutral variable (a standard fix unfolds only on a constructor) | mitigated; root cause open | proof extraction is quote(eval(..)), so it peels the Fix off an induct proof and leaves a Match on the concrete hypothesis with branches typed for the generic parameter: the ROOT CAUSE of F12. Coincides with the generic view for distinct-variable indices, which is why everything else worked. The equation path wraps its Fix in the transparent `__barrier_k` (kept folded by extraction); M63. Proper fix = a stuck-Fix neutral (about 80 sites in the evaluator, and it changes reference lockstep). The independent checker implements the standard rule and agreed on every test |
| F20 | Match terms are MODE-DEPENDENT: under `parameters` a branch must use the parameter's value, not its field binder | documented (design consequence) | kernel and independent checker both reject exactly cong, le_trans, transport when re-declared in parameter mode |
| F21 | a lambda CHECKED against a Pi never validated its binder annotation (infer did) | fixed | found by the independent checker; M60; S2/S3/S4/S6/S7 unaffected |

Other open items: M12 (`drain_pending` is effectively dead at the tactic level); motive-hole inference ignores parameters (untested path);
the default of the `parameters` option is OFF (a decision for the fingerprint stage).

## Implementation note: index generalisation with equations (S11)
If any non-parameter index slot is not a distinct variable (a concrete term, a repeated variable, or a variable that also occurs in a parameter), EVERY non-parameter slot gets an equation `Eq T_i marker_i c_i` as one more
(innermost) layer of the motive; the goal is left alone; the Fix is applied to the actual indices, `h`, the reverted hypotheses and `refl T_i c_i`. In the branch contexts the agent does NOT force at those slots (the equation
reads `Eq T r_i c_i` in terms of the constructor's own fields) while the kernel forces against the Fix's generic index, so `?sub` is passed the kernel-substituted value for any field the agent left unforced. Because no context
variable is substituted, the scrutinee is passed as the ORIGINAL `h`. The Fix is wrapped in `__barrier_k` (see F23). Refused: an index whose type depends on an earlier index (it would need heterogeneous equality).
