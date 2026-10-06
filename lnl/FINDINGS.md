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
| F12 | induct on a concrete index | **partial** | fixed for PARAMETER slots (S7); OPEN for true index slots (needs equation generalisation) |
| F13 | rewrite acts inside meta spines / accepts ill-typed motives | fixed | zonk + motive check; M27, M28 |
| F14 | exploited sibling branch wrapped in duplicated lambdas | fixed | `exploit_le` verifies in the hybrid kernel only |
| F15 | deferred constraint re-unified in the wrong context (abort under binders) | fixed | M31 |
| F16 | reference `apply` misbinds Pi-typed hypotheses | by construction | 7/7 variants; ex_elim |
| F17 | reference `intro` on a definition that unfolds to a Pi | fixed | NbE path; `not_not_intro`; M46 |
| F18 | reference aborts on ill-scoped terms (hybrid did too; now validated) | fixed in the hybrid | 288 ill-scoped terms rejected; M42-M44 |
| F19 | reference accepts a Match motive whose binder domain is not a type | hybrid is stricter | minimal case + 21 mutant disagreements; M45 |
| F20 | Match terms are MODE-DEPENDENT: under `parameters` a branch must use the parameter's value, not its field binder | documented (design consequence) | kernel and independent checker both reject exactly cong, le_trans, transport when re-declared in parameter mode |
| F21 | a lambda CHECKED against a Pi never validated its binder annotation (infer did) | fixed | found by the independent checker; M60; S2/S3/S4/S6/S7 unaffected |

Other open items: M12 (`drain_pending` is effectively dead at the tactic level); motive-hole inference ignores parameters (untested path);
the default of the `parameters` option is OFF (a decision for the fingerprint stage).
