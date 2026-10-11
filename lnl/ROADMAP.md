# Roadmap: from this kernel to IMO-level and undergraduate mathematics (a judgment, not a measurement)

## Where we are (measured)
A small, heavily tested trusted core and tactic layer: dependent types, inductive families with parameters and indices, structural recursion, a Prop/Type world with restricted elimination and the four classical axioms, tactics
(intro, apply, rewrite, cases, induct with equation generalisation, strong induction by bounded predicates, generated injectivity lemmas), an independent checker, differential gates, zero-tolerance random oracles and soundness
canaries. The library is: Nat, Eq, Le, add, And/Or/Ex/True/False/Not/Iff, a dozen lemmas.

## What is missing, with rough effort (expert person-months; treat as +-2x)
| Area | What | Effort |
|---|---|---|
| Kernel | universe polymorphism or cumulativity (see the status log: checked instantiation); structures/records with eta; mutual/nested inductives; well-founded recursion; **Nat/Int literals with accelerated arithmetic** (our Nat is unary: 2024 is 2024 constructors, so number-theory problems with concrete numbers are out of reach) | 2-4 |
| Elaboration | implicit arguments, instance resolution (or a disciplined structure-passing style), coercions N->Z->Q->R, notation, have/show/obtain/rcases/use/constructor, usable error messages | 2-4 |
| Automation | omega (linear integer arithmetic), ring/ring_nf, linarith/nlinarith, norm_num, simp, decide, positivity, field_simp, a general proof search; geometry needs coordinates/complex numbers or a dedicated deductive engine | 6-12 |
| Library | Z, Q, R (construct, or axiomatise as a complete ordered field: cheap but a trust risk), divisibility/gcd/primes/modular arithmetic, finite sets and sums, sequences/limits, polynomials, then linear algebra, groups/rings, topology | 12-24 for a competition subset; several person-years for undergraduate breadth |
| Search / learning | premise selection, proof-search policy, state encodings (the probing protocol and fingerprint stage are the substrate). Without guided search expect textbook lemmas, few competition problems. AlphaProof-class results used large-scale RL on top of Lean + Mathlib and long compute per problem; combinatorics and geometry remain hardest | open-ended research |

## The big lever
Lean 4 / Mathlib kernel compatibility, importing Mathlib's exported terms. Needed in the kernel: universe level expressions, Prop with definitional proof irrelevance, structure eta and projections, K-like reduction for Eq, quotients,
nested/mutual inductives with generated recursors, Nat/String literals with GMP-style acceleration, let, reducibility hints. That is a well-defined project (roughly 3-6 person-months, high risk of a long tail) and it would bring
~200k theorems and the whole algebraic hierarchy as CHECKABLE TERMS, with the independent checker doubling as a lean4checker-style validator. Our own tactics (apply, rewrite, simp, ring, search) would then run over imported lemmas.

## My honest estimate
By engineering effort, we are perhaps 5-10% of the way to "most IMO problems, confidently and efficiently" and 2-4% of the way to broad undergraduate mathematics. The percentages are low because the library and the automation are large;
the part that is hard to retrofit (a trustworthy kernel and tactic layer with an independent checker and tests that find real bugs) is in good shape. With the Mathlib-import route I would guess 1.5-3 further years of focused work for the
engineering, and the search/learning part could dominate beyond that; without it, more than 5. The uncertainty is a factor of 2-3 either way. None of this is a measurement.

## Calibrated plan (2026-10-10; estimates at the pace of Oct 5-10, uncertainty about 2x)
State: trusted core done (hybrid kernel, independent checker, 26 gates, 352 mutants); authoring layer with intro/apply/exact/rewrite/cases/induct/rw/have/obtain and a kernel-checked `ring` for unary Nat; library = 125 Nat
lemmas; the RL surface (probing, library index) is on the hybrid kernel, the gym environment / search / model are still on the old kernel.

| Block | Contents | Time |
|---|---|---|
| 1. Number tower and arithmetic automation | binary N (Pos/N, Peano induction derived as a theorem), Z (quotient of N x N), Q, R as ONE audited axiomatic complete ordered field; structures with a generic `ring`; linarith/nlinarith as kernel-checked certificates; omega-lite and norm_num | ~1 week |
| 2. Number theory / algebra vocabulary | divisibility, gcd/Bezout, primes, modular arithmetic, sums and products, floor/abs/sqrt/min/max, strong induction and descent, basic polynomials | ~1 week |
| 3. Gym environment on the hybrid kernel | observation/action surface, search and model code, curriculum Peano -> algebra -> number theory; can overlap with block 2 | ~1 week |
| 4. Competition bench | formalise ~100 olympiad problems, write reference proofs to find library/automation gaps; (miniF2F-style AMC/AIME first, then early IMO P1, then shortlist A1-A3 / N1-N3) | 1-3 months |

Scope: algebra and number theory only. About half of all IMO problems are geometry or combinatorics; geometry through coordinates/complex numbers (polynomial arithmetic + Groebner-style certificates) is possible for some
problems and is a separate project, combinatorics needs finite structures and is long-term. An autonomous learned agent that solves real IMO problems is a research outcome, not an engineering milestone; realistic first targets
are AMC/AIME-level and early-IMO P1-level problems.

Decisions: R is axiomatised as a single ordered-field structure instance (audit by constructing Q as a model of every non-completeness axiom); Q and Z are constructed; the Mathlib-import route is not pursued now.

## Suggested order
1. Nat/Int literals with accelerated arithmetic and `decide`/`norm_num` on them (self-contained, differential-testable against Python integers).
2. Universe polymorphism (or cumulativity) and propositional proof irrelevance (so Eq works on proofs, structures and subtypes become usable).
3. Structures/records, implicit arguments, a minimal instance mechanism.
4. omega, ring, linarith (certificates checked by the kernel).
5. Decide: build the library ourselves, or invest in Lean-kernel compatibility and import Mathlib.

## Status log (kernel features, newest first)
- **Definitional proof irrelevance** (gate S17, mutants M101-M113 and C1-C8). In the Prop/Type world two proofs of one proposition are convertible. The kernel decides it in the unifier: the structural comparison runs first;
  only a pair it rejects is a candidate, and then both sides must infer (by the ordinary inference rule) to types whose sort is Type0, and the two types must unify. A syntactic prefilter only skips inference for data
  and type formers. Partial effects of the rejected structural attempt (metavariable assignments, deferred constraints) are rolled back first. The independent checker implements the same rule with a conversion that
  carries the typing context. `conv_value` stays untyped and does no irrelevance (proofs inside a stuck match's captured environment are compared structurally: incomplete, never unsound). Without prop_elim there is no Prop
  and no irrelevance. Consequences: subtypes and structures with proof fields have the expected equality, a type may depend on a proof, and UIP / K are provable by `refl`.
- **Quotients** (gate S16): five kernel-built constants plus one reduction rule; funext is derivable from Quot.sound; the independent checker validates the roles and does not trust the labels.
- **Universe polymorphism: decision.** The kernel keeps concrete universes. A library declaration that is polymorphic is a construction function in the library layer, instantiated (and kernel-checked) once per level vector it is
  used at; no level variable ever reaches the kernel or the independent checker, so the trusted base does not grow. The competition and undergraduate libraries live in one data universe (Type1) plus Prop; Type2 is needed only for
  structure types (Group, Ring, ...). Real level expressions (Lean style) are the price of Mathlib import and can be added later without touching the rest.
