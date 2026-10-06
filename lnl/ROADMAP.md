# Roadmap: from this kernel to IMO-level and undergraduate mathematics (a judgment, not a measurement)

## Where we are (measured)
A small, heavily tested trusted core and tactic layer: dependent types, inductive families with parameters and indices, structural recursion, a Prop/Type world with restricted elimination and the four classical axioms, tactics
(intro, apply, rewrite, cases, induct with equation generalisation, strong induction by bounded predicates, generated injectivity lemmas), an independent checker, differential gates, zero-tolerance random oracles and soundness
canaries. The library is: Nat, Eq, Le, add, And/Or/Ex/True/False/Not/Iff, a dozen lemmas.

## What is missing, with rough effort (expert person-months; treat as +-2x)
| Area | What | Effort |
|---|---|---|
| Kernel | universe polymorphism or cumulativity; proof irrelevance; structures/records with eta; quotients (partly there); mutual/nested inductives; well-founded recursion; **Nat/Int literals with accelerated arithmetic** (our Nat is unary: 2024 is 2024 constructors, so number-theory problems with concrete numbers are out of reach) | 2-4 |
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

## Suggested order
1. Nat/Int literals with accelerated arithmetic and `decide`/`norm_num` on them (self-contained, differential-testable against Python integers).
2. Universe polymorphism (or cumulativity) and propositional proof irrelevance (so Eq works on proofs, structures and subtypes become usable).
3. Structures/records, implicit arguments, a minimal instance mechanism.
4. omega, ring, linarith (certificates checked by the kernel).
5. Decide: build the library ourselves, or invest in Lean-kernel compatibility and import Mathlib.
