# Roadmap: from this kernel to IMO-level and undergraduate mathematics (a judgment, not a measurement)

## Where we are (measured, 2026-10-12)
A small, heavily tested trusted core and tactic layer: dependent types, inductive families with parameters and indices, structural recursion, a Prop/Type world with definitional proof irrelevance and restricted elimination,
the four classical axioms, quotients (five kernel constants and one reduction rule; `funext` is a theorem), tactics (intro, apply, rewrite, cases, induct with equation generalisation, strong induction by bounded
predicates, generated injectivity lemmas; in the authoring layer exact/rw/rw_at/have/obtain/calc/induction_on/ring/linarith/nlinarith), an independent checker, differential gates against the frozen reference, zero-tolerance random
oracles, soundness canaries, and a deterministic work budget (a wrong action fails after a bounded number of steps instead of hanging). 46 gates, 1615 mutants defined; the whole suite is also run a second time with
`ITP_VERIFY_MEMO=1`, which re-decides every answer a sharing table or the quick conversion check gave.
Libraries, all built by tactic scripts and re-checked by the kernel (they use NO axioms; the four classical ones are an optional layer, S15): unary Nat (about 125 lemmas), the binary naturals `N` (Coq-style `Pos`/`N`, with the bridge to Nat and
the induction principle as a theorem), the integers `Z` as a quotient of `N x N` (ring laws, order, strict order, trichotomy, discreteness, integral domain, casts from N; 434 globals, about 25 s to build), a kernel-checked `ring` for `Nat`, `N` and `Z`,
`linarith` / `nlinarith` over `Z` (exact simplex search, certificate checked by the kernel), `linear_combination`, `interval_cases` and `by_contra`; a number-theory library over Z (divisibility, division with
remainder, Bezout, gcd, coprimality, primes, Euclid's lemma, every integer > 1 has a prime divisor; 46 theorems, one classical axiom `em`) and three olympiad statements proved (IMO 1959 P1 `gcd(21n+4, 14n+3) = 1`, IMO 2019 P1
`f(2a) + 2f(b) = f(f(a+b))` iff `f(n) = 2n + c`, the irrationality of sqrt 2).
Also built since (see the status log): the learner's environment with actions that SAY their term arguments (a typed term builder, claims, induction with a chosen principle, injectivity, forward reasoning), a recorder that replays the library's proofs as
action sequences (398 of 400), and a problem bank of formal olympiad problems with checked faithfulness (82 problems, nine of them IMO problems, levels 1-5) and dataset v1 on it (reference proofs in two regimes, a corpus of 398 library lemma proofs, leak-resistant splits: 56 train / 10 dev / 16 test).
Not yet: Q and R, a congruence calculus (`modeq`; squares mod 4 / 8 are proved case by case on remainders in the bank), `omega`/`positivity`/`norm_num`, sums over finite ranges, a trained policy (the first ones are the next step). Checking a big `ring`/`linarith` proof cost 0.1-0.5 s (finding F36); after the conversion work of 2026-10-12 (quick check on syntax, three sharing tables, Pi exposure) the final check of the large `nlinarith` proof is 88 thousand
steps instead of 1.98 million and the library build is 16% faster. Conversion of exponentially shared values is polynomial (F37, F38).

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

Progress against block 1 (2026-10-11): done -- binary N with its bridge, Z (quotient; ring laws, order, strict order, trichotomy, discreteness, integral domain, `eq_dec`, casts from N), `ring` over Nat, N and Z, `linarith` and
`nlinarith` over Z as kernel-checked certificates (gate S31), the typing memo that made big `ring` proofs 10-15x faster, the work budget. Next in block 1: a generic ordered-ring layer (one template instantiated for Z, Q, R),
Q as a quotient, R as the axiomatic ordered field validated on Q, `ring`/`linarith` instances for them, then `omega`-lite for N/Z, `positivity`, `norm_num`. Blocks 2-4 have not been started.

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
- **The bank grows to 82 problems; dataset v1** (gate S43; `bank_lib2.py`, `bank_all.py`, `bank_split.py`, `lib_corpus.py`, `dataset.py`; BANK.md sections 6-7b; findings F62-F67). 46 gates pass, in both modes (ordinary pass 748 s, `ITP_VERIFY_MEMO=1` 1299 s on the 2-core box).
  * S43 (500 s): the 54 problems added to the 28 of S41 -- levels 1-5, nine IMO problems in all (1959 P1, 1964 P1 (a), (b) and P2, 1969 P1, 1982 P1, 1984 P2, 1988 P6 by Vieta jumping, 2019 P1) -- each with its statement pinned by digest, a faithfulness
    check against an independent Python spec with negative controls (statement mutants: 687 over the first 28, 1415 over the 54; each caught or explained by a `skip_mutants` entry with its reason), a reference proof that replays as kernel-checked
    actions in the prefix regime and, where it proves its lemmas inline, in the strict regime, and the independent checker's verdict (all but 7 slow case enumerations: F67).
  * The machinery the new problems needed: `induction_on` with the principle's extra arguments (the starting point of `Z.induction_from`, strong induction) and INDUCT_WITH for it (F62); retired outer-goal hypotheses are not extras (F63); a hint that is the
    goal or a hypothesis (F64); `Spec(hatoms=..)` for the atoms of hypotheses (F65).
  * The DATASET (`dataset.py` writes it outside the repository: bank.json, library.json, splits.json, DATASET_CARD.md, manifest.json with digests): 82 reference proofs (1816 actions), 8 strict-regime proofs (546 actions), the proofs of 398 library lemmas (1978 actions),
    and the SPLITS: by family, a frozen hash (salt 25), 56 train / 10 dev / 16 test, the test split with three IMO problems and level-4 problems, the train split with the level-5 problem (F66).
- **The action language, the typed term builder, the recorder and the problem bank** (gates S37-S42; `gym_terms.py`, `gym_build.py`, `gym_acts.py`, `recorder.py`, `bank.py`, `bank_lib.py`, `semantic.py`, `bank_baseline.py`,
  `gym_policy.py`; ACTIONS.md, BANK.md; findings F51-F61; mutant series GA / GB / GE / GF / GL / GQ / GR / GS / GT / GX, 1615 mutants in all). 45 gates pass, in both modes (ordinary pass 677 s, `ITP_VERIFY_MEMO=1` 738 s on the 2-core box).
  * S37: `ring`, `linarith` and `linear_combination` see through the arithmetic in the arguments of an atom (`f (a + b)` and `f (b + a)` are one unknown; congruence proofs checked by the kernel).
  * S38: a term pool and the actions that carry terms (INSTANTIATE, USE), probes under a chosen work budget, goal identity up to irrelevant order, a cost search (IDA* with the policy's cost as a hook).
  * S39: a comparison that FAILS is decided by what it found: a refusal at data over a quotient type (`Z.add a b` against `Z.add a a`) costs a few hundred steps instead of 200,000 to 1,600,000; the retry by full normalisation runs
    only when the failure was uncertain (a case split, a fixpoint, the nesting bound).
  * S40 (587 checks): a second action language whose arguments are SAID, one typed choice at a time, by a builder that offers only the choices that keep the term well typed (CLAIM, REFINE, APPLY_WITH, REWRITE_WITH, REWRITE_AT, FORWARD,
    INDUCT_WITH with a chosen principle and starting point, INJECT, LINCOMB, USE with a code); actions as plain JSON; the recorder turns the library's authored proofs into kernel-checked action sequences (398 of 400, 1978 actions, each
    replayed again from JSON in a fresh engine and re-checked by the independent checker). Measured: world build 2.3-3 s (491 globals), a menu 6 ms, a probe 0.9 ms, menus of 55 actions on average (172 at most).
  * S41 (127 checks): the problem bank. 28 problems of levels 1-4 (identities, inequalities, divisibility and remainders, IMO 1959 P1, 1964 P1a/b, 1969 P1, 1984 P2, 2019 P1, the irrationality of sqrt 2), each with the problem in words,
    a SPEC written independently as Python (statements are checked to mean what the words say: windows, atoms, bodies of existentials, 687 mutants of the statements that must all be noticed or explained), a reference proof that replays
    through the action language in two regimes (prefix; strict, which hides the lemmas written for the problem and needs a self-contained proof), a JSON export with demonstrations, and every statement pinned by digest. The blind search
    (iterative deepening over the menus, no learning) solves 13 of the 56 (problem, regime) rows at 3000 probes: the identities and `sum_squares_zero`, and the prefix regime of `imo1964_p1a` where the helper is the statement.
  * S42 (245 checks): pins of the builder and the actions at their limits (shapes, sources, the matcher, the table of heads), written against the first survivors of the mutation run.
  * `gym_policy.py`: an action assembled from small typed choices (kind, candidate, hypothesis, term tokens, ...) for a policy without a text interface; the round trip over the recorded library proofs rebuilds 1962 of 1978 actions
    (the 16 others are the open item F61).
  * Mutation work on the new modules is DEFERRED by decision (it measures completeness of the tests, not the soundness of anything: every term and proof is kernel-checked), until the first trained agents exist. State: run A of the
    series (GB / GE, 216 mutants): 148 caught, 68 survivors; partial run B (GQ1-GQ174): 95 caught, 79 survivors; S42 was written against them and the runs were not repeated. Survivors still to triage (a test, or `KNOWN_EQUIVALENT` with a
    reason): run A: GB3 GB7 GB8 GB10 GB15 GB17-GB21 GB25 GB30 GB31 GB34 GB35 GB37 GB53 GB54 GB56 GB58 GB63-GB66 GB71 GB73 GB74 GB76-GB78 GB81-GB88 GB90 GB92 GB109-GB111 GB128 GB130 GB131 GB139-GB141 GB147 GB153 GB155 GB156 GB158
    GB159 GB165 GB170 GB173 GE67 GE72 GE77 GE85 GE86 GE89 GE97 GE99 GE103 GE104; run B (so far): the GQ mutants of the shape checks, principles, slots and the portable readers (GQ1-GQ174 minus the caught ones, see `logs/mut_s41_B.log`);
    the series has not been run to completion on the final tree (runs A and B are the only runs on snapshot 22; the others, e.g. GQ175-GQ193, GR and GT, are defined and unmeasured there). S42 sections E-H (the menu of the
    builder, principles, slots, a spent budget) are planned for that triage.
  * Next: the rest of the bank (26 more problems are written in `bank_lib2.py`, not yet part of the gated bank), the Gymnasium environment over the bank, imitation of the reference and library proofs, expert iteration against the
    blind search at equal probe budgets.
- **The learner's environment on the hybrid kernel, and falsification fuzzing** (gate S36, `gym_env.py`, `gym_search.py`, `gym_fuzz.py`; findings F45-F50; mutants CP / UF / CX / CL / LR / TL / GE / GS). Goals are DATA (a target and
  the hypothesis types, kernel handles; two routes to the same goal give the same object), a menu lists the legal actions (intro, cases, induction, apply and rewrite with library lemmas and hypotheses, ring, linarith, nlinarith,
  by_contra, interval_cases), and a PROBE tries an action atomically (checkpoint, one-goal proof state, action, read the goals, rewind). Nothing in the environment is trusted: a search returns a tree of actions, `commit` replays it
  on a live proof and the kernel checks the term. The search is an AND-OR depth-first search with iterative deepening. Results: 72 of 100 natural-number lemmas without ring and 263 of 401 with the full world are re-proved by
  search from the library that precedes them; the integer problems of S36 H too. `gym_fuzz.py` generates random true and false statements (naturals, integers), runs the search and reports REJECTED (a finished proof the kernel refuses),
  UNSOUND (a proof of a false statement) or ERROR: it found the unifier bug F45 and (earlier) the `cases` bug F46; it also showed how a fuzz can be empty (F47: the first integer run proved 22 of 109 true statements because a
  Python TypeError was counted as a refused action; strict engines and canaries now prevent that). The fixed fuzz: 0 problems in 6399 natural-number statements and in 150 integer ones (103 of the 109 true statements proved, the 41 false ones not). Checkpoints are validated (F48). 39 gates,
  933 mutants defined (the new series run against S36 alone where only it reads the file).
- **Number theory over Z, `linear_combination`, olympiad problems** (gate S34, `nt_lib.py`, `imo_lib.py`, `lincomb.py`; findings F44; mutants LC / RQ / LS / EP / NP / IV / BC). Divisibility `Z.dvd a b := exists k, b = a * k` with its algebra,
  division with remainder (existence and uniqueness), Bezout, `Z.is_gcd`, `Z.coprime`, `Z.prime`, Euclid's lemma, every integer above 1 has a prime divisor (46 theorems; the one classical axiom `em` is used by four of them:
  `by_contra`, `not_forall`, `not_imp`, `Z.exists_prime_dvd`). `linear_combination c1 h1 ... cn hn` closes an equation of a commutative ring that is the combination of the hypotheses (the proof is built from the ring's own
  congruence lemmas and a `ring` identity, and the cancellation lemma whose argument order is read off its statement); `interval_cases` splits an integer between two bounds into one goal per value (the bounds are proved by
  `linarith`); `by_contra`. IMO 1959 P1, IMO 2019 P1 (an iff) and the irrationality of sqrt 2 are proved, and the gate checks that each statement says what it should (negative controls: nearby statements are refuted, perturbed
  scripts are refused). The gate's randomised oracle generates 45 polynomial identities per type and requires `linear_combination` to accept exactly those the independent polynomial oracle says hold; it found four robustness
  bugs in the tactic layer in its first run (F44).
- **A `match` is checked on syntax** (gate S35, findings F42, F43; mutants MS1-MS17, P2, P3, M88, M89). The family of a scrutinee's type is read by unfolding definitions at its head one step at a time (at most 32), the type of the
  match and of its branches is the motive's body with the indices and the scrutinee substituted, and evaluation is only the fallback. Kernel steps are deterministic, so the gate pins the cost: `cases` on `h : Q big` (`big`
  computes to 16384 successors) takes about 200 steps instead of 200,000, and a proof with a hypothesis that depends on the scrutinee 384 steps instead of 2,556,726. Raw-match tests cover the 32-unfolding bound, zero-branch
  matches on an axiom or a stuck recursive type (refused), the order in which the motive receives the indices and the scrutinee (the induction principle of `Le` as a raw match), and a motive that is a global.
- **Conversion that scales** (gates S32, S33; findings F36-F39; mutants FX1-4, SC1-26, CM1-13, UM1-23, QM1-15). Four steps, each measured. (1) A Pi hidden behind a definition (`Not P`, `AllLe a`) is exposed on syntax, so the
  evaluator is not asked to normalise an expected type just to see a binder. (2) `SynConv`: a quick check on syntax in front of the evaluator that can only answer YES (identical nodes, congruence, binders, delta of the later-declared
  head, beta; budget 4000 steps; off with metavariables and inside the tactic scopes). (3) Sharing tables: values built from nested definitions share structure (`plus2 (plus2 (.. x))` holds the level below twice), and a
  frame-by-frame comparison is exponential in the depth (4.3 billion steps at depth 30); `ConvMemo`, `UnifyMemo` and `QuoteMemo` remember the pairs found equal and the syntax of stuck values, keyed by the identity of the
  shared blocks, and a table entry never outlives an undo of a metavariable solution (`MetaContext::undo_epoch`). Depth 100 now costs 24 thousand steps (conversion) / 744 thousand (unifier, still cubic: each stuck match opens its own
  table); a refused comparison of depth 100 costs 27 thousand. (4) `verify_memo` / `ITP_VERIFY_MEMO=1` re-decides every table hit and every quick-check yes with the tables bypassed and throws on a mismatch, so a table can
  make the kernel faster but not different: the gates run in both modes. Open: one table per unifier call instead of per stuck match (cubic to quadratic), closure comparison without quoting the domain, canonical syntax for stuck
  matches (F28 root cause), a memo on `eval` for DAG-shaped terms.
- **`linarith` / `nlinarith` over Z** (gate S31, `linarith.py`, mutants LA1-LA53 except LA38). Hypotheses and the negated goal become facts `0 <= p_i` (a proof-free polynomial reader shares the atoms of `ring`); an exact integer-preserving
  simplex (Edmonds fraction-free, Bland's rule; SciPy HiGHS only guesses a support for very large systems and its answer is always re-solved exactly) finds `sum m_i p_i = -k`, k > 0. The certificate is `Z.mul_nonneg`/`Z.add_nonneg`, a
  `ring` proof of `S = -k`, and `Z.not_nonneg_neg`; the whole proof is one `exact`, so a failed search leaves the proof state untouched. Untrusted search, kernel-checked result: a wrong certificate (injected by a mutant) is reported as an
  internal error, never accepted. `nlinarith` adds squares of the squared subterms and all pairwise products of facts (cap 600) and takes hint terms. Like Mathlib, `a^2 + b^2 >= 2ab` needs the hint `sq_nonneg (a - b)`.
- **Integer order library and robustness** (gate S30, S22 section L; mutants Z*, TL1-TL2). `Z.le x y := exists k:N, x + of_N k = y`; strict order, `mul_pos`, `mul_eq_zero`, cancellation, casts. A tactic that is called with no goal
  raised a raw C++ `IndexError`; every goal-indexed tactic now checks the goal first (finding F35).
- **`apply` on Pi goals** (gate S29, finding F32, mutants AR1-AR13). A goal that is an implication, a negation (`Not P` is `P -> False`) or a universal statement is met by a lemma that concludes such a Pi: the lemma's binders
  are peeled only as far as they exceed the goal's (a binder hidden behind a definition counts as one), and if that does not unify every binder is peeled, as before. It is the tactic layer's mode (`simp`) only: without it `apply`
  is the reference's, which the lockstep gates S3-S5 compare (the first full run after the change broke exactly those four gates, which is how the boundary was found).
- **Work budget** (gate S27, finding F30, mutants FU1-FU13). Every guarded recursion spends a step from a deterministic thread-local budget, set at each outermost Python-to-C++ call; exhaustion throws `FuelExhausted` and the
  budget stays spent for the rest of the call. `TacticTxn` makes the seven kernel tactics all-or-nothing against exceptions. Motivated by F29 (conversion is exponential on shared terms; a mistyped `apply` ran for minutes).
- **Head unfolding on syntax** (gate S28, finding F31, mutants U1-U9). `unfold_head_folded` is one delta step on syntax; `apply` and `cases` read a definition at the head of a goal or hypothesis through it, so the arguments stay as written.
- **Typing memo, node facts, shared-term traversals** (gate S26, mutants W/X series). Every node carries facts computed once (loose bound-variable range, context-variable range, tree size, mentions a metavariable); substitutions
  are guarded by them, big traversals are memoised per (node, binder depth), `infer` remembers the type of a large metavariable-free term per context, and proof extraction is linear in the DAG when nothing can reduce. 10-15x on big `ring`
  proofs. TRUSTED code, so the gate checks each piece against an oracle that shares no code with it. First mutation run of the series: 119 mutants, 95 caught, 2 documented equivalent, 22 survivors still to be triaged
  (stack-check removals G5-G9, W61, W63, typing-memo keys X2/X5-X9/X11-X15/X17/X19/X21, X27).
- **Integers, `ring` over N and Z** (gate S25, `int_lib.py`, `quot_lib.py`, `ring_lib.py`). `Z := Quot (N x N) Zrel`; addition, multiplication and negation by `Quot.lift`/`Quot.lift2`, computing on `Z.mk`; every law by quotient induction
  down to an equation of N that `ring` proves; effectiveness of the quotient from a Bool-valued test (no propext). `ring` is trace-based and untrusted: the finished term is checked by the kernel, and one lemma template is
  instantiated for Nat, N and Z. `funext` is derived from `Quot.sound` (S16), not assumed.
- **Definitional proof irrelevance** (gate S17, mutants M101-M113 and C1-C8). In the Prop/Type world two proofs of one proposition are convertible. The kernel decides it in the unifier: the structural comparison runs first;
  only a pair it rejects is a candidate, and then both sides must infer (by the ordinary inference rule) to types whose sort is Type0, and the two types must unify. A syntactic prefilter only skips inference for data
  and type formers. Partial effects of the rejected structural attempt (metavariable assignments, deferred constraints) are rolled back first. The independent checker implements the same rule with a conversion that
  carries the typing context. `conv_value` stays untyped and does no irrelevance (proofs inside a stuck match's captured environment are compared structurally: incomplete, never unsound). Without prop_elim there is no Prop
  and no irrelevance. Consequences: subtypes and structures with proof fields have the expected equality, a type may depend on a proof, and UIP / K are provable by `refl`.
- **Quotients** (gate S16): five kernel-built constants plus one reduction rule; funext is derivable from Quot.sound; the independent checker validates the roles and does not trust the labels.
- **Universe polymorphism: decision.** The kernel keeps concrete universes. A library declaration that is polymorphic is a construction function in the library layer, instantiated (and kernel-checked) once per level vector it is
  used at; no level variable ever reaches the kernel or the independent checker, so the trusted base does not grow. The competition and undergraduate libraries live in one data universe (Type1) plus Prop; Type2 is needed only for
  structure types (Group, Ring, ...). Real level expressions (Lean style) are the price of Mathlib import and can be added later without touching the rest.
