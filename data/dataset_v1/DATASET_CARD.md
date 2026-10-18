# Olympiad problem bank -- dataset version 1

Formal olympiad problems over the integers and naturals, stated in the hybrid kernel, each with a **kernel-checked reference proof written as a sequence of actions** of the gym action language (no text interface: the learner sees a goal and chooses actions).  82 problems, 8 of them with a second, self-contained proof for the *strict* regime.

## Contents

| level | problems | meaning |
|---|---|---|
| 1 | 18 | a single arithmetic step (ring / linarith / nlinarith) after the introductions |
| 2 | 32 | a short proof with one idea: a case split, a witness, a divisibility computation, one induction |
| 3 | 26 | an olympiad lemma: several ideas or several lemmas (15 - 60 actions) |
| 4 | 5 | an olympiad problem with a short solution (60 - 200 actions) |
| 5 | 1 | a hard olympiad problem |

kinds: answer 9, exists 2, prove 71.  sources: IMO 9, folklore 71, other 2.

topics: algebra 12, bounding 3, case-analysis 5, construction 4, descent 4, diophantine 10, divisibility 21, functional-equation 6, gcd 6, identity 11, induction 17, inequality 23, logic 1, modular 19, number-theory 43, parity 6, prime 3.

IMO problems (9): imo1959_p1 (IMO 1959 P1), imo1964_p1a (IMO 1964 P1 (a)), imo1964_p1b (IMO 1964 P1 (b)), imo1964_p2 (IMO 1964 P2), imo1969_p1 (IMO 1969 P1), imo1982_p1 (IMO 1982 P1), imo1984_p2 (IMO 1984 P2), imo1988_p6 (IMO 1988 P6), imo2019_p1 (IMO 2019 P1).

## Splits

Splits are by **family** (the two parts of one olympiad problem, a statement in other clothes, an argument with another constant stay together) and are a pure function of the family key (`bank_split.py`, salt number 25, fractions train 70%, dev 10%, test 20%): a later version of the bank never moves an existing problem from one split to another.

| split | problems | by level | IMO | demonstration steps |
|---|---|---|---|---|
| train | 56 | L1:14 L2:25 L3:13 L4:3 L5:1 | 6 | 1205 |
| dev | 10 | L1:2 L2:3 L3:5 | 0 | 293 |
| test | 16 | L1:2 L2:4 L3:8 L4:2 | 3 | 318 |

* **train**: `imo1959_p1`, `sqrt2_irrational`, `ident_sophie_germain`, `ident_consecutive_four`, `ident_fourth_powers`, `ineq_am_gm_two`, `ineq_square_sum_three`, `ineq_four_ab`, `ineq_cauchy_two`, `sum_squares_zero`, `odd_square_mod_eight`, `cube_mod_nine`, `cube_minus_n_div6`, `pow_sub_dvd`, `no_square_between`, `N.induction3`, `imo1984_p2`, `imo1969_p1`, `ident_cube_sum`, `ident_cube_diff`, `ident_three_cubes`, `ident_pell_product`, `ineq_three_squares`, `ineq_shifted_squares`, `ineq_cubes_two`, `ineq_cubes_three`, `ineq_pairs_eight`, `ineq_fourth_powers_three`, `ineq_four_fourth_powers`, `quad_form_zero`, `square_mod_three`, `consecutive_product_even`, `five_dvd_fifth_power`, `fourth_power_odd_16`, `diff_squares_not_2_mod_4`, `cube_sum_not_4_mod_9`, `gcd_shift`, `gcd_sum_product`, `no_square_pronic`, `five_dvd_six_pow_minus_one`, `three_dvd_four_pow_plus_five`, `fe_additive`, `no_f_f_succ`, `sqrt3_irrational`, `three_squares_mod_eight`, `schur_nonneg`, `ineq_cauchy_three`, `ineq_chebyshev_cubes`, `fe_f_x_plus_f_y`, `imo1982_p1`, `imo1988_p6`, `n5_minus_n_div30`, `imo1964_p2`, `fe_quadratic`, `ineq_a_plus_b_cubed`, `ineq_square_products`
* **dev**: `ident_diff_squares`, `not_dvd_between`, `square_mod_four`, `three_dvd_sum_squares`, `pythag_three`, `square_not_prime`, `two_pow_gt_n`, `bernoulli_int`, `find_all_x2_eq_x`, `pell_infinitely_many`
* **test**: `imo2019_p1`, `ident_brahmagupta`, `sum_two_squares_mod4`, `coprime_consecutive`, `imo1964_p1b`, `imo1964_p1a`, `ineq_cross_terms`, `three_consecutive_div3`, `dvd_coprime_product`, `three_consecutive_not_squarefree`, `find_all_xy_eq_x_plus_y`, `find_all_n_plus_1_dvd`, `frobenius_3_5`, `x2_y2_3z2`, `no_x2_minus_2y2_eq_3`, `n4_n2_1_composite`

## Regimes

A problem is attacked in the world in which the library precedes it.  In the **prefix** regime every global before the theorem is a premise (the lemmas written for the problem included); in the **strict** regime the lemmas written for the problem are hidden, so the agent must find the ideas.  `premises.prefix` / `premises.strict` of a problem is the number of globals the engine offers.

## Reference proofs

82 demonstrations, 1816 actions in all (shortest 1, longest 197).  Every one replays through the engine from its JSON in a fresh engine over a freshly built world, and the proof term it builds is checked by the kernel (and, for the proofs that are not too large, by the independent checker).

actions by kind: LINARITH 451, CASES 297, FORWARD 198, CLAIM 177, USE 159, APPLY_WITH 94, APPLY 82, REWRITE 78, RING 61, REWRITE_AT 54, LINCOMB 51, NLINARITH 41, REFINE 40, INTERVAL_CASES 19, INDUCT_WITH 14.

## Faithfulness of the statements

A kernel-checked proof proves the *formal* statement, not the problem.  Every statement is therefore read back into an AST and compared with an independent description of what the problem says (`semantic.Spec`: Python predicates, the numbers in the relation atoms, the body of existentials) on a box of values, and deliberately mutated statements (changed operators, constants, quantifiers, guards) must be told from it; a mutant that means the same is listed with its reason.  See `BANK.md`.  Inequalities that are over the reals in the original are stated for integers (noted in the problem).

## Library corpus

`library.json`: the proofs of the 398 lemmas of the base library (1978 actions) as action sequences, split by lemma name (train 282, dev 38, test 78).  3 lemma(s) have the statement of a bank problem and are marked `bank_duplicate`.

## Files

`bank.json` (problems and demonstrations), `splits.json`, `library.json`, `manifest.json` (digests).  World digest of the bank: `8eaef730fd209a69f35a75ddaa4858997acc54b40a801dbea6036427052222ca`.
