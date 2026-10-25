# Olympiad problem bank -- dataset version 2

Formal olympiad problems over the integers and naturals, stated in the hybrid kernel (no text interface: the learner sees a goal and chooses actions).  268 problems in two tiers: **187 proved** -- each with a **kernel-checked reference proof written as a sequence of actions** of the gym action language (8 of them with a second, self-contained proof for the *strict* regime) -- and **81 statement-only** -- the formal statement and its checked meaning, without a proof (a task all the same: whatever an agent proves is checked by the kernel).

## Contents

| level | problems | proved | meaning |
|---|---|---|---|
| 1 | 37 | 37 | a single arithmetic step (ring / linarith / nlinarith) after the introductions |
| 2 | 67 | 67 | a short proof with one idea: a case split, a witness, a divisibility computation, one induction |
| 3 | 75 | 68 | an olympiad lemma: several ideas or several lemmas (15 - 60 actions) |
| 4 | 45 | 14 | an olympiad problem with a short solution (60 - 200 actions) |
| 5 | 44 | 1 | a hard olympiad problem |

kinds: answer 121, exists 3, prove 144.  sources: AIME 31, AMC 72, IMO 41, ISL 18, folklore 71, other 35.

origin: adapted 19, contest 72, folklore 105, style 72 (contest: a problem of a named contest, paraphrased; adapted: a named problem changed to fit the vocabulary; style: written for the bank in the manner of a contest; folklore: a classical fact).  Source check: answer-checked 114, none 145, recalled 9 (answer-checked: compared with a public source -- the published answer of a "determine" problem (the formal answer is also found by brute force), the published statement of a "prove" problem; recalled: the source and the wording are from memory and could not be compared with the official statement; none: no check applies, the problem was written for the bank or is a classical fact).

topics: algebra 55, arithmetic 17, bounding 25, case-analysis 31, construction 9, descent 8, digits 10, diophantine 64, divisibility 68, functional-equation 23, gcd 15, identity 21, induction 29, inequality 35, modular 32, number-theory 163, parity 15, polynomial 2, prime 33, product 5, sequence 7, sum 7.

IMO problems (41): imo1959_p1 (IMO 1959 P1), imo1960_p1 (IMO 1960 P1), imo1962_p1 (IMO 1962 P1), imo1964_p1a (IMO 1964 P1 (a)), imo1964_p1b (IMO 1964 P1 (b)), imo1964_p2 (IMO 1964 P2), imo1969_p1 (IMO 1969 P1), imo1972_p3 (IMO 1972 P3), imo1977_p5 (IMO 1977 P5), imo1977_p6 (IMO 1977 P6), imo1978_p1 (IMO 1978 P1), imo1981_p3 (IMO 1981 P3), imo1982_p1 (IMO 1982 P1), imo1982_p4a (IMO 1982 P4 (a)), imo1982_p4b (IMO 1982 P4 (b)), imo1983_p6_integer (IMO 1983 P6), imo1984_p2 (IMO 1984 P2), imo1984_p6 (IMO 1984 P6), imo1986_p1 (IMO 1986 P1), imo1987_p4 (IMO 1987 P4), imo1988_p6 (IMO 1988 P6), imo1989_p5 (IMO 1989 P5), imo1990_p3 (IMO 1990 P3), imo1992_p1 (IMO 1992 P1), imo1994_p4 (IMO 1994 P4), imo1996_p4 (IMO 1996 P4), imo1997_p5 (IMO 1997 P5), imo1998_p4 (IMO 1998 P4), imo1999_p4 (IMO 1999 P4), imo2001_p6 (IMO 2001 P6), imo2003_p2 (IMO 2003 P2), imo2005_p4 (IMO 2005 P4), imo2006_p4 (IMO 2006 P4), imo2007_p5 (IMO 2007 P5), imo2009_p5 (IMO 2009 P5), imo2012_p4 (IMO 2012 P4), imo2015_p2 (IMO 2015 P2), imo2016_p4a (IMO 2016 P4 (a)), imo2016_p4b (IMO 2016 P4 (b)), imo2019_p1 (IMO 2019 P1), imo2022_p5 (IMO 2022 P5).

ISL problems (18): isl2000_n4 (ISL 2000 N4), isl2005_n3 (ISL 2005 N3), isl2006_n5 (ISL 2006 N5), isl2007_n1 (ISL 2007 N1), isl2007_n2 (ISL 2007 N2), isl2007_n6 (ISL 2007 N6), isl2008_n1 (ISL 2008 N1), isl2010_n2 (ISL 2010 N2), isl2012_n2 (ISL 2012 N2), isl2012_n4b (ISL 2012 N4 (b)), isl2012_n6 (ISL 2012 N6), isl2013_n1 (ISL 2013 N1), isl2014_n2 (ISL 2014 N2), isl2015_n2 (ISL 2015 N2), isl2016_n4 (ISL 2016 N4), isl2019_n2 (ISL 2019 N2), isl2021_n1 (ISL 2021 N1), isl2022_n1 (ISL 2022 N1).

AIME problems (31): aime1983_p6 (AIME 1983 P6), aime1984_p7 (AIME 1984 P7), aime1985_p13 (AIME 1985 P13), aime1985_p3 (AIME 1985 P3), aime1985_p5 (AIME 1985 P5), aime1985_p7 (AIME 1985 P7), aime1986_p1 (AIME 1986 P1), aime1986_p10 (AIME 1986 P10), aime1986_p4 (AIME 1986 P4), aime1986_p5 (AIME 1986 P5), aime1986_p6 (AIME 1986 P6), aime1987_p5 (AIME 1987 P5), aime1988_p8 (AIME 1988 P8), aime1989_p1 (AIME 1989 P1), aime1989_p3 (AIME 1989 P3), aime1989_p4 (AIME 1989 P4), aime1989_p7 (AIME 1989 P7), aime1989_p9 (AIME 1989 P9), aime1990_p15 (AIME 1990 P15), aime1990_p3 (AIME 1990 P3), aime1991_p1 (AIME 1991 P1), aime1991_p13 (AIME 1991 P13), aime1991_p8 (AIME 1991 P8), aime1992_p3 (AIME 1992 P3), aime1992_p8 (AIME 1992 P8), aime1993_p6 (AIME 1993 P6), aime1997_p3 (AIME 1997 P3), aime1998_p14 (AIME 1998 P14), aime1999_p3 (AIME 1999 P3), aime2002ii_p3 (AIME 2002 II P3), aime2009i_p1 (AIME 2009 I P1).

AMC problems (72, every one with the source "original, in the manner of the AMC"): amc_alg_01, amc_alg_02, amc_alg_03, amc_alg_04, amc_alg_05, amc_alg_06, amc_alg_07, amc_alg_08, amc_alg_09, amc_alg_10, amc_alg_11, amc_alg_12, amc_alg_13, amc_alg_14, amc_alg_15, amc_alg_16, amc_bound_01, amc_bound_02, amc_dig_01, amc_dig_02, amc_dig_03, amc_dio_01, amc_dio_02, amc_dio_03, amc_dio_04, amc_div_01, amc_div_02, amc_div_03, amc_div_04, amc_div_05, amc_div_06, amc_div_07, amc_div_08, amc_div_09, amc_div_10, amc_dvs_01, amc_dvs_02, amc_eq_01, amc_eq_02, amc_eq_03, amc_fe_01, amc_fe_02, amc_fe_03, amc_fe_04, amc_fe_05, amc_gcd_01, amc_ind_01, amc_ind_02, amc_ineq_01, amc_ineq_02, amc_ineq_03, amc_ineq_04, amc_ineq_05, amc_mod_01, amc_mod_02, amc_mod_03, amc_mod_04, amc_mod_05, amc_par_01, amc_prime_01, amc_prime_02, amc_prime_03, amc_prime_04, amc_prime_05, amc_seq_01, amc_seq_02, amc_seq_03, amc_seq_04, amc_sum_01, amc_sum_02, amc_sum_03, amc_sum_04.

## Splits

Splits are by **family** (the two parts of one olympiad problem, a statement in other clothes, an argument with another constant stay together) and are a pure function of the family key (`bank_split.py`, salt number 25, fractions train 70%, dev 10%, test 20%): a later version of the bank never moves an existing problem from one split to another.

| split | problems | proved | by level | IMO/ISL | AIME/AMC | demonstration steps |
|---|---|---|---|---|---|---|
| train | 190 | 132 | L1:27 L2:53 L3:43 L4:28 L5:39 | 43 | 75 | 4154 |
| dev | 20 | 15 | L1:5 L2:3 L3:9 L4:1 L5:2 | 3 | 9 | 822 |
| test | 58 | 40 | L1:5 L2:11 L3:23 L4:16 L5:3 | 13 | 19 | 1057 |

* **train**: `imo1959_p1`, `sqrt2_irrational`, `ident_sophie_germain`, `ident_consecutive_four`, `ident_fourth_powers`, `ineq_am_gm_two`, `ineq_square_sum_three`, `ineq_four_ab`, `ineq_cauchy_two`, `odd_square_mod_eight`, `pow_sub_dvd`, `no_square_between`, `imo1984_p2`, `imo1969_p1`, `ident_cube_sum`, `ident_cube_diff`, `ident_three_cubes`, `ineq_three_squares`, `ineq_shifted_squares`, `ineq_cubes_two`, `ineq_cubes_three`, `ineq_pairs_eight`, `ineq_fourth_powers_three`, `ineq_four_fourth_powers`, `quad_form_zero`, `consecutive_product_even`, `fourth_power_odd_16`, `diff_squares_not_2_mod_4`, `cube_sum_not_4_mod_9`, `gcd_shift`, `gcd_sum_product`, `no_square_pronic`, `five_dvd_six_pow_minus_one`, `three_dvd_four_pow_plus_five`, `fe_additive`, `no_f_f_succ`, `sqrt3_irrational`, `three_squares_mod_eight`, `schur_nonneg`, `ineq_cauchy_three`, `ineq_chebyshev_cubes`, `fe_f_x_plus_f_y`, `imo1982_p1`, `imo1988_p6`, `n5_minus_n_div30`, `imo1964_p2`, `fe_quadratic`, `ineq_a_plus_b_cubed`, `ineq_square_products`, `aime1985_p3`, `aime1987_p5`, `aime1986_p5`, `aime1999_p3`, `aime1989_p9`, `aime1990_p3`, `aime1989_p1`, `aime1992_p3`, `aime1991_p13`, `aime1986_p4`, `aime1983_p6`, `aime1986_p6`, `aime1984_p7`, `aime1985_p5`, `aime1992_p8`, `aime1998_p14`, `amc_alg_01`, `amc_alg_04`, `amc_alg_06`, `amc_alg_08`, `amc_alg_09`, `amc_alg_10`, `amc_alg_11`, `amc_div_01`, `amc_div_02`, `amc_div_03`, `amc_div_04`, `amc_div_05`, `amc_mod_01`, `amc_mod_03`, `amc_mod_04`, `amc_par_01`, `amc_prime_03`, `amc_prime_04`, `amc_prime_05`, `amc_ineq_01`, `amc_ineq_02`, `amc_ineq_04`, `amc_bound_01`, `amc_bound_02`, `amc_sum_01`, `amc_sum_02`, `amc_sum_03`, `amc_sum_04`, `amc_ind_01`, `amc_eq_01`, `amc_eq_02`, `amc_eq_03`, `amc_dio_02`, `amc_dio_03`, `amc_dvs_01`, `amc_div_07`, `amc_div_08`, `amc_div_09`, `amc_div_10`, `amc_dig_01`, `amc_dig_02`, `amc_mod_05`, `amc_alg_12`, `amc_alg_14`, `amc_alg_15`, `amc_alg_16`, `amc_fe_02`, `amc_fe_03`, `amc_fe_04`, `amc_fe_05`, `amc_seq_01`, `amc_seq_02`, `amc_seq_03`, `amc_seq_04`, `aime1986_p1`, `aime1990_p15`, `aime1991_p8`, `aime2002ii_p3`, `aime2009i_p1`, `imo1960_p1`, `imo1977_p5`, `imo1981_p3`, `imo1990_p3`, `imo1997_p5`, `imo1998_p4`, `imo1999_p4`, `imo2003_p2`, `imo2006_p4`, `imo2007_p5`, `imo2015_p2`, `imo2001_p6`, `imo1987_p4`, `imo1972_p3`, `imo1978_p1`, `imo1984_p6`, `imo1996_p4`, `imo2012_p4`, `imo2009_p5`, `isl2006_n5`, `isl2007_n6`, `isl2008_n1`, `isl2010_n2`, `isl2012_n2`, `isl2015_n2`, `isl2019_n2`, `isl2022_n1`, `composite_run`, `n2_3n_5_not_121`, `ineq_vasc`, `ineq_amgm_three`, `ineq_abc_2abc_1`, `imo1983_p6_integer`, `fe_ring_hom`, `imo1977_p6`, `imo2016_p4a`, `imo2016_p4b`, `isl2000_n4`, `isl2012_n4b`, `isl2012_n6`, `isl2013_n1`, `isl2014_n2`, `isl2021_n1`, `apmo2019_p1`, `classic_odd_prime_pow_sum`, `classic_descent_2xyz`, `classic_n4_4n`, `classic_fermat_little`, `classic_wilson`, `classic_wilson_converse`, `classic_two_squares`, `classic_four_squares`, `classic_gcd_pow2`, `classic_pythagorean`, `classic_2n_not_div`, `classic_xy_yx`, `classic_pq_qp`, `classic_bertrand`, `classic_primes_3mod4`, `classic_pyth_60`, `classic_3x_2y`, `classic_2x1_3y`, `classic_2x_3y_square`, `classic_x2_y3_1`, `classic_y2_x3_2`, `classic_fe_sum_squares`
* **dev**: `ident_diff_squares`, `pythag_three`, `square_not_prime`, `two_pow_gt_n`, `bernoulli_int`, `find_all_x2_eq_x`, `pell_infinitely_many`, `aime1991_p1`, `aime1997_p3`, `amc_alg_02`, `amc_alg_03`, `amc_mod_02`, `amc_prime_02`, `amc_ind_02`, `amc_dig_03`, `aime1985_p7`, `imo1994_p4`, `isl2007_n1`, `isl2016_n4`, `classic_two_coprime`
* **test**: `imo2019_p1`, `ident_brahmagupta`, `sum_two_squares_mod4`, `coprime_consecutive`, `imo1964_p1b`, `imo1964_p1a`, `ident_pell_product`, `ineq_cross_terms`, `three_consecutive_div3`, `three_consecutive_not_squarefree`, `find_all_xy_eq_x_plus_y`, `find_all_n_plus_1_dvd`, `frobenius_3_5`, `x2_y2_3z2`, `no_x2_minus_2y2_eq_3`, `n4_n2_1_composite`, `aime1985_p13`, `aime1989_p4`, `aime1989_p7`, `aime1993_p6`, `aime1988_p8`, `amc_alg_05`, `amc_alg_07`, `amc_div_06`, `amc_prime_01`, `amc_gcd_01`, `amc_ineq_03`, `amc_ineq_05`, `amc_dio_01`, `amc_dio_04`, `amc_dvs_02`, `amc_alg_13`, `amc_fe_01`, `aime1986_p10`, `aime1989_p3`, `imo1986_p1`, `imo1992_p1`, `imo2005_p4`, `imo2022_p5`, `imo1962_p1`, `imo1982_p4a`, `imo1982_p4b`, `imo1989_p5`, `n5_n4_1_composite`, `mersenne_base_2`, `mersenne_composite_exponent`, `isl2005_n3`, `ineq_cyclic_27_4`, `isl2007_n2`, `classic_ab_cd_sum`, `classic_fact_composite`, `classic_consecutive_product`, `classic_fermat_coprime`, `classic_fact_product_sum`, `classic_fact_sum_fact`, `classic_2p1_cube`, `classic_fe_n2_1`, `classic_fe_sum_prod`

## Regimes

Every problem is attacked with the **same library** (every global below `imo1959_p1`, the library the synthetic theorems are generated from) and its own block, and with nothing of the other problems: the theorems and helper lemmas of other problems are never offered.  In the **prefix** regime the problem's own helper lemmas are offered too; in the **strict** regime they are hidden, so the agent must find the ideas.  `premises.prefix` / `premises.strict` of a problem say what the engine offers: an integer n (the globals below n) or `{upto, hidden}` (the globals below `upto` except the id ranges in `hidden`).

## Reference proofs

187 demonstrations, 6033 actions in all (shortest 1, longest 466).  Every one replays through the engine from its JSON in a fresh engine over a freshly built world, and the proof term it builds is checked by the kernel (and, for the proofs that are not too large, by the independent checker).

actions by kind: LINARITH 1647, CLAIM 763, CASES 737, FORWARD 715, REWRITE_AT 518, APPLY_WITH 501, APPLY 317, USE 281, LINCOMB 122, REWRITE 103, NLINARITH 99, RING 85, REFINE 65, INTERVAL_CASES 51, INDUCT_WITH 29.

## Faithfulness of the statements

A kernel-checked proof proves the *formal* statement, not the problem.  Every statement is therefore read back into an AST and compared with an independent description of what the problem says (`semantic.Spec`: Python predicates, the numbers in the relation atoms, the body of existentials) on a box of values, and deliberately mutated statements (changed operators, constants, quantifiers, guards) must be told from it; a mutant that means the same is listed with its reason.  See `BANK.md`.  Inequalities that are over the reals in the original are stated for integers (noted in the problem).

## Library corpus

`library.json`: the proofs of the 453 lemmas of the base library (2677 actions) as action sequences, split by lemma name (train 319, dev 40, test 94).  4 lemma(s) have the statement of a bank problem and are marked `bank_duplicate`.

## Annotations and sub-theorems (hindsight)

`annotations.json`: for every action of every reference proof its **direction** (CLOSE / BACKWARD / FORWARD / SPLIT / CLAIM), the depth of its goal in the proof tree, the number of actions that close that goal (`subtree`), the goals it leaves, the goals open after it; per problem the techniques, the lemmas used, the CLAIMs (statement and size of their proofs), depth and width of the tree.  These are the labels for a manager that chooses the macro step and a worker that proves the pieces.

`subproblems.json`: the **theorems that the reference proofs contain**, each a problem of its own with a kernel-checked proof: 5649 closed subtrees (`solved`: a goal, closed over its hypotheses, with the actions that closed it) and 702 facts a single action derived (`derived`), 6033 actions in 187 proofs.  Every one was replayed in a fresh proof state and checked by the kernel when the file was made, and `subproblems.py --verify` does it again.  A sub-theorem belongs to the first of train, dev, test whose problem contains it (0 occurrences in later problems are recorded in `also_in`).  

26 of them (train: 26) have no `statement_code`: the typed term builder cannot say the statement (23 to 30 implications in the printed statement; a code has at most 400 tokens, and the longest statement that has one, with 22 implications, needs 385), so the environment cannot be started on it and the learners leave it out; `--verify` finds each again in the reference proof of its problem instead of restating it from its data (imo1983_p6_integer).

| split | solved | derived |
|---|---|---|
| train | 3879 | 527 |
| dev | 772 | 82 |
| test | 998 | 93 |

Use: hindsight relabelling (a failed attempt that proved a piece is turned into successes), goal-conditioned warm-up for the worker, a curriculum by `length`.  Train only on the `train` split.

## Files

`bank.json` (problems and demonstrations), `splits.json`, `library.json`, `annotations.json`, `subproblems.json`, `manifest.json` (digests).  World digest of the bank: `78850a291901432e4722af6ab7ff3ab7a1406985f090eff891d9e7b546018893`.
