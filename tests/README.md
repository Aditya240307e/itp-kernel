# tests/ -- the gates

A **gate** is a script that exits 0 and prints no `GATE: FAIL` iff the property it protects holds. `tests/gate_registry.py` is the one list of gates (and of the arguments they run with); `tests/run_gates.py`
runs them, `mutants/mutate.py` runs the kernel ones against every mutant of the code. Gates are scripts, not pytest tests: each is a self-contained program with its own fixtures, run from its own directory.

```
python3 tests/run_gates.py                        the 47 kernel gates, two at a time (about 13 minutes on two cores)
python3 tests/run_gates.py -j 1                   serially (use this when timing matters)
python3 tests/run_gates.py --ml                   + the learner's gate S44 (needs torch and gymnasium)
python3 tests/run_gates.py --repo                 + the repository gates S45 (layout) and S46 (examples)
python3 tests/run_gates.py --synth                + the synthetic theorem generator's gate S48 (about 70 seconds)
python3 tests/run_gates.py --all                  everything
python3 tests/run_gates.py s23_ring_tests.py s31_linarith_tests.py       only these, by file name (wherever they are registered)
ITP_VERIFY_MEMO=1 python3 tests/run_gates.py      the same, with the kernel re-deciding every answer it took from a sharing table (about twice as slow)
```

A gate can also be run by hand, from anywhere: `python3 tests/library/s23_ring_tests.py`.

## How the directory is organised

| directory | what it holds |
|---|---|
| `kernel/` | the kernel against the frozen reference (lockstep), its soundness, boundaries and performance guards, the independent checker, quotients, proof irrelevance, the work budget |
| `library/` | the libraries and the decision procedures built on them: `ring`, binary naturals, the integers, `linarith`, number theory |
| `environment/` | the learner's engine: observation, menus, probes, the action language, the typed term builder, the recorder |
| `problems/` | the problem bank: faithfulness of statements, reference proofs, dataset |
| `learner/` | features, policy network, search, Gymnasium environment, RL bookkeeping |
| `synth/` | the synthetic theorem generator (`src/python/synth`) and the datasets it writes: determinism, records replayed from their data, tamper tests, merge and splits, the command line |
| `repo/` | the repository itself: layout, registry, imports, documents, examples |
| `support/` | modules shared by gates (the differential harness, the self-consistency oracle, the soundness canaries, a Prop-world builder); `harness.py` is also a gate |
| `fuzz/`, `bench/` | the environment fuzzer and the kernel micro-benchmarks (not gates) |
| `golden/` | pinned data of S40, S41, S43, S48 (statements, action sequences, generated digests; `S40_UPDATE_GOLDEN=1` etc. rewrite them when a change is intended) |

Modules are imported by plain name from anywhere in `src/python` and `tests` (see `src/python/itp_paths.py`), so a gate can `import gym_env` or `import harness` without a path.

## The gates

Times are seconds on the 2-core box with two gates at a time (the last full run, 2026-10-18). A gate passes iff it exits 0 and prints no `GATE: FAIL`.

### Kernel gates (`tests/kernel`)

| gate | time | what it protects |
|---|---|---|
| `s2_tests.py 1500` | 1 s | The full typing core of the hybrid kernel against the reference kernel. |
| `s26_memo_tests.py` | 5 s | The facts every term node carries, the shared-term (DAG) traversals, the typing memo, and the fast proof extraction. |
| `s16_quot_tests.py` | 2 s | Quotients. |
| `s17_irrelevance_tests.py` | 8 s | Definitional proof irrelevance. |
| `s18_deep_tests.py` | 32 s | The kernel survives terms of any depth. |
| `s19_boundary_tests.py` | 5 s | The Python API boundary. Nothing a caller passes -- however malformed -- may crash the process, hang it, or be accepted when it is nonsense. |
| `s20_reservation_tests.py` | 1 s | The global environment stays WELL FOUNDED -- the reservation discipline. |
| `s21_view_tests.py` | 1 s | Reading terms back (`Kernel.view`) and the validity of term handles. |
| `s32_synconv_tests.py` | 2 s | The quick conversion check (SynConv) in front of the evaluator. |
| `s33_convmemo_tests.py` | 6 s | Conversion and unification over SHARED structure (the tables ConvMemo and UnifyMemo). |
| `s35_matchcheck_tests.py` | 1 s | The kernel checks a `match` against the type of its scrutinee AS WRITTEN, and the result type of the match is read off syntax too. |
| `s39_failfast_tests.py` | 2 s | A comparison that FAILS is decided by what it found, not by normalising both sides. |
| `s27_fuel_tests.py` | 51 s | The work budget. |
| `s28_headunfold_tests.py` | 0 s | A definition at the head of a hypothesis or a goal is unfolded ON SYNTAX, so that its arguments stay as the author wrote them. |
| `s29_apply_pi_goal_tests.py` | 0 s | `apply` of a lemma to a goal that is itself a Pi (an implication, a negation, a universal statement). |
| `s3_tests.py 600` | 3 s | Metas, unification, intro/apply/rewrite against the reference, as lockstep tactic scripts. |
| `s4_tests.py 800` | 4 s | Cases and induct against the reference, as lockstep tactic scripts. |
| `s5_tests.py 1200 3` | 5 s | The library index. apply_candidates must return EXACTLY the reference's candidate list (same entries, same order) for every goal of every state. |
| `s5_probe_tests.py 500 7` | 7 s | The search's probing protocol -- checkpoint / fork_single_goal / apply / read off / rewind -- and materialize_from_node. |
| `s6_logic_tests.py` | 1 s | The logical vocabulary (F7) with a shape probe per connective, and the F5 shape (elimination over a lambda predicate). |
| `s6_f6_tests.py` | 0 s | Induction with a Pi-headed goal, and with an explicit `generalizing` list. The classic case: add (S n) acc = add n (S acc) needs the IH for ALL acc. |
| `s7_param_tests.py 150 600` | 2 s | The parameter / index distinction (kernel option 'parameters'). Default mode is untouched (the lockstep gates cover it); here the option is ON. |
| `s8_indep_tests.py 100` | 2 s | The independently written checker against the kernels. |
| `s9_strong_tests.py` | 0 s | F6 probe: strong induction by a BOUNDED PREDICATE (Below P n := True / And (Below P k) (P k)), so the induction hypothesis is a plain proposition, not a forall. |
| `s10_noconf_tests.py` | 0 s | Injectivity: generated lemmas are accepted by the kernel's CHECKED path and by the independent checker, and they close the goals discrimination cannot. |
| `s11_eqgen_tests.py` | 2 s | Index generalisation with equations (induct on a hypothesis whose index is a concrete term). |
| `s12_probe_eq_tests.py 150` | 8 s | Probing protocol x index generalisation (hybrid only; the equation path does not exist in the reference). |
| `s13_prop_tests.py` | 0 s | `prop_elim`: level 0 is Prop, and an inductive family in Type0 that is not a syntactic subsingleton eliminates only into Prop. |
| `s14_prop_world_tests.py 600` | 4 s | The Prop/Type world (`prop_elim` + parameters, data in Type1): the library built by tactics in this mode, every global re-checked by the independent checker. |
| `s15_classical_tests.py 300` | 1 s | Classical logic in the Prop/Type world: the four axioms are the only additions to the trusted base; the derived principles are proved by tactics and re-checked. |
| `oracle_gate.py` | 33 s | The self-consistency oracle: random tactic scripts run in lockstep on both kernels and re-checked; the strict fragment must be perfectly clean, and the other distributions may only fail where a documented finding (F12) explains it. |

### Libraries and decision procedures (`tests/library`)

| gate | time | what it protects |
|---|---|---|
| `s22_simp_tests.py` | 0 s | The display normal form (`simp_normalize`), the `change` tactic and `apply` with folded premises. |
| `s23_ring_tests.py` | 28 s | The natural-number library and the `ring` tactic. |
| `s24_bin_tests.py` | 31 s | The binary natural numbers (bin_lib.py: `Pos`, `N`) and their bridge to the unary numerals. |
| `s25_nring_tests.py` | 36 s | `ring` over the binary naturals `N` (native, no detour through the unary numerals), and the lemma set it shares with the unary `ring`. |
| `s30_zlib_tests.py` | 27 s | The integers `Z` (int_lib.py) -- a quotient of pairs of binary naturals, with every law proved -- and `ring` over them. |
| `s31_linarith_tests.py` | 69 s | `linarith` and `nlinarith` over the integers (linarith.py). |
| `s34_nt_tests.py` | 69 s | Elementary number theory over the integers (nt_lib.py) and the first olympiad problems (imo_lib.py), `linear_combination` (lincomb.py). |
| `s37_atoms_tests.py` | 10 s | Atoms up to the arithmetic in their arguments (ring.py: Ring.canon / canon_arg, shared by `ring`, `linarith`, `linear_combination`). |

### The learner's environment (`tests/environment`)

| gate | time | what it protects |
|---|---|---|
| `s40_actions_tests.py` | 70 s | The second action language: the typed term builder (gym_build.py), the actions that say their arguments (gym_acts.py), their portable form, and the recorder (recorder.py) that turns authored proofs into action sequences. |
| `s42_pins_tests.py` | 4 s | Pins: small direct tests of the corner cases of the typed term builder (gym_build.py) and of the action language (gym_acts.py) that the larger gates (S36, S38, S40) only cross. |
| `s47_hindsight_tests.py` | 127 s | Hindsight: the proof tree of an episode (gym_tree.py) and the theorems it proved -- exact, replayed in a fresh engine, kernel-checked and judged by the independent checker; tamper tests; the restricted second opinion (recheck.py). |
| `s36_gym_tests.py` | 27 s | The learner's environment on the hybrid kernel (gym_env.py, gym_search.py), the `clear` tactic and the observation of goals. |
| `s38_terms_tests.py` | 16 s | The actions that carry terms (INSTANTIATE, USE), the term pool they draw from, probes with a chosen work budget, goal identity up to irrelevant order, and the search on cost. |

### The problem bank (`tests/problems`)

| gate | time | what it protects |
|---|---|---|
| `s41_bank_tests.py` | 184 s | The problem bank -- formal olympiad problems whose statements are CHECKED TO MEAN what the problem says, with reference proofs that replay through the action language. |
| `s43_bank2_tests.py` | 396 s | The extended problem bank -- the 54 problems of bank_lib2.py on top of the 28 of S41, and the infrastructure that they needed. |

### Differential harness (`tests/support`)

| gate | time | what it protects |
|---|---|---|
| `harness.py 1500 3` | 1 s | Differential harness: reference kernel (absolute levels) vs hybrid kernel (bound = index, context = level). |

### The learner (`tests/learner`; `run_gates.py --ml`)

| gate | time | what it protects |
|---|---|---|
| `s44_ml_tests.py` | 105 s | The learner's machinery -- features, the policy network, the search it guides, the Gymnasium environment (budget, goal-conditioned reset, achieved goals) and the policy-gradient bookkeeping. |

### The synthetic theorem generator (`tests/synth`; `run_gates.py --synth`)

| gate | time | what it protects |
|---|---|---|
| `s48_synth_tests.py` | 70 s | The synthetic theorem generator (`src/python/synth`, `docs/GENERATOR.md`) and its datasets -- the polynomial normal form behind "tidy"; records that are a function of (configuration, index) in any order, in a new generator and in other processes with other hash seeds; every record replayed from its own data in a fresh engine and judged by the independent checker; about thirty tamper cases; merge, splits (no recipe of `test_comp` elsewhere), exclusion of library lemmas, bank problems and sub-theorems; the command line end to end. |

### The repository (`tests/repo`; `run_gates.py --repo`)

| gate | time | what it protects |
|---|---|---|
| `s45_layout_tests.py` |  | The repository's layout is the layout the documentation describes, and stays so; the release archive is sound; the copies of the tree in which mutants and sanitizers run are complete. |
| `s46_examples_tests.py` |  | The examples run. |

## Adding a gate

1. Write `tests/<role>/sNN_name_tests.py`: a docstring (first line = what it protects), the bootstrap line (copy it from a neighbour), `FAIL`/`CHECKS` lists, an `expect(cond, what)` helper, and a `main()` that prints
   `SNN NAME GATE: PASS` or `... FAIL` with the failures and exits non-zero.
2. Register it in `tests/gate_registry.py` (`GATE_CMDS` if a mutant of the kernel or tactic layer could trip it, `ML_GATE_CMDS` for the learner, `REPO_GATE_CMDS` for the repository).
3. Add a row to the table above (S45 checks that every gate is listed here).
4. Mutate what it protects: add mutants to `mutants/catalogue.py` (`mutants/README.md`) and see that the gate catches them -- a gate that no mutant can trip has not been shown to check anything.
