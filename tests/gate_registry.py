"""gate_registry.py -- the gates: which scripts exist and with which arguments they run.  The one place to add a gate.

Every entry is [path relative to tests/, argument ...].  tests/run_gates.py runs them; mutants/mutate.py runs GATE_CMDS against every mutant (so a gate that a mutant of the kernel or of the tactic layer could
trip belongs there, and the order matters for speed only: a mutant stops at the first gate that fails).  A gate passes iff it exits 0 and prints no "GATE: FAIL".
"""
# Order matters only for speed (a mutant stops at the first gate that fails): fast, high-yield gates first, the 25-second oracle last.
GATE_CMDS = (['kernel/s2_tests.py', '1500'], ['kernel/s26_memo_tests.py'], ['kernel/s16_quot_tests.py'],
             ['kernel/s17_irrelevance_tests.py'], ['kernel/s18_deep_tests.py'], ['kernel/s19_boundary_tests.py'],
             ['kernel/s20_reservation_tests.py'], ['kernel/s21_view_tests.py'], ['library/s22_simp_tests.py'],
             ['library/s23_ring_tests.py'], ['library/s24_bin_tests.py'], ['library/s25_nring_tests.py'],
             ['library/s30_zlib_tests.py'], ['library/s31_linarith_tests.py'], ['library/s34_nt_tests.py'],
             ['kernel/s32_synconv_tests.py'], ['kernel/s33_convmemo_tests.py'], ['kernel/s35_matchcheck_tests.py'],
             ['environment/s40_actions_tests.py'], ['environment/s42_pins_tests.py'], ['problems/s41_bank_tests.py'],
             ['problems/s43_bank2_tests.py'], ['environment/s36_gym_tests.py'], ['library/s37_atoms_tests.py'],
             ['environment/s38_terms_tests.py'], ['kernel/s39_failfast_tests.py'], ['kernel/s27_fuel_tests.py'],
             ['kernel/s28_headunfold_tests.py'], ['kernel/s29_apply_pi_goal_tests.py'], ['support/harness.py', '1500', '3'],
             ['kernel/s3_tests.py', '600'], ['kernel/s4_tests.py', '800'], ['kernel/s5_tests.py', '1200', '3'],
             ['kernel/s5_probe_tests.py', '500', '7'], ['kernel/s6_logic_tests.py'], ['kernel/s6_f6_tests.py'],
             ['kernel/s7_param_tests.py', '150', '600'], ['kernel/s8_indep_tests.py', '100'], ['kernel/s9_strong_tests.py'],
             ['kernel/s10_noconf_tests.py'], ['kernel/s11_eqgen_tests.py'], ['kernel/s12_probe_eq_tests.py', '150'],
             ['kernel/s13_prop_tests.py'], ['kernel/s14_prop_world_tests.py', '600'], ['kernel/s15_classical_tests.py', '300'],
             ['kernel/oracle_gate.py'])
# the gates of the LEARNER's machinery (features, network, search, Gymnasium environment): they need torch and gymnasium, which the kernel and its tactics do not, and a mutant of the kernel is caught by the
# gates above long before these would matter, so the mutation driver does not run them; `run_gates.py --ml` does
ML_GATE_CMDS = (['learner/s44_ml_tests.py'],)
# the gates of the repository itself (its layout, its examples); `run_gates.py --repo`
REPO_GATE_CMDS = (['repo/s45_layout_tests.py'], ['repo/s46_examples_tests.py'])
