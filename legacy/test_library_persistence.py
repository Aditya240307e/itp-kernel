import itp_env

failures = 0
def check(cond, label):
    global failures
    if cond:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        failures += 1

# --- Build a real library in Kernel A, covering every declaration-event
# kind: an inductive family with constructors (Nat), another (Eq), a
# genuine RECURSIVE function (add), and a declare_checked'd lemma. ---
kA = itp_env.Kernel()
type0 = kA.mk_type(0)
prop0 = kA.mk_prop()

nat_id = kA.reserve_inductive_type('Nat', type0)
nat_ty = kA.mk_const(nat_id)
nat_h = kA.finish_inductive(nat_id, [itp_env.CtorSpec('Zero', []), itp_env.CtorSpec('Succ', [nat_ty])])
zero = kA.mk_const(nat_h.ctor_global_ids[0])
succ_c = kA.mk_const(nat_h.ctor_global_ids[1])

eq_sort = kA.mk_pi(type0, kA.mk_pi(kA.mk_var(0), kA.mk_pi(kA.mk_var(0), prop0)))
eq_id = kA.reserve_inductive_type('Eq', eq_sort)
eq_ty = kA.mk_const(eq_id)
eq_h = kA.finish_inductive(eq_id, [itp_env.CtorSpec('refl', [type0, kA.mk_var(0)], [kA.mk_var(0), kA.mk_var(1), kA.mk_var(1)])])
refl_c = kA.mk_const(eq_h.ctor_global_ids[0])

# add : Nat -> Nat -> Nat
# add n m = match n with { Zero => m ; Succ n' => Succ (add n' m) }
# Non-dependent motive: \_:Nat. Nat -- a Pi-chain of (index_count(=0)+1)
# layers, matching the shape confirmed in Module C's own EmptyRel work,
# not a bare Nat.
add_fn_type = kA.mk_pi(nat_ty, kA.mk_pi(nat_ty, nat_ty))
add_gid = kA.reserve_recursive_function('add', add_fn_type)
add_c = kA.mk_const(add_gid)

add_motive = kA.mk_lambda(nat_ty, nat_ty)
add_zero_branch = kA.mk_var(1)  # m
add_succ_branch = kA.mk_lambda(nat_ty, kA.mk_app(succ_c, kA.mk_app(kA.mk_app(add_c, kA.mk_var(2)), kA.mk_var(1))))
add_match = kA.mk_match(kA.mk_var(0), add_motive, [add_zero_branch, add_succ_branch])
add_body = kA.mk_lambda(nat_ty, kA.mk_lambda(nat_ty, add_match))

kA.finish_recursive_function_checked(add_gid, nat_id, 0, add_body)
print("[setup] add declared and checked in Kernel A")

one = kA.mk_app(succ_c, zero)
two = kA.mk_app(succ_c, one)
add_1_1 = kA.mk_app(kA.mk_app(add_c, one), one)
check(kA.normalize_node(add_1_1) == two, "add 1 1 reduces to 2 in Kernel A (sanity check before export)")

# A declare_checked'd lemma: one_plus_one_eq_two : Eq Nat (add 1 1) 2,
# proven directly via refl (add 1 1 reduces to 2 definitionally).
lemma_type = kA.mk_app(kA.mk_app(kA.mk_app(eq_ty, nat_ty), add_1_1), two)
lemma_value = kA.mk_app(kA.mk_app(refl_c, nat_ty), two)
lemma_gid = kA.declare_checked(lemma_type, lemma_value, 'one_plus_one_eq_two')
print(f"[setup] lemma declared and checked in Kernel A, gid={lemma_gid}")
print(f"[setup] Kernel A library_size: {kA.library_size()}")

# --- Export from Kernel A ---
events = kA.export_library()
print(f"[export] {len(events)} declaration events exported")
for e in events:
    print(f"  {e['kind']}: {e['name']}")

# --- Import into a COMPLETELY SEPARATE, fresh Kernel B ---
kB = itp_env.Kernel()
kB.import_library(events)
print(f"[import] Kernel B library_size after import: {kB.library_size()}")
check(kB.library_size() == kA.library_size(), "Kernel B's library size matches Kernel A's after import")

# --- Confirm the reconstructed library is GENUINELY usable in Kernel B:
# build a NEW proof there using the reconstructed add/Eq/refl, and
# confirm computation still works correctly. ---
globals_B = {name: gid for gid, name, ty, is_axiom in kB.list_globals()}
add_gid_B = globals_B['add']
nat_gid_B = globals_B['Nat']
zero_gid_B = globals_B['Zero']
succ_gid_B = globals_B['Succ']
eq_gid_B = globals_B['Eq']
refl_gid_B = globals_B['refl']

nat_ty_B = kB.mk_const(nat_gid_B)
zero_B = kB.mk_const(zero_gid_B)
succ_B = kB.mk_const(succ_gid_B)
add_B = kB.mk_const(add_gid_B)
eq_ty_B = kB.mk_const(eq_gid_B)
refl_B = kB.mk_const(refl_gid_B)

one_B = kB.mk_app(succ_B, zero_B)
two_B = kB.mk_app(succ_B, one_B)
three_B = kB.mk_app(succ_B, two_B)
add_1_2_B = kB.mk_app(kB.mk_app(add_B, one_B), two_B)
check(kB.normalize_node(add_1_2_B) == three_B,
      "reconstructed add computes correctly in Kernel B (add 1 2 = 3, NEW computation, not seen in Kernel A)")

# Use the reconstructed lemma directly in a NEW proof, entirely within
# Kernel B -- confirming it's genuinely usable, not just present.
lemma_gid_B = globals_B['one_plus_one_eq_two']
lemma_c_B = kB.mk_const(lemma_gid_B)
new_goal_B = kB.mk_app(kB.mk_app(kB.mk_app(eq_ty_B, nat_ty_B), add_1_2_B), three_B)
# Wait -- the reconstructed lemma proves add 1 1 = 2, not add 1 2 = 3.
# Use it for what it actually proves instead.
new_goal_B = kB.mk_app(kB.mk_app(kB.mk_app(eq_ty_B, nat_ty_B), kB.mk_app(kB.mk_app(add_B, one_B), one_B)), two_B)
ps_B = kB.start_proof(new_goal_B)
ps_B.apply(lemma_c_B)
check(ps_B.is_complete(), "reconstructed lemma directly closes a new goal in Kernel B via apply()")
term_B = ps_B.extract()
kB.whole_term_check(term_B, new_goal_B)
print("[check] whole-term check() of the Kernel-B-only proof, using the reconstructed lemma: PASSED")

if failures == 0:
    print("\nALL LIBRARY PERSISTENCE CHECKS PASSED")
else:
    print(f"\n{failures} CHECK(S) FAILED")
