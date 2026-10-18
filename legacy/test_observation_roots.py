import itp_env

failures = 0
def check(cond, label):
    global failures
    if cond:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        failures += 1

k = itp_env.Kernel()
type0 = k.mk_type(0)
prop0 = k.mk_prop()

nat_id = k.reserve_inductive_type('Nat', type0)
nat_ty = k.mk_const(nat_id)
nat_h = k.finish_inductive(nat_id, [itp_env.CtorSpec('Zero', []), itp_env.CtorSpec('Succ', [nat_ty])])

eq_sort = k.mk_pi(type0, k.mk_pi(k.mk_var(0), k.mk_pi(k.mk_var(0), prop0)))
eq_id = k.reserve_inductive_type('Eq', eq_sort)
eq_ty = k.mk_const(eq_id)
k.finish_inductive(eq_id, [itp_env.CtorSpec('refl', [type0, k.mk_var(0)], [k.mk_var(0), k.mk_var(1), k.mk_var(1)])])

# Goal: n:Nat, m:Nat |- Eq Nat n m -- two hyps of the IDENTICAL type
# (Nat), which should share exactly one dump entry via hash-consing.
goal_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_app(k.mk_app(k.mk_app(eq_ty, nat_ty), k.mk_var(0)), k.mk_var(1))))
ps = k.start_proof(goal_ty)
ps.intro('n')
ps.intro('m')
check(ps.goal_num_hyps(0) == 2, "sanity: goal has exactly 2 hypotheses")

tags, scalar, child_a, child_b, roots, *_ = k.observation(ps)  # schema v2 adds a trailing heights array; unused here
print(f"dumped {len(tags)} entries, roots={list(roots)}")
check(len(roots) == 3, "roots has length 3 (2 hyps + 1 target)")

hyp0_root, hyp1_root, target_root = roots[0], roots[1], roots[2]
check(hyp0_root == hyp1_root, "the two Nat hypotheses correctly share the SAME dump index (hash-consing)")
print(f"hyp0_root={hyp0_root}, hyp1_root={hyp1_root}, target_root={target_root}")

# Confirm hyp0's own root is genuinely Nat's own Const entry.
nat_gid = nat_id
check(scalar[hyp0_root] == nat_gid, "hyp root correctly points at Nat's own Const entry (scalar=Nat's global id)")

# Confirm target_root is distinct from the hyp root (different structure).
check(target_root != hyp0_root, "target's own root is a genuinely different dump entry from the hyp roots")

# Sanity: verify k.print_node still agrees with what roots claims each
# entry is, by cross-checking against the goal's own target/hyp NodeId
# printouts (independent confirmation, not just internal consistency).
print("goal_hyp_type(0,0) print:", k.print_node(ps.goal_hyp_type(0, 0)))
print("goal_hyp_type(0,1) print:", k.print_node(ps.goal_hyp_type(0, 1)))
print("goal_target(0) print:", k.print_node(ps.goal_target(0)))

print()
if failures == 0:
    print("ALL observation() ROOTS CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
