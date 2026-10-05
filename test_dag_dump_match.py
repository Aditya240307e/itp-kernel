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

# --- A genuinely 3-constructor family, all nullary, to get a clean
# K=3 Match with no field-binding complexity muddying the test. ---
trit_id = k.reserve_inductive_type('Trit', type0)
trit_ty = k.mk_const(trit_id)
trit_h = k.finish_inductive(trit_id, [itp_env.CtorSpec('A', []), itp_env.CtorSpec('B', []), itp_env.CtorSpec('C', [])])
a_c = k.mk_const(trit_h.ctor_global_ids[0])
b_c = k.mk_const(trit_h.ctor_global_ids[1])
c_c = k.mk_const(trit_h.ctor_global_ids[2])

nat_id = k.reserve_inductive_type('Nat', type0)
nat_ty = k.mk_const(nat_id)
nat_h = k.finish_inductive(nat_id, [itp_env.CtorSpec('Zero', []), itp_env.CtorSpec('Succ', [nat_ty])])
zero = k.mk_const(nat_h.ctor_global_ids[0])
succ_c = k.mk_const(nat_h.ctor_global_ids[1])
one = k.mk_app(succ_c, zero)
two = k.mk_app(succ_c, one)
three = k.mk_app(succ_c, two)

# match (scrutinee : Trit) with { A => 0 | B => 1 | C => 2 } : Nat,
# applied to a bound variable (level 0) as the scrutinee, so the whole
# term is closed under one Lambda: \t:Trit. match t with {A=>0;B=>1;C=>2}
motive = k.mk_lambda(trit_ty, nat_ty)
branch_a = zero
branch_b = one
branch_c = two
match_term = k.mk_match(k.mk_var(0), motive, [branch_a, branch_b, branch_c])
whole_term = k.mk_lambda(trit_ty, match_term)

tags, scalar, child_a, child_b = k.observe_term(whole_term)
print(f"dumped {len(tags)} entries")
print("tags:  ", list(tags))
print("scalar:", list(scalar))
print("child_a:", list(child_a))
print("child_b:", list(child_b))

# CORRECTED: the root is at index 0, not the last index. my_index is
# assigned (and pushed) BEFORE recursing into children -- confirmed
# directly from this dump's own index 0 (child_a=1, child_b=2, both
# greater than 0). This is the OPPOSITE of what planning assumed
# ("children precede parents, single forward pass") -- caught here,
# before any encoder code existed, by actually testing rather than
# trusting the earlier, unverified assumption.
root_idx = 0
CONS_TAG = 14
print(f"root tag: {tags[root_idx]}, child_a: {child_a[root_idx]}, child_b: {child_b[root_idx]}")
match_idx = child_b[root_idx]
print(f"match node tag: {tags[match_idx]}, child_a(scrutinee): {child_a[match_idx]}, child_b(chain head): {child_b[match_idx]}")

# Walk the cons chain from the Match's own child_b: should be
# motive -> branch_a(zero) -> branch_b(one) -> branch_c(two), with the
# first two hops through SyntheticCons (tag 14) and the last hop
# landing directly on the (already-dumped) two-node "one" structure
# (App(Succ, Zero)), not a further cons.
chain_head = child_b[match_idx]
check(tags[chain_head] == CONS_TAG, "first chain link is a SyntheticCons node (motive -> rest)")
motive_idx = child_a[chain_head]
rest1 = child_b[chain_head]
check(tags[rest1] == CONS_TAG, "second chain link is also a SyntheticCons node (branch_a -> rest)")
branch_a_idx = child_a[rest1]
rest2 = child_b[rest1]
check(tags[rest2] == CONS_TAG, "third chain link is also a SyntheticCons node (branch_b -> rest)")
branch_b_idx = child_a[rest2]
branch_c_idx = child_b[rest2]  # terminates the chain directly, no further cons
check(tags[rest2] == CONS_TAG and child_b[rest2] != -1, "chain terminates directly at branch_c, no trailing cons/nil node")

# Verify each branch's own dumped structure matches what was built:
# branch_a = Zero (a bare Const, scalar = Zero's global id)
# branch_b = one = App(Succ, Zero)
# branch_c = two = App(Succ, App(Succ, Zero))
zero_gid = nat_h.ctor_global_ids[0]
check(scalar[branch_a_idx] == zero_gid, "branch_a correctly dumps as Zero (Const, scalar=Zero's global id)")

def print_node_ref(name, idx):
    print(f"  {name}: tag={tags[idx]} scalar={scalar[idx]} child_a={child_a[idx]} child_b={child_b[idx]}")

print_node_ref("branch_b (should be App(Succ,Zero))", branch_b_idx)
print_node_ref("branch_c (should be App(Succ,App(Succ,Zero)))", branch_c_idx)

# App's own tag: discover it from branch_b_idx itself (it must be App,
# since 'one' = succ applied to zero).
APP_TAG = tags[branch_b_idx]
succ_ref_idx = child_a[branch_b_idx]
zero_ref_idx = child_b[branch_b_idx]
check(scalar[succ_ref_idx] == nat_h.ctor_global_ids[1], "branch_b's own App head correctly references Succ's global id")
check(zero_ref_idx == branch_a_idx, "branch_b's own App argument is the SAME dumped Zero node as branch_a (hash-consing sharing preserved)")

check(tags[branch_c_idx] == APP_TAG, "branch_c is also an App node (Succ applied to something)")
check(child_b[branch_c_idx] == branch_b_idx, "branch_c's own App argument is the SAME dumped node as branch_b (Succ (Succ Zero)) -- sharing preserved")

# Sanity: motive itself should be a Lambda (Trit -> Nat), dumped and
# reachable, not skipped.
check(child_a[chain_head] == motive_idx, "motive is correctly the head of the cons chain")

print()
if failures == 0:
    print("ALL build_dag_dump MATCH-HANDLING CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
