"""test_mul.py -- correctness gate for setup_mul (mul on Nat).

Same discipline as binary_nat.py's own testing: kernel acceptance of the definition is necessary
but not sufficient (that exact gap is what hid the binary_nat 5+5=12 bug until exhaustive,
value-level testing caught it). mul is computational, so this DOES normalize and compare against
Python's own multiplication directly, unlike Le/Lt's propositional testing.
"""
import sys

import itp_env
import curriculum as cur
import hypergraph as hg

failures = 0


def check(cond, msg):
    global failures
    if cond:
        print(f"[PASS] {msg}")
    else:
        failures += 1
        print(f"[FAIL] {msg}")


k, refs = cur.build_base_kernel()
# setup_mul now genuinely depends on add_comm_c (needed for mul_succ_right's own proof), which
# setup_add_comm provides -- which itself genuinely depends on add_zero_right_c/add_succ_right_c,
# which only exist once CURRICULUM has actually been run. This mirrors real prerequisite order, not
# an arbitrary convenience: establish it properly rather than hand-wiring just the pieces this test
# happens to need.
for goal in cur.CURRICULUM:
    lib = k.index_library(); gty = goal.build_type(k, refs)
    ps = k.start_proof(gty); rm = ps.goal_meta_id(0); dm = {}
    root = hg.get_or_create_node(k, ps, 0, dm)
    status = hg.search(k, root, lib, hg.UniformPolicy(), dm, max_simulations=200, top_k=16)
    if status == hg.Status.SOLVED and goal.lemma_name:
        term = hg.commit_and_verify(k, ps, root, rm, gty)
        gid = k.declare_checked(gty, term, goal.lemma_name)
        refs[goal.lemma_name + '_c'] = k.mk_const(gid)
refs.update(cur.setup_add_comm(k, refs))
mul_refs = cur.setup_mul(k, refs)
refs.update(mul_refs)


def mk_nat(v):
    t = refs['zero']
    for _ in range(v):
        t = k.mk_app(refs['succ_c'], t)
    return t


def nat_val_via_print(term):
    s = k.print_node(term)
    return s.count('Succ')


# --------------------------------------------------------------------------- 1. exhaustive mul correctness
N_EXHAUSTIVE = 20
ok = True
checked = 0
for a in range(0, N_EXHAUSTIVE + 1):
    for b in range(0, N_EXHAUSTIVE + 1):
        term = k.mk_app(k.mk_app(refs['mul_c'], mk_nat(a)), mk_nat(b))
        try:
            normalized = k.normalize_node(term)
        except Exception as e:
            ok = False
            print(f"   mul({a},{b}) normalize FAILED: {e}")
            break
        got = nat_val_via_print(normalized)
        expected = a * b
        if got != expected:
            ok = False
            print(f"   mul({a},{b}) = {got}, expected {expected}")
            break
        checked += 1
    if not ok:
        break
check(ok, f"mul(a,b) == a*b for every 0<=a,b<={N_EXHAUSTIVE} ({checked:,} pairs, "
          f"{(N_EXHAUSTIVE+1)**2:,} expected)")

# --------------------------------------------------------------------------- 2. a few larger, spot-checked pairs
# Deliberately small bounds: unary Nat's well-known scaling cost (the whole motivation for this
# project's earlier binary_nat work) makes larger products needlessly slow to normalize/print
# repeatedly here, and the exhaustive check above already gives strong, full coverage up to 20 --
# this is just a secondary, confirmatory spot-check slightly past that range, not a stress test.
import random
random.seed(0)
ok2 = True
for _ in range(50):
    a, b = random.randint(0, 25), random.randint(0, 25)
    term = k.mk_app(k.mk_app(refs['mul_c'], mk_nat(a)), mk_nat(b))
    got = nat_val_via_print(k.normalize_node(term))
    if got != a * b:
        ok2 = False
        print(f"   mul({a},{b}) = {got}, expected {a*b}")
check(ok2, "mul(a,b) == a*b for 50 random pairs with a,b up to 25")

# --------------------------------------------------------------------------- 3. real curriculum lemmas via search
def try_goal(name, build_type, sims=200, top_k=16):
    lib = k.index_library()
    gty = build_type(k, refs)
    ps = k.start_proof(gty)
    rm = ps.goal_meta_id(0)
    dm = {}
    root = hg.get_or_create_node(k, ps, 0, dm)
    status = hg.search(k, root, lib, hg.UniformPolicy(), dm, max_simulations=sims, top_k=top_k)
    solved = status == hg.Status.SOLVED
    verified = False
    if solved:
        try:
            hg.commit_and_verify(k, ps, root, rm, gty)
            verified = True
        except Exception as e:
            print(f"   {name}: search claimed SOLVED but independent verification FAILED: {e}")
    print(f"   {name}: search {'SOLVED' if solved else 'did not solve'} "
          f"({len(dm)} nodes){' -- independently VERIFIED' if verified else ''}")
    return solved and verified


def mul_app(k, refs, a, b):
    return k.mk_app(k.mk_app(refs['mul_c'], a), b)


def build_mul_zero_left(k, refs):
    # (n:Nat) -> Eq Nat (mul Zero n) Zero -- near-definitional (mul's own base case)
    return k.mk_pi(refs['nat_ty'],
             cur.eq_app(k, refs, refs['nat_ty'], mul_app(k, refs, refs['zero'], k.mk_var(0)), refs['zero']))


print("\n  -- unaided UniformPolicy search --")
solved_zero_left = try_goal("mul_zero_left   (n:Nat) -> Eq Nat (mul Zero n) Zero", build_mul_zero_left, sims=80)
check(solved_zero_left, "mul_zero_left solved and independently verified by plain search")

# mul_zero_right: confirmed NOT to bootstrap via plain UniformPolicy search within a modest budget
# (matching this project's own add_comm precedent) -- and, separately, confirmed that pushing the
# budget further risks a real, documented search-blowup from repeated self-re-induction on Nat
# (unrelated to mul specifically). Not re-attempted via search here for that reason; it is instead
# hand-built directly inside setup_mul (mirroring cong's own hand-built construction in
# build_base_kernel), already independently whole_term_check-verified there, and already present in
# refs by the time this test reaches here. Spot-check that it is, in fact, present and usable:
check('mul_zero_right_c' in refs, "mul_zero_right is present as a permanent library fact (hand-built in setup_mul)")
check('cong_add_zero_c' in refs, "cong_add_zero (the supporting congruence helper) is present as a permanent library fact")

# mul_succ_right, mul_comm, and their own supporting helpers (add_assoc, cong_add_left,
# cong_add_right) -- also hand-built, for the same reason (confirmed: plain search does not
# bootstrap any of these within a safe budget). mul_comm is this project's own "critical test" of
# whether curriculum-grading alone gets past comm-style stalls without a rewrite tactic: confirmed
# it does not, by itself, make mul_comm search-findable -- but confirmed the lemma is fully provable
# with the existing 4 tactics via direct, hand-built term construction.
for name in ('add_assoc_c', 'cong_add_left_c', 'cong_add_right_c', 'mul_succ_right_c', 'mul_comm_c'):
    check(name in refs, f"{name[:-2]} is present as a permanent library fact (hand-built in setup_mul)")

print()
print("ALL MUL TESTS PASSED" if failures == 0 else f"{failures} MUL TEST(S) FAILED")
sys.exit(1 if failures else 0)
