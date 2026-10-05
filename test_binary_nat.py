"""test_binary_nat.py -- correctness gate for binary_nat.py.

Two independent checks, neither sufficient alone: (1) the kernel's own type/termination checker
already accepted every definition in setup_binary_nat (raises loudly otherwise); (2) exhaustive
testing here, normalizing every operation for every value pair up to N_EXHAUSTIVE and comparing
against Python's own arithmetic -- this is what actually confirms the VALUES are right, not just
that the definitions are well-typed and terminating.
"""
import sys

import itp_env
import curriculum
import binary_nat as bn

failures = 0


def check(cond, msg):
    global failures
    if cond:
        print(f"[PASS] {msg}")
    else:
        failures += 1
        print(f"[FAIL] {msg}")


k, refs = curriculum.build_base_kernel()
br = bn.setup_binary_nat(k, refs)

N_EXHAUSTIVE = 200  # pairs (x, y) with 0 <= x, y <= N_EXHAUSTIVE: 40,401 additions, each independently checked

# --- round trip: int -> term -> int, for every value 0..N_EXHAUSTIVE*4 (covers int_to_pos/pos_to_int
# and int_to_n/n_to_int themselves, since every other check below depends on both being correct) ---
round_trip_ok = True
for v in range(0, N_EXHAUSTIVE * 4 + 1):
    got = bn.n_to_int(k, bn.int_to_n(k, br, v))
    if got != v:
        round_trip_ok = False
        print(f"   round-trip mismatch: {v} -> term -> {got}")
        break
check(round_trip_ok, f"int_to_n/n_to_int round-trip is exact for every value 0..{N_EXHAUSTIVE*4}")

# --- Pos.succ, exhaustive ---
succ_ok = True
for v in range(1, N_EXHAUSTIVE + 1):
    term = k.mk_app(br['pos_succ_c'], bn.int_to_pos(k, br, v))
    got = bn.pos_to_int(k, k.normalize_node(term))
    if got != v + 1:
        succ_ok = False
        print(f"   Pos.succ({v}) = {got}, expected {v + 1}")
        break
check(succ_ok, f"Pos.succ is exact for every value 1..{N_EXHAUSTIVE}")

# --- N.add, exhaustive over the full (x, y) grid: this exercises Pos.add AND Pos.add_carry
# together (every branch of both functions' 3x3 case tables is hit many times over this range) ---
add_ok = True
bad = None
for x in range(0, N_EXHAUSTIVE + 1):
    tx = bn.int_to_n(k, br, x)
    for y in range(0, N_EXHAUSTIVE + 1):
        ty = bn.int_to_n(k, br, y)
        term = k.mk_app(k.mk_app(br['n_add_c'], tx), ty)
        got = bn.n_to_int(k, k.normalize_node(term))
        if got != x + y:
            add_ok = False
            bad = (x, y, got)
            break
    if not add_ok:
        break
check(add_ok, f"N.add(x, y) == x + y for EVERY pair 0 <= x, y <= {N_EXHAUSTIVE} "
              f"({(N_EXHAUSTIVE + 1) ** 2:,} additions)" + (f" -- first mismatch {bad}" if bad else ""))

# --- specific hand-picked cases that exercise Pos.add_carry's OWN recursive call to itself
# (the deepest, most failure-prone path: XI,XI -> add_carry(p,q) at every level of a long
# all-ones-bits chain, e.g. 2^n - 1 + 2^n - 1) ---
carry_chain_ok = True
for n_bits in range(1, 12):
    x = (1 << n_bits) - 1  # all n_bits ones
    term = k.mk_app(k.mk_app(br['n_add_c'], bn.int_to_n(k, br, x)), bn.int_to_n(k, br, x))
    got = bn.n_to_int(k, k.normalize_node(term))
    if got != 2 * x:
        carry_chain_ok = False
        print(f"   (2^{n_bits}-1) + (2^{n_bits}-1): got {got}, expected {2*x}")
check(carry_chain_ok, "long carry-propagation chains (all-ones bit patterns, up to 11 bits) are exact")

# --- N.add is commutative and has N0 as identity, spot-checked over random larger values ---
import random
rng = random.Random(0)
comm_ok = True
for _ in range(500):
    x = rng.randrange(0, 2**40)
    y = rng.randrange(0, 2**40)
    t1 = bn.n_to_int(k, k.normalize_node(k.mk_app(k.mk_app(br['n_add_c'], bn.int_to_n(k, br, x)), bn.int_to_n(k, br, y))))
    t2 = bn.n_to_int(k, k.normalize_node(k.mk_app(k.mk_app(br['n_add_c'], bn.int_to_n(k, br, y)), bn.int_to_n(k, br, x))))
    if t1 != x + y or t2 != x + y:
        comm_ok = False
        break
check(comm_ok, "500 random pairs up to 2^40: N.add(x,y) == N.add(y,x) == x+y (kernel-checked commutativity spot check)")

print()
print("ALL BINARY_NAT TESTS PASSED" if failures == 0 else f"{failures} BINARY_NAT TEST(S) FAILED")
sys.exit(1 if failures else 0)
