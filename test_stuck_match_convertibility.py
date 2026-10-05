"""test_stuck_match_convertibility.py -- regression coverage for the conv_value/unify_values fix
to a genuine completeness gap (never a soundness one) in how two independently-reconstructed stuck
Match values get compared.

Root cause (found by direct diagnosis, not assumed -- see this session's own transcript for the
full derivation): check()'s own infer()-then-unify() fallback does not substitute an applied
argument as raw, unevaluated syntax -- it evaluates the argument to a Value and quotes that VALUE
back into the reconstructed type. For an argument whose own computation is itself stuck on a DEEPER
stuck computation (e.g. add's own Match stuck on mul's own, already-stuck result -- two levels, not
one), this produces a semantically-equal but differently-shaped term: one side's outer stuck Match
has its own captured_env holding the inner stuck value as a genuinely bound element (reached via a
real Lambda-application step); the other, quoted-back side's outer Match has no such binding step,
so its own captured_env just carries the flat ambient context straight through. The OLD fast path
compared captured_env pointwise and hard-failed on this representational mismatch, even though the
two sides are genuinely, definitionally equal (confirmed directly: normalize_in_context on each
independently yields the identical NodeId).

The fix (in both conv_value's and unify_values' own, "mirrored" VNeutral cases): when the
frame-by-frame structural spine walk fails anywhere, fall back to quoting BOTH full sides once and
comparing the resulting NodeId for equality, rather than hard-failing immediately. This is
deliberately NOT the same shape as the pre-existing captured_env-size-mismatch fallback
(branches_convertible): an earlier attempt tried exactly that and was found UNSAFE by a direct A/B
test -- branches_convertible mutually recurses with conv_value, and for two na/nb that are
genuinely, persistently different stuck applications of the SAME recursive function (an entirely
ordinary situation during search, not a rare edge case), every nested
branches_convertible->conv_value->(another stuck MatchFrame, same branches)->branches_convertible...
step still passes the branches==branches termination guard trivially, so the mutual recursion can
unwind arbitrarily deep before ever reaching a frame kind that lets it correctly return false --
confirmed directly: this segfaulted on an ordinary, unremarkable UniformPolicy search step at a
simulation budget of 50. quote() has no such risk: it is a ONE-SIDED, non-mutually-recursive
reconstruction, bounded by each side's own finite size, with no comparison-driven regress possible.
"""
import time

import itp_env
import curriculum as cur

failures = 0


def check(cond, msg):
    global failures
    if cond:
        print(f"[PASS] {msg}")
    else:
        failures += 1
        print(f"[FAIL] {msg}")


def build_mul(k, refs):
    """mul : Nat -> Nat -> Nat, recursing on the first argument -- mirrors setup_mul's own
    construction exactly, built standalone here so this file has no dependency on setup_add_comm/
    setup_mul's own, much larger prerequisite chain."""
    nat_ty = refs['nat_ty']; add_c = refs['add_c']
    mul_fn_type = k.mk_pi(nat_ty, k.mk_pi(nat_ty, nat_ty))
    mul_gid = k.reserve_recursive_function('mul', mul_fn_type)
    mul_c = k.mk_const(mul_gid)
    mul_motive = k.mk_lambda(nat_ty, nat_ty)
    mul_zero_branch = refs['zero']
    mul_succ_branch = k.mk_lambda(nat_ty,
        k.mk_app(k.mk_app(add_c, k.mk_app(k.mk_app(mul_c, k.mk_var(2)), k.mk_var(1))), k.mk_var(1)))
    mul_match = k.mk_match(k.mk_var(0), mul_motive, [mul_zero_branch, mul_succ_branch])
    mul_body = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty, mul_match))
    k.finish_recursive_function_checked(mul_gid, refs['nat_id'], 0, mul_body)
    return mul_c


k, refs = cur.build_base_kernel()
nat_ty = refs['nat_ty']; zero = refs['zero']; succ_c = refs['succ_c']; add_c = refs['add_c']
mul_c = build_mul(k, refs)


def mul_app(a, b):
    return k.mk_app(k.mk_app(mul_c, a), b)


def add_app(a, b):
    return k.mk_app(k.mk_app(add_c, a), b)


def eq2(a, x, y):
    return k.mk_app(k.mk_app(k.mk_app(refs['eq_ty'], a), x), y)


def refl_check(lhs, rhs, hyp_count=2):
    """Builds Pi^hyp_count. Eq Nat lhs rhs and checks it via refl(Nat, lhs) applied as the proof
    term -- the exact shape (a doubly-stuck computational fact proven purely by refl) that exposed
    the bug."""
    full_ty = eq2(nat_ty, lhs, rhs)
    term = k.mk_app(k.mk_app(refs['refl_c'], nat_ty), lhs)
    for _ in range(hyp_count):
        full_ty = k.mk_pi(nat_ty, full_ty)
        term = k.mk_lambda(nat_ty, term)
    k.whole_term_check(term, full_ty)


v0, v1 = k.mk_var(0), k.mk_var(1)

# --- Control: single-level stuck recursion (no inner stuck call) must already have worked, and
# must continue to. ---
try:
    refl_check(mul_app(v0, v1), mul_app(v0, v1))
    check(True, "single-level stuck mul(v0,v1) vs itself: refl-checks correctly")
except Exception as e:
    check(False, f"single-level stuck mul(v0,v1) vs itself: {e}")

# --- The original bug: doubly-nested stuck recursion (mul's own stuck recursive call feeding into
# add's own, also-stuck Match), both sides abstract. ---
lhs2 = mul_app(k.mk_app(succ_c, v0), v1)
rhs2 = add_app(mul_app(v0, v1), v1)
try:
    refl_check(lhs2, rhs2)
    check(True, "doubly-nested mul(Succ v0,v1) =?= add(mul(v0,v1),v1): refl-checks correctly")
except Exception as e:
    check(False, f"doubly-nested mul(Succ v0,v1) =?= add(mul(v0,v1),v1): {e}")

# --- Depth stress: triple- and quadruple-nested compositions must terminate promptly (not hang,
# not segfault) and still verify correctly. ---
lhs3 = mul_app(k.mk_app(succ_c, k.mk_app(succ_c, v0)), v1)
rhs3 = add_app(add_app(mul_app(v0, v1), v1), v1)
t0 = time.time()
try:
    refl_check(lhs3, rhs3)
    dt = time.time() - t0
    check(dt < 5.0, f"triple-nested (Succ^2) check: verified promptly ({dt:.3f}s)")
except Exception as e:
    check(False, f"triple-nested (Succ^2) check: {e}")

lhs4 = mul_app(k.mk_app(succ_c, k.mk_app(succ_c, k.mk_app(succ_c, v0))), v1)
rhs4 = add_app(add_app(add_app(mul_app(v0, v1), v1), v1), v1)
t0 = time.time()
try:
    refl_check(lhs4, rhs4)
    dt = time.time() - t0
    check(dt < 5.0, f"quadruple-nested (Succ^3) check: verified promptly ({dt:.3f}s)")
except Exception as e:
    check(False, f"quadruple-nested (Succ^3) check: {e}")

# --- Soundness negatives: genuinely FALSE doubly-nested statements must still be correctly
# rejected -- the fix must not make the checker more permissive than it should be. ---
rhs_wrong_swap = add_app(mul_app(v1, v0), v1)  # v0/v1 swapped inside mul -- wrong in general
try:
    refl_check(lhs2, rhs_wrong_swap)
    check(False, "NEGATIVE (v0/v1 swapped): incorrectly ACCEPTED -- soundness regression")
except itp_env.ITPError:
    check(True, "NEGATIVE (v0/v1 swapped): correctly rejected")

rhs_wrong_succ = k.mk_app(succ_c, add_app(mul_app(v0, v1), v1))  # spurious extra Succ
try:
    refl_check(lhs2, rhs_wrong_succ)
    check(False, "NEGATIVE (extra Succ): incorrectly ACCEPTED -- soundness regression")
except itp_env.ITPError:
    check(True, "NEGATIVE (extra Succ): correctly rejected")

try:
    refl_check(lhs2, zero)  # totally unrelated RHS
    check(False, "NEGATIVE (unrelated Zero): incorrectly ACCEPTED -- soundness regression")
except itp_env.ITPError:
    check(True, "NEGATIVE (unrelated Zero): correctly rejected")

if failures == 0:
    print("\nALL STUCK-MATCH CONVERTIBILITY CHECKS PASSED")
else:
    print(f"\n{failures} CHECK(S) FAILED")
