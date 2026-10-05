import itp_env
import copy

failures = 0
def check(cond, label):
    global failures
    if cond:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        failures += 1

kA = itp_env.Kernel()
type0 = kA.mk_type(0)
nat_id = kA.reserve_inductive_type('Nat', type0)
nat_ty = kA.mk_const(nat_id)
nat_h = kA.finish_inductive(nat_id, [itp_env.CtorSpec('Zero', []), itp_env.CtorSpec('Succ', [nat_ty])])
zero = kA.mk_const(nat_h.ctor_global_ids[0])
succ_c = kA.mk_const(nat_h.ctor_global_ids[1])

add_fn_type = kA.mk_pi(nat_ty, kA.mk_pi(nat_ty, nat_ty))
add_gid = kA.reserve_recursive_function('add', add_fn_type)
add_c = kA.mk_const(add_gid)
add_motive = kA.mk_lambda(nat_ty, nat_ty)
add_zero_branch = kA.mk_var(1)
add_succ_branch = kA.mk_lambda(nat_ty, kA.mk_app(succ_c, kA.mk_app(kA.mk_app(add_c, kA.mk_var(2)), kA.mk_var(1))))
add_match = kA.mk_match(kA.mk_var(0), add_motive, [add_zero_branch, add_succ_branch])
add_body = kA.mk_lambda(nat_ty, kA.mk_lambda(nat_ty, add_match))
kA.finish_recursive_function_checked(add_gid, nat_id, 0, add_body)
print("[setup] add declared with genuinely structurally-decreasing recursion")

events = kA.export_library()

# Tamper: in the Succ branch, replace the recursive call's own first
# argument -- currently the structurally-smaller, directly-matched
# field (mk_var(2), the "n'" bound by the Succ branch's own lambda) --
# with the OUTER, non-decreasing n (mk_var(0), the original scrutinee)
# instead. Still type-checks (both are Nat), but is no longer
# structurally decreasing -- this is the exact tampering shape
# is_structurally_decreasing exists to catch.
tampered = copy.deepcopy(events)
for e in tampered:
    if e.get('name') == 'add':
        # body = lambda(nat, lambda(nat, match(...))); match's own
        # branches[1] (Succ case) = lambda(nat, app(succ, app(app(add, VAR), var1)))
        succ_branch = e['body']['body']['body']['branches'][1]
        inner_app = succ_branch['body']['arg']  # app(app(add, VAR), var1)
        recursive_call_first_arg = inner_app['fn']['arg']
        print(f"  original recursive-call first arg: {recursive_call_first_arg}")
        # Replace level 2 (the smaller n') with level 0 -- but level 0
        # INSIDE the succ branch's own lambda no longer refers to the
        # outer n at all (it's now the branch's own bound n' itself,
        # differently). Simplest genuine tampering: replace with a
        # FRESH reference to the branch's own outer "n" via the actual
        # outer scrutinee level as seen from THIS depth -- level 0 here
        # is the succ branch's own n' (same thing we're replacing FROM).
        # Use level appropriate for "the original, outer n" instead:
        # within the match's own branch (nested inside 2 lambdas: add's
        # own n,m, then the branch's own n'), the ORIGINAL scrutinee is
        # unreachable as itself (it's what's being matched), so instead
        # substitute the STRUCTURALLY LARGER "Succ n'" -- still type-Nat,
        # never smaller.
        inner_app['fn']['arg'] = {'t': 'app', 'fn': {'t': 'const', 'name': 'Succ'}, 'arg': recursive_call_first_arg}
        break

kB = itp_env.Kernel()
try:
    kB.import_library(tampered)
    check(False, "import correctly rejects a recursive function whose tampered body no longer structurally decreases")
except itp_env.ITPError as ex:
    check('structural recursion' in str(ex), "import correctly rejects a recursive function whose tampered body no longer structurally decreases")
    print(f"  (rejected with: {str(ex)[:90]})")

# Control: the untampered export still imports and the guardedness
# check genuinely passes for it.
kB2 = itp_env.Kernel()
kB2.import_library(events)
check(kB2.library_size() == kA.library_size(), "control: the genuinely-decreasing, untampered add still imports successfully")

if failures == 0:
    print("\nALL GUARDEDNESS RE-VERIFICATION CHECKS PASSED")
else:
    print(f"\n{failures} CHECK(S) FAILED")
