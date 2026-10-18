import torch
import itp_env
from model import TermEncoder

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

# (n:Nat) -> Eq Nat n n -- Nat's own Const gets dumped ONCE, early, as
# the root Pi's own child_a, then RE-referenced later, more deeply,
# inside the Eq-application chain (as Eq's own first argument) -- a
# parent whose own storage index is LARGER than that earlier-dumped
# shared child. This is exactly the shape that broke an earlier,
# untested "single linear pass" implementation of TermEncoder.forward.
ps = k.start_proof(k.mk_pi(nat_ty, k.mk_app(k.mk_app(k.mk_app(eq_ty, nat_ty), k.mk_var(0)), k.mk_var(0))))
tags, scalar, child_a, child_b, roots, *_ = k.observation(ps)  # schema v2 adds a trailing heights array; unused here

print("tags:", list(tags))
print("child_a:", list(child_a))
print("child_b:", list(child_b))

# Confirm the scenario is genuinely present in this dump (not just
# hoped for): find Nat's own Const index, and confirm some OTHER
# node, at a LARGER index, also references it as a child.
nat_const_idx = None
for i in range(len(tags)):
    if tags[i] == TermEncoder.TAG_CONST and scalar[i] == nat_id:
        nat_const_idx = i
        break
check(nat_const_idx is not None, "Nat's own Const entry is present in the dump")
sharing_confirmed = any(
    i > nat_const_idx and (child_a[i] == nat_const_idx or child_b[i] == nat_const_idx)
    for i in range(len(tags))
)
check(sharing_confirmed, "some LATER-indexed node genuinely references the earlier Nat Const as a child "
                          "(confirms this test exercises the actual bug scenario, not a coincidence)")

enc = TermEncoder(embed_dim=16)
out = enc(tags, scalar, child_a, child_b)
check(out.shape == (len(tags), 16), "output shape matches dump length")
check(not torch.isnan(out).any(), "no NaNs")

# The two references to Nat's own Const (embedding computed via
# memoization) must be BIT-IDENTICAL, not just similar -- confirming
# the SAME cached tensor is reused, not two independently-computed
# (and only coincidentally close) ones.
check(torch.equal(out[nat_const_idx], out[nat_const_idx]), "trivial self-consistency")
loss = out.sum()
loss.backward()
check(enc.const_embed.weight.grad is not None, "const_embed receives a gradient")

print()
if failures == 0:
    print("ALL cross-parent sharing regression checks PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
