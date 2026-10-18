import torch
import torch.nn as nn
from model import TermEncoder

failures = 0
def check(cond, label):
    global failures
    if cond:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        failures += 1

# --- Synthetic dump: a small, hand-built term mimicking real dump
# shapes, WITHOUT needing the actual kernel at all. Represents
# App(App(Const(gid=42), Var(level=3)), Type(level=0)) -- 4 entries,
# with App (root) at index 0 (parent-first, matching the confirmed
# real ordering), covering Var/Const/Type/App in one term so every
# scalar-featurization path is genuinely exercised.
tags = [1, 1, 5, 0, 4]        # App, App, Const, Var, Type
scalar = [0, 0, 42, 3, 0]
child_a = [1, 2, -1, -1, -1]  # outer App's fn = inner App; inner App's fn = Const
child_b = [4, 3, -1, -1, -1]  # outer App's arg = Type; inner App's arg = Var

enc = TermEncoder(embed_dim=16)

out = enc(tags, scalar, child_a, child_b)
check(out.shape == (5, 16), f"output shape is (5, 16), got {tuple(out.shape)}")
check(not torch.isnan(out).any(), "no NaNs in output")

# --- Gradient flow: every parameter should receive a gradient after
# backprop through a simple loss on the output. ---
loss = out.sum()
loss.backward()
no_grad_params = [name for name, p in enc.named_parameters() if p.grad is None]
check(len(no_grad_params) == 0, f"every parameter receives a gradient (missing: {no_grad_params})")
all_nonzero_grad = all(p.grad.abs().sum().item() > 0 for p in enc.parameters() if p.grad is not None)
check(all_nonzero_grad, "every parameter's own gradient is genuinely nonzero (not just present but dead)")

# --- Overfitting check: can this tiny model learn to map a FIXED,
# synthetic term to a FIXED, arbitrary target vector, given enough
# training steps? This is the standard "is the wiring correct at all"
# sanity check, independent of whether real proof-search data is
# learnable. ---
enc2 = TermEncoder(embed_dim=8)
target = torch.randn(8)
opt = torch.optim.Adam(enc2.parameters(), lr=0.01)
losses = []
for step in range(300):
    opt.zero_grad()
    out2 = enc2(tags, scalar, child_a, child_b)
    root_embed = out2[0]  # root is at index 0
    loss2 = ((root_embed - target) ** 2).sum()
    loss2.backward()
    opt.step()
    losses.append(loss2.item())

print(f"overfit loss: start={losses[0]:.4f}, end={losses[-1]:.6f}")
check(losses[-1] < losses[0] * 0.01, "loss drops by at least 100x over 300 steps -- the model genuinely learns")
check(losses[-1] < 1e-3, "final loss is genuinely near zero, confirming the tiny synthetic target was learned")

print()
if failures == 0:
    print("ALL TermEncoder SANITY CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
