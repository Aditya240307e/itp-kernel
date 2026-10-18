import torch
import itp_env
import hypergraph as hg
from model import PolicyValueModel
import training

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
bool_id = k.reserve_inductive_type('Bool', type0)
bool_ty = k.mk_const(bool_id)
bool_h = k.finish_inductive(bool_id, [itp_env.CtorSpec('True', []), itp_env.CtorSpec('False', [])])
eq_sort = k.mk_pi(type0, k.mk_pi(k.mk_var(0), k.mk_pi(k.mk_var(0), prop0)))
eq_id = k.reserve_inductive_type('Eq', eq_sort)
eq_ty = k.mk_const(eq_id)
eq_h = k.finish_inductive(eq_id, [itp_env.CtorSpec('refl', [type0, k.mk_var(0)], [k.mk_var(0), k.mk_var(1), k.mk_var(1)])])

library = k.index_library()
goal_ty = k.mk_pi(bool_ty, k.mk_app(k.mk_app(k.mk_app(eq_ty, bool_ty), k.mk_var(0)), k.mk_var(0)))
ps = k.start_proof(goal_ty)
dedup_map = {}
root = hg.get_or_create_node(k, ps, 0, dedup_map)
policy = hg.UniformPolicy()
status = hg.search(k, root, library, policy, dedup_map, max_simulations=200, top_k=16)
check(status == hg.Status.SOLVED, "sanity: search solves the Bool goal")

policy_examples, value_examples = hg.extract_training_examples(k, dedup_map)
print(f"policy examples: {len(policy_examples)}, value examples: {len(value_examples)}")

model = PolicyValueModel(embed_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Single train_step: confirm it runs, produces sane loss values,
# and genuinely updates parameters. ---
params_before = [p.clone() for p in model.parameters()]
result = training.train_step(model, optimizer, policy_examples, value_examples)
print("single train_step result:", result)
check(result["policy_loss"] >= 0, "policy loss is non-negative")
check(result["value_loss"] >= 0, "value loss is non-negative")
params_changed = any(not torch.equal(a, b) for a, b in zip(params_before, model.parameters()))
check(params_changed, "at least one parameter genuinely changed after one train_step")

# --- Repeated steps: loss should trend down on this small, fixed
# batch (an overfitting sanity check -- not a claim about real
# generalization, just that the training loop is wired correctly). ---
model2 = PolicyValueModel(embed_dim=32)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.02)
losses = []
for step in range(60):
    r = training.train_step(model2, optimizer2, policy_examples, value_examples)
    losses.append(r["total_loss"])
print(f"loss trajectory: start={losses[0]:.4f}, end={losses[-1]:.4f}")
check(losses[-1] < losses[0], "total loss decreases over 60 steps on this small, fixed batch")
check(losses[-1] < losses[0] * 0.5, "loss drops by at least half -- genuine learning, not noise")

print()
if failures == 0:
    print("ALL train_step CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
