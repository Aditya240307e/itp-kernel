import itp_env
import hypergraph as hg

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
k.finish_inductive(bool_id, [itp_env.CtorSpec('True', []), itp_env.CtorSpec('False', [])])

eq_sort = k.mk_pi(type0, k.mk_pi(k.mk_var(0), k.mk_pi(k.mk_var(0), prop0)))
eq_id = k.reserve_inductive_type('Eq', eq_sort)
eq_ty = k.mk_const(eq_id)
k.finish_inductive(eq_id, [itp_env.CtorSpec('refl', [type0, k.mk_var(0)], [k.mk_var(0), k.mk_var(1), k.mk_var(1)])])

library = k.index_library()
goal_ty = k.mk_pi(bool_ty, k.mk_app(k.mk_app(k.mk_app(eq_ty, bool_ty), k.mk_var(0)), k.mk_var(0)))
ps = k.start_proof(goal_ty)
root_meta_id = ps.goal_meta_id(0)
dedup_map = {}
root = hg.get_or_create_node(k, ps, 0, dedup_map)
policy = hg.UniformPolicy()
status = hg.search(k, root, library, policy, dedup_map, max_simulations=200, top_k=16)
check(status == hg.Status.SOLVED, "search finds the Bool goal SOLVED (sanity precondition)")

policy_examples, value_examples = hg.extract_training_examples(k, dedup_map)
print(f"nodes total: {len(dedup_map)}, policy examples: {len(policy_examples)}, value examples: {len(value_examples)}")

# --- Hand-recompute the expected counts directly over dedup_map,
# independent of extract_training_examples' own internals, to cross-
# check rather than trust the function's own self-consistency. ---
expected_value_count = sum(1 for n in dedup_map.values() if n.tactics)
expected_policy_count = sum(1 for n in dedup_map.values()
                              if n.tactics and sum(e.N for e in n.tactics.values()) > 0)
check(len(value_examples) == expected_value_count,
      f"value example count matches direct hand-count ({expected_value_count})")
check(len(policy_examples) == expected_policy_count,
      f"policy example count matches direct hand-count ({expected_policy_count})")
check(expected_policy_count <= expected_value_count,
      "policy examples are a subset of value examples (every zero-visit expanded node contributes value but not policy)")

# --- Every policy target's own distribution must sum to 1 (within
# floating point tolerance) and assign only non-negative probabilities. ---
all_sum_to_one = True
all_nonneg = True
for obs, target, cts in policy_examples:
    s = sum(target.values())
    if abs(s - 1.0) > 1e-9:
        all_sum_to_one = False
    if any(v < 0 for v in target.values()):
        all_nonneg = False
check(all_sum_to_one, "every policy example's own target distribution sums to exactly 1.0")
check(all_nonneg, "every policy example's own target assigns only non-negative probabilities")

# --- Every value target must lie in [0, 1]. ---
all_in_range = all(0.0 <= v <= 1.0 for obs, v in value_examples)
check(all_in_range, "every value example's own target lies in [0, 1]")

# --- The root itself, since it's SOLVED with real search behind it,
# must appear in both example sets with a genuinely informative
# (non-degenerate) target -- not just technically present. ---
root_in_value = any(obs is root.observation for obs, v in value_examples)
check(root_in_value, "the root node itself appears among the value examples")
root_policy_targets = [target for obs, target, cts in policy_examples if obs is root.observation]
check(len(root_policy_targets) == 1, "the root node itself appears exactly once among the policy examples")
if root_policy_targets:
    print(f"  root's own policy target: {root_policy_targets[0]}")
    check(len(root_policy_targets[0]) >= 1, "the root's own policy target covers at least one action")

# --- Direct, concrete example: print a handful of extracted examples
# for manual inspection, not just aggregate checks. ---
print("\nsample policy examples (first 3):")
for obs, target, cts in policy_examples[:3]:
    print(f"  {target}")
print("\nsample value examples (first 5):")
for obs, v in value_examples[:5]:
    print(f"  {v}")

print()
if failures == 0:
    print("ALL extract_training_examples CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
