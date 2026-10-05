import random
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
bool_h = k.finish_inductive(bool_id, [itp_env.CtorSpec('True', []), itp_env.CtorSpec('False', [])])
eq_sort = k.mk_pi(type0, k.mk_pi(k.mk_var(0), k.mk_pi(k.mk_var(0), prop0)))
eq_id = k.reserve_inductive_type('Eq', eq_sort)
eq_ty = k.mk_const(eq_id)
eq_h = k.finish_inductive(eq_id, [itp_env.CtorSpec('refl', [type0, k.mk_var(0)], [k.mk_var(0), k.mk_var(1), k.mk_var(1)])])
library = k.index_library()
goal_ty = k.mk_pi(bool_ty, k.mk_app(k.mk_app(k.mk_app(eq_ty, bool_ty), k.mk_var(0)), k.mk_var(0)))

# --- Default (epsilon=0): completely deterministic, byte-identical
# across repeated runs -- confirming existing callers see NO behavior
# change unless they explicitly opt in. ---
def run_default():
    ps = k.start_proof(goal_ty)
    dedup_map = {}
    root = hg.get_or_create_node(k, ps, 0, dedup_map)
    policy = hg.UniformPolicy()
    hg.search(k, root, library, policy, dedup_map, max_simulations=50, top_k=16)
    return {a: e.prior for a, e in root.tactics.items()}

priors_1 = run_default()
priors_2 = run_default()
check(priors_1 == priors_2, "default (epsilon=0) search is fully deterministic across repeated runs")

# --- A stub policy that gives ONE action a strong, deliberately LOW
# prior (0.0) among several, so we can confirm noise gives it a genuine
# floor probability it would never otherwise get with epsilon=0. ---
class SkewedPolicy(hg.PolicyInterface):
    def predict(self, observation, actions, candidates):
        # First action gets 0.0, everything else shares the rest evenly.
        if not actions:
            return {}, 0.5
        n = len(actions)
        priors = {actions[0]: 0.0}
        for a in actions[1:]:
            priors[a] = 1.0 / max(1, n - 1)
        return priors, 0.5

ps = k.start_proof(goal_ty)
dedup_map = {}
root = hg.get_or_create_node(k, ps, 0, dedup_map)
policy = SkewedPolicy()
hg.search(k, root, library, policy, dedup_map, max_simulations=1, top_k=16, dirichlet_epsilon=0.0)
zero_action = list(root.tactics.keys())[0] if root.tactics else None
# Confirm the skewed policy really does give SOME action a 0.0 prior with no noise.
base = k.materialize_from_node(root.target, root.hyp_types, root.hyp_names)
actions, candidates, _rewrite_extra = hg.enumerate_legal_actions(k, base, base.root_meta_id, library)
priors_no_noise, _ = policy.predict(root.observation, actions, candidates)
check(len(actions) >= 1 and priors_no_noise.get(actions[0], -1) == 0.0,
      "sanity: the stub SkewedPolicy genuinely assigns 0.0 prior to its first action with no noise")

# --- With noise enabled, that same action should get a genuine,
# nonzero floor probability -- run several times (unseeded) and check
# it's ALWAYS nonzero, and that the results differ across runs
# (confirming genuine randomness, not a fixed deterministic tweak). ---
random.seed(1234)  # seed the STDLIB random module only, for a reproducible test run itself
noised_priors_runs = []
for _ in range(5):
    ps2 = k.start_proof(goal_ty)
    dedup_map2 = {}
    root2 = hg.get_or_create_node(k, ps2, 0, dedup_map2)
    hg.search(k, root2, library, policy, dedup_map2, max_simulations=1, top_k=16,
              dirichlet_epsilon=0.5, dirichlet_alpha=0.3)
    noised_priors_runs.append({a: e.prior for a, e in root2.tactics.items()})

first_action_priors = [p.get(actions[0], None) for p in noised_priors_runs if actions[0] in p]
check(len(first_action_priors) == 5, "the previously-zero-prior action now has AN EDGE at all in every run "
                                       "(it's no longer scored so low it never gets discovered)")
check(all(p is not None and p > 0.0 for p in first_action_priors),
      "with noise enabled, the action's own stored prior is genuinely nonzero every time")
check(len(set(first_action_priors)) > 1,
      f"the noised prior differs across repeated (unseeded-per-search) runs, confirming genuine "
      f"randomness rather than a fixed offset: {first_action_priors}")

print()
if failures == 0:
    print("ALL Dirichlet root noise CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
