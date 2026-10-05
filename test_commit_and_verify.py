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
check(status == hg.Status.SOLVED, "search finds the Bool goal SOLVED")

term = hg.commit_and_verify(k, ps, root, root_meta_id, goal_ty)
check(term is not None, "commit_and_verify returns a term")
check(ps.is_complete(), "the underlying ps is genuinely complete after commit_and_verify")
# whole_term_check already ran INSIDE commit_and_verify -- confirm it
# genuinely, independently checks out again here too, as an external
# sanity cross-check that the returned term really is what it claims.
k.whole_term_check(term, goal_ty)
print("[PASS] independently re-checking the returned term succeeds too")

# --- Negative case: commit_and_verify must propagate a genuine
# rejection, not swallow it. Build a hypergraph by hand where a node
# is (incorrectly) marked SOLVED via an edge that doesn't actually
# check out, and confirm commit_and_verify raises rather than
# returning silently. ---
ps2 = k.start_proof(goal_ty)
root_meta_id2 = ps2.goal_meta_id(0)
dedup_map2 = {}
root2 = hg.get_or_create_node(k, ps2, 0, dedup_map2)
hg.expand(k, root2, library, policy, dedup_map2, top_k=16)
intro_action = hg.Action(hg.ActionKind.INTRO, 0)
child = root2.tactics[intro_action].children[0]
# Manually, incorrectly force child to look SOLVED without any actual
# tactic ever having closed it -- an artificial inconsistency, to
# confirm commit_and_verify's own final check is the thing that
# actually catches a bad state, not just trusting node.status.
child.status = hg.Status.SOLVED
child.cached_value = 1.0
child.tactics[hg.Action(hg.ActionKind.APPLY, 0)] = hg.HyperEdge(prior=1.0, children=[])
child.cached_candidates = [k.mk_var(999)]  # a bogus, out-of-scope candidate that cannot possibly apply
hg.update_status(root2)

try:
    hg.commit_and_verify(k, ps2, root2, root_meta_id2, goal_ty)
    check(False, "commit_and_verify must NOT silently succeed on an artificially-forced, invalid SOLVED state")
except itp_env.ITPError as e:
    check(True, "commit_and_verify correctly raises on an invalid state rather than returning silently")
    print(f"  (raised: {str(e)[:80]})")

print()
if failures == 0:
    print("ALL commit_and_verify CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
