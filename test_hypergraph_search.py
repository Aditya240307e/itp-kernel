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
ps = k.start_proof(goal_ty)
root_meta_id = ps.goal_meta_id(0)

dedup_map = {}
root = hg.get_or_create_node(k, ps, 0, dedup_map)
policy = hg.UniformPolicy()

final_status = hg.search(k, root, library, policy, dedup_map, max_simulations=100, top_k=16)
print(f"final root status: {final_status.name}")
check(final_status == hg.Status.SOLVED, "search finds root SOLVED within budget")
check(ps.num_goals() == 1 and ps.goal_meta_id(0) == root_meta_id,
      "the ORIGINAL ps is STILL completely untouched after the whole search (only commit_solution should ever touch it)")

# --- Internal consistency: every solved node's own status is backed by
# a genuine, recursively-solved edge. ---
def check_solved_invariant(node, seen):
    if id(node) in seen:
        return True
    seen.add(id(node))
    if node.status != hg.Status.SOLVED:
        return True
    return any(all(check_solved_invariant(c, seen) and c.status == hg.Status.SOLVED for c in e.children)
               for e in node.tactics.values())

check(check_solved_invariant(root, set()), "SOLVED status is internally consistent (recursively, through the hypergraph)")

# --- The decisive check: commit the search's own solution to the REAL
# ps, and confirm the kernel independently agrees it's a genuine proof. ---
hg.commit_solution(k, ps, root, root_meta_id)
check(ps.is_complete(), "committing the search's own solution genuinely completes the kernel-side proof")

term = ps.extract()
k.whole_term_check(term, goal_ty)
print("[check] whole-term check() of the HYPERGRAPH-SEARCH-DISCOVERED proof: PASSED")

# --- The property the whole live_meta_id redesign exists for: build a
# SEPARATE, second search on a goal with genuinely multiple viable
# tactics from the SAME node, and confirm expand() having already run
# doesn't prevent exploring more than one of its discovered edges. ---
ps2 = k.start_proof(goal_ty)
root2_meta_id = ps2.goal_meta_id(0)
dedup_map2 = {}
root2 = hg.get_or_create_node(k, ps2, 0, dedup_map2)
hg.expand(k, root2, library, policy, dedup_map2, top_k=16)
check(len(root2.tactics) >= 1, "root2 has at least one discovered edge")
intro_action = hg.Action(hg.ActionKind.INTRO, 0)
edge2 = root2.tactics[intro_action]
child2 = edge2.children[0]
hg.expand(k, child2, library, policy, dedup_map2, top_k=16)
# child2 (b:Bool |- Eq Bool b b) should have discovered BOTH cases(0)
# AND induct(0) as SEPARATE, coexisting edges from the SAME node.
cases_action = hg.Action(hg.ActionKind.CASES, 0)
induct_action = hg.Action(hg.ActionKind.INDUCT, 0)
check(cases_action in child2.tactics and induct_action in child2.tactics,
      "the SAME node has BOTH cases(0) and induct(0) as separate, independently-discovered edges")
# Now genuinely commit BOTH, on two SEPARATE, independent proof
# attempts, confirming NEITHER consumes anything the other needs.
if cases_action in child2.tactics:
    cases_edge = child2.tactics[cases_action]
    if all(c.status != hg.Status.INVALID for c in cases_edge.children):
        # cases(0) alone doesn't close the goal (still need refl on
        # each branch) -- just confirm materialize_from_node can
        # commit this SPECIFIC edge for real, independent of the
        # OTHER edge (induct) also being present and untouched.
        fresh_fork = k.materialize_from_node(child2.target, child2.hyp_types, child2.hyp_names)
        fresh_fork.cases(0, target_meta_id=fresh_fork.root_meta_id)
        check(fresh_fork.num_goals() == 2, "cases(0) can be genuinely, independently re-applied to child2's own stored data")
if induct_action in child2.tactics:
    fresh_fork2 = k.materialize_from_node(child2.target, child2.hyp_types, child2.hyp_names)
    fresh_fork2.induct(0, target_meta_id=fresh_fork2.root_meta_id)
    check(fresh_fork2.num_goals() >= 1, "induct(0) can ALSO be independently, genuinely re-applied to the SAME child2 -- neither edge consumed the goal for the other")

print()
if failures == 0:
    print("ALL HYPERGRAPH SEARCH CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
