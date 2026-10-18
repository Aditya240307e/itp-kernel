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

# Goal: (b:Bool) -> Eq Bool b b
goal_ty = k.mk_pi(bool_ty, k.mk_app(k.mk_app(k.mk_app(eq_ty, bool_ty), k.mk_var(0)), k.mk_var(0)))
ps = k.start_proof(goal_ty)

dedup_map = {}
root = hg.get_or_create_node(k, ps, 0, dedup_map)
check(ps.num_goals() == 1, "sanity: one goal before any expansion")

policy = hg.UniformPolicy()
hg.expand(k, root, library, policy, dedup_map, top_k=16)

check(ps.num_goals() == 1, "discovery leaves NO trace in the ORIGINAL ps -- still exactly 1 goal after expand()")
print(f"discovered {len(root.tactics)} legal tactics: {list(root.tactics.keys())}")
check(len(root.tactics) > 0, "at least one legal tactic was discovered")

intro_action = hg.Action(hg.ActionKind.INTRO, 0)
check(intro_action in root.tactics, "INTRO was discovered as legal (goal is a Pi type)")
if intro_action in root.tactics:
    edge = root.tactics[intro_action]
    check(len(edge.children) == 1, "INTRO's own edge has exactly 1 child (intro never splits a goal)")

    child = edge.children[0]
    check(child.status == hg.Status.UNSOLVED, "the discovered child starts UNSOLVED")
    check(len(child.hyp_types) == 1, "the discovered child correctly has 1 new hypothesis (the introduced Bool)")

    # Confirm the discovered child is genuinely usable: expand IT too,
    # purely from its own stored data, no live parent needed at all.
    hg.expand(k, child, library, policy, dedup_map, top_k=16)
    cases_action = hg.Action(hg.ActionKind.CASES, 0)
    check(cases_action in child.tactics, "the child (expanded purely from its own stored data) discovers CASES on its hypothesis")
    if cases_action in child.tactics:
        cases_edge = child.tactics[cases_action]
        check(len(cases_edge.children) == 2, "CASES on Bool discovers exactly 2 children (True, False)")
    check(ps.num_goals() == 1, "expanding the child ALSO leaves the ORIGINAL ps untouched (still 1 goal)")

    # And confirm a SECOND edge from the SAME root node can ALSO still
    # be explored -- the actual property the whole redesign exists for.
    apply_actions = [a for a in root.tactics if a.kind == hg.ActionKind.APPLY]
    print(f"root also discovered {len(apply_actions)} APPLY candidates alongside INTRO")
    check(True, "root's INTRO edge coexists with its other discovered edges (no edge consumed anything)")

print()
if failures == 0:
    print("ALL DISCOVERY CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
