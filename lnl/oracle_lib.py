"""Shared well-formedness helpers for the hybrid-kernel oracles."""
from harness import *

def is_type(k, tree, ctxs):
    node = build_new(k, tree)
    return any(k.check_partial(node, build_new(k, ('type', lvl)), ctxs)[0] for lvl in range(4))

def metas_in(tree, out):
    if isinstance(tree, tuple):
        if len(tree) == 2 and tree[0] == 'meta': out.add(tree[1]); return
        for y in tree: metas_in(y, out)

def goals_meta_free(k, ps):
    """True when no goal's hypotheses/target mention an UNSOLVED meta, directly or behind solved ones. A partial proof is only well-typed
    relative to the eventual solutions of such metas, and the real unifier would solve them as a side effect of checking."""
    ids = set()
    for i in range(ps.num_goals()):
        metas_in(parse_raw(k.dump_raw(ps.goal_target(i)), True), ids)
        for j in range(ps.goal_num_hyps(i)): metas_in(parse_raw(k.dump_raw(ps.goal_hyp_type(i, j)), True), ids)
    seen = set(); work = list(ids)
    while work:
        m = work.pop()
        if m in seen: continue
        seen.add(m)
        if not k.meta_is_solved(m): return False
        inner = set(); metas_in(parse_raw(k.meta_solution_raw(m), True), inner); work.extend(inner - seen)
    return True

def goals_wellformed(k, ps):
    """every goal's hypotheses are types in their prefix context and the target is a type in the full context"""
    for i in range(ps.num_goals()):
        nh = ps.goal_num_hyps(i); ctxs = []
        for j in range(nh):
            tr = parse_raw(k.dump_raw(ps.goal_hyp_type(i, j)), True)
            if not is_type(k, tr, ctxs): return False
            ctxs.append(build_new(k, tr))
        if not is_type(k, parse_raw(k.dump_raw(ps.goal_target(i)), True), ctxs): return False
    return True

def is_mode2(h, g, k=None):
    """a hypothesis type whose index arguments are all distinct bare context variables (generic indices), or whose family has no indices.
    Anything else is a concrete-index induction (F12): the generic Fix cannot supply the specialised binder types of the branch goals.
    With the kernel `k` in parameter mode, PARAMETER slots may hold anything (they are fixed), only the index slots must be distinct variables."""
    args = []
    while h[0] == 'app': args.append(h[2]); h = h[1]
    args = args[::-1]
    if k is not None and k.get_option('parameters') and h[0] == 'const':
        slots = k.param_slots(h[1]); idxs = [a for i, a in enumerate(args) if not (i < len(slots) and slots[i])]
    else: idxs = args if h[1] != g['Nat'] else []
    return all(a[0] == 'fvar' for a in idxs) and len(set(idxs)) == len(idxs)
