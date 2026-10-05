"""test_induct_pi_ordering.py -- documents and verifies a tactic-sequencing discipline induct()
itself does not enforce: every Pi the eventual induction's own motive will need to abstract over
must already be intro()'d before induct() is called -- never left as a remaining Pi in the goal's
own target.

Root cause (found by direct diagnosis -- see this session's own transcript for the full
derivation): induct()'s own branch_hyps/branch_ec construction is, itself, correct (confirmed
directly: both are extended in lockstep, for every field and every IH, preserving Goal.hyps' own
documented invariant "hyps[i] describes level i" throughout induct()'s own body). The actual
divergence appears one step later, in intro(): a goal's remaining "Pi m. BODY" part, inherited
unchanged from before induct() ever ran (induct()'s own generalize()/substitute_vars machinery only
ever touches the scrutinee's own marker positions, never anything else the target happens to
mention), still has BODY's own reference to "m" hard-coded at whatever level "m" would have occupied
had it been intro()'d immediately, back when the goal was first written -- NOT the (larger) level it
will actually occupy once intro()'d AFTER induct() has already inserted the predecessor and IH ahead
of it. intro() itself assigns the new hypothesis's own level correctly (new_ec.size()); what it
inherits incorrectly is BODY's own stale internal reference to the variable being introduced.

This was never a soundness gap: the established, already-working curriculum (see curriculum.py's
own setup_mul, building mul_succ_right) already follows the correct discipline throughout --
intro() every quantified variable first, then induct() on the specific one wanted, by its own hyp
index. That ordering is also the standard, idiomatic way such proofs are written in mature proof
assistants generally, not a project-specific workaround. Before the guard below existed, violating
the discipline led to a goal containing a stale, wrong-typed reference that an ordinary
type-checking backstop caught -- a dead end, never a route to an incorrectly-accepted proof
(verified directly, both ways -- the colliding position's own type-incompatibility, and a rewrite
pattern built from the naive listing-index assumption being a safe no-op -- in this same session,
before cases()/induct() grew the guard this file now tests; see SESSION_NOTES_2026-10-03.md for
that derivation in full).

cases() and induct() now both reject outright, kernel-side, calling either while the goal's own
target is still a Pi (is_pi_headed(g.target)) -- turning what used to be a silent, several-steps-
downstream dead end into the same kind of clean, instant tactic failure any other inapplicable
action already produces. This matters beyond hand-written curriculum code: hypergraph.py's own
REWRITE_CANDIDATE_FNS and generic Eq-hypothesis path assume listing-index == level when building
Python-side mk_var(h) references, and an RL policy exploring the search tree is otherwise free to
call INDUCT before every variable is introduced, since enumerate_legal_actions offers INDUCT on any
inductive-typed hypothesis already present, regardless of what Pi's remain in the target -- the
guard is what now makes that exploration fail fast and cleanly instead of wasting search budget on
an unreachable goal. This file verifies the GOOD discipline's own alignment invariant, and that the
guard itself fires correctly for both cases() and induct() -- the malformed state the guard
prevents is no longer independently reconstructible through the public API, which is the point.
"""
import itp_env
import curriculum as cur
import hypergraph as hg

failures = 0


def check(cond, msg):
    global failures
    if cond:
        print(f"[PASS] {msg}")
    else:
        failures += 1
        print(f"[FAIL] {msg}")


k, refs = cur.build_base_kernel()
for goal in cur.CURRICULUM:
    lib = k.index_library(); gty = goal.build_type(k, refs)
    ps = k.start_proof(gty); rm = ps.goal_meta_id(0); dm = {}
    root = hg.get_or_create_node(k, ps, 0, dm)
    status = hg.search(k, root, lib, hg.UniformPolicy(), dm, max_simulations=200, top_k=16)
    if status == hg.Status.SOLVED and goal.lemma_name:
        term = hg.commit_and_verify(k, ps, root, rm, gty)
        gid = k.declare_checked(gty, term, goal.lemma_name)
        refs[goal.lemma_name + '_c'] = k.mk_const(gid)
        refs[goal.lemma_name + '_id'] = gid
refs.update(cur.setup_add_comm(k, refs))
refs.update(cur.setup_mul(k, refs))
refs.update(cur.setup_transport(k, refs))

nat_ty = refs['nat_ty']; mul_c = refs['mul_c']


def mul_app(a, b):
    return k.mk_app(k.mk_app(mul_c, a), b)


def build_goal():
    return k.mk_pi(nat_ty, k.mk_pi(nat_ty, cur.eq_app(k, refs, nat_ty,
        mul_app(k.mk_var(0), k.mk_var(1)), mul_app(k.mk_var(1), k.mk_var(0)))))


def find_step_meta(ps):
    return next(ps.goal_meta_id(i) for i in range(ps.num_goals()) if 'Succ' in k.print_node(ps.goal_target(i)))


def idx_of(ps, meta):
    return next(i for i in range(ps.num_goals()) if ps.goal_meta_id(i) == meta)


# --- GOOD discipline: intro every quantified variable first, then induct on the one wanted. ---
ps_good = k.start_proof(build_goal())
ps_good.intro('n', target_meta_id=ps_good.goal_meta_id(0))
ps_good.intro('m', target_meta_id=ps_good.goal_meta_id(0))
ps_good.induct(0, target_meta_id=ps_good.goal_meta_id(0))
step_meta = find_step_meta(ps_good)
si = idx_of(ps_good, step_meta)

nh = ps_good.goal_num_hyps(si)
check(nh == 4, f"good discipline: step goal has 4 hyps as expected (got {nh})")

# For every hyp, the h-th LISTED hypothesis's own declared type must genuinely match what
# mk_var(h) resolves to -- verified directly (not assumed) by extracting m's own NodeId straight
# from the goal's own target and checking raw NodeId equality against mk_var(1).
sp = k.decompose_app(ps_good.goal_target(si))
lhs_parts = k.decompose_app(sp[2])  # mul, (Succ n'), m
m_extracted = lhs_parts[2]
check(m_extracted == k.mk_var(1), "good discipline: m (listed at hyp index 1) genuinely IS mk_var(1) -- listing index == level")

succ_n_prime = lhs_parts[1]
n_prime_extracted = k.decompose_app(succ_n_prime)[1]
check(n_prime_extracted == k.mk_var(2), "good discipline: n' (listed at hyp index 2) genuinely IS mk_var(2) -- listing index == level")

ih_ty = ps_good.goal_hyp_type(si, 3)
check(not k.is_pi(ih_ty), "good discipline: the IH is a bare Eq, not Pi-quantified over m (m was already in scope when induct() ran)")


# --- GUARD: induct() while a Pi still remains (m not yet introduced) must now be rejected
# immediately, cleanly, at the induct() call itself -- not several steps downstream. ---
ps_bad = k.start_proof(build_goal())
ps_bad.intro('n', target_meta_id=ps_bad.goal_meta_id(0))
try:
    ps_bad.induct(0, target_meta_id=ps_bad.goal_meta_id(0))
    check(False, "GUARD: induct() with a remaining Pi incorrectly succeeded")
except itp_env.ITPError as e:
    check("still a Pi" in str(e), f"GUARD: induct() with a remaining Pi is rejected immediately, with a clear message ({e})")

# cases() shares the identical generalize()-based construction and so carries the identical risk
# -- confirm it gets the identical guard, not just induct().
ps_bad2 = k.start_proof(build_goal())
ps_bad2.intro('n', target_meta_id=ps_bad2.goal_meta_id(0))
try:
    ps_bad2.cases(0, target_meta_id=ps_bad2.goal_meta_id(0))
    check(False, "GUARD: cases() with a remaining Pi incorrectly succeeded")
except itp_env.ITPError as e:
    check("still a Pi" in str(e), f"GUARD: cases() with a remaining Pi is rejected immediately, with a clear message ({e})")

# Control: the SAME induct() call, on the SAME goal, succeeds once "m" is intro()'d first --
# confirms the guard rejects specifically the missing-intro condition, not something else about
# this particular goal.
ps_bad.intro('m', target_meta_id=ps_bad.goal_meta_id(0))
try:
    ps_bad.induct(0, target_meta_id=ps_bad.goal_meta_id(0))
    check(True, "GUARD control: the identical induct() call succeeds once m is intro()'d first")
except itp_env.ITPError as e:
    check(False, f"GUARD control: induct() unexpectedly still rejected after intro('m'): {e}")

if failures == 0:
    print("\nALL INDUCT/PI-ORDERING CHECKS PASSED")
else:
    print(f"\n{failures} CHECK(S) FAILED")
