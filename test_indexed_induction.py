"""Regression test: induct() on an INDEXED family (Le) with hypotheses that must be generalized.

Covers, each verified end-to-end through the independent whole_term_check():

  1. le_trans      (a b c)(h1:Le a b)(h2:Le b c) -> Le a c   -- induct on h2; h1 is reverted and
                   sits BEFORE the scrutinee.
  2. le_trans_rev  (a b c)(h2:Le b c)(h1:Le a b) -> Le a c   -- same lemma, hypotheses reordered:
                   h1 sits AFTER the scrutinee (the other direction of the dependency fixpoint).
  3. succ_le_succ  (n m)(h:Le n m) -> Le (S n) (S m)         -- goal mentions BOTH indices of Le;
                   only provable at all when the agent-facing branch substitution uses FORCED
                   result indices (le_S's first result index is a bare field variable).
  4. bare induct   (n m)(h:Le n m) -> Le Zero m              -- no reverted hyp at all.

Why each piece of the fix is pinned down here (all found by driving le_trans through
whole_term_check; each failure mode was confirmed with raw term dumps, not guessed):

  * selective (fixpoint) revert sweep, not cases()'s contiguous sweep -- guarded structurally
    (test_no_oversweep) and by a search-budget check, since the contiguous sweep exploded
    add_succ_right's search tree ~6x and, at larger budgets, aborted the process;
  * true-runtime branches must be wrapped with a Lambda per reverted hyp (le_n has zero
    constructor fields, so its branch was a bare value where the motive demands a function);
  * forced result indices on BOTH the agent-facing and true-runtime sides (Le a n' vs Le a b);
  * the recursive call must NOT be pre-applied to the reverted hyps (the IH is a function of them);
  * positions of reverted hyps in subgoal_applied must receive the branch-local specialization,
    since the Fix body is generic in the index;
  * solved-meta beta-reduction must substitute SIMULTANEOUSLY (sequential substitution clobbered
    `self` when an IH argument's re-embedded level equalled a later pending position).
"""
import itp_env, curriculum as cur, hypergraph as hg


def fresh_kernel():
    k, refs = cur.build_base_kernel()
    refs.update(cur.setup_nat_order(k, refs))
    for goal in cur.ORDER_CURRICULUM:
        lib = k.index_library()
        gty = goal.build_type(k, refs)
        ps = k.start_proof(gty); rm = ps.goal_meta_id(0); dm = {}
        root = hg.get_or_create_node(k, ps, 0, dm)
        st = hg.search(k, root, lib, hg.UniformPolicy(), dm, max_simulations=500, top_k=16)
        assert st == hg.Status.SOLVED, f"{goal.name} not solved"
        term = hg.commit_and_verify(k, ps, root, rm, gty)
        if goal.lemma_name:
            gid = k.declare_checked(gty, term, goal.lemma_name)
            refs[goal.lemma_name + '_c'] = k.mk_const(gid)
    return k, refs


k, refs = fresh_kernel()
nat = refs['nat_ty']
V = k.mk_var


def le(a, b):
    return cur.le_app(k, refs, a, b)


def S(x):
    return k.mk_app(refs['succ_c'], x)


def app(f, *args):
    for a in args:
        f = k.mk_app(f, a)
    return f


def hyps(ps, meta):
    gi = next(i for i in range(ps.num_goals()) if ps.goal_meta_id(i) == meta)
    return gi, [(j, ps.goal_hyp_name(gi, j), k.print_node(ps.goal_hyp_type(gi, j)))
                for j in range(ps.goal_num_hyps(gi))]


def goal_by(ps, pred):
    for i in range(ps.num_goals()):
        if pred(k.print_node(ps.goal_target(i))):
            return ps.goal_meta_id(i)
    raise AssertionError("no goal matches")


def finish(ps, goal_ty, name):
    assert ps.is_complete(), "proof not complete"
    term = ps.extract()
    k.whole_term_check(term, goal_ty)       # the independent, from-scratch re-verification
    gid = k.declare_checked(goal_ty, term, name)
    return term, gid


def intros(ps, *names):
    for n in names:
        ps.intro(n, target_meta_id=ps.goal_meta_id(0))


# ---------------------------------------------------------------- 1. le_trans
le_trans_ty = k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(le(V(0), V(1)),
              k.mk_pi(le(V(1), V(2)), le(V(0), V(2)))))))


def prove_le_trans():
    ps = k.start_proof(le_trans_ty)
    intros(ps, 'a', 'b', 'c', 'h1', 'h2')
    ps.induct(4, target_meta_id=ps.goal_meta_id(0))
    assert ps.num_goals() == 2
    base = goal_by(ps, lambda t: 'Succ' not in t)
    step = goal_by(ps, lambda t: 'Succ' in t)
    _, hb = hyps(ps, base)
    h1_resp = max(j for j, n, _ in hb if n == 'h1')           # re-specialized copy, last
    ps.apply(V(h1_resp), target_meta_id=base)
    _, hs = hyps(ps, step)
    # step goal: stale h1 : Le a b ; ..., c' (Nat), p, IH : Pi (h1 : Le a b). Le a c', h1' : Le a b
    ih = next(j for j, _, t in hs if t.startswith('(Pi') and 'Le' in t)
    cprime = ih - 2
    assert hs[cprime][2] == 'Nat'
    h1_orig = min(j for j, n, _ in hs if n == 'h1')
    # forced-index consistency: the re-specialized h1 in the step goal must read Le a b (not Le a n')
    h1_resp_s = max(j for j, n, _ in hs if n == 'h1')
    assert hs[h1_resp_s][2] == hs[h1_orig][2], (hs[h1_resp_s], hs[h1_orig])
    ps.apply(app(refs['le_s_c'], V(0), V(cprime), k.mk_app(V(ih), V(h1_orig))), target_meta_id=step)
    return finish(ps, le_trans_ty, 'le_trans')


term_le_trans, le_trans_gid = prove_le_trans()
print("PASS 1: le_trans (reverted hyp before scrutinee) -- whole_term_check verified")

# ------------------------------------------------- 2. le_trans, hypotheses reordered
le_trans_rev_ty = k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(le(V(1), V(2)),
                  k.mk_pi(le(V(0), V(1)), le(V(0), V(2)))))))


def prove_le_trans_rev():
    ps = k.start_proof(le_trans_rev_ty)
    intros(ps, 'a', 'b', 'c', 'h2', 'h1')
    ps.induct(3, target_meta_id=ps.goal_meta_id(0))               # h2 is at index 3 now
    assert ps.num_goals() == 2
    base = goal_by(ps, lambda t: 'Succ' not in t)
    step = goal_by(ps, lambda t: 'Succ' in t)
    _, hb = hyps(ps, base)
    h1_resp = max(j for j, n, _ in hb if n == 'h1')
    ps.apply(V(h1_resp), target_meta_id=base)
    _, hs = hyps(ps, step)
    ih = next(j for j, _, t in hs if t.startswith('(Pi') and 'Le' in t)
    cprime = ih - 2
    assert hs[cprime][2] == 'Nat'
    h1_orig = min(j for j, n, _ in hs if n == 'h1')
    ps.apply(app(refs['le_s_c'], V(0), V(cprime), k.mk_app(V(ih), V(h1_orig))), target_meta_id=step)
    return finish(ps, le_trans_rev_ty, 'le_trans_rev')


prove_le_trans_rev()
print("PASS 2: le_trans_rev (reverted hyp AFTER scrutinee) -- whole_term_check verified")

# ------------------------------------------------------------ 3. succ_le_succ
sls_ty = k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(le(V(0), V(1)), le(S(V(0)), S(V(1))))))


def prove_succ_le_succ():
    ps = k.start_proof(sls_ty)
    intros(ps, 'n', 'm', 'h')
    ps.induct(2, target_meta_id=ps.goal_meta_id(0))
    base = goal_by(ps, lambda t: t.count('Succ') == 2)            # Le (S n') (S n')
    step = goal_by(ps, lambda t: t.count('Succ') == 3)            # Le (S n) (S (S m'))
    ps.apply(refs['le_refl_c'], target_meta_id=base)
    _, hs = hyps(ps, step)
    ih = len(hs) - 1
    mf = ih - 2
    assert hs[mf][2] == 'Nat'
    # With FORCED result indices the step target is  Le (S n) (S (S m'))  -- mentioning the OUTER n,
    # consistent with the IH (Le (S n) (S m')) -- so this closes. With the field's unforced identity
    # (n') the target would mention n' while the IH mentions n, and no tactic could connect them.
    ps.apply(app(refs['le_s_c'], S(V(0)), S(V(mf)), V(ih)), target_meta_id=step)
    return finish(ps, sls_ty, 'succ_le_succ')


prove_succ_le_succ()
print("PASS 3: succ_le_succ (goal mentions both indices; forced-index substitution) -- verified")

# --------------------------------------------------------------- 4. bare induct
bare_ty = k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(le(V(0), V(1)), le(refs['zero'], V(1)))))


def prove_bare():
    ps = k.start_proof(bare_ty)
    intros(ps, 'n', 'm', 'h')
    ps.induct(2, target_meta_id=ps.goal_meta_id(0))
    base = goal_by(ps, lambda t: 'Succ' not in t)
    step = goal_by(ps, lambda t: 'Succ' in t)
    ps.apply(refs['zero_le_c'], target_meta_id=base)
    _, hs = hyps(ps, step)
    ih = len(hs) - 1
    ps.apply(app(refs['le_s_c'], refs['zero'], V(ih - 2), V(ih)), target_meta_id=step)
    return finish(ps, bare_ty, 'bare_le')


prove_bare()
print("PASS 4: bare induct on Le (no reverted hyp) -- verified")

# --------------------------- 5. targets with embedded bound structure: clean rejection
# PRE-EXISTING LIMITATION, reproduced on the UNPATCHED baseline with a plain non-indexed goal:
# induct() builds child goals by syntactic substitution, which does not re-level bound structure
# embedded in the target (a lambda body that refers to its own binder). The mis-leveled goal used to
# abort the whole process (Env::operator[] out of bounds) when a tactic later evaluated it. induct()
# now rejects such inputs cleanly (a dead end for search, not a killed worker). Soundness was never at
# stake -- whole_term_check re-verifies everything.  Found by mutation testing: reverting the motive
# scheme's shift amount (2*arity+3 -> arity+3) was undetectable, because the shift only ever moves
# exactly this embedded structure.
eq = lambda a, b: cur.eq_app(k, refs, nat, a, b)
lam_id_le = k.mk_lambda(nat, V(3))               # \x:Nat. x   -- binder at level 3 (intro'd ctx size)
idarg_ty = k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(le(V(0), V(1)), le(k.mk_app(lam_id_le, V(0)), V(1)))))
nat_idarg_ty = k.mk_pi(nat, eq(k.mk_app(k.mk_lambda(nat, V(1)), V(0)), V(0)))   # (n) -> Eq ((\x.x) n) n

# Positive control: an embedded binder that is NEVER referenced has nothing to mis-level; the guard
# must not reject it.   (n m)(h : Le n m) -> Le n ((\x:Nat. m) n)
const_lam_ty = k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(le(V(0), V(1)),
               le(V(0), k.mk_app(k.mk_lambda(nat, V(1)), V(0))))))


def prove_const_lambda():
    ps = k.start_proof(const_lam_ty)
    intros(ps, 'n', 'm', 'h')
    ps.induct(2, target_meta_id=ps.goal_meta_id(0))
    base = goal_by(ps, lambda t: 'Succ' not in t)
    step = goal_by(ps, lambda t: 'Succ' in t)
    ps.apply(refs['le_refl_c'], target_meta_id=base)
    _, hs = hyps(ps, step)
    ih = len(hs) - 1
    mf = ih - 2
    assert hs[mf][2] == 'Nat'
    ps.apply(app(refs['le_s_c'], V(0), V(mf), V(ih)), target_meta_id=step)
    return finish(ps, const_lam_ty, 'le_const_lambda')


prove_const_lambda()
print("PASS 5: unreferenced embedded binder is NOT rejected (guard precision) -- verified")

# ------------------------------------------------------------------ soundness
def must_raise(fn, what):
    try:
        fn()
    except itp_env.ITPError:
        print(f"PASS (rejected): {what}")
        return
    raise AssertionError(f"UNSOUND: {what} was accepted")


def neg_stale_h1_in_base():
    ps = k.start_proof(le_trans_ty)
    intros(ps, 'a', 'b', 'c', 'h1', 'h2')
    ps.induct(4, target_meta_id=ps.goal_meta_id(0))
    base = goal_by(ps, lambda t: 'Succ' not in t)
    # target is Le a n' ; the stale h1 is Le a b -- b and n' are unrelated variables here
    ps.apply(V(3), target_meta_id=base)


def neg_wrong_h1_for_ih_result():
    ps = k.start_proof(le_trans_ty)
    intros(ps, 'a', 'b', 'c', 'h1', 'h2')
    ps.induct(4, target_meta_id=ps.goal_meta_id(0))
    step = goal_by(ps, lambda t: 'Succ' in t)
    _, hs = hyps(ps, step)
    ih = next(j for j, _, t in hs if t.startswith('(Pi') and 'Le' in t)
    cprime = ih - 2
    # le_S a c' h1 : needs a proof of Le a c', but h1 proves Le a b
    ps.apply(app(refs['le_s_c'], V(0), V(cprime), V(3)), target_meta_id=step)


def neg_wrong_goal_for_extracted_term():
    # the (verified) le_trans proof must not check against a different statement
    wrong = k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(nat, k.mk_pi(le(V(0), V(1)),
            k.mk_pi(le(V(1), V(2)), le(V(2), V(0)))))))
    k.whole_term_check(term_le_trans, wrong)


must_raise(neg_stale_h1_in_base, "stale h1 (Le a b) used where the base goal needs Le a n'")
must_raise(neg_wrong_h1_for_ih_result, "le_S a c' h1 with h1 : Le a b (needs Le a c')")
must_raise(neg_wrong_goal_for_extracted_term, "extracted le_trans proof checked against Le c a")


def neg_embedded_binder_indexed():
    ps = k.start_proof(idarg_ty)
    intros(ps, 'n', 'm', 'h')
    ps.induct(2, target_meta_id=ps.goal_meta_id(0))


def neg_embedded_binder_nat():
    ps = k.start_proof(nat_idarg_ty)
    intros(ps, 'n')
    ps.induct(0, target_meta_id=ps.goal_meta_id(0))


must_raise(neg_embedded_binder_indexed, "induct on Le with a lambda-redex target (cleanly, no abort)")
must_raise(neg_embedded_binder_nat, "induct on Nat with a lambda-redex target (cleanly, no abort)")


# ------------------------------------------------- structural over-sweep guard
def test_no_oversweep():
    """Inducting on n with an UNRELATED later hypothesis m must NOT sweep m (cases()'s contiguous
    rule did, adding a spurious Pi to the IH and exploding double-induction search)."""
    eq = lambda a, b: cur.eq_app(k, refs, nat, a, b)
    ty = k.mk_pi(nat, k.mk_pi(nat, eq(V(0), V(0))))
    ps = k.start_proof(ty)
    intros(ps, 'n', 'm')
    ps.induct(0, target_meta_id=ps.goal_meta_id(0))
    step = goal_by(ps, lambda t: 'Succ' in t)
    _, hs = hyps(ps, step)
    assert len(hs) == 4, f"expected [n, m, n', IH], got {hs}"
    assert not hs[3][2].startswith('(Pi'), f"IH gained a spurious Pi layer: {hs[3]}"
    base = goal_by(ps, lambda t: 'Succ' not in t)
    _, hb = hyps(ps, base)
    assert len(hb) == 2, f"base goal gained spurious hyps: {hb}"


test_no_oversweep()
print("PASS: no over-sweep of unrelated hypotheses (structural guard)")


# ------------------------------------------------- search-budget regression guard
def test_search_budget_parity():
    k2, refs2 = cur.build_base_kernel()
    for goal in cur.CURRICULUM:
        lib = k2.index_library()
        gty = goal.build_type(k2, refs2)
        ps = k2.start_proof(gty); rm = ps.goal_meta_id(0); dm = {}
        root = hg.get_or_create_node(k2, ps, 0, dm)
        st = hg.search(k2, root, lib, hg.UniformPolicy(), dm, max_simulations=200, top_k=16)
        assert st == hg.Status.SOLVED, f"{goal.name} not solved at 200 sims"
        if goal.name == 'add_succ_right':
            # unpatched baseline: 749 nodes.  Over-sweeping build: 4744 and UNSOLVED.
            assert len(dm) < 1500, f"search tree exploded: {len(dm)} nodes"
        term = hg.commit_and_verify(k2, ps, root, rm, gty)
        if goal.lemma_name:
            gid = k2.declare_checked(gty, term, goal.lemma_name)
            refs2[goal.lemma_name + '_c'] = k2.mk_const(gid)
            refs2[goal.lemma_name + '_id'] = gid


test_search_budget_parity()
print("PASS: curriculum solved at 200 sims with baseline-comparable search trees")

# larger-budget exploration used to abort the process (Env::operator[] out of bounds) inside
# cases(); reaching the print below at all is the assertion.
k3, refs3 = cur.build_base_kernel()
for goal in cur.CURRICULUM:
    lib = k3.index_library(); gty = goal.build_type(k3, refs3)
    ps = k3.start_proof(gty); rm = ps.goal_meta_id(0); dm = {}
    root = hg.get_or_create_node(k3, ps, 0, dm)
    st = hg.search(k3, root, lib, hg.UniformPolicy(), dm, max_simulations=3000 if goal.name == 'add_succ_right' else 200, top_k=16)
    if st == hg.Status.SOLVED and goal.lemma_name:
        term = hg.commit_and_verify(k3, ps, root, rm, gty)
        gid = k3.declare_checked(gty, term, goal.lemma_name)
        refs3[goal.lemma_name + '_c'] = k3.mk_const(gid); refs3[goal.lemma_name + '_id'] = gid
print("PASS: 3000-simulation exploration of add_succ_right completes without aborting")

# ---------------------------------------------- library integration (curriculum.setup_le_trans)
# The scripted, whole_term_check-verified le_trans must be a USABLE library lemma, not just a proof:
# derive le_trans4 by applying it twice.
k5, refs5 = fresh_kernel()
refs5.update(cur.setup_le_trans(k5, refs5))
n5 = refs5['nat_ty']; V5 = k5.mk_var
le5 = lambda a, b: cur.le_app(k5, refs5, a, b)
ty4 = k5.mk_pi(n5, k5.mk_pi(n5, k5.mk_pi(n5, k5.mk_pi(n5, k5.mk_pi(le5(V5(0), V5(1)),
      k5.mk_pi(le5(V5(1), V5(2)), k5.mk_pi(le5(V5(2), V5(3)), le5(V5(0), V5(3)))))))))
ps5 = k5.start_proof(ty4)
for nm in ('a', 'b', 'c', 'd', 'h1', 'h2', 'h3'):
    ps5.intro(nm, target_meta_id=ps5.goal_meta_id(0))
lt = refs5['le_trans_c']
a5 = lambda f, *xs: __import__('functools').reduce(lambda acc, x: k5.mk_app(acc, x), xs, f)
inner = a5(lt, V5(0), V5(1), V5(2), V5(4), V5(5))             # Le a c
outer = a5(lt, V5(0), V5(2), V5(3), inner, V5(6))             # Le a d
ps5.apply(outer, target_meta_id=ps5.goal_meta_id(0))
assert ps5.is_complete()
k5.whole_term_check(ps5.extract(), ty4)
print("PASS: curriculum.setup_le_trans is a usable, verified library lemma (le_trans4 from two uses)")

print("\nALL INDEXED-INDUCTION TESTS PASSED")
