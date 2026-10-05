"""F3: nested induction must produce extracted proofs that whole_term_check accepts.
Regression for (1) Fix-blind traversals (shift_levels_from etc.) and (2) reverted hypotheses that mention the
induction variable itself (branch lambda domain must be specialised to the constructor)."""
import curriculum as cur

def make(n_vars):
    k, refs = cur.build_base_kernel(); nat = refs['nat_ty']; V = k.mk_var
    body = cur.eq_app(k, refs, nat, V(0), V(0))
    ty = body
    for _ in range(n_vars): ty = k.mk_pi(nat, ty)
    ps = k.start_proof(ty)
    for i in range(n_vars): ps.intro('v%d' % i, target_meta_id=ps.goal_meta_id(0))
    return k, refs, ps, ty

def run(spec, n_vars=2):
    k, refs, ps, ty = make(n_vars)
    cur_meta = ps.goal_meta_id(0)
    gi = lambda m: next(i for i in range(ps.num_goals()) if ps.goal_meta_id(i) == m)
    def last_nat(g): return max(j for j in range(ps.goal_num_hyps(g)) if k.print_node(ps.goal_hyp_type(g, j)) == 'Nat')
    for kind, which in (s.split(':') for s in spec):
        g = gi(cur_meta); h = last_nat(g) if which == 'last' else int(which)
        before = {ps.goal_meta_id(i) for i in range(ps.num_goals())}
        (ps.induct if kind == 'I' else ps.cases)(h, target_meta_id=cur_meta)
        new = [ps.goal_meta_id(i) for i in range(ps.num_goals()) if ps.goal_meta_id(i) not in before]
        for m in new[:-1]: ps.apply(refs['refl_c'], target_meta_id=m)
        cur_meta = new[-1]
    ps.apply(refs['refl_c'], target_meta_id=cur_meta)
    assert ps.is_complete()
    return k.whole_term_check_detailed(ps.extract(), ty)

CASES = [(['I:0'], 1), (['I:0', 'I:last'], 2), (['I:0', 'I:1'], 2), (['I:1', 'I:0'], 2), (['I:0', 'C:last'], 2),
         (['C:0', 'C:last'], 2), (['C:0', 'C:1'], 2),
         (['I:0', 'I:1', 'I:2'], 3), (['I:2', 'I:1', 'I:0'], 3), (['I:0', 'I:last', 'I:last'], 3), (['I:1', 'I:0', 'I:last'], 3)]
for spec, nv in CASES:
    r = run(spec, nv); assert r == 'OK', (spec, r); print('PASS  nested %-26s whole_term_check OK' % ','.join(spec))

import itertools
for perm in itertools.permutations([0, 1, 2]):
    spec = ['I:%d' % i for i in perm]; r = run(spec, 3); assert r == 'OK', (spec, r)
print('PASS  all 6 induction orders of 3 variables')
for perm in [(0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1), (1, 3, 0, 2), (3, 0, 2, 1)]:
    spec = ['I:%d' % i for i in perm]; r = run(spec, 4); assert r == 'OK', (spec, r)
print('PASS  5 induction orders of 4 variables (including fully reversed)')

# isolated: induction on a variable that a hypothesis depends on (the reverted-domain specialisation bug)
k, refs = cur.build_base_kernel(); nat = refs['nat_ty']; V = k.mk_var
E = lambda a, b: cur.eq_app(k, refs, nat, a, b)
ty = k.mk_pi(nat, k.mk_pi(E(V(0), V(0)), E(V(0), V(0))))
ps = k.start_proof(ty)
ps.intro('n', target_meta_id=ps.goal_meta_id(0)); ps.intro('h', target_meta_id=ps.goal_meta_id(0))
cm = ps.goal_meta_id(0); before = {ps.goal_meta_id(i) for i in range(ps.num_goals())}
ps.induct(0, target_meta_id=cm)
for m in [ps.goal_meta_id(i) for i in range(ps.num_goals()) if ps.goal_meta_id(i) not in before]: ps.apply(refs['refl_c'], target_meta_id=m)
assert k.whole_term_check_detailed(ps.extract(), ty) == 'OK'
print('PASS  induction on n with a hypothesis h : n = n that depends on n')
print('ALL NESTED-INDUCTION TESTS PASSED')
