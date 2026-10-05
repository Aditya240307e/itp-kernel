"""Core kernel micro-benchmarks (single thread). Used as the performance gate for the variable-representation
refactor: record at the starting commit, re-run after, compare ratios. Output is JSON on stdout."""
import json, time, statistics, curriculum as cur

def timeit(fn, reps, warm=3):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts)

def nested_proof():
    k, refs = cur.build_base_kernel(); nat = refs['nat_ty']; V = k.mk_var
    ty = k.mk_pi(nat, k.mk_pi(nat, cur.eq_app(k, refs, nat, V(0), V(0))))
    ps = k.start_proof(ty)
    for n in ('a', 'b'): ps.intro(n, target_meta_id=ps.goal_meta_id(0))
    cm = ps.goal_meta_id(0)
    for h in (0, 1):
        before = {ps.goal_meta_id(i) for i in range(ps.num_goals())}
        ps.induct(h, target_meta_id=cm)
        new = [ps.goal_meta_id(i) for i in range(ps.num_goals()) if ps.goal_meta_id(i) not in before]
        for m in new[:-1]: ps.apply(refs['refl_c'], target_meta_id=m)
        cm = new[-1]
    ps.apply(refs['refl_c'], target_meta_id=cm)
    return k, ty, ps.extract()

res = {}
k0, refs0 = cur.build_base_kernel()
res['B1_pipeline_2x_induct_extract_us'] = [x * 1e6 for x in timeit(lambda: nested_proof(), 60)]
k, ty, term = nested_proof()
res['B2_whole_term_check_us'] = [x * 1e6 for x in timeit(lambda: k.whole_term_check(term, ty), 1500)]
nat, zero, succ, add_c = refs0['nat_ty'], refs0['zero'], refs0['succ_c'], refs0['add_c']
def numeral(n):
    t = zero
    for _ in range(n): t = k0.mk_app(succ, t)
    return t
big = k0.mk_app(k0.mk_app(add_c, numeral(30)), numeral(30))
res['B3_normalize_add_30_30_us'] = [x * 1e6 for x in timeit(lambda: k0.normalize_node(big), 400)]
def intro_cases():
    ps = k0.start_proof(k0.mk_pi(nat, cur.eq_app(k0, refs0, nat, k0.mk_var(0), k0.mk_var(0))))
    ps.intro('n', target_meta_id=ps.goal_meta_id(0)); ps.cases(0, target_meta_id=ps.goal_meta_id(0))
res['B4_intro_cases_us'] = [x * 1e6 for x in timeit(intro_cases, 1500)]
print(json.dumps({k_: dict(median=round(v[0], 1), min=round(v[1], 1)) for k_, v in res.items()}, indent=1))
