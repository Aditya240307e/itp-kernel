"""S8 gate (F8): the independently written checker against the kernels."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, random, time, functools
sys.setrecursionlimit(30000)
import harness as hn
from harness import *
import s3_tests as s3
from s3_tests import V, app, Lockstep
import s4_tests as t4
import s6_logic_tests as t6
import indep_checker as ic

def build_env(k):
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid) if hasattr(k, 'quot_role') else 0, k.is_inductive(gid) if hasattr(k, 'is_inductive') else None,
                recursion=k.recursive_info(gid) if hasattr(k, 'recursive_info') else None, reserved=k.reserved_kind(gid) if hasattr(k, 'reserved_kind') else 0)
    return env

def kernel_redeclare_verdicts(k):
    """does the KERNEL re-accept every global that has a value, through its own checked declaration path (in its current mode)?"""
    out = {}
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        if val is None: continue
        try: k.declare_checked(ty, val, 'rechecked_' + name); out[name] = True
        except Exception: out[name] = False
    return out

def kernel_verdict(k, tree_h, stmt_h):
    try: return k.check_open(build_new(k, tree_h), build_new(k, stmt_h), [])[0]
    except Exception: return False

def run_definitions(k, params=False):
    chk = ic.Checker(build_env(k), params=params, fuel=2_000_000)
    return chk, chk.check_definitions()

# ---------------------------------------------------------------- scope-aware mutation of NAMED terms
def count_nodes(t):
    k = t[0]
    if k in ('v', 'c', 's'): return 1
    if k == 'app': return 1 + count_nodes(t[1]) + count_nodes(t[2])
    if k in ('lam', 'pi'): return 1 + count_nodes(t[2]) + count_nodes(t[3])
    if k == 'match': return 1 + count_nodes(t[1]) + count_nodes(t[2]) + sum(count_nodes(b) for b in t[3])
    if k == 'fix': return 1 + count_nodes(t[2]) + count_nodes(t[4])

def rebuild(t, scope, ctr, target, fn):
    """rebuild t, applying fn(node, scope) to the target-th node in a pre-order walk; ctr is a one-element list"""
    idx = ctr[0]; ctr[0] += 1
    if idx == target:
        r = fn(t, scope)
        return t if r is None else r
    k = t[0]
    if k in ('v', 'c', 's'): return t
    if k == 'app': return ('app', rebuild(t[1], scope, ctr, target, fn), rebuild(t[2], scope, ctr, target, fn))
    if k in ('lam', 'pi'): return (k, t[1], rebuild(t[2], scope, ctr, target, fn), rebuild(t[3], scope + (t[1],), ctr, target, fn))
    if k == 'match': return ('match', rebuild(t[1], scope, ctr, target, fn), rebuild(t[2], scope, ctr, target, fn), tuple(rebuild(b, scope, ctr, target, fn) for b in t[3]))
    if k == 'fix': return ('fix', t[1], rebuild(t[2], scope, ctr, target, fn), t[3], rebuild(t[4], scope + (t[3],), ctr, target, fn))

def mutate_named(t, rng, gids, fams):
    def fn(node, scope):
        k = node[0]
        if k == 'v':
            others = [y for y in scope if y != node[1]]
            return ('v', rng.choice(others)) if others and rng.random() < 0.8 else ('c', rng.choice(gids))
        if k == 'c': return ('c', rng.choice(gids))
        if k == 's': return ('s', max(0, node[1] + rng.choice([-1, 1])))
        if k == 'app':
            if rng.random() < 0.5: return ('app', node[2], node[1])
            return ('app', node[1], ('v', rng.choice(scope)) if scope else ('c', rng.choice(gids)))
        if k in ('lam', 'pi'): return (k, node[1], rng.choice([('s', 0), ('c', rng.choice(fams)), ('v', rng.choice(scope)) if scope else ('s', 0)]), node[3])
        if k == 'match':
            if len(node[3]) >= 2 and rng.random() < 0.6:
                b = list(node[3]); i, j = rng.sample(range(len(b)), 2); b[i], b[j] = b[j], b[i]; return ('match', node[1], node[2], tuple(b))
            return ('match', ('v', rng.choice(scope)) if scope else node[1], node[2], node[3])
        if k == 'fix': return ('fix', max(0, node[1] + rng.choice([-1, 1])), node[2], node[3], node[4])
        return None
    n = count_nodes(t); target = rng.randrange(n)
    return rebuild(t, (), [0], target, fn)

def collect_proofs(kref, knew, g, params):
    """tactic proofs that BOTH the kernel and the independent checker will be asked about. Environments are built AFTER the scripts run (derived globals exist by then)."""
    out = []
    stmts4 = t4.statements4(g); kit_s, kit_scripts, _ = t6.kit(g, params)
    items = [(n, stmts4[n], t4.SCRIPTS4[n](g)) for n in ('add_zero_r', 'zero_le', 'symm_cases', 'le_succ_ind', 'exploit_le', 'add_succ_r', 'le_trans_i', 'trans_cases') if n in t4.SCRIPTS4]
    items += [(n, kit_s[n], kit_scripts[n]) for n in kit_scripts]
    if params:
        import s7_param_tests as t7
        for n, (stmt, ops) in t7.kit_on(kref, knew, g).items(): items.append((n, stmt, ops))
    for name, stmt, ops in items:
        try:
            ps = knew.start_proof(build_new(knew, l2h(stmt, 0)))
            for op in ops:
                nh = ps.goal_num_hyps(0) if ps.num_goals() else 0
                if op[0] == 'intro': ps.intro('x')
                elif op[0] == 'cases': ps.cases(op[1])
                elif op[0] == 'induct': ps.induct(op[1])
                else: ps.apply(build_new(knew, l2h(op[1], nh, lenient=True)))
            if ps.num_goals() == 0 and ps.is_complete(): out.append((name, parse_raw(knew.dump_raw(ps.extract()), True), l2h(stmt, 0)))
        except Exception: pass
    return out

def differential(label, knew, proofs, params, per_proof, seed):
    chk = ic.Checker(build_env(knew), params=params, fuel=3_000_000)
    rng = random.Random(seed); gids = list(chk.g.keys()); fams = [i for i in gids if chk.is_family(i)]
    stats = dict(valid=0, valid_rejected=[], mutants=0, agree=0, both_accept=0, both_reject=0, kernel_accepts_indep_rejects=[], indep_accepts_kernel_rejects=[])
    for name, proof, stmt in proofs:
        okk = kernel_verdict(knew, proof, stmt); oki, msg = chk.check_closed(proof, stmt); stats['valid'] += 1
        if not (okk and oki): stats['valid_rejected'].append((name, okk, oki, msg))
        base = ic.from_hybrid(proof)
        for _ in range(per_proof):
            m = mutate_named(base, rng, gids, fams)
            try: mh = ic.to_hybrid(m)
            except ic.CheckError: continue                                  # the mutation produced an unbound variable
            if canon_eq(m, base): continue
            stats['mutants'] += 1; vk = kernel_verdict(knew, mh, stmt); vi, msg = chk.check_closed(mh, stmt)
            if vk == vi: stats['agree'] += 1; stats['both_accept'] += vk; stats['both_reject'] += (not vk)
            elif vk: stats['kernel_accepts_indep_rejects'].append((name, msg))
            else: stats['indep_accepts_kernel_rejects'].append((name, ic.canon(m)))
    return chk, stats

def canon_eq(a, b): return ic.canon(a) == ic.canon(b)

def guard_corpus(g):
    """Fix terms valid or invalid ONLY because of the structural-recursion guard (each is well typed otherwise). (label, expected, type, term), level trees.
    Levels: the Fix's self is 0 (its body sits one deeper), then each binder takes the next level."""
    NAT, Z, S = ('const', g['Nat']), ('const', g['Zero']), ('const', g['Succ'])
    N1, N2 = ('pi', NAT, NAT), ('pi', NAT, ('pi', NAT, NAT)); mot = ('lam', NAT, NAT)
    lam = lambda body: ('lam', NAT, body)
    fix = lambda ty, body: ('fix', 0, ty, body)
    f, n, k, j = V(0), V(1), V(2), V(3)                      # self, the decreasing parameter, a predecessor, a predecessor's predecessor
    structural  = fix(N1, lam(('match', n, mot, (Z, lam(app(S, app(f, k)))))))                                       # f (S k) = S (f k)
    same_arg    = fix(N1, lam(app(f, n)))                                                                           # f n = f n
    on_succ     = fix(N1, lam(('match', n, mot, (Z, lam(app(f, app(S, k)))))))                                      # f (S k) = f (S k)
    nested      = fix(N1, lam(('match', n, mot, (Z, lam(('match', k, mot, (Z, lam(app(f, j)))))))))                # f (S (S j)) = f j
    wrong_arg   = fix(N2, lam(lam(('match', n, mot, (k, lam(app(f, k, k)))))))                                     # recursion on m, not n  (k is m here)
    add         = fix(N2, lam(lam(('match', n, mot, (k, lam(app(S, app(f, j, k))))))))                              # add (S k') m = S (add k' m)
    not_a_field = fix(N1, lam(('match', Z, mot, (Z, lam(app(f, n))))))                                              # the match is not on the parameter
    return [("structural: f (S k) = S (f k)", True, N1, structural), ("self call on the SAME argument: f n", False, N1, same_arg),
            ("self call on S k (not smaller): f (S k)", False, N1, on_succ), ("nested match on a smaller variable: f j", True, N1, nested),
            ("recursion on the WRONG argument", False, N2, wrong_arg), ("add: f k' m", True, N2, add), ("call under a match that is not on the parameter", False, N1, not_a_field)]

if __name__ == '__main__':
    t0 = time.time(); FAIL = []
    kref, knew, g = t6.build_logic()
    chk, bad = run_definitions(knew)
    print("independent re-check of the %d globals of the kernel (value : type, every type is a type): %d rejected" % (len(chk.g), len(bad)))
    for b in bad[:8]: print("    REJECTED", b)
    if bad: FAIL.append('definitions')
    stmts, scripts, wrong = t6.kit(g)
    rows = t6.run_scripts(kref, knew, g)
    for (name, done_n, ver_n, done_r, ver_r, fd, L) in rows:
        if not (done_n and ver_n): continue
        proof = parse_raw(knew.dump_raw(L.psn.extract()), True); stmt_h = l2h(stmts[name], 0)
        ok, msg = chk.check_closed(proof, stmt_h)
        print("   proof %-14s kernel accepts True | independent checker accepts %s %s" % (name, ok, '' if ok else '(' + msg + ')'))
        if not ok: FAIL.append('independent checker rejects the valid proof ' + name)
    for name, (ty, term) in wrong.items():
        ok, msg = chk.check_closed(l2h(term, 0), l2h(ty, 0))
        print("   wrong %-44s independent checker accepts %s %s" % (name, ok, '(' + msg + ')' if not ok else ''))
        if ok: FAIL.append('independent checker ACCEPTS the wrong proof ' + name)
    kg, ng, gg = t6.build_logic(); chk_g = ic.Checker(build_env(ng))
    for label, expected, ty, term in guard_corpus(gg):
        vk = kernel_verdict(ng, l2h(term, 0), l2h(ty, 0)); vi, msg = chk_g.check_closed(l2h(term, 0), l2h(ty, 0))
        print("   guard corpus: %-62s expected %-5s kernel %-5s independent %-5s" % (label, expected, vk, vi))
        if not (vk == vi == expected): FAIL.append('guard corpus: ' + label)
    for label, params in (("legacy mode", False), ("parameter mode", True)):
        kref2, knew2, g2 = t6.build_logic()
        if params: knew2.set_option('parameters', True)
        proofs = collect_proofs(kref2, knew2, g2, params)
        if not params: pass
        chk2, st = differential(label, knew2, proofs, params, int(sys.argv[1]) if len(sys.argv) > 1 else 150, 7 if not params else 8)
        _, bad2 = run_definitions(knew2, params)
        kv = kernel_redeclare_verdicts(knew2); k_rejected = sorted(n for n, ok in kv.items() if not ok); i_rejected = sorted(b[1] for b in bad2)
        print("   definitions the KERNEL rejects when re-declared (%s): %s | the independent checker rejects: %s" % (label, k_rejected, i_rejected))
        if k_rejected != i_rejected: FAIL.append(label + ': kernel and independent checker disagree about which definitions are valid')
        print("%s: %d tactic proofs (derived globals included: %d globals); valid proofs rejected by either side: %d; definitions rejected: %d" % (label, st['valid'], len(chk2.g), len(st['valid_rejected']), len(bad2)))
        print("   mutants: %d, verdicts agree %d (both accept %d, both reject %d); kernel ACCEPTS / independent REJECTS: %d; independent ACCEPTS / kernel REJECTS: %d" %
              (st['mutants'], st['agree'], st['both_accept'], st['both_reject'], len(st['kernel_accepts_indep_rejects']), len(st['indep_accepts_kernel_rejects'])))
        for v in st['valid_rejected'][:4]: print("      VALID PROOF REJECTED:", v)
        for v in st['kernel_accepts_indep_rejects'][:3]: print("      kernel accepts, independent rejects:", v)
        for v in st['indep_accepts_kernel_rejects'][:2]: print("      independent accepts, kernel rejects:", str(v)[:200])
        if st['valid_rejected']: FAIL.append(label + ': valid proof rejected')
        if st['kernel_accepts_indep_rejects'] or st['indep_accepts_kernel_rejects']: FAIL.append(label + ': verdict disagreement on mutants')
    print("elapsed %.1fs" % (time.time() - t0)); print("S8 INDEP GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
