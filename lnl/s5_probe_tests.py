"""S5 part 2a gate: the search's probing protocol -- checkpoint / fork_single_goal / apply / read off / rewind -- and materialize_from_node."""
import sys, random, time, functools
import harness as hn
from harness import *
import s3_tests as s3
from s3_tests import V, app, Lockstep, library, canon_metas
import s4_tests as t4
import s5_tests as t5

def canon_state(k, ps, new, only=None):
    st = []
    for i in (range(ps.num_goals()) if only is None else [only]):
        nh = ps.goal_num_hyps(i)
        tgt = parse_raw(k.dump_raw(ps.goal_target(i)), new); hy = [parse_raw(k.dump_raw(ps.goal_hyp_type(i, j)), new) for j in range(nh)]
        if not new: tgt = l2h(tgt, nh, lenient=True); hy = [l2h(h, j, lenient=True) for j, h in enumerate(hy)]
        st.append((('meta', ps.goal_meta_id(i)), nh, tgt, tuple(hy)))
    return canon_metas(st)

def dispatch(k, g, ps, op, nh, new):
    bld, conv = (build_new, lambda t, n: l2h(t, n, lenient=True)) if new else (build_ref, lambda t, n: t)
    if op[0] == 'intro': ps.intro('x')
    elif op[0] == 'apply': ps.apply(bld(k, conv(op[1], nh)))
    elif op[0] == 'cases': ps.cases(op[1])
    elif op[0] == 'induct': ps.induct(op[1])
    elif op[0] == 'rewrite': ps.rewrite(bld(k, conv(V(op[1]), nh)), bld(k, conv(op[2], nh)), bld(k, conv(op[3], nh)), g['transport'], g['symm'])

# ---------------------------------------------------------------- (A) rewind restores the kernel and replays deterministically
def test_rewind_determinism():
    out = {}
    for sname, ops in (('exploit_le', [('intro',), ('cases', 0), ('apply', ('const', None))]), ('le_succ_ind', [('intro',)] * 3 + [('induct', 2)]), ('add_zero_r', [('intro',), ('induct', 0)])):
        kref, knew, g, defs = library(); stmt = t5.statements5(g)[sname]
        ops = [(o[0], ('const', g['refl'])) if o[0] == 'apply' else o for o in ops]
        n0 = (len(kref.list_globals()), len(knew.list_globals())); cps = (kref.checkpoint(), knew.checkpoint())
        runs = []; grown = []
        for attempt in range(2):
            L = Lockstep(kref, knew, g, stmt)
            for op in ops: L.step(op)
            runs.append((L.state(), kref.dump_raw(L.psr.extract()) if L.psr.num_goals() == 0 else None, knew.dump_raw(L.psn.extract()) if L.psn.num_goals() == 0 else None))
            grown.append((len(kref.list_globals()), len(knew.list_globals())))
            kref.rewind(cps[0]); knew.rewind(cps[1])
            assert (len(kref.list_globals()), len(knew.list_globals())) == n0, "rewind did not restore the global count"
        out[sname] = dict(identical_after_rewind=(runs[0] == runs[1]), states_agree=all(runs[0][0][0][0] == runs[0][0][1][0] for _ in [0]), globals_created=grown[0][1] - n0[1], restored=True)
    return out

# ---------------------------------------------------------------- (B) a stale memo entry must not survive a rewind (node indices are reused)
def test_loose_cache_after_rewind():
    kref, knew, g, defs = library(); cp = knew.checkpoint()
    A = build_new(knew, ('bvar', 4242)); ra = knew.loose_range(A)    # a node nobody has built before: loose range 4243, memoised under its node id
    knew.rewind(cp)
    B = build_new(knew, ('fvar', 9191)); rb = knew.loose_range(B)    # also new: the first variable node after the rewind reuses A's index; its loose range is 0
    return ra, rb

# ---------------------------------------------------------------- (C) the search's protocol, in lockstep, inside random scripts; (D) materialize_from_node
def probe_all(L, g, ops_to_probe, stats):
    mism = []
    before = (L.state(), len(L.kref.list_globals()), len(L.knew.list_globals()), L.psr.goal_meta_id(0), L.psn.goal_meta_id(0))
    for op in ops_to_probe:
        outs = []
        for k, ps, new in ((L.kref, L.psr, False), (L.knew, L.psn, True)):
            cp = k.checkpoint(); n_glob = len(k.list_globals())
            fk = k.fork_single_goal(ps, ps.goal_meta_id(0)); nh = fk.goal_num_hyps(0); ok = True; msg = ''
            try: dispatch(k, g, fk, op, nh, new)
            except Exception as e: ok = False; msg = str(e)
            st = canon_state(k, fk, new) if ok else None
            created = len(k.list_globals()) - n_glob
            k.rewind(cp)
            if len(k.list_globals()) != n_glob: mism.append(('globals not restored by rewind', op))
            outs.append((ok, msg, st, created))
        (ok_r, msg_r, st_r, cr_r), (ok_n, msg_n, st_n, cr_n) = outs
        stats['probes'] += 1; stats['probes_ok'] += ok_r and ok_n; stats['derived_globals_created_and_dropped'] += (cr_n > 0)
        if ok_r != ok_n:
            if (not ok_r) and ok_n and 'embedded bound structure' in msg_r: stats['relaxed'] += 1
            elif op[0] == 'rewrite' and ok_r and not ok_n: stats['f13'] += 1
            elif op[0] == 'apply' and op[1][0] == 'var' and (not ok_r) and ok_n and t4.hyp_is_pi(L, op[1][1]): stats['f16'] = stats.get('f16', 0) + 1
            else: mism.append(('probe verdict differs', op, ok_r, ok_n, msg_r[:70], msg_n[:70]))
        elif ok_r:
            if op[0] in ('intro', 'apply'):
                if st_r != st_n: mism.append(('probe state differs', op))
            elif t4.erase_vars(st_r) != t4.erase_vars(st_n): mism.append(('probe shape differs', op))
            if cr_r != cr_n: mism.append(('derived globals created differ', op, cr_r, cr_n))
    after = (L.state(), len(L.kref.list_globals()), len(L.knew.list_globals()), L.psr.goal_meta_id(0), L.psn.goal_meta_id(0))
    if after != before: mism.append(('the main state was disturbed by probing',))
    if L.knew.meta_is_solved(L.psn.goal_meta_id(0)): mism.append(('a probe left the live goal meta solved (meta rollback failed)',))
    return mism

def check_materialize(L, g, stats):
    """a state rebuilt from a goal's captured (target, hypothesis types, names) must equal the live goal, in both kernels"""
    res = []
    for k, ps, new in ((L.kref, L.psr, False), (L.knew, L.psn, True)):
        nh = ps.goal_num_hyps(0)
        mat = k.materialize_from_node(ps.goal_target(0), [ps.goal_hyp_type(0, j) for j in range(nh)], [ps.goal_hyp_name(0, j) for j in range(nh)])
        res.append((canon_state(k, mat, new), canon_state(k, ps, new, only=0)))
    stats['materialized'] += 1
    return [] if (res[0][0] == res[0][1] and res[1][0] == res[1][1] and res[0][0] == res[1][0]) else [('materialize_from_node differs from the live goal', res[0][0] == res[0][1], res[1][0] == res[1][1])]

def run_random(N, seed, max_steps=6, probes_per_step=4):
    kref, knew, g, defs = library(); stmts = t5.statements5(g); names = list(stmts); rng = random.Random(seed)
    stats = dict(scripts=0, steps=0, probes=0, probes_ok=0, derived_globals_created_and_dropped=0, relaxed=0, f13=0, materialized=0); bad = []
    consts = [g[n] for n in ('refl', 'cong', 'symm', 'trans', 'transport', 'le_n', 'le_S', 'le_trans', 'Succ', 'add')]
    ind = {g['Nat'], g['Eq'], g['Le']}; head = lambda x: head(x[1]) if x[0] == 'app' else x
    def gen_op(L):
        nh = L.psr.goal_num_hyps(0); tgt = parse_raw(kref.dump_raw(L.psr.goal_target(0)), False)
        hyps = [parse_raw(kref.dump_raw(L.psr.goal_hyp_type(0, j)), False) for j in range(nh)]
        indh = [j for j, h in enumerate(hyps) if head(h)[0] == 'const' and head(h)[1] in ind]
        eqh = [j for j, h in enumerate(hyps) if h[0] == 'app' and h[1][0] == 'app' and h[1][1][0] == 'app' and h[1][1][1] == ('const', g['Eq']) and not any(w in str(h) for w in ("'lam'", "'pi'", "'match'", "'fix'"))]
        r = rng.random()
        if tgt[0] == 'pi' and r < 0.4: return ('intro',)
        if indh and tgt[0] != 'pi' and r < 0.65: return (rng.choice(['cases', 'induct']), rng.choice(indh))
        if eqh and tgt[0] != 'pi' and r < 0.72: j = rng.choice(eqh); return ('rewrite', j, hyps[j][1][2], hyps[j][2])
        if nh and r < 0.85: return ('apply', V(rng.randrange(nh)))
        return ('apply', ('const', rng.choice(consts)))
    for i in range(N):
        sname = rng.choice(names); L = Lockstep(kref, knew, g, stmts[sname]); stats['scripts'] += 1
        for _ in range(max_steps):
            if L.psr.num_goals() == 0: break
            m = check_materialize(L, g, stats) + probe_all(L, g, [gen_op(L) for _ in range(probes_per_step)], stats)
            if m: bad.append((sname, L.ops[-3:], m)); break
            op = gen_op(L); ok, same, st = L.step(op); stats['steps'] += 1
            if L.stopped or not same:
                cat = t4.explain_divergence(L, stmts[sname], op, st)
                if cat is None: bad.append((sname, L.ops[-3:], [('unexplained lockstep divergence', str(st)[:160])]))
                break
        if bad: break
    return stats, bad

if __name__ == '__main__':
    t0 = time.time(); FAIL = []
    a = test_rewind_determinism()
    for k, v in a.items(): print("   rewind determinism %-12s identical after rewind: %s | kernels agree: %s | derived/new globals created then dropped: %d" % (k, v['identical_after_rewind'], v['states_agree'], v['globals_created']))
    if not all(v['identical_after_rewind'] and v['states_agree'] for v in a.values()): FAIL.append('rewind determinism')
    ra, rb = test_loose_cache_after_rewind(); print("   loose_range memo across a rewind: before %d, node reusing the index after rewind %d (must be 0)" % (ra, rb))
    if rb != 0 or ra != 4243: FAIL.append('stale loose_cache')
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    stats, bad = run_random(N, seed)
    print("S5 probing protocol: %d scripts, %d steps, %d probes (%d succeeded in both), %d probes created derived globals that rewind dropped, %d materialize_from_node checks; explained: relaxed-guard %d, F13 %d, F16 %d" %
          (stats['scripts'], stats['steps'], stats['probes'], stats['probes_ok'], stats['derived_globals_created_and_dropped'], stats['materialized'], stats['relaxed'], stats['f13'], stats.get('f16', 0)))
    for b in bad[:2]: print("   MISMATCH:", str(b)[:500])
    if bad: FAIL.append('probing protocol')
    print("elapsed %.1fs" % (time.time() - t0)); print("S5 PROBE GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
