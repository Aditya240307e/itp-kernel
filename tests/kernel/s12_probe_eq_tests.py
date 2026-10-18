"""Probing protocol x index generalisation (hybrid only; the equation path does not exist in the reference).
Per step: probes = checkpoint / fork_single_goal / tactic / read off / rewind. Required, with ZERO tolerance:
  * rewind never throws and restores the global count (a probe that is the first to need the barrier creates a global and must drop it);
  * a repeated probe gives the identical result (rewind restores enough of the kernel for a replay to be deterministic);
  * at the end of each script the probed run's goal states equal those of a fresh PROBE-FREE replay of the same tactics (probing leaves no residue);
  * every completed proof verifies."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, random, time
sys.setrecursionlimit(30000)
import harness as hn
from harness import *
from s3_tests import library, V, app
import s4_tests as t4
import oracle_stats as o

def _norm(t, m):
    """rename metavariable ids to their order of first appearance: two proof states of one kernel allocate different ids for the same situation"""
    if isinstance(t, tuple):
        if len(t) == 2 and t[0] == 'meta': return ('meta', m.setdefault(t[1], len(m)))
        return tuple(_norm(x, m) for x in t)
    return t

def canon(k, ps):
    raw = tuple((tuple(parse_raw(k.dump_raw(ps.goal_hyp_type(i, j)), True) for j in range(ps.goal_num_hyps(i))), parse_raw(k.dump_raw(ps.goal_target(i)), True)) for i in range(ps.num_goals()))
    return _norm(raw, {})

def run(N, seed, params, probes_per_step=3, max_steps=9):
    kref, knew, g, defs = library()
    if params: knew.set_option('parameters', True)
    stmts = dict(t4.statements4(g))
    C = lambda n: ('const', g[n]); NAT, ZERO, S, LE, EQ = C('Nat'), C('Zero'), C('Succ'), C('Le'), C('Eq')
    extra = {   # TRUE statements whose hypothesis has a concrete index: the equation path (and so the barrier) is hit constantly
      'E1: n <= 0 -> n = 0':           ('pi', NAT, ('pi', app(LE, V(0), ZERO), app(EQ, NAT, V(0), ZERO))),
      'E2: b <= S a -> b <= S a':      ('pi', NAT, ('pi', NAT, ('pi', app(LE, V(1), app(S, V(0))), app(LE, V(1), app(S, V(0)))))),
      'E3: S n <= S 0 -> n <= 0':      ('pi', NAT, ('pi', app(LE, app(S, V(0)), app(S, ZERO)), app(LE, V(0), ZERO)))}
    stmts.update(extra); names = list(t4.statements4(g)) + list(extra) * 40; rng = random.Random(seed)
    ind = {g['Nat'], g['Eq'], g['Le']}; consts = [g[n] for n in ('refl', 'cong', 'symm', 'trans', 'transport', 'le_n', 'le_S', 'le_trans', 'Succ', 'add')]
    head = lambda x: head(x[1]) if x[0] == 'app' else x
    st = dict(scripts=0, steps=0, probes=0, probes_ok=0, probes_created_globals=0, rewind_exceptions=0, globals_not_restored=0, nondeterministic=0, residue=0, completed=0, verified=0, induct_probes=0)
    first_bad = []
    def gen_op(ps):
        nh = ps.goal_num_hyps(0); tgt = parse_raw(knew.dump_raw(ps.goal_target(0)), True)
        hyps = [parse_raw(knew.dump_raw(ps.goal_hyp_type(0, j)), True) for j in range(nh)]
        indh = [j for j, h in enumerate(hyps) if head(h)[0] == 'const' and head(h)[1] in ind]
        eqh = [j for j, h in enumerate(hyps) if h[0] == 'app' and h[1][0] == 'app' and h[1][1][0] == 'app' and h[1][1][1] == ('const', g['Eq']) and not any(w in str(h) for w in ("'lam'", "'pi'", "'match'", "'fix'", "'meta'"))]
        r = rng.random()
        if tgt[0] == 'pi' and r < 0.35: return ('intro',)
        if indh and tgt[0] != 'pi' and r < 0.66: return (rng.choice(['cases', 'induct', 'induct']), rng.choice(indh))
        if eqh and tgt[0] != 'pi' and r < 0.72: j = rng.choice(eqh); return ('rewrite', j, hyps[j][1][2], hyps[j][2])
        if nh and r < 0.86: return ('apply', ('fvar', rng.randrange(nh)))
        return ('apply', ('const', rng.choice(consts)))
    for i in range(N):
        sname = rng.choice(names); stmt = build_new(knew, l2h(stmts[sname], 0)); ps = knew.start_proof(stmt); ops = []; st['scripts'] += 1; bad = None
        if sname in extra:      # start in the interesting state: the quantifiers introduced, so the random probes reach induct on the concrete-index hypothesis
            while parse_raw(knew.dump_raw(ps.goal_target(0)), True)[0] == 'pi': ps.intro('x'); ops.append(('intro',))
        for _ in range(max_steps):
            if ps.num_goals() == 0: break
            first_op = None; first_res = None
            for pi in range(probes_per_step):
                op = gen_op(ps); n_glob = len(knew.list_globals()); cp = knew.checkpoint()
                fk = knew.fork_single_goal(ps, ps.goal_meta_id(0)); ok = True
                try: o.apply_op(knew, g, fk, op, eq=True)
                except Exception: ok = False
                res = canon(knew, fk) if ok else None; created = len(knew.list_globals()) - n_glob
                try: knew.rewind(cp)
                except Exception as e: st['rewind_exceptions'] += 1; bad = bad or ('rewind threw', sname, ops[-3:], op, str(e)[:100]); break
                st['probes'] += 1; st['probes_ok'] += ok; st['probes_created_globals'] += created > 0; st['induct_probes'] += op[0] == 'induct'
                if len(knew.list_globals()) != n_glob: st['globals_not_restored'] += 1; bad = bad or ('globals not restored', sname, op)
                if pi == 0: first_op, first_res = op, res
            if bad: break
            if first_op is not None:     # determinism: the very same probe again, after all the others were rewound
                cp = knew.checkpoint(); fk = knew.fork_single_goal(ps, ps.goal_meta_id(0)); ok = True
                try: o.apply_op(knew, g, fk, first_op, eq=True)
                except Exception: ok = False
                res2 = canon(knew, fk) if ok else None; knew.rewind(cp)
                if res2 != first_res: st['nondeterministic'] += 1; bad = bad or ('probe not deterministic', sname, ops[-3:], first_op)
            if bad: break
            op = gen_op(ps)
            try: o.apply_op(knew, g, ps, op, eq=True)
            except Exception:
                ps = o.replay(knew, g, stmt, ops, eq=True); continue
            ops.append(op); st['steps'] += 1
        if bad: first_bad.append(bad); break
        # no residue: a fresh probe-free replay reaches the same goal states
        fresh = o.replay(knew, g, stmt, ops, eq=True)
        if canon(knew, fresh) != canon(knew, ps): st['residue'] += 1; first_bad.append(('probing left residue', sname, ops)); break
        if ps.num_goals() == 0 and ps.is_complete():
            st['completed'] += 1; st['verified'] += bool(knew.check_open(ps.extract(), stmt, [])[0])
    return st, first_bad

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300; FAIL = []; t0 = time.time()
    for params in (True, False):
        st, bad = run(N, 31 if params else 32, params)
        print("%s: %s" % ("parameters ON" if params else "legacy", st))
        for b in bad[:2]: print("    FIRST FAILURE:", str(b)[:300])
        if bad or st['rewind_exceptions'] or st['globals_not_restored'] or st['nondeterministic'] or st['residue'] or st['completed'] != st['verified'] or st['probes_created_globals'] < 1 or st['induct_probes'] < 20:
            FAIL.append(('params' if params else 'legacy') + ': ' + str(bad[:1]) if bad else ('params' if params else 'legacy') + ' counts')
    print("elapsed %.0fs" % (time.time() - t0)); print("S12 PROBE-EQ GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
