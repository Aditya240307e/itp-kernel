"""Self-consistency oracle for the hybrid tactics (no reference involved). After every successful tactic, in states with no unsolved meta in a goal's
types and no pending constraint:  (a) the partial proof type-checks against the original statement, (b) every goal's hypotheses/target are well-formed.
A failed tactic may leave side effects (solved metas, queued constraints), so the state is REBUILT by replaying the successful tactics, as the search does.
Failures after a concrete-index induction (F12: the generic Fix cannot supply the specialised binder types) are attributed, not exempted silently."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, os, random, time
import harness as hn
from harness import *
from s3_tests import library, V, app
import s4_tests as t4
from oracle_lib import *

def apply_op(k, g, ps, op, eq=False):
    if op[0] == 'intro': ps.intro('x')
    elif op[0] == 'cases': ps.cases(op[1])
    elif op[0] == 'induct': ps.induct(op[1], eq_gid=g['Eq'], refl_gid=g['refl']) if eq else ps.induct(op[1])
    elif op[0] == 'rewrite': ps.rewrite(build_new(k, ('fvar', op[1])), build_new(k, op[2]), build_new(k, op[3]), g['transport'], g['symm'])
    else: ps.apply(build_new(k, op[1]))

def replay(k, g, stmt, ops, eq=False):
    ps = k.start_proof(stmt)
    for op in ops: apply_op(k, g, ps, op, eq)
    return ps

def run(N, seed, verbose=True, mode2_only=False, no_q=False, params=False, eq=False):
    kref, knew, g, defs = library()
    if params: knew.set_option('parameters', True)
    stmts = t4.statements4(g); rng = random.Random(seed); names = list(stmts)
    ind = {g['Nat'], g['Eq'], g['Le']}; consts = [g[n] for n in ('refl', 'cong', 'symm', 'trans', 'transport', 'le_n', 'le_S', 'le_trans', 'Succ', 'add', 'Q')]
    if no_q: consts = [c for c in consts if c != g['Q']]
    head = lambda x: head(x[1]) if x[0] == 'app' else x
    mode2 = lambda h: is_mode2(h, g, knew if params else None)
    tot = {}; first_bad = {}; completed = verified = rebuilds = 0; t0 = time.time()
    for i in range(N):
        sname = rng.choice(names); stmt = build_new(knew, l2h(stmts[sname], 0)); ps = knew.start_proof(stmt); ops = []; m1 = False
        for _ in range(10):
            if ps.num_goals() == 0: break
            nh = ps.goal_num_hyps(0); tgt = parse_raw(knew.dump_raw(ps.goal_target(0)), True)
            hyps = [parse_raw(knew.dump_raw(ps.goal_hyp_type(0, j)), True) for j in range(nh)]
            indh = [j for j, h in enumerate(hyps) if head(h)[0] == 'const' and head(h)[1] in ind]
            eqh = [j for j, h in enumerate(hyps) if h[0] == 'app' and h[1][0] == 'app' and h[1][1][0] == 'app' and h[1][1][1] == ('const', g['Eq']) and not any(w in str(h) for w in ("'lam'", "'pi'", "'match'", "'fix'", "'meta'"))]
            r = rng.random()
            if tgt[0] == 'pi' and r < 0.35: op = ('intro',)
            elif indh and tgt[0] != 'pi' and r < 0.62:
                op = (rng.choice(['cases', 'induct']), rng.choice(indh))
                if mode2_only and op[0] == 'induct' and not mode2(hyps[op[1]]): continue
            elif eqh and tgt[0] != 'pi' and r < 0.70: j = rng.choice(eqh); op = ('rewrite', j, hyps[j][1][2], hyps[j][2])
            elif nh and r < 0.84: op = ('apply', ('fvar', rng.randrange(nh)))
            else: op = ('apply', ('const', rng.choice(consts)))
            try: apply_op(knew, g, ps, op, eq)
            except Exception:
                ps = replay(knew, g, stmt, ops, eq); rebuilds += 1; continue
            ops.append(op)
            if op[0] == 'induct' and not mode2(hyps[op[1]]) and not eq: m1 = True      # with equations a concrete index is supported: nothing is attributed
            d = tot.setdefault(op[0], [0, 0, 0, 0, 0])      # [ok, ill-typed partial proof, ill-formed goals, skipped, attributed to F12]
            if ps.num_pending() > 0 or not goals_meta_free(knew, ps): d[3] += 1; continue
            good = knew.check_partial(ps.extract(), stmt, [])[0]; wf = goals_wellformed(knew, ps) if ps.num_goals() else True
            if good and wf: d[0] += 1
            elif m1: d[4] += 1
            else:
                d[1] += (not good); d[2] += (not wf)
                if op[0] not in first_bad or len(ops) < len(first_bad[op[0]][1]): first_bad[op[0]] = (sname, list(ops), 'partial proof: ' + ('ok' if good else 'ILL-TYPED') + ' | goals well-formed: ' + str(wf))
        if ps.num_goals() == 0 and ps.is_complete():
            completed += 1; verified += bool(knew.check_open(ps.extract(), stmt, [])[0])
    if verbose:
        print("oracle over %d scripts (%.0fs, %d state rebuilds after failed tactics); per tactic [ok, ill-typed, ill-formed, skipped, attributed-F12]:" % (N, time.time() - t0, rebuilds))
        for k, v in sorted(tot.items()): print("   %-8s %s" % (k, v))
        print("   completed proofs: %d, verified by the kernel: %d" % (completed, verified))
        for k, v in first_bad.items(): print("   first failure after", k, ":", v[0], v[1][-5:], v[2])
    return tot, completed, verified, first_bad

if __name__ == '__main__':
    tot, c, v, fb = run(int(sys.argv[1]) if len(sys.argv) > 1 else 2000, int(sys.argv[2]) if len(sys.argv) > 2 else 1, mode2_only=bool(os.environ.get('ORACLE_MODE2')), no_q=bool(os.environ.get('ORACLE_NOQ')))
    bad = sum(x[1] + x[2] for x in tot.values()) + (c - v)
    print("ORACLE:", "PASS" if bad == 0 else "FAIL (%d unattributed)" % bad); sys.exit(1 if bad else 0)
