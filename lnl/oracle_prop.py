"""Random-script oracle in the Prop/Type world (zero tolerance): after EVERY tactic the partial proof must type-check and the goals must be well formed; every completed proof is verified by the kernel and,
at the end, by the independent checker (under the parameter rule and the elimination restriction)."""
import sys, random, time, functools
sys.setrecursionlimit(30000)
import harness as H
from harness import *
from s3_tests import V, app
import oracle_stats as o
from oracle_lib import goals_wellformed, goals_meta_free
import prop_world as pw
import s8_indep_tests as t8
import indep_checker as ic

T0, T1 = pw.T0, pw.T1

def statements(g):
    C = lambda n: ('const', g[n]); NAT, ZERO, S, EQ, LE, AND, OR, EX, FALSE = [C(n) for n in ('Nat', 'Zero', 'Succ', 'Eq', 'Le', 'And', 'Or', 'Ex', 'False')]
    eq = lambda A, a, b: app(EQ, A, a, b); le = lambda a, b: app(LE, a, b); add = lambda a, b: app(C('add'), a, b)
    chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)
    return {
     'refl_nat': chain([NAT], eq(NAT, V(0), V(0))),
     'symm_nat': chain([NAT, NAT, eq(NAT, V(0), V(1))], eq(NAT, V(1), V(0))),
     'trans_nat': chain([NAT, NAT, NAT, eq(NAT, V(0), V(1)), eq(NAT, V(1), V(2))], eq(NAT, V(0), V(2))),
     'cong_s': chain([NAT, NAT, eq(NAT, V(0), V(1))], eq(NAT, app(S, V(0)), app(S, V(1)))),
     'add_zero_r': chain([NAT], eq(NAT, add(V(0), ZERO), V(0))),
     'add_succ_r': chain([NAT, NAT], eq(NAT, add(V(0), app(S, V(1))), app(S, add(V(0), V(1))))),
     'zero_le': chain([NAT], le(ZERO, V(0))),
     'le_trans': chain([NAT, NAT, NAT, le(V(0), V(1)), le(V(1), V(2))], le(V(0), V(2))),
     'le_succ': chain([NAT, NAT, le(V(0), V(1))], le(V(0), app(S, V(1)))),
     'succ_le_succ': chain([NAT, NAT, le(V(0), V(1))], le(app(S, V(0)), app(S, V(1)))),
     'le_zero_eq': chain([NAT, le(V(0), ZERO)], eq(NAT, V(0), ZERO)),
     'no_succ_le_zero': chain([NAT, le(app(S, V(0)), ZERO)], FALSE),
     'zero_ne_succ': chain([NAT, eq(NAT, ZERO, app(S, V(0)))], FALSE),
     'and_swap': chain([T0, T0, app(AND, V(0), V(1))], app(AND, V(1), V(0))),
     'or_swap': chain([T0, T0, app(OR, V(0), V(1))], app(OR, V(1), V(0))),
     'ex_id': chain([T1, ('pi', V(0), T0), app(EX, V(0), V(1))], app(EX, V(0), V(1))),
     'le_inv': chain([NAT, NAT, le(V(1), app(S, V(0)))], app(OR, le(V(1), V(0)), eq(NAT, V(1), app(S, V(0))))),
    }

def run(N, seed, eq=True):
    k, g = pw.build_prop_world(); stmts = statements(g); names = list(stmts); rng = random.Random(seed)
    ind = {g[n] for n in ('Nat', 'Eq', 'Le', 'And', 'Or', 'Ex', 'False', 'True')}
    consts = [g[n] for n in ('refl', 'cong', 'symm', 'trans', 'transport', 'le_n', 'le_S', 'le_trans', 'Succ', 'add', 'conj', 'inl', 'inr', 'ex_intro', 'tt', 'zero_le')]
    head = lambda x: head(x[1]) if x[0] == 'app' else x
    tot = {}; first_bad = {}; completed = verified = rebuilds = 0; proofs = []; t0 = time.time()
    for i in range(N):
        sname = rng.choice(names); stmt = build_new(k, l2h(stmts[sname], 0)); ps = k.start_proof(stmt); ops = []
        for _ in range(10):
            if ps.num_goals() == 0: break
            nh = ps.goal_num_hyps(0); tgt = parse_raw(k.dump_raw(ps.goal_target(0)), True)
            hyps = [parse_raw(k.dump_raw(ps.goal_hyp_type(0, j)), True) for j in range(nh)]
            indh = [j for j, h in enumerate(hyps) if head(h)[0] == 'const' and head(h)[1] in ind]
            eqh = [j for j, h in enumerate(hyps) if h[0] == 'app' and h[1][0] == 'app' and h[1][1][0] == 'app' and h[1][1][1] == ('const', g['Eq']) and not any(w in str(h) for w in ("'lam'", "'pi'", "'match'", "'fix'", "'meta'"))]
            r = rng.random()
            if tgt[0] == 'pi' and r < 0.35: op = ('intro',)
            elif indh and tgt[0] != 'pi' and r < 0.62: op = (rng.choice(['cases', 'induct']), rng.choice(indh))
            elif eqh and tgt[0] != 'pi' and r < 0.70: j = rng.choice(eqh); op = ('rewrite', j, hyps[j][1][2], hyps[j][2])
            elif nh and r < 0.84: op = ('apply', ('fvar', rng.randrange(nh)))
            else: op = ('apply', ('const', rng.choice(consts)))
            try: o.apply_op(k, g, ps, op, eq)
            except Exception:
                ps = o.replay(k, g, stmt, ops, eq); rebuilds += 1; continue
            ops.append(op); d = tot.setdefault(op[0], [0, 0, 0, 0])      # [ok, ill-typed partial proof, ill-formed goals, skipped]
            if ps.num_pending() > 0 or not goals_meta_free(k, ps): d[3] += 1; continue
            good = k.check_open(ps.extract(), stmt, [])[0]; wf = goals_wellformed(k, ps) if ps.num_goals() else True
            if good and wf: d[0] += 1
            else:
                d[1] += (not good); d[2] += (not wf)
                if op[0] not in first_bad or len(ops) < len(first_bad[op[0]][1]): first_bad[op[0]] = (sname, list(ops), 'partial proof: ' + ('ok' if good else 'ILL-TYPED') + ' | goals well-formed: ' + str(wf))
        if ps.num_goals() == 0 and ps.is_complete():
            completed += 1; ok = bool(k.check_open(ps.extract(), stmt, [])[0]); verified += ok
            if ok: proofs.append((sname, parse_raw(k.dump_raw(ps.extract()), True)))
    chk = ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=4_000_000)
    indep_ok = sum(1 for sname, pr in proofs if chk.check_closed(pr, l2h(stmts[sname], 0))[0])
    return dict(tot=tot, completed=completed, verified=verified, indep_verified=indep_ok, indep_total=len(proofs), rebuilds=rebuilds, seconds=time.time() - t0, first_bad=first_bad)

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500; FAIL = []
    for seed in (41, 42):
        r = run(N, seed); bad = sum(x[1] + x[2] for x in r['tot'].values()) + (r['completed'] - r['verified']) + (r['indep_total'] - r['indep_verified'])
        print("prop world, seed %d, %d scripts (%.0fs): per tactic [ok, ill-typed, ill-formed, skipped] %s | completed %d, kernel-verified %d, independently verified %d/%d | failures %d" %
              (seed, N, r['seconds'], dict(sorted(r['tot'].items())), r['completed'], r['verified'], r['indep_verified'], r['indep_total'], bad))
        for kk, vv in list(r['first_bad'].items())[:2]: print("     first failure after", kk, ":", vv[0], vv[1][-4:], vv[2][:140])
        if bad: FAIL.append('seed %d' % seed)
    print("PROP ORACLE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
