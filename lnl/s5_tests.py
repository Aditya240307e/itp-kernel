"""S5 gate (part 1): the library index. apply_candidates must return EXACTLY the reference's candidate list (same entries, same order) for every goal of every state."""
import sys, random, time, functools
import harness as hn
from harness import *
import s3_tests as s3
from s3_tests import V, app, Lockstep, library
import s4_tests as t4

T0 = ('type', 0)
def statements5(g):
    NAT, EQ = ('const', g['Nat']), ('const', g['Eq'])
    eq = lambda a, b: app(EQ, NAT, a, b)
    P = ('pi', NAT, T0)
    chain = lambda doms, goal: functools.reduce(lambda t, d: ('pi', d, t), reversed(doms), goal)
    st = dict(t4.statements4(g))
    st.update({
      'pred_transport': chain([P, NAT, NAT, eq(V(1), V(2)), app(V(0), V(1))], app(V(0), V(2))),          # goal P b, hypotheses h : a = b, p : P a
      'pred_hyp':       chain([P, NAT, app(V(0), V(1))], app(V(0), V(1))),
      'pred_two':       chain([P, P, NAT, app(V(0), V(2)), app(V(1), V(2))], app(V(1), V(2))),
      'pred_pi_goal':   chain([P], ('pi', NAT, ('pi', app(V(0), V(1)), app(V(0), V(1))))),                # after one intro: a Pi target headed by a CONTEXT variable with bound arguments
      'pi_bound_head':  ('pi', P, ('pi', NAT, ('pi', app(V(0), V(1)), app(V(0), V(1))))),                  # the initial goal's conclusion is headed by a BOUND variable
      # a hypothesis whose OWN conclusion is headed by a bound variable: its key depends on the hypothesis's position
      'pred_pi_hyp':    chain([P, ('pi', P, app(V(1), ('const', g['Zero'])))], app(V(0), ('const', g['Zero']))),
      'pi_bound_pair':  ('pi', P, ('pi', P, ('pi', NAT, ('pi', app(V(0), V(2)), app(V(1), V(2)))))),
    })
    return st

def norm_candidates(k, nodes, new):
    out = []
    for n in nodes:
        t = parse_raw(k.dump_raw(n), new)
        out.append(('const', t[1]) if t[0] == 'const' else ('hyp', t[1]) if t[0] in ('var', 'fvar') else ('other', str(t)[:40]))
    return out

def head_kind(tree):
    while tree[0] == 'pi': tree = tree[2]
    while tree[0] == 'app': tree = tree[1]
    return tree[0]

def check(L, stats, sname, ops):
    lib_r = L.kref.index_library(); lib_n = L.knew.index_library()
    for i in range(L.psr.num_goals()):
        cr = norm_candidates(L.kref, L.kref.apply_candidates(L.psr, lib_r, target_meta_id=L.psr.goal_meta_id(i)), False)
        cn = norm_candidates(L.knew, L.knew.apply_candidates(L.psn, lib_n, target_meta_id=L.psn.goal_meta_id(i)), True)
        tgt = parse_raw(L.knew.dump_raw(L.psn.goal_target(i)), True)
        stats['compared'] += 1; stats['nonempty'] += bool(cr); stats['hyp_candidates'] += any(c[0] == 'hyp' for c in cr)
        stats['head_' + head_kind(tgt)] = stats.get('head_' + head_kind(tgt), 0) + 1
        stats['pi_target'] += tgt[0] == 'pi'; stats['distinct'].add(tuple(cr))
        if cr != cn: return (sname, ops, i, cr, cn)
    return None

def run_random(N, seed, max_steps=8):
    kref, knew, g, defs = library(); stmts = statements5(g); names = list(stmts); rng = random.Random(seed)
    stats = dict(compared=0, nonempty=0, hyp_candidates=0, pi_target=0, distinct=set(), scripts=0); bad = []
    consts = [g[n] for n in ('refl', 'cong', 'symm', 'trans', 'transport', 'le_n', 'le_S', 'le_trans', 'Succ', 'add')]
    ind = {g['Nat'], g['Eq'], g['Le']}
    head = lambda x: head(x[1]) if x[0] == 'app' else x
    for i in range(N):
        sname = rng.choice(names); L = Lockstep(kref, knew, g, stmts[sname]); ops = []; stats['scripts'] += 1
        m = check(L, stats, sname, ops)
        if m: bad.append(m)
        for _ in range(max_steps):
            if bad or L.psr.num_goals() == 0: break
            nh = L.psr.goal_num_hyps(0); tgt = parse_raw(kref.dump_raw(L.psr.goal_target(0)), False)
            hyps = [parse_raw(kref.dump_raw(L.psr.goal_hyp_type(0, j)), False) for j in range(nh)]
            indh = [j for j, h in enumerate(hyps) if head(h)[0] == 'const' and head(h)[1] in ind]
            r = rng.random()
            if tgt[0] == 'pi' and r < 0.45: op = ('intro',)
            elif indh and tgt[0] != 'pi' and r < 0.62: op = (rng.choice(['cases', 'induct']), rng.choice(indh))
            elif nh and r < 0.82: op = ('apply', V(rng.randrange(nh)))
            else: op = ('apply', ('const', rng.choice(consts)))
            ok, same, st = L.step(op); ops.append(op)
            if L.stopped or not same:                              # the two kernels' states differ: only an EXPLAINED category may end a script silently
                cat = t4.explain_divergence(L, stmts[sname], op, st) if not same or L.stopped else None
                stats['explained_' + str(cat)] = stats.get('explained_' + str(cat), 0) + 1
                if cat is None: bad.append(('unexplained lockstep divergence', sname, ops[-3:], str(st)[:200]))
                break
            if ok:
                m = check(L, stats, sname, ops)
                if m: bad.append(m); break
        if bad: break
    return stats, bad

if __name__ == '__main__':
    t0 = time.time(); N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    stats, bad = run_random(N, seed)
    heads = {k[5:]: v for k, v in stats.items() if k.startswith('head_')}
    expl = {k[10:]: v for k, v in stats.items() if k.startswith('explained_')}
    print("S5 candidates: %d scripts, %d goal states compared; non-empty candidate lists %d, with hypothesis candidates %d, Pi-targets %d, distinct lists %d; target heads %s; explained lockstep divergences %s" %
          (stats['scripts'], stats['compared'], stats['nonempty'], stats['hyp_candidates'], stats['pi_target'], len(stats['distinct']), heads, expl))
    for b in bad[:2]: print("   MISMATCH:", str(b)[:420])
    print("elapsed %.1fs" % (time.time() - t0)); print("S5 GATE:", "PASS" if not bad else "FAIL"); sys.exit(1 if bad else 0)
