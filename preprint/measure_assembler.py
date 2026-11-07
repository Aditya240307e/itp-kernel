#!/usr/bin/env python3
"""measure_assembler.py -- can the policy interface say every recorded action?  (no PyTorch needed)

    python3 -I preprint/measure_assembler.py --out preprint/measurements/assembler.json           # about three minutes, 2 GB
    python3 -I preprint/measure_assembler.py --quick --out /tmp/assembler_quick.json              # the first ten proofs of each source: a check that the script runs

The policy interface (src/python/environment/gym_policy.py) says an action as a sequence of small typed choices.  A recorded proof is a sequence of portable actions; it can be used to train or to test a policy
only if the interface can say each of its actions, that is, if at every choice the option the action wants is among those the interface offers (otherwise `Missing`), and the action the choices assemble is the
recorded one (otherwise `ActionError`).  This script replays every recorded proof in the dataset in a fresh engine and, at each action, asks the interface to assemble it, exactly as the data builder of the
learner does (src/python/learner/ml_data.extract, without the features):

  bank / prefix     the reference proofs of the bank's problems, with every global before the theorem on offer
  bank / strict     the second proofs of the problems that have one, with the helpers written for the problem hidden
  library           the recorded proofs of the library's lemmas (453 of the 459 the recorder could turn into actions; the other six are listed in data/dataset_v2/library.json)

For each source: the proofs, the actions, how many the interface said exactly, how many it could not (by kind of action, with the reason), and the wall-clock time of the assembly.
The numbers depend on the machine; the file records it.
"""
import os, sys, time, json, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src', 'python')); sys.path.insert(0, HERE)
import itp_paths  # noqa: E402,F401
sys.setrecursionlimit(100000)
import bank_all, gym_env as G, gym_acts as ACTS, gym_policy as P  # noqa: E402
import measure as M  # noqa: E402                                  (the machine description)
clock = time.perf_counter


def assemble_proof(eng, stmt, premises, steps, source, name, stats, failures):
    """replay `steps` and assemble each action; the same walk as ml_data.extract"""
    eng.premises = premises
    p = eng.start(stmt)
    for si, st in enumerate(steps):
        if not p.n_goals: break
        g = eng.goal_of(p, 0); m = eng.menu(g)
        menu = m if st.get('kind') in ACTS.MENU_KINDS else ACTS.NO_MENU
        a = ACTS.from_portable(eng, menu, st)
        stats['actions'] += 1
        t = clock()
        try:
            P.decompose(eng, g, m, a); stats['ok'] += 1
        except P.Missing as e:
            stats['missing'] += 1; failures.append({'source': source, 'proof': name, 'index': si, 'kind': str(st.get('kind')), 'error': 'Missing', 'reason': str(e)[:160]})
        except Exception as e:
            stats['error'] += 1; failures.append({'source': source, 'proof': name, 'index': si, 'kind': str(st.get('kind')), 'error': type(e).__name__, 'reason': str(e)[:160]})
        stats['seconds'] += clock() - t
        eng.replay(p, a, m)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True); ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'dataset_v2'))
    ap.add_argument('--quick', action='store_true', help='the first ten proofs of each source')
    a = ap.parse_args(argv)
    bank = json.load(open(os.path.join(a.data, 'bank.json'))); lib = json.load(open(os.path.join(a.data, 'library.json')))
    out = {'machine': M.machine(), 'args': {k: (v.replace(ROOT, '<repo>') if isinstance(v, str) else v) for k, v in vars(a).items()}}
    print('building the world ...', flush=True)
    t0 = clock(); bk = bank_all.load(); t_world = clock() - t0
    print('  %.1f s, %d globals, %d problems' % (t_world, bk.w.k.num_globals(), len(bk.placed)), flush=True)
    eng = G.Engine(bk.w, kinds=set(G.Kind), strict=True)
    srcs = collections.OrderedDict((k, collections.Counter()) for k in ('bank / prefix', 'bank / strict', 'library')); proofs = collections.Counter(); failures = []
    lim = 10 if a.quick else None
    n = 0
    for rec in bank['problems']:
        if rec['name'] not in bk.placed: continue
        pl = bk.placed[rec['name']]
        for dkey, regime, src in (('demonstration', 'prefix', 'bank / prefix'), ('demonstration_strict', 'strict', 'bank / strict')):
            if dkey not in rec: continue
            if lim and proofs[src] >= lim: continue
            assemble_proof(eng, pl.stmt, pl.premises(regime), rec[dkey]['steps'], src, rec['name'], srcs[src], failures); proofs[src] += 1
    for l in lib['lemmas']:
        if lim and proofs['library'] >= lim: break
        gid = l['gid']; stmt = bk.w.T(bk.k.global_info(gid)[1])
        assemble_proof(eng, stmt, gid, l['steps'], 'library', l['name'], srcs['library'], failures); proofs['library'] += 1
    rows = {}
    for src, st in srcs.items():
        rows[src] = {'proofs': proofs[src], 'actions': st['actions'], 'said': st['ok'], 'missing': st['missing'], 'errors': st['error'], 'seconds': round(st['seconds'], 3),
                     'ms_per_action': (1e3 * st['seconds'] / st['actions']) if st['actions'] else None}
        print('  %-14s %4d proofs  %5d actions  said %5d  missing %3d  other errors %3d  (%.2f ms per action)' % (src, proofs[src], st['actions'], st['ok'], st['missing'], st['error'], rows[src]['ms_per_action'] or 0), flush=True)
    out['sources'] = rows; out['failures'] = failures
    tot = {k: sum(r[k] for r in rows.values()) for k in ('proofs', 'actions', 'said', 'missing', 'errors')}
    tot['seconds'] = round(sum(r['seconds'] for r in rows.values()), 3); out['total'] = tot
    out['failures_by_kind'] = dict(collections.Counter(f['kind'] for f in failures))
    out['world_seconds'] = round(t_world, 2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f: json.dump(out, f, indent=1, sort_keys=True, default=str)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
