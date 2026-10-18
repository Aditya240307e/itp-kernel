"""bank_baseline.py -- how hard is each problem of the bank for BLIND proof search?

The plain searcher of gym_search.py (iterative deepening over the engine's menus, no learning) is run on the statement of every problem in a regime ('prefix': the library before the theorem; 'strict': without the
lemmas written for the problem) with a budget of probes. What it finds is replayed on a live proof and checked by the kernel (`gym_search.commit`), so a "solved" is a kernel-checked proof; a problem it does not solve
says nothing about the problem except that this search with this budget did not. The numbers go into the dataset as a rough, REPRODUCIBLE difficulty (probes, not seconds, decide):

    python3 bank_baseline.py                          all problems, both regimes, 4000 probes
    python3 bank_baseline.py --probes 20000 --json out.json ineq_cauchy_two imo1984_p2

    from bank_baseline import baseline
    rows = baseline(bank, ['ineq_cauchy_two'], regimes=('prefix', 'strict'), probes=4000)
"""
import os, sys, time, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_env as G, gym_search as S
from gym_env import Kind


def baseline(bank, names=None, regimes=('prefix', 'strict'), probes=4000, max_depth=10, time_limit=120.0, cost=None, engine=None, progress=None):
    """[{'name', 'regime', 'solved', 'probes', 'seconds', 'size', 'depth', 'kinds'}] for the problems (default: all) in the regimes; a problem whose strict regime is the prefix regime is run once"""
    eng = engine or G.Engine(bank.w)
    rows = []
    for n in (names or list(bank.placed)):
        pl = bank.placed[n]
        done = {}
        for regime in regimes:
            prem = pl.premises(regime)
            if prem in done:                                           # (no lemmas of its own: both regimes are one)
                row = dict(done[prem]); row['regime'] = regime; row['same_as'] = done[prem]['regime']; rows.append(row); continue
            eng.premises = prem
            t0 = time.time()
            try:
                term, tree, s = S.prove_statement(eng, pl.stmt, max_depth=max_depth, max_probes=probes, time_limit=time_limit, cost=cost)
                ok = term is not None
                row = {'name': n, 'regime': regime, 'solved': ok, 'probes': s.probes if s is not None else 0, 'seconds': round(time.time() - t0, 2),
                       'size': tree.size() if tree is not None else None, 'depth': tree.depth() if tree is not None else None,
                       'kinds': dict(collections.Counter(Kind(k).name if isinstance(k, int) else str(k) for k in (tree.kinds() if tree is not None else [])))}
            except Exception as e:                                       # (a bug in the search is a row, not a crash: the table says which)
                row = {'name': n, 'regime': regime, 'solved': False, 'probes': 0, 'seconds': round(time.time() - t0, 2), 'size': None, 'depth': None, 'kinds': {}, 'error': '%s: %s' % (type(e).__name__, str(e)[:120])}
            done[prem] = row; rows.append(row)
            if progress: progress(row)
        eng.premises = None
    return rows


def table(rows):
    lines = ['%-26s %-7s %-6s %7s %7s %5s %5s  %s' % ('problem', 'regime', 'solved', 'probes', 'secs', 'size', 'depth', 'actions')]
    for r in rows:
        lines.append('%-26s %-7s %-6s %7s %7.1f %5s %5s  %s%s' % (r['name'], r['regime'], 'yes' if r['solved'] else 'no', r['probes'], r['seconds'], r['size'] if r['size'] is not None else '-', r['depth'] if r['depth'] is not None else '-',
                                                                 ' '.join('%s:%d' % kv for kv in sorted(r['kinds'].items())), ('  [' + r['error'] + ']') if r.get('error') else ''))
    n = sum(1 for r in rows if r['solved'])
    lines.append('%d of %d solved' % (n, len(rows)))
    return '\n'.join(lines)


def main(argv):
    import bank_lib
    probes = 4000; out = None; names = []; regimes = ('prefix', 'strict'); cost = None; tl = 120.0
    it = iter(argv)
    for a in it:
        if a == '--probes': probes = int(next(it))
        elif a == '--json': out = next(it)
        elif a == '--regime': regimes = (next(it),)
        elif a == '--cost': cost = True
        elif a == '--time': tl = float(next(it))
        else: names.append(a)
    bank = bank_lib.load()
    rows = baseline(bank, names or None, regimes=regimes, probes=probes, cost=cost, time_limit=tl, progress=lambda r: print('%-26s %-7s %s probes=%s %.1fs' % (r['name'], r['regime'], 'SOLVED' if r['solved'] else '-', r['probes'], r['seconds']), flush=True))
    print(table(rows))
    if out:
        with open(out, 'w') as f: json.dump({'probes': probes, 'rows': rows}, f, indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
