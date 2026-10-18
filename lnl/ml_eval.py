"""ml_eval.py -- how many problems of the bank does a search solve with a given number of PROBES, with and without a learned policy?

    python3 ml_eval.py DATASET_DIR --split dev --probes 4000 --blind                      the baseline: gym_search.CostSearcher with its default costs (no learning)
    python3 ml_eval.py DATASET_DIR --split dev --probes 4000 --model model.pt             the policy-guided search (ml_search.PolicySearcher)

A search is deterministic, and its budget only truncates it, so one run at the largest budget gives the whole curve: a problem is solved at budget B iff the run found its proof after at most B probes.  Every
proof is replayed on a live proof and checked by the kernel (`solved` means a kernel-checked proof).  Rows: name, regime, level, split, solved, probes (when found, else the budget used), seconds, steps (length of
the proof found).
"""
import os, sys, json, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import bank_all, bank_split as BS
import gym_env as G, gym_search as S
import ml_data as D, ml_train as T, ml_search as MS


def run_blind(bk, eng, name, regime, probes, time_limit):
    pl = bk.placed[name]; eng.premises = pl.premises(regime)
    t0 = time.time()
    try:
        term, tree, s = S.prove_statement(eng, pl.stmt, max_depth=12, max_probes=probes, time_limit=time_limit, cost=True)
        return {'solved': term is not None, 'probes': s.probes if s is not None else 0, 'seconds': round(time.time() - t0, 2), 'steps': tree.size() if tree is not None else None}
    except Exception as e:
        return {'solved': False, 'probes': 0, 'seconds': round(time.time() - t0, 2), 'steps': None, 'error': '%s: %s' % (type(e).__name__, str(e)[:100])}


def run_policy(bk, eng, net, ctx, name, regime, probes, time_limit, **kw):
    pl = bk.placed[name]
    t0 = time.time()
    try:
        term, tree, s = MS.prove_statement(eng, pl.stmt, pl.premises(regime), net, ctx, max_probes=probes, time_limit=time_limit, **kw)
        row = {'solved': term is not None, 'probes': s.probes if s is not None else 0, 'seconds': round(time.time() - t0, 2), 'steps': tree.size() if tree is not None else None,
               'policy_seconds': round(s.t_policy, 2) if s is not None else 0.0, 'expansions': s.expansions if s is not None else 0}
        if term is not None and tree is not None: row['proof'] = MS.tree_portable(eng, pl.stmt, pl.premises(regime), tree)
        return row
    except Exception as e:
        import traceback
        return {'solved': False, 'probes': 0, 'seconds': round(time.time() - t0, 2), 'steps': None, 'error': '%s: %s' % (type(e).__name__, str(e)[:160])}


def evaluate(bk, names, method, probes, time_limit, net=None, ctx=None, regimes=('prefix', 'strict'), eng=None, progress=None, **kw):
    rows = []
    eng = eng or G.Engine(bk.w, kinds=set(G.Kind), strict=False) if method == 'policy' else (eng or G.Engine(bk.w))
    for n in names:
        pl = bk.placed[n]; done = {}
        for regime in regimes:
            prem = pl.premises(regime)
            if regime == 'strict' and prem != pl.premises('prefix') and not pl.problem.strict: continue          # (no reference proof that proves its own lemmas: the strict regime is not a task for it)
            if prem in done: row = dict(done[prem]); row['regime'] = regime; row['same_as'] = done[prem]['regime']
            else:
                row = run_blind(bk, eng, n, regime, probes, time_limit) if method == 'blind' else run_policy(bk, eng, net, ctx, n, regime, probes, time_limit, **kw)
                row['name'] = n; row['regime'] = regime
            row['level'] = pl.problem.level; row['split'] = BS.split_of(n); row['method'] = method
            done[prem] = row; rows.append(row)
            if progress: progress(row)
    return rows


def curve(rows, budgets):
    """{budget: number of rows solved with at most that many probes}"""
    return {b: sum(1 for r in rows if r['solved'] and r['probes'] <= b) for b in budgets}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('--split', default='dev'); ap.add_argument('--probes', type=int, default=4000); ap.add_argument('--time', type=float, default=120.0)
    ap.add_argument('--blind', action='store_true'); ap.add_argument('--model'); ap.add_argument('--out'); ap.add_argument('--regimes', default='prefix,strict'); ap.add_argument('--names', default='')
    ap.add_argument('--step-cost', type=float, default=0.2); ap.add_argument('--topk', type=int, default=24); ap.add_argument('--cap', type=float, default=9.0); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=1.0)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads); sys.setrecursionlimit(100000)
    bk = bank_all.load()
    names = a.names.split(',') if a.names else [n for n in bk.placed if BS.split_of(n) == a.split or a.split == 'all']
    regimes = tuple(a.regimes.split(','))
    prog = lambda r: print('%-26s %-7s %-6s probes %5d  %6.1fs  steps %s %s' % (r['name'], r['regime'], 'SOLVED' if r['solved'] else '-', r['probes'], r['seconds'], r['steps'], r.get('error', '')), flush=True)
    if a.blind:
        rows = evaluate(bk, names, 'blind', a.probes, a.time, regimes=regimes, progress=prog)
    else:
        ctx = D.DataContext(bk.w); net, cfg, extra = T.load(a.model, ctx.bags)
        rows = evaluate(bk, names, 'policy', a.probes, a.time, net=net, ctx=ctx, regimes=regimes, progress=prog, step_cost=a.step_cost, topk=a.topk, cap=a.cap, temperature=a.temperature)
    budgets = [b for b in (125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000) if b <= a.probes]
    print('rows %d; solved at budgets %s' % (len(rows), curve(rows, budgets)))
    if a.out: json.dump({'args': vars(a), 'rows': rows}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
