"""exp_bench.py -- throughput, memory and scaling of the whole stack, measured on this machine.

    python3 exp_bench.py DATASET_DIR --out runs/full/bench --model runs/full/ft/bc/s0/best.pt --synth runs/full/synth --workers 1,2,4

What it measures (bench.json, with the machine and the commit; one core unless said otherwise):

  world       the time to build the world of the bank, its size, the resident memory afterwards
  gym         decisions and probes per second through the Gymnasium interface (BankEnv) with uniformly random legal choices: the cost of the environment itself, observations included
  solvers     for each solver (blind, uniform, sample, greedy, ida, mcts, mw; the network ones need --model) on a fixed set of train problems: probes, decisions and actions per second, seconds per problem, the
              memory after the run.  A PROBE is one try of an action on a goal by the engine, whoever asks (ml_protocol): the unit the budgets of all the experiments are written in.
  workers     the same evaluation with 1, 2, 4, ... forked worker processes (the world is shared): wall-clock, probes per second, speed-up and efficiency against one worker
  training    steps per second of behaviour-cloning updates (forward + backward + optimiser, batch 64) on `--device`; synthetic proof extraction in theorems per second
  inference   decisions per second of the network inside the `sample` solver (batch of one decision, the engine's menu included)

Nothing is estimated: each number is a wall-clock measurement of the code the experiments run.  Numbers depend on the machine, so bench.json records it.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, math, random, argparse, tempfile, shutil, collections
import numpy as np
import torch
import bank as B, bank_split as BS
import gym_env as G
import ml_world as W, ml_meta as MT, ml_train as T, ml_data as D, ml_agents as AG, ml_evaluate as EV
import bank_env as BE

SOLVERS = ('blind', 'uniform', 'sample', 'greedy', 'ida', 'mcts', 'mw')


def pick_problems(bk, ddir, k):
    """k proved train problems, evenly spaced in the length of their reference proofs (deterministic)"""
    bank = json.load(open(os.path.join(ddir, 'bank.json')))
    cand = sorted((p['demonstration']['length'], p['name']) for p in bank['problems'] if 'demonstration' in p and p['split'] == 'train' and p['name'] in bk.placed and p['demonstration']['length'] >= 2)
    if not cand: cand = sorted((p['demonstration']['length'], p['name']) for p in bank['problems'] if 'demonstration' in p and p['name'] in bk.placed)
    if len(cand) <= k: return [n for _, n in cand]
    idx = sorted({int(round(i * (len(cand) - 1) / (k - 1))) for i in range(k)})
    return [cand[i][1] for i in idx]


def bench_gym(bk, ctx, eng, names, decisions=2000, time_limit=40.0, seed=0):
    env = BE.BankEnv(bk, names=names, ctx=ctx, engine=eng, rich=False, max_actions=60)
    rng = random.Random(seed); steps = probes = eps = 0; t0 = time.perf_counter()
    while steps < decisions and time.perf_counter() - t0 < time_limit:
        try: obs, info = env.reset(options={'problem': rng.choice(names)})
        except ValueError: continue
        eps += 1
        while True:
            n = int(obs['n_options'])
            if n <= 0: break
            obs, r, term, trunc, info = env.step(rng.randrange(n)); steps += 1
            if term or trunc: break
        probes += info['actions'] + info['refused']
    dt = time.perf_counter() - t0
    return {'decisions': steps, 'probes': probes, 'episodes': eps, 'seconds': round(dt, 3), 'decisions_per_s': steps / dt, 'probes_per_s': probes / dt}


def bench_solvers(bk, ctx, eng, net, names, specs, probes, time_limit, say=print):
    out = {}
    for spec in specs:
        name_, _ = AG.parse_spec(spec)
        if name_ in AG.NEEDS_NET and net is None: continue
        solver = AG.make_solver(spec, eng, net if name_ in AG.NEEDS_NET else None, ctx if name_ in AG.NEEDS_NET else None)
        tot = collections.Counter(); secs = 0.0; solved = 0
        for n in names:
            pl = bk.placed[n]
            row = solver.solve(pl.stmt, pl.premises('prefix'), probes, seed=0, name=n, regime='prefix', time_limit=time_limit)
            tot['probes'] += row.get('probes', 0); tot['decisions'] += row.get('decisions', 0) or 0; tot['actions'] += row.get('actions', 0) or 0; secs += row.get('seconds', 0.0); solved += int(bool(row.get('solved')))
        out[spec] = {'n': len(names), 'solved': solved, 'probes': tot['probes'], 'seconds': round(secs, 3), 'probes_per_s': tot['probes'] / secs if secs else None,
                     'decisions_per_s': tot['decisions'] / secs if secs and tot['decisions'] else None, 'actions_per_s': tot['actions'] / secs if secs and tot['actions'] else None,
                     'seconds_per_problem': secs / len(names), 'peak_rss_mb': MT.peak_rss_mb()}
        say('  %-10s %5.1f probes/s  %s solved of %d' % (spec, out[spec]['probes_per_s'] or 0, solved, len(names)))
    return out


def bench_workers(bk, ctx, eng, model, names, workers, probes, time_limit, say=print):
    solver = 'sample' if model else 'blind'
    res = {}
    for w in workers:
        d = tempfile.mkdtemp(prefix='bench_w%d_' % w)
        try:
            t0 = time.perf_counter()
            rows = EV.evaluate(bk, ctx, eng, [('bench', model)] if model else [], [solver], names, ('prefix',), probes, time_limit, d, workers=w, chunk=1, progress=lambda *a: None)
            dt = time.perf_counter() - t0
            pr = sum(r.get('probes', 0) for r in rows)
            res[str(w)] = {'seconds': round(dt, 3), 'probes': pr, 'probes_per_s': pr / dt, 'tasks': len(rows)}
            say('  %2d workers: %.1f s, %.1f probes/s' % (w, dt, pr / dt))
        finally:
            shutil.rmtree(d, ignore_errors=True)
    if res:
        base = res[str(min(workers))]
        for w in workers:
            r = res[str(w)]; r['speedup'] = base['seconds'] / r['seconds'] * (1.0); r['efficiency'] = r['speedup'] / (w / float(min(workers)))
    return res


def bench_training(a, bk, ctx, model, device, batches=12, say=print):
    data = T.build_steps(a.data, bk, ctx, G.Engine(bk.w, kinds=set(G.Kind), strict=True))
    steps = list(data['bank'].get('train', [])) + list(data['lib'].get('train', []))
    if len(steps) < 64: return {}
    net, cfg, _ = T.load(model, ctx.bags); net.train(); dev = torch.device(device); net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    rng = random.Random(0); bs = 64; times = []
    for i in range(batches + 2):
        idx = [rng.randrange(len(steps)) for _ in range(bs)]
        t0 = time.perf_counter()
        batch = D.collate([steps[j] for j in idx]); res = net(batch, aux={'size': 0.2, 'mgr': 0.2})
        opt.zero_grad(); res['loss'].backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if dev.type == 'cuda': torch.cuda.synchronize()
        if i >= 2: times.append(time.perf_counter() - t0)                       # (the first two batches warm caches up)
    sps = bs / float(np.mean(times))
    say('  training: %.0f steps/s (batch %d, %s)' % (sps, bs, device))
    return {'train_steps_per_s': sps, 'batch': bs, 'device': device, 'params': sum(p.numel() for p in net.parameters()), 'steps_available': len(steps)}


def bench_extract(a, bk, ctx, n=100, say=print):
    import ml_synth as MS
    recs = MS.load_records(a.synth, 'train', limit=n)
    if not recs: return {}
    t0 = time.perf_counter(); steps, stats = MS.extract_steps(recs, bk, ctx, bk.w.gid[B.LIBRARY_LIMIT], workers=0); dt = time.perf_counter() - t0
    say('  extraction: %.1f theorems/s' % (len(recs) / dt))
    return {'extract_theorems_per_s': len(recs) / dt, 'extract_steps_per_s': len(steps) / dt, 'extract_n': len(recs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('--out', required=True); ap.add_argument('--model', default=''); ap.add_argument('--synth', default='')
    ap.add_argument('--problems', type=int, default=12); ap.add_argument('--probes', type=int, default=200); ap.add_argument('--time-limit', type=float, default=20.0)
    ap.add_argument('--workers', default='1,2'); ap.add_argument('--worker-problems', type=int, default=16); ap.add_argument('--worker-probes', type=int, default=120)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=1); ap.add_argument('--no-gym', action='store_true'); ap.add_argument('--no-train', action='store_true')
    W.add_args(ap, lemmas=False)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter(); bk, ctx, eng = W.load(a); t_world = time.perf_counter() - t0
    world = {'seconds': round(t_world, 2), 'globals': bk.w.k.num_globals(), 'problems': len(bk.placed), 'rss_mb': MT.rss_mb()}
    model = a.model if a.model and os.path.exists(a.model) else ''
    net = None
    if model:
        net, cfg, _ = T.load(model, ctx.bags); net.eval(); world['params'] = sum(p.numel() for p in net.parameters())
    names = pick_problems(bk, a.data, a.problems)
    print('world %.1f s, %d globals, %.0f MB; %d problems for the solvers: %s' % (t_world, world['globals'], world['rss_mb'] or 0, len(names), names), flush=True)
    result = {'n_problems': len(names), 'problems': names, 'probes': a.probes, 'time_limit': a.time_limit}
    if not a.no_gym:
        print('gymnasium interface ...', flush=True); result['gym'] = bench_gym(bk, ctx, eng, names); print('  %.0f decisions/s, %.0f probes/s' % (result['gym']['decisions_per_s'], result['gym']['probes_per_s']), flush=True)
    print('solvers ...', flush=True)
    result['solvers'] = bench_solvers(bk, ctx, eng, net, names, SOLVERS, a.probes, a.time_limit)
    if 'sample' in result['solvers'] and result['solvers']['sample'].get('decisions_per_s'): world['infer_decisions_per_s'] = result['solvers']['sample']['decisions_per_s']
    ws = [int(x) for x in a.workers.split(',') if x]
    if len(ws) > 1:
        print('workers ...', flush=True)
        wn = pick_problems(bk, a.data, a.worker_problems)
        result['workers'] = bench_workers(bk, ctx, eng, model, wn, ws, a.worker_probes, a.time_limit)
    if model and not a.no_train:
        print('training ...', flush=True); tr = bench_training(a, bk, ctx, model, a.device); world.update(tr)
    if a.synth and os.path.exists(os.path.join(a.synth, 'train.jsonl.gz')):
        print('extraction ...', flush=True); world.update(bench_extract(a, bk, ctx))
    result['world'] = world
    result['meta'] = MT.run_meta(vars(a), {'peak_rss_mb': MT.peak_rss_mb()})
    MT.write_json(os.path.join(a.out, 'bench.json'), result)
    print('wrote', os.path.join(a.out, 'bench.json'))


if __name__ == '__main__':
    main()
