"""ml_evaluate.py -- measure provers on the problems of the bank, by the protocol of ml_protocol: kernel-checked proofs within a budget of probes.

    python3 ml_evaluate.py DATASET_DIR --split dev,test --solver blind --solver ida --solver mcts --model bc=runs/bc/s0/best.pt --probes 2000 --out runs/eval/bc_s0 --workers 4
    python3 ml_evaluate.py DATASET_DIR --split dev --solver blind --solver uniform --probes 500 --out runs/eval/baselines_dev

Every (model, solver) pair is run on every task (a problem in a regime) of the split(s); a solver that needs no network (blind, uniform) runs without a model.  The result is a directory with

    rows.jsonl     one row per (model, solver, task): method, model, spec, seed, name, regime, level, split, proved, origin, source, family, solved, probes, decisions, seconds, steps, proof (the portable
                   actions, when solved)
    meta.json      the arguments, the machine, the code (git commit), the dataset and world, the lemmas
    recheck.json   with --recheck: for every solved row the independent checker's verdict on the proof (checker/recheck.py)

`seed` in a row is the REPLICATE: the training seed of the model (read from its checkpoint, or --seed); a model-free solver has seed 0.  The run is RESUMABLE: a row already in rows.jsonl (same method, seed, name,
regime) is not computed again.  A deterministic solver (blind, ida) is run once at the largest budget and the curve "solved within B probes" is read off its rows (ml_aggregate); a seeded solver (sample, mcts, mw,
uniform) is seeded by the model's seed, so replicates are independent runs of the whole pipeline.  `--workers N` spreads the tasks over N forked processes that share the world.

A model trained with a library of bootstrapped lemmas (ml_ei --bootstrap) must be evaluated with it: `--lemmas auto` reads the lemma file next to the checkpoint (the checkpoint says which, and the library's
digest, which must match).  Models with different libraries are evaluated in separate runs.

Nothing is trusted: a row says `solved` only for a proof that was replayed in the engine and checked by the kernel; `--recheck` adds the verdict of the second, independently written checker.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, argparse, collections
import torch
import bank_split as BS
import gym_env as G, gym_acts as ACTS
import ml_train as T, ml_protocol as PR, ml_agents as AG, ml_world as W, ml_meta as MT
from ml_pool import Pool


def tasks_of(bk, names, regimes):
    """[(name, regime)]: the tasks of the problems (a regime that offers the same task as another is dropped: ml_protocol.regimes_of)"""
    out = []
    for n in names:
        for regime, same in PR.regimes_of(bk.placed[n], regimes):
            if same is None: out.append((n, regime))
    return out


def read_rows(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: rows.append(json.loads(line))
                    except ValueError: pass                      # (a line cut off by a crash)
    return rows


def parse_model(arg):
    """'label=path' (or just 'path': the label is the file's directory name) -> (label, path)"""
    if '=' in arg: label, path = arg.split('=', 1)
    else: path = arg; label = os.path.basename(os.path.dirname(os.path.abspath(arg))) or 'model'
    return label, path


def model_info(path):
    """(seed, extra) of a checkpoint: the training seed and what the trainer wrote beside the network (the lemma file and its digest, ...)"""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return int(ck['config'].get('seed', 0)), ck.get('extra', {}) or {}


def library_of(models, lemmas_arg):
    """the lemma file to evaluate under, from `--lemmas` ('' none, 'auto' read it from the checkpoints, else a path) and the models' own records: (path or '', {model path: block digest or None})"""
    digests = {}; files = set(); missing = []
    for label, path in models:
        _, extra = model_info(path)
        digests[path] = extra.get('block_digest')
        if extra.get('lemmas'):
            lf = os.path.join(os.path.dirname(os.path.abspath(path)), extra['lemmas'])
            if os.path.exists(lf): files.add(lf)
            else: missing.append((path, lf, extra.get('block_digest')))
    if lemmas_arg == 'auto':
        if missing: raise SystemExit('the checkpoint %s was trained with a library of bootstrapped lemmas (block %s) but its file %s is not there: restore it, or evaluate with --lemmas FILE' % (missing[0][0], missing[0][2], missing[0][1]))
        if len(files) > 1: raise SystemExit('the models were trained with different libraries (%s): evaluate them in separate runs' % sorted(files))
        lemmas_arg = next(iter(files), '')
        if lemmas_arg and any(d is None for d in digests.values()): raise SystemExit('some models were trained without the library and some with it: evaluate them in separate runs')
    return lemmas_arg, digests


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ workers
def make_worker_state(bk, ctx, eng, lib):
    def init():
        torch.set_num_threads(1)
        return {'bk': bk, 'ctx': ctx, 'eng': eng, 'lib': lib, 'nets': {}, 'solvers': {}}
    return init


def h_solve(st, p):
    """p: method, label, path, spec, seed (the replicate), probes, time_limit, tasks [(name, regime)] -> rows"""
    bk, lib = st['bk'], st['lib']
    key = (p['path'], p['spec'])
    solver = st['solvers'].get(key)
    if solver is None:
        name_, _ = AG.parse_spec(p['spec']); net = None
        if name_ in AG.NEEDS_NET:
            net = st['nets'].get(p['path'])
            if net is None: net = st['nets'][p['path']] = T.load(p['path'], st['ctx'].bags)[0]
        solver = st['solvers'][key] = AG.make_solver(p['spec'], st['eng'], net, st['ctx'] if net is not None else None)
    rows = []
    for n, regime in p['tasks']:
        pl = bk.placed[n]; prem = pl.premises(regime)
        if lib is not None: prem = lib.premises(prem)
        row = solver.solve(pl.stmt, prem, p['probes'], seed=p['seed'], name=n, regime=regime, time_limit=p['time_limit'])
        PR.annotate(row, bk, p['method'], p['seed']); row['model'] = p['label']; row['spec'] = solver.spec
        if lib is not None and row.get('proof'): row['boot_uses'] = sorted(lib.uses(row['proof']))
        rows.append(row)
    return rows


def evaluate(bk, ctx, eng, models, solvers, names, regimes, probes, time_limit, out_dir, lib=None, seed=None, workers=0, chunk=None, progress=print):
    """run `solvers` for each of `models` ([(label, path)]; empty: only the solvers that need no network) on `names`; rows go to out_dir/rows.jsonl as they are computed.  Returns all the rows (new and earlier)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'rows.jsonl')
    old = read_rows(path); done = {(r['method'], r['seed'], r['name'], r['regime']) for r in old}
    tasks = tasks_of(bk, names, regimes)
    runs = []                                                                     # (label, model path, replicate seed)
    for label, mpath in models:
        s, _ = model_info(mpath)
        runs.append((label, mpath, int(seed if seed is not None and len(models) == 1 else s)))
    if len({(l, s) for l, _, s in runs}) != len(runs): raise SystemExit('two models with the same label and the same seed: %s' % [(l, s) for l, _, s in runs])
    if not models: runs.append((None, None, 0))
    payloads = []
    for label, mpath, mseed in runs:
        for spec in solvers:
            name_, _ = AG.parse_spec(spec)
            if name_ in AG.NEEDS_NET and mpath is None: progress('skip %s: it needs a model' % spec); continue
            if name_ not in AG.NEEDS_NET and mpath is not None: continue            # (a model-free solver is run once, below, with no model)
            method = AG.canonical(spec) if name_ not in AG.NEEDS_NET else '%s+%s' % (label, AG.canonical(spec))
            sd = mseed if name_ in AG.NEEDS_NET else 0
            todo = [t for t in tasks if (method, sd, t[0], t[1]) not in done]
            ck = chunk or (1 if probes >= 500 else 4)
            for i in range(0, len(todo), ck):
                payloads.append({'method': method, 'label': label, 'path': mpath, 'spec': spec, 'seed': sd, 'probes': probes, 'time_limit': time_limit, 'tasks': todo[i:i + ck]})
    if models and any(AG.parse_spec(s)[0] not in AG.NEEDS_NET for s in solvers):          # model-free solvers: once
        for spec in solvers:
            name_, _ = AG.parse_spec(spec)
            if name_ in AG.NEEDS_NET: continue
            method = AG.canonical(spec); todo = [t for t in tasks if (method, 0, t[0], t[1]) not in done]
            ck = chunk or (1 if probes >= 500 else 4)
            for i in range(0, len(todo), ck):
                payloads.append({'method': method, 'label': None, 'path': None, 'spec': spec, 'seed': 0, 'probes': probes, 'time_limit': time_limit, 'tasks': todo[i:i + ck]})
    progress('%d tasks x %d runs: %d jobs to do (%d rows already there)' % (len(tasks), len(runs), len(payloads), len(old)))
    rows = list(old); t0 = time.time(); n_new = n_ok = 0
    pool = Pool(workers, make_worker_state(bk, ctx, eng, lib), {'solve': h_solve})
    try:
        with open(path, 'a') as logf:
            for _, part in pool.imap_unordered('solve', payloads):
                for row in part:
                    logf.write(json.dumps(row, default=str) + '\n'); rows.append(row); n_new += 1; n_ok += int(row['solved'])
                    if progress and (n_new % 25 == 0 or row['solved']):
                        progress('  %-34s %-26s %-8s %-6s probes %5d %6.1fs  %s' % (row['name'], row['method'], row['regime'], 'SOLVED' if row['solved'] else '-', row['probes'], row['seconds'], row.get('error', '') or ''))
                logf.flush()
    finally:
        pool.close()
    progress('%d new rows, %d solved, %.0f s' % (n_new, n_ok, time.time() - t0))
    return rows


def recheck_rows(bk, eng, rows, lib, out_path, progress=print):
    """replay the proof of every solved row in the engine and ask the independent checker (recheck.SecondOpinion) to accept the proof term for the statement.  Writes out_path, returns the verdicts."""
    import recheck as RC
    items = []
    for i, r in enumerate(rows):
        if not r.get('solved') or not r.get('proof') or r['name'] not in bk.placed: continue
        pl = bk.placed[r['name']]; prem = pl.premises(r['regime'])
        if lib is not None: prem = lib.premises(prem)
        try: term = ACTS.replay_portable(eng, pl.stmt, r['proof'], prem)
        except Exception as e: items.append((i, None, None, 'the proof does not replay: %s' % type(e).__name__)); continue
        items.append((i, term, pl.stmt, None))
    good = [t for _, t, s, w in items if t is not None]
    verdicts = []
    so = RC.SecondOpinion(bk.w.k, needed_by=[x for _, t, s, w in items if t is not None for x in (t, s)]) if good else None
    for i, term, stmt, why in items:
        if term is None: verdicts.append({'row': i, 'method': rows[i]['method'], 'seed': rows[i]['seed'], 'name': rows[i]['name'], 'ok': False, 'message': why}); continue
        ok, msg = so.accepts(term, stmt)
        verdicts.append({'row': i, 'method': rows[i]['method'], 'seed': rows[i]['seed'], 'name': rows[i]['name'], 'ok': bool(ok), 'message': '' if ok else str(msg)[:200]})
    n_ok = sum(1 for v in verdicts if v['ok'])
    progress('second opinion: %d of %d proofs accepted' % (n_ok, len(verdicts)))
    MT.write_json(out_path, {'checked': len(verdicts), 'accepted': n_ok, 'verdicts': verdicts})
    return verdicts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='dev', help="train, dev, test or all; several separated by commas"); ap.add_argument('--names', default='', help='evaluate these problems (comma separated) instead of a split')
    ap.add_argument('--regimes', default='prefix', help='prefix, strict or both (prefix,strict)')
    ap.add_argument('--solver', action='append', default=[], help='a solver spec (ml_agents): blind, uniform, sample, greedy, ida, mcts, mw, with options as name:key=value,...; repeatable')
    ap.add_argument('--model', action='append', default=[], help='LABEL=PATH of a checkpoint; repeatable')
    ap.add_argument('--probes', type=int, default=2000); ap.add_argument('--time-limit', type=float, default=120.0); ap.add_argument('--seed', type=int, default=None, help='the replicate of a single model (default: the seed in its checkpoint)')
    ap.add_argument('--recheck', action='store_true'); ap.add_argument('--threads', type=int, default=1); ap.add_argument('--workers', type=int, default=0, help='forked processes that share the world'); ap.add_argument('--limit', type=int, default=0, help='the first N problems only')
    ap.add_argument('--chunk', type=int, default=0, help='tasks per job (default: 1 for budgets of 500 probes or more)')
    W.add_args(ap)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    if not a.solver: ap.error('no --solver')
    models = [parse_model(m) for m in a.model]
    a.lemmas, digests = library_of(models, a.lemmas) if models else (a.lemmas, {})
    bk, ctx, eng = W.load(a)
    lib = W.load_library(a, bk, eng, ctx, a.data)
    for path, d in digests.items():
        if (d or None) != (lib.block_digest() if lib is not None else None): raise SystemExit('%s was trained with the library %s, this evaluation has %s' % (path, d, lib.block_digest() if lib is not None else None))
    if a.names: names = [n for n in a.names.split(',') if n in bk.placed]
    else:
        splits = a.split.split(',')
        names = [n for n in sorted(bk.placed) if 'all' in splits or BS.split_of(n) in splits]
    if a.limit: names = names[:a.limit]
    regimes = tuple(a.regimes.split(','))
    print('%d problems (%s), regimes %s, %d models, solvers %s, %d probes, %.0f s per problem' % (len(names), a.split, regimes, len(models), a.solver, a.probes, a.time_limit), flush=True)
    os.makedirs(a.out, exist_ok=True)
    meta = MT.run_meta(vars(a), {'world_globals': bk.w.k.num_globals(), 'problems': len(names), 'models': {l: {'path': p, 'digest': MT.file_digest(p)} for l, p in models},
                                 'dataset_manifest': MT.file_digest(os.path.join(a.data, 'manifest.json')), 'lemma_block': lib.block_digest() if lib is not None else None})
    MT.write_json(os.path.join(a.out, 'meta.json'), meta)
    rows = evaluate(bk, ctx, eng, models, a.solver, names, regimes, a.probes, a.time_limit, a.out, lib=lib, seed=a.seed, workers=a.workers, chunk=a.chunk or None)
    if a.recheck: recheck_rows(bk, eng, rows, lib, os.path.join(a.out, 'recheck.json'))
    print('rows %d; solved within budgets %s' % (len(rows), {b: n for b, n in PR.curve(rows).items() if b <= a.probes}))


if __name__ == '__main__':
    main()
