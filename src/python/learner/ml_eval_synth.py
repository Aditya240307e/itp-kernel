"""ml_eval_synth.py -- provers on held-out SYNTHETIC theorems: does the policy beat the model-free provers where n is in the hundreds?

    python3 src/python/learner/ml_eval_synth.py data/dataset_v2 runs/small/synth --out runs/small/eval/synth --split test_iid,test_comp --limit 120 \
        --solver blind --solver uniform --solver sample --model pre_n5000=runs/small/pre/n5000/s0/best.pt --probes 200 --time-limit 20 --workers 2

The olympiad dev and test splits have a few dozen problems each, so a comparison of provers on them cannot be told from noise unless the gap is large.  The synthetic splits `dev`, `test_iid` and `test_comp`
(`synth_cli.py merge`) hold hundreds to thousands of theorems that no training run has seen, each proved by a kernel-checked reference proof, so the question "does what the network learned help the search?"
can be answered with an interval that is narrow.  This tool runs the same solvers as `ml_evaluate.py` (`ml_agents`: blind, uniform, sample, greedy, ida, mcts, mw) on them and writes rows in the same form
(`rows.jsonl`: name, regime = 'library', split, depth, n_steps, method, model, seed, solved, probes, seconds, ...), so that `ml_report.py`-style tables and the intervals of `exp_stats.py` read them.

A synthetic theorem is stated and proved with the library only (its premises are the globals below the first problem of the bank, as in `ml_synth`): the world is the bank's, the statement is built from the
record's code.  A record whose statement cannot be built is a row with an `error` and counts as unsolved.  Rows are appended as they are computed; a second call with the same arguments continues.
Nothing here touches the olympiad test split: this is a development tool for the learner, and the held-out synthetic splits it reads are the generator's own.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import argparse, json, os, time
import torch
import gym_tree as GT
import ml_agents as AG, ml_protocol as PR, ml_synth as SY, ml_world as W, ml_meta as MT
import ml_train as T
import ml_evaluate as EV
from ml_pool import Pool

KEEP = ('depth', 'n_steps', 'n_lemmas', 'n_hyps', 'n_vars', 'area', 'root_lemma', 'family', 'recipe', 'split')


def tasks_of(ddir, splits, limit):
    """[record] of the splits, `limit` of each (the first ones: the file order is the index order), each with its split name"""
    out = []
    for sp in splits:
        recs = SY.load_records(ddir, sp, limit or None)
        for r in recs: r['split'] = sp
        out.extend(recs)
    return out


def make_state(bk, ctx, eng, premises):
    def init():
        torch.set_num_threads(1)
        return {'bk': bk, 'ctx': ctx, 'eng': eng, 'premises': premises, 'nets': {}, 'solvers': {}}
    return init


def h_solve(st, p):
    """p: method, label, path, spec, seed, probes, time_limit, recs [record] -> rows"""
    key = (p['path'], p['spec'])
    solver = st['solvers'].get(key)
    if solver is None:
        name_, _ = AG.parse_spec(p['spec']); net = None
        if name_ in AG.NEEDS_NET:
            net = st['nets'].get(p['path'])
            if net is None: net = st['nets'][p['path']] = T.load(p['path'], st['ctx'].bags)[0]
        solver = st['solvers'][key] = AG.make_solver(p['spec'], st['eng'], net, st['ctx'] if net is not None else None)
    rows = []
    for rec in p['recs']:
        try:
            stmt = GT.statement_from_code(st['eng'], rec['code'], st['premises'])
            row = solver.solve(stmt, st['premises'], p['probes'], seed=p['seed'], name=rec['id'], regime='library', time_limit=p['time_limit'])
        except Exception as e:                                                                  # (a statement that cannot be built: unsolved, and the row says why)
            row = {'solved': False, 'probes': 0, 'decisions': 0, 'seconds': 0.0, 'steps': None, 'name': rec['id'], 'regime': 'library', 'solver': p['spec'],
                   'error': '%s: %s' % (type(e).__name__, str(e)[:160])}
        for k in KEEP:
            if k in rec: row[k] = rec[k]
        row['level'] = rec.get('depth'); row['method'] = p['method']; row['model'] = p['label']; row['spec'] = solver.spec; row['seed'] = p['seed']; row.pop('proof', None)
        rows.append(row)
    return rows


def evaluate(bk, ctx, eng, models, solvers, recs, probes, time_limit, out_dir, workers=0, chunk=4, seed=None, progress=print):
    """rows of `solvers` (and of `models` for those that need a network) on `recs`; they go to out_dir/rows.jsonl as they are computed; the rows of an earlier call are kept and not computed again"""
    import bank as B
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'rows.jsonl')
    old = EV.read_rows(path); done = {(r['method'], r['seed'], r['name']) for r in old}
    premises = bk.w.gid[B.LIBRARY_LIMIT]
    runs = []
    for label, mpath in models:
        s, _ = EV.model_info(mpath)
        runs.append((label, mpath, int(seed if seed is not None and len(models) == 1 else s)))
    if len({(l, s) for l, _, s in runs}) != len(runs): raise SystemExit('two models with the same label and the same seed: %s' % [(l, s) for l, _, s in runs])
    payloads = []
    def add(method, label, mpath, spec, sd):
        todo = [r for r in recs if (method, sd, r['id']) not in done]
        for i in range(0, len(todo), chunk):
            payloads.append({'method': method, 'label': label, 'path': mpath, 'spec': spec, 'seed': sd, 'probes': probes, 'time_limit': time_limit, 'recs': todo[i:i + chunk]})
    for spec in solvers:                                                                         # (a solver without a network: once)
        if AG.parse_spec(spec)[0] not in AG.NEEDS_NET: add(AG.canonical(spec), None, None, spec, 0)
    for label, mpath, mseed in runs:
        for spec in solvers:
            if AG.parse_spec(spec)[0] in AG.NEEDS_NET: add('%s+%s' % (label, AG.canonical(spec)), label, mpath, spec, mseed)
    progress('%d theorems: %d jobs to do (%d rows already there)' % (len(recs), len(payloads), len(old)))
    rows = list(old); t0 = time.time(); n_new = n_ok = 0
    pool = Pool(workers, make_state(bk, ctx, eng, premises), {'solve': h_solve})
    try:
        with open(path, 'a') as logf:
            for _, part in pool.imap_unordered('solve', payloads):
                for row in part:
                    logf.write(json.dumps(row, default=str) + '\n'); rows.append(row); n_new += 1; n_ok += int(row['solved'])
                    if progress and n_new % 50 == 0: progress('  %d rows, %d solved, %.0f s' % (n_new, n_ok, time.time() - t0))
                logf.flush()
    finally:
        pool.close()
    progress('%d new rows, %d solved, %.0f s' % (n_new, n_ok, time.time() - t0))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('synth', help='the merged synthetic dataset (a folder with train/dev/test_iid/test_comp .jsonl.gz)'); ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test_iid,test_comp', help='dev, test_iid, test_comp (comma separated); train is refused: it is what the networks were trained on')
    ap.add_argument('--limit', type=int, default=0, help='at most this many theorems of each split (the first ones)')
    ap.add_argument('--solver', action='append', default=[], help='a solver spec (ml_agents); repeatable'); ap.add_argument('--model', action='append', default=[], help='LABEL=PATH of a checkpoint; repeatable')
    ap.add_argument('--probes', type=int, default=200); ap.add_argument('--time-limit', type=float, default=20.0); ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--threads', type=int, default=1); ap.add_argument('--workers', type=int, default=0, help='forked processes that share the world'); ap.add_argument('--chunk', type=int, default=4, help='theorems per job')
    ap.add_argument('--only', default='', help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    if not a.solver: ap.error('no --solver')
    splits = [s for s in a.split.split(',') if s]
    if 'train' in splits or not set(splits) <= {'dev', 'test_iid', 'test_comp'}: ap.error('--split: dev, test_iid, test_comp (the networks were trained on train)')
    models = [EV.parse_model(m) for m in a.model]
    a.lemmas = ''
    bk, ctx, eng = W.load(a)
    recs = tasks_of(a.synth, splits, a.limit)
    print('%d theorems (%s), %d models, solvers %s, %d probes, %.0f s per theorem' % (len(recs), a.split, len(models), a.solver, a.probes, a.time_limit), flush=True)
    os.makedirs(a.out, exist_ok=True)
    meta = MT.run_meta(vars(a), {'world_globals': bk.w.k.num_globals(), 'theorems': len(recs), 'models': {l: {'path': p, 'digest': MT.file_digest(p)} for l, p in models},
                                 'dataset_manifest': MT.file_digest(os.path.join(a.data, 'manifest.json')), 'synth_manifest': MT.file_digest(os.path.join(a.synth, 'manifest.json'))})
    MT.write_json(os.path.join(a.out, 'meta.json'), meta)
    rows = evaluate(bk, ctx, eng, models, a.solver, recs, a.probes, a.time_limit, a.out, workers=a.workers, chunk=a.chunk, seed=a.seed)
    by = {}
    for r in rows: by.setdefault(r['method'], []).append(r)
    print('rows %d' % len(rows))
    for m in sorted(by): print('  %-28s %4d theorems, solved within %s' % (m, len(by[m]), {b: n for b, n in PR.curve(by[m], (10, 20, 50, 100, 200, 500, 1000)).items() if b <= a.probes}))


if __name__ == '__main__':
    main()
