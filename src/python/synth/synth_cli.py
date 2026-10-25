"""synth_cli.py -- generate, merge and verify the synthetic theorem dataset.

    python3 src/python/synth/synth_cli.py gen    --out DIR --seed 1 --attempts 160000 --shard 2000 --jobs 2     # shards of records: DIR/shards/shard_<first index>.jsonl.gz
    python3 src/python/synth/synth_cli.py merge  --out DIR [--train-target 60000] [--shape-cap 2] [--skeleton-cap 25] [--exclude-dataset data/dataset_v2]   # DIR/{train,dev,test_iid,test_comp}.jsonl.gz, manifest.json
    python3 src/python/synth/synth_cli.py grow   --out DIR --train-target 60000 --attempts 450000 [--max-attempts 1350000] --shard 2000 --jobs 4 --exclude-dataset data/dataset_v2
                                                                                                               # gen, then count what the merge would keep; more shards until the train split has the target
    python3 src/python/synth/synth_cli.py verify --out DIR [--sample 1] [--no-second-opinion]                       # replay every record from its data in a fresh process
    python3 src/python/synth/synth_cli.py card   --out DIR                                                          # DIR/DATASET_CARD.md from the manifest

`gen` is resumable (a shard that exists is not remade) and its result does not depend on --jobs or --shard: the index decides every theorem (synth_gen).

HOW MANY THEOREMS.  About one attempt in four becomes a record; the caps on shapes and skeletons, the exclusion of the benchmark and the duplicates remove some of those (a pilot of 30 000 attempts kept
6 024 of 7 065); and 12% of what is kept is held out (dev, test_iid, test_comp).  So "60 000 theorems to train on" is `--train-target 60000`, not `--target`, which counts the records of all splits.  How many
attempts that takes depends on the caps (the more records there are, the more of them are the n-th instance of a skeleton), so `grow` does not guess: it generates, counts what the merge would keep, and
generates more until the train split has the target; if `--max-attempts` is reached first it says so and exits with status 3.  Records are taken in index order, so more shards never change the dataset
that a target selects.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import argparse, glob, json, multiprocessing, os, sys, time


def _work(args):
    cfg_json, a, b, path = args
    import synth_gen as SG
    cfg = SG.Config.from_json(cfg_json)
    return SG.run_range(cfg, a, b, path)


def leak_digests(w, subproblems=None):
    """the digests of the statements the synthetic data must not contain: every theorem of the world (library lemmas and bank problems) and, if given, the sub-theorems of the bank (a dataset made in THIS world)"""
    import synth_check as SC
    k = w.k; out = set()
    for gid in range(k.num_globals()):
        _, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        if val is None or k.is_inductive(gid): continue
        ok, sort = k.infer_open(ty, [])
        if ok and sort == k.mk_type(0): out.add(SC.digest(w.show(w.T(ty))))
    if subproblems:
        data = json.load(open(subproblems))
        for e in data['examples']: out.add(e['statement_digest'])
    return out


def dataset_exclusions(dirs):
    """(digests, shapes, sources) of the statements a synthetic dataset must not contain, from the files of datasets made by `problems/dataset.py`: the digests of the bank problems, the lemmas of the
    library and the sub-theorems, and the SHAPES (numerals and variable names forgotten) of the bank problems"""
    import synth_check as SC, synth_merge as SM
    digests = set(); shapes = set(); sources = []
    for d in dirs:
        bank = json.load(open(os.path.join(d, 'bank.json')))
        for p in bank['problems']:
            digests.add(p['statement_digest']); shapes.add(SM.shape_of(SC.pretty(p['statement'])))
        n_lib = n_sub = 0
        path = os.path.join(d, 'library.json')
        if os.path.exists(path):
            for l in json.load(open(path))['lemmas']: digests.add(l['statement_digest']); n_lib += 1
        path = os.path.join(d, 'subproblems.json')
        if os.path.exists(path):
            for e in json.load(open(path))['examples']: digests.add(e['statement_digest']); n_sub += 1
        sources.append({'dataset': os.path.basename(os.path.normpath(d)), 'world_digest': bank.get('world_digest'), 'problems': len(bank['problems']), 'lemmas': n_lib, 'sub_theorems': n_sub})
    return digests, shapes, sources


def make_shards(cfg, out, start, attempts, shard, jobs):
    """generate the shards of the indices start..start+attempts-1 into out/shards (the ones that exist are not remade); returns (shards made or found, records in them)"""
    shards = os.path.join(out, 'shards'); os.makedirs(shards, exist_ok=True)
    with open(os.path.join(out, 'config.json'), 'w') as f: json.dump(cfg.to_json(), f, indent=1)
    tasks = [(cfg.to_json(), s, min(s + shard, start + attempts), os.path.join(shards, 'shard_%09d.jsonl.gz' % s)) for s in range(start, start + attempts, shard)]
    t0 = time.time(); done = 0; recs = 0
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(jobs, maxtasksperchild=1) as pool:
        for meta in pool.imap_unordered(_work, tasks):
            done += 1; recs += meta['records']
            print('[%5.0f s] shard %d/%d  range %s  %d records  %.0f s  errors %d  disagreements %d' % (time.time() - t0, done, len(tasks), meta['range'], meta['records'], meta['seconds'],
                                                                                              meta['n_errors'], len(meta['disagreements'])), flush=True)
    return done, recs


def _config(a):
    import synth_gen as SG
    return SG.Config(seed=a.seed, closer_keep=a.closer_keep, area_weights=json.loads(a.area_weights) if a.area_weights else None)


def cmd_gen(a):
    done, recs = make_shards(_config(a), a.out, a.start, a.attempts, a.shard, a.jobs)
    print('done: %d shards, %d records' % (done, recs))


def _exclusions(w, a):
    """(digests, shapes, sources) of what the dataset must not contain: the theorems of the world, the sub-theorems of `--subproblems`, and the datasets of `--exclude-dataset`"""
    exclude = leak_digests(w, a.subproblems); shapes = set(); sources = []
    if a.exclude_dataset:
        d2, shapes, sources = dataset_exclusions(a.exclude_dataset); exclude |= d2
    else: print('WARNING: no --exclude-dataset: the bank problems are not excluded from the synthetic data (a dataset for the experiments must be merged with one)')
    return exclude, shapes, sources


def _round_up(n, k): return -(-int(n) // k) * k


def cmd_grow(a):
    """generate until the merge would keep `--train-target` train theorems (the same merge, run on what exists, without writing); exit 3 when `--max-attempts` is reached first"""
    import imo_lib, synth_gen as SG, synth_merge as SM
    cfg = _config(a)
    w = imo_lib.build_imo()
    exclude, shapes, sources = _exclusions(w, a)
    attempts = _round_up(a.attempts, a.shard); cap = max(attempts, _round_up(a.max_attempts or 3 * a.attempts, a.shard))
    rnd = 0
    while True:
        rnd += 1
        print('round %d: %d attempts' % (rnd, attempts), flush=True)
        make_shards(cfg, a.out, 0, attempts, a.shard, a.jobs)
        shards = sorted(glob.glob(os.path.join(a.out, 'shards', 'shard_*.jsonl.gz')))
        plan = SM.merge(shards, a.out, cfg, exclude=exclude, exclude_shapes=shapes, shape_cap=a.shape_cap, skeleton_cap=a.skeleton_cap, train_target=a.train_target, dry_run=True)
        n = plan['counts']['train']
        print('round %d: %d attempts -> %d records kept in all, %d to train on (target %d); dropped %s' % (rnd, attempts, plan['records'], n, a.train_target, plan['dropped_in_merge']), flush=True)
        if n >= a.train_target:
            print('done: the target is met after %d attempts (%d shards)' % (attempts, len(shards))); return
        if attempts >= cap:
            print('SHORT: %d attempts (--max-attempts) give %d of the %d train theorems wanted; raise --max-attempts, or relax the caps' % (attempts, n, a.train_target), file=sys.stderr)
            sys.exit(3)
        want = a.train_target / max(1, n)                                           # (how many times more are needed, if the yield stays; the caps make it fall, hence the margin)
        attempts = min(cap, _round_up(attempts * max(1.1, want * 1.05), a.shard))


def cmd_merge(a):
    import imo_lib, bank as B
    import synth_gen as SG, synth_merge as SM
    cfg = SG.Config.from_json(json.load(open(os.path.join(a.out, 'config.json'))))
    w = imo_lib.build_imo()
    shards = sorted(glob.glob(os.path.join(a.out, 'shards', 'shard_*.jsonl.gz')))
    if not shards: sys.exit('no shards in %s' % a.out)
    gens = {json.load(open(p + '.meta.json'))['generator'] for p in shards}
    if gens != {SG.generator_digest()}: print('WARNING: the shards were made by %s, the generator now is %s' % (sorted(gens), SG.generator_digest()))
    exclude, shapes, sources = _exclusions(w, a)
    man = SM.merge(shards, a.out, cfg, exclude=exclude, world_digest=B.world_digest(w, w.gid[cfg.limit_name]), generator=sorted(gens)[0] if len(gens) == 1 else sorted(gens), shape_cap=a.shape_cap, skeleton_cap=a.skeleton_cap,
                   target=a.target, train_target=a.train_target, progress=lambda p, kept: print('\r%s: %d kept' % (os.path.basename(p), kept), end='', flush=True), exclude_shapes=shapes, sources=sources)
    print('\n%d records %s; dropped %s' % (man['records'], man['counts'], man['dropped_in_merge']))
    if a.train_target and man['counts']['train'] < a.train_target:
        print('SHORT: %d train theorems, %d wanted (more attempts: `grow`)' % (man['counts']['train'], a.train_target), file=sys.stderr)
        if not a.allow_short: sys.exit(3)


def cmd_verify(a):
    import synth_verify as SV
    paths = [os.path.join(a.out, s + '.jsonl.gz') for s in ('train', 'dev', 'test_iid', 'test_comp')]
    cfg = json.load(open(os.path.join(a.out, 'config.json')))
    rep = SV.verify_files(paths, limit_name=cfg['limit_name'], second_opinion=not a.no_second_opinion, sample=a.sample, batch=a.batch,
                          progress=lambda n, bad, t: print('\r%d records, %d bad, %.0f s' % (n, bad, t), end='', flush=True))
    print()
    print(json.dumps({k: v for k, v in rep.items() if k not in ('failed', 'second_opinion_refused') or v}, indent=1))
    with open(os.path.join(a.out, 'verification.json'), 'w') as f: json.dump(rep, f, indent=1)
    sys.exit(0 if rep['ok'] else 1)


def cmd_card(a):
    import synth_card as SK
    man = json.load(open(os.path.join(a.out, 'manifest.json')))
    ver = json.load(open(os.path.join(a.out, 'verification.json'))) if os.path.exists(os.path.join(a.out, 'verification.json')) else None
    with open(os.path.join(a.out, 'DATASET_CARD.md'), 'w') as f: f.write(SK.card(man, ver))
    print('wrote', os.path.join(a.out, 'DATASET_CARD.md'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen'); g.add_argument('--out', required=True); g.add_argument('--seed', type=int, default=1); g.add_argument('--start', type=int, default=0)
    g.add_argument('--attempts', type=int, required=True); g.add_argument('--shard', type=int, default=2000); g.add_argument('--jobs', type=int, default=2)
    g.add_argument('--closer-keep', type=float, default=0.2); g.add_argument('--area-weights', default=None, help='JSON {area: weight}'); g.set_defaults(fn=cmd_gen)
    m = sub.add_parser('merge'); m.add_argument('--out', required=True); m.add_argument('--shape-cap', type=int, default=2); m.add_argument('--skeleton-cap', type=int, default=25)
    m.add_argument('--target', type=int, default=None, help='keep this many records in all (every split)'); m.add_argument('--train-target', type=int, default=None, help='keep this many TRAIN records (the held-out records come on top); exit 3 if there are fewer')
    m.add_argument('--allow-short', action='store_true', help='with --train-target: do not fail when there are fewer train records')
    m.add_argument('--subproblems', default=None); m.add_argument('--exclude-dataset', action='append', default=[], help='a dataset directory (problems/dataset.py): its bank problems (exactly and by shape), library lemmas and sub-theorems are excluded; repeatable')
    m.set_defaults(fn=cmd_merge)
    r = sub.add_parser('grow', help='gen + count until the merge keeps --train-target train theorems'); r.add_argument('--out', required=True); r.add_argument('--seed', type=int, default=1)
    r.add_argument('--train-target', type=int, required=True); r.add_argument('--attempts', type=int, required=True, help='the first round'); r.add_argument('--max-attempts', type=int, default=None, help='give up (exit 3) here; default 3 x --attempts')
    r.add_argument('--shard', type=int, default=2000); r.add_argument('--jobs', type=int, default=2); r.add_argument('--closer-keep', type=float, default=0.2); r.add_argument('--area-weights', default=None, help='JSON {area: weight}')
    r.add_argument('--shape-cap', type=int, default=2); r.add_argument('--skeleton-cap', type=int, default=25); r.add_argument('--subproblems', default=None)
    r.add_argument('--exclude-dataset', action='append', default=[], help='as for merge: the SAME exclusions, or the count is not the merge\'s'); r.set_defaults(fn=cmd_grow)
    v = sub.add_parser('verify'); v.add_argument('--out', required=True); v.add_argument('--sample', type=int, default=None); v.add_argument('--no-second-opinion', action='store_true')
    v.add_argument('--batch', type=int, default=500); v.set_defaults(fn=cmd_verify)
    c = sub.add_parser('card'); c.add_argument('--out', required=True); c.set_defaults(fn=cmd_card)
    a = ap.parse_args(argv); a.fn(a)


if __name__ == '__main__':
    main()
