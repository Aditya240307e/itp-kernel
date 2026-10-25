"""synth_cli.py -- generate, merge and verify the synthetic theorem dataset.

    python3 src/python/synth/synth_cli.py gen    --out DIR --seed 1 --attempts 160000 --shard 2000 --jobs 2     # shards of records: DIR/shards/shard_<first index>.jsonl.gz
    python3 src/python/synth/synth_cli.py merge  --out DIR [--target 60000] [--shape-cap 2] [--skeleton-cap 25] [--subproblems data/dataset_v2/subproblems.json]   # DIR/{train,dev,test_iid,test_comp}.jsonl.gz, manifest.json
    python3 src/python/synth/synth_cli.py verify --out DIR [--sample 1] [--no-second-opinion]                       # replay every record from its data in a fresh process
    python3 src/python/synth/synth_cli.py card   --out DIR                                                          # DIR/DATASET_CARD.md from the manifest

`gen` is resumable (a shard that exists is not remade) and its result does not depend on --jobs or --shard: the index decides every theorem (synth_gen).
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


def cmd_gen(a):
    import synth_gen as SG
    cfg = SG.Config(seed=a.seed, closer_keep=a.closer_keep, area_weights=json.loads(a.area_weights) if a.area_weights else None)
    shards = os.path.join(a.out, 'shards'); os.makedirs(shards, exist_ok=True)
    with open(os.path.join(a.out, 'config.json'), 'w') as f: json.dump(cfg.to_json(), f, indent=1)
    tasks = [(cfg.to_json(), s, min(s + a.shard, a.start + a.attempts), os.path.join(shards, 'shard_%09d.jsonl.gz' % s)) for s in range(a.start, a.start + a.attempts, a.shard)]
    t0 = time.time(); done = 0; recs = 0
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(a.jobs, maxtasksperchild=1) as pool:
        for meta in pool.imap_unordered(_work, tasks):
            done += 1; recs += meta['records']
            print('[%5.0f s] shard %d/%d  range %s  %d records  %.0f s  errors %d  disagreements %d' % (time.time() - t0, done, len(tasks), meta['range'], meta['records'], meta['seconds'],
                                                                                              meta['n_errors'], len(meta['disagreements'])), flush=True)
    print('done: %d shards, %d records' % (done, recs))


def cmd_merge(a):
    import imo_lib, bank as B
    import synth_gen as SG, synth_merge as SM
    cfg = SG.Config.from_json(json.load(open(os.path.join(a.out, 'config.json'))))
    w = imo_lib.build_imo()
    shards = sorted(glob.glob(os.path.join(a.out, 'shards', 'shard_*.jsonl.gz')))
    if not shards: sys.exit('no shards in %s' % a.out)
    gens = {json.load(open(p + '.meta.json'))['generator'] for p in shards}
    if gens != {SG.generator_digest()}: print('WARNING: the shards were made by %s, the generator now is %s' % (sorted(gens), SG.generator_digest()))
    exclude = leak_digests(w, a.subproblems)
    man = SM.merge(shards, a.out, cfg, exclude=exclude, world_digest=B.world_digest(w, w.gid[cfg.limit_name]), generator=sorted(gens)[0] if len(gens) == 1 else sorted(gens), shape_cap=a.shape_cap, skeleton_cap=a.skeleton_cap,
                   target=a.target, progress=lambda p, kept: print('\r%s: %d kept' % (os.path.basename(p), kept), end='', flush=True))
    print('\n%d records %s; dropped %s' % (man['records'], man['counts'], man['dropped_in_merge']))


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
    m = sub.add_parser('merge'); m.add_argument('--out', required=True); m.add_argument('--target', type=int, default=None); m.add_argument('--shape-cap', type=int, default=2); m.add_argument('--skeleton-cap', type=int, default=25)
    m.add_argument('--subproblems', default=None); m.set_defaults(fn=cmd_merge)
    v = sub.add_parser('verify'); v.add_argument('--out', required=True); v.add_argument('--sample', type=int, default=None); v.add_argument('--no-second-opinion', action='store_true')
    v.add_argument('--batch', type=int, default=500); v.set_defaults(fn=cmd_verify)
    c = sub.add_parser('card'); c.add_argument('--out', required=True); c.set_defaults(fn=cmd_card)
    a = ap.parse_args(argv); a.fn(a)


if __name__ == '__main__':
    main()
