"""ml_synth.py -- the synthetic theorems as training data for the policy.

    recs = load_records('data/synth_v2', 'train', limit=20000)                      # [dict]: id, code (the statement in the typed builder's tokens), steps (portable actions), n_steps, ...
    steps = extract_steps(recs, bk, ctx, premises, workers=4)                        # [Step]: every action of every proof, decomposed into the decisions of the assembler (ml_data)
    steps = cached_steps('data/synth_v2', 'train', bk, ctx, limit=20000, workers=4)  # the same, kept in a pickle next to the dataset

A synthetic theorem is proved with the library only (its premises are the globals below the first problem of the bank, `bank.LIBRARY_LIMIT`), so a policy pre-trained on these steps has seen nothing of any
problem of the bank.  Extraction replays every proof in the engine (about 0.05 s a theorem) and is spread over `workers` forked processes (ml_pool) that share the world.
Memory: a Step is about 8 KB (the goal, the overlap with the globals, the options of its decisions): 60 000 theorems of 5 steps are about 2.5 GB.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, gzip, json, time, math, pickle, hashlib, collections
import numpy as np
import gym_env as G, gym_tree as GT
import ml_data as D
from ml_pool import Pool

SPLITS = ('train', 'dev', 'test_iid', 'test_comp')


def read_records(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def load_records(ddir, split, limit=None):
    """the records of a split of a merged synthetic dataset (`synth_cli.py merge` wrote DIR/<split>.jsonl.gz), in file order; at most `limit` (the first ones: the file order is the index order)"""
    if split not in SPLITS: raise ValueError('split: one of %s' % (SPLITS,))
    path = os.path.join(ddir, split + '.jsonl.gz')
    out = []
    for r in read_records(path):
        out.append(r)
        if limit is not None and len(out) >= limit: break
    return out


def _extract_one(ctx, eng, rec, premises, stats):
    stmt = GT.statement_from_code(eng, rec['code'], premises)
    return D.extract(ctx, eng, stmt, premises, rec['steps'], name=rec['id'], stats=stats)


def _handler(st, recs):
    stats = collections.Counter(); out = []
    for rec in recs:
        try: steps = _extract_one(st['ctx'], st['eng'], rec, st['premises'], stats)
        except Exception as e:
            stats['record_error ' + type(e).__name__] += 1; continue
        for s in steps: s.goal.key = None
        out.append(steps)
    return out, dict(stats)


def extract_steps(records, bk, ctx, premises, workers=0, chunk=200, progress=None):
    """[Step] for the proofs of the records (all of them, in order); a record whose proof does not replay is counted in the returned statistics and skipped.  Returns (steps, stats)."""
    def init():
        return {'ctx': ctx, 'eng': G.Engine(bk.w, kinds=set(G.Kind), strict=True), 'premises': premises}
    pool = Pool(workers, init, {'extract': _handler})
    parts = [records[i:i + chunk] for i in range(0, len(records), chunk)]
    t0 = time.time(); steps = []; stats = collections.Counter()
    try:
        for i in range(0, len(parts), max(1, workers) * 4):
            res = pool.map('extract', parts[i:i + max(1, workers) * 4])
            for out, st in res:
                for ss in out: steps.extend(ss)
                stats.update(st)
            if progress: progress(min(len(records), (i + max(1, workers) * 4) * chunk), len(records), time.time() - t0)
    finally:
        pool.close()
    return steps, stats


def cached_steps(ddir, split, bk, ctx, limit=None, workers=0, progress=print, cache_dir=None):
    """`extract_steps` for a split, cached in a pickle named by the dataset's manifest, the split, the limit and the step format"""
    man = open(os.path.join(ddir, 'manifest.json')).read()
    key = hashlib.sha256((man + '|%s|%s|v5|%d' % (split, limit, bk.w.k.num_globals())).encode()).hexdigest()[:12]
    cpath = os.path.join(cache_dir or ddir, 'steps_%s_%s.pkl' % (split, key))
    if os.path.exists(cpath):
        try:
            with open(cpath, 'rb') as f: return pickle.load(f)
        except Exception as e: progress('the cache %s is unreadable (%s): rebuilding' % (cpath, type(e).__name__))
    import bank as B
    recs = load_records(ddir, split, limit)
    premises = bk.w.gid[B.LIBRARY_LIMIT]
    steps, stats = extract_steps(recs, bk, ctx, premises, workers, progress=lambda n, m, t: progress('  %s: %d/%d theorems, %.0f s' % (split, n, m, t)))
    progress('%s: %d theorems -> %d steps (%s)' % (split, len(recs), len(steps), dict(stats)))
    tmp = cpath + '.tmp%d' % os.getpid()
    with open(tmp, 'wb') as f: pickle.dump(steps, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cpath)
    return steps
