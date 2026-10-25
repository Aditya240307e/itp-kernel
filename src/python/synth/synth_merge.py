"""synth_merge.py -- from shards of records to a dataset: leakage exclusion, deduplication, a cap per shape, splits, statistics.

    stats = merge(shards, out_dir, cfg, exclude=set_of_digests, world_digest=..., target=60000)

WHAT HAPPENS TO THE RECORDS (in the order of their indices, so that the result does not depend on how the index space was cut into shards):
  1. EXCLUSION  a theorem whose statement is the statement of a lemma of the library, a problem of the bank or a sub-theorem of one is dropped (`exclude`: statement digests), and so is one that has
     the SHAPE of a problem of the bank (`exclude_shapes`: the statement with its numerals and variable names forgotten -- `x * 3 <= y` for `x * 5 <= y`): a learner trained on the data must not be
     handed a problem it is tested on, nor a variation of it, nor a lemma it will be asked to reprove.
  2. DUPLICATES the same statement (same digest) as an earlier record.
  3. SHAPE CAP  a SHAPE is a statement with its numerals and the names of its variables forgotten; at most `shape_cap` records have the same shape (`x * 3 <= y` and `x * 5 <= y` are the same exercise).
  3b. SKELETON CAP  a SKELETON is a proof with its terms forgotten: the sequence of its steps (kind and lemma). The lemmas of some areas combine in few ways, and the same two-step proof of
     `is_gcd x y 1 -> coprime y x` with a thousand different terms is one exercise a thousand times: at most `skeleton_cap` records have the same skeleton.
  4. SPLITS     a pure function of the record, not of its place in the file:
        test_comp   the record's RECIPE (the set of lemmas and decision procedures its proof uses) hashes into the held-out part: no training record has this recipe. A learner is asked to put known
                    lemmas together in a way it has not seen.
        test_iid / dev / train   by the statement's digest.
     (The salt is part of the configuration: another salt, another split.)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import collections, gzip, hashlib, json, os, re, time

SPLITS = ('train', 'dev', 'test_iid', 'test_comp')
_PREFIX = re.compile(r'^(forall \([^)]*\), )+')


def shape_of(pretty):
    """a statement with its numerals and variable names forgotten (variables numbered by first occurrence in the body)"""
    body = _PREFIX.sub('', pretty)
    body = re.sub(r'\b\d+\b', '#', body)
    names = {}
    def f(m):
        return 'v%d' % names.setdefault(m.group(0), len(names))
    return hashlib.sha256(re.sub(r'\bx\d+\b', f, body).encode()).hexdigest()[:16]


def _bucket(salt, kind, key, mod=1000):
    return int(hashlib.sha256(('%s|%s|%s' % (salt, kind, key)).encode()).hexdigest()[:8], 16) % mod


def skeleton_of(rec):
    """a proof with its terms forgotten: the kinds of its steps and the lemma each applies (global ids: within one dataset they name the lemmas), as a short hash"""
    parts = []
    for st in rec['steps']:
        kind = st.get('kind'); lemma = None
        if kind == 'APPLY' and isinstance(st.get('source'), list): lemma = st['source'][1]
        elif kind == 'APPLY_WITH' and st.get('codes') and st['codes'][0] and st['codes'][0][0][0] == 'c': lemma = st['codes'][0][0][1]
        parts.append('%s:%s' % (kind, lemma))
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]


def split_of(rec, salt='split-v1', permille=(60, 30, 30)):
    """the split of a record (see the module documentation): `permille` = (test_comp, test_iid, dev)"""
    comp, iid, dev = permille
    if _bucket(salt, 'recipe', rec['recipe']) < comp: return 'test_comp'
    b = _bucket(salt, 'stmt', rec['digest'])
    if b < iid: return 'test_iid'
    if b < iid + dev: return 'dev'
    return 'train'


def read_records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f: yield json.loads(line)


def merge(shards, out_dir, cfg, exclude=(), world_digest=None, generator=None, shape_cap=2, skeleton_cap=25, salt='split-v1', permille=(60, 30, 30), target=None, progress=None, exclude_shapes=(), sources=None,
          train_target=None, dry_run=False):
    """merge shard files (paths of .jsonl.gz written by synth_gen.run_range, each with its .meta.json) into the dataset in `out_dir`; returns the manifest (also written there).

    `target`: stop after that many records in all (every split); `train_target`: stop after that many TRAIN records -- what an experiment means by "60 000 synthetic theorems" (the held-out 12% of the records
    come on top of them).  Records are taken in index order, so a dataset made with a target is the prefix of the dataset made without one: more shards never change it.
    `dry_run`: count only -- nothing is written, the result is {'counts', 'records', 'dropped_in_merge'}."""
    if not dry_run: os.makedirs(out_dir, exist_ok=True)
    exclude = set(exclude); exclude_shapes = set(exclude_shapes); seen = set(); shapes = collections.Counter(); skeletons = collections.Counter(); dropped = collections.Counter()
    metas = [json.load(open(p + '.meta.json')) for p in shards]
    if world_digest is not None:                                             # (the shards of one dataset are made in one library: see synth_gen.run_range)
        other = sorted({str(m.get('world')) for m in metas} - {world_digest})
        if other: raise RuntimeError('the shards were made in the library %s (None: not recorded), the library now is %s -- the data of two libraries are not mixed: generate again, in another --out' % (other, world_digest))
    order = sorted(range(len(shards)), key=lambda i: metas[i]['range'][0])
    outs = {s: [] for s in SPLITS}; kept = 0
    for i in order:
        for rec in read_records(shards[i]):
            if target is not None and kept >= target: dropped['beyond the target count'] += 1; continue
            if train_target is not None and len(outs['train']) >= train_target: dropped['beyond the target count'] += 1; continue
            if rec['digest'] in exclude: dropped['statement of a library lemma, bank problem or sub-theorem'] += 1; continue
            if rec['digest'] in seen: dropped['duplicate statement'] += 1; continue
            sh = shape_of(rec['pretty'])
            if sh in exclude_shapes: dropped['shape of a bank problem'] += 1; continue
            if shapes[sh] >= shape_cap: dropped['shape cap'] += 1; continue
            sk = skeleton_of(rec)
            if skeletons[sk] >= skeleton_cap: dropped['skeleton cap'] += 1; continue
            seen.add(rec['digest']); shapes[sh] += 1; skeletons[sk] += 1
            rec['shape'] = sh; rec['skeleton'] = sk; rec['split'] = split_of(rec, salt, permille)
            outs[rec['split']].append(rec); kept += 1
        if progress: progress(shards[i], kept)
    if dry_run: return {'counts': {s: len(outs[s]) for s in SPLITS}, 'records': kept, 'dropped_in_merge': dict(dropped)}
    for s in SPLITS:
        with gzip.open(os.path.join(out_dir, s + '.jsonl.gz.tmp'), 'wt', encoding='utf-8') as f:
            for r in outs[s]: f.write(json.dumps(r, separators=(',', ':')) + '\n')
        os.replace(os.path.join(out_dir, s + '.jsonl.gz.tmp'), os.path.join(out_dir, s + '.jsonl.gz'))
    allrecs = [r for s in SPLITS for r in outs[s]]
    man = {'format': 1, 'name': 'synthetic', 'config': cfg.to_json(), 'config_digest': cfg.digest(), 'generator': generator, 'world_digest': world_digest, 'limit': cfg.limit_name,
           'split_rule': {'salt': salt, 'permille': {'test_comp': permille[0], 'test_iid': permille[1], 'dev': permille[2]}, 'shape_cap': shape_cap, 'skeleton_cap': skeleton_cap},
           'targets': {'records': target, 'train': train_target},
           'exclusions': {'digests': len(exclude), 'bank_shapes': len(exclude_shapes), 'sources': sources or []},
           'shards': len(shards), 'attempts': sum(m['range'][1] - m['range'][0] for m in metas), 'records_before_merge': sum(m['records'] for m in metas),
           'rejects': dict(sum((collections.Counter(m['rejects']) for m in metas), collections.Counter())), 'errors': sum(m['n_errors'] for m in metas),
           'disagreements': sum(len(m['disagreements']) for m in metas), 'dropped_in_merge': dict(dropped), 'records': len(allrecs),
           'counts': {s: len(outs[s]) for s in SPLITS}, 'generation_seconds': round(sum(m['seconds'] for m in metas)),
           'statistics': statistics(allrecs), 'statistics_by_split': {s: statistics(outs[s]) for s in SPLITS}}
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f: json.dump(man, f, indent=1, sort_keys=True)
    return man


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def statistics(recs):
    """what a set of records is made of: distributions of the labels"""
    if not recs: return {}
    n = len(recs); c = collections.Counter
    lem = c(l for r in recs for l in set(r['lemmas']))
    return {'n': n,
            'family': dict(c(r['family'] for r in recs)), 'area': dict(c(r['area'] for r in recs)),
            'depth': {str(k): v for k, v in sorted(c(r['depth'] for r in recs).items())},
            'n_hyps': {str(k): v for k, v in sorted(c(r['n_hyps'] for r in recs).items())},
            'n_vars': {str(k): v for k, v in sorted(c(r['n_vars'] for r in recs).items())},
            'n_steps': {str(k): v for k, v in sorted(c(r['n_steps'] for r in recs).items())},
            'size': {'mean': round(sum(r['size'] for r in recs) / n, 1), 'p50': _q([r['size'] for r in recs], 0.5), 'p90': _q([r['size'] for r in recs], 0.9), 'max': max(r['size'] for r in recs)},
            'closed_at_once_by_a_decision_procedure': round(sum(1 for r in recs if r['solved_by']) / n, 4),
            'uses_a_decision_procedure_in_the_proof': round(sum(1 for r in recs if r['closers']) / n, 4),
            'existential_conclusion': round(sum(1 for r in recs if r['semantic']['exact'] is False) / n, 4),
            'lemmas_used': len(lem), 'lemma_usage': dict(lem.most_common()), 'distinct_recipes': len({r['recipe'] for r in recs}), 'distinct_shapes': len({r.get('shape') for r in recs}),
            'distinct_skeletons': len({r.get('skeleton') for r in recs}),
            'root_lemmas': len({r['root_lemma'] for r in recs})}
