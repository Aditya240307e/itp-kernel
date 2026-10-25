"""synth_stats.py -- how saturated the synthetic generator is, and what its proofs are made of.

    python3 synth_stats.py DIR [--out DIR/stats.json]          # DIR: a merged synthetic dataset (synth_cli.py merge)

The manifest already counts areas, depths, lengths and lemma usage.  This adds what a reader of the paper asks next -- "is it just the same theorem again and again?":

  classes     for the skeleton (the proof without its terms: the kinds and lemmas of its steps), the shape (the statement without numerals and names) and the recipe (the set of lemmas and closers of the
              proof): the number observed, the singletons and doubletons, the Good-Turing coverage 1 - f1/n (the chance that the next theorem is of a class not seen yet is about f1/n), and Chao1's estimate
              of how many classes the generator could make (a lower bound, with its standard error)
  rarefaction the number of distinct classes (and distinct lemmas) among the first m theorems of a random order, for m up to n: a curve that is still rising steeply at n says the generator is far from
              exhausted; one that flattens says more theorems add repetitions
  kinds       the actions the proofs use (a step of each kind, and the share of theorems that use it): the generator is a backward-chaining generator and this says so
  lemmas      how many library lemmas the proofs apply, how evenly (entropy)

The records of train, dev and the two test sets are all pooled (they are one population split by recipe); `train` alone is also summarised.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, gzip, math, argparse, collections
import exp_stats as ST

SPLITS = ('train', 'dev', 'test_iid', 'test_comp')


def read(ddir, split):
    path = os.path.join(ddir, split + '.jsonl.gz')
    if not os.path.exists(path): return []
    out = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def class_summary(labels):
    cnt = collections.Counter(labels); n = len(labels)
    f1 = sum(1 for c in cnt.values() if c == 1); f2 = sum(1 for c in cnt.values() if c == 2)
    est, se = ST.chao1(list(cnt.values()))
    return {'observed': len(cnt), 'n': n, 'singletons': f1, 'doubletons': f2, 'good_turing_coverage': (1.0 - f1 / n) if n else None, 'chao1': est, 'chao1_se': se, 'largest_class': max(cnt.values()) if cnt else 0}


def points(n, first=100):
    pts = []; m = first
    while m < n: pts.append(m); m = int(m * 2.15) if m < 1000 else int(m * 2)
    pts.append(n)
    return sorted(set(p for p in pts if 0 < p <= n))


def entropy_bits(counts):
    tot = float(sum(counts))
    return -sum(c / tot * math.log2(c / tot) for c in counts if c) if tot else 0.0


def stats(ddir, repeats=10):
    recs = []
    by = {}
    for sp in SPLITS:
        r = read(ddir, sp); by[sp] = r; recs += r
    n = len(recs)
    out = {'dataset': os.path.abspath(ddir), 'n': {sp: len(by[sp]) for sp in SPLITS}, 'total': n}
    if not n: return out
    out['classes'] = {c: class_summary([r.get(c) for r in recs if r.get(c) is not None]) for c in ('skeleton', 'shape', 'recipe')}
    out['classes_train'] = {c: class_summary([r.get(c) for r in by['train'] if r.get(c) is not None]) for c in ('skeleton', 'shape', 'recipe')}
    pts = points(n)
    rare = {c: ST.rarefaction([r.get(c) for r in recs], pts, repeats=repeats) for c in ('skeleton', 'shape', 'recipe')}
    # distinct lemmas among the first m theorems of random orders
    lem = [tuple(r.get('lemmas', ())) for r in recs]
    import random
    rng = random.Random(0); rl = []
    for m in pts:
        vals = []
        for _ in range(repeats):
            idx = rng.sample(range(n), m); seen = set()
            for i in idx: seen.update(lem[i])
            vals.append(len(seen))
        mu = sum(vals) / len(vals); sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        rl.append((m, mu, sd))
    rare['lemmas'] = rl
    out['rarefaction'] = {c: [{'m': m, 'mean': mu, 'sd': sd} for m, mu, sd in v] for c, v in rare.items()}
    kinds = collections.Counter(); has = collections.Counter(); steps = 0
    for r in recs:
        ks = [s.get('kind') for s in r['steps']]; kinds.update(ks); has.update(set(ks)); steps += len(ks)
    out['kinds'] = {'steps': steps, 'per_kind': {k: {'steps': v, 'share_of_steps': v / steps, 'share_of_theorems': has[k] / n} for k, v in kinds.most_common()}, 'distinct_kinds': len(kinds)}
    lc = collections.Counter(l for r in recs for l in r.get('lemmas', ()))
    out['lemmas'] = {'distinct_applied': len(lc), 'applications': sum(lc.values()), 'entropy_bits': entropy_bits(lc.values()), 'effective_lemmas': 2 ** entropy_bits(lc.values()), 'top': lc.most_common(15),
                     'used_once': sum(1 for v in lc.values() if v == 1)}
    out['proofs'] = {'mean_steps': sum(len(r['steps']) for r in recs) / n, 'mean_depth': sum(r.get('depth', 0) for r in recs) / n, 'mean_size': sum(r.get('size', 0) for r in recs) / n,
                     'closed_at_once_share': sum(1 for r in recs if r.get('solved_by')) / n, 'mean_sat_rate': sum((r.get('semantic') or {}).get('sat_rate', 0) for r in recs) / n}
    out['by_depth'] = {str(d): {'n': c, 'mean_steps': sum(len(r['steps']) for r in recs if r.get('depth') == d) / c} for d, c in sorted(collections.Counter(r.get('depth') for r in recs).items()) if d is not None}
    return out


def summary_lines(s):
    if not s.get('total'): return ['(empty dataset)']
    c = s['classes']['skeleton']; sh = s['classes']['shape']
    L = ['%d theorems; %d distinct proof skeletons (%d singletons: coverage %.3f; Chao1 lower bound %.0f +- %.0f of how many the generator can make); %d distinct statement shapes.' % (
        s['total'], c['observed'], c['singletons'], c['good_turing_coverage'], c['chao1'], c['chao1_se'], sh['observed'])]
    k = s['kinds']['per_kind']
    L.append('Actions used: ' + ', '.join('%s %.0f%% of steps' % (name, 100 * v['share_of_steps']) for name, v in list(k.items())[:6]) + ' (%d kinds in all).' % s['kinds']['distinct_kinds'])
    lm = s['lemmas']
    L.append('%d library lemmas applied (effective number %.0f by entropy; %d used once).' % (lm['distinct_applied'], lm['effective_lemmas'], lm['used_once']))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dir'); ap.add_argument('--out', default=''); ap.add_argument('--repeats', type=int, default=10)
    a = ap.parse_args(argv)
    s = stats(a.dir, a.repeats)
    out = a.out or os.path.join(a.dir, 'stats.json')
    tmp = out + '.tmp'
    with open(tmp, 'w') as f: json.dump(s, f, indent=1)
    os.replace(tmp, out)
    for line in summary_lines(s): print(line)
    print('wrote', out)


if __name__ == '__main__':
    main()
