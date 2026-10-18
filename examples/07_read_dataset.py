"""07 -- the dataset on disk: what is in data/dataset_v1, how it is split, and how to check it.

    python3 examples/07_read_dataset.py                      (a second; no kernel needed)

The dataset (src/python/problems/dataset.py builds it, `dataset.py --verify DIR` replays it) is plain JSON:

    bank.json        the problems: text, source, topics, level, the formal statement (printed, as a syntax tree and as a code of the typed term builder), the kernel-checked reference proofs as sequences of
                     actions (the prefix regime and, for self-contained proofs, the strict regime), the split and the family of each problem
    splits.json      the split policy: problems of one FAMILY (variants of one idea) go to one split, by a salted hash; the lemmas of the base library are split the same way
    library.json     the proofs of the base library's lemmas (nat/bin/int/number-theory) as sequences of actions: practice material that is not a bank problem
    DATASET_CARD.md  a human-readable summary
    manifest.json    digests of the files (the files are deterministic: two builds from the same sources give the same bytes)

This script reads them without the kernel, prints the shape of the data, and checks the manifest.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import collections, hashlib, json, os

D = os.path.join(itp_paths.DATA, 'dataset_v1')
load = lambda name: json.load(open(os.path.join(D, name)))


def digest(path):                                                       # (the same digest as dataset.file_digest: the first 16 hex digits of the SHA-256 of the bytes)
    h = hashlib.sha256()
    with open(path, 'rb') as f: h.update(f.read())
    return h.hexdigest()[:16]


manifest = load('manifest.json')
for name, want in manifest['files'].items():
    got = digest(os.path.join(D, name))
    print('%-16s %s  %s' % (name, got, 'ok' if got == want else 'DIFFERS from the manifest (%s)' % want))
    assert got == want

bank, splits, library = load('bank.json'), load('splits.json'), load('library.json')
problems = bank['problems']
print('\nbank.json: %d problems over a world of %d globals (digest %s...)' % (len(problems), len(bank['globals']), bank['world_digest'][:12]))
print('  by level:   ', dict(sorted(collections.Counter(p['level'] for p in problems).items())))
print('  by split:   ', dict(collections.Counter(p['split'] for p in problems)))
print('  by kind:    ', dict(collections.Counter(p['kind'] for p in problems)))
print('  topics:     ', dict(collections.Counter(t for p in problems for t in p['topics']).most_common(8)), '...')
lengths = [p['demonstration']['length'] for p in problems if 'demonstration' in p]
print('  reference proofs: %d (prefix regime; %d more for the strict regime), %d to %d actions, mean %.1f'
      % (len(lengths), sum(1 for p in problems if 'demonstration_strict' in p), min(lengths), max(lengths), sum(lengths) / len(lengths)))
print('  IMO problems:', ', '.join(p['name'] for p in problems if p['source'].startswith('IMO')))

one = next(p for p in problems if p['name'] == 'imo1959_p1')
print('\none record (imo1959_p1): keys %s' % sorted(one))
print('  statement (printed):', one['statement'][:100] + ' ...')
print('  statement code:     %d tokens, first %s' % (len(one['statement_code']), one['statement_code'][:3]))
print('  first action:       %s' % one['demonstration']['steps'][0])

print('\nsplits.json: policy %s' % {k: splits[k] for k in ('dataset_version', 'fractions')})
print('  problems:', {s: len(ns) for s, ns in splits['problems'].items()}, '  library lemmas:', {s: len(ns) for s, ns in splits['lemmas'].items()})
print('library.json: %d lemmas with recorded proofs, %d of them not already a bank problem' % (len(library['lemmas']), sum(1 for l in library['lemmas'] if not l['bank_duplicate'])))

# the splits are consistent between the files
by_split = collections.defaultdict(set)
for p in problems: by_split[p['split']].add(p['name'])
assert {s: set(ns) for s, ns in splits['problems'].items()} == dict(by_split)
print('EXAMPLE OK')
