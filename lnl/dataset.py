"""dataset.py -- the dataset: the problem bank with its reference proofs, the splits, the library corpus and the card that says what is in them.

    python3 dataset.py OUT_DIR                 build everything into OUT_DIR (a few minutes): bank.json, splits.json, library.json, DATASET_CARD.md, manifest.json
    python3 dataset.py --verify OUT_DIR        read the files back and replay every demonstration in a fresh engine over a freshly built world

    import dataset
    info = dataset.build(out_dir)              # the files; info: counts and digests
    dataset.verify(out_dir)                    # raises if a demonstration does not finish its problem

The files (all JSON, deterministic: two builds of the same sources give the same digests):

    bank.json      {'format', 'globals' (names in id order), 'world_digest', 'problems': [..]}; a problem has its text, source, topics, level, kind, answer, the formal statement (printed, as an AST when the
                   evaluator reads it, and as the typed term builder's code), the premises of both regimes, the digests, its split and family, and DEMONSTRATIONS: kernel-checked sequences of actions in the
                   portable format of gym_acts, for the prefix regime and (when the proof is self-contained) the strict regime
    splits.json    the policy (family key, salt, fractions) and the splits of the problems and of the library lemmas
    library.json   the proofs of the lemmas of the base library as sequences of actions (lib_corpus.py)
    manifest.json  digests of the files
"""
import os, sys, json, hashlib, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank as B, bank_all, bank_split, lib_corpus, recorder as REC
import gym_env as G, gym_acts as A

VERSION = bank_split.DATASET_VERSION


def _canon(obj): return json.dumps(obj, sort_keys=True, separators=(',', ':'))
def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()[:16]


def splits_doc(names, lemma_names=()):
    sp = bank_split.splits(names)
    return {'dataset_version': VERSION, 'salt_number': bank_split.SALT_NUMBER, 'fractions': dict(bank_split.FRACTIONS), 'families': dict(sorted(bank_split.FAMILIES.items())),
            'problems': {s: list(ns) for s, ns in sp.items()}, 'lemmas': {s: [n for n in lemma_names if bank_split.lemma_split(n) == s] for s, _ in bank_split.FRACTIONS}}


def card(data, splits, library=None):
    """the dataset card (markdown) from the bank's JSON, the splits document and (optionally) the library corpus; deterministic"""
    probs = data['problems']; P = {p['name']: p for p in probs}
    L = []; add = L.append
    add('# Olympiad problem bank -- dataset version %d' % VERSION)
    add('')
    add('Formal olympiad problems over the integers and naturals, stated in the hybrid kernel, each with a **kernel-checked reference proof written as a sequence of actions** of the gym action language '
        '(no text interface: the learner sees a goal and chooses actions).  %d problems, %d of them with a second, self-contained proof for the *strict* regime.' % (len(probs), sum(1 for p in probs if 'demonstration_strict' in p)))
    add('')
    lv = collections.Counter(p['level'] for p in probs); kd = collections.Counter(p['kind'] for p in probs); tp = collections.Counter(t for p in probs for t in p['topics'])
    src = collections.Counter('IMO' if p['source'].startswith('IMO') else 'folklore' if p['source'].startswith('folklore') else 'other' for p in probs)
    add('## Contents')
    add('')
    add('| level | problems | meaning |'); add('|---|---|---|')
    for l in sorted(B.LEVELS): add('| %d | %d | %s |' % (l, lv[l], B.LEVELS[l]))
    add('')
    add('kinds: ' + ', '.join('%s %d' % (k, kd[k]) for k in sorted(kd)) + '.  sources: ' + ', '.join('%s %d' % kv for kv in sorted(src.items())) + '.')
    add('')
    add('topics: ' + ', '.join('%s %d' % kv for kv in sorted(tp.items())) + '.')
    add('')
    imo = sorted((p['year'] or 0, p['name'], p['source']) for p in probs if p['source'].startswith('IMO'))
    add('IMO problems (%d): ' % len(imo) + ', '.join('%s (%s)' % (n, s) for _, n, s in imo) + '.')
    add('')
    add('## Splits')
    add('')
    add('Splits are by **family** (the two parts of one olympiad problem, a statement in other clothes, an argument with another constant stay together) and are a pure function of the family key '
        '(`bank_split.py`, salt number %d, fractions %s): a later version of the bank never moves an existing problem from one split to another.' % (splits['salt_number'], ', '.join('%s %d%%' % kv for kv in splits['fractions'].items())))
    add('')
    add('| split | problems | by level | IMO | demonstration steps |'); add('|---|---|---|---|---|')
    for s, ns in splits['problems'].items():
        l2 = collections.Counter(P[n]['level'] for n in ns)
        add('| %s | %d | %s | %d | %d |' % (s, len(ns), ' '.join('L%d:%d' % (k, l2[k]) for k in sorted(l2)), sum(1 for n in ns if P[n]['source'].startswith('IMO')), sum(P[n]['demonstration']['length'] for n in ns if 'demonstration' in P[n])))
    add('')
    for s, ns in splits['problems'].items(): add('* **%s**: %s' % (s, ', '.join('`%s`' % n for n in ns)))
    add('')
    add('## Regimes')
    add('')
    add('A problem is attacked in the world in which the library precedes it.  In the **prefix** regime every global before the theorem is a premise (the lemmas written for the problem included); in the '
        '**strict** regime the lemmas written for the problem are hidden, so the agent must find the ideas.  `premises.prefix` / `premises.strict` of a problem is the number of globals the engine offers.')
    add('')
    add('## Reference proofs')
    add('')
    steps = [p['demonstration']['length'] for p in probs if 'demonstration' in p]
    add('%d demonstrations, %d actions in all (shortest %d, longest %d).  Every one replays through the engine from its JSON in a fresh engine over a freshly built world, and the proof term it builds is '
        'checked by the kernel (and, for the proofs that are not too large, by the independent checker).' % (len(steps), sum(steps), min(steps), max(steps)))
    kinds = collections.Counter(st['kind'] for p in probs if 'demonstration' in p for st in p['demonstration']['steps'])
    add('')
    add('actions by kind: ' + ', '.join('%s %d' % kv for kv in kinds.most_common()) + '.')
    add('')
    add('## Faithfulness of the statements')
    add('')
    add('A kernel-checked proof proves the *formal* statement, not the problem.  Every statement is therefore read back into an AST and compared with an independent description of what the problem says '
        '(`semantic.Spec`: Python predicates, the numbers in the relation atoms, the body of existentials) on a box of values, and deliberately mutated statements (changed operators, constants, '
        'quantifiers, guards) must be told from it; a mutant that means the same is listed with its reason.  See `BANK.md`.  Inequalities that are over the reals in the original are stated for integers (noted in the problem).')
    add('')
    if library is not None:
        ls = collections.Counter(l['split'] for l in library['lemmas'])
        add('## Library corpus')
        add('')
        add('`library.json`: the proofs of the %d lemmas of the base library (%d actions) as action sequences, split by lemma name (%s).  %d lemma(s) have the statement of a bank problem and are marked `bank_duplicate`.' % (
            len(library['lemmas']), sum(l['length'] for l in library['lemmas']), ', '.join('%s %d' % (k, ls[k]) for k in ('train', 'dev', 'test')), sum(1 for l in library['lemmas'] if l['bank_duplicate'])))
        add('')
    add('## Files')
    add('')
    add('`bank.json` (problems and demonstrations), `splits.json`, `library.json`, `manifest.json` (digests).  World digest of the bank: `%s`.' % data['world_digest'])
    add('')
    return '\n'.join(L)


def build(out_dir, library=True, progress=None):
    t0 = time.time(); say = progress or (lambda *a: None)
    os.makedirs(out_dir, exist_ok=True)
    bk = bank_all.load(); w2, recs = bk.recordings()
    eng = G.Engine(w2, kinds=set(G.Kind), strict=True)
    demos = {}; strict = {}
    for i, (n, pl) in enumerate(bk.placed.items()):
        demos[n] = bk.demonstrate(n, engine=eng)
        if not demos[n].ok: raise RuntimeError('the reference proof of %s does not replay: %s' % (n, demos[n].gap))
        if pl.problem.strict and pl.premises('strict') != pl.premises('prefix'):
            strict[n] = bk.demonstrate(n, engine=eng, regime='strict')
            if not strict[n].ok: raise RuntimeError('the strict reference proof of %s does not replay: %s' % (n, strict[n].gap))
        say('demonstration %d/%d %s' % (i + 1, len(bk.placed), n))
    path = os.path.join(out_dir, 'bank.json')
    data = bk.export(path, demos=demos, strict_demos=strict)
    for rec in data['problems']:
        rec['split'] = bank_split.split_of(rec['name']); rec['family'] = bank_split.family(rec['name'])
    with open(path, 'w') as f: json.dump(data, f, indent=1)
    lib = None
    if library:
        digs = {r['statement_digest'] for r in data['problems']}
        lib = lib_corpus.build(bank_digests=digs, progress=lambda i, n, nm: say('library %d/%d %s' % (i, n, nm)) if i % 50 == 0 else None)
        lib_corpus.write(lib, os.path.join(out_dir, 'library.json'))
    sp = splits_doc([r['name'] for r in data['problems']], [l['name'] for l in lib['lemmas']] if lib else ())
    with open(os.path.join(out_dir, 'splits.json'), 'w') as f: json.dump(sp, f, indent=1)
    with open(os.path.join(out_dir, 'DATASET_CARD.md'), 'w') as f: f.write(card(data, sp, lib))
    man = {'dataset_version': VERSION, 'files': {fn: file_digest(os.path.join(out_dir, fn)) for fn in ('bank.json', 'splits.json', 'DATASET_CARD.md') + (('library.json',) if lib else ())}}
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f: json.dump(man, f, indent=1)
    return {'problems': len(data['problems']), 'demonstrations': len(demos), 'strict_demonstrations': len(strict), 'actions': sum(len(r.portable) for r in demos.values()),
            'lemmas': len(lib['lemmas']) if lib else 0, 'seconds': round(time.time() - t0, 1), 'manifest': man}


def verify(out_dir, progress=None):
    """read the files back; every demonstration (problems, both regimes) must finish its problem in a fresh engine over a freshly built world, and the digests must be those of the manifest.  Returns the number of proofs replayed."""
    say = progress or (lambda *a: None)
    man = json.load(open(os.path.join(out_dir, 'manifest.json')))
    for fn, dg in man['files'].items():
        if file_digest(os.path.join(out_dir, fn)) != dg: raise RuntimeError('%s is not the file the manifest describes' % fn)
    data = json.load(open(os.path.join(out_dir, 'bank.json')))
    bk = bank_all.load(); eng = G.Engine(bk.w, kinds=set(G.Kind), strict=True)
    if B.world_digest(bk.w, bk.k.num_globals()) != data['world_digest']: raise RuntimeError('the world built now is not the world of the dataset')
    n = 0
    for rec in data['problems']:
        pl = bk.placed[rec['name']]
        if B.statement_digest(bk.w, pl.stmt) != rec['statement_digest']: raise RuntimeError('the statement of %s changed' % rec['name'])
        for key, regime in (('demonstration', 'prefix'), ('demonstration_strict', 'strict')):
            if key in rec:
                A.replay_portable(eng, pl.stmt, rec[key]['steps'], pl.premises(regime)); n += 1
        say('verified %s' % rec['name'])
    return n


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--verify':
        print('replayed %d demonstrations' % verify(args[1], progress=lambda m: None))
    else:
        info = build(args[0] if args else 'dataset_out', progress=lambda m: print('\r' + m[:70].ljust(70), end='', flush=True) if m.endswith(('0', '5')) else None)
        print(); print(json.dumps(info, indent=1))
