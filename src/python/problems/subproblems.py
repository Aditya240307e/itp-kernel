"""subproblems.py -- the hindsight half of the dataset: the theorems that the reference proofs contain, and a label for every action.

    python3 subproblems.py DATASET_DIR             build annotations.json and subproblems.json next to bank.json (about 4 minutes); update manifest.json and DATASET_CARD.md
    python3 subproblems.py --verify DATASET_DIR    replay every sub-theorem from its data in a fresh engine over a freshly built world (about 3 minutes)

A reference proof is a tree: each action is applied to a goal and leaves goals.  Every closed subtree is a theorem with a proof (the goal, closed over its hypotheses, and the actions of the subtree), and every action
that derives a fact proves that fact in the context it was applied in.  gym_tree.py extracts them, replaying each in a fresh proof state and checking it with the kernel; this module does it for the whole bank.

annotations.json -- one record per problem, one row per action of the reference proof (the labels of the manager / worker split, docs/ARCHITECTURE_RL.md):
    direction     what the action does to its goal: CLOSE (ring, linarith, ...), BACKWARD (apply, rewrite, refine, use, by_contra), FORWARD (adds facts), SPLIT (cases, induction), CLAIM (a stated lemma)
    depth         the depth of the goal in the proof tree        subtree   the number of actions that close this goal (including this one): what a worker needs to be able to do
    children      the goals the action leaves                    open_after   the goals open after the action          closes   the action leaves no goal
    parent_step   the action applied to the goal this one came from (None for the first)                   hyps   the hypotheses of the goal
  and per problem: the kinds of action and the techniques they stand for, the lemmas used, the CLAIMs (statement, length of its proof, length of the rest), depth, width, and what the hindsight extraction found.
subproblems.json -- every sub-theorem, as a problem of its own: its kind (solved: a closed subtree; derived: a fact one action proved), its direction, the closed statement (printed, and as the typed term builder's
    code), the number of globals it may use, the portable actions of its proof, where it came from.  The split of a sub-theorem is that of the first problem (train, then dev, then test) it occurs in: a theorem
    that a test problem shares with a training problem is not a test of anything and is listed under train, with the problems it also occurs in.

Everything is verified when it is built (gym_tree.py replays and kernel-checks each example) and again by `--verify`.  The files are deterministic: two builds give the same digests.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, collections, time
import bank as B, bank_all, bank_split
import gym_env as G, gym_acts as A, gym_tree as T

FORMAT = 1
MAX_LEN = 60                                       # the longest sub-proof kept (the longest reference proof has 53 actions: every proper subtree qualifies)
SPLIT_ORDER = ('train', 'dev', 'test')
TECHNIQUES = {'induction': ('INDUCT', 'INDUCT_WITH'), 'cases': ('CASES', 'INTERVAL_CASES'), 'claim': ('CLAIM',), 'contradiction': ('BY_CONTRA',), 'rewriting': ('REWRITE', 'REWRITE_WITH', 'REWRITE_AT'),
              'lemma application': ('APPLY', 'APPLY_WITH', 'REFINE', 'FORWARD'), 'witness': ('USE',), 'injectivity': ('INJECT',), 'instantiation': ('INSTANTIATE',), 'ring': ('RING',),
              'linarith': ('LINARITH', 'LINCOMB'), 'nlinarith': ('NLINARITH',)}


def replay_tree(bk, eng, name, steps, regime='prefix'):
    """the reference proof of `name` played on a tracking episode (it must finish)"""
    pl = bk.placed[name]
    ep = G.Episode(eng, pl.stmt, name, premises=pl.premises(regime), track=True)
    for st in steps:
        menu = ep.menu()
        ep.step(A.from_portable(eng, menu if st['kind'] in A.MENU_KINDS else A.NO_MENU, st))
    if not ep.done: raise RuntimeError('the reference proof of %s does not finish its problem' % name)
    return ep


def action_rows(tree):
    """the label of every action, in the order of the actions"""
    rows = []; open_ = 1
    for n in tree.by_step:
        open_ += len(n.children) - 1
        rows.append({'direction': T.DIRECTION[n.action.kind], 'depth': n.depth, 'subtree': n.last - n.step + 1, 'children': len(n.children), 'open_after': open_,
                     'closes': not n.children, 'parent_step': None if n.parent is None else n.parent.step, 'hyps': n.goal.n_hyps})
    return rows


def is_theorem(k, gid):
    """is the global a proof (its type is a proposition)? Not a function, a constructor or a type"""
    ok, sort = k.infer_open(k.global_info(gid)[1], [])
    return bool(ok) and sort == k.mk_type(0)


def lemmas_used(steps, names, keep=None):
    """the lemmas (theorems, by name) that the actions mention, in order of first use; `keep(gid)` says which globals count (default: all of them)"""
    out = []
    def add(gid):
        if 0 <= gid < len(names) and names[gid] not in out and (keep is None or keep(gid)): out.append(names[gid])
    for s in steps:
        src = s.get('source')
        if isinstance(src, list) and src[:1] == ['c']: add(src[1])
        for code in s.get('codes', ()):
            for tok in code:
                if tok and tok[0] == 'c': add(tok[1])
    return out


def claims_of(eng, tree):
    """the CLAIM actions of the proof: the plan a human would write down"""
    w = eng.w; out = []
    for n in tree.by_step:
        if n.action.kind != G.Kind.CLAIM or not n.children: continue
        first = n.children[0]
        out.append({'step': n.step, 'statement': w.show(w.T(first.goal.target)), 'proof_steps': first.last - first.step + 1, 'rest_steps': n.last - first.last})
    return out


def problem_record(eng, rec, ep, names):
    tree = ep.tree; rows = action_rows(tree); steps = rec['demonstration']['steps']
    kinds = collections.Counter(s['kind'] for s in steps)
    return {'actions': len(steps), 'depth': max(r['depth'] for r in rows), 'width': max(r['open_after'] for r in rows), 'by_direction': dict(sorted(collections.Counter(r['direction'] for r in rows).items())),
            'kinds': dict(sorted(kinds.items())), 'techniques': [t for t, ks in TECHNIQUES.items() if any(k in kinds for k in ks)], 'lemmas': lemmas_used(steps, names, lambda gid: is_theorem(eng.k, gid)),
            'claims': claims_of(eng, tree), 'rows': rows}


def example_record(rec, ex):
    o = ex.origin; d = ex.to_dict()
    ident = '%s#%d' % (rec['name'], o['step']) + ('' if ex.kind == 'solved' else '.%d' % o['fact'])
    return {'id': ident, 'problem': rec['name'], 'split': rec['split'], 'family': rec['family'], 'kind': ex.kind, 'direction': T.DIRECTION[G.Kind[o['kind']]],
            'length': d['length'], 'depth': o['depth'], 'statement': d['statement'], 'statement_digest': d['statement_digest'], 'statement_code': d['statement_code'], 'premises': d['premises'],
            'steps': d['steps'], 'origin': {k: v for k, v in o.items() if k in ('step', 'node', 'fact', 'kind')}}


def build_records(bk, eng, data, progress=None):
    """(annotations document, subproblems document) for the problems of `data` (bank.json); every sub-theorem verified"""
    say = progress or (lambda *a: None)
    names = data['globals']; probs = {r['name']: r for r in data['problems']}
    order = [r['name'] for s in SPLIT_ORDER for r in data['problems'] if r['split'] == s and 'demonstration' in r]          # (a statement-only problem has no proof to take sub-theorems from)
    ann = {}; examples = []; seen = {}; total = collections.Counter(); per_split = collections.Counter()
    for i, name in enumerate(order):
        rec = probs[name]; ep = replay_tree(bk, eng, name, rec['demonstration']['steps'])
        info = problem_record(eng, rec, ep, names)
        exs = ep.tree.examples(verify=True, max_len=MAX_LEN)
        st = dict(ep.tree.stats)
        kept = collections.Counter(); dup = 0
        for ex in exs:
            r = example_record(rec, ex)
            key = (r['statement_digest'], json.dumps(r['premises'], sort_keys=True))
            if key in seen:
                seen[key]['also_in'].append(name); dup += 1; continue
            r['also_in'] = []; seen[key] = r; examples.append(r); kept[ex.kind] += 1; per_split[(rec['split'], ex.kind)] += 1
        info['hindsight'] = {'solved': kept['solved'], 'derived': kept['derived'], 'duplicates_in_dataset': dup,
                             **{k: st[k] for k in ('too_long', 'context_changed', 'pi_fact', 'settled_at_once_derived', 'settled_at_once_solved', 'duplicates', 'not_portable_solved', 'not_portable_derived', 'not_replayed_solved', 'not_replayed_derived') if st.get(k)}}
        if info['hindsight'].get('not_replayed_solved') or info['hindsight'].get('not_replayed_derived'):
            raise RuntimeError('%s: a candidate sub-theorem did not replay (%s)' % (name, info['hindsight']))
        ann[name] = info
        for k in ('solved', 'derived'): total[k] += kept[k]
        total['duplicates_in_dataset'] += dup
        say('%d/%d %s: %d actions, %d solved + %d derived' % (i + 1, len(order), name, info['actions'], kept['solved'], kept['derived']))
    counts = {'problems': len(order), 'actions': sum(a['actions'] for a in ann.values()), 'solved': total['solved'], 'derived': total['derived'], 'duplicates_in_dataset': total['duplicates_in_dataset'],
              'by_split': {s: {k: per_split[(s, k)] for k in ('solved', 'derived')} for s in SPLIT_ORDER}}
    bank_order = [r['name'] for r in data['problems'] if r['name'] in ann]
    ann_doc = {'format': FORMAT, 'world_digest': data['world_digest'], 'directions': list(T.DIRECTIONS), 'problems': {n: ann[n] for n in bank_order}}
    sub_doc = {'format': FORMAT, 'world_digest': data['world_digest'], 'max_len': MAX_LEN, 'counts': counts, 'examples': examples}
    return ann_doc, sub_doc


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ files
def dump(path, doc, rows_key):
    """canonical JSON, one record per line (so that a diff of two builds is readable and the digest is stable)"""
    enc = lambda o: json.dumps(o, sort_keys=True, separators=(',', ':'))
    rows = doc[rows_key]; is_map = isinstance(rows, dict)
    head = ','.join('%s:%s' % (json.dumps(k), enc(v)) for k, v in doc.items() if k != rows_key)
    items = [json.dumps(k) + ':' + enc(v) for k, v in rows.items()] if is_map else [enc(v) for v in rows]
    with open(path, 'w') as f: f.write('{' + head + ',' + json.dumps(rows_key) + ':' + ('{' if is_map else '[') + '\n' + ',\n'.join(items) + '\n' + ('}' if is_map else ']') + '}\n')


def summary_lines(sub):
    """the part of the dataset card about these files"""
    c = sub['counts']; L = []
    L.append('## Annotations and sub-theorems (hindsight)')
    L.append('')
    L.append('`annotations.json`: for every action of every reference proof its **direction** (CLOSE / BACKWARD / FORWARD / SPLIT / CLAIM), the depth of its goal in the proof tree, the number of actions that close that goal '
             '(`subtree`), the goals it leaves, the goals open after it; per problem the techniques, the lemmas used, the CLAIMs (statement and size of their proofs), depth and width of the tree.  These are the labels for '
             'a manager that chooses the macro step and a worker that proves the pieces.')
    L.append('')
    L.append('`subproblems.json`: the **theorems that the reference proofs contain**, each a problem of its own with a kernel-checked proof: %d closed subtrees (`solved`: a goal, closed over its hypotheses, with the actions that closed it) '
             'and %d facts a single action derived (`derived`), %d actions in %d proofs.  Every one was replayed in a fresh proof state and checked by the kernel when the file was made, and `subproblems.py --verify` does it again.  '
             'A sub-theorem belongs to the first of train, dev, test whose problem contains it (%d occurrences in later problems are recorded in `also_in`).  '
             % (c['solved'], c['derived'], c['actions'], c['problems'], c['duplicates_in_dataset']))
    L.append('')
    L.append('| split | solved | derived |'); L.append('|---|---|---|')
    for s in SPLIT_ORDER: L.append('| %s | %d | %d |' % (s, c['by_split'][s]['solved'], c['by_split'][s]['derived']))
    L.append('')
    L.append('Use: hindsight relabelling (a failed attempt that proved a piece is turned into successes), goal-conditioned warm-up for the worker, a curriculum by `length`.  Train only on the `train` split.')
    L.append('')
    return L


def build(dataset_dir, progress=None):
    """annotations.json and subproblems.json for the dataset in `dataset_dir`; the manifest and the card are brought up to date"""
    import dataset
    t0 = time.time()
    data = json.load(open(os.path.join(dataset_dir, 'bank.json')))
    bk = bank_all.load()
    if B.world_digest(bk.w, bk.k.num_globals()) != data['world_digest']: raise RuntimeError('the world built now is not the world of the dataset')
    eng = G.Engine(bk.w, kinds=set(G.Kind), strict=True)
    ann, sub = build_records(bk, eng, data, progress)
    dump(os.path.join(dataset_dir, 'annotations.json'), ann, 'problems')
    dump(os.path.join(dataset_dir, 'subproblems.json'), sub, 'examples')
    update_card_and_manifest(dataset_dir, sub)
    return {'seconds': round(time.time() - t0, 1), **sub['counts']}


def update_card_and_manifest(dataset_dir, sub):
    import dataset
    splits = json.load(open(os.path.join(dataset_dir, 'splits.json'))); data = json.load(open(os.path.join(dataset_dir, 'bank.json')))
    lib = json.load(open(os.path.join(dataset_dir, 'library.json'))) if os.path.exists(os.path.join(dataset_dir, 'library.json')) else None
    with open(os.path.join(dataset_dir, 'DATASET_CARD.md'), 'w') as f: f.write(dataset.card(data, splits, lib, extra=summary_lines(sub)))
    man_path = os.path.join(dataset_dir, 'manifest.json'); man = json.load(open(man_path))
    for fn in ('annotations.json', 'subproblems.json', 'DATASET_CARD.md'): man['files'][fn] = dataset.file_digest(os.path.join(dataset_dir, fn))
    with open(man_path, 'w') as f: json.dump(man, f, indent=1)


def verify(dataset_dir, progress=None):
    """read the files back: digests as in the manifest; the annotations agree with a fresh replay of every reference proof; every sub-theorem, from its data alone, is proved in a fresh engine over a freshly
    built world by its own actions (the kernel checks the term) and says the statement it claims.  Returns the number of sub-theorems replayed."""
    import dataset
    say = progress or (lambda *a: None)
    man = json.load(open(os.path.join(dataset_dir, 'manifest.json')))
    for fn in ('annotations.json', 'subproblems.json'):
        if fn not in man['files'] or dataset.file_digest(os.path.join(dataset_dir, fn)) != man['files'][fn]: raise RuntimeError('%s is not the file the manifest describes' % fn)
    data = json.load(open(os.path.join(dataset_dir, 'bank.json')))
    ann = json.load(open(os.path.join(dataset_dir, 'annotations.json'))); sub = json.load(open(os.path.join(dataset_dir, 'subproblems.json')))
    bk = bank_all.load()
    if B.world_digest(bk.w, bk.k.num_globals()) != data['world_digest'] or ann['world_digest'] != data['world_digest'] or sub['world_digest'] != data['world_digest']:
        raise RuntimeError('the world built now is not the world of the dataset')
    eng = G.Engine(bk.w, kinds=set(G.Kind), strict=True); fresh = G.Engine(bk.w)
    for rec in data['problems']:
        if 'demonstration' not in rec: continue
        ep = replay_tree(bk, eng, rec['name'], rec['demonstration']['steps'])
        if action_rows(ep.tree) != ann['problems'][rec['name']]['rows']: raise RuntimeError('the labels of %s are not those of its proof' % rec['name'])
    n = 0
    for ex in sub['examples']:
        stmt = T.statement_from_code(fresh, ex['statement_code'], ex['premises'])
        if T.digest(bk.w.show(stmt)) != ex['statement_digest'] or bk.w.show(stmt) != ex['statement']: raise RuntimeError('%s does not say its statement' % ex['id'])
        A.replay_portable(fresh, stmt, ex['steps'], ex['premises']); n += 1
        if n % 200 == 0: say('verified %d/%d' % (n, len(sub['examples'])))
    return n


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--verify':
        print('replayed %d sub-theorems' % verify(args[1], progress=print))
    else:
        info = build(args[0] if args else os.path.join(itp_paths.DATA, 'dataset_v2'), progress=lambda m: print(m, flush=True))
        print(json.dumps(info, indent=1))
