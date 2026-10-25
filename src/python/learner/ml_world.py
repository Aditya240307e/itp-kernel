"""ml_world.py -- the world every experiment tool starts from, and the options they share.

    ap = argparse.ArgumentParser(); ml_world.add_args(ap); a = ap.parse_args()
    bk, ctx, eng = ml_world.load(a)                       # the bank's world (a minute), the data context, an engine that offers every kind of action
    lib = ml_world.load_library(a, bk, eng, ctx, ddir)    # the lemmas an earlier run proved (--lemmas FILE), or None

The world is the one THE DATASET describes: the reference proofs name library lemmas by global id, so a world with other ids would replay them wrongly.  `load` therefore reads `bank.json` of the dataset
(the positional argument `data` of every tool), builds the world it says -- the whole bank, or for a PARTIAL dataset (`dataset.py OUT --only a,b,c`: tests and the smoke runs of the experiment scripts, a world
in seconds instead of a minute) the problems it lists -- and refuses to go on when the world's digest is not the dataset's.  `--only` is an override for tests; with a dataset of another world it is an error.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time
import bank_all, bank_split as BS
import gym_env as G
import ml_data as D


def add_args(ap, lemmas=True):
    ap.add_argument('--only', default='', help='build the world for these problems only (comma separated names, or @FILE with a name on every line); normally not needed: a partial dataset (dataset.py --only) implies its world')
    if lemmas: ap.add_argument('--lemmas', default='', help='a file of lemmas an earlier run proved (ml_lemmas): they are added to the world, and to the premises of every task')


def names_of(arg):
    if not arg: return None
    if arg.startswith('@'): return {l.strip() for l in open(arg[1:]) if l.strip()}
    return {n for n in arg.split(',') if n}


def dataset_world_info(ddir):
    """(world digest, the problems of a partial dataset or None) of the dataset in `ddir`; (None, None) when there is no bank.json"""
    path = os.path.join(ddir or '', 'bank.json')
    if not ddir or not os.path.exists(path): return None, None
    with open(path) as f: data = json.load(f)
    return data.get('world_digest'), (set(data['only']) if data.get('only') else None)


def load(a, strict=False, kinds=None):
    """(bk, ctx, eng): the world of the dataset `a.data` (restricted by the dataset's `only`, or by --only), the data context and an engine with every kind of action"""
    import bank as B
    sys.setrecursionlimit(100000)
    digest, ds_only = dataset_world_info(getattr(a, 'data', None))
    only = names_of(getattr(a, 'only', '')) or ds_only
    t0 = time.time()
    bk = bank_all.load(only=only) if only else bank_all.load()
    if digest is not None and B.world_digest(bk.w, bk.k.num_globals()) != digest:
        raise SystemExit('the world built now (%d globals%s) is not the world of the dataset %s: the reference proofs name library lemmas by global id.  %s' % (
            bk.k.num_globals(), ', only %d problems' % len(only) if only else '', a.data,
            'Build a dataset for these problems with dataset.py OUT --only ... or drop --only.' if getattr(a, 'only', '') else 'Rebuild the dataset with dataset.py.'))
    ctx = D.DataContext(bk.w)
    eng = G.Engine(bk.w, kinds=set(G.Kind) if kinds is None else kinds, strict=strict)
    print('world: %d globals, %d problems, %.0f s' % (bk.w.k.num_globals(), len(bk.placed), time.time() - t0), flush=True)
    return bk, ctx, eng


def split_names(bk):
    """{split: sorted names} of the problems in the world"""
    out = {'train': [], 'dev': [], 'test': []}
    for n in sorted(bk.placed): out[BS.split_of(n)].append(n)
    return out


def forbidden_digests(ddir):
    """the statements a lemma the agent proves must not say: every problem of the bank (any split) and the sub-theorems of the dev and test problems"""
    out = set()
    bank = json.load(open(os.path.join(ddir, 'bank.json')))
    for p in bank['problems']: out.add(p['statement_digest'])
    path = os.path.join(ddir, 'subproblems.json')
    if os.path.exists(path):
        for e in json.load(open(path))['examples']:
            if e['split'] != 'train': out.add(e['statement_digest'])
    return out


def make_library(bk, eng, ctx, ddir, **kw):
    """an empty ml_lemmas.LemmaLibrary for this world: its `forbid` and `held_out` are the dataset's (see `forbidden_digests`); keywords: those of LemmaLibrary"""
    import ml_lemmas as L
    names = split_names(bk)
    return L.LemmaLibrary(bk, eng, ctx, forbid=forbidden_digests(ddir), held_out=names['dev'] + names['test'], **kw)


def load_library(a, bk, eng, ctx, ddir, nets=(), **kw):
    """the library of an earlier run (--lemmas), installed in this world and synchronised with the networks `nets`; None when no file was given"""
    path = getattr(a, 'lemmas', '')
    if not path: return None
    lib = make_library(bk, eng, ctx, ddir, **kw)
    n = lib.load(path)
    lib.sync(nets=nets)
    print('lemmas: %d installed from %s (block %s)' % (n, path, lib.block_digest()), flush=True)
    return lib
