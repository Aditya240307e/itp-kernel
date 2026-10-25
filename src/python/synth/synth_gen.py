"""synth_gen.py -- the synthetic theorem generator as a reproducible process: configuration, a theorem for every index, records, shards.

    cfg = Config(seed=1)                                   # everything that decides what is generated (and nothing else): `cfg.digest()` names it
    gen = Generator(cfg)                                   # builds the library world once (a few seconds), then makes theorems
    rec = gen.record(index)                                # the record of index i, or raises Reject(reason): a PURE FUNCTION of (cfg, i) -- the same record whatever else was generated before
    run_range(cfg, a, b, path)                             # the records of the indices a..b-1, written to a shard (jsonl.gz) with a small meta file; what the multi-process driver calls

THE INDEX IS THE IDENTITY.  Every theorem is made from `random.Random("<seed>/<index>")`: the plan (family, depth, number of variables), the root lemma, every random term. Nothing carries over from one index to
the next, so a shard can be remade alone, a record can be checked by regenerating it, and the driver can cut the index space into ranges and hand them to processes in any order. (The kernel and the engine do
keep caches across theorems; the gate S48 checks that they never change a result.)  An index that yields no theorem (`Reject`) is simply absent from the data -- the manifest says how many were and why.

A RECORD is a JSON-able dict with everything a user needs and nothing they have to trust: the statement (printed, readable, and as a code of the typed term builder, the language an agent states terms in),
the proof (portable actions: what `gym_acts.replay_portable` reads), and LABELS (family, root lemma and its area, the lemmas the proof uses, depth, hypotheses, the size, what the decision procedures
do with it, what the evaluator found out). `synth_verify` replays every record from this data alone, in a fresh process, and has the independent checker look at the proofs.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import collections, gzip, hashlib, json, os, random, time, traceback
import gym_env as G
import gym_acts as A
import synth_pool as SP
import synth_tree as ST
import synth_check as SC
from synth_tree import Reject, Plan

FORMAT = 1
HERE = os.path.dirname(os.path.abspath(__file__))


def generator_digest():
    """a hash of the generator's source (all the synth_*.py files except the command line, the merge, the card and the verification, which only read data): the data say which code made them"""
    h = hashlib.sha256()
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith('synth_') and fn.endswith('.py') and fn not in ('synth_verify.py', 'synth_cli.py', 'synth_merge.py', 'synth_card.py'):
            h.update(fn.encode()); h.update(open(os.path.join(HERE, fn), 'rb').read())
    return h.hexdigest()[:16]


DEFAULT_AREA_WEIGHTS = {'existence': 0.35, 'order': 1.5, 'equality': 1.5}      # (set so that the kept theorems are spread over the areas: the areas differ a lot in how often a draw becomes a theorem)


class Config:
    """what decides the data. Add a field and the digest changes; leave it out of `fields` and it does not decide anything."""
    fields = ('version', 'seed', 'limit_name', 'domains', 'families', 'depth_weights', 'nvars_weights', 'area_weights', 'closer_keep', 'max_hyps', 'min_sat', 'max_kids')
    def __init__(self, seed=1, limit_name='imo1959_p1', domains=('logic', 'Z'), families=None, depth_weights=None, nvars_weights=None, area_weights=None, closer_keep=0.2,
                 max_hyps=5, min_sat=5, max_kids=4):
        self.version = FORMAT; self.seed = seed; self.limit_name = limit_name; self.domains = tuple(domains)
        self.families = dict(families or {'backward_tree': 1.0})
        self.depth_weights = dict(depth_weights or {2: 0.10, 3: 0.20, 4: 0.22, 5: 0.18, 6: 0.15, 7: 0.10, 8: 0.05})
        self.nvars_weights = dict(nvars_weights or {3: 0.35, 4: 0.45, 5: 0.20})
        self.area_weights = dict(DEFAULT_AREA_WEIGHTS if area_weights is None else area_weights)           # {area: weight}; an area that is not listed has weight 1
        self.closer_keep = closer_keep                                  # the share of the theorems that a decision procedure closes at once that is kept (by the theorem's digest)
        self.max_hyps = max_hyps; self.min_sat = min_sat; self.max_kids = max_kids

    def to_json(self):
        d = {f: getattr(self, f) for f in self.fields}
        for f in ('depth_weights', 'nvars_weights'): d[f] = {str(k): v for k, v in sorted(d[f].items())}
        d['domains'] = list(d['domains']); d['area_weights'] = dict(sorted(d['area_weights'].items())); d['families'] = dict(sorted(d['families'].items()))
        return d

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        if d.pop('version', FORMAT) != FORMAT: raise ValueError('a configuration of another format')
        d['depth_weights'] = {int(k): v for k, v in d['depth_weights'].items()}; d['nvars_weights'] = {int(k): v for k, v in d['nvars_weights'].items()}
        return cls(**d)

    def digest(self): return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:16]


def index_rng(cfg, index):
    """the generator of index i: a string seed, which Python hashes with SHA-512 (the same on every machine and version)"""
    return random.Random('%d/%d' % (cfg.seed, index))


def keep_closable(cfg, digest):
    """a theorem that a decision procedure closes at once is kept with probability `closer_keep`, decided by its digest (so the choice is a function of the theorem alone)"""
    return int(digest[:8], 16) % 10_000 < cfg.closer_keep * 10_000


def world_digest_of(world, limit):
    """the digest of the library below `limit` (problems/bank.py has the one definition, by which a dataset names its world).  The data are a function of the configuration, the generator AND this: the pool
    of lemmas the theorems are grown from is the library, so a library that has grown makes other theorems for the same indices (the gate S48 pins the records of a range, and said so when ext_lib was added)."""
    import bank as B
    return B.world_digest(world, limit)


class Generator:
    def __init__(self, cfg, world=None, build=None):
        self.cfg = cfg
        if world is None:
            import imo_lib
            world = (build or imo_lib.build_imo)()
        self.w = world; self.k = world.k
        self.limit = world.gid[cfg.limit_name]
        self.pool = SP.Pool(world, self.limit, domains=cfg.domains)
        self.growers = {}
        self.checker = None
        self.world_digest = world_digest_of(world, self.limit)

    def grower(self, domain='Z'):
        g = self.growers.get(domain)
        if g is None: g = self.growers[domain] = ST.Grower(self.w, self.pool, self.limit, domain=domain)
        return g

    def plan(self, index):
        """(rng, family, Plan) of an index"""
        cfg = self.cfg; rng = index_rng(cfg, index)
        fams = sorted(cfg.families); family = rng.choices(fams, weights=[cfg.families[f] for f in fams])[0]
        ds = sorted(cfg.depth_weights); depth = rng.choices(ds, weights=[cfg.depth_weights[d] for d in ds])[0]
        ns = sorted(cfg.nvars_weights); nvars = rng.choices(ns, weights=[cfg.nvars_weights[n] for n in ns])[0]
        plan = Plan(depth=depth, nvars=nvars, max_hyps=cfg.max_hyps, min_sat=cfg.min_sat, max_kids=cfg.max_kids)
        plan.area_weights = cfg.area_weights or None
        return rng, family, plan

    def record(self, index):
        """the record of an index (see the module documentation); raises Reject(reason) when the index yields no theorem, ST.Disagreement when the evaluator and the kernel disagree (never expected)"""
        rng, family, plan = self.plan(index)
        if family != 'backward_tree': raise ValueError('unknown family %r' % family)
        gr = self.grower('Z')
        th = gr.theorem(rng, plan)
        info = th.info
        if info['code'] is None: raise Reject('the builder cannot say the statement')
        if info['solved_by'] and not keep_closable(self.cfg, info['digest']): raise Reject('closed at once by a decision procedure (not kept)')
        w = self.w; k = self.k
        names = [w.gname[g] for g in info['lemmas']]
        root_lemma = names[0] if names else None
        rec = collections.OrderedDict()
        rec['id'] = 's%d-%07d' % (self.cfg.seed, index); rec['index'] = index; rec['seed'] = self.cfg.seed; rec['family'] = 'backward_tree'; rec['domain'] = 'Z'
        rec['statement'] = info['text']; rec['digest'] = info['digest']; rec['pretty'] = info['pretty']
        rec['code'] = [list(t) for t in info['code']]
        rec['steps'] = json.loads(json.dumps(th.steps))
        rec['root_lemma'] = root_lemma; rec['area'] = self.pool.by_name[root_lemma].area if root_lemma in self.pool.by_name else None
        rec['lemmas'] = names
        rec['depth'] = info['depth']; rec['n_steps'] = info['n_steps']; rec['n_lemmas'] = info['n_lemmas']; rec['n_hyps'] = info['n_hyps']; rec['n_vars'] = info['n_vars']
        rec['closers'] = info['closers']; rec['size'] = info['size']; rec['solved_by'] = info['solved_by']
        sem = info['semantic']
        rec['semantic'] = {'sat_rate': sem['sat_rate'], 'sat_distinct': sem['sat_distinct'], 'exact': sem['exact'], 'n_needed': sem['n_needed'], 'valid': sem['valid']}
        rec['model'] = info['model']; rec['plan'] = {'depth': plan.depth, 'nvars': plan.nvars}
        recipe = sorted(set(names) | {'#' + c for c in info['closers']})
        rec['recipe'] = hashlib.sha256('|'.join(recipe).encode()).hexdigest()[:16]
        return rec

    def self_check(self, rec):
        """the record replays from its own data in a fresh engine (a new Engine object on the same kernel): statement rebuilt from the code, proof replayed and kernel-checked. Raises Reject if not."""
        import gym_tree as GT
        if self.checker is None: self.checker = G.Engine(self.w, kinds=set(G.Kind), strict=True)
        eng = self.checker
        try:
            stmt = GT.statement_from_code(eng, rec['code'], self.limit)
            if self.w.show(stmt) != rec['statement']: raise Reject('self-check: the code says another statement')
            A.replay_portable(eng, stmt, rec['steps'], premises=self.limit)
        except (A.ActionError, G.AuthorError, RuntimeError) as e:
            raise Reject('self-check: ' + str(e)[:80])


def run_range(cfg, a, b, path, world=None, log=None):
    """generate the indices a..b-1 and write the records to `path` (jsonl.gz) and a meta file `path + '.meta.json'`; returns the meta.
    A shard that exists is not remade -- if it is the shard asked for: made by this configuration, this generator and this library (`world`, the digest of the library below the limit), over this range.
    A shard of another configuration, generator or library is an error (shards of two generators, or of two libraries, must never be mixed in one dataset: use another directory); one of another range
    (the shard size changed) is made again."""
    meta_path = path + '.meta.json'
    t0 = time.time(); gen = None
    if os.path.exists(path) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta.get('config') != cfg.digest() or meta.get('generator') != generator_digest():
            raise RuntimeError('%s was made with configuration %s and generator %s; this is configuration %s and generator %s -- shards of different generators are not mixed: use another --out' %
                               (path, meta.get('config'), meta.get('generator'), cfg.digest(), generator_digest()))
        gen = Generator(cfg, world=world)                                    # (the library is needed to say which library the shard is for)
        if meta.get('world') != gen.world_digest:
            raise RuntimeError('%s was made in the library %s; the library now is %s -- shards of two libraries are not mixed (the same index makes another theorem when the pool of lemmas has changed): '
                               'use another --out' % (path, meta.get('world') or '(not recorded: a shard of an older generator)', gen.world_digest))
        if meta.get('range') == [a, b]: return meta
    if gen is None: gen = Generator(cfg, world=world)
    rejects = collections.Counter(); errors = []; disagreements = []; records = []
    for i in range(a, b):
        try:
            rec = gen.record(i)
            gen.self_check(rec)
            records.append(rec)
        except Reject as e:
            rejects[e.reason.split(':')[0]] += 1
        except ST.Disagreement as e:
            disagreements.append({'index': i, 'text': e.text[:300]}); rejects['DISAGREEMENT'] += 1
        except Exception as e:                                               # a bug of the generator: counted, kept, never silent
            errors.append({'index': i, 'error': '%s: %s' % (type(e).__name__, str(e)[:200]), 'trace': traceback.format_exc()[-600:]}); rejects['ERROR'] += 1
        if log and (i - a + 1) % 200 == 0: log('%d..%d: %d records, %.0f s' % (a, i + 1, len(records), time.time() - t0))
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r, separators=(',', ':')) + '\n')
    os.replace(tmp, path)
    meta = {'format': FORMAT, 'config': cfg.digest(), 'generator': generator_digest(), 'world': gen.world_digest, 'range': [a, b], 'records': len(records), 'rejects': dict(rejects), 'errors': errors[:20], 'n_errors': len(errors),
            'disagreements': disagreements, 'seconds': round(time.time() - t0, 1), 'stats': dict(gen.grower('Z').stats)}
    with open(meta_path + '.tmp', 'w') as f: json.dump(meta, f, indent=1)
    os.replace(meta_path + '.tmp', meta_path)
    return meta


def read_shard(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f: yield json.loads(line)
