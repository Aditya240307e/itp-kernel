"""exp_collect.py -- read a runs directory (exp_run.py) into one object the tables and figures are computed from.

    C = exp_collect.load('runs/full')                     # reads PROFILE.json, GRAPH.json, every evaluation's rows.jsonl, the training logs, the synthetic statistics, the benchmarks
    M = C.success('main', 'ei_her', 'sample', split='test', budget=1000)       # a Matrix: replicates (training seeds) x problems, 1.0 solved within the budget, 0.0 not
    M.rates()        # per replicate: the fraction of problems solved  (NaN-aware)
    C.curve('main', 'ei_her', 'sample', 'test', budgets)                       # solved fraction against the budget of probes, mean and interval over replicates

Definitions (they are the definitions of the paper):

  task        a problem of the bank in a regime (ml_protocol; the experiments use the prefix regime): the formal statement, the premises of the regime.
  solved      the solver returned a proof that was REPLAYED in the engine and CHECKED BY THE KERNEL (and, for the runs with --recheck, accepted by the independently written checker).
  within B    solved with at most B probes (a probe is one try of an action on a goal by the engine, whoever asks).  A solver is run once at the largest budget; the budget only truncates its run, so
              "solved within B" is read off its rows for every B (exact for sample, ida, mcts, mw and blind; a run stopped by the per-problem time limit counts as unsolved at every larger budget too).
  replicate   a whole run of the pipeline from its seed (pre-training, fine-tuning, reinforcement learning, evaluation).  Intervals over replicates say how much a result depends on the seed;
              intervals over problems (exp_stats.paired_problem_bootstrap) say how much it depends on WHICH problems the benchmark holds.
  subsets     `all` (every problem of the split), `proved` (those with a reference proof in the bank: they are solvable with the library, a statement-only problem may or may not be), `unproved` (statement
              only) and `free` (problems with no helper lemma of their own).
  regimes     `prefix` (every global before the problem is on the menu, the problem's own helper lemmas included: the default of all tables) and `strict` (the helper lemmas hidden; a different task only for the
              problems whose reference proof proves its lemmas inline).  A problem without helper lemmas is one task in both.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, glob, collections, math
import numpy as np
import exp_stats as ST, exp_repo as RP

NAN = float('nan')


def read_json(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError): return default


def read_jsonl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: out.append(json.loads(line))
                except ValueError: pass
    except OSError: pass
    return out


def resolve(path, near=None):
    """an existing path for `path` (relative to the working directory, to the repository root, or to the directory `near` belongs to), else ''"""
    if os.path.isabs(path): return path if os.path.exists(path) else ''
    for base in ('', itp_paths.ROOT, os.path.dirname(os.path.abspath(near)) if near else ''):
        cand = os.path.join(base, path) if base else path
        if os.path.exists(cand): return os.path.abspath(cand)
    return ''


def source_class(source):
    """the contest family of a source string: IMO, ISL, AIME, AMC, APMO, folklore or other (the dataset card's classes)"""
    s = source or ''
    for pre in ('IMO', 'ISL', 'AIME', 'AMC', 'APMO', 'folklore'):
        if s.startswith(pre): return pre
    if 'in the manner of the AMC' in s: return 'AMC'               # (the 72 problems written for the bank: `original, in the manner of the AMC`)
    return 'other'



def pre_converged(fin):
    """True if a pre-training run stopped because its dev loss had not improved for `patience` epochs, False if it ran out of epochs sooner than that (it was still improving), None if it cannot be told (early stopping off, or no history)"""
    hist = fin.get('history') or []; patience = int((fin.get('args') or {}).get('patience') or 0); best = fin.get('best_epoch')
    if not hist or best is None or patience <= 0: return None
    return len(hist) - 1 - int(best) >= patience

class Matrix:
    """replicates x problems; `seeds` and `names` label the axes; entries 1.0 / 0.0 / NaN (no row)"""
    def __init__(self, seeds, names, data):
        self.seeds = list(seeds); self.names = list(names); self.data = np.asarray(data, dtype=np.float64).reshape(len(self.seeds), len(self.names))
    @property
    def n_missing(self): return int(np.isnan(self.data).sum())
    def rates(self):
        if self.data.shape[1] == 0: return np.full(len(self.seeds), NAN)
        return np.nanmean(np.where(np.isnan(self.data), NAN, self.data), axis=1) if self.data.size and not np.isnan(self.data).all() else np.full(len(self.seeds), NAN)
    def subset(self, keep):
        idx = [i for i, n in enumerate(self.names) if n in keep]
        return Matrix(self.seeds, [self.names[i] for i in idx], self.data[:, idx])


class Collected:
    def __init__(self, R):
        self.R = R
        self.profile = read_json(os.path.join(R, 'PROFILE.json'), {})
        self.graph = read_json(os.path.join(R, 'GRAPH.json'), {})
        self.rows = []; self.eval_runs = {}; self.recheck = {}; self.meta = {}
        self.logs = {}; self.finals = {}; self.synth = {}; self.bench = {}; self.warnings = []
        self.info = {}                                                         # name -> {split, level, origin, source, proved} of the problems seen
        self._load()

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ loading
    def _load(self):
        R = self.R
        for rows_path in sorted(glob.glob(os.path.join(R, 'eval', '**', 'rows.jsonl'), recursive=True)):
            rd = os.path.dirname(rows_path)
            rid = os.path.relpath(rd, R).replace(os.sep, '/')
            parts = rid.split('/')                                              # eval/<group>[/<variant>[/s<k>]]
            group = parts[1]
            rows = read_jsonl(rows_path)
            self.eval_runs[rid] = {'group': group, 'rows': len(rows), 'meta': read_json(os.path.join(rd, 'meta.json'), {})}
            rc = read_json(os.path.join(rd, 'recheck.json'))
            if rc: self.recheck[rid] = {'checked': rc.get('checked', 0), 'accepted': rc.get('accepted', 0)}
            for r in rows:
                r = dict(r); r['group'] = group; r['run'] = rid
                m = r.get('method', '')
                r['variant'] = r.get('model') or ('baselines' if group == 'baselines' else m.split('+')[0])
                r['solver'] = r.get('spec') or r.get('solver') or (m.split('+', 1)[1] if '+' in m else m)
                self.rows.append(r)
                nm = r.get('name')
                if nm and nm not in self.info: self.info[nm] = {'split': r.get('split'), 'level': r.get('level'), 'origin': r.get('origin'), 'source': r.get('source'), 'proved': bool(r.get('proved', True)), 'family': r.get('family')}
        for stage in ('pre', 'ft', 'rl'):
            for rd in sorted(glob.glob(os.path.join(R, stage, '*', 's*'))):
                rid = os.path.relpath(rd, R).replace(os.sep, '/')
                lg = read_jsonl(os.path.join(rd, 'log.jsonl'))
                if lg: self.logs[rid] = lg
                fin = read_json(os.path.join(rd, 'final.json'))
                if fin: self.finals[rid] = fin
                lem = read_json(os.path.join(rd, 'lemmas.json'))
                if lem: self.finals.setdefault(rid, {})['lemma_file'] = {'records': len(lem.get('records', [])), 'block_digest': lem.get('block_digest'), 'rejected': lem.get('rejected', {}),
                                                                            'lengths': [r.get('length') for r in lem.get('records', [])], 'origins': [(r.get('origin') or {}).get('kind') for r in lem.get('records', [])]}
        ddir = resolve((self.profile or {}).get('dataset') or os.path.join(R, 'data'), R)
        self.dataset_dir = ddir
        self.bank = (read_json(os.path.join(ddir, 'bank.json'), {}) or {}).get('problems') or [] if ddir else []
        self.repo = RP.facts(); self.repo.update(RP.dataset_facts(ddir) if ddir else {})
        sd = os.path.join(R, 'synth')
        self.synth = {'stats': read_json(os.path.join(sd, 'stats.json')), 'manifest': read_json(os.path.join(sd, 'manifest.json')), 'verification': read_json(os.path.join(sd, 'verification.json'))}
        bd = os.path.join(R, 'bench')
        self.bench = {'throughput': read_json(os.path.join(bd, 'bench.json')), 'kernel': read_json(os.path.join(bd, 'kernel.json'))}
        self.all_rows = self.rows
        self.rows = [r for r in self.all_rows if r.get('regime', 'prefix') == 'prefix']          # the default view: every table is about the prefix regime unless it says otherwise (view('strict'))
        self.helpers = {p['name']: len(p.get('helpers') or []) for p in self.bank}
        for n in sorted(set((self.profile or {}).get('synth_sizes') or ()) - {0}):                    # (a pre-training run says how many synthetic theorems it really trained on)
            u = self.synth_used(n)
            if u is not None and u != n: self.warnings.append('pre-training of nominal size %d trained on %d synthetic theorems (the merged set has fewer train theorems than the size asked for)' % (n, u))
        if (self.profile or {}).get('pre_patience'):                                                # (early stopping is on: H1 is about CONVERGED networks, so a size that ran out of epochs while still improving is told)
            for n in sorted(set((self.profile or {}).get('synth_sizes') or ()) - {0}):
                open_ = sorted(rid for rid, fin in self.finals.items() if rid.startswith('pre/n%d/s' % n) and pre_converged(fin) is False)
                if open_: self.warnings.append('pre-training of size %d had not converged when its epochs ran out (%d of its runs: %s): the held-out loss of that size is an upper bound, and so the exponent of H1 over sizes is not a statement about converged networks -- raise `pre_epochs`' % (n, len(open_), ', '.join(open_[:4]) + (' ...' if len(open_) > 4 else '')))
        for rid, e in self.eval_runs.items():
            if e['meta']: self.meta = e['meta']; break
        if not self.meta:
            for fin in self.finals.values():
                pass

    def final_values(self, stage, name, path):
        """{seed: value} of `final.json[path[0]][path[1]]...` for the runs `stage/name/s*` (a held-out metric of the best epoch: ('best', 'synth_test_iid', 'loss'); the parameters: ('n_params',)); runs without the value are left out"""
        out = {}
        for rid, f in self.finals.items():
            parts = rid.split('/')
            if len(parts) != 3 or parts[0] != stage or parts[1] != name or not parts[2][1:].isdigit() or parts[2][0] != 's': continue
            v = f
            for k in path: v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v)): out[int(parts[2][1:])] = float(v)
        return out

    def synth_used(self, n):
        """how many synthetic theorems the pre-training runs of nominal size n trained on (the smallest over the seeds that finished; None: no final.json says)"""
        v = [f.get('synth_theorems') for rid, f in self.finals.items() if rid.startswith('pre/n%d/' % n) and isinstance(f.get('synth_theorems'), int)]
        return min(v) if v else None

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ selection
    def variants(self, group):
        return sorted({r['variant'] for r in self.rows if r['group'] == group})

    def solvers(self, group, variant):
        return sorted({r['solver'] for r in self.rows if r['group'] == group and r['variant'] == variant})

    def seeds(self, group, variant, solver=None):
        return sorted({int(r['seed']) for r in self.rows if r['group'] == group and r['variant'] == variant and (solver is None or r['solver'] == solver)})

    def splits_of(self, split):
        return [s for s in split.split('+')] if isinstance(split, str) else list(split)

    def names(self, split, subset='all', group=None):
        """the problems of the split(s) in the evaluation (those that appear in any row of `group`, or of any group), sorted"""
        sp = set(self.splits_of(split))
        names = {r['name'] for r in self.rows if (group is None or r['group'] == group) and r.get('split') in sp}
        if subset == 'proved': names = {n for n in names if self.info.get(n, {}).get('proved', True)}
        elif subset == 'unproved': names = {n for n in names if not self.info.get(n, {}).get('proved', True)}
        elif subset == 'free': names = {n for n in names if not self.helpers.get(n)}                     # problems with no helper lemma of their own: the prefix and the strict regime are one task
        return sorted(names)

    def view(self, regime):
        """a shallow copy of this object whose rows are those of one regime ('prefix': the default; 'strict': the problem's helper lemmas hidden, only for the problems where that is a different task)"""
        import copy
        v = copy.copy(self)
        v.rows = [r for r in self.all_rows if r.get('regime', 'prefix') == regime]
        return v

    def _rows_of(self, group, variant, solver):
        return [r for r in self.rows if r['group'] == group and r['variant'] == variant and r['solver'] == solver]

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ matrices
    def success(self, group, variant, solver, split='test', budget=None, subset='all', names=None, seeds=None):
        """Matrix of solved-within-`budget` for the replicates of (group, variant, solver) over the problems of the split(s).  Rows are matched by (seed, name); a task with no row is NaN."""
        rows = self._rows_of(group, variant, solver)
        seeds = sorted({int(r['seed']) for r in rows}) if seeds is None else list(seeds)
        names = self.names(split, subset, group) if names is None else list(names)
        ix = {n: j for j, n in enumerate(names)}; sx = {s: i for i, s in enumerate(seeds)}
        M = np.full((len(seeds), len(names)), NAN)
        for r in rows:
            i = sx.get(int(r['seed'])); j = ix.get(r['name'])
            if i is None or j is None: continue
            M[i, j] = 1.0 if (r.get('solved') and (budget is None or r.get('probes', 0) <= budget)) else 0.0
        return Matrix(seeds, names, M)

    def rate_ci(self, group, variant, solver, split='test', budget=None, subset='all'):
        """(mean, low, high, n_replicates) of the solved fraction over replicates; a single replicate gets a Wilson interval over problems"""
        M = self.success(group, variant, solver, split, budget, subset); r = M.rates(); r = r[~np.isnan(r)]
        if len(r) >= 2:
            m, lo, hi = ST.mean_ci(r); return m, lo, hi, len(r)
        if len(r) == 1:
            n = int((~np.isnan(M.data[0])).sum()); k = int(np.nansum(M.data[0])); p, lo, hi = ST.wilson(k, n); return p, lo, hi, 1
        return NAN, NAN, NAN, 0

    def curve(self, group, variant, solver, split, budgets, subset='all'):
        """{'budgets', 'mean', 'low', 'high', 'per_seed'}: the solved fraction within each budget; the interval is over replicates (Wilson over problems for one replicate)"""
        mean = []; low = []; high = []; per = []
        for b in budgets:
            m, lo, hi, n = self.rate_ci(group, variant, solver, split, b, subset)
            mean.append(m); low.append(lo); high.append(hi); per.append(list(self.success(group, variant, solver, split, b, subset).rates()))
        return {'budgets': list(budgets), 'mean': mean, 'low': low, 'high': high, 'per_seed': per}

    def efficiency(self, group, variant, solver, split='test', budget=None, subset='all'):
        """per replicate: (mean probes of the solved problems, mean proof length of the solved problems); lists over replicates (NaN when a replicate solved nothing)"""
        rows = self._rows_of(group, variant, solver); sp = set(self.splits_of(split))
        names = set(self.names(split, subset, group))
        by = collections.defaultdict(lambda: ([], []))
        for r in rows:
            if r.get('name') not in names or r.get('split') not in sp: continue
            if r.get('solved') and (budget is None or r.get('probes', 0) <= budget):
                by[int(r['seed'])][0].append(r['probes'])
                if r.get('steps') is not None: by[int(r['seed'])][1].append(r['steps'])
        seeds = sorted({int(r['seed']) for r in rows})
        pr = [float(np.mean(by[s][0])) if by[s][0] else NAN for s in seeds]; ln = [float(np.mean(by[s][1])) if by[s][1] else NAN for s in seeds]
        return pr, ln

    def seconds_per_problem(self, group, variant, solver, split='test'):
        rows = [r for r in self._rows_of(group, variant, solver) if r.get('split') in set(self.splits_of(split))]
        return float(np.mean([r['seconds'] for r in rows])) if rows else NAN

    def compare(self, group, a, b, solver, split='test', budget=None, subset='all', group_b=None, seed=0):
        """a against b (variants; `group_b`: the group of b if different): {'diff': paired_seed_diff of the rates, 'problem': paired problem bootstrap, 'a','b': rates per replicate}.
        Replicates are paired by seed (the learners that continue from `bc` share its checkpoint per seed)."""
        names = sorted(set(self.names(split, subset, group)) | set(self.names(split, subset, group_b or group)))
        Ma = self.success(group, a, solver, split, budget, subset, names=names); Mb = self.success(group_b or group, b, solver, split, budget, subset, names=names)
        common = sorted(set(Ma.seeds) & set(Mb.seeds))
        A = Ma.data[[Ma.seeds.index(s) for s in common]]; B = Mb.data[[Mb.seeds.index(s) for s in common]]
        ok = ~(np.isnan(A).any(axis=0) | np.isnan(B).any(axis=0))
        A = A[:, ok]; B = B[:, ok]
        ra = A.mean(axis=1) if A.size else np.array([]); rb = B.mean(axis=1) if B.size else np.array([])
        nan3 = {'mean': NAN, 'low': NAN, 'high': NAN}
        return {'seeds': common, 'a': list(ra), 'b': list(rb), 'n_problems': int(ok.sum()), 'dropped': int((~ok).sum()), 'names': [n for n, k in zip(names, ok) if k],
                'diff': ST.paired_seed_diff(ra, rb) if len(common) else {'mean': NAN, 'low': NAN, 'high': NAN, 'p': NAN, 'n': 0},
                'problem': ST.paired_problem_bootstrap(A, B, seed=seed) if A.size and B.size else dict(nan3),
                'crossed': ST.crossed_bootstrap(A, B, seed=seed) if A.size and B.size else dict(nan3),
                'mde80': ST.paired_mde(A, B) if A.size and B.size else NAN}

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ logs
    def log_runs(self, stage, variant):
        """{seed: [log rows]} of the runs `stage/variant/s*`"""
        out = {}
        for rid, lg in self.logs.items():
            parts = rid.split('/')
            if parts[0] == stage and parts[1] == variant: out[int(parts[2][1:])] = lg
        return out

    def log_curve(self, stage, variant, xkey, ykey, normalise=None):
        """aligned learning curves: for the rows that have `ykey`, the value of `xkey` (mean over replicates) and of `ykey` (mean and t-interval over replicates), by the iteration/epoch number.  `normalise`: a key
        of the row whose value divides y (dev_total)."""
        runs = self.log_runs(stage, variant)
        if not runs: return None
        idxkey = 'iter' if stage == 'rl' else 'epoch'
        table = collections.defaultdict(list)
        for seed, lg in runs.items():
            for r in lg:
                if r.get(ykey) is None or r.get(xkey) is None: continue
                y = float(r[ykey]) / (float(r[normalise]) if normalise and r.get(normalise) else 1.0)
                table[r[idxkey]].append((float(r[xkey]), y))
        xs = []; mean = []; low = []; high = []; ns = []
        for k in sorted(table):
            v = table[k]; m, lo, hi = ST.mean_ci([y for _, y in v])
            xs.append(float(np.mean([x for x, _ in v]))); mean.append(m); low.append(lo); high.append(hi); ns.append(len(v))
        return {'index': sorted(table), 'x': xs, 'mean': mean, 'low': low, 'high': high, 'n': ns}


def load(R):
    return Collected(R)
