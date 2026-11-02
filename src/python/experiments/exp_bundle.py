"""exp_bundle.py -- the RESULTS FOLDER: everything a paper is written FROM, saved apart from the paper.

    python3 src/python/experiments/exp_report.py runs/full --out results/full       # exp_report calls write_bundle after the tables and the figures

results/<name>/
    README.md                  what is in the folder, how each file was made, the definitions (read this first)
    MANIFEST.sha256            the SHA-256 of every other file:  cd results/<name> && sha256sum -c MANIFEST.sha256
    report.md                  the whole report in one readable file (warnings first)
    numbers.json, numbers.tex  every number the report states, by name
    tables/<key>.{md,tex,csv}  one file per table
    figures/<name>.{pdf,png,csv}   the csv holds what the figure draws: panel, series, kind, label, x, y, x_low, x_high, y_low, y_high
    per_problem/outcomes.csv   one row per ATTEMPT (method, seed, problem): solved, probes, steps, seconds and why it stopped
    per_problem/by_problem.csv one row per problem: the fraction of seeds that solved it, per method (and whether anyone did)
    per_problem/by_family.csv  one row per (method, split, group of problems): rate with its interval over seeds
    stats/rates.csv            one row per (method, split, subset, budget): mean, interval, the rate and the count of each seed
    stats/comparisons.csv      the paired comparisons of the ablation table: three intervals, p and Holm's p, the smallest difference this benchmark can detect
    stats/benchmark.csv        floor and ceiling of the benchmark, how it ends the attempts that fail, how stable the ranking of the methods is
    provenance/                PROFILE.json, GRAPH.json, runs.csv (every run: commit, code tree, host, seconds), machine.json, repo_facts.json, dataset.json

Nothing is typed in: every file is computed from the rows of the evaluations and the logs of the training runs (exp_collect), so the folder can be thrown away and made again from `runs/`.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, csv, glob, json, math, time, shutil, hashlib, collections
import numpy as np
import exp_stats as ST, exp_collect as EC, exp_tables as TB

NAN = float('nan')
SPLITS = ('dev', 'test')


def cell(x):
    """a CSV cell: empty for None / NaN, a short decimal for a float, 1 / 0 for a bool, JSON for a list or a dict"""
    if x is None: return ''
    if isinstance(x, (bool, np.bool_)): return int(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float, np.floating)): return '' if math.isnan(x) else format(float(x), '.6g')
    if isinstance(x, (list, tuple, dict)): return json.dumps(x, separators=(',', ':'), default=lambda v: None if isinstance(v, float) and math.isnan(v) else float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else str(v))
    return x


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for row in rows: w.writerow([cell(x) for x in row])
    return path


def stop_reason(r, P):
    """why an attempt ended: `solved`; `error` (the solver raised); `budget` (it used all the probes of the evaluation); `time` (the per-problem wall-clock limit, which is checked between probes only, ended it
    first); `gave_up` (the solver stopped on its own with probes and time left)"""
    if r.get('solved'): return 'solved'
    if r.get('error'): return 'error'
    maxp = P.get('eval_probes'); tl = P.get('eval_time_limit')
    if maxp and r.get('probes', 0) >= maxp: return 'budget'
    if tl and r.get('seconds', 0) >= tl: return 'time'
    return 'gave_up'


def methods(C, groups=None):
    """[(group, variant, solver)] of the default (prefix) view, baselines first, then the main variants in the order of the paper, then the rest"""
    seen = sorted({(r['group'], r['variant'], r['solver']) for r in C.rows if groups is None or r['group'] in groups})
    def key(m):
        g, v, s = m
        if g == 'baselines': return (0, s, '', '')
        if g == 'main' and v in TB.MAIN_ORDER: return (1, '%02d' % TB.MAIN_ORDER.index(v), s, '')
        return (2, g, v, s)
    return sorted(seen, key=key)


def mname(m): return '%s/%s/%s' % m


def seed_mean(M):
    """per problem: the mean over the replicates that have a result (NaN where none has)"""
    ok = ~np.isnan(M.data); n = ok.sum(axis=0)
    return np.where(n > 0, np.where(ok, M.data, 0.0).sum(axis=0) / np.maximum(n, 1), NAN)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ per problem
def outcomes(C, P, out):
    B = P.get('headline')
    header = ['group', 'variant', 'solver', 'regime', 'seed', 'name', 'split', 'level', 'family', 'origin', 'source', 'proved', 'solved', 'solved_at_headline', 'probes', 'decisions', 'steps', 'seconds', 'refused', 'stop', 'error']
    rows = []
    for r in sorted(C.all_rows, key=lambda r: (r['group'], r['variant'], r['solver'], r.get('regime', 'prefix'), int(r['seed']), r['name'])):
        solved = bool(r.get('solved'))
        rows.append([r['group'], r['variant'], r['solver'], r.get('regime', 'prefix'), int(r['seed']), r['name'], r.get('split'), r.get('level'), r.get('family'), r.get('origin'), r.get('source'), bool(r.get('proved', True)),
                     solved, solved and (B is None or r.get('probes', 0) <= B), r.get('probes'), r.get('decisions'), r.get('steps'), r.get('seconds'), r.get('refused'), stop_reason(r, P), (r.get('error') or '')[:200]])
    return write_csv(os.path.join(out, 'per_problem', 'outcomes.csv'), header, rows)


def problem_info(C, name):
    """(level, source class, proved, number of helper lemmas, length of the reference proof) of a problem"""
    if not hasattr(C, '_bank_ix'): C._bank_ix = {p['name']: p for p in C.bank}
    i = C.info.get(name, {}); d = C._bank_ix.get(name, {}).get('demonstration') or {}
    return i.get('level'), EC.source_class(i.get('source')), bool(i.get('proved', True)), C.helpers.get(name, ''), d.get('length', '')


def by_problem(C, P, out):
    B = P.get('headline'); ms = methods(C); cols = []; fr = {}
    for sp in SPLITS:
        for m in ms:
            M = C.success(m[0], m[1], m[2], sp, B)
            if not M.names: continue
            fr[(sp, m)] = dict(zip(M.names, seed_mean(M)))
    header = ['name', 'split', 'level', 'source_class', 'proved', 'helpers', 'reference_length'] + [mname(m) for m in ms] + ['n_methods', 'n_methods_solving', 'solved_by_any', 'solved_by_all']
    rows = []
    for sp in SPLITS:
        for n in C.names(sp, 'all'):
            vals = [fr.get((sp, m), {}).get(n, NAN) for m in ms]; ok = [v for v in vals if not math.isnan(v)]
            rows.append([n, sp] + list(problem_info(C, n)) + vals + [len(ok), sum(1 for v in ok if v > 0), bool(ok) and any(v > 0 for v in ok), bool(ok) and all(v >= 1.0 for v in ok)])
    return write_csv(os.path.join(out, 'per_problem', 'by_problem.csv'), header, rows)


def family_keys(C, split):
    """{(key type, key): [names]} for the groups of problems of a split: the contest family, the level, whether the bank has a reference proof"""
    groups = collections.OrderedDict()
    for n in C.names(split, 'all'):
        lvl, src, proved, _, _ = problem_info(C, n)
        groups.setdefault(('source', src), []).append(n); groups.setdefault(('level', 'level %s' % lvl), []).append(n); groups.setdefault(('reference', 'with a reference proof' if proved else 'statement only'), []).append(n)
    return groups


def by_family(C, P, out):
    B = P.get('headline'); header = ['group', 'variant', 'solver', 'split', 'key_type', 'key', 'n_problems', 'n_seeds', 'mean', 'ci_low', 'ci_high', 'per_seed']; rows = []
    fam = {sp: sorted(family_keys(C, sp).items(), key=lambda kv: (kv[0][0], str(kv[0][1]))) for sp in SPLITS}
    for m in methods(C):
        for sp in SPLITS:
            M = C.success(m[0], m[1], m[2], sp, B)
            if not M.names: continue
            for (kt, k), names in fam[sp]:
                sub = M.subset(set(names)); r = sub.rates(); r = r[~np.isnan(r)]
                mean, lo, hi = ST.mean_ci(list(r))
                rows.append([m[0], m[1], m[2], sp, kt, k, len(sub.names), len(r), mean, lo, hi, [round(float(x), 6) for x in sub.rates()]])
    return write_csv(os.path.join(out, 'per_problem', 'by_family.csv'), header, rows)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ statistics
def rates(C, P, out):
    B = P.get('headline'); budgets = sorted(set(P.get('eval_budgets') or ()) | ({B} if B else set()))
    header = ['group', 'variant', 'solver', 'split', 'subset', 'budget', 'n_seeds', 'n_problems', 'mean', 'ci_low', 'ci_high', 'ci_kind', 'solved_per_seed', 'problems_per_seed']; rows = []
    for m in methods(C):
        for sp in ('dev', 'test', 'dev+test'):
            for sub in ('all', 'proved', 'unproved', 'free'):
                for b in budgets:
                    M = C.success(m[0], m[1], m[2], sp, b, sub)
                    if not M.names: continue
                    k = [int(np.nansum(M.data[i])) for i in range(len(M.seeds))]; n = [int((~np.isnan(M.data[i])).sum()) for i in range(len(M.seeds))]
                    mean, lo, hi, ns = C.rate_ci(m[0], m[1], m[2], sp, b, sub)
                    rows.append([m[0], m[1], m[2], sp, sub, b, ns, len(M.names), mean, lo, hi, 't over seeds' if ns >= 2 else 'Wilson over problems', k, n])
    return write_csv(os.path.join(out, 'stats', 'rates.csv'), header, rows)


def comparisons(C, P, out):
    """the ablation table's comparisons (the family that Holm's correction is applied to, per split) with the three intervals: over seeds (paired t), over problems (bootstrap) and over both (crossed bootstrap),
    and the smallest true difference of the two methods that a paired analysis over THESE problems detects with 80% power"""
    B = P.get('headline'); header = ['split', 'key', 'comparison', 'a', 'b', 'budget', 'n_seeds', 'n_problems', 'dropped_problems', 'diff_pp', 'seed_low_pp', 'seed_high_pp', 'p', 'p_holm', 'significant_holm', 'problem_low_pp', 'problem_high_pp',
                                    'crossed_low_pp', 'crossed_high_pp', 'mde80_pp']; rows = []
    for sp in SPLITS:
        fam = []
        for key, label, a, b in TB.ABLATIONS:
            ga, gb = TB._find(C, a), TB._find(C, b)
            if ga is None or gb is None: continue
            c = C.compare(ga, a, b, 'sample', sp, B, 'all', group_b=gb)
            if not c['n_problems']: continue
            d = c['diff']; pb = c['problem']; cr = c['crossed']; pp = lambda x: 100 * x if x is not None and not math.isnan(x) else NAN
            fam.append([sp, key, label, a, b, B, len(c['seeds']), c['n_problems'], c['dropped'], pp(d['mean']), pp(d['low']), pp(d['high']), d['p'], NAN, None, pp(pb['low']), pp(pb['high']), pp(cr['low']), pp(cr['high']), pp(c['mde80'])])
        adj = ST.holm([r[12] for r in fam])
        for r, q in zip(fam, adj): r[13] = q; r[14] = (q < 0.05) if not math.isnan(q) else None
        rows += fam
    return write_csv(os.path.join(out, 'stats', 'comparisons.csv'), header, rows)


def rank_stability(M, repeats=200, seed=0):
    """M: (methods x problems) solve rates.  The mean Kendall tau between the rankings of the methods on two random halves of the problems: 1 means the benchmark orders the methods the same way whichever half of its
    problems it is given; 0, that the order is noise"""
    from scipy import stats
    k, n = M.shape
    if k < 3 or n < 8: return NAN
    rng = np.random.default_rng(seed); taus = []
    for _ in range(repeats):
        perm = rng.permutation(n); a, b = perm[:n // 2], perm[n // 2:2 * (n // 2)]
        ra, rb = M[:, a].mean(axis=1), M[:, b].mean(axis=1)
        if np.ptp(ra) == 0 or np.ptp(rb) == 0: continue
        t = stats.kendalltau(ra, rb).statistic
        if not math.isnan(t): taus.append(t)
    return float(np.mean(taus)) if taus else NAN


def benchmark(C, P, out):
    """what the benchmark can say: the problems nobody solves (floor) and everybody solves (ceiling) at the headline budget, how the failed attempts ended (budget / time / gave up), the stability of the ranking"""
    B = P.get('headline'); header = ['split', 'n_problems', 'n_methods', 'floor_none_solve', 'ceiling_all_solve', 'informative', 'floor_fraction', 'informative_fraction', 'failed_attempts', 'failed_by_budget', 'failed_by_time', 'failed_gave_up', 'failed_error',
                                    'rank_stability_tau']; rows = []
    for sp in SPLITS + ('dev+test',):
        ms = methods(C); names = C.names(sp, 'all')
        if not names: continue
        mat = []
        for m in ms:
            M = C.success(m[0], m[1], m[2], sp, B, names=names)
            if np.isnan(M.data).all(): continue
            mat.append(seed_mean(M))
        if not mat: continue
        X = np.array(mat); okp = ~np.isnan(X).any(axis=0); X = X[:, okp]
        floor = int((X.max(axis=0) == 0).sum()) if X.size else 0; ceil = int((X.min(axis=0) >= 1.0).sum()) if X.size else 0; n = X.shape[1]
        stops = collections.Counter(stop_reason(r, P) for r in C.rows if r.get('split') in set(sp.split('+')) and not r.get('solved'))
        failed = sum(stops.values())
        rows.append([sp, n, X.shape[0], floor, ceil, n - floor - ceil, floor / n if n else NAN, (n - floor - ceil) / n if n else NAN, failed, stops['budget'], stops['time'], stops['gave_up'], stops['error'], rank_stability(X)])
    return write_csv(os.path.join(out, 'stats', 'benchmark.csv'), header, rows)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ provenance
def provenance(R, C, P, out):
    d = os.path.join(out, 'provenance'); os.makedirs(d, exist_ok=True); files = []
    for name in ('PROFILE.json', 'GRAPH.json'):
        src = os.path.join(R, name)
        if os.path.exists(src): shutil.copyfile(src, os.path.join(d, name)); files.append(os.path.join(d, name))
    for src_name, name in (('bench.json', 'bench.json'), ('kernel.json', 'kernel_bench.json')):               # (the raw measurements behind the throughput tables and figures)
        src = os.path.join(R, 'bench', src_name)
        if os.path.exists(src): shutil.copyfile(src, os.path.join(d, name)); files.append(os.path.join(d, name))
    header = ['id', 'returncode', 'seconds', 'finished', 'host', 'commit', 'tree', 'tree_finish']; rows = []
    for p in sorted(glob.glob(os.path.join(R, '.status', '*.json'))):
        st = EC.read_json(p) or {}
        rows.append([st.get('id'), st.get('returncode'), st.get('seconds'), st.get('finished'), st.get('host'), st.get('commit'), st.get('tree'), st.get('tree_finish')])
    files.append(write_csv(os.path.join(d, 'runs.csv'), header, rows))
    for name, obj in (('machine.json', C.meta or {}), ('repo_facts.json', C.repo or {}), ('dataset.json', dataset_card(C))):
        with open(os.path.join(d, name), 'w') as f: json.dump(obj, f, indent=1, default=str)
        files.append(os.path.join(d, name))
    return files


def dataset_card(C):
    """the digests that say WHICH bank the numbers are about"""
    out = {'dir': os.path.basename(C.dataset_dir or ''), 'problems': len(C.bank)}
    man = EC.read_json(os.path.join(C.dataset_dir, 'manifest.json')) if C.dataset_dir else None
    if man: out['manifest'] = {k: v for k, v in man.items() if isinstance(v, (str, int, float))}
    out['statement_digests'] = hashlib.sha256('\n'.join(sorted('%s %s' % (p.get('name'), p.get('statement_digest')) for p in C.bank)).encode()).hexdigest() if C.bank else None
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ readme, manifest
def readme(R, C, P, N, warnings, out):
    L = []; add = L.append
    add('# Results folder -- profile `%s`' % P.get('name', '?')); add('')
    add('Made %s from `%s` by `src/python/experiments/exp_report.py`. Nothing in it is typed in: it can be deleted and made again from the runs directory.' % (time.strftime('%Y-%m-%d %H:%M'), os.path.basename(os.path.abspath(R)))); add('')
    add('* seeds: %s; headline budget: %s probes per problem; evaluation budget: %s probes, time limit %s s per problem.' % (P.get('seed_list'), P.get('headline'), P.get('eval_probes'), P.get('eval_time_limit')))
    add('* problems: %s dev, %s test (%s with a reference proof, %s statement-only).' % (N.get('n_dev'), N.get('n_test'), N.get('n_test_proved'), N.get('n_test_unproved')))
    add('* warnings: %d (see `report.md`, section Warnings).' % len(warnings)); add('')
    add('## Files'); add('')
    for path, what in (('report.md', 'the whole report, readable (what was run, warnings, every table, the figures)'), ('numbers.json / numbers.tex', 'every number the report states, by name'),
                       ('tables/<key>.{md,tex,csv}', 'one file per table; the csv holds the cells as the table prints them'), ('figures/<name>.{pdf,png,csv}', 'the csv is what the figure draws: panel, series, kind (line / interval / errorbar / bar / region), label, x, y, x_low, x_high, y_low, y_high'),
                       ('per_problem/outcomes.csv', 'one row per attempt: solved, probes, steps, seconds, `stop` = solved / budget / time / gave_up / error'),
                       ('per_problem/by_problem.csv', 'one row per problem: the fraction of seeds that solved it within the headline budget, per method; `solved_by_any`, `solved_by_all`'),
                       ('per_problem/by_family.csv', 'rates by contest family, level and reference proof, with the interval over seeds'), ('stats/rates.csv', 'every rate of every method, split, subset and budget with the count of each seed'),
                       ('stats/comparisons.csv', 'the ablation comparisons: difference in percentage points with the interval over seeds, over problems and over both, p, Holm-adjusted p, the smallest difference detectable with 80% power'),
                       ('stats/benchmark.csv', 'floor / ceiling of the benchmark, how failed attempts ended, stability of the ranking'), ('provenance/', 'the profile, the run graph, every run (commit, host, seconds), the machine, the repository facts, the dataset digests, the raw benchmark measurements (`bench.json`, `kernel_bench.json`)')):
        add('* `%s` -- %s' % (path, what))
    add('')
    add('## How to read the numbers'); add('')
    add('* A **replicate** is a whole run of the pipeline from its seed. `ci_kind = t over seeds`: the Student-t 95% interval of the per-seed rates; with one replicate the Wilson interval over problems.')
    add('* `comparisons.csv` has three intervals for a difference of two methods: **seed** (paired t over replicates, how much it depends on the seed), **problem** (bootstrap over the problems, how much it depends on WHICH problems the benchmark holds) and **crossed** (resampling both). '
        '`p_holm` is Holm\'s correction over the comparisons of the same split. `mde80_pp` is the smallest true difference that a paired analysis over this benchmark\'s problems detects with 80% power at the 5% level (z * sd of the per-problem differences / sqrt(problems)).')
    add('* `benchmark.csv`: `floor` problems are solved by no method, `ceiling` problems by all of them; a benchmark discriminates only through the rest (`informative`). `rank_stability_tau` is the mean Kendall tau of the ranking of the methods on two random halves of the problems.')
    add('* The tables `bench_*` and figures `fig_throughput`, `fig_breakdown`, `fig_latency` come from `provenance/bench.json`: throughput is the median of several passes over the same problems after a warm-up (the range of the passes is beside it); '
        'the time breakdown and the latency percentiles come from one more, instrumented pass (it finds the same proofs; `ratio_to_plain` in `bench.json` is its cost); memory is the proportional set size, which counts the world once however many workers share it; '
        '`determinism` says whether the rows of the in-process run and of every worker count are identical but for the clock. Timings are those of the machine in `bench_machine` and are only comparable with timings from a quiet machine of the same kind.')
    add('* The tables `arch`, `size`, `laws`, `scratch` and the figures `fig_arch`, `fig_laws`, `fig_scratch` are the studies of the network and of the start (only those that were run are there): other encoders of the goal and other widths with everything else held fixed, '
        'learners that start from random weights.  The exponent of a row of `laws` is that of `loss ~ x^e`, fitted on a log-log line separately for each seed over the range of x that was measured and averaged over seeds (95% t-interval over seeds); it is an empirical exponent '
        'over that range, with no asymptote assumed, and the R-squared column says how well a straight line describes the points.  Held-out losses and action accuracies are those of the best epoch of the pre-training run (`final.json`, `best`).')
    add('* `stop` in `outcomes.csv`: an attempt ends when it is solved, when it has used the evaluation budget of probes, when the wall-clock limit per problem (checked between probes) ends it, or when the solver gives up by itself.')
    add('')
    add('## Definitions'); add('')
    doc = EC.__doc__ or ''
    add(doc.split('Definitions (they are the definitions of the paper):')[1].rstrip() if 'Definitions (they are the definitions of the paper):' in doc else '')
    add('')
    add('## Check and remake'); add('')
    add('    cd %s && sha256sum -c MANIFEST.sha256        # the files are the ones that were written' % os.path.basename(os.path.abspath(out)))
    add('    python3 src/python/experiments/exp_report.py <runs directory> --out <this folder>      # remake it')
    add('')
    with open(os.path.join(out, 'README.md'), 'w') as f: f.write('\n'.join(L))
    return os.path.join(out, 'README.md')


def manifest(out):
    """MANIFEST.sha256: `sha256sum -c` format, every file below `out` except the manifest itself"""
    lines = []
    for root, _, files in os.walk(out):
        for fn in files:
            p = os.path.join(root, fn); rel = os.path.relpath(p, out).replace(os.sep, '/')
            if rel == 'MANIFEST.sha256': continue
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
            lines.append('%s  %s' % (h.hexdigest(), rel))
    lines.sort(key=lambda s: s.split('  ', 1)[1])
    with open(os.path.join(out, 'MANIFEST.sha256'), 'w') as f: f.write('\n'.join(lines) + '\n')
    return len(lines)


def verify(out):
    """problems with a results folder, as a list of strings (empty: it is what was written): a file of the manifest that is missing or has another digest, a file that the manifest does not list"""
    path = os.path.join(out, 'MANIFEST.sha256')
    if not os.path.exists(path): return ['%s has no MANIFEST.sha256' % out]
    listed = {}
    for line in open(path).read().split('\n'):
        if line.strip():
            h, _, rel = line.partition('  '); listed[rel] = h
    bad = []
    for rel, h in sorted(listed.items()):
        p = os.path.join(out, rel)
        if not os.path.exists(p): bad.append('missing: ' + rel); continue
        d = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''): d.update(chunk)
        if d.hexdigest() != h: bad.append('changed: ' + rel)
    for root, _, files in os.walk(out):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), out).replace(os.sep, '/')
            if rel != 'MANIFEST.sha256' and rel not in listed: bad.append('not in the manifest: ' + rel)
    return bad


def write_bundle(R, C, P, N, warnings, out):
    """the files this module adds to a report directory (per problem, statistics, provenance, README, manifest); returns the list of paths"""
    files = []
    if C.all_rows:
        files += [outcomes(C, P, out), by_problem(C, P, out), by_family(C, P, out), rates(C, P, out), comparisons(C, P, out), benchmark(C, P, out)]
    files += provenance(R, C, P, out)
    files.append(readme(R, C, P, N, warnings, out))
    return files


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='check a results folder against its manifest')
    ap.add_argument('--verify', metavar='DIR', required=True)
    a = ap.parse_args(argv); bad = verify(a.verify)
    for b in bad[:50]: print('  ' + b)
    print('%s: %s' % (a.verify, 'ok (every file is the one that was written)' if not bad else '%d problem(s)' % len(bad)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
