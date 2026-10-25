"""exp_report.py -- turn a runs directory into the paper's numbers: tables, figures, a Markdown report and LaTeX macros.

    python3 src/python/experiments/exp_report.py runs/full                       # writes runs/full/report/
    python3 src/python/experiments/exp_report.py runs/full --out paper/generated  # somewhere else (the paper's \\input directory)

Output directory:
    report.md        everything, readable: what was run, warnings, every table, the figures, the statement-only problems the agents solved
    tables/*.md|tex  one file per table (LaTeX with booktabs, \\input-able)
    figures/*.pdf|png
    numbers.json     every number the macros hold, with its name
    numbers.tex      \\newcommand{\\...}{...} for the numbers cited in the text of the paper

Nothing here is typed in: every cell comes from rows.jsonl of an evaluation or log.jsonl/final.json of a training run.  A runs directory with missing pieces still gives a report; the gaps are listed under
"Warnings" (missing evaluation rows, errors inside solvers, runs made with different commits).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, glob, time, math, argparse, collections
import numpy as np
import exp_collect as EC, exp_tables as TB, exp_stats as ST

DIGITS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']


def texname(key):
    """a LaTeX control-word name for a key: letters only (digits spelled out)"""
    out = []
    for part in key.replace('.', '_').replace(':', '_').replace('=', '_').replace('/', '_').split('_'):
        if not part: continue
        out.append(''.join(DIGITS[int(ch)] if ch.isdigit() else ch for ch in part).capitalize() if not part[0].isdigit() else ''.join(DIGITS[int(ch)] if ch.isdigit() else ch for ch in part))
    return ''.join(out)


def status_summary(R):
    """({commit: count}, done, failed, seconds) from the runner's status markers"""
    commits = collections.Counter(); done = failed = 0; secs = 0.0
    for p in glob.glob(os.path.join(R, '.status', '*.json')):
        try: st = json.load(open(p))
        except (OSError, ValueError): continue
        if st.get('returncode') == 0: done += 1; secs += st.get('seconds', 0); commits[st.get('commit') or '?'] += 1
        else: failed += 1
    return commits, done, failed, secs


def completeness(C):
    """warnings about evaluation rows that are missing (a replicate that did not finish a task) and about solvers that raised"""
    W = []
    keys = sorted({(r['group'], r['variant'], r['solver']) for r in C.rows})
    for g, v, s in keys:
        M = C.success(g, v, s, 'dev+test', None, 'all')
        if M.n_missing: W.append('%s / %s / %s: %d of %d (seed, problem) results are missing' % (g, v, s, M.n_missing, M.data.size))
    errs = collections.Counter((r['variant'], r['solver']) for r in getattr(C, 'all_rows', C.rows) if r.get('error'))
    for (v, s), n in errs.items(): W.append('%s / %s: %d rows carry a solver error (counted as unsolved)' % (v, s, n))
    return W


def numbers(C, T, P):
    N = collections.OrderedDict()
    B = P.get('headline')
    N['headline_budget'] = B; N['n_dev'] = len(C.names('dev', 'all')); N['n_test'] = len(C.names('test', 'all')); N['n_test_proved'] = len(C.names('test', 'proved')); N['n_test_unproved'] = len(C.names('test', 'unproved'))
    N['n_seeds'] = len(P.get('seed_list') or [])
    for k in ('kernel_lines', 'bindings_lines', 'checker_lines', 'python_modules', 'python_lines', 'test_lines', 'gates', 'mutants', 'mutants_equivalent', 'library_lemmas', 'subtheorems'):
        if (C.repo or {}).get(k) is not None: N['repo_' + k] = C.repo[k]
    if C.bank:
        N['bank_problems'] = len(C.bank); N['bank_proved'] = sum(1 for p in C.bank if 'demonstration' in p); N['bank_statement_only'] = N['bank_problems'] - N['bank_proved']
        for sp in ('train', 'dev', 'test'): N['bank_' + sp] = sum(1 for p in C.bank if p.get('split') == sp)
        N['bank_with_helpers'] = sum(1 for p in C.bank if p.get('helpers'))                       # (problems whose prefix and strict regimes differ)
        for k in ('IMO', 'ISL', 'AIME', 'AMC', 'APMO'): N['bank_' + k.lower()] = sum(1 for p in C.bank if EC.source_class(p.get('source')) == k)
    for group, variants in (('main', TB.MAIN_ORDER), ('baselines', ['baselines'])):
        for v in variants:
            if v not in C.variants(group): continue
            for s in C.solvers(group, v):
                for split in ('dev', 'test'):
                    m, lo, hi, n = C.rate_ci(group, v, s, split, B, 'all')
                    if math.isnan(m): continue
                    key = '%s_%s' % (split, s if group == 'baselines' else v)
                    N[key] = TB.pct(m, lo, hi, n); N[key + '_mean'] = round(100 * m, 2)
    for key, label, a, b in TB.ABLATIONS:
        ga, gb = TB._find(C, a), TB._find(C, b)
        if ga is None or gb is None: continue
        c = C.compare(ga, a, b, 'sample', 'test', B, 'all', group_b=gb)
        if c['diff']['n']: N['abl_' + key] = TB.pp(c['diff']); N['abl_' + key + '_p'] = TB.pval(c['diff']['p'])
    st = (C.synth or {}).get('stats')
    if st and st.get('total'):
        N['gen_total'] = st['total']; N['gen_skeletons'] = st['classes']['skeleton']['observed']; N['gen_skeleton_singletons'] = st['classes']['skeleton']['singletons']
        N['gen_chao'] = round(st['classes']['skeleton']['chao1']); N['gen_lemmas'] = st['lemmas']['distinct_applied']; N['gen_kinds'] = st['kinds']['distinct_kinds']
        N['gen_mean_steps'] = round(st['proofs']['mean_steps'], 2)
    pm = [f.get('n_params') for f in C.finals.values() if f.get('n_params')] or [((C.bench or {}).get('throughput') or {}).get('world', {}).get('params')]
    if pm and pm[0]: N['model_params_m'] = round(pm[0] / 1e6, 2)
    chk = sum(v['checked'] for v in C.recheck.values()); acc = sum(v['accepted'] for v in C.recheck.values())
    N['recheck_checked'] = chk; N['recheck_accepted'] = acc
    b = (C.bench or {}).get('throughput')
    if b and b.get('solvers'):
        for s, v in b['solvers'].items():
            if v.get('probes_per_s'): N['bench_probes_per_s_' + s] = round(v['probes_per_s'], 1)
    return N


def write_numbers(N, out):
    with open(os.path.join(out, 'numbers.json'), 'w') as f: json.dump(N, f, indent=1)
    L = ['% generated by exp_report.py -- do not edit']
    for k, v in N.items():
        s = str(v).replace('±', r'$\pm$').replace('%', r'\%').replace('_', r'\_')
        L.append(r'\newcommand{\%s}{%s}' % ('num' + texname(k), s))
    with open(os.path.join(out, 'numbers.tex'), 'w') as f: f.write('\n'.join(L) + '\n')


def md_report(R, C, T, P, figs, warnings, N):
    commits, done, failed, secs = status_summary(R)
    L = []; add = L.append
    add('# Experiment report -- profile `%s`' % P.get('name', '?')); add('')
    add('Generated %s from `%s`.' % (time.strftime('%Y-%m-%d %H:%M'), R)); add('')
    add('## What was run'); add('')
    add('* %d runs finished (%d failed), %.1f hours of wall-clock summed over runs; commits of the code that made them: %s.' % (done, failed, secs / 3600.0, ', '.join('%s (%d runs)' % (c[:10], n) for c, n in commits.items()) or 'unknown'))
    add('* seeds: %s; problems: %d dev, %d test (%d with a reference proof, %d statement-only); budget of the tables: %s probes per problem; time limit per problem %s s.' % (
        P.get('seed_list'), N.get('n_dev', 0), N.get('n_test', 0), N.get('n_test_proved', 0), N.get('n_test_unproved', 0), P.get('headline'), P.get('eval_time_limit')))
    add('* synthetic theorems: %s requested for pre-training; sizes %s.' % (max(P.get('synth_sizes', [0])), P.get('synth_sizes')))
    m = C.meta
    if m: add('* machine of the first evaluation: %s, %s cores, %s GB, torch %s, python %s; GPU: %s.' % (m.get('cpu'), m.get('cores'), m.get('memory_gb'), m.get('torch'), m.get('python'), m.get('gpu') or 'none'))
    add('')
    if len(commits) > 1: warnings = ['the runs were made with %d different commits (%s): the matrix mixes code versions' % (len(commits), ', '.join(c[:10] for c in commits))] + warnings
    add('## Warnings'); add('')
    if warnings:
        for w in warnings[:60]: add('* ' + w)
        if len(warnings) > 60: add('* ... and %d more' % (len(warnings) - 60))
    else: add('None: every evaluation row is present and no solver raised.')
    add('')
    order = [('system', 'The artefact', None), ('bank', 'The benchmark', None), ('main', 'Main results', 'fig_budget'), ('strict', 'Hiding the helper lemmas of a problem', None), ('ablations', 'Ablations', 'fig_ablations'), ('scaling', 'Scaling with the amount of synthetic data', 'fig_scaling'), ('solvers', 'Provers on the same networks', 'fig_solvers'),
             ('mw', 'Manager-worker agent', None), ('levels', 'By difficulty and by contest', 'fig_levels'), ('generator', 'The synthetic theorem set', 'fig_generator'), ('library', 'Bootstrapped library', 'fig_library'),
             ('cost', 'Training cost', 'fig_learning'), ('bench_solvers', 'Throughput', 'fig_throughput'), ('bench_workers', None, None), ('bench_world', None, None), ('bench_kernel', None, None), ('soundness', 'Soundness', None)]
    shown_figs = set()
    for key, title, fig in order:
        if key in T:
            if title: add('## ' + title); add('')
            add(T[key].md())
        if fig and fig in figs and fig not in shown_figs:
            add('![%s](figures/%s.png)' % (fig, fig)); add(''); shown_figs.add(fig)
    for fig in figs:
        if fig not in shown_figs: add('![%s](figures/%s.png)' % (fig, fig)); add('')
    d = TB.discoveries(C, P)
    add('## Statement-only problems solved'); add('')
    if d:
        add('Problems of the bank with a formal statement but no reference proof that some run proved (each proof was replayed and kernel-checked):'); add('')
        add('| problem | split | first by | solver | seed | probes | steps |'); add('|---|---|---|---|---|---|---|')
        for row in d: add('| %s |' % ' | '.join(str(x) for x in row))
    else: add('None.')
    add('')
    add('## Definitions'); add('')
    add(EC.__doc__.split('Definitions (they are the definitions of the paper):')[1].split('"""')[0].rstrip() if 'Definitions (they are the definitions of the paper):' in EC.__doc__ else '')
    add('')
    return '\n'.join(L)


def write_problems(C, out):
    """problems.csv: one row per problem of the bank"""
    import csv
    with open(os.path.join(out, 'problems.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['name', 'source', 'year', 'level', 'split', 'family', 'reference_proof', 'reference_length', 'helpers', 'origin', 'verified', 'statement_digest'])
        for p in C.bank:
            d = p.get('demonstration') or {}
            w.writerow([p.get('name'), p.get('source'), p.get('year'), p.get('level'), p.get('split'), p.get('family'), int('demonstration' in p), d.get('length', ''), len(p.get('helpers') or []),
                        p.get('origin', ''), p.get('verified', ''), p.get('statement_digest')])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('runs'); ap.add_argument('--out', default=''); ap.add_argument('--no-figures', action='store_true'); ap.add_argument('--data', default=''); ap.add_argument('--synth', default=''); ap.add_argument('--profile', default='')
    a = ap.parse_args(argv)
    R = a.runs; out = a.out or os.path.join(R, 'report')
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True)
    C = EC.load(R); P = C.profile or {}
    if not P: print('warning: %s has no PROFILE.json (was it made by exp_run.py?)' % R)
    T = TB.all_tables(C)
    for key, t in T.items():
        with open(os.path.join(out, 'tables', key + '.md'), 'w') as f: f.write(t.md())
        with open(os.path.join(out, 'tables', key + '.tex'), 'w') as f: f.write(t.tex())
    figs = {} if a.no_figures else __import__('exp_figures').make_all(C, os.path.join(out, 'figures'))
    warnings = completeness(C) + C.warnings
    N = numbers(C, T, P)
    write_numbers(N, out)
    if C.bank: write_problems(C, out)
    with open(os.path.join(out, 'report.md'), 'w') as f: f.write(md_report(R, C, T, P, figs, warnings, N))
    print('report: %d tables, %d figures, %d numbers, %d warnings -> %s' % (len(T), len(figs), len(N), len(warnings), out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
