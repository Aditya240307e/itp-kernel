"""exp_report.py -- turn a runs directory into the paper's numbers: tables, figures, a Markdown report and LaTeX macros.

    python3 src/python/experiments/exp_report.py runs/full                       # writes runs/full/report/
    python3 src/python/experiments/exp_report.py runs/full --out paper/generated  # somewhere else (the paper's \\input directory)

Output directory (the RESULTS FOLDER: everything a paper is written from, saved apart from the paper; exp_bundle.py has the per-problem and statistics files, README.md says what each holds):
    report.md        everything, readable: what was run, warnings, every table, the figures, the statement-only problems the agents solved
    tables/*.md|tex|csv  one file per table (LaTeX with booktabs, \\input-able)
    figures/*.pdf|png|csv  the csv holds the values the figure draws
    numbers.json     every number the macros hold, with its name
    numbers.tex      \\newcommand{\\...}{...} for the numbers cited in the text of the paper
    per_problem/ stats/ provenance/ README.md MANIFEST.sha256     (exp_bundle)

Nothing here is typed in: every cell comes from rows.jsonl of an evaluation or log.jsonl/final.json of a training run.  A runs directory with missing pieces still gives a report; the gaps are listed under
"Warnings" (missing evaluation rows, errors inside solvers, runs made with different versions of the code).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, glob, time, math, argparse, collections
import numpy as np
import exp_collect as EC, exp_tables as TB, exp_stats as ST, exp_bundle as EB

DIGITS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']


def texname(key):
    """a LaTeX control-word name for a key: letters only (digits spelled out)"""
    out = []
    for part in key.replace('.', '_').replace(':', '_').replace('=', '_').replace('/', '_').split('_'):
        if not part: continue
        out.append(''.join(DIGITS[int(ch)] if ch.isdigit() else ch for ch in part).capitalize() if not part[0].isdigit() else ''.join(DIGITS[int(ch)] if ch.isdigit() else ch for ch in part))
    return ''.join(out)


def status_summary(R):
    """({commit: count}, done, failed, seconds, {tree: count}, [runs whose code changed while they ran]) from the runner's status markers; a run that recorded no tree counts under '?'"""
    commits = collections.Counter(); trees = collections.Counter(); moved = []; done = failed = 0; secs = 0.0
    for p in sorted(glob.glob(os.path.join(R, '.status', '*.json'))):
        try: st = json.load(open(p))
        except (OSError, ValueError): continue
        if st.get('returncode') == 0:
            done += 1; secs += st.get('seconds', 0); commits[st.get('commit') or '?'] += 1; trees[st.get('tree') or '?'] += 1
            if st.get('tree') and st.get('tree_finish') and st['tree'] != st['tree_finish']: moved.append(st.get('id') or os.path.basename(p))
        else: failed += 1
    return commits, done, failed, secs, trees, moved


def code_warnings(commits, trees, moved):
    """what the record of the code says about the runs: made with different versions of the CODE (the digest of the tree: a commit that changed only the paper or a document is not another version), or with code that changed
    while they ran.  A runs directory that recorded no trees is judged by its commits, which is stricter than it should be"""
    W = []
    if any(t != '?' for t in trees):
        if len(trees) > 1:
            W.append('the runs were made with %d different versions of the code (tree %s%s): the matrix mixes code versions' % (len(trees), ', '.join(t[:12] for t in sorted(trees)),
                     '; %d run(s) recorded none' % trees['?'] if trees.get('?') else ''))
    elif len(commits) > 1:
        W.append('the runs were made with %d different commits (%s) and recorded no code tree: the matrix may mix code versions' % (len(commits), ', '.join(c[:10] for c in commits)))
    if moved: W.append('the code changed while %d run(s) were running (%s): their results belong to neither version' % (len(moved), ', '.join(moved[:6]) + (' ...' if len(moved) > 6 else '')))
    return W


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


def bench_warnings(C):
    """what the benchmark itself says is wrong with its measurements: a busy machine, rows that differ between worker counts, passes that did not reproduce, a ledger that does not add up, a proof the independent checker refused"""
    b = (C.bench or {}).get('throughput')
    if not b: return []
    W = []; mc = b.get('machine') or {}
    if mc.get('busy_at_start'): W.append('benchmark: the machine was busy when it started (load %s on %s usable cores): its timings measure the other jobs too' % (mc.get('loadavg_start'), mc.get('cores_usable')))
    det = b.get('determinism') or {}
    if det and not det.get('identical'): W.append('benchmark: the rows differ between %s (%d of %d tasks differ)' % (', '.join(det.get('runs', [])), det.get('n_differing', 0), det.get('tasks', 0)))
    for s, v in (b.get('solvers') or {}).items():
        pr = v.get('profile') or {}
        if v.get('stable') is False: W.append('benchmark: %s found different results in repeated passes over the same problems' % s)
        if pr and pr.get('same_outcome') is False: W.append('benchmark: the instrumented pass of %s found different results from the plain pass (its latency columns are not comparable)' % s)
        if pr and pr.get('balanced') is False: W.append('benchmark: the time ledger of %s is unbalanced (a timed call did not return)' % s)
        if pr.get('hooks_skipped'): W.append('benchmark: %s was profiled without %s (the code has no such method)' % (s, ', '.join(str(h) for h in pr['hooks_skipped'])))
    ck = b.get('check') or {}
    if ck.get('replay_failed'): W.append('benchmark: %d of %d reference proofs did not replay' % (ck['replay_failed'], ck.get('proofs', 0)))
    if ck.get('proofs') and ck.get('independent_accepted') is not None and ck['independent_accepted'] < ck['proofs'] - (ck.get('replay_failed') or 0): W.append('benchmark: the independent checker refused %d reference proofs that the kernel accepted' % (ck['proofs'] - (ck.get('replay_failed') or 0) - ck['independent_accepted']))
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
    for group in ('arch', 'size', 'scratch'):                                                     # (the studies of the network and of the start: one number per variant, as for the main methods)
        for v in C.variants(group):
            if 'sample' not in C.solvers(group, v): continue
            for split in ('dev', 'test'):
                m, lo, hi, n = C.rate_ci(group, v, 'sample', split, B, 'all')
                if math.isnan(m): continue
                N['%s_%s' % (split, v)] = TB.pct(m, lo, hi, n); N['%s_%s_mean' % (split, v)] = round(100 * m, 2)
    for r in (T['laws'].rows if 'laws' in T else ()):                                             # (the exponent of a scaling law, the mean as the table prints it)
        N['law_%s_%s' % ('data' if r[1].startswith('synthetic') else 'params', 'iid' if 'iid' in r[0] else 'comp')] = str(r[4]).split(' ')[0]
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
    smax = max(set(P.get('synth_sizes') or ()) - {0}, default=None)                               # (the parameters of THE network of the paper -- the default encoder and width, as fine-tuned or as pre-trained on everything -- not of the first network of a study)
    pm = [f.get('n_params') for rid, f in sorted(C.finals.items()) if f.get('n_params') and (rid.startswith('ft/bc/') or (smax and rid.startswith('pre/n%d/' % smax)))] \
        or [f.get('n_params') for rid, f in sorted(C.finals.items()) if f.get('n_params')] or [((C.bench or {}).get('throughput') or {}).get('world', {}).get('params')]
    if pm and pm[0]: N['model_params_m'] = round(pm[0] / 1e6, 2)
    chk = sum(v['checked'] for v in C.recheck.values()); acc = sum(v['accepted'] for v in C.recheck.values())
    N['recheck_checked'] = chk; N['recheck_accepted'] = acc
    bench_numbers((C.bench or {}).get('throughput'), N)
    return N


def bench_numbers(b, N):
    """the numbers of the benchmark that the text of the paper quotes (rounded as they are quoted); a number whose measurement is not in bench.json is left out"""
    if not b: return
    for s, v in (b.get('solvers') or {}).items():
        if v.get('probes_per_s'): N['bench_probes_per_s_' + s] = round(v['probes_per_s'], 1)
    g = b.get('gym') or {}
    if g.get('decisions_per_s'): N['bench_gym_decisions_per_s'] = round(g['decisions_per_s'])
    sh = TB.breakdown_shares(((g.get('profiled') or {}).get('breakdown')))
    if sh:
        N['bench_gym_decision_language_percent'] = round(100 * sh['decision language']); N['bench_gym_engine_percent'] = round(100 * (sh['tactics and kernel'] + sh['menu of a goal'] + sh['probe machinery']))
    lat = b.get('latency') or {}
    for key in ('probe', 'menu', 'replay', 'start'):
        d = lat.get(key)
        if d and d.get('n'): N['bench_%s_median_ms' % key] = round(d['p50_ms'], 2); N['bench_%s_p99_ms' % key] = round(d['p99_ms'], 1)
    ck = b.get('check') or {}
    kc, ic = (ck.get('kernel_check') or {}), (ck.get('independent_check') or {})
    if kc.get('n'): N['bench_kernel_check_median_ms'] = round(kc['p50_ms'], 2)
    if ic.get('n'):
        N['bench_independent_check_median_ms'] = round(ic['p50_ms'], 1)
        if kc.get('n') and kc['p50_ms'] > 0: N['bench_independent_check_ratio'] = round(ic['p50_ms'] / kc['p50_ms'])
    if ck.get('proofs'): N['bench_check_proofs'] = ck['proofs']; N['bench_check_accepted'] = ck.get('independent_accepted')
    for key, v in (b.get('archs') or {}).items():
        if v.get('train_steps_per_s'): N['bench_train_steps_%s' % key.replace('@', '_')] = round(v['train_steps_per_s'])
        if v.get('params'): N['bench_params_m_%s' % key.replace('@', '_')] = round(v['params'] / 1e6, 2)
    ws = sorted(int(w) for w in (b.get('workers') or {}))
    for w in ws:
        v = b['workers'][str(w)]; m = v.get('memory') or {}
        if v.get('speedup') is not None: N['bench_speedup_%d' % w] = round(v['speedup'], 2)
        if m.get('pss_mb_max'): N['bench_pss_mb_%d' % w] = round(m['pss_mb_max'])
        if m.get('rss_sum_mb_max'): N['bench_rss_sum_mb_%d' % w] = round(m['rss_sum_mb_max'])
    det = b.get('determinism') or {}
    if det: N['bench_deterministic'] = 'yes' if det.get('identical') else 'NO'; N['bench_deterministic_tasks'] = det.get('tasks')
    mc = b.get('machine') or {}
    if mc: N['bench_cores'] = mc.get('cores_usable') or mc.get('cores')
    ov = [v['profile']['ratio_to_plain'] for v in (b.get('solvers') or {}).values() if (v.get('profile') or {}).get('ratio_to_plain')]
    if ov: N['bench_instrumentation_ratio_max'] = round(max(ov), 2)


def write_numbers(N, out):
    with open(os.path.join(out, 'numbers.json'), 'w') as f: json.dump(N, f, indent=1)
    L = ['% generated by exp_report.py -- do not edit']
    for k, v in N.items():
        s = str(v).replace('±', r'$\pm$').replace('%', r'\%').replace('_', r'\_')
        L.append(r'\newcommand{\%s}{%s}' % ('num' + texname(k), s))
    with open(os.path.join(out, 'numbers.tex'), 'w') as f: f.write('\n'.join(L) + '\n')


def md_report(R, C, T, P, figs, warnings, N):
    commits, done, failed, secs, trees, moved = status_summary(R)
    L = []; add = L.append
    add('# Experiment report -- profile `%s`' % P.get('name', '?')); add('')
    add('Generated %s from `%s`. The files beside this one (per-problem outcomes, statistics, figure data, provenance) are described in `README.md`.' % (time.strftime('%Y-%m-%d %H:%M'), R)); add('')
    add('## What was run'); add('')
    add('* %d runs finished (%d failed), %.1f hours of wall-clock summed over runs; commits of the code that made them: %s; code trees: %s.' % (done, failed, secs / 3600.0, ', '.join('%s (%d runs)' % (c[:10], n) for c, n in commits.items()) or 'unknown',
        ', '.join('%s (%d runs)' % (t[:12], n) for t, n in sorted(trees.items())) or 'unknown'))
    add('* seeds: %s; problems: %d dev, %d test (%d with a reference proof, %d statement-only); budget of the tables: %s probes per problem; time limit per problem %s s.' % (
        P.get('seed_list'), N.get('n_dev', 0), N.get('n_test', 0), N.get('n_test_proved', 0), N.get('n_test_unproved', 0), P.get('headline'), P.get('eval_time_limit')))
    add('* synthetic theorems: %s requested for pre-training; sizes %s.' % (max(P.get('synth_sizes', [0])), P.get('synth_sizes')))
    m = C.meta
    if m: add('* machine of the first evaluation: %s, %s cores, %s GB, torch %s, python %s; GPU: %s.' % (m.get('cpu'), m.get('cores'), m.get('memory_gb'), m.get('torch'), m.get('python'), m.get('gpu') or 'none'))
    add('')
    warnings = code_warnings(commits, trees, moved) + warnings
    add('## Warnings'); add('')
    if warnings:
        for w in warnings[:60]: add('* ' + w)
        if len(warnings) > 60: add('* ... and %d more' % (len(warnings) - 60))
    else: add('None: every evaluation row is present and no solver raised.')
    add('')
    order = [('system', 'The artefact', None), ('bank', 'The benchmark', None), ('main', 'Main results', 'fig_budget'), ('strict', 'Hiding the helper lemmas of a problem', None), ('ablations', 'Ablations', 'fig_ablations'), ('scaling', 'Scaling with the amount of synthetic data', 'fig_scaling'), ('arch', 'Other encoders of the goal', 'fig_arch'), ('size', 'Other widths', None), ('laws', 'Scaling laws', 'fig_laws'),
             ('scratch', 'Reinforcement learning from random weights', 'fig_scratch'), ('solvers', 'Provers on the same networks', 'fig_solvers'), ('synth', 'Held-out synthetic theorems', 'fig_synth'), ('h7', 'The pre-registered comparisons on the synthetic theorems (H7, H8)', None),
             ('mw', 'Manager-worker agent', None), ('levels', 'By difficulty and by contest', 'fig_levels'), ('generator', 'The synthetic theorem set', 'fig_generator'), ('library', 'Bootstrapped library', 'fig_library'),
             ('cost', 'Training cost', 'fig_learning'), ('bench_solvers', 'Throughput', 'fig_throughput'), ('bench_workers', None, None), ('bench_breakdown', 'Where the time goes', 'fig_breakdown'), ('bench_latency', 'Latency', 'fig_latency'),
             ('bench_archs', 'Cost of the encoders and widths', None), ('bench_world', None, None), ('bench_kernel', None, None), ('bench_machine', None, None), ('soundness', 'Soundness', None)]
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
        with open(os.path.join(out, 'tables', key + '.csv'), 'w') as f: f.write(t.csv())
    figs = {} if a.no_figures else __import__('exp_figures').make_all(C, os.path.join(out, 'figures'))
    warnings = completeness(C) + bench_warnings(C) + C.warnings
    N = numbers(C, T, P)
    write_numbers(N, out)
    if C.bank: write_problems(C, out)
    with open(os.path.join(out, 'report.md'), 'w') as f: f.write(md_report(R, C, T, P, figs, warnings, N))
    extra = EB.write_bundle(R, C, P, N, warnings, out); nfiles = EB.manifest(out)
    print('report: %d tables, %d figures, %d numbers, %d warnings, %d more files, %d in the manifest -> %s' % (len(T), len(figs), len(N), len(warnings), len(extra), nfiles, out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
