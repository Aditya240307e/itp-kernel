"""S51 gate: the TRAINING SUITE -- hindsight, the growing library, the goals of the learners, the provers and their protocol, the experiment matrix and the statistics the paper is made of.

Nothing here is trusted by the kernel and nothing is trusted from a learner: an agent's proof counts when the engine replayed it and the kernel checked it; a lemma becomes a premise when the kernel checked its
proof; a number in a table is arithmetic on rows.  What this gate protects is that the layer between the kernel and the paper does what its documentation says -- a hindsight theorem is a theorem, a lemma of
the library is a theorem that no held-out problem can lean on, the curve "solved within B probes" read off one run is the curve of a run with budget B, an interval is the interval of the formula -- and that
the experiment matrix is a well-formed graph of commands whose products land where the collector looks for them.

  A. STATISTICS.     The estimators of exp_stats against values worked out by hand (Student-t, Wilson, Holm, Chao1, paired tests, the bootstrap on constant data, rarefaction, crossing points).
  B. THE MATRIX.     The run graph of every profile: unique ids, an acyclic graph whose needs exist, the checks every run promises, the closure of a target, the seeds, the overrides (an unknown field is an error), the
                     machine fields, the synthetic-set exclusion and the training commands (no test data in them), the estimator.
  C. THE COLLECTOR.  A runs directory written by hand (3 seeds, 2 learners, baselines) is read back: solved-within-budget matrices, rates with their intervals, comparisons, the strict view, the subsets, the
                     report and its numbers, the warnings for a missing result, the contest family of a source and the facts of a report made of the dataset alone (what the paper skeleton prints).
  D. HINDSIGHT.      On a small world (17 problems, a dataset built by dataset.py --only): what the Harvest keeps is exactly what the proof tree says was proved, every theorem replays in a fresh engine of another
                     configuration and its code builds the statement back; the filters do what they say; corrupted proofs are dropped and counted; claims.
  E. THE LIBRARY.    Lemmas the agent proved become premises: every rule of admission (one test per reason), the kernel's acceptance, the independent checker's acceptance of a proof that cites a lemma, the window of
                     premises (a lemma sees the library and the lemmas before it, no more), determinism across worlds, corrupted records, the file.
  F. THE GOALS.      Where the training episodes start: train problems and train sub-theorems only, reproducible sampling, the budget, plain data, statements resolved in the world.
  G. THE PROVERS.    The solver specs, the protocol (an unsolved search spends budget + 1 probes; "solved within B" read off the largest budget is what a run with budget B finds), `evaluate` (rows, annotations,
                     resumption, determinism, both regimes, the second opinion), the library of a checkpoint, the world digest of a dataset.
  H. EXPERT ITERATION.  A run of ml_ei.py with hindsight and bootstrapping on the small world, then ml_evaluate.py on its checkpoint with `--lemmas auto` (also when the run admitted no lemma, and the loud
                     failure when the file is gone).
  I. THE RUNNER.     The shell scripts parse (`bash -n`) and read their settings from env.sh (scripts/local.env wins over the defaults, the environment over both); a real scheduler on trivial commands
                     (needs, failure, skip, resume, cores, memory packing, `--force`, `--cascade`, the lock); dry-run and status through exp_run; the profile guard on machine fields.

Environment: S51_TRACE=1 prints timings, S51_ONLY=A,B selects sections, S51_DATA=DIR reuses a small dataset built earlier by dataset.py --only (development).  D-H build the small dataset (about two minutes).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, json, math, time, copy, glob, shutil, tempfile, argparse, itertools, subprocess, collections, traceback, statistics
sys.setrecursionlimit(200000)
for _sub in ('experiments',): _sys.path.insert(0, os.path.join(itp_paths.ROOT, 'src', 'python', _sub))

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S51_TRACE'))
ONLY = set(os.environ['S51_ONLY'].split(',')) if os.environ.get('S51_ONLY') else None


def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)


def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return bool(cond)


def raises(what, fn, exc):
    try: fn()
    except exc: return expect(True, what)
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:100]))
    return expect(False, what + ' (it raised nothing)')


def close(a, b, tol=1e-9): return abs(float(a) - float(b)) <= tol * (1.0 + abs(float(b)))


try:
    import numpy as np
    import scipy.stats as SS
except ImportError as e:
    print('S51 needs numpy and scipy: %s' % (e,)); print('S51 TRAINING SUITE GATE: FAIL'); sys.exit(1)
import exp_stats as ST, exp_spec as SP, exp_run as RUN, exp_estimate as EST, exp_collect as EC, exp_tables as TB, exp_report as REP, exp_repo as XR

T975_2 = 4.302652729911275                      # Student-t quantile, 0.975, 2 degrees of freedom
T975_1 = 12.706204736432095                     # ... 1 degree of freedom
T975_4 = 2.7764451051977987                     # ... 4 degrees of freedom
SMOKE = SP.SMOKE_PROBLEMS
NAN = float('nan')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. statistics
def sec_A(c):
    m, lo, hi = ST.mean_ci([1.0, 2.0, 3.0])
    h = T975_2 * 1.0 / math.sqrt(3)
    expect(close(m, 2.0) and close(lo, 2.0 - h, 1e-9) and close(hi, 2.0 + h, 1e-9), 'A1 mean_ci([1,2,3]) is the Student-t interval 2 +- %.5f (got %.5f, %.5f, %.5f)' % (h, m, lo, hi))
    m, lo, hi = ST.mean_ci([0.5])
    expect(m == 0.5 and math.isnan(lo) and math.isnan(hi), 'A2 one replicate has a mean and no interval')
    m, lo, hi = ST.mean_ci([])
    expect(math.isnan(m) and math.isnan(lo) and math.isnan(hi), 'A3 no replicate has nothing')
    m, lo, hi = ST.mean_ci([1.0, NAN, 3.0, None])
    expect(close(m, 2.0) and close(hi - lo, 2 * T975_1 * math.sqrt(2) / math.sqrt(2)), 'A4 mean_ci ignores NaN and None (two values left: 1 degree of freedom, got %.4f)' % (hi - lo))
    m, lo, hi = ST.mean_ci([0.2, 0.4, 0.6, 0.8, 1.0], level=0.95)
    sd = statistics.stdev([0.2, 0.4, 0.6, 0.8, 1.0])
    expect(close(hi - m, T975_4 * sd / math.sqrt(5)), 'A5 five replicates use 4 degrees of freedom')

    p, lo, hi = ST.wilson(5, 10)
    expect(close(p, 0.5) and abs(lo - 0.2366) < 5e-4 and abs(hi - 0.7634) < 5e-4, 'A6 wilson(5, 10) = 0.5 [0.2366, 0.7634] (got %.4f [%.4f, %.4f])' % (p, lo, hi))
    p, lo, hi = ST.wilson(0, 10)
    expect(p == 0.0 and lo == 0.0 and abs(hi - 0.2775) < 5e-4, 'A7 wilson(0, 10): the interval starts at 0 and ends near 0.2775 (got %.4f)' % hi)
    p, lo, hi = ST.wilson(10, 10)
    expect(p == 1.0 and close(hi, 1.0, 1e-12) and hi <= 1.0 and abs(lo - 0.7225) < 5e-4, 'A8 wilson(10, 10) ends at 1 (lower end %.4f)' % lo)
    expect(all(math.isnan(x) for x in ST.wilson(0, 0)), 'A9 wilson with no trial is NaN')
    expect(all(0.0 <= v <= 1.0 for k in range(0, 31) for v in ST.wilson(k, 30)), 'A10 every Wilson interval lies in [0, 1]')

    ad = ST.holm([0.01, 0.04, 0.03])
    expect([round(x, 12) for x in ad] == [0.03, 0.06, 0.06], 'A11 Holm: [0.01, 0.04, 0.03] -> [0.03, 0.06, 0.06] (got %s)' % ad)
    ad = ST.holm([0.5, NAN, 0.001])
    expect(close(ad[2], 0.002) and close(ad[0], 0.5) and math.isnan(ad[1]), 'A12 Holm keeps NaN and uses the number of real p-values (got %s)' % ad)
    expect(ST.holm([0.9, 0.8]) == [1.0, 1.0], 'A13 Holm of two large p-values: both 1 (the smaller one is multiplied by 2 and capped, the larger cannot go below it)')
    expect(all(x <= 1.0 for x in ST.holm([0.6, 0.7, 0.8])), 'A14 adjusted p-values are capped at 1')

    est, se = ST.chao1([1, 1, 2, 3, 5])
    expect(close(est, 5.5), 'A15 chao1 of abundances [1,1,2,3,5]: 5 + f1 (f1 - 1) / (2 (f2 + 1)) = 5.5 (got %s)' % est)
    est, se = ST.chao1([3, 4, 5])
    expect(close(est, 3.0) and se == 0.0, 'A16 chao1 with no singleton is the observed count')
    est, se = ST.chao1([1] * 10)
    expect(close(est, 10 + 10 * 9 / 2.0) and se > 0, 'A17 chao1 of ten singletons = 10 + 45 = 55')
    expect(ST.chao1([0, 0, 2])[0] == 1.0, 'A18 classes with zero count are not classes')

    r = ST.rarefaction(['a', 'b', 'c', 'd'], [1, 2, 4])
    expect([x[1] for x in r] == [1.0, 2.0, 4.0], 'A19 rarefaction of distinct labels is the sample size')
    r = ST.rarefaction(['a'] * 6, [1, 3, 6])
    expect([x[1] for x in r] == [1.0, 1.0, 1.0], 'A20 rarefaction of one label is 1')
    r = ST.rarefaction(['a', 'a', 'b', 'b'], [4])
    expect(r[0][1] == 2.0 and r[0][2] == 0.0, 'A21 all items drawn: the number of labels, no spread')

    expect(close(ST.first_crossing([1, 2, 3], [0.1, 0.3, 0.7], 0.5), 2.5), 'A22 first_crossing interpolates linearly between the points around the level')
    expect(ST.first_crossing([1, 2, 3], [0.1, 0.3, 0.4], 0.5) is None, 'A23 a curve that never reaches the level has no crossing')
    expect(ST.first_crossing([1, 2], [0.6, 0.7], 0.5) == 1, 'A24 a curve above the level at the start crosses at its start')

    d = ST.paired_seed_diff([0.5, 0.6, 0.7], [0.4, 0.5, 0.5])
    dd = np.array([0.1, 0.1, 0.2]); pv = float(SS.ttest_1samp(dd, 0.0).pvalue)
    expect(close(d['mean'], dd.mean()) and abs(d['p'] - pv) < 1e-12 and d['n'] == 3, 'A25 paired_seed_diff: the mean difference and the p of the paired t-test (p %.4f)' % d['p'])
    expect(ST.paired_seed_diff([0.5, 0.5], [0.5, 0.5])['p'] == 1.0, 'A26 identical replicates: p = 1')
    expect(ST.paired_seed_diff([0.6, 0.6], [0.5, 0.5])['p'] == 0.0, 'A27 a constant nonzero difference: p = 0 (no variance)')
    expect(math.isnan(ST.paired_seed_diff([0.5], [0.4])['p']), 'A28 one pair has no p')

    A = np.ones((3, 8)); B = np.zeros((3, 8))
    b = ST.paired_problem_bootstrap(A, B)
    expect(b['mean'] == 1.0 and b['low'] == 1.0 and b['high'] == 1.0, 'A29 the bootstrap of a constant difference is a point')
    b = ST.paired_problem_bootstrap(A, A)
    expect(b['mean'] == 0.0 and b['low'] == 0.0 and b['high'] == 0.0, 'A30 the bootstrap of identical methods is 0')
    rng = np.random.default_rng(3); A = (rng.random((4, 40)) < 0.7).astype(float); B = (rng.random((4, 40)) < 0.4).astype(float)
    b1 = ST.paired_problem_bootstrap(A, B, seed=5); b2 = ST.paired_problem_bootstrap(A, B, seed=5)
    expect(b1 == b2 and b1['low'] <= b1['mean'] <= b1['high'] and b1['mean'] > 0, 'A31 the bootstrap is reproducible from its seed and brackets the mean')
    raises('A32 two methods on different problems cannot be compared', lambda: ST.paired_problem_bootstrap(np.ones((2, 3)), np.ones((2, 4))), ValueError)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. the matrix
def sec_B(c):
    sizes = {}
    for name in sorted(SP.PROFILES):
        P = SP.profile(name); runs = SP.build(name, runs_dir='/runs/%s' % name)
        sizes[name] = len(runs)
        ids = list(runs)
        expect(len(set(ids)) == len(ids) and all(r.id == i for i, r in runs.items()), 'B1 %s: run ids are unique and keys' % name)
        pos = {i: k for k, i in enumerate(ids)}
        expect(all(n in runs and pos[n] < pos[i] for i, r in runs.items() for n in r.needs), 'B2 %s: every need exists and comes earlier (the graph is in dependency order, hence acyclic)' % name)
        expect(all(r.argv and r.check for r in runs.values() if r.stage != 'report') and all(r.argv for r in runs.values()), 'B3 %s: every run has a command and (but the report) promises files' % name)
        stages = [r.stage for r in runs.values()]
        expect(set(stages) == {'data', 'pre', 'ft', 'rl', 'eval', 'bench', 'report'}, 'B4 %s: the stages are data pre ft rl eval bench report (got %s)' % (name, sorted(set(stages))))
        seeds = P['seeds']
        for v in SP.RL_VARIANTS:
            expect(all(('rl/%s/s%d' % (v, s)) in runs for s in range(seeds)), 'B5 %s: variant %s has a run per seed' % (name, v))
        expect(all(('ft/%s/s%d' % (v, s)) in runs for v in ('bc', 'bc_scratch', 'bc_mw') for s in range(seeds)), 'B6 %s: the fine-tuning runs exist for every seed' % name)
        for i, r in runs.items():
            for n in r.needs:
                if runs[n].stage == 'rl' or runs[n].stage == 'eval': expect(r.stage in ('eval', 'report', 'bench'), 'B7 %s: only evaluations and the report need a learner (%s needs %s)' % (name, i, n))
        # the evaluation needs the checkpoint it evaluates, for every seed it names
        for i, r in runs.items():
            if r.stage != 'eval' or i == 'eval/baselines': continue
            for mv in [a for a in r.argv if '=' in a and a.endswith(('best.pt', 'last.pt'))]:
                path = mv.split('=', 1)[1]
                owner = [j for j in r.needs if ('/' + j + '/') in path]
                expect(len(owner) == 1, 'B8 %s: %s needs the run that makes %s' % (name, i, path))
        # no run trains on a dev or test problem, and the synthetic set excludes the benchmark
        merge = runs['data/synth_merge']
        expect('--exclude-dataset' in merge.argv, 'B9 %s: the synthetic set is merged with the benchmark excluded' % name)
        gen = runs['data/synth_gen']; arg = lambda r, flag: r.argv[r.argv.index(flag) + 1]
        expect(gen.argv[2] == 'grow' and int(arg(gen, '--train-target')) == P['synth_target'] == int(arg(merge, '--train-target')) and '--target' not in merge.argv,
               'B9b %s: the synthetic set is made for a number of TRAIN theorems, by `grow` and by the merge alike' % name)
        expect(int(arg(gen, '--max-attempts')) > int(arg(gen, '--attempts')) == P['synth_attempts'] and arg(gen, '--exclude-dataset') == arg(merge, '--exclude-dataset'),
               'B9c %s: grow counts with the merge\'s own exclusions, and may add rounds' % name)
        expect(all(int(arg(r, '--synth-max')) <= P['synth_target'] for r in runs.values() if '--synth-max' in r.argv) and max(P['synth_sizes']) == P['synth_target'],
               'B9d %s: no run asks for more synthetic theorems than the set has, and the largest pre-training is the whole set' % name)
        for i, r in runs.items():
            if r.stage in ('pre', 'ft', 'rl'):
                expect('--split' not in r.argv and '--test' not in r.argv, 'B10 %s: %s does not choose a split (the trainers read the train split and use dev only to select)' % (name, i))
        for s in range(min(seeds, 2)):
            pre = runs['pre/n%d/s%d' % (max(P['synth_sizes']), s)]
            expect('--no-bank' in pre.argv and '--no-lib' in pre.argv, 'B11 %s: pre-training sees no bank problem and no library lemma' % name)
            expect(runs['ft/bc/s%d' % s].needs == ('pre/n%d/s%d' % (max(P['synth_sizes']), s),), 'B12 %s: fine-tuning continues from the pre-trained network of its own seed' % name)
            expect('--init' not in runs['ft/bc_scratch/s%d' % s].argv, 'B13 %s: bc_scratch starts from random weights' % name)
        expect(runs['eval/baselines'].needs == ('data/dataset',) and not any('--model' in a for a in runs['eval/baselines'].argv), 'B14 %s: the baselines use no network' % name)
        expect(all('prefix,strict' in r.argv for i, r in runs.items() if r.stage == 'eval'), 'B15 %s: every evaluation runs both regimes' % name)
    expect(sizes['smoke'] < sizes['small'] and sizes['small'] < sizes['full'], 'B16 the profiles grow: smoke %d < small %d < full %d runs' % (sizes['smoke'], sizes['small'], sizes['full']))

    r5 = SP.build('full', runs_dir='/r'); r7 = SP.build('full', seeds=list(range(7)), runs_dir='/r')
    expect(len(r7) - len(r5) == 2 * len([i for i in r5 if i.endswith('/s0')]), 'B17 two more seeds add exactly the per-seed runs twice (%d -> %d)' % (len(r5), len(r7)))
    expect(all(('rl/ei_her/s%d' % s) in r7 for s in range(7)) and 'rl/ei_her/s6' not in r5, 'B18 --seeds 0-6 gives seven replicates')
    raises('B19 an unknown field of the profile is an error', lambda: SP.profile('full', {'rl_itres': 3}), ValueError)
    raises('B20 --set wants key=value', lambda: SP.parse_overrides(['rl_iters']), ValueError)
    ov = SP.parse_overrides(['rl_iters=20', 'synth_sizes=(0, 100)', 'device=cuda'])
    expect(ov == {'rl_iters': 20, 'synth_sizes': (0, 100), 'device': 'cuda'}, 'B21 overrides are Python literals, or strings (%s)' % ov)
    rs = SP.build('smoke', runs_dir='/r', overrides={'rl_iters': 7})
    expect('7' in rs['rl/ei_her/s0'].argv, 'B22 an override reaches the command')
    expect(SP.build('full', runs_dir='/r', overrides={'workers': 8})['rl/ei/s0'].cores == 8, 'B23 workers are the cores of a run')
    ep = SP.effective_profile('full', {'device': 'cuda'}, [0, 1])
    expect(ep['device'] == 'cuda' and ep['seed_list'] == [0, 1] and isinstance(ep['synth_sizes'], list), 'B24 the effective profile is plain JSON data (%s)' % sorted(ep)[:3])
    expect(RUN.parse_seeds('0-4') == [0, 1, 2, 3, 4] and RUN.parse_seeds('0,2,5') == [0, 2, 5] and RUN.parse_seeds('0-2,5') == [0, 1, 2, 5] and RUN.parse_seeds('3,1,3') == [1, 3], 'B25 seed lists')
    runs = SP.build('full', runs_dir='/r')
    cl = SP.closure(runs, ['eval/main/ei_her'])
    expect('rl/ei_her/s0' in cl and 'ft/bc/s0' in cl and 'pre/n60000/s0' in cl and 'data/synth_merge' in cl and 'data/dataset' in cl and 'rl/ppo/s0' not in cl, 'B26 the closure of an evaluation is its learners and their ancestors')
    expect(cl == [i for i in runs if i in set(cl)], 'B27 the closure keeps the order of the graph')
    raises('B28 the closure of an unknown run is an error', lambda: SP.closure(runs, ['rl/nothing']), KeyError)
    dep = RUN.dependents(runs, 'ft/bc/s0')
    expect('rl/ei/s0' in dep and 'eval/main/bc' in dep and 'report' not in dep and 'ft/bc/s1' not in dep, 'B29 dependents of ft/bc/s0: its learners and evaluations, not the other seeds')
    mix = {'device', 'threads', 'workers', 'mem_scale'}
    expect(set(RUN.MACHINE_FIELDS) == mix, 'B30 the machine fields are device, threads, workers, mem_scale')
    for k in mix: expect(k in SP.PROFILES['full'], 'B31 the machine field %s is a profile field' % k)
    # the boot variant is evaluated per seed with its own library
    boot = [i for i in runs if i.startswith('eval/main/ei_her_boot')]
    expect(len(boot) == 5 and all('--lemmas' in runs[i].argv and 'auto' in runs[i].argv for i in boot), 'B32 the bootstrapped learner is evaluated per seed, with the library it trained with')
    expect('ei_her_boot' not in ' '.join(runs['eval/main/ei_her'].argv) and '--lemmas' not in runs['eval/main/ei_her'].argv, 'B33 the other learners are not given a library')
    # the estimator
    P = SP.profile('full'); counts = {'train_steps': 20000, 'dev': 40, 'test': 80, 'strict_extra': 10, 'train': 200}
    est = EST.estimate(SP.build('full', runs_dir='/r'), P, counts, EST.RATES)
    tot = sum(v[1] for v in est.values())
    expect(set(est) == {'data', 'pre', 'ft', 'rl', 'eval', 'bench', 'report'} and all(v[1] > 0 and v[2] >= v[1] for v in est.values()), 'B34 the estimate has every stage, positive, core-seconds >= seconds')
    est2 = EST.estimate(SP.build('full', runs_dir='/r'), P, counts, EST.RATES, train_rate=EST.RATES['train_steps_per_s'] * 10)
    expect(est2['pre'][1] < est['pre'][1] / 5 and abs(est2['eval'][1] - est['eval'][1]) < 1e-6, 'B35 a faster trainer shortens training and nothing else')
    est3 = EST.estimate(SP.build('full', runs_dir='/r'), P, counts, EST.RATES, probe_rate=1e6)
    expect(est3['eval'][1] < est['eval'][1] / 10 and est3['rl'][1] < est['rl'][1], 'B36 faster probes shorten evaluation and search')
    p7 = dict(P); est7 = EST.estimate(SP.build('full', seeds=list(range(7)), runs_dir='/r'), p7, counts, EST.RATES)
    expect(sum(v[1] for v in est7.values()) > tot * 1.3, 'B37 seven seeds cost more than five')
    tmp = tempfile.mkdtemp(prefix='s51b_')
    try:
        bpath = os.path.join(tmp, 'bench.json'); json.dump({'solvers': {'sample': {'probes_per_s': 123.0}, 'ida': {'probes_per_s': 0}}, 'world': {'train_steps_per_s': 777.0}}, open(bpath, 'w'))
        r_, ok = EST.rates_from_bench(bpath)
        expect(ok and r_['probes_per_s']['sample'] == 123.0 and r_['probes_per_s']['ida'] == EST.RATES['probes_per_s']['ida'] and r_['train_steps_per_s'] == 777.0, 'B38 measured rates replace the defaults, a zero rate does not')
        r_, ok = EST.rates_from_bench(os.path.join(tmp, 'nothing.json'))
        expect(not ok and r_ == EST.RATES, 'B39 no bench file: the defaults')
        bank = {'problems': [{'split': 'train', 'demonstration': {'length': 5}, 'demonstration_strict': {'length': 4}, 'helpers': ['h']}, {'split': 'train', 'demonstration': {'length': 3}}, {'split': 'dev'}, {'split': 'test'}]}
        json.dump(bank, open(os.path.join(tmp, 'bank.json'), 'w'))
        json.dump({'lemmas': [{'split': 'train', 'length': 2}, {'split': 'dev', 'length': 9}]}, open(os.path.join(tmp, 'library.json'), 'w'))
        dc = EST.dataset_counts(tmp)
        expect(dc == {'train_steps': 5 + 4 + 3 + 2, 'dev': 1, 'test': 1, 'train': 2, 'strict_extra': 1}, 'B40 dataset_counts counts the train steps of both regimes and the train lemmas (%s)' % dc)
        expect(EST.dataset_counts(os.path.join(tmp, 'nope')) is None, 'B41 no dataset, no counts')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # the commands of the full profile are the ones of the documentation
    full = SP.build('full', runs_dir='/r')
    expect(full['pre/n60000/s0'].argv.count('--synth-max') == 1 and '60000' in full['pre/n60000/s0'].argv, 'B42 the full profile pre-trains on 60000 synthetic theorems')
    expect(sorted(int(i.split('/n')[1].split('/')[0]) for i in full if i.startswith('pre/') and i.endswith('/s0')) == [1000, 4000, 15000, 60000], 'B43 the sizes of the scaling study')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. the collector
BC_ROWS = {   # seed -> {problem: probes or None (unsolved)}   test problems t1..t4 (t4 is statement-only), dev d1, d2
    0: {'t1': 5, 't2': 50, 't3': None, 't4': None, 'd1': 2, 'd2': None},
    1: {'t1': 8, 't2': None, 't3': 90, 't4': None, 'd1': 2, 'd2': None},
    2: {'t1': 3, 't2': 20, 't3': None, 't4': 60, 'd1': 2, 'd2': None},
}
EI_ROWS = {
    0: {'t1': 4, 't2': 30, 't3': 40, 't4': None, 'd1': 2, 'd2': 70},
    1: {'t1': 6, 't2': 45, 't3': None, 't4': None, 'd1': 2, 'd2': None},
    2: {'t1': 2, 't2': 10, 't3': 120, 't4': 55, 'd1': 2, 'd2': 80},
}
SPLIT = {'t1': 'test', 't2': 'test', 't3': 'test', 't4': 'test', 'd1': 'dev', 'd2': 'dev'}


def make_row(variant, solver, seed, name, probes, budget=200, regime='prefix', group_method=None):
    solved = probes is not None
    return {'method': group_method or ('%s+%s' % (variant, solver)), 'model': variant, 'spec': solver, 'seed': seed, 'name': name, 'regime': regime, 'split': SPLIT[name], 'level': 'AMC' if name != 't4' else 'IMO',
            'origin': 'handwritten', 'source': 'AMC 2000' if name != 't4' else 'IMO 2000 P1', 'proved': name != 't4', 'family': name, 'solved': solved, 'probes': probes if solved else budget + 1,
            'steps': 3 if solved else None, 'seconds': 0.5, 'decisions': 4}


def write_fixture(R):
    os.makedirs(R, exist_ok=True)
    json.dump({'name': 'full', 'headline': 100, 'seed_list': [0, 1, 2], 'eval_probes': 200, 'eval_time_limit': 10.0, 'synth_sizes': [0, 100], 'eval_budgets': [10, 20, 50, 100, 200], 'dataset': '', 'workers': 1},
              open(os.path.join(R, 'PROFILE.json'), 'w'))
    for variant, table in (('bc', BC_ROWS), ('ei_her', EI_ROWS)):
        d = os.path.join(R, 'eval', 'main', variant); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'rows.jsonl'), 'w') as f:
            for seed, rows in table.items():
                for name, probes in rows.items():
                    if variant == 'ei_her' and seed == 2 and name == 't3': continue                          # a missing result: the report must say so
                    f.write(json.dumps(make_row(variant, 'sample', seed, name, probes)) + '\n')
            f.write(json.dumps(make_row(variant, 'sample', 0, 't1', 7, regime='strict')) + '\n')            # a strict-regime row: invisible in the default view
        json.dump({'argv': [], 'machine': {}}, open(os.path.join(d, 'meta.json'), 'w'))
        json.dump({'checked': 3, 'accepted': 3}, open(os.path.join(d, 'recheck.json'), 'w'))
    d = os.path.join(R, 'eval', 'baselines'); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'rows.jsonl'), 'w') as f:
        for name in SPLIT:
            f.write(json.dumps(make_row(None, 'blind', 0, name, 3 if name in ('d1', 't1') else None, group_method='blind') | {'model': None}) + '\n')
            f.write(json.dumps(make_row(None, 'uniform', 0, name, 9 if name == 't1' else None, group_method='uniform') | {'model': None}) + '\n')
    for variant, hist in (('bc', [(0, 0.9), (1, 0.7)]),):
        for seed in range(3):
            d = os.path.join(R, 'ft', variant, 's%d' % seed); os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, 'log.jsonl'), 'w') as f:
                for ep, loss in hist: f.write(json.dumps({'epoch': ep, 'train_loss': loss + 0.01 * seed, 'seconds': 10.0 * (ep + 1)}) + '\n')
            json.dump({'n_params': 1_500_000}, open(os.path.join(d, 'final.json'), 'w'))


def rate(table, names, budget):
    return statistics.mean(1.0 if (table[n] is not None and table[n] <= budget) else 0.0 for n in names)


def sec_C(c):
    tmp = tempfile.mkdtemp(prefix='s51c_')
    try:
        R = os.path.join(tmp, 'runs'); write_fixture(R)
        C = EC.load(R)
        expect(len(C.all_rows) == 2 * 3 * 6 + 2 - 1 + 12 and len(C.rows) == len(C.all_rows) - 2, 'C1 the rows: 35 sample rows + 2 strict + 12 baseline rows, the default view without the strict ones (got %d, %d)' % (len(C.all_rows), len(C.rows)))
        expect(C.variants('main') == ['bc', 'ei_her'] and C.solvers('main', 'bc') == ['sample'] and C.seeds('main', 'bc') == [0, 1, 2], 'C2 variants, solvers, seeds')
        expect(C.variants('baselines') == ['baselines'] and C.solvers('baselines', 'baselines') == ['blind', 'uniform'], 'C3 the baselines are one variant with two solvers')
        expect(C.names('test') == ['t1', 't2', 't3', 't4'] and C.names('dev') == ['d1', 'd2'] and C.names('dev+test') == ['d1', 'd2', 't1', 't2', 't3', 't4'], 'C4 names by split')
        expect(C.names('test', 'proved') == ['t1', 't2', 't3'] and C.names('test', 'unproved') == ['t4'] and C.names('test', 'free') == ['t1', 't2', 't3', 't4'], 'C5 subsets: proved, unproved, free (no helper known)')
        test = ['t1', 't2', 't3', 't4']
        for B in (5, 20, 50, 100, 200, None):
            M = C.success('main', 'bc', 'sample', 'test', B)
            exp = [rate({n: BC_ROWS[s][n] for n in test}, test, B if B is not None else 10 ** 9) for s in (0, 1, 2)]
            expect(M.seeds == [0, 1, 2] and M.names == test and all(close(a, b) for a, b in zip(M.rates(), exp)), 'C6 solved within %s probes, per seed: %s (got %s)' % (B, exp, list(M.rates())))
        M = C.success('main', 'bc', 'sample', 'test', 50)
        expect(M.data[0].tolist() == [1, 1, 0, 0] and M.data[1].tolist() == [1, 0, 0, 0] and M.data[2].tolist() == [1, 1, 0, 0], 'C7 the matrix at budget 50: t2 at exactly 50 probes counts, t3 at 90 and t4 at 60 do not (%s)' % M.data.tolist())
        m, lo, hi, n = C.rate_ci('main', 'bc', 'sample', 'test', 100)
        r = [0.5, 0.5, 0.75]; sd = statistics.stdev(r); h = T975_2 * sd / math.sqrt(3)
        expect(n == 3 and close(m, statistics.mean(r)) and close(lo, m - h) and close(hi, m + h), 'C8 rate_ci: mean %.4f +- %.4f over three replicates (got %.4f [%.4f, %.4f])' % (statistics.mean(r), h, m, lo, hi))
        m, lo, hi, n = C.rate_ci('baselines', 'baselines', 'blind', 'test', 100)
        p, wlo, whi = ST.wilson(1, 4)
        expect(n == 1 and close(m, 0.25) and close(lo, wlo) and close(hi, whi), 'C9 a single run gets a Wilson interval over problems (1 of 4: %.3f [%.3f, %.3f])' % (m, lo, hi))
        cv = C.curve('main', 'bc', 'sample', 'test', [5, 20, 50, 100, 200])
        expect(all(cv['mean'][i] <= cv['mean'][i + 1] + 1e-12 for i in range(4)) and len(cv['per_seed']) == 5, 'C10 the curve is nondecreasing in the budget (%s)' % [round(x, 3) for x in cv['mean']])
        pr, ln = C.efficiency('main', 'bc', 'sample', 'test', 100)
        expect(close(pr[0], statistics.mean([5, 50])) and close(pr[1], statistics.mean([8, 90])) and close(pr[2], statistics.mean([3, 20, 60])) and ln == [3.0, 3.0, 3.0], 'C11 efficiency: mean probes of the solved problems per replicate (%s)' % pr)
        cmp = C.compare('main', 'ei_her', 'bc', 'sample', 'test', 100)
        # seed 2 of ei_her has no row for t3: that problem is dropped from the comparison
        keep = ['t1', 't2', 't4']
        ra = [rate({n: EI_ROWS[s][n] for n in keep}, keep, 100) for s in (0, 1, 2)]; rb = [rate({n: BC_ROWS[s][n] for n in keep}, keep, 100) for s in (0, 1, 2)]
        expect(cmp['n_problems'] == 3 and cmp['dropped'] == 1 and close(cmp['diff']['mean'], statistics.mean([a - b for a, b in zip(ra, rb)])), 'C12 compare pairs replicates by seed and drops the problem with a missing result (%s vs %s)' % (ra, rb))
        expect(cmp['seeds'] == [0, 1, 2] and cmp['problem']['low'] <= cmp['problem']['mean'] <= cmp['problem']['high'], 'C13 compare: the bootstrap interval over problems brackets the difference')
        V = C.view('strict')
        expect(len(V.rows) == 2 and all(r['regime'] == 'strict' for r in V.rows) and len(C.rows) == len(C.all_rows) - 2, 'C14 the strict view holds the strict rows, the default view does not')
        expect(all(close(a, b) for a, b in zip(V.success('main', 'bc', 'sample', 'test', 100, names=['t1']).rates(), [NAN] * 0 + [1.0, NAN, NAN]) if not math.isnan(b)), 'C15 a strict row is found in the strict view')
        expect(close(C.seconds_per_problem('main', 'bc', 'sample', 'test'), 0.5), 'C16 seconds per problem')
        lc = C.log_curve('ft', 'bc', 'seconds', 'train_loss')
        expect(lc is not None and lc['index'] == [0, 1] and close(lc['mean'][0], statistics.mean([0.9, 0.91, 0.92])) and lc['n'] == [3, 3], 'C17 a learning curve is aligned by epoch and averaged over replicates (%s)' % (lc and lc['mean']))
        expect(C.recheck.get('eval/main/bc') == {'checked': 3, 'accepted': 3}, 'C18 the second opinion verdicts are read')
        # the report
        out = os.path.join(tmp, 'report')
        rc = REP.main([R, '--out', out, '--no-figures'])
        N = json.load(open(os.path.join(out, 'numbers.json')))
        expect(rc == 0 and close(N['test_bc_mean'], 100 * statistics.mean(r), 1e-3) and N['headline_budget'] == 100, 'C19 the report: numbers.json has the table cell of bc at the headline budget (%s)' % N.get('test_bc_mean'))
        expect(N['n_test'] == 4 and N['n_dev'] == 2 and N['n_test_proved'] == 3 and N['n_test_unproved'] == 1 and N['n_seeds'] == 3, 'C20 the report counts problems and seeds (%s)' % {k: N.get(k) for k in ('n_test', 'n_dev', 'n_test_proved', 'n_test_unproved', 'n_seeds')})
        expect(N['recheck_checked'] == 6 and N['recheck_accepted'] == 6, 'C21 the second-opinion totals')
        md = open(os.path.join(out, 'report.md')).read()
        expect('## Warnings' in md and 'ei_her' in md.split('## Warnings')[1].split('##')[0] and 'missing' in md.split('## Warnings')[1].split('##')[0], 'C22 the report warns about the missing result of ei_her')
        expect(all(os.path.exists(os.path.join(out, 'tables', t + '.md')) and os.path.exists(os.path.join(out, 'tables', t + '.tex')) for t in ('main', 'soundness', 'cost')), 'C23 tables are written in Markdown and LaTeX')
        tex = open(os.path.join(out, 'numbers.tex')).read()
        expect('\\newcommand{\\numTestBc}' in tex and '\\numHeadlineBudget' in tex, 'C24 numbers.tex defines the macros the paper cites')
        expect('Statement-only problems solved' in md and '| t4 |' in md, 'C25 the statement-only problem t4, solved by a run, is listed as a discovery')
        # a table is as wide as its content says and escapes what LaTeX would choke on
        t = TB.Table('demo', 'cap_tion 50%', ['a', 'b_c', 'd%'], [['x', '1.0 ± 2', '50%'], ['y', 'q_r', '–']])
        s = t.tex()
        expect('\\_' in s and '\\%' in s and '--' in s and '–' not in s and '\\begin{tabular}' in s, 'C26 LaTeX output escapes underscores, percent signs and dashes')
        # the contest family of a source, and the facts a report can state before any run (the paper's skeleton prints them: system and benchmark sections)
        expect(EC.source_class('original, in the manner of the AMC') == 'AMC' and EC.source_class('AIME 2001 I Problem 3') == 'AIME' and EC.source_class('IMO 1959 Problem 1') == 'IMO' and EC.source_class('folklore') == 'folklore'
               and EC.source_class('classical (Euclid)') == 'other' and EC.source_class(None) == 'other', 'C28 the contest family of a source: the problems written in the manner of the AMC are AMC problems of the bank (not "other")')
        ddir = os.path.join(itp_paths.ROOT, 'data', 'dataset_v2'); facts = XR.dataset_facts(ddir); sub = json.load(open(os.path.join(ddir, 'subproblems.json')))
        expect(facts.get('subtheorems') == len(sub['examples']) > 0 and facts.get('library_lemmas', 0) > 0, 'C29 the dataset facts count the sub-theorems (subproblems.json: "examples") and the library lemmas: %s' % facts)
        facts_runs = os.path.join(tmp, 'facts_runs'); os.makedirs(facts_runs); json.dump({'name': 'facts', 'dataset': ddir}, open(os.path.join(facts_runs, 'PROFILE.json'), 'w'))
        fo = os.path.join(tmp, 'facts_out'); REP.main([facts_runs, '--out', fo, '--no-figures']); F = json.load(open(os.path.join(fo, 'numbers.json')))
        bank = json.load(open(os.path.join(ddir, 'bank.json')))['problems']
        expect(F['bank_problems'] == len(bank) and F['bank_proved'] + F['bank_statement_only'] == len(bank) and F['bank_train'] + F['bank_dev'] + F['bank_test'] == len(bank)
               and F['bank_with_helpers'] == sum(1 for p in bank if p.get('helpers')) and F['bank_amc'] == sum(1 for p in bank if 'in the manner of the AMC' in p['source']) > 0,
               'C30 a report made of the dataset alone states the facts of the bank: %s' % {k: v for k, v in F.items() if k.startswith('bank_')})
        try:
            import matplotlib  # noqa: F401
            figs = __import__('exp_figures').make_all(C, os.path.join(out, 'figures'))
            expect('fig_budget' in figs and os.path.getsize(os.path.join(out, 'figures', 'fig_budget.png')) > 2000, 'C27 the budget figure is drawn (%s)' % sorted(figs))
        except ImportError:
            expect(True, 'C27 (matplotlib missing: figures not tested)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the small world
CFG = {'d': 32, 'layers': 1, 'heads': 2, 'ff': 64, 'dropout': 0.0}


class Mini:
    """the smoke profile's 17 problems as a dataset built by dataset.py (--only), verified, and the world it describes (a few seconds to load instead of a minute); `fresh()` builds another world of the same dataset"""
    def __init__(self):
        import dataset as DS, ml_world as W, ml_data as D, ml_train as T
        import gym_env as G
        self.DS, self.W, self.D, self.T, self.G = DS, W, D, T, G
        self.tmp = tempfile.mkdtemp(prefix='s51w_'); self.ddir = os.path.join(self.tmp, 'data')
        t0 = time.time()
        if os.environ.get('S51_DATA'):                                          # (development: reuse a dataset built earlier from the same problems instead of building one)
            self.ddir = os.environ['S51_DATA']
            bj = json.load(open(os.path.join(self.ddir, 'bank.json')))
            self.info = {'problems': len(bj['problems']), 'demonstrations': sum(1 for p in bj['problems'] if 'demonstration' in p)}
        else:
            self.info = DS.build(self.ddir, only=set(SMOKE), lib_max=12)
        trace('dataset %.1f s' % (time.time() - t0)); t0 = time.time()
        expect(set(SMOKE) <= {p['name'] for p in json.load(open(os.path.join(self.ddir, 'bank.json')))['problems']}, 'fixture: the small dataset holds every problem of the smoke list')
        self.n_verified = DS.verify(self.ddir)
        trace('dataset verified %.1f s' % (time.time() - t0))
        self.args = argparse.Namespace(data=self.ddir, only='')
        self.bk, self.ctx, self.eng = W.load(self.args)
        self.eng_plain = G.Engine(self.bk.w)                                  # another configuration of the engine: the bank's own, the first language's kinds
        self.bank = {p['name']: p for p in json.load(open(os.path.join(self.ddir, 'bank.json')))['problems']}
        self.proved = [n for n in sorted(self.bk.placed) if self.bk.placed[n].proved]
        import bank_split as BS
        self.BS = BS
        self.torch = None

    def fresh(self):
        return self.W.load(self.args)

    def play(self, name, upto=None, eng=None):
        """the reference proof of `name` (its first `upto` actions) through a tracking episode"""
        import gym_acts as A
        G = self.G; eng = eng or self.eng
        pl = self.bk.placed[name]; steps = self.bank[name]['demonstration']['steps']
        ep = G.Episode(eng, pl.stmt, name, premises=pl.premises('prefix'), track=True)
        for st in steps[:upto]:
            menu = ep.menu()
            ep.step(A.from_portable(eng, menu if st['kind'] in A.MENU_KINDS else A.NO_MENU, st))
        return ep

    def tiny_net(self, ctx=None, seed=0):
        import torch
        torch.manual_seed(seed)
        net = self.T.make_net((ctx or self.ctx).bags, CFG); net.eval()
        return net

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def candidates_of(fx, split='train', max_len=30):
    """[(problem, Found)]: the theorems the reference proofs of the problems of a split contain, longest first within a problem"""
    import ml_her as H
    out = []
    for n in fx.proved:
        if fx.BS.split_of(n) != split: continue
        ep = fx.play(n)
        h = H.Harvest(fx.eng, fx.bk.placed[n].premises('prefix'), n, max_len=max_len, max_n=10 ** 6, max_candidates=10 ** 6); h.episode(ep)
        out += [(n, f) for f in h.found()]
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. hindsight
def sec_D(c):
    import ml_her as H, ml_data as D, ml_agents as AG, gym_tree as GT, gym_premises as PM, gym_acts as ACTS, gym_env as G
    fx = c['mini']; bk = fx.bk; eng = fx.eng; eng2 = fx.eng_plain
    expect(fx.n_verified > 0 and fx.info['demonstrations'] == len(fx.proved) and fx.info['problems'] == len(bk.placed), 'D0 the small dataset verifies: %d proofs replayed (%d problems, %d proved)' % (fx.n_verified, fx.info['problems'], len(fx.proved)))
    tot = collections.Counter()
    for name in fx.proved:
        ep = fx.play(name); tree = ep.tree; pr = bk.placed[name].premises('prefix')
        h = H.Harvest(eng, pr, name, max_len=30, max_n=10 ** 6, max_candidates=10 ** 6); h.episode(ep); found = h.found()
        ex = tree.examples(verify=True, max_len=30, kinds=('solved',))
        a = {e.text: e.length for e in ex}; b = {f.text: f.length for f in found}
        expect(a == b, 'D1 %s: the Harvest keeps exactly the theorems the proof tree says were proved (%d vs %d)' % (name, len(a), len(b)))
        root = eng.w.show(bk.placed[name].stmt)
        expect(all(f.text != root for f in found), 'D2 %s: the problem itself is not one of its hindsight theorems' % name)
        expect(not h.stats.get('did not replay') and not h.stats.get('settled at once') and h.stats['kept'] == len(found) + h.stats.get('duplicate', 0), 'D3 %s: nothing was dropped silently (%s)' % (name, dict(h.stats)))
        for f in found:
            tot['found'] += 1
            ok = GT.replays(eng2, f.stmt, f.steps, f.premises) is not None
            expect(ok, 'D4 %s: a hindsight theorem replays in a fresh engine of another configuration (%s)' % (name, f.text[:60]))
            expect(1 <= f.length <= 30 and f.digest == GT.digest(f.text), 'D5 %s: length and digest of a hindsight theorem' % name)
            if f.code is not None:
                try: again = GT.statement_from_code(eng2, f.code, f.premises)
                except Exception: again = None
                expect(again is not None and again.n == f.stmt.n, 'D6 %s: the code of a hindsight theorem builds the statement back' % name)
            d = f.to_dict()
            expect(json.dumps(json.loads(json.dumps(d)), sort_keys=True) == json.dumps(d, sort_keys=True) and d['verified'] is True and d['length'] == f.length, 'D7 %s: a hindsight theorem is plain data' % name)
        expect([f.length for f in found] == sorted((f.length for f in found), reverse=True), 'D8 %s: longest proof first' % name)
        ties = [f for f in found]
        expect(all((ties[i].length > ties[i + 1].length) or (ties[i].digest <= ties[i + 1].digest) for i in range(len(ties) - 1)), 'D9 %s: ties by statement digest' % name)
    expect(tot['found'] >= 40, 'D10 the small world yields hindsight theorems (%d)' % tot['found'])
    # the filters
    name = 'pow_sub_dvd'; ep = fx.play(name); pr = bk.placed[name].premises('prefix'); tree = ep.tree
    lens = sorted((len(tree.subtree_steps(n)) for n in tree.solved_nodes() if n.parent is not None), reverse=True)
    h = H.Harvest(eng, pr, name, max_len=3, max_n=100, max_candidates=100); h.episode(ep)
    expect(all(f.length <= 3 for f in h.found()) and h.stats['too long'] == sum(1 for L in lens if L > 3) and len(h.found()) == sum(1 for L in lens if L <= 3), 'D11 max_len: longer subtrees are counted as too long, shorter ones kept (%s)' % dict(h.stats))
    h = H.Harvest(eng, pr, name, max_len=30, min_len=2, max_n=100, max_candidates=100); h.episode(ep)
    expect(all(f.length >= 2 for f in h.found()) and h.stats['too short'] == sum(1 for L in lens if L < 2) and len(h.found()) == sum(1 for L in lens if L >= 2), 'D12 min_len: one-action theorems are counted as too short')
    h = H.Harvest(eng, pr, name, max_len=30, max_n=3, max_candidates=100); h.episode(ep)
    expect([f.length for f in h.found()] == lens[:3], 'D13 max_n keeps the three longest proofs (%s vs %s)' % ([f.length for f in h.found()], lens[:3]))
    h = H.Harvest(eng, pr, name, max_len=30, max_n=100, max_candidates=2); h.episode(ep)
    expect(h.stats['kept'] == 2 and [f.length for f in h.found()] == lens[:2], 'D14 max_candidates tries only the largest candidates (%s)' % dict(h.stats))
    h = H.Harvest(eng, pr, name, max_len=30, max_n=100, max_candidates=100); h.episode(ep); n1 = len(h.found()); h.episode(fx.play(name))
    expect(len(h.found()) == n1 and h.stats['duplicate'] == n1, 'D15 the same theorem found twice is kept once and the repeat is counted (%s)' % dict(h.stats))
    # unfinished episodes give theorems too: the first n actions of a proof
    ep3 = fx.play(name, upto=5); h = H.Harvest(eng, pr, name, max_len=30, max_n=100, max_candidates=100); h.episode(ep3)
    expect(len(h.found()) >= 1 and all(GT.replays(eng2, f.stmt, f.steps, f.premises) is not None for f in h.found()), 'D16 an unfinished attempt still yields verified theorems (%d)' % len(h.found()))
    h = H.Harvest(eng, pr, name); h.episode(fx.play(name, upto=0))
    expect(h.found() == [] and not h.stats, 'D17 an attempt that did nothing yields nothing')
    # tampering: a Harvest believes only what replays
    found = H.Harvest(eng, pr, name, max_len=30, max_n=100, max_candidates=100)
    found.episode(ep)
    best = found.found()[0]
    for what, steps in (('an incomplete proof', best.steps[:-1]), ('a proof of something else', [{'kind': 'RING'}]), ('no proof', []), ('a repeated proof', best.steps + best.steps)):
        h = H.Harvest(eng, pr, name)
        h._make(best.stmt, steps, {'via': 'test'})
        expect(h.found() == [] and h.stats['did not replay'] + h.stats['settled at once'] == 1, 'D18 %s is not a theorem (%s)' % (what, dict(h.stats)))
    # claims: records of the manager-worker agent's worker
    rec = {'solved': True, 'statement_code': best.to_dict()['statement_code'], 'steps': best.steps, 'premises': best.to_dict()['premises']}
    h = H.Harvest(eng, pr, name, max_len=30); h.claims([rec])
    expect(len(h.found()) == 1 and h.found()[0].text == best.text and h.found()[0].origin['via'] == 'claim', 'D19 a claim the worker proved is a hindsight theorem')
    h = H.Harvest(eng, pr, name, max_len=30); h.claims([dict(rec, solved=False), dict(rec, steps=[]), dict(rec, statement_code=None)])
    expect(h.found() == [] and not h.stats, 'D20 a claim that was not proved, has no proof or no code is ignored')
    h = H.Harvest(eng, pr, name, max_len=30); h.claims([dict(rec, statement_code=[['zz', 7]])])
    expect(h.found() == [] and h.stats['claim not sayable'] == 1, 'D21 a claim whose code says nothing is counted (%s)' % dict(h.stats))
    h = H.Harvest(eng, pr, name, max_len=best.length - 1); h.claims([rec])
    expect(h.found() == [] and h.stats['too long'] == 1, 'D22 a claim with a long proof is counted')
    h = H.Harvest(eng, pr, name, max_len=30); h.claims([dict(rec, steps=best.steps[:-1])])
    expect(h.found() == [] and h.stats['did not replay'] + h.stats['settled at once'] == 1 and h.premises == pr, 'D23 a claim whose proof does not replay is dropped, and the premises of the Harvest are restored')
    # from hindsight theorems to training steps
    steps = H.steps_of(fx.ctx, eng, best)
    expect(len(steps) == best.length and all(isinstance(x, D.Step) for x in steps), 'D24 the proof of a hindsight theorem decomposes into one training step per action (%d of %d)' % (len(steps), best.length))
    bad = copy.copy(best); bad.steps = [{'kind': 'RING'}]
    expect(H.steps_of(fx.ctx, eng, bad) == [], 'D25 a proof that cannot be decomposed gives no steps')
    # the provers hand their attempts to the Harvest (every theorem it collects replays; the yields depend on the network, not asserted)
    net = fx.tiny_net()
    for spec in ('uniform', 'sample', 'ida', 'mcts'):
        s = AG.make_solver(spec, eng, net if spec in AG.NEEDS_NET else None, fx.ctx if spec in AG.NEEDS_NET else None)
        for name in ('pow_sub_dvd', 'imo1959_p1'):
            pr = bk.placed[name].premises('prefix'); h = H.Harvest(eng, pr, name, max_len=20, max_n=8)
            row = s.solve(bk.placed[name].stmt, pr, 60, seed=1, name=name, harvest=h)
            fs = h.found()
            expect('error' not in row and all(GT.replays(eng2, f.stmt, f.steps, pr) is not None and f.text != eng.w.show(bk.placed[name].stmt) for f in fs) and not h.stats.get('did not replay'),
                   'D26 %s on %s: %d theorems collected on the way, every one replays (%s)' % (spec, name, len(fs), dict(h.stats)))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. the library
def sec_E(c):
    import ml_lemmas as L, ml_her as H, gym_tree as GT, gym_premises as PM, gym_acts as ACTS, recheck as RC
    fx = c['mini']; BS = fx.BS
    bk, ctx, eng = fx.fresh(); w = bk.w; k = w.k
    fx2 = copy.copy(fx); fx2.bk, fx2.ctx, fx2.eng = bk, ctx, eng
    cands = candidates_of(fx2, 'train')
    lib = fx.W.make_library(bk, eng, ctx, fx.ddir, max_lemmas=50)
    held = fx.W.split_names(bk)
    expect(set(lib.held_out) == set(held['dev'] + held['test']) and len(lib.forbid) >= len(bk.placed), 'E0 the library forbids the statements of every problem and holds out dev and test (%d held out, %d forbidden)' % (len(lib.held_out), len(lib.forbid)))
    lo = lib.lo
    expect(lo == k.num_globals() and lib.library_end == w.gid[__import__('bank').LIBRARY_LIMIT] and lib.library_end <= lo, 'E1 the block of lemmas begins after everything in the world (%d), the library ends at %d' % (lo, lib.library_end))
    recs = []; why = collections.Counter()
    for n, f in cands:
        r, reason = lib.add(f.to_dict(), 'train', n)
        if r is not None: recs.append((n, f, r))
        else: why[reason] += 1
    expect(len(recs) == 10 and why == {'too short': 13, 'does not replay from the library': 6} and dict(lib.rejected) == dict(why), 'E2 admission on the small world: 10 lemmas, rejected %s (got %d, %s)' % (dict(why), len(recs), dict(why)))
    expect(lib.n_installed == 10 and lib.hi == lo + 10 and [r.name for _, _, r in recs] == ['boot_%04d' % i for i in range(10)], 'E3 names and ids are consecutive')
    for i, (n, f, r) in enumerate(recs):
        gid = w.gid[r.name]
        _, ty, val, cf, ci, ctors, rec_ = k.global_info(gid)
        stmt = GT.statement_from_code(eng, r.code, lib.lemma_premises(r.n_before))
        expect(r.status == 'ok' and gid == r.gid == lo + i and val is not None and ty == stmt.n, 'E4 %s: a global of the world with a value, whose type is the statement (declared through the kernel)' % r.name)
        expect(BS.split_of(r.source) == 'train' and r.digest not in lib.forbid and r.length == f.length and r.n_before == i, 'E5 %s: it comes from a train problem, says nothing the benchmark says, and was proved under the %d lemmas before it' % (r.name, i))
    expect(len({r.digest for _, _, r in recs}) == 10, 'E6 no two lemmas say the same')
    # the window of premises
    sees = lambda p_, g: (p_.sees(g) if hasattr(p_, 'sees') else 0 <= g < p_)
    base = bk.placed['pow_sub_dvd'].premises('prefix')
    expect(lib.lemma_premises(0) == lib.library_end and lib.premises(base, 0) == PM.of(base), 'E7 with no lemma installed a task keeps its premises')
    pw = lib.premises(base); pw3 = lib.premises(base, 3)
    expect(pw.upto == lib.hi and pw3.upto == lo + 3 and all(sees(pw, g) for g in range(lo, lib.hi)) and all(sees(pw3, g) for g in range(lo, lo + 3)) and not sees(pw3, lo + 3), 'E8 a task sees the lemmas installed when it started and not the later ones')
    expect(all(sees(pw, g) == sees(PM.of(base), g) for g in range(0, lo)), 'E9 widening adds the block and changes nothing below it')
    raises('E10 premises that reach into the block of lemmas are refused', lambda: lib.premises(lo + 2), ValueError)
    for _, _, r in recs:
        wnd = lib.lemma_premises(r.n_before)
        expect([j for j in range(10) if sees(wnd, lo + j)] == list(range(r.n_before)) and all(sees(wnd, g) for g in range(lib.library_end)) and not any(sees(wnd, g) for g in range(lib.library_end, lo)),
               'E11 %s: its proof saw the library and the %d lemmas before it, nothing of the bank and no lemma after' % (r.name, r.n_before))
    all_prem = lib.premises(lib.library_end)
    citable = []
    for _, _, r in recs:
        stmt = GT.statement_from_code(eng, r.code, lib.lemma_premises(r.n_before)); cite = [{'kind': 'APPLY', 'source': ['c', r.gid, 0]}]
        if GT.replays(eng, stmt, cite, all_prem) is not None: citable.append((r, stmt, cite))
    expect(len(citable) >= 5, 'E12 a lemma proves itself by `apply` once it is visible (%d of 10 do, in one action)' % len(citable))
    expect(all(GT.replays(eng, stmt, cite, lib.lemma_premises(r.n_before)) is None for r, stmt, cite in citable), 'E13 within the window of its own proof the same action fails: a proof cannot cite the lemma it proves')
    r0, st0, cite0 = citable[0]
    expect(lib.uses(cite0) == {r0.gid} and lib.uses([{'kind': 'APPLY', 'source': ['c', 5, 0]}]) == set() and lib.uses([{'kind': 'RING'}, {'kind': 'USE', 'codes': [[['c', r0.gid, 0]]]}]) == {r0.gid}, 'E14 which lemmas a proof cites')
    # the second opinion
    t0 = time.time()
    roots = []
    for _, _, r in recs:
        _, ty, val, *_ = k.global_info(r.gid); roots += [val, ty]
    so = RC.SecondOpinion(k, needed_by=roots)
    ok_all = all(so.accepts(k.global_info(r.gid)[2], k.global_info(r.gid)[1])[0] for _, _, r in recs)
    expect(ok_all, 'E15 the independent checker accepts the proof of every lemma against its statement')
    r_c, st_c, cite_c = max(citable, key=lambda t: t[0].n_before)
    term = ACTS.replay_portable(eng, st_c, cite_c, all_prem)
    so2 = RC.SecondOpinion(k, needed_by=[term, st_c])
    ok, msg = so2.accepts(term, st_c)
    expect(ok, 'E16 the independent checker accepts a proof that cites %s of the library (%s)' % (r_c.name, msg))
    expect(so2.problems() == [], 'E17 the globals that proof needs re-check without a complaint')
    trace('second opinion %.1f s' % (time.time() - t0))
    # determinism: another process of the same world installs the same block
    payload = lib.records_json()
    expect(json.loads(json.dumps(payload)) == payload, 'E18 the records are plain data')
    bk_b, ctx_b, eng_b = fx.fresh(); lib_b = fx.W.make_library(bk_b, eng_b, ctx_b, fx.ddir, max_lemmas=50)
    n = lib_b.install_records(payload)
    expect(n == 10 and lib_b.block_digest() == lib.block_digest() and [r.gid for r in lib_b.ok_records()] == [r.gid for r in lib.ok_records()] and lib_b.records_json() == payload, 'E19 a second world installs the same lemmas at the same ids (block %s)' % lib.block_digest())
    expect(lib_b.install_records(payload) == 0 and lib_b.n_installed == 10, 'E20 installing records already there does nothing')
    # corrupted records
    bk_c, ctx_c, eng_c = fx.fresh(); lib_c = fx.W.make_library(bk_c, eng_c, ctx_c, fx.ddir, max_lemmas=50)
    bad = copy.deepcopy(payload); bad[1]['steps'] = bad[0]['steps']; bad[2]['statement_code'] = bad[3]['statement_code']
    lib_c.install_records(bad)
    st = {r.name: r.status for r in lib_c.records}
    expect(st['boot_0001'] == 'failed' and st['boot_0002'] == 'failed' and st['boot_0000'] == 'ok', 'E21 a record with the proof of another lemma, or the statement of another, is not installed (%s)' % st)
    expect(lib_c.rejected['does not replay from the library'] == 2 and lib_c.n_installed == 8, 'E22 a record that does not replay is counted, the others are installed (%d)' % lib_c.n_installed)
    for r in lib_c.ok_records():
        orig = [x for _, _, x in recs if x.name == r.name][0]
        _, ty, val, *_ = bk_c.w.k.global_info(r.gid)
        stmt = GT.statement_from_code(eng_c, r.code, lib_c.lemma_premises(r.n_before))
        expect(ty == stmt.n and r.digest == orig.digest, 'E23 %s: what is installed after a corrupted record still has the statement it was given' % r.name)
    bk_d, ctx_d, eng_d = fx.fresh(); lib_d = fx.W.make_library(bk_d, eng_d, ctx_d, fx.ddir, max_lemmas=50)
    raises('E24 records that arrive out of order are an error', lambda: lib_d.install_records(payload[3:]), ValueError)
    # the file
    path = os.path.join(fx.tmp, 'lemmas.json'); lib.save(path, extra={'note': 'x'})
    d = json.load(open(path))
    expect(d['format'] == L.FORMAT and d['block_digest'] == lib.block_digest() and d['library_end'] == lib.library_end and len(d['records']) == 10 and d['note'] == 'x', 'E25 the lemma file says its format, library, block and records')
    bk_e, ctx_e, eng_e = fx.fresh(); lib_e = fx.W.make_library(bk_e, eng_e, ctx_e, fx.ddir, max_lemmas=50)
    expect(lib_e.load(path) == 10 and lib_e.block_digest() == lib.block_digest(), 'E26 a lemma file loads to the same block')
    for what, mod in (('another format', {'format': 99}), ('another library', {'library_end': lib.library_end + 1})):
        p2 = os.path.join(fx.tmp, 'bad.json'); json.dump(dict(d, **mod), open(p2, 'w'))
        lib_f = fx.W.make_library(bk_e, eng_e, ctx_e, fx.ddir, max_lemmas=50)
        raises('E27 a lemma file of %s is refused' % what, lambda: lib_f.load(p2), ValueError)
    open(os.path.join(fx.tmp, 'cut.json'), 'w').write(open(path).read()[:200])
    raises('E28 a truncated lemma file is an error', lambda: fx.W.make_library(bk_e, eng_e, ctx_e, fx.ddir).load(os.path.join(fx.tmp, 'cut.json')), ValueError)
    c['lemma_file'] = path; c['lemma_block'] = lib.block_digest(); c['lemma_names'] = [r.name for _, _, r in recs]
    sec_E_reasons(c, fx)


def sec_E_reasons(c, fx):
    """one test per reason a candidate is refused, on a world of its own (no lemma is ever admitted into it but by the controls)"""
    import ml_lemmas as L, gym_tree as GT, bank as B
    BS = fx.BS
    bk, ctx, eng = fx.fresh(); w = bk.w; k = w.k
    fx3 = copy.copy(fx); fx3.bk, fx3.ctx, fx3.eng = bk, ctx, eng
    cands = candidates_of(fx3, 'train')
    good = [f for n, f in cands if f.length >= 2][0]
    gd = good.to_dict()
    prefixes = itertools.count()
    def mk(**kw):
        return L.LemmaLibrary(bk, eng, ctx, forbid=kw.pop('forbid', ()), held_out=kw.pop('held_out', ()), name_prefix='lib%d_' % next(prefixes), **kw)
    lib = mk()
    for split in ('dev', 'test'):
        r, why = lib.add(gd, split); expect(r is None and why == 'not train', 'E30 a theorem of a %s problem is never a lemma' % split)
    r, why = lib.add({'steps': gd['steps']}, 'train'); expect(why == 'no code or no proof', 'E31 no code')
    r, why = lib.add({'statement_code': gd['statement_code'], 'steps': []}, 'train'); expect(why == 'no code or no proof', 'E32 no proof')
    one = [f for n, f in cands if f.length == 1][0]
    r, why = lib.add(one.to_dict(), 'train'); expect(why == 'too short', 'E33 a single action is not a lemma')
    r, why = mk(max_len=good.length - 1).add(gd, 'train'); expect(why == 'too long', 'E34 a proof longer than max_len')
    r, why = mk(max_tokens=1).add(gd, 'train'); expect(why == 'statement too long', 'E35 a statement longer than max_tokens')
    r, why = mk(max_lemmas=0).add(gd, 'train'); expect(why == 'library full', 'E36 a full library')
    r, why = mk(forbid={gd['statement_digest']}).add(gd, 'train'); expect(why == 'known statement', 'E37 a statement of the forbidden set')
    r, why = lib.add(dict(gd, statement_code=[['zz', 1]]), 'train'); expect(why == 'not sayable with the library', 'E38 a code that says nothing')
    r, why = lib.add(dict(gd, statement_code=[['c', 10 ** 7, 0]]), 'train'); expect(why == 'not sayable with the library', 'E39 a code that names a global that does not exist')
    r, why = lib.add(dict(gd, steps=[{'kind': 'RING'}, {'kind': 'RING'}]), 'train'); expect(why == 'does not replay from the library', 'E40 a proof that does not prove the statement')
    expect(lib.n_installed == 0 and k.num_globals() == lib.lo, 'E41 none of that declared anything (%d globals)' % k.num_globals())
    # a statement the world already proves: the type of a theorem of the library
    thm = None
    for gid in range(lib.library_end - 1, 0, -1):
        _, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        if val is None or k.is_inductive(gid): continue
        ok, sort = k.infer_open(ty, [])
        if ok and sort == k.mk_type(0): thm = gid; break
    prem0 = lib.library_end
    code = GT.statement_code(eng, w.T(k.global_info(thm)[1]), prem0)
    r, why = mk(min_len=1).add({'statement_code': [list(t) for t in code], 'steps': [{'kind': 'APPLY', 'source': ['c', thm, 0]}]}, 'train')
    expect(why == 'already a theorem of the world', 'E42 a statement the library already proves (%s)' % w.k.global_info(thm)[0])
    # the meaning test: a held-out problem is never one `apply` of a lemma away.  The statement of a bank problem is a theorem of the world already; its variants are not
    import bank_lib
    syms = bank_lib._syms(w); Z, Eq = syms.Z, syms.Eq
    with w.nota.using(Z):
        variants = {'commuted': w.forall_([Z, Z], lambda a_, c_: Eq((c_ + a_) * (a_ - c_), a_ ** 2 - c_ ** 2)),
                    'with the binders swapped': w.forall_([Z, Z], lambda c_, a_: Eq((a_ + c_) * (a_ - c_), a_ ** 2 - c_ ** 2))}
    held = bk.placed['ident_diff_squares']; prem = held.premises('prefix')
    expect(variants['commuted'].n != held.stmt.n and variants['with the binders swapped'].n != held.stmt.n, 'E43 the variants are other statements than the problem')
    cands_v = {}
    for nm, v in variants.items():
        code = GT.statement_code(eng, v, prem)
        cands_v[nm] = {'statement_code': [list(t) for t in code], 'steps': [{'kind': 'RING'}]}
        r, why = mk(held_out=['ident_diff_squares'], min_len=1).add(cands_v[nm], 'train')
        expect(r is None and why == 'says the same as a held-out problem', 'E44 a statement that says what a held-out problem says (%s) is refused (%s)' % (nm, why))
    code = GT.statement_code(eng, held.stmt, prem)
    r, why = mk(held_out=['ident_diff_squares'], min_len=1).add({'statement_code': [list(t) for t in code], 'steps': [{'kind': 'RING'}]}, 'train')
    expect(why == 'already a theorem of the world', 'E45 the problem\'s own statement is a theorem of the world (%s)' % why)
    r, why = mk(held_out=['ineq_am_gm_two'], min_len=1).add(cands_v['commuted'], 'train')
    expect(why is None and r.status == 'ok', 'E46 the same candidate is a lemma when the problem it resembles is not held out (%s)' % why)
    # the library never grows past max_lemmas
    lib2 = mk(max_lemmas=2); got = 0
    for n, f in cands:
        r, why = lib2.add(f.to_dict(), 'train')
        got += r is not None
    expect(got == 2 and lib2.rejected['library full'] >= 1 and lib2.n_installed == 2, 'E47 the library stops at max_lemmas (%d admitted, rejected %s)' % (got, dict(lib2.rejected)))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. goals
def sec_F(c):
    import ml_goals as GL, gym_tree as GT, gym_premises as PM
    fx = c['mini']; bk = fx.bk; BS = fx.BS
    bank, subs = GL.train_goals(bk, fx.ddir)
    train = sorted(n for n in bk.placed if BS.split_of(n) == 'train')
    expect([n for n, L in bank] == train, 'F1 the goals of the bank are the train problems of the world (%d)' % len(train))
    expect(all((L is None) == (not bk.placed[n].proved) for n, L in bank) and all(L == fx.bank[n]['demonstration']['length'] for n, L in bank if L is not None), 'F2 each train problem comes with the length of its reference proof, or None')
    allsubs = json.load(open(os.path.join(fx.ddir, 'subproblems.json')))['examples']
    expect(any(e['split'] != 'train' for e in allsubs) and len(subs) < len(allsubs), 'F3 the sub-theorem file holds dev and test examples ...')
    expect(all(e['split'] == 'train' and BS.split_of(e['problem']) == 'train' and e['length'] <= 30 and e.get('statement_code') is not None for e in subs), 'F4 ... and the learners are offered train ones only')
    expect({e['id'] for e in subs} == {e['id'] for e in allsubs if e['split'] == 'train' and e['length'] <= 30 and e.get('statement_code') is not None}, 'F5 all of them (the train sub-theorems of at most 30 actions)')
    synth = [{'id': 'synth#%d' % i, 'code': [['c', 1, 0]], 'n_steps': 3 + i % 5} for i in range(20)]
    G1 = GL.Goals(bk, bank, subs, synth=synth, p_sub=0.25, p_synth=0.1, p_hard=0.1)
    rng = np.random.default_rng(7); tasks = [G1.sample(rng) for _ in range(3000)]
    cnt = collections.Counter(t['source'] for t in tasks)
    expect(abs(cnt['sub'] / 3000 - 0.25) < 0.03 and abs(cnt['synth'] / 3000 - 0.10) < 0.02 and cnt['bank'] + cnt['sub'] + cnt['synth'] == 3000, 'F6 the sources are drawn with the probabilities asked for (%s)' % dict(cnt))
    expect(all(t['options']['problem'] in train for t in tasks if t['source'] == 'bank'), 'F7 a bank task is a train problem')
    expect(all(t['options']['goal']['statement_code'] for t in tasks if t['source'] == 'sub') and {t['name'] for t in tasks if t['source'] == 'sub'} <= {e['id'] for e in subs}, 'F8 a sub-theorem task is a train sub-theorem')
    expect(all(not t['name'].startswith(tuple(n for n in bk.placed if BS.split_of(n) != 'train')) for t in tasks if t['source'] == 'bank'), 'F9 no dev or test name appears')
    expect(all(json.loads(json.dumps(t)) == json.loads(json.dumps(t)) for t in tasks[:200]) and json.dumps(tasks[:200]), 'F10 a task is plain data (a worker process gets it as JSON)')
    rng2 = np.random.default_rng(7)
    expect([G1.sample(rng2) for _ in range(300)] == tasks[:300], 'F11 sampling is reproducible from the rng')
    G2 = GL.Goals(bk, bank, subs, synth=(), p_sub=0.0, p_synth=0.5, p_hard=0.0)
    expect(G2.p_synth == 0.0 and G2.p_sub == 0.0 and all(t['source'] == 'bank' for t in (G2.sample(np.random.default_rng(1)) for _ in range(200))), 'F12 a source with nothing in it is never drawn')
    expect([G1._budget(L) for L in (1, 2, 5, 10, 30, 100)] == [8, 10, 19, 34, 40, 40], 'F13 the step budget is 3 x the reference length + 4, between 8 and max_actions (%s)' % [G1._budget(L) for L in (1, 2, 5, 10, 30, 100)])
    G3 = GL.Goals(bk, [('a', None), ('b', None)], (), p_hard=0.3)
    expect(G3.easy == ['a', 'b'] and G3.hard == [] and G3.p_hard == 0.0, 'F14 with no problem of a known short proof the hard ones are the pool')
    raises('F15 no train problem is an error', lambda: GL.Goals(bk, [], ()), ValueError)
    # resolving a task in the world
    sub = [t for t in tasks if t['source'] == 'sub'][0]
    name, stmt, prem = GL.resolve(bk, fx.eng, None, sub)
    ex = [e for e in subs if e['id'] == sub['name']][0]
    expect(name == sub['name'] and prem == PM.of(ex['premises']), 'F16 a sub-theorem task resolves under the premises of its record')
    expect(GT.digest(bk.w.show(stmt)) == ex['statement_digest'], 'F17 ... to the statement of its record, exactly')
    bt = [t for t in tasks if t['source'] == 'bank'][0]
    name, stmt, prem = GL.resolve(bk, fx.eng, None, bt)
    expect(stmt == bk.placed[bt['name']].stmt and prem == bk.placed[bt['name']].premises(bt['options']['regime']), 'F18 a bank task resolves to the problem and the premises of its regime')
    class Stub:
        def premises(self, p): return PM.of(PM.Premises(bk.k.num_globals(), [(10, 20)]))
    n1, s1, p1 = GL.resolve(bk, fx.eng, Stub(), bt); n2, s2, p2 = GL.resolve(bk, fx.eng, Stub(), sub)
    expect(p1 == Stub().premises(None) and p2 == Stub().premises(None), 'F19 a library widens the premises of every task, bank problem or sub-theorem')
    differing = [n for n in bk.placed if bk.placed[n].problem.strict and bk.placed[n].premises('strict') != bk.placed[n].premises('prefix')]
    plain = [n for n in bk.placed if n not in differing]
    rng3 = np.random.default_rng(0)
    expect(all(GL.regime_for(bk, n, rng3) == 'prefix' for n in plain for _ in range(3)), 'F20 a problem with one task is attacked in the prefix regime')
    if differing: expect({GL.regime_for(bk, differing[0], rng3) for _ in range(60)} == {'prefix', 'strict'}, 'F21 a problem with two regimes is attacked in both (%s)' % differing[0])
    xs = list(range(10))
    expect(all(sum(GL.chunks(xs, k), []) == xs and len(GL.chunks(xs, k)) <= k for k in (1, 3, 4, 10, 20)) and GL.chunks([], 3) == [], 'F22 chunks partition a list into at most n consecutive parts')
    items = GL.dev_items(bk, ['ident_diff_squares', 'imo1959_p1'], ('prefix', 'strict'))
    expect(all(r in ('prefix', 'strict') for _, r in items) and ('ident_diff_squares', 'prefix') in items and ('ident_diff_squares', 'strict') not in items, 'F23 dev tasks: a regime that offers the same task twice is listed once')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ G. the provers
def sec_G(c):
    import hashlib, torch
    import ml_agents as AG, ml_protocol as PR, ml_evaluate as EV, ml_world as W, ml_train as T, gym_acts as ACTS, gym_env as G
    fx = c['mini']; bk = fx.bk; eng = fx.eng; BS = fx.BS
    name, opts = AG.parse_spec('mcts:value=critic,c_puct=2')
    expect(name == 'mcts' and opts['value'] == 'critic' and opts['c_puct'] == 2 and opts['widen_c'] == AG.DEFAULTS['mcts']['widen_c'] and isinstance(opts['c_puct'], int), 'G1 a spec sets options on top of the defaults, with their types')
    expect(AG.parse_spec('blind:all_kinds=true')[1]['all_kinds'] is True and AG.parse_spec('sample:T=0.5')[1]['T'] == 0.5, 'G2 booleans and floats')
    raises('G3 an unknown solver is an error', lambda: AG.parse_spec('oracle'), ValueError)
    raises('G4 an unknown option is an error', lambda: AG.parse_spec('mcts:depth=3'), ValueError)
    expect(AG.canonical('mcts:c_puct=1.5') == 'mcts' and AG.canonical('sample:T=0.5,max_actions=40') == 'sample:T=0.5' and AG.canonical('mw') == 'mw', 'G5 the canonical spec drops the defaults')
    raises('G6 a solver that needs a network refuses to be made without one', lambda: AG.make_solver('sample', eng), ValueError)
    expect(PR.stable_seed('sample', 0, 'a', 'prefix') == int.from_bytes(hashlib.sha256(b'sample/0/a/prefix').digest()[:8], 'little') >> 1 and PR.stable_seed('x', 1) != PR.stable_seed('x', 2), 'G7 seeds are derived from text, the same in every process')
    rows = [{'solved': True, 'probes': 5}, {'solved': True, 'probes': 50}, {'solved': False, 'probes': 101}, {'solved': True, 'probes': 100}]
    expect(PR.solved_within(rows, 5) == 1 and PR.solved_within(rows, 100) == 3 and PR.solved_within(rows, 4) == 0 and PR.curve(rows, (5, 50, 100)) == {5: 1, 50: 2, 100: 3}, 'G8 solved within B: solved and at most B probes')
    # the protocol: an unsolved search spends one probe more than its budget, and the curve read off the largest budget is the curve of the smaller ones
    blind = AG.make_solver('blind', eng); names = sorted(bk.placed)
    res = {}
    for B in (8, 16, 32):
        res[B] = {n: blind.solve(bk.placed[n].stmt, bk.placed[n].premises('prefix'), B, seed=0, name=n, time_limit=30.0) for n in names}
    expect(all(r['probes'] == B + 1 for B in res for r in res[B].values() if not r['solved']) and all(r['probes'] <= B for B in res for r in res[B].values() if r['solved']), 'G9 an unsolved search spends budget + 1 probes, a solved one at most the budget')
    bad = [(B, n) for B in (8, 16) for n in names if res[B][n]['solved'] != (res[32][n]['solved'] and res[32][n]['probes'] <= B)]
    expect(not bad and any(r['solved'] for r in res[32].values()) and any(not r['solved'] for r in res[32].values()), 'G10 solved within B read off the run with the largest budget is what a run with budget B finds (%s)' % bad[:3])
    expect(all(res[B][n]['probes'] == res[32][n]['probes'] for B in (8, 16) for n in names if res[B][n]['solved']), 'G11 and the probes it needed are the same')
    # evaluate: rows, annotations, resumption, determinism, both regimes, second opinion
    dev_test = [n for n in names if BS.split_of(n) in ('dev', 'test')]
    tasks = EV.tasks_of(bk, dev_test, ('prefix', 'strict'))
    expect(len(tasks) == len(dev_test) + sum(1 for n in dev_test if bk.placed[n].problem.strict and bk.placed[n].premises('strict') != bk.placed[n].premises('prefix')), 'G12 a strict task exists only for a problem whose strict regime differs (%d tasks for %d problems)' % (len(tasks), len(dev_test)))
    out1 = os.path.join(fx.tmp, 'ev1'); out2 = os.path.join(fx.tmp, 'ev2'); quiet = lambda *a: None
    rows1 = EV.evaluate(bk, fx.ctx, eng, [], ['blind', 'uniform'], dev_test, ('prefix', 'strict'), 40, 20.0, out1, progress=quiet)
    expect(len(rows1) == 2 * len(tasks) and sum(1 for r in rows1 if r['solved']) >= 3, 'G13 one row per task and solver (%d rows, %d solved)' % (len(rows1), sum(1 for r in rows1 if r['solved'])))
    fields = ('name', 'regime', 'solved', 'probes', 'seconds', 'steps', 'method', 'seed', 'level', 'split', 'proved', 'origin', 'source', 'family', 'solver')
    expect(all(all(f in r for f in fields) for r in rows1), 'G14 every row has the fields the collector reads')
    expect(all(r['split'] == BS.split_of(r['name']) and r['proved'] == bk.placed[r['name']].proved and r['seed'] == 0 and r['method'] in ('blind', 'uniform') for r in rows1), 'G15 rows are annotated from the bank')
    expect(all((('proof' in r) == r['solved']) and r['probes'] <= 41 and 'error' not in r for r in rows1), 'G16 a solved row carries its proof, an unsolved one does not; no solver raised')
    eng_fresh = G.Engine(bk.w)
    okp = 0
    for r in rows1:
        if r['solved']:
            pl = bk.placed[r['name']]
            try: term = ACTS.replay_portable(eng_fresh, pl.stmt, r['proof'], pl.premises(r['regime'])); okp += term is not None
            except Exception: pass
    expect(okp == sum(1 for r in rows1 if r['solved']), 'G17 the proof of every solved row replays in a fresh engine, checked by the kernel')
    lines = open(os.path.join(out1, 'rows.jsonl')).read().strip().split('\n')
    rows1b = EV.evaluate(bk, fx.ctx, eng, [], ['blind', 'uniform'], dev_test, ('prefix', 'strict'), 40, 20.0, out1, progress=quiet)
    expect(len(rows1b) == len(rows1) and open(os.path.join(out1, 'rows.jsonl')).read().strip().split('\n') == lines, 'G18 the evaluation resumes: a second call adds no row')
    open(os.path.join(out1, 'rows.jsonl'), 'a').write('{"half a li')
    rows1c = EV.evaluate(bk, fx.ctx, eng, [], ['blind', 'uniform'], dev_test, ('prefix', 'strict'), 40, 20.0, out1, progress=quiet)
    expect(len(rows1c) == len(rows1), 'G19 a line cut off by a crash is ignored')
    rows2 = EV.evaluate(bk, fx.ctx, eng, [], ['blind', 'uniform'], dev_test, ('prefix', 'strict'), 40, 20.0, out2, progress=quiet)
    key = lambda r: (r['method'], r['name'], r['regime'])
    a = {key(r): (r['solved'], r['probes'], r.get('proof'), r['steps']) for r in rows1}; b = {key(r): (r['solved'], r['probes'], r.get('proof'), r['steps']) for r in rows2}
    expect(a == b, 'G20 two evaluations give the same results (the baselines are deterministic, the seeded ones are seeded by name)')
    verdicts = EV.recheck_rows(bk, eng, rows1, None, os.path.join(out1, 'recheck.json'), progress=quiet)
    n_solved = sum(1 for r in rows1 if r['solved'])
    expect(len(verdicts) == n_solved and all(v['ok'] for v in verdicts), 'G21 the independent checker accepts every proof an evaluation found (%d of %d)' % (sum(1 for v in verdicts if v['ok']), len(verdicts)))
    rows_bad = [dict(rows1[[i for i, r in enumerate(rows1) if r['solved']][0]])]
    rows_bad[0]['proof'] = [{'kind': 'RING'}, {'kind': 'RING'}, {'kind': 'RING'}]; rows_bad[0]['name'] = [r['name'] for r in rows1 if not r['solved']][0]
    v2 = EV.recheck_rows(bk, eng, rows_bad, None, os.path.join(out1, 'recheck2.json'), progress=quiet)
    expect(len(v2) == 1 and not v2[0]['ok'], 'G22 a proof that does not prove the statement is not accepted')
    # a model: sampling is seeded and reproducible; a seed is part of the row
    net = fx.tiny_net(); s1 = AG.make_solver('sample', eng, net, fx.ctx); s2 = AG.make_solver('sample', eng, net, fx.ctx); g = AG.make_solver('greedy', eng, net, fx.ctx)
    for n in ('pow_sub_dvd', 'ineq_cauchy_two', 'imo1959_p1'):
        pl = bk.placed[n]; pr = pl.premises('prefix')
        r1 = s1.solve(pl.stmt, pr, 60, seed=3, name=n); r2 = s2.solve(pl.stmt, pr, 60, seed=3, name=n)
        expect((r1['solved'], r1['probes'], r1['decisions'], r1.get('proof')) == (r2['solved'], r2['probes'], r2['decisions'], r2.get('proof')), 'G23 %s: sampling with one seed gives one result' % n)
        g1 = g.solve(pl.stmt, pr, 60, seed=0, name=n); g2 = g.solve(pl.stmt, pr, 60, seed=9, name=n)
        expect((g1['solved'], g1['probes'], g1['decisions']) == (g2['solved'], g2['probes'], g2['decisions']), 'G24 %s: the greedy prover does not depend on the seed' % n)
        expect(r1['probes'] <= 60 + 40 and (not r1['solved'] or 'proof' in r1), 'G25 %s: the sampling prover respects its budget (within one episode) and returns a proof when it says solved' % n)
    # the library of a checkpoint
    ck = os.path.join(fx.tmp, 'ck'); os.makedirs(ck, exist_ok=True)
    def save_ck(fname, seed, extra): p = os.path.join(ck, fname); T.save(net, dict(CFG, seed=seed), p, extra); return p
    p_plain = save_ck('plain.pt', 3, {}); p_none = save_ck('none.pt', 4, {'lemmas': None, 'block_digest': None})
    expect(EV.model_info(p_plain) == (3, {}), 'G26 a checkpoint says its training seed')
    expect(EV.library_of([('a', p_plain), ('b', p_none)], 'auto') == ('', {p_plain: None, p_none: None}) and EV.library_of([('a', p_plain)], '')[0] == '' and EV.library_of([('a', p_plain)], 'x.json')[0] == 'x.json', 'G27 no library in the checkpoints: none, or what --lemmas says')
    shutil.copy(c['lemma_file'], os.path.join(ck, 'lemmas.json')) if 'lemma_file' in c else None
    p_lib = save_ck('lib.pt', 5, {'lemmas': 'lemmas.json', 'block_digest': c.get('lemma_block')})
    if 'lemma_file' in c:
        lf, dg = EV.library_of([('a', p_lib)], 'auto')
        expect(lf == os.path.join(ck, 'lemmas.json') and dg == {p_lib: c['lemma_block']}, 'G28 --lemmas auto reads the file next to the checkpoint')
        raises('G29 models with and without a library are not evaluated together', lambda: EV.library_of([('a', p_lib), ('b', p_plain)], 'auto'), SystemExit)
        os.makedirs(os.path.join(ck, 'other'), exist_ok=True); shutil.copy(c['lemma_file'], os.path.join(ck, 'other', 'lemmas.json'))
        T.save(net, dict(CFG, seed=6), os.path.join(ck, 'other', 'm.pt'), {'lemmas': 'lemmas.json', 'block_digest': c.get('lemma_block')})
        raises('G30 models with different libraries are not evaluated together', lambda: EV.library_of([('a', p_lib), ('b', os.path.join(ck, 'other', 'm.pt'))], 'auto'), SystemExit)
    p_gone = save_ck('gone.pt', 7, {'lemmas': 'missing.json', 'block_digest': 'abc'})
    raises('G31 a checkpoint whose lemma file is gone is refused with --lemmas auto', lambda: EV.library_of([('a', p_gone)], 'auto'), SystemExit)
    expect(EV.library_of([('a', p_gone)], 'elsewhere.json')[0] == 'elsewhere.json', 'G32 ... but not when a file is given')
    # the world digest of a dataset
    bad_dir = os.path.join(fx.tmp, 'baddata'); os.makedirs(bad_dir, exist_ok=True)
    bank_json = json.load(open(os.path.join(fx.ddir, 'bank.json'))); bank_json['world_digest'] = '0' * len(bank_json['world_digest'])
    json.dump(bank_json, open(os.path.join(bad_dir, 'bank.json'), 'w'))
    raises('G33 a dataset of another world is refused', lambda: W.load(argparse.Namespace(data=bad_dir, only='')), SystemExit)
    dg, only = W.dataset_world_info(fx.ddir)
    expect(dg == json.load(open(os.path.join(fx.ddir, 'bank.json')))['world_digest'] and only == set(fx.bk.placed), 'G34 a partial dataset says which problems its world holds (%d)' % len(only or ()))
    expect(W.dataset_world_info(os.path.join(fx.tmp, 'nothing')) == (None, None), 'G35 no dataset, no digest')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ H. expert iteration
def run_py(args, timeout=900):
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
    p = subprocess.run([sys.executable] + args, capture_output=True, text=True, timeout=timeout, env=env, cwd=itp_paths.ROOT)
    return p.returncode, p.stdout + p.stderr


def sec_H(c):
    import torch
    fx = c['mini']; root = itp_paths.ROOT
    ei = os.path.join(root, 'src', 'python', 'learner', 'ml_ei.py'); ev = os.path.join(root, 'src', 'python', 'learner', 'ml_evaluate.py')
    init = os.path.join(fx.tmp, 'init.pt'); fx.T.save(fx.tiny_net(), dict(CFG, seed=0), init, {})
    common = ['--init', init, '--seed', '0', '--workers', '0', '--device', 'cpu', '--threads', '1', '--iters', '1', '--episodes', '4', '--probes', '20', '--updates', '1', '--eval-every', '1',
              '--eval-probes', '10', '--actor', 'sample', '--her', '--bootstrap', '--time-limit', '5', '--eval-time-limit', '5']
    # (1) a run that admits no lemma still leaves the file its checkpoints name
    out0 = os.path.join(fx.tmp, 'ei0'); t0 = time.time()
    rc, log = run_py([ei, fx.ddir, '--out', out0] + common + ['--boot-max', '0'])                  # (a library that can hold no lemma: the case that once left the file unwritten)
    trace('ml_ei %.1f s' % (time.time() - t0))
    expect(rc == 0, 'H1 expert iteration with hindsight and bootstrapping runs (exit %d)\n%s' % (rc, log[-600:] if rc else ''))
    if rc == 0:
        lem = os.path.join(out0, 'lemmas.json')
        expect(os.path.exists(lem), 'H2 the lemma file exists after a run (also when no lemma was admitted)')
        d = json.load(open(lem)); fin = json.load(open(os.path.join(out0, 'final.json')))
        expect(d['format'] == 1 and d['records'] == [] and fin['library']['installed'] == 0 and d['block_digest'] == fin['library']['block_digest'], 'H3 the file is empty and agrees with the run\'s own summary (%d lemmas)' % len(d['records']))
        ck = torch.load(os.path.join(out0, 'last.pt'), map_location='cpu', weights_only=False)
        expect(ck['extra'].get('lemmas') == 'lemmas.json' and ck['extra'].get('block_digest') == d['block_digest'], 'H4 the checkpoint names the file and the block')
        expect(fin.get('n_params', 0) > 0 and len(fin['history']) == 1 and all(k in fin['history'][0] for k in ('lemmas', 'lemmas_added', 'boot_rejected', 'her_found', 'dev_solved')), 'H5 final.json carries the parameters and the per-iteration bookkeeping')
        o1 = os.path.join(fx.tmp, 'ev_h1'); t0 = time.time()
        rc, log = run_py([ev, fx.ddir, '--out', o1, '--split', 'dev', '--solver', 'sample', '--model', 'm=' + os.path.join(out0, 'last.pt'), '--lemmas', 'auto', '--probes', '20', '--time-limit', '5', '--recheck'])
        trace('ml_evaluate %.1f s' % (time.time() - t0))
        expect(rc == 0, 'H6 the evaluation of the checkpoint with --lemmas auto runs (exit %d)\n%s' % (rc, log[-600:] if rc else ''))
        if rc == 0:
            rows = [json.loads(l) for l in open(os.path.join(o1, 'rows.jsonl')) if l.strip()]
            meta = json.load(open(os.path.join(o1, 'meta.json')))
            expect(len(rows) >= 3 and all(r['model'] == 'm' and r['seed'] == 0 for r in rows) and meta['lemma_block'] == d['block_digest'] and os.path.exists(os.path.join(o1, 'recheck.json')), 'H7 rows, the library in the meta, the second opinion')
        os.remove(lem)
        o2 = os.path.join(fx.tmp, 'ev_h2')
        rc, log = run_py([ev, fx.ddir, '--out', o2, '--split', 'dev', '--solver', 'sample', '--model', 'm=' + os.path.join(out0, 'last.pt'), '--lemmas', 'auto', '--probes', '20'])
        expect(rc != 0 and 'not there' in log, 'H8 the file gone: the evaluation stops and says why (exit %d)' % rc)
    # (2) a run that starts from lemmas installs them, keeps them and the evaluation loads the same block
    if 'lemma_file' in c:
        out1 = os.path.join(fx.tmp, 'ei1')
        rc, log = run_py([ei, fx.ddir, '--out', out1, '--lemmas', c['lemma_file']] + common)
        expect(rc == 0, 'H9 expert iteration that starts from a lemma file runs (exit %d)\n%s' % (rc, log[-600:] if rc else ''))
        if rc == 0:
            d = json.load(open(os.path.join(out1, 'lemmas.json')))
            expect(len(d['records']) >= 10 and [r['name'] for r in d['records']][:10] == c['lemma_names'] and d['block_digest'] != json.load(open(os.path.join(out0, 'lemmas.json')))['block_digest'] if os.path.exists(os.path.join(out0, 'lemmas.json')) else len(d['records']) >= 10, 'H10 the lemmas it started from are in its file, first and in order')
            o3 = os.path.join(fx.tmp, 'ev_h3')
            rc, log = run_py([ev, fx.ddir, '--out', o3, '--split', 'dev', '--solver', 'sample', '--model', 'm=' + os.path.join(out1, 'last.pt'), '--lemmas', 'auto', '--probes', '20', '--time-limit', '5', '--recheck'])
            expect(rc == 0, 'H11 the evaluation under the bigger library runs (exit %d)\n%s' % (rc, log[-600:] if rc else ''))
            if rc == 0:
                rows = [json.loads(l) for l in open(os.path.join(o3, 'rows.jsonl')) if l.strip()]
                expect(len(rows) >= 3 and all('boot_uses' not in r or isinstance(r['boot_uses'], list) for r in rows) and json.load(open(os.path.join(o3, 'meta.json')))['lemma_block'] == d['block_digest'], 'H12 the rows and the meta carry the library')
            # a model is evaluated with the library it was trained with, not another
            o4 = os.path.join(fx.tmp, 'ev_h4')
            rc, log = run_py([ev, fx.ddir, '--out', o4, '--split', 'dev', '--solver', 'sample', '--model', 'm=' + os.path.join(out1, 'last.pt'), '--lemmas', c['lemma_file'], '--probes', '20', '--time-limit', '5'])
            expect(rc == 0 or 'trained with the library' in log, 'H13 (the file it started from is a prefix of its block: the digest decides) exit %d' % rc)



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ I. the runner and the scripts
def sec_I(c):
    root = itp_paths.ROOT
    scripts = sorted(glob.glob(os.path.join(root, 'scripts', '*.sh')))
    names = [os.path.basename(f) for f in scripts]
    expect(set(names) >= {'env.sh', '00_check_env.sh', '01_build_dataset.sh', '02_generate_synthetic.sh', '03_pretrain_finetune.sh', '04_reinforcement_learning.sh', '05_evaluate.sh', '06_benchmarks.sh', '07_report.sh', 'run_all.sh', 'smoke_test.sh', 'status.sh'}, 'I1 the scripts of the documentation exist (%s)' % names)
    for f in scripts:
        rc = subprocess.run(['bash', '-n', f], capture_output=True).returncode
        expect(rc == 0, 'I2 %s is valid bash' % os.path.basename(f))
        expect(os.access(f, os.X_OK), 'I3 %s is executable' % os.path.basename(f))
    env_sh = os.path.join(root, 'scripts', 'env.sh')
    clean = {k: v for k, v in os.environ.items() if k not in ('PROFILE', 'SEEDS', 'RUNS', 'JOBS', 'CORES', 'MEM_GB', 'DEVICE', 'WORKERS', 'SET', 'TARGET', 'PY')}
    def sh(cmd, **env):
        p = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, env=dict(clean, **env), cwd=root, timeout=300)
        return p.returncode, p.stdout + p.stderr
    for prof, seeds in (('smoke', '0-1'), ('small', '0-2'), ('full', '0-4')):
        rc, out = sh('bash scripts/env.sh', PROFILE=prof)
        expect(rc == 0 and ('PROFILE=%s SEEDS=%s RUNS=runs/%s' % (prof, seeds, prof)) in out, 'I4 profile %s: %d seeds by default, runs in runs/%s' % (prof, len(RUN.parse_seeds(seeds)), prof))
    rc, out = sh('bash scripts/env.sh', PROFILE='nonsense')
    expect(rc == 2 and 'PROFILE must be' in out, 'I5 an unknown profile is refused')
    rc, out = sh('bash scripts/env.sh', PROFILE='full', SEEDS='0-6', DEVICE='cuda', JOBS='3', WORKERS='5', SET='rl_iters=7')
    expect('SEEDS=0-6' in out and 'DEVICE=cuda' in out and 'JOBS=3' in out and 'WORKERS=5' in out and "SET='rl_iters=7'" in out, 'I6 settings from the environment (%s)' % out.strip()[:100])
    # every setting is an option of every script: `--seeds 0-6` runs seven seeds, `--set` is repeatable, `--opt=value` works, a typo is refused, the options reach the scripts that a script calls
    rc, out = sh('bash scripts/env.sh --profile full --seeds 0-6 --device cuda --jobs 3 --workers 5 --set rl_iters=7 --set eval_probes=40', PROFILE='smoke', SEEDS='0-1', DEVICE='cpu')
    expect(rc == 0 and 'PROFILE=full SEEDS=0-6' in out and 'DEVICE=cuda' in out and 'JOBS=3' in out and 'WORKERS=5' in out and "SET='rl_iters=7 eval_probes=40'" in out, 'I6b the options (--seeds 0-6, --set twice, ...) win over the environment (%s)' % out.strip()[:100])
    rc, out = sh('bash scripts/env.sh --seeds=0-6 --runs=/somewhere --mem-gb=12 --cores=3 --target=rl/ppo/s0', PROFILE='small')
    expect(rc == 0 and 'SEEDS=0-6' in out and 'RUNS=/somewhere' in out and 'MEM_GB=12' in out and 'CORES=3' in out and "TARGET='rl/ppo/s0'" in out, 'I6c --option=value works as well (%s)' % out.strip()[:100])
    rc, out = sh('bash scripts/env.sh --seed 0-6')
    expect(rc == 2 and "unknown option '--seed'" in out, 'I6d an unknown option is refused, not ignored')
    rc, out = sh('bash scripts/env.sh --seeds')
    expect(rc == 2 and 'needs a value' in out, 'I6e an option without a value is refused')
    rc, out = sh('bash scripts/env.sh --profile nonsense')
    expect(rc == 2 and 'PROFILE must be' in out, 'I6f an unknown profile given as an option is refused too')
    rc, out = sh('bash scripts/env.sh --help')
    expect(rc == 0 and '--seeds' in out and 'SEEDS' in out and 'PROFILE=full' not in out, 'I6g --help prints the settings and the options')
    rc, out = sh('. scripts/env.sh --seeds 0-6 --profile small --set rl_iters=3 >/dev/null; bash -c \'echo "child:$SEEDS:$PROFILE:$SET"\'')
    expect(rc == 0 and 'child:0-6:small:rl_iters=3' in out, 'I6h a script that calls another one hands its options on (run_all.sh calls 00_check_env.sh, 06_benchmarks.sh, 07_report.sh): %s' % out.strip()[-60:])
    tmp = tempfile.mkdtemp(prefix='s51i_')
    try:
        # local.env gives the usual settings; what the caller exports wins
        fake = os.path.join(tmp, 'repo'); os.makedirs(os.path.join(fake, 'scripts'))
        shutil.copy(env_sh, os.path.join(fake, 'scripts', 'env.sh'))
        open(os.path.join(fake, 'scripts', 'local.env'), 'w').write('PROFILE=small\nDEVICE=cuda\nWORKERS=3\n')
        p = subprocess.run(['bash', os.path.join(fake, 'scripts', 'env.sh')], capture_output=True, text=True, env=dict(clean, PROFILE='smoke')); out = p.stdout
        expect('PROFILE=smoke' in out and 'DEVICE=cuda' in out and 'WORKERS=3' in out, 'I7 local.env sets what the caller does not set, the caller wins (%s)' % out.strip()[:90])
        p = subprocess.run(['bash', os.path.join(fake, 'scripts', 'env.sh')], capture_output=True, text=True, env=dict(clean)); out = p.stdout
        expect('PROFILE=small' in out and 'SEEDS=0-2' in out, 'I8 local.env alone')
        p = subprocess.run(['bash', os.path.join(fake, 'scripts', 'env.sh'), '--device', 'cpu', '--seeds=0-6'], capture_output=True, text=True, env=dict(clean, PROFILE='smoke', DEVICE='cuda:1')); out = p.stdout
        expect('PROFILE=smoke' in out and 'DEVICE=cpu' in out and 'SEEDS=0-6' in out and 'WORKERS=3' in out, 'I8b an option wins over the environment and over local.env (%s)' % out.strip()[:90])
        # the runner through the scripts' own function: nothing is executed, the graph is printed
        R = os.path.join(tmp, 'runs')
        rc, out = sh('. scripts/env.sh >/dev/null; exp_run --dry-run', PROFILE='smoke', RUNS=R, JOBS='2')
        n = len(SP.build('smoke', seeds=[0, 1], runs_dir=R))
        expect(rc == 0 and ('%d runs' % n) in out and not os.path.exists(os.path.join(R, 'pre')), 'I9 a dry run lists the %d runs of the smoke profile and executes none' % n)
        rc, out = sh('. scripts/env.sh >/dev/null; exp_run --dry-run', PROFILE='smoke', RUNS=R, SEEDS='0-2', SET='rl_iters=3 eval_probes=40', TARGET='rl/ei_her/*')
        lines = [l for l in out.split('\n') if l.startswith('rl/ei_her/')]
        expect(rc == 0 and len(lines) == 3 and all('--iters 3' in l for l in lines) and 'rl/ei_her/s2' in out and 'rl/ppo/' not in out, 'I10 SEEDS, SET and TARGET reach the runner (%d lines)' % len(lines))
        rc, out = sh('. scripts/env.sh >/dev/null; exp_run --status', PROFILE='smoke', RUNS=R)
        expect(rc == 0 and ('%d of %d runs to do' % (n, n)) in out, 'I11 the status of an empty runs directory: everything is to do')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # the scheduler, with trivial commands instead of the matrix
    tmp = tempfile.mkdtemp(prefix='s51j_')
    orig_build = SP.build
    try:
        def mk(rid, code, needs=(), check=True, cores=1, stage='data'):
            return SP.Run(rid, stage, [sys.executable, '-c', code], needs=needs, check=[os.path.join(tmp, rid)] if check else [], mem_gb=0.1, cores=cores)
        log = os.path.join(tmp, 'ran.log')
        def body(rid, extra=''):
            return "import os,time; open(r'%s','a').write('%s\\n'); %s open(r'%s','w').write(str(time.time()))" % (log, rid, extra, os.path.join(tmp, rid))
        runs = collections.OrderedDict()
        runs['a'] = mk('a', body('a')); runs['b'] = mk('b', body('b', "assert os.path.exists(r'%s');" % os.path.join(tmp, 'a')), needs=['a'])
        runs['c'] = mk('c', "raise SystemExit(3)", check=False); runs['d'] = mk('d', body('d'), needs=['c'])
        runs['e'] = mk('e', "pass", check=False); runs['f'] = mk('f', "pass")              # f promises a file it does not write
        lines = []
        sch = RUN.Scheduler(runs, tmp, list(runs), jobs=2, cores=2, mem_gb=8, say=lines.append)
        ok = sch.run()
        expect(not ok and sch.done == {'a', 'b', 'e'} and set(sch.failed) == {'c', 'f'} and sch.skipped == {'d'}, 'I12 the scheduler runs a graph: done %s, failed %s, skipped %s' % (sorted(sch.done), sorted(sch.failed), sorted(sch.skipped)))
        expect(RUN.read_status(tmp, 'c')['returncode'] == 3 and RUN.read_status(tmp, 'a')['returncode'] == 0 and RUN.read_status(tmp, 'f')['returncode'] == -1 and RUN.read_status(tmp, 'f')['missing'], 'I13 the status of a run: its exit code, -1 when a promised file is missing')
        expect(RUN.is_done(tmp, runs['a']) and not RUN.is_done(tmp, runs['c']) and not RUN.is_done(tmp, runs['f']) and RUN.is_done(tmp, runs['e']) and not RUN.is_done(tmp, runs['d']), 'I14 done = exit 0 and the promised files exist')
        expect(sorted(open(log).read().split()) == ['a', 'b'], 'I15 a run whose need failed was not started, the others ran once (%s)' % open(log).read().split())
        expect(any(l.startswith('skip') and 'd' in l for l in lines) and any(l.startswith('FAILED') and 'c' in l for l in lines), 'I16 the scheduler says what failed and what it skipped')
        sch2 = RUN.Scheduler(runs, tmp, list(runs), jobs=2, cores=2, mem_gb=8, say=lambda *a: None)
        ok2 = sch2.run()
        expect(sorted(open(log).read().split()) == ['a', 'b'] and set(sch2.failed) == {'c', 'f'}, 'I17 a second scheduler does not repeat what is done (it retries the failures)')
        # a run that needs something outside the selection which is not done is skipped
        sch3 = RUN.Scheduler(runs, tmp, ['d'], jobs=1, cores=1, mem_gb=8, say=lambda *a: None)
        sch3.run()
        expect('d' in sch3.skipped and sorted(open(log).read().split()) == ['a', 'b'], 'I18 a selected run whose need is neither selected nor done is skipped')
        # cores: two runs of two cores each never overlap on two cores, two of one core do
        def span(rid): t = open(os.path.join(tmp, rid)).read().split(); return float(t[0]), float(t[1])
        def timed(rid, cores):
            code = "import time; t0=time.time(); time.sleep(0.8); open(r'%s','w').write('%%r %%r' %% (t0, time.time()))" % os.path.join(tmp, rid)
            return mk(rid, code, cores=cores)
        for cores, expect_overlap in ((2, False), (1, True)):
            rr = collections.OrderedDict((rid, timed('t%d%s' % (cores, rid), cores)) for rid in 'xy')
            for r in rr.values():
                try: os.remove(os.path.join(tmp, r.id))
                except OSError: pass
            RUN.Scheduler(rr, tmp, list(rr), jobs=2, cores=2, mem_gb=8, say=lambda *a: None).run()
            (a0, a1), (b0, b1) = [span(r.id) for r in rr.values()]
            overlap = min(a1, b1) > max(a0, b0)
            expect(overlap == expect_overlap, 'I19 two runs of %d core(s) on two cores %s (spans %.2f-%.2f, %.2f-%.2f)' % (cores, 'overlap' if expect_overlap else 'do not overlap', a0, a1, b0, b1))
        mm = collections.OrderedDict((rid, mk('m' + rid, "import time; time.sleep(0.6); open(r'%s','w').write('x')" % os.path.join(tmp, 'm' + rid), cores=1)) for rid in 'xy')
        for r in mm.values(): r.mem_gb = 5.0
        for r in mm.values():
            try: os.remove(os.path.join(tmp, r.id))
            except OSError: pass
        sm = RUN.Scheduler(mm, tmp, list(mm), jobs=2, cores=2, mem_gb=8, say=lambda *a: None)
        first = sm.runnable()
        expect(len(first) == 1, 'I20 two runs of 5 GB do not start together within 8 GB (%s)' % first)
        # the whole runner: profile guard, force, cascade
        SP.build = lambda *a, **k: runs
        R2 = os.path.join(tmp, 'R2'); os.makedirs(R2)
        rc = RUN.main(['smoke', '--runs', R2, '--jobs', '2', '--stage', 'data'])
        expect(rc == 1 and os.path.exists(os.path.join(R2, 'PROFILE.json')) and os.path.exists(os.path.join(R2, 'GRAPH.json')) and not os.path.exists(os.path.join(R2, '.lock')), 'I21 the runner writes the profile and the graph, removes its lock, and exits 1 when something failed')
        os.makedirs(os.path.join(R2, 'pre'))
        raises('I22 a runs directory made with another profile is not mixed with this one', lambda: RUN.main(['smoke', '--runs', R2, '--set', 'rl_iters=9', '--target', 'e']), SystemExit)
        rc = RUN.main(['smoke', '--runs', R2, '--set', 'workers=3', '--set', 'device=cuda', '--target', 'e'])
        expect(rc == 0, 'I23 the machine settings may change between invocations of one runs directory')
        open(os.path.join(R2, '.lock'), 'w').write(str(os.getpid()))
        raises('I24 two runners do not share a runs directory', lambda: RUN.main(['smoke', '--runs', R2, '--target', 'e']), SystemExit)
        os.remove(os.path.join(R2, '.lock'))
        raises('I25 --target of an unknown run is an error', lambda: RUN.main(['smoke', '--runs', R2, '--target', 'zzz']), SystemExit)
        before = open(log).read().split()
        RUN.main(['smoke', '--runs', R2, '--target', 'a,b', '--force', 'a', '--cascade'])
        after = open(log).read().split()
        expect(sorted(after[len(before):]) == ['a', 'b'], 'I26 --force with --cascade re-runs the run and what depends on it (%s)' % after[len(before):])
        before = open(log).read().split()
        RUN.main(['smoke', '--runs', R2, '--target', 'a,b'])
        expect(open(log).read().split() == before, 'I27 and a plain call after that repeats nothing')
    finally:
        SP.build = orig_build
        shutil.rmtree(tmp, ignore_errors=True)


SECTIONS = [('A', sec_A, False), ('B', sec_B, False), ('C', sec_C, False), ('D', sec_D, True), ('E', sec_E, True), ('F', sec_F, True), ('G', sec_G, True), ('H', sec_H, True), ('I', sec_I, False)]


def main():
    c = {}
    t_start = time.time()
    for name, fn, needs_world in SECTIONS:
        if ONLY is not None and name not in ONLY: continue
        if needs_world and 'mini' not in c:
            t0 = time.time()
            try: c['mini'] = Mini()
            except Exception:
                FAIL.append('the small world could not be built\n%s' % traceback.format_exc()); break
            print('S51 the small world: %d problems, %d globals, %.1f s' % (len(c['mini'].bk.placed), c['mini'].bk.w.k.num_globals(), time.time() - t0), flush=True)
        t0 = time.time(); n0 = CHECKS[0]; f0 = len(FAIL)
        try: fn(c)
        except Exception:
            FAIL.append('%s: section crashed\n%s' % (name, traceback.format_exc()))
        print('S51 %s: %d checks, %d failed, %.1f s' % (name, CHECKS[0] - n0, len(FAIL) - f0, time.time() - t0), flush=True)
    if 'mini' in c: c['mini'].close()
    for f in FAIL[:60]: print('  FAIL: ' + f)
    print('S51 %d checks, %d failures, %.0f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    print('S51 TRAINING SUITE GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
