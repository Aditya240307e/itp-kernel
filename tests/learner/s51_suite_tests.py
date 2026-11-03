"""S51 gate: the TRAINING SUITE -- hindsight, the growing library, the goals of the learners, the provers and their protocol, the experiment matrix and the statistics the paper is made of.

Nothing here is trusted by the kernel and nothing is trusted from a learner: an agent's proof counts when the engine replayed it and the kernel checked it; a lemma becomes a premise when the kernel checked its
proof; a number in a table is arithmetic on rows.  What this gate protects is that the layer between the kernel and the paper does what its documentation says -- a hindsight theorem is a theorem, a lemma of
the library is a theorem that no held-out problem can lean on, the curve "solved within B probes" read off one run is the curve of a run with budget B, an interval is the interval of the formula -- and that
the experiment matrix is a well-formed graph of commands whose products land where the collector looks for them.

  A. STATISTICS.     The estimators of exp_stats against values worked out by hand (Student-t, Wilson, Holm, Chao1, paired tests, the bootstrap on constant data, rarefaction, crossing points).
  B. THE MATRIX.     The run graph of every profile: unique ids, an acyclic graph whose needs exist, the checks every run promises, the closure of a target, the seeds, the overrides (an unknown field is an error), the
                     machine fields, the synthetic-set exclusion and the training commands (no test data in them), the estimator.
  C. THE COLLECTOR.  A runs directory written by hand (3 seeds, 2 learners, baselines) is read back: solved-within-budget matrices, rates with their intervals, comparisons, the strict view, the subsets, the
                     report and its numbers, the warnings for a missing result, the contest family of a source, the facts of a report made of the dataset alone (what the paper skeleton prints) and the LaTeX of a table (escapes; a mean ± half-width cell that cannot break).
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
  J. THE PROFILE.    The ledger of exp_profile (exclusive time that adds up, inclusive latency, percentiles), its hooks on a stand-in for the engine (installed and taken out again, exceptions, one profile at a time, other
                     threads), the memory of a process tree, what exp_bench counts as a result and what as the clock (the comparison of runs and its digest), the machine record.
  K. THE BENCH.      On the small world: a profiled pass finds what a plain one finds and accounts for the time, the probes of the ledger are the probes of the rows, the solvers in repeated passes, the network's share,
                     the Gymnasium interface, workers (memory, determinism of the rows), the cost of checking a proof, and the whole script.

  L. THE REPORT OF THE BENCH.  A bench.json made by hand with known numbers (real ledgers on a clock that is moved by hand, 1..100 ms latency samples) read back by the collector: every cell of the throughput,
                     time-breakdown, latency, worker, machine and fixed-cost tables, the figures' data, the numbers the text quotes, the warnings the measurements deserve (a busy machine, rows that differ between worker
                     counts, a pass that did not reproduce, a refused proof), a bench.json of an older format, and the results folder that carries it.

  M. THE STUDIES.    A results folder made by hand for the studies of the network and of the start (held-out synthetic metrics on exact power laws with a known exponent per seed; other encoders, other widths, a learner from random
                     weights; a benchmark of the encoders): every cell of the tables of encoders, widths, scaling laws, learning from scratch and the cost of the encoders, the data of the three figures, the numbers the text
                     quotes (and that the parameters quoted are those of THE network, not of the first network of a study), the sections of the report, a report of a folder with only a part of the studies.

  N. THE ENCODERS AS COMMANDS.  On the small world: ml_init makes an untrained network of each encoder (the same seed the same weights, the transformer the one make_net builds), ml_bc --arch / --init (an encoder of
                     its own, the checkpoint's, an error for another), expert iteration and PPO continue the encoder of their checkpoint, the evaluation loads any of them, and exp_bench --archs/--widths costs the
                     networks the experiments train.

  O. THE SOUNDNESS STAGE.  exp_soundness on records made by hand with known answers (a catalogue of six mutants, four gates, four fuzz campaigns, a comparison with the independent checker, six findings): every
                     summary (a record counts only for its own mutant and its own tree, the last record wins, the status is derived from what caught a mutant and from the catalogue), every verdict (FAILED, INCOMPLETE, CLEAN
                     and why), every cell of the tables and every number, the folder it writes (README with the numbers of the catalogue, manifest, the same bytes when rendered twice); the real tracker is fully tagged;
                     the records the real tools write (run_gates.py --json, gym_fuzz.py --json, S8_JSON); the real catalogue and inventory; and the whole stage on a tiny selection (resumed, reused only for the same tree,
                     failed gates and comparisons run again, verified).

  P. THE STATEMENT AUDIT.  exp_audit on ten records made by hand that have every caution, every kind of failure and every origin, and a ledger that puts them in every state: reading an AST (the constructs, an
                     equation that is not negated), a statement as a person reads it (precedence, the binder of a hypothesis), the failures and the cautions of a record, the ledger (every wrong line named, a reading counts for
                     the digest it was written for, the last line of a reader counts, author and second reader), import of a filled review sheet (all or nothing), the totals, the verdict (FAILED, INCOMPLETE, CLEAN), every
                     table cell and number, the review sheet (the order), the folder (manifest, README, macros); the real records of the small bank (the dataset's digests, every statement passes); and the whole thing on two
                     problems of the real bank (partial: INCOMPLETE; reused for the same tree and statement; records of another tree dropped; a reading imported and counted; --verify).

  Q. THE CODE THE EVIDENCE IS ABOUT.  Evidence is bound to the digest of the code (src, tests, mutants, the manifest of the dataset), not to the commit and not to the state of the whole working tree: `dirty` means that the
                     CODE is not committed (a document, the paper, a tool or a script that is edited is not; a new file below src, tests or mutants is) in both places that say it; a run records the digest when it starts and
                     when it ends; the report warns of runs made with different versions of the code, or with code that changed while they ran, and not of a commit that changed only the paper; runs.csv carries the digests.

Environment: S51_TRACE=1 prints timings, S51_ONLY=A,B selects sections, S51_DATA=DIR reuses a small dataset built earlier (development; `s51_suite_tests.py --build-data DIR` builds it).  D-H and P build the small dataset (about two minutes).
The tools these sections test are themselves tested by fault injection: mutants/check_tools.py injects faults into exp_soundness (section O), exp_audit (P) and the record of the code (Q), and these sections must fail for each (finding F88).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, io, json, math, time, copy, glob, shutil, tempfile, argparse, itertools, subprocess, collections, traceback, statistics, types
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

    # the exponent of a power law: the least-squares line of ln y on ln x
    s, i, r2, n = ST.loglog_fit([1, 4, 16], [2, 1, 0.5])
    expect(close(s, -0.5) and close(i, math.log(2)) and close(r2, 1.0) and n == 3, 'A33 loglog_fit of y = 2 x^-0.5 is that law exactly: slope -0.5, intercept ln 2, R^2 1 (got %s %s %s %s)' % (s, i, r2, n))
    s, i, r2, n = ST.loglog_fit([1, 2, 4], [1, 3, 2])
    expect(close(s, 0.5) and abs(i - 0.2507) < 5e-4 and abs(r2 - 0.3892) < 5e-4 and n == 3, 'A34 loglog_fit of points that are not on a line, worked out by hand: slope 0.5, intercept 0.2507, R^2 0.3892 (got %.4f %.4f %.4f)' % (s, i, r2))
    s, i, r2, n = ST.loglog_fit([1, 4, 16, 64, None, 256], [2, 1, 0.5, 0.0, 3, float('nan')])
    expect(close(s, -0.5) and n == 3, 'A35 a point with a zero, missing or NaN coordinate is left out of the fit (%d points used)' % n)
    expect(all(math.isnan(v) for v in ST.loglog_fit([2], [3])[:3]) and ST.loglog_fit([2], [3])[3] == 1 and all(math.isnan(v) for v in ST.loglog_fit([2, 2], [3, 4])[:3]), 'A36 one point, or no spread in x: no exponent')
    s, i, r2, n = ST.loglog_fit([1, 2, 4], [3, 3, 3])
    expect(s == 0.0 and math.isnan(r2), 'A37 a constant y has exponent 0 and no R^2 (nothing to explain)')
    m, lo, hi, k = ST.exponent_ci([1, 4, 16], [[2, 1, .5], [3, 1.5, .75], [1, 1, 1]])
    sd = statistics.stdev([-0.5, -0.5, 0.0]); h = T975_2 * sd / math.sqrt(3)
    expect(k == 3 and close(m, -1.0 / 3.0) and close(lo, m - h) and close(hi, m + h), 'A38 exponent_ci: the exponent fitted on each seed (-1/2, -1/2, 0), the t-interval over seeds (got %.4f [%.4f, %.4f], %d seeds)' % (m, lo, hi, k))
    m, lo, hi, k = ST.exponent_ci([1, 4, 16], [[2, 1, .5], [3, 1.5, .75], [None, None, 1]])
    expect(k == 2 and close(m, -0.5) and close(lo, -0.5) and close(hi, -0.5), 'A39 a seed that cannot be fitted (one point left) does not count (%d seeds)' % k)
    m, lo, hi, k = ST.exponent_ci([1, 4, 16], [[2, 1, .5]])
    expect(k == 1 and close(m, -0.5) and math.isnan(lo) and math.isnan(hi), 'A40 one seed: an exponent and no interval')


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
        expect(all('prefix,strict' in r.argv for i, r in runs.items() if r.stage == 'eval' and not i.startswith('eval/synth/')), 'B15 %s: every olympiad evaluation runs both regimes' % name)
        expect(all('ml_eval_synth.py' in ' '.join(r.argv) and 'prefix,strict' not in r.argv and r.tags.get('group') == 'synth' for i, r in runs.items() if i.startswith('eval/synth/')) and any(i.startswith('eval/synth/') for i in runs), 'B15b %s: the evaluations of the synthetic theorems are their own group, run by ml_eval_synth.py' % name)
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
    expect(sorted(int(i.split('/n')[1].split('/')[0]) for i in full if i.startswith('pre/n') and i.endswith('/s0')) == [1000, 4000, 15000, 60000], 'B43 the sizes of the scaling study')
    studies_in_matrix()


def studies_in_matrix():
    """the studies of the network and of the start: other encoders, other widths, reinforcement learning from random weights"""
    try:
        import ml_model
        expect(tuple(ml_model.ARCHS) == tuple(SP.ARCHS), 'B44 the encoders the matrix knows are the encoders the network has (%s)' % (SP.ARCHS,))
    except ImportError:
        expect(True, 'B44 (torch missing: the lists of encoders were not compared)')
    expect(SP.variant_name('bc') == 'bc' and SP.variant_name('bc', 'bigru') == 'bc_bigru' and SP.variant_name('ei_her', 'transformer', 64) == 'ei_her_w64' and SP.variant_name('ei_her', 'cnn', 128) == 'ei_her_cnn'
           and SP.parse_scratch('ei_her') == ('ei_her', 'transformer') and SP.parse_scratch('ppo@bow') == ('ppo', 'bow'), 'B45 the names of the variants: learner, encoder, width')
    expect(SP.net_args() == [] and SP.net_args('transformer', 128) == [] and SP.net_args('bow') == ['--arch', 'bow'] and SP.net_args('transformer', 64) == ['--d', 64, '--layers', 2, '--heads', 2, '--ff', 192]
           and SP.size_of(128) == dict(d=128, layers=3, heads=4, ff=384) and SP.size_of(256)['layers'] == 4 and SP.size_of(256)['heads'] == 8, 'B46 the default network needs no arguments, the others are named by the encoder and the width')
    for name in sorted(SP.PROFILES):
        P = SP.profile(name); runs = SP.build(name, runs_dir='/runs/%s' % name); smax = max(P['synth_sizes']); seeds = P['seeds']
        arg = lambda r, flag: r.argv[r.argv.index(flag) + 1]
        base_pre = runs['pre/n%d/s0' % smax]
        expect('--arch' not in base_pre.argv and '--d' not in base_pre.argv and '--arch' not in runs['ft/bc/s0'].argv, 'B47 %s: the default network is made without an architecture argument' % name)
        for arch in P['archs']:
            for s in range(seeds):
                pre = runs['pre/%s_n%d/s%d' % (arch, smax, s)]; ft = runs['ft/bc_%s/s%d' % (arch, s)]
                expect(arg(pre, '--arch') == arch and '--no-bank' in pre.argv and '--no-lib' in pre.argv and arg(pre, '--seed') == str(s) and arg(pre, '--synth-max') == str(smax), 'B48 %s: %s is pre-trained on synthetic theorems alone, with its own encoder and seed' % (name, arch))
                expect(ft.needs == (pre.id,) and arg(ft, '--init').endswith('/%s/best.pt' % pre.id) and '--arch' not in ft.argv, 'B49 %s: fine-tuning continues from the network of its own encoder and seed, which names its architecture itself' % name)
            for learner in P['arch_learners']:
                if learner == 'bc': continue
                for s in range(seeds):
                    r = runs['rl/%s_%s/s%d' % (learner, arch, s)]
                    expect(r.needs[0] == 'ft/bc_%s/s%d' % (arch, s) and arg(r, '--init').endswith('/ft/bc_%s/s%d/best.pt' % (arch, s)) and '--arch' not in r.argv, 'B50 %s: %s_%s continues the fine-tuned network of its encoder' % (name, learner, arch))
                    base = runs['rl/%s/s%d' % (learner, s)]
                    expect(r.argv[:3] == base.argv[:3] and r.argv[r.argv.index('--workers'):] == base.argv[base.argv.index('--workers'):] and r.tags['algo'] == base.tags['algo'], 'B51 %s: %s_%s is the same learner with the same settings (everything after the paths and the seed)' % (name, learner, arch))
        for w in P['widths']:
            sz = SP.size_of(w); pre = runs['pre/w%d_n%d/s0' % (w, smax)]
            expect([arg(pre, f) for f in ('--d', '--layers', '--heads', '--ff')] == [str(sz['d']), str(sz['layers']), str(sz['heads']), str(sz['ff'])] and '--arch' not in pre.argv and runs['ft/bc_w%d/s0' % w].needs == (pre.id,), 'B52 %s: width %d is a network of %s' % (name, w, sz))
        for item in P['scratch']:
            learner, arch = SP.parse_scratch(item); v = 'scratch_' + SP.variant_name(learner, arch)
            for s in range(seeds):
                r = runs['rl/%s/s%d' % (v, s)]; ini = runs['init/%s/s%d' % (arch, s)]; base = runs['rl/%s/s%d' % (learner, s)]
                expect(r.needs[0] == ini.id and arg(r, '--init').endswith('/init/%s/s%d/best.pt' % (arch, s)), 'B53 %s: %s starts from the random weights of its own encoder and seed' % (name, v))
                anc = SP.closure(runs, [r.id])
                expect(not [i for i in anc if i.startswith(('pre/', 'ft/'))] and 'data/synth_merge' in anc, 'B54 %s: %s has no behaviour cloning and no pre-training in its ancestry (%s)' % (name, v, [i for i in anc if i.startswith(('pre/', 'ft/'))]))
                expect(arg(r, '--p-synth') == str(SP.SCRATCH_P_SYNTH) and arg(base, '--p-synth') != arg(r, '--p-synth') and int(arg(r, '--iters')) == int(arg(base, '--iters')) * int(P['scratch_mult']), 'B55 %s: %s learns mostly from synthetic goals, for %d times the iterations' % (name, v, P['scratch_mult']))
            expect(arg(ini, '--seed') == str(seeds - 1) and arg(ini, '--arch') == arch if arch != 'transformer' else '--arch' not in ini.argv, 'B56 %s: the network of %s is made by ml_init (%s)' % (name, v, ini.argv[1].split('/')[-1]))
        for key in ('arch', 'size', 'scratch'):
            ev = [i for i, r in runs.items() if r.stage == 'eval' and r.tags.get('group') == key]
            want = {'arch': len(P['archs']) * (1 + len([l for l in P['arch_learners'] if l != 'bc'])), 'size': len(P['widths']) * (1 + len([l for l in P['width_learners'] if l != 'bc'])), 'scratch': len(P['scratch'])}[key]
            expect(len(ev) == want and all(i.startswith('eval/%s/' % key) for i in ev), 'B57 %s: one evaluation per variant of the %s study (%d)' % (name, key, want))
        bench = runs['bench/throughput']
        expect(arg(bench, '--archs').split(',') == ['transformer'] + list(P['archs']) and arg(bench, '--widths').split(',') == [str(w) for w in sorted({128} | set(P['widths']))], 'B58 %s: the benchmark costs the encoders and the widths of the studies' % name)
    # the studies can be switched off: what remains is the comparison of the learners
    off = SP.build('small', runs_dir='/r', overrides={'archs': (), 'widths': (), 'scratch': ()})
    expect(not [i for i in off if i.startswith(('eval/arch', 'eval/size', 'eval/scratch')) or (i.startswith('init/') and not i.startswith('init/transformer/'))] and not [i for i in off if '_bigru' in i or '_w64' in i or 'scratch_' in i]
           and len([i for i in off if i.startswith('init/')]) == int(SP.profile('small')['seeds']) and len([i for i in off if i.startswith('eval/synth/')]) == 9 and len(off) == 89,
           'B59 without the studies the matrix is that of the learners (77 runs), the untrained control (one init per seed) and the 9 evaluations of the synthetic theorems (%d runs)' % len(off))
    one = SP.build('small', runs_dir='/r', overrides={'archs': ('cnn',), 'arch_learners': ('bc',), 'widths': (), 'scratch': ('ppo@bow',)})
    expect('ft/bc_cnn/s0' in one and 'rl/ei_her_cnn/s0' not in one and 'rl/scratch_ppo_bow/s0' in one and {i for i in one if i.startswith('init/') and i.endswith('/s0')} == {'init/bow/s0', 'init/transformer/s0'}, 'B60 a single study can be asked for alone (the transformer init is the control of the synthetic theorems)')
    for what, ov in (('an unknown encoder', {'archs': ('lstm',)}), ('the transformer as an alternative to itself', {'archs': ('transformer',)}), ('an encoder twice', {'archs': ('cnn', 'cnn')}), ('a learner that needs a library', {'arch_learners': ('bc', 'ei_her_boot')}),
                     ('a learner that does not continue from bc', {'width_learners': ('ei_mw',)}), ('a width that is not a multiple of 32', {'widths': (48,)}), ('the default width', {'widths': (128,)}), ('a scratch learner that is unknown', {'scratch': ('sgd',)}),
                     ('a scratch encoder that is unknown', {'scratch': ('ppo@lstm',)}), ('no iterations', {'scratch_mult': 0})):
        raises('B61 %s is refused' % what, lambda ov=ov: SP.build('smoke', runs_dir='/r', overrides=ov), ValueError)
    # the estimate: every part has its share, the parts add up, a faster network trains faster
    Pf = SP.profile('full'); counts = {'train_steps': 20000, 'dev': 40, 'test': 80, 'strict_extra': 10, 'train': 200}; rf = SP.build('full', runs_dir='/r')
    whole = EST.estimate(rf, Pf, counts, EST.RATES); parts = EST.estimate_by_study(rf, Pf, counts, EST.RATES)
    expect(list(parts) == ['core', 'arch', 'size', 'scratch'] and sum(v[0] for e in parts.values() for v in e.values()) == sum(v[0] for v in whole.values()) == len(rf)
           and abs(sum(v[1] for e in parts.values() for v in e.values()) - sum(v[1] for v in whole.values())) < 1e-6, 'B62 the parts of the estimate add up to the whole (%s)' % list(parts))
    expect(EST.train_ratio(EST.RATES, {}) == 1.0 and EST.train_ratio(EST.RATES, {'arch': 'cnn'}) > 1.0 and EST.train_ratio(EST.RATES, {'width': 256}) < 1.0 and EST.train_ratio(EST.RATES, {'width': 384}) < EST.train_ratio(EST.RATES, {'width': 256})
           and EST.train_ratio({}, {'arch': 'bow'}) == 1.0, 'B63 a wider network trains slower, a CNN faster; a width with no measurement is estimated from its size')
    pre_cnn = {k: r for k, r in rf.items() if k == 'pre/cnn_n60000/s0'}; pre_tf = {k: r for k, r in rf.items() if k == 'pre/n60000/s0'}
    e1 = EST.estimate(pre_cnn, Pf, counts, EST.RATES)['pre'][1]; e2 = EST.estimate(pre_tf, Pf, counts, EST.RATES)['pre'][1]
    expect(abs(e2 / e1 - EST.RATES['train_ratio']['cnn@128']) < 1e-9, 'B64 the same pre-training takes 1/ratio of the time on the faster encoder')
    sc = {k: r for k, r in rf.items() if k in ('rl/scratch_ei_her/s0', 'rl/ei_her/s0')}
    es = EST.estimate({'rl/scratch_ei_her/s0': sc['rl/scratch_ei_her/s0']}, Pf, counts, EST.RATES)['rl'][1]; eb = EST.estimate({'rl/ei_her/s0': sc['rl/ei_her/s0']}, Pf, counts, EST.RATES)['rl'][1]
    expect(es > 2.5 * eb, 'B65 learning from scratch runs several times the iterations of fine-tuning with a learner (%.1f h against %.1f h)' % (es / 3600, eb / 3600))
    tmp = tempfile.mkdtemp(prefix='s51b2_')
    try:
        bp = os.path.join(tmp, 'bench.json')
        json.dump({'archs': {'transformer@128': {'train_steps_per_s': 100.0}, 'bigru@128': {'train_steps_per_s': 150.0}, 'transformer@64': {'train_steps_per_s': 300.0}}}, open(bp, 'w'))
        r_, ok = EST.rates_from_bench(bp)
        expect(ok and abs(r_['train_ratio']['bigru@128'] - 1.5) < 1e-9 and abs(r_['train_ratio']['transformer@64'] - 3.0) < 1e-9 and 'transformer@128' not in r_['train_ratio'], 'B66 the measured speed of each encoder and width is read from the benchmark, relative to the default network')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
        # a "mean ± half-width" cell never breaks (the first rendering of the paper showed "33.3 ±" and "0.0" on two lines): it is tied around the sign, and its column is as wide as the whole cell
        t2 = TB.Table('demo2', 'caption', ['method', 'dev'], [['a long method name that may wrap', '33.3 ± 0.0'], ['b', '20.0 ± 10.5']], align='lr')
        s2, w2 = t2.tex(), t2.column_widths()
        expect('33.3~$\\pm$~0.0' in s2 and '20.0~$\\pm$~10.5' in s2 and ' $\\pm$ ' not in s2 and '$\\sim$' not in s2, 'C31 a mean +- half-width cell is tied around the sign in LaTeX (it cannot break)')
        expect(w2[1] >= len('20.0±10.5') + 1, 'C32 the column of such a cell is as wide as the whole cell: %s' % w2)
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
        # ---- the results folder (exp_bundle): everything a paper is written from, saved apart from the paper
        import csv as _csv, hashlib, exp_bundle as EB
        TB.ABLATIONS = TB.ABLATIONS + [('ei_vs_bc', 'EI + HER vs BC (a comparison the fixture has both sides of)', 'ei_her', 'bc')]
        try:
            out = os.path.join(tmp, 'bundle'); REP.main([R, '--out', out, '--no-figures'])                 # (a folder of its own: the figures of C27 were added to the one before after its manifest was written)
        finally: TB.ABLATIONS = TB.ABLATIONS[:-1]
        def rd(*parts): return list(_csv.DictReader(open(os.path.join(out, *parts))))
        need = ['README.md', 'MANIFEST.sha256', 'report.md', 'numbers.json', 'per_problem/outcomes.csv', 'per_problem/by_problem.csv', 'per_problem/by_family.csv', 'stats/rates.csv', 'stats/comparisons.csv', 'stats/benchmark.csv',
                'provenance/PROFILE.json', 'provenance/runs.csv', 'provenance/machine.json', 'provenance/repo_facts.json', 'provenance/dataset.json', 'tables/main.csv']
        expect(all(os.path.exists(os.path.join(out, n)) for n in need), 'C33 the results folder holds the README, the manifest, the per-problem files, the statistics and the provenance (missing: %s)' % [n for n in need if not os.path.exists(os.path.join(out, n))])
        listed = dict(reversed(l.split('  ', 1)) for l in open(os.path.join(out, 'MANIFEST.sha256')).read().split('\n') if l)
        ondisk = sorted(os.path.relpath(os.path.join(r_, f_), out).replace(os.sep, '/') for r_, _, fs in os.walk(out) for f_ in fs if f_ != 'MANIFEST.sha256')
        digest = lambda rel: hashlib.sha256(open(os.path.join(out, rel), 'rb').read()).hexdigest()
        expect(sorted(listed) == ondisk and all(listed[n] == digest(n) for n in ondisk), 'C34 the manifest lists every file of the folder with its SHA-256 (%d files)' % len(ondisk))
        open(os.path.join(out, 'stats', 'rates.csv'), 'a').write('x\n')
        expect(listed['stats/rates.csv'] != digest('stats/rates.csv'), 'C35 a file that is changed no longer matches its line of the manifest')
        expect(EB.verify(out)[:1] == ['changed: stats/rates.csv'] and len(EB.verify(out)) == 1, 'C35b verify() names the file that was changed (%s)' % EB.verify(out))
        with open(os.path.join(out, 'stats', 'rates.csv'), 'rb') as f_: data_ = f_.read()
        open(os.path.join(out, 'stats', 'rates.csv'), 'wb').write(data_[:-2])                                    # (the "x\n" appended above is taken off again)
        bench_ = open(os.path.join(out, 'stats', 'benchmark.csv'), 'rb').read(); os.remove(os.path.join(out, 'stats', 'benchmark.csv')); open(os.path.join(out, 'new.txt'), 'w').write('x')
        v_ = EB.verify(out)
        open(os.path.join(out, 'stats', 'benchmark.csv'), 'wb').write(bench_); os.remove(os.path.join(out, 'new.txt'))
        expect(v_ == ['missing: stats/benchmark.csv', 'not in the manifest: new.txt'] and EB.verify(out) == [], 'C35c verify() reports a missing file and a file the manifest does not list, and nothing for the folder as it was written (%s, then %s)' % (v_, EB.verify(out)))
        oc = rd('per_problem', 'outcomes.csv')
        expect(len(oc) == len(C.all_rows) and {r['stop'] for r in oc} <= {'solved', 'budget', 'time', 'gave_up', 'error'}, 'C36 outcomes.csv holds one row per attempt of every regime (%d of %d)' % (len(oc), len(C.all_rows)))
        r1 = next(r for r in oc if r['variant'] == 'bc' and r['seed'] == '0' and r['name'] == 't2' and r['regime'] == 'prefix'); r2 = next(r for r in oc if r['variant'] == 'bc' and r['seed'] == '0' and r['name'] == 't3' and r['regime'] == 'prefix')
        expect(r1['solved'] == '1' and r1['probes'] == '50' and r1['stop'] == 'solved' and r1['solved_at_headline'] == '1' and r2['solved'] == '0' and r2['stop'] == 'budget' and r2['solved_at_headline'] == '0', 'C37 an attempt row says what happened and how it ended: %s, %s' % (r1, r2))
        Pp = {'eval_probes': 200, 'eval_time_limit': 10.0}
        expect([EB.stop_reason(x_, Pp) for x_ in ({'solved': True}, {'solved': False, 'error': 'boom'}, {'solved': False, 'probes': 201, 'seconds': 1.0}, {'solved': False, 'probes': 30, 'seconds': 10.2}, {'solved': False, 'probes': 30, 'seconds': 2.0})]
               == ['solved', 'error', 'budget', 'time', 'gave_up'], 'C38 an attempt ends solved / error / budget / time / gave_up, in that order of precedence')
        rt = next(r for r in rd('stats', 'rates.csv') if (r['group'], r['variant'], r['solver'], r['split'], r['subset'], r['budget']) == ('main', 'bc', 'sample', 'test', 'all', '100'))
        m_, lo_, hi_, n_ = C.rate_ci('main', 'bc', 'sample', 'test', 100, 'all')
        expect(abs(float(rt['mean']) - m_) < 1e-5 and abs(float(rt['ci_low']) - lo_) < 1e-5 and abs(float(rt['ci_high']) - hi_) < 1e-5 and json.loads(rt['solved_per_seed']) == [2, 2, 3] and json.loads(rt['problems_per_seed']) == [4, 4, 4]
               and rt['ci_kind'] == 't over seeds' and rt['n_seeds'] == '3', 'C39 rates.csv: the cell of the main table with the count of each seed (%s)' % rt)
        cm = {(r['split'], r['key']): r for r in rd('stats', 'comparisons.csv')}; cmp_t = C.compare('main', 'ei_her', 'bc', 'sample', 'test', 100)
        row = cm[('test', 'ei_vs_bc')]
        expect(abs(float(row['diff_pp']) - 100 * cmp_t['diff']['mean']) < 1e-4 and abs(float(row['p']) - cmp_t['diff']['p']) < 1e-5 and float(row['p_holm']) >= float(row['p']) - 1e-12 and int(row['n_problems']) == cmp_t['n_problems'] == 3
               and float(row['crossed_low_pp']) <= float(row['diff_pp']) <= float(row['crossed_high_pp']) and float(row['mde80_pp']) >= 0, 'C40 comparisons.csv: the ablation of the table with its seed, problem and crossed intervals, Holm-adjusted p and detectable difference (%s)' % row)
        bm = {r['split']: r for r in rd('stats', 'benchmark.csv')}
        expect(bm['test']['n_problems'] == '4' and bm['test']['n_methods'] == '4' and int(bm['test']['floor_none_solve']) + int(bm['test']['ceiling_all_solve']) + int(bm['test']['informative']) == 4 and bm['test']['ceiling_all_solve'] == '1'
               and int(bm['test']['failed_by_budget']) + int(bm['test']['failed_by_time']) + int(bm['test']['failed_gave_up']) + int(bm['test']['failed_error']) == int(bm['test']['failed_attempts']), 'C41 benchmark.csv: floor, ceiling and informative problems add up to the problems, the failed attempts to their causes (%s)' % bm['test'])
        bp = {r['name']: r for r in rd('per_problem', 'by_problem.csv')}
        expect(float(bp['t1']['main/bc/sample']) == 1.0 and bp['t1']['solved_by_any'] == '1' and bp['t3']['solved_by_all'] == '0' and bp['d2']['split'] == 'dev' and bp['t4']['source_class'] == 'IMO' and bp['t4']['proved'] == '0' and bp['t1']['source_class'] == 'AMC',
               'C42 by_problem.csv: the fraction of seeds that solved each problem, who solved it, what it is (%s)' % {k: bp['t1'][k] for k in ('main/bc/sample', 'solved_by_any', 'solved_by_all')})
        bf = rd('per_problem', 'by_family.csv'); fam = next(r for r in bf if (r['variant'], r['split'], r['key_type'], r['key']) == ('bc', 'test', 'source', 'AMC'))
        expect(fam['n_problems'] == '3' and fam['n_seeds'] == '3' and len(json.loads(fam['per_seed'])) == 3, 'C43 by_family.csv: a rate per contest family with its replicates (%s)' % fam)
        prov = json.load(open(os.path.join(out, 'provenance', 'PROFILE.json')))
        expect(prov['headline'] == 100 and prov['seed_list'] == [0, 1, 2] and 'Blind search' in open(os.path.join(out, 'tables', 'main.csv')).read() and open(os.path.join(out, 'README.md')).read().count('sha256sum -c') == 1, 'C44 provenance holds the profile; the table is a CSV; the README says how to check the folder')
        # the statistics behind it: a matrix of known answers
        A = [[1, 0, 1, 0]]; Bm = [[0, 0, 1, 1]]
        expect(abs(ST.paired_mde(A, Bm) - 2.8016 * math.sqrt(2.0 / 3.0) / 2) < 1e-3 and ST.paired_mde([[1, 1, 0]], [[1, 1, 0]]) == 0.0 and math.isnan(ST.paired_mde([[1]], [[0]])), 'C45 the detectable difference is (z_.975 + z_.8) * sd(d) / sqrt(n) of the per-problem differences, 0 for equal methods, NaN for one problem')
        cb = ST.crossed_bootstrap([[1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]]); cz = ST.crossed_bootstrap([[1, 0], [0, 1]], [[1, 0], [0, 1]])
        expect(cb == {'mean': 1.0, 'low': 1.0, 'high': 1.0} and cz == {'mean': 0.0, 'low': 0.0, 'high': 0.0}, 'C46 the crossed bootstrap of a constant difference is that difference, of two equal methods zero')
        wide = ST.crossed_bootstrap([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0]], [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]])
        prob = ST.paired_problem_bootstrap([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0]], [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]])
        expect(wide['low'] <= wide['mean'] <= wide['high'] and (wide['high'] - wide['low']) >= (prob['high'] - prob['low']) - 0.02, 'C47 resampling the replicates too does not narrow the interval (%.3f vs %.3f)' % (wide['high'] - wide['low'], prob['high'] - prob['low']))
        tb = TB.Table('demo3', 'cap', ['a', 'b'], [['x, y', '1.0 ± 2'], ['z', '"q"']]); got = list(_csv.reader(tb.csv().split('\n')[:-1]))
        expect(got == [['a', 'b'], ['x, y', '1.0 ± 2'], ['z', '"q"']], 'C48 a table is written as CSV with its cells as printed')
        try:
            import matplotlib  # noqa: F401
            EF = __import__('exp_figures'); plt = EF._mpl()
            fig, ax = plt.subplots(); EF.FEW_SEEDS = False
            EF.band(ax, [1, 2, 3], [0.2, 0.4, 0.6], [0.1, 0.3, 0.5], [0.3, 0.5, 0.9], EF.BLUE, '-', 'o', 'curve'); ax.bar([5], [0.5], width=0.5, label='bars'); ax.plot([7, 7], [0.2, 0.4], color='k'); ax.axvline(0)
            rows_ = EF.plotted_data(fig); lines_ = [r for r in rows_ if r[2] == 'line']; plt.close(fig)
            expect([r[4] for r in lines_] == [1.0, 2.0, 3.0] and [r[8] for r in lines_] == [0.1, 0.3, 0.5] and [r[9] for r in lines_] == [0.3, 0.5, 0.9] and any(r[2] == 'bar' and r[5] == 0.5 for r in rows_) and any(r[2] == 'interval' and r[8] == 0.2 and r[9] == 0.4 for r in rows_)
                   and not any(r[2] == 'region' for r in rows_) and len([r for r in rows_ if r[4] == 0.0]) == 0, 'C49 plotted_data gives the line with its band, the bar and the interval, and not the guide line (%s)' % rows_)
            fig, ax = plt.subplots(); EF.FEW_SEEDS = True; EF.N_SEEDS = 2
            EF.band(ax, [1, 2, 3], [0.2, 0.4, 0.6], [0.1, 0.3, 0.5], [0.3, 0.5, 0.9], EF.BLUE, '-', 'o', 'curve'); drawn_few = len(ax.collections); EF.FEW_SEEDS = False; EF.N_SEEDS = 0
            expect(drawn_few == 0, 'C50 with fewer than four seeds no interval is drawn around a curve')
            plt.close(fig)
        except ImportError:
            expect(True, 'C49-C50 (matplotlib missing: figure data not tested)')
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
    expect(set(names) >= {'env.sh', '00_check_env.sh', '01_build_dataset.sh', '02_generate_synthetic.sh', '03_pretrain_finetune.sh', '04_reinforcement_learning.sh', '05_evaluate.sh', '06_benchmarks.sh', '07_report.sh', '08_soundness.sh', '09_audit.sh', 'run_all.sh', 'smoke_test.sh', 'status.sh'}, 'I1 the scripts of the documentation exist (%s)' % names)
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


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ J. the profile and the arithmetic of the bench
def sec_J(c):
    import types, threading, subprocess
    import gym_env as G
    import exp_profile as PF, exp_bench as XB
    # J1. percentiles
    s = PF.summarize([0.001, 0.002, 0.003, 0.004, 0.005])
    expect(s['n'] == 5 and close(s['mean_ms'], 3.0) and close(s['p50_ms'], 3.0) and close(s['p90_ms'], 4.6) and close(s['p99_ms'], 4.96) and close(s['max_ms'], 5.0) and close(s['total_s'], 0.015), 'J1 percentiles of 1..5 ms (linear interpolation): %s' % (s,))
    expect(PF.summarize([]) == {'n': 0} and PF.summarize([0.002])['p99_ms'] == PF.summarize([0.002])['max_ms'] == 2.0, 'J1 no samples: n = 0; one sample: every percentile is it')
    # J2. the ledger: exclusive time, inclusive latency
    T_ = [0.0]; L = PF.Ledger(clock=lambda: T_[0])
    L.enter('a'); T_[0] = 1.0; L.enter('b'); T_[0] = 4.0; d_b = L.leave('kb', 'ok'); T_[0] = 6.0; d_a = L.leave('ka')
    expect(close(d_b, 3.0) and close(d_a, 6.0) and L.samples['kb'][0] == 3.0 and L.samples['ka'][0] == 6.0, 'J2 a latency is inclusive: the inner call 3, the outer one 6')
    expect(close(L.excl['a'], 3.0) and close(L.excl['b'], 3.0) and L.calls['a'] == L.calls['b'] == 1 and L.outcomes['kb'] == {'ok': 1} and 'ka' not in L.outcomes, 'J2 the time is charged to the innermost category only: a gets 1 + 2, b gets 3')
    r = L.report(6.0)
    expect(close(sum(v['seconds'] for v in r['categories'].values()) + r['other_s'], 6.0) and close(r['categories']['a']['share'], 0.5) and r['other_s'] == 0.0 and close(r['accounted_share'], 1.0) and list(r['categories']) == ['a', 'b'], 'J2 the categories add up to the wall time')
    r = L.report(8.0)
    expect(close(r['other_s'], 2.0) and close(r['other_share'], 0.25) and close(r['accounted_share'], 0.75), 'J2 what no category accounts for is `other`')
    expect(L.balanced(), 'J2 every call returned: the ledger is balanced')
    L.enter('x'); expect(not L.balanced(), 'J2 a call that has not returned leaves the ledger unbalanced')
    L.reset(); T_[0] = 0.0
    for cat, t_ in (('a', 0.0), ('b', 1.0), ('c', 2.0)): T_[0] = t_; L.enter(cat)
    for t_ in (5.0, 6.0, 9.0): T_[0] = t_; L.leave()
    expect([round(L.excl[x], 6) for x in 'abc'] == [4.0, 2.0, 3.0] and L.balanced(), 'J2 three levels deep: a 1 + 3, b 1 + 1, c 3 (9 in all)')
    # J3. the hooks, on a stand-in for the engine
    mod = types.ModuleType('s51_fake_engine'); T_ = [0.0]
    class Eng:
        def menu(self, g): T_[0] += 1.0; self.probe_status(g, types.SimpleNamespace(kind=types.SimpleNamespace(name='APPLY')), None); T_[0] += 2.0; return types.SimpleNamespace(actions=[1, 2, 3])
        def probe_status(self, g, action, menu, fuel=None): T_[0] += 4.0; return (None, 'refused')
        def boom(self): T_[0] += 1.0; raise KeyError('x')
    mod.Eng = Eng; sys.modules['s51_fake_engine'] = mod; saved = PF.HOOKS
    orig = {n: vars(Eng)[n] for n in ('menu', 'probe_status', 'boom')}
    PF.HOOKS = (('s51_fake_engine', 'Eng', 'menu', 'menu', lambda a, kw: 'menu', None, lambda r: len(r.actions)), ('s51_fake_engine', 'Eng', 'probe_status', 'probe', PF._probe_key, PF._probe_tag, None),
                ('s51_fake_engine', 'Eng', 'boom', 'boom', lambda a, kw: 'boom', None, None), ('s51_no_such_module', 'X', 'y', 'menu', None, None, None), ('s51_fake_engine', 'Eng', 'absent', 'menu', None, None, None))
    L = PF.Ledger(clock=lambda: T_[0])
    try:
        with PF.instrument(L) as inst:
            expect(inst is L and all(getattr(vars(Eng)[n], '__itp_profile__', False) for n in orig), 'J3 inside the block the methods are the wrappers')
            res = Eng().menu(None)
            expect(len(res.actions) == 3, 'J3 a wrapper returns what the method returns')
            expect(close(L.excl['menu'], 3.0) and close(L.excl['probe'], 4.0) and close(L.samples['menu'][0], 7.0) and close(L.samples['probe:APPLY'][0], 4.0) and L.outcomes['probe:APPLY'] == {'refused': 1} and list(L.values['menu#n']) == [3.0],
                   'J3 a menu that probes: its own 1 + 2 seconds, the probe 4 (7 inclusive), the probe is keyed by the kind of its action and tagged with how it ended, the menu with its number of options')
            raises('J3 an exception goes through the wrapper', lambda: Eng().boom(), KeyError)
            expect(L.balanced() and L.outcomes['boom'] == {'raised': 1} and close(L.excl['boom'], 1.0), 'J3 ... and the ledger is balanced, the call counted as raised')
            raises('J3 a second profile cannot be installed inside the first', lambda: PF.instrument(PF.Ledger()).__enter__(), RuntimeError)
            expect(vars(Eng)['menu'].__wrapped__ is orig['menu'] and PF._ACTIVE == [L], 'J3 ... and the refusal leaves the first profile as it was (each method wrapped once)')
            n_before = L.calls['probe']; got = []
            th = threading.Thread(target=lambda: got.append(Eng().probe_status(None, types.SimpleNamespace(kind=types.SimpleNamespace(name='APPLY')), None))); th.start(); th.join()
            expect(got == [(None, 'refused')] and L.calls['probe'] == n_before and L.balanced(), 'J3 a call from another thread runs, unmeasured, without touching the ledger')
        expect(all(vars(Eng)[n] is orig[n] for n in orig) and PF._ACTIVE == [], 'J3 after the block the methods are the originals and no profile is installed')
        expect(len(L.skipped) == 2 and any('s51_no_such_module' in x for x in L.skipped) and any('absent' in x for x in L.skipped), 'J3 a hook whose module or method is missing is skipped and reported (%s)' % (L.skipped,))
        try:
            with PF.instrument(PF.Ledger(clock=lambda: T_[0])): raise ValueError('inside')
        except ValueError: pass
        expect(all(vars(Eng)[n] is orig[n] for n in orig) and PF._ACTIVE == [], 'J3 an exception inside the block still takes the hooks out')
        with PF.instrument(PF.Ledger(clock=lambda: T_[0]), only={'boom'}):
            expect(getattr(vars(Eng)['boom'], '__itp_profile__', False) and vars(Eng)['menu'] is orig['menu'], 'J3 `only` installs the hooks of some categories')
    finally:
        PF.HOOKS = saved; sys.modules.pop('s51_fake_engine', None)
    expect(PF._probe_key((None, None, G.Action(G.Kind.APPLY, 0), None), {}) == 'probe:APPLY' and PF._probe_key((None, None), {'action': G.Action(G.Kind.RING, 0)}) == 'probe:RING' and PF._probe_key((None, None), {}) == 'probe:None', 'J3 the key of a probe is its kind (positional or by keyword)')
    expect(PF._probe_tag(([], 'ok')) == 'ok' and PF._probe_tag(None) is None, 'J3 the tag of a probe is its status')
    # J4. memory: the process tree
    if os.path.exists('/proc/self/smaps_rollup'):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            time.sleep(0.3)
            expect(child.pid in PF.children_of(os.getpid()) and os.getpid() not in PF.children_of(os.getpid()), 'J4 the children of this process include a process it started')
            expect(PF.pss_mb(os.getpid()) > 1.0 and PF.rss_of(child.pid) > 1.0 and PF.pss_mb(os.getpid()) <= PF.rss_of(os.getpid()) + 1.0 and PF.pss_mb(2 ** 22) is None and PF.rss_of(2 ** 22) is None, 'J4 proportional and resident set sizes (None for a process that does not exist)')
            t = PF.Tally(os.getpid()); t.take(); t.take(); res = t.result()
            expect(res['samples'] == 2 and res['processes_max'] >= 2 and res['pss_mb_max'] > 0 and res['rss_sum_mb_max'] >= res['pss_mb_max'] - 1.0, 'J4 a tally of the readings: the maxima of the sums over the process and its children (%s)' % (res,))
            with PF.ProcessSampler(every=0.1) as ps: time.sleep(2.5)
            res = ps.result()
            expect(res['samples'] >= 2 and res['processes_max'] >= 2 and res['pss_mb_max'] > 0 and res['rss_sum_mb_max'] >= res['pss_mb_max'] - 1.0, 'J4 the sampler (a separate process) sees the child (%s)' % (res,))
        finally:
            child.kill(); child.wait()
        expect(not any(p_ for p_ in PF.children_of(os.getpid())), 'J4 the sampler process is gone when the block ends')
    # J5. what is a result and what is the clock
    expect(all(XB.is_volatile(f) for f in ('seconds', 't_policy', 'time', 'wall_s', 'elapsed_s', 'host', 'worker')) and not any(XB.is_volatile(f) for f in ('solved', 'probes', 'decisions', 'steps', 'proof', 'name', 'regime', 'error', 'proof_error', 'boot_uses', 'method', 'seed', 'family')), 'J5 clock fields are volatile, results are not')
    row = {'name': 'p', 'regime': 'prefix', 'method': 'blind', 'seed': 0, 'solved': True, 'probes': 12, 'seconds': 0.5, 'steps': 3}
    expect(XB.canonical(row) == {'name': 'p', 'regime': 'prefix', 'method': 'blind', 'seed': 0, 'solved': True, 'probes': 12, 'steps': 3}, 'J5 canonical() drops the clock')
    def rows_of(**over):
        out = []
        for i, nm in enumerate(('a', 'b', 'c')):
            r = dict(row, name=nm, probes=10 + i, seconds=0.1 * (i + 1)); r.update(over.get(nm, {})); out.append(r)
        return out
    base = rows_of()
    same = XB.compare_rows({'w1': base, 'w2': list(reversed(rows_of(a={'seconds': 9.0}))), 'w0': rows_of(c={'seconds': 0.001})}, 100.0)
    expect(same['identical'] and same['tasks'] == 3 and same['n_differing'] == 0 and same['missing'] == 0 and same['excluded_slow'] == 0 and same['differing'] == [], 'J5 the same rows in another order and with other seconds are identical (%s)' % (same,))
    other = XB.compare_rows({'w1': base, 'w2': rows_of(b={'probes': 99})}, 100.0)
    expect(not other['identical'] and other['n_differing'] == 1 and other['differing'][0]['task'] == ['blind', 0, 'b', 'prefix'] and other['differing'][0]['fields'] == ['probes'] and other['differing'][0]['run'] == 'w2', 'J5 a different number of probes is reported with the task, the run and the field (%s)' % (other['differing'],))
    d1 = XB.compare_rows({'x': base}, 100.0)['outcome_digest']; d2 = XB.compare_rows({'y': rows_of(a={'seconds': 77.0})}, 100.0)['outcome_digest']; d3 = XB.compare_rows({'z': rows_of(a={'probes': 1})}, 100.0)['outcome_digest']
    expect(d1 == d2 == same['outcome_digest'] and d3 != d1 and len(d1) == 16, 'J5 the digest does not depend on the clock, the label of the run or the order of the rows, and does depend on the outcomes')
    miss = XB.compare_rows({'w1': base, 'w2': base[:2]}, 100.0)
    expect(not miss['identical'] and miss['missing'] == 1 and miss['tasks'] == 2, 'J5 a task one run lacks makes the runs different (%s)' % (miss,))
    slow = XB.compare_rows({'w1': base, 'w2': rows_of(a={'probes': 55, 'seconds': 95.0})}, 100.0)
    expect(slow['excluded_slow'] == 1 and slow['tasks'] == 2 and slow['identical'], 'J5 an attempt that used 90%% of the time limit is left out (the clock ended it) and counted (%s)' % (slow,))
    expect(not XB.compare_rows({'w1': []}, 100.0)['identical'] and not XB.compare_rows({}, 100.0)['identical'], 'J5 nothing compared is not "identical"')
    # J6. the machine
    m = XB.machine(1)
    expect(all(f in m for f in ('cpu', 'cores', 'cores_usable', 'memory_gb', 'kernel', 'python', 'torch', 'numpy', 'torch_threads', 'kernel_digest', 'loadavg_start', 'busy_at_start')) and XB.usable_cores() >= 1 and m['cores_usable'] >= 1, 'J6 the machine record has the fields the report reads')
    expect(len(m['kernel_digest']) == 16 and m['kernel_digest'] == XB.kernel_digest() and all(ch in '0123456789abcdef' for ch in m['kernel_digest']), 'J6 the digest of the kernel sources is 16 hex digits and stable')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ K. the bench on the small world
def sec_K(c):
    import gym_env as G, ml_agents as AG, ml_model as MM
    import exp_profile as PF, exp_bench as XB
    fx = c['mini']; bk, ctx, eng = fx.bk, fx.ctx, fx.eng; quiet = lambda *a: None
    names = XB.pick_problems(bk, fx.ddir, 3)
    expect(len(names) == 3 and all(n in bk.placed for n in names) and names == XB.pick_problems(bk, fx.ddir, 3), 'K1 the problems of the bench are picked deterministically from the train problems with a reference proof')
    # K2. a profile changes no result and accounts for the time
    solver = AG.make_solver('blind', eng)
    rows0, secs0 = XB.solve_all(solver, bk, names, 40, 60.0)
    L = PF.Ledger()
    with PF.instrument(L):
        t0 = time.perf_counter(); rows1, secs1 = XB.solve_all(solver, bk, names, 40, 60.0); wall = time.perf_counter() - t0
    expect(XB.outcome_of(rows0) == XB.outcome_of(rows1) and len(rows0) == 3, 'K2 the instrumented pass finds what the plain pass finds (same proofs, same probes)')
    expect(not getattr(G.Engine.menu, '__itp_profile__', False) and not getattr(G.Engine.probe_status, '__itp_profile__', False) and PF._ACTIVE == [], 'K2 the engine is as it was after the block')
    r = L.report(wall)
    expect(L.balanced() and L.calls['probe'] > 0 and L.calls['menu'] > 0 and L.calls['tactic'] > 0 and 0.3 < r['accounted_share'] <= 1.0 + 1e-9, 'K2 the engine\'s categories account for most of the wall time and not more than all of it (%.2f; %s)' % (r['accounted_share'], {k_: v['calls'] for k_, v in r['categories'].items()}))
    spent = sum(x['probes'] for x in rows1); unsolved = sum(1 for x in rows1 if not x['solved'])
    expect(0 <= spent - L.calls['probe'] <= unsolved, 'K2 the probes of the engine are the probes of the rows (an unsolved search counts the one it was refused): %d calls, %d counted, %d unsolved' % (L.calls['probe'], spent, unsolved))
    lat = XB.engine_latency(L)
    expect(lat['probe']['n'] == L.calls['probe'] and sum(v['n'] for v in lat['probe_by_kind'].values()) == lat['probe']['n'] and lat['menu']['n'] == L.calls['menu'] and lat['menu']['mean_options'] > 0, 'K2 the latency report: probes by kind add up, the menus have a number of options')
    expect(all(sum(v['outcomes'].values()) == v['n'] and set(v['outcomes']) <= {'ok', 'refused', 'fuel', 'bug', 'raised'} for v in lat['probe_by_kind'].values()), 'K2 every probe is tagged with how it ended')
    expect(lat['probe']['p50_ms'] <= lat['probe']['p90_ms'] <= lat['probe']['p99_ms'] <= lat['probe']['max_ms'] and lat['probe']['max_ms'] * lat['probe']['n'] >= lat['probe']['total_s'] * 1e3 - 1e-6, 'K2 the percentiles are ordered')
    # K3. the solvers: repeats, the instrumented pass, the network's share
    pool = ({}, {}, {}); net = fx.tiny_net(ctx)
    out = XB.bench_solvers(bk, ctx, eng, net, names, ('blind', 'uniform', 'sample'), 30, 60.0, say=quiet, repeats=2, pool=pool)
    expect(set(out) == {'blind', 'uniform', 'sample'} and all(v['repeats'] == 2 and len(v['probes_per_s_reps']) == 2 for v in out.values()), 'K3 every solver was measured in two passes')
    expect(all(v['probes_per_s_min'] <= v['probes_per_s'] <= v['probes_per_s_max'] for v in out.values()) and all(v['stable'] and v['profile']['same_outcome'] for v in out.values()), 'K3 the median is inside the range, the passes agree, and so does the instrumented one')
    expect(all(v['profile']['balanced'] and not v['profile']['hooks_skipped'] and v['profile']['ratio_to_plain'] > 0 for v in out.values()), 'K3 the instrumented passes are balanced and installed every hook')
    cats = out['sample']['profile']['breakdown']['categories']
    expect('policy' in cats and cats['policy']['calls'] > 0 and 'assembler' in cats and 'policy' not in out['blind']['profile']['breakdown']['categories'], 'K3 the network\'s time is its own category, and only for the solvers that use it (%s)' % (sorted(cats),))
    expect(pooled_ok(XB, pool), 'K3 the pooled latencies hold the probes of all three solvers')
    # K4. the Gymnasium interface
    g = XB.bench_gym(bk, ctx, eng, names, decisions=60, time_limit=30.0, profile_decisions=40)
    pr = g['profiled']
    expect(g['decisions'] >= 60 and g['decisions_per_s'] > 0 and g['episodes'] >= 1 and pr['latency']['decision']['n'] >= 40 and pr['latency']['reset']['n'] >= 1, 'K4 the interface: decisions per second, and the latency of a decision and of a reset')
    expect(pr['latency']['decision']['n'] == pr['latency'].get('decision_inner', {}).get('n', 0) + pr['latency'].get('decision_action', {}).get('n', 0), 'K4 a decision either completes an action or it does not')
    expect(0.3 < pr['breakdown']['accounted_share'] <= 1.0 + 1e-9 and 'assembler' in pr['breakdown']['categories'] and not pr['hooks_skipped'], 'K4 the time of the interface is accounted for (the options of the decision language, the features, the menu)')
    # K5. workers: scaling, memory, determinism
    wn = XB.pick_problems(bk, fx.ddir, 4)
    res, det = XB.bench_workers(bk, ctx, eng, '', wn, [1, 2], 25, 60.0, say=quiet)
    expect(set(res) == {'1', '2'} and res['1']['speedup'] == 1.0 and res['1']['efficiency'] == 1.0 and res['2']['tasks'] == res['1']['tasks'] and res['1']['probes'] == res['2']['probes'], 'K5 the same work with one and two workers (%s)' % ({k_: (v['tasks'], v['probes']) for k_, v in res.items()},))
    expect(all(v['memory']['pss_mb_max'] > 0 for v in res.values()) or not os.path.exists('/proc/self/smaps_rollup'), 'K5 the memory of the processes is measured (%s)' % ({k_: v['memory'] for k_, v in res.items()},))
    expect(det['identical'] and det['tasks'] == res['1']['tasks'] and det['runs'] == ['in process', '1 workers', '2 workers'] and det['n_differing'] == 0, 'K5 the rows of an in-process run and of one and two workers are identical but for the clock (%s)' % (det,))
    # K6. the cost of trusting a proof
    prem0 = eng.premises
    chk = XB.bench_check(bk, ctx, eng, fx.ddir, limit=3, say=quiet)
    expect(chk['proofs'] == 3 and chk['replay_failed'] == 0 and chk['independent_accepted'] == 3 and chk['kernel_check']['n'] == 3 and chk['independent_check']['n'] == 3 and chk['replay']['p50_ms'] >= chk['kernel_check']['p50_ms'], 'K6 three reference proofs: replayed (the kernel checks each term) and accepted by the independent checker (%s)' % ({k_: v for k_, v in chk.items() if not isinstance(v, dict)},))
    expect(eng.premises == prem0, 'K6 the replays leave the premises of the engine as they were')
    # K7. the whole script
    outdir = os.path.join(fx.tmp, 'bench_out'); t0 = time.time()
    XB.main([fx.ddir, '--out', outdir, '--problems', '3', '--probes', '30', '--repeats', '2', '--workers', '1,2', '--worker-problems', '3', '--worker-probes', '20', '--check-proofs', '2', '--no-train'])
    b = json.load(open(os.path.join(outdir, 'bench.json')))
    expect(set(b) >= {'n_problems', 'problems', 'probes', 'time_limit', 'repeats', 'gym', 'solvers', 'latency', 'overhead', 'check', 'workers', 'determinism', 'world', 'machine', 'meta'}, 'K7 bench.json has every block (%s)' % (sorted(b),))
    expect(set(b['solvers']) == {'blind', 'uniform'} and b['gym']['profiled']['latency']['decision']['n'] > 0 and b['latency']['probe']['n'] > 0 and set(b['overhead']) == {'blind', 'uniform'} and b['determinism']['identical'], 'K7 without a model: the two solvers that need none; the latencies and the determinism check are there')
    expect(b['machine']['kernel_digest'] == XB.kernel_digest() and 'loadavg_end' in b['machine'] and b['meta']['commit'] is not None, 'K7 the machine and the commit are recorded')
    expect(not any(f.endswith('.pyc') for f in os.listdir(outdir)) and sorted(os.listdir(outdir)) == ['bench.json'], 'K7 the script writes bench.json and nothing else')
    trace('K7 %.1f s' % (time.time() - t0))


def pooled_ok(XB, pool):
    lat = XB.pooled_latency(pool)
    return lat['probe']['n'] > 0 and lat['menu']['n'] > 0 and sum(v['n'] for v in lat['probe_by_kind'].values()) == lat['probe']['n']


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ L. the report of the bench
def bench_fixture():
    """a bench.json as exp_bench writes it, with numbers chosen by hand: the ledgers are real (exp_profile.Ledger on a clock that this function moves) and the latency summaries are exp_profile.summarize of
    1..100 ms, so every cell of every table is known in advance (a sample of 1..100 ms has median 50.5, 90th percentile 90.1, 99th percentile 99.01 and maximum 100)"""
    import exp_profile as PF
    clock = [0.0]
    def ledger(spend):
        L = PF.Ledger(clock=lambda: clock[0])
        for cat, secs in spend:
            L.enter(cat); clock[0] += secs; L.leave()
        return L
    S, ms100 = PF.summarize, [i / 1000.0 for i in range(1, 101)]
    gym_l = ledger([('assembler', 15.0), ('tactic', 0.5), ('menu', 1.0), ('probe', 0.5), ('features', 1.0)])                                       # of a wall time of 20 s: 75, 2.5, 5, 2.5, 5 and 10% in no category
    blind_l = ledger([('tactic', 4.0), ('menu', 3.0), ('probe', 1.5)])                                                                              # of 10 s: 40, 30, 15 and 15%
    samp_l = ledger([('policy', 2.5), ('tactic', 2.0), ('menu', 1.0), ('probe', 1.0), ('assembler', 1.0), ('features', 0.5)])                       # of 10 s: 25, 20, 10, 10, 10, 5 and 20%
    kinds = collections.OrderedDict(zip(['APPLY', 'LINARITH', 'CASES', 'REWRITE', 'NLINARITH', 'REWRITE_WITH', 'BY_CONTRA', 'REWRITE_AT', 'RING', 'APPLY_WITH', 'INDUCT_WITH', 'USE', 'FORWARD', 'SIMP'], [20, 15, 12, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1]))
    by_kind, i0 = {}, 0
    for k, n in kinds.items():
        by_kind[k] = dict(S(ms100[i0:i0 + n]), outcomes={'ok': n}); i0 += n
    def mem(pss, rss, procs): return {'pss_mb_max': pss, 'rss_sum_mb_max': rss, 'processes_max': procs, 'samples': 40}
    def entry(pps, reps, **kw):
        d = {'n': 3, 'solved': 1, 'probes': 120, 'seconds': 0.1, 'probes_per_s': pps, 'decisions_per_s': None, 'actions_per_s': None, 'seconds_per_problem': 0.034, 'peak_rss_mb': 2048.0, 'rss_mb': 2048.0, 'rss_delta_mb': 0.1,
             'repeats': len(reps), 'probes_per_s_reps': reps, 'probes_per_s_min': min(reps), 'probes_per_s_max': max(reps), 'stable': True}; d.update(kw); return d
    return {
        'n_problems': 3, 'problems': ['a', 'b', 'c'], 'probes': 40, 'time_limit': 20.0, 'repeats': 3,
        'gym': {'decisions': 2400, 'probes': 60, 'episodes': 4, 'seconds': 3.3, 'decisions_per_s': 725.4, 'probes_per_s': 18.1,
                'profiled': {'decisions': 1200, 'seconds': 20.0, 'breakdown': gym_l.report(20.0), 'engine_latency': {}, 'hooks_skipped': [],
                             'latency': {'reset': S([0.006, 0.006]), 'decision': S(ms100), 'decision_inner': S(ms100[:50]), 'decision_action': S(ms100[50:])}}},
        'solvers': {'blind': entry(1200.0, [1190.0, 1200.0, 1210.0], profile={'breakdown': blind_l.report(10.0), 'latency': {'probe': S(ms100)}, 'seconds': 10.0, 'plain_seconds_median': 8.0, 'ratio_to_plain': 1.25, 'same_outcome': True,
                                                                                'probe_calls': 120, 'probes_counted': 123, 'balanced': True, 'hooks_skipped': []}),
                    'sample': entry(20.0, [20.0], decisions_per_s=900.0, actions_per_s=15.0, peak_rss_mb=3072.0, profile={'breakdown': samp_l.report(10.0), 'latency': {'probe': S(ms100[:10])}, 'seconds': 10.0, 'plain_seconds_median': 11.0,
                                                                                                                        'ratio_to_plain': 0.9, 'same_outcome': True, 'probe_calls': 40, 'probes_counted': 40, 'balanced': True, 'hooks_skipped': []})},
        'latency': {'probe': S(ms100), 'probe_by_kind': by_kind, 'menu': dict(S(ms100), mean_options=35.0), 'replay': S(ms100[:50]), 'start': S([0.0005] * 6), 'verify': S(ms100[:10])},
        'overhead': {'blind': {'ratio': 1.25, 'same_outcome': True}, 'sample': {'ratio': 0.9, 'same_outcome': True}},
        'check': {'proofs': 5, 'replay_failed': 0, 'steps_mean': 6.4, 'replay': S([0.01] * 5), 'kernel_check': S([0.004] * 5), 'independent_check': S([0.4] * 5), 'independent_accepted': 5, 'independent_setup_s': 1.1},
        'workers': {'1': {'seconds': 10.0, 'probes': 100, 'probes_per_s': 10.0, 'tasks': 8, 'memory': mem(1000.0, 1100.0, 2), 'speedup': 1.0, 'efficiency': 1.0},
                    '2': {'seconds': 5.2, 'probes': 100, 'probes_per_s': 19.2, 'tasks': 8, 'memory': mem(1010.0, 2000.0, 3), 'speedup': 1.92, 'efficiency': 0.96},
                    '4': {'seconds': 2.9, 'probes': 100, 'probes_per_s': 35.0, 'tasks': 8, 'memory': mem(1020.0, 3900.0, 2), 'speedup': 3.5, 'efficiency': 0.875}},
        'determinism': {'runs': ['in process', '1 workers', '2 workers', '4 workers'], 'tasks': 8, 'identical': True, 'n_differing': 0, 'differing': [], 'missing': 0, 'excluded_slow': 0, 'outcome_digest': 'abcdef0123456789'},
        'world': {'seconds': 7.4, 'globals': 571, 'problems': 19, 'rss_mb': 2048.0, 'params': 2.5e6, 'train_steps_per_s': 55.0, 'infer_decisions_per_s': 300.0},
        'machine': {'cpu': 'Test CPU @ 2.0GHz', 'cores': 4, 'cores_usable': 2, 'memory_gb': 16.0, 'kernel': '6.1', 'platform': 'Linux-x', 'python': '3.13', 'torch': '2.0', 'numpy': '2.0', 'torch_threads': 1, 'kernel_digest': '0123456789abcdef',
                    'loadavg_start': [0.1, 0.1, 0.1], 'busy_at_start': False, 'loadavg_end': [0.3, 0.2, 0.1]},
        'meta': {'commit': 'a' * 40, 'tree_dirty': False}}


def sec_L(c):
    import types, csv as _csv, exp_bundle as EB, exp_figures as FG
    b = bench_fixture(); tmp = tempfile.mkdtemp(prefix='s51l_')
    try:
        R = os.path.join(tmp, 'runs'); os.makedirs(os.path.join(R, 'bench'))
        json.dump({'name': 'benchonly', 'seed_list': [0, 1]}, open(os.path.join(R, 'PROFILE.json'), 'w'))
        json.dump(b, open(os.path.join(R, 'bench', 'bench.json'), 'w'))
        kern = {'conv fib k': {'8': 1.0, '16': 3.0}, 'tiny normalize': {'1': 0.002}, 'empty': {}}; json.dump(kern, open(os.path.join(R, 'bench', 'kernel.json'), 'w'))
        C = EC.load(R); T = TB.throughput_tables(C)
        expect(set(T) == {'bench_solvers', 'bench_breakdown', 'bench_latency', 'bench_workers', 'bench_world', 'bench_machine', 'bench_kernel'}, 'L1 the tables of a bench: %s' % sorted(T))
        # ---- throughput
        t = T['bench_solvers']
        expect(t.header == ['solver', 'probes / s', 'range', 'decisions / s', 'actions / s', 'seconds / problem', 'probe median (ms)', 'probe p99 (ms)', 'peak RSS (GB)'], 'L2 the throughput table has the columns of the instrumented bench (%s)' % t.header)
        expect(t.rows[0] == ['blind', '1200.0', '1190.0–1210.0', '--', '--', '0.03', '50.50', '99.0', '2.00'], 'L3 a prover with repeated passes: the median, the range of the passes, its latency from the instrumented pass (%s)' % t.rows[0])
        expect(t.rows[1] == ['sample', '20.0', '--', '900.0', '15.0', '0.03', '5.50', '9.9', '3.00'], 'L4 a prover with a single pass has no range, and the decisions and actions of a policy (%s)' % t.rows[1])
        expect(any('median of 3 passes' in n for n in t.notes) and any('0.90 to 1.25 times' in n for n in t.notes), 'L5 the notes say how many passes the median is of and what the instrumented pass cost (%s)' % t.notes)
        # ---- where the time goes
        t = T['bench_breakdown']; groups = list(TB.BREAKDOWN_GROUPS)
        expect(t.header == ['share of the time (%)', 'Gymnasium interface', 'blind', 'sample'] and [r[0] for r in t.rows] == groups + [TB.OTHER_LABEL], 'L6 the columns are the interface and the provers, the rows the groups and the remainder (%s)' % [r[0] for r in t.rows])
        want = {'tactics and kernel': ['2.5', '40.0', '20.0'], 'menu of a goal': ['5.0', '30.0', '10.0'], 'probe machinery': ['2.5', '15.0', '10.0'], 'decision language': ['75.0', '0.0', '10.0'], 'features': ['5.0', '0.0', '5.0'],
                'policy network': ['0.0', '0.0', '25.0'], TB.OTHER_LABEL: ['10.0', '15.0', '20.0']}
        expect(all(r[1:] == want[r[0]] for r in t.rows), 'L7 the shares of the ledgers are the shares of the time spent (%s)' % t.rows)
        expect(all(abs(sum(float(r[k]) for r in t.rows) - 100.0) <= 0.31 for k in (1, 2, 3)), 'L8 every column of the breakdown adds up to 100%% (the remainder is what no category accounts for)')
        # ---- latency
        t = T['bench_latency']; rows = {r[0]: r for r in t.rows}
        expect(rows['probe, any kind'][1:6] == [100, '50.50', '90.10', '99.01', '100.0'] and rows['probe, any kind'][6] == 'ok 100', 'L9 the probe row: n, median, 90th and 99th percentile, maximum, the outcomes (%s)' % rows['probe, any kind'])
        kind_rows = [r for r in t.rows if r[0].startswith('  probe of ')]
        expect([r[0] for r in kind_rows] == ['  probe of ' + k for k in list(bench_fixture()['latency']['probe_by_kind'])[:TB.KIND_ROWS]] and [r[1] for r in kind_rows] == [20, 15, 12, 10, 9, 8, 7, 6, 5, 3, 2, 1],
               'L10 the kinds of action are listed by how often they were tried, the %d most frequent ones (%s)' % (TB.KIND_ROWS, [(r[0], r[1]) for r in kind_rows]))
        expect(any('The 2 rarer kinds of action are left out' in n for n in t.notes), 'L11 the table says that rarer kinds were left out (%s)' % t.notes)
        expect(rows['menu of a goal'][6] == '35 options on average' and rows['menu of a goal'][2] == '50.50', 'L12 the menu row carries the number of options')
        expect(rows['reference proof: the independent checker'][2] == '400.00' and rows['reference proof: the kernel\'s check of the finished term'][2] == '4.00' and rows['reference proof: replay, then kernel check'][6] == '5 proofs, 6.4 steps on average',
               'L13 the cost of checking a proof: replay, kernel, the independent checker (%s)' % [rows[k][2] for k in rows if k.startswith('reference')])
        expect([rows[k][2] for k in ('Gymnasium: reset', 'Gymnasium: a decision', '  a decision inside an action', '  a decision that completes an action')] == ['6.00', '50.50', '25.50', '75.50'], 'L14 the interface rows')
        tex = t.tex()
        expect('REWRITE\\_WITH' in tex and '\\begin{tabular}' in tex and '_' not in tex.split('\\midrule')[1].replace('\\_', ''), 'L15 the LaTeX of the latency table escapes the underscore of a kind of action')
        # ---- workers
        t = T['bench_workers']
        expect(t.rows[0] == ['1', '10.0', '1.00', '1.00', '1000', '1100', 2] and t.rows[1] == ['2', '19.2', '1.92', '0.96', '1010', '2000', 3] and t.rows[2][4:] == ['1020', '3900', 2], 'L16 the worker rows: throughput, speed-up, efficiency, memory counting the shared world once and the sum of the resident sets, the processes (%s)' % t.rows)
        expect(any('identical but for the clock (8 tasks compared; outcome digest abcdef0123456789)' in n for n in t.notes), 'L17 the note says the rows were identical and gives the digest (%s)' % t.notes)
        expect(any('With 4 workers the memory sampler saw fewer processes' in n for n in t.notes) and not any('With 2' in n for n in t.notes), 'L18 a run too short for the memory sampler is named (4 workers: 2 processes seen, 5 expected) and only that one (%s)' % t.notes)
        b2 = copy.deepcopy(b); b2['determinism'].update(identical=False, n_differing=3, differing=[{'task': 'x'}]); t2 = TB.throughput_tables(types.SimpleNamespace(bench={'throughput': b2, 'kernel': None}))['bench_workers']
        expect(any(n.startswith('THE ROWS DIFFER between in process, 1 workers, 2 workers, 4 workers: 3 of 8 tasks') for n in t2.notes), 'L19 rows that differ between worker counts are said loudly (%s)' % t2.notes)
        # ---- fixed costs, machine, kernel
        w = {r[0]: r[1] for r in T['bench_world'].rows}
        expect(w['policy network parameters'] == '2.50 M' and w['BC training steps / s (batch)'] == '55' and w['world build (the whole bank)'] == '7.4 s' and w['resident memory after the build'] == '2.00 GB', 'L20 the fixed costs (%s)' % w)
        m = {r[0]: r[1] for r in T['bench_machine'].rows}
        expect(m['processor'] == 'Test CPU @ 2.0GHz' and m['cores (usable by the benchmark)'] == '4 (2)' and m['commit'] == 'a' * 12 and not T['bench_machine'].notes, 'L21 the machine table (%s)' % m)
        b2 = copy.deepcopy(b); b2['machine']['busy_at_start'] = True; b2['meta']['tree_dirty'] = True; t2 = TB.throughput_tables(types.SimpleNamespace(bench={'throughput': b2, 'kernel': None}))['bench_machine']
        expect(any('machine was busy' in n for n in t2.notes) and dict((r[0], r[1]) for r in t2.rows)['commit'].endswith('(modified tree)'), 'L22 a busy machine and a modified tree are said')
        expect(T['bench_kernel'].rows == [['conv fib k', 16, '3.00'], ['tiny normalize', 1, '0.00']], 'L23 the kernel micro-benchmarks: the largest size of each series (%s)' % T['bench_kernel'].rows)
        # ---- an older bench.json
        old = {'n_problems': 3, 'solvers': {'sample': {'probes_per_s': 123.0, 'decisions_per_s': 4.0, 'actions_per_s': 1.0, 'seconds_per_problem': 0.5, 'peak_rss_mb': 1024.0}}, 'workers': {'1': {'probes_per_s': 5.0, 'speedup': 1.0, 'efficiency': 1.0}, '2': {'probes_per_s': 9.0, 'speedup': 1.8, 'efficiency': 0.9}},
               'world': {'seconds': 1.0, 'globals': 5, 'rss_mb': 100.0}}
        Co = types.SimpleNamespace(bench={'throughput': old, 'kernel': None}, profile={}, warnings=[]); To = TB.throughput_tables(Co)
        expect(set(To) == {'bench_solvers', 'bench_workers', 'bench_world'} and To['bench_solvers'].rows[0] == ['sample', '123.0', '--', '4.0', '1.0', '0.50', '--', '--', '1.00'] and To['bench_workers'].rows[1] == ['2', '9.0', '1.80', '0.90', '--', '--', '--'],
               'L24 a bench.json of the old format gives the old tables with "--" where the new measurements are missing (%s)' % {k: v.rows for k, v in To.items() if k != 'bench_world'})
        expect(TB.throughput_tables(types.SimpleNamespace(bench={'throughput': {}, 'kernel': None})) == {} and TB.throughput_tables(types.SimpleNamespace(bench={'throughput': None, 'kernel': None})) == {}
               and set(TB.throughput_tables(types.SimpleNamespace(bench={'throughput': {'solvers': {}}, 'kernel': None}))) <= {'bench_solvers'}, 'L25 no bench, an empty one: no table and no crash')
        # ---- the numbers the text quotes
        N = collections.OrderedDict(); REP.bench_numbers(b, N)
        wantN = {'bench_probes_per_s_blind': 1200.0, 'bench_probes_per_s_sample': 20.0, 'bench_gym_decisions_per_s': 725, 'bench_gym_decision_language_percent': 75, 'bench_gym_engine_percent': 10, 'bench_probe_median_ms': 50.5, 'bench_probe_p99_ms': 99.0,
                 'bench_menu_median_ms': 50.5, 'bench_replay_median_ms': 25.5, 'bench_start_median_ms': 0.5, 'bench_kernel_check_median_ms': 4.0, 'bench_independent_check_median_ms': 400.0, 'bench_independent_check_ratio': 100, 'bench_check_proofs': 5,
                 'bench_check_accepted': 5, 'bench_speedup_2': 1.92, 'bench_speedup_4': 3.5, 'bench_pss_mb_4': 1020, 'bench_rss_sum_mb_4': 3900, 'bench_deterministic': 'yes', 'bench_deterministic_tasks': 8, 'bench_cores': 2, 'bench_instrumentation_ratio_max': 1.25}
        bad = {k: (N.get(k), v) for k, v in wantN.items() if N.get(k) != v}
        expect(not bad, 'L26 the numbers of the benchmark: %s' % bad)
        N2 = collections.OrderedDict(); REP.bench_numbers(old, N2); N3 = collections.OrderedDict(); REP.bench_numbers(None, N3)
        expect(N2 == {'bench_probes_per_s_sample': 123.0, 'bench_speedup_1': 1.0, 'bench_speedup_2': 1.8} and not N3, 'L27 a bench of the old format gives the numbers it has (%s)' % dict(N2))
        expect(REP.texname('bench_pss_mb_4') == 'BenchPssMbFour' and REP.texname('bench_gym_decision_language_percent') == 'BenchGymDecisionLanguagePercent', 'L28 the macro names of these numbers are letters only (%s)' % REP.texname('bench_pss_mb_4'))
        # ---- warnings
        Cb = types.SimpleNamespace(bench={'throughput': b})
        expect(REP.bench_warnings(Cb) == [] and REP.bench_warnings(types.SimpleNamespace(bench={'throughput': None})) == [] and REP.bench_warnings(types.SimpleNamespace(bench={})) == [], 'L29 a clean bench says nothing (%s)' % REP.bench_warnings(Cb))
        def broken(f):
            x = copy.deepcopy(b); f(x); return REP.bench_warnings(types.SimpleNamespace(bench={'throughput': x}))
        def dets(x): x['determinism'].update(identical=False, n_differing=2)
        def busy(x): x['machine'].update(busy_at_start=True, loadavg_start=[3.0, 2.0, 1.0])
        def unstable(x): x['solvers']['sample']['stable'] = False
        def other(x): x['solvers']['blind']['profile']['same_outcome'] = False
        def unbal(x): x['solvers']['blind']['profile']['balanced'] = False
        def skipped(x): x['solvers']['sample']['profile']['hooks_skipped'] = ['gym_env.Engine.nf']
        def replay(x): x['check'].update(replay_failed=1, independent_accepted=4)
        def refused(x): x['check'].update(independent_accepted=3)
        for name, f, frag in (('determinism', dets, 'the rows differ between in process, 1 workers, 2 workers, 4 workers (2 of 8 tasks differ)'), ('busy', busy, 'the machine was busy when it started (load [3.0, 2.0, 1.0] on 2 usable cores)'),
                              ('unstable', unstable, 'sample found different results in repeated passes'), ('instrumented', other, 'the instrumented pass of blind found different results from the plain pass'),
                              ('ledger', unbal, 'the time ledger of blind is unbalanced'), ('hooks', skipped, 'sample was profiled without gym_env.Engine.nf'), ('replay', replay, '1 of 5 reference proofs did not replay'),
                              ('refused', refused, 'the independent checker refused 2 reference proofs that the kernel accepted')):
            ws = broken(f); expect(len(ws) == 1 and frag in ws[0] and ws[0].startswith('benchmark: '), 'L30 warning for %s: %s' % (name, ws))
        ws = broken(replay); expect(len(ws) == 1, 'L30b a proof that did not replay is not also counted as refused by the independent checker (%s)' % ws)
        # ---- the figures
        try:
            import matplotlib  # noqa: F401
            fdir = os.path.join(tmp, 'figs'); figs = FG.make_all(C, fdir, only={'fig_throughput', 'fig_breakdown', 'fig_latency'}, log=lambda *a: None)
            expect(set(figs) == {'fig_throughput', 'fig_breakdown', 'fig_latency'} and all(os.path.getsize(p) > 1500 for ps in figs.values() for p in ps) and not C.warnings, 'L31 the three figures are drawn as pdf and png (%s, warnings %s)' % (sorted(figs), C.warnings))
            def fcsv(name): return list(_csv.DictReader(open(os.path.join(fdir, name + '.csv'))))
            def fl(x): return float(x) if x not in ('', None) else None
            br = fcsv('fig_breakdown'); bars = [r for r in br if r['kind'] == 'bar']
            by_col = collections.defaultdict(float)
            for r in bars: by_col[r['label']] += fl(r['x'])
            expect(len(bars) == 3 * 7 and set(by_col) == {'Gymnasium interface', 'blind', 'sample'} and all(abs(v - 100.0) < 1e-6 for v in by_col.values()), 'L32 fig_breakdown: a bar per group per column, each column adding to 100 (%s)' % dict(by_col))
            expect(any(r['series'] == 'decision language' and r['label'] == 'Gymnasium interface' and abs(fl(r['x']) - 75.0) < 1e-9 for r in bars) and any(r['series'] == 'policy network' and r['label'] == 'sample' and abs(fl(r['x']) - 25.0) < 1e-9 for r in bars), 'L33 fig_breakdown: the segments carry the shares')
            lt = fcsv('fig_latency'); med = {(r['panel'], r['label']): fl(r['x']) for r in lt if r['kind'] == 'line' and r['series'] in ('engine', 'Gymnasium interface', 'checking a proof')}
            p90 = {(r['panel'], r['label']): fl(r['x']) for r in lt if r['kind'] == 'line' and r['series'] == '90th percentile'}
            iv = {(r['panel'], r['label']): (fl(r['x_low']), fl(r['x_high'])) for r in lt if r['kind'] == 'interval'}
            k_a = '(a) operations'; k_b = '(b) probes by kind of action'
            expect(close(med[(k_a, 'one probe, any kind')], 50.5) and close(p90[(k_a, 'one probe, any kind')], 90.1) and close(iv[(k_a, 'one probe, any kind')][1], 99.01) and close(iv[(k_a, 'one probe, any kind')][0], 50.5), 'L34 fig_latency: median, 90th and 99th percentile of a probe')
            expect(close(med[(k_a, 'independent checker')], 400.0) and close(med[(k_a, 'kernel check of a finished proof')], 4.0) and close(med[(k_a, 'one decision')], 50.5), 'L35 fig_latency: the checking rows and the interface rows')
            expect([k for (pn, k) in med if pn == k_b] == ['APPLY (20)', 'LINARITH (15)', 'CASES (12)', 'REWRITE (10)', 'NLINARITH (9)', 'REWRITE_WITH (8)', 'BY_CONTRA (7)', 'REWRITE_AT (6)', 'RING (5)', 'APPLY_WITH (3)'], 'L36 fig_latency: the ten most frequent kinds, with their counts')
            tp = fcsv('fig_throughput'); a_ = {r['label']: fl(r['x']) for r in tp if r['panel'].startswith('(a)') and r['kind'] == 'line'}; a_iv = {r['label']: (fl(r['x_low']), fl(r['x_high'])) for r in tp if r['panel'].startswith('(a)') and r['kind'] == 'interval'}
            expect(a_ == {'blind': 1200.0, 'sample': 20.0} and a_iv == {'blind': (1190.0, 1210.0)}, 'L37 fig_throughput (a): the medians of the passes and the range of the prover that was passed over three times (%s, %s)' % (a_, a_iv))
            sp = [(r['series'], fl(r['x']), fl(r['y'])) for r in tp if r['panel'].startswith('(b)')]; mm = [(r['series'], fl(r['x']), fl(r['y'])) for r in tp if r['panel'].startswith('(c)')]
            expect([y for s_, x, y in sp if s_ == 'measured'] == [1.0, 1.92, 3.5] and [y for s_, x, y in sp if s_ == 'ideal'] == [1.0, 2.0, 4.0], 'L38 fig_throughput (b): speed-up against the ideal (%s)' % sp)
            expect([round(y, 4) for s_, x, y in mm if s_ == 'counting shared pages once'] == [round(1000 / 1024, 4), round(1010 / 1024, 4), round(1020 / 1024, 4)] and [round(y, 4) for s_, x, y in mm if s_ == 'sum of resident sets'] == [round(1100 / 1024, 4), round(2000 / 1024, 4), round(3900 / 1024, 4)],
                   'L39 fig_throughput (c): memory in GB, the shared world counted once against the sum of the resident sets (%s)' % mm)
            Cx = types.SimpleNamespace(bench={'throughput': old, 'kernel': None}, profile={}, warnings=[]); only_a = FG.make_all(Cx, os.path.join(tmp, 'figs_old'), only={'fig_throughput', 'fig_breakdown', 'fig_latency'}, log=lambda *a: None)
            expect(set(only_a) == {'fig_throughput'} and not Cx.warnings, 'L40 a bench of the old format draws the throughput figure and leaves out the figures it has no data for (%s)' % sorted(only_a))
        except ImportError:
            expect(True, 'L31 (matplotlib missing: figures not tested)')
        # ---- the report and the results folder
        out = os.path.join(tmp, 'bundle'); rc = REP.main([R, '--out', out]); md = open(os.path.join(out, 'report.md')).read()
        sects = [h for h in ('## Throughput', '## Where the time goes', '## Latency') if h in md]
        expect(rc == 0 and sects == ['## Throughput', '## Where the time goes', '## Latency'] and md.index('## Throughput') < md.index('## Where the time goes') < md.index('## Latency'), 'L41 the report has the sections of the benchmark in order (%s)' % sects)
        expect(all(os.path.exists(os.path.join(out, 'tables', 'bench_%s.%s' % (k, e))) for k in ('solvers', 'breakdown', 'latency', 'workers', 'world', 'machine', 'kernel') for e in ('md', 'tex', 'csv')), 'L42 every table of the benchmark is written as md, tex and csv')
        expect(json.load(open(os.path.join(out, 'provenance', 'bench.json'))) == b and json.load(open(os.path.join(out, 'provenance', 'kernel_bench.json'))) == kern, 'L43 the raw measurements are in the results folder')
        Nr = json.load(open(os.path.join(out, 'numbers.json'))); tex = open(os.path.join(out, 'numbers.tex')).read()
        expect(Nr.get('bench_probe_median_ms') == 50.5 and '\\newcommand{\\numBenchProbeMedianMs}{50.5}' in tex, 'L44 the numbers of the benchmark are in numbers.json and numbers.tex')
        rd = open(os.path.join(out, 'README.md')).read()
        expect('provenance/bench.json' in rd and 'proportional set size' in rd and 'bench_*' in rd, 'L45 the README of the folder says where the benchmark tables come from and how memory is counted')
        expect(EB.verify(out) == [] and 'None: every evaluation row is present' in md, 'L46 the folder verifies against its manifest, and a clean bench adds no warning')
        figs_png = [f for f in os.listdir(os.path.join(out, 'figures')) if f.endswith('.png')] if os.path.isdir(os.path.join(out, 'figures')) else []
        try:
            import matplotlib  # noqa: F401
            expect({'fig_throughput.png', 'fig_breakdown.png', 'fig_latency.png'} <= set(figs_png) and all('![%s](figures/%s.png)' % (f, f) in md for f in ('fig_throughput', 'fig_breakdown', 'fig_latency')), 'L47 the report shows the three figures (%s)' % figs_png)
        except ImportError: pass
        json.dump(dict(b, machine=dict(b['machine'], busy_at_start=True, loadavg_start=[3.0, 2.0, 1.0])), open(os.path.join(R, 'bench', 'bench.json'), 'w'))
        out2 = os.path.join(tmp, 'bundle2'); REP.main([R, '--out', out2, '--no-figures']); md2 = open(os.path.join(out2, 'report.md')).read().split('## Warnings')[1].split('\n## ')[0]
        expect('benchmark: the machine was busy' in md2, 'L48 the warnings of the report carry what the benchmark says about itself (%s)' % md2.strip()[:120])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ M. the studies of the network and of the start
M_SEEDS = (0, 1, 2)
M_E = {'data_iid': (-0.20, -0.25, -0.30), 'data_comp': (-0.10, -0.15, -0.20), 'par_iid': (-0.05, -0.10, -0.15), 'par_comp': (-0.02, -0.07, -0.12)}        # the exponent of each seed's own power law
M_USED = {100: 100, 300: 300, 900: 800}                                                                                                                      # (the pre-training of nominal size 900 trained on 800 theorems)
M_PARAMS = {64: 380_000, 128: 1_500_000, 256: 6_100_000}
M_ACC = {100: (0.30, 0.20), 300: (0.40, 0.25), 900: (0.55, 0.35)}                                                                                          # n -> actions right (iid, compositional) of seed 0; every seed adds 0.02 and 0.01
M_WIDTH_ACC = {64: (0.45, 0.28), 256: (0.62, 0.40)}
M_ARCH = {'bigru': dict(li=(0.62, 0.64, 0.66), lc=(0.90, 0.95, 1.00), ai=(0.50, 0.52, 0.54), ac=(0.30, 0.32, 0.34), params=1_200_000),
          'bow': dict(li=(1.10, 1.15, 1.20), lc=(1.50, 1.60, 1.70), ai=(0.20, 0.22, 0.24), ac=(0.10, 0.12, 0.14), params=None)}                            # (the runs of the bag do not say their parameters: the table takes them from the benchmark)
M_SCRATCH = {0: [0, 0, 1, 1], 1: [0, 1, 1, 2], 2: [0, 0, 0, 1]}                                                                                             # dev problems solved (of 2) after each iteration of a learner from random weights


def m_data_loss(kind, x, s): return ((2.0 if kind == 'iid' else 3.0) + 0.5 * s) * x ** M_E['data_' + kind][s]                         # the law of seed s: loss = c_s x^e_s in the number of theorems
def m_par_loss(kind, p, s): return m_data_loss(kind, M_USED[900], s) * (p / float(M_PARAMS[128])) ** M_E['par_' + kind][s]            # ... and in the number of parameters, through the loss of the default network pre-trained on everything


def tab(*rows): return {s: dict(zip(('t1', 't2', 't3', 't4', 'd1', 'd2'), r)) for s, r in enumerate(rows)}


M_ROWS = {
    ('arch', 'bc_bigru'): tab((6, 90, None, None, 3, None), (9, 60, None, None, 3, 80), (7, None, None, 70, 3, None)),
    ('arch', 'ei_her_bigru'): tab((5, 40, 99, None, 2, 60), (8, 45, 101, None, 2, None), (4, 12, None, 52, 2, 75)),
    ('arch', 'bc_bow'): tab((None,) * 6, (40, None, None, None, None, None), (None,) * 6),
    ('size', 'bc_w64'): tab((12, None, None, None, 5, None), (15, None, None, None, 5, None), (9, 70, None, None, 5, None)),
    ('size', 'bc_w256'): tab((4, 40, 80, None, 2, 90), (5, 35, None, None, 2, 70), (3, 25, 95, 60, 2, 50)),
    ('size', 'ei_her_w64'): tab((10, 100, None, None, 4, None), (12, None, None, None, 4, 95), (8, 60, None, None, 4, None)),
    ('scratch', 'scratch_ei_her'): tab((30, None, None, None, 20, None), (None,) * 6, (45, None, None, None, 25, None)),
}


def put(R, rid, final=None, log=None):
    d = os.path.join(R, rid); os.makedirs(d, exist_ok=True)
    if final is not None:
        with open(os.path.join(d, 'final.json'), 'w') as f: json.dump(final, f)
    if log is not None:
        with open(os.path.join(d, 'log.jsonl'), 'w') as f:
            for row in log: f.write(json.dumps(row) + '\n')


def pct_cell(vals, digits=1):
    """the cell of a rate over three seeds as the module documents it: mean +- half-width of the Student-t interval, in percent; an interval that leaves [0, 100] is clipped and written in parentheses"""
    m = statistics.mean(vals); h = T975_2 * statistics.stdev(vals) / math.sqrt(len(vals)); lo, hi = m - h, m + h
    if lo < 0.0 or hi > 1.0: return '%.*f (%.0f–%.0f)' % (digits, 100 * m, 100 * max(0.0, lo), 100 * min(1.0, hi))
    return '%.*f ± %.*f' % (digits, 100 * m, digits, 100 * h)


def mci_cell(vals, digits=3):
    m = statistics.mean(vals); h = T975_2 * statistics.stdev(vals) / math.sqrt(len(vals))
    return '%.*f ± %.*f' % (digits, m, digits, h)


def studies_fixture(R):
    """the runs directory of C plus the studies: pre-training at three sizes, at other widths and with other encoders (held-out synthetic metrics on exact power laws, one exponent per seed), the evaluations of the
    variants of the groups arch / size / scratch, the iterations of a learner that starts from random weights, and a benchmark of the encoders.  Returns the final.json of every pre-training run."""
    import exp_profile as PF
    write_fixture(R); FIN = {}
    with open(os.path.join(R, 'PROFILE.json')) as f: prof = json.load(f)
    prof['synth_sizes'] = [0, 100, 300, 900]
    with open(os.path.join(R, 'PROFILE.json'), 'w') as f: json.dump(prof, f)
    def pre(name, s, li, lc, ai, ac, params, used):
        fin = {'best': {'synth_test_iid': {'loss': li, 'action_acc': ai}, 'synth_test_comp': {'loss': lc, 'action_acc': ac}}, 'synth_theorems': used}
        if params is not None: fin['n_params'] = params
        FIN.setdefault(name, {})[s] = fin; put(R, 'pre/%s/s%d' % (name, s), fin)
    for s in M_SEEDS:
        for n in (100, 300, 900):
            x = M_USED[n]; ai, ac = M_ACC[n]
            pre('n%d' % n, s, m_data_loss('iid', x, s), m_data_loss('comp', x, s), ai + 0.02 * s, ac + 0.01 * s, M_PARAMS[128], 850 if (n == 900 and s == 2) else x)
        for w in (64, 256):
            ai, ac = M_WIDTH_ACC[w]
            pre('w%d_n900' % w, s, m_par_loss('iid', M_PARAMS[w], s), m_par_loss('comp', M_PARAMS[w], s), ai + 0.01 * s, ac + 0.01 * s, M_PARAMS[w], 800)
        for a, v in M_ARCH.items():
            pre('%s_n900' % a, s, v['li'][s], v['lc'][s], v['ai'][s], v['ac'][s], v['params'], 800)
        put(R, 'rl/scratch_ei_her/s%d' % s, {'history': [{'iter': i} for i in range(4)], 'seconds': 7200, 'n_params': M_PARAMS[128]},
            [{'iter': i, 'dev_solved': M_SCRATCH[s][i], 'dev_total': 2, 'cum_probes': 1000 * (i + 1)} for i in range(4)])
        put(R, 'rl/ei_her/s%d' % s, {'history': [{'iter': i} for i in range(3)], 'seconds': 3600, 'n_params': M_PARAMS[128]})
        put(R, 'rl/ei_her_bigru/s%d' % s, {'history': [{'iter': i} for i in range(9)], 'seconds': 99999, 'n_params': 1_200_000})              # (another variant: its hours and iterations are not those of ei_her)
    for (group, variant), table in M_ROWS.items():
        d = os.path.join(R, 'eval', group, variant); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'rows.jsonl'), 'w') as f:
            for seed, rows in table.items():
                for name, probes in rows.items(): f.write(json.dumps(make_row(variant, 'sample', seed, name, probes)) + '\n')
        with open(os.path.join(d, 'meta.json'), 'w') as f: json.dump({'argv': [], 'machine': {}}, f)
    ms = [i / 1000.0 for i in range(1, 101)]
    def entry(arch, width, params, sps, lat):
        return {'arch': arch, 'width': width, 'params': params, 'train_steps_per_s': sps, 'batch': 64, 'device': 'cpu', 'encode': PF.summarize(lat), 'steps_available': 500}
    bench = {'archs': collections.OrderedDict([('transformer@128', entry('transformer', 128, M_PARAMS[128], 100.0, ms)), ('bigru@128', entry('bigru', 128, 1_200_000, 150.0, ms[:50])), ('cnn@128', entry('cnn', 128, 1_100_000, 250.0, ms[:20])),
                                               ('bow@128', entry('bow', 128, 900_000, 400.0, ms[:10])), ('transformer@64', entry('transformer', 64, M_PARAMS[64], 300.0, ms[:30])), ('transformer@256', entry('transformer', 256, M_PARAMS[256], 40.0, ms))])}
    os.makedirs(os.path.join(R, 'bench'), exist_ok=True)
    with open(os.path.join(R, 'bench', 'bench.json'), 'w') as f: json.dump(bench, f)
    return FIN


def sec_M(c):
    import types, csv as _csv, exp_bundle as EB
    tmp = tempfile.mkdtemp(prefix='s51m_')
    try:
        R = os.path.join(tmp, 'runs'); FIN = studies_fixture(R)
        C = EC.load(R); P = C.profile
        test = ['t1', 't2', 't3', 't4']; dev = ['d1', 'd2']; proved = ['t1', 't2', 't3']
        def cell(group, variant, names): return pct_cell([rate(M_ROWS[(group, variant)][s], names, 100) for s in M_SEEDS])
        def main_cell(variant, split, subset='all'): return TB._cell(C, 'main', variant, 'sample', split, 100, subset)[0]
        def fin(name, *path): return [functools_reduce(FIN[name][s], path) for s in M_SEEDS]
        def functools_reduce(d, path):
            for k in path: d = d[k]
            return d
        # ---- the collector reads the studies
        expect(C.variants('arch') == ['bc_bigru', 'bc_bow', 'ei_her_bigru'] and C.variants('size') == ['bc_w256', 'bc_w64', 'ei_her_w64'] and C.variants('scratch') == ['scratch_ei_her'], 'M1 the evaluation groups arch, size and scratch are read (%s, %s, %s)' % (C.variants('arch'), C.variants('size'), C.variants('scratch')))
        expect(C.final_values('pre', 'n900', ('best', 'synth_test_iid', 'loss')) == {s: FIN['n900'][s]['best']['synth_test_iid']['loss'] for s in M_SEEDS} and C.final_values('pre', 'w64_n900', ('n_params',)) == {s: 380000.0 for s in M_SEEDS}
               and C.final_values('pre', 'bow_n900', ('n_params',)) == {} and C.final_values('pre', 'n900', ('best', 'nope', 'loss')) == {}, 'M2 the held-out metric and the parameters of the pre-training runs of a name, by seed (a run without the value is left out)')
        expect(C.synth_used(900) == 800 and C.synth_used(100) == 100 and C.synth_used(300) == 300 and any('nominal size 900 trained on 800' in w for w in C.warnings) and not any('nominal size 100' in w for w in C.warnings),
               'M3 the number of theorems a pre-training really used is the smallest over its seeds (900 asked, 800 and 850 used), and the report warns about it (%s)' % C.warnings)
        # ---- the tables
        T = TB.all_tables(C); T2 = TB.throughput_tables(C)
        expect({'arch', 'size', 'laws', 'scratch'} <= set(T) and 'scaling' not in T and set(T2) == {'bench_archs'}, 'M4 the study tables exist and the scaling table (no run of an intermediate size) does not: %s %s' % (sorted(T), sorted(T2)))
        t = T['arch']
        expect(t.header == ['Encoder', 'parameters (M)', 'held-out synthetic: loss', 'actions right, iid (%)', 'actions right, compositional (%)', 'BC: test (%)', 'EI + HER: test (%)', 'training steps / s'], 'M5 the columns of the table of encoders (%s)' % t.header)
        expect([r[0] for r in t.rows] == ['Transformer', 'Bidirectional GRU', 'Bag of tokens'], 'M6 the rows are the encoders that have results, in the order of the paper (the dilated CNN has none here): %s' % [r[0] for r in t.rows])
        b = M_ARCH['bigru']; w_ = M_ARCH['bow']
        want = [['Transformer', '1.50', mci_cell(fin('n900', 'best', 'synth_test_iid', 'loss')), pct_cell(fin('n900', 'best', 'synth_test_iid', 'action_acc')), pct_cell(fin('n900', 'best', 'synth_test_comp', 'action_acc')),
                 main_cell('bc', 'test'), main_cell('ei_her', 'test'), '100'],
                ['Bidirectional GRU', '1.20', mci_cell(b['li']), pct_cell(b['ai']), pct_cell(b['ac']), cell('arch', 'bc_bigru', test), cell('arch', 'ei_her_bigru', test), '150'],
                ['Bag of tokens', '0.90', mci_cell(w_['li']), pct_cell(w_['ai']), pct_cell(w_['ac']), cell('arch', 'bc_bow', test), '--', '400']]
        for got, exp in zip(t.rows, want): expect(got == exp, 'M7 the row of %s: parameters (the bag from the benchmark, the others from the runs), held-out loss and accuracies of the pre-trained network, the learners on olympiad problems, training speed (got %s, wanted %s)' % (exp[0], got, exp))
        expect(any('One network per row' in n for n in t.notes), 'M8 the note says what is held fixed')
        t = T['size']
        expect(t.header == ['width', 'layers', 'parameters (M)', 'held-out synthetic: loss', 'actions right, iid (%)', 'BC: dev (%)', 'BC: test (%)', 'EI + HER: test (%)', 'training steps / s'], 'M9 the columns of the table of widths (%s)' % t.header)
        want = [[64, 2, '0.38', mci_cell(fin('w64_n900', 'best', 'synth_test_iid', 'loss')), pct_cell(fin('w64_n900', 'best', 'synth_test_iid', 'action_acc')), cell('size', 'bc_w64', dev), cell('size', 'bc_w64', test), cell('size', 'ei_her_w64', test), '300'],
                [128, 3, '1.50', mci_cell(fin('n900', 'best', 'synth_test_iid', 'loss')), pct_cell(fin('n900', 'best', 'synth_test_iid', 'action_acc')), main_cell('bc', 'dev'), main_cell('bc', 'test'), main_cell('ei_her', 'test'), '100'],
                [256, 4, '6.10', mci_cell(fin('w256_n900', 'best', 'synth_test_iid', 'loss')), pct_cell(fin('w256_n900', 'best', 'synth_test_iid', 'action_acc')), cell('size', 'bc_w256', dev), cell('size', 'bc_w256', test), '--', '40']]
        for got, exp in zip(t.rows, want): expect(got == exp, 'M10 the row of width %s: layers grow with the width, the default width takes its olympiad results from the main group, a learner that was not run at a width is "--" (got %s, wanted %s)' % (exp[0], got, exp))
        expect(len(T['size'].rows) == 3, 'M10b one row per width')
        t = T['laws']
        want = [['held-out synthetic loss, iid', 'synthetic training theorems', '100–800', 3, '-0.250 [-0.374, -0.126]', '1.00', 3],
                ['held-out synthetic loss, compositional', 'synthetic training theorems', '100–800', 3, '-0.150 [-0.274, -0.026]', '1.00', 3],
                ['held-out synthetic loss, iid', 'parameters', '0.38M–6.1M', 3, '-0.100 [-0.224, +0.024]', '1.00', 3],
                ['held-out synthetic loss, compositional', 'parameters', '0.38M–6.1M', 3, '-0.070 [-0.194, +0.054]', '1.00', 3]]
        expect(t.rows == want, 'M11 the scaling laws: the exponent of each seed (-0.20, -0.25, -0.30 for the data) averaged, with the t-interval over seeds, over the range that was measured (the number of theorems really used, 800 and not 900); R^2 1.00 for exact laws (got %s)' % t.rows)
        expect(any('no irreducible loss' in n for n in t.notes), 'M12 the note says that the exponent is empirical (no asymptote assumed)')
        t = T['scratch']
        bl = lambda s_: [TB._cell(C, 'baselines', 'baselines', s_, 'dev', 100, 'all')[0], TB._cell(C, 'baselines', 'baselines', s_, 'test', 100, 'all')[0], TB._cell(C, 'baselines', 'baselines', s_, 'test', 100, 'proved')[0]]
        mn = lambda v_: [main_cell(v_, 'dev'), main_cell(v_, 'test'), main_cell(v_, 'test', 'proved')]
        want = [['Uniform random actions', 'no network', 1, '--'] + bl('uniform') + ['--'], ['Blind search (IDA*, engine costs)', 'no network', 1, '--'] + bl('blind') + ['--'],
                ['EI + HER from random weights', 'random weights', 3, '4', cell('scratch', 'scratch_ei_her', dev), cell('scratch', 'scratch_ei_her', test), cell('scratch', 'scratch_ei_her', proved), '2.0'],
                ['BC: pre-trained + olympiad', 'pre-trained + BC', 3, '--'] + mn('bc') + ['--'], ['Expert iteration + hindsight', 'BC', 3, '3'] + mn('ei_her') + ['1.0']]
        expect(t.header == ['Method', 'starts from', 'seeds', 'iterations', 'dev', 'test', 'test, proved', 'training hours'] and t.rows == want, 'M13 the table of learning from random weights: the baselines, the learner from random weights with its iterations and hours, and the learners continued from behaviour cloning (the hours of ei_her are not those of ei_her_bigru) (got %s)' % t.rows)
        t = T2['bench_archs']
        want = [['Transformer', 128, '1.50', '100', '50.50', '99.0'], ['Bidirectional GRU', 128, '1.20', '150', '25.50', '49.5'], ['Dilated CNN', 128, '1.10', '250', '10.50', '19.8'], ['Bag of tokens', 128, '0.90', '400', '5.50', '9.9'],
                ['Transformer, width 64', 64, '0.38', '300', '15.50', '29.7'], ['Transformer, width 256', 256, '6.10', '40', '50.50', '99.0']]
        expect(t.header == ['network', 'width', 'parameters (M)', 'training steps / s', 'encode one goal: median (ms)', 'p99 (ms)'] and t.rows == want and any('on the cpu' in n for n in t.notes), 'M14 the cost of the encoders and widths, from the benchmark (got %s)' % t.rows)
        old = TB.throughput_tables(types.SimpleNamespace(bench={'throughput': {k: v for k, v in C.bench['throughput'].items() if k != 'archs'}, 'kernel': None}))
        expect('bench_archs' not in old, 'M15 a benchmark without the encoders gives no such table')
        # a study with a single point has no table of laws, and a single encoder no table of encoders
        P1 = dict(P, synth_sizes=[0, 100]); expect(TB.laws_table(C, P1) is None and TB.arch_table(C, dict(P, synth_sizes=[0])) is None and TB.size_table(C, dict(P, synth_sizes=[0])) is None and TB.laws_table(C, dict(P, synth_sizes=[0])) is None,
                                                  'M16 too few points for a trend, or no pre-training size at all: no table of laws, of encoders or of widths')
        # ---- the figures
        try:
            import matplotlib  # noqa: F401
            FG = __import__('exp_figures'); fdir = os.path.join(tmp, 'figs')
            figs = FG.make_all(C, fdir, only={'fig_arch', 'fig_laws', 'fig_scratch'}, log=lambda *a: None)
            expect(set(figs) == {'fig_arch', 'fig_laws', 'fig_scratch'} and all(os.path.getsize(p) > 1500 for ps in figs.values() for p in ps) and not [w for w in C.warnings if 'failed' in w], 'M17 the three figures are drawn as pdf and png (%s, %s)' % (sorted(figs), C.warnings))
            def fcsv(name): return list(_csv.DictReader(open(os.path.join(fdir, name + '.csv'))))
            def fl(x): return float(x) if x not in ('', None) else None
            fa = [r for r in fcsv('fig_arch') if r['kind'] == 'bar']
            a_ = {r['series']: fl(r['y']) for r in fa if r['panel'] == '(a) imitation'}
            ok = close(a_['Transformer, iid'], statistics.mean(fin('n900', 'best', 'synth_test_iid', 'action_acc')), 1e-5) and close(a_['Transformer, compositional'], statistics.mean(fin('n900', 'best', 'synth_test_comp', 'action_acc')), 1e-5) \
                and close(a_['Bidirectional GRU, iid'], 0.52, 1e-5) and close(a_['Bag of tokens, compositional'], 0.12, 1e-5) and len(a_) == 6
            expect(ok, 'M18 fig_arch (a): the actions right on held-out synthetic theorems of each encoder, iid and compositional (%s)' % a_)
            b_ = {(r['label'], r['series']): fl(r['y']) for r in fa if r['panel'] == '(b) olympiad problems'}
            rb = lambda g_, v_: statistics.mean(rate(M_ROWS[(g_, v_)][s], test, 100) for s in M_SEEDS)
            ok = len(b_) == 5 and close(b_[('BC', 'Transformer')], C.rate_ci('main', 'bc', 'sample', 'test', 100, 'all')[0], 1e-5) and close(b_[('BC', 'Bidirectional GRU')], rb('arch', 'bc_bigru'), 1e-5) and close(b_[('BC', 'Bag of tokens')], rb('arch', 'bc_bow'), 1e-5) \
                and close(b_[('EI + HER', 'Bidirectional GRU')], rb('arch', 'ei_her_bigru'), 1e-5) and ('EI + HER', 'Bag of tokens') not in b_
            expect(ok, 'M19 fig_arch (b): the olympiad test problems solved by each learner with each encoder, and no bar for a pair that was not run (%s)' % b_)
            c_ = {r['series']: fl(r['y']) for r in fa if r['panel'] == '(c) training speed'}
            expect(c_ == {'Transformer': 100.0, 'Bidirectional GRU': 150.0, 'Bag of tokens': 400.0}, 'M20 fig_arch (c): training steps per second from the benchmark (%s)' % c_)
            fw = [r for r in fcsv('fig_laws') if r['kind'] == 'line']
            def line(panel, series): return [(fl(r['x']), fl(r['y'])) for r in fw if r['panel'] == panel and r['series'] == series]
            def slope(pts): return (math.log(pts[-1][1]) - math.log(pts[0][1])) / (math.log(pts[-1][0]) - math.log(pts[0][0]))
            li = [statistics.mean(m_data_loss('iid', x, s) for s in M_SEEDS) for x in (100, 300, 800)]
            ok = [x for x, _ in line('(a) data', 'iid split')] == [100.0, 300.0, 800.0] and all(close(y, v, 1e-5) for (_, y), v in zip(line('(a) data', 'iid split'), li))
            expect(ok, 'M21 fig_laws (a): the mean held-out loss against the number of theorems really used (%s)' % line('(a) data', 'iid split'))
            fits = {(p_, s_): line(p_, s_) for p_ in ('(a) data', '(b) model size') for s_ in ('fit, iid split', 'fit, compositional split')}
            ok = all(len(v) == 20 for v in fits.values()) and abs(slope(fits[('(a) data', 'fit, iid split')]) + 0.25) < 1e-4 and abs(slope(fits[('(a) data', 'fit, compositional split')]) + 0.15) < 1e-4 \
                and abs(slope(fits[('(b) model size', 'fit, iid split')]) + 0.10) < 1e-4 and abs(slope(fits[('(b) model size', 'fit, compositional split')]) + 0.07) < 1e-4
            expect(ok, 'M22 fig_laws: the dashed line of each panel is the average of the lines fitted to the seeds, its slope the exponent of the table (%s)' % {k: round(slope(v), 4) for k, v in fits.items()})
            ok = [x for x, _ in line('(b) model size', 'iid split')] == [380000.0, 1500000.0, 6100000.0]
            expect(ok, 'M23 fig_laws (b): the x of a point is the parameter count of its network (%s)' % line('(b) model size', 'iid split'))
            dv = line('(c) olympiad problems, BC', 'dev'); ts = line('(c) olympiad problems, BC', 'test')
            wd = [('size', 'bc_w64'), ('main', 'bc'), ('size', 'bc_w256')]
            ok = len(dv) == 3 and len(ts) == 3 and close(ts[0][1], statistics.mean(rate(M_ROWS[('size', 'bc_w64')][s], test, 100) for s in M_SEEDS), 1e-5) and close(ts[1][1], C.rate_ci('main', 'bc', 'sample', 'test', 100, 'all')[0], 1e-5) \
                and close(dv[2][1], statistics.mean(rate(M_ROWS[('size', 'bc_w256')][s], dev, 100) for s in M_SEEDS), 1e-5)
            expect(ok, 'M24 fig_laws (c): olympiad problems solved by BC at each width (%s %s)' % (dv, ts))
            fs = [r for r in fcsv('fig_scratch') if r['kind'] == 'line' and r['series'] == 'EI + HER from random weights']
            a2 = [(fl(r['x']), fl(r['y'])) for r in fs if r['panel'] == '(a) by iteration']; b2 = [(fl(r['x']), fl(r['y'])) for r in fs if r['panel'] == '(b) by compute']
            want_y = [statistics.mean(M_SCRATCH[s][i] / 2.0 for s in M_SEEDS) for i in range(4)]
            expect([x for x, _ in a2] == [0.0, 1.0, 2.0, 3.0] and all(close(y, v, 1e-5) for (_, y), v in zip(a2, want_y)) and [x for x, _ in b2] == [1000.0, 2000.0, 3000.0, 4000.0] and all(close(y, v, 1e-5) for (_, y), v in zip(b2, want_y)),
                   'M25 fig_scratch: the dev problems solved after each iteration, against the iteration and against the probes spent (%s, %s)' % (a2, b2))
            expect(not [r for r in fcsv('fig_scratch') if r['series'] == '' and r['kind'] == 'line'], 'M26 the guide line of the uniform baseline is not data of the figure')
            Cn = types.SimpleNamespace(profile={}, warnings=[], variants=lambda g: [], bench={}, synth_used=lambda n: None)
            expect(FG.fig_arch(Cn, {}, tmp, FG._mpl()) is None and FG.fig_laws(Cn, {}, tmp, FG._mpl()) is None and FG.fig_scratch(Cn, {}, tmp, FG._mpl()) is None, 'M27 without the studies the three figures are not drawn')
        except ImportError:
            expect(True, 'M17 (matplotlib missing: figures not tested)')
        # ---- the report and the folder
        out = os.path.join(tmp, 'bundle'); rc = REP.main([R, '--out', out]); md = open(os.path.join(out, 'report.md')).read()
        heads = [h for h in ('## Other encoders of the goal', '## Other widths', '## Scaling laws', '## Reinforcement learning from random weights', '## Cost of the encoders and widths') if h in md]
        expect(rc == 0 and heads == ['## Other encoders of the goal', '## Other widths', '## Scaling laws', '## Reinforcement learning from random weights', '## Cost of the encoders and widths'] and [md.index(h) for h in heads] == sorted(md.index(h) for h in heads),
               'M28 the report has the sections of the studies, in order (%s)' % heads)
        expect(all(os.path.exists(os.path.join(out, 'tables', '%s.%s' % (k, e))) for k in ('arch', 'size', 'laws', 'scratch', 'bench_archs') for e in ('md', 'tex', 'csv')), 'M29 every table of the studies is written as md, tex and csv')
        try:
            import matplotlib  # noqa: F401
            expect(all('![%s](figures/%s.png)' % (f, f) in md and os.path.exists(os.path.join(out, 'figures', f + '.csv')) for f in ('fig_arch', 'fig_laws', 'fig_scratch')), 'M30 the report shows the figures of the studies and the folder has their data')
        except ImportError: pass
        N = json.load(open(os.path.join(out, 'numbers.json'))); tex = open(os.path.join(out, 'numbers.tex')).read()
        rates_bigru = [rate(M_ROWS[('arch', 'bc_bigru')][s], test, 100) for s in M_SEEDS]
        wantN = {'law_data_iid': '-0.250', 'law_data_comp': '-0.150', 'law_params_iid': '-0.100', 'law_params_comp': '-0.070', 'model_params_m': 1.5, 'bench_train_steps_bigru_128': 150, 'bench_train_steps_transformer_64': 300, 'bench_train_steps_transformer_256': 40,
                 'bench_params_m_bow_128': 0.9, 'test_bc_bigru': pct_cell(rates_bigru), 'test_bc_bigru_mean': round(100 * statistics.mean(rates_bigru), 2), 'dev_bc_w64': cell('size', 'bc_w64', dev), 'test_ei_her_w64': cell('size', 'ei_her_w64', test),
                 'test_scratch_ei_her': cell('scratch', 'scratch_ei_her', test)}
        bad = {k: (N.get(k), v) for k, v in wantN.items() if N.get(k) != v}
        expect(not bad, 'M31 the numbers of the studies in numbers.json (and the parameters of THE network, not of the first network of a study: 1.5 and not the 1.2 of the GRU): %s' % bad)
        expect('\\newcommand{\\numLawDataIid}{-0.250}' in tex and '\\numTestBcBigru}' in tex and '\\numBenchTrainStepsBigruOneTwoEight}{150}' in tex, 'M32 the macros of these numbers are letters only and are in numbers.tex')
        rd = open(os.path.join(out, 'README.md')).read()
        expect('fig_arch' in rd and 'empirical exponent' in rd and 'no asymptote' in rd, 'M33 the README of the folder says what the studies are and that an exponent is empirical')
        expect(EB.verify(out) == [], 'M34 the folder verifies against its manifest (%s)' % EB.verify(out))
        md_warn = md.split('## Warnings')[1].split('\n## ')[0]
        expect('nominal size 900 trained on 800' in md_warn, 'M35 the report carries the warning about the size actually used (%s)' % md_warn.strip()[:200])
        # a runs directory with only a part of the studies still gives a report
        shutil.rmtree(os.path.join(R, 'eval', 'size')); shutil.rmtree(os.path.join(R, 'pre', 'w64_n900')); shutil.rmtree(os.path.join(R, 'pre', 'w256_n900'))
        out2 = os.path.join(tmp, 'bundle2'); rc2 = REP.main([R, '--out', out2, '--no-figures']); T3 = json.load(open(os.path.join(out2, 'numbers.json')))
        have = set(os.listdir(os.path.join(out2, 'tables')))
        expect(rc2 == 0 and 'size.md' not in have and 'arch.md' in have and 'laws.md' in have and T3.get('law_data_iid') == '-0.250' and 'law_params_iid' not in T3, 'M36 without the widths there is no table of widths and no exponent in the parameters, and the rest of the report is the same (%s)' % sorted(have))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ N. the encoders as commands
def sec_N(c):
    import torch, ml_model as M, exp_bench as XB
    fx = c['mini']; L = os.path.join(itp_paths.ROOT, 'src', 'python', 'learner')
    py = lambda name: os.path.join(L, name)
    small = ['--d', '32', '--layers', '1', '--heads', '2', '--ff', '64']
    def load(path): return fx.T.load(path, fx.ctx.bags)
    def params(net): return {k: v.detach().clone() for k, v in net.named_parameters()}
    def same(a_, b_): return a_.keys() == b_.keys() and all(torch.equal(a_[k], b_[k]) for k in a_)
    classes = {'transformer': 'TransformerEncoder', 'bigru': 'BiGRUEncoder', 'cnn': 'ConvEncoder', 'bow': 'BagEncoder'}
    # (1) ml_init: an untrained network of every encoder, one command
    nets = {}; counts = {}
    for arch in M.ARCHS:
        out = os.path.join(fx.tmp, 'init_' + arch); t0 = time.time()
        rc, log = run_py([py('ml_init.py'), fx.ddir, '--out', out, '--arch', arch, '--seed', '0'] + small)
        trace('ml_init %s %.1f s' % (arch, time.time() - t0))
        if not expect(rc == 0 and os.path.exists(os.path.join(out, 'best.pt')) and os.path.exists(os.path.join(out, 'final.json')), 'N1 ml_init makes an untrained %s network (exit %d)\n%s' % (arch, rc, log[-500:] if rc else '')): continue
        net, cfg, extra = load(os.path.join(out, 'best.pt')); fin = json.load(open(os.path.join(out, 'final.json'))); nets[arch] = net; counts[arch] = sum(p.numel() for p in net.parameters())
        expect(cfg.get('arch') == arch and net.arch == arch and extra.get('untrained') is True and extra.get('epoch') == -1 and fin['untrained'] is True and fin['n_params'] == counts[arch] and fin['config'] == cfg, 'N2 the checkpoint of an untrained %s network says so (arch, untrained, parameters)' % arch)
        expect(type(net.enc).__name__ == classes[arch] and hasattr(net, 'pos') == (arch != 'bow') and net.pooled == ('first' if arch == 'transformer' else 'mean' if arch in ('cnn', 'bow') else net.pooled), 'N3 the %s network has the encoder %s%s' % (arch, classes[arch], ' and no table of positions' if arch == 'bow' else ''))
    expect(len(counts) == 4 and len(set(counts.values())) == 4 and counts.get('bow', 0) < min(v for k, v in counts.items() if k != 'bow'), 'N4 the four encoders are four different networks and the bag, which has no table of positions, is the smallest (%s)' % counts)
    # same seed, same weights; another seed, other weights; and ml_init makes the network that make_net makes under the same seed
    if 'bigru' in nets:
        rc1, _ = run_py([py('ml_init.py'), fx.ddir, '--out', os.path.join(fx.tmp, 'init_bigru_b'), '--arch', 'bigru', '--seed', '0'] + small)
        rc2, _ = run_py([py('ml_init.py'), fx.ddir, '--out', os.path.join(fx.tmp, 'init_bigru_c'), '--arch', 'bigru', '--seed', '1'] + small)
        if expect(rc1 == 0 and rc2 == 0, 'N5 ml_init twice more'):
            nb, _, _ = load(os.path.join(fx.tmp, 'init_bigru_b', 'best.pt')); nc, _, _ = load(os.path.join(fx.tmp, 'init_bigru_c', 'best.pt'))
            expect(same(params(nb), params(nets['bigru'])) and not same(params(nc), params(nets['bigru'])), 'N5 the same seed gives the same weights, another seed other weights')
    if 'transformer' in nets:
        torch.manual_seed(0); ref = fx.T.make_net(fx.ctx.bags, {'d': 32, 'layers': 1, 'heads': 2, 'ff': 64, 'dropout': 0.1, 'seed': 0, 'arch': 'transformer'})
        expect(same(params(ref), params(nets['transformer'])), 'N6 the untrained transformer of ml_init is the network make_net builds under the same seed')
    # (2) behaviour cloning: --arch chooses the encoder of a new network; --init keeps the one of the checkpoint; naming another one is an error
    common = ['--epochs', '1', '--batch', '16', '--seed', '0', '--no-lib'] + small
    out = os.path.join(fx.tmp, 'bc_bigru'); t0 = time.time()
    rc, log = run_py([py('ml_bc.py'), fx.ddir, '--out', out, '--arch', 'bigru'] + common)
    trace('ml_bc --arch bigru %.1f s' % (time.time() - t0))
    if expect(rc == 0, 'N7 behaviour cloning of a GRU network from scratch runs (exit %d)\n%s' % (rc, log[-500:] if rc else '')):
        net, cfg, _ = load(os.path.join(out, 'best.pt')); fin = json.load(open(os.path.join(out, 'final.json')))
        expect(cfg['arch'] == 'bigru' and net.arch == 'bigru' and fin['config']['arch'] == 'bigru' and fin['n_params'] == counts.get('bigru'), 'N8 its checkpoint is a bigru network of the size of ml_init\'s (%s parameters)' % fin['n_params'])
    out = os.path.join(fx.tmp, 'bc_cnn')
    rc, log = run_py([py('ml_bc.py'), fx.ddir, '--out', out, '--init', os.path.join(fx.tmp, 'init_cnn', 'best.pt')] + common[:6])
    if expect(rc == 0, 'N9 behaviour cloning that continues a CNN checkpoint, without naming the encoder, runs (exit %d)\n%s' % (rc, log[-500:] if rc else '')):
        net, cfg, _ = load(os.path.join(out, 'best.pt'))
        expect(cfg['arch'] == 'cnn' and net.arch == 'cnn', 'N10 and stays a CNN network')
    out = os.path.join(fx.tmp, 'bc_conflict')
    rc, log = run_py([py('ml_bc.py'), fx.ddir, '--out', out, '--init', os.path.join(fx.tmp, 'init_cnn', 'best.pt'), '--arch', 'bigru'] + common[:6])
    expect(rc != 0 and 'is a cnn network' in log and not os.path.exists(os.path.join(out, 'best.pt')), 'N11 naming another encoder than the checkpoint\'s is an error that says which one it is, and nothing is trained (exit %d)' % rc)
    # (3) the reinforcement learners continue the encoder of their checkpoint; the evaluation loads whatever the checkpoint is
    out = os.path.join(fx.tmp, 'ei_bow'); t0 = time.time()
    rc, log = run_py([py('ml_ei.py'), fx.ddir, '--out', out, '--init', os.path.join(fx.tmp, 'init_bow', 'best.pt'), '--seed', '0', '--workers', '0', '--device', 'cpu', '--threads', '1', '--iters', '1', '--episodes', '4', '--probes', '20',
                                       '--updates', '1', '--eval-every', '1', '--eval-probes', '10', '--actor', 'sample', '--her', '--time-limit', '5', '--eval-time-limit', '5'])
    trace('ml_ei from bow %.1f s' % (time.time() - t0))
    if expect(rc == 0, 'N12 expert iteration from a bag-of-tokens network runs (exit %d)\n%s' % (rc, log[-500:] if rc else '')):
        net, cfg, _ = load(os.path.join(out, 'last.pt')); fin = json.load(open(os.path.join(out, 'final.json')))
        expect(cfg['arch'] == 'bow' and net.arch == 'bow' and fin['n_params'] == counts.get('bow') and not hasattr(net, 'pos'), 'N13 and trains a bag-of-tokens network (%d parameters)' % fin['n_params'])
    out = os.path.join(fx.tmp, 'ppo_bigru'); t0 = time.time()
    rc, log = run_py([py('ml_ppo.py'), fx.ddir, '--init', os.path.join(fx.tmp, 'init_bigru', 'best.pt'), '--out', out, '--iters', '1', '--episodes', '4', '--epochs', '1', '--workers', '0', '--device', 'cpu', '--threads', '1', '--seed', '0', '--eval-every', '1', '--eval-probes', '10'])
    trace('ml_ppo from bigru %.1f s' % (time.time() - t0))
    if expect(rc == 0, 'N14 PPO from a GRU network runs (exit %d)\n%s' % (rc, log[-500:] if rc else '')):
        net, cfg, _ = load(os.path.join(out, 'last.pt'))
        expect(cfg['arch'] == 'bigru' and net.arch == 'bigru', 'N15 and trains a GRU network')
    out = os.path.join(fx.tmp, 'ev_cnn')
    rc, log = run_py([py('ml_evaluate.py'), fx.ddir, '--out', out, '--split', 'dev', '--solver', 'sample', '--model', 'm=' + os.path.join(fx.tmp, 'bc_cnn', 'best.pt'), '--probes', '20', '--time-limit', '5'])
    if expect(rc == 0, 'N16 the evaluation of a CNN checkpoint runs (exit %d)\n%s' % (rc, log[-500:] if rc else '')):
        rows = [json.loads(l) for l in open(os.path.join(out, 'rows.jsonl')) if l.strip()]
        expect(len(rows) >= 3 and all(r['model'] == 'm' and r['spec'] == 'sample' for r in rows), 'N17 and gives rows like any other model')
    # (4) the benchmark of the encoders costs the networks the experiments train
    outdir = os.path.join(fx.tmp, 'bench_archs'); t0 = time.time()
    XB.main([fx.ddir, '--out', outdir, '--problems', '2', '--probes', '10', '--repeats', '1', '--workers', '1', '--check-proofs', '0', '--no-gym', '--archs', 'bigru,bow', '--widths', '64'])
    trace('exp_bench archs %.1f s' % (time.time() - t0))
    b = json.load(open(os.path.join(outdir, 'bench.json'))); ar = b.get('archs') or {}
    expect(list(ar) == ['transformer@128', 'bigru@128', 'bow@128', 'transformer@64'], 'N18 bench.json costs the default network, the encoders asked for and the widths asked for (%s)' % list(ar))
    want = {}
    for key, (arch, width) in (('transformer@128', ('transformer', 128)), ('bigru@128', ('bigru', 128)), ('bow@128', ('bow', 128)), ('transformer@64', ('transformer', 64))):
        want[key] = sum(p.numel() for p in fx.T.make_net(fx.ctx.bags, dict(SP.size_of(width), dropout=0.1, arch=arch)).parameters())
    expect({k: v['params'] for k, v in ar.items()} == want and all(v['train_steps_per_s'] > 0 and v['encode']['n'] > 0 and v['encode']['p50_ms'] > 0 for v in ar.values()), 'N19 the networks it costs are those of the experiments (parameters %s) and each has a training speed and an encoding latency' % {k: v['params'] for k, v in ar.items()})
    Tb = TB.throughput_tables(types.SimpleNamespace(bench={'throughput': b, 'kernel': None}))
    expect('bench_archs' in Tb and [r[0] for r in Tb['bench_archs'].rows] == ['Transformer', 'Bidirectional GRU', 'Bag of tokens', 'Transformer, width 64'], 'N20 the report makes its table from it (%s)' % ([r[0] for r in Tb['bench_archs'].rows] if 'bench_archs' in Tb else None))
    t0 = time.time()
    raises('N21 an encoder that does not exist is refused', lambda: XB.main([fx.ddir, '--out', os.path.join(fx.tmp, 'bench_bad'), '--archs', 'bigru,lstm']), SystemExit)
    raises('N22 so is a width the heads cannot divide, and a width that is not a number', lambda: XB.main([fx.ddir, '--out', os.path.join(fx.tmp, 'bench_bad'), '--widths', '100']), SystemExit)
    raises('N22b (a width that is not a number)', lambda: XB.main([fx.ddir, '--out', os.path.join(fx.tmp, 'bench_bad'), '--widths', '64,x']), SystemExit)
    expect(time.time() - t0 < 5.0 and not os.path.exists(os.path.join(fx.tmp, 'bench_bad')), 'N23 ... before the world is loaded or anything is measured, and before the output directory is made (%.1f s)' % (time.time() - t0))
    expect(len(XB.pick_problems(fx.bk, fx.ddir, 1)) == 1 and XB.pick_problems(fx.bk, fx.ddir, 0) == [] and len(XB.pick_problems(fx.bk, fx.ddir, 3)) == 3 and XB.pick_problems(fx.bk, fx.ddir, 1)[0] in fx.bk.placed, 'N24 one problem (the one in the middle), none, and three problems can be picked')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ O. the soundness stage
def o_fixture():
    """a catalogue of six mutants, the records of a complete clean run, and the findings of a tiny tracker, all with known answers"""
    names = ['K1 a', 'K2 b', 'K3 c', 'T1 d', 'T2 e', 'E1 f']
    cat = {'names': names, 'file': {n: 'f%d.py' % i for i, n in enumerate(names)}, 'digest': {n: 'dg_' + n[:2] for n in names}, 'gates': ['g1.py', 'g2.py'], 'equivalent': {'K3 c': 'no input tells it apart'},
           'component': {'K1 a': 'kernel (C++)', 'K2 b': 'kernel (C++)', 'K3 c': 'kernel (C++)', 'T1 d': 'tactic layer and decision procedures', 'T2 e': 'tactic layer and decision procedures', 'E1 f': "learner's environment"}}
    def rec(n, kinds=(), tree='TREE', gates=('g1.py', 'g2.py'), status=None, **kw):
        r = {'name': n, 'digest': cat['digest'][n], 'tree': tree, 'gates': list(gates), 'full': False, 'seconds': 2.0, 'caught_by': [{'gate': 'g%d.py' % (i + 1), 'kind': k, 'first': 'x'} for i, k in enumerate(kinds)]}
        r['status'] = status or ('caught' if kinds else ('equivalent' if n in cat['equivalent'] else 'gap')); r.update(kw); return r
    muts = [rec('K1 a', ['assertion']), rec('K2 b', ['crash']), rec('K3 c'), rec('T1 d', ['timeout']), rec('T2 e', ['assertion']), rec('E1 f', ['assertion'])]
    gates = [{'gate': 'g1.py', 'args': [], 'group': 'kernel', 'ok': True, 'seconds': 60.0, 'verdict': 'G1: PASS', 'tree': 'TREE'}, {'gate': 'g2.py', 'args': ['5'], 'group': 'kernel', 'ok': True, 'seconds': 30.0, 'verdict': 'G2: PASS', 'tree': 'TREE'},
             {'gate': 'g3.py', 'args': [], 'group': 'problems', 'ok': True, 'seconds': 90.0, 'verdict': 'G3: PASS', 'tree': 'TREE'}, {'gate': 'g4.py', 'args': [], 'group': 'repo', 'ok': True, 'seconds': 6.0, 'verdict': 'G4: PASS', 'tree': 'TREE'}]
    def fz(world, noarith, seed, counts, true, false, tree='TREE', **kw):
        r = {'world': world, 'noarith': noarith, 'seed': seed, 'n': 10, 'depth': 5, 'probes': 400, 'seconds_per_statement': 5.0, 'counts': counts, 'true': true, 'false': false, 'statements': true + false,
             'seconds': 3.0, 'tree': tree, 'reports': []}; r.update(kw); return r
    fuzz = [fz('nat', False, 0, {'PROVED': 6, 'FAILED': 4, 'REJECTED': 0, 'UNSOUND': 0, 'ERROR': 0}, 6, 4), fz('nat', False, 1, {'PROVED': 5, 'FAILED': 5, 'REJECTED': 0, 'UNSOUND': 0, 'ERROR': 0}, 5, 5),
            fz('nat', True, 0, {'PROVED': 3, 'FAILED': 7, 'REJECTED': 0, 'UNSOUND': 0, 'ERROR': 0}, 4, 6), fz('int', False, 0, {'PROVED': 4, 'FAILED': 6, 'REJECTED': 0, 'UNSOUND': 0, 'ERROR': 0}, 5, 5)]
    mode = lambda p, g, m, ba, br: {'tactic_proofs': p, 'globals': g, 'mutants': m, 'agree': ba + br, 'both_accept': ba, 'both_reject': br, 'kernel_accepts_checker_rejects': 0, 'checker_accepts_kernel_rejects': 0,
                                    'valid_rejected': 0, 'definitions_agree': True, 'definitions_rejected_by_checker': 0, 'definitions_rejected_by_kernel': 0}
    diff = {'tree': 'TREE', 'passed': True, 'failures': [], 'definitions': 31, 'definitions_rejected': 0, 'wrong_proofs': 6, 'wrong_proofs_accepted': 0, 'guard_cases': 7, 'guard_cases_agree': 7, 'argv_mutants_per_proof': 100, 'seconds': 12.0,
            'modes': {'legacy mode': mode(12, 35, 200, 9, 191), 'parameter mode': mode(21, 36, 300, 13, 287)}}
    rows = [{'id': 'F1', 'title': 'one', 'status': '**fixed**', 'state': 'fixed', 'evidence': ''}, {'id': 'F2a', 'title': 'two a', 'status': 'fixed (baseline)', 'state': 'fixed', 'evidence': ''},
            {'id': 'F2b', 'title': 'two b', 'status': '**open**', 'state': 'open', 'evidence': ''}, {'id': 'F3', 'title': 'three', 'status': '**partial**', 'state': 'partial', 'evidence': ''},
            {'id': 'F4', 'title': 'four', 'status': 'by construction', 'state': 'by construction', 'evidence': ''}, {'id': 'F5', 'title': 'five', 'status': '**open**', 'state': 'open', 'evidence': ''}]
    tags = {'F1': {'layers': ['kernel'], 'effect': 'accepts', 'note': ''}, 'F2a': {'layers': ['tactic'], 'effect': 'wrong', 'note': ''}, 'F2b': {'layers': ['checker'], 'effect': 'refuses', 'note': ''},
            'F3': {'layers': ['kernel', 'checker'], 'effect': 'resource', 'note': ''}, 'F4': {'layers': ['engine'], 'effect': 'wrong', 'note': ''}, 'F5': {'layers': ['pipeline'], 'effect': 'measurement', 'note': ''}}
    return dict(cat=cat, muts=muts, gates=gates, fuzz=fuzz, diff=diff, rows=rows, tags=tags, rec=rec, fz=fz)


def sec_O(c):
    import exp_soundness as XS, exp_bundle as EB
    F = o_fixture(); cat, rec, fz = F['cat'], F['rec'], F['fz']
    # (1) small pieces
    tmp = tempfile.mkdtemp(prefix='s51_o_')
    try:
        p = os.path.join(tmp, 'x.jsonl')
        with open(p, 'w') as f: f.write('{"a": 1}\n\n{"a": 2, "b"\n{"a": 3}\n[1, 2]\n')
        expect([r['a'] for r in XS.read_jsonl(p)] == [1, 3] and XS.read_jsonl(os.path.join(tmp, 'none')) == [] and XS.read_json(os.path.join(tmp, 'none'), 7) == 7, 'O1 a JSON-lines file is read record by record: a line cut short and a line that is not a record are skipped, a missing file is empty')
        expect(XS.parse_seeds('0-2,5') == [0, 1, 2, 5] and XS.parse_seeds('3') == [3], 'O2 seeds: ranges and lists')
        expect(XS.pct(1646, 1647) == '99.9' and XS.pct(1647, 1647) == '100.0' and XS.pct(0, 5) == '0.0' and XS.pct(1, 3) == '33.3' and XS.pct(2, 3) == '66.6' and XS.pct(0, 0) == '--', 'O3 a percentage is cut, never rounded up: 1646 of 1647 is 99.9, only all of them is 100.0')
        states = {'**fixed**': 'fixed', 'fixed (baseline b777dc8); ported': 'fixed', 'fixed (baseline); by construction': 'fixed', '**fixed** in kernel and checker': 'fixed', 'hybrid is stricter': 'fixed', 'by construction': 'by construction',
                  '**by design**': 'by construction', '**open, documented**; an incompleteness only': 'open', '**mitigated by F30**; root cause open (no memo)': 'open', 'mitigated; root cause open': 'open', '**partial**': 'partial',
                  '**documented, avoided**': 'documented', 'documented (design consequence)': 'documented', '**open (accepted)**': 'open', 'mitigated': 'mitigated', 'whatever': 'other'}
        bad = {k: (XS.state_of(k), v) for k, v in states.items() if XS.state_of(k) != v}
        expect(not bad, 'O4 the state of a finding is read off the first words of its status: %r' % (bad,))
        expect(XS.short_title('the ASSEMBLER (`gym_policy.py`: an action said as a sequence) rebuilds 1962 of the 1978 actions; the other 16 are refine steps') == 'the ASSEMBLER rebuilds 1962 of the 1978 actions'
               and XS.short_title('kernel conversion (`conv`, NbE) is incomplete: two terms') == 'kernel conversion is incomplete' and XS.short_title('short') == 'short'
               and XS.short_title('word ' * 40).endswith(' ...') and len(XS.short_title('word ' * 40)) <= 105, 'O5 a title for a table: parentheticals dropped, cut at the first colon or semicolon, else at a word boundary')
        # (2) the tags file
        tp = os.path.join(tmp, 'tags.tsv')
        with open(tp, 'w') as f:
            f.write('# comment\n\nF1\tkernel\taccepts\tnote\nF2\tkernel,checker\trefuses\nF3\tnowhere\taccepts\nF4\tkernel\tbroken\nF1\tkernel\twrong\nF5\tkernel\n')
        tags, probs = XS.read_tags(tp)
        expect(sorted(tags) == ['F1', 'F2', 'F3', 'F4'] and tags['F1'] == {'layers': ['kernel'], 'effect': 'accepts', 'note': 'note'} and tags['F2']['layers'] == ['kernel', 'checker'] and tags['F2']['note'] == '', 'O6 tags: comments and blank lines skipped, layers split, the note optional')
        expect(len(probs) == 4 and any('F3' in x and 'nowhere' in x for x in probs) and any('F4' in x and 'broken' in x for x in probs) and any('tagged twice' in x for x in probs) and any('F5' not in x and 'fields' in x for x in probs),
               'O7 a layer or an effect that does not exist, a finding tagged twice and a line with too few fields are all reported: %r' % (probs,))
        expect(XS.read_tags(os.path.join(tmp, 'nothing.tsv'))[1] != [], 'O8 a tags file that does not exist is a problem')
        # (3) findings
        fs = XS.findings_summary(F['rows'], F['tags'])
        mx = fs['matrix']
        expect(fs['rows'] == 6 and fs['findings'] == 5 and fs['problems'] == [] and fs['by_state'] == {'fixed': 2, 'open': 2, 'partial': 1, 'by construction': 1}, 'O9 findings: rows (F2 has two), findings, states: %r' % ({k: fs[k] for k in ('rows', 'findings', 'by_state', 'problems')},))
        expect(fs['trusted'] == ['F1', 'F2b', 'F3'] and fs['trusted_n'] == 3 and fs['accepts'] == ['F1'] and fs['accepts_n'] == 1 and fs['open'] == ['F2b', 'F3'] and fs['open_n'] == 2,
               'O10 the findings that concern the trusted code (kernel or checker), those that accepted what they must refuse, those not closed (a measurement is not a risk: F5 is left out): %r' % ({k: fs[k] for k in ('trusted', 'accepts', 'open')},))
        expect(mx['kernel']['accepts'] == 1 and mx['kernel']['resource'] == 1 and mx['checker']['refuses'] == 1 and mx['checker']['resource'] == 1 and mx['tactic']['wrong'] == 1 and mx['engine']['wrong'] == 1 and mx['pipeline']['measurement'] == 1
               and sum(sum(d.values()) for d in mx.values()) == 7, 'O11 the layer-by-effect matrix counts a finding in two layers in both')
        fs2 = XS.findings_summary(F['rows'] + [{'id': 'F9', 'title': 't', 'status': 'x', 'state': 'other', 'evidence': ''}], {k: v for k, v in F['tags'].items() if k != 'F4'} | {'F8': F['tags']['F1']})
        expect(any('F4 has no tag' in x for x in fs2['problems']) and any('F8' in x and 'not in docs/FINDINGS.md' in x for x in fs2['problems']) and any('F9' in x and 'none of the known states' in x for x in fs2['problems']) and any('F9 has no tag' in x for x in fs2['problems']),
               'O12 a finding without a tag, a tag without a finding and a status that is none of the states are reported: %r' % (fs2['problems'],))
        # the real tracker is fully tagged, and the trusted-code figures are the ones the documents state
        rows = XS.read_findings(); rtags, rprobs = XS.read_tags(); rfs = XS.findings_summary(rows, rtags)
        head = open(XS.FINDINGS_PATH).readline()
        top = int(__import__('re').search(r'F1-F(\d+)', head).group(1))
        expect(rprobs == [] and rfs['problems'] == [] and rfs['findings'] == top and len(rows) == len({r['id'] for r in rows}), 'O13 every row of docs/FINDINGS.md has exactly one tag and every tag a row (%d findings, the title says F1-F%d): %r' % (rfs['findings'], top, (rprobs + rfs['problems'])[:3]))
        expect(rfs['accepts'] == ['F19', 'F21', 'F26'] and 'F26' in rfs['trusted'] and rtags['F26']['layers'] == ['kernel', 'checker'] and rtags['F26']['effect'] == 'accepts', 'O14 the findings that accepted what they must refuse are F19 (the baseline), F21 and F26 (the opaque type that proved anything): %r' % (rfs['accepts'],))
        # (4) the gates
        g = XS.gates_summary(F['gates'], 'TREE')
        expect(g['total'] == 4 and g['passed'] == 4 and g['failed'] == [] and list(g['groups']) == ['kernel', 'problems', 'repo'] and g['groups']['kernel'] == {'gates': 2, 'passed': 2, 'seconds': 90.0} and abs(g['seconds'] - 186.0) < 1e-9 and g['stale'] == 0, 'O15 gates: totals, groups in the order of the system, seconds')
        rerun = F['gates'] + [dict(F['gates'][1], ok=False, seconds=5.0, verdict='G2: FAIL'), dict(F['gates'][1], ok=True, seconds=31.0), dict(F['gates'][3], tree='OLD', ok=False)]
        g = XS.gates_summary(rerun, 'TREE')
        expect(g['total'] == 4 and g['passed'] == 4 and g['stale'] == 1 and g['groups']['kernel']['seconds'] == 91.0, 'O16 the last record of a gate counts (a failure that was run again and passed is a pass), a record of another tree is left out and counted')
        g = XS.gates_summary(F['gates'][:3] + [dict(F['gates'][3], ok=False)], 'TREE')
        expect(g['failed'] == ['g4.py'] and g['passed'] == 3 and g['groups']['repo']['passed'] == 0, 'O17 a failed gate is listed')
        expect(XS.gates_summary([dict(F['gates'][0], args=['1']), dict(F['gates'][0], args=['2'], ok=False)], None)['total'] == 2, 'O18 the same gate with other arguments is another gate')
        # (5) the mutants
        m = XS.mutants_summary(F['muts'], cat, 'TREE')
        expect(m['catalogue'] == 6 and m['run'] == 6 and m['caught'] == 5 and m['equivalent'] == 1 and m['gap'] == 0 and m['build_failed'] == 0 and m['missing'] == [] and m['stale'] == [] and m['kill_rate'] == 1.0
               and m['kinds'] == {'assertion': 3, 'crash': 1, 'timeout': 1} and m['gaps'] == [] and m['survivors'] == [('K3 c', 'no input tells it apart')] and m['depth'] is None and m['gate_sets'] == [['g1.py', 'g2.py']], 'O19 mutants: a clean run (caught by what, the survivor with its reason)')
        expect(m['components']['kernel (C++)'] == {'catalogue': 3, 'run': 3, 'caught': 2, 'equivalent': 1, 'gap': 0, 'build_failed': 0} and list(m['components'])[:2] == ['kernel (C++)', 'tactic layer and decision procedures']
               and m['series']['K'] == {'catalogue': 3, 'run': 3, 'caught': 2, 'equivalent': 1, 'gap': 0, 'build_failed': 0} and m['series']['E']['caught'] == 1, 'O20 by component (in the order of the system) and by series')
        bad_runs = [rec('K1 a', ['assertion'], digest='other'), rec('T1 d', ['timeout'], tree='OLD'), rec('T2 e', status='gap'), rec('E1 f', status='build_failed'), rec('K2 b', status='what?')]
        m = XS.mutants_summary([r for r in F['muts'] if r['name'] not in ('K1 a', 'T1 d', 'T2 e', 'E1 f', 'K2 b')] + bad_runs, cat, 'TREE')
        expect(sorted(m['stale']) == ['K1 a', 'K2 b', 'T1 d'] and m['run'] == 3 and m['gap'] == 1 and m['build_failed'] == 1 and m['caught'] == 0 and m['equivalent'] == 1 and m['gaps'] == ['T2 e', 'E1 f'] and m['kill_rate'] == 0.0,
               'O21 a record made for another mutant (the digest), another tree or with a status nobody knows is stale and does not count; a survivor with no reason, and a mutant that did not build, are gaps: %r' % ({k: m[k] for k in ('stale', 'run', 'gap', 'build_failed', 'caught')},))
        m = XS.mutants_summary([rec('K1 a', status='gap'), rec('K3 c', status='gap'), rec('T2 e', status='equivalent'), rec('T1 d', ['assertion'], status='equivalent')], cat, 'TREE')
        expect(m['caught'] == 1 and m['equivalent'] == 1 and m['gap'] == 2 and m['gaps'] == ['K1 a', 'T2 e'] and m['missing'] == ['K2 b', 'E1 f'],
               'O22 the status is derived from what caught the mutant and from the catalogue, not copied from the record: a mutant that a gate caught is caught, one that survives is equivalent only if the catalogue says why: %r' % ({k: m[k] for k in ('caught', 'equivalent', 'gap', 'missing')},))
        last_wins = XS.mutants_summary([rec('K1 a', status='gap'), rec('K1 a', ['crash'])], cat, 'TREE')
        expect(last_wins['caught'] == 1 and last_wins['run'] == 1 and last_wins['kinds'] == {'crash': 1}, 'O23 the last record of a mutant counts')
        full = [dict(r, full=True, caught_by=r['caught_by'] + ([{'gate': 'g2.py', 'kind': 'timeout', 'first': ''}] if r['name'] == 'K1 a' else []) + ([{'gate': 'g3.py', 'kind': 'assertion', 'first': ''}] * 2 if r['name'] == 'K2 b' else [])) for r in F['muts']]
        mf = XS.mutants_summary(full, cat, 'TREE')
        expect(mf['depth'] == {'1': 3, '2': 1, '3+': 1} and mf['kinds'] == {'assertion': 3, 'crash': 1, 'timeout': 1},
               'O24 run against every gate: how many gates fail on a mutant (exactly one: %r), and a mutant is counted by the kind of its FIRST catch even when other gates (of another kind) catch it too: %r' % (mf['depth'], mf['kinds']))
        expect(XS.mutants_summary([dict(r, gates=['g1.py']) for r in F['muts']], cat, 'TREE')['gate_sets'] == [['g1.py']] and len(XS.mutants_summary(F['muts'][:3] + [dict(r, gates=['g1.py']) for r in F['muts'][3:]], cat, 'TREE')['gate_sets']) == 2, 'O25 the sets of gates the mutants were run against are listed')
        expect([r['status'] for r in XS.mutants_summary(F['muts'][:2], cat, 'TREE')['rows']] == ['caught', 'caught', 'not run', 'not run', 'not run', 'not run'], 'O26 one row per mutant of the catalogue, also those that were not run')
        # (6) fuzz
        fzs = XS.fuzz_summary(F['fuzz'], 'TREE'); cl = fzs['classes']
        expect(fzs['campaigns'] == 4 and list(cl) == [('nat', False), ('nat', True), ('int', False)] and cl[('nat', False)]['seeds'] == {0, 1} and cl[('nat', False)]['statements'] == 20 and cl[('nat', False)]['PROVED'] == 11 and cl[('nat', False)]['true'] == 11
               and cl[('nat', True)]['false'] == 6 and fzs['total']['statements'] == 40 and fzs['total']['true'] == 20 and fzs['total']['false'] == 20 and fzs['total']['PROVED'] == 18 and fzs['total']['UNSOUND'] == 0, 'O27 fuzz: campaigns, seeds, statements, outcomes per class and in all')
        dup = XS.fuzz_summary(F['fuzz'] + [dict(F['fuzz'][0]), fz('int', False, 1, {'PROVED': 1}, 1, 0, tree='OLD')], 'TREE')
        expect(dup['campaigns'] == 4 and dup['stale'] == 1 and dup['total']['statements'] == 40, 'O28 a campaign run twice with the same settings is one campaign; one made for another tree is left out and counted')
        rep = XS.fuzz_summary([fz('nat', False, 0, {'UNSOUND': 1, 'PROVED': 1}, 1, 1, reports=[['UNSOUND', False, 'a = b', 'tree']])], 'TREE')
        expect(rep['total']['UNSOUND'] == 1 and rep['reports'] == [{'world': 'nat', 'noarith': False, 'seed': 0, 'report': ['UNSOUND', False, 'a = b', 'tree']}], 'O29 the reports of a campaign are kept')
        # (7) the comparison with the independent checker
        d = XS.differential_summary(F['diff'], 'TREE')
        expect(d['problems'] == [] and d['totals']['mutants'] == 500 and d['totals']['both_accept'] == 22 and d['totals']['both_reject'] == 478 and d['totals']['tactic_proofs'] == 33 and d['per_proof'] == 100, 'O30 the comparison: totals over both modes')
        expect(XS.differential_summary(None, 'TREE') is None and XS.differential_summary(F['diff'], 'OTHER') == {'stale': True}, 'O31 no comparison is None, one made for another tree is stale')
        broken = copy.deepcopy(F['diff']); broken['passed'] = False; broken['failures'] = ['legacy mode: verdict disagreement on mutants']
        broken['modes']['legacy mode'].update(kernel_accepts_checker_rejects=2, valid_rejected=1, definitions_agree=False); broken.update(wrong_proofs_accepted=1, definitions_rejected=3, guard_cases_agree=6)
        pr = XS.differential_summary(broken, 'TREE')['problems']
        expect(len(pr) == 7 and any('gate failed' in x for x in pr) and any('disagree on 2 mutated' in x for x in pr) and any('1 valid proofs rejected' in x for x in pr) and any('accepts 1 wrong' in x for x in pr) and any('rejects 3 definitions' in x for x in pr)
               and any('which definitions' in x for x in pr) and any('6 of 7' in x for x in pr), 'O32 every way the comparison can go wrong is reported: %r' % (pr,))
        # (8) the verdict
        stamp = {'commit': 'abcdef0123456789', 'dirty': False, 'tree': 'TREE', 'build_older_than_source': False, 'time': '2026-11-02T10:00:00+0530', 'host': 'h', 'cores': 2, 'python': '3.13', 'compiler': 'g++', 'kernel_source_sha256': 'aa', 'kernel_binary_sha256': 'bb'}
        def clean():
            return {'gates': XS.gates_summary(F['gates'], 'TREE'), 'gates_expected': 4, 'mutants': XS.mutants_summary(F['muts'], cat, 'TREE'), 'fuzz': XS.fuzz_summary(F['fuzz'], 'TREE'), 'differential': XS.differential_summary(F['diff'], 'TREE'),
                    'findings': XS.findings_summary(F['rows'], F['tags']), 'inventory': {'components': [{'component': 'kernel (C++)', 'role': 'trusted', 'files': 1, 'lines': 100, 'sha256': 'x'}, {'component': 'independent checker', 'role': 'second', 'files': 2, 'lines': 50, 'sha256': 'y'}], 'facts': {}},
                    'stamp': dict(stamp)}
        st, fl, inc = XS.assess(clean())
        expect((st, fl, inc) == ('CLEAN', [], []), 'O33 complete evidence for this tree, every claim backed: CLEAN (%r)' % ((st, fl, inc),))
        def broken_by(edit):
            S = clean(); edit(S); return XS.assess(S)
        S2 = clean(); S2['gates'] = XS.gates_summary(F['gates'][:3] + [dict(F['gates'][3], ok=False)], 'TREE')
        s_, f_, i_ = XS.assess(S2); expect(s_ == 'FAILED' and any('gates failed: g4.py' in x for x in f_), 'O34 a failing gate: FAILED')
        s_, f_, i_ = broken_by(lambda S: S.update(mutants=XS.mutants_summary(F['muts'][:4] + [rec('T2 e', status='gap'), F['muts'][5]], cat, 'TREE')))
        expect(s_ == 'FAILED' and any('1 mutant(s) survive without a documented reason' in x for x in f_), 'O35 a mutant that survives without a reason: FAILED')
        for what, edit, text in (('a false statement that was proved', lambda S: S['fuzz']['total'].update(UNSOUND=1), 'FALSE statement'), ('a proof the kernel refused', lambda S: S['fuzz']['total'].update(REJECTED=2), '2 proof(s)'),
                                 ('an error in the environment', lambda S: S['fuzz']['total'].update(ERROR=1), '1 error'), ('a campaign that proves nothing true', lambda S: S['fuzz']['classes'][('int', False)].update(PROVED=0), 'proved none of its 5 true'),
                                 ('a disagreement with the checker', lambda S: S['differential']['problems'].append('kernel and checker disagree'), 'kernel and checker disagree'), ('an untagged finding', lambda S: S['findings']['problems'].append('F7 has no tag'), 'F7 has no tag')):
            s_, f_, i_ = broken_by(edit); expect(s_ == 'FAILED' and any(text in x for x in f_), 'O36 %s: FAILED (%r)' % (what, f_))
        for what, edit, text in (('no gate results', lambda S: S.update(gates=XS.gates_summary([], 'TREE')), 'no gate results'), ('gates missing', lambda S: S.update(gates_expected=54), '4 of 54'),
                                 ('no mutants', lambda S: S.update(mutants=XS.mutants_summary([], cat, 'TREE')), 'no mutation results'), ('some mutants missing', lambda S: S.update(mutants=XS.mutants_summary(F['muts'][:4], cat, 'TREE')), '4 of 6 mutants'),
                                 ('a restricted gate set', lambda S: S.update(mutants=XS.mutants_summary([dict(r, gates=['g1.py']) for r in F['muts']], cat, 'TREE')), '1 of the 2 gates'),
                                 ('no fuzz', lambda S: S.update(fuzz=XS.fuzz_summary([], 'TREE')), 'no fuzz'), ('no comparison', lambda S: S.update(differential=None), 'no comparison'),
                                 ('a stale comparison', lambda S: S.update(differential={'stale': True}), 'another tree'), ('stale gate records', lambda S: S.update(gates=XS.gates_summary(F['gates'] + [dict(F['gates'][0], tree='OLD')], 'TREE')), '1 gate record'),
                                 ('a dirty tree', lambda S: S['stamp'].update(dirty=True), 'uncommitted'), ('an old build', lambda S: S['stamp'].update(build_older_than_source=True), 'older than its source'),
                                 ('a tree that changed', lambda S: S.update(tree_changed=('A', 'B')), 'changed while'), ('evidence of another tree', lambda S: S['stamp'].update(tree_stale='OLDTREE'), 'OLDTREE')):
            s_, f_, i_ = broken_by(edit); expect(s_ == 'INCOMPLETE' and f_ == [] and any(text in x for x in i_), 'O37 %s: INCOMPLETE (%r %r)' % (what, f_, i_))
        # (9) the tables and the numbers
        S = clean(); T = XS.build_tables(S); N = XS.numbers(S)
        expect(list(T) == ['sound_gates', 'sound_mutants', 'sound_survivors', 'sound_fuzz', 'sound_differential', 'sound_trusted', 'sound_findings', 'sound_open', 'sound_summary'], 'O38 the tables, in order: %r' % (list(T),))
        expect(T['sound_gates'].rows == [['kernel, independent checker, memoisation, work budget', 2, 2, '1.5'], ['problem bank and contest proofs', 1, 1, '1.5'], ['repository layout, examples', 1, 1, '0.1'], ['all gates', 4, 4, '3.1']], 'O39 the gates table: %r' % (T['sound_gates'].rows,))
        expect(T['sound_mutants'].rows == [['kernel (C++)', '3', '3', '2', '1', '0', '100.0'], ['tactic layer and decision procedures', '2', '2', '2', '0', '0', '100.0'], ["learner's environment", '1', '1', '1', '0', '0', '100.0'], ['all', '6', '6', '5', '1', '0', '100.0']],
               'O40 the mutants table: %r' % (T['sound_mutants'].rows,))
        expect(T['sound_survivors'].rows == [['K3 c', 'no input tells it apart']], 'O41 the survivors with their reasons')
        expect(T['sound_fuzz'].rows == [['naturals', 2, '20', '11', '9', '11', '0', '0', '0'], ['naturals, without the arithmetic tactics', 1, '10', '4', '6', '3', '0', '0', '0'], ['integers', 1, '10', '5', '5', '4', '0', '0', '0'], ['all', 4, '40', '20', '20', '18', '0', '0', '0']], 'O42 the fuzz table: %r' % (T['sound_fuzz'].rows,))
        expect(T['sound_differential'].rows == [['legacy mode', '12', '35', '200', '9', '191', '0', '0'], ['parameter mode', '21', '36', '300', '13', '287', '0', '0'], ['all', '33', '', '500', '22', '478', '0', '0']], 'O43 the comparison table: %r' % (T['sound_differential'].rows,))
        expect(T['sound_trusted'].rows[0] == ['kernel (C++)', 'trusted', 1, '100', '3'] and T['sound_trusted'].rows[1] == ['independent checker', 'second', 2, '50', '--'], 'O44 the trusted table: lines from the inventory, mutants from the catalogue (the checker has none in this fixture): %r' % (T['sound_trusted'].rows,))
        expect(T['sound_findings'].header == ['layer', 'accepts', 'refuses', 'wrong', 'resource', 'measurement', 'total'] and T['sound_findings'].rows[0] == ['kernel', 1, '', '', 1, '', 2] and T['sound_findings'].rows[-1][1:] == [1, 1, 2, 1, 1, 6]
               and [r[0] for r in T['sound_open'].rows] == ['F2b', 'F3'], 'O45 the findings tables (only the effects that occur; a finding in two layers is in both rows and once in the last): %r' % (T['sound_findings'].rows,))
        expect(dict(T['sound_summary'].rows)['mutants caught'].startswith('5 of 6 (100.0%') and dict(T['sound_summary'].rows)['gates'] == '4 of 4 pass' and dict(T['sound_summary'].rows)['false statements searched for a proof'] == '20 (40 statements in all), 0 proved'
               and dict(T['sound_summary'].rows)['mutated proofs given to kernel and checker'] == '500, 0 disagreements' and dict(T['sound_summary'].rows)['findings'].startswith('5 (3 concern the trusted code, 1 of them accepted'), 'O46 the summary: %r' % (T['sound_summary'].rows,))
        expect(N['sound_gates'] == 4 and N['sound_mutants'] == 6 and N['sound_mutants_caught'] == 5 and N['sound_mutants_caught_pct'] == '100.0' and N['sound_mutants_assertion'] == 3 and N['sound_mutants_crash'] == 1 and N['sound_mutants_timeout'] == 1
               and N['sound_fuzz_statements'] == 40 and N['sound_fuzz_false'] == 20 and N['sound_fuzz_unsound'] == 0 and N['sound_diff_mutants'] == 500 and N['sound_diff_disagreements'] == 0 and N['sound_diff_definitions'] == 31
               and N['sound_findings'] == 5 and N['sound_findings_trusted'] == 3 and N['sound_findings_accepts'] == 1 and N['sound_findings_open'] == 2 and N['sound_kernel_lines'] == 100 and N['sound_checker_lines'] == 50, 'O47 the numbers: %r' % (dict(N),))
        # (10) the folder
        out = os.path.join(tmp, 'folder'); os.makedirs(os.path.join(out, 'evidence'))
        with open(os.path.join(out, 'evidence', 'stamp.json'), 'w') as f: f.write('{}')
        files = XS.render(S, out)
        want = ['README.md', 'report.md', 'soundness.json', 'numbers.json', 'numbers.tex', 'MANIFEST.sha256', 'tables/sound_mutants.md', 'tables/sound_mutants.tex', 'tables/sound_mutants.csv', 'stats/mutants.csv', 'stats/mutants_by_series.csv', 'stats/gates.csv', 'stats/fuzz.csv', 'stats/findings.csv']
        expect(all(os.path.exists(os.path.join(out, w)) for w in want) and EB.verify(out) == [], 'O48 the folder holds the tables, statistics, numbers and README, and its manifest verifies: %r %r' % ([w for w in want if not os.path.exists(os.path.join(out, w))], EB.verify(out)[:3]))
        readme = open(os.path.join(out, 'README.md')).read()
        expect('Status: **CLEAN**' in readme and '4 of them (`tests/gate_registry.py`)' in readme and 'The 2 gates of the mutation suite' in readme and '6 of them, hand-written' in readme and '`abcdef012345`' in readme and 'make soundness' in readme,
               'O49 the README states the status and the definitions with the numbers of THIS catalogue (none typed in)')
        sj = json.load(open(os.path.join(out, 'soundness.json')))
        expect(sj['status'] == 'CLEAN' and sj['numbers']['sound_mutants_caught'] == 5 and sj['mutants']['kinds'] == {'assertion': 3, 'crash': 1, 'timeout': 1} and sj['fuzz']['total']['UNSOUND'] == 0 and sj['fuzz']['classes']['nat']['seeds'] == [0, 1]
               and sj['differential']['totals']['mutants'] == 500 and sj['findings']['accepts'] == ['F1'], 'O50 soundness.json is the summary')
        tex = open(os.path.join(out, 'numbers.tex')).read()
        expect('\\newcommand{\\numSoundMutantsCaught}{5}' in tex and '\\newcommand{\\numSoundFuzzUnsound}{0}' in tex, 'O51 numbers.tex defines the macros')
        csvm = open(os.path.join(out, 'stats', 'mutants.csv')).read().strip().split('\n')
        expect(len(csvm) == 7 and csvm[1].startswith('K1 a,kernel (C++),f0.py,caught,assertion,g1.py'), 'O52 stats/mutants.csv: one row per mutant: %r' % (csvm[:2],))
        first = {fn: open(os.path.join(out, fn), 'rb').read() for fn in ('README.md', 'report.md', 'soundness.json', 'MANIFEST.sha256', 'tables/sound_mutants.tex')}
        XS.render(S, out)
        expect(all(open(os.path.join(out, fn), 'rb').read() == b for fn, b in first.items()), 'O53 the folder is a pure function of the summaries: rendering twice gives the same bytes')
        with open(os.path.join(out, 'tables', 'sound_mutants.md'), 'a') as f: f.write('x')
        expect(any('changed: tables/sound_mutants.md' in x for x in EB.verify(out)), 'O54 a table edited by hand is noticed by the manifest')
        S_bad = clean(); S_bad['fuzz']['total']['UNSOUND'] = 1
        XS.render(S_bad, out)
        expect('Status: **FAILED**' in open(os.path.join(out, 'README.md')).read() and 'FALSE statement' in open(os.path.join(out, 'README.md')).read(), 'O55 a contradicted claim is the first thing the README says')
        # (11) the real tools write the records the stage reads
        import run_gates as RG
        gr = RG.gate_record(['kernel/s2_tests.py', '1500'], False, 12.34, 'line 1\nS2 GATE: FAIL -- reason\n' + 'x\n' * 20)
        expect(gr['gate'] == 's2_tests.py' and gr['args'] == ['1500'] and gr['group'] == 'kernel' and gr['ok'] is False and gr['seconds'] == 12.3 and gr['verdict'] == 'S2 GATE: FAIL -- reason' and len(gr['tail']) == 12 and 'time' in gr
               and 'tail' not in RG.gate_record(['kernel/s2_tests.py'], True, 1.0, 'S2 GATE: PASS'), 'O56 a gate record: name, arguments, group (the directory), verdict, the end of the output for a failure only')
        jp = os.path.join(tmp, 'gates.jsonl'); rc, log = run_py([os.path.join(itp_paths.ROOT, 'tests', 'run_gates.py'), '--json', jp, 's16_quot_tests.py'])
        gl = XS.read_jsonl(jp)
        expect(rc == 0 and len(gl) == 1 and gl[0]['gate'] == 's16_quot_tests.py' and gl[0]['ok'] is True and gl[0]['group'] == 'kernel' and 'PASS' in gl[0]['verdict'], 'O57 run_gates.py --json appends one record per gate (%s)' % log[-200:])
        fp = os.path.join(tmp, 'fuzz.jsonl'); rc, log = run_py([os.path.join(itp_paths.ROOT, 'tests', 'fuzz', 'gym_fuzz.py'), '--seed', '5', '--n', '4', '--world', 'nat', '--seconds', '2', '--json', fp])
        fl = XS.read_jsonl(fp)
        expect(rc == 0 and len(fl) == 1 and fl[0]['seed'] == 5 and fl[0]['n'] == 4 and fl[0]['world'] == 'nat' and sum(fl[0]['counts'].values()) == 4 and fl[0]['true'] + fl[0]['false'] == 4 and fl[0]['statements'] == 4 and fl[0]['noarith'] is False
               and fl[0]['probes'] == 400 and fl[0]['seconds_per_statement'] == 2.0 and fl[0]['reports'] == [] and 'probe_ok' in fl[0]['stats'], 'O58 gym_fuzz.py --json appends one record per campaign: the settings, the counts, true and false (%s)' % log[-200:])
        dp = os.path.join(tmp, 's8.json'); rc, log = run_py([os.path.join(itp_paths.ROOT, 'tests', 'kernel', 's8_indep_tests.py'), '5'], timeout=600) if False else (None, '')
        env = dict(os.environ, S8_JSON=dp); pr = subprocess.run([sys.executable, os.path.join(itp_paths.ROOT, 'tests', 'kernel', 's8_indep_tests.py'), '5'], capture_output=True, text=True, timeout=600, env=env, cwd=itp_paths.ROOT)
        dj = XS.read_json(dp, {})
        expect(pr.returncode == 0 and dj.get('passed') is True and dj.get('argv_mutants_per_proof') == 5 and set(dj.get('modes', {})) == {'legacy mode', 'parameter mode'} and dj['modes']['legacy mode']['mutants'] > 0 and dj['guard_cases'] == dj['guard_cases_agree'] == 7
               and dj['wrong_proofs'] == 6 and dj['wrong_proofs_accepted'] == 0, 'O59 S8_JSON: what the comparison with the independent checker counted (%s)' % (pr.stdout + pr.stderr)[-200:])
        ds = XS.differential_summary(dict(dj, tree='T'), 'T')
        expect(ds['problems'] == [] and ds['totals']['mutants'] == sum(v['mutants'] for v in dj['modes'].values()), 'O60 and the summary of it has no problem')
        # (12) the real catalogue and the real inventory
        real = XS.catalogue_info(); import gate_registry as REG
        comps = collections.Counter(real['component'].values())
        expect(len(real['names']) == len(set(real['names'])) == sum(comps.values()) and set(real['component']) == set(real['names']) and len(set(real['digest'].values())) == len(real['names']) and set(real['equivalent']) <= set(real['names'])
               and real['gates'] == [os.path.basename(x[0]) for x in REG.GATE_CMDS], 'O61 the catalogue: every mutant has a component and its own digest, the equivalent ones are in it, the suite is the gate registry\'s')
        expect(set(comps) <= set(XS.COMPONENT_ORDER), 'O62 every mutant targets a component the table knows (a new one is added to COMPONENT_ORDER): %r' % (sorted(set(comps) - set(XS.COMPONENT_ORDER)),))
        inv = XS.inventory(); facts = XR.facts(); byc = {x['component']: x for x in inv['components']}
        expect(byc['kernel (C++)']['lines'] == facts['kernel_lines'] and byc['kernel bindings (C++)']['lines'] == facts['bindings_lines'] and byc['independent checker']['lines'] == facts['checker_lines'] and all(x['lines'] > 0 and x['files'] > 0 for x in inv['components'])
               and XS.inventory()['components'] == inv['components'], 'O63 the inventory counts the kernel, the bindings and the checker as exp_repo does, every component is non-empty, and the digests are stable')
        st = XS.stamp(); gs = XR.git_state()      # a tarball is not a git checkout: then the stamp has no commit (the digest of the tree is what names the code), and in a checkout it has HEAD
        expect(st['tree'] == XS.tree_digest() and len(st['tree']) == 16 and st.get('commit') == gs.get('commit') and st.get('dirty') == gs.get('dirty') and (not gs or len(st['commit']) == 40) and st['python'] and st['cores'] >= 1
               and 'kernel_source_sha256' in st, 'O64 the stamp names the code: the digest of the tree, the commit and whether the code is committed (when it is a checkout), the kernel, the machine')
        # (13) the whole thing, small: the stages run through their commands, the records carry the tree, a second run reuses them, an unknown stage is refused
        out = os.path.join(tmp, 'run'); t0 = time.time()
        argv = [os.path.join(itp_paths.ROOT, 'src', 'python', 'experiments', 'exp_soundness.py'), 'run', '--out', out, '--gates', 's16_quot_tests.py', '--mut-gates', 's22_simp_tests.py', '--mutants', 'TL1 tactics', 'TL2 tactics',
                '--fuzz-seeds', '0-1', '--fuzz-n', '5', '--fuzz-seconds', '2', '--diff-per-proof', '5', '--jobs', '2']
        rc, log = run_py(argv, timeout=900); trace('exp_soundness run %.1f s' % (time.time() - t0))
        ev = os.path.join(out, 'evidence'); tree = XS.tree_digest()
        gl, ml, fl, dj2 = (XS.read_jsonl(os.path.join(ev, 'gates.jsonl')), XS.read_jsonl(os.path.join(ev, 'mutants.jsonl')), XS.read_jsonl(os.path.join(ev, 'fuzz.jsonl')), XS.read_json(os.path.join(ev, 'differential.json'), {}))
        expect(rc == 2 and 'INCOMPLETE' in log and len(gl) == 1 and gl[0]['tree'] == tree and len(ml) == 2 and all(r['tree'] == tree and r['status'] == 'caught' for r in ml) and len(fl) == 6 and all(r['tree'] == tree for r in fl)
               and sorted((r['world'], r['noarith'], r['seed']) for r in fl) == [('int', False, 0), ('int', False, 1), ('nat', False, 0), ('nat', False, 1), ('nat', True, 0), ('nat', True, 1)] and dj2.get('tree') == tree and dj2.get('passed') is True,
               'O65 the stages run through their commands and every record carries the tree (exit %d)\n%s' % (rc, log[-600:]))
        sj = XS.read_json(os.path.join(out, 'soundness.json'), {})
        expect(sj.get('status') == 'INCOMPLETE' and any('1 of ' in x and 'gates have a result' in x for x in sj.get('incomplete', [])) and any('2 of ' in x and 'mutants have a result' in x for x in sj.get('incomplete', [])) and sj['fuzz']['total']['statements'] == 30
               and sj['numbers']['sound_mutants_caught'] == 2 and sj['numbers']['sound_diff_disagreements'] == 0 and EB.verify(out) == [], 'O66 a partial run says it is incomplete, and what it has counted is right: %r' % (sj.get('incomplete'),))
        t1 = time.time(); rc2, log2 = run_py(argv, timeout=900)
        stp = XS.read_json(os.path.join(ev, 'stamp.json'), {})
        expect(rc2 == 2 and stp['commands']['gates']['ran'] == 0 and stp['commands']['gates']['reused'] == 1 and stp['commands']['fuzz']['ran'] == 0 and stp['commands']['fuzz']['reused'] == 6 and stp['commands']['differential'].get('reused') is True
               and len(XS.read_jsonl(os.path.join(ev, 'fuzz.jsonl'))) == 6 and len(XS.read_jsonl(os.path.join(ev, 'gates.jsonl'))) == 1, 'O67 a second run reuses what has a result for this tree (gates, campaigns, the comparison): %r' % (stp['commands'],))
        # what was made for another tree, did not pass, or was made with other settings is run again, not reused (and the old records are not kept).  (The records are read BEFORE the file is opened for writing.)
        def jl_edit(path, **change):
            recs = XS.read_jsonl(path)
            with open(path, 'w') as f: f.write(''.join(json.dumps(dict(r, **change), sort_keys=True) + '\n' for r in recs))
            return len(recs)
        def js_edit(path, **change):
            d = XS.read_json(path, {}); d.update(change)
            with open(path, 'w') as f: json.dump(d, f)
            return d
        gp, fp, dpath = (os.path.join(ev, x) for x in ('gates.jsonl', 'fuzz.jsonl', 'differential.json'))
        expect(jl_edit(gp, tree='OTHER') == 1 and jl_edit(fp, tree='OTHER') == 6 and js_edit(dpath, tree='OTHER', marker='old').get('passed') is True and all(r['tree'] == 'OTHER' for r in XS.read_jsonl(gp) + XS.read_jsonl(fp)),
               'O68a (the test itself) the records are really there, with the tree changed')
        rc7, log7 = run_py(argv + ['--stages', 'gates,fuzz,differential'], timeout=900)
        stp2 = XS.read_json(os.path.join(ev, 'stamp.json'), {}); gl2, fl2, dj3 = XS.read_jsonl(gp), XS.read_jsonl(fp), XS.read_json(dpath, {})
        expect(rc7 == 2 and stp2['commands']['gates']['ran'] == 1 and stp2['commands']['gates']['reused'] == 0 and stp2['commands']['fuzz']['ran'] == 6 and stp2['commands']['fuzz']['reused'] == 0 and 'reused' not in stp2['commands']['differential']
               and len(gl2) == 1 and len(fl2) == 6 and all(r['tree'] == tree for r in gl2 + fl2) and dj3.get('tree') == tree and 'marker' not in dj3 and dj3.get('passed') is True,
               'O68 what was made for another tree is run again, not reused, and the records of the other tree are not kept: %r\n%s' % (stp2['commands'], log7[-400:]))
        for what, change in (('failed', {'passed': False}), ('was made with another number of mutants per proof', {'argv_mutants_per_proof': 99})):
            expect(js_edit(dpath, marker='old', **change).get('tree') == tree, 'O69a (the test itself) the comparison is there, for this tree, with %s' % what)
            run_py([argv[0], 'run', '--out', out, '--stages', 'differential', '--diff-per-proof', '5'], timeout=900)
            dj4 = XS.read_json(dpath, {})
            expect('marker' not in dj4 and dj4.get('passed') is True and dj4.get('argv_mutants_per_proof') == 5 and dj4.get('tree') == tree, 'O69 a comparison that %s is made again: %r' % (what, sorted(dj4)[:6]))
        expect(jl_edit(gp, ok=False) == 1 and XS.read_jsonl(gp)[0]['tree'] == tree, 'O70a (the test itself) the gate record is there, for this tree, failed')
        run_py(argv + ['--stages', 'gates'], timeout=900); gl3 = XS.read_jsonl(gp)
        expect(len(gl3) == 1 and gl3[0]['ok'] is True and gl3[0]['tree'] == tree, 'O70 a gate that failed is run again, and the failure is not kept: %r' % ([(r['gate'], r['ok']) for r in gl3],))
        rc3, log3 = run_py([argv[0], 'run', '--out', out, '--stages', 'nothing'])
        expect(rc3 == 2 and 'unknown stage' in log3, 'O71 an unknown stage is refused')
        rc4, log4 = run_py([argv[0], '--verify', out])
        with open(os.path.join(out, 'README.md'), 'a') as f: f.write('x')
        rc5, log5 = run_py([argv[0], '--verify', out])
        expect(rc4 == 0 and 'ok' in log4 and rc5 == 1 and 'changed: README.md' in log5, 'O72 --verify: the folder is what was written, and an edited file is named')
        rc6, log6 = run_py([argv[0], 'report', '--out', out])
        expect(rc6 == 2 and EB.verify(out) == [], 'O73 report makes the folder again from the evidence')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ P. the statement audit
def p_rec(name, **kw):
    """a record of the audit as problem_record makes it, for a clean contest problem; `kw` changes fields (spec=/sem= are merged into the defaults)"""
    import hashlib
    r = {'name': name, 'tree': 'TREE', 'statement_digest': hashlib.sha256(name.encode()).hexdigest()[:16], 'text': 'Prove that ' + name + '.', 'notes': '', 'source': 'IMO 1959 P1', 'year': 1959, 'origin': 'contest',
         'verified': 'answer-checked', 'kind': 'prove', 'level': 2, 'topics': ['algebra'], 'proved': True, 'helpers': 0, 'split': 'train', 'family': name, 'formal': 'forall x0 : Z, x0 = x0', 'in_fragment': True, 'features': [],
         'spec': {'params': ['Z'], 'hyps': 0, 'atoms': 1, 'hatoms': None, 'body': False, 'vacuous': False, 'domains': False, 'cap': 1200, 'exemptions': []},
         'sem': {'ok': True, 'errors': [], 'points': 100, 'nonvacuous': 50, 'mutants': 20, 'caught': 20, 'equivalent': 0, 'survivors': 0, 'shape_caught': 2, 'unevaluable': 0, 'counterexamples': 0},
         'sayable': True, 'seconds': 0.1}
    for k, v in kw.items():
        if k in ('spec', 'sem') and isinstance(v, dict) and r[k] is not None: r[k] = dict(r[k], **v)
        else: r[k] = v
    return r


def p_fixture():
    """ten hand-made records that between them have every caution, every kind of failure and every origin, and a ledger that puts them in every state"""
    R = [p_rec('p01', split='test'),                                                                                                              # clean, contest, test split
         p_rec('p02', origin='adapted', verified='recalled', proved=False, split='dev', features=['exists'], spec={'exemptions': [['conj', 'the same by symmetry']]}, sem={'equivalent': 2}),
         p_rec('p03', origin='folklore', verified='none', features=['hyp_equation'], spec={'atoms': None, 'hyps': 1}),
         p_rec('p04', verified='none', features=['pred', 'function'], spec={'atoms': None}, sem={'nonvacuous': 3}),
         p_rec('p05', origin='style', sem={'ok': False, 'errors': ['part concl disagrees with the intended meaning at 3 of 100 points'], 'survivors': 2}),
         p_rec('p06', spec=None, sem=None),
         p_rec('p07', in_fragment=False, formal=None, features=[], sem=None),
         p_rec('p08', proved=False, sem={'counterexamples': 3}),
         p_rec('p09', sayable=False),
         p_rec('p10', spec={'vacuous': True}, sem={'nonvacuous': 0})]
    dg = {r['name']: r['statement_digest'] for r in R}
    E = [{'name': 'p01', 'digest': dg['p01'], 'reviewer': 'ann', 'role': 'second', 'date': '2026-11-01', 'verdict': 'faithful', 'note': '', 'line': 1},
         {'name': 'p02', 'digest': 'f' * 16, 'reviewer': 'ann', 'role': 'second', 'date': '2026-11-01', 'verdict': 'faithful', 'note': '', 'line': 2},                              # another statement: stale
         {'name': 'p03', 'digest': dg['p03'], 'reviewer': 'zed', 'role': 'author', 'date': '2026-11-01', 'verdict': 'faithful', 'note': '', 'line': 3},                             # only the author
         {'name': 'p04', 'digest': dg['p04'], 'reviewer': 'ann', 'role': 'second', 'date': '2026-11-01', 'verdict': 'ambiguous', 'note': 'which n?', 'line': 4},
         {'name': 'p08', 'digest': dg['p08'], 'reviewer': 'ann', 'role': 'second', 'date': '2026-11-01', 'verdict': 'faithful', 'note': '', 'line': 5},
         {'name': 'p08', 'digest': dg['p08'], 'reviewer': 'bob', 'role': 'second', 'date': '2026-11-02', 'verdict': 'unfaithful', 'note': 'the bound is n, not n+1', 'line': 6},        # the readers disagree
         {'name': 'p09', 'digest': dg['p09'], 'reviewer': 'ann', 'role': 'second', 'date': '2026-11-01', 'verdict': 'unfaithful', 'note': '', 'line': 7},
         {'name': 'p10', 'digest': dg['p10'], 'reviewer': 'ann', 'role': 'second', 'date': '2026-11-01', 'verdict': 'adapted', 'note': '', 'line': 8}]
    return R, E


def sec_P(c):
    import csv
    import exp_audit as AU, exp_bundle as EB, exp_soundness as XS
    fx = c['mini']; bk = fx.bk
    R, E = p_fixture(); byname = {r['name']: r for r in R}
    tmp = tempfile.mkdtemp(prefix='s51_p_')
    try:
        # (1) reading an AST
        Z = lambda *a: ('prim',) + a
        ast_gcd = ('all', 'Z', ('imp', Z('Z.le', (('int', 0), ('bv', 0))), Z('Z.is_gcd', (Z('Z.add', (Z('Z.mul', (('int', 21), ('bv', 1))), ('int', 4))), Z('Z.add', (Z('Z.mul', (('int', 14), ('bv', 1))), ('int', 3))), ('int', 1)))))
        ast_fe = ('all', 'Z', ('iff', ('eq', Z('Z.mul', (('bv', 0), ('bv', 0))), ('bv', 0)), ('or', ('eq', ('bv', 0), ('int', 0)), ('eq', ('bv', 0), ('int', 1)))))
        ast_ne = ('all', 'Z', ('all', 'Z', ('imp', ('not', ('eq', ('bv', 0), ('int', 0))), ('imp', ('eq', Z('Z.add', (('bv', 2), ('bv', 1))), ('int', 5)), ('eq', ('bv', 3), ('bv', 3))))))
        expect([n[0] for n in AU.walk(ast_fe)] == ['all', 'iff', 'eq', 'prim', 'bv', 'bv', 'bv', 'or', 'eq', 'bv', 'int', 'eq', 'bv', 'int'], 'P1 walk: every node, outermost first: %r' % ([n[0] for n in AU.walk(ast_fe)],))
        expect(len(AU.hypotheses_of(ast_gcd)) == 1 and AU.hypotheses_of(ast_gcd)[0][1] == 'Z.le' and AU.hypotheses_of(ast_fe) == [] and len(AU.hypotheses_of(ast_ne)) == 2, 'P2 the hypotheses of a statement: the premises after the leading quantifiers')
        expect(not AU.positive_equation(('not', ('eq', ('bv', 0), ('int', 0)))) and AU.positive_equation(('eq', ('bv', 0), ('int', 0))) and AU.positive_equation(('not', ('not', ('eq', ('bv', 0), ('int', 0)))))
               and not AU.positive_equation(ast_gcd), 'P3 an equation counts unless it is negated (q /= 0 holds almost everywhere; x^2 = 2 y^2 at few points)')
        expect(AU.features_of(ast_gcd) == [] and AU.features_of(ast_ne) == ['hyp_equation'] and AU.features_of(ast_fe) == [] and AU.features_of(('all', 'N', ('eq', Z('pred', (('bv', 0),)), ('bv', 0)))) == ['pred']
               and AU.features_of(('all', ('fn', 1), ('ex', 'Z', ('eq', ('call', 1, (('bv', 0),)), ('int', 0))))) == ['exists', 'function'] and AU.features_of(('all', ('fn', 1), ('eq', ('int', 0), ('int', 0)))) == ['function']
               and AU.features_of(('all', 'N', ('eq', Z('Z.sum', (('lam', 'Z', ('bv', 0)), ('bv', 0))), Z('Z.fact', (('bv', 0),))))) == ['factorial', 'sum']
               and AU.features_of(('call', 0, (('int', 1),))) == [], 'P4 the constructs of a statement (a call of a function says nothing by itself: the quantifier that binds the function does): %r' % (AU.features_of(ast_ne),))
        # (2) reading it as a person does
        cases = [(ast_gcd, 'forall x0 : Z, 0 <= x0 -> gcd(21 * x0 + 4, 14 * x0 + 3) = 1'), (ast_fe, 'forall x0 : Z, x0 * x0 = x0 <-> x0 = 0 \\/ x0 = 1'),
                 (ast_ne, 'forall x0 x1 : Z, ~(x1 = 0) -> x0 + x1 = 5 -> x0 = x0'),
                 (('all', 'Z', ('all', 'Z', ('eq', Z('Z.sub', (('bv', 1), Z('Z.sub', (('bv', 0), ('int', -2))))), Z('Z.pow', (('int', -2), ('bv', 0)))))), 'forall x0 x1 : Z, x0 - (x1 - -2) = (-2)^x1'),
                 (('all', 'Z', ('all', 'N', ('eq', Z('Z.mul', (Z('Z.add', (('bv', 1), ('int', 1))), ('bv', 1))), Z('Z.pow', (Z('Z.pow', (('bv', 1), ('bv', 0))), ('bv', 0)))))), 'forall x0 : Z, forall x1 : N, (x0 + 1) * x0 = (x0^x1)^x1'),
                 (('all', 'Z', ('eq', Z('Z.sub', (Z('Z.sub', (('bv', 0), ('int', 1))), ('int', 2))), Z('Z.pow', (('bv', 0), Z('Z.pow', (('int', 2), ('int', 3))))))), 'forall x0 : Z, x0 - 1 - 2 = x0^2^3'),
                 (('and', ('all', 'Z', ('eq', ('bv', 0), ('bv', 0))), ('all', 'Z', ('not', ('eq', ('bv', 0), ('bv', 0))))), '(forall x0 : Z, x0 = x0) /\\ forall x0 : Z, ~(x0 = x0)'),
                 (('imp', ('imp', ('true',), ('false',)), ('true',)), '(true -> false) -> true'), (('imp', ('true',), ('imp', ('true',), ('false',))), 'true -> true -> false'),
                 (('all', 'Z', ('not', Z('Z.prime', (('bv', 0),)))), 'forall x0 : Z, ~prime(x0)'), (('all', 'N', ('eq', Z('Z.fact', (Z('Z.add', (('bv', 0), ('int', 1))),)), Z('Z.sum', (('lam', 'Z', Z('Z.mul', (('bv', 0), ('bv', 0)))), ('bv', 0))))), 'forall x0 : N, (x0 + 1)! = sum(i0 < x0, i0 * i0)'),
                 (('all', ('fn', 1), ('all', 'Z', ('eq', ('call', 1, (('bv', 0),)), ('call', 1, (Z('Z.neg', (('bv', 0),)),))))), 'forall x0 : Z -> Z, forall x1 : Z, x0(x1) = x0(-x1)'),
                 (('all', 'Z', Z('Z.dvd', (Z('Z.sub', (('bv', 0), ('int', 1))), Z('Z.sub', (Z('Z.pow', (('bv', 0), ('int', 2))), ('int', 1)))))), 'forall x0 : Z, x0 - 1 | x0^2 - 1'),
                 (('all', 'N', Z('N.lt', (('bv', 0), Z('N.succ', (Z('N.add', (('bv', 0), ('bv', 0))),))))), 'forall x0 : N, x0 < x0 + x0 + 1'), (('all', 'N', Z('Z.le', (Z('Z.of_N', (('bv', 0),)), ('int', 3)))), 'forall x0 : N, x0 <= 3'),
                 (('all', 'Z', ('bv', 5)), 'forall x0 : Z, v5')]
        bad = [(AU.pretty(a), w) for a, w in cases if AU.pretty(a) != w]
        expect(not bad, 'P5 a statement as a person reads it (quantifiers merged, infix, precedence, a hypothesis is a binder nobody can mention, the variables are reached across it): %r' % (bad[:2],))
        # the real statements of the small bank read well
        pr = {n: AU.pretty(bk.statement_ast(n)) for n in bk.placed if bk.statement_ast(n) is not None}
        expect(pr['imo1959_p1'] == cases[0][1] and len(pr) == len(bk.placed) and all(v and 'Npos' not in v and 'XO' not in v and v.count('(') == v.count(')') for v in pr.values()), 'P6 the statements of the small bank read as formulas (no numerals as binary constructors, balanced parentheses): %r' % ([n for n, v in pr.items() if 'Npos' in v or v.count('(') != v.count(')')],))
        # (3) the checks as failures, and the cautions
        want_fail = {'p01': [], 'p02': [], 'p03': [], 'p04': [], 'p05': ['2 slip(s) of the formalisation go unnoticed by the box'], 'p06': ['no spec: the statement is not compared with anything'],
                     'p07': ['the statement is outside the fragment the evaluator reads'], 'p08': ['no proof, and the statement is FALSE at 3 point(s) of the box'], 'p09': ['the statement is not sayable in the typed term builder'], 'p10': []}
        got_fail = {n: AU.failures_of(r) for n, r in byname.items()}
        expect(got_fail == want_fail, 'P7 what is wrong with a statement: %r' % ({n: v for n, v in got_fail.items() if v != want_fail[n]},))
        expect(AU.failures_of(p_rec('x', sem={'ok': False, 'errors': []})) == ['the semantic check failed'] and AU.failures_of(p_rec('x', sem={'ok': False, 'errors': ['e1', 'e2']})) == ['semantics: e1']
               and AU.failures_of(p_rec('x', proved=True, sem={'counterexamples': 4})) == [], 'P8 a failed check with and without a message; a false point does not matter for a statement that has a proof')
        want_c = {'p01': [], 'p02': ['adapted', 'source_recalled', 'window_existential', 'exempted_slips'], 'p03': ['hypotheses_unchecked', 'atoms_unchecked'],
                  'p04': ['source_unchecked', 'truncated_arithmetic', 'function_quantifier', 'atoms_unchecked', 'thin_box'], 'p05': [], 'p06': [], 'p07': [], 'p08': [], 'p09': [], 'p10': []}
        got_c = {n: AU.cautions_of(r) for n, r in byname.items()}
        expect(got_c == want_c, 'P9 the cautions of each record: %r' % ({n: v for n, v in got_c.items() if v != want_c[n]},))
        expect(AU.cautions_of(p_rec('x', origin='folklore', verified='none')) == [] and AU.cautions_of(p_rec('x', origin='style', verified='none')) == [] and AU.cautions_of(p_rec('x', origin='adapted', verified='none')) == ['adapted', 'source_unchecked']
               and AU.cautions_of(p_rec('x', features=['hyp_equation'], spec={'hatoms': 1})) == [] and AU.cautions_of(p_rec('x', sem={'nonvacuous': 0})) == [] and AU.cautions_of(p_rec('x', spec=None, sem=None)) == []
               and AU.cautions_of(p_rec('x', spec={'atoms': None}, sem={'nonvacuous': AU.THIN})) == ['atoms_unchecked'] and AU.cautions_of(p_rec('x', spec={'atoms': None}, sem={'nonvacuous': AU.THIN - 1})) == ['atoms_unchecked', 'thin_box']
               and AU.cautions_of(p_rec('x', spec={'atoms': None, 'body': True}, sem={'nonvacuous': 1})) == ['atoms_unchecked'] and AU.cautions_of(p_rec('x', spec={'atoms': None, 'vacuous': True}, sem={'nonvacuous': 0})) == ['atoms_unchecked'],
               'P10 a caution is raised only where it applies: an invented problem has no published wording to compare, equation atoms compared, a thin box matters only where nothing but a truth value is compared there (not with a body, not for an impossibility), no spec')
        # (4) the ledger
        led = os.path.join(tmp, 'ledger.tsv'); dgs = {n: r['statement_digest'] for n, r in byname.items()}
        with open(led, 'w') as f:
            f.write('# a comment\n\n' + '\t'.join(['p01', dgs['p01'], 'ann', 'second', '2026-11-01', 'faithful']) + '\n' + '\t'.join(['p04', dgs['p04'], 'ann', 'second', '2026-11-02', 'ambiguous', 'which n?']) + '\n'
                    + 'p05\tshort\n' + '\t'.join(['p06', 'zz', 'ann', 'second', '2026-11-01', 'faithful']) + '\n' + '\t'.join(['p07', dgs['p07'], '', 'second', '2026-11-01', 'faithful']) + '\n'
                    + '\t'.join(['p08', dgs['p08'], 'ann', 'boss', '2026-11-01', 'faithful']) + '\n' + '\t'.join(['p09', dgs['p09'], 'ann', 'second', '1 Nov', 'faithful']) + '\n' + '\t'.join(['p10', dgs['p10'], 'ann', 'second', '2026-11-01', 'fine']) + '\n')
        ent, probs = AU.read_reviews(led)
        expect([(e['name'], e['verdict'], e['note'], e['line']) for e in ent] == [('p01', 'faithful', '', 3), ('p04', 'ambiguous', 'which n?', 4)] and len(probs) == 6 and probs[0].startswith('line 5:') and 'digest' in probs[1] and 'no reviewer' in probs[2]
               and 'role' in probs[3] and 'date' in probs[4] and 'verdict' in probs[5] and AU.read_reviews(os.path.join(tmp, 'none')) == ([], []), 'P11 the ledger: comments and blank lines skipped, every wrong line named with its reason (%d problems): %r' % (len(probs), probs))
        st = {n: AU.review_state(n, byname[n]['statement_digest'], E) for n in byname}
        expect({n: s[0] for n, s in st.items()} == {'p01': 'confirmed', 'p02': 'stale', 'p03': 'author only', 'p04': 'ambiguous', 'p05': 'unreviewed', 'p06': 'unreviewed', 'p07': 'unreviewed', 'p08': 'disputed', 'p09': 'unfaithful', 'p10': 'confirmed'}
               and st['p08'][1] == ['ann', 'bob'] and st['p03'][1] == [] and st['p01'][1] == ['ann'], 'P12 where a statement stands with the ledger: %r' % ({n: s[0] for n, s in st.items()},))
        later = E + [dict(E[0], reviewer='ann', verdict='unfaithful', date='2026-11-03', line=9)]
        expect(AU.review_state('p01', byname['p01']['statement_digest'], later)[0] == 'unfaithful' and AU.review_state('p01', byname['p01']['statement_digest'], later + [dict(E[0], reviewer='cy', line=10)])[0] == 'disputed'
               and AU.review_state('p01', byname['p01']['statement_digest'], later + [dict(E[0], line=10)])[0] == 'confirmed' and AU.review_state('p01', 'e' * 16, E)[0] == 'stale',
               'P13 of several lines of one reader the last counts; a reading counts for the digest it was written for')
        # appending, and importing a filled sheet
        newl = os.path.join(tmp, 'new_ledger.tsv'); AU.append_reviews(newl, [('p01', dgs['p01'], 'ann', 'second', '2026-11-01', 'faithful', '')])
        e2, pr2 = AU.read_reviews(newl)
        expect(open(newl).read().startswith('# The ledger of second readers') and len(e2) == 1 and pr2 == [] and open(newl).read().endswith('faithful\n'), 'P14 a new ledger starts with its header and what is appended reads back')
        sheet = os.path.join(tmp, 'sheet.csv')
        def write_sheet(rows):
            with open(sheet, 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['name', 'digest', 'reviewer', 'role', 'date', 'verdict', 'note'])
                for r in rows: w.writerow(r)
        write_sheet([['p01', dgs['p01'], '', '', '', 'faithful', ''], ['p02', dgs['p02'], 'bob', 'second', '2026-11-05', 'adapted', 'the\tnote\nspans lines'], ['p03', dgs['p03'], '', '', '', '', '']])
        l2 = os.path.join(tmp, 'l2.tsv'); n_, pb_ = AU.import_reviews(sheet, l2, reviewer='cy', today='2026-11-09'); e3, _ = AU.read_reviews(l2)
        expect(n_ == 2 and pb_ == [] and [(e['name'], e['reviewer'], e['role'], e['date'], e['verdict']) for e in e3] == [('p01', 'cy', 'second', '2026-11-09', 'faithful'), ('p02', 'bob', 'second', '2026-11-05', 'adapted')] and e3[1]['note'] == 'the note spans lines',
               'P15 a filled sheet becomes lines of the ledger: the defaults fill in what is missing, a row with no verdict is a statement not read yet, a note stays on one line: %r' % (e3,))
        write_sheet([['p01', dgs['p01'], 'ann', 'second', '', 'faithful', ''], ['p02', dgs['p02'], 'ann', 'second', '', 'good', ''], ['p03', 'abc', 'ann', 'second', '', 'faithful', ''], ['p04', dgs['p04'], '', 'second', '', 'faithful', '']])
        l3 = os.path.join(tmp, 'l3.tsv'); n_, pb_ = AU.import_reviews(sheet, l3, reviewer=None)
        expect(n_ == 0 and len(pb_) == 3 and not os.path.exists(l3) and 'verdict' in pb_[0] and 'digest' in pb_[1] and 'no reviewer' in pb_[2], 'P16 a sheet with a wrong row appends nothing at all (a verdict that is none of the four, a digest that was edited, no reviewer): %r' % (pb_,))
        # (5) summaries and the verdict
        S = AU.summarize(R + [dict(R[0], tree='OLD')], 'TREE', [r['name'] for r in R] + ['p11'], E)
        T = S['totals']
        expect(S['expected'] == 11 and S['run'] == 10 and S['missing'] == ['p11'] and S['stale_records'] == 1 and S['stale'] == [] and [r['name'] for r in S['rows']] == [r['name'] for r in R], 'P17 the records that count: made for this tree, in the order of the bank; the missing ones listed, a record of another tree left out and counted')
        Sd = AU.summarize([p_rec('c1', split='dev', sem={'ok': False, 'survivors': 1}), p_rec('c2'), p_rec('c1', split='test')], 'TREE', ['c1', 'c2'], []); Sr = AU.summarize([p_rec('c1', split='test'), p_rec('c2'), p_rec('c1', split='dev', sem={'ok': False, 'survivors': 1})], 'TREE', ['c1', 'c2'], [])
        expect([r['split'] for r in Sd['rows']] == ['test', 'train'] and Sd['totals']['failed'] == 0 and Sd['run'] == 2 and [r['split'] for r in Sr['rows']] == ['dev', 'train'] and Sr['totals']['failed'] == 1 and Sr['stale_records'] == 0,
               'P17b a problem checked again counts by its newest record, in whichever order the file holds them (a fixed statement is no longer failing; a statement that broke afterwards is): %r %r' % ([r['split'] for r in Sd['rows']], dict(Sr['totals'])))
        expect(T['problems'] == 10 and T['proved'] == 8 and T['statement_only'] == 2 and T['with_spec'] == 9 and T['in_fragment'] == 9 and T['sayable'] == 9 and T['failed'] == 5 and T['checked'] == 5 and T['cautioned'] == 3 and T['compared'] == 7 and T['mutants'] == 160 and T['caught'] == 160
               and T['points'] == 800 and T['equivalent'] == 2 and T['survivors'] == 2 and T['exemptions'] == 1 and T['answer'] == 0, 'P18 the totals: %r' % (dict(T),))
        expect(S['cautions']['thin_box'] == 1 and S['cautions']['adapted'] == 1 and S['cautions']['atoms_unchecked'] == 2 and sum(S['cautions'].values()) == 11 and list(S['cautions']) == list(AU.CAUTIONS) and S['review_states'] == {'confirmed': 2, 'ambiguous': 1, 'author only': 1, 'stale': 1, 'unreviewed': 3, 'disputed': 1, 'unfaithful': 1}
               and S['readers'] == ['ann', 'bob'], 'P19 cautions and review states are counted per problem: %r %r' % (S['cautions'], S['review_states']))
        expect(list(S['by_origin']) == ['contest', 'adapted', 'style', 'folklore'] and S['by_origin']['contest']['problems'] == 7 and S['by_origin']['adapted']['statement_only'] == 1 and S['by_origin']['contest']['confirmed'] == 2 and list(S['by_split']) == ['test', 'dev', 'train'] and S['by_split']['test']['confirmed'] == 1,
               'P20 by origin and by split (the order of the paper): %r' % ({k: v['problems'] for k, v in S['by_origin'].items()},))
        status, fl, inc = AU.assess(S)
        expect(status == 'FAILED' and len(fl) == 2 and fl[0].startswith('5 statement(s) fail their checks: p05') and 'p08, p09' in fl[1] and any('1 of 11 problems have no record' in x for x in inc) and any('1 record(s) made for another tree' in x for x in inc), 'P21 the verdict: failed checks and an unfaithful or disputed reading fail the audit: %r %r' % (fl, inc))
        def clean_summary(**kw):
            rows = [p_rec('c1', split='test'), p_rec('c2', origin='folklore', verified='none', split='dev')]
            return AU.summarize(rows, 'TREE', ['c1', 'c2'], kw.get('entries', []), kw.get('lp', ())), rows
        S2, rows2 = clean_summary(); S2['stamp'] = {'tree': 'TREE'}
        expect(AU.assess(S2) == ('CLEAN', [], []), 'P22 a statement audit with no failing check and no reviews is CLEAN: how many a second reader confirmed is a number of the report, not a condition: %r' % (AU.assess(S2),))
        for what, edit, kind, text in (('a ledger line that is wrong', lambda S: S.update(ledger_problems=['line 3: the verdict'] ), 'FAILED', 'the ledger: line 3'), ('a line for a problem that is not in the bank', lambda S: S.update(unknown_reviews=['zz']), 'FAILED', 'not in the bank: zz'),
                                       ('no records', lambda S: S.update(rows=[], missing=['c1', 'c2']), 'INCOMPLETE', 'no problem records'), ('a record of another tree', lambda S: S.update(stale_records=2), 'INCOMPLETE', '2 record(s) made for another tree'),
                                       ('a missing problem', lambda S: S.update(missing=['c2'], expected=2), 'INCOMPLETE', '1 of 2 problems'), ('a dirty tree', lambda S: S['stamp'].update(dirty=True), 'INCOMPLETE', 'uncommitted'),
                                       ('evidence of another tree', lambda S: S['stamp'].update(tree_stale='OLD'), 'INCOMPLETE', 'tree OLD'), ('a tree that changed', lambda S: S.update(tree_changed=('A', 'B')), 'INCOMPLETE', 'changed while')):
            S3, _ = clean_summary(); S3['stamp'] = {'tree': 'TREE'}; edit(S3); s_, f_, i_ = AU.assess(S3)
            expect(s_ == kind and any(text in x for x in (f_ if kind == 'FAILED' else i_)), 'P23 %s: %s (%r %r)' % (what, kind, f_, i_))
        S4, _ = clean_summary(entries=[dict(E[0], name='c1', digest=rows2[0]['statement_digest'])]); S4['stamp'] = {'tree': 'TREE'}
        expect(AU.assess(S4)[0] == 'CLEAN' and S4['review_states']['confirmed'] == 1, 'P24 a confirmed reading is reported and keeps the audit CLEAN')
        # (6) tables and numbers
        S['stamp'] = {'tree': 'TREE', 'commit': 'abc', 'time': 'now'}; S['entries_list'] = E; S['ledger'] = {'path': 'docs/x.tsv', 'lines': len(E), 'sha256': 'ff'}
        TB_ = AU.build_tables(S); N = AU.numbers(S)
        expect(list(TB_) == ['audit_summary', 'audit_origins', 'audit_cautions', 'audit_reviews', 'audit_failures'] and AU.build_tables({'rows': []}) == collections.OrderedDict() and AU.numbers({'rows': []}) == collections.OrderedDict(), 'P25 the tables, in order; nothing to tabulate without records')
        sm = dict(TB_['audit_summary'].rows)
        expect(sm['problems'] == '10 (8 with a reference proof, 2 statement-only)' and sm['statement read back and compared with its intended meaning'] == '5 of 10' and sm['slips of the formalisation tried (one plausible mistake each)'] == '160, 160 caught, 2 declared equivalent to the statement'
               and sm['confirmed by a second reader'] == '2 of 10 (20.0%)' and sm['second readers'] == 'ann, bob' and sm['published wording compared with a source'] == '7 of 10' and sm['problems without a current record'] == '1', 'P26 the summary table: %r' % (sm,))
        expect(TB_['audit_origins'].rows[0] == ['contest', '7', '6', '1', '3', '100', '100', '6', '2'] and TB_['audit_origins'].rows[-1][:3] == ['all', '10', '8'] and [r[0] for r in TB_['audit_origins'].rows] == ['contest', 'adapted', 'style', 'folklore', 'all'], 'P27 the origins table: %r' % (TB_['audit_origins'].rows,))
        cr = {r[0]: r for r in TB_['audit_cautions'].rows}
        expect(list(cr) == list(AU.CAUTIONS) and cr['thin_box'][1:] == ['1', '10.0', 'p04'] and cr['truncated_arithmetic'][3] == 'p04' and cr['source_unchecked'][1] == '1' and cr['atoms_unchecked'][3] == 'p03, p04', 'P28 the cautions table: one row per caution, with the first problems: %r' % (cr['thin_box'],))
        expect([r[0] for r in TB_['audit_reviews'].rows] == list(AU.REVIEW_STATES) and TB_['audit_reviews'].rows[0] == ['confirmed', '2', '2', '1'] and TB_['audit_reviews'].rows[4] == ['unreviewed', '3', '2', '0'], 'P29 the reviews table: all, contest and adapted, test split: %r' % (TB_['audit_reviews'].rows,))
        expect(len(TB_['audit_failures'].rows) == 5 and TB_['audit_failures'].rows[0][0] == 'p05' and AU.build_tables(AU.summarize([R[0]], 'TREE', ['p01'], []))['audit_failures'].rows == [['(none)', '']], 'P30 the failures table, and "(none)" when there are none')
        expect(N['audit_problems'] == 10 and N['audit_confirmed'] == 2 and N['audit_confirmed_pct'] == '20.0' and N['audit_confirmed_contest'] == 2 and N['audit_confirmed_test'] == 1 and N['audit_test'] == 1 and N['audit_unreviewed'] == 4 and N['audit_issues'] == 3
               and N['audit_readers'] == 2 and N['audit_mutants'] == 160 and N['audit_caution_thin_box'] == 1 and N['audit_contest'] == 8 and N['audit_failed'] == 5 and N['audit_compared'] == 7, 'P31 the numbers: %r' % (dict(N),))
        # (7) the sheet and the folder
        md, cols, srows = AU.review_sheet(S)
        order = [r[0] for r in srows]
        expect(order == ['p06', 'p07', 'p08', 'p09', 'p05', 'p02', 'p04', 'p03'] and 'p01' not in order and 'p10' not in order and '## p05' in md and '8 statement(s) to read' in md and cols[:2] == ['name', 'digest'] and all(len(r) == len(cols) for r in srows),
               'P32 the review sheet: what fails first, then the test split, contest before invented, more cautions first; what a second reader has confirmed is left out: %r' % (order,))
        expect(len(AU.review_sheet(S, only_open=False)[2]) == 10 and srows[0][1] == byname['p06']['statement_digest'] and srows[0][cols.index('role')] == 'second' and srows[0][cols.index('verdict')] == '' and 'digest `%s`' % byname['p06']['statement_digest'] in md,
               'P33 the sheet carries the digest to put in the ledger and empty columns for the reader')
        out = os.path.join(tmp, 'folder'); os.makedirs(os.path.join(out, 'evidence'))
        files = AU.render(S, out); names = sorted(os.path.relpath(f, out) for f in files)
        want = ['README.md', 'report.md', 'audit.json', 'numbers.json', 'numbers.tex', 'review_sheet.md', 'review_sheet.csv', 'MANIFEST.sha256', 'tables/audit_summary.md', 'tables/audit_cautions.tex', 'tables/audit_reviews.csv', 'stats/problems.csv', 'stats/cautions.csv', 'stats/exemptions.csv', 'stats/reviews.csv']
        miss = [w for w in want if w != 'MANIFEST.sha256' and w not in names]
        expect(not miss and EB.verify(out) == [] and os.path.exists(os.path.join(out, 'MANIFEST.sha256')), 'P34 the folder holds the tables, statistics, numbers, the sheet and the README, and its manifest verifies: %r %r' % (miss, EB.verify(out)[:2]))
        rd = open(os.path.join(out, 'README.md')).read(); aj = json.load(open(os.path.join(out, 'audit.json'))); nt = open(os.path.join(out, 'numbers.tex')).read()
        expect(rd.startswith('# Statement audit\n\nStatus: **FAILED**') and 'p05' in rd and 'What this audit does and does not show' in rd and 'tables/' in rd and aj['status'] == 'FAILED' and aj['numbers']['audit_confirmed'] == 2 and '\\newcommand{\\numAuditConfirmed}{2}' in nt
               and '\\newcommand{\\numAuditConfirmedPct}{20.0}' in nt and 'docs/x.tsv' in rd, 'P35 README, audit.json and the macros say the same: status, failures, numbers, the limits')
        rows_csv = list(csv.DictReader(open(os.path.join(out, 'stats', 'problems.csv'))))
        ex_csv = list(csv.DictReader(open(os.path.join(out, 'stats', 'exemptions.csv')))); rv_csv = list(csv.DictReader(open(os.path.join(out, 'stats', 'reviews.csv'))))
        expect(len(rows_csv) == 10 and rows_csv[0]['name'] == 'p01' and rows_csv[4]['checks_hold'] == '0' and rows_csv[0]['checks_hold'] == '1' and rows_csv[1]['cautions'] == 'adapted source_recalled window_existential exempted_slips' and ex_csv == [{'problem': 'p02', 'slip contains': 'conj', 'reason': 'the same by symmetry'}]
               and len(rv_csv) == 8 and [r['counts for the current statement'] for r in rv_csv] == ['1', '0', '1', '1', '1', '1', '1', '1'], 'P36 stats/: a row per problem, per exempted slip, per line of the ledger (and whether the line counts for the statement as it is now)')
        # the sheet, filled in, goes back through the ledger and the problem is then confirmed
        with open(os.path.join(out, 'review_sheet.csv'), newline='') as f: sh = list(csv.DictReader(f))
        sh[0].update(verdict='faithful', reviewer='dee', note='checked against the source')
        with open(sheet, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sh[0].keys())); w.writeheader(); w.writerows(sh)
        l4 = os.path.join(tmp, 'l4.tsv'); n_, pb_ = AU.import_reviews(sheet, l4, today='2026-11-10'); e4, _ = AU.read_reviews(l4)
        expect(n_ == 1 and pb_ == [] and e4[0]['name'] == 'p06' and e4[0]['digest'] == byname['p06']['statement_digest'] and e4[0]['reviewer'] == 'dee' and AU.review_state('p06', byname['p06']['statement_digest'], e4)[0] == 'confirmed',
               'P37 the sheet that was written, filled in by a reader, is what import-reviews reads: the statement is confirmed for the digest it was read at')
        # (8) the real records of the small bank
        recs = {n: AU.problem_record(bk, n, 'TREE') for n in bk.placed}
        r1 = recs['imo1959_p1']
        expect(r1['statement_digest'] == fx.bank['imo1959_p1']['statement_digest'] and r1['formal'] == cases[0][1] and r1['origin'] == 'contest' and r1['verified'] == 'answer-checked' and r1['proved'] is True and r1['in_fragment'] and r1['sayable']
               and r1['sem']['ok'] and r1['sem']['mutants'] > 0 and r1['sem']['caught'] == r1['sem']['mutants'] and r1['sem']['survivors'] == 0 and r1['spec']['atoms'] == 1 and r1['split'] in ('train', 'dev', 'test') and r1['tree'] == 'TREE',
               'P38 the record of a problem of the small bank: the digest is the dataset\'s, the statement reads as a formula, the checks ran')
        expect(all(AU.failures_of(r) == [] and r['statement_digest'] == fx.bank[n]['statement_digest'] and r['sem']['ok'] and r['sem']['survivors'] == 0 for n, r in recs.items()) and len(recs) == len(bk.placed)
               and recs['imo2019_p1']['features'] == ['exists', 'function'] and 'function_quantifier' in AU.cautions_of(recs['imo2019_p1']) and recs['imo2019_p1']['spec']['exemptions'] and recs['sqrt2_irrational']['features'] == [],
               'P39 every problem of the small bank passes its checks (the bank\'s gates say so; the audit says it from the records): %r' % ([n for n, r in recs.items() if AU.failures_of(r)],))
        expect(json.loads(json.dumps(recs['imo2019_p1'])) == recs['imo2019_p1'] and all(set(r) == set(r1) for r in recs.values()), 'P40 a record is plain data: it survives JSON, and every record has the same fields')
        class Unproved:                                                                   # (a bank in which every problem is statement-only)
            def __init__(self, b): self._b = b
            def __getattr__(self, k): return getattr(self._b, k)
            @property
            def placed(self):
                class NoProof:
                    proved = False
                    def __init__(self, p): self._p = p
                    def __getattr__(self, k): return getattr(self._p, k)
                return {n: NoProof(p) for n, p in self._b.placed.items()}
        ru = AU.problem_record(Unproved(bk), 'imo1959_p1', 'TREE')
        expect(ru['proved'] is False and r1['proved'] is True and dict(ru, proved=True, seconds=0) == dict(r1, seconds=0), 'P40b the record says whether the problem has a reference proof (and nothing else about it changes): %r' % (ru['proved'],))
        class Boom:                                                                       # (a bank whose semantic check raises)
            def __init__(self, b): self._b = b
            def __getattr__(self, k): return getattr(self._b, k)
            def check_semantics(self, name, **kw): raise RuntimeError('boom')
        rb = AU.problem_record(Boom(bk), 'imo1959_p1', 'TREE')
        expect(rb['error'] == 'RuntimeError: boom' and AU.failures_of(rb) == ['the audit crashed on this problem: RuntimeError: boom'] and rb['statement_digest'] == r1['statement_digest'] and rb['origin'] == 'contest' and rb['spec'] is not None and rb['sem'] is None,
               'P41 a check that raises does not stop the audit: it is recorded as the problem\'s error, which fails the problem (the audit fails closed): %r' % (rb.get('error'),))
        # (9) the whole thing, small: the world is built, the checks run, the folder is written, a second run reuses, the verdict is INCOMPLETE (a part of the bank)
        out2 = os.path.join(tmp, 'run'); led2 = os.path.join(tmp, 'run_ledger.tsv'); t0 = time.time()
        script = os.path.join(itp_paths.ROOT, 'src', 'python', 'experiments', 'exp_audit.py'); base = [script, 'run', '--out', out2, '--reviews', led2]; argv = base + ['--only', 'imo1959_p1', 'find_all_x2_eq_x']
        rc, log = run_py(argv, timeout=900); trace('exp_audit run %.1f s' % (time.time() - t0))
        ev = os.path.join(out2, 'evidence'); pl_ = XS.read_jsonl(os.path.join(ev, 'problems.jsonl')); stm = XS.read_json(os.path.join(ev, 'stamp.json'), {})
        expect(rc == 2 and 'INCOMPLETE' in log and [r['name'] for r in pl_] == ['imo1959_p1', 'find_all_x2_eq_x'] and all(r['tree'] == XS.tree_digest() for r in pl_) and stm['only'] == ['imo1959_p1', 'find_all_x2_eq_x'] and len(stm['bank_names']) >= 268
               and stm['ran'] == 2 and stm['reused'] == 0 and any('have no record for this tree' in x for x in log.split('\n')), 'P42 the audit runs on a part of the bank and says it is incomplete (exit %d)\n%s' % (rc, log[-500:]))
        sj = XS.read_json(os.path.join(out2, 'audit.json'), {})
        expect(sj.get('status') == 'INCOMPLETE' and sj['totals']['problems'] == 2 and sj['numbers']['audit_checked'] == 2 and sj['expected'] >= 268 and len(sj['missing']) == sj['expected'] - 2 and EB.verify(out2) == [], 'P43 what it has counted is right: %r' % (sj.get('incomplete'),))
        rc2, log2 = run_py(argv, timeout=900); stm2 = XS.read_json(os.path.join(ev, 'stamp.json'), {})
        expect(rc2 == 2 and stm2['ran'] == 0 and stm2['reused'] == 2 and len(XS.read_jsonl(os.path.join(ev, 'problems.jsonl'))) == 2, 'P44 a second run reuses the records that were made for this tree and this statement: %r' % ((stm2['ran'], stm2['reused']),))
        # a record is reused only if it was made for this tree AND for this statement AND did not fail: a record whose statement digest is not the bank's, and one that has an error, are checked again
        pl_ok = XS.read_jsonl(os.path.join(ev, 'problems.jsonl'))
        with open(os.path.join(ev, 'problems.jsonl'), 'w') as f: f.write(json.dumps(dict(pl_ok[0], statement_digest='0' * 16), sort_keys=True) + '\n' + json.dumps(dict(pl_ok[1], error='RuntimeError: boom'), sort_keys=True) + '\n')
        rcT, logT = run_py(argv, timeout=900); stmT = XS.read_json(os.path.join(ev, 'stamp.json'), {}); plT = XS.read_jsonl(os.path.join(ev, 'problems.jsonl'))
        expect(stmT['ran'] == 2 and stmT['reused'] == 0 and [(r['name'], r['statement_digest'], bool(r.get('error'))) for r in plT] == [(r['name'], r['statement_digest'], False) for r in pl_ok],
               'P44b a record made for another statement, and one that failed, are checked again and replaced: %r' % ((stmT['ran'], stmT['reused']),))
        # a run on some of the problems keeps the records of the others (the folder is the work of several partial runs); --redo checks again what has a record and still keeps what it was not asked for
        rcU, logU = run_py(base + ['--only', 'imo1959_p1'], timeout=900); stmU = XS.read_json(os.path.join(ev, 'stamp.json'), {}); plU = XS.read_jsonl(os.path.join(ev, 'problems.jsonl'))
        expect(stmU['ran'] == 0 and stmU['reused'] == 1 and [r['name'] for r in plU] == ['imo1959_p1', 'find_all_x2_eq_x'], 'P44c a run on some problems keeps the records of the others: %r' % ([r['name'] for r in plU], ))
        rcV, logV = run_py(base + ['--only', 'imo1959_p1', '--redo'], timeout=900); stmV = XS.read_json(os.path.join(ev, 'stamp.json'), {}); plV = XS.read_jsonl(os.path.join(ev, 'problems.jsonl'))
        expect(stmV['ran'] == 1 and stmV['reused'] == 0 and sorted(r['name'] for r in plV) == ['find_all_x2_eq_x', 'imo1959_p1'] and [r for r in plV if r['name'] == 'find_all_x2_eq_x'] == [r for r in plT if r['name'] == 'find_all_x2_eq_x'],
               'P44d --redo checks the problems asked for again and keeps the records of the others: %r' % ((stmV['ran'], stmV['reused'], [r['name'] for r in plV]),))
        out4 = os.path.join(tmp, 'run_noreport'); rcN, logN = run_py([script, 'run', '--out', out4, '--reviews', led2, '--only', 'imo1959_p1', '--no-report'], timeout=900)
        expect(rcN == 0 and os.path.exists(os.path.join(out4, 'evidence', 'problems.jsonl')) and not os.path.exists(os.path.join(out4, 'README.md')) and not os.path.exists(os.path.join(out4, 'audit.json')) and 'INCOMPLETE' not in logN,
               'P44e --no-report: the records are written and nothing is summarised (exit 0): %r' % (rcN,))
        rc3, log3 = run_py(base + ['--only', 'no_such_problem'])
        expect(rc3 == 2 and 'no such problem' in log3, 'P45 an unknown problem name is refused')
        with open(os.path.join(ev, 'problems.jsonl'), 'w') as f: f.write(''.join(json.dumps(dict(r, tree='OTHER'), sort_keys=True) + '\n' for r in pl_))
        rc4, log4 = run_py(base + ['--only', 'imo1959_p1'], timeout=900); pl4 = XS.read_jsonl(os.path.join(ev, 'problems.jsonl'))
        expect(len(pl4) == 1 and pl4[0]['tree'] == XS.tree_digest() and pl4[0]['name'] == 'imo1959_p1', 'P46 records of another tree are not reused and not kept: %r' % ([(r['name'], r['tree']) for r in pl4],))
        # a reading goes in through the ledger: the sheet of the folder is filled in and imported, and the next report counts it
        shp = os.path.join(out2, 'review_sheet.csv'); sh = list(csv.DictReader(open(shp, newline='')))
        expect(len(sh) == 1 and sh[0]['name'] == 'imo1959_p1' and sh[0]['text'].startswith('Prove that the fraction') and sh[0]['formal'].startswith('forall x0 : Z, 0 <= x0') and sh[0]['digest'] == fx.bank['imo1959_p1']['statement_digest'],
               'P47 the review sheet of the folder holds the problem text and the statement as a formula: %r' % (sh[:1],))
        sh[0].update(verdict='faithful', reviewer='eve')
        with open(shp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sh[0].keys())); w.writeheader(); w.writerows(sh)
        rc5, log5 = run_py([script, 'import-reviews', shp, '--reviews', led2]); rc6, log6 = run_py([script, 'report', '--out', out2, '--reviews', led2]); sj6 = XS.read_json(os.path.join(out2, 'audit.json'), {})
        expect(rc5 == 0 and '1 line(s) appended' in log5 and rc6 == 2 and sj6['numbers']['audit_confirmed'] == 1 and sj6['readers'] == ['eve'] and sj6['ledger']['lines'] == 1 and EB.verify(out2) == [], 'P48 import-reviews appends the reading, report counts it: %r' % (sj6.get('numbers', {}).get('audit_confirmed'),))
        before = open(led2).read(); bad_sheet = [dict(r, verdict='good') for r in sh]; bad_path = os.path.join(tmp, 'bad_sheet.csv')           # (not in the folder: its manifest would change)
        with open(bad_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sh[0].keys())); w.writeheader(); w.writerows(bad_sheet)
        rcB, logB = run_py([script, 'import-reviews', bad_path, '--reviews', led2])
        expect(rcB == 1 and 'nothing appended' in logB and open(led2).read() == before, 'P48b a sheet with a wrong row is refused on the command line (exit 1) and the ledger is not touched: %r' % (rcB,))
        rc7, log7 = run_py([script, '--verify', out2])
        with open(os.path.join(out2, 'README.md'), 'a') as f: f.write('x')
        rc8, log8 = run_py([script, '--verify', out2])
        expect(rc7 == 0 and 'ok' in log7 and rc8 == 1 and 'changed: README.md' in log8, 'P49 --verify: the folder is what was written, and an edited file is named')
        # what a folder says about the code it was made for: read from its stamp (the code changed while the records were made; the stamp is of other code than the code here; there is no stamp)
        cf = os.path.join(tmp, 'collected'); os.makedirs(os.path.join(cf, 'evidence')); noled = os.path.join(tmp, 'no_such_ledger.tsv')         # (a ledger that does not exist has no lines; one with a line for a problem that is not in this bank fails the audit)
        def put_evidence(stamp, recs):
            with open(os.path.join(cf, 'evidence', 'stamp.json'), 'w') as f: json.dump(stamp, f)
            with open(os.path.join(cf, 'evidence', 'problems.jsonl'), 'w') as f: f.write(''.join(json.dumps(r, sort_keys=True) + '\n' for r in recs))
        put_evidence({'bank_names': ['c1', 'c2'], 'tree_before': 'TREE', 'tree_after': 'TREE', 'tree': 'TREE', 'commit': 'abc', 'dirty': False}, [p_rec('c1'), p_rec('c2')]); Sc = AU.collect(cf, tree='TREE', reviews=newl)
        expect(Sc['run'] == 2 and Sc['expected'] == 2 and 'tree_changed' not in Sc and 'tree_stale' not in Sc['stamp'] and Sc['stamp']['tree'] == 'TREE' and Sc['stamp']['commit'] == 'abc' and Sc['ledger']['lines'] == 1 and len(Sc['entries_list']) == 1,
               'P50 collect: the records of the code that is here, the stamp it was made with, the ledger: %r' % ({k: Sc[k] for k in ('run', 'expected')},))
        put_evidence({'bank_names': ['c1', 'c2'], 'tree_before': 'A', 'tree_after': 'TREE', 'tree': 'TREE'}, [p_rec('c1'), p_rec('c2')]); Sc2 = AU.collect(cf, tree='TREE', reviews=noled); st2, fl2, inc2 = AU.assess(Sc2)
        expect(Sc2.get('tree_changed') == ('A', 'TREE') and st2 == 'INCOMPLETE' and any('changed while the evidence was collected (A, then TREE)' in x for x in inc2), 'P51 collect: code that changed while the records were made is said: %r %r' % (Sc2.get('tree_changed'), inc2))
        put_evidence({'bank_names': ['c1', 'c2'], 'tree_before': 'OLD', 'tree_after': 'OLD', 'tree': 'OLD'}, [dict(p_rec('c1'), tree='OLD'), dict(p_rec('c2'), tree='OLD')]); Sc3 = AU.collect(cf, tree='TREE', reviews=noled); st3, fl3, inc3 = AU.assess(Sc3)
        expect(Sc3['stamp'].get('tree_stale') == 'OLD' and Sc3['stamp']['tree'] == 'TREE' and Sc3['run'] == 0 and Sc3['stale_records'] == 2 and st3 == 'INCOMPLETE' and any('collected on tree OLD' in x for x in inc3), 'P52 collect: evidence of other code is stale, whatever its records say: %r %r' % (Sc3['stamp'].get('tree_stale'), inc3))
        put_evidence({}, [p_rec('c1'), p_rec('c2')]); os.remove(os.path.join(cf, 'evidence', 'stamp.json')); Sc4 = AU.collect(cf, tree='TREE', reviews=noled)
        expect(Sc4['stamp'].get('collected') is False and Sc4['expected'] == 2 and Sc4['run'] == 2 and 'tree_changed' not in Sc4, 'P53 collect: a folder with records and no stamp is read, and its stamp says it was not collected (the verdict is the assessor\'s): %r %r' % ({k: Sc4.get(k) for k in ('expected', 'run', 'tree_changed')}, Sc4['stamp'].get('collected')))
        # the real ledger of the repository
        ents, probs = AU.read_reviews(AU.REVIEWS_PATH)
        import bank_all
        expect(os.path.exists(AU.REVIEWS_PATH) and probs == [] and {e['name'] for e in ents} <= set(bank_all.names()), 'P50 the ledger of the repository (docs/statement_reviews.tsv) reads without a problem and names problems of the bank: %r' % (probs[:3],))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ Q. the code the evidence is about
def sec_Q(c):
    import csv, re
    import exp_repo as XR, exp_bundle as EB, exp_soundness as XS, ml_meta as MT
    tmp = tempfile.mkdtemp(prefix='s51_q_')
    try:
        # (1) dirty means that the CODE is not committed
        expect(XR.CODE_PATHS == itp_paths.CODE_PATHS == ('src', 'tests', 'mutants', 'data/dataset_v2/manifest.json') and itp_paths.CODE_DIRS + itp_paths.CODE_FILES == itp_paths.CODE_PATHS and itp_paths.CODE_PATHSPEC == itp_paths.CODE_PATHS + (':(exclude)*.md',), 'Q1 the code is src, tests, mutants and the manifest of the dataset, defined once')
        repo = os.path.join(tmp, 'repo'); os.makedirs(repo)
        def git(*a): return subprocess.run(['git', '-C', repo, '-c', 'user.name=T', '-c', 'user.email=t@example.org'] + list(a), capture_output=True, text=True, check=True).stdout
        def put(rel, text):
            os.makedirs(os.path.dirname(os.path.join(repo, rel)), exist_ok=True)
            with open(os.path.join(repo, rel), 'w') as f: f.write(text)
        git('init', '-q', '-b', 'main')
        files = ('src/a.py', 'tests/t.py', 'mutants/m.py', 'data/dataset_v2/manifest.json', 'docs/d.md', 'paper/p.tex', 'tools/t.py', 'scripts/s.sh', 'README.md', 'tests/README.md', 'src/NOTES.md')
        for rel in files: put(rel, 'one\n')
        git('add', '-A'); git('commit', '-q', '-m', 'one')
        def both():
            """what exp_repo and ml_meta say, and that they agree"""
            a = XR.git_state(repo); real = MT.ROOT; MT.ROOT = repo
            try: commit, dirty = MT.git_state()
            finally: MT.ROOT = real
            expect(a.get('commit') == commit and a.get('dirty') == dirty, 'Q2b exp_repo and ml_meta say the same of the code: %r and %r' % (a, (commit, dirty)))
            return dirty
        st = XR.git_state(repo)
        expect(re.fullmatch(r'[0-9a-f]{40}', st['commit']) and st['dirty'] is False and both() is False, 'Q2 a committed tree is not dirty: %r' % (st,))
        wrong = []
        for rel in ('docs/d.md', 'paper/p.tex', 'tools/t.py', 'scripts/s.sh', 'README.md', 'tests/README.md', 'src/NOTES.md'):
            put(rel, 'edited\n')
            if both(): wrong.append(rel)
            git('checkout', '-q', '--', rel)
        for rel in ('docs/new.md', 'paper/new.tex', 'tools/new.py', 'data/dataset_v2/other.json', 'notes.txt', 'tests/new.md', 'mutants/new.md'):
            put(rel, 'new\n')
            if both(): wrong.append('new ' + rel)
            os.remove(os.path.join(repo, rel))
        expect(not wrong, 'Q3 a document (also a .md file below src, tests or mutants), the paper, a tool, a script or the README that is edited or added, and a data file that is not the manifest, do not make the tree dirty: %r' % (wrong,))
        wrong = []
        for rel in ('src/a.py', 'tests/t.py', 'mutants/m.py', 'data/dataset_v2/manifest.json'):
            put(rel, 'edited\n')
            if not both(): wrong.append(rel)
            git('checkout', '-q', '--', rel)
        for rel in ('src/new.py', 'tests/new.py', 'mutants/new.py'):
            put(rel, 'new\n')
            if not both(): wrong.append('new ' + rel)
            os.remove(os.path.join(repo, rel))
        expect(not wrong and both() is False, 'Q4 a changed file of the code, and a new file below src, tests or mutants, make it dirty: %r' % (wrong,))
        nogit = os.path.join(tmp, 'nogit'); os.makedirs(nogit); real = MT.ROOT; MT.ROOT = nogit
        try: mt = MT.git_state()
        finally: MT.ROOT = real
        expect(XR.git_state(nogit) == {} and mt == ('', None), 'Q5 a directory that is not a git checkout has no commit and no dirty flag (not "clean"): %r %r' % (XR.git_state(nogit), mt))
        f = XR.facts(); here = XR.git_state()                                                  # (a copy of the tree without .git -- a tarball -- is not a checkout: the facts then name no commit and no flag, and the digest all the same)
        expect(f['tree'] == XR.tree_digest() == XS.tree_digest() and re.fullmatch(r'[0-9a-f]{16}', f['tree']) and {k: f.get(k) for k in ('commit', 'dirty')} == {k: here.get(k) for k in ('commit', 'dirty')}
               and (not here or (isinstance(f['dirty'], bool) and re.fullmatch(r'[0-9a-f]{40}', f['commit']))), 'Q6 the facts of the repository name the code: the digest, and the commit and whether the code is committed when it is a checkout (%r)' % (here,))
        # (2) a run records the code it started with and the code it ended with
        runs = {'a': SP.Run('a', 'data', [sys.executable, '-c', 'pass'], mem_gb=0.1)}
        seq = iter(['TREE-START', 'TREE-END'])
        real_ct = RUN.code_tree; RUN.code_tree = lambda: next(seq)
        try:
            rdir = os.path.join(tmp, 'runs'); os.makedirs(rdir)
            sch = RUN.Scheduler(runs, rdir, ['a'], jobs=1, cores=1, mem_gb=8, say=lambda *a: None); sch.run()
        finally: RUN.code_tree = real_ct
        stt = RUN.read_status(rdir, 'a')
        expect(stt['returncode'] == 0 and stt['tree'] == 'TREE-START' and stt['tree_finish'] == 'TREE-END', 'Q7 a run records the digest of the code when it starts and when it ends: %r' % ({k: stt.get(k) for k in ('tree', 'tree_finish')},))
        shutil.rmtree(os.path.join(rdir, '.status')); sch = RUN.Scheduler(runs, rdir, ['a'], jobs=1, cores=1, mem_gb=8, say=lambda *a: None); sch.run(); stt = RUN.read_status(rdir, 'a')
        expect(stt['tree'] == stt['tree_finish'] == XR.tree_digest() == RUN.code_tree(), 'Q8 with the real digest: the same code at the start and at the end (%r)' % (stt['tree'],))
        # (3) what the report says about them
        def status(R, rid, rc=0, commit='c1', tree=None, tree_finish=None):
            os.makedirs(os.path.join(R, '.status'), exist_ok=True)
            d = {'id': rid, 'returncode': rc, 'seconds': 10.0, 'commit': commit}
            if tree is not None: d['tree'] = tree
            if tree_finish is not None: d['tree_finish'] = tree_finish
            json.dump(d, open(os.path.join(R, '.status', rid.replace('/', '__') + '.json'), 'w'))
        def summary(*specs):
            R = tempfile.mkdtemp(dir=tmp)
            for i, sp in enumerate(specs): status(R, 'r%d' % i, **sp)
            commits, done, failed, secs, trees, moved = REP.status_summary(R)
            return REP.code_warnings(commits, trees, moved), (done, failed, dict(trees), moved)
        A, B = 'a' * 16, 'b' * 16
        w, info = summary(dict(tree=A, tree_finish=A, commit='c1'), dict(tree=A, tree_finish=A, commit='c2'), dict(tree=A, tree_finish=A, commit='c3'))
        expect(w == [] and info[2] == {A: 3}, 'Q9 runs made with one version of the code are one version, whatever commits (a commit that changed only the paper) they were made at: %r' % (w,))
        w, info = summary(dict(tree=A, tree_finish=A), dict(tree=B, tree_finish=B), dict(tree=B, tree_finish=B))
        expect(len(w) == 1 and 'different versions of the code' in w[0] and A[:12] in w[0] and B[:12] in w[0] and '2 different' in w[0], 'Q10 runs made with two versions of the code are named: %r' % (w,))
        w, info = summary(dict(tree=A, tree_finish=A), dict())
        expect(len(w) == 1 and 'different versions' in w[0] and '1 run(s) recorded none' in w[0], 'Q11 a run that recorded no digest next to one that did cannot be said to be the same version: %r' % (w,))
        w, info = summary(dict(commit='c1'), dict(commit='c2'))
        expect(len(w) == 1 and 'different commits' in w[0] and 'recorded no code tree' in w[0], 'Q12 an older runs directory, with no digests, is judged by its commits: %r' % (w,))
        w, info = summary(dict(commit='c1'), dict(commit='c1'))
        expect(w == [], 'Q13 and one commit is one version: %r' % (w,))
        w, info = summary(dict(tree=A, tree_finish=B), dict(tree=A, tree_finish=A))
        expect(len(w) == 1 and 'changed while 1 run(s) were running' in w[0] and 'r0' in w[0] and info[3] == ['r0'], 'Q14 a run whose code changed while it ran is named: %r' % (w,))
        w, info = summary(dict(tree=A, tree_finish=''), dict(tree='', tree_finish=B), dict(tree=A, tree_finish=A))
        expect(info[3] == [] and not [x for x in w if 'changed while' in x], 'Q14b a run that recorded no digest at the start, or none at the end, cannot be said to have moved: %r %r' % (w, info[3]))
        w, info = summary(dict(tree=A, tree_finish=A), dict(tree=B, tree_finish=B, rc=1))
        expect(w == [] and info[0] == 1 and info[1] == 1 and info[2] == {A: 1}, 'Q15 a failed run is not counted in the versions (it is counted as failed): %r %r' % (w, info))
        w, info = summary(*[dict(tree='%016x' % i, tree_finish='%016x' % i) for i in range(9)] + [dict(tree=A, tree_finish=B) for _ in range(8)])
        expect(any('(r10, r11, r12, r13, r14, r15 ...)' in x for x in w), 'Q16 many runs that moved are listed in part: %r' % (w[-1:],))
        # (4) the report and the bundle carry it
        R = os.path.join(tmp, 'rep'); write_fixture(R)
        status(R, 'ft/bc/s0', commit='c1', tree=A, tree_finish=A); status(R, 'ft/bc/s1', commit='c2', tree=A, tree_finish=A); status(R, 'ft/bc/s2', commit='c3', tree=B, tree_finish=B)
        out = os.path.join(tmp, 'rep_out'); rc = REP.main([R, '--out', out, '--no-figures']); md = open(os.path.join(out, 'report.md')).read()
        warn = md.split('## Warnings')[1].split('\n## ')[0]
        expect(rc == 0 and 'code trees: %s (2 runs), %s (1 runs)' % (A[:12], B[:12]) in md.split('## Warnings')[0] and 'the runs were made with 2 different versions of the code' in warn and 'different commits' not in warn,
               'Q17 the report names the versions of the code and warns of two of them (not of three commits): %r' % (warn[:300],))
        rows = list(csv.DictReader(open(os.path.join(out, 'provenance', 'runs.csv'), newline='')))
        expect(list(rows[0].keys())[-3:] == ['commit', 'tree', 'tree_finish'] and [(r['id'], r['tree'], r['tree_finish']) for r in rows] == [('ft/bc/s0', A, A), ('ft/bc/s1', A, A), ('ft/bc/s2', B, B)] and EB.verify(out) == [],
               'Q18 provenance/runs.csv has a column for the code at the start and at the end of every run: %r' % (rows[:1],))
        R3 = os.path.join(tmp, 'rep3'); write_fixture(R3)                                          # (a run that began with one version of the code and ended with another)
        status(R3, 'ft/bc/s0', commit='c1', tree=A, tree_finish=A); status(R3, 'ft/bc/s1', commit='c1', tree=A, tree_finish=B)
        out3 = os.path.join(tmp, 'rep_out3'); REP.main([R3, '--out', out3, '--no-figures']); warn3 = open(os.path.join(out3, 'report.md')).read().split('## Warnings')[1].split('\n## ')[0]
        rows3 = list(csv.DictReader(open(os.path.join(out3, 'provenance', 'runs.csv'), newline='')))
        expect('the code changed while 1 run(s) were running (ft/bc/s1)' in warn3 and [(r['id'], r['tree'], r['tree_finish']) for r in rows3] == [('ft/bc/s0', A, A), ('ft/bc/s1', A, B)],
               'Q18b the report names a run whose code changed while it ran, and runs.csv has the digest at the start and the digest at the end of every run: %r %r' % (warn3[:200], [(r['id'], r['tree'], r['tree_finish']) for r in rows3]))
        out2 = os.path.join(tmp, 'rep_out2'); R2 = os.path.join(tmp, 'rep2'); write_fixture(R2)
        status(R2, 'ft/bc/s0', commit='c1', tree=A, tree_finish=A); status(R2, 'ft/bc/s1', commit='c2', tree=A, tree_finish=A)
        REP.main([R2, '--out', out2, '--no-figures']); warn2 = open(os.path.join(out2, 'report.md')).read().split('## Warnings')[1].split('\n## ')[0]
        expect('different' not in warn2 and 'changed while' not in warn2, 'Q19 two runs of one version at two commits: nothing to warn of: %r' % (warn2[:200],))
    finally: shutil.rmtree(tmp, ignore_errors=True)


def sec_R(c):
    """R: the code on a machine it was not written on (macOS on Apple silicon): the memory, the cores and the CPU where /proc does not exist, the budgets the runner takes, the units of the peak memory, the pre-flight tool"""
    import ml_meta as MT, importlib.util
    real = (MT._read, MT._sysctl)
    try:
        mac = {'machdep.cpu.brand_string': 'Apple M3 Pro', 'hw.memsize': '19327352832', 'hw.perflevel0.logicalcpu': '6', 'hw.perflevel1.logicalcpu': '6'}
        MT._read = lambda path: ''; MT._sysctl = lambda name: mac.get(name, '')                          # (a Mac: no /proc, but sysctl)
        expect(MT.cpu_model() == 'Apple M3 Pro', 'R1 the CPU of a Mac is read from sysctl: %r' % (MT.cpu_model(),))
        expect(MT.memory_gb() == 18.0, 'R2 the memory of a Mac is hw.memsize: %r' % (MT.memory_gb(),))
        expect(MT.usable_cores() == 6, 'R3 an Apple-silicon Mac counts its performance cores only: %r' % (MT.usable_cores(),))
        MT._sysctl = lambda name: ''                                                                      # (nothing tells: the page size and the number of pages)
        m = MT.memory_gb(); expect(m is None or m > 0, 'R4 without /proc and sysctl the memory comes from sysconf or is unknown: %r' % (m,))
        expect(MT.usable_cores() >= 1 and MT.cpu_model(), 'R5 the cores and the CPU have an answer whatever the system')
        MT._read = lambda path: {'/proc/meminfo': 'MemTotal:       16384000 kB\nMemFree: 1 kB\n', '/proc/cpuinfo': 'processor\t: 0\nmodel name\t: Some CPU @ 3GHz\n',
                                 '/proc/self/status': 'VmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\n'}.get(path, '')
        expect(abs(MT.memory_gb() - 15.6) < 0.06 and MT.cpu_model() == 'Some CPU @ 3GHz' and MT.rss_mb() == 1.0 and MT.peak_rss_mb() == 2.0, 'R6 Linux is still read from /proc: %r %r %r %r' % (MT.memory_gb(), MT.cpu_model(), MT.rss_mb(), MT.peak_rss_mb()))
    finally: MT._read, MT._sysctl = real
    expect(MT.maxrss_mb(150994944, 'darwin') == 144.0 and MT.maxrss_mb(147456, 'linux') == 144.0, 'R7 ru_maxrss is bytes on macOS and kilobytes on Linux: %r %r' % (MT.maxrss_mb(150994944, 'darwin'), MT.maxrss_mb(147456, 'linux')))
    expect(RUN.mem_total_gb() > 0, 'R8 the runner knows the memory of the machine')
    expect(RUN.default_budgets(18.0, 6, 'darwin') == (6, 12.0) and RUN.default_budgets(36.0, 12, 'darwin') == (12, 30.0) and RUN.default_budgets(8.0, 8, 'darwin') == (8, 4.0),
           'R9 a Mac keeps 6 GB (or half) for the system: %r %r %r' % (RUN.default_budgets(18.0, 6, 'darwin'), RUN.default_budgets(36.0, 12, 'darwin'), RUN.default_budgets(8.0, 8, 'darwin')))
    cores, mem = RUN.default_budgets(64.0, 16, 'linux')
    expect(cores == 16 and close(mem, 54.4) and RUN.default_budgets(8.0, 0, 'linux')[0] == 1, 'R10 a server gives the runs 85%% of its memory and at least one core: %r' % ((cores, mem),))
    spec = importlib.util.spec_from_file_location('itp_preflight', os.path.join(itp_paths.ROOT, 'tools', 'preflight.py')); PF = importlib.util.module_from_spec(spec); spec.loader.exec_module(PF)
    ver = [("c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0", ('gcc', 13, True)), ("Apple clang version 16.0.0 (clang-1600.0.26.6)\nTarget: arm64-apple-darwin24.1.0", ('apple', 16, True)),
           ("Apple clang version 15.0.0 (clang-1500.3.9.4)", ('apple', 15, False)), ("Homebrew clang version 18.1.8\nTarget: arm64-apple-darwin", ('clang', 18, True)),
           ("clang version 16.0.6", ('clang', 16, False)), ("g++-14 (Homebrew GCC 14.2.0) 14.2.0", ('gcc', 14, True)), ("c++ (GCC) 12.2.0", ('gcc', 12, False)), ("something else", (None, 0, None))]
    got = [(PF.parse_compiler_version(t)[:2] + (PF.compiler_ok(*PF.parse_compiler_version(t)[:2]),)) for t, _ in ver]
    expect(got == [w for _, w in ver], 'R11 the compiler is told apart (Apple clang 16 has libc++ 17; clang needs 17, gcc 13): %r' % (got,))
    expect(PF.parse_power_source("Now drawing from 'AC Power'\n -InternalBattery-0 100%") == 'ac' and PF.parse_power_source("Now drawing from 'Battery Power'") == 'battery' and PF.parse_power_source('') is None,
           'R12 pmset -g batt: the power source')
    expect(PF.parse_low_power_mode(' lowpowermode         1\n sleep 1') is True and PF.parse_low_power_mode('lowpowermode 0') is False and PF.parse_low_power_mode('') is None, 'R13 pmset -g: Low Power Mode')
    expect(PF.parse_bash_version('3.2.57(1)-release') == (3, 2) and PF.parse_bash_version('5.2.37(1)-release\n') == (5, 2) and PF.parse_bash_version('') is None, 'R13b the shell version: the macOS default is bash 3.2')
    expect(PF.fork_probe_ok(0, 'x\nRESULT [1, 2, 3, 4]\n') and not PF.fork_probe_ok(1, 'RESULT [1, 2, 3, 4]') and not PF.fork_probe_ok(0, 'RESULT [1.0, 2.0, 3.0, 4.0]') and not PF.fork_probe_ok(0, ''), 'R13c the fork probe is judged by its exit code and its four results in order')
    if sys.platform.startswith('linux'):                                                                   # (the probe itself, as the pre-flight runs it: it printed floats once and was judged a failure on every machine)
        rc_, txt_ = PF.run([sys.executable, '-c', PF.FORK_PROBE % {'root': itp_paths.ROOT}], 150)
        expect(PF.fork_probe_ok(rc_, txt_), 'R13d the fork probe of the pre-flight succeeds where forked workers next to torch work (Linux): exit %s %r' % (rc_, txt_[-200:]))
    sw = PF.parse_swapusage('total = 2048.00M  used = 1023.25M  free = 1024.75M  (encrypted)')
    expect(sw == {'total_mb': 2048.0, 'used_mb': 1023.25, 'free_mb': 1024.75} and PF.parse_swapusage('total = 1.50G used = 0.00M free = 1.50G')['total_mb'] == 1536.0 and PF.parse_swapusage('garbage') == {}, 'R14 vm.swapusage: %r' % (sw,))
    # the held-out synthetic theorems: the runs of the graph, their cost, and the table (a fixture of rows: blind, and a network under two searches and two seeds)
    P_ = SP.profile('small'); runs_ = SP.build('small'); sy = {i: r for i, r in runs_.items() if i.startswith('eval/synth/')}
    expect('eval/synth/baselines' in sy and {'eval/synth/init', 'eval/synth/bc_scratch', 'eval/synth/bc', 'eval/synth/ppo', 'eval/synth/ei_her'} <= set(sy) and 'init/transformer/s0' in runs_ and all('eval/synth/pre_n%d' % n in sy for n in P_['synth_sizes'] if n),
           'R16 the graph evaluates the olympiad-only network, every pre-trained size, the fine-tuned and the reinforcement learners on the held-out synthetic theorems: %r' % (sorted(sy),))
    b = sy['eval/synth/baselines']; m5 = sy['eval/synth/pre_n500']
    expect('--solver' in b.argv and b.argv[b.argv.index('--solver') + 1] == 'blind' and '--model' not in b.argv and b.cores >= 1 and m5.argv.count('--model') == P_['seeds'] and 'pre/n500/s2' in ' '.join(m5.needs)
           and [m5.argv[i + 1] for i, x in enumerate(m5.argv) if x == '--solver'] == ['sample', 'ida'], 'R17 blind runs once without a network; a network is evaluated at every seed in one run, under the searches of the profile: %r' % (m5.argv[-12:],))
    expect([sy['eval/synth/init'].argv[i + 1] for i, x in enumerate(sy['eval/synth/init'].argv) if x == '--solver'] == ['ida'] and sy['eval/synth/init'].argv.count('--model') == P_['seeds'], 'R17b the untrained control runs under IDA* only, at every seed (sampling a random policy runs every theorem to the time limit)')
    pre_ = runs_['pre/n500/s0']; expect('--patience' in pre_.argv and pre_.argv[pre_.argv.index('--patience') + 1] == '2' and pre_.argv[pre_.argv.index('--epochs') + 1] == '14'
                                         and '--patience' not in SP.build('smoke')['pre/n100/s0'].argv, 'R18 pre-training stops early where the profile says so (and runs its epochs where it does not): %r' % (pre_.argv[-9:],))
    est = EST.estimate(runs_, P_, {'train_steps': 6476, 'dev': 20, 'test': 58, 'strict_extra': 0}, EST.RATES)
    est_off = EST.estimate(SP.build('small', overrides={'pre_patience': 0}), SP.profile('small', {'pre_patience': 0}), {'train_steps': 6476, 'dev': 20, 'test': 58, 'strict_extra': 0}, EST.RATES)
    expect(est['eval'][0] == len([r for r in runs_.values() if r.stage == 'eval']) and est['pre'][1] < est_off['pre'][1], 'R19 the estimate counts the synthetic evaluations, and early stopping lowers the pre-training estimate: %r %r' % (est['eval'][0], (est['pre'][1], est_off['pre'][1])))
    tmp = tempfile.mkdtemp(dir=os.environ.get('S51_TMP') or None)
    try:
        rng = np.random.default_rng(3)
        def put(variant, rows):
            d = os.path.join(tmp, 'eval', 'synth', variant); os.makedirs(d); open(os.path.join(d, 'rows.jsonl'), 'w').write('\n'.join(json.dumps(r) for r in rows) + '\n'); open(os.path.join(d, 'meta.json'), 'w').write('{}')
        names = {sp: ['%s_%d' % (sp, i) for i in range(40)] for sp in ('test_iid', 'test_comp')}
        def rows_of(method, model, spec, seed, p):
            return [dict(name=n, regime='library', split=sp, level=3, method=method, model=model, spec=spec, seed=seed, solved=bool(rng.random() < p), probes=50, seconds=0.1, steps=4) for sp in names for n in names[sp]]
        put('baselines', rows_of('blind', None, 'blind', 0, 0.5))
        put('pre_n500', [r for sd in (0, 1) for spec, p in (('ida', 0.8), ('sample', 0.3)) for r in rows_of('pre_n500+' + spec, 'pre_n500', spec, sd, p)])
        json.dump({'name': 'small', 'headline': 200, 'synth_eval_probes': 200, 'synth_sizes': [0, 500]}, open(os.path.join(tmp, 'PROFILE.json'), 'w')); json.dump({}, open(os.path.join(tmp, 'GRAPH.json'), 'w'))
        C = EC.load(tmp); T_ = TB.synth_table(C, C.profile)
        by = {(r[0], r[1]): r for r in T_.rows} if T_ else {}
        ida = by.get(('pre-trained on 500 synthetic theorems (no olympiad data)', 'ida')); smp = by.get(('pre-trained on 500 synthetic theorems (no olympiad data)', 'sample')); bl = by.get((TB.LABELS['blind'], 'blind'))
        expect(T_ is not None and bl is not None and ida is not None and smp is not None and ida[2] == 2 and T_.header[3:5] == ['test_iid', 'test_comp'] and ida[5].startswith('+') and smp[5].startswith('-') and bl[5] == '(reference)',
               'R20 the synthetic table has blind search as its reference and the gain (or loss) of every network and search: %r' % (T_.rows if T_ else None,))
        expect('synth' not in {r['group'] for r in C.rows} and 'synth' in {r['group'] for r in C.view('library').rows}, 'R21 the rows of the synthetic theorems are in the library regime, so the olympiad tables never see them')
    finally: shutil.rmtree(tmp, ignore_errors=True)
    def h7_fixture(init_thresholds, splits=('test_iid', 'test_comp')):
        """rows of the synthetic evaluation with known outcomes (a theorem is solved if its index is below the threshold; nested, so every gain is a superset): blind 20 of 40, the untrained network `init_thresholds` per seed, the networks of 500 and 1500 theorems"""
        tmp_ = tempfile.mkdtemp(dir=os.environ.get('S51_TMP') or None)
        names_ = {sp: ['%s_%d' % (sp, i) for i in range(40)] for sp in splits}
        def put_(variant, method, model, spec, thresholds, extra=0):
            d_ = os.path.join(tmp_, 'eval', 'synth', variant); os.makedirs(d_); open(os.path.join(d_, 'meta.json'), 'w').write('{}')
            rows_ = [dict(name=n, regime='library', split=sp, level=3, method=method, model=model, spec=spec, seed=sd, solved=i < t + extra, probes=50 if i < t else 1500, seconds=0.1, steps=4)
                     for sd, t in enumerate(thresholds) for sp in names_ for i, n in enumerate(names_[sp])]                          # (`extra`: theorems solved only far beyond the networks' budget, as blind search does)
            open(os.path.join(d_, 'rows.jsonl'), 'w').write('\n'.join(json.dumps(r) for r in rows_) + '\n')
        put_('baselines', 'blind', None, 'blind', [20], extra=5); put_('init', 'init+ida', 'init', 'ida', init_thresholds)
        put_('pre_n500', 'pre_n500+ida', 'pre_n500', 'ida', [26, 28, 30]); put_('pre_n1500', 'pre_n1500+ida', 'pre_n1500', 'ida', [32, 33, 34])
        json.dump({'name': 'small', 'headline': 200, 'synth_eval_probes': 200, 'synth_sizes': [0, 500, 1500], 'eval_budgets': [10, 20, 50, 100, 200], 'synth_eval_blind_mult': 10}, open(os.path.join(tmp_, 'PROFILE.json'), 'w')); json.dump({}, open(os.path.join(tmp_, 'GRAPH.json'), 'w'))
        return tmp_
    for thr, want, why in (([20, 22, 24], ['supported', 'supported'], 'a trained network that beats blind search and the untrained network'), ([32, 33, 34], ['supported', 'not distinguishable'], 'an untrained network that does as well as the trained one')):
        tmp = h7_fixture(thr)
        try:
            T_ = TB.h7_table(EC.load(tmp), EC.load(tmp).profile)
            rows_ = {(r[0].split(' minus ')[1].split(' (')[0], r[1]): r for r in T_.rows} if T_ else {}
            got = [rows_.get((k, 'test_iid'), [None] * 9)[8] for k in ('blind search', 'untrained network + ida')]
            exp_ = rows_.get(('pre_n500 + ida', 'test_comp'))
            ec_ = [r for r in (T_.rows if T_ else []) if 'equal compute' in r[0]]
            expect(len(ec_) == 2 and all(r[4].startswith('+20.0') for r in ec_) and 'with 2000 probes' in ec_[0][0] and T_ is not None and len(T_.rows) == 8 and got == want and all(rows_[(k, sp)][2] == 3 and rows_[(k, sp)][3] == 40 for k in ('blind search', 'untrained network + ida') for sp in ('test_iid', 'test_comp'))
                   and exp_ is not None and exp_[8] == 'exploratory' and exp_[6] == '--' and rows_[('blind search', 'test_iid')][4].startswith('+32.5'),
                   'R24 H7 (%s): the words of 3.5 for the gain over blind search and over the untrained network, the exploratory row without a verdict: %r' % (why, [(r[0], r[1], r[4], r[6], r[8]) for r in T_.rows] if T_ else None))
            if thr == [20, 22, 24]:                                                                               # (the figure of the same rows: one panel per split, a curve per search, its data in the csv)
                import csv as _csv
                FG_ = __import__('exp_figures'); C_ = EC.load(tmp); paths_ = FG_.fig_synth(C_, C_.profile, tmp, FG_._mpl())
                rows_f = list(_csv.DictReader(open(os.path.join(tmp, 'fig_synth.csv')))) if paths_ else []
                ser_ = {(r['panel'], r['series']) for r in rows_f if r['kind'] == 'line'}
                top_ = {r['x']: float(r['y']) for r in rows_f if r['kind'] == 'line' and r['series'] == 'pre-trained on 1500, IDA*' and r['panel'].startswith('test_iid')}
                bl_ = {r['x']: float(r['y']) for r in rows_f if r['kind'] == 'line' and r['series'] == 'blind search (to 2000 probes)' and r['panel'].startswith('test_iid')}
                expect(paths_ and all(os.path.getsize(q) > 1500 for q in paths_) and ('test_iid (40 theorems)', 'blind search (to 2000 probes)') in ser_ and ('test_comp (40 theorems)', 'untrained network, IDA*') in ser_ and ('test_iid (40 theorems)', 'pre-trained on 1500, sampled') not in ser_
                       and abs(top_.get('200', -1) - 0.825) < 1e-9 and top_.get('10') == 0.0 and abs(bl_.get('2000', -1) - 0.625) < 1e-9 and abs(bl_.get('200', -1) - 0.5) < 1e-9, 'R24b fig_synth draws a curve for each search of each split and keeps the numbers in its csv: %r %r' % (sorted(ser_), top_))
        finally: shutil.rmtree(tmp, ignore_errors=True)
    tmp = h7_fixture([20, 22, 24], splits=('dev',))                                                         # (a pilot evaluates the synthetic dev split only)
    try:
        C_ = EC.load(tmp); T1_ = TB.h7_table(C_, C_.profile); T2_ = TB.synth_table(C_, C_.profile)
        expect(T1_ is not None and {r[1] for r in T1_.rows} == {'dev'} and T2_ is not None and T2_.header[3:5] == ['dev', 'Δ vs blind, dev (pp)'], 'R25 a pilot that evaluates the synthetic dev split gets the same tables: %r' % (T1_.rows if T1_ else None,))
    finally: shutil.rmtree(tmp, ignore_errors=True)
    spec_ = importlib.util.spec_from_file_location('itp_run_gates', os.path.join(itp_paths.ROOT, 'tests', 'run_gates.py')); RG = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(RG)
    lim_ = [c for c in (list(RG.REG.GATE_CMDS) + list(RG.REG.BANK_GATE_CMDS) + list(RG.REG.ML_GATE_CMDS) + list(RG.REG.REPO_GATE_CMDS) + list(RG.REG.SYNTH_GATE_CMDS)) if RG.needs_rlimit_as(c)]
    expect(sorted(os.path.basename(c[0]) for c in lim_) == ['s19_boundary_tests.py', 's25_nring_tests.py', 's30_zlib_tests.py', 's31_linarith_tests.py', 's34_nt_tests.py', 's37_atoms_tests.py', 's40_actions_tests.py', 's50_proofs_tests.py'],
           'R26 the gates that need RLIMIT_AS are found from their source (a Mac skips them; docs/EXPERIMENTS.md lists them): %r' % (sorted(os.path.basename(c[0]) for c in lim_),))
    real_platform = sys.platform
    try:
        sys.platform = 'darwin'; so = sys.stdout; sys.stdout = io.StringIO()
        try: rc_ = RG.main(['s19_boundary_tests.py'])
        finally: out_ = sys.stdout.getvalue(); sys.stdout = so
    finally: sys.platform = real_platform
    expect(rc_ == 0 and 'SKIP' in out_ and 's19_boundary_tests.py' in out_ and '0/0 gates passed' in out_ and '1 skipped' in out_, 'R27 on a Mac a gate that needs RLIMIT_AS is skipped, said so and does not fail the run: %r' % (out_,))
    FGt_ = __import__('exp_figures')
    expect(FGt_.thin_ticks([10, 20, 30, 40, 60, 500, 600]) == [10, 20, 40, 60, 600] and FGt_.thin_ticks([10, 20, 50, 100, 200, 500, 1000, 2000]) == [10, 20, 50, 100, 200, 500, 1000, 2000] and FGt_.thin_ticks([]) == [] and FGt_.thin_ticks([7]) == [7],
           'R28 the labels of a log axis that would touch are thinned (the largest kept), the usual budgets are not: %r' % (FGt_.thin_ticks([10, 20, 30, 40, 60, 500, 600]),))
    fin_ = lambda n_hist, best, pat=2: {'history': [{}] * n_hist, 'best_epoch': best, 'args': {'patience': pat, 'epochs': 14}}
    expect([EC.pre_converged(f) for f in (fin_(6, 3), fin_(6, 4), fin_(14, 13), fin_(14, 12), fin_(14, 11), fin_(5, 2, 0), {})] == [True, False, False, False, True, None, None],
           'R22 a pre-training run has converged if its dev loss has not improved for `patience` epochs, has not if the epochs ran out sooner, and the question has no answer without early stopping')
    tmp = tempfile.mkdtemp(dir=os.environ.get('S51_TMP') or None)
    try:
        for n_, f_ in ((500, fin_(14, 13)), (1500, fin_(6, 3))):
            d_ = os.path.join(tmp, 'pre', 'n%d' % n_, 's0'); os.makedirs(d_); json.dump(f_, open(os.path.join(d_, 'final.json'), 'w'))
        json.dump({'name': 'small', 'headline': 200, 'synth_sizes': [0, 500, 1500], 'pre_patience': 2}, open(os.path.join(tmp, 'PROFILE.json'), 'w')); json.dump({}, open(os.path.join(tmp, 'GRAPH.json'), 'w'))
        w_ = [x for x in EC.load(tmp).warnings if 'had not converged' in x]
        expect(len(w_) == 1 and 'size 500' in w_[0] and 'pre/n500/s0' in w_[0] and 'H1' in w_[0], 'R23 the report warns for a size whose pre-training ran out of epochs while still improving, and for no other: %r' % (w_,))
    finally: shutil.rmtree(tmp, ignore_errors=True)
    tmp = tempfile.mkdtemp(dir=os.environ.get('S51_TMP') or None)
    try:
        so = sys.stdout; sys.stdout = open(os.devnull, 'w')
        try: rc = PF.main(['--quick', '--out', tmp])
        finally: sys.stdout.close(); sys.stdout = so
        rep = json.load(open(os.path.join(tmp, 'preflight.json')))
        expect(rc in (0, 1) and rep['machine']['cores_usable'] >= 1 and rep['checks'] and rep['recommended']['mem_gb'] > 0 and all(set(x) == {'ok', 'name', 'detail'} for x in rep['checks']),
               'R15 the quick pre-flight runs to the end and writes its record: %r' % (list(rep),))
        PF.REPORT.clear()
    finally: shutil.rmtree(tmp, ignore_errors=True)


SECTIONS = [('A', sec_A, False), ('B', sec_B, False), ('C', sec_C, False), ('D', sec_D, True), ('E', sec_E, True), ('F', sec_F, True), ('G', sec_G, True), ('H', sec_H, True), ('I', sec_I, False), ('J', sec_J, False), ('K', sec_K, True), ('L', sec_L, False), ('M', sec_M, False), ('N', sec_N, True), ('O', sec_O, False), ('P', sec_P, True), ('Q', sec_Q, False), ('R', sec_R, False)]


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
    if len(sys.argv) == 3 and sys.argv[1] == '--build-data':                # (the small dataset of sections D-H and P, built once into a directory that S51_DATA then names: a long run of mutants of the tools
        import dataset as DS                                                #  -- mutants/check_tools.py -- builds it once and not once per mutant)
        print('S51 small dataset in %s: %s' % (sys.argv[2], DS.build(sys.argv[2], only=set(SMOKE), lib_max=12))); sys.exit(0)
    sys.exit(main())
