"""exp_estimate.py -- how long will a profile take on this machine?

    python3 src/python/experiments/exp_estimate.py full                            # with the default rates (the reference machine: two slow cores, see RATES)
    python3 src/python/experiments/exp_estimate.py full --bench runs/full/bench/bench.json --cores 16 --seeds 0-4
    python3 src/python/experiments/exp_estimate.py full --train-rate 900 --set rl_iters=20

The estimate is arithmetic on the profile (exp_spec): the number of training steps of every run divided by a training rate (steps per second), the number of probes divided by a probe rate (per
solver), added up per stage.  The rates are MEASURED, not guessed: `scripts/06_benchmarks.sh` writes them to `RUNS/bench/bench.json` (`--bench`), and `--train-rate` / `--probe-rate` override them (for
example the training rate you measured with DEVICE=cuda).  The default rates are those of the machine the code was developed on.

What the estimate is NOT: an evaluation run that finds a proof stops early (so evaluation is an UPPER bound: every task costs min(budget / rate, time limit)); the steps per epoch of the olympiad data are read
from the dataset on disk; the cost of the world build (about a minute per process) and of the checks after a run are small and ignored.  Treat it as good to a factor of about two.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, argparse, collections
import exp_spec as SP, exp_run as RUN

# the reference machine (2 slow cores, no GPU): measured by scripts/06_benchmarks.sh on the smoke world; override with --bench / --train-rate / --probe-rate
RATES = {'train_steps_per_s': 55.0, 'probes_per_s': {'sample': 30.0, 'greedy': 30.0, 'ida': 18.0, 'mcts': 5.5, 'mw': 12.0, 'blind': 40.0, 'uniform': 60.0}}
MEAN_SYNTH_STEPS = 5.3                      # steps (actions) per synthetic theorem: the mean of the pilot's 6 024 theorems (5.26)
GEN_ATTEMPTS_PER_S = 12.0                   # generator attempts per second per core, world builds included: the pilot's 30 000 attempts took 2 473 core-seconds


def dataset_counts(ddir):
    """{'train_steps': actions of the bank's train proofs (both regimes) + the library lemmas' train proofs, 'dev': n, 'test': n, 'strict_extra': tasks that exist in the strict regime only} from the dataset on disk (None: unknown)"""
    try:
        bank = json.load(open(os.path.join(ddir, 'bank.json')))
    except (OSError, ValueError): return None
    steps = 0; n = collections.Counter(); strict = 0
    for p in bank['problems']:
        n[p['split']] += 1
        if p['split'] == 'train':
            steps += (p.get('demonstration') or {}).get('length', 0) + (p.get('demonstration_strict') or {}).get('length', 0)
        if p.get('helpers') and p.get('demonstration_strict'): strict += 1
    try:
        lib = json.load(open(os.path.join(ddir, 'library.json')))
        steps += sum(l.get('length', 0) for l in lib['lemmas'] if l.get('split') == 'train')
    except (OSError, ValueError, KeyError): pass
    return {'train_steps': steps, 'dev': n['dev'], 'test': n['test'], 'train': n['train'], 'strict_extra': strict}


def rates_from_bench(path):
    r = json.loads(json.dumps(RATES))
    try: b = json.load(open(path))
    except (OSError, ValueError): return r, False
    for s, v in (b.get('solvers') or {}).items():
        if v.get('probes_per_s'): r['probes_per_s'][s] = float(v['probes_per_s'])
    if (b.get('world') or {}).get('train_steps_per_s'): r['train_steps_per_s'] = float(b['world']['train_steps_per_s'])
    return r, True


def estimate(runs, P, counts, rates, train_rate=None, probe_rate=None, synth_mean=MEAN_SYNTH_STEPS):
    """{stage: [runs, seconds of wall-clock summed over runs, core-seconds]}: a run's wall-clock is for the run alone, its cores are what it occupies (a training run: one process, the GPU when there is one)"""
    tr = train_rate or rates['train_steps_per_s']
    pr = lambda solver: (probe_rate or rates['probes_per_s'].get(solver) or rates['probes_per_s']['sample'])
    steps_ft = (counts or {}).get('train_steps') or 8000
    n_dev = (counts or {}).get('dev') or 30; n_test = (counts or {}).get('test') or 55; n_eval = n_dev + n_test + ((counts or {}).get('strict_extra') or 0) // 3
    out = collections.OrderedDict()
    def add(stage, secs, cores=1):
        e = out.setdefault(stage, [0, 0.0, 0.0]); e[0] += 1; e[1] += secs; e[2] += secs * cores
    W = max(1, int(P['workers']))
    for r in runs.values():
        t = r.tags; st = r.stage
        if st == 'data':
            if r.id == 'data/synth_gen': add('data', P['synth_attempts'] / GEN_ATTEMPTS_PER_S / P['synth_jobs'], P['synth_jobs'])           # the first round of `grow`; it adds rounds when the train split falls short of its target
            elif r.id in ('data/synth_verify',): add('data', 60.0 + 0.003 * P['synth_target'])
            elif r.id == 'data/dataset': add('data', 1400.0 if t.get('verify_only') else 110.0)               # (measured: replaying the released dataset takes 1 362 s, building the smoke dataset 108 s)
            else: add('data', 60.0)
        elif st == 'pre': add('pre', P['pre_epochs'] * t['n_synth'] * synth_mean / tr)
        elif st == 'ft':
            epochs = P['ft_epochs'] if '--init' in r.argv else P['scratch_epochs']
            secs = epochs * steps_ft / tr
            if P['bc_eval_every']: secs += (epochs // max(1, P['bc_eval_every'])) * n_dev * P['bc_eval_probes'] / pr('sample') / W
            add('ft', secs)
        elif st == 'rl':
            v = r.id.split('/')[1]
            if t.get('algo') == 'ppo':
                eps = P['ppo_episodes'] * 100.0 / pr('sample') / W                                      # an episode is at most about a hundred probes
                add('rl', P['ppo_iters'] * (eps + 8 * 64 / tr) + (P['ppo_iters'] // max(1, P['rl_eval_every'])) * n_dev * P['rl_eval_probes'] / pr('sample') / W, W)
            else:
                actor = 'sample'
                a = SP.RL_VARIANTS[v]['args']
                if '--actor' in a: actor = a[a.index('--actor') + 1]
                per_iter = P['rl_episodes'] * P['rl_probes'] / pr(actor) / W + P['ei_updates'] * 64 / tr
                add('rl', P['rl_iters'] * per_iter + (P['rl_iters'] // max(1, P['rl_eval_every'])) * n_dev * P['rl_eval_probes'] / pr('sample') / W, W)
        elif st == 'eval':
            solvers = t.get('solvers') or ['sample']; secs = 0.0
            for s in solvers:
                base = s.split(':')[0]
                per = min(P['eval_probes'] / pr(base), P['eval_time_limit'])
                secs += n_eval * per * max(1, len(t.get('seeds') or [0])) / W
            add('eval', secs, W)
        elif st == 'bench': add('bench', 1800.0)
        elif st == 'report': add('report', 120.0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split('\n', 2)[2])
    ap.add_argument('profile', choices=sorted(SP.PROFILES)); ap.add_argument('--seeds', default=''); ap.add_argument('--set', action='append', default=[]); ap.add_argument('--bench', default='')
    ap.add_argument('--train-rate', type=float, default=0.0, help='training steps per second (default: bench.json or the reference machine)'); ap.add_argument('--probe-rate', type=float, default=0.0, help='probes per second for every solver')
    ap.add_argument('--cores', type=int, default=8, help='cores of the machine, for the wall-clock'); ap.add_argument('--dataset', default='')
    a = ap.parse_args(argv)
    over = SP.parse_overrides(a.set); seeds = RUN.parse_seeds(a.seeds) if a.seeds else None
    P = SP.profile(a.profile, over); runs = SP.build(a.profile, seeds=seeds, runs_dir='runs/' + a.profile, overrides=over)
    rates, from_bench = rates_from_bench(a.bench) if a.bench else (RATES, False)
    ddir = a.dataset or (P['dataset'] if P['dataset'] else '')
    counts = dataset_counts(os.path.join(itp_paths.ROOT, ddir)) if ddir else {'train_steps': 2000, 'dev': 3, 'test': 5, 'strict_extra': 0}          # (the smoke world: 17 problems)
    est = estimate(runs, P, counts, rates, a.train_rate or None, a.probe_rate or None)
    print('profile %s, %d seeds, %d runs; rates: %.0f training steps/s%s, probes/s %s (%s)' % (a.profile, len(seeds) if seeds else P['seeds'], len(runs), a.train_rate or rates['train_steps_per_s'], '' if not a.train_rate else ' (given)',
          ', '.join('%s %.0f' % kv for kv in sorted(rates['probes_per_s'].items())) if not a.probe_rate else '%.0f for all' % a.probe_rate, 'measured on this machine: ' + a.bench if from_bench else 'reference machine'))
    if counts: print('dataset: %d train steps per epoch of fine-tuning, %d dev + %d test problems' % (counts['train_steps'], counts['dev'], counts['test']))
    print('\n%-8s %6s %12s %12s %16s' % ('stage', 'runs', 'run-hours', 'core-hours', 'wall-clock, %d cores' % a.cores))
    tr_h = tc_h = tw = 0.0
    for stage, (n, secs, csecs) in est.items():
        rh, ch = secs / 3600.0, csecs / 3600.0; wall = max(ch / a.cores, (secs / n) / 3600.0)
        tr_h += rh; tc_h += ch; tw += wall
        print('%-8s %6d %12.1f %12.1f %16.1f' % (stage, n, rh, ch, wall))
    print('%-8s %6s %12.1f %12.1f %16.1f' % ('total', '', tr_h, tc_h, tw))
    print('\nrun-hours: the wall-clock of the runs added up, each run alone; core-hours: what they occupy (a training run one core, a search run `workers` = %d cores); wall-clock: core-hours / cores with perfect packing, and\n'
          'at least one run\'s length; stages overlap in run_all.sh, so the true total is a little lower.  Training on a GPU: pass --train-rate with the rate of scripts/06_benchmarks.sh (DEVICE=cuda); runs that share one GPU slow each other down.' % P['workers'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
