"""exp_spec.py -- the experiments of the paper as DATA: profiles, runs, their dependencies and their commands.

    runs = build('small', seeds=[0, 1, 2], runs_dir='runs/small')          # {id: Run}; nothing is executed here
    runs['rl/ei_her/s0'].argv                                               # the command, as a list; the working directory is the repository root

A PROFILE sets the sizes (seeds, synthetic theorems, epochs, iterations, budgets):

    smoke    17 problems, ~300 synthetic theorems, 2 seeds: ~1-2 hours on 2 cores.  It checks that the whole pipeline runs and that every table and figure is made; its numbers mean nothing.
    small    the whole bank, 5 000 synthetic theorems, 3 seeds: the first real numbers; the shape of every result.
    full     the paper: 60 000 synthetic theorems, 5 seeds (`--seeds 0-6` for 7), long budgets.

Any field of a profile can be changed on the command line (`exp_run.py ... --set rl_iters=20 --set eval_probes=500`); the effective profile is written to `PROFILE.json` in the runs directory and into the report.

A RUN is a node of a graph: an id (its directory under the runs directory, `rl/ei_her/s0`), a stage, the command that makes it, the runs it needs, the files that must exist when it has succeeded (`check`) and a
rough memory estimate.  `exp_run.py` executes the graph (resumably: a run that succeeded is not run again); `exp_report.py` reads the results through the same names.  The METHODS compared (the `variant`) are:

    pre            behaviour cloning on synthetic theorems only (a library-only world): zero-shot on the bank, no olympiad data
    bc_scratch     behaviour cloning on the bank's train split (and the library lemmas) from random weights: no synthetic pre-training
    bc             pre-trained on all the synthetic theorems, then behaviour cloning on the bank's train split: the starting point of every learner below
    bc_n<N>        the same with only N synthetic theorems of pre-training (the scaling study)
    bc_mw          bc with the manager-worker masks and labels
    ppo            proximal policy optimisation from bc (goals: train problems, sub-theorems, synthetic theorems)
    ppo_nosub      ppo without the sub-theorem goals (hindsight ablation)
    ei             expert iteration from bc: sample, keep the kernel-checked proofs, train on them (no hindsight)
    ei_her         + hindsight: the theorems an attempt proved on the way are goals and training data
    ei_her_boot    + bootstrapping: hindsight theorems enter the library as lemmas the agent can use (kernel-checked, train-side only)
    ei_mw          the manager-worker agent as the actor (from bc_mw)
    ei_mcts        MCTS as the actor           ei_ida   IDA* on the policy's costs as the actor

and the SOLVERS (ml_agents): blind, uniform (no model), sample, greedy, ida, mcts, mw.  The main comparison evaluates every model with the same solver (`sample`) on the dev and test splits.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, collections, ast

PY = 'python3'
ROOT = itp_paths.ROOT
FULL_DATASET = os.path.join('data', 'dataset_v2')

# the problems of the smoke world: a few of each split (bank_split), short reference proofs; the bank adds what they depend on
SMOKE_PROBLEMS = ('ident_cube_sum', 'ident_three_cubes', 'ident_cube_diff', 'ineq_am_gm_two', 'ineq_cauchy_two', 'ineq_four_ab', 'imo1959_p1', 'pow_sub_dvd', 'ineq_cauchy_three',        # train
                  'ident_diff_squares', 'not_dvd_between', 'two_pow_gt_n', 'find_all_x2_eq_x',                                                                                      # dev
                  'ident_brahmagupta', 'coprime_consecutive', 'ineq_cross_terms', 'n4_n2_1_composite')                                                                              # test

PROFILES = {
    'smoke': dict(
        seeds=2, dataset='', smoke_lib_max=60,
        synth_attempts=1800, synth_target=300, synth_jobs=1, synth_sizes=(0, 100, 300), synth_verify_sample=1, synth_seed=1,
        pre_epochs=2, pre_batch=32, ft_epochs=3, ft_lr=3e-4, scratch_epochs=3, bc_eval_every=0, bc_eval_probes=30,
        rl_iters=2, rl_episodes=8, rl_probes=40, ppo_iters=3, ppo_episodes=8, ei_updates=5, rl_eval_every=1, rl_eval_probes=30, boot_per_iter=4,
        eval_probes=60, headline=60, eval_time_limit=5.0, eval_budgets=(10, 20, 30, 40, 60), eval_splits='dev,test', recheck=1, workers=0, device='cpu', threads=1, mem_scale=0.4),
    'small': dict(
        seeds=3, dataset=FULL_DATASET,
        synth_attempts=40000, synth_target=5000, synth_jobs=2, synth_sizes=(0, 500, 1500, 5000), synth_verify_sample=1, synth_seed=1,
        pre_epochs=6, pre_batch=64, ft_epochs=12, ft_lr=3e-4, scratch_epochs=20, bc_eval_every=4, bc_eval_probes=100,
        rl_iters=10, rl_episodes=48, rl_probes=100, ppo_iters=60, ppo_episodes=32, ei_updates=40, rl_eval_every=2, rl_eval_probes=200, boot_per_iter=8,
        eval_probes=500, headline=200, eval_time_limit=30.0, eval_budgets=(10, 20, 50, 100, 200, 500), eval_splits='dev,test', recheck=1, workers=2, device='cpu', threads=1, mem_scale=1.0),
    'full': dict(
        seeds=5, dataset=FULL_DATASET,
        synth_attempts=400000, synth_target=60000, synth_jobs=4, synth_sizes=(0, 1000, 4000, 15000, 60000), synth_verify_sample=1, synth_seed=1,
        pre_epochs=6, pre_batch=64, ft_epochs=20, ft_lr=3e-4, scratch_epochs=30, bc_eval_every=5, bc_eval_probes=200,
        rl_iters=30, rl_episodes=128, rl_probes=200, ppo_iters=250, ppo_episodes=64, ei_updates=60, rl_eval_every=3, rl_eval_probes=500, boot_per_iter=16,
        eval_probes=2000, headline=1000, eval_time_limit=60.0, eval_budgets=(10, 20, 50, 100, 200, 500, 1000, 2000), eval_splits='dev,test', recheck=1, workers=4, device='cpu', threads=1, mem_scale=1.0),
}

# the learners that continue from a behaviour-cloned network: variant -> how
RL_VARIANTS = collections.OrderedDict([
    ('ppo',         dict(algo='ppo', init='bc',    args=['--p-sub', '0.25', '--p-synth', '0.1'])),
    ('ppo_nosub',   dict(algo='ppo', init='bc',    args=['--p-sub', '0', '--p-synth', '0.1'])),
    ('ei',          dict(algo='ei',  init='bc',    args=['--actor', 'sample', '--p-sub', '0', '--p-synth', '0.1'])),
    ('ei_her',      dict(algo='ei',  init='bc',    args=['--actor', 'sample', '--her', '--p-sub', '0.25', '--p-synth', '0.1'])),
    ('ei_her_boot', dict(algo='ei',  init='bc',    args=['--actor', 'sample', '--her', '--bootstrap', '--p-sub', '0.25', '--p-synth', '0.1'], library=True)),
    ('ei_mw',       dict(algo='ei',  init='bc_mw', args=['--actor', 'mw', '--her', '--p-sub', '0.25', '--p-synth', '0.1'])),
    ('ei_mcts',     dict(algo='ei',  init='bc',    args=['--actor', 'mcts', '--her', '--p-sub', '0.25', '--p-synth', '0.1'])),
    ('ei_ida',      dict(algo='ei',  init='bc',    args=['--actor', 'ida', '--her', '--p-sub', '0.25', '--p-synth', '0.1'])),
])

MAIN_SOLVER = 'sample'
SOLVER_STUDY = (('bc', 'ei_her'), ('sample', 'greedy', 'ida', 'mcts', 'mcts:value=critic'))          # the same networks under different provers
MW_STUDY = (('bc_mw', 'ei_mw'), ('mw', 'mw:gate=never', 'mw:gate=fixed', 'sample'))                  # the manager-worker agent and its ablations
BASELINES = ('blind', 'uniform')                                                                        # no network

# rough peak memory of one process in GB, by stage (the world of the whole bank is ~2.5 GB; a Step is ~8 KB)
MEM_GB = {'data': 3.0, 'pre': 3.5, 'ft': 3.5, 'rl': 4.0, 'eval': 3.5, 'bench': 3.5, 'report': 1.0}


class Run:
    __slots__ = ('id', 'stage', 'argv', 'needs', 'check', 'tags', 'mem_gb', 'cores')
    def __init__(self, id, stage, argv, needs=(), check=(), tags=None, mem_gb=None, cores=1):
        self.id = id; self.stage = stage; self.argv = [str(x) for x in argv]; self.needs = tuple(needs); self.check = tuple(check); self.tags = dict(tags or {}); self.mem_gb = MEM_GB[stage] if mem_gb is None else mem_gb
        self.cores = max(1, int(cores))
    def to_dict(self): return {'id': self.id, 'stage': self.stage, 'argv': self.argv, 'needs': list(self.needs), 'check': list(self.check), 'tags': self.tags, 'mem_gb': self.mem_gb, 'cores': self.cores}
    def __repr__(self): return 'Run(%s)' % self.id


def script(role, module):
    return os.path.join('src', 'python', role, module + '.py')


def parse_overrides(items):
    """['rl_iters=20', 'synth_sizes=(0,100)'] -> {'rl_iters': 20, 'synth_sizes': (0, 100)} (Python literals; anything else is a string)"""
    out = {}
    for it in items or ():
        k, eq, v = it.partition('=')
        if not k or not eq: raise ValueError('--set wants key=value, got %r' % it)
        try: out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError): out[k] = v
    return out


def profile(name, overrides=None):
    for k in overrides or {}:
        if k not in PROFILES[name]: raise ValueError('%r is not a field of the profile (%s)' % (k, ', '.join(sorted(PROFILES[name]))))
    P = dict(PROFILES[name]); P.update(overrides or {})
    P['name'] = name
    return P


def build(name, seeds=None, runs_dir=None, overrides=None):
    """the graph of runs ({id: Run}, in an order that respects the dependencies) for a profile; `seeds`: a list of ints (default: range(profile['seeds'])); `overrides`: profile fields to change"""
    P = profile(name, overrides)
    seeds = list(range(P['seeds'])) if seeds is None else list(seeds)
    R = runs_dir or os.path.join('runs', name)
    smoke = not P['dataset']
    D = os.path.join(R, 'data') if smoke else P['dataset']
    SY = os.path.join(R, 'synth')
    runs = collections.OrderedDict()
    def add(run): run.mem_gb = round(run.mem_gb * P['mem_scale'], 2); runs[run.id] = run; return run
    W_ = str(P['workers']); NC = max(1, int(P['workers'])); dev = ['--device', P['device'], '--threads', P['threads']]
    py = lambda role, module, *a: [PY, script(role, module)] + [str(x) for x in a]
    out = lambda rid: os.path.join(R, rid)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ data
    if smoke:
        add(Run('data/dataset', 'data', py('problems', 'dataset', D, '--only', ','.join(SMOKE_PROBLEMS), '--lib-max', P['smoke_lib_max']), check=[os.path.join(D, 'manifest.json')]))
    else:
        add(Run('data/dataset', 'data', py('problems', 'dataset', '--verify', D), check=[os.path.join(D, 'manifest.json')], tags={'verify_only': True}))
    ds = 'data/dataset'
    add(Run('data/synth_gen', 'data', py('synth', 'synth_cli', 'gen', '--out', SY, '--seed', P['synth_seed'], '--attempts', P['synth_attempts'], '--shard', 2000 if P['synth_attempts'] >= 8000 else 300,
                                          '--jobs', P['synth_jobs']), needs=[ds], check=[os.path.join(SY, 'config.json')], cores=P['synth_jobs']))
    add(Run('data/synth_merge', 'data', py('synth', 'synth_cli', 'merge', '--out', SY, '--target', P['synth_target'], '--exclude-dataset', D), needs=['data/synth_gen'], check=[os.path.join(SY, 'manifest.json')]))
    add(Run('data/synth_verify', 'data', py('synth', 'synth_cli', 'verify', '--out', SY, '--sample', P['synth_verify_sample']), needs=['data/synth_merge'], check=[os.path.join(SY, 'verification.json')]))
    add(Run('data/synth_card', 'data', py('synth', 'synth_cli', 'card', '--out', SY), needs=['data/synth_verify'], check=[os.path.join(SY, 'DATASET_CARD.md')]))
    add(Run('data/synth_stats', 'data', py('synth', 'synth_stats', SY, '--out', os.path.join(SY, 'stats.json')), needs=['data/synth_merge'], check=[os.path.join(SY, 'stats.json')], mem_gb=1.0))

    smax = max(P['synth_sizes'])
    sizes = sorted(set(P['synth_sizes']) - {0})

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ behaviour cloning
    def pre(n, seed):
        rid = 'pre/n%d/s%d' % (n, seed)
        a = py('learner', 'ml_bc', D, '--out', out(rid), '--synth', SY, '--synth-max', n, '--no-bank', '--no-lib', '--epochs', P['pre_epochs'], '--lr', 1e-3, '--seed', seed, '--batch', P['pre_batch'],
               '--workers', W_, *dev, '--resume')
        return add(Run(rid, 'pre', a, needs=['data/synth_merge'], check=[out(rid) + '/final.json', out(rid) + '/best.pt'], tags={'method': 'pre', 'n_synth': n, 'seed': seed},
                       mem_gb=MEM_GB['pre'] + 0.05 * n / 1000.0, cores=NC))

    def ft(variant, seed, init=None, extra=(), tags=None):
        rid = 'ft/%s/s%d' % (variant, seed)
        ev = ['--eval-every', P['bc_eval_every'], '--eval-probes', P['bc_eval_probes'], '--eval-time-limit', P['eval_time_limit']] if P['bc_eval_every'] else []
        a = py('learner', 'ml_bc', D, '--out', out(rid), '--epochs', P['ft_epochs'] if init else P['scratch_epochs'], '--lr', P['ft_lr'] if init else 1e-3, '--seed', seed, '--batch', P['pre_batch'],
               '--workers', W_, *dev) + ev + list(extra) + ['--resume']
        needs = [ds]
        if init: a += ['--init', out(init) + '/best.pt']; needs = [init]
        return add(Run(rid, 'ft', a, needs=needs, check=[out(rid) + '/final.json', out(rid) + '/best.pt'], tags=dict({'method': variant, 'seed': seed}, **(tags or {})), cores=NC))

    for seed in seeds:
        for n in sizes: pre(n, seed)
        top = 'pre/n%d/s%d' % (smax, seed)
        ft('bc', seed, init=top, tags={'n_synth': smax})
        ft('bc_scratch', seed, tags={'n_synth': 0})
        ft('bc_mw', seed, init=top, extra=['--mw'], tags={'n_synth': smax})
        for n in sizes:
            if n != smax: ft('bc_n%d' % n, seed, init='pre/n%d/s%d' % (n, seed), tags={'n_synth': n})

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ reinforcement learning
    for variant, v in RL_VARIANTS.items():
        for seed in seeds:
            rid = 'rl/%s/s%d' % (variant, seed); init = 'ft/%s/s%d' % (v['init'], seed)
            common = [D, '--init', out(init) + '/best.pt', '--out', out(rid), '--seed', seed, '--workers', W_, *dev, '--synth', SY, '--synth-max', P['synth_target'], '--eval-every', P['rl_eval_every'],
                      '--eval-probes', P['rl_eval_probes'], '--resume']
            if v['algo'] == 'ppo':
                a = py('learner', 'ml_ppo', *common, '--iters', P['ppo_iters'], '--episodes', P['ppo_episodes']) + v['args']
            else:
                a = py('learner', 'ml_ei', *common, '--iters', P['rl_iters'], '--episodes', P['rl_episodes'], '--probes', P['rl_probes'], '--updates', P['ei_updates'], '--time-limit', P['eval_time_limit'],
                       '--boot-per-iter', P['boot_per_iter'], '--eval-time-limit', P['eval_time_limit']) + v['args']
            add(Run(rid, 'rl', a, needs=[init, 'data/synth_merge'], check=[out(rid) + '/final.json', out(rid) + '/last.pt'],
                    tags={'method': variant, 'seed': seed, 'algo': v['algo'], 'library': bool(v.get('library'))}, mem_gb=MEM_GB['rl'] + 0.4 * int(W_), cores=NC))

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ evaluation
    def ckpt(variant, seed):
        """(path of the checkpoint to evaluate, id of the run that makes it): the best epoch of a behaviour-cloned model, the last iteration of a reinforcement learner (not selected on dev)"""
        if variant == 'pre': return out('pre/n%d/s%d' % (smax, seed)) + '/best.pt', 'pre/n%d/s%d' % (smax, seed)
        if variant in RL_VARIANTS: return out('rl/%s/s%d' % (variant, seed)) + '/last.pt', 'rl/%s/s%d' % (variant, seed)
        return out('ft/%s/s%d' % (variant, seed)) + '/best.pt', 'ft/%s/s%d' % (variant, seed)

    def evaluation(rid, group, variants, solvers, per_seed=False, lemmas=False):
        """one evaluation run per variant (all the seeds in one process: they share the world) or, with `per_seed` (a library of bootstrapped lemmas per run), per variant and seed"""
        for v in variants:
            for ss in ([[s] for s in seeds] if per_seed else [seeds]):
                r_id = '%s/%s%s' % (rid, v, ('/s%d' % ss[0]) if per_seed else '')
                a = py('learner', 'ml_evaluate', D, '--out', out(r_id), '--split', P['eval_splits'], '--probes', P['eval_probes'], '--time-limit', P['eval_time_limit'], '--workers', W_, '--threads', P['threads'])
                for s in solvers: a += ['--solver', s]
                needs = [ds]
                for s in ss:
                    path, need = ckpt(v, s); a += ['--model', '%s=%s' % (v, path)]; needs.append(need)
                if lemmas: a += ['--lemmas', 'auto']
                if P['recheck']: a += ['--recheck']
                add(Run(r_id, 'eval', a, needs=needs, check=[out(r_id) + '/rows.jsonl', out(r_id) + '/meta.json'], tags={'group': group, 'variant': v, 'seeds': ss, 'solvers': list(solvers)}, mem_gb=MEM_GB['eval'] + 0.4 * int(W_), cores=NC))

    boot = [v for v, d in RL_VARIANTS.items() if d.get('library')]
    main_variants = ['pre', 'bc_scratch', 'bc'] + [v for v in RL_VARIANTS if v not in boot]
    evaluation('eval/main', 'main', main_variants, [MAIN_SOLVER])
    evaluation('eval/main', 'main', boot, [MAIN_SOLVER], per_seed=True, lemmas=True)
    evaluation('eval/solvers', 'solvers', SOLVER_STUDY[0], SOLVER_STUDY[1])
    evaluation('eval/mw', 'mw', MW_STUDY[0], MW_STUDY[1])
    scale = ['bc_n%d' % n for n in sizes if n != smax]
    if scale: evaluation('eval/scale', 'scale', scale, [MAIN_SOLVER])
    a = py('learner', 'ml_evaluate', D, '--out', out('eval/baselines'), '--split', P['eval_splits'], '--probes', P['eval_probes'], '--time-limit', P['eval_time_limit'], '--workers', W_, '--threads', P['threads'])
    for s in BASELINES: a += ['--solver', s]
    add(Run('eval/baselines', 'eval', a, needs=[ds], check=[out('eval/baselines') + '/rows.jsonl'], tags={'group': 'baselines', 'variant': 'baselines', 'solvers': list(BASELINES)}, mem_gb=MEM_GB['eval'] + 0.4 * int(W_), cores=NC))

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ throughput, report
    add(Run('bench/throughput', 'bench', py('experiments', 'exp_bench', D, '--out', out('bench'), '--model', out('ft/bc/s%d' % seeds[0]) + '/best.pt', '--synth', SY,
                                              '--workers', ','.join(str(x) for x in sorted({1, 2, max(1, int(W_))}))), needs=['ft/bc/s%d' % seeds[0], 'data/synth_merge'], check=[out('bench') + '/bench.json']))
    add(Run('bench/kernel', 'bench', [PY, os.path.join('tests', 'bench', 'bench_kernel.py'), '--json', out('bench') + '/kernel.json'], needs=[], check=[out('bench') + '/kernel.json'], mem_gb=1.0))
    add(Run('report', 'report', py('experiments', 'exp_report', R, '--out', out('report')), needs=[], check=[out('report') + '/report.md']))
    return runs


def stages(runs):
    return list(collections.OrderedDict((r.stage, None) for r in runs.values()))


def closure(runs, ids):
    """the ids and everything they need (in graph order)"""
    want = set(); stack = list(ids)
    while stack:
        i = stack.pop()
        if i in want: continue
        if i not in runs: raise KeyError('no run %r' % i)
        want.add(i); stack += list(runs[i].needs)
    return [i for i in runs if i in want]


def effective_profile(name, overrides=None, seeds=None):
    """the profile as it is run (JSON-friendly)"""
    P = profile(name, overrides); P['seed_list'] = list(range(P['seeds'])) if seeds is None else list(seeds)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='print the run graph of a profile')
    ap.add_argument('profile', choices=sorted(PROFILES)); ap.add_argument('--seeds', default=''); ap.add_argument('--runs', default=''); ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    rs = build(a.profile, runs_dir=a.runs or None, overrides=parse_overrides(a.set))
    for r in rs.values(): print('%-34s %-7s needs %-40s %s' % (r.id, r.stage, ','.join(r.needs) or '-', ' '.join(r.argv[:3])))
    print(len(rs), 'runs')
