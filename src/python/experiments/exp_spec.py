"""exp_spec.py -- the experiments of the paper as DATA: profiles, runs, their dependencies and their commands.

    runs = build('small', seeds=[0, 1, 2], runs_dir='runs/small')          # {id: Run}; nothing is executed here
    runs['rl/ei_her/s0'].argv                                               # the command, as a list; the working directory is the repository root

A PROFILE sets the sizes (seeds, synthetic theorems, epochs, iterations, budgets):

    smoke    17 problems (19 with what they depend on), ~300 synthetic theorems, 2 seeds: about 70 minutes of run time on 2 cores.  It checks that the whole pipeline runs and that every table and figure is made; its numbers mean nothing.
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

and the STUDIES that vary the network or the start (each is a profile field, empty = not run: `--set 'archs=()'`, quoted: the shell reads the brackets):

    bc_<arch>      the same learner with another ENCODER of the goal (ml_model.ARCHS: bigru, cnn, bow; the transformer is `bc`): pre-trained on all the synthetic theorems with it, then fine-tuned on the bank
    ei_her_<arch>  ... and the learners of `arch_learners` continued from it (ppo_<arch>, ei_her_<arch>)                                                     [fields: archs, arch_learners]
    bc_w<d>        the transformer at another WIDTH (64, 256, ...; layers and heads grow with it, `size_of`): the model-size slice of the scaling study     [fields: widths, width_learners]
    scratch_<learner>[_<arch>]   reinforcement learning from RANDOM WEIGHTS: no demonstration, no pre-training; the goals are mostly synthetic theorems, the learner runs `scratch_mult` times as
                   many iterations as when it continues from a trained network                                                                            [fields: scratch ('ei_her', 'ppo', 'ei_her@bigru': learner[@arch]), scratch_mult]

and the SOLVERS (ml_agents): blind, uniform (no model), sample, greedy, ida, mcts, mw.  The main comparison evaluates every model with the same solver (`sample`) on the dev and test splits.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, re, json, collections, ast

PY = 'python3'
ROOT = itp_paths.ROOT
FULL_DATASET = os.path.join('data', 'dataset_v2')

# the problems of the smoke world: a few of each split (bank_split), short reference proofs; the bank adds what they depend on
SMOKE_PROBLEMS = ('ident_cube_sum', 'ident_three_cubes', 'ident_cube_diff', 'ineq_am_gm_two', 'ineq_cauchy_two', 'ineq_four_ab', 'imo1959_p1', 'pow_sub_dvd', 'ineq_cauchy_three',        # train
                  'ident_diff_squares', 'bernoulli_int', 'two_pow_gt_n', 'find_all_x2_eq_x',                                                                                      # dev
                  'ident_brahmagupta', 'coprime_consecutive', 'ineq_cross_terms', 'n4_n2_1_composite')                                                                              # test

PROFILES = {
    'smoke': dict(
        seeds=2, dataset='', smoke_lib_max=60,
        synth_attempts=1800, synth_target=300, synth_jobs=1, synth_sizes=(0, 100, 300), synth_verify_sample=1, synth_seed=1,
        pre_epochs=2, pre_patience=0, pre_batch=32, ft_epochs=3, ft_lr=3e-4, scratch_epochs=3, bc_eval_every=0, bc_eval_probes=30,
        rl_iters=2, rl_episodes=8, rl_probes=40, ppo_iters=3, ppo_episodes=8, ei_updates=5, rl_eval_every=1, rl_eval_probes=30, boot_per_iter=4,
        archs=('bigru',), arch_learners=('bc', 'ei_her'), widths=(64,), width_learners=('bc',), scratch=('ei_her',), scratch_mult=1,
        eval_probes=60, headline=60, eval_time_limit=5.0, eval_budgets=(10, 20, 30, 40, 60), eval_splits='dev,test', synth_eval_n=12, synth_eval_probes=60, synth_eval_blind_mult=10, synth_eval_splits='test_iid,test_comp', synth_eval_solvers=('blind', 'sample', 'ida'), recheck=1, workers=0, device='cpu', threads=1, mem_scale=0.4),
    'small': dict(
        seeds=3, dataset=FULL_DATASET,
        synth_attempts=40000, synth_target=5000, synth_jobs=2, synth_sizes=(0, 500, 1500, 5000), synth_verify_sample=1, synth_seed=1,
        pre_epochs=14, pre_patience=2, pre_batch=64, ft_epochs=12, ft_lr=3e-4, scratch_epochs=20, bc_eval_every=4, bc_eval_probes=100,
        rl_iters=10, rl_episodes=48, rl_probes=100, ppo_iters=60, ppo_episodes=32, ei_updates=40, rl_eval_every=2, rl_eval_probes=200, boot_per_iter=8,
        archs=('bigru', 'cnn', 'bow'), arch_learners=('bc', 'ei_her'), widths=(64, 256), width_learners=('bc',), scratch=('ei_her', 'ppo'), scratch_mult=3,
        eval_probes=500, headline=200, eval_time_limit=30.0, eval_budgets=(10, 20, 50, 100, 200, 500), eval_splits='dev,test', synth_eval_n=150, synth_eval_probes=200, synth_eval_blind_mult=10, synth_eval_splits='test_iid,test_comp', synth_eval_solvers=('blind', 'sample', 'ida'), recheck=1, workers=2, device='cpu', threads=1, mem_scale=1.0),
    'full': dict(
        seeds=5, dataset=FULL_DATASET,
        synth_attempts=450000, synth_target=60000, synth_jobs=4, synth_sizes=(0, 1000, 4000, 15000, 60000), synth_verify_sample=1, synth_seed=1,
        pre_epochs=14, pre_patience=2, pre_batch=64, ft_epochs=20, ft_lr=3e-4, scratch_epochs=30, bc_eval_every=5, bc_eval_probes=200,
        rl_iters=30, rl_episodes=128, rl_probes=200, ppo_iters=250, ppo_episodes=64, ei_updates=60, rl_eval_every=3, rl_eval_probes=500, boot_per_iter=16,
        archs=('bigru', 'cnn', 'bow'), arch_learners=('bc', 'ei_her', 'ppo'), widths=(64, 256), width_learners=('bc',), scratch=('ei_her', 'ppo', 'ei_her@bigru', 'ei_her@cnn', 'ei_her@bow'), scratch_mult=3,
        eval_probes=2000, headline=1000, eval_time_limit=60.0, eval_budgets=(10, 20, 50, 100, 200, 500, 1000, 2000), eval_splits='dev,test', synth_eval_n=500, synth_eval_probes=500, synth_eval_blind_mult=10, synth_eval_splits='test_iid,test_comp', synth_eval_solvers=('blind', 'sample', 'ida'), recheck=1, workers=4, device='cpu', threads=1, mem_scale=1.0),
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
SYNTH_EVAL_RL = ('ppo', 'ei_her')                                                                      # the reinforcement learners that are also run on the held-out synthetic theorems (the main comparison's)
BASELINES = ('blind', 'uniform')                                                                        # no network

ARCHS = ('transformer', 'bigru', 'cnn', 'bow')            # the encoders of ml_model.ARCHS (this module does not import torch; S51 checks that the lists agree)
DEFAULT_WIDTH = 128
SCRATCH_P_SYNTH = 0.7                                       # the share of the goals of a learner that starts from random weights that are synthetic theorems (the bank's problems are far out of reach of a random policy)


def size_of(d):
    """the network of width d: its layers, heads and feed-forward width.  The default (128) is 3 layers, 4 heads, 384 (ml_bc's defaults); the others grow with the width."""
    d = int(d)
    return dict(d=d, layers=2 if d <= 64 else 3 if d <= 128 else 4 if d <= 256 else 6, heads=max(2, d // 32), ff=3 * d)


def net_args(arch='transformer', width=None):
    """the command-line arguments that make a network of this encoder and width (none for the default network: its checkpoints and commands are those of before the studies existed)"""
    a = []
    if arch != 'transformer': a += ['--arch', arch]
    if width and int(width) != DEFAULT_WIDTH:
        sz = size_of(width); a += ['--d', sz['d'], '--layers', sz['layers'], '--heads', sz['heads'], '--ff', sz['ff']]
    return a


def variant_name(learner, arch='transformer', width=None):
    """bc, bc_bigru, bc_w64, ei_her_cnn: the learner, then the encoder (unless the transformer), then the width (unless the default)"""
    return learner + ('' if arch == 'transformer' else '_' + arch) + ('' if not width or int(width) == DEFAULT_WIDTH else '_w%d' % int(width))


def parse_scratch(item):
    """'ei_her' -> ('ei_her', 'transformer'); 'ei_her@bigru' -> ('ei_her', 'bigru')"""
    learner, _, arch = str(item).partition('@')
    return learner, arch or 'transformer'


def study_learners():
    """the learners a study can continue from a behaviour-cloned network: `bc` and every learner that starts from `bc` and needs no library of its own"""
    return ['bc'] + [v for v, d in RL_VARIANTS.items() if d['init'] == 'bc' and not d.get('library')]


def check_studies(P):
    ok = study_learners()
    bad = [a for a in P['archs'] if a not in ARCHS or a == 'transformer']
    if bad: raise ValueError('archs: %s (the encoders to compare with the transformer: %s)' % (', '.join(map(str, bad)), ', '.join(a for a in ARCHS if a != 'transformer')))
    if len(set(P['archs'])) != len(P['archs']): raise ValueError('archs lists an encoder twice')
    for key in ('arch_learners', 'width_learners'):
        bad = [l for l in P[key] if l not in ok]
        if bad: raise ValueError('%s: %s (learners that continue from a behaviour-cloned network: %s)' % (key, ', '.join(map(str, bad)), ', '.join(ok)))
    bad = [w for w in P['widths'] if not isinstance(w, int) or w % 32 or w <= 0 or w == DEFAULT_WIDTH]
    if bad or len(set(P['widths'])) != len(P['widths']): raise ValueError('widths: %s (multiples of 32, different from the default %d, each once)' % (', '.join(map(str, P['widths'])), DEFAULT_WIDTH))
    for item in P['scratch']:
        learner, arch = parse_scratch(item)
        if learner not in RL_VARIANTS or RL_VARIANTS[learner]['init'] != 'bc' or RL_VARIANTS[learner].get('library') or arch not in ARCHS:
            raise ValueError('scratch: %r (learner[@encoder]; the learner is one of %s, the encoder one of %s)' % (item, ', '.join(v for v in study_learners() if v != 'bc'), ', '.join(ARCHS)))
    if len(set(P['scratch'])) != len(P['scratch']): raise ValueError('scratch lists a learner twice')
    if int(P['scratch_mult']) < 1: raise ValueError('scratch_mult is at least 1')

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
    # synth_target is the number of TRAIN theorems (the held-out 12% come on top); `grow` generates until the merge would keep that many, starting from synth_attempts and giving up at three times as many
    add(Run('data/synth_gen', 'data', py('synth', 'synth_cli', 'grow', '--out', SY, '--seed', P['synth_seed'], '--train-target', P['synth_target'], '--attempts', P['synth_attempts'], '--max-attempts', 3 * P['synth_attempts'],
                                          '--shard', 2000 if P['synth_attempts'] >= 8000 else 300, '--jobs', P['synth_jobs'], '--exclude-dataset', D), needs=[ds], check=[os.path.join(SY, 'config.json')], cores=P['synth_jobs']))
    add(Run('data/synth_merge', 'data', py('synth', 'synth_cli', 'merge', '--out', SY, '--train-target', P['synth_target'], '--exclude-dataset', D), needs=['data/synth_gen'], check=[os.path.join(SY, 'manifest.json')]))
    add(Run('data/synth_verify', 'data', py('synth', 'synth_cli', 'verify', '--out', SY, '--sample', P['synth_verify_sample']), needs=['data/synth_merge'], check=[os.path.join(SY, 'verification.json')]))
    add(Run('data/synth_card', 'data', py('synth', 'synth_cli', 'card', '--out', SY), needs=['data/synth_verify'], check=[os.path.join(SY, 'DATASET_CARD.md')]))
    add(Run('data/synth_stats', 'data', py('synth', 'synth_stats', SY, '--out', os.path.join(SY, 'stats.json')), needs=['data/synth_merge'], check=[os.path.join(SY, 'stats.json')], mem_gb=1.0))

    smax = max(P['synth_sizes'])
    sizes = sorted(set(P['synth_sizes']) - {0})
    check_studies(P)
    where = {}                         # method -> (stage, the checkpoint that is evaluated): the best epoch of a behaviour-cloned model, the last iteration of a reinforcement learner (not selected on dev)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ behaviour cloning
    def pre(n, seed, arch='transformer', width=None):
        tag = ('' if arch == 'transformer' else arch + '_') + ('w%d_' % width if width else '')
        rid = 'pre/%sn%d/s%d' % (tag, n, seed)
        a = py('learner', 'ml_bc', D, '--out', out(rid), '--synth', SY, '--synth-max', n, '--no-bank', '--no-lib', '--epochs', P['pre_epochs'], '--lr', 1e-3, '--seed', seed, '--batch', P['pre_batch'],
               '--workers', W_, *dev) + (['--patience', P['pre_patience']] if P['pre_patience'] else []) + net_args(arch, width) + ['--resume']
        tags = {'method': 'pre', 'n_synth': n, 'seed': seed}
        if tag: tags.update(arch=arch, width=width or DEFAULT_WIDTH)
        return add(Run(rid, 'pre', a, needs=['data/synth_merge'], check=[out(rid) + '/final.json', out(rid) + '/best.pt'], tags=tags, mem_gb=MEM_GB['pre'] + 0.05 * n / 1000.0, cores=NC))

    def ft(variant, seed, init=None, extra=(), tags=None):
        rid = 'ft/%s/s%d' % (variant, seed)
        ev = ['--eval-every', P['bc_eval_every'], '--eval-probes', P['bc_eval_probes'], '--eval-time-limit', P['eval_time_limit']] if P['bc_eval_every'] else []
        a = py('learner', 'ml_bc', D, '--out', out(rid), '--epochs', P['ft_epochs'] if init else P['scratch_epochs'], '--lr', P['ft_lr'] if init else 1e-3, '--seed', seed, '--batch', P['pre_batch'],
               '--workers', W_, *dev) + ev + list(extra) + ['--resume']
        needs = [ds]
        if init: a += ['--init', out(init) + '/best.pt']; needs = [init]
        where[variant] = ('ft', 'best.pt')
        return add(Run(rid, 'ft', a, needs=needs, check=[out(rid) + '/final.json', out(rid) + '/best.pt'], tags=dict({'method': variant, 'seed': seed}, **(tags or {})), cores=NC))

    def init_run(arch, seed):
        """a network with random weights, saved like a trained one: where reinforcement learning FROM SCRATCH starts (ml_init)"""
        rid = 'init/%s/s%d' % (arch, seed)
        if rid in runs: return runs[rid]
        a = py('learner', 'ml_init', D, '--out', out(rid), '--seed', seed, '--threads', P['threads']) + net_args(arch)
        return add(Run(rid, 'pre', a, needs=[ds], check=[out(rid) + '/final.json', out(rid) + '/best.pt'], tags={'method': 'init', 'seed': seed, 'arch': arch, 'width': DEFAULT_WIDTH}, mem_gb=MEM_GB['data']))

    for seed in seeds:
        for n in sizes: pre(n, seed)
        top = 'pre/n%d/s%d' % (smax, seed)
        ft('bc', seed, init=top, tags={'n_synth': smax})
        ft('bc_scratch', seed, tags={'n_synth': 0})
        ft('bc_mw', seed, init=top, extra=['--mw'], tags={'n_synth': smax})
        for n in sizes:
            if n != smax: ft('bc_n%d' % n, seed, init='pre/n%d/s%d' % (n, seed), tags={'n_synth': n})

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ reinforcement learning
    def rl(variant, v, seed, init, tags=None, mult=1, p_synth=None):
        """a reinforcement learner (RL_VARIANTS entry `v`) continued from the checkpoint of the run `init`; `mult` times as many iterations, `p_synth` the share of synthetic goals (default: the variant's own)"""
        rid = 'rl/%s/s%d' % (variant, seed)
        args = list(v['args'])
        if p_synth is not None: args[args.index('--p-synth') + 1] = str(p_synth)
        common = [D, '--init', out(init) + '/best.pt', '--out', out(rid), '--seed', seed, '--workers', W_, *dev, '--synth', SY, '--synth-max', P['synth_target'], '--eval-every', P['rl_eval_every'],
                  '--eval-probes', P['rl_eval_probes'], '--resume']
        iters = (P['ppo_iters'] if v['algo'] == 'ppo' else P['rl_iters']) * int(mult)
        if v['algo'] == 'ppo':
            a = py('learner', 'ml_ppo', *common, '--iters', iters, '--episodes', P['ppo_episodes']) + args
        else:
            a = py('learner', 'ml_ei', *common, '--iters', iters, '--episodes', P['rl_episodes'], '--probes', P['rl_probes'], '--updates', P['ei_updates'], '--time-limit', P['eval_time_limit'],
                   '--boot-per-iter', P['boot_per_iter'], '--eval-time-limit', P['eval_time_limit']) + args
        actor = args[args.index('--actor') + 1] if '--actor' in args else 'sample'
        where[variant] = ('rl', 'last.pt')
        return add(Run(rid, 'rl', a, needs=[init, 'data/synth_merge'], check=[out(rid) + '/final.json', out(rid) + '/last.pt'],
                       tags=dict({'method': variant, 'seed': seed, 'algo': v['algo'], 'library': bool(v.get('library')), 'iters': iters, 'actor': actor}, **(tags or {})), mem_gb=MEM_GB['rl'] + 0.4 * int(W_), cores=NC))

    for variant, v in RL_VARIANTS.items():
        for seed in seeds: rl(variant, v, seed, 'ft/%s/s%d' % (v['init'], seed))

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ the studies of the network and of the start
    study = {'arch': [], 'size': [], 'scratch': []}

    def chain(arch, width, learners, key):
        """the network of this encoder and width: pre-trained on all the synthetic theorems, fine-tuned on the bank, and the learners (but `bc`, which is always there) continued from it"""
        bc = variant_name('bc', arch, width); nt = {'arch': arch, 'width': width or DEFAULT_WIDTH}
        for seed in seeds:
            ft(bc, seed, init=pre(smax, seed, arch, width).id, tags=dict(nt, n_synth=smax))
        study[key].append(bc)
        for learner in learners:
            if learner == 'bc': continue
            variant = variant_name(learner, arch, width)
            for seed in seeds: rl(variant, RL_VARIANTS[learner], seed, 'ft/%s/s%d' % (bc, seed), tags=nt)
            study[key].append(variant)

    for arch in P['archs']: chain(arch, None, P['arch_learners'], 'arch')
    for width in P['widths']: chain('transformer', width, P['width_learners'], 'size')
    for item in P['scratch']:
        learner, arch = parse_scratch(item); variant = 'scratch_' + variant_name(learner, arch)
        for seed in seeds:
            rl(variant, RL_VARIANTS[learner], seed, init_run(arch, seed).id, tags={'arch': arch, 'width': DEFAULT_WIDTH, 'scratch': True}, mult=P['scratch_mult'], p_synth=SCRATCH_P_SYNTH)
        study['scratch'].append(variant)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ evaluation
    def ckpt(variant, seed):
        """(path of the checkpoint to evaluate, id of the run that makes it)"""
        if variant == 'pre': return out('pre/n%d/s%d' % (smax, seed)) + '/best.pt', 'pre/n%d/s%d' % (smax, seed)
        if variant == 'init': r_ = init_run('transformer', seed); return out(r_.id) + '/best.pt', r_.id                    # (the untrained network of the same seed: the control of the synthetic evaluation)
        if re.fullmatch(r'pre_n\d+', variant): n = int(variant[5:]); return out('pre/n%d/s%d' % (n, seed)) + '/best.pt', 'pre/n%d/s%d' % (n, seed)
        stage, f = where[variant]; rid = '%s/%s/s%d' % (stage, variant, seed)
        return out(rid) + '/' + f, rid

    def evaluation(rid, group, variants, solvers, per_seed=False, lemmas=False):
        """one evaluation run per variant (all the seeds in one process: they share the world) or, with `per_seed` (a library of bootstrapped lemmas per run), per variant and seed"""
        for v in variants:
            for ss in ([[s] for s in seeds] if per_seed else [seeds]):
                r_id = '%s/%s%s' % (rid, v, ('/s%d' % ss[0]) if per_seed else '')
                a = py('learner', 'ml_evaluate', D, '--out', out(r_id), '--split', P['eval_splits'], '--regimes', 'prefix,strict', '--probes', P['eval_probes'], '--time-limit', P['eval_time_limit'], '--workers', W_, '--threads', P['threads'])
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
    for key, group in (('arch', 'arch'), ('size', 'size'), ('scratch', 'scratch')):
        if study[key]: evaluation('eval/' + group, group, study[key], [MAIN_SOLVER])
    a = py('learner', 'ml_evaluate', D, '--out', out('eval/baselines'), '--split', P['eval_splits'], '--regimes', 'prefix,strict', '--probes', P['eval_probes'], '--time-limit', P['eval_time_limit'], '--workers', W_, '--threads', P['threads'])
    for s in BASELINES: a += ['--solver', s]
    add(Run('eval/baselines', 'eval', a, needs=[ds], check=[out('eval/baselines') + '/rows.jsonl'], tags={'group': 'baselines', 'variant': 'baselines', 'solvers': list(BASELINES)}, mem_gb=MEM_GB['eval'] + 0.4 * int(W_), cores=NC))

    # the held-out SYNTHETIC theorems (ml_eval_synth): test_iid (drawn like the training theorems) and test_comp (compositions the training theorems do not contain), proved with the library only.  The olympiad splits have a few dozen
    # problems; these have hundreds, so "does what the network learned help the search?" gets an interval that is narrow.  Every network whose training saw synthetic theorems -- each pre-trained size, the behaviour-cloned and
    # the reinforcement learners -- and the olympiad-only network (the point at zero synthetic theorems) run under the network's own searches; the model-free baseline runs once.  `init` is the CONTROL: the network of the same
    # architecture and seed with random weights, under the same searches -- what the search does when the policy knows nothing, so that a gain over blind search can be attributed to what was learned and not to the search.
    def synth_evaluation():
        ssplits = P['synth_eval_splits']; n_eval = len([x for x in ssplits.split(',') if x]) * int(P['synth_eval_n'])
        base = [s for s in P['synth_eval_solvers'] if s.split(':')[0] in BASELINES]; nets = [s for s in P['synth_eval_solvers'] if s.split(':')[0] not in BASELINES]
        def common(r_id, probes=None):
            return py('learner', 'ml_eval_synth', D, SY, '--out', out(r_id), '--split', ssplits, '--limit', P['synth_eval_n'], '--probes', probes or P['synth_eval_probes'], '--time-limit', P['eval_time_limit'], '--workers', W_, '--threads', P['threads'])
        if base:
            bp = int(P['synth_eval_probes']) * max(1, int(P['synth_eval_blind_mult']))                                    # (the model-free search gets `mult` times the probes: a network's probe costs about that much more wall-clock, so the table has an equal-compute reference;
            a = common('eval/synth/baselines', bp)                                                                       #  its rows say at which probe each theorem was solved, so "within the network's probes" is read from the same rows)
            for s in base: a += ['--solver', s]
            add(Run('eval/synth/baselines', 'eval', a, needs=['data/synth_merge'], check=[out('eval/synth/baselines') + '/rows.jsonl', out('eval/synth/baselines') + '/meta.json'],
                    tags={'group': 'synth', 'variant': 'baselines', 'seeds': [0], 'solvers': base, 'synth_n': n_eval, 'probes': bp}, mem_gb=MEM_GB['eval'] + 0.4 * int(W_), cores=NC))
        if not nets: return
        variants = ['init', 'bc_scratch'] + ['pre_n%d' % n for n in sizes] + ['bc'] + [v for v in SYNTH_EVAL_RL if v in RL_VARIANTS and v not in boot]
        for v in variants:
            r_id = 'eval/synth/%s' % v; a = common(r_id); needs = ['data/synth_merge']
            sol = [s for s in nets if s.split(':')[0] == 'ida'] if v == 'init' else nets                  # (the control is about the search H7 compares: sampling an untrained network runs every theorem to its time limit and answers nothing)
            if not sol: continue
            for s in sol: a += ['--solver', s]
            for sd in seeds:
                path, need = ckpt(v, sd); a += ['--model', '%s=%s' % (v, path)]; needs.append(need)
            add(Run(r_id, 'eval', a, needs=needs, check=[out(r_id) + '/rows.jsonl', out(r_id) + '/meta.json'], tags={'group': 'synth', 'variant': v, 'seeds': list(seeds), 'solvers': sol, 'synth_n': n_eval}, mem_gb=MEM_GB['eval'] + 0.4 * int(W_), cores=NC))
    synth_evaluation()

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------ throughput, report
    add(Run('bench/throughput', 'bench', py('experiments', 'exp_bench', D, '--out', out('bench'), '--model', out('ft/bc/s%d' % seeds[0]) + '/best.pt', '--synth', SY,
                                              '--workers', ','.join(str(x) for x in sorted({1, 2, max(1, int(W_))})),
                                              '--archs', ','.join(['transformer'] + list(P['archs'])), '--widths', ','.join(str(w) for w in sorted({DEFAULT_WIDTH} | set(P['widths'])))),
            needs=['ft/bc/s%d' % seeds[0], 'data/synth_merge'], check=[out('bench') + '/bench.json']))
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
