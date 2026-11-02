"""exp_bench.py -- throughput, latency, memory and scaling of the whole stack, measured on this machine.

    python3 exp_bench.py DATASET_DIR --out runs/full/bench --model runs/full/ft/bc/s0/best.pt --synth runs/full/synth --workers 1,2,4 --repeats 3

What it measures (bench.json, with the machine and the commit; one core unless said otherwise):

  world        the time to build the world of the bank, its size, the resident memory afterwards
  gym          decisions and probes per second through the Gymnasium interface (BankEnv) with uniformly random legal choices: the cost of the environment itself, observations included.  Then a second, instrumented
               pass: the latency of a reset and of a decision (percentiles), of the decisions that complete an action separately (they probe it, replay it and list the next menu), and where the time goes
  solvers      for each solver (blind, uniform, sample, greedy, ida, mcts, mw; the network ones need --model) on a fixed set of train problems: probes, decisions and actions per second, seconds per problem.  The
               pass is repeated (--repeats; the first is cold, the engine and its caches are made in an untimed warm-up call, the later ones are warm); the number reported is the MEDIAN, with the range.  One more pass
               runs under the ledger of exp_profile: the share of the time by category (menu, probe, tactic, normal forms, policy, ...; the categories are exclusive and add up to the wall time with `other`), and the
               latency of a probe and of a menu.  A PROBE is one try of an action on a goal by the engine, whoever asks (ml_protocol): the unit the budgets of all the experiments are written in.
  latency      the probes and menus of all the instrumented passes together: percentiles of a probe by KIND of action (and how they ended: ok, refused, fuel), of a menu, a replay, the kernel's check of a finished proof
  overhead     the instrumented pass against the plain one on the same work (the price of the measurement), and whether it changed a result (it must not)
  check        the cost of trusting a proof: the reference proofs of the bank replayed through the engine (the kernel checks the finished term), and the independent checker's verdict on the same terms
  workers      the same evaluation with 1, 2, 4, ... forked worker processes (the world is shared): wall-clock, probes per second, speed-up and efficiency against one worker, and the memory (proportional set size:
               what the processes really cost together, not the sum of their resident sets, which counts the shared world again for every worker)
  determinism  the rows of all those runs, and of one in-process run, compared field by field (all but the clock): a result is a function of the task alone, whatever the number of workers, the order the tasks
               ran in or the state of the caches.  `outcome_digest` is a hash of the rows without clock fields: runs on two machines that print the same digest produced the same outcomes
  training     steps per second of behaviour-cloning updates (forward + backward + optimiser, batch 64) on `--device`; synthetic proof extraction in theorems per second
  inference    decisions per second of the network inside the `sample` solver (batch of one decision, the engine's menu included)
  archs        what each ENCODER (ml_model.ARCHS) and each width costs, on untrained networks of the sizes of the experiments (cost does not depend on the weights): parameters, training steps per second (batch 64
               of real steps, AdamW, on `--device`) and the encoder's latency on a single goal (batch of one, no gradient: the first thing a decision costs)
  machine      CPU, cores (and how many this process may use), memory, kernel, Python / torch / numpy, the digest of the kernel's sources, the load of the machine before and after (numbers measured on a busy
               machine measure the other jobs too: `busy_at_start` says so)

Nothing is estimated: each number is a wall-clock measurement (`time.perf_counter`) of the code the experiments run.  Numbers depend on the machine, so bench.json records it.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, math, random, argparse, tempfile, shutil, collections, hashlib, platform, re, array
import numpy as np
import torch
import bank as B, bank_split as BS
import gym_env as G
import ml_world as W, ml_meta as MT, ml_train as T, ml_data as D, ml_agents as AG, ml_evaluate as EV
import bank_env as BE
import exp_profile as PF, exp_spec as SP

SOLVERS = ('blind', 'uniform', 'sample', 'greedy', 'ida', 'mcts', 'mw')
clock = time.perf_counter


def pick_problems(bk, ddir, k):
    """k proved train problems, evenly spaced in the length of their reference proofs (deterministic)"""
    bank = json.load(open(os.path.join(ddir, 'bank.json')))
    cand = sorted((p['demonstration']['length'], p['name']) for p in bank['problems'] if 'demonstration' in p and p['split'] == 'train' and p['name'] in bk.placed and p['demonstration']['length'] >= 2)
    if not cand: cand = sorted((p['demonstration']['length'], p['name']) for p in bank['problems'] if 'demonstration' in p and p['name'] in bk.placed)
    if len(cand) <= k: return [n for _, n in cand]
    if k <= 0: return []
    if k == 1: return [cand[len(cand) // 2][1]]                                              # (one problem: the one in the middle)
    idx = sorted({int(round(i * (len(cand) - 1) / (k - 1))) for i in range(k)})
    return [cand[i][1] for i in idx]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the machine
def kernel_digest():
    """a hash of the kernel's sources (the header and the bindings of src/c++/hybrid): which kernel a number was measured with"""
    h = hashlib.sha256(); d = os.path.join(itp_paths.ROOT, 'src', 'c++', 'hybrid')
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else ():
        with open(os.path.join(d, n), 'rb') as f: h.update(n.encode()); h.update(f.read())
    return h.hexdigest()[:16]


def usable_cores():
    try: return len(os.sched_getaffinity(0))
    except (AttributeError, OSError): return os.cpu_count()


def machine(threads=1):
    """what the numbers were measured on (the load is read now: call it again at the end)"""
    try: load = [round(x, 2) for x in os.getloadavg()]
    except OSError: load = None
    cores = usable_cores()
    return {'cpu': MT.cpu_model(), 'cores': os.cpu_count(), 'cores_usable': cores, 'memory_gb': MT.memory_gb(), 'kernel': platform.release(), 'platform': platform.platform(), 'python': platform.python_version(),
            'torch': torch.__version__, 'numpy': np.__version__, 'torch_threads': int(threads), 'kernel_digest': kernel_digest(), 'loadavg_start': load,
            'busy_at_start': bool(load and cores and load[0] > 0.5 * cores)}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the Gymnasium interface
def gym_pass(env, names, decisions, time_limit, seed=0, lat=None):
    """random legal decisions on random problems until `decisions` were made or `time_limit` seconds passed.  With `lat` (a dict of lists) the latency of every reset and decision is saved: 'reset', 'decision',
    and 'decision_action' for the decisions that completed an action (taken or refused: they probe it, replay it and list the next menu) against 'decision_inner' for the others"""
    rng = random.Random(seed); steps = probes = eps = 0; t0 = clock()
    while steps < decisions and clock() - t0 < time_limit:
        t = clock()
        try: obs, info = env.reset(options={'problem': rng.choice(names)})
        except ValueError: continue
        if lat is not None: lat['reset'].append(clock() - t)
        eps += 1; spent = info['actions'] + info['refused']
        while True:
            n = int(obs['n_options'])
            if n <= 0: break
            a = rng.randrange(n)
            t = clock(); obs, r, term, trunc, info = env.step(a); dt = clock() - t; steps += 1
            if lat is not None:
                now = info['actions'] + info['refused']
                lat['decision'].append(dt); lat['decision_action' if now > spent else 'decision_inner'].append(dt); spent = now
            if term or trunc: break
        probes += info['actions'] + info['refused']
    dt = clock() - t0
    return {'decisions': steps, 'probes': probes, 'episodes': eps, 'seconds': round(dt, 3), 'decisions_per_s': steps / dt if dt else None, 'probes_per_s': probes / dt if dt else None}


def bench_gym(bk, ctx, eng, names, decisions=2000, time_limit=40.0, seed=0, profile_decisions=1000):
    env = BE.BankEnv(bk, names=names, ctx=ctx, engine=eng, rich=False, max_actions=60)
    out = gym_pass(env, names, decisions, time_limit, seed)
    if profile_decisions:
        lat = collections.defaultdict(lambda: array.array('d')); L = PF.Ledger()
        with PF.instrument(L):
            t0 = clock(); p = gym_pass(env, names, profile_decisions, min(time_limit, 20.0), seed + 1, lat); wall = clock() - t0
        out['profiled'] = {'decisions': p['decisions'], 'seconds': round(wall, 3), 'latency': {k: PF.summarize(v) for k, v in sorted(lat.items())}, 'breakdown': L.report(wall),
                           'engine_latency': engine_latency(L), 'hooks_skipped': list(L.skipped)}
    return out


def engine_latency(L):
    """the latencies the ledger saw, as plain data: a probe (all kinds, and by kind with how the probes ended), a menu (with the number of options), a replay, a start, a kernel check"""
    out = {}
    pk = [k for k in L.samples if k.startswith('probe:')]
    if pk:
        allp = array.array('d')
        for k in pk: allp.extend(L.samples[k])
        out['probe'] = PF.summarize(allp)
        out['probe_by_kind'] = {k.split(':', 1)[1]: dict(PF.summarize(L.samples[k]), outcomes=dict(L.outcomes.get(k, {}))) for k in sorted(pk)}
    for k in ('menu', 'replay', 'start', 'verify'):
        if k in L.samples:
            out[k] = PF.summarize(L.samples[k])
            if k + '#n' in L.values: out[k]['mean_options'] = float(np.mean(L.values[k + '#n']))
    return out


def merge_ledger(pool, L):
    """add the samples and outcomes of the ledger L to `pool` = (samples dict, outcomes dict) -- the latencies of several instrumented passes together"""
    for k, v in L.samples.items(): pool[0].setdefault(k, array.array('d')).extend(v)
    for k, c in L.outcomes.items(): pool[1].setdefault(k, collections.Counter()).update(c)
    for k, v in L.values.items(): pool[2].setdefault(k, array.array('d')).extend(v)


def pooled_latency(pool):
    L = PF.Ledger(); L.samples.update(pool[0]); L.outcomes.update(pool[1]); L.values.update(pool[2])
    return engine_latency(L)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the solvers
def solve_all(solver, bk, names, probes, time_limit):
    """one pass of `solver` over the problems: (rows, wall-clock seconds of each solve)"""
    rows, secs = [], []
    for n in names:
        pl = bk.placed[n]
        t = clock(); row = solver.solve(pl.stmt, pl.premises('prefix'), probes, seed=0, name=n, regime='prefix', time_limit=time_limit); secs.append(clock() - t); rows.append(row)
    return rows, secs


def outcome_of(rows):
    """what a pass produced, without the clock: the rows' fields that are not volatile (see is_volatile), as a comparable list"""
    return [canonical(r) for r in rows]


def bench_solvers(bk, ctx, eng, net, names, specs, probes, time_limit, say=print, repeats=1, pool=None):
    out = {}
    for spec in specs:
        name_, _ = AG.parse_spec(spec)
        if name_ in AG.NEEDS_NET and net is None: continue
        solver = AG.make_solver(spec, eng, net if name_ in AG.NEEDS_NET else None, ctx if name_ in AG.NEEDS_NET else None)
        pl = bk.placed[names[0]]
        solver.solve(pl.stmt, pl.premises('prefix'), 3, seed=0, name=names[0], regime='prefix', time_limit=time_limit)       # (a warm-up call: it makes the engines and the tables of the solver, which is not what is measured)
        rss0 = MT.rss_mb(); reps = []; first = None; stable = True
        for rep in range(max(1, int(repeats))):
            rows, secs = solve_all(solver, bk, names, probes, time_limit)
            tot = collections.Counter()
            for r in rows: tot['probes'] += r.get('probes', 0); tot['decisions'] += r.get('decisions', 0) or 0; tot['actions'] += r.get('actions', 0) or 0
            reps.append({'seconds': float(sum(secs)), 'probes': tot['probes'], 'decisions': tot['decisions'], 'actions': tot['actions'], 'solved': sum(int(bool(r.get('solved'))) for r in rows)})
            oc = outcome_of(rows)
            if first is None: first = oc
            elif oc != first: stable = False
        med = lambda key: float(np.median([r[key] / r['seconds'] for r in reps if r['seconds']])) if any(r['seconds'] for r in reps) and any(r[key] for r in reps) else None
        pps = [r['probes'] / r['seconds'] for r in reps if r['seconds']]
        secs_med = float(np.median([r['seconds'] for r in reps]))
        entry = {'n': len(names), 'solved': reps[0]['solved'], 'probes': reps[0]['probes'], 'seconds': round(secs_med, 3), 'probes_per_s': med('probes'), 'decisions_per_s': med('decisions'), 'actions_per_s': med('actions'),
                 'seconds_per_problem': secs_med / len(names), 'peak_rss_mb': MT.peak_rss_mb(), 'rss_mb': MT.rss_mb(), 'rss_delta_mb': (MT.rss_mb() - rss0) if rss0 is not None and MT.rss_mb() is not None else None,
                 'repeats': len(reps), 'probes_per_s_reps': pps, 'probes_per_s_min': min(pps) if pps else None, 'probes_per_s_max': max(pps) if pps else None, 'stable': stable}
        # one pass under the ledger: where the time goes, and what a probe and a menu cost; it must give the rows of the plain pass
        L = PF.Ledger()
        with PF.instrument(L):
            t0 = clock(); rows_i, secs_i = solve_all(solver, bk, names, probes, time_limit); wall = clock() - t0
        entry['profile'] = {'breakdown': L.report(wall), 'latency': engine_latency(L), 'seconds': round(wall, 3),
                            'plain_seconds_median': round(secs_med, 3), 'ratio_to_plain': wall / secs_med if secs_med else None, 'same_outcome': outcome_of(rows_i) == first,
                            'probe_calls': int(L.calls.get('probe', 0)), 'probes_counted': sum(r.get('probes', 0) for r in rows_i), 'balanced': L.balanced(), 'hooks_skipped': list(L.skipped)}
        if pool is not None: merge_ledger(pool, L)
        out[spec] = entry
        say('  %-10s %7.1f probes/s (%s)  %d of %d solved%s' % (spec, entry['probes_per_s'] or 0, ('%.1f-%.1f over %d passes' % (entry['probes_per_s_min'], entry['probes_per_s_max'], len(reps))) if len(reps) > 1 else 'one pass', entry['solved'], len(names),
                                                          '' if stable and entry['profile']['same_outcome'] else '  NOT REPRODUCED (%s)' % ('passes differ' if not stable else 'the instrumented pass differs')))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ determinism
_VOLATILE = re.compile(r'(^|_)(seconds|time|wall|elapsed|host|pid|worker|clock)(_|$)|^t_|_s$')


def is_volatile(key):
    """a field of a row that is a measurement of the clock or of the machine, not a result: seconds, time, t_policy, ..."""
    return bool(_VOLATILE.search(key))


def canonical(row): return {k: v for k, v in row.items() if not is_volatile(k)}


def compare_rows(runs, time_limit=None, slow=0.9):
    """are the rows of several runs of the same tasks the same, apart from the clock?  `runs`: {label: rows}; a task is identified by (method, seed, name, regime).  A task whose attempt used at least `slow` of
    the time limit in any run is left out (the time limit ends such an attempt wherever the clock says, which is not a property of the task) and counted.  The digest is a hash of the first run's rows
    without clock fields, in the order of the tasks: the same on two machines iff they found the same proofs after the same number of probes (for the tasks neither left out).
    Returns {'runs', 'tasks' (compared), 'identical' (every task is in every run and equal in all), 'n_differing', 'differing' (the first few: task, run, fields), 'missing', 'excluded_slow', 'outcome_digest'}"""
    labels = list(runs); keyed = {}
    for lab in labels:
        d = {}
        for r in runs[lab]: d[(r.get('method'), r.get('seed'), r.get('name'), r.get('regime'))] = r
        keyed[lab] = d
    allk = set().union(*[set(d) for d in keyed.values()]) if keyed else set()
    common = set.intersection(*[set(d) for d in keyed.values()]) if keyed else set()
    slow_k = {k for k in common if time_limit and any((keyed[l][k].get('seconds') or 0) >= slow * time_limit for l in labels)}
    use = sorted(common - slow_k, key=repr); differing = []; digest_rows = []
    for k in use:
        c0 = canonical(keyed[labels[0]][k]); digest_rows.append([list(k), c0])
        for l in labels[1:]:
            c1 = canonical(keyed[l][k])
            if c1 != c0: differing.append({'task': list(k), 'run': l, 'fields': sorted(f for f in set(c0) | set(c1) if c0.get(f) != c1.get(f))}); break
    digest = hashlib.sha256(json.dumps(digest_rows, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {'runs': labels, 'tasks': len(use), 'identical': bool(use) and not differing and len(common) == len(allk), 'n_differing': len(differing), 'differing': differing[:10], 'missing': len(allk - common),
            'excluded_slow': len(slow_k), 'outcome_digest': digest}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ workers
def bench_workers(bk, ctx, eng, model, names, workers, probes, time_limit, say=print, in_process=True):
    """the evaluation of `names` with each count of worker processes: ({workers: measurements}, determinism).  With `in_process` the same tasks also run once in this process (no workers), for the comparison
    of the rows only: it is not a point of the speed-up curve."""
    solver = 'sample' if model else 'blind'
    res = {}; runs = {}; workers = sorted({int(w) for w in workers if int(w) > 0})
    for w in ([0] if in_process else []) + workers:
        d = tempfile.mkdtemp(prefix='bench_w%d_' % w)
        try:
            with PF.ProcessSampler() as ps:
                t0 = clock()
                rows = EV.evaluate(bk, ctx, eng, [('bench', model)] if model else [], [solver], names, ('prefix',), probes, time_limit, d, workers=w, chunk=1, progress=lambda *a: None)
                dt = clock() - t0
            runs[w] = rows
            if w == 0:
                say('  in process: %.1f s (for the comparison of the rows)' % dt); continue
            pr = sum(r.get('probes', 0) for r in rows)
            res[str(w)] = {'seconds': round(dt, 3), 'probes': pr, 'probes_per_s': pr / dt, 'tasks': len(rows), 'memory': ps.result()}
            mem = res[str(w)]['memory']
            say('  %2d workers: %.1f s, %.1f probes/s, %s MB shared-aware (%s MB resident summed over %d processes)' % (w, dt, pr / dt, mem['pss_mb_max'], mem['rss_sum_mb_max'], mem['processes_max']))
        finally:
            shutil.rmtree(d, ignore_errors=True)
    if res:
        base = res[str(min(workers))]
        for w in workers:
            r = res[str(w)]; r['speedup'] = base['seconds'] / r['seconds'] * (1.0); r['efficiency'] = r['speedup'] / (w / float(min(workers)))
    det = compare_rows({('in process' if w == 0 else '%d workers' % w): rows for w, rows in runs.items()}, time_limit)
    say('  determinism: %s (%d tasks compared%s, digest %s)' % ('identical' if det['identical'] else 'DIFFERENT (%d tasks)' % det['n_differing'], det['tasks'], (', %d left out as slow' % det['excluded_slow']) if det['excluded_slow'] else '', det['outcome_digest']))
    return res, det


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the cost of trusting a proof
def bench_check(bk, ctx, eng, ddir, limit=40, say=print):
    """the reference proofs of the bank (up to `limit`, evenly spaced in length) replayed through the engine -- the kernel checks each finished term -- and the independent checker's verdict on the same terms"""
    import gym_acts as ACTS, recheck as RC
    bank = json.load(open(os.path.join(ddir, 'bank.json')))
    cand = sorted((p['demonstration']['length'], p['name'], p['demonstration']['steps']) for p in bank['problems'] if 'demonstration' in p and p['name'] in bk.placed and p['demonstration'].get('steps'))
    if not cand: return {}
    if len(cand) > limit: cand = [cand[i] for i in sorted({int(round(i * (len(cand) - 1) / (limit - 1))) for i in range(limit)})]
    L = PF.Ledger(); items = []; replay = array.array('d'); fail = 0; saved = eng.premises
    try:
        with PF.instrument(L, only={'verify'}):
            for length, name, steps in cand:
                pl = bk.placed[name]
                t = clock()
                try: term = ACTS.replay_portable(eng, pl.stmt, steps, pl.premises('prefix'))
                except Exception: fail += 1; continue
                replay.append(clock() - t); items.append((term, pl.stmt, length))
    finally:
        eng.premises = saved                                                          # (a replay sets the premises of the engine and leaves them)
    t = clock(); so = RC.SecondOpinion(bk.w.k, needed_by=[x for t_, s_, _ in items for x in (t_, s_)]) if items else None; setup = clock() - t
    indep = array.array('d'); ok = 0
    for term, stmt, _ in items:
        t = clock(); good, _msg = so.accepts(term, stmt); indep.append(clock() - t); ok += int(bool(good))
    out = {'proofs': len(items), 'replay_failed': fail, 'steps_mean': float(np.mean([n for _, _, n in items])) if items else None, 'replay': PF.summarize(replay), 'kernel_check': L.latency('verify'),
           'independent_check': PF.summarize(indep), 'independent_accepted': ok, 'independent_setup_s': round(setup, 3)}
    say('  check: %d reference proofs; replay %.1f ms, kernel check %.2f ms, independent checker %.1f ms (medians); %d accepted by the independent checker' % (
        len(items), out['replay'].get('p50_ms', 0), out['kernel_check'].get('p50_ms', 0), out['independent_check'].get('p50_ms', 0), ok))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ training, extraction
def bench_training(a, bk, ctx, model, device, batches=12, say=print):
    data = T.build_steps(a.data, bk, ctx, G.Engine(bk.w, kinds=set(G.Kind), strict=True))
    steps = list(data['bank'].get('train', [])) + list(data['lib'].get('train', []))
    if len(steps) < 64: return {}
    net, cfg, _ = T.load(model, ctx.bags); net.train(); dev = torch.device(device); net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    rng = random.Random(0); bs = 64; times = []
    for i in range(batches + 2):
        idx = [rng.randrange(len(steps)) for _ in range(bs)]
        t0 = time.perf_counter()
        batch = D.collate([steps[j] for j in idx]); res = net(batch, aux={'size': 0.2, 'mgr': 0.2})
        opt.zero_grad(); res['loss'].backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if dev.type == 'cuda': torch.cuda.synchronize()
        if i >= 2: times.append(time.perf_counter() - t0)                       # (the first two batches warm caches up)
    sps = bs / float(np.mean(times))
    say('  training: %.0f steps/s (batch %d, %s)' % (sps, bs, device))
    return {'train_steps_per_s': sps, 'batch': bs, 'device': device, 'params': sum(p.numel() for p in net.parameters()), 'steps_available': len(steps)}


def bench_archs(a, bk, ctx, archs, widths, device, batches=8, goals=100, say=print):
    """the cost of each encoder at the default width and of the transformer at each other width: {'<arch>@<width>': {...}}; untrained networks of the experiments' sizes (SP.size_of), batches of real steps"""
    data = T.build_steps(a.data, bk, ctx, G.Engine(bk.w, kinds=set(G.Kind), strict=True))
    steps = list(data['bank'].get('train', [])) + list(data['lib'].get('train', []))
    if len(steps) < 16: return {}
    dev = torch.device(device); out = collections.OrderedDict(); bs = 64
    combos = [(arch, SP.DEFAULT_WIDTH) for arch in archs] + [('transformer', w) for w in widths if w != SP.DEFAULT_WIDTH]
    gl = [D.collate([st]) for st in steps[:goals]]
    for arch, width in combos:
        torch.manual_seed(0)
        net = T.make_net(ctx.bags, dict(SP.size_of(width), dropout=0.1, arch=arch)); net.to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4); rng = random.Random(0); times = []
        net.train()
        for i in range(batches + 2):
            idx = [rng.randrange(len(steps)) for _ in range(bs)]
            t0 = clock(); batch = D.collate([steps[j] for j in idx]); res = net(batch, aux={'size': 0.2, 'mgr': 0.2})
            opt.zero_grad(); res['loss'].backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if dev.type == 'cuda': torch.cuda.synchronize()
            if i >= 2: times.append(clock() - t0)                              # (the first two batches warm caches up)
        net.eval(); lat = array.array('d')
        with torch.no_grad():
            for rep in range(2):
                for b in gl:
                    t = [torch.as_tensor(b[k]).to(dev) for k in ('tokens', 'seg', 'mask', 'hyp_of', 'n_hyps')]
                    t0 = clock(); net.encode(*t)
                    if dev.type == 'cuda': torch.cuda.synchronize()
                    if rep: lat.append(clock() - t0)                                # (the first pass warms up)
        key = '%s@%d' % (arch, width)
        out[key] = {'arch': arch, 'width': width, 'params': sum(p.numel() for p in net.parameters()), 'train_steps_per_s': bs / float(np.mean(times)), 'batch': bs, 'device': device, 'encode': PF.summarize(lat), 'steps_available': len(steps)}
        say('  %-16s %9d parameters, training %6.0f steps/s, encoding a goal: median %.2f ms' % (key, out[key]['params'], out[key]['train_steps_per_s'], out[key]['encode'].get('p50_ms', float('nan'))))
        del net, opt
    return out


def bench_extract(a, bk, ctx, n=100, say=print):
    import ml_synth as MS
    recs = MS.load_records(a.synth, 'train', limit=n)
    if not recs: return {}
    t0 = time.perf_counter(); steps, stats = MS.extract_steps(recs, bk, ctx, bk.w.gid[B.LIBRARY_LIMIT], workers=0); dt = time.perf_counter() - t0
    say('  extraction: %.1f theorems/s' % (len(recs) / dt))
    return {'extract_theorems_per_s': len(recs) / dt, 'extract_steps_per_s': len(steps) / dt, 'extract_n': len(recs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('--out', required=True); ap.add_argument('--model', default=''); ap.add_argument('--synth', default='')
    ap.add_argument('--problems', type=int, default=12); ap.add_argument('--probes', type=int, default=200); ap.add_argument('--time-limit', type=float, default=20.0)
    ap.add_argument('--repeats', type=int, default=3, help='passes of each solver over the problems (the first is cold); the median is reported')
    ap.add_argument('--workers', default='1,2'); ap.add_argument('--worker-problems', type=int, default=16); ap.add_argument('--worker-probes', type=int, default=120)
    ap.add_argument('--check-proofs', type=int, default=40, help='reference proofs for the cost of checking (0: skip)')
    ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=1); ap.add_argument('--no-gym', action='store_true'); ap.add_argument('--no-train', action='store_true')
    ap.add_argument('--archs', default='', help='encoders to cost (comma separated, ml_model.ARCHS; the default transformer is always costed with them); empty: skip'); ap.add_argument('--widths', default='', help='widths of the transformer to cost (comma separated); the default width is always costed with --archs')
    W.add_args(ap, lemmas=False)
    a = ap.parse_args(argv)
    archs = [x for x in a.archs.split(',') if x]
    try: widths = [int(x) for x in a.widths.split(',') if x]
    except ValueError: sys.exit('--widths: %r (comma separated whole numbers)' % a.widths)
    bad = [x for x in archs if x not in SP.ARCHS]
    if bad: sys.exit('--archs: unknown encoder %s (one of %s)' % (', '.join(bad), ', '.join(SP.ARCHS)))                       # (before the world is loaded and anything is measured)
    bad = [w for w in widths if w <= 0 or w % 32]
    if bad: sys.exit('--widths: %s (multiples of 32: the heads of the transformer are width / 32)' % ', '.join(map(str, bad)))
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    mach = machine(a.threads)
    if mach['busy_at_start']: print('WARNING: the machine is busy (load %s on %s usable cores): the numbers below measure the other jobs too' % (mach['loadavg_start'], mach['cores_usable']), flush=True)
    t0 = time.perf_counter(); bk, ctx, eng = W.load(a); t_world = time.perf_counter() - t0
    world = {'seconds': round(t_world, 2), 'globals': bk.w.k.num_globals(), 'problems': len(bk.placed), 'rss_mb': MT.rss_mb()}
    model = a.model if a.model and os.path.exists(a.model) else ''
    net = None
    if model:
        net, cfg, _ = T.load(model, ctx.bags); net.eval(); world['params'] = sum(p.numel() for p in net.parameters())
    names = pick_problems(bk, a.data, a.problems)
    print('world %.1f s, %d globals, %.0f MB; %d problems for the solvers: %s' % (t_world, world['globals'], world['rss_mb'] or 0, len(names), names), flush=True)
    result = {'n_problems': len(names), 'problems': names, 'probes': a.probes, 'time_limit': a.time_limit, 'repeats': a.repeats}
    if not a.no_gym:
        print('gymnasium interface ...', flush=True); g = result['gym'] = bench_gym(bk, ctx, eng, names)
        print('  %.0f decisions/s, %.0f probes/s' % (g['decisions_per_s'], g['probes_per_s']), flush=True)
        lt = (g.get('profiled') or {}).get('latency', {})
        if lt.get('decision', {}).get('n'): print('  a decision: median %.2f ms, p99 %.1f ms; one that completes an action: median %.1f ms' % (lt['decision']['p50_ms'], lt['decision']['p99_ms'], lt.get('decision_action', {}).get('p50_ms', float('nan'))), flush=True)
    print('solvers ...', flush=True)
    pool = ({}, {}, {})
    result['solvers'] = bench_solvers(bk, ctx, eng, net, names, SOLVERS, a.probes, a.time_limit, repeats=a.repeats, pool=pool)
    result['latency'] = pooled_latency(pool)
    result['overhead'] = {s: {'ratio': v['profile']['ratio_to_plain'], 'same_outcome': v['profile']['same_outcome']} for s, v in result['solvers'].items()}
    if 'sample' in result['solvers'] and result['solvers']['sample'].get('decisions_per_s'): world['infer_decisions_per_s'] = result['solvers']['sample']['decisions_per_s']
    if a.check_proofs:
        print('the cost of checking ...', flush=True); result['check'] = bench_check(bk, ctx, eng, a.data, limit=a.check_proofs)
    ws = [int(x) for x in a.workers.split(',') if x]
    if len(ws) > 1:
        print('workers ...', flush=True)
        wn = pick_problems(bk, a.data, a.worker_problems)
        result['workers'], result['determinism'] = bench_workers(bk, ctx, eng, model, wn, ws, a.worker_probes, a.time_limit)
    if model and not a.no_train:
        print('training ...', flush=True); tr = bench_training(a, bk, ctx, model, a.device); world.update(tr)
    if a.synth and os.path.exists(os.path.join(a.synth, 'train.jsonl.gz')):
        print('extraction ...', flush=True); world.update(bench_extract(a, bk, ctx))
    if (archs or widths) and not a.no_train:
        archs = ['transformer'] + [x for x in dict.fromkeys(archs) if x != 'transformer']                        # (the default network is always costed: the others are compared with it)
        print('encoders and widths ...', flush=True); result['archs'] = bench_archs(a, bk, ctx, archs, widths, a.device)
    result['world'] = world
    try: mach['loadavg_end'] = [round(x, 2) for x in os.getloadavg()]
    except OSError: pass
    result['machine'] = mach
    result['meta'] = MT.run_meta(vars(a), {'peak_rss_mb': MT.peak_rss_mb()})
    MT.write_json(os.path.join(a.out, 'bench.json'), result)
    print('wrote', os.path.join(a.out, 'bench.json'))


if __name__ == '__main__':
    main()
