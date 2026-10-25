"""ml_ei.py -- expert iteration: the policy attacks the TRAIN problems, and the proofs the kernel accepts become its training data.

    python3 ml_ei.py DATASET_DIR --init runs/ft/s0/best.pt --out runs/ei/s0 --actor sample --iters 30 --episodes 128 --probes 200 --workers 4 --seed 0
    python3 ml_ei.py DATASET_DIR --init ... --out runs/ei_full/s0 --actor mw --her --bootstrap --synth data/synth_v2 --workers 4

One ITERATION: (1) `--episodes` tasks are drawn from the goals (below); (2) workers attack them with the ACTOR -- any prover of ml_agents (`sample`, `mw`, `ida`, `mcts`, with options `mcts:value=critic`) -- each
with a budget of `--probes` probes; every proof found has been replayed on a live proof and checked by the kernel; (3) the learner trains on a mix of the proofs found, the reference proofs of the train split
and (with `--her`) the theorems the attempts proved on the way; (4) with `--bootstrap`, hindsight theorems are admitted to the LIBRARY (ml_lemmas) and every later attempt may use them.

    goals        the train problems of the bank (those with a reference proof of at most `--max-ref-len` actions; the others -- long proofs, and the problems nobody has proved -- with probability `--p-hard`), the
                 sub-theorems of train proofs (`--p-sub`), synthetic theorems (`--p-synth`, `--synth DIR`).  Nothing from the dev or test split is a goal, a demonstration or a lemma.
    hindsight    (`--her`) an attempt that fails still proves things: the goals it closed on the way (ml_her) are theorems with kernel-checked proofs, and the steps of those proofs are training data.  Without
                 `--her` only the proofs of whole tasks are learned from; with `--p-sub 0` also the static sub-theorem goals are off: the ablation of the paper.
    bootstrap    (`--bootstrap`) a hindsight theorem that can be said and proved from the library alone (and the lemmas admitted before it), that is not a statement of the bank and does not say what a dev or test
                 problem says, becomes the global `boot_NNNN` of the world: the kernel checked its proof, the independent checker will read it like any other lemma.  The lemmas are written to `lemmas.json`; a model
                 trained with them must be evaluated with them (`ml_evaluate.py --lemmas`).
    training     `--updates` gradient steps per iteration, each on a batch of `--batch` steps drawn from `--mix` (demo: reference proofs of the train split; full: proofs of whole tasks found by the actor; her:
                 steps of hindsight theorems).  With the actor `mw` the steps carry the manager-worker masks and labels (ml_hier.mw_steps).
    evaluation   every `--eval-every` iterations: the dev problems by `--eval-solver` within `--eval-probes` probes (kernel-checked solves), the number the log shows.

The run directory: `last.pt` (the network after the last iteration; the checkpoint of the experiments), `best.pt` (best dev solve count so far), `state.pt` (everything to continue: `--resume`), `log.jsonl` (a line per
iteration), `lemmas.json` (with `--bootstrap`), `final.json`.  Several seeds are several runs.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, math, argparse, collections
import numpy as np
import torch
import bank as B, bank_split as BS
import gym_env as G
import ml_data as D, ml_train as T, ml_protocol as PR, ml_agents as AG, ml_goals as GL, ml_her as HER, ml_hier as H, ml_world as W, ml_meta as MT
from ml_pool import Pool, WeightsFile


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ buffers
class Buffer:
    """positive examples: key -> (steps, proof length); the oldest go first when there are more than `max_steps` steps; a key seen again keeps the shorter proof"""
    def __init__(self, max_steps): self.max_steps = int(max_steps); self.items = collections.OrderedDict(); self.n = 0

    def add(self, key, steps, length):
        cur = self.items.get(key)
        if cur is not None and cur[1] <= length: return False
        if cur is not None: self.n -= len(cur[0]); del self.items[key]
        self.items[key] = (steps, length); self.n += len(steps)
        while self.n > self.max_steps and self.items:
            _, (ss, _) = self.items.popitem(last=False); self.n -= len(ss)
        return True

    def flat(self): return [s for ss, _ in self.items.values() for s in ss]
    def __len__(self): return len(self.items)
    def state(self): return [(k, ss, L) for k, (ss, L) in self.items.items()]
    @classmethod
    def from_state(cls, max_steps, st):
        b = cls(max_steps)
        for k, ss, L in st: b.items[tuple(k) if isinstance(k, list) else k] = (ss, L); b.n += len(ss)
        return b


def parse_mix(text):
    out = {}
    for item in filter(None, text.split(',')):
        k, _, v = item.partition('=')
        if k not in ('demo', 'full', 'her'): raise ValueError("--mix: demo=, full=, her= (got %r)" % k)
        out[k] = float(v)
    for k in ('demo', 'full', 'her'): out.setdefault(k, 0.0)
    return out


def draw(rng, pools, weights, n):
    """n steps from the non-empty `pools` (lists), in proportion to `weights` (the share of an empty pool goes to the others)"""
    live = [(p, w) for p, w in zip(pools, weights) if len(p) and w > 0]
    if not live: return []
    tot = sum(w for _, w in live); pv = np.asarray([w / tot for _, w in live], dtype=np.float64); pv /= pv.sum(); counts = rng.multinomial(n, pv)
    out = []
    for (p, _), c in zip(live, counts):
        if c: out.extend(p[int(i)] for i in rng.choice(len(p), size=int(c), replace=len(p) < c))
    return out


def strip(steps):
    for s in steps: s.goal.key = None                       # (the engine's key of the goal holds kernel handles: nothing to send to another process)
    return steps


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ workers
def make_worker_state(bk, ctx, eng, lib, cfg):
    def init():
        torch.set_num_threads(1)
        net = T.make_net(ctx.bags, cfg); net.eval()
        return {'bk': bk, 'ctx': ctx, 'eng': eng, 'lib': lib, 'net': net, 'cache': {}, 'bags': ctx.bags}
    return init


def sync_world(st, p):
    """bring a worker to the learner's version of the weights and of the library"""
    path, version = p['weights']
    WeightsFile.load_into(st['net'], path, version, st['cache'])
    lib = st['lib']
    if lib is not None and p.get('lemmas'):
        n = lib.install_records(p['lemmas'])
        if n: lib.sync(nets=[st['net']])
        if len(lib.records) != len(p['lemmas']) or lib.block_digest() != p['block_digest']:
            raise RuntimeError('the library of this worker (%s) is not the learner\'s (%s)' % (lib.block_digest(), p['block_digest']))
    if st['bags'] is not st['ctx'].bags:                      # (the world grew in this process: the learner's `lib.sync` made new global features)
        st['net'].set_globals(st['ctx'].bags)
    st['bags'] = st['ctx'].bags


def attempt(st, solver, base_ma, p, idx, task):
    bk, eng, ctx, lib = st['bk'], st['eng'], st['ctx'], st['lib']
    out = {'name': task['name'], 'source': task['source']}
    try:
        name, stmt, prem = GL.resolve(bk, eng, lib, task)
        o = task['options']; regime = o.get('regime', 'prefix')
        probes = p['probes'] if task['source'] == 'bank' else max(10, int(p['probes'] * p['sub_probe_frac']))
        if 'max_actions' in solver.opts: solver.opts['max_actions'] = base_ma if task['source'] == 'bank' else int(min(base_ma, o.get('budget', base_ma)))
        harvest = HER.Harvest(eng, prem, name, max_len=p['her_max_len'], max_n=p['her_max'], max_candidates=p['her_candidates']) if p['harvest'] else None
        seed = PR.stable_seed('ei', p['seed'], p['iter'], idx, name)
        row = solver.solve(stmt, prem, probes, seed=seed, name=name, regime=regime, time_limit=p['time_limit'], harvest=harvest)
        out.update({'regime': regime, 'solved': bool(row.get('solved')), 'probes': int(row.get('probes', 0)), 'seconds': float(row.get('seconds', 0.0)), 'length': row.get('steps'),
                    'decisions': int(row.get('decisions', 0)), 'error': row.get('error'), 'budget': probes})
        stats = collections.Counter(); proof = row.get('proof')
        if out['solved'] and proof:
            ss = D.extract(ctx, eng, stmt, prem, proof, name=name, stats=stats)
            if ss: out['steps'] = strip(ss)
            if lib is not None: out['uses'] = sorted(lib.uses(proof))
        her = []
        if harvest is not None:
            for f in harvest.found():
                ss = HER.steps_of(ctx, eng, f, name=f.digest, stats=stats)
                if ss: her.append({'rec': f.to_dict(), 'steps': strip(ss)})
            out['harvest'] = dict(harvest.stats)
        out['her'] = her; out['extract'] = dict(stats)
    except (MemoryError, KeyboardInterrupt): raise
    except Exception as e:
        out.setdefault('solved', False); out['error'] = '%s: %s' % (type(e).__name__, str(e)[:160]); out.setdefault('her', []); out.setdefault('probes', 0)
    return out


def h_attempts(st, p):
    sync_world(st, p)
    solver = AG.make_solver(p['actor'], st['eng'], st['net'], st['ctx'])
    base_ma = int(solver.opts.get('max_actions', 40))
    return [attempt(st, solver, base_ma, p, idx, task) for idx, task in p['tasks']]


def h_eval(st, p):
    sync_world(st, p)
    bk, lib = st['bk'], st['lib']
    solver = AG.make_solver(p['solver'], st['eng'], st['net'], st['ctx'])
    rows = []
    for name, regime in p['items']:
        pl = bk.placed[name]; prem = pl.premises(regime)
        if lib is not None: prem = lib.premises(prem)
        row = solver.solve(pl.stmt, prem, p['probes'], seed=PR.stable_seed('eval', p['seed'], name, regime), name=name, regime=regime, time_limit=p['time_limit'])
        row.pop('proof', None)
        rows.append(PR.annotate(row, bk, 'ei-dev', p['seed']))
    return rows


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the learner
def update(net, opt, pools, weights, a, rng, device):
    """`--updates` gradient steps of behaviour cloning on batches drawn from the pools; the averaged metrics"""
    net.train(); sums = collections.Counter(); n = 0
    aux = {'size': a.size_coef, 'mgr': a.mgr_coef}
    for _ in range(a.updates):
        steps = draw(rng, pools, weights, a.batch)
        if not steps: break
        res = net(D.collate(steps), aux=aux)
        opt.zero_grad(); res['loss'].backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        sums['policy_loss'] += T.num(res['policy_loss']); sums['size_loss'] += T.num(res['size_loss']); sums['mgr_loss'] += T.num(res['mgr_loss'])
        sums['acc'] += res['correct'] / max(res['n'], 1); n += 1
    net.eval()
    return {k: v / max(n, 1) for k, v in sums.items()} | {'updates': n}


def save_lemmas(lib, path):
    if lib is not None: lib.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('--init', required=True); ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--actor', default='sample', help="the prover that attacks the tasks: a solver spec of ml_agents (sample, mw, ida, mcts; options as 'mcts:value=critic')")
    ap.add_argument('--iters', type=int, default=30); ap.add_argument('--episodes', type=int, default=128, help='tasks per iteration'); ap.add_argument('--probes', type=int, default=200, help='probes per task')
    ap.add_argument('--sub-probe-frac', type=float, default=0.5, help='sub-theorems and synthetic theorems get this fraction of the probes'); ap.add_argument('--time-limit', type=float, default=60.0)
    ap.add_argument('--p-sub', type=float, default=0.0, help='probability that a task is a sub-theorem of a train proof'); ap.add_argument('--p-synth', type=float, default=0.0); ap.add_argument('--p-hard', type=float, default=0.3)
    ap.add_argument('--max-ref-len', type=int, default=30); ap.add_argument('--budget-mult', type=float, default=3.0)
    ap.add_argument('--synth', default='', help='a merged synthetic dataset: its train theorems are goals'); ap.add_argument('--synth-max', type=int, default=20000)
    ap.add_argument('--her', action='store_true', help='train on the theorems the attempts proved on the way'); ap.add_argument('--her-max', type=int, default=12); ap.add_argument('--her-max-len', type=int, default=20)
    ap.add_argument('--her-candidates', type=int, default=24)
    ap.add_argument('--bootstrap', action='store_true', help='admit hindsight theorems to the library'); ap.add_argument('--boot-per-iter', type=int, default=16); ap.add_argument('--boot-max', type=int, default=200)
    ap.add_argument('--boot-min-len', type=int, default=2); ap.add_argument('--boot-max-len', type=int, default=30)
    ap.add_argument('--mw', action='store_true', help='train under the manager-worker masks and labels (implied by an actor mw)')
    ap.add_argument('--updates', type=int, default=60); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--mix', default='demo=0.4,full=0.3,her=0.3'); ap.add_argument('--buffer-max', type=int, default=40000, help='steps kept per buffer')
    ap.add_argument('--size-coef', type=float, default=0.2); ap.add_argument('--mgr-coef', type=float, default=0.2)
    ap.add_argument('--eval-every', type=int, default=5); ap.add_argument('--eval-probes', type=int, default=500); ap.add_argument('--eval-solver', default='sample'); ap.add_argument('--eval-seeds', type=int, default=1)
    ap.add_argument('--eval-time-limit', type=float, default=60.0)
    ap.add_argument('--save-every', type=int, default=5, help='iterations between writes of state.pt (the buffers are large)')
    ap.add_argument('--workers', type=int, default=0); ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=1); ap.add_argument('--resume', action='store_true')
    W.add_args(ap)
    a = ap.parse_args(argv)
    aname = AG.parse_spec(a.actor)[0]
    if aname not in AG.NEEDS_NET: ap.error('--actor: a prover that uses the network (%s)' % ', '.join(AG.NEEDS_NET))
    mw = a.mw or aname == 'mw'
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    bk, ctx, eng = W.load(a)
    data = T.build_steps(a.data, bk, ctx, G.Engine(bk.w, kinds=set(G.Kind), strict=True))
    demo = list(data['bank'].get('train', [])) + list(data['lib'].get('train', []))
    if mw: demo = H.mw_steps(demo, label_mgr=True)
    lib = None; lem_path = os.path.join(a.out, 'lemmas.json')
    if a.bootstrap or a.lemmas:
        lib = W.make_library(bk, eng, ctx, a.data, max_lemmas=a.boot_max, min_len=a.boot_min_len, max_len=a.boot_max_len)
        if a.resume and os.path.exists(lem_path): lib.load(lem_path); print('resumed %d lemmas' % lib.n_installed, flush=True)
        elif a.lemmas: lib.load(a.lemmas); print('%d lemmas installed from %s' % (lib.n_installed, a.lemmas), flush=True)
        lib.sync()
        if not os.path.exists(lem_path): save_lemmas(lib, lem_path)                    # (always: the checkpoints name this file, also when no lemma has been admitted yet -- and a run that admits none ends with an empty one)
    bank, subs = GL.train_goals(bk, a.data, 'train', a.max_ref_len)
    synth = []
    if a.synth:
        import ml_synth as MSY
        synth = MSY.load_records(a.synth, 'train', limit=a.synth_max)
    goals = GL.Goals(bk, bank, subs, synth, a.p_sub, a.p_synth, a.p_hard, a.max_ref_len, 40, a.budget_mult, synth_premises=bk.w.gid[B.LIBRARY_LIMIT] if synth else None)
    net, cfg, _ = T.load(a.init, ctx.bags)
    dev = sorted(n for n in bk.placed if BS.split_of(n) == 'dev'); dev_tasks = GL.dev_items(bk, dev)
    weights_file = WeightsFile(os.path.join(a.out, 'weights'))
    pool = Pool(a.workers, make_worker_state(bk, ctx, eng, lib, cfg), {'attempts': h_attempts, 'eval': h_eval})          # (forked before any device is touched)
    device = torch.device(a.device); net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    mix = parse_mix(a.mix); full = Buffer(a.buffer_max); her = Buffer(a.buffer_max); her_seen = set(); hist = []; start = 0; best = -1; use_count = collections.Counter(); totals = collections.Counter()
    st_path = os.path.join(a.out, 'state.pt')
    if a.resume and os.path.exists(st_path):
        st = torch.load(st_path, map_location='cpu', weights_only=False)
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); hist = st['hist']; start = st['iter'] + 1; rng.bit_generator.state = st['rng']; best = st['best']
        full = Buffer.from_state(a.buffer_max, st['full']); her = Buffer.from_state(a.buffer_max, st['her']); her_seen = set(tuple(k) for k in st['her_seen']); use_count = collections.Counter(st['use_count']); totals = collections.Counter(st['totals'])
        print('resumed at iteration %d' % start, flush=True)
    print('%d demonstration steps; %d train problems (%d easy), %d sub-theorems, %d synthetic; %d dev problems; actor %s; %d workers; %s' % (
          len(demo), len(bank), len(goals.easy), len(subs), len(synth), len(dev_tasks), a.actor, a.workers, 'library %d lemmas' % lib.n_installed if lib is not None else 'no library'), flush=True)
    MT.write_json(os.path.join(a.out, 'meta.json'), MT.run_meta(vars(a), {'world_globals': bk.w.k.num_globals(), 'dataset_manifest': MT.file_digest(os.path.join(a.data, 'manifest.json')), 'mw': mw}))
    logf = open(os.path.join(a.out, 'log.jsonl'), 'a')
    harvest = bool(a.her or a.bootstrap)
    for it in range(start, a.iters):
        t1 = time.time()
        ver = weights_file.save(net)
        tasks = [(i, goals.sample(rng)) for i in range(a.episodes)]
        common = {'weights': (weights_file.path, ver), 'actor': a.actor, 'iter': it, 'seed': a.seed, 'probes': a.probes, 'sub_probe_frac': a.sub_probe_frac, 'time_limit': a.time_limit, 'harvest': harvest,
                  'her_max': a.her_max, 'her_max_len': a.her_max_len, 'her_candidates': a.her_candidates, 'lemmas': lib.records_json() if lib is not None else None,
                  'block_digest': lib.block_digest() if lib is not None else None}
        payloads = [dict(common, tasks=c) for c in GL.chunks(tasks, max(1, math.ceil(len(tasks) / 4)))]
        results = [r for part in pool.map('attempts', payloads) for r in part]; t_roll = time.time() - t1
        st = collections.Counter(); lens = []; cands = []
        for r in results:
            if 'solved' not in r or (r.get('error') and not r.get('probes') and not r['solved']): st['errors'] += 1; continue
            src = r['source']; st['episodes'] += 1; st[src] += 1; st['probes'] += r['probes']
            if r['solved']:
                st['solved'] += 1; st[src + '_solved'] += 1; st['solved_probes'] += r['probes']
                if r.get('length') is not None: lens.append(r['length'])
                if r.get('uses'): st['solved_with_boot'] += 1; use_count.update(r['uses'])
                if r.get('steps'):
                    ss = H.mw_steps(r['steps'], label_mgr=(src != 'synth')) if mw else r['steps']
                    key = (r['name'], r['regime']) if src == 'bank' else (r['name'],)
                    if full.add(key, ss, len(r['steps'])): st['full_new'] += 1
            for h in r.get('her', []):
                rec = h['rec']; key = (rec['statement_digest'], json.dumps(rec['premises'], sort_keys=True))
                st['her_found'] += 1
                if key not in her_seen:
                    her_seen.add(key); st['her_new'] += 1
                    if a.her: her.add(key, H.mw_steps(h['steps'], label_mgr=(r['source'] != 'synth')) if mw else h['steps'], rec['length'])
                if a.bootstrap: cands.append((rec, r['name']))
            for k, v in (r.get('harvest') or {}).items(): st['harvest ' + k] += v
        # --- the library grows
        added = 0; rejected0 = collections.Counter(lib.rejected) if lib is not None else None
        if a.bootstrap and cands and lib.n_installed < a.boot_max:
            seen_d = set(); cands.sort(key=lambda t: (-t[0]['length'], t[0]['statement_digest']))
            for rec, src in cands:
                if rec['statement_digest'] in seen_d: continue
                seen_d.add(rec['statement_digest'])
                if added >= a.boot_per_iter or lib.n_installed >= a.boot_max: break
                lr, why = lib.add(rec, split='train', source=src)
                if lr is not None: added += 1
            if added: lib.sync(nets=[net]); save_lemmas(lib, lem_path)
        t_roll_total = time.time() - t1
        # --- the learner
        t2 = time.time()
        pools = [demo, full.flat(), her.flat() if a.her else []]; wts = [mix['demo'], mix['full'], mix['her']]
        upd = update(net, opt, pools, wts, a, rng, device); t_upd = time.time() - t2
        totals.update({'probes': st['probes'], 'episodes': st['episodes'], 'solved': st['solved']})
        row = {'iter': it, 'episodes': st['episodes'], 'solved': st['solved'], 'bank': [st['bank_solved'], st['bank']], 'sub': [st['sub_solved'], st['sub']], 'synth': [st['synth_solved'], st['synth']],
               'probes': st['probes'], 'probes_per_solve': (st['solved_probes'] / st['solved']) if st['solved'] else None, 'mean_len': (sum(lens) / len(lens)) if lens else None,
               'errors': st['errors'], 'full_buffer': [len(full), full.n], 'her_buffer': [len(her), her.n], 'her_found': st['her_found'], 'her_new': st['her_new'],
               'cum_probes': totals['probes'], 'seconds_rollout': round(t_roll, 1), 'seconds_update': round(t_upd, 1), 'seconds': round(time.time() - t0)}
        if lib is not None:
            row.update({'lemmas': lib.n_installed, 'lemmas_added': added, 'solved_with_boot': st['solved_with_boot'], 'boot_rejected': {k: v - rejected0.get(k, 0) for k, v in lib.rejected.items() if v - rejected0.get(k, 0)}})
        row.update({k: round(v, 5) for k, v in upd.items()})
        if a.eval_every and (it % a.eval_every == a.eval_every - 1 or it == a.iters - 1):
            ver = weights_file.save(net)
            lem_now = lib.records_json() if lib is not None else None; dig_now = lib.block_digest() if lib is not None else None             # (the library as it is after this iteration's admissions)
            payloads = [{'weights': (weights_file.path, ver), 'items': c, 'solver': a.eval_solver, 'probes': a.eval_probes, 'time_limit': a.eval_time_limit, 'seed': a.seed * 1000 + s, 'lemmas': lem_now,
                         'block_digest': dig_now} for s in range(a.eval_seeds) for c in GL.chunks(dev_tasks, max(1, math.ceil(len(dev_tasks) / 4)))]
            rows = [r for part in pool.map('eval', payloads) for r in part]
            ok = [r for r in rows if r['solved']]
            row['dev_solved'] = len(ok) / max(a.eval_seeds, 1); row['dev_total'] = len(dev_tasks)
            row['dev_solved_proved'] = sum(1 for r in ok if r['proved']) / max(a.eval_seeds, 1); row['dev_total_proved'] = sum(1 for n, _ in dev_tasks if bk.placed[n].proved)
            row['dev_mean_len'] = (sum(r['steps'] for r in ok if r.get('steps') is not None) / len(ok)) if ok else None
            row['dev_mean_probes'] = (sum(r['probes'] for r in ok) / len(ok)) if ok else None
            if row['dev_solved'] > best:
                best = row['dev_solved']; T.save(net.cpu() if device.type != 'cpu' else net, cfg, os.path.join(a.out, 'best.pt'), {'iter': it, 'algo': 'ei', 'dev_solved': best, 'lemmas': 'lemmas.json' if lib is not None else None,
                                                                                                                                  'block_digest': lib.block_digest() if lib is not None else None}); net.to(device)
        row['rss_mb'] = round(MT.rss_mb() or 0)                          # (the memory of this process after the iteration: a run that leaks shows it here, long before the machine does)
        hist.append(row); logf.write(json.dumps(row) + '\n'); logf.flush()
        print('iter %3d  solved %3d/%3d (bank %d/%d sub %d/%d synth %d/%d)  probes/solve %s  len %s  her +%d  %s  loss %.3f  roll %.0fs upd %.0fs  %ds  rss %d MB%s' % (
              it, row['solved'], row['episodes'], row['bank'][0], row['bank'][1], row['sub'][0], row['sub'][1], row['synth'][0], row['synth'][1],
              '%.0f' % row['probes_per_solve'] if row['probes_per_solve'] else '-', '%.1f' % row['mean_len'] if row['mean_len'] else '-', row['her_new'],
              ('lemmas %d (+%d)' % (row['lemmas'], row['lemmas_added'])) if lib is not None else 'no library', row.get('policy_loss', 0), t_roll_total, t_upd, row['seconds'], row['rss_mb'],
              ('  | dev %.1f/%d' % (row['dev_solved'], row['dev_total'])) if 'dev_solved' in row else ''), flush=True)
        T.save(net.cpu() if device.type != 'cpu' else net, cfg, os.path.join(a.out, 'last.pt'), {'iter': it, 'algo': 'ei', 'actor': a.actor, 'lemmas': 'lemmas.json' if lib is not None else None,
                                                                                                'block_digest': lib.block_digest() if lib is not None else None}); net.to(device)
        if (a.save_every and it % a.save_every == a.save_every - 1) or it == a.iters - 1:
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'hist': hist, 'iter': it, 'rng': rng.bit_generator.state, 'best': best, 'full': full.state(), 'her': her.state(), 'her_seen': sorted(her_seen),
                        'use_count': dict(use_count), 'totals': dict(totals)}, st_path + '.tmp'); os.replace(st_path + '.tmp', st_path)
    final = {'args': vars(a), 'history': hist, 'seconds': round(time.time() - t0), 'peak_rss_mb': MT.peak_rss_mb(), 'n_params': sum(p.numel() for p in net.parameters())}
    if lib is not None: final['library'] = lib.summary(); final['lemma_use'] = {str(k): v for k, v in use_count.most_common()}
    MT.write_json(os.path.join(a.out, 'final.json'), final)
    pool.close(); weights_file.cleanup()


if __name__ == '__main__':
    main()
