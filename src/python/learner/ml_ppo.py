"""ml_ppo.py -- proximal policy optimisation in the problem bank's Gymnasium environment (bank_env.BankEnv).

The agent plays proof attempts: at every DECISION of the assembler (which kind of action, which lemma, which term-token, ...) it picks an option; the engine probes the finished action, the kernel checks a
finished proof, and the only reward is a kernel-checked proof (minus a small price per action taken and per action the engine refuses).  One time step of the RL problem is one ACTION (the engine acts once);
the decisions that assemble it are the micro-steps of the policy, which gets the action's advantage.

    reward      solve_reward at the action that finishes the proof, -action_penalty for every action taken, -fail_penalty for every action the engine refused
    critic      V(goal, fraction of the budget left, open goals / 8): a head of the same network, regression on the return
    advantage   GAE(gamma, lambda) over the actions of an episode (an episode ends by the proof or by the budget; the value after the end is 0, the budget being part of the critic's input), normalised over
                the batch
    policy      the clipped surrogate of PPO on each decision, ratio = pi_new / pi_old of the DECISION, with the advantage of the action it belongs to; an entropy bonus; a behaviour-cloning term on the
                demonstrations (`--bc-coef`) that keeps the policy near what the proofs of the train split do
    goals       the train problems of the bank (with a reference proof of at most `--max-ref-len` actions; the others only with probability `--p-hard`), the sub-theorems of train proofs (`--p-sub`: the
                hindsight dataset, easier goals with proofs of their own length as a budget hint) and synthetic theorems (`--p-synth`).  Nothing from the dev or test split is read in training.
    evaluation  every `--eval-every` iterations: the dev problems by the protocol of ml_protocol (kernel-checked solves within a probe budget by SAMPLING), which is also the number the log shows

    python3 ml_ppo.py DATASET_DIR --init bc.pt --out runs/ppo_s0 --iters 200 --episodes 64 --workers 4 --seed 0

The run directory gets `last.pt` (the network) and `state.pt` (network, optimiser, history, random state: `--resume` continues from it), `log.jsonl` (one JSON line per iteration) and `final.json`.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, math, argparse, collections, random
import numpy as np
import torch
import bank_split as BS
import gym_env as G
import ml_features as F, ml_data as D, ml_model as M, ml_train as T, ml_protocol as PR, ml_world as W, ml_meta as MT
import ml_goals as GL
from ml_goals import Goals, regime_for, train_goals, chunks, dev_items          # (they moved to ml_goals; the names stay here for the callers that import them from this module)
import bank_env as BE
from ml_pool import Pool, WeightsFile


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ episodes
def run_episode(env, net, ctx, gtab, rng, task, temperature=1.0):
    """one on-policy episode of a task: a dict {'name', 'source', 'solved', 'actions', 'refused', 'steps': [(Step, taken)]} (every Step with its old_logp and cfeat), or None when there is nothing to prove"""
    try:
        info, logps = PR.play(env, net, ctx, gtab, rng, dict(task['options']), temperature)
    except ValueError:
        return None
    budget0 = max(1, env.budget)
    if sum(len(d) for _, _, d, _ in env.trajectory) != len(logps): raise RuntimeError('the decisions of the trajectory and the log-probabilities disagree')
    steps = []; k = 0
    for (gf, ov, decisions, taken), (left, goals) in zip(env.trajectory, env.meta):
        n = len(decisions); lp = logps[k:k + n]; k += n
        if not decisions or taken is None: continue                       # (an action that was never completed was never answered by the engine: no signal)
        st = D.make_step(ctx, task['name'], len(steps), -1, None, decisions, 0, gf=gf, ov=ov, old_logp=lp, cfeat=np.asarray([left / budget0, min(goals, 16) / 8.0], dtype=np.float32))
        st.goal.key = None
        steps.append((st, taken))
    return {'name': task['name'], 'source': task['source'], 'solved': bool(env.solved), 'actions': env.actions, 'refused': env.fails, 'steps': steps}


def gae(rewards, values, gamma, lam):
    """advantages and returns of one episode: `values[t]` the critic at the state of step t; the value after the last step is 0 (the episode is over)"""
    n = len(rewards); adv = np.zeros(n, dtype=np.float32); last = 0.0
    for t in range(n - 1, -1, -1):
        nxt = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last; adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


def batch_values(net, steps, size=64):
    """the critic's values of the goals of Steps (no gradient), in the order of `steps`"""
    out = []
    net.eval()
    for i in range(0, len(steps), size):
        v, _, _ = net.goal_heads(D.collate(steps[i:i + size])); out.append(v)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def episodes_to_batch(net, eps, a, rng):
    """flatten the episodes into Steps that carry `ret` and a list of advantages; returns (steps, advantages, stats).  The critic is evaluated by `net` (the learner's network at collection time)."""
    steps = []; advs = []; stats = collections.Counter()
    for e in eps:
        sts = [s for s, _ in e['steps']]
        if not sts: continue
        v = batch_values(net, sts)
        r = np.zeros(len(sts), dtype=np.float32)
        for t, (s, taken) in enumerate(e['steps']): r[t] = -a.action_penalty if taken else -a.fail_penalty
        if e['solved']: r[-1] += a.solve_reward
        adv, ret = gae(r, v, a.gamma, a.lam)
        for s, rt in zip(sts, ret): s.ret = float(rt)
        steps.extend(sts); advs.append(adv)
        stats['episodes'] += 1; stats['solved'] += int(e['solved']); stats[e['source']] += 1; stats[e['source'] + '_solved'] += int(e['solved'])
        stats['actions'] += e['actions']; stats['refused'] += e['refused']
    adv = np.concatenate(advs) if advs else np.zeros(0, dtype=np.float32)
    return steps, adv, stats


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the learner
def ppo_update(net, opt, steps, adv, demo, a, rng, device):
    """a few epochs of PPO on the steps (advantages `adv`, normalised here); returns the averaged metrics"""
    if len(steps) == 0: return {}
    adv = (adv - adv.mean()) / (adv.std() + 1e-6) if len(adv) > 1 else adv
    net.eval(); sums = collections.Counter(); n_mb = 0                    # (eval mode: the ratio pi_new / pi_old compares the same function -- with dropout on, the first minibatch would already be off-policy)
    idx_all = np.arange(len(steps))
    stop = False
    for epoch in range(a.epochs):
        rng.shuffle(idx_all)
        for i in range(0, len(idx_all), a.minibatch):
            sel = idx_all[i:i + a.minibatch]
            batch = D.collate([steps[j] for j in sel], weights=adv[sel])
            res = net(batch, ppo={'clip': a.clip}, aux={})
            loss = res['policy_loss'] + a.vf_coef * res['critic_loss'] - a.ent_coef * res['entropy']
            if a.bc_coef > 0 and demo:
                bsel = rng.choice(len(demo), size=min(a.bc_batch, len(demo)), replace=False)
                bc = net(D.collate([demo[j] for j in bsel]), aux={'size': 0.2, 'mgr': 0.2}); loss = loss + a.bc_coef * bc['loss']; sums['bc_loss'] += T.num(bc['policy_loss'])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm); opt.step()
            n_mb += 1
            for k in ('policy_loss', 'critic_loss'): sums[k] += T.num(res[k])
            sums['entropy'] += T.num(res['entropy']); sums['clip_fraction'] += res['clip_fraction']; sums['approx_kl'] += res['approx_kl']
            if a.target_kl and res['approx_kl'] > 1.5 * a.target_kl: stop = True; break
        if stop: break
    out = {k: v / max(n_mb, 1) for k, v in sums.items()}; out['minibatches'] = n_mb
    return out


def make_worker_state(bk, ctx, cfg, args, lib=None):
    """what a worker holds: its environment over the (shared) world and its own copy of the network (`lib`: the lemmas an earlier run proved, a frozen library the tasks are widened by)"""
    def init():
        torch.set_num_threads(1)
        eng = G.Engine(bk.w, kinds=set(G.Kind), strict=False)
        env = BE.BankEnv(bk, [], ctx=ctx, engine=eng, max_actions=args.max_actions, max_decisions=args.max_decisions, seed=0, library=lib)
        net = T.make_net(ctx.bags, cfg); net.eval()
        return {'env': env, 'net': net, 'cache': {}, 'ctx': ctx, 'bk': bk}
    return init


def h_episodes(st, payload):
    path, version = payload['weights']
    WeightsFile.load_into(st['net'], path, version, st['cache'])
    net = st['net']
    with torch.no_grad(): gtab = net.global_table()
    out = []
    for i, task in enumerate(payload['tasks']):
        rng = np.random.default_rng(PR.stable_seed('ppo', payload['seed'], i, task['name']))
        e = run_episode(st['env'], net, st['ctx'], gtab, rng, task, payload.get('temperature', 1.0))
        if e is not None: out.append(e)
    return out


def h_eval(st, payload):
    path, version = payload['weights']
    WeightsFile.load_into(st['net'], path, version, st['cache'])
    net = st['net']
    with torch.no_grad(): gtab = net.global_table()
    rows = []
    for name, regime in payload['items']:
        row = PR.sample_row(st['env'], net, st['ctx'], gtab, name, regime, payload['probes'], payload['max_actions'], payload['seed'], payload.get('temperature', 1.0))
        rows.append(PR.annotate(row, st['bk'], 'ppo-sample', payload['seed']))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('--init', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=200); ap.add_argument('--episodes', type=int, default=64); ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--minibatch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--clip', type=float, default=0.2); ap.add_argument('--gamma', type=float, default=1.0); ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--vf-coef', type=float, default=0.5); ap.add_argument('--ent-coef', type=float, default=0.01); ap.add_argument('--bc-coef', type=float, default=0.3); ap.add_argument('--bc-batch', type=int, default=32)
    ap.add_argument('--target-kl', type=float, default=0.05); ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--max-actions', type=int, default=40); ap.add_argument('--max-decisions', type=int, default=600); ap.add_argument('--max-ref-len', type=int, default=30)
    ap.add_argument('--solve-reward', type=float, default=1.0); ap.add_argument('--action-penalty', type=float, default=0.01); ap.add_argument('--fail-penalty', type=float, default=0.005)
    ap.add_argument('--p-sub', type=float, default=0.25); ap.add_argument('--p-synth', type=float, default=0.1); ap.add_argument('--p-hard', type=float, default=0.1); ap.add_argument('--budget-mult', type=float, default=3.0)
    ap.add_argument('--synth', default='', help='a directory of the merged synthetic dataset (train.jsonl.gz); its train theorems are goals and, with --bc-synth, demonstrations')
    ap.add_argument('--synth-max', type=int, default=20000); ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--eval-every', type=int, default=10); ap.add_argument('--eval-probes', type=int, default=500); ap.add_argument('--eval-seeds', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    W.add_args(ap)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    bk, ctx, _eng = W.load(a)
    data = T.build_steps(a.data, bk, ctx, G.Engine(bk.w, kinds=set(G.Kind), strict=True))
    demo = list(data['bank'].get('train', [])) + list(data['lib'].get('train', []))
    lib = W.load_library(a, bk, _eng, ctx, a.data)                          # (a frozen library: lemmas an earlier run proved, installed before the workers are forked)
    bank, subs = train_goals(bk, a.data, 'train', a.max_ref_len)
    synth = []
    if a.synth:
        import ml_synth as MS
        synth = MS.load_records(a.synth, 'train', limit=a.synth_max)
    goals = Goals(bk, bank, subs, synth, a.p_sub, a.p_synth, a.p_hard, a.max_ref_len, a.max_actions, a.budget_mult, synth_premises=bk.w.gid[BS_LIMIT()] if synth else None)
    net0, cfg, _ = T.load(a.init, ctx.bags)
    dev = sorted(n for n in bk.placed if BS.split_of(n) == 'dev')
    weights = WeightsFile(os.path.join(a.out, 'weights'))
    pool = Pool(a.workers, make_worker_state(bk, ctx, cfg, a, lib), {'episodes': h_episodes, 'eval': h_eval})        # (forked before any device is touched)
    device = torch.device(a.device); net = net0.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    hist = []; start = 0; best = -1; cum_probes = 0
    st_path = os.path.join(a.out, 'state.pt')
    if a.resume and os.path.exists(st_path):
        st = torch.load(st_path, map_location='cpu', weights_only=False)
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); hist = st['hist']; start = st['iter'] + 1; rng.bit_generator.state = st['rng']; best = st.get('best', -1); cum_probes = st.get('cum_probes', 0)
        print('resumed at iteration %d' % start, flush=True)
    print('%d demonstration steps; %d train problems (%d easy), %d sub-theorems, %d synthetic; %d dev problems; %d workers' % (len(demo), len(bank), len(goals.easy), len(subs), len(synth), len(dev), a.workers), flush=True)
    MT.write_json(os.path.join(a.out, 'meta.json'), MT.run_meta(vars(a), {'world_globals': bk.w.k.num_globals(), 'dataset_manifest': MT.file_digest(os.path.join(a.data, 'manifest.json')), 'lemma_block': lib.block_digest() if lib is not None else None}))
    logf = open(os.path.join(a.out, 'log.jsonl'), 'a')
    for it in range(start, a.iters):
        t1 = time.time()
        ver = weights.save(net)
        tasks = [goals.sample(rng) for _ in range(a.episodes)]
        parts = pool.map('episodes', [{'weights': (weights.path, ver), 'tasks': c, 'seed': a.seed * 100003 + it * 997 + j, 'temperature': a.temperature} for j, c in enumerate(chunks(tasks, max(a.workers, 1)))])
        eps = [e for part in parts for e in part]; t_roll = time.time() - t1
        net.eval()
        steps, adv, st = episodes_to_batch(net, eps, a, rng)
        upd = ppo_update(net, opt, steps, adv, demo, a, rng, device)
        probes = st['actions'] + st['refused']; cum_probes += probes                          # (a probe is an action the engine took or refused: the unit of cost of every method)
        ok_eps = [e for e in eps if e['solved']]
        row = {'iter': it, 'episodes': st['episodes'], 'solved': st['solved'], 'bank': [st['bank_solved'], st['bank']], 'sub': [st['sub_solved'], st['sub']], 'synth': [st['synth_solved'], st['synth']],
               'actions': st['actions'], 'refused': st['refused'], 'probes': probes, 'cum_probes': cum_probes, 'steps': len(steps), 'seconds_rollout': round(t_roll, 1), 'seconds': round(time.time() - t0),
               'probes_per_solve': (sum(e['actions'] + e['refused'] for e in ok_eps) / len(ok_eps)) if ok_eps else None, 'mean_len': (sum(e['actions'] for e in ok_eps) / len(ok_eps)) if ok_eps else None}
        row.update({k: round(v, 5) for k, v in upd.items()})
        if a.eval_every and (it % a.eval_every == a.eval_every - 1 or it == a.iters - 1):
            ver = weights.save(net); items = dev_items(bk, dev)
            rows = [r for part in pool.map('eval', [{'weights': (weights.path, ver), 'items': c, 'probes': a.eval_probes, 'max_actions': a.max_actions, 'seed': a.seed + s * 1000, 'temperature': 1.0}
                                                    for s in range(a.eval_seeds) for c in chunks(items, max(a.workers, 1))]) for r in part]
            ok = [r for r in rows if r['solved'] and r['probes'] <= a.eval_probes]
            row['dev_solved'] = len(ok) / max(a.eval_seeds, 1); row['dev_total'] = len(items)
            row['dev_solved_proved'] = sum(1 for r in ok if r['proved']) / max(a.eval_seeds, 1); row['dev_total_proved'] = sum(1 for n, _ in items if bk.placed[n].proved)
            row['dev_mean_len'] = (sum(r['steps'] for r in ok if r.get('steps') is not None) / len(ok)) if ok else None
            row['dev_mean_probes'] = (sum(r['probes'] for r in ok) / len(ok)) if ok else None
            if row['dev_solved'] > best:
                best = row['dev_solved']; T.save(net.cpu() if device.type != 'cpu' else net, cfg, os.path.join(a.out, 'best.pt'), {'iter': it, 'algo': 'ppo', 'dev_solved': best}); net.to(device)
        hist.append(row); logf.write(json.dumps(row) + '\n'); logf.flush()
        print('iter %3d  solved %2d/%2d (bank %d/%d sub %d/%d)  ent %.3f kl %.4f clip %.2f  roll %.0fs  %ds%s' % (it, row['solved'], row['episodes'], row['bank'][0], row['bank'][1], row['sub'][0], row['sub'][1],
              row.get('entropy', 0), row.get('approx_kl', 0), row.get('clip_fraction', 0), t_roll, row['seconds'], ('  | dev %.1f/%d' % (row['dev_solved'], row['dev_total'])) if 'dev_solved' in row else ''), flush=True)
        T.save(net.cpu() if device.type != 'cpu' else net, cfg, os.path.join(a.out, 'last.pt'), {'iter': it, 'algo': 'ppo', 'block_digest': lib.block_digest() if lib is not None else None}); net.to(device)
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'hist': hist, 'iter': it, 'rng': rng.bit_generator.state, 'best': best, 'cum_probes': cum_probes}, st_path + '.tmp'); os.replace(st_path + '.tmp', st_path)
    MT.write_json(os.path.join(a.out, 'final.json'), {'args': vars(a), 'history': hist, 'seconds': round(time.time() - t0), 'peak_rss_mb': MT.peak_rss_mb(), 'n_params': sum(p.numel() for p in net.parameters())})
    pool.close(); weights.cleanup()


def BS_LIMIT():
    import bank as B
    return B.LIBRARY_LIMIT


if __name__ == '__main__':
    main()
