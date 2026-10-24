"""ml_rl.py -- reinforcement learning in BankEnv.

The policy plays the environment (one decision at a time, sampled), the kernel checks the finished proofs, and the network is updated from what happened:

  mode 'reinforce'   REINFORCE with a per-problem moving-average baseline.  Every decision of an episode is pushed up in proportion to (return - baseline); the decisions that assembled an action the engine
                     REFUSED are pushed down (`--refused-weight`).  The return is 1 for a kernel-checked proof, minus `--len-penalty` per action (shorter proofs are better), 0 otherwise.
  mode 'filter'      rejection-sampling fine-tuning (expert iteration by rollouts): behaviour cloning on the episodes that FINISHED the proof (the whole growing buffer of them), nothing learned from failures.

Both are mixed with behaviour cloning on the train demonstrations (`--bc-coef`) so the policy cannot drift away from what the demonstrations did.  Only TRAIN problems are played (the bank problems of the train
split in both regimes, and library lemmas of the train split: `--lemma-frac` of the episodes); dev problems are only used to watch progress (sampled rollouts at low temperature, `--dev-samples` per problem).
Nothing in the loop sees a dev or test problem's statement or proof.

    python3 ml_rl.py DATASET_DIR --init bc.pt --out rl.pt --iters 60 --episodes 24 --mode reinforce
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, random, math, argparse, collections
import numpy as np
import torch
import bank_all, bank_split as BS
import gym_env as G
import ml_features as F, ml_data as D, ml_model as M, ml_train as T
import bank_env as BE


def rollout(env, net, ctx, gtab, name, regime, rng, temperature=1.0, greedy=False, record=None):
    """one episode of `name`: (solved, [(Step, taken)], the final info).  A Step is one assembled action (the goal and its decisions); `taken` says whether the engine accepted it (True), refused it (False) or
    the episode ended before the action was complete (None).  `record`: a list that
    gets the log-probability (under the policy, before any temperature) of every decision made, in order -- what the tests compare with the network's own batched computation."""
    obs, info = env.reset(options={'problem': name, 'regime': regime})
    dec = None; last_gf = None; carry = None
    while True:
        ch = info.get('choice')
        if ch is None: break
        gf = info['goal_feat']
        if gf is not last_gf:
            dec = M.Decoder(net, ctx, None, gtab, gf=gf, ov=info['overlap']); last_gf = gf
        if info['in_action'] == 0: carry = dec.root()
        logp, oe, hs = dec.scores(carry, ch)
        if greedy: i = int(torch.argmax(logp))
        else:
            pr = torch.softmax(logp / temperature, dim=-1).numpy().astype(np.float64); pr /= pr.sum()
            i = int(rng.choice(len(pr), p=pr))
        if record is not None: record.append(float(logp[i]))
        carry = dec.child(carry, ch, i, oe, hs)
        obs, r, term, trunc, info = env.step(i)
        if term or trunc: break
    steps = []
    for gf, ov, decisions, taken in env.trajectory:
        if not decisions: continue
        steps.append((D.make_step(ctx, name, len(steps), -1, None, decisions, 0, gf=gf, ov=ov), taken))
    return env.solved, steps, info


def regime_for(bk, name, rng):
    pl = bk.placed[name]
    if not pl.problem.strict or pl.premises('strict') == pl.premises('prefix'): return 'prefix'          # (the strict regime is a task only for a problem with a self-contained reference proof)
    return 'prefix' if rng.random() < 0.5 else 'strict'


def dev_check(env, net, ctx, names, rng, samples=3, temperature=0.5):
    """pass@samples of sampled low-temperature rollouts on `names` (prefix regime)"""
    solved = []
    with torch.no_grad(): gtab = net.global_table()
    for n in names:
        for _ in range(samples):
            ok, _, _ = rollout(env, net, ctx, gtab, n, 'prefix', rng, temperature)
            if ok: solved.append(n); break
    return solved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('--init', required=True); ap.add_argument('--out', default='rl.pt')
    ap.add_argument('--iters', type=int, default=60); ap.add_argument('--episodes', type=int, default=24); ap.add_argument('--mode', default='reinforce', choices=('reinforce', 'filter'))
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--bc-coef', type=float, default=0.5); ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-actions', type=int, default=40); ap.add_argument('--max-decisions', type=int, default=400); ap.add_argument('--refused-weight', type=float, default=0.3)
    ap.add_argument('--len-penalty', type=float, default=0.005); ap.add_argument('--lemma-frac', type=float, default=0.35)
    ap.add_argument('--max-ref-len', type=int, default=30, help='only problems and lemmas whose reference proof has at most this many actions are played (a longer one cannot be finished within the episode limit)')
    ap.add_argument('--batch', type=int, default=32); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--eval-every', type=int, default=10); ap.add_argument('--dev-samples', type=int, default=3)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); sys.setrecursionlimit(100000)
    t0 = time.time()
    bk = bank_all.load(); eng = G.Engine(bk.w, kinds=set(G.Kind), strict=False)
    ctx = D.DataContext(bk.w)
    data = T.build_steps(a.data, bk, ctx, G.Engine(bk.w, kinds=set(G.Kind), strict=True))
    demo = list(data['bank'].get('train', [])) + list(data['lib'].get('train', []))
    net, cfg, _ = T.load(a.init, ctx.bags); net.train()
    ref_len = {r['name']: r['demonstration']['length'] for r in T.load_dataset(a.data)[0]['problems'] if 'demonstration' in r}          # (a statement-only problem has no reference proof: not played here)
    train_names = sorted(n for n in bk.placed if BS.split_of(n) == 'train' and n in ref_len and ref_len[n] <= a.max_ref_len); dev_names = sorted(n for n in bk.placed if BS.split_of(n) == 'dev')
    lemmas = BE.library_lemmas(bk, a.data, split='train', max_len=a.max_ref_len); lemma_names = sorted(lemmas)
    env = BE.BankEnv(bk, train_names, ctx=ctx, engine=eng, max_actions=a.max_actions, max_decisions=a.max_decisions, seed=a.seed, lemmas=lemmas)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    base = collections.defaultdict(float); seen = collections.Counter(); won = collections.Counter(); hist = []
    buf = []                                                  # steps of the finished episodes (filter mode)
    print('setup %.0fs; %d demonstration steps; %d train problems, %d train lemmas' % (time.time() - t0, len(demo), len(train_names), len(lemma_names)), flush=True)
    for it in range(a.iters):
        net.eval()
        with torch.no_grad(): gtab = net.global_table()
        eps = []
        for _ in range(a.episodes):
            if lemma_names and rng.random() < a.lemma_frac: name = lemma_names[int(rng.integers(len(lemma_names)))]; regime = 'prefix'
            else: name = train_names[int(rng.integers(len(train_names)))]; regime = regime_for(bk, name, rng)
            solved, st, info = rollout(env, net, ctx, gtab, name, regime, rng, a.temperature)
            eps.append((name, regime, solved, st, info))
        net.train()
        steps = []; weights = []; n_bank = s_bank = n_lem = s_lem = 0
        for name, regime, solved, st, info in eps:
            is_lem = name in lemmas
            if is_lem: n_lem += 1; s_lem += int(solved)
            else: n_bank += 1; s_bank += int(solved)
            R = (1.0 - a.len_penalty * info['actions']) if solved else 0.0
            key = name + ('' if regime == 'prefix' else '/strict')
            b = base[key]; adv = R - b; base[key] = 0.8 * b + 0.2 * R; seen[key] += 1; won[key] += int(solved)
            if a.mode == 'reinforce':
                for s, taken in st: steps.append(s); weights.append(-a.refused_weight if taken is False else adv)
            elif solved:
                buf.extend(s for s, taken in st if taken)
        if a.mode == 'filter':
            if buf:
                for _ in range(2):
                    idx = rng.choice(len(buf), size=min(len(buf), a.batch * 2), replace=False)
                    res = net(D.collate([buf[j] for j in idx]))
                    opt.zero_grad(); res['policy_loss'].backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        elif steps:
            order = rng.permutation(len(steps))
            for i in range(0, len(steps), a.batch * 2):
                sel = order[i:i + a.batch * 2]
                res = net(D.collate([steps[j] for j in sel], weights=[weights[j] for j in sel]))
                opt.zero_grad(); res['policy_loss'].backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        for _ in range(2):                                    # behaviour cloning on the demonstrations
            idx = rng.choice(len(demo), size=a.batch, replace=False)
            res = net(D.collate([demo[j] for j in idx])); loss = a.bc_coef * res['policy_loss']
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        row = {'iter': it, 'bank': [s_bank, n_bank], 'lemmas': [s_lem, n_lem], 'steps': len(steps), 'buffer': len(buf), 'seconds': round(time.time() - t0)}
        msg = 'iter %3d  bank %2d/%2d  lemmas %2d/%2d  %ds' % (it, s_bank, n_bank, s_lem, n_lem, row['seconds'])
        if a.eval_every and (it % a.eval_every == a.eval_every - 1 or it == a.iters - 1):
            net.eval(); ok = dev_check(env, net, ctx, dev_names, rng, a.dev_samples); net.train()
            row['dev_solved'] = ok; msg += '  | dev pass@%d: %d/%d %s' % (a.dev_samples, len(ok), len(dev_names), ok)
        hist.append(row); print(msg, flush=True)
        T.save(net, cfg, a.out, {'iter': it, 'mode': a.mode})
    json.dump({'args': vars(a), 'history': hist, 'won': dict(won), 'seen': dict(seen)}, open(os.path.splitext(a.out)[0] + '_history.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
