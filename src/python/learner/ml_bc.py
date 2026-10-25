"""ml_bc.py -- behaviour cloning of the policy network, in the two stages of the experiments:

    PRE-TRAINING   on the synthetic theorems (a library-only world: the proofs the generator grew)
        python3 ml_bc.py DATASET_DIR --out runs/pre_s0 --synth data/synth_v2 --synth-max 60000 --no-bank --epochs 6 --seed 0
    FINE-TUNING    on the bank's train split (the reference proofs of the hand-written problems, and the proofs of the library's lemmas)
        python3 ml_bc.py DATASET_DIR --out runs/ft_s0 --init runs/pre_s0/best.pt --lr 3e-4 --epochs 30 --seed 0

What is trained on is chosen by the flags: `--synth DIR` the synthetic train theorems (at most `--synth-max`), `--no-bank` leaves out the reference proofs of the bank's train problems, `--no-lib` the
library lemmas' proofs, `--no-strict` the second (strict-regime) reference proofs.  What is watched: the dev split of everything that is trained on (synthetic dev; bank dev and library dev), teacher-forced
decision and action accuracy; the epoch with the best `--select` metric (dev loss by default) is kept as `best.pt`.  Nothing of the bank's dev or test split is trained on, and the synthetic dev / test
records are different theorems from the train ones (a split by statement digest, and by recipe for `test_comp`).

Run directory: `best.pt` `last.pt` (checkpoints: network, configuration, extra), `state.pt` (everything to continue: `--resume`), `log.jsonl` (a line per epoch), `final.json` (arguments, history,
the metrics of the best network on every dev set and on the synthetic test sets).  Several seeds are several runs.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, math, argparse, random, collections
import numpy as np
import torch
import bank_all
import gym_env as G
import ml_data as D, ml_model as M, ml_train as T


def evaluate_sets(net, sets, size=64):
    out = {}
    for name, st in sets.items():
        if st: ev = T.evaluate(net, st, size); out[name] = {k: ev[k] for k in ('loss', 'acc', 'action_acc', 'value_loss', 'decisions', 'actions')}
    return out


def train(net, train_steps, dev_sets, a, out, device, log=print, resume=None):
    """the training loop: AdamW with a linear warm-up and a cosine decay over all the epochs; the best epoch by `a.select` is saved as best.pt, the last as last.pt"""
    rng = random.Random(a.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    nb = max(1, math.ceil(len(train_steps) / a.batch)); total = a.epochs * nb; warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, t / max(total, 1)))) * 0.95 + 0.05))
    hist = []; best = None; best_ep = -1; start = 0; t0 = time.time()
    cfg = net.cfg_for_save
    sp = os.path.join(out, 'state.pt')
    if resume and os.path.exists(sp):
        st = torch.load(sp, map_location='cpu', weights_only=False)
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); hist = st['hist']; best = st['best']; best_ep = st['best_ep']; start = st['epoch'] + 1; rng.setstate(st['rng'])
        log('resumed after epoch %d' % st['epoch'])
    logf = open(os.path.join(out, 'log.jsonl'), 'a')
    aux = {'size': a.size_coef, 'mgr': a.mgr_coef}
    for ep in range(start, a.epochs):
        net.train(); tl = tn = tc = 0; ts = 0.0; nbat = 0
        for bi in T.batches(train_steps, a.batch, rng):
            batch = D.collate([train_steps[j] for j in bi])
            res = net(batch, aux=aux)
            opt.zero_grad(); res['loss'].backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += float(res['policy_loss']) * res['n']; tn += res['n']; tc += res['correct']; ts += float(res['size_loss']); nbat += 1
        row = {'epoch': ep, 'train_loss': tl / max(tn, 1), 'train_acc': tc / max(tn, 1), 'train_size_loss': ts / max(nbat, 1), 'seconds': round(time.time() - t0)}
        row.update(evaluate_sets(net, dev_sets))
        hist.append(row)
        key = row.get(a.select_set, {}).get('loss', row['train_loss']) if a.select == 'loss' else -row.get(a.select_set, {}).get('action_acc', row['train_acc'])
        if best is None or key < best: best = key; best_ep = ep; T.save(net, cfg, os.path.join(out, 'best.pt'), {'epoch': ep, 'select': a.select, 'key': key})
        T.save(net, cfg, os.path.join(out, 'last.pt'), {'epoch': ep})
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'hist': hist, 'best': best, 'best_ep': best_ep, 'epoch': ep, 'rng': rng.getstate()}, sp)
        logf.write(json.dumps(row) + '\n'); logf.flush()
        log('epoch %2d  train loss %.3f acc %.3f | %s | %ds' % (ep, row['train_loss'], row['train_acc'], ' | '.join('%s loss %.3f acc %.3f act %.3f' % (n, row[n]['loss'], row[n]['acc'], row[n]['action_acc']) for n in dev_sets if n in row), row['seconds']))
        if a.patience and ep - best_ep >= a.patience: log('no improvement for %d epochs: stop' % a.patience); break
    return hist, best_ep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('--out', required=True); ap.add_argument('--init', default='')
    ap.add_argument('--synth', default='', help='a merged synthetic dataset directory'); ap.add_argument('--synth-max', type=int, default=60000); ap.add_argument('--synth-dev-max', type=int, default=3000)
    ap.add_argument('--no-bank', action='store_true'); ap.add_argument('--no-lib', action='store_true'); ap.add_argument('--no-strict', action='store_true')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--d', type=int, default=128); ap.add_argument('--layers', type=int, default=3); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--ff', type=int, default=384); ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--size-coef', type=float, default=0.2); ap.add_argument('--mgr-coef', type=float, default=0.2)
    ap.add_argument('--select', default='loss', choices=('loss', 'action_acc')); ap.add_argument('--select-set', default='', help='the dev set that picks the epoch (default: dev_bank when the bank is trained on, else dev_synth)')
    ap.add_argument('--patience', type=int, default=0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--workers', type=int, default=0, help='processes that extract the synthetic proofs'); ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); sys.setrecursionlimit(100000)
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    bk = bank_all.load(); ctx = D.DataContext(bk.w)
    print('world %.0fs' % (time.time() - t0), flush=True)
    data = T.build_steps(a.data, bk, ctx, G.Engine(bk.w, kinds=set(G.Kind), strict=True))
    train_steps = []; dev_sets = {}; extra_sets = {}
    if not a.no_bank:
        tr = list(data['bank'].get('train', []))
        if a.no_strict: tr = [s for s in tr if not s.name.endswith('/strict')]
        train_steps += tr; dev_sets['dev_bank'] = list(data['bank'].get('dev', []))
    if not a.no_lib:
        train_steps += list(data['lib'].get('train', [])); dev_sets['dev_lib'] = list(data['lib'].get('dev', []))
    if a.synth:
        import ml_synth as MS
        train_steps += MS.cached_steps(a.synth, 'train', bk, ctx, limit=a.synth_max, workers=a.workers)
        dev_sets['dev_synth'] = MS.cached_steps(a.synth, 'dev', bk, ctx, limit=a.synth_dev_max, workers=a.workers)
        for sp in ('test_iid', 'test_comp'): extra_sets['synth_' + sp] = MS.cached_steps(a.synth, sp, bk, ctx, limit=a.synth_dev_max, workers=a.workers)
    if not train_steps: sys.exit('nothing to train on')
    if not a.select_set: a.select_set = 'dev_bank' if (not a.no_bank and dev_sets.get('dev_bank')) else ('dev_synth' if 'dev_synth' in dev_sets else next(iter(dev_sets), ''))
    print('train steps %d; dev sets %s; epoch picked by %s of %s' % (len(train_steps), {k: len(v) for k, v in dev_sets.items()}, a.select, a.select_set), flush=True)
    if a.init:
        net, cfg, _ = T.load(a.init, ctx.bags); net.train()
    else:
        cfg = {'d': a.d, 'layers': a.layers, 'heads': a.heads, 'ff': a.ff, 'dropout': a.dropout, 'lr': a.lr, 'wd': a.wd, 'batch': a.batch, 'epochs': a.epochs, 'seed': a.seed}
        net = T.make_net(ctx.bags, cfg)
    net.cfg_for_save = cfg
    device = torch.device(a.device); net.to(device)
    print('parameters %d; device %s' % (sum(p.numel() for p in net.parameters()), device), flush=True)
    hist, best_ep = train(net, train_steps, dev_sets, a, a.out, device, log=lambda m: print(m, flush=True), resume=a.resume)
    best, _, extra = T.load(os.path.join(a.out, 'best.pt'), ctx.bags); best.to(device)
    final = {n: m for n, m in evaluate_sets(best, {**dev_sets, **extra_sets}).items()}
    json.dump({'args': vars(a), 'config': cfg, 'best_epoch': best_ep, 'history': hist, 'best': final, 'seconds': round(time.time() - t0)}, open(os.path.join(a.out, 'final.json'), 'w'), indent=1)
    print('best epoch %d: %s' % (best_ep, {n: (round(m['loss'], 3), round(m['action_acc'], 3)) for n, m in final.items()}))


if __name__ == '__main__':
    main()
