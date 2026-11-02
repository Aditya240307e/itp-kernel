"""ml_train.py -- behaviour cloning: train the policy network to take the actions of the reference proofs.

    python3 ml_train.py DATASET_DIR --out model.pt --epochs 30

The training data are the demonstrations of the dataset's TRAIN split: the bank problems' reference proofs (prefix regime, and the strict-regime proofs where there are two) and the proofs of the library
lemmas that `bank_split.lemma_split` puts in train.  Dev (bank problems and lemmas) is used to pick the epoch; test is never read here.  Every demonstration is replayed in the engine and each of its actions is
decomposed into the decisions of the policy assembler (ml_data); a decision is a classification over the options the engine offers at that point.

Metrics (teacher forced): decision accuracy (and per kind of decision), ACTION accuracy (all the decisions of an action right), value-head error.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, pickle, hashlib, random, collections, argparse
import numpy as np
import torch
import itp_env_lnl as E
import bank_all, bank_split as BS
import gym_env as G
import ml_features as F, ml_data as D, ml_model as M


def load_dataset(ddir):
    bank = json.load(open(os.path.join(ddir, 'bank.json')))
    lib = json.load(open(os.path.join(ddir, 'library.json'))) if os.path.exists(os.path.join(ddir, 'library.json')) else None
    return bank, lib


def build_steps(ddir, bk, ctx, eng, cache=None, progress=print):
    """{'bank': {split: [Step]}, 'lib': {split: [Step]}} for the dataset in `ddir` (cached in a pickle named by the manifest digest)"""
    man = open(os.path.join(ddir, 'manifest.json')).read()
    key = hashlib.sha256((man + 'v5|' + ','.join(sorted(bk.placed))).encode()).hexdigest()[:12]          # (a world with only some of the problems -- a test, a smoke run -- has steps of its own)
    cpath = cache or os.path.join(ddir, 'steps_%s.pkl' % key)
    if os.path.exists(cpath):
        try:
            with open(cpath, 'rb') as f: return pickle.load(f)
        except Exception as e: progress('the cache %s is unreadable (%s): rebuilding' % (cpath, type(e).__name__))
    bank, lib = load_dataset(ddir)
    out = {'bank': collections.defaultdict(list), 'lib': collections.defaultdict(list), 'stats': collections.Counter()}
    t0 = time.time()
    for rec in bank['problems']:
        if rec['name'] not in bk.placed: continue
        pl = bk.placed[rec['name']]; sp = rec['split']
        for dkey, regime in (('demonstration', 'prefix'), ('demonstration_strict', 'strict')):
            if dkey not in rec: continue
            st = D.extract(ctx, eng, pl.stmt, pl.premises(regime), rec[dkey]['steps'], name=rec['name'] + ('' if regime == 'prefix' else '/strict'), stats=out['stats'])
            out['bank'][sp].extend(st)
    progress('bank steps: %s (%.0fs)' % ({s: len(v) for s, v in out['bank'].items()}, time.time() - t0))
    if lib is not None:
        for l in lib['lemmas']:
            if l['bank_duplicate']: continue
            gid = l['gid']; stmt = bk.w.T(bk.k.global_info(gid)[1])
            st = D.extract(ctx, eng, stmt, gid, l['steps'], name=l['name'], stats=out['stats'])
            out['lib'][l['split']].extend(st)
        progress('library steps: %s (%.0fs)' % ({s: len(v) for s, v in out['lib'].items()}, time.time() - t0))
    out['bank'] = dict(out['bank']); out['lib'] = dict(out['lib'])
    for part in ('bank', 'lib'):
        for steps in out[part].values():
            for s in steps: s.goal.key = None                          # (the engine's key of the goal holds kernel handles: nothing to keep in a file)
    tmp = cpath + '.tmp'
    with open(tmp, 'wb') as f: pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cpath)
    return out


def num(x):
    """a Python float from a number or a tensor (detached: no warning about leaving the graph)"""
    return float(x.detach()) if hasattr(x, 'detach') else float(x)


def batches(steps, size, rng, shuffle=True):
    """batches of similar goal length"""
    idx = list(range(len(steps)))
    if shuffle:
        rng.shuffle(idx)
        chunks = [idx[i:i + size * 16] for i in range(0, len(idx), size * 16)]
        idx = [j for ch in chunks for j in sorted(ch, key=lambda j: len(steps[j].goal.tokens))]
    out = [idx[i:i + size] for i in range(0, len(idx), size)]
    if shuffle: rng.shuffle(out)
    return out


@torch.no_grad()
def evaluate(net, steps, size=32):
    net.eval()
    tot = n = cor = 0; vsum = 0.0; ok_actions = 0; per = collections.defaultdict(lambda: [0, 0]); by_kind = collections.defaultdict(lambda: [0, 0])
    for i in range(0, len(steps), size):
        chunk = steps[i:i + size]
        batch = D.collate(chunk)
        out = net(batch)
        tot += out['policy_loss'].item() * out['n']; n += out['n']; cor += out['correct']
        vsum += out['value_loss'].item() * len(chunk)
        ok = out['action_ok'].tolist(); ok_actions += sum(ok)
        for s, o in zip(chunk, ok):
            kk = by_kind[s.kind]; kk[0] += int(o); kk[1] += 1
        for nm, (c, t) in out['per_name'].items(): per[F.CHOICE_NAMES[nm]][0] += c; per[F.CHOICE_NAMES[nm]][1] += t
    net.train()
    return {'loss': tot / max(n, 1), 'acc': cor / max(n, 1), 'decisions': n, 'action_acc': ok_actions / max(len(steps), 1), 'actions': len(steps), 'value_loss': vsum / max(len(steps), 1),
            'per_name': {k: (v[0] / max(v[1], 1), v[1]) for k, v in per.items()}, 'by_kind': {k: (v[0] / max(v[1], 1), v[1]) for k, v in by_kind.items()}}


NEW_HEADS = ('critic.', 'mgr.', 'size.')


def make_net(bags, cfg):
    return M.PolicyNet(bags, d=cfg['d'], layers=cfg['layers'], heads=cfg['heads'], ff=cfg['ff'], dropout=cfg['dropout'], arch=cfg.get('arch', 'transformer'))      # (a checkpoint made before the encoders were swappable has no `arch`: it is a transformer)


def save(net, cfg, path, extra=None):
    torch.save({'state': net.state_dict(), 'config': cfg, 'extra': extra or {}}, path)


WORLD_BUFFERS = ('all_idx', 'all_off', 'con_idx', 'con_off', 'arity')            # (what a world says about its globals: not weights; a checkpoint made before they were left out of it has them)


def load_config(path):
    """the configuration a checkpoint was made with (architecture, sizes, training settings), without building the network"""
    return torch.load(path, map_location='cpu', weights_only=False)['config']


def load(path, bags):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = make_net(bags, ck['config'])
    state = {k: v for k, v in ck['state'].items() if k not in WORLD_BUFFERS}
    missing, unexpected = net.load_state_dict(state, strict=False)                 # (a model saved before the heads of the reinforcement learners existed lacks exactly those)
    bad = [k for k in missing if not k.startswith(NEW_HEADS)] + list(unexpected)
    if bad: raise RuntimeError('%s does not fit the network (missing or unexpected parameters: %s)' % (path, bad[:4]))
    net.eval()
    return net, ck['config'], ck.get('extra', {})


def train(net, train_steps, dev_sets, cfg, out=None, log=print, rng=None):
    rng = rng or random.Random(cfg.get('seed', 0))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    nb = max(1, (len(train_steps) + cfg['batch'] - 1) // cfg['batch'])
    total = cfg['epochs'] * nb; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=total, pct_start=0.1)
    best = None; hist = []; t0 = time.time(); it = 0
    for ep in range(cfg['epochs']):
        net.train(); tl = tn = tc = 0
        for bi in batches(train_steps, cfg['batch'], rng):
            batch = D.collate([train_steps[j] for j in bi])
            res = net(batch)
            opt.zero_grad(); res['loss'].backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); it += 1
            tl += res['policy_loss'].item() * res['n']; tn += res['n']; tc += res['correct']
        row = {'epoch': ep, 'train_loss': tl / max(tn, 1), 'train_acc': tc / max(tn, 1), 'seconds': time.time() - t0}
        for nm, st in dev_sets.items():
            if st: ev = evaluate(net, st); row[nm] = {k: ev[k] for k in ('loss', 'acc', 'action_acc', 'value_loss')}
        hist.append(row)
        dv = row.get('dev_all', {}).get('loss', row['train_loss'])
        log('epoch %2d  train loss %.3f acc %.3f | %s | %.0fs' % (ep, row['train_loss'], row['train_acc'], ' | '.join('%s loss %.3f acc %.3f act %.3f' % (nm, row[nm]['loss'], row[nm]['acc'], row[nm]['action_acc']) for nm in dev_sets if nm in row), row['seconds']))
        if out and (best is None or dv < best):
            best = dv; save(net, cfg, out, {'epoch': ep, 'dev_loss': dv})
    return hist


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('--out', default='model.pt'); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--d', type=int, default=128); ap.add_argument('--layers', type=int, default=3); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--ff', type=int, default=384)
    ap.add_argument('--arch', default='transformer', choices=M.ARCHS, help='the encoder of the goal (ml_model.ARCHS)')
    ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=32); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--no-lib', action='store_true'); ap.add_argument('--no-strict', action='store_true')
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    sys.setrecursionlimit(100000)
    t0 = time.time()
    bk = bank_all.load(); eng = G.Engine(bk.w, kinds=set(G.Kind), strict=True)
    ctx = D.DataContext(bk.w)
    print('world %.0fs' % (time.time() - t0), flush=True)
    data = build_steps(a.data, bk, ctx, eng)
    tr = list(data['bank'].get('train', [])); dv_bank = list(data['bank'].get('dev', []))
    tr_lib = list(data['lib'].get('train', [])); dv_lib = list(data['lib'].get('dev', []))
    if a.no_strict: tr = [s for s in tr if not s.name.endswith('/strict')]
    train_steps = tr + ([] if a.no_lib else tr_lib)
    print('train steps: bank %d + library %d; dev: bank %d, library %d' % (len(tr), 0 if a.no_lib else len(tr_lib), len(dv_bank), len(dv_lib)), flush=True)
    cfg = {'d': a.d, 'layers': a.layers, 'heads': a.heads, 'ff': a.ff, 'dropout': a.dropout, 'lr': a.lr, 'wd': a.wd, 'batch': a.batch, 'epochs': a.epochs, 'seed': a.seed, 'arch': a.arch}
    net = make_net(ctx.bags, cfg)
    print('parameters %d' % sum(p.numel() for p in net.parameters()), flush=True)
    dev_sets = {'dev_all': dv_bank + dv_lib, 'dev_bank': dv_bank, 'dev_lib': dv_lib}
    hist = train(net, train_steps, dev_sets, cfg, out=a.out, log=lambda m: print(m, flush=True))
    json.dump({'config': cfg, 'history': hist}, open(os.path.splitext(a.out)[0] + '_history.json', 'w'), indent=1)
    net2, cfg2, extra = load(a.out, ctx.bags)
    for nm, st in dev_sets.items():
        if st:
            ev = evaluate(net2, st)
            print('%s (best epoch %s): loss %.3f decisions acc %.3f (%d), actions acc %.3f (%d)' % (nm, extra.get('epoch'), ev['loss'], ev['acc'], ev['decisions'], ev['action_acc'], ev['actions']))
            if nm == 'dev_all': print('  per decision: ' + ', '.join('%s %.2f (%d)' % (k, v[0], v[1]) for k, v in sorted(ev['per_name'].items())))
    print('total %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
