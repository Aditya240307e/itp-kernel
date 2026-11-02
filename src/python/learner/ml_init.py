"""ml_init.py -- a network with random weights, saved like a trained one: the starting point of a learner that begins from nothing.

    python3 ml_init.py DATA --out DIR --arch bigru --d 128 --layers 3 --seed 0          # DIR/best.pt  DIR/final.json

Behaviour cloning, PPO and expert iteration all continue from a checkpoint (`--init`).  This makes the checkpoint of an untrained network, so that reinforcement learning FROM SCRATCH is the same command
as reinforcement learning from a pre-trained network with another `--init`, and so that the architecture (`--arch`, ml_model.ARCHS) and the size of a network are chosen in one place for the learners that
do not take them: a learner trains the architecture its checkpoint has.  Same seed, same architecture, same size: the same weights.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, time, argparse
import torch
import ml_model as M, ml_train as T, ml_world as W, ml_meta as MT


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('--out', required=True)
    ap.add_argument('--arch', default='transformer', choices=M.ARCHS, help='the encoder of the goal (ml_model.ARCHS)')
    ap.add_argument('--d', type=int, default=128); ap.add_argument('--layers', type=int, default=3); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--ff', type=int, default=384); ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--threads', type=int, default=1)
    W.add_args(ap, lemmas=False)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    bk, ctx, eng = W.load(a)
    cfg = {'d': a.d, 'layers': a.layers, 'heads': a.heads, 'ff': a.ff, 'dropout': a.dropout, 'seed': a.seed, 'arch': a.arch}
    net = T.make_net(ctx.bags, cfg); net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    T.save(net, cfg, os.path.join(a.out, 'best.pt'), {'epoch': -1, 'untrained': True})
    MT.write_json(os.path.join(a.out, 'final.json'), {'args': vars(a), 'config': cfg, 'n_params': n_params, 'untrained': True, 'seconds': round(time.time() - t0), 'peak_rss_mb': MT.peak_rss_mb()})
    print('an untrained %s network, %d parameters -> %s' % (a.arch, n_params, os.path.join(a.out, 'best.pt')), flush=True)


if __name__ == '__main__':
    main()
