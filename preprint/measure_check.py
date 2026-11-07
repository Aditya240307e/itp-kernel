#!/usr/bin/env python3
"""measure_check.py -- what trusting a proof costs, on the reference proofs of the problem bank (no PyTorch needed).

    python3 preprint/measure_check.py --out preprint/measurements/check.json                  # about ten minutes, 2 GB
    python3 preprint/measure_check.py --quick --out /tmp/check_quick.json                     # three small problems: a check that the script runs

What it measures (wall-clock; nothing is estimated):

  world      the time to build the world of the whole bank (libraries, 268 statements, 187 reference proofs), the peak resident memory, and how much of the time the kernel spent inside its checked
             entry points (declare_checked, finish_inductive, finish_recursive_function_checked): the price of the trusted part of every declaration, measured inside a real build
  proofs     the reference proofs of the bank: how many, how long (in actions)
  sample     a sample of them, evenly spaced in length: for each, the replay through the engine (the actions carried out one by one), the kernel's check of the finished term against the statement, the size of
             the proof term, and the independent checker's verdict with its time and memory.  The independent checker runs in a forked child with a wall-clock cap and a cap on the memory it allocates, so that a
             proof it cannot do in that budget is REPORTED (status 'timeout' or 'memory') and the next one goes on
  machine    the CPU, the memory, the versions, the commit

The numbers depend on the machine; the file records it.  Run it on an otherwise idle machine.
"""
import os, sys, time, json, argparse, signal, statistics, platform

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src', 'python')); sys.path.insert(0, HERE)
import itp_paths  # noqa: E402,F401
sys.setrecursionlimit(100000)
import itp_env_lnl as E  # noqa: E402
clock = time.perf_counter

# ------------------------------------------------------------------------------------------------------------------ time spent in the kernel's checked entry points
_TIMED = ('declare_checked', 'finish_inductive', 'finish_recursive_function_checked')


class TimedKernel(E.Kernel):
    stats = {n: [0.0, 0] for n in _TIMED}


def _wrap(name):
    base = getattr(E.Kernel, name)
    def f(self, *a, **kw):
        t = clock()
        try: return base(self, *a, **kw)
        finally:
            s = TimedKernel.stats[name]; s[0] += clock() - t; s[1] += 1
    f.__name__ = name
    return f


for _n in _TIMED:
    if hasattr(E.Kernel, _n): setattr(TimedKernel, _n, _wrap(_n))
E.Kernel = TimedKernel                                              # (author.World makes its kernel with E.Kernel())

import bank_all, gym_env as G, gym_acts as ACTS, recheck as RC  # noqa: E402
import measure as M  # noqa: E402                                  (the machine description)


def status_kb(field, pid='self'):
    try:
        for line in open('/proc/%s/status' % pid):
            if line.startswith(field + ':'): return int(line.split()[1])
    except OSError: pass
    return 0


def rss_mb(pid='self'): return status_kb('VmRSS', pid) / 1024.0
def hwm_mb(pid='self'): return status_kb('VmHWM', pid) / 1024.0


def run_isolated(fn, cap_s, cap_mb):
    """fn() in a forked child.  Returns a dict: {'status': 'ok'|'error'|'timeout'|'memory', ...what fn returned..., 'seconds', 'rss_delta_mb'}.  The child may allocate `cap_mb` MB beyond what it inherits"""
    r, wfd = os.pipe(); pid = os.fork()
    if pid == 0:
        code = 0
        try:
            os.close(r); base = rss_mb(); t = clock(); out = fn(); out = dict(out)
            out['seconds'] = clock() - t; out['rss_delta_mb'] = max(0.0, hwm_mb() - base); out['status'] = 'ok'
            os.write(wfd, json.dumps(out).encode())
        except BaseException as e:                                  # (the child must not run the parent's cleanup)
            try: os.write(wfd, json.dumps({'status': 'error', 'error': repr(e)[:300]}).encode())
            except BaseException: pass
            code = 1
        finally: os._exit(code)
    os.close(wfd)
    t0 = clock(); base = rss_mb(pid); verdict = None; peak = base
    while True:
        done, _st = os.waitpid(pid, os.WNOHANG)
        if done: break
        now = rss_mb(pid); peak = max(peak, now)
        if clock() - t0 > cap_s: os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0); verdict = 'timeout'; break
        if now - base > cap_mb: os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0); verdict = 'memory'; break
        time.sleep(0.05)
    try: raw = os.read(r, 1 << 20)
    finally: os.close(r)
    if verdict: return {'status': verdict, 'seconds': clock() - t0, 'rss_delta_mb': peak - base}
    try: return json.loads(raw.decode())
    except ValueError: return {'status': 'error', 'error': 'no answer from the child', 'seconds': clock() - t0}


def evenly(cand, k):
    if len(cand) <= k: return list(cand)
    if k == 1: return [cand[len(cand) // 2]]
    idx = sorted({int(round(i * (len(cand) - 1) / (k - 1))) for i in range(k)})
    return [cand[i] for i in idx]


def summary(xs):
    xs = sorted(xs)
    if not xs: return {}
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    return {'n': len(xs), 'min': xs[0], 'median': statistics.median(xs), 'p90': q(0.9), 'max': xs[-1], 'total': sum(xs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True); ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'dataset_v2'))
    ap.add_argument('--proofs', type=int, default=16, help='the size of the sample of reference proofs')
    ap.add_argument('--cap-s', type=float, default=120.0, help='wall-clock cap for the independent checker on one proof')
    ap.add_argument('--cap-mb', type=float, default=2500.0, help='the memory the independent checker may allocate on one proof')
    ap.add_argument('--quick', action='store_true', help='the three shortest problems only (the world in seconds)')
    a = ap.parse_args(argv)
    bank = json.load(open(os.path.join(a.data, 'bank.json')))
    demos = sorted((p['demonstration']['length'], p['name'], p['demonstration']['steps']) for p in bank['problems'] if p.get('demonstration') and p['demonstration'].get('steps'))
    lengths = [n for n, _, _ in demos]
    proofs = {'proved': len(demos), 'problems': len(bank['problems']), 'actions_total': sum(lengths), 'actions_median': statistics.median(lengths), 'actions_max': max(lengths), 'actions_min': min(lengths),
              'deciles': [lengths[min(len(lengths) - 1, int(i * len(lengths) / 10))] for i in range(10)]}
    only = None
    if a.quick:
        demos = demos[:3]; only = {n for _, n, _ in demos}; a.proofs = 3
    scrub = lambda v: ('<repo>' + v[len(ROOT):]) if isinstance(v, str) and v.startswith(ROOT) else v           # (no path of the machine in the file)
    out = {'machine': M.machine(), 'proofs': proofs, 'args': {k: scrub(v) for k, v in vars(a).items()}}
    print('building the world ...', flush=True)
    t0 = clock(); bk = bank_all.load(only=only) if only else bank_all.load(); t_world = clock() - t0
    k = bk.w.k
    kern = {n: {'seconds': round(s, 3), 'calls': c} for n, (s, c) in TimedKernel.stats.items()}
    kern_total = sum(v['seconds'] for v in kern.values())
    out['world'] = {'seconds': round(t_world, 2), 'globals': k.num_globals(), 'problems': len(bk.placed), 'peak_rss_mb': round(hwm_mb(), 1), 'rss_mb': round(rss_mb(), 1),
                    'kernel_checked_entry_points': kern, 'kernel_seconds': round(kern_total, 3), 'kernel_share': kern_total / t_world if t_world else None, 'python': platform.python_version()}
    print('  %.1f s, %d globals, peak %.0f MB; the kernel spent %.2f s (%.1f%%) in its checked entry points' % (t_world, k.num_globals(), hwm_mb(), kern_total, 100.0 * kern_total / t_world), flush=True)

    eng = G.Engine(bk.w, kinds=set(G.Kind), strict=True)
    verify_times = []
    orig_verify = eng.verify
    def timed_verify(stmt, p):
        t = clock()
        try: return orig_verify(stmt, p)
        finally: verify_times.append(clock() - t)
    eng.verify = timed_verify

    sample = evenly([d for d in demos if d[1] in bk.placed], a.proofs); items = []
    print('replaying %d reference proofs ...' % len(sample), flush=True)
    for length, name, steps in sample:
        pl = bk.placed[name]; verify_times.clear(); t = clock()
        try: term = ACTS.replay_portable(eng, pl.stmt, steps, pl.premises('prefix'))
        except Exception as e:
            items.append({'name': name, 'actions': length, 'replay_error': repr(e)[:200]}); continue
        dt = clock() - t
        size = k.node_info(term.n)[2]
        items.append({'name': name, 'actions': length, 'replay_s': dt, 'kernel_check_s': verify_times[-1] if verify_times else None, 'term_tree_nodes': size, '_term': term, '_stmt': pl.stmt})
    good = [it for it in items if '_term' in it]
    t = clock(); so = RC.SecondOpinion(k, needed_by=[x for it in good for x in (it['_term'], it['_stmt'])]) if good else None; setup = clock() - t
    print('independent checker: environment of %d globals made in %.1f s' % (len(so.checker.g) if so else 0, setup), flush=True)
    for it in good:
        term, stmt = it.pop('_term'), it.pop('_stmt')
        res = run_isolated(lambda: dict(zip(('accepted', 'message'), (lambda r: (bool(r[0]), str(r[1])[:200]))(so.accepts(term, stmt)))), a.cap_s, a.cap_mb)
        it['independent'] = res
        print('  %-26s %4d actions  replay %7.1f ms  kernel check %7.2f ms  %9d nodes  independent: %s %s' % (
            it['name'], it['actions'], 1e3 * it['replay_s'], 1e3 * (it['kernel_check_s'] or 0), it['term_tree_nodes'], res.get('status'),
            ('%.2f s, +%.0f MB, accepted=%s' % (res['seconds'], res['rss_delta_mb'], res.get('accepted'))) if res.get('status') == 'ok' else ('%.0f s' % res.get('seconds', 0))), flush=True)
    for it in items: it.pop('_term', None); it.pop('_stmt', None)
    ok = [it for it in good if it['independent'].get('status') == 'ok']
    out['independent_setup'] = {'seconds': round(setup, 2), 'globals': len(so.checker.g) if so else 0}
    out['sample'] = items
    out['summary'] = {'sampled': len(sample), 'replayed': len(good), 'independent_ok': len(ok), 'independent_accepted': sum(1 for it in ok if it['independent'].get('accepted')),
                      'independent_timeout': sum(1 for it in good if it['independent'].get('status') == 'timeout'), 'independent_memory': sum(1 for it in good if it['independent'].get('status') == 'memory'),
                      'independent_error': sum(1 for it in good if it['independent'].get('status') == 'error'),
                      'replay_s': summary([it['replay_s'] for it in good]), 'kernel_check_s': summary([it['kernel_check_s'] for it in good if it['kernel_check_s'] is not None]),
                      'independent_s': summary([it['independent']['seconds'] for it in ok]), 'independent_rss_delta_mb': summary([it['independent']['rss_delta_mb'] for it in ok])}
    out['peak_rss_mb'] = round(hwm_mb(), 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f: json.dump(out, f, indent=1, sort_keys=True, default=str)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
