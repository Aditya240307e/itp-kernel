"""check_env.py -- is this machine ready to run the experiments?  (scripts/00_check_env.sh calls it)

    python3 tools/check_env.py [--profile full] [--runs runs/full] [--device cuda]

Checks: Python and the libraries, that the kernel is built, cores / memory / free disk / GPU, the state of the git tree (the runs record the commit; a dirty tree means the results are not reproducible from
a commit), and prints rough needs of the profile.  Exit status 1 when something required is missing, 0 otherwise (warnings do not fail).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, glob, shutil, argparse, subprocess, importlib

ROOT = itp_paths.ROOT
bad = []; warn = []


def line(ok, what, detail=''):
    print('  [%s] %-34s %s' % ('ok' if ok is True else ('!!' if ok is False else '..'), what, detail))


def version_of(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, '__version__', '?'), m
    except Exception as e:
        return None, e


def mem_gb():
    try:
        for ln in open('/proc/meminfo'):
            if ln.startswith('MemTotal:'): return int(ln.split()[1]) / 1048576.0
    except OSError: pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--profile', default='full'); ap.add_argument('--runs', default=''); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    runs = a.runs or os.path.join('runs', a.profile)
    print('Python and libraries')
    ok = sys.version_info >= (3, 10); line(ok, 'python', sys.version.split()[0] + ('' if ok else '  (3.10 or newer is required, 3.13 was used for the paper)'))
    if not ok: bad.append('python')
    need = [('numpy', True), ('scipy', True), ('torch', True), ('gymnasium', True), ('matplotlib', False)]
    for mod, required in need:
        v, m = version_of(mod)
        line(v is not None or (None if not required else False), mod, v if v is not None else 'missing: pip install -r requirements.txt' + ('' if required else '  (only the figures of the report need it)'))
        if v is None: (bad if required else warn).append(mod)
    print('Kernel')
    so = sorted(glob.glob(os.path.join(os.environ.get('ITP_BUILD_DIR', os.path.join(ROOT, 'build')), 'itp_env_lnl*.so')))
    line(bool(so), 'build/itp_env_lnl*.so', os.path.basename(so[0]) if so else 'not built: make kernel  (needs g++ 13 or recent clang and nanobind)')
    if not so: bad.append('kernel')
    print('Machine')
    n = os.cpu_count() or 1; line(True, 'cores', str(n))
    m = mem_gb(); line(None if m is None else True, 'memory', '%.1f GB' % m if m else 'unknown')
    free = shutil.disk_usage(ROOT if not os.path.exists(runs) else runs).free / 2**30
    line(free > 20, 'free disk', '%.0f GB  (the full profile writes several tens of GB of checkpoints, caches and logs)' % free)
    if free <= 20: warn.append('disk')
    v, torch = version_of('torch')
    gpu = False
    if v is not None:
        try:
            gpu = torch.cuda.is_available()
            line(None if not gpu else True, 'cuda', ', '.join('%s (%.0f GB)' % (torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2**30) for i in range(torch.cuda.device_count())) if gpu else 'no GPU visible to torch')
        except Exception as e: line(None, 'cuda', 'unavailable (%s)' % e)
    if a.device.startswith('cuda') and not gpu:
        line(False, 'DEVICE=%s' % a.device, 'requested, but torch sees no GPU'); bad.append('device')
    elif a.device.startswith('cuda'): line(True, 'DEVICE=%s' % a.device, 'the networks are trained on the GPU; the proof search stays on the CPU. UNTESTED by the authors of this code on a GPU: run the smoke profile first')
    print('Repository')
    try:
        head = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True).stdout.strip()
        line(not dirty, 'git', '%s%s' % (head or 'not a git checkout', '  (the tree has uncommitted changes: commit them, the runs record only the commit)' if dirty else ''))
        if dirty: warn.append('dirty')
    except OSError: line(None, 'git', 'not available')
    ds = os.path.join(ROOT, 'data', 'dataset_v2')
    line(os.path.isdir(ds) or a.profile == 'smoke', 'data/dataset_v2', 'present' if os.path.isdir(ds) else ('built by the smoke profile itself' if a.profile == 'smoke' else 'missing'))
    if not os.path.isdir(ds) and a.profile != 'smoke': bad.append('dataset')
    try:
        import exp_spec as SP
        P = SP.PROFILES[a.profile]
        print('Profile %s: %d seeds, %s synthetic theorems, workers per run %s, up to %s probes per problem at evaluation' % (a.profile, P['seeds'], P['synth_target'], P['workers'], P['eval_probes']))
        runs_ = SP.build(a.profile, runs_dir=runs)
        peak = max(r.mem_gb for r in runs_.values())
        print('  %d runs in the graph; the largest run is estimated at %.1f GB of memory (the scheduler packs runs within the memory and cores it is given)' % (len(runs_), peak))
        if m and peak > 0.85 * m: warn.append('memory'); print('  !! one run may need more memory than this machine has')
    except Exception as e: print('  (no profile information: %s)' % e)
    print()
    for w in warn: print('warning:', w)
    if bad: print('NOT READY:', ', '.join(bad)); return 1
    print('ready'); return 0


if __name__ == '__main__':
    sys.exit(main())
