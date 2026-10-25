"""ml_meta.py -- what a result file says about the machine and the code that made it.

    meta = run_meta(args=vars(a))          # a dict: time, host, cpu, cores, memory, gpu, python/torch/numpy versions, the git commit of the code and whether the tree was clean, the arguments

Every file of results the experiments write (training logs, evaluation rows, benchmarks) carries this block, so that a number in the paper can be traced to the code and the machine that made it.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, time, json, platform, subprocess, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def _git(*args):
    try: return subprocess.run(['git', '-C', ROOT] + list(args), capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ''


def git_state():
    """(commit, dirty): the commit the code is at (empty when it is not a git checkout) and whether files of the tracked tree differ from it"""
    commit = _git('rev-parse', 'HEAD')
    dirty = bool(_git('status', '--porcelain', '--untracked-files=no')) if commit else None
    return commit, dirty


def cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.lower().startswith('model name'): return line.split(':', 1)[1].strip()
    except OSError: pass
    return platform.processor() or platform.machine()


def memory_gb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemTotal:'): return round(int(line.split()[1]) / 1048576, 1)
    except OSError: pass
    return None


def rss_mb():
    """the resident memory of this process in MB (Linux; None elsewhere)"""
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'): return int(line.split()[1]) / 1024.0
    except OSError: pass
    return None


def peak_rss_mb():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmHWM:'): return int(line.split()[1]) / 1024.0
    except OSError: pass
    return None


def run_meta(args=None, extra=None):
    commit, dirty = git_state()
    try:
        import torch
        tv = torch.__version__; gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception: tv = None; gpu = []
    try:
        import numpy; nv = numpy.__version__
    except Exception: nv = None
    m = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'host': platform.node(), 'platform': platform.platform(), 'cpu': cpu_model(), 'cores': os.cpu_count(), 'memory_gb': memory_gb(), 'gpu': gpu,
         'python': platform.python_version(), 'torch': tv, 'numpy': nv, 'commit': commit, 'tree_dirty': dirty, 'argv': list(sys.argv)}
    if args is not None: m['args'] = {k: v for k, v in args.items() if isinstance(v, (str, int, float, bool, type(None), list, tuple, dict))}
    if extra: m.update(extra)
    return m


def file_digest(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()[:n]


def write_json(path, obj):
    """write JSON atomically (a crash leaves the old file, not half a new one)"""
    tmp = '%s.tmp%d' % (path, os.getpid())
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
