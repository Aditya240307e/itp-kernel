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
    """(commit, dirty): the commit the code is at (empty when it is not a git checkout) and whether the CODE (itp_paths.CODE_PATHS: src, tests, mutants, the manifest of the dataset) differs from it -- a changed file, or a new one.
    A document, the paper, a tool or a script that is edited is not dirty: it cannot change what a run does"""
    commit = _git('rev-parse', 'HEAD')
    dirty = bool(_git('status', '--porcelain', '--untracked-files=normal', '--', *itp_paths.CODE_PATHSPEC)) if commit else None
    return commit, dirty


def _read(path):
    """the text of a file ('' where it cannot be read: /proc does not exist on a Mac)"""
    try:
        with open(path) as f: return f.read()
    except OSError: return ''


def _sysctl(name):
    """`sysctl -n NAME` ('' where there is no sysctl or no such name): how macOS and the BSDs tell what Linux keeps in /proc"""
    try:
        r = subprocess.run(['sysctl', '-n', name], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ''
    except Exception: return ''


def cpu_model():
    for line in _read('/proc/cpuinfo').splitlines():
        if line.lower().startswith('model name'): return line.split(':', 1)[1].strip()
    return _sysctl('machdep.cpu.brand_string') or platform.processor() or platform.machine()           # (macOS: 'Apple M3 Pro')


def memory_gb():
    """the physical memory in GB (Linux: /proc/meminfo; macOS: hw.memsize), None where neither tells"""
    for line in _read('/proc/meminfo').splitlines():
        if line.startswith('MemTotal:'): return round(int(line.split()[1]) / 1048576, 1)
    b = _sysctl('hw.memsize')
    if b.isdigit(): return round(int(b) / 2 ** 30, 1)
    try: return round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30, 1)
    except (ValueError, OSError, AttributeError): return None


def usable_cores():
    """how many processes can run at full speed at once: the cores this process may use (Linux: its affinity mask); on an Apple-silicon Mac only the PERFORMANCE cores (hw.perflevel0), because
    the efficiency cores run this kind of work at about a third of the speed and a job that lands on one makes the whole matrix wait for it"""
    perf = _sysctl('hw.perflevel0.logicalcpu')
    if perf.isdigit() and int(perf) > 0: return int(perf)
    try: return len(os.sched_getaffinity(0))
    except (AttributeError, OSError): return os.cpu_count() or 1


def rss_mb():
    """the resident memory of this process in MB (Linux: /proc; elsewhere `ps`; None if neither answers)"""
    for line in _read('/proc/self/status').splitlines():
        if line.startswith('VmRSS:'): return int(line.split()[1]) / 1024.0
    try:
        r = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True, timeout=10)
        return int(r.stdout.split()[0]) / 1024.0
    except Exception: return None


def maxrss_mb(v, platform_name):
    """MB from the `ru_maxrss` of getrusage: bytes on macOS, kilobytes on Linux and the BSDs"""
    return v / (1048576.0 if platform_name == 'darwin' else 1024.0)


def peak_rss_mb():
    """the largest resident memory of this process so far in MB (Linux: VmHWM; elsewhere getrusage)"""
    for line in _read('/proc/self/status').splitlines():
        if line.startswith('VmHWM:'): return int(line.split()[1]) / 1024.0
    try:
        import resource
        return maxrss_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform)
    except Exception: return None


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
