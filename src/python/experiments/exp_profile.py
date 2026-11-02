"""exp_profile.py -- where the time goes, how long one call takes, and how much memory the worker processes use.

    L = Ledger()
    with instrument(L):                       # patches the entry points of the engine / the environment / the policy for the duration of the block, in THIS process
        ... run provers, or step an environment ...
    L.report(wall_seconds)                    # {'wall_s', 'categories': {name: {'seconds', 'share', 'calls'}}, 'other_s', 'other_share', 'accounted_share'}
    L.latency('probe:APPLY')                  # {'n', 'mean_ms', 'p50_ms', 'p90_ms', 'p99_ms', 'max_ms', 'total_s'}
    L.outcomes['probe:APPLY']                 # Counter({'ok': 31, 'refused': 912, 'fuel': 2})

EXCLUSIVE TIME.  Calls nest (a menu probes, a probe runs a tactic, a tactic checks with the kernel), so the time of a call is charged to its own category and NOT to the wrapped calls it makes: the categories
then add up to the wall-clock time of the run, and the rest (`other`: the search's own bookkeeping, the Python glue between the calls) is what the categories leave over.  A latency sample is INCLUSIVE (the
time from the call to its return), because that is what a caller waits for.

THE CATEGORIES (`HOOKS`).  menu: listing the legal actions of a goal and the candidates behind them; probe: trying an action on a copy of the goal (checkpoint, rewind, settling the goals it leaves), without the
tactic itself; tactic: the tactic layer and the kernel calls it makes, inside `Engine._run`; nf: display normal forms; replay: applying the chosen action to the live proof; start / verify: a proof state for a
statement, and the final check of the finished term by the kernel; observe: the goal as arrays for the policy (C++); assembler: the options of the decision language; features: goal and option arrays for the
Gymnasium observation; policy: the network (encoder, decoder, scoring).

What it costs.  A wrapper adds a few microseconds to a call that takes tens of microseconds to seconds; `overhead` in bench.json measures the difference on the same work.  Nothing is estimated: the clock is
`time.perf_counter`, the ledger belongs to one thread (a call from another thread is passed through unmeasured), and a hook whose module cannot be imported (no torch, no gymnasium) is skipped and reported.

Memory.  The resident set of a forked worker counts the pages it shares with its parent again, so the sum over the workers overstates the cost of the world 1 + N times; the PROPORTIONAL set size (`Pss` of
/proc/PID/smaps_rollup) divides every shared page between the processes that map it.  `ProcessSampler` reads both for a process and its children while a run goes on, from a separate process
(Linux; None where /proc lacks them).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, time, json, array, threading, functools, collections, importlib
import numpy as np


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ percentiles
def summarize(xs, scale=1e3):
    """{'n', 'mean_ms', 'p50_ms', 'p90_ms', 'p99_ms', 'max_ms', 'total_s'} of a list of durations in seconds ({'n': 0} when empty).  Percentiles are linear interpolations between the observed values; with fewer than
    100 samples p99 is little more than the maximum, and the report says n."""
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0: return {'n': 0}
    q = np.percentile(a, [50, 90, 99])
    return {'n': int(a.size), 'mean_ms': float(a.mean() * scale), 'p50_ms': float(q[0] * scale), 'p90_ms': float(q[1] * scale), 'p99_ms': float(q[2] * scale), 'max_ms': float(a.max() * scale), 'total_s': float(a.sum())}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the ledger
class Ledger:
    """exclusive seconds, call counts and latency samples by name.  `enter(cat)` / `leave(key, tag)` bracket a wrapped call; `clock` is injectable (the tests drive it by hand)."""
    def __init__(self, clock=time.perf_counter):
        self.clock = clock; self.thread = threading.get_ident()
        self.excl = collections.defaultdict(float); self.calls = collections.Counter()
        self.samples = collections.defaultdict(lambda: array.array('d')); self.outcomes = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(lambda: array.array('d'))              # plain numbers that go with the calls (the number of options of a menu)
        self.stack = []; self.skipped = []                                           # skipped: hooks that could not be installed

    def enter(self, cat):
        now = self.clock()
        if self.stack: top = self.stack[-1]; self.excl[top[0]] += now - top[1]       # the caller's own clock stops while the callee runs
        self.stack.append([cat, now, now]); self.calls[cat] += 1                     # [category, start of its current exclusive stretch, start of the call]

    def leave(self, key=None, tag=None):
        now = self.clock(); cat, since, start = self.stack.pop(); self.excl[cat] += now - since
        if self.stack: self.stack[-1][1] = now                                       # the caller's clock restarts
        if key is not None:
            self.samples[key].append(now - start)
            if tag is not None: self.outcomes[key][tag] += 1
        return now - start

    def balanced(self): return not self.stack

    def latency(self, key, scale=1e3): return summarize(self.samples.get(key, ()), scale)

    def keys(self, prefix=''): return sorted(k for k in self.samples if k.startswith(prefix))

    def report(self, wall):
        """the shares of the wall-clock time `wall` (seconds): by category, what no category accounts for (`other`), and how much of the wall time the categories account for"""
        tot = float(sum(self.excl.values())); cats = {}
        for c in sorted(self.excl, key=lambda c: -self.excl[c]):
            cats[c] = {'seconds': float(self.excl[c]), 'share': (self.excl[c] / wall) if wall > 0 else None, 'calls': int(self.calls[c])}
        other = max(0.0, wall - tot)
        return {'wall_s': float(wall), 'categories': cats, 'other_s': other, 'other_share': (other / wall) if wall > 0 else None, 'accounted_share': (min(tot, wall) / wall) if wall > 0 else None}

    def reset(self):
        self.excl.clear(); self.calls.clear(); self.samples.clear(); self.outcomes.clear(); self.values.clear(); del self.stack[:]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the hooks
def _action_of(a, kw, pos):
    return a[pos] if len(a) > pos else kw.get('action')


def _probe_key(a, kw):
    act = _action_of(a, kw, 2)                                                       # probe_status(self, g, action, menu, fuel=None)
    kind = getattr(act, 'kind', None)
    return 'probe:%s' % (getattr(kind, 'name', None) or kind)


def _probe_tag(r): return r[1] if isinstance(r, tuple) and len(r) == 2 else None     # (goals, status) -> status


# (module, class or None, attribute, category, key function, tag function, values function)
# class None: a function of the module.  The key names the latency samples; tag names the outcome counted under that key; values returns a number saved under `key + '#n'` (the size of what the call returned).
HOOKS = (('gym_env', 'Engine', 'menu', 'menu', lambda a, kw: 'menu', None, lambda r: len(getattr(r, 'actions', ()))),
         ('gym_env', 'Engine', 'probe_status', 'probe', _probe_key, _probe_tag, None),
         ('gym_env', 'Engine', '_run', 'tactic', None, None, None),
         ('gym_env', 'Engine', 'nf', 'nf', None, None, None),
         ('gym_env', 'Engine', 'replay', 'replay', lambda a, kw: 'replay', None, None),
         ('gym_env', 'Engine', 'start', 'start', lambda a, kw: 'start', None, None),
         ('gym_env', 'Engine', 'verify', 'verify', lambda a, kw: 'verify', None, None),
         ('gym_env', 'Engine', 'observe', 'observe', None, None, None),
         ('gym_policy', 'Assembler', 'start', 'assembler', None, None, None),
         ('gym_policy', 'Assembler', 'choose', 'assembler', None, None, None),
         ('ml_features', None, 'goal_features', 'features', None, None, None),
         ('bank_env', 'BankEnv', '_obs', 'features', None, None, None),
         ('ml_model', 'Decoder', '__init__', 'policy', None, None, None),
         ('ml_model', 'Decoder', 'scores', 'policy', None, None, None),
         ('ml_model', 'Decoder', 'child', 'policy', None, None, None),
         ('ml_model', 'Decoder', 'critic', 'policy', None, None, None),
         ('ml_model', 'Decoder', 'manager_logit', 'policy', None, None, None))
CATEGORIES = ('menu', 'probe', 'tactic', 'nf', 'replay', 'start', 'verify', 'observe', 'assembler', 'features', 'policy')

_ACTIVE = []                                                                        # the ledger that is installed (one at a time: a wrapper inside a wrapper would count twice)


def _wrap(ledger, fn, cat, key, tag, values):
    @functools.wraps(fn)
    def wrapped(*a, **kw):
        if threading.get_ident() != ledger.thread: return fn(*a, **kw)             # (the ledger is not thread-safe: other threads run unmeasured)
        k = key(a, kw) if key is not None else None
        ledger.enter(cat)
        try: r = fn(*a, **kw)
        except BaseException:
            ledger.leave(k, 'raised' if k is not None else None); raise
        ledger.leave(k, tag(r) if tag is not None else None)
        if values is not None and k is not None: ledger.values[k + '#n'].append(values(r))
        return r
    wrapped.__itp_profile__ = True
    return wrapped


class instrument:
    """with instrument(ledger): ...   install the hooks of `HOOKS` for the duration of the block and take them out again, whatever happens.  `only`: a set of categories (default all)."""
    def __init__(self, ledger, only=None):
        self.ledger = ledger; self.only = set(only) if only is not None else None; self.saved = []

    def __enter__(self):
        if _ACTIVE: raise RuntimeError('a profile is already installed in this process')
        for mod, cls, attr, cat, key, tag, values in HOOKS:
            if self.only is not None and cat not in self.only: continue
            try: m = importlib.import_module(mod)
            except Exception as e:
                self.ledger.skipped.append('%s.%s%s: %s' % (mod, (cls + '.') if cls else '', attr, type(e).__name__)); continue
            owner = getattr(m, cls, None) if cls else m
            if owner is None or attr not in vars(owner):
                self.ledger.skipped.append('%s.%s%s: not found' % (mod, (cls + '.') if cls else '', attr)); continue
            fn = vars(owner)[attr]
            if getattr(fn, '__itp_profile__', False): raise RuntimeError('%s.%s is already wrapped' % (cls or mod, attr))
            setattr(owner, attr, _wrap(self.ledger, fn, cat, key, tag, values)); self.saved.append((owner, attr, fn))
        _ACTIVE.append(self.ledger)
        return self.ledger

    def __exit__(self, *exc):
        for owner, attr, fn in reversed(self.saved): setattr(owner, attr, fn)
        self.saved = []
        if _ACTIVE: _ACTIVE.pop()
        return False


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ memory
def _read(path):
    try:
        with open(path) as f: return f.read()
    except OSError: return ''


def pss_mb(pid):
    """the proportional set size of a process in MB (None where /proc/PID/smaps_rollup does not say)"""
    for line in _read('/proc/%d/smaps_rollup' % pid).splitlines():
        if line.startswith('Pss:'): return int(line.split()[1]) / 1024.0
    return None


def rss_of(pid):
    for line in _read('/proc/%d/status' % pid).splitlines():
        if line.startswith('VmRSS:'): return int(line.split()[1]) / 1024.0
    return None


def children_of(pid):
    """the pids of the processes below `pid`, all generations (Linux /proc; empty elsewhere)"""
    parent = {}
    try: names = os.listdir('/proc')
    except OSError: return []
    for n in names:
        if not n.isdigit(): continue
        st = _read('/proc/%s/stat' % n)
        if not st: continue
        try: parent[int(n)] = int(st[st.rindex(')') + 2:].split()[1])                  # (the command name may hold spaces and parentheses: read after the last ')')
        except (ValueError, IndexError): pass
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        for c, pp in parent.items():
            if pp == p and c not in out: out.append(c); todo.append(c)
    return sorted(out)


class Tally:
    """the maxima of the memory readings of a process and of the processes below it: the sum of the proportional set sizes (`pss_mb`), the sum of the resident sets (`rss_sum_mb`: what a naive count says) and
    the number of processes.  A reading counts only when every process answered (a process that ends between two reads would make the sum too small)."""
    def __init__(self, pid):
        self.pid = pid; self.pss_mb = None; self.rss_sum_mb = None; self.procs = 1; self.samples = 0

    def take(self, exclude=()):
        pids = [self.pid] + [c for c in children_of(self.pid) if c not in exclude]
        pss = [pss_mb(p) for p in pids]; rss = [rss_of(p) for p in pids]
        if all(v is not None for v in pss):
            s = sum(pss); self.pss_mb = s if self.pss_mb is None else max(self.pss_mb, s)
        if all(v is not None for v in rss):
            s = sum(rss); self.rss_sum_mb = s if self.rss_sum_mb is None else max(self.rss_sum_mb, s)
        self.procs = max(self.procs, len(pids)); self.samples += 1

    def result(self):
        r = lambda v: None if v is None else round(v, 1)
        return {'pss_mb_max': r(self.pss_mb), 'rss_sum_mb_max': r(self.rss_sum_mb), 'processes_max': self.procs, 'samples': self.samples}


def _sampler_main(argv):
    """the body of the sampler process: read the memory of process argv[0] and its children every argv[1] seconds until told to stop (SIGTERM) or until the process is gone, then print the result as one line of JSON"""
    import signal
    pid = int(argv[0]); every = float(argv[1]); me = os.getpid(); stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    t = Tally(pid)
    while not stop and os.path.exists('/proc/%d' % pid):
        try: t.take(exclude=(me,))
        except Exception: pass
        time.sleep(every)
    try: t.take(exclude=(me,))
    except Exception: pass
    print(json.dumps(t.result()), flush=True)


class ProcessSampler:
    """with ProcessSampler() as ps: ... run ...;  ps.result() -> {'pss_mb_max', 'rss_sum_mb_max', 'processes_max', 'samples'}.  The readings are made by a separate PROCESS (not a thread: this process forks its
    workers while it runs, and a fork must not find a thread of ours in the middle of reading a file), every `every` seconds, starting at once."""
    def __init__(self, every=0.25, pid=None):
        self.every = every; self.pid = pid or os.getpid(); self.proc = None; self.out = None

    def __enter__(self):
        import subprocess
        code = 'import sys; sys.path.insert(0, %r); import exp_profile; exp_profile._sampler_main(sys.argv[1:])' % os.path.dirname(os.path.abspath(__file__))
        try: self.proc = subprocess.Popen([sys.executable, '-I', '-c', code, str(self.pid), str(self.every)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError: self.proc = None
        return self

    def __exit__(self, *exc):
        if self.proc is None: return False
        try:
            self.proc.terminate()
            out, _ = self.proc.communicate(timeout=15)
            self.out = json.loads(out.strip().splitlines()[-1])
        except Exception:
            self.out = None
            try: self.proc.kill()
            except Exception: pass
        return False

    def result(self):
        return self.out or {'pss_mb_max': None, 'rss_sum_mb_max': None, 'processes_max': None, 'samples': 0}
