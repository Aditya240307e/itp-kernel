"""ml_pool.py -- processes that run episodes while the learner trains.

A world of the problem bank takes minutes to build and more than a gigabyte of memory, so the workers are FORKED from a process that already holds it: the pages are shared until a worker writes to them.
Fork the pool before anything that does not survive a fork (a CUDA context, threads started by torch) is created in the parent, and let the workers use the CPU only.

    pool = Pool(4, init=make_state, handlers={'episodes': run_episodes})       # init() -> state, in each worker, once, after the fork;  handler(state, payload) -> result
    results = pool.map('episodes', payloads)                                  # one result per payload, in the order of the payloads; the workers take payloads as they become free
    pool.close()

`Pool(0, ...)` runs everything in the calling process (tests, small machines): the same interface, the same results for the same payloads.  A handler that raises sends the traceback back and `map` raises
`WorkerError` with it; the other payloads still finish, so the pool is usable again.

The weights of the network are not sent with the payloads (they are megabytes): the learner saves them to a file (`WeightsFile`) and puts the file's name and a version in the payload; a worker loads a version
once.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, time, traceback, tempfile, multiprocessing as mp


class WorkerError(RuntimeError):
    pass


def _worker(init, handlers, tasks, results, wid):
    try:
        import torch
        torch.set_num_threads(1)
    except Exception: pass
    try:
        state = init()
    except BaseException:
        results.put((-1, False, 'worker %d: init failed:\n%s' % (wid, traceback.format_exc()))); return
    results.put((-1, True, wid))
    while True:
        job = tasks.get()
        if job is None: return
        idx, name, payload = job
        try: results.put((idx, True, handlers[name](state, payload)))
        except BaseException: results.put((idx, False, 'worker %d, %s:\n%s' % (wid, name, traceback.format_exc())))


class Pool:
    def __init__(self, n, init, handlers):
        self.n = int(n); self.handlers = handlers; self.procs = []; self._closed = False
        if self.n <= 0:
            self.state = init(); return
        ctx = mp.get_context('fork')
        self.tasks = ctx.Queue(); self.results = ctx.Queue()
        for w in range(self.n):
            p = ctx.Process(target=_worker, args=(init, handlers, self.tasks, self.results, w), daemon=True); p.start(); self.procs.append(p)
        ready = 0
        while ready < self.n:
            idx, ok, val = self.results.get()
            if idx != -1: continue
            if not ok: self.close(); raise WorkerError(val)
            ready += 1

    def map(self, name, payloads):
        payloads = list(payloads)
        if self.n <= 0:
            return [self.handlers[name](self.state, p) for p in payloads]
        for i, p in enumerate(payloads): self.tasks.put((i, name, p))
        out = [None] * len(payloads); errors = []; got = 0
        while got < len(payloads):
            try: idx, ok, val = self.results.get(timeout=5.0)
            except Exception:
                if any(not p.is_alive() for p in self.procs): self.close(); raise WorkerError('a worker process died (out of memory?)')
                continue
            if idx < 0: continue
            got += 1
            if ok: out[idx] = val
            else: errors.append(val)
        if errors: raise WorkerError(errors[0])
        return out

    def close(self):
        if self._closed: return
        self._closed = True
        if self.n > 0:
            for _ in self.procs: self.tasks.put(None)
            for p in self.procs:
                p.join(timeout=5.0)
                if p.is_alive(): p.terminate()


class WeightsFile:
    """the weights of the learner's network, handed to the workers through a file: `save(net)` writes a new version, `path` and `version` go into the payloads; `load_into(net, path, version, cache)` in a worker
    loads a version once"""
    def __init__(self, directory=None):
        self.dir = directory or tempfile.mkdtemp(prefix='itp_weights_'); os.makedirs(self.dir, exist_ok=True)
        self.version = 0; self.path = os.path.join(self.dir, 'weights.pt')
    def save(self, net):
        import torch
        self.version += 1
        tmp = self.path + '.tmp%d' % os.getpid()
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, tmp); os.replace(tmp, self.path)
        return self.version
    @staticmethod
    def load_into(net, path, version, cache):
        """load the file into `net` unless the worker's `cache` (a dict) says it already has this version; True if it loaded"""
        import torch
        if cache.get('version') == version: return False
        sd = torch.load(path, map_location='cpu', weights_only=True)
        net.load_state_dict(sd); net.eval(); cache['version'] = version
        return True
    def cleanup(self):
        import shutil; shutil.rmtree(self.dir, ignore_errors=True)
