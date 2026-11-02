"""exp_run.py -- run the experiment matrix of a profile: resumable, dependency-ordered, parallel.

    python3 src/python/experiments/exp_run.py smoke --runs runs/smoke                        # everything the profile defines (hours on 2 cores)
    python3 src/python/experiments/exp_run.py full --runs runs/full --seeds 0-4 --jobs 2     # the paper's matrix, two runs at a time
    python3 src/python/experiments/exp_run.py full --runs runs/full --stage data,pre,ft      # only some stages (data pre ft rl eval bench report)
    python3 src/python/experiments/exp_run.py full --runs runs/full --target rl/ei_her/s0    # one run and what it needs
    python3 src/python/experiments/exp_run.py full --runs runs/full --status                 # what is done, what is not
    python3 src/python/experiments/exp_run.py full --runs runs/full --dry-run                # the commands, nothing executed
    ... --set rl_iters=20 --set eval_probes=500                                               # change fields of the profile (recorded in PROFILE.json)

A run is DONE when its command exited with status 0 and the files it promises exist (exp_spec.Run.check); that is recorded in `RUNS/.status/<id>.json` (return code, times, host, commit, and the digest of the code when the run started and when it ended).  Running the same
command again skips what is done and re-runs the rest; every command of the matrix is itself resumable (a training run continues from its last checkpoint, an evaluation from its rows, the generator from its
shards), so an interrupted run loses at most the work since its last checkpoint.  `--force ID` re-runs a run (and, with --cascade, everything that depends on it).

Logs: `RUNS/logs/<id>.log`.  Resources: `--jobs N` runs at most N commands at once, `--cores C` (default: all) and `--mem-gb G` (default: 85% of the memory) are budgets the scheduler respects with the
estimates of the runs (exp_spec: a run's cores are its --workers).  A failed run blocks the runs that need it; the others go on (--fail-fast stops at the first failure).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, signal, argparse, subprocess, platform, collections
import exp_spec as SP, exp_repo as XR

ROOT = itp_paths.ROOT
MACHINE_FIELDS = ('device', 'threads', 'workers', 'mem_scale')          # profile fields about the machine, not the results: they may change between invocations of one runs directory


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ status
def sid(rid): return rid.replace('/', '__')


def status_path(R, rid): return os.path.join(R, '.status', sid(rid) + '.json')


def read_status(R, rid):
    try:
        with open(status_path(R, rid)) as f: return json.load(f)
    except (OSError, ValueError): return None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = '%s.tmp%d' % (path, os.getpid())
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def is_done(R, run):
    st = read_status(R, run.id)
    if not st or st.get('returncode') != 0: return False
    return all(os.path.exists(os.path.join(ROOT, c) if not os.path.isabs(c) else c) for c in run.check)


def parse_seeds(text):
    """'0-4' -> [0,1,2,3,4]; '0,2,5' -> [0,2,5]; '0-2,5' -> [0,1,2,5]"""
    out = []
    for part in text.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part: lo, hi = part.split('-'); out += list(range(int(lo), int(hi) + 1))
        else: out.append(int(part))
    return sorted(set(out))


def mem_total_gb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemTotal:'): return int(line.split()[1]) / 1048576.0
    except OSError: pass
    return 8.0


def git_commit():
    try: return subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ''


def code_tree():
    """the digest of the code as it is now ('' when it cannot be computed).  A run records it when it starts and when it ends: the report then says whether the whole matrix was made with one version of the code.  The commit
    is no help for that, since a commit that changed only the paper is another commit with the same code"""
    try: return XR.tree_digest()
    except Exception: return ''


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ selection
def dependents(runs, rid):
    out = set(); changed = True
    while changed:
        changed = False
        for r in runs.values():
            if r.id not in out and (rid in r.needs or any(n in out for n in r.needs)): out.add(r.id); changed = True
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ scheduler
class Scheduler:
    def __init__(self, runs, R, ids, jobs, cores, mem_gb, fail_fast=False, say=print):
        self.runs = runs; self.R = R; self.ids = list(ids); self.jobs = jobs; self.cores = cores; self.mem = mem_gb; self.fail_fast = fail_fast; self.say = say
        self.running = {}                                            # id -> (Popen, t0, logfile)
        self.tree0 = {}                                              # id -> the digest of the code when the run started
        self.failed = {}; self.skipped = set()
        self.done = {i for i in self.ids if is_done(R, runs[i])}

    def blocked_by_failure(self, rid):
        return [n for n in self.runs[rid].needs if n in self.failed or n in self.skipped]

    def runnable(self):
        used_c = sum(self.runs[i].cores for i in self.running); used_m = sum(self.runs[i].mem_gb for i in self.running)
        out = []
        for i in self.ids:
            if i in self.done or i in self.running or i in self.failed or i in self.skipped: continue
            r = self.runs[i]
            if any(n not in self.done and n in self.ids for n in r.needs): continue
            if any(n not in self.done and n not in self.ids and not is_done(self.R, self.runs[n]) for n in r.needs): self.skipped.add(i); self.say('skip %s: it needs %s, which is not done and not selected' % (i, [n for n in r.needs if n not in self.done]))
            if i in self.skipped: continue
            if len(self.running) + len(out) >= self.jobs: break
            if self.running or out:
                if used_c + r.cores > self.cores or used_m + r.mem_gb > self.mem: continue
            out.append(i); used_c += r.cores; used_m += r.mem_gb
        return out

    def start(self, rid):
        r = self.runs[rid]; os.makedirs(os.path.join(self.R, 'logs'), exist_ok=True)
        logp = os.path.join(self.R, 'logs', sid(rid) + '.log')
        lf = open(logp, 'a'); lf.write('\n===== %s  %s\n$ %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), rid, ' '.join(r.argv))); lf.flush()
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '1'), MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS', '1'))
        self.tree0[rid] = code_tree()
        p = subprocess.Popen(r.argv, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.running[rid] = (p, time.time(), lf)
        self.say('start  %-30s [%d running]  log %s' % (rid, len(self.running), shown(logp)))

    def finish(self, rid, rc):
        p, t0, lf = self.running.pop(rid); lf.close()
        r = self.runs[rid]; secs = time.time() - t0
        missing = [c for c in r.check if not os.path.exists(os.path.join(ROOT, c) if not os.path.isabs(c) else c)]
        ok = rc == 0 and not missing
        write_json(status_path(self.R, rid), {'id': rid, 'returncode': rc if not (rc == 0 and missing) else -1, 'missing': missing, 'seconds': round(secs, 1), 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                              'host': platform.node(), 'commit': git_commit(), 'tree': self.tree0.pop(rid, ''), 'tree_finish': code_tree(), 'argv': r.argv})
        if ok: self.done.add(rid); self.say('done   %-30s %s' % (rid, fmt_secs(secs)))
        else:
            self.failed[rid] = rc
            self.say('FAILED %-30s status %s%s after %s; see %s' % (rid, rc, (' (missing %s)' % missing) if missing else '', fmt_secs(secs), shown(os.path.join(self.R, 'logs', sid(rid) + '.log'))))

    def poll(self):
        for rid in list(self.running):
            rc = self.running[rid][0].poll()
            if rc is not None: self.finish(rid, rc)

    def stop_all(self):
        for rid, (p, t0, lf) in list(self.running.items()):
            try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError): pass
        deadline = time.time() + 20
        while self.running and time.time() < deadline:
            self.poll(); time.sleep(0.5)
        for rid, (p, t0, lf) in list(self.running.items()):
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError): pass
            lf.close(); self.running.pop(rid)

    def run(self):
        t_start = time.time(); last = time.time()
        try:
            while True:
                self.poll()
                if self.failed and self.fail_fast: break
                for rid in self.runnable(): self.start(rid)
                pending = [i for i in self.ids if i not in self.done and i not in self.failed and i not in self.skipped and not self.blocked_by_failure(i)]
                for i in self.ids:
                    if i not in self.done and i not in self.failed and i not in self.skipped and self.blocked_by_failure(i): self.skipped.add(i); self.say('skip   %-30s (needs %s)' % (i, self.blocked_by_failure(i)))
                if not self.running and not [i for i in pending if i not in self.skipped]: break
                if time.time() - last > 600 and self.running:
                    last = time.time(); self.say('... %d running (%s), %d done of %d, %s elapsed' % (len(self.running), ', '.join(sorted(self.running)[:4]), len(self.done), len(self.ids), fmt_secs(time.time() - t_start)))
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.say('interrupted: stopping %d running commands (they resume from their checkpoints next time)' % len(self.running))
            self.stop_all()
            return False
        finally:
            if self.running: self.stop_all()
        return not self.failed and not self.skipped


def shown(path):
    """a path as short as it can be written from the repository root"""
    rel = os.path.relpath(path, ROOT)
    return path if rel.startswith('..') else rel


def fmt_secs(s):
    s = int(s)
    return '%dh%02dm%02ds' % (s // 3600, s % 3600 // 60, s % 60) if s >= 3600 else '%dm%02ds' % (s // 60, s % 60)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ cli
def print_status(runs, R, ids):
    by = collections.OrderedDict()
    for i in ids:
        r = runs[i]; st = read_status(R, i); d = is_done(R, r)
        c = by.setdefault(r.stage, collections.Counter()); c['total'] += 1
        if d: c['done'] += 1; c['seconds'] += (st or {}).get('seconds', 0)
        elif st and st.get('returncode') not in (0, None): c['failed'] += 1
    print('%-8s %6s %6s %6s %12s' % ('stage', 'runs', 'done', 'failed', 'time (done)'))
    for stg, c in by.items(): print('%-8s %6d %6d %6d %12s' % (stg, c['total'], c['done'], c['failed'], fmt_secs(c['seconds'])))
    todo = [i for i in ids if not is_done(R, runs[i])]
    print('%d of %d runs to do' % (len(todo), len(ids)))
    for i in todo[:40]: print('   ', i, '(failed: status %s)' % (read_status(R, i) or {}).get('returncode') if (read_status(R, i) or {}).get('returncode') not in (0, None) else '')
    if len(todo) > 40: print('    ... and %d more' % (len(todo) - 40))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split('\n', 2)[2])
    ap.add_argument('profile', choices=sorted(SP.PROFILES)); ap.add_argument('--runs', default='', help='the runs directory (default: runs/PROFILE)')
    ap.add_argument('--seeds', default='', help='e.g. 0-4 or 0,1,2 (default: the first `seeds` of the profile)'); ap.add_argument('--set', action='append', default=[], help='field=value of the profile; repeatable')
    ap.add_argument('--stage', default='', help='only these stages, comma separated: ' + ' '.join(['data', 'pre', 'ft', 'rl', 'eval', 'bench', 'report']))
    ap.add_argument('--target', default='', help='only these runs and what they need (comma separated ids; a trailing * matches a prefix: rl/ei_her/*)')
    ap.add_argument('--jobs', type=int, default=1); ap.add_argument('--cores', type=int, default=0); ap.add_argument('--mem-gb', type=float, default=0.0)
    ap.add_argument('--force', default='', help='run these ids again even if done (comma separated)'); ap.add_argument('--cascade', action='store_true', help='with --force: also what depends on them')
    ap.add_argument('--status', action='store_true'); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--list', action='store_true'); ap.add_argument('--fail-fast', action='store_true')
    a = ap.parse_args(argv)
    over = SP.parse_overrides(a.set)
    seeds = parse_seeds(a.seeds) if a.seeds else None
    R = a.runs or os.path.join('runs', a.profile)
    runs = SP.build(a.profile, seeds=seeds, runs_dir=R, overrides=over)
    Rabs = R if os.path.isabs(R) else os.path.join(ROOT, R)
    stages = [s for s in a.stage.split(',') if s]
    for s in stages:
        if s not in {r.stage for r in runs.values()}: ap.error('no stage %r' % s)
    targets = []
    for t in [t for t in a.target.split(',') if t]:
        if t.endswith('*'): targets += [i for i in runs if i.startswith(t[:-1])]
        elif t in runs: targets.append(t)
        else: ap.error('no run %r (see --list)' % t)
    ids = SP.closure(runs, targets) if targets else list(runs)
    if stages: ids = [i for i in ids if runs[i].stage in stages]
    if a.list or a.dry_run:
        for i in ids:
            r = runs[i]; print('%-34s %-7s %s%s' % (i, r.stage, ' '.join(r.argv) if a.dry_run else ','.join(r.needs), '' if is_done(Rabs, r) else ''))
        print(len(ids), 'runs'); return 0
    if a.status: print_status(runs, Rabs, ids); return 0
    os.makedirs(Rabs, exist_ok=True)
    prof_path = os.path.join(Rabs, 'PROFILE.json'); prof = SP.effective_profile(a.profile, over, seeds)
    if os.path.exists(prof_path):
        old = json.load(open(prof_path))
        diff = {k: (old.get(k), prof.get(k)) for k in set(old) | set(prof) if old.get(k) != prof.get(k)}
        results = {k: v for k, v in diff.items() if k not in MACHINE_FIELDS}
        if results and any(os.path.isdir(os.path.join(Rabs, d)) for d in ('pre', 'ft', 'rl', 'eval')):
            sys.exit('%s holds runs made with another profile: %s\nUse the same --set/--seeds as before, or another --runs directory (a profile change would mix results).' % (R, results))
        if diff and not results: print('note: %s differs from the previous invocation only in machine settings (%s)' % (R, ', '.join(sorted(diff))), flush=True)
    write_json(prof_path, prof); write_json(os.path.join(Rabs, 'GRAPH.json'), {i: r.to_dict() for i, r in runs.items()})
    # --force
    force = []
    for t in [t for t in a.force.split(',') if t]:
        force += [i for i in runs if i.startswith(t[:-1])] if t.endswith('*') else [t]
    for f in force:
        if f not in runs: ap.error('--force: no run %r' % f)
    todo_force = set(force) | (set().union(*[dependents(runs, f) for f in force]) if (a.cascade and force) else set())
    for i in todo_force:
        try: os.remove(status_path(Rabs, i))
        except OSError: pass
    cores = a.cores or (os.cpu_count() or 1); mem = a.mem_gb or 0.85 * mem_total_gb()
    lock = os.path.join(Rabs, '.lock')
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip()); os.kill(pid, 0)
            sys.exit('%s is being run by process %d (remove %s if that is wrong)' % (R, pid, lock))
        except (ValueError, ProcessLookupError, PermissionError): pass
    with open(lock, 'w') as f: f.write(str(os.getpid()))
    try:
        sch = Scheduler(runs, Rabs, ids, max(1, a.jobs), cores, mem, a.fail_fast)
        print('%s: %d runs selected, %d done; %d jobs, %d cores, %.0f GB' % (a.profile, len(ids), len(sch.done), a.jobs, cores, mem), flush=True)
        ok = sch.run()
    finally:
        try: os.remove(lock)
        except OSError: pass
    print_status(runs, Rabs, ids)
    if not ok: print('not everything succeeded: see the logs under %s/logs' % R); return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
