#!/usr/bin/env python3
"""mutate.py -- mutation testing: every mutant re-breaks ONE rule of the system in an isolated copy of the tree, and the gates must FAIL.

    python3 mutants/mutate.py --check              verify the catalogue: every mutant must replace text that occurs EXACTLY ONCE in its file, must change it, and no name is used twice
    python3 mutants/mutate.py --selftest           verify the machinery: an UNMUTATED copy of the tree, made the way mutants' copies are, passes a gate (a few seconds)
    python3 mutants/mutate.py PREFIX [PREFIX ...]  run the mutants whose name starts with a prefix (no prefix: all of them -- many hours)
    MUT_ONLY=s2_tests.py,s16_quot_tests.py ...     run only these gates (by file name) against every mutant
    MUT_FULL=1     run EVERY gate on a mutant (default: stop at the first gate that fails)      MUT_VERBOSE=1   say which gate caught a mutant
    MUT_KEEP=1     keep the mutated copy of the tree (the directories of itp_paths.COPIED_DIRS + build/, in the system temp directory) for inspection
    --jobs N       run N mutants at once (each in its own copy of the tree; a C++ mutant is rebuilt in its copy, at most MUT_BUILD_JOBS=2 builds at a time); the time a gate may take before it counts as a
                   hang grows with the oversubscription of the cores, and a catch by timeout is recorded as such
    --json FILE    append one JSON line per finished mutant to FILE (name, status, the gates that caught it and how: assertion / crash / timeout, seconds, the digest of the mutant and of the tree it was run on)
                   and, when FILE exists, SKIP the mutants it already holds for this very tree and this very mutant (a long run can be interrupted and continued); --redo runs them again

The catalogue is catalogue.py (name -> (old text, new text[, file])); the gates are those of tests/gate_registry.py (GATE_CMDS: the kernel, the tactic layer, the environment and the bank -- not the learner).
A mutant is (old, new) for the kernel header, or (old, new, file) for another source file, named by its BASE NAME (the layout is searched for it: src/c++/hybrid, src/python/*).  A C++ target is rebuilt
in the copy (about a minute); a Python target needs no rebuild.  The copy is self-contained -- the directories of itp_paths.COPIED_DIRS (src, tests, legacy, data) and a build/ directory -- because every path
is derived from the location of src/python/itp_paths.py.

A mutant that no gate catches is a gap in the tests, unless it is listed in KNOWN_EQUIVALENT with the reason why no input can tell it from the original.  The UNMUTATED COPY must be green on every gate first
(not the original tree: the baseline has to run in the same kind of copy as the mutants, otherwise a copy that lacks something the gates need makes every gate fail, and every mutant looks caught).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa
import ast, glob, hashlib, json, re, shutil, subprocess, tempfile, threading, time
import concurrent.futures as cf
import gate_registry as REG
from catalogue import MUTANTS, KNOWN_EQUIVALENT, FAIL_RE

ROOT = itp_paths.ROOT
KERNEL_HEADER = 'itp_kernel.hpp'


def locate(base):
    """the path (relative to the root) of the source file with this base name: the hybrid kernel's header and bindings, or a Python module under src/python"""
    hits = [os.path.relpath(p, ROOT) for pat in ('src/c++/hybrid/*', 'src/python/*.py', 'src/python/*/*.py') for p in glob.glob(os.path.join(ROOT, pat)) if os.path.basename(p) == base]
    if len(hits) != 1: raise SystemExit('mutant target %r: %d candidate files (%s)' % (base, len(hits), ', '.join(hits)))
    return hits[0]


def mutant_parts(spec):
    """a mutant is (old, new) for the kernel header, or (old, new, file) for another source file; returns (old, new, base name of the file)"""
    return spec[0], spec[1], (spec[2] if len(spec) > 2 else KERNEL_HEADER)


def duplicate_names():
    """a dict literal keeps only the LAST of two equal keys, so a mutant whose name is reused silently disappears: read the keys off the source text instead"""
    tree = ast.parse(open(os.path.join(ROOT, 'mutants', 'catalogue.py')).read()); seen, dup = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MUTANTS' for t in node.targets) and isinstance(node.value, ast.Dict):
            for k in node.value.keys:
                if isinstance(k, ast.Constant):
                    if k.value in seen: dup.append(k.value)
                    seen.add(k.value)
    return dup


def check_anchors():
    """every mutant must replace text that occurs EXACTLY ONCE in its file, and must change it; a stale anchor would silently turn a mutant into a no-op (or abort an hour-long run)"""
    texts, bad, same = {}, [], []
    for n, spec in MUTANTS.items():
        old, new, f = mutant_parts(spec)
        if f not in texts: texts[f] = open(os.path.join(ROOT, locate(f))).read()
        if texts[f].count(old) != 1: bad.append((n, texts[f].count(old)))
        if old == new: same.append(n)
    return bad, same


def gate_failed(rc, text): return rc != 0 or bool(FAIL_RE.search(text))


def gate_command(tree, cmd):
    path = os.path.join(tree, 'tests', cmd[0])
    return [sys.executable, path] + cmd[1:], os.path.dirname(path)


def clean_env():
    return {k: v for k, v in os.environ.items() if k != 'ITP_BUILD_DIR'}      # every tree finds its own build/ directory


def make_copy(d, rebuild):
    """a self-contained copy of the tree to mutate: the directories of itp_paths.COPIED_DIRS and a build/ with the reference kernel (frozen, never rebuilt) and, for a Python-only mutant, the hybrid kernel too"""
    ign = shutil.ignore_patterns('__pycache__', '*.pyc')
    for name in itp_paths.COPIED_DIRS: shutil.copytree(os.path.join(ROOT, name), os.path.join(d, name), ignore=ign)
    os.makedirs(os.path.join(d, 'build'))
    for so in glob.glob(os.path.join(itp_paths.BUILD, 'itp_env_*.so')) + glob.glob(os.path.join(itp_paths.BUILD, 'itp_env_*.pyd')):
        if not rebuild or os.path.basename(so).startswith('itp_env_ref'): shutil.copy2(so, os.path.join(d, 'build'))


def series_of(name):
    """the series of a mutant: the letters its name starts with (`GE108 menu: ...` is of series GE)"""
    m = re.match(r'[A-Za-z]+', name); return m.group(0) if m else ''


def mutant_digest(nm):
    """what the mutant IS (its name, the text it replaces, the text it puts there, the file): a result saved for another digest is not a result for this mutant"""
    old, new, base = mutant_parts(MUTANTS[nm]); return hashlib.sha256('\0'.join((nm, old, new, base)).encode()).hexdigest()[:16]


def tree_digest():
    """the digest of the code the gates run on (itp_paths.CODE_PATHS: every file below src/, tests/ and mutants/ but caches, built files and .md documents, and the manifest of the dataset): a saved result is reused only for the tree it was measured on.
    A document (the paper, a tool, a script, or a .md file anywhere) is not part of it"""
    h = hashlib.sha256()
    for top in itp_paths.CODE_DIRS:
        for root, dirs, files in os.walk(os.path.join(ROOT, top)):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__')
            for fn in sorted(files):
                if fn.endswith(('.pyc', '.so', '.pyd', '.o') + tuple(itp_paths.DOC_SUFFIXES)): continue
                p = os.path.join(root, fn); h.update(os.path.relpath(p, ROOT).replace(os.sep, '/').encode() + b'\0')
                with open(p, 'rb') as f: h.update(hashlib.sha256(f.read()).digest())
    for rel in itp_paths.CODE_FILES:
        man = os.path.join(ROOT, *rel.split('/'))
        if os.path.exists(man):
            with open(man, 'rb') as f: h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:16]


def catch_kind(text):
    """how a gate caught a mutant: `timeout` (it did not stop), `crash` (it raised instead of checking), `assertion` (a check of the gate failed)"""
    if 'HANG (timeout)' in text: return 'timeout'
    return 'crash' if 'Traceback' in text and 'GATE: FAIL' not in text else 'assertion'


def read_results(path):
    """the last record of every mutant in a results file (JSON lines); a line that does not parse (a write that was cut short) is skipped"""
    out = {}
    if path and os.path.exists(path):
        for line in open(path):
            try: r = json.loads(line)
            except ValueError: continue
            if isinstance(r, dict) and 'name' in r: out[r['name']] = r
    return out


def baseline(cmds):
    """the unmutated COPY must be green on every gate, otherwise every mutant would look "caught" and the results would mean nothing.  It must be a copy made like the mutants' copies, not the tree itself: a copy
    that lacks something the gates need (legacy/, once) makes every gate fail, and the baseline run in the original tree would never show it.  Returns how long each gate takes on the unmutated build (under today's load)"""
    base_time = {}
    d0 = os.path.join(tempfile.gettempdir(), 'mut_baseline')
    if os.path.exists(d0): shutil.rmtree(d0)
    make_copy(d0, False)
    try:
        for cmd in cmds:
            t_b = time.time(); argv, cwd = gate_command(d0, cmd)
            r = subprocess.run(argv, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=400)
            base_time[cmd[0]] = time.time() - t_b
            if gate_failed(r.returncode, r.stdout + r.stderr):
                print("BASELINE IS RED on %s (in an unmutated copy of the tree) -- mutation results would be meaningless" % cmd[0]); print((r.stdout + r.stderr)[-500:]); sys.exit(2)
    finally: shutil.rmtree(d0, ignore_errors=True)
    return base_time


def run_mutant(nm, cmds, base_time, slow, builds):
    """one mutant: a copy of the tree with its change, the gates in order (the first that fails catches it; MUT_FULL=1 runs them all); returns the record"""
    old, new, base = mutant_parts(MUTANTS[nm]); d = os.path.join(tempfile.gettempdir(), 'mut_' + re.sub(r'\W+', '_', nm)[:24])
    rebuild = base.endswith(('.hpp', '.cpp'))
    rec = {'name': nm, 'id': nm.split()[0], 'series': series_of(nm), 'file': base, 'digest': mutant_digest(nm), 'caught_by': [], 'gates_run': []}
    if os.path.exists(d): shutil.rmtree(d)
    make_copy(d, rebuild)
    target = os.path.join(d, locate(base))
    h = open(target).read(); assert h.count(old) == 1, (nm, h.count(old))
    open(target, 'w').write(h.replace(old, new))
    t0 = time.time()
    if rebuild:
        with builds:
            b = subprocess.run([sys.executable, os.path.join(d, 'src', 'python', 'build_kernel.py'), 'hybrid', '--out', os.path.join(d, 'build')], cwd=d, env=clean_env(), capture_output=True, text=True)
        if 'Build succeeded' not in b.stdout:
            rec.update(status='build_failed', detail=b.stdout[-200:] + b.stderr[-200:], seconds=round(time.time() - t0, 1)); shutil.rmtree(d, ignore_errors=True); return rec
    for cmd in cmds:
        argv, cwd = gate_command(d, cmd); rec['gates_run'].append(os.path.basename(cmd[0]))
        try:
            r = subprocess.run(argv, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=max(100, 3 * base_time[cmd[0]] * slow))
            rc, text = r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            rc, text = 99, 'Traceback: HANG (timeout) -- the mutant does not terminate'
        if gate_failed(rc, text):
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            first = next((l[:110] for l in lines if FAIL_RE.search(l)), '')
            if first.startswith('Traceback') and lines: first = ('Traceback ... ' + lines[-1])[:150]          # (what raised, not just that something did)
            rec['caught_by'].append({'gate': os.path.basename(cmd[0]), 'kind': catch_kind(text), 'first': first})
            if not os.environ.get('MUT_FULL'): break           # MUT_FULL=1 runs every gate to show how many independently catch the mutant
    rec['status'] = 'caught' if rec['caught_by'] else ('equivalent' if nm in KNOWN_EQUIVALENT else 'gap')
    rec['seconds'] = round(time.time() - t0, 1)
    if not os.environ.get('MUT_KEEP'): shutil.rmtree(d, ignore_errors=True)          # (a mutant's copy of the tree is 6 MB: hundreds of them fill the disk; MUT_KEEP=1 keeps them for inspection)
    return rec


STATUS_TEXT = {'caught': 'CAUGHT', 'equivalent': 'SURVIVED (documented equivalent mutant)', 'gap': 'SURVIVED  <-- test gap', 'build_failed': 'BUILD FAILED'}


def run(names, jobs=1, json_path=None, redo=False):
    bad, same = check_anchors()
    if duplicate_names(): print("MUTANT NAMES USED TWICE (the later silently replaces the earlier):", duplicate_names()); sys.exit(2)
    if bad or same:
        print("STALE MUTANT ANCHORS (fix them before running):"); [print("   %d occurrences: %s" % (c, n)) for n, c in bad]; [print("   no-op mutant: %s" % n) for n in same]; sys.exit(2)
    cmds = [c for c in REG.GATE_CMDS if not os.environ.get('MUT_ONLY') or os.path.basename(c[0]) in os.environ['MUT_ONLY'].split(',')]
    tree = tree_digest(); saved = {} if redo else read_results(json_path); results = []; todo = []
    for nm in names:
        r = saved.get(nm)
        if r and r.get('digest') == mutant_digest(nm) and r.get('tree') == tree and r.get('gates') == [os.path.basename(c[0]) for c in cmds] and r.get('full') == bool(os.environ.get('MUT_FULL')) and r.get('status') in STATUS_TEXT:
            results.append(r); print("  %-62s %s  (saved)" % (nm, STATUS_TEXT[r['status']]), flush=True)
        else: todo.append(nm)
    if todo:
        base_time = baseline(cmds)
        print("baseline green on all %d gates (in an unmutated copy)" % len(cmds), flush=True)
    cores = usable_cores(); slow = max(1.0, jobs / float(cores)); builds = threading.BoundedSemaphore(int(os.environ.get('MUT_BUILD_JOBS', '2'))); lock = threading.Lock()
    meta = {'tree': tree, 'gates': [os.path.basename(c[0]) for c in cmds], 'full': bool(os.environ.get('MUT_FULL')), 'jobs': jobs}
    def one(nm):
        rec = run_mutant(nm, cmds, base_time, slow, builds); rec.update(meta); rec['time'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        with lock:
            if json_path:
                with open(json_path, 'a') as f: f.write(json.dumps(rec, sort_keys=True) + '\n')
            caught_by = rec['caught_by']
            print("  %-62s %s  (%.0fs)%s" % (nm, STATUS_TEXT[rec['status']], rec['seconds'], ('  | ' + ', '.join('%s: %s' % (c['gate'], c['first']) for c in caught_by)) if caught_by and os.environ.get('MUT_VERBOSE') else ''), flush=True)
        return rec
    if jobs <= 1: results += [one(nm) for nm in todo]
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool: results += list(pool.map(one, todo))
    order = {nm: i for i, nm in enumerate(names)}; results.sort(key=lambda r: order[r['name']])
    return results


def usable_cores():
    try: return max(1, len(os.sched_getaffinity(0)))
    except AttributeError: return max(1, os.cpu_count() or 1)


def selftest():
    """an UNMUTATED copy of the tree, made the way the mutants' copies are, passes a quick gate (S16, a few seconds), and the same gate fails in a copy that lacks legacy/ -- so a copy's deficiency is
    visible as what it is, and the baseline of `run` is meaningful.  Returns the list of problems."""
    problems = []; cmd = next(c for c in REG.GATE_CMDS if os.path.basename(c[0]) == 's16_quot_tests.py')
    d = os.path.join(tempfile.gettempdir(), 'mut_selftest')
    if os.path.exists(d): shutil.rmtree(d)
    try:
        make_copy(d, False)
        argv, cwd = gate_command(d, cmd)
        r = subprocess.run(argv, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=300)
        if gate_failed(r.returncode, r.stdout + r.stderr): problems.append('an unmutated copy fails %s: %s' % (cmd[0], (r.stdout + r.stderr)[-300:].replace('\n', ' | ')))
        if os.path.isdir(os.path.join(d, 'legacy')):
            shutil.rmtree(os.path.join(d, 'legacy'))
            r = subprocess.run(argv, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=300)
            if not gate_failed(r.returncode, r.stdout + r.stderr): problems.append('%s passes in a copy WITHOUT legacy/: the gates do not need it any more -- update itp_paths.COPIED_DIRS' % cmd[0])
    finally: shutil.rmtree(d, ignore_errors=True)
    return problems


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        probs = selftest()
        print('self-test: %s' % ('an unmutated copy passes the gate, a copy without legacy/ fails it' if not probs else 'FAILED'))
        for pr in probs: print('   ' + pr)
        sys.exit(1 if probs else 0)
    if '--check' in sys.argv:
        bad, same = check_anchors(); dup = duplicate_names(); print("%d mutants; %d stale anchors; %d no-op mutants; %d reused names" % (len(MUTANTS), len(bad), len(same), len(dup)))
        for n, c in bad: print("   %d occurrences: %s" % (c, n))
        for n in dup: print("   name used twice: %s" % n)
        sys.exit(1 if bad or same or dup else 0)
    argv = sys.argv[1:]; jobs, json_path, redo, prefixes = 1, None, False, []
    it = iter(argv)
    for a in it:
        if a == '--jobs': jobs = max(1, int(next(it)))
        elif a == '--json': json_path = next(it)
        elif a == '--redo': redo = True
        elif a.startswith('--'): print('unknown option %s' % a); sys.exit(2)
        else: prefixes.append(a)
    names = [n for n in MUTANTS if any(n.startswith(p) for p in prefixes)] if prefixes else list(MUTANTS)
    res = run(names, jobs=jobs, json_path=json_path, redo=redo)
    caught = sum(1 for r in res if r['status'] == 'caught'); eq = sum(1 for r in res if r['status'] == 'equivalent'); gaps = [r['name'] for r in res if r['status'] in ('gap', 'build_failed')]
    print("\n%d mutants: %d caught, %d documented equivalent, %d unexplained%s" % (len(res), caught, eq, len(gaps), ('   <-- ' + '; '.join(gaps)) if gaps else ''))
    sys.exit(1 if gaps else 0)
