#!/usr/bin/env python3
"""mutate.py -- mutation testing: every mutant re-breaks ONE rule of the system in an isolated copy of the tree, and the gates must FAIL.

    python3 mutants/mutate.py --check              verify the catalogue: every mutant must replace text that occurs EXACTLY ONCE in its file, must change it, and no name is used twice
    python3 mutants/mutate.py --selftest           verify the machinery: an UNMUTATED copy of the tree, made the way mutants' copies are, passes a gate (a few seconds)
    python3 mutants/mutate.py PREFIX [PREFIX ...]  run the mutants whose name starts with a prefix (no prefix: all of them -- many hours)
    MUT_ONLY=s2_tests.py,s16_quot_tests.py ...     run only these gates (by file name) against every mutant
    MUT_FULL=1     run EVERY gate on a mutant (default: stop at the first gate that fails)      MUT_VERBOSE=1   say which gate caught a mutant
    MUT_KEEP=1     keep the mutated copy of the tree (the directories of itp_paths.COPIED_DIRS + build/, in the system temp directory) for inspection

The catalogue is catalogue.py (name -> (old text, new text[, file])); the gates are those of tests/gate_registry.py (GATE_CMDS: the kernel, the tactic layer, the environment and the bank -- not the learner).
A mutant is (old, new) for the kernel header, or (old, new, file) for another source file, named by its BASE NAME (the layout is searched for it: src/c++/hybrid, src/python/*).  A C++ target is rebuilt
in the copy (about a minute); a Python target needs no rebuild.  The copy is self-contained -- the directories of itp_paths.COPIED_DIRS (src, tests, legacy, data) and a build/ directory -- because every path
is derived from the location of src/python/itp_paths.py.

A mutant that no gate catches is a gap in the tests, unless it is listed in KNOWN_EQUIVALENT with the reason why no input can tell it from the original.  The UNMUTATED COPY must be green on every gate first
(not the original tree: the baseline has to run in the same kind of copy as the mutants, otherwise a copy that lacks something the gates need makes every gate fail, and every mutant looks caught).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa
import ast, glob, re, shutil, subprocess, tempfile, time
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


def run(names):
    bad, same = check_anchors()
    if duplicate_names(): print("MUTANT NAMES USED TWICE (the later silently replaces the earlier):", duplicate_names()); sys.exit(2)
    if bad or same:
        print("STALE MUTANT ANCHORS (fix them before running):"); [print("   %d occurrences: %s" % (c, n)) for n, c in bad]; [print("   no-op mutant: %s" % n) for n in same]; sys.exit(2)
    results = []
    cmds = [c for c in REG.GATE_CMDS if not os.environ.get('MUT_ONLY') or os.path.basename(c[0]) in os.environ['MUT_ONLY'].split(',')]
    # the UNMUTATED COPY must be green on every gate, otherwise every mutant would look "caught" and the results would mean nothing.  It must be a copy made like the mutants' copies, not the tree itself: a copy
    # that lacks something the gates need (legacy/, once) makes every gate fail, and the baseline run in the original tree would never show it.
    base_time = {}                                                  # how long each gate takes on the unmutated build (under today's load): a mutant gets three times that, and at least 100 s, before it counts as a hang
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
    print("baseline green on all %d gates (in an unmutated copy)" % len(cmds), flush=True)
    for nm in names:
        old, new, base = mutant_parts(MUTANTS[nm]); d = os.path.join(tempfile.gettempdir(), 'mut_' + re.sub(r'\W+', '_', nm)[:24])
        rebuild = base.endswith(('.hpp', '.cpp'))
        if os.path.exists(d): shutil.rmtree(d)
        make_copy(d, rebuild)
        target = os.path.join(d, locate(base))
        h = open(target).read(); assert h.count(old) == 1, (nm, h.count(old))
        open(target, 'w').write(h.replace(old, new))
        t0 = time.time()
        if rebuild:
            b = subprocess.run([sys.executable, os.path.join(d, 'src', 'python', 'build_kernel.py'), 'hybrid', '--out', os.path.join(d, 'build')], cwd=d, env=clean_env(), capture_output=True, text=True)
            if 'Build succeeded' not in b.stdout: results.append((nm, 'BUILD FAILED', b.stdout[-200:] + b.stderr[-200:])); print("  %-62s BUILD FAILED" % nm, flush=True); continue
        caught_by = []
        for cmd in cmds:
            argv, cwd = gate_command(d, cmd)
            try:
                r = subprocess.run(argv, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=max(100, 3 * base_time[cmd[0]]))
                rc, text = r.returncode, r.stdout + r.stderr
            except subprocess.TimeoutExpired:
                rc, text = 99, 'Traceback: HANG (timeout) -- the mutant does not terminate'
            if gate_failed(rc, text):
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                first = next((l[:110] for l in lines if FAIL_RE.search(l)), '')
                if first.startswith('Traceback') and lines: first = ('Traceback ... ' + lines[-1])[:150]          # (what raised, not just that something did)
                caught_by.append((os.path.basename(cmd[0]), first))
                if not os.environ.get('MUT_FULL'): break           # MUT_FULL=1 runs every gate to show how many independently catch the mutant
        if caught_by: status = 'CAUGHT'
        elif nm in KNOWN_EQUIVALENT: status = 'SURVIVED (documented equivalent mutant)'
        else: status = 'SURVIVED  <-- test gap'
        results.append((nm, status, caught_by)); print("  %-62s %s  (%.0fs)%s" % (nm, status, time.time() - t0, ('  | ' + ', '.join('%s: %s' % c for c in caught_by)) if caught_by and os.environ.get('MUT_VERBOSE') else ''), flush=True)
        if not os.environ.get('MUT_KEEP'): shutil.rmtree(d, ignore_errors=True)          # (a mutant's copy of the tree is 6 MB: hundreds of them fill the disk; MUT_KEEP=1 keeps them for inspection)
    return results


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
    prefixes = [a for a in sys.argv[1:] if not a.startswith('--')]
    names = [n for n in MUTANTS if any(n.startswith(p) for p in prefixes)] if prefixes else list(MUTANTS)
    res = run(names)
    caught = sum(1 for r in res if r[1] == 'CAUGHT'); eq = sum(1 for r in res if r[1].startswith('SURVIVED (doc')); gaps = [r[0] for r in res if r[1].startswith('SURVIVED  <--') or r[1] == 'BUILD FAILED']
    print("\n%d mutants: %d caught, %d documented equivalent, %d unexplained%s" % (len(res), caught, eq, len(gaps), ('   <-- ' + '; '.join(gaps)) if gaps else ''))
    sys.exit(1 if gaps else 0)
