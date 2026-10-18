"""S45 gate: the repository's layout is the layout the documentation describes, and stays so.

The project is organised by ROLE (src/c++ and src/python/{authoring,checker,library,environment,problems,learner}; tests/{kernel,library,environment,problems,learner,repo,support};
mutants; examples; docs; data; legacy), while every module is imported by its plain name -- `import gym_env` -- because src/python/itp_paths.py puts each directory on sys.path.  That is convenient
and fragile in exactly the ways this gate watches:

  A. the gate registry (tests/gate_registry.py): every registered script exists, none is registered twice, and every gate script on disk is registered (a gate that nobody runs checks nothing);
  B. module names are UNIQUE across src/python, tests, mutants and tools (a duplicate would silently shadow another module), and every module imports in a FRESH interpreter started in an unrelated
     directory (so nothing depends on the working directory, on PYTHONPATH, or on the order of imports);
  C. every Python file of src/python/*/, tests and examples begins with the bootstrap line (`import itp_paths`) that makes `python3 path/to/file.py` work from anywhere;
  D. the frozen reference kernel is byte-for-byte the one whose checksums are recorded (src/c++/reference/SHA256SUMS);
  E. the Markdown documents: every relative link resolves, and tests/README.md lists every gate;
  F. no file mentions a path of the machine it was written on;
  G. requirements.txt names exactly the third-party packages the code imports;
  H. the required directories and files exist, and nothing else is lying around at the top level;
  I. the Makefile parses (`make -n` on each target) and its targets are the documented ones;
  J. the dataset on disk (data/dataset_v1): the manifest's digests are those of the files, and the files agree with each other;
  K. the release archive: tools/make_dist.py makes a tar and a zip that hold every file once, a manifest that checks, and nothing of the build or the machine;
  L. the copies of the tree in which the mutation driver and the sanitizer run the gates are complete (an unmutated copy passes a gate).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import ast, collections, glob, hashlib, json, os, re, shutil, subprocess, sys, tempfile
import gate_registry as REG

ROOT = itp_paths.ROOT
FAIL = []; CHECKS = [0]


def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


def rel(p): return os.path.relpath(p, ROOT)
def read(p):
    with open(p, encoding='utf-8') as f: return f.read()


def python_files(*patterns):
    return sorted(p for pat in patterns for p in glob.glob(os.path.join(ROOT, pat)) if '__pycache__' not in p)


SKIP_DIRS = {'.git', 'build', '__pycache__', 'artifacts', '.venv'}


def walk(top=ROOT):
    for d, dirs, files in os.walk(top):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files): yield os.path.join(d, f)


GATE_DIRS = ('kernel', 'library', 'environment', 'problems', 'learner', 'repo')
NOT_GATES = ('bench', 'fuzz', 'support')                        # scripts and helper modules under tests/ that are not gates


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. the registry
def sec_A():
    cmds = list(REG.GATE_CMDS) + list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS)
    paths = [c[0] for c in cmds]
    for c in cmds: expect(os.path.isfile(os.path.join(itp_paths.TESTS, c[0])), 'A  registered gate %s does not exist' % c[0])
    expect(len(paths) == len(set(paths)), 'A  a gate is registered twice: %s' % sorted(p for p, n in collections.Counter(paths).items() if n > 1))
    base = [os.path.basename(p) for p in paths]
    expect(len(base) == len(set(base)), 'A  two registered gates share a file name (run_gates.py selects gates by file name)')
    on_disk = {os.path.relpath(p, itp_paths.TESTS) for d in GATE_DIRS for p in glob.glob(os.path.join(itp_paths.TESTS, d, '*.py'))}
    expect(not (on_disk - set(paths)), 'A  scripts under tests/ that no registry lists: %s' % sorted(on_disk - set(paths)))
    expect(len(REG.GATE_CMDS) >= 46 and len(REG.ML_GATE_CMDS) >= 1 and len(REG.REPO_GATE_CMDS) >= 2, 'A  the registry has the kernel gates (%d), the learner gate and the repository gates' % len(REG.GATE_CMDS))
    expect(all(c in REG.REPO_GATE_CMDS for c in cmds if c[0].startswith('repo/')), 'A  the gates under tests/repo are the REPO_GATE_CMDS')
    for c in REG.REPO_GATE_CMDS: expect(c[0].startswith('repo/'), 'A  %s is registered as a repository gate but does not live in tests/repo' % c[0])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. modules
def sec_B():
    files = python_files('src/python/*.py', 'src/python/*/*.py', 'tests/*.py', 'tests/*/*.py', 'mutants/*.py', 'tools/*.py')
    names = collections.defaultdict(list)
    for p in files: names[os.path.basename(p)].append(rel(p))
    dup = {n: ps for n, ps in names.items() if len(ps) > 1}
    expect(not dup, 'B  module names are not unique: %s' % dup)
    # a fresh interpreter, started in an unrelated directory, imports every module of src/python and the shared modules of tests/support
    mods = [os.path.basename(p)[:-3] for p in python_files('src/python/*.py', 'src/python/*/*.py', 'tests/support/*.py')]
    code = ("import sys, importlib, traceback\n"
            "sys.path.insert(0, %r)\n"
            "import itp_paths\n"
            "bad = []\n"
            "for m in %r:\n"
            "    try: importlib.import_module(m)\n"
            "    except BaseException as e: bad.append((m, '%%s: %%s' %% (type(e).__name__, str(e)[:150])))\n"
            "import json; print(json.dumps(bad))\n") % (itp_paths.SRC_PY, mods)
    scratch = tempfile.mkdtemp(prefix='s45_')
    try:
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'ITP_BUILD_DIR')}
        r = subprocess.run([sys.executable, '-c', code], cwd=scratch, env=env, capture_output=True, text=True, timeout=300)
    finally: shutil.rmtree(scratch, ignore_errors=True)
    try: bad = json.loads(r.stdout.strip().split('\n')[-1])
    except Exception: bad = [('<interpreter>', (r.stdout + r.stderr)[-300:])]
    expect(not bad, 'B  modules that do not import in a fresh interpreter started elsewhere: %s' % bad[:5])
    expect(len(mods) >= 50, 'B  the import test covered %d modules' % len(mods))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. bootstrap line
def sec_C():
    files = python_files('src/python/*/*.py', 'tests/kernel/*.py', 'tests/library/*.py', 'tests/environment/*.py', 'tests/problems/*.py', 'tests/learner/*.py', 'tests/repo/*.py',
                         'tests/support/*.py', 'tests/fuzz/*.py', 'tests/bench/*.py', 'examples/*.py', 'examples/*/*.py')
    missing = [rel(p) for p in files if 'import itp_paths' not in read(p)]
    expect(not missing, 'C  files without the bootstrap line (import itp_paths): %s' % missing[:8])
    expect(len(files) >= 100, 'C  the bootstrap check covered %d files' % len(files))
    for p in python_files('src/python/*.py'):
        if os.path.basename(p) in ('itp_paths.py', 'build_kernel.py'): continue
        expect(False, 'C  src/python holds only itp_paths.py and build_kernel.py at its top level; found %s' % rel(p))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. the frozen kernel
def sec_D():
    d = os.path.join(itp_paths.SRC_CPP, 'reference')
    sums = {}
    for line in read(os.path.join(d, 'SHA256SUMS')).splitlines():
        if line.strip():
            h, name = line.split(None, 1); sums[name.strip().lstrip('*')] = h
    expect(set(sums) == {'itp_kernel.hpp', 'itp_bindings.cpp'}, 'D  SHA256SUMS lists the two files of the reference kernel (%s)' % sorted(sums))
    for name, h in sums.items():
        got = hashlib.sha256(open(os.path.join(d, name), 'rb').read()).hexdigest()
        expect(got == h, 'D  %s of the frozen reference kernel has changed (it is the oracle of the lockstep gates and must stay byte-identical)' % name)
    present = sorted(os.listdir(d))
    expect(sorted(present) == ['README.md', 'SHA256SUMS', 'itp_bindings.cpp', 'itp_kernel.hpp'], 'D  src/c++/reference holds the two sources, the checksums and a README (found %s)' % present)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. documents
LINK = re.compile(r'(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')


def sec_E():
    docs = [p for p in walk() if p.endswith('.md')]
    expect(len(docs) >= 12, 'E  the documents were found (%d)' % len(docs))
    for p in docs:
        for target in LINK.findall(read(p)):
            if target.startswith(('http://', 'https://', 'mailto:', '#')): continue
            path = target.split('#')[0]
            if path and not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(p), path))):
                expect(False, 'E  %s links to %s, which does not exist' % (rel(p), target))
    readme = read(os.path.join(itp_paths.TESTS, 'README.md'))
    for c in list(REG.GATE_CMDS) + list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS):
        expect(os.path.basename(c[0]) in readme, 'E  tests/README.md does not list the gate %s' % c[0])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. machine paths
MACHINE = re.compile('|'.join(['/' + 'home' + '/', '/' + 'mnt' + '/', '/' + 'root' + '/', '/tmp/' + 'claude', 'C:' + chr(92) + chr(92) + 'Users']))
TEXT = ('.py', '.md', '.json', '.txt', '.cpp', '.hpp', '.sh', '.cfg', '.toml', '.yml', '.yaml', 'Makefile', 'SHA256SUMS')


def sec_F():
    seen = 0
    for p in walk():
        if not p.endswith(TEXT): continue
        seen += 1
        text = read(p)
        m = MACHINE.search(text)
        if m: expect(False, 'F  %s mentions a machine-specific path (%r)' % (rel(p), text[max(0, m.start() - 20):m.end() + 30].replace('\n', ' ')))
    expect(seen >= 150, 'F  %d text files were scanned' % seen)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ G. requirements
def sec_G():
    local = {os.path.basename(p)[:-3] for p in python_files('src/python/*.py', 'src/python/*/*.py', 'tests/*.py', 'tests/*/*.py', 'mutants/*.py', 'legacy/*.py', 'tools/*.py')}
    local |= {'itp_env_lnl', 'itp_env_ref', 'itp_env'}                                    # the compiled kernels
    third = set()
    for p in python_files('src/python/*.py', 'src/python/*/*.py', 'tests/*.py', 'tests/*/*.py', 'mutants/*.py', 'tools/*.py', 'examples/*.py'):
        for node in ast.walk(ast.parse(read(p))):
            if isinstance(node, ast.Import): roots = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module: roots = [node.module.split('.')[0]]
            else: continue
            third |= {r for r in roots if r not in sys.stdlib_module_names and r not in local}
    req = set()
    for line in read(os.path.join(ROOT, 'requirements.txt')).splitlines():
        line = line.split('#')[0].strip()
        if line: req.add(re.split(r'[<>=!~\[ ;]', line)[0].lower().replace('_', '-'))
    expect({t.lower().replace('_', '-') for t in third} == req, 'G  requirements.txt (%s) is not the set of third-party imports (%s)' % (sorted(req), sorted(third)))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ H. layout
REQUIRED_FILES = ['README.md', 'Makefile', 'requirements.txt', '.gitignore',
                  'src/c++/hybrid/itp_kernel.hpp', 'src/c++/hybrid/itp_bindings.cpp', 'src/c++/reference/itp_kernel.hpp', 'src/c++/reference/itp_bindings.cpp', 'src/c++/reference/README.md',
                  'src/python/itp_paths.py', 'src/python/build_kernel.py',
                  'tests/README.md', 'tests/run_gates.py', 'tests/gate_registry.py', 'mutants/README.md', 'mutants/mutate.py', 'mutants/catalogue.py', 'legacy/README.md', 'tools/sanitize.py',
                  'docs/FINDINGS.md', 'docs/ROADMAP.md', 'docs/BANK.md', 'docs/ACTIONS.md', 'docs/KERNEL_DESIGN.md', 'docs/ARCHITECTURE_RL.md', 'docs/TRAINING.md',
                  'data/dataset_v1/bank.json', 'data/dataset_v1/splits.json', 'data/dataset_v1/library.json', 'data/dataset_v1/manifest.json', 'data/dataset_v1/DATASET_CARD.md']
REQUIRED_DIRS = ['src/python/authoring', 'src/python/checker', 'src/python/library', 'src/python/environment', 'src/python/problems', 'src/python/learner',
                 'tests/kernel', 'tests/library', 'tests/environment', 'tests/problems', 'tests/learner', 'tests/repo', 'tests/support', 'tests/golden', 'examples', 'data/results']
TOP_LEVEL = {'README.md', 'Makefile', 'requirements.txt', '.gitignore', '.git', 'build', 'dist', 'artifacts', 'src', 'tests', 'mutants', 'examples', 'docs', 'data', 'legacy', 'tools', '.venv',
             'MANIFEST.sha256', 'RELEASE.txt'}                      # (the last two: what tools/make_dist.py adds to a release archive, hence present in an extracted one)


def sec_H():
    for f in REQUIRED_FILES: expect(os.path.isfile(os.path.join(ROOT, f)), 'H  missing file %s' % f)
    for d in REQUIRED_DIRS: expect(os.path.isdir(os.path.join(ROOT, d)), 'H  missing directory %s' % d)
    stray = sorted(set(os.listdir(ROOT)) - TOP_LEVEL)
    expect(not stray, 'H  files or directories at the top level that belong somewhere else: %s' % stray)
    examples = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'examples', '[0-9][0-9]_*.py')))
    expect(len(examples) >= 7, 'H  the numbered examples are there (%s)' % examples)
    numbers = [e[:2] for e in examples]
    expect(numbers == sorted(set(numbers)), 'H  the examples are numbered once each (%s)' % numbers)
    expect(not [p for p in glob.glob(os.path.join(ROOT, 'src', 'python', '*', '*')) if os.path.isdir(p) and '__pycache__' not in p], 'H  src/python is two levels deep: a role directory holds modules, not directories')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ I. Makefile
def sec_I():
    text = read(os.path.join(ROOT, 'Makefile'))
    targets = re.findall(r'^([a-z][a-z0-9_-]*):', text, flags=re.M)
    wanted = {'help', 'kernel', 'gates', 'gates-ml', 'gates-repo', 'dataset-verify', 'examples', 'mutants-check', 'dist', 'clean'}
    expect(wanted <= set(targets), 'I  the Makefile lacks targets %s' % sorted(wanted - set(targets)))
    if shutil.which('make') is None: return
    for t in targets:
        r = subprocess.run(['make', '-n', t], cwd=ROOT, capture_output=True, text=True, timeout=60)
        expect(r.returncode == 0, 'I  `make -n %s` fails: %s' % (t, (r.stderr or r.stdout)[-200:]))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ J. the dataset on disk
def sec_J():
    d = os.path.join(ROOT, 'data', 'dataset_v1')
    man = json.load(open(os.path.join(d, 'manifest.json')))
    for name, want in man['files'].items():
        h = hashlib.sha256(open(os.path.join(d, name), 'rb').read()).hexdigest()[:16]
        expect(h == want, 'J  %s differs from the manifest (%s, not %s)' % (name, h, want))
    bank, splits = json.load(open(os.path.join(d, 'bank.json'))), json.load(open(os.path.join(d, 'splits.json')))
    names = [p['name'] for p in bank['problems']]
    expect(len(names) == len(set(names)) and len(names) >= 80, 'J  bank.json: %d distinct problems' % len(set(names)))
    in_splits = [n for s in splits['problems'].values() for n in s]
    expect(sorted(in_splits) == sorted(names), 'J  splits.json lists exactly the problems of bank.json, each once')
    by_split = collections.defaultdict(list)
    for p in bank['problems']: by_split[p['split']].append(p['name'])
    expect({s: sorted(v) for s, v in splits['problems'].items()} == {s: sorted(v) for s, v in by_split.items()}, 'J  the split recorded in each problem is the split of splits.json')
    fam = collections.defaultdict(set)
    for p in bank['problems']: fam[p['family']].add(p['split'])
    expect(all(len(s) == 1 for s in fam.values()), 'J  the problems of one family are in one split (%s)' % [f for f, s in fam.items() if len(s) > 1])
    expect(all('demonstration' in p and p['demonstration']['length'] == len(p['demonstration']['steps']) >= 1 for p in bank['problems']), 'J  every problem has a recorded proof')
    card = read(os.path.join(d, 'DATASET_CARD.md'))
    expect(('%d problems' % len(names)) in card, 'J  the dataset card states the number of problems (%d)' % len(names))
    for f in ('blind_train', 'blind_dev', 'blind_test'):
        rows = json.load(open(os.path.join(ROOT, 'data', 'results', f + '.json')))['rows']
        split = f.split('_')[1]
        expect(rows and {r['name'] for r in rows} <= set(splits['problems'][split]), 'J  data/results/%s.json describes problems of the %s split only' % (f, split))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ K. the release archive
def sec_K():
    """tools/make_dist.py: the archive (made here into a temporary directory, from this tree -- which may itself be an extracted archive) holds each file once, under one top-level directory, its MANIFEST
    checks against the bytes in the archive, it holds nothing of the build or of the machine, its top-level entries are the repository's, and the zip holds what the tar holds"""
    import tarfile, zipfile
    tmp = tempfile.mkdtemp(prefix='s45_dist_')
    try:
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'make_dist.py'), '--out', tmp, '--name', 'itp-s45'], capture_output=True, text=True, timeout=300)
        expect(r.returncode == 0, 'K  make_dist.py fails: %s' % (r.stderr or r.stdout)[-300:])
        tp, zp = os.path.join(tmp, 'itp-s45.tar.gz'), os.path.join(tmp, 'itp-s45.zip')
        if not (os.path.isfile(tp) and os.path.isfile(zp)): expect(False, 'K  make_dist.py made no archive'); return
        with tarfile.open(tp) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            names = [m.name for m in members]
            expect(len(names) == len(set(names)), 'K  a file occurs twice in the tar: %s' % sorted(n for n, c in collections.Counter(names).items() if c > 1)[:5])
            expect(all(n.startswith('itp-s45/') for n in names), 'K  every file of the tar is under the one top-level directory')
            inner = {n.split('/', 1)[1]: m for n, m in zip(names, members)}
            tops = {n.split('/')[0] for n in inner}
            expect(tops <= TOP_LEVEL, 'K  top-level entries of the archive that the repository does not have: %s' % sorted(tops - TOP_LEVEL))
            need = {'MANIFEST.sha256', 'RELEASE.txt', 'README.md', 'Makefile', 'requirements.txt', 'src', 'tests', 'examples', 'docs', 'data', 'mutants', 'tools', 'legacy'}
            expect(need <= tops, 'K  the archive lacks parts of the repository: %s' % sorted(need - tops))
            bad = [n for n in inner if n.endswith(('.so', '.pyc', '.o', '.pyd')) or set(n.split('/')) & {'.git', 'build', '__pycache__', 'dist', '.venv'}]
            expect(not bad, 'K  build products or caches in the archive: %s' % bad[:5])
            man = {}
            for line in tf.extractfile(inner['MANIFEST.sha256']).read().decode().splitlines():
                digest, _, path = line.partition('  ')
                man[path] = digest
            expect(set(man) == set(inner) - {'MANIFEST.sha256', 'RELEASE.txt'}, 'K  the manifest lists exactly the other files (missing %s, extra %s)' % (sorted(set(inner) - set(man) - {'MANIFEST.sha256', 'RELEASE.txt'})[:3], sorted(set(man) - set(inner))[:3]))
            wrong = [p for p, d in man.items() if p in inner and hashlib.sha256(tf.extractfile(inner[p]).read()).hexdigest() != d]
            expect(not wrong, 'K  the manifest\'s digest differs from the archived bytes of %s' % wrong[:5])
            rel_txt = tf.extractfile(inner['RELEASE.txt']).read().decode()
            expect(rel_txt.startswith('commit ') and 'files %d' % len(man) in rel_txt, 'K  RELEASE.txt names the commit and the number of files (%r)' % rel_txt[:120])
        with zipfile.ZipFile(zp) as zf:
            expect(sorted(n for n in zf.namelist() if not n.endswith('/')) == sorted(names), 'K  the zip holds the files of the tar')
        if os.path.isdir(os.path.join(ROOT, '.git')) and shutil.which('git'):
            r = subprocess.run(['git', '-C', ROOT, 'ls-files', '-co', '--exclude-standard'], capture_output=True, text=True)
            if r.returncode == 0:
                want = {n for n in r.stdout.split('\n') if n and os.path.isfile(os.path.join(ROOT, n))}
                got = set(inner) - {'MANIFEST.sha256', 'RELEASE.txt'}
                expect(got == want, 'K  the archive holds the files git knows (missing %s, extra %s)' % (sorted(want - got)[:3], sorted(got - want)[:3]))
    finally: shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ L. copies of the tree
def sec_L():
    """the mutation driver and the sanitizer run the gates in COPIES of the tree.  A copy that lacks something the gates need (it once lacked legacy/) makes every gate fail, and a mutation run then reports
    every mutant as caught -- the worst failure a test of tests can have, because it looks like success.  So: the copy lists are one list (itp_paths.COPIED_DIRS), and the driver's own self-test -- an
    unmutated copy passes a gate, the same copy without legacy/ fails it -- passes."""
    for d in itp_paths.COPIED_DIRS: expect(os.path.isdir(os.path.join(ROOT, d)), 'L  COPIED_DIRS names %s, which is not a directory of the repository' % d)
    for f in ('mutants/mutate.py', 'tools/sanitize.py'): expect('itp_paths.COPIED_DIRS' in read(os.path.join(ROOT, f)), 'L  %s does not copy the directories of itp_paths.COPIED_DIRS' % f)
    for d in ('src', 'tests', 'legacy'): expect(d in itp_paths.COPIED_DIRS, 'L  COPIED_DIRS lacks %s' % d)
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'mutants', 'mutate.py'), '--selftest'], capture_output=True, text=True, timeout=600)
    expect(r.returncode == 0, 'L  `mutate.py --selftest` fails: %s' % (r.stdout + r.stderr)[-400:].replace('\n', ' | '))


def main():
    import time
    t0 = time.time()
    for name, fn in (('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E), ('F', sec_F), ('G', sec_G), ('H', sec_H), ('I', sec_I), ('J', sec_J), ('K', sec_K), ('L', sec_L)):
        t1 = time.time(); n0, f0 = CHECKS[0], len(FAIL)
        try: fn()
        except Exception as e:
            import traceback
            FAIL.append('%s  section crashed: %s: %s\n%s' % (name, type(e).__name__, e, traceback.format_exc()[-600:]))
        if os.environ.get('S45_TRACE'): print('   [section %s: %d checks, %d failures, %.1f s]' % (name, CHECKS[0] - n0, len(FAIL) - f0, time.time() - t1), flush=True)
    print('%d checks' % CHECKS[0])
    if FAIL:
        print('S45 LAYOUT GATE: FAIL (%d)' % len(FAIL))
        for f in FAIL[:40]: print('   ', f)
        sys.exit(1)
    print('S45 LAYOUT GATE: PASS (%.1f s)' % (time.time() - t0))


if __name__ == '__main__':
    main()
