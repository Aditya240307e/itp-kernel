"""S45 gate: the repository's layout is the layout the documentation describes, and stays so.

The project is organised by ROLE (src/c++ and src/python/{authoring,checker,library,environment,problems,learner,synth,experiments}; tests/{kernel,library,environment,problems,learner,synth,repo,support};
mutants; examples; docs; data; legacy; scripts (the experiments, as shell scripts); paper (the TMLR manuscript); tools), while every module is imported by its plain name -- `import gym_env` -- because src/python/itp_paths.py puts each directory on sys.path.  That is convenient
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
  J. the dataset on disk (data/dataset_v2): the manifest's digests are those of the files, and the files agree with each other;
  K. the release archive: tools/make_dist.py makes a tar and a zip that hold every file once, a manifest that checks, and nothing of the build or the machine; with --git, a working repository (its history, branches and tags; no remote, user, hook or stash) in a small repository of its own;
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


GATE_DIRS = ('kernel', 'library', 'environment', 'problems', 'learner', 'synth', 'repo')
NOT_GATES = ('bench', 'fuzz', 'support')                        # scripts and helper modules under tests/ that are not gates


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. the registry
def sec_A():
    cmds = list(REG.GATE_CMDS) + list(REG.BANK_GATE_CMDS) + list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS) + list(REG.SYNTH_GATE_CMDS)
    paths = [c[0] for c in cmds]
    for c in cmds: expect(os.path.isfile(os.path.join(itp_paths.TESTS, c[0])), 'A  registered gate %s does not exist' % c[0])
    expect(len(paths) == len(set(paths)), 'A  a gate is registered twice: %s' % sorted(p for p, n in collections.Counter(paths).items() if n > 1))
    base = [os.path.basename(p) for p in paths]
    expect(len(base) == len(set(base)), 'A  two registered gates share a file name (run_gates.py selects gates by file name)')
    on_disk = {os.path.relpath(p, itp_paths.TESTS) for d in GATE_DIRS for p in glob.glob(os.path.join(itp_paths.TESTS, d, '*.py'))}
    expect(not (on_disk - set(paths)), 'A  scripts under tests/ that no registry lists: %s' % sorted(on_disk - set(paths)))
    expect(len(REG.GATE_CMDS) >= 46 and len(REG.ML_GATE_CMDS) >= 1 and len(REG.REPO_GATE_CMDS) >= 2, 'A  the registry has the kernel gates (%d), the learner gate and the repository gates' % len(REG.GATE_CMDS))
    expect(all(c in REG.REPO_GATE_CMDS for c in cmds if c[0].startswith('repo/')), 'A  the gates under tests/repo are the REPO_GATE_CMDS')
    expect(set(REG.HEAVY_GATES) <= set(base), 'A  every heavy gate (never run together with another one) is a registered gate (%s)' % sorted(set(REG.HEAVY_GATES) - set(base)))
    expect(len(REG.BANK_GATE_CMDS) >= 2 and all(c[0].startswith('problems/') for c in REG.BANK_GATE_CMDS) and not {tuple(c) for c in REG.BANK_GATE_CMDS} & {tuple(c) for c in REG.GATE_CMDS}, 'A  the gates of the contest problems (BANK_GATE_CMDS) are registered apart from the mutation driver\'s GATE_CMDS')
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
    files = python_files('src/python/*/*.py', 'tests/kernel/*.py', 'tests/library/*.py', 'tests/environment/*.py', 'tests/problems/*.py', 'tests/learner/*.py', 'tests/synth/*.py', 'tests/repo/*.py',
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
    for c in list(REG.GATE_CMDS) + list(REG.BANK_GATE_CMDS) + list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS) + list(REG.SYNTH_GATE_CMDS):
        expect(os.path.basename(c[0]) in readme, 'E  tests/README.md does not list the gate %s' % c[0])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. machine paths
MACHINE = re.compile('|'.join(['/' + 'home' + '/', '/' + 'mnt' + '/', '/' + 'root' + '/', '/tmp/' + 'claude', 'C:' + chr(92) + chr(92) + 'Users']))
TEXT = ('.py', '.md', '.json', '.txt', '.cpp', '.hpp', '.sh', '.cfg', '.toml', '.yml', '.yaml', '.tex', '.bib', '.example', 'Makefile', 'SHA256SUMS')


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
                  'tests/README.md', 'tests/run_gates.py', 'tests/gate_registry.py', 'mutants/README.md', 'mutants/mutate.py', 'mutants/catalogue.py', 'legacy/README.md', 'tools/sanitize.py', 'tools/make_dist.py', 'tools/check_env.py',
                  'docs/FINDINGS.md', 'docs/ROADMAP.md', 'docs/BANK.md', 'docs/ACTIONS.md', 'docs/KERNEL_DESIGN.md', 'docs/ARCHITECTURE_RL.md', 'docs/TRAINING.md', 'docs/GENERATOR.md', 'docs/EXPERIMENTS.md',
                  'scripts/env.sh', 'scripts/local.env.example', 'scripts/00_check_env.sh', 'scripts/01_build_dataset.sh', 'scripts/02_generate_synthetic.sh', 'scripts/03_pretrain_finetune.sh', 'scripts/04_reinforcement_learning.sh',
                  'scripts/05_evaluate.sh', 'scripts/06_benchmarks.sh', 'scripts/07_report.sh', 'scripts/run_all.sh', 'scripts/smoke_test.sh', 'scripts/status.sh',
                  'paper/main.tex', 'paper/main.bib', 'paper/Makefile', 'paper/tmlr.sty', 'paper/tmlr.bst', 'paper/generated/.gitkeep',
                  'data/dataset_v2/bank.json', 'data/dataset_v2/splits.json', 'data/dataset_v2/library.json', 'data/dataset_v2/manifest.json', 'data/dataset_v2/DATASET_CARD.md']
REQUIRED_DIRS = ['src/python/authoring', 'src/python/checker', 'src/python/library', 'src/python/environment', 'src/python/problems', 'src/python/learner', 'src/python/synth', 'src/python/experiments', 'scripts', 'paper',
                 'tests/kernel', 'tests/library', 'tests/environment', 'tests/problems', 'tests/learner', 'tests/synth', 'tests/repo', 'tests/support', 'tests/golden', 'examples', 'data/results']
TOP_LEVEL = {'README.md', 'Makefile', 'requirements.txt', '.gitignore', '.git', 'build', 'dist', 'artifacts', 'runs', 'src', 'tests', 'mutants', 'examples', 'docs', 'data', 'legacy', 'tools', 'scripts', 'paper', '.venv',
             'MANIFEST.sha256', 'RELEASE.txt'}                      # (runs: where the experiments write, untracked; the last two: what tools/make_dist.py adds to a release archive, hence present in an extracted one)


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
    wanted = {'help', 'kernel', 'gates', 'gates-ml', 'gates-repo', 'gates-synth', 'dataset-verify', 'examples', 'mutants-check', 'dist', 'clean', 'check-env', 'smoke', 'experiments', 'status', 'report', 'paper'}
    expect(wanted <= set(targets), 'I  the Makefile lacks targets %s' % sorted(wanted - set(targets)))
    if shutil.which('make') is None: return
    for t in targets:
        r = subprocess.run(['make', '-n', t], cwd=ROOT, capture_output=True, text=True, timeout=60)
        expect(r.returncode == 0, 'I  `make -n %s` fails: %s' % (t, (r.stderr or r.stdout)[-200:]))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ J. the dataset on disk
def sec_J():
    d = os.path.join(ROOT, 'data', 'dataset_v2')
    man = json.load(open(os.path.join(d, 'manifest.json')))
    for name, want in man['files'].items():
        h = hashlib.sha256(open(os.path.join(d, name), 'rb').read()).hexdigest()[:16]
        expect(h == want, 'J  %s differs from the manifest (%s, not %s)' % (name, h, want))
    bank, splits = json.load(open(os.path.join(d, 'bank.json'))), json.load(open(os.path.join(d, 'splits.json')))
    names = [p['name'] for p in bank['problems']]
    expect(len(names) == len(set(names)) == 268, 'J  bank.json: %d distinct problems (268 in dataset v2)' % len(set(names)))
    in_splits = [n for s in splits['problems'].values() for n in s]
    expect(sorted(in_splits) == sorted(names), 'J  splits.json lists exactly the problems of bank.json, each once')
    by_split = collections.defaultdict(list)
    for p in bank['problems']: by_split[p['split']].append(p['name'])
    expect({s: sorted(v) for s, v in splits['problems'].items()} == {s: sorted(v) for s, v in by_split.items()}, 'J  the split recorded in each problem is the split of splits.json')
    fam = collections.defaultdict(set)
    for p in bank['problems']: fam[p['family']].add(p['split'])
    expect(all(len(s) == 1 for s in fam.values()), 'J  the problems of one family are in one split (%s)' % [f for f, s in fam.items() if len(s) > 1])
    proved = [p for p in bank['problems'] if p.get('proved', True)]
    expect(len(proved) == 187 and all('demonstration' in p and p['demonstration']['length'] == len(p['demonstration']['steps']) >= 1 for p in proved), 'J  every proved problem (%d) has a recorded proof' % len(proved))
    expect(all('demonstration' not in p and p['gid'] is None for p in bank['problems'] if not p.get('proved', True)), 'J  a statement-only problem has no recorded proof and no theorem')
    expect(all(p.get('origin') in ('contest', 'adapted', 'style', 'folklore') and p.get('verified') in ('answer-checked', 'recalled', 'none') for p in bank['problems']), 'J  every problem says where it comes from (origin) and whether its text or answer was compared with a source (verified)')
    card = read(os.path.join(d, 'DATASET_CARD.md'))
    expect(('%d problems' % len(names)) in card, 'J  the dataset card states the number of problems (%d)' % len(names))
    # the numbers the documents state are the numbers of the data (F83): a document that outlives the data it describes is found here
    nprob, nproved = len(names), len(proved); nhelp = sum(1 for p in bank['problems'] if p.get('helpers'))
    lib = json.load(open(os.path.join(d, 'library.json'))); lsp = splits['lemmas']; nlib = len(lib['lemmas']); psp = splits['problems']
    psizes = '%d train / %d dev / %d test' % (len(psp['train']), len(psp['dev']), len(psp['test']))
    lsizes = 'train %d / dev %d / test %d' % (len(lsp['train']), len(lsp['dev']), len(lsp['test']))
    expect(nlib == sum(len(v) for v in lsp.values()), 'J  splits.json lists the library lemmas of library.json (%d)' % nlib)
    stated = {'README.md': ['**%d formal problems**' % nprob, '**%d have a kernel-checked reference proof**' % nproved],
              'docs/TRAINING.md': ['**%d proved**' % nproved, 'train %d / dev %d / test %d' % (len(psp['train']), len(psp['dev']), len(psp['test'])), '%d (%s)' % (nlib, lsizes), '%d of the %d' % (nhelp, nprob)],
              'docs/BANK.md': ['%d problems' % nprob, psizes, '%d demonstrations' % nproved, "%d lemmas" % nlib, lsizes],
              'docs/EXPERIMENTS.md': ['Only %d of the %d problems' % (nhelp, nprob)]}
    for doc, phrases in stated.items():
        text = read(os.path.join(ROOT, doc))
        for ph in phrases: expect(ph in text, 'J  %s does not state "%s" (the numbers of data/dataset_v2)' % (doc, ph))
    # the result files of the first runs (dataset v1) are a historical record under data/results/v1: they say so, and they are not read as results on v2
    rd = os.path.join(ROOT, 'data', 'results')
    expect(os.path.isfile(os.path.join(rd, 'README.md')) and 'dataset v1' in read(os.path.join(rd, 'README.md')), 'J  data/results/README.md says that the result files are those of dataset v1')
    expect(sorted(os.listdir(os.path.join(rd, 'v1'))) == ['bc1_history.json', 'blind_dev.json', 'blind_test.json', 'blind_train.json', 'rl_smoke_history.json', 'smoke_policy.json'], 'J  data/results/v1 holds the six result files of the first runs')
    for f in ('blind_train', 'blind_dev', 'blind_test'):
        res = json.load(open(os.path.join(rd, 'v1', f + '.json')))
        expect(res['rows'] and res['args']['data'] == 'data/dataset_v1' and all('name' in r for r in res['rows']), 'J  data/results/v1/%s.json describes a run on dataset v1' % f)


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
            need = {'MANIFEST.sha256', 'RELEASE.txt', 'README.md', 'Makefile', 'requirements.txt', 'src', 'tests', 'examples', 'docs', 'data', 'mutants', 'tools', 'legacy', 'scripts', 'paper'}
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
    if shutil.which('git'): k_git(); k_add()


def k_git():
    """make_dist.py --git, in a small repository of its own (the tool finds its repository from its own location, so the test copies it there; this also works in a copy of the tree that has no .git): the
    extracted directory, from the tar and from the zip, is a working repository at the commit of the archive -- its history, branches and tags, clean status, `git fsck`, a commit can be made -- and it holds
    nothing of the machine that made it: no remote, no user, no hook, no stash, no remote-tracking branch.  Git does not accept a repository without refs/heads, which is empty: the archive must hold
    directories.  An archive made from a tree with uncommitted changes says so, and the changes show in `git status`."""
    import tarfile, zipfile
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0')
    def git(cwd, *args, check=True):
        r = subprocess.run(['git', '-C', cwd] + list(args), capture_output=True, text=True, env=env)
        if check and r.returncode != 0: raise RuntimeError('git %s: %s' % (' '.join(args), (r.stderr or r.stdout).strip()[:300]))
        return r
    tmp = tempfile.mkdtemp(prefix='s45_distgit_')
    try:
        repo = os.path.join(tmp, 'repo'); os.makedirs(os.path.join(repo, 'tools')); os.makedirs(os.path.join(repo, 'src'))
        shutil.copy(os.path.join(ROOT, 'tools', 'make_dist.py'), os.path.join(repo, 'tools', 'make_dist.py'))
        def put(rel, text, mode=0o644):
            with open(os.path.join(repo, rel), 'w') as f: f.write(text)
            os.chmod(os.path.join(repo, rel), mode)
        put('README.md', 'one\n'); put('src/a.py', 'print(1)\n'); put('run.sh', '#!/bin/sh\necho hi\n', 0o755)
        git(repo, 'init', '-q', '-b', 'main'); git(repo, 'config', 'user.name', 'Somebody Else'); git(repo, 'config', 'user.email', 'else@example.org')
        git(repo, 'add', '-A'); git(repo, 'commit', '-q', '-m', 'one')
        put('README.md', 'one\ntwo\n'); git(repo, 'commit', '-q', '-am', 'two'); git(repo, 'tag', 'v1'); git(repo, 'branch', 'other')
        git(repo, 'remote', 'add', 'origin', '../private-origin.git'); git(repo, 'update-ref', 'refs/remotes/origin/main', 'HEAD')
        put('src/a.py', 'print(2)\n'); git(repo, 'stash', '-q')                                           # (a stash: local work in progress that must not travel)
        os.makedirs(os.path.join(repo, '.git', 'hooks'), exist_ok=True)
        with open(os.path.join(repo, '.git', 'hooks', 'pre-commit'), 'w') as f: f.write('#!/bin/sh\nexit 1\n')
        os.chmod(os.path.join(repo, '.git', 'hooks', 'pre-commit'), 0o755)
        head = git(repo, 'rev-parse', 'HEAD').stdout.strip()
        def make(name, *extra):
            r = subprocess.run([sys.executable, os.path.join(repo, 'tools', 'make_dist.py'), '--git', '--out', os.path.join(tmp, 'out'), '--name', name] + list(extra), capture_output=True, text=True, env=env, timeout=300)
            expect(r.returncode == 0, 'K  make_dist.py --git fails: %s' % (r.stderr or r.stdout)[-300:])
            return os.path.join(tmp, 'out', name + '.tar.gz'), os.path.join(tmp, 'out', name + '.zip')
        tp, zp = make('pkg')
        if not (os.path.isfile(tp) and os.path.isfile(zp)): expect(False, 'K  make_dist.py --git made no archive'); return
        xt, xz = os.path.join(tmp, 'xt'), os.path.join(tmp, 'xz'); os.makedirs(xt); os.makedirs(xz)
        with tarfile.open(tp) as tf:
            tf.extractall(xt, filter='fully_trusted')                                                   # (our own archive, with its permissions)
            dirs = {m.name for m in tf.getmembers() if m.isdir()}
            expect({'pkg/.git/refs/heads', 'pkg/.git/refs/tags', 'pkg/.git/objects'} <= dirs, 'K  the tar holds the directories of .git, the empty ones too (refs/heads, refs/tags): %s' % sorted(d for d in dirs if '.git' in d)[:6])
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(xz)
            expect({'pkg/.git/refs/heads/', 'pkg/.git/refs/tags/'} <= set(zf.namelist()), 'K  the zip holds the directories of .git as well')
        for kind, x in (('tar', os.path.join(xt, 'pkg')), ('zip', os.path.join(xz, 'pkg'))):
            expect(git(x, 'rev-parse', '--is-inside-work-tree', check=False).stdout.strip() == 'true', 'K  the extracted %s is a git working tree' % kind)
            expect(git(x, 'rev-parse', 'HEAD', check=False).stdout.strip() == head and git(x, 'rev-list', '--count', 'HEAD', check=False).stdout.strip() == '2', 'K  the extracted %s is at the commit of the archive with its history' % kind)
            expect(git(x, 'fsck', '--no-dangling', check=False).returncode == 0, 'K  git fsck is happy with the extracted %s: %s' % (kind, git(x, 'fsck', check=False).stderr.strip()[:200]))
            branches = set(git(x, 'for-each-ref', '--format=%(refname)', check=False).stdout.split())
            expect(branches == {'refs/heads/main', 'refs/heads/other', 'refs/tags/v1'}, 'K  the %s holds the branches and tags and nothing else (no stash, no remote-tracking branch): %s' % (kind, sorted(branches)))
            expect(git(x, 'remote', check=False).stdout.strip() == '' and 'private-origin' not in open(os.path.join(x, '.git', 'config')).read(), 'K  the extracted %s has no remote (a path of the machine that made it)' % kind)
            expect(git(x, 'config', '--local', '--get', 'user.name', check=False).returncode != 0 and git(x, 'config', '--local', '--get', 'user.email', check=False).returncode != 0,
                   'K  the extracted %s carries no user identity' % kind)
            hooks = os.path.join(x, '.git', 'hooks')
            expect(not os.path.isdir(hooks) or not [h for h in os.listdir(hooks) if not h.endswith('.sample')], 'K  the extracted %s carries no hook (code that runs when a commit is made)' % kind)
            expect(not os.path.exists(os.path.join(x, '.git', 'logs')), 'K  the extracted %s carries no reflog' % kind)
            if kind == 'tar':                                                                                 # (Python's zip extraction drops the executable bit, which git then reports)
                st = git(x, 'status', '--porcelain', check=False).stdout
                expect(st == '', 'K  git status of the extracted tar is clean (MANIFEST.sha256 and RELEASE.txt are not shown): %r' % st[:200])
            r = git(x, '-c', 'user.name=New Person', '-c', 'user.email=new@example.org', 'commit', '-q', '--allow-empty', '-m', 'a commit in the extracted directory', check=False)
            expect(r.returncode == 0 and git(x, 'log', '-1', '--format=%an', check=False).stdout.strip() == 'New Person', 'K  a commit can be made in the extracted %s, under the identity of whoever makes it: %s' % (kind, r.stderr.strip()[:200]))
        rel = open(os.path.join(xt, 'pkg', 'RELEASE.txt')).read()
        expect('history 2 commits' in rel and 'uncommitted' not in rel, 'K  RELEASE.txt of a clean tree names the history and no uncommitted changes (%r)' % rel[:200])
        put('README.md', 'one\ntwo\nthree\n')                                                        # an uncommitted change: the archive says so, and the extracted directory shows it
        tp2, _ = make('pkg2')
        os.makedirs(os.path.join(tmp, 'xt2')); tarfile.open(tp2).extractall(os.path.join(tmp, 'xt2'), filter='fully_trusted')
        x2 = os.path.join(tmp, 'xt2', 'pkg2')
        expect('uncommitted changes' in open(os.path.join(x2, 'RELEASE.txt')).read() and git(x2, 'status', '--porcelain', check=False).stdout.strip() == 'M README.md', 'K  an archive of a tree with uncommitted changes says so and `git status` shows them')
        expect(sorted(os.listdir(os.path.join(tmp, 'out'))) == ['pkg.tar.gz', 'pkg.zip', 'pkg2.tar.gz', 'pkg2.zip'], 'K  make_dist.py --git leaves nothing else in its output directory')
    finally: shutil.rmtree(tmp, ignore_errors=True)


def k_add():
    """make_dist.py --add: a build product that git does not hold (the PDF of the paper: ignored by .gitignore) goes into the archive at the path it has in the repository, or at another one, with its digest in
    MANIFEST.sha256 and its name in RELEASE.txt; without --add it is not there; a file that does not exist, a path outside the archive and a file that is in the archive already are refused; the extracted
    repository is clean (the file is ignored there as well)."""
    import tarfile, zipfile, hashlib
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0')
    def git(cwd, *args):
        return subprocess.run(['git', '-C', cwd] + list(args), capture_output=True, text=True, env=env)
    tmp = tempfile.mkdtemp(prefix='s45_distadd_')
    try:
        repo = os.path.join(tmp, 'repo'); os.makedirs(os.path.join(repo, 'tools')); os.makedirs(os.path.join(repo, 'paper'))
        shutil.copy(os.path.join(ROOT, 'tools', 'make_dist.py'), os.path.join(repo, 'tools', 'make_dist.py'))
        for rel, text in (('README.md', 'one\n'), ('.gitignore', '/paper/main.pdf\n'), ('paper/main.tex', 'x\n')):
            with open(os.path.join(repo, rel), 'w') as f: f.write(text)
        git(repo, 'init', '-q', '-b', 'main'); git(repo, 'config', 'user.name', 'T'); git(repo, 'config', 'user.email', 't@example.org'); git(repo, 'add', '-A'); git(repo, 'commit', '-q', '-m', 'one')
        pdf = b'%PDF-1.5 a build product\n\x00\x01'
        with open(os.path.join(repo, 'paper', 'main.pdf'), 'wb') as f: f.write(pdf)
        elsewhere = os.path.join(tmp, 'elsewhere.pdf')
        with open(elsewhere, 'wb') as f: f.write(b'%PDF other')
        def make(name, *extra):
            r = subprocess.run([sys.executable, os.path.join(repo, 'tools', 'make_dist.py'), '--git', '--out', os.path.join(tmp, 'out'), '--name', name] + list(extra), capture_output=True, text=True, env=env, timeout=300)
            return r, os.path.join(tmp, 'out', name + '.tar.gz'), os.path.join(tmp, 'out', name + '.zip')
        r, tp0, _ = make('plain')
        with tarfile.open(tp0) as tf: names0 = tf.getnames()
        expect(r.returncode == 0 and 'plain/paper/main.pdf' not in names0, 'K  without --add the ignored PDF is not in the archive')
        r, tp, zp = make('withpdf', '--add', 'paper/main.pdf', '--add', 'extra/other.pdf=' + elsewhere)
        expect(r.returncode == 0, 'K  make_dist.py --add fails: %s' % (r.stderr or r.stdout)[-300:])
        if r.returncode != 0: return
        x = os.path.join(tmp, 'x'); os.makedirs(x)
        with tarfile.open(tp) as tf: tf.extractall(x, filter='fully_trusted')
        d = os.path.join(x, 'withpdf')
        got = open(os.path.join(d, 'paper', 'main.pdf'), 'rb').read() if os.path.isfile(os.path.join(d, 'paper', 'main.pdf')) else None
        expect(got == pdf and open(os.path.join(d, 'extra', 'other.pdf'), 'rb').read() == b'%PDF other', 'K  --add puts the file in, byte for byte, at its path in the repository or at the path given (PATH=SOURCE)')
        man = dict((line.split('  ', 1)[1], line.split('  ', 1)[0]) for line in open(os.path.join(d, 'MANIFEST.sha256')).read().splitlines())
        expect(man.get('paper/main.pdf') == hashlib.sha256(pdf).hexdigest() and 'extra/other.pdf' in man, 'K  the added files are in MANIFEST.sha256 with their digests')
        rel = open(os.path.join(d, 'RELEASE.txt')).read()
        expect('added by hand, not in git: paper/main.pdf, extra/other.pdf' in rel, 'K  RELEASE.txt names the files that were added by hand (%r)' % rel[:300])
        with zipfile.ZipFile(zp) as zf: expect('withpdf/paper/main.pdf' in zf.namelist(), 'K  the zip holds the added file too')
        st = git(d, 'status', '--porcelain').stdout
        expect(st.strip() == '?? extra/' , 'K  the extracted repository shows only the file whose path .gitignore does not cover (%r)' % st[:200])
        for bad, why in (('paper/missing.pdf', 'a file that does not exist'), ('../outside.pdf=' + elsewhere, 'a path outside the archive'), ('README.md', 'a file that is in the archive already')):
            rb, _, _ = make('bad', '--add', bad)
            expect(rb.returncode != 0, 'K  --add refuses %s' % why)
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
