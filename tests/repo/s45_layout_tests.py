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
  M. the mutation driver's machinery: the digests that say which mutant and which tree a saved result is for, the results file (one JSON line per mutant, a cut line skipped), resumption (a mutant is run again when the
     tree, the mutant, the gates or the mode changed), the parallel run (results in the order asked for), the way a gate caught a mutant (assertion / crash / timeout).
  N. tools/check_numbers.py, which checks a LaTeX source typed by hand against the results folders (docs/PAPER_PLAN.md): the macros, tables and figures it cites exist, each folder is what its manifest says and is not
     FAILED or INCOMPLETE, the soundness and audit evidence is for the code that is here, folders agree about the facts they both count, no \\todo is left; exit status 0 / 1 / 2.  S45_ONLY=N runs only this section.
  O. mutants/check_tools.py and tool_catalogue.py: the faults injected into the tools that make the evidence (soundness folder, statement audit, number checker, identity of the code) are sound -- every one replaces text that
     occurs once, changes it and leaves a file that compiles, and names a gate section that exists -- and the runner does what it says (a red or empty baseline stops it, a fault no check notices is a survivor, a saved result
     is not run again); a change of a tool that moves the text a fault replaces fails this section until the catalogue is updated.
  P. the runner of those faults on doctored catalogues (O pins the real one): it says why a catalogue is stale, selects, stops on a red or empty baseline, reports a survivor, reuses only what was run for this fault and tree.
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
    # the counts the documents state about the repository are the counts of the repository: a number that was true when it was written goes stale with the next gate or mutant (1616 stood in five documents
    # after the catalogue had 1647)
    import importlib.util
    spec = importlib.util.spec_from_file_location('mutant_catalogue_s45', os.path.join(ROOT, 'mutants', 'catalogue.py')); cat = importlib.util.module_from_spec(spec); spec.loader.exec_module(cat)
    n_mut = len(cat.MUTANTS); n_gates = len({c[0] for c in list(REG.GATE_CMDS) + list(REG.BANK_GATE_CMDS) + list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS) + list(REG.SYNTH_GATE_CMDS)})
    def stated(path, pattern, first_lines=None):
        text = read(os.path.join(ROOT, path)); text = '\n'.join(text.split('\n')[:first_lines]) if first_lines else text
        return [int(re.sub(r'\D', '', x)) for x in re.findall(pattern, text)]
    for path, pattern, what, want, first in (('README.md', r'(\d[ \u00a0\u202f]?\d{3}) mutants', 'mutants', n_mut, None), ('README.md', r'(\d{2,3}) gates', 'gates', n_gates, None), ('mutants/README.md', r'(\d[ \u00a0\u202f]?\d{3}) mutants', 'mutants', n_mut, None),
                                             ('docs/ROADMAP.md', r'(\d[ \u00a0\u202f]?\d{3}) mutants defined', 'mutants', n_mut, 12), ('docs/ROADMAP.md', r'(\d{2,3}) gates \(', 'gates', n_gates, 12), ('docs/ACTIONS.md', r'(\d[ \u00a0\u202f]?\d{3}) mutants of the engine', 'mutants', n_mut, None), ('docs/EXPERIMENTS.md', r'The mutation catalogue \((\d[ \u00a0\u202f]?\d{3}) mutants', 'mutants', n_mut, None)):
        got = stated(path, pattern, first)
        expect(got and all(g == want for g in got), 'E  %s states %s %s but the repository has %d' % (path, what, got, want))


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
                  'tests/README.md', 'tests/run_gates.py', 'tests/gate_registry.py', 'mutants/README.md', 'mutants/mutate.py', 'mutants/catalogue.py', 'legacy/README.md', 'tools/sanitize.py', 'tools/make_dist.py', 'tools/check_env.py', 'tools/check_numbers.py',
                  'docs/FINDINGS.md', 'docs/ROADMAP.md', 'docs/BANK.md', 'docs/ACTIONS.md', 'docs/KERNEL_DESIGN.md', 'docs/ARCHITECTURE_RL.md', 'docs/TRAINING.md', 'docs/GENERATOR.md', 'docs/EXPERIMENTS.md', 'docs/PAPER_PLAN.md', 'docs/findings_tags.tsv', 'docs/statement_reviews.tsv',
                  'scripts/env.sh', 'scripts/local.env.example', 'scripts/00_check_env.sh', 'scripts/01_build_dataset.sh', 'scripts/02_generate_synthetic.sh', 'scripts/03_pretrain_finetune.sh', 'scripts/04_reinforcement_learning.sh',
                  'scripts/05_evaluate.sh', 'scripts/06_benchmarks.sh', 'scripts/07_report.sh', 'scripts/08_soundness.sh', 'scripts/09_audit.sh', 'scripts/run_all.sh', 'scripts/smoke_test.sh', 'scripts/status.sh',
                  'paper/main.tex', 'paper/main.bib', 'paper/Makefile', 'paper/tmlr.sty', 'paper/tmlr.bst', 'paper/generated/.gitkeep',
                  'data/dataset_v2/bank.json', 'data/dataset_v2/splits.json', 'data/dataset_v2/library.json', 'data/dataset_v2/manifest.json', 'data/dataset_v2/DATASET_CARD.md']
REQUIRED_DIRS = ['src/python/authoring', 'src/python/checker', 'src/python/library', 'src/python/environment', 'src/python/problems', 'src/python/learner', 'src/python/synth', 'src/python/experiments', 'scripts', 'paper',
                 'tests/kernel', 'tests/library', 'tests/environment', 'tests/problems', 'tests/learner', 'tests/synth', 'tests/repo', 'tests/support', 'tests/golden', 'examples', 'data/results']
TOP_LEVEL = {'README.md', 'Makefile', 'requirements.txt', '.gitignore', '.git', 'build', 'dist', 'artifacts', 'runs', 'results', 'src', 'tests', 'mutants', 'examples', 'docs', 'data', 'legacy', 'tools', 'scripts', 'paper', '.venv',
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
    wanted = {'help', 'kernel', 'gates', 'gates-ml', 'gates-repo', 'gates-synth', 'dataset-verify', 'examples', 'mutants-check', 'tool-mutants', 'dist', 'clean', 'check-env', 'smoke', 'experiments', 'status', 'report', 'soundness', 'audit', 'paper'}
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


def sec_M():
    """the machinery of a long mutation run, tested without running a gate: the gates and the copies of the tree are replaced by stand-ins, everything else is the driver's own code"""
    import importlib, io, contextlib, threading, time as _t
    sys.path.insert(0, os.path.join(ROOT, 'mutants')); MU = importlib.import_module('mutate')
    names = [n for n in MU.MUTANTS if n.startswith(('GA36 ', 'GA37 ', 'GE108 '))]; expect(len(names) == 3, 'M  the three mutants the test uses are in the catalogue (%s)' % names)
    expect(MU.series_of('GE108 menu: x') == 'GE' and MU.series_of('M113 irrelevance') == 'M' and MU.series_of('FU3 x') == 'FU' and MU.series_of('123') == '', 'M  the series of a mutant is the letters its name starts with')
    d1, d2 = MU.mutant_digest(names[0]), MU.mutant_digest(names[1]); saved = MU.MUTANTS[names[0]]
    expect(re.fullmatch(r'[0-9a-f]{16}', d1) and d1 != d2 and d1 == MU.mutant_digest(names[0]), 'M  the digest of a mutant is 16 hex digits, differs between mutants and is stable')
    try:
        MU.MUTANTS[names[0]] = (saved[0], saved[1] + ' ', *saved[2:]); d3 = MU.mutant_digest(names[0])
    finally: MU.MUTANTS[names[0]] = saved
    expect(d3 != d1 and MU.mutant_digest(names[0]) == d1, 'M  the digest of a mutant changes with the text it puts in')
    # the digest of the tree: of the files below src, tests and mutants (not the caches), and of nothing else
    tmp = tempfile.mkdtemp(prefix='s45m_'); real_root = MU.ROOT
    try:
        for top in ('src', 'tests', 'mutants'): os.makedirs(os.path.join(tmp, top, '__pycache__'))
        for rel_, txt in (('src/a.py', 'a'), ('tests/t.py', 't'), ('mutants/m.py', 'm')):
            with open(os.path.join(tmp, rel_), 'w') as f: f.write(txt)
        MU.ROOT = tmp; t0 = MU.tree_digest()
        with open(os.path.join(tmp, 'src', '__pycache__', 'a.pyc'), 'w') as f: f.write('x')
        with open(os.path.join(tmp, 'tests', '__pycache__', 'odd.txt'), 'w') as f: f.write('x')                 # (nothing below a cache directory counts, whatever its name)
        with open(os.path.join(tmp, 'src', 'b.so'), 'w') as f: f.write('x')
        with open(os.path.join(tmp, 'notes.txt'), 'w') as f: f.write('x')
        for rel_ in ('docs/d.md', 'paper/p.tex', 'tools/t.py', 'scripts/s.sh', 'README.md', 'data/dataset_v2/other.json', 'src/NOTES.md', 'tests/README.md', 'mutants/README.md'):    # (a document, the paper, a tool, a script, a .md file below a code directory: not the code)
            os.makedirs(os.path.dirname(os.path.join(tmp, rel_)), exist_ok=True)
            with open(os.path.join(tmp, rel_), 'w') as f: f.write('x')
        t1 = MU.tree_digest()
        with open(os.path.join(tmp, 'tests', 't.py'), 'w') as f: f.write('T')
        t2 = MU.tree_digest()
        with open(os.path.join(tmp, 'tests', 't.py'), 'w') as f: f.write('t')
        os.rename(os.path.join(tmp, 'src', 'a.py'), os.path.join(tmp, 'src', 'c.py')); t3 = MU.tree_digest()
        os.rename(os.path.join(tmp, 'src', 'c.py'), os.path.join(tmp, 'src', 'a.py'))
        with open(os.path.join(tmp, 'docs', 'd.md'), 'w') as f: f.write('edited')
        with open(os.path.join(tmp, 'paper', 'p.tex'), 'w') as f: f.write('edited')
        with open(os.path.join(tmp, 'tests', 'README.md'), 'w') as f: f.write('edited')
        with open(os.path.join(tmp, 'src', 'NOTES.md'), 'w') as f: f.write('edited')
        t6 = MU.tree_digest()
        with open(os.path.join(tmp, 'data', 'dataset_v2', 'manifest.json'), 'w') as f: f.write('{}')
        t4 = MU.tree_digest()
        with open(os.path.join(tmp, 'data', 'dataset_v2', 'manifest.json'), 'w') as f: f.write('{"a": 1}')
        t5 = MU.tree_digest()
    finally: MU.ROOT = real_root; shutil.rmtree(tmp, ignore_errors=True)
    expect(re.fullmatch(r'[0-9a-f]{16}', t0) and t0 == t1, 'M  the digest of the tree ignores caches, built files and what is outside src, tests and mutants (%s, %s)' % (t0, t1))
    expect(t2 != t0 and t3 != t0 and t2 != t3, 'M  the digest of the tree changes with the content of a file and with the name of a file')
    expect(t6 == t0 and t4 != t0 and t5 != t4, 'M  editing a document, the paper, or a .md file below src, tests or mutants does not change the digest of the tree (nothing a run or a gate does depends on it); the manifest of the dataset is part of the code (%s %s %s %s)' % (t0, t6, t4, t5))
    expect(itp_paths.CODE_PATHS == ('src', 'tests', 'mutants', 'data/dataset_v2/manifest.json') and itp_paths.CODE_DIRS == ('src', 'tests', 'mutants') and itp_paths.DOC_SUFFIXES == ('.md',) and itp_paths.CODE_PATHSPEC == itp_paths.CODE_PATHS + (':(exclude)*.md',), 'M  the code is defined once, in itp_paths (the digest, exp_repo and ml_meta read it from there)')
    expect(MU.catch_kind('Traceback: HANG (timeout) -- the mutant does not terminate') == 'timeout' and MU.catch_kind('Traceback (most recent call last):\nValueError: x') == 'crash'
           and MU.catch_kind('S40 ACTIONS GATE: FAIL') == 'assertion' and MU.catch_kind('Traceback (most recent call last):\n...\nS40 ACTIONS GATE: FAIL') == 'assertion', 'M  a gate catches a mutant by a failed check, by a crash or by not stopping, and the record says which')
    # the results file: the last record of a mutant counts and a line that was cut short is skipped
    f = os.path.join(tempfile.mkdtemp(prefix='s45m_'), 'res.jsonl')
    try:
        with open(f, 'w') as fh: fh.write(json.dumps({'name': 'A', 'status': 'gap'}) + '\n' + '{"name": "B", "sta' + '\n' + json.dumps({'name': 'A', 'status': 'caught'}) + '\n' + '[1, 2]\n' + json.dumps({'name': 'C'}) + '\n')
        got = MU.read_results(f)
        expect(sorted(got) == ['A', 'C'] and got['A']['status'] == 'caught', 'M  a results file: the last record of a mutant, a line cut short and a line that is no record are skipped (%s)' % sorted(got))
        expect(MU.read_results(os.path.join(os.path.dirname(f), 'nope.jsonl')) == {} and MU.read_results(None) == {}, 'M  no results file: no results')
        # a run with stand-ins for the baseline and for a mutant
        calls = []; lock = threading.Lock()
        def fake_run_mutant(nm, cmds, base_time, slow, builds):
            with lock: calls.append(nm)
            _t.sleep(0.05 * (3 - names.index(nm)))                                                  # (the first mutant finishes last)
            kind = 'equivalent' if nm == names[2] else 'caught'
            return {'name': nm, 'id': nm.split()[0], 'series': MU.series_of(nm), 'file': 'x.py', 'digest': MU.mutant_digest(nm), 'caught_by': [{'gate': 'g.py', 'kind': 'assertion', 'first': 'f'}] if kind == 'caught' else [],
                    'gates_run': ['g.py'], 'status': kind, 'seconds': 0.1}
        orig = (MU.run_mutant, MU.baseline, MU.tree_digest)
        MU.run_mutant = fake_run_mutant; MU.baseline = lambda cmds: {c[0]: 1.0 for c in cmds}; MU.tree_digest = lambda: 'a' * 16
        old_only = os.environ.get('MUT_ONLY'); old_full = os.environ.get('MUT_FULL'); os.environ['MUT_ONLY'] = 's16_quot_tests.py'; os.environ.pop('MUT_FULL', None)
        def go(**kw):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): res = MU.run(names, **kw)
            return res, buf.getvalue()
        try:
            os.remove(f)
            res, out = go(jobs=3, json_path=f)
            lines = [json.loads(l) for l in open(f)]
            expect([r['name'] for r in res] == names and len(calls) == 3 and sorted(r['name'] for r in lines) == sorted(names), 'M  a parallel run gives its results in the order asked for and writes a line per mutant (%s)' % [r['name'][:5] for r in res])
            expect(all(r['tree'] == 'a' * 16 and r['gates'] == ['s16_quot_tests.py'] and r['full'] is False and r['jobs'] == 3 and 'time' in r for r in lines), 'M  every line says which tree, which gates, which mode and how many jobs it was measured with')
            res2, out2 = go(jobs=1, json_path=f)
            expect(len(calls) == 3 and out2.count('(saved)') == 3 and [r['status'] for r in res2] == ['caught', 'caught', 'equivalent'] and 'baseline green' not in out2, 'M  a second run finds all three in the file and runs nothing (not even the baseline)')
            with open(f, 'a') as fh: fh.write('{"name": "cut')                                                       # (an interrupted write)
            res3, out3 = go(jobs=1, json_path=f)
            expect(len(calls) == 3 and out3.count('(saved)') == 3, 'M  a file whose last line was cut short is still read')
            MU.tree_digest = lambda: 'b' * 16; res4, out4 = go(jobs=2, json_path=f)
            expect(len(calls) == 6 and '(saved)' not in out4 and 'baseline green' in out4, 'M  another tree: every mutant is run again (a result is for the code it was measured on)')
            MU.tree_digest = lambda: 'a' * 16; calls[:] = []
            os.environ['MUT_ONLY'] = 's16_quot_tests.py,s17_irrelevance_tests.py'; res5, out5 = go(jobs=1, json_path=f)
            expect(len(calls) == 3, 'M  other gates: every mutant is run again')
            os.environ['MUT_ONLY'] = 's16_quot_tests.py'; os.environ['MUT_FULL'] = '1'; res6, out6 = go(jobs=1, json_path=f)
            expect(len(calls) == 6 and all(r['full'] for r in res6), 'M  another mode (MUT_FULL): every mutant is run again')
            os.environ.pop('MUT_FULL'); calls[:] = []
            res7, out7 = go(jobs=1, json_path=f, redo=True)
            expect(len(calls) == 3 and '(saved)' not in out7, 'M  --redo runs them all again')
            calls[:] = []; MU.MUTANTS[names[1]] = (MU.MUTANTS[names[1]][0], MU.MUTANTS[names[1]][1] + ' ', *MU.MUTANTS[names[1]][2:])
            try: res8, out8 = go(jobs=1, json_path=f)
            finally: MU.MUTANTS[names[1]] = (MU.MUTANTS[names[1]][0], MU.MUTANTS[names[1]][1][:-1], *MU.MUTANTS[names[1]][2:])
            expect(calls == [names[1]], 'M  a mutant whose text changed is run again and the others are not (%s)' % [c[:5] for c in calls])
            calls[:] = []; res9, out9 = go(jobs=1)
            expect(len(calls) == 3, 'M  no results file: everything is run')
        finally:
            MU.run_mutant, MU.baseline, MU.tree_digest = orig
            for k, v in (('MUT_ONLY', old_only), ('MUT_FULL', old_full)):
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
    finally: shutil.rmtree(os.path.dirname(f), ignore_errors=True)
    expect(MU.usable_cores() >= 1, 'M  the cores the driver may use')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'mutants', 'mutate.py'), '--bogus'], capture_output=True, text=True, timeout=120)
    expect(r.returncode == 2 and 'unknown option' in r.stdout, 'M  an option the driver does not know is refused (%s)' % (r.stdout + r.stderr)[-100:])


def sec_N():
    """tools/check_numbers.py, which checks a LaTeX source typed by hand against the results folders (docs/PAPER_PLAN.md): the number macros, tables and figures it cites exist, each folder is what its manifest says and
    is not FAILED or INCOMPLETE, the evidence is for the code that is here, the folders agree about the facts they both count, no \todo is left.  Tested on small folders made with the report's own writers (numbers, manifest)."""
    import importlib, io, contextlib, collections as _c
    sys.path.insert(0, os.path.join(ROOT, 'tools')); CN = importlib.import_module('check_numbers')
    import exp_report as ER, exp_bundle as EB
    tmp = tempfile.mkdtemp(prefix='s45n_'); TREE = 'a' * 16
    def write(path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f: f.write(text)
        return path
    def mk(name, macros, kind='report', status=None, tree=TREE, dirty=False, tables=(), figures=(), md_warnings=None, run_trees=None):
        d = os.path.join(tmp, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        ER.write_numbers(_c.OrderedDict(macros), d)
        for t in tables: write(os.path.join(d, 'tables', t + '.tex'), '\\begin{tabular}{l}x\\end{tabular}\n')
        for fg in figures: write(os.path.join(d, 'figures', fg), 'figure')
        if kind != 'report': write(os.path.join(d, kind + '.json'), json.dumps({'status': status, 'stamp': {'tree': tree, 'commit': 'c' * 40, 'dirty': dirty}}))
        else:
            write(os.path.join(d, 'report.md'), '\n'.join(['# Experiment report', '', '## What was run', '', '* x', '', '## Warnings', ''] + (['* ' + w for w in md_warnings] if md_warnings else ['None: every evaluation row is present and no solver raised.']) + ['', '## The artefact', '', '* not a warning']) + '\n')
            write(os.path.join(d, 'provenance', 'PROFILE.json'), json.dumps({'name': 'full'}))
            rts = (tree,) if run_trees is None else run_trees                          # (the code each run recorded: by default the folder's own; () = no runs.csv, as in a folder made before the digest was recorded)
            if rts: write(os.path.join(d, 'provenance', 'runs.csv'), 'id,commit,tree,tree_finish\n' + ''.join('r%d,%s,%s,%s\n' % (i, 'c' * 40, t or '', t or '') for i, t in enumerate(rts)))
        EB.manifest(d); return d
    def has(res, sev, *subs): return any(p[0] == sev and all(x in p[1] + ' ' + p[2] for x in subs) for p in res['problems'])
    def count(res, sev): return sum(1 for p in res['problems'] if p[0] == sev)
    ERR = []
    def cli(*args):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err): code = CN.main(list(args))
        ERR[:] = [err.getvalue()]
        return code, buf.getvalue()
    try:
        # the pieces: macros as the report writes them, comments, the macros a source defines itself, the numbers typed in a line
        d0 = mk('d0', {'test_bc': '41.0 ± 2.0%', 'bank_problems': 268, 'abl_pretrain_p': 'p = 0.03_x'}, tables=['main'])
        got = CN.read_macros(d0)
        expect(list(got) == ['numTestBc', 'numBankProblems', 'numAblPretrainP'] and got['numTestBc'] == r'41.0 $\pm$ 2.0\%' and got['numAblPretrainP'] == r'p = 0.03\_x' and got['numBankProblems'] == '268', 'N  numbers.tex is read back as the report wrote it, in order, escapes kept (%s)' % got)
        with open(os.path.join(d0, 'numbers.tex'), 'a') as f_: f_.write('\\newcommand{\\other}{1}\n\\newcommand{\\numlower}{2}\n% \\newcommand{\\numCommented}{3}\n\\newcommand{\\numBankProblems}{268} trailing\n')
        expect(list(CN.read_macros(d0)) == ['numTestBc', 'numBankProblems', 'numAblPretrainP'], 'N  only the lines that define a \\numName are macros: not another command, not a lower-case name, not a comment, not a line with something after it (%s)' % list(CN.read_macros(d0)))
        expect(CN.read_macros(os.path.join(tmp, 'nowhere')) is None, 'N  no numbers.tex: None')
        expect(CN.strip_comment(r'50\% of it \numKept % \numHidden') == r'50\% of it \numKept ' and CN.strip_comment('% all of it') == '' and CN.strip_comment('no comment') == 'no comment', 'N  a comment is a % that is not escaped')
        expect(CN.plain_number('1647') == '1647' and CN.plain_number('99.9') == '99.9' and CN.plain_number('12') is None and CN.plain_number('5') is None and CN.plain_number('0.5') is None and CN.plain_number('yes') is None
               and CN.plain_number('41.0 $\\pm$ 2.0') is None and CN.plain_number('') is None and CN.plain_number('0.25') == '0.25', 'N  only a plain number of three digits, or a decimal with two, is worth looking for in the text')
        line_ = r'1\,647 and 1{,}647 and 1,647 and 1647. Also 12,5 and 12,3456 and 3.14 and pollack1998believe and 50\% and \ref{x1000} and the 7th and 5seeds and sec3 and fig2.'
        expect(CN.typed_numbers(line_) == ['1647', '1647', '1647', '1647', '12', '5', '12', '3456', '3.14', '50'],
               'N  typed numbers: thousands separators removed (only before exactly three digits), digits inside a word or a command, or followed by letters, are not numbers (%s)' % CN.typed_numbers(line_))
        text = '\n'.join([r'% \numHidden \N{Hidden} \inputtable{hidden} \todo{hidden}', r'\newcommand{\numOwn}{3} \renewcommand{\numOwnRe}{1} \providecommand*{\numOwnToo}{4} \def\numOwnThree{5}',
                   r'Text \N{BankProblems}, \numBankProved and \numberwithin{equation}{section}, 50\% \numKept. % \numCommented', r'\inputtable{main} \input{generated/tables/bank.tex} \input{generated/numbers.tex} \input{sections/intro}',
                   r'\genfigure{fig_budget}{0.9}{cap} \includegraphics[width=2cm]{generated/figures/fig_levels.pdf} \includegraphics{mine/own.png}', r'\todo{x}', r'\todo {y} and \todo{z}',
                   r'\newcommand{\inputtable}[1]{\input{generated/tables/#1.tex}} \genfigure{#1} \newcommand{\tab}[1]{\inputtable{#1}}'])
        macros, tables, figures, own, todos = CN.citations([CN.strip_comment(l_) for l_ in text.split('\n')])
        expect(sorted(macros) == ['numBankProblems', 'numBankProved', 'numKept', 'numOwn', 'numOwnRe', 'numOwnThree', 'numOwnToo'] and {'numOwn', 'numOwnRe', 'numOwnToo', 'numOwnThree'} <= own and 'numCommented' not in macros and 'numHidden' not in macros,
               'N  the macros a source cites: \\N{X} and \\numX, not in comments, not \\numberwithin, and the ones it defines itself are its own (%s, own %s)' % (sorted(macros), sorted(own)))
        expect(sorted(tables) == ['bank', 'main'] and sorted(figures) == ['fig_budget', 'fig_levels'] and todos == [6, 7], 'N  tables (\\inputtable, \\input of a tables/ file), figures (\\genfigure, \\includegraphics of a figures/ file), \\todo lines; the definitions with #1 and other inputs are not citations (%s %s %s)' % (sorted(tables), sorted(figures), todos))
        # a source that cites only what exists, three folders that are all what they say
        R = mk('rep', {'bank_problems': 268, 'bank_proved': 187, 'repo_kernel_lines': 7474, 'test_bc': '41.0 ± 2.0%', 'n_seeds': 5}, tables=['main', 'bank'], figures=['fig_budget.pdf', 'fig_levels.png'])
        S = mk('snd', {'sound_mutants': 1647, 'sound_kernel_lines': 7474, 'sound_mutants_caught': 1625}, kind='soundness', status='CLEAN', tables=['sound_gates'])
        A = mk('aud', {'audit_problems': 268, 'audit_proved': 187}, kind='audit', status='CLEAN', tables=['audit_summary'])
        good = write(os.path.join(tmp, 'good.tex'), '\n'.join([r'\documentclass{article}', r'\begin{document}', r'We have \N{BankProblems} problems, \numBankProved with proofs, \N{SoundMutants} mutants (\numSoundMutantsCaught caught),',
                                                             r'\N{AuditProblems} statements, and \N{TestBc} on test.', r'\inputtable{main} \input{generated/tables/sound_gates} \inputtable{audit_summary}',
                                                             r'\genfigure{fig_budget}{0.9}{caption} \includegraphics[width=3cm]{generated/figures/fig_levels.pdf}', r'\end{document}']) + '\n')
        res = CN.analyse([good], [R, S, A], tree=TREE)
        expect(res['problems'] == [] and res['uses'] == (6, 3, 2) and len(res['macros']) == 10, 'N  three clean folders and a source that cites what exists: nothing to report (%s, %s)' % (res['problems'], res['uses']))
        code, out = cli(good, '--results', R, '--results', S, '--results', A, '--no-tree')
        expect(code == 0 and '0 errors, 0 warnings' in out and 'soundness, CLEAN' in out and 'manifest OK' in out and '6 macro uses, 3 tables, 2 figures' in out, 'N  the command line says so and exits 0 (%s)' % out[-200:])
        tiny = write(os.path.join(tmp, 'tiny.tex'), r'\N{BankProblems}' + '\n')
        # an undefined macro (with a suggestion), a missing table, a missing figure
        bad = write(os.path.join(tmp, 'bad.tex'), '\n'.join([r'\N{BankProblem} and \numAuditNonsense and \N{TestBc}.', r'\inputtable{nope} \input{generated/tables/main} \genfigure{fig_nope}{1}{c} \includegraphics{x/figures/fig_gone.pdf}']) + '\n')
        res = CN.analyse([bad], [R, S, A], tree=TREE)
        expect(count(res, 'ERROR') == 5 and has(res, 'ERROR', 'numBankProblem ', 'did you mean', 'numBankProblems') and has(res, 'ERROR', 'numAuditNonsense') and has(res, 'ERROR', "'nope'") and has(res, 'ERROR', "'fig_nope'") and has(res, 'ERROR', "'fig_gone'")
               and res['problems'][0][1].endswith('bad.tex:1'), 'N  an undefined macro (with the near ones), a table and a figure that no folder holds are errors, each with its line (%s)' % [p[1][-20:] + ' ' + p[2][:50] for p in res['problems']])
        code, out = cli(bad, '--results', R, '--results', S, '--results', A, '--no-tree')
        expect(code == 1 and '5 errors' in out, 'N  errors: exit status 1')
        local = write(os.path.join(tmp, 'local.tex'), r'\newcommand{\numLocal}{3} \numLocal and \N{BankProblems}' + '\n')
        res = CN.analyse([local], [R], tree=TREE)
        expect(res['problems'] == [] and res['uses'] == (3, 0, 0), 'N  a macro the source defines itself is not looked for in the folders (%s)' % res['problems'])
        # what a folder holds is found in any of them
        only_snd = write(os.path.join(tmp, 'o.tex'), r'\inputtable{sound_gates} \inputtable{main}' + '\n')
        res = CN.analyse([only_snd], [R, S], tree=TREE); res2 = CN.analyse([only_snd], [R], tree=TREE)
        expect(res['problems'] == [] and count(res2, 'ERROR') == 1 and has(res2, 'ERROR', "'sound_gates'"), 'N  a table is found in whichever folder holds it')
        # two folders that define a macro differently; the same fact counted twice and counted differently
        C1 = mk('c1', {'bank_problems': 268}, tables=[]); C2 = mk('c2', {'bank_problems': 269}); C3 = mk('c3', {'bank_problems': 268})
        res = CN.analyse([good], [C1, C2], tree=TREE); res3 = CN.analyse([good], [C1, C3], tree=TREE)
        expect(has(res, 'ERROR', 'numBankProblems is 269 here and 268 in') and not [p for p in res3['problems'] if p[2].startswith('\\numBankProblems is')], 'N  a macro defined differently in two folders is an error; defined alike it is not')
        S2 = mk('snd2', {'sound_kernel_lines': 7000, 'sound_mutants': 1647}, kind='soundness', status='CLEAN')
        res = CN.analyse([good], [R, S2], tree=TREE)
        expect(has(res, 'WARNING', 'numRepoKernelLines is 7474', 'numSoundKernelLines is 7000', 'counted twice') and count(res, 'WARNING') == 1, 'N  the same fact counted twice and counted differently (the kernel\'s lines) is a warning (%s)' % res['problems'])
        # the manifest: a changed file, an added file, a removed file, no manifest
        for what in ('changed', 'added', 'removed', 'none'):
            d = mk('man_' + what, {'bank_problems': 268}, tables=['main', 'bank'])
            if what == 'changed': write(os.path.join(d, 'tables', 'main.tex'), 'edited\n')
            elif what == 'added': write(os.path.join(d, 'tables', 'new.tex'), 'new\n')
            elif what == 'removed': os.remove(os.path.join(d, 'tables', 'bank.tex'))
            else: os.remove(os.path.join(d, 'MANIFEST.sha256'))
            res = CN.analyse([good], [d], tree=TREE)
            want = {'changed': 'changed: tables/main.tex', 'added': 'not in the manifest: tables/new.tex', 'removed': 'missing: tables/bank.tex', 'none': 'no MANIFEST.sha256'}[what]
            expect(has(res, 'ERROR', want, 'edited after it was made'), 'N  a folder that is not what its manifest says is an error (%s: %s)' % (what, [p[2][:80] for p in res['problems'] if 'manifest' in p[2]]))
        # the status of the soundness and audit folders; the tree they were made for; a dirty tree
        for status, sev, word in (('FAILED', 'ERROR', 'is FAILED'), ('INCOMPLETE', 'WARNING', 'is INCOMPLETE'), (None, 'WARNING', "is None, not CLEAN"), ('PARTIAL', 'WARNING', "is 'PARTIAL', not CLEAN"), ('CLEAN', None, '')):
            d = mk('st', {'sound_mutants': 1647}, kind='soundness', status=status)
            res = CN.analyse([tiny], [R, d], tree=TREE); hit = [p for p in res['problems'] if 'status' in p[2]]
            expect((sev is None and not hit) or (sev is not None and len(hit) == 1 and hit[0][0] == sev and word in hit[0][2]), 'N  a %s soundness folder is %s (%s)' % (status, sev, hit))
        expect(has(CN.analyse([tiny], [R, mk('st', {'sound_mutants': 1}, kind='soundness', status='FAILED')], tree=TREE), 'ERROR', 'FAILED', 'contradicts') and has(CN.analyse([tiny], [R, mk('st', {'sound_mutants': 1}, kind='audit', status='INCOMPLETE')], tree=TREE), 'WARNING', 'audit folder is INCOMPLETE'),
               'N  FAILED means the evidence contradicts a claim (an error); INCOMPLETE that it is partial (a warning); both for the audit as for the soundness folder')
        d_old = mk('old', {'sound_mutants': 1647}, kind='soundness', status='CLEAN', tree='b' * 16); d_new = mk('new', {'audit_problems': 268}, kind='audit', status='CLEAN', tree=TREE)
        res = CN.analyse([tiny], [R, d_old, d_new], tree=TREE)
        expect(has(res, 'WARNING', 'made for the tree ' + 'b' * 16, 'the code here is ' + TREE) and has(res, 'WARNING', '(folders)', 'different trees') and count(res, 'WARNING') == 2, 'N  evidence made for other code, and two folders made for different trees, are warnings (%s)' % [p[2][:60] for p in res['problems']])
        expect(CN.analyse([tiny], [R, d_old], tree=TREE, check_tree=False)['problems'] == [] and CN.analyse([tiny], [R, d_new], tree=TREE)['problems'] == [], 'N  no tree check on request; the same tree is silent')
        d_nt = mk('nt', {'sound_mutants': 1647}, kind='soundness', status='CLEAN', tree=None); d_dirty = mk('dirty', {'sound_mutants': 1647}, kind='soundness', status='CLEAN', dirty=True)
        expect(has(CN.analyse([tiny], [R, d_nt], tree=TREE), 'WARNING', 'does not record the tree') and has(CN.analyse([tiny], [R, d_dirty], tree=TREE), 'WARNING', 'uncommitted changes'), 'N  a folder that does not say which tree it is for, and one made from uncommitted changes, are warnings')
        # the code the runs of a report folder ran with (provenance/runs.csv)
        dr = mk('rt', {'bank_problems': 268}, run_trees=('b' * 16, TREE, TREE)); dr_b = mk('rt_b', {'bank_problems': 268}, run_trees=('b' * 16,))
        expect(CN.run_trees(dr) == [TREE, 'b' * 16] and CN.run_trees(R) == [TREE] and CN.run_trees(mk('rt0', {'bank_problems': 1}, run_trees=())) == [] and CN.run_trees(mk('rt1', {'bank_problems': 1}, run_trees=('',))) == [] and CN.run_trees(os.path.join(tmp, 'nowhere')) == [],
               'N  the code the runs recorded: the distinct digests, sorted; none for a folder without runs.csv, for runs that recorded none and for a folder that does not exist (%s %s)' % (CN.run_trees(dr), CN.run_trees(R)))
        res = CN.analyse([tiny], [dr_b], tree=TREE)
        expect(count(res, 'WARNING') == 1 and has(res, 'WARNING', dr_b, 'were made with the code ' + 'b' * 16 + ';', 'the code here is ' + TREE, 'about other code'), 'N  a report whose runs ran with other code than the one here is a warning (%s)' % [p[2][:80] for p in res['problems']])
        res = CN.analyse([tiny], [dr], tree=TREE)
        expect(count(res, 'WARNING') == 2 and has(res, 'WARNING', dr, 'were made with the code ' + TREE + ', ' + 'b' * 16) and has(res, 'WARNING', '(folders)', 'different trees', TREE, 'b' * 16), 'N  runs of two versions of the code: the folder is a warning, and so is the mixture (%s)' % [p[2][:80] for p in res['problems']])
        res = CN.analyse([tiny], [S, dr_b], tree=TREE)
        expect(count(res, 'WARNING') == 2 and has(res, 'WARNING', '(folders)', 'different trees', TREE, 'b' * 16), 'N  a soundness folder for this code and a report whose runs ran with other code: the folders are about different code (%s)' % [p[2][:80] for p in res['problems']])
        res = CN.analyse([tiny], [R, S], tree=TREE)
        expect(res['problems'] == [], 'N  a report whose runs ran with the code that is here, and a soundness folder for it: nothing to say (%s)' % res['problems'])
        for rts in ((), ('',)):
            res = CN.analyse([tiny], [mk('rt_none', {'bank_problems': 268}, run_trees=rts)], tree=TREE)
            expect(count(res, 'WARNING') == 1 and has(res, 'WARNING', 'did not record the code they ran with', 'nothing says which code'), 'N  runs that recorded no code (no runs.csv, or an empty value) are a warning, not silence (%s: %s)' % (rts, [p[2][:60] for p in res['problems']]))
        expect(CN.analyse([tiny], [dr, dr_b, mk('rt_none', {'bank_problems': 268}, run_trees=())], tree=TREE, check_tree=False)['problems'] == [], 'N  no tree check on request: runs of other code, of two versions and of none are silent')
        expect(has(CN.analyse([tiny], [os.path.join(tmp, 'nowhere'), tmp], tree=TREE), 'ERROR', 'no numbers.tex') and not has(CN.analyse([tiny], [tmp], tree=TREE), 'WARNING', 'did not record the code'), 'N  a directory that is no results folder is an error and is not also asked which code it ran with')
        res = CN.analyse([tiny], [R])                                                      # (no digest given: the code that is here is hashed -- the runs of the fixture ran with other code)
        expect(count(res, 'WARNING') == 1 and has(res, 'WARNING', R, 'were made with the code ' + TREE, 'the code here is', 'about other code') and not has(res, 'WARNING', 'could not be computed'), 'N  without a digest given, the code that is here is hashed and compared with the runs (%s)' % [p[2][:80] for p in res['problems']])
        real_td = CN.tree_digest
        def no_digest(): raise RuntimeError('cannot hash')
        CN.tree_digest = no_digest
        try: res = CN.analyse([tiny], [R, d_new])
        finally: CN.tree_digest = real_td
        expect(count(res, 'WARNING') == 1 and has(res, 'WARNING', '(code)', 'could not be computed', 'cannot hash', 'cannot be compared'), 'N  when the digest of the code cannot be computed, that is said and nothing is compared with it (%s)' % [p[2][:80] for p in res['problems']])
        # the warnings of a report are warnings of the check
        dw = mk('rw', {'bank_problems': 268}, md_warnings=['the runs were made with 2 different commits (a, b): the matrix mixes code versions', 'missing evaluation rows: 3'])
        res = CN.analyse([tiny], [dw], tree=TREE)
        expect(count(res, 'WARNING') == 2 and has(res, 'WARNING', 'the report says: the runs were made with 2 different commits') and not [p for p in CN.analyse([tiny], [R], tree=TREE)['problems'] if 'report says' in p[2]], 'N  the Warnings section of a report.md (and nothing after it) is read: each bullet is a warning (%s)' % [p[2][:50] for p in res['problems']])
        # \todo left, and --strict
        todo = write(os.path.join(tmp, 'todo.tex'), '\n'.join([r'We have \N{BankProblems}.', r'\todo{first}', 'fine', r'\todo{second} \todo{third}']) + '\n')
        res = CN.analyse([todo], [R], tree=TREE)
        expect(count(res, 'WARNING') == 1 and has(res, 'WARNING', '2 \\todo left', 'lines 2, 4'), 'N  \\todo left in the text is a warning that counts the lines (%s)' % res['problems'])
        code0, out0 = cli(todo, '--results', R, '--no-tree'); code2, out2 = cli(todo, '--results', R, '--no-tree', '--strict')
        expect(code0 == 0 and code2 == 2 and '0 errors, 1 warning\n' in out2, 'N  warnings exit 0, and 2 with --strict (%s %s)' % (code0, code2))
        code, out = cli(bad, '--results', R, '--no-tree', '--strict')
        expect(code == 1, 'N  an error exits 1 also with --strict')
        # a missing folder, a directory that is no results folder, no source
        res = CN.analyse([good], [os.path.join(tmp, 'nowhere'), tmp], tree=TREE)
        expect(has(res, 'ERROR', 'nowhere', 'does not exist') and has(res, 'ERROR', 'no numbers.tex'), 'N  a folder that does not exist, and a directory that is not a results folder, are errors')
        res = CN.analyse([os.path.join(tmp, 'nothing.tex')], [R], tree=TREE)
        expect(has(res, 'ERROR', 'nothing.tex', 'no such file') and has(res, 'ERROR', 'no LaTeX source') and has(CN.analyse([], [R], tree=TREE), 'ERROR', 'no LaTeX source'), 'N  a source that does not exist, and no source at all, are errors')
        code, out = cli('--results', R, '--no-tree')
        expect(code == 1 and 'give the LaTeX source' in ERR[0], 'N  no source on the command line: exit 1 and it says so')
        saved = CN.DEFAULT_FOLDERS; CN.DEFAULT_FOLDERS = ('no/such/folder',)
        try: code, out = cli(good, '--no-tree')
        finally: CN.DEFAULT_FOLDERS = saved
        expect(code == 1 and 'no results folder' in ERR[0] and out == '', 'N  no results folder at all: exit 1 and it says so')
        # a directory of sources: every .tex below it, and nothing else
        sd = os.path.join(tmp, 'src'); write(os.path.join(sd, 'a.tex'), r'\N{BankProblems}' + '\n'); write(os.path.join(sd, 'sub', 'b.tex'), r'\N{TestBc} \inputtable{main}' + '\n'); write(os.path.join(sd, 'c.txt'), r'\N{Missing}' + '\n')
        res = CN.analyse([sd], [R], tree=TREE)
        expect(res['problems'] == [] and res['uses'] == (2, 1, 0) and len(res['files']) == 2, 'N  a directory is read for its .tex files (%s)' % res['files'])
        mixed = write(os.path.join(tmp, 'mixed.tex'), '\n'.join([r'\N{SoundMutants} is 1647 and \N{Nonsense}.', r'\todo{x}']) + '\n')
        S_inc = mk('snd_inc', {'sound_mutants': 1647}, kind='soundness', status='INCOMPLETE')
        code, out = cli(mixed, '--results', R, '--results', S_inc, '--no-tree', '--literals', '--unused')
        expect(code == 1 and 0 < out.index('ERROR') < out.index('WARNING') < out.index('INFO') and out.rstrip().endswith('1 error, 2 warnings'), 'N  the report lists errors, then warnings, then information, and ends with the counts (%s)' % out[-300:])
        cwd = os.getcwd(); os.chdir(tmp)
        try:
            code_a, out_a = cli(good, '--results', 'rep', '--results', 'snd', '--results', 'aud', '--no-tree'); code_b, out_b = cli(good, '--results', 'tools', '--no-tree')
        finally: os.chdir(cwd)
        expect(code_a == 0 and code_b == 1 and os.path.join(ROOT, 'tools') in out_b, 'N  a relative folder is looked for where you are, and then in the repository (%s)' % out_b[:200])
        # numbers typed by hand that a macro holds (informational), macros that nothing cites (informational)
        lit = write(os.path.join(tmp, 'lit.tex'), '\n'.join([r'There are 1647 mutants, 1\,647 of them, 1{,}647, and 1,647. With 5 seeds and 12 problems.', r'\cite{pollack1998believe} % 1647 in a comment', r'\bibitem{x} 1647', r'\N{SoundMutants}']) + '\n')
        res = CN.analyse([lit], [R, S, A], tree=TREE, literals=True); hits = [p for p in res['problems'] if p[0] == 'INFO']
        expect(len(hits) == 4 and all('numSoundMutants' in h[2] and h[1].endswith('lit.tex:1') for h in hits) and count(res, 'ERROR') == 0 and count(res, 'WARNING') == 0, 'N  --literals: a number typed four ways is found four times on its line and the macro is named; a comment, a citation key and a bibitem are not (%s)' % [(h[1][-10:], h[2][:40]) for h in hits])
        res = CN.analyse([lit], [R, S, A], tree=TREE); expect([p for p in res['problems'] if p[0] == 'INFO'] == [], 'N  without --literals typed numbers are not looked for')
        code, out = cli(lit, '--results', R, '--results', S, '--no-tree', '--literals', '--strict')
        expect(code == 0 and 'INFO' in out, 'N  what is only information never changes the exit status')
        res = CN.analyse([good], [R, S, A], tree=TREE, unused=True); un = [p for p in res['problems'] if p[0] == 'INFO']
        expect(len(un) == 1 and '4 of 10 macros are cited nowhere: \\numRepoKernelLines, \\numNSeeds, \\numSoundKernelLines, \\numAuditProved' in un[0][2], 'N  --unused lists the macros no sentence cites (%s)' % un)
        # --list and --tree
        code, out = cli('--list', 'sound', '--results', R, '--results', S)
        expect(code == 0 and '\\numSoundMutants = 1647    [%s]' % S in out and 'numBankProblems' not in out and out.count('\n') == 3, 'N  --list prints the macros whose name matches, with value and folder (%s)' % out)
        code, out = cli('--list', '--results', R); expect(code == 0 and out.count('\n') == 5, 'N  --list without a pattern prints them all')
        code, out = cli('--list', 'zzzz', '--results', R); expect(code == 0 and out == '', 'N  --list: nothing matches, nothing printed')
        sys.path.insert(0, os.path.join(ROOT, 'mutants')); MU = importlib.import_module('mutate')
        d_a = MU.tree_digest(); code, out = cli('--tree'); d_b = MU.tree_digest()
        expect(code == 0 and out.strip() in (d_a, d_b) and re.fullmatch(r'[0-9a-f]{16}', out.strip()), 'N  --tree prints the digest of the code that the evidence is bound to (the driver\'s)')
        # the paper skeleton of this repository: its preamble defines \inputtable and \genfigure with #1, which are not citations
        def uncommented(line):
            for i_, ch in enumerate(line):
                if ch == '%' and (i_ == 0 or line[i_ - 1] != '\\'): return line[:i_]
            return line
        main_tex = '\n'.join(uncommented(l_) for l_ in read(os.path.join(ROOT, 'paper', 'main.tex')).split('\n')); res = CN.analyse([os.path.join(ROOT, 'paper', 'main.tex')], [R], tree=TREE)
        cited_t = set(re.findall(r'\\inputtable\{([^}#]+)\}', main_tex)); cited_f = set(re.findall(r'\\genfigure\{([^}#]+)\}', main_tex)); cited_m = set('num' + m for m in re.findall(r'\\N\{([A-Za-z0-9]+)\}', main_tex))
        miss_t = [p for p in res['problems'] if p[0] == 'ERROR' and 'the table' in p[2]]; miss_f = [p for p in res['problems'] if p[0] == 'ERROR' and 'the figure' in p[2]]; miss_m = [p for p in res['problems'] if p[0] == 'ERROR' and 'defined in no results folder' in p[2]]
        expect(res['uses'][1] == len(cited_t) and res['uses'][2] == len(cited_f) and len(miss_t) == len(cited_t - {'main', 'bank'}) and len(miss_f) == len(cited_f - {'fig_budget', 'fig_levels'}) and len(miss_m) == len(cited_m - set(R and CN.read_macros(R))),
               'N  the skeleton of the paper: %d tables, %d figures and %d macros are cited and found by an independent reading (%s)' % (len(cited_t), len(cited_f), len(cited_m), (res['uses'], len(miss_t), len(miss_f), len(miss_m))))
        expect(not [p for p in res['problems'] if '#' in p[2]], 'N  a definition with #1 is not a citation')
        # through the shell, as a user runs it
        env = dict(os.environ); cmd = [sys.executable, os.path.join(ROOT, 'tools', 'check_numbers.py')]
        r0 = subprocess.run(cmd + [good, '--results', R, '--results', S, '--results', A, '--no-tree'], capture_output=True, text=True, timeout=120, env=env)
        r1 = subprocess.run(cmd + [bad, '--results', R, '--no-tree'], capture_output=True, text=True, timeout=120, env=env)
        r2 = subprocess.run(cmd + [todo, '--results', R, '--no-tree', '--strict'], capture_output=True, text=True, timeout=120, env=env)
        expect((r0.returncode, r1.returncode, r2.returncode) == (0, 1, 2) and 'ERROR' in r1.stdout and 'WARNING' in r2.stdout, 'N  as a program: exit status 0, 1 and 2 (%s, %s, %s)' % (r0.returncode, r1.returncode, r2.returncode))
    finally: shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ O. the catalogue of faults of the tools
def sec_O():
    """mutants/tool_catalogue.py: the faults injected into the tools that make the evidence (the soundness folder, the statement audit, the number checker, the identity of the code, the runner of the faults) are sound -- each one
    replaces text that occurs once, changes it, leaves a file that compiles and names a gate section that exists.  An edit of a tool that moves the text a fault replaces fails this section until the catalogue is updated.
    (The section that tests the runner of the faults is P, which never looks at the real catalogue: a fault injected into the runner makes THIS section fail, for every fault, and so proves nothing.)"""
    import importlib
    sys.path.insert(0, os.path.join(ROOT, 'mutants')); CT = importlib.import_module('check_tools'); TC = importlib.import_module('tool_catalogue')
    probs = CT.check()
    expect(probs == [], 'O  every fault of the catalogue replaces text that occurs once, changes it and leaves a file that compiles: %s' % probs[:4])
    letters = {rel: set(re.findall(r"\('([A-Z])', sec_[A-Za-z]+", read(os.path.join(ROOT, rel)))) for rel in ('tests/learner/s51_suite_tests.py', 'tests/repo/s45_layout_tests.py')}
    var_of = {'tests/learner/s51_suite_tests.py': 'S51_ONLY', 'tests/repo/s45_layout_tests.py': 'S45_ONLY'}; bad = []
    for t, spec in TC.TOOLS.items():
        ls = set(spec['env'].get(var_of.get(spec['gate'], ''), '').split(','))
        if spec['gate'] not in var_of or ls == {''} or not ls <= letters[spec['gate']]: bad.append(t)
    expect(not bad, 'O  the gate section of every tool exists (%s)' % bad)
    expect(list(TC.TOOLS) == ['soundness', 'numbers', 'audit', 'identity', 'digest', 'runner'] and all(len(s_['mutants']) >= 5 for s_ in TC.TOOLS.values()), 'O  the tools of the catalogue (%s)' % {t: len(s_['mutants']) for t, s_ in TC.TOOLS.items()})
    expect(TC.TOOLS['runner']['env'] == {'S45_ONLY': 'P'}, 'O  the faults of the runner are checked by the section that tests the runner (P), not by this one')
    tmp = tempfile.mkdtemp(prefix='s45o_')
    try:
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'mutants', 'check_tools.py'), '--check'], capture_output=True, text=True, timeout=120, cwd=tmp)
        expect(r.returncode == 0 and 'every one replaces text that occurs once' in r.stdout, 'O  as a program, from any directory: --check exits 0 (%s)' % (r.stdout + r.stderr)[-200:])
    finally: shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ P. the runner of the faults of the tools
def sec_P():
    """mutants/check_tools.py does what it says: a stale catalogue is reported with the reason; faults are selected by tool and by name; the snapshot the faults are copied from leaves out the history; a gate that is red on the
    unmutated copy, or runs no check, stops the run (exit 2); a fault the gate does not notice is a survivor (exit 1); a result saved for this fault and this tree is not run again.  The real catalogue is NEVER looked at
    here (section O pins it): it holds faults of the runner, and a fault injected into the runner makes the real catalogue stale in the copy, so any look at it would fail for every fault and prove nothing.  The catalogue
    in force during this section is a small one of the number checker and the digest, made below."""
    import importlib, io, contextlib, collections as _c
    sys.path.insert(0, os.path.join(ROOT, 'mutants')); CT = importlib.import_module('check_tools'); TC = importlib.import_module('tool_catalogue'); MU = importlib.import_module('mutate')
    real_tools, real_root, real_mu_root = TC.TOOLS, CT.ROOT, MU.ROOT
    readme = read(os.path.join(ROOT, 'README.md')); line = next(l_ for l_ in readme.split('\n') if len(l_) > 20 and readme.count(l_) == 1)
    keep = [m for m in real_tools['numbers']['mutants'] if m[0] == 'main: strict exit']
    safe = _c.OrderedDict([('numbers', dict(real_tools['numbers'], mutants=keep + [('doc: a line of the README', 'README.md', line, line + ' (edited)')])), ('digest', real_tools['digest'])])
    TC.TOOLS = safe
    def cli(*args):
        buf, err = io.StringIO(), io.StringIO(); code = None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            try: code = CT.main(list(args))
            except SystemExit as e: code = e.code
        return code, buf.getvalue() + err.getvalue()
    def write(root, rel, text):
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        with open(os.path.join(root, rel), 'w') as f: f.write(text)
    tmp = tempfile.mkdtemp(prefix='s45p_')
    try:
        # (1) a catalogue that is not sound says why: a doctored one, in a small tree
        for rel, text in (('src/m.py', 'x = 1\ny = 2\nx = 1\n'), ('src/note.md', 'some text\n'), ('tests/g.py', 'print(1)\n')): write(tmp, rel, text)
        base = dict(gate='tests/g.py', env={'S45_ONLY': 'N'}, data=False, about='x')
        def problems(mutants, **over):
            spec = dict(base, mutants=mutants); spec.update(over); TC.TOOLS = _c.OrderedDict([('t', spec)]); CT.ROOT = tmp
            try: return CT.check()
            finally: TC.TOOLS, CT.ROOT = safe, real_root
        ok = ('ok', 'src/m.py', 'y = 2', 'y = 3')
        cases = [([ok], None), ([ok, ('doc', 'src/note.md', 'text', 'words')], None), ([('two', 'src/m.py', 'x = 1', 'x = 2')], 'occurs 2 times'), ([('none', 'src/m.py', 'z = 9', 'z = 8')], 'occurs 0 times'), ([('same', 'src/m.py', 'y = 2', 'y = 2')], 'changes nothing'),
                 ([('syntax', 'src/m.py', 'y = 2', 'y = (')], 'does not compile'), ([('gone', 'src/nope.py', 'a', 'b')], 'does not exist'), ([ok, ('ok', 'src/m.py', 'y = 2', 'y = 4')], 'used twice'), ([('short', 'src/m.py', 'y = 2')], 'not (name, file, old, new)'),
                 ([ok], 'does not exist', dict(gate='tests/nope.py')), ([ok], 'env must be', dict(env={'S45_ONLY': 1})), ([ok], 'env must be', dict(env='N')), ([ok], 'data must be', dict(data=1)), ([], 'no faults')]
        got = []
        for case in cases:
            muts, want = case[0], case[1]; over = case[2] if len(case) > 2 else {}; ps = problems(muts, **over)
            got.append((want is None and ps == []) or (want is not None and len(ps) >= 1 and any(want in p_ for p_ in ps)))
        expect(all(got), 'P  a stale catalogue is reported with the reason (anchor twice or not at all, no change, a file that does not compile, a missing file or gate, a name twice, a malformed fault, env, data, no fault): %s' % got)
        expect(problems([ok, ('ok2', 'src/m.py', 'y = 2', 'y = 5')]) == [], 'P  two faults that replace the same text under two names are fine (each is run on its own)')
        # (2) selection and digests
        fs = CT.faults(['numbers'], ['strict exit'])
        expect([f['name'] for f in fs] == ['main: strict exit'] and fs[0]['tool'] == 'numbers' and re.fullmatch(r'[0-9a-f]{16}', fs[0]['digest']) and CT.faults(None, ['no such fault']) == []
               and [(f['tool'], f['name']) for f in CT.faults()] == [(t, m[0]) for t, s_ in safe.items() for m in s_['mutants']] and {f['tool'] for f in CT.faults(['digest'])} == {'digest'} and len(CT.faults(['numbers'])) == 2,
               'P  faults are selected by tool and by a text of their name, in the order of the catalogue')
        d0 = CT.fault_digest('a', 'b', 'c', 'd', 'e')
        expect(d0 == CT.fault_digest('a', 'b', 'c', 'd', 'e') and len({d0, CT.fault_digest('x', 'b', 'c', 'd', 'e'), CT.fault_digest('a', 'x', 'c', 'd', 'e'), CT.fault_digest('a', 'b', 'x', 'd', 'e'), CT.fault_digest('a', 'b', 'c', 'x', 'e'), CT.fault_digest('a', 'b', 'c', 'd', 'x')}) == 6,
               'P  the digest of a fault is stable and changes with the tool, the name, the file, the text it replaces and the text it puts there')
        # (3) what is read of the output of a gate; the summary; the saved results
        expect(CT.checks_of('417 checks\nS45 LAYOUT GATE: PASS') == 417 and CT.checks_of('S51 Q: 7 checks, 0 failed\nS51 O: 12 checks, 0 failed\nS51 19 checks, 0 failures, 3 s') == 19 and CT.checks_of('S45 LAYOUT GATE: PASS') == 0, 'P  the number of checks a gate says it ran (the last count: the total)')
        expect(CT.first_failure('x\n  FAIL: O13 something\n  FAIL: second\nS51 TRAINING SUITE GATE: FAIL') == 'O13 something' and CT.first_failure('12 checks\nS45 LAYOUT GATE: FAIL (2)\n    N  a check\n    N  another') == 'N  a check'
               and CT.first_failure('Traceback (most recent call last):\n  File "x"\nValueError: boom') == 'ValueError: boom' and CT.first_failure('') == '' and len(CT.first_failure('FAIL: ' + 'x' * 400)) == 150, 'P  the first thing a gate complains of')
        recs = [dict(tool='numbers', name='a', status='caught', kind='assertion'), dict(tool='numbers', name='b', status='caught', kind='crash'), dict(tool='numbers', name='c', status='gap', kind=''), dict(tool='digest', name='d', status='caught', kind='timeout')]
        lines, gaps = CT.summary(recs)
        expect(gaps == ['c'] and len(lines) == 2 and lines[0] == 'numbers   3 faults, 2 caught (assertion 1, crash 1), 1 survived: c' and lines[1] == 'digest    1 faults, 1 caught (timeout 1), 0 survived', 'P  the summary: faults, caught (and how), survivors by name (%s)' % lines)
        jf = os.path.join(tmp, 'saved.jsonl')
        with open(jf, 'w') as f: f.write(json.dumps(dict(tool='a', name='n', status='gap')) + '\n' + 'not json\n' + '[1]\n' + json.dumps({'name': 'x'}) + '\n' + json.dumps(dict(tool='a', name='n', status='caught')) + '\n' + '{"tool": "b", "na')
        sv = CT.read_saved(jf)
        expect(list(sv) == [('a', 'n')] and sv[('a', 'n')]['status'] == 'caught' and CT.read_saved(os.path.join(tmp, 'nope')) == {} and CT.read_saved(None) == {}, 'P  saved results: the last record of a fault counts; a line that is cut, no record or no tool is skipped (%s)' % list(sv))
        # (4) the snapshot the faults are copied from: the code, the documents and the tools the gates read, the compiled kernels; not the history, the archives, the runs or the caches -- tested on a small tree that has all of them
        fake = os.path.join(tmp, 'fake')
        for rel in ('src/a.py', 'tests/t.py', 'legacy/l.txt', 'data/d.json', 'docs/d.md', 'tools/t.py', 'scripts/s.sh', 'paper/p.tex', 'mutants/m.py', 'mutants/__pycache__/m.pyc', 'README.md', 'Makefile', '.git/HEAD', 'dist/a.tar.gz', 'runs/smoke/x', 'build/junk.o', 'artifacts/m.pt', '.venv/bin/python', 'results/README.md', 'results/full/big.csv'): write(fake, rel, rel)
        snap = os.path.join(tmp, 'snap'); CT.ROOT = MU.ROOT = fake
        try: CT.make_snapshot(snap)
        finally: CT.ROOT, MU.ROOT = real_root, real_mu_root
        top = set(os.listdir(snap)); have = lambda *p_: os.path.exists(os.path.join(snap, *p_))
        expect(top == {'src', 'tests', 'legacy', 'data', 'docs', 'tools', 'scripts', 'paper', 'mutants', 'README.md', 'Makefile', 'build', 'results'} and have('mutants', 'm.py') and have('docs', 'd.md') and not have('mutants', '__pycache__') and not have('build', 'junk.o')
               and os.listdir(os.path.join(snap, 'results')) == ['README.md']
               and sorted(os.listdir(os.path.join(snap, 'build'))) == sorted(os.path.basename(p_) for p_ in glob.glob(os.path.join(itp_paths.BUILD, 'itp_env_*.so')) + glob.glob(os.path.join(itp_paths.BUILD, 'itp_env_*.pyd'))),
               'P  the snapshot holds the code, the documents, the tools and the compiled kernels, and no history, archive, runs or cache (%s)' % sorted(top))
        sp = real_tools['audit']; kept_env = {k: os.environ.pop(k) for k in ('S51_DATA',) if k in os.environ}; os.environ['ITP_BUILD_DIR'] = '/no/such/build'            # (whatever the caller's environment holds)
        try:
            expect(CT.gate_env(sp, '/some/data').get('S51_DATA') == '/some/data' and 'S51_DATA' not in CT.gate_env(sp, None) and 'S51_DATA' not in CT.gate_env(safe['numbers'], '/some/data') and CT.gate_env(sp, None)['S51_ONLY'] == 'P' and 'ITP_BUILD_DIR' not in CT.gate_env(sp, None),
                   'P  the environment of a gate: its section, the small dataset only for the tools that need it, and no build directory of the caller')
        finally: os.environ.pop('ITP_BUILD_DIR', None); os.environ.update(kept_env)
        # (5) the runner end to end, with a catalogue of two faults of the number checker: one the gate notices and one in a file the gate does not look at
        jf = os.path.join(tmp, 'run.jsonl'); code, out = cli('numbers', '--json', jf); recs = [json.loads(l_) for l_ in open(jf)]
        expect(code == 1 and 'baseline green on numbers' in out and re.search(r'main: strict exit\s+CAUGHT', out) and re.search(r'doc: a line of the README\s+SURVIVED', out) and 'TOOL MUTANTS: 1 SURVIVED' in out
               and 'numbers   2 faults, 1 caught (assertion 1), 1 survived: doc: a line of the README' in out, 'P  a fault the gate notices is caught, one it does not is a survivor, and the exit status is 1: %s' % out[-400:])
        expect(len(recs) == 2 and sorted(r['status'] for r in recs) == ['caught', 'gap'] and all(r['tree'] == MU.tree_digest() and r['digest'] and r['tool'] == 'numbers' for r in recs), 'P  one record per fault, with the tree it was run on')
        code, out = cli('numbers', '--json', jf); n_lines = len(open(jf).read().split('\n')) - 1
        expect(code == 1 and out.count('(saved)') == 2 and 'baseline' not in out and n_lines == 2, 'P  a second run takes both results from the file and runs nothing (%s lines, %s)' % (n_lines, out[-200:]))
        code, out = cli('numbers', '--json', jf, '--redo', '--only', 'strict exit'); n_lines = len(open(jf).read().split('\n')) - 1
        expect(code == 0 and 'baseline green' in out and 'doc: a line' not in out and 'ALL CAUGHT' in out and n_lines == 3, 'P  --redo runs a fault again although the file holds it; --only selects by name; all caught: exit 0 (%s lines)' % n_lines)
        recs = [json.loads(l_) for l_ in open(jf)]
        for r_ in recs: r_['tree'] = 'f' * 16
        with open(jf, 'w') as f: f.write(''.join(json.dumps(r_) + '\n' for r_ in recs))
        code, out = cli('numbers', '--json', jf, '--only', 'strict exit')
        expect(code == 0 and '(saved)' not in out and 'baseline green' in out, 'P  a result saved for another tree is not reused (%s)' % out[-200:])
        recs = [json.loads(l_) for l_ in open(jf)]
        for r_ in recs: r_['tree'] = MU.tree_digest(); r_['digest'] = 'f' * 16
        with open(jf, 'w') as f: f.write(''.join(json.dumps(r_) + '\n' for r_ in recs))
        code, out = cli('numbers', '--json', jf, '--only', 'strict exit')
        expect(code == 0 and '(saved)' not in out and 'baseline green' in out, 'P  a result saved for another fault (another digest) is not reused (%s)' % out[-200:])
        code, out = cli('numbers', '--only', 'no such fault')
        expect(code == 2 and 'no fault selected' in out, 'P  nothing selected: exit 2')
        # (6) a gate that is red, or that runs no check, on the unmutated copy stops the run
        TC.TOOLS = _c.OrderedDict([('empty', dict(safe['numbers'], env={'S45_ONLY': 'ZZ'}))])
        try: code, out = cli('empty')
        finally: TC.TOOLS = safe
        expect(code == 2 and 'BASELINE IS RED (or ran no check)' in out and 'caught' not in out.lower(), 'P  a gate section that selects nothing is green and checks nothing: the run stops (%s)' % out[:160])
        # the baseline is green only when the gate exits 0, prints no failure and says it ran a check (the gate itself is replaced by a stand-in that answers what it is told to)
        def baseline_with(rc, text):
            real_run = CT.run_gate; CT.run_gate = lambda d, spec, data, timeout: (rc, text, 0.5); buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    try: res = CT.baseline(snap, ['numbers'], [None, False])
                    except SystemExit as e: res = e.code
            finally: CT.run_gate = real_run
            return res, buf.getvalue()
        green, red_rc, red_text, empty = baseline_with(0, '12 checks\nS45 LAYOUT GATE: PASS (1 s)\n'), baseline_with(1, '12 checks\nS45 LAYOUT GATE: PASS\n'), baseline_with(0, '12 checks\nS45 LAYOUT GATE: FAIL (1)\n    N  x\n'), baseline_with(0, 'S45 LAYOUT GATE: PASS\n')
        expect(green[0] == {'numbers': 0.5} and red_rc[0] == 2 and 'BASELINE IS RED' in red_rc[1] and red_text[0] == 2 and 'BASELINE IS RED' in red_text[1] and empty[0] == 2, 'P  the baseline: green with an exit status 0, no failure and a check run; red for a bad exit status, for a failure printed, and for no check (%s %s %s %s)' % (green[0], red_rc[0], red_text[0], empty[0]))
        # (7) the command line
        code, out = cli('--list', 'digest'); code_c, out_c = cli('--check')
        expect(code == 0 and 'tree_digest: a short digest' in out and 'mutants/mutate.py' in out and 'S45_ONLY=M' in out and 'numbers' not in out, 'P  --list names the faults of the tools asked for')
        expect(code_c == 0 and 'every one replaces text that occurs once' in out_c and out_c.startswith('%d faults of 2 tools' % sum(len(s_['mutants']) for s_ in safe.values())), 'P  --check: exit 0 and the number of faults (%s)' % out_c[:120])
        code, out = cli('nonsense'); expect(code == 2 and 'unknown tool' in out, 'P  an unknown tool: exit 2')
        TC.TOOLS = _c.OrderedDict([('t', dict(base, mutants=[('two', 'src/m.py', 'x = 1', 'x = 2')]))]); CT.ROOT = tmp
        try: code, out = cli('--check')
        finally: TC.TOOLS, CT.ROOT = safe, real_root
        expect(code == 2 and 'THE CATALOGUE IS STALE' in out and 'occurs 2 times' in out, 'P  a stale catalogue stops --check and a run: exit 2 (%s)' % out[:160])
    finally:
        TC.TOOLS, CT.ROOT, MU.ROOT = real_tools, real_root, real_mu_root; shutil.rmtree(tmp, ignore_errors=True)

def main():
    import time
    t0 = time.time()
    only = set(os.environ.get('S45_ONLY', '').replace(',', ' ').split())                   # (S45_ONLY=N: only these sections, for a mutation check of one of them)
    for name, fn in (('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E), ('F', sec_F), ('G', sec_G), ('H', sec_H), ('I', sec_I), ('J', sec_J), ('K', sec_K), ('L', sec_L), ('M', sec_M), ('N', sec_N), ('O', sec_O), ('P', sec_P)):
        if only and name not in only: continue
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
