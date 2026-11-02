"""exp_repo.py -- facts about the repository itself (size of the trusted kernel, number of gates and mutants, size of the libraries), counted from the files, for the system section of the paper.

    facts = exp_repo.facts()          # {'kernel_lines': 7474, 'gates': 52, 'mutants': 1647, ...}

Nothing is typed in: a number that the paper quotes about the code is counted here from the code.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, glob, json, importlib.util, subprocess

ROOT = itp_paths.ROOT


def count_lines(paths):
    n = 0
    for p in paths:
        try:
            with open(p, 'rb') as f: n += sum(1 for _ in f)
        except OSError: pass
    return n


def python_files(sub):
    return [p for p in glob.glob(os.path.join(ROOT, sub, '**', '*.py'), recursive=True) if '__pycache__' not in p]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


CODE_PATHS = itp_paths.CODE_PATHS                                                     # the code, the tests, the mutants and the manifest of the dataset; a document, the paper, a tool or a script is not in it


def tree_digest():
    """the digest of the code (CODE_PATHS): the results of the gates, the soundness and audit folders and every run of the matrix are bound to it and not to the commit, so that typing the paper or editing a document never makes them stale"""
    d = os.path.join(ROOT, 'mutants')
    if d not in sys.path: sys.path.insert(0, d)
    import mutate
    return mutate.tree_digest()


def git_state(root=None):
    """{'commit': HEAD, 'dirty': the CODE has changes that are not committed (a tracked file of CODE_PATHS is changed, or a file there is new)}: the commit then does not identify the code, the tree digest does.
    A change to a document, the paper, a tool or a script is not dirty.  Both keys are missing when it is not a git checkout"""
    root = root or ROOT
    try:
        commit = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=20).stdout.strip()
        if not commit: return {}
        return {'commit': commit, 'dirty': bool(subprocess.run(['git', '-C', root, 'status', '--porcelain', '--untracked-files=normal', '--'] + list(itp_paths.CODE_PATHSPEC), capture_output=True, text=True, timeout=20).stdout.strip())}
    except Exception: return {}


def facts():
    out = {}
    hyb = os.path.join(ROOT, 'src', 'c++', 'hybrid')
    out['kernel_lines'] = count_lines([os.path.join(hyb, 'itp_kernel.hpp')]); out['bindings_lines'] = count_lines([os.path.join(hyb, 'itp_bindings.cpp')])
    out['checker_lines'] = count_lines(glob.glob(os.path.join(ROOT, 'src', 'python', 'checker', '*.py')))
    out['python_lines'] = count_lines(python_files('src/python')); out['test_lines'] = count_lines(python_files('tests'))
    out['python_modules'] = len(python_files('src/python'))
    try:
        reg = load_module(os.path.join(ROOT, 'tests', 'gate_registry.py'), 'gate_registry_facts')
        scripts = set()
        for group in ('GATE_CMDS', 'BANK_GATE_CMDS', 'ML_GATE_CMDS', 'REPO_GATE_CMDS', 'SYNTH_GATE_CMDS'):
            scripts |= {cmd[0] for cmd in getattr(reg, group, ())}
        out['gates'] = len(scripts); out['gates_mutation_suite'] = len({cmd[0] for cmd in reg.GATE_CMDS})
    except Exception as e: out['gates_error'] = str(e)
    try:
        cat = load_module(os.path.join(ROOT, 'mutants', 'catalogue.py'), 'mutant_catalogue_facts')
        out['mutants'] = len(cat.MUTANTS); out['mutants_equivalent'] = len(cat.KNOWN_EQUIVALENT)
    except Exception as e: out['mutants_error'] = str(e)
    out.update(git_state())
    try: out['tree'] = tree_digest()
    except Exception as e: out['tree_error'] = str(e)
    return out


def dataset_facts(ddir):
    """what the dataset directory holds: library lemmas, annotated actions, sub-theorem goals"""
    out = {}
    for name, key in (('library.json', 'library_lemmas'), ('subproblems.json', 'subtheorems')):
        try:
            data = json.load(open(os.path.join(ddir, name)))
            recs = data.get('lemmas') or data.get('examples') or data.get('problems') or data.get('records') or (data if isinstance(data, list) else [])     # ('examples': subproblems.json)
            out[key] = len(recs)
        except (OSError, ValueError): pass
    return out


if __name__ == '__main__':
    print(json.dumps(facts(), indent=1))
