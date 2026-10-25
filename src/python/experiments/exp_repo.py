"""exp_repo.py -- facts about the repository itself (size of the trusted kernel, number of gates and mutants, size of the libraries), counted from the files, for the system section of the paper.

    facts = exp_repo.facts()          # {'kernel_lines': 7474, 'gates': 52, 'mutants': 1616, ...}

Nothing is typed in: a number that the paper quotes about the code is counted here from the code.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, glob, json, importlib.util, subprocess

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
    try:
        out['commit'] = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=20).stdout.strip()
        out['dirty'] = bool(subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception: pass
    return out


def dataset_facts(ddir):
    """what the dataset directory holds: library lemmas, annotated actions, sub-theorem goals"""
    out = {}
    for name, key in (('library.json', 'library_lemmas'), ('subproblems.json', 'subtheorems')):
        try:
            data = json.load(open(os.path.join(ddir, name)))
            recs = data.get('lemmas') or data.get('problems') or data.get('records') or (data if isinstance(data, list) else [])
            out[key] = len(recs)
        except (OSError, ValueError): pass
    return out


if __name__ == '__main__':
    print(json.dumps(facts(), indent=1))
