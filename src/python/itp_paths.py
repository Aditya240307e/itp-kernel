"""itp_paths.py -- where everything lives.  The only module that knows the layout of the repository.

Every script and module of the project begins with one line that puts src/python on sys.path and imports this module:

    import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa

(the number of '..' depends on the depth of the file; `tests/run_gates.py` and the like have their own).  Importing it puts on sys.path

  * the build directory, which holds the compiled kernels (itp_env_lnl, itp_env_ref); $ITP_BUILD_DIR overrides it,
  * src/python and each of its subdirectories (the library is organised by ROLE, but a module is imported by its plain name: `import gym_env`),
  * tests and each of its subdirectories (the gates and the modules they share).

Module names are unique across all of these directories (the layout gate, tests/repo/s45_layout_tests.py, checks it), so the order does not matter.  The root is derived from THIS file's location, never from
the working directory or from an environment variable: a copy of src/ and tests/ (the mutation driver tests mutated copies) is a complete, self-consistent tree.

Constants: ROOT SRC_PY SRC_CPP TESTS GOLDEN MUTANTS EXAMPLES DOCS DATA LEGACY BUILD
"""
import os, sys

SRC_PY = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SRC_PY))
SRC_CPP = os.path.join(ROOT, 'src', 'c++')
TESTS = os.path.join(ROOT, 'tests')
GOLDEN = os.path.join(TESTS, 'golden')
MUTANTS = os.path.join(ROOT, 'mutants')
EXAMPLES = os.path.join(ROOT, 'examples')
DOCS = os.path.join(ROOT, 'docs')
DATA = os.path.join(ROOT, 'data')
LEGACY = os.path.join(ROOT, 'legacy')
BUILD = os.environ.get('ITP_BUILD_DIR') or os.path.join(ROOT, 'build')

_NOT_SOURCE = ('golden',)


def subdirs(base):
    """the directories directly under `base` that hold modules (not hidden, not caches, not data)"""
    if not os.path.isdir(base): return []
    return sorted(os.path.join(base, n) for n in os.listdir(base)
                  if os.path.isdir(os.path.join(base, n)) and not n.startswith(('.', '_')) and n not in _NOT_SOURCE)


def source_dirs():
    """every directory whose modules are imported by their plain names, in the order they are put on sys.path"""
    return [BUILD, SRC_PY] + subdirs(SRC_PY) + [TESTS] + subdirs(TESTS)


def setup():
    for p in reversed(source_dirs()):
        if p not in sys.path: sys.path.insert(0, p)


setup()
