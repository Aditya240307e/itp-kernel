#!/usr/bin/env python3
"""sanitize.py -- build the hybrid kernel with AddressSanitizer + UndefinedBehaviorSanitizer in a scratch directory and run the gates under it.

    python3 tools/sanitize.py                         build (about 80 s) and run every kernel gate (about 4 minutes)
    python3 tools/sanitize.py s2_tests.py s16_quot_tests.py        only these gates
    python3 tools/sanitize.py --keep DIR              build into DIR and leave it there (default: a temporary directory that is deleted afterwards)

Why: the evaluator owns its memory by hand (intrusive reference counts, single-allocation sequences, a trampoline that frees deep chains iteratively), and
a bug there would not show up as a wrong answer -- it would be a use-after-free or a leak that the functional gates cannot see. The sanitizers turn such a
bug into an immediate, located error. Run this after any change to itp_kernel.hpp's value representation, Env, SharedSeq or the release machinery.
Leak detection is off here (the Python interpreter itself leaks at exit); the evaluator's leaks are checked separately by compiling a standalone driver with
-fsanitize=address and running it with detect_leaks=1.

The scratch directory is a copy of src/ and tests/ with its own build/: the sanitized hybrid kernel, and the (ordinary) reference kernel copied from the repository's build/.
"""
import glob, os, shutil, subprocess as sp, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    keep = None; gates = []
    it = iter(argv)
    for a in it:
        if a == '--keep': keep = next(it)
        else: gates.append(a)
    work = keep or tempfile.mkdtemp(prefix='itp_asan_')
    os.makedirs(work, exist_ok=True)
    try:
        ign = shutil.ignore_patterns('__pycache__', '*.pyc')
        for d in ('src', 'tests'): shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d), dirs_exist_ok=True, ignore=ign)
        os.makedirs(os.path.join(work, 'build'), exist_ok=True)
        for so in glob.glob(os.path.join(ROOT, 'build', 'itp_env_ref*')): shutil.copy2(so, os.path.join(work, 'build'))
        env = dict(os.environ, ITP_EXTRA_CXXFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g')
        env.pop('ITP_BUILD_DIR', None)
        print('building the sanitizer variant in %s ...' % work, flush=True)
        r = sp.run([sys.executable, os.path.join(work, 'src', 'python', 'build_kernel.py'), 'hybrid', '--out', os.path.join(work, 'build'), '--force'], cwd=work, env=env, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); return 2
        libs = [sp.run(['gcc', '-print-file-name=' + n], capture_output=True, text=True).stdout.strip() for n in ('libasan.so', 'libstdc++.so')]
        env = dict(os.environ, LD_PRELOAD=':'.join(libs), ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1')
        env.pop('ITP_BUILD_DIR', None)
        print('running the gates under ASan + UBSan ...', flush=True)
        return sp.run([sys.executable, os.path.join(work, 'tests', 'run_gates.py'), '-j', '2'] + gates, cwd=work, env=env).returncode
    finally:
        if not keep: shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
