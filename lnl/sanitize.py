#!/usr/bin/env python3
"""sanitize.py -- build the kernel with AddressSanitizer + UndefinedBehaviorSanitizer in a scratch directory and run the gates under it.

    python3 sanitize.py                         build (about 80 s) and run every gate (about 4 minutes)
    python3 sanitize.py s2_tests.py s16_quot_tests.py        only these gates
    python3 sanitize.py --keep DIR              build into DIR and leave it there (default: a temporary directory that is deleted afterwards)

Why: the evaluator owns its memory by hand (intrusive reference counts, single-allocation sequences, a trampoline that frees deep chains iteratively), and
a bug there would not show up as a wrong answer -- it would be a use-after-free or a leak that the functional gates cannot see. The sanitizers turn such a
bug into an immediate, located error. Run this after any change to itp_kernel.hpp's value representation, Env, SharedSeq or the release machinery.
Leak detection is off here (the Python interpreter itself leaks at exit); the evaluator's leaks are checked separately by compiling a standalone driver with
-fsanitize=address and running it with detect_leaks=1 (see PERFORMANCE.md).
"""
import os, shutil, subprocess, subprocess as sp, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.environ.get('ITP_REF_DIR', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'itp_ref'))


def main(argv):
    keep = None; gates = []
    it = iter(argv)
    for a in it:
        if a == '--keep': keep = next(it)
        else: gates.append(a)
    work = keep or tempfile.mkdtemp(prefix='itp_asan_')
    os.makedirs(work, exist_ok=True)
    try:
        for name in os.listdir(HERE):
            if name in ('.git', '__pycache__') or name.endswith('.so'): continue
            src = os.path.join(HERE, name)
            if os.path.isdir(src): shutil.copytree(src, os.path.join(work, name), dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
            else: shutil.copy2(src, os.path.join(work, name))
        env = dict(os.environ, ITP_EXTRA_CXXFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g')
        print('building the sanitizer variant in %s ...' % work, flush=True)
        r = sp.run([sys.executable, 'build_kernel.py'], cwd=work, env=env, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); return 2
        libs = [sp.run(['gcc', '-print-file-name=' + n], capture_output=True, text=True).stdout.strip() for n in ('libasan.so', 'libstdc++.so')]
        env = dict(os.environ, ITP_PY_ROOT=os.path.dirname(HERE), ITP_REF_DIR=REF_DIR, LNL_DIR=work, LD_PRELOAD=':'.join(libs),
                   ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1')
        print('running the gates under ASan + UBSan ...', flush=True)
        return sp.run([sys.executable, 'run_gates.py', '-j', '2'] + gates, cwd=work, env=env).returncode
    finally:
        if not keep: shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
