#!/usr/bin/env python3
"""sanitize.py -- build the hybrid kernel with AddressSanitizer + UndefinedBehaviorSanitizer in a scratch directory and run the gates under it.

    python3 tools/sanitize.py                         build (about 80 s) and run the kernel and bank gates that can run under the sanitizers (long: ASan makes the slow gates several times slower; it says which it leaves out and why)
    python3 tools/sanitize.py s2_tests.py s16_quot_tests.py        only these gates
    python3 tools/sanitize.py --keep DIR              build into DIR and leave it there (default: a temporary directory that is deleted afterwards)

Why: the evaluator owns its memory by hand (intrusive reference counts, single-allocation sequences, a trampoline that frees deep chains iteratively), and
a bug there would not show up as a wrong answer -- it would be a use-after-free or a leak that the functional gates cannot see. The sanitizers turn such a
bug into an immediate, located error. Run this after any change to itp_kernel.hpp's value representation, Env, SharedSeq or the release machinery.
Leak detection is off here (the Python interpreter itself leaks at exit); the evaluator's leaks are checked separately by compiling a standalone driver with
-fsanitize=address and running it with detect_leaks=1.

The scratch directory is a copy of the directories of itp_paths.COPIED_DIRS (src, tests, legacy, data) with its own build/: the sanitized hybrid kernel, and the (ordinary) reference kernel copied from the
repository's build/.
"""
import glob, importlib.util, os, shutil, subprocess as sp, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402
import gate_registry as REG  # noqa: E402

ROOT = itp_paths.ROOT
# Gates that cannot say anything under the sanitizers (they are not run by default; name one on the command line to run it anyway):
NOT_UNDER_ASAN = {
    's24_bin_tests.py': 'its time limits (a 1500-bit product in 5 s) are for an unsanitized build, which ASan makes about ten times slower',
    's27_fuel_tests.py': 'it checks that an answer that runs out of fuel comes back within a few seconds, which an unsanitized build does and ASan (about ten times slower) does not',
    's26_memo_tests.py': 'its child throws out of a recursion a gigabyte deep: ASan then warns "__asan_handle_no_return: ignoring ... false positive error reports may follow" (google/sanitizers#189) and aborts; the same on the kernel before the macOS port',
}


def default_gates():
    """(the gates to run, [(gate, why not)]): the kernel and bank gates of `run_gates.py` without the ones that limit the address space (RLIMIT_AS: ASan reserves terabytes of it and cannot start) and those of NOT_UNDER_ASAN"""
    spec = importlib.util.spec_from_file_location('itp_run_gates', os.path.join(ROOT, 'tests', 'run_gates.py')); RG = importlib.util.module_from_spec(spec); spec.loader.exec_module(RG)
    run, skip = [], []
    for c in list(REG.GATE_CMDS) + list(REG.BANK_GATE_CMDS):
        name = os.path.basename(c[0])
        if RG.needs_rlimit_as(c): skip.append((name, 'it limits the address space (RLIMIT_AS), and ASan cannot start under a limit'))
        elif name in NOT_UNDER_ASAN: skip.append((name, NOT_UNDER_ASAN[name]))
        else: run.append(name)
    return run, skip


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
        for d in itp_paths.COPIED_DIRS: shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d), dirs_exist_ok=True, ignore=ign)
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
        if not gates:
            gates, skip = default_gates()
            print('not run under the sanitizers (%d of %d; name a gate to run it anyway):' % (len(skip), len(skip) + len(gates)), flush=True)
            for name, why in skip: print('  %-28s %s' % (name, why), flush=True)
        print('running %d gates under ASan + UBSan ...' % len(gates), flush=True)
        return sp.run([sys.executable, os.path.join(work, 'tests', 'run_gates.py'), '-j', '2'] + gates, cwd=work, env=env).returncode
    finally:
        if not keep: shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
