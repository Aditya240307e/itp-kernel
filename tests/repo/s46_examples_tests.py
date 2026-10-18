"""S46 gate: the examples run.

Every `examples/NN_name.py` is started in a fresh interpreter, from an unrelated working directory (they must not depend on it), and must exit 0, print no traceback, and end with the line
`EXAMPLE OK` (the example's own assertions -- a proof accepted by the kernel AND by the independent checker, a false statement refused, a recorded proof replayed -- sit before that line).  An example
that rots is documentation that lies; this gate is what keeps the README honest.

    S46_JOBS=2 python3 s46_examples_tests.py          run two at a time (default: one);  S46_ONLY=03,04 selects examples by number
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import concurrent.futures as cf
import glob, os, shutil, subprocess, sys, tempfile, time

FAIL = []; CHECKS = [0]


def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


def run_example(path, timeout=900):
    scratch = tempfile.mkdtemp(prefix='s46_')
    t0 = time.time()
    try:
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'ITP_BUILD_DIR')}
        r = subprocess.run([sys.executable, path], cwd=scratch, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr, time.time() - t0
    except subprocess.TimeoutExpired as e:
        return 99, (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), 'TIMEOUT after %d s' % timeout, time.time() - t0
    finally: shutil.rmtree(scratch, ignore_errors=True)


def main():
    t0 = time.time()
    only = [x for x in os.environ.get('S46_ONLY', '').split(',') if x]
    paths = sorted(glob.glob(os.path.join(itp_paths.EXAMPLES, '[0-9][0-9]_*.py')))
    if only: paths = [p for p in paths if os.path.basename(p)[:2] in only]
    expect(len(paths) >= 7 or only, 'there are at least seven examples (%d)' % len(paths))
    jobs = int(os.environ.get('S46_JOBS', '1'))
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {p: pool.submit(run_example, p) for p in paths}
        for p in paths:
            rc, out, err, dt = futs[p].result(); name = os.path.basename(p)
            lines = [l for l in out.strip().split('\n') if l.strip()]
            expect(rc == 0, '%s exits with status %d: %s' % (name, rc, (err or out)[-300:].replace('\n', ' | ')))
            expect('Traceback' not in err and 'Traceback' not in out, '%s printed a traceback' % name)
            expect(lines and lines[-1].strip() == 'EXAMPLE OK', '%s does not end with EXAMPLE OK (last line: %r)' % (name, lines[-1][:80] if lines else ''))
            expect(len(lines) >= 5, '%s printed only %d lines' % (name, len(lines)))
            if os.environ.get('S46_TRACE'): print('   [%s: %.1f s, rc %d, %d lines]' % (name, dt, rc, len(lines)), flush=True)
    print('%d checks' % CHECKS[0])
    if FAIL:
        print('S46 EXAMPLES GATE: FAIL (%d)' % len(FAIL))
        for f in FAIL[:30]: print('   ', f)
        sys.exit(1)
    print('S46 EXAMPLES GATE: PASS (%d examples, %.0f s)' % (len(paths), time.time() - t0))


if __name__ == '__main__':
    main()
