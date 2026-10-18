#!/usr/bin/env python3
"""run_gates.py -- run the gates with the arguments the mutation driver uses and report PASS/FAIL per gate.

    python3 tests/run_gates.py                       the kernel gates (the ones the mutation driver runs: kernel, tactics, environment, bank), 2 at a time
    python3 tests/run_gates.py -j 1                  serial (use this when timing matters)
    python3 tests/run_gates.py --ml                  + the learner's gate (s44; needs torch and gymnasium)
    python3 tests/run_gates.py --repo                + the repository gates (s45 layout, s46 examples)
    python3 tests/run_gates.py --all                 every gate
    python3 tests/run_gates.py s14_prop_world_tests.py s15_classical_tests.py       only these (by file name; the learner's and the repository's gates are found by name too)

The gate list lives in gate_registry.py, the one place to add a gate.  A gate passes iff it exits 0 and prints no "GATE: FAIL".  Exit status is 0 iff every selected gate passed.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa
import concurrent.futures as cf
import re, subprocess, time
import gate_registry as REG


def label(cmd): return ' '.join([os.path.basename(cmd[0])] + cmd[1:])


def run_one(cmd, timeout):
    t0 = time.time()
    path = os.path.join(itp_paths.TESTS, cmd[0])
    try:
        r = subprocess.run([sys.executable, path] + cmd[1:], cwd=os.path.dirname(path), capture_output=True, text=True, timeout=timeout)
        text, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        text, rc = ((e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or '')) + '\nTIMEOUT', 99
    ok = rc == 0 and not re.search(r'GATE: FAIL', text)
    return cmd, ok, time.time() - t0, text


def main(argv):
    jobs, timeout, names, extra = 2, 1800, [], []
    it = iter(argv)
    for a in it:
        if a == '-j': jobs = int(next(it))
        elif a == '--timeout': timeout = int(next(it))
        elif a == '--ml': extra += list(REG.ML_GATE_CMDS)
        elif a == '--repo': extra += list(REG.REPO_GATE_CMDS)
        elif a == '--all': extra += list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS)
        else: names.append(a)
    everything = list(REG.GATE_CMDS) + list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS)
    base = [os.path.basename(c[0]) for c in everything]
    unknown = set(names) - set(base)
    if unknown:
        print('unknown gate(s): %s' % ', '.join(sorted(unknown))); return 2
    if names: cmds = [c for c in everything if os.path.basename(c[0]) in names]          # named gates: exactly these, wherever they are registered
    else: cmds = list(REG.GATE_CMDS) + [c for c in extra if c not in REG.GATE_CMDS]
    t0, failed = time.time(), []
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(run_one, c, timeout) for c in cmds]
        for f in cf.as_completed(futs):
            cmd, ok, dt, text = f.result()
            last = next((l.strip() for l in reversed(text.strip().split('\n')) if 'GATE' in l or 'PASS' in l or 'FAIL' in l), text.strip().split('\n')[-1] if text.strip() else '')
            print('%-5s %-34s %6.1fs  %s' % ('PASS' if ok else 'FAIL', label(cmd), dt, last[:90]), flush=True)
            if not ok:
                failed.append(os.path.basename(cmd[0])); print('      ---- tail of output ----\n      ' + '\n      '.join(text.strip().split('\n')[-12:]), flush=True)
    print('%d/%d gates passed in %.0fs%s' % (len(cmds) - len(failed), len(cmds), time.time() - t0, ('   FAILED: ' + ', '.join(failed)) if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
