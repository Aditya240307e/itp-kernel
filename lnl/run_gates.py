#!/usr/bin/env python3
"""run_gates.py -- run every gate with the arguments the mutation driver uses and report PASS/FAIL per gate.

    python3 run_gates.py                      all gates, 2 at a time
    python3 run_gates.py -j 1                 serial (use this when timing matters)
    python3 run_gates.py s14_prop_world_tests.py s15_classical_tests.py     only these

The gate list lives in mutate.py (GATE_CMDS) so there is exactly one place to add a gate. A gate passes iff it
exits 0 and prints no "GATE: FAIL". Exit status is 0 iff every selected gate passed.
"""
import concurrent.futures as cf
import os, re, subprocess, sys, time
import mutate

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(cmd, timeout):
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, cmd[0])] + cmd[1:], cwd=HERE, env=mutate.gate_env(HERE),
                           capture_output=True, text=True, timeout=timeout)
        text, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        text, rc = ((e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or '')) + '\nTIMEOUT', 99
    ok = rc == 0 and not re.search(r'GATE: FAIL', text)
    return cmd, ok, time.time() - t0, text


def main(argv):
    jobs, timeout, names = 2, 1800, []
    it = iter(argv)
    for a in it:
        if a == '-j': jobs = int(next(it))
        elif a == '--timeout': timeout = int(next(it))
        else: names.append(a)
    cmds = [c for c in mutate.GATE_CMDS if not names or c[0] in names]
    unknown = set(names) - {c[0] for c in mutate.GATE_CMDS}
    if unknown:
        print('unknown gate(s): %s' % ', '.join(sorted(unknown))); return 2
    t0, failed = time.time(), []
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(run_one, c, timeout) for c in cmds]
        for f in cf.as_completed(futs):
            cmd, ok, dt, text = f.result()
            last = next((l.strip() for l in reversed(text.strip().split('\n')) if 'GATE' in l or 'PASS' in l or 'FAIL' in l), text.strip().split('\n')[-1] if text.strip() else '')
            print('%-5s %-34s %6.1fs  %s' % ('PASS' if ok else 'FAIL', ' '.join(cmd), dt, last[:90]), flush=True)
            if not ok:
                failed.append(cmd[0]); print('      ---- tail of output ----\n      ' + '\n      '.join(text.strip().split('\n')[-12:]), flush=True)
    print('%d/%d gates passed in %.0fs%s' % (len(cmds) - len(failed), len(cmds), time.time() - t0, ('   FAILED: ' + ', '.join(failed)) if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
