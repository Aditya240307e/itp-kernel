#!/usr/bin/env python3
"""run_listings.py -- run the listings of the report and keep what they print.

    python3 -I preprint/run_listings.py           run listings/ex_*.py and write listings/ex_*.out (the output printed in the report)
    python3 -I preprint/run_listings.py --check   run them again and compare with the .out files that exist (exit 1 if one differs)
    python3 -I preprint/run_listings.py dev_motive dev_positivity       only these (by name, without .py)

The report prints each listing next to its output; the output is the file written here, never typed in.
The listings are run in isolated mode (-I) from their own directory; a listing that exits non-zero, or writes a traceback, is an error.
ex_gym.py builds the world of the problem bank (about a minute and 1.7 GB with PyTorch loaded).
"""
import os, sys, subprocess, glob, time

HERE = os.path.dirname(os.path.abspath(__file__)); LST = os.path.join(HERE, 'listings')


def run(path):
    t0 = time.time()
    p = subprocess.run([sys.executable, '-I', os.path.basename(path)], cwd=os.path.dirname(path), capture_output=True, text=True, timeout=1800)
    if p.returncode != 0 or 'Traceback' in p.stderr:
        raise SystemExit('%s failed (exit %d):\n%s' % (path, p.returncode, p.stderr[-2000:]))
    return p.stdout, time.time() - t0


def main(argv):
    check = '--check' in argv
    only = [a for a in argv if not a.startswith('--')]
    bad = 0
    for path in sorted(glob.glob(os.path.join(LST, 'ex_*.py')) + glob.glob(os.path.join(LST, 'dev_*.py'))):
        if only and os.path.basename(path)[:-3] not in only: continue
        out, sec = run(path); name = os.path.basename(path)[:-3]; target = os.path.join(LST, name + '.out')
        if check:
            same = os.path.exists(target) and open(target).read() == out
            print('%-10s %s  (%.1f s)' % (name, 'same' if same else 'DIFFERENT', sec)); bad += not same
        else:
            with open(target, 'w') as f: f.write(out)
            print('%-10s wrote %s  (%.1f s)' % (name, os.path.relpath(target, HERE), sec))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
