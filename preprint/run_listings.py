#!/usr/bin/env python3
"""run_listings.py -- run the listings of the report and keep what they print.

    python3 -I preprint/run_listings.py           run listings/ex_*.py, dev_*.py and dev_lean_*.lean and write the .out files (the output printed in the report)
    python3 -I preprint/run_listings.py --check   run them again and compare with the .out files that exist (exit 1 if one differs)
    python3 -I preprint/run_listings.py dev_motive dev_positivity       only these (by name, without the extension)

The report prints each listing next to its output; the output is the file written here, never typed in.
The Python listings are run in isolated mode (-I) from their own directory; a listing that exits non-zero, or writes a traceback, is an error.
ex_gym.py builds the world of the problem bank (about a minute and 1.7 GB with PyTorch loaded).

The Lean listings (Appendix A) are a cross-check in a second proof assistant and need a Lean 4 on the PATH, or named by the environment variable LEAN;
the report used Lean 4.15.0 with the core library only (no Mathlib, no lake project).  Without a Lean they are skipped, which is said and is not an error.
Lean's messages (what it writes to the standard output) are the output; each listing must end with the exit status given in LEAN_EXIT,
because the first one is a refusal on purpose.
"""
import os, sys, subprocess, glob, time, shutil

HERE = os.path.dirname(os.path.abspath(__file__)); LST = os.path.join(HERE, 'listings')
LEAN_EXIT = {'dev_lean_decl': 1,      # Lean's kernel refuses the three declarations
             'dev_lean_false': 0}     # the file checks: False follows from the rules of R and Classical.propComplete


def run(path):
    t0 = time.time()
    p = subprocess.run([sys.executable, '-I', os.path.basename(path)], cwd=os.path.dirname(path), capture_output=True, text=True, timeout=1800)
    if p.returncode != 0 or 'Traceback' in p.stderr:
        raise SystemExit('%s failed (exit %d):\n%s' % (path, p.returncode, p.stderr[-2000:]))
    return p.stdout, time.time() - t0


def find_lean():
    return os.environ.get('LEAN') or shutil.which('lean')


def run_lean(path, lean):
    name = os.path.splitext(os.path.basename(path))[0]
    if name not in LEAN_EXIT: raise SystemExit('%s: give the exit status that Lean must end with in LEAN_EXIT' % name)
    t0 = time.time()
    p = subprocess.run([lean, os.path.basename(path)], cwd=os.path.dirname(path), capture_output=True, text=True, timeout=1800)
    if p.returncode != LEAN_EXIT[name] or p.stderr.strip():
        raise SystemExit('%s: Lean ended with status %d (expected %d)\n%s%s' % (path, p.returncode, LEAN_EXIT[name], p.stdout[-2000:], p.stderr[-2000:]))
    return p.stdout, time.time() - t0


def main(argv):
    check = '--check' in argv
    only = [a for a in argv if not a.startswith('--')]
    bad = 0
    lean = find_lean(); said = False
    paths = glob.glob(os.path.join(LST, 'ex_*.py')) + glob.glob(os.path.join(LST, 'dev_*.py')) + glob.glob(os.path.join(LST, 'dev_lean_*.lean'))
    for path in sorted(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        if only and name not in only: continue
        if path.endswith('.lean'):
            if not lean:
                print('%-14s skipped (no Lean 4 on the PATH, and LEAN is not set)' % name); continue
            if not said:
                said = True; print('Lean: %s' % subprocess.run([lean, '--version'], capture_output=True, text=True).stdout.strip())
            out, sec = run_lean(path, lean)
        else:
            out, sec = run(path)
        target = os.path.join(LST, name + '.out')
        if check:
            same = os.path.exists(target) and open(target).read() == out
            print('%-14s %s  (%.1f s)' % (name, 'same' if same else 'DIFFERENT', sec)); bad += not same
        else:
            with open(target, 'w') as f: f.write(out)
            print('%-14s wrote %s  (%.1f s)' % (name, os.path.relpath(target, HERE), sec))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
