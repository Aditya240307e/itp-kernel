#!/usr/bin/env python3
"""run_gates.py -- run the gates with the arguments the mutation driver uses and report PASS/FAIL per gate.

    python3 tests/run_gates.py                       the kernel gates (the ones the mutation driver runs: kernel, tactics, environment, bank) and the gates of the contest problems, 2 at a time (never two of the
                                                     gates that build the whole bank's world at once: gate_registry.HEAVY_GATES)
    python3 tests/run_gates.py -j 1                  serial (use this when timing matters)
    python3 tests/run_gates.py --ml                  + the learner's gates (s44 machinery, s51 training suite; need torch and gymnasium)
    python3 tests/run_gates.py --repo                + the repository gates (s45 layout, s46 examples)
    python3 tests/run_gates.py --synth               + the gate of the synthetic theorem generator (s48)
    python3 tests/run_gates.py --all                 every gate
    python3 tests/run_gates.py s14_prop_world_tests.py s15_classical_tests.py       only these (by file name; the learner's, the repository's and the generator's gates are found by name too)
    python3 tests/run_gates.py --all --json gates.jsonl                              also append one JSON line per finished gate to the file (gate, its arguments, group, ok, seconds, the verdict line and,
                                                                                     for a gate that failed, the end of its output); exp_soundness.py reads it

The gate list lives in gate_registry.py, the one place to add a gate.  A gate passes iff it exits 0 and prints no "GATE: FAIL".  Exit status is 0 iff every selected gate passed.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa
import concurrent.futures as cf
import json, re, subprocess, time
import gate_registry as REG


def label(cmd): return ' '.join([os.path.basename(cmd[0])] + cmd[1:])


def group_of(cmd):
    """the directory of a gate below tests/ (kernel, library, environment, problems, learner, repo, synth, support): which part of the system it tests"""
    return os.path.dirname(cmd[0]) or '.'


def verdict_line(text):
    """the line of a gate's output that states its verdict (the last one that mentions GATE, PASS or FAIL)"""
    lines = text.strip().split('\n')
    return next((l.strip() for l in reversed(lines) if 'GATE' in l or 'PASS' in l or 'FAIL' in l), lines[-1] if text.strip() else '')


def gate_record(cmd, ok, dt, text):
    """what is saved of a finished gate (--json): enough to tell, afterwards, which gate ran with which arguments, how long it took and why it failed"""
    rec = {'gate': os.path.basename(cmd[0]), 'args': list(cmd[1:]), 'group': group_of(cmd), 'ok': bool(ok), 'seconds': round(dt, 1), 'verdict': verdict_line(text)[:200],
           'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    if not ok: rec['tail'] = text.strip().split('\n')[-12:]
    return rec


def needs_rlimit_as(cmd):
    """True if the gate limits the address space (RLIMIT_AS): Linux enforces that limit, macOS does not, so there such a gate would fail without a defect behind it"""
    try:
        with open(os.path.join(itp_paths.TESTS, cmd[0])) as f: return 'resource.RLIMIT_AS' in f.read()                  # (a gate that sets the limit through the resource module)
    except OSError: return False


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
    jobs, timeout, names, extra, json_path = 2, 5400, [], [], None  # (a gate that runs longer than this is hung: S50 takes about half an hour, S43 and S46 up to ten minutes)
    it = iter(argv)
    for a in it:
        if a == '-j': jobs = int(next(it))
        elif a == '--json': json_path = next(it)
        elif a == '--timeout': timeout = int(next(it))
        elif a == '--ml': extra += list(REG.ML_GATE_CMDS)
        elif a == '--repo': extra += list(REG.REPO_GATE_CMDS)
        elif a == '--synth': extra += list(REG.SYNTH_GATE_CMDS)
        elif a == '--all': extra += list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS) + list(REG.SYNTH_GATE_CMDS)
        else: names.append(a)
    everything = list(REG.GATE_CMDS) + list(REG.BANK_GATE_CMDS) + list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS) + list(REG.SYNTH_GATE_CMDS)
    base = [os.path.basename(c[0]) for c in everything]
    unknown = set(names) - set(base)
    if unknown:
        print('unknown gate(s): %s' % ', '.join(sorted(unknown))); return 2
    if names: cmds = [c for c in everything if os.path.basename(c[0]) in names]          # named gates: exactly these, wherever they are registered
    else: cmds = list(REG.GATE_CMDS) + list(REG.BANK_GATE_CMDS) + [c for c in extra if c not in REG.GATE_CMDS]
    skipped = []
    if sys.platform != 'linux':                                                                    # (a Mac: the gates that need RLIMIT_AS are skipped and said so; the rest is a real run, but not the full one)
        skipped = [c for c in cmds if needs_rlimit_as(c)]; cmds = [c for c in cmds if c not in skipped]
        if skipped: print('SKIP  %d gate(s) that limit the address space (RLIMIT_AS, which only Linux enforces): %s -- this is NOT the full set' % (len(skipped), ', '.join(os.path.basename(c[0]) for c in skipped)), flush=True)
    t0, failed = time.time(), []
    pending, running = list(cmds), {}
    heavy = lambda c: os.path.basename(c[0]) in REG.HEAVY_GATES
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        while pending or running:
            heavy_running = any(heavy(c) for c in running.values()); i = 0
            while len(running) < jobs and i < len(pending):                              # (the next gate that may start: not a second heavy one)
                if heavy(pending[i]) and heavy_running: i += 1; continue
                c = pending.pop(i); running[pool.submit(run_one, c, timeout)] = c; heavy_running = heavy_running or heavy(c)
            done, _ = cf.wait(list(running), return_when=cf.FIRST_COMPLETED)
            for f in done:
                running.pop(f)
                cmd, ok, dt, text = f.result()
                last = verdict_line(text)
                if json_path:
                    with open(json_path, 'a') as jf: jf.write(json.dumps(gate_record(cmd, ok, dt, text), sort_keys=True) + '\n')
                print('%-5s %-34s %6.1fs  %s' % ('PASS' if ok else 'FAIL', label(cmd), dt, last[:90]), flush=True)
                if not ok:
                    failed.append(os.path.basename(cmd[0])); print('      ---- tail of output ----\n      ' + '\n      '.join(text.strip().split('\n')[-12:]), flush=True)
    print('%d/%d gates passed in %.0fs%s%s' % (len(cmds) - len(failed), len(cmds), time.time() - t0, ('   FAILED: ' + ', '.join(failed)) if failed else '', ('   (%d skipped on this platform)' % len(skipped)) if skipped else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
