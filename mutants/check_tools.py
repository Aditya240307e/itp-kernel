#!/usr/bin/env python3
"""check_tools.py -- mutation testing of the TESTS OF THE TOOLS that produce the evidence of the paper (the soundness folder, the statement audit, the number checker, the identity of the code).

    python3 mutants/check_tools.py --check                verify the catalogue (tool_catalogue.py): every fault replaces text that occurs EXACTLY ONCE in its file, changes it, leaves a file that still compiles,
                                                          no name is used twice in a tool, every gate exists (a second, pinned by S45; no gate is run)
    python3 mutants/check_tools.py --list [TOOL ...]      the tools and their faults
    python3 mutants/check_tools.py [TOOL ...]             inject the faults one at a time, each into its own copy of the tree, and run the gate of the tool: the gate must FAIL for every one
      --only TEXT [TEXT ...]    only the faults whose name contains one of these texts
      --jobs N                  N copies at once (about 35 MB each, removed when the fault is done); the time a gate may take grows with the oversubscription of the cores
      --json FILE               append one JSON line per finished fault to FILE and, when FILE exists, SKIP the faults it holds for this very fault and this very tree (a long run can be interrupted and
                                continued); --redo runs them again
    TOOL_MUT_KEEP=1             keep the copy of the tree of a fault that survived, for inspection (the directory is printed)

The exit status is 0 when every fault was caught, 1 when one survived (a gap in the tests of the tool: write the check that fails for it, or -- when no input can tell the fault from the original -- remove the line
it re-breaks), 2 when the catalogue is stale or the gate is red on the unmutated copy.  A catch by `crash` (the section raised instead of checking) is a catch, but a weaker one: it is reported by kind.

The unmutated copy must be green first (otherwise every fault would look caught) and must run at least one check (a gate section that selects nothing is green too).  The small dataset that the gate of the
statement audit needs is built once, by `tests/learner/s51_suite_tests.py --build-data`, or taken from S51_DATA.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, concurrent.futures as cf, hashlib, json, re, shutil, subprocess, tempfile, threading, time
import mutate as MU
import tool_catalogue as TC

ROOT = itp_paths.ROOT
NOT_COPIED = ('.git', 'dist', 'runs', 'build', 'artifacts', 'results', '.venv', '__pycache__')      # (build/ is copied by MU.make_copy: the compiled kernels and nothing else; the others can be large and no gate of a tool reads them)


def faults(tools=None, only=None):
    """the faults to inject, in catalogue order: dicts (tool, name, file, old, new, digest); `tools` selects tools by name, `only` faults by a text of their name"""
    out = []
    for tool, spec in TC.TOOLS.items():
        if tools and tool not in tools: continue
        for name, f, old, new in spec['mutants']:
            if only and not any(o in name for o in only): continue
            out.append({'tool': tool, 'name': name, 'file': f, 'old': old, 'new': new, 'digest': fault_digest(tool, name, f, old, new)})
    return out


def fault_digest(tool, name, f, old, new):
    """what the fault IS: a result saved for another digest is not a result for this fault"""
    return hashlib.sha256('\0'.join((tool, name, f, old, new)).encode()).hexdigest()[:16]


def check():
    """the problems of the catalogue (empty: it is sound)"""
    problems, texts = [], {}
    if not TC.TOOLS: problems.append('the catalogue has no tools')
    for tool, spec in TC.TOOLS.items():
        gate = spec.get('gate', '')
        if not os.path.isfile(os.path.join(ROOT, gate)): problems.append('%s: the gate %r does not exist' % (tool, gate))
        if not (isinstance(spec.get('env'), dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in spec['env'].items())): problems.append('%s: env must be a dict of strings' % tool)
        if not isinstance(spec.get('data'), bool): problems.append('%s: data must be True or False' % tool)
        if not spec.get('mutants'): problems.append('%s: no faults' % tool)
        names = [m[0] for m in spec.get('mutants', [])]
        for n in sorted({n for n in names if names.count(n) > 1}): problems.append('%s: the name %r is used twice' % (tool, n))
        for entry in spec.get('mutants', []):
            if len(entry) != 4 or not all(isinstance(x, str) for x in entry): problems.append('%s: %r is not (name, file, old, new)' % (tool, entry[:1])); continue
            name, f, old, new = entry
            if f not in texts:
                p = os.path.join(ROOT, f); texts[f] = open(p).read() if os.path.isfile(p) else None
            if texts[f] is None: problems.append('%s / %s: the file %s does not exist' % (tool, name, f)); continue
            n = texts[f].count(old)
            if n != 1: problems.append('%s / %s: the text to replace occurs %d times in %s (it must occur once)' % (tool, name, n, f)); continue
            if old == new: problems.append('%s / %s: the fault changes nothing' % (tool, name)); continue
            if f.endswith('.py'):
                try: compile(texts[f].replace(old, new), f, 'exec')
                except SyntaxError as e: problems.append('%s / %s: the file does not compile with the fault (%s): a fault must break a rule, not the syntax' % (tool, name, e))
    return problems


def make_snapshot(d):
    """the tree as it is now, to copy from (a fault is run in a copy of the snapshot, not of the live tree: an edit made while a long run goes on cannot reach the faults still to come): what MU.make_copy copies
    (itp_paths.COPIED_DIRS, a build/ with the compiled kernels) and everything else the gates of the tools read (docs, tools, scripts, the manuscript)"""
    MU.make_copy(d, False)
    ign = shutil.ignore_patterns('__pycache__', '*.pyc')
    for name in sorted(os.listdir(ROOT)):
        if name in NOT_COPIED or name in itp_paths.COPIED_DIRS: continue
        src = os.path.join(ROOT, name)
        if os.path.isdir(src): shutil.copytree(src, os.path.join(d, name), ignore=ign)
        else: shutil.copy2(src, os.path.join(d, name))
    readme = os.path.join(ROOT, 'results', 'README.md')                                # (the one tracked file of results/: a link target of the documents)
    if os.path.isfile(readme): os.makedirs(os.path.join(d, 'results'), exist_ok=True); shutil.copy2(readme, os.path.join(d, 'results', 'README.md'))


def copy_snapshot(snap, d):
    shutil.copytree(snap, d, symlinks=True, dirs_exist_ok=True)


def gate_env(spec, data):
    env = MU.clean_env(); env.update(spec['env'])
    if spec['data'] and data: env['S51_DATA'] = data
    return env


def run_gate(d, spec, data, timeout):
    """the gate of a tool in the copy at d: (exit status, its output, seconds); a gate that does not stop is a catch by timeout"""
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.join(d, spec['gate'])], cwd=d, env=gate_env(spec, data), capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return 99, 'Traceback: HANG (timeout) -- the gate does not terminate', time.time() - t0


def checks_of(text):
    """how many checks the gate says it ran (S45: `N checks`; S51: `S51 N checks, M failures`)"""
    m = re.findall(r'(\d+) checks', text); return int(m[-1]) if m else 0


def first_failure(text):
    """the first thing the gate complains of: S51 lists `FAIL: ...` lines, S45 lists its failures below the `GATE: FAIL` line, a crash ends with the exception"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for l in lines:
        if l.startswith('FAIL: '): return l[6:][:150]
    for i, l in enumerate(lines):
        if 'GATE: FAIL' in l and i + 1 < len(lines): return lines[i + 1][:150]
    return (lines[-1] if lines else '')[:150]


def baseline(snap, todo_tools, data_dir):
    """the unmutated copy: green on the gate of every tool in todo_tools, with at least one check run (the seconds each gate takes are the base of the time a fault may take)"""
    d0 = tempfile.mkdtemp(prefix='tool_mut_base_'); base = {}
    try:
        copy_snapshot(snap, d0)
        if any(TC.TOOLS[t]['data'] for t in todo_tools) and not data_dir[0]:
            data_dir[0] = os.path.join(tempfile.mkdtemp(prefix='tool_mut_data_'), 'data'); data_dir[1] = True
            r = subprocess.run([sys.executable, os.path.join(d0, 'tests', 'learner', 's51_suite_tests.py'), '--build-data', data_dir[0]], cwd=d0, env=MU.clean_env(), capture_output=True, text=True)
            if r.returncode != 0: print('THE SMALL DATASET COULD NOT BE BUILT:\n' + (r.stdout + r.stderr)[-600:]); sys.exit(2)
        for tool in todo_tools:
            spec = TC.TOOLS[tool]; rc, text, secs = run_gate(d0, spec, data_dir[0], 1800)
            if MU.gate_failed(rc, text) or checks_of(text) < 1:
                print('BASELINE IS RED (or ran no check) for %s on an unmutated copy of the tree -- the results would mean nothing:' % tool); print(text[-700:]); sys.exit(2)
            base[tool] = secs
    finally: shutil.rmtree(d0, ignore_errors=True)
    return base


def run_fault(snap, ft, base, data, slow, keep=False):
    """one fault: a copy of the snapshot with the change, the gate of its tool; returns the record"""
    spec = TC.TOOLS[ft['tool']]; d = tempfile.mkdtemp(prefix='tool_mut_')
    rec = {k: ft[k] for k in ('tool', 'name', 'file', 'digest')}; t0 = time.time(); kept = False
    try:
        copy_snapshot(snap, d); target = os.path.join(d, ft['file']); text = open(target).read()
        assert text.count(ft['old']) == 1, (ft['tool'], ft['name'], text.count(ft['old']))
        with open(target, 'w') as fh: fh.write(text.replace(ft['old'], ft['new']))
        rc, out, _ = run_gate(d, spec, data, max(120, 5 * base[ft['tool']] * slow))
        if MU.gate_failed(rc, out):
            rec.update(status='caught', kind=MU.catch_kind(out), first=first_failure(out))
        else:
            rec.update(status='gap', kind='', first=''); kept = keep
    finally:
        if kept: print('   kept: %s' % d, flush=True)
        else: shutil.rmtree(d, ignore_errors=True)
    rec['seconds'] = round(time.time() - t0, 1); return rec


def read_saved(path):
    out = {}
    if path and os.path.exists(path):
        for line in open(path):
            try: r = json.loads(line)
            except ValueError: continue
            if isinstance(r, dict) and 'tool' in r and 'name' in r: out[(r['tool'], r['name'])] = r
    return out


def run(selected, jobs=1, json_path=None, redo=False):
    """inject the faults; returns the records (in catalogue order)"""
    tree = MU.tree_digest(); saved = {} if redo else read_saved(json_path); results, todo = {}, []
    for ft in selected:
        r = saved.get((ft['tool'], ft['name']))
        if r and r.get('digest') == ft['digest'] and r.get('tree') == tree and r.get('status') in ('caught', 'gap'):
            results[(ft['tool'], ft['name'])] = r; print('  %-9s %-52s %s  (saved)' % (ft['tool'], ft['name'][:52], r['status'].upper()), flush=True)
        else: todo.append(ft)
    data_dir = [os.environ.get('S51_DATA'), False]                                  # [directory of the small dataset, built here (and removed at the end)]
    if todo:
        snap = tempfile.mkdtemp(prefix='tool_mut_snap_')
        try: return finish(selected, results, todo, snap, tree, jobs, json_path, data_dir)
        finally: shutil.rmtree(snap, ignore_errors=True)
    return [results[(ft['tool'], ft['name'])] for ft in selected]


def finish(selected, results, todo, snap, tree, jobs, json_path, data_dir):
    """run the faults that have no saved result, in copies of a snapshot of the tree (the tree must be the one the digest is of)"""
    make_snapshot(snap)
    if MU.tree_digest() != tree: print('THE CODE CHANGED WHILE THE TREE WAS COPIED -- run again'); sys.exit(2)
    try:
        base = baseline(snap, sorted({ft['tool'] for ft in todo}, key=list(TC.TOOLS).index), data_dir)
        print('baseline green on %s (in an unmutated copy of the tree)' % ', '.join('%s (%.0f s)' % (t, s) for t, s in base.items()), flush=True)
        cores = MU.usable_cores(); slow = max(1.0, jobs / float(cores)); lock = threading.Lock(); keep = bool(os.environ.get('TOOL_MUT_KEEP'))
        def one(ft):
            rec = run_fault(snap, ft, base, data_dir[0], slow, keep); rec.update(tree=tree, jobs=jobs, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
            with lock:
                if json_path:
                    with open(json_path, 'a') as f: f.write(json.dumps(rec, sort_keys=True) + '\n')
                print('  %-9s %-52s %-8s (%4.1f s) %s' % (rec['tool'], rec['name'][:52], 'CAUGHT' if rec['status'] == 'caught' else 'SURVIVED', rec['seconds'],
                                                      ('[%s] %s' % (rec['kind'], rec['first'][:90])) if rec['status'] == 'caught' else '<-- a gap in the tests of the tool'), flush=True)
            return rec
        if jobs <= 1: new = [one(ft) for ft in todo]
        else:
            with cf.ThreadPoolExecutor(max_workers=jobs) as pool: new = list(pool.map(one, todo))
    finally:
        if data_dir[1]: shutil.rmtree(os.path.dirname(data_dir[0]), ignore_errors=True)
    for r in new: results[(r['tool'], r['name'])] = r
    if MU.tree_digest() != tree: print('NOTE: the code was edited while the faults ran; they ran on the tree as it was at the start (%s)' % tree)
    return [results[(ft['tool'], ft['name'])] for ft in selected]


def summary(records):
    """one line per tool: the faults, how many were caught (and how), the names of the survivors"""
    lines, gaps = [], []
    for tool in TC.TOOLS:
        rs = [r for r in records if r['tool'] == tool]
        if not rs: continue
        caught = [r for r in rs if r['status'] == 'caught']; surv = [r['name'] for r in rs if r['status'] != 'caught']
        kinds = ', '.join('%s %d' % (k, sum(1 for r in caught if r.get('kind') == k)) for k in ('assertion', 'crash', 'timeout') if any(r.get('kind') == k for r in caught))
        lines.append('%-9s %d faults, %d caught (%s), %d survived%s' % (tool, len(rs), len(caught), kinds or 'none', len(surv), (': ' + '; '.join(surv)) if surv else ''))
        gaps += surv
    return lines, gaps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tools', nargs='*', help='the tools to check (default: all): %s' % ', '.join(TC.TOOLS))
    ap.add_argument('--check', action='store_true', help='verify the catalogue and stop')
    ap.add_argument('--list', action='store_true', help='list the faults and stop')
    ap.add_argument('--only', nargs='+', default=[], metavar='TEXT', help='only the faults whose name contains one of these texts')
    ap.add_argument('--jobs', type=int, default=1, help='copies of the tree at once (default 1)')
    ap.add_argument('--json', default=None, metavar='FILE', help='append the records to FILE and skip what it holds for this fault and this tree')
    ap.add_argument('--redo', action='store_true', help='run the faults again although FILE holds them')
    a = ap.parse_args(argv)
    unknown = [t for t in a.tools if t not in TC.TOOLS]
    if unknown: ap.error('unknown tool %s (known: %s)' % (', '.join(unknown), ', '.join(TC.TOOLS)))
    problems = check()
    if problems:
        print('THE CATALOGUE IS STALE (fix it before running):'); [print('   ' + p) for p in problems]; return 2
    selected = faults(a.tools, a.only)
    if a.check:
        print('%d faults of %d tools: every one replaces text that occurs once, changes it and leaves a file that compiles' % (sum(len(s['mutants']) for s in TC.TOOLS.values()), len(TC.TOOLS))); return 0
    if a.list:
        for tool, spec in TC.TOOLS.items():
            if a.tools and tool not in a.tools: continue
            print('%s: %s\n   gate %s %s%s' % (tool, spec['about'], spec['gate'], ' '.join('%s=%s' % kv for kv in spec['env'].items()), ' (needs the small dataset)' if spec['data'] else ''))
            for ft in selected:
                if ft['tool'] == tool: print('   %-52s %s' % (ft['name'][:52], ft['file']))
        return 0
    if not selected: print('no fault selected'); return 2
    records = run(selected, max(1, a.jobs), a.json, a.redo)
    lines, gaps = summary(records)
    print('\n' + '\n'.join(lines)); print('TOOL MUTANTS: %s' % ('ALL CAUGHT' if not gaps else '%d SURVIVED' % len(gaps)))
    return 1 if gaps else 0


if __name__ == '__main__':
    sys.exit(main())
