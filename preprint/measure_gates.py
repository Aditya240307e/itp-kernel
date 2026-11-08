#!/usr/bin/env python3
"""measure_gates.py -- run the gates whose counts the report quotes, and keep what they print.

    python3 -I preprint/measure_gates.py                      run the quick gates (about 1 minute) and write preprint/measurements/gates/*.log and preprint/measurements/gates.json
    python3 -I preprint/measure_gates.py --with-env           also gate S40, the action language (about 80 seconds; it holds the recorder's coverage of the libraries)
    python3 -I preprint/measure_gates.py --parse-only         read the logs that exist again (no gate is run)
    python3 -I preprint/measure_gates.py --sanitizers LOG     read the log of `tools/sanitize.py` (made by hand: it builds the kernel again, about 80 seconds, and runs the gates under ASan and UBSan)

What a gate prints is the record: the log is kept next to the numbers parsed from it, and every number the text of the report quotes from a gate is read from `gates.json` by make_numbers.py.
A pattern that does not match is an error, not a missing number: the report must not quote a count that the gate no longer prints.
The gates are run from their own directories (each is a self-contained program), at the commit that is checked out; the tree must be clean below src/, tests/ and mutants/.
"""
import os, sys, re, json, subprocess, time, argparse, datetime

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(HERE, 'measurements', 'gates'); JSON = os.path.join(HERE, 'measurements', 'gates.json')

# name -> (directory below tests/, script, arguments)
QUICK = {
    's2': ('kernel', 's2_tests.py', ['1500']),
    'harness': ('support', 'harness.py', ['1500', '3']),
    'oracle': ('kernel', 'oracle_gate.py', []),
    's8': ('kernel', 's8_indep_tests.py', ['100']),
    's15': ('kernel', 's15_classical_tests.py', ['300']),
    's16': ('kernel', 's16_quot_tests.py', []),
    's19': ('kernel', 's19_boundary_tests.py', []),
}
ENV = {'s40': ('environment', 's40_actions_tests.py', [])}


def grab(text, pattern, what, flags=0, many=False):
    """the groups of the match(es) of `pattern` in `text`; an error when there is none"""
    ms = list(re.finditer(pattern, text, flags))
    if not ms: raise SystemExit('measure_gates: the gate no longer prints %s (pattern %r)' % (what, pattern))
    if many: return [m.groups() for m in ms]
    return ms[0].groups()


def ints(t): return [int(x) for x in t]


def parse(name, text):
    """the numbers the report quotes from the log of gate `name`"""
    r = {}
    if name == 's2':
        a, b, c, d, e = ints(grab(text, r'Match typing, (\d+) random typed terms with Match: .*?: (\d+) checked, (\d+) disagreements \| ill-typed matches rejected by both: (\d+) checked, (\d+) disagreements', 'the Match typing line'))
        assert a == b and c == 0 and e == 0, 'a disagreement'
        r.update(match_terms=a, match_disagreements=c, ill_typed_rejected=d, ill_typed_disagreements=e)
        n, acc, dis = ints(grab(text, r'dependent typing .*?: (\d+) checked \((\d+) accepted by the reference\) \| disagreements (\d+)', 'the dependent-typing line'))
        r.update(dep_checked=n, dep_accepted=acc, dep_disagreements=dis)
        n, fx, ok, rej = ints(grab(text, r'proof corpus: (\d+) real Fix\+Match proofs \((\d+) Fix nodes\): accepted at their statement: (\d+) \| proofs of FALSE statements rejected: (\d+)', 'the proof-corpus line'))
        r.update(corpus_proofs=n, corpus_fix_nodes=fx, corpus_accepted=ok, corpus_false_rejected=rej)
        b_block = text.split('B. admission corpus')[1].split('C. structural-recursion')[0]
        c_block = text.split('C. structural-recursion')[1].split('D. Match typing')[0]
        pat = r'expect (accept|reject) ref (accept|reject) new (accept|reject) OK'
        rows_ = re.findall(pat, b_block)              # the rows where the reference and the hybrid kernel agree, and those where the hybrid kernel is stricter on purpose (the reference accepts; F92)
        r['admission_cases'] = sum(1 for e_, a_, b_ in rows_ if a_ == b_); r['admission_departures'] = sum(1 for e_, a_, b_ in rows_ if a_ != b_); r['guard_cases'] = len(re.findall(pat, c_block))
        assert r['admission_cases'] and r['guard_cases'] and all((a_, b_) == ('accept', 'reject') for e_, a_, b_ in rows_ if a_ != b_)
        r['library_globals_redeclared'] = ints(grab(text, r'global types equal after conversion: yes \((\d+) vs (\d+) globals\)', 'the re-declared globals'))[0]
    elif name == 'harness':
        rows = grab(text, r'^\s+(norm|norm_open|conv|typing|typing_open|reject)\s+checked\s+(\d+) \| disagreements (\d+)', 'the comparison lines', re.M, many=True)
        r['comparisons'] = {k: int(n) for k, n, d in rows}; r['disagreements'] = sum(int(d) for k, n, d in rows)
        r['terms'] = int(grab(text, r'seed (\d+), N=(\d+) terms', 'the term count')[1]); r['seed'] = int(grab(text, r'seed (\d+), N=(\d+) terms', 'the seed')[0])
        n, fx, st, dis = ints(grab(text, r'proof corpus \(.*?\): (\d+) normalisations, (\d+) Fix nodes in the proofs, (\d+) with a stuck Fix/Match \| disagreements (\d+)', 'the proof-corpus line'))
        r.update(corpus_normalisations=n, corpus_fix=fx, corpus_stuck=st, corpus_disagreements=dis)
        assert r['disagreements'] == 0 and dis == 0
    elif name == 'oracle':
        rows = grab(text, r'^oracle (.*?)\s+unattributed failures: (\d+) \(completed (\d+), verified (\d+)\)', 'the oracle lines', re.M, many=True)
        r['distributions'] = [{'name': n.strip(), 'unattributed': int(u), 'completed': int(c), 'verified': int(v)} for n, u, c, v in rows]
        r['completed'] = sum(x['completed'] for x in r['distributions']); r['verified'] = sum(x['verified'] for x in r['distributions']); r['unattributed'] = sum(x['unattributed'] for x in r['distributions'])
        assert r['completed'] == r['verified'] and r['unattributed'] == 0
    elif name == 's8':
        rows = grab(text, r'mutants: (\d+), verdicts agree (\d+) \(both accept (\d+), both reject (\d+)\); kernel ACCEPTS / independent REJECTS: (\d+); independent ACCEPTS / kernel REJECTS: (\d+)', 'the mutant lines', many=True)
        r['runs'] = [dict(zip(('mutants', 'agree', 'both_accept', 'both_reject', 'k_acc_i_rej', 'i_acc_k_rej'), map(int, row))) for row in rows]
        for x in r['runs']: assert x['mutants'] == x['agree'] == x['both_accept'] + x['both_reject'] and x['k_acc_i_rej'] == 0 and x['i_acc_k_rej'] == 0
        r['mutants'] = sum(x['mutants'] for x in r['runs']); r['both_accept'] = sum(x['both_accept'] for x in r['runs']); r['both_reject'] = sum(x['both_reject'] for x in r['runs'])
    elif name == 's15':
        att, nfalse, comp = ints(grab(text, r'soundness canary \(classical axioms available\): (\d+) random proof attempts on (\d+) FALSE statements, (\d+) completed, verified proof of a false statement: None', 'the classical canary'))
        patt, pcomp, pver = ints(grab(text, r'canary POWER \(.*?\): (\d+) attempts, (\d+) completed, (\d+) verified', 'the classical power run'))
        r.update(attempts=att, false_statements=nfalse, completed=comp, power_attempts=patt, power_completed=pcomp, power_verified=pver)
    elif name == 's16':
        att, nfalse, comp = ints(grab(text, r'soundness canary \(Quot constants available\): (\d+) random proof attempts on (\d+) FALSE statements about quotients, (\d+) completed, verified proof of a false statement: None', 'the quotient canary'))
        ntrue, patt, pcomp, pver = ints(grab(text, r'canary POWER \(same generator, (\d+) TRUE statements\): (\d+) attempts, (\d+) completed, (\d+) verified', 'the quotient power run'))
        r.update(attempts=att, false_statements=nfalse, completed=comp, true_statements=ntrue, power_attempts=patt, power_completed=pcomp, power_verified=pver)
    elif name == 's19':
        cases, passed, failed = ints(grab(text, r'A\. boundary cases: (\d+), passed (\d+), failed (\d+)', 'the boundary cases'))
        runs = grab(text, r'B\. fuzz seed (\d+): (\d+) calls, no crash, no hang', 'the fuzz lines', many=True)
        r.update(cases=cases, passed=passed, failed=failed, fuzz_seeds=len(runs), fuzz_calls=sum(int(c) for s, c in runs), fuzz_calls_per_seed=int(runs[0][1]))
        assert failed == 0
    elif name == 's40':
        r.update(parse_s40(text))
    return r


def parse_s40(text):
    """the number of checks of the gate of the action language (the recorder's coverage, 453 of 459 proofs, is one of them; the gate fails below it)"""
    n, bad = ints(grab(text, r'^S40: (\d+) checks, (\d+) failures', 'the count of checks', re.M))
    assert bad == 0
    return {'checks': n}


def scrub(text):
    """the logs are published: no absolute path of the machine they were made on belongs in them"""
    text = text.replace(ROOT, '<repo>')
    home = os.path.expanduser('~')
    if home not in ('', '/'): text = text.replace(home, '~')
    return re.sub(r'/tmp/\S*', '<tmp>', text)


def run_gate(name, spec, timeout=1800):
    d, script, args = spec
    cwd = os.path.join(ROOT, 'tests', d)
    t0 = time.time()
    p = subprocess.run([sys.executable, script] + args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    sec = time.time() - t0
    text = scrub(p.stdout + p.stderr)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, name + '.log'), 'w') as f: f.write(text)
    print('%-8s exit %d  %.1f s' % (name, p.returncode, sec), flush=True)
    return {'cmd': 'python3 %s %s' % (os.path.join('tests', d, script), ' '.join(args)), 'exit': p.returncode, 'seconds': round(sec, 2), 'pass': p.returncode == 0 and 'GATE: FAIL' not in text}


def git(*a): return subprocess.run(['git', '-C', ROOT] + list(a), capture_output=True, text=True, check=True).stdout.strip()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-env', action='store_true'); ap.add_argument('--parse-only', action='store_true'); ap.add_argument('--sanitizers', metavar='LOG')
    a = ap.parse_args(argv)
    res = json.load(open(JSON)) if os.path.exists(JSON) else {'gates': {}}
    res['commit'] = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--', 'src', 'tests', 'mutants')
    if dirty and not a.parse_only: raise SystemExit('measure_gates: the code is modified:\n' + dirty)
    res['code_clean'] = not dirty
    res['date'] = datetime.date.today().isoformat()
    # the gates that were run again alone after the run of all of them (measurements/gates_rerun.jsonl): the commit of that run (the last commit before its first record), the commit that
    # corrected the gate, and the files below src/, tests/ and mutants/ that changed between the run and this commit.  The report states them; they are read here, where the history exists.
    ga_, rr_ = os.path.join(HERE, 'measurements', 'gates_all.jsonl'), os.path.join(HERE, 'measurements', 'gates_rerun.jsonl')
    if os.path.exists(ga_) and os.path.exists(rr_):
        again_ = [json.loads(l) for l in open(rr_) if l.strip()]
        first_ = git('log', '-1', '--before=' + min(json.loads(l)['time'] for l in open(ga_) if l.strip()), '--format=%H')
        scripts_ = sorted('tests/%s/%s' % (r_['group'], r_['gate']) for r_ in again_)
        res['rerun'] = {'gates': sorted(r_['gate'] for r_ in again_), 'first_commit': first_, 'fix_commit': git('log', '-1', '--format=%H', '--', *scripts_),
                        'changed': sorted(git('diff', '--name-only', first_, res['commit'], '--', 'src', 'tests', 'mutants').split()), 'scripts': scripts_}
    else: res.pop('rerun', None)
    todo = dict(QUICK); todo.update(ENV if a.with_env else {})
    if a.sanitizers:
        text = scrub(open(a.sanitizers).read())
        rows = grab(text, r'^(PASS|FAIL)\s+(.*?)\s+([\d.]+)s\s', 'the sanitizer lines', re.M, many=True)
        m = re.search(r'^(\d+)/(\d+) gates passed in (\d+)s', text, re.M)
        passed, total, secs = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert passed == total == len(rows), 'a gate failed under the sanitizers, or a line was not read'
        sys.path[:0] = [os.path.join(ROOT, 'src', 'python'), os.path.join(ROOT, 'tests'), os.path.join(ROOT, 'tools')]
        import importlib.util
        spec = importlib.util.spec_from_file_location('itp_sanitize', os.path.join(ROOT, 'tools', 'sanitize.py')); sz = importlib.util.module_from_spec(spec); spec.loader.exec_module(sz)
        run, skip = sz.default_gates()
        res['sanitizers'] = {'gates': [{'gate': g.strip(), 'pass': v == 'PASS', 'seconds': float(t)} for v, g, t in rows], 'seconds': secs,
                             'default_set': len(run), 'left_out_by_the_tool': [n for n, why in skip], 'build_flags': '-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g',
                             'options': 'ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1'}
        os.makedirs(OUTDIR, exist_ok=True)
        with open(os.path.join(OUTDIR, 'sanitize.log'), 'w') as f: f.write(text)
    for name, spec in todo.items():
        if not a.parse_only:
            meta = run_gate(name, spec)
        else:
            if not os.path.exists(os.path.join(OUTDIR, name + '.log')): continue
            meta = res['gates'].get(name, {})
        text = open(os.path.join(OUTDIR, name + '.log')).read()
        if not meta.get('pass', True): raise SystemExit('measure_gates: gate %s failed' % name)
        meta['parsed'] = parse(name, text)
        res['gates'][name] = meta
    with open(JSON, 'w') as f: json.dump(res, f, indent=1, sort_keys=True)
    print('wrote', JSON)


if __name__ == '__main__':
    main(sys.argv[1:])
