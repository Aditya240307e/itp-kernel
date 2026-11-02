"""exp_soundness.py -- the SOUNDNESS EVIDENCE of the paper, collected in one stamped folder.

    python3 src/python/experiments/exp_soundness.py run [--out results/soundness] [--stages gates,mutants,fuzz,differential,inventory,findings] [--jobs 2]
                      [--mutants PREFIX ...] [--mut-gates s22_simp_tests.py,..] [--mut-full] [--fuzz-seeds 0-4] [--fuzz-n 300] [--fuzz-seconds 5] [--diff-per-proof 1000] [--redo]
    python3 src/python/experiments/exp_soundness.py report [--out results/soundness]        # tables, numbers and README from the saved evidence; runs nothing
    python3 src/python/experiments/exp_soundness.py --verify results/soundness                # is every file the one that was written?

What it holds (the claims of a verifier that others are asked to trust):
  gates         every gate passes -- tests/run_gates.py --all --json
  mutants       the mutation catalogue (seeded faults, one rule re-broken each, in an isolated copy of the tree) is caught by the gates: how many, by what (assertion / crash / timeout),
                which survive and why (a documented equivalent mutant, or a gap in the tests) -- mutants/mutate.py --json
  fuzz          falsification campaigns through the learner's environment: statements that are FALSE at some point are searched for proofs; a proof of one would be a bug of the kernel (UNSOUND),
                a proof the kernel refuses a bug of the tactic layer (REJECTED) -- tests/fuzz/gym_fuzz.py --json, over seeds x worlds (naturals with and without the arithmetic tactics, integers)
  differential  the kernel against the independently written checker (tests/kernel/s8_indep_tests.py): every library definition, tactic proofs, thousands of mutated proofs, in both modes
  inventory     what the trusted code is (lines, files, digests) against what is not, counted from the files
  findings      the findings (docs/FINDINGS.md) classified by layer and effect (docs/findings_tags.tsv), the open ones listed

Folder:
    README.md  report.md  soundness.json  numbers.json  numbers.tex  MANIFEST.sha256
    tables/<key>.{md,tex,csv}        sound_summary sound_gates sound_mutants sound_survivors sound_fuzz sound_differential sound_trusted sound_findings sound_open
    stats/*.csv                      mutants.csv (one row per mutant of the catalogue), mutants_by_series.csv, gates.csv, fuzz.csv, findings.csv
    evidence/                        stamp.json gates.jsonl mutants.jsonl fuzz.jsonl differential.json inventory.json logs/   (the raw records; the rest is computed from them)

Everything but evidence/ is a pure function of evidence/ (and of docs/FINDINGS.md, docs/findings_tags.tsv and the mutation catalogue): `report` makes it again.  A record is evidence for THIS code only if it
carries the digest of the tree it ran on (mutants/mutate.py tree_digest); a record of another tree is stale and does not count -- the folder says so instead of quoting it.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, re, json, time, glob, math, shutil, hashlib, platform, argparse, subprocess, collections
import concurrent.futures as cf
import exp_tables as TB, exp_bundle as EB, exp_report as ER, exp_repo as RP

ROOT = itp_paths.ROOT
STAGES = ('gates', 'mutants', 'fuzz', 'differential', 'inventory', 'findings')
DEFAULT_OUT = os.path.join(ROOT, 'results', 'soundness')
TAGS_PATH = os.path.join(ROOT, 'docs', 'findings_tags.tsv')
FINDINGS_PATH = os.path.join(ROOT, 'docs', 'FINDINGS.md')

# falsification campaigns: (world, noarith).  In the integer world every statement needs ring / linarith / nlinarith / interval_cases, so "without arithmetic" would prove nothing there.
FUZZ_CAMPAIGNS = (('nat', False), ('nat', True), ('int', False))
FUZZ_LABEL = {('nat', False): 'naturals', ('nat', True): 'naturals, without the arithmetic tactics', ('int', False): 'integers'}
FUZZ_DEFAULTS = dict(seeds='0-4', n=300, depth=5, probes=400, seconds=5.0)
DIFF_DEFAULT = 1000

GROUP_LABEL = collections.OrderedDict([
    ('kernel', 'kernel, independent checker, memoisation, work budget'), ('library', 'tactic libraries (simp, ring, linarith, number theory)'),
    ('environment', "learner's environment (actions, term builder, probes)"), ('problems', 'problem bank and contest proofs'), ('learner', 'learner and experiment suite'),
    ('repo', 'repository layout, examples'), ('synth', 'synthetic theorem generator'), ('support', 'randomised lockstep harness')])
COMPONENT = {'checker': 'independent checker', 'authoring': 'tactic layer and decision procedures', 'environment': "learner's environment", 'library': 'libraries',
             'learner': 'learner', 'problems': 'problem bank', 'synth': 'theorem generator', 'experiments': 'experiment tooling'}
COMPONENT_ORDER = ('kernel (C++)', 'independent checker', 'tactic layer and decision procedures', "learner's environment", 'libraries')

LAYERS = collections.OrderedDict([('kernel', 'the trusted C++ kernel: typing, conversion, admission, work budget, memoisation'), ('checker', 'the independent checker'),
                                  ('tactic', 'the tactic layer, the unifier and the libraries (untrusted: the kernel checks every term they build)'),
                                  ('engine', "the learner's environment: engine, probe protocol, action language, term builder, search (untrusted)"),
                                  ('data', 'the problem bank, recorder, dataset, hindsight, the synthetic generator and the semantic checks (untrusted)'),
                                  ('pipeline', 'the training suite, the experiment tooling, estimates, documents'), ('tooling', 'the mutation catalogue and driver, the repository layout, the gates')])
EFFECTS = collections.OrderedDict([('accepts', 'trusted code accepted a term it must refuse (found in the baseline, in an earlier version of this kernel or in the checker)'),
                                   ('refuses', 'trusted code refused, aborted on or mishandled a VALID input (fails closed: no false theorem, but a true one is lost)'),
                                   ('wrong', 'untrusted code produced a wrong, ill-typed or misleading result, or failed on valid input (the kernel check refuses it, or a test caught it first)'),
                                   ('resource', 'unbounded or excessive time or memory'), ('data', 'the benchmark or the training data was wrong, duplicated or misleading'),
                                   ('missing', 'a capability the platform needs did not exist yet (a design decision, not a defect of a released rule)'),
                                   ('obligation', 'a soundness-critical rule that had to be designed or ported correctly (nothing wrong was released)'),
                                   ('process', 'tooling, estimates, documents, the test harness itself'), ('measurement', 'not a defect: a measurement recorded with the findings')])
TRUSTED_LAYERS = ('kernel', 'checker')
STATES = ('fixed', 'by construction', 'documented', 'mitigated', 'partial', 'open')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ reading evidence
def read_jsonl(path):
    """the records (dicts) of a JSON-lines file, in order; a line that does not parse (a write that was cut short) is skipped; [] when the file does not exist"""
    out = []
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: r = json.loads(line)
                except ValueError: continue
                if isinstance(r, dict): out.append(r)
    return out


def read_json(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError): return default


def sha256_of(path, n=16):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    except OSError: return None
    return h.hexdigest()[:n]


def pct(a, b, digits=1):
    """a/b in percent, truncated and never rounded up: 1646/1647 prints 99.9, so 100.0 means every one; '--' when b is 0"""
    if not b: return '--'
    scale = 10 ** digits
    return ('%%.%df' % digits) % (math.floor(100.0 * a / b * scale + 1e-9) / scale)


def num(n): return '{:,}'.format(int(n))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the mutation catalogue
def _mutate():
    """the mutation driver as a module (it lives outside the import path: mutants/)"""
    d = os.path.join(ROOT, 'mutants')
    if d not in sys.path: sys.path.insert(0, d)
    import mutate
    return mutate


def component_of(path):
    """the part of the system a source file (relative to the root) belongs to"""
    p = path.replace(os.sep, '/')
    if p.startswith('src/c++/'): return 'kernel (C++)'
    parts = p.split('/')
    if len(parts) >= 3 and parts[:2] == ['src', 'python']: return COMPONENT.get(parts[2], parts[2])
    return p


def catalogue_info():
    """what the summary needs to know of the catalogue: {names, file, component, digest, equivalent (name -> reason), gates (the mutation suite)}"""
    M = _mutate(); cache = {}
    info = {'names': list(M.MUTANTS), 'file': {}, 'component': {}, 'digest': {}, 'equivalent': dict(M.KNOWN_EQUIVALENT), 'gates': [os.path.basename(c[0]) for c in M.REG.GATE_CMDS]}
    for n in info['names']:
        base = M.mutant_parts(M.MUTANTS[n])[2]
        if base not in cache: cache[base] = component_of(M.locate(base))
        info['file'][n] = base; info['component'][n] = cache[base]; info['digest'][n] = M.mutant_digest(n)
    return info


def tree_digest():
    return _mutate().tree_digest()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ summaries
def gates_summary(recs, tree=None):
    """{'total', 'passed', 'failed': [gate], 'seconds', 'groups': {group: {gates, passed, seconds}}, 'stale': n} -- the LAST record of every (gate, arguments) counts, one made for another tree does not"""
    last, stale = collections.OrderedDict(), 0
    for r in recs:
        if tree and r.get('tree') != tree: stale += 1; continue
        last[(r.get('gate'), tuple(r.get('args') or ()))] = r
    groups = collections.OrderedDict(); failed = []
    for (g, _), r in last.items():
        d = groups.setdefault(r.get('group') or '?', {'gates': 0, 'passed': 0, 'seconds': 0.0}); d['gates'] += 1; d['passed'] += bool(r.get('ok')); d['seconds'] += float(r.get('seconds') or 0)
        if not r.get('ok'): failed.append(g)
    order = {k: i for i, k in enumerate(GROUP_LABEL)}
    groups = collections.OrderedDict(sorted(groups.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])))
    return {'total': len(last), 'passed': len(last) - len(failed), 'failed': failed, 'seconds': sum(d['seconds'] for d in groups.values()), 'groups': groups, 'stale': stale,
            'records': [dict(gate=g, args=list(a), group=r.get('group'), ok=bool(r.get('ok')), seconds=r.get('seconds'), verdict=r.get('verdict')) for (g, a), r in last.items()]}


def mutants_summary(recs, cat, tree=None):
    """what the mutation records say, against the catalogue `cat` (catalogue_info()).  A mutant counts only through a record made for THIS mutant (its digest) on THIS tree; the status is derived afresh from what caught it
    ('caught' = some gate failed; else 'equivalent' if the catalogue documents why no input can tell it from the original, else 'gap')"""
    last = {}
    for r in recs: last[r.get('name')] = r
    cur, stale = {}, []
    for n in cat['names']:
        r = last.get(n)
        if r is None: continue
        if cat.get('digest') and r.get('digest') != cat['digest'].get(n): stale.append(n); continue
        if tree and r.get('tree') != tree: stale.append(n); continue
        if r.get('status') not in ('caught', 'equivalent', 'gap', 'build_failed'): stale.append(n); continue
        cur[n] = r
    status = {}
    for n, r in cur.items():
        if r.get('caught_by'): status[n] = 'caught'
        elif r.get('status') == 'build_failed': status[n] = 'build_failed'
        elif n in cat['equivalent']: status[n] = 'equivalent'
        else: status[n] = 'gap'
    cnt = collections.Counter(status.values())
    kinds = collections.Counter(cur[n]['caught_by'][0].get('kind', '?') for n in cur if status[n] == 'caught')
    denom = cnt['caught'] + cnt['gap'] + cnt['build_failed']
    full = bool(cur) and all(r.get('full') for r in cur.values())
    depth = None
    if full:
        c = collections.Counter(min(len(r['caught_by']), 3) for n, r in cur.items() if status[n] == 'caught'); depth = {'1': c[1], '2': c[2], '3+': c[3]}
    comps = collections.OrderedDict()
    order = {k: i for i, k in enumerate(COMPONENT_ORDER)}
    for n in cat['names']:
        d = comps.setdefault(cat['component'][n], {'catalogue': 0, 'run': 0, 'caught': 0, 'equivalent': 0, 'gap': 0, 'build_failed': 0}); d['catalogue'] += 1
        if n in cur: d['run'] += 1; d[status[n]] += 1
    comps = collections.OrderedDict(sorted(comps.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])))
    series = collections.OrderedDict()
    for n in cat['names']:
        s = re.match(r'[A-Za-z]+', n); s = s.group(0) if s else ''
        d = series.setdefault(s, {'catalogue': 0, 'run': 0, 'caught': 0, 'equivalent': 0, 'gap': 0, 'build_failed': 0}); d['catalogue'] += 1
        if n in cur: d['run'] += 1; d[status[n]] += 1
    gate_sets = sorted({tuple(r.get('gates') or ()) for r in cur.values()})
    return {'catalogue': len(cat['names']), 'run': len(cur), 'missing': [n for n in cat['names'] if n not in cur and n not in stale], 'stale': stale,
            'caught': cnt['caught'], 'equivalent': cnt['equivalent'], 'gap': cnt['gap'], 'build_failed': cnt['build_failed'], 'kinds': dict(kinds),
            'kill_rate': (cnt['caught'] / denom) if denom else None, 'depth': depth, 'components': comps, 'series': series,
            'gaps': [n for n in cur if status[n] in ('gap', 'build_failed')], 'survivors': [(n, cat['equivalent'][n]) for n in cat['names'] if status.get(n) == 'equivalent'],
            'gate_sets': [list(g) for g in gate_sets], 'suite': list(cat.get('gates') or ()), 'seconds': sum(float(r.get('seconds') or 0) for r in cur.values()),
            'rows': [dict(name=n, component=cat['component'][n], file=cat['file'][n], status=status.get(n, 'not run'), kind=(cur[n]['caught_by'][0].get('kind') if n in cur and status[n] == 'caught' else ''),
                          gates=';'.join(c.get('gate', '') for c in cur[n].get('caught_by', [])) if n in cur else '') for n in cat['names']]}


def fuzz_summary(recs, tree=None):
    """per campaign class (world, noarith): campaigns, seeds, statements (true / false), the outcomes -- and the sum.  A campaign made for another tree does not count."""
    by = collections.OrderedDict(); stale = 0; seen = set()
    for r in recs:
        if tree and r.get('tree') != tree: stale += 1; continue
        key = (r.get('world'), bool(r.get('noarith')))
        sig = (key, r.get('seed'), r.get('n'), r.get('depth'), r.get('probes'), r.get('seconds_per_statement'))
        if sig in seen: continue                                                # (a campaign run twice with the same settings is one campaign)
        seen.add(sig); c = r.get('counts') or {}
        d = by.setdefault(key, {'campaigns': 0, 'seeds': set(), 'statements': 0, 'true': 0, 'false': 0, 'PROVED': 0, 'FAILED': 0, 'REJECTED': 0, 'UNSOUND': 0, 'ERROR': 0, 'seconds': 0.0})
        d['campaigns'] += 1; d['seeds'].add(r.get('seed')); d['statements'] += int(r.get('statements') or sum(c.values())); d['true'] += int(r.get('true') or 0); d['false'] += int(r.get('false') or 0)
        d['seconds'] += float(r.get('seconds') or 0)
        for k in ('PROVED', 'FAILED', 'REJECTED', 'UNSOUND', 'ERROR'): d[k] += int(c.get(k) or 0)
    order = {k: i for i, k in enumerate(FUZZ_CAMPAIGNS)}
    classes = collections.OrderedDict(sorted(by.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])))
    tot = collections.Counter()
    for d in classes.values():
        for k, v in d.items():
            if k != 'seeds': tot[k] += v
    return {'classes': classes, 'total': dict(tot), 'stale': stale, 'campaigns': sum(d['campaigns'] for d in classes.values()),
            'reports': [dict(world=r.get('world'), noarith=bool(r.get('noarith')), seed=r.get('seed'), report=x) for r in recs if not (tree and r.get('tree') != tree) for x in (r.get('reports') or [])][:50]}


def differential_summary(d, tree=None):
    """the S8 report (S8_JSON) as {modes, totals, problems}; None when there is none or it was made for another tree"""
    if not d: return None
    if tree and d.get('tree') != tree: return {'stale': True}
    modes = d.get('modes') or {}
    tot = collections.Counter()
    for m in modes.values():
        for k in ('tactic_proofs', 'mutants', 'agree', 'both_accept', 'both_reject', 'kernel_accepts_checker_rejects', 'checker_accepts_kernel_rejects', 'valid_rejected'): tot[k] += int(m.get(k) or 0)
    problems = []
    if not d.get('passed', False): problems.append('the S8 gate failed: ' + '; '.join(d.get('failures') or ['(no reason recorded)']))
    if tot['kernel_accepts_checker_rejects'] or tot['checker_accepts_kernel_rejects']: problems.append('kernel and checker disagree on %d mutated proofs' % (tot['kernel_accepts_checker_rejects'] + tot['checker_accepts_kernel_rejects']))
    if tot['valid_rejected']: problems.append('%d valid proofs rejected' % tot['valid_rejected'])
    if d.get('wrong_proofs_accepted'): problems.append('the checker accepts %d wrong proofs' % d['wrong_proofs_accepted'])
    if d.get('definitions_rejected'): problems.append('the checker rejects %d definitions of the library' % d['definitions_rejected'])
    if any(not m.get('definitions_agree', True) for m in modes.values()): problems.append('kernel and checker disagree about which definitions are valid')
    if d.get('guard_cases_agree') != d.get('guard_cases'): problems.append('the structural-recursion guard corpus: %s of %s agree' % (d.get('guard_cases_agree'), d.get('guard_cases')))
    return {'stale': False, 'modes': modes, 'totals': dict(tot), 'problems': problems, 'definitions': d.get('definitions'), 'wrong_proofs': d.get('wrong_proofs'), 'wrong_proofs_accepted': d.get('wrong_proofs_accepted'),
            'guard_cases': d.get('guard_cases'), 'guard_cases_agree': d.get('guard_cases_agree'), 'per_proof': d.get('argv_mutants_per_proof'), 'seconds': d.get('seconds')}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the findings
def read_findings(path=FINDINGS_PATH):
    """[{id, title, status, state, evidence}] -- the rows of the table of docs/FINDINGS.md (a row begins with `| F<number>`)"""
    rows = []
    if not os.path.exists(path): return rows
    for line in open(path):
        if not re.match(r'\|\s*F\d+[a-z]?\s*\|', line): continue
        cells = [c.strip() for c in line.rstrip('\n').strip().strip('|').split(' | ')]
        if len(cells) < 3: continue
        rows.append({'id': cells[0], 'title': cells[1], 'status': cells[2], 'state': state_of(cells[2]), 'evidence': cells[3] if len(cells) > 3 else ''})
    return rows


def state_of(status):
    """a finding's state from the first words of its status cell: open (also 'mitigated; root cause open'), partial, documented, fixed, by construction (also 'by design'), mitigated; 'other' when it is none of them"""
    s = status.replace('**', '').strip().lower()
    if s.startswith('open'): return 'open'
    if s.startswith('mitigated'): return 'open' if 'open' in s else 'mitigated'
    if s.startswith('partial'): return 'partial'
    if s.startswith('documented'): return 'documented'
    if s.startswith('by construction') or s.startswith('by design'): return 'by construction'
    if s.startswith('fixed') or s.startswith('hybrid is stricter'): return 'fixed'
    return 'other'


def read_tags(path=TAGS_PATH):
    """{id: {layers: [..], effect, note}} from the tab-separated tags file; '#' starts a comment line.  Returns (tags, problems)"""
    tags, problems = collections.OrderedDict(), []
    if not os.path.exists(path): return tags, ['no tags file: ' + path]
    for k, line in enumerate(open(path), 1):
        if not line.strip() or line.lstrip().startswith('#'): continue
        c = line.rstrip('\n').split('\t')
        if len(c) < 3: problems.append('line %d: %d fields, 3 or 4 wanted' % (k, len(c))); continue
        fid, layers, effect = c[0].strip(), [x.strip() for x in c[1].split(',') if x.strip()], c[2].strip()
        if fid in tags: problems.append('line %d: %s is tagged twice' % (k, fid)); continue
        bad = [x for x in layers if x not in LAYERS]
        if bad or not layers: problems.append('line %d: %s: layer %s' % (k, fid, ', '.join(bad) or '(none)'))
        if effect not in EFFECTS: problems.append('line %d: %s: effect %r' % (k, fid, effect))
        tags[fid] = {'layers': layers, 'effect': effect, 'note': c[3].strip() if len(c) > 3 else ''}
    return tags, problems


def findings_summary(rows, tags):
    """the findings with their tags: counts by layer and effect, by state, the trusted-code ones, the open ones; `problems` lists a finding without a tag, a tag without a finding and a state that is none of STATES"""
    problems = []
    ids = [r['id'] for r in rows]
    for i in ids:
        if i not in tags: problems.append('%s has no tag in docs/findings_tags.tsv' % i)
    for i in tags:
        if i not in ids: problems.append('docs/findings_tags.tsv tags %s, which is not in docs/FINDINGS.md' % i)
    for r in rows:
        if r['state'] not in STATES: problems.append('%s: the status %r is none of the known states' % (r['id'], r['status'][:40]))
    matrix = collections.OrderedDict((l, collections.OrderedDict((e, 0) for e in EFFECTS)) for l in LAYERS)
    by_state, trusted, accepts, opened = collections.Counter(), [], [], []
    for r in rows:
        t = tags.get(r['id'])
        if not t: continue
        by_state[r['state']] += 1
        for l in t['layers']:
            if l in matrix and t['effect'] in matrix[l]: matrix[l][t['effect']] += 1
        if any(l in TRUSTED_LAYERS for l in t['layers']): trusted.append(r['id'])
        if t['effect'] == 'accepts': accepts.append(r['id'])
        if r['state'] in ('open', 'partial', 'mitigated') and t['effect'] != 'measurement': opened.append(r['id'])          # (a measurement is not a defect: F70 is a research question, not a risk)
    num_of = lambda i: re.match(r'F(\d+)', i).group(1)
    numeric = {num_of(i) for i in ids}
    return {'rows': len(rows), 'findings': len(numeric), 'matrix': matrix, 'by_state': dict(by_state), 'trusted': trusted, 'accepts': accepts, 'open': opened, 'problems': problems,
            'trusted_n': len({num_of(i) for i in trusted}), 'accepts_n': len({num_of(i) for i in accepts}), 'open_n': len({num_of(i) for i in opened}),
            'table': [dict(id=r['id'], title=r['title'], status=r['status'], state=r['state'], layers=','.join(tags[r['id']]['layers']) if r['id'] in tags else '', effect=tags[r['id']]['effect'] if r['id'] in tags else '') for r in rows]}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the inventory
def short_title(t, n=100):
    """a finding's title for a table: backticks and bold marks removed, parentheticals dropped, cut at the first colon or semicolon, else at a word boundary, with an ellipsis"""
    t = t.replace('`', '').replace('**', '').strip(); prev = None
    while prev != t: prev = t; t = re.sub(r'\s\([^()]*\)', '', t)
    for sep in (': ', '; '):
        i = t.find(sep)
        if 12 <= i <= n: return t[:i]
    if len(t) <= n: return t
    return t[:n].rsplit(' ', 1)[0] + ' ...'


def lines_of(paths):
    return RP.count_lines(paths)


def inventory():
    """what the trusted code is, counted from the files: {components: [{component, role, files, lines, sha256}], facts, tests}"""
    hyb = os.path.join(ROOT, 'src', 'c++', 'hybrid'); ref = os.path.join(ROOT, 'src', 'c++', 'reference')
    py = lambda sub: sorted(RP.python_files(os.path.join('src', 'python', sub)))
    comps = [('kernel (C++)', 'trusted: decides every verdict (the typing rules, conversion, admission, the work budget)', [os.path.join(hyb, 'itp_kernel.hpp')]),
             ('kernel bindings (C++)', 'trusted: the interface a verdict passes through', [os.path.join(hyb, 'itp_bindings.cpp')]),
             ('independent checker', 'second opinion: written from the specification, shares no code with the kernel', py('checker')),
             ('tactic layer and decision procedures', 'untrusted: every term it builds is checked by the kernel', py('authoring')),
             ('libraries', 'untrusted: definitions and lemmas, each admitted through the checked path', py('library')),
             ("learner's environment", 'untrusted: proposes actions; a finished proof is checked by the kernel', py('environment')),
             ('problem bank, learner, generator, experiments', 'untrusted: data and training; nothing here is on the path of a verdict', py('problems') + py('learner') + py('synth') + py('experiments')),
             ('reference kernel (frozen baseline)', 'only a test oracle: the lockstep tests compare the new kernel against it', sorted(glob.glob(os.path.join(ref, '*')))),
             ('tests', 'gates, fuzzers, harnesses', sorted(RP.python_files('tests')))]
    out = []
    for name, role, files in comps:
        files = [f for f in files if os.path.isfile(f)]
        h = hashlib.sha256()
        for f in files:
            with open(f, 'rb') as fh: h.update(os.path.relpath(f, ROOT).encode() + b'\0' + hashlib.sha256(fh.read()).digest())
        out.append({'component': name, 'role': role, 'files': len(files), 'lines': lines_of(files), 'sha256': h.hexdigest()[:16]})
    return {'components': out, 'facts': RP.facts()}


def stamp(extra=None):
    """who ran what on which code: the commit, whether the tree is clean, the digest of the tree the records must carry, of the kernel's source and of the compiled kernel, the machine"""
    facts = RP.facts(); build = sorted(glob.glob(os.path.join(itp_paths.BUILD, 'itp_env_lnl*')))
    src = [os.path.join(ROOT, 'src', 'c++', 'hybrid', n) for n in ('itp_kernel.hpp', 'itp_bindings.cpp')]
    try: cores = len(os.sched_getaffinity(0))
    except AttributeError: cores = os.cpu_count() or 1
    try: gxx = subprocess.run(['g++', '--version'], capture_output=True, text=True, timeout=20).stdout.split('\n')[0]
    except Exception: gxx = ''
    st = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'commit': facts.get('commit'), 'dirty': facts.get('dirty'), 'tree': tree_digest(), 'kernel_source_sha256': sha256_of(src[0], 16),
          'bindings_source_sha256': sha256_of(src[1], 16), 'kernel_binary': os.path.basename(build[0]) if build else None, 'kernel_binary_sha256': sha256_of(build[0], 16) if build else None,
          'build_older_than_source': bool(build and os.path.getmtime(build[0]) < max(os.path.getmtime(p) for p in src if os.path.exists(p))),
          'python': platform.python_version(), 'platform': platform.platform(), 'machine': platform.machine(), 'host': platform.node(), 'cores': cores, 'compiler': gxx}
    st.update(extra or {})
    return st


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the verdict
def assess(S):
    """(status, failures, incomplete): status is FAILED when the evidence contradicts a claim (a gate fails, a mutant survives without a reason, a false statement is proved, the checker disagrees), INCOMPLETE when
    evidence is missing, partial or stale, CLEAN when every claim is backed by evidence for this very code"""
    failures, incomplete = [], []
    g, m, f, d, fi = S.get('gates'), S.get('mutants'), S.get('fuzz'), S.get('differential'), S.get('findings')
    if g is None or not g['total']: incomplete.append('no gate results for this tree')
    else:
        if g['failed']: failures.append('gates failed: ' + ', '.join(g['failed']))
        if g['stale']: incomplete.append('%d gate record(s) made for another tree are ignored' % g['stale'])
        want = S.get('gates_expected')
        if want and g['total'] < want: incomplete.append('%d of %d gates have a result' % (g['total'], want))
    if m is None or not m['run']: incomplete.append('no mutation results for this tree')
    else:
        if m['gap'] or m['build_failed']: failures.append('%d mutant(s) survive without a documented reason and %d did not build: %s' % (m['gap'], m['build_failed'], '; '.join(n[:40] for n in m['gaps'][:5])))
        if m['run'] < m['catalogue']: incomplete.append('%d of %d mutants have a result for this tree (%d stale, %d never run)' % (m['run'], m['catalogue'], len(m['stale']), len(m['missing'])))
        if len(m['gate_sets']) > 1: incomplete.append('mutants were run against %d different sets of gates' % len(m['gate_sets']))
        elif m['gate_sets'] and m['suite'] and m['gate_sets'][0] != m['suite']: incomplete.append('mutants were run against %d of the %d gates of the mutation suite' % (len(m['gate_sets'][0]), len(m['suite'])))
    if f is None or not f['campaigns']: incomplete.append('no fuzz campaigns for this tree')
    else:
        t = f['total']
        if t.get('UNSOUND'): failures.append('%d FALSE statement(s) were proved: a bug of the kernel' % t['UNSOUND'])
        if t.get('REJECTED'): failures.append('%d proof(s) of the tactic layer were refused by the kernel' % t['REJECTED'])
        if t.get('ERROR'): failures.append('%d error(s) inside the environment' % t['ERROR'])
        if f['stale']: incomplete.append('%d fuzz campaign(s) made for another tree are ignored' % f['stale'])
        for key, c in f['classes'].items():
            if c['true'] and not c['PROVED']: failures.append('fuzz %s proved none of its %d true statements: the campaign cannot show anything' % (FUZZ_LABEL.get(key, key), c['true']))
    if d is None: incomplete.append('no comparison with the independent checker')
    elif d.get('stale'): incomplete.append('the comparison with the independent checker was made for another tree')
    else: failures += d['problems']
    if fi is None: incomplete.append('findings not classified')
    elif fi['problems']: failures += fi['problems'][:5]
    st = S.get('stamp') or {}
    if st.get('tree_stale'): incomplete.append('the evidence was collected on tree %s, the code is now %s' % (st['tree_stale'], st.get('tree')))
    if st.get('dirty'): incomplete.append('the code (src/, tests/, mutants/, the manifest of the dataset) has uncommitted changes: the commit does not identify it (the tree digest does)')
    if st.get('build_older_than_source'): incomplete.append('the compiled kernel is older than its source: rebuild it (make kernel) and collect again')
    if S.get('tree_changed'): incomplete.append('the tree changed while the evidence was collected (%s, then %s)' % S['tree_changed'])
    return ('FAILED' if failures else 'INCOMPLETE' if incomplete else 'CLEAN'), failures, incomplete


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ tables
def build_tables(S):
    """the tables of the folder (exp_tables.Table) from the summaries S = {gates, mutants, fuzz, differential, inventory, findings}"""
    T = collections.OrderedDict(); g, m, f, d, inv, fi = (S.get(k) for k in ('gates', 'mutants', 'fuzz', 'differential', 'inventory', 'findings'))
    if g and g['total']:
        rows = [[GROUP_LABEL.get(k, k), v['gates'], v['passed'], '%.1f' % (v['seconds'] / 60.0)] for k, v in g['groups'].items()]
        rows.append(['all gates', g['total'], g['passed'], '%.1f' % (g['seconds'] / 60.0)])
        T['sound_gates'] = TB.Table('sound_gates', 'Gates: every gate is a program that exits non-zero when one of its checks fails.', ['part of the system', 'gates', 'passed', 'minutes'], rows, align='lrrr',
                                    notes=['Minutes are summed over gates; gates run two at a time.'])
    if m and m['run']:
        rows = []
        for k, v in m['components'].items():
            rows.append([k, num(v['catalogue']), num(v['run']), num(v['caught']), num(v['equivalent']), num(v['gap'] + v['build_failed']), pct(v['caught'], v['caught'] + v['gap'] + v['build_failed'])])
        c, r, ca, eq, ga = m['catalogue'], m['run'], m['caught'], m['equivalent'], m['gap'] + m['build_failed']
        rows.append(['all', num(c), num(r), num(ca), num(eq), num(ga), pct(ca, ca + ga)])
        kinds = m['kinds']; kn = ', '.join('%s %s' % (k, num(kinds[k])) for k in ('assertion', 'crash', 'timeout') if kinds.get(k))
        notes = ['A mutant re-breaks ONE rule of the system in an isolated copy of the tree; it is caught when some gate then fails (%s). Equivalent mutants (documented: no input can tell them from the original) are left out of the rate.' % (kn or 'no gate failed')]
        if m['depth']: notes.append('Run against every gate: %s mutants fail exactly one gate, %s two, %s three or more.' % (num(m['depth']['1']), num(m['depth']['2']), num(m['depth']['3+'])))
        T['sound_mutants'] = TB.Table('sound_mutants', 'Mutation testing: seeded faults caught by the gates, by the part of the system that was mutated.', ['mutated component', 'mutants', 'run', 'caught', 'equivalent', 'survive', 'caught (%)'],
                                      rows, align='lrrrrrr', notes=notes)
        T['sound_survivors'] = TB.Table('sound_survivors', 'Mutants that no gate catches, and why no test could.', ['mutant', 'reason'], [[n, why] for n, why in m['survivors']] or [['(none)', '']], align='ll')
    if f and f['campaigns']:
        rows = []
        for key, v in f['classes'].items():
            rows.append([FUZZ_LABEL.get(key, '%s%s' % (key[0], ', no arithmetic' if key[1] else '')), v['campaigns'], num(v['statements']), num(v['true']), num(v['false']), num(v['PROVED']), num(v['UNSOUND']), num(v['REJECTED']), num(v['ERROR'])])
        t = f['total']; rows.append(['all', f['campaigns'], num(t['statements']), num(t['true']), num(t['false']), num(t['PROVED']), num(t['UNSOUND']), num(t['REJECTED']), num(t['ERROR'])])
        T['sound_fuzz'] = TB.Table('sound_fuzz', 'Falsification campaigns: random statements, true and false, searched for proofs through the learner\'s environment.',
                                   ['statements about', 'campaigns', 'statements', 'true', 'false', 'proved (true)', 'false proved', 'refused by kernel', 'errors'], rows, align='lrrrrrrrr',
                                   notes=['A false statement that is proved would be a bug of the kernel; a proof that the kernel then refuses is a bug of the tactic layer or the unifier. A campaign in which nothing true is proved would prove nothing: the column "proved (true)" is there to show the search is able to prove.'])
    if d and not d.get('stale'):
        rows = []
        for k, v in d['modes'].items():
            rows.append([k, num(v.get('tactic_proofs', 0)), num(v.get('globals', 0)), num(v.get('mutants', 0)), num(v.get('both_accept', 0)), num(v.get('both_reject', 0)),
                         num(v.get('kernel_accepts_checker_rejects', 0)), num(v.get('checker_accepts_kernel_rejects', 0))])
        t = d['totals']; rows.append(['all', num(t['tactic_proofs']), '', num(t['mutants']), num(t['both_accept']), num(t['both_reject']), num(t['kernel_accepts_checker_rejects']), num(t['checker_accepts_kernel_rejects'])])
        T['sound_differential'] = TB.Table('sound_differential', 'The kernel against the independently written checker: mutated proofs given to both.',
                                           ['mode', 'tactic proofs', 'globals', 'mutated proofs', 'both accept', 'both reject', 'only kernel accepts', 'only checker accepts'], rows, align='lrrrrrrr',
                                           notes=['Every definition of the library (%s) is re-checked by the checker; it accepts all but those the kernel also refuses, and %d of %d wrong proofs are accepted. Structural-recursion guard corpus: %s of %s cases agree.' %
                                                  (d.get('definitions'), d.get('wrong_proofs_accepted') or 0, d.get('wrong_proofs') or 0, d.get('guard_cases_agree'), d.get('guard_cases'))])
    if inv:
        mut = collections.Counter()
        if m:
            for k, v in m['components'].items(): mut[k] = v['catalogue']
        mut_of = {'kernel (C++)': mut['kernel (C++)'], 'independent checker': mut['independent checker'], 'tactic layer and decision procedures': mut['tactic layer and decision procedures'],
                  'libraries': mut['libraries'], "learner's environment": mut["learner's environment"]}
        rows = [[c['component'], c['role'], c['files'], num(c['lines']), num(mut_of[c['component']]) if mut_of.get(c['component']) else '--'] for c in inv['components']]
        T['sound_trusted'] = TB.Table('sound_trusted', 'What has to be trusted: the code that decides a verdict, against the code that only proposes.', ['component', 'role', 'files', 'lines', 'mutants'], rows, align='llrrr',
                                      notes=['Lines are counted from the files. Mutants of the kernel bindings are listed with the kernel.'])
    if fi:
        cols = [e for e in EFFECTS if sum(fi['matrix'][l][e] for l in LAYERS)]
        rows = [[l] + [fi['matrix'][l][e] or '' for e in cols] + [sum(fi['matrix'][l].values())] for l in LAYERS]
        rows.append(['all (a finding in two layers counts once)'] + [sum(1 for r in fi['table'] if r['effect'] == e) for e in cols] + [fi['rows']])
        T['sound_findings'] = TB.Table('sound_findings', 'The %d findings recorded while building the system (%d rows: F2 has two parts), by layer and effect.' % (fi['findings'], fi['rows']), ['layer'] + cols + ['total'], rows,
                                       align='l' + 'r' * (len(cols) + 1), notes=['accepts: trusted code accepted what it must refuse. refuses: trusted code refused valid input (fails closed). wrong: untrusted code produced a wrong result, which the kernel check refuses. See README.md for the definitions.'])
        orows = [[r['id'], short_title(r['title']), r['layers'], r['effect'], r['state']] for r in fi['table'] if r['id'] in fi['open']]
        T['sound_open'] = TB.Table('sound_open', 'Findings that are not closed.', ['id', 'finding', 'layer', 'effect', 'state'], orows or [['(none)', '', '', '', '']], align='lllll')
    rows = []
    st = S.get('stamp') or {}
    if g and g['total']: rows.append(['gates', '%d of %d pass' % (g['passed'], g['total'])])
    if m and m['run']: rows += [['mutants caught', '%s of %s (%s%% of those that can be told apart from the original)' % (num(m['caught']), num(m['catalogue']), pct(m['caught'], m['caught'] + m['gap'] + m['build_failed']))],
                                ['mutants that survive', '%s documented as equivalent, %s unexplained' % (num(m['equivalent']), num(m['gap'] + m['build_failed']))]]
    if f and f['campaigns']: rows.append(['false statements searched for a proof', '%s (%s statements in all), %s proved' % (num(f['total']['false']), num(f['total']['statements']), num(f['total']['UNSOUND']))])
    if d and not d.get('stale'): rows.append(['mutated proofs given to kernel and checker', '%s, %s disagreements' % (num(d['totals']['mutants']), num(d['totals']['kernel_accepts_checker_rejects'] + d['totals']['checker_accepts_kernel_rejects']))])
    if fi: rows.append(['findings', '%d (%d concern the trusted code, %d of them accepted what they must refuse), %d not closed' % (fi['findings'], fi['trusted_n'], fi['accepts_n'], fi['open_n'])])
    if rows: T['sound_summary'] = TB.Table('sound_summary', 'The soundness evidence in one table.', ['evidence', 'result'], rows, align='ll',
                                           notes=['Code: commit %s, tree %s.' % ((st.get('commit') or '?')[:10], st.get('tree') or '?')])
    return T


def numbers(S):
    """the numbers the text may quote, by name (numbers.tex defines \\numSound... for each)"""
    N = collections.OrderedDict(); g, m, f, d, fi, inv = (S.get(k) for k in ('gates', 'mutants', 'fuzz', 'differential', 'findings', 'inventory'))
    if g and g['total']: N.update(sound_gates=g['total'], sound_gates_passed=g['passed'])
    if m and m['run']:
        N.update(sound_mutants=m['catalogue'], sound_mutants_run=m['run'], sound_mutants_caught=m['caught'], sound_mutants_equivalent=m['equivalent'], sound_mutants_unexplained=m['gap'] + m['build_failed'],
                 sound_mutants_caught_pct=pct(m['caught'], m['caught'] + m['gap'] + m['build_failed']))
        for k in ('assertion', 'crash', 'timeout'): N['sound_mutants_' + k] = m['kinds'].get(k, 0)
    if f and f['campaigns']:
        t = f['total']; N.update(sound_fuzz_campaigns=f['campaigns'], sound_fuzz_statements=t['statements'], sound_fuzz_true=t['true'], sound_fuzz_false=t['false'], sound_fuzz_proved=t['PROVED'],
                                 sound_fuzz_unsound=t['UNSOUND'], sound_fuzz_rejected=t['REJECTED'], sound_fuzz_errors=t['ERROR'])
    if d and not d.get('stale'):
        t = d['totals']; N.update(sound_diff_proofs=t['tactic_proofs'], sound_diff_mutants=t['mutants'], sound_diff_disagreements=t['kernel_accepts_checker_rejects'] + t['checker_accepts_kernel_rejects'],
                                  sound_diff_definitions=d.get('definitions') or 0)
    if fi: N.update(sound_findings=fi['findings'], sound_findings_trusted=fi['trusted_n'], sound_findings_accepts=fi['accepts_n'], sound_findings_open=fi['open_n'])
    if inv:
        by = {c['component']: c for c in inv['components']}
        for key, name in (('kernel_lines', 'kernel (C++)'), ('bindings_lines', 'kernel bindings (C++)'), ('checker_lines', 'independent checker')):
            if name in by: N['sound_' + key] = by[name]['lines']
    return N


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the folder
README_DEFS = """## Definitions

* **gate**: a test program (`tests/**/s*_tests.py`) that exits non-zero when one of its checks fails; {gates} of them (`tests/gate_registry.py`). The {suite} gates of the mutation suite are the ones mutants are run against.
* **mutant**: a copy of the code with ONE rule re-broken (`mutants/catalogue.py`: the text to replace, the text to put there, the file; {mutants} of them, hand-written, each pinned to the rule it breaks). **caught**: some gate fails on the copy,
  by an *assertion* (a check of the gate failed), a *crash* (the gate raised instead of checking) or a *timeout* (it did not stop). **equivalent**: no gate fails, and `KNOWN_EQUIVALENT` says why no input can tell the mutant from the original.
  **survives / gap**: no gate fails and there is no such reason -- a hole in the tests.
* **fuzz campaign**: `tests/fuzz/gym_fuzz.py` makes statements about the naturals or the integers, decides by evaluation which are true, and runs the environment's proof search on each. `PROVED` = a true statement was proved (the search is able to prove);
  `UNSOUND` = a statement that is false at some point was proved (a bug of the kernel); `REJECTED` = the search finished a proof that the kernel then refused (a bug of the tactic layer or the unifier); `ERROR` = an exception out of the engine.
* **differential testing**: the independently written checker (`src/python/checker/indep_checker.py`, written from the specification with named variables, sharing no code with the kernel) gets the same terms as the kernel; the verdicts must agree.
* **layer** of a finding: kernel, checker, tactic, engine, data, pipeline, tooling (docs/findings_tags.tsv has the definitions). **effect**: accepts, refuses, wrong, resource, data, missing, obligation, process, measurement.
  `accepts` is the class that matters for soundness: trusted code accepted what it must refuse. The tags are the author's classification of each row of docs/FINDINGS.md, whose text says what happened; the rule for each class is in the tags file.
"""

README_LIMITS = """## What this evidence does and does not show

* Mutation testing measures how well the gates detect the faults of the catalogue. The catalogue is written by hand, by the author of the code: a high rate shows the tests are sharp against the faults someone thought of, not against every fault.
* The fuzz campaigns find bugs by falsification: no false statement was proved in the statements generated. This is evidence, not a proof; the kernel is small (see the trusted table) so that it can be audited.
* The checker is independent in code and in representation, not in authorship; a misreading of the specification common to both would not show.
* The statement of a theorem is outside the verified chain: the kernel shows that a term has the type that was written down, not that the type says what the problem says (see the statement audit).
* A record counts only for the tree it ran on (the digest in `stamp.json`); anything else is listed as stale and left out.
"""


def render(S, out):
    """write the derived files of the folder from the summaries S; returns the list of paths.  S = {gates, mutants, fuzz, differential, inventory, findings, stamp, ...}"""
    for sub in ('tables', 'stats'): shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
    for fn in ('README.md', 'report.md', 'soundness.json', 'numbers.json', 'numbers.tex'):
        if os.path.exists(os.path.join(out, fn)): os.remove(os.path.join(out, fn))
    os.makedirs(os.path.join(out, 'tables')); os.makedirs(os.path.join(out, 'stats')); files = []
    status, failures, incomplete = assess(S); T = build_tables(S); N = numbers(S)
    for key, t in T.items():
        for ext, text in (('md', t.md()), ('tex', t.tex()), ('csv', t.csv())):
            p = os.path.join(out, 'tables', '%s.%s' % (key, ext))
            with open(p, 'w') as fh: fh.write(text)
            files.append(p)
    g, m, f, fi = S.get('gates'), S.get('mutants'), S.get('fuzz'), S.get('findings')
    if m and m['run']:
        files.append(EB.write_csv(os.path.join(out, 'stats', 'mutants.csv'), ['name', 'component', 'file', 'status', 'first_catch', 'caught_by'], [[r['name'], r['component'], r['file'], r['status'], r['kind'], r['gates']] for r in m['rows']]))
        files.append(EB.write_csv(os.path.join(out, 'stats', 'mutants_by_series.csv'), ['series', 'catalogue', 'run', 'caught', 'equivalent', 'gap', 'build_failed'],
                                  [[k, v['catalogue'], v['run'], v['caught'], v['equivalent'], v['gap'], v['build_failed']] for k, v in m['series'].items()]))
    if g and g['total']: files.append(EB.write_csv(os.path.join(out, 'stats', 'gates.csv'), ['gate', 'args', 'group', 'ok', 'seconds', 'verdict'], [[r['gate'], ' '.join(r['args']), r['group'], r['ok'], r['seconds'], r['verdict']] for r in g['records']]))
    if f and f['campaigns']:
        files.append(EB.write_csv(os.path.join(out, 'stats', 'fuzz.csv'), ['world', 'noarith', 'campaigns', 'seeds', 'statements', 'true', 'false', 'PROVED', 'FAILED', 'REJECTED', 'UNSOUND', 'ERROR'],
                                  [[k[0], k[1], v['campaigns'], sorted(x for x in v['seeds'] if x is not None), v['statements'], v['true'], v['false'], v['PROVED'], v['FAILED'], v['REJECTED'], v['UNSOUND'], v['ERROR']] for k, v in f['classes'].items()]))
    if fi: files.append(EB.write_csv(os.path.join(out, 'stats', 'findings.csv'), ['id', 'finding', 'status', 'state', 'layers', 'effect'], [[r['id'], r['title'], r['status'], r['state'], r['layers'], r['effect']] for r in fi['table']]))
    summary = {'status': status, 'failures': failures, 'incomplete': incomplete, 'numbers': N, 'stamp': S.get('stamp'),
               'gates': {k: v for k, v in (g or {}).items() if k != 'records'}, 'mutants': {k: v for k, v in (m or {}).items() if k not in ('rows', 'series', 'missing', 'stale')} | ({'missing': len((m or {}).get('missing', [])), 'stale': len((m or {}).get('stale', []))} if m else {}),
               'fuzz': {'total': (f or {}).get('total'), 'campaigns': (f or {}).get('campaigns'), 'classes': {('%s%s' % (k[0], '/noarith' if k[1] else '')): dict(v, seeds=sorted(x for x in v['seeds'] if x is not None)) for k, v in ((f or {}).get('classes') or {}).items()}},
               'differential': S.get('differential'), 'findings': {k: v for k, v in (fi or {}).items() if k not in ('table', 'matrix')} | ({'matrix': fi['matrix']} if fi else {}), 'inventory': S.get('inventory')}
    p = os.path.join(out, 'soundness.json')
    with open(p, 'w') as fh: json.dump(summary, fh, indent=1, sort_keys=False, default=str)
    files.append(p)
    ER.write_numbers(N, out); files += [os.path.join(out, 'numbers.json'), os.path.join(out, 'numbers.tex')]
    st = S.get('stamp') or {}
    L = ['# Soundness evidence', '', 'Status: **%s**' % status, '']
    if failures: L += ['**Contradicted claims** (fix these before quoting anything):', ''] + ['* ' + x for x in failures] + ['']
    if incomplete: L += ['**Incomplete or stale evidence:**', ''] + ['* ' + x for x in incomplete] + ['']
    L += ['Code: commit `%s`%s, tree digest `%s`, kernel source `%s`, compiled kernel `%s`. Collected %s on %s (%s cores, %s, %s).' % ((st.get('commit') or '?')[:12], ' (uncommitted changes)' if st.get('dirty') else '', st.get('tree'),
          st.get('kernel_source_sha256'), st.get('kernel_binary_sha256'), st.get('time'), st.get('host'), st.get('cores'), st.get('python'), st.get('compiler')), '']
    L += ['Read `report.md` for the tables, `numbers.tex` for the macros, `soundness.json` for everything machine-readable; the raw records are in `evidence/`.', '', '## Reproduce', '',
          '    make soundness                     # the whole stage, resumable: the gates, every mutant, the fuzz campaigns, the checker comparison (many hours; see docs/EXPERIMENTS.md)',
          '    python3 src/python/experiments/exp_soundness.py report   # the tables again from the saved evidence', '    sha256sum -c MANIFEST.sha256     # in this folder: every file is the one that was written', '']
    counts = dict(gates=S.get('gates_expected') or (g or {}).get('total') or '?', suite=len((m or {}).get('suite') or ()) or '?', mutants=(m or {}).get('catalogue') or '?')
    L += README_DEFS.format(**counts).split('\n') + [''] + README_LIMITS.split('\n')
    L += ['', '## Files', '', '* `tables/` -- one md/tex/csv per table: ' + ', '.join('`%s`' % k for k in T), '* `stats/` -- one row per mutant, per series, per gate, per campaign class, per finding',
          '* `numbers.json`, `numbers.tex` -- the numbers by name (`\\numSoundMutantsCaught`, ...)', '* `soundness.json` -- the whole summary', '* `evidence/` -- the raw records (the only files that are not computed)', '* `MANIFEST.sha256` -- the SHA-256 of every other file']
    p = os.path.join(out, 'README.md')
    with open(p, 'w') as fh: fh.write('\n'.join(L) + '\n')
    files.append(p)
    R = ['# Soundness evidence -- report', '', 'Status: **%s**' % status, ''] + (['Contradicted claims:', ''] + ['* ' + x for x in failures] + [''] if failures else []) + (['Incomplete or stale:', ''] + ['* ' + x for x in incomplete] + [''] if incomplete else [])
    for key, t in T.items(): R += [t.md(), '']
    p = os.path.join(out, 'report.md')
    with open(p, 'w') as fh: fh.write('\n'.join(R))
    files.append(p)
    EB.manifest(out)
    return files


def collect(out, tree=None, cat=None, tags_path=TAGS_PATH, findings_path=FINDINGS_PATH):
    """read the evidence of `out` and summarise it (against `tree`, default the digest of the code as it is now)"""
    ev = os.path.join(out, 'evidence')
    tree = tree or tree_digest(); cat = cat or catalogue_info()
    st = read_json(os.path.join(ev, 'stamp.json'), {})
    rows = read_findings(findings_path); tags, tag_problems = read_tags(tags_path)
    fs = findings_summary(rows, tags); fs['problems'] = tag_problems + fs['problems']
    S = {'gates': gates_summary(read_jsonl(os.path.join(ev, 'gates.jsonl')), tree), 'mutants': mutants_summary(read_jsonl(os.path.join(ev, 'mutants.jsonl')), cat, tree),
         'fuzz': fuzz_summary(read_jsonl(os.path.join(ev, 'fuzz.jsonl')), tree), 'differential': differential_summary(read_json(os.path.join(ev, 'differential.json')), tree),
         'inventory': read_json(os.path.join(ev, 'inventory.json')), 'findings': fs, 'stamp': dict(st, tree=tree) if st else stamp({'collected': False})}
    try:
        import gate_registry as REG
        S['gates_expected'] = len({os.path.basename(c[0]) + ' ' + ' '.join(c[1:]) for grp in ('GATE_CMDS', 'BANK_GATE_CMDS', 'ML_GATE_CMDS', 'REPO_GATE_CMDS', 'SYNTH_GATE_CMDS') for c in getattr(REG, grp, ())})
    except Exception: pass
    if st and st.get('tree_before') and st.get('tree_after') and st['tree_before'] != st['tree_after']: S['tree_changed'] = (st['tree_before'], st['tree_after'])
    if st and st.get('tree') and st['tree'] != tree: S['stamp']['tree_stale'] = st['tree']
    return S


def report(out):
    S = collect(out); files = render(S, out)
    return S, files


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ running the stages
def parse_seeds(text):
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if not part: continue
        if '-' in part: lo, hi = part.split('-'); out += list(range(int(lo), int(hi) + 1))
        else: out.append(int(part))
    return sorted(set(out))


def run_cmd(argv, log, env=None, cwd=None):
    """run a command, append its output to the log file; returns the exit status"""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'a') as fh:
        fh.write('$ ' + ' '.join(argv) + '\n'); fh.flush()
        r = subprocess.run(argv, cwd=cwd or ROOT, env=dict(os.environ, **(env or {})), stdout=fh, stderr=subprocess.STDOUT)
        fh.write('[exit %d]\n' % r.returncode)
    return r.returncode


def append_jsonl(path, recs):
    with open(path, 'a') as fh:
        for r in recs: fh.write(json.dumps(r, sort_keys=True) + '\n')


def stage_gates(ev, a, tree, log):
    import gate_registry as REG
    everything = list(REG.GATE_CMDS) + list(REG.BANK_GATE_CMDS) + list(REG.ML_GATE_CMDS) + list(REG.REPO_GATE_CMDS) + list(REG.SYNTH_GATE_CMDS)
    path = os.path.join(ev, 'gates.jsonl'); old = [] if a.redo else [r for r in read_jsonl(path) if r.get('tree') == tree]
    done = {(r['gate'], tuple(r.get('args') or ())) for r in old if r.get('ok')}
    todo = []
    for c in everything:
        key = (os.path.basename(c[0]), tuple(c[1:]))
        if key not in done and os.path.basename(c[0]) not in todo: todo.append(os.path.basename(c[0]))
    if a.gates: todo = [n for n in todo if n in a.gates]
    keep = [r for r in old if (r['gate'], tuple(r.get('args') or ())) in done]
    tmp = path + '.new'
    if os.path.exists(tmp): os.remove(tmp)
    rc = 0
    if todo: rc = run_cmd([sys.executable, os.path.join(ROOT, 'tests', 'run_gates.py'), '-j', str(a.jobs), '--json', tmp] + todo, log)
    new = read_jsonl(tmp)
    for r in new: r['tree'] = tree
    with open(path, 'w') as fh:
        for r in keep + new: fh.write(json.dumps(r, sort_keys=True) + '\n')
    if os.path.exists(tmp): os.remove(tmp)
    return {'argv': 'tests/run_gates.py -j %d --json <file> %s' % (a.jobs, ' '.join(todo) if len(todo) < len(everything) else '(every gate)'), 'exit': rc, 'ran': len(new), 'reused': len(keep)}


def stage_mutants(ev, a, tree, log):
    path = os.path.join(ev, 'mutants.jsonl'); env = {}
    if a.mut_gates: env['MUT_ONLY'] = a.mut_gates
    if a.mut_full: env['MUT_FULL'] = '1'
    argv = [sys.executable, os.path.join(ROOT, 'mutants', 'mutate.py'), '--jobs', str(a.jobs), '--json', path] + (['--redo'] if a.redo else []) + list(a.mutants or [])
    rc = run_cmd(argv, log, env=env)
    return {'argv': ' '.join(['mutants/mutate.py'] + argv[2:]), 'env': env, 'exit': rc}


def fuzz_key(r): return (r.get('world'), bool(r.get('noarith')), r.get('seed'), r.get('n'), r.get('depth'), r.get('probes'), r.get('seconds_per_statement'))


def stage_fuzz(ev, a, tree, log):
    path = os.path.join(ev, 'fuzz.jsonl'); old = [] if a.redo else [r for r in read_jsonl(path) if r.get('tree') == tree]
    have = {fuzz_key(r) for r in old}; jobs = []
    for seed in parse_seeds(a.fuzz_seeds):
        for world, noarith in FUZZ_CAMPAIGNS:
            key = (world, noarith, seed, a.fuzz_n, FUZZ_DEFAULTS['depth'], FUZZ_DEFAULTS['probes'], float(a.fuzz_seconds))
            if key not in have: jobs.append((key, seed, world, noarith))
    def one(job):
        key, seed, world, noarith = job; tmp = path + '.%s.%s.%d.tmp' % (world, 'na' if noarith else 'ar', seed)
        if os.path.exists(tmp): os.remove(tmp)
        argv = [sys.executable, os.path.join(ROOT, 'tests', 'fuzz', 'gym_fuzz.py'), '--seed', str(seed), '--n', str(a.fuzz_n), '--world', world, '--seconds', str(a.fuzz_seconds), '--json', tmp] + (['--noarith'] if noarith else [])
        rc = run_cmd(argv, os.path.join(os.path.dirname(log), 'fuzz', '%s_%s_%d.log' % (world, 'noarith' if noarith else 'arith', seed)))
        recs = read_jsonl(tmp)
        if os.path.exists(tmp): os.remove(tmp)
        for r in recs: r['tree'] = tree; r['exit'] = rc
        return recs
    new = []
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        for recs in pool.map(one, jobs): new += recs
    keep = old
    with open(path, 'w') as fh:
        for r in keep + new: fh.write(json.dumps(r, sort_keys=True) + '\n')
    return {'argv': 'tests/fuzz/gym_fuzz.py --seed {%s} --world {nat,int} [--noarith] --n %d --seconds %s' % (a.fuzz_seeds, a.fuzz_n, a.fuzz_seconds), 'ran': len(new), 'reused': len(keep)}


def stage_differential(ev, a, tree, log):
    path = os.path.join(ev, 'differential.json'); old = read_json(path)
    if old and not a.redo and old.get('tree') == tree and old.get('passed') and old.get('argv_mutants_per_proof') == a.diff_per_proof: return {'reused': True}
    tmp = path + '.tmp'
    rc = run_cmd([sys.executable, os.path.join(ROOT, 'tests', 'kernel', 's8_indep_tests.py'), str(a.diff_per_proof)], log, env={'S8_JSON': tmp})
    d = read_json(tmp, {})
    if os.path.exists(tmp): os.remove(tmp)
    d.update(tree=tree, exit=rc, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with open(path, 'w') as fh: json.dump(d, fh, indent=1, sort_keys=True)
    return {'argv': 'tests/kernel/s8_indep_tests.py %d' % a.diff_per_proof, 'exit': rc}


def stage_inventory(ev, a, tree, log):
    inv = inventory(); inv['tree'] = tree
    with open(os.path.join(ev, 'inventory.json'), 'w') as fh: json.dump(inv, fh, indent=1, sort_keys=True)
    return {}


def stage_findings(ev, a, tree, log):
    return {}                                                           # (the findings are read from docs/ when the report is made)


STAGE_FN = {'gates': stage_gates, 'mutants': stage_mutants, 'fuzz': stage_fuzz, 'differential': stage_differential, 'inventory': stage_inventory, 'findings': stage_findings}


def run(a):
    out = os.path.abspath(a.out); ev = os.path.join(out, 'evidence'); os.makedirs(os.path.join(ev, 'logs'), exist_ok=True)
    stages = [s for s in STAGES if s in a.stages] if a.stages else list(STAGES)
    bad = [s for s in (a.stages or []) if s not in STAGES]
    if bad: print('unknown stage(s): %s (stages: %s)' % (', '.join(bad), ', '.join(STAGES))); return 2
    tree = tree_digest(); st0 = read_json(os.path.join(ev, 'stamp.json'), {})
    st = stamp({'tree_before': tree, 'stages': stages, 'commands': {}})
    if st0.get('tree') == tree: st['commands'] = st0.get('commands') or {}; st['started'] = st0.get('started') or st['time']
    else: st['started'] = st['time']
    for s in stages:
        t0 = time.time(); print('== %s' % s, flush=True)
        st['commands'][s] = dict(STAGE_FN[s](ev, a, tree, os.path.join(ev, 'logs', s + '.log')), seconds=round(time.time() - t0, 1))
        print('   done in %.0f s' % (time.time() - t0), flush=True)
    st['tree_after'] = tree_digest(); st['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    if st['tree_after'] != tree: st['tree'] = tree
    with open(os.path.join(ev, 'stamp.json'), 'w') as fh: json.dump(st, fh, indent=1, sort_keys=True)
    if a.no_report: return 0
    S, files = report(out)
    status, failures, incomplete = assess(S)
    print('%s: %s' % (out, status)); [print('   FAILED: ' + x) for x in failures]; [print('   incomplete: ' + x) for x in incomplete]
    return {'CLEAN': 0, 'FAILED': 1, 'INCOMPLETE': 2}[status]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--verify']:
        bad = EB.verify(argv[1]) if len(argv) > 1 else ['--verify needs a folder']
        for b in bad[:50]: print('  ' + b)
        print('%s: %s' % (argv[1] if len(argv) > 1 else '', 'ok (every file is the one that was written)' if not bad else '%d problem(s)' % len(bad)))
        return 1 if bad else 0
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run the stages and write the folder'); rp = sub.add_parser('report', help='write the tables and the README from the saved evidence')
    for p in (r, rp): p.add_argument('--out', default=DEFAULT_OUT)
    r.add_argument('--stages', type=lambda t: [x for x in t.split(',') if x], default=None, help='comma-separated, of: ' + ', '.join(STAGES))
    r.add_argument('--gates', type=lambda t: [x for x in t.split(',') if x], default=None, help='run only these gates (file names; a quick check: the folder then says it is incomplete)')
    r.add_argument('--jobs', type=int, default=2); r.add_argument('--redo', action='store_true', help='run again what has a result already')
    r.add_argument('--mutants', nargs='*', default=None, help='only the mutants whose name starts with one of these (default: all)')
    r.add_argument('--mut-gates', default=None, help='MUT_ONLY: run the mutants against these gates only (a quick check; the folder then says it is incomplete)'); r.add_argument('--mut-full', action='store_true', help='run EVERY gate against each mutant (how many gates catch it)')
    r.add_argument('--fuzz-seeds', default=FUZZ_DEFAULTS['seeds']); r.add_argument('--fuzz-n', type=int, default=FUZZ_DEFAULTS['n']); r.add_argument('--fuzz-seconds', type=float, default=FUZZ_DEFAULTS['seconds'])
    r.add_argument('--diff-per-proof', type=int, default=DIFF_DEFAULT, help='mutated proofs per tactic proof in the comparison with the independent checker'); r.add_argument('--no-report', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'run': return run(a)
    S, files = report(os.path.abspath(a.out)); status, failures, incomplete = assess(S)
    print('%s: %s (%d files)' % (a.out, status, len(files))); [print('   FAILED: ' + x) for x in failures]; [print('   incomplete: ' + x) for x in incomplete]
    return {'CLEAN': 0, 'FAILED': 1, 'INCOMPLETE': 2}[status]


if __name__ == '__main__':
    sys.exit(main())
