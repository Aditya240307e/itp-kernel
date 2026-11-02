#!/usr/bin/env python3
"""check_numbers.py -- do the numbers, tables and figures that a LaTeX source cites exist in the results folders, are the folders what they were when they were made, and do they agree with each other and with the code?

    python3 tools/check_numbers.py paper.tex --results results/full --results results/soundness --results results/audit
    python3 tools/check_numbers.py paper.tex --strict               # before a submission: warnings fail too
    python3 tools/check_numbers.py paper.tex --literals --unused    # numbers typed by hand that a macro already holds; macros nothing cites
    python3 tools/check_numbers.py --list sound                     # the macros whose name matches, with their values and folders
    python3 tools/check_numbers.py --tree                           # the digest of the code the soundness and audit evidence is bound to

A paper typed by hand takes its numbers from the results folders (`results/<profile>`, `results/soundness`, `results/audit`; docs/PAPER_PLAN.md says which). This tool reads the LaTeX source(s) (a file, or a directory: every
.tex below it) and reports

  ERROR    a number macro (`\\N{Name}` or `\\numName`) defined in no folder, a table (`\\inputtable{key}`, `\\input{.../tables/key}`) or figure (`\\genfigure{name}`, `\\includegraphics{.../figures/name}`) that no folder holds;
           a folder that is not what its manifest says (a file changed after it was made, or added); a soundness or audit folder whose status is FAILED; a macro defined differently in two folders; a folder that does not exist
  WARNING  a soundness or audit folder that is INCOMPLETE, or made while the code had uncommitted changes; a folder made for (its runs made with) other code than the code here, or for other code than another folder (the digest of the
           code, not the commit: documents, the paper and the tools are not code); warnings the report lists itself (missing rows, mixed code versions); a fact that two folders count twice and count differently (the kernel's lines, the number of problems, ...); `\\todo` left in the text
  INFO     (with --literals) a number typed in the text that equals the value of a macro -- it should be the macro, so that it follows the runs; (with --unused) macros nothing cites

Exit status: 0 when there is no error (and, with --strict, no warning), 1 when there is an error, 2 when there are warnings and --strict was given.  With no --results the folders results/full, results/soundness and
results/audit are used when they exist.  Nothing is written.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import argparse, collections, csv, difflib, json, os, re, sys

ROOT = itp_paths.ROOT
DEFAULT_FOLDERS = ('results/full', 'results/soundness', 'results/audit')

# the same fact counted by two folders (the report counts the repository and the bank it was run on; the soundness folder and the audit count them again): they must agree
INVARIANTS = (('RepoKernelLines', 'SoundKernelLines'), ('RepoBindingsLines', 'SoundBindingsLines'), ('RepoCheckerLines', 'SoundCheckerLines'), ('RepoMutants', 'SoundMutants'), ('RepoGates', 'SoundGates'),
              ('BankProblems', 'AuditProblems'), ('BankProved', 'AuditProved'), ('BankStatementOnly', 'AuditStatementOnly'))

MACRO_DEF = re.compile(r'^\\newcommand\{\\(num[A-Z][A-Za-z]*)\}\{(.*)\}\s*$')
OWN_DEF = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\{\\(num[A-Z][A-Za-z]*)\}|\\def\\(num[A-Z][A-Za-z]*)')
USE_N = re.compile(r'\\N\{([A-Za-z0-9]+)\}')
USE_NUM = re.compile(r'\\(num[A-Z][A-Za-z]*)')
USE_TABLE = re.compile(r'\\inputtable\{([^}]*)\}')
USE_FIG = re.compile(r'\\genfigure\{([^}]*)\}')
USE_INPUT = re.compile(r'\\input\{([^}]*)\}')
USE_GRAPHICS = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}')
TODO = re.compile(r'\\todo\s*\{')
ERROR, WARNING, INFO = 'ERROR', 'WARNING', 'INFO'


def tree_digest():
    """the digest of the code (mutants/mutate.py tree_digest: every file below src/, tests/ and mutants/ but caches, built files and .md documents, and the manifest of the dataset)"""
    d = os.path.join(ROOT, 'mutants')
    if d not in sys.path: sys.path.insert(0, d)
    import mutate
    return mutate.tree_digest()


def read_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f: return json.load(f)
    except (OSError, ValueError): return default


def read_macros(folder):
    """{name: value} from the folder's numbers.tex (\\newcommand{\\numName}{value}); None when there is no such file"""
    path = os.path.join(folder, 'numbers.tex')
    if not os.path.isfile(path): return None
    out = collections.OrderedDict()
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = MACRO_DEF.match(line.rstrip('\n'))
            if m: out[m.group(1)] = m.group(2)
    return out


def report_warnings(folder):
    """the bullets of the 'Warnings' section of a report.md ([] when there are none or no such file)"""
    try:
        with open(os.path.join(folder, 'report.md'), encoding='utf-8') as f: text = f.read()
    except OSError: return []
    out = []; on = False
    for line in text.split('\n'):
        if line.startswith('## '):
            on = line.strip() == '## Warnings'; continue
        if on and line.startswith('* '): out.append(line[2:].strip())
    return out


def run_trees(folder):
    """the digests of the code that the runs of a results folder recorded (provenance/runs.csv, column `tree`), distinct and sorted; [] when there is no such file or no run recorded one"""
    try:
        with open(os.path.join(folder, 'provenance', 'runs.csv'), newline='', encoding='utf-8') as f: rows = list(csv.DictReader(f))
    except (OSError, csv.Error): return []
    return sorted({r['tree'] for r in rows if r.get('tree')})


def folder_info(path):
    """what a results folder says about itself: its kind (report, soundness, audit), macros, status, the tree it was made for (the digests of the code its runs recorded, for a report), the manifest problems"""
    info = {'path': path, 'exists': os.path.isdir(path), 'kind': 'report', 'macros': None, 'status': None, 'tree': None, 'commit': None, 'dirty': None, 'manifest': None, 'warnings': [], 'profile': None, 'run_trees': []}
    if not info['exists']: return info
    info['macros'] = read_macros(path)
    for kind in ('soundness', 'audit'):
        d = read_json(os.path.join(path, kind + '.json'))
        if d is not None:
            info['kind'] = kind; info['status'] = d.get('status'); st = d.get('stamp') or {}
            info['tree'] = st.get('tree'); info['commit'] = st.get('commit'); info['dirty'] = st.get('dirty')
    if info['kind'] == 'report':
        info['warnings'] = report_warnings(path); info['run_trees'] = run_trees(path)
        prof = read_json(os.path.join(path, 'provenance', 'PROFILE.json'))
        if isinstance(prof, dict): info['profile'] = prof.get('name')
    import exp_bundle as EB
    info['manifest'] = EB.verify(path)
    return info


def strip_comment(line):
    """a line of LaTeX without its comment (a % that is not escaped)"""
    return re.sub(r'(?<!\\)%.*$', '', line)


def read_sources(paths):
    """[(file, [lines without comments])] for the .tex files given (a directory: every .tex below it)"""
    out = []
    for p in paths:
        files = [p]
        if os.path.isdir(p): files = sorted(os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs if f.endswith('.tex'))
        for fp in files:
            with open(fp, encoding='utf-8', errors='replace') as f: out.append((fp, [strip_comment(l.rstrip('\n')) for l in f]))
    return out


def citations(lines):
    """({macro: [line numbers]}, {table key: [...]}, {figure name: [...]}, own, todos) for the lines of one source; `own` are the macros the source defines itself"""
    macros, tables, figures, todos = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), []
    own = set()
    for i, line in enumerate(lines, 1):
        for m in OWN_DEF.finditer(line): own.add(m.group(1) or m.group(2))
        for m in USE_N.finditer(line): macros['num' + m.group(1)].append(i)
        for m in USE_NUM.finditer(line): macros[m.group(1)].append(i)
        for m in USE_TABLE.finditer(line):
            if '#' not in m.group(1): tables[m.group(1).strip()].append(i)
        for m in USE_FIG.finditer(line):
            if '#' not in m.group(1): figures[m.group(1).strip()].append(i)
        for m in USE_INPUT.finditer(line):
            path = m.group(1).strip()
            if '#' in path or 'tables/' not in path.replace('\\', '/'): continue
            key = os.path.basename(path.replace('\\', '/'))
            tables[key[:-4] if key.endswith('.tex') else key].append(i)
        for m in USE_GRAPHICS.finditer(line):
            path = m.group(1).strip().replace('\\', '/')
            if '#' in path or 'figures/' not in path: continue
            figures[os.path.splitext(os.path.basename(path))[0]].append(i)
        if TODO.search(line): todos.append(i)
    return macros, tables, figures, own, todos


def has_table(folder, key): return os.path.isfile(os.path.join(folder, 'tables', key + '.tex'))
def has_figure(folder, name): return any(os.path.isfile(os.path.join(folder, 'figures', name + ext)) for ext in ('.pdf', '.png'))


NUMBER = re.compile(r'(?<![\w.\\{])\d+(?:\.\d+)?(?![\w}])')


def plain_number(value):
    """the value of a macro when it is a plain number worth looking for in the text (three digits or more, or a decimal with two), else None"""
    v = value.strip()
    if not re.fullmatch(r'\d+(?:\.\d+)?', v): return None
    digits = v.replace('.', '')
    return v if len(digits) >= 3 or (('.' in v) and len(v.split('.')[1]) >= 2) else None


def typed_numbers(line):
    """the numbers typed in a line of text, thousands separators removed (1647, 1\\,647, 1{,}647 and 1,647 are all 1647)"""
    s = re.sub(r'(?<=\d)(?:\\,|\{,\}|\\;|~|\u2009|,)(?=\d{3}(?!\d))', '', line)
    return NUMBER.findall(s)


def analyse(sources, folders, tree=None, check_tree=True, literals=False, unused=False):
    """the findings, as {'problems': [(severity, where, message)], 'folders': [info], 'uses': (macros, tables, figures), 'macros': {name: (value, folder)}}.
    `sources` are the paths of .tex files or directories, `folders` the results folders; `tree` the digest of the code to compare the evidence with (None: compute it)"""
    P = []
    def add(sev, where, msg): P.append((sev, where, msg))
    infos = [folder_info(f) for f in folders]
    known = collections.OrderedDict()
    for inf in infos:
        if not inf['exists']: add(ERROR, inf['path'], 'the results folder does not exist'); continue
        if inf['macros'] is None: add(ERROR, inf['path'], 'there is no numbers.tex: this is not a results folder'); continue
        for pr in inf['manifest']: add(ERROR, inf['path'], 'the folder is not what its manifest says (%s): it was edited after it was made' % pr)
        if inf['status'] == 'FAILED': add(ERROR, inf['path'], 'the status of this %s folder is FAILED: its evidence contradicts a claim' % inf['kind'])
        elif inf['status'] == 'INCOMPLETE': add(WARNING, inf['path'], 'the status of this %s folder is INCOMPLETE: numbers cited from it describe partial evidence' % inf['kind'])
        elif inf['kind'] in ('soundness', 'audit') and inf['status'] != 'CLEAN': add(WARNING, inf['path'], 'the status of this %s folder is %r, not CLEAN' % (inf['kind'], inf['status']))
        if inf['dirty']: add(WARNING, inf['path'], 'this %s folder was made while the code had uncommitted changes: the commit does not identify the code' % inf['kind'])
        for w in inf['warnings']: add(WARNING, inf['path'], 'the report says: ' + w)
        for k, v in inf['macros'].items():
            if k in known and known[k][0] != v: add(ERROR, inf['path'], '\\%s is %s here and %s in %s' % (k, v, known[k][0], known[k][1]))
            else: known.setdefault(k, (v, inf['path']))
    if check_tree:
        current = tree or None
        bound = [inf for inf in infos if inf['exists'] and inf['kind'] in ('soundness', 'audit')]
        ran = [inf for inf in infos if inf['exists'] and inf['run_trees']]                                # (a results folder whose runs recorded the code they ran with)
        if (bound or ran) and current is None:
            try: current = tree_digest()
            except Exception as e: add(WARNING, '(code)', 'the digest of the code could not be computed (%s): the evidence cannot be compared with it' % e)
        for inf in bound:
            if not inf['tree']: add(WARNING, inf['path'], 'the folder does not record the tree it was made for')
            elif current and inf['tree'] != current: add(WARNING, inf['path'], 'made for the tree %s; the code here is %s: the evidence is about other code' % (inf['tree'], current))
        for inf in infos:
            if inf['exists'] and inf['kind'] == 'report' and inf['macros'] is not None and not inf['run_trees']:
                add(WARNING, inf['path'], 'the runs of this folder did not record the code they ran with (provenance/runs.csv has no tree): nothing says which code the numbers are about')
        for inf in ran:
            if current and inf['run_trees'] != [current]: add(WARNING, inf['path'], 'the runs of this folder were made with the code %s; the code here is %s: the numbers are about other code' % (', '.join(inf['run_trees']), current))
        trees = sorted({inf['tree'] for inf in bound if inf['tree']} | {t for inf in ran for t in inf['run_trees']})
        if len(trees) > 1: add(WARNING, '(folders)', 'the folders were made for different trees (%s): the evidence is about different code' % ', '.join(trees))
    for a, b in INVARIANTS:
        ka, kb = 'num' + a, 'num' + b
        if ka in known and kb in known and known[ka][0] != known[kb][0]:
            add(WARNING, '(folders)', '\\%s is %s in %s but \\%s is %s in %s: the same fact, counted twice, differs (are the folders from the same code and bank?)' % (ka, known[ka][0], known[ka][1], kb, known[kb][0], known[kb][1]))
    used = set(); n_uses = 0; n_tables = set(); n_figs = set()
    for sp in sources:
        if not os.path.exists(sp): add(ERROR, sp, 'there is no such file or directory')
    texts = read_sources([sp for sp in sources if os.path.exists(sp)])
    if not texts: add(ERROR, ', '.join(sources) or '(no source)', 'there is no LaTeX source to check')
    for fp, lines in texts:
        macros, tables, figures, own, todos = citations(lines)
        for name, at in sorted(macros.items()):
            n_uses += len(at)
            if name in own: continue
            used.add(name)
            if name not in known:
                near = difflib.get_close_matches(name, list(known), n=3, cutoff=0.75)
                add(ERROR, '%s:%d' % (fp, at[0]), '\\%s is defined in no results folder%s' % (name, (' (did you mean %s?)' % ', '.join('\\' + n for n in near)) if near else ''))
        for key, at in sorted(tables.items()):
            n_tables.add(key)
            if not any(inf['exists'] and has_table(inf['path'], key) for inf in infos): add(ERROR, '%s:%d' % (fp, at[0]), 'the table %r is in no results folder' % key)
        for name, at in sorted(figures.items()):
            n_figs.add(name)
            if not any(inf['exists'] and has_figure(inf['path'], name) for inf in infos): add(ERROR, '%s:%d' % (fp, at[0]), 'the figure %r is in no results folder' % name)
        if todos: add(WARNING, '%s:%d' % (fp, todos[0]), '%d \\todo left in the text (lines %s)' % (len(todos), ', '.join(str(i) for i in todos[:8]) + (' ...' if len(todos) > 8 else '')))
        if literals:
            by_value = collections.defaultdict(list)
            for k, (v, _) in known.items():
                n = plain_number(v)
                if n: by_value[n].append(k)
            for i, line in enumerate(lines, 1):
                if re.match(r'\s*\\(?:bibitem|usepackage|label|bibliography|includegraphics)', line): continue
                for n in typed_numbers(line):
                    if n in by_value: add(INFO, '%s:%d' % (fp, i), 'the number %s typed here is the value of %s: use the macro' % (n, ', '.join('\\' + k for k in by_value[n])))
    if unused:
        left = [k for k in known if k not in used]
        if left: add(INFO, '(folders)', '%d of %d macros are cited nowhere: %s' % (len(left), len(known), ', '.join('\\' + k for k in left[:30]) + (' ...' if len(left) > 30 else '')))
    return {'problems': P, 'folders': infos, 'uses': (n_uses, len(n_tables), len(n_figs)), 'macros': known, 'files': [fp for fp, _ in texts]}


def summary_line(inf):
    if not inf['exists']: return '%-22s (missing)' % inf['path']
    bits = [inf['kind']]
    if inf['kind'] != 'report': bits.append(inf['status'] or '?')
    elif inf['profile']: bits.append('profile ' + inf['profile'])
    if inf['tree']: bits.append('tree ' + inf['tree'])
    elif inf['commit']: bits.append('commit ' + inf['commit'][:10])
    bits.append('manifest ' + ('OK' if inf['manifest'] == [] else 'BAD' if inf['manifest'] else '?'))
    bits.append('%d numbers' % len(inf['macros'] or ()))
    return '%-22s %s' % (inf['path'], ', '.join(bits))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\n'.join(__doc__.split('\n')[1:]))
    ap.add_argument('sources', nargs='*', help='LaTeX files, or directories (every .tex below them)')
    ap.add_argument('--results', action='append', default=[], metavar='DIR', help='a results folder (repeatable; default: %s, those that exist)' % ', '.join(DEFAULT_FOLDERS))
    ap.add_argument('--strict', action='store_true', help='warnings fail too (exit status 2)')
    ap.add_argument('--literals', action='store_true', help='list numbers typed in the text that a macro already holds')
    ap.add_argument('--unused', action='store_true', help='list the macros that nothing cites')
    ap.add_argument('--list', nargs='?', const='', default=None, metavar='PATTERN', help='print the macros whose name matches PATTERN (a regular expression, case-insensitive) with value and folder, and stop')
    ap.add_argument('--tree', action='store_true', help='print the digest of the code and stop')
    ap.add_argument('--no-tree', action='store_true', help='do not compare the evidence (the soundness and audit folders, the runs of a results folder) with the code here (when the code is not the checkout it was made from)')
    a = ap.parse_args(argv)
    if a.tree: print(tree_digest()); return 0
    folders = a.results or [f for f in DEFAULT_FOLDERS if os.path.isdir(os.path.join(ROOT, f))]
    folders = [f if os.path.isabs(f) or os.path.exists(f) else os.path.join(ROOT, f) for f in folders]
    if not folders: print('no results folder: give --results DIR (default: %s)' % ', '.join(DEFAULT_FOLDERS), file=sys.stderr); return 1
    if a.list is not None:
        rx = re.compile(a.list, re.I); n = 0
        for f in folders:
            m = read_macros(f)
            if m is None: print('%s: no numbers.tex' % f, file=sys.stderr); continue
            for k, v in m.items():
                if rx.search(k): print('\\%s = %s    [%s]' % (k, v, f)); n += 1
        if not n: print('no macro matches %r' % a.list, file=sys.stderr)
        return 0
    if not a.sources: print('give the LaTeX source to check (or --list, --tree)', file=sys.stderr); return 1
    R = analyse(a.sources, folders, check_tree=not a.no_tree, literals=a.literals, unused=a.unused)
    for inf in R['folders']: print(summary_line(inf))
    nm, nt, nf = R['uses']
    print('%s: %d macro uses, %d tables, %d figures' % (', '.join(R['files']) or '(no source)', nm, nt, nf))
    order = {ERROR: 0, WARNING: 1, INFO: 2}
    for sev, where, msg in sorted(R['problems'], key=lambda p: order[p[0]]): print('%-8s %s  %s' % (sev, where, msg))
    errors = sum(1 for p in R['problems'] if p[0] == ERROR); warnings = sum(1 for p in R['problems'] if p[0] == WARNING)
    print('%d error%s, %d warning%s' % (errors, '' if errors == 1 else 's', warnings, '' if warnings == 1 else 's'))
    if errors: return 1
    return 2 if warnings and a.strict else 0


if __name__ == '__main__':
    sys.exit(main())
