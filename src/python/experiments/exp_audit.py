"""exp_audit.py -- the STATEMENT AUDIT of the problem bank, in one stamped folder.

    python3 src/python/experiments/exp_audit.py run [--out results/audit] [--only NAME ...] [--redo] [--no-report]
    python3 src/python/experiments/exp_audit.py report [--out results/audit] [--reviews docs/statement_reviews.tsv]      # tables, numbers, README, review sheet from the saved evidence; runs nothing
    python3 src/python/experiments/exp_audit.py import-reviews SHEET.csv [--reviewer NAME] [--role second] [--reviews docs/statement_reviews.tsv]     # a filled review sheet -> lines of the ledger
    python3 src/python/experiments/exp_audit.py --verify results/audit

A benchmark of olympiad problems is only as good as its statements: a proof of a statement that is not the problem proves nothing about the problem, and the kernel cannot tell (it checks proofs of a statement, not the statement).
The bank therefore checks every formal statement against an independent description of its meaning (semantic.py): the statement is read back from the kernel's term, evaluated on a box of points and compared with Python
predicates part by part (each hypothesis, the conclusion or both sides of an equivalence) and with the numbers inside its relation atoms, it must be non-vacuous, and every plausible one-slip mutant of it must differ from it
somewhere (or the spec says why it means the same).  This audit runs those checks for the whole bank in one place and adds what they cannot show:

  evidence      one record per problem (evidence/problems.jsonl): its provenance, the digest of the statement, what the spec checks, what the box found, the constructs that make the evidence weaker
  cautions      derived from a record: a contest problem whose wording was never compared with the published one, an existential quantifier (the box sees small witnesses only), a function quantifier, hypotheses
                whose numbers are not compared, slips that were declared equivalent, a box with few points that satisfy the hypotheses, ... (CAUTIONS)
  reviews       docs/statement_reviews.tsv: a SECOND READER's verdict on a statement, bound to the digest of the statement that was read, so that a statement that changed afterwards is no longer reviewed.  The checks
                above are written by the people who wrote the statements: a misreading of the English that both share is invisible to them, and only a second reader can see it.

Verdict: FAILED (exit 1) when a check fails or a second reader found a statement unfaithful, INCOMPLETE (exit 2) when records are missing or stale (made for another tree), CLEAN (exit 0) otherwise.  The share of statements that
a second reader confirmed is REPORTED (numbers.tex: \\numAuditConfirmed ...), never assumed: the paper says what the folder says.

Folder:
    README.md  report.md  audit.json  numbers.json  numbers.tex  MANIFEST.sha256  review_sheet.md  review_sheet.csv
    tables/<key>.{md,tex,csv}   audit_summary audit_origins audit_cautions audit_reviews audit_failures
    stats/*.csv                 problems.csv (one row per problem), cautions.csv, exemptions.csv, reviews.csv
    evidence/                   stamp.json problems.jsonl   (the raw records; the rest is computed from them, from the ledger and from the code)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, re, csv, json, time, shutil, hashlib, argparse, collections
import exp_tables as TB, exp_bundle as EB, exp_report as ER, exp_soundness as XS

ROOT = itp_paths.ROOT
DEFAULT_OUT = os.path.join(ROOT, 'results', 'audit')
REVIEWS_PATH = os.path.join(ROOT, 'docs', 'statement_reviews.tsv')
VERDICTS = collections.OrderedDict([('faithful', 'the formal statement says what the problem says'), ('adapted', 'it says what the problem says AFTER the adaptation the notes describe (an integer variant, one part of several)'),
                                    ('ambiguous', 'the English admits several readings; the formal statement picks one (the reader says which in the note)'), ('unfaithful', 'the formal statement does not say what the problem says')])
ROLES = ('second', 'author')                    # second: someone who did not write the statement; author: the person who did (recorded, but not an independent reading)
THIN = 5                                        # fewer points of the box satisfying the hypotheses than this: a thin box
ORIGINS = ('contest', 'adapted', 'style', 'folklore')
SPLITS = ('test', 'dev', 'train')
CAUTIONS = collections.OrderedDict([
    ('adapted', 'the problem was changed to fit the vocabulary (an integer variant of a statement about reals, one part of several): the notes say how'),
    ('source_unchecked', 'a contest problem whose published wording was not compared with a public source'),
    ('source_recalled', 'a contest problem whose source and wording are from memory'),
    ('truncated_arithmetic', 'the statement uses the predecessor on the naturals, which stops at zero'),
    ('window_existential', 'the statement has an existential quantifier: the box searches its witnesses in a finite window, so it only sees small witnesses'),
    ('function_quantifier', 'the statement quantifies over a function: it is tested on a family of test functions, not on all functions'),
    ('hypotheses_unchecked', 'a hypothesis contains an equation and the numbers inside it are not compared with the intended ones (only whether it holds, and an equation holds at few points)'),
    ('atoms_unchecked', 'the numbers inside the relation atoms of the conclusion are not compared with the intended ones (only whether the conclusion holds)'),
    ('exempted_slips', 'some slips of the statement are not tested because the spec declares them equivalent to it (each has its reason in stats/exemptions.csv)'),
    ('thin_box', 'fewer than %d points of the box satisfy the hypotheses and nothing but the truth value of the conclusion is compared there (no atoms, no body)' % THIN)])
REVIEW_STATES = ('confirmed', 'ambiguous', 'author only', 'stale', 'unreviewed', 'disputed', 'unfaithful')          # (in this order in the tables)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ reading a statement
def kids(a):
    """the sub-ASTs of an AST node (semantic.py)"""
    t = a[0]
    if t in ('prim', 'call'): return tuple(a[2])
    if t in ('all', 'ex', 'lam'): return (a[2],)
    if t in ('eq', 'imp', 'and', 'andd', 'or', 'iff'): return (a[1], a[2])
    if t == 'not': return (a[1],)
    return ()


def walk(a):
    """every node of an AST, the outermost first"""
    yield a
    for k in kids(a): yield from walk(k)


def positive_equation(a, neg=False):
    """does the AST contain an equation that is not under a negation?  (`q /= 0` holds almost everywhere, so a truth table sees a slip in it; `x^2 + y^2 = 3 z^2` holds at few points, and a slip that keeps its few
    solutions is invisible to a truth table: the numbers inside the atom must be compared)"""
    if a[0] == 'eq': return not neg
    return any(positive_equation(k, neg != (a[0] == 'not')) for k in kids(a))


def hypotheses_of(a):
    """the hypotheses of a statement  forall x1 .. xn, H1 -> .. -> Hm -> C  (the ASTs of H1 .. Hm)"""
    while a[0] == 'all': a = a[2]
    hyps = []
    while a[0] == 'imp': hyps.append(a[1]); a = a[2]
    return hyps


def features_of(a):
    """the constructs of a statement the audit cares about: 'pred' (the predecessor on the naturals: truncated at 0), 'exists', 'function' (a quantifier over functions: a call of a function variable is always under
    the quantifier that binds it, so the call itself says nothing more), 'sum', 'product', 'factorial', 'hyp_equation' (a hypothesis contains an equation that is not negated)"""
    f = set()
    for n in walk(a):
        t = n[0]
        if t == 'prim':
            if n[1] == 'pred': f.add('pred')
            elif n[1] == 'Z.sum': f.add('sum')
            elif n[1] == 'Z.prod': f.add('product')
            elif n[1] == 'Z.fact': f.add('factorial')
        elif t == 'ex': f.add('exists')
        if t in ('all', 'ex') and isinstance(n[1], tuple): f.add('function')
    if any(positive_equation(h) for h in hypotheses_of(a)): f.add('hyp_equation')
    return sorted(f)


_INFIX = {'Z.add': ('+', 7), 'N.add': ('+', 7), 'add': ('+', 7), 'Pos.add': ('+', 7), 'Z.sub': ('-', 7), 'Z.mul': ('*', 8), 'N.mul': ('*', 8), 'mul': ('*', 8), 'Pos.mul': ('*', 8)}
_POW = ('Z.pow', 'N.pow', 'pow', 'Pos.pow', 'Z.ppow')
_REL = {'Z.le': '<=', 'Z.lt': '<', 'N.le': '<=', 'N.lt': '<', 'Le': '<=', 'Lt': '<', 'Z.dvd': '|'}
_SAME = ('Z.of_N', 'N.to_nat', 'N.of_nat', 'Pos.to_nat', 'Npos', 'Z.zero', 'Z.one')              # (a coercion: written as its argument)
_BIN = {'iff': ('<->', 2, 2), 'imp': ('->', 3, 2), 'or': ('\\/', 3, 4), 'and': ('/\\', 4, 5), 'andd': ('/\\', 4, 5)}          # symbol, precedence needed on the left, on the right


def _kind(k): return 'Z -> Z' if k == ('fn', 1) else ('Z%s -> Z' % (''.join(', Z' for _ in range(k[1] - 1)))) if isinstance(k, tuple) else str(k)


def pretty(a, names=(), prec=0, tail=True):
    """a statement as a person reads it:  forall x0 : Z, 0 <= x0 -> gcd(21*x0 + 4, 14*x0 + 3) = 1.  `names`: the binders (None for the proof of a hypothesis); `prec`: the precedence the context needs; `tail`: nothing
    follows this expression inside its parentheses, so a quantifier needs none"""
    t = a[0]; wrap = lambda s, p: '(' + s + ')' if p < prec else s
    if t == 'int': return wrap(str(a[1]), 9 if a[1] < 0 else 11)
    if t == 'bv': return names[-1 - a[1]] or '_' if a[1] < len(names) else 'v%d' % a[1]
    if t in ('true', 'false'): return t
    if t == 'eq': return wrap('%s = %s' % (pretty(a[1], names, 7), pretty(a[2], names, 7)), 6)
    if t == 'not': return wrap('~' + pretty(a[1], names, 11), 5)
    if t in _BIN:
        sym, pl, pr = _BIN[t]; p = {'iff': 1, 'imp': 2, 'or': 3, 'and': 4, 'andd': 4}[t]
        right = pretty(a[2], names + (None,) if t in ('imp', 'andd') else names, pr, tail)
        return wrap('%s %s %s' % (pretty(a[1], names, pl, False), sym, right), p)
    if t in ('all', 'ex'):
        word = 'forall' if t == 'all' else 'exists'; kind = a[1]; vs = []; body = a
        while body[0] == t and body[1] == kind: vs.append('x%d' % (sum(1 for n in names if n is not None) + len(vs))); body = body[2]
        s = '%s %s : %s, %s' % (word, ' '.join(vs), _kind(kind), pretty(body, names + tuple(vs), 0, tail))
        return s if tail else '(' + s + ')'
    if t == 'lam': return 'fun i%d => %s' % (sum(1 for n in names if n and n.startswith('i')), pretty(a[2], names + ('i%d' % sum(1 for n in names if n and n.startswith('i')),), 0))
    if t == 'call': return '%s(%s)' % (pretty(('bv', a[1]), names), ', '.join(pretty(x, names) for x in a[2]))
    nm, args = a[1], a[2]
    if nm in _INFIX and len(args) == 2:
        sym, p = _INFIX[nm]; return wrap('%s %s %s' % (pretty(args[0], names, p), sym, pretty(args[1], names, p + 1)), p)
    if nm in _POW and len(args) == 2: return wrap('%s^%s' % (pretty(args[0], names, 11), pretty(args[1], names, 10)), 10)
    if nm in _REL and len(args) == 2: return wrap('%s %s %s' % (pretty(args[0], names, 7), _REL[nm], pretty(args[1], names, 7)), 6)
    if nm == 'Z.neg' and len(args) == 1: return wrap('-' + pretty(args[0], names, 10), 9)
    if nm == 'Z.sqr' and len(args) == 1: return wrap('%s^2' % pretty(args[0], names, 11), 10)
    if nm in ('N.succ', 'Succ', 'Pos.succ') and len(args) == 1: return wrap('%s + 1' % pretty(args[0], names, 7), 7)
    if nm == 'N.double' and len(args) == 1: return wrap('2*%s' % pretty(args[0], names, 9), 8)
    if nm in _SAME and len(args) == 1: return pretty(args[0], names, prec, tail)
    if nm in _SAME and not args: return {'Z.zero': '0', 'Z.one': '1'}.get(nm, nm)
    if nm == 'Z.is_gcd' and len(args) == 3: return wrap('gcd(%s, %s) = %s' % (pretty(args[0], names), pretty(args[1], names), pretty(args[2], names, 7)), 6)
    if nm == 'Z.fact' and len(args) == 1: return pretty(args[0], names, 11) + '!'
    if nm in ('Z.sum', 'Z.prod') and len(args) == 2 and args[0][0] == 'lam':
        i = 'i%d' % sum(1 for n in names if n and n.startswith('i'))
        return '%s(%s < %s, %s)' % ('sum' if nm == 'Z.sum' else 'prod', i, pretty(args[1], names), pretty(args[0][2], names + (i,)))
    return '%s(%s)' % (nm.replace('Z.', '').replace('N.', ''), ', '.join(pretty(x, names) for x in args)) if args else nm


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ a record per problem
def problem_record(bk, name, tree):
    """the evidence for one problem of a loaded bank (bank.LoadedBank): provenance, the digest of the statement, what the spec checks, what the box found, the constructs of the statement.  A check that raises is
    recorded as the problem's `error` (the audit fails closed: a problem it could not check is a failure), it does not stop the audit"""
    import bank as B
    pl = bk.placed[name]; pr = pl.problem; t0 = time.time()
    try:
        import bank_split as BS
        split, family = BS.split_of(name), BS.family(name)
    except Exception: split = family = None
    rec = {'name': name, 'tree': tree, 'statement_digest': B.statement_digest(bk.w, pl.stmt), 'text': pr.text, 'notes': pr.notes, 'source': pr.source, 'year': pr.year, 'origin': pr.origin, 'verified': pr.verified,
           'kind': pr.kind, 'level': pr.level, 'topics': list(pr.topics), 'proved': pl.proved, 'helpers': len(pr.helpers), 'split': split, 'family': family,
           'formal': None, 'in_fragment': False, 'features': [], 'spec': None, 'sem': None, 'sayable': False}
    try:
        ast = bk.statement_ast(name)
        if ast is not None: rec.update(formal=pretty(ast), in_fragment=True, features=features_of(ast))
        sp = pr.spec
        if sp is not None:
            rec['spec'] = {'params': [str(x) for x in sp.params], 'hyps': len(sp.hyps), 'atoms': None if sp.atoms is None else len(sp.atoms), 'hatoms': None if sp.hatoms is None else len(sp.hatoms),
                           'body': sp.body is not None, 'vacuous': bool(sp.vacuous), 'domains': sp.domains is not None, 'cap': sp.points, 'exemptions': sorted([k, v] for k, v in sp.skip_mutants.items())}
            rep = bk.check_semantics(name)
            rec['sem'] = {'ok': bool(rep.ok), 'errors': [e[:300] for e in rep.errors[:3]], 'points': rep.points, 'nonvacuous': rep.nonvacuous, 'mutants': rep.mutants, 'caught': rep.caught,
                          'equivalent': rep.equivalent, 'survivors': len(rep.survivors), 'shape_caught': rep.shape_caught, 'unevaluable': rep.unevaluable, 'counterexamples': rep.n_counter}
        rec['sayable'] = bk.statement_code(name) is not None
    except Exception as e:
        rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
    rec['seconds'] = round(time.time() - t0, 2)
    return rec


def failures_of(rec):
    """what is wrong with the statement according to the checks (a list of short texts; [] when the checks hold)"""
    out = []; sem = rec.get('sem')
    if rec.get('error'): return ['the audit crashed on this problem: ' + rec['error'][:200]]
    if rec.get('spec') is None: out.append('no spec: the statement is not compared with anything')
    elif not rec.get('in_fragment'): out.append('the statement is outside the fragment the evaluator reads')
    elif sem is None: out.append('the semantic check did not run')
    else:
        if sem['survivors']: out.append('%d slip(s) of the formalisation go unnoticed by the box' % sem['survivors'])
        if sem['errors'] and not sem['survivors']: out.append('semantics: ' + sem['errors'][0][:160])
        elif not sem['ok'] and not sem['errors']: out.append('the semantic check failed')
        if not rec.get('proved') and sem['counterexamples']: out.append('no proof, and the statement is FALSE at %d point(s) of the box' % sem['counterexamples'])
    if not rec.get('sayable'): out.append('the statement is not sayable in the typed term builder')
    return out


def cautions_of(rec):
    """the cautions that apply to a record (keys of CAUTIONS, in their order)"""
    sp, sem, ft = rec.get('spec') or {}, rec.get('sem') or {}, set(rec.get('features') or ()); c = set()
    if rec.get('origin') == 'adapted': c.add('adapted')
    if rec.get('origin') in ('contest', 'adapted') and rec.get('verified') == 'none': c.add('source_unchecked')
    if rec.get('verified') == 'recalled': c.add('source_recalled')
    if 'pred' in ft: c.add('truncated_arithmetic')
    if 'exists' in ft: c.add('window_existential')
    if 'function' in ft: c.add('function_quantifier')
    if 'hyp_equation' in ft and sp and sp.get('hatoms') is None: c.add('hypotheses_unchecked')
    if sp and sp.get('atoms') is None: c.add('atoms_unchecked')
    if sp and sp.get('exemptions'): c.add('exempted_slips')
    if sem and sp and not sp.get('vacuous') and sem.get('nonvacuous', 0) < THIN and sp.get('atoms') is None and not sp.get('body'): c.add('thin_box')
    return [k for k in CAUTIONS if k in c]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the ledger of second readers
def read_reviews(path=REVIEWS_PATH):
    """(entries, problems): the lines of the ledger  name <TAB> statement digest <TAB> reviewer <TAB> role <TAB> date <TAB> verdict [<TAB> note];  comments (#) and blank lines are skipped"""
    entries, problems = [], []
    if not path or not os.path.exists(path): return entries, problems
    with open(path) as f:
        for k, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith('#'): continue
            p = [x.strip() for x in line.rstrip('\n').split('\t')]
            if len(p) < 6: problems.append('line %d: %d fields, 6 are needed (name, digest, reviewer, role, date, verdict)' % (k, len(p))); continue
            name, dg, who, role, date, verdict = p[:6]; note = p[6] if len(p) > 6 else ''
            bad = []
            if not re.fullmatch(r'[0-9a-f]{16}', dg): bad.append('the digest %r is not 16 hex digits' % dg)
            if not who: bad.append('no reviewer')
            if role not in ROLES: bad.append('the role %r is none of %s' % (role, ', '.join(ROLES)))
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date): bad.append('the date %r is not YYYY-MM-DD' % date)
            if verdict not in VERDICTS: bad.append('the verdict %r is none of %s' % (verdict, ', '.join(VERDICTS)))
            if bad: problems.append('line %d (%s): %s' % (k, name, '; '.join(bad))); continue
            entries.append({'name': name, 'digest': dg, 'reviewer': who, 'role': role, 'date': date, 'verdict': verdict, 'note': note, 'line': k})
    return entries, problems


def review_state(name, digest, entries):
    """(state, readers): where a problem stands with the ledger.  A line counts only for the digest of the statement it was written for; of several lines of one reader, the last counts.
    confirmed: every second reader says faithful or adapted; ambiguous: some says ambiguous, none unfaithful; unfaithful / disputed: all / some (not all) second readers say unfaithful;
    author only: only the author's own lines; stale: lines for other digests only; unreviewed: no line"""
    mine = [e for e in entries if e['name'] == name]
    if not mine: return 'unreviewed', []
    cur = [e for e in mine if e['digest'] == digest]
    if not cur: return 'stale', []
    last = {}
    for e in cur: last[(e['reviewer'], e['role'])] = e
    second = [e for (who, role), e in last.items() if role == 'second']
    if not second: return 'author only', []
    v = {e['verdict'] for e in second}; readers = sorted(e['reviewer'] for e in second)
    if v <= {'faithful', 'adapted'}: return 'confirmed', readers
    if 'unfaithful' in v: return ('unfaithful' if v == {'unfaithful'} else 'disputed'), readers
    return 'ambiguous', readers


def append_reviews(path, lines):
    """add lines (name, digest, reviewer, role, date, verdict, note) to the ledger (created with its header when it does not exist)"""
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        if new: f.write(LEDGER_HEADER)
        for ln in lines: f.write('\t'.join(str(x) for x in ln).rstrip('\t') + '\n')


LEDGER_HEADER = """# The ledger of second readers: one line per reading of one statement, appended (never edited: a later line of the same reader replaces an earlier one).
# name <TAB> statement digest <TAB> reviewer <TAB> role <TAB> date <TAB> verdict <TAB> note
#   statement digest  the 16 hex digits printed in review_sheet.md next to the statement that was read; a line counts only for the statement it was written for, so a statement that changed is no longer reviewed
#   role              second (someone who did not write the statement; counts as an independent reading) | author (the person who wrote it; recorded, does not count)
#   verdict           faithful | adapted (faithful after the adaptation the notes describe) | ambiguous (the English admits several readings: say which one the statement picks) | unfaithful
# Make lines with `python3 src/python/experiments/exp_audit.py import-reviews review_sheet.csv --reviewer NAME` from the filled sheet, or by hand.
"""


def import_reviews(sheet, ledger=REVIEWS_PATH, reviewer=None, role='second', today=None):
    """append the lines of a filled review sheet (review_sheet.csv with the columns verdict, and optionally reviewer, role, date, note filled in) to the ledger.  A row with no verdict is a statement not read yet and is skipped.
    All or nothing: when a row is wrong (a verdict that is none of VERDICTS, no reviewer, a role that is none of ROLES, no digest) nothing is appended.  Returns (lines appended, problems)."""
    today = today or time.strftime('%Y-%m-%d'); lines, problems, n = [], [], 0
    with open(sheet, newline='') as f:
        for k, row in enumerate(csv.DictReader(f), 2):
            verdict = (row.get('verdict') or '').strip()
            if not verdict: continue
            n += 1; who = (row.get('reviewer') or reviewer or '').strip(); r = (row.get('role') or role or '').strip(); date = (row.get('date') or today).strip(); dg = (row.get('digest') or '').strip()
            bad = []
            if verdict not in VERDICTS: bad.append('the verdict %r is none of %s' % (verdict, ', '.join(VERDICTS)))
            if not who: bad.append('no reviewer (a column of the sheet or --reviewer)')
            if r not in ROLES: bad.append('the role %r is none of %s' % (r, ', '.join(ROLES)))
            if not re.fullmatch(r'[0-9a-f]{16}', dg): bad.append('the digest %r is not 16 hex digits (do not edit that column)' % dg)
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date): bad.append('the date %r is not YYYY-MM-DD' % date)
            if bad: problems.append('row %d (%s): %s' % (k, row.get('name'), '; '.join(bad))); continue
            lines.append((row['name'].strip(), dg, who, r, date, verdict, (row.get('note') or '').replace('\t', ' ').replace('\n', ' ').strip()))
    if problems: return 0, problems
    if lines: append_reviews(ledger, lines)
    return len(lines), []


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ summaries
def split_rank(r): return SPLITS.index(r['split']) if r.get('split') in SPLITS else len(SPLITS)


def derive(rec, entries):
    """a record with what follows from it: its failures, its cautions and where it stands with the ledger"""
    state, readers = review_state(rec['name'], rec.get('statement_digest'), entries)
    return dict(rec, failures=failures_of(rec), cautions=cautions_of(rec), review=state, readers=readers)


def summarize(recs, tree, expected, entries, ledger_problems=()):
    """the records that count (the last of each problem, made for `tree`) with their derived parts, and the totals.  `expected`: the names of the bank, in order (None: those that have a record)"""
    last = collections.OrderedDict(); stale = []
    for r in recs:
        if not r.get('name'): continue
        if tree and r.get('tree') != tree: stale.append(r['name']); continue
        last[r['name']] = r
    names = list(expected) if expected else sorted(last)
    rows = [derive(last[n], entries) for n in names if n in last]
    S = {'rows': rows, 'expected': len(names), 'run': len(rows), 'missing': [n for n in names if n not in last], 'stale': sorted(set(stale) - set(last)), 'stale_records': len(stale),
         'unknown_reviews': sorted({e['name'] for e in entries} - set(names)) if expected else [], 'ledger_problems': list(ledger_problems), 'entries': len(entries)}
    C = collections.Counter(); T = collections.OrderedDict()
    def total(key, n): T[key] = T.get(key, 0) + n
    for r in rows:
        total('problems', 1); total('proved', 1 if r['proved'] else 0); total('statement_only', 0 if r['proved'] else 1); total('answer', 1 if r['kind'] == 'answer' else 0)
        total('with_spec', 1 if r.get('spec') else 0); total('in_fragment', 1 if r.get('in_fragment') else 0); total('sayable', 1 if r.get('sayable') else 0)
        total('checked', 1 if not r['failures'] else 0); total('failed', 1 if r['failures'] else 0); total('cautioned', 1 if r['cautions'] else 0)
        sem = r.get('sem') or {}
        for k in ('points', 'nonvacuous', 'mutants', 'caught', 'equivalent', 'survivors', 'shape_caught', 'unevaluable'): total(k, sem.get(k, 0))
        total('exemptions', len((r.get('spec') or {}).get('exemptions') or ()))
        for c in r['cautions']: C[c] += 1
        total('compared', 1 if r['verified'] == 'answer-checked' else 0)
    S['totals'] = T; S['cautions'] = {k: C.get(k, 0) for k in CAUTIONS}
    S['by_origin'] = collections.OrderedDict()
    for o in ORIGINS:
        sub = [r for r in rows if r['origin'] == o]
        if sub: S['by_origin'][o] = group_totals(sub)
    S['by_split'] = collections.OrderedDict()
    for s in SPLITS + (None,):
        sub = [r for r in rows if r.get('split') == s]
        if sub: S['by_split'][s or '(none)'] = group_totals(sub)
    S['review_states'] = {k: sum(1 for r in rows if r['review'] == k) for k in REVIEW_STATES}
    S['readers'] = sorted({w for r in rows for w in r['readers']})
    return S


def group_totals(rows):
    sem = lambda r: r.get('sem') or {}
    return {'problems': len(rows), 'proved': sum(1 for r in rows if r['proved']), 'statement_only': sum(1 for r in rows if not r['proved']), 'checked': sum(1 for r in rows if not r['failures']),
            'mutants': sum(sem(r).get('mutants', 0) for r in rows), 'caught': sum(sem(r).get('caught', 0) for r in rows), 'equivalent': sum(sem(r).get('equivalent', 0) for r in rows),
            'compared': sum(1 for r in rows if r['verified'] == 'answer-checked'), 'cautioned': sum(1 for r in rows if r['cautions']), 'confirmed': sum(1 for r in rows if r['review'] == 'confirmed'),
            'states': {k: sum(1 for r in rows if r['review'] == k) for k in REVIEW_STATES}}


def assess(S):
    """(status, failures, incomplete): FAILED when a statement fails its checks or a second reader found one unfaithful (or readers disagree) or the ledger is wrong; INCOMPLETE when records are missing or were made for
    other code; CLEAN otherwise.  How many statements a second reader confirmed is a number of the report, not a condition."""
    failures, incomplete = [], []
    rows = S.get('rows') or []
    if not rows: incomplete.append('no problem records for this tree')
    bad = [r for r in rows if r['failures']]
    if bad: failures.append('%d statement(s) fail their checks: %s' % (len(bad), '; '.join('%s (%s)' % (r['name'], r['failures'][0][:70]) for r in bad[:5])))
    unf = [r['name'] for r in rows if r['review'] in ('unfaithful', 'disputed')]
    if unf: failures.append('a second reader found %d statement(s) unfaithful, or the readers disagree: %s' % (len(unf), ', '.join(unf[:8])))
    if S.get('ledger_problems'): failures += ['the ledger: ' + x for x in S['ledger_problems'][:5]]
    if S.get('unknown_reviews'): failures.append('the ledger has lines for problems that are not in the bank: %s' % ', '.join(S['unknown_reviews'][:8]))
    if rows and S.get('missing'): incomplete.append('%d of %d problems have no record for this tree (run the audit; --only makes a partial folder)' % (len(S['missing']), S['expected']))
    if rows and S.get('stale_records'): incomplete.append('%d record(s) made for another tree are ignored' % S['stale_records'])
    st = S.get('stamp') or {}
    if st.get('tree_stale'): incomplete.append('the evidence was collected on tree %s, the code is now %s' % (st['tree_stale'], st.get('tree')))
    if st.get('dirty'): incomplete.append('the code (src/, tests/, mutants/, the manifest of the dataset) has uncommitted changes: the commit does not identify it (the tree digest does)')
    if S.get('tree_changed'): incomplete.append('the tree changed while the evidence was collected (%s, then %s)' % S['tree_changed'])
    return ('FAILED' if failures else 'INCOMPLETE' if incomplete else 'CLEAN'), failures, incomplete


def pct(a, b): return XS.pct(a, b)


def num(x): return XS.num(x)


def priority(r):
    """the order of the review sheet: statements that fail first, then the hidden test split, then contest problems before invented ones, then more cautions first"""
    return (0 if r['failures'] else 1, split_rank(r), ORIGINS.index(r['origin']) if r['origin'] in ORIGINS else len(ORIGINS), -len(r['cautions']), r['name'])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ tables
def build_tables(S):
    """the tables of the folder (exp_tables.Table) from the summary S"""
    T = collections.OrderedDict(); t = S.get('totals') or {}; rows = S.get('rows') or []
    if not rows: return T
    n = t['problems']; RS = S['review_states']
    summary = [['problems', '%s (%s with a reference proof, %s statement-only)' % (num(n), num(t['proved']), num(t['statement_only']))],
               ['statement read back and compared with its intended meaning', '%s of %s' % (num(t['checked']), num(n))],
               ['points of the box evaluated', num(t['points'])],
               ['slips of the formalisation tried (one plausible mistake each)', '%s, %s caught, %s declared equivalent to the statement' % (num(t['mutants']), num(t['caught']), num(t['equivalent']))],
               ['published wording compared with a source', '%s of %s' % (num(t['compared']), num(n))],
               ['statements with at least one caution', '%s of %s' % (num(t['cautioned']), num(n))],
               ['confirmed by a second reader', '%s of %s (%s%%)' % (num(RS['confirmed']), num(n), pct(RS['confirmed'], n))],
               ['second readers', ', '.join(S['readers']) or '(none)']]
    if S.get('stale_records') or S.get('missing'): summary.append(['problems without a current record', num(len(S.get('missing') or ()))])
    T['audit_summary'] = TB.Table('audit_summary', 'The statement audit in numbers.', ['what', 'value'], summary, align='ll')
    orows = []
    for o, g in S['by_origin'].items(): orows.append([o, num(g['problems']), num(g['proved']), num(g['statement_only']), num(g['checked']), num(g['mutants']), num(g['caught']), num(g['compared']), num(g['confirmed'])])
    orows.append(['all', num(n), num(t['proved']), num(t['statement_only']), num(t['checked']), num(t['mutants']), num(t['caught']), num(t['compared']), num(RS['confirmed'])])
    T['audit_origins'] = TB.Table('audit_origins', 'Statements by origin: contest (formalised as stated), adapted (changed to fit the vocabulary), style (written in the manner of a contest), folklore.',
                                  ['origin', 'problems', 'with proof', 'statement-only', 'checks hold', 'slips tried', 'caught', 'wording compared', 'second reader'], orows, align='lrrrrrrrr',
                                  notes=['A statement-only problem is a task all the same: the kernel checks any proof an agent finds; the bank does not hold one.', 'Slips are caught when the box tells the mutant statement from the original; the rest are declared equivalent in the spec, with a reason.'])
    crow = []
    for k, why in CAUTIONS.items():
        names = [r['name'] for r in rows if k in r['cautions']]
        crow.append([k, num(len(names)), pct(len(names), n), ', '.join(names[:3]) + (', ...' if len(names) > 3 else '')])
    T['audit_cautions'] = TB.Table('audit_cautions', 'Cautions: where the automatic evidence for a statement is weaker, so a reader should look.', ['caution', 'statements', '%', 'for example'], crow, align='lrrl',
                                   notes=['%s: %s' % (k, why) for k, why in CAUTIONS.items()])
    rrows = []
    cw = [r for r in rows if r['origin'] in ('contest', 'adapted')]
    for s in REVIEW_STATES:
        rrows.append([s, num(RS[s]), num(sum(1 for r in cw if r['review'] == s)), num(sum(1 for r in rows if r.get('split') == 'test' and r['review'] == s))])
    T['audit_reviews'] = TB.Table('audit_reviews', 'Second readers: where each statement stands with the ledger (docs/statement_reviews.tsv).', ['state', 'all', 'contest and adapted', 'test split'], rrows, align='lrrr',
                                  notes=['A reading counts for the statement it was written for (by digest); a statement that changed since is stale.', 'confirmed: every second reader says faithful or adapted. unreviewed: nobody has read it yet.'])
    bad = [r for r in rows if r['failures']]
    T['audit_failures'] = TB.Table('audit_failures', 'Statements that fail their checks.', ['problem', 'what'], [[r['name'], '; '.join(r['failures'])[:200]] for r in bad] or [['(none)', '']], align='ll')
    return T


def numbers(S):
    """the numbers the text may quote, by name (numbers.tex defines \\numAudit... for each)"""
    N = collections.OrderedDict(); t = S.get('totals') or {}
    if not S.get('rows'): return N
    RS = S['review_states']; cw = [r for r in S['rows'] if r['origin'] in ('contest', 'adapted')]; tst = [r for r in S['rows'] if r.get('split') == 'test']
    N.update(audit_problems=t['problems'], audit_proved=t['proved'], audit_statement_only=t['statement_only'], audit_checked=t['checked'], audit_failed=t['failed'], audit_points=t['points'], audit_mutants=t['mutants'],
             audit_mutants_caught=t['caught'], audit_mutants_equivalent=t['equivalent'], audit_compared=t['compared'], audit_cautioned=t['cautioned'], audit_contest=len(cw),
             audit_confirmed=RS['confirmed'], audit_confirmed_pct=pct(RS['confirmed'], t['problems']), audit_confirmed_contest=sum(1 for r in cw if r['review'] == 'confirmed'),
             audit_confirmed_test=sum(1 for r in tst if r['review'] == 'confirmed'), audit_test=len(tst), audit_unreviewed=RS['unreviewed'] + RS['stale'], audit_issues=RS['ambiguous'] + RS['disputed'] + RS['unfaithful'],
             audit_readers=len(S['readers']))
    for k in CAUTIONS: N['audit_caution_' + k] = S['cautions'][k]
    return N


README_DEFS = """## Definitions

* **statement check** (`src/python/problems/semantic.py`): the formal statement is read back from the kernel's term into a small language (integers and naturals, sums and products, divisibility, gcd, primality, quantifiers over a
  window of values, functions given by test functions) and evaluated on a *box* of points. Each part (every hypothesis, the conclusion, or both sides of an equivalence) must agree with a Python predicate written separately from the
  statement (the problem's *spec*) at every point, and so must the numbers inside the relation atoms (the two sides of each equation or inequality), so that a slip in a constant or in the direction of an inequality cannot hide
  behind a truth table. The statement must be **non-vacuous** (some point of the box satisfies the hypotheses), a statement that has no proof must not be false at any point, and it must be sayable in the typed term builder the agent uses.
* **slip** (negative control): a copy of the statement with ONE plausible mistake of a formaliser (a constant off by one, a comparison weakened, a sum for a product, a swapped subtraction, a quantifier dropped, ...). Every slip must
  differ from the statement somewhere on the box; one that cannot (it means the same thing) is **declared equivalent** in the spec, with its reason (`stats/exemptions.csv`). A slip nobody notices would fail the check.
* **caution**: where the automatic evidence is weaker, so that a reader should look: `audit_cautions` lists each with its meaning.
* **second reader**: a person who did not write the statement and read it against the problem's text; the ledger (`docs/statement_reviews.tsv`) holds the verdict (faithful, adapted, ambiguous, unfaithful) with the digest of the
  statement that was read, so that a statement that changed afterwards is no longer counted as reviewed.
"""

README_LIMITS = """## What this audit does and does not show

* The spec is written by the people who wrote the statement. The checks catch slips between the two (a transcription error, a wrong constant, a missing hypothesis), not a misreading of the English that both share. Only a second
  reader can: that is what the ledger is for, and the number of statements one confirmed is reported as it is.
* The box is a finite sample (integers up to 8 in absolute value, naturals up to 12, unless the spec says otherwise; `domains` for a problem whose answer is a large number). A quantifier is evaluated over that window, so an
  existential statement is checked through small witnesses, a statement about all functions through a family of test functions. The cautions mark those.
* `verified` (the published wording was compared with a public source) is declared by whoever added the problem; the audit counts it and does not compute it.
* A statement-only problem has no reference proof, so its truth rests on the box and on the brute-force screen of `imo_truth.py`, not on a proof; an agent that proves it is checked by the kernel all the same.
* The audit covers the bank's problems, whose statements formalise a text. The synthetic theorems are statements of kernel-checked theorems by construction and are described by the generator's own statistics.
"""


def review_sheet(S, only_open=True):
    """(markdown, rows of the csv): the statements a second reader should read, in the order of `priority`, each with its text, the statement as a person reads it, the cautions and the digest to put in the ledger"""
    rows = [r for r in S.get('rows') or [] if not only_open or r['review'] != 'confirmed']
    rows.sort(key=priority)
    L = ['# Review sheet', '', 'Read each statement against the problem text, then put your verdict in `review_sheet.csv` (columns `reviewer`, `verdict`, optionally `note`) and run '
         '`python3 src/python/experiments/exp_audit.py import-reviews review_sheet.csv --reviewer NAME`, or append a line to `docs/statement_reviews.tsv` by hand.', '',
         '**Verdicts:** ' + '; '.join('`%s` -- %s' % (k, v) for k, v in VERDICTS.items()) + '. A verdict counts for the statement whose digest is printed below. %d statement(s) to read.' % len(rows), '']
    cols = ['name', 'digest', 'source', 'origin', 'verified', 'level', 'split', 'proved', 'cautions', 'state', 'text', 'formal', 'notes', 'reviewer', 'role', 'date', 'verdict', 'note']
    out = []
    for r in rows:
        L += ['## %s' % r['name'], '', '`%s` -- %s -- %s, level %s, %s split, %s -- digest `%s` -- state: %s' % (r['name'], r['source'], r['origin'], r['level'], r.get('split') or '?', 'with a reference proof' if r['proved'] else 'statement-only', r['statement_digest'], r['review']), '',
              '**Problem.** ' + (r['text'] or ''), '', '**Formal statement.** `%s`' % (r.get('formal') or '(outside the fragment the evaluator reads)'), '']
        if r['notes']: L += ['**Notes of the formaliser.** ' + r['notes'], '']
        if r['cautions']: L += ['**Cautions.** ' + '; '.join('%s' % k for k in r['cautions']), '']
        if r['failures']: L += ['**Fails its checks.** ' + '; '.join(r['failures']), '']
        out.append([r['name'], r['statement_digest'], r['source'], r['origin'], r['verified'], r['level'], r.get('split') or '', 'yes' if r['proved'] else 'no', ' '.join(r['cautions']), r['review'], r['text'], r.get('formal') or '', r['notes'], '', 'second', '', '', ''])
    return '\n'.join(L) + '\n', cols, out


def render(S, out):
    """write the derived files of the folder from the summary S; returns the list of paths"""
    for sub in ('tables', 'stats'): shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
    for fn in ('README.md', 'report.md', 'audit.json', 'numbers.json', 'numbers.tex', 'review_sheet.md', 'review_sheet.csv'):
        if os.path.exists(os.path.join(out, fn)): os.remove(os.path.join(out, fn))
    os.makedirs(os.path.join(out, 'tables')); os.makedirs(os.path.join(out, 'stats')); files = []
    status, failures, incomplete = assess(S); T = build_tables(S); N = numbers(S); rows = S.get('rows') or []
    for key, t in T.items():
        for ext, text in (('md', t.md()), ('tex', t.tex()), ('csv', t.csv())):
            p = os.path.join(out, 'tables', '%s.%s' % (key, ext))
            with open(p, 'w') as fh: fh.write(text)
            files.append(p)
    if rows:
        files.append(EB.write_csv(os.path.join(out, 'stats', 'problems.csv'), ['name', 'source', 'origin', 'verified', 'kind', 'level', 'proved', 'split', 'points', 'nonvacuous', 'mutants', 'caught', 'equivalent', 'survivors', 'checks_hold', 'failures', 'cautions', 'review', 'digest'],
                                  [[r['name'], r['source'], r['origin'], r['verified'], r['kind'], r['level'], r['proved'], r.get('split') or '', (r.get('sem') or {}).get('points', ''), (r.get('sem') or {}).get('nonvacuous', ''), (r.get('sem') or {}).get('mutants', ''),
                                    (r.get('sem') or {}).get('caught', ''), (r.get('sem') or {}).get('equivalent', ''), (r.get('sem') or {}).get('survivors', ''), not r['failures'], '; '.join(r['failures']), ' '.join(r['cautions']), r['review'], r['statement_digest']] for r in rows]))
        files.append(EB.write_csv(os.path.join(out, 'stats', 'cautions.csv'), ['problem', 'caution'], [[r['name'], c] for r in rows for c in r['cautions']]))
        files.append(EB.write_csv(os.path.join(out, 'stats', 'exemptions.csv'), ['problem', 'slip contains', 'reason'], [[r['name'], k, v] for r in rows for k, v in (r.get('spec') or {}).get('exemptions') or ()]))
        files.append(EB.write_csv(os.path.join(out, 'stats', 'reviews.csv'), ['problem', 'digest', 'reviewer', 'role', 'date', 'verdict', 'counts for the current statement', 'note'],
                                  [[e['name'], e['digest'], e['reviewer'], e['role'], e['date'], e['verdict'], any(r['name'] == e['name'] and r['statement_digest'] == e['digest'] for r in rows), e['note']] for e in S.get('entries_list') or ()]))
    md, cols, srows = review_sheet(S)
    with open(os.path.join(out, 'review_sheet.md'), 'w') as fh: fh.write(md)
    files.append(os.path.join(out, 'review_sheet.md'))
    files.append(EB.write_csv(os.path.join(out, 'review_sheet.csv'), cols, srows))
    summary = {'status': status, 'failures': failures, 'incomplete': incomplete, 'numbers': N, 'stamp': S.get('stamp'), 'totals': S.get('totals'), 'cautions': S.get('cautions'), 'by_origin': S.get('by_origin'), 'by_split': S.get('by_split'),
               'review_states': S.get('review_states'), 'readers': S.get('readers'), 'expected': S.get('expected'), 'run': S.get('run'), 'missing': S.get('missing'), 'stale': S.get('stale'), 'ledger': S.get('ledger')}
    p = os.path.join(out, 'audit.json')
    with open(p, 'w') as fh: json.dump(summary, fh, indent=1, default=str)
    files.append(p)
    ER.write_numbers(N, out); files += [os.path.join(out, 'numbers.json'), os.path.join(out, 'numbers.tex')]
    st = S.get('stamp') or {}
    L = ['# Statement audit', '', 'Status: **%s**' % status, '']
    if failures: L += ['**Failed checks** (fix these before quoting anything):', ''] + ['* ' + x for x in failures] + ['']
    if incomplete: L += ['**Incomplete or stale evidence:**', ''] + ['* ' + x for x in incomplete] + ['']
    L += ['Code: commit `%s`%s, tree digest `%s`. Collected %s on %s (%s).' % ((st.get('commit') or '?')[:12], ' (uncommitted changes)' if st.get('dirty') else '', st.get('tree'), st.get('time'), st.get('host'), st.get('python')), '']
    lg = S.get('ledger') or {}
    L += ['Ledger: `%s` (%s lines, sha256 `%s`).' % (lg.get('path'), lg.get('lines'), lg.get('sha256')), '']
    L += ['Read `report.md` for the tables, `numbers.tex` for the macros, `audit.json` for everything machine-readable and `review_sheet.md` to read the statements; the raw records are in `evidence/`.', '', '## Reproduce', '',
          '    make audit                          # the checks for the whole bank (about a minute for the world, then a few minutes), then this folder',
          '    python3 src/python/experiments/exp_audit.py report        # the tables again from the saved evidence and the ledger', '    sha256sum -c MANIFEST.sha256      # in this folder: every file is the one that was written', '']
    L += README_DEFS.split('\n') + [''] + README_LIMITS.split('\n')
    L += ['', '## Files', '', '* `tables/` -- one md/tex/csv per table: ' + ', '.join('`%s`' % k for k in T), '* `stats/` -- one row per problem, per caution, per exempted slip, per line of the ledger',
          '* `review_sheet.md`, `review_sheet.csv` -- the statements still to be read by a second reader, most important first', '* `numbers.json`, `numbers.tex` -- the numbers by name (`\\numAuditConfirmed`, ...)',
          '* `audit.json` -- the whole summary', '* `evidence/` -- the raw records (the only files that are not computed)', '* `MANIFEST.sha256` -- the SHA-256 of every other file']
    p = os.path.join(out, 'README.md')
    with open(p, 'w') as fh: fh.write('\n'.join(L) + '\n')
    files.append(p)
    R = ['# Statement audit -- report', '', 'Status: **%s**' % status, ''] + (['Failed checks:', ''] + ['* ' + x for x in failures] + [''] if failures else []) + (['Incomplete or stale:', ''] + ['* ' + x for x in incomplete] + [''] if incomplete else [])
    for key, t in T.items(): R += [t.md(), '']
    p = os.path.join(out, 'report.md')
    with open(p, 'w') as fh: fh.write('\n'.join(R))
    files.append(p)
    EB.manifest(out)
    return files


def collect(out, tree=None, reviews=REVIEWS_PATH):
    """read the evidence of `out` and the ledger and summarise them (against `tree`, default the digest of the code as it is now)"""
    ev = os.path.join(out, 'evidence'); tree = tree or XS.tree_digest()
    st = XS.read_json(os.path.join(ev, 'stamp.json'), {})
    entries, lprobs = read_reviews(reviews)
    S = summarize(XS.read_jsonl(os.path.join(ev, 'problems.jsonl')), tree, st.get('bank_names'), entries, lprobs)
    S['entries_list'] = entries
    S['ledger'] = {'path': os.path.relpath(reviews, ROOT) if reviews and reviews.startswith(ROOT) else reviews, 'lines': len(entries), 'sha256': XS.sha256_of(reviews, 16) if reviews and os.path.exists(reviews) else None}
    S['stamp'] = dict(st, tree=tree) if st else XS.stamp({'collected': False})
    if st and st.get('tree_before') and st.get('tree_after') and st['tree_before'] != st['tree_after']: S['tree_changed'] = (st['tree_before'], st['tree_after'])
    if st and st.get('tree') and st['tree'] != tree: S['stamp']['tree_stale'] = st['tree']
    return S


def report(out, reviews=REVIEWS_PATH):
    S = collect(out, reviews=reviews); files = render(S, out)
    return S, files


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ running
def run(a):
    import bank_all
    out = os.path.abspath(a.out); ev = os.path.join(out, 'evidence'); os.makedirs(ev, exist_ok=True)
    names_all = bank_all.names(); only = list(a.only) if a.only else None
    unknown = sorted(set(only or ()) - set(names_all))
    if unknown: print('no such problem(s): %s' % ', '.join(unknown)); return 2
    tree = XS.tree_digest(); path = os.path.join(ev, 'problems.jsonl')
    st = XS.stamp({'bank_names': names_all, 'only': only, 'tree_before': tree})
    t0 = time.time(); bk = bank_all.load(only=set(only) if only else None); print('== world built in %.0f s, %d problems' % (time.time() - t0, len(bk.placed)), flush=True)
    import bank as B
    old = collections.OrderedDict((r['name'], r) for r in XS.read_jsonl(path) if r.get('tree') == tree)
    want = [n for n in (only or names_all) if n in bk.placed]
    done = set() if a.redo else {n for n in want if n in old and old[n].get('statement_digest') == B.statement_digest(bk.w, bk.placed[n].stmt) and not old[n].get('error')}      # (--redo: nothing is reused, but what is not asked for is kept)
    ran = 0
    with open(path, 'w') as fh:                                                   # (the records of other trees are dropped; each new record is written as soon as it is made)
        for n, r in old.items():
            if n not in want or n in done: fh.write(json.dumps(r, sort_keys=True) + '\n')
        for i, n in enumerate(want):
            if n in done: continue
            fh.write(json.dumps(problem_record(bk, n, tree), sort_keys=True) + '\n'); fh.flush(); ran += 1
            if ran % 25 == 0: print('   %d checked, %d / %d' % (ran, i + 1, len(want)), flush=True)
    st.update(tree_after=XS.tree_digest(), finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'), ran=ran, reused=len(done), seconds=round(time.time() - t0, 1))
    with open(os.path.join(ev, 'stamp.json'), 'w') as fh: json.dump(st, fh, indent=1, sort_keys=True)
    print('== %d checked, %d reused, %.0f s' % (ran, len(done), time.time() - t0), flush=True)
    if a.no_report: return 0
    S, files = report(out, a.reviews)
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
    r = sub.add_parser('run', help='check the statements and write the folder'); rp = sub.add_parser('report', help='write the tables and the README from the saved evidence and the ledger')
    im = sub.add_parser('import-reviews', help='append the verdicts of a filled review sheet to the ledger')
    for p in (r, rp): p.add_argument('--out', default=DEFAULT_OUT); p.add_argument('--reviews', default=REVIEWS_PATH)
    r.add_argument('--only', nargs='*', default=None, help='check only these problems (a quick look: the folder then says it is incomplete)'); r.add_argument('--redo', action='store_true', help='check again what has a record already')
    r.add_argument('--no-report', action='store_true')
    im.add_argument('sheet'); im.add_argument('--reviewer', default=None, help='the name to use for rows that have none'); im.add_argument('--role', default='second', choices=ROLES); im.add_argument('--reviews', default=REVIEWS_PATH)
    a = ap.parse_args(argv)
    if a.cmd == 'run': return run(a)
    if a.cmd == 'import-reviews':
        n, probs = import_reviews(a.sheet, a.reviews, a.reviewer, a.role)
        for x in probs: print('  ' + x)
        print('%d line(s) appended to %s' % (n, a.reviews) if not probs else 'nothing appended: %d row(s) to fix' % len(probs))
        return 1 if probs else 0
    S, files = report(os.path.abspath(a.out), a.reviews); status, failures, incomplete = assess(S)
    print('%s: %s (%d files)' % (a.out, status, len(files))); [print('   FAILED: ' + x) for x in failures]; [print('   incomplete: ' + x) for x in incomplete]
    return {'CLEAN': 0, 'FAILED': 1, 'INCOMPLETE': 2}[status]


if __name__ == '__main__':
    sys.exit(main())
