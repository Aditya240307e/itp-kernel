#!/usr/bin/env python3
"""review_packs.py -- split the review sheet of an audit folder into packs for second readers.

    python3 tools/review_packs.py AUDIT_DIR --readers 4 [--copies 2] [--split test] [--states unreviewed,stale] [--out packs]

AUDIT_DIR holds review_sheet.md and review_sheet.csv (made by `exp_audit.py`, i.e. `make audit`).  Every statement of the chosen split whose state is one of --states goes to --copies different readers
(round-robin over the statements sorted by level and name, so that each reader gets a similar mix of levels).  Reader K gets

    reader_K.md   the statements assigned to K, in the format of the review sheet
    reader_K.csv  the same statements as the rows of the review sheet (reviewer empty, role second): K fills in `reviewer`, `verdict` and `note`
    READ_ME_FIRST.md   a copy of docs/SECOND_READER.md (the handout: notation, checklist, verdicts, worked examples)

and assignment.tsv lists, for every statement, its digest and the readers that read it.  The packs are deterministic (the same folder and arguments give the same files).  Nothing in the repository or in
the ledger changes: the filled CSV goes back to the author, who runs

    python3 src/python/experiments/exp_audit.py import-reviews reader_K.csv [--reviewer NAME]

The digest column ties a verdict to the statement that was read.  A pack made from one audit folder is valid for another only while the statements are the same (the digests say).

Exit status: 0 packs written; 2 a problem (named on standard error), nothing written.
"""
import argparse, csv, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HANDOUT = os.path.join(HERE, '..', 'docs', 'SECOND_READER.md')
MINUTES_PER_STATEMENT = 5                       # (the figure of docs/PAPER_PLAN.md R10; a rough guide, not a measurement)


def die(msg):
    sys.stderr.write('review_packs.py: %s\n' % msg)
    sys.exit(2)


def read_sheet(folder):
    """(header text of the .md, {name: section text}, fieldnames, rows) of the review sheet in `folder`"""
    md_path, csv_path = os.path.join(folder, 'review_sheet.md'), os.path.join(folder, 'review_sheet.csv')
    for p in (md_path, csv_path):
        if not os.path.isfile(p): die('%s is missing (is %s an audit folder made by exp_audit.py?)' % (p, folder))
    text = open(md_path, encoding='utf-8').read()
    parts = re.split(r'(?m)^(?=## )', text)
    header, sections = parts[0], {}
    for s in parts[1:]:
        name = s.split('\n', 1)[0][3:].strip()
        if name in sections: die('the sheet lists %s twice' % name)
        sections[name] = s.rstrip('\n') + '\n'
    with open(csv_path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f); fields = list(rd.fieldnames or []); rows = list(rd)
    need = {'name', 'digest', 'level', 'split', 'state', 'reviewer', 'role', 'verdict', 'note'}
    if not need <= set(fields): die('review_sheet.csv lacks the columns %s' % sorted(need - set(fields)))
    names = [r['name'] for r in rows]
    if len(set(names)) != len(names): die('review_sheet.csv lists a problem twice')
    if set(names) != set(sections): die('the .md and the .csv of the sheet do not list the same problems (%s only in one of them)' % sorted(set(names) ^ set(sections))[:5])
    return header, sections, fields, rows


def level_of(row):
    try: return int(row['level'])
    except (TypeError, ValueError): return 99


def assign(rows, readers, copies):
    """{name: [reader numbers 1..readers]}: statement i (sorted by level, then name) goes to readers i, i+1, .. i+copies-1 (mod readers): loads differ by at most one, and no reader gets a statement twice"""
    order = sorted(rows, key=lambda r: (level_of(r), r['name']))
    return {r['name']: [(i + j) % readers + 1 for j in range(copies)] for i, r in enumerate(order)}, order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', help='an audit folder (results/audit) with review_sheet.md and review_sheet.csv')
    ap.add_argument('--readers', type=int, default=4, help='number of second readers (default 4)')
    ap.add_argument('--copies', type=int, default=1, help='how many readers read each statement (default 1; 2 gives a statement a second opinion)')
    ap.add_argument('--split', default='test', help='train, dev, test or all (default test: the statements the paper stands on)')
    ap.add_argument('--states', default='unreviewed,stale', help='which states of the ledger to include (default unreviewed,stale; others: confirmed, ambiguous, author only, disputed, unfaithful)')
    ap.add_argument('--out', default='review_packs', help='the folder to write (default review_packs; it must not exist or be empty)')
    a = ap.parse_args(argv)
    if a.readers < 1: die('--readers must be at least 1')
    if not 1 <= a.copies <= a.readers: die('--copies must be between 1 and --readers (%d)' % a.readers)
    if a.split not in ('train', 'dev', 'test', 'all'): die('--split is train, dev, test or all')
    states = {s.strip() for s in a.states.split(',') if s.strip()}
    header, sections, fields, rows = read_sheet(a.folder)
    chosen = [r for r in rows if (a.split == 'all' or r['split'] == a.split) and r['state'] in states]
    if not chosen: die('no statement of split %s is in state %s' % (a.split, ', '.join(sorted(states))))
    if os.path.exists(a.out) and os.listdir(a.out): die('%s exists and is not empty' % a.out)
    if not os.path.isfile(HANDOUT): die('%s is missing' % HANDOUT)
    who, order = assign(chosen, a.readers, a.copies)
    os.makedirs(a.out, exist_ok=True)
    handout = open(HANDOUT, encoding='utf-8').read()
    with open(os.path.join(a.out, 'READ_ME_FIRST.md'), 'w', encoding='utf-8') as f: f.write(handout)
    with open(os.path.join(a.out, 'assignment.tsv'), 'w', encoding='utf-8') as f:
        f.write('name\tdigest\tlevel\tsplit\treaders\n')
        for r in order: f.write('%s\t%s\t%s\t%s\t%s\n' % (r['name'], r['digest'], r['level'], r['split'], ','.join(str(k) for k in who[r['name']])))
    counts = {}
    for k in range(1, a.readers + 1):
        mine = [r for r in order if k in who[r['name']]]
        counts[k] = len(mine)
        with open(os.path.join(a.out, 'reader_%d.md' % k), 'w', encoding='utf-8') as f:
            f.write(header.rstrip('\n') + '\n\n')
            f.write('**Pack %d of %d: %d statements.** Read `READ_ME_FIRST.md` first (notation, checklist, the four verdicts). Fill in `reader_%d.csv`: the columns `reviewer`, `verdict` and, unless the verdict is plain `faithful`, `note`.\n\n' % (k, a.readers, len(mine), k))
            for r in mine: f.write(sections[r['name']] + '\n')
        with open(os.path.join(a.out, 'reader_%d.csv' % k), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader()
            for r in mine:
                row = dict(r); row['reviewer'] = ''; row['verdict'] = ''; row['note'] = ''; row['role'] = 'second'; row['date'] = ''
                w.writerow(row)
    print('%d statements (split %s, state %s), %d readers, %d copies each:' % (len(chosen), a.split, '/'.join(sorted(states)), a.readers, a.copies))
    for k in range(1, a.readers + 1):
        print('  reader %d: %2d statements, about %.1f hours at %d minutes each' % (k, counts[k], counts[k] * MINUTES_PER_STATEMENT / 60.0, MINUTES_PER_STATEMENT))
    print('written to %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
