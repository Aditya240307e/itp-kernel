"""ml_report.py -- tables from the result files of ml_eval.py.

    python3 ml_report.py blind=runs/blind_dev.json bc1=runs/bc1_dev.json [--budgets 250,500,1000,2000,4000,8000] [--split dev] [--markdown]

For every file (a method): how many (problem, regime) rows were solved within each budget of probes, per split; and, per problem, the probes that each method needed (`-` when it did not find a proof).  A
search is deterministic and the budget only truncates it, so a result file made with the largest budget holds the whole curve.  Rows are (problem, regime) pairs; a problem without lemmas of its own is
one row in both regimes' names but counted once (`same_as` rows are skipped).
"""
import os, sys, json, argparse, collections

BUDGETS = (125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def load(path):
    d = json.load(open(path))
    return [r for r in d['rows'] if 'same_as' not in r], d.get('args', {})


def solved_at(rows, b): return sum(1 for r in rows if r['solved'] and r['probes'] <= b)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+'); ap.add_argument('--budgets', default=','.join(map(str, BUDGETS))); ap.add_argument('--split', default=''); ap.add_argument('--markdown', action='store_true')
    ap.add_argument('--per-problem', action='store_true')
    a = ap.parse_args(argv)
    budgets = [int(x) for x in a.budgets.split(',')]
    methods = []
    for f in a.files:
        name, path = f.split('=', 1) if '=' in f else (os.path.splitext(os.path.basename(f))[0], f)
        rows, args = load(path)
        if a.split: rows = [r for r in rows if r.get('split') == a.split]
        methods.append((name, rows))
    allrows = {}
    for name, rows in methods:
        for r in rows: allrows.setdefault((r['name'], r['regime']), {})[name] = r
    splits = sorted({r['split'] for _, rows in methods for r in rows})
    sep = ' | ' if a.markdown else '  '
    def line(cells): return ('| ' + ' | '.join(cells) + ' |') if a.markdown else sep.join(cells)
    mx = max(budgets)
    for sp in splits + ['all']:
        sel = {k: v for k, v in allrows.items() if sp == 'all' or any(r['split'] == sp for r in v.values())}
        print('\n%s: %d rows (problem, regime)' % (sp, len(sel)))
        print(line(['method'.ljust(14)] + [('%d' % b).rjust(6) for b in budgets] + ['seconds'.rjust(8)]))
        if a.markdown: print('|' + '---|' * (len(budgets) + 2))
        for name, rows in methods:
            rr = [r for r in rows if sp == 'all' or r['split'] == sp]
            print(line([name.ljust(14)] + [str(solved_at(rr, b)).rjust(6) for b in budgets] + [('%.0f' % sum(r['seconds'] for r in rr)).rjust(8)]))
    if a.per_problem:
        print('\nprobes needed per (problem, regime); - = not solved within the largest budget')
        names = [n for n, _ in methods]
        print(line(['problem'.ljust(28), 'regime'.ljust(7), 'lvl', 'split'.ljust(5)] + [n.rjust(8) for n in names]))
        for (pn, rg), v in sorted(allrows.items(), key=lambda kv: (next(iter(kv[1].values()))['split'], next(iter(kv[1].values()))['level'], kv[0])):
            any_r = next(iter(v.values()))
            cells = [pn.ljust(28), rg.ljust(7), str(any_r['level']).rjust(3), any_r['split'].ljust(5)]
            for n in names:
                r = v.get(n)
                cells.append(('-' if r is None or not r['solved'] else str(r['probes'])).rjust(8))
            print(line(cells))


if __name__ == '__main__':
    main()
