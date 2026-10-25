"""exp_tables.py -- the tables of the paper, computed from a collected runs directory (exp_collect) and rendered as Markdown and LaTeX (booktabs).

    T = exp_tables.all_tables(C)            # {key: Table}; a table whose data is missing is left out
    T['main'].md()   T['main'].tex()

Every cell is computed here from the rows of the evaluations and the logs of the training runs; nothing is typed in.  A percentage with two or more replicates is `mean +- half-width` of the 95% Student-t
interval over replicates (training seeds); with ONE replicate (the model-free baselines) it is the Wilson interval over problems in parentheses (low–high); a t-interval that would leave [0, 100] (two or three
replicates on a handful of problems) is clipped to it and written in parentheses too.  Differences between methods are paired by seed; the p-values of one family of comparisons carry Holm's correction.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, math, collections
import numpy as np
import exp_stats as ST, exp_collect as EC

NAN = float('nan')

LABELS = collections.OrderedDict([
    ('uniform', 'Uniform random actions'), ('blind', 'Blind search (IDA*, engine costs)'),
    ('pre', 'Pre-trained on synthetic theorems only'), ('bc_scratch', 'BC on olympiad data only'), ('bc', 'BC: pre-trained + olympiad'), ('bc_mw', 'BC (manager-worker labels)'),
    ('ppo', 'PPO'), ('ppo_nosub', 'PPO without sub-theorem goals'),
    ('ei', 'Expert iteration'), ('ei_her', 'Expert iteration + hindsight'), ('ei_her_boot', 'Expert iteration + hindsight + bootstrapped library'),
    ('ei_mw', 'Expert iteration, manager-worker actor'), ('ei_mcts', 'Expert iteration, MCTS actor'), ('ei_ida', 'Expert iteration, IDA* actor')])
MAIN_ORDER = ['pre', 'bc_scratch', 'bc', 'ppo', 'ppo_nosub', 'ei', 'ei_her', 'ei_her_boot', 'ei_mw', 'ei_mcts', 'ei_ida']

ABLATIONS = [      # (key, label, a, b, group_a, group_b): "a minus b"
    ('pretrain', 'Synthetic pre-training (BC vs BC from scratch)', 'bc', 'bc_scratch'),
    ('finetune', 'Olympiad fine-tuning (BC vs pre-trained only)', 'bc', 'pre'),
    ('ei', 'Expert iteration vs its starting point (BC)', 'ei', 'bc'),
    ('ppo', 'PPO vs its starting point (BC)', 'ppo', 'bc'),
    ('her', 'Hindsight relabelling (EI + HER vs EI)', 'ei_her', 'ei'),
    ('boot', 'Bootstrapped library (EI + HER + library vs EI + HER)', 'ei_her_boot', 'ei_her'),
    ('sub', 'Sub-theorem goals for PPO (with vs without)', 'ppo', 'ppo_nosub'),
    ('algo', 'Expert iteration + HER vs PPO', 'ei_her', 'ppo'),
    ('mw', 'Manager-worker actor vs flat actor (EI + HER)', 'ei_mw', 'ei_her'),
    ('mcts', 'MCTS actor vs sampling actor (EI + HER)', 'ei_mcts', 'ei_her'),
    ('ida', 'IDA* actor vs sampling actor (EI + HER)', 'ei_ida', 'ei_her')]


class Table:
    def __init__(self, key, caption, header, rows, notes=(), align=None):
        self.key = key; self.caption = caption; self.header = list(header); self.rows = [list(r) for r in rows]; self.notes = list(notes)
        self.align = align or ('l' + 'r' * (len(header) - 1))
    def md(self):
        L = ['**%s**' % self.caption, '', '| ' + ' | '.join(self.header) + ' |', '|' + '|'.join((':---' if a == 'l' else '---:') for a in self.align) + '|']
        for r in self.rows: L.append('| ' + ' | '.join(str(c) for c in r) + ' |')
        if self.notes: L += [''] + ['*%s*' % n for n in self.notes]
        return '\n'.join(L) + '\n'
    def column_widths(self):
        """relative widths of the columns from their content: the longest unbreakable word of a column is its minimum, the longest cell (capped) its preference"""
        w = []
        for k in range(len(self.header)):
            cells = [str(self.header[k])] + [str(r[k]) for r in self.rows if k < len(r)]
            words = [c.replace(' ± ', '±') for c in cells]                                    # (a "33.3 ± 0.0" cell is written with ties around the sign: it does not break, so it is one word)
            lo = max((len(t) for c in words for t in c.split()), default=1); hi = min(max((len(c) for c in cells), default=1), 36)
            w.append(max(lo + 1.5, 0.5 * hi))
        return w
    def tex(self):
        def esc(s):
            s = str(s).replace(' ± ', '\x00').replace('±', r'$\pm$').replace('Δ', r'$\Delta$').replace('%', r'\%').replace('&', r'\&').replace('_', r'\_').replace('≤', r'$\leq$').replace('×', r'$\times$').replace('~', r'$\sim$').replace('–', '--')
            return s.replace('\x00', r'~$\pm$~')                                                      # (a "mean ± half-width" cell is tied around the sign: it never breaks)
        w = self.column_widths(); tot = sum(w);
        size = r'\scriptsize' if tot > 95 else (r'\footnotesize' if tot > 70 else r'\small')
        spec = ''.join((r'>{\raggedright\arraybackslash}' if (a == 'l' or k == 0) else r'>{\raggedleft\arraybackslash}') + r'p{\dimexpr %.4f\linewidth-2\tabcolsep\relax}' % (w[k] / tot)
                       for k, a in enumerate(self.align))
        L = [r'\begin{table}[t]', r'\centering', size, r'\caption{%s}\label{tab:%s}' % (esc(self.caption), self.key), r'\begin{tabular}{%s}' % spec, r'\toprule',
             ' & '.join(esc(h) for h in self.header) + r' \\', r'\midrule']
        for r in self.rows: L.append(' & '.join(esc(c) for c in r) + r' \\')
        L += [r'\bottomrule', r'\end{tabular}']
        if self.notes: L.append(r'\par\vspace{2pt}{\footnotesize\raggedright ' + ' '.join(esc(n) for n in self.notes) + r'\par}')
        L.append(r'\end{table}')
        return '\n'.join(L) + '\n'
    def to_dict(self): return {'key': self.key, 'caption': self.caption, 'header': self.header, 'rows': self.rows, 'notes': self.notes}


def pct(m, lo, hi, n, digits=1):
    if m is None or (isinstance(m, float) and math.isnan(m)): return '--'
    f = '%%.%df' % digits
    if n >= 2 and not math.isnan(lo):
        if lo < 0.0 or hi > 1.0: return ('%s (%.0f–%.0f)' % (f, 100 * max(0.0, lo), 100 * min(1.0, hi))) % (100 * m)           # (a t-interval that leaves [0, 100] -- few replicates, few problems -- is clipped and written like the Wilson interval)
        return (f + ' ± ' + f) % (100 * m, 50 * (hi - lo))
    if n == 1 and not math.isnan(lo): return ('%s (%.0f–%.0f)' % (f, 100 * lo, 100 * hi)) % (100 * m)
    return f % (100 * m)


def num(x, digits=1):
    return '--' if x is None or (isinstance(x, float) and math.isnan(x)) else ('%%.%df' % digits) % x


def mean_of(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.mean(xs)) if xs else NAN


def pval(p):
    if p is None or math.isnan(p): return '--'
    return '<0.001' if p < 0.001 else '%.3f' % p


def pp(d, digits=1):
    """a difference of fractions as percentage points with mean [low, high]"""
    if d is None or math.isnan(d['mean']): return '--'
    if math.isnan(d.get('low', NAN)): return '%+.*f' % (digits, 100 * d['mean'])
    return '%+.*f [%+.*f, %+.*f]' % (digits, 100 * d['mean'], digits, 100 * d['low'], digits, 100 * d['high'])


def _cell(C, group, variant, solver, split, budget, subset):
    m, lo, hi, n = C.rate_ci(group, variant, solver, split, budget, subset)
    return pct(m, lo, hi, n), n


def _find(C, variant, solver='sample'):
    """the evaluation group that holds (variant, solver): main first, then the others; None when absent"""
    for g in ('baselines', 'main', 'solvers', 'mw', 'scale'):
        if variant in C.variants(g) and solver in C.solvers(g, variant): return g
    return None


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ tables
def main_table(C, P):
    B = P.get('headline'); rows = []
    spec = [('baselines', 'baselines', 'uniform', 'uniform'), ('baselines', 'baselines', 'blind', 'blind')] + [('main', v, 'sample', v) for v in MAIN_ORDER]
    for group, variant, solver, key in spec:
        if variant not in C.variants(group) or solver not in C.solvers(group, variant): continue
        dev, _ = _cell(C, group, variant, solver, 'dev', B, 'all'); test, n = _cell(C, group, variant, solver, 'test', B, 'all')
        tp, _ = _cell(C, group, variant, solver, 'test', B, 'proved'); tu, _ = _cell(C, group, variant, solver, 'test', B, 'unproved'); tf, _ = _cell(C, group, variant, solver, 'test', B, 'free')
        pr, ln = C.efficiency(group, variant, solver, 'test', B)
        rows.append([LABELS.get(key, key), n, dev, test, tp, tu, tf, num(mean_of(pr), 0), num(mean_of(ln), 1)])
    if not rows: return None
    nt = len(C.names('test', 'all')); nd = len(C.names('dev', 'all')); ntp = len(C.names('test', 'proved')); ntu = len(C.names('test', 'unproved'))
    return Table('main', 'Problems solved (%%) within %s probes per problem: dev (%d problems) and test (%d problems; %d with a reference proof in the bank, %d statement-only).' % (B, nd, nt, ntp, ntu),
                 ['Method', 'seeds', 'dev', 'test', 'test, proved', 'test, statement-only', 'test, no helpers', 'probes / solved', 'steps / proof'], rows,
                 notes=['mean ± half-width of the 95% t-interval over seeds; a single run (the model-free baselines) shows its Wilson interval over problems in parentheses, and so does a t-interval that would leave [0, 100] (clipped).  "Solved" means a proof that the engine replayed and the kernel checked; '
                        'probes and steps are means over the problems a method solved (so a method that solves more, harder problems can need more).  The test split was not used to choose anything.'],
                 align='lrrrrrrrr')


def strict_table(C, P, variants=('bc_scratch', 'bc', 'ppo', 'ei_her', 'ei_her_boot')):
    """the problems whose helper lemmas can be hidden: the same models with the helpers on the menu (prefix) and hidden (strict)"""
    S = C.view('strict'); B = P.get('headline')
    names = sorted({r['name'] for r in S.rows if r['group'] == 'main' and r.get('split') in ('dev', 'test')})
    if not names:                                       # (a world without such problems -- a smoke run -- has no strict task; the table says so, so that the paper's reference to it resolves)
        return Table('strict', 'Dev and test problems whose helper lemmas can be hidden: there are none in this run.',
                     ['Method', 'prefix (helpers available)', 'strict (helpers hidden)'], [['--', '--', '--']],
                     notes=['No dev or test problem of this run has helper lemmas, so the prefix and the strict regime are the same task and there is nothing to compare.'], align='lrr')
    rows = []
    for v in variants:
        if v not in C.variants('main'): continue
        cells = []
        for view in (C, S):
            M = view.success('main', v, 'sample', 'dev+test', B, 'all', names=names); r = M.rates(); r = r[~np.isnan(r)]
            cells.append(pct(*(ST.mean_ci(r) + (len(r),)), digits=1) if len(r) >= 2 else (pct(float(r[0]), NAN, NAN, 1) if len(r) == 1 else '--'))
        rows.append([LABELS.get(v, v), cells[0], cells[1]])
    if not rows: return None
    return Table('strict', 'The %d dev and test problems whose helper lemmas can be hidden: solved (%%) within %s probes with the helpers on the menu (prefix regime) and hidden (strict regime).' % (len(names), B),
                 ['Method', 'prefix (helpers available)', 'strict (helpers hidden)'], rows,
                 notes=['Strict is the harder task: the lemmas the reference proof builds its argument on are not given.  Problems without helper lemmas are the same task in both regimes and are not in this table.'], align='lrr')


def ablation_table(C, P, split='test'):
    B = P.get('headline'); rows = []; ps = []
    items = []
    for key, label, a, b in ABLATIONS:
        ga, gb = _find(C, a), _find(C, b)
        if ga is None or gb is None: continue
        cmp_t = C.compare(ga, a, b, 'sample', split, B, 'all', group_b=gb)
        cmp_d = C.compare(ga, a, b, 'sample', 'dev', B, 'all', group_b=gb)
        items.append((label, cmp_t, cmp_d)); ps.append(cmp_t['diff']['p'])
    if not items: return None
    adj = ST.holm(ps)
    for (label, ct, cd), padj in zip(items, adj):
        rows.append([label, ct['diff']['n'], pp(ct['diff']), pval(ct['diff']['p']), pval(padj), pp(ct['problem']), pp(cd['diff'])])
    return Table('ablations', 'Ablations: difference in problems solved (percentage points) at %s probes, first minus second; %s split unless stated.' % (B, split),
                 ['Comparison', 'pairs', 'Δ test, over seeds [95% CI]', 'p (paired t)', 'p (Holm)', 'Δ test, over problems [95% CI]', 'Δ dev, over seeds'], rows,
                 notes=['Seeds are paired (learners that continue from BC share its checkpoint of the same seed); the first interval is the t-interval of the per-seed differences, the second a bootstrap over problems '
                        '(the same resampled problems in both methods).  With few seeds the first interval is wide; the second ignores training noise.  Holm\'s correction covers this family of comparisons.'],
                 align='lrrrrrr')


def solver_table(C, P, group, key, caption):
    B = P.get('headline'); rows = []
    for v in C.variants(group):
        for s in C.solvers(group, v):
            dev, _ = _cell(C, group, v, s, 'dev', B, 'all'); test, n = _cell(C, group, v, s, 'test', B, 'all')
            pr, ln = C.efficiency(group, v, s, 'test', B)
            rows.append([LABELS.get(v, v), s, n, dev, test, num(mean_of(pr), 0), num(mean_of(ln), 1), num(C.seconds_per_problem(group, v, s, 'test'), 1)])
    if not rows: return None
    return Table(key, caption % B, ['Network', 'Solver', 'seeds', 'dev', 'test', 'probes / solved', 'steps / proof', 'seconds / problem'], rows, align='llrrrrrr')


def scaling_table(C, P):
    sizes = sorted(set(P.get('synth_sizes', ())) - {0}); smax = max(sizes) if sizes else None
    B = P.get('headline'); rows = []
    def add(n, group, variant, label):
        if variant not in C.variants(group): return
        dev, k = _cell(C, group, variant, 'sample', 'dev', B, 'all'); test, _ = _cell(C, group, variant, 'sample', 'test', B, 'all'); tp, _ = _cell(C, group, variant, 'sample', 'test', B, 'proved')
        u = C.synth_used(n) if n else None
        rows.append([n if (u is None or u == n) else '%d (%d used)' % (n, u), label, k, dev, test, tp])
    add(0, 'main', 'bc_scratch', 'BC, no pre-training')
    for n in sizes:
        if n == smax: add(n, 'main', 'bc', 'BC, pre-trained + olympiad')
        else: add(n, 'scale', 'bc_n%d' % n, 'BC, pre-trained + olympiad')
    if len(rows) < 2: return None
    if 'pre' in C.variants('main') and smax:
        dev, k = _cell(C, 'main', 'pre', 'sample', 'dev', B, 'all'); test, _ = _cell(C, 'main', 'pre', 'sample', 'test', B, 'all'); tp, _ = _cell(C, 'main', 'pre', 'sample', 'test', B, 'proved')
        rows.append([smax, 'pre-trained only (zero-shot, no olympiad data)', k, dev, test, tp])
    return Table('scaling', 'Olympiad performance (%% solved within %s probes) against the number of synthetic theorems used for pre-training.' % B, ['synthetic theorems', 'model', 'seeds', 'dev', 'test', 'test, proved'], rows, align='rlrrrr')


def level_table(C, P, variants=('bc_scratch', 'bc', 'ppo', 'ei_her', 'ei_her_boot')):
    B = P.get('headline'); levels = sorted({C.info[n]['level'] for n in C.names('test', 'all') if C.info.get(n, {}).get('level') is not None})
    srcs = ['IMO', 'ISL', 'AIME', 'AMC', 'APMO', 'folklore', 'other']
    present = [v for v in variants if v in C.variants('main')]
    if not present or not levels: return None
    def cell(v, names):
        if not names: return '--'
        M = C.success('main', v, 'sample', 'test', B, 'all', names=names); r = M.rates(); r = r[~np.isnan(r)]
        if len(r) >= 2: m, lo, hi = ST.mean_ci(r); return pct(m, lo, hi, len(r))
        if len(r) == 1: return pct(float(r[0]), NAN, NAN, 1)
        return '--'
    rows = []
    for l in levels:
        names = [n for n in C.names('test', 'all') if C.info[n]['level'] == l]
        rows.append(['level %s' % l, len(names)] + [cell(v, names) for v in present])
    for s in srcs:
        names = [n for n in C.names('test', 'all') if EC.source_class(C.info[n]['source']) == s]
        if names: rows.append([s, len(names)] + [cell(v, names) for v in present])
    return Table('levels', 'Test problems solved (%%) within %s probes by level of the bank (1 = one routine step ... 5 = a hard olympiad problem) and by the contest the problem comes from.' % B, ['group', 'problems'] + [LABELS.get(v, v) for v in present], rows,
                 notes=['A row with few problems has an interval wider than its entry shows; read it as a profile, not a ranking.'], align='lr' + 'r' * len(present))


def bank_table(C):
    """the benchmark itself: problems by contest family, with or without a reference proof, by split; then by level and by provenance"""
    probs = C.bank
    if not probs: return None
    classes = ['IMO', 'ISL', 'AIME', 'AMC', 'APMO', 'folklore', 'other']
    def row(label, ps):
        c = collections.Counter(); 
        for p in ps: c['proved' if 'demonstration' in p else 'statement'] += 1; c[p.get('split', '?')] += 1
        return [label, len(ps), c['proved'], c['statement'], c['train'], c['dev'], c['test']]
    rows = []
    for k in classes:
        ps = [p for p in probs if EC.source_class(p.get('source')) == k]
        if ps: rows.append(row(k, ps))
    rows.append(row('all problems', probs))
    for l in sorted({p.get('level') for p in probs if p.get('level') is not None}): rows.append(row('level %s' % l, [p for p in probs if p.get('level') == l]))
    for field, values, title in (('origin', ('contest', 'adapted', 'style', 'folklore'), 'origin'), ('verified', ('answer-checked', 'recalled', 'none'), 'checked against a public source')):
        if any(p.get(field) for p in probs):
            for v in values:
                ps = [p for p in probs if p.get(field) == v]
                if ps: rows.append(row('%s: %s' % (title, v), ps))
    return Table('bank', 'The benchmark: formal problems by contest family, level and provenance, with a kernel-checked reference proof or the statement only, and their split (train / dev / test).',
                 ['', 'problems', 'with proof', 'statement only', 'train', 'dev', 'test'], rows,
                 notes=['The split is by problem family (a hash of the family, so variants of one problem are never separated); origin: a problem of a named contest paraphrased (contest), a named problem changed to fit the vocabulary (adapted), '
                        'written for the bank in the manner of a contest (style), or a classical fact (folklore); "checked" means the statement or answer was compared with a public source.'],
                 align='lrrrrrr')


def system_table(C):
    r = C.repo or {}
    if not r.get('kernel_lines'): return None
    rows = [['trusted kernel (C++23, header)', '%d lines' % r['kernel_lines']], ['Python bindings (nanobind)', '%d lines' % r.get('bindings_lines', 0)],
            ['independent checker (Python, written from the typing rules)', '%d lines' % r.get('checker_lines', 0)],
            ['untrusted Python: authoring, libraries, gym, bank, generator, learner', '%d modules, %d lines' % (r.get('python_modules', 0), r.get('python_lines', 0))],
            ['gate scripts (tests of the kernel, tactics, gym, bank, generator, learner)', '%d scripts, %d lines' % (r.get('gates', 0), r.get('test_lines', 0))],
            ['mutants of the code in the catalogue (%d listed as equivalent to the original)' % r.get('mutants_equivalent', 0), r.get('mutants', '--')]]
    if r.get('library_lemmas'): rows.append(['library lemmas whose proofs are recorded as actions (the corpus the learner imitates)', r['library_lemmas']])
    if r.get('subtheorems'): rows.append(['hindsight sub-theorem goals extracted from the reference proofs', r['subtheorems']])
    rows.append(['commit', (r.get('commit') or '?')[:10] + (' (modified)' if r.get('dirty') else '')])
    return Table('system', 'The artefact in numbers (counted from the repository).', ['', 'value'], rows, notes=['Lines are physical lines of the files, comments and blank lines included.'], align='lr')


HYPER = [      # (profile key, description)
    ('seeds', 'training seeds (replicates)'), ('synth_target', 'synthetic train theorems (the largest pre-training pool)'), ('synth_attempts', 'generator attempts'), ('synth_sizes', 'synthetic theorems used for pre-training (scaling study)'),
    ('pre_epochs', 'pre-training epochs'), ('pre_batch', 'batch size (decisions)'), ('ft_epochs', 'fine-tuning epochs (after pre-training)'), ('ft_lr', 'fine-tuning peak learning rate'), ('scratch_epochs', 'epochs from scratch (no pre-training)'),
    ('rl_iters', 'expert-iteration iterations'), ('rl_episodes', 'tasks per iteration'), ('rl_probes', 'probe budget per task during training'), ('ei_updates', 'gradient steps per iteration'), ('boot_per_iter', 'library lemmas admitted per iteration at most'),
    ('ppo_iters', 'PPO iterations'), ('ppo_episodes', 'PPO episodes per iteration'),
    ('eval_probes', 'largest probe budget at evaluation'), ('headline', 'budget of the tables'), ('eval_budgets', 'budgets of the curves'), ('eval_time_limit', 'time limit per problem (s)'), ('eval_splits', 'evaluated splits'),
    ('device', 'device of the training runs'), ('workers', 'worker processes per run'), ('threads', 'torch threads per process')]


def hyper_table(C):
    P = C.profile or {}
    if not P: return None
    def show(v):
        if isinstance(v, (list, tuple)): return ', '.join(str(x) for x in v)
        return str(v)
    rows = [[d, show(P[k])] for k, d in HYPER if k in P]
    rows.insert(0, ['profile', str(P.get('name'))]); rows.insert(1, ['seed list', show(P.get('seed_list') or [])])
    return Table('hyper', 'Configuration of the experiments (the profile of the runs directory, exp_spec.py).', ['setting', 'value'], rows, align='ll')


def generator_table(C):
    st = (C.synth or {}).get('stats'); man = (C.synth or {}).get('manifest')
    if not st or not st.get('total'): return None
    rows = []
    cl = st['classes']
    rows.append(['theorems (train / dev / test iid / test comp)', ' / '.join(str(st['n'][k]) for k in ('train', 'dev', 'test_iid', 'test_comp'))])
    rows.append(['mean proof length (steps), mean depth', '%.1f, %.1f' % (st['proofs']['mean_steps'], st['proofs']['mean_depth'])])
    rows.append(['share closed at once by a decision procedure', '%.1f%%' % (100 * st['proofs']['closed_at_once_share'])])
    k = st['kinds']['per_kind']
    rows.append(['action kinds used (share of steps)', ', '.join('%s %.0f%%' % (n, 100 * v['share_of_steps']) for n, v in list(k.items())[:6]) + ('' if st['kinds']['distinct_kinds'] <= 6 else ', ...')])
    rows.append(['library lemmas applied (effective number by entropy)', '%d (%.0f)' % (st['lemmas']['distinct_applied'], st['lemmas']['effective_lemmas'])])
    for name, key in (('proof skeletons', 'skeleton'), ('statement shapes', 'shape'), ('recipes (lemma sets)', 'recipe')):
        c = cl[key]
        rows.append(['distinct %s: observed / singletons / Chao1 estimate' % name, '%d / %d / %.0f ± %.0f' % (c['observed'], c['singletons'], c['chao1'], c['chao1_se'])])
    rows.append(['Good-Turing coverage of skeletons (chance a new theorem repeats a seen skeleton)', '%.3f' % cl['skeleton']['good_turing_coverage']])
    if man:
        rows.append(['attempts, kept after merge', '%s, %s' % (man.get('attempts'), man.get('records'))])
        ex = man.get('exclusions') or {}
        rows.append(['excluded as a bank problem / library lemma / sub-theorem (digests), bank shapes', '%s, %s' % (ex.get('digests'), ex.get('bank_shapes'))])
    return Table('generator', 'The synthetic theorem set: composition and saturation.', ['quantity', 'value'], rows, align='lr')


def library_table(C, P):
    runs = {rid: f for rid, f in C.finals.items() if rid.startswith('rl/ei_her_boot/')}
    if not runs: return None
    rows = []
    for rid in sorted(runs):
        f = runs[rid]; lib = f.get('library') or {}; lf = f.get('lemma_file') or {}
        rows.append([rid.split('/')[-1], lib.get('installed', lf.get('records', '--')), num(lib.get('mean_length'), 1), ', '.join('%s %d' % kv for kv in sorted((lib.get('origins') or {}).items(), key=lambda kv: -kv[1])[:3]) or '--',
                     sum((lib.get('rejected') or {}).values()) if lib else '--'])
    # use of the lemmas at test time
    used = tot = 0
    for r in C.rows:
        if r['variant'] == 'ei_her_boot' and r['group'] == 'main' and r.get('solved') and r.get('split') == 'test':
            tot += 1; used += 1 if r.get('boot_uses') else 0
    notes = ['%d of %d proofs found for test problems by the library-equipped agent apply at least one bootstrapped lemma.' % (used, tot) if tot else 'No test problem was solved by the library-equipped agent.']
    return Table('library', 'Bootstrapped lemma library: lemmas admitted per run (train problems only, each kernel-checked and replayed from the library alone).', ['seed', 'lemmas', 'mean length', 'main origins', 'rejected'], rows, notes=notes, align='lrrlr')


def cost_table(C, P):
    by = collections.OrderedDict()
    for rid, f in sorted(C.finals.items()):
        parts = rid.split('/'); stage, variant = parts[0], parts[1] if stage_ok(parts) else None
        if variant is None: continue
        key = (stage, variant)
        by.setdefault(key, []).append(f)
    rows = []
    for (stage, variant), fs in by.items():
        secs = [f.get('seconds') for f in fs if f.get('seconds') is not None]; rss = [f.get('peak_rss_mb') for f in fs if f.get('peak_rss_mb') is not None]
        par = [f.get('n_params') for f in fs if f.get('n_params')]
        extra = ''
        if stage in ('ft', 'pre'):
            logs = [C.logs.get('%s/%s/s%s' % (stage, variant, s)) for s in range(0, 20)]
            sps = [r.get('steps_per_second') for lg in logs if lg for r in lg if r.get('steps_per_second')]
            extra = '%s steps/s' % num(mean_of(sps), 0) if sps else ''
        else:
            cp = []
            for rid2, lg in C.logs.items():
                if rid2.startswith('%s/%s/' % (stage, variant)) and lg and lg[-1].get('cum_probes') and lg[-1].get('seconds'): cp.append(lg[-1]['cum_probes'] / max(lg[-1]['seconds'], 1e-9))
            extra = '%s probes/s' % num(mean_of(cp), 1) if cp else ''
        rows.append(['%s/%s' % (stage, variant), len(fs), num(mean_of(secs) / 60.0 if secs else NAN, 1), num(mean_of(rss) / 1024.0 if rss else NAN, 2), num(mean_of(par) / 1e6 if par else NAN, 2), extra])
    if not rows: return None
    return Table('cost', 'Training cost per run (means over seeds): wall-clock minutes, peak resident memory, parameters, throughput.', ['run', 'runs', 'minutes', 'peak RSS (GB)', 'parameters (M)', 'throughput'], rows, align='lrrrrr')


def stage_ok(parts): return len(parts) >= 3 and parts[0] in ('pre', 'ft', 'rl')


def soundness_table(C):
    chk = sum(v['checked'] for v in C.recheck.values()); acc = sum(v['accepted'] for v in C.recheck.values())
    allr = getattr(C, 'all_rows', C.rows)
    solved = sum(1 for r in allr if r.get('solved')); err = sum(1 for r in allr if r.get('error'))
    noproof = sum(1 for r in allr if r.get('solved') and not r.get('proof'))
    rows = [['rows (task x method x seed) evaluated', len(allr)], ['rows with a kernel-checked proof', solved], ['rows that raised an error in the solver', err],
            ['solved rows without a portable proof (they cannot be replayed or re-checked)', noproof],
            ['proofs given to the independent checker', chk], ['accepted by the independent checker', acc], ['rejected', chk - acc]]
    return Table('soundness', 'Soundness of the evaluation.', ['quantity', 'count'], rows, notes=['A row counts as solved only after the engine replayed its proof and the kernel accepted the term; the independently written checker (recheck.py) re-checks the same terms.'], align='lr')


def throughput_tables(C):
    b = (C.bench or {}).get('throughput')
    if not b: return {}
    out = {}
    if b.get('solvers'):
        rows = [[s, num(v.get('probes_per_s'), 1), num(v.get('decisions_per_s'), 1), num(v.get('actions_per_s'), 1), num(v.get('seconds_per_problem'), 2), num(v.get('peak_rss_mb', NAN) / 1024.0 if v.get('peak_rss_mb') else NAN, 2)]
                for s, v in b['solvers'].items()]
        out['bench_solvers'] = Table('bench_solvers', 'Engine throughput on %d train problems (one core): probes (tries of an action), decisions of the policy, actions per second.' % b.get('n_problems', 0),
                                     ['solver', 'probes / s', 'decisions / s', 'actions / s', 'seconds / problem', 'peak RSS (GB)'], rows, align='lrrrrr')
    if b.get('workers'):
        rows = [[w, num(v.get('probes_per_s'), 1), num(v.get('speedup'), 2), num(v.get('efficiency'), 2)] for w, v in sorted(b['workers'].items(), key=lambda kv: int(kv[0]))]
        out['bench_workers'] = Table('bench_workers', 'Scaling of the evaluation with worker processes (the world is shared by fork).', ['workers', 'probes / s', 'speed-up', 'efficiency'], rows, align='rrrr')
    if b.get('world'):
        w = b['world']
        rows = [['world build (the whole bank)', num(w.get('seconds'), 1) + ' s'], ['globals in the world', w.get('globals')], ['resident memory after the build', num(w.get('rss_mb', NAN) / 1024.0, 2) + ' GB'],
                ['policy network parameters', num(w.get('params', NAN) / 1e6, 2) + ' M'], ['BC training steps / s (batch)', num(w.get('train_steps_per_s'), 0)], ['network inference decisions / s (batch 1)', num(w.get('infer_decisions_per_s'), 0)]]
        out['bench_world'] = Table('bench_world', 'Fixed costs.', ['quantity', 'value'], rows, align='lr')
    k = (C.bench or {}).get('kernel')
    if k and isinstance(k, dict):
        rows = []
        for name, series in k.items():
            if not isinstance(series, dict): continue
            pts = [(int(n), v) for n, v in series.items() if v is not None]
            if pts:
                n, ms = max(pts); rows.append([name, n, num(ms, 2)])
        if rows: out['bench_kernel'] = Table('bench_kernel', 'Kernel micro-benchmarks (largest size within the time budget): milliseconds for one call.', ['case', 'size n', 'ms'], rows, align='lrr')
    return out


def discoveries(C, P):
    """statement-only problems of any split that some evaluated run solved: [(name, split, variant, solver, seed, probes, steps)] (new, kernel-checked proofs)"""
    out = {}
    for r in getattr(C, 'all_rows', C.rows):
        if r.get('solved') and not C.info.get(r['name'], {}).get('proved', True):
            key = r['name']; cur = out.get(key)
            if cur is None or r['probes'] < cur[5]: out[key] = (r['name'], r.get('split'), r['variant'], r['solver'], r['seed'], r['probes'], r.get('steps'))
    return sorted(out.values())


def all_tables(C):
    P = C.profile or {}
    T = collections.OrderedDict()
    for key, t in (('system', system_table(C)), ('bank', bank_table(C)), ('main', main_table(C, P)), ('strict', strict_table(C, P)), ('ablations', ablation_table(C, P)),
                   ('solvers', solver_table(C, P, 'solvers', 'solvers', 'The same networks under different provers: problems solved (%%) within %s probes.')),
                   ('mw', solver_table(C, P, 'mw', 'mw', 'The manager-worker agent and its ablations: problems solved (%%) within %s probes.')),
                   ('scaling', scaling_table(C, P)), ('levels', level_table(C, P)), ('generator', generator_table(C)), ('library', library_table(C, P)), ('cost', cost_table(C, P)), ('soundness', soundness_table(C)), ('hyper', hyper_table(C))):
        if t is not None: T[key] = t
    for k, t in throughput_tables(C).items(): T[k] = t
    return T
