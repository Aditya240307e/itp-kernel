"""exp_figures.py -- the figures of the paper, from a collected runs directory (exp_collect).

    paths = exp_figures.make_all(C, 'runs/full/report/figures')        # {name: [pdf path, png path]}; a figure whose data is missing is left out

Conventions (dataviz): one hue per FAMILY of methods (behaviour cloning blue, PPO aqua, expert iteration orange, and one hue each for the manager-worker, MCTS and IDA* actors), variants inside a family
told apart by line style and marker; model-free baselines in neutral grey; intervals are the 95% t-interval over training seeds as a light band (Wilson over problems for a single run); every curve has a
direct label at its end and the legend repeats it, so no figure depends on colour alone; thin lines, markers of at least 5 pt, recessive grid; logarithmic axes where the quantity is a budget or a size.
Vector PDF for the paper and PNG for the report.  Matplotlib is imported when a figure is made (the rest of the pipeline does not need it).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, math, collections
import numpy as np
import exp_stats as ST, exp_collect as EC, exp_tables as TB

NAN = float('nan')
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#ffffff'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'
GREY, GREY2 = '#52514e', '#9a998f'

# variant -> (colour, line style, marker, label)
STYLE = collections.OrderedDict([
    ('uniform', (GREY2, (0, (1, 1.5)), 'x', 'uniform random')), ('blind', (GREY, (0, (4, 2)), '+', 'blind search')),
    ('pre', (BLUE, (0, (1, 1.5)), 'o', 'pre-trained only')), ('bc_scratch', (BLUE, (0, (4, 2)), 'o', 'BC from scratch')), ('bc', (BLUE, '-', 'o', 'BC')), ('bc_mw', (BLUE, (0, (3, 1, 1, 1)), 'o', 'BC (mw labels)'),),
    ('ppo', (AQUA, '-', 's', 'PPO')), ('ppo_nosub', (AQUA, (0, (4, 2)), 's', 'PPO, no sub-goals')),
    ('ei', (ORANGE, (0, (4, 2)), '^', 'EI')), ('ei_her', (ORANGE, '-', '^', 'EI + HER')), ('ei_her_boot', (ORANGE, (0, (1, 1.5)), '^', 'EI + HER + library')),
    ('ei_mw', (VIOLET, '-', 'D', 'EI, manager-worker')), ('ei_mcts', (MAGENTA, '-', 'v', 'EI, MCTS actor')), ('ei_ida', (YELLOW, '-', 'P', 'EI, IDA* actor'))])
SOLVER_STYLE = collections.OrderedDict([('sample', (BLUE, '-', 'o', 'sample')), ('greedy', (GREY, (0, (4, 2)), '+', 'greedy')), ('ida', (ORANGE, '-', '^', 'IDA*')), ('mcts', (VIOLET, '-', 'D', 'MCTS (rollouts)')),
                                        ('mcts:value=critic', (MAGENTA, '-', 'v', 'MCTS (critic)')), ('mw', (AQUA, '-', 's', 'manager-worker')), ('mw:gate=never', (AQUA, (0, (4, 2)), 's', 'mw, no claims')),
                                        ('mw:gate=fixed', (YELLOW, (0, (4, 2)), 'P', 'mw, fixed gate'))])


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6, 'lines.markersize': 5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 170, 'figure.dpi': 110})
    return plt


def save(plt, fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, '%s.%s' % (name, ext)); fig.savefig(p, bbox_inches='tight'); paths.append(p)
    plt.close(fig)
    return paths


def _finite(xs): return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _dodge(ys, gap):
    """y positions for direct labels: sorted, pushed apart by at least `gap` (in the same units), keeping the order"""
    order = sorted(range(len(ys)), key=lambda i: ys[i]); out = list(ys)
    for a, i in enumerate(order[1:], 1):
        prev = order[a - 1]
        if out[i] - out[prev] < gap: out[i] = out[prev] + gap
    return out


def end_labels(ax, items, xpad=0.0):
    """items: [(x, y, text, colour)] at the right end of each curve; placed in the margin, pushed apart"""
    if not items: return
    ymin, ymax = ax.get_ylim(); gap = (ymax - ymin) * 0.055
    ys = _dodge([y for _, y, _, _ in items], gap)
    for (x, y, text, col), yy in zip(items, ys):
        ax.annotate(text, xy=(x, y), xytext=(4 + xpad, 0), textcoords='offset points', va='center', ha='left', fontsize=6.8, color=INK2, annotation_clip=False,
                    arrowprops=None)
        ax.plot([x], [y], marker='', linestyle='')


def band(ax, x, mean, low, high, colour, ls, marker, label, ms=4.5, alpha=0.14, zorder=3):
    x = np.asarray(x, float); m = np.asarray(mean, float); lo = np.asarray(low, float); hi = np.asarray(high, float)
    ok = ~np.isnan(m)
    ax.plot(x[ok], m[ok], color=colour, linestyle=ls, marker=marker, markersize=ms, label=label, zorder=zorder, markeredgecolor=SURFACE, markeredgewidth=0.6)
    ok2 = ok & ~np.isnan(lo) & ~np.isnan(hi)
    if ok2.any(): ax.fill_between(x[ok2], np.clip(lo[ok2], 0.0, 1.0), np.clip(hi[ok2], 0.0, 1.0), color=colour, alpha=alpha, linewidth=0, zorder=zorder - 1)       # (a rate lives in [0, 1]; with two seeds the t-interval does not)


def pct_axis(ax, axis='y'):
    import matplotlib.ticker as mt
    f = mt.FuncFormatter(lambda v, _: '%d' % round(100 * v))
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(f)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ figures
def fig_budget(C, P, out, plt):
    budgets = [b for b in P.get('eval_budgets', ()) if b <= P.get('eval_probes', 1e18)]
    if not budgets: return None
    spec = [('baselines', 'uniform', 'uniform'), ('baselines', 'blind', 'blind')] + [('main', v, 'sample') for v in ('bc_scratch', 'bc', 'ppo', 'ei', 'ei_her', 'ei_her_boot', 'ei_mw')]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    drawn = 0
    for ax, split in zip(axes, ('dev', 'test')):
        ends = []
        for group, variant, solver in spec:
            if variant not in C.variants(group) or solver not in C.solvers(group, variant): continue
            key = solver if group == 'baselines' else variant
            colour, ls, mk, label = STYLE[key]
            cv = C.curve(group, variant, solver, split, budgets)
            band(ax, budgets, cv['mean'], cv['low'], cv['high'], colour, ls, mk, label); drawn += 1
            fin = _finite(cv['mean'])
            if fin: ends.append((budgets[-1], cv['mean'][len(cv['mean']) - 1 - [not math.isnan(v) for v in cv['mean'][::-1]].index(True)], label, colour))
        ax.set_xscale('log'); ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets], rotation=0); ax.minorticks_off()
        ax.set_xlabel('probes per problem (log scale)'); ax.set_title('%s split (%d problems)' % (split, len(C.names(split, 'all'))))
        ax.set_ylim(bottom=0); pct_axis(ax)
        end_labels(ax, ends)
    axes[0].set_ylabel('problems solved (%)')
    if not drawn: plt.close(fig); return None
    h, l = axes[1].get_legend_handles_labels()
    if h: fig.legend(h, l, loc='lower center', ncol=min(5, len(h)), frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); return save(plt, fig, out, 'fig_budget')


def fig_scaling(C, P, out, plt):
    sizes = sorted(set(P.get('synth_sizes', ())) - {0}); smax = max(sizes) if sizes else None
    if len(sizes) < 1: return None
    B = P.get('headline'); budgets = sorted({B, max(1, B // 4)})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    drawn = 0
    for ax, split in zip(axes, ('dev', 'test')):
        xs = [max(1, min(sizes) / 3.0)] + sizes   # N = 0 is drawn at the left, labelled 0
        for b, (ls, mk) in zip(budgets[::-1], (('-', 'o'), ((0, (4, 2)), 's'))):
            mean = []; low = []; high = []
            for n in [0] + sizes:
                if n == 0: g, v = 'main', 'bc_scratch'
                elif n == smax: g, v = 'main', 'bc'
                else: g, v = 'scale', 'bc_n%d' % n
                if v not in C.variants(g): mean.append(NAN); low.append(NAN); high.append(NAN); continue
                m, lo, hi, k = C.rate_ci(g, v, 'sample', split, b, 'all'); mean.append(m); low.append(lo); high.append(hi)
            if _finite(mean): band(ax, xs, mean, low, high, BLUE, ls, mk, 'within %d probes' % b); drawn += 1
        if 'pre' in C.variants('main'):
            m, lo, hi, k = C.rate_ci('main', 'pre', 'sample', split, B, 'all')
            if not math.isnan(m):
                ax.errorbar([xs[-1]], [m], yerr=[[m - lo if not math.isnan(lo) else 0], [hi - m if not math.isnan(hi) else 0]], fmt='X', color=GREY, markersize=6, capsize=2, label='pre-trained only (%d probes)' % B, zorder=4)
        ax.set_xscale('log'); ax.set_xticks(xs); ax.set_xticklabels(['0'] + [str(n) for n in sizes]); ax.minorticks_off()
        ax.set_xlabel('synthetic theorems used for pre-training'); ax.set_title('%s split' % split); ax.set_ylim(bottom=0); pct_axis(ax)
    axes[0].set_ylabel('problems solved (%)')
    if not drawn: plt.close(fig); return None
    h, l = axes[1].get_legend_handles_labels()
    if h: fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); return save(plt, fig, out, 'fig_scaling')


def fig_learning(C, P, out, plt):
    variants = [v for v in ('ppo', 'ppo_nosub', 'ei', 'ei_her', 'ei_her_boot', 'ei_mw', 'ei_mcts', 'ei_ida') if C.log_runs('rl', v)]
    if not variants: return None
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    drawn = 0
    for v in variants:
        colour, ls, mk, label = STYLE[v]
        for ax, xkey, ykey, norm in ((axes[0], 'iter', 'dev_solved', 'dev_total'), (axes[1], 'cum_probes', 'dev_solved', 'dev_total'), (axes[2], 'iter', 'dev_mean_len', None)):
            cv = C.log_curve('rl', v, xkey if xkey != 'iter' else 'iter', ykey, normalise=norm)
            if not cv or not cv['x']: continue
            band(ax, cv['x'], cv['mean'], cv['low'], cv['high'], colour, ls, mk, label, ms=4); drawn += 1
    if not drawn: plt.close(fig); return None
    axes[0].set_xlabel('iteration'); axes[0].set_ylabel('dev problems solved (%)'); pct_axis(axes[0]); axes[0].set_ylim(bottom=0)
    axes[1].set_xscale('log'); axes[1].set_xlabel('cumulative probes of rollouts (log)'); pct_axis(axes[1]); axes[1].set_ylim(bottom=0)
    axes[2].set_xlabel('iteration'); axes[2].set_ylabel('steps per dev proof (mean)'); axes[2].set_ylim(bottom=0)
    for ax, t in zip(axes, ('(a) by iteration', '(b) by compute', '(c) proof length')): ax.set_title(t)
    h, l = axes[0].get_legend_handles_labels()
    seen = {}
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()): seen.setdefault(ll, hh)
    fig.legend(list(seen.values()), list(seen.keys()), loc='lower center', ncol=min(4, len(seen)), frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(); return save(plt, fig, out, 'fig_learning')


def fig_bc(C, P, out, plt):
    sizes = sorted(set(P.get('synth_sizes', ())) - {0}); smax = max(sizes) if sizes else None
    series = []
    if smax and C.log_runs('pre', 'n%d' % smax): series.append(('pre', 'pre', 'n%d' % smax, 'dev_synth'))
    for v in ('bc_scratch', 'bc'):
        if C.log_runs('ft', v): series.append((v, 'ft', v, 'dev_bank'))
    if not series: return None
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6)); drawn = 0
    for key, stage, variant, dev in series:
        colour, ls, mk, label = STYLE[key]
        for ax, y in zip(axes, ('train_loss', ('%s' % dev, 'loss'), ('%s' % dev, 'action_acc'))):
            runs = C.log_runs(stage, variant)
            table = collections.defaultdict(list)
            for seed, lg in runs.items():
                for r in lg:
                    val = r.get(y) if isinstance(y, str) else (r.get(y[0]) or {}).get(y[1])
                    if val is not None: table[r['epoch']].append(float(val))
            if not table: continue
            ep = sorted(table); st = [ST.mean_ci(table[e]) for e in ep]
            band(ax, ep, [s[0] for s in st], [s[1] for s in st], [s[2] for s in st], colour, ls, mk, label, ms=3.5); drawn += 1
    if not drawn: plt.close(fig); return None
    for ax, t, yl in zip(axes, ('(a) training loss', '(b) held-out loss', '(c) held-out action accuracy'), ('loss', 'loss (dev)', 'accuracy')): ax.set_title(t); ax.set_xlabel('epoch'); ax.set_ylabel(yl)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(h), frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); return save(plt, fig, out, 'fig_bc')


def fig_ablations(C, P, out, plt):
    B = P.get('headline'); items = []
    for key, label, a, b in TB.ABLATIONS:
        ga, gb = TB._find(C, a), TB._find(C, b)
        if ga is None or gb is None: continue
        items.append((label, C.compare(ga, a, b, 'sample', 'test', B, 'all', group_b=gb)))
    if not items: return None
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(items) + 1.0))
    ys = list(range(len(items)))[::-1]
    for y, (label, c) in zip(ys, items):
        d = c['diff']; pb = c['problem']
        if not math.isnan(pb['mean']): ax.plot([100 * pb['low'], 100 * pb['high']], [y - 0.13] * 2, color=GREY2, linewidth=1.2, solid_capstyle='butt')
        if not math.isnan(d['mean']):
            if not math.isnan(d['low']): ax.plot([100 * d['low'], 100 * d['high']], [y + 0.1] * 2, color=BLUE, linewidth=2.2, solid_capstyle='butt')
            ax.plot([100 * d['mean']], [y + 0.1], marker='o', color=BLUE, markeredgecolor=SURFACE, markersize=6)
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.set_yticks(ys); ax.set_yticklabels([l for l, _ in items], fontsize=7); ax.set_xlabel('difference in problems solved on test (percentage points), first minus second')
    ax.grid(axis='y', visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=BLUE, lw=2.2, marker='o', label='95% interval over seeds'), Line2D([0], [0], color=GREY2, lw=1.2, label='95% interval over problems (bootstrap)')], loc='lower right', frameon=False)
    fig.tight_layout(); return save(plt, fig, out, 'fig_ablations')


def fig_solvers(C, P, out, plt):
    groups = [(g, v) for g in ('solvers', 'mw') for v in C.variants(g)]
    budgets = [b for b in P.get('eval_budgets', ()) if b <= P.get('eval_probes', 1e18)]
    if not groups or not budgets: return None
    fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 2.8), squeeze=False); axes = axes[0]; drawn = 0
    for ax, (g, v) in zip(axes, groups):
        for s in C.solvers(g, v):
            colour, ls, mk, label = SOLVER_STYLE.get(s, (GREY, '-', 'o', s))
            cv = C.curve(g, v, s, 'test', budgets)
            band(ax, budgets, cv['mean'], cv['low'], cv['high'], colour, ls, mk, label, ms=4); drawn += 1
        ax.set_xscale('log'); ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets], fontsize=6.5); ax.minorticks_off(); ax.set_ylim(bottom=0); pct_axis(ax)
        ax.set_title('%s (test)' % TB.LABELS.get(v, v), fontsize=7.5); ax.set_xlabel('probes per problem (log)')
        ax.legend(frameon=False, loc='upper left', fontsize=6.3)
    axes[0].set_ylabel('problems solved (%)')
    if not drawn: plt.close(fig); return None
    fig.tight_layout(); return save(plt, fig, out, 'fig_solvers')


def fig_generator(C, P, out, plt):
    st = (C.synth or {}).get('stats')
    if not st or not st.get('rarefaction'): return None
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    for key, colour, ls, label in (('skeleton', BLUE, '-', 'proof skeletons'), ('shape', ORANGE, '-', 'statement shapes'), ('recipe', AQUA, '-', 'recipes (lemma sets)'), ('lemmas', VIOLET, '-', 'library lemmas applied')):
        pts = st['rarefaction'].get(key)
        if not pts: continue
        m = [p['m'] for p in pts]; mu = [p['mean'] for p in pts]; sd = [p['sd'] for p in pts]
        ax.plot(m, mu, color=colour, linestyle=ls, marker='o', markersize=3.5, label=label)
        ax.fill_between(m, np.array(mu) - np.array(sd), np.array(mu) + np.array(sd), color=colour, alpha=0.15, linewidth=0)
    n = st['total']; ax.plot([1, n], [1, n], color=GREY2, linestyle=(0, (1, 2)), linewidth=0.9, label='every theorem new')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('theorems drawn'); ax.set_ylabel('distinct classes'); ax.set_title('(a) saturation'); ax.legend(frameon=False, loc='lower right', fontsize=6)
    ax = axes[1]
    k = st['kinds']['per_kind']
    names = list(k)[:8]; vals = [100 * k[x]['share_of_steps'] for x in names]
    ax.barh(range(len(names)), vals, color=BLUE, height=0.6); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel('share of proof steps (%)'); ax.set_title('(b) actions in the proofs (%d kinds of %d)' % (st['kinds']['distinct_kinds'], 15)); ax.grid(axis='y', visible=False)
    ax = axes[2]
    bd = st.get('by_depth') or {}
    ds = sorted(int(d) for d in bd); ns = [bd[str(d)]['n'] for d in ds]
    ax.bar(ds, ns, color=ORANGE, width=0.7); ax.set_xlabel('proof-tree depth'); ax.set_ylabel('theorems'); ax.set_title('(c) depth'); ax.grid(axis='x', visible=False)
    fig.tight_layout(); return save(plt, fig, out, 'fig_generator')


def fig_library(C, P, out, plt):
    cv = C.log_curve('rl', 'ei_her_boot', 'iter', 'lemmas')
    if not cv or not cv['x']: return None
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axes[0]
    band(ax, cv['index'], cv['mean'], cv['low'], cv['high'], ORANGE, '-', '^', 'lemmas in the library')
    ax.set_xlabel('iteration'); ax.set_ylabel('bootstrapped lemmas'); ax.set_title('(a) library growth'); ax.set_ylim(bottom=0)
    ax = axes[1]
    lens = []
    for rid, f in C.finals.items():
        if rid.startswith('rl/ei_her_boot/'): lens += [x for x in (f.get('lemma_file') or {}).get('lengths', []) if x]
    if lens:
        vals, cnt = np.unique(lens, return_counts=True); ax.bar(vals, cnt, color=ORANGE, width=0.7)
    ax.set_xlabel('proof length of the lemma (steps)'); ax.set_ylabel('lemmas'); ax.set_title('(b) what was admitted'); ax.grid(axis='x', visible=False)
    fig.tight_layout(); return save(plt, fig, out, 'fig_library')


def fig_levels(C, P, out, plt):
    B = P.get('headline'); variants = [v for v in ('bc_scratch', 'bc', 'ppo', 'ei_her', 'ei_her_boot') if v in C.variants('main')]
    levels = sorted({C.info[n]['level'] for n in C.names('test', 'all') if C.info.get(n, {}).get('level') is not None})
    if not variants or not levels: return None
    fig, ax = plt.subplots(figsize=(6.4, 2.7)); w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        colour, ls, mk, label = STYLE[v]
        for j, l in enumerate(levels):
            names = [n for n in C.names('test', 'all') if C.info[n]['level'] == l]
            M = C.success('main', v, 'sample', 'test', B, 'all', names=names); r = M.rates(); r = r[~np.isnan(r)]
            if not len(r): continue
            m, lo, hi = ST.mean_ci(r) if len(r) > 1 else (float(r[0]), NAN, NAN)
            x = j + (i - (len(variants) - 1) / 2) * w
            ax.bar([x], [m], width=w * 0.9, color=colour, alpha=0.5 + 0.5 * (i % 2), edgecolor=colour, hatch=None if ls == '-' else '//', linewidth=0.8, label=label if j == 0 else None)
            if not math.isnan(lo): ax.plot([x, x], [lo, hi], color=INK2, linewidth=0.8)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(['level %s\n(%d)' % (l, sum(1 for n in C.names('test', 'all') if C.info[n]['level'] == l)) for l in levels]); pct_axis(ax)
    ax.set_ylabel('test problems solved (%)'); ax.grid(axis='x', visible=False); ax.legend(frameon=False, ncol=3, fontsize=6.5, loc='upper right')
    fig.tight_layout(); return save(plt, fig, out, 'fig_levels')


def fig_throughput(C, P, out, plt):
    b = (C.bench or {}).get('throughput')
    if not b or not (b.get('solvers') or b.get('workers')): return None
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axes[0]
    if b.get('solvers'):
        names = list(b['solvers']); vals = [b['solvers'][n].get('probes_per_s') or 0 for n in names]
        ax.barh(range(len(names)), vals, color=BLUE, height=0.6); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7); ax.invert_yaxis()
        ax.set_xlabel('probes per second (one core)'); ax.set_title('(a) engine throughput'); ax.grid(axis='y', visible=False)
    ax = axes[1]
    if b.get('workers'):
        ws = sorted(int(w) for w in b['workers']); sp = [b['workers'][str(w)].get('speedup') or NAN for w in ws]
        ax.plot(ws, sp, color=ORANGE, marker='o', label='measured'); ax.plot(ws, ws, color=GREY2, linestyle=(0, (4, 2)), label='ideal')
        ax.set_xlabel('worker processes'); ax.set_ylabel('speed-up'); ax.set_title('(b) scaling'); ax.legend(frameon=False)
    fig.tight_layout(); return save(plt, fig, out, 'fig_throughput')


FIGURES = [('fig_budget', fig_budget), ('fig_scaling', fig_scaling), ('fig_learning', fig_learning), ('fig_bc', fig_bc), ('fig_ablations', fig_ablations), ('fig_solvers', fig_solvers),
           ('fig_generator', fig_generator), ('fig_library', fig_library), ('fig_levels', fig_levels), ('fig_throughput', fig_throughput)]


def make_all(C, outdir, only=None, log=print):
    plt = _mpl(); P = C.profile or {}; out = {}
    for name, fn in FIGURES:
        if only and name not in only: continue
        try: paths = fn(C, P, outdir, plt)
        except Exception as e:
            log('figure %s failed: %s: %s' % (name, type(e).__name__, e)); C.warnings.append('figure %s failed: %s' % (name, e)); continue
        if paths: out[name] = paths
        else: log('figure %s: no data' % name)
    return out
