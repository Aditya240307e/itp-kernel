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
import exp_stats as ST, exp_collect as EC, exp_tables as TB, exp_spec as SP

NAN = float('nan')
FEW_SEEDS = False; N_SEEDS = 0              # set by make_all: with fewer than FEW_SEEDS_BELOW replicates an interval over seeds is not drawn (a t-interval of two or three values is as wide as the plot)
FEW_SEEDS_BELOW = 4
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


PLOT_COLUMNS = ['panel', 'series', 'kind', 'label', 'x', 'y', 'x_low', 'x_high', 'y_low', 'y_high']


def plotted_data(fig):
    """what a figure draws, as rows (PLOT_COLUMNS): every visible line (with the band `band` drew around it), every error bar, every interval drawn as a short line, every bar and every other filled region.  The
    values the figure shows, so that a caption or a plot of one's own can be made from the numbers and not from the picture.  `label`: the text of the tick at the row's position, for a figure whose axis names
    things (the comparisons of fig_ablations, the bars of fig_generator)."""
    import matplotlib.collections as mc, matplotlib.container as mcont
    rows = []
    def f(v): return float(v)
    for pi, ax in enumerate(fig.axes):
        panel = ax.get_title() or 'panel %d' % (pi + 1); handled = set()
        xl = {round(f(t), 6): tx.get_text() for t, tx in zip(ax.get_xticks(), ax.get_xticklabels()) if tx.get_text()}
        yl = {round(f(t), 6): tx.get_text() for t, tx in zip(ax.get_yticks(), ax.get_yticklabels()) if tx.get_text()}
        # a label is only meaningful where the axis names things: the ticks are text that is not just a number
        def isnum(t):
            if '$' in t: return True                                                               # (the tick of a logarithmic axis: $10^{2}$)
            try: float(t.replace('\u2212', '-')); return True
            except ValueError: return False
        xl = {k: v for k, v in xl.items() if not isnum(v)}; yl = {k: v for k, v in yl.items() if not isnum(v)}
        def near(table, v):
            """the text of the tick nearest to v, when v is within half the spacing of the ticks (an interval is drawn a little off its row)"""
            if not table or v in ('', None): return ''
            ks = sorted(table); gap = min([b - a for a, b in zip(ks, ks[1:])] or [1.0]); k = min(ks, key=lambda t: abs(t - f(v)))
            return table[k] if abs(k - f(v)) <= gap / 2 + 1e-9 else ''
        def lab(x, y): return near(yl, y) or near(xl, x)
        for cont in ax.containers:
            name = getattr(cont, '_itp_series', None) or (cont.get_label() if not cont.get_label().startswith('_') else ''); grp = getattr(cont, '_itp_group', '')
            if isinstance(cont, mcont.ErrorbarContainer):
                dl, _, bars = cont.lines; handled.add(id(dl)); segs = [seg for bc in bars for seg in bc.get_segments()]
                for xv, yv, seg in zip(dl.get_xdata(), dl.get_ydata(), segs): rows.append([panel, name, 'errorbar', lab(xv, yv), f(xv), f(yv), '', '', f(min(seg[:, 1])), f(max(seg[:, 1]))])
            elif isinstance(cont, mcont.BarContainer):
                horiz = getattr(cont, 'orientation', 'vertical') == 'horizontal'
                for pt in cont.patches:
                    if horiz: pos, val = pt.get_y() + pt.get_height() / 2, pt.get_width(); rows.append([panel, name, 'bar', grp or yl.get(round(f(pos), 6), ''), f(val), f(pos), '', '', '', ''])
                    else: pos, val = pt.get_x() + pt.get_width() / 2, pt.get_height(); rows.append([panel, name, 'bar', grp or xl.get(round(f(pos), 6), ''), f(pos), f(val), '', '', '', ''])
        for ln in ax.get_lines():
            if id(ln) in handled: continue
            x, y = ln.get_xdata(orig=False), ln.get_ydata(orig=False)
            if len(x) == 0 or len(x) != len(y): continue
            if ln.get_marker() in ('', 'None', None) and ln.get_linestyle() in ('None', '', ' '): continue
            name = getattr(ln, '_itp_series', None) or (ln.get_label() if not ln.get_label().startswith('_') else '')           # (a line kept out of the legend can still be named for the data file: the fit of fig_laws)
            if len(x) == 2 and not name and ln.get_transform().contains_branch(ax.transData):        # an unlabelled two-point line in data coordinates: an interval (a guide such as axvline is in axes coordinates)
                if f(x[0]) == f(x[1]) and f(y[0]) == f(y[1]): rows.append([panel, '', 'interval', lab(x[0], y[0]), f(x[0]), f(y[0]), f(x[0]), f(x[0]), f(y[0]), f(y[0])])      # (zero width: a point)
                elif f(x[0]) == f(x[1]): rows.append([panel, '', 'interval', lab(x[0], ''), f(x[0]), '', '', '', f(min(y)), f(max(y))])
                elif f(y[0]) == f(y[1]): rows.append([panel, '', 'interval', lab('', y[0]), '', f(y[0]), f(min(x)), f(max(x)), '', ''])
                continue
            if len(x) == 2 and not name: continue
            band = getattr(ln, '_itp_band', {})
            for xv, yv in zip(x, y):
                lo, hi = band.get(f(xv), ('', ''))
                rows.append([panel, name, 'line', lab(xv, yv), f(xv), f(yv), '', '', lo, hi])
        for coll in ax.collections:
            if isinstance(coll, mc.PolyCollection) and not getattr(coll, '_itp_skip', False):
                byx = collections.defaultdict(list)
                for path in coll.get_paths():
                    for vx, vy in path.vertices: byx[round(f(vx), 9)].append(f(vy))
                for vx in sorted(byx): rows.append([panel, '', 'region', '', vx, '', '', '', min(byx[vx]), max(byx[vx])])
    return rows


def save(plt, fig, outdir, name, seeds_note=True):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    if FEW_SEEDS and seeds_note: fig.text(0.995, 0.004, '%d seeds: no intervals over seeds are drawn' % N_SEEDS, ha='right', va='bottom', fontsize=6, color=INK2)
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, '%s.%s' % (name, ext)); fig.savefig(p, bbox_inches='tight'); paths.append(p)
    try:
        import csv as _csv
        with open(os.path.join(outdir, name + '.csv'), 'w', newline='') as f:
            w = _csv.writer(f); w.writerow(PLOT_COLUMNS)
            for row in plotted_data(fig): w.writerow(['%.6g' % v if isinstance(v, float) else v for v in row])
    except Exception as e:                                                         # (a figure is never lost to its data file)
        print('warning: the data of %s were not written: %s' % (name, e))
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
    """items: [(x, y, text, colour)] at the right end of each curve; the labels stand in the margin, pushed apart (a thin leader in the curve's colour joins a label that was moved to its curve)"""
    if not items: return
    ymin, ymax = ax.get_ylim(); gap = (ymax - ymin) * 0.055
    ys = _dodge([y for _, y, _, _ in items], gap)
    log = ax.get_xscale() == 'log'; x0, x1 = ax.get_xlim()
    for (x, y, text, col), yy in zip(items, ys):
        xt = x * (x1 / x0) ** (0.035 + xpad) if log else x + (0.035 + xpad) * (x1 - x0)
        moved = abs(yy - y) > 1e-9
        ax.annotate(text, xy=(x, y), xytext=(xt, yy), textcoords='data', va='center', ha='left', fontsize=6.8, color=INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.6, shrinkA=1, shrinkB=2) if moved else None)
        ax.plot([x], [y], marker='', linestyle='')


def band(ax, x, mean, low, high, colour, ls, marker, label, ms=4.5, alpha=0.14, zorder=3, clip=True):
    """a line of means with the band of its interval; `clip`: the quantity is a rate, which lives in [0, 1] (a loss does not: clip=False)"""
    x = np.asarray(x, float); m = np.asarray(mean, float); lo = np.asarray(low, float); hi = np.asarray(high, float)
    ok = ~np.isnan(m)
    line, = ax.plot(x[ok], m[ok], color=colour, linestyle=ls, marker=marker, markersize=ms, label=label, zorder=zorder, markeredgecolor=SURFACE, markeredgewidth=0.6)
    ok2 = ok & ~np.isnan(lo) & ~np.isnan(hi)
    cl = (lambda v: np.clip(v, 0.0, 1.0)) if clip else (lambda v: v)
    line._itp_band = {float(a): (float(cl(b)), float(cl(c))) for a, b, c in zip(x[ok2], lo[ok2], hi[ok2])} if not FEW_SEEDS else {}      # (what the band says, for plotted_data)
    if ok2.any() and not FEW_SEEDS:
        poly = ax.fill_between(x[ok2], cl(lo[ok2]), cl(hi[ok2]), color=colour, alpha=alpha, linewidth=0, zorder=zorder - 1)       # (a rate lives in [0, 1]; with two seeds the t-interval does not)
        poly._itp_skip = True


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
                lo_ = m if math.isnan(lo) else min(m, max(0.0, lo)); hi_ = m if math.isnan(hi) else max(m, min(1.0, hi))               # (a rate lives in [0, 1]; the t-interval of two seeds does not)
                ax.errorbar([xs[-1]], [m], yerr=[[m - lo_], [hi_ - m]], fmt='X', color=GREY, markersize=6, capsize=2, label='pre-trained only (%d probes)' % B, zorder=4)
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
    import matplotlib.ticker as mt
    for ax in (axes[0], axes[2]): ax.xaxis.set_major_locator(mt.MaxNLocator(integer=True))                                     # (iterations are whole numbers)
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
    import matplotlib.ticker as mt
    for ax in axes: ax.xaxis.set_major_locator(mt.MaxNLocator(integer=True))                                                  # (epochs are whole numbers)
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
            if not math.isnan(d['low']) and not FEW_SEEDS: ax.plot([100 * d['low'], 100 * d['high']], [y + 0.1] * 2, color=BLUE, linewidth=2.2, solid_capstyle='butt')
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
    ncols = min(2, len(groups)); nrows = -(-len(groups) // ncols)                                     # (a grid of at most two columns: a row of four panels is unreadable at the width of a page)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows), squeeze=False); axes = list(axes.ravel()); drawn = 0
    for ax in axes[len(groups):]: ax.set_visible(False)
    for ax, (g, v) in zip(axes, groups):
        for s in C.solvers(g, v):
            colour, ls, mk, label = SOLVER_STYLE.get(s, (GREY, '-', 'o', s))
            cv = C.curve(g, v, s, 'test', budgets)
            band(ax, budgets, cv['mean'], cv['low'], cv['high'], colour, ls, mk, label, ms=4); drawn += 1
        ax.set_xscale('log'); ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets], fontsize=6.5); ax.minorticks_off(); ax.set_ylim(bottom=0); pct_axis(ax)
        ax.set_title('%s (test)' % TB.LABELS.get(v, v), fontsize=7.5); ax.set_xlabel('probes per problem (log)')
        ax.legend(frameon=False, loc='upper left', fontsize=6.3)
    for ax in axes[:len(groups):ncols]: ax.set_ylabel('problems solved (%)')
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
            bc = ax.bar([x], [m], width=w * 0.9, color=colour, alpha=0.5 + 0.5 * (i % 2), edgecolor=colour, hatch=None if ls == '-' else '//', linewidth=0.8, label=label if j == 0 else None)
            bc._itp_series = label; bc._itp_group = 'level %s' % l
            if not math.isnan(lo) and not FEW_SEEDS: ax.plot([x, x], [max(0.0, lo), min(1.0, hi)], color=INK2, linewidth=0.8)             # (a rate lives in [0, 1]; with two seeds the t-interval leaves it)
    ax.set_ylim(0.0, 1.0); ax.set_xticks(range(len(levels))); ax.set_xticklabels(['level %s\n(%d)' % (l, sum(1 for n in C.names('test', 'all') if C.info[n]['level'] == l)) for l in levels]); pct_axis(ax)
    ax.set_ylabel('test problems solved (%)'); ax.grid(axis='x', visible=False); ax.legend(frameon=False, ncol=3, fontsize=6.5, loc='upper right')
    fig.tight_layout(); return save(plt, fig, out, 'fig_levels')


def _bench(C): return (C.bench or {}).get('throughput') or {}


def _ms(x):
    """a latency in milliseconds as a short label"""
    return '%.2f' % x if x < 10 else ('%.1f' % x if x < 100 else '%.0f' % x)


def fig_throughput(C, P, out, plt):
    """(a) probes per second of each prover on one core, the median of the passes with the range of the passes (a logarithmic axis: the provers differ by orders of magnitude); (b) speed-up with the worker
    processes; (c) the memory of the evaluation, counting the world once (proportional set size) next to the sum of the resident sets of the processes (what a naive measurement says)"""
    b = _bench(C)
    if not b or not (b.get('solvers') or b.get('workers')): return None
    ws = sorted(int(w) for w in (b.get('workers') or {}))
    mem = [(w, b['workers'][str(w)].get('memory') or {}) for w in ws]; mem = [(w, m) for w, m in mem if m.get('pss_mb_max')]
    n = (1 if b.get('solvers') else 0) + (1 if ws else 0) + (1 if mem else 0)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 0.6, 2.8)); axes = list(np.atleast_1d(axes)); letters = iter('abc')
    if b.get('solvers'):
        ax = axes.pop(0); S = b['solvers']; names = list(S); ys = list(range(len(names))); med = [S[s].get('probes_per_s') or NAN for s in names]
        for i, s in enumerate(names):                                                                   # the range of the passes (a thin line; one pass: none)
            lo, hi = S[s].get('probes_per_s_min'), S[s].get('probes_per_s_max')
            if (S[s].get('repeats') or 1) > 1 and lo and hi and hi > lo: ax.plot([lo, hi], [i, i], color=INK2, linewidth=1.0, zorder=2, solid_capstyle='butt')
        ax.plot(med, ys, linestyle='', marker='o', color=BLUE, markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.6, label='median of the passes', zorder=3)
        fin = _finite(med)
        if fin: ax.set_xlim(min(fin) / 1.8, max(fin) * 3.2)
        ax.set_xscale('log'); ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=7); ax.set_ylim(len(names) - 0.5, -0.5)
        for i, v in enumerate(med):
            if v == v: ax.annotate('%.0f' % v if v >= 100 else '%.1f' % v, xy=(v, i), xytext=(6, 0), textcoords='offset points', va='center', fontsize=6.8, color=INK2)
        ax.set_xlabel('probes per second (one core)'); ax.set_title('(%s) engine throughput' % next(letters)); ax.grid(axis='y', visible=False)
    if ws:
        ax = axes.pop(0); sp = [b['workers'][str(w)].get('speedup') or NAN for w in ws]
        ax.plot(ws, sp, color=ORANGE, marker='o', label='measured'); ax.plot(ws, ws, color=GREY2, linestyle=(0, (4, 2)), label='ideal')
        ax.set_xticks(ws); ax.set_xlabel('worker processes'); ax.set_ylabel('speed-up'); ax.set_title('(%s) scaling' % next(letters)); ax.legend(frameon=False, loc='upper left')
    if mem:
        ax = axes.pop(0); xs = [w for w, _ in mem]; pss = [m['pss_mb_max'] / 1024.0 for _, m in mem]; rss = [(m.get('rss_sum_mb_max') or NAN) / 1024.0 for _, m in mem]
        ax.plot(xs, pss, color=BLUE, marker='o', label='counting shared pages once')
        ax.plot(xs, rss, color=GREY, linestyle=(0, (4, 2)), marker='+', markersize=6, label='sum of resident sets')
        top = max(_finite(pss + rss) or [1.0]); ax.set_ylim(0.0, top * 1.35); ax.set_xticks(xs)
        ax.set_xlabel('worker processes'); ax.set_ylabel('memory (GB)'); ax.set_title('(%s) memory' % next(letters)); ax.legend(frameon=False, loc='upper left', fontsize=6.3)
    fig.tight_layout(); return save(plt, fig, out, 'fig_throughput', seeds_note=False)


BREAKDOWN_COLOURS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, GREY2]           # one hue per group of categories, in the fixed order of TB.BREAKDOWN_GROUPS; the remainder is grey
WHITE_TEXT = {BLUE, VIOLET}                                                         # (the fills a white number is readable on; on the others the number is black)


def fig_breakdown(C, P, out, plt):
    """where the time goes: for the Gymnasium interface (a random walk over the decisions) and each prover, the share of the wall-clock time of an instrumented pass in each group of operations (stacked, 100%)"""
    b = _bench(C); cols = []
    sh = TB.breakdown_shares((((b.get('gym') or {}).get('profiled')) or {}).get('breakdown'))
    if sh: cols.append(('Gymnasium interface', sh))
    for s, v in (b.get('solvers') or {}).items():
        sh = TB.breakdown_shares((v.get('profile') or {}).get('breakdown'))
        if sh: cols.append((s, sh))
    if not cols: return None
    labels = list(TB.BREAKDOWN_GROUPS) + [TB.OTHER_LABEL]; ys = np.arange(len(cols)); left = np.zeros(len(cols))
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(cols) + 1.5))
    for k, lab in enumerate(labels):
        w = np.array([sh[lab] for _, sh in cols]) * 100.0; col = BREAKDOWN_COLOURS[k]
        ax.barh(ys, w, left=left, height=0.62, color=col, edgecolor=SURFACE, linewidth=1.0, label=lab, zorder=3)
        for y, x0, ww in zip(ys, left, w):
            if ww >= 9.0: ax.text(x0 + ww / 2, y, '%.0f' % ww, ha='center', va='center', fontsize=6.5, color='white' if col in WHITE_TEXT else INK, zorder=4)
        left += w
    ax.set_yticks(ys); ax.set_yticklabels([n for n, _ in cols]); ax.set_ylim(len(cols) - 0.5, -0.5); ax.set_xlim(0, 100); ax.set_xlabel('share of the time of an instrumented pass (%)'); ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.45, -0.3), fontsize=6.5)
    fig.tight_layout(); return save(plt, fig, out, 'fig_breakdown', seeds_note=False)


LATENCY_GROUPS = collections.OrderedDict([('engine', BLUE), ('Gymnasium interface', ORANGE), ('checking a proof', AQUA)])


def latency_rows(b):
    """[(group, label, summary)] of the operations of the stack, in the order of a proof's life; an operation that was not measured is left out"""
    lat = b.get('latency') or {}; ck = b.get('check') or {}; gl = (((b.get('gym') or {}).get('profiled')) or {}).get('latency') or {}
    items = [('engine', 'one probe, any kind', lat.get('probe')), ('engine', 'the menu of a goal', lat.get('menu')), ('engine', 'a proof state for a statement', lat.get('start')), ('engine', 'replay of an action', lat.get('replay')),
             ('Gymnasium interface', 'reset', gl.get('reset')), ('Gymnasium interface', 'one decision', gl.get('decision')),
             ('checking a proof', 'kernel check of a finished proof', ck.get('kernel_check')), ('checking a proof', 'independent checker', ck.get('independent_check'))]
    return [(g, l, d) for g, l, d in items if d and d.get('n') and d.get('p50_ms') is not None]


def _latency_panel(ax, rows, colours, title):
    """rows: [(group, label, summary)]: the median (dot, its value to the left), the 90th percentile (tick) and the line to the 99th percentile, on a logarithmic axis"""
    ys = list(range(len(rows)))
    for i, (_, _, d) in enumerate(rows):
        c = colours[rows[i][0]]
        if d['p99_ms'] > d['p50_ms']: ax.plot([d['p50_ms'], d['p99_ms']], [i, i], color=c, linewidth=1.2, zorder=2, solid_capstyle='butt')
    for g, c in colours.items():
        sel = [i for i, r in enumerate(rows) if r[0] == g]
        if sel: ax.plot([rows[i][2]['p50_ms'] for i in sel], sel, linestyle='', marker='o', color=c, markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.6, label=g, zorder=3)
    ax.plot([d['p90_ms'] for _, _, d in rows], ys, linestyle='', marker='|', color=INK, markersize=7, markeredgewidth=1.0, label='90th percentile', zorder=4)
    for i, (_, _, d) in enumerate(rows): ax.annotate(_ms(d['p50_ms']), xy=(d['p50_ms'], i), xytext=(-6, 0), textcoords='offset points', va='center', ha='right', fontsize=6.5, color=INK2)
    lo = min(d['p50_ms'] for _, _, d in rows); hi = max(d['p99_ms'] for _, _, d in rows)
    ax.set_xscale('log'); ax.set_xlim(lo / 4.0, hi * 1.5); ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in rows], fontsize=7); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel('milliseconds per call (logarithmic axis)'); ax.set_title(title); ax.grid(axis='y', visible=False)


def fig_latency(C, P, out, plt):
    """(a) the latency of the operations of the stack; (b) of a probe by the kind of action (the kinds in the order of how often they are tried; the number of probes in brackets)"""
    b = _bench(C); rows = latency_rows(b)
    kinds = sorted(((k, v) for k, v in ((b.get('latency') or {}).get('probe_by_kind') or {}).items() if v.get('n') and v.get('p50_ms') is not None), key=lambda kv: -kv[1]['n'])[:10]
    krows = [('engine', '%s (%d)' % (k, v['n']), v) for k, v in kinds]
    if not rows and not krows: return None
    panels = [p for p in (rows, krows) if p]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels) + 0.4, 0.26 * max(len(p) for p in panels) + 1.5), gridspec_kw={'width_ratios': [1.15, 1.0][:len(panels)]}); axes = list(np.atleast_1d(axes))
    if rows: _latency_panel(axes.pop(0), rows, LATENCY_GROUPS, '(a) operations' if krows else 'operations')
    if krows: _latency_panel(axes.pop(0), krows, LATENCY_GROUPS, '(b) probes by kind of action' if rows else 'probes by kind of action')
    from matplotlib.lines import Line2D
    used = [g for g in LATENCY_GROUPS if any(r[0] == g for r in rows + krows)]
    handles = [Line2D([], [], linestyle='', marker='o', color=LATENCY_GROUPS[g], markersize=5, label='median, ' + g) for g in used]
    handles += [Line2D([], [], linestyle='', marker='|', color=INK, markersize=7, markeredgewidth=1.0, label='90th percentile'), Line2D([], [], color=INK2, linewidth=1.2, label='line to the 99th percentile')]
    fig.legend(handles=handles, frameon=False, ncol=min(len(handles), 5), loc='lower center', fontsize=6.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); return save(plt, fig, out, 'fig_latency', seeds_note=False)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the studies of the network and of the start
ARCH_COLOURS = collections.OrderedDict([('transformer', BLUE), ('bigru', ORANGE), ('cnn', AQUA), ('bow', VIOLET)])
ARCH_SHORT = {'transformer': 'Transformer', 'bigru': 'BiGRU', 'cnn': 'CNN', 'bow': 'Bag'}
SCRATCH_ORDER = [('ei_her', 'transformer'), ('ppo', 'transformer'), ('ei_her', 'bigru'), ('ei_her', 'cnn'), ('ei_her', 'bow'), ('ppo', 'bigru'), ('ppo', 'cnn'), ('ppo', 'bow')]      # (a method keeps its colour whatever else is drawn)
SCRATCH_MARKS = ['^', 's', 'D', 'v', 'P', 'o', 'X', '<']


def _bar_whisker(ax, x, vals, width, colour, label=None, series=None, group=None, hatch=None, alpha=1.0):
    """a bar of the mean of per-replicate values at x, with the interval over replicates as a thin line (not drawn with few replicates); returns True when something was drawn"""
    xs = _finite(vals)
    if not xs: return False
    m, lo, hi = ST.mean_ci(xs) if len(xs) > 1 else (float(xs[0]), NAN, NAN)
    bc = ax.bar([x], [m], width=width, color=colour, alpha=alpha, edgecolor=colour, hatch=hatch, linewidth=0.8, label=label)
    bc._itp_series = series or label or ''; bc._itp_group = group or ''
    if not math.isnan(lo) and not FEW_SEEDS: ax.plot([x, x], [lo, hi], color=INK2, linewidth=0.8)
    return True


def fig_arch(C, P, out, plt):
    """(a) imitation of held-out synthetic proofs by encoder (iid and compositional splits), (b) olympiad test problems solved by learner and encoder, (c) training speed of each encoder"""
    smax = TB._smax(P); B = P.get('headline')
    archs = [a for a in TB.ARCH_LABEL if a == 'transformer' or ('bc_' + a) in C.variants('arch')]
    if len(archs) < 2 or smax is None: return None
    bench = _bench(C).get('archs') or {}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9)); drawn = 0
    ax = axes[0]; w = 0.38
    for i, a in enumerate(archs):
        name = '%sn%d' % ('' if a == 'transformer' else a + '_', smax)
        for j, (sname, hatch, alpha) in enumerate((('synth_test_iid', None, 1.0), ('synth_test_comp', '//', 0.55))):
            vals = list(C.final_values('pre', name, ('best', sname, 'action_acc')).values())
            drawn += _bar_whisker(ax, i + (j - 0.5) * w, vals, w * 0.92, ARCH_COLOURS[a], series='%s, %s' % (TB.ARCH_LABEL[a], 'iid' if j == 0 else 'compositional'), group=ARCH_SHORT[a], hatch=hatch, alpha=alpha)
    ax.set_xticks(range(len(archs))); ax.set_xticklabels([ARCH_SHORT[a] for a in archs], fontsize=7); ax.set_ylim(0, 1); pct_axis(ax); ax.grid(axis='x', visible=False)
    ax.set_ylabel('actions right on held-out synthetic theorems (%)'); ax.set_title('(a) imitation')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GREY2, edgecolor=GREY2, label='iid split'), Patch(facecolor=GREY2, edgecolor=GREY2, alpha=0.55, hatch='//', label='compositional split')], frameon=False, loc='lower left', fontsize=6.3)
    ax = axes[1]
    learners = [('bc', 'BC')] + [(l, TB.LEARNER_LABEL[l]) for l in ('ei_her', 'ppo') if any('%s_%s' % (l, a) in C.variants('arch') for a in archs if a != 'transformer')]
    wb = 0.8 / len(archs)
    for k, (l, lab) in enumerate(learners):
        for i, a in enumerate(archs):
            v = l if a == 'transformer' else '%s_%s' % (l, a); g = 'main' if a == 'transformer' else 'arch'
            if v not in C.variants(g) or 'sample' not in C.solvers(g, v): continue
            M = C.success(g, v, 'sample', 'test', B, 'all'); r = M.rates(); r = list(r[~np.isnan(r)])
            drawn += _bar_whisker(ax, k + (i - (len(archs) - 1) / 2) * wb, r, wb * 0.92, ARCH_COLOURS[a], series=TB.ARCH_LABEL[a], group=lab)
    ax.set_xticks(range(len(learners))); ax.set_xticklabels([lab for _, lab in learners], fontsize=7); ax.set_ylim(bottom=0); pct_axis(ax); ax.grid(axis='x', visible=False)
    ax.set_ylabel('olympiad test problems solved (%)'); ax.set_title('(b) olympiad problems')
    ax = axes[2]
    for i, a in enumerate(archs):
        b = bench.get('%s@%d' % (a, SP.DEFAULT_WIDTH))
        if not b or not b.get('train_steps_per_s'): continue
        bc = ax.bar([i], [b['train_steps_per_s']], width=0.62, color=ARCH_COLOURS[a], edgecolor=ARCH_COLOURS[a]); bc._itp_series = TB.ARCH_LABEL[a]; bc._itp_group = ARCH_SHORT[a]; drawn += 1
        ax.annotate('%.0f' % b['train_steps_per_s'], xy=(i, b['train_steps_per_s']), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=6.8, color=INK2)
    ax.set_xticks(range(len(archs))); ax.set_xticklabels([ARCH_SHORT[a] for a in archs], fontsize=7); ax.set_ylim(bottom=0); ax.grid(axis='x', visible=False)
    ax.set_ylabel('training steps per second'); ax.set_title('(c) training speed')
    if not drawn: plt.close(fig); return None
    fig.legend(handles=[Patch(facecolor=ARCH_COLOURS[a], edgecolor=ARCH_COLOURS[a], label=TB.ARCH_LABEL[a]) for a in archs], loc='lower center', ncol=len(archs), frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); return save(plt, fig, out, 'fig_arch')


def _law_points(C, stage, names, sname):
    """per x: (mean, low, high) of the held-out loss over the seeds that have it"""
    out = []
    for n in names:
        v = list(C.final_values(stage, n, ('best', sname, 'loss')).values())
        out.append(ST.mean_ci(v) if len(v) > 1 else ((float(v[0]), NAN, NAN) if v else (NAN, NAN, NAN)))
    return out


def fig_laws(C, P, out, plt):
    """(a) held-out synthetic loss against the number of synthetic training theorems, (b) against the number of parameters, each with the fitted line of the table of scaling laws; (c) olympiad problems solved against parameters"""
    smax = TB._smax(P); sizes = sorted(set(P.get('synth_sizes', ())) - {0}); B = P.get('headline')
    if smax is None: return None
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8)); drawn = 0
    series = (('synth_test_iid', BLUE, 'o', 'iid split'), ('synth_test_comp', ORANGE, '^', 'compositional split'))
    def panel(ax, xs, names, xlabel):
        n = 0
        for sname, colour, mk, label in series:
            pts = _law_points(C, 'pre', names, sname)
            if not _finite([p[0] for p in pts]): continue
            band(ax, xs, [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], colour, '-', mk, label, ms=4, clip=False); n += 1
            al = TB._aligned(C, 'pre', names, ('best', sname, 'loss'))
            fits = [f for f in (ST.loglog_fit(xs, y) for y in al.values()) if not math.isnan(f[0])] if len(xs) >= 3 and al else []
            if fits:
                e = float(np.mean([f[0] for f in fits])); ic = float(np.mean([f[1] for f in fits]))         # (the average of the lines fitted to each seed: its slope is the exponent of the table of scaling laws)
                gx = np.geomspace(min(xs), max(xs), 20); fit, = ax.plot(gx, np.exp(ic) * gx ** e, color=colour, linestyle=(0, (4, 2)), linewidth=1.0, zorder=2, label='_fit'); fit._itp_series = 'fit, ' + label
                ax.annotate('e = %+.2f' % e, xy=(gx[-1], np.exp(ic) * gx[-1] ** e), xytext=(4, 0), textcoords='offset points', fontsize=6.5, color=colour, va='center')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(xlabel); ax.set_ylabel('held-out loss per decision (log)'); ax.minorticks_off()
        return n
    drawn += panel(axes[0], [float(C.synth_used(n) or n) for n in sizes], ['n%d' % n for n in sizes], 'synthetic training theorems (log)')
    axes[0].set_title('(a) data')
    ws = sorted({int(v.rsplit('_w', 1)[1]) for v in C.variants('size') if v.startswith('bc_w') and v.rsplit('_w', 1)[1].isdigit()} | {SP.DEFAULT_WIDTH})
    names = ['%sn%d' % ('' if w == SP.DEFAULT_WIDTH else 'w%d_' % w, smax) for w in ws]
    par = [TB.mean_of(list(C.final_values('pre', n, ('n_params',)).values())) for n in names]
    if len(ws) >= 2 and not any(math.isnan(x) for x in par):
        drawn += panel(axes[1], par, names, 'parameters (log)'); axes[1].set_title('(b) model size')
        ax = axes[2]
        for split, colour, mk, label in (('dev', GREY, 's', 'dev'), ('test', BLUE, 'o', 'test')):
            mean = []; low = []; high = []
            for w in ws:
                g, v = ('main', 'bc') if w == SP.DEFAULT_WIDTH else ('size', 'bc_w%d' % w)
                if v not in C.variants(g): mean.append(NAN); low.append(NAN); high.append(NAN); continue
                m, lo, hi, k = C.rate_ci(g, v, 'sample', split, B, 'all'); mean.append(m); low.append(lo); high.append(hi)
            if _finite(mean): band(ax, par, mean, low, high, colour, '-', mk, label, ms=4); drawn += 1
        ax.set_xscale('log'); ax.set_xlabel('parameters (log)'); ax.set_ylabel('olympiad problems solved (%)'); ax.set_ylim(bottom=0); pct_axis(ax); ax.minorticks_off(); ax.set_title('(c) olympiad problems, BC')
        ax.legend(frameon=False, loc='lower right', fontsize=6.5)
    else:
        axes[1].set_visible(False); axes[2].set_visible(False)
    if not drawn: plt.close(fig); return None
    h, l = axes[0].get_legend_handles_labels()
    if h: fig.legend(h, [x for x in l], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); return save(plt, fig, out, 'fig_laws')


def fig_scratch(C, P, out, plt):
    """the dev problems solved by learners that start from random weights: (a) by iteration, (b) by the probes spent (log)"""
    vs = [v for v in C.variants('scratch')]
    runs = [v for v in vs if C.log_runs('rl', v)]
    if not runs: return None
    def key(v):
        base, arch, _ = TB.split_variant(v); learner = base[len('scratch_'):] if base.startswith('scratch_') else base
        return (learner, arch)
    runs.sort(key=lambda v: SCRATCH_ORDER.index(key(v)) if key(v) in SCRATCH_ORDER else len(SCRATCH_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8)); drawn = 0
    for v in runs:
        idx = SCRATCH_ORDER.index(key(v)) if key(v) in SCRATCH_ORDER else len(SCRATCH_ORDER) - 1
        colour = [ORANGE, AQUA, VIOLET, MAGENTA, YELLOW, BLUE, GREEN, RED][idx % 8]; mk = SCRATCH_MARKS[idx % len(SCRATCH_MARKS)]
        for ax, xkey in zip(axes, ('iter', 'cum_probes')):
            cv = C.log_curve('rl', v, xkey, 'dev_solved', normalise='dev_total')
            if not cv or not cv['x']: continue
            band(ax, cv['x'], cv['mean'], cv['low'], cv['high'], colour, '-', mk, TB.scratch_label(v), ms=3.5); drawn += 1
    if not drawn: plt.close(fig); return None
    base = C.rate_ci('baselines', 'baselines', 'uniform', 'dev', P.get('headline'), 'all') if 'uniform' in C.solvers('baselines', 'baselines') else None
    for ax in axes:
        if base and not math.isnan(base[0]): ax.axhline(base[0], color=GREY2, linestyle=(0, (1, 1.5)), linewidth=1.0, zorder=1)
        ax.set_ylim(bottom=0); pct_axis(ax)
    if base and not math.isnan(base[0]): axes[0].annotate('uniform random actions', xy=(axes[0].get_xlim()[0], base[0]), xytext=(2, 3), textcoords='offset points', fontsize=6.3, color=INK2)
    axes[0].set_xlabel('iteration'); axes[0].set_ylabel('dev problems solved (%)'); axes[0].set_title('(a) by iteration')
    axes[1].set_xscale('log'); axes[1].set_xlabel('cumulative probes of rollouts (log)'); axes[1].set_title('(b) by compute')
    import matplotlib.ticker as mt
    axes[0].xaxis.set_major_locator(mt.MaxNLocator(integer=True))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=min(3, len(h)), frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); return save(plt, fig, out, 'fig_scratch')


FIGURES = [('fig_budget', fig_budget), ('fig_scaling', fig_scaling), ('fig_learning', fig_learning), ('fig_bc', fig_bc), ('fig_ablations', fig_ablations), ('fig_solvers', fig_solvers),
           ('fig_generator', fig_generator), ('fig_library', fig_library), ('fig_levels', fig_levels), ('fig_throughput', fig_throughput), ('fig_breakdown', fig_breakdown), ('fig_latency', fig_latency),
           ('fig_arch', fig_arch), ('fig_laws', fig_laws), ('fig_scratch', fig_scratch)]


def make_all(C, outdir, only=None, log=print):
    global FEW_SEEDS, N_SEEDS
    plt = _mpl(); P = C.profile or {}; out = {}
    N_SEEDS = len(P.get('seed_list') or []); FEW_SEEDS = 0 < N_SEEDS < FEW_SEEDS_BELOW
    for name, fn in FIGURES:
        if only and name not in only: continue
        try: paths = fn(C, P, outdir, plt)
        except Exception as e:
            log('figure %s failed: %s: %s' % (name, type(e).__name__, e)); C.warnings.append('figure %s failed: %s' % (name, e)); continue
        if paths: out[name] = paths
        else: log('figure %s: no data' % name)
    return out
