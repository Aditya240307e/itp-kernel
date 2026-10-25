"""exp_stats.py -- the statistics of the experiments, small and explicit.

    mean_ci([0.31, 0.28, 0.35])                         # (mean, low, high): Student-t interval of the mean over replicates (training seeds); NaN bounds for fewer than two
    paired_seed_diff(a, b)                               # replicates paired by seed: mean difference, its t-interval, the two-sided p of the paired t-test
    paired_problem_bootstrap(A, B)                       # A, B: (seeds x problems) 0/1 matrices: mean difference with a bootstrap interval over PROBLEMS (resampled together in both methods)
    wilson(k, n)                                         # interval of a proportion from ONE run (a baseline without replicates)
    chao1(counts)                                        # (estimate of the number of classes, standard error): how many skeletons a generator could still make
    rarefaction(labels, points)                          # distinct classes among the first m items, for m in points
    holm(pvals)                                          # Holm-Bonferroni adjusted p-values for a family of comparisons

A replicate is a whole run of the pipeline from its seed (pre-training, fine-tuning, reinforcement learning, evaluation), so the interval over replicates is the interval that says how much a result depends on
the seed.  The bootstrap over problems says how much it depends on WHICH problems the benchmark happens to hold; both are reported, neither replaces the other.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import math
import numpy as np

NAN = float('nan')


def _t_crit(df, level):
    from scipy import stats
    return float(stats.t.ppf(0.5 + level / 2.0, df))


def mean_ci(x, level=0.95):
    """(mean, low, high) of the values `x` with the Student-t interval; (mean, nan, nan) for one value, (nan, nan, nan) for none"""
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    n = len(x)
    if n == 0: return NAN, NAN, NAN
    m = float(x.mean())
    if n == 1: return m, NAN, NAN
    h = _t_crit(n - 1, level) * float(x.std(ddof=1)) / math.sqrt(n)
    return m, m - h, m + h


def paired_seed_diff(a, b, level=0.95):
    """{mean, low, high, p, n}: the differences a_i - b_i of replicates paired by index (the same seed), their Student-t interval, and the two-sided p of the paired t-test (NaN for fewer than two pairs)"""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    m, lo, hi = mean_ci(d, level); n = len(d)
    p = NAN
    if n >= 2 and float(d.std(ddof=1)) > 0:
        from scipy import stats
        p = float(stats.ttest_1samp(d, 0.0).pvalue)
    elif n >= 2: p = 1.0 if float(d.mean()) == 0 else 0.0
    return {'mean': m, 'low': lo, 'high': hi, 'p': p, 'n': n}


def paired_problem_bootstrap(A, B, n_boot=4000, level=0.95, seed=0):
    """A, B: arrays (replicates x problems) of 0/1 (solved) for two methods over the same problems.  The statistic is mean(A) - mean(B); the interval resamples the PROBLEMS with replacement (the same
    problems in both methods, all replicates of each problem together).  Returns {mean, low, high}."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if A.shape[1:] != B.shape[1:]: raise ValueError('the two methods must be measured on the same problems')
    pa = A.mean(axis=0); pb = B.mean(axis=0); d = pa - pb; n = len(d)
    if n == 0: return {'mean': NAN, 'low': NAN, 'high': NAN}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {'mean': float(d.mean()), 'low': float(lo), 'high': float(hi)}


def wilson(k, n, level=0.95):
    """(p, low, high): the Wilson interval of k successes in n trials"""
    if n == 0: return NAN, NAN, NAN
    from scipy import stats
    z = float(stats.norm.ppf(0.5 + level / 2.0)); p = k / n
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def chao1(counts):
    """(S_est, se) of the number of classes in a population from the abundance counts of the classes seen (Chao 1984, bias-corrected: S_obs + f1 (f1 - 1) / (2 (f2 + 1)))"""
    counts = [c for c in counts if c > 0]
    s_obs = len(counts)
    f1 = sum(1 for c in counts if c == 1); f2 = sum(1 for c in counts if c == 2)
    est = s_obs + f1 * (f1 - 1) / (2.0 * (f2 + 1))
    # variance of the bias-corrected estimator (Chao 1987; Colwell's EstimateS)
    a = (f1 * (f1 - 1)) / (2.0 * (f2 + 1))
    var = a + f1 * (2 * f1 - 1) ** 2 / (4.0 * (f2 + 1) ** 2) + f1 ** 2 * f2 * (f1 - 1) ** 2 / (4.0 * (f2 + 1) ** 4)
    return float(est), float(math.sqrt(max(var, 0.0)))


def rarefaction(labels, points, repeats=20, seed=0):
    """for each m in `points`: the mean number of distinct labels among m items drawn without replacement from `labels` (repeats random orders), and the standard deviation"""
    labels = list(labels); n = len(labels)
    ids = {l: i for i, l in enumerate(dict.fromkeys(labels))}
    arr = np.asarray([ids[l] for l in labels], dtype=np.int64)
    rng = np.random.default_rng(seed); out = []
    pts = [min(int(m), n) for m in points]
    res = np.zeros((repeats, len(pts)))
    for r in range(repeats):
        order = rng.permutation(n); seen = np.zeros(len(ids), dtype=bool); k = 0; j = 0
        cum = np.zeros(n, dtype=np.int64)
        for t, v in enumerate(arr[order]):
            if not seen[v]: seen[v] = True; k += 1
            cum[t] = k
        res[r] = [cum[m - 1] if m > 0 else 0 for m in pts]
    return [(m, float(res[:, i].mean()), float(res[:, i].std())) for i, m in enumerate(pts)]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a family of comparisons (NaN stays NaN; the adjusted values are monotone and capped at 1)"""
    idx = [i for i, p in enumerate(pvals) if p is not None and not (isinstance(p, float) and math.isnan(p))]
    out = [NAN] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i]); m = len(order); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i])); out[i] = run
    return out


def first_crossing(xs, ys, level):
    """the first x at which y reaches `level` (linear interpolation between the points around the crossing); None when it never does"""
    prev = None
    for x, y in zip(xs, ys):
        if y is None or (isinstance(y, float) and math.isnan(y)): continue
        if y >= level:
            if prev is None or prev[1] >= level: return x
            (x0, y0) = prev
            return x0 + (x - x0) * (level - y0) / (y - y0) if y > y0 else x
        prev = (x, y)
    return None
