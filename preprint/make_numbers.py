#!/usr/bin/env python3
"""make_numbers.py -- turn the measurement files of preprint/measurements into LaTeX macros.

    python3 preprint/make_numbers.py                 writes preprint/generated_numbers.tex
    python3 preprint/make_numbers.py --print         also prints every macro with its value (to check the text against)

Every number the report quotes that was measured (as opposed to counted in the source or stated by a definition) comes from here, so that the text cannot drift from the files:

    measurements/libs_run1.json, libs_run2.json     measure.py        the library chain: build, sharing, kernel re-check, independent checker, micro-benchmarks
    measurements/bench/bench.json                   exp_bench.py      the problem-bank world, the latency of the environment, throughput, two workers, determinism
    measurements/check.json                         measure_check.py  what trusting a proof costs: kernel and independent checker on a sample of the bank's reference proofs

The sizes of the source (lines per component) are counted from the files git tracks, at the commit the measurements were made at.
Nothing is rounded here except for display; the JSON files hold the raw numbers.
"""
import os, sys, json, subprocess, argparse, re

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MEAS = os.path.join(HERE, 'measurements')
OUT = os.path.join(HERE, 'generated_numbers.tex')


def load(*parts): return json.load(open(os.path.join(MEAS, *parts)))


def g3(x):
    """three significant digits, no exponent"""
    if x == 0: return '0'
    from math import log10, floor
    d = max(0, 2 - int(floor(log10(abs(x)))))
    s = '%.*f' % (d, x)
    return s


def thousands(n):
    """12345 -> 12{,}345 (a number inside math or text)"""
    s = '{:,}'.format(int(n))
    return s.replace(',', '{,}')


def ms(x_s): return g3(1e3 * x_s)


def tex_escape(s): return s.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')


macros = {}


def M(name, value):
    """a macro with a text value; the name must consist of letters"""
    assert re.fullmatch(r'[A-Za-z]+', name), name
    macros[name] = str(value)


# ----------------------------------------------------------------------------------------------------------------------------------------------- the machine
r1, r2 = load('libs_run1.json'), load('libs_run2.json')
bench = load('bench', 'bench.json')
chk = load('check.json')
mach = r1['machine']
M('mCPU', 'Intel Xeon @ 2.10\\,GHz'); M('mCores', mach['cores']); M('mMemGB', g3(mach['memory_gb']))
M('mOS', 'Linux ' + mach['os'].split('-')[1]); M('mPython', mach['python']); M('mCompiler', 'g++ ' + re.search(r'(\d+\.\d+\.\d+)', mach['compiler']).group(1))
M('mCommit', mach['commit'][:7]); M('mCommitFull', mach['commit'])
assert chk['machine']['commit'] == mach['commit'] == bench['meta']['commit'], 'the measurements were made at different commits'

# ----------------------------------------------------------------------------------------------------------------------------------------------- the library chain
b1, b2 = r1['build'], r2['build']
M('libGlobals', b1['globals']); M('libBuildA', g3(b1['seconds'])); M('libBuildB', g3(b2['seconds']))
M('libPeakRSS', '%d' % round(b1['peak_rss_mb']))
cen = r1['census']['by_kind']
for key, name in (('inductive type', 'censusInd'), ('constructor', 'censusCtor'), ('definition', 'censusDef'), ('recursive function', 'censusRec'), ('theorem', 'censusThm'), ('quotient constant', 'censusQuot'), ('axiom', 'censusAxiom')):
    M(name, cen[key])
M('censusAxiomName', ', '.join(r1['census']['axioms']))
sh = r1['sharing']
M('shareTree', thousands(sh['tree_size_of_values'])); M('shareDistinct', thousands(sh['distinct_nodes_of_types_and_values'])); M('shareRatio', '%.1f' % sh['ratio_tree_to_distinct'])
M('shareLargest', thousands(sh['largest_value']['nodes_as_tree'])); M('shareLargestName', tex_escape(sh['largest_value']['name'])); M('shareValues', sh['values'])
k1, k2 = r1['kernel'], r2['kernel']
M('kerDecls', k1['declarations']); M('kerRecheckA', g3(k1['seconds'])); M('kerRecheckB', g3(k2['seconds'])); M('kerMedianUs', '%d' % round(1e3 * k1['median_ms'])); M('kerRejected', len(k1['rejected']))
i1, i2 = r1['independent'], r2['independent']
M('indLoadA', g3(i1['load_seconds'])); M('indLoadB', g3(i2['load_seconds'])); M('indCheckA', '%d' % round(i1['check_seconds'])); M('indCheckB', '%d' % round(i2['check_seconds']))
M('indGlobals', i1['globals']); M('indRejected', len(i1['rejected']))
M('indOverKernel', '%d' % round(0.5 * (i1['check_seconds'] / k1['seconds'] + i2['check_seconds'] / k2['seconds'])))
M('worldDigest', r1['determinism']['world_digest']); assert r1['determinism']['world_digest'] == r2['determinism']['world_digest']
M('worldDigestShort', r1['determinism']['world_digest'][:16])
ks = r1['kernel_stats']; mh, mm, ms_ = ks['typing_memo (hits, misses, ?)']
M('memoHits', thousands(mh)); M('memoMisses', thousands(mm)); M('memoShare', '%.0f' % (100.0 * mh / (mh + mm)))
M('synTried', thousands(ks['quick_conversion (tried, settled)'][0])); M('synSettled', thousands(ks['quick_conversion (tried, settled)'][1]))
mi1, mi2 = r1['micro'], r2['micro']
M('ffiNsA', '%d' % round(1e3 * mi1['ffi_call_baseline']['median_us'])); M('ffiNsB', '%d' % round(1e3 * mi2['ffi_call_baseline']['median_us']))
M('rewPairNsA', '%d' % round(1e3 * mi1['checkpoint_rewind_empty']['median_us'])); M('rewPairNsB', '%d' % round(1e3 * mi2['checkpoint_rewind_empty']['median_us']))
M('rewNodes', thousands(mi1['rewind_after_allocating']['nodes_built_about']))
M('nodeBuildNsA', '%d' % round(1e3 * mi1['rewind_after_allocating']['build_us_per_node_median'])); M('nodeBuildNsB', '%d' % round(1e3 * mi2['rewind_after_allocating']['build_us_per_node_median']))
M('rewindUsA', '%.0f' % mi1['rewind_after_allocating']['rewind_us_median']); M('rewindUsB', '%.0f' % mi2['rewind_after_allocating']['rewind_us_median'])
M('rewindNsNodeA', '%.1f' % mi1['rewind_after_allocating']['rewind_ns_per_node_median']); M('rewindNsNodeB', '%.1f' % mi2['rewind_after_allocating']['rewind_ns_per_node_median'])
for idx, tag in enumerate(('Small', 'Large')):
    pa, pb = mi1['binary_product_by_reduction'][idx], mi2['binary_product_by_reduction'][idx]
    M('prod%sBitsA' % tag, pa['bits_a']); M('prod%sBitsB' % tag, pa['bits_b']); M('prod%sMsA' % tag, '%d' % round(1e3 * pa['seconds_median'])); M('prod%sMsB' % tag, '%d' % round(1e3 * pb['seconds_median']))
    assert pa['product_decided_equal'] and pb['product_decided_equal']

# the rows of the table of library stages: (group, added, seconds run 1, seconds run 2)
rows = []
for ga, gb in zip(b1['groups'], b2['groups']):
    assert ga['group'] == gb['group'] and ga['added'] == gb['added']
    rows.append('%s & %d & %s & %s \\\\' % (tex_escape(ga['group']), ga['added'], g3(ga['seconds']), g3(gb['seconds'])))
M('libGroupRows', '\n'.join(rows))

# ----------------------------------------------------------------------------------------------------------------------------------------------- the bank
w = bench['world']
M('bankWorldSec', '%d' % round(w['seconds'])); M('bankGlobals', w['globals']); M('bankProblems', w['problems'])
cw = chk['world']
M('bankWorldSecB', '%d' % round(cw['seconds'])); M('bankRSSnoTorch', '%.1f' % (cw['peak_rss_mb'] / 1024.0)); M('bankRSSTorch', '%.1f' % (w['rss_mb'] / 1024.0))
M('kernelShare', '%.0f' % (100.0 * cw['kernel_share'])); M('kernelSecInWorld', g3(cw['kernel_seconds'])); M('kernelCallsInWorld', cw['kernel_checked_entry_points']['declare_checked']['calls'])
pr = chk['proofs']
M('proofsProved', pr['proved']); M('proofsTotalActions', thousands(pr['actions_total'])); M('proofsMedian', '%g' % pr['actions_median']); M('proofsMax', pr['actions_max']); M('proofsMin', pr['actions_min'])
M('proofsNinety', pr['deciles'][9])
M('statementOnly', w['problems'] - pr['proved'])

# the composition of the bank, counted from the released dataset file (data/dataset_v2/bank.json)
import collections
bank_json = json.load(open(os.path.join(ROOT, 'data', 'dataset_v2', 'bank.json')))
bps = bank_json['problems']
assert len(bps) == w['problems'] and sum(1 for p_ in bps if p_.get('proved')) == pr['proved'], 'the dataset file and the measurements describe different banks'
oc = collections.Counter(p_['origin'] for p_ in bps)
M('bankOrigContest', oc['contest']); M('bankOrigAdapted', oc['adapted']); M('bankOrigStyle', oc['style']); M('bankOrigFolklore', oc['folklore'])
assert sum(oc.values()) == len(bps)
sp = collections.Counter(p_['split'] for p_ in bps)
M('bankSplitTrain', sp['train']); M('bankSplitDev', sp['dev']); M('bankSplitTest', sp['test'])
lv_all = collections.Counter(p_['level'] for p_ in bps); lv_pr = collections.Counter(p_['level'] for p_ in bps if p_.get('proved'))
rows = []
for lv in range(1, 6):
    rows.append('%d & %d & %d & %d \\\\' % (lv, lv_pr[lv], lv_all[lv] - lv_pr[lv], lv_all[lv]))
rows.append('\\midrule\nall & %d & %d & %d \\\\' % (sum(lv_pr.values()), len(bps) - sum(lv_pr.values()), len(bps)))
M('bankLevelRows', '\n'.join(rows))
M('bankLevelFiveProved', lv_pr[5]); M('bankLevelFiveTotal', lv_all[5]); M('bankLevelFourProved', lv_pr[4]); M('bankLevelFourTotal', lv_all[4])
M('bankLevelsLowProved', lv_pr[1] + lv_pr[2] + lv_pr[3])
kd = collections.Counter(p_['kind'] for p_ in bps)
M('bankKindProve', kd['prove']); M('bankKindAnswer', kd['answer']); M('bankKindExists', kd['exists'])
tp = collections.Counter(t for p_ in bps for t in p_['topics'])
M('bankTopicNT', tp['number-theory']); M('bankTopicCombinatorics', tp['combinatorics']); M('bankTopicPolynomial', tp['polynomial']); M('bankTopicInequality', tp['inequality'])
src_ = collections.Counter(p_['source'].split()[0] for p_ in bps)
M('bankFromIMO', src_['IMO']); M('bankFromISL', src_['ISL']); M('bankFromAIME', src_['AIME']); M('bankFromAPMO', src_['APMO'])
imo_like = [p_ for p_ in bps if p_['source'].split()[0] in ('IMO', 'ISL', 'APMO')]
M('bankImoLike', len(imo_like)); M('bankImoLikeProved', sum(1 for p_ in imo_like if p_.get('proved')))

# ----------------------------------------------------------------------------------------------------------------------------------------------- what trusting a proof costs
sample = chk['sample']
summ = chk['summary']
M('chkSampled', summ['sampled']); M('chkAccepted', summ['independent_accepted']); M('chkMemory', summ['independent_memory']); M('chkTimeout', summ['independent_timeout'])
M('chkCapS', '%d' % round(chk['args']['cap_s'])); M('chkCapMB', thousands(chk['args']['cap_mb']))
M('chkSetupSec', g3(chk['independent_setup']['seconds'])); M('chkSetupGlobals', chk['independent_setup']['globals'])
ok = [s for s in sample if s['independent'].get('status') == 'ok']
ratios = [s['independent']['seconds'] / s['kernel_check_s'] for s in ok]
M('chkRatioMin', '%d' % round(min(ratios))); M('chkRatioMax', '%d' % round(max(ratios)))
ratios_sorted = sorted(ratios); M('chkRatioMedian', '%d' % round(ratios_sorted[len(ratios_sorted) // 2]))
M('chkKernelMedianMs', g3(1e3 * summ['kernel_check_s']['median'])); M('chkKernelMaxS', g3(summ['kernel_check_s']['max'])); M('chkIndMedianS', g3(summ['independent_s']['median'])); M('chkIndMaxS', g3(summ['independent_s']['max']))
M('chkIndMemMedianMB', '%d' % round(summ['independent_rss_delta_mb']['median'])); M('chkIndMemMaxMB', thousands(round(summ['independent_rss_delta_mb']['max'])))
big = [s for s in sample if s['independent'].get('status') != 'ok']
if big:
    b = big[0]; M('chkCapName', tex_escape(b['name'])); M('chkCapActions', b['actions']); M('chkCapNodes', thousands(b['term_tree_nodes'])); M('chkCapKernelS', g3(b['kernel_check_s'])); M('chkCapStatus', b['independent']['status'])
    M('chkCapAfterS', '%d' % round(b['independent']['seconds']))
rows = []
for s in sample:
    ind = s['independent']
    if ind.get('status') == 'ok':
        rows.append('\\texttt{%s} & %d & %s & %s & %s & %s & %s \\\\' % (tex_escape(s['name']), s['actions'], thousands(s['term_tree_nodes']), g3(1e3 * s['kernel_check_s']), g3(ind['seconds'] * 1e3), '%d' % round(ind['seconds'] / s['kernel_check_s']), '%d' % round(ind['rss_delta_mb'])))
    else:
        rows.append('\\texttt{%s} & %d & %s & %s & \\multicolumn{3}{c}{stopped at the memory cap after %d\\,s} \\\\' % (tex_escape(s['name']), s['actions'], thousands(s['term_tree_nodes']), g3(1e3 * s['kernel_check_s']), round(ind['seconds'])))
M('checkRows', '\n'.join(rows))
# the points of the scatter plot: tree size against time in ms (the kernel, the independent checker where it finished)
M('scatterKernel', ' '.join('(%d,%.3f)' % (s['term_tree_nodes'], 1e3 * s['kernel_check_s']) for s in sample))
M('scatterIndep', ' '.join('(%d,%.3f)' % (s['term_tree_nodes'], 1e3 * s['independent']['seconds']) for s in ok))
M('scatterIndepCap', ' '.join('(%d,%.3f)' % (s['term_tree_nodes'], 1e3 * s['independent']['seconds']) for s in big))

# ----------------------------------------------------------------------------------------------------------------------------------------------- the environment
lat = bench['latency']
pk = lat['probe']
M('probeN', thousands(pk['n'])); M('probeMed', g3(pk['p50_ms'])); M('probeNinety', g3(pk['p90_ms'])); M('probeNinetyNine', g3(pk['p99_ms'])); M('probeMean', g3(pk['mean_ms'])); M('probeMax', thousands(round(pk['max_ms'])))
ap = lat['probe_by_kind']['APPLY']['outcomes']; M('applyProbes', sum(ap.values())); M('applyFuel', ap['fuel']); M('applyFuelShare', '%.0f' % (100.0 * ap['fuel'] / sum(ap.values())))
M('probeTotalFuel', sum(v['outcomes'].get('fuel', 0) for v in lat['probe_by_kind'].values())); M('probeTotalRefused', sum(v['outcomes'].get('refused', 0) for v in lat['probe_by_kind'].values())); M('probeTotalOk', sum(v['outcomes'].get('ok', 0) for v in lat['probe_by_kind'].values()))
mn = lat['menu']
M('menuN', mn['n']); M('menuMed', g3(mn['p50_ms'])); M('menuNinety', g3(mn['p90_ms'])); M('menuMeanOpts', '%.0f' % mn['mean_options']); M('menuNinetyNine', g3(mn['p99_ms']))
st = lat['start']; M('startMed', g3(st['p50_ms'])); M('startN', st['n']); M('startNinety', g3(st['p90_ms'])); M('startNinetyNine', g3(st['p99_ms']))
rp = lat['replay']; M('replayMed', g3(rp['p50_ms'])); M('replayN', rp['n']); M('replayNinety', g3(rp['p90_ms'])); M('replayNinetyNine', g3(rp['p99_ms']))
kinds = ['APPLY', 'CASES', 'REWRITE', 'REWRITE_AT', 'LINARITH', 'NLINARITH', 'RING', 'BY_CONTRA', 'INDUCT_WITH', 'CLAIM', 'INTERVAL_CASES']
rows = []
for kd in kinds:
    d = lat['probe_by_kind'][kd]; oc = d['outcomes']
    rows.append('\\texttt{%s} & %d & %s & %s & %s & %d / %d / %d \\\\' % (kd.replace('_', '\\_'), d['n'], g3(d['p50_ms']), g3(d['p90_ms']), g3(d['p99_ms']), oc.get('ok', 0), oc.get('refused', 0), oc.get('fuel', 0)))
M('probeKindRows', '\n'.join(rows))
M('probeKindsTotal', len(lat['probe_by_kind']))
gy = bench['gym']['profiled']
dl = gy['latency']['decision_inner']
M('decN', thousands(dl['n'])); M('decMed', g3(dl['p50_ms'])); M('decNinety', g3(dl['p90_ms'])); M('decNinetyNine', g3(dl['p99_ms'])); M('decMean', g3(dl['mean_ms']))
M('gymDecPerS', thousands(round(bench['gym']['decisions_per_s']))); M('gymProbesPerS', '%.0f' % bench['gym']['probes_per_s'])
M('resetMs', g3(gy['latency']['reset']['p50_ms']))
cats = gy['breakdown']['categories']
M('gymAssemblerShare', '%.0f' % (100 * cats['assembler']['share'])); M('gymFeaturesShare', '%.0f' % (100 * cats['features']['share']))
M('gymMenuShare', '%.1f' % (100 * cats['menu']['share']))
M('gymEngineShare', '%.1f' % (100 * sum(cats[c]['share'] for c in ('probe', 'tactic', 'start', 'replay', 'nf'))))     # probes, the tactics that carry them out, episode starts, replays, normal forms: the engine's work apart from the menus
M('decPerProbeGym', '%.0f' % (bench['gym']['decisions'] / bench['gym']['probes']))
M('applyMaxMs', '%.0f' % lat['probe_by_kind']['APPLY']['max_ms'])
bl, un = bench['solvers']['blind'], bench['solvers']['uniform']
M('blindProbesPerS', '%.0f' % bl['probes_per_s']); M('blindProbesMin', '%.0f' % bl['probes_per_s_min']); M('blindProbesMax', '%.0f' % bl['probes_per_s_max'])
M('blindProblems', bl['n']); M('blindProbes', thousands(bl['probes'])); M('blindSecPerProblem', g3(bl['seconds_per_problem'])); M('blindSolved', bl['solved'])
M('blindOverhead', '%.2f' % bench['overhead']['blind']['ratio'])
bc = bl['profile']['breakdown']['categories']
M('blindTacticShare', '%.0f' % (100 * bc['tactic']['share'])); M('blindProbeShare', '%.0f' % (100 * bc['probe']['share'])); M('blindMenuShare', '%.0f' % (100 * bc['menu']['share']))
bp = bl['profile']['latency']['probe']
M('blindProbeMed', g3(bp['p50_ms'])); M('blindProbeMean', g3(bp['mean_ms'])); M('blindProbeNinety', g3(bp['p90_ms'])); M('blindProbeNinetyNine', g3(bp['p99_ms']))
bm = bl['profile']['latency']['menu']; M('blindMenuMed', g3(bm['p50_ms']))
M('uniProbesPerS', '%.0f' % un['probes_per_s']); M('uniDecPerS', thousands(round(un['decisions_per_s']))); M('uniActPerS', '%.0f' % un['actions_per_s'])
M('uniAssemblerShare', '%.0f' % (100 * un['profile']['breakdown']['categories']['assembler']['share']))
M('decPerProbeUni', '%.0f' % (un['decisions_per_s'] / un['probes_per_s']))
M('uniLimitS', '%d' % round(bench['time_limit'])); M('benchProbeBudget', bench['probes']); M('benchRepeats', bench['repeats'])
loads = [r1['machine']['loadavg_start'][0], r2['machine']['loadavg_start'][0], bench['machine']['loadavg_start'][0], chk['machine']['loadavg_start'][0]]
M('loadStartMax', '%.1f' % max(loads))
wk = bench['workers']
M('wkOneSec', g3(wk['1']['seconds'])); M('wkTwoSec', g3(wk['2']['seconds'])); M('wkOneRate', '%.0f' % wk['1']['probes_per_s']); M('wkTwoRate', '%.0f' % wk['2']['probes_per_s'])
M('wkSpeedup', '%.2f' % wk['2']['speedup']); M('wkEff', '%.2f' % wk['2']['efficiency']); M('wkTasks', wk['1']['tasks']); M('wkProbes', wk['1']['probes'])
M('wkPSS', thousands(round(wk['2']['memory']['pss_mb_max'])))
de = bench['determinism']
assert de['identical'] and de['n_differing'] == 0
M('detTasks', de['tasks']); M('detDigest', de['outcome_digest']); M('detRuns', len(de['runs']))
M('kernelDigest', bench['machine']['kernel_digest'])

# ----------------------------------------------------------------------------------------------------------------------------------------------- the gates (preprint/measure_gates.py) and the catalogues (counted)
gj = load('gates.json')
assert gj['commit'] == mach['commit'], 'the gates were run at a different commit than the measurements'
assert gj['code_clean']
G = {k: v['parsed'] for k, v in gj['gates'].items()}
for k, v in gj['gates'].items(): assert v['pass'], k
g = G['s2']
M('lockMatchTerms', thousands(g['match_terms'])); M('lockIllTyped', thousands(g['ill_typed_rejected'])); M('lockDepChecked', thousands(g['dep_checked'])); M('lockDepAccepted', thousands(g['dep_accepted']))
M('lockAdmission', g['admission_cases']); M('lockDepartures', g['admission_departures']); M('lockGuard', g['guard_cases']); M('lockCorpusProofs', g['corpus_proofs']); M('lockCorpusFalse', g['corpus_false_rejected']); M('lockGlobals', g['library_globals_redeclared'])
g = G['harness']
M('harnessTerms', thousands(g['terms'])); M('harnessComparisons', thousands(sum(g['comparisons'].values()))); M('harnessCorpusNorm', g['corpus_normalisations'])
g = G['oracle']
M('oracleDists', len(g['distributions'])); M('oracleCompleted', g['completed']); M('oracleVerified', g['verified'])
g = G['s8']
M('indepMutated', thousands(g['mutants'])); M('indepBothAccept', g['both_accept']); M('indepBothReject', thousands(g['both_reject']))
g = G['s15']
M('canaryClassAttempts', thousands(g['attempts'])); M('canaryClassFalse', g['false_statements']); M('canaryClassPower', thousands(g['power_attempts'])); M('canaryClassPowerOk', g['power_completed'])
assert g['completed'] == 0 and g['power_completed'] == g['power_verified']
g = G['s16']
M('canaryQuotAttempts', thousands(g['attempts'])); M('canaryQuotFalse', g['false_statements']); M('canaryQuotPower', thousands(g['power_attempts'])); M('canaryQuotPowerOk', g['power_completed']); M('canaryQuotTrue', g['true_statements'])
assert g['completed'] == 0 and g['power_completed'] == g['power_verified']
g = G['s19']
M('boundaryCases', g['cases']); M('boundarySeeds', g['fuzz_seeds']); M('boundaryCallsSeed', thousands(g['fuzz_calls_per_seed'])); M('boundaryCalls', thousands(g['fuzz_calls']))
if 's40' in G: M('actionChecks', G['s40']['checks'])
if 'sanitizers' in gj:
    sz = gj['sanitizers']['gates']
    M('sanGates', len(sz)); assert all(x['pass'] for x in sz)
    M('sanDefault', gj['sanitizers']['default_set']); M('sanSeconds', gj['sanitizers']['seconds']); M('sanLeftOut', len(gj['sanitizers']['left_out_by_the_tool']))

def load_module(name, rel, paths):
    import importlib.util
    sys.path[:0] = [os.path.join(ROOT, p) for p in paths]
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    finally:
        del sys.path[:len(paths)]
    return mod

reg = load_module('gate_registry', 'tests/gate_registry.py', ['tests', 'src/python'])
M('gatesDefault', len(reg.GATE_CMDS) + len(reg.BANK_GATE_CMDS)); M('gatesML', len(reg.ML_GATE_CMDS)); M('gatesRepo', len(reg.REPO_GATE_CMDS)); M('gatesSynth', len(reg.SYNTH_GATE_CMDS))
M('gatesAll', len(reg.GATE_CMDS) + len(reg.BANK_GATE_CMDS) + len(reg.ML_GATE_CMDS) + len(reg.REPO_GATE_CMDS) + len(reg.SYNTH_GATE_CMDS))
cat = load_module('catalogue', 'mutants/catalogue.py', ['mutants', 'src/python', 'tests'])
import collections
by = collections.Counter()
for nm, v in cat.MUTANTS.items():
    f = v[2] if len(v) > 2 else 'itp_kernel.hpp'
    by[f] += 1
GROUPS = {'kernel': ['itp_kernel.hpp', 'itp_bindings.cpp'], 'checker': ['indep_checker.py'], 'tactic': ['ring.py', 'linarith.py', 'lincomb.py', 'tactics.py', 'author.py'],
          'library': ['bin_lib.py', 'int_lib.py', 'ring_lib.py', 'quot_lib.py'], 'env': ['gym_acts.py', 'gym_env.py', 'gym_build.py', 'recorder.py', 'gym_search.py', 'gym_terms.py']}
assert sum(by.values()) == sum(by[f] for fs in GROUPS.values() for f in fs), 'a mutant targets a file that no group names: %s' % sorted(set(by) - {f for fs in GROUPS.values() for f in fs})
M('mutTotal', thousands(len(cat.MUTANTS))); M('mutEquiv', len(cat.KNOWN_EQUIVALENT))
for grp, fs in GROUPS.items(): M('mut' + {'kernel': 'Kernel', 'checker': 'Checker', 'tactic': 'Tactic', 'library': 'Library', 'env': 'Env'}[grp], thousands(sum(by[f] for f in fs)))
M('mutKernelHeader', thousands(by['itp_kernel.hpp'])); M('mutBindings', by['itp_bindings.cpp'])
tcat = load_module('tool_catalogue', 'mutants/tool_catalogue.py', ['mutants', 'src/python', 'tests'])
M('toolFaults', sum(len(v['mutants']) for v in tcat.TOOLS.values())); M('toolCount', len(tcat.TOOLS))

# the log of the findings: docs/FINDINGS.md (the table) and docs/findings_tags.tsv (the layer and the effect of each row, tagged by the author when the finding was written)
frows = []
for line in open(os.path.join(ROOT, 'docs', 'FINDINGS.md')).read().split('\n'):
    if line.startswith('| F'): frows.append([c.strip() for c in line.strip().strip('|').split('|')])
nums = sorted({int(re.match(r'F(\d+)', r[0]).group(1)) for r in frows})
assert nums == list(range(1, 94)), 'the findings are not F1..F93'
stat = collections.Counter()
for r in frows:
    st = re.sub(r'\*', '', r[2]).lower()
    kind = ('open' if st.startswith('open') else 'partial' if st.startswith('partial') else 'mitigated' if st.startswith('mitigated') else 'fixed' if st.startswith('fixed')
            else 'construction' if st.startswith('by construction') else 'documented')
    stat[kind] += 1
assert sum(stat.values()) == len(frows)
M('findNumbers', len(nums)); M('findRows', len(frows))
for kd, nm in (('fixed', 'findFixed'), ('construction', 'findConstruction'), ('mitigated', 'findMitigated'), ('partial', 'findPartial'), ('open', 'findOpen'), ('documented', 'findDocumented')): M(nm, stat[kd])
tags = [l.rstrip('\n').split('\t') for l in open(os.path.join(ROOT, 'docs', 'findings_tags.tsv')) if l.strip() and not l.startswith('#')]
assert sorted(t[0] for t in tags) == sorted(r[0] for r in frows), 'the tags and the findings differ'
layer = collections.Counter(); eff = collections.Counter()
for t in tags:
    for x in t[1].split(','): layer[x] += 1
    eff[t[2]] += 1
for lay, nm in (('kernel', 'findKernel'), ('checker', 'findChecker'), ('tactic', 'findTactic'), ('engine', 'findEngine'), ('data', 'findData'), ('pipeline', 'findPipeline'), ('tooling', 'findTooling')): M(nm, layer[lay])
M('findAccepts', eff['accepts']); M('findRefuses', eff['refuses']); M('findResource', eff['resource'])
M('findAcceptsIds', ', '.join(t[0] for t in tags if t[2] == 'accepts'))

# ----------------------------------------------------------------------------------------------------------------------------------------------- source sizes (counted from the files git tracks)
def lines(prefixes):
    out = subprocess.run(['git', '-C', ROOT, 'ls-files'] + prefixes, capture_output=True, text=True, check=True).stdout.split()
    n = 0
    for f in out:
        if f.endswith(('.py', '.hpp', '.cpp', '.h')):
            with open(os.path.join(ROOT, f), 'rb') as fh: n += sum(1 for _ in fh)
    return n


SIZES = [('kernel', ['src/c++/hybrid']), ('refkernel', ['src/c++/reference']), ('authoring', ['src/python/authoring']), ('checker', ['src/python/checker']), ('envdir', ['src/python/environment']),
         ('libdir', ['src/python/library']), ('probdir', ['src/python/problems']), ('learner', ['src/python/learner']), ('synth', ['src/python/synth']), ('exper', ['src/python/experiments']),
         ('tests', ['tests']), ('mutants', ['mutants']), ('tools', ['tools']), ('examples', ['examples'])]
for name, pre in SIZES: M('loc' + name, thousands(lines(pre)))
M('locKernelHeader', thousands(lines(['src/c++/hybrid/itp_kernel.hpp']))); M('locBindings', thousands(lines(['src/c++/hybrid/itp_bindings.cpp'])))

# ----------------------------------------------------------------------------------------------------------------------------------------------- measurements made late (a placeholder macro stands in until the file exists)
MISSING = []


def maybe_load(*parts):
    path = os.path.join(MEAS, *parts)
    if not os.path.exists(path):
        MISSING.append('/'.join(parts)); return None
    return json.load(open(path))


def placeholder(*names):
    for n in names: M(n, '\\textbf{??}')


# the digest of the code: every file below src/, tests/ and mutants/ but caches, built files and documents, and the manifest of the dataset (tools/check_numbers.py --tree)
M('codeDigest', subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'tools', 'check_numbers.py'), '--tree'], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip())

asm = maybe_load('assembler.json')
if asm:
    assert asm['machine']['commit'] == mach['commit'], 'the assembler round trip was run at a different commit than the measurements'
    tot = asm['total']
    M('asmProofs', tot['proofs']); M('asmActions', thousands(tot['actions'])); M('asmSaid', thousands(tot['said'])); M('asmMissing', tot['missing']); M('asmErrors', tot['errors'])
    M('asmMsPerAction', g3(1e3 * tot['seconds'] / tot['actions']))
    M('asmMissingKinds', ', '.join('\\code{%s} (%d)' % (k_.replace('_', '\\_'), n_) for k_, n_ in sorted(asm['failures_by_kind'].items(), key=lambda kv: -kv[1])) or 'none')
    for src_, nm_ in (('bank / prefix', 'Prefix'), ('bank / strict', 'Strict'), ('library', 'Lib')):
        r_ = asm['sources'][src_]; M('asm%sProofs' % nm_, r_['proofs']); M('asm%sActions' % nm_, thousands(r_['actions'])); M('asm%sMissing' % nm_, r_['missing'] + r_['errors'])
else: placeholder('asmProofs', 'asmActions', 'asmSaid', 'asmMissing', 'asmErrors', 'asmMsPerAction', 'asmMissingKinds', 'asmPrefixProofs', 'asmPrefixActions', 'asmPrefixMissing', 'asmStrictProofs', 'asmStrictActions', 'asmStrictMissing', 'asmLibProofs', 'asmLibActions', 'asmLibMissing')

# the gates: the run of all of them at this commit (measurements/gates_all.jsonl, one JSON line per finished gate) and the index of what each protects (tests/README.md)
ga = os.path.join(MEAS, 'gates_all.jsonl')
if os.path.exists(ga):
    recs = [json.loads(l) for l in open(ga) if l.strip()]
    assert len(recs) == 54, 'the run holds %d gates, not 54' % len(recs)
    # a gate that failed in the run of all of them and was run again alone (measurements/gates_rerun.jsonl): the record kept for it is that of the second run, and the report says so.
    # measure_gates.py recorded, where the history exists, the commit of the first run, the commit that corrected the gate, and the files below src/, tests/ and mutants/ that changed in between;
    # what is checked here is that the gate failed in the first run and that those files are the scripts of the gates that were run again and no others.
    rerun_note = ''
    rr = os.path.join(MEAS, 'gates_rerun.jsonl')
    if os.path.exists(rr):
        again = [json.loads(l) for l in open(rr) if l.strip()]
        rf_ = gj.get('rerun'); assert rf_, 'gates_rerun.jsonl exists but gates.json holds no "rerun" (run measure_gates.py again)'
        assert [r_['gate'] for r_ in again] == ['s51_suite_tests.py'] == rf_['gates'], 'gates_rerun.jsonl: one record, of the gate S51'
        assert all(any(r_['gate'] == a_['gate'] and not r_['ok'] for r_ in recs) and a_['ok'] for a_ in again), 'gates_rerun.jsonl: the gate must fail in the run of all of them and pass in the second run'
        assert rf_['changed'] == rf_['scripts'] == ['tests/learner/s51_suite_tests.py'], 'since the run of all the gates the files below src/, tests/ and mutants/ that changed are %s, not only the scripts of the gates run again' % rf_['changed']
        recs = [next(a_ for a_ in again if a_['gate'] == r_['gate']) if r_['gate'] in {b_['gate'] for b_ in again} else r_ for r_ in recs]
        rerun_note = (' The run of all of them was made at commit %s. In it the gate S51 failed one of its checks: the check pinned the list of the findings that had accepted what they must refuse, and the list did not yet name F92 and F93, the two findings of this version, which belong to it on purpose (the log of that run shows the failure).'
                      ' The check was corrected in commit %s, the only change to \\code{src/}, \\code{tests/} and \\code{mutants/} between the two commits, and S51 was then run again alone, at the commit of the other measurements of this report; the record of the table for S51 is that of the second run.') % (rf_['first_commit'][:7], rf_['fix_commit'][:7])
    npass = sum(1 for r_ in recs if r_['ok'])
    M('fullGatesPassed', npass); M('fullGatesSeconds', thousands(round(sum(r_['seconds'] for r_ in recs))))
    lg_ = os.path.join(MEAS, 'gates_all.log'); mw_ = re.search(r'(\d+)/(\d+) gates passed in (\d+)s', open(lg_).read()) if os.path.exists(lg_) else None
    if mw_: M('fullGatesWallMin', '%.0f' % (int(mw_.group(3)) / 60.0))
    else: MISSING.append('gates_all.log'); placeholder('fullGatesWallMin')
    desc = {}
    for line in open(os.path.join(ROOT, 'tests', 'README.md')).read().split('\n'):
        m_ = re.match(r'\|\s*`([^`\s]+)(?:\s[^`]*)?`\s*\|\s*[^|]*\|\s*(.*?)\s*\|\s*$', line)
        if m_: desc[os.path.basename(m_.group(1))] = m_.group(2)
    def short(t):
        t = re.split(r'(?<=[a-z\)`])\.\s|;\s| -- ', t)[0].strip().rstrip('.')
        t = t.replace('`', '')
        return (t[:117].rsplit(' ', 1)[0] + ' ...') if len(t) > 120 else t
    def gate_id(name):
        m_ = re.match(r's(\d+)', name)
        return 'S%s' % m_.group(1) if m_ else name.split('.')[0].replace('_gate', '')
    order = {'kernel': 0, 'library': 1, 'environment': 2, 'problems': 3, 'support': 4, 'learner': 5, 'synth': 6, 'repo': 7}
    def sortkey(r_):
        m_ = re.match(r's(\d+)', r_['gate']); return (order.get(r_['group'].split('/')[-1], 9), int(m_.group(1)) if m_ else 999, r_['gate'])
    rows = []
    for r_ in sorted(recs, key=sortkey):
        assert r_['gate'] in desc, 'no description of gate %s in tests/README.md' % r_['gate']
        rows.append('%s & \\code{%s/\\allowbreak{}%s} & %s & %s \\\\' % (gate_id(r_['gate']), r_['group'].split('/')[-1], tex_escape(r_['gate']).replace('\\_', '\\_\\allowbreak{}'), ('%.0f' % r_['seconds']) if r_['seconds'] >= 1 else '<1', tex_escape(short(desc[r_['gate']]))))
    M('gateIndexRows', '\n'.join(rows))
    if npass == 54 and rerun_note: M('fullGatesSentence', 'All %d gates pass.%s The logs, one record per gate, are in the supplementary material.' % (npass, rerun_note))
    elif npass == 54: M('fullGatesSentence', 'All %d gates pass at commit %s. The log of the run, one record per gate, is in the supplementary material.' % (npass, mach['commit'][:7]))
    else: M('fullGatesSentence', '%d of the %d gates pass at commit %s; the others are listed with their verdicts in the supplementary material.' % (npass, len(recs), mach['commit'][:7]))
else:
    MISSING.append('gates_all.jsonl'); placeholder('fullGatesPassed', 'fullGatesSeconds', 'fullGatesWallMin', 'gateIndexRows', 'fullGatesSentence')

# the two commands of the usage section, timed: measurements/kernel_build.log is `build_kernel.py all --force` into an empty directory, measurements/gates_quick.log is `make gates-quick`
def real_seconds(path):
    mm = re.search(r'^real\s+(\d+)m([\d.]+)s', open(path).read(), re.M)
    return int(mm.group(1)) * 60 + float(mm.group(2))
for nm_, fn_, fmt_ in (('kernelBuildS', 'kernel_build.log', '%.0f'), ('gatesQuickS', 'gates_quick.log', '%.1f')):
    p_ = os.path.join(MEAS, fn_)
    if os.path.exists(p_): M(nm_, fmt_ % real_seconds(p_))
    else: MISSING.append(fn_); placeholder(nm_)

# ----------------------------------------------------------------------------------------------------------------------------------------------- write
with open(OUT, 'w') as f:
    f.write('%% generated by preprint/make_numbers.py from preprint/measurements/*.json and the files git tracks -- do not edit by hand\n')
    for k in sorted(macros):
        f.write('\\newcommand{\\%s}{%s}\n' % (k, macros[k]))
print('wrote %s: %d macros' % (OUT, len(macros)))
if MISSING and '--allow-missing' not in sys.argv:
    print('MISSING measurement files (run again when they exist, or pass --allow-missing for a draft): %s' % ', '.join(MISSING)); sys.exit(1)
if '--print' in sys.argv:
    for k in sorted(macros): print('%-22s %s' % (k, macros[k].replace('\n', ' | ')[:150]))
