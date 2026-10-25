"""S49 gate: the CONTEST problems -- the AMC/AIME tier and the IMO tier of the bank (modules bank_amc, bank_amc2, bank_aime2, bank_imo, bank_imo2, bank_imo3, bank_imo4, bank_classic): 196 problems on top of the
72 graded problems of S41/S43 -- their provenance, their statements, and the machinery of the STATEMENT-ONLY tier.

The two earlier bank gates pin the graded problems.  The contest problems differ in three ways, and this gate is about those:

  * TIERS.  A problem is PROVED (a kernel-checked reference proof) or STATEMENT-ONLY (the formal statement and its checked meaning, no proof: 81 of the 268 problems, among them 80 of the 102 IMO-tier problems).
    A statement-only problem is only as good as the evidence that its statement is TRUE and MEANS what the problem says: the faithfulness check of semantic.py, the truth screen on the box, and brute-force
    claims over ranges far beyond the box (imo_truth.py).
  * PROVENANCE.  `origin` (contest / adapted / style / folklore) and `verified` (answer-checked / recalled / none) say where a problem comes from and whether its text or answer was compared with a public
    source.  The composition of the two tiers is PINNED here, because it is what a paper has to report: 72 of the 103 AMC/AIME-tier problems are `style` problems written for the bank, not AMC problems.
  * SPLITS.  The numbers of problems per split, in the whole bank and in the contest tiers, are pinned (they are the sizes of the pre-training and fine-tuning sets of the experiments).

  A. THE REGISTRY.     268 problems in the order of the modules; the tiers; the pinned composition (contest class x origin x verified x tier), levels and kinds; the pinned splits.
  B. THE EVIDENCE.     every statement-only problem has a brute-force claim and every claim holds; a claim that does not hold is reported.
  C. THE STATEMENTS.   in the built world: every contest problem passes its check (the statement is read back and agrees with its spec on the box, every mutant is told from it or explained, the statement is
                       sayable in the typed term builder, a statement without a proof is true on the box); statements are pinned by digest (s49_golden.json) and pairwise different.
  D. THE TIER.         negative controls: a false statement-only problem is rejected, a true one accepted; a statement-only problem that is no proposition, that declares a theorem, that has helpers, an unknown origin
                       or an unknown `verified` is refused.
  E. DUPLICATES.       no two problems say the same thing (bank_dups.py): the same statement, the same meaning under any renaming of the variables (the statements are decoded and compared by the numbers their
                       relation atoms are made of), alike texts (in one family, or reviewed and entered in REVIEWED_DIFFERENT with the reason); negative controls for the screens.

Environment: S49_TRACE=1 prints timings, S49_ONLY=A,C selects sections, S49_UPDATE_GOLDEN=1 rewrites the golden digests.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, time, json, collections, traceback
sys.setrecursionlimit(200000)
import itp_env_lnl as E
import bank as B, bank_lib, bank_all, bank_split as BS, bank_dups as DUPS, imo_truth, semantic as SEM
import bank_kit as K
from bank_spec import spec, LT, LE, EQ, V as X
from author import AuthorError

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S49_TRACE'))
ONLY = set(os.environ['S49_ONLY'].split(',')) if os.environ.get('S49_ONLY') else None
GOLDEN = os.path.join(itp_paths.GOLDEN, 's49_golden.json')


def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)


def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return bool(cond)


def raises(what, fn, exc):
    try: fn()
    except exc: return expect(True, what)
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:100]))
    return expect(False, what + ' (it raised nothing)')


AMC_AIME_MODULES = ('bank_amc', 'bank_amc2', 'bank_aime2')
IMO_MODULES = ('bank_imo', 'bank_imo2', 'bank_imo3', 'bank_imo4', 'bank_classic')
CONTEST_MODULES = AMC_AIME_MODULES + IMO_MODULES
OLDER_IMO = ('imo1959_p1', 'imo1964_p1a', 'imo1964_p1b', 'imo1964_p2', 'imo1969_p1', 'imo1982_p1', 'imo1984_p2', 'imo1988_p6', 'imo2019_p1')       # the real IMO problems of bank_lib / bank_lib2 (gates S41, S43)
N_FIRST, N_SECOND = 21, 51                                                                                                                         # the problems of bank_lib, of bank_lib2

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ pins
TOTAL, PROVED, STATEMENT_ONLY = 268, 187, 81
AMC_AIME_TIER, IMO_TIER = 103, 102          # the AMC/AIME tier: the three AMC/AIME modules; the IMO tier: the five IMO modules and the nine older IMO problems
# (class of the source, origin, verified, tier) -> number of problems.  A paper reports this table: it moves only on purpose.
COMPOSITION = {
    'amc_aime': {('AIME', 'adapted', 'answer-checked', 'proved'): 8, ('AIME', 'contest', 'answer-checked', 'proved'): 21, ('AIME', 'contest', 'answer-checked', 'statement-only'): 1,
                 ('AIME', 'contest', 'recalled', 'proved'): 1, ('other', 'style', 'none', 'proved'): 72},
    'imo': {('APMO', 'contest', 'answer-checked', 'statement-only'): 1, ('IMO', 'adapted', 'answer-checked', 'statement-only'): 5, ('IMO', 'adapted', 'recalled', 'proved'): 1,
            ('IMO', 'contest', 'answer-checked', 'proved'): 9, ('IMO', 'contest', 'answer-checked', 'statement-only'): 22, ('IMO', 'contest', 'recalled', 'proved'): 1,
            ('IMO', 'contest', 'recalled', 'statement-only'): 3, ('ISL', 'adapted', 'answer-checked', 'statement-only'): 4, ('ISL', 'adapted', 'recalled', 'statement-only'): 1,
            ('ISL', 'contest', 'answer-checked', 'statement-only'): 11, ('ISL', 'contest', 'recalled', 'proved'): 1, ('ISL', 'contest', 'recalled', 'statement-only'): 1,
            ('folklore', 'folklore', 'none', 'proved'): 10, ('other', 'folklore', 'answer-checked', 'statement-only'): 32},
}
LEVELS = {'amc_aime': {1: 20, 2: 42, 3: 38, 4: 3}, 'imo': {2: 1, 3: 18, 4: 39, 5: 44}}
KINDS = {'amc_aime': {'answer': 68, 'exists': 1, 'prove': 34}, 'imo': {'answer': 47, 'exists': 1, 'prove': 54}}
# the splits (salt number 25, by family): problems per split, and how many of them are proved
SPLITS_ALL = {'train': (190, 132), 'dev': (20, 15), 'test': (58, 40)}
SPLITS_TIERS = {'amc_aime': {'train': (75, 75), 'dev': (9, 8), 'test': (19, 19)}, 'imo': {'train': (72, 14), 'dev': (4, 0), 'test': (26, 8)}}
MAX_SKIPPED_MUTANTS = 30                    # mutants explained (not tested) in the contest modules: the explanations are few and each has a reason


def contest_class(source):
    for pre in ('IMO', 'ISL', 'AIME', 'AMC', 'APMO', 'folklore'):
        if source.startswith(pre): return pre
    return 'other'


def tiers():
    """{'amc_aime': [names], 'imo': [names]}"""
    mods = bank_all.MODULES
    return {'amc_aime': [n for m in AMC_AIME_MODULES for n in mods[m]], 'imo': [n for m in IMO_MODULES for n in mods[m]] + list(OLDER_IMO)}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. the registry
def sec_A(c):
    P = B.BANK.problems; names = list(P); mods = bank_all.MODULES
    expect(len(names) == len(set(names)) == TOTAL and bank_all.names() == names, 'A1 the registry holds %d problems, each once (%d)' % (TOTAL, len(names)))
    expect(sorted(mods) == sorted(CONTEST_MODULES + ('bank_lib2',)), 'A1 bank_all knows the modules %s (%s)' % (CONTEST_MODULES + ('bank_lib2',), sorted(mods)))
    flat = [n for m in ('bank_lib2',) + CONTEST_MODULES for n in mods[m]]
    expect(names[:N_FIRST] + flat == names and len(mods['bank_lib2']) == N_SECOND, 'A2 the registry is the problems of bank_lib (%d), bank_lib2 (%d) and then the contest modules, in this order' % (N_FIRST, N_SECOND))
    expect(all(n in P for ns in mods.values() for n in ns) and sum(len(ns) for ns in mods.values()) == TOTAL - N_FIRST, 'A2 the modules list exactly the problems they register')
    proved = bank_all.proved_names(); stmt = bank_all.statement_only_names()
    expect(len(proved) == PROVED and len(stmt) == STATEMENT_ONLY and set(proved) | set(stmt) == set(names) and not set(proved) & set(stmt), 'A3 %d problems are proved and %d are statement-only (%d, %d)' % (PROVED, STATEMENT_ONLY, len(proved), len(stmt)))
    expect(all(P[n].build is not None for n in stmt), 'A3 a statement-only problem is built by the function that says its statement')
    T = tiers()
    expect(len(T['amc_aime']) == AMC_AIME_TIER and len(T['imo']) == IMO_TIER and not set(T['amc_aime']) & set(T['imo']) and all(n in P for ns in T.values() for n in ns), 'A4 the AMC/AIME tier has %d problems and the IMO tier %d, disjoint (%d, %d)' % (AMC_AIME_TIER, IMO_TIER, len(T['amc_aime']), len(T['imo'])))
    expect(len(set(T['amc_aime']) | set(T['imo'])) == AMC_AIME_TIER + IMO_TIER == 205 and len(set(names) - set(T['amc_aime']) - set(T['imo'])) == TOTAL - 205, 'A4 together 205 contest problems; the other %d are the graded ones of S41/S43' % (TOTAL - 205))
    expect(all(P[n].source.startswith('IMO') for n in OLDER_IMO), 'A4 the nine older problems of the IMO tier are IMO problems')
    # ---- A5. provenance
    for tier, ns in T.items():
        comp = collections.Counter((contest_class(P[n].source), P[n].origin, P[n].verified, 'proved' if P[n].proved else 'statement-only') for n in ns)
        expect(dict(comp) == COMPOSITION[tier], 'A5 the composition of the %s tier is the pinned one (differences: %s)' % (tier, sorted((k, comp.get(k, 0), COMPOSITION[tier].get(k, 0)) for k in set(comp) | set(COMPOSITION[tier]) if comp.get(k, 0) != COMPOSITION[tier].get(k, 0))[:6]))
        expect(dict(collections.Counter(P[n].level for n in ns)) == LEVELS[tier], 'A5 ... the levels of the %s tier are the pinned ones (%s)' % (tier, dict(collections.Counter(P[n].level for n in ns))))
        expect(dict(collections.Counter(P[n].kind for n in ns)) == KINDS[tier], 'A5 ... and its kinds (%s)' % (dict(collections.Counter(P[n].kind for n in ns)),))
    # the rules between the fields
    bad = []
    for n in names:
        p = P[n]
        if p.origin == 'style' and (p.verified != 'none' or p.proved is False): bad.append((n, 'style problems carry no source check and have proofs'))
        if p.origin in ('contest', 'adapted') and p.verified == 'none': bad.append((n, 'a problem of a contest is answer-checked or recalled'))
        if p.origin in ('contest', 'adapted') and not (p.source.startswith(('IMO', 'ISL', 'AIME', 'APMO', 'AMC')) and p.year): bad.append((n, 'a contest problem names its contest and year'))
        if p.kind == 'answer' and not p.answer: bad.append((n, 'an answer problem has its answer'))
        if not (p.text.isascii() and len(p.text) > 20): bad.append((n, 'ASCII text'))
        if not p.proved and p.spec is None: bad.append((n, 'a statement without a proof needs its spec'))
        if p.verified == 'answer-checked' and p.origin != 'style' and not p.proved and n not in imo_truth.CLAIMS: bad.append((n, 'an answer-checked statement-only problem has its brute-force claim'))
    expect(not bad, 'A6 the rules between origin, verified, source and tier hold (%s)' % (bad[:3],))
    # the AMC-style problems are style problems: this is what the paper must say
    style = [n for n in names if P[n].origin == 'style']
    expect(len(style) == 72 and set(style) == set(mods['bank_amc2']), 'A7 the 72 problems of bank_amc2 are the `style` problems (written for the bank in the manner of AMC; no claim to be AMC problems)')
    # ---- A8. the splits
    sp = BS.splits(names)
    expect({s: (len(ns), sum(1 for n in ns if P[n].proved)) for s, ns in sp.items()} == SPLITS_ALL, 'A8 the splits of the bank are the pinned ones (%s)' % ({s: (len(ns), sum(1 for n in ns if P[n].proved)) for s, ns in sp.items()},))
    for tier, ns in T.items():
        s_ = set(ns)
        got = {s: (len([n for n in v if n in s_]), len([n for n in v if n in s_ and P[n].proved])) for s, v in sp.items()}
        expect(got == SPLITS_TIERS[tier], 'A8 ... and those of the %s tier (%s)' % (tier, got))
    assign = {n: s for s, ns in sp.items() for n in ns}
    expect(BS.straddling(assign) == [] and BS.coverage({n: P[n] for n in names}) == [], 'A8 no family straddles two splits and every split covers the levels and topics it should (%s)' % (BS.coverage({n: P[n] for n in names}),))
    expect(all(n in P for n in BS.FAMILIES), 'A8 the families name problems of the bank')
    # ---- A9. alike texts: one family, or reviewed (the screen of bank_dups.py)
    pairs = DUPS.text_pairs({n: P[n] for n in names})
    seen = {frozenset((a, b)) for _, a, b in pairs}
    open_ = [(r, a, b) for r, a, b in pairs if BS.family(a) != BS.family(b) and frozenset((a, b)) not in DUPS.REVIEWED_DIFFERENT]
    expect(not open_, 'A9 every pair of problems with alike texts (ratio >= %.2f) is one family or has been reviewed as two problems: %d pairs, open: %s -- put them in bank_split.FAMILIES or in bank_dups.REVIEWED_DIFFERENT, with the reason' % (DUPS.TEXT_RATIO, len(pairs), open_[:3]))
    stale = [sorted(k) for k in DUPS.REVIEWED_DIFFERENT if k not in seen]
    expect(not stale, 'A9 every reviewed pair is still alike (a stale entry hides nothing but rots: %s)' % (stale[:3],))
    contradict = [sorted(k) for k in DUPS.REVIEWED_DIFFERENT if len({BS.family(x) for x in k}) == 1]
    expect(not contradict and all(len(why) >= 40 for why in DUPS.REVIEWED_DIFFERENT.values()) and all(all(x in P for x in k) for k in DUPS.REVIEWED_DIFFERENT), 'A9 a reviewed pair is two families, with a reason, of problems that exist (%s)' % (contradict[:2],))
    # the screen itself: it finds alike texts and renamed variables, and not different problems
    import types
    fake = {k: types.SimpleNamespace(text=t) for k, t in (('p', 'For all integers a and b: 2ab <= a^2 + b^2.'), ('q', 'For all integers x and y: 2xy <= x^2 + y^2.'), ('r', 'Find all pairs of positive integers (x, y) with x^2 - y^2 = 15.'),
                                                           ('s', 'Prove that n^4 + 4 is not a prime for any integer n > 1.'))}
    got = DUPS.text_pairs(fake)
    expect([(a, b) for _, a, b in got] == [('p', 'q')], 'A9 the text screen finds a statement written twice with other variables, and nothing else (%s)' % (got,))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. the evidence
def sec_B(c):
    stmt = set(bank_all.statement_only_names()); claims = set(imo_truth.CLAIMS)
    expect(claims == stmt, 'B1 every statement-only problem has a brute-force claim and every claim is about one (missing %s, extra %s)' % (sorted(stmt - claims)[:4], sorted(claims - stmt)[:4]))
    t0 = time.time(); bad = imo_truth.run(None, out=lambda s: trace('B ' + s))
    expect(not bad, 'B2 the brute-force claim of every statement-only problem holds (%d claims, %.0f s; failed: %s)' % (len(imo_truth.CLAIMS), time.time() - t0, bad[:4]))
    # a claim that does not hold is reported
    saved = dict(imo_truth.CLAIMS)
    try:
        imo_truth.CLAIMS['s49_false_claim'] = (lambda: False, 'nothing', 'range')
        imo_truth.CLAIMS['s49_true_claim'] = (lambda: True, 'nothing', 'range')
        got = imo_truth.run(['s49_false_claim', 's49_true_claim'], out=lambda s: None)
    finally: imo_truth.CLAIMS.clear(); imo_truth.CLAIMS.update(saved)
    expect(got == ['s49_false_claim'], 'B3 a claim that is false is reported, a true one is not (%s)' % (got,))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. the statements
class Fixture:
    def __init__(self):
        t0 = time.time()
        self.bk = bank_all.load(); self.w = self.bk.w; self.k = self.w.k
        trace('bank world: %d globals, %d problems, %.1f s' % (self.k.num_globals(), len(self.bk.placed), time.time() - t0))


def sec_C(c):
    fx = c['fx']; bk = fx.bk; w = fx.w; P = bk.placed
    contest = [n for m in CONTEST_MODULES for n in bank_all.MODULES[m]]
    expect(len(P) == TOTAL and list(P) == bank_all.names(), 'C1 the world has all %d problems, in the order of the registry' % TOTAL)
    expect(len(contest) == 196, 'C1 %d problems in the contest modules' % len(contest))
    t0 = time.time(); bad = []; stats = collections.Counter(); slow = []
    for n in contest:
        t1 = time.time(); rep = bk.check(n)
        if time.time() - t1 > 20: slow.append((n, round(time.time() - t1, 1)))
        if not rep.ok: bad.append((n, rep.errors[:1]))
        s = rep.semantics
        if s is not None:
            stats['points'] += s.points; stats['mutants'] += s.mutants; stats['caught'] += s.caught; stats['equivalent'] += s.equivalent; stats['survivors'] += len(s.survivors)
            if s.nonvacuous == 0 and not P[n].problem.spec.vacuous: bad.append((n, 'vacuous'))
            if not P[n].proved and s.n_counter: bad.append((n, 'a statement without a proof is false on the box'))
        if rep.code is not None: stats['sayable'] += 1
    expect(not bad, 'C2 every contest problem passes its check: statement read, meaning agrees with the spec on the box, every mutant caught or explained, sayable, and (without a proof) true on the box (%s)' % (bad[:3],))
    expect(stats['sayable'] == len(contest), 'C2 every statement is sayable in the typed term builder (%d of %d)' % (stats['sayable'], len(contest)))
    expect(stats['mutants'] >= 8000 and stats['caught'] == stats['mutants'] and not stats['survivors'], 'C3 the negative controls: %d mutants of the contest statements, all of them caught (%d), %d more explained as equivalent, %d box points' % (stats['mutants'], stats['caught'], stats['equivalent'], stats['points']))
    trace('C2 checks %.1f s, %s, slow: %s' % (time.time() - t0, dict(stats), slow))
    skips = [(n, d, why) for n in contest if P[n].problem.spec is not None for d, why in P[n].problem.spec.skip_mutants.items()]
    expect(all(len(why) >= 25 for _, _, why in skips) and len(skips) <= MAX_SKIPPED_MUTANTS, 'C4 the mutants explained as equivalent are few (%d <= %d) and each has a reason' % (len(skips), MAX_SKIPPED_MUTANTS))
    # ---- C5. the tiers in the world
    stmt_only = [n for n in P if not P[n].proved]
    expect(len(stmt_only) == STATEMENT_ONLY and all(P[n].gid is None and n not in w.gid for n in stmt_only), 'C5 no statement-only problem is a theorem of the world (%d)' % len(stmt_only))
    expect(all(P[n].gid is not None and w.gid[n] == P[n].gid and P[n].start <= P[n].defs_end <= P[n].gid for n in P if P[n].proved), 'C5 a proved problem is the theorem of its name, after its block')
    starts = [P[n].start for n in P]
    expect(all(a <= b for a, b in zip(starts, starts[1:])), 'C5 the blocks of the problems follow one another in the order of the registry')
    k = w.k
    expect(all(w.infer(P[n].stmt).n == w.PROP.n for n in P), 'C5 every statement is a proposition')
    # ---- C6. pinned statements
    digests = {n: B.statement_digest(w, P[n].stmt) for n in contest}
    if os.environ.get('S49_UPDATE_GOLDEN'):
        with open(GOLDEN, 'w') as f: json.dump({'statements': digests}, f, indent=0, sort_keys=True)
        print('   golden digests written: %d statements' % len(digests))
    gold = json.load(open(GOLDEN)) if os.path.exists(GOLDEN) else {'statements': {}}
    diff = sorted(n for n in set(gold['statements']) | set(digests) if gold['statements'].get(n) != digests.get(n))
    expect(not diff, 'C6 the statements are those in %s: %d of %d differ (%s) -- if the change is intended: S49_UPDATE_GOLDEN=1 python3 s49_contest_tests.py' % (os.path.basename(GOLDEN), len(diff), len(digests), diff[:4]))
    alld = {n: B.statement_digest(w, P[n].stmt) for n in P}
    dup = [(a, b) for a in alld for b in alld if a < b and alld[a] == alld[b]]
    expect(not dup, 'C6 no two problems of the whole bank have the same statement (%s)' % (dup[:3],))
    old = json.load(open(os.path.join(itp_paths.GOLDEN, 's41_golden.json')))['statements']; old2 = json.load(open(os.path.join(itp_paths.GOLDEN, 's43_golden.json')))['statements']
    expect(all(alld.get(n) == d for n, d in list(old.items()) + list(old2.items())) and len(old) + len(old2) == N_FIRST + N_SECOND, 'C6 the statements of the %d graded problems are still those of s41_golden.json and s43_golden.json: the world below the contest problems did not move' % (N_FIRST + N_SECOND))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. the tier
def sec_D(c):
    # ---- D1. the metadata refuses what is wrong
    mk = lambda **kw: B.Problem(kw.pop('name', 'p'), kw.pop('text', 'a text of some length'), kw.pop('source', 'test'), kw.pop('topics', ('algebra',)), kw.pop('level', 1), **kw)
    raises('D1 an unknown origin is refused', lambda: mk(origin='rumour'), ValueError)
    raises('D1 an unknown `verified` is refused', lambda: mk(verified='trust me'), ValueError)
    raises('D1 a statement-only problem needs the function that says its statement', lambda: mk(proved=False), ValueError)
    raises('D1 a statement-only problem has no helper lemmas', lambda: mk(proved=False, build=lambda w, b: None, helpers=('h',)), ValueError)
    p1 = mk(source='folklore (an exercise)'); p2 = mk(source='IMO 1959 P1'); p3 = mk(source='IMO 1959 P1', origin='adapted', verified='recalled')
    expect((p1.origin, p2.origin, p3.origin, p3.verified, p2.verified) == ('folklore', 'contest', 'adapted', 'recalled', 'none'), 'D1 the origin follows the source unless it is given; `verified` is `none` unless it is given')
    # ---- D2. a false statement-only problem is rejected, a true one accepted
    bank = B.Bank()

    @bank.statement('s49_false', text='For every positive integer n: n^2 < n.', source='test', topics=('algebra',), level=1, spec=spec(('Z',), [LT(0, X(0))], LT(lambda n: n * n, X(0))))
    def _(w, b):
        v = K.Vocab(w)
        with v.ints(): return K.statement(w, [v.Z], [lambda n: v.pos(n)], lambda n: v.lt(n * n, n))

    @bank.statement('s49_true', text='For every integer n: 0 <= n^2.', source='test', topics=('algebra', 'inequality'), level=1, spec=spec(('Z',), [], LE(0, lambda n: n * n)))
    def _(w, b):
        v = K.Vocab(w)
        with v.ints(): return K.statement(w, [v.Z], [], lambda n: v.le(v.zero, n * n))

    @bank.statement('s49_wrong_meaning', text='For every integer n: 0 <= n^2.', source='test', topics=('algebra', 'inequality'), level=1, spec=spec(('Z',), [], LE(0, lambda n: n * n)))
    def _(w, b):
        v = K.Vocab(w)
        with v.ints(): return K.statement(w, [v.Z], [], lambda n: v.lt(v.zero, n * n))

    w1, placed = B.build_world(bank_lib.base, bank); lb = B.LoadedBank(w1, placed, bank, bank_lib.base)
    rep = lb.check('s49_false', code=False)
    expect(not rep.ok and any('FALSE' in e for e in rep.errors), 'D2 a statement-only problem that is false on the box is rejected (%s)' % (rep.errors[:1],))
    rep = lb.check('s49_true')
    expect(rep.ok and rep.code is not None and rep.semantics is not None and rep.semantics.n_counter == 0, 'D2 a true statement-only problem with the right spec is accepted (%s)' % (rep.errors[:1],))
    rep = lb.check('s49_wrong_meaning', code=False)
    expect(not rep.ok and rep.semantics is not None and not rep.semantics.ok, 'D2 a statement that does not mean what its spec says is rejected as for a proved problem (%s)' % (rep.errors[:1],))
    pl = placed['s49_true']
    expect(not pl.proved and pl.gid is None and 's49_true' not in w1.gid and pl.premises('prefix') == pl.premises('strict'), 'D2 a statement-only problem has no theorem; its two regimes offer the same globals')
    ep = lb.episode('s49_true')
    expect(ep.n_goals == 1 and not ep.done, 'D2 ... and is still a task: an episode starts on its statement')
    rec = lb.record('s49_true')
    expect(rec['proved'] is False and rec['gid'] is None and 'demonstration' not in rec and rec['origin'] == 'contest' and rec['verified'] == 'none', 'D2 its record says so (%s)' % ({k: rec[k] for k in ('proved', 'gid', 'origin', 'verified')},))
    # ---- D3. what the build function of a statement-only problem may not do
    def one(builder, **kw):
        b2 = B.Bank(); b2.statement('s49_x', text='a statement for a test', source='test', topics=('algebra',), level=1, **kw)(builder)
        return lambda: B.build_world(bank_lib.base, b2)

    def not_a_prop(w, b):
        v = K.Vocab(w)
        with v.ints(): return v.lit(3)
    raises('D3 a statement that is no proposition is refused', one(not_a_prop), ValueError)

    def declares_theorem(w, b):
        from nt_lib import _syms
        from lnl_base import thm
        s = _syms(w); v = K.Vocab(w)
        with v.ints():
            thm(w, 's49_x', [s.Z], lambda a: s.Eq(a, a))(lambda p: (p.intro('a'), p.refl()))
            return K.statement(w, [v.Z], [], lambda n: v.le(v.zero, n * n))
    raises('D3 a statement-only problem that declares a theorem of its own name is refused', one(declares_theorem), ValueError)



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. duplicates
def sec_E(c):
    fx = c['fx']; bk = fx.bk; P = bk.placed
    t0 = time.time(); reads = DUPS.readable(bk); pairs = DUPS.meaning_pairs(bk)
    expect(len(reads) >= 200, 'E1 the screen reads %d of the %d statements (the others quantify over functions or are too large for the evaluator; their texts are screened in A9)' % (len(reads), len(P)))
    expect(not pairs, 'E1 no two problems of the bank have the same meaning (the statements, decoded and evaluated on a grid, have the same atoms up to the order and the names of the variables): %s' % (pairs[:3],))
    trace('E1 meaning screen: %d statements, %.1f s' % (len(reads), time.time() - t0))
    # ---- E2. the screen itself, on a bank of its own: the same inequality written three ways and a different one
    bank = B.Bank()

    def stmt(name, kinds, build, params, conclusion):
        @bank.statement(name, text='a statement for a test of the duplicate screen', source='test', topics=('inequality',), level=1, spec=spec(params, [], conclusion))
        def _(w, b):
            v = K.Vocab(w)
            with v.ints(): return K.statement(w, [v.Z] * kinds, [], lambda *xs: build(v, *xs))
    stmt('s49_dup_a', 2, lambda v, a, c: v.le(2 * a * c, a ** 2 + c ** 2), ('Z', 'Z'), LE(lambda a, c: 2 * a * c, lambda a, c: a * a + c * c))
    stmt('s49_dup_b', 2, lambda v, y, x: v.le(x * y * 2, x * x + y * y), ('Z', 'Z'), LE(lambda y, x: x * y * 2, lambda y, x: x * x + y * y))                 # (the parameters in the other order, x*x for x^2)
    stmt('s49_dup_c', 3, lambda v, p, q, r: v.le(2 * p * r, p ** 2 + r ** 2), ('Z', 'Z', 'Z'), LE(lambda p, q, r: 2 * p * r, lambda p, q, r: p * p + r * r))   # (a parameter that is not used: not the same statement)
    stmt('s49_other', 2, lambda v, a, c: v.le(4 * a * c, (a + c) ** 2), ('Z', 'Z'), LE(lambda a, c: 4 * a * c, lambda a, c: (a + c) ** 2))
    w1, placed = B.build_world(bank_lib.base, bank); lb = B.LoadedBank(w1, placed, bank, bank_lib.base)
    got = DUPS.meaning_pairs(lb)
    expect([(a, b) for a, b, _ in got] == [('s49_dup_a', 's49_dup_b')], 'E2 the screen finds the inequality written twice (other order of the parameters, other notation) and neither the version with a parameter it does not use nor a different inequality (%s)' % (got,))
    expect(B.statement_digest(w1, placed['s49_dup_a'].stmt) != B.statement_digest(w1, placed['s49_dup_b'].stmt), 'E2 ... although the two statements are different terms (the digest screen of C6 does not see them)')


SECTIONS = [('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E)]


def main():
    t_start = time.time()
    c = {}
    for name, fn in SECTIONS:
        if ONLY and name not in ONLY: continue
        if name in ('C', 'E') and 'fx' not in c: c['fx'] = Fixture()
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:300])); traceback.print_exc(limit=8)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
        if FAIL and os.environ.get('S49_FAILFAST'): break
    print('S49: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:60]: print('  FAIL:', f)
    print('S49 CONTEST GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
