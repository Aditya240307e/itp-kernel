"""S50 gate: the PREMISES that mask the world, and the PROOFS of the contest tier.

A problem of the bank is attacked in a world that holds the library and the blocks of ALL the problems; what an attempt may USE is decided by its premises (gym_premises.py): the library, plus the problem's own
definitions and helper lemmas, and nothing of any other problem.  Without that mask a late problem would see the theorems of the early ones -- the solutions of other problems, the leak between a training
split and a test split, and not the library a model was pre-trained on.  This gate pins the mask, and then the thing every claim of the project rests on: the reference proofs of the 115 PROVED problems of the
contest tiers (the 72 graded problems are pinned by S41/S43) are kernel-checked action sequences that use nothing the mask hides, and a second, independently written checker accepts them as well.

  A. PREMISES AS VALUES.   gym_premises.Premises: normal form, `sees`, `count`, equality and hashing, the JSON form and its reading, `for_block`; what is refused.
  B. THE LEAK.             a bank of three small problems in a world of its own: with a plain prefix of the world the theorem of an EARLIER problem is on the menu of a later one and one action finishes it; with the mask
                           it is not on the menu, the typed term builder refuses to say it, the action that uses it is refused, and the recorder reports the gap when a library lemma a proof needs is hidden.
  C. THE MASK IN THE BANK. in the whole world (268 problems): the premises of every problem are exactly the library and its own block; no theorem, helper or definition of another problem is visible; a proved problem
                           does not see its own theorem; contest problems (no helpers) see the library and their definitions; the engine agrees with the premises on every global; on a sample of problems the menu under the mask is
                           the plain-prefix menu without hidden globals (and the mask removes something).
  D. THE PROOFS.           each of the 115 proved contest problems: its reference proof replays as a kernel-checked sequence of actions in its regime; the steps survive a round trip through JSON and replay again; the proof term
                           mentions only globals the problem's premises offer (and the ones the replay itself derived); the proof depends on no axiom and on the quotient primitives only (audit.py).
  E. THE SECOND OPINION.   the independent checker (recheck.SecondOpinion) accepts every one of those proofs for its statement and rejects a proof for another problem's statement.

Environment: S50_TRACE=1 prints timings and memory, S50_ONLY=A,B selects sections, S50_LIMIT=n restricts D and E to the first n proved contest problems (a quick run; the gate itself runs them all).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, time, json, random, collections, traceback
sys.setrecursionlimit(200000)
import itp_env_lnl as E
import bank as B, bank_lib, bank_all
import gym_env as G, gym_acts as A, gym_build as GB, gym_premises as PM, recorder as REC
from gym_env import Kind, Action
from author import AuthorError
from nt_lib import _syms
from lnl_base import thm
import recheck as RC, audit as AU

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S50_TRACE'))
ONLY = set(os.environ['S50_ONLY'].split(',')) if os.environ.get('S50_ONLY') else None
LIMIT = int(os.environ['S50_LIMIT']) if os.environ.get('S50_LIMIT') else None


def rss_mb():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'): return int(line.split()[1]) // 1024
    except OSError: pass
    return 0


def trace(msg):
    if TRACE: print('   [trace %s | %d MB]' % (msg, rss_mb()), flush=True)


def expect(cond, what):
    CHECKS[0] += 1
    if not cond:
        FAIL.append(what)
        if TRACE: print('   [FAIL %s]' % what, flush=True)
    return bool(cond)


def raises(what, fn, exc):
    try: fn()
    except exc: return expect(True, what)
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:100]))
    return expect(False, what + ' (it raised nothing)')


REFUSED = (G.ActionError, AuthorError, E.ITPError)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ pins
LIBRARY_END = 549                       # the globals below the first problem of the bank: the library every problem is attacked with
N_PROVED_CONTEST = 115                  # the proved problems of the modules after bank_lib2 (the 72 graded problems are S41/S43's)
TOTAL, N_GRADED = 268, 72
MIN_ACTIONS = 1000                      # the reference proofs of the contest tier are at least this many actions in all (calibrated below)
QUOT = {'Quot', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound'}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. premises as values
def sec_A(c):
    P = PM.Premises
    # ---- A1. the normal form
    p = P(20, [(5, 8), (2, 6), (8, 9)])
    expect(p.hidden == ((2, 9),), 'A1 overlapping and adjacent ranges are merged (%s)' % (p.hidden,))
    expect(P(10, [(8, 15), (12, 14)]).hidden == ((8, 10),), 'A1 a range beyond `upto` is clipped, one wholly beyond it dropped (%s)' % (P(10, [(8, 15), (12, 14)]).hidden,))
    expect(P(10, [(3, 3), (4, 4)]).hidden == () and P(10).hidden == (), 'A1 empty ranges are dropped')
    expect(P(10, [(6, 7), (1, 2)]).hidden == ((1, 2), (6, 7)), 'A1 the ranges are sorted')
    expect(P(0).count() == 0 and not P(0).sees(0), 'A1 premises of nothing offer nothing')
    raises('A1 a negative `upto` is refused', lambda: P(-1), ValueError)
    raises('A1 a boolean `upto` is refused', lambda: P(True), ValueError)
    raises('A1 a float `upto` is refused', lambda: P(3.0), ValueError)
    raises('A1 a range that runs backwards is refused', lambda: P(10, [(5, 3)]), ValueError)
    raises('A1 a range with a negative start is refused', lambda: P(10, [(-1, 3)]), ValueError)
    # ---- A2. sees and count agree, on random premises
    rng = random.Random(50); bad = []
    for _ in range(300):
        upto = rng.randrange(0, 40); hid = []
        for _ in range(rng.randrange(0, 4)):
            a = rng.randrange(0, 45); hid.append((a, a + rng.randrange(0, 12)))
        p = P(upto, hid)
        brute = [g for g in range(-3, 60) if 0 <= g < upto and not any(a <= g < b for a, b in hid)]
        if [g for g in range(-3, 60) if p.sees(g)] != brute or p.count() != len(brute): bad.append((upto, hid))
    expect(not bad, 'A2 `sees` and `count` agree with the definition on 300 random premises (%s)' % (bad[:2],))
    expect(P(10, [(3, 5)]).sees(2) and not P(10, [(3, 5)]).sees(3) and not P(10, [(3, 5)]).sees(4) and P(10, [(3, 5)]).sees(5) and P(10, [(3, 5)]).sees(9) and not P(10, [(3, 5)]).sees(10), 'A2 the boundaries: a hidden range is half open, `upto` is exclusive')
    # ---- A3. a value: equality, hashing
    a, b2 = P(20, [(5, 8), (2, 6)]), P(20, [(2, 8)])
    expect(a == b2 and hash(a) == hash(b2) and a != P(20, [(2, 7)]) and a != P(21, [(2, 8)]) and a != 20 and {a: 1}[b2] == 1, 'A3 premises with the same normal form are equal and hash alike; others differ; a Premises is not an int')
    raises('A3 premises are immutable (an attribute cannot be set)', lambda: setattr(a, 'upto', 3), AttributeError)
    raises('A3 ... or deleted', lambda: delattr(a, 'hidden'), AttributeError)
    import pickle, copy
    expect(pickle.loads(pickle.dumps(a)) == a and copy.deepcopy(a) == a and copy.copy(a) == a and hash(pickle.loads(pickle.dumps(a))) == hash(a), 'A3 premises survive pickle and copy')
    # ---- A4. JSON
    for p in (P(7), P(20, [(2, 8)]), P(100, [(10, 20), (30, 31)])):
        j = json.loads(json.dumps(p.to_json()))
        expect(PM.of(j) == (p if p.hidden else p.upto), 'A4 %r survives JSON (%r)' % (p, j))
    expect(P(7).to_json() == 7 and P(20, [(2, 8)]).to_json() == {'upto': 20, 'hidden': [[2, 8]]}, 'A4 the JSON form is an int when nothing is hidden, else {upto, hidden}')
    expect(PM.of(None) is None and PM.of(5) == 5 and PM.of(P(5)) == 5 and PM.of(P(9, [(1, 2)])) == P(9, [(1, 2)]) and PM.of({'upto': 9}) == 9, 'A4 `of` reads None, an int, a Premises and the JSON form')
    raises('A4 `of` refuses a boolean', lambda: PM.of(True), ValueError)
    raises('A4 `of` refuses a string', lambda: PM.of('7'), ValueError)
    raises('A4 `of` refuses a dict with a key it does not know', lambda: PM.of({'upto': 5, 'hide': []}), ValueError)
    raises('A4 `of` refuses a dict without `upto`', lambda: PM.of({'hidden': []}), ValueError)
    expect(PM.to_json(None) is None and PM.to_json(5) == 5 and PM.to_json(P(9, [(1, 2)])) == {'upto': 9, 'hidden': [[1, 2]]}, 'A4 `to_json` of None, an int, a Premises')
    # ---- A5. the module functions
    expect(PM.upto(None, 30) == 30 and PM.upto(50, 30) == 30 and PM.upto(10, 30) == 10 and PM.upto(P(40, [(2, 4)]), 30) == 30 and PM.upto(P(8, [(2, 4)]), 30) == 8, 'A5 `upto`: the bound in a world of n globals')
    expect(PM.sees(None, 4) and not PM.sees(None, -1) and PM.sees(5, 4) and not PM.sees(5, 5) and not PM.sees(P(9, [(2, 4)]), 3) and PM.sees(P(9, [(2, 4)]), 4), 'A5 `sees` for None, an int and a Premises')
    expect(not PM.sees(None, 30, 30) and PM.sees(None, 29, 30) and not PM.sees(50, 30, 30), 'A5 ... and in a world of n globals nothing at or beyond n is seen')
    # ---- A6. the premises of a problem's block
    expect(PM.for_block(100, 80, 90) == 90 and PM.for_block(100, 100, 120) == 120, 'A6 `for_block`: a block inside the library is an ordinary prefix (an int)')
    f = PM.for_block(100, 130, 140)
    expect(f == P(140, [(100, 130)]) and f.count() == 110 and f.sees(99) and not f.sees(100) and not f.sees(129) and f.sees(130) and f.sees(139) and not f.sees(140), 'A6 a block after the library: the library and the block, nothing between (%r)' % (f,))
    f0 = PM.for_block(100, 130, 130)
    expect(f0.count() == 100 and not f0.sees(130), 'A6 a block of nothing offers exactly the library (%r)' % (f0,))
    raises('A6 premises that end before the block starts are refused', lambda: PM.for_block(100, 130, 120), ValueError)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. the leak
def leak_bank():
    """three problems in a world of their own:  A (with a helper lemma) and B and C state the same inequality, so that the theorem of A proves B in ONE action -- if A is visible"""
    bank = B.Bank()

    def mk(name, helper=False, hint='Z.sq_nonneg'):
        kw = dict(text='For all integers a and b: 2ab <= a^2 + b^2.', source='test', topics=('inequality',), level=1)
        if helper: kw['helpers'] = ('s50_leak_h',)
        @bank.problem(name, **kw)
        def _(w, b):
            s = _syms(w); Z, zle = s.Z, s.zle
            with w.nota.using(Z):
                if helper:
                    @thm(w, 's50_leak_h', [Z, Z], lambda a, c: zle(2 * a * c, a ** 2 + c ** 2))
                    def _(p): p.intro('a', 'c'); p.nlinarith((hint, lambda q: q.h('a') - q.h('c')))
                @thm(w, name, [Z, Z], lambda a, c: zle(2 * a * c, a ** 2 + c ** 2))
                def _(p): p.intro('a', 'c'); p.nlinarith((hint, lambda q: q.h('a') - q.h('c')))
    mk('s50_leak_a', helper=True); mk('s50_leak_b'); mk('s50_leak_c')
    return bank


def can_say(eng, ep, gid):
    """can the agent SAY the global `gid` in the first goal of the episode (the typed term builder offers only the premises of the episode)?"""
    g = ep.goal(); eng.premises = ep.premises
    b = eng.space(g).builder(g.target)
    try: b.push(('c', gid, 0))
    except GB.BuildError as e: return 'no global' not in str(e)
    return True


def sec_B(c):
    bank = leak_bank(); t0 = time.time()
    w, placed = B.build_world(bank_lib.base, bank); lb = B.LoadedBank(w, placed, bank, bank_lib.base); k = w.k; eng = lb.engine
    trace('B the small world: %d globals, %.1f s' % (k.num_globals(), time.time() - t0))
    a, b, cc = placed['s50_leak_a'], placed['s50_leak_b'], placed['s50_leak_c']; h = w.gid['s50_leak_h']; lib = a.library_end
    expect(lib == w.gid[B.LIBRARY_LIMIT] and a.start == lib + 6 and h == a.start and a.gid == h + 1 and b.start == b.gid == a.gid + 1 and cc.start == cc.gid == b.gid + 1, 'B1 the blocks follow the library one after another (%d; %s)' % (lib, [(p.start, p.defs_end, p.gid) for p in (a, b, cc)]))
    # ---- B1. the premises of the blocks
    pa, sa = a.premises('prefix'), a.premises('strict')
    expect(pa.sees(h) and not pa.sees(a.gid) and sa.count() == lib and not sa.sees(h) and pa.count() == lib + 1, 'B1 A: the prefix regime offers the library and its helper, not its own theorem; the strict one the library alone')
    for p in (b, cc):
        for regime in ('prefix', 'strict'):
            ps = p.premises(regime)
            expect(ps.count() == lib and not any(ps.sees(g) for g in (h, a.gid, b.gid, cc.gid)) and all(ps.sees(g) for g in range(lib)), 'B1 %s: the %s regime offers exactly the library -- not the helper or the theorem of A, not its own' % (p.problem.name, regime))
    # ---- B2. WITHOUT the mask the theorem of A finishes B in one action; WITH it the theorem is not on the menu
    node_a, node_h = k.mk_const(a.gid), k.mk_const(h)
    plain = G.Episode(eng, b.stmt, 's50_leak_b', premises=b.gid)
    menu = plain.menu(); idx = [i for i, cd in enumerate(menu.cands) if cd[0] == node_a]
    expect(len(idx) == 1, 'B2 with a plain prefix of the world (premises = the id of the theorem) the theorem of A is on the menu of B')
    if idx:
        port = A.to_portable(eng, menu, Action(Kind.APPLY, idx[0]))
        plain.step(Action(Kind.APPLY, idx[0]))
        expect(plain.done and plain.finish() is not None, 'B2 ... and one action finishes B, kernel-checked: this is the leak')
        masked = lb.episode('s50_leak_b')
        mm = masked.menu()
        expect(not any(cd[0] in (node_a, node_h) for cd in mm.cands), 'B2 with the mask neither the theorem nor the helper of A is on the menu of B')
        pm = {cd[0] for cd in mm.cands}; pp = {cd[0] for cd in menu.cands}
        expect(pm < pp and pp - pm == {node_a, node_h} & pp, 'B2 the menu under the mask is the plain one less the hidden globals (%d < %d)' % (len(pm), len(pp)))
        # ---- B3. the typed term builder
        fresh = G.Episode(eng, b.stmt, 's50_leak_b', premises=b.gid)                       # (the first episode is finished: it has no goal left)
        expect(can_say(eng, fresh, a.gid) is True, 'B3 with a plain prefix the typed term builder can say the theorem of A')
        expect(not can_say(eng, masked, a.gid) and not can_say(eng, masked, h), 'B3 with the mask it refuses to say the theorem and the helper of A (BuildError: no global)')
        expect(can_say(eng, masked, w.gid['Z.sq_nonneg']), 'B3 ... and still says a library lemma')
        # ---- B4. the action itself
        e1 = B.G.Engine(w, kinds=set(G.Kind), strict=True)
        t = A.replay_portable(e1, b.stmt, [port], b.gid)
        expect(t is not None, 'B4 the action `apply A` replays under a plain prefix')
        try: A.replay_portable(e1, b.stmt, [port], b.premises('prefix')); refused = None
        except G.ActionError as e: refused = str(e)
        expect(refused is not None and ('no global %d' % a.gid) in refused, 'B4 under the mask it is refused: ActionError "no global %d" (%s)' % (a.gid, refused))
        # the same through a stale, hand-written action: a portable step that names the helper
        step_h = dict(port); step_h['source'] = ['c', h, 0]
        raises('B4 an action that names the helper of another problem is refused under the mask', lambda: A.replay_portable(e1, b.stmt, [step_h], b.premises('prefix')), REFUSED)
        raises('B4 ... and C, which comes after B, cannot name the theorem of B either', lambda: A.replay_portable(e1, cc.stmt, [dict(port, source=['c', b.gid, 0])], cc.premises('prefix')), REFUSED)
    # ---- B5. the engine agrees with the premises on every global
    n = k.num_globals(); bad = []
    for prem in (None, 549, n + 10, b.premises('prefix'), a.premises('prefix'), PM.Premises(n, [(100, 200), (300, 310)]), PM.Premises(30)):
        eng.premises = prem
        if [g for g in range(-2, n + 12) if eng.sees(g)] != [g for g in range(-2, n + 12) if PM.sees(prem, g, n)]: bad.append(prem)
    expect(not bad, 'B5 the engine sees exactly what the premises offer, for None, an int and a Premises (%s)' % (bad[:2],))
    # ---- B6. the recorder: a library lemma the proof needs is hidden -> a gap, not a silent success
    holder = {}

    def build():
        w_, pl_ = B.build_world(bank_lib.base, bank); holder['placed'] = pl_; return w_
    w2, recs = REC.record_world(build); pl2 = holder['placed']; eng2 = G.Engine(w2, kinds=set(G.Kind), strict=True)
    pb = pl2['s50_leak_b']; r = REC.replay(eng2, recs['s50_leak_b'], premises=pb.premises('prefix'))
    expect(r.ok and len(r.portable) >= 2, 'B6 the proof of B replays under the mask (%s)' % (r.gap,))
    sq = w2.gid['Z.sq_nonneg']
    r2 = REC.replay(eng2, recs['s50_leak_b'], premises=PM.Premises(pb.gid, [(pb.library_end, pb.start), (sq, sq + 1)]))
    expect(not r2.ok and r2.gap is not None and r2.gap.reason == 'premise', 'B6 with the library lemma Z.sq_nonneg hidden as well the proof does not replay: gap "%s"' % (r2.gap.reason if r2.gap else None,))
    r3 = REC.replay(eng2, recs['s50_leak_b'], premises=PM.Premises(pb.gid, [(sq, sq + 1)]))
    expect(not r3.ok, 'B6 ... also when only that lemma is hidden')
    r4 = REC.replay(eng2, recs['s50_leak_b'], premises=PM.Premises(pb.gid, [(pb.library_end, pb.start), (5, 6)]))
    expect(r4.ok, 'B6 hiding a library lemma the proof does not use changes nothing')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the whole world
class Fixture:
    """the bank's world built with every proof recorded (ONE whole-bank world: about 2.5 GB), and the engine over it"""
    def __init__(self):
        t0 = time.time(); holder = {}

        def build():
            w, placed = B.build_world(bank_lib.base, B.BANK); holder['placed'] = placed; return w
        self.w, self.recs = REC.record_world(build); self.placed = holder['placed']; self.k = self.w.k
        self.eng = G.Engine(self.w, kinds=set(G.Kind), strict=True)
        trace('the recorded world: %d globals, %d problems, %d recordings, %.1f s' % (self.k.num_globals(), len(self.placed), len(self.recs), time.time() - t0))
        self.contest = [n for n in bank_all.proved_names() if n not in self.graded()]
        if LIMIT: self.contest = self.contest[:LIMIT]
        self._results = None

    def graded(self):
        names = bank_all.names()
        return set(names[:N_GRADED])

    @property
    def results(self):
        """{name: recorder.Result} of the reference proofs of the proved contest problems, replayed under the mask of the prefix regime"""
        if self._results is None:
            t0 = time.time(); self._results = {}
            for n in self.contest:
                t1 = time.time(); self._results[n] = REC.replay(self.eng, self.recs[n], premises=self.placed[n].premises('prefix'))
                if time.time() - t1 > 20: trace('D slow: %s %.1f s' % (n, time.time() - t1))
            trace('D the %d reference proofs replayed: %.1f s' % (len(self._results), time.time() - t0))
        return self._results


def visible_set(p, regime, lib):
    """the globals the premises of problem `p` should offer, from first principles: the library and the block up to the end of the regime"""
    upto = (p.gid if p.gid is not None else p.defs_end) if regime == 'prefix' else p.defs_end
    return set(range(min(lib, upto))) | set(range(max(p.start, lib), upto))


def sec_C(c):
    fx = c['fx']; w = fx.w; k = fx.k; P = fx.placed; n = k.num_globals(); lib = w.gid[B.LIBRARY_LIMIT]
    expect(lib == LIBRARY_END and all(p.library_end == lib for p in P.values()), 'C1 the library below the problems has %d globals, for every problem (%d)' % (LIBRARY_END, lib))
    expect(len(P) == TOTAL and all(p.start >= lib or p.problem.build is None for p in P.values()), 'C1 every problem with a block of its own starts at or after the end of the library')
    bad = []; stats = collections.Counter()
    for name, p in P.items():
        for regime in ('prefix', 'strict'):
            ps = PM.of(p.premises(regime)); want = visible_set(p, regime, lib)
            got = {g for g in range(n) if PM.sees(ps, g, n)}
            if got != want: bad.append((name, regime, len(got), len(want)))
            stats[regime] += 1
    expect(not bad, 'C2 the premises of every problem, in both regimes, offer exactly the library and its own block up to the end of the regime (%d of %d differ: %s)' % (len(bad), sum(stats.values()), bad[:3]))
    # ---- C3. nothing of another problem is visible; a problem does not see its own theorem
    blocks = {}
    for name, p in P.items():
        end = p.gid + 1 if p.gid is not None else p.defs_end
        if p.start >= lib: blocks[name] = set(range(p.start, end))
    leaks = []
    for name, p in P.items():
        for regime in ('prefix', 'strict'):
            ps = PM.of(p.premises(regime)); seen = {g for g in range(lib, n) if PM.sees(ps, g, n)}
            foreign = [q for q, blk in blocks.items() if q != name and seen & blk]
            if foreign: leaks.append((name, regime, foreign[:2]))
            if p.gid is not None and PM.sees(ps, p.gid, n): leaks.append((name, regime, 'sees its own theorem'))
    expect(not leaks, 'C3 no problem sees a global of another problem, and none sees its own theorem (%s)' % (leaks[:3],))
    thms = {p.gid for p in P.values() if p.gid is not None and p.start >= lib}
    expect(all(not ({g for g in range(lib, n) if PM.sees(PM.of(p.premises(r)), g, n)} & thms) for p in P.values() for r in ('prefix', 'strict')), 'C3 the theorem of a problem of the bank is visible to no problem at all (%d theorems)' % len(thms))
    # ---- C4. the contest problems
    contest = [nm for m in ('bank_amc', 'bank_amc2', 'bank_aime2', 'bank_imo', 'bank_imo2', 'bank_imo3', 'bank_imo4', 'bank_classic') for nm in bank_all.MODULES[m]]
    expect(len(contest) == 196 and all(not P[nm].problem.helpers for nm in contest), 'C4 the 196 problems of the contest modules have no helper lemmas')
    same = [nm for nm in contest if PM.of(P[nm].premises('prefix')) != PM.of(P[nm].premises('strict'))]
    expect(not same, 'C4 for each of them the two regimes offer the same globals (%s)' % (same[:3],))
    own = collections.Counter(PM.of(P[nm].premises('prefix')).count() - lib for nm in contest)
    expect(all(PM.of(P[nm].premises('prefix')).count() == lib + (P[nm].defs_end - P[nm].start) for nm in contest), 'C4 ... the library and the definitions the statement needs: %s' % dict(own))
    expect(all(P[nm].defs_end == P[nm].start for nm in contest if P[nm].proved), 'C4 a proved contest problem declares nothing before its theorem')
    expect(sum(1 for nm in contest if not P[nm].proved and P[nm].defs_end > P[nm].start) >= 1, 'C4 some statement-only problems declare definitions, which their premises offer')
    # ---- C5. the engine agrees with the premises on every global of the world
    bad = []
    for name in list(P)[::9]:
        for regime in ('prefix', 'strict'):
            ps = P[name].premises(regime); fx.eng.premises = ps
            if [g for g in range(n) if fx.eng.sees(g)] != [g for g in range(n) if PM.sees(ps, g, n)]: bad.append((name, regime))
    expect(not bad, 'C5 the engine sees exactly what the premises offer, for a sample of problems (%s)' % (bad[:3],))
    # ---- C6. the menus: under the mask the plain-prefix menu less hidden globals; and the mask removes something
    t0 = time.time(); sample = list(P)[7::17]; removed = 0; bad = []; checked = 0
    for name in sample:
        p = P[name]
        if p.start <= lib: continue
        upto = p.gid if p.gid is not None else p.defs_end
        try:
            ep_m = G.Episode(fx.eng, p.stmt, name, premises=p.premises('prefix')); mm = ep_m.menu()
            ep_p = G.Episode(fx.eng, p.stmt, name, premises=upto); pp = ep_p.menu()
        except Exception as e: bad.append((name, type(e).__name__, str(e)[:60])); continue
        hidden = {k.mk_const(g) for g in range(lib, upto) if not PM.sees(p.premises('prefix'), g, n)}
        cm = {cd[0] for cd in mm.cands}; cp = {cd[0] for cd in pp.cands}
        if not (cm <= cp and cp - cm <= hidden): bad.append((name, 'menu differs by more than the hidden globals', len(cp - cm - hidden)))
        removed += len(cp - cm); checked += 1
    expect(not bad and checked >= 10, 'C6 under the mask the menu of a problem is the plain-prefix menu less hidden globals (%d problems; %s)' % (checked, bad[:2]))
    expect(removed >= 1, 'C6 ... and the mask does remove candidates from the menus (%d in all): it is not vacuous' % removed)
    trace('C6 menus: %d problems, %d candidates removed, %.1f s' % (checked, removed, time.time() - t0))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. the proofs
def sec_D(c):
    fx = c['fx']; w = fx.w; k = fx.k; P = fx.placed; res = fx.results; lib = w.gid[B.LIBRARY_LIMIT]
    expect(bool(LIMIT) or len(fx.contest) == N_PROVED_CONTEST, 'D1 %d proved problems in the contest modules (%d)' % (N_PROVED_CONTEST, len(fx.contest)))
    bad = {n: r.gap for n, r in res.items() if not r.ok}
    expect(not bad, 'D1 the reference proof of every proved contest problem replays as a sequence of kernel-checked actions under the mask of its regime (%d fail: %s)' % (len(bad), list(bad.items())[:2]))
    steps = sum(len(r.actions) for r in res.values() if r.ok)
    kinds = collections.Counter(); [kinds.update(r.kinds) for r in res.values() if r.ok]
    expect(bool(LIMIT) or (steps >= MIN_ACTIONS and len(kinds) >= 12), 'D1 ... %d actions in all, of %d kinds (%s)' % (steps, len(kinds), dict(kinds)))
    trace('D1 %d proofs, %d actions, kinds %s' % (len(res), steps, dict(kinds)))
    # ---- D2. through JSON: the steps are plain data, and they prove the statement again in another engine
    eng3 = G.Engine(w, kinds=set(G.Kind), strict=True); terms = {}; bad = []; t0 = time.time()
    for n, r in res.items():
        if not r.ok: continue
        p = P[n]; steps_j = json.loads(json.dumps(r.portable))
        if json.dumps(steps_j, sort_keys=True) != json.dumps(r.portable, sort_keys=True) or len(steps_j) != len(r.actions): bad.append((n, 'json'))
        n_before = k.num_globals()
        try: t = A.replay_portable(eng3, p.stmt, steps_j, p.premises('prefix'))
        except Exception as e: bad.append((n, type(e).__name__, str(e)[:80])); continue
        terms[n] = (t, n_before)
    expect(not bad, 'D2 the steps of every proof survive a round trip through JSON and prove the statement again in a second engine (%d proofs; %s)' % (len(terms), bad[:2]))
    trace('D2 portable replays: %.1f s' % (time.time() - t0))
    # ---- D3. the proof term mentions only what the premises offer (and what the replay derived itself)
    bad = []; derived = 0
    for n, (t, n_before) in terms.items():
        ps = P[n].premises('prefix'); used = RC.consts_of(k.dump_raw(t.n))
        out = [g for g in used if not PM.sees(ps, g, k.num_globals()) and g < n_before]
        derived += sum(1 for g in used if g >= n_before)
        if out: bad.append((n, [k.global_info(g)[0] for g in sorted(out)[:3]]))
    expect(not bad, 'D3 no proof term mentions a global that its premises hide (%s)' % (bad[:3],))
    # ---- D4. the trusted base
    t0 = time.time(); memo = {}; axioms = set(); quots = set(); per = {}
    for n, (t, _) in terms.items():
        ax, qu = AU.dependencies(k, RC.parse_raw(k.dump_raw(t.n)), memo)
        axioms |= ax; quots |= qu; per[n] = (len(ax), len(qu))
    expect(not axioms, 'D4 no proof of the contest tier depends on an axiom (%s)' % (sorted(axioms),))
    expect(quots <= QUOT, 'D4 the only primitives beyond the kernel\'s declarations that any of them reaches are the quotient primitives (%s)' % (sorted(quots),))
    nq = sum(1 for v in per.values() if v[1])
    trace('D4 audit: %d proofs, %d reach the quotient primitives %s, %.1f s' % (len(per), nq, sorted(quots), time.time() - t0))
    c['terms'] = {n: t for n, (t, _) in terms.items()}
    # ---- D5. a proof is a proof of its statement only
    longs = [n for n in terms if len(res[n].portable) >= 8]
    if len(longs) >= 8:
        wrong = 0; tried = 0
        for i, n in enumerate(longs[:6]):
            m = longs[(i + 5) % len(longs)]
            if m == n: continue
            tried += 1
            try: A.replay_portable(eng3, P[m].stmt, json.loads(json.dumps(res[n].portable)), P[m].premises('prefix'))
            except REFUSED: wrong += 1
        expect(tried >= 5 and wrong == tried, 'D5 the actions of one problem do not finish another problem (%d of %d refused)' % (wrong, tried))
        n = longs[0]; r = res[n]
        raises('D5 a truncated proof is not a proof', lambda: A.replay_portable(eng3, P[n].stmt, json.loads(json.dumps(r.portable))[:-1], P[n].premises('prefix')), REFUSED)
        raises('D5 a proof with a step too many is refused', lambda: A.replay_portable(eng3, P[n].stmt, json.loads(json.dumps(r.portable)) + [r.portable[-1]], P[n].premises('prefix')), REFUSED)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. the second opinion
def sec_E(c):
    fx = c['fx']; w = fx.w; k = fx.k; P = fx.placed
    terms = c.get('terms')
    if terms is None:                                           # section D was not run: replay what is needed
        eng3 = G.Engine(w, kinds=set(G.Kind), strict=True); terms = {}
        for n, r in fx.results.items():
            if r.ok: terms[n] = A.replay_portable(eng3, P[n].stmt, r.portable, P[n].premises('prefix'))
    names = list(terms)
    t0 = time.time()
    so = RC.SecondOpinion(k, fuel=400_000_000, needed_by=[terms[n] for n in names] + [P[n].stmt for n in names])
    trace('E the independent checker\'s environment: %d globals of %d, %.1f s' % (len(so.checker.g), k.num_globals(), time.time() - t0))
    bad = []; t0 = time.time(); slow = []
    for n in names:
        t1 = time.time(); ok, msg = so.accepts(terms[n], P[n].stmt)
        if not ok: bad.append((n, str(msg)[:100]))
        if time.time() - t1 > 30: slow.append((n, round(time.time() - t1)))
    expect(not bad, 'E1 the independent checker accepts the proof of every proved contest problem for its statement (%d proofs; %s)' % (len(names), bad[:2]))
    trace('E1 %d proofs checked in %.1f s; slow: %s' % (len(names), time.time() - t0, slow))
    # ---- E2. negative controls: a proof for the statement of another problem
    wrong = []; tried = 0
    for i, n in enumerate(names[:8]):
        m = names[(i * 5 + 3) % len(names)]
        if m == n or k.dump_raw(P[m].stmt.n) == k.dump_raw(P[n].stmt.n): continue
        tried += 1
        ok, msg = so.accepts(terms[n], P[m].stmt)
        if ok: wrong.append((n, m))
    expect(tried >= 6 and not wrong, 'E2 the independent checker rejects the proof of one problem as a proof of another (%d pairs tried; accepted by mistake: %s)' % (tried, wrong[:2]))
    # a lemma of the library is no proof of a problem
    lem = next((w.gid[nm] for nm in ('Z.sq_nonneg', 'Z.add_comm', 'Z.mul_comm', 'Z.le_refl', 'Z.le_trans') if nm in w.gid and w.gid[nm] in so.gids), None)
    ok, msg = so.accepts(k.mk_const(lem), P[names[0]].stmt) if lem is not None else (True, 'no library lemma in the environment')
    expect(not ok and 'not made for' not in str(msg), 'E2 ... and a lemma of the library is no proof of a problem (%s)' % (str(msg)[:80],))


SECTIONS = [('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E)]


def main():
    t_start = time.time(); c = {}
    for name, fn in SECTIONS:
        if ONLY and name not in ONLY: continue
        if name in 'CDE' and 'fx' not in c: c['fx'] = Fixture()
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:300])); traceback.print_exc(limit=8)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
        if FAIL and os.environ.get('S50_FAILFAST'): break
    print('S50: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:60]: print('  FAIL:', f)
    print('S50 PROOFS GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
