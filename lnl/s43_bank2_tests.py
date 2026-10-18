"""S43 gate: the extended problem bank -- the 54 problems of bank_lib2.py on top of the 28 of S41, and the infrastructure that they needed.

S41 pins the first 28 problems (their statements, their faithfulness checks, their reference proofs); this gate does the same for the problems added after them, in the same world (the world below a problem
never changes when problems are added after it: every digest of s41_golden.json is still the one of the 28), and for what the new problems taught the machinery:

  A. THE MACHINERY.  `Spec(hatoms=..)`: the numbers of the atoms of the HYPOTHESES (a Diophantine hypothesis `x^2 + y^2 = 3 z^2` has the same solutions as `... = -3 z^2` over the integers: only the numbers see
     the slip); the recorder's handling of a claim made under the hypotheses of the outer goal (those are retired, not extras: `sqrt3_irrational`, `x2_y2_3z2`) and of a `linarith` hint that is the goal
     (`schur_nonneg`); `induction_on` with extra arguments of the principle (`Z.induction_from`, `Z.strong_induction`).
  B. THE PROBLEMS.  Every new problem: metadata, a spec, the faithfulness check with its negative controls (no unexplained survivor), the statement sayable in the typed term builder, the statement pinned by digest
     (s43_golden.json).
  C. THE REFERENCE PROOFS.  Every new problem has a reference proof that replays as a sequence of kernel-checked actions; one that proves its lemmas inline replays again with the lemmas hidden (STRICT regime);
     the independent checker takes the proofs; they replay from their JSON in a fresh engine over a rebuilt world.
  D. REGIMES AND EPISODES.  The helper lemmas written for the new problems are on the menu in the prefix regime and hidden in the strict one; the reference actions played one by one finish the episode.
  E. THE DATASET.  The whole bank (bank_all.py): the splits (a pure function of the problem's family; no family straddles two splits; every split covers the levels), the export, the dataset card.
"""
import os, sys, io, time, json, math, copy, collections, tempfile, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(200000)
import itp_env_lnl as E
import bank as B, bank_lib, bank_lib2, bank_all, semantic as SEM, recorder as REC
import gym_env as G, gym_build as GB, gym_acts as A
from gym_env import Kind, Action
from author import AuthorError
import indep_checker as ic
from harness import parse_raw
from nt_lib import _syms
from lnl_base import thm
from semantic import Spec

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S43_TRACE'))
def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond
def raises(what, fn, exc):
    try: fn()
    except exc: return expect(True, what)
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:100]))
    return expect(False, what + ' (it raised nothing)')

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's43_golden.json')
NEW = list(bank_lib2.NAMES)
# the proofs the independent checker needs more than ~15 s for (case enumerations with large certificates, up to minutes); S43_HEAVY=1 checks them too
HEAVY = ('three_squares_mod_eight', 'n5_minus_n_div30', 'pythag_three', 'cube_sum_not_4_mod_9', 'five_dvd_fifth_power', 'schur_nonneg', 'imo1988_p6')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ fixture
class Fixture:
    def __init__(self):
        t0 = time.time()
        self.bk = bank_all.load(); self.w = self.bk.w; self.k = self.w.k
        trace('bank world: %d globals, %d problems, %.1f s' % (self.k.num_globals(), len(self.bk.placed), time.time() - t0))
        t0 = time.time()
        self.w2, self.recs = self.bk.recordings(); self.k2 = self.w2.k
        self.eng2 = G.Engine(self.w2, kinds=set(G.Kind), strict=True)
        self.results = {}; self.strict = {}
        for n in NEW:
            pl = self.bk.placed[n]
            self.results[n] = self.bk.demonstrate(n, engine=self.eng2)
            if pl.problem.strict and pl.premises('strict') != pl.premises('prefix'): self.strict[n] = self.bk.demonstrate(n, engine=self.eng2, regime='strict')
        trace('recordings and replays: %d new proofs (%d strict), %.1f s' % (len(self.results), len(self.strict), time.time() - t0))
        self._ic = None; self._data = None; self._bk2 = None
    @property
    def data(self):
        """the new problems as the JSON file the export writes, read back"""
        if self._data is None:
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, 'bank.json')
                self.bk.export(path, demos=self.results, strict_demos=self.strict)
                self._data = json.load(open(path))
        return self._data
    @property
    def bk2(self):
        """the bank built AGAIN, from scratch (another world, the same library)"""
        if self._bk2 is None: self._bk2 = bank_all.load()
        return self._bk2
    @property
    def indep(self):
        if self._ic is None:
            k = self.k2; env = ic.Env()
            for gid in range(k.num_globals()):
                name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
                env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid), recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
            self._ic = ic.Checker(env, params=True, fuel=400_000_000, prop_elim=True)
        return self._ic
    def ic_check(self, term, stmt):
        k, w = self.k2, self.w2
        return self.indep.check_closed(parse_raw(k.dump_raw(term.n), True), parse_raw(k.dump_raw(w.term(stmt, 'statement').n), True))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. the machinery
def sec_A(c):
    fx = c['fx']; bk = fx.bk; P = bk.placed; sem = bk.semantics
    # ---- A1. the numbers of the atoms of the hypotheses
    pl = P['x2_y2_3z2']; sp = pl.problem.spec; ast = bk.statement_ast('x2_y2_3z2')
    expect(sp.hatoms is not None and len(sp.hatoms) == 1, 'A1 x2_y2_3z2 gives the numbers of the one atom of its hypothesis')
    r = SEM.check_spec(sem, ast, sp)
    expect(r.ok and r.mutants >= 10 and r.caught == r.mutants, 'A1 with them every mutant of the statement x^2 + y^2 = 3 z^2 -> x = y = z = 0 is told from it (%d of %d; %s)' % (r.caught, r.mutants, r.errors[:1]))
    sp0 = copy.copy(sp); sp0.hatoms = None
    r0 = SEM.check_spec(sem, ast, sp0)
    expect(not r0.ok and r0.survivors, 'A1 without them a mutant of the hypothesis (-3 z^2 for 3 z^2: the same solutions over the integers) is not told from the statement (%d survivors)' % len(r0.survivors))
    sp1 = copy.copy(sp); sp1.hatoms = [(sp.hatoms[0][0], lambda x, y, z: 2 * z * z)]
    r1 = SEM.check_spec(sem, ast, sp1, mutate=False)
    expect(not r1.ok and 'atom 0 of the hypotheses' in ' '.join(r1.errors), 'A1 the wrong number (2 z^2 for 3 z^2) in the hypothesis is rejected (%s)' % (r1.errors[:1],))
    sp2 = copy.copy(sp); sp2.hatoms = [sp.hatoms[0], sp.hatoms[0]]
    r2 = SEM.check_spec(sem, ast, sp2, mutate=False)
    expect(not r2.ok and 'relation atoms' in ' '.join(r2.errors), 'A1 a spec with another number of atoms of the hypotheses than the statement has is rejected (%s)' % (r2.errors[:1],))
    sp3 = copy.copy(sp); sp3.hatoms = [(sp.hatoms[0][0],)]
    r3 = SEM.check_spec(sem, ast, sp3, mutate=False)
    expect(not r3.ok and 'arguments' in ' '.join(r3.errors), 'A1 an atom with the wrong number of sides is rejected (%s)' % (r3.errors[:1],))
    # a problem without Diophantine hypotheses is not asked for them
    expect(all(P[n].problem.spec.hatoms is None or len(P[n].problem.spec.hatoms) >= 1 for n in P if P[n].problem.spec is not None), 'A1 ... hatoms are never an empty list')
    # ---- A2. a claim made under the hypotheses of the outer goal: the hypotheses are retired, not extras
    for n in ('sqrt3_irrational', 'x2_y2_3z2'):
        r = fx.results[n]
        expect(r.ok and r.kinds['CLAIM'] >= 1, 'A2 the proof of %s has claims under the hypotheses of the goal and replays (%s; CLAIM x %d)' % (n, r.gap if not r.ok else 'ok', r.kinds['CLAIM']))
        expect(n in fx.strict and fx.strict[n].ok, 'A2 ... also in the strict regime, with its lemma hidden')
    # ---- A3. a linarith hint that is the goal closes it by itself
    r = fx.results['schur_nonneg']
    expect(r.ok and r.kinds['LINARITH'] + r.kinds['NLINARITH'] >= 2, 'A3 schur_nonneg, whose last step is a linarith with the goal itself as a hint, replays (%s)' % (r.gap if not r.ok else 'ok',))
    # ---- A4. induction by a principle with extra arguments
    n_ind = sum(1 for n in NEW if fx.results[n].kinds['INDUCT_WITH'])
    expect(n_ind >= 8, 'A4 %d of the new proofs use induction by a chosen principle (INDUCT_WITH)' % n_ind)
    # a small world of its own: the principle with its starting point as an extra argument, and what the DSL refuses
    w = bank_lib.base(); s = _syms(w); Z, N, Eq, zle, zlt = s.Z, s.N, s.Eq, s.zle, s.zlt
    with w.nota.using(Z):
        lit = w.nota.lit
        def good(p):
            p.intro('k', 'hk')
            p.induction_on('Z.induction_from', 'k', extra=(lit(1),))
            p.nlinarith()
            p.intro('j', 'hj', 'ih'); p.nlinarith()
            p.assumption()
        thm(w, 's43_probe_from', [Z, lambda k: zle(lit(1), k)], lambda k, hk: zle(k, k * k))(good)
        expect('s43_probe_from' in w.gid, 'A4 induction from a starting point (the extra argument 1) proves  1 <= k -> k <= k*k')
        def wrong_type(p): p.intro('k', 'hk'); p.induction_on('N.induction_on', 'k')
        raises('A4 a principle for another type than the variable\'s is refused', lambda: thm(w, 's43_probe_bad1', [Z, lambda k: zle(lit(1), k)], lambda k, hk: zle(k, k * k))(wrong_type), (AuthorError, E.ITPError))
        def too_many(p): p.intro('k', 'hk'); p.induction_on('Z.induction_from', 'k', extra=(lit(1),) * 7)
        try: thm(w, 's43_probe_bad2', [Z, lambda k: zle(lit(1), k)], lambda k, hk: zle(k, k * k))(too_many); expect(False, 'A4 extra arguments that are too many are refused (it raised nothing)')
        except AuthorError as e: expect('fewer arguments' in str(e), 'A4 extra arguments that are too many are refused and the message says so (%s)' % str(e)[:80])
        except Exception as e: expect(False, 'A4 extra arguments that are too many are refused with an AuthorError (%s: %s)' % (type(e).__name__, str(e)[:80]))
        def no_goal(p): p.intro('k', 'hk'); p.induction_on('Z.induction_from', 'k')
        raises('A4 a principle that needs its starting point and gets none is refused', lambda: thm(w, 's43_probe_bad3', [Z, lambda k: zle(lit(1), k)], lambda k, hk: zle(k, k * k))(no_goal), (AuthorError, E.ITPError))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. the problems
def sec_B(c):
    fx = c['fx']; bk = fx.bk; w = fx.w; P = bk.placed
    expect(len(P) == 82 and len(NEW) == 54 and list(P)[28:] == NEW, 'B1 the bank has 82 problems: the 28 of S41 and the 54 of bank_lib2, after them (%d, %d)' % (len(P), len(NEW)))
    expect(len(set(NEW)) == len(NEW) and all(n in P for n in NEW), 'B1 ... each once')
    lv = collections.Counter(P[n].problem.level for n in NEW); kd = collections.Counter(P[n].problem.kind for n in NEW)
    expect(all(lv[i] >= 5 for i in (1, 2, 3)) and lv[4] >= 3 and lv[5] >= 1, 'B1 the new problems cover the levels 1..5 (%s)' % dict(lv))
    expect(kd['prove'] >= 35 and kd['answer'] >= 6, 'B1 ... and the kinds (%s)' % dict(kd))
    topics = collections.Counter(t for n in NEW for t in P[n].problem.topics)
    expect(all(t in B.TOPICS for t in topics) and {'descent', 'functional-equation', 'inequality', 'diophantine', 'modular'} <= set(topics), 'B1 ... and the topics (%d: %s)' % (len(topics), dict(topics)))
    imo = [n for n in P if P[n].problem.source.startswith('IMO')]
    expect(len(imo) == 9 and sum(1 for n in imo if n in NEW) == 3, 'B1 the bank has 9 IMO problems, 3 of them new (%s)' % (imo,))
    expect(all(P[n].problem.text.isascii() and len(P[n].problem.text) > 20 for n in NEW), 'B1 every problem has its text, in ASCII')
    expect(all(P[n].problem.spec is not None for n in NEW), 'B2 every new problem has a spec of its meaning')
    t0 = time.time(); bad = []; stats = collections.Counter()
    for n in NEW:
        rep = bk.check(n)
        if not rep.ok: bad.append((n, rep.errors[:1]))
        if rep.semantics is not None: stats['points'] += rep.semantics.points; stats['mutants'] += rep.semantics.mutants; stats['caught'] += rep.semantics.caught; stats['equivalent'] += rep.semantics.equivalent
        if rep.code is not None: stats['sayable'] += 1
    expect(not bad, 'B2 every new problem passes its check: statement read, meaning agrees with the spec on the box, every mutant caught or explained, statement sayable (%s)' % (bad[:2],))
    expect(stats['sayable'] == len(NEW), 'B2 every statement is sayable in the typed term builder (%d of %d)' % (stats['sayable'], len(NEW)))
    expect(stats['mutants'] >= 1300 and stats['caught'] == stats['mutants'], 'B2 the negative controls: %d mutants of the statements, all of them caught (%d), %d more explained as equivalent' % (stats['mutants'], stats['caught'], stats['equivalent']))
    trace('B2 checks %.1f s, %s' % (time.time() - t0, dict(stats)))
    skips = [(n, d, why) for n in NEW for d, why in P[n].problem.spec.skip_mutants.items()]
    expect(all(len(why) >= 25 for _, _, why in skips) and len(skips) <= 12, 'B3 the mutants explained as equivalent are few (%d) and each has a reason' % len(skips))
    digests = {n: B.statement_digest(w, P[n].stmt) for n in NEW}
    if os.environ.get('S43_UPDATE_GOLDEN'):
        with open(GOLDEN, 'w') as f: json.dump({'statements': digests}, f, indent=0, sort_keys=True)
        print('   golden digests written: %d statements' % len(digests))
    gold = json.load(open(GOLDEN)) if os.path.exists(GOLDEN) else {'statements': {}}
    diff = sorted(n for n in set(gold['statements']) | set(digests) if gold['statements'].get(n) != digests.get(n))
    expect(not diff, 'B4 the statements are those in %s: %d of %d differ (%s) -- if the change is intended: S43_UPDATE_GOLDEN=1 python3 s43_bank2_tests.py' % (os.path.basename(GOLDEN), len(diff), len(digests), diff[:4]))
    old = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 's41_golden.json')))['statements']
    allp = {n: B.statement_digest(w, P[n].stmt) for n in P}
    expect(all(allp[n] == old[n] for n in old if n in allp) and len(set(allp.values())) == len(allp), 'B4 the first 28 statements are still those of s41_golden.json, and no two problems of the whole bank have the same statement')
    # the world below the first new problem is the world of S41: adding problems after them changed nothing below
    first = P[NEW[0]]
    expect(first.start == first.defs_end or first.start <= first.gid, 'B5 the new problems come after the old ones in the world')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. the reference proofs
def sec_C(c):
    fx = c['fx']; bk = fx.bk; P = bk.placed; res = fx.results
    bad = {n: r.gap for n, r in res.items() if not r.ok}
    expect(not bad, 'C1 the reference proof of every new problem replays as a sequence of kernel-checked actions (%s)' % (list(bad.items())[:2],))
    steps = sum(len(r.actions) for r in res.values() if r.ok)
    expect(steps >= 1300, 'C1 ... %d actions in all' % steps)
    kinds = collections.Counter(); [kinds.update(r.kinds) for r in res.values() if r.ok]
    expect(len(kinds) >= 15 and kinds['CLAIM'] >= 20 and kinds['INDUCT_WITH'] >= 8 and kinds['LINCOMB'] >= 5 and kinds['INTERVAL_CASES'] >= 8, 'C1 ... they use many kinds of action (%d), among them claims, induction by a chosen principle, linear combinations, interval cases (%s)' % (len(kinds), dict(kinds)))
    trace('C1 kinds %s' % dict(kinds))
    # ---- C2. the strict regime
    with_helpers = [n for n in NEW if P[n].problem.helpers and P[n].problem.build is not None]
    expect(len(with_helpers) >= 5 and set(with_helpers) <= set(fx.strict), 'C2 every new problem with lemmas of its own that claims a self-contained proof was replayed in the strict regime (%s)' % (with_helpers,))
    sbad = {n: r.gap for n, r in fx.strict.items() if not r.ok}
    expect(not sbad, 'C2 ... and they replay (%s)' % (list(sbad.items())[:2],))
    for n in with_helpers:
        pl = P[n]
        expect(pl.premises('strict') < pl.premises('prefix') and pl.premises('prefix') == pl.gid, 'C2 the strict regime of %s offers fewer globals than the prefix regime (%d < %d)' % (n, pl.premises('strict'), pl.premises('prefix')))
    # every new problem whose build function declares a lemma of its own lists it as a helper (the strict regime hides exactly those)
    undeclared = []
    for n in NEW:
        pl = P[n]
        block = [nm for nm, g in fx.w.gid.items() if pl.start <= g < pl.gid]
        if set(block) != set(pl.problem.helpers): undeclared.append((n, sorted(set(block) ^ set(pl.problem.helpers))[:3]))
    expect(not undeclared, 'C2 the helpers a problem declares are the globals of its block (%s)' % (undeclared[:3],))
    # ---- C3. the independent checker takes the proofs
    t_c3 = time.time()
    eng3 = G.Engine(fx.w2, kinds=set(G.Kind), strict=True); n_ic = 0; bad_ic = []
    other = fx.recs['ineq_cauchy_three'].stmt
    for n in NEW:
        if n in HEAVY and not os.environ.get('S43_HEAVY'): continue
        r = res[n]; rec = fx.recs[n]; t1 = time.time()
        term = A.replay_portable(eng3, rec.stmt, r.portable, rec.gid)
        good, msg = fx.ic_check(term, rec.stmt); n_ic += 1
        if not good: bad_ic.append((n, msg))
        trace('C3 %s %.1f s' % (n, time.time() - t1))
        if n in ('ineq_chebyshev_cubes', 'schur_nonneg'):
            expect(not fx.ic_check(term, other)[0], 'C3 the independent checker rejects the proof of %s as a proof of another problem' % n)
    trace('C3 independent checker: %.1f s for %d proofs' % (time.time() - t_c3, n_ic))
    expect(n_ic >= len(NEW) - len(HEAVY) and not bad_ic, 'C3 the independent checker accepts the proofs by actions of %d new problems (%s)' % (n_ic, bad_ic[:2]))
    # ---- C4. from JSON, in a fresh engine over a freshly built world
    t_c4 = time.time()
    data = fx.data; bk2 = fx.bk2; eng_f = G.Engine(bk2.w, kinds=set(G.Kind), strict=True); bad_j = []; n_j = 0
    recs = {r['name']: r for r in data['problems']}
    for n in NEW:
        rec = recs[n]; pl2 = bk2.placed[n]
        for key, regime in (('demonstration', 'prefix'), ('demonstration_strict', 'strict')):
            if key not in rec: continue
            try:
                A.replay_portable(eng_f, pl2.stmt, rec[key]['steps'], pl2.premises(regime)); n_j += 1
                if rec[key]['length'] != len(rec[key]['steps']): bad_j.append((n, 'length'))
            except Exception as e: bad_j.append((n, key, str(e)[:100]))
    expect(n_j >= len(NEW) + len(fx.strict) and not bad_j, 'C4 the demonstrations read back from the JSON file finish the new problems in a fresh engine over a rebuilt world (%d proofs; %s)' % (n_j, bad_j[:2]))
    trace('C4 JSON replays: %.1f s' % (time.time() - t_c4))
    a, b = 'ineq_chebyshev_cubes', 'ineq_cauchy_three'
    raises('C4 the demonstration of one problem does not prove another', lambda: A.replay_portable(eng_f, bk2.placed[b].stmt, recs[a]['demonstration']['steps'], bk2.placed[b].premises('prefix')), (G.ActionError, AuthorError, E.ITPError))
    st = recs['imo1982_p1']['demonstration']['steps']; pl_ = bk2.placed['imo1982_p1']
    raises('C4 a truncated demonstration is not a proof', lambda: A.replay_portable(eng_f, pl_.stmt, st[:-1], pl_.premises('prefix')), (G.ActionError, AuthorError, E.ITPError))
    raises('C4 a demonstration with a step too many is refused', lambda: A.replay_portable(eng_f, pl_.stmt, st + [st[-1]], pl_.premises('prefix')), (G.ActionError, AuthorError, E.ITPError))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. regimes and episodes
def helper_visible(eng, ep, gid):
    """can the agent SAY the global `gid` in the first goal of the episode (the typed term builder offers only the premises of the regime)?"""
    g = ep.goal(); eng.premises = ep.premises
    b = eng.space(g).builder(g.target)
    try: b.push(('c', gid, 0))
    except GB.BuildError as e: return 'no global' not in str(e)
    return True


def sec_D(c):
    fx = c['fx']; bk = fx.bk; P = bk.placed; w = fx.w; eng = bk.engine
    for n, hs in (('sqrt3_irrational', ('Z.sqrt3_aux',)), ('x2_y2_3z2', ('Z.x2_y2_3z2_aux',)), ('three_squares_mod_eight', ('sq_mod_eight',)), ('schur_nonneg', ('schur_ordered',)), ('imo1988_p6', ('imo1988_jump', 'imo1988_descent'))):
        for h in hs:
            gid = w.gid[h]
            e1 = bk.episode(n, 'prefix'); v1 = helper_visible(eng, e1, gid)
            e2 = bk.episode(n, 'strict'); v2 = helper_visible(eng, e2, gid)
            expect(v1 and not v2, 'D1 the lemma %s written for %s can be said in the prefix regime (%s) and not in the strict one (%s)' % (h, n, v1, v2))
    # ---- D2. the reference actions, played step by step through the Episode API, finish the episode
    done = 0; bad = []
    for n in ('ineq_cauchy_three', 'schur_nonneg', 'fe_f_x_plus_f_y', 'imo1982_p1', 'sqrt3_irrational', 'n5_minus_n_div30', 'imo1964_p2'):
        pl = P[n]
        for regime in ('prefix', 'strict'):
            if regime == 'strict' and n not in fx.strict: continue
            steps_r = fx.results[n].portable if regime == 'prefix' else fx.strict[n].portable
            ep = bk.episode(n, regime)
            try:
                for d in steps_r:
                    if ep.done: raise G.ActionError('over')
                    menu = ep.menu() if d.get('kind') in A.MENU_KINDS else A.NO_MENU
                    ep.step(A.from_portable(eng, menu, d))
                term = ep.finish(); done += 1
            except Exception as e: bad.append((n, regime, type(e).__name__, str(e)[:80]))
    expect(done >= 9 and not bad, 'D2 the reference actions played one by one through the Episode API finish the episode (%d; %s)' % (done, bad[:2]))
    ep = bk.episode('imo1982_p1'); steps = fx.results['imo1982_p1'].portable
    for d in steps[:5]:
        menu = ep.menu() if d.get('kind') in A.MENU_KINDS else A.NO_MENU
        ep.step(A.from_portable(eng, menu, d))
    expect(not ep.done, 'D2 ... and the first steps alone do not')
    # ---- D3. the actions of one problem do not finish another
    ep = bk.episode('ineq_square_products'); fin = False
    try:
        for d in fx.results['ineq_chebyshev_cubes'].portable:
            menu = ep.menu() if d.get('kind') in A.MENU_KINDS else A.NO_MENU
            ep.step(A.from_portable(eng, menu, d))
        ep.finish(); fin = True
    except (G.ActionError, AuthorError, E.ITPError): pass
    expect(not fin, 'D3 the reference actions of ineq_chebyshev_cubes do not finish ineq_square_products')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. the dataset
def sec_E(c):
    import bank_split as BS, dataset as DS, lib_corpus as LC
    fx = c['fx']; bk = fx.bk; P = bk.placed; names = list(P)
    probs = {n: P[n].problem for n in names}
    # ---- E1. the splits
    sp = BS.splits(names)
    flat = [n for ns in sp.values() for n in ns]
    expect(sorted(flat) == sorted(names) and len(flat) == len(set(flat)) == 82, 'E1 the splits partition the 82 problems: every problem is in exactly one split (%s)' % {s: len(ns) for s, ns in sp.items()})
    assign = {n: s for s, ns in sp.items() for n in ns}
    expect(BS.straddling(assign) == [], 'E1 no family straddles two splits (no train problem is a variant of a test problem)')
    multi = collections.defaultdict(list)
    for n in names: multi[BS.family(n)].append(n)
    expect(sum(1 for f, m in multi.items() if len(m) > 1) >= 10 and all(n in P for n in BS.FAMILIES), 'E1 the families are real: %d families of more than one problem, all named problems exist' % sum(1 for f, m in multi.items() if len(m) > 1))
    bad = dict(assign); bad['imo1964_p1a'] = 'train'; bad['imo1964_p1b'] = 'test'
    expect(BS.straddling(bad) == ['imo1964_p1'], 'E1 ... and a split that separates the two parts of IMO 1964 P1 is noticed')
    expect(BS.coverage(probs) == [], 'E1 every split covers the levels and topics it should (%s)' % (BS.coverage(probs),))
    expect(all(BS.split_of(n) == assign[n] for n in reversed(names)) and BS.splits(names[:40])['test'] == [n for n in sp['test'] if n in names[:40]], 'E1 the split of a problem is a function of its family alone: the first 40 problems alone are split as in the whole bank')
    other = BS.splits(names, salt_number=BS.SALT_NUMBER + 1)
    expect(other != sp, 'E1 another salt would split the bank differently (so the frozen salt is a choice, recorded)')
    lvl_test = collections.Counter(probs[n].level for n in sp['test'])
    expect(sum(1 for n in sp['test'] if probs[n].source.startswith('IMO')) >= 2, 'E1 the test split has olympiad problems (levels %s)' % dict(lvl_test))
    # the lemma split: a function of the name alone
    ln = ['Z.add_comm', 'Z.sq_nonneg', 'N.add_comm', 'le_trans', 'Z.mul_pos']
    expect([BS.lemma_split(n) for n in ln] == [BS.lemma_split(n) for n in ln] and set(BS.lemma_split(n) for n in ['L%d' % i for i in range(300)]) == {'train', 'dev', 'test'}, 'E1 the lemma split uses all three splits and is deterministic')
    cnt = collections.Counter(BS.lemma_split('lemma%d' % i) for i in range(2000))
    expect(0.62 < cnt['train'] / 2000 < 0.78 and 0.05 < cnt['dev'] / 2000 < 0.15 and 0.15 < cnt['test'] / 2000 < 0.25, 'E1 ... in the proportions 70/10/20 (%s)' % dict(cnt))
    # ---- E2. the card and the splits document
    spd = DS.splits_doc(names, ln)
    expect(spd['dataset_version'] == BS.DATASET_VERSION and spd['problems'] == {s: list(ns) for s, ns in sp.items()} and set(spd['lemmas']) == {'train', 'dev', 'test'} and sorted(sum(spd['lemmas'].values(), [])) == sorted(ln), 'E2 the splits document lists the problems and the lemmas of each split')
    data = fx.data
    for rec in data['problems']: rec['split'] = BS.split_of(rec['name']); rec['family'] = BS.family(rec['name'])
    card1 = DS.card(data, spd); card2 = DS.card(data, spd)
    expect(card1 == card2 and '82 problems' in card1 and data['world_digest'] in card1 and all(('**%s**' % s) in card1 for s in ('train', 'dev', 'test')), 'E2 the dataset card is deterministic and states the counts, the splits and the world digest')
    expect(all(('`%s`' % n) in card1 for n in names), 'E2 ... and names every problem under its split')
    # ---- E3. the library corpus: lemmas of the base library as action sequences, replayable in a fresh engine
    t0 = time.time()
    some = {'Z.add_comm', 'Z.sq_nonneg', 'N.induction_on', 'Z.le_trans', 'Z.dvd_trans', 'Z.mul_pos', 'Z.lt_trichotomy'}
    digs = {B.statement_digest(fx.w, P[n].stmt) for n in names}
    corp = LC.build(names=some, bank_digests=digs)
    got = {l['name'] for l in corp['lemmas']}
    expect(len(got) >= 5 and not corp['unreplayed'], 'E3 lemmas of the library replay as action sequences (%d of %d; unreplayed: %s)' % (len(got), len(some), [u['name'] for u in corp['unreplayed']]))
    expect(all(l['split'] == BS.lemma_split(l['name']) and not l['bank_duplicate'] and l['length'] == len(l['steps']) >= 1 for l in corp['lemmas']), 'E3 ... each with its split, none a copy of a problem of the bank')
    bk2 = fx.bk2; eng_f = G.Engine(bk2.w, kinds=set(G.Kind), strict=True); bad_l = []
    for l in corp['lemmas']:
        gid = bk2.w.gid[l['name']]
        try: A.replay_portable(eng_f, bk2.w.T(bk2.k.global_info(gid)[1]), l['steps'], gid)
        except Exception as e: bad_l.append((l['name'], str(e)[:80]))
    expect(not bad_l, 'E3 ... and the steps read from the corpus finish the lemma in a fresh engine over a rebuilt world (%s)' % (bad_l[:2],))
    expect(corp['world_digest'] == B.world_digest(bk2.w, corp['globals']) and LC.digest(corp) == LC.digest(json.loads(json.dumps(corp))), 'E3 the corpus names the world it was built in (the first %d globals of the bank\'s world) and its digest survives JSON' % corp['globals'])
    trace('E3 library corpus %.1f s' % (time.time() - t0))


SECTIONS = [('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E)]


def main():
    t_start = time.time()
    c = {'fx': Fixture()}
    only = os.environ.get('S43_ONLY')
    for name, fn in SECTIONS:
        if only and name not in only.split(','): continue
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:300])); traceback.print_exc(limit=8)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
        if FAIL and os.environ.get('S43_FAILFAST'): break
    print('S43: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:60]: print('  FAIL:', f)
    print('S43 BANK GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
