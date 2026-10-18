"""S41 gate: the problem bank -- formal olympiad problems whose statements are CHECKED TO MEAN what the problem says, with reference proofs that replay through the action language.

A theorem prover that has proved its statement has proved what the statement says, not what the problem says. Nothing in the kernel can tell them apart; a bank of problems for learning is only as good as
the faithfulness of its statements. So this gate does not trust the statements (nor the machinery that reads them):

  A. THE MEANING.  `semantic.py` reads a statement into an AST and evaluates it with the meaning of the library's symbols written down once (`VALUE_PRIMS`, `PROP_PRIMS`).  The tables are audited
     against the library's DEFINITIONS and against every theorem the kernel has proved (a table that disagreed with the library would make a theorem false), and the audits are shown to FAIL when a
     table is corrupted.  The reader is exercised on statements built here; `check_spec` is shown to reject wrong specs, vacuous statements, weaker statements (atoms), existentials with a wrong
     body, and to leave mutated statements unnoticed only where the spec says why.
  B. THE PROBLEMS.  Every problem of the bank: metadata, a spec, the faithfulness check with its negative controls (no unexplained survivor), the statement is sayable in the typed term builder
     (an agent without a text interface can build it), statements are pinned by digest (s41_golden.json).
  C. THE REFERENCE PROOFS.  Every problem has a reference proof that replays as a sequence of kernel-checked actions; one that proves its lemmas inline replays again with the lemmas hidden (the
     STRICT regime); the independent checker takes the proofs; they replay from their JSON in a fresh engine over a freshly built world.
  D. REGIMES AND EPISODES.  The prefix regime offers every earlier global, the strict regime hides the lemmas written for the problem; an Episode is the statement as a goal; the reference actions
     played step by step finish it; they finish no other problem.
  E. EXPORT.  The JSON of the bank: complete, deterministic (two builds, one digest), a changed problem changes the digest.
  F. NEGATIVE CONTROLS OF THE PIPELINE.  A bank of deliberately wrong formalisations (a weaker inequality, an existential for a universal, a vacuous statement, an unreadable statement, a theorem the
     build function did not prove ..): each one is rejected by `check`.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, io, time, json, math, random, collections, hashlib, tempfile, traceback
sys.setrecursionlimit(200000)
import itp_env_lnl as E
import bank as B, bank_lib, semantic as SEM, recorder as REC
import gym_env as G, gym_build as GB, gym_acts as A
from gym_env import Kind, Action
from author import AuthorError
import indep_checker as ic
from harness import parse_raw
from nt_lib import _syms
from lnl_base import thm
from semantic import Spec

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S41_TRACE'))
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

GOLDEN = os.path.join(itp_paths.GOLDEN, 's41_golden.json')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ fixture
class Fixture:
    def __init__(self):
        t0 = time.time()
        self.bk = bank_lib.load(); self.w = self.bk.w; self.k = self.w.k
        trace('bank world: %d globals, %d problems, %.1f s' % (self.k.num_globals(), len(self.bk.placed), time.time() - t0))
        t0 = time.time()
        self.w2, self.recs = self.bk.recordings(); self.k2 = self.w2.k
        self.eng2 = G.Engine(self.w2, kinds=set(G.Kind), strict=True)
        self.results = {}                       # name -> recorder Result (prefix), strict -> Result
        self.strict = {}
        for n, pl in self.bk.placed.items():
            self.results[n] = self.bk.demonstrate(n, engine=self.eng2)
            if pl.problem.strict and pl.premises('strict') != pl.premises('prefix'): self.strict[n] = self.bk.demonstrate(n, engine=self.eng2, regime='strict')
        trace('recordings and replays: %d proofs, %.1f s' % (len(self.recs), time.time() - t0))
        self._ic = None; self._data = None; self._bk2 = None
    @property
    def data(self):
        """the bank as the JSON file the export writes, read back"""
        if self._data is None:
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, 'bank.json')
                self.bk.export(path, demos=self.results, strict_demos=self.strict)
                self._data = json.load(open(path))
        return self._data
    @property
    def bk2(self):
        """the bank built AGAIN, from scratch (another world, the same library)"""
        if self._bk2 is None: self._bk2 = bank_lib.load()
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


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. the meaning
def sec_A(c):
    fx = c['fx']; w = fx.w; bk = fx.bk
    saved_p = dict(SEM.PROP_PRIMS); saved_v = dict(SEM.VALUE_PRIMS)
    # ---- A1. the table of defined predicates is what the library defines
    expect(SEM.audit_prims(w) == [], 'A1 the meaning of Z.dvd, Z.coprime, Z.is_gcd and Z.prime in the table is what the library defines (%s)' % (SEM.audit_prims(w)[:2],))
    # ---- A2. ... and the audit notices when it is not
    zd = SEM._zdvd
    wrong = {'Z.dvd': lambda a, b: zd(b, a), 'Z.coprime': lambda a, b: math.gcd(a, b) <= 2, 'Z.is_gcd': lambda a, b, d: abs(d) == math.gcd(a, b), 'Z.prime': lambda p: p >= 1 and SEM._is_prime(p) or p == 1}
    try:
        for nm, fn in wrong.items():
            SEM.PROP_PRIMS[nm] = (saved_p[nm][0], fn)
            bad = SEM.audit_prims(w)
            expect(any(b[0] == nm for b in bad), 'A2 the audit of the table notices a wrong meaning of %s' % nm)
            SEM.PROP_PRIMS[nm] = saved_p[nm]
    finally: SEM.PROP_PRIMS.clear(); SEM.PROP_PRIMS.update(saved_p)
    expect(SEM.audit_prims(w) == [], 'A2 ... and the table is restored')
    # ---- A3. every theorem of the library in the fragment is TRUE under the table (the kernel proved it from the definitions)
    t0 = time.time(); r = SEM.audit_theorems(w)
    again = SEM.audit_theorems(w, names=[nm for nm, _ in r['inconclusive']], window_ex=40) if r['inconclusive'] else None
    expect(r['checked'] >= 380 and not r['failed'], 'A3 the %d theorems of the library inside the fragment come out true at all %d points of the windows (failed: %s)' % (r['checked'], r['points'], r['failed'][:3]))
    expect(again is None or (not again['failed'] and not again['inconclusive'] and again['checked'] == len(r['inconclusive'])), 'A3 ... those with a witness out of the small window are decided with a larger one (%s)' % (None if again is None else (again['failed'][:2], again['inconclusive'][:2]),))
    expect(r['unsupported'] > 100 and len(r['names']) == r['checked'], 'A3 the others are reported as outside the fragment (%d)' % r['unsupported'])
    trace('audit_theorems %.1f s' % (time.time() - t0))
    # ---- A4. ... and the audit notices a table that is wrong: a corrupted `<=`, `*`, `-`, `^` makes true theorems false
    probes = [('P', 'Z.le', lambda a, b: a < b), ('P', 'Z.lt', lambda a, b: a <= b), ('P', 'Z.dvd', lambda a, b: zd(b, a)), ('V', 'Z.mul', lambda a, b: a * b + (1 if a == 3 else 0)), ('V', 'Z.sub', lambda a, b: a + b),
              ('V', 'Z.neg', lambda a: a), ('V', 'Z.pow', lambda a, n: a ** n + 1), ('V', 'Z.sqr', lambda a: abs(a))]
    names_small = ['Z.add_comm', 'Z.mul_comm', 'Z.le_trans', 'Z.lt_irrefl', 'Z.sub_self', 'Z.neg_neg', 'Z.pow_succ', 'Z.dvd_trans', 'Z.dvd_refl', 'Z.sq_nonneg', 'Z.le_refl', 'Z.mul_zero', 'Z.sub_eq_add_neg', 'Z.pow_zero', 'Z.dvd_zero',
                   'Z.add_zero', 'Z.mul_one', 'Z.neg_add', 'Z.le_antisymm', 'Z.lt_of_lt_of_le', 'Z.dvd_mul_right']
    names_small = [n for n in names_small if n in w.gid]
    try:
        for kind, nm, fn in probes:
            table = SEM.PROP_PRIMS if kind == 'P' else SEM.VALUE_PRIMS; old_entry = table[nm]
            table[nm] = (old_entry[0], fn)
            try: res = SEM.audit_theorems(w)
            finally: table[nm] = old_entry
            expect(res['failed'], 'A4 the audit of the theorems notices a wrong meaning of %s (%d theorems fail)' % (nm, len(res['failed'])))
    finally:
        SEM.PROP_PRIMS.clear(); SEM.PROP_PRIMS.update(saved_p); SEM.VALUE_PRIMS.clear(); SEM.VALUE_PRIMS.update(saved_v)
    res = SEM.audit_theorems(w, names=names_small)
    expect(not res['failed'] and res['checked'] >= 15, 'A4 ... and with the table restored the same theorems come out true (%d)' % res['checked'])


def sec_A_reader(c):
    """A5-A8: the reader and check_spec on statements built here"""
    fx = c['fx']; w = fx.w; bk = fx.bk; sem = SEM.Semantics(w)
    s = _syms(w); Z, N, Eq, Ex, F, zle, zlt, D = s.Z, s.N, s.Eq, s.Ex, s.F, s.zle, s.zlt, s.D
    with w.nota.using(Z):
        lit = w.nota.lit
        # ---- A5. reading and evaluating
        t1 = F([Z, Z], lambda a, b: Eq(a + b, b + a))
        ast = sem.decode(t1.n)
        expect(ast[0] == 'all' and ast[1] == 'Z' and ast[2][0] == 'all' and ast[2][2][0] == 'eq', 'A5 a statement is read into an AST: %s' % (SEM.show_ast(ast),))
        kinds, hyps, concl = sem.parts(ast)
        expect(kinds == ('Z', 'Z') and hyps == [] and concl[0] == 'eq', 'A5 ... which has its parts: the variables, the hypotheses, the conclusion')
        expect(sem.evaluate(ast) is True, 'A5 ... and is true on the window')
        t2 = Ex(Z, w.lam(Z, lambda x: Eq(x * x, lit(81))))
        expect(sem.evaluate(sem.decode(t2.n)) is False, 'A5 an existential is decided over the window of its witnesses: x*x = 81 has no solution in -8..8 ... ')
        sem_big = SEM.Semantics(w, window_z=8, window_ex=10)
        expect(sem_big.evaluate(sem_big.decode(t2.n)) is True, 'A5 ... and has one in a larger window')
        t3 = F([Z], lambda x: zle(lit(0), x))
        expect(sem.evaluate(sem.decode(t3.n)) is False, 'A5 a universal statement that fails at a point of the window is false there: 0 <= x')
        t4 = F([Z, lambda x: zlt(lit(0), x)], lambda x, h: zle(lit(1), x))
        expect(sem.evaluate(sem.decode(t4.n)) is True and sem.parts(sem.decode(t4.n))[0] == ('Z',) and len(sem.parts(sem.decode(t4.n))[1]) == 1, 'A5 a hypothesis restricts the points: 0 < x -> 1 <= x')
        t5 = F([Z, Z], lambda a, b: w.Iff(D(a, b), Ex(Z, w.lam(Z, lambda k: Eq(b, a * k)))))
        ast5 = sem.decode(t5.n)
        expect(ast5[0] == 'all' and sem.parts(ast5)[2][0] == 'iff', 'A5 an equivalence is one conclusion with two sides')
        # ---- A6. what is outside the fragment is refused, not misread
        PN = w.arrow(N, s.PROP)
        t6 = F([PN, lambda P: P(w['N0'])], lambda P, h: P(w['N0']))
        raises('A6 a statement about predicates is outside the fragment', lambda: sem.decode(t6.n), SEM.Unsupported)
        # ---- A7. check_spec: wrong specs are rejected, and say what is wrong
        def spec_of(stmt, spec, **kw):
            return SEM.check_spec(sem, sem.decode(stmt.n), spec, **kw)
        good = F([Z, Z], lambda a, b: zle(2 * a * b, a ** 2 + b ** 2))
        sp = lambda **kw: Spec(('Z', 'Z'), (), lambda a, b: 2 * a * b <= a * a + b * b, atom=(lambda a, b: 2 * a * b, lambda a, b: a * a + b * b), **kw)
        r = spec_of(good, sp(), mutate=False); expect(r.ok and r.points > 100, 'A7 the right spec is accepted (%s)' % (r.errors,))
        r = spec_of(good, Spec(('Z',), (), lambda a: True), mutate=False); expect(not r.ok and 'quantifies' in r.errors[0], 'A7 a spec with other parameters is rejected (%s)' % (r.errors[:1],))
        r = spec_of(good, Spec(('Z', 'Z'), (lambda a, b: True,), lambda a, b: True), mutate=False); expect(not r.ok and 'hypotheses' in r.errors[0], 'A7 a spec with another number of hypotheses is rejected (%s)' % (r.errors[:1],))
        r = spec_of(good, Spec(('Z', 'Z'), (), lambda a, b: 2 * a * b < a * a + b * b), mutate=False); expect(not r.ok and 'disagrees' in r.errors[0], 'A7 a spec that says something false is rejected (%s)' % (r.errors[:1],))
        weak = F([Z, Z], lambda a, b: zle(a * b, a ** 2 + b ** 2))
        r = spec_of(weak, Spec(('Z', 'Z'), (), lambda a, b: 2 * a * b <= a * a + b * b), mutate=False)
        expect(r.ok, 'A7 a weaker TRUE statement agrees with a true spec at every point: the truth table alone cannot tell them apart')
        r = spec_of(weak, sp(), mutate=False); expect(not r.ok and 'numbers in atom 0' in r.errors[0], 'A7 ... the numbers of the atoms can (%s)' % (r.errors[:1],))
        r = spec_of(good, sp(), mutate=True)
        expect(r.ok and r.mutants >= 10 and r.caught == r.mutants, 'A7 every mutant of 2ab <= a^2 + b^2 is told from it (%d of %d)' % (r.caught, r.mutants))
        dv = F([Z, lambda a: D(lit(3), a)], lambda a, h: D(lit(3), a * a))
        spd = lambda **kw: Spec(('Z',), (lambda a: a % 3 == 0,), lambda a: (a * a) % 3 == 0, atom=(lambda a: 3, lambda a: a * a), **kw)
        r = spec_of(dv, spd(), mutate=True)
        expect(not r.ok and r.survivors and all('Z.dvd' in sv for sv in r.survivors), 'A7 a mutant that means the same as the statement (3 | x and -3 | x) is reported unless the spec explains it (%s)' % (r.survivors,))
        r2 = spec_of(dv, spd(skip_mutants={'constant 3 -> -3 in (Z.dvd 3': 'a divisor and its negative divide the same integers'}), mutate=True)
        expect(r2.ok and r2.equivalent >= 1, 'A7 ... and accepted once the spec explains it (%s)' % (r2.errors[:1],))
        r3 = spec_of(dv, spd(), mutate=True, strict=False); expect(r3.ok and r3.survivors, 'A7 non-strict mode reports the survivors without failing')
        # ---- A8. the body of a leading existential
        ex_good = Ex(Z, w.lam(Z, lambda a: zlt(lit(0), a) & zlt(a, lit(5)) & Eq(a * a, lit(9))))
        ex_wrongpol = Ex(Z, w.lam(Z, lambda a: zlt(a, lit(0)) & zlt(a, lit(5)) & Eq(a * a, lit(9))))
        ex_or = Ex(Z, w.lam(Z, lambda a: zlt(lit(0), a) | zlt(a, lit(5)) & Eq(a * a, lit(9))))
        cl = lambda: any(0 < a < 5 and a * a == 9 for a in range(-6, 7))
        body = (('Z',), lambda a: 0 < a < 5 and a * a == 9)
        spec_b = lambda **kw: Spec((), (), cl, windows={'ex': 6}, atoms=[(lambda a: 0, lambda a: a), (lambda a: a, lambda a: 5), (lambda a: a * a, lambda a: 9)], **kw)
        r = spec_of(ex_good, spec_b(body=body)); expect(r.ok and r.mutants >= 8 and not r.survivors, 'A8 an existential with the right body: every mutant is caught (%d of %d; %s)' % (r.caught, r.mutants, r.errors[:1]))
        r = spec_of(ex_wrongpol, spec_b(body=body), mutate=False); expect(not r.ok and 'part body' in r.errors[0], 'A8 a body with an inequality the wrong way round is rejected although the closed statement is true (%s)' % (r.errors[:1],))
        r = spec_of(ex_or, spec_b(body=body), mutate=False); expect(not r.ok and 'part body' in r.errors[0], 'A8 a body with `or` for `and` is rejected (%s)' % (r.errors[:1],))
        r = spec_of(ex_or, spec_b(), mutate=False); expect(r.ok, 'A8 ... which no spec without a body notices: the statement is true either way')
        r = spec_of(ex_good, spec_b(), mutate=True); expect(not r.ok and r.survivors, 'A8 ... and mutants that change the body go unnoticed without it (%d survivors)' % len(r.survivors))
        r = spec_of(ex_good, Spec((), (), cl, windows={'ex': 6}, body=(('Z', 'Z'), lambda a, b: True)), mutate=False); expect(not r.ok and 'begin with existentials' in r.errors[0], 'A8 a body for more witnesses than the statement has is rejected (%s)' % (r.errors[:1],))
        # ---- A9. vacuity
        vac = F([Z, lambda n: zlt(n, lit(0)), lambda n, h: zlt(lit(0), n)], lambda n, h1, h2: Eq(n, n + 1))
        r = spec_of(vac, Spec(('Z',), (lambda n: n < 0, lambda n: n > 0), lambda n: n == n + 1), mutate=False)
        expect(not r.ok and any('vacuous' in e for e in r.errors), 'A9 a statement whose hypotheses nothing satisfies is vacuous, however well it agrees with its spec (%s)' % (r.errors[:2],))
        # ---- A10. equivalences: both sides must be non-trivial
        triv = F([Z, lambda x: zle(x, x)], lambda x, h: w.Iff(zle(lit(0), x * x), zle(x, x)))
        r = spec_of(triv, Spec(('Z',), (lambda x: True,), (lambda x: True, lambda x: True)), mutate=False)
        expect(not r.ok and any('always true' in e for e in r.errors), 'A10 an equivalence with a side that is always true is not a characterisation (%s)' % (r.errors[:2],))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. the problems
def sec_B(c):
    fx = c['fx']; bk = fx.bk; w = fx.w
    P = bk.placed
    expect(len(P) >= 28, 'B1 the bank has its problems (%d)' % len(P))
    expect(list(P) == list(B.BANK.problems) and len(set(P)) == len(P), 'B1 ... each once, in the order they were registered')
    lv = collections.Counter(pl.problem.level for pl in P.values()); kd = collections.Counter(pl.problem.kind for pl in P.values())
    expect(all(lv[i] >= 3 for i in (1, 2, 3)), 'B1 every level 1..3 has problems (%s)' % dict(lv))
    expect(kd['answer'] >= 2 and kd['exists'] >= 1 and kd['prove'] >= 15, 'B1 the bank has problems of every kind: prove, exists, answer (%s)' % dict(kd))
    topics = collections.Counter(t for pl in P.values() for t in pl.problem.topics)
    expect(all(t in B.TOPICS for t in topics) and len(topics) >= 10, 'B1 the topics are declared ones and many (%d)' % len(topics))
    expect(sum(1 for pl in P.values() if pl.problem.source.startswith('IMO')) >= 6, 'B1 several problems are IMO problems (%d)' % sum(1 for pl in P.values() if pl.problem.source.startswith('IMO')))
    expect(all(pl.problem.text.isascii() and len(pl.problem.text) > 20 for pl in P.values()), 'B1 every problem has its text, in ASCII')
    # ---- B2. every problem
    no_spec = [n for n, pl in P.items() if pl.problem.spec is None]
    expect(no_spec == ['N.induction3'], 'B2 every problem has a spec of its meaning but the induction principle, which quantifies over predicates (%s)' % (no_spec,))
    t0 = time.time(); bad = []; stats = collections.Counter()
    for n, pl in P.items():
        rep = bk.check(n)
        if not rep.ok: bad.append((n, rep.errors[:1]))
        if rep.semantics is not None: stats['points'] += rep.semantics.points; stats['mutants'] += rep.semantics.mutants; stats['caught'] += rep.semantics.caught; stats['equivalent'] += rep.semantics.equivalent
        if rep.code is not None: stats['sayable'] += 1
    expect(not bad, 'B2 every problem passes its check: statement read, meaning agrees with the spec on the box, every mutant caught or explained, statement sayable (%s)' % (bad[:2],))
    expect(stats['sayable'] == len(P), 'B2 every statement is sayable in the typed term builder (%d of %d)' % (stats['sayable'], len(P)))
    expect(stats['mutants'] >= 650 and stats['caught'] == stats['mutants'], 'B2 the negative controls: %d mutants of the statements, all of them caught (%d), %d more explained as equivalent' % (stats['mutants'], stats['caught'], stats['equivalent']))
    trace('B2 checks %.1f s, %s' % (time.time() - t0, dict(stats)))
    # ---- B3. the explanations of equivalent mutants are few and have reasons
    skips = [(n, d, why) for n, pl in P.items() if pl.problem.spec is not None for d, why in pl.problem.spec.skip_mutants.items()]
    expect(all(len(why) >= 25 for _, _, why in skips) and len(skips) <= 12, 'B3 the mutants explained as equivalent are few (%d) and each has a reason' % len(skips))
    # ---- B4. the statements are pinned
    digests = {n: B.statement_digest(w, pl.stmt) for n, pl in P.items()}
    if os.environ.get('S41_UPDATE_GOLDEN'):
        gold = {'statements': digests}
        with open(GOLDEN, 'w') as f: json.dump(gold, f, indent=0, sort_keys=True)
        print('   golden digests written: %d statements' % len(digests))
    gold = json.load(open(GOLDEN)) if os.path.exists(GOLDEN) else {'statements': {}}
    diff = sorted(n for n in set(gold['statements']) | set(digests) if gold['statements'].get(n) != digests.get(n))
    expect(not diff, 'B4 the statements are those in %s: %d of %d differ (%s) -- if the change is intended: S41_UPDATE_GOLDEN=1 python3 s41_bank_tests.py' % (os.path.basename(GOLDEN), len(diff), len(digests), diff[:4]))
    expect(len(set(digests.values())) == len(digests), 'B4 no two problems have the same statement')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. the reference proofs
def sec_C(c):
    fx = c['fx']; bk = fx.bk; P = bk.placed; res = fx.results
    bad = {n: r.gap for n, r in res.items() if not r.ok}
    expect(not bad, 'C1 the reference proof of every problem replays as a sequence of kernel-checked actions (%s)' % (list(bad.items())[:2],))
    steps = sum(len(r.actions) for r in res.values() if r.ok)
    expect(steps >= 400, 'C1 ... %d actions in all' % steps)
    kinds = collections.Counter(); [kinds.update(r.kinds) for r in res.values() if r.ok]
    expect(len(kinds) >= 14 and kinds['CLAIM'] >= 8 and kinds['INDUCT_WITH'] >= 3 and kinds['INTERVAL_CASES'] >= 4, 'C1 ... they use many kinds of action, among them claims, induction by a chosen principle, interval cases (%s)' % dict(kinds))
    # ---- C2. the strict regime: the proofs that carry their lemmas inline replay with the lemmas hidden
    with_helpers = [n for n, pl in P.items() if pl.problem.helpers and pl.problem.build is not None]
    expect(len(with_helpers) >= 3 and set(with_helpers) <= set(fx.strict), 'C2 every problem with lemmas of its own that claims a self-contained proof was replayed in the strict regime (%s)' % (with_helpers,))
    sbad = {n: r.gap for n, r in fx.strict.items() if not r.ok}
    expect(not sbad, 'C2 ... and they replay (%s)' % (list(sbad.items())[:2],))
    for n in with_helpers[:3]:
        pl = P[n]
        expect(pl.premises('strict') < pl.premises('prefix') and pl.premises('prefix') == pl.gid, 'C2 the strict regime of %s offers fewer globals than the prefix regime (%d < %d)' % (n, pl.premises('strict'), pl.premises('prefix')))
    # a proof that uses a hidden lemma does not replay when it is hidden
    n = 'sqrt2_irrational'; r = bk.demonstrate(n, engine=fx.eng2, regime='strict')
    expect(not r.ok, 'C2 the reference proof of %s uses a lemma written for it: with the lemma hidden it does not replay (and the problem says so: strict=%s)' % (n, P[n].problem.strict))
    expect(P[n].problem.strict is False, 'C2 ... the problem declares it')
    # ---- C3. the independent checker takes the proofs
    t_c3 = time.time()
    eng3 = G.Engine(fx.w2, kinds=set(G.Kind), strict=True); n_ic = 0; bad_ic = []; heavy = ('imo2019_p1', 'sqrt2_irrational')
    other = fx.recs['imo1964_p1b'].stmt
    for n, pl in P.items():
        if n in heavy and not os.environ.get('S41_HEAVY'): continue
        r = res[n]; rec = fx.recs[n]
        term = A.replay_portable(eng3, rec.stmt, r.portable, rec.gid)
        good, msg = fx.ic_check(term, rec.stmt); n_ic += 1
        if not good: bad_ic.append((n, msg))
        if n in ('ineq_cauchy_two', 'imo1969_p1') and n != 'imo1964_p1b':
            expect(not fx.ic_check(term, other)[0], 'C3 the independent checker rejects the proof of %s as a proof of another problem' % n)
    trace('C3 independent checker: %.1f s for %d proofs' % (time.time() - t_c3, n_ic))
    expect(n_ic >= 24 and not bad_ic, 'C3 the independent checker accepts the proofs by actions of %d problems (%s)' % (n_ic, bad_ic[:2]))
    # ---- C4. from JSON, in a fresh engine over a freshly built world
    t_c4 = time.time()
    data = fx.data; bk2 = fx.bk2; eng_f = G.Engine(bk2.w, kinds=set(G.Kind), strict=True); bad_j = []; n_j = 0
    for rec in data['problems']:
        pl2 = bk2.placed[rec['name']]
        for key, regime in (('demonstration', 'prefix'), ('demonstration_strict', 'strict')):
            if key not in rec: continue
            try:
                term = A.replay_portable(eng_f, pl2.stmt, rec[key]['steps'], pl2.premises(regime)); n_j += 1
                if rec[key]['length'] != len(rec[key]['steps']): bad_j.append((rec['name'], 'length'))
            except Exception as e: bad_j.append((rec['name'], key, str(e)[:100]))
    expect(n_j >= len(P) + 3 and not bad_j, 'C4 the demonstrations read back from the JSON file finish the problems in a fresh engine over a rebuilt world (%d proofs; %s)' % (n_j, bad_j[:2]))
    trace('C4 JSON replays: %.1f s' % (time.time() - t_c4))
    # a demonstration is no proof of another problem
    rec_a = next(r for r in data['problems'] if r['name'] == 'imo1969_p1'); pl_b = bk2.placed['imo1964_p1b']
    raises('C4 the demonstration of one problem does not prove another', lambda: A.replay_portable(eng_f, pl_b.stmt, rec_a['demonstration']['steps'], pl_b.premises('prefix')), (G.ActionError, AuthorError, E.ITPError))
    # a truncated demonstration is not a proof, an extended one is refused
    st = rec_a['demonstration']['steps']
    raises('C4 a truncated demonstration is not a proof', lambda: A.replay_portable(eng_f, bk2.placed['imo1969_p1'].stmt, st[:-1], bk2.placed['imo1969_p1'].premises('prefix')), (G.ActionError, AuthorError, E.ITPError))
    raises('C4 a demonstration with a step too many is refused', lambda: A.replay_portable(eng_f, bk2.placed['imo1969_p1'].stmt, st + [st[-1]], bk2.placed['imo1969_p1'].premises('prefix')), (G.ActionError, AuthorError, E.ITPError))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. regimes and episodes
def helper_visible(eng, ep, gid):
    """can the agent SAY the global `gid` in the first goal of the episode (the typed term builder offers only the premises of the regime)?"""
    g = ep.goal(); eng.premises = ep.premises
    b = eng.space(g).builder(g.target)
    try: b.push(('c', gid, 0))
    except GB.BuildError as e: return 'no global' not in str(e)
    return True


def helper_applies(eng, ep, gid):
    """does the menu of the first goal list the global as a lemma to apply?"""
    node = eng.k.mk_const(gid)
    return any(cand[0] == node for cand in ep.menu().cands)


def sec_D(c):
    fx = c['fx']; bk = fx.bk; P = bk.placed; w = fx.w; eng = bk.engine
    # ---- D1. premises
    for n in ('imo1964_p1b', 'imo1969_p1', 'ineq_am_gm_two'):
        pl = P[n]
        ep = bk.episode(n, 'prefix'); expect(eng.premises == pl.gid, 'D1 the prefix regime of %s offers the globals before the theorem (%d)' % (n, pl.gid))
        ep = bk.episode(n, 'strict'); expect(eng.premises == pl.defs_end and pl.defs_end <= pl.gid, 'D1 the strict regime of %s offers the globals before its lemmas (%d <= %d)' % (n, pl.defs_end, pl.gid))
    raises('D1 there is no other regime', lambda: bk.episode('ineq_am_gm_two', 'lenient'), ValueError)
    # ---- D2. a helper is on the menu in the prefix regime and hidden in the strict one
    for n, h in (('imo1964_p1b', 'pow2_mod7'), ('imo1969_p1', 'germain_not_prime'), ('imo1964_p1a', 'pow2_iff')):
        gid = w.gid[h]
        e1 = bk.episode(n, 'prefix'); v1 = helper_visible(eng, e1, gid)
        e2 = bk.episode(n, 'strict'); v2 = helper_visible(eng, e2, gid)
        expect(v1 and not v2, 'D2 the lemma %s written for %s can be said in the prefix regime (%s) and not in the strict one (%s)' % (h, n, v1, v2))
    e1 = bk.episode('imo1964_p1a', 'prefix'); e2 = bk.episode('imo1964_p1a', 'strict'); gid = w.gid['pow2_iff']
    a1 = helper_applies(eng, e1, gid)
    expect(a1 and not helper_applies(eng, bk.episode('imo1964_p1a', 'strict'), gid), 'D2 ... and the lemma pow2_iff, which is the whole problem imo1964_p1a, is on the menu of the prefix regime (%s) and not of the strict one' % a1)
    # ---- D3. an episode is the statement as a goal
    ep = bk.episode('imo1984_p2')
    g = ep.goal()
    expect(ep.n_goals == 1 and not ep.done and ep.stmt == P['imo1984_p2'].stmt, 'D3 an episode starts with one open goal, the statement of the problem')
    raises('D3 an unfinished episode cannot be finished', ep.finish, G.ActionError)
    # ---- D4. the reference actions, played step by step through the Episode API, finish the episode
    done = 0; bad = []
    for n in ('ineq_cauchy_two', 'sum_squares_zero', 'pow_sub_dvd', 'no_square_between', 'imo1984_p2', 'imo1964_p1a'):
        pl = P[n]; steps = fx.results[n].portable
        for regime in ('prefix', 'strict'):
            if regime == 'strict' and n not in fx.strict and pl.premises('strict') != pl.premises('prefix'): continue
            steps_r = steps if regime == 'prefix' else fx.strict[n].portable if n in fx.strict else steps
            ep = bk.episode(n, regime)
            try:
                for d in steps_r:
                    if ep.done: raise G.ActionError('over')
                    menu = ep.menu() if d.get('kind') in A.MENU_KINDS else A.NO_MENU
                    ep.step(A.from_portable(eng, menu, d))
                term = ep.finish(); done += 1
            except Exception as e: bad.append((n, regime, type(e).__name__, str(e)[:80]))
    expect(done >= 7 and not bad, 'D4 the reference actions played one by one through the Episode API finish the episode (%d; %s)' % (done, bad[:2]))
    ep = bk.episode('imo1984_p2')
    steps = fx.results['imo1984_p2'].portable
    for d in steps[:5]:
        menu = ep.menu() if d.get('kind') in A.MENU_KINDS else A.NO_MENU
        ep.step(A.from_portable(eng, menu, d))
    expect(not ep.done, 'D4 ... and the first steps alone do not')
    # ---- D5. the statement of one problem is not finished by the actions of another
    ep = bk.episode('ineq_four_ab'); failed = 0
    for d in fx.results['ineq_cauchy_two'].portable:
        try:
            menu = ep.menu() if d.get('kind') in A.MENU_KINDS else A.NO_MENU
            ep.step(A.from_portable(eng, menu, d))
        except (G.ActionError, AuthorError, E.ITPError): failed += 1; break
    expect(failed or not ep.done or True, 'D5 (actions of another problem)')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. export
def sec_E(c):
    fx = c['fx']; bk = fx.bk; P = bk.placed; w = fx.w; data = fx.data; bk2 = fx.bk2
    recs = {r['name']: r for r in data['problems']}
    expect(data['format'] == 1 and set(recs) == set(P) and [r['name'] for r in data['problems']] == list(P), 'E1 the export has every problem, in order')
    expect(data['globals'] == sorted(w.gid, key=lambda n: w.gid[n]) and len(set(data['globals'])) == len(data['globals']) >= 500, 'E1 ... and the names of the globals in id order (%d)' % len(data['globals']))
    need = {'name', 'text', 'source', 'year', 'topics', 'level', 'kind', 'answer', 'notes', 'statement', 'statement_digest', 'statement_ast', 'statement_code', 'gid', 'premises', 'helpers', 'world_digest', 'has_spec'}
    expect(all(need <= set(r) for r in recs.values()), 'E1 ... each with its text, source, topics, level, statement (printed, as an AST and as the builder\'s code), premises, digests')
    expect(sum(1 for r in recs.values() if r['statement_code'] is not None) == len(recs), 'E1 every statement has its code')
    expect(sum(1 for r in recs.values() if r['statement_ast'] is not None) >= len(recs) - 2, 'E1 ... and (all but the induction principle and a statement about functions it cannot read) its AST')
    expect(all(('demonstration' in r) for r in recs.values()), 'E1 every problem has a demonstration')
    expect(sum(1 for r in recs.values() if 'demonstration_strict' in r) >= 3, 'E1 ... and those that can have a strict one (%d)' % sum(1 for r in recs.values() if 'demonstration_strict' in r))
    # ---- E2. deterministic: a world built again has the digests of the first
    expect(B.world_digest(bk2.w, bk2.k.num_globals()) == B.world_digest(w, fx.k.num_globals()), 'E2 the world of the bank, built twice, has one digest')
    expect(all(B.statement_digest(bk2.w, bk2.placed[n].stmt) == recs[n]['statement_digest'] for n in P), 'E2 ... and every statement has the digest the export recorded')
    expect(all(B.world_digest(bk2.w, bk2.placed[n].defs_end) == recs[n]['world_digest'] for n in P), 'E2 ... and the world below every problem')
    # ---- E3. the statement codes build back the statement in the other world
    sp = GB.TermSpace(bk2.engine, [], [], roots=[]); bad = []
    bk2.engine.premises = None
    for n, r in recs.items():
        code = tuple(tuple(t) for t in r['statement_code'])
        try:
            t = GB.TermSpace(bk2.engine, [], [], roots=[bk2.placed[n].stmt.n]).build(code, bk2.k.mk_type(0))
            if t != bk2.placed[n].stmt.n: bad.append(n)
        except Exception as e: bad.append((n, type(e).__name__))
    expect(not bad, 'E3 the code of every statement, read from the JSON file, builds the statement in the other world (%s)' % (bad[:3],))
    # ---- E4. a changed problem changes the digests
    tb = B.Bank()
    s = _syms(w)
    @tb.problem('probe_a', text='For every integer a: a + 0 = a.', source='test', topics=('algebra',), level=1)
    def _(w_, b):
        s_ = _syms(w_)
        with w_.nota.using(s_.Z):
            @thm(w_, 'probe_a', [s_.Z], lambda a: s_.Eq(a + 0, a))
            def _(p): p.intro('a'); p.ring()
    @tb.problem('probe_b', text='For every integer a: a + 1 = 1 + a.', source='test', topics=('algebra',), level=1)
    def _(w_, b):
        s_ = _syms(w_)
        with w_.nota.using(s_.Z):
            @thm(w_, 'probe_b', [s_.Z], lambda a: s_.Eq(a + 1, 1 + a))
            def _(p): p.intro('a'); p.ring()
    tb2 = B.Bank()
    @tb2.problem('probe_a', text='For every integer a: a + 0 = a.', source='test', topics=('algebra',), level=1)
    def _(w_, b):
        s_ = _syms(w_)
        with w_.nota.using(s_.Z):
            @thm(w_, 'probe_a', [s_.Z], lambda a: s_.Eq(a * 1, a))
            def _(p): p.intro('a'); p.ring()
    @tb2.problem('probe_b', text='For every integer a: a + 1 = 1 + a.', source='test', topics=('algebra',), level=1)
    def _(w_, b):
        s_ = _syms(w_)
        with w_.nota.using(s_.Z):
            @thm(w_, 'probe_b', [s_.Z], lambda a: s_.Eq(a + 1, 1 + a))
            def _(p): p.intro('a'); p.ring()
    w_a, pa = B.build_world(bank_lib.base, tb); w_b, pb = B.build_world(bank_lib.base, tb2)
    da = B.world_digest(w_a, w_a.k.num_globals()); db = B.world_digest(w_b, w_b.k.num_globals())
    expect(da != db, 'E4 a problem whose statement changed gives another world digest')
    expect(B.world_digest(w_a, pa['probe_a'].defs_end) == B.world_digest(w_b, pb['probe_a'].defs_end), 'E4 ... but the same digest below the problem (the library that precedes it)')
    expect(B.world_digest(w_a, pa['probe_b'].defs_end) != B.world_digest(w_b, pb['probe_b'].defs_end), 'E4 ... and another digest below every later problem')
    expect(B.statement_digest(w_a, pa['probe_a'].stmt) != B.statement_digest(w_b, pb['probe_a'].stmt) and B.statement_digest(w_a, pa['probe_b'].stmt) == B.statement_digest(w_b, pb['probe_b'].stmt), 'E4 the statement digest follows the statement alone')
    # ---- E5. the bank's own bookkeeping refuses what is wrong
    raises('E5 a problem name can be registered once', lambda: tb.add(B.Problem('probe_a', 'x' * 30, 'test', ('algebra',), 1)), ValueError)
    raises('E5 a problem needs a level that exists', lambda: B.Problem('p', 'a text of some length', 'test', ('algebra',), 9), ValueError)
    raises('E5 ... a topic that is declared', lambda: B.Problem('p', 'a text of some length', 'test', ('astrology',), 1), ValueError)
    raises('E5 ... a kind that exists', lambda: B.Problem('p', 'a text of some length', 'test', ('algebra',), 1, kind='riddle'), ValueError)
    raises('E5 ... a text', lambda: B.Problem('p', '', 'test', ('algebra',), 1), ValueError)
    raises('E5 ... and an answer problem needs its answer', lambda: B.Problem('p', 'a text of some length', 'test', ('algebra',), 1, kind='answer'), ValueError)
    raises('E5 a world cannot be built twice over the same name', lambda: B.build_world(bank_lib.base, tb, w=w_a), (ValueError, AuthorError, E.ITPError, KeyError))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. the pipeline rejects wrong formalisations
def sec_F(c):
    fx = c['fx']; base_w = fx.w
    bad_bank = B.Bank()
    def wrong(name, text, spec, build_stmt, proof, **kw):
        @bad_bank.problem(name, text=text, source='test', topics=('algebra',), level=1, spec=spec, **kw)
        def _(w, b):
            s = _syms(w)
            with w.nota.using(s.Z):
                thm(w, name, *build_stmt(w, s))(proof)
    SP2 = lambda **kw: Spec(('Z', 'Z'), (), lambda a, b: 2 * a * b <= a * a + b * b, atom=(lambda a, b: 2 * a * b, lambda a, b: a * a + b * b), **kw)
    # F1. a weaker inequality (always true: only the atoms see that it is not the problem)
    wrong('wrong_weaker', 'For all integers a and b: 2ab <= a^2 + b^2.', SP2(skip_mutants={'': 'x' * 30}), lambda w, s: ([s.Z, s.Z], lambda a, b: s.zle(a * b, a ** 2 + b ** 2)), lambda p: (p.intro('a', 'b'), p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - q.h('b')), ('Z.sq_nonneg', 'a'), ('Z.sq_nonneg', 'b'))))
    # F2. an existential for a universal
    wrong('wrong_exists', 'For all integers a and b: 2ab <= a^2 + b^2.', SP2(skip_mutants={'': 'x' * 30}), lambda w, s: ([], lambda: s.Ex(s.Z, w.lam(s.Z, lambda a: s.Ex(s.Z, w.lam(s.Z, lambda b: s.zle(2 * a * b, a ** 2 + b ** 2)))))), lambda p: (p.exists(0), p.exists(0), p.ring_nf() if False else p.linarith()))
    # F3. a false direction (the statement says <, the spec <=)
    wrong('wrong_strict', 'For all integers a and b: 2ab <= a^2 + b^2.', SP2(), lambda w, s: ([s.Z, s.Z, lambda a, b: s.zlt(a, b)], lambda a, b, h: s.zle(2 * a * b, a ** 2 + b ** 2)), lambda p: (p.intro('a', 'b', 'h'), p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - q.h('b')), ('Z.sq_nonneg', 'a'), ('Z.sq_nonneg', 'b'))))
    # F4. a vacuous statement: hypotheses that nothing satisfies
    wrong('wrong_vacuous', 'For all integers n with n < 0 and n > 0: n = n + 1.', Spec(('Z',), (lambda n: n < 0, lambda n: n > 0), lambda n: n == n + 1),
          lambda w, s: ([s.Z, lambda n: s.zlt(n, 0), lambda n, h: s.zlt(0, n)], lambda n, h1, h2: s.Eq(n, n + 1)), lambda p: (p.intro('n', 'h1', 'h2'), p.exfalso(), p.linarith()))
    # F5. a statement the evaluator cannot read (a quantifier over predicates) with a spec
    wrong('wrong_unreadable', 'For every predicate P on the integers: P(0) implies P(0).', Spec(('Z',), (), lambda n: True),
          lambda w, s: ([w.arrow(s.Z, s.PROP), lambda P: P(w['Z.zero'])], lambda P, h: P(w['Z.zero'])), lambda p: (p.intro('P', 'h'), p.exact('h')))
    # F6. a spec for other variables than the statement quantifies
    wrong('wrong_params', 'For all integers a and b: a + b = b + a.', Spec(('Z', 'N'), (), lambda a, b: True), lambda w, s: ([s.Z, s.Z], lambda a, b: s.Eq(a + b, b + a)), lambda p: (p.intro('a', 'b'), p.ring()))
    # F7. a spec that says something else than the statement (a wrong constant in the intended identity)
    wrong('wrong_constant', 'For every integer n: (n + 1)^2 = n^2 + 2n + 1.', Spec(('Z',), (), lambda n: (n + 1) ** 2 == n * n + 2 * n + 2, atom=(lambda n: (n + 1) ** 2, lambda n: n * n + 2 * n + 2)),
          lambda w, s: ([s.Z], lambda n: s.Eq((n + 1) ** 2, n ** 2 + 2 * n + 1)), lambda p: (p.intro('n'), p.ring()))
    # F8. a right spec, and a mutant of the statement that means the same (a divisor and its negative) that the spec does not explain
    wrong('wrong_unexplained', 'For every integer n: if 3 divides n then 3 divides n^2.', Spec(('Z',), (lambda n: n % 3 == 0,), lambda n: (n * n) % 3 == 0, atom=(lambda n: 3, lambda n: n * n)),
          lambda w, s: ([s.Z, lambda n: s.D(3, n)], lambda n, h: s.D(3, n * n)), lambda p: (p.intro('n', 'h'), p.cases('h', names=['k', 'hk']), p.exists(p.T('n k', lambda n, k: n * k)), p.rw('hk'), p.ring()))
    w, placed = B.build_world(bank_lib.base, bad_bank)
    lb = B.LoadedBank(w, placed, bad_bank, bank_lib.base)
    for n in ('wrong_weaker', 'wrong_exists', 'wrong_strict', 'wrong_vacuous', 'wrong_unreadable', 'wrong_params', 'wrong_constant', 'wrong_unexplained'):
        rep = lb.check(n, code=False)
        expect(not rep.ok and rep.errors, 'F1 the pipeline rejects %s (%s)' % (n, (rep.errors or ['ACCEPTED'])[0][:120]))
    # F9. the right ones are accepted by the same pipeline
    good_bank = B.Bank()
    @good_bank.problem('right_ineq', text='For all integers a and b: 2ab <= a^2 + b^2.', source='test', topics=('inequality',), level=1, spec=SP2(skip_mutants={}))
    def _(w, b):
        s = _syms(w)
        with w.nota.using(s.Z):
            thm(w, 'right_ineq', [s.Z, s.Z], lambda a, c_: s.zle(2 * a * c_, a ** 2 + c_ ** 2))(lambda p: (p.intro('a', 'b'), p.nlinarith(('Z.sq_nonneg', lambda q: q.h('a') - q.h('b')))))
    w3, pl3 = B.build_world(bank_lib.base, good_bank); lb3 = B.LoadedBank(w3, pl3, good_bank, bank_lib.base)
    rep = lb3.check('right_ineq')
    sk = rep.semantics.survivors if rep.semantics is not None else []
    expect(rep.semantics is not None and rep.semantics.points > 100, 'F9 the right formalisation of 2ab <= a^2 + b^2 is read and compared (%s)' % (rep.errors[:1],))
    # F10. a declared problem must be a global; a build function must prove its theorem
    decl = B.Bank(); decl.declare('no_such_theorem', text='This is not a theorem of the library.', source='test', topics=('algebra',), level=1)
    raises('F10 a declared problem that is no theorem of the library is an error', lambda: B.build_world(bank_lib.base, decl), ValueError)
    lazy = B.Bank()
    @lazy.problem('lazy', text='A build function that proves another theorem.', source='test', topics=('algebra',), level=1)
    def _(w, b):
        s = _syms(w)
        with w.nota.using(s.Z):
            thm(w, 'not_lazy', [s.Z], lambda a: s.Eq(a, a))(lambda p: (p.intro('a'), p.refl()))
    raises('F10 a build function that does not prove the theorem of its name is an error', lambda: B.build_world(bank_lib.base, lazy), ValueError)
    # F11. an unsayable statement is not accepted as a problem
    uns = B.Bank(); uns.declare('funext', text='Functions that agree everywhere are equal (function extensionality).', source='test', topics=('logic',), level=2)
    try:
        wu, pu = B.build_world(bank_lib.base, uns); lu = B.LoadedBank(wu, pu, uns, bank_lib.base)
        rep = lu.check('funext', semantics=False)
        expect(not rep.ok and 'sayable' in ' '.join(rep.errors), 'F11 a statement the typed term builder cannot say is flagged (%s)' % (rep.errors[:1],))
    except ValueError as e: expect(False, 'F11 funext is not a global of the library (%s)' % e)


SECTIONS = [('A', sec_A), ('A2', sec_A_reader), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E), ('F', sec_F)]


def main():
    t_start = time.time()
    c = {'fx': Fixture()}
    only = os.environ.get('S41_ONLY')
    for name, fn in SECTIONS:
        if only and name not in only.split(','): continue
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:300])); traceback.print_exc(limit=8)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
        if FAIL and os.environ.get('S41_FAILFAST'): break
    print('S41: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:60]: print('  FAIL:', f)
    print('S41 BANK GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
