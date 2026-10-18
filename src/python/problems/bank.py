"""bank.py -- the problem bank: formal olympiad problems with their provenance, their intended meaning, a reference proof and the regimes they are attacked in.

A PROBLEM is a theorem of the bank's world, a library built from scripts (`imo_lib.build_imo` and the problems of `bank_lib`), plus what a dataset needs to know about it:

    name        the unique name of the problem AND of the theorem that proves it (`imo1969_p1`); no text interface is implied: the agent sees the statement as a goal
    text        the problem as it is stated in words (ASCII; this is what a human checks the formal statement against)
    source      where it comes from (`IMO 1959 P1`, `ISL 2019 N1`, `folklore`); `year`
    topics      from TOPICS;  level  1..5 (LEVELS);  kind  'prove' | 'answer' (the statement contains the answer: "determine all ..") | 'exists' (an example is asked for)
    spec        the intended meaning as Python (`semantic.Spec`): the formal statement is READ back into an AST and compared with it on a box, mutated statements are the negative controls
    helpers     names of the lemmas of the library that were written for this problem (they are hidden in the STRICT regime)
    strict      True when the reference proof is SELF-CONTAINED (it proves its lemmas inline: it replays in the strict regime too); default: True for a problem with a build function
    build       the function that declares the helpers and the theorem in the world: `build(w, b)`; None when a library stage already built it

REGIMES. A problem is attacked inside the world in which the library precedes it: the engine's `premises` is a number of globals.  'prefix': every global before the theorem, the problem's own helper lemmas
included (the reference proof's lemmas are on the menu); 'strict': every global before the problem's block, its definitions aside (the helpers are not offered: the agent must find the ideas).  The
world is deterministic, so a problem is identified by its name and the world's digest below its block.

    bank = bank_lib.load()                 # builds the world once (a couple of seconds), the index of the problems
    ep = bank.episode('imo1959_p1')        # a gym Episode for the statement, with the premises of the prefix regime
    report = bank.check('imo1959_p1')      # faithfulness (semantics), sayability of the statement as a term, reference demonstration
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, json, hashlib, collections, time, traceback
import itp_env_lnl as E
from author import AuthorError
import semantic as SEM
import gym_env as G
import gym_build as GB
import gym_acts as ACTS
import recorder as REC

LEVELS = {1: 'a single arithmetic step (ring / linarith / nlinarith) after the introductions',
          2: 'a short proof with one idea: a case split, a witness, a divisibility computation, one induction',
          3: 'an olympiad lemma: several ideas or several lemmas (15 - 60 actions)',
          4: 'an olympiad problem with a short solution (60 - 200 actions)',
          5: 'a hard olympiad problem'}
KINDS = ('prove', 'answer', 'exists')
TOPICS = frozenset({'algebra', 'identity', 'inequality', 'number-theory', 'divisibility', 'gcd', 'prime', 'parity', 'modular', 'diophantine', 'induction', 'functional-equation',
                    'polynomial', 'sequence', 'combinatorics', 'logic', 'descent', 'construction', 'bounding', 'case-analysis'})
REGIMES = ('prefix', 'strict')


class Problem:
    def __init__(self, name, text, source, topics, level, kind='prove', spec=None, answer=None, helpers=(), year=None, notes='', build=None, strict=None):
        self.name = name; self.text = text; self.source = source; self.topics = tuple(topics); self.level = level; self.kind = kind
        self.spec = spec; self.answer = answer; self.helpers = tuple(helpers); self.year = year; self.notes = notes; self.build = build
        self.strict = (not self.helpers or build is not None) if strict is None else strict         # the reference proof is self-contained: it replays when the lemmas written for the problem are hidden
        bad = []
        if level not in LEVELS: bad.append('level %r' % (level,))
        if kind not in KINDS: bad.append('kind %r' % (kind,))
        for t in self.topics:
            if t not in TOPICS: bad.append('topic %r' % (t,))
        if not self.topics: bad.append('no topics')
        if not text: bad.append('no text')
        if kind == 'answer' and not answer: bad.append('an answer problem needs its answer')
        if bad: raise ValueError('problem %s: %s' % (name, ', '.join(bad)))


class Block:
    """what a problem's build function gets besides the world: it marks where the definitions the STATEMENT needs end (what follows is hidden in the strict regime)"""
    def __init__(self, w): self.w = w; self.start = w.k.num_globals(); self.defs_end = None
    def statement_ready(self): self.defs_end = self.w.k.num_globals()


class Placed:
    """a problem in a built world: `start` the first global of its block, `defs_end` the end of its definitions, `gid` the theorem, `stmt` the statement (a Term)"""
    def __init__(self, problem, start, defs_end, gid, stmt):
        self.problem = problem; self.start = start; self.defs_end = defs_end; self.gid = gid; self.stmt = stmt
    def premises(self, regime):
        if regime == 'prefix': return self.gid
        if regime == 'strict': return self.defs_end
        raise ValueError('no regime %r' % (regime,))


class Bank:
    def __init__(self): self.problems = collections.OrderedDict()
    def add(self, problem):
        if problem.name in self.problems: raise ValueError('the problem %s is already in the bank' % problem.name)
        self.problems[problem.name] = problem; return problem
    def problem(self, name, **kw):
        """decorator: `@BANK.problem('name', text=.., source=.., topics=.., level=.., spec=..)` over `def build(w, b)` that declares the helpers and the theorem `name`"""
        def deco(fn):
            self.add(Problem(name, build=fn, **kw)); return fn
        return deco
    def declare(self, name, **kw):
        """a problem whose theorem a library stage already proves (no build function)"""
        return self.add(Problem(name, build=None, **kw))


BANK = Bank()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ building the world
def build_world(base, bank=None, w=None, only=None):
    """the world of the bank: `base(w)` (a library builder such as imo_lib.build_imo) and then the build function of every problem, in the order they were registered. Returns (w, {name: Placed}).
    `only`: names to build (the declared problems are always located). A theorem that a build function should have proved but did not is an error."""
    bank = bank or BANK
    w = base(w) if w is not None else base()
    placed = collections.OrderedDict(); k = w.k
    for p in bank.problems.values():
        if p.build is not None and (only is None or p.name in only):
            if p.name in w.gid: raise ValueError('the name %s is already a global of the world' % p.name)
            b = Block(w); p.build(w, b)
            if p.name not in w.gid: raise ValueError('the build function of %s did not prove a theorem of that name' % p.name)
            start = b.start; defs_end = b.defs_end if b.defs_end is not None else b.start
        elif p.build is None:
            if p.name not in w.gid: raise ValueError('the declared problem %s is not a global of the base library' % p.name)
            start = None; defs_end = None
        else: continue
        gid = w.gid[p.name]
        if start is None:
            gids = [w.gid[h] for h in p.helpers] + [gid]
            for h in p.helpers:
                if h not in w.gid: raise ValueError('problem %s: the helper %s is not a global' % (p.name, h))
            start = defs_end = min(gids)
        stmt = w.T(k.global_info(gid)[1])
        placed[p.name] = Placed(p, start, defs_end, gid, stmt)
    return w, placed


def world_digest(w, upto):
    """a hash of the library below `upto`: the name and the printed type of every global, in id order. A dataset names its world by it."""
    k = w.k; names = sorted(w.gid, key=lambda n: w.gid[n]); h = hashlib.sha256()
    for nm in names:
        gid = w.gid[nm]
        if gid >= upto: break
        h.update(('%d %s : %s\n' % (gid, nm, w.show(w.T(k.global_info(gid)[1])))).encode())
    return h.hexdigest()


def statement_digest(w, stmt): return hashlib.sha256(w.show(stmt).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the checks
class ProblemReport:
    def __init__(self, name):
        self.name = name; self.errors = []; self.notes = []; self.semantics = None; self.code = None; self.demo = None; self.demo_seconds = None; self.ok_demo = None
    @property
    def ok(self): return not self.errors
    def __repr__(self): return 'ProblemReport(%s ok=%s errors=%s)' % (self.name, self.ok, self.errors[:2])


class LoadedBank:
    """a built world with the index of its problems; the engine is made on demand and shared (its `premises` is set per episode)"""
    def __init__(self, w, placed, bank=None, base=None):
        self.w = w; self.k = w.k; self.placed = placed; self.bank = bank or BANK; self.base = base; self._eng = None; self._sem = None; self._recs = None
    def names(self, level=None, topic=None):
        return [n for n, pl in self.placed.items() if (level is None or pl.problem.level == level) and (topic is None or topic in pl.problem.topics)]
    @property
    def engine(self):
        if self._eng is None: self._eng = G.Engine(self.w)
        return self._eng
    @property
    def semantics(self):
        if self._sem is None: self._sem = SEM.Semantics(self.w)
        return self._sem

    def episode(self, name, regime='prefix', engine=None):
        pl = self.placed[name]; eng = engine or self.engine
        return G.Episode(eng, pl.stmt, name, premises=pl.premises(regime))

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- the statement as data
    def statement_text(self, name): return self.w.show(self.placed[name].stmt)

    def statement_ast(self, name):
        try: return self.semantics.decode(self.placed[name].stmt.n)
        except SEM.Unsupported: return None

    def statement_code(self, name):
        """the canonical code of the statement in the typed term builder (a tuple of tokens, with global ids), checked by building it back; None when the statement is not sayable"""
        pl = self.placed[name]; eng = self.engine; eng.premises = None
        space = GB.TermSpace(eng, [], [], roots=[pl.stmt.n])
        try:
            code = space.encode(pl.stmt.n)
            t = space.build(code, self.k.mk_type(0))
        except (GB.NotEncodable, GB.BuildError, AuthorError, E.ITPError, RuntimeError): return None
        return code if t == pl.stmt.n else None

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- checks
    def check_semantics(self, name, **kw):
        pl = self.placed[name]; pr = pl.problem
        if pr.spec is None: return None
        ast = self.statement_ast(name)
        if ast is None: rep = SEM.Report(); rep.errors.append('the statement is outside the fragment the evaluator reads'); return rep
        return SEM.check_spec(self.semantics, ast, pr.spec, **kw)

    def recordings(self):
        """({name: Recording}) of a build with every proof logged: the world is built AGAIN (the log needs the proofs to run on a RecordingProof)"""
        if self._recs is None:
            if self.base is None: raise ValueError('this bank was loaded without its base builder')
            w2, recs = REC.record_world(lambda: build_world(self.base, self.bank)[0])
            self._recs = (w2, recs)
        return self._recs

    def demonstrate(self, name, engine=None, regime='prefix'):
        """replay the reference proof of a problem through the action language, in the premises of a regime: the recorder's Result"""
        w2, recs = self.recordings()
        eng = engine or G.Engine(w2)
        return REC.replay(eng, recs[name], premises=self.placed[name].premises(regime))

    def check(self, name, semantics=True, code=True, demo=False):
        rep = ProblemReport(name); pl = self.placed[name]; rep.strict_ok = None
        if semantics and pl.problem.spec is not None:
            rep.semantics = self.check_semantics(name)
            if not rep.semantics.ok: rep.errors.extend('semantics: ' + e for e in rep.semantics.errors)
        if code:
            rep.code = self.statement_code(name)
            if rep.code is None: rep.errors.append('the statement is not sayable in the typed term builder')
        if demo:
            t0 = time.time(); res = self.demonstrate(name); rep.demo_seconds = time.time() - t0; rep.demo = res; rep.ok_demo = res.ok
            if not res.ok: rep.errors.append('the reference proof does not replay in the action language: %s' % (res.gap,))
            if pl.problem.strict and pl.premises('strict') != pl.premises('prefix'):
                res2 = self.demonstrate(name, regime='strict'); rep.strict_ok = res2.ok
                if not res2.ok: rep.errors.append('the reference proof is not self-contained (it does not replay when the lemmas written for the problem are hidden): %s' % (res2.gap,))
        return rep

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- export
    def record(self, name, demo=None, strict_demo=None):
        """the problem as a plain dict (JSON-ready): see the module documentation. `demo`, `strict_demo`: recorder Results (prefix regime, strict regime) whose portable steps are included"""
        pl = self.placed[name]; pr = pl.problem; w = self.w
        code = self.statement_code(name); ast = self.statement_ast(name)
        d = {'name': name, 'text': pr.text, 'source': pr.source, 'year': pr.year, 'topics': list(pr.topics), 'level': pr.level, 'kind': pr.kind, 'answer': pr.answer, 'notes': pr.notes,
             'statement': w.show(pl.stmt), 'statement_digest': statement_digest(w, pl.stmt), 'statement_ast': ast, 'statement_code': None if code is None else [list(t) for t in code],
             'gid': pl.gid, 'premises': {'prefix': pl.premises('prefix'), 'strict': pl.premises('strict')}, 'helpers': list(pr.helpers),
             'world_digest': world_digest(w, pl.defs_end), 'has_spec': pr.spec is not None}
        if demo is not None and demo.ok: d['demonstration'] = {'regime': 'prefix', 'steps': demo.portable, 'length': len(demo.portable)}
        if strict_demo is not None and strict_demo.ok: d['demonstration_strict'] = {'regime': 'strict', 'steps': strict_demo.portable, 'length': len(strict_demo.portable)}
        return d

    def export(self, path, demos=None, strict_demos=None):
        """write the bank to a JSON file: {'globals': [names in id order], 'world_digest', 'problems': [record, ..]}; `demos`, `strict_demos`: {name: Result}"""
        k = self.k; names = sorted(self.w.gid, key=lambda n: self.w.gid[n])
        data = {'format': 1, 'globals': names, 'world_digest': world_digest(self.w, k.num_globals()),
                'problems': [self.record(n, (demos or {}).get(n), (strict_demos or {}).get(n)) for n in self.placed]}
        with open(path, 'w') as f: json.dump(data, f, indent=1)
        return data


def load(base=None, bank=None, only=None):
    """build the bank's world once and index its problems; `base`: the library builder below the problems"""
    if base is None: raise ValueError('bank.load needs the base library builder (see bank_lib.load)')
    w, placed = build_world(base, bank, only=only)
    return LoadedBank(w, placed, bank, base)
