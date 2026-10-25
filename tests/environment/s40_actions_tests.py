"""S40 gate: the second action language -- the typed term builder (gym_build.py), the actions that say their arguments (gym_acts.py), their portable form, and the recorder (recorder.py) that turns the
library's authored proofs into sequences of such actions.

An agent without a text interface still has to be able to SAY a statement, a witness, an induction principle and its starting point, the arguments of a lemma. Nothing is trusted -- a term is built by the
builder, an action is carried out by the tactic layer, the finished proof is checked by the kernel (and here also by the independent checker) -- so what this gate pins down is that the machinery does
what its documentation says, for every kind of input, including those that a learner will produce by accident:

  A. THE BUILDER.  Every term of the library's proofs (more than 700: check A1) is said and built back as the same kernel term; random walks over the choices it offers (many contexts, many types) give terms the kernel
     accepts, every offered choice is legal, a code means the same in another engine, an illegal choice is refused and changes nothing, codes mutated at random never crash it; numerals; a head applied to MORE
     arguments than its type has binders (`Z.strong_induction P a H n h x e`); a principle whose conclusion is `P n` and whose arguments wait for the motive (`Z.int_induction`); premises limit the library.
  B. THE ACTIONS.  CLAIM, REFINE (goals `('g',)`), APPLY_WITH (both modes), REWRITE_WITH, REWRITE_AT, FORWARD, INDUCT_WITH (also generalising), INJECT, LINCOMB and USE with a code, and CASES / INDUCT that
     generalise: the goals they leave, the hypotheses they add, and, for each, the actions it refuses (wrong shape, wrong type, a goal it cannot determine, a fact that is there already) -- refusals change nothing.
  C. THE PORTABLE FORM.  Actions as plain JSON data: a round trip changes nothing, the menu's own action is found again and otherwise the equivalent action that says its arguments, pool-relative actions have
     no portable form, and data that is no action (a missing key, a number where a list is wanted, a global that does not exist) is ActionError, never another exception.
  D. PROBES.  Probing leaves the kernel as it was, accepted or refused; the outcome depends on the budget only by running out of it; the same probe gives the same goals.
  E. THE MENU.  The enumerators of the second language (FORWARD, REWRITE_AT, INDUCT_WITH, INJECT, CLAIM hints): bounded, deterministic, every action found again from its signature, nearly all of them
     carried out by the probe, restricted by the engine's kinds.
  F. THE RECORDER.  The authored proofs of the library replay as sequences of actions (>= 453 of 459; the six that do not are the documented ones, docs/BANK.md section 7b), every replay re-checked by the kernel and by the
     independent checker, every one replays again from its JSON in a fresh engine, changed or truncated or extended sequences are refused or end in a proof the kernel checks, a proof is no proof of
     another statement.
  G. NAMES.  The engine's tag for the names it introduces (the recorder's, so that they never clash with an author's).
  H. FREE PLAY.  Random play with the actions of both languages on true and false statements: a refused step changes nothing, a live step leaves the goals the probe predicted, what is finished is
     checked by the kernel and by the independent checker, a false statement is never finished.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, io, time, json, random, collections, functools, hashlib, traceback
sys.setrecursionlimit(200000)
import itp_env_lnl as E
import imo_lib, recorder as REC
import gym_env as G, gym_build as GB, gym_acts as A
from gym_env import Kind, Action
from author import AuthorError, Term
import indep_checker as ic
from harness import parse_raw

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S40_TRACE'))
def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond
REFUSALS = (G.ActionError, AuthorError, E.ITPError)
def refuses(what, fn, exc=REFUSALS):
    """fn() must raise one of exc"""
    try: fn()
    except exc: return expect(True, what)
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:80]))
    return expect(False, what + ' (it raised nothing)')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ helpers
def state(k, w):
    """everything a probe must leave as it was"""
    return (tuple(k.pool_sizes()), k.num_globals(), k.get_fuel(), w.probing)


def refusal_effect(eng, before, after, n_noprogress):
    """what a refused probe may have done to the kernel: nothing, except that an action the tactic layer carried out but that made no progress (the goal came back as it was) leaves the nodes of that goal
    (the kernel keeps what a successful probe made). Returns a description of what is wrong, or None."""
    if before[1:] != after[1:]: return 'globals, fuel or probing changed'
    if any(b < a for a, b in zip(before[0], after[0])): return 'a pool shrank'
    if before[0] != after[0] and eng.stats['probe_noprogress'] == n_noprogress: return 'nodes left behind by a refusal that was not "no progress" (%s)' % [(i, a, b) for i, (a, b) in enumerate(zip(before[0], after[0])) if a != b][:3]
    return None


def nodes_of(rec):
    """the nodes of the log of a recorded proof"""
    st = [rec.root]
    while st:
        n = st.pop()
        yield n
        st.extend(n.kids)


class Fixture:
    """the world of the library (every theorem proved on a recording proof), one engine that offers every kind of action, and the replay of every recording (made once, on first use)"""
    def __init__(self):
        t0 = time.time()
        self.w, self.recs = REC.record_world(imo_lib.build_imo)
        self.k = self.w.k; self.gid = self.w.gid
        self.eng = G.Engine(self.w, kinds=set(G.Kind), strict=True)
        self._results = None
        trace('fixture %.1f s' % (time.time() - t0))
    @property
    def results(self):
        if self._results is None:
            t0 = time.time()
            self._results = {nm: REC.replay(self.eng, rec) for nm, rec in self.recs.items()}
            self.eng.premises = None
            trace('replay of all recordings %.1f s' % (time.time() - t0))
        return self._results
    def walk(self, name, hook):
        """replay the portable actions of a recorded proof as an agent would; `hook(i, p, g, act, menu)` is called before each step (g: the goal acted on)"""
        res = self.results[name]
        assert res.ok, (name, res.gap)
        eng = self.eng; eng.premises = self.recs[name].gid
        try:
            p = eng.start(self.recs[name].stmt)
            for i, d in enumerate(res.portable):
                g = eng.goal_of(p, 0)
                menu = eng.menu(g) if d['kind'] in A.MENU_KINDS else A.NO_MENU
                act = A.from_portable(eng, menu, d)
                hook(i, p, g, act, menu)
                eng.replay(p, act, menu)
            return p
        finally: eng.premises = None
    def first_goal(self, name):
        """the first (settled) goal of a recorded theorem, with the premises it had (None when settling closes it)"""
        eng = self.eng; eng.premises = self.recs[name].gid
        try:
            p = eng.start(self.recs[name].stmt)
            return eng.goal_of(p, 0) if p.n_goals else None
        finally: eng.premises = None
    def step_of(self, name, pred):
        """(i, goal, action) of the first step of a recorded proof whose action satisfies pred"""
        found = []
        def hook(i, p, g, act, menu):
            if not found and pred(act): found.append((i, g, act))
        self.walk(name, hook)
        return found[0] if found else None


def act(kind, *codes, arg=0, hyps=()):
    return Action(kind, arg, (), tuple(tuple(c) for c in codes), tuple(hyps))


def ref(gid): return (('c', gid, 0),)
def hyp(l): return (('v', l, 0),)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. the builder
WALK_TYPES = ('Z', 'Prop', 'Z->Z', 'Z->Prop', 'N')


STRUCTURE_TAGS = ('lam', 'all', 'imp', 'dep', 'arr', 's')
TOKEN_RANK = {t: 0 for t in STRUCTURE_TAGS}; TOKEN_RANK.update({'g': 1, 'n': 2, 'v': 3, 'c': 4, 'p': 5})


def token_order_problem(toks):
    """the order in which a builder lists its choices: binders and sorts, the goal, numerals, variables (the latest first), globals (by id), pool terms. None, or what is wrong."""
    ranks = [TOKEN_RANK[t[0]] for t in toks]
    if ranks != sorted(ranks): return 'the kinds of choice are not in the documented order: %s' % [t[0] for t in toks][:12]
    vs = [t[1] for t in toks if t[0] == 'v']; cs = [t[1] for t in toks if t[0] == 'c']
    if vs != sorted(vs, reverse=True) or len(set(vs)) != len(vs): return 'the variables are not listed latest first, once each: %s' % vs[:8]
    if cs != sorted(cs) or len(set(cs)) != len(cs): return 'the globals are not listed by id, once each: %s' % cs[:8]
    if len(set(toks)) != len(toks): return 'a choice is listed twice'
    return None


def builder_fingerprint(b):
    """everything about a builder that a push could change (the frames of the stack down to their dicts)"""
    return (tuple(b.code), b.hole_type, b.done, tuple(b.ctx), tuple(b.goals), b.allow_goals, tuple(h.ident for h in b.opened),
            tuple((f.kind, f.j, f.stage, f.kk, f.dom, f.level, f.hole, f.flex, f.expect, None if f.subst is None else tuple(sorted(f.subst.items())), None if f.auto is None else tuple(sorted(f.auto))) for f in b.stack))


def walk_term(b, rng, max_tokens=24, legal_every=0):
    """a random walk over the choices a builder offers: (the builder, number of tokens, the tokens that were offered but refused -- and, under the token None, other things that are wrong). Past half the
    tokens only the choices that add no hole are taken."""
    n = 0; illegal = []
    while not b.done:
        toks = b.tokens()
        if not toks: return b, n, illegal
        prob = token_order_problem(toks)
        if prob: illegal.append((None, prob))
        if legal_every and n % legal_every == 0:
            fp = builder_fingerprint(b)
            for t in rng.sample(toks, min(len(toks), 40)):
                bb = b.copy()
                try: bb.push(t)
                except GB.BuildError as e: illegal.append((t, str(e)[:80]))
            if builder_fingerprint(b) != fp: illegal.append((None, 'pushing on a copy of a builder changed the builder'))
        if n >= max_tokens // 2:
            small = [t for t in toks if t[0] == 'n' or (t[0] in 'vc' and t[2] == 0)]
            toks = small or toks
        b.push(rng.choice(toks)); n += 1
        if n > max_tokens: return b, n, illegal
    return b, n, illegal


def builder_snapshot(b):
    return (tuple(b.code), b.hole_type, len(b.stack), tuple(b.tokens()), b.done)


def sec_A(c):
    fx = c['fx']; w, k, eng = fx.w, fx.k, fx.eng
    eng.premises = None
    # ---- A1. the terms of the library's proofs: said, built back as the same term
    tot = collections.Counter(); bad = []
    for nm, rec in fx.recs.items():
        eng.premises = rec.gid
        for n in nodes_of(rec):
            if n.op not in ('exact', 'have', 'exists') or n.info is None: continue
            t = n.info.get({'exact': 'term', 'have': 'P', 'exists': 'witness'}[n.op])
            if t is None: continue
            node = t.n if isinstance(t, Term) else t; pre = n.pre
            ok, ty = k.infer_open(node, pre.hyp_types)
            if not ok: tot['illtyped'] += 1; continue
            tot['terms'] += 1
            space = GB.TermSpace(eng, pre.hyp_types, pre.hyp_names, roots=[pre.target])
            try: code = space.encode(node, ty)
            except GB.BuildError as e: tot['unsayable'] += 1; bad.append((nm, str(e)[:80])); continue
            tot['tokens'] += len(code)
            back = GB.TermSpace(eng, pre.hyp_types, pre.hyp_names, roots=[pre.target]).build(code, ty)
            if back != node: tot['mismatch'] += 1; bad.append((nm, 'builds a different term'))
            if not k.check_open(back, ty, pre.hyp_types)[0]: tot['unchecked'] += 1
            if len(code) > GB.MAX_CODE: tot['toolong'] += 1
    eng.premises = None
    expect(tot['terms'] >= 700 and tot['tokens'] >= 4000, 'A1 the proofs of the library have many terms to say (%s)' % dict(tot))
    expect(tot['mismatch'] == 0 and tot['unchecked'] == 0 and tot['illtyped'] == 0 and tot['toolong'] == 0, 'A1 every term is built back as itself, and the kernel accepts it (%s; %s)' % (dict(tot), bad[:3]))
    expect(tot['unsayable'] <= 1 and all(b[0] == 'Z.induction_on' for b in bad), 'A1 the only term that cannot be said is the `match` of Z.induction_on (%s)' % (bad[:3],))
    # ---- A2. random walks over the choices: terms the kernel accepts, offered choices legal, codes mean the same elsewhere
    eng2 = G.Engine(w, kinds=set(G.Kind))
    contexts = [('imo2019_p1_fwd', fx.first_goal('imo2019_p1_fwd')), ('Z.sqrt2_aux', fx.first_goal('Z.sqrt2_aux')), ('N.beq_eq', fx.first_goal('N.beq_eq'))]
    PROP = k.mk_type(0)
    stats = collections.Counter()
    for ci, (nm, g) in enumerate(contexts):
        eng.premises = fx.recs[nm].gid; eng2.premises = fx.recs[nm].gid
        sp = GB.TermSpace(eng, g.hyp_types, g.hyp_names, roots=[g.target]); sp2 = GB.TermSpace(eng2, g.hyp_types, g.hyp_names, roots=[g.target])
        Zt, Nt = w['Z'].n, w['N'].n
        expected = {'Z': Zt, 'Prop': PROP, 'Z->Z': k.mk_pi(Zt, Zt), 'Z->Prop': k.mk_pi(Zt, PROP), 'N': Nt}
        for label in WALK_TYPES:
            rng = random.Random(1000 * ci + len(label)); done = 0
            for i in range(24):
                b, n, illegal = walk_term(sp.builder(expected[label]), rng, legal_every=3 if i < 8 else 0)
                stats['offered illegal'] += sum(1 for x in illegal if x[0] is not None); stats['order or copy problems'] += sum(1 for x in illegal if x[0] is None)
                if illegal: stats['illegal example'] = illegal[0]
                if not b.done: stats['unfinished'] += 1; continue
                done += 1; term = b.term(); stats['walks'] += 1; stats['tokens'] += n
                if not k.check_open(term, expected[label], g.hyp_types)[0]: stats['ill-typed'] += 1
                if sp2.build(tuple(b.code), expected[label]) != term: stats['other engine'] += 1             # the same code in another engine, a fresh space
                if GB.TermSpace(eng, g.hyp_types, g.hyp_names).build(tuple(b.code), expected[label], check=False) != term: stats['other space'] += 1
                try:
                    code = sp.encode(term, expected[label])
                    if sp2.build(code, expected[label]) != term: stats['encode round trip'] += 1
                    if len(code) > n: stats['encode longer than the walk'] += 1
                except GB.NotEncodable: stats['unencodable'] += 1
            stats['unfinished-by-type ' + label] += 24 - done
    eng.premises = None; eng2.premises = None
    expect(stats['walks'] >= 150 and stats['tokens'] >= 300, 'A2 the walks finish and are not trivial (%s)' % dict(stats))
    expect(stats['offered illegal'] == 0, 'A2 every choice the builder offers is legal (%s)' % (stats['illegal example'],))
    expect(stats['order or copy problems'] == 0, 'A2 the choices are listed in the documented order, once each, and a push on a copy of a builder leaves the builder as it was (%s)' % (stats['illegal example'],))
    expect(stats['ill-typed'] == 0, 'A2 every finished term has the type that was wanted (the kernel says so)')
    expect(stats['other engine'] == 0 and stats['other space'] == 0, 'A2 a code builds the same term in another engine and in a fresh space')
    expect(stats['encode round trip'] == 0 and stats['encode longer than the walk'] == 0 and stats['unencodable'] <= 3, 'A2 the canonical code of a built term builds it back and is not longer than the code it came from (%s)' % dict(stats))
    # ---- A3. illegal choices are refused and change nothing
    g = contexts[0][1]; eng.premises = fx.recs[contexts[0][0]].gid
    sp = GB.TermSpace(eng, g.hyp_types, g.hyp_names, roots=[g.target])
    Zt = w['Z'].n; gid = fx.gid; zle = gid['Z.le']
    wrong = [('lam',), ('all',), ('imp',), ('arr',), ('s', 0), ('s', 7), ('g',), ('p', 0), ('c', 10 ** 6, 0), ('c', -1, 0), ('v', 999, 0), ('v', -1, 0), ('n', 10 ** 19), ('n', True), ('n', 1.5), ('n',), ('zz',), (),
             'n', ['n', 1], ('c', zle, 2, 0), ('c', zle, -1), ('c', gid['Z.zero'], -1), ('v', 0, -1), ('v', 0, 10 ** 9), ('c', gid['Z.add'], 1)]
    b = sp.builder(Zt); before = builder_snapshot(b)
    for t in wrong:
        refuses('A3 the choice %r where an integer is wanted is refused' % (t,), lambda: b.push(t), GB.BuildError)
        expect(builder_snapshot(b) == before, 'A3 ... and the builder is as it was (%r)' % (t,))
    b = sp.builder(PROP); before = builder_snapshot(b)
    for t in (('n', 1), ('lam',), ('arr',), ('s', 0), ('s', -1), ('g',), ('v', 0, 0), ('c', gid['Z.add'], 1)):
        refuses('A3 the choice %r where a proposition is wanted is refused' % (t,), lambda: b.push(t), GB.BuildError)
        expect(builder_snapshot(b) == before, 'A3 ... and the builder is as it was (%r)' % (t,))
    # (a head applied to ALL its arguments is accepted wherever its conclusion might have the wanted type by computation, and over-applied heads are accepted: the kernel decides when the term is finished)
    for label, hole, codes in (('a proposition for an integer', Zt, (('c', zle, 2), ('n', 1), ('n', 2))), ('an integer for a proposition', PROP, (('c', gid['Z.add'], 2), ('n', 1), ('n', 2))),
                               ('a sum applied to three integers', Zt, (('c', gid['Z.add'], 3), ('n', 1), ('n', 2), ('n', 3))), ('a function applied to two integers', Zt, (('v', 0, 2), ('n', 1), ('n', 2)))):
        refuses('A3 %s is not a term' % label, lambda: sp.build(codes, hole), GB.BuildError)
    b = sp.builder(Zt); b.push(('n', 3))
    expect(b.done and b.tokens() == [], 'A3 a finished term offers no choice')
    refuses('A3 ... and takes none', lambda: b.push(('n', 3)), GB.BuildError)
    refuses('A3 a finished term cannot be filled', lambda: b.fill(b.term()), GB.BuildError)
    refuses('A3 an unfinished code is no term', lambda: sp.build((('c', gid['Z.add'], 2), ('n', 1)), Zt), GB.BuildError)
    refuses('A3 a code with tokens left over is no term', lambda: sp.build((('n', 1), ('n', 2)), Zt), GB.BuildError)
    refuses('A3 a term of the wrong type is no term (a numeral for a proposition)', lambda: sp.build((('n', 1),), PROP), GB.BuildError)
    refuses('A3 a term built with a hypothesis of another type is refused by `fill`', lambda: sp.builder(Zt).fill(k.mk_fvar(1)), GB.BuildError)
    expect(sp.builder(Zt).fill(sp.build((('n', 4),), Zt)) is None and True, 'A3 a term of the right type can be filled in')
    # ---- A3b. the sorts, and how a statement is begun
    TYPE1, TYPE2 = k.mk_type(1), k.mk_type(2)
    def tk(hole, **kw): return sp.builder(hole).tokens(**kw)
    expect(tk(PROP)[:2] == [('all',), ('imp',)] and ('dep',) not in tk(PROP) and ('dep',) in tk(PROP, rare=True), 'A3 a proposition is begun by a quantifier or an implication (a dependent implication only when asked for)')
    t1 = tk(TYPE1)
    expect(('arr',) in t1 and ('s', 0) in t1 and ('s', 1) not in t1 and ('lam',) not in t1 and ('all',) not in t1, 'A3 a type is begun by a function type, or is the sort of propositions: not Type 1 (%s)' % (t1[:4],))
    bq = sp.builder(PROP); bq.push(('all',)); tq = bq.tokens()
    expect(('arr',) in tq and ('s', 0) in tq and ('s', 1) in tq, 'A3 the domain of a quantifier may also be the sort Type 1 itself (a statement about all types)')
    bq.push(('s', 1))
    expect(bq.hole_type == PROP and not bq.done, 'A3 ... and the body that follows is a proposition')
    refuses('A3 the sort Type 1 is no type outside the domain of a quantifier', lambda: sp.builder(TYPE1).push(('s', 1)), GB.BuildError)
    t2 = tk(TYPE2)
    expect(('s', 1) in t2 and ('s', 0) not in t2 and ('s', 2) not in t2 and sp.builder(TYPE2).push(('s', 1)) is None, 'A3 a hole of type Type 2 is filled by the sort Type 1')
    # ---- A4. the numerals
    N_, Nat_ = w['N'].n, w['Nat'].n
    expect(sp.build((('n', 5),), Zt) != sp.build((('n', 6),), Zt) and sp.build((('n', -3),), Zt) is not None, 'A4 the numerals of the integers include the negative ones')
    refuses('A4 ... those of N do not', lambda: sp.build((('n', -1),), N_), GB.BuildError)
    refuses('A4 ... nor those of Nat', lambda: sp.build((('n', -1),), Nat_), GB.BuildError)
    refuses('A4 a numeral beyond the limit is refused', lambda: sp.build((('n', 10 ** 18 + 1),), Zt), GB.BuildError)
    expect(sp.build((('n', 10 ** 18),), Zt) is not None, 'A4 ... and one at the limit is not')
    refuses('A4 a numeral of Nat is unary: beyond its limit it is refused', lambda: sp.build((('n', GB.MAX_NUMERAL['unary'] + 1),), Nat_), GB.BuildError)
    toks = sp.builder(Zt).tokens()
    expect(('n', 0) in toks and ('n', 13) not in toks and ('n', -1) in toks, 'A4 the small numerals are offered, others are not (but still legal)')
    expect(('c', gid['Z.zero'], 0) not in toks and sp.build((('c', gid['Z.zero'], 0),), Zt) == sp.build((('n', 0),), Zt), 'A4 the constant that spells the numeral 0 is not offered (the numeral says it) and means the same')
    expect(('n', -1) not in sp.builder(N_).tokens(), 'A4 negative numerals are not offered for N')
    expect(sp.encode(sp.build((('n', 12),), Zt), Zt) == (('n', 12),), 'A4 the canonical code of a numeral is a numeral')
    def numerals_goal(vals):                                          # (a statement that has these numerals: a <= v for each, as one conjunction)
        with w.nota.using(w['Z']):
            return w.forall_([w['Z']], lambda a: functools.reduce(lambda p_, q_: w.And(q_, p_), [w['Z.le'](a, w.nota.lit(v)) for v in vals]))
    many = (17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)
    sp_many = GB.TermSpace(eng, [], [], roots=[w.term(numerals_goal(many), 'statement').n])
    offered = [t[1] for t in sp_many.builder(Zt).tokens() if t[0] == 'n']
    extras = [v for v in offered if v not in GB.DEFAULT_NUMERALS]
    expect(offered[:len(GB.DEFAULT_NUMERALS)] == list(GB.DEFAULT_NUMERALS) and extras == list(many[:GB.MAX_GOAL_NUMERALS]), 'A4 the numerals of the goal are offered after the small ones: the %d smallest of those it has (%s)' % (GB.MAX_GOAL_NUMERALS, extras))
    lim = GB.MAX_NUMERAL['unary']                                       # (the numerals of Nat are unary: those beyond the limit are not offered, though the goal has them)
    with w.nota.using(w['Nat']):
        st_nat = w.forall_([w['Nat']], lambda a: w.And(w.Eq(a, w.nota.lit(lim)), w.Eq(a, w.nota.lit(lim + 1))))
    sp_nat = GB.TermSpace(eng, [], [], roots=[w.term(st_nat, 'statement').n])
    offered = [t[1] for t in sp_nat.builder(Nat_).tokens() if t[0] == 'n']
    expect(lim in offered and lim + 1 not in offered and 1000 in offered and all(v >= 0 for v in offered), 'A4 a numeral of the goal beyond the limit of its type is not offered, one at the limit is (%s)' % offered[-4:])
    eng.premises = None
    # ---- A5. over-application: a head applied to more arguments than its type has binders
    i_s, g_s, act_s = fx.step_of('Z.sqrt2_aux', lambda a: a.kind == Kind.REFINE and a.codes[0][0][:2] == ('c', gid['Z.strong_induction']))
    code = act_s.codes[0]; kk = code[0][2]
    sp = eng.space(g_s)
    n_binders = sp.table.heads[gid['Z.strong_induction']].arity
    expect(kk > n_binders, 'A5 the recorded proof applies Z.strong_induction to more arguments (%d) than its type has binders (%d)' % (kk, n_binders))
    t, goals = sp.build_refine(code, g_s.target)
    expect(not goals and k.check_open(t, g_s.target, g_s.hyp_types)[0], 'A5 ... the term is accepted and has the goal\'s type')
    expect(sp.encode(t, g_s.target) == code, 'A5 ... and its canonical code is the code (the arguments beyond the binders are all said)')
    refuses('A5 one more argument than the result can take is refused', lambda: sp.build_refine((code[0][:2] + (kk + 1,),) + code[1:], g_s.target), GB.BuildError)
    refuses('A5 one argument fewer leaves tokens over', lambda: sp.build_refine((code[0][:2] + (kk - 1,),) + code[1:], g_s.target), GB.BuildError)
    refuses('A5 an application to more arguments than the limit allows is refused', lambda: sp.build_refine((code[0][:2] + (n_binders + GB.MAX_ARITY + 1,),) + code[1:], g_s.target), GB.BuildError)
    refuses('A5 the last argument must have its type (a proof of the wrong statement is refused)', lambda: sp.build_refine(code[:-1] + (('v', 1, 0),), g_s.target), GB.BuildError)
    for extra_ in (GB.MAX_ARITY, GB.MAX_ARITY + 1):
        bb = eng.builder(g_s, goals=True)
        try: bb.push(code[0][:2] + (n_binders + extra_,)); r_ = 'accepted'
        except GB.BuildError: r_ = 'refused'
        expect(r_ == ('accepted' if extra_ == GB.MAX_ARITY else 'refused'), 'A5 a head applied to %d more arguments than it has binders is %s (the limit is %d)' % (extra_, r_, GB.MAX_ARITY))
    # a code is at most MAX_CODE tokens: x + x + .. + x with one `-` in it
    spz = GB.TermSpace(eng, [Zt], ['x'])
    def chain(m, wrap=False):
        leaves = [('v', 0, 0)] * m
        if wrap: leaves[-1:] = [('c', gid['Z.neg'], 1), ('v', 0, 0)]
        return (('c', gid['Z.add'], 2),) * (m - 1) + tuple(leaves)
    c_ok, c_over = chain(GB.MAX_CODE // 2, wrap=True), chain(GB.MAX_CODE // 2 + 1)
    expect(len(c_ok) == GB.MAX_CODE and len(c_over) == GB.MAX_CODE + 1, 'A5 (the codes: %d and %d tokens)' % (len(c_ok), len(c_over)))
    expect(spz.build(c_ok, Zt) is not None, 'A5 a code of %d tokens, the most there can be, is built' % GB.MAX_CODE)
    refuses('A5 a code of one token more is refused', lambda: spz.build(c_over, Zt), GB.BuildError)
    # ---- A6. a principle whose conclusion is `P n`: its arguments wait for the motive
    i_f, g_f, act_f = fx.step_of('imo2019_p1_fwd', lambda a: a.kind == Kind.REFINE and a.codes[0][0][:2] == ('c', gid['Z.int_induction']))
    code = act_f.codes[0]; sp = eng.space(g_f)
    b = eng.builder(g_f, goals=True)
    expect(code[0] not in b.tokens() and code[0] in b.tokens(rare=True), 'A6 a principle with conclusion `P n` is offered only when asked for (it fits every proposition once the motive is said)')
    b.push(code[0])
    pi = b.expose_pi(b.hole_type)
    expect(pi is not None and pi[1] == k.mk_type(0) and ('lam',) in b.tokens(), 'A6 after the head, the next hole is the motive, a predicate (%s)' % (b.tokens()[:3],))
    t, goals = sp.build_refine(code, g_f.target)
    expect(not goals and k.check_open(t, g_f.target, g_f.hyp_types)[0] and sp.encode(t, g_f.target) == code, 'A6 the recorded code (a motive and the proofs of the three premises) builds a term the kernel accepts, and is its canonical code')
    b = eng.builder(g_f, goals=True); b.push(code[0]); app = b.stack[0]; j0 = app.j; used = 1
    while app.j == j0: b.push(code[used]); used += 1
    expect(not b.done and b.stack[0] is app and app.j == j0 + 1, 'A6 the motive is the first argument said (%d tokens), then come the premises' % used)
    for _ in range(3): b.push(('g',))
    expect(b.done, 'A6 ... and after the three premises the term is finished: the argument n is not said, the goal determines it once the motive is known')
    t3, goals3 = sp.close_goals(b)
    expect(len(goals3) == 3 and goals3 == [g_f.hyp_types[6], g_f.hyp_types[7], g_f.hyp_types[8]], 'A6 the three goals are the base case and the two steps, with the motive applied and reduced (the hypotheses a6 a7 a8 of the goal say the same)')
    expect(eng.probe(g_f, act(Kind.REFINE, code[:used] + (('g',),) * 3), A.NO_MENU) == [], 'A6 REFINE with these goals closes the goal: the goals are hypotheses already')
    k_ctx = len(g_f.hyp_types)
    wrong_motive = code[:1] + (('lam',), ('c', gid['Eq'], 3), ('c', gid['Z'], 0), ('v', k_ctx, 0), ('v', k_ctx, 0)) + (('g',),) * 3
    refuses('A6 a motive that does not fit the goal leaves the argument n undetermined: the code is unfinished', lambda: sp.build_refine(wrong_motive, g_f.target), GB.BuildError)
    refuses('A6 ... and with n said the term has the wrong type', lambda: sp.build_refine(wrong_motive + (('v', 9, 0),), g_f.target), GB.BuildError)
    b = eng.builder(g_f, goals=True); b.push(code[0]); b.push(('lam',))
    expect(('g',) not in b.tokens(), 'A6 no goal is offered under a binder of the term')
    refuses('A6 ... and none is accepted', lambda: b.push(('g',)), GB.BuildError)
    expect(('g',) in eng.builder(g_f, goals=True).tokens() and ('g',) not in eng.builder(g_f, goals=False).tokens(), 'A6 goals are offered only when the builder allows them')
    # ---- A6b. a term leaves at most MAX_GOALS goals
    def conj_chain(m):                                                  # P /\ (P /\ (.. /\ P)) with m conjuncts, P = 0 <= 0
        with w.nota.using(w['Z']):
            P = w['Z.le'](w.nota.lit(0), w.nota.lit(0)); st = P
            for _ in range(m - 1): st = w.And(P, st)
        return w.term(st, 'statement').n
    sp0 = GB.TermSpace(eng, [], [])
    for m_ in (GB.MAX_GOALS, GB.MAX_GOALS + 1):
        code_ = (('c', gid['conj'], 4), ('g',)) * (m_ - 1) + (('g',),)
        try: _, gs_ = sp0.build_refine(code_, conj_chain(m_)); r_ = len(gs_)
        except GB.BuildError: r_ = None
        expect(r_ == m_ if m_ == GB.MAX_GOALS else r_ is None, 'A6 a term that leaves %d goals is %s (the limit is %d)' % (m_, 'built' if r_ else 'refused', GB.MAX_GOALS))
    # ---- A7. premises limit the library
    eng.premises = gid['Z.int_induction']
    sp = GB.TermSpace(eng, g_f.hyp_types, g_f.hyp_names)
    refuses('A7 a global beyond the premises is refused', lambda: sp.build_refine(code, g_f.target), GB.BuildError)
    bz = sp.builder(PROP)
    expect(all(not (t[0] == 'c' and t[1] >= gid['Z.int_induction']) for t in bz.tokens(rare=True)), 'A7 ... and not offered')
    eng.premises = None
    # ---- A8. codes changed at random are refused or built, never anything else
    rng = random.Random(40); n_mut = collections.Counter()
    sources = []
    for nm in ('Z.sqrt2_aux', 'imo2019_p1_fwd', 'imo1959_p1', 'Z.prime_dvd_sq', 'N.beq_eq'):
        def hook(i, p, g, a, menu, nm=nm):
            if a.kind in (Kind.REFINE, Kind.CLAIM) and a.codes: sources.append((nm, g, a.kind, a.codes[0]))
        fx.walk(nm, hook)
    for nm, g, kind, code in sources[:40]:
        eng.premises = fx.recs[nm].gid
        sp = GB.TermSpace(eng, g.hyp_types, g.hyp_names, roots=[g.target])
        expected = sp.prop if kind == Kind.CLAIM else g.target
        for j in range(8):
            c2 = list(code); m = rng.randrange(4); pos = rng.randrange(len(c2))
            if m == 0: del c2[pos]
            elif m == 1: c2.insert(pos, rng.choice(code))
            elif m == 2: c2[pos] = rng.choice(code)
            else:
                tk = c2[pos]
                c2[pos] = tk[:-1] + (tk[-1] + rng.choice((-1, 1, 5)),) if len(tk) > 1 and isinstance(tk[-1], int) else tk
            try:
                t, goals = sp.build_refine(tuple(c2), expected)
                n_mut['built'] += 1
                if not k.check_open(t, expected if not goals else k.mk_pi(goals[0], expected) if len(goals) == 1 else t and expected, g.hyp_types)[0] and not goals: n_mut['ill-typed'] += 1
            except GB.BuildError: n_mut['refused'] += 1
            except Exception as e: n_mut['crashed'] += 1; n_mut['crash example'] = '%s: %s' % (type(e).__name__, str(e)[:100])
    eng.premises = None
    expect(n_mut['crashed'] == 0, 'A8 a code changed at random is refused or built, never a crash (%s)' % dict(n_mut))
    expect(n_mut['refused'] >= 100 and n_mut['ill-typed'] == 0, 'A8 most changed codes are refused, and what is built has its type (%s)' % dict(n_mut))
    # ---- A9. every statement of the library is said, and built back as itself (a problem statement is said like this)
    tot = collections.Counter(); bad = []; used = collections.Counter()
    for g_ in range(k.num_globals()):
        name, ty = k.global_info(g_)[0], k.global_info(g_)[1]
        if name.startswith('__') or k.reserved_kind(g_): continue
        ok_, sort_ = k.infer_open(ty, [])
        if not ok_ or sort_ != PROP: continue
        eng.premises = g_; tot['statements'] += 1
        try: code_ = GB.TermSpace(eng, [], []).encode(ty, PROP)
        except GB.BuildError: tot['unsayable'] += 1; bad.append(name); continue
        used.update(t[0] for t in code_); tot['tokens'] += len(code_)
        if GB.TermSpace(eng, [], []).build(code_, PROP) != ty: tot['different'] += 1; bad.append(name)
    eng.premises = None
    expect(tot['statements'] >= 400 and tot['tokens'] >= 6500 and tot['different'] == 0 and bad == ['funext'], 'A9 every statement of the library is said and built back as itself, but `funext`, which quantifies over a family of types (%s; %s)' % (dict(tot), bad[:3]))
    expect(all(used[t] > 0 for t in ('v', 'c', 'n', 'lam', 'all', 'imp', 'arr', 's', 'dep')), 'A9 ... with every kind of token among them (%s)' % dict(used))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. the actions
def goals_of(eng, ep): return [eng.goal_of(ep.p, i) for i in range(ep.n_goals)]


def ep_of(c, stmt, premises=None):
    c['fx'].eng.premises = premises
    return G.Episode(c['fx'].eng, stmt)


def refused(c, ep, a, what):
    """the action is refused (ActionError) and changes nothing"""
    fx = c['fx']; k, w = fx.k, fx.w
    ep.menu(); g = ep.goal(); before = (state(k, w), ep.n_goals, g.key)
    try: ep.step(a)
    except G.ActionError: pass
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:100]))
    else: return expect(False, what + ' (it was accepted)')
    return expect(before == (state(k, w), ep.n_goals, ep.goal().key), what + ' -- and nothing changed')


def do(c, ep, a, what):
    """carry out the action: the goals it leaves in place of the first (after checking that the probe predicted exactly them); None if it is refused"""
    eng = c['fx'].eng; g = ep.goal(); m = ep.menu()
    pred = eng.probe(g, a, m)
    if not expect(pred is not None, what + ' (the probe refuses it)'): return None
    before = ep.n_goals; ep.step(a); now = goals_of(eng, ep)
    expect([x.key for x in now[:len(pred)]] == [x.key for x in pred] and ep.n_goals == before - 1 + len(pred), what + ': the live step leaves the goals the probe predicted')
    return now[:len(pred)]


def in_goal(c, g, fn):
    """fn(fvars...) -> a term, built in the context of the goal g; the kernel node"""
    fx = c['fx']
    with fx.eng.in_context(g.hyp_types, g.hyp_names), fx.w.nota.using(fx.w['Z']):
        return fn(*[fx.w.T(fx.k.mk_fvar(i)) for i in range(min(g.n_hyps, fn.__code__.co_argcount))]).n


def statements(w):
    Z = w['Z']; F = w.forall_; Eq = w.Eq; zle = w['Z.le']; Ex = w['Ex']; Nat = w['Nat']
    out = {}
    with w.nota.using(Z):
        lit = w.nota.lit
        out['amgm'] = F([Z, Z], lambda a, b: zle(2 * a * b, a * a + b * b))
        out['trans'] = F([Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h1: zle(y, z)], lambda x, y, z, h1, h2: zle(x, z))
        out['trans_part'] = F([Z, Z, Z, lambda a, b, c_: zle(a, b)], lambda a, b, c_, h: zle(a, c_))                            # (false: only the goals it leaves matter)
        out['ex49'] = Ex(Z, w.lam(Z, lambda x: Eq(x * x, lit(49))))
        out['conj'] = F([Z], lambda a: w.And(zle(lit(0), a * a), zle(a * a, a * a + lit(1))))
        out['sq'] = F([Z], lambda a: zle(w['Z.zero'], a * a))                                                                    # (0 as the lemmas spell it)
        out['rwg'] = F([Z, Z, Z], lambda a, b, c_: Eq((a + b) * c_, (b + a) * c_))
        out['rwa'] = F([Z, Z, Z, lambda a, b, c_: Eq(a + b, c_)], lambda a, b, c_, h: Eq(b + a, c_))
        out['ind'] = F([Z, lambda n: zle(lit(1), n)], lambda n, h: zle(lit(1), n * n))
        out['ind2'] = F([Z, Z, lambda n, m: zle(lit(1), n), lambda n, m, h1: zle(n, m)], lambda n, m, h1, h2: zle(n * n, m * m))
        out['ind_free'] = F([Z, Z, lambda n, m: zle(lit(1), n)], lambda n, m, h: zle(lit(0), m * m))
        out['inj'] = F([Nat, Nat, lambda n, m: Eq(w['Succ'](n), w['Succ'](m))], lambda n, m, h: Eq(n, m))
        out['injz'] = F([Z, Z, Z, lambda a, b, c_: Eq(c_ + a, c_ + b)], lambda a, b, c_, h: Eq(a, b))
        out['lc'] = F([Z, Z, lambda a, b: Eq(a + b, lit(5)), lambda a, b, h1: Eq(a - b, lit(1))], lambda a, b, h1, h2: Eq(2 * a, lit(6)))
        out['lc_le'] = F([Z, Z, lambda a, b: Eq(a + b, lit(5)), lambda a, b, h1: Eq(a - b, lit(1))], lambda a, b, h1, h2: zle(a, lit(6)))
        out['mul_one'] = F([Z], lambda a: Eq(a * lit(1), a))                                                                       # (the numeral 1 is `Z.of_N (Npos XH)`, the lemma's is `Z.one`: equal by computation only)
        out['ind_motive'] = F([Z], lambda a: zle(w['Z.zero'], a * a + a))
        out['closed'] = F([lambda: zle(lit(0), lit(1))], lambda h: zle(lit(0), lit(1) + lit(1)))                                    # (closed hypothesis, closed target)
        out['rwdir'] = F([Z], lambda a: Eq(a * w['Z.zero'], w['Z.zero']))                                                          # (`Z.mul_zero a` closes it from left to right; from right to left it does not)
        out['inj2'] = F([Nat, Nat, lambda n, m: Eq(w['Succ'](n), w['Succ'](m))], lambda n, m, h: w.And(Eq(n, m), Eq(n, m)))      # (the fact n = m does not close the goal)
        out['ind_renum'] = F([Z, lambda x: zle(lit(0), x), Z, lambda x, h, c_: zle(c_, x)], lambda x, h, c_, hb: zle(lit(0), x * x))   # (the starting point c comes after the hypothesis 0 <= x that is generalised)
        out['chain4'] = F([Z, Z, Z, Z, lambda x, y, z, v: zle(x, y), lambda x, y, z, v, h1: zle(y, z), lambda x, y, z, v, h1, h2: zle(z, v)], lambda x, y, z, v, h1, h2, h3: zle(x, v))
    return out


def sec_B(c):
    fx = c['fx']; w, k, eng = fx.w, fx.k, fx.eng
    gid = fx.gid; S = statements(w); Z = w['Z']
    with w.nota.using(Z): lit = w.nota.lit
    # ---- B1. CLAIM
    ep = ep_of(c, S['amgm']); g = ep.goal()
    claim = in_goal(c, g, lambda a, b: w['Z.sq_nonneg'](a - b)); claim_ty = k.infer_open(claim, g.hyp_types)[1]
    code = eng.space(g).encode(claim_ty, eng.space(g).prop)
    out = do(c, ep, act(Kind.CLAIM, code), 'B1 CLAIM 0 <= (a - b)^2')
    expect(out is not None and len(out) == 2 and out[0].target == claim_ty and out[0].hyp_types == g.hyp_types, 'B1 ... the first goal is the claim, in the context of the goal')
    expect(out is not None and out[1].target == g.target and out[1].hyp_types == g.hyp_types + [claim_ty], 'B1 ... the second is the goal with the claim as a new hypothesis')
    refused(c, ep, act(Kind.CLAIM, code + (('n', 1),)), 'B1 a claim with tokens left over')
    refused(c, ep, act(Kind.CLAIM, code[:-1]), 'B1 an unfinished claim')
    refused(c, ep, act(Kind.CLAIM, (('v', 0, 0),)), 'B1 a claim that is no proposition (an integer)')
    refused(c, ep, act(Kind.CLAIM, (('n', 1),)), 'B1 a numeral is no proposition')
    refused(c, ep, act(Kind.CLAIM), 'B1 CLAIM without a code')
    refused(c, ep, act(Kind.CLAIM, code, code), 'B1 CLAIM with two codes')
    refused(c, ep, act(Kind.CLAIM, code, hyps=(0,)), 'B1 CLAIM with a hypothesis')
    refused(c, ep, act(Kind.CLAIM, code, arg=1), 'B1 CLAIM with an argument')
    refused(c, ep, Action(Kind.CLAIM, 0, (1,), (code,)), 'B1 CLAIM with a term of the pool')
    refused(c, ep, act(Kind.CLAIM, ()), 'B1 CLAIM with the empty code')
    n2 = do(c, ep, act(Kind.REFINE, (('c', gid['Z.sq_nonneg'], 1),)), 'B1 REFINE closes the claim with `Z.sq_nonneg` (its argument is determined by the claim)')
    expect(n2 is not None and n2 == [] and ep.n_goals == 1, 'B1 ... the claim is proved and the goal with the claim is left')
    refused(c, ep, act(Kind.CLAIM, code), 'B1 the claim is a hypothesis now: claiming it again is refused')
    ep.menu()
    nl = next(a for a in ep.menu().actions if a.kind == Kind.NLINARITH)
    ep.step(nl)
    expect(ep.done and ep.finish() is not None, 'B1 the proof (claim, refine, nlinarith) is finished and the kernel accepts it')
    # ---- B2. REFINE
    ep = ep_of(c, S['ex49'])
    out = do(c, ep, act(Kind.REFINE, (('c', gid['ex_intro'], 4), ('n', 7), ('g',))), 'B2 REFINE ex_intro 7 ?_')
    expect(out == [] and ep.done and ep.finish() is not None, 'B2 ... 7 * 7 = 49 is closed by computation: the proof is finished')
    ep = ep_of(c, S['ex49'])
    out = do(c, ep, act(Kind.REFINE, (('c', gid['ex_intro'], 4), ('n', 8), ('g',))), 'B2 REFINE ex_intro 8 ?_')
    expect(out is not None and len(out) == 1 and out[0].target == k.mk_app(k.mk_app(k.mk_app(k.mk_const(gid['Eq']), w['Z'].n), in_goal(c, out[0], lambda: lit(8) * lit(8))), in_goal(c, out[0], lambda: lit(49))),
           'B2 ... leaves the goal 8 * 8 = 49')
    expect(eng.probe(ep.goal(), Action(Kind.RING, 0), ep.menu()) is None, 'B2 ... which ring does not prove')
    refused(c, ep_of(c, S['ex49']), act(Kind.REFINE, (('c', gid['ex_intro'], 4), ('g',), ('g',))), 'B2 a witness cannot be left as a goal (it is no proposition)')
    refused(c, ep_of(c, S['ex49']), act(Kind.REFINE, (('c', gid['conj'], 4), ('g',), ('g',))), 'B2 a conjunction for an existential goal')
    ep = ep_of(c, S['sq'])
    out = do(c, ep, act(Kind.REFINE, (('c', gid['Z.sq_nonneg'], 1),)), 'B2 REFINE with a term that proves the goal outright (the argument is determined by the goal)')
    expect(out == [] and ep.done and ep.finish() is not None, 'B2 ... closes it')
    ep = ep_of(c, S['sq'])
    refused(c, ep, act(Kind.REFINE, (('c', gid['Z.sq_nonneg'], 1), ('v', 0, 0))), 'B2 REFINE with an argument the goal determines (a term has one canonical code: what the goal determines is not said)')
    refused(c, ep, act(Kind.REFINE, (('c', gid['Z.sq_nonneg'], 0),)), 'B2 REFINE with a lemma not applied to its argument: the whole lemma is not a proof of the goal')
    ep = ep_of(c, S['conj'])
    out = do(c, ep, act(Kind.REFINE, (('c', gid['conj'], 4), ('g',), ('g',))), 'B2 REFINE conj ?_ ?_')
    expect(out is not None and len(out) == 2 and out[0].target == in_goal(c, out[0], lambda a: w['Z.le'](lit(0), a * a)) and out[1].target == in_goal(c, out[1], lambda a: w['Z.le'](a * a, a * a + lit(1))),
           'B2 ... leaves the two conjuncts, in order')
    refused(c, ep, act(Kind.REFINE, (('g',),)), 'B2 REFINE ?_ (the whole goal) makes no progress')
    refused(c, ep, act(Kind.REFINE, (('c', gid['conj'], 4), ('g',), ('g',)), arg=1), 'B2 REFINE with an argument')
    refused(c, ep, act(Kind.REFINE, (('c', gid['conj'], 4), ('g',), ('g',)), hyps=(0,)), 'B2 REFINE with a hypothesis')
    refused(c, ep, Action(Kind.REFINE, 0, (0,), ((('g',),),)), 'B2 REFINE with a term of the pool')
    refused(c, ep, act(Kind.REFINE), 'B2 REFINE without a code')
    refused(c, ep, act(Kind.REFINE, ()), 'B2 REFINE with the empty code')
    refused(c, ep, act(Kind.REFINE, (('c', gid['conj'], 4), ('g',), ('g',)), (('g',),)), 'B2 REFINE with two codes')
    # ---- B3. APPLY_WITH
    ep = ep_of(c, S['trans'])
    out = do(c, ep, act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), hyp(1)), 'B3 apply Z.le_trans y')
    expect(out == [] and ep.done and ep.finish() is not None, 'B3 ... both premises are hypotheses: the proof is finished')
    ep = ep_of(c, S['trans_part']); g = ep.goal()
    out = do(c, ep, act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), hyp(1)), 'B3 apply Z.le_trans b (one premise is a hypothesis)')
    expect(out is not None and len(out) == 1 and out[0].target == in_goal(c, g, lambda a, b, c_, h: w['Z.le'](b, c_)) and out[0].hyp_types == g.hyp_types, 'B3 ... the other premise b <= c is the goal left')
    for what, a in (('too few arguments (the intermediate integer is not said)', act(Kind.APPLY_WITH, ref(gid['Z.le_trans']))),
                    ('too many arguments', act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), hyp(1), hyp(1))),
                    ('a proof where an integer is wanted', act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), hyp(3))),
                    ('a lemma that does not fit the goal', act(Kind.APPLY_WITH, ref(gid['Z.mul_comm']))),
                    ('a source of two tokens', act(Kind.APPLY_WITH, (('c', gid['Z.le_trans'], 0), ('n', 1)), hyp(1))),
                    ('a source that is applied', act(Kind.APPLY_WITH, (('c', gid['Z.le_trans'], 2),), hyp(1))),
                    ('a source that is no global (beyond the library)', act(Kind.APPLY_WITH, ref(10 ** 6), hyp(1))),
                    ('a source that is no hypothesis', act(Kind.APPLY_WITH, hyp(99), hyp(1))),
                    ('a source that is a variable of the wrong type', act(Kind.APPLY_WITH, hyp(0), hyp(1))),
                    ('an unknown mode', act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), hyp(1), arg=2)),
                    ('a negative mode', act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), hyp(1), arg=-1)),
                    ('no source', act(Kind.APPLY_WITH)),
                    ('hypotheses', act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), hyp(1), hyps=(0,)))):
        refused(c, ep_of(c, S['trans_part']), a, 'B3 mode 0: ' + what)
    # the unifier mode: leading arguments positional, the unifier finds the others (also up to computation)
    ep = ep_of(c, S['mul_one']); g = ep.goal()
    out = do(c, ep, act(Kind.APPLY_WITH, ref(gid['Z.mul_one']), arg=1), 'B3 mode 1: apply Z.mul_one to a * 1 = a (equal up to computation)')
    expect(out == [] and ep.done and ep.finish() is not None, 'B3 ... the unifier finds a and the goal is closed')
    ep = ep_of(c, S['mul_one'])
    out = do(c, ep, act(Kind.APPLY_WITH, ref(gid['Z.mul_one']), hyp(0)), 'B3 mode 0 with the argument said')
    expect(out == [] and ep.done, 'B3 ... closes it as well')
    ep = ep_of(c, S['trans_part'])
    refused(c, ep, act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), arg=1), 'B3 mode 1: apply Z.le_trans leaves the intermediate integer to nothing: the two goals share it')
    out = do(c, ep, act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), hyp(0), hyp(1), arg=1), 'B3 mode 1: apply Z.le_trans x y (the LEADING arguments are said, in the order of the binders)')
    expect(out is not None and len(out) == 1 and out[0].target == in_goal(c, out[0], lambda a, b, c_, h: w['Z.le'](b, c_)), 'B3 ... the same goal as in mode 0: b <= c')
    refused(c, ep_of(c, S['trans_part']), act(Kind.APPLY_WITH, ref(gid['Z.le_trans']), hyp(1), arg=1), 'B3 mode 1: apply Z.le_trans b says x = b, which the goal a <= c contradicts')
    ep = ep_of(c, S['ind_motive']); g = ep.goal()
    motive = (('lam',), ('c', gid['Z.le'], 2), ('n', 0), ('c', gid['Z.add'], 2), ('c', gid['Z.mul'], 2), ('v', g.n_hyps, 0), ('v', g.n_hyps, 0), ('v', g.n_hyps, 0))
    out = do(c, ep, act(Kind.APPLY_WITH, ref(gid['Z.int_induction']), motive, arg=1), 'B3 mode 1: apply Z.int_induction with the motive fun n => 0 <= n * n + n')
    expect(out is not None and len(out) == 3, 'B3 ... the unifier finds n = a, the base case and the two steps are left (%s)' % (None if out is None else len(out)))
    if out and len(out) == 3: expect(out[0].target == in_goal(c, out[0], lambda: w['Z.le'](w['Z.zero'], w['Z.zero'] * w['Z.zero'] + w['Z.zero'])) and out[1].n_hyps == 3 and out[2].n_hyps == 3, 'B3 ... the first is the base case 0 <= 0 * 0 + 0, the others are the steps')
    ep = ep_of(c, S['ind_motive'])
    out2 = do(c, ep, act(Kind.APPLY_WITH, ref(gid['Z.int_induction']), motive), 'B3 mode 0: the same with the goal determining the rest')
    expect(out2 is not None and out is not None and [x.key for x in out2] == [x.key for x in out], 'B3 ... gives the same three goals')
    # ---- B4. REWRITE_WITH
    ep = ep_of(c, S['rwg'])
    out = do(c, ep, act(Kind.REWRITE_WITH, ref(gid['Z.add_comm']), hyp(0), hyp(1)), 'B4 rw Z.add_comm a b')
    expect(out == [] and ep.done and ep.finish() is not None, 'B4 ... turns a + b into b + a: both sides are equal and the goal is closed')
    ep = ep_of(c, S['rwg'])
    out = do(c, ep, act(Kind.REWRITE_WITH, ref(gid['Z.add_comm']), hyp(0), hyp(1), arg=1), 'B4 rw <- Z.add_comm a b')
    expect(out == [] and ep.done, 'B4 ... right to left: b + a into a + b, the goal is closed')
    ep = ep_of(c, S['rwg'])
    out = do(c, ep, act(Kind.REWRITE_WITH, ref(gid['Z.add_comm'])), 'B4 rw Z.add_comm (the first instance is found by matching)')
    expect(out == [] and ep.done, 'B4 ... closes the goal too')
    ep = ep_of(c, S['rwdir'])
    out = do(c, ep, act(Kind.REWRITE_WITH, ref(gid['Z.mul_zero']), hyp(0)), 'B4 rw Z.mul_zero a')
    expect(out == [] and ep.done, 'B4 ... a * 0 into 0: the goal is closed')
    ep = ep_of(c, S['rwdir'])
    out = do(c, ep, act(Kind.REWRITE_WITH, ref(gid['Z.mul_zero']), hyp(0), arg=1), 'B4 rw <- Z.mul_zero a')
    expect(out is not None and len(out) == 1, 'B4 ... from right to left every 0 becomes a * 0: a goal is left (the direction matters)')
    for what, a in (('a pattern that does not occur', act(Kind.REWRITE_WITH, ref(gid['Z.mul_comm']), hyp(0), hyp(0))),
                    ('a lemma that is no equation', act(Kind.REWRITE_WITH, ref(gid['Z.le_trans']))),
                    ('too many arguments', act(Kind.REWRITE_WITH, ref(gid['Z.add_comm']), hyp(0), hyp(1), hyp(2))),
                    ('an argument of the wrong type', act(Kind.REWRITE_WITH, ref(gid['Z.add_comm']), hyp(0), (('c', gid['Z.le'], 0),))),
                    ('an unknown direction', act(Kind.REWRITE_WITH, ref(gid['Z.add_comm']), arg=2)),
                    ('no source', act(Kind.REWRITE_WITH)),
                    ('hypotheses', act(Kind.REWRITE_WITH, ref(gid['Z.add_comm']), hyps=(0,)))):
        refused(c, ep_of(c, S['rwg']), a, 'B4 ' + what)
    # ---- B5. REWRITE_AT
    ep = ep_of(c, S['rwa']); g = ep.goal()
    out = do(c, ep, act(Kind.REWRITE_AT, ref(gid['Z.add_comm']), hyps=(3,)), 'B5 rw Z.add_comm at h')
    expect(out == [] and ep.done and ep.finish() is not None, 'B5 ... h becomes b + a = c, which is the goal: the proof is finished')
    ep = ep_of(c, S['rwa'])
    out = do(c, ep, act(Kind.REWRITE_AT, ref(gid['Z.add_comm']), hyp(0), hyp(1), hyps=(3,)), 'B5 rw Z.add_comm a b at h')
    expect(out == [] and ep.done, 'B5 ... with the arguments said')
    refused(c, ep_of(c, S['rwa']), act(Kind.REWRITE_AT, ref(gid['Z.add_comm']), hyp(0), hyp(1), hyps=(3,), arg=1), 'B5 from right to left with the arguments said: b + a does not occur in the hypothesis')
    for what, a in (('a hypothesis that does not exist', act(Kind.REWRITE_AT, ref(gid['Z.add_comm']), hyps=(9,))),
                    ('a negative position', act(Kind.REWRITE_AT, ref(gid['Z.add_comm']), hyps=(-1,))),
                    ('a variable instead of a hypothesis', act(Kind.REWRITE_AT, ref(gid['Z.add_comm']), hyps=(0,))),
                    ('no hypothesis', act(Kind.REWRITE_AT, ref(gid['Z.add_comm']))),
                    ('two hypotheses', act(Kind.REWRITE_AT, ref(gid['Z.add_comm']), hyps=(3, 3))),
                    ('a hypothesis rewritten by itself', act(Kind.REWRITE_AT, hyp(3), hyps=(3,))),
                    ('a pattern that does not occur', act(Kind.REWRITE_AT, ref(gid['Z.mul_comm']), hyps=(3,))),
                    ('an unknown direction', act(Kind.REWRITE_AT, ref(gid['Z.add_comm']), hyps=(3,), arg=2))):
        refused(c, ep_of(c, S['rwa']), a, 'B5 ' + what)
    # ---- B6. FORWARD
    ep = ep_of(c, S['trans']); g = ep.goal()
    out = do(c, ep, act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), hyp(3), hyp(4)), 'B6 have := Z.le_trans _ _ _ h1 h2 (the integers are determined by the hypotheses)')
    expect(out == [] and ep.done and ep.finish() is not None, 'B6 ... x <= z is the goal: finished')
    ep = ep_of(c, S['sq']); g = ep.goal()
    out = do(c, ep, act(Kind.FORWARD, ref(gid['Z.mul_comm']), (('v', 0, 0),), (('v', 0, 0),)), 'B6 have := Z.mul_comm a a (the arguments said)')
    expect(out is not None and len(out) == 1 and out[0].hyp_types == g.hyp_types + [in_goal(c, g, lambda a: w.Eq(a * a, a * a))], 'B6 ... the new hypothesis is the instance a * a = a * a')
    ep = ep_of(c, S['trans'])
    for what, a in (('an argument that nothing determines', act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), (), ())),
                    ('a hypothesis of the wrong type for the premise', act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), hyp(4), hyp(3))),
                    ('more slots than binders', act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), hyp(3), hyp(4), ())),
                    ('a term of the wrong type for an integer', act(Kind.FORWARD, ref(gid['Z.le_trans']), hyp(3), (), ())),
                    ('no source', act(Kind.FORWARD)),
                    ('a source that does not exist', act(Kind.FORWARD, ref(10 ** 6))),
                    ('a hypothesis that does not exist as a premise', act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), hyp(99), hyp(4))),
                    ('a hypothesis at the end of the context as a premise', act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), hyp(ep.goal().n_hyps), hyp(4))),
                    ('an integer that the hypotheses force to be another', act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (('n', 5),), (), hyp(3), hyp(4))),
                    ('hypotheses', act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), hyp(3), hyp(4), hyps=(0,))),
                    ('an argument that is not a code', Action(Kind.FORWARD, 0, (), (ref(gid['Z.le_trans']), 5), ()))):
        refused(c, ep, a, 'B6 ' + what)
    ep = ep_of(c, S['trans_part'])                                  # a fact that is there already
    do(c, ep, act(Kind.FORWARD, ref(gid['Z.le_refl']), hyp(0)), 'B6 have := Z.le_refl a')
    refused(c, ep, act(Kind.FORWARD, ref(gid['Z.le_refl']), hyp(0)), 'B6 the same fact again is refused')
    # ---- B7. INDUCT_WITH
    ep = ep_of(c, S['ind'])
    out = do(c, ep, act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),), hyps=(0,)), 'B7 induction on n by Z.induction_from 1')
    expect(out is not None and len(out) == 2 and out[0].n_hyps == 0 and out[1].n_hyps == 3, 'B7 ... two goals: the base case (no hypothesis) and the step (the variable, the bound and the induction hypothesis)')
    if out and len(out) == 2 and out[1].n_hyps == 3:
        one = w['Z.one'].n
        expect(out[0].target == in_goal(c, out[0], lambda: w['Z.le'](lit(1), lit(1) * lit(1))), 'B7 ... the base case is 1 <= 1 * 1')
        expect(out[1].target == in_goal(c, out[1], lambda x, h, ih: w['Z.le'](lit(1), (x + w['Z.one']) * (x + w['Z.one']))) and out[1].hyp_types[1] == in_goal(c, out[1], lambda x: w['Z.le'](lit(1), x))
               and out[1].hyp_types[2] == in_goal(c, out[1], lambda x, h: w['Z.le'](lit(1), x * x)), 'B7 ... the step goes from the induction hypothesis 1 <= x * x to 1 <= (x + 1) * (x + 1)')
    ep = ep_of(c, S['ind2'])
    out = do(c, ep, act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),), hyps=(0,)), 'B7 induction on n with a hypothesis n <= m that is not the bound')
    expect(out is not None and len(out) == 2 and out[0].n_hyps == 2 and out[1].n_hyps == 5 and k.view(out[1].hyp_types[3])[0] == 'pi', 'B7 ... the other hypothesis is generalised: the base case has m and 1 <= m, the step m, x, 1 <= x, the induction hypothesis x <= m -> x * x <= m * m and x + 1 <= m')
    if out and len(out) == 2 and out[1].n_hyps == 5:
        ih = k.mk_pi(in_goal(c, out[1], lambda m, x: w['Z.le'](x, m)), in_goal(c, out[1], lambda m, x: w['Z.le'](x * x, m * m)))
        expect(out[1].hyp_types[3] == ih and out[1].hyp_types[4] == in_goal(c, out[1], lambda m, x, h, i_: w['Z.le'](x + w['Z.one'], m)), 'B7 ... stated as an implication with the generalised hypothesis in front, and the hypothesis is back for x + 1')
    for what, a in (('a lemma that is no principle', act(Kind.INDUCT_WITH, ref(gid['Z.le_trans']), (('n', 1),), hyps=(0,))),
                    ('a principle about another type', act(Kind.INDUCT_WITH, ref(gid['N.ind']), hyps=(0,))),
                    ('a proof as the variable', act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),), hyps=(2,))),
                    ('no starting point', act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), hyps=(0,))),
                    ('too many arguments', act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),), (('n', 1),), hyps=(0,))),
                    ('a hypothesis that does not exist', act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),), hyps=(9,))),
                    ('no variable', act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),))),
                    ('two variables', act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),), hyps=(0, 1))),
                    ('a starting point that is a proof', act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), hyp(2), hyps=(0,))),
                    ('a source that is a hypothesis', act(Kind.INDUCT_WITH, hyp(2), (('n', 1),), hyps=(0,))),
                    ('an argument', act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),), hyps=(0,), arg=1))):
        refused(c, ep_of(c, S['ind2']), a, 'B7 ' + what)
    ep = ep_of(c, S['ind_renum']); g = ep.goal(); m = ep.menu()
    by_start = {a.codes[1]: a for a in m.actions if a.kind == Kind.INDUCT_WITH and a.codes[0] == ref(gid['Z.induction_from']) and a.hyps == (0,)}
    expect((('v', 2, 0),) in by_start and len(by_start) == 2, 'B7 the starting points that the hypotheses show: c (which comes after the hypothesis 0 <= x) and 0 (%s)' % (sorted(by_start),))
    if (('v', 2, 0),) in by_start:
        out = do(c, ep, by_start[(('v', 2, 0),)], 'B7 induction from c, generalising 0 <= x (which comes before c in the context: the starting point is renumbered)')
        expect(out is not None and len(out) == 2 and out[0].n_hyps == 2 and out[0].hyp_types[0] == w['Z'].n and out[0].target == in_goal(c, out[0], lambda c_, h: w['Z.le'](lit(0), c_ * c_)), 'B7 ... the base case is for c itself: 0 <= c * c with 0 <= c')
        expect(out is not None and len(out) == 2 and out[1].n_hyps == 5, 'B7 ... the step has c, x, c <= x, and the induction hypothesis 0 <= x -> 0 <= x * x with 0 <= x + 1 (%s)' % (None if out is None else out[1].n_hyps,))
    ep = ep_of(c, S['ind_free'])
    refused(c, ep, act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),), hyps=(0,)), 'B7 induction on n when only the bound 1 <= n mentions it (the motive would not mention n)')
    expect(not any(a.kind == Kind.INDUCT_WITH and a.hyps == (0,) and a.codes[0] == ref(gid['Z.induction_from']) for a in ep.menu().actions), 'B7 ... and the menu does not list it')
    # ---- B8. INJECT
    ep = ep_of(c, S['inj']); g = ep.goal()
    out = do(c, ep, Action(Kind.INJECT, 2), 'B8 injection of S n = S m')
    expect(out == [] and ep.done and ep.finish() is not None, 'B8 ... gives n = m, which is the goal')
    ep = ep_of(c, S['injz']); g = ep.goal()
    out = do(c, ep, Action(Kind.INJECT, 3), 'B8 injection of c + a = c + b (the left cancellation is the one that applies)')
    expect(out == [] and ep.done and ep.finish() is not None, 'B8 ... gives a = b')
    ep = ep_of(c, S['inj'])
    for what, a in (('a variable', Action(Kind.INJECT, 0)), ('a hypothesis that does not exist', Action(Kind.INJECT, 9)), ('a negative position', Action(Kind.INJECT, -1)),
                    ('hypotheses', Action(Kind.INJECT, 2, (), (), (0,))), ('codes', Action(Kind.INJECT, 2, (), ((('n', 1),),))), ('terms of the pool', Action(Kind.INJECT, 2, (0,)))):
        refused(c, ep, a, 'B8 ' + what)
    ep = ep_of(c, S['inj2'])
    out = do(c, ep, Action(Kind.INJECT, 2), 'B8 injection when the fact does not close the goal')
    expect(out is not None and len(out) == 1 and out[0].n_hyps == 4, 'B8 ... the fact n = m is a new hypothesis: one goal')
    refused(c, ep, Action(Kind.INJECT, 2), 'B8 the facts are hypotheses now: injecting again is refused')
    expect(not any(a.kind == Kind.INJECT for a in ep.menu().actions), 'B8 ... and not offered either')
    ep = ep_of(c, S['trans'])
    refused(c, ep, Action(Kind.INJECT, 3), 'B8 an inequality has no injectivity lemma')
    # ---- B9. LINCOMB
    ep = ep_of(c, S['lc'])
    out = do(c, ep, act(Kind.LINCOMB, (('n', 1),), (('n', 1),), hyps=(2, 3)), 'B9 the sum of the two equations')
    expect(out == [] and ep.done and ep.finish() is not None, 'B9 ... is the goal 2 a = 6: finished')
    for what, a in (('wrong coefficients', act(Kind.LINCOMB, (('n', 2),), (('n', 1),), hyps=(2, 3))),
                    ('one coefficient for two equations', act(Kind.LINCOMB, (('n', 1),), hyps=(2, 3))),
                    ('two coefficients for one equation', act(Kind.LINCOMB, (('n', 1),), (('n', 1),), hyps=(2,))),
                    ('no equation', act(Kind.LINCOMB)),
                    ('a coefficient that is a proof', act(Kind.LINCOMB, hyp(2), (('n', 1),), hyps=(2, 3))),
                    ('an empty coefficient', act(Kind.LINCOMB, (), (('n', 1),), hyps=(2, 3))),
                    ('a variable instead of an equation', act(Kind.LINCOMB, (('n', 1),), (('n', 1),), hyps=(0, 3))),
                    ('an equation that does not exist', act(Kind.LINCOMB, (('n', 1),), (('n', 1),), hyps=(2, 9))),
                    ('an argument', act(Kind.LINCOMB, (('n', 1),), (('n', 1),), hyps=(2, 3), arg=1))):
        refused(c, ep_of(c, S['lc']), a, 'B9 ' + what)
    refused(c, ep_of(c, S['lc_le']), act(Kind.LINCOMB, (('n', 1),), (('n', 1),), hyps=(2, 3)), 'B9 a goal that is no equation')
    # ---- B10. USE with a witness said
    ep = ep_of(c, S['ex49'])
    out = do(c, ep, act(Kind.USE, (('n', 7),)), 'B10 use 7')
    expect(out == [] and ep.done and ep.finish() is not None, 'B10 ... 7 * 7 = 49 is closed')
    ep = ep_of(c, S['ex49'])
    out = do(c, ep, act(Kind.USE, (('n', -7),)), 'B10 use -7')
    expect(out == [] and ep.done, 'B10 ... works as well')
    for what, a in (('a witness that is a proposition', act(Kind.USE, (('c', gid['Z.le'], 2), ('n', 1), ('n', 2)))), ('a witness of the wrong type (a function)', act(Kind.USE, (('lam',), ('n', 1)))),
                    ('a witness and a pool term', Action(Kind.USE, 0, (0,), ((('n', 7),),))), ('two witnesses', act(Kind.USE, (('n', 7),), (('n', 7),))), ('an empty witness', act(Kind.USE, ())),
                    ('hypotheses', act(Kind.USE, (('n', 7),), hyps=(0,)))):
        refused(c, ep_of(c, S['ex49']), a, 'B10 ' + what)
    refused(c, ep_of(c, S['mul_one']), act(Kind.USE, (('n', 7),)), 'B10 a goal that is no existential (an equation)')
    ep = ep_of(c, S['amgm'])
    out = do(c, ep, act(Kind.USE, (('n', 7),)), 'B10 use 7 for an inequality (an inequality is an existential behind its definition: x <= y is the existence of the difference)')
    expect(out is not None and len(out) == 1, 'B10 ... leaves the equation 2 a b + 7 = a * a + b * b')
    ep = ep_of(c, S['ex49'])
    out = do(c, ep, act(Kind.USE, (('n', 8),)), 'B10 use 8 (a false witness)')
    expect(out is not None and len(out) == 1 and eng.probe(ep.goal(), Action(Kind.RING, 0), ep.menu()) is None, 'B10 ... leaves 8 * 8 = 49, which nothing proves')
    # ---- B11. CASES / INDUCT that generalise
    st = w.global_type('add_left_cancel'); ep = ep_of(c, st, premises=gid['add_left_cancel']); g = ep.goal()
    plain = eng.probe(g, Action(Kind.INDUCT, 0), ep.menu()); gen = eng.probe(g, Action(Kind.INDUCT, 0, (), (), (1, 2)), ep.menu())
    expect(plain is not None and gen is not None and [x.key for x in plain] != [x.key for x in gen], 'B11 INDUCT that generalises a hypothesis leaves other goals than the plain INDUCT (add_left_cancel: %s hyps)' % (g.n_hyps,))
    if plain and gen:
        expect(len(plain) == len(gen) == 1 and k.view(gen[0].hyp_types[1])[0] == 'pi' and k.view(plain[0].hyp_types[3])[0] == 'pi' and gen[0].hyp_types[1] != plain[0].hyp_types[3],
               'B11 ... (the base case is closed by computation) the induction hypothesis of the generalising one is for all values of the generalised variables, the plain one is for these')
    for what, a in (('a duplicate', Action(Kind.INDUCT, 0, (), (), (3, 3))), ('the variable itself', Action(Kind.INDUCT, 0, (), (), (0,))), ('a position that does not exist', Action(Kind.INDUCT, 0, (), (), (99,))),
                    ('a negative position', Action(Kind.INDUCT, 0, (), (), (-1,))), ('codes', Action(Kind.INDUCT, 0, (), ((('n', 1),),), (3,)))):
        refused(c, ep, a, 'B11 INDUCT with ' + what)
    refused(c, ep, Action(Kind.CASES, 0, (), (), (0,)), 'B11 CASES that generalises the variable itself')
    refused(c, ep, Action(Kind.CASES, 0, (), (), (99,)), 'B11 CASES that generalises a position that does not exist')
    expect(eng.probe(g, Action(Kind.CASES, 0, (), (), (3,)), ep.menu()) is not None, 'B11 CASES that generalises a hypothesis is accepted')
    expect(eng.menu(g).signature(Action(Kind.INDUCT, 0, (), (), (3,)))[3] == (G.GEN_MARK, (3,)) and eng.menu(g).find(eng.menu(g).signature(Action(Kind.INDUCT, 0, (), (), (3,)))) == Action(Kind.INDUCT, 0, (), (), (3,)), 'B11 the signature of a generalising INDUCT names the hypotheses, and finds it again')
    eng.premises = None


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. the portable form
BAD_PORTABLE = [None, 5, [], 'REFINE', {}, {'kind': 5}, {'kind': 'NOPE'}, {'kind': None}, {'kind': []}, {'kind': 'REFINE'}, {'kind': 'REFINE', 'codes': 5}, {'kind': 'REFINE', 'codes': [5]},
                {'kind': 'REFINE', 'codes': [['x']]}, {'kind': 'REFINE', 'codes': [[[1]]]}, {'kind': 'REFINE', 'codes': [[['c', 1, 0]]], 'hyps': 'a'}, {'kind': 'REFINE', 'codes': [[['c', 1, 0]]], 'hyps': [None]},
                {'kind': 'REFINE', 'codes': [[['c', 1, 0]]], 'arg': 'x'}, {'kind': 'REFINE', 'codes': [[['c', 1, 0]]], 'arg': None}, {'kind': 'REFINE', 'codes': [[['c', 1.5, 0]]]}, {'kind': 'REFINE', 'codes': [[['c', True, 0]]]},
                {'kind': 'APPLY'}, {'kind': 'APPLY', 'source': 5}, {'kind': 'APPLY', 'source': [1, 2]}, {'kind': 'APPLY', 'source': ['c', 'x', 0]}, {'kind': 'APPLY', 'source': ['c', 1, 0, 0]}, {'kind': 'APPLY', 'source': ['x', 1, 0]},
                {'kind': 'APPLY', 'source': ['c', 1, 1]}, {'kind': 'APPLY', 'source': ['c', 10 ** 9, 0]}, {'kind': 'APPLY', 'source': ['c', -1, 0]}, {'kind': 'APPLY', 'source': ['v', 99, 0]}, {'kind': 'APPLY', 'source': ['c', True, 0]},
                {'kind': 'APPLY', 'source': ['v', -1, 0]}, {'kind': 'REWRITE', 'source': ['c', 302, 0], 'rev': 'x'}, {'kind': 'REWRITE', 'source': ['c', 302, 0], 'rev': 2}, {'kind': 'INTERVAL_CASES'},
                {'kind': 'INTERVAL_CASES', 'range': 5}, {'kind': 'INTERVAL_CASES', 'range': [0, 1, 2]}, {'kind': 'INTERVAL_CASES', 'range': ['a', 1, 2]}, {'kind': 'INSTANTIATE'}, {'kind': 'USE', 'arg': 0},
                {'kind': 'CLAIM', 'codes': [[['n', 1]]], 'hyps': [0]}, {'kind': 'INTRO'}, {'kind': 'RING', 'arg': 3}, {'kind': 'RING', 'codes': [[['c', 1, 0]]]}, {'kind': 'REFINE', 'codes': [[['c', 1, 0]]], 'extra': 7}]


def sec_C(c):
    fx = c['fx']; w, k, eng = fx.w, fx.k, fx.eng
    eng.premises = None
    res = fx.results; gid = fx.gid; S = statements(w)
    # ---- C1. every demonstration replays from its JSON in a fresh engine
    eng2 = G.Engine(w, kinds=set(G.Kind), strict=True)
    demos = REC.demonstrations(res)
    js = json.loads(json.dumps(demos))
    expect(js == demos, 'C1 the demonstrations are plain data: a round trip through JSON changes nothing')
    n_steps = 0; bad = []; kinds = collections.Counter()
    for d in js:
        try:
            A.replay_portable(eng2, fx.recs[d['name']].stmt, d['steps'], d['gid']); n_steps += len(d['steps'])
            for s_ in d['steps']: kinds[s_['kind']] += 1
        except Exception as e: bad.append((d['name'], type(e).__name__, str(e)[:80]))
    expect(not bad and len(js) >= 453 and n_steps >= 2600, 'C1 every demonstration replays from its JSON in a fresh engine and the kernel accepts it (%d proofs, %d steps; %s)' % (len(js), n_steps, bad[:3]))
    expect(all(kinds[x] > 0 for x in ('APPLY', 'REWRITE', 'CASES', 'INDUCT', 'CLAIM', 'FORWARD', 'APPLY_WITH', 'RING', 'USE', 'REFINE', 'INDUCT_WITH', 'LINARITH', 'REWRITE_AT', 'REWRITE_WITH', 'LINCOMB', 'NLINARITH')),
           'C1 the demonstrations use every kind of action there is to demonstrate (%s)' % dict(kinds))
    # the actions that say everything themselves are their own portable form: to_portable / from_portable is the identity
    n_id = 0; bad = []
    for nm, r in res.items():
        if not r.ok: continue
        for (a, sig), d in zip(r.actions, r.portable):
            if d['kind'] in A.MENU_KINDS: continue
            n_id += 1
            if A.from_portable(eng, A.NO_MENU, d) != a or A.to_portable(eng, A.NO_MENU, a) != d: bad.append((nm, d['kind']))
    expect(n_id >= 900 and not bad, 'C1 an action that says its arguments is its own portable form (%d actions; %s)' % (n_id, bad[:3]))
    # ---- C2. the actions that name a menu entry: by lemma or hypothesis, found again in the menu, else the equivalent action that says it
    ep = ep_of(c, S['rwg']); g = ep.goal(); m = ep.menu()
    ap = [a for a in m.actions if a.kind == Kind.REWRITE]
    expect(len(ap) >= 1, 'C2 the menu of a x + y rewrite goal lists rewrites (%d)' % len(ap))
    for a in ap[:6]:
        d = A.to_portable(eng, m, a)
        expect(d['kind'] == 'REWRITE' and d['source'][0] in ('c', 'v') and d['rev'] in (0, 1) and d['rev'] == int(bool(m.cands[a.arg][1])), 'C2 a rewrite is named by its lemma and direction (%s)' % (d,))
        expect(A.from_portable(eng, m, json.loads(json.dumps(d))) == a, 'C2 ... and found again in the menu after a round trip through JSON')
        flipped = dict(d); flipped['rev'] = 1 - d['rev']
        expect(A.from_portable(eng, m, flipped) != a, 'C2 ... in its direction: the other direction is not the same action')
        fb = A.from_portable(eng, A.NO_MENU, d)
        expect(fb.kind == Kind.REWRITE_WITH and fb.arg == d['rev'] and fb.codes == ((tuple(d['source']),),), 'C2 ... without the menu it is the rewrite that says its source (%r)' % (fb,))
        p1 = eng.probe(g, a, m); p2 = eng.probe(g, fb, A.NO_MENU)
        expect(p1 is not None and p2 is not None and [x.key for x in p1] == [x.key for x in p2], 'C2 ... and both leave the same goals')
    ep = ep_of(c, S['trans_part']); g = ep.goal(); m = ep.menu()
    ap = [a for a in m.actions if a.kind == Kind.APPLY]
    expect(len(ap) >= 1, 'C2 the menu lists applications (%d)' % len(ap))
    for a in ap[:8]:
        d = A.to_portable(eng, m, a)
        fb = A.from_portable(eng, A.NO_MENU, d)
        expect(A.from_portable(eng, m, d) == a and fb.kind == Kind.APPLY_WITH and fb.arg == 1 and fb.codes == ((tuple(d['source']),),), 'C2 an application is named by its source; found again, else APPLY_WITH in the unifier mode (%s)' % (d,))
        p1 = eng.probe(g, a, m); p2 = eng.probe(g, fb, A.NO_MENU)
        expect((p1 is None) == (p2 is None) or p2 is None, 'C2 ... the menu\'s application and the equivalent action agree whether the application works (%s)' % (d,))
        if p1 is not None and p2 is not None: expect([x.key for x in p1] == [x.key for x in p2], 'C2 ... and what it leaves (%s)' % (d,))
    with w.nota.using(w['Z']):
        lit = w.nota.lit; F = w.forall_
        ivs = F([w['Z'], lambda x: w['Z.le'](lit(0), x), lambda x, h0: w['Z.le'](x, lit(3))], lambda x, h0, h1: w['Z.le'](x * x, lit(9)))
    ep = ep_of(c, ivs); g = ep.goal(); m = ep.menu()
    iv = [a for a in m.actions if a.kind == Kind.INTERVAL_CASES]
    expect(len(iv) == 1, 'C2 the menu offers the interval split of 0 <= x <= 3')
    if iv:
        d = A.to_portable(eng, m, iv[0])
        expect(d['kind'] == 'INTERVAL_CASES' and d['range'] == [0, 0, 3] and A.from_portable(eng, m, d) == iv[0], 'C2 an interval split is named by its range (%s)' % (d,))
        refuses('C2 ... and has no meaning without the menu that lists it', lambda: A.from_portable(eng, A.NO_MENU, d), G.ActionError)
        refuses('C2 ... nor for another range', lambda: A.from_portable(eng, m, {'kind': 'INTERVAL_CASES', 'range': [0, 0, 4]}), G.ActionError)
    # the actions that are relative to the menu's pool have no portable form
    ep = ep_of(c, S['ex49']); m = ep.menu()
    use = [a for a in m.actions if a.kind == Kind.USE and a.args]
    expect(len(use) >= 1, 'C2 the menu offers witnesses from its pool')
    for a in use[:3]: refuses('C2 a witness by position in the pool has no portable form', lambda: A.to_portable(eng, m, a), G.ActionError)
    ok_use = A.to_portable(eng, m, act(Kind.USE, (('n', 7),)))
    expect(ok_use == {'kind': 'USE', 'codes': [[['n', 7]]]} and A.from_portable(eng, m, ok_use) == act(Kind.USE, (('n', 7),)), 'C2 ... a witness said by a code does (%s)' % (ok_use,))
    with w.nota.using(w['Z']):
        lin = F([w.arrow(w['Z'], w['Z']), lambda f: F([w['Z']], lambda n: w.Eq(f(n), 2 * n + 1))], lambda f, h: w.Eq(f(lit(3)) + f(lit(3)), lit(14)))
    ep = ep_of(c, lin); m = ep.menu()
    ins = [a for a in m.actions if a.kind == Kind.INSTANTIATE]
    expect(len(ins) >= 1, 'C2 the menu offers instances of the hypothesis')
    for a in ins[:3]: refuses('C2 an instance by positions in the pool has no portable form', lambda: A.to_portable(eng, m, a), G.ActionError)
    # ---- C3. data that is no action
    st = w.term(S['trans'], 'statement'); n_bad = collections.Counter()
    for d in BAD_PORTABLE:
        try:
            m = eng.menu(eng.goal_of(eng.start(st), 0)) if isinstance(d, dict) and isinstance(d.get('kind'), str) and d['kind'] in A.MENU_KINDS else A.NO_MENU
            a = A.from_portable(eng, m, d)
        except G.ActionError: n_bad['refused as data'] += 1; continue
        except Exception as e: n_bad['other exception'] += 1; n_bad['example'] = '%r -> %s: %s' % (d, type(e).__name__, str(e)[:60]); continue
        try: eng.replay(eng.start(st), a, m); n_bad['carried out'] += 1
        except G.ActionError: n_bad['refused as action'] += 1
        except Exception as e: n_bad['other exception'] += 1; n_bad['example'] = '%r -> %s: %s' % (d, type(e).__name__, str(e)[:60])
    last = k.num_globals() - 1
    refuses('C3 a global one past the last is no global', lambda: A.from_portable(eng, A.NO_MENU, {'kind': 'APPLY', 'source': ['c', last + 1, 0]}), G.ActionError)
    expect(A.from_portable(eng, A.NO_MENU, {'kind': 'APPLY', 'source': ['c', last, 0]}).codes == ((('c', last, 0),),), 'C3 ... the last one is')
    expect(A.from_portable(eng, A.NO_MENU, {'kind': 'REWRITE', 'source': ['c', 0, 0], 'rev': True}).arg == 1, 'C3 the direction of a rewrite may be a boolean (JSON false / true)')
    expect(n_bad['other exception'] == 0, 'C3 data that is no action is ActionError, as data or as an action, never another exception (%s)' % dict(n_bad))
    expect(n_bad['refused as data'] >= 30 and n_bad['carried out'] <= 1, 'C3 nearly all of it is refused as data (%s)' % dict(n_bad))
    for what, steps in (('a number', 5), ('None', None), ('a dict', {}), ('a list of numbers', [1, 2]), ('a list of None', [None]), ('a list of lists', [[1]]), ('a step without a kind', [{}])):
        refuses('C3 the steps of a proof as %s' % what, lambda: A.replay_portable(eng, st, steps), G.ActionError)
    # ---- C4. a proof by portable actions is a proof of its statement, nothing less
    name = 'Z.sqrt2_aux'; r = res[name]; rec = fx.recs[name]
    steps = json.loads(json.dumps(r.portable))
    expect(A.replay_portable(eng, rec.stmt, steps, rec.gid) is not None, 'C4 (the proof of %s, %d steps, replays)' % (name, len(steps)))
    refuses('C4 with a step missing the proof is not finished', lambda: A.replay_portable(eng, rec.stmt, steps[:-1], rec.gid), G.ActionError)
    refuses('C4 with a step too many the proof is already finished', lambda: A.replay_portable(eng, rec.stmt, steps + [steps[-1]], rec.gid), G.ActionError)
    refuses('C4 with no step at all the goal is open', lambda: A.replay_portable(eng, rec.stmt, [], rec.gid), G.ActionError)
    refuses('C4 the proof of a theorem that uses a lemma beyond the premises given is refused', lambda: A.replay_portable(eng, rec.stmt, steps, 1), G.ActionError)
    eng.premises = None; A.replay_portable(eng, rec.stmt, steps, rec.gid)
    expect(eng.premises == rec.gid, 'C4 ... the premises given stay on the engine')
    eng.premises = None
    other = fx.recs['Z.prime_dvd_sq'].stmt
    refuses('C4 the steps of one theorem do not prove another', lambda: A.replay_portable(eng, other, steps, rec.gid), G.ActionError)
    eng.premises = None
    p = eng.start(rec.stmt)
    refuses('C4 an unfinished proof state cannot be verified', lambda: eng.verify(rec.stmt, p))
    eng.premises = rec.gid; p = eng.start(rec.stmt)
    for d in steps:
        g_ = eng.goal_of(p, 0); m_ = eng.menu(g_) if d['kind'] in A.MENU_KINDS else A.NO_MENU
        eng.replay(p, A.from_portable(eng, m_, d), m_)
    eng.verify(rec.stmt, p)
    refuses('C4 a finished proof is not a proof of another statement (the kernel says no)', lambda: eng.verify(other, p))
    eng.premises = None
    # tampering: a changed step is refused, or the proof it gives is checked by the kernel all the same
    rng = random.Random(4); tally = collections.Counter()
    for nm in ('Z.sqrt2_aux', 'imo1959_p1', 'imo2019_p1_bwd', 'Z.prime_dvd_sq', 'N.beq_eq', 'Z.exists_divmod', 'Pos.to_nat_add', 'mul_left_cancel'):
        if nm not in res or not res[nm].ok: continue
        base = json.loads(json.dumps(res[nm].portable)); rec = fx.recs[nm]
        for t in range(8):
            steps = json.loads(json.dumps(base))
            if not steps: break
            i = rng.randrange(len(steps)); how = rng.choice(('drop', 'repeat', 'swap', 'kind', 'token'))
            if how == 'drop': del steps[i]
            elif how == 'repeat': steps.insert(i, dict(steps[i]))
            elif how == 'swap' and len(steps) > 1: j = (i + 1) % len(steps); steps[i], steps[j] = steps[j], steps[i]
            elif how == 'kind': steps[i]['kind'] = rng.choice(('RING', 'LINARITH', 'CASES', 'INDUCT', 'APPLY', 'FORWARD', 'REFINE'))
            else:
                cs = steps[i].get('codes')
                if cs and cs[0]: cs[0][rng.randrange(len(cs[0]))] = ['c', rng.randrange(40), 0]
                else: steps[i]['hyps'] = [0]
            try:
                A.replay_portable(eng, rec.stmt, steps, rec.gid); tally['a proof, checked by the kernel'] += 1
            except G.ActionError: tally['refused'] += 1
            except Exception as e: tally['other: %s %s' % (type(e).__name__, str(e)[:60])] += 1
    eng.premises = None
    expect(not [x for x in tally if x.startswith('other')], 'C4 a changed sequence is refused or ends in a proof the kernel has checked, never another exception (%s)' % dict(tally))
    expect(tally['refused'] >= 20, 'C4 most changes are refused (%s)' % dict(tally))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. probes
MALFORMED = [Action('x', 0), Action(99, 0), Action(-1, 0), Action(Kind.REFINE, 0.5, (), ((('g',),),), ()), Action(Kind.REFINE, 0, 'abc', ((('g',),),), ()), Action(Kind.REFINE, 0, (), 5, ()), Action(Kind.REFINE, 0, (), None, ()),
             Action(Kind.REFINE, 0, (), (5,), ()), Action(Kind.REFINE, 0, (), ((5,),), ()), Action(Kind.REFINE, 0, (), ((('c', 'x', 0),),), ()), Action(Kind.REFINE, 0, (), ((('c', 1.5, 0),),), ()),
             Action(Kind.REFINE, 0, (), ((('c', 1, 0, 0, 0),),), ()), Action(Kind.REFINE, 0, (), (((),),), ()), Action(Kind.REFINE, 0, (), ((('zz', 1, 0),),), ()), Action(Kind.REFINE, 0, (), ((('c', 1, 0),),), 5),
             Action(Kind.REFINE, 0, (), ((('c', 1, 0),),), (None,)), Action(Kind.REFINE, 0, (), ((('c', 1, 0),),), (True,)), Action(Kind.REFINE, True, (), ((('c', 1, 0),),), ()), Action(Kind.CLAIM, 0, (), (), ()),
             Action(Kind.APPLY_WITH, 0, (), ((('v', 0, 0),),) * 30, ()), Action(Kind.REFINE, 0, (), ((('c', 1, 0),) * 500,), ()), Action(Kind.INDUCT_WITH, 0, (), ((('c', 435, 0),),), (10 ** 9,)), Action(Kind.FORWARD, 0, (), (), ()),
             Action(Kind.LINCOMB, 0, (), ((('n', 1),),), (0, 0)), Action(Kind.INJECT, 10 ** 9), Action(Kind.INJECT, -5), Action(Kind.USE, 0, (), ((),), ()), Action(Kind.USE, 3, (), ((('n', 1),),), ()),
             Action(Kind.CASES, 0, (), ((('n', 1),),), (1,)), Action(Kind.INSTANTIATE, 0, (), ((('n', 1),),), ())]


def sec_D(c):
    fx = c['fx']; w, k, eng = fx.w, fx.k, fx.eng; res = fx.results
    samples = []
    for nm in ('Z.sqrt2_aux', 'imo2019_p1_fwd', 'imo1959_p1', 'Z.prime_dvd_sq', 'Z.exists_divmod', 'N.beq_eq', 'Z.add_wd', 'Pos.to_nat_add'):
        def hook(i, p, g, a, menu, nm=nm): samples.append((nm, fx.recs[nm].gid, g, a, menu))
        fx.walk(nm, hook)
    expect(len(samples) >= 120, 'D  there are many recorded actions to probe (%d)' % len(samples))
    kinds = collections.Counter(Kind(a.kind).name for _, _, _, a, _ in samples)
    # ---- D1. a probe leaves the kernel as it was; probing again gives the same
    tally = collections.Counter(); rng = random.Random(5)
    for nm, gid, g, a, menu in samples:
        eng.premises = gid
        variants = [a]
        if a.kind in A.CODE_KINDS and a.codes and a.codes[0] and a.codes[0][0][0] in ('c', 'v'):             # a refused variant: another source
            tok = a.codes[0][0]; other = (tok[0], (tok[1] + 1 + rng.randrange(40)) % (gid if tok[0] == 'c' else max(1, g.n_hyps))) + tok[2:]
            variants.append(Action(a.kind, a.arg, (), ((other,) + a.codes[0][1:],) + a.codes[1:], a.hyps))
        for v in variants:
            s0 = state(k, w); np0 = eng.stats['probe_noprogress']
            out1, st1 = eng.probe_status(g, v, menu); s1 = state(k, w)
            tally['probes'] += 1; tally[st1] += 1
            if st1 != 'ok': tally['state changed by a refusal'] += (refusal_effect(eng, s0, s1, np0) is not None)
            else: tally['state changed by an acceptance'] += (s0[1:] != s1[1:])
            out2, st2 = eng.probe_status(g, v, menu); s2 = state(k, w)
            tally['not repeatable'] += (st1 != st2 or (out1 is None) != (out2 is None) or (out1 is not None and [x.key for x in out1] != [x.key for x in out2]))
            tally['pools grew on the second probe'] += (s2 != s1)
    eng.premises = None
    expect(tally['probes'] >= 200 and tally['ok'] >= 100 and tally['refused'] >= 30, 'D1 the probes are accepted and refused alike (%s)' % dict(tally))
    expect(tally['state changed by a refusal'] == 0 and tally['state changed by an acceptance'] == 0, 'D1 a probe leaves the kernel as it was: a refusal changes nothing (but for the goal of an action that made no progress), an acceptance only the goals it leaves (%s)' % dict(tally))
    expect(tally['not repeatable'] == 0 and tally['pools grew on the second probe'] == 0, 'D1 the same probe gives the same goals, and the second leaves no trace (%s)' % dict(tally))
    # ---- D2. the outcome depends on the budget only by running out of it
    levels = (1, 20, 200, 2000, 20000, 200000, 2000000)
    pick = [s_ for s_ in samples if s_[3].kind in A.CODE_KINDS or s_[3].kind == Kind.INJECT][::2][:70]
    bad = []; shapes = collections.Counter()
    for nm, gid, g, a, menu in pick:
        eng.premises = gid; seq = []; goals = []
        for f in levels:
            out, st = eng.probe_status(g, a, menu, fuel=f); seq.append(st); goals.append(None if out is None else [x.key for x in out])
        first = next((i for i, s_ in enumerate(seq) if s_ != G.ST_FUEL), len(seq))
        shapes[(seq[0], seq[-1])] += 1
        if len(set(seq[first:])) > 1 or (first < len(seq) and any(goals[i] != goals[first] for i in range(first, len(seq)))): bad.append((nm, Kind(a.kind).name, seq))
    eng.premises = None
    expect(not bad, 'D2 for every budget an action runs out of work or has its one outcome (%s)' % (bad[:3],))
    expect(shapes[(G.ST_FUEL, G.ST_OK)] >= 30, 'D2 with a budget of one step nothing is carried out, with a large one most are (%s)' % dict(shapes))
    # a budget too small for the goal itself is running out of work, not a bug
    ep = ep_of(c, statements(w)['trans']); g = ep.goal()
    out, st = eng.probe_status(g, act(Kind.APPLY_WITH, ref(fx.gid['Z.le_trans']), hyp(1)), ep.menu(), fuel=1)
    expect(out is None and st == G.ST_FUEL, 'D2 a budget of one step: out of work (%s)' % (st,))
    out, st = eng.probe_status(g, Action(Kind.INTRO, 0), ep.menu(), fuel=1)
    expect(st in (G.ST_FUEL, G.ST_REFUSED) and out is None, 'D2 ... for every kind of action (%s)' % (st,))
    # ---- D3. actions that mean nothing are refused by the probe, never a bug
    n_bugs = len(eng.bugs); ep = ep_of(c, statements(w)['conj']); g = ep.goal(); m = ep.menu(); bad = []
    for a_ in MALFORMED:
        out, st = eng.probe_status(g, a_, m)
        if st != G.ST_REFUSED or out is not None: bad.append((a_, st))
        try: ep.step(a_)
        except G.ActionError: pass
        except Exception as e: bad.append((a_, type(e).__name__))
        else: bad.append((a_, 'accepted'))
    expect(not bad and len(eng.bugs) == n_bugs, 'D3 an action that means nothing is refused, by the probe and by the episode, and is no bug (%s)' % (bad[:3],))
    expect(ep.n_goals == 1 and ep.goal().key == g.key, 'D3 ... and changed nothing')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. the menu
SECOND = (Kind.FORWARD, Kind.REWRITE_AT, Kind.INDUCT_WITH, Kind.INJECT, Kind.CLAIM)


def sec_E(c):
    fx = c['fx']; w, k, eng = fx.w, fx.k, fx.eng; gid = fx.gid; S = statements(w)
    # ---- E1. the goals of many proofs: bounded, deterministic, found again, listed once, nearly all carried out
    goals = []
    for nm in list(fx.recs)[::9]:
        g0 = fx.first_goal(nm)
        if g0 is not None: goals.append((nm, fx.recs[nm].gid, g0))
    for nm in ('Z.sqrt2_aux', 'imo2019_p1_fwd', 'Z.exists_divmod', 'Z.prime_dvd_sq', 'mul_left_cancel'):
        def hook(i, p, g, a, menu, nm=nm):
            if i % 2 == 0: goals.append((nm, fx.recs[nm].gid, g))
        fx.walk(nm, hook)
    eng2 = G.Engine(w, kinds=set(G.Kind), strict=True)
    tally = collections.Counter(); sigs_seen = []
    for nm, gid_, g in goals:
        eng.premises = gid_; eng2.premises = gid_
        m = eng.menu(g); g2 = G.GoalData(g.target, g.hyp_types, g.hyp_names); m2 = eng2.menu(g2)
        sig = [m.signature(a) for a in m.actions]
        tally['goals'] += 1; tally['actions'] += len(sig)
        tally['not deterministic'] += (sig != [m2.signature(a) for a in m2.actions])
        tally['listed twice'] += (len(set(sig)) != len(sig))
        tally['signature does not find the action'] += sum(1 for a in m.actions if m.find(m.signature(a)) != a)
        cnt = collections.Counter(a.kind for a in m.actions)
        tally['over a cap'] += (cnt[Kind.FORWARD] > A.MAX_FORWARD or cnt[Kind.REWRITE_AT] > A.MAX_REWRITE_AT or cnt[Kind.INDUCT_WITH] > A.MAX_INDUCT)
        for kd in SECOND: tally['kind ' + kd.name] += cnt[kd]
        for a in m.actions:
            if a.kind in SECOND:
                tally['second-language actions'] += 1
                try: eng.check_action(a, m, g.n_hyps)
                except G.ActionError: tally['malformed listed action'] += 1
                if a.kind in (Kind.FORWARD, Kind.INJECT) or tally['second-language actions'] % 3 == 0:
                    out, st = eng.probe_status(g, a, m); tally['probed'] += 1; tally['probed ok'] += (st == G.ST_OK)
                    tally['probed ' + Kind(a.kind).name] += 1; tally['probed ok ' + Kind(a.kind).name] += (st == G.ST_OK)
    eng.premises = None; eng2.premises = None
    trace('E1 tally %s' % sorted(tally.items()))
    expect(tally['goals'] >= 60 and tally['actions'] > 1000, 'E1 many goals, many actions (%s)' % dict(tally))
    expect(tally['not deterministic'] == 0 and tally['listed twice'] == 0 and tally['signature does not find the action'] == 0, 'E1 menus are deterministic across engines, list every action once, and every action is found again from its signature (%s)' % dict(tally))
    expect(tally['over a cap'] == 0 and tally['malformed listed action'] == 0, 'E1 the caps hold and every listed action has the shape of its kind (%s)' % dict(tally))
    expect(all(tally['kind ' + kd.name] > 0 for kd in (Kind.FORWARD, Kind.REWRITE_AT, Kind.INDUCT_WITH, Kind.INJECT)), 'E1 the enumerators list their kinds (%s)' % {kd.name: tally['kind ' + kd.name] for kd in SECOND})
    expect(tally['probed'] >= 100 and tally['probed ok'] >= 0.95 * tally['probed'], 'E1 nearly all the listed actions are carried out by the probe (%d of %d)' % (tally['probed ok'], tally['probed']))
    for kd, floor, least in ((Kind.FORWARD, 1.0, 30), (Kind.INJECT, 1.0, 1), (Kind.INDUCT_WITH, 1.0, 200), (Kind.REWRITE_AT, 0.97, 300)):      # (a rewrite can be undone by the normaliser: `Npos 2` is `N.of_nat (Pos.to_nat 2)`)
        n_, ok_ = tally['probed ' + kd.name], tally['probed ok ' + kd.name]
        expect(n_ >= least and ok_ >= floor * n_, 'E1 the listed %s actions are carried out by the probe: %d of %d (at least %d%%)' % (kd.name, ok_, n_, round(100 * floor)))
    # ---- E2. what the enumerators list, on goals where it is known
    ep = ep_of(c, S['rwa']); g = ep.goal(); m = ep.menu()
    ra = [a for a in m.actions if a.kind == Kind.REWRITE_AT]
    expect(any(a.codes == (ref(gid['Z.add_comm']),) and a.hyps == (3,) for a in ra), 'E2 REWRITE_AT offers Z.add_comm at the hypothesis a + b = c (%d offered)' % len(ra))
    ep = ep_of(c, S['inj']); g = ep.goal(); m = ep.menu()
    expect([a for a in m.actions if a.kind == Kind.INJECT] == [Action(Kind.INJECT, 2)], 'E2 INJECT is offered for S n = S m (and for nothing else)')
    ep.step(Action(Kind.INJECT, 2))
    ep = ep_of(c, S['injz']); m = ep.menu()
    expect([a for a in m.actions if a.kind == Kind.INJECT] == [Action(Kind.INJECT, 3)], 'E2 ... and for c + a = c + b')
    ep = ep_of(c, S['trans']); m = ep.menu()
    expect(not any(a.kind == Kind.INJECT for a in m.actions), 'E2 INJECT is not offered where no hypothesis is an equation of a constructor')
    ep = ep_of(c, S['ind']); g = ep.goal(); m = ep.menu()
    ia = [a for a in m.actions if a.kind == Kind.INDUCT_WITH]
    expect(any(a.codes == (ref(gid['Z.induction_from']), (('n', 1),)) and a.hyps == (0,) for a in ia), 'E2 INDUCT_WITH offers Z.induction_from with the starting point 1 that the hypothesis 1 <= n shows')
    expect(any(a.codes == (ref(gid['Z.int_induction']),) and a.hyps == (0,) for a in ia), 'E2 ... and Z.int_induction, which needs no starting point')
    expect(not any(a.codes and a.codes[0] == ref(gid['N.ind']) for a in ia), 'E2 ... but no principle about another type')
    expect(not any(a.hyps == (1,) for a in ia), 'E2 ... nor induction on a proof')
    expect(all(eng.probe(g, a, m) is not None for a in ia if a.codes[0] != ref(gid['Z.strong_induction'])), 'E2 ... all of them carried out by the probe (%d offered)' % len(ia))
    ep = ep_of(c, S['ind_motive']); m = ep.menu()
    starts = [a.codes[1] for a in m.actions if a.kind == Kind.INDUCT_WITH and a.codes[0] == ref(gid['Z.induction_from']) and a.hyps == (0,)]
    expect(starts == [(('n', 0),), (('n', 1),)], 'E2 where no hypothesis shows a starting point, INDUCT_WITH tries the numerals 0 and 1 (%s)' % (starts,))
    # ---- E3. FORWARD from the lemmas the engine is allowed to instantiate, and CLAIM from its hints
    eng3 = G.Engine(w, kinds=set(G.Kind), strict=True, inst_lemmas=('Z.le_trans', 'Z.mul_comm'))
    ep = G.Episode(eng3, S['trans']); g = ep.goal(); m = ep.menu()
    fa = [a for a in m.actions if a.kind == Kind.FORWARD]
    expect(act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), hyp(3), hyp(4)) in fa, 'E3 FORWARD offers Z.le_trans with the two hypotheses for its premises (%d offered)' % len(fa))
    expect(all(eng3.probe(g, a, m) is not None for a in fa), 'E3 ... every one carried out by the probe')
    ep.step(act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), hyp(3), hyp(4)))
    expect(ep.done, 'E3 (it proves the goal)')
    ep = G.Episode(eng3, S['chain4']); m = ep.menu()
    first = act(Kind.FORWARD, ref(gid['Z.le_trans']), (), (), (), hyp(4), hyp(5))
    expect(first in [a for a in m.actions if a.kind == Kind.FORWARD], 'E3 FORWARD offers the transitivity of the first two inequalities of a chain')
    ep.step(first)
    expect(first not in [a for a in ep.menu().actions if a.kind == Kind.FORWARD] and ep.n_goals == 1, 'E3 a fact that is a hypothesis now is not offered again')
    ep = G.Episode(eng3, S['trans_part']); m = ep.menu()
    fa2 = [a for a in m.actions if a.kind == Kind.FORWARD]
    expect(not any(a.codes[0] == ref(gid['Z.le_trans']) for a in fa2), 'E3 FORWARD of Z.le_trans is not offered when its premises are not hypotheses')
    Z = w['Z']; zle = w['Z.le']; Eq = w.Eq; F = w.forall_
    with w.nota.using(Z):
        h_sq = F([Z], lambda a: zle(w['Z.zero'], a * a)).n
        h_comm = F([Z, Z], lambda a, b: Eq(a * b, b * a)).n
        h_hyp = zle(w.nota.lit(0), w.nota.lit(1)).n                       # closed: the hypothesis of S['closed']
        h_goal = zle(w.nota.lit(0), w.nota.lit(1) + w.nota.lit(1)).n      # closed: its target
    eng4 = G.Engine(w, kinds={Kind.CLAIM}, strict=True, claims=[h_sq, h_comm, h_sq, h_hyp, h_goal])
    ep = G.Episode(eng4, S['amgm']); m = ep.menu(); sp = eng4.space(ep.goal())
    ca = [a for a in m.actions if a.kind == Kind.CLAIM]
    expect(ca == [act(Kind.CLAIM, sp.encode(h, sp.prop)) for h in (h_sq, h_comm, h_hyp, h_goal)], 'E3 CLAIM offers the engine\'s hints, each once (%d offered)' % len(ca))
    expect(all(a.kind == Kind.CLAIM for a in m.actions), 'E3 an engine restricted to CLAIM offers nothing else (%s)' % collections.Counter(Kind(a.kind).name for a in m.actions))
    expect(all(eng4.probe(ep.goal(), a, m) is not None for a in ca), 'E3 ... and the probe carries every one out')
    sp9 = GB.TermSpace(eng4, [w['Z'].n] * 9, None)                                         # (a context size no goal had yet)
    w.probing += 1
    try: inside = eng4.claim_codes(sp9)
    finally: w.probing -= 1
    outside = eng4.claim_codes(sp9)
    expect(inside == [] and len(outside) == 4 and eng4.claim_codes(sp9) is outside, 'E3 the codes of the hints are made outside probes (whose nodes are reclaimed) and once for each size of context (%d, then %d)' % (len(inside), len(outside)))
    ep.step(ca[0])
    expect(ep.n_goals == 2, 'E3 a claim stepped from the menu leaves the claim and the goal')
    ep.step(act(Kind.REFINE, (('c', gid['Z.sq_nonneg'], 1),)))
    expect(ep.n_goals == 1 and not any(a.codes == ca[0].codes for a in ep.menu().actions if a.kind == Kind.CLAIM), 'E3 a claim that is a hypothesis now is not offered again')
    ep = G.Episode(eng4, S['closed']); m = ep.menu(); sp = eng4.space(ep.goal())
    ca = [a for a in m.actions if a.kind == Kind.CLAIM]
    expect(ca == [act(Kind.CLAIM, sp.encode(h, sp.prop)) for h in (h_sq, h_comm)], 'E3 a hint that is a hypothesis, or the goal itself, is not offered as a claim (%d offered)' % len(ca))
    # ---- E4. the engine's kinds decide what is listed; every action is legal either way
    e1 = G.Engine(w, strict=True)                                   # the first language only
    seen = set()
    for nm, gid_, g in goals[:25]:
        e1.premises = gid_; seen |= {a.kind for a in e1.menu(g).actions}
    expect(not (seen & set(A.CODE_KINDS | {Kind.INJECT})), 'E4 an engine of the first language lists no action of the second (%s)' % sorted(Kind(x).name for x in seen))
    eng5 = G.Engine(w, kinds={Kind.FORWARD}, strict=True, inst_lemmas=('Z.le_trans',))
    ep = G.Episode(eng5, S['trans']); m = ep.menu()
    expect(m.actions and all(a.kind == Kind.FORWARD for a in m.actions), 'E4 an engine restricted to FORWARD lists only FORWARD')
    g = ep.goal()
    expect(eng5.probe(g, Action(Kind.APPLY_WITH, 1, (), ((('c', gid['Z.le_trans'], 0),), (('v', 0, 0),), (('v', 1, 0),))), m) is not None, 'E4 ... and still carries out an action of a kind it does not list')
    eng.premises = None
    # ---- E5. the premises are a hard limit: a lemma at the limit is not listed, the one before it is
    eng6 = G.Engine(w, kinds=set(G.Kind), strict=True, inst_lemmas=('Z.le_trans',))
    inj_lemmas = sorted(eng.inj_index.get(gid['Succ'], ()))
    expect(inj_lemmas, 'E5 (there is an injectivity lemma for Succ)')
    cases = [('REWRITE_AT', eng, 'rwa', Kind.REWRITE_AT, lambda a: a.codes[0] == ref(gid['Z.add_comm']), gid['Z.add_comm']),
             ('INDUCT_WITH', eng, 'ind', Kind.INDUCT_WITH, lambda a: a.codes[0] == ref(gid['Z.induction_from']), gid['Z.induction_from']),
             ('FORWARD', eng6, 'trans', Kind.FORWARD, lambda a: a.codes[0] == ref(gid['Z.le_trans']), gid['Z.le_trans']),
             ('INJECT', eng, 'inj', Kind.INJECT, lambda a: True, inj_lemmas[0] if inj_lemmas else 0)]
    for label, en, stmt, kind, pred, lemma in cases:
        for prem, want in ((lemma, False), (lemma + 1, True)):
            en.premises = prem
            m_ = G.Episode(en, S[stmt]).menu()
            got = any(a.kind == kind and pred(a) for a in m_.actions)
            expect(got == want, 'E5 with %d premises a lemma %s %s by %s (it is number %d)' % (prem, 'at the limit is' if not want else 'before the limit is', 'listed' if got else 'not listed', label, lemma))
    eng.premises = None; eng6.premises = None
    # ---- E6. the caps hold, and they are reached; FORWARD is for sources with few premises
    with w.nota.using(w['Z']):
        lit = w.nota.lit
        def total(xs): return functools.reduce(lambda a_, b_: a_ + b_, xs)
        many_vars = w.forall_([w['Z']] * 12, lambda *xs: w.Eq(total(xs), total(list(reversed(xs)))))                         # (12 variables to induct on)
        eq_binders = [w['Z']] * 10 + [(lambda i: lambda *vs: w.Eq(vs[i % 10] + vs[(i + 1) % 10], vs[(i + 2 + i // 10) % 10]))(i) for i in range(40)]
        many_eqs = w.forall_(eq_binders, lambda *vs: w.Eq(vs[0], vs[0] + lit(0)))                                           # (40 equations to rewrite)
        fwd_binders = [w['Z']] * 4 + [(lambda j: lambda *vs: w['Z.le'](lit(0), vs[j]))(j) for j in range(4)] + \
                      [(lambda c_: lambda *vs: w.forall_([w['Z'], lambda x: w['Z.le'](lit(0), x)], lambda x, h: w['Z.le'](lit(0), x + lit(c_ + 1))))(c_) for c_ in range(20)]
        many_fwd = w.forall_(fwd_binders, lambda *vs: w['Z.le'](lit(0), vs[0] + lit(50)))                                    # (20 sources with a premise, 4 facts to fill it)
        def with_premises(n):                                                                                                # a, 0 <= a, a source with n premises 0 <= a, the goal 0 <= a + 7
            def source(a): return w.forall_([(lambda i: lambda *vs: w['Z.le'](lit(0), a))(i) for i in range(n)], lambda *vs: w['Z.le'](lit(0), a + lit(n)))
            return w.forall_([w['Z'], lambda a: w['Z.le'](lit(0), a), lambda a, h: source(a)], lambda a, h, src: w['Z.le'](lit(0), a + lit(7)))
        by_premises = {n_: with_premises(n_) for n_ in (A.MAX_PREMISES, A.MAX_PREMISES + 1)}
    for label, stmt, kind, cap in (('INDUCT_WITH', many_vars, Kind.INDUCT_WITH, A.MAX_INDUCT), ('REWRITE_AT', many_eqs, Kind.REWRITE_AT, A.MAX_REWRITE_AT), ('FORWARD', many_fwd, Kind.FORWARD, A.MAX_FORWARD)):
        m_ = G.Episode(eng, stmt).menu(); acts_ = [a for a in m_.actions if a.kind == kind]
        expect(len(acts_) == cap and len(set(m_.signature(a) for a in acts_)) == cap, 'E6 a goal with more %s actions than the cap lists exactly %d, each once (%d)' % (label, cap, len(acts_)))
    for n_prem, want in ((A.MAX_PREMISES, True), (A.MAX_PREMISES + 1, False)):
        m_ = G.Episode(eng, by_premises[n_prem]).menu()
        got = any(a.kind == Kind.FORWARD and a.codes[0] == (('v', 2, 0),) for a in m_.actions)
        expect(got == want, 'E6 FORWARD from a hypothesis with %d premises (all hypotheses) is %s (the limit is %d)' % (n_prem, 'listed' if got else 'not listed', A.MAX_PREMISES))
    eng.premises = None


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. the recorder
KNOWN_GAPS = {'Z.induction_on': 'not_encodable', 'Z.mk_inj': 'not_encodable',        # (a term with a `match`; a term that is a proof by computation only)
              'Z.dvd_fact': 'stale_ref', 'Z.one_le_pow': 'stale_ref',                  # (the author's script uses an induction hypothesis the engine does not have: the author's induction is not the engine's)
              'Z.pow_nonneg': 'probe_apply', 'Z.pow_pos': 'probe_apply'}               # (the author's `apply` leaves one goal where the engine's action leaves none: the engine settles the goal itself)


def indep(c):
    """the independent checker, with the whole kernel as its environment (made on first use)"""
    if 'ic' not in c:
        k = c['fx'].k; env = ic.Env()
        for gid in range(k.num_globals()):
            name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
            env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid), recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
        c['ic'] = ic.Checker(env, params=True, fuel=400_000_000, prop_elim=True)
    return c['ic']


def ic_check(c, term, stmt):
    """(accepted, message): does the independent checker take `term` for a proof of `stmt`?"""
    fx = c['fx']; k, w = fx.k, fx.w
    return indep(c).check_closed(parse_raw(k.dump_raw(term.n), True), parse_raw(k.dump_raw(w.term(stmt, 'statement').n), True))


GOLDEN = os.path.join(itp_paths.GOLDEN, 's40_golden.json')
PINNED_KINDS = {'REWRITE': 534, 'CASES': 346, 'APPLY': 340, 'CLAIM': 217, 'LINARITH': 190, 'APPLY_WITH': 187, 'FORWARD': 155, 'RING': 140, 'USE': 132, 'REFINE': 115, 'REWRITE_AT': 96, 'INDUCT_WITH': 75,
                'LINCOMB': 46, 'INDUCT': 38, 'REWRITE_WITH': 37, 'INTERVAL_CASES': 15, 'NLINARITH': 14}       # (the 453 proofs; the 398 older ones gave REWRITE 425, APPLY 303, CASES 294, ... and the 55 added by the library extension the rest)
PINNED_OPS = {'rw': 571, 'apply': 527, 'have': 375, 'cases': 346, 'linarith': 190, 'ring': 140, 'exists': 132, 'exact': 113, 'rw_at': 96, 'induction_on': 75, 'linear_combination': 46, 'induct': 38, 'interval_cases': 15, 'nlinarith': 14}


def demo_digest(k, steps):
    """a short digest of the portable actions of one proof, the global ids replaced by their names (so that a lemma added to the library does not change it)"""
    def tok(t): return [t[0], k.global_info(t[1])[0]] + list(t[2:]) if t and t[0] == 'c' else list(t)
    canon = []
    for d in steps:
        d = dict(d)
        if 'source' in d: d['source'] = tok(d['source'])
        if 'codes' in d: d['codes'] = [[tok(t) for t in code] for code in d['codes']]
        canon.append(d)
    return hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:10]


def sec_F(c):
    fx = c['fx']; w, k, eng = fx.w, fx.k, fx.eng; res = fx.results; recs = fx.recs
    ok = sorted(nm for nm, r in res.items() if r.ok); bad = {nm: r for nm, r in res.items() if not r.ok}
    # ---- F1. coverage
    expect(len(res) == len(recs) and len(res) >= 400, 'F1 every recorded proof of the library was replayed (%d)' % len(res))
    expect(len(ok) >= 453 and set(bad) <= set(KNOWN_GAPS), 'F1 all but the six known proofs are replayed as kernel-checked action sequences (%d of %d; not replayed: %s)' % (len(ok), len(res), sorted(bad)))
    expect(all(r.gap.reason == KNOWN_GAPS.get(nm) and r.gap.detail and r.where and r.view for nm, r in bad.items()), 'F1 a proof that is not replayed says why (the documented reason), where, and shows both goals (%s)' % {nm: (r.gap.reason, r.where) for nm, r in bad.items()})
    expect(sum(r.soft['target_differs'] for r in res.values()) == 0, 'F1 the engine\'s goal was always the author\'s')
    # ---- F2. what a result records
    wrong = []; steps = 0; empty = 0
    for nm in ok:
        r = res[nm]; steps += len(r.actions); empty += (not r.actions)
        if not (r.gap is None and len(r.actions) == len(r.portable) == sum(r.kinds.values()) and all(isinstance(d, dict) and isinstance(d.get('kind'), str) for d in r.portable)
                and [Kind(a.kind).name for a, _ in r.actions] == [d['kind'] for d in r.portable] and (not r.actions or sum(r.ops.values()) > 0)): wrong.append(nm)
    expect(not wrong, 'F2 an action sequence has one portable form, one signature and one kind per action (%s)' % (wrong[:4],))
    expect(steps >= 1900 and empty <= 60, 'F2 the sequences are long enough to mean something (%d actions; %d proofs need none: settling closes them)' % (steps, empty))
    ops_seen = {n.op for rec in recs.values() for n in nodes_of(rec)}
    expect(ops_seen - {'intro'} <= set(REC.TRANSLATORS), 'F2 every call the author makes has a translation, but `intro` (which settling does) (%s)' % sorted(ops_seen - {'intro'} - set(REC.TRANSLATORS)))
    expect(len(ops_seen) >= 15, 'F2 the library uses many kinds of call (%d)' % len(ops_seen))
    # ---- F3. the same recording gives the same actions, in another engine
    eng2 = G.Engine(w, kinds=set(G.Kind), strict=True); diff = []
    for nm in ok[::13]:
        r2 = REC.replay(eng2, recs[nm])
        if not (r2.ok and r2.portable == res[nm].portable and [sg for _, sg in r2.actions] == [sg for _, sg in res[nm].actions] and r2.kinds == res[nm].kinds): diff.append(nm)
    expect(not diff, 'F3 a recording gives the same actions, the same signatures and the same kinds in another engine (%d replayed; different: %s)' % (len(ok[::13]), diff[:4]))
    # ---- F4. the proofs by actions are proofs: the independent checker takes them, and takes none for another statement
    eng3 = G.Engine(w, kinds=set(G.Kind), strict=True)
    heavy = ('imo2019_p1_fwd', 'imo2019_p1_bwd', 'Z.sqrt2_aux', 'Z.exists_divmod')                    # (the independent checker takes 3 to 15 s for each: S40_HEAVY=1)
    named = [nm for nm in ('imo1959_p1', 'Z.prime_dvd_sq', 'Z.coprime_symm', 'N.beq_eq', 'Z.add_wd', 'Pos.to_nat_add') + (heavy if os.environ.get('S40_HEAVY') else ()) if nm in res and res[nm].ok]
    sample = named + [nm for nm in ok[::17] if nm not in named and nm not in heavy]
    other = recs['Z.add_comm'].stmt; n_ic = 0; bad_ic = []
    for nm in sample:
        rec = recs[nm]
        term = A.replay_portable(eng3, rec.stmt, res[nm].portable, rec.gid)
        good, msg = ic_check(c, term, rec.stmt); n_ic += 1
        if not good: bad_ic.append((nm, msg))
        if nm in named[:4] or nm in heavy[:1]:
            expect(not ic_check(c, term, other)[0], 'F4 the independent checker rejects the proof of %s as a proof of Z.add_comm' % nm)
    eng3.premises = None
    expect(n_ic >= 28 and not bad_ic, 'F4 the independent checker accepts the proofs by actions of %d theorems (%s)' % (n_ic, bad_ic[:2]))
    # ---- F5. the report
    out = io.StringIO(); stats = REC.report(res, out); text = out.getvalue()
    expect(stats['n'] == len(res) and stats['ok'] == len(ok) and stats['actions'] == steps and stats['gaps'] == dict(collections.Counter(r.gap.reason for r in bad.values())), 'F5 the report counts the proofs, the actions and the gaps (%s)' % ({x: stats[x] for x in ('n', 'ok', 'actions', 'gaps')},))
    expect('replayed as kernel-checked action sequences: %d of %d' % (len(ok), len(res)) in text and ('gaps:' in text) == bool(bad) and all(nm in text for nm in bad), 'F5 ... and says so, with the proofs that are not replayed')
    expect(sum(v[0] for v in stats['groups'].values()) == len(res) and sum(v[1] for v in stats['groups'].values()) == len(ok) and sum(stats['kinds'].values()) == steps, 'F5 ... by part of the library and by kind of action')
    demos = REC.demonstrations(res)
    expect(len(demos) == len(ok) and all(set(d) == {'name', 'gid', 'steps'} for d in demos) and json.loads(json.dumps(demos)) == demos and sum(len(d['steps']) for d in demos) == steps, 'F5 the demonstrations are the replayed proofs as plain data (JSON)')
    # ---- F6. the demonstrations are what they were: how often each kind of action is used (a change in the recorder's choice of plan moves these), and the action sequence of each proof
    kinds_ = stats['kinds']; ops_ = collections.Counter()
    for nm in ok: ops_.update(res[nm].ops)
    off = {kd: (kinds_.get(kd, 0), want) for kd, want in PINNED_KINDS.items() if abs(kinds_.get(kd, 0) - want) > max(2, want // 10)}
    off_ops = {op: (ops_.get(op, 0), want) for op, want in PINNED_OPS.items() if abs(ops_.get(op, 0) - want) > max(2, want // 10)}
    n_compact = sum(1 for nm in ok for d in res[nm].portable if d['kind'] == 'FORWARD' and any(len(cd) == 0 for cd in d['codes'][1:]))
    expect(not off and set(kinds_) <= set(PINNED_KINDS) and not off_ops and set(ops_) <= set(PINNED_OPS), 'F6 the demonstrations use each kind of action and translate each kind of call as often as they did, within 10%% (actions %s; calls %s)' % (off, off_ops))
    expect(abs(n_compact - 86) <= 8, 'F6 FORWARD is said in its compact form (the arguments the others determine left out) as often as it was (%d of %d)' % (n_compact, kinds_.get('FORWARD', 0)))
    digests = {nm: demo_digest(k, res[nm].portable) for nm in ok}
    if os.environ.get('S40_UPDATE_GOLDEN'):
        with open(GOLDEN, 'w') as f_: json.dump(digests, f_, indent=0, sort_keys=True)
        print('   golden digests written: %d proofs' % len(digests))
    gold = json.load(open(GOLDEN)) if os.path.exists(GOLDEN) else {}
    diff = sorted(nm for nm in set(gold) | set(digests) if gold.get(nm) != digests.get(nm))
    expect(not diff, 'F6 the action sequences of the library\'s proofs are those in %s: %d of %d differ (%s) -- if the change is intended: S40_UPDATE_GOLDEN=1 python3 s40_actions_tests.py' % (os.path.basename(GOLDEN), len(diff), len(digests), diff[:5]))
    # ---- F7. a replay checks its proof with the kernel, against the statement of the theorem, and uses the library that precedes the theorem
    engv = G.Engine(w, kinds=set(G.Kind), strict=True); seen = {'verify': [], 'premises': set()}
    orig_verify, orig_menu = engv.verify, engv.menu
    def spy_verify(stmt, p): seen['verify'].append(stmt); return orig_verify(stmt, p)
    def spy_menu(g): seen['premises'].add(engv.premises); return orig_menu(g)
    engv.verify = spy_verify; engv.menu = spy_menu
    rec = recs['Z.sqrt2_aux']; r = REC.replay(engv, rec)
    expect(r.ok and len(seen['verify']) == 1 and seen['verify'][0] is rec.stmt, 'F7 a replayed proof is checked by the kernel once, against the statement of the theorem')
    expect(seen['premises'] == {rec.gid}, 'F7 ... and the menus of the replay are those of the library that precedes the theorem (%s)' % (seen['premises'],))
    def refuse_verify(stmt, p): raise AuthorError('the kernel refuses this proof')
    engv.verify = refuse_verify
    r = REC.replay(engv, rec)
    expect(not r.ok and r.gap is not None, 'F7 a proof that the kernel refuses is not a replayed proof')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ G. names
def sec_G(c):
    fx = c['fx']; w, k = fx.w, fx.k; S = statements(w); gid = fx.gid
    Z = w['Z']; F = w.forall_; Eq = w.Eq
    with w.nota.using(Z):
        lit = w.nota.lit
        fn = F([w.arrow(Z, Z), lambda f: F([Z], lambda x: Eq(f(x), x))], lambda f, h: Eq(f(lit(2)), lit(2)))          # (a hypothesis to instantiate)
    add_comm = ref(gid['Z.add_comm'])
    for tag in ('', 'e_', 'zz'):
        eng = G.Engine(w, kinds=set(G.Kind), strict=True, terms=True); eng.name_tag = tag
        ep = G.Episode(eng, S['lc'])
        expect(ep.goal().hyp_names == [tag + 'a%d' % i for i in range(4)], 'G (tag %r) the quantifiers that are introduced are named a<position> (%s)' % (tag, ep.goal().hyp_names))
        ep.step(act(Kind.REWRITE_AT, add_comm, hyps=(2,)))
        expect(ep.goal().hyp_names == [tag + n for n in ('a0', 'a1', 'a3', 'r4')], 'G (tag %r) REWRITE_AT replaces the hypothesis by r<count> (%s)' % (tag, ep.goal().hyp_names))
        ep.step(act(Kind.REWRITE_AT, add_comm, hyps=(3,)))
        expect(ep.goal().hyp_names[-1] == tag + 'r4_', 'G (tag %r) a name that is taken gets an underscore (%s)' % (tag, ep.goal().hyp_names))
        ep.step(act(Kind.REWRITE_AT, add_comm, hyps=(3,)))
        expect(ep.goal().hyp_names[-1] == tag + 'r4', 'G (tag %r) ... and the underscore goes when the name is free again (%s)' % (tag, ep.goal().hyp_names))
        ep = G.Episode(eng, S['sq']); ep.step(act(Kind.FORWARD, ref(gid['Z.mul_comm']), hyp(0), hyp(0)))
        expect(ep.goal().hyp_names == [tag + 'a0', tag + 'i1'], 'G (tag %r) FORWARD names the fact i<count> (%s)' % (tag, ep.goal().hyp_names))
        ep = G.Episode(eng, fn); m = ep.menu(); ins = [a for a in m.actions if a.kind == Kind.INSTANTIATE]
        if expect(ins, 'G (tag %r) a goal with a hypothesis to instantiate' % tag):
            ep.step(ins[0])
            expect(ep.goal().hyp_names == [tag + 'a0', tag + 'a1', tag + 'i2'], 'G (tag %r) INSTANTIATE names the fact i<count> (%s)' % (tag, ep.goal().hyp_names))
        ep = G.Episode(eng, S['ind2'])
        ep.step(act(Kind.INDUCT_WITH, ref(gid['Z.induction_from']), (('n', 1),), hyps=(0,)))
        names = [eng.goal_of(ep.p, i).hyp_names for i in range(ep.n_goals)]
        expect(len(names) == 2 and all(len(set(nm)) == len(nm) for nm in names) and all(x.startswith(tag) for nm in names for x in nm), 'G (tag %r) after an induction the names are all different, and all have the tag (%s)' % (tag, names))
    # the hypotheses of a goal get different names (the tactic layer lets a name be used twice, as the author's scripts do): the later one gets underscores, and the kernel is told
    from tactics import Proof
    Zt = w['Z'].n; tgt = w['Z.le'](w['Z.zero'], w['Z.zero']).n
    for names, count, want in ((['x', 'x', 'x_', 'y'], 1, ['x', 'x__', 'x_', 'y']), (['x', 'x', 'x', 'x'], 1, ['x', 'x_', 'x__', 'x___']), (['x', 'x', 'y', 'y'], 0, ['x', 'x', 'y', 'y'])):
        p_ = Proof.from_goal(w, tgt, [Zt] * 4, names); fx.eng._unique_names(p_, count)
        got = list(p_._names[p_.ps.goal_meta_id(0)]); kern = [p_.ps.goal_hyp_name(0, h) for h in range(4)]
        expect(got == want and kern == want, 'G the names %s of the first %d goals become %s, in the engine and in the kernel (%s, %s)' % (names, count, want, got, kern))
    # the recorder names what it introduces with its own tag, and gives the engine back as it was
    eng = fx.eng; eng.name_tag = ''; seen = collections.Counter(); orig = eng._intro_at
    def spy(p, i): seen[eng.name_tag] += 1; return orig(p, i)
    eng._intro_at = spy
    try: r = REC.replay(eng, fx.recs['Z.add_comm'])
    finally: del eng._intro_at
    expect(r.ok and set(seen) == {REC.TAG} and eng.name_tag == '', 'G the recorder introduces with the tag %r, and restores the engine\'s (%s)' % (REC.TAG, dict(seen)))
    eng.name_tag = 'zz_'
    r2 = REC.replay(eng, fx.recs['Z.induction_on'])
    expect(not r2.ok and eng.name_tag == 'zz_', 'G ... also when the replay fails')
    r3 = REC.replay(eng, fx.recs['Z.add_comm'])
    expect(r3.ok and eng.name_tag == 'zz_' and r3.portable == fx.results['Z.add_comm'].portable, 'G ... and the actions do not depend on the tag the engine had')
    eng.name_tag = ''; eng.premises = None


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ H. free play
TRUE_PLAY = ('amgm', 'trans', 'ex49', 'conj', 'sq', 'rwg', 'rwa', 'ind', 'ind2', 'ind_free', 'inj', 'injz', 'lc', 'lc_le', 'mul_one', 'ind_motive')
SECOND_CODE_KINDS = (Kind.APPLY_WITH, Kind.REWRITE_WITH, Kind.REWRITE_AT, Kind.FORWARD, Kind.INDUCT_WITH, Kind.LINCOMB, Kind.USE, Kind.CLAIM, Kind.REFINE)


def false_statements(w):
    Z = w['Z']; F = w.forall_; Eq = w.Eq; zle = w['Z.le']; Ex = w['Ex']; Nat = w['Nat']
    out = {}
    with w.nota.using(Z):
        lit = w.nota.lit
        out['idem'] = F([Z], lambda a: Eq(a * a, a))
        out['addmul'] = F([Z, Z], lambda a, b: Eq(a + b, a * b))
        out['nonneg'] = F([Z], lambda a: zle(w['Z.zero'], a))
        out['pred'] = F([Z], lambda a: zle(a, a - lit(1)))
        out['ex50'] = Ex(Z, w.lam(Z, lambda x: Eq(x * x, lit(50))))
        out['nat_eq'] = F([Nat, Nat], lambda n, m: Eq(n, m))
        out['lin'] = F([Z, Z, lambda a, b: Eq(a + b, lit(5))], lambda a, b, h: Eq(a - b, lit(1)))
    return out


def rand_slot(rng, g, lemmas):
    r = rng.random()
    if r < 0.25: return ()
    if r < 0.55 and g.n_hyps: return hyp(rng.randrange(g.n_hyps))
    if r < 0.85: return (('n', rng.choice((0, 1, 2, 3, 7, -1))),)
    return ref(rng.choice(lemmas))


def rand_source(rng, g, lemmas):
    if g.n_hyps and rng.random() < 0.35: return hyp(rng.randrange(g.n_hyps))
    return ref(rng.choice(lemmas))


def mutate_action(a, rng, g, lemmas):
    """a listed action with one thing changed: its source, a slot, its mode, a hypothesis"""
    codes = list(a.codes); hyps = list(a.hyps); arg = a.arg; r = rng.randrange(5)
    if r == 0 and codes: codes[0] = rand_source(rng, g, lemmas)
    elif r == 1 and codes: codes.pop()
    elif r == 2: codes.append(rand_slot(rng, g, lemmas))
    elif r == 3 and a.kind in A.CODE_KINDS: arg = 1 - arg if arg in (0, 1) else 0
    elif r == 4 and hyps: hyps[rng.randrange(len(hyps))] = rng.randrange(max(g.n_hyps, 1))
    return Action(a.kind, arg, a.args, tuple(codes), tuple(hyps))


def random_action(eng, g, m, rng, lemmas, principles):
    """a random action for the goal: one of the menu's, one of the menu's changed a little, a menu's APPLY or REWRITE said in the second language, or one of any kind said with random arguments
    (CLAIM and REFINE by a random walk of the builder, LINCOMB over the equations)"""
    k = eng.k; r = rng.random(); nh = max(g.n_hyps, 1)
    by_kind = collections.defaultdict(list)
    for a in m.actions: by_kind[a.kind].append(a)
    pick = lambda: rng.choice(by_kind[rng.choice(sorted(by_kind))])                 # (a kind first, then an action: INSTANTIATE would be most of the menu)
    if r < 0.35 and m.actions: return pick()
    if r < 0.45 and m.actions: return mutate_action(pick(), rng, g, lemmas)
    if r < 0.6:
        cands = [a for a in m.actions if a.kind in (Kind.APPLY, Kind.REWRITE)]
        if cands:
            a = rng.choice(cands); term, rev, _ = m.cands[a.arg]
            try: tok = A.source_token(eng, term)
            except G.ActionError: tok = None
            if tok is not None:
                if a.kind == Kind.APPLY: return Action(Kind.APPLY_WITH, rng.randrange(2), (), ((tok,),), ())
                return Action(Kind.REWRITE_WITH, int(bool(rev)), (), ((tok,),), ())
    kind = rng.choice(SECOND_CODE_KINDS + (Kind.INJECT, Kind.CASES, Kind.INDUCT))
    src = rand_source(rng, g, lemmas); slots = tuple(rand_slot(rng, g, lemmas) for _ in range(rng.randrange(0, 4)))
    if kind in (Kind.APPLY_WITH, Kind.REWRITE_WITH, Kind.FORWARD): return Action(kind, rng.randrange(2) if kind != Kind.FORWARD else 0, (), (src,) + slots, ())
    if kind == Kind.REWRITE_AT: return Action(kind, rng.randrange(2), (), (src,) + slots, (rng.randrange(nh),))
    if kind == Kind.INDUCT_WITH: return Action(kind, 0, (), (ref(rng.choice(principles)),) + slots[:1], (rng.randrange(nh),))
    if kind == Kind.INJECT: return Action(kind, rng.randrange(nh))
    if kind in (Kind.CASES, Kind.INDUCT): return Action(kind, rng.randrange(nh), (), (), tuple(rng.sample(range(nh), min(nh, rng.randrange(0, 3)))))
    if kind == Kind.LINCOMB:
        eqs = [h for h in range(g.n_hyps) if is_equation(eng, g.hyp_types[h])] or [0]
        hs = rng.sample(eqs, min(len(eqs), rng.randrange(1, 3))); return Action(kind, 0, (), tuple((('n', rng.choice((1, 1, 2, -1, 3))),) for _ in hs), tuple(hs))
    if kind == Kind.USE: return Action(kind, 0, (), ((slots[0] if slots and slots[0] else (('n', rng.randrange(0, 9)),)),), ())
    sp = eng.space(g)
    b = sp.builder(sp.prop) if kind == Kind.CLAIM else eng.builder(g, goals=True)
    b, n, _ = walk_term(b, rng, max_tokens=12)
    if b.done: return Action(kind, 0, (), (tuple(b.code),), ())
    return pick() if m.actions else Action(Kind.INJECT, 0)


def is_equation(eng, ty):
    k = eng.k; head, args = GB.spine(k, ty); v = k.view(head)
    return v[0] == 'const' and v[1] == eng.w.gid['Eq'] and len(args) == 3


def play(c, eng, stmt, rng, steps, tally, lemmas, principles, deadline, label='-'):
    """one episode of random play; the invariants are counted in `tally`. Returns the episode."""
    k, w = c['fx'].k, c['fx'].w
    ep = G.Episode(eng, stmt)
    for _ in range(steps):
        if ep.done or not ep.n_goals or time.time() > deadline: break
        g = ep.goal(); m = ep.menu(); names = g.hyp_names
        tally['goals'] += 1; tally['names not unique'] += (len(set(names)) != len(names))
        a = random_action(eng, g, m, rng, lemmas, principles)
        s_before = state(k, w); np0 = eng.stats['probe_noprogress']
        out, st = eng.probe_status(g, a, m); tally['status ' + st] += 1; tally['kind ' + Kind(a.kind).name] += 1
        pre = (state(k, w), ep.n_goals, g.key, tuple(names))                         # (the step probes once more: that one must leave nothing)
        if st != G.ST_OK:
            why = refusal_effect(eng, s_before, pre[0], np0)
            if why: tally['a refused probe changed the kernel'] += 1; c.setdefault('examples', []).append((label, repr(a), why))
        try: ep.step(a)
        except G.ActionError:
            tally['refused'] += 1; tally['probe ok, step refused'] += (out is not None)
            g2 = ep.goal(); post = (state(k, w), ep.n_goals, g2.key, tuple(g2.hyp_names))
            if pre != post:
                tally['a refusal changed something'] += 1; c.setdefault('examples', []).append((label, repr(a), [(x, y) for x, y in zip(pre, post) if x != y][:2]))
        except Exception as e:
            tally['step raised ' + type(e).__name__] += 1; break
        else:
            tally['accepted'] += 1; tally['probe refused, step accepted'] += (out is None); tally['accepted ' + Kind(a.kind).name] += 1
            s1 = state(k, w); tally['accepted step changed globals, fuel or probing'] += (pre[0][1:] != s1[1:])
            if out is not None:
                now = goals_of(eng, ep)
                tally['goals not as predicted'] += not ([x.key for x in now[:len(out)]] == [x.key for x in out] and ep.n_goals == pre[1] - 1 + len(out))
    return ep


def sec_H(c):
    fx = c['fx']; w, k = fx.w, fx.k; gid = fx.gid; S = statements(w); Fs = false_statements(w)
    lemmas = [gid[nm] for nm in sorted(fx.recs) if nm.startswith('Z.') and nm in gid]
    principles = [gid[nm] for nm in ('Z.induction_from', 'Z.int_induction', 'Z.strong_induction', 'N.ind', 'Z.induction_on') if nm in gid]
    eng = G.Engine(w, kinds=set(G.Kind), strict=True, terms=True, inst_lemmas=tuple(nm for nm in ('Z.le_trans', 'Z.add_comm', 'Z.mul_comm', 'Z.add_assoc', 'Z.le_refl', 'Z.sq_nonneg', 'Z.add_left_cancel') if nm in gid))
    runs = int(os.environ.get('S40_RUNS', '4')); steps = 14; deadline = time.time() + float(os.environ.get('S40_PLAY_SECONDS', '60'))
    tally = collections.Counter(); finished = []; rng = random.Random(40)
    t0 = time.time(); per = {}
    for label, stmt, true in [(nm, S[nm], True) for nm in TRUE_PLAY] + [(nm, Fs[nm], False) for nm in Fs]:
        t1 = time.time(); t = collections.Counter()
        for r in range(runs):
            ep = play(c, eng, stmt, rng, steps, t, lemmas, principles, deadline, label)
            if ep.done:
                t['finished'] += 1; term = ep.finish()
                if true: finished.append((label, stmt, term))
                else: t['FALSE STATEMENT FINISHED'] += 1
        tally.update(t); per[label] = time.time() - t1
    trace('H play %.1f s: slowest %s' % (time.time() - t0, sorted(per.items(), key=lambda x: -x[1])[:4]))
    trace('H tally %s' % sorted(tally.items()))
    expect(tally['goals'] >= 300 and tally['accepted'] >= 120 and tally['refused'] >= 120, 'H the play is long and both accepts and refuses (%s)' % ({x: tally[x] for x in ('goals', 'accepted', 'refused', 'finished', 'status ok', 'status refused', 'status fuel', 'status bug')},))
    kinds_seen = {Kind[x[5:]] for x in tally if x.startswith('kind ')}
    expect(all(kd in kinds_seen for kd in SECOND_CODE_KINDS + (Kind.INJECT, Kind.CASES, Kind.INDUCT)), 'H ... with actions of every kind of the second language (missing: %s)' % sorted(kd.name for kd in set(SECOND_CODE_KINDS + (Kind.INJECT,)) - kinds_seen))
    expect(tally['a refused probe changed the kernel'] == 0, 'H a refused probe leaves the kernel as it was (but for the goal of an action that made no progress) (%s)' % (c.get('examples', ['-'])[:1],))
    expect(tally['a refusal changed something'] == 0, 'H a refused action changes nothing: no goal, no name, no node of the kernel (%d refusals; %s)' % (tally['refused'], c.get('examples', ['-'])[:1]))
    expect(tally['probe ok, step refused'] == 0 and tally['probe refused, step accepted'] == 0, 'H the probe and the step agree on what is legal')
    expect(tally['goals not as predicted'] == 0 and tally['accepted step changed globals, fuel or probing'] == 0, 'H an accepted step leaves exactly the goals the probe predicted, and changes nothing else (%d accepted)' % tally['accepted'])
    expect(tally['names not unique'] == 0, 'H the hypotheses of every goal have different names (%d goals)' % tally['goals'])
    expect(tally['status bug'] == 0 and not eng.bugs and not [x for x in tally if x.startswith('step raised')], 'H nothing is mistaken for a refusal: no bug, no other exception (%s)' % ({x: v for x, v in tally.items() if x.startswith('step raised')},))
    expect(tally['FALSE STATEMENT FINISHED'] == 0, 'H no false statement is proved')
    expect(len(finished) >= 8, 'H proofs of true statements are found by random play and finished (%d)' % len(finished))
    for label, stmt, term in finished[:10]:
        expect(ic_check(c, term, stmt)[0], 'H the independent checker accepts the proof of %s that random play found' % label)


SECTIONS = [('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E), ('F', sec_F), ('G', sec_G), ('H', sec_H)]


def main():
    t_start = time.time()
    c = {'fx': Fixture()}
    only = os.environ.get('S40_ONLY')
    for name, fn in SECTIONS:
        if only and name not in only.split(','): continue
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:200])); traceback.print_exc(limit=6)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
        if FAIL and os.environ.get('S40_FAILFAST'): break           # (for mutation testing, where one failure is all that is wanted)
    fx = c['fx']
    expect(not fx.eng.bugs and fx.eng.stats['probe_bug'] == 0, 'Z  the engine of the gate met no bug: every exception inside a probe was a refusal (%s)' % (fx.eng.bugs[:1],))
    print('S40: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:60]: print('  FAIL:', f)
    print('S40 ACTIONS GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
