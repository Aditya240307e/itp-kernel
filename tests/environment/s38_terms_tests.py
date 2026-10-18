"""S38 gate: the actions that carry terms (INSTANTIATE, USE), the term pool they draw from, probes with a chosen work budget, goal identity up to irrelevant order, and the search on cost.

A learner without a text interface still has to say WHICH term: the value to put into `forall a b, f (a + b) = f a + f b`, the witness of `exists x, ...`. The term pool (gym_terms.py) is the answer for the
number types: a deterministic numbered list of well-typed terms of the goal's context, and an action names its terms by position, so it cannot be ill-typed or mean something else on another machine.
Nothing here is trusted -- an action is carried out by the tactic layer and everything is re-checked by the kernel -- so what this gate pins down is that the machinery does what its documentation says:

  A. THE POOL.  Determinism (same goal, same list, from any engine); every term has the type the pool says; no two terms of a type are equal as polynomials; the content and order are the documented ones
     (variables, small numerals, the numerals of the goal, sums and differences, doubles, squares, values of the context's functions); naturals get no negatives and no subtraction; the pool is bounded.
  B. USE.  Offered on existential goals (also behind a definition: `a | b`), with the pool's terms of the right type; applying one leaves `P witness`, with no metavariable in the goal; bad indices, a
     term of the wrong type and a goal that is no existential are refused and change nothing.
  C. INSTANTIATE.  A hypothesis (or an allowed lemma) applied to chosen terms becomes a new hypothesis whose statement is the instance; premises stay premises; a fact that is already there is neither offered
     nor accepted; the bounds on proposals hold; wrong numbers of terms are refused.
  D. THE MENU WITH TERMS.  Every action is found again from its signature (the signature names the terms); kinds can be restricted; without `terms` no pool is built; random play with term actions keeps the
     invariants of gate S36: the live step leaves the goals the probe predicted, refused steps change nothing, nothing is mistaken for a bug.
  E. PROBES WITH A BUDGET.  `probe_status` tells a refusal from running out of work; the outcome of an action never depends on the budget except by running out of it (monotone); the budget levels of a kind;
     the kernel is left as it was in every case.
  F. GOAL IDENTITY.  `canon_key` ignores the order of hypotheses that nothing refers to, and nothing else.
  G. `exists` leaves no metavariable behind and still works through definitions.
  H. THE SEARCH.  Witness problems and instances of hypotheses and lemmas are solved by the plain and by the cost search (proofs accepted by the kernel and by the independent checker); a false existential
     is not proved; the cost search honours its budgets, is deterministic, remembers what probes told it, ends exactly when the space is exhausted, and with unit costs finds at cost d what deepening finds
     at depth d.
  I. ARGUMENTS ARE VALIDATED.  An action whose kind, index, hypothesis, candidate or terms mean nothing in the menu is refused with `ActionError` (never an exception from the inside of the engine), for
     every kind, and changes nothing.
  J. REWRITING COSTS.  Equations that only reorder a term are marked permutative (in the profile and in the menu); the default cost of an action counts its kind, its terms' places in the pool, the library
     and permutative rewrites.
  K. THE LEMMAS `apply` OFFERS.  A lemma about another type than the goal's is not offered (and none that would have applied is lost); the introduction rules of the connectives are told from the other
     lemmas; applying a lemma of the library costs a step more than applying a hypothesis or an introduction rule.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, sys, time

sys.setrecursionlimit(200000)
import itp_env_lnl as E
import nat_lib, int_lib, ring as R
import gym_env as G, gym_search as S, gym_terms as GT
from author import AuthorError
from tactics import Proof
import indep_checker as ic
from harness import parse_raw

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S38_TRACE'))
def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond
REFUSALS = (G.ActionError, AuthorError, E.ITPError)
def refuses(what, fn, exc=REFUSALS):
    try: fn()
    except exc: return expect(True, what)
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:80]))
    return expect(False, what + ' (it raised nothing)')


def state(k, w):
    """everything a probe must leave as it was"""
    return (tuple(k.pool_sizes()), k.num_globals(), k.get_fuel(), w.probing)


def find(m, kind, pred=lambda a: True):
    return [a for a in m.actions if a.kind == kind and pred(a)]


def labels(m, a): return tuple(m.pool.labels[i] for i in a.args)


def has_meta(k, node):
    """does the kernel term contain a metavariable?"""
    tags = k.observe_term(node)[0]
    return bool((tags == G.TAG['META']).any())


def use_with(m, label):
    """the USE action whose witness has this label in the pool, or None"""
    return next((a for a in find(m, G.Kind.USE) if m.pool.labels[a.args[0]] == label), None)


def inst_with(m, source, lab):
    """the INSTANTIATE action of the candidate `source` (a kernel handle) at the terms with these labels, or None"""
    return next((a for a in find(m, G.Kind.INSTANTIATE) if m.cands[a.arg][0] == source and labels(m, a) == tuple(lab)), None)


def inst_sig(k, w, m, a):
    """a readable name of an INSTANTIATE: (the source, the labels of its terms)"""
    src = k.view(m.cands[a.arg][0])
    return ('hyp%d' % src[1] if src[0] == 'fvar' else w.gname.get(src[1], '?'), labels(m, a))


class Fixture:
    """the integer world and the goals the tests use"""
    def __init__(self):
        self.w = w = int_lib.build_int(); self.k = w.k; self.Z = Z = w['Z']; self.N = w['N']; self.Nat = w['Nat']
        F = w.forall_; Eq = w.Eq; Ex = w['Ex']; zle = w['Z.le']; PROP = w.PROP
        self.ZZ = ZZ = w.arrow(Z, Z)
        with w.nota.using(Z):
            lit = w.nota.lit
            w.define('Z.dvd', w.forall_([Z, Z], lambda a, b: PROP), w.fun([Z, Z], lambda a, b: Ex(Z, w.lam(Z, lambda k_: Eq(b, a * k_)))))      # as in nt_lib
            self.dvd = w['Z.dvd'](lit(3), lit(12))                                                                                           # 3 | 12
            self.sq49 = Ex(Z, w.lam(Z, lambda x: Eq(x * x, lit(49))))                                                                       # exists x, x * x = 49
            self.sq50 = Ex(Z, w.lam(Z, lambda x: Eq(x * x, lit(50))))                                                                       # false
            self.any_w = Ex(Z, w.lam(Z, lambda x: Eq(x + lit(0), x)))                                                                       # every numeral is a witness
            self.pair = Ex(Z, w.lam(Z, lambda a: Ex(Z, w.lam(Z, lambda b: w.And(Eq(a + b, lit(5)), Eq(a - b, lit(1)))))))                  # exists a b, a + b = 5 /\ a - b = 1
            self.lin = F([ZZ, lambda f: F([Z], lambda n: Eq(f(n), 2 * n + 1))], lambda f, h: Eq(f(lit(3)) + f(lit(3)), lit(14)))            # (forall n, f n = 2 n + 1) -> f 3 + f 3 = 14
            self.add = F([ZZ, lambda f: F([Z, Z], lambda a, b: Eq(f(a + b), f(a) + f(b)))], lambda f, h: Eq(f(lit(3)), 3 * f(lit(1))))     # f additive -> f 3 = 3 f 1
            self.add2 = F([ZZ, lambda f: F([Z, Z], lambda a, b: Eq(f(a + b), f(a) + f(b)))], lambda f, h: Eq(f(lit(2)), 2 * f(lit(1))))     # f additive -> f 2 = 2 f 1
            self.amgm = F([Z, Z], lambda a, b: zle(2 * a * b, a * a + b * b))                                                              # 2 a b <= a a + b b
            self.vars = F([Z, Z, ZZ], lambda a, b, f: Ex(Z, w.lam(Z, lambda x: Eq(f(x), a * b + lit(17)))))                                # a goal with variables and a function
            self.prem = F([ZZ, lambda f: F([Z, lambda a: zle(lit(0), a)], lambda a, h: Eq(f(a), a))], lambda f, h: Eq(f(lit(2)), lit(2)))   # (forall a, 0 <= a -> f a = a) -> f 2 = 2
            self.comm = F([Z, Z], lambda a, b: Eq(a + b, b + a))
            self.mix = F([Z, lambda x: zle(lit(0), x), lambda x, h0: zle(x, lit(3))], lambda x, h0, h1: Eq(x + lit(1), lit(1) + x))        # an interval and lemmas
            self.big = F([Z] * 9, lambda *xs: Ex(Z, w.lam(Z, lambda y: Eq(y, xs[0] + xs[8]))))                                              # nine variables
            self.mixed = F([self.N], lambda n: Ex(Z, w.lam(Z, lambda x: Eq(x, w['Z.of_N'](n)))))                                           # numbers of two types


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. the pool
def sec_A(c):
    fx = c['fx']; w, k, Z = fx.w, fx.k, fx.Z
    eng = G.Engine(w, terms=True, strict=True)
    ep = G.Episode(eng, fx.vars); g = ep.goal(); m = ep.menu(); pool = m.pool
    expect(len(pool) > 20 and len(pool.labels) == len(pool.terms) == len(pool.types), 'A  the pool of a goal with variables is not small (%d terms)' % len(pool))
    # determinism: another engine, the pool built again
    m2 = G.Episode(G.Engine(w, terms=True, strict=True), fx.vars).menu()
    expect(m2.pool.labels == pool.labels and m2.pool.terms == pool.terms, 'A  the pool is the same for another engine (same terms, same order)')
    with eng.in_context(g.hyp_types, g.hyp_names):
        again = GT.build_pool(w, g.hyp_types, g.hyp_names, [g.target] + list(g.hyp_types), {Z.n})
    expect(again.terms == pool.terms and again.labels == pool.labels and again.types == pool.types, 'A  building the pool again gives the same terms in the same order')
    # types: every term has the type the pool says, and of_type lists exactly its indices
    bad = []
    for i, (t, ty) in enumerate(zip(pool.terms, pool.types)):
        ok, got = k.infer_open(t, g.hyp_types)
        if not ok or got != ty: bad.append((i, pool.labels[i]))
    expect(not bad, 'A  every term of the pool has the type the pool says (%s)' % (bad[:3],))
    expect(sorted(i for ids in pool.of_type.values() for i in ids) == list(range(len(pool))) and all(pool.types[i] == ty for ty, ids in pool.of_type.items() for i in ids), 'A  of_type partitions the indices by type')
    expect(set(pool.of_type) <= {Z.n, fx.N.n, fx.Nat.n}, 'A  only number types are listed')
    # no two terms of the integers are equal as polynomials
    rg = R.Ring(w, R.instance_for(w, Z)); seen = {}; dup = []
    with eng.in_context(g.hyp_types, g.hyp_names):
        for i in pool.of_type.get(Z.n, ()):
            nf, _ = rg.nf(w.T(pool.terms[i])); key = tuple((mo.vars, mo.coeff) for mo in nf.monos)
            if key in seen: dup.append((pool.labels[seen[key]], pool.labels[i]))
            else: seen[key] = i
    expect(not dup and len(seen) == len(pool.of_type[Z.n]), 'A  no two integer terms of the pool are equal as polynomials (%s)' % (dup[:3],))
    # content and order: variables first, then the small numerals, sums and differences, doubles, ...; the values of the function after the base terms
    zl = [pool.labels[i] for i in pool.of_type[Z.n]]
    expect(zl[:2] == ['a0', 'a1'], 'A  the variables come first, in order (%s)' % (zl[:3],))
    expect(zl[2:8] == ['0', '1', '-1', '2', '3', '-2'], 'A  then the small numerals 0 1 -1 2 3 -2 (%s)' % (zl[2:9],))
    for want in ('(a0 + a1)', '(a0 - a1)', '(a1 - a0)', '(2 * a0)', '(a0 + 1)', '(a0 - 1)', '(a0 * a1)', '(a0 * a0)', '-a0', '17', '10', '4'):
        expect(want in zl, 'A  the pool has %s' % want)
    expect('(a1 + a0)' not in zl and '(a1 * a0)' not in zl and '(1 + a0)' not in zl, 'A  ... and no second spelling of a sum or a product')
    fvals = [l for l in zl if l.startswith('a2(')]
    expect(len(fvals) == GT.APPLY_TO and fvals[:3] == ['a2(a0)', 'a2(a1)', 'a2(0)'] and 'a2(17)' not in fvals, 'A  the context function is applied to the first %d base terms: %s' % (GT.APPLY_TO, fvals[:4]))
    expect(zl.index('(a0 + a1)') < zl.index(fvals[0]) < zl.index('(a0 * a1)'), 'A  function values come after the base terms and before the rest')
    expect(len(set(zl)) == len(zl), 'A  all labels differ')
    expect('1000' not in zl and zl.index('17') > zl.index('(a0 * a1)'), 'A  a numeral that does not occur is not offered unless it is small; the numerals of the goal come with the rest')
    # a goal without variables: the numerals only; the numerals of the goal come first after the small ones
    pz = G.Episode(eng, fx.sq49).menu().pool
    zl2 = [pz.labels[i] for i in pz.of_type[Z.n]]
    expect(zl2[:7] == ['0', '1', '-1', '2', '3', '-2', '49'] and zl2[7:14] == ['4', '5', '6', '7', '8', '9', '10'], 'A  without variables: 0 1 -1 2 3 -2, the numeral of the goal, then 4..10 (%s)' % (zl2[:14],))
    expect(all(l.lstrip('-').isdigit() for l in pz.labels), 'A  ... and nothing but numerals in the pool, for any type')
    # the bound, and the variables that are combined (the latest ones)
    eb = G.Engine(w, terms=True, strict=True); pb = G.Episode(eb, fx.big).menu().pool
    bl = [pb.labels[i] for i in pb.of_type[Z.n]]
    expect(0 < len(pb) <= GT.MAX_POOL, 'A  the pool is bounded (%d terms for nine variables)' % len(pb))
    expect(all(('a%d' % j) in bl for j in range(3, 9)) and not any(l in bl for l in ('a0', 'a1', 'a2')), 'A  only the latest %d variables of a type are combined (%s)' % (GT.MAX_VARS, bl[:8]))
    # naturals: no negative numerals, no subtraction, no negation
    w3 = c['w3']; Nat = w3['Nat']; F3 = w3.forall_; Ex3 = w3['Ex']; Eq3 = w3.Eq
    st3 = F3([Nat, Nat], lambda a, b: Ex3(Nat, w3.lam(Nat, lambda x: Eq3(w3['add'](x, a), b))))
    m3 = G.Episode(G.Engine(w3, terms=True, strict=True), st3).menu()
    nl = list(m3.pool.labels)
    expect(len(nl) > 10 and not any('-' in l for l in nl), 'A  the pool of the naturals has no subtraction and no negative numerals (%s)' % (nl[:12],))
    expect(nl[:2] == ['a0', 'a1'] and '(a0 + a1)' in nl and '(a0 * a1)' in nl, 'A  ... but variables, sums and products (%s)' % (nl[:12],))
    trace('A done')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. USE
def sec_B(c):
    fx = c['fx']; w, k, Z = fx.w, fx.k, fx.Z
    eng = G.Engine(w, terms=True, strict=True)
    ep = G.Episode(eng, fx.sq49); m = ep.menu(); g = ep.goal()
    uses = find(m, G.Kind.USE)
    zi = m.pool.of_type[Z.n]
    expect(len(uses) == min(G.MAX_USE, len(zi)) and [a.args[0] for a in uses] == zi[:len(uses)], 'B  USE is offered for the first %d terms of the pool of the right type, in order' % G.MAX_USE)
    expect(all(a.arg == 0 and len(a.args) == 1 for a in uses), 'B  ... each with one term')
    expect(not find(m, G.Kind.INSTANTIATE), 'B  no INSTANTIATE where nothing can be instantiated')
    lab = {m.pool.labels[a.args[0]]: a for a in uses}
    expect('7' in lab and '-7' not in lab and '0' in lab, 'B  7 is among the witnesses')
    s0 = state(k, w)
    ch = eng.probe(g, lab['7'], m)
    expect(ch == [] and state(k, w) == s0, 'B  USE 7 closes x * x = 49 (the instance is closed by computation) and the probe leaves the kernel as it was')
    ch0 = eng.probe(g, lab['0'], m)
    expect(ch0 is not None and len(ch0) == 1, 'B  USE 0 leaves the goal 0 * 0 = 49 (which is false, but a legal step)')
    if ch0:
        shown = eng.render(ch0[0]).split('|-')[1]
        expect('?' not in shown and 'Ex' not in shown and 'fun' not in shown, 'B  ... without a metavariable, without the quantifier and without a left-over lambda (%s)' % shown.strip()[:70])
    n = ep.step(lab['7'])
    expect(n == 0 and ep.done and ep.finish() is not None, 'B  the episode "exists x, x * x = 49" is finished by USE 7 and the kernel accepts the proof')
    # two witnesses and a conjunction
    ep2 = G.Episode(eng, fx.pair); m2 = ep2.menu()
    ep2.step(use_with(m2, '3')); m2b = ep2.menu(); ep2.step(use_with(m2b, '2'))
    expect(not ep2.done and ep2.n_goals == 1, 'B  after both witnesses the conjunction is left')
    m2c = ep2.menu(); cj = next(a for a in find(m2c, G.Kind.APPLY) if k.view(m2c.cands[a.arg][0]) == ('const', w.gid['conj']))
    n = ep2.step(cj)
    expect(n == 1, 'B  conj leaves 3 - 2 = 1 (3 + 2 = 5 is closed by computation, a difference of integers is not: Z is a quotient)')
    rg = find(ep2.menu(), G.Kind.RING)
    if expect(len(rg) == 1, 'B  ... which ring is offered for'):
        ep2.step(rg[0]); expect(ep2.done and ep2.finish() is not None, 'B  ... and closes: the pair (3, 2) is a proof')
    # a witness that is wrong is a legal step that leads nowhere
    ep3 = G.Episode(eng, fx.sq50); m3 = ep3.menu(); ep3.step(use_with(m3, '7'))
    expect(not ep3.done and ep3.n_goals == 1, 'B  USE 7 for x * x = 50 leaves 7 * 7 = 50, which is not closed')
    # through a definition: 3 | 12 is an existential
    ed = G.Episode(eng, fx.dvd); md = ed.menu()
    expect(len(find(md, G.Kind.USE)) > 0, 'B  USE is offered on a divisibility goal (an existential behind a definition)')
    a4 = use_with(md, '4')
    if expect(a4 is not None, 'B  ... among them 4'):
        ed.step(a4); expect(ed.done and ed.finish() is not None, 'B  ... and USE 4 proves 3 | 12')
    # refusals: no existential, bad indices, the wrong number of terms, a term of the wrong type
    el = G.Episode(eng, fx.lin); ml = el.menu(); kl = el.goal().key; sl = state(k, w)
    expect(not find(ml, G.Kind.USE) and len(ml.pool) > 0, 'B  no USE on a goal that is no existential (though its pool is not empty)')
    refuses('B  USE on a goal that is no existential is refused', lambda: el.step(G.Action(G.Kind.USE, 0, (0,))), exc=(G.ActionError,))
    expect(el.goal().key == kl and el.n_goals == 1 and state(k, w) == sl, 'B  ... and changes nothing')
    ep4 = G.Episode(eng, fx.sq49); m4 = ep4.menu(); key4 = ep4.goal().key; s4 = state(k, w)
    for what, act in (('an index outside the pool', G.Action(G.Kind.USE, 0, (len(m4.pool),))), ('a negative index', G.Action(G.Kind.USE, 0, (-1,))), ('no term', G.Action(G.Kind.USE, 0, ())),
                      ('two terms', G.Action(G.Kind.USE, 0, (0, 1))), ('a non-integer index', G.Action(G.Kind.USE, 0, ('0',)))):
        refuses('B  USE with %s is refused' % what, lambda act=act: ep4.step(act), exc=(G.ActionError,))
    expect(ep4.goal().key == key4 and ep4.n_goals == 1 and state(k, w) == s4, 'B  ... and each leaves the episode and the kernel as they were')
    em = G.Episode(eng, fx.mixed); mm = em.menu()
    n_idx = mm.pool.of_type.get(fx.N.n, [])
    if expect(len(n_idx) > 0 and Z.n in mm.pool.of_type, 'B  a goal about N and Z has terms of both types in its pool'):
        km = em.goal().key
        refuses('B  USE with a natural number as the witness of an integer existential is refused', lambda: em.step(G.Action(G.Kind.USE, 0, (n_idx[0],))), exc=(G.ActionError,))
        expect(em.goal().key == km, 'B  ... and changes nothing')
        expect(all(mm.pool.types[a.args[0]] == Z.n for a in find(mm, G.Kind.USE)), 'B  ... and only integers are offered')
    trace('B done')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. INSTANTIATE
def sec_C(c):
    fx = c['fx']; w, k, Z = fx.w, fx.k, fx.Z
    eng = G.Engine(w, terms=True, strict=True)
    ep = G.Episode(eng, fx.lin); g = ep.goal(); m = ep.menu()
    ins = find(m, G.Kind.INSTANTIATE)
    expect(0 < len(ins) <= G.MAX_INST, 'C  a hypothesis "forall n, f n = 2 n + 1" is offered for instantiation (%d proposals)' % len(ins))
    expect(all(len(a.args) == 1 for a in ins) and len({inst_sig(k, w, m, a) for a in ins}) == len(ins), 'C  ... with one term each, all different')
    expect(len(ins) <= G.INST_ONE, 'C  ... at most %d of them (the first terms of the type)' % G.INST_ONE)
    src1 = k.mk_fvar(1)
    t3 = inst_with(m, src1, ('3',))
    expect(t3 is not None and g.hyp_names[1] == 'a1', 'C  the source is the hypothesis, named by its position (a1)')
    s0 = state(k, w); ch = eng.probe(g, t3, m)
    expect(ch is not None and len(ch) == 1 and ch[0].n_hyps == g.n_hyps + 1, 'C  INSTANTIATE at 3 leaves one goal with one more hypothesis')
    expect(state(k, w) == s0, 'C  ... and the probe leaves the kernel as it was')
    g1 = ch[0]; new = g1.hyp_types[-1]
    expect(g1.hyp_names[-1] == 'i%d' % g.n_hyps, 'C  ... named i<position> (%s)' % g1.hyp_names[-1])
    with eng.in_context(g.hyp_types, g.hyp_names):
        with w.nota.using(Z):
            lit = w.nota.lit; f = w.T(k.mk_fvar(0)); expected = w.Eq(f(lit(3)), 2 * lit(3) + 1)
    expect(k.conv_open(new, expected.n, g.n_hyps), 'C  ... whose statement is the instance f 3 = 2 * 3 + 1')
    ep.step(t3)
    expect(ep.goal().n_hyps == g.n_hyps + 1, 'C  the episode takes the step')
    # the same fact is neither offered again nor accepted
    m_again = ep.menu()
    expect(inst_with(m_again, src1, ('3',)) is None, 'C  ... and the same instance is not offered again')
    ci = next((i for i, cd in enumerate(m_again.cands) if cd[0] == src1 and any(a.kind == G.Kind.INSTANTIATE and a.arg == i for a in m_again.actions)), None)
    i3 = next(i for i in m_again.pool.of_type[Z.n] if m_again.pool.labels[i] == '3')
    if expect(ci is not None, 'C  (the hypothesis is still offered at other terms)'):
        refuses('C  a fact that is already a hypothesis is refused', lambda: ep.step(G.Action(G.Kind.INSTANTIATE, ci, (i3,))), exc=(G.ActionError,))
    # the goal is then closed by arithmetic
    lin = find(ep.menu(), G.Kind.LINARITH)
    if expect(len(lin) == 1, 'C  linarith is offered with the new fact'):
        ep.step(lin[0]); expect(ep.done and ep.finish() is not None, 'C  ... and closes f 3 + f 3 = 14 from f 3 = 2 * 3 + 1')
    # two terms: the additive function
    ep2 = G.Episode(eng, fx.add); m2 = ep2.menu(); ins2 = find(m2, G.Kind.INSTANTIATE)
    expect(len(ins2) > 0 and len(ins2) <= G.INST_TWO * G.INST_TWO and all(len(a.args) == 2 for a in ins2), 'C  a hypothesis with two data quantifiers is offered at pairs of terms (%d proposals)' % len(ins2))
    zi = m2.pool.of_type[Z.n]
    expect(all(zi.index(i) < G.INST_TWO for a in ins2 for i in a.args), 'C  ... drawn from the first %d terms of the type' % G.INST_TWO)
    a11 = inst_with(m2, src1, ('1', '1')); a21 = inst_with(m2, src1, ('2', '1'))
    expect(a11 is not None and a21 is not None, 'C  ... among them (1, 1) and (2, 1)')
    sig21 = m2.signature(a21)
    ep2.step(a11); m2b = ep2.menu(); a21b = m2b.find(sig21)
    expect(a21b is not None, 'C  the second instance is still offered after the first (found again by its signature)')
    if a21b is not None:
        ep2.step(a21b); lin2 = find(ep2.menu(), G.Kind.LINARITH)
        if expect(len(lin2) == 1, 'C  linarith is offered'):
            ep2.step(lin2[0]); expect(ep2.done and ep2.finish() is not None, 'C  f 3 = 3 f 1 follows from f (1+1) = 2 f 1 and f (2+1) = f 2 + f 1')
    # one term where there are two quantifiers is legal (the engine does not list it)
    ep2b = G.Episode(eng, fx.add); m2x = ep2b.menu(); c1 = next(i for i, cd in enumerate(m2x.cands) if cd[0] == src1)
    i1 = next(i for i in m2x.pool.of_type[Z.n] if m2x.pool.labels[i] == '1')
    ep2b.step(G.Action(G.Kind.INSTANTIATE, c1, (i1,)))
    expect(k.view(ep2b.goal().hyp_types[-1])[0] == 'pi', 'C  one term where there are two quantifiers leaves a hypothesis that still quantifies')
    # a premise after the data quantifier stays a premise
    ep3 = G.Episode(eng, fx.prem); m3 = ep3.menu(); i3s = find(m3, G.Kind.INSTANTIATE)
    expect(len(i3s) > 0 and all(len(a.args) == 1 for a in i3s), 'C  "forall a, 0 <= a -> f a = a" is instantiated at one term (the premise stops it)')
    t2 = inst_with(m3, src1, ('2',))
    if expect(t2 is not None, 'C  ... at 2 for example'):
        chp = eng.probe(ep3.goal(), t2, m3); fact = chp[0].hyp_types[-1]
        expect(k.view(fact)[0] == 'pi', 'C  ... the new hypothesis is an implication: 0 <= 2 -> f 2 = 2')
    # lemmas: only the allowed ones, only within the premises
    ea = G.Engine(w, terms=True, strict=True, inst_lemmas=('Z.sq_nonneg', 'no_such_lemma'))
    epa = G.Episode(ea, fx.amgm); ma = epa.menu(); ia = find(ma, G.Kind.INSTANTIATE)
    sq = k.mk_const(w.gid['Z.sq_nonneg'])
    expect(len(ia) > 0 and all(ma.cands[a.arg][0] == sq for a in ia), 'C  with inst_lemmas the lemma is offered for instantiation (the unknown name is ignored): %d proposals' % len(ia))
    eb = G.Engine(w, terms=True, strict=True, inst_lemmas=('Z.sq_nonneg',), premises=w.gid['Z.sq_nonneg'])
    expect(not find(G.Episode(eb, fx.amgm).menu(), G.Kind.INSTANTIATE), 'C  ... but not when it is beyond the premises')
    amb = inst_with(ma, sq, ('(a0 - a1)',))
    if expect(amb is not None, 'C  the lemma is offered at a0 - a1'):
        epa.step(amb); lin3 = find(epa.menu(), G.Kind.LINARITH)
        if expect(len(lin3) == 1, 'C  linarith is offered after the instance'):
            epa.step(lin3[0]); expect(epa.done and epa.finish() is not None, 'C  2ab <= a^2 + b^2 follows from 0 <= (a - b)^2')
    # refusals
    ep5 = G.Episode(eng, fx.lin); m5 = ep5.menu(); k5 = ep5.goal().key; s5 = state(k, w); src = find(m5, G.Kind.INSTANTIATE)[0].arg
    for what, act in (('no term', G.Action(G.Kind.INSTANTIATE, src, ())), ('an index outside the pool', G.Action(G.Kind.INSTANTIATE, src, (len(m5.pool),))), ('a negative index', G.Action(G.Kind.INSTANTIATE, src, (-1,))),
                      ('a candidate that does not exist', G.Action(G.Kind.INSTANTIATE, len(m5.cands) + 3, (0,))), ('too many terms', G.Action(G.Kind.INSTANTIATE, src, (0, 1, 2)))):
        refuses('C  INSTANTIATE with %s is refused' % what, lambda act=act: ep5.step(act), exc=(G.ActionError,))
    expect(ep5.goal().key == k5 and state(k, w) == s5, 'C  ... and each leaves the episode and the kernel as they were')
    trace('C done')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. the menu with terms
def sec_D(c):
    fx = c['fx']; w, k = fx.w, fx.k
    rng = random.Random(38)
    eng = G.Engine(w, terms=True, strict=True, inst_lemmas=('Z.sq_nonneg',))
    bad = []; n_steps = n_refused = n_term = n_fin = 0
    stmts = [fx.sq49, fx.sq50, fx.pair, fx.lin, fx.add, fx.amgm, fx.vars, fx.any_w, fx.prem]
    for trial in range(90):
        st = stmts[trial % len(stmts)]
        ep = G.Episode(eng, st)
        for step in range(8):
            if ep.done: break
            m = ep.menu(); g = ep.goal()
            if not m.actions: break
            for a in m.actions:
                if m.find(m.signature(a)) != a: bad.append(('signature round trip', G.KIND_NAMES[a.kind]))
            tact = [a for a in m.actions if a.kind in G.TERM_KINDS]; closing = [a for a in m.actions if a.kind in (G.Kind.RING, G.Kind.LINARITH)]
            roll = rng.random()
            a = rng.choice(tact) if tact and roll < 0.55 else (rng.choice(closing) if closing and roll < 0.75 else rng.choice(m.actions))
            pred = eng.probe(g, a, m); n0 = ep.n_goals; s0 = state(k, w)
            if pred is None:
                n_refused += 1
                try: ep.step(a); bad.append(('a refused action was taken', G.KIND_NAMES[a.kind]))
                except G.ActionError: pass
                if ep.n_goals != n0 or eng.goal_of(ep.p, 0).key != g.key or state(k, w) != s0: bad.append(('a refused action changed the state', G.KIND_NAMES[a.kind]))
                continue
            n = ep.step(a); n_steps += 1; n_term += a.kind in G.TERM_KINDS
            if n != n0 - 1 + len(pred): bad.append(('wrong number of goals', G.KIND_NAMES[a.kind]))
            for i, ch in enumerate(pred):
                if eng.goal_of(ep.p, i).key != ch.key: bad.append(('the live goal differs from the predicted one', G.KIND_NAMES[a.kind], i))
        if ep.done:
            try: ep.finish(); n_fin += 1
            except Exception as ex: bad.append(('a finished episode was not accepted', str(ex)[:200]))
    expect(not bad, 'D  random play with term actions: %s' % (bad[:3],))
    expect(not eng.bugs and eng.stats['probe_bug'] == 0, 'D  ... and nothing was mistaken for a refusal (%s)' % (eng.bugs[:1],))
    expect(n_steps >= 150 and n_term >= 60 and n_refused >= 3 and n_fin >= 1, 'D  random play is not trivial (%d steps, %d with terms, %d refused, %d finished)' % (n_steps, n_term, n_refused, n_fin))
    # kinds and the switch
    m = G.Episode(G.Engine(w, kinds=[G.Kind.USE]), fx.sq49).menu()
    expect(m.actions and {a.kind for a in m.actions} == {G.Kind.USE} and len(m.cands) == 0 and len(m.pool) > 0, 'D  kinds=[USE] offers only USE (and asking for a term kind turns the pool on by itself)')
    m = G.Episode(G.Engine(w, kinds=[G.Kind.INSTANTIATE]), fx.lin).menu()
    expect({a.kind for a in m.actions} == {G.Kind.INSTANTIATE} and len(m.cands) >= 1 and all(a.arg < len(m.cands) for a in m.actions), 'D  kinds=[INSTANTIATE] offers only INSTANTIATE, with its candidates renumbered')
    m = G.Episode(G.Engine(w, terms=False), fx.sq49).menu()
    expect(len(m.pool) == 0 and not any(a.kind in G.TERM_KINDS for a in m.actions), 'D  without terms there is no pool and no term action')
    m = G.Episode(G.Engine(w), fx.sq49).menu()
    expect(len(m.pool) == 0 and not any(a.kind in G.TERM_KINDS for a in m.actions), 'D  by default (TERMS_DEFAULT is off) the menu has no term actions, so the behaviour of gate S36 is unchanged')
    mk_ = G.Episode(G.Engine(w, kinds=[G.Kind.RING, G.Kind.USE]), fx.sq49).menu()
    expect(mk_.actions and all(a.kind in (G.Kind.RING, G.Kind.USE) for a in mk_.actions), 'D  mixed kinds')
    # without a hypothesis or an existential there is nothing to build a pool for
    expect(len(G.Episode(G.Engine(w, terms=True), fx.comm).menu().pool) == 0, 'D  a goal with no existential and nothing to instantiate has no pool')
    expect(len(G.Episode(eng, fx.comm).menu().pool) > 0, 'D  ... but one that a lemma allowed for instantiation can be applied to has')
    # the features of the pool
    mf = G.Episode(G.Engine(w, terms=True), fx.vars).menu()
    expect(len(mf.pool_tags) == len(mf.pool) == len(mf.pool_scalars) and mf.pool_tags.dtype.name == 'uint32' and mf.pool_scalars.dtype.name == 'uint32', 'D  the menu has one (tag, scalar) pair per pool term')
    tg, sc = k.term_features(list(mf.pool.terms))
    expect([int(x) for x in mf.pool_tags] == [int(x) for x in tg] and [int(x) for x in mf.pool_scalars] == [int(x) for x in sc], 'D  ... which are the features of the pool terms, in pool order')
    one = k.term_features([mf.pool.terms[9]])
    expect(int(one[0][0]) == int(mf.pool_tags[9]) and int(one[1][0]) == int(mf.pool_scalars[9]), 'D  ... and each is the feature of that very term')
    trace('D done')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. probes with a budget
def sec_E(c):
    fx = c['fx']; w, k = fx.w, fx.k
    eng = G.Engine(w, terms=True, strict=True)
    ep = G.Episode(eng, fx.sq49); g = ep.goal(); m = ep.menu()
    use7 = use_with(m, '7')
    ch, st = eng.probe_status(g, use7, m)
    expect(st == G.ST_OK and ch == [] and eng.probe(g, use7, m) == ch, 'E  probe_status of a working action: the goals and "ok" (the same as probe)')
    # too little work: running out of budget is not a refusal
    s0 = state(k, w); n_fuel0 = eng.stats['probe_fuel']
    ch, st = eng.probe_status(g, use7, m, fuel=1)
    expect(ch is None and st == G.ST_FUEL, 'E  with a budget of one step the same action is "fuel", not "refused" (%s)' % st)
    expect(state(k, w) == s0, 'E  ... and the kernel, the probing flag and the budget are as they were')
    expect(eng.stats['probe_fuel'] == n_fuel0 + 1, 'E  ... and it is counted')
    # an action that is refused on logical grounds: cases on a function
    el = G.Episode(eng, fx.lin); gl = el.goal(); ml = el.menu()
    ch, st = eng.probe_status(gl, G.Action(G.Kind.CASES, 0), ml)
    expect(ch is None and st == G.ST_REFUSED, 'E  cases on a function is refused (%s)' % st)
    # outcomes by budget: running out, then always the same answer
    small = G.Engine(w, terms=True, strict=True, inst_lemmas=('Z.sq_nonneg',), probe_fuel=60000); small.probe_fuel = {}        # one top budget for every kind
    n_checked = 0; bad = []; kinds_seen = set(); n_fuel_seen = 0
    for stmt in (fx.sq49, fx.lin, fx.add, fx.amgm, fx.mix):
        e = G.Episode(small, stmt); gg = e.goal(); mm = e.menu()
        for a in mm.actions[:70]:
            outs = []
            for fuel in small.fuel_levels(a.kind):
                ch, st = small.probe_status(gg, a, mm, fuel); outs.append((st, None if ch is None else tuple(x.key for x in ch)))
            n_checked += 1; kinds_seen.add(a.kind)
            first = next((i for i, o in enumerate(outs) if o[0] != G.ST_FUEL), len(outs)); n_fuel_seen += first > 0
            if any(o != outs[first] for o in outs[first:]): bad.append((G.KIND_NAMES[a.kind], [o[0] for o in outs]))
            full = small.probe(gg, a, mm)
            if first < len(outs) and outs[-1][1] != (None if full is None else tuple(x.key for x in full)): bad.append(('the largest level differs from probe', G.KIND_NAMES[a.kind]))
    expect(not bad and n_checked > 150 and len(kinds_seen) >= 5, 'E  over %d actions of %d kinds the outcome by budget is: running out, then always the same answer (%s)' % (n_checked, len(kinds_seen), bad[:3]))
    expect(n_fuel_seen > 0, 'E  ... and some of them did run out at the smaller budgets (%d)' % n_fuel_seen)
    # the levels
    for kind in G.Kind:
        lv = eng.fuel_levels(kind)
        expect(list(lv) == sorted(set(lv)) and lv[-1] == eng.fuel_of(G.Action(kind, 0)) and lv[0] >= 1 and 1 <= len(lv) <= 5, 'E  the budget levels of %s ascend to its budget (%s)' % (G.KIND_NAMES[kind], lv))
    lv = G.Engine(w, probe_fuel=10).fuel_levels(G.Kind.APPLY)
    expect(lv[-1] == 10 and len(lv) <= 3 and list(lv) == sorted(set(lv)), 'E  a small budget gives few, distinct levels (%s)' % (lv,))
    # the flag on the tactic layer's refusal
    with w.nota.using(fx.Z): lit7 = w.nota.lit(7)
    p = Proof(w, w.term(fx.sq49, 'statement'), 'fuel_flag'); saved = k.get_fuel(); k.set_fuel(1)
    try:
        try: p.exists(lit7)
        except AuthorError as e_: expect(e_.fuel is True, 'E  a tactic that ran out of budget says so (AuthorError.fuel)')
        else: expect(False, 'E  a tactic with a budget of one step should have failed')
    finally: k.set_fuel(saved)
    p2 = Proof(w, w.term(fx.amgm, 'statement'), 'refuse_flag')
    try: p2.apply('Z.le_refl')
    except AuthorError as e_: expect(e_.fuel is False, 'E  a logical refusal is not flagged as running out of budget')
    else: expect(False, 'E  apply le_refl to a quantified goal should be refused')
    expect(AuthorError('x').fuel is False, 'E  the default of AuthorError.fuel is false')
    trace('E done')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. goal identity
def sec_F(c):
    fx = c['fx']; w, k, Z = fx.w, fx.k, fx.Z
    eng = G.Engine(w)
    zle = w['Z.le']
    with eng.in_context([Z.n]):
        with w.nota.using(Z):
            lit = w.nota.lit; x = w.T(k.mk_fvar(0))
            facts = {'a': zle(lit(0), x).n, 'b': zle(x, lit(5)).n, 'c': zle(lit(1), x).n, 'd': zle(lit(2), x).n}; target = w.Eq(x, x).n
    def goal(order): return G.GoalData(target, [Z.n] + [facts[o] for o in order], ['x'] + list(order))
    abc, bca, cab, ab = goal('abc'), goal('bca'), goal('cab'), goal('ab')
    expect(abc.key != bca.key and abc.key != cab.key, 'F  different orders are different goals for the menus (the exact keys differ)')
    ca, cb, cc = eng.canon_key(abc), eng.canon_key(bca), eng.canon_key(cab)
    expect(ca == cb == cc, 'F  ... but one problem: the canonical keys of the three orders agree')
    expect(eng.canon_key(abc) is ca and abc.ckey is ca, 'F  the canonical key is computed once')
    expect(eng.canon_key(ab) != ca, 'F  a goal with fewer hypotheses is a different problem')
    expect(eng.canon_key(goal('abc')) == ca and eng.canon_key(goal('dbc')) != ca, 'F  ... and so is a goal with another hypothesis')
    expect(eng.canon_key(goal('dcba')) == eng.canon_key(goal('abcd')), 'F  any order of four unreferenced hypotheses has one key')
    expect(eng.canon_key(G.GoalData(target, [Z.n], ['x'])) == (target, Z.n), 'F  a goal whose hypotheses are all mentioned keeps its order')
    # a variable that the target mentions separates the hypotheses on its two sides
    with eng.in_context([Z.n]):
        with w.nota.using(Z):
            lit = w.nota.lit; x0 = w.T(k.mk_fvar(0)); h1 = zle(lit(0), x0).n; h2 = zle(lit(2), x0).n
    def with_y(first, second):
        with eng.in_context([Z.n, first, Z.n]):
            y = w.T(k.mk_fvar(2)); tgt = w.Eq(y, y).n
        return G.GoalData(tgt, [Z.n, first, Z.n, second], ['x', 'f', 'y', 'g'])
    g_a, g_b = with_y(h1, h2), with_y(h2, h1)
    expect(eng.canon_key(g_a) == (g_a.target, Z.n, h1, Z.n, h2) and eng.canon_key(g_b) == (g_b.target, Z.n, h2, Z.n, h1) and eng.canon_key(g_a) != eng.canon_key(g_b), 'F  hypotheses on the two sides of a variable that is mentioned are not exchanged')
    trace('F done')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ G. exists
def sec_G(c):
    fx = c['fx']; w, k, Z = fx.w, fx.k, fx.Z
    Ex = w['Ex']; Eq = w.Eq; F = w.forall_
    with w.nota.using(Z):
        lit = w.nota.lit
        st = F([Z], lambda a: Ex(Z, w.lam(Z, lambda x: Eq(x + a, a + lit(3)))))
    p = Proof(w, w.term(st, 'statement'), 'ex1'); p.intro('a'); n0 = len(p.trace)
    with w.nota.using(Z): p.exists(lit(3))
    with p.focus(): tgt = w.show(p.goal().target)
    expect(not has_meta(k, p.goal().target.n) and 'fun' not in tgt, 'G  exists leaves no metavariable and no beta-redex in the goal (%s)' % tgt)
    steps = p.trace[n0:]
    expect(len(steps) == 1 and steps[0][0] == 'apply', 'G  ... because the kernel is given ex_intro A P witness at once (steps: %s)' % ([s[0] for s in steps],))
    if steps:
        head, args = w.spine(w.T(steps[0][1]))
        expect(w.head_name(head) == 'ex_intro' and len(args) == 3, 'G  ... with the type, the predicate and the witness')
    expect(p.n_goals == 1, 'G  one goal is left')
    # the older way (apply ex_intro, exact the witness) ends at the same goal
    p2 = Proof(w, w.term(st, 'statement'), 'ex2'); p2.intro('a'); p2.apply('ex_intro')
    with w.nota.using(Z): p2.exact(lit(3))
    p2._norm_all()
    t1, t2 = p.goal().target, p2.goal().target
    expect(t1 == t2 or k.conv_open(t1.n, t2.n, 1), 'G  ... the goal that the older way leaves')
    p.ring()
    expect(p.done, 'G  and the goal x + a = a + 3 at x = 3 is closed by ring')
    w.check(p.term(), w.term(st, 'statement')); expect(True, 'G  the kernel accepts the proof')
    # a definition at the head: divisibility
    pd = Proof(w, w.term(fx.dvd, 'statement'), 'ex_dvd')
    with w.nota.using(Z): pd.exists(lit(4))
    with pd.focus(): shown = w.show(pd.goal().target) if not pd.done else ''
    expect(not pd.done and not has_meta(k, pd.goal().target.n), 'G  exists works behind a definition (3 | 12): %s' % shown)
    if not pd.done: pd.ring()
    if expect(pd.done, 'G  ... and ring closes 12 = 3 * 4'):
        w.check(pd.term(), w.term(fx.dvd, 'statement')); expect(True, 'G  ... and the proof of 3 | 12 is accepted by the kernel')
    # a goal that is no existential
    pn = Proof(w, w.term(fx.amgm, 'statement'), 'ex_no'); pn.intro('a', 'b'); before = str(pn)
    with w.nota.using(Z): zero = lit(0)
    refuses('G  exists on a goal that is no existential is refused', lambda: pn.exists(zero))
    expect(str(pn) == before, 'G  ... and changes nothing')
    # a witness of the wrong type
    pw = Proof(w, w.term(fx.sq49, 'statement'), 'ex_wrong'); before = str(pw)
    refuses('G  exists with a witness of the wrong type is refused', lambda: pw.exists(w['N0']))
    expect(str(pw) == before, 'G  ... and changes nothing')
    # a proved existential: the kernel accepts the whole proof
    pp = Proof(w, w.term(fx.sq49, 'statement'), 'ex_ok')
    with w.nota.using(Z): pp.exists(lit(7))
    pp.ring()
    expect(pp.done, 'G  x * x = 49 at 7 is closed by ring')
    w.check(pp.term(), w.term(fx.sq49, 'statement')); expect(True, 'G  the kernel accepts the proof through the explicit witness')
    trace('G done')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ H. the search
def sec_H(c):
    fx = c['fx']; w, k = fx.w, fx.k
    K = G.Kind; witness = [K.USE, K.APPLY, K.RING]; instance = [K.INSTANTIATE, K.LINARITH]
    # (the kinds are restricted where the unrestricted space is far too large for blind search: every library lemma that could be applied, every rewrite, and a hundred instances at every node)
    both = ('id', 'cost'); cost_only = ('cost',)            # (blind deepening takes long on the two with the larger spaces; the cheap ones show that it works)
    problems = [('exists x, x * x = 49', fx.sq49, None, (), both), ('exists a b, a + b = 5 /\\ a - b = 1', fx.pair, witness, (), cost_only),
                ('(forall n, f n = 2 n + 1) gives f 3 + f 3 = 14', fx.lin, instance, (), both), ('f additive gives f 2 = 2 f 1', fx.add2, instance, (), cost_only),
                ('2ab <= a^2 + b^2 (with the lemma sq_nonneg)', fx.amgm, instance, ('Z.sq_nonneg',), both), ('(forall a, 0 <= a -> f a = a) gives f 2 = 2', fx.prem, None, (), both)]
    proofs = []; t_all = time.time()
    for label, st, kinds, lemmas, methods in problems:
        eng = G.Engine(w, terms=True, strict=True, kinds=kinds, inst_lemmas=lemmas)
        for method in methods:
            t0 = time.time()
            term, tree, s = S.prove_statement(eng, st, max_depth=(4 if method == 'id' else 12), max_probes=6000, time_limit=60, cost=(True if method == 'cost' else None))
            if expect(term is not None, 'H  %s is proved by the %s search (%s probes)' % (label, method, s.probes if s else '-')):
                expect(k.check_open(term.n, w.term(st, 'statement').n, [])[0], 'H  ... and the kernel accepts the proof')
                proofs.append((label, st, term))
            expect(not eng.bugs, 'H  ... with nothing mistaken for a refusal')
            trace('H %s %s: %.1f s, %s probes' % (label, method, time.time() - t0, s.probes if s else '-'))
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid), recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    chk = ic.Checker(env, params=True, fuel=400_000_000, prop_elim=True)
    for label, st, term in proofs:
        ok, msg = chk.check_closed(parse_raw(k.dump_raw(term.n), True), parse_raw(k.dump_raw(w.term(st, 'statement').n), True))
        expect(ok, 'H  the independent checker accepts the proof of %s (%s)' % (label, msg))
    # a false existential is not proved
    for method in ('id', 'cost'):
        term, tree, s = S.prove_statement(G.Engine(w, terms=True, strict=True), fx.sq50, max_depth=(3 if method == 'id' else 8), max_probes=500, time_limit=30, cost=(True if method == 'cost' else None))
        expect(term is None, 'H  exists x, x * x = 50 is not proved by the %s search' % method)
    # the cost search: determinism, budgets, memory
    def fresh(stmt, **kw):
        eng = G.Engine(w, terms=True, strict=True, **kw); p = eng.start(w.term(stmt, 'statement')); return eng, eng.goal_of(p, 0)
    e1, g1 = fresh(fx.pair, kinds=witness); s1 = S.CostSearcher(e1, max_probes=4000, time_limit=60); t1 = s1.prove(g1, max_cost=12)
    e2, g2 = fresh(fx.pair, kinds=witness); s2 = S.CostSearcher(e2, max_probes=4000, time_limit=60); t2 = s2.prove(g2, max_cost=12)
    expect(t1 is not None and t2 is not None and t1.signature == t2.signature and t1.size() == t2.size() and s1.probes == s2.probes, 'H  the cost search is deterministic (%s probes both times)' % s1.probes)
    p0 = s1.probes; again = s1.prove(g1, max_cost=12)
    expect(again is t1 and s1.probes == p0, 'H  asking again for a solved goal costs no probe')
    e3, g3 = fresh(fx.pair, kinds=witness); s3 = S.CostSearcher(e3, max_probes=5, time_limit=60); t3 = s3.prove(g3, max_cost=12)
    expect(t3 is None and s3.probes <= 6, 'H  the probe budget is honoured (%d probes with a budget of 5)' % s3.probes)
    e4, g4 = fresh(fx.pair, kinds=witness); s4 = S.CostSearcher(e4, time_limit=0.0)
    expect(s4.prove(g4, max_cost=12) is None, 'H  a time limit of zero stops the search at once')
    e5, g5 = fresh(fx.sq50); s5 = S.CostSearcher(e5, max_probes=3000, time_limit=60); s5.prove(g5, max_cost=4); p_a = s5.probes
    s5.prove(g5, max_cost=4)
    expect(s5.probes == p_a, 'H  a pass that is repeated costs no probe (%d)' % p_a)
    s5.prove(g5, max_cost=5)
    expect(s5.probes > p_a, 'H  a larger budget explores more (%d, then %d probes)' % (p_a, s5.probes))
    # exhaustion: with USE as the only kind the space of a false existential is finite -- one step, then nothing to do -- and the search ends exactly when the dearest step fits the budget with room to spare
    e6, g6 = fresh(fx.sq50, kinds=[G.Kind.USE]); m6 = e6.menu(g6); use6 = find(m6, G.Kind.USE)
    top = max(S.default_cost(e6, g6, m6, a) for a in use6)
    s6 = S.CostSearcher(e6, max_probes=3000, time_limit=60); t6 = s6.prove(g6, max_cost=40)
    expect(t6 is None and s6.passes == top + 1 and s6.probes == len(use6), 'H  a space that is exhausted ends the search at the first budget that cuts nothing (%d passes for a dearest step of %d; %d probes for %d actions)' % (s6.passes, top, s6.probes, len(use6)))
    # unit costs make the cost search a depth search: the same depth as deepening, on lemmas proved without the decision procedures
    w3 = c['w3']; plain = [G.Kind.INTRO, G.Kind.CASES, G.Kind.INDUCT, G.Kind.APPLY, G.Kind.REWRITE]
    unit = lambda eng_, g_, m_, a_: 1
    n_both = 0
    for name in ('add_assoc', 'le_trans', 'add_eq_zero', 'succ_ne_self'):
        stmt = w3.global_type(name)
        e = G.Engine(w3, premises=w3.gid[name], kinds=plain); term_id, tree_id, s_id = S.prove_statement(e, stmt, max_depth=7, max_probes=4000, time_limit=60)
        e = G.Engine(w3, premises=w3.gid[name], kinds=plain); pc = e.start(stmt); gc = e.goal_of(pc, 0)
        sc = S.CostSearcher(e, cost=unit, fuel_step=0, max_probes=4000, time_limit=60); tree_c = sc.prove(gc, max_cost=7)
        if expect(term_id is not None and tree_c is not None, 'H  %s is proved by both searches without the decision procedures' % name):
            n_both += 1
            expect(tree_c.depth() == tree_id.depth(), 'H  ... at the same depth: unit costs are depth (%s: %d and %d)' % (name, tree_id.depth(), tree_c.depth()))
            expect(S.commit(e, stmt, tree_c) is not None, 'H  ... and the cost search\'s tree replays on a live proof')
    expect(n_both == 4, 'H  all four lemmas were compared')
    # a price decides what is tried first: a witness out of reach stops the proof
    eng7, g7 = fresh(fx.sq49, kinds=[G.Kind.USE, G.Kind.RING])         # (more kinds would let the passes below the price wander off through the lemmas at a unit price)
    pricey = lambda eng_, g_, m_, a_: 50 if a_.kind == G.Kind.USE else 1
    expect(S.CostSearcher(eng7, cost=pricey, max_probes=2000, time_limit=30).prove(g7, max_cost=10) is None, 'H  a price that puts every witness out of reach stops the proof (budget 10 < 50)')
    expect(S.CostSearcher(eng7, cost=pricey, max_probes=2000, time_limit=30).prove(g7, max_cost=60) is not None, 'H  ... and with a budget above the price it is found')
    trace('H done %.1f s' % (time.time() - t_all))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ I. arguments are validated
def sec_I(c):
    fx = c['fx']; w, k = fx.w, fx.k
    eng = G.Engine(w, terms=True, strict=True)
    ep = G.Episode(eng, fx.mix); g = ep.goal(); m = ep.menu(); s0 = state(k, w)
    ivs = [i for i, cd in enumerate(m.cands) if cd[0] is None]; lem = [i for i, cd in enumerate(m.cands) if cd[0] is not None]
    expect(len(ivs) == 1 and len(lem) > 0, 'I  the goal 0 <= x <= 3 -> x + 1 = 1 + x has an interval and lemmas among its candidates (%d, %d)' % (len(ivs), len(lem)))
    tried = []
    def bad(what, act):
        tried.append(what)
        refuses('I  %s is refused' % what, lambda: ep.step(act), exc=(G.ActionError,))
        _, st = eng.probe_status(g, act, m)
        expect(st == G.ST_REFUSED, 'I  ... also by the probe, as a refusal and not as a bug (%s: %s)' % (what, st))
    n = len(m.cands)
    for kind in (G.Kind.APPLY, G.Kind.REWRITE, G.Kind.INSTANTIATE, G.Kind.INTERVAL_CASES):
        nm = G.KIND_NAMES[kind]
        bad('%s with the candidate past the end' % nm, G.Action(kind, n))
        bad('%s with a negative candidate' % nm, G.Action(kind, -1))
        bad('%s with a candidate that is no number' % nm, G.Action(kind, 'x'))
        bad('%s with a float candidate' % nm, G.Action(kind, 1.0))
    bad('APPLY of an interval candidate', G.Action(G.Kind.APPLY, ivs[0]))
    bad('REWRITE by an interval candidate', G.Action(G.Kind.REWRITE, ivs[0]))
    bad('INTERVAL_CASES on a lemma', G.Action(G.Kind.INTERVAL_CASES, lem[0]))
    nh = g.n_hyps
    for kind in (G.Kind.CASES, G.Kind.INDUCT):
        nm = G.KIND_NAMES[kind]
        bad('%s on a hypothesis past the end' % nm, G.Action(kind, nh))
        bad('%s on a negative hypothesis' % nm, G.Action(kind, -1))
        bad('%s on a hypothesis that is no number' % nm, G.Action(kind, None))
    bad('a kind that does not exist', G.Action(99, 0))
    bad('a negative kind', G.Action(-1, 0))
    bad('a kind that is no number', G.Action('APPLY', 0))
    bad('a boolean as the argument of RING', G.Action(G.Kind.RING, True))
    for kind in (G.Kind.RING, G.Kind.LINARITH, G.Kind.NLINARITH, G.Kind.BY_CONTRA, G.Kind.INTRO, G.Kind.CASES, G.Kind.APPLY):
        bad('%s with terms' % G.KIND_NAMES[kind], G.Action(kind, 0, (0,)))
    expect(state(k, w) == s0 and ep.goal().key == g.key and ep.n_goals == 1, 'I  none of the %d refused actions changed the episode or the kernel' % len(tried))
    # every action the menu lists passes the check
    for a in m.actions: eng.check_action(a, m, g.n_hyps)
    expect(True, 'I  every listed action passes the check')
    expect(not eng.bugs and eng.stats['probe_bug'] == 0, 'I  nothing was mistaken for a refusal')
    trace('I done')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ J. rewriting costs
def sec_J(c):
    fx = c['fx']; w, k = fx.w, fx.k
    eng = G.Engine(w, strict=True)
    perm = {n: eng.profile(w.gid[n]).permutative for n in ('Z.add_comm', 'Z.mul_comm', 'Z.add_assoc', 'Z.mul_assoc', 'Z.add_left_comm', 'Z.mul_add', 'Z.add_zero', 'Z.zero_add', 'Z.mul_one', 'Z.neg_neg') if n in w.gid}
    for n in ('Z.add_comm', 'Z.mul_comm', 'Z.add_assoc', 'Z.mul_assoc'):
        if n in perm: expect(perm[n] is True, 'J  %s only reorders a term: permutative' % n)
    for n in ('Z.mul_add', 'Z.add_zero', 'Z.zero_add', 'Z.mul_one', 'Z.neg_neg'):
        if n in perm: expect(perm[n] is False, 'J  %s changes the term: not permutative' % n)
    expect(sum(perm.values()) >= 3 and not all(perm.values()), 'J  some of the equations are permutative and some are not (%s)' % perm)
    # in the menu: bit 1 of cand_flags
    ep = G.Episode(eng, fx.comm); m = ep.menu(); g = ep.goal()
    rw = [(w.gname.get(k.view(m.cands[a.arg][0])[1]), int(m.cand_flags[a.arg])) for a in find(m, G.Kind.REWRITE) if k.view(m.cands[a.arg][0])[0] == 'const']
    expect(rw and all((f & 2) == (2 if perm.get(nm) else 0) for nm, f in rw if nm in perm), 'J  the menu marks the permutative candidates (bit 1 of cand_flags): %s' % (rw[:6],))
    expect(any(f & 2 for _, f in rw), 'J  ... and there are some on a + b = b + a')
    # an equation among the hypotheses
    with w.nota.using(fx.Z):
        lit = w.nota.lit
        st = w.forall_([fx.Z, fx.Z, lambda a, b: w.Eq(a + b, b + a), lambda a, b, h: w.Eq(a * b, b * a)], lambda a, b, h, h2: w.Eq(a + b + lit(1), b + a + lit(1)))
    ep2 = G.Episode(eng, st); m2 = ep2.menu(); g2 = ep2.goal()
    hyp_rw = [(k.view(m2.cands[a.arg][0]), int(m2.cand_flags[a.arg])) for a in find(m2, G.Kind.REWRITE) if k.view(m2.cands[a.arg][0])[0] == 'fvar']
    expect(hyp_rw and all(f & 2 for _, f in hyp_rw), 'J  a hypothesis a + b = b + a is permutative too (%s)' % (hyp_rw,))
    # the default cost
    expect(S.default_cost(eng, g, m, G.Action(G.Kind.RING, 0)) == S.KIND_COST[G.Kind.RING] == 1, 'J  a decision procedure costs one step')
    is_lib = lambda mm, a: k.view(mm.cands[a.arg][0])[0] == 'const'
    rws = find(m, G.Kind.REWRITE)
    perm_rw = next((a for a in rws if is_lib(m, a) and int(m.cand_flags[a.arg]) & 2), None)
    lem_rw = next((a for a in rws if is_lib(m, a) and not int(m.cand_flags[a.arg]) & 2), None)
    if expect(perm_rw is not None, 'J  (a permutative library rewrite is on offer)'):
        expect(S.default_cost(eng, g, m, perm_rw) == S.KIND_COST[G.Kind.REWRITE] + 1 + 2, 'J  a permutative library rewrite costs the step, one for the library and two for the permutation')
    if lem_rw is not None: expect(S.default_cost(eng, g, m, lem_rw) == S.KIND_COST[G.Kind.REWRITE] + 1, 'J  a library rewrite that is not permutative costs the step and one more')
    hr = next((a for a in find(m2, G.Kind.REWRITE) if not is_lib(m2, a)), None)
    if expect(hr is not None, 'J  (a rewrite by a hypothesis is on offer)'):
        expect(S.default_cost(eng, g2, m2, hr) == S.KIND_COST[G.Kind.REWRITE] + (2 if int(m2.cand_flags[hr.arg]) & 2 else 0), 'J  a rewrite by a hypothesis costs the step, and two more when it is permutative')
    # terms: the later in the pool the dearer, by the binary logarithm of the place
    et = G.Engine(w, terms=True, strict=True); et_ep = G.Episode(et, fx.sq49); mt = et_ep.menu(); gt = et_ep.goal()
    uses = find(mt, G.Kind.USE); costs = [S.default_cost(et, gt, mt, a) for a in uses]
    expect(costs == [S.KIND_COST[G.Kind.USE] + (a.args[0] + 1).bit_length() - 1 for a in uses] and costs[0] == 1 and costs == sorted(costs) and costs[-1] > costs[0], 'J  USE costs one step and the binary logarithm of the term\'s place in the pool: %s' % costs[:10])
    el = G.Episode(et, fx.add); ml = el.menu(); gl = el.goal()
    i2 = find(ml, G.Kind.INSTANTIATE)
    expect(i2 and all(S.default_cost(et, gl, ml, a) == S.KIND_COST[G.Kind.INSTANTIATE] + sum((i + 1).bit_length() - 1 for i in a.args) for a in i2), 'J  INSTANTIATE costs its step and the same for each of its terms')
    # the kinds are ordered as intended
    expect(S.KIND_COST[G.Kind.NLINARITH] > S.KIND_COST[G.Kind.LINARITH] and S.KIND_COST[G.Kind.INDUCT] >= S.KIND_COST[G.Kind.CASES] and all(v >= 1 for v in S.KIND_COST.values()) and set(S.KIND_COST) == set(G.Kind), 'J  every kind has a cost of at least one step; nlinarith costs more than linarith')
    trace('J done')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ K. the lemmas apply offers
def sec_K(c):
    fx = c['fx']; w, k = fx.w, fx.k; Z, N, Nat = fx.Z, fx.N, fx.Nat
    eng = G.Engine(w, kinds=[G.Kind.APPLY, G.Kind.RING], strict=True)
    fp = lambda name: eng.lemma_fps(w.gid[name])
    expect(eng.type_fp(Z.n) == w.gid['Quot'] and eng.type_fp(N.n) == w.gid['N'] and eng.type_fp(Nat.n) == w.gid['Nat'], 'K  a type is told by its inductive family, or by the quotient behind its definition (Z)')
    expect(eng.type_fp(fx.ZZ.n) is None and eng.type_fp(w['Z.add'].n) is None, 'K  a function type and a function are no type of this kind')
    expect(fp('Z.add_zero') == (w.gid['Quot'], None, None) and fp('N.add_zero') == (w.gid['N'], None, None) and fp('add_zero') == (w.gid['Nat'], None, None) and fp('refl') == (None, None, None),
           'K  the fingerprint of a lemma: the types in the arguments of its conclusion (%s %s %s)' % (fp('Z.add_zero'), fp('N.add_zero'), fp('refl')))
    expect(G.fps_compatible((1, None), (1, 5)) and G.fps_compatible((None, 2), (4, 2)) and not G.fps_compatible((1, None), (2, None)) and G.fps_compatible((), (3,)), 'K  two fingerprints clash where both name a type and the types differ')
    def eq_goal(ty):
        with w.nota.using(ty): return w.forall_([ty, ty], lambda a, b: w.Eq(a + b, b + a))
    def names(mm): return {w.gname.get(k.view(mm.cands[a.arg][0])[1]) for a in find(mm, G.Kind.APPLY) if k.view(mm.cands[a.arg][0])[0] == 'const'}
    lost = []; nlemmas = {}
    for tname, ty, own, others in (('Z', Z, 'Z.add_zero', ('N.add_zero', 'add_zero')), ('N', N, 'N.add_zero', ('Z.add_zero', 'add_zero')), ('Nat', Nat, 'add_zero', ('Z.add_zero', 'N.add_zero'))):
        ep = G.Episode(eng, eq_goal(ty)); g = ep.goal(); m_on = ep.menu(); nm = names(m_on); nlemmas[tname] = len(nm)
        expect(own in nm, 'K  on an equation of %s the lemmas about %s are offered (%s)' % (tname, tname, own))
        expect(not any(o in nm for o in others), 'K  ... and none about the other types (%s)' % sorted(o for o in others if o in nm))
        expect({'refl', 'symm'} <= nm, 'K  ... and the lemmas that are about every type')
        eng.type_filter = False
        try: m_off = eng.menu(g)
        finally: eng.type_filter = True
        expect(len(names(m_off)) > len(nm), 'K  without the filter there are more lemmas on offer than with it (%d, %d)' % (len(names(m_off)), len(nm)))
        for a in find(m_off, G.Kind.APPLY):
            term = m_off.cands[a.arg][0]
            if k.view(term)[0] != 'const': continue
            nme = w.gname.get(k.view(term)[1])
            if nme in nm: continue
            if eng.probe_status(g, a, m_off)[1] == G.ST_OK: lost.append((tname, nme))
    expect(not lost, 'K  no lemma that the filter removed would have applied (%s)' % (lost,))
    # the introduction rules
    for nm_, want in (('conj', True), ('ex_intro', True), ('add_eq_zero', True), ('and_comm', False), ('or_comm', False), ('Z.add_zero', False), ('refl', False), ('symm', False), ('f_equal', False)):
        if nm_ in w.gid: expect(eng.introduces(w.gid[nm_]) is want, 'K  %s %s an introduction rule' % (nm_, 'is' if want else 'is not'))
    # the cost: a hypothesis or an introduction rule is one step, any other lemma one more
    with w.nota.using(Z): both = w.forall_([Z, Z], lambda a, b: w.And(w.Eq(a, b), w.Eq(b, a)))
    ep = G.Episode(eng, both); g = ep.goal(); m = ep.menu()
    cost_of = {w.gname.get(k.view(m.cands[a.arg][0])[1]): S.default_cost(eng, g, m, a) for a in find(m, G.Kind.APPLY) if k.view(m.cands[a.arg][0])[0] == 'const'}
    expect(cost_of.get('conj') == S.KIND_COST[G.Kind.APPLY] and cost_of.get('and_comm') == S.KIND_COST[G.Kind.APPLY] + 1, 'K  applying conj costs a step, applying and_comm one more (%s)' % ({n: cost_of.get(n) for n in ('conj', 'and_comm')},))
    ep = G.Episode(eng, fx.prem); g = ep.goal(); m = ep.menu()
    hyp = [a for a in find(m, G.Kind.APPLY) if k.view(m.cands[a.arg][0])[0] == 'fvar']; lem = [a for a in find(m, G.Kind.APPLY) if k.view(m.cands[a.arg][0])[0] == 'const']
    if expect(hyp and lem, 'K  (a goal with a hypothesis to apply and lemmas on offer)'):
        expect(all(S.default_cost(eng, g, m, a) == S.KIND_COST[G.Kind.APPLY] for a in hyp) and all(S.default_cost(eng, g, m, a) == S.KIND_COST[G.Kind.APPLY] + 1 for a in lem if not eng.introduces(k.view(m.cands[a.arg][0])[1])),
               'K  applying a hypothesis costs a step, applying a lemma of the library one more')
    trace('K done (lemmas on offer on an equation: %s)' % nlemmas)


SECTIONS = [('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E), ('F', sec_F), ('G', sec_G), ('H', sec_H), ('I', sec_I), ('J', sec_J), ('K', sec_K)]


def main():
    t_start = time.time()
    c = {'fx': Fixture(), 'w3': nat_lib.build_nat()}
    only = os.environ.get('S38_ONLY')
    for name, fn in SECTIONS:
        if only and name not in only.split(','): continue
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            import traceback
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:200])); traceback.print_exc(limit=6)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
        if FAIL and os.environ.get('S38_FAILFAST'): break           # (for mutation testing, where one failure is all that is wanted)
    print('S38: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:60]: print('  FAIL:', f)
    print('S38 TERMS GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
