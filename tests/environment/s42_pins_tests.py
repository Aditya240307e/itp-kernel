"""S42 gate: PINS -- small direct tests of the boundaries and corner cases of the typed term builder (gym_build.py) and of the action language (gym_acts.py) that the larger gates (S36, S38, S40) only cross
by accident.

The mutation driver (mutate.py) showed which behaviours no gate pinned down: a limit off by one, a check that is never reached by the actions a library proof needs, a cache key, a fallback that is only
taken for terms that are convertible and not equal, the shape a statement must have to be an induction principle. None of them can make a false theorem (the kernel checks every term and every proof),
but each is part of the documented behaviour of the interface an agent acts through, and a change in one of them changes what an agent can say or is told. Each section below pins a group; every check
is a small scenario with ONE expected outcome, accepted as well as refused (a check that only refuses cannot tell a strict implementation from one that refuses everything).

  A. SHAPES.        `check_shape`: what codes, hypotheses and arguments each kind of action takes, at the limits.
  B. SOURCES.       `ref_of`: a lemma or a hypothesis named by a one-token code; `positional_args`, `eq_sides`.
  C. MATCHING.      `match`, `match_any`, `variants`, `holes_of`: the first-order matcher with its convertibility fallback and its unfolding of definitions at the head.
  D. THE TABLE.     `HeadTable`: which globals a term may start with; extended when the library grows; never made inside a probe.
The sections below are the ones this gate has (A-D). Planned and DEFERRED (the mutation work that motivated them is parked until the first agents have trained): E. THE MENU (`Builder.tokens`: what is
offered at a hole -- heads, numerals, goals, pool terms -- and `push`: what is refused), F. PRINCIPLES (`analyze_principle`, `induct_plan`), G. SLOTS (`resolve_slots`), H. FUEL (a spent work budget is
the answer of the whole call). The drafts are not in the tree; the open mutation survivors are listed in ROADMAP.md.

Environment: S42_TRACE=1 prints the sections, S42_ONLY=A,B runs some of them, S42_FAILFAST=1 stops at the first failure.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, time, traceback
sys.setrecursionlimit(100000)
import itp_env_lnl as E
import imo_lib
import gym_env as G, gym_build as GB, gym_acts as A, gym_terms as GT
from gym_env import Kind, Action, ActionError
from author import AuthorError

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S42_TRACE'))
FAILFAST = bool(os.environ.get('S42_FAILFAST'))
ONLY = set(os.environ['S42_ONLY'].split(',')) if os.environ.get('S42_ONLY') else None


def expect(cond, what):
    CHECKS[0] += 1
    if not cond:
        FAIL.append(what)
        if FAILFAST: print('FAIL: ' + what, flush=True); report(); sys.exit(1)
    return bool(cond)


REFUSALS = (ActionError, AuthorError, E.ITPError, GB.BuildError)


def refuses(what, fn, exc=REFUSALS):
    """fn() must raise one of exc"""
    try: fn()
    except exc: return expect(True, what)
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:100]))
    return expect(False, what + ' (it raised nothing)')


def accepts(what, fn):
    """fn() must not raise"""
    try: fn()
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:100]))
    return expect(True, what)


def section(name):
    if ONLY is not None and name not in ONLY: return False
    if TRACE: print('-- section %s' % name, flush=True)
    return True


def report():
    print('S42: %d checks, %d failures' % (CHECKS[0], len(FAIL)))
    for f in FAIL[:40]: print('  FAIL: ' + f)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ fixture
class Fixture:
    """the library, one engine that offers every kind of action, a few helpers to make statements and goals"""
    def __init__(self):
        t0 = time.time()
        self.w = imo_lib.build_imo(); self.k = self.w.k
        self.eng = G.Engine(self.w, kinds=set(Kind), strict=True)
        self.n_lib = self.k.num_globals()
        s = self.w
        self.Z = s['Z']; self.N = s['N']; self.Eq = s.Eq
        if TRACE: print('   [fixture %.1f s, %d globals]' % (time.time() - t0, self.n_lib), flush=True)
    def gid(self, name): return self.w.gid[name]
    def start(self, stmt, premises=None):
        self.eng.premises = premises
        p = self.eng.start(stmt)
        return p, self.eng.goal_of(p, 0)
    def space(self, hyp_types=(), names=None, pool=None, roots=()):
        return GB.TermSpace(self.eng, list(hyp_types), None if names is None else list(names), pool=pool, roots=list(roots))
    def num(self, ty, v):
        inst = GB.R.instance_for(self.w, self.w.T(ty if not hasattr(ty, 'n') else ty.n))
        return inst.lit(v).n


F = None


def act(kind, *codes, arg=0, hyps=(), args=()):
    return Action(kind, arg, tuple(args), tuple(tuple(c) for c in codes), tuple(hyps))


def shape(action, n_hyps=3):
    """does check_shape accept it?"""
    try: A.check_shape(F.eng, action, n_hyps)
    except ActionError: return False
    return True


TOK = ('n', 1)
CODE1 = (TOK,)
SRC = (('c', 5, 0),)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. shapes
def sec_A():
    if not section('A'): return
    # --- what a code is: a tuple of tokens, a token a tuple of at most 3 elements whose first is its name
    lim = GB.MAX_CODE
    expect(shape(act(Kind.CLAIM, (TOK,) * lim)), 'A1 a code of exactly MAX_CODE tokens is a code')
    expect(not shape(act(Kind.CLAIM, (TOK,) * (lim + 1))), 'A1 a code of MAX_CODE + 1 tokens is not')
    expect(shape(act(Kind.CLAIM, (('lam',),))), 'A2 a token of one element is a token')
    expect(shape(act(Kind.CLAIM, (('c', 3, 2),))), 'A2 a token of three elements is a token')
    expect(not shape(act(Kind.CLAIM, (('c', 3, 2, 1),))), 'A2 a token of four elements is not')
    expect(not shape(act(Kind.CLAIM, ((),))), 'A2 an empty token is not')
    expect(not shape(act(Kind.CLAIM, ((1, 2, 3),))), 'A2 a token whose name is not a string is not')
    expect(not shape(act(Kind.CLAIM, (['n', 1],))), 'A2 a token that is a list is not (codes are tuples of tuples)')
    expect(not shape(Action(Kind.CLAIM, 0, (), [CODE1], ())), 'A2 the codes of an action are a tuple, not a list')
    expect(not shape(Action(Kind.CLAIM, 0, (), (list(CODE1),), ())), 'A2 a code is a tuple, not a list')
    # --- the number of codes and of hypotheses
    mc = A.MAX_CODES
    expect(shape(act(Kind.FORWARD, SRC, *([()] * (mc - 1)))), 'A3 the most codes an action takes (%d) are accepted' % mc)
    expect(not shape(act(Kind.FORWARD, SRC, *([()] * mc))), 'A3 one code more is refused')
    mh = A.MAX_HYPS
    expect(shape(act(Kind.INDUCT, hyps=tuple(range(1, mh + 1))), n_hyps=mh + 10), 'A4 the most hypotheses an action takes (%d) are accepted' % mh)
    expect(not shape(act(Kind.INDUCT, hyps=tuple(range(1, mh + 2))), n_hyps=mh + 10), 'A4 one hypothesis more is refused')
    expect(shape(act(Kind.CASES, hyps=(1, 2), arg=0)), 'A5 CASES generalising two hypotheses other than the one split')
    expect(not shape(act(Kind.CASES, hyps=(1, 1))), 'A5 the hypotheses to generalise are distinct')
    expect(not shape(act(Kind.CASES, hyps=(0, 1), arg=0)), 'A5 the hypothesis split is not one of those generalised')
    expect(not shape(act(Kind.INDUCT, hyps=(True,))), 'A6 a boolean is no hypothesis position')
    expect(not shape(act(Kind.INDUCT, hyps=(-1,))), 'A6 a negative position is none')
    expect(not shape(act(Kind.INDUCT, hyps=(3,)), n_hyps=3), 'A6 a position past the last hypothesis is none')
    expect(shape(act(Kind.INDUCT, hyps=(2,)), n_hyps=3), 'A6 the last hypothesis is one')
    expect(not shape(Action(Kind.INDUCT, 0, (), (), [1])), 'A6 the hypotheses are a tuple, not a list')
    try:
        import numpy as np
        expect(shape(Action(Kind.INDUCT, 0, (), (), (np.int64(1),))), 'A6 a numpy integer is a position')
    except ImportError: pass
    expect(not shape(act(Kind.CASES, CODE1)), 'A7 CASES takes no codes')
    expect(not shape(act(Kind.INDUCT, CODE1)), 'A7 INDUCT takes no codes')
    # --- pool terms (args) belong to INSTANTIATE and USE only
    for kd, cs, hs in ((Kind.CLAIM, (CODE1,), ()), (Kind.REFINE, (CODE1,), ()), (Kind.FORWARD, (SRC,), ()), (Kind.APPLY_WITH, (SRC,), ()), (Kind.REWRITE_WITH, (SRC,), ()),
                       (Kind.REWRITE_AT, (SRC,), (1,)), (Kind.INDUCT_WITH, (SRC,), (1,)), (Kind.LINCOMB, (CODE1,), (1,))):
        expect(shape(act(kd, *cs, hyps=hs)), 'A8 %s with its codes' % kd.name)
        expect(not shape(act(kd, *cs, hyps=hs, args=(0,))), 'A8 %s takes no pool terms' % kd.name)
    # --- USE: a pool position, or a witness code, not both
    expect(shape(Action(Kind.USE, 0, (0,), (), ())), 'A9 USE of a pool term')
    expect(shape(act(Kind.USE, CODE1)), 'A9 USE of a witness code')
    expect(not shape(act(Kind.USE, CODE1, args=(0,))), 'A9 USE with a witness code AND a pool term')
    expect(not shape(act(Kind.USE, CODE1, CODE1)), 'A9 USE takes exactly one witness')
    expect(not shape(act(Kind.USE, ())), 'A9 USE takes a witness that is not empty')
    expect(not shape(act(Kind.USE, CODE1, hyps=(1,))), 'A9 USE takes no hypotheses')
    expect(not shape(Action(Kind.USE, 0, (0,), (), (1,))), 'A9 USE of a pool term takes no hypotheses either')
    # --- the other kinds of the first language take neither codes nor hypotheses
    for kd in (Kind.RING, Kind.LINARITH, Kind.NLINARITH, Kind.BY_CONTRA, Kind.APPLY, Kind.REWRITE, Kind.INTRO, Kind.INSTANTIATE, Kind.INTERVAL_CASES, Kind.INJECT):
        expect(shape(Action(kd, 0, (), (), ())), 'A10 %s with nothing else' % kd.name)
        expect(not shape(act(kd, CODE1)), 'A10 %s takes no codes' % kd.name)
        expect(not shape(act(kd, hyps=(1,))), 'A10 %s takes no hypotheses' % kd.name)
    # --- CLAIM and REFINE: exactly one non-empty code, no hypotheses, argument 0
    for kd in (Kind.CLAIM, Kind.REFINE):
        n = kd.name
        expect(shape(act(kd, CODE1)), 'A11 %s of one code' % n)
        expect(not shape(act(kd)), 'A11 %s needs a code' % n)
        expect(not shape(act(kd, CODE1, CODE1)), 'A11 %s takes one code, not two' % n)
        expect(not shape(act(kd, ())), 'A11 the code of %s is not empty' % n)
        expect(not shape(act(kd, CODE1, hyps=(1,))), 'A11 %s takes no hypotheses' % n)
        expect(not shape(act(kd, CODE1, arg=1)), 'A11 the argument of %s is 0' % n)
    # --- sources and their arguments
    for kd in (Kind.FORWARD, Kind.APPLY_WITH, Kind.REWRITE_WITH, Kind.REWRITE_AT, Kind.INDUCT_WITH):
        n = kd.name; hs = (1,) if kd in (Kind.REWRITE_AT, Kind.INDUCT_WITH) else ()
        expect(shape(act(kd, SRC, hyps=hs)), 'A12 %s of a source alone' % n)
        expect(shape(act(kd, SRC, CODE1, hyps=hs)), 'A12 %s of a source and an argument' % n)
        expect(not shape(act(kd, (), hyps=hs)), 'A12 the source of %s is not empty' % n)
        expect(not shape(act(kd, hyps=hs)), 'A12 %s needs a source' % n)
        expect(len(hs) == 0 or not shape(act(kd, SRC)), 'A12 %s names a hypothesis' % n)
        expect(len(hs) == 1 or not shape(act(kd, SRC, hyps=(1,))), 'A12 %s names no hypothesis' % n)
        expect(len(hs) == 0 or not shape(act(kd, SRC, hyps=(1, 2))), 'A12 %s names ONE hypothesis' % n)
        if kd != Kind.FORWARD: expect(not shape(act(kd, SRC, (), hyps=hs)), 'A12 an argument of %s is not empty' % n)
        else: expect(shape(act(kd, SRC, (), CODE1)), 'A12 a slot of FORWARD may be empty (an argument the others determine)')
    # --- the argument of a kind
    for kd, ok_args in ((Kind.REWRITE_WITH, (0, 1)), (Kind.REWRITE_AT, (0, 1)), (Kind.APPLY_WITH, (0, 1))):
        hs = (1,) if kd == Kind.REWRITE_AT else ()
        for a in ok_args: expect(shape(act(kd, SRC, arg=a, hyps=hs)), 'A13 %s with argument %d' % (kd.name, a))
        for a in (2, -1): expect(not shape(act(kd, SRC, arg=a, hyps=hs)), 'A13 %s with argument %d' % (kd.name, a))
    for kd, cs, hs in ((Kind.FORWARD, (SRC,), ()), (Kind.INDUCT_WITH, (SRC,), (1,)), (Kind.LINCOMB, (CODE1,), (1,))):
        expect(not shape(act(kd, *cs, hyps=hs, arg=1)), 'A13 the argument of %s is 0' % kd.name)
    # --- LINCOMB: one non-empty coefficient for each equation, at least one
    expect(shape(act(Kind.LINCOMB, CODE1, CODE1, hyps=(0, 1))), 'A14 LINCOMB with a coefficient for each of two equations')
    expect(not shape(act(Kind.LINCOMB, CODE1, hyps=(0, 1))), 'A14 LINCOMB with fewer coefficients than equations')
    expect(not shape(act(Kind.LINCOMB, CODE1, CODE1, hyps=(0,))), 'A14 LINCOMB with more coefficients than equations')
    expect(not shape(act(Kind.LINCOMB)), 'A14 a linear combination of nothing is none')
    expect(not shape(act(Kind.LINCOMB, (), hyps=(0,))), 'A14 a coefficient is not empty')
    expect(not shape(act(Kind.LINCOMB, CODE1, (), hyps=(0, 1))), 'A14 no coefficient of a linear combination is empty')
    # --- an unknown kind is not an action (check_action says so before check_shape is reached)
    refuses('A15 an unknown kind', lambda: F.eng.check_action(Action(99, 0, (), (), ()), A.NO_MENU, 3))
    refuses('A15 a boolean argument', lambda: F.eng.check_action(Action(Kind.RING, True, (), (), ()), A.NO_MENU, 3))
    refuses('A15 hypothesis past the last', lambda: F.eng.check_action(Action(Kind.INJECT, 3, (), (), ()), A.NO_MENU, 3))
    accepts('A15 the last hypothesis', lambda: F.eng.check_action(Action(Kind.INJECT, 2, (), (), ()), A.NO_MENU, 3))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ small kernel helpers
def kmk(f, *args): return f(*args)


class T:
    """terms of the fixture world as kernel handles: T.app(name, args..), T.fv(level), T.num(Z|N, v)"""
    @staticmethod
    def const(name): return F.k.mk_const(F.w.gid[name])
    @staticmethod
    def app(head, *args):
        n = T.const(head) if isinstance(head, str) else head
        for a in args: n = F.k.mk_app(n, a)
        return n
    @staticmethod
    def fv(level): return F.k.mk_fvar(level)
    @staticmethod
    def num(tyname, v): return GB.R.instance_for(F.w, F.w[tyname]).lit(v).n
    @staticmethod
    def ty(l): return F.k.mk_type(l)
    @staticmethod
    def pi(dom, cod): return F.k.mk_pi(dom, cod)


def fresh_space(types=(), names=None, pool=None):
    return GB.TermSpace(F.eng, list(types), None if names is None else list(names), pool=pool)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. sources
def sec_B():
    if not section('B'): return
    Zt = T.const('Z'); k = F.k; eng = F.eng
    sp = fresh_space([Zt, Zt, T.const('N')], ['a', 'b', 'n'])
    # --- ref_of: a source is a code of one token, ('v', level, 0) or ('c', id, 0)
    def ref(code):
        eng.premises = None
        return A.ref_of(eng, sp, code)
    h0, n0 = ref((('v', 0, 0),))
    expect(n0 == k.mk_fvar(0) and h0.tag == 'v' and h0.ident == 0, 'B1 a hypothesis is a source: the variable and its head')
    expect(ref((('v', 2, 0),))[1] == k.mk_fvar(2), 'B1 the last hypothesis is a source')
    refuses('B1 the hypothesis after the last is none', lambda: ref((('v', 3, 0),)))
    refuses('B1 a negative hypothesis number is none', lambda: ref((('v', -1, 0),)))
    refuses('B1 a boolean is no hypothesis number', lambda: ref((('v', True, 0),)))
    refuses('B1 a boolean is no global number', lambda: ref((('c', True, 0),)))
    refuses('B1 a float is no number', lambda: ref((('v', 1.0, 0),)))
    refuses('B1 a source takes no arguments', lambda: ref((('v', 0, 1),)))
    refuses('B1 a source is not a numeral', lambda: ref((('n', 3),)))
    refuses('B1 a source is one token', lambda: ref((('v', 0, 0), ('v', 1, 0))))
    refuses('B1 a source is not empty', lambda: ref(()))
    refuses('B1 a source is a tuple', lambda: ref([('v', 0, 0)]))
    refuses('B1 an unknown tag is no source', lambda: ref((('x', 0, 0),)))
    g = F.gid('Z.add_comm')
    h1, n1 = ref((('c', g, 0),))
    expect(n1 == k.mk_const(g) and h1.tag == 'c' and h1.ident == g, 'B2 a global is a source')
    refuses('B2 a global beyond the library is none', lambda: ref((('c', k.num_globals(), 0),)))
    refuses('B2 a negative global is none', lambda: ref((('c', -1, 0),)))
    eng.premises = g + 1
    try:
        accepts('B3 the last global below the premises limit is a source', lambda: A.ref_of(eng, sp, (('c', g, 0),)))
        refuses('B3 the premises limit hides the global at the limit', lambda: A.ref_of(eng, sp, (('c', g + 1, 0),)))
        refuses('B3 ... and all those above it', lambda: A.ref_of(eng, sp, (('c', g + 5, 0),)))
        refuses('B3 ... also the last global of the library', lambda: A.ref_of(eng, sp, (('c', k.num_globals() - 1, 0),)))
    finally: eng.premises = None
    # --- positional_args: the arguments are built against the binders in turn, later binders see the earlier arguments
    dvd_add = F.gid('Z.dvd_add'); ty = k.global_info(dvd_add)[1]                 # forall x0 x1 x2, x0 | x1 -> x0 | x2 -> x0 | x1 + x2
    sp3 = fresh_space([Zt, Zt, T.app('Z.dvd', T.fv(0), T.fv(1))], ['a', 'b', 'h'])
    terms = A.positional_args(eng, sp3, ty, [(('v', 0, 0),), (('v', 1, 0),), (('v', 1, 0),), (('v', 2, 0),)])
    expect(terms == [T.fv(0), T.fv(1), T.fv(1), T.fv(2)], 'B4 positional arguments: three numbers and a proof whose type mentions the first two (the earlier arguments are substituted into the later binders)')
    refuses('B4 a proof of the wrong statement for a later binder', lambda: A.positional_args(eng, sp3, ty, [(('v', 1, 0),), (('v', 0, 0),), (('v', 1, 0),), (('v', 2, 0),)]))
    refuses('B4 more arguments than binders', lambda: A.positional_args(eng, sp3, k.global_info(F.gid('Z.add_comm'))[1], [(('n', 1),), (('n', 2),), (('n', 3),)]))
    expect(A.positional_args(eng, sp3, k.global_info(F.gid('Z.add_comm'))[1], []) == [], 'B4 no arguments, no terms')
    # --- eq_sides: only an equation (the constant Eq with three arguments) has sides
    a, b_ = T.fv(0), T.fv(1)
    eqt = T.app('Eq', Zt, a, b_)
    expect(list(A.eq_sides(eng, eqt)) == [Zt, a, b_], 'B5 the sides of an equation (the carrier, left, right)')
    expect(A.eq_sides(eng, T.app('Z.is_gcd', a, b_, T.fv(2))) is None, 'B5 another constant with three arguments has no sides')
    expect(A.eq_sides(eng, T.app('Z.dvd', a, b_)) is None, 'B5 a divisibility has no sides')
    expect(A.eq_sides(eng, T.app('Eq', Zt, a)) is None, 'B5 an equation with an argument missing has none')
    expect(A.eq_sides(eng, Zt) is None, 'B5 a type has none')
    expect(A.eq_sides(eng, k.mk_fvar(0)) is None, 'B5 a variable has none')
    # --- expose_pi (acts): a function type, also behind a definition
    notp = T.app('Not', T.app('Z.lt', a, b_))
    e = A.expose_pi(k, notp)
    expect(e is not None and e[0] == T.app('Z.lt', a, b_) and e[1] == T.const('False'), 'B6 a definition at the head that hides a function type is unfolded (Not P is P -> False)')
    expect(A.expose_pi(k, Zt) is None and A.expose_pi(k, T.app('Z.lt', a, b_)) is None, 'B6 a type that is no function has none (also one whose unfoldings are not functions)')
    e2 = A.expose_pi(k, T.pi(Zt, Zt))
    expect(e2 is not None and e2[0] == Zt, 'B6 a function type is its own')
    # --- conv_fn: a bounded convertibility test; running out of work is "no"
    conv = A.conv_fn(eng, 0)
    seven = T.num('Z', 7); fortynine = T.num('Z', 49)
    expect(conv(T.app('Z.mul', seven, seven), fortynine), 'B7 7 * 7 and 49 are convertible')
    expect(not conv(T.app('Z.mul', seven, seven), T.num('Z', 50)), 'B7 7 * 7 and 50 are not')
    big = T.app('Z.pow', T.num('Z', 3), T.num('N', 2000))
    expect(not conv(T.app('Z.add', big, T.num('Z', 1)), T.app('Z.add', big, T.num('Z', 2))), 'B7 a conversion that runs out of work is not a "yes" (3^2000 + 1 against 3^2000 + 2)')
    expect(k.get_fuel() == eng.k.get_fuel(), 'B7 (the fuel is restored)')
    f_ = T.fv(0)
    conv1 = A.conv_fn(eng, 1)
    expect(conv1(T.app(f_, T.app('Z.mul', seven, seven)), T.app(f_, fortynine)) is True, 'B7 a context variable may occur in the terms (the context has one variable)')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. matching
def sec_C():
    if not section('C'): return
    k = F.k; Zt = T.const('Z'); HOLE = GB.HOLE
    v0, v1 = T.fv(HOLE), T.fv(HOLE + 1)
    seven, fortynine = T.num('Z', 7), T.num('Z', 49)
    conv = A.conv_fn(F.eng, 0)
    # --- match: a pattern variable that occurs twice must meet the same term twice -- or, with a conversion test, terms that are convertible
    pat = T.app('Eq', Zt, v0, v0)
    same = T.app('Eq', Zt, fortynine, fortynine)
    cvt = T.app('Eq', Zt, T.app('Z.mul', seven, seven), fortynine)
    diff = T.app('Eq', Zt, T.app('Z.mul', seven, seven), T.num('Z', 50))
    expect(GB.match(k, pat, same, {}) and GB.match(k, pat, same, {}, conv), 'C1 a variable meeting the same term twice')
    b_ = {}; GB.match(k, pat, same, b_)
    expect(b_ == {HOLE: fortynine}, 'C1 ... is bound to it')
    expect(not GB.match(k, pat, cvt, {}), 'C1 two convertible terms are two terms without a conversion test')
    expect(GB.match(k, pat, cvt, {}, conv), 'C1 ... and the same with one')
    expect(not GB.match(k, pat, diff, {}, conv), 'C1 two terms that are not convertible are two terms with one')
    expect(not GB.match(k, T.app('Eq', Zt, v0, v1), T.app('Eq', T.const('N'), fortynine, fortynine), {}), 'C1 a carrier that differs is no match')
    expect(not GB.match(k, T.app('Z.add', v0, v1), T.app('Z.mul', seven, seven), {}), 'C1 another head is no match')
    expect(not GB.match(k, T.app('Z.neg', v0), T.app('Z.neg', k.mk_bvar(0)), {}), 'C1 a pattern variable is not bound to a term with a loose bound variable')
    # --- variants: the type itself and what unfolding the definitions at its head gives, one step at a time
    a, b = T.fv(0), T.fv(1)
    lt = T.app('Z.lt', a, b)
    vs = GB.variants(k, lt)
    expect(len(vs) == 3 and vs[0] == lt, 'C2 a < b has two unfoldings (a + 1 <= b, then an existential): %d variants' % len(vs))
    expect(len(GB.variants(k, lt, count=1)) == 2 and len(GB.variants(k, lt, count=0)) == 1, 'C2 the number of unfoldings is limited by the count')
    expect(GB.variants(k, T.const('N')) == [T.const('N')], 'C2 a type that is no definition is its own and only variant')
    expect(len(GB.variants(k, T.app('Z.dvd', a, b))) == 2, 'C2 a divisibility has one unfolding')
    expect(GB.VARIANTS >= 3, 'C2 (the default count reaches the existential behind <)')
    # --- match_any: the pattern against the type or what unfolding its head gives
    dvd = T.app('Z.dvd', a, b); ex_dvd = GB.variants(k, dvd)[1]
    pat_ex = T.app('Ex', v0, v1)                                            # Ex ?A ?P
    bnd = {}
    expect(GB.match_any(k, pat_ex, dvd, bnd) and bnd == {HOLE: Zt, HOLE + 1: k.view(ex_dvd)[2]}, 'C3 a pattern that only the unfolded type matches; the failed attempt on the type leaves no trace in the binding')
    bnd = {HOLE + 7: Zt}
    GB.match_any(k, pat_ex, dvd, bnd)
    expect(bnd.get(HOLE + 7) == Zt and len(bnd) == 3, 'C3 match_any extends the binding it is given')
    bnd = {HOLE: T.const('N')}
    expect(not GB.match_any(k, pat_ex, dvd, bnd) and bnd == {HOLE: T.const('N')}, 'C3 on failure the binding is as it was (also when an attempt had bound more)')
    expect(not GB.match_any(k, T.app('Z.le', v0, v1), Zt, {}), 'C3 no match at all')
    pat_dvd = T.app('Z.dvd', v0, v1)
    bnd = {}
    expect(A.match_any(k, pat_dvd, ex_dvd, bnd) and bnd == {HOLE: a, HOLE + 1: b}, 'C4 (acts) the pattern is unfolded as well: a divisibility against the existential it means')
    expect(A.match_any(k, pat_ex, dvd, {}), 'C4 (acts) ... and the type is unfolded')
    bnd = {HOLE: T.const('N')}
    expect(not A.match_any(k, pat_dvd, ex_dvd, bnd) and bnd == {HOLE: T.const('N')}, 'C4 (acts) on failure the binding is as it was')
    # --- holes_of: the pattern variables a term mentions (the context variables with levels from HOLE up), and nothing else
    expect(GB.holes_of(k, T.app('Eq', Zt, v0, T.app('Z.add', v1, a))) == {HOLE, HOLE + 1}, 'C5 the pattern variables of a term')
    expect(GB.holes_of(k, T.app('Z.add', a, b)) == set(), 'C5 a term without any')
    expect(GB.holes_of(k, k.mk_bvar(HOLE + 5)) == set(), 'C5 a bound variable with a large index is no pattern variable')
    expect(GB.holes_of(k, T.fv(HOLE - 1)) == set() and GB.holes_of(k, T.fv(HOLE)) == {HOLE}, 'C5 the first pattern variable is the first level from HOLE')
    # --- open_pis, sym_of
    ty = k.global_info(F.gid('Z.dvd_trans'))[1]
    doms, concl = GB.open_pis(k, ty)
    expect(len(doms) == 5 and concl == T.app('Z.dvd', v0, T.fv(HOLE + 2)), 'C6 the binders of a lemma become pattern variables (3 numbers, 2 premises) and what is left is the conclusion')
    doms2, concl2 = GB.open_pis(k, ty, count=2)
    expect(len(doms2) == 2, 'C6 ... as many as asked for')
    d3, c3 = GB.open_pis(k, T.app('Not', lt))
    expect(len(d3) == 1 and c3 == T.const('False'), 'C6 a definition at the head that hides a function type: Not P has the binder P and the rest False')
    expect(GB.sym_of(k, Zt) == ('c', F.gid('Z')) and GB.sym_of(k, T.pi(Zt, Zt)) == 'pi' and GB.sym_of(k, T.ty(0)) == ('t', 0) and GB.sym_of(k, T.fv(3)) == ('v', 3)
           and GB.sym_of(k, T.fv(HOLE)) is None and GB.sym_of(k, T.app(T.fv(HOLE), a)) is None, 'C7 the symbol a type is headed by: a constant, a function type, a sort, a variable; a pattern variable is a wildcard')
    # --- beta_head, inst_beta: a motive said as a lambda
    lam = k.mk_lambda(Zt, T.app('Eq', Zt, k.mk_bvar(0), a))
    red = GB.beta_head(k, T.app(lam, b))
    expect(red == T.app('Eq', Zt, b, a), 'C8 the beta-redex at the head is contracted')
    expect(GB.beta_head(k, T.app('Z.neg', b)) == T.app('Z.neg', b), 'C8 nothing to contract')
    ib = GB.inst_beta(k, T.app(T.fv(HOLE), b), {HOLE: lam})
    expect(ib == T.app('Eq', Zt, b, a), 'C8 a pattern variable replaced by a lambda, applied: the redex is contracted')
    expect(GB.inst_beta(k, T.app('Z.add', v0, v1), {HOLE: a}) == T.app('Z.add', a, v1), 'C8 only the variables in the map are replaced')
    expect(GB.inst_beta(k, lt, {}) == lt, 'C8 an empty map changes nothing')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. the table of heads
def sec_D():
    if not section('D'): return
    w = F.w; k = F.k
    t = GB.head_table(w)
    n0 = k.num_globals()
    expect(t.scanned == n0 and F.gid('Z.add_comm') in t.heads, 'D1 the table covers the library')
    expect(GB.head_table(w) is t, 'D1 it is made once')
    PROP = w.T(k.mk_type(0))
    Zw = w['Z']
    ty = w.forall_([Zw], lambda x: w.Eq(x, x))
    w.axiom('s42_ax_visible', ty)
    g_new = w.gid['s42_ax_visible']
    expect(g_new == n0 and t.scanned == n0, 'D2 (a global is declared after the table was made)')
    t2 = GB.head_table(w)
    expect(t2 is t and t.scanned == n0 + 1 and g_new in t.heads and t.heads[g_new].arity == 1, 'D2 the table is extended when the library has grown')
    expect(any(h.ident == g_new for h in t.candidates([('c', F.gid('Eq'))])), 'D2 ... and indexed by the symbol its conclusion has')
    w.axiom('__s42_hidden', ty)
    g_hid = w.gid['__s42_hidden']
    GB.head_table(w)
    expect(g_hid not in t.heads and t.scanned == k.num_globals(), 'D3 a helper global (name starting with __) cannot be used in a term')
    # a global that is only reserved (a recursive function being defined) cannot be used either
    seen = {}
    def body(f):
        GB.head_table(w)                                                     # (made while `f` is reserved)
        seen['gid'] = w.gid['s42_rec']
        seen['in'] = seen['gid'] in w.__dict__['_head_table'].heads
        return w.lam(w['N'], lambda n: n)
    try:
        w.recursive('s42_rec', w.arrow(w['N'], w['N']), body, on=0)
    except AuthorError: pass
    expect('gid' in seen and seen['in'] is False, 'D4 a global that is being declared cannot be used in a term')
    GB.head_table(w)
    expect('gid' in seen and seen['gid'] in t.heads and not t.pending and w.k.reserved_kind(seen['gid']) == 0, 'D4 ... but once it is finished the table takes it in')
    # the table is made and extended outside probes only
    w.probing += 1
    try:
        refuses('D5 the table cannot be extended inside a probe', lambda: t.extend(), AuthorError)
        saved = w.__dict__.pop('_head_table')
        try: refuses('D5 ... nor made inside a probe', lambda: GB.head_table(w), AuthorError)
        finally: w.__dict__['_head_table'] = saved
    finally: w.probing -= 1
    # candidates: the heads whose conclusion has one of the symbols (or is a wildcard), in the order of the ids
    cands = t.candidates([('c', F.gid('Z.le'))])
    ids = [h.ident for h in cands]
    expect(ids == sorted(ids) and F.gid('Z.le_refl') in ids and F.gid('Z.add_comm') not in ids, 'D6 candidates: those with the symbol (and the wildcards), in the order of the ids')
    expect(F.gid('absurd') in [h.ident for h in t.candidates([('c', F.gid('Z.le'))])], 'D6 ... a wildcard conclusion (a pattern variable) fits every symbol')
    h = t.heads[F.gid('Z.dvd_trans')]
    expect(h.arity == 5 and len(h.doms) == 5 and h.syms == (('c', F.gid('Z.dvd')), ('c', F.gid('Ex'))), 'D7 a head: its arity and the symbols its conclusion has once unfolded, one step at a time (Z.dvd, then Ex)')
    hs = t.heads[F.gid('Z.strong_induction')]
    expect(hs.flex == 0 and hs.csort == 0, 'D7 a head whose conclusion is a motive applied to an argument knows which binder is the motive and the sort the conclusion has')
    expect(t.heads[F.gid('Z.dvd_trans')].flex is None and t.heads[F.gid('Z.dvd_trans')].csort is None, 'D7 ... a head with an ordinary conclusion has neither')
    ha = t.heads[F.gid('absurd')]
    expect(ha.flex is None and ha.csort == 0 and None in ha.syms, 'D7 a conclusion that is a bare pattern variable: a wildcard whose sort is known, and not a motive')
    expect(h.rest(k, 0) == h.ty and h.rest(k, 5) == h.concl, 'D8 the type left after kk binders: the whole type, and the conclusion')
    expect(len(GB.open_pis(k, h.rest(k, 3))[0]) == 2, 'D8 ... after three binders two are left')
    # reserved/hidden globals are skipped, the rest of the library is not
    expect(len(t.heads) > 400, 'D9 (the table has the library: %d heads)' % len(t.heads))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ main
SECTIONS = (('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D))


def main():
    global F
    t0 = time.time()
    F = Fixture()
    for name, fn in SECTIONS:
        t1 = time.time()
        try: fn()
        except Exception as e:
            expect(False, 'section %s crashed: %s: %s' % (name, type(e).__name__, str(e)[:200]))
            if TRACE or FAILFAST: traceback.print_exc()
        if TRACE: print('   [section %s %.1f s, %d checks so far, %d failures]' % (name, time.time() - t1, CHECKS[0], len(FAIL)), flush=True)
    report()
    print('S42 PINS GATE: %s (%.1f s)' % ('PASS' if not FAIL else 'FAIL', time.time() - t0))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
