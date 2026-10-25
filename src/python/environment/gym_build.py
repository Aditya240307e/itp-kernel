"""gym_build.py -- the typed term builder: any well-typed term of a goal's context, said one discrete choice at a time.

A learner with no text interface still has to be able to say a STATEMENT ("for all a, f (2 a) = 2 f a - f 0"), a WITNESS ("the integer 2 n + c"), an ARGUMENT of a lemma. The term pool (gym_terms.py) lists a
few dozen simple terms; this module is the general answer: a term is built in prefix order, one choice per step, and at every step only the choices that keep the term well typed are offered.

    space = TermSpace(eng, hyp_types, hyp_names)             # the context of a goal (kernel handles, as in GoalData)
    b = space.builder(expected_type)                         # a term of this type is wanted
    b.tokens()                                               # the choices at the first hole, in a fixed order
    b.push(token)                                            # take one; the next hole is the first unfilled one, left to right
    b.done, b.term()                                         # finished: the kernel handle
    code = space.encode(term, expected_type)                 # the canonical way to say an existing term (the inverse of `build`)
    term = space.build(code, expected_type)                  # a whole code at once

TOKENS (a CODE is a tuple of them, prefix order, no brackets: what a head is applied to is said by its number of arguments)
    ('v', level, k)     the variable of the context at this level (a hypothesis, or a binder opened by the term itself), applied to k arguments
    ('c', gid, k)       the global with this id (a constructor, a function, a predicate, a type former, a lemma), applied to k arguments
    ('n', value)        the numeral of the hole's number type (Nat, N, Z: whatever `ring` has an instance for)
    ('lam',)            a lambda: the hole must be a function type; its body is the next hole, with the argument in scope (a proof of an implication is one)
    ('all',)            a universally quantified proposition: a type (the domain), then the proposition (the body, with the bound variable in scope)
    ('imp',)            an implication: two propositions.       ('arr',)  a function type: two types.       ('s', l)  the sort Type l (a term of type Type (l+1))
    ('dep',)            an implication whose proof the conclusion mentions (rare: statements about a function that takes a proof); not listed by `tokens()` unless asked for with `rare`
    ('p', i)            the i-th term of the pool given to the space (accepted wherever its type fits; listed only when asked for)
    ('g',)              a GOAL: a proposition that is left to be proved (only when the builder allows goals, and not under a binder of the term itself); the term is then a `refine`, see `TermSpace.build_refine`
An application head takes the arguments its type needs to reach the hole's type, minus the ones that the expected type DETERMINES (the implicit arguments: the carrier of `Quot.mk A r a : Quot A r`,
the terms of `f_equal`'s conclusion): those are not said. A head may be applied to fewer arguments than its type has binders when the hole is itself a function type.

NOTHING HERE IS TRUSTED. A term built here is an ordinary kernel term; whatever is done with it is done by the tactic layer and checked by the kernel, and `build` checks the finished term against the
expected type with the kernel as well. A wrong menu wastes a step; it cannot make a false theorem appear.

WORK BUDGET.  The kernel calls an assembler makes (the check of a finished term, the sort of a hole's type) run under the budget the engine gives the action that carries the term, `Engine.default_fuel`
(5 * 10^6 steps), and not under the library's (`author.DEFAULT_FUEL`, 2 * 10^9, about half a minute and several GB: a term a learner has said at random, `Z.sumNat` of nested products of a numeral in unary,
type-checked against a proposition it does not prove, spent them all, F86). A term whose check spends the budget is REFUSED (`BuildError`), as the engine would have refused the action; inside a probe the
probe's own budget decides, and spending it is `FuelExhausted`, as everywhere (`spent`).

PROBES.  Kernel nodes made during a probe are reclaimed at its rewind. The table of the library's heads (`head_table`) is made outside probes (by `Engine.refresh`) and never stored from inside one; a
`TermSpace` and its builders are made for one goal and thrown away with it.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, collections, functools
import itp_env_lnl as E
import ring as R
from author import AuthorError

HOLE = 1 << 28                      # pattern variables of a head's telescope: context variables with levels no real context reaches (the same convention as tactics.HOLE_BASE)
MAX_CODE = 400                      # tokens in one term
MAX_NUMERAL = {'binary': 10 ** 18, 'unary': 2000}
DEFAULT_NUMERALS = tuple(range(0, 11)) + tuple(range(-1, -6, -1)) + (12, 16, 20, 24, 32, 64, 100, 1000)
MAX_GOAL_NUMERALS = 12
MAX_ARITY = 64
VARIANTS = 6                        # how many definitions at the head of a type are unfolded when looking for a match
CONV_FUEL = 20_000                  # the work budget of one conversion check made while offering or matching heads
GOAL_BASE = 1 << 27                 # the variables that stand for the goals a `refine` leaves (context levels no real context reaches; below HOLE)
MAX_GOALS = 16                      # goals one term may leave

TAGS = ('v', 'c', 'n', 'lam', 'all', 'dep', 'imp', 'arr', 's', 'p', 'g')


class BuildError(Exception):
    """a code that is not a well-typed term of the expected type (or a token that does not fit the hole)"""


def spent(e):
    """re-raise a spent work budget (call it in an `except` block that handles the kernel's errors): it is the answer of the whole call -- a probe tells "out of work" from a refusal, and a bigger budget
    may succeed -- never "this term is ill-typed" or "this is no proposition"
    """
    if isinstance(e, E.FuelExhausted): raise e


def bounded(fn):
    """a method of a `Builder` or a `TermSpace` that calls the kernel, run under the assembler's work budget: `Engine.default_fuel` steps for each call into the kernel (see WORK BUDGET above). Spending it
    is a refused term (`BuildError`). Left alone: a probe (`world.probing`: its own budget, set by the engine, decides, and spending it is `FuelExhausted`, which `spent` lets through), a caller that has set
    a budget at most as large already (the engine carrying an action out), and an engine whose default budget is unlimited (<= 0)."""
    @functools.wraps(fn)
    def run(self, *args, **kw):
        sp = getattr(self, 'space', None) or self
        k = sp.k; cap = int(sp.eng.default_fuel); saved = k.get_fuel()
        if cap <= 0 or sp.w.probing or 0 < saved <= cap: return fn(self, *args, **kw)
        k.set_fuel(cap)
        try: return fn(self, *args, **kw)
        except E.FuelExhausted: raise BuildError('the kernel needs more than %d steps to check this: refused' % cap) from None
        finally: k.set_fuel(saved)
    return run


class NotEncodable(BuildError):
    """a term that the builder cannot say (a match, a let, a head applied to more arguments than its type shows, ...)"""


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ reading terms
def spine(k, n):
    """(head, arguments) of an application, on kernel handles"""
    args = []
    while True:
        v = k.view(n)
        if v[0] != 'app': break
        args.append(v[2]); n = v[1]
    args.reverse()
    return n, args


def match(k, pat, term, binding, conv=None):
    """first-order matching of `pat` (pattern variables are the context variables with levels >= HOLE) against `term`; extends `binding` (level -> node) or fails. A pattern variable can only be bound
    to a term without loose bound variables. A variable that occurs twice must meet the same term twice -- or, with `conv`, two terms that `conv` says are convertible (`refl : a = a` for `7 * 7 = 49`)."""
    stack = [(pat, term)]
    while stack:
        a, b = stack.pop()
        va = k.view(a)
        if va[0] == 'fvar' and va[1] >= HOLE:
            if va[1] in binding:
                if binding[va[1]] != b and not (conv is not None and conv(binding[va[1]], b)): return False
            else:
                if k.loose_range(b) != 0: return False
                binding[va[1]] = b
            continue
        if a == b: continue
        vb = k.view(b)
        if va[0] != vb[0]: return False
        kind = va[0]
        if kind == 'app': stack.append((va[1], vb[1])); stack.append((va[2], vb[2]))
        elif kind in ('lam', 'pi'): stack.append((va[1], vb[1])); stack.append((va[2], vb[2]))
        elif kind == 'let': stack += [(va[1], vb[1]), (va[2], vb[2]), (va[3], vb[3])]
        elif kind == 'match':
            if len(va[3]) != len(vb[3]): return False
            stack += [(va[1], vb[1]), (va[2], vb[2])] + list(zip(va[3], vb[3]))
        elif kind == 'fix':
            if va[2] != vb[2]: return False
            stack += [(va[1], vb[1]), (va[3], vb[3])]
        elif va[1:] != vb[1:]: return False
    return True


def beta_head(k, t):
    """t with the beta-redex at its head contracted (repeatedly)"""
    while True:
        head, args = spine(k, t)
        v = k.view(head)
        if v[0] != 'lam' or not args: return t
        t = k.instantiate(v[2], [args[0]])
        for a in args[1:]: t = k.mk_app(t, a)


def inst_beta(k, t, reps):
    """`t` with the pattern variables replaced as in `reps` ({level: closed node}) and the beta-redexes that this makes contracted (`P w` for a motive P that was said as a lambda)"""
    if not reps: return t
    memo = {}
    def go(n):
        r = memo.get(n)
        if r is not None: return r
        v = k.view(n); kind = v[0]
        if kind == 'fvar': r = reps.get(v[1], n)
        elif kind == 'app':
            head, args = spine(k, n); hv = k.view(head)
            if hv[0] == 'fvar' and hv[1] in reps:
                r = reps[hv[1]]
                for a in args: r = k.mk_app(r, go(a))
                r = beta_head(k, r)
            else: r = k.mk_app(go(v[1]), go(v[2]))
        elif kind == 'lam': r = k.mk_lambda(go(v[1]), go(v[2]))
        elif kind == 'pi': r = k.mk_pi(go(v[1]), go(v[2]))
        elif kind == 'let': r = k.mk_let(go(v[1]), go(v[2]), go(v[3]))
        elif kind == 'match': r = k.mk_match(go(v[1]), go(v[2]), [go(b) for b in v[3]])
        elif kind == 'fix': r = k.mk_fix(go(v[1]), v[2], go(v[3]))
        else: r = n
        memo[n] = r
        return r
    return go(t)


def match_any(k, pat, ty, binding, conv=None):
    """match `pat` against the type `ty` or what unfolding the definitions at its head gives; on success `binding` is extended (and True returned), on failure it is left as it was"""
    for x in variants(k, ty):
        trial = dict(binding)
        if match(k, pat, x, trial, conv):
            binding.clear(); binding.update(trial)
            return True
    return False


def holes_of(k, node):
    """the pattern variables (levels) that a term mentions"""
    tags, scalar = k.observe_term(node)[:2]
    return set(scalar[(tags == E.OBS_TAG_NAMES.index('FVAR')) & (scalar >= HOLE)].tolist())


def open_pis(k, ty, base=HOLE, count=None):
    """the leading binders of `ty` opened as pattern variables base, base + 1, ...: (the domains, in terms of the variables before them; the rest). A definition at the head that unfolds to a Pi is unfolded
    (`Not P` has the binder `P` and the rest `False`)."""
    doms = []; cur = ty
    top = MAX_ARITY if count is None else min(count, MAX_ARITY)
    while len(doms) < top:
        v = k.view(cur)
        if v[0] != 'pi':
            nxt = k.unfold_head(cur)
            if nxt is None: break
            v = k.view(nxt)
            if v[0] != 'pi': break
        doms.append(v[1]); cur = k.instantiate(v[2], [k.mk_fvar(base + len(doms) - 1)])
    return doms, cur


def variants(k, ty, count=VARIANTS):
    """the type itself and what unfolding the definitions at its head gives, one step at a time (a statement `a | b` is, to a proof, an existential)"""
    out = [ty]; cur = ty
    for _ in range(count):
        cur = k.unfold_head(cur)
        if cur is None: break
        out.append(cur)
    return out


def sym_of(k, node):
    """the symbol a type is headed by, for indexing: ('t', level) for a sort; 'pi'; ('c', gid); ('v', level) for a context variable at the head; None (a wildcard) for a pattern variable, a bound variable,
    a lambda or a match at the head"""
    v = k.view(node)
    if v[0] == 'type': return ('t', v[1])
    if v[0] == 'pi': return 'pi'
    h, _ = spine(k, node); hv = k.view(h)
    if hv[0] == 'const': return ('c', hv[1])
    if hv[0] == 'fvar': return None if hv[1] >= HOLE else ('v', hv[1])
    return None


class Head:
    """something a term can start with: a variable of the context or a global. `ty`: its type; `doms`: the domains of its leading binders (`open_pis`: pattern variables stand for the earlier binders);
    `concl`: what is left; `syms`: the symbols the conclusion has once its definitions are unfolded, one by one (for indexing)"""
    __slots__ = ('tag', 'ident', 'ty', 'doms', 'concl', 'syms', 'arity', '_rest', 'csort', 'flex')
    def __init__(self, k, tag, ident, ty):
        self.tag = tag; self.ident = ident; self.ty = ty
        doms, concl = open_pis(k, ty)
        self.doms = tuple(doms); self.concl = concl; self.arity = len(doms); self._rest = {0: ty, len(doms): concl}
        seen = []
        for v in variants(k, concl):
            s = sym_of(k, v)
            if s not in seen: seen.append(s)
        self.syms = tuple(seen)
        self.csort = None                              # when a pattern variable heads the conclusion (`P y`, `A`): the sort level its declared type gives the conclusion (a type of Type l)
        chead, cargs = spine(k, concl); cv = k.view(chead)
        self.flex = cv[1] - HOLE if (cargs and cv[0] == 'fvar' and HOLE <= cv[1] < HOLE + len(doms)) else None          # a conclusion `P n` headed by a variable of the telescope (a motive): the index of that variable
        if cv[0] == 'fvar' and cv[1] >= HOLE and cv[1] - HOLE < len(doms):
            d = doms[cv[1] - HOLE]
            for _ in cargs:
                dv = k.view(d)
                if dv[0] != 'pi': d = None; break
                d = dv[2]
            if d is not None and k.view(d)[0] == 'type': self.csort = k.view(d)[1]
    def rest(self, k, kk):
        """the type left after kk binders (pattern variables for those)"""
        r = self._rest.get(kk)
        if r is None: r = self._rest[kk] = open_pis(k, self.ty, count=kk)[1]
        return r
    def __repr__(self): return 'Head(%s %s /%d)' % (self.tag, self.ident, self.arity)


class HeadTable:
    """the globals a term can start with, indexed by the symbols their conclusions have (`Head.syms`), the wildcards under None; in the order of the global ids. Shared by the engines of a world; made
    outside probes."""
    def __init__(self, w):
        self.w = w; self.k = w.k; self.scanned = 0; self.heads = {}; self.index = collections.defaultdict(list); self.wild = []; self.pending = []
        self.extend()
    def extend(self):
        """take in the globals made since the last scan, and those that were only reserved (a function or a type being defined) at the last scan and are finished now"""
        k = self.k; w = self.w
        if w.probing: raise AuthorError('the table of heads cannot be extended inside a probe')
        todo = self.pending + list(range(self.scanned, k.num_globals())); self.pending = []
        for gid in todo:
            try:
                if k.reserved_kind(gid): self.pending.append(gid); continue
                info = k.global_info(gid)
                if info[0].startswith('__'): continue
                h = Head(k, 'c', gid, info[1])
            except (E.ITPError, RuntimeError): continue
            self.heads[gid] = h
            if None in h.syms: self.wild.append(h)
            else:
                for s in h.syms: self.index[s].append(h)
        self.scanned = k.num_globals()
    def candidates(self, syms):
        """the heads whose conclusion has one of these symbols (or is a wildcard), in the order of the ids"""
        out = {}
        for s in syms:
            for h in self.index.get(s, ()): out[h.ident] = h
        for h in self.wild: out[h.ident] = h
        return [out[g] for g in sorted(out)]


def head_table(w):
    """the table of the world (made on first use, extended when the library has grown)"""
    t = w.__dict__.get('_head_table')
    if t is None:
        if w.probing: raise AuthorError('the table of heads must be made outside probes (Engine.refresh makes it)')
        t = w.__dict__['_head_table'] = HeadTable(w)
    elif t.scanned < w.k.num_globals() or t.pending:
        t.extend()
    return t


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ frames
class Frame:
    """a partly built term: an application waiting for arguments, or a binder waiting for its body"""
    __slots__ = ('kind', 'head', 'kk', 'doms', 'subst', 'auto', 'j', 'dom', 'level', 'stage', 'hole', 'concl', 'flex', 'expect')
    def __init__(self, kind):
        self.kind = kind; self.head = self.kk = self.doms = self.subst = self.auto = self.dom = self.level = self.hole = self.concl = self.flex = self.expect = None; self.j = 0; self.stage = 0
    def copy(self):
        f = Frame(self.kind); f.head = self.head; f.kk = self.kk; f.doms = self.doms; f.subst = None if self.subst is None else dict(self.subst)
        f.auto = None if self.auto is None else set(self.auto); f.j = self.j; f.dom = self.dom; f.level = self.level; f.stage = self.stage; f.hole = self.hole; f.concl = self.concl; f.flex = self.flex; f.expect = self.expect
        return f


class Builder:
    """one term being built. `tokens()` lists the choices at the first hole, `push(token)` takes one."""
    def __init__(self, space, expected):
        self.space = space; self.k = space.k; self.expected = expected
        self.ctx = list(space.ctx)                 # the context: the goal's, then the binders opened by the term so far
        self.opened = []                           # the heads of the binders opened by the term so far (levels len(space.ctx), ...)
        self.stack = []; self.result = None
        self.code = []
        self.allow_goals = False                   # may a proposition be left as a goal, `('g',)`?
        self.goals = []                            # the propositions left as goals so far, in order (the term mentions them as the variables GOAL_BASE, GOAL_BASE + 1, ...)
    @property
    def heads(self):
        """the variables that can start a term: the goal's, then the opened binders"""
        return self.space.local_heads + self.opened
    def head_of(self, level):
        n = len(self.space.ctx)
        if level < n: return self.space.local_head(level)
        return self.opened[level - n] if level - n < len(self.opened) else None
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ state
    @property
    def done(self): return self.result is not None
    def term(self):
        if self.result is None: raise BuildError('the term is not finished')
        return self.result
    @property
    def hole_type(self):
        """the type wanted at the first hole (None when the term is finished)"""
        if self.result is not None: return None
        if not self.stack: return self.expected
        f = self.stack[-1]
        if f.kind == 'app': return self._arg_type(f)
        return f.hole
    def copy(self):
        b = Builder.__new__(Builder)
        b.space = self.space; b.k = self.k; b.expected = self.expected; b.ctx = list(self.ctx); b.opened = list(self.opened)
        b.stack = [f.copy() for f in self.stack]; b.result = self.result; b.code = list(self.code)
        b.allow_goals = self.allow_goals; b.goals = list(self.goals)
        return b
    def _arg_type(self, f):
        n = len(f.doms)
        if f.j < n:
            d = f.doms[f.j]
            return inst_beta(self.k, d, {HOLE + i: f.subst[HOLE + i] for i in range(f.j) if HOLE + i in f.subst})
        # an argument beyond the telescope of the head's type (the head is applied to MORE arguments than its type shows binders: its conclusion, `P n` for a motive P that is a lambda, is a function
        # type once the arguments are in): the type of the application so far, with its Pi exposed
        k = self.k
        ty = inst_beta(k, f.concl, {HOLE + i: f.subst[HOLE + i] for i in range(n) if HOLE + i in f.subst})
        for i in range(n, f.j):
            e = self.expose_pi(beta_head(k, ty))
            if e is None: raise BuildError('the head applied to %d arguments is not a function (it takes no argument %d)' % (i, i))
            ty = k.instantiate(e[1], [f.subst[HOLE + i]])
        e = self.expose_pi(beta_head(k, ty))
        if e is None: raise BuildError('the head applied to %d arguments is not a function (it takes no argument %d)' % (f.j, f.j))
        return e[0]

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ types
    def expose_pi(self, ty):
        """(domain, codomain) of a function type, looking through definitions at its head (`Not P` is `P -> False`); None for any other type"""
        k = self.k; cur = ty
        for _ in range(VARIANTS + 2):
            v = k.view(cur)
            if v[0] == 'pi': return v[1], v[2]
            cur = k.unfold_head(cur)
            if cur is None: return None
        return None

    def _in_all_domain(self):
        """is the first hole the domain of a quantifier? (it may be the sort Type 1 itself: a statement about all types)"""
        return bool(self.stack) and self.stack[-1].kind == 'all' and self.stack[-1].stage == 0

    def is_sort(self, ty, level):
        v = self.k.view(ty); return v[0] == 'type' and v[1] == level

    def number_instance(self, ty):
        return self.space.number_instance(ty)

    def declared_type(self, t):
        """the type of a term read off the declared types of its head and arguments (no evaluation, so definitions stay folded); None where that does not work"""
        k = self.k
        head, args = spine(k, t); hv = k.view(head)
        if hv[0] == 'const': ty = k.global_info(hv[1])[1]
        elif hv[0] == 'fvar' and hv[1] < len(self.ctx): ty = self.ctx[hv[1]]
        elif hv[0] == 'type' and not args: return k.mk_type(hv[1] + 1)
        else: return None
        for a in args:
            e = self.expose_pi(ty)
            if e is None: return None
            ty = k.instantiate(e[1], [a])
        return ty

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the menu
    @bounded
    def tokens(self, pool=False, rare=False):
        """the choices at the first hole, in a fixed order: binders and sorts, numerals, variables (the latest first), globals (by id), then (with `pool`) the pool terms of the hole's type"""
        H = self.hole_type
        if H is None: return []
        k = self.k; out = []
        if self.expose_pi(H) is not None: out.append(('lam',))
        if self.is_sort(H, 0):
            out += [('all',), ('imp',)]
            if rare: out.append(('dep',))
        elif self.is_sort(H, 1):
            out += [('arr',), ('s', 0)]
            if self._in_all_domain(): out.append(('s', 1))                          # (`forall A : Type, ...`)
        else:
            v = k.view(H)
            if v[0] == 'type' and v[1] >= 2: out.append(('s', v[1] - 1))
        if self._goal_ok(H): out.append(('g',))
        inst = self.number_instance(H)
        if inst is not None:
            for val in self.space.numerals:
                if val < 0 and not getattr(inst, 'is_ring', False): continue
                if abs(val) > self.space.max_numeral(inst): continue
                out.append(('n', val))
        for h, kk, _ in self._head_options(H, rare):
            out.append((h.tag, h.ident, kk))
        if pool and self.space.pool is not None:
            out += [('p', i) for i in self.space.pool_of_type(H)]
        return out

    def _goal_ok(self, H):
        """may the hole be left as a goal? The builder allows goals, the term has opened no binder (a goal under a binder would need the binder in its context), the hole's type is a proposition
        that no pattern variable is left in, and there are not too many goals already"""
        if not self.allow_goals or self.opened or len(self.goals) >= MAX_GOALS or holes_of(self.k, H): return False
        try: ok, sort = self.k.infer_open(H, self.ctx)
        except (E.ITPError, RuntimeError) as e: spent(e); return False
        return bool(ok) and sort == self.space.prop

    @bounded
    def is_prop(self, H):
        """is the type H a proposition (in the context of the term so far)?"""
        try: ok, sort = self.k.infer_open(H, self.ctx)
        except (E.ITPError, RuntimeError) as e: spent(e); return False
        return bool(ok) and sort == self.space.prop

    @bounded
    def fill(self, term):
        """put a term that is already built into the first hole (the kernel checks it against the hole's type)"""
        H = self.hole_type
        if H is None: raise BuildError('the term is finished: nothing to fill')
        try: ok, msg = self.k.check_open(term, H, self.ctx)
        except (E.ITPError, RuntimeError) as e: spent(e); raise BuildError('the kernel could not check the term: %s' % (e,)) from None
        if not ok: raise BuildError('the term does not have the type %s: %s' % (self._show(H), msg))
        self._deliver(term)

    def _head_options(self, H, rare=False):
        """(head, number of arguments, the pattern variables the hole determines) for every head that fits the hole. A principle whose conclusion is a motive applied to arguments (`P n`) fits every
        proposition once the motive is said; it is listed only with `rare` (INDUCT_WITH is the way to use one) -- but a code may always use it"""
        k = self.k; space = self.space
        hd, hconcl = open_pis(k, H)
        m = len(hd)
        syms = []
        for v in variants(k, hconcl):
            s = sym_of(k, v)
            if s not in syms: syms.append(s)
        out = []
        for h in reversed(self.heads):                                            # the variables first, the latest first
            if h.arity < m or not (None in h.syms or any(s in h.syms for s in syms)): continue
            kk = h.arity - m
            if h.flex is not None and kk == h.arity and not rare: continue
            bound = self._match_head(h, kk, H)
            if bound is not None: out.append((h, kk, bound))
        eng = space.eng
        for h in space.table.candidates(syms):
            if not eng.sees(h.ident) or h.arity < m: continue
            kk = h.arity - m
            if h.flex is not None and kk == h.arity and not rare: continue
            if self._is_numeral_spelling(h, kk, H): continue
            bound = self._match_head(h, kk, H)
            if bound is not None: out.append((h, kk, bound))
        return out

    def _is_numeral_spelling(self, h, kk, H):
        """is this a constant that spells a numeral of the hole's number type (so that a numeral token says the same)?"""
        if kk != 0 or h.tag != 'c': return False
        inst = self.number_instance(H)
        if inst is None: return False
        return self.space.canonical_numeral(inst, h.ident)

    def _match_head(self, h, kk, H):
        """the pattern variables of the head's first kk binders that the hole determines -- {level: node} -- or None when the head applied to kk arguments cannot have type H. The conclusion is matched
        against the hole's type (and the definitions at the head of either unfolded); what is then still free is looked for in the declared types of the arguments found (`f_equal`: the carrier `A`
        is the domain of the type of `f`); a value whose declared type is not the domain it fills is refused."""
        k = self.k
        if h.flex is not None and kk == h.arity:                                   # `P n`: the motive P is said first; what the hole then says about n is read after it (see _resolve_flex)
            if h.csort is not None and not self._sort_fits(H, h.csort): return None
            return {}
        R_k = h.rest(k, kk)
        hole_vs = variants(k, H)
        rest_vs = variants(k, R_k)                                                 # (also when arguments are missing: `foo a` of  foo : forall x, Not (P x)  has the type  P a -> False)
        binding = None
        for hv in hole_vs:
            for rv in rest_vs:
                trial = {}
                if match(k, rv, hv, trial, self._convertible): binding = trial; break
            if binding is not None: break
        if binding is None:
            return None
        if h.csort is not None and not self._sort_fits(H, h.csort): return None
        bound = {lv: n for lv, n in binding.items() if HOLE <= lv < HOLE + kk}
        # (the pattern variables of binders beyond kk would be those of the rest, which `open_pis` leaves as a Pi: none are bound)
        return self._close_binding(h, kk, bound)

    def _sort_fits(self, H, level):
        """can the hole type H (a type) be of sort `level`, i.e. is H : Type level? Unknown (a hole that is itself a function type, an ill-typed context) counts as yes"""
        k = self.k
        if k.view(H)[0] == 'pi': return True
        if holes_of(k, H): return True
        ok, sort = k.infer_open(H, self.ctx)
        if not ok: return True
        sv = k.view(sort)
        return sv[0] != 'type' or sv[1] == level

    def _close_binding(self, h, kk, bound):
        """extend `bound` by the declared types of the values (until nothing changes), and check the values against the domains they fill: None if one does not fit"""
        k = self.k; doms = h.doms
        changed = True; rounds = 0
        while changed and rounds < kk + 1:
            changed = False; rounds += 1
            for lv in sorted(bound):
                i = lv - HOLE
                if i >= kk: continue
                free = holes_of(k, doms[i]) - set(bound)
                if not free: continue
                dty = self.declared_type(bound[lv])
                if dty is None: continue
                trial = dict(bound)
                if match(k, doms[i], dty, trial):
                    new = {l: n for l, n in trial.items() if HOLE <= l < HOLE + kk}
                    if new != bound: bound = new; changed = True
        for lv in sorted(bound):                                                    # the check
            i = lv - HOLE
            if i >= kk: continue
            dd = inst_beta(k, doms[i], bound)
            if holes_of(k, dd): continue
            dty = self.declared_type(bound[lv])
            if dty is None or dty == dd: continue
            if not self._convertible(dd, dty): return None
        return bound

    def _convertible(self, a, b):
        """are two terms convertible, within a small work budget (a menu must not compute)? The answers are remembered by the space"""
        key = (a, b, len(self.ctx)); memo = self.space.conv_memo
        r = memo.get(key)
        if r is None:
            k = self.k; saved = k.get_fuel(); k.set_fuel(CONV_FUEL)
            try: r = k.conv_open(a, b, len(self.ctx))
            except (E.ITPError, RuntimeError): r = False
            finally: k.set_fuel(saved)
            memo[key] = r
        return r

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ pushing a token
    @bounded
    def push(self, tok):
        H = self.hole_type
        if H is None: raise BuildError('the term is finished: no token %r' % (tok,))
        if len(self.code) >= MAX_CODE: raise BuildError('a term of more than %d tokens' % MAX_CODE)
        if not isinstance(tok, tuple) or not tok or tok[0] not in TAGS: raise BuildError('not a token: %r' % (tok,))
        k = self.k; tag = tok[0]
        def ints(n):
            if len(tok) != n or not all(isinstance(x, int) and not isinstance(x, bool) for x in tok[1:]): raise BuildError('malformed token %r' % (tok,))
        try:
            if tag == 'lam':
                ints(1)
                pi = self.expose_pi(H)
                if pi is None: raise BuildError('a lambda where %s is wanted' % self._show(H))
                dom, cod = pi
                f = Frame('lam'); f.dom = dom; f.level = len(self.ctx)
                self._open(dom)
                f.hole = k.instantiate(cod, [k.mk_fvar(f.level)])
                self.stack.append(f)
            elif tag in ('all', 'imp', 'dep'):
                ints(1)
                if not self.is_sort(H, 0): raise BuildError('a quantified statement where %s is wanted' % self._show(H))
                f = Frame(tag); f.hole = self.space.data_sort if tag == 'all' else self.space.prop
                self.stack.append(f)
            elif tag == 'arr':
                ints(1)
                if not self.is_sort(H, 1): raise BuildError('a function type where %s is wanted' % self._show(H))
                f = Frame('arr'); f.hole = H
                self.stack.append(f)
            elif tag == 's':
                ints(2)
                if tok[1] < 0 or not (self.is_sort(H, tok[1] + 1) or (tok[1] == 1 and self._in_all_domain())): raise BuildError('the sort Type %d where %s is wanted' % (tok[1], self._show(H)))
                self.code.append(tok); self._deliver(k.mk_type(tok[1])); return
            elif tag == 'n':
                ints(2)
                inst = self.number_instance(H)
                if inst is None: raise BuildError('a numeral where %s is wanted' % self._show(H))
                if abs(tok[1]) > self.space.max_numeral(inst) or (tok[1] < 0 and not getattr(inst, 'is_ring', False)): raise BuildError('no numeral %d of this type' % tok[1])
                self.code.append(tok); self._deliver(inst.lit(tok[1]).n); return
            elif tag == 'p':
                ints(2)
                pool = self.space.pool
                if pool is None or not 0 <= tok[1] < len(pool): raise BuildError('no pool term in %r' % (tok,))
                if pool.types[tok[1]] != H and not self._convertible(pool.types[tok[1]], H): raise BuildError('the pool term %d does not have the type %s' % (tok[1], self._show(H)))
                self.code.append(tok); self._deliver(pool.terms[tok[1]]); return
            elif tag == 'g':
                ints(1)
                if not self._goal_ok(H): raise BuildError('%s cannot be left as a goal here' % self._show(H))
                fv = k.mk_fvar(GOAL_BASE + len(self.goals)); self.goals.append(H)
                self.code.append(tok); self._deliver(fv); return
            else:                                                # 'v' / 'c': a head with kk arguments
                ints(3)
                kk = tok[2]
                if kk < 0: raise BuildError('malformed token %r' % (tok,))
                if tag == 'v':
                    h = self.head_of(tok[1]) if tok[1] >= 0 else None
                    if h is None: raise BuildError('no variable at level %d' % tok[1])
                    node = k.mk_fvar(tok[1])
                else:
                    if not self.space.eng.sees(tok[1]): raise BuildError('no global %d' % tok[1])
                    h = self.space.table.heads.get(tok[1])
                    if h is None: raise BuildError('the global %d cannot be used in a term' % tok[1])
                    node = k.mk_const(tok[1])
                if kk > h.arity + MAX_ARITY: raise BuildError('%d arguments for a head that takes %d' % (kk, h.arity))
                if kk > h.arity: bound = {}                                     # (more arguments than the head's type has binders: the hole determines none of them, they are all said, and the kernel checks the finished term)
                else: bound = self._match_head(h, kk, H)
                if bound is None and kk == h.arity: bound = {}                  # (a head applied to ALL its arguments whose conclusion is not the hole's type by matching may still have it by computation -- `f_equal .. : Pred (Succ n) = Pred (Succ m)` for `n = m`: nothing is determined, every argument is said, and the kernel decides when the term is finished)
                if bound is None: raise BuildError('%s applied to %d arguments does not have the type %s' % (self._show(node), kk, self._show(H)))
                f = Frame('app'); f.head = node; f.kk = kk; f.doms = h.doms; f.concl = h.concl; f.subst = dict(bound); f.auto = set(i for i in range(min(kk, h.arity)) if HOLE + i in bound)
                if h.flex is not None and kk == h.arity: f.flex = h.flex; f.expect = H
                self.stack.append(f); self.code.append(tok)
                self._ready()
                return
        except (E.ITPError, RuntimeError) as e:
            spent(e); raise BuildError('%s' % (e,)) from None
        self.code.append(tok)

    def _show(self, node):
        try:
            w = self.space.w; names = list(self.space.names) + ['x%d' % i for i in range(len(self.space.names), len(self.ctx))]
            return w.show(w.T(node), names=names)
        except Exception: return '<term>'

    def _open(self, dom):
        """a binder enters the context"""
        level = len(self.ctx); self.ctx.append(dom)
        self.opened.append(Head(self.k, 'v', level, dom))

    def _close(self):
        self.ctx.pop(); self.opened.pop()

    def _ready(self):
        """the application on top of the stack: skip the arguments the hole determined, and finish it if nothing is left to say"""
        f = self.stack[-1]
        if f.kind != 'app': return
        while f.j < f.kk and f.j in f.auto: f.j += 1
        if f.j < f.kk: return
        self.stack.pop(); self._deliver(self._apply(f))

    def _resolve_flex(self, f):
        """the motive of an application whose conclusion is `P n` has been said: the arguments that its conclusion, with the motive in, determines through the hole's type are determined now (a motive
        that is a lambda makes `P n` a proposition about n, whose shape the hole shows)"""
        k = self.k; f.flex = None
        reps = {HOLE + i: f.subst[HOLE + i] for i in range(len(f.doms)) if HOLE + i in f.subst}
        concl = inst_beta(k, f.concl, reps)
        for hv in variants(k, f.expect):
            for cv in variants(k, concl):
                trial = {}
                if match(k, cv, hv, trial, self._convertible):
                    new = {lv: n for lv, n in trial.items() if HOLE <= lv < HOLE + f.kk and lv not in f.subst}
                    f.subst.update(new); f.auto.update(lv - HOLE for lv in new)
                    return

    def _apply(self, f):
        k = self.k; node = f.head
        for i in range(f.kk): node = k.mk_app(node, f.subst[HOLE + i])
        return node

    def _deliver(self, node):
        """a finished sub-term goes to the frame below it, or is the result; the frames that this completes finish in turn"""
        k = self.k
        while True:
            if not self.stack: self.result = node; return
            f = self.stack[-1]; kind = f.kind
            if kind == 'app':
                f.subst[HOLE + f.j] = node
                if f.flex is not None and f.j == f.flex: f.j += 1; self._resolve_flex(f)
                else: f.j += 1
                while f.j < f.kk and f.j in f.auto: f.j += 1
                if f.j < f.kk: return
                self.stack.pop(); node = self._apply(f)
            elif kind == 'lam':
                self._close(); self.stack.pop(); node = k.mk_lambda(f.dom, k.abstract_fvars(node, [f.level]))
            elif kind in ('all', 'dep'):
                if f.stage == 0:
                    f.dom = node; f.stage = 1; f.hole = self.space.prop; f.level = len(self.ctx); self._open(node); return
                self._close(); self.stack.pop(); node = k.mk_pi(f.dom, k.abstract_fvars(node, [f.level]))
            else:                                                     # 'imp' / 'arr'
                if f.stage == 0: f.dom = node; f.stage = 1; return
                self.stack.pop(); node = k.mk_pi(f.dom, node)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the space
class TermSpace:
    """the terms of one context. `hyp_types` / `hyp_names`: the goal's hypotheses (kernel handles; the names are for display only); `pool`: an optional gym_terms.Pool whose terms may be named by
    ('p', i); `roots`: terms whose numerals are worth offering (the goal)."""
    def __init__(self, eng, hyp_types, hyp_names=None, pool=None, roots=()):
        self.eng = eng; self.w = eng.w; self.k = eng.k
        self.ctx = list(hyp_types); self.names = list(hyp_names) if hyp_names is not None else ['x%d' % i for i in range(len(self.ctx))]
        self.pool = pool if pool is not None and len(pool) else None
        self.prop = self.k.mk_type(0); self.data_sort = self.k.mk_type(1)
        self.table = head_table(self.w)
        self._local = {}; self._locals_all = None
        self._instances = {}; self._canon = {}; self.conv_memo = {}
        self.roots = tuple(roots); self._numerals_cache = None

    @property
    def local_heads(self):
        """the heads of the context's variables, in order (made on first use)"""
        if self._locals_all is None: self._locals_all = [self.local_head(lv) for lv in range(len(self.ctx))]
        return self._locals_all
    def local_head(self, lv):
        h = self._local.get(lv)
        if h is None: h = self._local[lv] = Head(self.k, 'v', lv, self.ctx[lv])
        return h
    @property
    def numerals(self):
        if self._numerals_cache is None: self._numerals_cache = self._numerals(self.roots)
        return self._numerals_cache

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ numbers
    def number_instance(self, ty):
        """the ring instance of a number type (Nat, N, Z) or None"""
        if ty in self._instances: return self._instances[ty]
        try: inst = R.instance_for(self.w, self.w.T(ty))
        except (AuthorError, E.ITPError, RuntimeError): inst = None
        self._instances[ty] = inst
        return inst

    def max_numeral(self, inst):
        return MAX_NUMERAL['binary' if type(inst).__name__ != 'SemiringNat' else 'unary']

    def canonical_numeral(self, inst, gid):
        """is the constant a spelling of a numeral that `inst.lit` makes (Zero, N0, Z.zero)?"""
        key = (id(inst), gid)
        r = self._canon.get(key)
        if r is None:
            try: r = self.k.mk_const(gid) == inst.lit(0).n
            except (AuthorError, E.ITPError, RuntimeError): r = False
            self._canon[key] = r
        return r

    def _numerals(self, roots):
        """the numerals worth offering: the small ones and those of the goal"""
        import gym_terms as GT
        occurring = set()
        for name in GT.NUMBER_TYPES:
            if name in self.w.gid:
                try: inst = R.instance_for(self.w, self.w[name])
                except (AuthorError, E.ITPError, RuntimeError): inst = None
                if inst is not None:
                    for v in GT.numerals_in(self.k, inst, self.w, list(roots) + list(self.ctx)): occurring.add(v)
        occ = sorted(occurring, key=lambda v: (abs(v), v))[:MAX_GOAL_NUMERALS]
        out = []
        for v in tuple(DEFAULT_NUMERALS) + tuple(occ):
            if v not in out: out.append(v)
        return tuple(out)

    def pool_of_type(self, ty):
        return self.pool.of_type.get(ty, ()) if self.pool is not None else ()

    def builder(self, expected):
        return Builder(self, expected)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ whole terms
    @bounded
    def build(self, code, expected, check=True):
        """the term that a code says, of the expected type; raises BuildError"""
        b = Builder(self, expected)
        for tok in code: b.push(tuple(tok) if isinstance(tok, list) else tok)
        if not b.done: raise BuildError('the term is not finished (%d tokens said)' % len(code))
        t = b.term()
        if check:
            try:
                ok, msg = self.k.check_open(t, expected, self.ctx)
            except (E.ITPError, RuntimeError) as e: spent(e); raise BuildError('the kernel could not check the term: %s' % (e,)) from None
            if not ok: raise BuildError('the term does not have the expected type: %s' % (msg,))
        return t

    def build_refine(self, code, expected):
        """a term of the expected type that may leave propositions as goals (`('g',)` in the code): (the term, the goals' propositions, in order). With goals the term is the function  fun g1 .. gn => t  of
        type  g1 -> .. -> gn -> expected  (`apply` of it leaves the goals g1 .. gn); the kernel has checked it. Without goals it is t itself, of the expected type. Raises BuildError."""
        b = Builder(self, expected); b.allow_goals = True
        for tok in code: b.push(tuple(tok) if isinstance(tok, list) else tok)
        return self.close_goals(b)

    @bounded
    def close_goals(self, b):
        """(the term of a finished builder with its goals abstracted, the goals): see `build_refine`"""
        k = self.k
        if not b.done: raise BuildError('the term is not finished (%d tokens said)' % len(b.code))
        t = b.term(); goals = list(b.goals); expected = b.expected
        for g in goals:
            tags, scalar = k.observe_term(g)[:2]
            if bool(((tags == E.OBS_TAG_NAMES.index('FVAR')) & (scalar >= GOAL_BASE)).any()): raise BuildError('a goal that depends on another goal')
        lam = t; pi = expected
        for i in reversed(range(len(goals))):
            lam = k.mk_lambda(goals[i], k.abstract_fvars(lam, [GOAL_BASE + i])); pi = k.mk_pi(goals[i], pi)
        try: ok, msg = k.check_open(lam, pi, self.ctx)
        except (E.ITPError, RuntimeError) as e: spent(e); raise BuildError('the kernel could not check the term: %s' % (e,)) from None
        if not ok: raise BuildError('the term does not have the expected type: %s' % (msg,))
        return lam, goals

    @bounded
    def encode(self, term, expected=None):
        """the canonical code of a term (numerals as numerals, a head applied to exactly the arguments the expected type does not determine); raises NotEncodable"""
        k = self.k
        if expected is None:
            ok, expected = k.infer_open(term, self.ctx)
            if not ok: raise NotEncodable('ill-typed term: %s' % (expected,))
        b = Builder(self, expected)
        self._enc(b, term)
        if not b.done: raise NotEncodable('the term was not finished')
        if b.term() != term: raise NotEncodable('the code builds a different term: %s' % b._show(b.term()))
        return tuple(b.code)

    def _enc(self, b, t):
        k = self.k; H = b.hole_type
        if H is None: raise NotEncodable('more terms than holes')
        v = k.view(t)
        inst = b.number_instance(H)
        if inst is not None:
            try: val = inst.numeral(self.w.T(t))
            except (AuthorError, E.ITPError, RuntimeError): val = None
            if val is not None and abs(val) <= self.max_numeral(inst) and (val >= 0 or getattr(inst, 'is_ring', False)):
                try: canonical = inst.lit(val).n == t
                except (AuthorError, E.ITPError, RuntimeError): canonical = False
                if canonical: self._push(b, ('n', val)); return
        if v[0] == 'type':
            self._push(b, ('s', v[1])); return
        if v[0] == 'lam':
            self._push(b, ('lam',))
            self._enc(b, k.instantiate(v[2], [k.mk_fvar(len(b.ctx) - 1)])); return
        if v[0] == 'pi':
            if b.is_sort(H, 1):
                if k.loose_range(v[2]) != 0: raise NotEncodable('a dependent function type')
                self._push(b, ('arr',)); self._enc(b, v[1]); self._enc(b, v[2]); return
            if not b.is_sort(H, 0): raise NotEncodable('a quantifier where %s is wanted' % b._show(H))
            ok, sort = k.infer_open(v[1], b.ctx)
            if not ok: raise NotEncodable('ill-typed domain')
            if sort == self.prop:
                if k.loose_range(v[2]) != 0:
                    self._push(b, ('dep',)); self._enc(b, v[1])
                    self._enc(b, k.instantiate(v[2], [k.mk_fvar(len(b.ctx) - 1)])); return
                self._push(b, ('imp',)); self._enc(b, v[1]); self._enc(b, v[2]); return
            self._push(b, ('all',)); self._enc(b, v[1])
            self._enc(b, k.instantiate(v[2], [k.mk_fvar(len(b.ctx) - 1)])); return
        head, args = spine(k, t); hv = k.view(head)
        if hv[0] == 'fvar': tag, ident = 'v', hv[1]
        elif hv[0] == 'const': tag, ident = 'c', hv[1]
        else: raise NotEncodable('a term of kind %s cannot be said' % hv[0])
        depth = len(b.stack)
        self._push(b, (tag, ident, len(args)))
        if len(b.stack) != depth + 1: return                                      # every argument was determined: the application is finished
        auto = b.stack[-1].auto
        for i in range(len(args)):
            if i not in auto: self._enc(b, args[i])

    def _push(self, b, tok):
        try: b.push(tok)
        except BuildError as e: raise NotEncodable('%s' % (e,)) from None
