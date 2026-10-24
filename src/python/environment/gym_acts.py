"""gym_acts.py -- the second action language: actions whose arguments are SAID (terms, statements, references), carried out by the tactic layer on top of the typed term builder (gym_build.py).

The first nine kinds of gym_env.py are a finite menu: the learner picks from a list. These are the kinds that an olympiad proof also needs and a list cannot offer, because the argument is an arbitrary
well-typed term of the goal's context. Such an argument is a CODE: a tuple of builder tokens, prefix order (gym_build.py). Every such action is a plain `Action` with

    codes    a tuple of codes (the terms said, in the order the kind documents)
    hyps     a tuple of hypothesis positions (the hypotheses it acts on)
    arg      a flag or a hypothesis position, where the kind says so

and its SIGNATURE is the whole of it, so it is found again (`Menu.find`) in any menu, also one that does not list it. A SOURCE (a lemma, a hypothesis) is named by a code of one token: ('c', gid, 0)
for a global, ('v', level, 0) for a hypothesis of the goal.

    CLAIM          codes = (statement,)           `have h : statement`: two goals, the statement, then the goal with `h` added
    REFINE         codes = (term,)                the term has the goal's type; each `('g',)` in it is a proposition left as a goal (`exact` when there is none; witnesses, splits, `apply f x ?_`)
    APPLY_WITH     codes = (source, t1, ..)       `apply source t1 ..`: the t_i are the arguments the goal does not determine (the data ones, in order); the premises become goals.
                                                  arg 1 (the unifier finds them): the t_i are the leading arguments, premises too, and the unifier finds the others by matching the conclusion
                                                  with the goal, up to computation (`apply pow_one` for `n * 1 = n`, a motive for an induction principle); the goals it leaves must not share a
                                                  metavariable (an argument that nothing determined is refused)
    REWRITE_WITH   codes = (source, t1, ..)       `rw source t1 ..` (arg 1: right to left): the t_i are the leading arguments, the rest is found by matching, the first instance is rewritten
    REWRITE_AT     codes = (source, t1, ..)       the same inside the hypothesis hyps[0], which is replaced by the rewritten one
    FORWARD        codes = (source, s1, ..)       `have h := source s1 ..`: one slot per leading binder; a slot is the code of a term, the code of a hypothesis for a premise, or () for an argument that
                                                  the others determine (the carrier, `a` in `a | b` from `a | c` and `c | b`)
    INDUCT_WITH    codes = (principle, e1, ..), hyps = (x,)   induction on the variable x by a library principle (`Z.induction_from`, `N.ind`, ..): the motive is read off the goal (the hypotheses that
                                                  mention x are generalised), the e_i are the principle's arguments that the type of x does not determine (the starting point)
    INJECT         arg = h                        from the equation h : c a.. = c b.. of a constructor (or an injective function) the facts that its injectivity lemmas give: a = b ..
    LINCOMB        hyps = (h1, ..), codes = (c1, ..)   `linear_combination (c1, h1) ..`: the goal is the combination of the equations h_i with the coefficients c_i, by `ring`
    USE            codes = (witness,)             (instead of a position in the pool)

NOTHING HERE IS TRUSTED. A term is built by the builder and checked by the kernel; the tactic layer does the rest and the finished proof is checked again (`Engine.verify`). A wrong action is refused
(`ActionError`) and changes nothing; it cannot make a false theorem appear.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys
import itp_env_lnl as E
from author import AuthorError
import gym_env as G
import gym_build as GB
from gym_env import Kind, Action, ActionError

CODE_KINDS = G.CODE_KINDS
MAX_CODES = 24                   # codes in one action
MAX_HYPS = 24
MAX_FORWARD = 60                 # FORWARD actions listed per goal
MAX_REWRITE_AT = 60
MAX_INDUCT = 40
MAX_PREMISES = 3                 # FORWARD is listed for sources with at most this many premises (all of them filled from the hypotheses)
EXTRA_NUMERALS = (0, 1)          # the starting points tried for a principle whose starting point no hypothesis shows


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ small helpers
def space_of(eng, p, i=0):
    """(a fresh term space for goal i of a live proof state, the goal as data). Fresh: this runs inside probes, whose nodes are reclaimed, so nothing is kept."""
    g = eng.goal_of(p, i)
    return GB.TermSpace(eng, g.hyp_types, g.hyp_names, roots=[g.target]), g


def build(space, code, expected):
    try: return space.build(code, expected)
    except GB.BuildError as e: raise ActionError('%s' % (e,)) from None


def expose_pi(k, ty):
    """(domain, codomain) of a function type, looking through definitions at its head; None for anything else"""
    cur = ty
    for _ in range(GB.VARIANTS + 2):
        v = k.view(cur)
        if v[0] == 'pi': return v[1], v[2]
        cur = k.unfold_head(cur)
        if cur is None: return None
    return None


def conv_fn(eng, nctx):
    """a bounded convertibility test for matching: are two terms of a context of nctx variables convertible within a small work budget?"""
    k = eng.k
    def conv(a, b):
        saved = k.get_fuel(); k.set_fuel(GB.CONV_FUEL)
        try: return bool(k.conv_open(a, b, nctx))
        except (E.ITPError, RuntimeError): return False
        finally: k.set_fuel(saved)
    return conv


def match_any(k, pat, ty, binding, conv=None):
    """match the pattern (variables from GB.HOLE up) against the closed type, either of them with the definitions at its head unfolded, one step at a time; on success `binding` is extended and True
    returned, on failure it is left as it was"""
    for pv in GB.variants(k, pat):
        for tv in GB.variants(k, ty):
            trial = dict(binding)
            if GB.match(k, pv, tv, trial, conv):
                binding.clear(); binding.update(trial)
                return True
    return False


def ref_of(eng, space, code):
    """(head, term) of a source reference: a code of exactly one token, ('v', level, 0) or ('c', gid, 0)"""
    if not isinstance(code, tuple) or len(code) != 1 or not isinstance(code[0], tuple) or len(code[0]) != 3: raise ActionError('a source is a code of one token (v, level, 0) or (c, id, 0), not %r' % (code,))
    tag, ident, kk = code[0]
    if kk != 0 or isinstance(ident, bool) or not isinstance(ident, int): raise ActionError('a source is a code of one token (v, level, 0) or (c, id, 0), not %r' % (code,))
    k = eng.k
    if tag == 'v':
        if not 0 <= ident < len(space.ctx): raise ActionError('no hypothesis %r' % (ident,))
        return space.local_head(ident), k.mk_fvar(ident)
    if tag == 'c':
        if not eng.sees(ident): raise ActionError('no global %r' % (ident,))
        h = space.table.heads.get(ident)
        if h is None: raise ActionError('the global %r cannot be used in a term' % (ident,))
        return h, k.mk_const(ident)
    raise ActionError('a source is a code of one token (v, level, 0) or (c, id, 0), not %r' % (code,))


def fresh_name(p, base, tag=''):
    ps = p.ps; names = p._names[ps.goal_meta_id(0)]; n = ps.goal_num_hyps(0)
    nm = '%s%s%d' % (tag, base, n)
    while nm in names: nm += '_'
    return nm


def hyp_term(eng, lvl): return eng.w.T(eng.k.mk_fvar(lvl))


def apply_type(k, ty, values):
    """the type of  source values  for a source of type `ty`: the binders instantiated one by one (None if there are fewer binders than values)"""
    for v in values:
        pi = expose_pi(k, ty)
        if pi is None: return None
        ty = k.instantiate(pi[1], [v])
    return ty


def positional_args(eng, space, ty, codes):
    """the terms that `codes` say for the leading binders of the type `ty`, in order: each is built against its binder's type (the earlier ones substituted). Returns the terms."""
    k = eng.k; out = []
    for i, c in enumerate(codes):
        pi = expose_pi(k, ty)
        if pi is None: raise ActionError('too many arguments: the type has only %d binders' % i)
        a = build(space, c, pi[0])
        out.append(a); ty = k.instantiate(pi[1], [a])
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ shapes
def _is_code(c, limit=GB.MAX_CODE):
    return isinstance(c, tuple) and len(c) <= limit and all(isinstance(t, tuple) and 0 < len(t) <= 3 and isinstance(t[0], str) for t in c)


def check_shape(eng, action, n_hyps):
    """the codes and hypotheses of an action must be of the shape its kind takes; raises ActionError. (What the codes MEAN is up to the builder and the tactic layer.)"""
    kind = Kind(action.kind)
    codes, hyps, args = action.codes, action.hyps, action.args
    if not isinstance(codes, tuple) or not all(_is_code(c) for c in codes) or len(codes) > MAX_CODES: raise ActionError('%r: the codes must be a tuple of tuples of tokens' % (action,))
    if not isinstance(hyps, tuple) or len(hyps) > MAX_HYPS or not all(isinstance(h, (int, G.np.integer)) and not isinstance(h, bool) and 0 <= h < n_hyps for h in hyps):
        raise ActionError('%r: the hypotheses must be a tuple of positions of the goal (it has %d)' % (action, n_hyps))
    if kind in CODE_KINDS and args: raise ActionError('%r: this kind of action takes no pool terms' % (action,))
    if kind in (Kind.CASES, Kind.INDUCT):                                       # (the hypotheses are the ones to generalise: the induction hypothesis / the cases are stated for all of them)
        if codes: raise ActionError('%r: this kind of action takes no codes' % (action,))
        if len(set(hyps)) != len(hyps) or action.arg in hyps: raise ActionError('%r: the hypotheses to generalise are distinct and not the one split' % (action,))
        return
    if kind not in CODE_KINDS and kind != Kind.USE and (codes or hyps): raise ActionError('%r: this kind of action takes no codes or hypotheses' % (action,))
    if kind == Kind.USE:
        if hyps: raise ActionError('%r: USE takes no hypotheses' % (action,))
        if codes and (args or len(codes) != 1 or not codes[0]): raise ActionError('%r: USE takes a witness code or a pool position, not both, and exactly one witness' % (action,))
        return
    if kind not in CODE_KINDS: return
    if kind in (Kind.REWRITE_AT, Kind.REWRITE_WITH) and action.arg not in (0, 1): raise ActionError('%r: the direction is 0 (left to right) or 1' % (action,))
    if kind not in (Kind.REWRITE_AT, Kind.REWRITE_WITH, Kind.APPLY_WITH) and action.arg != 0: raise ActionError('%r: the argument of this kind of action is 0' % (action,))
    if kind == Kind.APPLY_WITH and action.arg not in (0, 1): raise ActionError('%r: the mode is 0 (the goal determines the arguments) or 1 (the unifier finds them)' % (action,))
    want = {Kind.CLAIM: (1, 1, 0), Kind.REFINE: (1, 1, 0), Kind.INDUCT_WITH: (1, MAX_CODES, 1), Kind.REWRITE_AT: (1, MAX_CODES, 1)}.get(kind)
    if kind == Kind.LINCOMB:
        if not codes or len(codes) != len(hyps): raise ActionError('%r: one coefficient for each equation' % (action,))
        if not all(codes): raise ActionError('%r: a coefficient is a non-empty code' % (action,))
        return
    lo, hi, nh = want if want else (1, MAX_CODES, 0)
    if not lo <= len(codes) <= hi: raise ActionError('%r: %d codes (this kind takes %d to %d)' % (action, len(codes), lo, hi))
    if len(hyps) != nh: raise ActionError('%r: %d hypotheses (this kind takes %d)' % (action, len(hyps), nh))
    if kind in (Kind.CLAIM, Kind.REFINE) and not codes[0]: raise ActionError('%r: the code is empty' % (action,))
    if kind in (Kind.INDUCT_WITH, Kind.FORWARD, Kind.REWRITE_AT, Kind.REWRITE_WITH, Kind.APPLY_WITH) and not codes[0]: raise ActionError('%r: the source is empty' % (action,))
    if kind in (Kind.INDUCT_WITH, Kind.REWRITE_AT, Kind.REWRITE_WITH, Kind.APPLY_WITH) and not all(codes[1:]): raise ActionError('%r: an argument is a non-empty code' % (action,))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the runners
def run(eng, p, action):
    """carry out an action of one of the code kinds (and INJECT, and USE with a witness code) on the first goal of a live proof state. Returns what `clean_after` needs, or None."""
    kind = Kind(action.kind)
    try:
        if kind == Kind.CLAIM: return run_claim(eng, p, action)
        if kind == Kind.REFINE: return run_refine(eng, p, action)
        if kind == Kind.APPLY_WITH: return run_apply_with(eng, p, action)
        if kind == Kind.REWRITE_WITH: return run_rewrite_with(eng, p, action)
        if kind == Kind.REWRITE_AT: return run_rewrite_at(eng, p, action)
        if kind == Kind.FORWARD: return run_forward(eng, p, action)
        if kind == Kind.INDUCT_WITH: return run_induct_with(eng, p, action)
        if kind == Kind.INJECT: return run_inject(eng, p, action)
        if kind == Kind.LINCOMB: return run_lincomb(eng, p, action)
        if kind == Kind.USE: return run_use(eng, p, action)
    except GB.BuildError as e: raise ActionError('%s' % (e,)) from None
    raise ActionError('%r: not an action of the second language' % (action,))


def run_claim(eng, p, action):
    space, g = space_of(eng, p); w = eng.w
    stmt = build(space, action.codes[0], space.prop)
    if stmt in g.hyp_types: raise ActionError('the claim is already a hypothesis')
    p.apply(w.const('cut'), w.T(stmt), w.T(g.target))


def run_refine(eng, p, action):
    space, g = space_of(eng, p); w = eng.w
    t, goals = space.build_refine(action.codes[0], g.target)
    p.apply(w.T(t))


def run_apply_with(eng, p, action):
    space, g = space_of(eng, p); w = eng.w; k = eng.k
    h, node = ref_of(eng, space, action.codes[0])
    if action.arg == 1:                                                               # the unifier finds the arguments that the leading ones do not give
        args = positional_args(eng, space, h.ty, action.codes[1:])
        p.apply(w.T(node), *[w.T(a) for a in args]); return
    hd, _ = GB.open_pis(k, g.target); kk = h.arity - len(hd)
    if kk < 0: raise ActionError('the goal has more binders than the source')
    b = space.builder(g.target); b.allow_goals = True
    b.push((h.tag, h.ident, kk))
    rest = iter(action.codes[1:])
    while not b.done:
        H = b.hole_type
        if b.is_prop(H): b.push(('g',))
        else:
            c = next(rest, None)
            if c is None: raise ActionError('too few arguments: the next one is %s' % b._show(H))
            b.fill(build(space, c, H))
    if next(rest, None) is not None: raise ActionError('too many arguments')
    t, goals = space.close_goals(b)
    p.apply(w.T(t))


def run_rewrite_with(eng, p, action):
    space, g = space_of(eng, p); w = eng.w
    h, node = ref_of(eng, space, action.codes[0])
    args = positional_args(eng, space, h.ty, action.codes[1:])
    p.rw(w.T(node), *[w.T(a) for a in args], rev=bool(action.arg))


def run_rewrite_at(eng, p, action):
    space, g = space_of(eng, p); w = eng.w
    h, node = ref_of(eng, space, action.codes[0])
    x = action.hyps[0]
    if node == eng.k.mk_fvar(x): raise ActionError('a hypothesis is not rewritten by itself')
    args = positional_args(eng, space, h.ty, action.codes[1:])
    p.rw_at(hyp_term(eng, x), w.T(node), *[w.T(a) for a in args], rev=bool(action.arg), new=fresh_name(p, 'r', eng.name_tag))


def run_use(eng, p, action):
    space, g = space_of(eng, p); w = eng.w
    parts = p._ex_parts()
    if parts is None: raise ActionError('USE: the goal is no existential')
    wit = build(space, action.codes[0], parts[0].n)
    p.exists(w.T(wit))


def run_lincomb(eng, p, action):
    space, g = space_of(eng, p); w = eng.w
    with eng.in_context(g.hyp_types, g.hyp_names):
        try: A, _, _ = w.eq_parts(w.T(g.target))
        except AuthorError: raise ActionError('LINCOMB: the goal is no equation') from None
    pairs = [(w.T(build(space, c, A.n)), hyp_term(eng, h)) for c, h in zip(action.codes, action.hyps)]
    p.linear_combination(*pairs)


def resolve_slots(eng, space, src_ty, slots, nctx=None):
    """the arguments for the leading binders of `src_ty` that the slots say. A slot is the code of a term, or of a hypothesis (one token, (v, level, 0)) -- whose type is matched against the binder's, which
    fixes the arguments that occur in it -- or () for an argument the others determine. Returns the terms, one for each slot."""
    k = eng.k; n = len(slots)
    doms, _ = GB.open_pis(k, src_ty, count=n)
    if len(doms) < n: raise ActionError('%d arguments for a source that takes %d' % (n, len(doms)))
    conv = conv_fn(eng, len(space.ctx) if nctx is None else nctx)
    binding = {}; hyp_slot = {}
    for i, c in enumerate(slots):
        if len(c) == 1 and len(c[0]) == 3 and c[0][0] == 'v' and c[0][2] == 0 and isinstance(c[0][1], int) and 0 <= c[0][1] < len(space.ctx): hyp_slot[i] = c[0][1]
    for i in sorted(hyp_slot):                                                                  # first the hypotheses: their types fix what the binders mention
        lvl = hyp_slot[i]
        if not match_any(k, doms[i], space.ctx[lvl], binding, conv): raise ActionError('hypothesis %d does not have the type the argument %d needs' % (lvl, i))
        binding[GB.HOLE + i] = k.mk_fvar(lvl)
    values = []
    for i, c in enumerate(slots):
        if i in hyp_slot: values.append(binding[GB.HOLE + i]); continue
        if not c:
            if GB.HOLE + i not in binding: raise ActionError('argument %d is determined by nothing that was said' % i)
            values.append(binding[GB.HOLE + i]); continue
        dom = GB.inst_beta(k, doms[i], binding)
        if GB.holes_of(k, dom): raise ActionError('the type of argument %d is not known yet (say the arguments it depends on first)' % i)
        a = build(space, c, dom)
        old = binding.get(GB.HOLE + i)
        if old is not None and old != a and not conv(old, a): raise ActionError('argument %d is not what the hypotheses force' % i)
        binding[GB.HOLE + i] = a; values.append(a)
    return values


def add_fact(eng, p, src_node, values, base='i'):
    """add the fact  src values  to the hypotheses of the first goal; a fact that is already a hypothesis is refused"""
    w = eng.w; ps = p.ps; n = ps.goal_num_hyps(0)
    with p.focus(): proof = w.app(w.T(src_node), *[w.T(v) for v in values])
    before = {ps.goal_hyp_type(0, h) for h in range(n)}
    p.have(fresh_name(p, base, eng.name_tag), proof)
    if ps.goal_hyp_type(0, ps.goal_num_hyps(0) - 1) in before: raise ActionError('the fact is already a hypothesis')


def run_forward(eng, p, action):
    space, g = space_of(eng, p)
    h, node = ref_of(eng, space, action.codes[0])
    values = resolve_slots(eng, space, h.ty, action.codes[1:])
    add_fact(eng, p, node, values)


# --- injectivity
def eq_sides(eng, ty):
    """(type, left, right) of an equation as kernel handles, or None"""
    k = eng.k; eq = eng.w.gid.get('Eq')
    head, args = GB.spine(k, ty); hv = k.view(head)
    if eq is None or hv[0] != 'const' or hv[1] != eq or len(args) != 3: return None
    return args


def head_const(eng, t):
    k = eng.k; head, args = GB.spine(k, t); hv = k.view(head)
    return (hv[1], len(args)) if hv[0] == 'const' else None


def injectivity_lemmas(eng, ty):
    """the library lemmas (global ids) of the injectivity index that can apply to an equation of this type: both sides applications of the same constant"""
    sides = eq_sides(eng, ty)
    if sides is None: return []
    a, b = head_const(eng, sides[1]), head_const(eng, sides[2])
    if a is None or a != b: return []
    return [gid for gid in eng.inj_index.get(a[0], ()) if eng.sees(gid)]


def inject_facts(eng, space, h):
    """[(lemma global id, arguments)] for the injectivity lemmas that apply to the equation hypothesis h"""
    k = eng.k; out = []
    for gid in injectivity_lemmas(eng, space.ctx[h]):
        hd = space.table.heads.get(gid)
        if hd is None: continue
        slots = [() for _ in range(hd.arity - 1)] + [(('v', h, 0),)]
        try: out.append((gid, resolve_slots(eng, space, hd.ty, slots)))
        except (ActionError, GB.BuildError): continue
    return out


def run_inject(eng, p, action):
    space, g = space_of(eng, p); k = eng.k
    facts = inject_facts(eng, space, action.arg)
    if not facts: raise ActionError('INJECT: no injectivity lemma applies to hypothesis %d' % action.arg)
    made = 0
    for gid, values in facts:
        try: add_fact(eng, p, k.mk_const(gid), values)
        except ActionError: continue
        made += 1
    if not made: raise ActionError('INJECT: the facts are all hypotheses already')


# --- induction by a principle
class Principle:
    """what an induction principle needs, read off its statement  forall .. (P : T -> Prop) .. , premises -> P n  (n a binder): `mi`, `ni` the binders of the motive and of the variable; `T` the type
    the motive is about (with the pattern variables of the binders it mentions); `auto` the binders that T determines (the carrier of a quotient), `free` the others that must be said (the starting
    point); `cases` the premises that mention the motive; `side` the premises that mention the variable and not the motive (`a <= n`); `last` the last binder that is given by position"""
    __slots__ = ('gid', 'head', 'mi', 'ni', 'T', 'auto', 'free', 'cases', 'side', 'last', 'major_explicit')


def analyze_principle(eng, gid):
    """the Principle of a library statement, or None when it is not of that shape"""
    h = eng.table.heads.get(gid)
    if h is None: return None
    k = eng.k; n = h.arity; PROP = k.mk_type(0); HOLE = GB.HOLE
    head, args = GB.spine(k, h.concl); hv = k.view(head)
    if hv[0] != 'fvar' or not HOLE <= hv[1] < HOLE + n or len(args) != 1: return None
    av = k.view(args[0])
    if av[0] != 'fvar' or not HOLE <= av[1] < HOLE + n: return None
    mi, ni = hv[1] - HOLE, av[1] - HOLE
    if mi >= ni: return None
    mv = k.view(h.doms[mi])
    if mv[0] != 'pi' or k.loose_range(mv[2]) != 0 or mv[2] != PROP: return None                    # P : T -> Prop
    T = mv[1]
    if h.doms[ni] != T: return None
    try: real, _ = G.open_binders(k, k.global_info(gid)[1], 0)
    except (E.ITPError, RuntimeError): return None
    if len(real) != n: return None
    prop = []
    for j in range(n):
        ok, sort = k.infer_open(real[j], real[:j])
        if not ok: return None
        prop.append(sort == PROP)
    if prop[mi] or prop[ni]: return None
    first_prop = min((j for j in range(n) if prop[j]), default=n)
    holes_T = GB.holes_of(k, T)
    extras = [j for j in range(n) if j not in (mi, ni) and not prop[j]]
    auto = [j for j in extras if HOLE + j in holes_T]; free = [j for j in extras if j not in auto]
    need = [mi] + extras
    if max(need) >= first_prop: return None                                                         # something that must be said comes after a premise: positional `apply` cannot reach it
    pr = Principle(); pr.gid = gid; pr.head = h; pr.mi = mi; pr.ni = ni; pr.T = T; pr.auto = auto; pr.free = free
    pr.major_explicit = ni < first_prop
    pr.last = max(need + ([ni] if pr.major_explicit else []))
    pr.cases = [j for j in range(n) if prop[j] and HOLE + mi in GB.holes_of(k, h.doms[j])]
    pr.side = [j for j in range(n) if prop[j] and j > ni and HOLE + ni in GB.holes_of(k, h.doms[j]) and HOLE + mi not in GB.holes_of(k, h.doms[j])]
    if not any(k.view(h.doms[j])[0] == 'pi' for j in pr.cases): return None                     # an induction has a case that is a step (`transport : P x -> P y` has none)
    return pr


def mentions(eng, node, lvl): return lvl in G.atoms_of(eng.k, node)[1]


def induct_plan(eng, space, g, pr, x, extras):
    """what an induction on the variable x of goal g by the principle needs: (the arguments of the principle given by position except the motive, as {binder: term}; the hypotheses to generalise;
    the hypotheses the principle's side premises consume). Raises ActionError."""
    k = eng.k; HOLE = GB.HOLE
    if len(extras) != len(pr.free): raise ActionError('the principle needs %d more arguments (%s), %d were said' % (len(pr.free), pr.free, len(extras)))
    conv = conv_fn(eng, len(g.hyp_types))
    binding = {}
    if not match_any(k, pr.T, g.hyp_types[x], binding, conv): raise ActionError('the principle is not about the type of hypothesis %d' % x)
    for j in pr.auto:
        if HOLE + j not in binding: raise ActionError('the type of hypothesis %d does not determine the principle\'s argument %d' % (x, j))
    for j, c in zip(pr.free, extras):
        dom = GB.inst_beta(k, pr.head.doms[j], binding)
        if GB.holes_of(k, dom): raise ActionError('the type of the principle\'s argument %d is not known yet' % j)
        binding[HOLE + j] = build(space, c, dom)
    binding[HOLE + pr.ni] = k.mk_fvar(x)
    side = [GB.inst_beta(k, pr.head.doms[j], binding) for j in pr.side]
    consumed = set(); general = []
    for d in range(g.n_hyps):
        if d == x or not mentions(eng, g.hyp_types[d], x): continue
        if any(g.hyp_types[d] == s or conv(g.hyp_types[d], s) for s in side): consumed.add(d); continue
        ok, sort = k.infer_open(g.hyp_types[d], g.hyp_types[:d])
        if not ok or sort != space.prop: raise ActionError('hypothesis %d is a variable that depends on the one induction is on' % d)
        general.append(d)
    return binding, general, consumed


def revert(eng, p, levels):
    """move the hypotheses at these levels (propositions that nothing else mentions) of the first goal into the goal, in their order: the goal becomes  H1 -> .. -> Hn -> G  without them"""
    w = eng.w
    for d in sorted(levels, reverse=True):
        ps = p.ps
        D = ps.goal_hyp_type(0, d); G0 = ps.goal_target(0)
        p.apply(w.const('cut'), w.T(D), w.T(G0), hyp_term(eng, d))
        p.clear([d])


def run_induct_with(eng, p, action):
    space, g = space_of(eng, p); w = eng.w; k = eng.k; HOLE = GB.HOLE
    h, node = ref_of(eng, space, action.codes[0])
    if h.tag != 'c': raise ActionError('a principle is a library statement')
    pr = eng.principles.get(h.ident)
    if pr is None: raise ActionError('the global %r is not an induction principle' % (h.ident,))
    x = action.hyps[0]
    binding, general, consumed = induct_plan(eng, space, g, pr, x, action.codes[1:])
    surviving = [l for l in range(g.n_hyps) if l not in general]
    xl = surviving.index(x)
    new_of = {old: new for new, old in enumerate(surviving)}
    for lv in list(binding):                       # (the arguments say the variables of the goal as it is now; reverting the hypotheses that are generalised moves the others up: renumber them)
        used = G.atoms_of(k, binding[lv])[1]
        if any(u not in new_of for u in used): raise ActionError('an argument of the principle mentions a hypothesis that the induction generalises')
        repl = [(u, k.mk_fvar(new_of[u])) for u in used if new_of[u] != u]
        if repl: binding[lv] = k.replace_fvars(binding[lv], repl)
    if general: revert(eng, p, general)
    ps = p.ps; target = ps.goal_target(0)
    if xl not in G.atoms_of(k, target)[1]: raise ActionError('the goal does not mention hypothesis %d' % x)
    T = ps.goal_hyp_type(0, xl)
    M = k.mk_lambda(T, k.abstract_fvars(target, [xl]))
    args = []
    for j in range(pr.last + 1):
        if j == pr.mi: args.append(M)
        elif j == pr.ni: args.append(k.mk_fvar(xl))
        else: args.append(binding[HOLE + j])
    p.apply(w.T(node), *[w.T(a) for a in args])
    return {'x': xl, 'consumed': [surviving.index(c) for c in sorted(consumed)]}


def clean_induct(eng, p, info, made, n_before):
    """after the settled goals of an induction: remove, from each goal it made, the variable the induction was on and the hypotheses its side premises consumed, unless the goal still mentions them"""
    for i in range(made):
        eng._clear_dead(p, i, {info['x']} | set(info['consumed']))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ indexes
def scan(eng, lo, hi):
    """index the globals lo .. hi - 1: the induction principles and the injectivity lemmas"""
    k = eng.k; eq = eng.w.gid.get('Eq')
    for gid in range(lo, hi):
        h = eng.table.heads.get(gid)
        if h is None: continue
        try:
            pr = analyze_principle(eng, gid)
            if pr is not None: eng.principles[gid] = pr
            if eq is not None and h.arity >= 1:
                ps = eq_sides(eng, h.doms[-1]); cs = eq_sides(eng, h.concl)
                if ps is not None and cs is not None and ps[1] != ps[2]:
                    a, b = head_const(eng, ps[1]), head_const(eng, ps[2])
                    if a is not None and a == b:
                        args1, args2 = GB.spine(k, ps[1])[1], GB.spine(k, ps[2])[1]
                        if any(cs[1] == x and cs[2] == y for x, y in zip(args1, args2)): eng.inj_index.setdefault(a[0], []).append(gid)       # injective in one argument: c .. x .. = c .. y .. -> x = y
        except (E.ITPError, RuntimeError): continue


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the menu
def extend_menu(eng, g, actions):
    """add the actions of the second language that can be listed to the menu of goal g (CLAIM only from the engine's hints; the others are said, not listed)"""
    kinds = eng.kinds
    if not (kinds & {Kind.INDUCT_WITH, Kind.INJECT, Kind.FORWARD, Kind.REWRITE_AT, Kind.CLAIM}): return
    space = eng.space(g)
    if Kind.INJECT in kinds: actions.extend(_inject_actions(eng, space, g))
    if Kind.INDUCT_WITH in kinds: actions.extend(_induct_actions(eng, space, g))
    if Kind.FORWARD in kinds: actions.extend(_forward_actions(eng, space, g))
    if Kind.REWRITE_AT in kinds: actions.extend(_rewrite_at_actions(eng, space, g))
    if Kind.CLAIM in kinds: actions.extend(_claim_actions(eng, space, g))


def _inject_actions(eng, space, g):
    out = []; k = eng.k
    existing = set(g.hyp_types) | {k.simp_normalize(t, g.n_hyps) for t in g.hyp_types}
    for h in range(g.n_hyps):
        if not injectivity_lemmas(eng, g.hyp_types[h]): continue
        fresh = False
        for gid, values in inject_facts(eng, space, h):
            ty = apply_type(k, space.table.heads[gid].ty, values)
            if ty is not None and k.simp_normalize(ty, g.n_hyps) not in existing: fresh = True
        if fresh: out.append(Action(Kind.INJECT, h))
    return out


def _induct_actions(eng, space, g):
    out = []; k = eng.k; HOLE = GB.HOLE
    if not eng.principles: return out
    conv = conv_fn(eng, g.n_hyps)
    for x in range(g.n_hyps):
        ok, sort = k.infer_open(g.hyp_types[x], g.hyp_types[:x])
        if not ok or sort == space.prop: continue                                                      # a variable of a data type
        if not any(mentions(eng, t, x) for t in [g.target] + list(g.hyp_types[x + 1:])): continue
        for gid in sorted(eng.principles):
            if not eng.sees(gid): continue
            pr = eng.principles[gid]
            binding = {}
            if not match_any(k, pr.T, g.hyp_types[x], binding, conv) or any(HOLE + j not in binding for j in pr.auto): continue
            for extras in _extra_choices(eng, space, g, pr, x, binding, conv):
                if len(out) >= MAX_INDUCT: return out
                a = Action(Kind.INDUCT_WITH, 0, (), ((('c', gid, 0),),) + extras, (x,))
                try: _, general, _ = induct_plan(eng, space, g, pr, x, extras)
                except (ActionError, GB.BuildError): continue
                if not (mentions(eng, g.target, x) or any(mentions(eng, g.hyp_types[d], x) for d in general)): continue          # (the motive would not mention the variable: only the side premises do)
                out.append(a)
    return out


def _extra_choices(eng, space, g, pr, x, binding, conv):
    """the ways to say the arguments of a principle that its variable's type does not determine: the starting points that the hypotheses show (a hypothesis  1 <= x  is the side premise  a <= x  with a = 1),
    else the numerals 0 and 1. Each way is a tuple of codes."""
    k = eng.k; HOLE = GB.HOLE
    if not pr.free: return [()]
    if len(pr.free) > 1: return []
    j = pr.free[0]; found = []
    for sj in pr.side:
        pat = pr.head.doms[sj]
        trial0 = dict(binding); trial0[HOLE + pr.ni] = k.mk_fvar(x)
        for d in range(g.n_hyps):
            trial = dict(trial0)
            if d != x and match_any(k, pat, g.hyp_types[d], trial, conv) and HOLE + j in trial:
                try: code = space.encode(trial[HOLE + j], GB.inst_beta(k, pr.head.doms[j], binding))
                except GB.NotEncodable: continue
                if (code,) not in found: found.append((code,))
    if found: return found
    out = []
    for v in EXTRA_NUMERALS:
        code = (('n', v),)
        try: space.build(code, GB.inst_beta(k, pr.head.doms[j], binding))
        except GB.BuildError: continue
        out.append((code,))
    return out


def _forward_actions(eng, space, g):
    """modus ponens from the hypotheses: a hypothesis (or an allowed lemma) with at most MAX_PREMISES premises, each filled by a hypothesis whose type matches, the arguments they mention determined by the match;
    only the facts that are not hypotheses yet"""
    out = []; k = eng.k; HOLE = GB.HOLE; w = eng.w
    conv = conv_fn(eng, g.n_hyps)
    existing = set(g.hyp_types) | {k.simp_normalize(t, g.n_hyps) for t in g.hyp_types}
    sources = []
    for h in range(g.n_hyps): sources.append((('v', h, 0), space.local_head(h)))
    for name in eng.inst_lemmas:
        gid = w.gid.get(name)
        if gid is not None and eng.sees(gid) and gid in space.table.heads: sources.append((('c', gid, 0), space.table.heads[gid]))
    for tok, head in sources:
        if len(out) >= MAX_FORWARD: break
        n = head.arity
        if n == 0: continue
        doms = head.doms
        ctxt = [] if tok[0] == 'c' else list(g.hyp_types)
        try: real, _ = G.open_binders(k, head.ty, len(ctxt))
        except (E.ITPError, RuntimeError): continue
        prem = []
        for j in range(n):
            ok, sort = k.infer_open(real[j], ctxt + real[:j])
            prem.append(bool(ok) and sort == space.prop)
        if not any(prem) or sum(prem) > MAX_PREMISES: continue
        pj = [j for j in range(n) if prem[j]]
        def go(idx, binding, chosen):
            if len(out) >= MAX_FORWARD: return
            if idx == len(pj):
                if any(HOLE + j not in binding for j in range(n) if not prem[j] and j <= pj[-1]): return
                slots = []
                for j in range(pj[-1] + 1):
                    slots.append((('v', chosen[pj.index(j)], 0),) if prem[j] else ())
                try:
                    vals = resolve_slots(eng, space, head.ty, slots)
                    ty = apply_type(k, head.ty, vals)
                except (ActionError, GB.BuildError, E.ITPError, RuntimeError): return
                if ty is None: return
                if ty in existing or k.simp_normalize(ty, g.n_hyps) in existing: return
                out.append(Action(Kind.FORWARD, 0, (), (((tok),),) + tuple(slots)))
                return
            j = pj[idx]
            for d in range(g.n_hyps):
                if tok[0] == 'v' and d == tok[1]: continue
                trial = dict(binding)
                if match_any(k, doms[j], g.hyp_types[d], trial, conv):
                    trial[HOLE + j] = k.mk_fvar(d)
                    go(idx + 1, trial, chosen + [d])
        go(0, {}, [])
    return out


def _rewrite_at_actions(eng, space, g):
    out = []; k = eng.k; shim = G._MatchShim(k)
    for x in range(g.n_hyps):
        ok, sort = k.infer_open(g.hyp_types[x], g.hyp_types[:x])
        if not ok or sort != space.prop: continue
        heads = G.subterms_by_head(k, g.hyp_types[x])
        if not heads: continue
        def occurs(head, pat): return any(shim.match(pat, t, {}) for t in heads.get(head, ()))
        t_consts = G.atoms_of(k, g.hyp_types[x])[0]
        for head in sorted(t_consts):
            for gid, rev, pat in eng._rw_index.get(head, ()):
                if not eng.sees(gid) or not occurs(head, pat): continue
                if len(out) >= MAX_REWRITE_AT: return out
                out.append(Action(Kind.REWRITE_AT, 1 if rev else 0, (), ((('c', gid, 0),),), (x,)))
        for h in range(g.n_hyps):
            if h == x: continue
            pr = eng.hyp_profile(g, h)
            for rev, head, pat in pr.rw:
                if occurs(head, pat):
                    if len(out) >= MAX_REWRITE_AT: return out
                    out.append(Action(Kind.REWRITE_AT, 1 if rev else 0, (), ((('v', h, 0),),), (x,)))
    return out


def _claim_actions(eng, space, g):
    out = []
    for code in eng.claim_codes(space):
        try: stmt = space.build(code, space.prop)
        except GB.BuildError: continue
        if stmt in g.hyp_types or stmt == g.target: continue
        out.append(Action(Kind.CLAIM, 0, (), (code,)))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ portable form
PORTABLE_SOURCE_KINDS = (Kind.APPLY, Kind.REWRITE)


def source_token(eng, node):
    """the one-token code of a term that is a global or a hypothesis of the goal"""
    v = eng.k.view(node)
    if v[0] == 'const': return ('c', v[1], 0)
    if v[0] == 'fvar': return ('v', v[1], 0)
    raise ActionError('a candidate that is neither a global nor a hypothesis has no portable form')


def to_portable(eng, menu, action):
    """the action as plain data -- strings, ints and lists, no kernel handle and no index into a menu -- so that it can be written to a file, sent to another process, and read back by
    `from_portable` in any engine of the same world (the same global ids). The actions of the second language are their own portable form already (codes and hypothesis positions); an APPLY or a
    REWRITE is named by the lemma or hypothesis it uses instead of its place in the menu's list. An action that is relative to the menu's term pool (INSTANTIATE, USE by position) has no portable
    form: say the terms by codes (FORWARD, USE with a witness code)."""
    kind = Kind(action.kind); d = {'kind': kind.name}
    if kind in PORTABLE_SOURCE_KINDS:
        term, rev, _ = menu.cands[action.arg]
        d['source'] = list(source_token(eng, term))
        if kind == Kind.REWRITE: d['rev'] = int(bool(rev))
        return d
    if kind == Kind.INTERVAL_CASES:
        d['range'] = [int(x) for x in menu.cands[action.arg][2]]; return d
    if kind == Kind.INSTANTIATE or (kind == Kind.USE and action.args): raise ActionError('%r is relative to the menu\'s term pool: it has no portable form' % (action,))
    if action.arg: d['arg'] = int(action.arg)
    if action.codes: d['codes'] = [[list(t) for t in c] for c in action.codes]
    if action.hyps: d['hyps'] = [int(h) for h in action.hyps]
    return d


def _plain_int(x, what):
    """x as an int; booleans, floats, strings and the rest are not numbers here (data from a file or another process is not trusted to be well formed)"""
    if isinstance(x, bool) or not isinstance(x, int): raise ActionError('%s: %r is no integer' % (what, x))
    return x


def _plain_list(x, what):
    if not isinstance(x, (list, tuple)): raise ActionError('%s: %r is no list' % (what, x))
    return x


def _portable_code(c):
    """the code (a tuple of tokens) of portable data: a list of tokens, each a list whose first element is its name and the others integers; anything else is ActionError (what the tokens MEAN is the builder's)"""
    out = []
    for t in _plain_list(c, 'a code'):
        t = _plain_list(t, 'a token')
        if not t or not isinstance(t[0], str): raise ActionError('a token starts with its name: %r' % (t,))
        out.append((t[0],) + tuple(_plain_int(x, 'a token') for x in t[1:]))
    return tuple(out)


def from_portable(eng, menu, d):
    """the action that portable data `d` (see `to_portable`) says, for the goal of `menu`: the menu's own action when the menu lists it, else the equivalent action that says its arguments (APPLY
    -> APPLY_WITH with no arguments left to say, in the mode where the unifier finds them; REWRITE -> REWRITE_WITH). Raises ActionError for data that is no action -- whatever is wrong with it
    (a missing key, a number where a list is wanted, a global that does not exist): the data may come from a file or another process."""
    if not isinstance(d, dict): raise ActionError('%r: an action is a dict' % (d,))
    try: kind = Kind[d['kind']]
    except (KeyError, TypeError): raise ActionError('%r: no such kind of action' % (d,)) from None
    k = eng.k
    if kind in PORTABLE_SOURCE_KINDS:
        if 'source' not in d: raise ActionError('%r: no source' % (d,))
        tok = _portable_code([d['source']])
        if len(tok) != 1: raise ActionError('%r: a source is one token' % (d,))
        tok = tok[0]
        rev = d.get('rev', 0)
        if isinstance(rev, bool): rev = int(rev)
        if _plain_int(rev, 'rev') not in (0, 1): raise ActionError('%r: rev is 0 or 1' % (d,))
        if len(tok) != 3 or tok[2] != 0 or tok[0] not in ('c', 'v') or tok[1] < 0: raise ActionError('%r: a source is one token (c, id, 0) or (v, level, 0)' % (d,))
        if tok[0] == 'c' and tok[1] >= k.num_globals(): raise ActionError('%r: no global %d' % (d, tok[1]))
        node = k.mk_const(tok[1]) if tok[0] == 'c' else k.mk_fvar(tok[1])
        for a in menu.actions:
            if a.kind == kind and menu.cands[a.arg][0] == node and bool(menu.cands[a.arg][1]) == bool(rev): return a
        if kind == Kind.APPLY: return Action(Kind.APPLY_WITH, 1, (), ((tok,),))
        return Action(Kind.REWRITE_WITH, rev, (), ((tok,),))
    if kind == Kind.INTERVAL_CASES:
        if 'range' not in d: raise ActionError('%r: no range' % (d,))
        want = tuple(_plain_int(x, 'range') for x in _plain_list(d['range'], 'range'))
        for a in menu.actions:
            if a.kind == kind and tuple(menu.cands[a.arg][2]) == want: return a
        raise ActionError('%r: the menu does not list this split' % (d,))
    codes = tuple(_portable_code(c) for c in _plain_list(d.get('codes', ()), 'codes'))
    hyps = tuple(_plain_int(h, 'hyps') for h in _plain_list(d.get('hyps', ()), 'hyps'))
    return Action(kind, _plain_int(d.get('arg', 0), 'arg'), (), codes, hyps)


NO_MENU = G.Menu([], [], G.np.zeros(0, dtype=G.np.uint32), G.np.zeros(0, dtype=G.np.uint32), G.np.zeros(0, dtype=G.np.uint8))       # for the actions that say everything themselves
MENU_KINDS = frozenset(k.name for k in (Kind.APPLY, Kind.REWRITE, Kind.INTERVAL_CASES))


def replay_portable(eng, stmt, steps, premises=None):
    """prove `stmt` by portable actions (`to_portable`), as an agent would, with nothing but the engine: each action is read for the first open goal, checked, carried out and the goals settled.
    Returns the proof term, which the kernel has checked against the statement (ActionError if an action is refused, the actions run out before the proof is finished, or some are left over).
    `premises` (the number of globals the actions may use: a theorem is proved with the library that precedes it) is set on the engine and stays."""
    steps = _plain_list(steps, 'the steps')
    if premises is not None: eng.premises = premises
    p = eng.start(stmt)
    for i, d in enumerate(steps):
        if not p.n_goals: raise ActionError('the proof is finished, %d actions are left over' % (len(steps) - i))
        name = d.get('kind') if isinstance(d, dict) else None
        menu = eng.menu(eng.goal_of(p, 0)) if isinstance(name, str) and name in MENU_KINDS else NO_MENU
        act = from_portable(eng, menu, d)
        eng.replay(p, act, menu)
    if p.n_goals: raise ActionError('the actions are over and %d goals are open' % p.n_goals)
    return eng.verify(stmt, p)
