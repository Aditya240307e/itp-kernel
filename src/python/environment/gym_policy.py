"""gym_policy.py -- the policy-facing view of the action language: an action is ASSEMBLED by a sequence of discrete CHOICES.

A learner with no text interface does not write an `Action`; it makes decisions, one at a time, each among a finite list of OPTIONS that the engine says are legal at that point:

    kind                which kind of action (the legal ones for this goal)
    cand / hyp / source which lemma, which hypothesis (as a menu candidate, a position, or the one-token code of a source)
    token               the next token of a term it is SAYING (statement, witness, argument), among the tokens that keep the term well typed (gym_build.Builder.tokens)
    more / slot / gen   whether another argument, which kind of slot, whether to generalise another hypothesis

Every decision is a `Choice`: a `name`, a list of `options` -- rows (otype, a, b, flags), plain integers a network can embed -- and the context the decision is made in (`hole`: the type of the term being
said). Nothing is trusted: the assembled action is probed like any other (Engine.probe) and the finished proof is checked by the kernel; the choices only make well-typed and well-shaped actions likely.

    asm = Assembler(eng, goal, menu)          # menu = eng.menu(goal)
    ch = asm.start()                          # the first Choice
    while ch is not None: ch = asm.choose(i)  # i: the index of the chosen option
    asm.action                                # the Action (or asm.failed: why the choices led nowhere)

With `script=action` the assembler also computes, for each choice, which option the given action says (`Choice.want`): that is how a demonstration is decomposed into decisions (`decompose`), and the
tests check that choosing the wanted options rebuilds exactly the same action.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys
import itp_env_lnl as E
import gym_env as G
import gym_acts as ACTS
import gym_build as GB
from gym_env import Kind, Action, ActionError

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ options
OT_KIND, OT_HYP, OT_GLOBAL, OT_VAR, OT_NUM, OT_STRUCT, OT_STOP, OT_SLOT, OT_DIR, OT_RANGE, OT_YES = range(11)
OT_NAMES = ('kind', 'hyp', 'global', 'var', 'num', 'struct', 'stop', 'slot', 'dir', 'range', 'yes')
N_OT = len(OT_NAMES)

S_LAM, S_ALL, S_IMP, S_ARR, S_DEP, S_GOAL, S_SORT0, S_SORT1, S_SORT2, S_SORT3 = range(10)
STRUCT_OF = {'lam': S_LAM, 'all': S_ALL, 'imp': S_IMP, 'arr': S_ARR, 'dep': S_DEP, 'g': S_GOAL}
N_STRUCT = 10

SLOT_EMPTY, SLOT_TERM = 0, 1          # OT_SLOT options (a hypothesis in a slot is an OT_HYP option)

CHOICE_NAMES = ('kind', 'cand', 'hyp', 'source', 'token', 'more', 'slot', 'gen', 'dir', 'lin', 'range')
CHOICE_ID = {n: i for i, n in enumerate(CHOICE_NAMES)}

# the kinds of the policy's menu (in this order, as a decision's options); APPLY_WITH has two modes (see gym_acts.run_apply_with)
MENU_KINDS = (Kind.CASES, Kind.INDUCT, Kind.APPLY, Kind.REWRITE, Kind.RING, Kind.LINARITH, Kind.NLINARITH, Kind.BY_CONTRA, Kind.INTERVAL_CASES, Kind.INJECT)
NULLARY = (Kind.RING, Kind.LINARITH, Kind.NLINARITH, Kind.BY_CONTRA)
POLICY_KINDS = frozenset(MENU_KINDS) | frozenset({Kind.USE, Kind.CLAIM, Kind.INDUCT_WITH, Kind.FORWARD, Kind.REWRITE_AT, Kind.REFINE, Kind.APPLY_WITH, Kind.REWRITE_WITH, Kind.LINCOMB})
ENGINE_KINDS = frozenset(MENU_KINDS)          # what an engine for a policy lists in its menus (the code kinds are said, not listed)


class Infeasible(Exception):
    """the choices made so far lead to an action that cannot be (a source that does not fit, a type that is not known yet): the assembly stops"""


class Choice:
    """one decision. `options`: rows (otype, a, b, flags); `hole`: the kernel handle of the type of the term being said (token choices, the argument about to be said), else None; `slot`: the index of
    the code being said; `pos`: the position in it; `want`: the index of the option the script says (None when there is no script, or it says none of the options); `builder`: for tokens, the builder"""
    __slots__ = ('name', 'options', 'hole', 'slot', 'pos', 'want', 'builder', 'refs', 'kind', 'nctx')
    def __init__(self, name, options, hole=None, slot=0, pos=0, want=None, builder=None, refs=None, kind=None, nctx=0):
        self.name = name; self.options = options; self.hole = hole; self.slot = slot; self.pos = pos; self.want = want; self.builder = builder; self.refs = refs; self.kind = kind; self.nctx = nctx
    def __repr__(self): return 'Choice(%s, %d options%s)' % (self.name, len(self.options), '' if self.want is None else ', want %d' % self.want)


def token_option(tok):
    """the option row of a builder token"""
    t = tok[0]
    if t == 'c': return (OT_GLOBAL, tok[1], tok[2], 0)
    if t == 'v': return (OT_VAR, tok[1], tok[2], 0)
    if t == 'n': return (OT_NUM, tok[1], 0, 0)
    if t == 's': return (OT_STRUCT, S_SORT0 + min(tok[1], 3), 0, 0)
    if t in STRUCT_OF: return (OT_STRUCT, STRUCT_OF[t], 0, 0)
    raise ValueError('no option for the token %r' % (tok,))


def kind_option(kind, mode=0): return (OT_KIND, int(kind), int(mode), 0)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the assembler
class Assembler:
    """assembles one action for one goal by choices (see the module documentation). `menu`: the goal's menu (`eng.menu(g)`); `script`: an action whose choices `Choice.want` reports."""
    def __init__(self, eng, g, menu, script=None, rare=True):
        self.eng = eng; self.g = g; self.menu = menu; self.script = script; self.rare = rare
        self.k = eng.k; self.w = eng.w
        self.space = eng.space(g)
        self.action = None; self.failed = None; self.choice = None; self.trace = []      # trace: (choice name, index) taken
        self._gen = self._main()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------- driving
    def start(self):
        return self._advance(None)

    def choose(self, i):
        if self.choice is None: raise ActionError('the action is complete')
        if not 0 <= i < len(self.choice.options): raise ActionError('no option %r (the choice %s has %d)' % (i, self.choice.name, len(self.choice.options)))
        self.trace.append((self.choice.name, i))
        return self._advance(i)

    def _advance(self, i):
        try:
            self.choice = self._gen.send(i)
        except StopIteration as e:
            self.choice = None; self.action = e.value
        except (Infeasible, GB.BuildError, ActionError, E.ITPError, RuntimeError) as e:
            if isinstance(e, E.FuelExhausted): raise
            self.choice = None; self.action = None; self.failed = '%s: %s' % (type(e).__name__, e)
        return self.choice

    @property
    def done(self): return self.choice is None

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------- helpers
    def _want(self, options, key):
        """the index of the option row `key` among the rows (None if the script says nothing, or says something that is not offered)"""
        if self.script is None or key is None: return None
        try: return options.index(key)
        except ValueError: return None

    def _menu_actions(self, kind): return [a for a in self.menu.actions if a.kind == kind]

    def _legal_kinds(self):
        g = self.g; eng = self.eng; menu = self.menu; k = self.k
        out = []
        have = {a.kind for a in menu.actions}
        for kd in MENU_KINDS:
            if kd in have: out.append((kd, 0))
        out.append((Kind.CLAIM, 0)); out.append((Kind.REFINE, 0))
        out.append((Kind.APPLY_WITH, 0)); out.append((Kind.APPLY_WITH, 1))
        out.append((Kind.REWRITE_WITH, 0))
        if g.n_hyps: out.append((Kind.FORWARD, 0))
        if any(self._is_prop_hyp(h) for h in range(g.n_hyps)): out.append((Kind.REWRITE_AT, 0))
        if eng.principles and self._induct_vars(): out.append((Kind.INDUCT_WITH, 0))
        if self._lincomb_ok(): out.append((Kind.LINCOMB, 0))
        if eng._exists_type(g.target) is not None: out.append((Kind.USE, 0))
        order = {kd: i for i, kd in enumerate(sorted(POLICY_KINDS))}
        return sorted(out, key=lambda x: (order[x[0]], x[1]))

    def _is_prop_hyp(self, h):
        try: ok, sort = self.k.infer_open(self.g.hyp_types[h], self.g.hyp_types[:h])
        except (E.ITPError, RuntimeError): return False
        return bool(ok) and sort == self.space.prop

    def _eq_hyps(self):
        return [h for h in range(self.g.n_hyps) if ACTS.eq_sides(self.eng, self.g.hyp_types[h]) is not None]

    def _lincomb_ok(self):
        return ACTS.eq_sides(self.eng, self.g.target) is not None and bool(self._eq_hyps())

    def _induct_vars(self):
        """the hypotheses that are variables of a data type which the goal (or a later hypothesis) mentions: what an induction can be on"""
        g = self.g; k = self.k; out = []
        for x in range(g.n_hyps):
            try: ok, sort = k.infer_open(g.hyp_types[x], g.hyp_types[:x])
            except (E.ITPError, RuntimeError): continue
            if not ok or sort == self.space.prop: continue
            if not any(ACTS.mentions(self.eng, t, x) for t in [g.target] + list(g.hyp_types[x + 1:])): continue
            out.append(x)
        return out

    def _sources(self, kind):
        """the heads that can be the source of an action of this kind, as option rows with the one-token codes (c, gid, 0) / (v, level, 0)"""
        eng = self.eng; g = self.g; k = self.k; limit = eng.limit
        rows = []; toks = []
        heads = self.space.table.heads
        for gid in sorted(heads):
            if gid >= limit: break
            h = heads[gid]
            if kind in (Kind.REWRITE_WITH, Kind.REWRITE_AT) and ACTS.eq_sides(eng, h.concl) is None: continue
            if kind == Kind.FORWARD and h.arity == 0: continue
            rows.append((OT_GLOBAL, gid, 0, 0)); toks.append(('c', gid, 0))
        for lv in range(g.n_hyps):
            h = self.space.local_head(lv)
            if kind in (Kind.REWRITE_WITH, Kind.REWRITE_AT) and ACTS.eq_sides(eng, h.concl) is None: continue
            if kind == Kind.FORWARD and h.arity == 0: continue
            if kind == Kind.REWRITE_AT and False: pass
            rows.append((OT_VAR, lv, 0, 0)); toks.append(('v', lv, 0))
        return rows, toks

    def _hyp_options(self, hyps, extra=()):
        return [(OT_HYP, h, 0, 0) for h in hyps] + list(extra)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------- sub-generators
    def _code(self, space, expected, goals=False, slot=0, kind=None):
        """say a term of the expected type, token by token: the finished Builder"""
        b = space.builder(expected); b.allow_goals = goals
        sc = None
        if self.script is not None and slot < len(self.script.codes): sc = self.script.codes[slot]
        while not b.done:
            toks = b.tokens(rare=self.rare)
            if not toks: raise Infeasible('no token fits %s' % b._show(b.hole_type))
            opts = [token_option(t) for t in toks]
            pos = len(b.code)
            want = None
            if sc is not None and pos < len(sc):
                try: want = toks.index(tuple(sc[pos]))
                except ValueError: want = None
            i = yield Choice('token', opts, hole=b.hole_type, slot=slot, pos=pos, want=want, builder=b, refs=toks, kind=kind, nctx=len(b.ctx))
            b.push(toks[i])
        return b

    def _source(self, kind, slot=0):
        rows, toks = self._sources(kind)
        if not rows: raise Infeasible('no source')
        want = None
        if self.script is not None and self.script.codes and len(self.script.codes[slot]) == 1:
            t = tuple(self.script.codes[slot][0])
            try: want = toks.index(t)
            except ValueError: want = None
        i = yield Choice('source', rows, slot=slot, want=want, refs=toks, kind=kind)
        return toks[i]

    def _more(self, hole, kind, n_taken, n_script_extra):
        """a decision whether another argument follows: options [stop, yes]"""
        want = None
        if self.script is not None: want = 1 if n_taken < n_script_extra else 0
        i = yield Choice('more', [(OT_STOP, 0, 0, 0), (OT_YES, 0, 0, 0)], hole=hole, want=want, kind=kind)
        return i == 1

    def _positional(self, space, ty, first_slot, kind):
        """codes for the leading binders of the type `ty`, as many as the policy wants (gym_acts.positional_args, incrementally): (the codes, the terms)"""
        k = self.k; codes = []; terms = []
        extra = (len(self.script.codes) - first_slot) if self.script is not None else 0
        while True:
            pi = ACTS.expose_pi(k, ty)
            if pi is None: break
            more = yield from self._more(pi[0], kind, len(codes), extra)
            if not more: break
            b = yield from self._code(space, pi[0], slot=first_slot + len(codes), kind=kind)
            a = b.term(); codes.append(tuple(b.code)); terms.append(a)
            ty = k.instantiate(pi[1], [a])
        return codes, terms

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------- the main generator
    def _main(self):
        kinds = self._legal_kinds()
        opts = [kind_option(kd, mode) for kd, mode in kinds]
        want = None
        if self.script is not None:
            sk = Kind(self.script.kind); key = kind_option(sk, self.script.arg if sk == Kind.APPLY_WITH else 0)
            want = self._want(opts, key)
        i = yield Choice('kind', opts, want=want, refs=kinds)
        kind, mode = kinds[i]
        handler = getattr(self, '_k_' + kind.name.lower())
        return (yield from handler(mode))

    # --- menu kinds
    def _menu_pick(self, kind, name='cand'):
        acts = self._menu_actions(kind)
        if not acts: raise Infeasible('the menu has no %s' % kind.name)
        rows = []
        for a in acts:
            term, rev, aux = self.menu.cands[a.arg]
            flags = (1 if rev else 0) | (2 if self.menu.cand_flags[a.arg] & 2 else 0)
            if kind == Kind.INTERVAL_CASES:
                h, lo, hi = aux; rows.append((OT_RANGE, int(h), int(lo), int(hi) & 0xffff))
                continue
            v = self.k.view(term)
            if v[0] == 'const': rows.append((OT_GLOBAL, v[1], 0, flags))
            elif v[0] == 'fvar': rows.append((OT_VAR, v[1], 0, flags))
            else: raise Infeasible('a candidate that is neither a global nor a hypothesis')
        want = None
        if self.script is not None and self.script.kind == kind:
            for j, a in enumerate(acts):
                if a.arg == self.script.arg: want = j
        i = yield Choice(name, rows, want=want, kind=kind, refs=acts)
        return acts[i]

    def _k_apply(self, mode):
        a = yield from self._menu_pick(Kind.APPLY)
        return a

    def _k_rewrite(self, mode):
        a = yield from self._menu_pick(Kind.REWRITE)
        return a

    def _k_interval_cases(self, mode):
        a = yield from self._menu_pick(Kind.INTERVAL_CASES, 'range')
        return a

    def _k_inject(self, mode):
        acts = self._menu_actions(Kind.INJECT)
        want = None
        if self.script is not None:
            for j, a in enumerate(acts):
                if a.arg == self.script.arg: want = j
        i = yield Choice('hyp', self._hyp_options([a.arg for a in acts]), want=want, kind=Kind.INJECT, refs=acts)
        return acts[i]

    def _nullary(self, kind):
        acts = self._menu_actions(kind)
        if not acts: raise Infeasible('the menu has no %s' % kind.name)
        return acts[0]
        yield                                                          # (a generator, though nothing is chosen)

    def _k_ring(self, mode): return (yield from self._nullary(Kind.RING))
    def _k_linarith(self, mode): return (yield from self._nullary(Kind.LINARITH))
    def _k_nlinarith(self, mode): return (yield from self._nullary(Kind.NLINARITH))
    def _k_by_contra(self, mode): return (yield from self._nullary(Kind.BY_CONTRA))

    def _split(self, kind):
        """CASES / INDUCT: the hypothesis (a menu action), then the hypotheses to generalise (any others, in the order said, until stop)"""
        acts = self._menu_actions(kind)
        want = None
        if self.script is not None:
            for j, a in enumerate(acts):
                if a.arg == self.script.arg: want = j
        i = yield Choice('hyp', self._hyp_options([a.arg for a in acts]), want=want, kind=kind, refs=acts)
        x = acts[i].arg; chosen = []
        sh = list(self.script.hyps) if self.script is not None else []
        while True:
            left = [h for h in range(self.g.n_hyps) if h != x and h not in chosen]
            if not left: break
            opts = [(OT_STOP, 0, 0, 0)] + self._hyp_options(left)
            want = None
            if self.script is not None: want = 0 if len(chosen) >= len(sh) else 1 + left.index(sh[len(chosen)]) if sh[len(chosen)] in left else None
            j = yield Choice('gen', opts, want=want, kind=kind, refs=left)
            if j == 0: break
            chosen.append(left[j - 1])
        return Action(kind, x, (), (), tuple(chosen))

    def _k_cases(self, mode): return (yield from self._split(Kind.CASES))
    def _k_induct(self, mode): return (yield from self._split(Kind.INDUCT))

    # --- the kinds that say their arguments
    def _k_claim(self, mode):
        b = yield from self._code(self.space, self.space.prop, kind=Kind.CLAIM)
        return Action(Kind.CLAIM, 0, (), (tuple(b.code),))

    def _k_refine(self, mode):
        b = yield from self._code(self.space, self.g.target, goals=True, kind=Kind.REFINE)
        return Action(Kind.REFINE, 0, (), (tuple(b.code),))

    def _k_use(self, mode):
        A = self.eng._exists_type(self.g.target)
        if A is None: raise Infeasible('the goal is no existential')
        b = yield from self._code(self.space, A, kind=Kind.USE)
        return Action(Kind.USE, 0, (), (tuple(b.code),))

    def _k_apply_with(self, mode):
        space = self.space; k = self.k; g = self.g
        src = yield from self._source(Kind.APPLY_WITH)
        h, node = ACTS.ref_of(self.eng, space, (src,))
        if mode == 1:
            codes, _ = yield from self._positional(space, h.ty, 1, Kind.APPLY_WITH)
            return Action(Kind.APPLY_WITH, 1, (), ((src,),) + tuple(codes))
        hd, _ = GB.open_pis(k, g.target); kk = h.arity - len(hd)
        if kk < 0: raise Infeasible('the goal has more binders than the source')
        b = space.builder(g.target); b.allow_goals = True
        b.push((h.tag, h.ident, kk))
        codes = []
        while not b.done:
            H = b.hole_type
            if b.is_prop(H): b.push(('g',))
            else:
                sub = yield from self._code(space, H, slot=1 + len(codes), kind=Kind.APPLY_WITH)
                codes.append(tuple(sub.code)); b.fill(sub.term())
        space.close_goals(b)
        return Action(Kind.APPLY_WITH, 0, (), ((src,),) + tuple(codes))

    def _rewrite_tail(self, kind, hyps):
        space = self.space
        src = yield from self._source(kind)
        h, node = ACTS.ref_of(self.eng, space, (src,))
        want = None
        if self.script is not None: want = int(self.script.arg)
        d = yield Choice('dir', [(OT_DIR, 0, 0, 0), (OT_DIR, 1, 0, 0)], want=want, kind=kind)
        codes, _ = yield from self._positional(space, h.ty, 1, kind)
        return Action(kind, d, (), ((src,),) + tuple(codes), tuple(hyps))

    def _k_rewrite_with(self, mode): return (yield from self._rewrite_tail(Kind.REWRITE_WITH, ()))

    def _k_rewrite_at(self, mode):
        cands = [h for h in range(self.g.n_hyps) if self._is_prop_hyp(h)]
        want = None
        if self.script is not None and self.script.hyps and self.script.hyps[0] in cands: want = cands.index(self.script.hyps[0])
        i = yield Choice('hyp', self._hyp_options(cands), want=want, kind=Kind.REWRITE_AT, refs=cands)
        x = cands[i]
        return (yield from self._rewrite_tail(Kind.REWRITE_AT, (x,)))

    def _k_forward(self, mode):
        space = self.space; k = self.k; g = self.g; eng = self.eng; HOLE = GB.HOLE
        src = yield from self._source(Kind.FORWARD)
        h, node = ACTS.ref_of(eng, space, (src,))
        sc = list(self.script.codes[1:]) if self.script is not None else []
        # phase A: the kind of each slot
        kinds_ = []                                                  # ('E',) | ('T',) | ('H', level)
        while len(kinds_) < h.arity:
            opts = [(OT_STOP, 0, 0, 0), (OT_SLOT, SLOT_EMPTY, 0, 0), (OT_SLOT, SLOT_TERM, 0, 0)] + self._hyp_options(range(g.n_hyps))
            want = None
            if self.script is not None:
                j = len(kinds_)
                if j >= len(sc): want = 0
                else:
                    c = sc[j]
                    if len(c) == 0: want = 1
                    elif len(c) == 1 and len(c[0]) == 3 and c[0][0] == 'v' and c[0][2] == 0 and isinstance(c[0][1], int) and 0 <= c[0][1] < g.n_hyps: want = 3 + c[0][1]
                    else: want = 2
            i = yield Choice('slot', opts, want=want, kind=Kind.FORWARD, slot=1 + len(kinds_))
            if i == 0: break
            kinds_.append(('E',) if i == 1 else ('T',) if i == 2 else ('H', i - 3))
        n = len(kinds_)
        if n == 0: raise Infeasible('a forward step without slots')
        # phase B: resolve (gym_acts.resolve_slots, with the terms said when they are needed)
        doms, _ = GB.open_pis(k, h.ty, count=n)
        if len(doms) < n: raise Infeasible('%d slots for a source that takes %d' % (n, len(doms)))
        conv = ACTS.conv_fn(eng, len(space.ctx))
        binding = {}
        for j, sk in enumerate(kinds_):
            if sk[0] != 'H': continue
            if not ACTS.match_any(k, doms[j], space.ctx[sk[1]], binding, conv): raise Infeasible('hypothesis %d does not fit the slot %d' % (sk[1], j))
            binding[HOLE + j] = k.mk_fvar(sk[1])
        codes = [(src,)]
        for j, sk in enumerate(kinds_):
            if sk[0] == 'H': codes.append((('v', sk[1], 0),)); continue
            if sk[0] == 'E':
                if HOLE + j not in binding: raise Infeasible('slot %d is determined by nothing' % j)
                codes.append(()); continue
            dom = GB.inst_beta(k, doms[j], binding)
            if GB.holes_of(k, dom): raise Infeasible('the type of slot %d is not known yet' % j)
            sub = yield from self._code(space, dom, slot=1 + j, kind=Kind.FORWARD)
            old = binding.get(HOLE + j)
            a = sub.term()
            if old is not None and old != a and not conv(old, a): raise Infeasible('slot %d is not what the hypotheses force' % j)
            binding[HOLE + j] = a; codes.append(tuple(sub.code))
        return Action(Kind.FORWARD, 0, (), tuple(codes))

    def _k_induct_with(self, mode):
        eng = self.eng; g = self.g; k = self.k; space = self.space; HOLE = GB.HOLE
        xs = self._induct_vars()
        want = None
        if self.script is not None and self.script.hyps and self.script.hyps[0] in xs: want = xs.index(self.script.hyps[0])
        i = yield Choice('hyp', self._hyp_options(xs), want=want, kind=Kind.INDUCT_WITH, refs=xs)
        x = xs[i]
        conv = ACTS.conv_fn(eng, g.n_hyps)
        prs = []
        for gid in sorted(eng.principles):
            if gid >= eng.limit: continue
            pr = eng.principles[gid]; binding = {}
            if ACTS.match_any(k, pr.T, g.hyp_types[x], binding, conv) and all(HOLE + j in binding for j in pr.auto): prs.append(pr)
        if not prs: raise Infeasible('no principle is about this variable')
        rows = [(OT_GLOBAL, pr.gid, 0, 0) for pr in prs]
        want = None
        if self.script is not None and self.script.codes: want = self._want(rows, (OT_GLOBAL, self.script.codes[0][0][1], 0, 0)) if len(self.script.codes[0]) == 1 else None
        i = yield Choice('source', rows, want=want, kind=Kind.INDUCT_WITH, refs=prs)
        pr = prs[i]
        binding = {}
        ACTS.match_any(k, pr.T, g.hyp_types[x], binding, conv)
        codes = [(('c', pr.gid, 0),)]
        for n, j in enumerate(pr.free):
            dom = GB.inst_beta(k, pr.head.doms[j], binding)
            if GB.holes_of(k, dom): raise Infeasible('the type of the principle\'s argument %d is not known yet' % j)
            sub = yield from self._code(space, dom, slot=1 + n, kind=Kind.INDUCT_WITH)
            binding[HOLE + j] = sub.term(); codes.append(tuple(sub.code))
        return Action(Kind.INDUCT_WITH, 0, (), tuple(codes), (x,))

    def _k_lincomb(self, mode):
        eng = self.eng; g = self.g; space = self.space; w = self.w
        with eng.in_context(g.hyp_types, g.hyp_names):
            try: A, _, _ = w.eq_parts(w.T(g.target))
            except Exception: raise Infeasible('the goal is no equation')
        eqs = self._eq_hyps(); hyps = []; codes = []
        sh = list(self.script.hyps) if self.script is not None else []
        while True:
            left = [h for h in eqs if h not in hyps]
            if not left: break
            opts = ([(OT_STOP, 0, 0, 0)] if hyps else []) + self._hyp_options(left)
            off = 1 if hyps else 0
            want = None
            if self.script is not None:
                if len(hyps) >= len(sh): want = 0 if hyps else None
                elif sh[len(hyps)] in left: want = off + left.index(sh[len(hyps)])
            i = yield Choice('lin', opts, want=want, kind=Kind.LINCOMB, refs=left)
            if hyps and i == 0: break
            h = left[i - off]
            sub = yield from self._code(space, A.n, slot=len(hyps), kind=Kind.LINCOMB)
            hyps.append(h); codes.append(tuple(sub.code))
        if not hyps: raise Infeasible('a linear combination of nothing')
        return Action(Kind.LINCOMB, 0, (), tuple(codes), tuple(hyps))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ decomposing an action
class Missing(Exception):
    """the action says something that is not among the options of the choice"""


def decompose(eng, g, menu, action):
    """the decisions (list of (Choice, index)) that assemble `action` for goal g, or raise Missing. The choices are returned as they were made (their `builder` is the live builder, so a caller that wants
    features of the hole must read them in the loop of `follow`)."""
    out = []
    for ch, i in follow(eng, g, menu, action): out.append((ch, i))
    return out


def follow(eng, g, menu, action, rare=True):
    """a generator of (Choice, index): the assembly of `action` by choosing each wanted option; raises Missing where the action says what is not offered, and ActionError when the rebuilt action differs"""
    asm = Assembler(eng, g, menu, script=action, rare=rare)
    ch = asm.start()
    while ch is not None:
        if ch.want is None:
            raise Missing('%s (%s): the action says none of the %d options' % (ch.name, ch.kind.name if ch.kind is not None else '-', len(ch.options)))
        yield ch, ch.want
        ch = asm.choose(ch.want)
    if asm.failed is not None: raise Missing('assembly failed: ' + asm.failed)
    if menu.signature(asm.action) != menu.signature(action):
        raise ActionError('the assembly rebuilt %r, not %r' % (asm.action, action))
