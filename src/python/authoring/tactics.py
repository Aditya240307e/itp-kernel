"""tactics.py -- a scripting layer over the kernel's proof states: named hypotheses, normalisation of goals, rewriting by pattern, and a recorder of the primitive steps.

The kernel's own tactics (intro, apply, cases, induct, exact, rewrite) are the whole logical interface; what is added here is convenience that ends up as calls to them, so everything is
still checked by the kernel when the finished proof is admitted (World.theorem -> declare_checked):

  * hypotheses have NAMES (the kernel leaves the fields and induction hypotheses of a case unnamed): `p.induct('n', names=['m', 'ih'])`;
  * `p.norm()` replaces the goal by its normal form (a `change` through an identity function), so that `add (Succ m) n` is seen as `Succ (add m n)`;
  * `p.rw(lemma, *args)` rewrites by an equation chosen by PATTERN: the lemma's leading quantifiers become pattern variables, the first matching subterm of the goal fixes them, and the
    kernel's rewrite does the work with the fully instantiated equation. Like Lean's `rw`, it then closes a goal that has become `x = x`;
  * every primitive step is recorded (`p.trace`), which is the form a learner trains on.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys
from contextlib import contextmanager
import itp_env_lnl as E
from author import AuthorError, Term, World

HOLE_BASE = 1 << 28                      # pattern variables are context variables with levels no real context reaches


class Goal:
    """a read-only snapshot of one goal"""
    def __init__(self, p, idx):
        self.p = p; ps = p.ps; w = p.w
        self.meta = ps.goal_meta_id(idx)
        n = ps.goal_num_hyps(idx)
        self.hyps = [(p._names[self.meta][h] if self.meta in p._names else ps.goal_hyp_name(idx, h), w.T(ps.goal_hyp_type(idx, h))) for h in range(n)]
        self.target = w.T(ps.goal_target(idx))
    def __str__(self):
        names = [nm for nm, _ in self.hyps]; w = self.p.w
        lines = ['%s : %s' % (nm, w.show(ty, names=names[:i])) for i, (nm, ty) in enumerate(self.hyps)]
        return '\n'.join(lines + ['|- ' + w.show(self.target, names=names)])


class Proof:
    def __init__(self, w, stmt, name=None, auto_norm=True):
        self.w = w; self.k = w.k; self.name = name; self.stmt = w.term(stmt, 'statement'); self.auto_norm = auto_norm
        if w.ctx: raise AuthorError('a proof must be started at the top level')
        self.ps = self.k.start_proof(self.stmt.n)
        self._names = {}                  # goal meta id -> display names of its hypotheses (the kernel's own names, completed)
        self.trace = []                   # (tactic, arguments) in the order applied; arguments are handles / numbers
        self._sync_names()

    @classmethod
    def from_goal(cls, w, target, hyp_types, hyp_names, auto_norm=True):
        """a proof state with ONE goal given as data (the target and the types and names of the hypotheses, kernel handles): what a search works on. There is no statement: such a state is for
        trying tactics and reading the goals they leave, and `term()` of a finished one is a proof of the goal in its context, not of a closed statement."""
        if w.ctx: raise AuthorError('a proof must be started at the top level')
        self = cls.__new__(cls)
        self.w = w; self.k = w.k; self.name = None; self.stmt = None; self.auto_norm = auto_norm
        self.ps = w.k.materialize_from_node(target, list(hyp_types), list(hyp_names))
        self._names = {}; self.trace = []
        self._sync_names()
        return self

    # ------------------------------------------------------------------------------------------------------------------ goals
    @property
    def n_goals(self): return self.ps.num_goals()
    @property
    def done(self): return self.ps.num_goals() == 0
    def goal(self, i=0):
        if i >= self.ps.num_goals(): raise AuthorError('no goal %d (%d open)' % (i, self.ps.num_goals()))
        return Goal(self, i)
    def __str__(self):
        if self.done: return 'no goals'
        return '\n\n'.join(('goal %d:\n' % i if self.n_goals > 1 else '') + str(self.goal(i)) for i in range(self.n_goals))
    def _fail(self, what, e):
        if self.w.probing: err = AuthorError('%s failed: %s' % (what, e))                  # a search throws these messages away: do not print the proof state for each
        else: err = AuthorError('%s failed: %s\n%s' % (what, e, self))
        err.fuel = isinstance(e, E.FuelExhausted)
        raise err from None

    def _sync_names(self, supplied=(), skip=0):
        """give every new goal a name for each hypothesis: the kernel remembers the name of a hypothesis (also when cases/induct revert it and introduce it again), so a hypothesis that has one
        keeps it; new unnamed ones take the supplied names, then defaults, and the kernel is told what was chosen"""
        ps = self.ps; supplied = list(supplied)
        for i in range(ps.num_goals()):
            meta = ps.goal_meta_id(i)
            if meta in self._names: continue
            n = ps.goal_num_hyps(i); kern = [ps.goal_hyp_name(i, h) for h in range(n)]
            names = []; to_skip = skip
            for h in range(n):
                if kern[h]: names.append(kern[h])
                elif to_skip > 0: names.append('p%d' % h); to_skip -= 1          # the copies of the family's parameters that a constructor pattern brings along
                elif supplied: names.append(supplied.pop(0))
                else: names.append('h%d' % h)
            self._names[meta] = names
            for h in range(n):
                if not kern[h]: ps.set_goal_hyp_name(i, h, names[h])          # the kernel keeps the name when it reverts and introduces the hypothesis again (cases/induct with later hypotheses)

    def _check_goal(self, i):
        if not 0 <= i < self.ps.num_goals(): raise AuthorError('no goal %d (%d open)' % (i, self.ps.num_goals()))

    def _context(self, i=0):
        self._check_goal(i); ps = self.ps; n = ps.goal_num_hyps(i)
        return [ps.goal_hyp_type(i, h) for h in range(n)], list(self._names[ps.goal_meta_id(i)])

    @contextmanager
    def focus(self, i=0):
        """terms built inside the `with` see the hypotheses of goal i as context variables (use p.h(name))"""
        w = self.w; saved = (w.ctx, w.ctx_names); types, names = self._context(i)
        w.ctx, w.ctx_names = types, names
        try: yield
        finally: w.ctx, w.ctx_names = saved

    def T(self, names, f=None, i=0):
        """terms built under the hypotheses of goal i: `p.T('a b', lambda a, b: a * b)` applies f to the hypotheses called a and b (the notation of the world applies); without `f`, the list of those hypotheses"""
        with self.focus(i):
            hs = [self.h(n, i) for n in names.split()]
            return f(*hs) if f is not None else hs

    def h(self, name, i=0):
        """the hypothesis called `name` in goal i (the latest one if the name was reused) as a term"""
        self._check_goal(i); names = self._names[self.ps.goal_meta_id(i)]
        for lvl in range(len(names) - 1, -1, -1):
            if names[lvl] == name: return self.w.T(self.k.mk_fvar(lvl))
        raise AuthorError('no hypothesis called %r; the hypotheses are %s' % (name, names))
    def _level(self, name_or_term, i=0):
        if isinstance(name_or_term, int):
            self._check_goal(i)
            if not 0 <= name_or_term < self.ps.goal_num_hyps(i): raise AuthorError('no hypothesis at level %d; goal %d has %d hypotheses' % (name_or_term, i, self.ps.goal_num_hyps(i)))
            return name_or_term
        t = self.h(name_or_term, i) if isinstance(name_or_term, str) else name_or_term
        v = self.k.view(t.n)
        if v[0] != 'fvar': raise AuthorError('expected a hypothesis, got %s' % self.w.show(t))
        return v[1]

    # ------------------------------------------------------------------------------------------------------------------ resolving arguments
    def t(self, x, i=0):
        """a term from: a Term; a hypothesis name, then a global name; an int (a numeral); a tuple (f, *args); a function of the Proof"""
        w = self.w
        if isinstance(x, Term): return x
        if isinstance(x, bool): raise AuthorError('expected a term, got a bool')
        if isinstance(x, int): return w.term(x)
        if isinstance(x, str):
            names = self._names[self.ps.goal_meta_id(i)] if self.ps.num_goals() > i else []
            if x in names: return self.h(x, i)
            return w.const(x)
        if isinstance(x, tuple):
            head = self.t(x[0], i)
            with self.focus(i):
                return w.app(head, *[self.t(a, i) for a in x[1:]])
        if callable(x):
            with self.focus(i): return w.term(x(self))
        raise AuthorError('cannot make a term out of %r' % (x,))

    # ------------------------------------------------------------------------------------------------------------------ the primitive tactics
    def _step(self, what, fn, supplied=(), i=0, skip=0):
        try: fn()
        except (E.ITPError, RuntimeError) as e: self._fail(what() if callable(what) else what, e)       # `what` may be a function: describing a big term is not free
        self._sync_names(supplied=supplied, skip=skip)

    def _n_params(self, lvl):
        """how many arguments of the type of hypothesis `lvl` are parameters of its family"""
        with self.focus():
            ty = self.w.T(self.ps.goal_hyp_type(0, lvl)); v = None
            for _ in range(32):                    # the kernel splits on the type with the definitions at its head unfolded (Iff is an And): one step at a time, on syntax -- normalising the whole type would unfold what is inside the arguments as well
                fam, _ = self.w.spine(ty); v = self.k.view(fam.n)
                if v[0] == 'const' and self.k.is_inductive(v[1]): break
                nxt = self.k.unfold_head(ty.n)
                if nxt is None: v = None; break
                ty = self.w.T(nxt)
            else: v = None
            if v is None:                          # the family appears only by computation
                fam, _ = self.w.spine(self.w.normalize(self.w.T(self.ps.goal_hyp_type(0, lvl)))); v = self.k.view(fam.n)
        return sum(self.k.param_slots(v[1])) if v[0] == 'const' else 0

    def intro(self, *names):
        for nm in (names or ('',)):
            if self.done: raise AuthorError('intro: no goals')
            lvl = self.ps.goal_num_hyps(0); nm_ = nm or 'a%d' % lvl
            self._step('intro %s' % nm_, lambda: self.ps.intro(nm_, None, True), supplied=())
            self.trace.append(('intro', nm_))
        return self
    def intros(self, *names): return self.intro(*names)

    def apply(self, term, *args):
        t = self.t(term)
        if args:
            with self.focus(): t = self.w.app(t, *[self.t(a) for a in args])
        self._step(lambda: 'apply %s' % self.w.show(t), lambda: self.ps.apply(t.n, None, True))
        self.trace.append(('apply', t.n))
        if self.auto_norm: self._norm_all()
        return self

    def split(self): return self.apply('conj')          # A /\ B : two goals
    def left(self): return self.apply('inl')            # A \/ B : prove A
    def right(self): return self.apply('inr')           # A \/ B : prove B
    def exfalso(self): return self.apply('false_elim')  # prove anything from False: the goal becomes False
    def _ex_parts(self):
        """(A, P) when the first goal is `Ex A P` -- looking through definitions at its head (`a | b` is one) -- else None"""
        w = self.w; k = self.k; ex = w.gid.get('Ex')
        if ex is None or self.done: return None
        cur = self.ps.goal_target(0)
        for _ in range(8):
            n = cur; args = []
            while True:
                v = k.view(n)
                if v[0] != 'app': break
                args.append(v[2]); n = v[1]
            args.reverse()
            v = k.view(n)
            if v[0] != 'const': return None
            if v[1] == ex and len(args) == 2: return w.T(args[0]), w.T(args[1])
            cur = k.unfold_head(cur)
            if cur is None: return None
        return None
    def exists(self, witness):
        """for a goal `Ex A P`: choose the witness; the goal becomes `P witness`. The kernel is given the whole of `ex_intro A P witness` at once, so no metavariable for the witness is left in the goal
        (a goal that mentions an assigned metavariable is expensive to compare with its normal form)."""
        parts = self._ex_parts()
        if parts is None: self.apply('ex_intro'); self.exact(witness); self._norm_all(); return self
        w = self.t(witness)
        return self.apply('ex_intro', parts[0], parts[1], w)

    def have(self, name, fact, script=None):
        """add a fact to the hypotheses. `have('h', proof_term)` reads its statement off the term; `have('h', statement, script)` first proves the statement by `script(p)` (which must close exactly
        that goal). Both are the kernel's `apply` of the library lemma `cut : forall P Q : Prop, P -> (P -> Q) -> Q`, followed by an `intro`."""
        w = self.w
        with self.focus():
            G = self.goal().target
            if script is None:
                proof = self.t(fact); P = w.simp(w.type_of(proof))
            else:
                P = self.t(fact); proof = None
        before = self.n_goals
        self.apply('cut', P, G, *([proof] if proof is not None else []))
        if proof is None:
            script(self)
            if self.n_goals != before: raise AuthorError('have %s: the script did not close exactly the statement it was asked to prove\n%s' % (name, self))
        return self.intro(name)
    def obtain(self, names, fact, script=None):
        """`have` followed by `cases`: names for the parts of the fact (a conjunction or an existential, nested to the right: pass a list of names)"""
        tmp = '_obtain'
        self.have(tmp, fact, script)
        return self.cases(tmp, names=names)

    def exact(self, term):
        t = self.t(term)
        self._step(lambda: 'exact %s' % self.w.show(t), lambda: self.ps.exact(t.n))
        self.trace.append(('exact', t.n)); return self

    def cases(self, var, names=None, generalizing=()):
        lvl = self._level(var); gen = [self._level(g) for g in generalizing]
        sup = [x for grp in (names or []) for x in (grp if isinstance(grp, (list, tuple)) else [grp])]
        self._step('cases %s' % var, lambda: self.ps.cases(lvl, None, gen, True), supplied=sup, skip=self._n_params(lvl))
        self.trace.append(('cases', lvl))
        if self.auto_norm: self._norm_all()
        return self
    def induct(self, var, names=None, generalizing=(), indexed=False):
        lvl = self._level(var); gen = [self._level(g) for g in generalizing]
        sup = [x for grp in (names or []) for x in (grp if isinstance(grp, (list, tuple)) else [grp])]
        eq = self.w.gid.get('Eq') if indexed else None; rf = self.w.gid.get('refl') if indexed else None
        self._step('induct %s' % var, lambda: self.ps.induct(lvl, None, gen, eq, rf, True), supplied=sup, skip=self._n_params(lvl))
        self.trace.append(('induct', lvl, tuple(gen)))
        if self.auto_norm: self._norm_all()
        return self

    def clear(self, levels, i=0):
        """remove the hypotheses at these levels from goal i (the goal and the other hypotheses must not mention them): the proof of the new goal is the proof of the old one, with the hypotheses that stay. The
        survivors keep their names and order and are renumbered."""
        self._check_goal(i); levels = sorted(set(levels))
        if not levels: return self
        self._step('clear %s' % (levels,), lambda: self.ps.clear(levels, self.ps.goal_meta_id(i)), i=i)
        self.trace.append(('clear', tuple(levels))); return self

    # ------------------------------------------------------------------------------------------------------------------ normalisation
    def norm(self, i=0):
        """replace goal i by its normal form (definitional unfolding of everything that computes)"""
        g = self.goal(i); types, names = self._context(i)
        nf = self.w.T(self.k.simp_normalize(g.target.n, len(types)))
        if nf == g.target: return self
        self._step('norm', lambda: self.ps.change(nf.n, self.ps.goal_meta_id(i)), i=i)
        self.trace.append(('change', nf.n)); return self
    def _norm_all(self):
        for i in range(self.n_goals): self.norm(i)

    # ------------------------------------------------------------------------------------------------------------------ equality
    def refl(self):
        g = self.goal()
        with self.focus():
            A, a, b = self.w.eq_parts(g.target); r = self.w.app(self.w['refl'], A, a)
        return self.exact(r)

    def rewrite(self, eq, rev=False):
        """rewrite with an equation given as a term (or hypothesis / global name): occurrences of its left side become its right side (rev: the other way)"""
        e = self.t(eq); g = self.goal()
        types, names = self._context()
        with self.focus():
            pty = self.w.simp(self.w.type_of(e))
            A, a, b = self.w.eq_parts(pty)
        x, y = (b, a) if rev else (a, b)
        if rev:
            with self.focus(): e = self.w.app(self.w['symm'], A, a, b, e)
        self._step(lambda: 'rewrite %s' % self.w.show(e), lambda: self.ps.rewrite(e.n, x.n, y.n, self.w.gid['transport'], self.w.gid['symm']))
        self.trace.append(('rewrite', e.n, x.n, y.n))
        if self.auto_norm: self._norm_all()
        return self

    # --- rewriting by pattern -------------------------------------------------------------------------------------------
    def _peel(self, lemma_ty, nargs=None):
        """open the leading quantifiers of a lemma's type as pattern variables: returns (hole terms, body with the holes in)"""
        w = self.w; holes = []; ty = lemma_ty
        while nargs is None or len(holes) < nargs:
            p = w.pi_parts(ty)
            if p is None: break
            dom, cod = p
            hole = w.T(self.k.mk_fvar(HOLE_BASE + len(holes))); holes.append((hole, dom))
            ty = w.instantiate(cod, hole)
        return holes, ty

    def _match(self, pat, term, binding):
        """first-order matching of `pat` (holes are context variables >= HOLE_BASE) against `term`; extends `binding` (hole level -> node) or fails"""
        k = self.k; stack = [(pat, term)]
        while stack:
            a, b = stack.pop()
            va = k.view(a)
            if va[0] == 'fvar' and va[1] >= HOLE_BASE:
                if va[1] in binding:
                    if binding[va[1]] != b: return False
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

    def _find(self, pat, target):
        """the first subterm of `target` (pre-order, function before argument) with no loose bound variable that matches `pat`; returns the binding or None"""
        k = self.k; seen = set(); stack = [target]
        while stack:
            t = stack.pop()
            if t in seen: continue
            seen.add(t)
            if k.loose_range(t) == 0:
                binding = {}
                if self._match(pat, t, binding): return binding, t
            v = k.view(t); kind = v[0]
            if kind == 'app': stack.append(v[2]); stack.append(v[1])
            elif kind in ('lam', 'pi'): stack.append(v[2]); stack.append(v[1])
            elif kind == 'let': stack += [v[3], v[2], v[1]]
            elif kind == 'match': stack += list(reversed(v[3])) + [v[2], v[1]]
            elif kind == 'fix': stack += [v[3], v[1]]
        return None

    def rw(self, lemma, *args, rev=False, close=True):
        """rewrite by a lemma or hypothesis: its quantified variables not given in `args` are found by matching its left side (right side if rev) in the goal"""
        w = self.w; e = self.t(lemma); types, names = self._context()
        self.norm()
        g = self.goal(); target = g.target.n
        with self.focus():
            args_t = [self.t(a) for a in args]
            ety = w.type_of(w.app(e, *args_t)) if args_t else w.type_of(e)
            ety = w.simp(ety)
            holes, body = self._peel(ety)
            A, lhs, rhs = w.eq_parts(body)
        pat = rhs if rev else lhs
        found = self._find(pat.n, target)
        if found is None:
            raise AuthorError('rw %s: no instance of the pattern %s in the goal\n%s' % (lemma if isinstance(lemma, str) else w.show(e), w.show(pat, names=[]), self))
        binding, inst = found
        missing = [hole for hole, dom in holes if self.k.view(hole.n)[1] not in binding]
        if missing: raise AuthorError('rw %s: the pattern does not determine all the quantified variables of the lemma (give them as arguments)' % lemma)
        with self.focus():
            full = w.app(e, *args_t, *[w.T(binding[self.k.view(hole.n)[1]]) for hole, dom in holes])
        self.rewrite(full, rev=rev)
        if close and not self.done: self.try_refl()
        return self

    def _replace(self, t, old, new):
        """`t` with every occurrence of the term `old` (which has no loose bound variable) replaced by `new`"""
        k = self.k; memo = {}
        def go(x):
            if x == old: return new
            r = memo.get(x)
            if r is not None: return r
            v = k.view(x); kind = v[0]
            if kind == 'app': r = k.mk_app(go(v[1]), go(v[2]))
            elif kind == 'lam': r = k.mk_lambda(go(v[1]), go(v[2]))
            elif kind == 'pi': r = k.mk_pi(go(v[1]), go(v[2]))
            elif kind == 'match': r = k.mk_match(go(v[1]), go(v[2]), [go(b) for b in v[3]])
            elif kind in ('let', 'fix'):
                if self._contains(x, old): raise AuthorError('rewriting inside a %s is not supported' % kind)
                r = x
            else: r = x
            memo[x] = r; return r
        return go(t)

    def _contains(self, t, sub):
        k = self.k; seen = set(); stack = [t]
        while stack:
            x = stack.pop()
            if x == sub: return True
            if x in seen: continue
            seen.add(x)
            v = k.view(x); kind = v[0]
            if kind == 'app': stack += [v[1], v[2]]
            elif kind in ('lam', 'pi'): stack += [v[1], v[2]]
            elif kind == 'let': stack += [v[1], v[2], v[3]]
            elif kind == 'match': stack += [v[1], v[2]] + list(v[3])
            elif kind == 'fix': stack += [v[1], v[3]]
        return False

    def rw_at(self, hyp, lemma, *args, rev=False, new=None):
        """rewrite inside a hypothesis: like `rw`, but the pattern is looked for in the type of hypothesis `hyp`. The rewritten hypothesis is added under the same name (or `new`); the original stays in the
        context, shadowed. The proof is `transport` along the equation, with the motive read off the occurrences of the instance in the hypothesis."""
        w = self.w; k = self.k
        e = self.t(lemma); h = self.h(hyp) if isinstance(hyp, str) else hyp
        with self.focus():
            hty = w.simp(w.type_of(h))
            args_t = [self.t(a) for a in args]
            ety = w.simp(w.type_of(w.app(e, *args_t)) if args_t else w.type_of(e))
            holes, body = self._peel(ety)
            A, lhs, rhs = w.eq_parts(body)
        pat = rhs if rev else lhs
        found = self._find(pat.n, hty.n)
        if found is None:
            raise AuthorError('rw %s at %s: no instance of the pattern %s in the hypothesis\n%s' % (lemma if isinstance(lemma, str) else w.show(e), hyp, w.show(pat, names=[]), self))
        binding, inst = found
        missing = [hole for hole, dom in holes if k.view(hole.n)[1] not in binding]
        if missing: raise AuthorError('rw %s at %s: the pattern does not determine all the quantified variables of the lemma (give them as arguments)' % (lemma, hyp))
        with self.focus():
            full = w.app(e, *args_t, *[w.T(binding[k.view(hole.n)[1]]) for hole, dom in holes])
            A_ty, a, b = w.eq_parts(w.simp(w.type_of(full)))
            x, y = (b, a) if rev else (a, b)
            eq_xy = w.app(w['symm'], A_ty, a, b, full) if rev else full
            L = HOLE_BASE + 1
            motive = w.T(k.mk_lambda(A_ty.n, k.abstract_fvars(self._replace(hty.n, x.n, k.mk_fvar(L)), [L])))
            proof = w.app(w['transport'], A_ty, motive, x, y, eq_xy, h)
        name = new or (hyp if isinstance(hyp, str) else 'h')
        return self.have(name, proof)

    def try_refl(self):
        """close the first goal if it is an equation whose sides are the same term"""
        if self.done: return False
        g = self.goal()
        with self.focus():
            try: A, a, b = self.w.eq_parts(g.target)
            except AuthorError: return False
            same = a == b or self.w.conv(a, b)
        if same:
            self.refl(); return True
        return False

    def ring(self):
        """close the first goal, an equation between polynomial expressions of natural numbers, by normalising both sides (see ring.py: the proof it builds is checked by the kernel like any other)"""
        from ring import ring as _ring
        return _ring(self)

    def linear_combination(self, *pairs):
        """close the first goal, an equation of a commutative ring, by a linear combination  sum c_i * (l_i - r_i)  of hypotheses  h_i : l_i = r_i, given as pairs (c_i, h_i) (see lincomb.py: kernel-checked)"""
        from lincomb import linear_combination as _lc
        return _lc(self, *pairs)

    def linarith(self, *extra):
        """close the first goal by linear arithmetic over the integers from the hypotheses, and from the proofs in `extra` (see linarith.py: a certificate found by a simplex, checked by the kernel)"""
        from linarith import linarith as _lin
        return _lin(self, extra)

    def nlinarith(self, *extra):
        """like `linarith`, after adding the squares of the sub-terms  t * t, t ^ 2  and the products of every two facts (a certificate of degree two)"""
        from linarith import linarith as _lin
        return _lin(self, extra, nonlinear=True)

    def by_contra(self, name='h'):
        """classical: to prove P, assume  ~P  (named `name`) and prove False. Needs the world's `by_contra` (excluded middle)."""
        if 'by_contra' not in self.w.gid: raise AuthorError('by_contra: this world has no classical logic (no lemma by_contra)')
        self.apply('by_contra'); return self.intro(name)

    def interval_cases(self, var, lo, hi, name=None):
        """case analysis on an integer hypothesis `var`: the bounds  lo <= var <= hi  are proved by `linarith` from the context, and the goal is replaced by hi - lo + 1 goals, in increasing order of the value,
        each with a hypothesis  <name or var_eq> : var = k  (a few helper hypotheses `_ic_*` stay in the context). The proof is a chain of the library's `Z.lt_or_eq_of_le`."""
        import ring as R
        w = self.w; name = name or (var + '_eq')
        if not isinstance(lo, int) or not isinstance(hi, int) or isinstance(lo, bool) or isinstance(hi, bool): raise AuthorError('interval_cases: the bounds must be Python ints')
        if hi < lo: raise AuthorError('interval_cases: empty range %d..%d' % (lo, hi))
        if hi - lo >= 500: raise AuthorError('interval_cases: %d cases are too many' % (hi - lo + 1))
        with self.focus():
            A = w.infer(self.h(var))
        if 'Z' not in w.gid or A != w['Z']: raise AuthorError('interval_cases: %s is not an integer' % var)
        lit = R.instance_for(w, A).lit; le = w['Z.le']
        self.have('_ic_lo', self.T(var, lambda x: le(lit(lo), x)), lambda q: q.linarith())
        self.have('_ic_hi', self.T(var, lambda x: le(x, lit(hi))), lambda q: q.linarith())
        for k in range(hi, lo, -1):
            self.obtain(['_ic_c', name], ('Z.lt_or_eq_of_le', var, lit(k), '_ic_hi'))          # var < k  \/  var = k : the first case goes on
            self.have('_ic_hi', self.T(var, lambda x, k=k: le(x, lit(k - 1))), lambda q: q.linarith())
        self.have(name, self.T(var, lambda x: w.Eq(x, lit(lo))), lambda q: q.linarith())
        return self

    def induction_on(self, lemma, var, names=(), extra=()):
        """induction by a library principle on a hypothesis `var`, for a type that `induct` cannot split (a quotient) or a principle of another shape than the type's own (induction from a starting point, strong
        induction, in both directions). The principle is  lemma : forall P [extra..] [z], premises -> [forall n, side premises ->] P z.  The motive P is read off the goal by abstracting `var`; `extra` are the
        principle's other data arguments that come before its premises (the starting point `a` of `Z.induction_from` and `Z.strong_induction`); the variable itself is passed too when the principle takes it there
        (`N.induction_on`, `Z.induction_on`), and otherwise found by unification with the goal. The goals are the premises, in order: the cases first, then the side premises that only talk about the variable
        (`a <= n`: `assumption` or `linarith` closes them). `names` introduce the binders of the first goal. Hypotheses that depend on `var` are not generalised: do the induction before introducing them."""
        w = self.w; lvl = self._level(var)
        with self.focus():
            x = self.h(var) if isinstance(var, str) else var; g = self.goal().target
            ty = w.T(self.ps.goal_hyp_type(0, lvl))
            motive = w.T(self.k.mk_lambda(ty.n, self.k.abstract_fvars(g.n, [lvl])))
            given = [self.t(e) for e in extra]
            cur = w.type_of(self.t(lemma))                    # the principle's type, past the motive and the extras
            for a in [motive] + given:
                parts = w.pi_parts(cur)
                if parts is None: raise AuthorError('induction_on: %s takes fewer arguments than the motive and %d more' % (lemma, len(given)))
                cur = w.instantiate(parts[1], a)
            parts = w.pi_parts(cur)
            takes_var = False
            if parts is not None:                             # the next binder is the variable's if it is data (a premise is a proposition)
                ok, sort = self.k.infer_open(parts[0].n, w.ctx)
                takes_var = bool(ok) and sort != w.PROP.n
        self.last_induction = {'extra': given, 'takes_var': takes_var}
        self.apply(lemma, motive, *given, *([x] if takes_var else []))
        if names: self.intro(*names)
        return self

    def calc(self, e0, *steps):
        """a chain of equations  e0 = e1 = ... = en  closing the first goal (which must be `e0 = en`, up to computation). Each step is `(justification, next_expression)`; the justification of `e_i = e_{i+1}` is
        'ring' (both sides normalise to the same polynomial), a proof (a term or hypothesis name whose statement is that equation, either way round), or a function of the Proof that proves it by tactics.
        Expressions may be terms or functions of the Proof (so that they can mention the hypotheses by name)."""
        w = self.w
        with self.focus():
            cur = self.t(e0); A = w.infer(cur); exprs = [cur]
            for _, nxt in steps: exprs.append(self.t(nxt))
        names = []
        for i, (just, _) in enumerate(steps):
            with self.focus(): stmt = w.Eq(exprs[i], exprs[i + 1])
            if just == 'ring': script = lambda q: q.ring()
            elif callable(just) and not isinstance(just, Term): script = just
            else: script = lambda q, j=just, a=exprs[i], b=exprs[i + 1]: q._calc_exact(j, a, b)
            name = '_calc%d' % i; self.have(name, stmt, script); names.append(name)
        with self.focus():
            acc = self.h(names[0])
            for i in range(1, len(names)): acc = w.app(w['trans'], A, exprs[0], exprs[i], exprs[i + 1], acc, self.h(names[i]))
        return self.exact(acc)

    def _calc_exact(self, j, prev, nxt):
        w = self.w
        with self.focus():
            jt = self.t(j); A, a, b = w.eq_parts(w.simp(w.type_of(jt)))
            fwd = w.conv(a, prev) and w.conv(b, nxt); rev = (not fwd) and w.conv(b, prev) and w.conv(a, nxt)
            if not (fwd or rev): raise AuthorError('calc: the justification proves %s = %s, not the step %s = %s' % (w.show(a), w.show(b), w.show(prev), w.show(nxt)))
            proof = jt if fwd else w.app(w['symm'], A, a, b, jt)
        return self.exact(proof)

    def assumption(self):
        """close the first goal with a hypothesis whose type is the goal (up to computation)"""
        g = self.goal(); types, names = self._context()
        for lvl in range(len(types) - 1, -1, -1):
            if self.k.conv_open(g.hyps[lvl][1].n, g.target.n, len(types)):
                return self.exact(self.w.T(self.k.mk_fvar(lvl)))
        raise AuthorError('assumption: no hypothesis matches\n%s' % self)

    # ------------------------------------------------------------------------------------------------------------------ finishing
    def term(self):
        if not self.done: raise AuthorError('the proof is not finished:\n%s' % self)
        return self.w.T(self.ps.extract())
