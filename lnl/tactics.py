"""tactics.py -- a scripting layer over the kernel's proof states: named hypotheses, normalisation of goals, rewriting by pattern, and a recorder of the primitive steps.

The kernel's own tactics (intro, apply, cases, induct, exact, rewrite) are the whole logical interface; what is added here is convenience that ends up as calls to them, so everything is
still checked by the kernel when the finished proof is admitted (World.theorem -> declare_checked):

  * hypotheses have NAMES (the kernel leaves the fields and induction hypotheses of a case unnamed): `p.induct('n', names=['m', 'ih'])`;
  * `p.norm()` replaces the goal by its normal form (a `change` through an identity function), so that `add (Succ m) n` is seen as `Succ (add m n)`;
  * `p.rw(lemma, *args)` rewrites by an equation chosen by PATTERN: the lemma's leading quantifiers become pattern variables, the first matching subterm of the goal fixes them, and the
    kernel's rewrite does the work with the fully instantiated equation. Like Lean's `rw`, it then closes a goal that has become `x = x`;
  * every primitive step is recorded (`p.trace`), which is the form a learner trains on.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        self._sync_names(prev=None)

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
        raise AuthorError('%s failed: %s\n%s' % (what, e, self)) from None

    def _sync_names(self, prev, supplied=(), skip=0):
        """give every goal a name for each hypothesis: a hypothesis that is the same one as before keeps its name, new unnamed ones take the supplied names, then defaults"""
        ps = self.ps; supplied = list(supplied)
        for i in range(ps.num_goals()):
            meta = ps.goal_meta_id(i)
            if meta in self._names: continue
            n = ps.goal_num_hyps(i); kern = [ps.goal_hyp_name(i, h) for h in range(n)]
            old_alias, old_kern = prev if prev else ([], [])
            names = []; to_skip = skip
            for h in range(n):
                if h < len(old_alias) and kern[h] == old_kern[h]: names.append(old_alias[h])
                elif kern[h]: names.append(kern[h])
                elif to_skip > 0: names.append('p%d' % h); to_skip -= 1          # the copies of the family's parameters that a constructor pattern brings along
                elif supplied: names.append(supplied.pop(0))
                else: names.append('h%d' % h)
            self._names[meta] = names

    def _context(self, i=0):
        ps = self.ps; n = ps.goal_num_hyps(i)
        return [ps.goal_hyp_type(i, h) for h in range(n)], list(self._names[ps.goal_meta_id(i)])

    @contextmanager
    def focus(self, i=0):
        """terms built inside the `with` see the hypotheses of goal i as context variables (use p.h(name))"""
        w = self.w; saved = (w.ctx, w.ctx_names); types, names = self._context(i)
        w.ctx, w.ctx_names = types, names
        try: yield
        finally: w.ctx, w.ctx_names = saved

    def h(self, name, i=0):
        """the hypothesis called `name` in goal i (the latest one if the name was reused) as a term"""
        names = self._names[self.ps.goal_meta_id(i)]
        for lvl in range(len(names) - 1, -1, -1):
            if names[lvl] == name: return self.w.T(self.k.mk_fvar(lvl))
        raise AuthorError('no hypothesis called %r; the hypotheses are %s' % (name, names))
    def _level(self, name_or_term, i=0):
        if isinstance(name_or_term, int): return name_or_term
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
        before_meta = self.ps.goal_meta_id(i); prev = (list(self._names[before_meta]), [self.ps.goal_hyp_name(i, h) for h in range(self.ps.goal_num_hyps(i))])
        try: fn()
        except (E.ITPError, RuntimeError) as e: self._fail(what() if callable(what) else what, e)       # `what` may be a function: describing a big term is not free
        self._sync_names(prev, supplied=supplied, skip=skip)

    def _n_params(self, lvl):
        """how many arguments of the type of hypothesis `lvl` are parameters of its family"""
        with self.focus():
            ty = self.w.normalize(self.w.T(self.ps.goal_hyp_type(0, lvl)))        # the kernel splits on the type with every definition unfolded (Iff is an And)
            fam, _ = self.w.spine(ty); v = self.k.view(fam.n)
        return sum(self.k.param_slots(v[1])) if v[0] == 'const' else 0

    def intro(self, *names):
        for nm in (names or ('',)):
            if self.done: raise AuthorError('intro: no goals')
            lvl = self.ps.goal_num_hyps(0); nm_ = nm or 'a%d' % lvl
            self._step('intro %s' % nm_, lambda: self.ps.intro(nm_), supplied=())
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
    def exists(self, witness):
        """for a goal `Ex A P`: choose the witness; the goal becomes `P witness`"""
        self.apply('ex_intro'); self.exact(witness); self._norm_all(); return self

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
