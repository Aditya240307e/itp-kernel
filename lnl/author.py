"""author.py -- writing mathematics against the hybrid kernel from Python.

NOTHING HERE IS TRUSTED. This module builds terms and drives the kernel's tactics; every declaration it makes goes through the kernel's CHECKED entry points (declare_checked,
finish_recursive_function_checked, finish_inductive), so a bug here can make a library fail to build, never make a false theorem appear. What it adds on top of the raw `mk_*` API:

  * `Term`   a kernel term together with the world that owns it (a handle from another kernel is an error here, not a memory hazard), with `t(a, b)` for application and operators
             (`+ * ** <= <`, `& | ~`, `>>`) that build whatever the world's current NOTATION says they mean;
  * `World`  one kernel plus a name table, and higher-order-abstract-syntax binders: `w.pi(A, lambda x: B(x))`, `w.lam(...)`, `w.forall_([A, B], lambda a, b: ...)`, written in the
             LOCALLY NAMELESS style (a bound variable is a fresh context variable while the body is built, abstracted when the binder is closed);
  * declarations `w.inductive`, `w.define`, `w.recursive`, `w.theorem`, all kernel-checked;
  * reading terms back (`w.spine`, `w.view`) and a printer (`w.show`) for diagnostics.

The tactic layer (named hypotheses, rewriting with patterns) is in tactics.py, the equality-proof combinators in eqproof.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itp_env_lnl as E


class AuthorError(Exception):
    """a mistake in how the library is being written (not a kernel soundness matter): ill-typed input, unknown name, a proof script that did not finish"""


class Term:
    """A term of one world. Immutable; two Terms are equal iff they are the same node of the same world (terms are hash-consed, so this is syntactic equality)."""
    __slots__ = ('w', 'n')

    def __init__(self, w, n): self.w = w; self.n = n
    def __call__(self, *args): return self.w.app(self, *args)
    def __eq__(self, other): return isinstance(other, Term) and self.w is other.w and self.n == other.n
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(self.n)
    def __repr__(self): return self.w.show(self)
    def __bool__(self): raise AuthorError('a Term has no truth value (write eq(a, b) for the proposition a = b, and == only to compare terms as syntax: use `is_same`)')
    def is_same(self, other): return self.__eq__(other)
    # ---- notation: these build whatever the world's current notation says (see World.notation)
    def __add__(self, o): return self.w.nota.add(self, o)
    def __radd__(self, o): return self.w.nota.add(o, self)
    def __mul__(self, o): return self.w.nota.mul(self, o)
    def __rmul__(self, o): return self.w.nota.mul(o, self)
    def __pow__(self, o): return self.w.nota.pow(self, o)
    def __sub__(self, o): return self.w.nota.sub(self, o)
    def __neg__(self): return self.w.nota.neg(self)
    def __le__(self, o): return self.w.nota.le(self, o)
    def __lt__(self, o): return self.w.nota.lt(self, o)
    def __ge__(self, o): return self.w.nota.le(o, self)
    def __gt__(self, o): return self.w.nota.lt(o, self)
    def __and__(self, o): return self.w.And(self, o)
    def __or__(self, o): return self.w.Or(self, o)
    def __invert__(self): return self.w.Not(self)
    def __rshift__(self, o): return self.w.imp(self, o)          # a >> b : a -> b


class Notation:
    """what the operators of Term mean in the current context; a library installs one for its numbers (nat_lib.NatNotation)"""
    def _no(self, what): raise AuthorError('no notation for %s is installed in this world' % what)
    def add(self, a, b): self._no('+')
    def mul(self, a, b): self._no('*')
    def pow(self, a, b): self._no('**')
    def sub(self, a, b): self._no('-')
    def neg(self, a): self._no('unary -')
    def le(self, a, b): self._no('<=')
    def lt(self, a, b): self._no('<')
    def lit(self, n): self._no('numerals')


class DispatchNotation(Notation):
    """the operators and numerals of a world with several kinds of number: each registered Notation serves one type, and an operator picks the one of its operand's type (a Python int takes the type of
    the other operand, or the `default` type). So `a + b` is `add` on Nat and `N.add` on N with the same source text."""
    def __init__(self, w): self.w = w; self.by_type = {}; self.default = None
    def register(self, ty, nota, default=False):
        self.by_type[self.w._n(ty)] = nota
        if default or self.default is None: self.default = nota
    def using(self, ty):
        """`with w.nota.using(N):` -- Python ints without a Term operand mean numerals of this type"""
        d = self; nota = self.by_type[self.w._n(ty)]
        class _Using:
            def __enter__(s): s.prev = d.default; d.default = nota
            def __exit__(s, *exc): d.default = s.prev
        return _Using()
    def _pick(self, *ops):
        if len(self.by_type) <= 1: return self.default                 # one kind of number: no need to look at the operands
        for o in ops:
            if isinstance(o, Term):
                nt = self.by_type.get(self.w.type_of(o).n)
                if nt is not None: return nt
        return self.default
    def add(self, a, b): return self._pick(a, b).add(a, b)
    def mul(self, a, b): return self._pick(a, b).mul(a, b)
    def pow(self, a, b): return self._pick(a).pow(a, b)
    def sub(self, a, b): return self._pick(a, b).sub(a, b)
    def neg(self, a): return self._pick(a).neg(a)
    def le(self, a, b): return self._pick(a, b).le(a, b)
    def lt(self, a, b): return self._pick(a, b).lt(a, b)
    def lit(self, n): return self.default.lit(n)


# The work budget of one call into the kernel, in steps (see FuelExhausted in itp_kernel.hpp): a wrong or pathological action fails after this much work instead of running on. Every call to a library
# so far has needed under 10^7 steps; this is two orders of magnitude of headroom for the large proof terms of a later library (about half a minute). ITP_FUEL overrides it (0 = unlimited).
DEFAULT_FUEL = int(os.environ.get('ITP_FUEL', 2 * 10 ** 9))


class World:
    def __init__(self, parameters=True, prop_elim=True, fuel=None):
        self.k = E.Kernel()
        self.k.set_option('prop_elim', prop_elim)
        self.k.set_option('parameters', parameters)
        self.k.set_fuel(DEFAULT_FUEL if fuel is None else fuel)
        self.ctx = []                       # types (handles) of the open context variables, outermost first; the variable at level l is mk_fvar(l)
        self.ctx_names = []                 # display names of those variables
        self.gid = {}                       # global name -> id
        self.gname = {}                     # id -> name
        self.nota = Notation()
        self.infix = {}                     # global name -> (symbol, precedence) for the printer
        self._const_cache = {}
        self.proofs = {}                    # theorem name -> the primitive steps of its proof (what a learner trains on)
        self.PROP = self.T(self.k.mk_type(0))      # propositions live in Type0, data in Type1 (the Prop/Type world)
        self.TYPE = self.T(self.k.mk_type(1))

    # ------------------------------------------------------------------------------------------------------------------ handles
    def T(self, n): return Term(self, n)
    def _n(self, t, what='term'):
        if isinstance(t, Term):
            if t.w is not self: raise AuthorError('a term of another world was used as the %s' % what)
            return t.n
        if isinstance(t, bool) or not isinstance(t, int): raise AuthorError('expected a Term as the %s, got %r' % (what, t))
        return self.nota.lit(t).n                          # a Python int stands for a numeral of the current notation
    def term(self, x, what='term'): return x if isinstance(x, Term) and x.w is self else self.T(self._n(x, what))

    # ------------------------------------------------------------------------------------------------------------------ constants and application
    def const(self, name):
        t = self._const_cache.get(name)
        if t is None:
            if name not in self.gid: raise AuthorError('unknown global %r' % name)
            t = self._const_cache[name] = self.T(self.k.mk_const(self.gid[name]))
        return t
    c = const
    def __getitem__(self, name): return self.const(name)
    def __contains__(self, name): return name in self.gid

    def app(self, f, *args):
        n = self._n(f, 'function')
        for a in args: n = self.k.mk_app(n, self._n(a, 'argument'))
        return self.T(n)

    # ------------------------------------------------------------------------------------------------------------------ binders (locally nameless)
    def _bind(self, dom, name=None):
        lvl = len(self.ctx); self.ctx.append(self._n(dom, 'binder type')); self.ctx_names.append(name or 'x%d' % lvl)
        return lvl, self.T(self.k.mk_fvar(lvl))
    def _unbind(self, count=1):
        for _ in range(count): self.ctx.pop(); self.ctx_names.pop()

    @staticmethod
    def _argname(fn):
        code = getattr(fn, '__code__', None)
        return code.co_varnames[0] if code is not None and code.co_argcount else None

    def _binder(self, mk, dom, fn):
        dom = self.term(dom, 'binder type'); lvl, x = self._bind(dom, self._argname(fn))
        try: body = self.term(fn(x), 'body')
        finally: self._unbind()
        return self.T(mk(dom.n, self.k.abstract_fvars(body.n, [lvl])))
    def lam(self, dom, fn): return self._binder(self.k.mk_lambda, dom, fn)
    def pi(self, dom, fn): return self._binder(self.k.mk_pi, dom, fn)
    def arrow(self, a, b): return self.T(self.k.mk_pi(self._n(a, 'domain'), self._n(b, 'codomain')))      # non-dependent: the codomain has no loose bound variable
    imp = arrow

    def _telescope(self, kind, doms, fn):
        """doms: each a Term or a function of the variables bound so far; fn receives all the variables"""
        if not doms: return self.term(fn(), 'body')
        base = len(self.ctx); vars_ = []; domt = []; names = []
        try:
            for d in doms:
                if isinstance(d, tuple): nm, d = d
                else: nm = None
                dom = self.term(d(*vars_) if (callable(d) and not isinstance(d, Term)) else d, 'binder type')
                lvl, x = self._bind(dom, nm); vars_.append(x); domt.append(dom)
            if hasattr(fn, '__code__'):                     # name the variables after the parameters of the Python function, for readable goals
                for i, nm in enumerate(fn.__code__.co_varnames[:fn.__code__.co_argcount]):
                    if i < len(vars_): self.ctx_names[base + i] = nm
            body = self.term(fn(*vars_), 'body')
        finally: self._unbind(len(vars_))
        return self.T(self.k.close_telescope(kind, [d.n for d in domt], body.n, base))
    def forall_(self, doms, fn): return self._telescope('pi', doms, fn)
    def fun(self, doms, fn): return self._telescope('lam', doms, fn)

    # ------------------------------------------------------------------------------------------------------------------ typing
    def infer(self, t):
        ok, ty = self.k.infer_open(self._n(t), self.ctx)
        if not ok: raise AuthorError('ill-typed term: %s\n  %s' % (ty, self.show(t)))
        return self.T(ty)
    def type_of(self, t):
        """the type of t read off the DECLARED types of its head and the arguments (no evaluation, so definitions stay folded: the kernel's own `infer` hands back delta-normalised types);
        falls back to `infer` when a Pi has to be exposed by unfolding"""
        t = self.term(t); head, args = self.spine(t); v = self.k.view(head.n)
        if v[0] == 'const': ty = self.T(self.k.global_info(v[1])[1])
        elif v[0] == 'fvar' and v[1] < len(self.ctx): ty = self.T(self.ctx[v[1]])
        else: return self.infer(t)
        for a in args:
            parts = self.pi_parts(ty)
            if parts is None: return self.infer(t)
            ty = self.instantiate(parts[1], a)
        return ty
    def check(self, t, ty):
        ok, msg = self.k.check_open(self._n(t), self._n(ty), self.ctx)
        if not ok: raise AuthorError('the term does not have the stated type: %s\n  term: %s\n  type: %s' % (msg, self.show(t), self.show(ty)))
    def normalize(self, t): return self.T(self.k.normalize_open(self._n(t), len(self.ctx)))
    def simp(self, t):
        """the display / rewriting normal form: computation where a recursive function's decreasing argument is a constructor, every definition left folded"""
        return self.T(self.k.simp_normalize(self._n(t), len(self.ctx)))
    def conv(self, a, b): return self.k.conv_open(self._n(a), self._n(b), len(self.ctx))

    # ------------------------------------------------------------------------------------------------------------------ reading terms
    def view(self, t): return self.k.view(self._n(t))
    def spine(self, t):
        """(head, [arguments]) of an application, iteratively"""
        n = self._n(t); args = []
        while True:
            v = self.k.view(n)
            if v[0] != 'app': break
            args.append(v[2]); n = v[1]
        args.reverse()
        return self.T(n), [self.T(a) for a in args]
    def head_name(self, t):
        """the name of the global at the head of an application, or None"""
        h, _ = self.spine(t); v = self.k.view(h.n)
        return self.gname.get(v[1]) if v[0] == 'const' else None
    def is_app_of(self, t, name, nargs=None):
        h, args = self.spine(t); v = self.k.view(h.n)
        return v[0] == 'const' and self.gname.get(v[1]) == name and (nargs is None or len(args) == nargs)
    def pi_parts(self, t):
        """(domain, codomain) of a Pi (codomain with a loose bound variable 0), or None"""
        v = self.k.view(self._n(t)); return (self.T(v[1]), self.T(v[2])) if v[0] == 'pi' else None
    def instantiate(self, body, *vals):
        """replace the loose bound variables of `body`: vals[i] for bound variable i"""
        return self.T(self.k.instantiate(self._n(body), [self._n(v) for v in vals]))
    def global_type(self, name): return self.T(self.k.global_info(self.gid[name])[1])
    def global_value(self, name):
        v = self.k.global_info(self.gid[name])[2]; return None if v is None else self.T(v)

    # ------------------------------------------------------------------------------------------------------------------ declarations
    def _register(self, name, gid):
        if name in self.gid and self.gid[name] != gid: raise AuthorError('the name %r is already declared' % name)
        self.gid[name] = gid; self.gname[gid] = name
    def _at_top(self, what):
        if self.ctx: raise AuthorError('%s must be declared at the top level, not under %d open binder(s)' % (what, len(self.ctx)))

    def axiom(self, name, ty):
        self._at_top('an axiom'); self._fresh_name(name)
        self._register(name, self.k.declare_unchecked(self._n(ty), None, name)); return self.const(name)
    def _fresh_name(self, name):
        if name in self.gid: raise AuthorError('the name %r is already declared' % name)
    def define(self, name, ty, val):
        """a checked definition: the kernel type-checks `val` against `ty`"""
        self._at_top('a definition'); self._fresh_name(name)
        try: gid = self.k.declare_checked(self._n(ty), self._n(val), name)
        except (E.ITPError, RuntimeError) as e: raise AuthorError('definition %s was rejected: %s\n  type: %s\n  value: %s' % (name, e, self.show(ty), self.show(val))) from None
        self._register(name, gid); return self.const(name)

    def inductive(self, name, ty, ctors):
        """declare an inductive family. ty: its type (a sort, or a Pi telescope ending in one: parameters and indices alike).
        ctors: list of (ctor_name, fields, indices): `fields` is a list of binder types (a Term, or a function of the fields so far; the family itself is available as w[name]) and
        `indices` is a function of the field variables returning the list of index terms of the constructor's result (omit it, or give None, for a family without indices)."""
        self._at_top('an inductive type'); self._fresh_name(name)
        tid = self.k.reserve_inductive_type(name, self._n(ty)); self._register(name, tid)
        specs = []
        for spec in ctors:
            cname, fields, *rest = spec; idx_fn = rest[0] if rest else None
            vars_ = []; doms = []
            try:
                for d in fields:
                    dom = self.term(d(*vars_) if (callable(d) and not isinstance(d, Term)) else d, 'constructor field type')
                    lvl, x = self._bind(dom); vars_.append(x); doms.append(dom)
                idxs = [self.term(i, 'constructor result index') for i in (idx_fn(*vars_) if idx_fn else [])]
            finally: self._unbind(len(vars_))
            specs.append(E.CtorSpec(cname, [d.n for d in doms], [i.n for i in idxs]))
        try: h = self.k.finish_inductive(tid, specs)
        except (E.ITPError, RuntimeError) as e: raise AuthorError('inductive type %s was rejected: %s' % (name, e)) from None
        for (cname, *_), cg in zip(ctors, h.ctor_global_ids): self._register(cname, cg)
        return self.const(name)

    def recursive(self, name, ty, body_fn, on=0, over=None):
        """a structurally recursive function. `body_fn(f)` receives the function itself (for the recursive calls) and returns the whole body (usually lambdas around a match).
        `on`: the position, among the Pi arguments of `ty`, of the argument the recursion decreases; `over`: the name of the inductive type of that argument (read off `ty` when omitted)."""
        self._at_top('a recursive function'); self._fresh_name(name)
        if over is None: over = self._rec_family(ty, on)
        gid = self.k.reserve_recursive_function(name, self._n(ty)); self._register(name, gid)
        try:
            body = self.term(body_fn(self.const(name)), 'function body')
            self.k.finish_recursive_function_checked(gid, self.gid[over], on, body.n)
        except (E.ITPError, RuntimeError) as e: raise AuthorError('recursive function %s was rejected: %s' % (name, e)) from None
        return self.const(name)
    def _rec_family(self, ty, on):
        cur = self.term(ty)
        for i in range(on + 1):
            p = self.pi_parts(cur)
            if p is None: raise AuthorError('the type of the function has no argument %d' % on)
            dom, cur = p
        h, _ = self.spine(dom); v = self.k.view(h.n)
        if v[0] != 'const' or v[1] not in self.gname: raise AuthorError('cannot tell which inductive type argument %d of the function has; pass over=' % on)
        return self.gname[v[1]]

    # ------------------------------------------------------------------------------------------------------------------ matching (non-indexed families, parameters allowed)
    def ctor_fields(self, ctor_name, params=()):
        """the field types of a constructor, as functions of the earlier fields: returns the list of binder types for `fields`, instantiated at `params`
        (a constructor type is  Pi params. Pi fields. Family params indices ; with the parameter option the parameters are not abstracted in a branch)"""
        ty = self.global_type(ctor_name); out = []
        for p in params:
            parts = self.pi_parts(ty)
            if parts is None: raise AuthorError('constructor %s has fewer parameters than supplied' % ctor_name)
            ty = self.instantiate(parts[1], p)
        return ty

    def match(self, scrut, motive, branches):
        """case analysis. motive: a function of the scrutinee giving the result type (or a Term, an already built lambda).
        branches: one per constructor, in declaration order; each a Term (for a constructor without fields) or a function of the constructor's field variables."""
        scrut = self.term(scrut, 'scrutinee'); sty = self.infer(scrut)
        fam, fargs = self.spine(sty); v = self.k.view(fam.n)
        if v[0] != 'const': raise AuthorError('cannot match on a term whose type is not an inductive family: %s' % self.show(sty))
        fam_gid = v[1]
        slots = self.k.param_slots(fam_gid); nparams = sum(slots)
        if list(slots[:nparams]) != [1] * nparams: raise AuthorError('match: the parameters of %s are not a prefix of its arguments' % self.gname[fam_gid])
        if isinstance(motive, Term): mot = motive
        else:
            if len(fargs) > nparams: raise AuthorError('a motive for an indexed family must be passed as a Term (a lambda over the parameters, the indices and the scrutinee)')
            mot = self._motive(fam_gid, fargs[:nparams], sty, motive)
        ctors = self.constructors(fam_gid)
        if len(ctors) != len(branches): raise AuthorError('%s has %d constructors, %d branches given' % (self.gname[fam_gid], len(ctors), len(branches)))
        bs = [br if isinstance(br, Term) else self._branch(cname, fargs[:nparams], br) for cname, br in zip(ctors, branches)]
        return self.T(self.k.mk_match(scrut.n, mot.n, [b.n for b in bs]))

    def _motive(self, fam_gid, params, sty, fn):
        """the kernel's motive of a match on a family WITH PARAMETERS and no indices, from a function of the scrutinee: one unused binder per parameter slot, then the scrutinee (whose type mentions the actual
        parameters). Without parameters this is just `fun (x : T) => fn x`."""
        ty = self.T(self.k.global_info(fam_gid)[1]); doms = []; base = len(self.ctx)
        try:
            for prm in params:
                parts = self.pi_parts(ty)
                if parts is None: raise AuthorError('%s has fewer parameters than supplied' % self.gname[fam_gid])
                self._bind(parts[0]); doms.append(parts[0]); ty = self.instantiate(parts[1], prm)
            lvl, q = self._bind(sty, self._argname(fn)); doms.append(sty)
            body = self.term(fn(q), 'motive body')
        finally: self._unbind(len(self.ctx) - base)
        return self.T(self.k.close_telescope('lam', [d.n for d in doms], body.n, base))

    def constructors(self, family):
        """the constructor names of an inductive family (a name or a global id), in declaration order"""
        gid = self.gid[family] if isinstance(family, str) else family
        return [self.gname[c] for c in self.k.global_info(gid)[5]]

    def _branch(self, cname, params, fn):
        """one branch of a match: `fn` receives the constructor's FIELDS. The kernel's branches of a family with parameters also carry one binder per parameter slot, in front and unused (their types are those of the
        constructor's leading Pi's), while the types of the fields mention the scrutinee's own parameters: `match p with fun (A' : Type) (B' : Type) (a : A) (b : B) => a` for `p : Prod A B`."""
        ty = self.global_type(cname); vars_ = []; doms = []
        base = len(self.ctx)
        try:
            for prm in params:
                parts = self.pi_parts(ty)
                if parts is None: raise AuthorError('constructor %s has fewer parameters than supplied' % cname)
                lvl, x = self._bind(parts[0]); doms.append(parts[0]); ty = self.instantiate(parts[1], prm)
            nparams = len(doms)
            while True:
                p = self.pi_parts(ty)
                if p is None: break
                dom = self.instantiate_with(p[0], vars_)
                lvl, x = self._bind(dom); vars_.append(x); doms.append(dom)
                ty = p[1]
            if hasattr(fn, '__code__'):
                for i, nm in enumerate(fn.__code__.co_varnames[:fn.__code__.co_argcount]):
                    if i < len(vars_): self.ctx_names[base + nparams + i] = nm
            body = self.term(fn(*vars_), 'branch')
        finally: self._unbind(len(self.ctx) - base)
        return self.T(self.k.close_telescope('lam', [d.n for d in doms], body.n, base))
    def instantiate_with(self, body, vars_):
        """`body` has loose bound variables for the vars_ bound so far (innermost last): instantiate them"""
        if not vars_: return body
        return self.instantiate(body, *reversed(vars_))

    def prove(self, name, stmt, script, auto_norm=True):
        """prove `stmt` with a tactic script (a function of a tactics.Proof), and admit the extracted term through the kernel's checked path"""
        from tactics import Proof
        self._at_top('a theorem'); self._fresh_name(name)
        stmt = self.term(stmt, 'statement'); p = Proof(self, stmt, name, auto_norm)
        script(p)
        if not p.done: raise AuthorError('the proof of %s did not finish:\n%s' % (name, p))
        term = p.term()
        try: gid = self.k.declare_checked(stmt.n, term.n, name)
        except (E.ITPError, RuntimeError) as e: raise AuthorError('theorem %s was rejected by the kernel: %s\n  statement: %s' % (name, e, self.show(stmt))) from None
        self._register(name, gid); self.proofs[name] = p.trace; return self.const(name)

    # ------------------------------------------------------------------------------------------------------------------ logic and equality
    def Eq(self, a, b):
        a = self.term(a, 'left side'); b = self.term(b, 'right side')
        return self.app(self['Eq'], self.infer(a), a, b)
    eq = Eq
    def And(self, a, b): return self.app(self['And'], a, b)
    def Or(self, a, b): return self.app(self['Or'], a, b)
    def Not(self, a): return self.app(self['Not'], a)
    def Iff(self, a, b): return self.app(self['Iff'], a, b)
    def Ex(self, A, fn): return self.app(self['Ex'], A, self.lam(A, fn))

    def eq_parts(self, prop):
        """(A, a, b) of a proposition  Eq A a b, seen through the definitions at its head (one syntactic delta step at a time: the sides are left as written) -- and, failing that, through its display
        normal form"""
        cand = prop
        for _ in range(32):
            h, args = self.spine(cand)
            if len(args) == 3 and self.head_name(cand) == 'Eq': return args[0], args[1], args[2]
            nxt = self.k.unfold_head(self._n(cand))
            if nxt is None: break
            cand = self.T(nxt)
        cand = self.simp(prop)
        h, args = self.spine(cand)
        if len(args) == 3 and self.head_name(cand) == 'Eq': return args[0], args[1], args[2]
        raise AuthorError('not an equation: %s' % self.show(prop))

    # ------------------------------------------------------------------------------------------------------------------ the printer
    def show(self, t, depth=40, names=None):
        names = list(self.ctx_names) if names is None else list(names)
        try: return self._show(self._n(t), names, [], 0)
        except RecursionError: return '<term too deep to print>'
    def _name_of_const(self, gid): return self.gname.get(gid, 'C%d' % gid)
    def _fresh_bound(self, bound, names):
        base = 'x'; i = len(bound)
        while True:
            nm = '%s%d' % (base, i)
            if nm not in bound and nm not in names: return nm
            i += 1
    def _show(self, n, names, bound, prec):
        k = self.k; v = k.view(n); kind = v[0]
        P = lambda s, p: ('(' + s + ')') if prec > p else s
        if kind == 'type': return {0: 'Prop', 1: 'Type'}.get(v[1], 'Type%d' % v[1])
        if kind == 'bvar': return bound[-1 - v[1]] if v[1] < len(bound) else '#%d' % v[1]
        if kind == 'fvar': return names[v[1]] if v[1] < len(names) else '?f%d' % v[1]
        if kind == 'const': return self._name_of_const(v[1])
        if kind == 'meta': return '?m%d' % v[1]
        if kind == 'app':
            args = []; h = n
            while True:
                vv = k.view(h)
                if vv[0] != 'app': break
                args.append(vv[2]); h = vv[1]
            args.reverse(); hv = k.view(h)
            if hv[0] == 'const':
                nm = self._name_of_const(hv[1])
                if nm == 'Eq' and len(args) == 3: return P('%s = %s' % (self._show(args[1], names, bound, 5), self._show(args[2], names, bound, 5)), 4)
                if nm in self.infix and len(args) == 2:
                    sym, pr = self.infix[nm]
                    return P('%s %s %s' % (self._show(args[0], names, bound, pr + 1), sym, self._show(args[1], names, bound, pr + 1)), pr)
            return P(' '.join([self._show(h, names, bound, 10)] + [self._show(a, names, bound, 11) for a in args]), 10)
        if kind == 'lam':
            nm = self._fresh_bound(bound, names)
            return P('fun (%s : %s) => %s' % (nm, self._show(v[1], names, bound, 0), self._show(v[2], names, bound + [nm], 0)), 0)
        if kind == 'pi':
            nm = self._fresh_bound(bound, names)
            if self._uses_bvar0(v[2]): return P('forall (%s : %s), %s' % (nm, self._show(v[1], names, bound, 0), self._show(v[2], names, bound + [nm], 0)), 0)
            return P('%s -> %s' % (self._show(v[1], names, bound, 1), self._show(v[2], names, bound + ['_'], 0)), 0)
        if kind == 'let':
            nm = self._fresh_bound(bound, names)
            return P('let %s : %s := %s in %s' % (nm, self._show(v[1], names, bound, 0), self._show(v[2], names, bound, 0), self._show(v[3], names, bound + [nm], 0)), 0)
        if kind == 'match':
            return P('match %s with %s' % (self._show(v[1], names, bound, 0), ' | '.join(self._show(b, names, bound, 1) for b in v[3])), 0)
        if kind == 'fix': return P('fix<%d> %s' % (v[2], self._show(v[3], names, bound, 0)), 0)
        return '<%s>' % kind
    def _uses_bvar0(self, n, depth=0):
        """does bound variable `depth` occur in n? (DAG walk with a seen set)"""
        seen = set(); stack = [(n, depth)]
        while stack:
            t, d = stack.pop()
            if (t, d) in seen: continue
            seen.add((t, d)); v = self.k.view(t); kd = v[0]
            if kd == 'bvar':
                if v[1] == d: return True
            elif kd in ('app',): stack += [(v[1], d), (v[2], d)]
            elif kd in ('lam', 'pi'): stack += [(v[1], d), (v[2], d + 1)]
            elif kd == 'let': stack += [(v[1], d), (v[2], d), (v[3], d + 1)]
            elif kd == 'match': stack += [(v[1], d), (v[2], d)] + [(b, d) for b in v[3]]
            elif kd == 'fix': stack += [(v[1], d), (v[3], d + 1)]
        return False
