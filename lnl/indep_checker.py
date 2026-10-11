"""indep_checker.py -- an independently written type checker for the kernel's fragment (finding F8).

Written from the SPECIFICATION, not from the C++ kernel, and deliberately built differently:
  * terms use NAMED variables with explicit capture-avoiding substitution (no de Bruijn indices, no levels);
  * conversion is by weak-head normalisation (no NbE, no closures, no shared caches);
  * the universe rule, the structural-recursion guard, the parameter analysis and the Match typing rule are re-implemented here.
It shares only the *input format* (hybrid term trees, as printed by the kernel) with the code under test.

Terms:  ('v', name)  ('c', gid)  ('s', k)  ('app', f, a)  ('lam', x, A, b)  ('pi', x, A, B)
        ('match', scrutinee, motive, (branch, ...))  ('fix', d, fn_type, self_name, body)
Specification summary (the points where kernels can differ):
  sorts:      Type k : Type (k+1); no cumulativity. Pi : Type (imax ...) where a codomain in Type 0 makes the Pi live in Type 0 (impredicative), else the max.
  delta:      a constant with a value unfolds whenever it is the head of a term (recursive functions are ordinary lambda/match definitions).
  iota:       match on a fully applied constructor selects the branch and applies it to the constructor's arguments.
  fix:        (fix d T self. body) a_0 .. a_n unfolds to body[self := fix] a_0 .. a_n when a_d is a constructor application; guarded by structural recursion.
  eta:        function eta in both directions: (lam x : A. b) is convertible with any term t that is not itself a lambda (after weak-head normalisation)
              exactly when b is convertible with  t x  for a fresh variable x. The binder annotation is not compared (the two sides are assumed to have the same type).
  match:      scrutinee type  F c_0 .. c_{k-1}.  For constructor c with fields x_j : A_j and result indices r_i, scan the slots left to right with a map sigma:
              a slot whose pattern is a bare field variable v forces v := c_i (if v is already forced, the forced value is matched against c_i instead);
              two applications of the same constructor match argument by argument; two DIFFERENT constructors of one family refute the branch (the scan stops);
              anything else is ignored.  Field types are specialised by ALL forced entries; the result indices and the constructor application are specialised only by
              the fields that occur at most once as a bare result index, or that a PARAMETER slot determined (parameter mode).  The branch must have the type
              Pi (x_j : A_j sigma). motive r'_0 .. r'_{k-1} (c x'_0 ..).  The motive must be well typed and have k+1 Pi layers ending in a sort.
  quotients:  the five roles type / mk / lift / ind / sound are ordinary constants (no value). The checker does NOT trust the kernel's labels: it derives j and the
              equality family E from the declared type of `sound`, then demands that every labelled constant has EXACTLY the type below (alpha-equivalence), that E is
              E : (A : Type j) -> A -> A -> Type 0 with the single constructor refl : (A : Type j) -> (a : A) -> E A a a, and that j >= 1. Only then does the one extra
              reduction rule exist:   lift A r B f h (mk A' r' a) ~> f a   (the sixth argument is a mk applied to exactly three arguments). Types, with Prop = Type 0:
                Quot : (A : Type j) -> (A -> A -> Prop) -> Type j          mk : (A : Type j) -> (r : A -> A -> Prop) -> A -> Quot A r
                lift : (A : Type j) -> (r) -> (B : Type j) -> (f : A -> B) -> ((a b : A) -> r a b -> E B (f a) (f b)) -> Quot A r -> B
                ind  : (A : Type j) -> (r) -> (P : Quot A r -> Prop) -> ((a : A) -> P (mk A r a)) -> (q : Quot A r) -> P q
                sound: (A : Type j) -> (r) -> (a b : A) -> r a b -> E (Quot A r) (mk A r a) (mk A r b)
  parameters: a prefix of the family's slots; slot i is a parameter when in EVERY constructor result_i is a bare field variable different from the earlier parameters'
              variables and EVERY occurrence of the family in any field type is fully applied and passes that same variable at slot i.
"""
import itertools

class CheckError(Exception): pass
class Unsupported(Exception): pass

_ctr = itertools.count()
def fresh(base='x'): return '%s#%d' % (base.split('#')[0], next(_ctr))

# ---------------------------------------------------------------- conversion from / to the kernel's hybrid trees
def from_hybrid(t, names=()):
    k = t[0]
    if k == 'bvar':
        if t[1] >= len(names): raise CheckError('loose bound variable')
        return ('v', names[len(names) - 1 - t[1]])
    if k == 'fvar': return ('v', 'free%d' % t[1])
    if k == 'const': return ('c', t[1])
    if k == 'type': return ('s', t[1])
    if k == 'app': return ('app', from_hybrid(t[1], names), from_hybrid(t[2], names))
    if k in ('lam', 'pi'):
        x = fresh(); return (k, x, from_hybrid(t[1], names), from_hybrid(t[2], names + (x,)))
    if k == 'match': return ('match', from_hybrid(t[1], names), from_hybrid(t[2], names), tuple(from_hybrid(b, names) for b in t[3]))
    if k == 'fix':
        x = fresh('self'); return ('fix', t[1], from_hybrid(t[2], names), x, from_hybrid(t[3], names + (x,)))
    raise Unsupported(k)

def to_hybrid(t, names=()):
    k = t[0]
    if k == 'v':
        if t[1] in names: return ('bvar', len(names) - 1 - max(i for i, n in enumerate(names) if n == t[1]))
        if t[1].startswith('free'): return ('fvar', int(t[1][4:]))
        raise CheckError('unbound variable ' + t[1])
    if k == 'c': return ('const', t[1])
    if k == 's': return ('type', t[1])
    if k == 'app': return ('app', to_hybrid(t[1], names), to_hybrid(t[2], names))
    if k in ('lam', 'pi'): return (k, to_hybrid(t[2], names), to_hybrid(t[3], names + (t[1],)))
    if k == 'match': return ('match', to_hybrid(t[1], names), to_hybrid(t[2], names), tuple(to_hybrid(b, names) for b in t[3]))
    if k == 'fix': return ('fix', t[1], to_hybrid(t[2], names), to_hybrid(t[4], names + (t[3],)))
    raise Unsupported(k)

# ---------------------------------------------------------------- syntax helpers
def spine(t):
    args = []
    while t[0] == 'app': args.append(t[2]); t = t[1]
    return t, args[::-1]
def apply_args(h, args):
    for a in args: h = ('app', h, a)
    return h
def fv(t):
    k = t[0]
    if k == 'v': return {t[1]}
    if k in ('c', 's'): return set()
    if k == 'app': return fv(t[1]) | fv(t[2])
    if k in ('lam', 'pi'): return fv(t[2]) | (fv(t[3]) - {t[1]})
    if k == 'match':
        r = fv(t[1]) | fv(t[2])
        for b in t[3]: r |= fv(b)
        return r
    if k == 'fix': return fv(t[2]) | (fv(t[4]) - {t[3]})
    raise Unsupported(k)
def subst(t, x, s, fvs=None):
    """capture-avoiding substitution t[x := s]"""
    if fvs is None: fvs = fv(s)
    k = t[0]
    if k == 'v': return s if t[1] == x else t
    if k in ('c', 's'): return t
    if k == 'app': return ('app', subst(t[1], x, s, fvs), subst(t[2], x, s, fvs))
    if k in ('lam', 'pi'):
        _, y, A, B = t; A2 = subst(A, x, s, fvs)
        if y == x: return (k, y, A2, B)
        if y in fvs: y2 = fresh(y); B = subst(B, y, ('v', y2)); y = y2
        return (k, y, A2, subst(B, x, s, fvs))
    if k == 'match': return ('match', subst(t[1], x, s, fvs), subst(t[2], x, s, fvs), tuple(subst(b, x, s, fvs) for b in t[3]))
    if k == 'fix':
        _, d, fty, y, body = t; fty2 = subst(fty, x, s, fvs)
        if y == x: return ('fix', d, fty2, y, body)
        if y in fvs: y2 = fresh(y); body = subst(body, y, ('v', y2)); y = y2
        return ('fix', d, fty2, y, subst(body, x, s, fvs))
    raise Unsupported(k)
def canon(t, env=()):
    """alpha-canonical form: bound names become indices"""
    k = t[0]
    if k == 'v':
        for i in range(len(env) - 1, -1, -1):
            if env[i] == t[1]: return ('b', len(env) - 1 - i)
        return ('f', t[1])
    if k in ('c', 's'): return t
    if k == 'app': return ('app', canon(t[1], env), canon(t[2], env))
    if k in ('lam', 'pi'): return (k, canon(t[2], env), canon(t[3], env + (t[1],)))
    if k == 'match': return ('match', canon(t[1], env), canon(t[2], env), tuple(canon(b, env) for b in t[3]))
    if k == 'fix': return ('fix', t[1], canon(t[2], env), canon(t[4], env + (t[3],)))
    raise Unsupported(k)

# ---------------------------------------------------------------- the global environment
QUOT_ROLE_NAMES = {0: None, 1: 'type', 2: 'mk', 3: 'lift', 4: 'ind', 5: 'sound'}      # the kernel's numbering (Kernel.quot_role)

def consts_in(t):
    """the set of global ids a named term mentions (iterative: dumps of deep terms must not hit the interpreter's recursion limit)"""
    out, stack = set(), [t]
    while stack:
        u = stack.pop(); k = u[0]
        if k == 'c': out.add(u[1])
        elif k == 'app': stack.append(u[1]); stack.append(u[2])
        elif k in ('lam', 'pi'): stack.append(u[2]); stack.append(u[3])
        elif k == 'match': stack.append(u[1]); stack.append(u[2]); stack.extend(u[3])
        elif k == 'fix': stack.append(u[2]); stack.append(u[4])
        elif k not in ('v', 's'): raise Unsupported(k)
    return out

def replace_const(t, gid, repl):
    """t with every occurrence of the global `gid` replaced by the term `repl` (which must have no free variables that t binds)"""
    k = t[0]
    if k == 'c': return repl if t[1] == gid else t
    if k in ('v', 's'): return t
    if k == 'app': return ('app', replace_const(t[1], gid, repl), replace_const(t[2], gid, repl))
    if k in ('lam', 'pi'): return (k, t[1], replace_const(t[2], gid, repl), replace_const(t[3], gid, repl))
    if k == 'match': return ('match', replace_const(t[1], gid, repl), replace_const(t[2], gid, repl), tuple(replace_const(b, gid, repl) for b in t[3]))
    if k == 'fix': return ('fix', t[1], replace_const(t[2], gid, repl), t[3], replace_const(t[4], gid, repl))
    raise Unsupported(k)

class Glob:
    def __init__(self, gid, name, type_, value, ctor_family, ctor_idx, ctors, quot_role=None, inductive=None, recursion=None, reserved=0):
        self.gid, self.name, self.type, self.value, self.ctor_family, self.ctor_idx, self.ctors, self.quot_role = gid, name, type_, value, ctor_family, ctor_idx, ctors, quot_role
        # recursion: (the inductive family, the decreasing argument's position) for a global the kernel finished as a structurally recursive function, else None.
        # reserved: 0 = complete, 1/2 = still being defined (a reserved inductive type / recursive function); nothing may mention such a global.
        self.recursion, self.reserved = (tuple(recursion) if recursion is not None else None), reserved
        # inductive: the global was declared as an inductive family (possibly with no constructors). None = the exporter did not say; then a family is
        # taken to be inductive when it is not a quotient constant and has constructors (the old behaviour, which cannot see an empty inductive).
        self.inductive = inductive

class Env:
    def __init__(self): self.g = {}
    def add(self, gid, name, type_tree, value_tree, ctor_family, ctor_idx, ctors, quot_role=0, inductive=None, recursion=None, reserved=0):
        self.g[gid] = Glob(gid, name, from_hybrid(type_tree), None if value_tree is None else from_hybrid(value_tree),
                           None if ctor_family is None or ctor_family < 0 else ctor_family, None if ctor_idx is None or ctor_idx < 0 else ctor_idx, list(ctors),
                           QUOT_ROLE_NAMES[quot_role], inductive, recursion, reserved)

class Checker:
    MAX_DEPTH = 200

    def __init__(self, env, params=False, fuel=400000, prop_elim=False):
        self.env, self.g, self.params, self.fuel0 = env, env.g, params, fuel
        self.prop_elim = prop_elim; self._restricted = {}; self._pi_depth = 0; self.limit_hit = False; self.irrelevance_inferences = 0
        self.fuel = fuel; self._arity = {}
        self.quot, self.quot_problem = None, None
        self._validate_quot()

    def tick(self):
        self.fuel -= 1
        if self.fuel < 0: raise CheckError('fuel exhausted (treated as a rejection)')

    # ------------------------------------------------------------ global facts
    def arity(self, fam):
        if fam not in self._arity:
            t, n = self.g[fam].type, 0
            while t[0] == 'pi': n += 1; t = t[3]
            self._arity[fam] = n
        return self._arity[fam]
    def is_family(self, gid):
        """an inductive family: declared as one (not an opaque type constant such as an axiom `Foo : Type` or Quot), with a type ending in a sort.
        Matching on a value of an opaque type would otherwise be a branchless match, i.e. a proof of anything (finding F26)."""
        gl = self.g[gid]
        if gl.value is not None or gl.ctor_family is not None or gl.quot_role is not None: return False
        if gl.inductive is False: return False
        if gl.inductive is None and not gl.ctors: return False
        t = gl.type
        while t[0] == 'pi': t = t[3]
        return t[0] == 's'
    def ctor_fields(self, cg):
        t, n = self.g[cg].type, 0
        while t[0] == 'pi': n += 1; t = t[3]
        return n
    def open_ctor(self, cg):
        ty, fields = self.g[cg].type, []
        while ty[0] == 'pi':
            x = fresh('f'); fields.append((x, ty[2])); ty = subst(ty[3], ty[1], ('v', x))
        h, res = spine(ty)
        fam = self.g[cg].ctor_family
        if h != ('c', fam) or len(res) != self.arity(fam): raise CheckError('malformed constructor: its result is not its family applied to exactly %d arguments' % self.arity(fam))
        return fields, res
    def ctor_app(self, t):
        h, args = spine(t)
        if h[0] == 'c':
            gl = self.g[h[1]]
            if gl.ctor_family is not None and len(args) == self.ctor_fields(h[1]): return h[1], args
        return None

    # ------------------------------------------------------------ quotients (independent implementation of the stated rule)
    @staticmethod
    def quot_types(eq, j, gq, gm):
        """the canonical type of each role, as named terms (binder names are irrelevant: comparison is up to alpha)"""
        P = lambda x, d, b: ('pi', x, d, b)
        v = lambda x: ('v', x)
        ap = lambda f, *xs: apply_args(f, list(xs))
        T, Prop = ('s', j), ('s', 0)
        A, r, B, f, h, q, a, b, Pm = v('A'), v('r'), v('B'), v('f'), v('h'), v('q'), v('a'), v('b'), v('P')
        rel = P('x', A, P('y', A, Prop))
        Q = lambda X, R: ap(('c', gq), X, R)
        M = lambda X, R, x: ap(('c', gm), X, R, x)
        E = lambda *xs: ap(('c', eq), *xs)
        return {
            'type': P('A', T, P('r', rel, T)),
            'mk': P('A', T, P('r', rel, P('a', A, Q(A, r)))),
            'lift': P('A', T, P('r', rel, P('B', T, P('f', P('x', A, B), P('h', P('a', A, P('b', A, P('p', ap(r, a, b), E(B, ap(f, a), ap(f, b))))), P('q', Q(A, r), B)))))),
            'ind': P('A', T, P('r', rel, P('P', P('x', Q(A, r), Prop), P('h', P('a', A, ap(Pm, M(A, r, a))), P('q', Q(A, r), ap(Pm, q)))))),
            'sound': P('A', T, P('r', rel, P('a', A, P('b', A, P('p', ap(r, a, b), E(Q(A, r), M(A, r, a), M(A, r, b))))))),
        }

    def _validate_quot(self):
        roles = {}
        for gid, gl in self.g.items():
            if gl.quot_role is not None: roles.setdefault(gl.quot_role, []).append(gid)
        if not roles: return
        bad = lambda msg: setattr(self, 'quot_problem', msg)
        if sorted(roles) != sorted(QUOT_ROLE_NAMES[i] for i in range(1, 6)) or any(len(v) != 1 for v in roles.values()): return bad('the quotient roles are not each held by exactly one global')
        by = {r: v[0] for r, v in roles.items()}
        if any(self.g[x].value is not None or self.g[x].ctor_family is not None for x in by.values()): return bad('a quotient constant has a value or is a constructor')
        t = self.g[by['sound']].type
        if t[0] != 'pi' or t[2][0] != 's': return bad('sound has an unexpected shape')
        j = t[2][1]
        for _ in range(5):
            if t[0] != 'pi': return bad('sound has an unexpected shape')
            t = t[3]
        h, args = spine(t)
        if h[0] != 'c' or len(args) != 3: return bad('sound does not conclude with an equality')
        eq = h[1]; ge = self.g[eq]
        if j < 1: return bad('the universe level of the quotient must be >= 1')
        A, a, b = ('v', 'A'), ('v', 'a'), ('v', 'b')
        if ge.value is not None or ge.ctor_family is not None or canon(ge.type) != canon(('pi', 'A', ('s', j), ('pi', 'a', A, ('pi', 'b', A, ('s', 0))))): return bad('the equality family has the wrong type')
        if len(ge.ctors) != 1 or self.g[ge.ctors[0]].ctor_family != eq: return bad('the equality family must have exactly one constructor')
        if canon(self.g[ge.ctors[0]].type) != canon(('pi', 'A', ('s', j), ('pi', 'a', A, apply_args(('c', eq), [A, a, a])))): return bad('the equality constructor is not refl : (A : Type j) -> (a : A) -> Eq A a a')
        for role, expected in self.quot_types(eq, j, by['type'], by['mk']).items():
            if canon(self.g[by[role]].type) != canon(expected): return bad('quotient constant %r does not have its specified type' % role)
        self.quot = by

    # ------------------------------------------------------------ prop_elim: Type0 is Prop (independent implementation of the stated rule)
    def elim_restricted(self, fam):
        """a family in Type0 that is not a syntactic subsingleton can only be eliminated into Type0. Subsingleton = no constructor, or exactly one whose every field is a proof (its type lives in Type0)
        or occurs as a bare result index. Families above Type0 are data and never restricted."""
        if fam in self._restricted: return self._restricted[fam]
        t = self.g[fam].type
        while t[0] == 'pi': t = t[3]
        r = False
        if t[0] == 's' and t[1] == 0:
            ctors = self.g[fam].ctors
            if len(ctors) > 1: r = True
            elif len(ctors) == 1:
                try:
                    fields, res = self.open_ctor(ctors[0]); ctx = {}
                    for x, A in fields:
                        if self.sort_of(ctx, A) != 0 and ('v', x) not in res: r = True
                        ctx[x] = A
                except (CheckError, Unsupported): r = True
        self._restricted[fam] = r
        return r

    # ------------------------------------------------------------ parameters (independent implementation of the stated rule)
    def uniform(self, t, fam, k, i, r):
        h, args = spine(t)
        if h[0] == 'c' and h[1] == fam:
            if len(args) != k or args[i] != r: return False
            return all(self.uniform(a, fam, k, i, r) for a in args)
        kk = t[0]
        if kk in ('v', 'c', 's'): return True
        if kk == 'app': return self.uniform(t[1], fam, k, i, r) and self.uniform(t[2], fam, k, i, r)
        if kk in ('lam', 'pi'): return self.uniform(t[2], fam, k, i, r) and self.uniform(t[3], fam, k, i, r)
        if kk == 'match': return all(self.uniform(x, fam, k, i, r) for x in (t[1], t[2]) + tuple(t[3]))
        if kk == 'fix': return self.uniform(t[2], fam, k, i, r) and self.uniform(t[4], fam, k, i, r)
        return False
    def param_slots(self, fam):
        k = self.arity(fam); ctors = self.g[fam].ctors
        params = [False] * k
        if k == 0 or not ctors: return params
        opened = [self.open_ctor(c) for c in ctors]
        for i in range(k):
            ok = True
            for fields, res in opened:
                names = {x for x, _ in fields}
                if len(res) != k: ok = False; break
                r = res[i]
                if not (r[0] == 'v' and r[1] in names): ok = False; break
                if any(res[j] == r for j in range(i)): ok = False; break
                if not all(self.uniform(A, fam, k, i, r) for _, A in fields): ok = False; break
            if not ok: break
            params[i] = True
        return params

    # ------------------------------------------------------------ reduction
    def whnf(self, t, folded=False):
        """weak head normal form. With `folded`, a structurally recursive function is unfolded only when its decreasing argument is a constructor application: a call that is stuck
        stays a call (the default unfolds it into the stuck match on its decreasing argument, whose branches still mention the other arguments unevaluated)."""
        while True:
            self.tick()
            h, args = spine(t)
            k = h[0]
            if k == 'lam' and args:
                t = apply_args(subst(h[3], h[1], args[0]), args[1:]); continue
            if k == 'c':
                gl = self.g[h[1]]; v = gl.value
                if v is not None and folded and gl.recursion is not None:
                    d = gl.recursion[1]
                    if len(args) <= d or self.ctor_app(self.whnf(args[d], True)) is None: return t
                if v is not None: t = apply_args(v, args); continue
                if self.quot is not None and h[1] == self.quot['lift'] and len(args) >= 6:       # lift A r B f h (mk A' r' a) ~> f a
                    qh, qargs = spine(self.whnf(args[5]))
                    if qh[0] == 'c' and qh[1] == self.quot['mk'] and len(qargs) == 3:
                        t = apply_args(('app', args[3], qargs[2]), args[6:]); continue
                return t
            if k == 'fix':
                _, d, fty, selfn, body = h
                if len(args) > d and self.ctor_app(self.whnf(args[d])) is not None:
                    t = apply_args(subst(body, selfn, h), args); continue
                return t
            if k == 'match':
                ca = self.ctor_app(self.whnf(h[1]))
                if ca is not None:
                    cg, cargs = ca; t = apply_args(h[3][self.g[cg].ctor_idx], cargs + args); continue
                return t
            return t

    def index_nf(self, t, fuel_depth=0):
        """normalise through constructor applications (enough to expose the structure an index pattern is matched against)"""
        t = self.whnf(t); ca = self.ctor_app(t)
        if ca is None or fuel_depth > 60: return t
        h, args = spine(t)
        return apply_args(h, [self.index_nf(a, fuel_depth + 1) for a in args])

    def conv(self, a, b, depth=0, ctx=None):
        """definitional equality. `ctx` (variable -> type) is the typing context the two terms live in; with it, and in the Prop/Type world, two
        PROOFS of one proposition are equal (proof irrelevance). Without it (ctx is None) the comparison is purely structural.
        A chain of one constructor (the unary numeral Succ (Succ (... Zero)), a list) is walked by this loop, not by recursion: `conv_structural` compares every argument of a constructor
        application but the last and hands the last pair back, so the length of a chain is limited by the fuel only, not by the recursion depth (a numeral of length 200 used to be
        rejected as a 'type mismatch'). Proof irrelevance still gets its try at every level, innermost first, exactly as in the recursive formulation."""
        pending = []                                      # the levels that agreed on everything but their last argument
        while True:
            r = self.conv_structural(a, b, depth, ctx)
            if r.__class__ is not tuple: break
            pending.append((a, b, depth)); a, b = r
        ok = r or self.proof_irrelevant(a, b, depth, ctx)
        while pending:
            pa, pb, pd = pending.pop()
            ok = ok or self.proof_irrelevant(pa, pb, pd, ctx)
        return ok

    def proof_irrelevant(self, a, b, depth, ctx):
        return self.prop_elim and ctx is not None and self.irrelevant(a, b, depth, ctx)

    def irrelevant(self, a, b, depth, ctx):
        """(PI)   G |- a : P    G |- b : P'    G |- P : Type 0    P == P'   =>   a == b .   Decided by inference, never by the shape of a term. Types and type formers
        (anything whose own type is a sort) are skipped before inferring anything: they are not proofs."""
        if depth > self.MAX_DEPTH or self._pi_depth >= 4: self.limit_hit = True; return False
        wa, wb = self.whnf(a), self.whnf(b)
        if wa[0] in ('s', 'pi', 'lam') or wb[0] in ('s', 'pi', 'lam'): return False
        if self.data_ctor_app(wa) or self.data_ctor_app(wb): return False         # a numeral, a list: data (inferring its type below would give the same verdict, after type-checking the whole term)
        self._pi_depth += 1; self.irrelevance_inferences += 1
        try:
            pa = self.infer(ctx, a)
            if self.sort_of(ctx, pa) != 0: return False
            pb = self.infer(ctx, b)
            return self.conv(pa, pb, depth + 1, ctx)
        except (CheckError, Unsupported, KeyError, IndexError): return False
        finally: self._pi_depth -= 1

    def data_ctor_app(self, t):
        """t (in weak head normal form) is an application of a constructor whose family is declared in Type k, k >= 1: its type is in Type k, so it is data, never a proof"""
        h = spine(t)[0]
        if h[0] != 'c': return False
        fam = self.g[h[1]].ctor_family
        if fam is None: return False
        s = self.g[fam].type
        while s[0] == 'pi': s = s[3]
        return s[0] == 's' and s[1] >= 1

    def conv_structural(self, a, b, depth, ctx):
        self.tick()
        if canon(a) == canon(b): return True
        if depth > self.MAX_DEPTH: self.limit_hit = True; return False
        # folded first: two stuck calls of one recursive function are equal when their arguments are (unfolding them instead gives two stuck matches whose branches mention the other
        # arguments unevaluated, and comparing those never ends when an argument is not literally the same term, e.g. `add x Zero` and `add x (mul Zero x)`)
        a, b = self.whnf(a, True), self.whnf(b, True)
        if canon(a) == canon(b): return True
        ha0, aa0 = spine(a); hb0, ab0 = spine(b)
        if ha0[0] == 'c' and hb0[0] == 'c' and ha0[1] == hb0[1] and len(aa0) == len(ab0) and self.g[ha0[1]].recursion is not None:
            if all(self.conv(x, y, depth + 1, ctx) for x, y in zip(aa0, ab0)): return True
        a, b = self.whnf(a), self.whnf(b)
        if canon(a) == canon(b): return True
        ka, kb = a[0], b[0]
        ext = lambda z, A: None if ctx is None else {**ctx, z[1]: A}          # the context under a binder: the fresh variable has the binder's annotation as its type
        if ka == 's' and kb == 's': return a[1] == b[1]
        if ka in ('lam', 'pi') and kb == ka:
            if not self.conv(a[2], b[2], depth + 1, ctx): return False
            z = ('v', fresh()); return self.conv(subst(a[3], a[1], z), subst(b[3], b[1], z), depth + 1, ext(z, a[2]))
        if (ka == 'lam') != (kb == 'lam'):                  # eta:  (lam x. body) == t   iff   body == t x   (x fresh), t not a lambda
            lam, other = (a, b) if ka == 'lam' else (b, a)
            z = ('v', fresh()); body, app = subst(lam[3], lam[1], z), ('app', other, z); ctx2 = ext(z, lam[2])
            return self.conv(body, app, depth + 1, ctx2) if ka == 'lam' else self.conv(app, body, depth + 1, ctx2)
        ha, aa = spine(a); hb, ab = spine(b)
        if len(aa) != len(ab): return False
        if ha[0] == 'v' and hb[0] == 'v': same = ha[1] == hb[1]
        elif ha[0] == 'c' and hb[0] == 'c': same = ha[1] == hb[1]
        elif ha[0] == 'match' and hb[0] == 'match':
            same = (len(ha[3]) == len(hb[3]) and self.conv(ha[1], hb[1], depth + 1, ctx) and self.conv(ha[2], hb[2], depth + 1, ctx)
                    and all(self.conv(x, y, depth + 1, ctx) for x, y in zip(ha[3], hb[3])))
        elif ha[0] == 'fix' and hb[0] == 'fix':
            z = ('v', fresh('self'))
            same = (ha[1] == hb[1] and self.conv(ha[2], hb[2], depth + 1, ctx)
                    and self.conv(subst(ha[4], ha[3], z), subst(hb[4], hb[3], z), depth + 1, ext(z, ha[2])))
        else: return False
        if not same: return False
        if aa and ha[0] == 'c' and self.g[ha[1]].ctor_family is not None:       # a constructor application: all arguments but the last here, the last by the caller's loop (see `conv`)
            if not all(self.conv(x, y, depth + 1, ctx) for x, y in zip(aa[:-1], ab[:-1])): return False
            return (aa[-1], ab[-1])
        return all(self.conv(x, y, depth + 1, ctx) for x, y in zip(aa, ab))

    # ------------------------------------------------------------ typing
    def sort_of(self, ctx, A):
        s = self.whnf(self.infer(ctx, A))
        if s[0] != 's': raise CheckError('not a type: expected a sort')
        return s[1]

    def infer(self, ctx, t):
        self.tick()
        k = t[0]
        if k == 'v':
            if t[1] not in ctx: raise CheckError('unbound variable ' + t[1])
            return ctx[t[1]]
        if k == 'c': return self.g[t[1]].type
        if k == 's': return ('s', t[1] + 1)
        if k == 'app':
            tf = self.whnf(self.infer(ctx, t[1]))
            if tf[0] != 'pi': raise CheckError('applying a term that is not a function')
            self.check(ctx, t[2], tf[2])
            return subst(tf[3], tf[1], t[2])
        if k == 'lam':
            _, x, A, b = t; self.sort_of(ctx, A)
            y = fresh(); ctx2 = dict(ctx); ctx2[y] = A
            return ('pi', y, A, self.infer(ctx2, subst(b, x, ('v', y))))
        if k == 'pi':
            _, x, A, B = t; a = self.sort_of(ctx, A)
            y = fresh(); ctx2 = dict(ctx); ctx2[y] = A
            b = self.sort_of(ctx2, subst(B, x, ('v', y)))
            return ('s', 0 if b == 0 else max(a, b))
        if k == 'match': return self.infer_match(ctx, t)
        if k == 'fix':
            _, d, fty, selfn, body = t; self.sort_of(ctx, fty)
            sx = fresh('self'); ctx2 = dict(ctx); ctx2[sx] = fty; body2 = subst(body, selfn, ('v', sx))
            self.check(ctx2, body2, fty)
            if not self.guard_ok(d, fty, sx, body2): raise CheckError('structural recursion guard failed')
            return fty
        raise Unsupported(k)

    def check(self, ctx, t, expected):
        inferred = self.infer(ctx, t)
        if not self.conv(inferred, expected, 0, ctx):
            raise CheckError('type mismatch' + (' (a depth limit of the conversion check was reached: treated as a rejection)' if self.limit_hit else ''))

    # ------------------------------------------------------------ the Match rule
    def match_index(self, pat, conc, fnames, sigma, refuted):
        if refuted[0]: return
        if pat[0] == 'v' and pat[1] in fnames:
            if pat[1] in sigma: self.match_index(sigma[pat[1]], conc, fnames, sigma, refuted)
            else: sigma[pat[1]] = conc
            return
        if canon(pat) == canon(conc): return
        hp, ap = spine(pat); hc, ac = spine(self.whnf(conc))
        if hp[0] == 'c' and hc[0] == 'c':
            if hp[1] == hc[1]:
                if len(ap) == len(ac):
                    for x, y in zip(ap, ac):
                        self.match_index(x, y, fnames, sigma, refuted)
                        if refuted[0]: return
                return
            gp, gc = self.g[hp[1]], self.g[hc[1]]
            if gp.ctor_family is not None and gp.ctor_family == gc.ctor_family: refuted[0] = True
        # anything else is left alone

    def infer_match(self, ctx, t):
        _, s, motive, brs = t
        ts = self.whnf(self.infer(ctx, s)); h, idx = spine(ts)
        if h[0] != 'c' or not self.is_family(h[1]): raise CheckError('match scrutinee is not an inductive family application')
        fam = h[1]; k = self.arity(fam); ctors = self.g[fam].ctors
        if len(idx) != k: raise CheckError('family not fully applied')
        cur = self.whnf(self.infer(ctx, motive))                                   # the motive must be well typed ...
        for _ in range(k + 1):                                                     # ... have k+1 Pi layers ...
            if cur[0] != 'pi': raise CheckError('motive has the wrong shape')
            cur = self.whnf(subst(cur[3], cur[1], ('v', fresh())))
        if cur[0] != 's': raise CheckError('motive does not return a sort')       # ... and end in a sort
        if self.prop_elim and cur[1] != 0 and self.elim_restricted(fam): raise CheckError('large elimination from a proposition')
        if len(brs) != len(ctors): raise CheckError('wrong number of branches')
        conc = [self.index_nf(a) for a in idx]
        params = self.param_slots(fam) if self.params else [False] * k
        for ci, cg in enumerate(ctors):
            fields, res = self.open_ctor(cg); fnames = {x for x, _ in fields}
            sigma, pvars, refuted = {}, set(), [False]
            for j in range(k):
                if refuted[0]: break
                self.match_index(res[j], conc[j], fnames, sigma, refuted)
                if params[j] and res[j][0] == 'v' and res[j][1] in fnames: pvars.add(res[j][1])
            count = {}
            for r in res:
                if r[0] == 'v' and r[1] in fnames: count[r[1]] = count.get(r[1], 0) + 1
            narrow = {v: val for v, val in sigma.items() if count.get(v, 0) <= 1 or v in pvars}
            def sub_many(term, mp):
                for v, val in mp.items(): term = subst(term, v, val)
                return term
            res2 = [sub_many(r, narrow) for r in res]
            capp = apply_args(('c', cg), [narrow.get(x, ('v', x)) for x, _ in fields])
            body_ty = apply_args(motive, res2 + [capp])
            expected = body_ty
            for x, A in reversed(fields): expected = ('pi', x, sub_many(A, sigma), expected)
            self.check(ctx, brs[ci], expected)
        return apply_args(motive, list(idx) + [s])

    # ------------------------------------------------------------ the structural-recursion guard
    def guard_ok(self, d, fty, sx, body):
        cur = fty
        for _ in range(d):
            if cur[0] != 'pi': return False
            cur = cur[3]
        if cur[0] != 'pi': return False
        dh, _ = spine(cur[2]); fam = dh[1] if dh[0] == 'c' else None
        names, b = [], body
        while b[0] == 'lam': names.append(b[1]); b = b[3]
        if len(names) <= d: return True and sx not in fv(body)
        dec = names[d]

        def walk(t, smaller):
            k = t[0]
            if k == 'v': return t[1] != sx
            if k in ('c', 's'): return True
            if k == 'app':
                h, args = spine(t)
                if h == ('v', sx):
                    if len(args) <= d: return False
                    a = args[d]
                    if not (a[0] == 'v' and a[1] in smaller): return False
                    return all(walk(x, smaller) for x in args)
                return walk(h, smaller) and all(walk(x, smaller) for x in args)
            if k in ('lam', 'pi'): return walk(t[2], smaller) and walk(t[3], smaller)
            if k == 'match':
                ok = walk(t[1], smaller) and walk(t[2], smaller)
                structural = t[1][0] == 'v' and (t[1][1] == dec or t[1][1] in smaller)
                for br in t[3]:
                    if not structural: ok = ok and walk(br, smaller); continue
                    sm, bb = set(smaller), br
                    while bb[0] == 'lam':
                        ok = ok and walk(bb[2], smaller)
                        hh, _ = spine(bb[2])
                        if hh[0] == 'c' and hh[1] == fam: sm.add(bb[1])
                        bb = bb[3]
                    ok = ok and walk(bb, sm)
                return ok
            if k == 'fix': return walk(t[2], smaller) and walk(t[4], smaller)
            return False
        return walk(body, set())

    # ------------------------------------------------------------ entry points
    def run(self, f, *a):
        self.fuel = self.fuel0; self.limit_hit = False
        try: f(*a); return True, 'OK'
        except CheckError as e: return False, str(e)
        except Unsupported as e: return False, 'unsupported node ' + str(e)
        except RecursionError: return False, 'recursion limit (treated as a rejection)'
        except (IndexError, KeyError, TypeError, AttributeError, ValueError) as e: return False, 'internal error on a malformed input (treated as a rejection): %s: %s' % (type(e).__name__, e)

    def check_closed(self, term_tree, type_tree):
        return self.run(lambda: (self.sort_of({}, from_hybrid(type_tree)), self.check({}, from_hybrid(term_tree), from_hybrid(type_tree))))

    def well_founded_problems(self):
        """(gid, name, message) for every way the environment fails to be WELL FOUNDED. Typing each global against the whole environment cannot see this: `f := fun x => g x` and
        `g := fun x => f x` each type-check, and `f 0 : False` follows from f's declared type. So: (1) no global may mention a global that is still being defined, (2) the graph of
        'mentions' (in the type or the value) has no cycle, and (3) the only self-reference is a recursive function's call to itself (whose guard is recursion_problems' business)."""
        bad, deps = [], {}
        for gid, gl in self.g.items():
            d = consts_in(gl.type)
            if gl.value is not None: d |= consts_in(gl.value)
            deps[gid] = d
        for gid, d in deps.items():
            for h in sorted(d):
                if h not in self.g: bad.append((gid, self.g[gid].name, 'mentions the unknown global %d' % h))
                elif self.g[h].reserved and h != gid: bad.append((gid, self.g[gid].name, "mentions global '%s', which is declared but not yet finished" % self.g[h].name))
            if gid in d and self.g[gid].recursion is None: bad.append((gid, self.g[gid].name, 'refers to itself without being a structurally recursive function'))
        # Tarjan's strongly connected components, iteratively; a self-loop is not a cycle here (it was judged above)
        index, low, on, stack, counter, sccs = {}, {}, set(), [], [0], []
        for root in self.g:
            if root in index: continue
            work = [(root, iter(sorted(h for h in deps[root] if h in self.g and h != root)))]
            index[root] = low[root] = counter[0]; counter[0] += 1; stack.append(root); on.add(root)
            while work:
                v, it = work[-1]
                advanced = False
                for w in it:
                    if w not in index:
                        index[w] = low[w] = counter[0]; counter[0] += 1; stack.append(w); on.add(w)
                        work.append((w, iter(sorted(h for h in deps[w] if h in self.g and h != w)))); advanced = True; break
                    elif w in on: low[v] = min(low[v], index[w])
                if advanced: continue
                work.pop()
                if work: low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop(); on.discard(w); comp.append(w)
                        if w == v: break
                    if len(comp) > 1: sccs.append(sorted(comp))
        for comp in sccs: bad.append((comp[0], self.g[comp[0]].name, 'cyclic definitions: ' + ', '.join(self.g[c].name for c in comp)))
        return bad

    def recursion_problems(self):
        """(gid, name, message) for every global the kernel recorded as structurally recursive whose body does not pass this checker's own guard. The body's calls to the global itself are
        judged exactly like the self-calls of a `fix` term: the global becomes a variable, and every call must pass a strict subterm of the decreasing argument."""
        bad = []
        for gid, gl in self.g.items():
            if gl.recursion is None: continue
            fam, d = gl.recursion
            if gl.value is None: bad.append((gid, gl.name, 'recorded as recursive but has no body')); continue
            if fam not in self.g or not self.is_family(fam): bad.append((gid, gl.name, 'recursion over %s, which is not an inductive family' % fam)); continue
            cur = gl.type
            for _ in range(d):
                if cur[0] != 'pi': cur = None; break
                cur = cur[3]
            dh = spine(cur[2])[0] if cur is not None and cur[0] == 'pi' else None
            if dh != ('c', fam): bad.append((gid, gl.name, 'the decreasing argument %d is not of the recorded family' % d)); continue
            sx = fresh('self')
            if not self.guard_ok(d, gl.type, sx, replace_const(gl.value, gid, ('v', sx))): bad.append((gid, gl.name, 'structural recursion guard failed'))
        return bad

    def check_definitions(self):
        """re-check every definition of the environment (value : type) and that every family/constructor/axiom type is a type; the environment must be well founded and its recursive globals guarded"""
        bad = []
        if self.quot_problem is not None:
            gid0 = min(gid for gid, gl in self.g.items() if gl.quot_role is not None)
            bad.append((gid0, self.g[gid0].name, 'quotients: ' + self.quot_problem))
        bad += self.well_founded_problems()
        bad += self.recursion_problems()
        for gid, gl in self.g.items():
            ok, msg = self.run(lambda: (self.sort_of({}, gl.type), None if gl.value is None else self.check({}, gl.value, gl.type)))
            if not ok: bad.append((gid, gl.name, msg))
        return bad
