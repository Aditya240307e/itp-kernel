"""semantic.py -- a reference semantics for the statements of the problem bank.

A formal statement is a kernel term. Nothing in the kernel says that it MEANS what the problem says: a proof only shows the statement follows from the library's axioms and definitions. A wrong
constant, a `<` for a `<=`, a missing hypothesis gives a theorem that is true and proved and is not the problem. This module is the independent check of that: it READS a statement into a small AST,
EVALUATES it on Python integers with the meaning of the library's symbols written down once (`VALUE_PRIMS`, `PROP_PRIMS`: a few dozen lines to audit) and compares its parts with what the problem is
supposed to say, as Python predicates written by the person who formalised it.

    sem = Semantics(w)                              # w: the world (a library built by lnl_base.thm)
    ast = sem.decode(statement_node)                # the statement as an AST (Unsupported if it uses something the evaluator does not know)
    parts = sem.parts(ast)                          # (params, hyps, conclusion): forall x1 .. xn, H1 -> .. -> Hm -> C   (C may be an Iff: two parts)
    check_spec(sem, ast, spec)                      # the formal parts against the intended ones on a box of small values, and the NEGATIVE CONTROLS: mutated statements must differ

QUANTIFIERS over Z and N are evaluated over WINDOWS (Z: -W..W, N: 0..W). A statement that holds on the window is not thereby true, and one that fails on the window may be true: this is a check of
MEANING on small values, never of truth. The kernel proves; this module tells whether what was proved is what was meant. The meaning of a defined predicate (`Z.is_gcd`, `Z.prime`) is exact, not
windowed (`PROP_PRIMS`), and `audit_prims` compares each against its definition in the library, evaluated over a large window, so that the table cannot drift from the library.
"""
import math, itertools, random, collections
import ring as R


class Unsupported(Exception):
    """the statement uses something outside the fragment the evaluator reads"""


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the meaning of the library's symbols
def _zdvd(a, b): return b == 0 if a == 0 else b % a == 0
def _zgcd(a, b): return math.gcd(a, b)
def _pow(a, n):
    if n < 0 or n > 200 or (abs(a) > 1 and n * max(1, abs(a).bit_length()) > 4000): raise Unsupported('power out of range')
    return a ** n
def _is_prime(p):
    if p < 2: return False
    i = 2
    while i * i <= p:
        if p % i == 0: return False
        i += 1
    return True

# (arity, function): values are Python ints (Z and N alike) or callables
VALUE_PRIMS = {
    'Z.zero': (0, lambda: 0), 'Z.one': (0, lambda: 1), 'N0': (0, lambda: 0), 'XH': (0, lambda: 1), 'Zero': (0, lambda: 0),
    'Z.add': (2, lambda a, b: a + b), 'Z.mul': (2, lambda a, b: a * b), 'Z.neg': (1, lambda a: -a), 'Z.sub': (2, lambda a, b: a - b), 'Z.sqr': (1, lambda a: a * a),
    'Z.pow': (2, _pow), 'Z.of_N': (1, lambda n: n),
    'N.succ': (1, lambda n: n + 1), 'N.add': (2, lambda a, b: a + b), 'N.mul': (2, lambda a, b: a * b), 'N.pow': (2, _pow), 'N.double': (1, lambda a: 2 * a),
    'Npos': (1, lambda p: p), 'XO': (1, lambda p: 2 * p), 'XI': (1, lambda p: 2 * p + 1),
    'Succ': (1, lambda n: n + 1), 'add': (2, lambda a, b: a + b), 'mul': (2, lambda a, b: a * b), 'pow': (2, _pow), 'pred': (1, lambda n: max(n - 1, 0)),
    'Pos.add': (2, lambda a, b: a + b), 'Pos.mul': (2, lambda a, b: a * b), 'Pos.pow': (2, _pow), 'Pos.sqr': (1, lambda a: a * a), 'Pos.succ': (1, lambda a: a + 1), 'Z.ppow': (2, _pow),
    'N.to_nat': (1, lambda n: n), 'N.of_nat': (1, lambda n: n), 'Pos.to_nat': (1, lambda n: n),
}
PROP_PRIMS = {
    'Z.le': (2, lambda a, b: a <= b), 'Z.lt': (2, lambda a, b: a < b), 'N.le': (2, lambda a, b: a <= b), 'N.lt': (2, lambda a, b: a < b), 'Le': (2, lambda a, b: a <= b), 'Lt': (2, lambda a, b: a < b),
    'Z.dvd': (2, _zdvd),
    'Z.coprime': (2, lambda a, b: _zgcd(a, b) == 1),                                          # exists x y, a x + b y = 1
    'Z.is_gcd': (3, lambda a, b, d: d == _zgcd(a, b)),                                         # 0 <= d, d | a, d | b, every common divisor divides d
    'Z.prime': (1, _is_prime),                                                                 # 1 < p and every positive divisor of p is 1 or p
}
LOGIC = {'And': 'and', 'Or': 'or', 'Iff': 'iff', 'Not': 'not', 'True': 'true', 'False': 'false'}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the AST
# ('int', n) | ('bv', i) | ('prim', name, args) | ('call', i, args) | ('eq', a, b) | ('all', kind, body) | ('ex', kind, body) | ('imp', a, b)  [b has a dummy binder: the proof of a]
# | ('and', a, b) | ('or', a, b) | ('iff', a, b) | ('not', a) | ('true',) | ('false',)       kinds: 'Z' 'N' ('fn', n_args)
def show_ast(a, names=()):
    t = a[0]
    if t == 'int': return str(a[1])
    if t == 'bv': return names[-1 - a[1]] if a[1] < len(names) else 'v%d' % a[1]
    if t == 'prim': return '(%s %s)' % (a[1], ' '.join(show_ast(x, names) for x in a[2])) if a[2] else a[1]
    if t == 'call': return '(%s %s)' % (names[-1 - a[1]] if a[1] < len(names) else 'v%d' % a[1], ' '.join(show_ast(x, names) for x in a[2]))
    if t == 'eq': return '(%s = %s)' % (show_ast(a[1], names), show_ast(a[2], names))
    if t in ('all', 'ex'):
        nm = 'x%d' % len(names)
        return '(%s %s : %s, %s)' % ('forall' if t == 'all' else 'exists', nm, a[1], show_ast(a[2], names + (nm,)))
    if t == 'imp': return '(%s -> %s)' % (show_ast(a[1], names), show_ast(a[2], names + ('_',)))
    if t in ('and', 'or', 'iff'): return '(%s %s %s)' % (show_ast(a[1], names), {'and': '/\\', 'or': '\\/', 'iff': '<->'}[t], show_ast(a[2], names))
    if t == 'not': return '~%s' % show_ast(a[1], names)
    return a[0]


class Semantics:
    def __init__(self, w, window_z=8, window_n=12, window_ex=None):
        self.w = w; self.k = w.k; self.wz = window_z; self.wn = window_n; self.wex = window_ex; self.watom = 3          # wex: the window of the witnesses of an existential (default: the window of the universal quantifiers)
        self.zinst = R.instance_for(w, w['Z']) if 'Z' in w.gid else None
        self.ninst = R.instance_for(w, w['N']) if 'N' in w.gid else None
        self.name = {}; self._fns = {}; self._cc = {}
        self.Z = w['Z'].n if 'Z' in w.gid else None; self.N = w['N'].n if 'N' in w.gid else None

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- reading a term
    def _gname(self, gid):
        n = self.name.get(gid)
        if n is None: n = self.name[gid] = self.k.global_info(gid)[0]
        return n

    def _spine(self, n):
        k = self.k; args = []
        while True:
            v = k.view(n)
            if v[0] != 'app': break
            args.append(v[2]); n = v[1]
        args.reverse()
        return n, args

    def _number_kind(self, node):
        """'Z' or 'N' when `node` is one of the number types, else None"""
        v = self.k.view(node)
        if v[0] == 'const':
            nm = self._gname(v[1])
            if nm == 'Z': return 'Z'
            if nm in ('N', 'Nat'): return 'N'
            if nm == 'Pos': return 'P'
        return None

    def _kind(self, dom, stack):
        """the kind of a binder domain: 'Z', 'N', ('fn', n) for a function type between numbers, 'prop' for a proposition (an implication's premise), else Unsupported"""
        k = self.k
        nk = self._number_kind(dom)
        if nk is not None: return nk
        if self._is_prop(dom): return 'prop'
        if k.view(dom)[0] == 'pi':
            n = 0; cur = dom
            while k.view(cur)[0] == 'pi':
                dv = k.view(cur)
                if self._number_kind(dv[1]) is None: raise Unsupported('a function binder with a domain that is not a number type')
                n += 1; cur = dv[2]
            if self._number_kind(cur) is None: raise Unsupported('a function binder with a codomain that is not a number type')
            return ('fn', n)
        raise Unsupported('a binder over %s' % self.w.show(self.w.T(dom)))

    def _is_prop(self, node):
        k = self.k
        while k.view(node)[0] == 'pi': node = k.view(node)[2]
        head, args = self._spine(node); hv = k.view(head)
        if hv[0] == 'const':
            nm = self._gname(hv[1])
            return nm in PROP_PRIMS or nm in LOGIC or nm in ('Eq', 'Ex')
        return False

    def decode(self, node, stack=()):
        """the AST of a statement (or a term), `stack`: the kinds of the enclosing binders, innermost last"""
        k = self.k; v = k.view(node); t = v[0]
        if t == 'bvar': return ('bv', v[1])
        if t == 'pi':
            kind = self._kind(v[1], stack)
            if kind == 'prop': return ('imp', self.decode(v[1], stack), self.decode(v[2], stack + ('prop',)))
            return ('all', kind, self.decode(v[2], stack + (kind,)))
        if t == 'lam': raise Unsupported('a lambda outside an existential')
        if t not in ('const', 'app'): raise Unsupported('a term of kind %s' % t)
        head, args = self._spine(node); hv = k.view(head)
        if hv[0] == 'bvar':
            return ('call', hv[1], tuple(self.decode(a, stack) for a in args))
        if hv[0] != 'const': raise Unsupported('a head of kind %s' % hv[0])
        nm = self._gname(hv[1])
        num = self._numeral(nm, node)
        if num is not None: return ('int', num)
        if nm == 'Eq':
            if len(args) != 3: raise Unsupported('Eq not applied to three arguments')
            return ('eq', self.decode(args[1], stack), self.decode(args[2], stack))
        if nm == 'Ex':
            if len(args) != 2: raise Unsupported('Ex not applied to two arguments')
            kind = self._kind(args[0], stack)
            if kind not in ('Z', 'N', 'P'): raise Unsupported('an existential over %s' % kind)
            lv = k.view(args[1])
            if lv[0] != 'lam': raise Unsupported('an existential whose predicate is not a lambda')
            return ('ex', kind, self.decode(lv[2], stack + (kind,)))
        if nm in LOGIC:
            op = LOGIC[nm]; need = {'and': 2, 'or': 2, 'iff': 2, 'not': 1, 'true': 0, 'false': 0}[op]
            if len(args) != need: raise Unsupported('%s applied to %d arguments' % (nm, len(args)))
            return (op,) + tuple(self.decode(a, stack) for a in args)
        if nm in VALUE_PRIMS or nm in PROP_PRIMS:
            ar = (VALUE_PRIMS.get(nm) or PROP_PRIMS[nm])[0]
            if len(args) != ar: raise Unsupported('%s applied to %d arguments (it takes %d)' % (nm, len(args), ar))
            return ('prim', nm, tuple(self.decode(a, stack) for a in args))
        raise Unsupported('the symbol %s has no meaning in the evaluator' % nm)

    def _numeral(self, nm, node):
        if nm in ('Z.of_N', 'Z.neg', 'Z.zero', 'Z.one') and self.zinst is not None:
            try: return self.zinst.numeral(self.w.T(node))
            except Exception: return None
        if nm in ('Npos', 'N0') and self.ninst is not None:
            try: return self.ninst.numeral(self.w.T(node))
            except Exception: return None
        return None

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- evaluating
    def window(self, kind, witness=False):
        if kind == 'Z':
            r = self.wex if witness and self.wex is not None else self.wz
            return range(-r, r + 1)
        if kind == 'N': return range(0, (self.wex if witness and self.wex is not None else self.wn) + 1)
        if kind == 'P': return range(1, (self.wex if witness and self.wex is not None else self.wn) + 1)
        if isinstance(kind, tuple) and kind[0] == 'fn': return self.functions(kind[1])
        raise Unsupported('no window for %r' % (kind,))

    def atom_window(self, kind):
        """the values the bound variables of an atom take when its numbers are compared"""
        r = self.watom
        if kind == 'Z': return range(-r, r + 1)
        if kind == 'N': return range(0, 2 * r + 1)
        if kind == 'P': return range(1, 2 * r + 1)
        raise Unsupported('no atom window for %r' % (kind,))

    def functions(self, arity):
        """the test functions a quantifier over functions of this arity ranges over (callables)"""
        if arity not in self._fns: self._fns[arity] = [f for _, f in standard_functions(arity)]
        return self._fns[arity]

    def compile(self, a):
        """the AST as a Python closure  env -> value  (env: a tuple of the values of the bound variables, innermost last). Quantifiers range over the windows as they are WHEN THE CLOSURE RUNS."""
        c = self._cc.get(a)
        if c is None: c = self._cc[a] = self._compile(a)
        return c

    def _compile(self, a):
        t = a[0]; comp = self.compile
        if t == 'int':
            n = a[1]; return lambda env: n
        if t == 'bv':
            j = -1 - a[1]; return lambda env: env[j]
        if t == 'true': return lambda env: True
        if t == 'false': return lambda env: False
        if t == 'prim':
            fn = (VALUE_PRIMS.get(a[1]) or PROP_PRIMS[a[1]])[1]; cs = [comp(x) for x in a[2]]
            if len(cs) == 0: return lambda env: fn()
            if len(cs) == 1:
                c0 = cs[0]; return lambda env: fn(c0(env))
            if len(cs) == 2:
                c0, c1 = cs; return lambda env: fn(c0(env), c1(env))
            return lambda env: fn(*[c(env) for c in cs])
        if t == 'call':
            j = -1 - a[1]; cs = [comp(x) for x in a[2]]
            if len(cs) == 1:
                c0 = cs[0]; return lambda env: env[j](c0(env))
            return lambda env: env[j](*[c(env) for c in cs])
        if t == 'eq':
            c0, c1 = comp(a[1]), comp(a[2]); return lambda env: c0(env) == c1(env)
        if t == 'all':
            kind = a[1]; cb = comp(a[2]); window = self.window
            def f_all(env):
                for x in window(kind):
                    if not cb(env + (x,)): return False
                return True
            return f_all
        if t == 'ex':
            kind = a[1]; cb = comp(a[2]); window = self.window
            def f_ex(env):
                for x in window(kind, True):
                    if cb(env + (x,)): return True
                return False
            return f_ex
        if t == 'imp':
            c0, c1 = comp(a[1]), comp(a[2]); return lambda env: (not c0(env)) or c1(env + (None,))
        if t == 'andd':
            c0, c1 = comp(a[1]), comp(a[2]); return lambda env: bool(c0(env)) and bool(c1(env + (None,)))
        if t == 'and':
            c0, c1 = comp(a[1]), comp(a[2]); return lambda env: bool(c0(env)) and bool(c1(env))
        if t == 'or':
            c0, c1 = comp(a[1]), comp(a[2]); return lambda env: bool(c0(env)) or bool(c1(env))
        if t == 'iff':
            c0, c1 = comp(a[1]), comp(a[2]); return lambda env: bool(c0(env)) == bool(c1(env))
        if t == 'not':
            c0 = comp(a[1]); return lambda env: not c0(env)
        raise Unsupported('cannot evaluate %r' % (t,))

    def evaluate(self, a, env=()):
        """the value of an AST in an environment (a tuple, innermost binder last); quantifiers range over the windows"""
        return self.compile(a)(tuple(env))

    def parts(self, a):
        """take a statement apart:  forall x1 .. xn, H1 -> .. -> Hm -> C  ->  (kinds of the x_i, [H_i ASTs], C AST). The hypotheses and C are evaluated in the environment of the x_i
        (the proofs of the hypotheses are dummy binders, so an AST under them is shifted back: no hypothesis name occurs in a conclusion of this shape)."""
        kinds = []
        while a[0] == 'all': kinds.append(a[1]); a = a[2]
        hyps = []
        while a[0] == 'imp':
            hyps.append(a[1]); a = shift(a[2], -1, 0)
            if a is None: raise Unsupported('a conclusion that mentions the proof of a hypothesis')
        return tuple(kinds), hyps, a


def shift(a, d, cut):
    """the AST with every bound variable of index >= cut moved by d (None when a variable that would be removed occurs: d = -1 and an index equal to cut)"""
    t = a[0]
    if t == 'bv':
        if a[1] < cut: return a
        if d < 0 and a[1] + d < cut: return None
        return ('bv', a[1] + d)
    if t == 'int' or t in ('true', 'false'): return a
    if t == 'prim' or t == 'call':
        xs = [shift(x, d, cut) for x in a[2]]
        if any(x is None for x in xs): return None
        if t == 'call':
            i = shift(('bv', a[1]), d, cut)
            if i is None: return None
            return ('call', i[1], tuple(xs))
        return ('prim', a[1], tuple(xs))
    if t in ('all', 'ex'):
        b = shift(a[2], d, cut + 1)
        return None if b is None else (t, a[1], b)
    if t in ('imp', 'andd'):
        x = shift(a[1], d, cut); y = shift(a[2], d, cut + 1)
        return None if x is None or y is None else (t, x, y)
    if t == 'not':
        x = shift(a[1], d, cut); return None if x is None else ('not', x)
    xs = [shift(x, d, cut) for x in a[1:]]
    return None if any(x is None for x in xs) else (t,) + tuple(xs)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ mutants (the negative controls)
SWAP_PRIM = {'Z.le': 'Z.lt', 'Z.lt': 'Z.le', 'N.le': 'N.lt', 'N.lt': 'N.le', 'Z.add': 'Z.sub', 'Z.sub': 'Z.add', 'Z.mul': 'Z.add', 'Le': 'Lt', 'Lt': 'Le', 'N.add': 'N.mul', 'N.mul': 'N.add'}
SWAP_LOGIC = {'and': 'or', 'or': 'and'}


def mutants(a):
    """every statement that differs from `a` by one plausible slip of the formaliser: a constant off by one or negated, a comparison weakened or strengthened, + for -, the arguments of a subtraction
    or a divisibility swapped, a relation negated, a hypothesis dropped, and for or, forall for exists, a negation dropped. Yields (description, AST); the description names the atom the slip is in."""
    yield from _mut(a, (), '')


def _in(desc, ctx): return desc + (' in ' + ctx if ctx else '')


RELATIONS = frozenset(PROP_PRIMS)


def _mut(a, names, ctx):
    t = a[0]
    if t == 'int':
        n = a[1]
        yield _in('constant %d -> %d' % (n, n + 1), ctx), ('int', n + 1)
        yield _in('constant %d -> %d' % (n, n - 1), ctx), ('int', n - 1)
        if n != 0: yield _in('constant %d -> %d' % (n, -n), ctx), ('int', -n)
        return
    if t == 'prim':
        here = show_ast(a, names) if a[1] in RELATIONS else ctx
        if a[1] in SWAP_PRIM: yield _in('%s -> %s' % (a[1], SWAP_PRIM[a[1]]), here), ('prim', SWAP_PRIM[a[1]], a[2])
        if a[1] == 'Z.sub' and len(a[2]) == 2: yield _in('swap the arguments of Z.sub', here), ('prim', 'Z.sub', (a[2][1], a[2][0]))
        if a[1] == 'Z.dvd' and len(a[2]) == 2: yield _in('swap the arguments of Z.dvd', here), ('prim', a[1], (a[2][1], a[2][0]))
        if a[1] in RELATIONS: yield _in('negate %s' % a[1], here), ('not', a)
        for i, x in enumerate(a[2]):
            for d, m in _mut(x, names, here): yield d, ('prim', a[1], a[2][:i] + (m,) + a[2][i + 1:])
        return
    if t == 'call':
        for i, x in enumerate(a[2]):
            for d, m in _mut(x, names, ctx): yield d, ('call', a[1], a[2][:i] + (m,) + a[2][i + 1:])
        return
    if t == 'eq':
        here = show_ast(a, names)
        yield _in('negate an equation', here), ('not', a)
        for d, m in _mut(a[1], names, here): yield d, ('eq', m, a[2])
        for d, m in _mut(a[2], names, here): yield d, ('eq', a[1], m)
        return
    if t in ('all', 'ex'):
        nm = 'x%d' % len(names)
        yield 'quantifier %s -> %s in %s' % (t, 'ex' if t == 'all' else 'all', show_ast(a, names)[:140]), ('ex' if t == 'all' else 'all', a[1], a[2])
        for d, m in _mut(a[2], names + (nm,), ctx): yield d, (t, a[1], m)
        return
    if t == 'imp':
        b = shift(a[2], -1, 0)
        if b is not None: yield 'drop the hypothesis %s' % show_ast(a[1], names), b
        yield 'hypothesis %s -> conjunct' % show_ast(a[1], names), ('andd', a[1], a[2])
        for d, m in _mut(a[1], names, ctx): yield d, ('imp', m, a[2])
        for d, m in _mut(a[2], names + ('_',), ctx): yield d, ('imp', a[1], m)
        return
    if t == 'andd':
        for d, m in _mut(a[1], names, ctx): yield d, ('andd', m, a[2])
        for d, m in _mut(a[2], names + ('_',), ctx): yield d, ('andd', a[1], m)
        return
    if t in ('and', 'or'):
        yield '%s -> %s in %s' % (t, SWAP_LOGIC[t], show_ast(a, names)), (SWAP_LOGIC[t], a[1], a[2])
        yield 'drop the second part of %s' % show_ast(a, names), a[1]
        yield 'drop the first part of %s' % show_ast(a, names), a[2]
        for d, m in _mut(a[1], names, ctx): yield d, (t, m, a[2])
        for d, m in _mut(a[2], names, ctx): yield d, (t, a[1], m)
        return
    if t == 'iff':
        sh = shift(a[2], 1, 0)
        if sh is not None: yield 'iff -> left to right only', ('imp', a[1], sh)
        for d, m in _mut(a[1], names, ctx): yield d, ('iff', m, a[2])
        for d, m in _mut(a[2], names, ctx): yield d, ('iff', a[1], m)
        return
    if t == 'not':
        yield 'drop the negation of %s' % show_ast(a[1], names), a[1]
        for d, m in _mut(a[1], names, ctx): yield d, ('not', m)
        return
    return


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ test functions
def standard_functions(arity=1, seed=7, tables=24):
    """a family of functions Z -> Z (arity 1) or Z x Z -> Z (arity 2) that separates a good many statements about functions: constants, linear and quadratic functions with small
    coefficients, |x|, parity-dependent functions, and pseudo-random tables. [(name, callable)]"""
    out = []
    rng = random.Random(seed + arity)
    if arity == 1:
        for a in range(-3, 4):
            for b in range(-4, 5): out.append(('%d*x%+d' % (a, b), lambda n, a=a, b=b: a * n + b))
        for c in range(-2, 3): out.append(('x^2%+d' % c, lambda n, c=c: n * n + c))
        out.append(('2*x^2', lambda n: 2 * n * n)); out.append(('-x^2', lambda n: -n * n))
        out.append(('|x|', lambda n: abs(n)))
        out.append(('x mod 2', lambda n: n % 2)); out.append(('2*(x mod 2)', lambda n: 2 * (n % 2))); out.append(('x+(x mod 2)', lambda n: n + n % 2))
        out.append(('x if even else 0', lambda n: n if n % 2 == 0 else 0)); out.append(('2x if even else 2x+1', lambda n: 2 * n if n % 2 == 0 else 2 * n + 1))
        out.append(('2x+(x mod 2)', lambda n: 2 * n + (n % 2)))
        out.append(('x^3', lambda n: n ** 3)); out.append(('sign', lambda n: (n > 0) - (n < 0)))
        for i in range(tables):
            tab = {}
            def f(n, tab=tab, rng=rng):
                if n not in tab: tab[n] = rng.randint(-3, 3)
                return tab[n]
            out.append(('table%d' % i, f))
    elif arity == 2:
        for a in range(-2, 3):
            for b in range(-2, 3):
                for c in (-1, 0, 2): out.append(('%d*x%+d*y%+d' % (a, b, c), lambda x, y, a=a, b=b, c=c: a * x + b * y + c))
        out.append(('x*y', lambda x, y: x * y)); out.append(('x^2+y^2', lambda x, y: x * x + y * y)); out.append(('x*y+1', lambda x, y: x * y + 1))
        out.append(('max', max)); out.append(('min', min)); out.append(('|x-y|', lambda x, y: abs(x - y)))
        for i in range(tables):
            tab = {}
            def f(x, y, tab=tab, rng=rng):
                if (x, y) not in tab: tab[(x, y)] = rng.randint(-3, 3)
                return tab[(x, y)]
            out.append(('table%d' % i, f))
    else: raise Unsupported('no test functions of arity %d' % arity)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ checking a statement against its intended meaning
class Spec:
    """what a statement is supposed to say, as Python:  params  the kinds of the leading quantified variables ('Z', 'N', 'P', or 'fn' for a function given by the test functions);
    hyps  one predicate for each hypothesis; concl  the conclusion, or (lhs, rhs) when it is an equivalence; each takes the values of the parameters (and nothing else).
    `windows` overrides the evaluator's windows {'z': .., 'n': .., 'ex': ..} ('ex': the window of the witnesses of existentials); `fns` the test functions for the parameters of kind 'fn' (default: `standard_functions`), `extra_fns` more of them (the functions
    that satisfy the equation of the problem only partly are the ones that tell a quantifier from a slip of it); `points` caps the box;
    `skip_mutants`: {description substring: reason}: mutants whose description contains the substring are known to mean the same thing as the statement (the reason says why); they are not tested.
    `atoms`: the intended NUMBERS of the relation atoms of the conclusion, one tuple of Python functions for each atom, in the order the atoms are written (equations, `<=`, `<`, `Z.dvd`,
    `Z.is_gcd`, .. ; inside a quantifier too, but not inside a quantifier over functions): the two sides of `l = r`, `l <= r`, `l < r` (an equation may be written either way round), `(a, b, d)` for
    `Z.is_gcd a b d`, `(a, b)` for `Z.dvd a b`.  The functions take the parameters of the statement and then the variables of the quantifiers the atom is under, outermost first (hypotheses
    of an implication inside the conclusion are not variables): for  forall n, exists k, n^2 = 4k  the atom is  (lambda n, k: n * n, lambda n, k: 4 * k).  `atom=(f, g, ..)` is `atoms=[(f, g, ..)]`.
    A true statement has a true conclusion wherever it matters, so a truth table cannot tell it from a weaker or different true one (`2ab <= a^2 + b^2` from `ab <= a^2 + b^2`,
    `gcd(21n+4, 14n+3) = 1` from `gcd(20n+4, 14n+3) = 1`, an existential from one with another witness equation): the numbers the atoms are made of can.
    The bound variables take the values of a small window (`windows['atom']`, default 3).
    `body=(kinds, pred)`: for a conclusion that begins with existentials (`exists a, exists b, B(a, b)`): the kinds of the witnesses and the intended meaning of the BODY as a predicate of the parameters and then the
    witnesses (`pred(*params, a, b)`).  A closed existential is true as soon as ONE witness works, so the truth tables above do not see whether a connective or an inequality inside the body is the right way round
    (a body with `or` for `and` has more witnesses, and still has the one that the problem needs); the body is compared at every combination of a point and the witnesses of the window `windows['ex']` (a sample when
    that is a lot), and so are the bodies of the mutants.
    `hatoms`: the same for the relation atoms of the HYPOTHESES (all of them, in the order they are written; no quantifier inside a hypothesis): the numbers of  x^2 + y^2 = 3 z^2  are
    `(lambda x, y, z: x * x + y * y, lambda x, y, z: 3 * z * z)`.  A truth table does not see a slip in a hypothesis that has the same solutions (`-3 z^2` for `3 z^2` over the integers: both have only the
    origin), and a Diophantine hypothesis is the usual case."""
    def __init__(self, params, hyps=(), concl=None, windows=None, fns=None, points=1200, skip_mutants=None, max_mutants=120, note='', atom=None, atoms=None, extra_fns=(), body=None, hatoms=None):
        self.params = tuple(params); self.hyps = tuple(hyps); self.concl = concl; self.windows = windows or {}; self.fns = fns; self.points = points; self.extra_fns = tuple(extra_fns)
        self.hatoms = None if hatoms is None else [tuple(a) for a in hatoms]
        self.skip_mutants = skip_mutants or {}; self.max_mutants = max_mutants; self.note = note
        self.atoms = None if atoms is None and atom is None else ([tuple(atom)] if atom is not None else [tuple(a) for a in atoms])
        self.body = None if body is None else (tuple(body[0]), body[1])


class Report:
    def __init__(self): self.errors = []; self.points = 0; self.nonvacuous = 0; self.mutants = 0; self.caught = 0; self.survivors = []; self.shape_caught = 0; self.equivalent = 0; self.unevaluable = 0
    @property
    def ok(self): return not self.errors
    def __repr__(self): return 'Report(ok=%s points=%d nonvacuous=%d mutants=%d caught=%d equivalent=%d survivors=%d errors=%s)' % (
        self.ok, self.points, self.nonvacuous, self.mutants, self.caught, self.equivalent, len(self.survivors), self.errors[:2])


def _box(sem, kinds, fns, cap, rng, extra=()):
    """the points of the box: every combination of the windows (functions: the test functions, `fns` of the spec or the standard ones) when that is at most `cap`, else a random sample"""
    dom = []
    for kd in kinds:
        if kd in ('Z', 'N', 'P'): dom.append(list(sem.window(kd)))
        else: dom.append((list(fns) if fns is not None else list(sem.functions(kd[1]))) + list(extra))
    total = 1
    for d in dom: total *= len(d)
    if total <= cap: return list(itertools.product(*dom)), total
    seen = set(); out = []
    while len(out) < cap:
        p = tuple(rng.randrange(len(d)) for d in dom)
        if p in seen: continue
        seen.add(p); out.append(tuple(dom[i][j] for i, j in enumerate(p)))
    return out, total


class ShapeChange(Exception):
    """a mutant quantifies a different number of variables than the statement"""


BODY_CAP = 3000


def body_rows(sem, bodykinds, pts, cap=BODY_CAP, seed=777):
    """the rows (a point followed by the values of the witnesses) at which the body of a leading existential is compared: every combination of the points and the witness window when there are at most
    `cap` of them, else a deterministic sample (the same one for the same points)"""
    wins = [list(sem.window(kd, True)) for kd in bodykinds]
    total = len(pts)
    for wv in wins: total *= len(wv)
    if total <= cap: return [tuple(pts[i]) + tuple(c) for i in range(len(pts)) for c in itertools.product(*wins)]
    rng = random.Random(seed); seen = set(); rows = []
    while len(rows) < cap:
        c = (rng.randrange(len(pts)),) + tuple(rng.randrange(len(wv)) for wv in wins)
        if c in seen: continue
        seen.add(c); rows.append(tuple(pts[c[0]]) + tuple(wv[j] for wv, j in zip(wins, c[1:])))
    return rows


def strip_witnesses(concl, bodykinds):
    """the body of a conclusion `exists x1, .., exists xk, B` (the kinds must be the given ones; ShapeChange otherwise)"""
    for kd in bodykinds:
        if concl[0] != 'ex' or concl[1] != kd: raise ShapeChange()
        concl = concl[2]
    return concl


def _canon_atom(kind, vals):
    """the numbers of a hypothesis atom at one point, as they are compared: an equation may be written either way round"""
    return (kind,) + (tuple(sorted(vals)) if kind == 'eq' else tuple(vals))


def _tables(sem, ast, pts, kinds0=None, bodykinds=None, hyp_atoms=False):
    """the values of the parts of the statement at the points: (kinds, {'hyp0': [bool], .., 'concl': [bool] | 'lhs', 'rhs'}, atoms) where atoms = [(kind, [column of each argument])] for the relation
    atoms of the conclusion's quantifier-free skeleton; with `bodykinds` also 'body': the body of the leading existentials at the `body_rows`.  Raises Unsupported, ShapeChange (kinds0 given and different)."""
    kinds, hyps, concl = sem.parts(ast)
    if kinds0 is not None and kinds != kinds0: raise ShapeChange()
    tabs = {}
    for i, h in enumerate(hyps):
        f = sem.compile(h); tabs['hyp%d' % i] = [bool(f(p)) for p in pts]
    if hyp_atoms:
        j = 0
        for h in hyps:
            for kind, args, binders in conclusion_atoms(h):
                if binders: raise Unsupported('a relation under a quantifier inside a hypothesis: `hatoms` does not cover it')
                fs = [sem.compile(x) for x in args]
                tabs['hatom%d' % j] = [_canon_atom(kind, [f(p) for f in fs]) for p in pts]; j += 1
    if concl[0] == 'iff':
        f, g = sem.compile(concl[1]), sem.compile(concl[2])
        tabs['lhs'] = [bool(f(p)) for p in pts]; tabs['rhs'] = [bool(g(p)) for p in pts]
    else:
        f = sem.compile(concl); tabs['concl'] = [bool(f(p)) for p in pts]
    if bodykinds is not None:
        f = sem.compile(strip_witnesses(concl, bodykinds)); tabs['body'] = [bool(f(r)) for r in body_rows(sem, bodykinds, pts)]
    return kinds, tabs, _atom_tables(sem, concl, pts)


def conclusion_atoms(a):
    """the relation atoms of a conclusion, in syntactic order: [(kind, [argument ASTs], binders)], binders: the kinds of the quantifiers the atom is under, outermost first, None for the (dummy)
    binder of the proof of an implication's premise; quantifiers over functions are not entered"""
    out = []
    def go(a, binders):
        t = a[0]
        if t == 'eq': out.append(('eq', [a[1], a[2]], binders))
        elif t == 'prim' and a[1] in RELATIONS: out.append((a[1], list(a[2]), binders))
        elif t == 'not': go(a[1], binders)
        elif t in ('and', 'or', 'iff'): go(a[1], binders); go(a[2], binders)
        elif t in ('all', 'ex'):
            if not isinstance(a[1], tuple): go(a[2], binders + (a[1],))
        elif t in ('imp', 'andd'): go(a[1], binders); go(a[2], binders + (None,))
    go(a, ())
    return out


class AtomTable:
    """the numbers an atom of the conclusion is made of: `cols` one column per argument, over the `rows` (the point and the values of the bound variables, as the spec's functions take them)"""
    __slots__ = ('kind', 'binders', 'cols', 'rows')
    def __init__(self, kind, binders, cols, rows): self.kind = kind; self.binders = binders; self.cols = cols; self.rows = rows


def _atom_tables(sem, concl, pts, cap=2500):
    out = []
    for kind, args, binders in conclusion_atoms(concl):
        real = tuple(b for b in binders if b is not None)
        wins = [list(sem.atom_window(b)) for b in real]
        total = len(pts)
        for wv in wins: total *= len(wv)
        combos = list(itertools.product(range(len(pts)), *wins)) if total <= cap else None
        if combos is None:
            rng = random.Random(12345); seen = set(); combos = []
            while len(combos) < cap:
                c = (rng.randrange(len(pts)),) + tuple(rng.choice(wv) for wv in wins)
                if c in seen: continue
                seen.add(c); combos.append(c)
        rows = []; envs = []
        for c in combos:
            p = pts[c[0]]; ws = c[1:]; it = iter(ws)
            rows.append(tuple(p) + tuple(ws)); envs.append(tuple(p) + tuple(None if b is None else next(it) for b in binders))
        cols = [[f(e) for e in envs] for f in (sem.compile(x) for x in args)]
        out.append(AtomTable(kind, real, cols, rows))
    return out


def _atoms_same(x, y):
    """two lists of atom tables describe the same atoms: the same relations, quantifiers and numbers (an equation may be written either way round)"""
    if len(x) != len(y): return False
    for u, v in zip(x, y):
        if u.kind != v.kind or u.binders != v.binders or len(u.cols) != len(v.cols): return False
        if u.cols != v.cols and not (u.kind == 'eq' and u.cols[0] == v.cols[1] and u.cols[1] == v.cols[0]): return False
    return True


def check_spec(sem, ast, spec, mutate=True, seed=1, strict=True):
    """the formal statement against its intended meaning: every part must agree with the Python predicate at every point of the box; the intended atom (if the spec gives one) must be made of the same
    numbers; the statement must be NON-VACUOUS (some point satisfies all the hypotheses; the sides of an equivalence are each true somewhere and false somewhere); and the NEGATIVE CONTROLS: every
    mutant of the statement (one plausible slip each, see `mutants`) must differ from it in some part or in the numbers of the atom -- unless the spec explains why it does not (`skip_mutants`): with
    `strict` an unexplained survivor is an error, because the box would not notice that slip. Returns a Report."""
    rep = Report(); rng = random.Random(seed)
    saved = (sem.wz, sem.wn, sem.wex, sem.watom)
    if 'atom' in spec.windows: sem.watom = spec.windows['atom']
    if 'z' in spec.windows: sem.wz = spec.windows['z']
    if 'n' in spec.windows: sem.wn = spec.windows['n']
    if 'ex' in spec.windows: sem.wex = spec.windows['ex']
    try:
        try: kinds, hyps, concl = sem.parts(ast)
        except Unsupported as e: rep.errors.append('the statement is outside the fragment: %s' % (e,)); return rep
        if tuple('fn' if isinstance(x, tuple) else x for x in kinds) != tuple(spec.params): rep.errors.append('the statement quantifies %s, the spec says %s' % (kinds, spec.params)); return rep
        if len(hyps) != len(spec.hyps): rep.errors.append('the statement has %d hypotheses, the spec %d' % (len(hyps), len(spec.hyps))); return rep
        if (concl[0] == 'iff') != isinstance(spec.concl, tuple): rep.errors.append('the conclusion is %s, the spec says %s' % ('an equivalence' if concl[0] == 'iff' else 'not an equivalence', 'an equivalence' if isinstance(spec.concl, tuple) else 'a single statement')); return rep
        pts, total = _box(sem, kinds, spec.fns, spec.points, rng, spec.extra_fns)
        rep.points = len(pts)
        bk = None if spec.body is None else spec.body[0]
        ha = spec.hatoms is not None
        try: _, tabs, atoms = _tables(sem, ast, pts, None, bk, ha)
        except Unsupported as e: rep.errors.append('evaluation: %s' % (e,)); return rep
        except ShapeChange: rep.errors.append('the conclusion does not begin with existentials of the kinds %s the spec gives for its body' % (bk,)); return rep
        want = {}; where = {}
        for i, h in enumerate(spec.hyps): want['hyp%d' % i] = [bool(h(*p)) for p in pts]
        if isinstance(spec.concl, tuple): want['lhs'] = [bool(spec.concl[0](*p)) for p in pts]; want['rhs'] = [bool(spec.concl[1](*p)) for p in pts]
        else: want['concl'] = [bool(spec.concl(*p)) for p in pts]
        if bk is not None:
            rows = body_rows(sem, bk, pts); want['body'] = [bool(spec.body[1](*r)) for r in rows]; where['body'] = rows
        for nm, ws in want.items():
            at = where.get(nm, pts)
            bad = [j for j in range(len(at)) if tabs[nm][j] != ws[j]]
            if bad: rep.errors.append('part %s disagrees with the intended meaning at %d of %d points, e.g. at %s (statement says %s)' % (nm, len(bad), len(at), _showpt(at[bad[0]]), tabs[nm][bad[0]]))
        if spec.atoms is not None:
            if len(spec.atoms) != len(atoms): rep.errors.append('the conclusion has %d relation atoms (%s), the spec gives the numbers of %d' % (len(atoms), ', '.join(t.kind for t in atoms), len(spec.atoms)))
            else:
                for i, (at, fs) in enumerate(zip(atoms, spec.atoms)):
                    if len(fs) != len(at.cols): rep.errors.append('atom %d of the conclusion (%s) has %d arguments, the spec gives %d' % (i, at.kind, len(at.cols), len(fs))); continue
                    want_cols = [[f(*row) for row in at.rows] for f in fs]
                    if not (at.cols == want_cols or (at.kind == 'eq' and at.cols == want_cols[::-1])):
                        bad = [j for j in range(len(at.rows)) if tuple(c[j] for c in at.cols) != tuple(c[j] for c in want_cols)]
                        rep.errors.append('the numbers in atom %d of the conclusion (%s) differ from the intended ones at %d of %d rows, e.g. at %s: statement %s, intended %s' % (
                            i, at.kind, len(bad), len(at.rows), _showpt(at.rows[bad[0]]), tuple(c[bad[0]] for c in at.cols), tuple(c[bad[0]] for c in want_cols)))
        if ha:
            names = sorted((nm for nm in tabs if nm.startswith('hatom')), key=lambda nm: int(nm[5:]))
            if len(spec.hatoms) != len(names): rep.errors.append('the hypotheses have %d relation atoms, the spec gives the numbers of %d' % (len(names), len(spec.hatoms)))
            else:
                for i, (nm, fs) in enumerate(zip(names, spec.hatoms)):
                    kind = tabs[nm][0][0]
                    if len(fs) != len(tabs[nm][0]) - 1: rep.errors.append('atom %d of the hypotheses (%s) has %d arguments, the spec gives %d' % (i, kind, len(tabs[nm][0]) - 1, len(fs))); continue
                    want_h = [_canon_atom(kind, [f(*p) for f in fs]) for p in pts]
                    bad = [j for j in range(len(pts)) if tabs[nm][j] != want_h[j]]
                    if bad: rep.errors.append('the numbers in atom %d of the hypotheses (%s) differ from the intended ones at %d of %d points, e.g. at %s: statement %s, intended %s' % (
                        i, kind, len(bad), len(pts), _showpt(pts[bad[0]]), tabs[nm][bad[0]][1:], want_h[bad[0]][1:]))
        hy = [tabs['hyp%d' % i] for i in range(len(hyps))]
        rep.nonvacuous = sum(1 for j in range(len(pts)) if all(h[j] for h in hy))
        if rep.nonvacuous == 0: rep.errors.append('no point satisfies the hypotheses: the problem is vacuous on this box')
        if concl[0] == 'iff':
            for side in ('lhs', 'rhs'):
                if not any(tabs[side]): rep.errors.append('the %s of the equivalence is never true on the box: add test points / functions' % side)
                if all(tabs[side]): rep.errors.append('the %s of the equivalence is always true on the box: add test points / functions' % side)
        if mutate and rep.ok:
            ms = list(mutants(ast)); keep = []
            for d, m in ms:
                if any(s in d for s in spec.skip_mutants): rep.equivalent += 1
                else: keep.append((d, m))
            ms = keep
            rng2 = random.Random(seed + 1)
            if len(ms) > spec.max_mutants: ms = rng2.sample(ms, spec.max_mutants)
            idx = list(range(len(pts))); rng2.shuffle(idx); small = sorted(idx[:250])
            spts = [pts[j] for j in small]
            if bk is not None: _, ref_small, ref_small_at = _tables(sem, ast, spts, None, bk, ha)         # (the sample of the body rows depends on the points)
            else: ref_small = {nm: [tabs[nm][j] for j in small] for nm in tabs}; ref_small_at = _atom_tables(sem, concl, spts) if spec.atoms is not None else None
            for d, m in ms:
                rep.mutants += 1
                differ = None
                for sub in (small, None):                                # a few points first; the whole box only for a mutant that these do not tell apart
                    use = spts if sub is not None else pts
                    try: kd2, tb2, at2 = _tables(sem, m, use, kinds, bk, ha)
                    except ShapeChange: rep.shape_caught += 1; differ = True; break
                    except Unsupported: rep.unevaluable += 1; differ = True; break
                    except (ArithmeticError, TypeError, ValueError, RecursionError, IndexError): rep.unevaluable += 1; differ = True; break
                    if set(tb2) != set(tabs): rep.shape_caught += 1; differ = True; break
                    if sub is not None: ref = ref_small; ref_at = ref_small_at
                    else: ref = tabs; ref_at = atoms
                    if any(tb2[nm] != ref[nm] for nm in tabs): differ = True; break
                    if spec.atoms is not None and not _atoms_same(at2, ref_at): differ = True; break
                if differ: rep.caught += 1
                else: rep.survivors.append(d)
            if rep.survivors and strict:
                rep.errors.append('%d slips of the formalisation go unnoticed by the box (explain each in `skip_mutants` or sharpen the spec): %s' % (len(rep.survivors), '; '.join(rep.survivors[:6])))
    finally: sem.wz, sem.wn, sem.wex, sem.watom = saved
    return rep


def _showpt(p): return '(' + ', '.join(getattr(x, '__name__', None) or str(x) for x in p) + ')'


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ auditing the table
def audit_prims(w, grid=range(-4, 5), big=12):
    """compare the exact meaning of each defined predicate in `PROP_PRIMS` with its DEFINITION in the library, evaluated over a window that contains every relevant witness and divisor of the
    values in `grid`. Returns the list of disagreements ([] when the table is faithful to the definitions)."""
    sem = Semantics(w, window_z=big, window_n=big)
    bad = []; k = w.k
    for nm, arity in (('Z.dvd', 2), ('Z.coprime', 2), ('Z.is_gcd', 3), ('Z.prime', 1)):
        if nm not in w.gid: continue
        body = k.global_info(w.gid[nm])[2]
        if body is None: bad.append((nm, 'has no definition')); continue
        n = 0; cur = body
        while k.view(cur)[0] == 'lam': cur = k.view(cur)[2]; n += 1
        if n != arity: bad.append((nm, 'the definition has %d lambdas, the table %d arguments' % (n, arity))); continue
        try: ast = sem.decode(cur)
        except Unsupported as e: bad.append((nm, 'the definition is outside the fragment: %s' % (e,))); continue
        fn = PROP_PRIMS[nm][1]; f = sem.compile(ast)
        for args in itertools.product(grid, repeat=arity):
            got = bool(f(tuple(args))); want = bool(fn(*args))
            if got != want: bad.append((nm, args, 'definition says %s, table says %s' % (got, want))); break
    return bad


def _has_ex(a):
    if isinstance(a, tuple):
        if a and a[0] == 'ex': return True
        return any(_has_ex(x) for x in a)
    return False


def audit_theorems(w, window_z=6, window_n=8, cap=1500, seed=3, names=None, window_ex=None):
    """THE TABLE AGAINST THE KERNEL. Every theorem of the library inside the evaluator's fragment was proved by the kernel from the library's definitions, so under the table's meaning it must come out
    TRUE at every point of the windows: a point where it is false means the table (or the reader) disagrees with the library. A statement with an existential may fail for want of a witness inside
    the window: those are reported apart (`inconclusive`). Returns {'checked': n, 'unsupported': n, 'failed': [(name, point)], 'inconclusive': [(name, point)], 'points': n}."""
    sem = Semantics(w, window_z=window_z, window_n=window_n, window_ex=window_ex); k = w.k; rng = random.Random(seed)
    out = {'checked': 0, 'unsupported': 0, 'failed': [], 'inconclusive': [], 'points': 0, 'names': []}
    for nm in (names if names is not None else sorted(w.gid)):
        gid = w.gid[nm]
        try:
            ty = k.global_info(gid)[1]
            ast = sem.decode(ty)
            kinds, hyps, concl = sem.parts(ast)
        except (Unsupported, RecursionError): out['unsupported'] += 1; continue
        pts, _ = _box(sem, kinds, None, cap, rng)
        hc = [sem.compile(h) for h in hyps]; cc = sem.compile(concl)
        out['checked'] += 1; out['names'].append(nm)
        for p in pts:
            try:
                ok = (not all(h(p) for h in hc)) or bool(cc(p))
            except Unsupported: ok = True; break
            out['points'] += 1
            if not ok:
                (out['inconclusive'] if _has_ex(ast) else out['failed']).append((nm, tuple(getattr(x, '__name__', x) for x in p)))
                break
    return out
