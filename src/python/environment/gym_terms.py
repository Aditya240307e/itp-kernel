"""gym_terms.py -- the term pool: small well-typed terms for the actions that carry term arguments (INSTANTIATE, USE).

A learner that has no text interface still has to say WHICH term: which value to put into `forall a b, f (a + b) = ...`, which witness to give to `exists x, ...`. The pool is the answer for the number
types (Nat, N, Z: the types `ring` knows): for a goal, a deterministic, numbered list of terms in the goal's context -- its variables, small numerals (and the numerals that occur in the goal), sums,
differences, products, doubles and squares of those, and the values of its functions on the simplest of them -- so an action names its arguments by position and cannot be ill-typed. Expressions that
are equal as polynomials (`x + y` and `y + x`) are listed once, as the first, simplest spelling.

    pool = build_pool(w, hyp_types, hyp_names, roots)       # roots: the kernel terms whose numerals are worth having (the target and the hypotheses)
    pool.terms      kernel handles, in the order actions refer to them (simplest first)
    pool.types      the type (a kernel handle) of each
    pool.labels     a short readable spelling of each (for tests and logs)
    pool.of_type    {type handle: [indices]}

NOTHING HERE IS TRUSTED. The terms are built from the number type's own operations; whatever an action does with them is done by the tactic layer and checked by the kernel. A term that does not mean
what its label says wastes a probe; it cannot make a false statement provable.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import ring as R
from author import AuthorError

NUMBER_TYPES = ('Z', 'N', 'Nat')       # the order the pool lists them in (integers first: they are the olympiad world)
SMALL = (0, 1, -1, 2, 3, -2)           # numerals every pool has (the negative ones only for rings)
EXTRA = (4, 5, 6, 7, 8, 9, 10)         # further numerals: witnesses, not combined with anything
MAX_VARS = 6                           # variables of one type that are combined (the latest ones)
MAX_OCCURRING = 12                     # numerals of the goal that are offered
APPLY_TO = 16                          # functions are applied to the first this many terms of their domain
MAX_POOL = 200


class Pool:
    __slots__ = ('terms', 'types', 'labels', 'of_type')
    def __init__(self, terms, types, labels):
        self.terms = terms; self.types = types; self.labels = labels; self.of_type = {}
        for i, t in enumerate(types): self.of_type.setdefault(t, []).append(i)
    def __len__(self): return len(self.terms)
    def index(self, term):
        """the position of a term in the pool, or None"""
        try: return self.terms.index(term)
        except ValueError: return None


EMPTY = Pool([], [], [])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ polynomials
# an expression: ('var', level) ('num', c) ('add', a, b) ('sub', a, b) ('mul', a, b) ('neg', a);   a polynomial: {monomial: coefficient}, a monomial a tuple of (variable, exponent) sorted by variable
def _padd(p, q, sign=1):
    r = dict(p)
    for m, c in q.items():
        c2 = r.get(m, 0) + sign * c
        if c2: r[m] = c2
        else: r.pop(m, None)
    return r


def _pmul(p, q):
    r = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            d = dict(m1)
            for v, e in m2: d[v] = d.get(v, 0) + e
            m = tuple(sorted(d.items())); c = r.get(m, 0) + c1 * c2
            if c: r[m] = c
            else: r.pop(m, None)
    return r


def poly_of(e):
    k = e[0]
    if k == 'var': return {((e[1], 1),): 1}
    if k == 'num': return {(): e[1]} if e[1] else {}
    if k == 'add': return _padd(poly_of(e[1]), poly_of(e[2]))
    if k == 'sub': return _padd(poly_of(e[1]), poly_of(e[2]), -1)
    if k == 'neg': return _padd({}, poly_of(e[1]), -1)
    if k == 'mul': return _pmul(poly_of(e[1]), poly_of(e[2]))
    raise ValueError(k)


def poly_key(p): return tuple(sorted(p.items()))


def show_expr(e, names):
    k = e[0]
    if k == 'var': return names[e[1]] if e[1] < len(names) and names[e[1]] else 'x%d' % e[1]
    if k == 'num': return str(e[1])
    if k == 'neg': return '-%s' % show_expr(e[1], names)
    sym = {'add': '+', 'sub': '-', 'mul': '*'}[k]
    return '(%s %s %s)' % (show_expr(e[1], names), sym, show_expr(e[2], names))


def enumerate_exprs(var_levels, numerals, is_ring):
    """the expressions over the given variables and numerals as three groups, each simplest first, equal polynomials once: (the variables and the numerals; the sums and differences of two variables, doubles, and
    a variable plus or minus one; the rest: products, squares, negations, a variable plus or minus a small numeral, multiples)"""
    seen = set(); groups = ([], [], [])
    def add(g, e):
        key = poly_key(poly_of(e))
        if key in seen: return
        seen.add(key); groups[g].append(e)
    v = [('var', l) for l in var_levels]
    for x in v[:4]: add(0, x)                                # a few variables first: the generic instances come before the special ones
    for c in numerals: add(0, ('num', c))
    for x in v[4:]: add(0, x)
    for i, x in enumerate(v):
        for y in v[i + 1:]:
            add(1, ('add', x, y))
            if is_ring: add(1, ('sub', x, y)); add(1, ('sub', y, x))
    for x in v:
        add(1, ('mul', ('num', 2), x)); add(1, ('add', x, ('num', 1)))
        if is_ring: add(1, ('sub', x, ('num', 1)))
    for i, x in enumerate(v):
        add(2, ('mul', x, x))
        for y in v[i + 1:]: add(2, ('mul', x, y))
        if is_ring: add(2, ('neg', x))
    for x in v:
        for c in (2, 3):
            add(2, ('add', x, ('num', c))); add(2, ('mul', ('num', c), x))
            if is_ring: add(2, ('sub', x, ('num', c)))
    return groups


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ building the pool
def numerals_in(k, inst, w, roots, limit=3000):
    """the values of the numerals that occur in the terms `roots` (kernel handles), sorted. A numeral is not looked into."""
    found = set(); seen = set(); stack = list(roots)
    while stack and len(seen) < limit:
        t = stack.pop()
        if t in seen: continue
        seen.add(t)
        v = k.view(t)
        if v[0] in ('app', 'const'):
            val = inst.numeral(w.T(t))
            if val is not None:
                if abs(val) <= 10 ** 6: found.add(val)
                continue
        if v[0] == 'app': stack.append(v[1]); stack.append(v[2])
        elif v[0] in ('lam', 'pi'): stack.append(v[1]); stack.append(v[2])
        elif v[0] == 'let': stack += [v[1], v[2], v[3]]
        elif v[0] == 'match': stack += [v[1], v[2]] + list(v[3])
        elif v[0] == 'fix': stack += [v[1], v[3]]
    return sorted(found)


def _build(w, inst, e):
    """the kernel term of an expression over the number type of `inst`"""
    k = e[0]
    if k == 'var': return w.T(w.k.mk_fvar(e[1]))
    if k == 'num': return inst.lit(e[1])
    a = _build(w, inst, e[1])
    if k == 'neg': return inst._neg(a)
    b = _build(w, inst, e[2])
    if k == 'add': return inst.add(a, b)
    if k == 'mul': return inst.mul(a, b)
    if k == 'sub': return inst._sub(a, b)
    raise ValueError(k)


def build_pool(w, hyp_types, hyp_names, roots, want=()):
    """the pool of a goal with these hypothesis types (kernel handles) and names. `want`: the number types (handles) that the actions in view need terms of, besides those the goal mentions -- a type that
    neither the goal nor `want` mentions gets only the small numerals. Call it outside probes (it makes nodes that must outlive them)."""
    k = w.k; systems = []
    for name in NUMBER_TYPES:
        if name in w.gid:
            A = w[name]; inst = R.instance_for(w, A)
            if inst is not None: systems.append((A, inst))
    if not systems: return EMPTY
    names = list(hyp_names)
    def name_of(l): return names[l] if l < len(names) and names[l] else 'x%d' % l
    # --- per number type: the first two groups of expressions, then (below) the values of functions, then the rest
    base = {}                                      # type handle -> [(term, label)] in pool order, before the function values
    rest = {}                                      # type handle -> [(term, label)] the third group and the extra numerals
    for A, inst in systems:
        is_ring = getattr(inst, 'is_ring', False)
        var_levels = [l for l, t in enumerate(hyp_types) if t == A.n][-MAX_VARS:]
        occurring = [c for c in numerals_in(k, inst, w, list(roots) + list(hyp_types)) if is_ring or c >= 0]
        combine = [c for c in SMALL if is_ring or c >= 0]
        relevant = bool(var_levels) or bool(occurring) or A.n in want
        g0, g1, g2 = enumerate_exprs(var_levels, combine, is_ring)
        if not relevant: g1 = g2 = []
        listed = {poly_key(poly_of(e)) for e in g0 + g1 + g2}
        extra = []
        for c in (occurring[:MAX_OCCURRING] + list(EXTRA)) if relevant else ():
            e = ('num', c); key = poly_key(poly_of(e))
            if key in listed or (c < 0 and not is_ring): continue
            listed.add(key); extra.append(e)
        build = lambda es: [(_build(w, inst, e), show_expr(e, names)) for e in es]
        base[A.n] = build(g0 + g1); rest[A.n] = build(g2 + extra)
    # --- the values of the functions of the context (number type to number type) on the simplest terms of their domain
    apps = {A.n: [] for A, _ in systems}
    for l, ty in enumerate(hyp_types):
        v = k.view(ty)
        if v[0] != 'pi' or v[1] not in base or v[2] not in base or k.loose_range(v[2]) != 0: continue
        for t, lab in base[v[1]][:APPLY_TO]:
            apps[v[2]].append((w.T(k.mk_app(k.mk_fvar(l), t.n)), '%s(%s)' % (name_of(l), lab)))
    terms = []; types = []; labels = []; seen = set()
    for A, _ in systems:
        for t, lab in base[A.n] + apps[A.n] + rest[A.n]:
            if len(terms) >= MAX_POOL: break
            if t.n in seen: continue
            seen.add(t.n); terms.append(t.n); types.append(A.n); labels.append(lab)
    return Pool(terms, types, labels)
