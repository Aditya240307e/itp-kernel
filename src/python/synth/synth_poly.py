"""synth_poly.py -- polynomials over the integers, to tell whether a term is written the way a person would write it.

    p = tree_poly(tree)                     # the polynomial of a term of the generator's vocabulary (synth_terms.Vocab.parse: ('v', level) | ('n', value) | (role, child, ..))
    p = ast_poly(ast, depth)                # the polynomial of an arithmetic AST of semantic.py (None if it is not arithmetic)
    reducible(p, occ)                       # does a term with `occ` variable occurrences (and these variables) have a polynomial that is simpler -- the term can be written with fewer variables?

A random term such as `((a + b) - c) - a` is a perfectly good term and nobody would write it: the `a` cancels. The generator draws terms at random and a theorem about such a term is a theorem about
`b - c` in disguise; the kernel is not fooled, but a training set made of such statements teaches the wrong things. The test is by POLYNOMIAL NORMAL FORM (not by pattern): a term is REDUCIBLE when a variable
occurs in it that does not occur in its polynomial (it cancels), or when it has at least two more variable occurrences than its polynomial has. A polynomial is a dict  monomial -> coefficient,
a monomial a sorted tuple of (variable, exponent).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
ARITH = {'Z.add': 2, 'Z.mul': 2, 'Z.sub': 2, 'Z.neg': 1}


def padd(p, q):
    r = dict(p)
    for m, c in q.items():
        v = r.get(m, 0) + c
        if v: r[m] = v
        else: r.pop(m, None)
    return r


def pneg(p): return {m: -c for m, c in p.items()}


def pmul(p, q):
    r = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            d = dict(m1)
            for v, e in m2: d[v] = d.get(v, 0) + e
            m = tuple(sorted(d.items())); c = r.get(m, 0) + c1 * c2
            if c: r[m] = c
            else: r.pop(m, None)
    return r


def const(n): return {(): n} if n else {}


def var(v): return {((v, 1),): 1}


def tree_poly(tree):
    """the polynomial of a parsed term of the generator's vocabulary; None if it contains something else"""
    t = tree[0]
    if t == 'v': return var(tree[1])
    if t == 'n': return const(tree[1])
    kids = [tree_poly(c) for c in tree[1:]]
    if any(k is None for k in kids): return None
    if t == 'add': return padd(kids[0], kids[1])
    if t == 'sub': return padd(kids[0], pneg(kids[1]))
    if t == 'mul': return pmul(kids[0], kids[1])
    if t == 'neg': return pneg(kids[0])
    return None


def tree_vars(tree, acc=None):
    """the variables written in a parsed term, with multiplicity: (set of variables, number of occurrences)"""
    acc = [set(), 0] if acc is None else acc
    if tree[0] == 'v': acc[0].add(tree[1]); acc[1] += 1
    elif tree[0] != 'n':
        for c in tree[1:]: tree_vars(c, acc)
    return acc


def ast_poly(a, depth):
    """the polynomial of an arithmetic AST of semantic.py; a bound variable is named by its absolute level (`depth` binders enclose the term); None if the AST is not a ring expression"""
    t = a[0]
    if t == 'int': return const(a[1])
    if t == 'bv': return var(depth - 1 - a[1])
    if t == 'prim' and a[1] in ARITH:
        kids = [ast_poly(x, depth) for x in a[2]]
        if any(x is None for x in kids): return None
        op = a[1]
        if op == 'Z.add': return padd(kids[0], kids[1])
        if op == 'Z.sub': return padd(kids[0], pneg(kids[1]))
        if op == 'Z.mul': return pmul(kids[0], kids[1])
        return pneg(kids[0])
    return None


def ast_vars(a, depth, acc=None):
    """the same for an AST: (set of variable levels, occurrences); only meaningful for ring expressions"""
    acc = [set(), 0] if acc is None else acc
    t = a[0]
    if t == 'bv': acc[0].add(depth - 1 - a[1]); acc[1] += 1
    elif t == 'prim':
        for x in a[2]: ast_vars(x, depth, acc)
    return acc


def poly_vars(p):
    """(the variables of a polynomial, the number of variable factors it is written with: x*x + y counts three)"""
    vs = set(); occ = 0
    for m in p:
        for v, e in m: vs.add(v); occ += e
    return vs, occ


def reducible(p, vs, occ, slack=2):
    """is a term with variables `vs` and `occ` variable occurrences simpler as the polynomial `p`?  A variable cancels, or the polynomial needs `slack` fewer variable factors"""
    pv, pocc = poly_vars(p)
    if pv != vs: return True
    return occ - pocc >= slack
