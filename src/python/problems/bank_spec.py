"""bank_spec.py -- a short way to write the `Spec` of a problem: the intended meaning of a statement as Python, one relation atom at a time.

A statement is a list of hypotheses and a conclusion; the faithfulness check (semantic.py) wants for each of them (1) the predicate, and (2) the numbers its relation atoms are made of -- so that a slip in a constant, a
sign or the direction of an inequality cannot hide behind a truth table.  Writing both by hand says everything twice; here each relation is written ONCE as a `Rel`, and `spec` derives the predicates and the atoms:

    spec(('Z', 'Z'),                                          # the kinds of the variables
         [LT(0, V(0)), LT(0, V(1)), EQ(lambda x, y: x * y + x + y, 71)],     # hypotheses: one relation each
         [EQ(lambda x, y: x * x + y * y, 146)],                # conclusion: a relation, or a list (their conjunction)
         domains=(R, R))

`V(i)` is the i-th variable; a number is a constant; a function takes the values of the variables.  The relations are  EQ  LT  LE  DVD  COP (coprime)  GCD (a, b, d)  PRIME, with the arguments in the order the
formal statement has them (`Z.lt x y` for `x < y`: a statement that says `x > y` is `LT(y, x)`).  A hypothesis or conclusion that is not a single relation (or a conjunction) is passed as a function, with `hatoms=` or
`atoms=` giving the numbers of ALL its relation atoms (see semantic.Spec); `hatoms=False` leaves the numbers of the hypotheses unchecked (a hypothesis with a quantifier inside cannot be covered).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import math
from semantic import Spec, _is_prime

R = range(-8, 9)


def V(i):
    """the i-th variable of the statement (the variables of the Spec in order, then the variables of the quantifiers an atom is under)"""
    return lambda *a: a[i]


def _fn(x): return x if callable(x) else (lambda *a, c=x: c)


def dvd_(a, b): return b == 0 if a == 0 else b % a == 0


def is_prime_(n):
    """primality of an integer (exact; the Miller-Rabin test of semantic.py, so that powers are no problem)"""
    return _is_prime(n)


class Rel:
    """one relation atom with the numbers it is made of: `Rel('eq', l, r)`, ... ; calling it with the values of the variables tells whether it holds"""
    __slots__ = ('kind', 'args')
    ARITY = {'eq': 2, 'lt': 2, 'le': 2, 'dvd': 2, 'coprime': 2, 'gcd': 3, 'prime': 1}

    def __init__(self, kind, *args):
        assert len(args) == self.ARITY[kind], (kind, len(args))
        self.kind = kind; self.args = tuple(_fn(a) for a in args)

    def __call__(self, *v):
        a = [f(*v) for f in self.args]; k = self.kind
        if k == 'eq': return a[0] == a[1]
        if k == 'lt': return a[0] < a[1]
        if k == 'le': return a[0] <= a[1]
        if k == 'dvd': return dvd_(a[0], a[1])
        if k == 'coprime': return math.gcd(a[0], a[1]) == 1
        if k == 'gcd': return a[2] == math.gcd(a[0], a[1])
        return is_prime_(a[0])


def EQ(l, r): return Rel('eq', l, r)
def LT(l, r): return Rel('lt', l, r)
def LE(l, r): return Rel('le', l, r)
def DVD(a, b): return Rel('dvd', a, b)
def COP(a, b): return Rel('coprime', a, b)
def GCD(a, b, d): return Rel('gcd', a, b, d)
def PRIME(p): return Rel('prime', p)


def AND_(*fs):
    """the conjunction of predicates (functions of the variables)"""
    return lambda *a: all(f(*a) for f in fs)


def spec(params, hyps=(), concl=None, domains=None, atoms=None, hatoms=None, **kw):
    """`Spec(params, hyps, concl, ...)` with the predicates and the atoms taken from the `Rel`s: `hyps` a list whose entries are Rels (a hypothesis that is one relation) or functions (then `hatoms` is needed
    for all of the hypotheses); `concl` a Rel, a list of Rels (their conjunction), a function (then `atoms` is needed), or a pair (lhs, rhs) of functions for an equivalence (then `atoms` too).
    Everything else (`skip_mutants`, `points`, `windows`, `body`, ...) is Spec's."""
    hyps = list(hyps)
    if hatoms is None:
        if not all(isinstance(h, Rel) for h in hyps): raise ValueError('a hypothesis that is not a single relation needs hatoms= (False: the numbers of the hypotheses are not checked)')
        hatoms = [h.args for h in hyps]
    elif hatoms is False: hatoms = None
    if isinstance(concl, Rel): parts = [concl]
    elif isinstance(concl, list) and all(isinstance(c, Rel) for c in concl): parts = concl
    else: parts = None
    if parts is not None:
        if atoms is None: atoms = [c.args for c in parts]
        concl = parts[0] if len(parts) == 1 else AND_(*parts)
    elif atoms is None: raise ValueError('a conclusion that is not a conjunction of relations needs atoms=')
    return Spec(tuple(params), tuple(hyps), concl, domains=domains, atoms=atoms, hatoms=hatoms, **kw)
