"""02 -- the integers: `ring`, `linarith` and `nlinarith`, each a kernel-checked certificate.

    python3 examples/02_ring_linarith.py                 (a few seconds)

The integer world (src/python/library/int_lib.py) builds Z as a *quotient* of pairs of binary naturals (N x N, with (a,b) ~ (c,d) iff a+d = c+b), on top of the binary-natural library.  The three decision
procedures are untrusted: each finds a certificate (`ring`: a chain of ring-lemma applications; `linarith`: non-negative multipliers found by a simplex; `nlinarith`: the same after adding products and
squares) and builds an ordinary proof term out of it, which the kernel then checks like any other.  So when one of them is wrong, the worst that happens is a refusal.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import int_lib, recheck, audit
from author import AuthorError

w = int_lib.build_int()
Z, Eq, forall = w['Z'], w.Eq, w.forall_
print('the integer world: %d globals' % w.k.num_globals())

# ring: an identity of commutative-ring arithmetic
w.prove('ex_difference_of_squares', forall([Z, Z], lambda x, y: Eq((x + y) * (x - y), x * x - y * y)), lambda p: (p.intro('x', 'y'), p.ring()))
print('ring:       (x + y)(x - y) = x*x - y*y')

# linarith: linear arithmetic over Z from the hypotheses (strict inequalities are strengthened: x < y means x + 1 <= y)
w.prove('ex_linear', forall([Z, Z, lambda x, y: x + y <= 10, lambda x, y, h: 3 <= x], lambda x, y, h1, h2: y <= 7), lambda p: (p.intro('x', 'y', 'h1', 'h2'), p.linarith()))
print('linarith:   x + y <= 10, 3 <= x  |-  y <= 7')
w.prove('ex_no_integer_between', forall([Z, Z, lambda x, y: x < y, lambda x, y, h: y < x + 1], lambda x, y, h1, h2: w['False']), lambda p: (p.intro('x', 'y', 'h1', 'h2'), p.linarith()))
print('linarith:   x < y < x + 1  |-  False   (no integer lies strictly between consecutive integers)')

# nlinarith: products of hypotheses and squares; hints are proofs of extra facts
w.prove('ex_product_nonneg', forall([Z, Z, lambda a, b: 0 <= a, lambda a, b, h: 0 <= b], lambda a, b, ha, hb: 0 <= a * b), lambda p: (p.intro('a', 'b', 'ha', 'hb'), p.nlinarith()))
print('nlinarith:  0 <= a, 0 <= b  |-  0 <= a*b        (a product of two hypotheses)')


def am_gm(p):
    p.intro('x', 'y')
    p.nlinarith(p.T('x y', lambda x, y: w['Z.sq_nonneg'](x - y)))          # the hint: 0 <= (x - y)^2, a proof built under the goal's hypotheses
w.prove('ex_am_gm', forall([Z, Z], lambda x, y: 2 * x * y <= x * x + y * y), am_gm)
print('nlinarith:  2xy <= x^2 + y^2 with the hint (x - y)^2 >= 0')

# a claim that is false is refused, with the reason
try:
    w.prove('ex_false_claim', forall([Z, Z, lambda x, y: x <= y], lambda x, y, h: y <= x), lambda p: (p.intro('x', 'y', 'h'), p.linarith()))
    raise SystemExit('linarith proved a false claim?!')
except AuthorError as e:
    print('linarith refuses  x <= y |- y <= x :', str(e).splitlines()[0])

count, problems = recheck.recheck_world(w)
print('independent checker: %d globals re-checked, %d problems' % (count, len(problems)))
assert not problems, problems[:3]
axioms, quotient_constants = audit.dependencies_of_global(w.k, w.gid['ex_am_gm'])
print('ex_am_gm rests on: axioms %s; quotient primitives %s' % (sorted(axioms) or 'none', sorted(quotient_constants)))
assert not axioms                       # no axiom at all: Z is built from the kernel's quotient primitives, and funext is a theorem
print('EXAMPLE OK')
