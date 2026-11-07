import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths
import int_lib, recheck, audit
from author import AuthorError

w = int_lib.build_int()                       # the integers: a quotient of pairs of binary naturals
Z, Eq, forall = w['Z'], w.Eq, w.forall_

w.prove('diff_squares', forall([Z, Z], lambda x, y: Eq((x + y) * (x - y), x * x - y * y)),
        lambda p: (p.intro('x', 'y'), p.ring()))

# linarith: non-negative multipliers found by an exact simplex, the sum checked by `ring`
w.prove('bound', forall([Z, Z, lambda x, y: x + y <= 10, lambda x, y, h: 3 <= x], lambda x, y, h1, h2: y <= 7),
        lambda p: (p.intro('x', 'y', 'h1', 'h2'), p.linarith()))

# nlinarith: products and squares, with a hint (a proof of 0 <= (x - y)^2)
def am_gm(p):
    p.intro('x', 'y')
    p.nlinarith(p.T('x y', lambda x, y: w['Z.sq_nonneg'](x - y)))
w.prove('am_gm', forall([Z, Z], lambda x, y: 2 * x * y <= x * x + y * y), am_gm)

try:                                          # a false claim is refused
    w.prove('false_claim', forall([Z, Z, lambda x, y: x <= y], lambda x, y, h: y <= x),
            lambda p: (p.intro('x', 'y', 'h'), p.linarith()))
except AuthorError as e:
    print('linarith refuses:', str(e).splitlines()[0])

n, problems = recheck.recheck_world(w)
print('independent checker: %d globals, problems: %s' % (n, problems))
axioms, quotients = audit.dependencies_of_global(w.k, w.gid['am_gm'])
print('axioms:', sorted(axioms), '| quotient constants:', sorted(quotients))
