import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths
import itp_env_lnl as E                       # the compiled kernel
import nat_lib, recheck, audit
from author import AuthorError

w = nat_lib.build_nat()                       # a world: a kernel holding the library, every lemma admitted by it
Nat, Zero, Succ, Eq, forall = w['Nat'], w['Zero'], w['Succ'], w.Eq, w.forall_

# a recursive function (the kernel checks that the recursion is structural) ...
w.recursive('twice', w.arrow(Nat, Nat),
            lambda f: w.fun([Nat], lambda n: w.match(n, lambda _: Nat, [Zero, lambda k: Succ(Succ(f(k)))])))
twice = w['twice']

# ... and a theorem about it, proved by a tactic script
def by_induction(p):
    p.intro('n')
    p.induct('n', names=['k', 'ih'])          # two goals: Zero, and Succ k with the hypothesis ih
    p.refl()                                  # twice Zero = Zero + Zero, by computation
    p.rw('ih')
    p.rw('add_succ')                          # n + Succ m = Succ (n + m)
w.prove('twice_eq_add_self', forall([Nat], lambda n: Eq(twice(n), n + n)), by_induction)

# a polynomial identity, closed by `ring`
w.prove('binomial', forall([Nat, Nat], lambda a, b: Eq((a + b) ** 2, a ** 2 + 2 * a * b + b ** 2)),
        lambda p: (p.intro('a', 'b'), p.ring()))

# a false identity is refused, with both normal forms
try:
    w.prove('bad', forall([Nat, Nat], lambda a, b: Eq((a + b) ** 2, a ** 2 + b ** 2)),
            lambda p: (p.intro('a', 'b'), p.ring()))
except AuthorError as e:
    print('ring refuses:', str(e).splitlines()[0])

# the kernel does not trust the tactics: a forged proof of 0 = 1 is refused
try:
    w.k.declare_checked(Eq(Zero, Succ(Zero)).n, w['refl'](Nat, Zero).n, 'forged')
except (E.ITPError, RuntimeError) as e:
    print('kernel refuses:', str(e).splitlines()[0])

print('independent checker: %d globals re-checked, problems: %s' % (recheck.recheck_world(w)[0], recheck.recheck_world(w)[1]))
axioms, quotients = audit.dependencies_of_global(w.k, w.gid['binomial'])      # what the theorem rests on besides the kernel's rules
print('axioms:', sorted(axioms), '| quotient constants:', sorted(quotients))
