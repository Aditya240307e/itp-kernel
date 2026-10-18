"""01 -- prove theorems with the tactic layer.  The tactics propose; the kernel and a second, independent checker decide.

    python3 examples/01_prove_with_tactics.py            (a few seconds; the hybrid kernel must be built first: python3 src/python/build_kernel.py)

The natural-number library (src/python/library/nat_lib.py) is a *world*: a kernel that holds inductive types, recursive functions and lemmas, each of them proved by a tactic script.  Here we add to it

  1. a function defined by recursion, and a theorem about it proved by induction;
  2. a polynomial identity closed by `ring` (which assembles a proof out of the library's ring lemmas);
  3. a false identity, which `ring` refuses -- showing both normal forms, so that you can see why;
  4. a forged proof, which the kernel refuses: whatever the tactic layer does, the kernel re-checks the finished term.

Then the independent checker (src/python/checker/indep_checker.py, written separately from the kernel) re-checks every global of the world, and `audit` lists what a theorem rests on.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import itp_env_lnl as E                  # the compiled hybrid kernel
import nat_lib, recheck, audit
from author import AuthorError

w = nat_lib.build_nat()
Nat, Zero, Succ, Eq, forall = w['Nat'], w['Zero'], w['Succ'], w.Eq, w.forall_
print('the natural-number world: %d globals (types, functions, lemmas), all admitted by the kernel' % w.k.num_globals())

# ---------------------------------------------------------------------------------------------------------------------------------------- 1. a recursive function, and induction
# ex_double (Succ k) = Succ (Succ (ex_double k)):  the body is a `match`, and the kernel checks that the recursion is structural
w.recursive('ex_double', w.arrow(Nat, Nat), lambda f: w.fun([Nat], lambda n: w.match(n, lambda _: Nat, [Zero, lambda k: Succ(Succ(f(k)))])))
double = w['ex_double']


def by_induction(p):
    p.intro('n')
    p.induct('n', names=['k', 'ih'])          # two goals: n = Zero, and n = Succ k with the hypothesis ih
    p.refl()                                  # ex_double Zero = Zero + Zero  holds by computation
    p.rw('ih')                                # Succ (Succ (ex_double k)) = Succ k + Succ k  -->  use the hypothesis
    p.rw('add_succ')                          #   ... and the library's  n + Succ m = Succ (n + m)


w.prove('ex_double_eq_add_self', forall([Nat], lambda n: Eq(double(n), n + n)), by_induction)
print('1. proved by induction:      ', w.show(w.global_type('ex_double_eq_add_self')))

# ---------------------------------------------------------------------------------------------------------------------------------------- 2. ring
w.prove('ex_binomial', forall([Nat, Nat], lambda a, b: Eq((a + b) ** 2, a ** 2 + 2 * a * b + b ** 2)), lambda p: (p.intro('a', 'b'), p.ring()))
print('2. proved by ring:           (a + b)^2 = a^2 + 2*a*b + b^2   (numerals are Succ chains in this world: the statement is %d characters long)' % len(w.show(w.global_type('ex_binomial'))))

# ---------------------------------------------------------------------------------------------------------------------------------------- 3. a false identity is refused
try:
    w.prove('ex_not_binomial', forall([Nat, Nat], lambda a, b: Eq((a + b) ** 2, a ** 2 + b ** 2)), lambda p: (p.intro('a', 'b'), p.ring()))
    raise SystemExit('ring proved a false identity?!')
except AuthorError as e:
    lines = str(e).splitlines()
    print('3. ring refuses (a+b)^2 = a^2 + b^2:')
    for line in lines[:3]: print('      ' + line)

# ---------------------------------------------------------------------------------------------------------------------------------------- 4. the kernel does not trust the tactics
claim = Eq(Zero, Succ(Zero))                  # 0 = 1
forged = w['refl'](Nat, Zero)                 # a proof of 0 = 0, offered as a proof of 0 = 1
try:
    w.k.declare_checked(claim.n, forged.n, 'ex_forged')
    raise SystemExit('the kernel accepted a forged proof?!')
except (E.ITPError, RuntimeError) as e:
    print('4. the kernel refuses a forged proof of 0 = 1:', str(e).splitlines()[0])

# ---------------------------------------------------------------------------------------------------------------------------------------- the second opinion, and what a theorem rests on
count, problems = recheck.recheck_world(w)
print('independent checker: %d globals re-checked, %d problems' % (count, len(problems)))
assert not problems, problems[:3]
axioms, quotient_constants = audit.dependencies_of_global(w.k, w.gid['ex_binomial'])
print('ex_binomial rests on: axioms %s, quotient primitives %s' % (sorted(axioms) or 'none', sorted(quotient_constants) or 'none'))
assert not axioms and not quotient_constants
print('EXAMPLE OK')
