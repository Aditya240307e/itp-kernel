"""S37 gate: atoms up to the arithmetic in their arguments (ring.py: Ring.canon / canon_arg, shared by `ring`, `linarith`, `linear_combination`).

`ring` treats a term it cannot read as arithmetic -- `f x`, `g a b`, `Z.of_N n` -- as an ATOM. Atoms used to be compared by node identity, so `f (a + b)` and `f (b + a)` were two different unknowns
and `ring` / `linarith` could not see that a hypothesis about one is a hypothesis about the other. Now the arguments of an application that are numbers (of a number system `ring` knows: Nat, N, Z) are
replaced by their simplest spelling (the zero, the numeral of the value, the atom they consist of, or their normal form), with a CONGRUENCE proof (`f_equal`, checked by the kernel like everything else),
so atoms are equal exactly when their arguments are equal as polynomials. Nothing here is trusted: a wrong canonical form can only make a tactic fail or produce a proof the kernel refuses; what this
gate pins down is that the canonical forms are the right ones, that the proofs are what they claim to be, and that what is not safe to change is left alone:

  A. `Ring.canon` on its own: the canonical spelling and the type of the proof (`t = canonical form`); sums in either order, products, differences, numerals in all spellings, naturals under `Z.of_N`,
     two arguments, nested atoms (the inner one first); idempotence (the canonical form of a canonical form is itself, with no proof); atoms that must NOT become equal stay different (`f (a + b)`
     and `f (a - b)`, `g a b` and `g b a`); a term that is no application, an application of a bound variable that is not a function of numbers, a dependent head type -- left alone, no error; a
     context the ring cannot see into -- left alone, no error.
  B. `ring` end to end: the statements are proved and admitted by the kernel; false ones are refused and leave the proof state untouched.
  C. `linarith` / `nlinarith` / `linear_combination` use the same atoms: a contradiction between `f (a + b) <= 3` and `4 <= f (b + a)`, the same with an unrelated `f (a - b)` is refused.
  D. random: `f E1 = f E2` for expressions that are equal as polynomials (rewritten by identities, with nested applications inside) is proved; for expressions that are not, refused.
  E. the independent checker re-checks every theorem proved here.
"""
import os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(200000)
import itp_env_lnl as E
import int_lib, nt_lib, ring as R
from nt_lib import _syms
from author import AuthorError
from tactics import Proof
import indep_checker as ic
from harness import parse_raw
from s30_zlib_tests import Syntax, rand_expr, equiv, zpoly, zclean, terms_bound

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S37_TRACE'))
def trace(msg):
    if TRACE: print('   [%s]' % msg, flush=True)
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    t_start = time.time(); rng = random.Random(37)
    w = int_lib.build_int(); k = w.k; lib_end = k.num_globals()
    s = _syms(w); Z, Eq, F, zzero, zone = s.Z, s.Eq, s.F, s.zzero, s.zone
    N, Nat = w['N'], w['Nat']
    ZZ = w.arrow(Z, Z); ZZZ = w.arrow(Z, w.arrow(Z, Z)); NZ = w.arrow(N, Z)
    zof, nadd, nmul = w['Z.of_N'], w['N.add'], w['N.mul']
    inst = R.instance_for(w, Z); N0 = w['N0']
    zle = w['Z.le']
    proved = []

    def prove_ok(name, stmt, script):
        """admit a theorem through the kernel's checked path"""
        try: w.prove(name, stmt, script)
        except AuthorError as e: return expect(False, '%s: %s' % (name, str(e).split('\n')[0][:140]))
        proved.append(name); return True

    def refused(what, stmt, script):
        """the script must fail (or leave the proof unfinished), and a failing tactic must leave the proof state as it was"""
        p = Proof(w, stmt, 'refused')
        try: script(p)
        except AuthorError: return expect(True, what)
        return expect(False, what + ' (nothing was refused)' if p.done else what + ' (the script ended with goals open)')

    def in_context(stmt, intro_names, fn):
        """run fn(ring, vars) inside the goal that opens `stmt` (the variables are the context variables the names introduce)"""
        p = Proof(w, stmt, 'ctx'); p.intro(*intro_names)
        with p.focus():
            vs = [w.T(k.mk_fvar(i)) for i in range(len(intro_names))]
            return fn(R.Ring(w, inst), vs)

    # ------------------------------------------------------------------------------------------------------------------ A. the canonical forms
    with w.nota.using(Z):
        def check_canon(rg, t, expected=None, same=None, differs=None, what=''):
            ct, pc = rg.canon(t)
            if expected is not None: expect(ct == expected, 'A  %s: the canonical form of %s is %s (got %s)' % (what, w.show(t), w.show(expected), w.show(ct)))
            if pc is not None:
                try: w.check(pc.proof, w.Eq(t, ct)); expect(True, 'ok')
                except AuthorError as e: expect(False, 'A  %s: the proof is not a proof of %s = %s: %s' % (what, w.show(t), w.show(ct), str(e)[:80]))
                expect(pc.lhs == t and pc.rhs == ct, 'A  %s: the proof records its sides' % what)
            else: expect(ct == t, 'A  %s: no proof means no change' % what)
            ct2, pc2 = rg.canon(ct); expect(ct2 == ct and pc2 is None, 'A  %s: the canonical form is canonical' % what)
            return ct

        def a_tests(rg, vs):
            f, g, a, b, c = vs
            # one argument
            check_canon(rg, f(zzero + b), f(b), what='0 + b')
            check_canon(rg, f(b + zzero), f(b), what='b + 0')
            check_canon(rg, f(b * 1), f(b), what='b * 1')
            check_canon(rg, f(a - a + b), f(b), what='a - a + b')
            check_canon(rg, f(b), f(b), what='already simple')
            c1 = check_canon(rg, f(a + b), what='a + b'); c2 = check_canon(rg, f(b + a), what='b + a')
            expect(c1 == c2, 'A  f (a + b) and f (b + a) have the same canonical form')
            c3 = check_canon(rg, f(a * b + 1), what='a * b + 1'); c4 = check_canon(rg, f(1 + b * a), what='1 + b * a')
            expect(c3 == c4, 'A  f (a * b + 1) and f (1 + b * a) have the same canonical form')
            c5 = check_canon(rg, f(a - b), what='a - b'); c6 = check_canon(rg, f(a + b), what='a + b')
            expect(c5 != c6, 'A  f (a - b) and f (a + b) stay different')
            expect(check_canon(rg, f(a + 1), what='a + 1') != check_canon(rg, f(a), what='a'), 'A  f (a + 1) and f a stay different')
            expect(check_canon(rg, f(a * b), what='a * b') != check_canon(rg, f(a + b), what='a + b'), 'A  f (a * b) and f (a + b) stay different')
            # numerals in all their spellings
            n3 = check_canon(rg, f(zone + zone + zone), what='1 + 1 + 1')
            expect(n3 == check_canon(rg, f(inst.lit(3)), what='3') == check_canon(rg, f(zof(inst.N.lit(3))), what='Z.of_N 3'), 'A  f (1 + 1 + 1), f 3 and f (Z.of_N 3) are one atom')
            expect(check_canon(rg, f(a * 0 + 3)) == check_canon(rg, f(inst.lit(3))), 'A  f (a * 0 + 3) is f 3')
            expect(check_canon(rg, f(a * 0)) == check_canon(rg, f(zzero)), 'A  f (a * 0) is f 0')
            # nested: the inner application first
            check_canon(rg, f(f(a + zzero) + zzero), f(f(a)), what='f (f (a + 0) + 0)')
            expect(check_canon(rg, f(f(a + b) + c)) == check_canon(rg, f(f(b + a) + c)), 'A  nested atoms: f (f (a + b) + c) is f (f (b + a) + c)')
            expect(check_canon(rg, f(f(a + b) + c)) == check_canon(rg, f(c + f(b + a))), 'A  nested atoms: f (f (a + b) + c) is f (c + f (b + a))')
        # the context: f g a b c with g a two-argument function is not what introduces here -- two statements, one for the one-argument functions and one for the two-argument ones
        in_context(F([ZZ, ZZ, Z, Z, Z], lambda f, g, a, b, c: Eq(a, a)), ('f', 'g', 'a', 'b', 'c'), a_tests)

        def a2_tests(rg, vs):
            g, a, b, c = vs
            check_canon(rg, g(a + zzero, zzero + b), g(a, b), what='two arguments, both simplified')
            expect(check_canon(rg, g(a + b, c)) == check_canon(rg, g(b + a, c)), 'A  g (a + b) c is g (b + a) c')
            expect(check_canon(rg, g(c, a + b)) == check_canon(rg, g(c, b + a)), 'A  g c (a + b) is g c (b + a)')
            expect(check_canon(rg, g(a, b)) != check_canon(rg, g(b, a)), 'A  g a b and g b a stay different')
            expect(check_canon(rg, g(a + b, a)) != check_canon(rg, g(a, a + b)), 'A  g (a + b) a and g a (a + b) stay different')
            expect(check_canon(rg, g(a, b + zzero)) == g(a, b), 'A  only the second argument changes: g a (b + 0) is g a b')
            expect(check_canon(rg, g(zzero + a, b)) == g(a, b), 'A  only the first argument changes: g (0 + a) b is g a b')
        in_context(F([ZZZ, Z, Z, Z], lambda g, a, b, c: Eq(a, a)), ('g', 'a', 'b', 'c'), a2_tests)

        def a3_tests(rg, vs):
            h, n, m, a = vs
            # naturals under a function from N to Z, and under Z.of_N (a constant function of one natural argument)
            expect(check_canon(rg, h(nadd(n, m))) == check_canon(rg, h(nadd(m, n))), 'A  h (n + m) is h (m + n) for h : N -> Z')
            check_canon(rg, h(nadd(n, N0)), h(n), what='h (n + 0)')
            check_canon(rg, zof(nadd(n, N0)), zof(n), what='Z.of_N (n + 0)')
            expect(check_canon(rg, zof(nadd(n, m))) == check_canon(rg, zof(nadd(m, n))), 'A  Z.of_N (n + m) is Z.of_N (m + n)')
            expect(check_canon(rg, zof(nmul(n, m))) == check_canon(rg, zof(nmul(m, n))), 'A  Z.of_N (n * m) is Z.of_N (m * n)')
            expect(check_canon(rg, zof(nmul(n, m))) != check_canon(rg, zof(nadd(n, m))), 'A  Z.of_N (n * m) and Z.of_N (n + m) stay different')
            # a variable is no application: left alone
            for t in (a, n):
                ct, pc = rg.canon(t); expect(pc is None and ct == t, 'A  a variable is its own canonical form')
        in_context(F([NZ, N, N, Z], lambda h, n, m, a: Eq(a, a)), ('h', 'n', 'm', 'a'), a3_tests)

        # a head whose type is dependent on the argument that would change: left alone (f_equal would be ill-typed there)
        FT = w.forall_([Z], lambda x: w.arrow(Eq(x, x), Z))
        def a4_tests(rg, vs):
            Fd, b = vs
            t = Fd(zzero + b, w.app(w['refl'], Z, zzero + b))
            ct, pc = rg.canon(t)
            expect(pc is None and ct == t, 'A  an application whose function type depends on the argument is left alone')
        in_context(F([FT, Z], lambda Fd, b: Eq(b, b)), ('Fd', 'b'), a4_tests)

        # a context the ring cannot see into: no error, no change
        p = Proof(w, F([ZZ, Z], lambda f, b: Eq(b, b)), 'blind'); p.intro('f', 'b')
        with p.focus():
            f, b = [w.T(k.mk_fvar(i)) for i in range(2)]; t = f(zzero + b)
        rg0 = R.Ring(w, inst)
        try: ct, pc = rg0.canon(t); expect(pc is None and ct == t, 'A  outside the goal that has the variables, a term is left alone')
        except Exception as e: expect(False, 'A  canon outside the context raised %s: %s' % (type(e).__name__, str(e)[:100]))

    trace('A done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ B. ring
    with w.nota.using(Z):
        S = {}
        S['one_arg_zero'] = F([ZZ, Z], lambda f, b: Eq(f(zzero + b), f(b)))
        S['one_arg_comm'] = F([ZZ, Z, Z], lambda f, a, b: Eq(f(a + b), f(b + a)))
        S['one_arg_prod'] = F([ZZ, Z, Z], lambda f, a, b: Eq(f(a * b + 1), f(1 + b * a)))
        S['one_arg_minus'] = F([ZZ, Z], lambda f, a: Eq(f(f(a + 1 - 1)), f(f(a))))
        S['one_arg_numeral'] = F([ZZ, Z], lambda f, a: Eq(f(a * 0 + 3), f(1 + 2)))
        S['two_args'] = F([ZZZ, Z, Z], lambda g, a, b: Eq(g(a + zzero, zzero + b), g(a, b)))
        S['two_args_comm'] = F([ZZZ, Z, Z], lambda g, a, b: Eq(g(a + b, b * a), g(b + a, a * b)))
        S['sum_of_atoms'] = F([ZZ, Z, Z], lambda f, a, b: Eq(f(a + b) + f(b + a), 2 * f(a + b)))
        S['nested'] = F([ZZ, Z, Z], lambda f, a, b: Eq(f(f(a + b) + a), f(a + f(b + a))))
        S['of_N'] = F([N, N], lambda n, m: Eq(zof(nadd(n, m)), zof(nadd(m, n))))
        S['of_N_sum'] = F([N, N], lambda n, m: Eq(zof(nadd(n, m)) + zof(nadd(m, n)), 2 * zof(nadd(n, m))))
        S['in_hypothesis'] = F([ZZ, Z, Z, lambda f, a, b: Eq(f(a + b), 5)], lambda f, a, b, h: Eq(f(b + a) + 1, 6))
        N_VARS = {'one_arg_zero': 2, 'one_arg_comm': 3, 'one_arg_prod': 3, 'one_arg_minus': 2, 'one_arg_numeral': 2, 'two_args': 3, 'two_args_comm': 3, 'sum_of_atoms': 3,
                  'nested': 3, 'of_N': 2, 'of_N_sum': 2, 'in_hypothesis': 4}
        for nm, st in S.items():
            def run(p, nm=nm):
                p.intro(*['v%d' % i for i in range(N_VARS[nm])])
                if nm == 'in_hypothesis': p.linarith()
                else: p.ring()
            prove_ok('s37_' + nm, st, run)
        # false ones
        R_ = {}
        R_['different arguments'] = (F([ZZ, Z, Z], lambda f, a, b: Eq(f(a), f(b))), 3)
        R_['plus one'] = (F([ZZ, Z], lambda f, a: Eq(f(a + 1), f(a))), 2)
        R_['sum against difference'] = (F([ZZ, Z, Z], lambda f, a, b: Eq(f(a + b), f(a - b))), 3)
        R_['swapped arguments'] = (F([ZZZ, Z, Z], lambda g, a, b: Eq(g(a, b), g(b, a))), 3)
        R_['additive function'] = (F([ZZ, Z, Z], lambda f, a, b: Eq(f(a + b), f(a) + f(b))), 3)
        R_['product against sum'] = (F([N, N], lambda n, m: Eq(zof(nmul(n, m)), zof(nadd(n, m)))), 2)
        for what, (st, nv) in R_.items():
            refused('B  ring refuses: %s' % what, st, lambda p, nv=nv: (p.intro(*['v%d' % i for i in range(nv)]), p.ring()))
        # a refused tactic leaves the state as it was
        p = Proof(w, F([ZZ, Z], lambda f, a: Eq(f(a + 1), f(a))), 'untouched'); p.intro('f', 'a'); before = str(p)
        try: p.ring()
        except AuthorError: pass
        expect(str(p) == before and not p.done, 'B  a refused ring leaves the proof state untouched')
    trace('B done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ C. linarith, nlinarith, linear_combination
    with w.nota.using(Z):
        def intro_run(nv, tac):
            def run(p):
                p.intro(*['v%d' % i for i in range(nv)]); tac(p)
            return run
        prove_ok('s37_la_contra', F([ZZ, Z, Z, lambda f, a, b: zle(f(a + b), 3), lambda f, a, b, h1: zle(4, f(b + a))], lambda f, a, b, h1, h2: w['False']),
                 intro_run(5, lambda p: p.linarith()))
        prove_ok('s37_la_zero', F([ZZ, Z, lambda f, b: zle(f(zzero + b), 0), lambda f, b, h: zle(1, f(b))], lambda f, b, h1, h2: w['False']),
                 intro_run(4, lambda p: p.linarith()))
        prove_ok('s37_la_consequence', F([ZZ, Z, Z, lambda f, a, b: zle(f(a + b), 3)], lambda f, a, b, h: zle(f(b + a) + 1, 4)),
                 intro_run(4, lambda p: p.linarith()))
        prove_ok('s37_nla', F([ZZ, Z, Z, lambda f, a, b: zle(0, f(a + b)), lambda f, a, b, h: zle(0, f(b + a))], lambda f, a, b, h1, h2: zle(0, f(a + b) * f(b + a))),
                 intro_run(5, lambda p: p.nlinarith()))
        prove_ok('s37_lincomb', F([ZZ, Z, Z, lambda f, a, b: Eq(f(a + b), 3)], lambda f, a, b, h: Eq(f(b + a), 3)),
                 intro_run(4, lambda p: p.linear_combination((1, 'v3'))))
        refused('C  linarith keeps f (a + b) and f (a - b) apart', F([ZZ, Z, Z, lambda f, a, b: zle(f(a + b), 3), lambda f, a, b, h1: zle(4, f(a - b))], lambda f, a, b, h1, h2: w['False']),
                intro_run(5, lambda p: p.linarith()))
        refused('C  linarith: f (a + 1) is not f a', F([ZZ, Z, lambda f, a: zle(f(a), 3), lambda f, a, h1: zle(4, f(a + 1))], lambda f, a, h1, h2: w['False']),
                intro_run(4, lambda p: p.linarith()))
    trace('C done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ D. random arguments
    SY = Syntax(w, None)
    n_eq = n_ne = 0; tries = 0
    while n_eq < 24 and tries < 400:
        tries += 1
        e1 = rand_expr(rng, 2, 3)
        if terms_bound(e1) > 12: continue
        e2 = equiv(e1, rng)
        if terms_bound(e2) > 24: continue
        if zclean(zpoly(e1, 2)) != zclean(zpoly(e2, 2)): continue                  # (the rewriter keeps the value; this is a guard against a bug in the helper, not in the code under test)
        def stmt_of(ea, eb):
            # f, g are function variables; the expressions are over x, y with g applied to them ('app' nodes)
            return F([ZZ, ZZ, Z, Z], lambda f, g, x, y: Eq(f(SY.build(ea, [x, y], g)), f(SY.build(eb, [x, y], g))))
        run = lambda p: (p.intro('f', 'g', 'x', 'y'), p.ring())
        if prove_ok('s37_rand_eq_%d' % n_eq, stmt_of(e1, e2), run): n_eq += 1
        # a different expression: the statement is false for an injective f, so ring must refuse
        for _ in range(20):
            e3 = rand_expr(rng, 2, 3)
            if terms_bound(e3) <= 12 and zclean(zpoly(e3, 2)) != zclean(zpoly(e1, 2)): break
        else: continue
        if refused('D  ring refuses f E1 = f E3 for E1, E3 that differ as polynomials', stmt_of(e1, e3), run): n_ne += 1
    expect(n_eq >= 20 and n_ne >= 20, 'D  enough random cases (%d equal, %d different)' % (n_eq, n_ne))
    trace('D done at %.1f s' % (time.time() - t_start))

    # ------------------------------------------------------------------------------------------------------------------ E. the independent checker
    t2 = time.time()
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid), recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    chk = ic.Checker(env, params=True, fuel=200_000_000, prop_elim=True)
    n_checked = 0; bad = []
    for nm in proved:
        gid = w.gid[nm]; gl = chk.g[gid]
        ok, msg = chk.run(lambda: (chk.sort_of({}, gl.type), chk.check({}, gl.value, gl.type)))
        n_checked += 1
        if not ok: bad.append((nm, msg))
    expect(n_checked == len(proved) and not bad, 'E  the independent checker re-checks the %d theorems proved here without a rejection: %s' % (len(proved), bad[:3]))
    print("   %d checks; %d theorems proved and admitted (%d random equal, %d random different refused); independent re-check %.1f s (%.1f s total)" %
          (CHECKS[0], len(proved), n_eq, n_ne, time.time() - t2, time.time() - t_start))
    for f in FAIL[:40]: print("   FAIL " + f)
    if len(FAIL) > 40: print("   ... and %d more" % (len(FAIL) - 40))
    print("S37 ATOMS GATE: " + ("PASS" if not FAIL else "FAIL"))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
