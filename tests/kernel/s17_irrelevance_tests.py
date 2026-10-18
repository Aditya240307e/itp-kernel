"""S17 gate: definitional proof irrelevance.

In the Prop/Type world (option prop_elim) two terms that are proofs of one proposition are DEFINITIONALLY equal:
        G |- p : P    G |- q : P    G |- P : Prop   ==>   p == q .
Nothing else is irrelevant. The kernel decides it in its unifier (structural comparison first; on failure both sides must infer to types that are propositions, and the two
types must unify); the independent checker decides it by inference too, with a conversion that carries the typing context. What must hold, and where it is tested:
  A  a table of terms, each judged by the KERNEL and by the INDEPENDENT checker, which must both give the expected verdict. Accepted: equality of subtype values with different proofs,
     a function applied to different proofs, a TYPE that depends on a proof, proofs under binders, opaque proofs (axioms), a constructor proof with different data in it (Ex), a stuck match
     proof, library proofs of different shapes, nested irrelevance (a proof of an equation between terms that themselves contain proofs), Quot.lift with different respects-proofs.
     Rejected: data, functions, propositions themselves, data hidden next to a proof, and the same subtype statement in a world WITHOUT prop_elim (no Prop there, no irrelevance)
  B  the unifier on ill-typed comparisons (outside the contract of conversion, but the answer must still be safe): proofs of DIFFERENT propositions, two elements of a Type1, two propositions
  C  the tactic layer: `apply refl` closes an equation between terms that differ only in proofs (the unifier runs with metavariables), and the extracted proof is kernel-verified
  D  trusted base: the accepted proofs use no axiom beyond the opaque proofs the test itself declared
  E  the random soundness canary on false statements in this world, with a power run on true ones"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, time
sys.setrecursionlimit(30000)
import harness as H
from harness import *
from s3_tests import V
import prop_world as pw
import s7_param_tests as t7
import s8_indep_tests as t8
import indep_checker as ic
import canary_lib as canlib
import audit
from named import *

T0, T1 = pw.T0, pw.T1

SUB_SORT = ('pi', T1, ('pi', ('pi', V(0), T0), T1))
SUB_CTOR = ('mk', [T1, ('pi', ('var', 0), T0), ('var', 0), ('app', ('var', 1), ('var', 2))], [('var', 0), ('var', 1)])     # mk : (A : Type1) (p : A -> Prop) (a : A) (h : p a) -> Sub A p

# a PROPOSITION with two data fields and a proof:  W A B P : Prop ,  w : (A B : Type1) (P : Prop) (a : A) (b : B) (h : P) -> W A B P  (so two proofs of it can differ in their data)
W_SORT = ('pi', T1, ('pi', T1, ('pi', T0, T0)))
W_CTOR = ('w', [T1, T1, T0, ('var', 0), ('var', 1), ('var', 2)], [('var', 0), ('var', 1), ('var', 2)])

# a PROPOSITION whose LAST field is data:  NE A : Prop ,  ne : (A : Type1) (a : A) -> NE A   (two proofs ne 0 and ne 1 are equal although their last arguments are different data)
NE_SORT = ('pi', T1, T0)
NE_CTOR = ('ne', [T1, ('var', 0)], [('var', 0)])

def world(quot=False):
    """the Prop/Type library, a subtype family Sub (data in Type1 carrying a proof), a proposition W with two data fields, a proposition NE whose last field is data, and two OPAQUE proofs ax1 ax2 of one proposition (Le 0 1)"""
    k, g = pw.build_prop_world()
    t7.declare_family_new(k, 'Sub', SUB_SORT, [SUB_CTOR]); t7.declare_family_new(k, 'W', W_SORT, [W_CTOR]); t7.declare_family_new(k, 'NE', NE_SORT, [NE_CTOR]); g = gmap(k); C = lambda n: ('const', g[n])
    le01 = ap(C('Le'), C('Zero'), ap(C('Succ'), C('Zero')))
    k.declare_unchecked(build_new(k, HY(le01)), None, 'ax1'); k.declare_unchecked(build_new(k, HY(le01)), None, 'ax2')
    if quot: k.declare_quot(g['Eq'])
    return k, gmap(k)

def legacy_world():
    """the same shapes WITHOUT prop_elim: Type0 is an ordinary universe, there is no Prop and so no irrelevance"""
    k = H.Kernel(); k.set_option('parameters', True)
    fam = lambda name, sort, ctors: t7.declare_family_new(k, name, sort, ctors)
    fam('Nat', T1, [('Zero', [], []), ('Succ', [('self',)], [])])
    fam('Eq', ('pi', T1, ('pi', V(0), ('pi', V(0), T0))), [('refl', [T1, ('var', 0)], [('var', 0), ('var', 1), ('var', 1)])])
    fam('Sub', SUB_SORT, [SUB_CTOR])
    return k, gmap(k)

def checker(k, prop): return ic.Checker(t8.build_env(k), params=True, prop_elim=prop, fuel=3_000_000)

# ================================================================================================================ A. the table
def build_cases(g):
    C = lambda n: ('const', g[n])
    NAT, ZERO, S, EQ, REFL, SUB, MK = C('Nat'), C('Zero'), C('Succ'), C('Eq'), C('refl'), C('Sub'), C('mk')
    LE, LE_N, ZERO_LE, LE_TRANS, EX, EX_INTRO, TRUE, TT = C('Le'), C('le_n'), C('zero_le'), C('le_trans'), C('Ex'), C('ex_intro'), C('True'), C('tt')
    eq = lambda A, a, b: ap(EQ, A, a, b); refl = lambda A, a: ap(REFL, A, a)
    one = ap(S, ZERO)
    PK = lam('_', NAT, 'P')                       # the constant predicate  fun _ => P
    subP = ap(SUB, NAT, PK)
    mk0 = lambda h: ap(MK, NAT, PK, ZERO, h)
    cases = []
    def case(name, stmt, proof, expected): cases.append((name, proof, stmt, expected))

    # ---- accepted
    case('subtype values with different proofs are equal', pi('P', T0, pi('h1', 'P', pi('h2', 'P', eq(subP, mk0('h1'), mk0('h2'))))),
         lam('P', T0, lam('h1', 'P', lam('h2', 'P', refl(subP, mk0('h1'))))), True)
    case('a function applied to different proofs', pi('P', T0, pi('f', arrow('P', NAT), pi('h1', 'P', pi('h2', 'P', eq(NAT, ap('f', 'h1'), ap('f', 'h2')))))),
         lam('P', T0, lam('f', arrow('P', NAT), lam('h1', 'P', lam('h2', 'P', refl(NAT, ap('f', 'h1')))))), True)
    case('a TYPE that depends on a proof: x : T h1 is accepted at T h2',
         pi('P', T0, pi('T', arrow('P', T1), pi('h1', 'P', pi('h2', 'P', pi('x', ap('T', 'h1'), eq(ap('T', 'h2'), 'x', 'x')))))),
         lam('P', T0, lam('T', arrow('P', T1), lam('h1', 'P', lam('h2', 'P', lam('x', ap('T', 'h1'), refl(ap('T', 'h2'), 'x')))))), True)
    case('proofs under a binder: (fun n => f h1 n) = (fun n => f h2 n)',
         pi('P', T0, pi('f', arrow('P', arrow(NAT, NAT)), pi('h1', 'P', pi('h2', 'P', eq(arrow(NAT, NAT), lam('n', NAT, ap('f', 'h1', 'n')), lam('n', NAT, ap('f', 'h2', 'n'))))))),
         lam('P', T0, lam('f', arrow('P', arrow(NAT, NAT)), lam('h1', 'P', lam('h2', 'P', refl(arrow(NAT, NAT), lam('n', NAT, ap('f', 'h1', 'n'))))))), True)
    case('opaque proofs (two axioms of Le 0 1)', pi('f', arrow(ap(LE, ZERO, one), NAT), eq(NAT, ap('f', C('ax1')), ap('f', C('ax2')))),
         lam('f', arrow(ap(LE, ZERO, one), NAT), refl(NAT, ap('f', C('ax1')))), True)
    pT = lam('_', NAT, TRUE)
    case('a constructor proof with different data inside: ex_intro 0 tt = ex_intro 1 tt', pi('f', arrow(ap(EX, NAT, pT), NAT),
         eq(NAT, ap('f', ap(EX_INTRO, NAT, pT, ZERO, TT)), ap('f', ap(EX_INTRO, NAT, pT, one, TT)))),
         lam('f', arrow(ap(EX, NAT, pT), NAT), refl(NAT, ap('f', ap(EX_INTRO, NAT, pT, ZERO, TT)))), True)
    leNN = lambda n: ap(LE, n, n)
    stuck = match('n', lam('k', NAT, leNN('k')), ap(LE_N, ZERO), lam('m', NAT, ap(LE_N, ap(S, 'm'))))     # a proof of Le n n that is a stuck match on the variable n
    case('a stuck match proof against a constructor proof', pi('n', NAT, pi('f', arrow(leNN('n'), NAT), eq(NAT, ap('f', ap(LE_N, 'n')), ap('f', stuck)))),
         lam('n', NAT, lam('f', arrow(leNN('n'), NAT), refl(NAT, ap('f', ap(LE_N, 'n'))))), True)
    case('library proofs of different shapes (zero_le n against le_trans 0 0 n ..)', pi('n', NAT, pi('f', arrow(ap(LE, ZERO, 'n'), NAT),
         eq(NAT, ap('f', ap(ZERO_LE, 'n')), ap('f', ap(LE_TRANS, ZERO, ZERO, 'n', ap(LE_N, ZERO), ap(ZERO_LE, 'n')))))),
         lam('n', NAT, lam('f', arrow(ap(LE, ZERO, 'n'), NAT), refl(NAT, ap('f', ap(ZERO_LE, 'n'))))), True)
    case('proofs under a binder that MENTION the bound variable: (fun n => f n (u n)) = (fun n => f n (v n))',
         pi('Q', arrow(NAT, T0), pi('f', pi('n', NAT, arrow(ap('Q', 'n'), NAT)), pi('u', pi('n', NAT, ap('Q', 'n')), pi('v', pi('n', NAT, ap('Q', 'n')),
         eq(arrow(NAT, NAT), lam('n', NAT, ap('f', 'n', ap('u', 'n'))), lam('n', NAT, ap('f', 'n', ap('v', 'n')))))))),
         lam('Q', arrow(NAT, T0), lam('f', pi('n', NAT, arrow(ap('Q', 'n'), NAT)), lam('u', pi('n', NAT, ap('Q', 'n')), lam('v', pi('n', NAT, ap('Q', 'n')),
         refl(arrow(NAT, NAT), lam('n', NAT, ap('f', 'n', ap('u', 'n')))))))), True)
    # nested irrelevance: f (refl (mk h1)) = f (refl (mk h2)) where  f : Eq S (mk h1) (mk h1) -> Nat . The argument's TYPE contains h2, so typing the application needs irrelevance,
    # and then the two arguments are proofs whose own types are convertible only by irrelevance again
    case('nested: f (refl (mk h1)) = f (refl (mk h2))', pi('P', T0, pi('h1', 'P', pi('h2', 'P', pi('f', arrow(eq(subP, mk0('h1'), mk0('h1')), NAT),
         eq(NAT, ap('f', refl(subP, mk0('h1'))), ap('f', refl(subP, mk0('h2')))))))),
         lam('P', T0, lam('h1', 'P', lam('h2', 'P', lam('f', arrow(eq(subP, mk0('h1'), mk0('h1')), NAT), refl(NAT, ap('f', refl(subP, mk0('h1')))))))), True)
    # irrelevance whose own type check needs irrelevance: u1 : Q (mk h1) and u2 : Q (mk h2) are proofs of propositions that differ only in a proof (inside the argument of Q)
    Qmk = lambda h: ap('Q', mk0(h))
    case('nested twice: g u1 = g u2 where u1 : Q (mk h1), u2 : Q (mk h2), g : Q (mk h1) -> Nat', pi('P', T0, pi('h1', 'P', pi('h2', 'P', pi('Q', arrow(subP, T0), pi('u1', Qmk('h1'),
         pi('u2', Qmk('h2'), pi('g', arrow(Qmk('h1'), NAT), eq(NAT, ap('g', 'u1'), ap('g', 'u2'))))))))),
         lam('P', T0, lam('h1', 'P', lam('h2', 'P', lam('Q', arrow(subP, T0), lam('u1', Qmk('h1'), lam('u2', Qmk('h2'), lam('g', arrow(Qmk('h1'), NAT), refl(NAT, ap('g', 'u1'))))))))), True)
    case('irrelevance is transitive through a chain of three proofs', pi('P', T0, pi('f', arrow('P', NAT), pi('h1', 'P', pi('h2', 'P', pi('h3', 'P',
         pi('e1', eq(NAT, ap('f', 'h1'), ap('f', 'h2')), pi('e2', eq(NAT, ap('f', 'h2'), ap('f', 'h3')), eq(NAT, ap('f', 'h1'), ap('f', 'h3'))))))))),
         lam('P', T0, lam('f', arrow('P', NAT), lam('h1', 'P', lam('h2', 'P', lam('h3', 'P', lam('e1', eq(NAT, ap('f', 'h1'), ap('f', 'h2')),
         lam('e2', eq(NAT, ap('f', 'h2'), ap('f', 'h3')), refl(NAT, ap('f', 'h1'))))))))), True)

    # a CLOSED equation (the typing context is empty, not absent): two opaque proofs inside a subtype value
    leP = lam('_', NAT, ap(LE, ZERO, one))
    case('a closed equation between proof-different values: mk 0 ax1 = mk 0 ax2 (empty context)', eq(ap(SUB, NAT, leP), ap(MK, NAT, leP, ZERO, C('ax1')), ap(MK, NAT, leP, ZERO, C('ax2'))),
         ap(C('refl'), ap(SUB, NAT, leP), ap(MK, NAT, leP, ZERO, C('ax1'))), True)
    # a proposition whose LAST field is data: the comparison of two constructor applications compares every argument but the last, then goes on with the last (see Checker.conv): when that last pair
    # is different DATA, the two applications are still proofs of one proposition, so the level above must get its own try at proof irrelevance
    NE, NE_C, ADD = C('NE'), C('ne'), C('add')
    def num(n):
        t = ZERO
        for _ in range(n): t = ap(S, t)
        return t
    neP = ap(NE, NAT)
    case('a proposition whose last field is data: ne 0 = ne 1', pi('f', arrow(neP, NAT), eq(NAT, ap('f', ap(NE_C, NAT, ZERO)), ap('f', ap(NE_C, NAT, one)))),
         lam('f', arrow(neP, NAT), refl(NAT, ap('f', ap(NE_C, NAT, ZERO)))), True)
    case('the same with long numerals in the last field: ne (250) = ne (251)', pi('f', arrow(neP, NAT), eq(NAT, ap('f', ap(NE_C, NAT, num(250))), ap('f', ap(NE_C, NAT, num(251))))),
         lam('f', arrow(neP, NAT), refl(NAT, ap('f', ap(NE_C, NAT, num(250))))), True)
    # unary numerals longer than the independent checker's conversion depth (200): the chain Succ (Succ (..)) is walked, not recursed into
    case('a closed sum beyond the old depth limit: 150 + 150 = 300', eq(NAT, ap(ADD, num(150), num(150)), num(300)), refl(NAT, num(300)), True)
    case('a closed sum that is long on both sides: 260 + 140 = 400', eq(NAT, ap(ADD, num(260), num(140)), num(400)), refl(NAT, num(400)), True)
    case('a closed numeral computation under a binder (the typing context is not empty): 220 + 0 = 220', pi('n', NAT, eq(NAT, ap(ADD, num(220), ZERO), num(220))),
         lam('n', NAT, refl(NAT, num(220))), True)

    # ---- rejected
    case('NOT: an off-by-one numeral claim beyond the old depth limit: 150 + 150 = 301', eq(NAT, ap(ADD, num(150), num(150)), num(301)), refl(NAT, num(301)), False)
    case('NOT: the same claim with the proof of the true numeral: refl 300 : 150 + 150 = 301', eq(NAT, ap(ADD, num(150), num(150)), num(301)), refl(NAT, num(300)), False)
    case('NOT: a long numeral is not the numeral one shorter: 260 + 140 = 399', eq(NAT, ap(ADD, num(260), num(140)), num(399)), refl(NAT, num(399)), False)
    case('NOT: the last argument of a constructor is compared (S a = S b for a b : Nat)', pi('a', NAT, pi('b', NAT, eq(NAT, ap(S, 'a'), ap(S, 'b')))),
         lam('a', NAT, lam('b', NAT, refl(NAT, ap(S, 'a')))), False)
    case('NOT: data is not irrelevant (a = b for a b : Nat)', pi('a', NAT, pi('b', NAT, eq(NAT, 'a', 'b'))), lam('a', NAT, lam('b', NAT, refl(NAT, 'a'))), False)
    case('NOT: functions are not irrelevant', pi('f', arrow(NAT, NAT), pi('g', arrow(NAT, NAT), eq(arrow(NAT, NAT), 'f', 'g'))), lam('f', arrow(NAT, NAT), lam('g', arrow(NAT, NAT), refl(arrow(NAT, NAT), 'f'))), False)
    case('NOT: propositions are not irrelevant (P = Q for P Q : Prop)', pi('P', T0, pi('Q', T0, eq(T0, 'P', 'Q'))), lam('P', T0, lam('Q', T0, refl(T0, 'P'))), False)
    case('NOT: data next to a proof is still data (mk 0 h1 = mk 1 h2)', pi('P', T0, pi('h1', 'P', pi('h2', 'P', eq(subP, mk0('h1'), ap(MK, NAT, PK, one, 'h2'))))),
         lam('P', T0, lam('h1', 'P', lam('h2', 'P', refl(subP, mk0('h1'))))), False)
    case('NOT: a function of data and a proof (f 0 h = f 1 h)', pi('P', T0, pi('f', arrow(NAT, arrow('P', NAT)), pi('h', 'P', eq(NAT, ap('f', ZERO, 'h'), ap('f', one, 'h'))))),
         lam('P', T0, lam('f', arrow(NAT, arrow('P', NAT)), lam('h', 'P', refl(NAT, ap('f', ZERO, 'h'))))), False)
    case('NOT: the proofs must be proofs of the SAME proposition (h1 : P, h2 : Q)', pi('P', T0, pi('Q', T0, pi('h1', 'P', pi('h2', 'Q', eq(subP, mk0('h1'), mk0('h2')))))),
         lam('P', T0, lam('Q', T0, lam('h1', 'P', lam('h2', 'Q', refl(subP, mk0('h1')))))), None)       # ill-typed statement: both checkers must refuse the STATEMENT (None = "statement must not type")
    return cases

def section_a():
    k, g = world(); chk = checker(k, True); cases = build_cases(g); bad = 0; n_ok = 0
    for name, proof, stmt, expected in cases:
        if expected is None:
            sk, si = kernel_type_ok(k, stmt), indep_type_ok(chk, stmt)
            ok = (not sk) and (not si)
            print("   %-86s statement typed: kernel %s indep %s %s" % (name, sk, si, 'ok' if ok else 'FAILED'))
            expect(ok, 'A: ill-typed statement accepted: ' + name); bad += not ok; continue
        kv, iv = kernel_ok(k, proof, stmt), indep_ok(chk, proof, stmt)
        good = kv == expected and iv == expected; bad += not good; n_ok += good
        print("   %-86s kernel %-5s indep %-5s %s" % (name, kv, iv, 'ok' if good else 'FAILED (expected %s)' % expected))
        expect(good, 'A: ' + name)
    print("A. %d cases, %d wrong verdicts" % (len(cases), bad))

    # the same shape as the first case, with Quot.lift: two different respects-proofs give the same value
    kq, gq = world(quot=True); cq = checker(kq, True); C = lambda n: ('const', gq[n])
    NAT, ZERO, EQ = C('Nat'), C('Zero'), C('Eq'); eq = lambda A, a, b: ap(EQ, A, a, b)
    eqn = lam('a', NAT, lam('b', NAT, eq(NAT, 'a', 'b')))
    hty = pi('a', NAT, pi('b', NAT, pi('p', eq(NAT, 'a', 'b'), eq(NAT, 'a', 'b'))))      # the respects-proof for the identity function: Eq a b -> Eq (id a) (id b)
    lift = lambda h: ap(C('Quot.lift'), NAT, eqn, NAT, lam('a1', NAT, 'a1'), h, ap(C('Quot.mk'), NAT, eqn, ZERO))
    stmt = pi('h1', hty, pi('h2', hty, eq(NAT, lift('h1'), lift('h2'))))
    proof = lam('h1', hty, lam('h2', hty, ap(C('refl'), NAT, lift('h1'))))
    kv, iv = kernel_ok(kq, proof, stmt), indep_ok(cq, proof, stmt)
    print("   Quot.lift with two different respects-proofs h1 h2 gives equal values: kernel %s indep %s" % (kv, iv))
    expect(kv and iv, 'A: Quot.lift with different respects-proofs')

    # without prop_elim there is no Prop: the first statement must be REFUSED by both checkers
    kl, gl = legacy_world(); cl = checker(kl, False); C = lambda n: ('const', gl[n])
    NAT, ZERO, SUB, MK, EQ = C('Nat'), C('Zero'), C('Sub'), C('mk'), C('Eq'); PK = lam('_', NAT, 'P'); subP = ap(SUB, NAT, PK); mk0 = lambda h: ap(MK, NAT, PK, ZERO, h)
    stmt = pi('P', T0, pi('h1', 'P', pi('h2', 'P', ap(EQ, subP, mk0('h1'), mk0('h2'))))); proof = lam('P', T0, lam('h1', 'P', lam('h2', 'P', ap(C('refl'), subP, mk0('h1')))))
    sk, si = kernel_type_ok(kl, stmt), indep_type_ok(cl, stmt); kv, iv = kernel_ok(kl, proof, stmt), indep_ok(cl, proof, stmt)
    print("   WITHOUT prop_elim: statement typed (kernel %s, indep %s); the irrelevance proof is accepted: kernel %s, indep %s (both must be False)" % (sk, si, kv, iv))
    expect(sk and si, 'A: the legacy world should type the statement'); expect(not kv and not iv, 'A: proof irrelevance leaked into a world without prop_elim')

# ================================================================================================================ B. ill-typed comparisons
def indep_conv(chk, a, b, names, types):
    """the independent checker's conversion of two OPEN terms in the context  names : types  (variables are named free0, free1, .. as in ic.from_hybrid)"""
    tr = lambda t, ctxn: ic.from_hybrid(l2h(lv(t, tuple(ctxn)), len(ctxn)))
    ctx = {'free%d' % i: tr(t, names[:i]) for i, t in enumerate(types)}
    chk.fuel = chk.fuel0
    try: return bool(chk.conv(tr(a, names), tr(b, names), 0, ctx))
    except ic.CheckError: return False

def section_b():
    k, g = world(); chk = checker(k, True); C = lambda n: ('const', g[n])
    hyc = lambda t, ctx: build_new(k, l2h(lv(t, ctx), len(ctx)))
    def unify(a, b, ctx_names, ctx_types):
        tys = [hyc(t, tuple(ctx_names[:i])) for i, t in enumerate(ctx_types)]
        return k.unify_open(hyc(a, tuple(ctx_names)), hyc(b, tuple(ctx_names)), tys)[0]
    NAT = C('Nat')
    rows = [('two proofs of one proposition (h1 h2 : P)', ['P', 'h1', 'h2'], [T0, 'P', 'P'], 'h1', 'h2', True),
            ('proofs of DIFFERENT propositions (h1 : P, h2 : Q)', ['P', 'Q', 'h1', 'h2'], [T0, T0, 'P', 'Q'], 'h1', 'h2', False),
            ('proofs of propositions that are only equal after reduction (P vs (fun X => X) P)', ['P', 'h1', 'h2'], [T0, 'P', ap(lam('X', T0, 'X'), 'P')], 'h1', 'h2', True),
            ('two elements of a Type1 (a b : A, A : Type1)', ['A', 'a', 'b'], [T1, 'A', 'A'], 'a', 'b', False),
            ('two natural numbers', ['a', 'b'], [NAT, NAT], 'a', 'b', False),
            ('two propositions (P Q : Prop)', ['P', 'Q'], [T0, T0], 'P', 'Q', False),
            ('a proof and an element of a Type1 (ill-typed)', ['P', 'A', 'h', 'a'], [T0, T1, 'P', 'A'], 'h', 'a', False),
            ('a proof and a proposition (ill-typed)', ['P', 'h'], [T0, 'P'], 'h', 'P', False),
            ('two functions into Prop (p q : A -> Prop are data of a Pi type)', ['A', 'p', 'q'], [T1, arrow('A', T0), arrow('A', T0)], 'p', 'q', False),
            ('two proofs of a universally quantified proposition (u v : forall x : Nat, P x)', ['P', 'u', 'v'], [arrow(NAT, T0), pi('x', NAT, ap('P', 'x')), pi('x', NAT, ap('P', 'x'))], 'u', 'v', True),
            ('two proofs of a proposition that mentions a variable (h1 : P x, h2 : P x)', ['P', 'x', 'h1', 'h2'], [arrow(NAT, T0), NAT, ap('P', 'x'), ap('P', 'x')], 'h1', 'h2', True),
            ('two proofs of propositions that differ in the variable (h1 : P x, h2 : P y)', ['P', 'x', 'y', 'h1', 'h2'], [arrow(NAT, T0), NAT, NAT, ap('P', 'x'), ap('P', 'y')], 'h1', 'h2', False)]
    bad = 0
    for name, names, types, a, b, expected in rows:
        got, ind = unify(a, b, names, types), indep_conv(chk, a, b, names, types); good = got == expected and ind == expected; bad += not good
        print("   %-92s kernel %-5s indep %-5s %s" % (name, got, ind, 'ok' if good else 'FAILED (expected %s)' % expected))
        expect(good, 'B: ' + name)
    print("B. %d comparisons (kernel unifier and independent conversion), %d wrong" % (len(rows), bad))

    # an equality obtained by irrelevance must not keep the commitments of the structural attempt that failed before it. Two proofs of  W Nat Nat P  that differ in the SECOND
    # data field; the first is (a) a metavariable the structural attempt solves, or (b) a metavariable applied to a non-variable, which it defers
    names = ['P', 'h1', 'h2']; types = [T0, 'P', 'P']; ctxP = tuple(names)
    tys = [hyc(t, tuple(names[:i])) for i, t in enumerate(types)]
    NATn = build_new(k, HY(NAT)); ZERO = C('Zero'); one = ap(C('Succ'), C('Zero'))
    W = lambda c, d, h: ap(C('w'), NAT, NAT, 'P', c, d, h)
    def hy_with_meta(c_hybrid, d, h):         # W Nat Nat P <c_hybrid> d h, built as a hybrid tree because c_hybrid may mention a metavariable
        base = lambda t: l2h(lv(t, ctxP), 3)
        head = ('app', ('app', ('app', base(C('w')), base(NAT)), base(NAT)), base('P'))
        return build_new(k, ('app', ('app', ('app', head, c_hybrid), base(d)), base(h)))
    pi_ctx = k.close_telescope('pi', tys, NATn, 0)                          # ?m : forall (P : Prop) (h1 h2 : P), Nat   -- applied to every context variable
    m = k.fresh_meta(pi_ctx, 3); mt = ('meta', m)
    for i in range(3): mt = ('app', mt, ('fvar', i))
    rhs = build_new(k, l2h(lv(W(one, one, 'h2'), ctxP), 3))
    ok, msg, pend = k.unify_open(hy_with_meta(mt, ZERO, 'h1'), rhs, tys); solved = k.meta_is_solved(m)
    print("   structural attempt solves ?m := 1, then fails on the data (0 vs 1); irrelevance equates the two proofs: unified %s, ?m still solved: %s (must be False), pending %d" % (ok, solved, pend))
    expect(ok and not solved and pend == 0, 'B: an equality obtained by irrelevance kept a metavariable assignment from the failed structural attempt')
    f = k.fresh_meta(k.close_telescope('pi', [NATn], NATn, 0), 0); ft = ('app', ('meta', f), l2h(lv(one, ctxP), 3))      # ?f 1 : not a pattern, so  ?f 1 =?= 1  is DEFERRED
    ok2, msg2, pend2 = k.unify_open(hy_with_meta(ft, ZERO, 'h1'), rhs, tys)
    print("   structural attempt defers  ?f 1 =?= 1  and then fails on the data; irrelevance equates the two proofs: unified %s, deferred constraints left: %d (must be 0)" % (ok2, pend2))
    expect(ok2 and pend2 == 0, 'B: an equality obtained by irrelevance kept a deferred constraint from the failed structural attempt')

# ================================================================================================================ C. tactics
def section_c():
    k, g = world(); C = lambda n: ('const', g[n]); NAT, ZERO, EQ, SUB, MK = C('Nat'), C('Zero'), C('Eq'), C('Sub'), C('mk')
    PK = lam('_', NAT, 'P'); subP = ap(SUB, NAT, PK); mk0 = lambda h: ap(MK, NAT, PK, ZERO, h)
    stmt = pi('P', T0, pi('h1', 'P', pi('h2', 'P', ap(EQ, subP, mk0('h1'), mk0('h2')))))
    s = build_new(k, HY(stmt)); ps = k.start_proof(s)
    for _ in range(3): ps.intro('x')
    ps.apply(build_new(k, ('const', g['refl'])))
    done = ps.num_goals() == 0 and ps.is_complete(); proof = ps.extract() if done else None
    verified = bool(done and k.check_open(proof, s, [])[0])
    chk = checker(k, True)
    indep = bool(done and chk.check_closed(parse_raw(k.dump_raw(proof), True), parse_raw(k.dump_raw(s), True))[0])
    print("C. `apply refl` closes  mk 0 h1 = mk 0 h2  (after three intros): complete %s, kernel-verified %s, independent checker %s" % (done, verified, indep))
    expect(done and verified and indep, 'C: apply refl on an equation between proof-different terms')
    # the same tactic must NOT close an equation between different data
    stmt2 = pi('a', NAT, pi('b', NAT, ap(EQ, NAT, 'a', 'b'))); s2 = build_new(k, HY(stmt2)); ps2 = k.start_proof(s2); ps2.intro('x'); ps2.intro('x')
    try: ps2.apply(build_new(k, ('const', g['refl']))); closed = ps2.num_goals() == 0
    except Exception: closed = False
    print("   `apply refl` on  a = b  (a b : Nat): closes the goal: %s (must be False)" % closed)
    expect(not closed, 'C: apply refl closed a = b for natural numbers')

# ================================================================================================================ D. trusted base
def section_d():
    k, g = world(); chk = checker(k, True); cases = build_cases(g)
    used = {}
    for name, proof, stmt, expected in cases:
        if expected is True:
            ax, qs = audit.dependencies(k, HY(proof)); used[name] = (sorted(ax), sorted(qs))
    extra = {n: v for n, v in used.items() if set(v[0]) - {'ax1', 'ax2'} or v[1]}
    print("D. accepted proofs: axioms used outside the two test axioms or quotient constants: %s" % (extra if extra else 'none'))
    expect(not extra, 'D: a proof irrelevance proof rests on an unexpected axiom: %s' % list(extra))
    bad = chk.check_definitions(); print("   independent re-check of all %d globals: %d rejected" % (k.num_globals(), len(bad))); expect(not bad, 'D: re-check rejected %s' % bad[:3])

# ================================================================================================================ E. canary
def section_e(N):
    k, g = world(); C = lambda n: ('const', g[n]); NAT, ZERO, S, EQ, SUB, MK, FALSE = C('Nat'), C('Zero'), C('Succ'), C('Eq'), C('Sub'), C('mk'), C('False')
    one = ap(S, ZERO); PK = lam('_', NAT, 'P'); subP = ap(SUB, NAT, PK); mk = lambda n, h: ap(MK, NAT, PK, n, h)
    eq = lambda A, a, b: ap(EQ, A, a, b)
    false_stmts = {
        'a = b for naturals': pi('a', NAT, pi('b', NAT, eq(NAT, 'a', 'b'))),
        'mk 0 h1 = mk 1 h2': pi('P', T0, pi('h1', 'P', pi('h2', 'P', eq(subP, mk(ZERO, 'h1'), mk(one, 'h2'))))),
        'P = Q for propositions': pi('P', T0, pi('Q', T0, eq(T0, 'P', 'Q'))),
        'f h = 1 for f h : Nat': pi('P', T0, pi('f', arrow('P', NAT), pi('h', 'P', eq(NAT, ap('f', 'h'), one)))),
        'f 0 h = f 1 h': pi('P', T0, pi('f', arrow(NAT, arrow('P', NAT)), pi('h', 'P', eq(NAT, ap('f', ZERO, 'h'), ap('f', one, 'h'))))),
        'False': FALSE,
        'forall P, P': pi('P', T0, 'P'),
    }
    true_stmts = {
        'mk 0 h1 = mk 0 h2': pi('P', T0, pi('h1', 'P', pi('h2', 'P', eq(subP, mk(ZERO, 'h1'), mk(ZERO, 'h2'))))),
        'f h1 = f h2': pi('P', T0, pi('f', arrow('P', NAT), pi('h1', 'P', pi('h2', 'P', eq(NAT, ap('f', 'h1'), ap('f', 'h2')))))),
        'a = a': pi('a', NAT, eq(NAT, 'a', 'a')),
        'f 0 h1 = f 0 h2': pi('P', T0, pi('f', arrow(NAT, arrow('P', NAT)), pi('h1', 'P', pi('h2', 'P', eq(NAT, ap('f', ZERO, 'h1'), ap('f', ZERO, 'h2')))))),
        'mk 1 h1 = mk 1 h2': pi('P', T0, pi('h1', 'P', pi('h2', 'P', eq(subP, mk(one, 'h1'), mk(one, 'h2'))))),
    }
    consts = [g[n] for n in ('refl', 'tt', 'mk', 'Zero', 'Succ', 'cong', 'symm', 'trans', 'ax1', 'ax2', 'zero_le', 'le_n')]
    ind = {g[n] for n in ('Nat', 'Eq', 'Le', 'And', 'Or', 'Ex', 'False', 'True', 'Sub')}
    fl = {n: lv(t) for n, t in false_stmts.items()}; tr = {n: lv(t) for n, t in true_stmts.items()}
    tried, completed, verified, first = canlib.run(k, g, fl, consts, ind, N, 171)
    print("E. soundness canary (proof irrelevance on): %d random proof attempts on %d FALSE statements, %d completed, verified proof of a false statement: %s" % (tried, len(fl), completed, first))
    expect(first is None, 'E: canary proved a false statement: %s' % first)
    tried2, completed2, verified2, _ = canlib.run(k, g, tr, consts, ind, 3 * N, 172, stop_on_verified=False)
    print("   canary POWER (same generator, %d TRUE statements): %d attempts, %d completed, %d verified" % (len(tr), tried2, completed2, verified2))
    expect(verified2 >= 5, 'E: the canary has no power: %d true statements proved' % verified2)
    expect(completed2 == verified2, 'E: %d completed proof states did not verify' % (completed2 - verified2))

# ================================================================================================================ F. the independent checker's own limits
def section_f():
    """The independent checker is allowed to be INCOMPLETE (it may refuse what the kernel accepts) but never unsound, and a refusal that comes from one of its resource limits must say so.
    (1) a chain of one constructor (a unary numeral) is walked by a loop, so its length is not limited by the recursion depth; refusing a wrong claim about long numerals is cheap, because
        proof irrelevance is never inferred for data (counted: `irrelevance_inferences`);
    (2) a comparison that does hit the depth limit (two chains of applications of a variable, equal only after beta reduction at every level) is refused as 'a depth limit was reached', not as an
        ordinary type mismatch, while the kernel accepts it."""
    k, g = world(); chk = checker(k, True); C = lambda n: ('const', g[n])
    NAT, ZERO, S, EQ, ADD = C('Nat'), C('Zero'), C('Succ'), C('Eq'), C('add'); eq = lambda A, a, b: ap(EQ, A, a, b)
    def num(n):
        t = ZERO
        for _ in range(n): t = ap(S, t)
        return t
    chk.irrelevance_inferences = 0; t = time.time()
    stmt = eq(NAT, ap(ADD, num(300), num(300)), num(601)); proof = ap(C('refl'), NAT, num(601))
    refused = not indep_ok(chk, proof, stmt); dt = time.time() - t
    print("F. a wrong claim about long numerals (300 + 300 = 601): refused %s in %.2f s with %d irrelevance inferences (must be refused, at most 2: the comparison of the two equations themselves)" % (refused, dt, chk.irrelevance_inferences))
    expect(refused, 'F: the independent checker accepted 300 + 300 = 601'); expect(chk.irrelevance_inferences <= 2, 'F: proof irrelevance was inferred for data constructors (%d times)' % chk.irrelevance_inferences)
    expect(dt < 30, 'F: refusing a wrong claim about long numerals took %.1f s' % dt)
    # proofs are NOT data: the inference does happen for them
    chk.irrelevance_inferences = 0
    stmt2 = pi('P', T0, pi('h1', 'P', pi('h2', 'P', eq(ap(C('Sub'), NAT, lam('_', NAT, 'P')), ap(C('mk'), NAT, lam('_', NAT, 'P'), ZERO, 'h1'), ap(C('mk'), NAT, lam('_', NAT, 'P'), ZERO, 'h2')))))
    proof2 = lam('P', T0, lam('h1', 'P', lam('h2', 'P', ap(C('refl'), ap(C('Sub'), NAT, lam('_', NAT, 'P')), ap(C('mk'), NAT, lam('_', NAT, 'P'), ZERO, 'h1')))))
    ok2 = indep_ok(chk, proof2, stmt2)
    print("   proofs of one proposition are still equated by inference: accepted %s with %d irrelevance inferences (must be accepted, at least 1)" % (ok2, chk.irrelevance_inferences))
    expect(ok2 and chk.irrelevance_inferences >= 1, 'F: the irrelevance of proofs is no longer inferred')

    def depth_case(label, k_, g_, chk_):
        Cc = lambda n: ('const', g_[n]); NATc, EQc = Cc('Nat'), Cc('Eq'); eqc = lambda A, a, b: ap(EQc, A, a, b)
        depth = chk_.MAX_DEPTH + 60
        lam_f = lambda y_: ap(lam('y', NATc, ap('f', 'y')), y_)
        plain, lazy = 'x', 'x'
        for _ in range(depth): plain = ap('f', plain); lazy = lam_f(lazy)
        stmt3 = pi('f', arrow(NATc, NATc), pi('x', NATc, eqc(NATc, plain, lazy))); proof3 = lam('f', arrow(NATc, NATc), lam('x', NATc, ap(Cc('refl'), NATc, plain)))
        kv = kernel_ok(k_, proof3, stmt3); iv, msg = chk_.check_closed(HY(proof3), HY(stmt3))
        print("   %s: %d nested applications of a variable against the same with a beta-redex at every level: kernel %s, independent checker %s (%s)" % (label, depth, kv, iv, msg))
        expect(kv, 'F: the kernel refused two convertible chains of applications (%s)' % label)
        expect(not iv and 'depth limit' in msg, 'F: the independent checker %s (%s); it must refuse and name its depth limit (%s)' % ('accepted' if iv else 'refused', msg, label))
        # the flag belongs to ONE run: an ordinary mismatch afterwards is reported as one
        stmt_w = eqc(NATc, ap(Cc('Succ'), Cc('Zero')), Cc('Zero')); iv2, msg2 = chk_.check_closed(HY(ap(Cc('refl'), NATc, Cc('Zero'))), HY(stmt_w))
        print("   %s: an ordinary mismatch afterwards (1 = 0): %s" % (label, msg2))
        expect(not iv2 and msg2.startswith('type mismatch') and 'depth limit' not in msg2, 'F: an ordinary mismatch is reported as %r (%s)' % (msg2, label))
    depth_case('with proof irrelevance', k, g, chk)
    # WITHOUT prop_elim the conversion never calls the irrelevance check, whose own depth guard would also raise the flag: the guard of the comparison itself must report the limit
    kl, gl = legacy_world(); depth_case('without prop_elim', kl, gl, checker(kl, False))

if __name__ == '__main__':
    t0 = time.time(); N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    section_a(); section_b(); section_c(); section_d(); section_f(); section_e(N)
    print("elapsed %.0fs" % (time.time() - t0)); print("S17 IRRELEVANCE GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
