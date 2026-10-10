"""S16 gate: quotients.

Quot, Quot.mk, Quot.lift, Quot.ind and Quot.sound are five ordinary constants whose types the kernel builds itself (declare_quot), plus ONE reduction rule
        Quot.lift A r B f h (Quot.mk A' r' a)  ~>  f a .
What must hold, and where it is tested:
  A  the five types are exactly the specified ones (compared with trees written independently in this file), at two universe levels; declare_quot refuses every ill-formed
     equality family, mode, repeat and clash, and leaves nothing declared when it refuses
  B  a table of well- and ill-typed terms, each judged by the KERNEL and by the INDEPENDENT checker, which must both give the expected verdict: computation (also under a binder, nested,
     over-applied), no firing on a stuck or non-mk argument, ill-typed h / f / q / motive / sound rejected; every rejected case has an accepted twin that differs in one place
  C  eta (F27), and the conditional contradiction that checking h exists to prevent
  D  real mathematics: all elements of Quot Nat total are equal (ind + sound), the eta law for Quot, and FUNEXT derived from Quot.sound (it needs eta), verified by both checkers and
     audited: it rests on quotient constants only, none of the classical axioms
  E  the independent checker refuses an environment whose quotient constants were tampered with, and then does NOT enable the reduction rule
  F  F26: an opaque type constant (an axiom `Foo : Type1`, or Quot itself) is not an inductive family: a branchless match / cases / induct on its inhabitants is refused by both
  G  the random soundness canary on false statements about quotients, with a power run on true ones"""
import sys, time, functools
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

T0, T1, T2 = pw.T0, pw.T1, ('type', 2)

from named import *

def quot_types(EQ, Q, MK, j):
    """the specified types of the five constants, written out here, independently of the kernel's declare_quot and of the checker's quot_types"""
    Tj = ('type', j); rel = lambda A: arrow(A, arrow(A, T0))
    return {
        'Quot': pi('A', Tj, pi('r', rel('A'), Tj)),
        'Quot.mk': pi('A', Tj, pi('r', rel('A'), pi('a', 'A', ap(Q, 'A', 'r')))),
        'Quot.lift': pi('A', Tj, pi('r', rel('A'), pi('B', Tj, pi('f', arrow('A', 'B'),
                        pi('h', pi('a', 'A', pi('b', 'A', pi('p', ap('r', 'a', 'b'), ap(EQ, 'B', ap('f', 'a'), ap('f', 'b'))))), pi('q', ap(Q, 'A', 'r'), 'B')))))),
        'Quot.ind': pi('A', Tj, pi('r', rel('A'), pi('P', arrow(ap(Q, 'A', 'r'), T0), pi('h', pi('a', 'A', ap('P', ap(MK, 'A', 'r', 'a'))), pi('q', ap(Q, 'A', 'r'), ap('P', 'q')))))),
        'Quot.sound': pi('A', Tj, pi('r', rel('A'), pi('a', 'A', pi('b', 'A', pi('p', ap('r', 'a', 'b'), ap(EQ, ap(Q, 'A', 'r'), ap(MK, 'A', 'r', 'a'), ap(MK, 'A', 'r', 'b'))))))),
    }

# ---------------------------------------------------------------------------------------------------------------- the standard worlds
def std_world():
    k, _ = pw.build_prop_world(); g0 = gmap(k)
    handle = k.declare_quot(g0['Eq']); return k, gmap(k), handle

def fresh_k():
    k = H.Kernel(); k.set_option('prop_elim', True); k.set_option('parameters', True); return k

EQSORT = lambda Tj: ('pi', Tj, ('pi', V(0), ('pi', V(0), T0)))
REFLC = lambda Tj, name='refl': (name, [Tj, ('var', 0)], [('var', 0), ('var', 1), ('var', 1)])

# ================================================================================================================ A. the types, and declare_quot's refusals
def section_a():
    k, g, handle = std_world(); C = lambda n: ('const', g[n])
    roles = [k.quot_role(x) for x in handle]
    expect(list(handle) == [g['Quot'], g['Quot.mk'], g['Quot.lift'], g['Quot.ind'], g['Quot.sound']] and roles == [1, 2, 3, 4, 5], 'A: handle / roles')
    want = quot_types(C('Eq'), C('Quot'), C('Quot.mk'), 1)
    same = {n: k.global_info(g[n])[1] == build_new(k, HY(t)) for n, t in want.items()}
    print("A. level 1: the five declared types equal the independently written ones: %s" % same)
    expect(all(same.values()), 'A: a Quot type differs from the specification')
    expect(all(k.global_info(g[n])[2] is None and k.global_info(g[n])[3] < 0 and not k.is_inductive(g[n]) for n in want), 'A: a quotient constant has a value, is a constructor or is inductive')
    chk = ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=2_000_000)
    expect(chk.quot is not None and chk.quot_problem is None, 'A: the independent checker did not validate the quotient constants: %s' % chk.quot_problem)
    bad = chk.check_definitions(); print("   independent checker validated the roles: %s; re-check of all %d globals: %d rejected" % (chk.quot is not None, len(g), len(bad)))
    expect(not bad, 'A: independent re-check rejected %s' % bad[:3])

    # refusals, each in a fresh world; nothing may be left declared
    def attempt(name, build, needle):
        k2, gid_fn = build(); n0 = k2.num_globals()
        ok = refuses(lambda: k2.declare_quot(gid_fn()), needle) and k2.num_globals() == n0
        print("   refuse %-52s %s" % (name, 'ok' if ok else 'FAILED'))
        expect(ok, 'A: declare_quot should refuse: ' + name)
    def with_family(name, sort, ctors):
        def build():
            kk = fresh_k(); t7.declare_family_new(kk, 'True', T0, [('tt', [], [])])
            tid = t7.declare_family_new(kk, name, sort, ctors)
            return kk, lambda: tid
        return build
    def legacy():       # the same equality family, but in a world without prop_elim (the option cannot be switched off once globals exist)
        kk = H.Kernel(); eq = t7.declare_family_new(kk, 'Eq', ('pi', T1, ('pi', V(0), ('pi', V(0), T0))), [('refl', [T1, ('var', 0)], [('var', 0), ('var', 1), ('var', 1)])])
        return kk, lambda: eq
    def second():
        kk, gg, _h = std_world(); return kk, lambda: gg['Eq']
    def nonexistent():
        kk, gg = pw.build_prop_world(); return kk, lambda: 99999
    def non_eq():
        kk, gg = pw.build_prop_world(); return kk, lambda: gg['And']
    def definition():
        kk, gg = pw.build_prop_world(); return kk, lambda: gg['Not']
    def clash():
        kk, gg = pw.build_prop_world(); kk.declare_unchecked(kk.mk_type(1), None, 'Quot.lift'); return kk, lambda: gg['Eq']
    attempt('outside the Prop/Type world (prop_elim off)', legacy, 'Prop/Type world')
    attempt('a second declaration', second, 'already declared')
    attempt('a global id that does not exist', nonexistent, 'no such global')
    attempt('a family that is not an equality (And)', non_eq, 'equality family')
    attempt('a definition (Not) instead of a family', definition, 'equality family')
    attempt('a name clash (a global called Quot.lift)', clash, 'already exists')
    attempt('an equality family with two constructors', with_family('Eq2', EQSORT(T1), [REFLC(T1), REFLC(T1, 'refl2')]), 'exactly one constructor')
    attempt('an equality family whose constructor proves every a = b', with_family('Eq3', EQSORT(T1), [('triv', [T1, ('var', 0), ('var', 0)], [('var', 0), ('var', 1), ('var', 2)])]), 'refl : (A : Type j)')
    attempt('an equality family at Type0 (j = 0)', with_family('Eq0', ('pi', T0, ('pi', V(0), ('pi', V(0), T0))), [REFLC(T0)]), 'j >= 1')
    attempt('an "equality" valued in Type1 instead of Prop', with_family('Eq1', ('pi', T1, ('pi', V(0), ('pi', V(0), T1))), [REFLC(T1)]), 'type (A : Type j)')

    # level generality: the same construction at j = 2 (neither the kernel nor the checker may have 1 hard-wired)
    k2 = fresh_k(); t7.declare_family_new(k2, 'True', T0, [('tt', [], [])]); t7.declare_family_new(k2, 'Eq2', EQSORT(T2), [REFLC(T2, 'refl2')])
    g2 = gmap(k2); k2.declare_quot(g2['Eq2']); g2 = gmap(k2); C2 = lambda n: ('const', g2[n])
    want2 = quot_types(C2('Eq2'), C2('Quot'), C2('Quot.mk'), 2)
    same2 = {n: k2.global_info(g2[n])[1] == build_new(k2, HY(t)) for n, t in want2.items()}
    print("   level 2: types equal the specification: %s" % same2)
    expect(all(same2.values()), 'A: level-2 Quot types differ from the specification')
    chk2 = ic.Checker(t8.build_env(k2), params=True, prop_elim=True, fuel=2_000_000)
    expect(chk2.quot is not None and not chk2.check_definitions(), 'A: the checker rejected the level-2 world')
    total1 = lam('a', T1, lam('b', T1, C2('True')))
    lft = ap(C2('Quot.lift'), T1, total1, T1, lam('a1', T1, T0), lam('a2', T1, lam('b2', T1, lam('p2', C2('True'), ap(C2('refl2'), T1, T0)))))
    stmt2 = ap(C2('Eq2'), T1, ap(lft, ap(C2('Quot.mk'), T1, total1, T0)), T0)
    ok2 = kernel_ok(k2, ap(C2('refl2'), T1, T0), stmt2) and indep_ok(chk2, ap(C2('refl2'), T1, T0), stmt2)
    wrong2 = ap(C2('refl2'), T1, arrow(T0, T0)); bad2 = kernel_ok(k2, wrong2, stmt2) or indep_ok(chk2, wrong2, stmt2)
    print("   level 2: lift (mk _) computes (both checkers accept): %s; a wrong value is refused by both: %s" % (ok2, not bad2))
    expect(ok2 and not bad2, 'A: level-2 computation')

# ================================================================================================================ the cases (B, C, D share one world)
class Terms:
    def __init__(self, k, g):
        self.k, self.g = k, g; C = lambda n: ('const', g[n]); self.C = C
        self.NAT, self.ZERO, self.S, self.TRUE, self.TT, self.FALSE = C('Nat'), C('Zero'), C('Succ'), C('True'), C('tt'), C('False')
        self.EQ, self.REFL, self.ADD, self.TRANSPORT = C('Eq'), C('refl'), C('add'), C('transport')
        self.Q, self.MK, self.LIFT, self.IND, self.SOUND = C('Quot'), C('Quot.mk'), C('Quot.lift'), C('Quot.ind'), C('Quot.sound')
        NAT = self.NAT
        self.total = lam('a', NAT, lam('b', NAT, self.TRUE))
        self.eqn = lam('a', NAT, lam('b', NAT, self.eq(NAT, 'a', 'b')))
        self.QT, self.QE = self.quot(NAT, self.total), self.quot(NAT, self.eqn)
        # K0 : Quot Nat total -> Nat, constant 0.   ID : Quot Nat (Eq Nat) -> Nat, the identity.   ID2 : Quot Nat total -> Nat -> Nat, returns its second argument.
        self.K0 = ap(self.LIFT, NAT, self.total, NAT, lam('a1', NAT, self.ZERO), lam('a2', NAT, lam('b2', NAT, lam('p2', self.TRUE, self.refl(NAT, self.ZERO)))))
        self.ID = ap(self.LIFT, NAT, self.eqn, NAT, lam('a1', NAT, 'a1'), lam('a2', NAT, lam('b2', NAT, lam('p2', self.eq(NAT, 'a2', 'b2'), 'p2'))))
        NN = arrow(NAT, NAT); self.NN = NN
        self.ID2 = ap(self.LIFT, NAT, self.total, NN, lam('a1', NAT, lam('n1', NAT, 'n1')), lam('a2', NAT, lam('b2', NAT, lam('p2', self.TRUE, self.refl(NN, lam('n2', NAT, 'n2'))))))
        self.LS = ap(self.LIFT, NAT, self.eqn, NAT, lam('a1', NAT, ap(self.S, 'a1')), lam('a2', NAT, lam('b2', NAT, lam('p2', self.eq(NAT, 'a2', 'b2'), ap(C('cong'), 'a2', 'b2', 'p2')))))
    def nat(self, n): return functools.reduce(lambda t, _: ap(self.S, t), range(n), self.ZERO)
    def eq(self, A, a, b): return ap(self.EQ, A, a, b)
    def refl(self, A, a): return ap(self.REFL, A, a)
    def quot(self, A, r): return ap(self.Q, A, r)
    def mk(self, A, r, a): return ap(self.MK, A, r, a)

def funext_statement(T):
    EQ = T.EQ
    return pi('A', T1, pi('B', arrow('A', T1), pi('f', pi('x1', 'A', ap('B', 'x1')), pi('g', pi('x2', 'A', ap('B', 'x2')),
              pi('h', pi('x3', 'A', ap(EQ, ap('B', 'x3'), ap('f', 'x3'), ap('g', 'x3'))), ap(EQ, pi('x4', 'A', ap('B', 'x4')), 'f', 'g'))))))

def funext_proof(T):
    """funext from Quot.sound:  extfun : Quot Fn rel -> Fn,  extfun (mk f) == f  (computation + eta);  congrArg extfun (sound f g h) : extfun (mk f) = extfun (mk g)"""
    EQ, REFL, TRANSPORT, Q, MK, LIFT, SOUND = T.EQ, T.REFL, T.TRANSPORT, T.Q, T.MK, T.LIFT, T.SOUND
    Fn = pi('x5', 'A', ap('B', 'x5'))
    rel = lam('u', Fn, lam('v', Fn, pi('x6', 'A', ap(EQ, ap('B', 'x6'), ap('u', 'x6'), ap('v', 'x6')))))
    QF = ap(Q, Fn, rel)
    extfun = lam('z', QF, lam('x7', 'A', ap(LIFT, Fn, rel, ap('B', 'x7'), lam('u1', Fn, ap('u1', 'x7')),
                  lam('u2', Fn, lam('v2', Fn, lam('p2', ap(rel, 'u2', 'v2'), ap('p2', 'x7')))), 'z')))
    mkf = ap(MK, Fn, rel, 'f')
    return lam('A', T1, lam('B', arrow('A', T1), lam('f', pi('x1', 'A', ap('B', 'x1')), lam('g', pi('x2', 'A', ap('B', 'x2')),
           lam('h', pi('x3', 'A', ap(EQ, ap('B', 'x3'), ap('f', 'x3'), ap('g', 'x3'))),
               ap(TRANSPORT, QF, lam('z1', QF, ap(EQ, Fn, ap(extfun, mkf), ap(extfun, 'z1'))), mkf, ap(MK, Fn, rel, 'g'), ap(SOUND, Fn, rel, 'f', 'g', 'h'),
                  ap(REFL, Fn, ap(extfun, mkf))))))))

def build_cases(T):
    NAT, ZERO, S, TRUE, TT, FALSE = T.NAT, T.ZERO, T.S, T.TRUE, T.TT, T.FALSE
    EQ, REFL, ADD, TRANSPORT = T.EQ, T.REFL, T.ADD, T.TRANSPORT
    Q, MK, LIFT, IND, SOUND = T.Q, T.MK, T.LIFT, T.IND, T.SOUND
    nat, eq, refl, quot, mk = T.nat, T.eq, T.refl, T.quot, T.mk
    total, eqn, QT, QE, K0, ID, ID2, LS, NN = T.total, T.eqn, T.QT, T.QE, T.K0, T.ID, T.ID2, T.LS, T.NN
    cases = []
    def case(name, proof, stmt, expected): cases.append((name, proof, stmt, expected))
    # ---- computation: accepted
    case('lift K0 (mk 1) = 0, by refl', refl(NAT, ZERO), eq(NAT, ap(K0, mk(NAT, total, nat(1))), ZERO), True)
    case('lift ID (mk 5) = 5', refl(NAT, nat(5)), eq(NAT, ap(ID, mk(NAT, eqn, nat(5))), nat(5)), True)
    case('over-application: ID2 (mk 5) 7 = 7', refl(NAT, nat(7)), eq(NAT, ap(ID2, mk(NAT, total, nat(5)), nat(7)), nat(7)), True)
    case('partial application: ID2 (mk 5) = fun n => n', refl(NN, lam('n3', NAT, 'n3')), eq(NN, ap(ID2, mk(NAT, total, nat(5))), lam('n4', NAT, 'n4')), True)
    case('the argument of mk is computed first: ID (mk (1+1)) = 2', refl(NAT, nat(2)), eq(NAT, ap(ID, mk(NAT, eqn, ap(ADD, nat(1), nat(1)))), nat(2)), True)
    case('mk is a congruence: mk (1+1) = mk 2', refl(QE, mk(NAT, eqn, nat(2))), eq(QE, mk(NAT, eqn, ap(ADD, nat(1), nat(1))), mk(NAT, eqn, nat(2))), True)
    case('under a binder: forall n, K0 (mk n) = 0', lam('n', NAT, refl(NAT, ZERO)), pi('n', NAT, eq(NAT, ap(K0, mk(NAT, total, 'n')), ZERO)), True)
    case('lift with f = Succ: LS (mk 2) = 3', refl(NAT, nat(3)), eq(NAT, ap(LS, mk(NAT, eqn, nat(2))), nat(3)), True)
    case('nested: K0 (mk (ID (mk 3))) = 0', refl(NAT, ZERO), eq(NAT, ap(K0, mk(NAT, total, ap(ID, mk(NAT, eqn, nat(3))))), ZERO), True)
    case('nested the other way: ID (mk (LS (mk 1))) = 2', refl(NAT, nat(2)), eq(NAT, ap(ID, mk(NAT, eqn, ap(LS, mk(NAT, eqn, nat(1))))), nat(2)), True)
    case('lift into a universe: Quot Nat total -> Type0', ap(LIFT, NAT, total, T0, lam('a1', NAT, TRUE), lam('a2', NAT, lam('b2', NAT, lam('p2', TRUE, refl(T0, TRUE))))), arrow(QT, T0), True)
    # ---- computation: rejected
    case('ID (mk 5) = 0 is false', refl(NAT, ZERO), eq(NAT, ap(ID, mk(NAT, eqn, nat(5))), ZERO), False)
    case('K0 (mk 1) = 1 is false', refl(NAT, nat(1)), eq(NAT, ap(K0, mk(NAT, total, nat(1))), nat(1)), False)
    case('over-application returns the SECOND argument: ID2 (mk 5) 7 = 5 is false', refl(NAT, nat(5)), eq(NAT, ap(ID2, mk(NAT, total, nat(5)), nat(7)), nat(5)), False)
    case('stuck on a variable: forall q, ID q = 0', lam('q', QE, refl(NAT, ZERO)), pi('q', QE, eq(NAT, ap(ID, 'q'), ZERO)), False)
    case('stuck on a non-mk neutral with three arguments', lam('w', arrow(NAT, arrow(NAT, arrow(NAT, QE))), refl(NAT, ZERO)),
         pi('w', arrow(NAT, arrow(NAT, arrow(NAT, QE))), eq(NAT, ap(ID, ap('w', ZERO, ZERO, ZERO)), ZERO)), False)
    case('stuck: forall q, K0 q = 0 is NOT definitional (it is a theorem, below)', lam('q', QT, refl(NAT, ZERO)), pi('q', QT, eq(NAT, ap(K0, 'q'), ZERO)), False)
    case('mk 0 = mk 1 is not definitional in Quot Nat (Eq Nat)', refl(QE, mk(NAT, eqn, ZERO)), eq(QE, mk(NAT, eqn, ZERO), mk(NAT, eqn, nat(1))), False)
    # ---- typing of lift: the twin pairs (an accepted term, then the same term with ONE thing wrong)
    bad_h = lam('a2', NAT, lam('b2', NAT, lam('p2', TRUE, refl(NAT, 'a2'))))
    case('twin: lift of the constant function over the total relation is a function Quot Nat total -> Nat', K0, arrow(QT, NAT), True)
    case('twin: lift of the identity over Eq Nat (h := the relation proof itself) is a function', ID, arrow(QE, NAT), True)
    case('lift of the identity over the TOTAL relation with h := fun a b _ => refl a  (h is wrong: a = b is not provable)', ap(LIFT, NAT, total, NAT, lam('a1', NAT, 'a1'), bad_h), arrow(QT, NAT), False)
    case('lift with h := Zero (not a proof at all)', ap(LIFT, NAT, total, NAT, lam('a1', NAT, 'a1'), ZERO), arrow(QT, NAT), False)
    case('lift with f of the wrong type (f a := True, B := Nat)', ap(LIFT, NAT, total, NAT, lam('a1', NAT, TRUE), lam('a2', NAT, lam('b2', NAT, lam('p2', TRUE, refl(NAT, ZERO))))), arrow(QT, NAT), False)
    case('lift into a PROPOSITION type True : Type0 is not Type1', ap(LIFT, NAT, total, TRUE, lam('a1', NAT, TT), lam('a2', NAT, lam('b2', NAT, lam('p2', TRUE, refl(TRUE, TT))))), arrow(QT, TRUE), False)
    case('K0 applied to a natural number, not a quotient element', ap(K0, ZERO), NAT, False)
    case('K0 applied to an element of a DIFFERENT quotient (Quot Nat (Eq Nat))', ap(K0, mk(NAT, eqn, nat(1))), NAT, False)
    case('mk of an element of the wrong type', mk(NAT, total, TRUE), QT, False)
    # ---- ind and sound
    case('sound 0 1 : mk 0 = mk 1 in Quot Nat total', ap(SOUND, NAT, total, ZERO, nat(1), TT), eq(QT, mk(NAT, total, ZERO), mk(NAT, total, nat(1))), True)
    case('sound 0 1 over Eq Nat is refused (the premise Eq Nat 0 1 is false)', ap(SOUND, NAT, eqn, ZERO, nat(1), refl(NAT, ZERO)), eq(QE, mk(NAT, eqn, ZERO), mk(NAT, eqn, nat(1))), False)
    case('sound with tt where the relation is Eq Nat', ap(SOUND, NAT, eqn, ZERO, nat(1), TT), eq(QE, mk(NAT, eqn, ZERO), mk(NAT, eqn, nat(1))), False)
    case('ind into a proposition: forall q, True', lam('q', QT, ap(IND, NAT, total, lam('q1', QT, TRUE), lam('a', NAT, TT), 'q')), pi('q', QT, TRUE), True)
    case('ind with a motive into TYPE (Nat), refused: it would define a function out of the quotient', lam('q', QT, ap(IND, NAT, total, lam('q1', QT, NAT), lam('a', NAT, 'a'), 'q')), pi('q', QT, NAT), False)
    case('ind with a motive into Type1 (Type0 itself), refused', lam('q', QT, ap(IND, NAT, total, lam('q1', QT, T0), lam('a', NAT, TRUE), 'q')), pi('q', QT, T1), False)
    # ---- eta (F27)
    case('eta: fun x => f x == f', lam('f', NN, refl(NN, 'f')), pi('f', NN, eq(NN, lam('x', NAT, ap('f', 'x')), 'f')), True)
    case('eta, the other direction: f == fun x => f x', lam('f', NN, refl(NN, 'f')), pi('f', NN, eq(NN, 'f', lam('x', NAT, ap('f', 'x')))), True)
    case('eta, two arguments: fun x y => f x y == f', lam('f', arrow(NAT, NN), refl(arrow(NAT, NN), 'f')),
         pi('f', arrow(NAT, NN), eq(arrow(NAT, NN), lam('x', NAT, lam('y', NAT, ap('f', 'x', 'y'))), 'f')), True)
    case('eta for the recursive function add: fun a => add a == add', refl(arrow(NAT, NN), ADD), eq(arrow(NAT, NN), lam('a', NAT, ap(ADD, 'a')), ADD), True)
    case('eta for add, two arguments', refl(arrow(NAT, NN), ADD), eq(arrow(NAT, NN), lam('a', NAT, lam('b', NAT, ap(ADD, 'a', 'b'))), ADD), True)
    case('NOT eta: fun x => f x == g', lam('f', NN, lam('g', NN, refl(NN, 'f'))), pi('f', NN, pi('g', NN, eq(NN, lam('x', NAT, ap('f', 'x')), 'g'))), False)
    case('NOT eta: fun x => S (f x) == f', lam('f', NN, refl(NN, 'f')), pi('f', NN, eq(NN, lam('x', NAT, ap(S, ap('f', 'x'))), 'f')), False)
    case('NOT eta: fun x => f (S x) == f', lam('f', NN, refl(NN, 'f')), pi('f', NN, eq(NN, lam('x', NAT, ap('f', ap(S, 'x'))), 'f')), False)
    # ---- C. the conditional contradiction that the check of h exists to prevent
    H_ty = pi('a', NAT, pi('b', NAT, eq(NAT, 'a', 'b')))
    def contradiction(hproof):
        F = ap(LIFT, NAT, total, NAT, lam('a1', NAT, 'a1'), hproof)
        return ap(TRANSPORT, QT, lam('x', QT, eq(NAT, ap(F, mk(NAT, total, ZERO)), ap(F, 'x'))), mk(NAT, total, ZERO), mk(NAT, total, nat(1)), ap(SOUND, NAT, total, ZERO, nat(1), TT),
                  refl(NAT, ap(F, mk(NAT, total, ZERO))))
    case('UNDER the false hypothesis H : forall a b, a = b we CAN derive 0 = 1 (the rule and sound work together)',
         lam('H', H_ty, contradiction(lam('a2', NAT, lam('b2', NAT, lam('p2', TRUE, ap('H', 'a2', 'b2')))))), pi('H', H_ty, eq(NAT, ZERO, nat(1))), True)
    case('...but a made-up h gives no closed proof of 0 = 1', contradiction(bad_h), eq(NAT, ZERO, nat(1)), False)
    case('...nor does h := Zero', contradiction(ZERO), eq(NAT, ZERO, nat(1)), False)
    # ---- D. real use
    case('all elements of Quot Nat total are equal (ind + sound)',
         lam('q1', QT, lam('q2', QT, ap(IND, NAT, total, lam('x1', QT, eq(QT, 'x1', 'q2')),
              lam('a', NAT, ap(IND, NAT, total, lam('x2', QT, eq(QT, mk(NAT, total, 'a'), 'x2')), lam('b', NAT, ap(SOUND, NAT, total, 'a', 'b', TT)), 'q2')), 'q1'))),
         pi('q1', QT, pi('q2', QT, eq(QT, 'q1', 'q2'))), True)
    liftmk = ap(LIFT, NAT, total, QT, ap(MK, NAT, total), lam('a2', NAT, lam('b2', NAT, lam('p2', TRUE, ap(SOUND, NAT, total, 'a2', 'b2', 'p2')))))
    case('the eta law for Quot: lift mk q = q', lam('q', QT, ap(IND, NAT, total, lam('x', QT, eq(QT, ap(liftmk, 'x'), 'x')), lam('a', NAT, refl(QT, mk(NAT, total, 'a'))), 'q')),
         pi('q', QT, eq(QT, ap(liftmk, 'q'), 'q')), True)
    case('K0 q = 0 for every q (ind: the computation rule at the mk case)', lam('q', QT, ap(IND, NAT, total, lam('x', QT, eq(NAT, ap(K0, 'x'), ZERO)), lam('a', NAT, refl(NAT, ZERO)), 'q')),
         pi('q', QT, eq(NAT, ap(K0, 'q'), ZERO)), True)
    case('...and the same with the wrong value (K0 q = 1) has no such proof', lam('q', QT, ap(IND, NAT, total, lam('x', QT, eq(NAT, ap(K0, 'x'), nat(1))), lam('a', NAT, refl(NAT, nat(1))), 'q')),
         pi('q', QT, eq(NAT, ap(K0, 'q'), nat(1))), False)
    fx = funext_statement(T)
    case('FUNEXT derived from Quot.sound (needs the computation rule AND eta)', funext_proof(T), fx, True)
    return cases

def section_bcd():
    k, g, handle = std_world(); T = Terms(k, g); cases = build_cases(T)
    chk = ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=4_000_000)
    acc = rej = dis = 0; lines = []
    for name, proof, stmt, expected in cases:
        st_k, st_i = kernel_type_ok(k, stmt), indep_type_ok(chk, stmt)
        kv, iv = kernel_ok(k, proof, stmt), indep_ok(chk, proof, stmt)
        good = st_k and st_i and kv == expected and iv == expected
        if expected: acc += 1
        else: rej += 1
        if not good: dis += 1; lines.append("   DISAGREEMENT %-90s statement type-correct: kernel %s indep %s | proof: kernel %s indep %s | expected %s" % (name[:90], st_k, st_i, kv, iv, expected))
        expect(good, 'B: ' + name)
    print("B/C/D. %d cases (%d expected accepted, %d expected rejected): the kernel and the independent checker both gave the expected verdict on %d of them" % (len(cases), acc, rej, len(cases) - dis))
    for l in lines: print(l)
    expect(acc >= 20 and rej >= 18, 'B: too few cases (a vacuous table)')

    # the derived funext: declare it through the CHECKED path, re-check the whole world, audit its dependencies, compare its statement with the classical axiom's
    k.declare_checked(build_new(k, HY(funext_statement(T))), build_new(k, HY(funext_proof(T))), 'funext')
    g2 = gmap(k); chk2 = ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=4_000_000); bad = chk2.check_definitions()
    axioms, quots = audit.dependencies_of_global(k, g2['funext'])
    print("D. funext declared as a THEOREM (kernel-checked); independent re-check of the %d globals: %d rejected; it rests on axioms %s and quotient constants %s" % (len(g2), len(bad), sorted(axioms), sorted(quots)))
    expect(not bad, 'D: independent re-check rejected %s' % bad[:3])
    expect(axioms == set() and 'Quot.sound' in quots and quots <= {'Quot', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound'}, 'D: funext dependencies %s %s' % (axioms, quots))
    kc, gc = pw.build_prop_world(classical=True)
    name_of = lambda kk: {gid: nm for gid, nm, _, _ in kk.list_globals()}
    def named(kk, tree):
        nm = name_of(kk)
        def go(t):
            if t[0] == 'const': return ('const', nm[t[1]])
            if t[0] in ('app',): return ('app', go(t[1]), go(t[2]))
            if t[0] in ('lam', 'pi'): return (t[0], go(t[1]), go(t[2]))
            return t
        return go(tree)
    axiom_ty = named(kc, parse_raw(kc.dump_raw(kc.global_info(gc['funext'])[1]), True))
    thm_ty = named(k, parse_raw(k.dump_raw(k.global_info(g2['funext'])[1]), True))
    print("   the theorem's statement is the classical axiom's statement: %s" % (axiom_ty == thm_ty))
    expect(axiom_ty == thm_ty, 'D: derived funext has a different statement from the classical axiom')
    # trusted-base accounting for the whole Quot world: no axiom at all, exactly five quotient constants
    kinds = {}
    for gid in range(k.num_globals()): kinds.setdefault(audit.kind(k, gid), []).append(k.global_info(gid)[0])
    print("   trusted-base accounting of the quotient world (%d globals): axioms %s, quotient constants %s" % (k.num_globals(), kinds.get('axiom', []), kinds.get('quot', [])))
    expect(kinds.get('axiom', []) == [] and sorted(kinds.get('quot', [])) == sorted(['Quot', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound']), 'D: trusted base of the quotient world')

# ================================================================================================================ B2. neutrals headed by GLOBAL axioms, and eta in conversion
def section_b2():
    """the reduction rule must fire only on Quot.mk. A local variable can never be mistaken for it, but a GLOBAL axiom applied to three arguments has the same neutral shape as `Quot.mk A r a`"""
    k, g, handle = std_world(); T = Terms(k, g); NAT, ZERO, ID, QE = T.NAT, T.ZERO, T.ID, T.QE
    k.declare_unchecked(build_new(k, HY(arrow(NAT, arrow(NAT, arrow(NAT, QE))))), None, 'wq3'); k.declare_unchecked(build_new(k, HY(QE)), None, 'wq0'); g = gmap(k); C = lambda n: ('const', g[n])
    w3 = ap(C('wq3'), ZERO, ZERO, ZERO)
    cases = [('ID (wq3 0 0 0) = 0 is NOT definitional: the global wq3 is not Quot.mk, so the lift is stuck', T.refl(NAT, ZERO), T.eq(NAT, ap(ID, w3), ZERO), False),
             ('ID wq0 = 0 is NOT definitional (an opaque element of Quot Nat (Eq Nat))', T.refl(NAT, ZERO), T.eq(NAT, ap(ID, C('wq0')), ZERO), False),
             ('twin: a stuck lift is still equal to itself', T.refl(NAT, ap(ID, w3)), T.eq(NAT, ap(ID, w3), ap(ID, w3)), True),
             ('twin: the real Quot.mk with three arguments does fire', T.refl(NAT, T.nat(3)), T.eq(NAT, ap(ID, T.mk(NAT, T.eqn, T.nat(3))), T.nat(3)), True)]
    chk = ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=2_000_000); bad = 0
    for name, proof, stmt, expected in cases:
        kv, iv = kernel_ok(k, proof, stmt), indep_ok(chk, proof, stmt)
        if not (kv == expected and iv == expected): bad += 1; print("   DISAGREEMENT %s: kernel %s indep %s expected %s" % (name, kv, iv, expected))
        expect(kv == expected and iv == expected, 'B2: ' + name)
    print("B2. global opaque neutrals: %d cases, %d wrong verdicts" % (len(cases), bad))
    # eta in CONVERSION (the type checker uses the unifier; conv_open is the other entry point and must agree)
    NN = T.NN; nn_ctx = ('f',)
    HYC = lambda t, ctx=(): build_new(k, l2h(lv(t, ctx), len(ctx)))
    eta1 = lam('x', NAT, ap('f', 'x')); eta_bad1 = lam('x', NAT, ap(T.S, ap('f', 'x'))); eta_bad2 = lam('x', NAT, ap('f', ap(T.S, 'x')))
    eta2 = lam('a', NAT, ap(T.ADD, 'a')); eta3 = lam('a', NAT, lam('b', NAT, ap(T.ADD, 'a', 'b')))
    res = {'fun x => f x  ==  f': k.conv_open(HYC(eta1, nn_ctx), HYC('f', nn_ctx), 1), 'f  ==  fun x => f x': k.conv_open(HYC('f', nn_ctx), HYC(eta1, nn_ctx), 1),
           'fun a => add a  ==  add': k.conv_open(HYC(eta2), HYC(T.ADD), 0), 'add  ==  fun a => add a': k.conv_open(HYC(T.ADD), HYC(eta2), 0),
           'fun a b => add a b  ==  add': k.conv_open(HYC(eta3), HYC(T.ADD), 0), 'add  ==  fun a b => add a b': k.conv_open(HYC(T.ADD), HYC(eta3), 0)}
    neg = {'fun x => S (f x)  ==  f': k.conv_open(HYC(eta_bad1, nn_ctx), HYC('f', nn_ctx), 1), 'f  ==  fun x => f (S x)': k.conv_open(HYC('f', nn_ctx), HYC(eta_bad2, nn_ctx), 1)}
    print("   conv_open eta: %s" % res); print("   conv_open non-eta (must be False): %s" % neg)
    expect(all(res.values()), 'C2: eta in conv_open: %s' % [n for n, v in res.items() if not v]); expect(not any(neg.values()), 'C2: conv_open accepted a non-eta pair')
    # eta for a RAW (local) recursive function. A global such as `add` evaluates to a lambda, a bare Fix node to an unapplied VFix: the kernel needs its own Closure-vs-VFix cases
    # (in conv_value for conv_open, in unify_values for the type checker), and the independent checker's eta rule must agree.
    mot = lam('_', NAT, NAT)
    FIXD = ('Fix', 0, 'self', NN, lam('n', NAT, match('n', mot, ZERO, lam('m', NAT, ap(T.S, ap(T.S, ap('self', 'm')))))))        # fix self. fun n. match n with 0 => 0 | S m => S (S (self m))
    etaF = lam('x', NAT, ap(FIXD, 'x')); etaF_bad1 = lam('x', NAT, ap(T.S, ap(FIXD, 'x'))); etaF_bad2 = lam('x', NAT, ap(FIXD, ap(T.S, 'x')))
    resF = {'fun x => F x  ==  F': k.conv_open(HYC(etaF), HYC(FIXD), 0), 'F  ==  fun x => F x': k.conv_open(HYC(FIXD), HYC(etaF), 0)}
    negF = {'fun x => S (F x)  ==  F': k.conv_open(HYC(etaF_bad1), HYC(FIXD), 0), 'F  ==  fun x => F (S x)': k.conv_open(HYC(FIXD), HYC(etaF_bad2), 0)}
    print("   conv_open eta for a raw Fix: %s ; non-eta (must be False): %s" % (resF, negF))
    expect(all(resF.values()), 'C2: eta for a raw Fix in conv_open: %s' % [n for n, v in resF.items() if not v]); expect(not any(negF.values()), 'C2: conv_open accepted a non-eta pair with a raw Fix')
    FN = T.NN
    cases_f = [('type checker eta: fun x => F x == F (raw Fix)', T.refl(FN, FIXD), T.eq(FN, etaF, FIXD), True), ('type checker eta: F == fun x => F x (raw Fix)', T.refl(FN, FIXD), T.eq(FN, FIXD, etaF), True),
               ('type checker eta, the proof mentions the lambda: refl (fun x => F x) : (fun x => F x) = F', T.refl(FN, etaF), T.eq(FN, etaF, FIXD), True),
               ('type checker eta, the proof mentions the lambda: refl (fun x => F x) : F = (fun x => F x)', T.refl(FN, etaF), T.eq(FN, FIXD, etaF), True),
               ('NOT eta: fun x => S (F x) == F', T.refl(FN, FIXD), T.eq(FN, etaF_bad1, FIXD), False), ('NOT eta: F == fun x => F (S x)', T.refl(FN, FIXD), T.eq(FN, FIXD, etaF_bad2), False)]
    for name, proof, stmt, expected in cases_f:
        kv, iv = kernel_ok(k, proof, stmt), indep_ok(chk, proof, stmt)
        if not (kv == expected and iv == expected): print("   DISAGREEMENT %s: kernel %s indep %s expected %s" % (name, kv, iv, expected))
        expect(kv == expected and iv == expected, 'B2: ' + name)

# ================================================================================================================ E. tampering
def section_e():
    k, g, handle = std_world(); T = Terms(k, g)
    names = {n: g[n] for n in ('Quot', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound')}
    ref_stmt = T.eq(T.NAT, ap(T.K0, T.mk(T.NAT, T.total, T.nat(1))), T.ZERO); ref_proof = T.refl(T.NAT, T.ZERO)
    def splice(t, i): return t[3] if i == 0 else ('pi', t[1], t[2], splice(t[3], i - 1))
    def map_conclusion(t, f): return ('pi', t[1], t[2], map_conclusion(t[3], f)) if t[0] == 'pi' else f(t)
    def respine(t, f):
        h, args = ic.spine(t); return ic.apply_args(h, f(args))
    def t_no_h(env): env.g[names['Quot.lift']].type = splice(env.g[names['Quot.lift']].type, 4)
    def t_vacuous_sound(env): env.g[names['Quot.sound']].type = map_conclusion(env.g[names['Quot.sound']].type, lambda c: respine(c, lambda a: [a[0], a[1], a[1]]))
    def dom_of(t, i): return t[2] if i == 0 else dom_of(t[3], i - 1)
    def set_dom(t, i, d): return ('pi', t[1], d, t[3]) if i == 0 else ('pi', t[1], t[2], set_dom(t[3], i - 1, d))
    def t_weak_h(env):          # lift : A r B f h q ; in h's conclusion replace  Eq B (f a) (f b)  by  Eq B (f a) (f a)
        lt = env.g[names['Quot.lift']].type
        env.g[names['Quot.lift']].type = set_dom(lt, 4, map_conclusion(dom_of(lt, 4), lambda c: respine(c, lambda a: [a[0], a[1], a[1]])))
    def t_ind_type_motive(env):
        it = env.g[names['Quot.ind']].type; pdom = dom_of(it, 2)
        env.g[names['Quot.ind']].type = set_dom(it, 2, ('pi', pdom[1], pdom[2], ('s', 1)))
    def t_second_mk(env): env.g[g['Nat']].quot_role = 'mk'
    def t_role_moved(env): env.g[names['Quot.mk']].quot_role = None; env.g[names['Quot.ind']].quot_role = 'mk'
    def t_eq_extra_ctor(env): env.g[g['Eq']].ctors.append(g['Zero'])
    def t_eq_wrong_refl(env): env.g[g['refl']].type = env.g[g['tt']].type
    tamper = {'Quot.lift without its h argument': t_no_h, 'Quot.sound concluding mk a = mk a': t_vacuous_sound, 'Quot.lift whose h only asks f a = f a': t_weak_h,
              'Quot.ind with a motive into Type1': t_ind_type_motive, 'a second global holding the mk role': t_second_mk, 'the mk role moved onto ind': t_role_moved,
              'Eq with an extra constructor': t_eq_extra_ctor, 'Eq whose constructor is not refl': t_eq_wrong_refl}
    base = ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=2_000_000)
    base_ok = indep_ok(base, ref_proof, ref_stmt)
    print("E. untampered environment: roles validated %s; the computation case is accepted: %s" % (base.quot is not None, base_ok))
    expect(base.quot is not None and base_ok, 'E: the baseline environment is not accepted')
    for name, fn in tamper.items():
        env = t8.build_env(k); fn(env)
        chk = ic.Checker(env, params=True, prop_elim=True, fuel=2_000_000)
        reported = [b for b in chk.check_definitions() if 'quotients' in b[2]]
        disabled = chk.quot is None and chk.quot_problem is not None
        computes = indep_ok(chk, ref_proof, ref_stmt)
        ok = disabled and len(reported) == 1 and not computes          # refused, reported once, and the reduction rule is OFF (so the computation case is no longer accepted)
        print("   %-44s -> quotient rule %s, definitions reported %-3d, case still computes: %-5s %s" % (name, 'DISABLED' if disabled else 'ENABLED', len(reported), computes, 'ok' if ok else 'FAILED'))
        expect(ok, 'E: tampered environment was not refused: ' + name)

# ================================================================================================================ F. F26
def section_f():
    k, g, handle = std_world(); T = Terms(k, g); NAT, FALSE = T.NAT, T.FALSE
    q0 = T.mk(NAT, T.total, T.ZERO)
    motive = lam('A', T1, lam('r', arrow('A', arrow('A', T0)), lam('q', ap(T.Q, 'A', 'r'), FALSE)))
    bad = match(q0, motive)
    chk = ic.Checker(t8.build_env(k), params=True, prop_elim=True, fuel=2_000_000)
    kv, iv = kernel_ok(k, bad, FALSE), indep_ok(chk, bad, FALSE)
    try: inf = bool(k.infer_open(build_new(k, HY(bad)), [])[0])
    except Exception: inf = False
    print("F. branchless match on a Quot value : False -> kernel %s, independent %s (both must refuse); by inference alone (explicit motive): %s" % (kv, iv, inf))
    expect(not inf, 'F26: infer() accepted a branchless match on a Quot value')
    expect(not kv and not iv, 'F26: a branchless match on a Quot value proves False')
    # an axiom `Foo : Type1` with an element `bar : Foo`
    k2, _ = pw.build_prop_world(); foo = k2.declare_unchecked(k2.mk_type(1), None, 'Foo'); bar = k2.declare_unchecked(k2.mk_const(foo), None, 'bar'); g2 = gmap(k2)
    FALSE2 = ('const', g2['False'])
    m2 = match(('const', g2['bar']), lam('x', ('const', g2['Foo']), FALSE2))
    chk2 = ic.Checker(t8.build_env(k2), params=True, prop_elim=True, fuel=2_000_000)
    kv2, iv2 = kernel_ok(k2, m2, FALSE2), indep_ok(chk2, m2, FALSE2)
    print("   branchless match on an element of an opaque type Foo : False -> kernel %s, independent %s" % (kv2, iv2))
    expect(not kv2 and not iv2, 'F26: a branchless match on an element of an opaque type proves False')
    # the tactics refuse as well
    for label, kk, ty_node in (('cases on a Quot hypothesis', k, build_new(k, HY(pi('q', T.QT, FALSE)))), ('cases on an opaque hypothesis', k2, build_new(k2, HY(pi('x', ('const', g2['Foo']), FALSE2))))):
        ps = kk.start_proof(ty_node); ps.intro('x')
        ok = refuses(lambda: ps.cases(0), 'not an inductive type'); print("   %-34s refused: %s" % (label, ok)); expect(ok, 'F26: ' + label)
        ps = kk.start_proof(ty_node); ps.intro('x')
        ok = refuses(lambda: ps.induct(0), 'not an inductive type'); print("   %-34s refused: %s" % (label.replace('cases', 'induct'), ok)); expect(ok, 'F26: ' + label.replace('cases', 'induct'))
    # empty inductives are still inductive (and still eliminate): False, an extra empty family
    k3, _ = pw.build_prop_world(); empty = t7.declare_family_new(k3, 'Empty', T0, [])
    flags = {n: k.is_inductive(g[n]) for n in ('Nat', 'False', 'Eq', 'And', 'Le', 'Quot', 'Quot.mk', 'Quot.lift', 'add', 'Not', 'refl')}
    print("   is_inductive flags: %s; a freshly declared empty family: %s" % (flags, k3.is_inductive(empty)))
    expect(flags == {'Nat': True, 'False': True, 'Eq': True, 'And': True, 'Le': True, 'Quot': False, 'Quot.mk': False, 'Quot.lift': False, 'add': False, 'Not': False, 'refl': False} and k3.is_inductive(empty), 'F26: inductive flags')
    g3 = gmap(k3); stmt = pi('x', ('const', g3['Empty']), ('const', g3['False']))
    ps = k3.start_proof(build_new(k3, HY(stmt))); ps.intro('x'); ps.cases(0)
    ok = ps.num_goals() == 0 and ps.is_complete() and k3.check_open(ps.extract(), build_new(k3, HY(stmt)), [])[0]
    print("   cases on an inhabitant of the empty inductive family still proves `Empty -> False`: %s" % ok)
    expect(ok, 'F26: empty inductive families must still eliminate')

# ================================================================================================================ G. the canary
def section_g(N):
    k, g, handle = std_world(); T = Terms(k, g); lvt = lambda t: lv(t)
    NAT, ZERO, S, EQ, FALSE = T.NAT, T.ZERO, T.S, T.EQ, T.FALSE
    false_stmts = {
        'Eq Nat 0 1': T.eq(NAT, ZERO, T.nat(1)),
        'False': FALSE,
        'mk 0 = mk 1 in Quot Nat (Eq Nat)': T.eq(T.QE, T.mk(NAT, T.eqn, ZERO), T.mk(NAT, T.eqn, T.nat(1))),
        'forall q : Quot Nat (Eq Nat), ID q = 0': pi('q', T.QE, T.eq(NAT, ap(T.ID, 'q'), ZERO)),
        'forall (A : Type1) r (a b : A), mk a = mk b': pi('A', T1, pi('r', arrow('A', arrow('A', T0)), pi('a', 'A', pi('b', 'A', T.eq(ap(T.Q, 'A', 'r'), ap(T.MK, 'A', 'r', 'a'), ap(T.MK, 'A', 'r', 'b')))))),
        'forall q : Quot Nat total, K0 q = 1': pi('q', T.QT, T.eq(NAT, ap(T.K0, 'q'), T.nat(1))),
        'all elements of Quot Nat (Eq Nat) are equal': pi('q1', T.QE, pi('q2', T.QE, T.eq(T.QE, 'q1', 'q2'))),
    }
    true_stmts = {
        'K0 (mk 3) = 0': T.eq(NAT, ap(T.K0, T.mk(NAT, T.total, T.nat(3))), ZERO),
        'forall n, ID (mk n) = n': pi('n', NAT, T.eq(NAT, ap(T.ID, T.mk(NAT, T.eqn, 'n')), 'n')),
        'forall q : Quot Nat total, True': pi('q', T.QT, T.TRUE),
        'forall n, K0 (mk n) = 0': pi('n', NAT, T.eq(NAT, ap(T.K0, T.mk(NAT, T.total, 'n')), ZERO)),
        'mk 0 = mk 1 in Quot Nat total': T.eq(T.QT, T.mk(NAT, T.total, ZERO), T.mk(NAT, T.total, T.nat(1))),
    }
    consts = [g[n] for n in ('refl', 'tt', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound', 'Zero', 'Succ', 'cong', 'symm', 'trans')]
    ind = {g[n] for n in ('Nat', 'Eq', 'Le', 'And', 'Or', 'Ex', 'False', 'True')}
    fl = {n: lv(t) for n, t in false_stmts.items()}; tr = {n: lv(t) for n, t in true_stmts.items()}
    tried, completed, verified, first = canlib.run(k, g, fl, consts, ind, N, 71)
    print("G. soundness canary (Quot constants available): %d random proof attempts on %d FALSE statements about quotients, %d completed, verified proof of a false statement: %s" % (tried, len(fl), completed, first))
    expect(first is None, 'G: canary proved a false statement: %s' % first)
    tried2, completed2, verified2, _ = canlib.run(k, g, tr, consts, ind, N, 72, stop_on_verified=False)
    print("   canary POWER (same generator, %d TRUE statements): %d attempts, %d completed, %d verified" % (len(tr), tried2, completed2, verified2))
    expect(verified2 >= 5, 'G: the canary has no power: %d true statements proved' % verified2)
    expect(completed2 == verified2, 'G: %d completed proof states did not verify' % (completed2 - verified2))

if __name__ == '__main__':
    t0 = time.time(); N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    section_a(); section_bcd(); section_b2(); section_e(); section_f(); section_g(N)
    print("elapsed %.0fs" % (time.time() - t0)); print("S16 QUOT GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
