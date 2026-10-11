"""S28 gate: a definition at the head of a hypothesis or a goal is unfolded ON SYNTAX, so that its arguments stay as the author wrote them.

`cases` on a hypothesis `h : Lt a b` where `Lt n m := Le (Succ n) m`, and `apply` of a lemma about `Le` to a goal `Lt a b`, have to see through the definition. They used to do it by evaluating the type
with every definition unfolded, which unfolds the definitions inside the arguments as well: `Lt two y` became `Le 3 y` with `two` written out as `Succ (Succ Zero)`, and (for the integers) `Z.zero` became
`Quot.mk ...`, so that a `rw` by a lemma about `Z.zero` found nothing to rewrite. The tactic layer now takes the head definition apart with the kernel's `unfold_head_folded` (one delta step on syntax: the
body with the arguments substituted) and reads the family and its arguments from that, so the arguments are exactly the terms of the statement.

Nothing here is trusted (the extracted proof is checked as a whole); this gate pins the PRESENTATION down, because everything downstream -- `rw` by pattern, `exact` of a hypothesis, a learner reading the
goal -- depends on it:

  A. `cases` on a hypothesis whose type is headed by a definition: the new hypotheses and goals mention the defined constant `two`, not its value, in both branches;
  B. `cases` on a hypothesis whose head is a definition of a definition (two steps of unfolding); one headed by a definition that returns a function (surplus arguments); one headed by the family itself;
  C. `apply` of a lemma whose conclusion is about the family to a goal headed by a definition: the new goal mentions `two`; the same through two definitions and through a definition with fewer lambdas than
     arguments;
  D. every proof is finished and admitted by the kernel.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(100000)
import itp_env_lnl as E
import lnl_base
from author import AuthorError
from tactics import Proof

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


VALUE_OF_TWO = 'Succ (Succ Zero)'


def build():
    w = lnl_base.build_base()
    Nat, PROP, Le, S, Z0 = w['Nat'], w.PROP, w['Le'], w['Succ'], w['Zero']
    w.define('two', Nat, S(S(Z0)))
    w.define('Lt', w.forall_([Nat, Nat], lambda n, m: PROP), w.fun([Nat, Nat], lambda n, m: Le(S(n), m)))                 # Lt n m := Le (S n) m
    w.define('Lt2', w.forall_([Nat, Nat], lambda n, m: PROP), w.fun([Nat, Nat], lambda n, m: w['Lt'](n, m)))             # a definition of a definition
    w.define('LtHalf', w.arrow(Nat, w.arrow(Nat, PROP)), w.fun([Nat], lambda n: Le(S(n))))                               # one lambda, and the body is a function: LtHalf n m has an argument more than the lambdas
    return w


def finish_ok(w, p, name):
    expect(p.done, '%s: the proof was not finished:\n%s' % (name, p))
    if not p.done: return
    try: w.k.declare_checked(p.stmt.n, p.term().n, name)
    except Exception as e: expect(False, '%s: the kernel rejected the proof: %s' % (name, str(e)[:200]))


def part_AB(w):
    Nat, Le, S, two = w['Nat'], w['Le'], w['Succ'], w['two']
    # forall y, D two y -> Le (Succ two) (Succ y)   for a head D that needs unfolding to reach Le (Succ two) y
    for dname in ('Lt', 'Lt2', 'LtHalf', 'Le(Succ)'):
        D = (lambda a, b: Le(S(a), b)) if dname == 'Le(Succ)' else w[dname]
        name = 'hu_cases_' + dname.replace('(', '_').replace(')', '')
        p = Proof(w, w.forall_([Nat, lambda y: D(two, y)], lambda y, h: Le(S(two), S(y))), name)
        p.intro('y', 'h'); before = str(p)
        expect('two' in before, '%s: setup: %s' % (dname, before))
        p.cases('h', names=['m', 'h2'])
        t = str(p)
        expect(VALUE_OF_TWO not in t, '%s: cases wrote the value of two out in a hypothesis or goal:\n%s' % (dname, t))
        expect(t.count('two') >= 3, '%s: the branches do not mention two (the goal of the first, the hypothesis and the goal of the second):\n%s' % (dname, t))
        expect('h2 : ' in t and 'h2 : Succ two <= m' in t.replace('Le (Succ two) m', 'Succ two <= m'), '%s: the field of the second branch is Succ two <= m:\n%s' % (dname, t))
        p.apply('le_S'); p.apply('le_n')                    # y = Succ two:  Succ two <= Succ (Succ two)
        p.apply('le_S'); p.apply('le_S'); p.exact('h2')     # y = Succ m, h2 : Succ two <= m
        finish_ok(w, p, name)


def part_C(w):
    Nat, Le, S, two = w['Nat'], w['Le'], w['Succ'], w['two']
    for dname in ('Lt', 'Lt2', 'LtHalf'):
        D = w[dname]
        name = 'hu_apply_' + dname
        # D two (Succ b) unfolds to Le (Succ two) (Succ b); le_S's conclusion is  Le ?n (Succ ?m),  so the new goal is  Le (Succ two) b
        p = Proof(w, w.forall_([Nat, lambda b: w['Le'](S(two), b)], lambda b, h: D(two, S(b))), name)
        p.intro('b', 'h'); p.apply('le_S')
        t = str(p)
        expect(p.n_goals == 1, '%s: apply le_S leaves one goal:\n%s' % (dname, t))
        expect(VALUE_OF_TWO not in t, '%s: apply wrote the value of two out in the new goal:\n%s' % (dname, t))
        expect('|- Succ two <= b' in t.replace('Le (Succ two) b', 'Succ two <= b'), '%s: the new goal is Succ two <= b:\n%s' % (dname, t))
        p.exact('h')
        finish_ok(w, p, name)


def main():
    w = build()
    part_AB(w); part_C(w)
    print('%d checks' % CHECKS[0])
    if FAIL:
        print('S28 HEAD UNFOLDING GATE: FAIL (%d)' % len(FAIL))
        for f in FAIL[:20]: print('   ', f)
        sys.exit(1)
    print('S28 HEAD UNFOLDING GATE: PASS')


if __name__ == '__main__':
    main()
