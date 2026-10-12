"""S29 gate: `apply` of a lemma to a goal that is itself a Pi (an implication, a negation, a universal statement).

`apply` used to peel EVERY binder of the lemma's type into a metavariable and unify what was left with the goal. For a goal `a <> b` (`Not (a = b)`, which is `a = b -> False`) and a lemma
`ne : forall a b, a < b -> a <> b` that peels the negation's premise as well, leaves `False` as the conclusion, and fails to match the goal. It now follows Lean's rule: the lemma's binders are peeled only as
far as they exceed the goal's (counting a binder hidden behind a definition, `Not P`, as one), so the conclusion that remains is a Pi like the goal; if that does not unify, every binder is peeled as before.

Nothing here is trusted (the extracted proof is checked); this pins down WHICH goals `apply` can close and what it leaves open -- the exact statements of the goals it leaves, because a learner (or `rw` by
pattern) reads them:

  A. a negation goal and a lemma that concludes a negation (with and without premises, with an explicit argument, written as an arrow, a hypothesis of the goal's own shape): closed, leaving only
     the lemma's premises -- no goal for the negation's premise;
  B. an implication goal and a lemma that concludes the same implication, a universal goal and a lemma that proves it, a lemma with more premises than the goal has binders (the surplus ones, the
     FIRST ones, become goals), the same through a definition that hides the Pi (`Imp2 A B`, `AllLe`), and a lemma whose conclusion is a bare variable (`P -> P`: the tight reading closes the goal);
  C. a negation goal and a lemma that concludes `False` from the negated statement (the premise IS the goal's binder);
  D. goals that are not Pis, and lemmas that do not fit: the lemma's negation premise is still peeled when the goal is `False`; a lemma with a Pi conclusion applied to a non-Pi goal peels it; a lemma whose
     conclusion is a bare variable still applies to a Pi goal (the fallback: every binder peeled, the goal becomes the variable); an `apply` that cannot unify fails and leaves the state as it was;
  E. a binder hidden behind a RECURSIVE type-level function (not a definition) is still peeled when the goal is not a Pi; the kernel call WITHOUT `simp` is the reference's (the rule is the tactic layer's);
     every proof is finished and admitted;
  F. a lemma given an EXPLICIT PREDICATE (`apply nat_ind (fun n => n <= n -> n <= n)`): its conclusion `P n` is then a Pi hidden behind the lambda, so the smallest count of binders to peel does not fit and peeling
     every binder (the hidden one too) leaves a bare predicate; Lean tries every count from the smallest up to the lemma's own number of binders, and so does `apply` now (finding F41).
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(100000)
import itp_env_lnl as E
import lnl_base
from lnl_base import thm
from author import AuthorError
from tactics import Proof

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


def build():
    w = lnl_base.build_base()
    Nat, PROP, Le, S, Z0, Eq, F = w['Nat'], w.PROP, w['Le'], w['Succ'], w['Zero'], w.Eq, w['False']
    # a lemma that concludes a negation, with and without premises
    @thm(w, 'succ_ne_zero', [Nat], lambda n: ~Eq(S(n), Z0))
    def _(p): p.intro('n', 'h'); p.cases('h')

    @thm(w, 'ne_symm', [Nat, Nat, lambda a, b: ~Eq(a, b)], lambda a, b, h: ~Eq(b, a))
    def _(p): p.intro('a', 'b', 'h', 'e'); p.apply('h'); p.apply('symm'); p.exact('e')

    # lemmas that conclude an implication / a universal statement: one binder, one premise, two premises
    @thm(w, 'le_succ', [Nat], lambda n: Le(n, S(n)))
    def _(p): p.intro('n'); p.apply('le_S'); p.apply('le_n')

    @thm(w, 'le_to_succ', [Nat, lambda n: Le(n, n)], lambda n, h: Le(n, S(n)))
    def _(p): p.intro('n', 'h'); p.apply('le_S'); p.exact('h')

    @thm(w, 'le_to_succ2', [Nat, lambda n: Le(n, n), lambda n, a: Le(n, n)], lambda n, a, b: Le(n, S(n)))
    def _(p): p.intro('n', 'a', 'b'); p.apply('le_S'); p.exact('a')

    # lemmas that conclude False from the negated statement
    @thm(w, 'succ_zero_false', [Nat, lambda n: Eq(S(n), Z0)], lambda n, h: F)
    def _(p): p.intro('n', 'h'); p.cases('h')

    @thm(w, 'absurd_succ', [Nat, lambda n: Le(n, n), lambda n, a: Eq(S(n), Z0)], lambda n, a, h: F)
    def _(p): p.intro('n', 'a', 'h'); p.cases('h')

    # a lemma whose conclusion is a bare variable, and one that is a variable under a premise
    @thm(w, 'id_fn', [PROP, lambda P: P], lambda P, x: P)
    def _(p): p.intro('P', 'x'); p.exact('x')

    # induction, and a lemma that returns its predicate: both are applied to an EXPLICIT predicate in part F
    @thm(w, 'nat_ind', [w.arrow(Nat, PROP), lambda P: P(Z0), lambda P, h0: w.forall_([Nat, lambda n: P(n)], lambda n, h: P(S(n)))], lambda P, h0, hs: w.forall_([Nat], lambda n: P(n)))
    def _(p):
        p.intro('P', 'h0', 'hs', 'n'); p.induct('n', names=['m', 'ih']); p.exact('h0'); p.apply('hs'); p.exact('ih')

    @thm(w, 'id_pred', [w.arrow(Nat, PROP), Nat, lambda P, n: P(n)], lambda P, n, h: P(n))
    def _(p): p.intro('P', 'n', 'h'); p.exact('h')

    # definitions that hide the Pi of a goal: an implication, and a universal statement
    w.define('Imp2', w.forall_([PROP, PROP], lambda a, b: PROP), w.fun([PROP, PROP], lambda a, b: w.arrow(a, b)))
    w.define('AllLe', PROP, w.forall_([Nat], lambda n: Le(n, S(n))))
    # a RECURSIVE type-level function whose value is a Pi:  Arrows (S n) A B = A -> Arrows n A B
    w.recursive('Arrows', w.forall_([Nat, PROP, PROP], lambda n, A, B: PROP),
                lambda Ar: w.fun([Nat, PROP, PROP], lambda n, A, B: w.match(n, lambda _: PROP, [B, lambda n1: w.arrow(A, Ar(n1, A, B))])))
    return w


def finish_ok(w, p, name):
    expect(p.done, '%s: the proof was not finished:\n%s' % (name, p))
    if not p.done: return
    try: w.k.declare_checked(p.stmt.n, p.term().n, name)
    except Exception as e: expect(False, '%s: the kernel rejected the proof: %s' % (name, str(e)[:200]))


def n_metas(w):
    """how many metavariables exist (the id the next one gets); asking creates one, which the callers allow for"""
    return w.k.fresh_meta(w.k.mk_type(0), 0)


def targets(p):
    """the target of every open goal, as printed"""
    return [str(p.goal(i)).split('|- ')[-1] for i in range(p.n_goals)]


def run_case(w, name, stmt, intros, apply_args, left, close=()):
    """prove `stmt`: introduce `intros`, `apply(*apply_args)`; the goals left must be exactly `left` (their targets as printed); `close` is the list of scripts closing them, in order"""
    p = Proof(w, stmt, name)
    if intros: p.intro(*intros)
    try: p.apply(*apply_args)
    except (AuthorError, E.ITPError) as e:
        expect(False, '%s: apply %s failed:\n%s' % (name, ' '.join(map(str, apply_args)), str(e)[:300])); return p
    got = targets(p)
    expect(got == left, '%s: the goals left are %r, expected %r' % (name, got, left))
    for script in close:
        if p.done: expect(False, '%s: fewer goals than closing scripts' % name); break
        try: script(p)
        except AuthorError as e: expect(False, '%s: closing a goal failed:\n%s' % (name, str(e)[:300])); return p
    finish_ok(w, p, name)
    return p


def part_A(w):
    Nat, Le, S, Z0, Eq, F = w['Nat'], w['Le'], w['Succ'], w['Zero'], w.Eq, w['False']
    # A1: a negation lemma with a premise: only the premise is left, and as the lemma wrote it (a negation, not unfolded to  (x = y) -> False)
    run_case(w, 'ap_ne_symm', w.forall_([Nat, Nat, lambda x, y: ~Eq(x, y)], lambda x, y, h: ~Eq(y, x)), ['x', 'y', 'h'], ['ne_symm'],
             ['Not (x = y)'], [lambda p: p.exact('h')])
    # A2/A3: a negation lemma with no premise: closed outright; the same with its argument given
    run_case(w, 'ap_succ_ne_zero', w.forall_([Nat], lambda k: ~Eq(S(k), Z0)), ['k'], ['succ_ne_zero'], [])
    run_case(w, 'ap_succ_ne_zero_arg', w.forall_([Nat], lambda k: ~Eq(S(k), Z0)), ['k'], ['succ_ne_zero', 'k'], [])
    # A4: the goal is written as the arrow that the negation lemma's conclusion unfolds to
    run_case(w, 'ap_succ_ne_zero_arrow', w.forall_([Nat], lambda k: w.arrow(Eq(S(k), Z0), F)), ['k'], ['succ_ne_zero'], [])
    # A5: a hypothesis of the goal's own shape, applied (the premise of the negation is NOT peeled into a new goal)
    run_case(w, 'ap_neg_hyp', w.forall_([w.PROP, lambda P: ~P], lambda P, h: ~P), ['P', 'h'], ['h'], [])
    # A6: a hypothesis that is a negation, applied to a goal that is not a Pi: the premise is peeled as before
    run_case(w, 'ap_neg_hyp_false', w.forall_([Nat, Nat, lambda x, y: Eq(x, y), lambda x, y, e: ~Eq(x, y)], lambda x, y, e, h: F), ['x', 'y', 'e', 'h'], ['h'],
             ['x = y'], [lambda p: p.exact('e')])


def part_B(w):
    Nat, Le, S, Z0, Eq, F = w['Nat'], w['Le'], w['Succ'], w['Zero'], w.Eq, w['False']
    # B1: an implication goal, a lemma that proves it
    run_case(w, 'ap_imp', w.forall_([Nat], lambda k: w.arrow(Le(k, k), Le(k, S(k)))), ['k'], ['le_to_succ'], [])
    # B2/B3: a universal goal; a goal that is a universal statement of an implication
    run_case(w, 'ap_all', w.forall_([Nat], lambda n: Le(n, S(n))), [], ['le_succ'], [])
    run_case(w, 'ap_all_imp', w.forall_([Nat, lambda n: Le(n, n)], lambda n, h: Le(n, S(n))), [], ['le_to_succ'], [])
    # B4: a local hypothesis proving a universal implication, applied to an implication goal
    run_case(w, 'ap_hyp_imp', w.forall_([w.forall_([Nat, lambda n: Le(n, n)], lambda n, h: Le(n, S(n))), Nat], lambda f, k: w.arrow(Le(k, k), Le(k, S(k)))), ['f', 'k'], ['f'], [])
    # B5: more premises than the goal has binders: the FIRST premise is left as a goal (the last one is the goal's own binder)
    run_case(w, 'ap_surplus', w.forall_([Nat], lambda k: w.arrow(Le(k, k), Le(k, S(k)))), ['k'], ['le_to_succ2'], ['k <= k'], [lambda p: p.apply('le_n')])
    # B6: the goal has as many binders as the lemma minus the quantified variable
    run_case(w, 'ap_two_prem', w.forall_([Nat], lambda k: w.arrow(Le(k, k), w.arrow(Le(k, k), Le(k, S(k))))), ['k'], ['le_to_succ2'], [])
    # B7/B8: the Pi of the goal hidden behind a definition: an implication, a universal statement
    run_case(w, 'ap_defn_imp', w.forall_([Nat], lambda k: w['Imp2'](Le(k, k), Le(k, S(k)))), ['k'], ['le_to_succ'], [])
    run_case(w, 'ap_defn_all', w['AllLe'], [], ['le_succ'], [])
    # B9: the lemma's conclusion is a bare variable under one premise: the tight reading  ?P -> ?P  against  A -> A  closes the goal (no goal `A -> A` is left over)
    run_case(w, 'ap_id_fn', w.forall_([Nat], lambda k: w.arrow(Le(k, k), Le(k, k))), ['k'], ['id_fn'], [])


def part_C(w):
    Nat, Le, S, Z0, Eq, F = w['Nat'], w['Le'], w['Succ'], w['Zero'], w.Eq, w['False']
    # C1: the lemma concludes False from `S n = 0`: that premise is the negation's own binder
    run_case(w, 'ap_false_concl', w.forall_([Nat], lambda k: ~Eq(S(k), Z0)), ['k'], ['succ_zero_false'], [])
    # C2/C3: ... with a premise in front: it is the only goal left (with the argument given explicitly too)
    run_case(w, 'ap_false_concl_prem', w.forall_([Nat], lambda k: ~Eq(S(k), Z0)), ['k'], ['absurd_succ'], ['k <= k'], [lambda p: p.apply('le_n')])
    run_case(w, 'ap_false_concl_prem_arg', w.forall_([Nat], lambda k: ~Eq(S(k), Z0)), ['k'], ['absurd_succ', 'k'], ['k <= k'], [lambda p: p.apply('le_n')])


def part_D(w):
    Nat, Le, S, Z0, Eq, F = w['Nat'], w['Le'], w['Succ'], w['Zero'], w.Eq, w['False']
    # D1: a goal that is not a Pi: the lemma's binders are all peeled, the premise is left
    run_case(w, 'ap_plain', w.forall_([Nat], lambda k: Le(k, S(k))), ['k'], ['le_to_succ'], ['k <= k'], [lambda p: p.apply('le_n')])
    # D2/D3: a lemma whose conclusion is a bare variable applied to a Pi goal: the tight reading fails (False against the goal's premise), then every binder is peeled and the variable is the goal
    run_case(w, 'ap_exfalso_imp', w.forall_([Nat, lambda k: F], lambda k, f: w.arrow(Le(k, k), Le(k, S(k)))), ['k', 'f'], ['false_elim'], ['False'], [lambda p: p.exact('f')])
    run_case(w, 'ap_exfalso_neg', w.forall_([Nat, lambda k: F], lambda k, f: ~Eq(S(k), Z0)), ['k', 'f'], ['false_elim'], ['False'], [lambda p: p.exact('f')])
    # D4: a lemma that does not fit a Pi goal fails, and the state is what it was: the same goal, no metavariable left behind, and the right lemma still applies afterwards
    for name, goal, wrong in (('neg', ~Eq(S(Z0), Z0), 'le_succ'), ('imp3', w.arrow(Le(Z0, Z0), w.arrow(Le(Z0, S(Z0)), Le(Z0, Z0))), 'le_to_succ')):
        p = Proof(w, goal, 'ap_fail_' + name)
        before = str(p); metas0 = n_metas(w); pending0 = p.ps.num_pending()
        try: p.apply(wrong); expect(False, 'ap_fail_%s: apply %s succeeded' % (name, wrong))
        except AuthorError: pass
        expect(str(p) == before and p.n_goals == 1, 'ap_fail_%s: the state changed:\n%s\n--- expected\n%s' % (name, p, before))
        expect('?m' not in str(p), 'ap_fail_%s: a metavariable is left behind:\n%s' % (name, p))
        metas1 = n_metas(w)
        expect(metas1 == metas0 + 1, 'ap_fail_%s: the refused apply left %d metavariable(s) behind' % (name, metas1 - metas0 - 1))     # (each probe is one new metavariable itself)
        expect(p.ps.num_pending() == pending0, 'ap_fail_%s: the refused apply left deferred constraints behind' % name)
        if name == 'neg':
            try: p.apply('succ_ne_zero')
            except AuthorError as e: expect(False, 'ap_fail_neg: the right lemma does not apply afterwards:\n%s' % str(e)[:300]); continue
            finish_ok(w, p, 'ap_fail_' + name)


def part_E(w):
    Nat, Le, S, Z0, Eq, F = w['Nat'], w['Le'], w['Succ'], w['Zero'], w.Eq, w['False']
    two = S(S(Z0))
    # E1: the hypothesis is `Arrows 2 A B` = A -> A -> B through a RECURSIVE type-level function; the goal is B, not a Pi: both premises are peeled (two goals)
    run_case(w, 'ap_recursive_pi', w.forall_([Nat, lambda k: w['Arrows'](two, Le(k, k), Le(k, S(k)))], lambda k, h: Le(k, S(k))), ['k', 'h'], ['h'],
             ['k <= k', 'k <= k'], [lambda p: p.apply('le_n'), lambda p: p.apply('le_n')])
    # E2: the kernel's apply WITHOUT simp is the reference's: every binder of the lemma is peeled, so a lemma that concludes a negation, an implication or a universal statement does not apply to a goal of
    # that shape (the lockstep gates S3-S5 compare this mode with the frozen reference; the improvements live in the tactic layer's mode). It fails cleanly; on a goal that is not a Pi it works.
    for name, stmt, intros, lemma in (('neg', w.forall_([Nat, Nat, lambda x, y: ~Eq(x, y)], lambda x, y, h: ~Eq(y, x)), ['x', 'y', 'h'], 'ne_symm'),
                                      ('all', w.forall_([Nat], lambda n: Le(n, S(n))), [], 'le_succ'),
                                      ('imp', w.forall_([Nat], lambda k: w.arrow(Le(k, k), Le(k, S(k)))), ['k'], 'le_to_succ')):
        p = Proof(w, stmt, 'ap_raw_' + name)
        if intros: p.intro(*intros)
        before = str(p); metas0 = n_metas(w); pending0 = p.ps.num_pending()
        try: p.ps.apply(p.t(lemma).n, None, False); expect(False, 'ap_raw_%s: the kernel apply without simp applied %s to a Pi goal:\n%s' % (name, lemma, p))
        except E.ITPError: pass
        expect(str(p) == before and p.n_goals == 1 and n_metas(w) == metas0 + 1 and p.ps.num_pending() == pending0, 'ap_raw_%s: the refused kernel apply changed the state:\n%s' % (name, p))
    p = Proof(w, w.forall_([Nat], lambda k: Le(k, S(k))), 'ap_raw_plain'); p.intro('k')
    p.ps.apply(p.t('le_to_succ').n, None, False); p._sync_names()
    expect(p.n_goals == 1, 'ap_raw_plain: the kernel apply without simp leaves one goal, the premise:\n%s' % p)
    p.ps.apply(p.t('le_n').n, None, False); p._sync_names()
    finish_ok(w, p, 'ap_raw_plain')


def part_F(w):
    Nat, Le, S, Z0, Eq, F = w['Nat'], w['Le'], w['Succ'], w['Zero'], w.Eq, w['False']
    motive = w.lam(Nat, lambda n: w.arrow(Le(n, n), Le(n, n)))
    # F1: the lemma's own binders are the base case, the step and the variable; the Pi in its conclusion is the goal's second binder: peel the first two (the count that fits is neither the smallest nor all)
    run_case(w, 'ap_motive_ind', w.forall_([Nat, lambda n: Le(n, n)], lambda n, h: Le(n, n)), [], ['nat_ind', motive],
             ['Zero <= Zero -> Zero <= Zero', 'forall (x0 : Nat), (x0 <= x0 -> x0 <= x0) -> Succ x0 <= Succ x0 -> Succ x0 <= Succ x0'],
             [lambda p: (p.intro('h'), p.exact('h')), lambda p: (p.intro('n', 'ih', 'h'), p.exact('h'))])
    # F2: the count that fits is ALL the lemma's own binders: the variable and the premise are peeled, the Pi in the conclusion is the goal
    run_case(w, 'ap_motive_id', w.arrow(Le(Z0, Z0), Le(Z0, Z0)), [], ['id_pred', motive], ['Zero <= Zero -> Zero <= Zero'], [lambda p: (p.intro('h'), p.exact('h'))])
    # F3: the same lemma with the variable given: nothing but the premise is left
    run_case(w, 'ap_motive_id_arg', w.arrow(Le(Z0, Z0), Le(Z0, Z0)), [], ['id_pred', motive, Z0], ['Zero <= Zero -> Zero <= Zero'], [lambda p: (p.intro('h'), p.exact('h'))])
    # F4: and a predicate that is NOT a Pi still works as before (the lemma's conclusion is the goal)
    run_case(w, 'ap_plain_pred', w.forall_([Nat], lambda n: Le(n, n)), [], ['nat_ind', w.lam(Nat, lambda n: Le(n, n))],
             ['Zero <= Zero', 'forall (x0 : Nat), x0 <= x0 -> Succ x0 <= Succ x0'], [lambda p: p.apply('le_n'), lambda p: (p.intro('n', 'ih'), p.apply('le_n'))])


def main():
    w = build()
    part_A(w); part_B(w); part_C(w); part_D(w); part_E(w); part_F(w)
    print('%d checks' % CHECKS[0])
    if FAIL:
        print('S29 APPLY ON PI GOALS GATE: FAIL (%d)' % len(FAIL))
        for f in FAIL[:20]: print('   ', f)
        sys.exit(1)
    print('S29 APPLY ON PI GOALS GATE: PASS')


if __name__ == '__main__':
    main()
