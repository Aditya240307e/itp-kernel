"""S22 gate: the display normal form (`simp_normalize`), the `change` tactic and `apply` with folded premises.

These three exist so that goals can be SHOWN and REWRITTEN in a readable form (`add (Succ m) n` as `Succ (add m n)`, `add m n` left alone) without the kernel's conversion or typing ever seeing a
different notion of equality: every result below is convertible with its input, and the proof terms built from them are re-checked by the ordinary checker. What this gate pins down:

  A. simp_normalize: a recursive function is unfolded ONE layer exactly when its decreasing argument (whichever position that is) is a constructor application, every other definition stays folded,
     closed arithmetic computes to the right numeral (checked against Python's own arithmetic), the result is convertible with the input and a fixed point, and the mode does not leak into
     ordinary normalisation;
  B. change: accepts a definitionally equal target (at any depth of context), refuses a different one, a non-type and an ill-typed one, replaces the goal in place and leaves the proof term as it was;
  C. apply with simp: the premises of a lemma applied to arguments come out folded (read off the declared type of the head, not off the delta-normalised inferred one), through hypotheses and
     definitions too, with the same proof term as the unfolded mode.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, sys, time

import itp_env_lnl as E
import nat_lib
from author import AuthorError
from tactics import Proof

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond

def raises(thunk, needle=''):
    try: thunk()
    except Exception as e: return needle in str(e) and type(e).__name__ in ('ITPError', 'RuntimeError', 'AuthorError')
    return False


class Ctx:
    """`with Ctx(w, 'm', 'n') as (m, n):` -- the terms are open variables of type Nat (or the given type) while the block runs"""
    def __init__(self, w, *names, ty=None): self.w = w; self.names = names; self.ty = ty
    def __enter__(self):
        ty = self.ty if self.ty is not None else self.w['Nat']
        self.vars = [self.w._bind(ty, nm)[1] for nm in self.names]
        return self.vars
    def __exit__(self, *exc): self.w._unbind(len(self.names))


class GoalCtx:
    """`with GoalCtx(w, ps):` -- terms built inside see the hypotheses of the first goal of the proof state as context variables"""
    def __init__(self, w, ps, i=0): self.w = w; self.ps = ps; self.i = i
    def __enter__(self):
        ps, i = self.ps, self.i; self.saved = (self.w.ctx, self.w.ctx_names)
        self.w.ctx = [ps.goal_hyp_type(i, h) for h in range(ps.goal_num_hyps(i))]; self.w.ctx_names = [ps.goal_hyp_name(i, h) or 'h%d' % h for h in range(ps.goal_num_hyps(i))]
    def __exit__(self, *exc): self.w.ctx, self.w.ctx_names = self.saved


def main():
    t0 = time.time()
    w = nat_lib.build_nat(); k = w.k
    Nat, S, Z, add, mul, pw, pred = w['Nat'], w['Succ'], w['Zero'], w['add'], w['mul'], w['pow'], w['pred']
    lit = w.nota.lit

    # a recursive function whose decreasing argument is NOT the first one and which has an argument after it: iter f n x = f (f ... x)
    w.recursive('iter', w.forall_([w.arrow(Nat, Nat), Nat, Nat], lambda f, n, x: Nat),
                lambda it: w.fun([w.arrow(Nat, Nat), Nat, Nat], lambda f, n, x: w.match(n, lambda _: Nat, [x, lambda n1: f(it(f, n1, x))])), on=1)
    it = w['iter']

    # ------------------------------------------------------------------------------------------------------------------------------ A. simp_normalize
    with Ctx(w, 'm', 'n', 'k') as (m, n, kk):
        P = w.PROP
        cases = [
            ('add (S m) n',            add(S(m), n),             S(add(m, n))),
            ('add m n stays',          add(m, n),                add(m, n)),
            ('add (S (S m)) n',        add(S(S(m)), n),          S(S(add(m, n)))),
            ('add Zero n',             add(Z, n),                n),
            ('add (add m n) k stays',  add(add(m, n), kk),       add(add(m, n), kk)),
            ('add (S m) (S n)',        add(S(m), S(n)),          S(add(m, S(n)))),
            ('mul (S m) n',            mul(S(m), n),             add(n, mul(m, n))),
            ('mul m n stays',          mul(m, n),                mul(m, n)),
            ('mul Zero n',             mul(Z, n),                Z),
            ('pow b (S e)',            pw(m, S(n)),              mul(m, pw(m, n))),
            ('pow (S b) e stays',      pw(S(m), n),              pw(S(m), n)),
            ('pow b Zero',             pw(m, Z),                 S(Z)),
            ('pred (S m)',             pred(S(m)),               m),
            ('pred m stays',           pred(m),                  pred(m)),
            ('pred Zero',              pred(Z),                  Z),
            ('iter f (S n) x',         it(w.lam(Nat, lambda y: S(y)), S(n), kk), S(it(w.lam(Nat, lambda y: S(y)), n, kk))),
            ('iter f n x stays',       it(w.lam(Nat, lambda y: S(y)), n, kk),    it(w.lam(Nat, lambda y: S(y)), n, kk)),
            ('iter f Zero x',          it(w.lam(Nat, lambda y: S(y)), Z, kk),    kk),
            ('definition stays: Lt',   w['Lt'](Z, S(Z)),         w['Lt'](Z, S(Z))),
            ('definition stays: Not',  w['Not'](w['Not'](P)),    w['Not'](w['Not'](P))),
            ('definition stays: Le',   w['Le'](add(m, Z), n),    w['Le'](add(m, Z), n)),
            ('beta',                   w.lam(Nat, lambda x: add(x, Z))(S(m)),   S(add(m, Z))),
            ('match on a constructor', w.match(S(m), lambda _: Nat, [Z, lambda p: p]), m),
            ('match on a variable stays', w.match(m, lambda _: Nat, [Z, lambda p: p]), w.match(m, lambda _: Nat, [Z, lambda p: p])),
        ]
        for name, term, want in cases:
            got = w.simp(term)
            expect(got == want, 'A1 %s: got %s, wanted %s' % (name, w.show(got), w.show(want)))
            expect(w.conv(term, got), 'A2 %s: the result is not convertible with the input' % name)
            expect(w.simp(got) == got, 'A3 %s: not a fixed point' % name)
        # under binders
        t = w.lam(Nat, lambda x: add(S(x), n)); want = w.lam(Nat, lambda x: S(add(x, n)))
        expect(w.simp(t) == want, 'A4 under a lambda: %s' % w.show(w.simp(t)))
        t = w.forall_([Nat], lambda x: w.Eq(add(S(x), Z), x)); want = w.forall_([Nat], lambda x: w.Eq(S(add(x, Z)), x))
        expect(w.simp(t) == want, 'A5 under a Pi, inside an equation: %s' % w.show(w.simp(t)))
        # a partial application of a recursive function is not saturated: it stays folded; with its last argument supplied it computes
        t = add(S(m)); got = w.simp(t)
        expect(got == t, 'A6 add (S m) unapplied stays folded: %s' % w.show(got))
        got = w.simp(w.lam(Nat, lambda y: add(S(m), y)))
        expect(got == w.lam(Nat, lambda y: S(add(m, y))), 'A7 and the function with its argument supplied computes: %s' % w.show(got))
        # the mode does not leak into ordinary normalisation: full delta still unfolds the stuck recursion and the definitions
        full = w.normalize(add(m, n))
        expect(full != add(m, n) and w.view(full)[0] == 'match', 'A8 normalize (add m n) is no longer the stuck match -- the folded mode leaked: %s' % w.show(full))
        expect(w.normalize(w['Not'](P)) != w['Not'](P), 'A9 normalize unfolds a definition (the no-delta mode leaked)')

        # random open terms: convertible with the input and a fixed point
        rng = random.Random(22)
        def gen(d):
            r = rng.random()
            if d == 0 or r < 0.2: return rng.choice([m, n, kk, Z, S(Z)])
            f = rng.choice(['S', 'add', 'mul', 'pow', 'pred', 'add', 'S'])
            if f == 'S': return S(gen(d - 1))
            if f == 'pred': return pred(gen(d - 1))
            return w[f](gen(d - 1), gen(d - 1))
        for i in range(400):
            t = gen(rng.randint(1, 5)); got = w.simp(t)
            expect(w.conv(t, got), 'A10 random open term %d: simp is not convertible with the input: %s  ~>  %s' % (i, w.show(t), w.show(got)))
            expect(w.simp(got) == got, 'A11 random open term %d: not a fixed point: %s' % (i, w.show(t)))

    # closed arithmetic computes: the result is the numeral Python computes
    def gen_closed(d):
        """(term, value) with a value kept small"""
        r = rng.random()
        if d == 0 or r < 0.25:
            v = rng.randint(0, 4); return lit(v), v
        f = rng.choice(['S', 'add', 'mul', 'pow', 'pred', 'add', 'iter'])
        a, av = gen_closed(d - 1)
        if f == 'S': return S(a), av + 1
        if f == 'pred': return pred(a), max(av - 1, 0)
        b, bv = gen_closed(d - 1)
        if f == 'add': return add(a, b), av + bv
        if f == 'mul': return mul(a, b), av * bv
        if f == 'pow':
            if bv > 3 or av > 5: return a, av
            return pw(a, b), av ** bv
        step = w.lam(Nat, lambda y: S(y))                        # iter step a b = b + a... : applies the step `a` times to b
        return it(step, a, b), av + bv
    checked = 0
    for i in range(500):
        t, v = gen_closed(rng.randint(1, 4))
        if v > 120: continue
        got = w.simp(t); checked += 1
        expect(got == lit(v), 'A12 closed term %d: %s should compute to %d, got %s' % (i, w.show(t), v, w.show(got)))
        expect(w.normalize(t) == lit(v), 'A13 closed term %d: full normalisation disagrees with Python' % i)
    expect(checked > 300, 'A14 the closed-term sample was too small (%d)' % checked)

    # a recursive function that, once its decreasing argument is a constructor, still splits on ANOTHER argument: cmp x y = match x [y, p => match y [p, q => cmp p q]]. The unfolded layer of
    # `cmp (S m) n` is a stuck `match n with ...`, which is not shown: the call stays folded until n is a constructor too
    w.recursive('cmp', w.forall_([Nat, Nat], lambda x, y: Nat),
                lambda f: w.fun([Nat, Nat], lambda x, y: w.match(x, lambda _: Nat, [y, lambda p: w.match(y, lambda _: Nat, [p, lambda q: f(p, q)])])))
    cmp_ = w['cmp']
    with Ctx(w, 'm', 'n') as (m, n):
        for name, term, want in [
                ('cmp (S m) n stays (unfolding would only expose a split on n)', cmp_(S(m), n), cmp_(S(m), n)),
                ('cmp (S m) Zero',                                             cmp_(S(m), Z), m),
                ('cmp (S m) (S n)',                                            cmp_(S(m), S(n)), cmp_(m, n)),
                ('cmp Zero n',                                                 cmp_(Z, n), n),
                ('cmp m n stays',                                              cmp_(m, n), cmp_(m, n)),
                ('cmp (S m) unapplied stays',                                  cmp_(S(m)), cmp_(S(m)))]:
            got = w.simp(term)
            expect(got == want, 'A15 %s: got %s, wanted %s' % (name, w.show(got), w.show(want)))
            expect(w.conv(term, got) and w.simp(got) == got, 'A16 %s: not convertible with the input or not a fixed point' % name)
        st = w.view(w.normalize(cmp_(S(m), n)))
        expect(st[0] == 'match', 'A17 full normalisation still unfolds the stuck call (the guard leaked into conversion)')

    # ------------------------------------------------------------------------------------------------------------------------------ B. change
    Eq = w.Eq
    def fresh_proof(stmt_): return k.start_proof(stmt_.n)
    def admit(ps, stmt_, name): return k.declare_checked(stmt_.n, ps.extract(), name)
    def intros(ps, *names):
        for nm in names: ps.intro(nm)
        return [w.T(k.mk_fvar(i)) for i in range(len(names))]
    def hyp_names(ps, i=0): return ['h%d' % h for h in range(ps.goal_num_hyps(i))]

    stmt = w.forall_([Nat], lambda n: Eq(add(S(Z), n), S(n)))
    ps = fresh_proof(stmt); (nv,) = intros(ps, 'n')
    meta0 = ps.goal_meta_id(0); tgt0 = w.T(ps.goal_target(0))
    with GoalCtx(w, ps):
        new1 = Eq(S(add(Z, nv)), S(nv)); new2 = Eq(S(nv), S(nv)); refl_t = w['refl'](Nat, S(nv))
        expect(w.conv(tgt0, new1) and w.conv(tgt0, new2), 'B0 the targets used below are convertible with the goal (test setup)')
    ps.change(new1.n)
    expect(ps.num_goals() == 1 and ps.goal_target(0) == new1.n, 'B1 change puts the new target in the goal, in place')
    expect(ps.goal_meta_id(0) != meta0, 'B2 the goal is a new goal (the old metavariable is solved by it)')
    closed = Eq(Z, Z)                                                   # a target that mentions no hypothesis, so that only the goal lookup can refuse it
    expect(raises(lambda: ps.change(closed.n, meta0), 'no goal with the given meta_id'), 'B3 the superseded goal cannot be changed again')
    ps.change(new2.n)
    ps.exact(refl_t.n)
    expect(ps.is_complete(), 'B4 the proof is complete')
    admit(ps, stmt, 'b_with_change')
    ps2 = fresh_proof(stmt); intros(ps2, 'n'); ps2.exact(refl_t.n)
    expect(ps.extract() == ps2.extract(), 'B5 change leaves the proof term exactly as it was (the same term with and without change)')

    # refusals leave the proof state untouched
    ps = fresh_proof(stmt); intros(ps, 'n'); meta0 = ps.goal_meta_id(0); tgt0 = ps.goal_target(0)
    with GoalCtx(w, ps):
        bad = {
            'a different proposition': (Eq(S(nv), nv).n, 'definitionally equal'),
            'a term that is not a type': (S(nv).n, 'not a type'),
            'an ill-typed target': (w.app(w['Eq'], Nat, Z, w['True']).n, ''),
        }
    for name, (tgt, needle) in bad.items():
        expect(raises(lambda: ps.change(tgt), needle), 'B6 change to %s was not refused%s' % (name, (' with ' + repr(needle)) if needle else ''))
        expect(ps.num_goals() == 1 and ps.goal_meta_id(0) == meta0 and ps.goal_target(0) == tgt0, 'B7 a refused change altered the proof state (%s)' % name)
    expect(raises(lambda: ps.change(closed.n, 987654), 'no goal with the given meta_id'), 'B8 change on an unknown goal id')
    # an ill-typed target that nevertheless REDUCES to the goal (the function ignores its argument, and Zero is not a Prop): conversion alone would accept it, the type check must not
    with GoalCtx(w, ps):
        sneaky = w.lam(w.PROP, lambda x: w.T(tgt0))(Z)
    expect(w.view(sneaky)[0] == 'app', 'B8b (test setup) the sneaky target is a beta redex')
    expect(raises(lambda: ps.change(sneaky.n)), 'B8c an ill-typed target that reduces to the goal was accepted')
    expect(ps.num_goals() == 1 and ps.goal_meta_id(0) == meta0 and ps.goal_target(0) == tgt0, 'B8d and the refusal left the proof state alone')
    # a target mentioning something outside the context is refused at the boundary
    expect(raises(lambda: ps.change(k.mk_fvar(5)), 'ill-scoped'), 'B8e change refuses a target with a variable beyond the context')
    # the extracted proof keeps definitions folded (no delta in extraction), also after the folded normal form has been used in this very process
    ps = fresh_proof(stmt); (nv,) = intros(ps, 'n')
    with GoalCtx(w, ps): ex = w['refl'](Nat, add(S(Z), nv))
    ps.exact(ex.n)
    want = w.fun([Nat], lambda n: w['refl'](Nat, add(S(Z), n)))
    expect(ps.extract() == want.n, 'B8f the extracted proof is exactly the term given (definitions folded), got %s' % w.show(w.T(ps.extract())))

    # the conversion is made at the right depth of context: with one variable g in scope, `fun x => g` and `fun x => x` must differ (a binder at a wrong level would equate them)
    NN = w.arrow(Nat, Nat)
    stmt_g = w.forall_([Nat], lambda g: Eq(w.lam(Nat, lambda x: g), w.lam(Nat, lambda x: g)))
    ps = fresh_proof(stmt_g); (gv,) = intros(ps, 'g')
    with GoalCtx(w, ps):
        wrong = Eq(w.lam(Nat, lambda x: gv), w.lam(Nat, lambda x: x))
        right = Eq(w.lam(Nat, lambda x: add(Z, gv)), w.lam(Nat, lambda x: gv))
        ident = w['refl'](NN, w.lam(Nat, lambda x: gv))
    expect(raises(lambda: ps.change(wrong.n), 'definitionally equal'), 'B9 `fun x => g` is not `fun x => x` (the conversion ran at the wrong context depth)')
    ps.change(right.n)
    expect(ps.goal_target(0) == right.n, 'B10 a convertible target under a lambda is accepted')
    ps.exact(ident.n)
    expect(ps.is_complete(), 'B11 and the proof finishes')
    admit(ps, stmt_g, 'b_binder')

    # several goals: change replaces the chosen one in place and keeps the order
    both = w.forall_([Nat], lambda n: w.And(Eq(add(Z, n), n), Eq(add(Z, n), n)))
    ps = fresh_proof(both); (nv,) = intros(ps, 'n')
    with GoalCtx(w, ps):
        conj_app = w['conj'](Eq(add(Z, nv), nv), Eq(add(Z, nv), nv)); eqnn = Eq(nv, nv); rn = w['refl'](Nat, nv)
    ps.apply(conj_app.n, None, False)
    expect(ps.num_goals() == 2, 'B12 two goals after applying conj (test setup)')
    m0, m1 = ps.goal_meta_id(0), ps.goal_meta_id(1); t0_ = ps.goal_target(0)
    ps.change(eqnn.n, m1)
    expect(ps.num_goals() == 2 and ps.goal_meta_id(0) == m0 and ps.goal_meta_id(1) != m1 and ps.goal_target(0) == t0_ and ps.goal_target(1) == eqnn.n,
           'B13 changing the second goal keeps the first untouched and the order')
    ps.exact(rn.n, ps.goal_meta_id(1)); ps.exact(rn.n)
    expect(ps.is_complete(), 'B14 both goals are closed')
    admit(ps, both, 'b_two_goals')

    # ------------------------------------------------------------------------------------------------------------------------------ C. apply with folded premises
    # forall m n k, add m n = add m k -> n = k   by  apply (add_left_cancel m)
    stmt_c = w.forall_([Nat, Nat, Nat, lambda m, n, kk: Eq(add(m, n), add(m, kk))], lambda m, n, kk, h: Eq(n, kk))
    results = {}
    for simp in (True, False):
        ps = fresh_proof(stmt_c); mv, nv, kv, hv = intros(ps, 'm', 'n', 'k', 'h')
        with GoalCtx(w, ps): cand = w['add_left_cancel'](mv); folded = Eq(add(mv, nv), add(mv, kv))
        ps.apply(cand.n, None, simp)
        expect(ps.num_goals() == 1, 'C0 apply (add_left_cancel m) leaves one goal (simp=%s)' % simp)
        tgt = w.T(ps.goal_target(0))
        with GoalCtx(w, ps): same_meaning = w.conv(tgt, folded)
        if simp: expect(tgt == folded, 'C1 folded premise expected, got %s' % w.show(tgt, names=['m', 'n', 'k', 'h']))
        else: expect(tgt != folded, 'C2 (control) the unfolded mode really does unfold the premise: got %s' % w.show(tgt, names=['m', 'n', 'k', 'h']))
        expect(same_meaning, 'C3 the premise is convertible with the folded one (simp=%s)' % simp)
        ps.exact(hv.n)
        expect(ps.is_complete(), 'C4 finished (simp=%s)' % simp)
        results[simp] = ps.extract(); admit(ps, stmt_c, 'c_cancel_%s' % simp)
    expect(results[True] == results[False], 'C5 the proof term is the same with and without simp')

    # a hypothesis applied to an argument:  h : forall x, add m x = add x m -> x = 0,  goal  k = 0 ;  apply (h k)
    stmt_h = w.forall_([Nat, lambda m: w.forall_([Nat], lambda x: w.arrow(Eq(add(m, x), add(x, m)), Eq(x, Z))), Nat], lambda m, h, kk: w.arrow(Eq(add(m, kk), add(kk, m)), Eq(kk, Z)))
    ps = fresh_proof(stmt_h); mv, hv, kv, ev = intros(ps, 'm', 'h', 'k', 'e')
    with GoalCtx(w, ps): cand = hv(kv); folded = Eq(add(mv, kv), add(kv, mv)); h0 = w.T(ps.goal_target(0))
    ps.apply(cand.n, None, True)
    tgt = w.T(ps.goal_target(0))
    expect(ps.num_goals() == 1 and tgt == folded, 'C6 premise of a hypothesis applied to an argument is folded, got %s' % w.show(tgt, names=['m', 'h', 'k', 'e']))
    ps.exact(ev.n); expect(ps.is_complete(), 'C6b finished'); admit(ps, stmt_h, 'c_hyp')

    # a hypothesis whose type is a definition that hides the Pi:  h : Not P,  goal False ;  apply h  gives the goal P
    stmt_n = w.forall_([w.PROP, lambda P: w['Not'](P), lambda P, h: P], lambda P, h, p: w['False'])
    ps = fresh_proof(stmt_n); Pv, hv, pv = intros(ps, 'P', 'h', 'p')
    ps.apply(hv.n, None, True)
    expect(ps.num_goals() == 1 and ps.goal_target(0) == Pv.n, 'C7 apply of `h : Not P` to the goal False leaves the goal P')
    ps.exact(pv.n); expect(ps.is_complete(), 'C8 finished'); admit(ps, stmt_n, 'c_not')
    # ... and with the argument already given: the Pi to peel is hidden behind the definition, so the declared type cannot be read off syntactically and the inferred one is used
    ps = fresh_proof(stmt_n); Pv, hv, pv = intros(ps, 'P', 'h', 'p')
    with GoalCtx(w, ps): cand = hv(pv)
    ps.apply(cand.n, None, True)
    expect(ps.is_complete(), 'C9 apply (h p) closes the goal False outright'); admit(ps, stmt_n, 'c_not2')

    # a head that is neither a constant nor a variable: the inferred type is used and the answer is still right
    ps = fresh_proof(stmt_c); mv, nv, kv, hv = intros(ps, 'm', 'n', 'k', 'h')
    with GoalCtx(w, ps): cand = w.lam(Nat, lambda x: w['add_left_cancel'](x))(mv); folded = Eq(add(mv, nv), add(mv, kv))
    ps.apply(cand.n, None, True)
    with GoalCtx(w, ps): ok = ps.num_goals() == 1 and w.conv(w.T(ps.goal_target(0)), folded)
    expect(ok, 'C10 a lambda-headed candidate still gives the right premise')
    ps.exact(hv.n); expect(ps.is_complete(), 'C11 finished')

    # intro through a definition: with simp the hypothesis is what the user wrote (`n < n`), not what `<` is defined as; without it the definition is unfolded all the way
    stmt_nl = w.forall_([Nat], lambda n: w['Not'](w['Lt'](n, n)))
    for simp in (True, False):
        ps = fresh_proof(stmt_nl); intros(ps, 'n'); nv = w.T(k.mk_fvar(0))
        ps.intro('h', None, simp)
        ty = w.T(ps.goal_hyp_type(0, 1)); folded = w['Lt'](nv, nv)
        if simp: expect(ty == folded, 'C14 intro of `Not (n < n)` gives the hypothesis n < n, got %s' % w.show(ty, names=['n']))
        else: expect(ty != folded, 'C15 (control) without simp the hypothesis is the unfolded definition')
        with GoalCtx(w, ps): expect(w.conv(ty, folded), 'C16 the hypothesis is convertible with n < n (simp=%s)' % simp)
        expect(ps.goal_target(0) == w['False'].n, 'C17 the new goal is False (simp=%s)' % simp)
        ps.exact(w['lt_irrefl'](nv, w.T(k.mk_fvar(1))).n); expect(ps.is_complete(), 'C18 finished (simp=%s)' % simp); admit(ps, stmt_nl, 'c_intro_not_%s' % simp)
    ps = fresh_proof(stmt_nl); intros(ps, 'n'); ps.intro('h', None, True)
    expect(raises(lambda: ps.intro('x', None, True)), 'C19 intro on a goal that is not a Pi (False) is refused with simp too')
    stmt_ndef = w.forall_([Nat, Nat], lambda n, m: Eq(n, m))
    ps = fresh_proof(stmt_ndef); intros(ps, 'n', 'm')
    expect(raises(lambda: ps.intro('x', None, True)), 'C20 intro on an equation is refused with simp')
    ps = fresh_proof(w.forall_([Nat], lambda n: w['Not'](w['Not'](w['Le'](n, n))))); intros(ps, 'n'); ps.intro('h', None, True)
    with GoalCtx(w, ps): expect(w.T(ps.goal_hyp_type(0, 1)) == w['Not'](w['Le'](w.T(k.mk_fvar(0)), w.T(k.mk_fvar(0)))), 'C21 intro of `Not (Not P)` gives `Not P`, still folded')

    # a candidate with the right head but an argument that does not fit: refused, state unchanged
    ps = fresh_proof(stmt_c); intros(ps, 'm', 'n', 'k', 'h'); meta0 = ps.goal_meta_id(0)
    expect(raises(lambda: ps.apply(w['add_left_cancel'](w['True']).n, None, True)), 'C12 a candidate applied to an argument of the wrong type is refused')
    expect(ps.num_goals() == 1 and ps.goal_meta_id(0) == meta0, 'C13 a refused apply leaves the state alone')

    # ------------------------------------------------------------------------------------------------------------------------------ E. typing leaves types unevaluated
    def nodes_of(t):
        seen = set(); stack = [t.n]; out = []
        while stack:
            n = stack.pop()
            if n in seen: continue
            seen.add(n); out.append(n); v = k.view(n)
            if v[0] in ('app', 'lam', 'pi'): stack += [v[1], v[2]]
            elif v[0] == 'let': stack += [v[1], v[2], v[3]]
            elif v[0] == 'match': stack += [v[1], v[2]] + list(v[3])
            elif v[0] == 'fix': stack += [v[1], v[3]]
        return out
    def has_match(t): return any(k.view(n)[0] == 'match' for n in nodes_of(t))
    def has_redex(t): return any(k.view(n)[0] == 'app' and k.view(k.view(n)[1])[0] == 'lam' for n in nodes_of(t))

    with Ctx(w, 'm', 'n') as (m, n):
        ty = w.infer(w['add_comm'](m, n))
        expect(ty == Eq(add(m, n), add(n, m)), 'E1 the type of a lemma applied to arguments is the lemma\'s statement with the arguments substituted, nothing evaluated: %s' % w.show(ty))
        expect(not has_match(ty), 'E1b ... no stuck match in it')
        ty = w.infer(w['f_equal'](Nat, Nat, S, m, n, w._bind(Eq(m, n), 'h')[1])) if False else None
        _, h = w._bind(Eq(m, n), 'h')
        try:
            ty = w.infer(w['f_equal'](Nat, Nat, S, m, n, h))
            expect(ty == Eq(S(m), S(n)), 'E2 a dependent application of six arguments gets every argument into its place: %s' % w.show(ty))
            P = w.lam(Nat, lambda z: Eq(add(z, Z), z))
            _, px = w._bind(w.app(P, m), 'px')
            try:
                ty = w.infer(w['transport'](Nat, P, m, n, h, px))
                expect(ty == w.app(P, n), 'E3 the result is the unreduced instance `P n` of the motive (comparison, not the type, does the computing): %s' % w.show(ty))
                expect(has_redex(ty), 'E3b ... which is a beta-redex')
                expect(w.conv(ty, Eq(add(n, Z), n)), 'E3c ... and convertible with the reduced statement')
            finally: w._unbind(1)
            expect(raises(lambda: w.infer(w['f_equal'](Nat, Nat, S, m, n, px if False else m))), 'E4 an argument of the wrong type in the last position is refused')
        finally: w._unbind(1)
        expect(raises(lambda: w.infer(w['add_comm'](w['True'], n))), 'E5 a wrong argument in the first position is refused')
        expect(raises(lambda: w.infer(w['add_comm'](m, w['True']))), 'E5b ... in the last position is refused')
        expect(raises(lambda: w.infer(w['add_comm'](m, n, n))), 'E5c a term applied to more arguments than its type has is refused')
        expect(raises(lambda: w.infer(Z(Z))), 'E5d a constructor of a type applied as a function is refused')

    # a function whose type is a definition that unfolds to a Pi (the Pi is not visible in the syntax), alone and after a visible Pi
    w.define('Fn', w.TYPE, w.arrow(Nat, Nat)); w.define('Fam', w.arrow(Nat, w.TYPE), w.lam(Nat, lambda n: w.arrow(Nat, Nat)))
    Fn, Fam = w['Fn'], w['Fam']
    with Ctx(w, 'f', ty=Fn) as (f,):
        _, g = w._bind(w.pi(Nat, lambda n: Fam(n)), 'g')
        try:
            expect(w.infer(f(Z)) == Nat, 'E6 a function whose type is a definition unfolding to a Pi can be applied')
            expect(raises(lambda: w.infer(f(Z, Z))), 'E6b ... but not to too many arguments')
            expect(raises(lambda: w.infer(f(w['True']))), 'E6c ... nor to an argument of the wrong type')
            expect(w.infer(g(Z, S(Z))) == Nat, 'E7 a Pi hidden behind a definition after a visible Pi: g 0 1 : Nat')
            expect(w.infer(g(Z)) is not None and w.conv(w.infer(g(Z)), w.arrow(Nat, Nat)), 'E7b ... and g 0 is the function type')
            expect(raises(lambda: w.infer(g(Z, Z, Z))), 'E7c ... but not to too many arguments')
            expect(raises(lambda: w.infer(g(w['True'], Z))), 'E7d ... nor to a wrong first argument')
        finally: w._unbind(1)
    # a lambda checked against an expected type that is a Pi, a definition unfolding to one, or a visible Pi followed by such a definition
    chk_ok = lambda tm, ty: not raises(lambda: w.check(tm, ty)) 
    expect(chk_ok(w.lam(Nat, lambda x: x), Fn), 'E8 a lambda is accepted against a definition that unfolds to a Pi')
    expect(chk_ok(w.lam(Nat, lambda a: w.lam(Nat, lambda b: a)), w.arrow(Nat, Fn)), 'E8b ... against a Pi whose codomain is such a definition')
    expect(chk_ok(w.lam(Nat, lambda a: w.lam(Nat, lambda b: b)), w.pi(Nat, lambda a: Fam(a))), 'E8c ... against a dependent Pi whose codomain is an application of such a family')
    expect(not chk_ok(w.lam(Nat, lambda a: w.lam(Nat, lambda b: w.lam(Nat, lambda c: a))), Fn), 'E9 a lambda with more binders than the type has Pis is refused')
    expect(not chk_ok(w.lam(w['True'], lambda a: a), Fn), 'E9b a binder annotation that is not the expected domain is refused')
    expect(not chk_ok(w.lam(Nat, lambda a: w.lam(Nat, lambda b: w['True'])), w.pi(Nat, lambda a: Fam(a))), 'E9c a body of the wrong type is refused')
    expect(not chk_ok(w.lam(Nat, lambda a: Z), Nat), 'E9d a lambda against a type that is not a function type is refused')
    # identical types are accepted without evaluation, different but convertible ones by evaluation, different ones are refused
    expect(chk_ok(w['refl'](Nat, add(S(Z), S(Z))), Eq(S(S(Z)), add(S(Z), S(Z)))), 'E10 convertible but not identical types are accepted')
    expect(not chk_ok(w['refl'](Nat, S(Z)), Eq(S(Z), S(S(Z)))), 'E10b a type that is not convertible is refused')

    # a lambda whose body is a match with a HOLE for its motive: the motive is built from the expected type, which must therefore mention the binder as a variable, not as a loose index
    mot_ty = w.pi(Nat, lambda n: w.PROP)
    def hole_match():
        meta = k.mk_meta(k.fresh_meta(mot_ty.n, 0))
        return w.lam(Nat, lambda n: w.T(k.mk_match(n.n, meta, [w['refl'](Nat, Z).n, w.lam(Nat, lambda j: w['refl'](Nat, S(j))).n])))
    ok1, _ = k.check_partial(hole_match().n, w.pi(Nat, lambda n: Eq(n, n)).n, [])
    ok2, _ = k.check_partial(hole_match().n, w.pi(Nat, lambda n: Eq(n, S(n))).n, [])
    expect(ok1, 'E11 fun n => match n with ?motive | refl 0 | fun j => refl (S j)  has the type  forall n, n = n  (the motive is inferred from the expected type)')
    expect(not ok2, 'E11b ... and not the type  forall n, n = S n')

    # ------------------------------------------------------------------------------------------------------------------------------ F. folded goals through the tactic layer
    from tactics import Proof
    stmt_iff = w.forall_([Nat, Nat], lambda n, m: w.Iff(w['Le'](n, m), w.Ex(Nat, lambda j: Eq(add(n, j), m))))
    p = Proof(w, stmt_iff, 'f_iff'); p.intro('n', 'm'); p.apply('conj')
    goals_ok = p.n_goals == 2 and all(not has_match(p.goal(i).target) and not has_redex(p.goal(i).target) for i in range(2))
    expect(goals_ok, 'F1 the two goals of `apply conj` on an iff are written with definitions folded (no stuck match, no beta-redex):\n%s' % p)
    p.intro('h'); p.apply('le_exists'); p.exact('h'); p.intro('h'); p.apply('exists_le'); p.exact('h')
    expect(p.done, 'F1b ... and the proof goes through'); w.k.declare_checked(stmt_iff.n, p.term().n, 'f_iff')
    stmt_el = w.forall_([Nat, Nat, lambda n, m: w.Ex(Nat, lambda j: Eq(add(n, j), m))], lambda n, m, h: w['Le'](n, m))
    p = Proof(w, stmt_el, 'f_el'); p.intro('n', 'm', 'h'); p.cases('h', names=['j', 'hj'])
    names = [nm for nm, _ in p.goal().hyps]
    with GoalCtx(w, p.ps):
        hj_ty = p.goal().hyps[names.index('hj')][1]
        fv = lambda nm: w.T(k.mk_fvar(names.index(nm)))
        expect(hj_ty == Eq(add(fv('n'), fv('j')), fv('m')), 'F2 `cases` on an existential gives the witness property folded, not a beta-redex: %s' % w.show(hj_ty))
    p.apply('exists_le'); p.exact(lambda pp: w['ex_intro'](Nat, w.lam(Nat, lambda j: Eq(add(pp.h('n'), j), pp.h('m'))), pp.h('j'), pp.h('hj')))
    expect(p.done, 'F2b ... and the proof goes through')
    stmt_lar = w.forall_([Nat, Nat], lambda n, j: w['Le'](n, add(n, j)))
    p = Proof(w, stmt_lar, 'f_lar'); p.intro('n', 'j'); p.apply('le_add_right')
    expect(p.done, 'F3 `apply` unifies a lemma\'s folded conclusion (le_add_right: n <= n + j) with a goal whose arithmetic is stuck')
    stmt_az = w.forall_([Nat], lambda n: Eq(add(n, Z), n))
    p = Proof(w, stmt_az, 'f_az'); p.intro('n'); p.induct('n', names=['j', 'ih'])
    with GoalCtx(w, p.ps, 1):
        names = [nm for nm, _ in p.goal(1).hyps]
        ih_ty = p.goal(1).hyps[names.index('ih')][1]
        fv = lambda nm: w.T(k.mk_fvar(names.index(nm)))
        expect(ih_ty == Eq(add(fv('j'), Z), fv('j')), 'F4 the induction hypothesis is written with the arithmetic folded: %s' % w.show(ih_ty))

    # ------------------------------------------------------------------------------------------------------------------------------ G. a hypothesis of an indexed family whose index is a stuck application
    Le = w['Le']
    stmt_gc = w.forall_([Nat, Nat, Nat, lambda a, x, y: Le(a, add(x, y))], lambda a, x, y, h: Le(a, add(y, x)))
    p = Proof(w, stmt_gc, 'g_comm'); p.intro('a', 'x', 'y', 'h'); p.induct('h', names=['m', 'n', 'ih'], indexed=True)
    expect(p.n_goals == 2, 'G1 induct on `a <= x + y` (index a stuck call of add) gives the two cases')
    leaks = [(i, nm) for i in range(p.n_goals) for nm, ty in p.goal(i).hyps if has_match(ty)]
    expect(not leaks, 'G2 no hypothesis of either case contains a stuck match: the equations are written with `x + y`, as in the hypothesis (%s):\n%s' % (leaks, p))
    with GoalCtx(w, p.ps, 0):
        names = [nm for nm, _ in p.goal(0).hyps]; fv = lambda nm: w.T(k.mk_fvar(names.index(nm)))
        m_ty = p.goal(0).hyps[names.index('m')][1]
        expect(m_ty == Eq(fv('a'), add(fv('x'), fv('y'))), 'G3 the equation of the first case is `a = x + y` exactly: %s' % w.show(m_ty))
    p.rw('m'); p.rw('add_comm', 'x', 'y'); p.apply('le_refl')
    expect(p.n_goals == 1, 'G4 `rw` sees `x + y` in the equation of the first case (the first case is closed)')
    p.apply('le_trans', 'a', 'n', (w['add'], 'y', 'x')); p.exact('ih')
    p.apply('le_trans', 'n', (w['Succ'], 'n'), (w['add'], 'y', 'x')); p.apply('le_succ')
    p.rw('h8'); p.rw('add_comm', 'x', 'y'); p.apply('le_refl')
    expect(p.done, 'G5 ... and so is the second case: the whole proof goes through')
    w.k.declare_checked(stmt_gc.n, p.term().n, 'g_comm')
    # `cases` on the same hypothesis: the same folded reading, whatever it can say about the cases
    p = Proof(w, stmt_gc, 'g_cases'); p.intro('a', 'x', 'y', 'h'); p.cases('h', names=['m', 'hm'])
    leaks = [(i, nm) for i in range(p.n_goals) for nm, ty in p.goal(i).hyps if has_match(ty)]
    expect(p.n_goals == 2 and not leaks, 'G6 cases on the same hypothesis: two cases and no stuck match in any hypothesis (%s)' % leaks)

    stmt_gs = w.forall_([Nat, Nat, Nat, lambda a, j, y: Le(a, add(S(j), y))], lambda a, j, y, h: Eq(a, a))
    p = Proof(w, stmt_gs, 'g_succ'); p.intro('a', 'j', 'y', 'h'); p.cases('h', names=['m', 'hm'])
    with GoalCtx(w, p.ps, 1):
        names = [nm for nm, _ in p.goal(1).hyps]; fv = lambda nm: w.T(k.mk_fvar(names.index(nm)))
        hm_ty = p.goal(1).hyps[names.index('hm')][1] if 'hm' in names else None
        expect(hm_ty is not None and hm_ty == Le(fv('a'), add(fv('j'), fv('y'))), 'G7 `a <= S j + y` read as `a <= S (j + y)`: the le_S case knows its predecessor is `j + y`: %s' % (w.show(hm_ty) if hm_ty is not None else None))

    # ------------------------------------------------------------------------------------------------------------------------------ H. apply reaches a lemma's conclusion through a definition
    # sum3 x y z := (x + y) + z is a DEFINITION. add_assoc's conclusion `(?n + ?m) + ?k = ?n + (?m + ?k)` meets the goal `sum3 a b c = a + (b + c)` only by unfolding sum3; with every recursive call unfolded
    # as well the left side is a stuck match on the metavariable ?n, which the unifier can only defer, and apply would hand ?n ?m ?k back as three open goals
    w.define('sum3', w.forall_([Nat, Nat, Nat], lambda x, y, z: Nat), w.fun([Nat, Nat, Nat], lambda x, y, z: add(add(x, y), z)))
    stmt_h3 = w.forall_([Nat, Nat, Nat], lambda a, b, c: Eq(w['sum3'](a, b, c), add(a, add(b, c))))
    p = Proof(w, stmt_h3, 'h_sum3'); p.intro('a', 'b', 'c'); p.apply('add_assoc')
    expect(p.done, 'H1 apply add_assoc to a goal stated with a definition: the arguments are found by unification, no goal is left\n%s' % p)
    w.k.declare_checked(stmt_h3.n, p.term().n, 'h_sum3')
    # the same through two definitions, and with an argument that is not a variable
    w.define('sum3b', w.forall_([Nat, Nat, Nat], lambda x, y, z: Nat), w.fun([Nat, Nat, Nat], lambda x, y, z: w['sum3'](x, y, z)))
    stmt_h4 = w.forall_([Nat, Nat], lambda a, b: Eq(w['sum3b'](S(a), b, a), add(S(a), add(b, a))))
    p = Proof(w, stmt_h4, 'h_sum3b'); p.intro('a', 'b'); p.apply('add_assoc')
    expect(p.done, 'H2 ... through two definitions and with a non-variable argument\n%s' % p)
    w.k.declare_checked(stmt_h4.n, p.term().n, 'h_sum3b')
    # a conclusion that does not fit is still refused
    stmt_h5 = w.forall_([Nat, Nat, Nat], lambda a, b, c: Eq(w['sum3'](a, b, c), add(a, add(c, b))))
    p = Proof(w, stmt_h5, 'h_bad'); p.intro('a', 'b', 'c')
    n_before = p.n_goals
    try: p.apply('add_assoc'); ok = not p.done
    except AuthorError: ok = True
    expect(ok, 'H3 add_assoc does not close sum3 a b c = a + (c + b)')

    # a failed attempt leaves nothing behind. `konst x := Zero` ignores its argument, so the first-order match of `konst ?x` against `konst a` (the first attempt, on the folded forms) guesses ?x := a --
    # and then fails on the other side of the equation, where the lemma still has `konst` and the goal has Zero. The attempt with the definitions unfolded then finds ?x not constrained at all, and the
    # lemma's argument comes back as an open goal; had the guess survived, that goal would be gone, and the result would depend on an accident of the order of the attempts
    w.define('konst', w.arrow(Nat, Nat), w.lam(Nat, lambda x: Z))
    def _konst_refl(pp): pp.intro('x'); pp.refl()
    w.prove('konst_refl', w.forall_([Nat], lambda x: Eq(w['konst'](x), w['konst'](x))), _konst_refl)
    stmt_k = w.forall_([Nat], lambda a: Eq(w['konst'](a), Z))
    ps = fresh_proof(stmt_k); (av,) = intros(ps, 'a')
    ps.apply(w['konst_refl'].n, None, True)
    expect(ps.num_goals() == 1 and ps.goal_target(0) == Nat.n, 'H4 apply konst_refl to `konst a = 0`: the argument is not determined by the goal, so it is the one open goal (no guess of the first attempt survives)')
    ps.exact(av.n); expect(ps.is_complete(), 'H4b ... and it can be chosen freely'); admit(ps, stmt_k, 'h_konst')
    # a refused apply leaves the state as it was: the goal, and nothing guessed or deferred (`?f ?x = 0` against `S a = S b` is a constraint the unifier first defers, then loses on `0 = S b`)
    stmt_r = w.forall_([Nat, Nat, lambda a, b: w.forall_([w.arrow(Nat, Nat), Nat], lambda f, x: Eq(f(x), Z))], lambda a, b, hh: Eq(S(a), S(b)))
    ps = fresh_proof(stmt_r); intros(ps, 'a', 'b', 'hh'); meta0 = ps.goal_meta_id(0)
    expect(raises(lambda: ps.apply(w.T(k.mk_fvar(2)).n, None, True)), 'H5 apply of `forall f x, f x = 0` to `S a = S b` is refused')
    expect(ps.num_goals() == 1 and ps.goal_meta_id(0) == meta0 and ps.num_pending() == 0, 'H5b a refused apply leaves the goal alone and no deferred constraint behind (%d pending)' % ps.num_pending())
    expect(raises(lambda: ps.apply(w.T(k.mk_fvar(2)).n, None, False)) and ps.num_pending() == 0, 'H5c ... also without simp')

    # a lemma whose arrow sits behind a definition (`Imp A B := A -> B`): only the full-delta reading sees the conclusion, but the premise and the arguments solved by unification must still read as written
    PROP = w.PROP
    w.define('Imp', w.forall_([PROP, PROP], lambda a, b: PROP), w.fun([PROP, PROP], lambda a, b: w.arrow(a, b)))
    Imp = w['Imp']
    def _imp_le(pp): pp.intro('x'); pp.intro('h'); pp.apply('le_refl')
    w.prove('imp_le', w.forall_([Nat], lambda x: Imp(Le(Z, x), Le(x, x))), _imp_le)
    sum3 = w['sum3']
    stmt_i = w.forall_([Nat, Nat, Nat], lambda a, b, c: Le(sum3(a, b, c), sum3(a, b, c)))
    for simp in (True, False):
        ps = fresh_proof(stmt_i); av, bv, cv = intros(ps, 'a', 'b', 'c')
        ps.apply(w['imp_le'].n, None, simp)
        tgt = w.T(ps.goal_target(0)); folded = Le(Z, sum3(av, bv, cv))
        expect(ps.num_goals() == 1, 'H6 apply imp_le leaves the premise as the one goal (simp=%s)' % simp)
        if simp: expect(tgt == folded, 'H6b the premise reads `0 <= sum3 a b c`, the argument found by unification stays folded: %s' % w.show(tgt, names=['a', 'b', 'c']))
        else: expect(tgt != folded, 'H6c (control) without simp the argument is not read back folded: %s' % w.show(tgt, names=['a', 'b', 'c']))
        with GoalCtx(w, ps): expect(w.conv(tgt, folded), 'H6d the premise is convertible with `0 <= sum3 a b c` (simp=%s)' % simp)
        ps.apply(w['zero_le'].n, None, simp); expect(ps.is_complete(), 'H6e finished (simp=%s)' % simp); admit(ps, stmt_i, 'h_imp_le_%s' % simp)
    # the same lemma with its first argument and the proof of the first premise given: the arrow that is left is behind the definition too, and its premise is read off the declared type
    def _imp2_le(pp): pp.intro('x'); pp.intro('h'); pp.intro('h2'); pp.apply('le_refl')
    w.prove('imp2_le', w.forall_([Nat], lambda x: Imp(Le(Z, x), Imp(Le(x, S(x)), Le(x, x)))), _imp2_le)
    stmt_j = w.forall_([Nat, Nat, Nat, lambda a, b, c: Le(Z, sum3(a, b, c))], lambda a, b, c, hz: Le(sum3(a, b, c), sum3(a, b, c)))
    ps = fresh_proof(stmt_j); av, bv, cv, zv = intros(ps, 'a', 'b', 'c', 'hz')
    with GoalCtx(w, ps): cand = w['imp2_le'](sum3(av, bv, cv), zv); folded = Le(sum3(av, bv, cv), S(sum3(av, bv, cv)))
    ps.apply(cand.n, None, True)
    expect(ps.num_goals() == 1 and ps.goal_target(0) == folded.n, 'H7 apply (imp2_le t hz): the premise behind the definition is read folded, `sum3 a b c <= S (sum3 a b c)`: %s' % w.show(w.T(ps.goal_target(0)), names=['a', 'b', 'c', 'hz']))
    # the arrow behind a RECURSIVE function (`Imps n A B` is A -> ... -> B, n times): the folded reading gives up, and everything still works through full delta
    w.recursive('Imps', w.forall_([Nat, PROP, PROP], lambda n, a, b: PROP),
                lambda f: w.fun([Nat, PROP, PROP], lambda n, a, b: w.match(n, lambda _: PROP, [b, lambda m: w.arrow(a, f(m, a, b))])))
    Imps = w['Imps']
    stmt_m = w.forall_([PROP, PROP, lambda a, b: w.arrow(a, w.arrow(a, b))], lambda a, b, f: Imps(S(S(Z)), a, b))
    ps = fresh_proof(stmt_m); av, bv, fv_ = intros(ps, 'A', 'B', 'f')
    ps.intro('x', None, True); ps.intro('y', None, True)
    expect(ps.num_goals() == 1 and ps.goal_target(0) == bv.n and ps.goal_num_hyps(0) == 5 and ps.goal_hyp_type(0, 3) == av.n and ps.goal_hyp_type(0, 4) == av.n, 'C22 intro through a recursive function that computes to an arrow gives its premises and the conclusion')
    ps.exact(w.T(k.mk_app(k.mk_app(fv_.n, k.mk_fvar(3)), k.mk_fvar(4))).n); expect(ps.is_complete(), 'C22b finished'); admit(ps, stmt_m, 'c_imps')
    ps = fresh_proof(w.forall_([PROP, PROP, lambda a, b: Imps(S(Z), a, b)], lambda a, b, h: a)); intros(ps, 'A', 'B', 'h')
    with GoalCtx(w, ps): cand = w.T(k.mk_fvar(2))(w['Zero'])
    expect(raises(lambda: ps.apply(cand.n, None, True)), 'C22c an argument of the wrong type given to a hypothesis whose type hides its arrow behind a recursive function is refused')
    ps = fresh_proof(w.forall_([PROP, PROP, lambda a, b: Imps(S(Z), a, b), lambda a, b, h: a], lambda a, b, h, x: b)); intros(ps, 'A', 'B', 'h', 'x')
    with GoalCtx(w, ps): cand = w.T(k.mk_fvar(2))(w.T(k.mk_fvar(3)))
    ps.apply(cand.n, None, True); expect(ps.is_complete(), 'C23 apply (h x) with h : Imps 1 A B and x : A closes the goal B (the hidden arrow is only seen by full delta)'); admit(ps, w.forall_([PROP, PROP, lambda a, b: Imps(S(Z), a, b), lambda a, b, h: a], lambda a, b, h, x: b), 'c_imps2')
    stmt_ap = w.forall_([PROP, PROP, lambda a, b: Imps(S(Z), a, b)], lambda a, b, h: w.arrow(a, b))
    ps = fresh_proof(stmt_ap); av, bv, hv = intros(ps, 'A', 'B', 'h'); ps.intro('x', None, True)
    ps.apply(hv.n, None, True)
    expect(ps.num_goals() == 1 and ps.goal_target(0) == av.n, 'C23b apply h with h : Imps 1 A B to the goal B gives the premise A (the arrow is behind a recursive function: found by full delta)')
    # a negated hypothesis whose argument mentions a definition: the premise of `apply h` is what the hypothesis says, not what the definitions unfold to
    stmt_nh = w.forall_([Nat, Nat, Nat, lambda a, b, c: w['Not'](Le(a, sum3(a, b, c)))], lambda a, b, c, h: w['False'])
    ps = fresh_proof(stmt_nh); av, bv, cv, hv = intros(ps, 'a', 'b', 'c', 'h')
    ps.apply(hv.n, None, True)
    with GoalCtx(w, ps): folded = Le(av, sum3(av, bv, cv))
    expect(ps.num_goals() == 1 and w.T(ps.goal_target(0)) == folded, 'C25 apply (h : Not (a <= sum3 a b c)) to False leaves `a <= sum3 a b c`: %s' % w.show(w.T(ps.goal_target(0)), names=['a', 'b', 'c', 'h']))
    # a definition with two arguments: intro substitutes them in order (a mix-up would give the hypothesis B and the goal A)
    w.define('imp2', w.forall_([PROP, PROP], lambda a, b: PROP), w.fun([PROP, PROP], lambda a, b: w.arrow(a, b)))
    stmt_d = w.forall_([PROP, PROP, lambda a, b: a], lambda a, b, x: w['imp2'](b, a))
    ps = fresh_proof(stmt_d); av, bv, xv = intros(ps, 'A', 'B', 'x')
    ps.intro('y', None, True)
    expect(ps.goal_num_hyps(0) == 4 and ps.goal_hyp_type(0, 3) == bv.n and ps.goal_target(0) == av.n, 'C24 intro of `imp2 B A` gives the hypothesis B and the goal A')
    ps.exact(xv.n); expect(ps.is_complete(), 'C24b finished'); admit(ps, stmt_d, 'c_imp2')

    # ------------------------------------------------------------------------------------------------------------------------------ I. cases and induct write what they introduce as a person would
    # a parameterised family whose parameter is a lambda: the field `R a b` is `(fun x y => y = S x) a z` after the parameter is substituted, and must read `z = S a`
    TYPE = w.TYPE
    rel = lambda A: w.arrow(A, w.arrow(A, PROP))
    w.inductive('Star', w.forall_([TYPE, lambda A: rel(A), lambda A, R: A, lambda A, R, a: A], lambda A, R, a, b: PROP),
                [('star_refl', [TYPE, lambda A: rel(A), lambda A, R: A], lambda A, R, a: [A, R, a, a]),
                 ('star_step', [TYPE, lambda A: rel(A), lambda A, R: A, lambda A, R, a: A, lambda A, R, a, b: A, lambda A, R, a, b, c: R(a, b), lambda A, R, a, b, c, h: w['Star'](A, R, b, c)],
                  lambda A, R, a, b, c, h1, h2: [A, R, a, c])])
    Star = w['Star']
    step = w.lam(Nat, lambda x: w.lam(Nat, lambda y: Eq(y, S(x))))
    stmt_s = w.forall_([Nat, Nat, lambda a, b: Star(Nat, step, a, b)], lambda a, b, h: Eq(a, a))
    for how in ('induct', 'cases'):
        p = Proof(w, stmt_s, 'i_star_' + how); p.intro('a', 'b', 'h')
        if how == 'induct': p.induct('h', names=['x', 'y', 'z', 'r', 'hs', 'ih'])
        else: p.cases('h', names=['x', 'y', 'z', 'r', 'hs'])
        names = [nm for nm, _ in p.goal(1).hyps]
        expect('hs' in names, 'I1 %s: the step case has the hypothesis hs (%s)' % (how, names))
        if 'hs' in names:
            with GoalCtx(w, p.ps, 1):
                fv = lambda nm: w.T(k.mk_fvar(names.index(nm)))
                hs_ty = p.goal(1).hyps[names.index('hs')][1]
                expect(hs_ty == Eq(fv('z'), S(fv('a'))), 'I2 %s: the relation field reads `z = S a`, not a beta-redex: %s' % (how, w.show(hs_ty, names=names)))
                expect(not has_redex(hs_ty), 'I3 %s: no beta-redex in the field' % how)

    # a hypothesis after the scrutinee is reverted and introduced again: it is shown folded (`Zero < m`, not what `<` unfolds to) and keeps its name, also through a second cases
    stmt_l = w.forall_([Nat, Nat, lambda n, m: w['Lt'](n, m)], lambda n, m, hh: Eq(n, n))
    p = Proof(w, stmt_l, 'i_lt'); p.intro('n', 'm', 'hh'); p.cases('n', names=['k'])
    expect(p.n_goals == 2, 'I4 cases n gives two goals')
    names0 = [nm for nm, _ in p.goal(0).hyps]; names1 = [nm for nm, _ in p.goal(1).hyps]
    expect(names0 == ['n', 'm', 'hh', 'm', 'hh'] and names1 == ['n', 'm', 'hh', 'k', 'm', 'hh'], 'I5 the hypotheses that were reverted come back under their names: %s / %s' % (names0, names1))
    with GoalCtx(w, p.ps, 0):
        mv = w.T(k.mk_fvar(3)); expect(p.goal(0).hyps[4][1] == w['Lt'](Z, mv), 'I6 the reverted hypothesis is `Zero < m` (folded): %s' % w.show(p.goal(0).hyps[4][1], names=names0))
    with GoalCtx(w, p.ps, 1):
        kv, mv = w.T(k.mk_fvar(3)), w.T(k.mk_fvar(4)); expect(p.goal(1).hyps[5][1] == w['Lt'](S(kv), mv), 'I7 ... and `Succ k < m` in the other case: %s' % w.show(p.goal(1).hyps[5][1], names=names1))
    p.refl()                                                                                   # the first case
    p.cases(1, names=['j'])                                                                    # the original m: everything after it is reverted, among it k, which only the tactic layer named
    names0 = [nm for nm, _ in p.goal(0).hyps]; names1 = [nm for nm, _ in p.goal(1).hyps]
    expect(names0 == ['n', 'm', 'hh', 'k', 'm', 'hh', 'hh', 'k', 'm', 'hh'] and names1 == ['n', 'm', 'hh', 'k', 'm', 'hh', 'j', 'hh', 'k', 'm', 'hh'], 'I8 a second cases keeps the names of what it reverted, including the one only the tactic layer had given: %s / %s' % (names0, names1))
    expect(p.n_goals == 2 and p.ps.goal_hyp_name(0, 7) == 'k' and p.ps.goal_hyp_name(1, 8) == 'k', 'I9 the kernel knows those names too')

    # ------------------------------------------------------------------------------------------------------------------------------ J. rewriting inside a hypothesis (`rw_at`)
    # `rw_at h lemma` finds the lemma's left side (right side with rev) in the display form of h, and adds the rewritten hypothesis under the same name (or `new`); the original stays, shadowed.
    # The proof is `transport` along the instantiated equation, so everything below is also re-checked by the kernel when the theorem is declared.
    F_ = w.forall_
    def finish(p_, stmt_, name_, last):
        p_.exact(last); expect(p_.done, 'J %s: the proof is finished' % name_)
        k.declare_checked(w.term(stmt_, 's').n, p_.term().n, name_)
    def hyp_summary(p_, i=0):
        return [(nm, w.show(ty, names=[n for n, _ in p_.goal(i).hyps][:j])) for j, (nm, ty) in enumerate(p_.goal(i).hyps)]
    stmt_j1 = F_([Nat, Nat, lambda a, b: Eq(add(a, Z), b)], lambda a, b, h: Eq(a, b))
    p = Proof(w, stmt_j1, 'j1'); p.intro('a', 'b', 'h'); p.rw_at('h', 'add_zero')
    with GoalCtx(w, p.ps):
        a_, b_ = w.T(k.mk_fvar(0)), w.T(k.mk_fvar(1)); hs = p.goal().hyps
        expect(len(hs) == 4 and hs[3][0] == 'h' and hs[3][1] == Eq(a_, b_), 'J1 rw_at adds the rewritten hypothesis `h : a = b`: %s' % hyp_summary(p))
        expect(hs[2][0] == 'h' and hs[2][1] == Eq(add(a_, Z), b_), 'J1b the original stays, shadowed: %s' % hyp_summary(p))
        expect(p.goal().target == Eq(a_, b_), 'J1c the goal is not touched')
    expect(k.view(p.h('h').n) == ('fvar', 3), 'J1d the name `h` now means the new hypothesis (level 3), not the shadowed one')
    finish(p, stmt_j1, 'j1_thm', 'h')
    # a hypothesis that is not called `h`: the rewritten one keeps ITS name
    stmt_j1q = F_([Nat, Nat, lambda a, b: Eq(add(a, Z), b)], lambda a, b, hq: Eq(a, b))
    p = Proof(w, stmt_j1q, 'j1q'); p.intro('a', 'b', 'hq'); p.rw_at('hq', 'add_zero')
    expect([nm for nm, _ in p.goal().hyps] == ['a', 'b', 'hq', 'hq'], 'J1e the rewritten hypothesis keeps the name `hq`: %s' % hyp_summary(p))
    finish(p, stmt_j1q, 'j1q_thm', 'hq')
    # given as a term, the new hypothesis is called `h`
    p = Proof(w, stmt_j1q, 'j1t'); p.intro('a', 'b', 'hq'); p.rw_at(p.h('hq'), 'add_zero')
    expect([nm for nm, _ in p.goal().hyps] == ['a', 'b', 'hq', 'h'], 'J1f a hypothesis given as a term: the new one is called h: %s' % hyp_summary(p))
    finish(p, stmt_j1q, 'j1t_thm', 'h')
    # `new=` names the result; the original keeps its name AND its type
    p = Proof(w, stmt_j1, 'j2'); p.intro('a', 'b', 'h'); p.rw_at('h', 'add_zero', new='h2')
    with GoalCtx(w, p.ps):
        a_, b_ = w.T(k.mk_fvar(0)), w.T(k.mk_fvar(1)); hs = p.goal().hyps
        expect([nm for nm, _ in hs] == ['a', 'b', 'h', 'h2'] and hs[2][1] == Eq(add(a_, Z), b_) and hs[3][1] == Eq(a_, b_), 'J2 new= names the result and the original is unchanged: %s' % hyp_summary(p))
    finish(p, stmt_j1, 'j2_thm', 'h2')
    # reverse direction, and the pattern is read off the DISPLAY form: h is `Succ a + b = c`, shown as `Succ (a + b) = c`, which is what `add_succ` read right to left looks for
    stmt_j3 = F_([Nat, Nat, Nat, lambda a, b, c: Eq(add(S(a), b), c)], lambda a, b, c, h: Eq(add(a, S(b)), c))
    p = Proof(w, stmt_j3, 'j3'); p.intro('a', 'b', 'c', 'h'); p.rw_at('h', 'add_succ', rev=True)
    with GoalCtx(w, p.ps):
        a_, b_, c_ = (w.T(k.mk_fvar(i)) for i in range(3))
        expect(p.goal().hyps[-1][1] == Eq(add(a_, S(b_)), c_), 'J3 rev: `Succ (a + b) = c` becomes `a + Succ b = c`: %s' % hyp_summary(p))
    finish(p, stmt_j3, 'j3_thm', 'h')
    # explicit arguments choose the instance; without them the first instance in the display form is taken (outermost, leftmost)
    stmt_j4 = F_([Nat, Nat, Nat, lambda a, b, c: Eq(add(add(a, b), add(b, a)), c)], lambda a, b, c, h: Eq(add(add(a, b), add(a, b)), c))
    p = Proof(w, stmt_j4, 'j4a'); p.intro('a', 'b', 'c', 'h'); p.rw_at('h', 'add_comm', 'b', 'a')
    with GoalCtx(w, p.ps):
        a_, b_, c_ = (w.T(k.mk_fvar(i)) for i in range(3))
        expect(p.goal().hyps[-1][1] == Eq(add(add(a_, b_), add(a_, b_)), c_), 'J4a add_comm b a: only the instance `b + a` is rewritten: %s' % hyp_summary(p))
    finish(p, stmt_j4, 'j4a_thm', 'h')
    p = Proof(w, stmt_j4, 'j4b'); p.intro('a', 'b', 'c', 'h'); p.rw_at('h', 'add_comm')
    with GoalCtx(w, p.ps):
        a_, b_, c_ = (w.T(k.mk_fvar(i)) for i in range(3))
        expect(p.goal().hyps[-1][1] == Eq(add(add(b_, a_), add(a_, b_)), c_), 'J4b add_comm: the first instance found is the outermost sum (b + a + (a + b)): %s' % hyp_summary(p))
    # every occurrence of the instance is rewritten, not only the first
    stmt_j5 = F_([Nat, lambda a: Eq(add(a, Z), S(add(a, Z)))], lambda a, h: Eq(a, S(a)))
    p = Proof(w, stmt_j5, 'j5'); p.intro('a', 'h'); p.rw_at('h', 'add_zero')
    with GoalCtx(w, p.ps):
        a_ = w.T(k.mk_fvar(0)); expect(p.goal().hyps[-1][1] == Eq(a_, S(a_)), 'J5 both occurrences of `a + 0` are rewritten: %s' % hyp_summary(p))
    finish(p, stmt_j5, 'j5_thm', 'h')
    # no instance: an error that names the pattern, and the proof state is as before
    stmt_j6 = F_([Nat, Nat, lambda a, b: Eq(a, b)], lambda a, b, h: Eq(a, b))
    p = Proof(w, stmt_j6, 'j6'); p.intro('a', 'b', 'h'); before = hyp_summary(p)
    try: p.rw_at('h', 'add_zero'); msg = ''
    except AuthorError as e: msg = str(e)
    expect('no instance of the pattern' in msg and hyp_summary(p) == before and p.n_goals == 1, 'J6 no instance of the pattern: refused, nothing added (%s)' % msg[:60])
    # a variable of the lemma that the pattern does not mention must be given
    w.prove('j_extra', F_([Nat, Nat], lambda n, m: Eq(add(n, Z), n)), lambda q: (q.intro('n', 'm'), q.rw('add_zero')))
    p = Proof(w, stmt_j1, 'j7'); p.intro('a', 'b', 'h')
    try: p.rw_at('h', 'j_extra'); msg = ''
    except AuthorError as e: msg = str(e)
    expect('does not determine' in msg and len(p.goal().hyps) == 3, 'J7 the pattern does not determine the quantified variables: refused (%s)' % msg[:60])
    try: p.rw_at('h', 'j_extra', 'a'); msg = ''
    except AuthorError as e: msg = str(e)
    expect('does not determine' in msg and len(p.goal().hyps) == 3, 'J7b ... also when only the first is given (%s)' % msg[:60])
    p.rw_at('h', 'j_extra', 'a', 'b')
    with GoalCtx(w, p.ps):
        a_, b_ = w.T(k.mk_fvar(0)), w.T(k.mk_fvar(1)); expect(p.goal().hyps[-1][1] == Eq(a_, b_), 'J7c given both, the lemma applies: %s' % hyp_summary(p))
    finish(p, stmt_j1, 'j7_thm', 'h')
    # the LEMMA is read in display form too: `Succ n + m = m + Succ n` reads `Succ (n + m) = m + Succ n`, and that is what it finds in the (display form of the) hypothesis `Succ a + b = c`
    w.prove('j_sac', F_([Nat, Nat], lambda n, m: Eq(add(S(n), m), add(m, S(n)))), lambda q: (q.intro('n', 'm'), q.exact(lambda q2: w['add_comm'](S(q2.h('n')), q2.h('m')))))
    stmt_j9 = F_([Nat, Nat, Nat, lambda a, b, c: Eq(add(S(a), b), c)], lambda a, b, c, h: Eq(add(b, S(a)), c))
    p = Proof(w, stmt_j9, 'j9'); p.intro('a', 'b', 'c', 'h'); p.rw_at('h', 'j_sac')
    with GoalCtx(w, p.ps):
        a_, b_, c_ = (w.T(k.mk_fvar(i)) for i in range(3))
        expect(p.goal().hyps[-1][1] == Eq(add(b_, S(a_)), c_), 'J9 a lemma whose left side is a recursive call on a constructor is found in the display form of the hypothesis: %s' % hyp_summary(p))
    finish(p, stmt_j9, 'j9_thm', 'h')
    # something that is not an equation cannot be rewritten with
    p = Proof(w, stmt_j1, 'j8'); p.intro('a', 'b', 'h')
    expect(raises(lambda: p.rw_at('h', 'succ_ne_zero'), ''), 'J8 a lemma that is not an equation is refused')
    expect(len(p.goal().hyps) == 3, 'J8b ... and nothing was added')

    # ------------------------------------------------------------------------------------------------------------------------------ K. which reading of the lemma meets the goal, and how what unification found is written
    # apply with simp tries its readings of the lemma's conclusion in a fixed order (see apply); each case below is one that only a particular reading, or a particular way of writing down a solution, gets right
    # K1: the goal reaches the conclusion through a definition AND has a constructor where the lemma has a metavariable. `k_def2 (S a) b` is `add (S a) b` once the definition is unfolded, and that computes (its
    # decreasing argument is a constructor) to `S (add a b)`: the lemma's `R (S (add ?n ?m))` meets it first order. Without that reading, the full-delta one meets two stuck matches, defers a constraint and hands ?n ?m back as goals
    w.define('k_def2', w.forall_([Nat, Nat], lambda x, y: Nat), w.fun([Nat, Nat], lambda x, y: add(x, y)))
    w.axiom('k_R', w.arrow(Nat, PROP)); w.axiom('k_lemS', w.forall_([Nat, Nat], lambda n, m: w['k_R'](S(add(n, m)))))
    stmt_k1 = w.forall_([Nat, Nat], lambda a, b: w['k_R'](w['k_def2'](S(a), b)))
    ps = fresh_proof(stmt_k1); intros(ps, 'a', 'b')
    ps.apply(w['k_lemS'].n, None, True)
    expect(ps.num_goals() == 0 and ps.num_pending() == 0, 'K1 apply (R (S (n + m))) to `R (def2 (S a) b)`: both arguments found by first-order matching, nothing left over (%d goals, %d pending)' % (ps.num_goals(), ps.num_pending()))
    expect(ps.is_complete(), 'K1b ... and the proof is complete')
    if ps.is_complete(): admit(ps, stmt_k1, 'k_def2_thm')

    # K2: a lemma whose arrow sits behind a recursive function whose decreasing argument is itself hidden behind a definition (`Imps one A B`, one := S Zero). The folded reading cannot see the arrow (`one` is not
    # a constructor) while full delta can: the folded reading is then abandoned -- it must not be kept (it never found a conclusion), and it must leave nothing behind -- and the lemma is read with every definition
    # unfolded. The premise and the function that unification finds for `f` (a LAMBDA, which is written down when it is quoted) must still read as the lemma states them: `Q (fun x => x + 0)`
    w.define('k_one', Nat, S(Z))
    w.axiom('k_Q', w.arrow(NN, PROP)); w.axiom('k_Q2', w.arrow(NN, PROP)); kQ, kQ2 = w['k_Q'], w['k_Q2']
    w.axiom('k_lemH', w.forall_([NN], lambda f: Imps(w['k_one'], kQ(f), kQ2(f))))          # Q f -> Q2 f, behind `Imps one`
    w.axiom('k_lemA', w.forall_([NN], lambda f: w.arrow(kQ(f), kQ2(f))))                   # the same lemma with its arrow in plain sight
    fn0 = w.lam(Nat, lambda x: add(x, Z))
    w.axiom('k_qq', kQ(fn0))
    stmt_kh = w.forall_([Nat], lambda a: kQ2(fn0))
    made = {}
    for lem in ('k_lemH', 'k_lemA'):
        ps = fresh_proof(stmt_kh); intros(ps, 'a'); m0 = k.fresh_meta(Nat.n, 0)
        try: ps.apply(w[lem].n, None, True); err = ''
        except Exception as e: err = str(e)[:80]
        made[lem] = k.fresh_meta(Nat.n, 0) - m0 - 1
        expect(not err and ps.num_goals() == 1, 'K2 apply %s to `Q2 (fun x => x + 0)` leaves the premise as the one goal (%s)' % (lem, err or '%d goals' % ps.num_goals()))
        if err or ps.num_goals() != 1: continue
        expect(ps.goal_target(0) == kQ(fn0).n, 'K2b %s: the premise reads `Q (fun x => x + 0)`, not an unfolded function: %s' % (lem, w.show(w.T(ps.goal_target(0)), names=['a'])))
        ps.exact(w['k_qq'].n)
        expect(ps.is_complete() and ps.extract() == w.fun([Nat], lambda a: w[lem](fn0, w['k_qq'])).n, 'K2c %s: and the function found for f is written folded in the proof term: %s' % (lem, w.show(w.T(ps.extract()))))
    expect(len(made) == 2 and made['k_lemH'] == made['k_lemA'], 'K2d the abandoned reading leaves no metavariable behind: %s created for the hidden arrow, %s for the plain one' % (made.get('k_lemH'), made.get('k_lemA')))
    # ... and without simp the solution is written the way unification found it, definitions unfolded: the folded writing belongs to the simp readings alone (it must neither be missing from them nor outlive them --
    # this runs after many simp applications, so a folding mode that is never switched off again shows here)
    unfolded = w.normalize(fn0)
    expect(unfolded != fn0 and w.conv(unfolded, fn0), 'K3 (test setup) the normal form of `fun x => x + 0` is another, convertible term')
    for lem in ('k_lemH', 'k_lemA'):
        ps = fresh_proof(stmt_kh); intros(ps, 'a'); ps.apply(w[lem].n, None, False); ps.exact(w['k_qq'].n)
        expect(ps.is_complete() and ps.extract() == w.fun([Nat], lambda a: w[lem](unfolded, w['k_qq'])).n, 'K3b %s without simp: the function found for f is the unfolded one: %s' % (lem, w.show(w.T(ps.extract()))))

    # K4: an arrow behind TWO definitions (`Imp2 A B := Imp A B`, `Imp A B := A -> B`): the folded reading unfolds one definition after the other until the arrow shows, so the premise is the lemma's, not its unfolding
    w.define('k_Imp2', w.forall_([PROP, PROP], lambda a, b: PROP), w.fun([PROP, PROP], lambda a, b: Imp(a, b)))
    w.axiom('k_lem2', w.forall_([Nat, Nat], lambda x, y: w['k_Imp2'](Eq(add(x, y), add(y, x)), Eq(add(x, y), add(y, x)))))
    stmt_k4 = w.forall_([Nat, Nat], lambda x, y: Eq(add(x, y), add(y, x)))
    ps = fresh_proof(stmt_k4); xv, yv = intros(ps, 'x', 'y')
    with GoalCtx(w, ps): folded = Eq(add(xv, yv), add(yv, xv)); pf = w['add_comm'](xv, yv)
    ps.apply(w['k_lem2'].n, None, True)
    expect(ps.num_goals() == 1 and ps.goal_target(0) == folded.n, 'K4 the premise behind two definitions reads `x + y = y + x`: %s' % [w.show(w.T(ps.goal_target(i)), names=['x', 'y']) for i in range(ps.num_goals())])
    try: ps.exact(pf.n)
    except Exception as e: expect(False, 'K4b the premise is closed by add_comm x y: %s' % str(e)[:80])
    expect(ps.is_complete(), 'K4c finished')
    if ps.is_complete(): admit(ps, stmt_k4, 'k_lem2_thm')

    # K5: a candidate headed by a HYPOTHESIS whose second argument goes through a hidden arrow: the premise that is left is written the way the hypothesis states it, not with `+` unfolded
    w.axiom('k_Q0', w.arrow(Nat, PROP)); kQ0 = w['k_Q0']
    stmt_k5 = w.forall_([Nat, lambda n: w.forall_([Nat], lambda x: Imp(Le(x, x), Imp(Eq(add(x, x), x), kQ0(x)))), lambda n, h: Le(n, n)], lambda n, h, hl: kQ0(n))
    ps = fresh_proof(stmt_k5); nv, hv, hlv = intros(ps, 'n', 'h', 'hl')
    with GoalCtx(w, ps): cand = hv(nv, hlv); folded = Eq(add(nv, nv), nv)
    ps.apply(cand.n, None, True)
    expect(ps.num_goals() == 1 and ps.goal_target(0) == folded.n, 'K5 apply (h n hl) leaves `n + n = n` as the premise: %s' % [w.show(w.T(ps.goal_target(i)), names=['n', 'h', 'hl']) for i in range(ps.num_goals())])

    # K6: a constructor field that is a definition (`Zero < n`, with `x < y` defined as `Succ x <= y`): `cases` shows the field as the constructor states it
    w.inductive('k_Box', w.forall_([Nat], lambda n: PROP), [('k_mk_box', [Nat, lambda n: w['Lt'](Z, n)], lambda n, h: [n])])
    stmt_k6 = w.forall_([Nat, lambda m: w['k_Box'](m)], lambda m, b: Eq(m, m))
    p = Proof(w, stmt_k6, 'k_box'); p.intro('m', 'b'); p.cases('b')
    with GoalCtx(w, p.ps):
        mv = w.T(k.mk_fvar(0)); field_ty = p.goal().hyps[-1][1]
        expect(field_ty == w['Lt'](Z, mv), 'K6 the field of the only case reads `0 < m`, not what `<` unfolds to: %s' % w.show(field_ty, names=[nm for nm, _ in p.goal().hyps][:-1]))
    p.refl(); expect(p.done, 'K6b finished')

    # K7: a recursive function whose body is a constructor, not a match on the decreasing argument: on a variable it must stay folded (its unfolding would be shown whatever the argument was), on a constructor
    # it computes one layer
    w.recursive('k_hh', w.arrow(Nat, Nat), lambda f: w.fun([Nat], lambda n: S(w.match(n, lambda _: Nat, [Z, lambda p: f(p)]))), on=0)
    khh = w['k_hh']
    with Ctx(w, 'm') as (m,):
        expect(w.simp(khh(m)) == khh(m), 'K7 `hh m` stays folded: %s' % w.show(w.simp(khh(m))))
        expect(w.simp(khh(S(m))) == S(khh(m)), 'K7b `hh (S m)` computes one layer: %s' % w.show(w.simp(khh(S(m)))))
        expect(w.simp(khh(Z)) == S(Z), 'K7c `hh 0` computes: %s' % w.show(w.simp(khh(Z))))

    # L: a finished proof has no goal: every tactic that needs one says so as an author error -- never a C++ exception from the kernel's goal table (`rw`, `focus` and `h` used to reach it)
    def finished():
        pf = Proof(w, w.forall_([Nat], lambda n: w.Eq(n, n)), 'l_done'); pf.intro('n'); pf.refl(); return pf
    ladd0 = w['add_zero']
    no_goal_calls = {
        'intro': lambda p: p.intro('x'), 'apply': lambda p: p.apply('add_zero'), 'exact': lambda p: p.exact('add_zero'), 'refl': lambda p: p.refl(), 'cases': lambda p: p.cases('n'),
        'induct': lambda p: p.induct('n'), 'rw': lambda p: p.rw('add_zero'), 'rw_at': lambda p: p.rw_at('n', 'add_zero'), 'rewrite': lambda p: p.rewrite('add_zero'),
        'have': lambda p: p.have('e', w.Eq(Z, Z)), 'obtain': lambda p: p.obtain(['a', 'b'], 'add_zero'), 'exists': lambda p: p.exists('n'), 'ring': lambda p: p.ring(),
        'calc': lambda p: p.calc(Z), 'assumption': lambda p: p.assumption(), 'norm': lambda p: p.norm(), 'h': lambda p: p.h('n'), 'focus': lambda p: p.focus().__enter__(),
        'goal': lambda p: p.goal(), 'left': lambda p: p.left(), 'right': lambda p: p.right(), 'split': lambda p: p.split(), 'exfalso': lambda p: p.exfalso(), 'induction_on': lambda p: p.induction_on('nat_ind', 'n'),
    }
    expect(finished().done, 'L0 the probe proof is finished')
    for label, call in no_goal_calls.items():
        try: call(finished()); got = 'no error'
        except AuthorError: got = 'AuthorError'
        except Exception as e: got = '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:70])
        expect(got == 'AuthorError', 'L1 %s on a finished proof is an author error (got %s)' % (label, got))
    pf = Proof(w, w.forall_([Nat, Nat], lambda a, b: w.Eq(add(a, b), add(b, a))), 'l_two'); pf.intro('a', 'b')
    for label, call in (('h', lambda: pf.h('a', 1)), ('focus', lambda: pf.focus(1).__enter__()), ('goal', lambda: pf.goal(1)), ('h negative', lambda: pf.h('a', -1))):
        try: call(); got = 'no error'
        except AuthorError: got = 'AuthorError'
        except Exception as e: got = '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:70])
        expect(got == 'AuthorError', 'L2 %s of a goal that does not exist is an author error (got %s)' % (label, got))
    expect(pf.h('a') is not None and not pf.done, 'L3 the open goal is still usable after the refused calls')

    # nothing leaks out of the folded mode into later work
    with Ctx(w, 'm', 'n') as (m, n):
        expect(w.view(w.normalize(add(m, n)))[0] == 'match', 'D1 after the folded-mode calls, normalisation is full delta again')
    print("   %d checks (%.1f s)" % (CHECKS[0], time.time() - t0))
    for f in FAIL[:40]: print("   FAIL " + f)
    if len(FAIL) > 40: print("   ... and %d more" % (len(FAIL) - 40))
    print("S22 SIMP GATE: " + ("PASS" if not FAIL else "FAIL"))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
