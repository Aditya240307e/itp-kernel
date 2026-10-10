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
import os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itp_env_lnl as E
import nat_lib
from author import AuthorError

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
        # a partial application of a recursive function
        t = add(S(m)); got = w.simp(t)
        expect(w.view(got)[0] == 'lam' and w.conv(t, got), 'A6 add (S m) unapplied becomes a function: %s' % w.show(got))
        expect(w.simp(got) == got and w.conv(got, w.lam(Nat, lambda y: S(add(m, y)))), 'A7 and the function is the expected one')
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
