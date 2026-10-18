"""S35 gate: the kernel checks a `match` against the type of its scrutinee AS WRITTEN, and the result type of the match is read off syntax too.

A `match` is typed in three steps: the family the scrutinee's type belongs to and its arguments (`scrutinee_family`), the type each branch must have (`check_match_branches`), and the type of the whole
match (`infer`/`check_match`: the motive applied to the indices and the scrutinee). Each of them used to evaluate the types it handled and quote them back, which unfolds every definition INSIDE the
arguments as well: for a conjunction of divisibilities over Z that was `Z.mul` expanded to its quotient construction on every match (70 ms for a three line proof), and for a hypothesis that depends on the
scrutinee (the usual `cases` with something to revert) the match was typed as `(match h ...) hyp` and the whole result type was normalised: 2.5 million steps for a five line proof about `Q big`, where
`big` is a definition that computes to 16384 successors. The three steps now work on syntax: a definition at the head of the scrutinee's type is unfolded one step at a time (`unfold_head_folded`) until
the family appears, and the motive is applied by BETA ON SYNTAX (`motive_result_syntactic`). Evaluation is the fallback, for a type that reaches its family only by computation.

Nothing here is trusted (the independent checker re-checks every accepted term and every refused one is refused by it as well); the gate pins down

  A. COST OF PROOFS BUILT BY `cases`, in kernel steps (`Kernel.fuel_used`, deterministic): a hypothesis `h : Q big` is taken apart in a few hundred steps whatever `big` computes to; so is one headed by a
     chain of ten definitions, and one with a hypothesis that depends on the scrutinee (the case that cost 2.5 million);
  B. THE FALLBACK: a type that reaches its family only by computation (`T Zero` with `T` a definition by cases, `RT Zero` with `RT` recursive) is still accepted -- by evaluation, which is expensive;
  C. RAW MATCHES, below the tactic layer: the loop that unfolds definitions at the head of the scrutinee's type stops after 32 unfoldings (31 are read on syntax: cheap; the 32nd is read by
     evaluation: accepted, but at the price of computing `big`), a branch of the wrong type is refused under every route;
  D. A SCRUTINEE WHOSE TYPE IS NOT AN INDUCTIVE FAMILY is refused, however the type is reached (a function type, a definition that unfolds to one, a type variable, an axiom, a recursive function stuck on a
     variable), for matches with one branch and with none (a match without branches on an axiom would prove anything), whether the match is checked against a type or is the head of an application;
  E. THE ORDER IN WHICH THE MOTIVE IS APPLIED: for a family with two indices (`Le a b`) the motive's body receives the indices and then the scrutinee: the induction principle of `Le`, written as a raw match
     and as the head of an application, is accepted; the same with the arguments of the motive swapped, and with a hypothesis of the wrong type, is refused; a motive that is not a lambda (a global) works;
  F. DEFINITIONS AT THE HEAD OF AN INDEXED FAMILY under `cases` with something to revert (`EqN a b := a = b`, `Le'`): the proofs are accepted;
  G. every accepted term is accepted by the independent checker, every refused one is refused by it.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, time

sys.setrecursionlimit(100000)
import itp_env_lnl as E
import lnl_base
import indep_checker as ic
from author import AuthorError, Term
from tactics import Proof
from harness import parse_raw

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S35_TRACE'))
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


COSTS = [True]               # the cost assertions are made in the world where `big` computes to 16384 successors (the independent checker is naive: it is given the small one)
CHEAP = 5000                  # steps: what a proof that is read on syntax costs (a few hundred); computing `big` costs ~200000 and more
DEAR = 50000                  # steps: what a proof that has to evaluate `big` costs at the least
DEPTH = 33                    # the longest chain of definitions built


def num(w, n):
    t = w['Zero']
    for _ in range(n): t = w['Succ'](t)
    return t


def build(log2):
    w = lnl_base.build_base()
    Nat, PROP, TYPE, S, Z0, Eq, And, TRUE, FALSE = w['Nat'], w.PROP, w.TYPE, w['Succ'], w['Zero'], w.Eq, w['And'], w['True'], w['False']
    # a term that is cheap to write and expensive to compute: big = pow2 log2  (16384 successors for log2 = 14)
    w.recursive('pow2', w.arrow(Nat, Nat), lambda R: w.lam(Nat, lambda n: w.match(n, lambda _: Nat, [S(Z0), lambda m: w['add'](R(m), R(m))])))
    w.define('big', Nat, w['pow2'](num(w, log2)))
    # Q n := n = n /\ True, and a chain of definitions above it:  CH1 n := Q n,  CH2 n := CH1 n, ...
    w.define('Q', w.arrow(Nat, PROP), w.lam(Nat, lambda n: And(Eq(n, n), TRUE)))
    prev = w['Q']
    for j in range(1, DEPTH + 1):
        w.define('CH%d' % j, w.arrow(Nat, PROP), w.lam(Nat, lambda n, pv=prev: pv(n)))
        prev = w['CH%d' % j]
    # a type that reaches its family only by computation: by cases (not recursive), and recursive
    w.define('T', w.arrow(Nat, PROP), w.lam(Nat, lambda n: w.match(n, lambda _: PROP, [w['Q'](w['big']), lambda m: FALSE])))
    w.recursive('RT', w.arrow(Nat, PROP), lambda R: w.lam(Nat, lambda n: w.match(n, lambda _: PROP, [w['Q'](w['big']), lambda m: R(m)])))
    # a recursive type-level function with an argument that stays a variable: the family it computes to mentions the local variable under a binder (the arguments of the family are quoted at the depth of the context)
    w.recursive('RE2', w.arrow(Nat, w.arrow(Nat, PROP)), lambda R: w.fun([Nat, Nat], lambda a, n: w.match(n, lambda _: PROP, [And(w.forall_([Nat], lambda x: Eq(x, a)), TRUE), lambda m: R(a, m)])), on=1)
    # definitions at the head of indexed families
    w.define('EqN', w.forall_([Nat, Nat], lambda a, b: PROP), w.fun([Nat, Nat], lambda a, b: Eq(a, b)))
    w.define('Le2', w.forall_([Nat, Nat], lambda a, b: PROP), w.fun([Nat, Nat], lambda a, b: w['Le'](a, b)))
    # an axiom that is a proposition, and a motive that is a global
    w.axiom('ax', PROP)
    w.define('natmot', w.arrow(Nat, TYPE), w.lam(Nat, lambda n: w.arrow(Nat, Nat)))
    Le = w['Le']
    w.define('lemot', w.forall_([Nat, Nat, lambda n, m: Le(n, m)], lambda n, m, s: PROP), w.fun([Nat, Nat, lambda n, m: Le(n, m)], lambda n, m, s: TRUE))
    w.define('lemotF', w.forall_([Nat, Nat, lambda n, m: Le(n, m)], lambda n, m, s: PROP), w.fun([Nat, Nat, lambda n, m: Le(n, m)], lambda n, m, s: FALSE))
    return w


def kernel(w, val, ty):
    """(accepted, steps, message): the kernel's verdict on `val : ty` and what it cost"""
    k = w.k; k.set_fuel(0)
    ok, msg = k.check_open(val.n, ty.n, [])
    if TRACE: print('    [%s] %d steps%s' % ('ok' if ok else 'refused', k.fuel_used(), '' if ok else ': ' + str(msg)[:80]))
    return bool(ok), k.fuel_used(), msg


def tactic(w, name, stmt, script):
    p = Proof(w, stmt, name); script(p)
    expect(p.done, '%s: the proof was not finished:\n%s' % (name, p))
    return p


def raw_and(w, sty, params, fn, motive_res=None):
    """the term  fun (h : sty) => match h with conj a b => fn a b   as a raw match: `sty` is a definition that unfolds to  And params...  (a type `w.match` cannot take apart), the motive returns `motive_res`"""
    res = w['True'] if motive_res is None else motive_res
    def build_(h):
        mot = w._motive(w.gid['And'], params, sty, lambda s: res)
        br = w._branch('conj', params, fn)
        return w.T(w.k.mk_match(h.n, mot.n, [br.n]))
    return w.lam(sty, build_)


def part_A(w, jobs):
    Nat, Eq, And, TRUE, PROP = w['Nat'], w.Eq, w['And'], w['True'], w.PROP
    big, Q = w['big'], w['Q']

    def sc(p): p.intro('h'); p.cases('h', names=['a', 'b']); p.apply('conj'); p.exact('a'); p.exact('b')
    for label, D in (('Q', Q), ('CH1', w['CH1']), ('CH10', w['CH10']), ('CH14', w['CH14'])):
        stmt = w.forall_([D(big)], lambda h: And(Eq(big, big), TRUE))
        p = tactic(w, 'ma_cases_' + label, stmt, sc)
        ok, steps, msg = kernel(w, p.term(), p.stmt)
        expect(ok, 'A %s big: the kernel rejected the proof of a conjunction taken apart by cases: %s' % (label, msg))
        if COSTS[0]: expect(steps < CHEAP, 'A %s big: taking the hypothesis apart took %d kernel steps (more than %d): the type of the scrutinee was evaluated, `big` computed' % (label, steps, CHEAP))
        jobs.append(('A ' + label, p.term(), p.stmt, True))

    # the tactic layer unfolds at most 15 definitions on syntax (16 types are read); behind a longer chain the proof is still found and accepted, but from the evaluated type: `big` is computed
    stmt = w.forall_([w['CH15'](big)], lambda h: And(Eq(big, big), TRUE))
    p = tactic(w, 'ma_cases_CH15', stmt, sc)
    ok, steps, msg = kernel(w, p.term(), p.stmt)
    expect(ok, 'A CH15 big: the kernel rejected the proof: %s' % msg)
    if COSTS[0]: expect(steps > DEAR, 'A CH15 big: only %d kernel steps: cases is meant to read the type on syntax up to 15 unfoldings and to evaluate it behind a longer chain of definitions' % steps)
    jobs.append(('A CH15', p.term(), p.stmt, True))

    # a hypothesis that depends on the scrutinee: cases reverts it, the match becomes the head of an application and the motive has to be applied without evaluating anything
    stmt = w.forall_([Q(big), lambda h: w.arrow(Q(big), PROP), lambda h, P: P(h)], lambda h, P, x: TRUE)
    def sc_dep(p): p.intro('h', 'P', 'x'); p.cases('h', names=['a', 'b']); p.apply('tt')
    p = tactic(w, 'ma_cases_dep', stmt, sc_dep)
    ok, steps, msg = kernel(w, p.term(), p.stmt)
    expect(ok, 'A dependent hypothesis: the kernel rejected the proof: %s' % msg)
    if COSTS[0]: expect(steps < CHEAP, 'A dependent hypothesis: %d kernel steps (more than %d): the type of the match (the motive applied) or the type of a branch was normalised' % (steps, CHEAP))
    jobs.append(('A dependent', p.term(), p.stmt, True))
    # ... and one that uses the reverted hypothesis, so that the branch really has the extra binders
    stmt = w.forall_([Q(big), lambda h: w.arrow(Q(big), PROP), lambda h, P: P(h)], lambda h, P, x: P(h))
    def sc_dep2(p): p.intro('h', 'P', 'x'); p.cases('h', names=['a', 'b']); p.exact('x')
    p = tactic(w, 'ma_cases_dep2', stmt, sc_dep2)
    ok, steps, msg = kernel(w, p.term(), p.stmt)
    expect(ok, 'A dependent hypothesis, used: the kernel rejected the proof: %s' % msg)
    if COSTS[0]: expect(steps < CHEAP, 'A dependent hypothesis, used: %d steps' % steps)
    jobs.append(('A dependent, used', p.term(), p.stmt, True))


def part_B(w, jobs):
    Nat, Eq, And, TRUE, FALSE = w['Nat'], w.Eq, w['And'], w['True'], w['False']
    big, Z0, S = w['big'], w['Zero'], w['Succ']
    def sc(p): p.intro('h'); p.cases('h', names=['a', 'b']); p.apply('conj'); p.exact('a'); p.exact('b')
    for label, fam, arg in (('T Zero', w['T'], Z0), ('RT Zero', w['RT'], Z0), ('RT (Succ Zero)', w['RT'], S(Z0))):
        stmt = w.forall_([fam(arg)], lambda h: And(Eq(big, big), TRUE))
        try: p = tactic(w, 'mb_' + label.replace(' ', '_').replace('(', '').replace(')', ''), stmt, sc)
        except AuthorError as e: expect(False, 'B %s: the tactic layer failed: %s' % (label, str(e)[:200])); continue
        ok, steps, msg = kernel(w, p.term(), p.stmt)
        expect(ok, 'B %s: a hypothesis whose type reaches its family only by computation was not accepted: %s' % (label, msg))
        jobs.append(('B ' + label, p.term(), p.stmt, True))
    # ... with a local variable inside the arguments of the family, under a binder
    RE2 = w['RE2']
    stmt = w.forall_([Nat, lambda a: RE2(a, Z0)], lambda a, h: w.forall_([Nat], lambda x: Eq(x, a)))
    def sc2(p): p.intro('a', 'h'); p.cases('h', names=['f', 't']); p.exact('f')
    try: p = tactic(w, 'mb_RE2', stmt, sc2)
    except AuthorError as e: expect(False, 'B RE2: the tactic layer failed: %s' % str(e)[:200]); return
    ok, steps, msg = kernel(w, p.term(), p.stmt)
    expect(ok, 'B a recursive type-level function that computes to a conjunction mentioning a local variable under a binder: not accepted: %s' % msg)
    jobs.append(('B RE2', p.term(), p.stmt, True))


def part_C(w, jobs):
    Nat, Eq, And, TRUE = w['Nat'], w.Eq, w['And'], w['True']
    big = w['big']; params = [Eq(big, big), TRUE]
    tt = w['tt']
    costs = {}
    for j in (1, 5, 15, 29, 30, 31, 32, 33):
        D = w['CH%d' % j]; sty = D(big)
        val = raw_and(w, sty, params, lambda a, b: tt); ty = w.forall_([sty], lambda h: TRUE)
        ok, steps, msg = kernel(w, val, ty); costs[j] = steps
        expect(ok, 'C a raw match on a hypothesis headed by %d definitions was rejected: %s' % (j + 1, msg))
        jobs.append(('C chain %d' % j, val, ty, True))
    for j in (1, 5, 15, 29, 30):
        if COSTS[0]: expect(costs[j] < CHEAP, 'C %d unfoldings (%d definitions above Q and Q itself): %d kernel steps (more than %d): not read on syntax' % (j + 1, j, costs[j], CHEAP))
    for j in (31, 32, 33):
        if COSTS[0]: expect(costs[j] > DEAR, 'C %d unfoldings: only %d kernel steps (fewer than %d): the loop that unfolds definitions on syntax is meant to give up after 32 unfoldings and evaluate' % (j + 1, costs[j], DEAR))
    # a branch of the wrong type (a proof of  big = big  where True is wanted) is refused under each route: one definition, the longest syntactic chain, the first that needs evaluation
    for j in (0, 1, 30, 31):
        D = w['Q'] if j == 0 else w['CH%d' % j]; sty = D(big)
        val = raw_and(w, sty, params, lambda a, b: a); ty = w.forall_([sty], lambda h: TRUE)
        ok, steps, msg = kernel(w, val, ty)
        expect(not ok, 'C a branch of the wrong type (a : big = big where True is wanted) was accepted under %d unfoldings' % (j + 1))
        jobs.append(('C wrong branch %d' % j, val, ty, False))
    # ... and one that swaps the two fields (b : True where  big = big  is wanted)
    sty = w['Q'](big)
    val = raw_and(w, sty, params, lambda a, b: b, motive_res=Eq(big, big)); ty = w.forall_([sty], lambda h: Eq(big, big))
    ok, steps, msg = kernel(w, val, ty)
    expect(not ok, 'C the second field of the conjunction was accepted as a proof of  big = big')
    jobs.append(('C swapped fields', val, ty, False))
    val = raw_and(w, sty, params, lambda a, b: a, motive_res=Eq(big, big)); ty = w.forall_([sty], lambda h: Eq(big, big))
    ok, steps, msg = kernel(w, val, ty)
    expect(ok, 'C the first field is a proof of  big = big: the kernel rejected it: %s' % msg)
    if COSTS[0]: expect(steps < CHEAP, 'C the first field: %d steps' % steps)
    jobs.append(('C first field', val, ty, True))


def part_D(w, jobs):
    Nat, Eq, And, TRUE, FALSE, PROP, TYPE = w['Nat'], w.Eq, w['And'], w['True'], w['False'], w.PROP, w.TYPE
    Z0, S, big = w['Zero'], w['Succ'], w['big']
    k = w.k
    ax = w['ax']; RT = w['RT']
    NotF = w['Not'](FALSE)
    NN = w.arrow(Nat, Nat)
    cases = []
    # (label, type of the scrutinee as a function of the context variables, binder types of the context)
    # every one is tried with a match that has one branch (Zero) and with a match that has none (the dangerous shape: it eliminates into any type), checked against a type and as the head of an application
    def attempt(label, doms, sty_fn, nbr):
        def val_check(*vs):
            h = vs[-1]; sty = sty_fn(*vs[:-1])
            mot = w.lam(sty, lambda s: FALSE)
            return w.T(k.mk_match(h.n, mot.n, [Z0.n] * nbr))
        def val_head(*vs):
            h = vs[-1]; sty = sty_fn(*vs[:-1])
            mot = w.lam(sty, lambda s: NN)
            return w.app(w.T(k.mk_match(h.n, mot.n, [w.lam(Nat, lambda x: x).n] * nbr)), Z0)
        for kind, vf, res in (('checked', val_check, FALSE), ('head', val_head, Nat)):
            ctx_doms = list(doms) + [lambda *vs: sty_fn(*vs)]
            try:
                val = w.fun(ctx_doms, vf); ty = w.forall_(ctx_doms, lambda *vs: res)
            except AuthorError as e:
                expect(False, 'D %s (%s, %d branches): could not build the raw term: %s' % (label, kind, nbr, str(e)[:160])); continue
            ok, steps, msg = kernel(w, val, ty)
            expect(not ok, 'D %s (%s, %d branches): a match on a term whose type is not an inductive family was accepted' % (label, kind, nbr))
            expect('inductive' in msg, 'D %s (%s, %d branches): refused, but not for that reason: %s' % (label, kind, nbr, msg))
            jobs.append(('D %s %s %d' % (label, kind, nbr), val, ty, False))
    for nbr in (0, 1):
        attempt('a function type', [], lambda: NN, nbr)
        attempt('Not False (a definition that unfolds to a function type)', [], lambda: NotF, nbr)
        attempt('a type variable', [TYPE, ], lambda A: A, nbr)
        attempt('an axiom', [], lambda: ax, nbr)
        attempt('a recursive function stuck on a variable', [Nat], lambda n: RT(n), nbr)
        attempt('Type', [], lambda: TYPE, nbr)


def part_E(w, jobs):
    Nat, PROP, TRUE, S, Z0 = w['Nat'], w.PROP, w['True'], w['Succ'], w['Zero']
    Le = w['Le']; tt = w['tt']
    Le_n = lambda n: w.app(w['le_n'], n)
    Le_S = lambda n, m, h: w.app(w['le_S'], n, m, h)
    LEHYP = lambda n, m: Le(n, m)
    # `Le a b` has one parameter (its first argument, which every constructor keeps) and one index: the kernel's branches for it are  le_n => (no field),  le_S => (m, h2)  with the parameter `a` in the types.
    # The induction principle of Le:  forall P, (forall n, P n n (le_n n)) -> (forall n m h, P n (S m) (le_S n m h)) -> forall a b (h : Le a b), P a b h
    # (`wrong`: the second hypothesis is  P n m h  instead, which does not prove what the second constructor needs)
    def ind_tel(wrong):
        return [w.forall_([Nat, Nat, lambda n, m: Le(n, m)], lambda n, m, s: PROP),
                lambda P: w.forall_([Nat], lambda n: P(n, n, Le_n(n))),
                lambda P, hp1: w.forall_([Nat, Nat, lambda n, m: Le(n, m)], lambda n, m, h: P(n, m, h) if wrong else P(n, S(m), Le_S(n, m, h))),
                Nat, Nat, lambda P, hp1, hp2, a, b: Le(a, b)]
    ind_concl = lambda P, hp1, hp2, a, b, h: P(a, b, h)
    ind_motive = lambda P, extra=None: w.fun([Nat, Nat, lambda n, m: Le(n, m)], lambda n, m, s: P(n, m, s) if extra is None else w.arrow(extra, P(n, m, s)))
    for wrong in (False, True):
        tel = ind_tel(wrong); stmt = w.forall_(tel, ind_concl)
        val = w.fun(tel, lambda P, hp1, hp2, a, b, h: w.match(h, ind_motive(P), [lambda: hp1(a), lambda m, h2: hp2(a, m, h2)]))
        ok, steps, msg = kernel(w, val, stmt)
        expect(ok != wrong, 'E the induction principle of Le as a raw match (checked against its type): accepted=%s with %s (%s)' % (ok, 'a wrong hypothesis for the second constructor' if wrong else 'the right hypotheses', msg))
        if COSTS[0]: expect(steps < CHEAP, 'E the induction principle of Le took %d kernel steps' % steps)
        jobs.append(('E induction principle wrong=%s' % wrong, val, stmt, not wrong))
        # the same as the head of an application:  (match h with motive (fun n m s => True -> P n m s) ...) tt
        val = w.fun(tel, lambda P, hp1, hp2, a, b, h: w.app(w.match(h, ind_motive(P, TRUE), [lambda: w.lam(TRUE, lambda _: hp1(a)), lambda m, h2: w.lam(TRUE, lambda _: hp2(a, m, h2))]), tt))
        ok, steps, msg = kernel(w, val, stmt)
        expect(ok != wrong, 'E the induction principle of Le as the head of an application: accepted=%s with %s (%s)' % (ok, 'a wrong hypothesis for the second constructor' if wrong else 'the right hypotheses', msg))
        jobs.append(('E head position wrong=%s' % wrong, val, stmt, not wrong))
    # in the head position the type of the match is  True -> P a b h:  applied to a natural number it must fail
    tel = ind_tel(False); stmt = w.forall_(tel, ind_concl)
    val = w.fun(tel, lambda P, hp1, hp2, a, b, h: w.app(w.match(h, ind_motive(P, TRUE), [lambda: w.lam(TRUE, lambda _: hp1(a)), lambda m, h2: w.lam(TRUE, lambda _: hp2(a, m, h2))]), Z0))
    ok, steps, msg = kernel(w, val, stmt)
    expect(not ok, 'E the match as the head of an application was applied to a natural number where its type asks for a proof of True')
    jobs.append(('E head position, wrong argument', val, stmt, False))
    # the indices in the result of the match:  (match h : Le a b ...) : R a b -> R a b  applied to a proof of R a b is accepted, applied to a proof of R b a it is refused
    for good in (True, False):
        tel = [w.arrow(Nat, w.arrow(Nat, PROP)), Nat, Nat, lambda R, a, b: Le(a, b), (lambda R, a, b, h: R(a, b)) if good else (lambda R, a, b, h: R(b, a))]
        stmt = w.forall_(tel, lambda R, a, b, h, hr: R(a, b))
        def body(R, a, b, h, hr):
            mot = w.fun([Nat, Nat, lambda n, m: Le(n, m)], lambda n, m, s: w.arrow(R(n, m), R(n, m)))
            return w.app(w.match(h, mot, [lambda: w.lam(R(a, a), lambda r: r), lambda m_, h2: w.lam(R(a, S(m_)), lambda r: r)]), hr)
        val = w.fun(tel, body)
        ok, steps, msg = kernel(w, val, stmt)
        expect(ok == good, 'E (match h : Le a b with result R a b -> R a b) applied to a proof of %s: accepted=%s (%s)' % ('R a b' if good else 'R b a', ok, msg))
        jobs.append(('E R a b applied, good=%s' % good, val, stmt, good))
    # a motive that is a global and not a lambda: the result of the match is  natmot n,  evaluated to a function type when the match is applied
    natmot = w['natmot']
    br = lambda: [w.lam(Nat, lambda x: x), lambda m: w.lam(Nat, lambda x: S(x))]
    stmt = w.forall_([Nat], lambda n: Nat)
    val = w.fun([Nat], lambda n: w.app(w.match(n, natmot, br()), Z0))
    ok, steps, msg = kernel(w, val, stmt)
    expect(ok, 'E a match whose motive is a global, as the head of an application: %s' % msg)
    jobs.append(('E global motive head', val, stmt, True))
    stmt2 = w.forall_([Nat], lambda n: w.arrow(Nat, Nat))
    val = w.fun([Nat], lambda n: w.match(n, natmot, br()))
    ok, steps, msg = kernel(w, val, stmt2)
    expect(ok, 'E a match whose motive is a global, checked against  Nat -> Nat : %s' % msg)
    jobs.append(('E global motive checked', val, stmt2, True))
    ok, steps, msg = kernel(w, val, stmt)
    expect(not ok, 'E a match whose motive is a global, checked against Nat (its type is  natmot n,  a function type)')
    jobs.append(('E global motive checked, wrong type', val, stmt, False))
    # ... and on a family with a parameter and an index (Le): the result of the match is  lemot a b h,  the type of a branch  lemot a a (le_n a),  both evaluated when they are compared
    for mot, good in (('lemot', True), ('lemotF', False)):
        stmt3 = w.forall_([Nat, Nat, lambda a, b: Le(a, b)], lambda a, b, h: TRUE)
        val = w.fun([Nat, Nat, lambda a, b: Le(a, b)], lambda a, b, h: w.match(h, w[mot], [lambda: tt, lambda m, h2: tt]))
        ok, steps, msg = kernel(w, val, stmt3)
        expect(ok == good, 'E a match on  Le a b  whose motive is the global %s: accepted=%s (%s)' % (mot, ok, msg))
        jobs.append(('E global motive %s' % mot, val, stmt3, good))
    val = w.fun([Nat], lambda n: w.app(w.match(n, natmot, br()), TRUE))
    ok, steps, msg = kernel(w, val, stmt)
    expect(not ok, 'E a match whose motive is a global, applied to a proposition where a natural number is asked for')
    jobs.append(('E global motive wrong argument', val, stmt, False))


def part_G(w, jobs):
    """every accepted term is accepted by the independent checker, every refused one is refused by it"""
    k = w.k
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid),
                recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    chk = ic.Checker(env, params=True, fuel=400_000_000, prop_elim=True)
    expect(chk.quot_problem is None and not chk.well_founded_problems() and not chk.recursion_problems(), 'G the independent checker accepts the declarations of the test world')
    agree = {'accepted': 0, 'refused': 0, 'unsupported': 0}
    for label, val, ty, want in jobs:
        try: ok, msg = chk.check_closed(parse_raw(k.dump_raw(val.n), True), parse_raw(k.dump_raw(ty.n), True))
        except ic.Unsupported as e: agree['unsupported'] += 1; continue
        if ok == want: agree['accepted' if ok else 'refused'] += 1
        else: expect(False, 'G %s: the kernel and the independent checker disagree: the kernel says %s, the independent checker %s (%s)' % (label, want, ok, str(msg)[:120]))
    expect(agree['accepted'] >= 20 and agree['refused'] >= 30, 'G the cross-check is not vacuous: %s' % agree)
    return agree


def main():
    t0 = time.time()
    w = build(14)
    jobs = []
    COSTS[0] = True
    part_A(w, jobs); part_B(w, jobs); part_C(w, jobs); part_D(w, jobs); part_E(w, jobs)
    n_big = CHECKS[0]
    # the same terms in a world where `big` computes to 16 successors, for the independent checker (a plain tree rewriter: the 16384 successors of the first world take it hours)
    w2 = build(4); jobs2 = []
    COSTS[0] = False
    part_A(w2, jobs2); part_B(w2, jobs2); part_C(w2, jobs2); part_D(w2, jobs2); part_E(w2, jobs2)
    t1 = time.time(); agree = part_G(w2, jobs2)
    if TRACE: print('G independent checker: %s in %.1f s' % (agree, time.time() - t1))
    print('S35: %d checks (%d in the big world), %d terms cross-checked, %.1f s' % (CHECKS[0], n_big, len(jobs2), time.time() - t0))
    for f in FAIL[:30]: print('  FAIL:', f)
    print('S35 MATCH-CHECK GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    if FAIL: sys.exit(1)


if __name__ == '__main__':
    main()
