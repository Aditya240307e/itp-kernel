"""S27 gate: the work budget.

The stack guard bounds how DEEP a kernel computation goes; the work budget bounds how LONG it runs. Every guarded function of the kernel (eval, quote, instantiate, conversion, unification, type inference,
...) spends one step per call (ITP_STACK_CHECK), a call into the kernel starts with `Kernel.set_fuel(n)` steps, and a call that spends them raises `FuelExhausted`. Conversion of two stuck values that share
structure is exponential in the sharing, so a tactic that is merely WRONG (an `apply` with an argument in the wrong position) used to run for minutes inside a proof search; now it is a failed action.

The budget is trusted code only in the sense that it must never turn a failure into an acceptance or leave a half-done effect behind, so this gate pins down:

  A. ACCOUNTING: the steps a call spends are deterministic (the same call twice spends the same), `set_fuel(n)` allows exactly n steps (a call that needs n passes, one that needs n+1 fails), the budget
     is per call (two successive calls each get all of it), 0 or less means unlimited, `FuelExhausted` is an `ITPError`, and the counters (`fuel_used`, `fuel_peak`) agree with it;
  B. A PATHOLOGICAL INPUT (two chains of doubling definitions whose conversion walks 3^n paths) fails within the budget and in a fraction of a second, and the kernel is fully usable afterwards;
  C. FAIL CLOSED: a declaration checked under every budget from 1 step upward is either rejected with `FuelExhausted` and leaves NO trace (no new global), or accepted with the same type and value as
     with an unlimited budget; an ill-typed declaration is never accepted at any budget; a recursive function whose check is interrupted stays unfinished and can be finished afterwards;
  D. TACTICS ARE ALL OR NOTHING: each kernel tactic (intro, apply, exact, cases, induct, rewrite, change) is interrupted at EVERY step it takes; after each interruption the proof state is exactly as it
     was (the goals, their hypotheses and targets, which metavariables are solved and to what, how many exist, the deferred constraints), and the proof can then be completed and is accepted by the kernel.
"""
import os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(100000)
import itp_env_lnl as E
import nat_lib
from author import AuthorError
from tactics import Proof

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


def raises_fuel(fn):
    try: fn()
    except E.FuelExhausted: return True
    return False


# ---------------------------------------------------------------------------------------------------------------------------------- A, B
def doubling_chains(w, n):
    """dbl^n x and (fun t => t + t)^n x written out: equal after unfolding, and their stuck values share structure, so comparing them walks 3^n paths"""
    Nat, add = w['Nat'], w['add']
    if 'dbl' not in w.gid: w.define('dbl', w.arrow(Nat, Nat), w.fun([Nat], lambda t: add(t, t)))
    dbl = w['dbl']
    x = w._bind(Nat, 'x')[1]
    try:
        a = b = x
        for _ in range(n): a = dbl(a); b = add(b, b)
    finally: w._unbind(1)
    return a, b


def conv1(w, a, b):
    """are two terms with one free variable (context level 0) convertible? One call into the kernel."""
    return w.k.conv_open(w._n(a), w._n(b), 1)


def part_AB(w):
    k = w.k
    k.set_fuel(0)
    a, b = doubling_chains(w, 12)
    for _ in range(2): expect(conv1(w, a, b) is True, 'A the chains of 12 doublings are convertible')          # warm: the first call fills the caches of the definitions it unfolds
    costs = []
    for _ in range(3):
        conv1(w, a, b); costs.append(k.fuel_used())
    expect(len(set(costs)) == 1, 'A the steps a call spends are deterministic: %s' % costs)
    used = costs[0]
    expect(used > 10000, 'A a conversion that walks 3^12 paths spends many steps (%d)' % used)
    expect(k.fuel_peak() >= used, 'A fuel_peak covers the biggest call so far (%d < %d)' % (k.fuel_peak(), used))
    k.set_fuel(used)
    expect(conv1(w, a, b) is True, 'A a budget of exactly the steps needed is enough')
    expect(k.fuel_used() == used, 'A ... and spends exactly that: %d' % k.fuel_used())
    expect(conv1(w, a, b) is True and conv1(w, a, b) is True, 'A the budget is per call: every call gets all of it')
    k.set_fuel(used - 1)
    expect(raises_fuel(lambda: conv1(w, a, b)), 'A one step short is a FuelExhausted')
    expect(k.fuel_used() > used - 1, 'A ... after spending the whole budget (%d)' % k.fuel_used())
    k.set_fuel(used)
    expect(conv1(w, a, b) is True, 'A the call after an exhausted one gets a fresh budget')
    for limit in (0, -1, -10 ** 12):
        k.set_fuel(limit)
        expect(conv1(w, a, b) is True and k.get_fuel() == limit, 'A a budget of %d means unlimited' % limit)
    try: k.set_fuel(3); conv1(w, a, b)
    except E.FuelExhausted as e:
        expect(isinstance(e, E.ITPError) and isinstance(e, Exception) and 'work budget' in str(e), 'A FuelExhausted is an ITPError with a message that names the budget: %s' % (type(e).__mro__[:3],))
    else: expect(False, 'A three steps cannot convert the chains')
    k.set_fuel(0); k.reset_fuel_peak()
    expect(k.fuel_peak() == k.fuel_used(), 'A reset_fuel_peak starts counting again: only the current call is left (%d vs %d)' % (k.fuel_peak(), k.fuel_used()))
    conv1(w, a, b)
    expect(k.fuel_peak() == used, 'A ... and the peak is then the one call made: %d vs %d' % (k.fuel_peak(), used))
    a2, b2 = doubling_chains(w, 3)
    conv1(w, a2, b2); small = k.fuel_used()
    expect(small < used, 'A a small call spends less than a big one (%d, %d)' % (small, used))
    expect(k.fuel_peak() == used, 'A the peak remembers the biggest call after smaller ones: %d vs %d' % (k.fuel_peak(), used))

    # B: 3^24 paths, a budget of 10^7 steps
    a24, b24 = doubling_chains(w, 24)
    k.set_fuel(10 ** 7); t0 = time.time()
    failed = raises_fuel(lambda: conv1(w, a24, b24)); dt = time.time() - t0
    expect(failed, 'B the conversion of 3^24 paths is cut off by a budget of 10^7 steps')
    expect(dt < 10, 'B ... and quickly (%.2f s)' % dt)
    expect(k.fuel_used() > 10 ** 7, 'B ... having spent the budget (%d)' % k.fuel_used())
    expect(conv1(w, a, b) is True, 'B the kernel is usable after the cut-off')
    k.set_fuel(0)
    return used


# ---------------------------------------------------------------------------------------------------------------------------------- C
def part_C(w):
    k = w.k; Nat = w['Nat']; S, Z0 = w['Succ'], w['Zero']
    k.set_fuel(0)
    num = Z0
    for _ in range(40): num = S(num)
    ty, val = w._n(Nat), w._n(num)
    n0 = k.num_globals()
    gid = k.declare_checked(ty, val, 'fuel_num_ref'); cost = k.fuel_used()
    info_ref = k.global_info(gid)
    expect(cost > 20, 'C declaring a numeral of 40 successors spends steps (%d)' % cost)
    # The steps a check spends are a function of the input AND of what earlier checks left in the typing memo (a failed check keeps the types it had found, which are facts), so a later attempt can be cheaper
    # than the first one. What must hold for every budget: the attempt either fails with FuelExhausted and leaves no trace, or succeeds with the same declaration; and the outcomes are monotone.
    outcomes = []
    for limit in range(1, cost + 3):
        k.set_fuel(limit); n_before = k.num_globals()
        try:
            g2 = k.declare_checked(ty, val, 'fuel_num_%d' % limit)
        except E.FuelExhausted:
            outcomes.append(False)
            expect(k.num_globals() == n_before, 'C an interrupted declaration leaves no global behind (budget %d)' % limit)
            continue
        outcomes.append(True)
        expect(limit >= 1 and k.num_globals() == n_before + 1, 'C an accepted declaration adds exactly one global')
        i2 = k.global_info(g2)
        expect(i2[1] == info_ref[1] and i2[2] == info_ref[2], 'C the declaration accepted under budget %d has the same type and value' % limit)
    first = outcomes.index(True) if True in outcomes else None
    expect(first is not None and first + 1 <= cost, 'C some budget up to the cold cost accepts: %s' % first)
    expect(first is not None and all(outcomes[first:]) and not any(outcomes[:first]), 'C the outcomes are monotone in the budget: %s' % ''.join('S' if o else 'F' for o in outcomes))
    expect(first is not None and first >= 5, 'C a handful of steps is not enough to check 40 successors (accepted at %s)' % first)

    # an ill-typed declaration is never accepted, whatever the budget: Nat is not the type of a numeral applied to a numeral
    bad_val = w._n(num(num)) if False else None
    bad_ty = w._n(w['Nat'])                                                                       # the value is a type, not a number: `Nat : Nat` is ill-typed
    n_before = k.num_globals(); outcomes = set()
    for limit in list(range(1, 60)) + [10 ** 3, 10 ** 5, 0]:
        k.set_fuel(limit)
        try: k.declare_checked(bad_ty, w._n(w['Nat']), 'fuel_bad_%d' % limit); outcomes.add('ACCEPTED')
        except E.FuelExhausted: outcomes.add('fuel')
        except E.ITPError: outcomes.add('rejected')
    expect('ACCEPTED' not in outcomes, 'C an ill-typed declaration was accepted under some budget: %s' % outcomes)
    expect(outcomes == {'fuel', 'rejected'}, 'C a small budget runs out and a large one rejects the ill-typed declaration: %s' % outcomes)
    expect(k.num_globals() == n_before, 'C nothing ill-typed was declared')

    # a recursive function whose check is interrupted stays unfinished, and can be finished afterwards
    k.set_fuel(0)
    add = w['add']; add_ty = w._n(w.forall_([Nat, Nat], lambda n, m: Nat))
    def body_of(f): return w.fun([Nat, Nat], lambda n, m: w.match(n, lambda _: Nat, [m, lambda n1: S(f(n1, m))]))
    gid2 = k.reserve_recursive_function('fuel_add', add_ty); w._register('fuel_add', gid2)
    body = w.term(body_of(w.const('fuel_add')), 'body').n
    nat_gid = w.gid['Nat']
    k.set_fuel(3)
    expect(raises_fuel(lambda: k.finish_recursive_function_checked(gid2, nat_gid, 0, body)), 'C finishing a recursive function under 3 steps is interrupted')
    k.set_fuel(0)
    expect(k.recursive_info(gid2) is None or not k.recursive_info(gid2), 'C ... and leaves it unfinished: %s' % (k.recursive_info(gid2),))
    try: k.finish_recursive_function_checked(gid2, nat_gid, 0, body); fin = True
    except Exception as e: fin = False; FAIL.append('C the interrupted function could not be finished afterwards: %s' % e)
    expect(fin and w.conv(w['fuel_add'](num, Z0), num), 'C ... and once finished it computes')


# ---------------------------------------------------------------------------------------------------------------------------------- D
def canon(p):
    """the printed goals with the metavariable numbers taken out (they are different in every run)"""
    return re.sub(r'\?m\d+', '?m', str(p))


def snapshot(p):
    k, ps = p.k, p.ps
    goals = tuple((ps.goal_meta_id(i), ps.goal_target(i), tuple((ps.goal_hyp_name(i, h), ps.goal_hyp_type(i, h)) for h in range(ps.goal_num_hyps(i)))) for i in range(ps.num_goals()))
    probe = k.fresh_meta(k.mk_type(0), 0)                      # the id a new metavariable would get = how many exist (the probe itself stays: callers compare counts accordingly)
    metas = tuple((k.meta_is_solved(m), k.meta_solution_raw(m) if k.meta_is_solved(m) else None) for m in range(probe))
    return {'goals': goals, 'n_metas': probe, 'metas': metas, 'pending': ps.num_pending()}


def sweep(w, name, make, op):
    """run `op(p)` on a fresh state `make()` under every budget from 1 step up to what it needed the first time: each budget either fails with FuelExhausted and leaves the state exactly as it was, or succeeds
    with the very result of an unlimited run; the outcomes are monotone (a larger budget never fails where a smaller one succeeded)"""
    k = w.k
    k.set_fuel(0)
    p = make(); op(p); need = k.fuel_used(); ref_text = canon(p)
    expect(need >= 1, 'D %s spends steps' % name)
    outcomes = []; bad = 0
    for limit in range(1, need + 2):
        p = make(); before = snapshot(p)
        k.set_fuel(limit)
        try: op(p); err = None
        except E.FuelExhausted as e: err = e
        k.set_fuel(0)
        if err is None:
            outcomes.append(True)
            expect(canon(p) == ref_text, 'D %s: the result under budget %d differs from the unlimited one' % (name, limit))
            continue
        outcomes.append(False)
        after = snapshot(p)
        if not (after['goals'] == before['goals'] and after['n_metas'] == before['n_metas'] + 1 and after['metas'][:before['n_metas'] + 1] == before['metas'] + ((False, None),) and after['pending'] == before['pending']):
            bad += 1
            if bad <= 3:
                FAIL.append('D %s interrupted at step %d left a trace: goals %s, metas %d -> %d, solved before %s after %s, pending %d -> %d' % (
                    name, limit, after['goals'] == before['goals'], before['n_metas'], after['n_metas'] - 1,
                    [i for i, m in enumerate(before['metas']) if m[0]], [i for i, m in enumerate(after['metas']) if m[0]], before['pending'], after['pending']))
    expect(bad == 0, 'D %s: %d of the interruptions left a trace' % (name, bad))
    first = outcomes.index(True) if True in outcomes else None
    expect(first is not None and all(outcomes[first:]) and not any(outcomes[:first]), 'D %s: the outcomes are monotone in the budget: %s' % (name, ''.join('S' if o else 'F' for o in outcomes)))
    expect(first is not None and first >= 1, 'D %s: one step is not enough' % name)
    return need


def part_D(w):
    k = w.k; Nat = w['Nat']; Eq = w.Eq; S = w['Succ']; Z0 = w['Zero']
    k.set_fuel(0)

    def trans_state():
        stmt = w.forall_([Nat, Nat, Nat, lambda a, b, c: Eq(a, b), lambda a, b, c, h1: Eq(b, c)], lambda a, b, c, h1, h2: Eq(a, c))
        p = Proof(w, stmt, 'sweep'); p.intro('a', 'b', 'c', 'h1', 'h2'); p.apply('trans', Nat, 'a')          # goals: ?y, a = ?y, ?y = c
        return p
    p = trans_state()
    expect(p.n_goals == 3, 'D setup: apply trans leaves three goals (%d)' % p.n_goals)

    def apply_h1_to_second_goal(p):      # solves the metavariable of the FIRST goal by unification, while working on the second
        with p.focus(1): h1 = p.h('h1', 1)
        p.ps.apply(h1.n, p.ps.goal_meta_id(1), True)
    need = sweep(w, 'apply (solving another goal\'s metavariable)', trans_state, apply_h1_to_second_goal)
    expect(need > 50, 'D that apply spends a few hundred steps (%d)' % need)
    p = trans_state(); apply_h1_to_second_goal(p)
    expect(p.ps.num_goals() == 2 and k.meta_is_solved(p.ps.goal_meta_id(0)), 'D setup: that apply closes the second goal and solves the first goal\'s metavariable by unification')

    def finish_trans(p):
        p._sync_names(); p.exact('b'); p.exact('h2')
        return p

    def exact_to_second_goal(p):
        with p.focus(1): h1 = p.h('h1', 1)
        p.ps.exact(h1.n, p.ps.goal_meta_id(1))
    sweep(w, 'exact (unifying the first goal\'s metavariable)', trans_state, exact_to_second_goal)

    def intro_state():
        return Proof(w, w.forall_([Nat, Nat, lambda a, b: Eq(a, b)], lambda a, b, h: Eq(b, a)), 'sweep_intro')
    sweep(w, 'intro', intro_state, lambda p: p.ps.intro('a', None, True))

    def cases_state():
        p = Proof(w, w.forall_([Nat, Nat, lambda a, b: w['Le'](a, b)], lambda a, b, h: w['Le'](a, S(b))), 'sweep_cases'); p.intro('a', 'b', 'h'); return p
    sweep(w, 'cases (on an indexed family)', cases_state, lambda p: p.ps.cases(2, None, [], True))

    def induct_state():
        p = Proof(w, w.forall_([Nat], lambda n: Eq(w['add'](n, Z0), n)), 'sweep_induct'); p.intro('n'); return p
    sweep(w, 'induct', induct_state, lambda p: p.ps.induct(0, None, [], None, None, True))

    def rewrite_state():
        p = Proof(w, w.forall_([Nat, Nat, Nat, lambda a, b, c: Eq(a, b)], lambda a, b, c, h1: Eq(S(a), S(c))), 'sweep_rw'); p.intro('a', 'b', 'c', 'h1'); return p
    def rewrite_op(p):
        with p.focus(): h1, a, b = p.h('h1'), p.h('a'), p.h('b')
        p.ps.rewrite(h1.n, a.n, b.n, w.gid['transport'], w.gid['symm'])
    sweep(w, 'rewrite', rewrite_state, rewrite_op)

    def change_op(p):
        with p.focus(): target = Eq(S(p.h('a')), S(p.h('c')))
        p.ps.change(w._n(target))
    sweep(w, 'change', rewrite_state, change_op)

    # the proof can be carried on and is accepted after all those interruptions: interrupt the apply at a few budgets, then do it for real
    p = trans_state(); before = snapshot(p)
    for limit in (1, 7, 40, need - 1):
        k.set_fuel(limit)
        try: apply_h1_to_second_goal(p); FAIL.append('D the apply succeeded under a budget of %d although it needs %d' % (limit, need))
        except E.FuelExhausted: pass
        k.set_fuel(0)
    after = snapshot(p)
    expect(after['goals'] == before['goals'] and after['metas'][:before['n_metas'] + 1] == before['metas'] + ((False, None),), 'D four interruptions in a row leave the state as it was')
    apply_h1_to_second_goal(p); finish_trans(p)
    expect(p.done, 'D the interrupted proof is finished by the real steps')
    term = p.term()
    gid = k.declare_checked(p.stmt.n, term.n, 'fuel_trans_thm')
    expect(gid is not None, 'D ... and the kernel accepts it')


def main():
    t0 = time.time()
    w = nat_lib.build_nat()
    print('  nat library built in %.2f s' % (time.time() - t0), flush=True)
    for name, fn in (('A,B', lambda: part_AB(w)), ('C', lambda: part_C(w)), ('D', lambda: part_D(w))):
        t1 = time.time()
        try: fn()
        except Exception as ex:
            import traceback; traceback.print_exc()
            expect(False, '%s raised %s: %s' % (name, type(ex).__name__, str(ex)[:300]))
        print('  part %s done in %.1fs (%d checks so far)' % (name, time.time() - t1, CHECKS[0]), flush=True)
    print('%d checks in %.1fs' % (CHECKS[0], time.time() - t0))
    if FAIL:
        print('S27 FUEL GATE: FAIL (%d)' % len(FAIL))
        for f in FAIL[:30]: print('   ', f)
        sys.exit(1)
    print('S27 FUEL GATE: PASS')


if __name__ == '__main__':
    main()
