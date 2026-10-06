"""F6 gate: induction with a Pi-headed goal, and with an explicit `generalizing` list. The classic case: add (S n) acc = add n (S acc) needs the IH for ALL acc."""
import sys, functools, time
import harness as hn
from harness import *
import s3_tests as s3
from s3_tests import V, app, Lockstep, library

def statement(g):
    NAT, EQ, ADD, S = ('const', g['Nat']), ('const', g['Eq']), ('const', g['add']), ('const', g['Succ'])
    body = app(EQ, NAT, app(ADD, app(S, V(0)), V(1)), app(ADD, V(0), app(S, V(1))))      # Eq (add (S n) acc) (add n (S acc))
    return NAT, ('pi', NAT, ('pi', NAT, body))

def run_pi_headed(kref, knew, g):
    NAT, stmt = statement(g); refl, cong = ('const', g['refl']), ('const', g['cong'])
    ops = [('intro',), ('induct', 0), ('intro',), ('apply', refl), ('intro',), ('apply', cong), ('apply', app(V(2), V(3)))]   # step: IH (hyp 2) applied to acc (hyp 3)
    L = Lockstep(kref, knew, g, stmt); first_div = None
    for i, op in enumerate(ops):
        ok, same, st = L.step(op)
        if first_div is None and not same: first_div = (i, op[0], 'ref ok=%s new ok=%s' % (L.log[-1][1], L.log[-1][2]), L.last_msgs[0][:70])
    ver = knew.check_open(L.psn.extract(), build_new(knew, l2h(stmt, 0)), [])[0] if L.psn.is_complete() else False
    return L.psn.is_complete(), ver, first_div

def run_generalizing(knew, g):
    NAT, stmt = statement(g); refl, cong = ('const', g['refl']), ('const', g['cong'])
    st = build_new(knew, l2h(stmt, 0)); ps = knew.start_proof(st); b = lambda t, n: build_new(knew, l2h(t, n, lenient=True))
    ps.intro('n'); ps.intro('acc')                                  # ctx [n, acc]; the goal is NOT Pi-headed
    results = {}
    try: ps.induct(0); ps.apply(b(refl, 2))                          # without generalizing the IH is for this one acc: the step cannot be closed
    except Exception: pass
    ps = knew.start_proof(st); ps.intro('n'); ps.intro('acc')
    ps.induct(0, generalizing=[1])                                   # revert acc: the IH becomes  forall acc'
    ps.intro if False else None
    ps.apply(b(refl, ps.goal_num_hyps(0)))                           # base:  add 1 acc' = add 0 (S acc')
    nh = ps.goal_num_hyps(0)                                         # step: ctx [n, acc, x, IH, acc']
    ps.apply(b(cong, nh)); ps.apply(b(app(V(3), V(4)), nh))
    ver = knew.check_open(ps.extract(), st, [])[0] if ps.is_complete() else False
    return ps.is_complete(), ver, nh

def run_cases_generalizing(knew, g):
    """cases already reverts every hypothesis AFTER the scrutinee; `generalizing` is for those BEFORE it. ctx [acc, n], cases n: with generalizing=[0] acc is reverted too"""
    NAT, EQ, Z = ('const', g['Nat']), ('const', g['Eq']), ('const', g['Zero'])
    stmt = ('pi', NAT, ('pi', NAT, app(EQ, NAT, V(0), V(0)))); counts = {}
    for label, gen in (('plain', []), ('generalizing acc', [0])):
        ps = knew.start_proof(build_new(knew, l2h(stmt, 0))); ps.intro('acc'); ps.intro('n'); ps.cases(1, generalizing=gen)
        counts[label] = [ps.goal_num_hyps(i) for i in range(ps.num_goals())]
    return counts

if __name__ == '__main__':
    kref, knew, g, defs = library(); FAIL = []
    c, v, fd = run_pi_headed(kref, knew, g)
    print("Pi-headed induction (intro n; induct n with goal  forall acc. ...): hybrid complete %s, verifies %s | first lockstep divergence: %s" % (c, v, fd))
    if not (c and v): FAIL.append('pi-headed induction')
    c2, v2, nh = run_generalizing(knew, g)
    print("induct n generalizing acc (acc is independent of n): hybrid complete %s, verifies %s (step goal has %d hypotheses)" % (c2, v2, nh))
    if not (c2 and v2): FAIL.append('generalizing')
    cc = run_cases_generalizing(knew, g)
    print("cases n (ctx [acc, n]) hypothesis counts per branch: %s" % cc)
    if cc['plain'] != [2, 3] or cc['generalizing acc'] != [3, 4]: FAIL.append('cases generalizing')
    print("S6 F6 GATE:", "PASS" if not FAIL else "FAIL -- " + "; ".join(FAIL)); sys.exit(1 if FAIL else 0)
