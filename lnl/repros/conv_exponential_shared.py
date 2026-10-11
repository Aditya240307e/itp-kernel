"""Finding F29: `conv` is exponential in the DEPTH of a shared (DAG) term when the two sides unfold differently.

    python3 repros/conv_exponential_shared.py [fuel]

`dbl t := t + t` applied n times to x, against `t + t` written out n times: the two terms have 2n+1 distinct nodes, yet comparing them costs about 3^n guarded steps (the steps double every
two chain links: n=16 takes 4.4e7 steps, 2 s; n=18 3.9e8 steps, 22 s). Nothing is wrong with the answer; the cost is the problem, and an `apply` that unified a wrongly-instantiated
lemma against a big goal (z16 in the findings) spent minutes there. Since the kernel has a deterministic work budget (F30, gate S27) such a call now fails with FuelExhausted, with the proof
state unchanged, instead of hanging. The root cause (no memo on pairs of values in `conv_value`) is OPEN.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(100000)
import itp_env_lnl as E
import nat_lib
w = nat_lib.build_nat()
w.k.set_fuel(int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 10)
Nat = w['Nat']; add = w['add']
w.define('dbl', w.arrow(Nat, Nat), w.fun([Nat], lambda t: add(t, t)))
dbl = w['dbl']
x = w._bind(Nat, 'x')[1]
def chain(n, op):
    t = x
    for _ in range(n): t = op(t)
    return t
for n in (2, 4, 6, 8, 10, 12, 14, 16, 18):
    A = chain(n, lambda t: dbl(t)); B = chain(n, lambda t: add(t, t))
    t0 = time.time()
    try: r = w.conv(A, B); used = w.k.fuel_used()
    except E.FuelExhausted: r = 'FUEL EXHAUSTED'; used = w.k.fuel_used()
    dt = time.time() - t0
    print('n=%2d conv = %s  %.3f s  steps %d' % (n, r, dt, used), flush=True)
    if dt > 10 or r == 'FUEL EXHAUSTED': break
