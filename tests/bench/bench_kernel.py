"""bench_kernel.py -- micro-benchmarks for the hybrid kernel's evaluator and type checker (run in isolation: `python3 bench_kernel.py [--json out.json]`).

Every case reports milliseconds for ONE call on a freshly built prop world. The workloads are the ones that decide whether reflection (`ring`) and proof search are usable:
unary numerals (deep constructor spines), structural recursion (`add`, `fib`), conversion of big closed terms, and the typing path (`infer`/`check`) over them.
A case is skipped (reported as null) when its previous size already took more than the budget, so the script always terminates."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, time, json, os
import harness as H
from harness import *
import prop_world as PW
import s2_tests

BUDGET_MS = float(os.environ.get('BENCH_BUDGET_MS', 4000))

def world():
    k, g = PW.build_prop_world()
    mk = lambda name: k.mk_const(g[name])
    S, Z, ADD, EQ, NAT, REFL = (mk(n) for n in ('Succ', 'Zero', 'add', 'Eq', 'Nat', 'refl'))
    # fib : Nat -> Nat   (fib 0 = 0, fib 1 = 1, fib (k+2) = fib k + fib (k+1)); the second call is on the pattern variable m = S k itself
    NATc, Sc, Zc, ADDc = ('const', g['Nat']), ('const', g['Succ']), ('const', g['Zero']), ('const', g['add'])
    fty = ('pi', NATc, NATc); motive = ('lam', NATc, NATc)
    inner = ('match', ('var', 1), motive, (('app', Sc, Zc), ('lam', NATc, ('app', ('app', ADDc, ('app', ('self',), ('var', 2))), ('app', ('self',), ('var', 1))))))
    body = ('lam', NATc, ('match', ('var', 0), motive, (Zc, ('lam', NATc, inner))))
    gid = k.reserve_recursive_function('fib', build_new(k, l2h(fty, 0)))
    k.finish_recursive_function_checked(gid, g['Nat'], 0, build_new(k, l2h(s2_tests.subst_self(body, gid), 0, lenient=True)))
    FIB = k.mk_const(gid)
    def num(n):
        t = Z
        for _ in range(n): t = k.mk_app(S, t)
        return t
    return k, g, dict(S=S, Z=Z, ADD=ADD, EQ=EQ, NAT=NAT, REFL=REFL, FIB=FIB, num=num)

def timed(f):
    t = time.perf_counter(); r = f(); return (time.perf_counter() - t) * 1e3, r

def series(name, sizes, make, results):
    """make(n) -> thunk; stops growing once a call exceeds the budget"""
    out = {}
    for n in sizes:
        try:
            thunk = make(n)
            ms, r = timed(thunk)
        except Exception as e:
            out[n] = None; print("  %-28s n=%-6d FAILED %s" % (name, n, str(e)[:80]), flush=True); break
        out[n] = ms; print("  %-28s n=%-6d %10.2f ms" % (name, n, ms), flush=True)
        if ms > BUDGET_MS: break
    results[name] = out

def main():
    k, g, w = world(); S, Z, ADD, EQ, NAT, REFL, FIB, num = (w[x] for x in ('S', 'Z', 'ADD', 'EQ', 'NAT', 'REFL', 'FIB', 'num'))
    app = k.mk_app; results = {}
    sizes = [50, 100, 200, 400, 800, 1600, 3200]
    series('infer numeral', sizes, lambda n: (lambda a=num(n): k.infer_open(a, [])), results)
    series('normalize add n n', sizes, lambda n: (lambda a=num(n): k.normalize_open(app(app(ADD, a), a), 0)), results)
    series('conv add n n == 2n', sizes, lambda n: (lambda a=num(n), b=num(2 * n): k.conv_open(app(app(ADD, a), a), b, 0)), results)
    def refl_check(n):
        a, b = num(n), num(2 * n); stmt = app(app(app(EQ, NAT), app(app(ADD, a), a)), b); prf = app(app(REFL, NAT), b)
        return lambda: k.check_open(prf, stmt, [])
    series('check refl: add n n = 2n', sizes, refl_check, results)
    fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
    series('conv fib k == fib_k', [8, 12, 14, 16, 17, 18, 19], lambda n: (lambda: k.conv_open(app(FIB, num(n)), num(fibs[n]), 0)), results)
    # typing path: statement with a bigger term, checked by `apply refl` in a proof state
    def tactic_refl(n):
        def go():
            a, b = num(n), num(2 * n); stmt = app(app(app(EQ, NAT), app(app(ADD, a), a)), b)
            ps = k.start_proof(stmt); ps.apply(app(REFL, NAT)); assert ps.is_complete()
        return go
    series('tactic: apply refl', [50, 100, 200, 400, 800], tactic_refl, results)
    # per-call overhead on tiny terms (microseconds per call): this is what a proof-search loop pays thousands of times a second
    def tiny(name, reps, f):
        t = time.perf_counter()
        for _ in range(reps): f()
        us = (time.perf_counter() - t) / reps * 1e6; results[name] = {1: us / 1e3}; print("  %-28s %8.2f us/call" % (name, us), flush=True)
    a3, a2, a5 = num(3), num(2), num(5); sum32 = app(app(ADD, a3), a2); stmt32 = app(app(app(EQ, NAT), sum32), a5); prf5 = app(app(REFL, NAT), a5)
    tiny('tiny normalize add 3 2', 20000, lambda: k.normalize_open(sum32, 0))
    tiny('tiny conv add 3 2 == 5', 20000, lambda: k.conv_open(sum32, a5, 0))
    tiny('tiny infer add 3 2', 20000, lambda: k.infer_open(sum32, []))
    tiny('tiny check refl 3+2=5', 10000, lambda: k.check_open(prf5, stmt32, []))
    def tiny_tactic():
        ps = k.start_proof(stmt32); ps.apply(app(REFL, NAT))
    tiny('tiny tactic apply refl', 5000, tiny_tactic)
    if '--json' in sys.argv:
        json.dump(results, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    return results

if __name__ == '__main__':
    main()
