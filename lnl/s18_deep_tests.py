"""S18 gate: the kernel survives terms of any depth.

Every kernel function that recurses on the depth of a term starts with ITP_STACK_CHECK(), which turns "the C stack is about to overflow" into a RuntimeError instead of a segmentation
fault (before the guard, a unary numeral 20000 deep killed the process). This gate builds terms of many shapes -- numerals, nests of lambdas / Pis / lets / matches / beta redexes, a
long application spine -- far deeper than a small thread stack allows and drives every entry point that traverses a term. Each (shape, mode) runs in a CHILD PROCESS, so a crash is a
test failure with a name instead of a dead gate:

  small : the work runs on a thread with a 1 MiB stack, on a term deep enough to overflow it -- every operation must either finish or raise RuntimeError('stack exhausted'); the
          kernel must stay usable afterwards (a small term still checks) and a rejected deep term must not have changed what is accepted.
  big   : the same operations on the main thread (stack limit raised by the module) on a term that is deep but legitimate -- they must FINISH, and give the right answers.

Never a crash, never a hang, never a wrong answer."""
import sys, os, subprocess, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = ['numeral', 'neutral', 'lambda', 'pi', 'let', 'match', 'beta', 'spine', 'release']
N_SMALL, N_BIG = 150000, 30000

def child(shape, mode):
    sys.path.insert(0, HERE)
    import harness as H
    from harness import build_new, l2h
    import prop_world as PW
    k, g = PW.build_prop_world()
    mk = lambda name: k.mk_const(g[name])
    S, Z, ADD, EQ, NAT, REFL = (mk(x) for x in ('Succ', 'Zero', 'add', 'Eq', 'Nat', 'refl'))
    app = k.mk_app; n = N_SMALL if mode == 'small' else N_BIG
    def numeral(m):
        t = Z
        for _ in range(m): t = app(S, t)
        return t
    ident = k.mk_lambda(NAT, k.mk_bvar(0))
    spine_f = None
    if shape == 'numeral': t = numeral(n)
    elif shape == 'neutral':            # f (f (... Z)) with f an opaque function: a stuck application nested n deep, not a constructor chain
        f_gid = k.declare_unchecked(k.mk_pi(NAT, NAT), None, 'deep_opaque'); F = k.mk_const(f_gid); t = Z
        for _ in range(n): t = app(F, t)
    elif shape == 'lambda':
        t = Z
        for _ in range(n): t = k.mk_lambda(NAT, t)
    elif shape == 'pi':
        t = NAT
        for _ in range(n): t = k.mk_pi(NAT, t)
    elif shape == 'let':
        t = Z
        for _ in range(n): t = k.mk_let(NAT, Z, t)
    elif shape == 'match':
        motive = k.mk_lambda(NAT, NAT); pred = k.mk_lambda(NAT, k.mk_bvar(0)); t = Z
        for _ in range(n): t = k.mk_match(t, motive, [Z, pred])
    elif shape == 'beta':
        t = Z
        for _ in range(n): t = app(ident, t)
    elif shape == 'spine':
        ft = NAT
        for _ in range(n // 100): ft = k.mk_pi(NAT, ft)
        spine_f = k.declare_unchecked(ft, None, 'deep_f'); t = k.mk_const(spine_f)
        for _ in range(n // 100): t = app(t, Z)
    if shape == 'release':
        # values built iteratively (so nothing recurses on the way IN) and dropped: freeing them must not recurse either. A million levels is far beyond any stack.
        t = Z
    results = []
    def op(name, f):
        t_op = time.perf_counter()
        try: r = f(); results.append((name, 'ok', r))
        except RuntimeError as e: results.append((name, 'stack' if 'stack exhausted' in str(e) else 'error', str(e)[:80]))
        except Exception as e: results.append((name, 'error', '%s: %s' % (type(e).__name__, str(e)[:80])))
        print("%s %s" % (name, results[-1][1]), flush=True)
        if os.environ.get('S18_TIMES'): print("   %-16s %8.0f ms" % (name, (time.perf_counter() - t_op) * 1e3), file=sys.stderr, flush=True)
    def run():
        if shape == 'release':
            for kind, what in ((0, 'ctor chain'), (1, 'environment'), (2, 'closure chain')):
                op('release ' + what.replace(' ', '_'), lambda kind=kind: k.stress_value_release(1_000_000, kind))
            return
        op('infer', lambda: k.infer_open(t, []))
        op('normalize', lambda: k.normalize_open(t, 0))
        op('lift', lambda: k.lift(t, 1))
        op('loose_range', lambda: k.loose_range(t))
        op('dump_raw', lambda: len(k.dump_raw(t)))
        op('instantiate', lambda: k.instantiate(k.lift(t, 1) if shape != 'lambda' else t, [Z]))
        op('abstract_fvars', lambda: k.abstract_fvars(t, [0]))
        if shape in ('numeral', 'neutral', 'match', 'beta', 'let'):          # terms of type Nat
            if shape == 'neutral':
                other = app(app(ADD, Z), Z)
                for _ in range(n): other = app(F, other)         # the same stuck chain, reached through a different (unevaluated) argument
            else: other = app(app(ADD, numeral(n - 1)), app(S, Z)) if shape == 'numeral' else Z
            op('conv', lambda: k.conv_open(t, other, 0))
            op('unify', lambda: k.unify_open(t, other, []))
            op('check', lambda: k.check_open(t, NAT, []))
            op('declare_checked', lambda: k.declare_checked(NAT, t, 'deep_' + shape))
            stmt = app(app(app(EQ, NAT), t), other)
            op('check_refl', lambda: k.check_open(app(app(REFL, NAT), t), stmt, []))
            op('tactic_refl', lambda: (lambda ps: (ps.apply(app(REFL, NAT)), ps.is_complete())[1])(k.start_proof(stmt)))
    if mode == 'small':
        threading.stack_size(1 << 20); th = threading.Thread(target=run); th.start(); th.join()
    else:
        run()
    # the kernel must still work, and must not have been changed by the rejected work
    a, b = numeral(3), numeral(5); ok = k.check_open(app(app(REFL, NAT), b), app(app(app(EQ, NAT), app(app(ADD, numeral(2)), a)), b), [])[0]
    print("AFTER " + ("ok" if ok else "BROKEN"), flush=True)
    bad = [r for r in results if r[1] == 'error' and not ('ill-typed' in r[2] or 'mismatch' in r[2].lower() or 'type' in r[2].lower() or 'not a function' in r[2].lower() or 'scope' in r[2].lower() or 'ITPError' in r[2] or 'loose' in r[2].lower() or 'unbound' in r[2].lower())]
    print("DONE %d %d" % (sum(1 for r in results if r[1] == 'ok'), sum(1 for r in results if r[1] == 'stack')), flush=True)

def main():
    fails = []; t0 = time.time()
    for mode in ('small', 'big'):
        for shape in SHAPES:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', shape, mode], capture_output=True, text=True, timeout=300, cwd=HERE)
            lines = r.stdout.strip().split('\n') if r.stdout.strip() else []
            if r.returncode != 0 or not lines or not lines[-1].startswith('DONE'):
                fails.append("%s/%s: the process died (exit %s) after: %s" % (mode, shape, r.returncode, lines[-1] if lines else '<no output>')); continue
            if 'AFTER ok' not in lines: fails.append("%s/%s: the kernel was broken after the deep work" % (mode, shape))
            stack = [l for l in lines if l.endswith(' stack')]; errs = [l for l in lines if l.endswith(' error')]; oks = [l for l in lines if l.endswith(' ok') and not l.startswith('AFTER')]
            if mode == 'big' and stack: fails.append("%s/%s: stack exhausted on a term the raised limit should allow: %s" % (mode, shape, stack))
            if mode == 'small' and not stack and shape != 'release': fails.append("%s/%s: nothing overflowed -- the term is not deep enough to test the guard" % (mode, shape))
            if shape == 'release' and (len(oks) != 3 or stack or errs): fails.append("%s/%s: dropping a deep value chain did not simply succeed: ok %d, stack %d, errors %d" % (mode, shape, len(oks), len(stack), len(errs)))
            print("   %-5s %-8s ok %2d  stack-guard %2d  other-error %2d" % (mode, shape, len(oks), len(stack), len(errs)), flush=True)
            for e in errs:
                if mode == 'big' and shape in ('numeral', 'neutral', 'match', 'beta', 'let') and e.split()[0] in ('infer', 'normalize', 'conv', 'check', 'declare_checked', 'check_refl', 'tactic_refl', 'unify'):
                    fails.append("%s/%s: %s failed with an ordinary error on a well-typed term" % (mode, shape, e))
    print("   (%.0f s)" % (time.time() - t0))
    for f in fails: print("   FAIL " + f)
    print("S18 DEEP-TERM GATE: " + ("PASS" if not fails else "FAIL"))
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child': child(sys.argv[2], sys.argv[3])
    else: main()
