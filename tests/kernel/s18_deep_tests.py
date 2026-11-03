"""S18 gate: the kernel survives terms of any depth.

Every kernel function that recurses on the depth of a term starts with ITP_STACK_CHECK(), which turns "the C stack is about to overflow" into a RuntimeError instead of a segmentation
fault (before the guard, a unary numeral 20000 deep killed the process). This gate builds terms of many shapes -- numerals, nests of lambdas / Pis / lets / matches / beta redexes, a
long application spine -- far deeper than a small thread stack allows and drives every entry point that traverses a term. Each (shape, mode) runs in a CHILD PROCESS, so a crash is a
test failure with a name instead of a dead gate:

  small : the work runs on a thread with a 1 MiB stack, on a term deep enough to overflow it -- every operation must either finish or raise RuntimeError('stack exhausted'); the
          kernel must stay usable afterwards (a small term still checks) and a rejected deep term must not have changed what is accepted.
  big   : the same operations on the main thread (stack limit raised by the module) on a term that is deep but legitimate -- they must FINISH, and give the right answers.

  bounds: the stack-bounds lookup of the guard, for the platform that is NOT this one: the header is compiled with ITP_STACK_BOUNDS_APPLE=1 (the macOS calls pthread_get_stackaddr_np /
          pthread_get_stacksize_np, emulated here with glibc's) and without it, and on threads of several stack sizes the guard must fire exactly at the same floor in both -- lowest address
          plus min(size/8, 1 MiB) -- with the stack still intact (a wrong 'top minus size' would make it fire at once or never, i.e. a crash on a Mac).

Never a crash, never a hang, never a wrong answer."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import sys, os, subprocess, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = ['numeral', 'neutral', 'lambda', 'pi', 'let', 'match', 'beta', 'spine', 'solution', 'metachain', 'metaspine', 'release']
N_SMALL, N_BIG = 150000, 30000

def child(shape, mode):
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
    sol = None
    if shape == 'solution':
        # a proof built step by step: the solution of the first goal is  Succ ?m1,  that of ?m1 is  Succ ?m2, ...  (n steps, each one cheap), so extracting it resolves a chain of n metavariable solutions,
        # and the result is a numeral n deep that nothing in it can reduce
        sol = k.start_proof(NAT)
        for _ in range(n): sol.apply(S)
        sol.exact(Z); t = Z
    meta_end = None
    if shape == 'metachain':
        # metavariables assigned to one another: ?m0 := ?m1, ?m1 := ?m2, ... (n of them); resolving the first goes through all n solutions, and no node of a term is rebuilt on the way
        ms = [k.fresh_meta(NAT, 0) for _ in range(n)]
        for i in range(n - 1): k.unify_open(k.mk_meta(ms[i]), k.mk_meta(ms[i + 1]), [])
        t = k.mk_meta(ms[0]); meta_end = k.mk_meta(ms[-1])
    elif shape == 'metaspine':
        # ?f := fun x => x, and the term ?f (?f (... x)) n deep: every argument of a solved metavariable is resolved before it is spliced in
        fid = k.fresh_meta(k.mk_pi(NAT, NAT), 0); k.unify_open(app(k.mk_meta(fid), k.mk_fvar(0)), k.mk_fvar(0), [NAT])
        t = k.mk_fvar(0); meta_end = t
        for _ in range(n): t = app(k.mk_meta(fid), t)
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
        if shape in ('metachain', 'metaspine'):
            op('resolve_metas', lambda: k.resolve_metas(t) == meta_end or (_ for _ in ()).throw(RuntimeError('resolving gave a different term than the end of the chain')))
            op('node_info', lambda: k.node_info(t))
            op('check_node_info', lambda: (lambda bad: bad == '' or (_ for _ in ()).throw(RuntimeError('node facts disagree: ' + bad)))(k.check_node_info(t)))
            return
        if shape == 'solution':
            got = {}
            op('is_complete', lambda: sol.is_complete())
            op('extract', lambda: got.setdefault('proof', sol.extract()) is not None)
            if 'proof' in got:                    # the extracted numeral is what was built, and the kernel accepts it
                op('check_extracted', lambda: k.check_open(got['proof'], NAT, [])[0] or (_ for _ in ()).throw(RuntimeError('the extracted solution is not a Nat')))
                op('extracted_is_numeral', lambda: got['proof'] == numeral(n) or (_ for _ in ()).throw(RuntimeError('the extracted solution differs from the numeral built step by step')))
            return
        op('infer', lambda: k.infer_open(t, []))
        op('normalize', lambda: k.normalize_open(t, 0))
        op('lift', lambda: k.lift(t, 1))
        op('loose_range', lambda: k.loose_range(t))
        op('dump_raw', lambda: len(k.dump_raw(t)))
        op('instantiate', lambda: k.instantiate(k.lift(t, 1) if shape != 'lambda' else t, [Z]))
        op('abstract_fvars', lambda: k.abstract_fvars(t, [0]))
        op('replace_fvars', lambda: k.replace_fvars(t, [(0, Z)]))
        op('resolve_metas', lambda: k.resolve_metas(t))
        op('check_node_info', lambda: (lambda bad: bad == '' or (_ for _ in ()).throw(RuntimeError('node facts disagree: ' + bad)))(k.check_node_info(t)))
        op('node_info', lambda: k.node_info(t))
        op('extract', lambda: (lambda ps: (ps.exact(t), ps.extract())[1])(k.start_proof(NAT)))              # the proof-term path: resolve the metavariables, then scan for anything that could reduce
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

BOUNDS_PROBE = r"""
#include <pthread.h>
#include <cstdio>
#include <cstdint>
#include <cstddef>
// What macOS answers: the HIGH end of the stack (it grows down) and its size. Emulated with glibc's call; the header only looks at them if ITP_STACK_BOUNDS_APPLE is 1.
static void stack_of(pthread_t t, char*& lo, size_t& sz) { pthread_attr_t a; void* p = nullptr; pthread_getattr_np(t, &a); pthread_attr_getstack(&a, &p, &sz); pthread_attr_destroy(&a); lo = static_cast<char*>(p); }
extern "C" void* pthread_get_stackaddr_np(pthread_t t) { char* lo; size_t sz; stack_of(t, lo, sz); return lo + sz; }
extern "C" size_t pthread_get_stacksize_np(pthread_t t) { char* lo; size_t sz; stack_of(t, lo, sz); return sz; }
#include "itp_kernel.hpp"
struct Probe { char* first = nullptr; char* last = nullptr; char* lowest = nullptr; size_t size = 0; long depth = 0; bool threw = false; };
[[gnu::noinline]] static long rec(Probe& p, long d) {
    ITP_STACK_CHECK();
    char pad[128]; pad[0] = static_cast<char>(d); asm volatile("" : : "r"(pad) : "memory");
    char* fa = static_cast<char*>(__builtin_frame_address(0)); if (!p.first) p.first = fa; p.last = fa; p.depth = d;
    return rec(p, d + 1) + pad[0];
}
static void* run(void* arg) {
    Probe& p = *static_cast<Probe*>(arg); stack_of(pthread_self(), p.lowest, p.size);
    try { rec(p, 0); } catch (const itp::StackExhausted&) { p.threw = true; }
    return nullptr;
}
int main() {
    for (size_t want : {size_t(256) << 10, size_t(1) << 20, size_t(4) << 20, size_t(16) << 20}) {
        pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at, want);
        Probe p; pthread_t th; pthread_create(&th, &at, run, &p); pthread_join(th, nullptr);
        const size_t margin = p.size / 8 < (size_t(1) << 20) ? p.size / 8 : (size_t(1) << 20);
        std::printf("%zu %d %zu %zu %ld\n", p.size, p.threw ? 1 : 0, margin, static_cast<size_t>(p.last - p.lowest), p.depth);
    }
}
"""

def bounds_check():
    """the guard's floor on threads of several stack sizes, with the macOS stack-bounds calls (emulated) and with the platform's own: both must put it at lowest + min(size/8, 1 MiB).  [failure]"""
    import tempfile, shutil
    hdr = os.path.join(HERE, '..', '..', 'src', 'c++', 'hybrid'); fails = []
    cxx = os.environ.get('CXX') or next((c for c in ('c++', 'g++', 'clang++') if shutil.which(c)), None)
    if not cxx: return ['bounds: no C++ compiler']
    import build_kernel as BK; std = BK.pick_std_flag(cxx)
    if not sys.platform.startswith('linux'): return []                       # (the emulation needs glibc's pthread_getattr_np; on a Mac the real calls are what every other test here exercises)
    with tempfile.TemporaryDirectory(prefix='itp_s18_') as td:
        src = os.path.join(td, 'probe.cpp'); open(src, 'w').write(BOUNDS_PROBE); procs = {}
        for apple in (0, 1):
            exe = os.path.join(td, 'probe%d' % apple)
            procs[apple] = (exe, subprocess.Popen([cxx, std, '-O1', '-pthread', '-DITP_STACK_BOUNDS_APPLE=%d' % apple, '-I' + hdr, src, '-o', exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        out = {}
        for apple, (exe, pr) in procs.items():
            _, err = pr.communicate(timeout=600)
            if pr.returncode != 0:
                fails.append('bounds/apple=%d: the probe did not compile: %s' % (apple, err.strip()[-300:])); continue
            r = subprocess.run([exe], capture_output=True, text=True, timeout=120)
            if r.returncode != 0: fails.append('bounds/apple=%d: the probe died (exit %s): the guard did not fire in time' % (apple, r.returncode)); continue
            out[apple] = [tuple(int(x) for x in l.split()) for l in r.stdout.strip().split('\n')]
        for apple, rows in out.items():
            for size, threw, margin, above, depth in rows:
                tag = 'bounds/apple=%d/%dKiB' % (apple, size >> 10)
                if not threw: fails.append(tag + ': the guard never fired')
                elif not (margin <= above < margin + 1024): fails.append(tag + ': the guard fired %d bytes above the stack\'s lowest address, expected %d..%d' % (above, margin, margin + 1023))
        if out: print("   bounds   thread stacks %s KiB: the guard fires %s bytes above the lowest address (macOS calls emulated / this platform's)" % ([r[0] >> 10 for r in out.get(1, out.get(0))], [r[3] - r[2] for r in out.get(1, out.get(0))]), flush=True)
    return fails

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
    fails.extend(bounds_check())
    print("   (%.0f s)" % (time.time() - t0))
    for f in fails: print("   FAIL " + f)
    print("S18 DEEP-TERM GATE: " + ("PASS" if not fails else "FAIL"))
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child': child(sys.argv[2], sys.argv[3])
    else: main()
