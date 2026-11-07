#!/usr/bin/env python3
"""measure.py -- the numbers of the technical report that come from the kernel and the libraries.

    python3 preprint/measure.py --out preprint/measurements/libs.json            # about three minutes
    python3 preprint/measure.py --quick --out /tmp/libs_quick.json               # the Nat library only: a check that the script runs

What it measures (everything is a wall-clock measurement of the code the repository runs; nothing is estimated):

  machine      CPU, cores, memory, operating system, Python, compiler, the commit and whether the tree was clean
  build        the library chain built from scratch in ONE process, timed stage by stage: seconds, globals added, globals so far; peak resident memory afterwards
  census       the globals of the finished world by kind (inductive type, constructor, recursive function, definition, theorem, axiom, quotient constant)
  sharing      the term nodes the world holds, against the size of its proof terms as TREES (the effect of hash-consing)
  kernel       every checked declaration of the finished world submitted to the kernel again, in an empty context: total time and the slowest declarations
  independent  the independent checker over the same world: seconds to load it (export, parse) and to check it, and the number of declarations it rejected
  micro        checkpoint and rewind with and without allocation; the evaluation of binary-numeral products (a 793-bit by 843-bit product is decided by reduction)
  determinism  the digest of the world (names and types of all globals): two runs of this script must print the same one

The numbers depend on the machine; the file records it.  Run it on an otherwise idle machine.
"""
import os, sys, time, json, platform, resource, subprocess, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src', 'python'))
import itp_paths  # noqa: E402,F401
sys.setrecursionlimit(100000)
clock = time.perf_counter


def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception: return ''


def cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.lower().startswith('model name'): return line.split(':', 1)[1].strip()
    except OSError: pass
    return platform.processor() or 'unknown'


def mem_gb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemTotal'): return round(int(line.split()[1]) / 1048576, 2)
    except OSError: pass
    return None


def rss_mb():
    try: return int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1048576
    except OSError: return None


def peak_rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: kilobytes


def machine():
    try: load = [round(x, 2) for x in os.getloadavg()]
    except OSError: load = None
    try: usable = len(os.sched_getaffinity(0))
    except AttributeError: usable = os.cpu_count()
    return {'cpu': cpu_model(), 'cores': os.cpu_count(), 'cores_usable': usable, 'memory_gb': mem_gb(), 'os': platform.platform(), 'python': platform.python_version(),
            'compiler': (sh('g++ --version').splitlines() or [''])[0], 'commit': sh('git -C %s rev-parse HEAD' % ROOT), 'tree_clean_outside_preprint': not [l for l in sh('git -C %s status --porcelain' % ROOT).splitlines() if 'preprint/' not in l],
            'loadavg_start': load}


# ---------------------------------------------------------------------------------------------------------------------------------------------- stage timing
class Stages:
    """wraps the stage functions of the library modules (each takes the world first) so that the build records how long each took and how many globals it added"""
    def __init__(self): self.rows = []; self.depth = 0
    def wrap(self, mod, name, label=None):
        f = getattr(mod, name); rows = self.rows; modname = mod.__name__; me = self
        def g(w, *a, **kw):
            n0 = w.k.num_globals(); t = clock(); d = me.depth; me.depth += 1
            try: r = f(w, *a, **kw)
            finally: me.depth -= 1
            dt = clock() - t; n1 = w.k.num_globals()
            rows.append({'module': modname, 'stage': label or name, 'depth': d, 'seconds': dt, 'added': n1 - n0, 'total': n1, 'rss_mb': rss_mb()})      # (a stage called by another stage is counted in its caller: depth > 0)
            return r
        setattr(mod, name, g)


def instrument(st):
    import lnl_base, nat_lib, bin_lib, quot_lib, int_lib, nt_lib, ext_lib, imo_lib, ring_lib
    for n in ('declare_types', 'declare_add', 'equality_lemmas', 'logic_lemmas'): st.wrap(lnl_base, n)
    for n in ('declare_arith', 'succ_lemmas', 'add_lemmas', 'mul_lemmas', 'pow_lemmas', 'arith_lemmas', 'order_lemmas', 'ring_lemmas'): st.wrap(nat_lib, n)
    for n in ('declare_binary', 'pos_lemmas', 'n_lemmas', 'n_library', 'n_order'): st.wrap(bin_lib, n)
    st.wrap(ring_lib, 'ring_lemma_set', 'ring_lemma_set (N)')
    for n in ('declare_quotients', 'declare_prod', 'quot_lift2'): st.wrap(quot_lib, n)
    for n in int_lib.STAGES: st.wrap(int_lib, n)
    for n in nt_lib.STAGES: st.wrap(nt_lib, n)
    for n in ext_lib.STAGES: st.wrap(ext_lib, n)
    for n in imo_lib.STAGES: st.wrap(imo_lib, n)


GROUPS = [('lnl_base', 'logic and equality (lnl_base)'), ('nat_lib', 'unary naturals (nat_lib)'), ('bin_lib', 'binary naturals (bin_lib)'), ('ring_lib', 'ring lemmas over N (ring_lib)'),
          ('quot_lib', 'quotients (quot_lib)'), ('int_lib', 'integers (int_lib)'), ('nt_lib', 'number theory (nt_lib)'), ('ext_lib', 'library extension (ext_lib)'), ('imo_lib', 'olympiad problems (imo_lib)')]


def grouped(rows):
    out = []
    for mod, label in GROUPS:
        rs = [r for r in rows if r['module'] == mod and r['depth'] == 0]
        if rs: out.append({'group': label, 'module': mod, 'seconds': sum(r['seconds'] for r in rs), 'added': sum(r['added'] for r in rs), 'total_after': rs[-1]['total']})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- census and sharing
def census(w):
    k = w.k; c = collections.Counter(); props = []
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        if k.is_inductive(gid): kind = 'inductive type'
        elif cf >= 0: kind = 'constructor'
        elif k.quot_role(gid): kind = 'quotient constant'
        elif val is None: kind = 'axiom'
        elif rec: kind = 'recursive function'
        else:
            ok, s = k.infer_open(ty, [])
            v = k.view(s) if ok else None
            kind = 'theorem' if (v and v[0] == 'type' and v[1] == 0) else 'definition'
        c[kind] += 1
        if kind == 'axiom': props.append(name)
    return dict(c), props


def reachable_nodes(k, roots):
    """the number of DISTINCT term nodes reachable from `roots` (hash-consed: equal terms are one node)"""
    seen = set(); stack = list(roots)
    while stack:
        n = stack.pop(); key = (n.tag, n.index)
        if key in seen: continue
        seen.add(key)
        for x in k.view(n)[1:]:
            if hasattr(x, 'tag'): stack.append(x)
            elif isinstance(x, (list, tuple)): stack.extend(y for y in x if hasattr(y, 'tag'))
    return len(seen)


def sharing(w):
    """the size of the values of the declarations as TREES (what a system without sharing would store) against the distinct nodes the kernel holds for them (types and values)"""
    k = w.k; tree = 0; n_vals = 0; biggest = (0, ''); roots = []; saturated = 0
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        roots.append(ty)
        if val is None: continue
        roots.append(val)
        size = k.node_info(val)[2]; tree += size; n_vals += 1
        if size >= 0x7FFFFFFF: saturated += 1
        if size > biggest[0]: biggest = (size, name)
    distinct = reachable_nodes(k, roots)
    return {'values': n_vals, 'tree_size_of_values': tree, 'distinct_nodes_of_types_and_values': distinct, 'ratio_tree_to_distinct': tree / distinct if distinct else None,
            'values_with_saturated_size': saturated, 'largest_value': {'nodes_as_tree': biggest[0], 'name': biggest[1]}, 'pools': k.pool_sizes()}


def kernel_recheck(w, top=8):
    k = w.k; rows = []; bad = []
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        if val is None: continue
        t = clock(); ok, msg = k.check_open(val, ty, []); dt = clock() - t
        rows.append((dt, name))
        if not ok: bad.append((name, msg[:100]))
    rows.sort(reverse=True); tot = sum(r[0] for r in rows)
    return {'declarations': len(rows), 'seconds': tot, 'rejected': bad, 'slowest': [{'name': n, 'seconds': round(d, 4)} for d, n in rows[:top]], 'median_ms': sorted(r[0] for r in rows)[len(rows) // 2] * 1e3 if rows else None}


def independent(w):
    import recheck
    k = w.k
    t = clock(); so = recheck.SecondOpinion(k); t_load = clock() - t
    t = clock(); bad = so.problems(); t_check = clock() - t
    return {'globals': len(so.checker.g), 'load_seconds': t_load, 'check_seconds': t_check, 'rejected': [(g, n, str(m)[:100]) for g, n, m in bad]}


# ---------------------------------------------------------------------------------------------------------------------------------------------- micro-benchmarks
def median(xs): xs = sorted(xs); return xs[len(xs) // 2]


def micro(w, with_binary=True, repeats=5):
    k = w.k; out = {}
    n = 200000
    def timed(fn):
        runs = []
        for _ in range(repeats):
            t = clock(); fn(); runs.append((clock() - t) / n * 1e6)
        return {'median_us': median(runs), 'min_us': min(runs), 'max_us': max(runs)}
    def trivial():
        for _ in range(n): k.num_globals()
    def pair():
        for _ in range(n): c = k.checkpoint(); k.rewind(c)
    out['ffi_call_baseline'] = timed(trivial)               # (the price of one trivial call from Python into the kernel: part of every number below)
    out['checkpoint_rewind_empty'] = timed(pair)
    # a rewind that reclaims allocated nodes: a term of m fresh applications of the library's functions
    if 'Nat' in w.gid and 'add' in w.gid:
        m = 2000; reps = 30; alloc = []; rew = []
        for rep in range(reps):
            c = k.checkpoint(); t = clock()
            x = w['Zero']
            for i in range(m): x = w['add'](x, w['Succ'](x))
            alloc.append(clock() - t); t = clock(); k.rewind(c); rew.append(clock() - t)
        out['rewind_after_allocating'] = {'nodes_built_about': m * 3, 'build_us_per_node_median': median(alloc) / (m * 3) * 1e6, 'rewind_us_median': median(rew) * 1e6, 'rewind_ns_per_node_median': median(rew) / (m * 3) * 1e9}
    if with_binary and 'N' in w.gid and 'N.mul' in w.gid:
        import ring as R
        S = R.SemiringN(w); res = []
        for (ea, eb) in ((200, 120), (500, 300)):
            a, b = 3 ** ea, 7 ** eb; c = a * b
            A, B, C = S.lit(a), S.lit(b), S.lit(c)
            lhs = S.mul(A, B); ts = []
            for _ in range(3):
                t = clock(); ok = k.conv_open(lhs.n, C.n, 0); ts.append(clock() - t)
            res.append({'bits_a': a.bit_length(), 'bits_b': b.bit_length(), 'product_decided_equal': bool(ok), 'seconds_median': median(ts)})
        out['binary_product_by_reduction'] = res
    return out


def run(a):
    res = {'machine': machine()}
    st = Stages(); instrument(st)
    t0 = clock()
    if a.quick:
        import nat_lib; w = nat_lib.build_nat()
    else:
        import imo_lib; w = imo_lib.build_imo()
    build_s = clock() - t0
    k = w.k
    res['build'] = {'seconds': build_s, 'globals': k.num_globals(), 'peak_rss_mb': peak_rss_mb(), 'rss_mb': rss_mb(), 'stages': st.rows, 'groups': grouped(st.rows)}
    c, axioms = census(w); res['census'] = {'by_kind': c, 'axioms': axioms}
    res['sharing'] = sharing(w)
    import bank as B
    res['determinism'] = {'world_digest': B.world_digest(w, k.num_globals()), 'globals': k.num_globals()}
    res['kernel_stats'] = {'typing_memo (hits, misses, ?)': list(k.memo_stats()), 'quick_conversion (tried, settled)': list(k.syn_stats()), 'extraction (fast, slow)': list(k.extract_stats()),
                           'conv_memo_hits': k.conv_memo_hits(), 'unify_memo_hits': k.unify_memo_hits()}
    print('build %.1f s, %d globals, peak %.0f MB' % (build_s, k.num_globals(), peak_rss_mb()), flush=True)
    print('kernel re-check ...', flush=True); res['kernel'] = kernel_recheck(w)
    print('  %d declarations, %.2f s, %d rejected' % (res['kernel']['declarations'], res['kernel']['seconds'], len(res['kernel']['rejected'])), flush=True)
    if not a.no_indep:
        print('independent checker ...', flush=True); res['independent'] = independent(w)
        print('  load %.1f s, check %.1f s, %d rejected' % (res['independent']['load_seconds'], res['independent']['check_seconds'], len(res['independent']['rejected'])), flush=True)
    print('micro ...', flush=True); res['micro'] = micro(w, with_binary=not a.quick)
    try: res['machine']['loadavg_end'] = [round(x, 2) for x in os.getloadavg()]
    except OSError: pass
    res['machine']['peak_rss_mb_total'] = peak_rss_mb()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True); ap.add_argument('--quick', action='store_true', help='the Nat library only'); ap.add_argument('--no-indep', action='store_true')
    a = ap.parse_args(argv)
    res = run(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f: json.dump(res, f, indent=1, default=str)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
