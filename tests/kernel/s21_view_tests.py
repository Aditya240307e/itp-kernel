"""S21 gate: reading terms back (`Kernel.view`) and the validity of term handles.

Python code that proves things -- tactics, the ring normaliser, feature extraction for the learner -- has to READ terms, not only build them. `view(t)` returns the structure of ONE node:

    ('type', l) ('bvar', i) ('fvar', l) ('app', f, a) ('lam', A, b) ('pi', A, B) ('let', A, v, b) ('const', gid) ('match', s, motive, [branches]) ('fix', type, dec, body) ('meta', id)

with children as handles, so sharing is preserved and one call is O(1). What this gate pins down:

  A. each node kind reads back as documented (fields in the documented order);
  B. ROUND TRIP: rebuilding any node from its view with the mk_* functions gives back the very same handle (terms are hash-consed, so this is exact equality). Checked on random terms of every
     shape and on the type and value of EVERY global of the proof library, which contains the Match / Fix terms the tactics produce;
  C. `view` agrees with `dump_raw`, an independent traversal written in C++ (the text rendered from the views must equal the dump, character for character);
  D. a handle that does not name a node of this kernel -- one from another kernel, or one reclaimed by a rewind -- is REFUSED by every entry point that takes one, never followed (the kernel indexes
     its node tables directly, so following it would read out of bounds).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, sys, time

import itp_env_lnl as E
import prop_world as PW

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond

def raises(thunk, needle=''):
    try: thunk()
    except Exception as e: return needle in str(e) and type(e).__name__ in ('ITPError', 'RuntimeError')
    return False


# ---------------------------------------------------------------------------------------------------------------- rebuilding from views
def children(v):
    """the child handles of a viewed node, in the order the views list them"""
    kind = v[0]
    if kind in ('app', 'lam', 'pi'): return (v[1], v[2])
    if kind == 'let': return (v[1], v[2], v[3])
    if kind == 'match': return (v[1], v[2], *v[3])
    if kind == 'fix': return (v[1], v[3])
    return ()


def rebuild(k, t, memo):
    """the node t, rebuilt bottom-up from its view with the mk_* constructors (iterative: a unary numeral is tens of thousands deep)"""
    stack = [t]
    while stack:
        n = stack[-1]
        if n in memo: stack.pop(); continue
        v = k.view(n); kind = v[0]
        kids = children(v)
        todo = [c for c in kids if c not in memo]
        if todo: stack.extend(todo); continue
        stack.pop(); m = lambda c: memo[c]
        if kind == 'type': r = k.mk_type(v[1])
        elif kind == 'bvar': r = k.mk_bvar(v[1])
        elif kind == 'fvar': r = k.mk_fvar(v[1])
        elif kind == 'app': r = k.mk_app(m(v[1]), m(v[2]))
        elif kind == 'lam': r = k.mk_lambda(m(v[1]), m(v[2]))
        elif kind == 'pi': r = k.mk_pi(m(v[1]), m(v[2]))
        elif kind == 'let': r = k.mk_let(m(v[1]), m(v[2]), m(v[3]))
        elif kind == 'const': r = k.mk_const(v[1])
        elif kind == 'match': r = k.mk_match(m(v[1]), m(v[2]), [m(b) for b in v[3]])
        elif kind == 'fix': r = k.mk_fix(m(v[1]), v[2], m(v[3]))
        elif kind == 'meta': r = k.mk_meta(v[1])
        else: raise AssertionError('unknown kind ' + kind)
        memo[n] = r
    return memo[t]


def render(k, t, memo):
    """the text of dump_raw, computed from views (iteratively, with a memo: shared subterms are rendered once)"""
    stack = [t]
    while stack:
        n = stack[-1]
        if n in memo: stack.pop(); continue
        v = k.view(n); kind = v[0]
        kids = children(v)
        todo = [c for c in kids if c not in memo]
        if todo: stack.extend(todo); continue
        stack.pop(); r = lambda c: memo[c]
        if kind == 'type': s = 'T%d' % v[1]
        elif kind == 'bvar': s = 'b%d' % v[1]
        elif kind == 'fvar': s = 'f%d' % v[1]
        elif kind == 'app': s = '(%s %s)' % (r(v[1]), r(v[2]))
        elif kind == 'lam': s = '(lam %s. %s)' % (r(v[1]), r(v[2]))
        elif kind == 'pi': s = '(pi %s. %s)' % (r(v[1]), r(v[2]))
        elif kind == 'let': s = '(let %s := %s in %s)' % (r(v[1]), r(v[2]), r(v[3]))
        elif kind == 'const': s = 'C%d' % v[1]
        elif kind == 'meta': s = '?%d' % v[1]
        elif kind == 'match': s = '(match %s [%s]%s)' % (r(v[1]), r(v[2]), ''.join(' | ' + r(b) for b in v[3]))
        elif kind == 'fix': s = '(fix<%d> %s := %s)' % (v[2], r(v[1]), r(v[3]))
        memo[n] = s
    return memo[t]


def random_term(k, rng, depth, nglobals, nmetas):
    """a random (usually ill-typed, always constructible) term using every node kind"""
    if depth <= 0 or rng.random() < 0.18:
        c = rng.random()
        if c < 0.25: return k.mk_type(rng.randrange(0, 4))
        if c < 0.5: return k.mk_bvar(rng.randrange(0, 5))
        if c < 0.65: return k.mk_fvar(rng.randrange(0, 5))
        if c < 0.9: return k.mk_const(rng.randrange(0, nglobals))
        return k.mk_meta(rng.randrange(0, nmetas))
    sub = lambda: random_term(k, rng, depth - 1, nglobals, nmetas)
    c = rng.random()
    if c < 0.34: return k.mk_app(sub(), sub())
    if c < 0.5: return k.mk_lambda(sub(), sub())
    if c < 0.64: return k.mk_pi(sub(), sub())
    if c < 0.72: return k.mk_let(sub(), sub(), sub())
    if c < 0.88: return k.mk_match(sub(), sub(), [sub() for _ in range(rng.choice([0, 1, 2, 3]))])
    return k.mk_fix(sub(), rng.randrange(0, 4), sub())


def main():
    t0 = time.time()
    k, g = PW.build_prop_world(classical=True)
    NAT, Z, S = (k.mk_const(g[n]) for n in ('Nat', 'Zero', 'Succ'))

    # ------------------------------------------------------------------------------------------------ A. each kind, with its fields in the documented order
    T0, T2 = k.mk_type(0), k.mk_type(2)
    b3, f4 = k.mk_bvar(3), k.mk_fvar(4)
    a = k.mk_app(S, Z); lam = k.mk_lambda(NAT, k.mk_bvar(0)); pi = k.mk_pi(NAT, T0); let = k.mk_let(NAT, Z, k.mk_bvar(0))
    mt = k.mk_match(Z, k.mk_lambda(NAT, NAT), [Z, k.mk_lambda(NAT, k.mk_bvar(0))]); fx = k.mk_fix(k.mk_pi(NAT, NAT), 1, k.mk_lambda(NAT, k.mk_bvar(0)))
    meta_id = k.fresh_meta(NAT, 0); me = k.mk_meta(meta_id)
    expect(k.view(T2) == ('type', 2), 'A1 type: %r' % (k.view(T2),))
    expect(k.view(b3) == ('bvar', 3), 'A2 bvar: %r' % (k.view(b3),))
    expect(k.view(f4) == ('fvar', 4), 'A3 fvar: %r' % (k.view(f4),))
    expect(k.view(a) == ('app', S, Z), 'A4 app: function first, argument second')
    expect(k.view(lam) == ('lam', NAT, k.mk_bvar(0)), 'A5 lam: binder type first, body second')
    expect(k.view(pi) == ('pi', NAT, T0), 'A6 pi: domain first, codomain second')
    expect(k.view(let) == ('let', NAT, Z, k.mk_bvar(0)), 'A7 let: type, value, body')
    expect(k.view(mt) == ('match', Z, k.mk_lambda(NAT, NAT), [Z, k.mk_lambda(NAT, k.mk_bvar(0))]), 'A8 match: scrutinee, motive, branches in order')
    expect(k.view(fx) == ('fix', k.mk_pi(NAT, NAT), 1, k.mk_lambda(NAT, k.mk_bvar(0))), 'A9 fix: type, decreasing argument, body')
    expect(k.view(me) == ('meta', meta_id), 'A10 meta')
    expect(k.view(Z) == ('const', g['Zero']), 'A11 const: the global id')
    # an application is binary: a spine of three arguments is three nested nodes, the function innermost
    sp = k.mk_app(k.mk_app(k.mk_app(k.mk_const(g['Eq']), NAT), Z), Z); v1 = k.view(sp)
    expect(v1[0] == 'app' and v1[2] == Z and k.view(v1[1])[2] == Z and k.view(k.view(v1[1])[1])[2] == NAT, 'A12 app spine nesting')
    expect(isinstance(k.view(mt)[3], list), 'A13 match branches come back as a list')

    # ------------------------------------------------------------------------------------------------ B. round trip
    memo = {}
    nodes = 0
    for gid in range(k.num_globals()):
        name, ty, val, *_ = k.global_info(gid)
        expect(rebuild(k, ty, memo) == ty, 'B1 round trip of the type of global %s' % name)
        if val is not None: expect(rebuild(k, val, memo) == val, 'B2 round trip of the value of global %s' % name)
    nodes += len(memo)
    kinds_seen = {k.view(n)[0] for n in memo}
    for need in ('app', 'lam', 'pi', 'const', 'match', 'fix', 'type', 'bvar'):
        expect(need in kinds_seen, 'B3 the library contains no %s node, so the round trip does not cover it' % need)
    rng = random.Random(21); ng = k.num_globals()
    for i in range(400):
        t = random_term(k, rng, rng.choice([2, 4, 6, 8]), ng, meta_id + 1)
        expect(rebuild(k, t, {}) == t, 'B4 round trip of a random term #%d' % i)
    # a unary numeral 50000 deep (the walk is iterative on both sides)
    big = Z
    for _ in range(50000): big = k.mk_app(S, big)
    expect(rebuild(k, big, {}) == big, 'B5 round trip of a deep numeral')

    # ------------------------------------------------------------------------------------------------ C. agreement with dump_raw
    for gid in range(k.num_globals()):
        name, ty, val, *_ = k.global_info(gid)
        for part, t in (('type', ty), ('value', val)):
            if t is None: continue
            expect(render(k, t, {}) == k.dump_raw(t), 'C1 views and dump_raw disagree on the %s of global %s' % (part, name))
    for i in range(400):
        t = random_term(k, rng, rng.choice([2, 4, 6]), ng, meta_id + 1)
        expect(render(k, t, {}) == k.dump_raw(t), 'C2 views and dump_raw disagree on random term #%d' % i)

    # ------------------------------------------------------------------------------------------------ D. handles that are not nodes of this kernel
    # The victim is a fresh library kernel (a few thousand nodes); the foreign handles come from another kernel whose pools are far larger.
    k, g = PW.build_prop_world(); NAT, Z, S = (k.mk_const(g[n]) for n in ('Nat', 'Zero', 'Succ'))
    other = E.Kernel(); o = other.mk_type(0)
    for _ in range(200_000): o = other.mk_app(o, o)                       # an App handle whose index is far beyond anything the library kernel has
    far = o
    far_lam = far_pi = far_match = far_fix = far_let = far
    for i in range(20_000):                                               # a long chain of each kind, so that the last handle of each pool is far beyond the victim's pool of that kind
        far_lam = other.mk_lambda(far_lam, far_lam); far_pi = other.mk_pi(far_pi, far_pi); far_match = other.mk_match(far_match, far, [far_match]); far_fix = other.mk_fix(far_fix, i % 3, far); far_let = other.mk_let(far_let, far, far_let)
    probes = {
        'view': lambda: k.view(far), 'dump_raw': lambda: k.dump_raw(far), 'loose_range': lambda: k.loose_range(far), 'loose_range_of': lambda: k.loose_range_of(far),
        'mk_app (function)': lambda: k.mk_app(far, Z), 'mk_app (argument)': lambda: k.mk_app(Z, far),
        'mk_lambda (type)': lambda: k.mk_lambda(far, Z), 'mk_lambda (body)': lambda: k.mk_lambda(NAT, far), 'mk_pi (domain)': lambda: k.mk_pi(far, NAT), 'mk_pi (codomain)': lambda: k.mk_pi(NAT, far),
        'mk_let (type)': lambda: k.mk_let(far, Z, Z), 'mk_let (value)': lambda: k.mk_let(NAT, far, Z), 'mk_let (body)': lambda: k.mk_let(NAT, Z, far),
        'mk_match (scrutinee)': lambda: k.mk_match(far, NAT, [Z]), 'mk_match (motive)': lambda: k.mk_match(Z, far, [Z]), 'mk_match (branch)': lambda: k.mk_match(Z, NAT, [Z, far]),
        'mk_fix (type)': lambda: k.mk_fix(far, 0, Z), 'mk_fix (body)': lambda: k.mk_fix(NAT, 0, far),
        'infer_open': lambda: k.infer_open(far, []), 'check_open (term)': lambda: k.check_open(far, NAT, []), 'check_open (expected)': lambda: k.check_open(Z, far, []), 'check_partial': lambda: k.check_partial(far, NAT, []),
        'normalize_open': lambda: k.normalize_open(far, 0), 'conv_open': lambda: k.conv_open(far, Z, 0), 'conv_open (rhs)': lambda: k.conv_open(Z, far, 0), 'unify_open': lambda: k.unify_open(far, Z, []),
        'declare_checked (type)': lambda: k.declare_checked(far, Z, 'zz1'), 'declare_checked (value)': lambda: k.declare_checked(NAT, far, 'zz2'), 'declare_unchecked': lambda: k.declare_unchecked(far, None, 'zz3'),
        'lift': lambda: k.lift(far, 1, 0), 'instantiate (term)': lambda: k.instantiate(far, [Z]), 'instantiate (value)': lambda: k.instantiate(NAT, [far]), 'abstract_fvars': lambda: k.abstract_fvars(far, [0]),
        'close_telescope (domain)': lambda: k.close_telescope('pi', [far], NAT, 0), 'close_telescope (body)': lambda: k.close_telescope('lam', [NAT], far, 0),
        'start_proof': lambda: k.start_proof(far), 'fresh_meta': lambda: k.fresh_meta(far, 0), 'reserve_inductive_type': lambda: k.reserve_inductive_type('Zz', far),
        'reserve_recursive_function': lambda: k.reserve_recursive_function('zf', far), 'materialize_from_node': lambda: k.materialize_from_node(far, [], []),
    }
    far_const = far_meta = None
    for i in range(3000):                                                  # many globals and metas, so that the last const / meta handle is beyond the victim's pools
        far_const = other.mk_const(other.reserve_inductive_type('G%d' % i, other.mk_type(0))); far_meta = other.mk_meta(other.fresh_meta(other.mk_type(0), 0))
    probes['view of a foreign meta node'] = lambda: k.view(far_meta)
    probes['mk_app of a foreign meta node'] = lambda: k.mk_app(far_meta, Z)
    for kind_name, mk in (('lam', far_lam), ('pi', far_pi), ('match', far_match), ('fix', far_fix), ('let', far_let), ('const', far_const)):
        probes['view of a foreign %s node' % kind_name] = (lambda mk=mk: k.view(mk))
    n_before = k.num_globals()
    for name, thunk in probes.items():
        expect(raises(thunk, 'invalid term handle'), 'D1 %s: a handle from another kernel was not refused cleanly' % name)
    expect(k.num_globals() == n_before, 'D2 a refused call changed the global environment')
    # in range => accepted (so the guard is not "refuse everything")
    expect(k.view(Z) == ('const', g['Zero']) and k.infer_open(Z, [])[0], 'D3 a valid handle was refused')

    # a handle reclaimed by a rewind
    cp = k.checkpoint(); t = Z
    for _ in range(2000): t = k.mk_app(S, t)
    last = k.mk_app(t, t)
    expect(k.view(last)[0] == 'app', 'D4 a fresh node is viewable')
    k.rewind(cp)
    expect(raises(lambda: k.view(last), 'invalid term handle'), 'D5 a handle reclaimed by a rewind was followed')
    expect(raises(lambda: k.mk_app(last, Z), 'invalid term handle'), 'D6 a handle reclaimed by a rewind was used to build a term')
    expect(k.view(Z) == ('const', g['Zero']), 'D7 the surviving nodes are unaffected by the rewind')

    # the FIRST node of each kind created after the checkpoint sits exactly at the end of its pool once the checkpoint is restored: the boundary case of the range check
    T0 = k.mk_type(0); cp = k.checkpoint()
    v_ = k.mk_bvar(123457); ty_ = k.mk_type(987654)
    first = {'var': v_, 'type': ty_, 'app': k.mk_app(v_, ty_), 'lam': k.mk_lambda(v_, ty_), 'pi': k.mk_pi(v_, ty_), 'let': k.mk_let(v_, ty_, v_), 'match': k.mk_match(v_, ty_, [v_]), 'fix': k.mk_fix(v_, 0, ty_)}
    first['meta'] = k.mk_meta(k.fresh_meta(T0, 0))
    for kind_name, h in first.items(): expect(k.view(h)[0] in ('bvar', 'type', 'app', 'lam', 'pi', 'let', 'match', 'fix', 'meta'), 'D8 a fresh %s node is viewable' % kind_name)
    k.rewind(cp)
    for kind_name, h in first.items(): expect(raises(lambda: k.view(h), 'invalid term handle'), 'D9 the first reclaimed %s node (the boundary of its pool) was followed' % kind_name)

    print("   %d checks, %d nodes round-tripped in the library (%.1f s)" % (CHECKS[0], nodes, time.time() - t0))
    for f in FAIL[:40]: print("   FAIL " + f)
    if len(FAIL) > 40: print("   ... and %d more" % (len(FAIL) - 40))
    print("S21 VIEW GATE: " + ("PASS" if not FAIL else "FAIL"))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
