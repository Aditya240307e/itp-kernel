"""S23 gate: the natural-number library and the `ring` tactic.

`ring` (ring.py) is UNTRUSTED: it normalises both sides of an equation to a canonical sum of monomials and builds, at every step, an application of one of the library's `ring_*` lemmas; the kernel
checks the finished term like any other proof. So what this gate pins down is (1) that `ring` is useful -- it proves every true identity of commutative-semiring arithmetic it is given, decided against
plain Python polynomial arithmetic, and refuses the false ones with the two normal forms -- (2) that nothing it produces is trusted -- the kernel AND the independently written checker accept every
proof, and both reject (and agree about) corrupted ones -- and (3) that checking such proofs stays fast.

  A. the library's statements are the ones intended (a table of ~50 theorems, compared as terms);
  B. the independent checker re-checks every global of the library, the ring lemmas and every theorem proved below;
  C. a battery of fixed identities (powers of sums, several atoms, numerals, `Succ`, applications such as `pred n` as atoms, hypotheses in the context);
  D. random expressions: an equivalent rewriting must be proved, a perturbed one is proved iff the two polynomials (computed by Python) agree, otherwise `ring` refuses;
  E. corrupted ring proofs: the kernel and the independent checker give the same verdict on every one, and the kernel rejects most;
  F. conversion of stuck recursive calls (`add x Zero` against `add x (mul Zero x)`): the kernel and the independent checker agree on a corpus of right and wrong equations;
  G. checking time: the expansion of (a+b)^8 and (a+b+c)^4 is admitted in well under a few seconds (the check used to be exponential in the nesting of the terms).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, sys, time

sys.setrecursionlimit(200000)
import itp_env_lnl as E
import nat_lib, ring as R
from author import AuthorError
from tactics import Proof
import indep_checker as ic
from harness import parse_raw

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


# ---------------------------------------------------------------------------------------------------------------------- the independent checker over a world
def build_env(k):
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid),
                recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    return env


# ---------------------------------------------------------------------------------------------------------------------- Python polynomial arithmetic (the oracle)
def padd(p, q):
    r = dict(p)
    for m, c in q.items(): r[m] = r.get(m, 0) + c
    return r
def pmul(p, q):
    r = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = tuple(a + b for a, b in zip(m1, m2)); r[m] = r.get(m, 0) + c1 * c2
    return r
def ppow(p, k, n):
    r = {tuple([0] * n): 1}
    for _ in range(k): r = pmul(r, p)
    return r

def poly(e, natoms):
    """expression -> {exponent vector: coefficient}; the atoms are x_0..x_{n-1} and pred x_0 .. pred x_{n-1}"""
    n = 2 * natoms; one = {tuple([0] * n): 1}
    k = e[0]
    if k == 'x': v = [0] * n; v[e[1]] = 1; return {tuple(v): 1}
    if k == 'pred': v = [0] * n; v[natoms + e[1]] = 1; return {tuple(v): 1}
    if k == 'n': return {tuple([0] * n): e[1]} if e[1] else {}
    if k == '+': return padd(poly(e[1], natoms), poly(e[2], natoms))
    if k == '*': return pmul(poly(e[1], natoms), poly(e[2], natoms))
    if k == '^': return ppow(poly(e[1], natoms), e[2], n)
    if k == 'S': return padd(poly(e[1], natoms), one)
    raise ValueError(k)

def clean(p): return {m: c for m, c in p.items() if c}

def rand_expr(rng, natoms, depth):
    if depth == 0 or rng.random() < 0.22:
        c = rng.random()
        if c < 0.55: return ('x', rng.randrange(natoms))
        if c < 0.82: return ('n', rng.randrange(0, 6))
        return ('pred', rng.randrange(natoms))
    r = rng.random()
    if r < 0.4: return ('+', rand_expr(rng, natoms, depth - 1), rand_expr(rng, natoms, depth - 1))
    if r < 0.75: return ('*', rand_expr(rng, natoms, depth - 1), rand_expr(rng, natoms, depth - 1))
    if r < 0.9: return ('^', rand_expr(rng, natoms, depth - 1), rng.randrange(0, 4))
    return ('S', rand_expr(rng, natoms, depth - 1))

def equiv(e, rng):
    """a rewriting of e that has the same value for every assignment of the atoms"""
    k = e[0]; r = rng.random()
    if k in '+*':
        a, b = e[1], e[2]
        if r < 0.3: return (k, equiv(b, rng), equiv(a, rng))
        if r < 0.45 and a[0] == k: return (k, a[1], (k, a[2], b))
        if r < 0.55 and k == '*' and b[0] == '+': return ('+', ('*', a, b[1]), ('*', a, b[2]))
        if r < 0.65 and k == '*' and a[0] == '+': return ('+', ('*', a[1], b), ('*', a[2], b))
        return (k, equiv(a, rng), equiv(b, rng))
    if k == '^':
        a, n = e[1], e[2]
        if r < 0.3 and n >= 1: return ('*', equiv(a, rng), ('^', a, n - 1))
        if r < 0.4 and n == 0: return ('n', 1)
        return ('^', equiv(a, rng), n)
    if k == 'S': return ('+', equiv(e[1], rng), ('n', 1)) if r < 0.4 else ('S', equiv(e[1], rng))
    if k == 'n':
        n = e[1]
        if n >= 1 and r < 0.4: return ('+', ('n', n - 1), ('n', 1))
        if r < 0.5: return ('*', ('n', n), ('n', 1))
        return e
    return e

def perturb(e, rng, natoms):
    """a small change at a random node (it may or may not change the polynomial)"""
    k = e[0]
    if k in '+*' and rng.random() < 0.75:
        if rng.random() < 0.5: return (k, perturb(e[1], rng, natoms), e[2])
        return (k, e[1], perturb(e[2], rng, natoms))
    if k == '^':
        if rng.random() < 0.5: return ('^', e[1], max(0, e[2] + rng.choice([-1, 1])))
        return ('^', perturb(e[1], rng, natoms), e[2])
    if k == 'S': return ('S', perturb(e[1], rng, natoms)) if rng.random() < 0.5 else e[1]
    if k == 'n': return ('n', max(0, e[1] + rng.choice([-1, 1])))
    if k == 'x': return ('x', rng.randrange(natoms)) if rng.random() < 0.5 else ('n', 1)
    if k == 'pred': return ('x', e[1])
    if k in '+*': return ('+' if k == '*' else '*', e[1], e[2])
    return e

def size(e): return 1 + sum(size(c) for c in e[1:] if isinstance(c, tuple))

def show_expr(e):
    k = e[0]
    if k == 'x': return 'xyz'[e[1]]
    if k == 'pred': return 'pred(%s)' % 'xyz'[e[1]]
    if k == 'n': return str(e[1])
    if k == 'S': return 'S(%s)' % show_expr(e[1])
    if k == '^': return '(%s)^%d' % (show_expr(e[1]), e[2])
    return '(%s %s %s)' % (show_expr(e[1]), k, show_expr(e[2]))

def build_term(w, e, xs):
    k = e[0]
    if k == 'x': return xs[e[1]]
    if k == 'pred': return w['pred'](xs[e[1]])
    if k == 'n': return w.nota.lit(e[1])
    if k == 'S': return w['Succ'](build_term(w, e[1], xs))
    if k == '^': return build_term(w, e[1], xs) ** e[2]
    a, b = build_term(w, e[1], xs), build_term(w, e[2], xs)
    return a + b if k == '+' else a * b


# ---------------------------------------------------------------------------------------------------------------------- term surgery (DAG-preserving) for the corruption test
def rebuild(k, t, repl):
    memo = {}
    def go(n):
        r = memo.get(n)
        if r is not None: return r
        if n in repl: res = repl[n]
        else:
            v = k.view(n); kind = v[0]
            if kind == 'app': res = k.mk_app(go(v[1]), go(v[2]))
            elif kind == 'lam': res = k.mk_lambda(go(v[1]), go(v[2]))
            elif kind == 'pi': res = k.mk_pi(go(v[1]), go(v[2]))
            elif kind in ('const', 'bvar', 'fvar', 'type'): res = n
            else: raise ValueError(kind)
        memo[n] = res; return res
    return go(t)

def dag_nodes(k, t):
    seen = []; seen_set = set(); stack = [t]
    while stack:
        n = stack.pop()
        if n in seen_set: continue
        seen_set.add(n); seen.append(n)
        v = k.view(n)
        if v[0] in ('app', 'lam', 'pi'): stack.append(v[1]); stack.append(v[2])
    return seen


def main():
    t0 = time.time(); rng = random.Random(23)
    w = nat_lib.build_nat(); k = w.k
    Nat, S, Z, add, mul, pw, pred = w['Nat'], w['Succ'], w['Zero'], w['add'], w['mul'], w['pow'], w['pred']
    Eq = w.Eq; F = w.forall_; one = S(Z)

    # ------------------------------------------------------------------------------------------------------------------ A. statements
    spec = {
        'add_zero': F([Nat], lambda n: Eq(n + Z, n)), 'zero_add': F([Nat], lambda n: Eq(Z + n, n)),
        'add_succ': F([Nat, Nat], lambda n, m: Eq(n + S(m), S(n + m))), 'succ_add': F([Nat, Nat], lambda n, m: Eq(S(n) + m, S(n + m))),
        'add_assoc': F([Nat, Nat, Nat], lambda n, m, j: Eq((n + m) + j, n + (m + j))), 'add_comm': F([Nat, Nat], lambda n, m: Eq(n + m, m + n)),
        'add_left_comm': F([Nat, Nat, Nat], lambda n, m, j: Eq(n + (m + j), m + (n + j))), 'add_right_comm': F([Nat, Nat, Nat], lambda n, m, j: Eq((n + m) + j, (n + j) + m)),
        'add_left_cancel': F([Nat, Nat, Nat, lambda n, m, j: Eq(n + m, n + j)], lambda n, m, j, h: Eq(m, j)),
        'add_right_cancel': F([Nat, Nat, Nat, lambda n, m, j: Eq(n + m, j + m)], lambda n, m, j, h: Eq(n, j)),
        'mul_zero': F([Nat], lambda n: Eq(n * Z, Z)), 'zero_mul': F([Nat], lambda n: Eq(Z * n, Z)),
        'succ_mul': F([Nat, Nat], lambda n, m: Eq(S(n) * m, m + n * m)), 'mul_succ': F([Nat, Nat], lambda n, m: Eq(n * S(m), n + n * m)),
        'one_mul': F([Nat], lambda n: Eq(one * n, n)), 'mul_one': F([Nat], lambda n: Eq(n * one, n)), 'mul_comm': F([Nat, Nat], lambda n, m: Eq(n * m, m * n)),
        'add_mul': F([Nat, Nat, Nat], lambda n, m, j: Eq((n + m) * j, n * j + m * j)), 'mul_add': F([Nat, Nat, Nat], lambda n, m, j: Eq(n * (m + j), n * m + n * j)),
        'mul_assoc': F([Nat, Nat, Nat], lambda n, m, j: Eq((n * m) * j, n * (m * j))),
        'mul_left_comm': F([Nat, Nat, Nat], lambda n, m, j: Eq(n * (m * j), m * (n * j))), 'mul_right_comm': F([Nat, Nat, Nat], lambda n, m, j: Eq((n * m) * j, (n * j) * m)),
        'pow_zero': F([Nat], lambda n: Eq(n ** Z, one)), 'pow_succ': F([Nat, Nat], lambda n, m: Eq(n ** S(m), n * n ** m)), 'pow_one': F([Nat], lambda n: Eq(n ** one, n)),
        'one_pow': F([Nat], lambda n: Eq(one ** n, one)), 'zero_pow': F([Nat], lambda n: Eq(Z ** S(n), Z)),
        'pow_add': F([Nat, Nat, Nat], lambda n, m, j: Eq(n ** (m + j), n ** m * n ** j)), 'mul_pow': F([Nat, Nat, Nat], lambda n, m, j: Eq((n * m) ** j, n ** j * m ** j)),
        'pow_mul': F([Nat, Nat, Nat], lambda n, m, j: Eq(n ** (m * j), (n ** m) ** j)),
        'succ_inj': F([Nat, Nat, lambda n, m: Eq(S(n), S(m))], lambda n, m, h: Eq(n, m)), 'succ_ne_zero': F([Nat, lambda n: Eq(S(n), Z)], lambda n, h: w['False']),
        'le_refl': F([Nat], lambda n: n <= n), 'zero_le': F([Nat], lambda n: Z <= n), 'le_succ': F([Nat], lambda n: n <= S(n)),
        'le_trans': F([Nat, Nat, Nat, lambda n, m, j: n <= m, lambda n, m, j, h: m <= j], lambda n, m, j, h, g: n <= j),
        'le_antisymm': F([Nat, Nat, lambda n, m: n <= m, lambda n, m, h: m <= n], lambda n, m, h, g: Eq(n, m)),
        'le_total': F([Nat, Nat], lambda n, m: (n <= m) | (m <= n)),
        'add_le_add': F([Nat, Nat, Nat, Nat, lambda a, b, c, d: a <= b, lambda a, b, c, d, h: c <= d], lambda a, b, c, d, h, g: a + c <= b + d),
        'mul_le_mul': F([Nat, Nat, Nat, Nat, lambda a, b, c, d: a <= b, lambda a, b, c, d, h: c <= d], lambda a, b, c, d, h, g: a * c <= b * d),
        'ring_add_congr': F([Nat, Nat, Nat, Nat, lambda a, b, c, d: Eq(a, c), lambda a, b, c, d, h: Eq(b, d)], lambda a, b, c, d, h, g: Eq(a + b, c + d)),
        'ring_mul_congr': F([Nat, Nat, Nat, Nat, lambda a, b, c, d: Eq(a, c), lambda a, b, c, d, h: Eq(b, d)], lambda a, b, c, d, h, g: Eq(a * b, c * d)),
        'ring_pow_congr': F([Nat, Nat, Nat, lambda a, c, e: Eq(a, c)], lambda a, c, e, h: Eq(a ** e, c ** e)),
        'ring_atom': F([Nat], lambda x: Eq(x, x ** one * one + Z)), 'ring_succ': F([Nat], lambda x: Eq(S(x), x + one)),
        'ring_add_lt': F([Nat, Nat, Nat, Nat, lambda a, A, B, C: Eq(A + B, C)], lambda a, A, B, C, h: Eq((a + A) + B, a + C)),
        'ring_add_gt': F([Nat, Nat, Nat, Nat, lambda b, A, B, C: Eq(A + B, C)], lambda b, A, B, C, h: Eq(A + (b + B), b + C)),
        'ring_mono_add': F([Nat, Nat, Nat, Nat, Nat, lambda x, e, a, b, s: Eq(a + b, s)], lambda x, e, a, b, s, h: Eq(x ** e * a + x ** e * b, x ** e * s)),
        'ring_scale_nil': F([Nat], lambda M: Eq(M * Z, Z)), 'ring_mulp_nil': F([Nat], lambda Q: Eq(Z * Q, Z)), 'ring_pow_zero': F([Nat], lambda P: Eq(P ** Z, one + Z)),
    }
    for name, stmt in spec.items():
        expect(name in w and w.global_type(name) == stmt, 'A  %s has the intended statement' % name)
    names_before = set(range(k.num_globals()))

    # ------------------------------------------------------------------------------------------------------------------ C. fixed identities
    counter = [0]
    def prove_ring(doms_names, stmt_fn, label, extra_intro=()):
        counter[0] += 1; nm = 's23_%d' % counter[0]
        doms = [Nat] * len(doms_names)
        stmt = F(doms, stmt_fn)
        try:
            w.prove(nm, stmt, lambda p: (p.intro(*doms_names), p.ring()))
            return True
        except AuthorError as e:
            FAIL.append('C  ring failed on %s: %s' % (label, str(e).splitlines()[0])); return False

    prove_ring(['a', 'b'], lambda a, b: Eq(a + b, b + a), 'comm')
    prove_ring(['a', 'b'], lambda a, b: Eq((a + b) ** 2, a ** 2 + 2 * a * b + b ** 2), '(a+b)^2')
    prove_ring(['a', 'b'], lambda a, b: Eq((a + b) ** 3, a ** 3 + 3 * a ** 2 * b + 3 * a * b ** 2 + b ** 3), '(a+b)^3')
    prove_ring(['a', 'b', 'c'], lambda a, b, c: Eq((a + b + c) ** 2, a ** 2 + b ** 2 + c ** 2 + 2 * a * b + 2 * b * c + 2 * a * c), '(a+b+c)^2')
    prove_ring(['a', 'b'], lambda a, b: Eq((a + b) * (a + b), a * a + 2 * a * b + b * b), '(a+b)(a+b)')
    prove_ring(['a'], lambda a: Eq(S(a) * S(a), a * a + 2 * a + 1), 'Succ')
    prove_ring(['n'], lambda n: Eq((pred(n) + 1) ** 2, pred(n) ** 2 + 2 * pred(n) + 1), 'application as an atom')
    prove_ring(['a', 'b'], lambda a, b: Eq((a + b) ** 2 + (a + b) ** 2, 2 * a ** 2 + 4 * a * b + 2 * b ** 2), 'doubling')
    prove_ring(['a'], lambda a: Eq(a ** 0, 1), 'a^0')
    prove_ring(['a'], lambda a: Eq(a ** 1 * 1 + 0, a), 'a^1')
    prove_ring(['a', 'b'], lambda a, b: Eq(0 * a + b * 0, 0), 'zeros')
    prove_ring(['a', 'b', 'c'], lambda a, b, c: Eq(a * (b + c) * (b + c), a * b * b + 2 * a * b * c + a * c * c), 'a(b+c)^2')
    # numerals only
    try:
        w.prove('s23_num', Eq((w.nota.lit(3) + w.nota.lit(4)) * w.nota.lit(5), w.nota.lit(35)), lambda p: p.ring()); ok = True
    except AuthorError as e: ok = False
    expect(ok, 'C  ring proves a closed numeral identity')
    # an extra hypothesis in the context does not matter
    try:
        stmt = F([Nat, Nat, lambda a, b: Eq(a, b)], lambda a, b, h: Eq((a + b) ** 2, a ** 2 + 2 * a * b + b ** 2))
        w.prove('s23_hyp', stmt, lambda p: (p.intro('a', 'b', 'h'), p.ring())); ok = True
    except AuthorError as e: ok = False
    expect(ok, 'C  ring works with a hypothesis in the context')
    # the 'wrong' ones are refused with both normal forms
    for label, stmt_fn, doms in (('(a+b)^2 = a^2+b^2', lambda a, b: Eq((a + b) ** 2, a ** 2 + b ** 2), 2), ('a+1 = a', lambda a: Eq(a + 1, a), 1), ('a*b = a+b', lambda a, b: Eq(a * b, a + b), 2)):
        msg = ''
        try:
            nms = ['a', 'b'][:doms]
            w.prove('s23_bad_%d' % doms, F([Nat] * doms, stmt_fn), lambda p: (p.intro(*nms), p.ring()))
        except AuthorError as e: msg = str(e)
        expect('different normal forms' in msg and 'left:' in msg and 'right:' in msg, 'C  ring refuses the false identity %s with both normal forms' % label)
    # goals that are not equations of naturals
    for label, stmt, intros in (('a <= a', F([Nat], lambda a: a <= a), ['a']), ('True', w['True'], []), ('a = a /\\ a = a', F([Nat], lambda a: Eq(a, a) & Eq(a, a)), ['a'])):
        msg = ''
        try: w.prove('s23_ne_%d' % len(label), stmt, lambda p: (p.intro(*intros) if intros else None, p.ring()))
        except (AuthorError, E.ITPError) as e: msg = str(e)
        expect('not an equation' in msg, 'C  ring refuses a goal that is not an equation (%s): %s' % (label, msg[:80]))

    # ------------------------------------------------------------------------------------------------------------------ D. random expressions against the Python oracle
    equal_proved = unequal_refused = unequal_but_equal = failures = 0
    for case in range(260):
        natoms = rng.choice([1, 2, 2, 3]); depth = rng.choice([2, 3, 3, 4])
        e1 = rand_expr(rng, natoms, depth)
        if size(e1) > 40: continue
        if rng.random() < 0.5: e2 = equiv(equiv(e1, rng), rng)
        else: e2 = perturb(equiv(e1, rng), rng, natoms)
        if size(e2) > 90: continue
        same = clean(poly(e1, natoms)) == clean(poly(e2, natoms))
        names = ['a', 'b', 'c'][:natoms]
        stmt = F([Nat] * natoms, lambda *xs: Eq(build_term(w, e1, xs), build_term(w, e2, xs)))
        nm = 's23_r%d' % case
        try:
            w.prove(nm, stmt, lambda p: (p.intro(*names), p.ring()))
            got = 'proved'
        except AuthorError as e:
            got = 'refused' if 'different normal forms' in str(e) else 'error: ' + str(e).splitlines()[0]
        except (E.ITPError, RuntimeError) as e:
            got = 'KERNEL REJECTED: ' + str(e)[:200]
        if same:
            if got == 'proved': equal_proved += 1
            else: failures += 1; FAIL.append('D  ring should prove %s = %s but: %s' % (show_expr(e1), show_expr(e2), got))
        else:
            if got == 'refused': unequal_refused += 1
            else: failures += 1; FAIL.append('D  ring should refuse %s = %s (different polynomials) but: %s' % (show_expr(e1), show_expr(e2), got))
    expect(equal_proved >= 80 and unequal_refused >= 30, 'D  the random test is not vacuous (proved %d true identities, refused %d false ones)' % (equal_proved, unequal_refused))

    # ------------------------------------------------------------------------------------------------------------------ G. time
    t1 = time.time()
    prove_ring(['a', 'b'], lambda a, b: Eq((a + b) ** 8, sum_terms(w, [(comb(8, i), 8 - i, i) for i in range(9)], a, b)), '(a+b)^8')
    dt8 = time.time() - t1; t1 = time.time()
    prove_ring(['a', 'b', 'c'], lambda a, b, c: Eq((a + b + c) ** 4, (c + b + a) ** 2 * (a + c + b) ** 2), '(a+b+c)^4')
    dt4 = time.time() - t1
    expect(dt8 < 6.0, 'G  (a+b)^8 is admitted in under 6 s (took %.2f s)' % dt8)
    expect(dt4 < 6.0, 'G  (a+b+c)^4 is admitted in under 6 s (took %.2f s)' % dt4)

    # ------------------------------------------------------------------------------------------------------------------ B. the independent checker over everything declared so far
    t2 = time.time()
    env = build_env(k); chk = ic.Checker(env, params=True, fuel=60_000_000, prop_elim=True)
    bad = chk.check_definitions()
    expect(len(chk.g) >= 280 and not bad, 'B  the independent checker re-checks all %d globals (the library, the ring lemmas and the ring theorems) without a rejection: %s' % (len(chk.g), bad[:3]))
    tb = time.time() - t2

    # ------------------------------------------------------------------------------------------------------------------ E. corrupted ring proofs
    t3 = time.time()
    st = F([Nat, Nat], lambda a, b: Eq((a + b) ** 2, a ** 2 + 2 * a * b + b ** 2))
    pr = Proof(w, st, 'e'); pr.intro('a', 'b'); pr.ring(); proof = pr.term()
    stmt_h = parse_raw(k.dump_raw(st.n), True)
    pool = [n for n in dag_nodes(k, proof.n) if k.view(n)[0] in ('app', 'const', 'lam')]
    ring_consts = [k.mk_const(w.gid[nm]) for nm in w.gid if nm.startswith('ring_')]
    kinds = {'kernel_rej': 0, 'kernel_acc': 0, 'disagree': 0, 'undecided': 0}
    for case in range(70):
        target = rng.choice(pool); v = k.view(target); r = rng.random()
        if r < 0.3: repl = rng.choice(pool)
        elif r < 0.5 and v[0] == 'app' and k.view(v[1])[0] == 'app':
            inner = k.view(v[1]); repl = k.mk_app(k.mk_app(inner[1], v[2]), inner[2])          # swap the last two arguments
        elif r < 0.7: repl = rng.choice(ring_consts)
        elif r < 0.85: repl = k.mk_const(w.gid['Zero'])
        else: repl = k.mk_app(k.mk_const(w.gid['Succ']), target) if v[0] != 'lam' else target
        if repl == target: continue
        try: mutant = rebuild(k, proof.n, {target: repl})
        except ValueError: continue
        if mutant == proof.n: continue
        try: kok = bool(k.check_open(mutant, st.n, [])[0])
        except Exception: kok = False
        try: iok, imsg = chk.check_closed(parse_raw(k.dump_raw(mutant), True), stmt_h)
        except ic.Unsupported: continue
        if not iok and 'fuel' in imsg: kinds['undecided'] += 1; continue
        if kok != iok: kinds['disagree'] += 1; FAIL.append('E  the kernel (%s) and the independent checker (%s: %s) disagree about a corrupted ring proof' % (kok, iok, imsg))
        elif kok: kinds['kernel_acc'] += 1
        else: kinds['kernel_rej'] += 1
    expect(kinds['kernel_rej'] >= 25, 'E  the corruption test is not vacuous (%s)' % kinds)
    # the untouched proof is accepted by both
    expect(k.check_open(proof.n, st.n, [])[0] and chk.check_closed(parse_raw(k.dump_raw(proof.n), True), stmt_h)[0], 'E  the uncorrupted proof is accepted by both checkers')
    te = time.time() - t3

    # ------------------------------------------------------------------------------------------------------------------ F. stuck recursive calls
    x = None
    def eqn(x_, a, b): return Eq(a, b)
    corpus = []          # (label, expected, statement builder over x, proof builder over x)
    refl_of = lambda x_, t: w['refl'](Nat, t)
    corpus.append(('add x (mul Zero x) = add x Zero', True, lambda x_: Eq(add(x_, mul(Z, x_)), add(x_, Z)), lambda x_: refl_of(x_, add(x_, Z))))
    corpus.append(('mul x (add Zero Zero) = mul x Zero', True, lambda x_: Eq(mul(x_, add(Z, Z)), mul(x_, Z)), lambda x_: refl_of(x_, mul(x_, Z))))
    corpus.append(('x^2 = x*(x*1)', True, lambda x_: Eq(pw(x_, S(S(Z))), mul(x_, mul(x_, one))), lambda x_: refl_of(x_, mul(x_, mul(x_, one)))))
    corpus.append(('add x Zero = mul x Zero  (false)', False, lambda x_: Eq(add(x_, Z), mul(x_, Z)), lambda x_: refl_of(x_, add(x_, Z))))
    corpus.append(('add x (mul Zero x) = x  (false: stuck)', False, lambda x_: Eq(add(x_, mul(Z, x_)), x_), lambda x_: refl_of(x_, x_)))
    corpus.append(('add x Zero = add x 1  (false)', False, lambda x_: Eq(add(x_, Z), add(x_, one)), lambda x_: refl_of(x_, add(x_, Z))))
    corpus.append(('mul x 1 = x  (false: stuck)', False, lambda x_: Eq(mul(x_, one), x_), lambda x_: refl_of(x_, x_)))
    corpus.append(('add x x = mul x 2  (false: stuck)', False, lambda x_: Eq(add(x_, x_), mul(x_, S(S(Z)))), lambda x_: refl_of(x_, add(x_, x_))))
    corpus.append(('add (S x) 1 = S (add x 1)', True, lambda x_: Eq(add(S(x_), one), S(add(x_, one))), lambda x_: refl_of(x_, S(add(x_, one)))))
    corpus.append(('mul (S x) x = add x (mul x x)', True, lambda x_: Eq(mul(S(x_), x_), add(x_, mul(x_, x_))), lambda x_: refl_of(x_, add(x_, mul(x_, x_)))))
    corpus.append(('mul x (S x) = add x (mul x x)  (false: stuck)', False, lambda x_: Eq(mul(x_, S(x_)), add(x_, mul(x_, x_))), lambda x_: refl_of(x_, add(x_, mul(x_, x_)))))
    for label, expected, stmt_fn, proof_fn in corpus:
        ty = w.pi(Nat, lambda x_: stmt_fn(x_)); term = w.lam(Nat, lambda x_: proof_fn(x_))
        try: kok = bool(k.check_open(term.n, ty.n, [])[0])
        except Exception: kok = False
        iok, imsg = chk.check_closed(parse_raw(k.dump_raw(term.n), True), parse_raw(k.dump_raw(ty.n), True))
        expect(kok == expected, 'F  kernel verdict on %s: expected %s, got %s' % (label, expected, kok))
        expect(iok == expected, 'F  independent verdict on %s: expected %s, got %s (%s)' % (label, expected, iok, imsg))

    # ------------------------------------------------------------------------------------------------------------------ H. an identity of open polynomials, with a time bound
    # (declared AFTER the independent re-check above: the Python checker evaluates unary numerals far too slowly for coefficients in the hundreds)
    # Typing the statement and the proof compares equal terms. That stays polynomial ONLY because identical terms are never evaluated (the shortcuts in unify); without them the
    # cost doubles with every increase of the exponent (about 32 s here instead of about 1.5 s at (a+b)^12, minutes beyond).
    t1 = time.time()
    prove_ring(['a', 'b'], lambda a, b: Eq((a + b) ** 12, sum_terms(w, [(comb(12, i), 12 - i, i) for i in range(13)], a, b)), '(a+b)^12')
    dt12 = time.time() - t1
    expect(dt12 < 10.0, 'H  (a+b)^12 is admitted in under 10 s (took %.2f s)' % dt12)

    print("   %d checks; random expressions: %d true identities proved, %d false ones refused; independent re-check of %d globals in %.1f s; %d corrupted proofs %s in %.1f s (%.1f s total)" %
          (CHECKS[0], equal_proved, unequal_refused, len(chk.g), tb, sum(kinds.values()), kinds, te, time.time() - t0))
    for f in FAIL[:40]: print("   FAIL " + f)
    if len(FAIL) > 40: print("   ... and %d more" % (len(FAIL) - 40))
    print("S23 RING GATE: " + ("PASS" if not FAIL else "FAIL"))
    sys.exit(1 if FAIL else 0)


def comb(n, r):
    from math import comb as c
    return c(n, r)

def sum_terms(w, terms, a, b):
    """sum of c * a^i * b^j over (c, i, j), as a right-nested Term"""
    out = None
    for (c, i, j) in terms:
        t = w.nota.lit(c)
        if i: t = t * (a ** i) if t is not None else a ** i
        if j: t = t * (b ** j)
        out = t if out is None else out + t
    return out


if __name__ == '__main__':
    main()
