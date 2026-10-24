"""ext_lib.py -- the library extension: facts the problems of the bank lean on, finite sums and products, factorials, and the lemmas about powers an olympiad proof needs.

It is the last part of the LIBRARY: `imo_lib.build_imo` builds it before `imo1959_p1`, the global that ends the library (bank.LIBRARY_LIMIT).  Everything here is therefore available to every problem of the
bank, to the generator of synthetic theorems, and to a model that is pre-trained on the library; no problem of the bank is among the lemmas (a lemma that was a problem is not a problem any more).

Stages (each a function of the world; `STAGES` is their order):

    nt_facts     the classical facts of elementary number theory that several problems use: no multiple of d strictly between dk and d(k+1), squares mod 3 / 4, cubes mod 9, 6 | n^3 - n,
                 3 | a^2 + b^2 forces 3 | a and 3 | b, 5 | n^5 - n, a | c and b | c and gcd(a, b) = 1 give ab | c, a^2 + b^2 = 0 forces a = b = 0, induction in steps of three,
                 a product of two integers above 1 is not prime
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys
from nt_lib import _syms
from lnl_base import thm


def fin(p, n0=None):
    """close the first goal by `ring` unless a rewrite has already closed it by computation (`n0`: the number of goals before the rewrite; without it, the proof is finished when no goal is left)"""
    if (not p.done) if n0 is None else (p.n_goals >= n0): p.ring()


def destruct(p, fact, names):
    r"""`obtain` for a fact that is a chain  exists x, exists y, A /\ B /\ C ..  (nested to the right): one name per component, in order"""
    if len(names) == 2: return p.obtain(names, fact)
    p.obtain([names[0], '_d1'], fact)
    for i in range(1, len(names) - 2): p.cases('_d%d' % i, names=[names[i], '_d%d' % (i + 1)])
    p.cases('_d%d' % (len(names) - 2), names=[names[-2], names[-1]])


def nt_facts(w):
    s = _syms(w); Z, N, Eq, Ex, F, zzero, zone, zle, zlt, D, PROP = s.Z, s.N, s.Eq, s.Ex, s.F, s.zzero, s.zone, s.zle, s.zlt, s.D, w.PROP
    prime = w['Z.prime']
    with w.nota.using(Z):
        lit = w.nota.lit

        @thm(w, 'not_dvd_between', [Z, Z, Z, lambda d, k, x: zlt(zzero, d), lambda d, k, x, h: zlt(d * k, x), lambda d, k, x, h, h1: zlt(x, d * (k + 1))], lambda d, k, x, h, h1, h2: ~D(d, x))
        def _(p):
            p.intro('d', 'k', 'x', 'hd', 'h1', 'h2', 'hx')
            p.cases('hx', names=['m', 'hm'])
            p.obtain(['c', 'c'], ('Z.le_or_lt', 'm', 'k'))
            p.nlinarith(); p.nlinarith()

        @thm(w, 'square_mod_four', [Z], lambda n: Ex(Z, w.lam(Z, lambda k: Eq(n ** 2, 4 * k) | Eq(n ** 2, 4 * k + 1))))
        def _(p):
            p.intro('n')
            p.have('h2', p.T('', lambda: zlt(zzero, lit(2))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(2), 'h2', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 1)
            p.rw('r_eq'); p.exists(p.T('q', lambda q: q ** 2)); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: q ** 2 + q)); p.right(); p.ring()

        @thm(w, 'square_mod_three', [Z], lambda n: Ex(Z, w.lam(Z, lambda k: Eq(n ** 2, 3 * k) | Eq(n ** 2, 3 * k + 1))))
        def _(p):
            p.intro('n')
            p.have('h3', p.T('', lambda: zlt(zzero, lit(3))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 2)
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 2)); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 2 + 2 * q)); p.right(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 2 + 4 * q + 1)); p.right(); p.ring()

        @thm(w, 'cube_mod_nine', [Z], lambda n: Ex(Z, w.lam(Z, lambda k: Eq(n ** 3, 9 * k) | (Eq(n ** 3, 9 * k + 1) | Eq(n ** 3, 9 * k + 8)))))
        def _(p):
            p.intro('n')
            p.have('h3', p.T('', lambda: zlt(zzero, lit(3))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 2)
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 3)); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 3 + 3 * q ** 2 + q)); p.right(); p.left(); p.ring()
            p.rw('r_eq'); p.exists(p.T('q', lambda q: 3 * q ** 3 + 6 * q ** 2 + 4 * q)); p.right(); p.right(); p.ring()

        @thm(w, 'cube_minus_n_div6', [Z], lambda n: D(lit(6), n ** 3 - n))
        def _(p):
            p.intro('n')
            p.have('h6', p.T('', lambda: zlt(zzero, lit(6))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(6), 'h6', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 5)
            for k in range(6):                      # n = 6q + k:  n^3 - n = 6 (36 q^3 + 18 k q^2 + 3 k^2 q - q + (k^3 - k) / 6)
                p.rw('r_eq')
                p.exists(p.T('q', lambda q, k=k: lit(36) * q ** 3 + lit(18 * k) * q ** 2 + lit(3 * k * k) * q - q + lit((k ** 3 - k) // 6)))
                p.ring()

        @thm(w, 'three_dvd_sum_squares', [Z, Z, lambda a, c: D(lit(3), a ** 2 + c ** 2)], lambda a, c, h: D(lit(3), a) & D(lit(3), c))
        def _(p):
            p.intro('a', 'c', 'h'); p.cases('h', names=['k', 'hk'])
            p.have('h3', p.T('', lambda: zlt(zzero, lit(3))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'a'), ['q', 'r', 'e1', 'h10', 'h11'])
            destruct(p, ('Z.exists_divmod', lit(3), 'h3', 'c'), ['t', 'u', 'e2', 'h20', 'h21'])
            p.rw_at('hk', 'e1'); p.rw_at('hk', 'e2')
            p.interval_cases('r', 0, 2)
            for rv in range(3):
                p.interval_cases('u', 0, 2)
                for uv in range(3):
                    p.rw_at('hk', 'r_eq'); p.rw_at('hk', 'u_eq')
                    if rv == 0 and uv == 0:
                        p.split()
                        p.exists('q'); p.rw('e1'); p.rw('r_eq'); p.ring()
                        p.exists('t'); p.rw('e2'); p.rw('u_eq'); p.ring()
                    else:
                        x = rv * rv + uv * uv; k0 = x // 3
                        p.exfalso()
                        p.apply('not_dvd_between', lit(3), lit(k0), lit(x))
                        p.linarith(); p.linarith(); p.linarith()
                        p.exists(p.T('k q t', lambda k_, q_, t_: k_ - 3 * q_ ** 2 - 2 * rv * q_ - 3 * t_ ** 2 - 2 * uv * t_))
                        p.linear_combination((1, 'hk'))

        @thm(w, 'sum_squares_zero', [Z, Z, lambda a, c: Eq(a ** 2 + c ** 2, zzero)], lambda a, c, h: Eq(a, zzero) & Eq(c, zzero))
        def _(p):
            p.intro('a', 'c', 'h')
            p.have('ha', p.T('a', lambda a_: Eq(a_ * a_, zzero)), lambda q: q.nlinarith(('Z.sq_nonneg', q.h('a')), ('Z.sq_nonneg', q.h('c'))))
            p.have('hc', p.T('c', lambda c_: Eq(c_ * c_, zzero)), lambda q: q.nlinarith(('Z.sq_nonneg', q.h('a')), ('Z.sq_nonneg', q.h('c'))))
            p.split()
            p.obtain(['u', 'u'], ('Z.mul_eq_zero', 'a', 'a', 'ha')); p.exact('u'); p.exact('u')
            p.obtain(['v', 'v'], ('Z.mul_eq_zero', 'c', 'c', 'hc')); p.exact('v'); p.exact('v')

        @thm(w, 'five_dvd_fifth_power', [Z], lambda n: D(lit(5), n ** 5 - n))
        def _(p):
            p.intro('n')
            p.have('h5', p.T('', lambda: zlt(zzero, lit(5))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(5), 'h5', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.rw('e')
            p.interval_cases('r', 0, 4)
            for k in range(5):                      # (5q + k)^5 - (5q + k) = 5 (625 q^5 + 625 k q^4 + 250 k^2 q^3 + 50 k^3 q^2 + (5 k^4 - 1) q + (k^5 - k) / 5)
                p.rw('r_eq')
                p.exists(p.T('q', lambda q, k=k: lit(625) * q ** 5 + lit(625 * k) * q ** 4 + lit(250 * k * k) * q ** 3 + lit(50 * k ** 3) * q ** 2 + lit(5 * k ** 4 - 1) * q + lit((k ** 5 - k) // 5)))
                p.ring()

        zgcd = w['Z.is_gcd']

        @thm(w, 'dvd_coprime_product', [Z, Z, Z, lambda a, c, d: zgcd(a, c, zone), lambda a, c, d, h: D(a, d), lambda a, c, d, h, h1: D(c, d)], lambda a, c, d, h, h1, h2: D(a * c, d))
        def _(p):                                                              # c = a m and b | a m with gcd(b, a) = 1 give b | m
            p.intro('a', 'c', 'd', 'h', 'ha', 'hb')
            p.cases('ha', names=['m', 'hm'])
            p.have('hc', p.T('a c', lambda a_, c_: w['Z.coprime'](c_, a_)), lambda q: q.exact(('Z.coprime_symm', 'a', 'c', ('Z.coprime_of_is_gcd_one', 'a', 'c', 'h'))))
            p.have('hbm', p.T('a c m', lambda a_, c_, m_: D(c_, a_ * m_)), lambda q: (q.rw_at('hb', 'hm'), q.exact('hb')))
            p.obtain(['n', 'hn'], ('Z.coprime_dvd_mul', 'c', 'a', 'm', 'hc', 'hbm'))
            p.exists('n'); p.rw('hm'); p.rw('hn'); p.ring()

        @thm(w, 'not_prime_mul', [Z, Z, lambda a, c: zlt(zone, a), lambda a, c, h: zlt(zone, c)], lambda a, c, h1, h2: ~prime(a * c))
        def _(p):
            p.intro('a', 'c', 'h1', 'h2', 'hp'); p.cases('hp', names=['g1', 'g2'])
            p.have('ha', p.T('a', lambda a_: zlt(zzero, a_)), lambda q: q.linarith())
            p.obtain(['e', 'e'], ('g2', 'a', 'ha', ('Z.dvd_mul_right', 'a', 'c')))
            p.linarith()
            p.nlinarith()

    # induction in steps of three
    nsucc, N0 = w['N.succ'], w['N0']
    PN = w.arrow(N, PROP)

    def S3(m): return nsucc(nsucc(nsucc(m)))
    @thm(w, 'N.induction3', [PN, N, lambda P, n: P(N0), lambda P, n, h0: P(nsucc(N0)), lambda P, n, h0, h1: P(nsucc(nsucc(N0))),
                              lambda P, n, h0, h1, h2: F([N, lambda m: P(m)], lambda m, hm: P(S3(m)))],
         lambda P, n, h0, h1, h2, hs: P(n))
    def _(p):
        p.intro('P', 'n', 'h0', 'h1', 'h2', 'hs')
        st = p.T('P', lambda P: F([N], lambda k: P(k) & (P(nsucc(k)) & P(nsucc(nsucc(k))))))
        def main(q):
            q.intro('k'); q.induction_on('N.induction_on', 'k')
            q.split(); q.exact('h0'); q.split(); q.exact('h1'); q.exact('h2')
            q.intro('m', 'hm'); q.cases('hm', names=['a', 'bc']); q.cases('bc', names=['b', 'c'])
            q.split(); q.exact('b'); q.split(); q.exact('c'); q.apply('hs'); q.exact('a')
        p.have('hq', st, main)
        p.obtain(['a', 'bc'], ('hq', 'n')); p.exact('a')



def sums(w):
    """finite sums  Z.sum f n = f 0 + f 1 + .. + f (n-1)  (f : Z -> Z, n : N), by recursion on the unary numeral of n; the lemmas that induction proofs about sums use"""
    s = _syms(w); Z, N, Eq, Ex, F, zzero, zone, zle, zlt, D, PROP = s.Z, s.N, s.Eq, s.Ex, s.F, s.zzero, s.zone, s.zle, s.zlt, s.D, w.PROP
    Nat = w['Nat']
    zadd, zmul, zof = w['Z.add'], w['Z.mul'], w['Z.of_N']
    nsucc, N0 = w['N.succ'], w['N0']
    ntn, ofn = w['N.to_nat'], w['N.of_nat']
    fZ = w.arrow(Z, Z); fN = w.arrow(Nat, Z); PZ = w.arrow(Z, PROP)

    w.recursive('Z.sumNat', w.forall_([fN, Nat], lambda f, n: Z),
                lambda sn: w.fun([fN, Nat], lambda f, n: w.match(n, lambda _: Z, [zzero, lambda m: zadd(sn(f, m), f(m))])), on=1)
    sumNat = w['Z.sumNat']
    w.define('Z.sum', w.forall_([fZ, N], lambda f, n: Z), w.fun([fZ, N], lambda f, n: sumNat(w.lam(Nat, lambda k: f(zof(ofn(k)))), ntn(n))))
    zsum = w['Z.sum']

    def rng(n, body):                     # forall i, 0 <= i -> i < n -> body i
        return F([Z, lambda i: zle(zzero, i), lambda i, h: zlt(i, zof(n))], lambda i, h1, h2: body(i))

    with w.nota.using(Z):
        @thm(w, 'Z.range_weaken', [PZ, N, lambda P, m: rng(nsucc(m), P)], lambda P, m, h: rng(m, P))
        def _(p):
            p.intro('P', 'm', 'h', 'i', 'a', 'b'); p.apply('h'); p.exact('a'); p.linarith(('Z.of_N_succ', 'm'))

        @thm(w, 'Z.range_last', [PZ, N, lambda P, m: rng(nsucc(m), P)], lambda P, m, h: P(zof(m)))
        def _(p):
            p.intro('P', 'm', 'h'); p.apply('h'); p.exact(('Z.nonneg_of_N', 'm')); p.linarith(('Z.of_N_succ', 'm'))

        @thm(w, 'Z.range_extend', [PZ, N, lambda P, m: rng(m, P), lambda P, m, h1: P(zof(m))], lambda P, m, h1, h2: rng(nsucc(m), P))
        def _(p):
            p.intro('P', 'm', 'h1', 'h2', 'i', 'a', 'b')
            p.have('b2', p.T('i m', lambda i, m: zlt(i, zof(m) + zone)), lambda q: q.linarith(('Z.of_N_succ', 'm')))
            p.obtain(['c', 'c'], ('Z.lt_or_eq_of_le', 'i', (zof, 'm'), ('Z.le_of_lt_add_one', 'i', (zof, 'm'), 'b2')))
            p.apply('h1'); p.exact('a'); p.exact('c')
            p.rw('c'); p.exact('h2')

        @thm(w, 'Z.sum_def', [fZ, N], lambda f, n: Eq(zsum(f, n), sumNat(w.lam(Nat, lambda k: f(zof(ofn(k)))), ntn(n))))
        def _(p): p.intro('f', 'n'); p.refl()

        @thm(w, 'Z.sum_zero', [fZ], lambda f: Eq(zsum(f, N0), zzero))
        def _(p): p.intro('f'); p.refl()

        @thm(w, 'Z.sum_succ', [fZ, N], lambda f, n: Eq(zsum(f, nsucc(n)), zsum(f, n) + f(zof(n))))
        def _(p):
            p.intro('f', 'n'); p.rw('Z.sum_def'); p.rw('Z.sum_def'); p.rw('N.to_nat_succ'); p.rw('N.of_nat_to_nat')

        @thm(w, 'Z.sum_congr', [fZ, fZ, N, lambda f, g, n: rng(n, lambda i: Eq(f(i), g(i)))], lambda f, g, n, h: Eq(zsum(f, n), zsum(g, n)))
        def _(p):
            p.intro('f', 'g', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h'); p.refl()
            p.intro('m', 'ih', 'h')
            p.have('h1', p.T('f g m', lambda f, g, m: rng(m, lambda i: Eq(f(i), g(i)))), lambda q: q.exact(('Z.range_weaken', q.T('f g', lambda f, g: w.lam(Z, lambda i: Eq(f(i), g(i)))), 'm', 'h')))
            p.have('h2', p.T('f g m', lambda f, g, m: Eq(f(zof(m)), g(zof(m)))), lambda q: q.exact(('Z.range_last', q.T('f g', lambda f, g: w.lam(Z, lambda i: Eq(f(i), g(i)))), 'm', 'h')))
            p.rw('Z.sum_succ'); p.rw('Z.sum_succ'); p.rw('ih', 'h1'); p.rw('h2')

        @thm(w, 'Z.sum_add', [fZ, fZ, N], lambda f, g, n: Eq(zsum(w.lam(Z, lambda i: f(i) + g(i)), n), zsum(f, n) + zsum(g, n)))
        def _(p):
            p.intro('f', 'g', 'n'); p.induction_on('N.induction_on', 'n')
            p.refl()
            p.intro('m', 'ih')
            p.rw('Z.sum_succ'); p.rw('Z.sum_succ'); p.rw('Z.sum_succ'); p.rw('ih'); p.ring()

        @thm(w, 'Z.sum_mul_left', [Z, fZ, N], lambda c, f, n: Eq(zsum(w.lam(Z, lambda i: c * f(i)), n), c * zsum(f, n)))
        def _(p):
            p.intro('c', 'f', 'n'); p.induction_on('N.induction_on', 'n')
            p.rw('Z.sum_zero'); p.rw('Z.sum_zero'); p.ring()
            p.intro('m', 'ih')
            p.rw('Z.sum_succ'); p.rw('Z.sum_succ'); p.rw('ih'); p.ring()

        @thm(w, 'Z.sum_const', [Z, N], lambda c, n: Eq(zsum(w.lam(Z, lambda i: c), n), zof(n) * c))
        def _(p):
            p.intro('c', 'n'); p.induction_on('N.induction_on', 'n')
            p.rw('Z.sum_zero'); p.rw('Z.of_N_zero'); p.ring()
            p.intro('m', 'ih')
            p.rw('Z.sum_succ'); p.rw('ih'); p.rw('Z.of_N_succ'); p.ring()

        @thm(w, 'Z.sum_nonneg', [fZ, N, lambda f, n: rng(n, lambda i: zle(zzero, f(i)))], lambda f, n, h: zle(zzero, zsum(f, n)))
        def _(p):
            p.intro('f', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h'); p.rw('Z.sum_zero'); p.linarith()
            p.intro('m', 'ih', 'h')
            p.have('h1', p.T('f m', lambda f, m: rng(m, lambda i: zle(zzero, f(i)))), lambda q: q.exact(('Z.range_weaken', q.T('f', lambda f: w.lam(Z, lambda i: zle(zzero, f(i)))), 'm', 'h')))
            p.have('h2', p.T('f m', lambda f, m: zle(zzero, f(zof(m)))), lambda q: q.exact(('Z.range_last', q.T('f', lambda f: w.lam(Z, lambda i: zle(zzero, f(i)))), 'm', 'h')))
            p.rw('Z.sum_succ'); p.linarith(('ih', 'h1'))

        @thm(w, 'Z.sum_le', [fZ, fZ, N, lambda f, g, n: rng(n, lambda i: zle(f(i), g(i)))], lambda f, g, n, h: zle(zsum(f, n), zsum(g, n)))
        def _(p):
            p.intro('f', 'g', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h'); p.rw('Z.sum_zero'); p.rw('Z.sum_zero'); p.linarith()
            p.intro('m', 'ih', 'h')
            p.have('h1', p.T('f g m', lambda f, g, m: rng(m, lambda i: zle(f(i), g(i)))), lambda q: q.exact(('Z.range_weaken', q.T('f g', lambda f, g: w.lam(Z, lambda i: zle(f(i), g(i)))), 'm', 'h')))
            p.have('h2', p.T('f g m', lambda f, g, m: zle(f(zof(m)), g(zof(m)))), lambda q: q.exact(('Z.range_last', q.T('f g', lambda f, g: w.lam(Z, lambda i: zle(f(i), g(i)))), 'm', 'h')))
            p.rw('Z.sum_succ'); p.rw('Z.sum_succ'); p.linarith(('ih', 'h1'))

        @thm(w, 'Z.sum_id', [N], lambda n: Eq(2 * zsum(w.lam(Z, lambda i: i), n), zof(n) * (zof(n) - 1)))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            n0 = p.n_goals; p.rw('Z.sum_zero'); fin(p, n0)
            p.intro('m', 'ih'); p.rw('Z.sum_succ'); p.rw('Z.of_N_succ'); p.linear_combination((1, 'ih'))

        @thm(w, 'Z.sum_first', [N], lambda n: Eq(2 * zsum(w.lam(Z, lambda i: i + 1), n), zof(n) * (zof(n) + 1)))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            n0 = p.n_goals; p.rw('Z.sum_zero'); fin(p, n0)
            p.intro('m', 'ih'); p.rw('Z.sum_succ'); p.rw('Z.of_N_succ'); p.linear_combination((1, 'ih'))

        @thm(w, 'Z.sum_odd', [N], lambda n: Eq(zsum(w.lam(Z, lambda i: 2 * i + 1), n), zof(n) ** 2))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            n0 = p.n_goals; p.rw('Z.sum_zero'); fin(p, n0)
            p.intro('m', 'ih'); p.rw('Z.sum_succ'); p.rw('Z.of_N_succ'); p.linear_combination((1, 'ih'))

        @thm(w, 'Z.sum_sq', [N], lambda n: Eq(6 * zsum(w.lam(Z, lambda i: i ** 2), n), (zof(n) - 1) * zof(n) * (2 * zof(n) - 1)))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            n0 = p.n_goals; p.rw('Z.sum_zero'); fin(p, n0)
            p.intro('m', 'ih'); p.rw('Z.sum_succ'); p.rw('Z.of_N_succ'); p.linear_combination((1, 'ih'))

        @thm(w, 'Z.sum_cube', [N], lambda n: Eq(4 * zsum(w.lam(Z, lambda i: i ** 3), n), (zof(n) * (zof(n) - 1)) ** 2))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            n0 = p.n_goals; p.rw('Z.sum_zero'); fin(p, n0)
            p.intro('m', 'ih'); p.rw('Z.sum_succ'); p.rw('Z.of_N_succ'); p.linear_combination((1, 'ih'))

        @thm(w, 'Z.sum_telescope', [fZ, N], lambda g, n: Eq(zsum(w.lam(Z, lambda i: g(i + zone) - g(i)), n), g(zof(n)) - g(zzero)))
        def _(p):
            p.intro('g', 'n'); p.induction_on('N.induction_on', 'n')
            p.rw('Z.sum_zero'); p.rw('Z.of_N_zero'); p.ring()
            p.intro('m', 'ih')
            p.rw('Z.sum_succ'); p.rw('ih'); p.rw('Z.of_N_succ'); p.ring()


def prods(w):
    """finite products  Z.prod f n = f 0 * f 1 * .. * f (n-1),  the factorial  Z.fact n = 1 * 2 * .. * n"""
    s = _syms(w); Z, N, Eq, Ex, F, zzero, zone, zle, zlt, D, PROP = s.Z, s.N, s.Eq, s.Ex, s.F, s.zzero, s.zone, s.zle, s.zlt, s.D, w.PROP
    Nat = w['Nat']
    zadd, zmul, zof, zpow = w['Z.add'], w['Z.mul'], w['Z.of_N'], w['Z.pow']
    nsucc, N0 = w['N.succ'], w['N0']
    ntn, ofn = w['N.to_nat'], w['N.of_nat']
    fZ = w.arrow(Z, Z); fN = w.arrow(Nat, Z); PZ = w.arrow(Z, PROP)

    w.recursive('Z.prodNat', w.forall_([fN, Nat], lambda f, n: Z),
                lambda pn: w.fun([fN, Nat], lambda f, n: w.match(n, lambda _: Z, [zone, lambda m: zmul(pn(f, m), f(m))])), on=1)
    prodNat = w['Z.prodNat']
    w.define('Z.prod', w.forall_([fZ, N], lambda f, n: Z), w.fun([fZ, N], lambda f, n: prodNat(w.lam(Nat, lambda k: f(zof(ofn(k)))), ntn(n))))
    zprod = w['Z.prod']
    w.define('Z.fact', w.arrow(N, Z), w.fun([N], lambda n: zprod(w.lam(Z, lambda i: zadd(i, zone)), n)))
    zfact = w['Z.fact']

    def rng(n, body):                     # forall i, 0 <= i -> i < n -> body i
        return F([Z, lambda i: zle(zzero, i), lambda i, h: zlt(i, zof(n))], lambda i, h1, h2: body(i))

    with w.nota.using(Z):
        @thm(w, 'Z.prod_def', [fZ, N], lambda f, n: Eq(zprod(f, n), prodNat(w.lam(Nat, lambda k: f(zof(ofn(k)))), ntn(n))))
        def _(p): p.intro('f', 'n'); p.refl()

        @thm(w, 'Z.prod_zero', [fZ], lambda f: Eq(zprod(f, N0), zone))
        def _(p): p.intro('f'); p.refl()

        @thm(w, 'Z.prod_succ', [fZ, N], lambda f, n: Eq(zprod(f, nsucc(n)), zprod(f, n) * f(zof(n))))
        def _(p):
            p.intro('f', 'n'); p.rw('Z.prod_def'); p.rw('Z.prod_def'); p.rw('N.to_nat_succ'); p.rw('N.of_nat_to_nat')

        @thm(w, 'Z.prod_congr', [fZ, fZ, N, lambda f, g, n: rng(n, lambda i: Eq(f(i), g(i)))], lambda f, g, n, h: Eq(zprod(f, n), zprod(g, n)))
        def _(p):
            p.intro('f', 'g', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h'); p.refl()
            p.intro('m', 'ih', 'h')
            p.have('h1', p.T('f g m', lambda f, g, m: rng(m, lambda i: Eq(f(i), g(i)))), lambda q: q.exact(('Z.range_weaken', q.T('f g', lambda f, g: w.lam(Z, lambda i: Eq(f(i), g(i)))), 'm', 'h')))
            p.have('h2', p.T('f g m', lambda f, g, m: Eq(f(zof(m)), g(zof(m)))), lambda q: q.exact(('Z.range_last', q.T('f g', lambda f, g: w.lam(Z, lambda i: Eq(f(i), g(i)))), 'm', 'h')))
            p.rw('Z.prod_succ'); p.rw('Z.prod_succ'); p.rw('ih', 'h1'); p.rw('h2')

        @thm(w, 'Z.prod_const', [Z, N], lambda c, n: Eq(zprod(w.lam(Z, lambda i: c), n), zpow(c, n)))
        def _(p):
            p.intro('c', 'n'); p.induction_on('N.induction_on', 'n')
            p.refl()
            p.intro('m', 'ih')
            p.rw('Z.prod_succ'); p.rw('ih'); p.rw('Z.pow_succ'); p.ring()

        @thm(w, 'Z.prod_nonneg', [fZ, N, lambda f, n: rng(n, lambda i: zle(zzero, f(i)))], lambda f, n, h: zle(zzero, zprod(f, n)))
        def _(p):
            p.intro('f', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h'); p.rw('Z.prod_zero'); p.linarith()
            p.intro('m', 'ih', 'h')
            p.have('h1', p.T('f m', lambda f, m: rng(m, lambda i: zle(zzero, f(i)))), lambda q: q.exact(('Z.range_weaken', q.T('f', lambda f: w.lam(Z, lambda i: zle(zzero, f(i)))), 'm', 'h')))
            p.have('h2', p.T('f m', lambda f, m: zle(zzero, f(zof(m)))), lambda q: q.exact(('Z.range_last', q.T('f', lambda f: w.lam(Z, lambda i: zle(zzero, f(i)))), 'm', 'h')))
            p.rw('Z.prod_succ'); p.apply('Z.mul_nonneg'); p.exact(('ih', 'h1')); p.exact('h2')

        @thm(w, 'Z.prod_pos', [fZ, N, lambda f, n: rng(n, lambda i: zlt(zzero, f(i)))], lambda f, n, h: zlt(zzero, zprod(f, n)))
        def _(p):
            p.intro('f', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h'); p.rw('Z.prod_zero'); p.linarith()
            p.intro('m', 'ih', 'h')
            p.have('h1', p.T('f m', lambda f, m: rng(m, lambda i: zlt(zzero, f(i)))), lambda q: q.exact(('Z.range_weaken', q.T('f', lambda f: w.lam(Z, lambda i: zlt(zzero, f(i)))), 'm', 'h')))
            p.have('h2', p.T('f m', lambda f, m: zlt(zzero, f(zof(m)))), lambda q: q.exact(('Z.range_last', q.T('f', lambda f: w.lam(Z, lambda i: zlt(zzero, f(i)))), 'm', 'h')))
            p.rw('Z.prod_succ'); p.apply('Z.mul_pos'); p.exact(('ih', 'h1')); p.exact('h2')

        @thm(w, 'Z.fact_def', [N], lambda n: Eq(zfact(n), zprod(w.lam(Z, lambda i: i + zone), n)))
        def _(p): p.intro('n'); p.refl()

        @thm(w, 'Z.fact_zero', [], lambda: Eq(zfact(N0), zone))
        def _(p): p.refl()

        @thm(w, 'Z.fact_succ', [N], lambda n: Eq(zfact(nsucc(n)), (zof(n) + zone) * zfact(n)))
        def _(p):
            p.intro('n'); p.rw('Z.fact_def'); p.rw('Z.fact_def'); p.rw('Z.prod_succ'); p.ring()

        @thm(w, 'Z.fact_pos', [N], lambda n: zlt(zzero, zfact(n)))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            p.rw('Z.fact_zero'); p.linarith()
            p.intro('m', 'ih')
            p.rw('Z.fact_succ'); p.apply('Z.mul_pos'); p.linarith(('Z.nonneg_of_N', 'm')); p.exact('ih')

        @thm(w, 'Z.dvd_fact', [N, Z, lambda n, k: zlt(zzero, k), lambda n, k, h: zle(k, zof(n))], lambda n, k, h1, h2: D(k, zfact(n)))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            p.intro('k', 'hk', 'hn'); p.exfalso(); p.linarith('Z.of_N_zero')
            p.intro('m', 'ih', 'k', 'hk', 'hn')
            p.have('h3', p.T('k m', lambda k, m: zle(k, zof(m) + zone)), lambda q: q.linarith(('Z.of_N_succ', 'm')))
            p.rw('Z.fact_succ')
            p.obtain(['c', 'c'], ('Z.lt_or_eq_of_le', 'k', (zadd, (zof, 'm'), zone), 'h3'))
            p.apply('Z.dvd_mul_of_dvd_right'); p.apply('ih'); p.exact('hk'); p.exact(('Z.le_of_lt_add_one', 'k', (zof, 'm'), 'c'))
            p.rw('c'); p.apply('Z.dvd_mul_right')


def pows(w):
    """powers: the sign and size of x^n, (xy)^n, x^(mn), and x^n <= y^n; x - y divides x^n - y^n; the Bernoulli congruence (1 + x)^n = 1 + nx + x^2 k"""
    s = _syms(w); Z, N, Eq, Ex, F, zzero, zone, zle, zlt, D, PROP = s.Z, s.N, s.Eq, s.Ex, s.F, s.zzero, s.zone, s.zle, s.zlt, s.D, w.PROP
    zpow, nmul, zofN = w['Z.pow'], w['N.mul'], w['Z.of_N']
    nsucc, N0 = w['N.succ'], w['N0']
    with w.nota.using(Z):
        @thm(w, 'Z.one_pow', [N], lambda n: Eq(zpow(zone, n), zone))
        def _(p):
            p.intro('n'); p.induction_on('N.induction_on', 'n')
            p.refl()
            p.intro('m', 'ih'); p.rw('Z.pow_succ'); p.rw('ih'); fin(p)

        @thm(w, 'Z.pow_nonneg', [Z, N, lambda x, n: zle(zzero, x)], lambda x, n, h: zle(zzero, zpow(x, n)))
        def _(p):
            p.intro('x', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h'); p.linarith()
            p.intro('m', 'ih', 'h'); p.rw('Z.pow_succ'); p.apply('Z.mul_nonneg'); p.exact('h'); p.exact(('ih', 'h'))

        @thm(w, 'Z.pow_pos', [Z, N, lambda x, n: zlt(zzero, x)], lambda x, n, h: zlt(zzero, zpow(x, n)))
        def _(p):
            p.intro('x', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h'); p.linarith()
            p.intro('m', 'ih', 'h'); p.rw('Z.pow_succ'); p.apply('Z.mul_pos'); p.exact('h'); p.exact(('ih', 'h'))

        @thm(w, 'Z.one_le_pow', [Z, N, lambda x, n: zle(zone, x)], lambda x, n, h: zle(zone, zpow(x, n)))
        def _(p):
            p.intro('x', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h'); p.linarith()
            p.intro('m', 'ih', 'h'); p.rw('Z.pow_succ'); p.nlinarith(('ih', 'h'))

        @thm(w, 'Z.mul_pow', [Z, Z, N], lambda x, y, n: Eq(zpow(x * y, n), zpow(x, n) * zpow(y, n)))
        def _(p):
            p.intro('x', 'y', 'n'); p.induction_on('N.induction_on', 'n')
            p.refl()
            p.intro('m', 'ih'); p.rw('Z.pow_succ'); p.rw('Z.pow_succ'); p.rw('Z.pow_succ'); p.rw('ih'); fin(p)

        @thm(w, 'Z.pow_mul', [Z, N, N], lambda x, m, n: Eq(zpow(x, nmul(m, n)), zpow(zpow(x, m), n)))
        def _(p):
            p.intro('x', 'm', 'n'); p.induction_on('N.induction_on', 'n')
            p.rw('N.mul_zero')
            p.intro('k', 'ih'); p.rw('N.mul_succ'); p.rw('Z.pow_add'); p.rw('ih'); p.rw('Z.pow_succ')

        @thm(w, 'Z.pow_le_pow_left', [Z, Z, N, lambda a, b, n: zle(zzero, a), lambda a, b, n, h: zle(a, b)], lambda a, b, n, h1, h2: zle(zpow(a, n), zpow(b, n)))
        def _(p):
            p.intro('a', 'b', 'n', 'ha', 'hab')
            p.have('hb', p.T('b', lambda b: zle(zzero, b)), lambda q: q.linarith())
            p.induction_on('N.induction_on', 'n')
            p.linarith()
            p.intro('m', 'ih'); p.rw('Z.pow_succ'); p.rw('Z.pow_succ')
            p.nlinarith(('Z.pow_nonneg', 'a', 'm', 'ha'), ('Z.pow_nonneg', 'b', 'm', 'hb'))

        @thm(w, 'Z.sub_dvd_pow_sub', [Z, Z, N], lambda x, y, n: D(x - y, zpow(x, n) - zpow(y, n)))
        def _(p):
            p.intro('x', 'y', 'n'); p.induction_on('N.induction_on', 'n')
            p.exists(zzero); p.ring()
            p.intro('m', 'ih'); p.cases('ih', names=['k', 'hk'])
            p.exists(p.T('x y m k', lambda x, y, m, k: x * k + zpow(y, m)))
            p.rw('Z.pow_succ'); p.rw('Z.pow_succ')
            p.linear_combination((p.T('x', lambda x: x), 'hk'))

        @thm(w, 'Z.pow_one_add', [Z, N], lambda x, n: Ex(Z, w.lam(Z, lambda k: Eq(zpow(1 + x, n), 1 + zofN(n) * x + x ** 2 * k))))
        def _(p):
            p.intro('x', 'n'); p.induction_on('N.induction_on', 'n')
            p.exists(zzero); p.ring()
            p.intro('m', 'ih'); p.cases('ih', names=['k', 'hk'])
            p.exists(p.T('x m k', lambda x, m, k: k + zofN(m) + x * k))
            p.rw('Z.pow_succ'); p.rw('Z.of_N_succ')
            p.linear_combination((p.T('x', lambda x: 1 + x), 'hk'))


def squares(w):
    s = _syms(w); Z, Eq, Ex, zzero, zle, zlt = s.Z, s.Eq, s.Ex, s.zzero, s.zle, s.zlt
    with w.nota.using(Z):
        lit = w.nota.lit

        @thm(w, 'Z.sq_eq_sq', [Z, Z, lambda a, b: Eq(a ** 2, b ** 2)], lambda a, b, h: Eq(a, b) | Eq(a, -b))
        def _(p):
            p.intro('a', 'b', 'h')
            p.have('e', p.T('a b', lambda a, b: Eq((a - b) * (a + b), zzero)), lambda q: q.linear_combination((1, 'h')))
            p.obtain(['u', 'u'], ('Z.mul_eq_zero', p.T('a b', lambda a, b: a - b), p.T('a b', lambda a, b: a + b), 'e'))
            p.left(); p.linarith()
            p.right(); p.linarith()

        @thm(w, 'Z.even_or_odd', [Z], lambda n: Ex(Z, w.lam(Z, lambda k: Eq(n, 2 * k) | Eq(n, 2 * k + 1))))
        def _(p):
            p.intro('n')
            p.have('h2', p.T('', lambda: zlt(zzero, lit(2))), lambda q: q.linarith())
            destruct(p, ('Z.exists_divmod', lit(2), 'h2', 'n'), ['q', 'r', 'e', 'h0', 'h1'])
            p.interval_cases('r', 0, 1)
            p.exists('q'); p.left(); p.linarith()
            p.exists('q'); p.right(); p.linarith()

        @thm(w, 'not_square_between', [Z, Z, lambda y, k: zle(zzero, k), lambda y, k, h: zlt(k ** 2, y ** 2), lambda y, k, h, h1: zlt(y ** 2, (k + 1) ** 2)], lambda y, k, h, h1, h2: w['False'])
        def _(p):
            p.intro('y', 'k', 'hk', 'h1', 'h2')
            p.obtain(['c', 'c'], ('Z.le_or_lt', 'y', 'k'))
            p.obtain(['d', 'd'], ('Z.le_or_lt', p.T('k', lambda k: -k), 'y'))
            p.nlinarith(); p.nlinarith(); p.nlinarith()


def primes(w):
    s = _syms(w); Z, N, Eq, zzero, zone, zle, zlt, D = s.Z, s.N, s.Eq, s.zzero, s.zone, s.zle, s.zlt, s.D
    zprime = w['Z.prime']
    with w.nota.using(Z):
        lit = w.nota.lit
        for q, nm in ((3, 'three'), (5, 'five'), (7, 'seven'), (11, 'eleven'), (13, 'thirteen')):
            @thm(w, 'Z.prime_' + nm, [], lambda q=q: zprime(lit(q)))
            def _(p, q=q):
                p.split(); p.linarith()
                p.intro('d', 'h1', 'h2')
                p.have('hq', p.T('', lambda: zlt(zzero, lit(q))), lambda r: r.linarith())
                p.have('hle', p.T('d', lambda d: zle(d, lit(q))), lambda r: r.exact(('Z.le_of_dvd', 'd', lit(q), 'hq', 'h2')))
                p.interval_cases('d', 1, q)
                for d in range(1, q + 1):
                    if d == 1: p.left(); p.linarith()
                    elif d == q: p.right(); p.linarith()
                    else:
                        p.exfalso(); p.rw_at('h2', 'd_eq')
                        p.apply('not_dvd_between', lit(d), lit(q // d), lit(q)); p.linarith(); p.linarith(); p.linarith(); p.exact('h2')

        @thm(w, 'Z.prime_dvd_pow', [Z, lambda p: zprime(p), Z, N, lambda p, hp, x, n: D(p, x ** n)], lambda p, hp, x, n, h: D(p, x))
        def _(p):
            p.intro('p', 'hp', 'x', 'n'); p.induction_on('N.induction_on', 'n')
            p.intro('h')
            p.cases('hp', names=['g1', 'g2'])
            p.have('hone', p.T('', lambda: zlt(zzero, zone)), lambda q: q.linarith())
            p.have('hle', p.T('p', lambda p_: zle(p_, zone)), lambda q: q.exact(('Z.le_of_dvd', 'p', zone, 'hone', 'h')))
            p.exfalso(); p.linarith()
            p.intro('m', 'ih', 'h'); p.rw_at('h', 'Z.pow_succ')
            p.obtain(['c', 'c'], ('Z.prime_dvd_mul', 'p', 'hp', 'x', p.T('x m', lambda x_, m_: x_ ** m_), 'h'))
            p.exact('c')
            p.exact(('ih', 'c'))


STAGES = ('nt_facts', 'sums', 'prods', 'pows', 'squares', 'primes')


def build_ext(w, stages=STAGES):
    for name in stages: globals()[name](w)
    return w
