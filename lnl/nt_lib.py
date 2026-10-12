"""nt_lib.py -- elementary number theory over the integers `Z` (int_lib.py), every theorem proved by a script and re-checked by the kernel.

    nt_induction   integer induction in three forms (from 0 both ways, from a lower bound, strong induction from a lower bound)
    nt_dvd         `Z.dvd a b := exists k, b = a * k` and its calculus; positive divisors; `a | 1`; divisors are bounded
    nt_divmod      division with remainder: existence and uniqueness of  a = b q + r,  0 <= r < b  for b > 0
    nt_classical   the ONE axiom of this part of the library, excluded middle (`em`), and the logic that rests on it (`by_contra`, `not_forall`, `not_exists`, `not_imp`, `not_or`)
    nt_gcd         Bezout's identity, greatest common divisors (`Z.is_gcd`) and coprimality (`Z.coprime`: Bezout form) with Euclid's lemma
    nt_prime       primes (`Z.prime p := 1 < p and every positive divisor of p is 1 or p`), Euclid's lemma for primes, every n > 1 has a prime divisor

Everything before `nt_classical` is constructive (no axiom); the gate S34 pins down that exactly the lemmas that need it use `em`. Nothing here is trusted: statements are plain terms, proofs are kernel-checked."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import int_lib
from lnl_base import thm


def _syms(w):
    """the vocabulary the stages share"""
    class S: pass
    s = S()
    s.Z, s.N, s.Eq, s.PROP, s.F, s.Ex = w['Z'], w['N'], w.Eq, w.PROP, w.forall_, w['Ex']
    s.zle, s.zlt, s.zof, s.zsub, s.zneg = w['Z.le'], w['Z.lt'], w['Z.of_N'], w['Z.sub'], w['Z.neg']
    s.zzero, s.zone = w['Z.zero'], w['Z.one']
    s.PZ = w.arrow(s.Z, s.PROP)
    if 'Z.dvd' in w.gid: s.D = w['Z.dvd']
    return s


# ---------------------------------------------------------------------------------------------------------------------------------------------- induction
def nt_induction(w):
    s = _syms(w); Z, N, Eq, PROP, F = s.Z, s.N, s.Eq, s.PROP, s.F
    zle, zlt, zof, zsub, zneg, zzero, zone, PZ = s.zle, s.zlt, s.zof, s.zsub, s.zneg, s.zzero, s.zone, s.PZ
    N0, nsucc = w['N0'], w['N.succ']
    with w.nota.using(Z):
        @thm(w, 'Z.int_induction',
             [PZ, lambda P: P(zzero),
              lambda P, h0: F([Z, lambda n: P(n)], lambda n, hn: P(n + zone)),
              lambda P, h0, hs: F([Z, lambda n: P(n)], lambda n, hn: P(n - zone)),
              Z],
             lambda P, h0, hs, hp, z: P(z))
        def _(p):
            p.intro('P', 'h0', 'hs', 'hp', 'z')
            with p.focus(): P = p.h('P'); st1 = F([N], lambda n: P(zof(n))); st2 = F([N], lambda n: P(zneg(zof(n))))
            def up(q):
                q.intro('n'); q.induction_on('N.induction_on', 'n')
                q.rw('Z.of_N_zero'); q.exact('h0')
                q.intro('m', 'hm'); q.rw('Z.of_N_succ'); q.apply('hs'); q.exact('hm')
            p.have('up', st1, up)
            def down(q):
                q.intro('n'); q.induction_on('N.induction_on', 'n')
                q.have('e0', Eq(zneg(zof(N0)), zzero), lambda r: r.ring())
                q.rw('e0'); q.exact('h0')
                q.intro('m', 'hm')
                q.have('e1', q.T('m', lambda m: Eq(zneg(zof(nsucc(m))), zneg(zof(m)) - zone)), lambda r: (r.rw('Z.of_N_succ', close=False), r.ring()))
                q.rw('e1'); q.apply('hp'); q.exact('hm')
            p.have('down', st2, down)
            p.obtain(['h', 'h'], ('Z.le_total', zzero, 'z'))
            p.obtain(['k', 'hk'], ('Z.nonneg_exists', 'z', 'h'))
            p.rw('hk', rev=True); p.apply('up')
            p.have('h2', p.T('z', lambda z: zle(zzero, zneg(z))), lambda q: q.linarith())
            p.obtain(['k', 'hk'], ('Z.nonneg_exists', (zneg, 'z'), 'h2'))
            p.have('e', p.T('z k', lambda z, k: Eq(z, zneg(zof(k)))), lambda q: q.linarith())
            p.rw('e'); p.apply('down')

        @thm(w, 'Z.induction_from',
             [PZ, Z, lambda P, a: P(a),
              lambda P, a, h0: F([Z, lambda n: zle(a, n), lambda n, hn: P(n)], lambda n, hn, hp: P(n + zone)),
              Z, lambda P, a, h0, hs, n: zle(a, n)],
             lambda P, a, h0, hs, n, hn: P(n))
        def _(p):
            p.intro('P', 'a', 'h0', 'hs', 'n', 'hn')
            st = p.T('P a', lambda P, a: F([N], lambda k: P(a + zof(k))))
            def up(q):
                q.intro('k'); q.induction_on('N.induction_on', 'k')
                q.have('e0', q.T('a', lambda a: Eq(a + zof(N0), a)), lambda r: r.ring())
                q.rw('e0'); q.exact('h0')
                q.intro('m', 'hm')
                q.have('e1', q.T('a m', lambda a, m: Eq(a + zof(nsucc(m)), (a + zof(m)) + zone)), lambda r: (r.rw('Z.of_N_succ', close=False), r.ring()))
                q.have('h1', q.T('a m', lambda a, m: zle(a, a + zof(m))), lambda r: r.linarith(('Z.nonneg_of_N', 'm')))
                q.rw('e1'); q.apply('hs'); q.exact('h1'); q.exact('hm')
            p.have('up', st, up)
            p.obtain(['k', 'hk'], ('Z.nonneg_exists', (zsub, 'n', 'a'), ('Z.sub_nonneg_of_le', 'a', 'n', 'hn')))
            p.have('e', p.T('n a k', lambda n, a, k: Eq(n, a + zof(k))), lambda q: q.linarith())
            p.rw('e'); p.apply('up')

        @thm(w, 'Z.strong_induction',
             [PZ, Z, lambda P, a: F([Z, lambda n: zle(a, n), lambda n, hn: F([Z, lambda m: zle(a, m), lambda m, h1: zlt(m, n)], lambda m, h1, h2: P(m))], lambda n, hn, ih: P(n)),
              Z, lambda P, a, H, n: zle(a, n)],
             lambda P, a, H, n, hn: P(n))
        def _(p):
            p.intro('P', 'a', 'H', 'n', 'hn')
            with p.focus():
                P = p.h('P'); a = p.h('a')
                Qf = lambda t: F([Z, lambda m: zle(a, m), lambda m, h1: zlt(m, t)], lambda m, h1, h2: P(m))
                st = F([Z, lambda t: zle(a, t)], lambda t, ht: Qf(t))
                motive = w.lam(Z, Qf)
                st_base = Qf(a); st_step = F([Z, lambda t: zle(a, t), lambda t, ht: Qf(t)], lambda t, ht, ih: Qf(t + zone))
            def base(q):
                q.intro('m', 'h1', 'h2'); q.exfalso(); q.linarith()
            def step(q):
                q.intro('t', 'ht', 'IH', 'm', 'h1', 'h2')
                q.obtain(['h3', 'h3'], ('Z.lt_or_eq_of_le', 'm', 't', ('Z.le_of_lt_add_one', 'm', 't', 'h2')))
                q.apply('IH'); q.exact('h1'); q.exact('h3')
                q.rw('h3'); q.apply('H'); q.exact('ht'); q.exact('IH')
            p.have('base', st_base, base)
            p.have('step', st_step, step)
            def main(q):
                q.intro('t', 'ht'); q.exact((('Z.induction_from', motive, 'a', 'base', 'step', 't', 'ht')))
            p.have('hq', st, main)
            p.apply('H'); p.exact('hn'); p.apply('hq'); p.exact('hn')


# ---------------------------------------------------------------------------------------------------------------------------------------------- divisibility
def nt_dvd(w):
    s = _syms(w); Z, N, Eq, PROP, F, Ex = s.Z, s.N, s.Eq, s.PROP, s.F, s.Ex
    zle, zlt, zzero, zone = s.zle, s.zlt, s.zzero, s.zone
    with w.nota.using(Z):
        w.define('Z.dvd', w.forall_([Z, Z], lambda a, b: PROP), w.fun([Z, Z], lambda a, b: Ex(Z, w.lam(Z, lambda k: Eq(b, a * k)))))
    D = w['Z.dvd']
    with w.nota.using(Z):
        @thm(w, 'Z.dvd_trans', [Z, Z, Z, lambda a, b, c: D(a, b), lambda a, b, c, h: D(b, c)], lambda a, b, c, h1, h2: D(a, c))
        def _(p):
            p.intro('a', 'b', 'c', 'h1', 'h2'); p.cases('h1', names=['k', 'hk']); p.cases('h2', names=['l', 'hl'])
            p.exists(p.T('k l', lambda k, l: k * l)); p.rw('hl', close=False); p.rw('hk', close=False); p.ring()

        @thm(w, 'Z.dvd_refl', [Z], lambda a: D(a, a))
        def _(p): p.intro('a'); p.exists(zone); p.ring()

        @thm(w, 'Z.dvd_zero', [Z], lambda a: D(a, zzero))
        def _(p): p.intro('a'); p.exists(zzero); p.ring()

        @thm(w, 'Z.one_dvd', [Z], lambda a: D(zone, a))
        def _(p): p.intro('a'); p.exists('a'); p.ring()

        @thm(w, 'Z.dvd_mul_right', [Z, Z], lambda a, b: D(a, a * b))
        def _(p): p.intro('a', 'b'); p.exists('b'); p.ring()

        @thm(w, 'Z.dvd_mul_left', [Z, Z], lambda a, b: D(a, b * a))
        def _(p): p.intro('a', 'b'); p.exists('b'); p.ring()

        @thm(w, 'Z.dvd_add', [Z, Z, Z, lambda a, b, c: D(a, b), lambda a, b, c, h: D(a, c)], lambda a, b, c, h1, h2: D(a, b + c))
        def _(p):
            p.intro('a', 'b', 'c', 'h1', 'h2'); p.cases('h1', names=['k', 'hk']); p.cases('h2', names=['l', 'hl'])
            p.exists(p.T('k l', lambda k, l: k + l)); p.linear_combination((1, 'hk'), (1, 'hl'))

        @thm(w, 'Z.dvd_neg', [Z, Z, lambda a, b: D(a, b)], lambda a, b, h: D(a, -b))
        def _(p):
            p.intro('a', 'b', 'h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('k', lambda k: -k)); p.linear_combination((-1, 'hk'))

        @thm(w, 'Z.dvd_of_dvd_neg', [Z, Z, lambda a, b: D(a, -b)], lambda a, b, h: D(a, b))
        def _(p):
            p.intro('a', 'b', 'h'); p.cases('h', names=['k', 'hk']); p.exists(p.T('k', lambda k: -k)); p.linear_combination((-1, 'hk'))

        @thm(w, 'Z.dvd_sub', [Z, Z, Z, lambda a, b, c: D(a, b), lambda a, b, c, h: D(a, c)], lambda a, b, c, h1, h2: D(a, b - c))
        def _(p):
            p.intro('a', 'b', 'c', 'h1', 'h2'); p.cases('h1', names=['k', 'hk']); p.cases('h2', names=['l', 'hl'])
            p.exists(p.T('k l', lambda k, l: k - l)); p.linear_combination((1, 'hk'), (-1, 'hl'))

        @thm(w, 'Z.neg_dvd', [Z, Z, lambda a, b: D(a, b)], lambda a, b, h: D(-a, b))
        def _(p):
            p.intro('a', 'b', 'h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('k', lambda k: -k)); p.linear_combination((1, 'hk'))

        @thm(w, 'Z.dvd_of_neg_dvd', [Z, Z, lambda a, b: D(-a, b)], lambda a, b, h: D(a, b))
        def _(p):
            p.intro('a', 'b', 'h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('k', lambda k: -k)); p.linear_combination((1, 'hk'))

        @thm(w, 'Z.dvd_mul_of_dvd_left', [Z, Z, Z, lambda a, b, c: D(a, b)], lambda a, b, c, h: D(a, b * c))
        def _(p):
            p.intro('a', 'b', 'c', 'h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('k c', lambda k, c: k * c)); p.linear_combination((p.h('c'), 'hk'))

        @thm(w, 'Z.dvd_mul_of_dvd_right', [Z, Z, Z, lambda a, b, c: D(a, c)], lambda a, b, c, h: D(a, b * c))
        def _(p):
            p.intro('a', 'b', 'c', 'h'); p.cases('h', names=['k', 'hk'])
            p.exists(p.T('k b', lambda k, b: b * k)); p.linear_combination((p.h('b'), 'hk'))

        @thm(w, 'Z.mul_dvd_mul', [Z, Z, Z, Z, lambda a, b, c, d: D(a, b), lambda a, b, c, d, h: D(c, d)], lambda a, b, c, d, h1, h2: D(a * c, b * d))
        def _(p):
            p.intro('a', 'b', 'c', 'd', 'h1', 'h2'); p.cases('h1', names=['k', 'hk']); p.cases('h2', names=['l', 'hl'])
            p.exists(p.T('k l', lambda k, l: k * l)); p.linear_combination((p.h('d'), 'hk'), (p.T('a k', lambda a, k: a * k), 'hl'))

        @thm(w, 'Z.dvd_of_dvd_add_left', [Z, Z, Z, lambda a, b, c: D(a, b), lambda a, b, c, h: D(a, b + c)], lambda a, b, c, h1, h2: D(a, c))
        def _(p):
            p.intro('a', 'b', 'c', 'h1', 'h2'); p.cases('h1', names=['k', 'hk']); p.cases('h2', names=['l', 'hl'])
            p.exists(p.T('k l', lambda k, l: l - k)); p.linear_combination((1, 'hl'), (-1, 'hk'))

        @thm(w, 'Z.dvd_of_dvd_add_right', [Z, Z, Z, lambda a, b, c: D(a, c), lambda a, b, c, h: D(a, b + c)], lambda a, b, c, h1, h2: D(a, b))
        def _(p):
            p.intro('a', 'b', 'c', 'h1', 'h2'); p.cases('h1', names=['k', 'hk']); p.cases('h2', names=['l', 'hl'])
            p.exists(p.T('k l', lambda k, l: l - k)); p.linear_combination((1, 'hl'), (-1, 'hk'))

        @thm(w, 'Z.eq_zero_of_zero_dvd', [Z, lambda a: D(zzero, a)], lambda a, h: Eq(a, zzero))
        def _(p):
            p.intro('a', 'h'); p.cases('h', names=['k', 'hk']); p.linear_combination((1, 'hk'))

        @thm(w, 'Z.eq_one_of_mul_eq_one_pos', [Z, Z, lambda a, k: zlt(zzero, a), lambda a, k, h: Eq(zone, a * k)], lambda a, k, h, e: Eq(a, zone))
        def _(p):
            p.intro('a', 'k', 'h', 'e')
            p.obtain(['h1', 'h1'], ('Z.le_or_lt', 'k', zzero))
            p.exfalso(); p.nlinarith()
            p.nlinarith()

        @thm(w, 'Z.dvd_one', [Z, lambda a: D(a, zone)], lambda a, h: Eq(a, zone) | Eq(a, -zone))
        def _(p):
            p.intro('a', 'h'); p.cases('h', names=['k', 'hk'])
            p.obtain(['h1', 'h1'], ('Z.lt_trichotomy', 'a', zzero))
            # a < 0
            p.apply('inr')
            p.have('ev', p.T('a k', lambda a, k: Eq(zone, (-a) * (-k))), lambda q: q.linear_combination((1, 'hk')))
            p.have('pv', p.T('a', lambda a: zlt(zzero, -a)), lambda q: q.linarith())
            p.obtain(['h2', 'h2'], ('Z.eq_one_of_mul_eq_one_pos', (w['Z.neg'], 'a'), (w['Z.neg'], 'k'), 'pv', 'ev'))
            p.linarith()
            # a = 0  or  0 < a
            p.cases('h1', names=['h2', 'h2'])
            p.exfalso()
            p.have('e', Eq(zone, zzero), lambda q: q.linear_combination((1, 'hk'), (q.h('k'), 'h2')))
            p.linarith()
            p.apply('inl'); p.apply('Z.eq_one_of_mul_eq_one_pos', 'a', 'k'); p.exact('h2'); p.exact('hk')

        @thm(w, 'Z.le_of_dvd', [Z, Z, lambda a, b: zlt(zzero, b), lambda a, b, h: D(a, b)], lambda a, b, h, d: zle(a, b))
        def _(p):
            p.intro('a', 'b', 'h', 'd'); p.cases('d', names=['k', 'hk'])
            p.obtain(['h1', 'h1'], ('Z.le_or_lt', 'a', zzero))
            p.linarith()
            p.have('hk1', p.T('k', lambda k: zle(zone, k)), lambda q: q.nlinarith())
            p.nlinarith()

        @thm(w, 'Z.dvd_antisymm', [Z, Z, lambda a, b: zle(zzero, a), lambda a, b, h: zle(zzero, b), lambda a, b, h, h2: D(a, b), lambda a, b, h, h2, h3: D(b, a)], lambda a, b, h1, h2, h3, h4: Eq(a, b))
        def _(p):
            p.intro('a', 'b', 'h1', 'h2', 'h3', 'h4')
            p.obtain(['h5', 'h5'], ('Z.lt_or_eq_of_le', zzero, 'a', 'h1'))
            p.obtain(['h6', 'h6'], ('Z.lt_or_eq_of_le', zzero, 'b', 'h2'))
            # 0 < a, 0 < b
            p.have('l1', p.T('a b', lambda a, b: zle(a, b)), lambda q: q.exact(('Z.le_of_dvd', 'a', 'b', 'h6', 'h3')))
            p.have('l2', p.T('a b', lambda a, b: zle(b, a)), lambda q: q.exact(('Z.le_of_dvd', 'b', 'a', 'h5', 'h4')))
            p.linarith()
            # 0 < a, 0 = b: a = b * k = 0
            p.cases('h4', names=['k', 'hk'])
            p.exfalso()
            p.have('e', p.T('a', lambda a: Eq(a, zzero)), lambda q: q.linear_combination((1, 'hk'), (q.T('k', lambda k: -k), 'h6')))
            p.linarith()
            # 0 = a
            p.cases('h3', names=['k', 'hk'])
            p.linear_combination((p.T('k', lambda k: k - zone), 'h5'), (-1, 'hk'))


# ---------------------------------------------------------------------------------------------------------------------------------------------- division with remainder
def nt_divmod(w):
    s = _syms(w); Z, Eq, F, Ex = s.Z, s.Eq, s.F, s.Ex
    zle, zlt, zzero, zone = s.zle, s.zlt, s.zzero, s.zone
    with w.nota.using(Z):
        def Pdm(b):
            return lambda a: Ex(Z, w.lam(Z, lambda q: Ex(Z, w.lam(Z, lambda r: Eq(a, b * q + r) & (zle(zzero, r) & zlt(r, b))))))

        @thm(w, 'Z.exists_divmod', [Z, lambda b: zlt(zzero, b), Z], lambda b, hb, a: Pdm(b)(a))
        def _(p):
            p.intro('b', 'hb', 'a')
            with p.focus():
                b = p.h('b'); P = Pdm(b); motive = w.lam(Z, P)
                st_base = P(zzero)
                st_up = F([Z, lambda n: P(n)], lambda n, hn: P(n + zone))
                st_down = F([Z, lambda n: P(n)], lambda n, hn: P(n - zone))
            def base(q):
                q.exists(zzero); q.exists(zzero); q.split(); q.ring(); q.split(); q.linarith(); q.exact('hb')
            def up(q):
                q.intro('n', 'hn'); q.cases('hn', names=['m', 'hm']); q.cases('hm', names=['r', 'hr']); q.cases('hr', names=['e', 'hrr']); q.cases('hrr', names=['h0', 'h1'])
                q.obtain(['c', 'c'], ('Z.lt_or_eq_of_le', q.T('r', lambda r: r + zone), 'b', 'h1'))
                q.exists('m'); q.exists(q.T('r', lambda r: r + zone)); q.split(); q.linear_combination((1, 'e')); q.split(); q.linarith(); q.exact('c')
                q.exists(q.T('m', lambda m: m + zone)); q.exists(zzero); q.split(); q.linear_combination((1, 'e'), (1, 'c')); q.split(); q.linarith(); q.exact('hb')
            def down(q):
                q.intro('n', 'hn'); q.cases('hn', names=['m', 'hm']); q.cases('hm', names=['r', 'hr']); q.cases('hr', names=['e', 'hrr']); q.cases('hrr', names=['h0', 'h1'])
                q.obtain(['c', 'c'], ('Z.lt_or_eq_of_le', zzero, 'r', 'h0'))
                q.exists('m'); q.exists(q.T('r', lambda r: r - zone)); q.split(); q.linear_combination((1, 'e')); q.split(); q.linarith(); q.linarith()
                q.exists(q.T('m', lambda m: m - zone)); q.exists(q.T('b', lambda b: b - zone)); q.split(); q.linear_combination((1, 'e'), (-1, 'c')); q.split(); q.linarith(); q.linarith()
            p.have('base', st_base, base)
            p.have('up', st_up, up)
            p.have('down', st_down, down)
            p.exact(lambda q: w['Z.int_induction'](motive, q.h('base'), q.h('up'), q.h('down'), q.h('a')))

        @thm(w, 'Z.divmod_unique',
             [Z, Z, Z, Z, Z, Z, lambda a, b, q1, r1, q2, r2: zlt(zzero, b),
              lambda a, b, q1, r1, q2, r2, hb: Eq(a, b * q1 + r1) & (zle(zzero, r1) & zlt(r1, b)),
              lambda a, b, q1, r1, q2, r2, hb, h1: Eq(a, b * q2 + r2) & (zle(zzero, r2) & zlt(r2, b))],
             lambda a, b, q1, r1, q2, r2, hb, h1, h2: Eq(q1, q2) & Eq(r1, r2))
        def _(p):
            p.intro('a', 'b', 'q1', 'r1', 'q2', 'r2', 'hb', 'h1', 'h2')
            p.cases('h1', names=['e1', 'g1']); p.cases('g1', names=['g10', 'g11'])
            p.cases('h2', names=['e2', 'g2']); p.cases('g2', names=['g20', 'g21'])
            p.have('hq', p.T('q1 q2', lambda q1, q2: Eq(q1, q2)), lambda q: q.nlinarith())
            p.split(); p.exact('hq')
            p.linear_combination((-1, 'e1'), (1, 'e2'), (p.T('b', lambda b: -b), 'hq'))


# ---------------------------------------------------------------------------------------------------------------------------------------------- classical logic
def nt_classical(w):
    """excluded middle is the one axiom of the number theory; the lemmas below are its everyday consequences"""
    s = _syms(w); Z, Eq, F, Ex, PROP = s.Z, s.Eq, s.F, s.Ex, s.PROP
    w.axiom('em', F([PROP], lambda P: P | ~P))

    @thm(w, 'by_contra', [PROP, lambda P: ~~P], lambda P, h: P)
    def _(p):
        p.intro('P', 'h'); p.obtain(['h1', 'h1'], ('em', 'P'))
        p.exact('h1')
        p.exfalso(); p.apply('h'); p.exact('h1')

    @thm(w, 'not_forall', [w.TYPE, lambda A: w.arrow(A, PROP), lambda A, P: ~F([A], lambda x: P(x))], lambda A, P, h: Ex(A, w.lam(A, lambda x: ~P(x))))
    def _(p):
        p.intro('A', 'P', 'h'); p.apply('by_contra'); p.intro('h2'); p.apply('h'); p.intro('x'); p.apply('by_contra'); p.intro('h3'); p.apply('h2'); p.exists('x'); p.exact('h3')

    @thm(w, 'not_exists', [w.TYPE, lambda A: w.arrow(A, PROP), lambda A, P: ~Ex(A, w.lam(A, lambda x: P(x)))], lambda A, P, h: F([A], lambda x: ~P(x)))
    def _(p):
        p.intro('A', 'P', 'h', 'x', 'hx'); p.apply('h'); p.exists('x'); p.exact('hx')

    @thm(w, 'not_imp', [PROP, PROP, lambda P, Q: ~(P >> Q)], lambda P, Q, h: P & ~Q)
    def _(p):
        p.intro('P', 'Q', 'h'); p.obtain(['h1', 'h1'], ('em', 'P'))
        p.split(); p.exact('h1'); p.intro('hq'); p.apply('h'); p.intro('hp0'); p.exact('hq')
        p.exfalso(); p.apply('h'); p.intro('hp0'); p.exfalso(); p.apply('h1'); p.exact('hp0')

    @thm(w, 'not_or', [PROP, PROP, lambda P, Q: ~(P | Q)], lambda P, Q, h: ~P & ~Q)
    def _(p):
        p.intro('P', 'Q', 'h'); p.split(); p.intro('hp'); p.apply('h'); p.left(); p.exact('hp'); p.intro('hq'); p.apply('h'); p.right(); p.exact('hq')


# ---------------------------------------------------------------------------------------------------------------------------------------------- Bezout, gcd, coprime
def nt_gcd(w):
    s = _syms(w); Z, Eq, F, Ex, PROP = s.Z, s.Eq, s.F, s.Ex, s.PROP
    zle, zlt, zzero, zone, D = s.zle, s.zlt, s.zzero, s.zone, w['Z.dvd']
    def Bz(a, b):
        return Ex(Z, w.lam(Z, lambda d: Ex(Z, w.lam(Z, lambda x: Ex(Z, w.lam(Z, lambda y: zle(zzero, d) & (D(d, a) & (D(d, b) & Eq(d, a * x + b * y)))))))))
    with w.nota.using(Z):
        @thm(w, 'Z.bezout_aux', [Z, lambda b: zle(zzero, b), Z], lambda b, hb, a: Bz(a, b))
        def _(p):
            p.intro('b', 'hb')
            with p.focus():
                Pm = lambda m: F([Z], lambda a: Bz(a, m)); motive = w.lam(Z, Pm)
                Hst = F([Z, lambda n: zle(zzero, n), lambda n, hn: F([Z, lambda m: zle(zzero, m), lambda m, h1: zlt(m, n)], lambda m, h1, h2: Pm(m))], lambda n, hn, ih: Pm(n))
            def H(q):
                q.intro('n', 'hn', 'IH', 'a')
                q.obtain(['c', 'c'], ('Z.lt_or_eq_of_le', zzero, 'n', 'hn'))
                # 0 < n:  a = n m + r,  0 <= r < n,  and  gcd(r, n)  by the induction hypothesis
                q.obtain(['m', 't'], ('Z.exists_divmod', 'n', 'c', 'a'))
                q.cases('t', names=['r', 'tr']); q.cases('tr', names=['e', 'tg']); q.cases('tg', names=['g0', 'g1'])
                q.obtain(['d', 't2'], ('IH', 'r', 'g0', 'g1', 'n'))
                q.cases('t2', names=['x1', 'u2']); q.cases('u2', names=['y1', 'u3'])
                q.cases('u3', names=['i0', 'u4']); q.cases('u4', names=['i1', 'u5']); q.cases('u5', names=['i2', 'i3'])
                q.exists('d'); q.exists('y1'); q.exists(q.T('x1 m y1', lambda x1, m, y1: x1 - m * y1))
                q.split(); q.exact('i0'); q.split()
                q.have('da', q.T('d n m r', lambda d, n, m, r: D(d, n * m + r)),
                       lambda r_: r_.exact(('Z.dvd_add', 'd', r_.T('n m', lambda n, m: n * m), 'r', ('Z.dvd_mul_of_dvd_left', 'd', 'n', 'm', 'i1'), 'i2')))
                q.rw('e'); q.exact('da')
                q.split(); q.exact('i1')
                q.linear_combination((1, 'i3'), (q.T('y1', lambda y: -y), 'e'))
                # n = 0:  the gcd is a or -a
                q.obtain(['s', 's'], ('Z.le_or_lt', zzero, 'a'))
                q.exists('a'); q.exists(zone); q.exists(zzero); q.split(); q.exact('s'); q.split(); q.apply('Z.dvd_refl')
                q.split(); q.rw('c', rev=True); q.apply('Z.dvd_zero')
                q.rw('c', rev=True); q.ring()
                q.exists(q.T('a', lambda a: -a)); q.exists(-zone); q.exists(zzero); q.split(); q.linarith(); q.split()
                q.apply('Z.neg_dvd'); q.apply('Z.dvd_refl')
                q.split(); q.rw('c', rev=True); q.apply('Z.dvd_zero')
                q.rw('c', rev=True); q.ring()
            p.have('H', Hst, H)
            p.exact(lambda q: w['Z.strong_induction'](motive, zzero, q.h('H'), q.h('b'), q.h('hb')))

        @thm(w, 'Z.bezout', [Z, Z], lambda a, b: Bz(a, b))
        def _(p):
            p.intro('a', 'b')
            p.obtain(['s', 's'], ('Z.le_or_lt', zzero, 'b'))
            p.exact(('Z.bezout_aux', 'b', 's', 'a'))
            p.have('h2', p.T('b', lambda b: zle(zzero, -b)), lambda q: q.linarith())
            p.obtain(['d', 't'], ('Z.bezout_aux', (w['Z.neg'], 'b'), 'h2', 'a'))
            p.cases('t', names=['x', 'u2']); p.cases('u2', names=['y', 'u3']); p.cases('u3', names=['i0', 'u4']); p.cases('u4', names=['i1', 'u5']); p.cases('u5', names=['i2', 'i3'])
            p.exists('d'); p.exists('x'); p.exists(p.T('y', lambda y: -y)); p.split(); p.exact('i0'); p.split(); p.exact('i1'); p.split()
            p.apply('Z.dvd_of_dvd_neg'); p.exact('i2')
            p.linear_combination((1, 'i3'))

    with w.nota.using(Z):
        w.define('Z.coprime', w.forall_([Z, Z], lambda a, b: PROP),
                 w.fun([Z, Z], lambda a, b: Ex(Z, w.lam(Z, lambda x: Ex(Z, w.lam(Z, lambda y: Eq(a * x + b * y, zone)))))))
        w.define('Z.is_gcd', w.forall_([Z, Z, Z], lambda a, b, d: PROP),
                 w.fun([Z, Z, Z], lambda a, b, d: zle(zzero, d) & (D(d, a) & (D(d, b) & F([Z, lambda e: D(e, a), lambda e, h1: D(e, b)], lambda e, h1, h2: D(e, d))))))
    zcop = w['Z.coprime']; zgcd = w['Z.is_gcd']
    with w.nota.using(Z):
        @thm(w, 'Z.exists_gcd', [Z, Z], lambda a, b: Ex(Z, w.lam(Z, lambda d: zgcd(a, b, d))))
        def _(p):
            p.intro('a', 'b')
            p.obtain(['d', 't'], ('Z.bezout', 'a', 'b'))
            p.cases('t', names=['x', 'u2']); p.cases('u2', names=['y', 'u3']); p.cases('u3', names=['i0', 'u4']); p.cases('u4', names=['i1', 'u5']); p.cases('u5', names=['i2', 'i3'])
            p.exists('d')
            p.split(); p.exact('i0'); p.split(); p.exact('i1'); p.split(); p.exact('i2')
            p.intro('e', 'h1', 'h2'); p.rw('i3')
            p.exact(('Z.dvd_add', 'e', p.T('a x', lambda a, x: a * x), p.T('b y', lambda b, y: b * y),
                     ('Z.dvd_mul_of_dvd_left', 'e', 'a', 'x', 'h1'), ('Z.dvd_mul_of_dvd_left', 'e', 'b', 'y', 'h2')))

        @thm(w, 'Z.is_gcd_unique', [Z, Z, Z, Z, lambda a, b, d, d2: zgcd(a, b, d), lambda a, b, d, d2, h1: zgcd(a, b, d2)], lambda a, b, d, d2, h1, h2: Eq(d, d2))
        def _(p):
            p.intro('a', 'b', 'd', 'd2', 'h1', 'h2')
            p.cases('h1', names=['n0', 'h1b']); p.cases('h1b', names=['d1a', 'h1c']); p.cases('h1c', names=['d1b', 'm1'])
            p.cases('h2', names=['n2', 'h2b']); p.cases('h2b', names=['d2a', 'h2c']); p.cases('h2c', names=['d2b', 'm2'])
            p.apply('Z.dvd_antisymm'); p.exact('n0'); p.exact('n2')
            p.apply('m2'); p.exact('d1a'); p.exact('d1b')
            p.apply('m1'); p.exact('d2a'); p.exact('d2b')

        @thm(w, 'Z.coprime_symm', [Z, Z, lambda a, b: zcop(a, b)], lambda a, b, h: zcop(b, a))
        def _(p):
            p.intro('a', 'b', 'h'); p.cases('h', names=['x', 'hx']); p.cases('hx', names=['y', 'hy'])
            p.exists('y'); p.exists('x'); p.linear_combination((1, 'hy'))

        @thm(w, 'Z.coprime_dvd_mul', [Z, Z, Z, lambda a, b, c: zcop(a, b), lambda a, b, c, h: D(a, b * c)], lambda a, b, c, h1, h2: D(a, c))
        def _(p):
            p.intro('a', 'b', 'c', 'h1', 'h2')
            p.cases('h1', names=['x', 'hx']); p.cases('hx', names=['y', 'hy']); p.cases('h2', names=['m', 'hm'])
            p.exists(p.T('c x m y', lambda c, x, m, y: c * x + m * y))
            p.linear_combination((p.T('c', lambda c: -c), 'hy'), (p.h('y'), 'hm'))

        @thm(w, 'Z.coprime_mul_right', [Z, Z, Z, lambda a, b, c: zcop(a, b), lambda a, b, c, h: zcop(a, c)], lambda a, b, c, h1, h2: zcop(a, b * c))
        def _(p):
            p.intro('a', 'b', 'c', 'h1', 'h2')
            p.cases('h1', names=['x1', 'hx1']); p.cases('hx1', names=['y1', 'hy1']); p.cases('h2', names=['x2', 'hx2']); p.cases('hx2', names=['y2', 'hy2'])
            p.exists(p.T('a b c x1 y1 x2 y2', lambda a, b, c, x1, y1, x2, y2: x1 * a * x2 + x1 * c * y2 + b * y1 * x2))
            p.exists(p.T('y1 y2', lambda y1, y2: y1 * y2))
            p.linear_combination((p.T('a c x2 y2', lambda a, c, x2, y2: a * x2 + c * y2), 'hy1'), (1, 'hy2'))

        @thm(w, 'Z.is_gcd_one_of_coprime', [Z, Z, lambda a, b: zcop(a, b)], lambda a, b, h: zgcd(a, b, zone))
        def _(p):
            p.intro('a', 'b', 'h'); p.cases('h', names=['x', 'hx']); p.cases('hx', names=['y', 'hy'])
            p.split(); p.linarith(); p.split(); p.apply('Z.one_dvd'); p.split(); p.apply('Z.one_dvd')
            p.intro('e', 'h1', 'h2')
            p.have('hd', p.T('e a b x y', lambda e, a, b, x, y: D(e, a * x + b * y)),
                   lambda q: q.exact(('Z.dvd_add', 'e', q.T('a x', lambda a, x: a * x), q.T('b y', lambda b, y: b * y),
                                      ('Z.dvd_mul_of_dvd_left', 'e', 'a', 'x', 'h1'), ('Z.dvd_mul_of_dvd_left', 'e', 'b', 'y', 'h2'))))
            p.rw_at('hd', 'hy'); p.exact('hd')

        @thm(w, 'Z.coprime_of_is_gcd_one', [Z, Z, lambda a, b: zgcd(a, b, zone)], lambda a, b, h: zcop(a, b))
        def _(p):
            p.intro('a', 'b', 'h'); p.cases('h', names=['n0', 'h1']); p.cases('h1', names=['d1', 'h2']); p.cases('h2', names=['d2', 'm'])
            p.obtain(['d', 't'], ('Z.bezout', 'a', 'b'))
            p.cases('t', names=['x', 'u2']); p.cases('u2', names=['y', 'u3']); p.cases('u3', names=['i0', 'u4']); p.cases('u4', names=['i1', 'u5']); p.cases('u5', names=['i2', 'i3'])
            p.have('hd', p.T('d', lambda d: D(d, zone)), lambda q: q.exact(('m', 'd', 'i1', 'i2')))
            p.obtain(['ec', 'ec'], ('Z.dvd_one', 'd', 'hd'))
            p.exists('x'); p.exists('y'); p.linear_combination((-1, 'i3'), (1, 'ec'))
            p.exfalso(); p.linarith()


# ---------------------------------------------------------------------------------------------------------------------------------------------- primes
def nt_prime(w):
    s = _syms(w); Z, Eq, F, Ex, PROP = s.Z, s.Eq, s.F, s.Ex, s.PROP
    zle, zlt, zzero, zone, D = s.zle, s.zlt, s.zzero, s.zone, w['Z.dvd']
    zcop = w['Z.coprime']
    with w.nota.using(Z):
        w.define('Z.prime', w.forall_([Z], lambda p: PROP),
                 w.fun([Z], lambda p: zlt(zone, p) & F([Z, lambda d: zlt(zzero, d), lambda d, h: D(d, p)], lambda d, h1, h2: Eq(d, zone) | Eq(d, p))))
    zprime = w['Z.prime']
    with w.nota.using(Z):
        lit = lambda n: w.nota.lit(n)

        @thm(w, 'Z.prime_dvd_or_coprime', [Z, lambda p: zprime(p), Z], lambda p, hp, a: D(p, a) | zcop(p, a))
        def _(p):
            p.intro('p', 'hp', 'a'); p.cases('hp', names=['gt1', 'hd'])
            p.obtain(['d', 't'], ('Z.bezout', 'p', 'a'))
            p.cases('t', names=['x', 'u2']); p.cases('u2', names=['y', 'u3']); p.cases('u3', names=['i0', 'u4']); p.cases('u4', names=['i1', 'u5']); p.cases('u5', names=['i2', 'i3'])
            p.obtain(['c', 'c'], ('Z.lt_or_eq_of_le', zzero, 'd', 'i0'))
            # 0 < d: d is 1 or p
            p.obtain(['ec', 'ec'], ('hd', 'd', 'c', 'i1'))
            p.right(); p.exists('x'); p.exists('y'); p.linear_combination((-1, 'i3'), (1, 'ec'))
            p.left(); p.rw_at('i2', 'ec'); p.exact('i2')
            # d = 0 would make p = 0
            p.exfalso(); p.rw_at('i1', 'c', rev=True)
            p.have('pz', p.T('p', lambda p_: Eq(p_, zzero)), lambda q: q.exact(('Z.eq_zero_of_zero_dvd', 'p', 'i1')))
            p.linarith()

        @thm(w, 'Z.prime_dvd_mul', [Z, lambda p: zprime(p), Z, Z, lambda p, hp, a, b: D(p, a * b)], lambda p, hp, a, b, h: D(p, a) | D(p, b))
        def _(p):
            p.intro('p', 'hp', 'a', 'b', 'h')
            p.obtain(['c', 'c'], ('Z.prime_dvd_or_coprime', 'p', 'hp', 'a'))
            p.left(); p.exact('c')
            p.right(); p.exact(('Z.coprime_dvd_mul', 'p', 'a', 'b', 'c', 'h'))

        @thm(w, 'Z.prime_dvd_sq', [Z, lambda p: zprime(p), Z, lambda p, hp, a: D(p, a * a)], lambda p, hp, a, h: D(p, a))
        def _(p):
            p.intro('p', 'hp', 'a', 'h'); p.obtain(['c', 'c'], ('Z.prime_dvd_mul', 'p', 'hp', 'a', 'a', 'h'))
            p.exact('c'); p.exact('c')

        @thm(w, 'Z.prime_two', [], lambda: zprime(lit(2)))
        def _(p):
            p.split(); p.linarith()
            p.intro('d', 'h1', 'h2')
            p.have('hp2', zlt(zzero, lit(2)), lambda q: q.linarith())
            p.have('hle', p.T('d', lambda d: zle(d, lit(2))), lambda q: q.exact(('Z.le_of_dvd', 'd', lit(2), 'hp2', 'h2')))
            p.obtain(['c', 'c'], ('Z.lt_or_eq_of_le', 'd', lit(2), 'hle'))
            p.left(); p.linarith()
            p.right(); p.exact('c')

        def Pep(n): return zlt(zone, n) >> Ex(Z, w.lam(Z, lambda p: zprime(p) & D(p, n)))

        @thm(w, 'Z.exists_prime_dvd', [Z, lambda n: zlt(zone, n)], lambda n, h: Ex(Z, w.lam(Z, lambda p: zprime(p) & D(p, n))))
        def _(p):
            p.intro('n0', 'h0')
            with p.focus():
                motive = w.lam(Z, Pep)
                Hst = F([Z, lambda n: zle(zzero, n), lambda n, hn: F([Z, lambda m: zle(zzero, m), lambda m, h1: zlt(m, n)], lambda m, h1, h2: Pep(m))], lambda n, hn, ih: Pep(n))
            def H(q):
                q.intro('n', 'hn', 'IH', 'h1')
                q.obtain(['hp', 'hp'], ('em', q.T('n', lambda n: zprime(n))))
                q.exists('n'); q.split(); q.exact('hp'); q.apply('Z.dvd_refl')
                # n is not prime: it has a divisor d with 1 < d < n
                stB = q.T('n', lambda n: ~F([Z, lambda d: zlt(zzero, d), lambda d, h: D(d, n)], lambda d, h1, h2: Eq(d, zone) | Eq(d, n)))
                q.have('hB', stB, lambda r: (r.intro('hB'), r.apply('hp'), r.split(), r.exact('h1'), r.exact('hB')))
                Pd = q.T('n', lambda n: w.lam(Z, lambda d: zlt(zzero, d) >> (D(d, n) >> (Eq(d, zone) | Eq(d, n)))))
                q.obtain(['d', 'hd'], ('not_forall', Z, Pd, 'hB'))
                q.obtain(['hd1', 'hd2'], ('not_imp', q.T('d', lambda d: zlt(zzero, d)), q.T('d n', lambda d, n: D(d, n) >> (Eq(d, zone) | Eq(d, n))), 'hd'))
                q.obtain(['hd3', 'hd4'], ('not_imp', q.T('d n', lambda d, n: D(d, n)), q.T('d n', lambda d, n: Eq(d, zone) | Eq(d, n)), 'hd2'))
                q.obtain(['hd5', 'hd6'], ('not_or', q.T('d', lambda d: Eq(d, zone)), q.T('d n', lambda d, n: Eq(d, n)), 'hd4'))
                q.have('npos', q.T('n', lambda n: zlt(zzero, n)), lambda r: r.linarith())
                q.have('hle', q.T('d n', lambda d, n: zle(d, n)), lambda r: r.exact(('Z.le_of_dvd', 'd', 'n', 'npos', 'hd3')))
                q.have('hlt', q.T('d n', lambda d, n: zlt(d, n)), lambda r: (r.obtain(['hh', 'hh'], ('Z.lt_or_eq_of_le', 'd', 'n', 'hle')), r.exact('hh'), r.exfalso(), r.apply('hd6'), r.exact('hh')))
                q.have('hd0', q.T('d', lambda d: zle(zzero, d)), lambda r: r.linarith())
                q.have('hgt', q.T('d', lambda d: zlt(zone, d)),
                       lambda r: (r.have('h1d', r.T('d', lambda d: zle(zone, d)), lambda s_: s_.linarith()), r.obtain(['hh', 'hh'], ('Z.lt_or_eq_of_le', zone, 'd', 'h1d')),
                                  r.exact('hh'), r.exfalso(), r.apply('hd5'), r.linarith()))
                q.obtain(['pp', 'hpp'], ('IH', 'd', 'hd0', 'hlt', 'hgt'))
                q.cases('hpp', names=['hp1', 'hp2'])
                q.exists('pp'); q.split(); q.exact('hp1')
                q.exact(('Z.dvd_trans', 'pp', 'd', 'n', 'hp2', 'hd3'))
            p.have('H', Hst, H)
            p.have('h1', p.T('n0', lambda n: zle(zzero, n)), lambda q: q.linarith())
            p.exact(lambda q: w['Z.strong_induction'](motive, zzero, q.h('H'), q.h('n0'), q.h('h1'))(q.h('h0')))


STAGES = ('nt_induction', 'nt_dvd', 'nt_divmod', 'nt_classical', 'nt_gcd', 'nt_prime')


def build_nt(w=None, stages=STAGES):
    """the integers (int_lib), then every stage of this module in order"""
    if w is None: w = int_lib.build_int()
    for name in stages: globals()[name](w)
    return w


if __name__ == '__main__':
    import time
    t0 = time.time(); w = build_nt()
    print('%d globals in %.2f s' % (w.k.num_globals(), time.time() - t0))
