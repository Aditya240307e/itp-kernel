"""imo_lib.py -- olympiad problems, formalised over the integers and proved on top of `nt_lib` (every proof is a script, re-checked by the kernel).

    imo1959_p1          IMO 1959 Problem 1: the fraction (21n+4)/(14n+3) is irreducible for every natural number n
                        formal statement:  forall n >= 0,  Z.is_gcd (21n+4) (14n+3) 1   (1 is a greatest common divisor)
    sqrt2_irrational    sqrt 2 is irrational:  no integers p, q with q /= 0 and p^2 = 2 q^2
    imo2019_p1          IMO 2019 Problem 1: all functions f : Z -> Z with  f(2a) + 2 f(b) = f(f(a+b))  for all a, b are  f = 0  and  f(n) = 2n + c
                        formal statement:  forall f, (forall a b, f(2a) + 2 f(b) = f(f(a+b)))  <->  (forall n, f n = 0) or (exists c, forall n, f n = 2n + c)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys
import nt_lib
from nt_lib import _syms
from lnl_base import thm


def imo_1959_p1(w):
    s = _syms(w); Z, zle, zone = s.Z, s.zle, s.zone
    zgcd = w['Z.is_gcd']
    with w.nota.using(Z):
        lit = lambda n: w.nota.lit(n)

        @thm(w, 'imo1959_p1', [Z, lambda n: zle(s.zzero, n)], lambda n, h: zgcd(21 * n + 4, 14 * n + 3, zone))
        def _(p):
            p.intro('n', 'h'); p.apply('Z.is_gcd_one_of_coprime')
            p.exists(-lit(2)); p.exists(lit(3)); p.ring()


def sqrt2(w):
    s = _syms(w); Z, Eq, F, zle, zlt, zzero, zone, D = s.Z, s.Eq, s.F, s.zle, s.zlt, s.zzero, s.zone, w['Z.dvd']
    with w.nota.using(Z):
        lit = lambda n: w.nota.lit(n)
        def Psq(q): return zlt(zzero, q) >> F([Z], lambda p: ~Eq(p * p, 2 * (q * q)))

        @thm(w, 'Z.sqrt2_aux', [Z, lambda q: zle(zzero, q)], lambda q, h: Psq(q))
        def _(p):
            p.intro('q0', 'h0')
            with p.focus():
                motive = w.lam(Z, Psq)
                Hst = F([Z, lambda n: zle(zzero, n), lambda n, hn: F([Z, lambda m: zle(zzero, m), lambda m, h1: zlt(m, n)], lambda m, h1, h2: Psq(m))], lambda n, hn, ih: Psq(n))
            def H(q):
                # descent: p^2 = 2 q^2 makes p = 2 r even, then q^2 = 2 r^2 makes q = 2 s even, and r^2 = 2 s^2 with 0 < s < q
                q.intro('q', 'hq', 'IH', 'hpos', 'p', 'hpp')
                q.have('h2p', q.T('p', lambda p_: D(lit(2), p_ * p_)), lambda r: (r.exists(r.T('q', lambda q_: q_ * q_)), r.linear_combination((1, 'hpp'))))
                q.have('h3', q.T('p', lambda p_: D(lit(2), p_)), lambda r: r.exact(('Z.prime_dvd_sq', lit(2), w['Z.prime_two'], 'p', 'h2p')))
                q.cases('h3', names=['r', 'hr'])
                q.rw_at('hpp', 'hr')
                q.have('hqq', q.T('q r', lambda q_, r_: Eq(q_ * q_, 2 * (r_ * r_))), lambda t: t.linarith())
                q.have('h2q', q.T('q', lambda q_: D(lit(2), q_ * q_)), lambda t: (t.exists(t.T('r', lambda r_: r_ * r_)), t.linear_combination((1, 'hqq'))))
                q.have('h4', q.T('q', lambda q_: D(lit(2), q_)), lambda t: t.exact(('Z.prime_dvd_sq', lit(2), w['Z.prime_two'], 'q', 'h2q')))
                q.cases('h4', names=['s', 'hs'])
                q.have('spos', q.T('s', lambda s_: zlt(zzero, s_)), lambda t: t.linarith())
                q.have('slt', q.T('s q', lambda s_, q_: zlt(s_, q_)), lambda t: t.linarith())
                q.have('s0', q.T('s', lambda s_: zle(zzero, s_)), lambda t: t.linarith())
                q.rw_at('hqq', 'hs')
                q.have('hrs', q.T('r s', lambda r_, s_: Eq(r_ * r_, 2 * (s_ * s_))), lambda t: t.linarith())
                q.exact(('IH', 's', 's0', 'slt', 'spos', 'r', 'hrs'))
            p.have('H', Hst, H)
            p.exact(lambda q: w['Z.strong_induction'](motive, zzero, q.h('H'), q.h('q0'), q.h('h0')))

        @thm(w, 'sqrt2_irrational', [Z, Z, lambda p, q: ~Eq(q, zzero)], lambda p, q, h: ~Eq(p * p, 2 * (q * q)))
        def _(p):
            p.intro('p', 'q', 'h', 'hpp')
            p.obtain(['c', 'c'], ('Z.le_or_lt', zzero, 'q'))
            p.obtain(['c2', 'c2'], ('Z.lt_or_eq_of_le', zzero, 'q', 'c'))
            p.exact(('Z.sqrt2_aux', 'q', 'c', 'c2', 'p', 'hpp'))
            p.apply('h'); p.linarith()
            p.have('c3', p.T('q', lambda q_: zlt(zzero, -q_)), lambda r: r.linarith())
            p.have('c4', p.T('q', lambda q_: zle(zzero, -q_)), lambda r: r.linarith())
            p.have('hpp2', p.T('p q', lambda p_, q_: Eq(p_ * p_, 2 * ((-q_) * (-q_)))), lambda r: r.linarith())
            p.exact(('Z.sqrt2_aux', p.T('q', lambda q_: -q_), 'c4', 'c3', 'p', 'hpp2'))


def imo_2019_p1(w):
    s = _syms(w); Z, Eq, F, Ex, zzero, zone = s.Z, s.Eq, s.F, s.Ex, s.zzero, s.zone
    fT = w.arrow(Z, Z)
    with w.nota.using(Z):
        def cond(f): return F([Z, Z], lambda a, b: Eq(f(2 * a) + 2 * f(b), f(f(a + b))))
        def sol(f): return F([Z], lambda n: Eq(f(n), zzero)) | Ex(Z, w.lam(Z, lambda c: F([Z], lambda n: Eq(f(n), 2 * n + c))))

        @thm(w, 'imo2019_p1_fwd', [fT, lambda f: cond(f)], lambda f, H: sol(f))
        def _(p):
            p.intro('f', 'H')
            with p.focus():
                f = p.h('f')
                st1 = F([Z], lambda b: Eq(f(zzero) + 2 * f(b), f(f(b))))
                st2 = F([Z, Z], lambda a, b: Eq(f(2 * a) + 2 * f(b), 2 * f(a + b) + f(zzero)))
                st3 = F([Z], lambda a: Eq(f(2 * a), 2 * f(a) - f(zzero)))
                st4 = F([Z, Z], lambda a, b: Eq(f(a + b), f(a) + f(b) - f(zzero)))
                Pn = lambda n: Eq(f(n), n * (f(zone) - f(zzero)) + f(zzero)); motive = w.lam(Z, Pn)
                st5 = F([Z], lambda n: Pn(n)); Pn_zero = Pn(zzero)
                st_up = F([Z, lambda n: Pn(n)], lambda n, hn: Pn(n + zone))
                st_down = F([Z, lambda n: Pn(n)], lambda n, hn: Pn(n - zone))
            # f(0) + 2 f(b) = f(f(b))                       (a = 0)
            def s1(q):
                q.intro('b')
                q.have('e', ('H', zzero, 'b'))
                q.have('e1', Eq(2 * zzero, zzero), lambda r: r.ring())
                q.have('e2', q.T('b', lambda b_: Eq(zzero + b_, b_)), lambda r: r.ring())
                q.rw_at('e', 'e1'); q.rw_at('e', 'e2'); q.exact('e')
            p.have('h1', st1, s1)
            # f(2a) + 2 f(b) = 2 f(a+b) + f(0)
            def s2(q):
                q.intro('a', 'b')
                q.have('e1', ('H', 'a', 'b'))
                q.have('e2', ('h1', q.T('a b', lambda a, b: a + b)))
                q.linarith()
            p.have('h2', st2, s2)
            # f(2a) = 2 f(a) - f(0)                          (b = 0)
            def s3(q):
                q.intro('a')
                q.have('e', ('h2', 'a', zzero))
                q.have('e1', q.T('a', lambda a_: Eq(a_ + zzero, a_)), lambda r: r.ring())
                q.rw_at('e', 'e1'); q.linarith()
            p.have('h3', st3, s3)
            # f is additive up to the constant f(0)
            def s4(q):
                q.intro('a', 'b')
                q.have('e1', ('h2', 'a', 'b')); q.have('e2', ('h3', 'a'))
                q.linarith()
            p.have('h4', st4, s4)
            # f(n) = n (f(1) - f(0)) + f(0) for every integer n, by induction in both directions
            p.have('base', Pn_zero, lambda q: q.ring())
            def u5(q):
                q.intro('n', 'hn')
                q.have('e', ('h4', 'n', zone))
                q.linarith()
            p.have('up', st_up, u5)
            def d5(q):
                q.intro('n', 'hn')
                q.have('e', ('h4', q.T('n', lambda n_: n_ - zone), zone))
                q.have('e1', q.T('n', lambda n_: Eq((n_ - zone) + zone, n_)), lambda r: r.ring())
                q.rw_at('e', 'e1')
                q.linarith()
            p.have('down', st_down, d5)
            p.have('h5', st5, lambda q: (q.intro('z'), q.exact(lambda r: w['Z.int_induction'](motive, r.h('base'), r.h('up'), r.h('down'), r.h('z')))))
            # m = f(1) - f(0) and c = f(0) satisfy  2c = c m  and  2 f(1) = f(1) m,  hence  m (m - 2) = 0
            def hA(q):
                q.have('e1', ('h1', zzero))
                q.have('e2', ('h5', q.T('f', lambda f_: f_(zzero))))
                q.linarith()
            p.have('hA', p.T('f', lambda f_: Eq(2 * f_(zzero), f_(zzero) * (f_(zone) - f_(zzero)))), hA)
            def hB(q):
                q.have('e1', ('h1', zone))
                q.have('e2', ('h5', q.T('f', lambda f_: f_(zone))))
                q.linarith()
            p.have('hB', p.T('f', lambda f_: Eq(2 * f_(zone), f_(zone) * (f_(zone) - f_(zzero)))), hB)
            p.have('hC', p.T('f', lambda f_: Eq((f_(zone) - f_(zzero)) * ((f_(zone) - f_(zzero)) - 2), zzero)), lambda q: q.linarith())
            p.obtain(['hm', 'hm'], ('Z.mul_eq_zero', p.T('f', lambda f_: f_(zone) - f_(zzero)), p.T('f', lambda f_: (f_(zone) - f_(zzero)) - 2), 'hC'))
            # m = 0: then c = 0 and f = 0
            p.have('ha', p.T('f', lambda f_: Eq(f_(zzero), zzero)), lambda q: (q.rw_at('hA', 'hm'), q.linarith()))
            p.left(); p.intro('n')
            p.have('e', ('h5', 'n'))
            p.linear_combination((1, 'e'), (p.h('n'), 'hm'), (1, 'ha'))
            # m = 2: f(n) = 2 n + f(0)
            p.right(); p.exists(p.T('f', lambda f_: f_(zzero))); p.intro('n')
            p.have('e', ('h5', 'n'))
            p.linear_combination((1, 'e'), (p.h('n'), 'hm'))

        @thm(w, 'imo2019_p1_bwd', [fT, lambda f: sol(f)], lambda f, h: cond(f))
        def _(p):
            p.intro('f', 'h', 'a', 'b')
            p.cases('h', names=['hz', 'hz'])
            p.have('e1', ('hz', p.T('a', lambda a_: 2 * a_)))
            p.have('e2', ('hz', 'b'))
            p.have('e3', ('hz', p.T('f a b', lambda f_, a_, b_: f_(a_ + b_))))
            p.have('e4', ('hz', p.T('f a b', lambda f_, a_, b_: f_(f_(a_ + b_)))))
            p.linarith()
            p.cases('hz', names=['c', 'hc'])
            p.have('e1', ('hc', p.T('a', lambda a_: 2 * a_)))
            p.have('e2', ('hc', 'b'))
            p.have('e3', ('hc', p.T('f a b', lambda f_, a_, b_: f_(a_ + b_))))
            p.have('e4', ('hc', p.T('a b', lambda a_, b_: a_ + b_)))
            p.linarith()

        @thm(w, 'imo2019_p1', [fT], lambda f: w.Iff(cond(f), sol(f)))
        def _(p):
            p.intro('f'); p.split()
            p.intro('h'); p.exact(('imo2019_p1_fwd', 'f', 'h'))
            p.intro('h'); p.exact(('imo2019_p1_bwd', 'f', 'h'))


STAGES = ('imo_1959_p1', 'sqrt2', 'imo_2019_p1')


def build_imo(w=None, stages=STAGES):
    """the number-theory world (nt_lib), then the problems"""
    if w is None: w = nt_lib.build_nt()
    for name in stages: globals()[name](w)
    return w


if __name__ == '__main__':
    import time
    t0 = time.time(); w = build_imo()
    print('%d globals in %.2f s' % (w.k.num_globals(), time.time() - t0))
