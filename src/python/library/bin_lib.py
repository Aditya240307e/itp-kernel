"""bin_lib.py -- binary natural numbers (Coq's `positive` / `N`) with arithmetic that the kernel computes in time proportional to the number of BITS, and a bridge to the unary `Nat` that makes the whole
library available for them.

    Pos ::= XH            1
          | XO Pos        2p
          | XI Pos        2p + 1
    N   ::= N0 | Npos Pos

Why two layers. The unary numerals are the Peano axioms made literal (and the curriculum starts there), but a numeral of value k is a term of size k, so `2023 * 2023` is millions of kernel steps. The binary
operations (`Pos.add`, `Pos.mul`, ...) take a few hundred steps for numbers of any size a competition problem uses. They are defined here by plain structural recursion, with no mutual recursion and no
carry flags (the carry of an addition is a `succ` of the recursive sum); the price is a quadratic worst case in the bit length, which does not matter below a few hundred bits.

Nothing about the binary operations is trusted. Each is related to the unary one it implements by a THEOREM (`Pos.to_nat_add : to_nat (x + y) = to_nat x + to_nat y`, ...), and `to_nat` is shown to be injective, so every equation
between binary expressions follows from the unary equation (`transfer`). The induction principle of N (`N.ind`) is derived the same way. Operations that are not worth implementing in binary (subtraction, division, gcd)
are defined as the unary operation read back through the bridge; numeric facts about them are proved by CERTIFICATES (`a = b*q + r` with `r < b`, checked with the fast operations), which is also how a norm_num tactic works.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys
from author import World, AuthorError, Term, Notation, DispatchNotation
import lnl_base
from lnl_base import thm


class BinNotation(Notation):
    """operators and numerals of N: a numeral is its binary digits (the leading one is `XH`, then `XO`/`XI` for the other digits, most significant first)"""
    def __init__(self, w): self.w = w
    def _t(self, x): return self.lit(x) if isinstance(x, int) and not isinstance(x, bool) else x
    def add(self, a, b): return self.w['N.add'](self._t(a), self._t(b))
    def mul(self, a, b): return self.w['N.mul'](self._t(a), self._t(b))
    def pow(self, a, b): return self.w['N.pow'](self._t(a), self._t(b))
    def le(self, a, b): return self.w['N.le'](self._t(a), self._t(b))
    def lt(self, a, b): return self.w['N.lt'](self._t(a), self._t(b))
    def pos(self, n):
        t = self.w['XH']
        for bit in bin(n)[3:]: t = self.w['XI' if bit == '1' else 'XO'](t)
        return t
    def lit(self, n):
        if n < 0: raise AuthorError('N has no negative numerals')
        return self.w['N0'] if n == 0 else self.w['Npos'](self.pos(n))


# ---------------------------------------------------------------------------------------------------------------------------------------------- definitions
def declare_binary(w):
    TYPE = w.TYPE; Nat = w['Nat']; S, Z = w['Succ'], w['Zero']
    w.inductive('Pos', TYPE, [('XH', []), ('XO', [lambda *_: w['Pos']]), ('XI', [lambda *_: w['Pos']])])
    Pos = w['Pos']; XH, XO, XI = w['XH'], w['XO'], w['XI']
    w.inductive('N', TYPE, [('N0', []), ('Npos', [Pos])])
    N = w['N']; N0, Npos = w['N0'], w['Npos']

    # successor: 1 -> 2, 2p -> 2p+1, 2p+1 -> 2(p+1)
    w.recursive('Pos.succ', w.arrow(Pos, Pos),
                lambda succ: w.fun([Pos], lambda x: w.match(x, lambda _: Pos, [XO(XH), lambda p: XI(p), lambda p: XO(succ(p))])))
    psucc = w['Pos.succ']
    # addition, by recursion on the first argument; the carry of (2p+1) + (2q+1) = 2(p+q+1) is a successor of the recursive sum
    def add_body(add):
        def by_x(x, y):
            return w.match(x, lambda _: Pos, [
                psucc(y),
                lambda p: w.match(y, lambda _: Pos, [XI(p), lambda q: XO(add(p, q)), lambda q: XI(add(p, q))]),
                lambda p: w.match(y, lambda _: Pos, [XO(psucc(p)), lambda q: XI(add(p, q)), lambda q: XO(psucc(add(p, q)))])])
        return w.fun([Pos, Pos], by_x)
    w.recursive('Pos.add', w.forall_([Pos, Pos], lambda x, y: Pos), add_body)
    padd = w['Pos.add']
    # multiplication: 1*y = y, 2p*y = 2(p*y), (2p+1)*y = y + 2(p*y)
    w.recursive('Pos.mul', w.forall_([Pos, Pos], lambda x, y: Pos),
                lambda mul: w.fun([Pos, Pos], lambda x, y: w.match(x, lambda _: Pos, [y, lambda p: XO(mul(p, y)), lambda p: padd(y, XO(mul(p, y)))])))
    pmul = w['Pos.mul']
    w.define('Pos.sqr', w.arrow(Pos, Pos), w.fun([Pos], lambda z: pmul(z, z)))           # a function, so that the recursive result of pow is computed once and used twice
    psqr = w['Pos.sqr']
    w.recursive('Pos.pow', w.forall_([Pos, Pos], lambda x, y: Pos),
                lambda pw: w.fun([Pos, Pos], lambda x, y: w.match(y, lambda _: Pos, [x, lambda q: psqr(pw(x, q)), lambda q: pmul(x, psqr(pw(x, q)))])), on=1)

    # N: zero or a positive numeral. These are declared as recursive functions without a recursive call: the display normal form then computes them where the argument they match on is a constructor
    # (`Npos p + N0` is shown as `Npos p`), exactly as for `pred` on the unary numerals.
    w.recursive('N.succ', w.arrow(N, N), lambda f: w.fun([N], lambda n: w.match(n, lambda _: N, [Npos(XH), lambda p: Npos(psucc(p))])))
    w.recursive('N.add', w.forall_([N, N], lambda n, m: N),
                lambda f: w.fun([N, N], lambda n, m: w.match(n, lambda _: N, [m, lambda p: w.match(m, lambda _: N, [Npos(p), lambda q: Npos(padd(p, q))])])))
    w.recursive('N.mul', w.forall_([N, N], lambda n, m: N),
                lambda f: w.fun([N, N], lambda n, m: w.match(n, lambda _: N, [N0, lambda p: w.match(m, lambda _: N, [N0, lambda q: Npos(pmul(p, q))])])))
    w.recursive('N.pow', w.forall_([N, N], lambda a, b: N),
                lambda f: w.fun([N, N], lambda a, b: w.match(b, lambda _: N, [Npos(XH), lambda q: w.match(a, lambda _: N, [N0, lambda p: Npos(w['Pos.pow'](p, q))])])), on=1)
    w.recursive('N.double', w.arrow(N, N), lambda f: w.fun([N], lambda n: w.match(n, lambda _: N, [N0, lambda p: Npos(XO(p))])))          # 2n; used to read a unary sum a + a back as a binary numeral

    # the bridge to the unary numerals
    w.recursive('Pos.to_nat', w.arrow(Pos, Nat),
                lambda tn: w.fun([Pos], lambda x: w.match(x, lambda _: Nat, [S(Z), lambda p: w['add'](tn(p), tn(p)), lambda p: S(w['add'](tn(p), tn(p)))])))
    w.recursive('N.to_nat', w.arrow(N, Nat), lambda f: w.fun([N], lambda n: w.match(n, lambda _: Nat, [Z, lambda p: w['Pos.to_nat'](p)])))
    w.recursive('N.of_nat', w.arrow(Nat, N), lambda of: w.fun([Nat], lambda k: w.match(k, lambda _: N, [N0, lambda j: w['N.succ'](of(j))])))
    w.infix.update({'N.add': ('+', 6), 'N.mul': ('*', 7), 'N.pow': ('^', 8), 'Pos.add': ('+', 6), 'Pos.mul': ('*', 7), 'Pos.pow': ('^', 8)})


def pos_lemmas(w):
    """the operations on Pos against the unary numerals: to_nat is a homomorphism, and positive"""
    Nat, S, Z = w['Nat'], w['Succ'], w['Zero']; Pos = w['Pos']; Eq = w.Eq
    tn = w['Pos.to_nat']; psucc, padd, pmul, ppow, psqr = w['Pos.succ'], w['Pos.add'], w['Pos.mul'], w['Pos.pow'], w['Pos.sqr']
    XH, XO, XI = w['XH'], w['XO'], w['XI']

    @thm(w, 'Pos.to_nat_succ', [Pos], lambda x: Eq(tn(psucc(x)), S(tn(x))))
    def _(p):
        p.intro('x'); p.induct('x', names=['q', 'ih', 'q', 'ih'])
        p.refl()
        p.refl()
        p.rw('ih'); p.ring()

    @thm(w, 'Pos.to_nat_add', [Pos, Pos], lambda x, y: Eq(tn(padd(x, y)), tn(x) + tn(y)))
    def _(p):
        p.intro('x'); p.induct('x', names=['q', 'ih', 'q', 'ih'])
        p.exact('Pos.to_nat_succ')
        p.intro('y'); p.cases('y', names=['r', 'r'])
        p.ring()
        p.rw('ih'); p.ring()
        p.rw('ih'); p.ring()
        p.intro('y'); p.cases('y', names=['r', 'r'])
        p.rw('Pos.to_nat_succ'); p.ring()
        p.rw('ih'); p.ring()
        p.rw('Pos.to_nat_succ'); p.rw('ih'); p.ring()

    @thm(w, 'Pos.to_nat_mul', [Pos, Pos], lambda x, y: Eq(tn(pmul(x, y)), tn(x) * tn(y)))
    def _(p):
        p.intro('x'); p.induct('x', names=['q', 'ih', 'q', 'ih'])
        p.intro('y'); p.ring()
        p.intro('y'); p.rw('ih'); p.ring()
        p.intro('y'); p.rw('Pos.to_nat_add'); p.rw('ih'); p.ring()

    @thm(w, 'Pos.to_nat_sqr', [Pos], lambda x: Eq(tn(psqr(x)), tn(x) * tn(x)))
    def _(p): p.intro('x'); p.apply('Pos.to_nat_mul', 'x', 'x')

    @thm(w, 'Pos.to_nat_pow', [Pos, Pos], lambda x, y: Eq(tn(ppow(x, y)), tn(x) ** tn(y)))
    def _(p):
        p.intro('x', 'y'); p.induct('y', names=['q', 'ih', 'q', 'ih'])
        p.ring()
        p.rw('Pos.to_nat_sqr'); p.rw('ih'); p.rw('pow_add')
        p.rw('Pos.to_nat_mul'); p.rw('Pos.to_nat_sqr'); p.rw('ih'); p.rw('pow_add')

    @thm(w, 'Pos.to_nat_pos', [Pos], lambda x: w.Ex(Nat, lambda m: Eq(tn(x), S(m))))
    def _(p):
        p.intro('x'); p.induct('x', names=['q', 'ih', 'q', 'ih'])
        p.exists('Zero'); p.refl()
        p.cases('ih', names=['m', 'hm']); p.rw('hm'); p.exists((w['add'], 'm', (S, 'm'))); p.refl()
        p.exists((w['add'], (tn, 'q'), (tn, 'q'))); p.refl()


def n_lemmas(w):
    """N against the unary numerals: the bridge is a homomorphism and a bijection; Peano induction for N"""
    Nat, S, Z = w['Nat'], w['Succ'], w['Zero']; Pos, N = w['Pos'], w['N']; Eq = w.Eq
    tn, ntn, ofn = w['Pos.to_nat'], w['N.to_nat'], w['N.of_nat']
    N0, Npos, nsucc, ndouble = w['N0'], w['Npos'], w['N.succ'], w['N.double']
    nadd, nmul, npow = w['N.add'], w['N.mul'], w['N.pow']

    @thm(w, 'N.to_nat_succ', [N], lambda n: Eq(ntn(nsucc(n)), S(ntn(n))))
    def _(p):
        p.intro('n'); p.cases('n', names=['q'])
        p.refl()
        p.apply('Pos.to_nat_succ')

    @thm(w, 'N.to_nat_add', [N, N], lambda a, b: Eq(ntn(nadd(a, b)), ntn(a) + ntn(b)))
    def _(p):
        p.intro('a', 'b'); p.cases('a', names=['q'])
        p.refl()
        p.cases('b', names=['r'])
        p.ring()
        p.apply('Pos.to_nat_add')

    @thm(w, 'N.to_nat_mul', [N, N], lambda a, b: Eq(ntn(nmul(a, b)), ntn(a) * ntn(b)))
    def _(p):
        p.intro('a', 'b'); p.cases('a', names=['q'])
        p.refl()
        p.cases('b', names=['r'])
        p.ring()
        p.apply('Pos.to_nat_mul')

    @thm(w, 'N.to_nat_pow', [N, N], lambda a, b: Eq(ntn(npow(a, b)), ntn(a) ** ntn(b)))
    def _(p):
        p.intro('a', 'b'); p.cases('b', names=['q'])
        p.refl()
        p.cases('a', names=['r'])
        p.obtain(['m', 'hm'], ('Pos.to_nat_pos', 'q')); p.rw('hm')
        p.apply('Pos.to_nat_pow')

    # the bridge is a bijection
    @thm(w, 'N.to_nat_of_nat', [Nat], lambda k: Eq(ntn(ofn(k)), k))
    def _(p):
        p.intro('k'); p.induct('k', names=['j', 'ih'])
        p.refl()
        p.rw('N.to_nat_succ'); p.rw('ih')

    @thm(w, 'N.double_succ', [N], lambda n: Eq(ndouble(nsucc(n)), nsucc(nsucc(ndouble(n)))))
    def _(p):
        p.intro('n'); p.cases('n', names=['q'])
        p.refl()
        p.refl()

    @thm(w, 'N.of_nat_double', [Nat], lambda a: Eq(ofn(a + a), ndouble(ofn(a))))
    def _(p):
        p.intro('a'); p.induct('a', names=['j', 'ih'])
        p.refl()
        p.rw('add_succ'); p.rw('ih'); p.rw('N.double_succ')

    @thm(w, 'Pos.of_nat_to_nat', [Pos], lambda x: Eq(ofn(tn(x)), Npos(x)))
    def _(p):
        p.intro('x'); p.induct('x', names=['q', 'ih', 'q', 'ih'])
        p.refl()
        p.rw('N.of_nat_double'); p.rw('ih')
        p.rw('N.of_nat_double'); p.rw('ih')

    @thm(w, 'N.of_nat_to_nat', [N], lambda n: Eq(ofn(ntn(n)), n))
    def _(p):
        p.intro('n'); p.cases('n', names=['q'])
        p.refl()
        p.apply('Pos.of_nat_to_nat')

    @thm(w, 'N.to_nat_inj', [N, N, lambda a, b: Eq(ntn(a), ntn(b))], lambda a, b, h: Eq(a, b))
    def _(p):
        p.intro('a', 'b', 'h')
        p.rw('N.of_nat_to_nat', 'a', rev=True); p.rw('h'); p.apply('N.of_nat_to_nat')

    # Peano induction
    @thm(w, 'N.ind', [w.arrow(N, w.PROP), lambda P: P(N0), lambda P, h0: w.forall_([N, lambda n: P(n)], lambda n, hn: P(nsucc(n)))], lambda P, h0, hs: w.forall_([N], lambda n: P(n)))
    def _(p):
        p.intro('P', 'h0', 'hs', 'n')
        p.have('key', lambda q: w.forall_([Nat], lambda k: q.h('P')(ofn(k))),
               lambda q: (q.intro('k'), q.induct('k', names=['j', 'ih']), q.exact('h0'), q.apply('hs'), q.exact('ih')))
        p.rw('N.of_nat_to_nat', 'n', rev=True)
        p.apply('key')


# ---------------------------------------------------------------------------------------------------------------------------------------------- transfer
HOMS = ('N.to_nat_succ', 'N.to_nat_add', 'N.to_nat_mul', 'N.to_nat_pow', 'N.to_nat_of_nat')


def _no_instance(e): return 'no instance of the pattern' in str(e)


def hom_goal(p):
    """rewrite every `to_nat (a op b)` in the first goal by the unary operation on `to_nat a`, `to_nat b`"""
    changed = True
    while changed:
        changed = False
        for lem in HOMS:
            try: p.rw(lem, close=False)
            except AuthorError as e:
                if not _no_instance(e): raise
                continue
            changed = True
    return p


def hom_at(p, h, new=None):
    """the same inside hypothesis `h` (the rewritten hypothesis is added under the name `new`, default the same)"""
    cur = h; first = True
    changed = True
    while changed:
        changed = False
        for lem in HOMS:
            try: p.rw_at(cur, lem, new=new or h)
            except AuthorError as e:
                if not _no_instance(e): raise
                continue
            cur = new or h; changed = True
    return p


def to_nat_hyp(p, h, new=None):
    """from a hypothesis about N (an equation, `<=` or `<`) the same fact about the unary numerals `to_nat a`, with the operations pushed through: h : a = b gives new : to_nat a = to_nat b, h : a <= b gives
    new : to_nat a <= to_nat b"""
    w = p.w; new = new or h + '_nat'
    with p.focus():
        hy = p.h(h); ty = w.simp(w.type_of(hy))
        if w.is_app_of(ty, 'Eq', 3):
            A, a, b = w.eq_parts(ty)
            proof = w.app(w['f_equal'], A, w['Nat'], w['N.to_nat'], a, b, hy)
        elif w.is_app_of(ty, 'N.le', 2):
            _, args = w.spine(ty); proof = w.app(w['N.le_to_nat'], args[0], args[1], hy)
        elif w.is_app_of(ty, 'N.lt', 2):
            _, args = w.spine(ty); proof = w.app(w['N.lt_to_nat'], args[0], args[1], hy)
        else: raise AuthorError('to_nat_hyp: %s is not an equation, <= or < on N' % h)
    p.have(new, proof)
    return hom_at(p, new)


def transfer(p):
    """first goal: an equation, `<=` or `<` between expressions of N; it becomes the same statement about the unary numerals (the operations pushed through `to_nat`), where the unary library and `ring` apply"""
    w = p.w; g = p.goal()
    with p.focus(): ty = w.simp(g.target)
    if w.is_app_of(ty, 'Eq', 3): p.apply('N.to_nat_inj')
    elif w.is_app_of(ty, 'N.le', 2): p.apply('N.of_le')
    elif w.is_app_of(ty, 'N.lt', 2): p.apply('N.of_lt')
    else: raise AuthorError('transfer: the goal is not an equation, <= or < on N')
    return hom_goal(p)


def n_library(w):
    """the laws of N, each proved by moving the statement to the unary numerals and applying the unary law (or `ring`)"""
    N, Nat, Eq, Le = w['N'], w['Nat'], w.Eq, w['Le']; S, Z = w['Succ'], w['Zero']
    ntn, ofn, nsucc, N0, Npos, XH = w['N.to_nat'], w['N.of_nat'], w['N.succ'], w['N0'], w['Npos'], w['XH']
    one = Npos(XH)
    F = w['False']

    # ---- order on N: a <= b iff some k has a + k = b (a certificate, which the fast operations check for numerals of any size)
    w.define('N.le', w.forall_([N, N], lambda a, b: w.PROP), w.fun([N, N], lambda a, b: w['Ex'](N, w.lam(N, lambda k: Eq(a + k, b)))))
    w.define('N.lt', w.forall_([N, N], lambda a, b: w.PROP), w.fun([N, N], lambda a, b: w['N.le'](nsucc(a), b)))
    w.infix.update({'N.le': ('<=', 4), 'N.lt': ('<', 4)})

    # ---- the equation laws
    def eq_thm(name, nvars, body, how='ring'):
        names = 'abcdefgh'[:nvars]
        @thm(w, name, [N] * nvars, body)
        def _(p):
            p.intro(*names)
            if how == 'refl': p.refl(); return
            transfer(p)
            if how == 'ring': p.ring()
            else: p.apply(how)
    with w.nota.using(N):
        eq_thm('N.add_zero', 1, lambda a: Eq(a + N0, a))
        eq_thm('N.zero_add', 1, lambda a: Eq(N0 + a, a), 'refl')
        eq_thm('N.add_succ', 2, lambda a, b: Eq(a + nsucc(b), nsucc(a + b)))
        eq_thm('N.succ_add', 2, lambda a, b: Eq(nsucc(a) + b, nsucc(a + b)))
        eq_thm('N.add_comm', 2, lambda a, b: Eq(a + b, b + a))
        eq_thm('N.add_assoc', 3, lambda a, b, c: Eq((a + b) + c, a + (b + c)))
        eq_thm('N.add_left_comm', 3, lambda a, b, c: Eq(a + (b + c), b + (a + c)))
        eq_thm('N.add_right_comm', 3, lambda a, b, c: Eq((a + b) + c, (a + c) + b))
        eq_thm('N.mul_zero', 1, lambda a: Eq(a * N0, N0))
        eq_thm('N.zero_mul', 1, lambda a: Eq(N0 * a, N0), 'refl')
        eq_thm('N.mul_one', 1, lambda a: Eq(a * one, a))
        eq_thm('N.one_mul', 1, lambda a: Eq(one * a, a))
        eq_thm('N.mul_succ', 2, lambda a, b: Eq(a * nsucc(b), a + a * b))
        eq_thm('N.succ_mul', 2, lambda a, b: Eq(nsucc(a) * b, b + a * b))
        eq_thm('N.mul_comm', 2, lambda a, b: Eq(a * b, b * a))
        eq_thm('N.mul_assoc', 3, lambda a, b, c: Eq((a * b) * c, a * (b * c)))
        eq_thm('N.mul_left_comm', 3, lambda a, b, c: Eq(a * (b * c), b * (a * c)))
        eq_thm('N.mul_right_comm', 3, lambda a, b, c: Eq((a * b) * c, (a * c) * b))
        eq_thm('N.mul_add', 3, lambda a, b, c: Eq(a * (b + c), a * b + a * c))
        eq_thm('N.add_mul', 3, lambda a, b, c: Eq((a + b) * c, a * c + b * c))
        eq_thm('N.succ_eq_add_one', 1, lambda a: Eq(nsucc(a), a + one))
        eq_thm('N.pow_zero', 1, lambda a: Eq(a ** N0, one), 'refl')
        eq_thm('N.pow_succ', 2, lambda a, b: Eq(a ** nsucc(b), a * a ** b), 'pow_succ')
        eq_thm('N.pow_one', 1, lambda a: Eq(a ** one, a), 'pow_one')
        eq_thm('N.one_pow', 1, lambda a: Eq(one ** a, one), 'one_pow')
        eq_thm('N.pow_add', 3, lambda a, b, c: Eq(a ** (b + c), a ** b * a ** c), 'pow_add')
        eq_thm('N.mul_pow', 3, lambda a, b, c: Eq((a * b) ** c, a ** c * b ** c), 'mul_pow')
        eq_thm('N.pow_mul', 3, lambda a, b, c: Eq(a ** (b * c), (a ** b) ** c), 'pow_mul')

    # ---- laws with hypotheses
    @thm(w, 'N.succ_inj', [N, N, lambda a, b: Eq(nsucc(a), nsucc(b))], lambda a, b, h: Eq(a, b))
    def _(p):
        p.intro('a', 'b', 'h'); p.apply('N.to_nat_inj'); to_nat_hyp(p, 'h', 'h2')
        p.apply('succ_inj'); p.exact('h2')

    @thm(w, 'N.succ_ne_zero', [N, lambda a: Eq(nsucc(a), N0)], lambda a, h: F)
    def _(p):
        p.intro('a', 'h'); to_nat_hyp(p, 'h', 'h2')
        p.apply('succ_ne_zero', (ntn, 'a')); p.exact('h2')

    @thm(w, 'N.zero_ne_succ', [N, lambda a: Eq(N0, nsucc(a))], lambda a, h: F)
    def _(p):
        p.intro('a', 'h'); to_nat_hyp(p, 'h', 'h2')
        p.apply('zero_ne_succ', (ntn, 'a')); p.exact('h2')

    @thm(w, 'N.add_left_cancel', [N, N, N, lambda a, b, c: Eq(w['N.add'](a, b), w['N.add'](a, c))], lambda a, b, c, h: Eq(b, c))
    def _(p):
        p.intro('a', 'b', 'c', 'h'); p.apply('N.to_nat_inj'); to_nat_hyp(p, 'h', 'h2')
        p.apply('add_left_cancel', (ntn, 'a')); p.exact('h2')

    @thm(w, 'N.add_right_cancel', [N, N, N, lambda a, b, c: Eq(w['N.add'](a, c), w['N.add'](b, c))], lambda a, b, c, h: Eq(a, b))
    def _(p):
        p.intro('a', 'b', 'c', 'h'); p.apply('N.to_nat_inj'); to_nat_hyp(p, 'h', 'h2')
        p.apply('add_right_cancel', (ntn, 'a'), (ntn, 'c'), (ntn, 'b')); p.exact('h2')

    @thm(w, 'N.add_eq_zero', [N, N, lambda a, b: Eq(w['N.add'](a, b), N0)], lambda a, b, h: Eq(a, N0) & Eq(b, N0))
    def _(p):
        p.intro('a', 'b', 'h'); to_nat_hyp(p, 'h', 'h2')
        p.obtain(['h3', 'h4'], ('add_eq_zero', (ntn, 'a'), (ntn, 'b'), 'h2'))
        p.split()
        p.apply('N.to_nat_inj'); p.exact('h3')
        p.apply('N.to_nat_inj'); p.exact('h4')

    @thm(w, 'N.mul_eq_zero', [N, N, lambda a, b: Eq(w['N.mul'](a, b), N0)], lambda a, b, h: Eq(a, N0) | Eq(b, N0))
    def _(p):
        p.intro('a', 'b', 'h'); to_nat_hyp(p, 'h', 'h2')
        p.obtain(['e', 'e'], ('mul_eq_zero', (ntn, 'a'), (ntn, 'b'), 'h2'))
        p.left(); p.apply('N.to_nat_inj'); p.exact('e')
        p.right(); p.apply('N.to_nat_inj'); p.exact('e')

    @thm(w, 'N.mul_left_cancel', [N, N, N, lambda a, b, c: ~Eq(a, N0), lambda a, b, c, h1: Eq(w['N.mul'](a, b), w['N.mul'](a, c))], lambda a, b, c, h1, h2: Eq(b, c))
    def _(p):
        p.intro('a', 'b', 'c', 'h1', 'h2'); p.apply('N.to_nat_inj'); to_nat_hyp(p, 'h2', 'h3')
        p.apply('mul_left_cancel', (ntn, 'a'))
        p.intro('e'); p.apply('h1'); p.apply('N.to_nat_inj'); p.exact('e')
        p.exact('h3')

    @thm(w, 'N.eq_dec', [N, N], lambda a, b: Eq(a, b) | ~Eq(a, b))
    def _(p):
        p.intro('a', 'b'); p.obtain(['e', 'e'], ('eq_dec', (ntn, 'a'), (ntn, 'b')))
        p.left(); p.apply('N.to_nat_inj'); p.exact('e')
        p.right(); p.intro('h'); p.apply('e'); p.rw('h')

    # ---- order
    @thm(w, 'N.le_to_nat', [N, N, lambda a, b: w['N.le'](a, b)], lambda a, b, h: Le(ntn(a), ntn(b)))
    def _(p):
        p.intro('a', 'b', 'h'); p.cases('h', names=['k', 'hk'])
        p.apply('exists_le'); p.exists((ntn, 'k')); to_nat_hyp(p, 'hk', 'hk2'); p.exact('hk2')

    @thm(w, 'N.of_le', [N, N, lambda a, b: Le(ntn(a), ntn(b))], lambda a, b, h: w['N.le'](a, b))
    def _(p):
        p.intro('a', 'b', 'h'); p.obtain(['m', 'hm'], ('le_exists', (ntn, 'a'), (ntn, 'b'), 'h'))
        p.exists((ofn, 'm')); p.apply('N.to_nat_inj'); hom_goal(p); p.exact('hm')

    @thm(w, 'N.lt_to_nat', [N, N, lambda a, b: w['N.lt'](a, b)], lambda a, b, h: w['Lt'](ntn(a), ntn(b)))
    def _(p):
        p.intro('a', 'b', 'h'); p.have('h2', (w['N.le_to_nat'], (nsucc, 'a'), 'b', 'h'))
        hom_at(p, 'h2'); p.exact('h2')

    @thm(w, 'N.of_lt', [N, N, lambda a, b: w['Lt'](ntn(a), ntn(b))], lambda a, b, h: w['N.lt'](a, b))
    def _(p):
        p.intro('a', 'b', 'h'); p.apply('N.of_le'); hom_goal(p); p.exact('h')


def n_order(w):
    """the order of N: every law is the unary law, reached by `transfer` (the statement) and `to_nat_hyp` (the hypotheses)"""
    N, Nat, Eq, Le = w['N'], w['Nat'], w.Eq, w['Le']; S, Z = w['Succ'], w['Zero']
    ntn, ofn, nsucc, N0 = w['N.to_nat'], w['N.of_nat'], w['N.succ'], w['N0']
    nle, nlt = w['N.le'], w['N.lt']
    nadd, nmul = w['N.add'], w['N.mul']
    tnn = lambda x: (ntn, x)

    def order_thm(name, doms, concl, lemma, hyps, args=()):
        """doms: the variables (all N) followed by hypotheses; the proof is: transfer the goal, transfer every hypothesis `h<i>`, apply the unary `lemma` (to `args`) and close its premises by the transferred hypotheses"""
        nv = sum(1 for d in doms if d == N)
        names = 'abcdefgh'[:nv]
        @thm(w, name, doms, concl)
        def _(p):
            p.intro(*names, *['h%d' % i for i in range(1, len(doms) - nv + 1)])
            transfer(p)
            for i in range(1, len(doms) - nv + 1): to_nat_hyp(p, 'h%d' % i)
            p.apply(lemma, *args)
            for i in range(1, len(doms) - nv + 1): p.exact('h%d_nat' % i)

    @thm(w, 'N.le_refl', [N], lambda a: nle(a, a))
    def _(p): p.intro('a'); p.exists(N0); p.apply('N.add_zero')

    @thm(w, 'N.zero_le', [N], lambda a: nle(N0, a))
    def _(p): p.intro('a'); p.exists('a'); p.refl()

    @thm(w, 'N.le_succ', [N], lambda a: nle(a, nsucc(a)))
    def _(p): p.intro('a'); transfer(p); p.apply('le_succ')

    @thm(w, 'N.le_add_right', [N, N], lambda a, k: nle(a, nadd(a, k)))
    def _(p): p.intro('a', 'k'); transfer(p); p.apply('le_add_right')

    order_thm('N.le_trans', [N, N, N, lambda a, b, c: nle(a, b), lambda a, b, c, h1: nle(b, c)], lambda a, b, c, h1, h2: nle(a, c), 'le_trans', 2, (tnn('a'), tnn('b'), tnn('c')))
    order_thm('N.le_antisymm', [N, N, lambda a, b: nle(a, b), lambda a, b, h1: nle(b, a)], lambda a, b, h1, h2: Eq(a, b), 'le_antisymm', 2, (tnn('a'), tnn('b')))
    order_thm('N.succ_le_succ', [N, N, lambda a, b: nle(a, b)], lambda a, b, h1: nle(nsucc(a), nsucc(b)), 'succ_le_succ', 1, (tnn('a'), tnn('b')))
    order_thm('N.le_of_succ_le_succ', [N, N, lambda a, b: nle(nsucc(a), nsucc(b))], lambda a, b, h1: nle(a, b), 'le_of_succ_le_succ', 1, (tnn('a'), tnn('b')))
    order_thm('N.add_le_add_left', [N, N, N, lambda a, b, c: nle(b, c)], lambda a, b, c, h1: nle(nadd(a, b), nadd(a, c)), 'add_le_add_left', 1, (tnn('a'), tnn('b'), tnn('c')))
    order_thm('N.add_le_add_right', [N, N, N, lambda a, b, c: nle(a, b)], lambda a, b, c, h1: nle(nadd(a, c), nadd(b, c)), 'add_le_add_right', 1, (tnn('a'), tnn('b'), tnn('c')))
    order_thm('N.add_le_add', [N, N, N, N, lambda a, b, c, d: nle(a, b), lambda a, b, c, d, h1: nle(c, d)], lambda a, b, c, d, h1, h2: nle(nadd(a, c), nadd(b, d)), 'add_le_add', 2,
              (tnn('a'), tnn('b'), tnn('c'), tnn('d')))
    order_thm('N.mul_le_mul_left', [N, N, N, lambda a, b, c: nle(b, c)], lambda a, b, c, h1: nle(nmul(a, b), nmul(a, c)), 'mul_le_mul_left', 1, (tnn('a'), tnn('b'), tnn('c')))
    order_thm('N.mul_le_mul_right', [N, N, N, lambda a, b, c: nle(a, b)], lambda a, b, c, h1: nle(nmul(a, c), nmul(b, c)), 'mul_le_mul_right', 1, (tnn('a'), tnn('b'), tnn('c')))
    order_thm('N.mul_le_mul', [N, N, N, N, lambda a, b, c, d: nle(a, b), lambda a, b, c, d, h1: nle(c, d)], lambda a, b, c, d, h1, h2: nle(nmul(a, c), nmul(b, d)), 'mul_le_mul', 2,
              (tnn('a'), tnn('b'), tnn('c'), tnn('d')))

    @thm(w, 'N.le_total', [N, N], lambda a, b: nle(a, b) | nle(b, a))
    def _(p):
        p.intro('a', 'b'); p.obtain(['e', 'e'], ('le_total', tnn('a'), tnn('b')))
        p.left(); p.apply('N.of_le'); p.exact('e')
        p.right(); p.apply('N.of_le'); p.exact('e')

    @thm(w, 'N.lt_irrefl', [N], lambda a: ~nlt(a, a))
    def _(p):
        p.intro('a', 'h'); to_nat_hyp(p, 'h')
        p.apply('lt_irrefl', tnn('a')); p.exact('h_nat')

    @thm(w, 'N.lt_succ_self', [N], lambda a: nlt(a, nsucc(a)))
    def _(p): p.intro('a'); p.apply('N.le_refl')

    @thm(w, 'N.lt_of_succ_le', [N, N, lambda a, b: nle(nsucc(a), b)], lambda a, b, h: nlt(a, b))
    def _(p): p.intro('a', 'b', 'h'); p.exact('h')

    @thm(w, 'N.succ_le_of_lt', [N, N, lambda a, b: nlt(a, b)], lambda a, b, h: nle(nsucc(a), b))
    def _(p): p.intro('a', 'b', 'h'); p.exact('h')

    @thm(w, 'N.lt_of_lt_of_le', [N, N, N, lambda a, b, c: nlt(a, b), lambda a, b, c, h1: nle(b, c)], lambda a, b, c, h1, h2: nlt(a, c))
    def _(p): p.intro('a', 'b', 'c', 'h1', 'h2'); p.apply('N.le_trans', (nsucc, 'a'), 'b', 'c'); p.exact('h1'); p.exact('h2')

    @thm(w, 'N.lt_of_le_of_lt', [N, N, N, lambda a, b, c: nle(a, b), lambda a, b, c, h1: nlt(b, c)], lambda a, b, c, h1, h2: nlt(a, c))
    def _(p): p.intro('a', 'b', 'c', 'h1', 'h2'); p.apply('N.le_trans', (nsucc, 'a'), (nsucc, 'b'), 'c'); p.apply('N.succ_le_succ'); p.exact('h1'); p.exact('h2')

    @thm(w, 'N.lt_trans', [N, N, N, lambda a, b, c: nlt(a, b), lambda a, b, c, h1: nlt(b, c)], lambda a, b, c, h1, h2: nlt(a, c))
    def _(p): p.intro('a', 'b', 'c', 'h1', 'h2'); p.apply('N.lt_of_lt_of_le', 'a', 'b', 'c'); p.exact('h1'); p.apply('N.le_trans', 'b', (nsucc, 'b'), 'c'); p.apply('N.le_succ'); p.exact('h2')


def build_bin_defs():
    import nat_lib
    w = nat_lib.build_nat()
    declare_binary(w)
    pos_lemmas(w); n_lemmas(w)
    w.nota.register(w['N'], BinNotation(w))
    n_library(w); n_order(w)
    import ring_lib
    ring_lib.ring_lemma_set(w, ring_lib.n_desc(w))
    return w


if __name__ == '__main__':
    import time
    t0 = time.time(); w = build_bin_defs()
    print('binary definitions: %d globals in %.2f s' % (w.k.num_globals(), time.time() - t0))
