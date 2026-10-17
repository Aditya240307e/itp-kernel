"""ring.py -- the `ring` tactic for commutative semirings and rings (the unary naturals `Nat`, the binary naturals `N`, the integers `Z`), proof-producing and checked by the kernel.

NOTHING HERE IS TRUSTED. `ring` normalises both sides of an equation to a canonical sum of monomials and, at every step, builds the proof term that justifies it: an application of one of the
`ring_*` lemmas of the number system (ring_lib.py), each a theorem proved from the semiring laws. The finished term is handed to the kernel like any other proof, so a bug in this file can make `ring`
fail or produce a term the kernel rejects, never make a false equation provable.

Normal form (the same shape Mathlib's `ring` uses, so that every merge is one lemma application):

    polynomial   m1 + (m2 + (... + (mk + 0)))            monomials in strictly decreasing key order, coefficients >= 1
    monomial     x1 ^ e1 * (x2 ^ e2 * (... * (xj ^ ej * c)))   atoms in increasing index order, exponents >= 1 (numerals), the coefficient c a numeral last

The ORDER on monomials (`cmp_vars`) is the lexicographic order on exponent vectors with the first atom most significant; it is a total order that is compatible with multiplication, which is what lets
`scale` (monomial times polynomial) keep a polynomial sorted without merging. Numerals are the number system's own canonical numerals (`S (S Zero)` for Nat, the binary digits for N); the arithmetic on
them (`3 + 4 = 7`, `12 * 12 = 144`) is checked by the kernel's evaluation, which for N takes time proportional to the number of bits, so coefficients of any size a competition problem has cost nothing.
Powers by a numeral use repeated squaring (`ring_pow_split`), so `(a + b) ^ 20` takes five multiplications, not nineteen.

Statement-carrying proofs: `Pf(proof, lhs, rhs)` knows what it proves, so each combinator can build the next lemma instance without asking the kernel for a type.

A number system is an INSTANCE (`SemiringNat`, `SemiringN`, `SemiringZ`): its terms for zero, one, `+`, `*`, `^`, its numerals and how to read them back, and the prefix of its `ring_*` lemmas. `instance_for(w, A)`
finds the one for the type A of an equation.

Rings: the integers add negation and subtraction (rewritten as multiplication by the numeral -1), coefficients of either sign, and monomials that cancel (their coefficients add up to 0 and the monomial
leaves the sum). The arithmetic on integer numerals is no longer `refl` -- `Z.mk 3 5` and `Z.mk 0 2` are different terms of the quotient -- so `SemiringZ.num_eq` computes the pair of naturals each side
evaluates to (the kernel's own computation, mirrored on Python integers) and proves the equation with `Z.mk_eq` and a `refl` of N, both of which the kernel checks.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from author import AuthorError, Term, NodeCache


class Pf:
    """a proof term together with the equation it proves: proof : lhs = rhs"""
    __slots__ = ('proof', 'lhs', 'rhs')
    def __init__(self, proof, lhs, rhs): self.proof = proof; self.lhs = lhs; self.rhs = rhs


class Mono:
    """a monomial: vars = ((atom index, exponent), ...) in increasing atom index, coeff >= 1"""
    __slots__ = ('vars', 'coeff')
    def __init__(self, vars_, coeff): self.vars = vars_; self.coeff = coeff
    def key(self): return (self.vars, self.coeff)


def cmp_vars(a, b):
    """compare the variable parts of two monomials: 1 if a comes first (a is the larger), -1 if b does, 0 if they are the same"""
    for (i, e), (j, f) in zip(a, b):
        if i != j: return 1 if i < j else -1
        if e != f: return 1 if e > f else -1
    if len(a) == len(b): return 0
    return 1 if len(a) > len(b) else -1


class NF:
    """a normalised polynomial: a tuple of Monos, with its kernel term"""
    __slots__ = ('monos', 'term')
    def __init__(self, monos, term): self.monos = monos; self.term = term


# ---------------------------------------------------------------------------------------------------------------------------------------------- number systems
class Semiring:
    """what `Ring` needs to know about a number system. Subclasses set `ty` (the carrier), `ety` (the type of exponents), `zero`, `one`, `pfx` (the prefix of the `ring_*` lemmas), `ops` (global id -> 'add' | 'mul' |
    'pow' | 'succ'), the number system's own `zero_add` and `add_zero` laws, and provide the term builders add / mul / pow / eadd, the numerals lit / elit and their readers numeral / enumeral."""
    pfx = 'ring_'
    def num_eq(self, lhs, rhs):
        """a proof of  lhs = rhs  for two closed numeral expressions of the carrier with the same value: the kernel checks it by evaluation"""
        return Pf(self.w['refl'](self.ty, rhs), lhs, rhs)
    def enum_eq(self, lhs, rhs):
        """the same for the exponent type"""
        return Pf(self.w['refl'](self.ety, rhs), lhs, rhs)


class SemiringNat(Semiring):
    """the unary natural numbers with the library's names"""
    pfx = 'ring_'
    def __init__(self, w):
        self.w = w; self.ty = self.ety = w['Nat']; self.zero = w['Zero']; self._succ = w['Succ']
        self._add, self._mul, self._pow = w['add'], w['mul'], w['pow']
        self.ops = {w.gid['add']: 'add', w.gid['mul']: 'mul', w.gid['pow']: 'pow', w.gid['Succ']: 'succ'}
        self._zero_id, self._succ_id = w.gid['Zero'], w.gid['Succ']
        self.zero_add, self.add_zero = w['zero_add'], w['add_zero']
        self.lit_cache = NodeCache(w); self.num_cache = NodeCache(w)
        self.one = self.lit(1)
    def add(self, a, b): return self._add(a, b)
    def mul(self, a, b): return self._mul(a, b)
    def pow(self, a, e): return self._pow(a, e)
    eadd = add
    def lit(self, k):
        t = self.lit_cache.get(k)
        if t is None:
            if k < 0: raise AuthorError('the natural numbers have no negative numerals (%d)' % k)
            t = self.zero if k == 0 else self._succ(self.lit(k - 1)) if k < 200 else self._lit_big(k)
            self.lit_cache[k] = t
        return t
    elit = lit
    def _lit_big(self, k):
        t = self.lit(199)
        for _ in range(k - 199): t = self._succ(t)
        return t
    def numeral(self, t):
        """the value k when t is the numeral S^k Zero, else None"""
        hit = self.num_cache.get(t)
        if hit is not None: return None if hit < 0 else hit
        view = self.w.k.view; zero = self._zero_id; succ = self._succ_id
        k = 0; n = t.n
        while True:
            v = view(n)
            if v[0] == 'const' and v[1] == zero: break
            if v[0] == 'app':
                f = view(v[1])
                if f[0] == 'const' and f[1] == succ: n = v[2]; k += 1; continue
            self.num_cache[t] = -1; return None
        self.num_cache[t] = k
        return k
    enumeral = numeral


class SemiringN(Semiring):
    """the binary natural numbers (bin_lib.py): a numeral is N0 or `Npos` of its binary digits"""
    pfx = 'N.ring_'
    def __init__(self, w):
        self.w = w; self.ty = self.ety = w['N']; self.zero = w['N0']
        self._add, self._mul, self._pow, self._succ = w['N.add'], w['N.mul'], w['N.pow'], w['N.succ']
        self._npos, self._xh, self._xo, self._xi = w['Npos'], w['XH'], w['XO'], w['XI']
        self._ids = (w.gid['N0'], w.gid['Npos'], w.gid['XH'], w.gid['XO'], w.gid['XI'])
        self.ops = {w.gid['N.add']: 'add', w.gid['N.mul']: 'mul', w.gid['N.pow']: 'pow', w.gid['N.succ']: 'succ'}
        self.zero_add, self.add_zero = w['N.zero_add'], w['N.add_zero']
        self.lit_cache = NodeCache(w); self.num_cache = NodeCache(w)
        self.one = self.lit(1)
    def add(self, a, b): return self._add(a, b)
    def mul(self, a, b): return self._mul(a, b)
    def pow(self, a, e): return self._pow(a, e)
    eadd = add
    def lit(self, k):
        t = self.lit_cache.get(k)
        if t is None:
            if k < 0: raise AuthorError('N has no negative numerals (%d)' % k)
            if k == 0: t = self.zero
            else:
                p = self._xh
                for bit in bin(k)[3:]: p = (self._xi if bit == '1' else self._xo)(p)
                t = self._npos(p)
            self.lit_cache[k] = t
        return t
    elit = lit
    def _parse(self, t):
        """the value k when t is spelled as the canonical numeral of k (N0, or Npos of digits XH / XO / XI), else None; `npos` tells whether t is an application of Npos at all"""
        view = self.w.k.view; n0, npos, xh, xo, xi = self._ids
        v = view(t.n); res = None; is_npos = False
        if v[0] == 'const' and v[1] == n0: res = 0
        elif v[0] == 'app':
            f = view(v[1])
            if f[0] == 'const' and f[1] == npos:
                is_npos = True; bits = []; n = v[2]
                while True:
                    u = view(n)
                    if u[0] == 'const' and u[1] == xh: break
                    if u[0] != 'app': bits = None; break
                    g = view(u[1])
                    if g[0] != 'const' or g[1] not in (xo, xi): bits = None; break
                    bits.append(1 if g[1] == xi else 0); n = u[2]
                if bits is not None:
                    res = 1
                    for b in reversed(bits): res = res * 2 + b
        return res, is_npos
    def numeral(self, t):
        """the value k when t is a numeral of k, else None: the canonical spelling (N0, or Npos of digits XH / XO / XI), or a closed `Npos` term whose digits the kernel computes to it (the display
        normal form leaves `1 ^ 2` as `Npos (Pos.sqr XH)`). The two spellings are convertible, which the kernel checks when it meets the proof"""
        hit = self.num_cache.get(t)
        if hit is not None: return None if hit < 0 else hit
        res, is_npos = self._parse(t)
        if res is None and is_npos:
            try: res = self._parse(self.w.normalize(t))[0]
            except Exception: res = None             # a term with a free variable inside the digits is not a numeral
        self.num_cache[t] = -1 if res is None else res
        return res
    enumeral = numeral


class SemiringZ(Semiring):
    """the integers (int_lib.py). A numeral is `Z.of_N` of a numeral of N, or the negation of one (0 is `Z.zero`); the exponents are binary naturals"""
    pfx = 'Z.ring_'
    is_ring = True
    def __init__(self, w):
        self.w = w; self.ty = w['Z']; self.ety = w['N']; self.zero = w['Z.zero']
        self.N = SemiringN(w)
        self._add, self._mul, self._pow, self._neg, self._sub, self._of_N = w['Z.add'], w['Z.mul'], w['Z.pow'], w['Z.neg'], w['Z.sub'], w['Z.of_N']
        self._ids = {n: w.gid[n] for n in ('Z.zero', 'Z.one', 'Z.of_N', 'Z.neg', 'Z.add', 'Z.mul')}
        self.ops = {w.gid['Z.add']: 'add', w.gid['Z.mul']: 'mul', w.gid['Z.pow']: 'pow', w.gid['Z.neg']: 'neg', w.gid['Z.sub']: 'sub', w.gid['Z.sqr']: 'sqr'}      # (the display normal form turns `x ^ 2` into `Z.sqr x`)
        self.sqr_def = w['Z.sqr_def']
        self.zero_add, self.add_zero = w['Z.zero_add'], w['Z.add_zero']
        self.lit_cache = NodeCache(w); self.num_cache = NodeCache(w)
        self.one = self.lit(1); self.m1 = self.lit(-1)
    def add(self, a, b): return self._add(a, b)
    def mul(self, a, b): return self._mul(a, b)
    def pow(self, a, e): return self._pow(a, e)
    def eadd(self, a, b): return self.N.add(a, b)
    def lit(self, k):
        t = self.lit_cache.get(k)
        if t is None:
            t = self.zero if k == 0 else self._of_N(self.N.lit(k)) if k > 0 else self._neg(self._of_N(self.N.lit(-k)))
            self.lit_cache[k] = t
        return t
    def elit(self, k): return self.N.lit(k)
    def enumeral(self, t): return self.N.numeral(t)
    def numeral(self, t):
        """the value k when t is a numeral: `Z.zero`, `Z.one`, `Z.of_N` of a numeral of N, or the negation of that (else None). The forms other than the canonical `lit(k)` are definitionally equal to it"""
        hit = self.num_cache.get(t)
        if hit is not None: return None if hit is False else hit
        view = self.w.k.view; ids = self._ids; v = view(t.n); res = None
        if v[0] == 'const':
            res = 0 if v[1] == ids['Z.zero'] else 1 if v[1] == ids['Z.one'] else None
        elif v[0] == 'app':
            f = view(v[1])
            if f[0] == 'const' and f[1] == ids['Z.of_N']: res = self.N.numeral(self.w.T(v[2]))
            elif f[0] == 'const' and f[1] == ids['Z.neg']:
                u = view(v[2])
                if u[0] == 'app':
                    g = view(u[1])
                    if g[0] == 'const' and g[1] == ids['Z.of_N']:
                        kk = self.N.numeral(self.w.T(u[2])); res = None if kk is None else -kk
        self.num_cache[t] = False if res is None else res
        return res
    def _rep(self, t):
        """the pair (a, b) such that the closed integer expression t evaluates to `Z.mk a b` -- the kernel's computation on the representatives, done on Python integers"""
        view = self.w.k.view; ids = self._ids; v = view(t.n)
        if v[0] == 'const':
            if v[1] == ids['Z.zero']: return (0, 0)
            if v[1] == ids['Z.one']: return (1, 0)
        elif v[0] == 'app':
            f = view(v[1]); x = self.w.T(v[2])
            if f[0] == 'const' and f[1] == ids['Z.of_N']:
                k = self.N.numeral(x)
                if k is not None: return (k, 0)
            elif f[0] == 'const' and f[1] == ids['Z.neg']:
                a, b = self._rep(x); return (b, a)
            elif f[0] == 'app':
                g = view(f[1]); y = self.w.T(f[2])
                if g[0] == 'const' and g[1] == ids['Z.add']:
                    a, b = self._rep(y); c, d = self._rep(x); return (a + c, b + d)
                if g[0] == 'const' and g[1] == ids['Z.mul']:
                    a, b = self._rep(y); c, d = self._rep(x); return (a * c + b * d, a * d + b * c)
        raise AuthorError('ring: %s is not a closed integer numeral expression' % self.w.show(t))
    def num_eq(self, lhs, rhs):
        """a proof of  lhs = rhs  for closed numeral expressions of equal value. The two sides evaluate to  Z.mk a b  and  Z.mk c d  with  a + d = c + b  (not to the same term, as for the naturals); `Z.mk_eq` turns that equation of
        naturals, which the kernel checks by evaluation, into the equation of integers, and the kernel recognises its two sides as the given ones by evaluating them too"""
        a, b = self._rep(lhs); c, d = self._rep(rhs)
        if a + d != c + b: raise AuthorError('ring: internal error, the numerals %s and %s differ' % (self.w.show(lhs), self.w.show(rhs)))
        N = self.N; w = self.w
        return Pf(w['Z.mk_eq'](N.lit(a), N.lit(b), N.lit(c), N.lit(d), w['refl'](self.ety, N.lit(a + d))), lhs, rhs)


def _make_nat(w, A):
    if 'Nat' in w and 'ring_add_lt' in w and A == w['Nat']: return SemiringNat(w)
def _make_n(w, A):
    if 'N' in w and 'N.ring_add_lt' in w and A == w['N']: return SemiringN(w)
def _make_z(w, A):
    if 'Z' in w and 'Z.ring_add_lt' in w and A == w['Z']: return SemiringZ(w)
INSTANCE_MAKERS = [_make_nat, _make_n, _make_z]


def instance_for(w, A):
    """the number system of the type A (a Term) -- or None when `ring` knows none for it. Instances are kept per world: their numeral tables are worth keeping between calls."""
    cache = w.__dict__.get('_ring_instances')
    if cache is None: cache = w.__dict__['_ring_instances'] = NodeCache(w)
    inst = cache.get(A.n)
    if inst is None:
        for make in INSTANCE_MAKERS:
            inst = make(w, A)
            if inst is not None: cache[A.n] = inst; break
    return inst


# ---------------------------------------------------------------------------------------------------------------------------------------------- the normaliser
class Ring:
    def __init__(self, w, inst):
        self.w = w; self.R = inst; self.A = inst.ty
        self.atoms = []; self.atom_idx = {}
        self.zero_nf = NF((), inst.zero)
        self._poly_memo = {}; self._mono_memo = {}; self._add_memo = {}; self._mul_memo = {}; self._nf_memo = {}; self._pow_memo = {}; self._cexp_memo = {}; self._exp = None
        self._lc = {}; self._refl = w['refl']; self._symm = w['symm']; self._trans = w['trans']; self._zero_add = inst.zero_add; self._add_zero = inst.add_zero

    # ------------------------------------------------------------------------------------------------------------------ terms of normal forms
    def mono_term(self, m):
        t = self._mono_memo.get(m.key())
        if t is None:
            R = self.R
            t = R.lit(m.coeff)
            for (i, e) in reversed(m.vars): t = R.mul(R.pow(self.atoms[i], R.elit(e)), t)
            self._mono_memo[m.key()] = t
        return t
    def poly(self, monos):
        monos = tuple(monos); key = tuple(m.key() for m in monos)
        nf = self._poly_memo.get(key)
        if nf is None:
            R = self.R; t = R.zero
            for m in reversed(monos): t = R.add(self.mono_term(m), t)
            nf = NF(monos, t); self._poly_memo[key] = nf
        return nf
    def tail(self, nf): return self.poly(nf.monos[1:])
    def head_term(self, nf): return self.mono_term(nf.monos[0])

    # ------------------------------------------------------------------------------------------------------------------ proof combinators
    def L(self, name, *args):
        c = self._lc.get(name)
        if c is None: c = self._lc[name] = self.w[self.R.pfx + name]
        return c(*args)
    def refl(self, a): return Pf(self._refl(self.A, a), a, a)
    def symm(self, p): return Pf(self._symm(self.A, p.lhs, p.rhs, p.proof), p.rhs, p.lhs)
    def trans(self, p, q):
        if p.rhs != q.lhs: raise AuthorError('ring: internal error, transitivity of unrelated equations')
        return Pf(self._trans(self.A, p.lhs, p.rhs, q.rhs, p.proof, q.proof), p.lhs, q.rhs)
    def add_congr(self, p, q): return Pf(self.L('add_congr', p.lhs, q.lhs, p.rhs, q.rhs, p.proof, q.proof), self.R.add(p.lhs, q.lhs), self.R.add(p.rhs, q.rhs))
    def mul_congr(self, p, q): return Pf(self.L('mul_congr', p.lhs, q.lhs, p.rhs, q.rhs, p.proof, q.proof), self.R.mul(p.lhs, q.lhs), self.R.mul(p.rhs, q.rhs))
    def pow_congr(self, p, e): return Pf(self.L('pow_congr', p.lhs, p.rhs, e, p.proof), self.R.pow(p.lhs, e), self.R.pow(p.rhs, e))
    def pow_exp_congr(self, a, p): return Pf(self.L('pow_exp_congr', a, p.lhs, p.rhs, p.proof), self.R.pow(a, p.lhs), self.R.pow(a, p.rhs))

    # ------------------------------------------------------------------------------------------------------------------ sums
    def add(self, A, B):
        """(S, proof of A + B = S)"""
        key = (A.term, B.term); hit = self._add_memo.get(key)
        if hit is not None: return hit
        R = self.R; AB = R.add(A.term, B.term)
        if not A.monos: res = (B, Pf(self._zero_add(B.term), AB, B.term))
        elif not B.monos: res = (A, Pf(self._add_zero(A.term), AB, A.term))
        else:
            a, b = A.monos[0], B.monos[0]; c = cmp_vars(a.vars, b.vars)
            At, Bt = self.tail(A), self.tail(B); at, bt = self.mono_term(a), self.mono_term(b)
            if c > 0:
                Rr, h = self.add(At, B); S = self.poly((a,) + Rr.monos)
                res = (S, Pf(self.L('add_lt', at, At.term, B.term, Rr.term, h.proof), AB, S.term))
            elif c < 0:
                Rr, h = self.add(A, Bt); S = self.poly((b,) + Rr.monos)
                res = (S, Pf(self.L('add_gt', bt, A.term, Bt.term, Rr.term, h.proof), AB, S.term))
            else:
                m, h1 = self.mono_add(a, b); Rr, h2 = self.add(At, Bt)
                if m.coeff == 0:              # the coefficients cancel: the monomial leaves the sum (rings only)
                    res = (Rr, Pf(self.L('add_cancel', at, bt, At.term, Bt.term, Rr.term, h1.proof, h2.proof), AB, Rr.term))
                else:
                    S = self.poly((m,) + Rr.monos)
                    res = (S, Pf(self.L('add_eq', at, bt, self.mono_term(m), At.term, Bt.term, Rr.term, h1.proof, h2.proof), AB, S.term))
        self._add_memo[key] = res
        return res

    def mono_add(self, a, b):
        """two monomials with the same variable part: (their sum as a monomial, proof of  a + b = sum)"""
        coeff = a.coeff + b.coeff; m = Mono(a.vars, coeff)
        return m, self._mono_add_proof(a.vars, a.coeff, b.coeff)
    def _mono_add_proof(self, vars_, ca, cb):
        R = self.R
        # the proof is built from the inside out: constants first, then one `mono_add` lemma per variable. When the coefficients cancel the sum is the zero of the ring, and at every level
        # x^e * 0 is reduced to 0 at once (`scale_nil`), so that the whole monomial sum is proved equal to 0
        vanish = ca + cb == 0
        pf = R.num_eq(R.add(R.lit(ca), R.lit(cb)), R.lit(ca + cb))
        for j in range(len(vars_) - 1, -1, -1):
            i, e = vars_[j]; x = self.atoms[i]; et = R.elit(e); xe = R.pow(x, et)
            a_t = self.mono_term(Mono(vars_[j + 1:], ca)); b_t = self.mono_term(Mono(vars_[j + 1:], cb))
            s_t = R.zero if vanish else self.mono_term(Mono(vars_[j + 1:], ca + cb))
            pf = Pf(self.L('mono_add', x, et, a_t, b_t, s_t, pf.proof), R.add(R.mul(xe, a_t), R.mul(xe, b_t)), R.mul(xe, s_t))
            if vanish: pf = self.trans(pf, Pf(self.L('scale_nil', xe), R.mul(xe, R.zero), R.zero))
        return pf

    # ------------------------------------------------------------------------------------------------------------------ products
    def mono_mul(self, M, N):
        """(M * N as a monomial, proof of  M.term * N.term = result.term)"""
        R = self.R; INF = 1 << 60
        iM = M.vars[0][0] if M.vars else INF; iN = N.vars[0][0] if N.vars else INF
        mt, nt = self.mono_term(M), self.mono_term(N); mn = R.mul(mt, nt)
        if iM == INF and iN == INF:
            r = Mono((), M.coeff * N.coeff)
            return r, R.num_eq(mn, self.mono_term(r))
        if iM < iN:
            (i, e) = M.vars[0]; M1 = Mono(M.vars[1:], M.coeff); Rr, h = self.mono_mul(M1, N)
            r = Mono(((i, e),) + Rr.vars, Rr.coeff)
            return r, Pf(self.L('mono_lt', self.atoms[i], R.elit(e), self.mono_term(M1), nt, self.mono_term(Rr), h.proof), mn, self.mono_term(r))
        if iM > iN:
            (j, f) = N.vars[0]; N1 = Mono(N.vars[1:], N.coeff); Rr, h = self.mono_mul(M, N1)
            r = Mono(((j, f),) + Rr.vars, Rr.coeff)
            return r, Pf(self.L('mono_gt', self.atoms[j], R.elit(f), mt, self.mono_term(N1), self.mono_term(Rr), h.proof), mn, self.mono_term(r))
        (i, e) = M.vars[0]; (_, f) = N.vars[0]
        M1 = Mono(M.vars[1:], M.coeff); N1 = Mono(N.vars[1:], N.coeff); Rr, h2 = self.mono_mul(M1, N1)
        h1 = R.enum_eq(R.eadd(R.elit(e), R.elit(f)), R.elit(e + f))
        r = Mono(((i, e + f),) + Rr.vars, Rr.coeff)
        return r, Pf(self.L('mono_eq', self.atoms[i], R.elit(e), R.elit(f), R.elit(e + f), self.mono_term(M1), self.mono_term(N1), self.mono_term(Rr), h1.proof, h2.proof), mn, self.mono_term(r))

    def scale(self, M, Q):
        """(M * Q, proof of  M.term * Q.term = result.term) for a monomial M and a polynomial Q"""
        R = self.R; mt = self.mono_term(M)
        if not Q.monos: return self.zero_nf, Pf(self.L('scale_nil', mt), R.mul(mt, Q.term), R.zero)
        N, Qt = Q.monos[0], self.tail(Q)
        MN, h1 = self.mono_mul(M, N); Rr, h2 = self.scale(M, Qt)
        S = self.poly((MN,) + Rr.monos)        # the key of M*N is above every key of R: multiplication by M keeps the order, so no merge is needed
        return S, Pf(self.L('scale_cons', mt, self.mono_term(N), Qt.term, self.mono_term(MN), Rr.term, h1.proof, h2.proof), R.mul(mt, Q.term), S.term)

    def mul(self, A, B):
        key = (A.term, B.term); hit = self._mul_memo.get(key)
        if hit is not None: return hit
        R = self.R
        if not A.monos: res = (self.zero_nf, Pf(self.L('mulp_nil', B.term), R.mul(A.term, B.term), R.zero))
        else:
            M, P = A.monos[0], self.tail(A)
            X, h1 = self.scale(M, B); Y, h2 = self.mul(P, B); C, h3 = self.add(X, Y)
            res = (C, Pf(self.L('mulp_cons', self.mono_term(M), P.term, B.term, X.term, Y.term, C.term, h1.proof, h2.proof, h3.proof), R.mul(A.term, B.term), C.term))
        self._mul_memo[key] = res
        return res

    def pow(self, A, k):
        """(A^k, proof of  A.term ^ k = result.term) by repeated squaring: A^(2j) = A^j * A^j, A^(2j+1) = A^1 * A^(2j)"""
        key = (A.term, k); hit = self._pow_memo.get(key)
        if hit is not None: return hit
        R = self.R
        if k == 0:
            one = self.poly((Mono((), 1),))
            res = (one, Pf(self.L('pow_zero', A.term), R.pow(A.term, R.elit(0)), one.term))
        elif k == 1: res = (A, Pf(self.L('pow_one', A.term), R.pow(A.term, R.elit(1)), A.term))
        else:
            e1 = k // 2 if k % 2 == 0 else 1; e2 = k - e1
            P1, h1 = self.pow(A, e1); P2, h2 = self.pow(A, e2); C, h3 = self.mul(P1, P2)
            he = R.enum_eq(R.eadd(R.elit(e1), R.elit(e2)), R.elit(k))
            res = (C, Pf(self.L('pow_split', A.term, R.elit(e1), R.elit(e2), R.elit(k), P1.term, P2.term, C.term, he.proof, h1.proof, h2.proof, h3.proof), R.pow(A.term, R.elit(k)), C.term))
        self._pow_memo[key] = res
        return res

    # ------------------------------------------------------------------------------------------------------------------ reading an expression
    def atom(self, t):
        i = self.atom_idx.get(t)
        if i is None: i = self.atom_idx[t] = len(self.atoms); self.atoms.append(t)
        return i

    def nf(self, t):
        """(normal form of t, proof of  t = normal form)"""
        hit = self._nf_memo.get(t)
        if hit is not None: return hit
        w = self.w; R = self.R
        k = R.numeral(t)
        if k is not None:
            if k == 0: res = (self.zero_nf, Pf(self._refl(self.A, t), t, R.zero))       # (t may be another spelling of 0 than the zero of the ring: `refl t` proves it by conversion)
            else:
                S = self.poly((Mono((), k),))
                res = (S, Pf(self._symm(self.A, R.add(t, R.zero), t, self._add_zero(t)), t, S.term))
        else:
            head, args = w.spine(t); hv = w.k.view(head.n)
            kind = R.ops.get(hv[1]) if hv[0] == 'const' else None
            if kind == 'add' and len(args) == 2:
                A, pa = self.nf(args[0]); B, pb = self.nf(args[1]); S, ps = self.add(A, B)
                res = (S, self.trans(self.add_congr(pa, pb), ps))
            elif kind == 'mul' and len(args) == 2:
                A, pa = self.nf(args[0]); B, pb = self.nf(args[1]); S, ps = self.mul(A, B)
                res = (S, self.trans(self.mul_congr(pa, pb), ps))
            elif kind == 'pow' and len(args) == 2 and self.const_exp(args[1])[0] is not None:
                e, pe = self.const_exp(args[1]); A, pa = self.nf(args[0]); S, ps = self.pow(A, e)
                pw = self.trans(self.pow_congr(pa, R.elit(e)), ps)
                res = (S, pw if pe is None else self.trans(self.pow_exp_congr(args[0], pe), pw))
            elif kind == 'neg' and len(args) == 1:
                eq = Pf(self.L('neg', args[0]), t, R.mul(R.m1, args[0]))
                S, ps = self.nf(R.mul(R.m1, args[0]))
                res = (S, self.trans(eq, ps))
            elif kind == 'sub' and len(args) == 2:
                e2 = R.add(args[0], R.mul(R.m1, args[1]))
                eq = Pf(self.L('sub', args[0], args[1]), t, e2)
                S, ps = self.nf(e2)
                res = (S, self.trans(eq, ps))
            elif kind == 'sqr' and len(args) == 1:
                e2 = R.mul(args[0], args[0])
                eq = Pf(R.sqr_def(args[0]), t, e2)
                S, ps = self.nf(e2)
                res = (S, self.trans(eq, ps))
            elif kind == 'succ' and len(args) == 1:
                eq = Pf(self.L('succ', args[0]), t, R.add(args[0], R.one))
                S, ps = self.nf(R.add(args[0], R.one))
                res = (S, self.trans(eq, ps))
            else:
                i = self.atom(t); S = self.poly((Mono(((i, 1),), 1),))
                res = (S, Pf(self.L('atom', t), t, S.term))
        self._nf_memo[t] = res
        return res

    def exp_ring(self):
        """the normaliser for the exponents (this one, when they are numbers of the same kind)"""
        if self._exp is None:
            R = self.R
            if R.ety == R.ty: self._exp = self
            else:
                einst = instance_for(self.w, R.ety)
                if einst is None: raise AuthorError('ring: no commutative-semiring structure is known for the exponent type %s' % self.w.show(R.ety))
                self._exp = Ring(self.w, einst)
        return self._exp

    def const_exp(self, t):
        """(k, proof of  t = numeral k) when the exponent t is a closed expression with the value k -- the proof is None when t already is the numeral -- else (None, None)"""
        hit = self._cexp_memo.get(t)
        if hit is not None: return hit
        R = self.R; e = R.enumeral(t)
        if e is not None: res = (e, None)
        else:
            ex = self.exp_ring(); Ee, pe = ex.nf(t)                 # pe : t = Ee.term, in the exponent type
            if not Ee.monos: res = (0, pe)                           # Ee.term is the zero of the exponents, which is their numeral 0
            elif len(Ee.monos) == 1 and not Ee.monos[0].vars:
                k = Ee.monos[0].coeff; ER = ex.R
                res = (k, ex.trans(pe, Pf(ER.add_zero(ER.elit(k)), Ee.term, ER.elit(k))))        # numeral k + 0 = numeral k
            else: res = (None, None)
        self._cexp_memo[t] = res
        return res

    def show_nf(self, nf):
        def mono(m):
            parts = [('%s^%d' % (self.w.show(self.atoms[i]), e)) if e > 1 else self.w.show(self.atoms[i]) for i, e in m.vars]
            return '*'.join(([str(m.coeff)] if (m.coeff != 1 or not parts) else []) + parts)
        return ' + '.join(mono(m) for m in nf.monos) or '0'

    def prove_eq(self, a, b):
        """a proof of  a = b  when both sides have the same normal form, else None together with the two normal forms"""
        A, pa = self.nf(a); B, pb = self.nf(b)
        if A.term != B.term: return None, (A, B)
        return self.trans(pa, self.symm(pb)), (A, B)


def ring(p):
    """close the first goal, an equation of expressions in a commutative semiring or ring (Nat, N or Z), by normalising both sides"""
    w = p.w; g = p.goal()
    if sys.getrecursionlimit() < 30000: sys.setrecursionlimit(30000)
    with p.focus():
        A, a, b = w.eq_parts(g.target)
        inst = instance_for(w, A)
        if inst is None: raise AuthorError('ring: no commutative-semiring structure is known for the type %s' % w.show(A))
        rg = Ring(w, inst)
        pf, (na, nb) = rg.prove_eq(a, b)
        if pf is None: raise AuthorError('ring failed: the two sides have different normal forms\n  left:  %s\n  right: %s\n%s' % (rg.show_nf(na), rg.show_nf(nb), p))
        proof = pf.proof
    p.exact(proof)
    p.trace.append(('ring',))
    return p
