"""ring.py -- the `ring` tactic for commutative semirings (here: the natural numbers), proof-producing and checked by the kernel.

NOTHING HERE IS TRUSTED. `ring` normalises both sides of an equation to a canonical sum of monomials and, at every step, builds the proof term that justifies it: an application of one of the
library's `ring_*` lemmas (nat_lib.ring_lemmas), each a theorem proved from the semiring laws. The finished term is handed to the kernel like any other proof, so a bug in this file can make `ring`
fail or produce a term the kernel rejects, never make a false equation provable.

Normal form (the same shape Mathlib's `ring` uses, so that every merge is one lemma application):

    polynomial   m1 + (m2 + (... + (mk + 0)))            monomials in strictly decreasing key order, coefficients >= 1
    monomial     x1 ^ e1 * (x2 ^ e2 * (... * (xj ^ ej * c)))   atoms in increasing index order, exponents >= 1 (numerals), the coefficient c a numeral last

The ORDER on monomials (`cmp_vars`) is the lexicographic order on exponent vectors with the first atom most significant; it is a total order that is compatible with multiplication, which is what lets
`scale` (monomial times polynomial) keep a polynomial sorted without merging.

Statement-carrying proofs: `Pf(proof, lhs, rhs)` knows what it proves, so each combinator can build the next lemma instance without asking the kernel for a type.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from author import AuthorError, Term


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


class SemiringNat:
    """the instance: the natural numbers with the library's names"""
    def __init__(self, w):
        self.w = w; self.ty = w['Nat']; self.zero = w['Zero']; self.succ = w['Succ']
        self.lit_cache = {}; self.num_cache = {}
        self.one = self.lit(1)
    def lit(self, k):
        t = self.lit_cache.get(k)
        if t is None:
            t = self.zero if k == 0 else self.succ(self.lit(k - 1)) if k < 200 else self._lit_big(k)
            self.lit_cache[k] = t
        return t
    def _lit_big(self, k):
        t = self.lit(199)
        for _ in range(k - 199): t = self.succ(t)
        return t
    def numeral(self, t):
        """the value k when t is the numeral S^k Zero, else None"""
        hit = self.num_cache.get(t)
        if hit is not None: return None if hit < 0 else hit
        view = self.w.k.view; zero = self.w.gid['Zero']; succ = self.w.gid['Succ']
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


class Ring:
    def __init__(self, w, inst=None):
        self.w = w; self.R = inst or SemiringNat(w); self.A = self.R.ty
        self.atoms = []; self.atom_idx = {}
        self.zero_nf = NF((), self.R.zero)
        self._poly_memo = {}; self._mono_memo = {}; self._add_memo = {}; self._mul_memo = {}; self._nf_memo = {}

    # ------------------------------------------------------------------------------------------------------------------ terms of normal forms
    def mono_term(self, m):
        t = self._mono_memo.get(m.key())
        if t is None:
            t = self.R.lit(m.coeff)
            for (i, e) in reversed(m.vars): t = self.atoms[i] ** self.R.lit(e) * t
            self._mono_memo[m.key()] = t
        return t
    def poly(self, monos):
        monos = tuple(monos)
        nf = self._poly_memo.get(tuple(m.key() for m in monos))
        if nf is None:
            t = self.R.zero
            for m in reversed(monos): t = self.mono_term(m) + t
            nf = NF(monos, t); self._poly_memo[tuple(m.key() for m in monos)] = nf
        return nf
    def tail(self, nf): return self.poly(nf.monos[1:])
    def head_term(self, nf): return self.mono_term(nf.monos[0])

    # ------------------------------------------------------------------------------------------------------------------ proof combinators
    def L(self, name, *args): return self.w['ring_' + name](*args)
    def refl(self, a): return Pf(self.w['refl'](self.A, a), a, a)
    def symm(self, p): return Pf(self.w['symm'](self.A, p.lhs, p.rhs, p.proof), p.rhs, p.lhs)
    def trans(self, p, q):
        if p.rhs != q.lhs: raise AuthorError('ring: internal error, transitivity of unrelated equations')
        return Pf(self.w['trans'](self.A, p.lhs, p.rhs, q.rhs, p.proof, q.proof), p.lhs, q.rhs)
    def add_congr(self, p, q): return Pf(self.L('add_congr', p.lhs, q.lhs, p.rhs, q.rhs, p.proof, q.proof), p.lhs + q.lhs, p.rhs + q.rhs)
    def mul_congr(self, p, q): return Pf(self.L('mul_congr', p.lhs, q.lhs, p.rhs, q.rhs, p.proof, q.proof), p.lhs * q.lhs, p.rhs * q.rhs)
    def pow_congr(self, p, e): return Pf(self.L('pow_congr', p.lhs, p.rhs, e, p.proof), p.lhs ** e, p.rhs ** e)
    def by_computation(self, lhs, rhs): return Pf(self.w['refl'](self.A, rhs), lhs, rhs)           # lhs and rhs compute to the same numeral: the kernel checks it by evaluation

    # ------------------------------------------------------------------------------------------------------------------ sums
    def add(self, A, B):
        """(S, proof of A + B = S)"""
        key = (A.term, B.term); hit = self._add_memo.get(key)
        if hit is not None: return hit
        w = self.w
        if not A.monos: res = (B, Pf(w['zero_add'](B.term), A.term + B.term, B.term))
        elif not B.monos: res = (A, Pf(w['add_zero'](A.term), A.term + B.term, A.term))
        else:
            a, b = A.monos[0], B.monos[0]; c = cmp_vars(a.vars, b.vars)
            At, Bt = self.tail(A), self.tail(B); at, bt = self.mono_term(a), self.mono_term(b)
            if c > 0:
                R, h = self.add(At, B); S = self.poly((a,) + R.monos)
                res = (S, Pf(self.L('add_lt', at, At.term, B.term, R.term, h.proof), A.term + B.term, S.term))
            elif c < 0:
                R, h = self.add(A, Bt); S = self.poly((b,) + R.monos)
                res = (S, Pf(self.L('add_gt', bt, A.term, Bt.term, R.term, h.proof), A.term + B.term, S.term))
            else:
                m, h1 = self.mono_add(a, b); R, h2 = self.add(At, Bt); S = self.poly((m,) + R.monos)
                res = (S, Pf(self.L('add_eq', at, bt, self.mono_term(m), At.term, Bt.term, R.term, h1.proof, h2.proof), A.term + B.term, S.term))
        self._add_memo[key] = res
        return res

    def mono_add(self, a, b):
        """two monomials with the same variable part: (their sum as a monomial, proof of  a + b = sum)"""
        coeff = a.coeff + b.coeff; m = Mono(a.vars, coeff)
        return m, self._mono_add_proof(a.vars, a.coeff, b.coeff, 0)
    def _mono_add_proof(self, vars_, ca, cb, depth):
        w = self.w
        # the proof is built from the inside out: constants first, then one `mono_add` lemma per variable
        sc = self.R.lit(ca + cb)
        pf = self.by_computation(self.R.lit(ca) + self.R.lit(cb), sc)
        for j in range(len(vars_) - 1, -1, -1):
            i, e = vars_[j]; x = self.atoms[i]; et = self.R.lit(e)
            a_t = self.mono_term(Mono(vars_[j + 1:], ca)); b_t = self.mono_term(Mono(vars_[j + 1:], cb)); s_t = self.mono_term(Mono(vars_[j + 1:], ca + cb))
            pf = Pf(self.L('mono_add', x, et, a_t, b_t, s_t, pf.proof), x ** et * a_t + x ** et * b_t, x ** et * s_t)
        return pf

    # ------------------------------------------------------------------------------------------------------------------ products
    def mono_mul(self, M, N):
        """(M * N as a monomial, proof of  M.term * N.term = result.term)"""
        w = self.w; INF = 1 << 60
        iM = M.vars[0][0] if M.vars else INF; iN = N.vars[0][0] if N.vars else INF
        mt, nt = self.mono_term(M), self.mono_term(N)
        if iM == INF and iN == INF:
            r = Mono((), M.coeff * N.coeff)
            return r, self.by_computation(mt * nt, self.mono_term(r))
        if iM < iN:
            (i, e) = M.vars[0]; M1 = Mono(M.vars[1:], M.coeff); R, h = self.mono_mul(M1, N)
            r = Mono(((i, e),) + R.vars, R.coeff)
            return r, Pf(self.L('mono_lt', self.atoms[i], self.R.lit(e), self.mono_term(M1), nt, self.mono_term(R), h.proof), mt * nt, self.mono_term(r))
        if iM > iN:
            (j, f) = N.vars[0]; N1 = Mono(N.vars[1:], N.coeff); R, h = self.mono_mul(M, N1)
            r = Mono(((j, f),) + R.vars, R.coeff)
            return r, Pf(self.L('mono_gt', self.atoms[j], self.R.lit(f), mt, self.mono_term(N1), self.mono_term(R), h.proof), mt * nt, self.mono_term(r))
        (i, e) = M.vars[0]; (_, f) = N.vars[0]
        M1 = Mono(M.vars[1:], M.coeff); N1 = Mono(N.vars[1:], N.coeff); R, h2 = self.mono_mul(M1, N1)
        h1 = self.by_computation(self.R.lit(e) + self.R.lit(f), self.R.lit(e + f))
        r = Mono(((i, e + f),) + R.vars, R.coeff)
        return r, Pf(self.L('mono_eq', self.atoms[i], self.R.lit(e), self.R.lit(f), self.R.lit(e + f), self.mono_term(M1), self.mono_term(N1), self.mono_term(R), h1.proof, h2.proof), mt * nt, self.mono_term(r))

    def scale(self, M, Q):
        """(M * Q, proof of  M.term * Q.term = result.term) for a monomial M and a polynomial Q"""
        w = self.w; mt = self.mono_term(M)
        if not Q.monos: return self.zero_nf, Pf(self.L('scale_nil', mt), mt * Q.term, self.R.zero)
        N, Qt = Q.monos[0], self.tail(Q)
        MN, h1 = self.mono_mul(M, N); R, h2 = self.scale(M, Qt)
        S = self.poly((MN,) + R.monos)        # the key of M*N is above every key of R: multiplication by M keeps the order, so no merge is needed
        return S, Pf(self.L('scale_cons', mt, self.mono_term(N), Qt.term, self.mono_term(MN), R.term, h1.proof, h2.proof), mt * Q.term, S.term)

    def mul(self, A, B):
        key = (A.term, B.term); hit = self._mul_memo.get(key)
        if hit is not None: return hit
        if not A.monos: res = (self.zero_nf, Pf(self.L('mulp_nil', B.term), A.term * B.term, self.R.zero))
        else:
            M, P = A.monos[0], self.tail(A)
            X, h1 = self.scale(M, B); Y, h2 = self.mul(P, B); C, h3 = self.add(X, Y)
            res = (C, Pf(self.L('mulp_cons', self.mono_term(M), P.term, B.term, X.term, Y.term, C.term, h1.proof, h2.proof, h3.proof), A.term * B.term, C.term))
        self._mul_memo[key] = res
        return res

    def pow(self, A, k):
        if k == 0:
            one = self.poly((Mono((), 1),))
            return one, Pf(self.L('pow_zero', A.term), A.term ** self.R.lit(0), one.term)
        R, h1 = self.pow(A, k - 1); B, h2 = self.mul(A, R)
        return B, Pf(self.L('pow_succ', A.term, self.R.lit(k - 1), R.term, B.term, h1.proof, h2.proof), A.term ** self.R.lit(k), B.term)

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
            if k == 0: res = (self.zero_nf, self.refl(t))
            else:
                S = self.poly((Mono((), k),))
                res = (S, Pf(w['symm'](self.A, t + R.zero, t, w['add_zero'](t)), t, S.term))
        else:
            name = w.head_name(t); head, args = w.spine(t)
            if name == 'add' and len(args) == 2:
                A, pa = self.nf(args[0]); B, pb = self.nf(args[1]); S, ps = self.add(A, B)
                res = (S, self.trans(self.add_congr(pa, pb), ps))
            elif name == 'mul' and len(args) == 2:
                A, pa = self.nf(args[0]); B, pb = self.nf(args[1]); S, ps = self.mul(A, B)
                res = (S, self.trans(self.mul_congr(pa, pb), ps))
            elif name == 'pow' and len(args) == 2 and R.numeral(args[1]) is not None:
                e = R.numeral(args[1]); A, pa = self.nf(args[0]); S, ps = self.pow(A, e)
                res = (S, self.trans(self.pow_congr(pa, args[1]), ps))
            elif name == 'Succ' and len(args) == 1:
                eq = Pf(self.L('succ', args[0]), t, args[0] + R.one)
                S, ps = self.nf(args[0] + R.one)
                res = (S, self.trans(eq, ps))
            else:
                i = self.atom(t); S = self.poly((Mono(((i, 1),), 1),))
                res = (S, Pf(self.L('atom', t), t, S.term))
        self._nf_memo[t] = res
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
    """close the first goal, an equation of natural-number expressions, by normalising both sides"""
    w = p.w; g = p.goal()
    with p.focus():
        A, a, b = w.eq_parts(g.target)
        rg = Ring(w)
        pf, (na, nb) = rg.prove_eq(a, b)
        if pf is None: raise AuthorError('ring failed: the two sides have different normal forms\n  left:  %s\n  right: %s\n%s' % (rg.show_nf(na), rg.show_nf(nb), p))
        proof = pf.proof
    p.exact(proof)
    p.trace.append(('ring',))
    return p
