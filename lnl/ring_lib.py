"""ring_lib.py -- the lemmas the `ring` normaliser (ring.py) instantiates, written ONCE for every commutative semiring that has the usual laws, and instantiated for the unary naturals (`ring_*`) and the
binary naturals (`N.ring_*`).

Nothing here is trusted: each lemma is a theorem proved by a tactic script from the semiring laws of its number system, admitted through the kernel's checked path like every other theorem. `ring`
(ring.py) only ever APPLIES them, so a mistake in ring.py can make a proof fail to check, never make a false equation provable.

A number system is described by a `RingDesc`: the carrier T of the numbers and the type ET of the exponents (the same for the naturals; for a ring of integers the exponents are still naturals), the zero and
one, and `law(name)`, the name under which the system's own law (`add_assoc`, `mul_add`, ...) is declared. The laws used are exactly these:

    add_zero  add_assoc  add_left_comm  mul_zero  mul_one  mul_assoc  mul_left_comm  mul_add  add_mul  pow_one  pow_add

and the exponent type needs `pow_add : x ^ (e + f) = x ^ e * x ^ f` (stated for ET addition).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from author import AuthorError
from lnl_base import thm


class RingDesc:
    """what the lemma template needs to know about one commutative semiring (carrier T, exponents in ET)"""
    def __init__(self, w, T, ET, pfx, law, zero, one, ezero, eone, succ=None, succ_script=None, neg=None, sub=None, m1=None, neg_script=None, sub_script=None, pow_one_computes=False, zero_mul_computes=True):
        self.w = w; self.T = T; self.ET = ET; self.pfx = pfx; self.law = law
        self.zero = zero; self.one = one; self.ezero = ezero; self.eone = eone
        self.succ = succ; self.succ_script = succ_script
        # a RING (not just a semiring) also has negation and subtraction, and the numeral -1 (the lemmas `neg` and `sub` express both through multiplication by it)
        self.neg = neg; self.sub = sub; self.m1 = m1; self.neg_script = neg_script; self.sub_script = sub_script
        # `x ^ 1` computes to `x` (the power is a recursive function on the binary digits of the exponent): the goals are then displayed with it computed, and the library's `pow_one` reads `x = x`
        self.pow_one_computes = pow_one_computes
        # `0 * q` computes to `0` for the naturals (the product recurses on its first argument); for a quotient it is the law `zero_mul`
        self.zero_mul_computes = zero_mul_computes


def ring_lemma_set(w, d):
    """declare the `ring_*` lemmas (named d.pfx + ...) for the semiring described by d"""
    T, ET, Z, one, ez, eo = d.T, d.ET, d.zero, d.one, d.ezero, d.eone
    Eq = w.Eq; L = d.law; R = lambda name: d.pfx + name

    @thm(w, R('add_congr'), [T, T, T, T, lambda a, b, c, d_: Eq(a, c), lambda a, b, c, d_, h1: Eq(b, d_)], lambda a, b, c, d_, h1, h2: Eq(a + b, c + d_))
    def _(p): p.intro('a', 'b', 'c', 'd', 'h1', 'h2'); p.cases('h1'); p.cases('h2'); p.refl()

    @thm(w, R('mul_congr'), [T, T, T, T, lambda a, b, c, d_: Eq(a, c), lambda a, b, c, d_, h1: Eq(b, d_)], lambda a, b, c, d_, h1, h2: Eq(a * b, c * d_))
    def _(p): p.intro('a', 'b', 'c', 'd', 'h1', 'h2'); p.cases('h1'); p.cases('h2'); p.refl()

    @thm(w, R('pow_congr'), [T, T, ET, lambda a, c, e: Eq(a, c)], lambda a, c, e, h: Eq(a ** e, c ** e))
    def _(p): p.intro('a', 'c', 'e', 'h'); p.cases('h'); p.refl()

    @thm(w, R('pow_exp_congr'), [T, ET, ET, lambda a, e, f: Eq(e, f)], lambda a, e, f, h: Eq(a ** e, a ** f))
    def _(p): p.intro('a', 'e', 'f', 'h'); p.cases('h'); p.refl()

    @thm(w, R('atom'), [T], lambda x: Eq(x, x ** eo * one + Z))
    def _(p):
        p.intro('x')
        if not d.pow_one_computes: p.rw(L('pow_one'))
        p.rw(L('mul_one')); p.rw(L('add_zero'))

    if d.succ is not None:
        @thm(w, R('succ'), [T], lambda x: Eq(d.succ(x), x + one))
        def _(p): d.succ_script(p)

    # sums of normal forms (monomials sorted by key)
    @thm(w, R('add_lt'), [T, T, T, T, lambda a, A, B, C: Eq(A + B, C)], lambda a, A, B, C, h: Eq((a + A) + B, a + C))
    def _(p): p.intro('a', 'A', 'B', 'C', 'h'); p.rw(L('add_assoc')); p.rw('h')

    @thm(w, R('add_gt'), [T, T, T, T, lambda b, A, B, C: Eq(A + B, C)], lambda b, A, B, C, h: Eq(A + (b + B), b + C))
    def _(p): p.intro('b', 'A', 'B', 'C', 'h'); p.rw(L('add_left_comm')); p.rw('h')

    @thm(w, R('add_eq'), [T, T, T, T, T, T, lambda a, b, c, A, B, C: Eq(a + b, c), lambda a, b, c, A, B, C, h1: Eq(A + B, C)],
         lambda a, b, c, A, B, C, h1, h2: Eq((a + A) + (b + B), c + C))
    def _(p):
        p.intro('a', 'b', 'c', 'A', 'B', 'C', 'h1', 'h2')
        p.rw(L('add_assoc')); p.rw(L('add_left_comm'), 'A'); p.rw(L('add_assoc'), rev=True); p.rw('h1'); p.rw('h2')

    @thm(w, R('mono_add'), [T, ET, T, T, T, lambda x, e, a, b, s: Eq(a + b, s)], lambda x, e, a, b, s, h: Eq(x ** e * a + x ** e * b, x ** e * s))
    def _(p): p.intro('x', 'e', 'a', 'b', 's', 'h'); p.rw(L('mul_add'), rev=True); p.rw('h')

    if d.neg is not None:
        # a monomial whose coefficients cancel disappears from the sum
        @thm(w, R('add_cancel'), [T, T, T, T, T, lambda a, b, A, B, C: Eq(a + b, Z), lambda a, b, A, B, C, h1: Eq(A + B, C)], lambda a, b, A, B, C, h1, h2: Eq((a + A) + (b + B), C))
        def _(p):
            p.intro('a', 'b', 'A', 'B', 'C', 'h1', 'h2')
            p.rw(L('add_assoc')); p.rw(L('add_left_comm'), 'A'); p.rw(L('add_assoc'), rev=True); p.rw('h1'); p.rw('h2'); p.rw(L('zero_add'))

        # negation and subtraction are multiplication by the numeral -1
        @thm(w, R('neg'), [T], lambda x: Eq(d.neg(x), d.m1 * x))
        def _(p): d.neg_script(p)

        @thm(w, R('sub'), [T, T], lambda x, y: Eq(d.sub(x, y), x + d.m1 * y))
        def _(p): d.sub_script(p)

    # products of monomials  x^e * (y^f * ... * c)
    @thm(w, R('mono_lt'), [T, ET, T, T, T, lambda x, e, A, B, R_: Eq(A * B, R_)], lambda x, e, A, B, R_, h: Eq((x ** e * A) * B, x ** e * R_))
    def _(p): p.intro('x', 'e', 'A', 'B', 'R', 'h'); p.rw(L('mul_assoc')); p.rw('h')

    @thm(w, R('mono_gt'), [T, ET, T, T, T, lambda y, f, A, B, R_: Eq(A * B, R_)], lambda y, f, A, B, R_, h: Eq(A * (y ** f * B), y ** f * R_))
    def _(p): p.intro('y', 'f', 'A', 'B', 'R', 'h'); p.rw(L('mul_left_comm')); p.rw('h')

    @thm(w, R('mul_mul_mul'), [T, T, T, T], lambda a, b, c, d_: Eq((a * b) * (c * d_), (a * c) * (b * d_)))
    def _(p): p.intro('a', 'b', 'c', 'd'); p.rw(L('mul_assoc')); p.rw(L('mul_left_comm'), 'b'); p.rw(L('mul_assoc'), rev=True)

    @thm(w, R('mono_eq'), [T, ET, ET, ET, T, T, T, lambda x, e, f, g, A, B, R_: Eq(e + f, g), lambda x, e, f, g, A, B, R_, h1: Eq(A * B, R_)],
         lambda x, e, f, g, A, B, R_, h1, h2: Eq((x ** e * A) * (x ** f * B), x ** g * R_))
    def _(p):
        p.intro('x', 'e', 'f', 'g', 'A', 'B', 'R', 'h1', 'h2')
        p.rw(R('mul_mul_mul')); p.rw(L('pow_add'), rev=True); p.rw('h1'); p.rw('h2')

    # products of polynomials
    @thm(w, R('scale_nil'), [T], lambda M: Eq(M * Z, Z))
    def _(p): p.intro('M'); p.rw(L('mul_zero'))

    @thm(w, R('scale_cons'), [T, T, T, T, T, lambda M, N_, Q, MN, R_: Eq(M * N_, MN), lambda M, N_, Q, MN, R_, h1: Eq(M * Q, R_)],
         lambda M, N_, Q, MN, R_, h1, h2: Eq(M * (N_ + Q), MN + R_))
    def _(p): p.intro('M', 'N', 'Q', 'MN', 'R', 'h1', 'h2'); p.rw(L('mul_add')); p.rw('h1'); p.rw('h2')

    @thm(w, R('mulp_nil'), [T], lambda Q: Eq(Z * Q, Z))
    def _(p):
        p.intro('Q')
        if d.zero_mul_computes: p.refl()
        else: p.rw(L('zero_mul'))

    @thm(w, R('mulp_cons'), [T, T, T, T, T, T, lambda M, P, Q, A, B, C: Eq(M * Q, A), lambda M, P, Q, A, B, C, h1: Eq(P * Q, B), lambda M, P, Q, A, B, C, h1, h2: Eq(A + B, C)],
         lambda M, P, Q, A, B, C, h1, h2, h3: Eq((M + P) * Q, C))
    def _(p): p.intro('M', 'P', 'Q', 'A', 'B', 'C', 'h1', 'h2', 'h3'); p.rw(L('add_mul')); p.rw('h1'); p.rw('h2'); p.rw('h3')

    # powers by a numeral, by repeated squaring: P^e = P^e1 * P^e2 when e = e1 + e2
    @thm(w, R('pow_zero'), [T], lambda P: Eq(P ** ez, one + Z))
    def _(p): p.intro('P'); p.refl()

    @thm(w, R('pow_one'), [T], lambda P: Eq(P ** eo, P))
    def _(p):
        p.intro('P')
        if d.pow_one_computes: p.refl()
        else: p.rw(L('pow_one'))

    @thm(w, R('pow_split'), [T, ET, ET, ET, T, T, T, lambda P, e1, e2, e, A, B, C: Eq(e1 + e2, e), lambda P, e1, e2, e, A, B, C, he: Eq(P ** e1, A),
                              lambda P, e1, e2, e, A, B, C, he, h1: Eq(P ** e2, B), lambda P, e1, e2, e, A, B, C, he, h1, h2: Eq(A * B, C)],
         lambda P, e1, e2, e, A, B, C, he, h1, h2, h3: Eq(P ** e, C))
    def _(p):
        p.intro('P', 'e1', 'e2', 'e', 'A', 'B', 'C', 'he', 'h1', 'h2', 'h3')
        p.cases('he'); p.rw(L('pow_add')); p.rw('h1'); p.rw('h2'); p.exact('h3')


# ---------------------------------------------------------------------------------------------------------------------------------------------- the two instances
def nat_desc(w):
    """the unary naturals: the laws are the library's own (`add_assoc`, ...)"""
    Nat, S, Z = w['Nat'], w['Succ'], w['Zero']
    def succ_script(p): p.intro('x'); p.rw('add_succ'); p.rw('add_zero')
    return RingDesc(w, Nat, Nat, 'ring_', lambda name: name, Z, S(Z), Z, S(Z), succ=S, succ_script=succ_script)


def n_desc(w):
    """the binary naturals: the laws are the `N.*` ones (proved by transfer to the unary numerals)"""
    N, N0, one = w['N'], w['N0'], w['Npos'](w['XH'])
    def succ_script(p): p.intro('x'); p.apply('N.succ_eq_add_one')
    return RingDesc(w, N, N, 'N.ring_', lambda name: 'N.' + name, N0, one, N0, one, succ=w['N.succ'], succ_script=succ_script)


def z_desc(w):
    """the integers (int_lib.py), with exponents in the binary naturals; `-1` is the negation of the numeral 1"""
    Z, N, N0 = w['Z'], w['N'], w['N0']; one_n = w['Npos'](w['XH']); zof, zneg = w['Z.of_N'], w['Z.neg']
    m1 = zneg(zof(one_n))
    def neg_script(p):
        p.intro('x'); p.apply('symm', Z); p.rw('Z.neg_mul'); p.apply('f_equal', Z, Z, zneg); p.exact(lambda q: w['Z.one_mul'](q.h('x')))
    def sub_script(p):
        p.intro('x', 'y'); p.rw('Z.sub_def'); p.rw('Z.ring_neg', 'y')
    return RingDesc(w, Z, N, 'Z.ring_', lambda name: 'Z.' + name, w['Z.zero'], w['Z.one'], N0, one_n, neg=zneg, sub=w['Z.sub'], m1=m1, neg_script=neg_script, sub_script=sub_script, pow_one_computes=True, zero_mul_computes=False)
