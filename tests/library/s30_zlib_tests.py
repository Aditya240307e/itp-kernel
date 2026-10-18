"""S30 gate: the integers `Z` (int_lib.py) -- a quotient of pairs of binary naturals, with every law proved -- and `ring` over them.

Nothing in the integer library is trusted: `Z` is `Quot (Prod N N) Zrel`, the operations are `Quot.lift` / `Quot.lift2` on representatives, the laws are theorems proved by quotient induction down to equations of N
(which the trace-based `ring` proves and the kernel checks), `Z.mk_inj` comes from a Bool-valued test lifted out of the quotient (no `propext`), and `funext` is derived. This gate pins down WHAT the library says
(a mistake in a statement would still be a theorem, just not the one the library claims to have) and that `ring` over `Z` is sound and complete for commutative rings:

  A. the statements: ~110 theorems of the library, written out here with explicit operations and compared as terms with what the library declared (and: nothing in the integer part of the world is undeclared in the
     table); the definitions have the intended types and the intended computation (a table of conversion checks, with negative controls: `Z.mk 1 0` and `Z.mk 2 1` are different terms); the world has NO axiom and
     every theorem rests on the quotient primitives only; `funext` is a theorem of the kernel plus `Quot.sound` and proves a function equality that is not definitional;
  B. numerals: `lit` builds exactly the canonical numeral of a value, `numeral` reads back exactly the four spellings (`Z.zero`, `Z.one`, `Z.of_N n`, `Z.neg (Z.of_N n)`) and nothing else; the kernel's evaluation of
     `+`, `*`, `-`, negation and powers on numerals of up to 70 bits is the Python integer arithmetic;
  C. fixed ring identities proved natively (the proof mentions the `Z.ring_*` lemmas and no lemma of the naturals' rings): binomials, differences of powers, Sophie Germain, Brahmagupta, Lagrange, numerals of
     100 bits, applications as atoms, a hypothesis in the context; the false ones are refused with both normal forms (coefficients of either sign); goals that are no equation, types with no ring structure;
  D. ~200 random expressions with `+ - * neg ^`, numerals of either sign and several spellings, against Python polynomial arithmetic: a rewriting is proved, a perturbation is proved iff the polynomials agree
     and refused otherwise;
  E. the normal form itself is the Python polynomial, atom by atom (monomials, exponents, SIGNED coefficients);
  F. the order, semantically: for random integers (up to 70 bits) the library proves `a <= b` / `a < b` exactly when Python says so, and refutes it exactly when Python says not, with its own lemmas;
  G. the independent checker re-checks every global of the integer part of the world (and the theorems proved here) -- the quotient construction, `Quot.lift2`, the laws, `ring`'s lemmas, the order;
  H. corrupted ring proofs over Z: the kernel and the independent checker agree on every one; speed: (a - b)^8, (a + b + c)^4, numeral exponents of 70 bits.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, sys, time

sys.setrecursionlimit(200000)
import itp_env_lnl as E
import bin_lib, int_lib, ring as R, audit
from author import AuthorError
from tactics import Proof
import indep_checker as ic
from harness import parse_raw
from s23_ring_tests import rebuild, dag_nodes, padd, pmul, ppow, size
from s25_nring_tests import ring_spec, consts_in, refused_with

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S30_TRACE'))
INDEP_MAX_TREE = int(os.environ.get('S30_INDEP_TREE', '20000'))     # only the identities proved here whose proof tree has at most this many nodes go to the (slow) independent checker as well
def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond


# ---------------------------------------------------------------------------------------------------------------------- A. the statements
def zspec(w):
    """the statements of the integer library, written out with explicit operations (no notation, nothing taken from int_lib)"""
    TYPE, PROP = w.TYPE, w.PROP; Eq = w.Eq; F = w.forall_; Ex = w['Ex']; Fls = w['False']
    N, Pos, Z = w['N'], w['Pos'], w['Z']; N0, N1 = w['N0'], w['Npos'](w['XH'])
    nadd, nmul, nsucc, nle, nlt = w['N.add'], w['N.mul'], w['N.succ'], w['N.le'], w['N.lt']; psucc = w['Pos.succ']
    mk, zadd, zmul, zneg, zsub, zpow, zof, zsqr, zppow = (w[n] for n in ('Z.mk', 'Z.add', 'Z.mul', 'Z.neg', 'Z.sub', 'Z.pow', 'Z.of_N', 'Z.sqr', 'Z.ppow'))
    Z0, Z1 = w['Z.zero'], w['Z.one']; zle, zlt = w['Z.le'], w['Z.lt']
    S = {}
    # ---- the quotient: soundness, induction, effectiveness; computation on representatives
    S['Z.mk_eq'] = F([N, N, N, N, lambda a, b, c, d: Eq(nadd(a, d), nadd(c, b))], lambda a, b, c, d, h: Eq(mk(a, b), mk(c, d)))
    S['Z.mk_inj'] = F([N, N, N, N, lambda a, b, c, d: Eq(mk(a, b), mk(c, d))], lambda a, b, c, d, h: Eq(nadd(a, d), nadd(c, b)))
    S['Z.induction_on'] = F([w.arrow(Z, PROP), Z, lambda P, z: F([N, N], lambda a, b: P(mk(a, b)))], lambda P, z, h: P(z))
    S['Z.neg_mk'] = F([N, N], lambda a, b: Eq(zneg(mk(a, b)), mk(b, a)))
    S['Z.add_mk'] = F([N] * 4, lambda a, b, c, d: Eq(zadd(mk(a, b), mk(c, d)), mk(nadd(a, c), nadd(b, d))))
    S['Z.mul_mk'] = F([N] * 4, lambda a, b, c, d: Eq(zmul(mk(a, b), mk(c, d)), mk(nadd(nmul(a, c), nmul(b, d)), nadd(nmul(a, d), nmul(b, c)))))
    S['Z.sub_def'] = F([Z, Z], lambda x, y: Eq(zsub(x, y), zadd(x, zneg(y))))
    # ---- the commutative-ring laws
    S['Z.add_comm'] = F([Z, Z], lambda x, y: Eq(zadd(x, y), zadd(y, x)))
    S['Z.add_assoc'] = F([Z, Z, Z], lambda x, y, z: Eq(zadd(zadd(x, y), z), zadd(x, zadd(y, z))))
    S['Z.add_zero'] = F([Z], lambda x: Eq(zadd(x, Z0), x))
    S['Z.zero_add'] = F([Z], lambda x: Eq(zadd(Z0, x), x))
    S['Z.add_left_comm'] = F([Z, Z, Z], lambda x, y, z: Eq(zadd(x, zadd(y, z)), zadd(y, zadd(x, z))))
    S['Z.add_right_comm'] = F([Z, Z, Z], lambda x, y, z: Eq(zadd(zadd(x, y), z), zadd(zadd(x, z), y)))
    S['Z.add_neg_cancel'] = F([Z], lambda x: Eq(zadd(x, zneg(x)), Z0))
    S['Z.neg_add_cancel'] = F([Z], lambda x: Eq(zadd(zneg(x), x), Z0))
    S['Z.neg_neg'] = F([Z], lambda x: Eq(zneg(zneg(x)), x))
    S['Z.neg_add'] = F([Z, Z], lambda x, y: Eq(zneg(zadd(x, y)), zadd(zneg(x), zneg(y))))
    S['Z.mul_comm'] = F([Z, Z], lambda x, y: Eq(zmul(x, y), zmul(y, x)))
    S['Z.mul_assoc'] = F([Z, Z, Z], lambda x, y, z: Eq(zmul(zmul(x, y), z), zmul(x, zmul(y, z))))
    S['Z.mul_zero'] = F([Z], lambda x: Eq(zmul(x, Z0), Z0))
    S['Z.zero_mul'] = F([Z], lambda x: Eq(zmul(Z0, x), Z0))
    S['Z.mul_one'] = F([Z], lambda x: Eq(zmul(x, Z1), x))
    S['Z.one_mul'] = F([Z], lambda x: Eq(zmul(Z1, x), x))
    S['Z.mul_left_comm'] = F([Z, Z, Z], lambda x, y, z: Eq(zmul(x, zmul(y, z)), zmul(y, zmul(x, z))))
    S['Z.mul_right_comm'] = F([Z, Z, Z], lambda x, y, z: Eq(zmul(zmul(x, y), z), zmul(zmul(x, z), y)))
    S['Z.mul_add'] = F([Z, Z, Z], lambda x, y, z: Eq(zmul(x, zadd(y, z)), zadd(zmul(x, y), zmul(x, z))))
    S['Z.add_mul'] = F([Z, Z, Z], lambda x, y, z: Eq(zmul(zadd(x, y), z), zadd(zmul(x, z), zmul(y, z))))
    S['Z.mul_neg'] = F([Z, Z], lambda x, y: Eq(zmul(x, zneg(y)), zneg(zmul(x, y))))
    S['Z.neg_mul'] = F([Z, Z], lambda x, y: Eq(zmul(zneg(x), y), zneg(zmul(x, y))))
    # ---- powers
    S['Z.sqr_def'] = F([Z], lambda x: Eq(zsqr(x), zmul(x, x)))
    S['Z.pow_zero'] = F([Z], lambda x: Eq(zpow(x, N0), Z1))
    S['Z.pow_one'] = F([Z], lambda x: Eq(zpow(x, N1), x))
    S['Z.ppow_succ'] = F([Z, Pos], lambda x, q: Eq(zppow(x, psucc(q)), zmul(x, zppow(x, q))))
    S['Z.pow_succ'] = F([Z, N], lambda x, n: Eq(zpow(x, nsucc(n)), zmul(x, zpow(x, n))))
    S['Z.pow_add'] = F([Z, N, N], lambda x, m, n: Eq(zpow(x, nadd(m, n)), zmul(zpow(x, m), zpow(x, n))))
    S['N.induction_on'] = F([w.arrow(N, PROP), N, lambda P, n: P(N0), lambda P, n, h0: F([N, lambda m: P(m)], lambda m, hm: P(nsucc(m)))], lambda P, n, h0, hs: P(n))
    # ---- the naturals inside the integers; cancellation; 0 /= 1
    S['Z.of_N_def'] = F([N], lambda n: Eq(zof(n), mk(n, N0)))
    S['Z.of_N_zero'] = Eq(zof(N0), Z0)
    S['Z.of_N_one'] = Eq(zof(N1), Z1)
    S['Z.of_N_add'] = F([N, N], lambda m, n: Eq(zof(nadd(m, n)), zadd(zof(m), zof(n))))
    S['Z.of_N_mul'] = F([N, N], lambda m, n: Eq(zof(nmul(m, n)), zmul(zof(m), zof(n))))
    S['Z.of_N_succ'] = F([N], lambda n: Eq(zof(nsucc(n)), zadd(zof(n), Z1)))
    S['Z.of_N_inj'] = F([N, N, lambda a, b: Eq(zof(a), zof(b))], lambda a, b, h: Eq(a, b))
    S['Z.add_left_cancel'] = F([Z, Z, Z, lambda x, y, z: Eq(zadd(x, y), zadd(x, z))], lambda x, y, z, h: Eq(y, z))
    S['Z.add_right_cancel'] = F([Z, Z, Z, lambda x, y, z: Eq(zadd(x, z), zadd(y, z))], lambda x, y, z, h: Eq(x, y))
    S['Z.zero_ne_one'] = ~Eq(Z0, Z1)
    # ---- the order:  x <= y  iff  some natural k has  x + k = y;  x < y  iff  x + 1 <= y
    S['Z.le_refl'] = F([Z], lambda x: zle(x, x))
    S['Z.le_trans'] = F([Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h: zle(y, z)], lambda x, y, z, h1, h2: zle(x, z))
    S['Z.le_antisymm'] = F([Z, Z, lambda x, y: zle(x, y), lambda x, y, h: zle(y, x)], lambda x, y, h1, h2: Eq(x, y))
    S['Z.le_of_mk'] = F([N, N, N, N, lambda a, b, c, d: nle(nadd(a, d), nadd(c, b))], lambda a, b, c, d, h: zle(mk(a, b), mk(c, d)))
    S['Z.mk_le'] = F([N, N, N, N, lambda a, b, c, d: zle(mk(a, b), mk(c, d))], lambda a, b, c, d, h: nle(nadd(a, d), nadd(c, b)))
    S['Z.le_total'] = F([Z, Z], lambda x, y: zle(x, y) | zle(y, x))
    S['Z.nonneg_of_N'] = F([N], lambda k: zle(Z0, zof(k)))
    S['Z.add_le_add_right'] = F([Z, Z, Z, lambda x, y, z: zle(x, y)], lambda x, y, z, h: zle(zadd(x, z), zadd(y, z)))
    S['Z.add_le_add_left'] = F([Z, Z, Z, lambda x, y, z: zle(x, y)], lambda x, y, z, h: zle(zadd(z, x), zadd(z, y)))
    S['Z.add_le_add'] = F([Z, Z, Z, Z, lambda x, y, u, v: zle(x, y), lambda x, y, u, v, h: zle(u, v)], lambda x, y, u, v, h1, h2: zle(zadd(x, u), zadd(y, v)))
    S['Z.nonneg_exists'] = F([Z, lambda x: zle(Z0, x)], lambda x, h: Ex(N, w.lam(N, lambda k: Eq(zof(k), x))))
    S['Z.mul_nonneg'] = F([Z, Z, lambda x, y: zle(Z0, x), lambda x, y, h: zle(Z0, y)], lambda x, y, h1, h2: zle(Z0, zmul(x, y)))
    S['Z.le_of_le_eq'] = F([Z, Z, Z, lambda a, b, c: zle(a, b), lambda a, b, c, h: Eq(b, c)], lambda a, b, c, h1, h2: zle(a, c))
    S['Z.le_of_eq_le'] = F([Z, Z, Z, lambda a, b, c: Eq(a, b), lambda a, b, c, h: zle(b, c)], lambda a, b, c, h1, h2: zle(a, c))
    S['Z.sub_nonneg_of_le'] = F([Z, Z, lambda x, y: zle(x, y)], lambda x, y, h: zle(Z0, zsub(y, x)))
    S['Z.le_of_sub_nonneg'] = F([Z, Z, lambda x, y: zle(Z0, zsub(y, x))], lambda x, y, h: zle(x, y))
    S['Z.neg_le_neg'] = F([Z, Z, lambda x, y: zle(x, y)], lambda x, y, h: zle(zneg(y), zneg(x)))
    S['Z.add_nonneg'] = F([Z, Z, lambda x, y: zle(Z0, x), lambda x, y, h: zle(Z0, y)], lambda x, y, hx, hy: zle(Z0, zadd(x, y)))
    S['Z.sq_nonneg'] = F([Z], lambda x: zle(Z0, zmul(x, x)))
    S['Z.mul_le_mul_of_nonneg_left'] = F([Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h: zle(Z0, z)], lambda x, y, z, h, hz: zle(zmul(z, x), zmul(z, y)))
    S['Z.mul_le_mul_of_nonneg_right'] = F([Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h: zle(Z0, z)], lambda x, y, z, h, hz: zle(zmul(x, z), zmul(y, z)))
    S['Z.le_succ'] = F([Z], lambda x: zle(x, zadd(x, Z1)))
    S['Z.le_of_lt'] = F([Z, Z, lambda x, y: zlt(x, y)], lambda x, y, h: zle(x, y))
    S['Z.lt_irrefl'] = F([Z, lambda x: zlt(x, x)], lambda x, h: Fls)
    S['Z.lt_of_lt_of_le'] = F([Z, Z, Z, lambda x, y, z: zlt(x, y), lambda x, y, z, h: zle(y, z)], lambda x, y, z, h1, h2: zlt(x, z))
    S['Z.lt_of_le_of_lt'] = F([Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h: zlt(y, z)], lambda x, y, z, h1, h2: zlt(x, z))
    S['Z.lt_trans'] = F([Z, Z, Z, lambda x, y, z: zlt(x, y), lambda x, y, z, h: zlt(y, z)], lambda x, y, z, h1, h2: zlt(x, z))
    S['N.zero_or_succ'] = F([N], lambda k: Eq(k, N0) | Ex(N, w.lam(N, lambda j: Eq(k, nsucc(j)))))
    S['Z.lt_or_eq_of_le'] = F([Z, Z, lambda x, y: zle(x, y)], lambda x, y, h: zlt(x, y) | Eq(x, y))
    S['Z.le_or_lt'] = F([Z, Z], lambda x, y: zle(x, y) | zlt(y, x))
    S['Z.not_le_of_lt'] = F([Z, Z, lambda x, y: zlt(x, y)], lambda x, y, h: ~zle(y, x))
    S['Z.not_lt_of_le'] = F([Z, Z, lambda x, y: zle(x, y)], lambda x, y, h: ~zlt(y, x))
    S['Z.le_of_not_lt'] = F([Z, Z, lambda x, y: ~zlt(x, y)], lambda x, y, h: zle(y, x))
    S['Z.lt_of_not_le'] = F([Z, Z, lambda x, y: ~zle(x, y)], lambda x, y, h: zlt(y, x))
    S['Z.ne_of_lt'] = F([Z, Z, lambda x, y: zlt(x, y)], lambda x, y, h: ~Eq(x, y))
    S['Z.eq_dec'] = F([Z, Z], lambda x, y: Eq(x, y) | ~Eq(x, y))
    S['Z.lt_trichotomy'] = F([Z, Z], lambda x, y: zlt(x, y) | (Eq(x, y) | zlt(y, x)))
    S['Z.zero_lt_one'] = zlt(Z0, Z1)
    # ---- the strict order with sums and products; discreteness; the integral domain; the casts of naturals
    S['Z.le_of_eq'] = F([Z, Z, lambda x, y: Eq(x, y)], lambda x, y, h: zle(x, y))
    S['Z.lt_add_one'] = F([Z], lambda x: zlt(x, zadd(x, Z1)))
    S['Z.add_lt_add_right'] = F([Z, Z, Z, lambda x, y, z: zlt(x, y)], lambda x, y, z, h: zlt(zadd(x, z), zadd(y, z)))
    S['Z.add_lt_add_left'] = F([Z, Z, Z, lambda x, y, z: zlt(x, y)], lambda x, y, z, h: zlt(zadd(z, x), zadd(z, y)))
    S['Z.add_lt_add'] = F([Z, Z, Z, Z, lambda x, y, u, v: zlt(x, y), lambda x, y, u, v, h: zlt(u, v)], lambda x, y, u, v, h1, h2: zlt(zadd(x, u), zadd(y, v)))
    S['Z.add_lt_add_of_lt_of_le'] = F([Z, Z, Z, Z, lambda x, y, u, v: zlt(x, y), lambda x, y, u, v, h: zle(u, v)], lambda x, y, u, v, h1, h2: zlt(zadd(x, u), zadd(y, v)))
    S['Z.add_lt_add_of_le_of_lt'] = F([Z, Z, Z, Z, lambda x, y, u, v: zle(x, y), lambda x, y, u, v, h: zlt(u, v)], lambda x, y, u, v, h1, h2: zlt(zadd(x, u), zadd(y, v)))
    S['Z.pos_of_neg'] = F([Z, lambda x: zlt(x, Z0)], lambda x, h: zlt(Z0, zneg(x)))
    S['Z.neg_of_pos'] = F([Z, lambda x: zlt(Z0, x)], lambda x, h: zlt(zneg(x), Z0))
    S['Z.neg_lt_neg'] = F([Z, Z, lambda x, y: zlt(x, y)], lambda x, y, h: zlt(zneg(y), zneg(x)))
    S['Z.sub_pos_of_lt'] = F([Z, Z, lambda x, y: zlt(x, y)], lambda x, y, h: zlt(Z0, zsub(y, x)))
    S['Z.lt_of_sub_pos'] = F([Z, Z, lambda x, y: zlt(Z0, zsub(y, x))], lambda x, y, h: zlt(x, y))
    S['Z.le_of_lt_add_one'] = F([Z, Z, lambda x, y: zlt(x, zadd(y, Z1))], lambda x, y, h: zle(x, y))
    S['Z.lt_of_le_of_ne'] = F([Z, Z, lambda x, y: zle(x, y), lambda x, y, h: ~Eq(x, y)], lambda x, y, h, hne: zlt(x, y))
    S['Z.mul_pos'] = F([Z, Z, lambda x, y: zlt(Z0, x), lambda x, y, h: zlt(Z0, y)], lambda x, y, hx, hy: zlt(Z0, zmul(x, y)))
    S['Z.mul_lt_mul_of_pos_left'] = F([Z, Z, Z, lambda x, y, z: zlt(x, y), lambda x, y, z, h: zlt(Z0, z)], lambda x, y, z, h, hz: zlt(zmul(z, x), zmul(z, y)))
    S['Z.mul_lt_mul_of_pos_right'] = F([Z, Z, Z, lambda x, y, z: zlt(x, y), lambda x, y, z, h: zlt(Z0, z)], lambda x, y, z, h, hz: zlt(zmul(x, z), zmul(y, z)))
    S['Z.mul_pos_ne_zero'] = F([Z, Z, lambda x, y: zlt(Z0, x), lambda x, y, h: zlt(Z0, y), lambda x, y, h, h2: Eq(zmul(x, y), Z0)], lambda x, y, hx, hy, h: Fls)
    S['Z.mul_eq_zero'] = F([Z, Z, lambda x, y: Eq(zmul(x, y), Z0)], lambda x, y, h: Eq(x, Z0) | Eq(y, Z0))
    S['Z.mul_left_cancel'] = F([Z, Z, Z, lambda x, y, z: ~Eq(x, Z0), lambda x, y, z, h: Eq(zmul(x, y), zmul(x, z))], lambda x, y, z, hx, h: Eq(y, z))
    S['Z.mul_right_cancel'] = F([Z, Z, Z, lambda x, y, z: ~Eq(z, Z0), lambda x, y, z, h: Eq(zmul(x, z), zmul(y, z))], lambda x, y, z, hz, h: Eq(x, y))
    S['Z.of_N_le'] = F([N, N, lambda a, b: nle(a, b)], lambda a, b, h: zle(zof(a), zof(b)))
    S['Z.le_of_of_N_le'] = F([N, N, lambda a, b: zle(zof(a), zof(b))], lambda a, b, h: nle(a, b))
    S['Z.of_N_lt'] = F([N, N, lambda a, b: nlt(a, b)], lambda a, b, h: zlt(zof(a), zof(b)))
    S['Z.lt_of_of_N_lt'] = F([N, N, lambda a, b: zlt(zof(a), zof(b))], lambda a, b, h: nlt(a, b))
    S['Z.not_nonneg_neg'] = F([Pos, lambda q: zle(Z0, zneg(zof(w['Npos'](q))))], lambda q, h: Fls)
    # ---- the Bool-valued equality tests behind `Z.mk_inj`, and the lift of a function of two arguments out of a quotient (the library's own, quot_lib.py)
    Bool, BT, BF = w['Bool'], w['Bool.true'], w['Bool.false']; pbeq, nbeq = w['Pos.beq'], w['N.beq']
    S['Bool.false_ne_true'] = F([Eq(BF, BT)], lambda h: Fls)
    S['Pos.beq_refl'] = F([Pos], lambda x: Eq(pbeq(x, x), BT))
    S['Pos.beq_eq'] = F([Pos, Pos, lambda x, y: Eq(pbeq(x, y), BT)], lambda x, y, h: Eq(x, y))
    S['N.beq_refl'] = F([N], lambda x: Eq(nbeq(x, x), BT))
    S['N.beq_eq'] = F([N, N, lambda x, y: Eq(nbeq(x, y), BT)], lambda x, y, h: Eq(x, y))
    S['N.eq_beq'] = F([N, N, lambda x, y: Eq(x, y)], lambda x, y, h: Eq(nbeq(x, y), BT))
    S['Bool.ext'] = F([Bool, Bool, lambda a, b: w.arrow(Eq(a, BT), Eq(b, BT)), lambda a, b, h: w.arrow(Eq(b, BT), Eq(a, BT))], lambda a, b, h1, h2: Eq(a, b))
    Quot, MKq, LIFT2 = w['Quot'], w['Quot.mk'], w['Quot.lift2']
    lift2_doms = [TYPE, lambda A: w.arrow(A, w.arrow(A, PROP)), lambda A, r: TYPE, lambda A, r, B: w.arrow(A, w.arrow(A, B)), lambda A, r, B, f: F([A], lambda a: r(a, a)),
                  lambda A, r, B, f, hr: F([A, A, A, A, lambda a, a2, b, b2: r(a, a2), lambda a, a2, b, b2, h1: r(b, b2)], lambda a, a2, b, b2, h1, h2: Eq(f(a, b), f(a2, b2)))]
    S['Quot.lift2_mk'] = F(lift2_doms + [lambda A, r, B, f, hr, h: A, lambda A, r, B, f, hr, h, a: A],
                           lambda A, r, B, f, hr, h, a, b: Eq(LIFT2(A, r, B, f, hr, h, MKq(A, r, a), MKq(A, r, b)), f(a, b)))
    return S


# the plumbing of the quotient construction: the well-definedness of the operations on representatives, and the Bool test behind `Z.mk_inj` (checked by the kernel; no user ever states them)
PLUMBING = {'Z.shift', 'Z.test_wd', 'Z.test', 'Z.neg_wd', 'Z.add_wd', 'Z.mul_wd1', 'Z.mul_wd2', 'Z.mul_wd'}
# the definitions (not theorems): their types are in `definition_types`, their behaviour in the conversion checks
DEFINITIONS = {'Z', 'Zrel', 'Z.mk', 'Z.neg', 'Z.add', 'Z.mul', 'Z.zero', 'Z.one', 'Z.sub', 'Z.of_N', 'Z.sqr', 'Z.ppow', 'Z.pow', 'Z.le', 'Z.lt', 'funext', 'Quot.lift2'}


def definition_types(w):
    TYPE, PROP = w.TYPE, w.PROP; F = w.forall_; Eq = w.Eq
    N, Pos, Z, Prod, Bool = w['N'], w['Pos'], w['Z'], w['Prod'], w['Bool']; NN = Prod(N, N)
    A = w.arrow
    return {'Z': TYPE, 'Zrel': A(NN, A(NN, PROP)), 'Z.mk': A(N, A(N, Z)), 'Z.neg': A(Z, Z), 'Z.add': A(Z, A(Z, Z)), 'Z.mul': A(Z, A(Z, Z)), 'Z.zero': Z, 'Z.one': Z, 'Z.sub': A(Z, A(Z, Z)),
            'Z.of_N': A(N, Z), 'Z.sqr': A(Z, Z), 'Z.ppow': A(Z, A(Pos, Z)), 'Z.pow': A(Z, A(N, Z)), 'Z.le': A(Z, A(Z, PROP)), 'Z.lt': A(Z, A(Z, PROP)), 'Z.test': A(N, A(N, A(Z, Bool))),
            'Quot.lift2': F([TYPE, lambda T: A(T, A(T, PROP)), lambda T, r: TYPE, lambda T, r, B: A(T, A(T, B)), lambda T, r, B, f: F([T], lambda a: r(a, a)),
                                lambda T, r, B, f, hr: F([T, T, T, T, lambda a, a2, b, b2: r(a, a2), lambda a, a2, b, b2, h1: r(b, b2)], lambda a, a2, b, b2, h1, h2: Eq(f(a, b), f(a2, b2)))],
                               lambda T, r, B, f, hr, h: A(w['Quot'](T, r), A(w['Quot'](T, r), B))),
            'funext': F([TYPE, lambda T: A(T, TYPE), lambda T, B: F([T], lambda x: B(x)), lambda T, B, f: F([T], lambda x: B(x)), lambda T, B, f, g: F([T], lambda x: Eq(f(x), g(x)))], lambda T, B, f, g, h: Eq(f, g))}


# ---------------------------------------------------------------------------------------------------------------------- Python side: integer polynomials and expression trees
def pscale(p, c): return {m: c * v for m, v in p.items()}

def zpoly(e, natoms):
    """expression -> {exponent vector: signed coefficient}; the atoms are x_0..x_{n-1} and g x_0 .. g x_{n-1} for a function variable g; `sqr x_i` is the square x_i * x_i (what the display normal form makes of x_i ^ 2)"""
    n = 2 * natoms; k = e[0]
    if k == 'x': v = [0] * n; v[e[1]] = 1; return {tuple(v): 1}
    if k == 'sqr': v = [0] * n; v[e[1]] = 2; return {tuple(v): 1}
    if k == 'app': v = [0] * n; v[natoms + e[1]] = 1; return {tuple(v): 1}
    if k == 'n': return {tuple([0] * n): e[1]} if e[1] else {}
    if k == '+': return padd(zpoly(e[1], natoms), zpoly(e[2], natoms))
    if k == '-': return padd(zpoly(e[1], natoms), pscale(zpoly(e[2], natoms), -1))
    if k == '*': return pmul(zpoly(e[1], natoms), zpoly(e[2], natoms))
    if k == 'neg': return pscale(zpoly(e[1], natoms), -1)
    if k == '^': return ppow(zpoly(e[1], natoms), e[2], n)
    raise ValueError(k)

def zclean(p): return {m: c for m, c in p.items() if c}

def terms_bound(e):
    """an upper bound on the number of monomials of the expanded expression (so that a power tower of sums is not asked for: the proof of it would be millions of steps)"""
    k = e[0]
    if k in ('x', 'sqr', 'app', 'n'): return 1
    if k == 'neg': return terms_bound(e[1])
    if k in '+-': return terms_bound(e[1]) + terms_bound(e[2])
    if k == '*': return terms_bound(e[1]) * terms_bound(e[2])
    if k == '^': return terms_bound(e[1]) ** max(e[2], 1)
    raise ValueError(k)

def zshow(e):
    k = e[0]
    if k == 'x': return 'xyz'[e[1]]
    if k == 'sqr': return 'sqr(%s)' % 'xyz'[e[1]]
    if k == 'app': return 'g(%s)' % 'xyz'[e[1]]
    if k == 'n': return str(e[1])
    if k == 'neg': return '-(%s)' % zshow(e[1])
    if k == '^': return '(%s)^%d' % (zshow(e[1]), e[2])
    return '(%s %s %s)' % (zshow(e[1]), k, zshow(e[2]))

def rand_num(rng):
    c = rng.random(); s = rng.choice([-1, 1, 1])
    if c < 0.5: return s * rng.randrange(0, 8)
    if c < 0.8: return s * rng.randrange(0, 1 << 8)
    if c < 0.95: return s * rng.randrange(0, 1 << 24)
    return s * rng.randrange(0, 1 << 40)

def small_num(rng): return rng.choice([-1, 1]) * rng.randrange(0, 6)

def rand_expr(rng, natoms, depth, numeral=rand_num):
    if depth == 0 or rng.random() < 0.22:
        c = rng.random()
        if c < 0.55: return ('x', rng.randrange(natoms))
        if c < 0.82: return ('n', numeral(rng))
        if c < 0.92: return ('app', rng.randrange(natoms))
        return ('sqr', rng.randrange(natoms))
    r = rng.random()
    if r < 0.28: return ('+', rand_expr(rng, natoms, depth - 1, numeral), rand_expr(rng, natoms, depth - 1, numeral))
    if r < 0.5: return ('-', rand_expr(rng, natoms, depth - 1, numeral), rand_expr(rng, natoms, depth - 1, numeral))
    if r < 0.78: return ('*', rand_expr(rng, natoms, depth - 1, numeral), rand_expr(rng, natoms, depth - 1, numeral))
    if r < 0.9: return ('^', rand_expr(rng, natoms, depth - 1, numeral), rng.randrange(0, 4))
    return ('neg', rand_expr(rng, natoms, depth - 1, numeral))

def equiv(e, rng):
    """a rewriting of e with the same value for every assignment of the atoms"""
    k = e[0]; r = rng.random()
    if k in '+*':
        a, b = e[1], e[2]
        if r < 0.3: return (k, equiv(b, rng), equiv(a, rng))
        if r < 0.45 and a[0] == k: return (k, a[1], (k, a[2], b))
        if r < 0.55 and k == '*' and b[0] == '+': return ('+', ('*', a, b[1]), ('*', a, b[2]))
        if r < 0.65 and k == '*' and a[0] == '+': return ('+', ('*', a[1], b), ('*', a[2], b))
        if r < 0.7 and k == '*' and b[0] == '-': return ('-', ('*', a, b[1]), ('*', a, b[2]))
        if r < 0.75 and k == '*' and a[0] == '-': return ('-', ('*', a[1], b), ('*', a[2], b))
        if r < 0.8 and k == '+': return ('-', equiv(a, rng), ('neg', equiv(b, rng)))
        return (k, equiv(a, rng), equiv(b, rng))
    if k == '-':
        a, b = e[1], e[2]
        if r < 0.25: return ('+', equiv(a, rng), ('neg', equiv(b, rng)))
        if r < 0.4: return ('+', equiv(a, rng), ('*', ('n', -1), equiv(b, rng)))
        if r < 0.55: return ('neg', ('-', equiv(b, rng), equiv(a, rng)))
        return ('-', equiv(a, rng), equiv(b, rng))
    if k == 'neg':
        a = e[1]
        if r < 0.25: return ('*', ('n', -1), equiv(a, rng))
        if r < 0.4: return ('-', ('n', 0), equiv(a, rng))
        if r < 0.5 and a[0] == 'neg': return equiv(a[1], rng)
        if r < 0.6: return ('neg', ('neg', ('neg', equiv(a, rng))))
        return ('neg', equiv(a, rng))
    if k == '^':
        a, n = e[1], e[2]
        if r < 0.3 and n >= 1: return ('*', equiv(a, rng), ('^', a, n - 1))
        if r < 0.4 and n == 0: return ('n', 1)
        if r < 0.5 and a[0] == '*': return ('*', ('^', a[1], n), ('^', a[2], n))
        if r < 0.55 and n == 2: return ('*', equiv(a, rng), equiv(a, rng))
        return ('^', equiv(a, rng), n)
    if k == 'n':
        n = e[1]
        if r < 0.25: return ('+', ('n', n - 1), ('n', 1))
        if r < 0.4: return ('-', ('n', n + 3), ('n', 3))
        if r < 0.5: return ('neg', ('n', -n))
        if r < 0.6: return ('*', ('n', n), ('n', 1))
        return e
    return e

def perturb(e, rng, natoms):
    """a small change at a random node (it may or may not change the polynomial)"""
    k = e[0]
    if k in '+-*' and rng.random() < 0.75:
        if rng.random() < 0.5: return (k, perturb(e[1], rng, natoms), e[2])
        return (k, e[1], perturb(e[2], rng, natoms))
    if k == '^':
        if rng.random() < 0.5: return ('^', e[1], max(0, e[2] + rng.choice([-1, 1])))
        return ('^', perturb(e[1], rng, natoms), e[2])
    if k == 'neg': return ('neg', perturb(e[1], rng, natoms)) if rng.random() < 0.5 else e[1]
    if k == 'n': return ('n', e[1] + rng.choice([-1, 1]))
    if k == 'x': return ('x', rng.randrange(natoms)) if rng.random() < 0.5 else ('n', 1)
    if k in ('sqr', 'app'): return ('x', e[1])
    if k in '+-*': return (rng.choice([c for c in '+-*' if c != k]), e[1], e[2])
    return e


class Syntax:
    """builds expression trees over Z; a numeral gets one of its spellings (the canonical `lit`, or `Z.of_N` / `Z.neg (Z.of_N ..)` with the raw N numeral -- also for zero)"""
    def __init__(self, w, rng=None):
        self.w = w; self.rng = rng; self.inst = R.instance_for(w, w['Z']); self.nlit = self.inst.N.lit
        self.add, self.mul, self.neg, self.sub, self.sqr, self.pw, self.of_N = (w[n] for n in ('Z.add', 'Z.mul', 'Z.neg', 'Z.sub', 'Z.sqr', 'Z.pow', 'Z.of_N'))
    def numeral(self, k):
        if self.rng is not None and self.rng.random() < 0.3:
            return self.of_N(self.nlit(k)) if k >= 0 else self.neg(self.of_N(self.nlit(-k)))
        return self.inst.lit(k)
    def build(self, e, xs, g=None):
        """the term of the expression over the variables xs; `g` is the function variable that the nodes ('app', i) apply to xs[i]"""
        k = e[0]
        if k == 'x': return xs[e[1]]
        if k == 'sqr': return self.sqr(xs[e[1]])
        if k == 'app':
            if g is None: raise ValueError('an application node needs the function variable')
            return g(xs[e[1]])
        if k == 'n': return self.numeral(e[1])
        if k == 'neg': return self.neg(self.build(e[1], xs, g))
        if k == '^': return self.pw(self.build(e[1], xs, g), self.nlit(e[2]))
        a, b = self.build(e[1], xs, g), self.build(e[2], xs, g)
        return self.add(a, b) if k == '+' else self.sub(a, b) if k == '-' else self.mul(a, b)


def read_z(w, inst_n, t):
    """the integer a closed term of Z evaluates to: the kernel's normal form is `Quot.mk _ _ (pair N N a b)` with binary numerals a, b, and the integer is a - b"""
    nt = w.normalize(t); head, args = w.spine(nt)
    if w.head_name(nt) != 'Quot.mk': return None
    h2, args2 = w.spine(args[-1])
    if w.head_name(args[-1]) != 'pair' or len(args2) != 4: return None
    a, b = inst_n.numeral(args2[2]), inst_n.numeral(args2[3])
    return None if a is None or b is None else a - b


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    t_start = time.time(); rng = random.Random(30)
    t0 = time.time(); w0 = bin_lib.build_bin_defs(); n_base = w0.k.num_globals()
    w = int_lib.build_int(w0); k = w.k; tb = time.time() - t0; lib_end = k.num_globals()
    indep = set()                       # the theorems proved here that the independent checker re-checks as well (the whole library is re-checked in any case)
    TYPE, PROP = w.TYPE, w.PROP; Eq = w.Eq; F = w.forall_
    N, Pos, Z = w['N'], w['Pos'], w['Z']; N0, N1 = w['N0'], w['Npos'](w['XH'])
    zadd, zmul, zneg, zsub, zpow, zof, zsqr, mk = (w[n] for n in ('Z.add', 'Z.mul', 'Z.neg', 'Z.sub', 'Z.pow', 'Z.of_N', 'Z.sqr', 'Z.mk'))
    Z0, Z1 = w['Z.zero'], w['Z.one']; zle, zlt = w['Z.le'], w['Z.lt']
    nadd, nmul = w['N.add'], w['N.mul']
    inst = R.instance_for(w, Z); inst_n = R.instance_for(w, N)
    expect(isinstance(inst, R.SemiringZ) and isinstance(inst_n, R.SemiringN), 'A  the instances of Z and N are found')
    SY = Syntax(w); ZZ = w.arrow(Z, Z)

    # ------------------------------------------------------------------------------------------------------------------ A. statements
    spec = zspec(w); n_ok = 0
    for name, stmt in spec.items():
        if expect(name in w, 'A  the library declares %s' % name):
            n_ok += expect(w.global_type(name) == stmt, 'A  %s has the intended statement' % name)
            expect(k.global_info(w.gid[name])[2] is not None, 'A  %s is a theorem (it has a value)' % name)
    I_z = dict(T=Z, ET=N, Z=Z0, one=Z1, add=zadd, mul=zmul, pow=zpow, eadd=nadd, succ=lambda x: zadd(x, Z1), ez=N0, eo=N1, pfx='Z.ring_')       # (Z has no `succ` lemma: the entry is dropped below)
    rspec = {n: s for n, s in ring_spec(w, I_z).items() if n != 'succ'}
    m1 = zneg(zof(N1))
    rspec['add_cancel'] = F([Z] * 5 + [lambda a, b, A, B, C: Eq(zadd(a, b), Z0), lambda a, b, A, B, C, h: Eq(zadd(A, B), C)], lambda a, b, A, B, C, h1, h2: Eq(zadd(zadd(a, A), zadd(b, B)), C))
    rspec['neg'] = F([Z], lambda x: Eq(zneg(x), zmul(m1, x)))
    rspec['sub'] = F([Z, Z], lambda x, y: Eq(zsub(x, y), zadd(x, zmul(m1, y))))
    for name, stmt in rspec.items():
        have = 'Z.ring_' + name
        if expect(have in w, 'A  Z declares %s' % have): n_ok += expect(w.global_type(have) == stmt, 'A  %s has the intended statement' % have)
    extra = [n for n in w.gid if n.startswith('Z.ring_') and n[len('Z.ring_'):] not in rspec]
    expect(not extra, 'A  the Z ring lemma set has lemmas the table does not know: %s' % extra[:5])
    # nothing in the integer part of the world escapes the tables
    known = set(spec) | {'Z.ring_' + n for n in rspec} | PLUMBING | DEFINITIONS | {'Bool', 'Bool.false', 'Bool.true'}
    unknown = [k.global_info(g)[0] for g in range(n_base, k.num_globals()) if k.global_info(g)[0] not in known and not k.global_info(g)[0].startswith('__same_ctor_') and k.global_info(g)[0] not in ('Pos.beq', 'N.beq', 'Quot', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound', 'Prod', 'pair', 'fst', 'snd')]
    expect(not unknown, 'A  the integer part of the world declares things the tables do not cover: %s' % unknown[:8])
    for name, ty in definition_types(w).items():
        expect(w.global_type(name) == ty, 'A  %s has the intended type' % name)
    # the definitions compute as they should (conversion checks, closed instances; the last column is the negative control)
    def nl(v): return inst_n.lit(v)
    def M(a, b): return mk(nl(a), nl(b))
    Prod = w['Prod']; NN = Prod(N, N); pair = w['pair']
    conv_yes = [
        ('Z unfolds to the quotient', Z, w['Quot'](NN, w['Zrel'])),
        ('Z.zero is Z.mk 0 0', Z0, M(0, 0)), ('Z.one is Z.mk 1 0', Z1, M(1, 0)), ('Z.of_N 7 is Z.mk 7 0', zof(nl(7)), M(7, 0)),
        ('Z.neg (mk 3 5) is mk 5 3', zneg(M(3, 5)), M(5, 3)),
        ('Z.add (mk 3 5) (mk 10 2) is mk 13 7', zadd(M(3, 5), M(10, 2)), M(13, 7)),
        ('Z.mul (mk 3 5) (mk 10 2) is mk (3*10 + 5*2) (3*2 + 5*10)', zmul(M(3, 5), M(10, 2)), mk(nadd(nmul(nl(3), nl(10)), nmul(nl(5), nl(2))), nadd(nmul(nl(3), nl(2)), nmul(nl(5), nl(10))))),
        ('Z.sub x y is x + -y', zsub(M(3, 5), M(10, 2)), zadd(M(3, 5), zneg(M(10, 2)))),
        ('Z.pow x 0 is 1', zpow(M(2, 5), nl(0)), Z1), ('Z.pow x 1 is x', zpow(M(2, 5), nl(1)), M(2, 5)),
        ('Z.pow x 2 is x * x', zpow(M(2, 5), nl(2)), zmul(M(2, 5), M(2, 5))), ('Z.pow x 3 is x * (x * x)', zpow(M(2, 5), nl(3)), zmul(M(2, 5), zmul(M(2, 5), M(2, 5)))),
        ('Z.pow x 6 is (x*x*x)^2', zpow(M(2, 5), nl(6)), zsqr(zmul(M(2, 5), zmul(M(2, 5), M(2, 5))))),
        ('Z.sqr x is x * x', zsqr(M(2, 5)), zmul(M(2, 5), M(2, 5))),
        ('Z.le x y unfolds to the witness', zle(M(1, 2), M(4, 4)), w['Ex'](N, w.lam(N, lambda j: Eq(zadd(M(1, 2), zof(j)), M(4, 4))))),
        ('Z.lt x y is Z.le (x + 1) y', w['Z.lt'](M(1, 2), M(4, 4)), zle(zadd(M(1, 2), Z1), M(4, 4))),
        ('Zrel on pairs is the cross-sum equation', w['Zrel'](pair(N, N, nl(3), nl(5)), pair(N, N, nl(4), nl(6))), Eq(nadd(nl(3), nl(6)), nadd(nl(4), nl(5)))),
    ]
    conv_no = [
        ('Z.mk 1 0 and Z.mk 2 1 are different terms (the quotient is not collapsed by conversion)', M(1, 0), M(2, 1)), ('Z.zero and Z.one differ', Z0, Z1),
        ('Z.lt is not Z.le', w['Z.lt'](M(1, 2), M(4, 4)), zle(M(1, 2), M(4, 4))), ('Z.sub x y is not Z.add y (-x)', zsub(M(3, 5), M(10, 2)), zadd(M(10, 2), zneg(M(3, 5)))),
        ('Z.add is not Z.mul', zadd(M(3, 5), M(10, 2)), zmul(M(3, 5), M(10, 2))), ('Z.neg x is not x', zneg(M(3, 5)), M(3, 5)),
        ('Z.pow x 3 is not x * x', zpow(M(2, 5), nl(3)), zmul(M(2, 5), M(2, 5))), ('Z.of_N is not Z.neg . Z.of_N', zof(nl(7)), zneg(zof(nl(7)))),
    ]
    for label, a, b in conv_yes: expect(w.conv(a, b), 'A  conversion: ' + label)
    for label, a, b in conv_no: expect(not w.conv(a, b), 'A  no conversion: ' + label)
    # no axioms: a global without a value is an inductive type, a constructor or one of the five quotient primitives
    axioms = [k.global_info(g)[0] for g in range(k.num_globals()) if audit.kind(k, g) == 'axiom']
    expect(axioms == [], 'A  the integer world has NO axiom (found %s)' % axioms[:5])
    memo = {}; deps_bad = []; quots_seen = set()
    for name in list(spec) + ['Z.ring_' + n for n in rspec] + ['funext', 'Z.add', 'Z.mul', 'Z.neg', 'Z.le']:
        ax, qs = audit.dependencies_of_global(k, w.gid[name], memo)
        if ax or not qs <= {'Quot', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound'}: deps_bad.append((name, sorted(ax), sorted(qs)))
        quots_seen |= qs
    expect(not deps_bad, 'A  every theorem of the integer part rests on the quotient primitives only: %s' % deps_bad[:3])
    ax_fe, qs_fe = audit.dependencies_of_global(k, w.gid['funext'], memo)
    expect(ax_fe == set() and 'Quot.sound' in qs_fe, 'A  funext is a theorem of the kernel plus Quot.sound (axioms %s, quotient constants %s)' % (sorted(ax_fe), sorted(qs_fe)))
    expect(quots_seen == {'Quot', 'Quot.mk', 'Quot.lift', 'Quot.ind', 'Quot.sound'}, 'A  the library uses all five quotient primitives (%s)' % sorted(quots_seen))
    # funext proves a function equality that is not definitional
    f1 = w.fun([N], lambda x: nadd(x, N0)); f2 = w.fun([N], lambda x: x)
    expect(not w.conv(f1, f2), 'A  (fun x => x + 0) and (fun x => x) are not convertible')
    w.prove('s30_fe', Eq(f1, f2), lambda p: p.exact(lambda q: w['funext'](N, w.lam(N, lambda _x: N), f1, f2, w.fun([N], lambda x: w['N.add_zero'](x)))))
    expect('s30_fe' in w, 'A  funext proves (fun x => x + 0) = (fun x => x)')
    ta = time.time() - t_start
    trace('section A done at %.1f s' % ta)

    # ------------------------------------------------------------------------------------------------------------------ B. numerals and arithmetic on numerals
    zn = w.nota.by_type[Z.n]
    def binary_numeral(v):
        if v == 0: return N0
        p = w['XH']
        for bit in bin(v)[3:]: p = (w['XI'] if bit == '1' else w['XO'])(p)
        return w['Npos'](p)
    def canonical(v):
        """written out here: Z.zero, Z.of_N of the binary numeral, or its negation"""
        if v == 0: return Z0
        return zof(binary_numeral(v)) if v > 0 else zneg(zof(binary_numeral(-v)))
    values = list(range(-12, 13)) + [s * x for s in (1, -1) for x in (63, 64, 65, 127, 128, 255, 256, 1000, 2**31 - 1, 2**31, 2**32, 2**63, 2**64 - 1, 2**64, 10**18, 10**30 + 7, 2**100, 2**200 + 12345)]
    values += [rng.choice([-1, 1]) * rng.randrange(0, 1 << rng.choice([5, 20, 70])) for _ in range(30)]
    for v in values:
        t = inst.lit(v)
        expect(t == canonical(v), 'B  lit(%d) is the canonical numeral' % v)
        expect(inst.numeral(t) == v, 'B  numeral(lit(%d)) reads back' % v)
        expect(inst.numeral(zn.lit(v)) == v, 'B  numeral reads the notation spelling of %d' % v)
        expect(w.normalize(t) == w.normalize(zn.lit(v)) or read_z(w, inst_n, t) == v, 'B  the two spellings of %d denote one integer' % v)
        expect(read_z(w, inst_n, t) == v, 'B  the kernel evaluates lit(%d) to the integer %d (a - b of its representative)' % (v, v))
    nonnum = {
        '1 + 1': zadd(Z1, Z1), '--1': zneg(zneg(zof(N1))), '-0 (negated Z.zero)': zneg(Z0), '-1 (negated Z.one)': zneg(Z1), 'Z.mk 1 0': mk(N1, N0), 'Z.of_N (N.succ 0)': zof(w['N.succ'](N0)),
        'Z.of_N (1 + 1)': zof(nadd(N1, N1)), '3 - 1': zsub(inst.lit(3), Z1), 'sqr 2': zsqr(inst.lit(2)), '2 * 2': zmul(inst.lit(2), inst.lit(2)), '2 ^ 2': zpow(inst.lit(2), inst.lit(2)),
        'a bound variable': w.T(k.mk_bvar(0)), 'a natural number': inst_n.lit(5), 'the pair': pair(N, N, N1, N0),
    }
    for label, t in nonnum.items(): expect(inst.numeral(w.T(t) if not hasattr(t, 'n') else t) is None, 'B  numeral() refuses %s' % label)
    # the spellings that ARE numerals: zero in four ways, the negation of a spelled zero
    for label, t, v in (('Z.zero', Z0, 0), ('Z.one', Z1, 1), ('Z.of_N 0', zof(N0), 0), ('-(Z.of_N 0)', zneg(zof(N0)), 0), ('Z.of_N 1', zof(N1), 1), ('-(Z.of_N 5)', zneg(zof(nl(5))), -5)):
        expect(inst.numeral(t) == v, 'B  numeral() reads %s as %d' % (label, v))
    # arithmetic: the kernel's evaluation is Python's
    pairs = [(rng.choice(values), rng.choice(values)) for _ in range(120)] + [(0, 0), (0, -5), (-5, 0), (1, -1), (-1, -1), (7, -7), (2**64, -2**64), (-2**70, 2**70 + 1)]
    n_ar = 0
    for a, b in pairs:
        ta_, tb_ = inst.lit(a), inst.lit(b)
        for op, py, term in (('+', a + b, zadd(ta_, tb_)), ('-', a - b, zsub(ta_, tb_)), ('*', a * b, zmul(ta_, tb_))):
            got = read_z(w, inst_n, term); n_ar += 1
            expect(got == py, 'B  the kernel computes %d %s %d = %d (got %s)' % (a, op, b, py, got))
    for a in values[:40]:
        expect(read_z(w, inst_n, zneg(inst.lit(a))) == -a, 'B  the kernel computes -(%d)' % a)
    for a in (-7, -2, -1, 0, 1, 2, 3, 10, 2**20, -(2**20), 2**35 + 1):
        for e in (0, 1, 2, 3, 4, 5, 8, 13):
            if e * a.bit_length() > 400: continue
            expect(read_z(w, inst_n, zpow(inst.lit(a), inst_n.lit(e))) == a ** e, 'B  the kernel computes (%d)^%d' % (a, e))
    # the identities between numerals: ring proves them (the two sides are NOT convertible: the representatives differ)
    counter = [0]
    def prove_closed(stmt, label, heavy=False):
        counter[0] += 1; nm = 's30_c%d' % counter[0]
        try: w.prove(nm, stmt, lambda p: p.ring()); ok = True
        except AuthorError: ok = False
        expect(ok, 'B  ring proves the closed identity %s' % label)
        if heavy: kernel_only.add(nm)
        else: indep.add(nm)
        return ok
    kernel_only = set()
    for a, b in ((3, -5), (-3, -4), (0, 0), (7, -7), (-1, 1), (2**64, -(2**63)), (-(10**30), 10**30 + 1)):
        prove_closed(Eq(zadd(inst.lit(a), inst.lit(b)), inst.lit(a + b)), '%d + %d = %d' % (a, b, a + b), heavy=max(abs(a), abs(b)) >= 2**40)
        prove_closed(Eq(zmul(inst.lit(a), inst.lit(b)), inst.lit(a * b)), '%d * %d = %d' % (a, b, a * b), heavy=max(abs(a), abs(b)) >= 2**40)
        prove_closed(Eq(zsub(inst.lit(a), inst.lit(b)), inst.lit(a - b)), '%d - %d = %d' % (a, b, a - b), heavy=max(abs(a), abs(b)) >= 2**40)
    prove_closed(Eq(zpow(inst.lit(-3), inst_n.lit(5)), inst.lit(-243)), '(-3)^5 = -243')
    prove_closed(Eq(zneg(zneg(inst.lit(9))), inst.lit(9)), '--9 = 9')
    tb2 = time.time() - t_start
    trace('section B done at %.1f s' % tb2)

    # ------------------------------------------------------------------------------------------------------------------ C. fixed identities
    ring_ids = {g for n, g in w.gid.items() if n.startswith('Z.ring_')}
    unary_ring_ids = {w.gid[n] for n in w.gid if n.startswith('ring_') and not n.startswith('Z.')}
    n_ring_ids = {g for n, g in w.gid.items() if n.startswith('N.ring_')}
    def prove_z(names, stmt_fn, label, hyp=None, heavy=False, exp_ring=False):
        counter[0] += 1; nm = 's30_%d' % counter[0]
        if heavy: kernel_only.add(nm)
        else: indep.add(nm)
        doms = [Z] * len(names) + ([hyp] if hyp is not None else [])
        stmt = F(doms, stmt_fn)
        try: w.prove(nm, stmt, lambda p: (p.intro(*names, *(['h'] if hyp is not None else [])), p.ring()))
        except AuthorError as e: FAIL.append('C  ring failed on %s: %s' % (label, str(e).splitlines()[0])); return None
        used = consts_in(k, w.global_value(nm).n)
        expect(not (used & unary_ring_ids), 'C  the proof of %s uses no ring lemma of the unary naturals' % label)
        expect(used & ring_ids, 'C  the proof of %s uses the Z.ring_* lemmas' % label)
        if not exp_ring: expect(not (used & n_ring_ids), 'C  the proof of %s needs no ring lemma of N (its exponents are numerals)' % label)
        return nm
    L = inst.lit; ZP = lambda a, e: zpow(a, inst_n.lit(e)); add, mul, sub, neg = zadd, zmul, zsub, zneg
    prove_z(['a', 'b'], lambda a, b: Eq(add(a, b), add(b, a)), 'comm')
    prove_z(['a', 'b'], lambda a, b: Eq(ZP(add(a, b), 2), add(add(ZP(a, 2), mul(L(2), mul(a, b))), ZP(b, 2))), '(a+b)^2')
    prove_z(['a', 'b'], lambda a, b: Eq(ZP(sub(a, b), 2), sub(add(ZP(a, 2), ZP(b, 2)), mul(L(2), mul(a, b)))), '(a-b)^2')
    prove_z(['a', 'b'], lambda a, b: Eq(mul(sub(a, b), add(a, b)), sub(ZP(a, 2), ZP(b, 2))), '(a-b)(a+b) = a^2 - b^2')
    prove_z(['a', 'b'], lambda a, b: Eq(sub(ZP(a, 3), ZP(b, 3)), mul(sub(a, b), add(add(ZP(a, 2), mul(a, b)), ZP(b, 2)))), 'a^3 - b^3')
    prove_z(['a', 'b'], lambda a, b: Eq(add(ZP(a, 3), ZP(b, 3)), mul(add(a, b), add(sub(ZP(a, 2), mul(a, b)), ZP(b, 2)))), 'a^3 + b^3')
    prove_z(['a', 'b', 'c'], lambda a, b, c: Eq(sub(add(add(ZP(a, 3), ZP(b, 3)), ZP(c, 3)), mul(L(3), mul(a, mul(b, c)))),
                                                mul(add(add(a, b), c), sub(sub(sub(add(add(ZP(a, 2), ZP(b, 2)), ZP(c, 2)), mul(a, b)), mul(b, c)), mul(c, a)))), 'a^3+b^3+c^3-3abc')
    prove_z(['x', 'y'], lambda x, y: Eq(add(ZP(x, 4), mul(L(4), ZP(y, 4))), mul(add(add(ZP(x, 2), mul(L(2), ZP(y, 2))), mul(L(2), mul(x, y))), sub(add(ZP(x, 2), mul(L(2), ZP(y, 2))), mul(L(2), mul(x, y))))),
            'Sophie Germain: x^4 + 4y^4')
    prove_z(['a', 'b', 'c', 'd'], lambda a, b, c, d: Eq(mul(add(ZP(a, 2), ZP(b, 2)), add(ZP(c, 2), ZP(d, 2))), add(ZP(sub(mul(a, c), mul(b, d)), 2), ZP(add(mul(a, d), mul(b, c)), 2))), 'Brahmagupta-Fibonacci')
    prove_z(['a', 'b', 'c', 'd'], lambda a, b, c, d: Eq(mul(add(ZP(a, 2), ZP(b, 2)), add(ZP(c, 2), ZP(d, 2))), add(ZP(add(mul(a, c), mul(b, d)), 2), ZP(sub(mul(a, d), mul(b, c)), 2))), 'Lagrange')
    prove_z(['a', 'b', 'c'], lambda a, b, c: Eq(mul(sub(a, b), add(add(sub(b, c), sub(c, a)), sub(a, b))), L(0)), '(a-b)((b-c)+(c-a)+(a-b)) = 0')
    prove_z(['a'], lambda a: Eq(sub(a, a), L(0)), 'a - a = 0')
    prove_z(['a', 'b'], lambda a, b: Eq(sub(a, b), neg(sub(b, a))), 'a - b = -(b - a)')
    prove_z(['a'], lambda a: Eq(neg(neg(a)), a), '--a = a')
    prove_z(['a', 'b'], lambda a, b: Eq(neg(add(a, b)), add(neg(a), neg(b))), '-(a+b) = -a + -b')
    prove_z(['a'], lambda a: Eq(mul(L(-1), mul(L(-1), a)), a), '(-1)(-1)a = a')
    prove_z(['a'], lambda a: Eq(ZP(a, 0), L(1)), 'a^0')
    prove_z(['a'], lambda a: Eq(ZP(a, 1), a), 'a^1')
    prove_z(['a'], lambda a: Eq(ZP(neg(a), 2), ZP(a, 2)), '(-a)^2 = a^2')
    prove_z(['a'], lambda a: Eq(ZP(neg(a), 3), neg(ZP(a, 3))), '(-a)^3 = -a^3')
    prove_z(['a', 'b'], lambda a, b: Eq(add(mul(L(0), a), mul(b, L(0))), L(0)), 'zeros')
    prove_z(['a'], lambda a: Eq(ZP(add(zsqr(a), L(1)), 2), add(add(ZP(zsqr(a), 2), mul(L(2), zsqr(a))), L(1))), '(sqr a + 1)^2: Z.sqr is the square')
    prove_z(['a'], lambda a: Eq(mul(zsqr(a), zsqr(a)), ZP(a, 4)), 'sqr a * sqr a = a^4')
    prove_z(['a'], lambda a: Eq(zsqr(a), mul(a, a)), 'sqr a = a * a')
    prove_z(['a'], lambda a: Eq(ZP(zsqr(add(a, L(1))), 2), ZP(add(a, L(1)), 4)), '(sqr (a + 1))^2 = (a + 1)^4')
    def prove_zg(names, stmt_fn, label):
        counter[0] += 1; nm = 's30_g%d' % counter[0]; indep.add(nm)
        try: w.prove(nm, F([ZZ] + [Z] * len(names), stmt_fn), lambda p: (p.intro('g', *names), p.ring()))
        except AuthorError as e: FAIL.append('C  ring failed on %s: %s' % (label, str(e).splitlines()[0]))
    prove_zg(['a'], lambda g, a: Eq(ZP(add(g(a), L(1)), 2), add(add(ZP(g(a), 2), mul(L(2), g(a))), L(1))), 'an application of a function variable is an atom')
    prove_zg(['a'], lambda g, a: Eq(mul(g(a), g(a)), ZP(g(a), 2)), 'g a * g a = (g a)^2')
    prove_zg(['a', 'b'], lambda g, a, b: Eq(add(g(a), g(b)), add(g(b), g(a))), 'g a + g b = g b + g a')
    prove_zg(['a', 'b'], lambda g, a, b: Eq(mul(sub(g(a), g(b)), add(g(a), g(b))), sub(ZP(g(a), 2), ZP(g(b), 2))), '(g a - g b)(g a + g b) = (g a)^2 - (g b)^2')
    prove_z(['a'], lambda a: Eq(mul(add(a, L(-3)), add(a, L(5))), add(add(ZP(a, 2), mul(L(2), a)), L(-15))), '(a-3)(a+5) = a^2 + 2a - 15')
    big = 10**30 + 7
    prove_z(['a'], lambda a: Eq(ZP(sub(a, L(big)), 2), add(sub(ZP(a, 2), mul(L(2 * big), a)), L(big * big))), '100-bit coefficients', heavy=True)
    prove_z(['a', 'b'], lambda a, b: Eq(mul(add(a, L(-(2**64))), sub(b, L(2**64 + 1))), add(sub(sub(mul(a, b), mul(L(2**64 + 1), a)), mul(L(2**64), b)), L(2**64 * (2**64 + 1)))), '64-bit coefficients')
    prove_z(['a'], lambda a: Eq(ZP(a, 5), mul(ZP(a, 2), ZP(a, 3))), 'a^5 = a^2 a^3')
    prove_z(['a'], lambda a: Eq(zpow(a, nadd(inst_n.lit(2), inst_n.lit(3))), mul(ZP(a, 2), ZP(a, 3))), 'an exponent that is a sum of numerals', exp_ring=True)
    prove_z(['a'], lambda a: Eq(zpow(a, nmul(inst_n.lit(2), inst_n.lit(3))), ZP(ZP(a, 2), 3)), 'an exponent that is a product of numerals', exp_ring=True)
    prove_z(['a'], lambda a: Eq(ZP(ZP(a, 10), 10), ZP(a, 100)), 'a power of a power')
    # how an exponent is read: a numeral as it is (no proof), a closed composite expression through the ring of N (with a proof), anything else is no constant
    rgz = R.Ring(w, inst)
    for v in (0, 1, 2, 5, 100, 2**40): expect(rgz.const_exp(inst_n.lit(v)) == (v, None), 'C  the numeral exponent %d is read as it is, without a proof' % v)
    for label, e, v in (('2 + 3', nadd(nl(2), nl(3)), 5), ('2 * 3', nmul(nl(2), nl(3)), 6), ('(1 + 1) * (1 + 2)', nmul(nadd(nl(1), nl(1)), nadd(nl(1), nl(2))), 6), ('0 + 0', nadd(nl(0), nl(0)), 0)):
        kv, pv = rgz.const_exp(e)
        expect(kv == v and pv is not None, 'C  the exponent %s is the constant %d, with a proof' % (label, v))
    from tactics import Proof
    pv = Proof(w, F([N], lambda n_: Eq(n_, n_)), 's30_probe'); pv.intro('n')
    with pv.focus():
        n_var = pv.h('n')
        expect(rgz.const_exp(n_var) == (None, None), 'C  an exponent that is a variable is no constant')
        expect(rgz.const_exp(nadd(n_var, nl(1))) == (None, None), 'C  an exponent that is a variable plus one is no constant')
        expect(rgz.const_exp(nmul(n_var, nl(0)))[0] in (None, 0), 'C  an exponent that is a variable times zero is zero or no constant')
    prove_z(['a', 'b'], lambda a, b, h: Eq(ZP(add(a, b), 2), add(add(ZP(a, 2), mul(L(2), mul(a, b))), ZP(b, 2))), 'a hypothesis in the context', hyp=lambda a, b: Eq(a, b))
    prove_z(['a', 'n'], lambda a, n_: Eq(mul(a, n_), mul(n_, a)), 'two atoms')
    # a power with a variable exponent is an atom: an identity that needs it as an atom only
    counter[0] += 1
    w.prove('s30_vexp', F([Z, N], lambda a, n_: Eq(mul(zpow(a, n_), zpow(a, n_)), ZP(zpow(a, n_), 2))), lambda p: (p.intro('a', 'n'), p.ring()))
    # binomial theorem for (a + 1)^10 and (a - 1)^7
    from math import comb
    def mono_sum(terms):
        out = None
        for (c, t) in terms:
            tt = mul(L(c), t) if c != 1 else t
            out = tt if out is None else add(out, tt)
        return out
    prove_z(['a'], lambda a: Eq(ZP(add(a, L(1)), 10), mono_sum([(comb(10, i), ZP(a, i) if i else L(1)) for i in range(11)])), '(a+1)^10')
    prove_z(['a'], lambda a: Eq(ZP(sub(a, L(1)), 7), mono_sum([((-1) ** (7 - i) * comb(7, i), ZP(a, i) if i else L(1)) for i in range(8)])), '(a-1)^7')
    # the false ones are refused with both normal forms
    for label, stmt_fn, names in (('(a+b)^2 = a^2+b^2', lambda a, b: Eq(ZP(add(a, b), 2), add(ZP(a, 2), ZP(b, 2))), ['a', 'b']), ('a+1 = a', lambda a: Eq(add(a, L(1)), a), ['a']),
                                  ('(a-b)^2 = (a+b)^2', lambda a, b: Eq(ZP(sub(a, b), 2), ZP(add(a, b), 2)), ['a', 'b']), ('a - b = b - a', lambda a, b: Eq(sub(a, b), sub(b, a)), ['a', 'b']),
                                  ('-a = a', lambda a: Eq(neg(a), a), ['a']), ('a*b = a+b', lambda a, b: Eq(mul(a, b), add(a, b)), ['a', 'b']),
                                  ('2^64 a = 2^64 a + 1', lambda a: Eq(mul(L(2**64), a), add(mul(L(2**64), a), L(1))), ['a']), ('(-a)^3 = a^3', lambda a: Eq(ZP(neg(a), 3), ZP(a, 3)), ['a'])):
        counter[0] += 1
        msg = refused_with(lambda: w.prove('s30_bad%d' % counter[0], F([Z] * len(names), stmt_fn), lambda p: (p.intro(*names), p.ring())))
        expect('different normal forms' in msg and 'left:' in msg and 'right:' in msg, 'C  ring refuses the false identity %s with both normal forms' % label)
    def nfs(stmt, names):
        counter[0] += 1
        msg = refused_with(lambda: w.prove('s30_nf%d' % counter[0], stmt, lambda p: (p.intro(*names) if names else None, p.ring())))
        lines = {l.split(':')[0].strip(): l.split(':', 1)[1].strip() for l in msg.split('\n') if l.strip().startswith(('left:', 'right:'))}
        return lines.get('left'), lines.get('right')
    for label, stmt, names, want in (
            ('(a-b)^2 = a^2+b^2', F([Z, Z], lambda a, b: Eq(ZP(sub(a, b), 2), add(ZP(a, 2), ZP(b, 2)))), ['a', 'b'], ('a^2 + -2*a*b + b^2', 'a^2 + b^2')),
            ('a - b = b - a', F([Z, Z], lambda a, b: Eq(sub(a, b), sub(b, a))), ['a', 'b'], ('a + -1*b', '-1*a + b')),
            ('a+1 = a', F([Z], lambda a: Eq(add(a, L(1)), a)), ['a'], ('a + 1', 'a')),
            ('-a = 5', F([Z], lambda a: Eq(neg(a), L(5))), ['a'], ('-1*a', '5')),
            ('a - a = 1', F([Z], lambda a: Eq(sub(a, a), L(1))), ['a'], ('0', '1')),
            ('-3 = 3', Eq(L(-3), L(3)), [], ('-3', '3')),
            ('a^(0*3) = a', F([Z], lambda a: Eq(zpow(a, nmul(inst_n.lit(0), inst_n.lit(3))), a)), ['a'], ('1', 'a'))):
        expect(nfs(stmt, names) == want, 'C  the refusal of %s shows the normal forms %s (got %s)' % (label, want, nfs(stmt, names)))
    # an exponent that is a closed expression with value zero or one is the numeral
    prove_z(['a'], lambda a: Eq(zpow(a, nmul(inst_n.lit(0), inst_n.lit(3))), L(1)), 'a^(0*3) = 1', exp_ring=True)
    prove_z(['a'], lambda a: Eq(zpow(a, nmul(inst_n.lit(1), inst_n.lit(1))), a), 'a^(1*1) = a', exp_ring=True)
    # goals that are not equations of a ring
    for label, stmt, intros in (('a <= a', F([Z], lambda a: zle(a, a)), ['a']), ('True', w['True'], []), ('a = a /\\ a = a', F([Z], lambda a: Eq(a, a) & Eq(a, a)), ['a'])):
        counter[0] += 1
        msg = refused_with(lambda: w.prove('s30_ne%d' % counter[0], stmt, lambda p: (p.intro(*intros) if intros else None, p.ring())))
        expect('not an equation' in msg, 'C  ring refuses a goal that is not an equation (%s): %s' % (label, msg[:80]))
    # a type with no ring lemmas (the pairs of naturals)
    counter[0] += 1
    msg = refused_with(lambda: w.prove('s30_np%d' % counter[0], F([NN, NN], lambda a, b: Eq(pair(N, N, nadd(w['fst'](N, N, a), w['fst'](N, N, b)), N0), pair(N, N, nadd(w['fst'](N, N, b), w['fst'](N, N, a)), N0))),
                                       lambda p: (p.intro('a', 'b'), p.ring())))
    expect('no commutative-semiring structure' in msg and 'Prod' in msg, 'C  ring says it knows no number system for pairs: %s' % msg[:100])
    # distinct applications of a function variable are distinct atoms
    counter[0] += 1
    msg = refused_with(lambda: w.prove('s30_ga%d' % counter[0], F([ZZ, Z, Z], lambda g, a, b: Eq(g(a), g(b))), lambda p: (p.intro('g', 'a', 'b'), p.ring())))
    expect('different normal forms' in msg, 'C  ring does not identify g a with g b: %s' % msg[:80])
    tc = time.time() - t_start
    trace('section C done at %.1f s' % tc)

    # ------------------------------------------------------------------------------------------------------------------ D. random expressions against the Python oracle
    SYr = Syntax(w, rng)
    equal_proved = unequal_refused = 0
    for case in range(150):
        natoms = rng.choice([1, 2, 2, 3]); depth = rng.choice([2, 3, 3, 4])
        e1 = rand_expr(rng, natoms, depth)
        if size(e1) > 40 or terms_bound(e1) > 200: continue
        if rng.random() < 0.5: e2 = equiv(equiv(e1, rng), rng)
        else: e2 = perturb(equiv(e1, rng), rng, natoms)
        if size(e2) > 90 or terms_bound(e2) > 400: continue
        same = zclean(zpoly(e1, natoms)) == zclean(zpoly(e2, natoms))
        names = ['a', 'b', 'c'][:natoms]
        stmt = F([ZZ] + [Z] * natoms, lambda g, *xs: Eq(SYr.build(e1, xs, g), SYr.build(e2, xs, g)))
        t_case = time.time()
        try:
            w.prove('s30_r%d' % case, stmt, lambda p: (p.intro('g', *names), p.ring())); got = 'proved'
            if case % 8 == 0: indep.add('s30_r%d' % case)
        except AuthorError as e: got = 'refused' if 'different normal forms' in str(e) else 'error: ' + str(e).splitlines()[0]
        except (E.ITPError, RuntimeError) as e: got = 'KERNEL REJECTED: ' + str(e)[:200]
        if time.time() - t_case > 3: trace('D case %d took %.1f s: %s = %s' % (case, time.time() - t_case, zshow(e1), zshow(e2)))
        if same:
            if got == 'proved': equal_proved += 1
            else: FAIL.append('D  ring should prove %s = %s but: %s' % (zshow(e1), zshow(e2), got))
        else:
            if got == 'refused': unequal_refused += 1
            else: FAIL.append('D  ring should refuse %s = %s (different polynomials) but: %s' % (zshow(e1), zshow(e2), got))
    expect(equal_proved >= 45 and unequal_refused >= 20, 'D  the random test is not vacuous (proved %d true identities, refused %d false ones)' % (equal_proved, unequal_refused))
    td = time.time() - t_start
    trace('section D done at %.1f s' % td)

    # ------------------------------------------------------------------------------------------------------------------ E. the normal form is the Python polynomial
    n_nf = 0; SY0 = Syntax(w)
    for case in range(70):
        natoms = rng.choice([1, 2, 3]); e = rand_expr(rng, natoms, rng.choice([2, 3, 4]), small_num)
        if size(e) > 50 or terms_bound(e) > 200: continue
        oracle = zclean(zpoly(e, natoms)); width = 2 * natoms; names = ['a', 'b', 'c'][:natoms]
        p = Proof(w, F([ZZ] + [Z] * natoms, lambda g, *xs: Eq(SY0.build(e, xs, g), SY0.build(e, xs, g))), 'e%d' % case); p.intro('g', *names)
        xs = [p.h(nm) for nm in names]; gv = p.h('g')
        rg = R.Ring(w, inst); NFe, _ = rg.nf(SY0.build(e, xs, gv))
        slot = {}
        for j in range(natoms): slot[xs[j]] = j; slot[gv(xs[j])] = natoms + j
        got = {}
        for m in NFe.monos:
            vec = [0] * width
            for (i, ex) in m.vars: vec[slot[rg.atoms[i]]] = ex
            got[tuple(vec)] = m.coeff
        expect(got == oracle, 'E  the normal form of %s is the polynomial (got %s, expected %s)' % (zshow(e), got, oracle))
        expect(all(m.coeff != 0 for m in NFe.monos), 'E  the normal form of %s has no monomial with coefficient 0' % zshow(e))
        n_nf += 1
    expect(n_nf >= 50, 'E  the normal-form test is not vacuous (%d normal forms compared)' % n_nf)
    te = time.time() - t_start
    trace('section E done at %.1f s' % te)

    # ------------------------------------------------------------------------------------------------------------------ F. the order, semantically
    def prove_le_true(a_t, b_t, v, label):
        """a proof of  a_t <= b_t  with the witness v = b - a >= 0 (None when the library cannot prove it)"""
        counter[0] += 1; nm = 's30_le%d' % counter[0]
        try: w.prove(nm, zle(a_t, b_t), lambda p: (p.exists(inst_n.lit(v)), p.ring()))
        except AuthorError as e: FAIL.append('F  %s: the witness %d does not prove it: %s' % (label, v, str(e).splitlines()[0])); return None
        if counter[0] % 10 == 0 and v < 2**40: indep.add(nm)
        return w[nm]
    def checks(term, ty, label):
        try: w.check(term, ty); return True
        except AuthorError as e: FAIL.append('F  %s: %s' % (label, str(e).splitlines()[0])); return False
    n_le = n_lt = 0
    int_values = [rng.choice(values) for _ in range(40)] + [0, 1, -1, 2, -2]
    cmp_pairs = [(rng.choice(int_values), rng.choice(int_values)) for _ in range(40)] + [(a, a) for a in (0, 1, -1, 2**65, -(2**65))] + [(a, a + 1) for a in (0, -1, 2**64, -(2**64))] + [(a + 1, a) for a in (0, -1, 2**64, -(2**64))]
    for a, b in cmp_pairs:
        A_, B_ = L(a), L(b)
        if a <= b:                                                      # a <= b : the witness is b - a
            pf = prove_le_true(A_, B_, b - a, '%d <= %d' % (a, b)); n_le += pf is not None
        else:                                                           # ~(a <= b) from  b < a  (b + 1 <= a, the witness a - b - 1)
            h = prove_le_true(zadd(L(b), Z1), A_, a - b - 1, '%d < %d' % (b, a))
            if h is not None: n_le += checks(w['Z.not_le_of_lt'](L(b), A_, h), ~zle(A_, B_), '~(%d <= %d)' % (a, b))
        if a < b:                                                       # a < b : a + 1 <= b, the witness b - a - 1
            pf = prove_le_true(zadd(A_, Z1), B_, b - a - 1, '%d < %d' % (a, b)); n_lt += pf is not None
        else:                                                           # ~(a < b) from  b <= a
            h = prove_le_true(B_, A_, a - b, '%d <= %d' % (b, a))
            if h is not None: n_lt += checks(w['Z.not_lt_of_le'](B_, A_, h), ~zlt(A_, B_), '~(%d < %d)' % (a, b))
    expect(n_le >= 45 and n_lt >= 45, 'F  the order test is not vacuous (%d, %d)' % (n_le, n_lt))
    # the library's three-way split instantiates at numerals
    for a, b in cmp_pairs[:30]:
        A_, B_ = L(a), L(b)
        expect(checks(w['Z.lt_trichotomy'](A_, B_), zlt(A_, B_) | (Eq(A_, B_) | zlt(B_, A_)), 'trichotomy at %d, %d' % (a, b)), 'F  trichotomy instantiates at %d, %d' % (a, b))
    tf = time.time() - t_start
    trace('section F done at %.1f s' % tf)

    # ------------------------------------------------------------------------------------------------------------------ H1. time (before the independent re-check)
    def sum_terms_z(terms, a, b):
        out = None
        for (c, i, j) in terms:
            t = L(c)
            if i: t = mul(t, ZP(a, i))
            if j: t = mul(t, ZP(b, j))
            out = t if out is None else add(out, t)
        return out
    t1 = time.time()
    prove_z(['a', 'b'], lambda a, b: Eq(ZP(sub(a, b), 8), sum_terms_z([((-1) ** i * comb(8, i), 8 - i, i) for i in range(9)], a, b)), '(a-b)^8')
    dt8 = time.time() - t1; t1 = time.time()
    prove_z(['a', 'b', 'c'], lambda a, b, c: Eq(ZP(add(add(a, b), c), 4), mul(ZP(add(add(c, b), a), 2), ZP(add(add(a, c), b), 2))), '(a+b+c)^4')
    dt4 = time.time() - t1
    expect(dt8 < 8.0, 'H  (a-b)^8 over Z is admitted in under 8 s (took %.2f s)' % dt8)
    expect(dt4 < 8.0, 'H  (a+b+c)^4 over Z is admitted in under 8 s (took %.2f s)' % dt4)

    # ------------------------------------------------------------------------------------------------------------------ G. the independent checker over the integer part of the world
    t2 = time.time()
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        if name in kernel_only: continue
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid),
                recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    chk = ic.Checker(env, params=True, fuel=200_000_000, prop_elim=True)
    bad = []
    if chk.quot_problem is not None: bad.append(('quotients: ' + chk.quot_problem,))
    bad += chk.well_founded_problems() + chk.recursion_problems()
    sizes = {nm: k.node_info(w.global_value(nm).n)[2] for nm in indep}
    trace('sizes of the picked theorems: %s' % sorted(((v, nm) for nm, v in sizes.items()), reverse=True)[:12])
    indep = {nm for nm in indep if sizes[nm] <= INDEP_MAX_TREE}
    n_checked = 0; t_lib = 0.0; times_g = []
    for gid, gl in chk.g.items():
        if gid < n_base: continue                                    # the base (S24/S25 re-check it)
        if gid >= lib_end and gl.name not in indep: continue         # the library, and the theorems proved here that were picked
        t_g = time.time()
        ok, msg = chk.run(lambda: (chk.sort_of({}, gl.type), None if gl.value is None else chk.check({}, gl.value, gl.type)))
        n_checked += 1; t_lib += time.time() - t_g; times_g.append((time.time() - t_g, gl.name))
        if not ok: bad.append((gid, gl.name, msg))
    expect(n_checked >= 200 and not bad, 'G  the independent checker re-checks the %d globals of the integer part (the library and %d of the identities proved here; the %d theorems about 100-bit numerals are the kernel\'s alone) without a rejection: %s' % (n_checked, len(indep), len(kernel_only), bad[:3]))
    tg = time.time() - t2
    trace('section G done: %d globals in %.1f s; slowest: %s' % (n_checked, tg, [(round(a, 1), b) for a, b in sorted(times_g, reverse=True)[:14]]))

    # ------------------------------------------------------------------------------------------------------------------ H2. corrupted ring proofs over Z
    t3 = time.time()
    st = F([Z, Z], lambda a, b: Eq(mul(sub(a, b), add(a, b)), sub(mul(a, a), mul(b, b))))
    pr = Proof(w, st, 'e'); pr.intro('a', 'b'); pr.ring(); proof = pr.term()
    stmt_h = parse_raw(k.dump_raw(st.n), True)
    pool = [n for n in dag_nodes(k, proof.n) if k.view(n)[0] in ('app', 'const', 'lam')]
    ring_consts = [k.mk_const(w.gid[nm]) for nm in w.gid if nm.startswith('Z.ring_')]
    kinds = {'kernel_rej': 0, 'kernel_acc': 0, 'disagree': 0, 'undecided': 0}
    chk_small = ic.Checker(env, params=True, fuel=1_500_000, prop_elim=True)       # a corrupted proof can send a checker into a huge conversion problem: the verdict 'undecided' is allowed, a minute of Python is not
    for case in range(40):
        target = rng.choice(pool); v = k.view(target); r = rng.random()
        if r < 0.3: repl = rng.choice(pool)
        elif r < 0.5 and v[0] == 'app' and k.view(v[1])[0] == 'app':
            inner = k.view(v[1]); repl = k.mk_app(k.mk_app(inner[1], v[2]), inner[2])          # swap the last two arguments
        elif r < 0.7: repl = rng.choice(ring_consts)
        elif r < 0.8: repl = k.mk_const(w.gid['Z.zero'])
        elif r < 0.9: repl = k.mk_const(w.gid['Z.neg'])
        else: repl = k.mk_app(k.mk_const(w.gid['Z.neg']), target) if v[0] != 'lam' else target
        if repl == target: continue
        try: mutant = rebuild(k, proof.n, {target: repl})
        except ValueError: continue
        if mutant == proof.n: continue
        try: kok = bool(k.check_open(mutant, st.n, [])[0])
        except Exception: kok = False
        try: iok, imsg = chk_small.check_closed(parse_raw(k.dump_raw(mutant), True), stmt_h)
        except ic.Unsupported: continue
        if not iok and 'fuel' in imsg: kinds['undecided'] += 1; continue
        if kok != iok: kinds['disagree'] += 1; FAIL.append('H  the kernel (%s) and the independent checker (%s: %s) disagree about a corrupted ring proof over Z' % (kok, iok, imsg))
        elif kok: kinds['kernel_acc'] += 1
        else: kinds['kernel_rej'] += 1
    expect(kinds['kernel_rej'] >= 18, 'H  the corruption test is not vacuous (%s)' % kinds)
    expect(k.check_open(proof.n, st.n, [])[0] and chk.check_closed(parse_raw(k.dump_raw(proof.n), True), stmt_h)[0], 'H  the uncorrupted proof is accepted by both checkers')
    tf2 = time.time() - t3

    # (declared AFTER the independent re-check: the checker reads proofs as TEXT, which spells out the tree of a proof with this much sharing)
    t1 = time.time()
    prove_z(['a'], lambda a: Eq(ZP(sub(a, L(1)), 32), mul(ZP(sub(a, L(1)), 16), ZP(sub(a, L(1)), 16))), '(a-1)^32 = (a-1)^16 (a-1)^16')
    dt32 = time.time() - t1
    expect(dt32 < 25.0, 'H  (a-1)^32 over Z is admitted in under 25 s (took %.2f s)' % dt32)
    t1 = time.time()
    nm = prove_z(['a', 'b'], lambda a, b: Eq(mul(ZP(a, 2**70), L(0)), mul(L(0), b)), 'a numeral exponent of 70 bits against a zero coefficient')
    prove_z(['a'], lambda a: Eq(mul(ZP(a, 2**70), ZP(a, 2**70)), ZP(a, 2**71)), 'a^(2^70) * a^(2^70) = a^(2^71)')
    dt70 = time.time() - t1
    expect(nm is not None and k.node_info(w.global_value(nm).n)[2] == 2**31 - 1, 'H  the tree of that proof is too big to count (its size saturates)')
    expect(dt70 < 12.0, 'H  powers with 70-bit exponents are admitted in under 12 s (took %.2f s)' % dt70)

    print("   %d checks; world built in %.1f s; %d statements and %d ring lemmas compared; %d closed evaluations; random expressions: %d true identities proved, %d false ones refused; %d normal forms; "
          "%d order facts; independent re-check of %d globals in %.1f s; %d corrupted proofs %s in %.1f s; (a-b)^8 %.2f s, (a-1)^32 %.2f s (%.1f s total)" %
          (CHECKS[0], tb, len(spec), len(rspec), n_ar, equal_proved, unequal_refused, n_nf, n_le + n_lt, n_checked, tg, sum(kinds.values()), kinds, tf2, dt8, dt32, time.time() - t_start))
    print("   sections (cumulative s): A %.1f, B %.1f, C %.1f, D %.1f, E %.1f, F %.1f" % (ta, tb2, tc, td, te, tf))
    for f in FAIL[:40]: print("   FAIL " + f)
    if len(FAIL) > 40: print("   ... and %d more" % (len(FAIL) - 40))
    print("S30 INTEGER GATE: " + ("PASS" if not FAIL else "FAIL"))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
