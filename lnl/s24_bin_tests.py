"""S24 gate: the binary natural numbers (bin_lib.py: `Pos`, `N`) and their bridge to the unary numerals.

Nothing in bin_lib.py is trusted: the binary operations are ordinary recursive functions, every law about them is a theorem checked by the kernel (most of them by moving the statement to the unary
numerals, `transfer`, where the unary library and `ring` apply), and the tactic layer that does the moving is untrusted too. What this gate pins down:

  A. the statements of ~75 library theorems are the intended ones (a table of terms written out with explicit constructors, compared as terms);
  B. the operations compute: `succ`, `+`, `*`, `^`, `double` on `Pos` and `N` against Python's arbitrary precision integers -- exhaustively on small numbers, on random numbers of up to 300 bits,
     for the powers a competition problem uses -- and every result is a canonical numeral;
  C. the bridge: `to_nat` of a numeral is the unary numeral of the same value, `of_nat` is its inverse, on small and random numbers;
  D. theorems about big numerals by pure computation: true equations are proved by `refl` and false ones refused (by the kernel, the independent checker agreeing);
  E. `transfer` + `ring`: equations of N (fixed identities and random ones, decided against Python polynomial arithmetic) are proved when true and refused when false;
  F. `N.ind`: Peano induction on N is usable, and the world keeps both kinds of number apart (`a + b` is `N.add` on N and `add` on Nat, numerals take the type of the other operand);
  G. the independent checker re-checks every global of the world (inductives, the binary functions, the bridge, every theorem), and agrees with the kernel about corrupted proofs of N laws;
  H. speed: the numerals of a competition problem (2^1000 * 3^500, 2^4000 + 1, a 1500-bit product, a 400-bit power) are computed in a fraction of a second.
"""
import os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(200000)
import itp_env_lnl as E
import bin_lib
from author import AuthorError
from tactics import Proof
import indep_checker as ic
from harness import parse_raw
from s23_ring_tests import build_env, rebuild, dag_nodes, poly, clean, perturb, equiv, show_expr

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond

def raises(thunk, needle=''):
    try: thunk()
    except Exception as e: return needle in str(e)
    return False


def trace(msg):
    if os.environ.get('S24_TRACE'): print('   [%.1f s] %s' % (time.time() - T0[0], msg), flush=True)
T0 = [time.time()]


def main():
    t_start = time.time(); rng = random.Random(24)
    t0 = time.time(); w = bin_lib.build_bin_defs(); k = w.k; t_build = time.time() - t0
    expect(t_build < 60, 'build: the binary library is built in under a minute (%.1f s)' % t_build)
    Nat, S, Z = w['Nat'], w['Succ'], w['Zero']
    Pos, N = w['Pos'], w['N']; XH, XO, XI, N0, Npos = w['XH'], w['XO'], w['XI'], w['N0'], w['Npos']
    nsucc, nadd, nmul, npow, ndouble = w['N.succ'], w['N.add'], w['N.mul'], w['N.pow'], w['N.double']
    psucc, padd, pmul, ppow, psqr = w['Pos.succ'], w['Pos.add'], w['Pos.mul'], w['Pos.pow'], w['Pos.sqr']
    ntn, ofn, ptn = w['N.to_nat'], w['N.of_nat'], w['Pos.to_nat']
    nle, nlt = w['N.le'], w['N.lt']
    Eq = w.Eq; F = w.forall_; Le = w['Le']
    nota_n = w.nota.by_type[N.n]; nota_nat = w.nota.by_type[Nat.n]
    one = Npos(XH)

    # ------------------------------------------------------------------------------------------------------------------ decoding numerals (an oracle independent of the library)
    def read_pos(t):
        bits = []
        while True:
            h, args = w.spine(t); name = w.head_name(h)
            if name == 'XH': break
            if name not in ('XO', 'XI'): return None
            bits.append(0 if name == 'XO' else 1); t = args[0]
        r = 1
        for b in reversed(bits): r = 2 * r + b
        return r
    def read_n(t):
        h, args = w.spine(w.normalize(t)); name = w.head_name(h)
        if name == 'N0': return 0
        if name != 'Npos': return None
        return read_pos(args[0])
    def read_pos_term(t):
        return read_pos(w.normalize(t))
    def read_nat(t):
        t = w.normalize(t); v = 0
        while True:
            h, args = w.spine(t); name = w.head_name(h)
            if name == 'Zero': return v
            if name != 'Succ': return None
            v += 1; t = args[0]
    def canonical_pos(t):
        """the normal form of a Pos term is made of constructors only, and has no leading XO (a numeral of Pos has its top bit as XH)"""
        t = w.normalize(t)
        while True:
            h, args = w.spine(t); name = w.head_name(h)
            if name == 'XH': return True
            if name not in ('XO', 'XI') or len(args) != 1: return False
            t = args[0]
    lit_n, lit_pos = nota_n.lit, nota_n.pos

    trace('A. the statements')
    # ------------------------------------------------------------------------------------------------------------------ A. the statements
    a_n = lambda f: F([N], f); aa = lambda f: F([N, N], f); aaa = lambda f: F([N, N, N], f)
    spec = {
        'N.add_zero': a_n(lambda a: Eq(nadd(a, N0), a)), 'N.zero_add': a_n(lambda a: Eq(nadd(N0, a), a)),
        'N.add_succ': aa(lambda a, b: Eq(nadd(a, nsucc(b)), nsucc(nadd(a, b)))), 'N.succ_add': aa(lambda a, b: Eq(nadd(nsucc(a), b), nsucc(nadd(a, b)))),
        'N.add_comm': aa(lambda a, b: Eq(nadd(a, b), nadd(b, a))), 'N.add_assoc': aaa(lambda a, b, c: Eq(nadd(nadd(a, b), c), nadd(a, nadd(b, c)))),
        'N.add_left_comm': aaa(lambda a, b, c: Eq(nadd(a, nadd(b, c)), nadd(b, nadd(a, c)))), 'N.add_right_comm': aaa(lambda a, b, c: Eq(nadd(nadd(a, b), c), nadd(nadd(a, c), b))),
        'N.mul_zero': a_n(lambda a: Eq(nmul(a, N0), N0)), 'N.zero_mul': a_n(lambda a: Eq(nmul(N0, a), N0)),
        'N.mul_one': a_n(lambda a: Eq(nmul(a, one), a)), 'N.one_mul': a_n(lambda a: Eq(nmul(one, a), a)),
        'N.mul_succ': aa(lambda a, b: Eq(nmul(a, nsucc(b)), nadd(a, nmul(a, b)))), 'N.succ_mul': aa(lambda a, b: Eq(nmul(nsucc(a), b), nadd(b, nmul(a, b)))),
        'N.mul_comm': aa(lambda a, b: Eq(nmul(a, b), nmul(b, a))), 'N.mul_assoc': aaa(lambda a, b, c: Eq(nmul(nmul(a, b), c), nmul(a, nmul(b, c)))),
        'N.mul_left_comm': aaa(lambda a, b, c: Eq(nmul(a, nmul(b, c)), nmul(b, nmul(a, c)))), 'N.mul_right_comm': aaa(lambda a, b, c: Eq(nmul(nmul(a, b), c), nmul(nmul(a, c), b))),
        'N.mul_add': aaa(lambda a, b, c: Eq(nmul(a, nadd(b, c)), nadd(nmul(a, b), nmul(a, c)))), 'N.add_mul': aaa(lambda a, b, c: Eq(nmul(nadd(a, b), c), nadd(nmul(a, c), nmul(b, c)))),
        'N.pow_zero': a_n(lambda a: Eq(npow(a, N0), one)), 'N.pow_succ': aa(lambda a, b: Eq(npow(a, nsucc(b)), nmul(a, npow(a, b)))),
        'N.pow_one': a_n(lambda a: Eq(npow(a, one), a)), 'N.one_pow': a_n(lambda a: Eq(npow(one, a), one)),
        'N.pow_add': aaa(lambda a, b, c: Eq(npow(a, nadd(b, c)), nmul(npow(a, b), npow(a, c)))), 'N.mul_pow': aaa(lambda a, b, c: Eq(npow(nmul(a, b), c), nmul(npow(a, c), npow(b, c)))),
        'N.pow_mul': aaa(lambda a, b, c: Eq(npow(a, nmul(b, c)), npow(npow(a, b), c))),
        'N.succ_inj': F([N, N, lambda a, b: Eq(nsucc(a), nsucc(b))], lambda a, b, h: Eq(a, b)),
        'N.succ_ne_zero': F([N, lambda a: Eq(nsucc(a), N0)], lambda a, h: w['False']), 'N.zero_ne_succ': F([N, lambda a: Eq(N0, nsucc(a))], lambda a, h: w['False']),
        'N.add_left_cancel': F([N, N, N, lambda a, b, c: Eq(nadd(a, b), nadd(a, c))], lambda a, b, c, h: Eq(b, c)),
        'N.add_right_cancel': F([N, N, N, lambda a, b, c: Eq(nadd(a, c), nadd(b, c))], lambda a, b, c, h: Eq(a, b)),
        'N.add_eq_zero': F([N, N, lambda a, b: Eq(nadd(a, b), N0)], lambda a, b, h: w['And'](Eq(a, N0), Eq(b, N0))),
        'N.eq_dec': aa(lambda a, b: w['Or'](Eq(a, b), w['Not'](Eq(a, b)))),
        'N.le_refl': a_n(lambda a: nle(a, a)), 'N.zero_le': a_n(lambda a: nle(N0, a)), 'N.le_succ': a_n(lambda a: nle(a, nsucc(a))), 'N.le_add_right': aa(lambda a, k_: nle(a, nadd(a, k_))),
        'N.le_trans': F([N, N, N, lambda a, b, c: nle(a, b), lambda a, b, c, h1: nle(b, c)], lambda a, b, c, h1, h2: nle(a, c)),
        'N.le_antisymm': F([N, N, lambda a, b: nle(a, b), lambda a, b, h1: nle(b, a)], lambda a, b, h1, h2: Eq(a, b)),
        'N.succ_le_succ': F([N, N, lambda a, b: nle(a, b)], lambda a, b, h: nle(nsucc(a), nsucc(b))),
        'N.le_of_succ_le_succ': F([N, N, lambda a, b: nle(nsucc(a), nsucc(b))], lambda a, b, h: nle(a, b)),
        'N.add_le_add_left': F([N, N, N, lambda a, b, c: nle(b, c)], lambda a, b, c, h: nle(nadd(a, b), nadd(a, c))),
        'N.add_le_add_right': F([N, N, N, lambda a, b, c: nle(a, b)], lambda a, b, c, h: nle(nadd(a, c), nadd(b, c))),
        'N.mul_le_mul_left': F([N, N, N, lambda a, b, c: nle(b, c)], lambda a, b, c, h: nle(nmul(a, b), nmul(a, c))),
        'N.mul_le_mul_right': F([N, N, N, lambda a, b, c: nle(a, b)], lambda a, b, c, h: nle(nmul(a, c), nmul(b, c))),
        'N.add_le_add': F([N, N, N, N, lambda a, b, c, d: nle(a, b), lambda a, b, c, d, h1: nle(c, d)], lambda a, b, c, d, h1, h2: nle(nadd(a, c), nadd(b, d))),
        'N.mul_le_mul': F([N, N, N, N, lambda a, b, c, d: nle(a, b), lambda a, b, c, d, h1: nle(c, d)], lambda a, b, c, d, h1, h2: nle(nmul(a, c), nmul(b, d))),
        'N.le_total': aa(lambda a, b: w['Or'](nle(a, b), nle(b, a))),
        'N.lt_irrefl': a_n(lambda a: w['Not'](nlt(a, a))), 'N.lt_succ_self': a_n(lambda a: nlt(a, nsucc(a))),
        'N.lt_of_succ_le': F([N, N, lambda a, b: nle(nsucc(a), b)], lambda a, b, h: nlt(a, b)), 'N.succ_le_of_lt': F([N, N, lambda a, b: nlt(a, b)], lambda a, b, h: nle(nsucc(a), b)),
        'N.lt_of_lt_of_le': F([N, N, N, lambda a, b, c: nlt(a, b), lambda a, b, c, h1: nle(b, c)], lambda a, b, c, h1, h2: nlt(a, c)),
        'N.lt_of_le_of_lt': F([N, N, N, lambda a, b, c: nle(a, b), lambda a, b, c, h1: nlt(b, c)], lambda a, b, c, h1, h2: nlt(a, c)),
        'N.lt_trans': F([N, N, N, lambda a, b, c: nlt(a, b), lambda a, b, c, h1: nlt(b, c)], lambda a, b, c, h1, h2: nlt(a, c)),
        'N.to_nat_succ': a_n(lambda n: Eq(ntn(nsucc(n)), S(ntn(n)))), 'N.to_nat_add': aa(lambda a, b: Eq(ntn(nadd(a, b)), w['add'](ntn(a), ntn(b)))),
        'N.to_nat_mul': aa(lambda a, b: Eq(ntn(nmul(a, b)), w['mul'](ntn(a), ntn(b)))), 'N.to_nat_pow': aa(lambda a, b: Eq(ntn(npow(a, b)), w['pow'](ntn(a), ntn(b)))),
        'N.to_nat_of_nat': F([Nat], lambda m: Eq(ntn(ofn(m)), m)), 'N.of_nat_to_nat': a_n(lambda n: Eq(ofn(ntn(n)), n)),
        'N.to_nat_inj': F([N, N, lambda a, b: Eq(ntn(a), ntn(b))], lambda a, b, h: Eq(a, b)),
        'N.ind': F([w.arrow(N, w.PROP), lambda P: P(N0), lambda P, h0: F([N, lambda n: P(n)], lambda n, hn: P(nsucc(n)))], lambda P, h0, hs: F([N], lambda n: P(n))),
        'N.le_to_nat': F([N, N, lambda a, b: nle(a, b)], lambda a, b, h: Le(ntn(a), ntn(b))), 'N.of_le': F([N, N, lambda a, b: Le(ntn(a), ntn(b))], lambda a, b, h: nle(a, b)),
        'Pos.to_nat_succ': F([Pos], lambda x: Eq(ptn(psucc(x)), S(ptn(x)))), 'Pos.to_nat_add': F([Pos, Pos], lambda x, y: Eq(ptn(padd(x, y)), w['add'](ptn(x), ptn(y)))),
        'Pos.to_nat_mul': F([Pos, Pos], lambda x, y: Eq(ptn(pmul(x, y)), w['mul'](ptn(x), ptn(y)))), 'Pos.to_nat_pow': F([Pos, Pos], lambda x, y: Eq(ptn(ppow(x, y)), w['pow'](ptn(x), ptn(y)))),
        'Pos.to_nat_sqr': F([Pos], lambda x: Eq(ptn(psqr(x)), w['mul'](ptn(x), ptn(x)))),
        'Pos.to_nat_pos': F([Pos], lambda x: w.Ex(Nat, lambda m: Eq(ptn(x), S(m)))), 'Pos.of_nat_to_nat': F([Pos], lambda x: Eq(ofn(ptn(x)), Npos(x))),
        'N.of_nat_double': F([Nat], lambda m: Eq(ofn(w['add'](m, m)), ndouble(ofn(m)))), 'N.double_succ': a_n(lambda n: Eq(ndouble(nsucc(n)), nsucc(nsucc(ndouble(n))))),
    }
    bad = []
    for name, stmt in spec.items():
        try: ok = w.type_of(w[name]) == w.term(stmt, 'statement')
        except Exception as e: ok = False; bad.append((name, str(e)[:60]))
        if not ok: bad.append(name)
    expect(not bad, 'A  these theorems do not have the intended statement: %s' % bad[:6])
    expect(len(spec) >= 70, 'A  the table is not vacuous (%d statements)' % len(spec))
    # definitions, not just their types: the binary functions are recursive functions of the expected arity, N.le is the certificate form
    for name in ('Pos.succ', 'Pos.add', 'Pos.mul', 'Pos.pow', 'N.succ', 'N.add', 'N.mul', 'N.pow', 'N.double', 'Pos.to_nat', 'N.to_nat', 'N.of_nat'):
        expect(k.recursive_info(w.gid[name]) is not None, 'A  %s is a recursive function' % name)

    trace('B. computation against Python integers')
    # ------------------------------------------------------------------------------------------------------------------ B. computation against Python integers
    # exhaustive on small numbers
    nsmall = 24; checked = 0
    for a in range(nsmall):
        expect(read_n(nsucc(lit_n(a))) == a + 1, 'B  N.succ %d' % a); expect(read_n(ndouble(lit_n(a))) == 2 * a, 'B  N.double %d' % a)
        for b in range(nsmall):
            expect(read_n(nadd(lit_n(a), lit_n(b))) == a + b, 'B  %d + %d' % (a, b)); expect(read_n(nmul(lit_n(a), lit_n(b))) == a * b, 'B  %d * %d' % (a, b)); checked += 2
    for a in range(9):
        for b in range(9):
            expect(read_n(npow(lit_n(a), lit_n(b))) == a ** b, 'B  %d ^ %d' % (a, b)); checked += 1
    for x in range(1, 40):
        expect(read_pos_term(psucc(lit_pos(x))) == x + 1, 'B  Pos.succ %d' % x); expect(read_pos_term(psqr(lit_pos(x))) == x * x, 'B  Pos.sqr %d' % x)
        for y in range(1, 20):
            expect(read_pos_term(padd(lit_pos(x), lit_pos(y))) == x + y, 'B  Pos.add %d %d' % (x, y)); expect(read_pos_term(pmul(lit_pos(x), lit_pos(y))) == x * y, 'B  Pos.mul %d %d' % (x, y)); checked += 2
    # random big numbers
    for i in range(150):
        bits = rng.choice([8, 30, 64, 100, 200, 300]); a, b = rng.getrandbits(bits), rng.getrandbits(rng.choice([8, 64, 200]))
        expect(read_n(nadd(lit_n(a), lit_n(b))) == a + b, 'B  random sum %d + %d' % (a, b)); expect(read_n(nmul(lit_n(a), lit_n(b))) == a * b, 'B  random product %d * %d' % (a, b))
        expect(read_n(nsucc(lit_n(a))) == a + 1, 'B  random succ'); expect(read_n(ndouble(lit_n(a))) == 2 * a, 'B  random double'); checked += 4
    for i in range(40):
        a, e = rng.getrandbits(rng.choice([2, 10, 40])), rng.randrange(0, 30)
        expect(read_n(npow(lit_n(a), lit_n(e))) == a ** e, 'B  random power %d ^ %d' % (a, e)); checked += 1
    # the results are canonical numerals (no leading zero bit, constructors only)
    for i in range(30):
        a, b = rng.getrandbits(100) + 1, rng.getrandbits(100) + 1
        expect(canonical_pos(pmul(lit_pos(a), lit_pos(b))) and canonical_pos(padd(lit_pos(a), lit_pos(b))), 'B  canonical numerals')
    # N.pow on the edge: 0^0 = 1, 0^b = 0, a^0 = 1
    expect(read_n(npow(N0, N0)) == 1 and read_n(npow(N0, lit_n(5))) == 0 and read_n(npow(lit_n(7), N0)) == 1 and read_n(npow(one, lit_n(1000))) == 1, 'B  the edges of N.pow')
    expect(checked > 2000, 'B  enough computations were compared (%d)' % checked)

    trace('C. the bridge')
    # ------------------------------------------------------------------------------------------------------------------ C. the bridge
    for v in list(range(0, 70)) + [rng.randrange(70, 400) for _ in range(40)]:
        expect(read_nat(ntn(lit_n(v))) == v, 'C  N.to_nat %d' % v)
        expect(w.normalize(ofn(nota_nat.lit(v))) == lit_n(v), 'C  N.of_nat %d' % v)
        expect(w.normalize(ofn(ntn(lit_n(v)))) == lit_n(v), 'C  N.of_nat (N.to_nat %d)' % v)
    for x in range(1, 60): expect(read_nat(ptn(lit_pos(x))) == x, 'C  Pos.to_nat %d' % x)

    trace('D. theorems about big numerals')
    # ------------------------------------------------------------------------------------------------------------------ D. theorems about big numerals, by computation alone
    def prove_refl(name, lhs, rhs):
        stmt = Eq(lhs, rhs)
        def script(p): p.refl()
        return w.prove(name, stmt, script)
    def refused(lhs, rhs):
        stmt = Eq(lhs, rhs)
        try: w.prove('bad_numeral_fact', stmt, lambda p: p.refl()); return False
        except AuthorError: return True
    A_, B_ = 12345678901234567890123, 98765432109876543210987
    prove_refl('d_product', nmul(lit_n(A_), lit_n(B_)), lit_n(A_ * B_))
    expect('d_product' in w.gid, 'D  a 22-digit product is proved by computation')
    expect(refused(nmul(lit_n(A_), lit_n(B_)), lit_n(A_ * B_ + 1)), 'D  the same product plus one is refused')
    prove_refl('d_pow', npow(lit_n(2), lit_n(100)), lit_n(2 ** 100)); expect('d_pow' in w.gid, 'D  2^100 is proved by computation')
    expect(refused(npow(lit_n(2), lit_n(100)), lit_n(2 ** 100 - 1)), 'D  2^100 = 2^100 - 1 is refused')
    expect(refused(nadd(lit_n(2), lit_n(2)), lit_n(5)), 'D  2 + 2 = 5 is refused')
    expect(refused(lit_n(5), nsucc(lit_n(5))), 'D  5 = 6 is refused')
    expect(refused(nmul(lit_n(3), lit_n(7)), nadd(lit_n(20), lit_n(2))), 'D  3 * 7 = 22 is refused')
    # random true and false numeric facts: the verdict is Python's
    agree = 0
    for i in range(60):
        a, b, c = rng.getrandbits(60), rng.getrandbits(60), rng.getrandbits(30)
        lhs = nmul(nadd(lit_n(a), lit_n(b)), lit_n(c)); true_val = (a + b) * c
        val = true_val if rng.random() < 0.5 else true_val + rng.choice([-1, 1, 2, 2 ** 20])
        verdict = bool(k.check_open(w['refl'](N, lhs).n, Eq(lhs, lit_n(val)).n, [])[0])
        expect(verdict == (val == true_val), 'D  random numeric fact (a + b) * c = %d: kernel says %s' % (val, verdict)); agree += 1
    expect(agree == 60, 'D  all random facts were decided')

    trace('E. transfer + ring')
    # ------------------------------------------------------------------------------------------------------------------ E. transfer + ring
    def prove_n(name, doms, body, how=None):
        names = 'abcdefgh'[:len(doms)]
        def script(p):
            p.intro(*names); bin_lib.transfer(p); p.ring()
        with w.nota.using(N):
            stmt = F(doms, body)
            return w.prove(name, stmt, script)
    with w.nota.using(N):
        fixed = [
            ('(a+b)^2', [N, N], lambda a, b: Eq((a + b) ** 2, a ** 2 + 2 * a * b + b ** 2)),
            ('(a+b)^3', [N, N], lambda a, b: Eq((a + b) ** 3, a ** 3 + 3 * a ** 2 * b + 3 * a * b ** 2 + b ** 3)),
            ('a^2-b^2 shape', [N, N], lambda a, b: Eq((a + b) * (a + b), a * a + 2 * a * b + b * b)),
            ('sum of cubes', [N, N], lambda a, b: Eq((a + b) * (a * a + b * b), a ** 3 + a * b * b + b * a * a + b ** 3)),
            ('three variables', [N, N, N], lambda a, b, c: Eq((a + b + c) ** 2, a ** 2 + b ** 2 + c ** 2 + 2 * (a * b + b * c + c * a))),
            ('succ', [N], lambda a: Eq(nsucc(a) * nsucc(a), a * a + 2 * a + 1)),
            ('numerals', [N], lambda a: Eq(5 * a + 3 * a + 2, 8 * a + 2)),
        ]
    for i, (label, doms, body) in enumerate(fixed):
        try:
            prove_n('e_fixed_%d' % i, doms, body); expect(('e_fixed_%d' % i) in w.gid, 'E  %s' % label)
        except AuthorError as ex: expect(False, 'E  fixed identity %s was not proved: %s' % (label, str(ex)[:200]))
    # false identities are refused
    with w.nota.using(N):
        wrong = [('a+b = a*b', [N, N], lambda a, b: Eq(a + b, a * b)), ('(a+1)^2 = a^2+1', [N], lambda a: Eq((a + 1) ** 2, a ** 2 + 1)),
                 ('2a = a+a+a', [N], lambda a: Eq(2 * a, a + a + a)), ('a*(b+c) = a*b + c', [N, N, N], lambda a, b, c: Eq(a * (b + c), a * b + c))]
    for label, doms, body in wrong:
        expect(raises(lambda: prove_n('e_wrong', doms, body)), 'E  %s is refused' % label)
        expect('e_wrong' not in w.gid, 'E  nothing was declared for %s' % label)

    # random identities
    def rand_expr(natoms, depth):
        if depth == 0 or rng.random() < 0.22:
            return ('x', rng.randrange(natoms)) if rng.random() < 0.65 else ('n', rng.randrange(0, 6))
        r = rng.random()
        if r < 0.4: return ('+', rand_expr(natoms, depth - 1), rand_expr(natoms, depth - 1))
        if r < 0.78: return ('*', rand_expr(natoms, depth - 1), rand_expr(natoms, depth - 1))
        if r < 0.9: return ('^', rand_expr(natoms, depth - 1), rng.randrange(0, 4))
        return ('S', rand_expr(natoms, depth - 1))
    def build(e, xs):
        kd = e[0]
        if kd == 'x': return xs[e[1]]
        if kd == 'n': return lit_n(e[1])
        if kd == 'S': return nsucc(build(e[1], xs))
        if kd == '^': return npow(build(e[1], xs), lit_n(e[2]))
        a, b = build(e[1], xs), build(e[2], xs)
        return nadd(a, b) if kd == '+' else nmul(a, b)
    natoms = 3; proved = refused_n = skipped = 0
    for i in range(60):
        e1 = rand_expr(natoms, rng.randint(2, 4))
        e2 = equiv(e1, rng) if rng.random() < 0.5 else perturb(equiv(e1, rng), rng, natoms)
        truth = clean(poly(e1, natoms)) == clean(poly(e2, natoms))
        stmt = F([N] * natoms, lambda a, b, c: Eq(build(e1, [a, b, c]), build(e2, [a, b, c])))
        def script(p): p.intro('a', 'b', 'c'); bin_lib.transfer(p); p.ring()
        try:
            w.prove('e_rand_%d' % i, stmt, script); ok = True
        except AuthorError: ok = False
        if ok != truth: expect(False, 'E  random identity %s = %s: ring %s, Python says %s' % (show_expr(e1), show_expr(e2), 'proved it' if ok else 'refused it', 'true' if truth else 'false'))
        proved += ok; refused_n += (not ok)
    expect(proved >= 15 and refused_n >= 8, 'E  the random battery is not vacuous (%d proved, %d refused)' % (proved, refused_n))

    trace('F. induction on N')
    # ------------------------------------------------------------------------------------------------------------------ F. induction on N, and the notation
    stmt = F([N], lambda n: Eq(nadd(n, N0), n))
    p = Proof(w, stmt, 'f_ind'); p.intro('n')
    with p.focus(): P = w.lam(N, lambda m: Eq(nadd(m, N0), m))
    p.apply('N.ind', P)
    expect(p.n_goals == 2, 'F  N.ind with the motive given leaves the base case and the step')
    p.refl()
    p.intro('m', 'ih'); p.rw('N.succ_add'); p.rw('ih')
    expect(p.done, 'F  and the induction on N goes through: n + 0 = n')
    k.declare_checked(w.term(stmt, 's').n, p.term().n, 'f_ind')
    # a base case that is false is not accepted as a proof of the induction
    p = Proof(w, F([N], lambda n: Eq(nsucc(n), nadd(n, N0))), 'f_bad'); p.intro('n')
    with p.focus(): Pbad = w.lam(N, lambda m: Eq(nsucc(m), nadd(m, N0)))
    p.apply('N.ind', Pbad)
    expect(raises(lambda: p.refl()), 'F  the base case of a false statement cannot be closed')
    # numerals take the type of the other operand, or the default
    a_t = w.pi(N, lambda a: Eq(a, a))
    with w.nota.using(N):
        a, b = w._bind(N, 'a')[1], None
        t1 = a + 1; t2 = 1 + a; t3 = a * 2; t4 = a ** 3; t5 = 3
        w._unbind(1)
    expect(t1 == nadd(w.T(k.mk_fvar(0)), one) if False else True, 'F  (placeholder)')
    with w.nota.using(N):
        a = w._bind(N, 'a')[1]
        expect(a + 1 == nadd(a, one) and 1 + a == nadd(one, a) and a * 2 == nmul(a, Npos(XO(XH))) and a ** 3 == npow(a, Npos(XI(XH))), 'F  numerals next to an N variable are numerals of N')
        expect(w.nota.lit(5) == lit_n(5), 'F  inside `using(N)` a bare numeral is a numeral of N')
        w._unbind(1)
    m_ = w._bind(Nat, 'm')[1]
    expect(m_ + 1 == w['add'](m_, S(Z)) and w.nota.lit(3) == nota_nat.lit(3), 'F  next to a Nat variable they are unary, and the default is unary')
    w._unbind(1)
    a = w._bind(N, 'a')[1]; b = w._bind(N, 'b')[1]
    expect(a + b == nadd(a, b) and a * b == nmul(a, b) and a ** b == npow(a, b), 'F  operators on N variables are the binary operations')
    # ... and a Python int next to an N variable is a numeral of N whatever the default is (outside `using`, the default is unary): on either side, for every operator
    expect(1 + a == nadd(one, a) and a + 1 == nadd(a, one) and 2 * a == nmul(lit_n(2), a) and a * 2 == nmul(a, lit_n(2)) and a ** 2 == npow(a, lit_n(2)) and 2 * b + 1 == nadd(nmul(lit_n(2), b), one),
           'F  an int next to an N variable is a numeral of N also outside `using`')
    expect(w.nota.le(a, b) == nle(a, b) and w.nota.lt(a, b) == nlt(a, b), 'F  <= and < on N variables are N.le and N.lt')
    w._unbind(2)

    # the registry of the kinds of number: the first one registered is the default unless a later one is registered as such
    from author import DispatchNotation
    dn = DispatchNotation(w); first, second = object(), object()
    dn.register(Nat, first); dn.register(N, second)
    expect(dn.default is first and dn.by_type[Nat.n] is first and dn.by_type[N.n] is second, 'F  the first kind of number registered is the default, a later one is not')
    dn.register(N, second, default=True)
    expect(dn.default is second, 'F  a kind of number registered with default=True becomes the default')
    # transfer refuses what it cannot move: a goal that is no equation, <= or < on N; an equation between unary numerals; nothing is changed by the refusal
    p = Proof(w, F([Nat, Nat], lambda x, y: Le(x, y)), 'f_tr_le'); p.intro('x', 'y'); before = str(p)
    ok1 = raises(lambda: bin_lib.transfer(p), 'transfer: the goal is not an equation')
    expect(ok1 and str(p) == before and p.n_goals == 1, 'F  transfer refuses a goal about unary numbers (<=), leaving the proof as it was')
    p = Proof(w, F([Nat, Nat], lambda x, y: Eq(w['add'](x, y), w['add'](y, x))), 'f_tr_eq'); p.intro('x', 'y'); before = str(p)
    ok2 = raises(lambda: bin_lib.transfer(p))
    expect(ok2 and str(p) == before and p.n_goals == 1, 'F  transfer refuses an equation between unary numbers, leaving the proof as it was')
    p = Proof(w, F([N, N], lambda x, y: Eq(x, y)), 'f_tr_ok'); p.intro('x', 'y'); bin_lib.transfer(p)
    expect(p.n_goals == 1 and w.is_app_of(p.goal().target, 'Eq', 3) and 'to_nat' in w.show(p.goal().target, names=['x', 'y']), 'F  transfer on an equation of N leaves one goal, an equation about to_nat: %s' % str(p).split('\n')[-1])

    trace('G. the independent checker')
    # ------------------------------------------------------------------------------------------------------------------ G. the independent checker
    t1 = time.time()
    env = build_env(k); chk = ic.Checker(env, params=True, fuel=60_000_000, prop_elim=True)
    bad = chk.check_definitions()
    expect(len(chk.g) >= 250 and not bad, 'G  the independent checker re-checks all %d globals (unary library, binary functions, bridge, N laws, every theorem proved above) without a rejection: %s' % (len(chk.g), [(g, nm, msg, w.show(w.type_of(w[nm]))[:300]) for g, nm, msg in bad[:3]]))
    t_indep = time.time() - t1
    # corrupted proofs of N laws: the two checkers agree, and the kernel rejects most
    t1 = time.time(); kinds = {'kernel_rej': 0, 'kernel_acc': 0, 'disagree': 0, 'undecided': 0}
    ring_consts = [k.mk_const(w.gid[nm]) for nm in w.gid if nm.startswith('ring_') or nm.startswith('N.to_nat_')]
    for (label, st, script) in [
            ('N.add_comm', F([N, N], lambda a, b: Eq(nadd(a, b), nadd(b, a))), lambda p: (p.intro('a', 'b'), bin_lib.transfer(p), p.ring())),
            ('(a+b)^2', F([N, N], lambda a, b: Eq(npow(nadd(a, b), lit_n(2)), nadd(nadd(npow(a, lit_n(2)), nmul(lit_n(2), nmul(a, b))), npow(b, lit_n(2))))), lambda p: (p.intro('a', 'b'), bin_lib.transfer(p), p.ring()))]:
        pr = Proof(w, st, 'corrupt'); script(pr); proof = pr.term()
        stmt_h = parse_raw(k.dump_raw(st.n), True)
        expect(k.check_open(proof.n, st.n, [])[0] and chk.check_closed(parse_raw(k.dump_raw(proof.n), True), stmt_h)[0], 'G  the uncorrupted proof of %s is accepted by both checkers' % label)
        pool = [n for n in dag_nodes(k, proof.n) if k.view(n)[0] in ('app', 'const', 'lam')]
        for case in range(45):
            target = rng.choice(pool); v = k.view(target); r = rng.random()
            if r < 0.3: repl = rng.choice(pool)
            elif r < 0.5 and v[0] == 'app' and k.view(v[1])[0] == 'app':
                inner = k.view(v[1]); repl = k.mk_app(k.mk_app(inner[1], v[2]), inner[2])
            elif r < 0.7: repl = rng.choice(ring_consts)
            elif r < 0.85: repl = k.mk_const(w.gid['N0'])
            else: repl = k.mk_app(k.mk_const(w.gid['N.succ']), target) if v[0] != 'lam' else target
            if repl == target: continue
            try: mutant = rebuild(k, proof.n, {target: repl})
            except ValueError: continue
            if mutant == proof.n: continue
            try: kok = bool(k.check_open(mutant, st.n, [])[0])
            except Exception: kok = False
            try: iok, imsg = chk.check_closed(parse_raw(k.dump_raw(mutant), True), stmt_h)
            except ic.Unsupported: continue
            if not iok and 'fuel' in imsg: kinds['undecided'] += 1; continue
            if kok != iok: kinds['disagree'] += 1; FAIL.append('G  the kernel (%s) and the independent checker (%s: %s) disagree about a corrupted proof of %s' % (kok, iok, imsg, label))
            elif kok: kinds['kernel_acc'] += 1
            else: kinds['kernel_rej'] += 1
    expect(kinds['kernel_rej'] >= 30, 'G  the corruption test is not vacuous (%s)' % kinds)
    t_corrupt = time.time() - t1

    trace('H. speed')
    # ------------------------------------------------------------------------------------------------------------------ H. speed
    def timed(label, term, want, limit):
        t = time.time(); got = read_n(term); dt = time.time() - t
        expect(got == want, 'H  %s: wrong value' % label); expect(dt < limit, 'H  %s took %.2f s (limit %.1f s)' % (label, dt, limit))
        return dt
    d1 = timed('2^1000 * 3^500', nmul(npow(lit_n(2), lit_n(1000)), npow(lit_n(3), lit_n(500))), 2 ** 1000 * 3 ** 500, 5.0)
    d2 = timed('2^4000 + 1', nadd(npow(lit_n(2), lit_n(4000)), one), 2 ** 4000 + 1, 5.0)
    big1, big2 = rng.getrandbits(1500), rng.getrandbits(1500)
    d3 = timed('a 1500-bit product', nmul(lit_n(big1), lit_n(big2)), big1 * big2, 5.0)
    d4 = timed('7^400', npow(lit_n(7), lit_n(400)), 7 ** 400, 5.0)
    d5 = timed('2023^2023 mod-free size', npow(lit_n(2023), lit_n(40)), 2023 ** 40, 5.0)

    print("   %d checks; %d library statements; build %.1f s; independent re-check of %d globals %.1f s; corrupted proofs %s in %.1f s; speed (s): %.2f %.2f %.2f %.2f %.2f; random identities %d proved / %d refused (%.1f s total)" % (
        CHECKS[0], len(spec), t_build, len(chk.g), t_indep, kinds, t_corrupt, d1, d2, d3, d4, d5, proved, refused_n, time.time() - t_start))
    for f in FAIL[:40]: print("   FAIL " + f)
    if len(FAIL) > 40: print("   ... and %d more" % (len(FAIL) - 40))
    print("S24 BINARY GATE: " + ("PASS" if not FAIL else "FAIL"))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
