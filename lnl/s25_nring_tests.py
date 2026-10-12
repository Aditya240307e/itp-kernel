"""S25 gate: `ring` over the binary naturals `N` (native, no detour through the unary numerals), and the lemma set it shares with the unary `ring`.

`ring` (ring.py) is UNTRUSTED. It normalises both sides of an equation to a sum of monomials and builds, at every step, an application of one of the `ring_*` lemmas of the number system (ring_lib.py: one
template, instantiated for `Nat` with the prefix `ring_` and for `N` with the prefix `N.ring_`); the kernel checks the finished term like any other proof. S23 pins down `ring` on `Nat`; this gate pins down what
is new with `N`, and the instance machinery both share:

  A. the 21 lemmas of the template have the intended statements for BOTH number systems (a table written out here with explicit operations, compared as terms with what the library declared);
  B. the numerals: `lit` builds exactly the canonical numeral of a value (digits of the binary representation, `Succ` chains for Nat) and `numeral` reads back exactly those -- every other term, including every
     unevaluated expression and every term that is nearly a numeral, is not a numeral;
  C. fixed identities over N, proved natively (the proof mentions the `N.ring_*` lemmas and neither a `ring_*` lemma of Nat nor the bridge `to_nat`): powers of sums, several atoms, numerals of 100 bits,
     `N.succ`, applications as atoms, exponents that are closed expressions, a hypothesis in the context; the false ones are refused with both normal forms; a goal that is no equation is refused; an
     equation over a type with no number-system instance is refused with its own message;
  D. random expressions with numerals up to 70 bits against Python polynomial arithmetic: a rewriting of an expression is proved, a perturbed one is proved iff the polynomials agree, else refused;
  E. the normal form itself, for N and for Nat, is the Python polynomial of the expression (monomials, exponents and coefficients), atom by atom;
  F. corrupted ring proofs over N: the kernel and the independent checker give the same verdict on every one, and the kernel rejects most;
  G. the independent checker re-checks every global of the world (the whole library, the ring lemmas of both systems, every theorem proved here);
  H. speed: (a+b)^8, (a+b+c)^4 and a power of a sum with 33 terms are admitted in a few seconds.
"""
import os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(200000)
import itp_env_lnl as E
import bin_lib, ring as R
from author import AuthorError
from tactics import Proof
import indep_checker as ic
from harness import parse_raw
from s23_ring_tests import build_env, rebuild, dag_nodes, poly, clean, perturb, equiv, show_expr, size

FAIL = []; CHECKS = [0]
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond

def refused_with(thunk, *needles):
    """the message of the AuthorError the thunk raises ('' when it does not raise)"""
    try: thunk()
    except AuthorError as e: return str(e)
    return ''


def consts_in(k, term):
    """the global ids a term mentions (walked over the DAG, once per node)"""
    seen, out, stack = set(), set(), [term]
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); v = k.view(n)
        if v[0] == 'const': out.add(v[1])
        elif v[0] in ('app', 'lam', 'pi'): stack.append(v[1]); stack.append(v[2])
    return out


# ---------------------------------------------------------------------------------------------------------------------- the two number systems, described independently of ring_lib
def instances(w):
    Nat, N = w['Nat'], w['N']
    return {
        'Nat': dict(T=Nat, ET=Nat, Z=w['Zero'], one=w['Succ'](w['Zero']), add=w['add'], mul=w['mul'], pow=w['pow'], eadd=w['add'], succ=w['Succ'], pfx='ring_'),
        'N':   dict(T=N, ET=N, Z=w['N0'], one=w['Npos'](w['XH']), add=w['N.add'], mul=w['N.mul'], pow=w['N.pow'], eadd=w['N.add'], succ=w['N.succ'], pfx='N.ring_'),
    }


def ring_spec(w, I):
    """the statements of the 21 lemmas of the ring template, written out with explicit operations (not the library's template, not the notation)"""
    T, ET, Z, one = I['T'], I['ET'], I['Z'], I['one']
    ez, eo = I.get('ez', Z), I.get('eo', one)       # the exponents are of the same kind as the numbers (for a ring of integers they are naturals: the instance says so)
    add, mul, pw, eadd, succ = I['add'], I['mul'], I['pow'], I['eadd'], I['succ']
    Eq = w.Eq; F = w.forall_
    S = {}
    S['add_congr'] = F([T, T, T, T, lambda a, b, c, d: Eq(a, c), lambda a, b, c, d, h: Eq(b, d)], lambda a, b, c, d, h, g: Eq(add(a, b), add(c, d)))
    S['mul_congr'] = F([T, T, T, T, lambda a, b, c, d: Eq(a, c), lambda a, b, c, d, h: Eq(b, d)], lambda a, b, c, d, h, g: Eq(mul(a, b), mul(c, d)))
    S['pow_congr'] = F([T, T, ET, lambda a, c, e: Eq(a, c)], lambda a, c, e, h: Eq(pw(a, e), pw(c, e)))
    S['pow_exp_congr'] = F([T, ET, ET, lambda a, e, f: Eq(e, f)], lambda a, e, f, h: Eq(pw(a, e), pw(a, f)))
    S['atom'] = F([T], lambda x: Eq(x, add(mul(pw(x, eo), one), Z)))
    S['succ'] = F([T], lambda x: Eq(succ(x), add(x, one)))
    S['add_lt'] = F([T, T, T, T, lambda a, A, B, C: Eq(add(A, B), C)], lambda a, A, B, C, h: Eq(add(add(a, A), B), add(a, C)))
    S['add_gt'] = F([T, T, T, T, lambda b, A, B, C: Eq(add(A, B), C)], lambda b, A, B, C, h: Eq(add(A, add(b, B)), add(b, C)))
    S['add_eq'] = F([T, T, T, T, T, T, lambda a, b, c, A, B, C: Eq(add(a, b), c), lambda a, b, c, A, B, C, h1: Eq(add(A, B), C)],
                    lambda a, b, c, A, B, C, h1, h2: Eq(add(add(a, A), add(b, B)), add(c, C)))
    S['mono_add'] = F([T, ET, T, T, T, lambda x, e, a, b, s: Eq(add(a, b), s)], lambda x, e, a, b, s, h: Eq(add(mul(pw(x, e), a), mul(pw(x, e), b)), mul(pw(x, e), s)))
    S['mono_lt'] = F([T, ET, T, T, T, lambda x, e, A, B, R_: Eq(mul(A, B), R_)], lambda x, e, A, B, R_, h: Eq(mul(mul(pw(x, e), A), B), mul(pw(x, e), R_)))
    S['mono_gt'] = F([T, ET, T, T, T, lambda y, f, A, B, R_: Eq(mul(A, B), R_)], lambda y, f, A, B, R_, h: Eq(mul(A, mul(pw(y, f), B)), mul(pw(y, f), R_)))
    S['mul_mul_mul'] = F([T, T, T, T], lambda a, b, c, d: Eq(mul(mul(a, b), mul(c, d)), mul(mul(a, c), mul(b, d))))
    S['mono_eq'] = F([T, ET, ET, ET, T, T, T, lambda x, e, f, g, A, B, R_: Eq(eadd(e, f), g), lambda x, e, f, g, A, B, R_, h1: Eq(mul(A, B), R_)],
                     lambda x, e, f, g, A, B, R_, h1, h2: Eq(mul(mul(pw(x, e), A), mul(pw(x, f), B)), mul(pw(x, g), R_)))
    S['scale_nil'] = F([T], lambda M: Eq(mul(M, Z), Z))
    S['scale_cons'] = F([T, T, T, T, T, lambda M, N_, Q, MN, R_: Eq(mul(M, N_), MN), lambda M, N_, Q, MN, R_, h1: Eq(mul(M, Q), R_)],
                        lambda M, N_, Q, MN, R_, h1, h2: Eq(mul(M, add(N_, Q)), add(MN, R_)))
    S['mulp_nil'] = F([T], lambda Q: Eq(mul(Z, Q), Z))
    S['mulp_cons'] = F([T, T, T, T, T, T, lambda M, P, Q, A, B, C: Eq(mul(M, Q), A), lambda M, P, Q, A, B, C, h1: Eq(mul(P, Q), B), lambda M, P, Q, A, B, C, h1, h2: Eq(add(A, B), C)],
                       lambda M, P, Q, A, B, C, h1, h2, h3: Eq(mul(add(M, P), Q), C))
    S['pow_zero'] = F([T], lambda P: Eq(pw(P, ez), add(one, Z)))
    S['pow_one'] = F([T], lambda P: Eq(pw(P, eo), P))
    S['pow_split'] = F([T, ET, ET, ET, T, T, T, lambda P, e1, e2, e, A, B, C: Eq(eadd(e1, e2), e), lambda P, e1, e2, e, A, B, C, he: Eq(pw(P, e1), A),
                        lambda P, e1, e2, e, A, B, C, he, h1: Eq(pw(P, e2), B), lambda P, e1, e2, e, A, B, C, he, h1, h2: Eq(mul(A, B), C)],
                       lambda P, e1, e2, e, A, B, C, he, h1, h2, h3: Eq(pw(P, e), C))
    return S


# ---------------------------------------------------------------------------------------------------------------------- expressions with big numerals (the oracle is S23's polynomial arithmetic)
def rand_numeral(rng):
    c = rng.random()
    if c < 0.45: return rng.randrange(0, 8)
    if c < 0.7: return rng.randrange(0, 1 << 8)
    if c < 0.9: return rng.randrange(0, 1 << 24)
    return rng.randrange(0, 1 << 40)

def small_numeral(rng): return rng.randrange(0, 7)

def rand_expr(rng, natoms, depth, numeral=rand_numeral):
    if depth == 0 or rng.random() < 0.22:
        c = rng.random()
        if c < 0.55: return ('x', rng.randrange(natoms))
        if c < 0.82: return ('n', numeral(rng))
        return ('pred', rng.randrange(natoms))             # an application (the doubling function of the library, `pred` for Nat): an atom of its own
    r = rng.random()
    if r < 0.4: return ('+', rand_expr(rng, natoms, depth - 1, numeral), rand_expr(rng, natoms, depth - 1, numeral))
    if r < 0.75: return ('*', rand_expr(rng, natoms, depth - 1, numeral), rand_expr(rng, natoms, depth - 1, numeral))
    if r < 0.9: return ('^', rand_expr(rng, natoms, depth - 1, numeral), rng.randrange(0, 4))
    return ('S', rand_expr(rng, natoms, depth - 1, numeral))


class Syntax:
    """builds expression trees in one of the two number systems"""
    def __init__(self, w, name):
        self.w = w; self.name = name
        if name == 'N':
            self.lit = w.nota.by_type[w['N'].n].lit; self.succ = w['N.succ']; self.app_atom = w['N.double']
            self.add, self.mul, self.pw = w['N.add'], w['N.mul'], w['N.pow']
        else:
            self.lit = w.nota.by_type[w['Nat'].n].lit; self.succ = w['Succ']; self.app_atom = w['pred']
            self.add, self.mul, self.pw = w['add'], w['mul'], w['pow']
    def build(self, e, xs):
        k = e[0]
        if k == 'x': return xs[e[1]]
        if k == 'pred': return self.app_atom(xs[e[1]])
        if k == 'n': return self.lit(e[1])
        if k == 'S': return self.succ(self.build(e[1], xs))
        if k == '^': return self.pw(self.build(e[1], xs), self.lit(e[2]))
        a, b = self.build(e[1], xs), self.build(e[2], xs)
        return self.add(a, b) if k == '+' else self.mul(a, b)


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))       # a mistake that builds a unary numeral of 2^70 must end in a MemoryError, not take the machine down
    t_start = time.time(); rng = random.Random(25)
    t0 = time.time(); w = bin_lib.build_bin_defs(); k = w.k
    Nat, N = w['Nat'], w['N']; Eq = w.Eq; F = w.forall_
    I = instances(w); SN, SNat = Syntax(w, 'N'), Syntax(w, 'Nat')
    lit_n = SN.lit; nadd, nmul, npow = w['N.add'], w['N.mul'], w['N.pow']

    # ------------------------------------------------------------------------------------------------------------------ A. the statements of the template lemmas, for both systems
    for sysname, inst in I.items():
        spec = ring_spec(w, inst); n_ok = 0
        for name, stmt in spec.items():
            have = inst['pfx'] + name
            if expect(have in w, 'A  %s declares %s' % (sysname, have)): n_ok += expect(w.global_type(have) == stmt, 'A  %s has the intended statement (%s)' % (have, sysname))
        extra = [n for n in w.gid if n.startswith(inst['pfx']) and n[len(inst['pfx']):] not in spec]
        expect(not extra, 'A  the %s ring lemma set has lemmas the table does not know: %s' % (sysname, extra[:5]))

    # ------------------------------------------------------------------------------------------------------------------ B. numerals
    inst_nat, inst_n = R.instance_for(w, Nat), R.instance_for(w, N)
    expect(isinstance(inst_nat, R.SemiringNat) and isinstance(inst_n, R.SemiringN), 'B  the instance of each carrier is found')
    expect(R.instance_for(w, w['Pos']) is None and R.instance_for(w, w['Bool'] if 'Bool' in w else w['Pos']) is None, 'B  a type with no ring lemmas has no instance')
    def binary_numeral(v):
        """written out here: N0, or Npos of the digits below the leading one, most significant first"""
        if v == 0: return w['N0']
        p = w['XH']
        for bit in bin(v)[3:]: p = (w['XI'] if bit == '1' else w['XO'])(p)
        return w['Npos'](p)
    def unary_numeral(v):
        t = w['Zero']
        for _ in range(v): t = w['Succ'](t)
        return t
    values = list(range(0, 40)) + [63, 64, 65, 127, 128, 255, 256, 1000, 1023, 1024, 2**31 - 1, 2**31, 2**32, 2**63, 2**64 - 1, 2**64, 10**18, 10**30 + 7, 2**100, 2**200 + 12345]
    values += [rng.randrange(0, 1 << rng.choice([5, 20, 70, 150])) for _ in range(30)]
    for v in values:
        t = inst_n.lit(v)
        expect(t == binary_numeral(v), 'B  N lit(%d) is the canonical numeral' % v)
        expect(inst_n.numeral(t) == v, 'B  N numeral(lit(%d)) reads back' % v)
        expect(w.normalize(t) == t, 'B  the numeral %d is a normal form' % v)
    for v in list(range(0, 30)) + [60, 199, 200, 201, 250]:
        t = inst_nat.lit(v)
        expect(t == unary_numeral(v), 'B  Nat lit(%d) is the canonical numeral' % v)
        expect(inst_nat.numeral(t) == v, 'B  Nat numeral(lit(%d)) reads back' % v)
    # the bridge agrees with `lit` (the kernel computes it): of_nat of the unary numeral is the binary one, to_nat of the binary one is the unary one
    for v in list(range(0, 24)) + [50, 100, 200, 255, 256]:
        expect(w.normalize(w['N.of_nat'](inst_nat.lit(v))) == inst_n.lit(v), 'B  of_nat (lit %d) computes to the binary numeral' % v)
        expect(w.normalize(w['N.to_nat'](inst_n.lit(v))) == inst_nat.lit(v), 'B  to_nat (lit %d) computes to the unary numeral' % v)
    # everything else is not a numeral: unevaluated expressions, nearly-numerals, numerals of the other system
    one_n, two_n = inst_n.lit(1), inst_n.lit(2)
    near_n = {
        'N.succ 0': w['N.succ'](w['N0']), '1 + 1': nadd(one_n, one_n), 'Npos (Pos.succ XH)': w['Npos'](w['Pos.succ'](w['XH'])), 'Npos (XO (Pos.succ XH))': w['Npos'](w['XO'](w['Pos.succ'](w['XH']))),
        'XH (a Pos, not an N)': w['XH'], 'Npos XH + 0': nadd(one_n, w['N0']), 'N.double 1': w['N.double'](one_n), 'a unary numeral': inst_nat.lit(2), 'a bound variable': k.mk_bvar(0),
    }
    for label, t in near_n.items():
        t = t if hasattr(t, 'n') else w.T(t)
        expect(inst_n.numeral(t) is None, 'B  N numeral() refuses %s' % label)
    near_nat = {'pred 1': w['pred'](inst_nat.lit(1)), 'Succ (add 0 0)': w['Succ'](w['add'](w['Zero'], w['Zero'])), 'add 1 1': w['add'](inst_nat.lit(1), inst_nat.lit(1)), 'mul 0 0': w['mul'](w['Zero'], w['Zero']), 'a binary numeral': inst_n.lit(2),
                'a bound variable': w.T(k.mk_bvar(0))}
    for label, t in near_nat.items():
        expect(inst_nat.numeral(t) is None, 'B  Nat numeral() refuses %s' % label)

    # ------------------------------------------------------------------------------------------------------------------ C. fixed identities, proved natively
    counter = [0]; native_lemmas = {inst_n.pfx + n for n in ring_spec(w, I['N'])}
    gid_of = {n: g for n, g in w.gid.items()}
    nat_ring_ids = {gid_of[I['Nat']['pfx'] + n] for n in ring_spec(w, I['Nat'])}
    bridge_ids = {g for n, g in w.gid.items() if 'to_nat' in n or n in ('N.of_nat',)}
    kernel_only = set()          # theorems about numerals of 100+ bits: the Python checker evaluates binary arithmetic by structural recursion and takes minutes on them; the kernel takes milliseconds
    def prove_n(names, stmt_fn, label, hyp=None, heavy=False):
        counter[0] += 1; nm = 's25_%d' % counter[0]
        if heavy: kernel_only.add(nm)
        doms = [N] * len(names) + ([hyp] if hyp is not None else [])
        stmt = F(doms, stmt_fn)
        try:
            w.prove(nm, stmt, lambda p: (p.intro(*names, *(['h'] if hyp is not None else [])), p.ring()))
        except AuthorError as e:
            FAIL.append('C  ring failed on %s: %s' % (label, str(e).splitlines()[0])); return None
        used = consts_in(k, w.global_value(nm).n)
        expect(not (used & nat_ring_ids) and not (used & bridge_ids), 'C  the proof of %s uses neither the unary ring lemmas nor the bridge' % label)
        expect(any(w.gname.get(g, '').startswith('N.ring_') for g in used) or label.startswith('numerals'), 'C  the proof of %s uses the N.ring_* lemmas' % label)
        return nm
    two = lit_n(2); three = lit_n(3)
    prove_n(['a', 'b'], lambda a, b: Eq(nadd(a, b), nadd(b, a)), 'comm')
    prove_n(['a', 'b'], lambda a, b: Eq(npow(nadd(a, b), two), nadd(nadd(npow(a, two), nmul(two, nmul(a, b))), npow(b, two))), '(a+b)^2')
    prove_n(['a', 'b'], lambda a, b: Eq(npow(nadd(a, b), three), nadd(nadd(npow(a, three), nmul(three, nmul(npow(a, two), b))), nadd(nmul(three, nmul(a, npow(b, two))), npow(b, three)))), '(a+b)^3')
    prove_n(['a', 'b', 'c'], lambda a, b, c: Eq(npow(nadd(nadd(a, b), c), two), nadd(nadd(nadd(npow(a, two), npow(b, two)), npow(c, two)), nmul(two, nadd(nadd(nmul(a, b), nmul(b, c)), nmul(a, c))))), '(a+b+c)^2')
    prove_n(['a'], lambda a: Eq(nmul(w['N.succ'](a), w['N.succ'](a)), nadd(nadd(nmul(a, a), nmul(two, a)), lit_n(1))), 'N.succ')
    prove_n(['a'], lambda a: Eq(npow(nadd(w['N.double'](a), lit_n(1)), two), nadd(nadd(npow(w['N.double'](a), two), nmul(two, w['N.double'](a))), lit_n(1))), 'application as an atom')
    prove_n(['a'], lambda a: Eq(npow(a, lit_n(0)), lit_n(1)), 'a^0')
    prove_n(['a'], lambda a: Eq(nadd(nmul(npow(a, lit_n(1)), lit_n(1)), lit_n(0)), a), 'a^1')
    prove_n(['a', 'b'], lambda a, b: Eq(nadd(nmul(lit_n(0), a), nmul(b, lit_n(0))), lit_n(0)), 'zeros')
    big = 10**30 + 7
    prove_n(['a'], lambda a: Eq(npow(nadd(a, lit_n(big)), two), nadd(nadd(npow(a, two), nmul(lit_n(2 * big), a)), lit_n(big * big))), '100-bit coefficients', heavy=True)
    prove_n(['a', 'b'], lambda a, b: Eq(nmul(nadd(a, lit_n(2**64)), nadd(b, lit_n(2**64 + 1))), nadd(nadd(nmul(a, b), nmul(lit_n(2**64 + 1), a)), nadd(nmul(lit_n(2**64), b), lit_n(2**64 * (2**64 + 1))))), '64-bit coefficients')
    prove_n(['a'], lambda a: Eq(npow(a, nadd(two, three)), nmul(npow(a, two), npow(a, three))), 'an exponent that is a sum of numerals')
    prove_n(['a'], lambda a: Eq(npow(a, nmul(two, three)), npow(npow(a, two), three)), 'an exponent that is a product of numerals')
    prove_n(['a'], lambda a: Eq(npow(npow(a, lit_n(10)), lit_n(10)), npow(a, lit_n(100))), 'a power of a power')
    # closed numerals only, in the world's own arithmetic
    for x, y, op, heavy in ((rng.randrange(1 << 64), rng.randrange(1 << 64), 'add', False), (rng.randrange(1 << 40), rng.randrange(1 << 40), 'mul', False), (rng.randrange(1 << 150), rng.randrange(1 << 150), 'add', True),
                            (rng.randrange(1 << 100), rng.randrange(1 << 100), 'mul', True), (12345678901234567890, 3, 'pow', True)):
        if op == 'add': s = Eq(nadd(lit_n(x), lit_n(y)), lit_n(x + y))
        elif op == 'mul': s = Eq(nmul(lit_n(x), lit_n(y)), lit_n(x * y))
        else: s = Eq(npow(lit_n(x), lit_n(y)), lit_n(x ** y))
        counter[0] += 1; nm = 's25_num%d' % counter[0]
        if heavy: kernel_only.add(nm)
        try: w.prove(nm, s, lambda p: p.ring()); ok = True
        except AuthorError as e: ok = False
        expect(ok, 'C  ring proves a closed identity of numerals (%s, %d-bit operands)' % (op, max(x, y).bit_length()))
    # a hypothesis in the context does not matter
    prove_n(['a', 'b'], lambda a, b, h: Eq(npow(nadd(a, b), two), nadd(nadd(npow(a, two), nmul(two, nmul(a, b))), npow(b, two))), 'a hypothesis in the context', hyp=lambda a, b: Eq(a, b))
    # the false ones are refused with both normal forms
    for label, stmt_fn, names in (('(a+b)^2 = a^2+b^2', lambda a, b: Eq(npow(nadd(a, b), two), nadd(npow(a, two), npow(b, two))), ['a', 'b']), ('a+1 = a', lambda a: Eq(nadd(a, lit_n(1)), a), ['a']),
                                  ('a*b = a+b', lambda a, b: Eq(nmul(a, b), nadd(a, b)), ['a', 'b']), ('2^64 a = 2^64 a + 1', lambda a: Eq(nmul(lit_n(2**64), a), nadd(nmul(lit_n(2**64), a), lit_n(1))), ['a']),
                                  ('a*(2^64+1) = a*2^64', lambda a: Eq(nmul(a, lit_n(2**64 + 1)), nmul(a, lit_n(2**64))), ['a'])):
        counter[0] += 1
        msg = refused_with(lambda: w.prove('s25_bad%d' % counter[0], F([N] * len(names), stmt_fn), lambda p: (p.intro(*names), p.ring())))
        expect('different normal forms' in msg and 'left:' in msg and 'right:' in msg, 'C  ring refuses the false identity %s with both normal forms' % label)
    # what the refusal shows: the two normal forms, spelled out (coefficients, exponents, products, sums; a constant alone; zero)
    def nfs(stmt, names):
        counter[0] += 1
        msg = refused_with(lambda: w.prove('s25_nf%d' % counter[0], stmt, lambda p: (p.intro(*names), p.ring())))
        lines = {l.split(':')[0].strip(): l.split(':', 1)[1].strip() for l in msg.split('\n') if l.strip().startswith(('left:', 'right:'))}
        return lines.get('left'), lines.get('right')
    for label, stmt, names, want in (
            ('(a+b)^2 = a^2+b^2', F([N, N], lambda a, b: Eq(npow(nadd(a, b), two), nadd(npow(a, two), npow(b, two)))), ['a', 'b'], ('a^2 + 2*a*b + b^2', 'a^2 + b^2')),
            ('a+1 = a', F([N], lambda a: Eq(nadd(a, lit_n(1)), a)), ['a'], ('a + 1', 'a')),
            ('a*b = 5', F([N, N], lambda a, b: Eq(nmul(a, b), lit_n(5))), ['a', 'b'], ('a*b', '5')),
            ('0*a = 1', F([N], lambda a: Eq(nmul(lit_n(0), a), lit_n(1))), ['a'], ('0', '1')),
            ('a^(0*3) = a', F([N], lambda a: Eq(npow(a, nmul(lit_n(0), lit_n(3))), a)), ['a'], ('1', 'a')),
            ('3*a^2*b + a*b^3 = a*b', F([N, N], lambda a, b: Eq(nadd(nmul(lit_n(3), nmul(npow(a, two), b)), nmul(a, npow(b, three))), nmul(a, b))), ['a', 'b'], ('3*a^2*b + a*b^3', 'a*b'))):
        expect(nfs(stmt, names) == want, 'C  the refusal of %s shows the normal forms %s (got %s)' % (label, want, nfs(stmt, names)))
    # an exponent that is a closed expression with value zero or one is the numeral
    prove_n(['a'], lambda a: Eq(npow(a, nmul(lit_n(0), lit_n(3))), lit_n(1)), 'a^(0*3) = 1')
    prove_n(['a'], lambda a: Eq(npow(a, nadd(lit_n(0), lit_n(0))), lit_n(1)), 'a^(0+0) = 1')
    prove_n(['a'], lambda a: Eq(npow(a, nmul(lit_n(1), lit_n(1))), a), 'a^(1*1) = a')
    # goals that are not equations of a ring
    for label, stmt, intros in (('a <= a', F([N], lambda a: w['N.le'](a, a)), ['a']), ('True', w['True'], []), ('a = a /\\ a = a', F([N], lambda a: Eq(a, a) & Eq(a, a)), ['a'])):
        counter[0] += 1
        msg = refused_with(lambda: w.prove('s25_ne%d' % counter[0], stmt, lambda p: (p.intro(*intros) if intros else None, p.ring())))
        expect('not an equation' in msg, 'C  ring refuses a goal that is not an equation (%s): %s' % (label, msg[:80]))
    # an equation over a type that has no ring lemmas
    Pos = w['Pos']; padd = w['Pos.add']
    counter[0] += 1
    msg = refused_with(lambda: w.prove('s25_pos%d' % counter[0], F([Pos, Pos], lambda a, b: Eq(padd(a, b), padd(b, a))), lambda p: (p.intro('a', 'b'), p.ring())))
    expect('no commutative-semiring structure' in msg and 'Pos' in msg, 'C  ring says it knows no number system for Pos: %s' % msg[:100])
    # a power with an exponent that is not a numeral is an atom (symbolic exponents are not supported yet): the identity that only needs it as an atom is proved
    prove_n(['a', 'n'], lambda a, n_: Eq(nmul(npow(a, n_), npow(a, n_)), npow(npow(a, n_), two)), 'a power with a variable exponent is an atom')

    # ------------------------------------------------------------------------------------------------------------------ D. random expressions against the Python oracle
    equal_proved = unequal_refused = failures = 0
    for case in range(220):
        natoms = rng.choice([1, 2, 2, 3]); depth = rng.choice([2, 3, 3, 4])
        e1 = rand_expr(rng, natoms, depth)
        if size(e1) > 40: continue
        if rng.random() < 0.5: e2 = equiv(equiv(e1, rng), rng)
        else: e2 = perturb(equiv(e1, rng), rng, natoms)
        if size(e2) > 90: continue
        same = clean(poly(e1, natoms)) == clean(poly(e2, natoms))
        names = ['a', 'b', 'c'][:natoms]
        stmt = F([N] * natoms, lambda *xs: Eq(SN.build(e1, xs), SN.build(e2, xs)))
        try:
            w.prove('s25_r%d' % case, stmt, lambda p: (p.intro(*names), p.ring())); got = 'proved'
        except AuthorError as e: got = 'refused' if 'different normal forms' in str(e) else 'error: ' + str(e).splitlines()[0]
        except (E.ITPError, RuntimeError) as e: got = 'KERNEL REJECTED: ' + str(e)[:200]
        if same:
            if got == 'proved': equal_proved += 1
            else: failures += 1; FAIL.append('D  ring should prove %s = %s but: %s' % (show_expr(e1), show_expr(e2), got))
        else:
            if got == 'refused': unequal_refused += 1
            else: failures += 1; FAIL.append('D  ring should refuse %s = %s (different polynomials) but: %s' % (show_expr(e1), show_expr(e2), got))
    expect(equal_proved >= 70 and unequal_refused >= 25, 'D  the random test is not vacuous (proved %d true identities, refused %d false ones)' % (equal_proved, unequal_refused))

    # ------------------------------------------------------------------------------------------------------------------ E. the normal form is the Python polynomial, for both systems
    n_nf = 0
    for case in range(120):
        natoms = rng.choice([1, 2, 3]); e = rand_expr(rng, natoms, rng.choice([2, 3, 4]), small_numeral)      # (unary numerals: nothing big)
        if size(e) > 50: continue
        oracle = clean(poly(e, natoms)); width = 2 * natoms
        for syn, sysname in ((SN, 'N'), (SNat, 'Nat')):
            A = N if sysname == 'N' else Nat
            names = ['a', 'b', 'c'][:natoms]
            p = Proof(w, F([A] * natoms, lambda *xs: Eq(syn.build(e, xs), syn.build(e, xs))), 'e%d' % case); p.intro(*names)
            xs = [p.h(nm) for nm in names]
            rg = R.Ring(w, R.instance_for(w, A)); NFe, _ = rg.nf(syn.build(e, xs))
            slot = {}
            for j in range(natoms): slot[xs[j]] = j; slot[syn.app_atom(xs[j])] = natoms + j
            got = {}
            for m in NFe.monos:
                vec = [0] * width
                for (i, ex) in m.vars: vec[slot[rg.atoms[i]]] = ex
                got[tuple(vec)] = m.coeff
            expect(got == oracle, 'E  the %s normal form of %s is the polynomial (got %s, expected %s)' % (sysname, show_expr(e), got, oracle))
            n_nf += 1
    expect(n_nf >= 150, 'E  the normal-form test is not vacuous (%d normal forms compared)' % n_nf)

    # ------------------------------------------------------------------------------------------------------------------ H. time (before the independent re-check, which is slow on big numerals)
    def comb(n, r):
        from math import comb as c
        return c(n, r)
    def sum_terms_n(terms, a, b):
        out = None
        for (c, i, j) in terms:
            t = lit_n(c)
            if i: t = nmul(t, npow(a, lit_n(i)))
            if j: t = nmul(t, npow(b, lit_n(j)))
            out = t if out is None else nadd(out, t)
        return out
    t1 = time.time()
    prove_n(['a', 'b'], lambda a, b: Eq(npow(nadd(a, b), lit_n(8)), sum_terms_n([(comb(8, i), 8 - i, i) for i in range(9)], a, b)), '(a+b)^8')
    dt8 = time.time() - t1; t1 = time.time()
    prove_n(['a', 'b', 'c'], lambda a, b, c: Eq(npow(nadd(nadd(a, b), c), lit_n(4)), nmul(npow(nadd(nadd(c, b), a), two), npow(nadd(nadd(a, c), b), two))), '(a+b+c)^4')
    dt4 = time.time() - t1; t1 = time.time()
    expect(dt8 < 6.0, 'H  (a+b)^8 over N is admitted in under 6 s (took %.2f s)' % dt8)
    expect(dt4 < 6.0, 'H  (a+b+c)^4 over N is admitted in under 6 s (took %.2f s)' % dt4)

    # ------------------------------------------------------------------------------------------------------------------ G. the independent checker over everything declared so far
    t2 = time.time()
    if os.environ.get('S25_TRACE'):
        print('   biggest proofs (tree size, name):', sorted(((k.node_info(k.global_info(g)[2])[2], k.global_info(g)[0]) for g in range(k.num_globals()) if k.global_info(g)[2] is not None), reverse=True)[:8], flush=True)
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        if name in kernel_only: continue
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid),
                recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    chk = ic.Checker(env, params=True, fuel=200_000_000, prop_elim=True)
    if os.environ.get('S25_TRACE'):
        times = []
        for gid, gl in chk.g.items():
            t_g = time.time(); chk.run(lambda: (chk.sort_of({}, gl.type), None if gl.value is None else chk.check({}, gl.value, gl.type))); times.append((time.time() - t_g, gl.name))
        times.sort(reverse=True); print('   slowest independent checks:', [(round(a, 1), b) for a, b in times[:12]], 'total %.1f s' % sum(a for a, _ in times), flush=True)
    bad = chk.check_definitions()
    expect(len(chk.g) >= 380 and not bad, 'G  the independent checker re-checks all %d globals (all but the %d theorems about 100-bit numerals) without a rejection: %s' % (len(chk.g), len(kernel_only), bad[:3]))
    tg = time.time() - t2

    # ------------------------------------------------------------------------------------------------------------------ F. corrupted ring proofs over N
    t3 = time.time()
    st = F([N, N], lambda a, b: Eq(npow(nadd(a, b), two), nadd(nadd(npow(a, two), nmul(two, nmul(a, b))), npow(b, two))))
    pr = Proof(w, st, 'e'); pr.intro('a', 'b'); pr.ring(); proof = pr.term()
    stmt_h = parse_raw(k.dump_raw(st.n), True)
    pool = [n for n in dag_nodes(k, proof.n) if k.view(n)[0] in ('app', 'const', 'lam')]
    ring_consts = [k.mk_const(w.gid[nm]) for nm in w.gid if nm.startswith('N.ring_')]
    kinds = {'kernel_rej': 0, 'kernel_acc': 0, 'disagree': 0, 'undecided': 0}
    for case in range(70):
        target = rng.choice(pool); v = k.view(target); r = rng.random()
        if r < 0.3: repl = rng.choice(pool)
        elif r < 0.5 and v[0] == 'app' and k.view(v[1])[0] == 'app':
            inner = k.view(v[1]); repl = k.mk_app(k.mk_app(inner[1], v[2]), inner[2])          # swap the last two arguments
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
        if kok != iok: kinds['disagree'] += 1; FAIL.append('F  the kernel (%s) and the independent checker (%s: %s) disagree about a corrupted ring proof over N' % (kok, iok, imsg))
        elif kok: kinds['kernel_acc'] += 1
        else: kinds['kernel_rej'] += 1
    expect(kinds['kernel_rej'] >= 25, 'F  the corruption test is not vacuous (%s)' % kinds)
    expect(k.check_open(proof.n, st.n, [])[0] and chk.check_closed(parse_raw(k.dump_raw(proof.n), True), stmt_h)[0], 'F  the uncorrupted proof is accepted by both checkers')
    tf = time.time() - t3

    # (declared AFTER the independent re-check: the checker reads proofs as TEXT, which spells out the tree of a proof with this much sharing)
    t1 = time.time()
    prove_n(['a'], lambda a: Eq(npow(nadd(a, lit_n(1)), lit_n(32)), nmul(npow(nadd(lit_n(1), a), lit_n(16)), npow(nadd(a, lit_n(1)), lit_n(16)))), '(a+1)^32 = (1+a)^16 (a+1)^16')
    dt32 = time.time() - t1
    expect(dt32 < 20.0, 'H  (a+1)^32 over N is admitted in under 20 s (took %.2f s)' % dt32)
    # a power with an exponent of 70 bits: repeated squaring makes the proof 70 levels deep, and the tree it spells out has 2^70 nodes (the kernel's facts about the term saturate; nothing may walk the tree)
    t1 = time.time()
    nm = prove_n(['a', 'b'], lambda a, b: Eq(nmul(npow(a, lit_n(2**70)), lit_n(0)), nmul(lit_n(0), b)), 'a numeral exponent of 70 bits against a zero coefficient')
    prove_n(['a'], lambda a: Eq(nmul(npow(a, lit_n(2**70)), npow(a, lit_n(2**70))), npow(a, lit_n(2**71))), 'a^(2^70) * a^(2^70) = a^(2^71)')
    dt70 = time.time() - t1
    expect(nm is not None and k.node_info(w.global_value(nm).n)[2] == 2**31 - 1, 'H  the tree of that proof is too big to count (its size saturates)')
    expect(dt70 < 10.0, 'H  powers with 70-bit exponents are admitted in under 10 s (took %.2f s)' % dt70)

    print("   %d checks; random expressions: %d true identities proved, %d false ones refused; %d normal forms compared; independent re-check of %d globals in %.1f s; %d corrupted proofs %s in %.1f s; (a+b)^8 %.2f s, (a+1)^32 %.2f s (%.1f s total)" %
          (CHECKS[0], equal_proved, unequal_refused, n_nf, len(chk.g), tg, sum(kinds.values()), kinds, tf, dt8, dt32, time.time() - t_start))
    for f in FAIL[:40]: print("   FAIL " + f)
    if len(FAIL) > 40: print("   ... and %d more" % (len(FAIL) - 40))
    print("S25 N-RING GATE: " + ("PASS" if not FAIL else "FAIL"))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
