"""lincomb.py -- `linear_combination`: close an equation of a commutative ring from other equations, given the coefficients.

    p.linear_combination((c1, 'h1'), (c2, 'h2'), ...)         goal  lhs = rhs  from  h_i : l_i = r_i

is the kernel-checked version of "the goal is the sum of c_i times the hypotheses": it succeeds when  lhs - rhs = sum c_i (l_i - r_i)  holds as a polynomial identity. A coefficient is a term, a function of
the Proof (so that it can mention the hypotheses by name), or a Python int (a numeral of the ring, of either sign).

The proof is built from the ring's own lemmas and checked by the kernel like every other:
    E  :  S1 = S2                         with  S1 = sum c_i * l_i ,  S2 = sum c_i * r_i        (the congruence lemmas `ring_mul_congr` / `ring_add_congr` applied to the hypotheses)
    P  :  lhs + S2 = rhs + S1             by `ring` (the identity above, rearranged so that no subtraction is needed)
    then  rhs + S1 = rhs + S2  from E, so  lhs + S2 = rhs + S2  and the common summand is cancelled by the ring's `add_right_cancel`.
Nothing is trusted: a wrong coefficient makes `ring` fail (and both normal forms are shown), a wrong hypothesis makes the kernel refuse the congruence."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import ring as R
from author import AuthorError


def _cancel_order(p, ty, name):
    """the positions (a, b, c) of the three leading arguments of the cancellation lemma  a + c = b + c -> a = b  : the libraries state it with the variables in different orders (`N.add_right_cancel a b c`,
    but `add_right_cancel a c b` for the unary naturals), so read the order off the statement"""
    holes, body = p._peel(ty, 3)
    if len(holes) != 3: raise AuthorError('linear_combination: the lemma %s does not have the expected shape' % name)
    parts = p.w.pi_parts(body)
    if parts is None: raise AuthorError('linear_combination: the lemma %s does not have the expected shape' % name)
    _, concl = parts
    _, left, right = p.w.eq_parts(concl)
    ids = [h.n for h, _ in holes]
    if left.n not in ids or right.n not in ids or left.n == right.n: raise AuthorError('linear_combination: the lemma %s does not have the expected shape' % name)
    ia, ib = ids.index(left.n), ids.index(right.n); ic = ({0, 1, 2} - {ia, ib}).pop()
    return ia, ib, ic


def linear_combination(p, *pairs):
    w = p.w
    if not pairs: return p.ring()
    g = p.goal()
    with p.focus():
        A, lhs, rhs = w.eq_parts(g.target)
        inst = R.instance_for(w, A)
        if inst is None: raise AuthorError('linear_combination: no ring structure is known for the type %s' % w.show(A))
        pfx = inst.pfx
        try:
            cancel_name = pfx.replace('ring_', '') + 'add_right_cancel'
            cancel_ty = w.global_type(cancel_name)
            add_congr, mul_congr = w[pfx + 'add_congr'], w[pfx + 'mul_congr']
        except Exception:
            raise AuthorError('linear_combination: the library has no congruence / cancellation lemmas for the type %s' % w.show(A)) from None
        order = _cancel_order(p, cancel_ty, cancel_name)
        refl = lambda t: w['refl'](A, t)
        S1 = S2 = E = None
        for c, h in pairs:
            ct = inst.lit(c) if isinstance(c, int) and not isinstance(c, bool) else p.t(c)
            ht = p.t(h); B, l, r = w.eq_parts(w.type_of(ht))
            if not w.conv(B, A): raise AuthorError('linear_combination: the hypothesis %s is an equation in %s, the goal one in %s' % (h if isinstance(h, str) else w.show(ht), w.show(B), w.show(A)))
            ei = mul_congr(ct, l, ct, r, refl(ct), ht)
            if S1 is None: S1, S2, E = inst.mul(ct, l), inst.mul(ct, r), ei
            else: E = add_congr(S1, inst.mul(ct, l), S2, inst.mul(ct, r), E, ei); S1, S2 = inst.add(S1, inst.mul(ct, l)), inst.add(S2, inst.mul(ct, r))
        rg = R.Ring(w, inst)
        pf, (na, nb) = rg.prove_eq(inst.add(lhs, S2), inst.add(rhs, S1))
        if pf is None:
            raise AuthorError('linear_combination failed: the combination does not close the goal\n  left:  %s\n  right: %s\n%s' % (rg.show_nf(na), rg.show_nf(nb), p))
        step1 = add_congr(rhs, S1, rhs, S2, refl(rhs), E)
        step2 = w['trans'](A, inst.add(lhs, S2), inst.add(rhs, S1), inst.add(rhs, S2), pf.proof, step1)
        args = [None, None, None]
        args[order[0]], args[order[1]], args[order[2]] = lhs, rhs, S2           # the first and second summands and the common one, in the order this ring's lemma takes them
        proof = w[cancel_name](*args, step2)
    p.exact(proof); p.trace.append(('linear_combination',))
    return p
