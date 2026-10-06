"""noconf.py -- injectivity lemmas for constructors (the missing half of F2b; discrimination is cases + same_ctor).

For a family F whose slots are ALL parameters (arity 0 included: Nat, trees; and List, And, Or, Ex ...), constructor c and payload field q (a field that is not itself a parameter and whose
type mentions no other payload field) this declares, with p the family's parameters,
    Inj_c_q : forall p, F p -> F p -> Type0      match x { d f.. => match y { e g.. => if d = e = c then Eq A_q f_q g_q else True } }
    c_inj_q : forall p a.. b.., Eq (F p) (c p a..) (c p b..) -> Eq A_q a_q b_q
            = fun p a.. b.. h => transport (F p) (fun z => Inj_c_q p (c p a..) z) (c p a..) (c p b..) h (refl A_q a_q)
(a.. and b.. are the payload fields of c, in order). The lemma's type equals Inj_c_q p (c p a..) (c p b..) by delta and iota, and refl : Inj_c_q p (c p a..) (c p a..) likewise.
The branch binders of Inj carry the types the kernel's Match rule forces (a constructor's parameter fields := the shared parameters). Terms are built as NAMED terms (indep_checker's
format), converted to hybrid trees, and declared through the kernel's CHECKED path. Not generated: families with a true index (Eq, Le, Vec), and dependent payload fields (they need HEq)."""
import indep_checker as ic
from indep_checker import fresh

def lams(fields, body):
    for x, A in reversed(fields): body = ('lam', x, A, body)
    return body
def pis(fields, body):
    for x, A in reversed(fields): body = ('pi', x, A, body)
    return body

def generate_injectivity(knew, build_new, parse_raw, chk_factory, fam_gid, gids):
    """declares the lemmas in `knew`; returns {ctor name: [lemma global name per field, or None for a parameter field or a dependent payload field]}"""
    chk = chk_factory(); k = chk.arity(fam_gid)
    if not all(chk.param_slots(fam_gid)): raise ValueError('only families whose every slot is a parameter (no true index)')
    F = ('c', fam_gid); ctors = list(chk.g[fam_gid].ctors); TRUE, EQ, TR, RF = [('c', gids[n]) for n in ('True', 'Eq', 'transport', 'refl')]
    eq = lambda A, a, b: ic.apply_args(EQ, [A, a, b])
    fam_type = chk.g[fam_gid].type
    def telescope(names):                       # [(name, type)] for the given binder names, each type in terms of the earlier names
        t, out = fam_type, []
        for nm in names: out.append((nm, t[2])); t = ic.subst(t[3], t[1], ('v', nm))
        return out
    fam_app = lambda names: ic.apply_args(F, [('v', n) for n in names])
    declared = {}
    def declare(name, ty, val): return knew.declare_checked(build_new(knew, ic.to_hybrid(ty)), build_new(knew, ic.to_hybrid(val)), name)
    def forced_fields(d, pn):                   # ctor d's fields with their types specialised the way the kernel's Match rule does: its parameter fields := the shared parameter variables
        fd, rd = chk.open_ctor(d); sub = {rd[i][1]: ('v', pn[i]) for i in range(k)}
        def sp(t):
            for nm, val in sub.items(): t = ic.subst(t, nm, val)
            return t
        return [(x, sp(A)) for x, A in fd], rd
    for cg in ctors:
        cname = chk.g[cg].name; fields, res = chk.open_ctor(cg); slot_of = {res[i][1]: i for i in range(k)}
        payload = {x for x, _ in fields if x not in slot_of}; declared[cname] = []
        for qi, (qn, qt) in enumerate(fields):
            if qn in slot_of or any(n2 in ic.fv(qt) for n2 in payload): declared[cname].append(None); continue       # a parameter field, or a dependent payload field
            pn = [fresh('p') for _ in range(k)]; pt = telescope(pn); F_ps = fam_app(pn); x, y, z = fresh('x'), fresh('y'), fresh('z')
            mn = [fresh('i') for _ in range(k)]; mot = lams(telescope(mn) + [(fresh('s'), fam_app(mn))], ('s', 0))
            def outer(d):
                fd, rd = forced_fields(d, pn)
                def inner(e):
                    fe, re_ = forced_fields(e, pn)
                    return lams(fe, eq(fd[qi][1], ('v', fd[qi][0]), ('v', fe[qi][0])) if (d == cg and e == cg) else TRUE)
                return lams(fd, ('match', ('v', y), mot, tuple(inner(e) for e in ctors)))
            inj_val = lams(pt + [(x, F_ps), (y, F_ps)], ('match', ('v', x), mot, tuple(outer(d) for d in ctors)))
            inj_ty = pis(pt + [(fresh('x'), F_ps), (fresh('y'), F_ps)], ('s', 0))
            inj_name = 'Inj__%s_%d' % (cname, qi); inj_gid = declare(inj_name, inj_ty, inj_val)
            fc, rc = forced_fields(cg, pn); slot_c = {rc[i][1]: i for i in range(k)}; Aq = fc[qi][1]; qn_c = fc[qi][0]      # ONE field list (the forced one) for the whole lemma
            pay_order = [xj for xj, _ in fc if xj not in slot_c]
            h = fresh('h'); a_of = {xj: fresh('a') for xj in pay_order}; b_of = {xj: fresh('b') for xj in pay_order}
            def rename(t, m):                       # a payload field's type may mention an EARLIER payload field (Ex's proof mentions its witness): rename those to this side's binders
                for xk, v in m.items(): t = ic.subst(t, xk, ('v', v))
                return t
            a_f = [(a_of[xj], rename(A, a_of)) for (xj, A) in fc if xj not in slot_c]; b_f = [(b_of[xj], rename(A, b_of)) for (xj, A) in fc if xj not in slot_c]
            def capp(m): return ic.apply_args(('c', cg), [('v', pn[slot_c[xj]]) if xj in slot_c else ('v', m[xj]) for xj, _ in fc])
            h_ty = eq(F_ps, capp(a_of), capp(b_of))
            stmt = pis(pt + a_f + b_f + [(h, h_ty)], eq(Aq, ('v', a_of[qn_c]), ('v', b_of[qn_c])))
            proof = lams(pt + a_f + b_f + [(h, h_ty)], ic.apply_args(TR, [F_ps, ('lam', z, F_ps, ic.apply_args(('c', inj_gid), [('v', p) for p in pn] + [capp(a_of), ('v', z)])),
                                                                         capp(a_of), capp(b_of), ('v', h), ic.apply_args(RF, [Aq, ('v', a_of[qn_c])])]))
            lem_name = '%s_inj_%d' % (cname, qi); declare(lem_name, stmt, proof); declared[cname].append(lem_name)
    return declared
