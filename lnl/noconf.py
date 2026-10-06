"""noconf.py -- injectivity lemmas for constructors (the missing half of F2b; discrimination is cases + same_ctor).

For a NON-INDEXED, NON-PARAMETRIC family F (arity 0) with constructor c and payload field q (a field whose type mentions no other field) this declares
    Inj_c_q : F -> F -> Type0       match x { d f.. => match y { e g.. => if d = e = c then Eq A_q f_q g_q else True } }
    c_inj_q : forall a.. b.., Eq F (c a..) (c b..) -> Eq A_q a_q b_q
            = fun a.. b.. h => transport F (fun z => Inj_c_q (c a..) z) (c a..) (c b..) h (refl A_q a_q)
The lemma's type equals Inj_c_q (c a..) (c b..) by delta and iota, and refl : Inj_c_q (c a..) (c a..) likewise. Terms are built as NAMED terms (indep_checker's format),
converted to hybrid trees, and declared through the kernel's CHECKED path. Not generated: indexed or parametric families, and dependent fields (those need HEq)."""
import indep_checker as ic
from indep_checker import fresh

def lams(fields, body):
    for x, A in reversed(fields): body = ('lam', x, A, body)
    return body
def pis(fields, body):
    for x, A in reversed(fields): body = ('pi', x, A, body)
    return body

def generate_injectivity(knew, build_new, parse_raw, chk_factory, fam_gid, gids):
    """declares the lemmas in `knew`; returns {ctor name: [lemma global name per field, or None for a dependent field]}"""
    chk = chk_factory()
    if chk.arity(fam_gid) != 0: raise ValueError('only non-indexed, non-parametric families (arity 0)')
    F = ('c', fam_gid); ctors = list(chk.g[fam_gid].ctors); TRUE, EQ, TR, RF = [('c', gids[n]) for n in ('True', 'Eq', 'transport', 'refl')]
    eq = lambda A, a, b: ic.apply_args(EQ, [A, a, b])
    declared = {}
    def declare(name, ty, val): return knew.declare_checked(build_new(knew, ic.to_hybrid(ty)), build_new(knew, ic.to_hybrid(val)), name)
    for cg in ctors:
        cname = chk.g[cg].name; fields, _ = chk.open_ctor(cg); names = {x for x, _ in fields}; declared[cname] = []
        for qi, (qn, qt) in enumerate(fields):
            if any(n2 in ic.fv(qt) for n2 in names): declared[cname].append(None); continue          # a dependent field
            x, y, z = fresh('x'), fresh('y'), fresh('z'); mot = ('lam', z, F, ('s', 0))
            def outer(d):
                fd, _ = chk.open_ctor(d)
                def inner(e):
                    fe, _ = chk.open_ctor(e)
                    res = eq(qt, ('v', fd[qi][0]), ('v', fe[qi][0])) if (d == cg and e == cg) else TRUE
                    return lams(fe, res)
                return lams(fd, ('match', ('v', y), mot, tuple(inner(e) for e in ctors)))
            inj_val = ('lam', x, F, ('lam', y, F, ('match', ('v', x), mot, tuple(outer(d) for d in ctors))))
            inj_ty = ('pi', fresh('x'), F, ('pi', fresh('y'), F, ('s', 0)))
            inj_name = 'Inj__%s_%d' % (cname, qi); inj_gid = declare(inj_name, inj_ty, inj_val)
            a_f = [(fresh('a'), A) for _, A in fields]; b_f = [(fresh('b'), A) for _, A in fields]; h = fresh('h')
            capp = lambda fs: ic.apply_args(('c', cg), [('v', x_) for x_, _ in fs])
            h_ty = eq(F, capp(a_f), capp(b_f))
            stmt = pis(a_f + b_f + [(h, h_ty)], eq(qt, ('v', a_f[qi][0]), ('v', b_f[qi][0])))
            proof = lams(a_f + b_f + [(h, h_ty)], ic.apply_args(TR, [F, ('lam', z, F, ic.apply_args(('c', inj_gid), [capp(a_f), ('v', z)])), capp(a_f), capp(b_f), ('v', h), ic.apply_args(RF, [qt, ('v', a_f[qi][0])])]))
            lem_name = '%s_inj_%d' % (cname, qi); declare(lem_name, stmt, proof); declared[cname].append(lem_name)
    return declared
