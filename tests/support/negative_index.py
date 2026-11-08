"""negative_index.py -- finding F92: a family that occurs in an index of its own constructors is inconsistent with two standard classical axioms.

    R : Prop -> Type        mkr : forall A : Prop, R (R A -> False)

R has itself to the left of an arrow inside a RESULT INDEX.  Write E x := (R x -> False).  The only constructor gives R (E B) for every proposition B, and the elimination of an inhabitant of R x into a
proposition gives  exists B, E B = x.  So R x is inhabited exactly when x is a value of E: E x is refuted when x is a value of E (K1) and holds when it is not (K3), and neither lemma uses an axiom.  Excluded
middle and propositional extensionality make every proposition equal to True or to False, so E has at most two values, and the cases for E False and E True each contradict "E x holds exactly when x is not a
value of E".  `boom : False` is the result.  (In a model that interprets Prop by two truth values R has no interpretation: E would be a map of {0,1} to itself with E x = 1 exactly when x is not in its
image, and none of the four such maps is.)  The fault is in the rules for R, not in the encoding: Lean's kernel refuses R.

build(checked) is used by the gate S15 (part C).  With checked=True the family goes through the kernel's checked entry, which must refuse it (the positivity condition looks in the result indices).  With
checked=False it goes through the UNCHECKED entry, which exists for tests: then the kernel proves False from the two axioms (the exploit is real and the gate is not vacuous), and the independent checker
must refuse the world because of R's admission and for no other reason."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import itp_env_lnl as E
import author
import lnl_base


def declare_unchecked(w, name, ty, ctors):
    """author.World.inductive, finished through the kernel's UNCHECKED entry: what a front end that skipped the admission checks would do"""
    tid = w.k.reserve_inductive_type(name, w._n(ty)); w._register(name, tid)
    specs = []
    for cname, fields, *rest in ctors:
        idx_fn = rest[0] if rest else None
        vars_, doms = [], []
        try:
            for d in fields:
                dom = w.term(d(*vars_) if (callable(d) and not isinstance(d, author.Term)) else d, 'constructor field type')
                lvl, x = w._bind(dom); vars_.append(x); doms.append(dom)
            idxs = [w.term(i, 'constructor result index') for i in (idx_fn(*vars_) if idx_fn else [])]
        finally: w._unbind(len(vars_))
        specs.append(E.CtorSpec(cname, [d.n for d in doms], [i.n for i in idxs]))
    h = w.k.finish_inductive_unchecked(tid, specs)
    for (cname, *_), cg in zip(ctors, h.ctor_global_ids): w._register(cname, cg)
    return w.const(name)


def build(checked):
    """the base library, excluded middle, propositional extensionality and the family R; when R is admitted, the proof of False.  Raises author.AuthorError at the declaration of R when checked is True."""
    w = lnl_base.build_base()
    PROP, TYPE, F, T = w.PROP, w.TYPE, w['False'], w['True']
    fa, Eq, Ex, app = w.forall_, w.Eq, w.Ex, w.app
    w.axiom('LEM', fa([PROP], lambda P: P | ~P))
    w.axiom('propext', fa([PROP, PROP, lambda a, b: w.Iff(a, b)], lambda a, b, h: Eq(a, b)))
    ctors = [('mkr', [PROP], lambda A: [w.arrow(w['R'](A), F)])]
    if checked: w.inductive('R', w.arrow(PROP, TYPE), ctors)
    else: declare_unchecked(w, 'R', w.arrow(PROP, TYPE), ctors)
    R = w['R']
    Efn = lambda x: w.arrow(R(x), F)            # E x := (R x -> False)
    prove = w.prove
    # an element of R x comes from some B with E B = x ... and conversely
    prove('R_inv', fa([PROP, lambda x: R(x)], lambda x, r: Ex(PROP, lambda B: Eq(Efn(B), x))), lambda p: (p.intro('x', 'r'), p.cases('r', names=['A']), p.exists('A'), p.refl()))
    prove('R_of_eq', fa([PROP, PROP, lambda B, x: Eq(Efn(B), x)], lambda B, x, h: R(x)), lambda p: (p.intro('B', 'x', 'h'), p.cases('h'), p.apply('mkr')))
    # K1: if x is a value of E, then E x is refuted;  K3: if x is not a value of E, then E x holds
    prove('K1', fa([PROP, PROP, lambda B, x: Eq(Efn(B), x), lambda B, x, h: Efn(x)], lambda B, x, h, e: F),
          lambda p: (p.intro('B', 'x', 'h', 'e'), p.apply('e'), p.exact(('R_of_eq', 'B', 'x', 'h'))))
    prove('K3', fa([PROP, lambda x: w.arrow(Ex(PROP, lambda B: Eq(Efn(B), x)), F)], lambda x, hn: Efn(x)), lambda p: (p.intro('x', 'hn', 'r'), p.apply('hn'), p.exact(('R_inv', 'x', 'r'))))
    # small facts about Prop (the last two use propext)
    tr_ = lambda a, b, c, h1, h2: app(w['trans'], PROP, a, b, c, h1, h2)
    prove('tf', fa([Eq(T, F)], lambda h: F), lambda p: (p.intro('h'), p.exact(lambda p: app(w['transport'], PROP, w.lam(PROP, lambda P: P), T, F, p.h('h'), w['tt']))))
    prove('H4', fa([PROP, lambda X: Eq(X, F), lambda X, a: Eq(X, T)], lambda X, a, b: F),
          lambda p: (p.intro('X', 'a', 'b'), p.apply('tf'), p.exact(lambda p: tr_(T, p.h('X'), F, app(w['symm'], PROP, p.h('X'), T, p.h('b')), p.h('a')))))
    prove('eq_true', fa([PROP, lambda P: P], lambda P, h: Eq(P, T)), lambda p: (p.intro('P', 'h'), p.apply('propext'), p.apply('conj'), p.intro('x'), p.apply('tt'), p.intro('x'), p.exact('h')))
    prove('eq_false', fa([PROP, lambda P: ~P], lambda P, n: Eq(P, F)), lambda p: (p.intro('P', 'n'), p.apply('propext'), p.apply('conj'), p.intro('x'), p.apply('n'), p.exact('x'), p.intro('x'), p.cases('x')))
    Efun = w.lam(PROP, lambda P: Efn(P))
    congr = lambda p, v, e: app(w['f_equal'], PROP, PROP, Efun, p.h('B'), v, p.h(e))

    def boom(p):
        # excluded middle and propext make E False and E True each equal to True or to False
        p.have('c0', ('LEM', Efn(F))); p.cases('c0', names=['h0', 'h0'])
        # E False holds, so E False = True: True is a value of E, so E True is refuted, so E True = False: False is a value of E, so E False is refuted
        p.have('q0', ('eq_true', Efn(F), 'h0')); p.have('n1', ('K1', F, T, 'q0')); p.have('q1', ('eq_false', Efn(T), 'n1')); p.have('n0', ('K1', T, F, 'q1')); p.exact(('n0', 'h0'))
        # E False fails, so E False = False
        p.have('q0', ('eq_false', Efn(F), 'h0')); p.have('c1', ('LEM', Efn(T))); p.cases('c1', names=['h1', 'h1'])
        # E True holds, so E True = True: True is a value of E, so E True is refuted
        p.have('q1', ('eq_true', Efn(T), 'h1')); p.have('n1', ('K1', T, T, 'q1')); p.exact(('n1', 'h1'))
        # E True fails too: both values of E are False, so True is not a value of E, so E True holds
        p.have('q1', ('eq_false', Efn(T), 'h1'))

        def not_in_image(p):
            p.intro('hex'); p.cases('hex', names=['B', 'hB'])
            p.have('cB', ('LEM', 'B')); p.cases('cB', names=['b', 'b'])
            p.have('e1', ('eq_true', 'B', 'b')); p.have('e2', lambda p: congr(p, T, 'e1')); p.have('e3', lambda p: tr_(Efn(p.h('B')), Efn(T), F, p.h('e2'), p.h('q1')))
            p.exact(lambda p: app(w['H4'], Efn(p.h('B')), p.h('e3'), p.h('hB')))
            p.have('e1', ('eq_false', 'B', 'b')); p.have('e2', lambda p: congr(p, F, 'e1')); p.have('e3', lambda p: tr_(Efn(p.h('B')), Efn(F), F, p.h('e2'), p.h('q0')))
            p.exact(lambda p: app(w['H4'], Efn(p.h('B')), p.h('e3'), p.h('hB')))
        p.have('ni', w.arrow(Ex(PROP, lambda B: Eq(Efn(B), T)), F), not_in_image)
        p.exact(lambda p: app(p.h('h1'), app(w['K3'], T, p.h('ni'))))

    prove('boom', F, boom)
    return w
