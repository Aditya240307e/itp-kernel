#!/usr/bin/env python3
"""S20 gate: the global environment stays WELL FOUNDED -- the reservation discipline.

A recursive function or an inductive type is declared in two steps: reserve_* gives it an id and a type, finish_* supplies the body / the constructors. Between the two the global exists
but means nothing. If anything could mention it in that window, two reserved globals could be defined in terms of each other, and each would still pass its own check:

    f : Nat -> False := fun x => g x          g : Nat -> False := fun x => f x                 =>  f 0 : False
    T1 with  c1 : T2 -> T1     T2 with  c2 : (T1 -> False) -> T2                               =>  a negative occurrence strict positivity never sees

Both were accepted by the checked path until this gate was written. The rule now (GlobalEnv::declare, recursive_function_problem, require_ctor_specs_complete, declare_checked, check_open):
a type or value may mention only FINISHED globals and, while it is being finished, the global itself. This gate checks

  A. what must still work: ordinary recursion, recursive types, a function calling an earlier one, and the sound replacement for mutual recursion (ONE function with an extra argument:
     binary addition with a carry flag, checked against Python integers);
  B. both exploits, in every order, and every other way a declaration could reach a reserved global (types, values, constructor fields and indices, axioms, strict checks);
  C. that a refused finish leaves everything usable, and that rewind refuses to cross a finished function body (its nodes would be reclaimed under a surviving global);
  D. that the INDEPENDENT checker finds the same problems by itself on hand-built environments (cycles, mentions of reserved globals, unguarded or unrecorded self-reference);
  E. a term with 2^60 root-to-leaf paths (the check must visit each shared node once) -- in a child process, so a hang is a failure with a name;
  F. the option prop_elim cannot be changed once anything is declared: a Prop inductive declared without the option records no elimination restriction, and with the option switched on
     later (definitional proof irrelevance) its large elimination proves False;
  G. a declared type, a recursive function's type, a statement and its hypotheses must be well-formed TYPES, not merely terms that evaluate to one.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))

FAIL = []; CHECKS = [0]
def expect(cond, msg):
    CHECKS[0] += 1
    if not cond: FAIL.append(msg)

def must_reject(fn, needle, label, cls=None):
    """fn() must raise an exception whose text contains `needle` (and, when `cls` is given, whose Python class has that name: ITPError = a TypeError of the kernel, RuntimeError = a protocol violation)"""
    CHECKS[0] += 1
    try: fn()
    except Exception as e:
        if needle not in str(e) or (cls is not None and type(e).__name__ != cls):
            FAIL.append('%s: rejected with the wrong error: %s: %s' % (label, type(e).__name__, str(e)[:170]))
        return
    FAIL.append('%s: ACCEPTED' % label)

def child_e():
    import prop_world as PW
    k, g = PW.build_prop_world()
    t = k.mk_const(g['Nat'])
    for _ in range(60): t = k.mk_app(t, t)                    # 2^60 paths, 61 distinct nodes
    k.declare_unchecked(t, None, 'big_dag_type')              # not well typed -- declare_unchecked does not look; the point is that every walk over it terminates
    k.declare_unchecked(k.mk_const(g['Nat']), t, 'big_dag_value')
    r = k.check_partial(t, k.mk_const(g['Nat']), [])         # an ill-typed term: rejected (cleanly), not a hang
    print('E OK')

def main():
    import harness
    from harness import build_new, l2h, H as KM
    from s3_tests import V, app
    import s7_param_tests as t7
    import prop_world as pw
    import indep_checker as ic
    import s8_indep_tests as t8
    T0, T1, SELF = ('type', 0), ('type', 1), ('self',)
    t0 = time.time()

    def world():
        k, g = pw.build_prop_world()
        return k, g, (lambda t, n0=0, lenient=False: build_new(k, l2h(t, n0, lenient=lenient)))
    def gmap(k): return {name: gid for gid, name, _, _ in k.list_globals()}
    def spec(k, name, fields, indices=()):
        return KM.CtorSpec(name, [build_new(k, l2h(a, j, lenient=True)) for j, a in enumerate(fields)], [build_new(k, l2h(r, len(fields), lenient=True)) for r in indices])

    # ------------------------------------------------------------------------------------------------ A. what must still work
    k, g, T = world(); NAT, S, Z, FALSE = ('const', g['Nat']), ('const', g['Succ']), ('const', g['Zero']), ('const', g['False'])
    fty = ('pi', NAT, NAT)
    double = k.reserve_recursive_function('double', T(fty)); DOUBLE = ('const', double)
    double_body = ('lam', NAT, ('match', V(0), ('lam', NAT, NAT), (Z, ('lam', NAT, app(S, app(S, app(DOUBLE, V(1))))))))
    k.finish_recursive_function_checked(double, g['Nat'], 0, T(double_body))
    expect(k.reserved_kind(double) == 0 and k.recursive_info(double) == (g['Nat'], 0), 'A1: a finished function is no longer reserved and records its recursion')
    k.declare_checked(T(fty), T(('lam', NAT, app(DOUBLE, app(DOUBLE, V(0))))), 'quad')                     # A2: uses the finished function
    t7.declare_family_new(k, 'Tree', T1, [('leaf', [], []), ('node', [SELF, SELF], [])])                  # A3: a recursive type
    g = gmap(k)
    expect(k.reserved_kind(g['Tree']) == 0 and k.is_inductive(g['Tree']), 'A3: a finished type is complete')
    ch, bad = t8.run_definitions(k, True)
    expect(not [b for b in bad if b[1] not in ('cong', 'le_trans', 'transport')], 'A: the independent checker rejects a valid world: %s' % [b for b in bad if b[1] not in ('cong', 'le_trans', 'transport')])

    # A4: binary addition with a carry flag -- one function, recursion on the second argument, the flag changes in the recursive call
    k, g, T = world()
    t7.declare_family_new(k, 'Bit', T1, [('b0', [], []), ('b1', [], [])])
    t7.declare_family_new(k, 'Pos', T1, [('xh', [], []), ('xo', [SELF], []), ('xi', [SELF], [])])
    g = gmap(k); BIT, POS = ('const', g['Bit']), ('const', g['Pos']); B0, B1, XH, XO, XI = (('const', g[n]) for n in ('b0', 'b1', 'xh', 'xo', 'xi'))
    mot, motb = ('lam', POS, POS), ('lam', BIT, POS)
    M = lambda scrut, motive, *branches: ('match', scrut, motive, tuple(branches))
    succ = k.reserve_recursive_function('Pos_succ', T(('pi', POS, POS))); SUCC = ('const', succ)
    k.finish_recursive_function_checked(succ, g['Pos'], 0, T(('lam', POS, M(V(0), mot, app(XO, XH), ('lam', POS, app(XI, V(1))), ('lam', POS, app(XO, app(SUCC, V(1))))))))
    addc = k.reserve_recursive_function('Pos_addc', T(('pi', BIT, ('pi', POS, ('pi', POS, POS))))); ADDC = ('const', addc)
    c = lambda flag, p, q: app(ADDC, flag, p, q)
    # levels: c=0, x=1, y=2; a field of x is level 3, a field of y inside it level 4 (inside the xh branch, level 3)
    body = ('lam', BIT, ('lam', POS, ('lam', POS, M(V(1), mot,
        # x = 1
        M(V(0), motb,
            M(V(2), mot, app(XO, XH), ('lam', POS, app(XI, V(3))), ('lam', POS, app(XO, app(SUCC, V(3))))),            # carry 0:  1+1, 1+2q, 1+2q+1
            M(V(2), mot, app(XI, XH), ('lam', POS, app(XO, app(SUCC, V(3)))), ('lam', POS, app(XI, app(SUCC, V(3)))))),   # carry 1
        # x = 2p
        ('lam', POS, M(V(0), motb,
            M(V(2), mot, app(XI, V(3)), ('lam', POS, app(XO, c(B0, V(3), V(4)))), ('lam', POS, app(XI, c(B0, V(3), V(4))))),
            M(V(2), mot, app(XO, app(SUCC, V(3))), ('lam', POS, app(XI, c(B0, V(3), V(4)))), ('lam', POS, app(XO, c(B1, V(3), V(4))))))),
        # x = 2p+1
        ('lam', POS, M(V(0), motb,
            M(V(2), mot, app(XO, app(SUCC, V(3))), ('lam', POS, app(XI, c(B0, V(3), V(4)))), ('lam', POS, app(XO, c(B1, V(3), V(4))))),
            M(V(2), mot, app(XI, app(SUCC, V(3))), ('lam', POS, app(XO, c(B1, V(3), V(4)))), ('lam', POS, app(XI, c(B1, V(3), V(4)))))))))))
    k.finish_recursive_function_checked(addc, g['Pos'], 1, T(body))
    expect(k.recursive_info(addc) == (g['Pos'], 1), 'A4: the carry-flag function records its decreasing argument')
    def pos(n):
        bits = bin(n)[3:]; t = XH
        for b in bits: t = app(XI if b == '1' else XO, t)
        return t
    def decode(s):
        toks = s.replace('(', ' ( ').replace(')', ' ) ').split(); i = [0]
        names = {g['xh']: 'xh', g['xo']: 'xo', g['xi']: 'xi'}
        def parse():
            tok = toks[i[0]]; i[0] += 1
            if tok == '(':
                f = parse(); a = parse(); assert toks[i[0]] == ')'; i[0] += 1
                return (f, a)
            return names[int(tok[1:])]
        def val(t):
            if t == 'xh': return 1
            return 2 * val(t[1]) + (1 if t[0] == 'xi' else 0)
        return val(parse())
    def evalpos(t): return decode(k.dump_raw(k.normalize_open(T(t), 0)))
    wrong = []
    for x in range(1, 40):
        for y in range(1, 40):
            for flag, f in ((0, B0), (1, B1)):
                if evalpos(app(ADDC, f, pos(x), pos(y))) != x + y + flag: wrong.append((x, y, flag))
    rng = random.Random(20)
    for _ in range(60):
        x, y, flag = rng.getrandbits(rng.randint(1, 70)) | 1, rng.getrandbits(rng.randint(1, 70)) | 1, rng.randint(0, 1)
        if evalpos(app(ADDC, B1 if flag else B0, pos(x), pos(y))) != x + y + flag: wrong.append((x, y, flag))
    expect(not wrong, 'A4: the carry-flag addition disagrees with Python integers on %d inputs, e.g. %s' % (len(wrong), wrong[:3]))
    ch, bad = t8.run_definitions(k, True)
    expect(not [b for b in bad if b[1] not in ('cong', 'le_trans', 'transport')], 'A4: the independent checker rejects the carry-flag world: %s' % [b for b in bad if b[1] not in ('cong', 'le_trans', 'transport')])

    # ------------------------------------------------------------------------------------------------ B. the exploits and every other way to reach a reserved global
    N_REJ = 'which is declared but not yet finished'
    for order in ((0, 1), (1, 0)):
        k, g, T = world(); NAT, Z, FALSE = ('const', g['Nat']), ('const', g['Zero']), ('const', g['False'])
        fty = ('pi', NAT, FALSE); ids = [k.reserve_recursive_function('f', T(fty)), k.reserve_recursive_function('g', T(fty))]
        F, G = ('const', ids[0]), ('const', ids[1]); bodies = [('lam', NAT, app(G, V(0))), ('lam', NAT, app(F, V(0)))]
        for i in order:
            must_reject(lambda i=i: k.finish_recursive_function_checked(ids[i], g['Nat'], 0, T(bodies[i])), N_REJ, 'B1 mutual functions, finishing %s first' % 'fg'[order[0]], 'ITPError')
            must_reject(lambda i=i: k.finish_recursive_function(ids[i], g['Nat'], 0, T(bodies[i])), N_REJ, 'B1 mutual functions (unchecked finish)', 'RuntimeError')
        expect(all(k.reserved_kind(i) == 2 for i in ids), 'B1: after the refusals both functions must still be reserved (and have no body)')
        expect(all(k.global_info(i)[2] is None for i in ids), 'B1: a refused finish left a body installed')
        must_reject(lambda: k.declare_checked(T(FALSE), T(app(F, Z)), 'absurd'), N_REJ, 'B3 declare_checked: a proof that uses a reserved function', 'ITPError')
        must_reject(lambda: k.declare_unchecked(T(FALSE), T(app(F, Z)), 'absurd'), N_REJ, 'B3 declare_unchecked: a proof that uses a reserved function', 'RuntimeError')
        must_reject(lambda: k.check_open(T(app(F, Z)), T(FALSE), []), N_REJ, 'B3 check_open: a proof that uses a reserved function', 'ITPError')
        must_reject(lambda: k.declare_checked(T(('pi', NAT, FALSE)), T(('lam', NAT, app(G, V(0)))), 'h'), N_REJ, 'B6 a definition via a reserved function', 'ITPError')

    # B2 / B9
    k, g, T = world(); NAT, S, Z, FALSE = ('const', g['Nat']), ('const', g['Succ']), ('const', g['Zero']), ('const', g['False'])
    fty = ('pi', NAT, NAT); f = k.reserve_recursive_function('f', T(fty)); h = k.reserve_recursive_function('h', T(fty)); F, Hh = ('const', f), ('const', h)
    must_reject(lambda: k.finish_recursive_function_checked(f, g['Nat'], 0, T(('lam', NAT, app(Hh, V(0))))), N_REJ, 'B2 a function whose body uses ANOTHER reserved function (not a loop, still mutual)', 'ITPError')
    own = ('lam', NAT, ('match', V(0), ('lam', NAT, NAT), (Z, ('lam', NAT, app(S, app(F, V(1)))))))
    k.finish_recursive_function_checked(f, g['Nat'], 0, T(own))                                              # B9: still possible after the refusal
    must_reject(lambda: k.finish_recursive_function_checked(f, g['Nat'], 0, T(own)), 'is not a reserved', 'B9 a function is finished once', 'ITPError')
    k.finish_recursive_function_checked(h, g['Nat'], 0, T(('lam', NAT, ('match', V(0), ('lam', NAT, NAT), (Z, ('lam', NAT, app(S, app(F, app(Hh, V(1))))))))))   # now h may use the FINISHED f
    expect(k.reserved_kind(f) == 0 and k.reserved_kind(h) == 0, 'B9: both functions are finished at the end')

    # B4: two types that need each other, in both orders -- and the refused one can still be finished properly
    for first in ('T1', 'T2'):
        k, g, T = world(); FALSE = ('const', g['False'])
        t2 = k.reserve_inductive_type('T2', T(T1)); t1 = k.reserve_inductive_type('T1', T(T1)); C1, C2 = ('const', t1), ('const', t2)
        specs = {'T1': (t1, lambda: spec(k, 'c1', [C2])), 'T2': (t2, lambda: spec(k, 'c2', [('pi', C1, FALSE)]))}
        gid, mk = specs[first]
        must_reject(lambda: k.finish_inductive(gid, [mk()]), N_REJ, 'B4 mutually inductive types, %s first (checked)' % first, 'RuntimeError')
        must_reject(lambda: k.finish_inductive_unchecked(gid, [mk()]), N_REJ, 'B4 mutually inductive types, %s first (unchecked)' % first, 'RuntimeError')
        expect(k.reserved_kind(gid) == 1 and not k.is_inductive(gid) and not any(d[1] in ('c1', 'c2') for d in k.list_globals()), 'B4: a refused finish_inductive left constructors or state behind')
        k.finish_inductive(gid, [spec(k, 'ok_ctor', [])])                                                    # no mention of the other type: fine
        expect(k.is_inductive(gid) and k.reserved_kind(gid) == 0, 'B4: the type could not be finished properly after a refusal')
    # the exploit script itself, end to end: it must stop at the first finish
    k, g, T = world(); FALSE = ('const', g['False'])
    t2 = k.reserve_inductive_type('T2', T(T1)); t1 = k.reserve_inductive_type('T1', T(T1)); C1, C2 = ('const', t1), ('const', t2)
    must_reject(lambda: k.finish_inductive(t1, [spec(k, 'c1', [C2])]), N_REJ, 'B4 the Russell-style exploit')

    # B7: types and axioms
    k, g, T = world(); NAT, FALSE, Z = ('const', g['Nat']), ('const', g['False']), ('const', g['Zero'])
    rf = k.reserve_recursive_function('rf', T(('pi', NAT, NAT))); RF = ('const', rf)
    rt = k.reserve_inductive_type('RT', T(T1)); RT = ('const', rt)
    must_reject(lambda: k.reserve_recursive_function('f2', T(('pi', RT, NAT))), N_REJ, 'B7 a function type that mentions a reserved type', 'RuntimeError')
    must_reject(lambda: k.reserve_inductive_type('Ix0', T(('pi', app(('const', g['Eq']), NAT, app(RF, Z), Z), T1))), N_REJ, 'B7 a sort that mentions a reserved function', 'RuntimeError')
    must_reject(lambda: k.declare_unchecked(T(('pi', RT, FALSE)), None, 'ax'), N_REJ, 'B7 an axiom about a reserved type', 'RuntimeError')
    must_reject(lambda: k.declare_checked(T(('pi', RT, FALSE)), T(('lam', RT, V(0))), 'badty'), N_REJ, 'B7 declare_checked: the TYPE mentions a reserved type', 'ITPError')
    must_reject(lambda: k.check_open(T(('lam', NAT, V(0))), T(('pi', RT, FALSE)), []), N_REJ, 'B7 check_open: the EXPECTED TYPE mentions a reserved type', 'ITPError')
    must_reject(lambda: k.check_open(T(('const', g['tt'])), T(('const', g['True'])), [T(RT)]), N_REJ, 'B7 check_open: a HYPOTHESIS type mentions a reserved type', 'ITPError')
    must_reject(lambda: k.declare_unchecked(T(('pi', NAT, NAT)), T(('lam', NAT, app(RF, V(0)))), 'viaRF'), N_REJ, 'B7 an unchecked definition through a reserved function', 'RuntimeError')
    # B8: constructor fields and indices
    ix = k.reserve_inductive_type('Ix', T(('pi', NAT, T1)))
    must_reject(lambda: k.finish_inductive(ix, [spec(k, 'mk', [], [app(RF, Z)])]), N_REJ, 'B8 a constructor INDEX that mentions a reserved function (checked)', 'RuntimeError')
    must_reject(lambda: k.finish_inductive_unchecked(ix, [spec(k, 'mk', [], [app(RF, Z)])]), N_REJ, 'B8 a constructor INDEX that mentions a reserved function (unchecked)', 'RuntimeError')
    must_reject(lambda: k.finish_inductive(ix, [spec(k, 'ok', [], [Z]), spec(k, 'mk2', [app(('const', g['Eq']), NAT, app(RF, Z), Z)], [Z])]), N_REJ, 'B8 a constructor FIELD that mentions a reserved function', 'RuntimeError')
    must_reject(lambda: k.finish_inductive_unchecked(ix, [spec(k, 'ok', [], [Z]), spec(k, 'mk2', [app(('const', g['Eq']), NAT, app(RF, Z), Z)], [Z])]), N_REJ, 'B8 a constructor FIELD that mentions a reserved function (unchecked)', 'RuntimeError')
    expect(not any(d[1] in ('mk', 'ok', 'mk2') for d in k.list_globals()), 'B8: the FIRST constructor of a refused family must not be declared (a half-finished type)')
    k.finish_inductive(ix, [spec(k, 'ok', [], [Z])]); expect(k.is_inductive(ix), 'B8: the family can still be finished properly')

    # ------------------------------------------------------------------------------------------------ C. checkpoint / rewind
    k, g, T = world(); NAT, S, Z = ('const', g['Nat']), ('const', g['Succ']), ('const', g['Zero'])
    fty = ('pi', NAT, NAT); f = k.reserve_recursive_function('f', T(fty)); F = ('const', f)
    cp = k.checkpoint()
    k.finish_recursive_function_checked(f, g['Nat'], 0, T(('lam', NAT, ('match', V(0), ('lam', NAT, NAT), (Z, ('lam', NAT, app(S, app(F, V(1)))))))))
    must_reject(lambda: k.rewind(cp), 'recursive function was finished since this checkpoint', 'C1 rewind across a finished function body', 'RuntimeError')
    k, g, T = world(); NAT, S, Z, TRUE = ('const', g['Nat']), ('const', g['Succ']), ('const', g['Zero']), ('const', g['True'])
    f = k.reserve_recursive_function('f', T(fty)); F = ('const', f)
    cp = k.checkpoint()
    must_reject(lambda: k.finish_recursive_function_checked(f, g['Nat'], 0, T(('lam', NAT, ('match', V(0), ('lam', NAT, NAT), (TRUE, ('lam', NAT, app(S, app(F, V(1))))))))), '', 'C2 an ill-typed body is refused')
    try: k.rewind(cp); ok = True
    except Exception as e: ok = False
    expect(ok, 'C2: after a REFUSED (rolled back) finish a rewind must be allowed')
    k.finish_recursive_function_checked(f, g['Nat'], 0, T(('lam', NAT, ('match', V(0), ('lam', NAT, NAT), (Z, ('lam', NAT, app(S, app(F, V(1)))))))))
    expect(k.reserved_kind(f) == 0, 'C2: and the function can still be finished correctly afterwards')

    k, g, T = world(); NAT, S, Z = ('const', g['Nat']), ('const', g['Succ']), ('const', g['Zero'])
    f = k.reserve_recursive_function('f', T(fty)); F = ('const', f)
    k.finish_recursive_function_checked(f, g['Nat'], 0, T(('lam', NAT, ('match', V(0), ('lam', NAT, NAT), (Z, ('lam', NAT, app(S, app(F, V(1)))))))))
    cp = k.checkpoint(); ps = k.start_proof(T(('pi', NAT, NAT))); ps.intro('x')
    try: k.rewind(cp); ok = True
    except Exception as e: ok = False
    expect(ok, 'C3: a checkpoint taken AFTER a function was finished must be rewindable (nothing has changed since)')

    # C4: a refused finish leaves no trace in the evaluator -- while the body is checked it is installed provisionally, and evaluating the function caches its value
    k, g, T = world(); NAT, Z, S = ('const', g['Nat']), ('const', g['Zero']), ('const', g['Succ'])
    f = k.reserve_recursive_function('f', T(('pi', NAT, NAT))); F = k.mk_const(f)
    nat, z, eq, refl, tt = (k.mk_const(g[n]) for n in ('Nat', 'Zero', 'Eq', 'refl', 'tt'))
    eq_ty = k.mk_app(k.mk_app(k.mk_app(eq, nat), k.mk_app(F, k.mk_bvar(0))), z)            # Eq Nat (f m) Zero      (m = bvar 0; the check must evaluate f to compare it)
    s_branch = k.mk_lambda(nat, k.mk_let(eq_ty, k.mk_app(k.mk_app(refl, nat), z), tt))      # fun m => let p : Eq Nat (f m) Zero := refl Nat Zero in tt     -- ill-typed twice over
    bad = k.mk_lambda(nat, k.mk_match(k.mk_bvar(0), k.mk_lambda(nat, nat), [z, s_branch]))
    must_reject(lambda: k.finish_recursive_function_checked(f, g['Nat'], 0, bad), '', 'C4 an ill-typed body is refused', 'ITPError')
    nf = k.dump_raw(k.normalize_open(k.mk_app(F, k.mk_app(k.mk_const(g['Succ']), z)), 0))
    expect(nf == '(C%d (C%d C%d))' % (f, g['Succ'], g['Zero']), 'C4: after a refused finish the function still unfolds (a stale cached value): f (S Z) normalises to %s' % nf)

    # ------------------------------------------------------------------------------------------------ D. the independent checker on hand-built environments
    def base_env():
        e = ic.Env()
        e.add(0, 'Nat', ('type', 1), None, None, None, [1, 2], 0, True)
        e.add(1, 'Zero', ('const', 0), None, 0, 0, [], 0, None)
        e.add(2, 'Succ', ('pi', ('const', 0), ('const', 0)), None, 0, 1, [], 0, None)
        e.add(3, 'False', ('type', 0), None, None, None, [], 0, True)
        return e
    nat_to_false = ('pi', ('const', 0), ('const', 3))
    def problems(env): return ic.Checker(env, params=True, fuel=200000).check_definitions()
    e = base_env(); e.add(4, 'f', nat_to_false, ('lam', ('const', 0), ('app', ('const', 5), ('bvar', 0))), None, None, [], 0, None); e.add(5, 'g', nat_to_false, ('lam', ('const', 0), ('app', ('const', 4), ('bvar', 0))), None, None, [], 0, None)
    expect(any('cyclic definitions' in b[2] for b in problems(e)), 'D1: the independent checker accepted the f/g loop: %s' % problems(e))
    # the loop hidden inside a match branch (a walk that skips branches would miss it)
    e = base_env(); nat_motive = ('lam', ('const', 0), ('const', 3))
    e.add(4, 'f', nat_to_false, ('lam', ('const', 0), ('match', ('bvar', 0), nat_motive, (('app', ('const', 5), ('const', 1)), ('lam', ('const', 0), ('app', ('const', 5), ('bvar', 0)))))), None, None, [], 0, None)
    e.add(5, 'g', nat_to_false, ('lam', ('const', 0), ('app', ('const', 4), ('bvar', 0))), None, None, [], 0, None)
    expect(any('cyclic definitions' in b[2] for b in problems(e)), 'D1b: the independent checker missed a cycle that runs through match branches: %s' % problems(e))
    e = base_env(); e.add(4, 'RT', ('type', 1), None, None, None, [], 0, True, None, 1); e.add(5, 'ax', ('pi', ('const', 4), ('const', 3)), None, None, None, [], 0, None)
    expect(any('not yet finished' in b[2] for b in problems(e)), 'D2b: the independent checker accepted a TYPE that mentions a reserved type: %s' % problems(e))
    e = base_env(); e.add(4, 'dangling', ('const', 3), ('app', ('const', 99), ('const', 1)), None, None, [], 0, None)
    expect(any('unknown global' in b[2] for b in problems(e)), 'D7: the independent checker accepted a mention of a global that does not exist: %s' % problems(e))
    e = base_env(); e.add(4, 'f', nat_to_false, None, None, None, [], 0, None, None, 2); e.add(5, 'absurd', ('const', 3), ('app', ('const', 4), ('const', 1)), None, None, [], 0, None)
    expect(any('not yet finished' in b[2] for b in problems(e)), 'D2: the independent checker accepted a mention of a reserved function: %s' % problems(e))
    e = base_env(); e.add(4, 'k', ('pi', ('const', 0), ('const', 0)), ('lam', ('const', 0), ('app', ('const', 4), ('bvar', 0))), None, None, [], 0, None)
    expect(any('refers to itself' in b[2] for b in problems(e)), 'D3: the independent checker accepted a self-reference of an unrecorded global: %s' % problems(e))
    e = base_env(); e.add(4, 'loop', ('pi', ('const', 0), ('const', 0)), ('lam', ('const', 0), ('app', ('const', 4), ('bvar', 0))), None, None, [], 0, None, (0, 0))
    expect(any('guard' in b[2] for b in problems(e)), 'D4: the independent checker accepted an unguarded recursive global: %s' % problems(e))
    double_h = ('lam', ('const', 0), ('match', ('bvar', 0), ('lam', ('const', 0), ('const', 0)), (('const', 1), ('lam', ('const', 0), ('app', ('const', 2), ('app', ('const', 2), ('app', ('const', 4), ('bvar', 0))))))))
    e = base_env(); e.add(4, 'double', ('pi', ('const', 0), ('const', 0)), double_h, None, None, [], 0, None, (0, 0))
    expect(problems(e) == [], 'D5: the independent checker rejected a valid recursive global: %s' % problems(e))
    e = base_env(); e.add(4, 'double', ('pi', ('const', 0), ('const', 0)), double_h, None, None, [], 0, None, (3, 0))
    expect(any('not an inductive family' in b[2] or 'not of the recorded family' in b[2] for b in problems(e)), 'D6: the checker accepted a recursion recorded over the wrong family: %s' % problems(e))

    # ------------------------------------------------------------------------------------------------ F. the option prop_elim is fixed once anything is declared
    # The attack: declare a Prop inductive (Or) with prop_elim OFF -- no elimination restriction is recorded for it -- switch the option ON, and eliminate it into Bool. With definitional proof
    # irrelevance (inl tt == inr tt) that is a proof of False; the other direction (declare under prop_elim, switch it off) lets a Type0 family with a large field be eliminated freely.
    def attack_world():
        kk = KM.Kernel()
        for name, sort, ctors in (('Bool', T1, [('b0', [], []), ('b1', [], [])]), ('True', T0, [('tt', [], [])]),
                                  ('Or', ('pi', T0, ('pi', T0, T0)), [('inl', [T0, T0, ('var', 0)], [('var', 0), ('var', 1)]), ('inr', [T0, T0, ('var', 1)], [('var', 0), ('var', 1)])])):
            t7.declare_family_new(kk, name, sort, ctors)
        return kk
    kk = attack_world()
    must_reject(lambda: kk.set_option('prop_elim', True), 'cannot be changed once globals exist', 'F1 switching prop_elim ON after declarations', 'RuntimeError')
    expect(kk.get_option('prop_elim') is False, 'F1: a refused set_option changed the option anyway')
    kk.set_option('prop_elim', False)                                                            # the value it already has: fine
    kk2 = KM.Kernel(); kk2.set_option('prop_elim', True); kk2.set_option('parameters', True); t7.declare_family_new(kk2, 'True', T0, [('tt', [], [])])
    must_reject(lambda: kk2.set_option('prop_elim', False), 'cannot be changed once globals exist', 'F2 switching prop_elim OFF after declarations', 'RuntimeError')
    kk2.set_option('prop_elim', True)
    kk2.set_option('parameters', False); kk2.set_option('parameters', True)                      # the parameter analysis is a pure function of the declared family: still free to change
    kk3 = KM.Kernel(); kk3.set_option('prop_elim', True); kk3.set_option('prop_elim', False); kk3.set_option('prop_elim', True)       # while the environment is empty any change is fine
    expect(kk3.get_option('prop_elim') is True, 'F3: prop_elim could not be changed on an empty environment')

    # ------------------------------------------------------------------------------------------------ G. declared types, statements and contexts must be TYPES
    # `check` takes its expected type on trust. An ill-typed term that merely evaluates to the right thing -- (fun x : Nat => Nat) Type0 -- used to be stored as a global's type and "proved".
    k, g, T = world(); NAT, Z, S, FALSE = ('const', g['Nat']), ('const', g['Zero']), ('const', g['Succ']), ('const', g['False'])
    junk = app(('lam', NAT, NAT), ('type', 0))                  # evaluates to Nat, but applies a function on Nat to Type0
    must_reject(lambda: k.declare_checked(T(junk), T(Z), 'g1'), '', 'G1 declare_checked with an ill-typed (but evaluating) declared type', 'ITPError')
    expect(not any(d[1] == 'g1' for d in k.list_globals()), 'G1: the ill-typed declaration was installed anyway')
    must_reject(lambda: k.declare_checked(T(Z), T(Z), 'g2'), '', 'G2 declare_checked with a non-type as the declared type', 'ITPError')
    junk_fn = app(('lam', NAT, ('pi', NAT, NAT)), ('type', 0))
    f = k.reserve_recursive_function('f', T(junk_fn)); F = ('const', f)
    own = ('lam', NAT, ('match', V(0), ('lam', NAT, NAT), (Z, ('lam', NAT, app(S, app(F, V(1)))))))
    must_reject(lambda: k.finish_recursive_function_checked(f, g['Nat'], 0, T(own)), '', 'G3 a recursive function reserved with an ill-typed type', 'ITPError')
    expect(k.reserved_kind(f) == 2 and k.global_info(f)[2] is None, 'G3: the refused function was finished or given a body anyway')
    ok, msg = k.check_open(T(Z), T(junk), [])
    expect(not ok, 'G4: check_open "proved" an ill-typed statement: %s' % msg)
    ok, msg = k.check_open(T(Z), T(NAT), [T(junk)])
    expect(not ok and 'context' in msg, 'G5: check_open accepted an ill-typed hypothesis type: %s' % msg)
    ok, msg = k.check_open(T(Z), T(NAT), [T(NAT)])
    expect(ok, 'G6: a well-formed statement and context must still check: %s' % msg)
    ok, msg = k.check_partial(T(Z), T(NAT), [T(NAT)]); expect(ok, 'G6: check_partial on a well-formed statement: %s' % msg)

    # ------------------------------------------------------------------------------------------------ E. shared structure
    r = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', 'e'], capture_output=True, text=True, timeout=40, cwd=HERE)
    expect(r.returncode == 0 and 'E OK' in r.stdout, 'E: a term with 2^60 paths was not handled in time/at all: exit %s %s' % (r.returncode, (r.stdout + r.stderr)[-200:]))

    print("   %d checks (%.1f s)" % (CHECKS[0], time.time() - t0))
    for f in FAIL: print("   FAIL " + f)
    print("S20 RESERVATION GATE: " + ("PASS" if not FAIL else "FAIL"))
    sys.exit(1 if FAIL else 0)

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--child': child_e()
    else: main()
