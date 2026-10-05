"""binary_nat.py -- a binary numeral representation, additive to the existing (unary) Nat.

Motivation: normalize(add k k) on the existing unary Nat is quadratic in k (confirmed directly,
bench_unary_scaling.cpp, back in the throughput benchmarking session) -- a REPRESENTATION problem
(a unary numeral of value k is itself a term of size O(k), so any structural recursion over it needs
O(k) steps, each touching an O(k)-sized term), not a fixable property of how `add` was written: it
would be equally quadratic recursing on either argument. No amount of kernel-side optimization to
`add` itself changes that a unary Zero/Succ representation of k is O(k) bits.

Design: Coq's own standard-library split (BinNums.positive / BinNums.N), chosen deliberately over
inventing a new scheme -- it is the most-exercised, best-understood binary numeral design available,
which matters here because a numeral system used for arithmetic is exactly the kind of thing where a
subtle bug would be easy to miss and costly to trust. `Pos` (least-significant-bit first) represents
strictly positive integers with O(log v) sized terms; `N` adds a zero case on top.

  Pos ::= XH            (value 1)
        | XO Pos         (given p : value v, XO p : value 2v)
        | XI Pos         (given p : value v, XI p : value 2v+1)
  N   ::= N0 | Npos Pos

Everything here is built through kernel primitives ALREADY exposed to Python (reserve_inductive_type
/finish_inductive, reserve_recursive_function/finish_recursive_function_checked, declare_checked) --
no kernel or binding changes were made or are needed. In particular, mutual recursion (Pos.add and
Pos.add_carry each call the other) needed no dedicated mutual-recursion primitive: verified directly
that finish_recursive_function_checked's own single-function structural-decrease check treats a call
to a DIFFERENT, already-reserved (but not yet finished) global as an ordinary opaque call -- it only
constrains a function's calls to ITSELF -- so reserving both functions' global ids up front, building
each body free to reference the other, and finishing each independently (each function's own
self-calls checked against its own decrease, the other function trusted by its own, separate,
already-passed check) works and is validated by the kernel's own independent checker on each
definition, exactly as any other definition in this codebase is trusted.

Pos.add/add_carry's branch table is Coq's own (re-derived independently here, by hand, from the
value semantics above, then cross-checked against Coq's actual definition -- they agree). Coq's own
argument for why this is correct is a standard, decades-old piece of library code; that pedigree does
not substitute for this project's own verification discipline, so every function below is checked
twice over: once by the kernel's own type/termination checker (finish_recursive_function_checked
raises if it disagrees), and again by exhaustive testing against Python's own `+` for every pair of
values up to N_EXHAUSTIVE (see test_binary_nat.py) -- neither check alone would be enough on its own.
One real bug WAS caught this way during development: a single branch (add_carry's own XO/XO case) had
been copy-pasted from a neighboring case and called the wrong function with the wrong constructor;
every OTHER small-value pair passed by coincidence (they simply never reached that specific branch),
and only the exhaustive sweep (specifically x=y=5) surfaced it. This is exactly the kind of narrow,
single-branch error that "looks done" after a few spot checks and is precisely why exhaustive testing
over dozens of thousands of pairs, not a handful of examples, is the bar here.

Performance (bench_binary_nat_final.py): normalize(N.add(k, k)) scales quadratically in BIT LENGTH
(each doubling of bits costs ~4x, not ~2x -- Pos.add_carry's own recursive structure does a little
more than a single pass per bit), NOT in VALUE -- e.g. 64 bits (value ~1.8e19) takes ~1.6ms, 256 bits
~30ms, 2048 bits ~1.9s. The old unary representation was quadratic in VALUE itself (k=1280 took
561ms; k=1280 has only 11 bits). Since bits = log2(value), quadratic-in-bits is exponentially better
than quadratic-in-value for anything of real size: values this project's own curricula, or any
plausible competition-math problem, would ever need (well under 64 bits) now cost low milliseconds
instead of being computationally unreachable.

KNOWN LIMIT, confirmed NOT specific to this module: normalize() crashes (a C++ segfault, presumably
a recursive evaluator's own call-stack exhaustion) somewhere between 3040 and 3072 BITS of recursion
depth for N.add(k, k) here -- but the SAME crash, at a similar depth, was independently reproduced
using nothing but the pre-existing unary Nat and its own `add` (a long chain, no binary_nat code
involved at all): confirming this is a pre-existing, general kernel recursion-depth ceiling, not a
bug introduced by or unique to this module. It sits far above anything this module's own use case
needs (competition-math-scale numbers, comfortably under 64-128 bits), but is worth knowing about,
and worth fixing at the kernel level (an iterative/explicit-stack evaluator, or a larger stack) as a
separate, future, genuinely kernel-touching task -- deliberately not attempted here.
"""
from __future__ import annotations

import itp_env


def setup_binary_nat(k, refs: dict) -> dict:
    """Adds Pos, N, and their operations to an already-built kernel (see curriculum.build_base_kernel
    for `k`/`refs`). Purely additive: does not touch Nat, add, or anything else already in `refs`, so
    every existing curriculum goal is completely unaffected whether or not this is called. Returns a
    dict of the new NodeIds/global ids, in the same style as build_base_kernel's own `refs`."""
    type0 = refs['type0']

    # -------------------------------------------------------------------- Pos
    pos_id = k.reserve_inductive_type('Pos', type0)
    pos_ty = k.mk_const(pos_id)
    pos_h = k.finish_inductive(pos_id, [
        itp_env.CtorSpec('XH', []),
        itp_env.CtorSpec('XO', [pos_ty]),
        itp_env.CtorSpec('XI', [pos_ty]),
    ])
    xh_c, xo_c, xi_c = (k.mk_const(g) for g in pos_h.ctor_global_ids)

    # ---- Pos.succ : Pos -> Pos --------------------------------------------------------------
    # succ(XH) = XO XH ; succ(XO p) = XI p ; succ(XI p) = XO (succ p)
    # Single self-recursion, decreasing on the sole argument (level 0) -- the same shape as the
    # existing unary `add`, just over Pos instead of Nat.
    succ_fn_ty = k.mk_pi(pos_ty, pos_ty)
    succ_gid = k.reserve_recursive_function('Pos_succ', succ_fn_ty)
    succ_c = k.mk_const(succ_gid)
    succ_motive = k.mk_lambda(pos_ty, pos_ty)
    succ_xh_branch = k.mk_app(xo_c, xh_c)
    succ_xo_branch = k.mk_lambda(pos_ty, k.mk_app(xi_c, k.mk_var(1)))       # var1 = p (the XO's own arg)
    succ_xi_branch = k.mk_lambda(pos_ty, k.mk_app(xo_c, k.mk_app(succ_c, k.mk_var(1))))
    succ_body = k.mk_lambda(pos_ty, k.mk_match(k.mk_var(0), succ_motive, [succ_xh_branch, succ_xo_branch, succ_xi_branch]))
    k.finish_recursive_function_checked(succ_gid, pos_id, 0, succ_body)

    # ---- Pos.add / Pos.add_carry : Pos -> Pos -> Pos (mutually recursive) -------------------
    # Both match on the FIRST argument (level 0, matching this codebase's existing `add`
    # convention), with a NESTED match on the second (level 1, still directly addressable inside
    # every branch of the first match: levels are stable under further nested binders by
    # construction -- introducing a branch binder for the first match's own constructor argument
    # never changes what level the second argument was bound at). See this module's own
    # docstring for the branch-by-branch derivation.
    add_fn_ty = k.mk_pi(pos_ty, k.mk_pi(pos_ty, pos_ty))
    add_gid = k.reserve_recursive_function('Pos_add', add_fn_ty)
    add_carry_gid = k.reserve_recursive_function('Pos_add_carry', add_fn_ty)
    add_c = k.mk_const(add_gid)
    add_carry_c = k.mk_const(add_carry_gid)

    def _mk_body(self_c, other_c):
        """self_c is THIS function's own const (used for its own decreasing recursive calls on the
        destructed first argument); other_c is the mutually-recursive partner's const."""
        # y-level (the second, outer-bound argument) is ALWAYS var(1), everywhere below, regardless
        # of how many further branch binders (for x's or y's own destructed predecessor) are
        # introduced -- that is exactly what using levels (not indices) buys here.
        y_motive = k.mk_lambda(pos_ty, pos_ty)

        # --- x = XH: no new binder; y still var(1) ---
        # y=XH -> XO XH (add) / XI XH (add_carry)
        # y=XO q [q@1, since XH introduces no binder, q is the 2nd overall bound var: x@0,y@1,q@... ]
        # NOTE: inside the x=XH branch there is no x-binder, so the CURRENT depth is still 2 (x@0,y@1);
        # y's own OWN match branch for XO/XI introduces q at level 2.
        xh_y_xh = k.mk_app(xo_c, xh_c) if self_c is add_c else k.mk_app(xi_c, xh_c)
        xh_y_xo = k.mk_lambda(pos_ty, k.mk_app(xi_c, k.mk_var(2)) if self_c is add_c
                             else k.mk_app(xo_c, k.mk_app(succ_c, k.mk_var(2))))
        xh_y_xi = k.mk_lambda(pos_ty, k.mk_app(xo_c, k.mk_app(succ_c, k.mk_var(2))) if self_c is add_c
                             else k.mk_app(xi_c, k.mk_app(succ_c, k.mk_var(2))))
        x_xh_branch = k.mk_match(k.mk_var(1), y_motive, [xh_y_xh, xh_y_xo, xh_y_xi])

        # --- x = XO p [p@2]: y still var(1) ---
        xo_y_xh = k.mk_app(xi_c, k.mk_var(2)) if self_c is add_c else k.mk_app(xo_c, k.mk_app(succ_c, k.mk_var(2)))
        # y = XO q [q@3]: p@2, q@3
        xo_y_xo = k.mk_lambda(pos_ty,
            k.mk_app(xo_c, k.mk_app(k.mk_app(add_c, k.mk_var(2)), k.mk_var(3))) if self_c is add_c
            else k.mk_app(xi_c, k.mk_app(k.mk_app(add_c, k.mk_var(2)), k.mk_var(3))))
        xo_y_xi = k.mk_lambda(pos_ty,
            k.mk_app(xi_c, k.mk_app(k.mk_app(add_c, k.mk_var(2)), k.mk_var(3))) if self_c is add_c
            else k.mk_app(xo_c, k.mk_app(k.mk_app(add_carry_c, k.mk_var(2)), k.mk_var(3))))
        x_xo_branch = k.mk_lambda(pos_ty, k.mk_match(k.mk_var(1), y_motive, [xo_y_xh, xo_y_xo, xo_y_xi]))

        # --- x = XI p [p@2]: y still var(1) ---
        xi_y_xh = k.mk_app(xo_c, k.mk_app(succ_c, k.mk_var(2))) if self_c is add_c else k.mk_app(xi_c, k.mk_app(succ_c, k.mk_var(2)))
        xi_y_xo = k.mk_lambda(pos_ty,
            k.mk_app(xi_c, k.mk_app(k.mk_app(add_c, k.mk_var(2)), k.mk_var(3))) if self_c is add_c
            else k.mk_app(xo_c, k.mk_app(k.mk_app(add_carry_c, k.mk_var(2)), k.mk_var(3))))
        xi_y_xi = k.mk_lambda(pos_ty,
            k.mk_app(xo_c, k.mk_app(k.mk_app(add_carry_c, k.mk_var(2)), k.mk_var(3))) if self_c is add_c
            else k.mk_app(xi_c, k.mk_app(k.mk_app(add_carry_c, k.mk_var(2)), k.mk_var(3))))
        x_xi_branch = k.mk_lambda(pos_ty, k.mk_match(k.mk_var(1), y_motive, [xi_y_xh, xi_y_xo, xi_y_xi]))

        x_motive = k.mk_lambda(pos_ty, pos_ty)
        return k.mk_lambda(pos_ty, k.mk_lambda(pos_ty,
            k.mk_match(k.mk_var(0), x_motive, [x_xh_branch, x_xo_branch, x_xi_branch])))

    add_body = _mk_body(add_c, add_carry_c)
    add_carry_body = _mk_body(add_carry_c, add_c)
    # Each finished independently: add's own checker validates only add's OWN self-calls (all on
    # the destructed x, at level 2, strictly smaller than x itself) -- calls to add_carry are, from
    # add's own checker's point of view, ordinary calls to an already-reserved (typed) function,
    # exactly as calling `succ` is. Then add_carry is finished the same way, its own self-calls
    # checked, calls to (now-finished) add trusted by add's own already-passed check. Verified
    # directly this works, with a minimal is_even/is_odd pair, before committing to this design.
    k.finish_recursive_function_checked(add_gid, pos_id, 0, add_body)
    k.finish_recursive_function_checked(add_carry_gid, pos_id, 0, add_carry_body)

    # -------------------------------------------------------------------- N
    n_id = k.reserve_inductive_type('N', type0)
    n_ty = k.mk_const(n_id)
    n_h = k.finish_inductive(n_id, [itp_env.CtorSpec('N0', []), itp_env.CtorSpec('Npos', [pos_ty])])
    n0_c, npos_c = (k.mk_const(g) for g in n_h.ctor_global_ids)

    # N.add : N -> N -> N. NOT self-referential (it only ever calls Pos.add, never itself), so
    # declare_checked suffices -- no recursive-function machinery needed, same as this codebase's
    # own `cong` (a match-containing, non-recursive definition via plain declare_checked).
    n_add_ty = k.mk_pi(n_ty, k.mk_pi(n_ty, n_ty))
    n_motive = k.mk_lambda(n_ty, n_ty)
    # x = N0: y=N0 -> N0 ; y=Npos q -> Npos q   [q@2, since x's N0 branch adds no binder: x@0,y@1,q@2]
    x_n0_y_n0 = n0_c
    x_n0_y_npos = k.mk_lambda(pos_ty, k.mk_app(npos_c, k.mk_var(2)))
    x_n0_branch = k.mk_match(k.mk_var(1), n_motive, [x_n0_y_n0, x_n0_y_npos])
    # x = Npos p [p@2]: y=N0 -> Npos p ; y=Npos q [q@3] -> Npos (Pos.add p q)
    x_npos_y_n0 = k.mk_app(npos_c, k.mk_var(2))
    x_npos_y_npos = k.mk_lambda(pos_ty, k.mk_app(npos_c, k.mk_app(k.mk_app(add_c, k.mk_var(2)), k.mk_var(3))))
    x_npos_branch = k.mk_lambda(pos_ty, k.mk_match(k.mk_var(1), n_motive, [x_npos_y_n0, x_npos_y_npos]))
    n_add_body = k.mk_lambda(n_ty, k.mk_lambda(n_ty, k.mk_match(k.mk_var(0), n_motive, [x_n0_branch, x_npos_branch])))
    n_add_gid = k.declare_checked(n_add_ty, n_add_body, 'N_add')
    n_add_c = k.mk_const(n_add_gid)

    return dict(
        pos_id=pos_id, pos_ty=pos_ty, xh_c=xh_c, xo_c=xo_c, xi_c=xi_c,
        pos_succ_id=succ_gid, pos_succ_c=succ_c,
        pos_add_id=add_gid, pos_add_c=add_c, pos_add_carry_id=add_carry_gid, pos_add_carry_c=add_carry_c,
        n_id=n_id, n_ty=n_ty, n0_c=n0_c, npos_c=npos_c,
        n_add_id=n_add_gid, n_add_c=n_add_c,
    )


# --------------------------------------------------------------------------------------------
# Python <-> term helpers. Building a term for a Python int is O(log v) (just walks its bits);
# it never goes through unary Nat and never normalizes anything to construct a numeral -- exactly
# the point, since even CONSTRUCTING a unary numeral of value v is already an O(v) term.
# --------------------------------------------------------------------------------------------

def int_to_pos(k, br: dict, v: int):
    """v >= 1. Builds bit-by-bit from the LEAST significant bit outward (matching Pos's own
    least-significant-bit-first structure): the innermost constructor is XH (the leading 1 bit,
    i.e. the MOST significant), each XO/XI wrapping it going back out lower and lower."""
    if v < 1:
        raise ValueError(f"int_to_pos: v must be >= 1, got {v}")
    bits = []
    while v > 1:
        bits.append(v & 1)
        v >>= 1
    # v is now 1 (the leading bit) -- build from the leading bit inward, applying remaining bits
    # from MOST to LEAST significant (i.e. reversed, since `bits` was collected least-significant-first).
    term = br['xh_c']
    for b in reversed(bits):
        term = k.mk_app(br['xi_c'] if b else br['xo_c'], term)
    return term


def int_to_n(k, br: dict, v: int):
    if v == 0:
        return br['n0_c']
    return k.mk_app(br['npos_c'], int_to_pos(k, br, v))


def pos_to_int(k, term) -> int:
    """Decodes an already-NORMALIZED Pos term (no Match/Meta/free Var -- a closed value built purely
    from XH/XO/XI) back to a Python int, for testing. Uses print_node's own text form rather than
    walking NodeIds directly, since this module has no dependency on the kernel's own C++ constructor
    introspection beyond what print_node already exposes -- fine for a test/debug helper, not on any
    hot path."""
    s = k.print_node(term).strip()
    return _parse_pos(s)


def _parse_pos(s: str) -> int:
    s = s.strip()
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1].strip()
    if s == 'XH':
        return 1
    if s.startswith('XO '):
        return 2 * _parse_pos(s[3:])
    if s.startswith('XI '):
        return 2 * _parse_pos(s[3:]) + 1
    raise ValueError(f"pos_to_int: cannot parse {s!r} as a normalized Pos term")


def n_to_int(k, term) -> int:
    s = k.print_node(term).strip()
    if s == 'N0':
        return 0
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1].strip()
    if s.startswith('Npos '):
        return _parse_pos(s[5:])
    raise ValueError(f"n_to_int: cannot parse {s!r} as a normalized N term")
