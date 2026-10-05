"""
curriculum.py -- the expert-iteration training loop.

Round structure: build a FRESH kernel from the fixed base library each
round (not carried over across rounds -- deliberate, so "the model got
better" and "the library accumulated more lemmas" stay separable
signals, per this session's own established measurement-validity
principle). Within one round, curriculum goals are attempted IN ORDER;
a solved goal with its own lemma_name gets declare_checked'd
immediately, extending the library for the REST of that same round.
Held-out evaluation always uses a fresh, base-only kernel, isolating
genuine policy improvement from library growth.
"""

from __future__ import annotations
import gc
from dataclasses import dataclass
from typing import Callable, Optional
import itp_env
import hypergraph as hg
from model import PolicyValueModel
from trainable_policy import TrainablePolicy
import training


def eq_app(k, refs, a, x, y):
    return k.mk_app(k.mk_app(k.mk_app(refs['eq_ty'], a), x), y)


def setup_nat_order(k, refs: dict) -> dict:
    """Adds Le/Lt on Nat to an already-built kernel, purely additively (nothing about Nat, add, Eq,
    or any existing goal changes). Mirrors Coq's own standard definition exactly, for the same reason
    binary_nat.py mirrored Coq's positive/N: it is the best-understood, most-exercised choice
    available, which matters most exactly where a subtle definitional error would be easy to miss.

        Le n : Nat -> Prop :=
          | le_n :  Le n n
          | le_S :  (m:Nat) -> Le n m -> Le n (Succ m)
        Lt n m := Le (Succ n) m
    """
    nat_ty = refs['nat_ty']; succ_c = refs['succ_c']; prop0 = refs['prop0']

    le_sort = k.mk_pi(nat_ty, k.mk_pi(nat_ty, prop0))
    le_id = k.reserve_inductive_type('Le', le_sort)
    le_ty = k.mk_const(le_id)
    le_h = k.finish_inductive(le_id, [
        itp_env.CtorSpec('le_n', [nat_ty], [k.mk_var(0), k.mk_var(0)]),
        itp_env.CtorSpec('le_S', [nat_ty, nat_ty, k.mk_app(k.mk_app(le_ty, k.mk_var(0)), k.mk_var(1))],
                          [k.mk_var(0), k.mk_app(succ_c, k.mk_var(1))]),
    ])
    le_n_c = k.mk_const(le_h.ctor_global_ids[0])
    le_s_c = k.mk_const(le_h.ctor_global_ids[1])

    lt_fn_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty, prop0))
    lt_body = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty,
        k.mk_app(k.mk_app(le_ty, k.mk_app(succ_c, k.mk_var(0))), k.mk_var(1))))
    lt_gid = k.declare_checked(lt_fn_ty, lt_body, 'lt')

    return dict(le_id=le_id, le_ty=le_ty, le_n_c=le_n_c, le_s_c=le_s_c,
                lt_id=lt_gid, lt_c=k.mk_const(lt_gid))


def le_app(k, refs, n, m):
    return k.mk_app(k.mk_app(refs['le_ty'], n), m)


def lt_app(k, refs, n, m):
    return k.mk_app(k.mk_app(refs['lt_c'], n), m)


def setup_le_trans(k, refs: dict) -> dict:
    """Adds le_trans : (a b c:Nat) -> Le a b -> Le b c -> Le a c to a kernel whose refs already
    include setup_nat_order's (le_ty, le_s_c).

    Hand-scripted via tactics -- intro x5, induct on h2, then ONE apply per branch -- rather than
    found by search, and independently verified (whole_term_check) before being declared:

        intro a b c h1 h2;  induct h2        -- h1 : Le a b is GENERALIZED (reverted): it depends on
                                             -- b, one of h2's indices, so the IH is strengthened
                                             -- to  IH : (h1 : Le a b) -> Le a c'
        base (le_n) :  exact h1'             -- h1' : Le a n', the branch-local copy of h1
        step (le_S) :  apply le_S a c' (IH h1)

    This is the textbook proof (induction on the SECOND hypothesis, generalizing the first). It was
    for a long time a documented, open project limitation: induct() could not generalize an earlier
    hypothesis that depends on an index of the scrutinee. The full chain of causes (a missing revert
    mechanism, a too-coarse sweep, unforced result indices, an unwrapped zero-field branch, a
    pre-applied IH, a reverted position not specialized inside the generic Fix body, and a
    sequential-substitution collision in solved-meta beta-reduction) is written up in
    INDUCT_REVERT_SWEEP_NOTES.md; test_indexed_induction.py pins each one down (mutation-tested).
    """
    nat_ty = refs['nat_ty']; le_s_c = refs['le_s_c']
    V = k.mk_var
    ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_pi(le_app(k, refs, V(0), V(1)),
         k.mk_pi(le_app(k, refs, V(1), V(2)), le_app(k, refs, V(0), V(2)))))))

    def goal_index(ps, meta):
        return next(i for i in range(ps.num_goals()) if ps.goal_meta_id(i) == meta)

    ps = k.start_proof(ty)
    for name in ('a', 'b', 'c', 'h1', 'h2'):
        ps.intro(name, target_meta_id=ps.goal_meta_id(0))
    ps.induct(4, target_meta_id=ps.goal_meta_id(0))       # h2 : Le b c is hypothesis index 4
    # induct() emits one goal per constructor, in declaration order: le_n (base), then le_S (step)
    base_meta, step_meta = ps.goal_meta_id(0), ps.goal_meta_id(1)

    # base goal context: [a, b, c, h1, h2, n', h1']      -- h1' (the branch-local h1) is last
    nb = ps.goal_num_hyps(goal_index(ps, base_meta))
    ps.apply(V(nb - 1), target_meta_id=base_meta)

    # step goal context: [a, b, c, h1, h2, n', c', p, IH, h1']
    ns = ps.goal_num_hyps(goal_index(ps, step_meta))
    ih, c_prime = ns - 2, ns - 4
    ps.apply(k.mk_app(k.mk_app(k.mk_app(le_s_c, V(0)), V(c_prime)), k.mk_app(V(ih), V(3))),
             target_meta_id=step_meta)

    assert ps.is_complete(), "setup_le_trans: scripted proof did not close every goal"
    term = ps.extract()
    k.whole_term_check(term, ty)                           # independent, from-scratch re-verification
    gid = k.declare_checked(ty, term, 'le_trans')
    return dict(le_trans_id=gid, le_trans_c=k.mk_const(gid))


def setup_add_comm(k, refs: dict) -> dict:
    """Adds add_comm : (n m:Nat) -> Eq Nat (add n m) (add m n) to a kernel whose refs already
    include add_zero_right_c and add_succ_right_c (i.e. call this AFTER running CURRICULUM through
    run_round/evaluate or an equivalent loop -- those two lemmas are genuine prerequisites, not
    merely convenient ones).

    Hand-built directly via explicit term application (intro, induct, then one fully-constructed
    term per branch), NOT via plain tactic search -- kept this way deliberately, not out of
    continued necessity: this was originally a real, documented, open project finding (search found
    a tactic-level SOLVED path for add_comm once add_zero_right/add_succ_right were available, but
    that path's own extracted term failed independent verification with "different variable/
    constant heads") whose root cause is now understood and fixed, in a later session. That symptom
    traces to cases()/induct()'s own generalize() step: when either is called while the goal's own
    target still has a remaining Pi (i.e. before every quantified variable is intro()'d -- exactly
    what a UniformPolicy search exploring freely is otherwise free to do, since enumerate_legal_actions
    offers INDUCT on any inductive-typed hypothesis already present regardless of what Pi's remain),
    a pre-existing partial fix for this (shift_levels_from, from a still-earlier session) under-
    shifts by exactly one level for any branch that also introduces an IH (the recursive case),
    letting a later-introduced hypothesis's own listed index and genuine level silently diverge.
    cases()/induct() now reject this outright (is_pi_headed's own guard in itp_kernel.hpp), turning
    it into an immediate, clean tactic failure instead of a proof that reports solved but fails
    whole_term_check. Confirmed directly, this session: a full, independently-verified add_comm
    proof via rewrite (not cases/induct/apply's own congruence-lemma style below) now exists
    end-to-end, using exactly the add_succ_left_unfold lemma this function also declares, below.
    This function's own hand-built proof is kept as-is regardless -- not because anything about it
    was ever broken, but because it predates rewrite and remains a perfectly good, independent proof
    of the same lemma, and because mul_comm's own hand-built proof (setup_mul) is kept the same way
    for the same reason.

        base case   (n=Zero):  symm(add_zero_right(m)) : Eq Nat m (add m Zero)
                                 -- matches "Eq Nat (add Zero m) (add m Zero)" definitionally
        step case (n=Succ n'): trans(cong(IH), symm(add_succ_right(m, n')))
                                 : Eq Nat (Succ (add n' m)) (add m (Succ n'))
    """
    nat_ty = refs['nat_ty']; add_c = refs['add_c']; succ_c = refs['succ_c']
    symm_c = refs['symm_c']; trans_c = refs['trans_c']; cong_c = refs['cong_c']
    add_zero_right_c = refs['add_zero_right_c']; add_succ_right_c = refs['add_succ_right_c']

    def add_app(a, b):
        return k.mk_app(k.mk_app(add_c, a), b)

    add_comm_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty,
        eq_app(k, refs, nat_ty, add_app(k.mk_var(0), k.mk_var(1)), add_app(k.mk_var(1), k.mk_var(0)))))

    ps = k.start_proof(add_comm_ty)
    ps.intro('n', target_meta_id=ps.goal_meta_id(0))
    ps.intro('m', target_meta_id=ps.goal_meta_id(0))
    ps.induct(0, target_meta_id=ps.goal_meta_id(0))

    m_var = k.mk_var(1)
    base_term = k.mk_app(k.mk_app(k.mk_app(k.mk_app(symm_c, nat_ty),
        add_app(m_var, refs['zero'])), m_var), k.mk_app(add_zero_right_c, m_var))
    ps.apply(base_term, target_meta_id=ps.goal_meta_id(0))

    n_prime = k.mk_var(2)
    ih = k.mk_var(3)
    cong_step = k.mk_app(k.mk_app(k.mk_app(cong_c, add_app(n_prime, m_var)), add_app(m_var, n_prime)), ih)
    asr_step = k.mk_app(k.mk_app(add_succ_right_c, m_var), n_prime)
    symm_asr = k.mk_app(k.mk_app(k.mk_app(k.mk_app(symm_c, nat_ty),
        add_app(m_var, k.mk_app(succ_c, n_prime))), k.mk_app(succ_c, add_app(m_var, n_prime))), asr_step)
    x_val = k.mk_app(succ_c, add_app(n_prime, m_var))
    y_val = k.mk_app(succ_c, add_app(m_var, n_prime))
    z_val = add_app(m_var, k.mk_app(succ_c, n_prime))
    step_term = k.mk_app(k.mk_app(k.mk_app(k.mk_app(k.mk_app(
        trans_c, nat_ty), x_val), y_val), z_val), cong_step)
    step_term = k.mk_app(step_term, symm_asr)
    ps.apply(step_term, target_meta_id=ps.goal_meta_id(0))

    assert ps.is_complete()
    add_comm_term = ps.extract()
    k.whole_term_check(add_comm_term, add_comm_ty)  # independent re-verification
    add_comm_gid = k.declare_checked(add_comm_ty, add_comm_term, 'add_comm')

    # add_succ_left_unfold : (n m:Nat) -> Eq Nat (add (Succ n) m) (Succ (add n m)) -- a direct
    # restatement of add's own computation rule, built purely via refl, exactly mirroring
    # mul_succ_left_unfold's own construction (see setup_mul). Together with add_succ_right
    # (already above) and add_zero_right (from CURRICULUM), this is what let add_comm itself be
    # re-derived via rewrite + the intro-both-then-induct discipline cases()/induct() now enforce
    # (see is_pi_headed's own guard in itp_kernel.hpp) -- independently confirmed, this session,
    # to verify end-to-end. This does NOT retroactively confirm the ORIGINAL, separately-discovered
    # automatic-extraction bug this function's own docstring describes is fixed by THIS proof
    # existing (it still isn't fixed by this hand-built path, same as before) -- but the guard
    # above is the directly-confirmed root-cause fix for the exact symptom ("different
    # variable/constant heads") that bug's own docstring describes, and a full, independently-
    # verified rewrite-based reproof of add_comm, impossible before today's fixes, now exists
    # end-to-end as a result.
    def build_add_succ_left_unfold_ty():
        n_t, m_t = k.mk_var(0), k.mk_var(1)
        lhs = add_app(k.mk_app(succ_c, n_t), m_t)
        rhs = k.mk_app(succ_c, add_app(n_t, m_t))
        return k.mk_pi(nat_ty, k.mk_pi(nat_ty, eq_app(k, refs, nat_ty, lhs, rhs)))

    asl_ty = build_add_succ_left_unfold_ty()
    asl_refl_lhs = add_app(k.mk_app(succ_c, k.mk_var(0)), k.mk_var(1))
    asl_term = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty, k.mk_app(k.mk_app(refs['refl_c'], nat_ty), asl_refl_lhs)))
    k.whole_term_check(asl_term, asl_ty)  # independent re-verification
    asl_gid = k.declare_checked(asl_ty, asl_term, 'add_succ_left_unfold')

    return dict(add_comm_id=add_comm_gid, add_comm_c=k.mk_const(add_comm_gid),
                add_succ_left_unfold_id=asl_gid, add_succ_left_unfold_c=k.mk_const(asl_gid))


def build_rewrite_registry(k, refs: dict) -> list:
    """Builds hypergraph.py's own REWRITE_CANDIDATE_FNS list: one small closure per "rewrite-
    eligible" library lemma (a Pi-chain of plain Nat arguments ending in an Eq conclusion), each
    trying every combination of in-scope Nat-typed hypotheses for that lemma's own arguments and
    constructing (eq_term, x, y) directly the same, safe way this project's own hand-built proofs
    already do -- never via infer()'s own eager-unfolding reconstruction (see rewrite_eq's own,
    extensive comment on exactly why that matters). REQUIRES refs to already include mul_c,
    mul_zero_right_c, mul_succ_right_c, mul_succ_left_unfold_c, add_succ_right_c,
    add_succ_left_unfold_c (i.e. CURRICULUM + setup_add_comm + setup_mul must already have been
    run -- setup_add_comm provides add_succ_left_unfold_c, setup_mul provides the rest).
    """
    nat_ty = refs['nat_ty']; succ_c = refs['succ_c']; add_c = refs['add_c']; mul_c = refs['mul_c']

    def add_app(a, b):
        return k.mk_app(k.mk_app(add_c, a), b)

    def mul_app(a, b):
        return k.mk_app(k.mk_app(mul_c, a), b)

    def is_nat(k_, ty):
        return ty == nat_ty

    def nat_hyp_levels(hyp_types):
        return [h for h, ty in enumerate(hyp_types) if is_nat(k, ty)]

    # mul_succ_right(n, m) : Eq Nat (mul n (Succ m)) (add (mul n m) n) -- arity 2
    def mul_succ_right_candidates(kernel, ps, idx, hyp_types, levels):
        out = []
        ns = nat_hyp_levels(hyp_types)
        for n_h in ns:
            for m_h in ns:
                n_t, m_t = kernel.mk_var(n_h), kernel.mk_var(m_h)
                eq_term = kernel.mk_app(kernel.mk_app(refs['mul_succ_right_c'], n_t), m_t)
                x = mul_app(n_t, kernel.mk_app(succ_c, m_t))
                y = add_app(mul_app(n_t, m_t), n_t)
                out.append((eq_term, x, y))
        return out

    # mul_zero_right(n) : Eq Nat (mul n Zero) Zero -- arity 1
    def mul_zero_right_candidates(kernel, ps, idx, hyp_types, levels):
        out = []
        for n_h in nat_hyp_levels(hyp_types):
            n_t = kernel.mk_var(n_h)
            eq_term = kernel.mk_app(refs['mul_zero_right_c'], n_t)
            x = mul_app(n_t, refs['zero'])
            y = refs['zero']
            out.append((eq_term, x, y))
        return out

    # add_succ_right(n, m) : Eq Nat (add n (Succ m)) (Succ (add n m)) -- arity 2
    def add_succ_right_candidates(kernel, ps, idx, hyp_types, levels):
        out = []
        ns = nat_hyp_levels(hyp_types)
        for n_h in ns:
            for m_h in ns:
                n_t, m_t = kernel.mk_var(n_h), kernel.mk_var(m_h)
                eq_term = kernel.mk_app(kernel.mk_app(refs['add_succ_right_c'], n_t), m_t)
                x = add_app(n_t, kernel.mk_app(succ_c, m_t))
                y = kernel.mk_app(succ_c, add_app(n_t, m_t))
                out.append((eq_term, x, y))
        return out

    # add_zero_right(n) : Eq Nat (add n Zero) n -- arity 1
    def add_zero_right_candidates(kernel, ps, idx, hyp_types, levels):
        out = []
        for n_h in nat_hyp_levels(hyp_types):
            n_t = kernel.mk_var(n_h)
            eq_term = kernel.mk_app(refs['add_zero_right_c'], n_t)
            x = add_app(n_t, refs['zero'])
            y = n_t
            out.append((eq_term, x, y))
        return out

    # mul_succ_left_unfold(n, m) : Eq Nat (mul (Succ n) m) (add (mul n m) m) -- arity 2
    def mul_succ_left_unfold_candidates(kernel, ps, idx, hyp_types, levels):
        out = []
        ns = nat_hyp_levels(hyp_types)
        for n_h in ns:
            for m_h in ns:
                n_t, m_t = kernel.mk_var(n_h), kernel.mk_var(m_h)
                eq_term = kernel.mk_app(kernel.mk_app(refs['mul_succ_left_unfold_c'], n_t), m_t)
                x = mul_app(kernel.mk_app(succ_c, n_t), m_t)
                y = add_app(mul_app(n_t, m_t), m_t)
                out.append((eq_term, x, y))
        return out

    # add_succ_left_unfold(n, m) : Eq Nat (add (Succ n) m) (Succ (add n m)) -- arity 2
    def add_succ_left_unfold_candidates(kernel, ps, idx, hyp_types, levels):
        out = []
        ns = nat_hyp_levels(hyp_types)
        for n_h in ns:
            for m_h in ns:
                n_t, m_t = kernel.mk_var(n_h), kernel.mk_var(m_h)
                eq_term = kernel.mk_app(kernel.mk_app(refs['add_succ_left_unfold_c'], n_t), m_t)
                x = add_app(kernel.mk_app(succ_c, n_t), m_t)
                y = kernel.mk_app(succ_c, add_app(n_t, m_t))
                out.append((eq_term, x, y))
        return out

    return [mul_succ_right_candidates, mul_zero_right_candidates,
            add_succ_right_candidates, add_zero_right_candidates,
            mul_succ_left_unfold_candidates, add_succ_left_unfold_candidates]


def setup_transport(k, refs: dict) -> dict:
    """Adds transport : (A:Type) -> (P:A->Type) -> (x y:A) -> Eq A x y -> P x -> P y (the standard
    Eq.subst principle) as a permanent library fact. Hand-built directly via mk_match, mirroring
    cong's own construction exactly (same 4-binder motive shape: A, x', y', proof -- A declared but
    not referenced, x'/y' typed with k.mk_var(0) directly). This is the foundation the rewrite
    tactic (itp_kernel.hpp's rewrite_eq, exposed as ProofStateHandle.rewrite) is built on: given
    h:Eq A x y and a proof of P(y), transport(A,P,y,x,symm(h),proof) : P(x) -- recovering the
    ORIGINAL goal from a proof of the REWRITTEN one.
    """
    type0 = refs['type0']; refl_c = refs['refl_c']

    def eq2(a, x, y):
        return k.mk_app(k.mk_app(k.mk_app(refs['eq_ty'], a), x), y)

    transport_ty = k.mk_pi(type0, k.mk_pi(k.mk_pi(k.mk_var(0), type0), k.mk_pi(k.mk_var(0), k.mk_pi(k.mk_var(0),
        k.mk_pi(eq2(k.mk_var(0), k.mk_var(2), k.mk_var(3)),
                k.mk_pi(k.mk_app(k.mk_var(1), k.mk_var(2)), k.mk_app(k.mk_var(1), k.mk_var(3))))))))
    motive = k.mk_lambda(type0, k.mk_lambda(k.mk_var(0), k.mk_lambda(k.mk_var(0),
        k.mk_lambda(eq2(k.mk_var(0), k.mk_var(6), k.mk_var(7)),
                    k.mk_pi(k.mk_app(k.mk_var(1), k.mk_var(6)), k.mk_app(k.mk_var(1), k.mk_var(7)))))))
    refl_branch = k.mk_lambda(type0, k.mk_lambda(k.mk_var(0),
        k.mk_lambda(k.mk_app(k.mk_var(1), k.mk_var(6)), k.mk_var(7))))
    transport_body = k.mk_lambda(type0, k.mk_lambda(k.mk_pi(k.mk_var(0), type0), k.mk_lambda(k.mk_var(0), k.mk_lambda(k.mk_var(0),
        k.mk_lambda(eq2(k.mk_var(0), k.mk_var(2), k.mk_var(3)), k.mk_match(k.mk_var(4), motive, [refl_branch]))))))
    k.whole_term_check(transport_body, transport_ty)
    transport_gid = k.declare_checked(transport_ty, transport_body, 'transport')
    return dict(transport_id=transport_gid, transport_c=k.mk_const(transport_gid))


def setup_mul(k, refs: dict) -> dict:
    """Adds mul : Nat -> Nat -> Nat, and its basic algebra (mul_zero_right, mul_succ_right,
    mul_comm, plus the supporting congruence/associativity helpers cong_add_zero, add_assoc,
    cong_add_left, cong_add_right), to an already-built kernel. mul itself mirrors add's own
    construction exactly (same shape, same recursion argument): recurses on the FIRST argument
    (decreasing_arg=0, matching add_gid's own convention in build_base_kernel), NOT the second --

        mul Zero y := Zero
        mul (Succ x') y := add (mul x' y) y

    REQUIRES refs['add_comm_c'] to already be present (i.e. call setup_add_comm(k, refs) first,
    which itself requires add_zero_right_c/add_succ_right_c -- i.e. CURRICULUM to have already been
    run): mul_succ_right's own proof genuinely needs add_comm, not merely add_zero_right/
    add_succ_right. This is a real prerequisite chain, not an arbitrary convenience.
    """
    nat_ty = refs['nat_ty']; add_c = refs['add_c']; succ_c = refs['succ_c']

    def mul_app(a, b):
        return k.mk_app(k.mk_app(mul_c, a), b)

    mul_fn_type = k.mk_pi(nat_ty, k.mk_pi(nat_ty, nat_ty))
    mul_gid = k.reserve_recursive_function('mul', mul_fn_type)
    mul_c = k.mk_const(mul_gid)
    mul_motive = k.mk_lambda(nat_ty, nat_ty)
    mul_zero_branch = refs['zero']
    # x=0 (Succ's own field x', level 2), y=1 (ambient) -- add(mul(x', y), y)
    mul_succ_branch = k.mk_lambda(nat_ty,
        k.mk_app(k.mk_app(add_c, k.mk_app(k.mk_app(mul_c, k.mk_var(2)), k.mk_var(1))), k.mk_var(1)))
    mul_match = k.mk_match(k.mk_var(0), mul_motive, [mul_zero_branch, mul_succ_branch])
    mul_body = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty, mul_match))
    k.finish_recursive_function_checked(mul_gid, refs['nat_id'], 0, mul_body)

    # cong_add_zero : (n m:Nat) -> Eq Nat n m -> Eq Nat (add n Zero) (add m Zero) -- a genuine,
    # reusable congruence lemma (add respects equality in its first argument), hand-built directly
    # via mk_match, mirroring cong's own construction exactly (down to typing the motive's x'/y'
    # binders with nat_ty directly rather than by referencing the motive's own generic A binder --
    # confirmed, not assumed, that departing from this exact shape makes the motive rejected as
    # "not well-typed"). Needed because mul_zero_right's own step case needs to rewrite through an
    # IH naming a different, compound (mul-headed) expression -- exactly the kind of fact a future
    # rewrite/subst tactic would generalize, built by hand here for the same reason cong itself was:
    # confirmed, not assumed, that plain UniformPolicy search does not find mul_zero_right within a
    # modest budget (matching this project's own earlier add_comm precedent).
    type0 = refs['type0']

    def eq2(a, x, y):
        return k.mk_app(k.mk_app(k.mk_app(refs['eq_ty'], a), x), y)

    def add_zero_app(t):
        return k.mk_app(k.mk_app(add_c, t), refs['zero'])

    cong_add_zero_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_pi(eq2(nat_ty, k.mk_var(0), k.mk_var(1)),
        eq2(nat_ty, add_zero_app(k.mk_var(0)), add_zero_app(k.mk_var(1))))))
    caz_motive = k.mk_lambda(type0, k.mk_lambda(nat_ty, k.mk_lambda(nat_ty,
        k.mk_lambda(eq2(nat_ty, k.mk_var(4), k.mk_var(5)),
                    eq2(nat_ty, add_zero_app(k.mk_var(4)), add_zero_app(k.mk_var(5)))))))
    caz_refl_branch = k.mk_lambda(type0, k.mk_lambda(nat_ty,
        k.mk_app(k.mk_app(refs['refl_c'], nat_ty), add_zero_app(k.mk_var(4)))))
    cong_add_zero_body = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty, k.mk_lambda(eq2(nat_ty, k.mk_var(0), k.mk_var(1)),
        k.mk_match(k.mk_var(2), caz_motive, [caz_refl_branch]))))
    cong_add_zero_gid = k.declare_checked(cong_add_zero_ty, cong_add_zero_body, 'cong_add_zero')
    cong_add_zero_c = k.mk_const(cong_add_zero_gid)

    # mul_zero_right : (n:Nat) -> Eq Nat (mul n Zero) Zero -- hand-built via explicit tactic calls
    # (intro, induct, apply), using cong_add_zero for the step case; NOT via mk_match/Fix directly,
    # since the ordinary tactic layer already handles plain Nat induction (no indexed-relation
    # cross-dependency issue here, unlike le_trans) -- only the final rewriting step needed help.
    def build_mul_zero_right_ty():
        return k.mk_pi(nat_ty, eq2(nat_ty, k.mk_app(k.mk_app(mul_c, k.mk_var(0)), refs['zero']), refs['zero']))

    mzr_ty = build_mul_zero_right_ty()
    mzr_ps = k.start_proof(mzr_ty)
    mzr_ps.intro('n', target_meta_id=mzr_ps.goal_meta_id(0))
    mzr_ps.induct(0, target_meta_id=mzr_ps.goal_meta_id(0))
    mzr_ps.apply(k.mk_app(k.mk_app(refs['refl_c'], nat_ty), refs['zero']), target_meta_id=mzr_ps.goal_meta_id(0))
    mzr_mul_v1_zero = k.mk_app(k.mk_app(mul_c, k.mk_var(1)), refs['zero'])
    mzr_candidate = k.mk_app(k.mk_app(k.mk_app(cong_add_zero_c, mzr_mul_v1_zero), refs['zero']), k.mk_var(2))
    mzr_ps.apply(mzr_candidate, target_meta_id=mzr_ps.goal_meta_id(0))
    assert mzr_ps.is_complete()
    mzr_term = mzr_ps.extract()
    k.whole_term_check(mzr_term, mzr_ty)  # independent re-verification, same discipline as everywhere else
    mul_zero_right_gid = k.declare_checked(mzr_ty, mzr_term, 'mul_zero_right')
    mul_zero_right_c = k.mk_const(mul_zero_right_gid)

    # mul_succ_left_unfold : (n m:Nat) -> Eq Nat (mul (Succ n) m) (add (mul n m) m) -- a direct
    # restatement of mul's OWN computation rule (mul (Succ x') y := add (mul x' y) y, see this
    # function's own docstring), built purely via refl relying on definitional equality -- NOT a
    # new, independent fact. Exists as its own permanent library lemma (rather than reconstructed
    # ad hoc each time) because the rewrite tactic's own exact-NodeId matching is syntactic, not
    # up-to-computation: a goal containing the un-reduced "mul (Succ n) m" needs an explicit
    # rewrite step to expose "add (mul n m) m" as a genuine, matchable subterm before any further
    # rewrite (e.g. through an IH naming "mul n m") can find its own target. Same role for mul's
    # OWN recursion argument that mul_succ_zero_unfold (an early, test-script-only, Zero-second-
    # argument special case of exactly this) played during this project's first rewrite_eq trial.
    def build_mul_succ_left_unfold_ty():
        n_t, m_t = k.mk_var(0), k.mk_var(1)
        lhs = mul_app(k.mk_app(succ_c, n_t), m_t)
        rhs = k.mk_app(k.mk_app(add_c, mul_app(n_t, m_t)), m_t)
        return k.mk_pi(nat_ty, k.mk_pi(nat_ty, eq2(nat_ty, lhs, rhs)))

    msl_ty = build_mul_succ_left_unfold_ty()
    msl_refl_lhs = mul_app(k.mk_app(succ_c, k.mk_var(0)), k.mk_var(1))
    msl_term = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty, k.mk_app(k.mk_app(refs['refl_c'], nat_ty), msl_refl_lhs)))
    k.whole_term_check(msl_term, msl_ty)  # independent re-verification: confirms the computation rule holds exactly as stated
    mul_succ_left_unfold_gid = k.declare_checked(msl_ty, msl_term, 'mul_succ_left_unfold')
    mul_succ_left_unfold_c = k.mk_const(mul_succ_left_unfold_gid)

    # add_assoc, cong_add_left, cong_add_right, mul_succ_right, mul_comm: all hand-built for the
    # same reason mul_zero_right was (confirmed, not assumed, that plain search does not bootstrap
    # them within a safe budget) -- this is the project's own "critical test" of whether curriculum-
    # grading alone gets past comm-style stalls without a rewrite tactic: confirmed it does NOT, by
    # itself, make these search-findable, but confirmed the underlying LEMMAS are fully provable
    # with the existing 4 tactics via careful, direct term construction. The repeated pattern here
    # (build a small congruence helper, chain it with symm/trans) is exactly what a future rewrite
    # tactic would need to automate -- this hand-built path is evidence for its design, not a
    # substitute for it.
    def symm(a, x, y, h):
        return k.mk_app(k.mk_app(k.mk_app(k.mk_app(refs['symm_c'], a), x), y), h)

    def trans2(a, x, y, z, h1, h2):
        return k.mk_app(k.mk_app(k.mk_app(k.mk_app(k.mk_app(k.mk_app(refs['trans_c'], a), x), y), z), h1), h2)

    def congf(x, y, h):
        return k.mk_app(k.mk_app(k.mk_app(refs['cong_c'], x), y), h)

    # add_assoc : (a b c:Nat) -> Eq Nat (add (add a b) c) (add a (add b c))
    add_assoc_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_pi(nat_ty,
        eq2(nat_ty,
            k.mk_app(k.mk_app(add_c, k.mk_app(k.mk_app(add_c, k.mk_var(0)), k.mk_var(1))), k.mk_var(2)),
            k.mk_app(k.mk_app(add_c, k.mk_var(0)), k.mk_app(k.mk_app(add_c, k.mk_var(1)), k.mk_var(2)))))))
    aa_ps = k.start_proof(add_assoc_ty)
    for nm in ('a', 'b', 'c'):
        aa_ps.intro(nm, target_meta_id=aa_ps.goal_meta_id(0))
    aa_ps.induct(0, target_meta_id=aa_ps.goal_meta_id(0))
    aa_ps.apply(k.mk_app(k.mk_app(refs['refl_c'], nat_ty), k.mk_app(k.mk_app(add_c, k.mk_var(1)), k.mk_var(2))),
                target_meta_id=aa_ps.goal_meta_id(0))
    aa_lhs = k.mk_app(k.mk_app(add_c, k.mk_app(k.mk_app(add_c, k.mk_var(3)), k.mk_var(1))), k.mk_var(2))
    aa_rhs = k.mk_app(k.mk_app(add_c, k.mk_var(3)), k.mk_app(k.mk_app(add_c, k.mk_var(1)), k.mk_var(2)))
    aa_ps.apply(congf(aa_lhs, aa_rhs, k.mk_var(4)), target_meta_id=aa_ps.goal_meta_id(0))
    assert aa_ps.is_complete()
    add_assoc_term = aa_ps.extract()
    k.whole_term_check(add_assoc_term, add_assoc_ty)
    add_assoc_gid = k.declare_checked(add_assoc_ty, add_assoc_term, 'add_assoc')
    add_assoc_c = k.mk_const(add_assoc_gid)

    # cong_add_left : (a x y:Nat) -> Eq Nat x y -> Eq Nat (add a x) (add a y)
    def add_app(a2, b2):
        return k.mk_app(k.mk_app(add_c, a2), b2)

    cal_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_pi(eq2(nat_ty, k.mk_var(1), k.mk_var(2)),
        eq2(nat_ty, add_app(k.mk_var(0), k.mk_var(1)), add_app(k.mk_var(0), k.mk_var(2)))))))
    cal_motive = k.mk_lambda(type0, k.mk_lambda(nat_ty, k.mk_lambda(nat_ty,
        k.mk_lambda(eq2(nat_ty, k.mk_var(5), k.mk_var(6)),
                    eq2(nat_ty, add_app(k.mk_var(0), k.mk_var(5)), add_app(k.mk_var(0), k.mk_var(6)))))))
    cal_branch = k.mk_lambda(type0, k.mk_lambda(nat_ty, k.mk_app(k.mk_app(refs['refl_c'], nat_ty), add_app(k.mk_var(0), k.mk_var(5)))))
    cal_body = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty, k.mk_lambda(nat_ty,
        k.mk_lambda(eq2(nat_ty, k.mk_var(1), k.mk_var(2)), k.mk_match(k.mk_var(3), cal_motive, [cal_branch])))))
    k.whole_term_check(cal_body, cal_ty)
    cal_gid = k.declare_checked(cal_ty, cal_body, 'cong_add_left')
    cong_add_left_c = k.mk_const(cal_gid)

    # cong_add_right : (a b c:Nat) -> Eq Nat a b -> Eq Nat (add a c) (add b c)
    car_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_pi(eq2(nat_ty, k.mk_var(0), k.mk_var(1)),
        eq2(nat_ty, add_app(k.mk_var(0), k.mk_var(2)), add_app(k.mk_var(1), k.mk_var(2)))))))
    car_motive = k.mk_lambda(type0, k.mk_lambda(nat_ty, k.mk_lambda(nat_ty,
        k.mk_lambda(eq2(nat_ty, k.mk_var(5), k.mk_var(6)),
                    eq2(nat_ty, add_app(k.mk_var(5), k.mk_var(2)), add_app(k.mk_var(6), k.mk_var(2)))))))
    car_branch = k.mk_lambda(type0, k.mk_lambda(nat_ty, k.mk_app(k.mk_app(refs['refl_c'], nat_ty), add_app(k.mk_var(5), k.mk_var(2)))))
    car_body = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty, k.mk_lambda(nat_ty,
        k.mk_lambda(eq2(nat_ty, k.mk_var(0), k.mk_var(1)), k.mk_match(k.mk_var(3), car_motive, [car_branch])))))
    k.whole_term_check(car_body, car_ty)
    car_gid = k.declare_checked(car_ty, car_body, 'cong_add_right')
    cong_add_right_c = k.mk_const(car_gid)

    # mul_succ_right : (n m:Nat) -> Eq Nat (mul n (Succ m)) (add (mul n m) n)
    msr_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty,
        eq2(nat_ty, mul_app(k.mk_var(0), k.mk_app(succ_c, k.mk_var(1))),
            add_app(mul_app(k.mk_var(0), k.mk_var(1)), k.mk_var(0)))))
    msr_ps = k.start_proof(msr_ty)
    msr_ps.intro('n', target_meta_id=msr_ps.goal_meta_id(0))
    msr_ps.intro('m', target_meta_id=msr_ps.goal_meta_id(0))
    msr_ps.induct(0, target_meta_id=msr_ps.goal_meta_id(0))
    msr_ps.apply(k.mk_app(k.mk_app(refs['refl_c'], nat_ty), refs['zero']), target_meta_id=msr_ps.goal_meta_id(0))

    n_p = k.mk_var(2); m_v = k.mk_var(1); ih = k.mk_var(3)
    kk = mul_app(n_p, m_v)
    step1 = k.mk_app(k.mk_app(k.mk_app(k.mk_app(cong_add_right_c, mul_app(n_p, k.mk_app(succ_c, m_v))),
        add_app(kk, n_p)), k.mk_app(succ_c, m_v)), ih)
    middle_A = k.mk_app(k.mk_app(k.mk_app(add_assoc_c, kk), n_p), m_v)
    middle_B = k.mk_app(k.mk_app(k.mk_app(k.mk_app(cong_add_left_c, kk), add_app(n_p, m_v)), add_app(m_v, n_p)),
        k.mk_app(k.mk_app(refs['add_comm_c'], n_p), m_v))
    middle_C = symm(nat_ty, add_app(add_app(kk, m_v), n_p), add_app(kk, add_app(m_v, n_p)),
        k.mk_app(k.mk_app(k.mk_app(add_assoc_c, kk), m_v), n_p))
    middle_eq = trans2(nat_ty, add_app(add_app(kk, n_p), m_v), add_app(kk, add_app(n_p, m_v)),
        add_app(add_app(kk, m_v), n_p), middle_A,
        trans2(nat_ty, add_app(kk, add_app(n_p, m_v)), add_app(kk, add_app(m_v, n_p)),
               add_app(add_app(kk, m_v), n_p), middle_B, middle_C))
    cong_middle = congf(add_app(add_app(kk, n_p), m_v), add_app(add_app(kk, m_v), n_p), middle_eq)
    asr1 = k.mk_app(k.mk_app(refs['add_succ_right_c'], add_app(kk, n_p)), m_v)
    asr2 = symm(nat_ty, add_app(add_app(kk, m_v), k.mk_app(succ_c, n_p)), k.mk_app(succ_c, add_app(add_app(kk, m_v), n_p)),
        k.mk_app(k.mk_app(refs['add_succ_right_c'], add_app(kk, m_v)), n_p))
    combined_right = trans2(nat_ty, add_app(add_app(kk, n_p), k.mk_app(succ_c, m_v)),
        k.mk_app(succ_c, add_app(add_app(kk, n_p), m_v)), add_app(add_app(kk, m_v), k.mk_app(succ_c, n_p)), asr1,
        trans2(nat_ty, k.mk_app(succ_c, add_app(add_app(kk, n_p), m_v)), k.mk_app(succ_c, add_app(add_app(kk, m_v), n_p)),
               add_app(add_app(kk, m_v), k.mk_app(succ_c, n_p)), cong_middle, asr2))
    msr_final = trans2(nat_ty, add_app(mul_app(n_p, k.mk_app(succ_c, m_v)), k.mk_app(succ_c, m_v)),
        add_app(add_app(kk, n_p), k.mk_app(succ_c, m_v)), add_app(add_app(kk, m_v), k.mk_app(succ_c, n_p)), step1, combined_right)
    msr_ps.apply(msr_final, target_meta_id=msr_ps.goal_meta_id(0))
    assert msr_ps.is_complete()
    msr_term = msr_ps.extract()
    k.whole_term_check(msr_term, msr_ty)
    mul_succ_right_gid = k.declare_checked(msr_ty, msr_term, 'mul_succ_right')
    mul_succ_right_c = k.mk_const(mul_succ_right_gid)

    # mul_comm : (n m:Nat) -> Eq Nat (mul n m) (mul m n)
    mc_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty, eq2(nat_ty, mul_app(k.mk_var(0), k.mk_var(1)), mul_app(k.mk_var(1), k.mk_var(0)))))
    mc_ps = k.start_proof(mc_ty)
    mc_ps.intro('n', target_meta_id=mc_ps.goal_meta_id(0))
    mc_ps.intro('m', target_meta_id=mc_ps.goal_meta_id(0))
    mc_ps.induct(0, target_meta_id=mc_ps.goal_meta_id(0))
    mc_m = k.mk_var(1)
    base = symm(nat_ty, mul_app(mc_m, refs['zero']), refs['zero'], k.mk_app(mul_zero_right_c, mc_m))
    mc_ps.apply(base, target_meta_id=mc_ps.goal_meta_id(0))
    mc_np = k.mk_var(2); mc_ih = k.mk_var(3)
    mc_k = mul_app(mc_np, mc_m)
    cong_step = k.mk_app(k.mk_app(k.mk_app(k.mk_app(cong_add_right_c, mc_k), mul_app(mc_m, mc_np)), mc_m), mc_ih)
    msr_symm = symm(nat_ty, mul_app(mc_m, k.mk_app(succ_c, mc_np)), add_app(mul_app(mc_m, mc_np), mc_m),
        k.mk_app(k.mk_app(mul_succ_right_c, mc_m), mc_np))
    step_final = trans2(nat_ty, add_app(mc_k, mc_m), add_app(mul_app(mc_m, mc_np), mc_m),
        mul_app(mc_m, k.mk_app(succ_c, mc_np)), cong_step, msr_symm)
    mc_ps.apply(step_final, target_meta_id=mc_ps.goal_meta_id(0))
    assert mc_ps.is_complete()
    mc_term = mc_ps.extract()
    k.whole_term_check(mc_term, mc_ty)
    mul_comm_gid = k.declare_checked(mc_ty, mc_term, 'mul_comm')

    return dict(mul_id=mul_gid, mul_c=mul_c,
                cong_add_zero_id=cong_add_zero_gid, cong_add_zero_c=cong_add_zero_c,
                mul_zero_right_id=mul_zero_right_gid, mul_zero_right_c=mul_zero_right_c,
                mul_succ_left_unfold_id=mul_succ_left_unfold_gid, mul_succ_left_unfold_c=mul_succ_left_unfold_c,
                add_assoc_id=add_assoc_gid, add_assoc_c=add_assoc_c,
                cong_add_left_id=cal_gid, cong_add_left_c=cong_add_left_c,
                cong_add_right_id=car_gid, cong_add_right_c=cong_add_right_c,
                mul_succ_right_id=mul_succ_right_gid, mul_succ_right_c=mul_succ_right_c,
                mul_comm_id=mul_comm_gid, mul_comm_c=k.mk_const(mul_comm_gid))


def _build_le_refl_goal(k, refs):
    # (n:Nat) -> Le n n -- near-definitional (le_n itself), a sanity check on the definition more
    # than a real search test; confirmed solved by plain, unaided UniformPolicy search (6 nodes).
    return k.mk_pi(refs['nat_ty'], le_app(k, refs, k.mk_var(0), k.mk_var(0)))


def _build_zero_le_goal(k, refs):
    # (n:Nat) -> Le Zero n -- a genuine inductive proof (base case + step case via le_S), confirmed
    # solved by plain, unaided UniformPolicy search (267 nodes), independently verified.
    return k.mk_pi(refs['nat_ty'], le_app(k, refs, refs['zero'], k.mk_var(0)))


def build_base_kernel(kernel_cls=itp_env.Kernel):
    """Builds the shared, fixed BASE library: Nat, Bool, Eq/refl, add
    (recursive, checked). Deliberately does NOT include symm -- that is
    the first curriculum goal instead (solved via search, not hand-
    built), after a hand-built dependent-motive construction attempt
    was tried, found to have a real, unisolated bug, and abandoned in
    favor of the same tactic-driven approach already proven correct
    throughout this whole project (see the session notes on why: a
    hand-rolled motive is genuinely error-prone to get right, and
    letting the ALREADY-VALIDATED search machinery find the proof
    avoids that risk entirely, while still yielding the same lemma).

    kernel_cls: the class to instantiate as the kernel (default
    itp_env.Kernel). Lets a caller pass a subclass -- e.g. one that
    overrides specific methods to add timing, for benchmarking --
    without duplicating any of this function's own construction logic.

    Returns (kernel, refs): refs holds every NodeId/global id a
    curriculum or held-out goal needs to build its own type.
    """
    k = kernel_cls()
    type0 = k.mk_type(0)
    prop0 = k.mk_prop()

    nat_id = k.reserve_inductive_type('Nat', type0)
    nat_ty = k.mk_const(nat_id)
    nat_h = k.finish_inductive(nat_id, [itp_env.CtorSpec('Zero', []), itp_env.CtorSpec('Succ', [nat_ty])])
    zero = k.mk_const(nat_h.ctor_global_ids[0])
    succ_c = k.mk_const(nat_h.ctor_global_ids[1])

    bool_id = k.reserve_inductive_type('Bool', type0)
    bool_ty = k.mk_const(bool_id)
    bool_h = k.finish_inductive(bool_id, [itp_env.CtorSpec('True', []), itp_env.CtorSpec('False', [])])

    eq_sort = k.mk_pi(type0, k.mk_pi(k.mk_var(0), k.mk_pi(k.mk_var(0), prop0)))
    eq_id = k.reserve_inductive_type('Eq', eq_sort)
    eq_ty = k.mk_const(eq_id)
    eq_h = k.finish_inductive(eq_id, [itp_env.CtorSpec('refl', [type0, k.mk_var(0)],
                                                        [k.mk_var(0), k.mk_var(1), k.mk_var(1)])])
    refl_c = k.mk_const(eq_h.ctor_global_ids[0])

    add_fn_type = k.mk_pi(nat_ty, k.mk_pi(nat_ty, nat_ty))
    add_gid = k.reserve_recursive_function('add', add_fn_type)
    add_c = k.mk_const(add_gid)
    add_motive = k.mk_lambda(nat_ty, nat_ty)
    add_zero_branch = k.mk_var(1)
    add_succ_branch = k.mk_lambda(nat_ty, k.mk_app(succ_c, k.mk_app(k.mk_app(add_c, k.mk_var(2)), k.mk_var(1))))
    add_match = k.mk_match(k.mk_var(0), add_motive, [add_zero_branch, add_succ_branch])
    add_body = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty, add_match))
    k.finish_recursive_function_checked(add_gid, nat_id, 0, add_body)

    eq_n_m_ty = eq_app(k, dict(eq_ty=eq_ty), nat_ty, k.mk_var(0), k.mk_var(1))
    cong_concl = eq_app(k, dict(eq_ty=eq_ty), nat_ty, k.mk_app(succ_c, k.mk_var(0)), k.mk_app(succ_c, k.mk_var(1)))
    cong_ty = k.mk_pi(nat_ty, k.mk_pi(nat_ty, k.mk_pi(eq_n_m_ty, cong_concl)))
    cong_motive = k.mk_lambda(type0, k.mk_lambda(nat_ty, k.mk_lambda(nat_ty,
        k.mk_lambda(eq_app(k, dict(eq_ty=eq_ty), nat_ty, k.mk_var(4), k.mk_var(5)),
                    eq_app(k, dict(eq_ty=eq_ty), nat_ty, k.mk_app(succ_c, k.mk_var(4)), k.mk_app(succ_c, k.mk_var(5)))))))
    refl_branch = k.mk_lambda(type0, k.mk_lambda(nat_ty, k.mk_app(k.mk_app(refl_c, nat_ty), k.mk_app(succ_c, k.mk_var(4)))))
    cong_body = k.mk_lambda(nat_ty, k.mk_lambda(nat_ty, k.mk_lambda(eq_n_m_ty,
                                                                     k.mk_match(k.mk_var(2), cong_motive, [refl_branch]))))
    cong_gid = k.declare_checked(cong_ty, cong_body, 'cong')

    refs = dict(
        type0=type0, prop0=prop0,
        nat_id=nat_id, nat_ty=nat_ty, zero=zero, succ_c=succ_c,
        bool_id=bool_id, bool_ty=bool_ty,
        true_c=k.mk_const(bool_h.ctor_global_ids[0]), false_c=k.mk_const(bool_h.ctor_global_ids[1]),
        eq_id=eq_id, eq_ty=eq_ty, refl_c=refl_c,
        add_id=add_gid, add_c=add_c,
        cong_id=cong_gid, cong_c=k.mk_const(cong_gid),
    )
    return k, refs


@dataclass
class CurriculumGoal:
    name: str
    build_type: Callable  # (kernel, refs) -> NodeId
    lemma_name: Optional[str] = None  # if solved, declare_checked as this new global, for LATER goals in the SAME round


def _build_symm_goal(k, refs):
    # (A:Type)(x:A)(y:A) -> Eq A x y -> Eq A y x
    return k.mk_pi(refs['type0'],
             k.mk_pi(k.mk_var(0),
               k.mk_pi(k.mk_var(0),
                 k.mk_pi(eq_app(k, refs, k.mk_var(0), k.mk_var(1), k.mk_var(2)),
                         eq_app(k, refs, k.mk_var(0), k.mk_var(2), k.mk_var(1))))))


def _build_add_zero_right_goal(k, refs):
    # (n:Nat) -> Eq Nat (add n Zero) n
    return k.mk_pi(refs['nat_ty'],
             eq_app(k, refs, refs['nat_ty'],
                    k.mk_app(k.mk_app(refs['add_c'], k.mk_var(0)), refs['zero']),
                    k.mk_var(0)))


def _build_n_eq_add_n_zero_goal(k, refs):
    # (n:Nat) -> Eq Nat n (add n Zero) -- genuinely needs symm + add_zero_right
    return k.mk_pi(refs['nat_ty'],
             eq_app(k, refs, refs['nat_ty'],
                    k.mk_var(0),
                    k.mk_app(k.mk_app(refs['add_c'], k.mk_var(0)), refs['zero'])))


def _build_bool_refl_goal(k, refs):
    # (b:Bool) -> Eq Bool b b -- held-out; different type than any training goal
    return k.mk_pi(refs['bool_ty'], eq_app(k, refs, refs['bool_ty'], k.mk_var(0), k.mk_var(0)))


def _build_add_zero_left_goal(k, refs):
    # (n:Nat) -> Eq Nat (add Zero n) n -- held-out; provable by refl ALONE
    # (add's own match is on its FIRST arg, here a concrete Zero), a
    # genuinely different-shaped goal from add_zero_right despite
    # looking superficially similar.
    return k.mk_pi(refs['nat_ty'],
             eq_app(k, refs, refs['nat_ty'],
                    k.mk_app(k.mk_app(refs['add_c'], refs['zero']), k.mk_var(0)),
                    k.mk_var(0)))


def _build_add_succ_right_goal(k, refs):
    # (n m:Nat) -> Eq Nat (add n (Succ m)) (Succ (add n m)) -- has TWO
    # Nat hyps (n, m) simultaneously in scope before induction, testing
    # multi-hypothesis induction (does INDUCT(0) vs INDUCT(1) genuinely
    # get distinguished, exercising this session's own position-
    # encoding fix). Confirmed directly, via search with UniformPolicy,
    # to be provable with the existing tactic set (369 nodes, SOLVED,
    # independently verified) before being added here.
    #
    # trans and add_comm are DELIBERATELY not included anywhere in this
    # curriculum.
    #
    # trans: this comment previously said trans was "confirmed genuinely
    # unprovable with the current cases()/induct() tactic implementation",
    # citing cases()'s own inability to generalize a sibling hypothesis
    # sharing a variable with the target (h1: Eq A x y and h2: Eq A y z
    # both genuinely need y in scope at once). That limitation was found
    # and fixed later the same session (cases()'s own extended_targets/
    # revert_from construction now correctly reverts a sibling hypothesis
    # whose type depends on a generalized variable) -- confirmed directly,
    # this session: trans now solves via plain search (UniformPolicy, no
    # hand-built proof) and its extracted term passes an independent
    # commit_and_verify from scratch. It is left out of this curriculum
    # only because adding it (as a goal, a lemma, or both) is a curriculum-
    # design choice -- what position, whether it should have its own
    # lemma_name, whether a goal exercising it belongs in CURRICULUM or
    # HELD_OUT -- not because it remains unprovable.
    #
    # add_comm, in contrast, IS still genuinely broken today, but for an
    # unrelated, separately-discovered reason: given add_zero_right and
    # add_succ_right already declared as available lemmas, search finds a
    # tactic-level SOLVED path, but the extracted term fails independent
    # verification ("different variable/constant heads", a different
    # symptom and root cause than the induction/extraction bug this same
    # session found and fixed for add_zero_right/add_succ_right themselves
    # -- not yet isolated). Do not add add_comm to this curriculum, with
    # or without a lemma_name, until this is investigated and fixed; doing
    # so today would let a tactic-level SOLVED reach commit_and_verify and
    # correctly raise, which run_round now catches and logs rather than
    # crashing on, but which would still, correctly, never actually solve
    # or declare the lemma.
    return k.mk_pi(refs['nat_ty'],
             k.mk_pi(refs['nat_ty'],
               eq_app(k, refs, refs['nat_ty'],
                      k.mk_app(k.mk_app(refs['add_c'], k.mk_var(0)), k.mk_app(refs['succ_c'], k.mk_var(1))),
                      k.mk_app(refs['succ_c'], k.mk_app(k.mk_app(refs['add_c'], k.mk_var(0)), k.mk_var(1))))))


def _build_trans_goal(k, refs):
    # (A:Type)(x y z:A) -> Eq A x y -> Eq A y z -> Eq A x z -- confirmed
    # provable via plain search (UniformPolicy, no hand-built proof) and
    # independently re-verified via commit_and_verify, following this
    # session's own cases() fix for sibling-hypothesis dependent
    # generalization (h1: Eq A x y and h2: Eq A y z both genuinely need
    # y in scope at once -- see cases()'s own extended_targets/
    # revert_from construction, in itp_kernel.hpp, for the fix itself).
    # Absolute levels (this kernel's own convention: a variable's level
    # never changes with nesting depth): A=0, x=1, y=2, z=3, h1=4, h2=5.
    return k.mk_pi(refs['type0'],
             k.mk_pi(k.mk_var(0),
               k.mk_pi(k.mk_var(0),
                 k.mk_pi(k.mk_var(0),
                   k.mk_pi(eq_app(k, refs, k.mk_var(0), k.mk_var(1), k.mk_var(2)),
                     k.mk_pi(eq_app(k, refs, k.mk_var(0), k.mk_var(2), k.mk_var(3)),
                             eq_app(k, refs, k.mk_var(0), k.mk_var(1), k.mk_var(3))))))))


def _build_cong_add_zero_right_combo_goal(k, refs):
    # (n:Nat) -> Eq Nat (Succ (add n Zero)) (Succ n) -- held-out; needs
    # cong applied to add_zero_right(n), a DIFFERENT lemma combination
    # than any curriculum goal exercises (n_eq_add_n_zero combines
    # symm+add_zero_right; this combines cong+add_zero_right instead)
    # -- a genuine test of whether the model generalizes to a novel
    # pairing of lemmas it separately learned to use, not just the
    # specific combination it was trained on.
    return k.mk_pi(refs['nat_ty'],
             eq_app(k, refs, refs['nat_ty'],
                    k.mk_app(refs['succ_c'], k.mk_app(k.mk_app(refs['add_c'], k.mk_var(0)), refs['zero'])),
                    k.mk_app(refs['succ_c'], k.mk_var(0))))


CURRICULUM = [
    CurriculumGoal('symm', _build_symm_goal, lemma_name='symm'),
    CurriculumGoal('trans', _build_trans_goal, lemma_name='trans'),
    CurriculumGoal('add_zero_right', _build_add_zero_right_goal, lemma_name='add_zero_right'),
    CurriculumGoal('add_succ_right', _build_add_succ_right_goal, lemma_name='add_succ_right'),
    CurriculumGoal('n_eq_add_n_zero', _build_n_eq_add_n_zero_goal, lemma_name=None),
]

# Order-relation goals: additive, like setup_nat_order itself. A caller wanting these needs
# build_base_kernel() -> setup_nat_order(k, refs) -> refs updated -> ORDER_CURRICULUM (not folded
# directly into CURRICULUM, since CURRICULUM's own goals only ever assume build_base_kernel()'s
# refs; adding Le-dependent goals there directly would break any existing caller that doesn't also
# call setup_nat_order). le_trans is deliberately NOT in this search-driven list, but for a different
# reason than it once was: the kernel limitation that used to block it (induct() on an indexed-relation
# hypothesis that an earlier hypothesis depends on) is fixed -- see INDUCT_REVERT_SWEEP_NOTES.md -- and
# le_trans is a permanent, independently verified library lemma via setup_le_trans() (hand-scripted,
# like mul_zero_right in setup_mul). It stays out of ORDER_CURRICULUM because plain UniformPolicy search
# does not find it: no solve in 600 or 1200 simulations for either hypothesis order (262K-node trees,
# ~3.6 GB), and 3000 simulations exhausts memory. That is a search-efficiency finding, not a kernel one
# (no aborts in any run), and it makes le_trans a natural early benchmark for a learned policy.
# le_refl and zero_le are found by search, independently verified, and stay available to the agent as
# ordinary library facts it can apply, exactly like any other solved curriculum lemma.
ORDER_CURRICULUM = [
    CurriculumGoal('le_refl', _build_le_refl_goal, lemma_name='le_refl'),
    CurriculumGoal('zero_le', _build_zero_le_goal, lemma_name='zero_le'),
]


def _build_mul_zero_left_goal(k, refs):
    # (n:Nat) -> Eq Nat (mul Zero n) Zero -- near-definitional (mul's own base case); confirmed
    # solved by plain, unaided UniformPolicy search (6 nodes). mul_zero_right and cong_add_zero are
    # NOT search-discoverable goals here (confirmed: UniformPolicy does not find mul_zero_right
    # within a modest budget) -- they are built directly inside setup_mul itself, exactly like cong
    # is hand-built inside build_base_kernel for the same reason, and are already permanent library
    # facts (refs['mul_zero_right_c'], refs['cong_add_zero_c']) by the time setup_mul returns.
    return k.mk_pi(refs['nat_ty'],
             eq_app(k, refs, refs['nat_ty'], k.mk_app(k.mk_app(refs['mul_c'], refs['zero']), k.mk_var(0)), refs['zero']))


# mul-related goals: additive, like setup_mul itself. A caller wanting these needs
# build_base_kernel() -> setup_mul(k, refs) -> refs updated -> MUL_CURRICULUM.
MUL_CURRICULUM = [
    CurriculumGoal('mul_zero_left', _build_mul_zero_left_goal, lemma_name='mul_zero_left'),
]

HELD_OUT = [
    CurriculumGoal('bool_refl', _build_bool_refl_goal),
    CurriculumGoal('add_zero_left', _build_add_zero_left_goal),
    CurriculumGoal('cong_add_zero_right_combo', _build_cong_add_zero_right_combo_goal),
]


def make_const_vocab(base_library_size: int, headroom: int = 20) -> None:
    """Deprecated: TermEncoder now hashes any global id directly (see
    its own docstring) and needs no pre-sized vocabulary at all. Kept
    only so old call sites fail loudly rather than silently, rather
    than deleted outright."""
    raise RuntimeError("make_const_vocab is obsolete -- PolicyValueModel no longer takes a "
                        "const_vocab argument; pass const_table_size instead (or nothing, "
                        "for the default).")


def run_round(model: PolicyValueModel, curriculum: list[CurriculumGoal],
              max_simulations: int = 200, top_k: int = 200, kernel_cls=itp_env.Kernel,
              dirichlet_epsilon: float = 0.0, dirichlet_alpha: float = 0.3) -> tuple[list, list, list[dict]]:
    """Attempts every curriculum goal, in order, against one fresh,
    round-scoped kernel (base library + whatever earlier goals in THIS
    round have already declared). Returns (policy_examples,
    value_examples, log) -- log is one dict per goal (name, solved,
    nodes, and verification_error if the tactic layer claimed SOLVED but
    the independent check below rejected it) for reporting.

    dirichlet_epsilon > 0 mixes Dirichlet root exploration noise into
    each goal's own root expansion (see hg.search's own docstring) --
    on by default here would NOT be right: this parameter defaults to
    0 (off), since a caller who wants it should opt in explicitly, but
    self-play/training-data generation (this function) is precisely
    where root noise is standard practice, unlike evaluate() below,
    which deliberately never adds it (an evaluation run should reflect
    the model's own true, unperturbed judgment).

    Every goal the search reports SOLVED is independently re-verified,
    via commit_and_verify, before being reported solved=True here -- not
    only goals with their own lemma_name, as an earlier version of this
    function did. A tactic-level SOLVED is necessary but not sufficient
    (see commit_and_verify's own docstring, and this project's own,
    directly-confirmed history of a tactic-level success that an
    independent check then correctly rejected): without this, a goal
    with no lemma_name -- or any goal in evaluate() below, which never
    called commit_and_verify at all -- would report solved=True, and
    contribute training examples as such, purely on the search's own,
    unverified bookkeeping. commit_and_verify's own failure is caught
    here, not left to propagate: a single goal whose extracted proof
    does not survive independent re-checking is logged (with the
    kernel's own reason, under verification_error) and reported
    unsolved, rather than crashing this entire round and losing every
    other goal's own genuine progress along with it.

    Training-data extraction (policy_examples, value_examples) is
    itself gated on this same verification: a goal the tactic layer
    claimed SOLVED but that failed commit_and_verify contributes NO
    examples at all (see the inline comment at the extraction call
    below for why) -- a genuinely unsolved goal (the tactic layer never
    claimed a win in the first place) still contributes normally, since
    nothing there asserts a false "this wins" signal."""
    k, refs = build_base_kernel(kernel_cls)
    all_policy_examples: list = []
    all_value_examples: list = []
    log: list[dict] = []
    for goal in curriculum:
        library = k.index_library()
        goal_ty = goal.build_type(k, refs)
        ps = k.start_proof(goal_ty)
        root_meta_id = ps.goal_meta_id(0)
        dedup_map: dict = {}
        root = hg.get_or_create_node(k, ps, 0, dedup_map)
        policy = TrainablePolicy(model, k)
        status = hg.search(k, root, library, policy, dedup_map, max_simulations=max_simulations, top_k=top_k,
                            dirichlet_epsilon=dirichlet_epsilon, dirichlet_alpha=dirichlet_alpha)
        tactic_solved = status == hg.Status.SOLVED
        solved = False
        verification_error = None
        if tactic_solved:
            try:
                term = hg.commit_and_verify(k, ps, root, root_meta_id, goal_ty)
                solved = True
                if goal.lemma_name:
                    new_gid = k.declare_checked(goal_ty, term, goal.lemma_name)
                    refs[goal.lemma_name + '_c'] = k.mk_const(new_gid)
            except Exception as exc:
                verification_error = str(exc)

        # A verification failure means every node.cached_value along the
        # committed path was backed up (during search, via
        # update_status) from a node.status the tactic layer marked
        # SOLVED -- and nothing here goes back and corrects those,
        # retroactively, once commit_and_verify later disagrees.
        # Extracting from this dedup_map anyway would train the policy
        # and value heads to prefer a path that does not actually
        # produce a valid proof: a real reward-hacking risk, not a
        # hypothetical one, since this is exactly the shape of mistake
        # a search exploring more novel combinations (more lemmas, more
        # induction) is more likely to make, not less. Skipping
        # extraction entirely here is deliberately conservative -- some
        # OTHER part of this same tree (an unrelated sibling branch,
        # correctly scored as a dead end) might still have been
        # trustworthy, but this dedup_map has no cheap way to tell
        # "tainted by the bad path" apart from "genuinely independent"
        # node by node, and training on a false "this wins" signal is a
        # worse failure mode than losing one round's worth of data from
        # one goal. A goal the tactic layer never even claimed solved
        # carries no such risk (nothing here asserts "this is a win"
        # without justification) and is still extracted from normally.
        if not (tactic_solved and verification_error is not None):
            pe, ve = hg.extract_training_examples(k, dedup_map)
            all_policy_examples.extend(pe)
            all_value_examples.extend(ve)
        entry = dict(name=goal.name, solved=solved, nodes=len(dedup_map))
        if verification_error is not None:
            entry['verification_error'] = verification_error
        log.append(entry)
    # Defensive, not strictly required: the hypergraph a round builds is a deliberately
    # deduplicated graph (see hypergraph.HyperEdge's own docstring) that can contain
    # genuine reference cycles (a self-looping "no progress" edge, or two structurally-
    # coincident goals referencing each other) -- confirmed harmless and confirmed already
    # collected fully by Python's own AUTOMATIC, threshold-triggered gc with no manual
    # intervention at all (RSS stays flat across many rounds either way). Collecting
    # explicitly here just reclaims THIS round's cycles before the next one starts,
    # rather than leaving that to whenever the automatic threshold next fires -- cheap
    # insurance for a long, unattended training run, not a fix for a real leak.
    gc.collect()
    return all_policy_examples, all_value_examples, log


def evaluate(model: PolicyValueModel, held_out: list[CurriculumGoal],
             max_simulations: int = 200, top_k: int = 200, kernel_cls=itp_env.Kernel) -> list[dict]:
    """Runs the given model (no training) against a FRESH, BASE-ONLY
    kernel on each held-out goal. Deliberately never sees any lemma a
    training round may have declared -- isolates genuine model
    improvement from library growth.

    As in run_round above, a tactic-level SOLVED is independently
    re-verified via commit_and_verify before being reported solved=True
    here (an earlier version of this function reported the search's own,
    unverified status directly) -- see run_round's own docstring for why
    this matters, especially here: this function's own results are
    exactly the headline "solve rate" a report on this system's own
    capability would cite."""
    k, refs = build_base_kernel(kernel_cls)
    results = []
    for goal in held_out:
        library = k.index_library()
        goal_ty = goal.build_type(k, refs)
        ps = k.start_proof(goal_ty)
        root_meta_id = ps.goal_meta_id(0)
        dedup_map: dict = {}
        root = hg.get_or_create_node(k, ps, 0, dedup_map)
        policy = TrainablePolicy(model, k)
        status = hg.search(k, root, library, policy, dedup_map, max_simulations=max_simulations, top_k=top_k)
        tactic_solved = status == hg.Status.SOLVED
        solved = False
        verification_error = None
        if tactic_solved:
            try:
                hg.commit_and_verify(k, ps, root, root_meta_id, goal_ty)
                solved = True
            except Exception as exc:
                verification_error = str(exc)
        entry = dict(name=goal.name, solved=solved, nodes=len(dedup_map))
        if verification_error is not None:
            entry['verification_error'] = verification_error
        results.append(entry)
    gc.collect()  # same defensive, not-strictly-required insurance as run_round's own -- see there
    return results
