"""S19 gate: the Python API boundary. Nothing a caller passes -- however malformed -- may crash the process, hang it, or be accepted when it is nonsense.

A training loop builds terms in Python and calls the kernel thousands of times a second; one bad argument (an out-of-range global id, a variable that is not in scope, a Match with
too few branches) must be an exception the caller can catch, never a segmentation fault that ends the run. The kernel itself assumes its terms are well-scoped and uses unchecked
indexing for speed, so EVERY entry point that takes a term or an id validates it first (itp_bindings.cpp: require_scoped, require_global, require_meta, ...). This gate is the test
that those checks exist, are the right ones, and are not too strict:

  A. deterministic cases -- for each entry point, inputs that must be REJECTED (an ill-scoped term, an unknown id, a malformed node, an inconsistent argument list) and a few that
     must be ACCEPTED (so a guard that rejects everything is also caught). A case passes when the call raises or reports failure; it fails if the process dies, the call is accepted,
     or an accepting call is rejected.
  B. a random API fuzzer -- thousands of calls with arguments drawn from typed pools (well-typed terms, ill-scoped terms, junk ids) against every method; only "no crash, no hang" is
     checked, and every crash is reported with the call that caused it and a seed that reproduces it.

Both run in CHILD processes that print a marker before each call; when a child dies the parent names the call and carries on with the rest."""
import sys, os, subprocess, time, random

HERE = os.path.dirname(os.path.abspath(__file__))
FUZZ_SEEDS = [int(x) for x in os.environ.get('S19_SEEDS', '1,2,3,4').split(',')]
FUZZ_CALLS = int(os.environ.get('S19_CALLS', 15000))
HUGE = 4_000_000


class W:
    """a fresh prop world with the terms the cases need"""
    def __init__(self):
        sys.path.insert(0, HERE)
        import prop_world as PW, itp_env_lnl as E
        self.E = E; self.k, self.g = PW.build_prop_world(); k = self.k
        mk = lambda n: k.mk_const(self.g[n])
        self.NAT, self.Z, self.S, self.ADD, self.EQ, self.REFL, self.TRUE, self.FALSE = (mk(n) for n in ('Nat', 'Zero', 'Succ', 'add', 'Eq', 'Refl' if 'Refl' in self.g else 'refl', 'True', 'False'))
        self.T0, self.T1 = k.mk_type(0), k.mk_type(1)
        self.app = k.mk_app
        self.one = self.app(self.S, self.Z)
        self.stmt = self.app(self.app(self.app(self.EQ, self.NAT), self.Z), self.Z)                    # Eq Nat Zero Zero
        self.refl_z = self.app(self.app(self.REFL, self.NAT), self.Z)                               # refl Nat Zero : Eq Nat Zero Zero
        self.b0, self.b1 = k.mk_bvar(0), k.mk_bvar(1)
        self.f0, self.f1, self.f2, self.f5 = (k.mk_fvar(i) for i in (0, 1, 2, 5))
        self.lam_b1 = k.mk_lambda(self.NAT, self.b1)                                                 # fun _ : Nat => #1   -- one loose bound variable
        self.ngl = k.num_globals()


def rejected(r):
    """did the call report failure (as opposed to returning a result)?"""
    return isinstance(r, tuple) and len(r) >= 1 and r[0] is False


def cases():
    """(name, expectation, function of W); expectation 'reject' = must raise or report failure, 'accept' = must succeed"""
    C = []
    def add(name, expect, fn): C.append((name, expect, fn))
    # ---- evaluation entry points: a variable that is not in scope would index past the end of the environment
    add('normalize_open: loose bound variable', 'reject', lambda w: w.k.normalize_open(w.b0, 0))
    add('normalize_open: loose bound variable under a binder', 'reject', lambda w: w.k.normalize_open(w.lam_b1, 0))
    add('normalize_open: context variable beyond n0', 'reject', lambda w: w.k.normalize_open(w.f0, 0))
    add('normalize_open: context variable beyond n0 (n0 = 2)', 'reject', lambda w: w.k.normalize_open(w.f2, 2))
    add('normalize_open: nested ill-scoped variable', 'reject', lambda w: w.k.normalize_open(w.app(w.S, w.app(w.S, w.f5)), 3))
    add('normalize_open: well-scoped variable', 'accept', lambda w: w.k.normalize_open(w.f1, 2))
    add('normalize_open: closed term', 'accept', lambda w: w.k.normalize_open(w.app(w.app(w.ADD, w.one), w.one), 0))
    add('conv_open: loose bound variable', 'reject', lambda w: w.k.conv_open(w.b0, w.Z, 0))
    add('conv_open: loose bound variable on the right', 'reject', lambda w: w.k.conv_open(w.Z, w.lam_b1, 0))
    add('conv_open: context variable beyond n0', 'reject', lambda w: w.k.conv_open(w.f0, w.f0, 0))
    add('conv_open: context variable beyond n0 (right side)', 'reject', lambda w: w.k.conv_open(w.Z, w.f2, 2))
    add('conv_open: well-scoped', 'accept', lambda w: w.k.conv_open(w.f1, w.f1, 2))
    add('infer_open: loose bound variable', 'reject', lambda w: w.k.infer_open(w.b0, []))
    add('infer_open: context variable beyond the context', 'reject', lambda w: w.k.infer_open(w.f0, []))
    add('infer_open: context variable beyond the context (one hypothesis)', 'reject', lambda w: w.k.infer_open(w.f1, [w.NAT]))
    add('infer_open: hypothesis type mentions itself', 'reject', lambda w: w.k.infer_open(w.Z, [w.f0]))
    add('infer_open: hypothesis type mentions a later hypothesis', 'reject', lambda w: w.k.infer_open(w.Z, [w.NAT, w.f1]))
    add('infer_open: variable in scope', 'accept', lambda w: w.k.infer_open(w.f0, [w.NAT]))
    add('check_open: ill-scoped term', 'reject', lambda w: w.k.check_open(w.f0, w.NAT, []))
    add('check_open: ill-scoped expected type', 'reject', lambda w: w.k.check_open(w.Z, w.f0, []))
    add('check_open: ill-scoped hypothesis type', 'reject', lambda w: w.k.check_open(w.Z, w.NAT, [w.f0]))
    add('check_open: loose bound variable in the term', 'reject', lambda w: w.k.check_open(w.lam_b1, w.NAT, []))
    add('check_open: well-typed', 'accept', lambda w: w.k.check_open(w.refl_z, w.stmt, []))
    add('unify_open: ill-scoped left', 'reject', lambda w: w.k.unify_open(w.f0, w.Z, []))
    add('unify_open: ill-scoped right', 'reject', lambda w: w.k.unify_open(w.Z, w.f2, [w.NAT]))
    add('unify_open: loose bound variable', 'reject', lambda w: w.k.unify_open(w.b0, w.b0, []))
    add('unify_open: ill-scoped hypothesis type', 'reject', lambda w: w.k.unify_open(w.Z, w.Z, [w.f0]))
    add('unify_open: well-scoped', 'accept', lambda w: w.k.unify_open(w.f0, w.f0, [w.NAT]))
    add('unify_drain_open: ill-scoped left', 'reject', lambda w: w.k.unify_drain_open(w.f0, w.Z, []))
    add('unify_drain_open: loose bound variable', 'reject', lambda w: w.k.unify_drain_open(w.Z, w.lam_b1, []))
    add('unify_drain_open: well-scoped', 'accept', lambda w: w.k.unify_drain_open(w.Z, w.Z, []))
    # ---- declarations and proof states
    add('declare_checked: ill-scoped type', 'reject', lambda w: w.k.declare_checked(w.f0, w.Z, 'bad1'))
    add('declare_checked: ill-scoped value', 'reject', lambda w: w.k.declare_checked(w.NAT, w.f0, 'bad2'))
    add('declare_checked: loose bound variable', 'reject', lambda w: w.k.declare_checked(w.NAT, w.lam_b1, 'bad3'))
    add('declare_unchecked: ill-scoped type', 'reject', lambda w: w.k.declare_unchecked(w.f0, None, 'bad4'))
    add('declare_unchecked: ill-scoped value', 'reject', lambda w: w.k.declare_unchecked(w.NAT, w.b0, 'bad5'))
    add('start_proof: ill-scoped goal', 'reject', lambda w: w.k.start_proof(w.f0))
    add('start_proof: loose bound variable', 'reject', lambda w: w.k.start_proof(w.b0))
    add('materialize_from_node: ill-scoped target', 'reject', lambda w: w.k.materialize_from_node(w.f1, [w.NAT], ['h']))
    add('materialize_from_node: hypothesis mentions itself', 'reject', lambda w: w.k.materialize_from_node(w.NAT, [w.f0], ['h']))
    add('materialize_from_node: name list too short', 'reject', lambda w: w.k.materialize_from_node(w.NAT, [w.NAT, w.NAT], ['h']))
    add('materialize_from_node: well-scoped', 'accept', lambda w: w.k.materialize_from_node(w.NAT, [w.NAT], ['h']).num_goals())
    add('ProofState.apply: ill-scoped term', 'reject', lambda w: w.k.start_proof(w.stmt).apply(w.f0))
    add('ProofState.apply: loose bound variable', 'reject', lambda w: w.k.start_proof(w.stmt).apply(w.lam_b1))
    add('ProofState.exact: ill-scoped term', 'reject', lambda w: w.k.start_proof(w.stmt).exact(w.f0))
    add('ProofState.exact: well-typed term', 'accept', lambda w: w.k.start_proof(w.stmt).exact(w.refl_z))
    add('ProofState.rewrite: ill-scoped equation', 'reject', lambda w: w.k.start_proof(w.stmt).rewrite(w.f0, w.Z, w.Z, w.g['transport'], w.g['symm']))
    add('ProofState.rewrite: ill-scoped x', 'reject', lambda w: w.k.start_proof(w.stmt).rewrite(w.refl_z, w.f0, w.Z, w.g['transport'], w.g['symm']))
    add('ProofState.rewrite: ill-scoped y', 'reject', lambda w: w.k.start_proof(w.stmt).rewrite(w.refl_z, w.Z, w.f5, w.g['transport'], w.g['symm']))
    add('ProofState.rewrite: unknown transport lemma', 'reject', lambda w: w.k.start_proof(w.stmt).rewrite(w.refl_z, w.Z, w.Z, HUGE, w.g['symm']))
    add('ProofState.rewrite: unknown symmetry lemma', 'reject', lambda w: w.k.start_proof(w.stmt).rewrite(w.refl_z, w.Z, w.Z, w.g['transport'], HUGE))
    add('ProofState.cases: no such hypothesis', 'reject', lambda w: w.k.start_proof(w.stmt).cases(7))
    add('ProofState.cases: generalizing a missing hypothesis', 'reject', lambda w: w.k.start_proof(w.stmt).cases(0, None, [9]))
    add('ProofState.induct: no such hypothesis', 'reject', lambda w: w.k.start_proof(w.stmt).induct(7))
    add('ProofState.induct: unknown equality family', 'reject', lambda w: w.k.start_proof(w.stmt).induct(0, None, [], HUGE, HUGE))
    add('ProofState.induct: unknown equality family (reflexivity lemma fine)', 'reject', lambda w: w.k.start_proof(w.stmt).induct(0, None, [], HUGE, w.g['refl']))
    add('ProofState.induct: unknown reflexivity lemma (equality family fine)', 'reject', lambda w: w.k.start_proof(w.stmt).induct(0, None, [], w.g['Eq'], HUGE))
    add('mk_bvar: an index too large to shift', 'reject', lambda w: w.k.mk_bvar(2**32 - 1))
    add('mk_fvar: a level too large to extend', 'reject', lambda w: w.k.mk_fvar(2**32 - 1))
    add('mk_fix: a decreasing argument too large', 'reject', lambda w: w.k.mk_fix(w.k.mk_pi(w.NAT, w.NAT), 2**32 - 1, w.k.mk_lambda(w.NAT, w.b0)))
    add('normalize_open: an absurd context size (would allocate billions of bindings)', 'reject', lambda w: w.k.normalize_open(w.Z, 2**32 - 1))
    add('conv_open: an absurd context size', 'reject', lambda w: w.k.conv_open(w.Z, w.Z, 2**32 - 1))
    add('fresh_meta: an absurd scope', 'reject', lambda w: w.k.fresh_meta(w.NAT, 2**32 - 1))
    add('lift: an absurd cutoff', 'reject', lambda w: w.k.lift(w.Z, 1, 2**32 - 1))
    add('lift: an absurd amount', 'reject', lambda w: w.k.lift(w.Z, 2**32 - 1, 0))
    add('close_telescope: an absurd base', 'reject', lambda w: w.k.close_telescope('pi', [w.NAT], w.NAT, 2**32 - 1))
    add('close_telescope: well-formed', 'accept', lambda w: w.k.close_telescope('lam', [w.NAT], w.b0, 0))
    add('ProofState.intro: not a Pi', 'reject', lambda w: w.k.start_proof(w.stmt).intro('x'))
    add('ProofState.apply: no such goal', 'reject', lambda w: w.k.start_proof(w.stmt).apply(w.refl_z, 12345))
    add('ProofState.goal_target: no such goal', 'reject', lambda w: w.k.start_proof(w.stmt).goal_target(3))
    add('ProofState.goal_hyp_type: no such hypothesis', 'reject', lambda w: w.k.start_proof(w.stmt).goal_hyp_type(0, 3))
    add('an unfinished proof extracted and offered to check_open', 'reject', lambda w: (lambda ps: w.k.check_open(ps.extract(), w.stmt, []))(w.k.start_proof(w.stmt)))
    add('an unfinished proof extracted and offered to declare_checked', 'reject', lambda w: (lambda ps: w.k.declare_checked(w.stmt, ps.extract(), 'forged_partial'))(w.k.start_proof(w.stmt)))
    add('check_partial: an unfinished proof is a well-typed term with holes', 'accept', lambda w: (lambda ps: w.k.check_partial(ps.extract(), w.stmt, []))(w.k.start_proof(w.stmt)))
    add('check_partial: a hole of the wrong type is still ill-typed', 'reject', lambda w: w.k.check_partial(w.k.mk_meta(w.k.fresh_meta(w.NAT, 0)), w.stmt, []))
    add('check_partial: ill-scoped terms are refused as before', 'reject', lambda w: w.k.check_partial(w.f0, w.NAT, []))
    # ---- inductive types and recursive functions
    add('reserve_inductive_type: ill-scoped sort', 'reject', lambda w: w.k.reserve_inductive_type('Bad1', w.f0))
    add('finish_inductive: ill-scoped field type', 'reject', lambda w: w.k.finish_inductive(w.k.reserve_inductive_type('Bad2', w.T1), [w.E.CtorSpec('c', [w.f5], [])]))
    add('finish_inductive: field type mentions a later field', 'reject', lambda w: w.k.finish_inductive(w.k.reserve_inductive_type('Bad3', w.T1), [w.E.CtorSpec('c', [w.f1, w.NAT], [])]))
    add('finish_inductive: ill-scoped result index', 'reject', lambda w: w.k.finish_inductive(w.k.reserve_inductive_type('Bad4', w.T1), [w.E.CtorSpec('c', [], [w.f2])]))
    add('finish_inductive: unknown type id', 'reject', lambda w: w.k.finish_inductive(HUGE, [w.E.CtorSpec('c', [], [])]))
    add('finish_inductive_unchecked: ill-scoped field type', 'reject', lambda w: w.k.finish_inductive_unchecked(w.k.reserve_inductive_type('Bad5', w.T1), [w.E.CtorSpec('c', [w.f5], [])]))
    add('finish_inductive_unchecked: unknown type id', 'reject', lambda w: w.k.finish_inductive_unchecked(HUGE, [w.E.CtorSpec('c', [], [])]))
    add('finish_inductive: a type that is already finished', 'reject', lambda w: w.k.finish_inductive(w.g['Nat'], [w.E.CtorSpec('c', [], [])]))
    add('reserve_recursive_function: ill-scoped type', 'reject', lambda w: w.k.reserve_recursive_function('bad_rec1', w.f0))
    add('finish_recursive_function_checked: ill-scoped body', 'reject', lambda w: w.k.finish_recursive_function_checked(w.k.reserve_recursive_function('bad_rec2', w.app(w.app(w.k.mk_pi(w.NAT, w.NAT), w.Z), w.Z)), w.g['Nat'], 0, w.f0))
    add('finish_recursive_function_checked: unknown function id', 'reject', lambda w: w.k.finish_recursive_function_checked(HUGE, w.g['Nat'], 0, w.Z))
    add('finish_recursive_function_checked: unknown inductive id', 'reject', lambda w: w.k.finish_recursive_function_checked(w.k.reserve_recursive_function('bad_rec3', w.k.mk_pi(w.NAT, w.NAT)), HUGE, 0, w.Z))
    add('finish_recursive_function: unknown function id', 'reject', lambda w: w.k.finish_recursive_function(HUGE, w.g['Nat'], 0, w.Z))
    add('finish_recursive_function: ill-scoped body', 'reject', lambda w: w.k.finish_recursive_function(w.k.reserve_recursive_function('bad_rec4', w.k.mk_pi(w.NAT, w.NAT)), w.g['Nat'], 0, w.b0))
    add('finish_recursive_function_checked: decreasing argument out of range', 'reject', lambda w: w.k.finish_recursive_function_checked(w.k.reserve_recursive_function('bad_rec5', w.k.mk_pi(w.NAT, w.NAT)), w.g['Nat'], 50, w.k.mk_lambda(w.NAT, w.b0)))
    # ---- ids that name nothing
    add('mk_const: unknown global', 'reject', lambda w: w.k.mk_const(HUGE))
    add('mk_const: the first id past the end', 'reject', lambda w: w.k.mk_const(w.k.num_globals()))
    add('mk_const: the last existing id', 'accept', lambda w: w.k.mk_const(w.k.num_globals() - 1))
    add('mk_meta: the first id past the end', 'reject', lambda w: (lambda m: w.k.mk_meta(m + 1))(w.k.fresh_meta(w.NAT, 0)))
    add('mk_meta: the last existing id', 'accept', lambda w: (lambda m: w.k.mk_meta(m))(w.k.fresh_meta(w.NAT, 0)))
    add('global_info: the first id past the end', 'reject', lambda w: w.k.global_info(w.k.num_globals()))
    add('param_slots: the first id past the end', 'reject', lambda w: w.k.param_slots(w.k.num_globals()))
    add('finish_recursive_function_checked: finishing a function twice', 'reject', lambda w: (lambda fid: (w.k.finish_recursive_function_checked(fid, w.g['Nat'], 0, w.k.mk_lambda(w.NAT, w.b0)), w.k.finish_recursive_function_checked(fid, w.g['Nat'], 0, w.k.mk_lambda(w.NAT, w.Z)))[1])(w.k.reserve_recursive_function('twice1', w.k.mk_pi(w.NAT, w.NAT))))
    add('finish_recursive_function: finishing a function twice', 'reject', lambda w: (lambda fid: (w.k.finish_recursive_function(fid, w.g['Nat'], 0, w.k.mk_lambda(w.NAT, w.b0)), w.k.finish_recursive_function(fid, w.g['Nat'], 0, w.k.mk_lambda(w.NAT, w.Z)))[1])(w.k.reserve_recursive_function('twice2', w.k.mk_pi(w.NAT, w.NAT))))
    add('finish_recursive_function_checked: a function that was declared as an ordinary axiom', 'reject', lambda w: w.k.finish_recursive_function_checked(w.k.declare_unchecked(w.k.mk_pi(w.NAT, w.NAT), None, 'ax_fn'), w.g['Nat'], 0, w.k.mk_lambda(w.NAT, w.b0)))
    add('finish_recursive_function_checked: recursion over a type that is not inductive', 'reject', lambda w: w.k.finish_recursive_function_checked(w.k.reserve_recursive_function('over_axiom', w.k.mk_pi(w.NAT, w.NAT)), w.g['add'], 0, w.k.mk_lambda(w.NAT, w.b0)))
    add('finish_recursive_function_checked: a well-formed non-recursive function', 'accept', lambda w: w.k.finish_recursive_function_checked(w.k.reserve_recursive_function('ident', w.k.mk_pi(w.NAT, w.NAT)), w.g['Nat'], 0, w.k.mk_lambda(w.NAT, w.b0)))
    add('finish_inductive_unchecked: finishing a type twice', 'reject', lambda w: (lambda tid: (w.k.finish_inductive_unchecked(tid, [w.E.CtorSpec('a', [], [])]), w.k.finish_inductive_unchecked(tid, [w.E.CtorSpec('b', [], [])]))[1])(w.k.reserve_inductive_type('Twice', w.T1)))
    add('finish_inductive: a global that was declared as an axiom', 'reject', lambda w: w.k.finish_inductive(w.k.declare_unchecked(w.T1, None, 'ax_type'), [w.E.CtorSpec('c', [], [])]))
    add('finish_inductive: a reserved type is finished once', 'accept', lambda w: w.k.finish_inductive(w.k.reserve_inductive_type('Once', w.T1), [w.E.CtorSpec('c', [], [])]).type_global_id)
    add('mk_meta: unknown meta variable', 'reject', lambda w: w.k.mk_meta(HUGE))
    add('meta_is_solved: unknown meta variable', 'reject', lambda w: w.k.meta_is_solved(HUGE))
    add('meta_solution_raw: unknown meta variable', 'reject', lambda w: w.k.meta_solution_raw(HUGE))
    add('fresh_meta: ill-scoped type', 'reject', lambda w: w.k.fresh_meta(w.f0, 0))
    add('fresh_meta: well-scoped type', 'accept', lambda w: w.k.fresh_meta(w.NAT, 0))
    add('fork_single_goal: unknown meta', 'reject', lambda w: w.k.fork_single_goal(w.k.start_proof(w.stmt), HUGE))
    add('param_slots: unknown family', 'reject', lambda w: w.k.param_slots(HUGE))
    add('param_slots: a family', 'accept', lambda w: w.k.param_slots(w.g['Eq']))
    add('global_info: unknown global', 'reject', lambda w: w.k.global_info(HUGE))
    add('is_inductive: unknown global', 'reject', lambda w: w.k.is_inductive(HUGE))
    add('quot_role: unknown global', 'reject', lambda w: w.k.quot_role(HUGE))
    add('declare_quot: unknown equality family', 'reject', lambda w: w.k.declare_quot(HUGE))
    add('declare_quot: not the equality family', 'reject', lambda w: w.k.declare_quot(w.g['Nat']))
    add('set_option: unknown option', 'reject', lambda w: w.k.set_option('no_such_option', True))
    # ---- malformed nodes: built from valid parts, but not terms of any type
    add('infer_open: constant of an unknown global used after a bad mk_const', 'reject', lambda w: w.k.infer_open(w.k.mk_const(HUGE), []))
    add('infer_open: Fix over a non-Pi type', 'reject', lambda w: w.k.infer_open(w.k.mk_fix(w.NAT, 0, w.Z), []))
    add('infer_open: Fix with a decreasing argument out of range', 'reject', lambda w: w.k.infer_open(w.k.mk_fix(w.k.mk_pi(w.NAT, w.NAT), 99, w.k.mk_lambda(w.NAT, w.b0)), []))
    add('check_open: Fix with a decreasing argument out of range', 'reject', lambda w: w.k.check_open(w.k.mk_fix(w.k.mk_pi(w.NAT, w.NAT), 99, w.k.mk_lambda(w.NAT, w.b0)), w.k.mk_pi(w.NAT, w.NAT), []))
    add('infer_open: Match with no branches on Nat', 'reject', lambda w: w.k.infer_open(w.k.mk_match(w.Z, w.k.mk_lambda(w.NAT, w.NAT), []), []))
    add('infer_open: Match with too many branches', 'reject', lambda w: w.k.infer_open(w.k.mk_match(w.Z, w.k.mk_lambda(w.NAT, w.NAT), [w.Z] * 9), []))
    add('infer_open: Match on a function', 'reject', lambda w: w.k.infer_open(w.k.mk_match(w.S, w.k.mk_lambda(w.NAT, w.NAT), [w.Z, w.Z]), []))
    add('normalize_open: Match with too few branches', 'reject', lambda w: w.k.normalize_open(w.k.mk_match(w.one, w.k.mk_lambda(w.NAT, w.NAT), [w.Z]), 0))
    add('normalize_open: Match with no branches', 'reject', lambda w: w.k.normalize_open(w.k.mk_match(w.Z, w.k.mk_lambda(w.NAT, w.NAT), []), 0))
    add('conv_open: Match with too few branches', 'reject', lambda w: w.k.conv_open(w.k.mk_match(w.one, w.k.mk_lambda(w.NAT, w.NAT), [w.Z]), w.Z, 0))
    add('normalize_open: Match on a non-inductive value (stuck, not a crash)', 'accept', lambda w: w.k.normalize_open(w.k.mk_match(w.NAT, w.k.mk_lambda(w.NAT, w.NAT), [w.Z, w.Z]), 0))
    add('infer_open: the highest allowed universe has no type (the level does not wrap around)', 'reject', lambda w: w.k.infer_open(w.k.mk_type(2**30), []))
    add('infer_open: a high universe below the limit has a type', 'accept', lambda w: w.k.infer_open(w.k.mk_type(2**30 - 1), []))
    # ---- a metavariable is a hole, never a proof
    add('declare_checked: an unsolved metavariable offered as a proof of False', 'reject', lambda w: w.k.declare_checked(w.FALSE, w.k.mk_meta(w.k.fresh_meta(w.FALSE, 0)), 'forged1'))
    add('declare_checked: a metavariable inside a larger term', 'reject', lambda w: w.k.declare_checked(w.stmt, w.app(w.app(w.REFL, w.NAT), w.k.mk_meta(w.k.fresh_meta(w.NAT, 0))), 'forged2'))
    add('declare_checked: a metavariable in the statement', 'reject', lambda w: w.k.declare_checked(w.k.mk_meta(w.k.fresh_meta(w.T1, 0)), w.Z, 'forged3'))
    add('check_open: an unsolved metavariable offered as a proof of False', 'reject', lambda w: w.k.check_open(w.k.mk_meta(w.k.fresh_meta(w.FALSE, 0)), w.FALSE, []))
    add('check_open: a metavariable in the expected type', 'reject', lambda w: w.k.check_open(w.Z, w.k.mk_meta(w.k.fresh_meta(w.T1, 0)), []))
    add('declare_unchecked: a metavariable in the type', 'reject', lambda w: w.k.declare_unchecked(w.k.mk_meta(w.k.fresh_meta(w.T1, 0)), None, 'forged4'))
    add('start_proof: a metavariable in the goal', 'reject', lambda w: w.k.start_proof(w.k.mk_meta(w.k.fresh_meta(w.T1, 0))))
    add('finish_recursive_function_checked: a metavariable in the body', 'reject', lambda w: w.k.finish_recursive_function_checked(w.k.reserve_recursive_function('forged_rec', w.k.mk_pi(w.NAT, w.NAT)), w.g['Nat'], 0, w.k.mk_lambda(w.NAT, w.k.mk_meta(w.k.fresh_meta(w.NAT, 0)))))
    add('finish_inductive: a metavariable in a field type', 'reject', lambda w: w.k.finish_inductive(w.k.reserve_inductive_type('Forged5', w.T1), [w.E.CtorSpec('c', [w.k.mk_meta(w.k.fresh_meta(w.T1, 0))], [])]))
    add('extract on a finished proof is fine, and the result passes the strict check', 'accept', lambda w: (lambda ps: (ps.exact(w.refl_z), w.k.check_open(ps.extract(), w.stmt, []))[1])(w.k.start_proof(w.stmt)))
    add('declare_checked: a finished proof extracted by tactics', 'accept', lambda w: (lambda ps: (ps.apply(w.app(w.REFL, w.NAT)), w.k.declare_checked(w.stmt, ps.extract(), 'fine1'))[1])(w.k.start_proof(w.stmt)))
    add('mk_type: a level too large to have a successor', 'reject', lambda w: w.k.mk_type(2**32 - 1))
    add('close_telescope: unknown kind', 'reject', lambda w: w.k.close_telescope('bogus', [w.NAT], w.NAT, 0))
    add('lift: shifting a bound variable past the end of the index range', 'reject', lambda w: w.k.lift(w.b0, 2**32 - 1, 0))
    add('instantiate: more values than binders', 'accept', lambda w: w.k.instantiate(w.b0, [w.Z, w.Z, w.Z]))
    return C


def classify(expect, fn, w):
    try: r = fn(w)
    except BaseException as e:                                      # nanobind turns C++ exceptions into Python ones; anything Python-level counts as a rejection
        if isinstance(e, (KeyboardInterrupt, SystemExit)): raise
        if isinstance(e, MemoryError): return ('FAIL', 'MemoryError: an absurd size was not rejected up front, the kernel tried to allocate it')
        return ('ok', 'raised %s' % type(e).__name__) if expect == 'reject' else ('FAIL', 'a call that must succeed raised %s: %s' % (type(e).__name__, str(e)[:100]))
    if rejected(r): return ('ok', 'reported failure') if expect == 'reject' else ('FAIL', 'a call that must succeed reported failure: %s' % (r,))
    return ('FAIL', 'ACCEPTED: %r' % (r if isinstance(r, (bool, int, str, tuple)) else type(r).__name__,)) if expect == 'reject' else ('ok', 'accepted')


def run_cases(start):
    cs = cases()
    for i in range(start, len(cs)):
        name, expect, fn = cs[i]
        print('BEGIN %d %s' % (i, name), flush=True)
        verdict, msg = classify(expect, fn, W())                    # a fresh world per case: a case that declares something cannot affect the next one
        print('END %d %s %s' % (i, verdict, msg), flush=True)
    print('DONE', flush=True)


# ================================================================== B. the random API fuzzer
import re
SIG = re.compile(r'^(\w+)\((.*)\) -> ')

def parse_params(doc):
    """'name(self, a: T, b: U = d) -> R'  ->  [(name, type, has_default)]"""
    m = SIG.match(doc.strip().split('\n')[0])
    inside = m.group(2); parts = []; depth = 0; cur = ''
    for ch in inside:
        if ch in '[(': depth += 1
        if ch in '])': depth -= 1
        if ch == ',' and depth == 0: parts.append(cur.strip()); cur = ''
        else: cur += ch
    parts.append(cur.strip()); out = []
    for p in parts[1:]:
        if not p: continue
        default = ' = ' in p; p = p.split(' = ')[0]; name, ty = p.split(': ', 1); out.append((name.strip(), ty.strip(), default))
    return out


class Fuzz:
    SKIP = {'checkpoint', 'rewind', 'set_option', 'declare_quot'}      # stale-node protocols are covered by the probe gates; options/quot would change the world for the whole run
    def __init__(self, seed):
        self.rng = random.Random(seed); self.w = W(); w = self.w; k = w.k; self.k = k
        self.nodes = [w.NAT, w.Z, w.S, w.ADD, w.EQ, w.REFL, w.T0, w.T1, w.one, w.stmt, w.refl_z, w.b0, w.b1, w.f0, w.f1, w.f2, w.f5, w.lam_b1,
                      w.TRUE, w.FALSE, k.mk_pi(w.NAT, w.NAT), k.mk_lambda(w.NAT, w.b0), k.mk_lambda(w.NAT, k.mk_app(w.S, w.b0)), w.app(w.app(w.ADD, w.one), w.one)]
        for n in ('And', 'Or', 'Ex', 'Le', 'Not', 'Iff', 'transport', 'symm', 'trans', 'cong', 'le_n', 'le_S', 'zero_le', 'conj', 'inl', 'inr', 'tt'):
            self.nodes.append(k.mk_const(w.g[n]))
        self.gids = sorted(set(w.g.values())); self.metas = []
        self.ps = [k.start_proof(w.stmt), k.start_proof(k.mk_pi(w.NAT, w.app(w.app(w.app(w.EQ, w.NAT), w.b0), w.b0))),
                   k.start_proof(k.mk_pi(w.NAT, k.mk_pi(w.NAT, w.app(w.app(w.app(w.EQ, w.NAT), w.app(w.app(w.ADD, w.Z), w.b0)), w.b0))))]
        self.libs = [k.index_library()]; self.names = ['', 'x', 'h', 'n', 'Nat', 'f', 'a b', 'é', 'fz' + 'z' * 40]
        self.ctors = []
        methods = [m for m in dir(type(k)) if not m.startswith('_') and m not in self.SKIP]
        self.kmethods = [(m, parse_params(getattr(type(k), m).__doc__)) for m in methods]
        pm = [m for m in dir(type(self.ps[0])) if not m.startswith('_')]
        self.pmethods = [(m, parse_params(getattr(type(self.ps[0]), m).__doc__)) for m in pm]
        self.garbage = [w.b0, w.b1, w.f0, w.f1, w.f2, w.f5, w.lam_b1, k.mk_app(w.S, w.f1), k.mk_lambda(w.NAT, k.mk_bvar(3)), k.mk_pi(w.f0, w.NAT), k.mk_meta(k.fresh_meta(w.NAT, 0)),
                        k.mk_fix(k.mk_pi(w.NAT, w.NAT), 0, k.mk_lambda(w.NAT, w.b0)), k.mk_match(w.Z, k.mk_lambda(w.NAT, w.NAT), [w.Z]), k.mk_let(w.NAT, w.Z, w.f0)]
        self.refresh_typed()
        self.log = []

    # ---- well-typed material, so that calls get past the boundary checks and exercise the kernel proper
    def gen_nat(self, depth, nb):
        """a term of type Nat under nb bound variables of type Nat"""
        r = self.rng; w = self.w; k = self.k
        c = r.random()
        if depth <= 0 or c < 0.25: return w.Z if (nb == 0 or r.random() < 0.5) else k.mk_bvar(r.randrange(nb))
        if c < 0.55: return k.mk_app(w.S, self.gen_nat(depth - 1, nb))
        return k.mk_app(k.mk_app(w.ADD, self.gen_nat(depth - 1, nb)), self.gen_nat(depth - 1, nb))
    def gen_stmt(self, depth, nb):
        r = self.rng; w = self.w; k = self.k; g = w.g; c = r.random()
        eq = lambda: k.mk_app(k.mk_app(k.mk_app(w.EQ, w.NAT), self.gen_nat(2, nb)), self.gen_nat(2, nb))
        if depth <= 0 or c < 0.35: return eq()
        if c < 0.5: return k.mk_app(k.mk_app(k.mk_const(g['Le']), self.gen_nat(2, nb)), self.gen_nat(2, nb))
        if c < 0.62: return k.mk_app(k.mk_app(k.mk_const(g['And']), self.gen_stmt(depth - 1, nb)), self.gen_stmt(depth - 1, nb))
        if c < 0.72: return k.mk_app(k.mk_app(k.mk_const(g['Or']), self.gen_stmt(depth - 1, nb)), self.gen_stmt(depth - 1, nb))
        if c < 0.78: return k.mk_app(k.mk_const(g['Not']), self.gen_stmt(depth - 1, nb))
        if c < 0.82: return w.TRUE if r.random() < 0.5 else w.FALSE
        return k.mk_pi(w.NAT, self.gen_stmt(depth - 1, nb + 1))
    def gen_proofish(self, depth, nb):
        """a proof-shaped term: usually ill-typed, sometimes right"""
        r = self.rng; w = self.w; k = self.k; g = w.g; c = r.random()
        if depth <= 0 or c < 0.2: return k.mk_app(k.mk_app(w.REFL, w.NAT), self.gen_nat(2, nb))
        if c < 0.3: return k.mk_const(g['tt'])
        if c < 0.45: return k.mk_app(k.mk_const(g['le_n']), self.gen_nat(2, nb))
        if c < 0.6: return k.mk_app(k.mk_app(k.mk_app(k.mk_app(k.mk_const(g['conj']), self.gen_stmt(1, nb)), self.gen_stmt(1, nb)), self.gen_proofish(depth - 1, nb)), self.gen_proofish(depth - 1, nb))
        if c < 0.75: return k.mk_lambda(w.NAT, self.gen_proofish(depth - 1, nb + 1))
        if c < 0.85: return k.mk_app(k.mk_const(g['zero_le']), self.gen_nat(2, nb))
        return self.gen_nat(2, nb)
    def refresh_typed(self):
        r = self.rng; n = self.nodes
        for _ in range(6):
            n.append(self.gen_nat(3, 0)); n.append(self.gen_stmt(3, 0)); n.append(self.gen_proofish(3, 0))
        for _ in range(2): self.ps.append(self.k.start_proof(self.gen_stmt(3, 0)))
        if len(self.ps) > 40: del self.ps[3:len(self.ps) - 30]
        if len(n) > 800: del n[24:len(n) - 500]
    def small(self):
        return self.rng.choice([0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 8, 20])
    def an_int(self, pname):
        r = self.rng
        if 'gid' in pname or 'global' in pname or pname in ('gid', 'ind_gid', 'type_global_id', 'family_gid'):
            return r.choice(self.gids) if r.random() < 0.85 else r.choice([len(self.gids) + r.randrange(0, 5), HUGE, r.randrange(0, 3 * len(self.gids))])
        if 'meta' in pname or pname == 'id':
            return r.choice(self.metas) if self.metas and r.random() < 0.7 else r.choice([0, 1, 2, 5, 100, HUGE])
        if pname == 'level': return r.choice([0, 1, 2, 3, 7])
        if pname in ('i', 'h'): return self.small() % 4
        return self.small()
    def a_node(self):
        r = self.rng; c = r.random()
        if c < 0.12: return r.choice(self.garbage)                       # ill-scoped / junk: must be refused cleanly
        if c < 0.55: return r.choice(self.nodes[24:]) if len(self.nodes) > 24 else r.choice(self.nodes)     # the closed, mostly well-typed material
        return r.choice(self.nodes)
    def arg(self, pname, ty):
        r = self.rng
        if ty == 'int': return self.an_int(pname)
        if ty == 'int | None': return None if r.random() < 0.6 else self.an_int('meta_id')
        if ty == 'bool': return r.random() < 0.5
        if ty == 'str': return r.choice(self.names) if pname != 'kind' else r.choice(['lam', 'pi', 'x'])
        if ty == 'itp_env_lnl.NodeId | None': return None if r.random() < 0.5 else self.a_node()
        if ty == 'itp_env_lnl.NodeId': return self.a_node()
        if 'Sequence[itp_env_lnl.NodeId]' in ty: return [self.a_node() for _ in range(r.choice([0, 0, 1, 1, 2, 3, 4]))]
        if 'Sequence[int]' in ty: return [self.small() for _ in range(r.choice([0, 0, 1, 2, 3]))]
        if 'Sequence[str]' in ty: return [r.choice(self.names) for _ in range(r.choice([0, 1, 1, 2, 3]))]
        if 'Sequence[itp_env_lnl.CtorSpec]' in ty:
            return [self.w.E.CtorSpec(r.choice(self.names), [self.a_node() for _ in range(r.choice([0, 0, 1, 2]))], [self.a_node() for _ in range(r.choice([0, 0, 1]))]) for _ in range(r.choice([0, 1, 1, 2, 3]))]
        if ty == 'itp_env_lnl.ProofState': return r.choice(self.ps)
        if ty == 'itp_env_lnl.Library': return r.choice(self.libs)
        raise RuntimeError('fuzzer: no generator for parameter type %r' % ty)
    def args_for(self, params):
        out = []
        for (pname, ty, has_default) in params:
            if has_default and self.rng.random() < 0.4: continue
            out.append(self.arg(pname, ty))
        # a defaulted parameter may only be dropped from the end: pass everything up to the last one we generated
        return out
    def absorb(self, res):
        E = self.w.E
        if isinstance(res, E.NodeId):
            if len(self.nodes) < 600: self.nodes.append(res)
            else: self.nodes[self.rng.randrange(24, len(self.nodes))] = res
        elif isinstance(res, E.ProofState):
            if len(self.ps) < 40: self.ps.append(res)
            else: self.ps[self.rng.randrange(3, len(self.ps))] = res
        elif isinstance(res, E.InductiveHandle):
            self.gids.append(res.type_global_id); self.gids.extend(res.ctor_global_ids)
        elif isinstance(res, E.Library):
            self.libs.append(res)
        elif isinstance(res, int) and not isinstance(res, bool):
            pass
    def directed(self, describe):
        """what a prover does: work on a goal with the tactics, the way the search would -- candidates from the library index, intro / cases / induct on hypotheses, then
        extract and admit the finished proof. Reaches the code that random single calls almost never get to."""
        r = self.rng; k = self.k; ps = r.choice(self.ps[3:] if len(self.ps) > 3 else self.ps)
        lib = k.index_library() if r.random() < 0.3 or not self.libs else r.choice(self.libs)
        for _ in range(r.choice([2, 4, 8, 12])):
            if describe: print('CALL directed-step', flush=True)
            if ps.num_goals() == 0: break
            try:
                nh = ps.goal_num_hyps(0); c = r.random()
                if c < 0.25: ps.intro('v%d' % nh)
                elif c < 0.7:
                    cands = k.apply_candidates(ps, lib)
                    if cands: ps.apply(r.choice(cands))
                    else: ps.intro('w')
                elif c < 0.85 and nh: ps.cases(r.randrange(nh))
                elif nh: ps.induct(r.randrange(nh))
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)): raise
        try:
            if ps.num_goals() == 0 and ps.is_complete():
                t = ps.extract(); self.nodes.append(t)
                stmt = ps.goal_target(0) if ps.num_goals() else None
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)): raise

    def step(self, i, describe=False):
        r = self.rng; k = self.k
        if i % 25 == 0: self.refresh_typed()
        if r.random() < 0.12:
            self.directed(describe); return
        kind = r.random()
        if kind < 0.28: pool, obj = self.pmethods, r.choice(self.ps)
        else: pool, obj = self.kmethods, k
        name, params = pool[r.randrange(len(pool))]
        # bias: construct before consuming
        if kind > 0.28 and kind < 0.5:
            cand = [m for m in pool if m[0].startswith('mk_')]; name, params = cand[r.randrange(len(cand))]
        fn = getattr(obj, name)
        # defaulted parameters are positional-or-keyword: build positional args up to the last non-defaulted one, plus optionally some defaulted ones in order
        pos = []
        for (pname, ty, has_default) in params:
            if has_default and r.random() < 0.5: break
            pos.append(self.arg(pname, ty))
        if describe: print('CALL %s%s' % (name, tuple(describe_arg(self.k, a) for a in pos)), flush=True)
        if name == 'start_proof' or name == 'materialize_from_node': pass
        try:
            res = fn(*pos)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)): raise
            return
        if name in ('reserve_inductive_type', 'reserve_recursive_function', 'declare_checked', 'declare_unchecked', 'fresh_meta'): self.gids.append(res) if name != 'fresh_meta' else self.metas.append(res)
        self.absorb(res)
        if isinstance(res, tuple):
            for x in res: self.absorb(x)
        if name == 'start_proof' or name == 'materialize_from_node' or name == 'fork_single_goal':
            try: self.metas.append(res.goal_meta_id(0))
            except Exception: pass


def describe_arg(k, a):
    E = sys.modules['itp_env_lnl']
    if isinstance(a, E.NodeId):
        try: return k.dump_raw(a)
        except Exception: return '<node>'
    if isinstance(a, list): return [describe_arg(k, x) for x in a]
    if isinstance(a, E.CtorSpec): return 'CtorSpec'
    if isinstance(a, E.ProofState): return '<ProofState %d goals>' % a.num_goals()
    return a


def run_fuzz(seed, calls, start, skip, describe_at):
    f = Fuzz(seed)
    for i in range(calls):
        if i in skip:
            f.rng.random(); continue              # a call that crashed the process: do not execute it again (and move the random stream on, or the next call would be the same one)
        print('BEGIN %d' % i, flush=True)
        f.step(i, describe=(describe_at == i))
        if i < start: pass
    print('DONE', flush=True)


# ================================================================== the gate
def child_main(argv):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))      # an input that is not rejected up front must fail fast with MemoryError, not eat the machine
    mode = argv[0]
    if mode == 'cases': run_cases(int(argv[1]))
    elif mode == 'fuzz':
        seed, calls = int(argv[1]), int(argv[2]); skip = set(int(x) for x in argv[3].split(',') if x); describe_at = int(argv[4]) if len(argv) > 4 else -1
        run_fuzz(seed, calls, 0, skip, describe_at)


def spawn(args, timeout):
    return subprocess.run([sys.executable, os.path.abspath(__file__), '--child'] + [str(a) for a in args], capture_output=True, text=True, timeout=timeout, cwd=HERE)


def main():
    t0 = time.time(); fails = []
    cs = cases(); n_ok = 0; i = 0; crashes = 0
    while i < len(cs):
        try: r = spawn(['cases', i], 300); out = r.stdout.strip().split('\n') if r.stdout.strip() else []; rc = r.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')).strip().split('\n'); rc = 'timeout'
        begun = [l for l in out if l.startswith('BEGIN ')]; ended = [l for l in out if l.startswith('END ')]
        for l in ended:
            _, idx, verdict, *msg = l.split(' ', 3)
            if verdict == 'ok': n_ok += 1
            else: fails.append('case %s [%s]: %s' % (idx, cs[int(idx)][0], ' '.join(msg)))
        if rc == 0 and out and out[-1] == 'DONE': break
        last = int(begun[-1].split()[1]) if begun else i
        if len(ended) == 0 or int(ended[-1].split()[1]) < last:
            fails.append('case %d [%s]: the process %s (exit %s)' % (last, cs[last][0], 'hung' if rc == 'timeout' else 'died', rc)); crashes += 1
        i = last + 1
    print('   A. boundary cases: %d, passed %d, failed %d' % (len(cs), n_ok, len(cs) - n_ok), flush=True)
    for seed in FUZZ_SEEDS:
        skip = []; done = False; seed_crashes = 0
        while not done and seed_crashes < 6:
            try: r = spawn(['fuzz', seed, FUZZ_CALLS, ','.join(map(str, skip))], 600); out = r.stdout.strip().split('\n') if r.stdout.strip() else []; rc = r.returncode
            except subprocess.TimeoutExpired as e:
                out = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')).strip().split('\n'); rc = 'timeout'
            if rc == 0 and out and out[-1] == 'DONE': done = True; break
            begun = [l for l in out if l.startswith('BEGIN ')]
            if not begun: fails.append('fuzz seed %d: the child failed before the first call (exit %s): %s' % (seed, rc, (r.stderr if rc != 'timeout' else '')[-300:])); break
            last = int(begun[-1].split()[1])
            try:
                d = spawn(['fuzz', seed, FUZZ_CALLS, ','.join(map(str, skip)), last], 600); calls = [l for l in d.stdout.split('\n') if l.startswith('CALL ')]
                desc = calls[-1][:400] if calls else '<could not reproduce>'
            except subprocess.TimeoutExpired: desc = '<reproduction timed out>'
            fails.append('fuzz seed %d call %d: the process %s (exit %s): %s' % (seed, last, 'hung' if rc == 'timeout' else 'died', rc, desc))
            skip.append(last); seed_crashes += 1
        print('   B. fuzz seed %d: %d calls, %s' % (seed, FUZZ_CALLS, 'no crash, no hang' if not seed_crashes else '%d CRASHES' % seed_crashes), flush=True)
    print('   (%.0f s)' % (time.time() - t0))
    for f in fails: print('   FAIL ' + f)
    print('S19 BOUNDARY GATE: ' + ('PASS' if not fails else 'FAIL'))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child': child_main(sys.argv[2:])
    else: main()
