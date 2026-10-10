// Bindings for the hybrid-variable kernel (refactor spike S1): terms, NbE, declarations, a small typing fragment and the
// diagnostics the differential harness needs. Tactics, unification and the full typing core are ported in S2-S4.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <functional>
#include <sys/resource.h>
#include "itp_kernel.hpp"

namespace nb = nanobind;
using namespace nb::literals;

struct ITPError : std::runtime_error { using std::runtime_error::runtime_error; };


struct KernelWrapper;
// Is this id a node of the ACTIVE kernel's pools? (Call after `auto g = activate();`.) A NodeId is a (tag, index) pair into per-kind pools, so an id that did not come from this kernel -- or that was
// reclaimed by a rewind -- may index past a pool; the kernel reads pools directly, so such an id must be refused here, never followed. This cannot detect an id from ANOTHER kernel that happens to be in
// range (that is aliasing, not a crash); the Python authoring layer wraps ids with their owner for that.
inline bool node_ok(itp::NodeId n) {
    auto& c = itp::ctx();
    switch (n.tag) {
        case itp::NodeType::VAR:       return n.index < c.vars.storage.size();
        case itp::NodeType::APP:       return n.index < c.apps.storage.size();
        case itp::NodeType::LAMBDA:    return n.index < c.lambdas.storage.size();
        case itp::NodeType::PI:        return n.index < c.pis.storage.size();
        case itp::NodeType::TYPE:      return n.index < c.types.storage.size();
        case itp::NodeType::CONST:     return n.index < c.consts.storage.size();
        case itp::NodeType::MATCH:     return n.index < c.matches.storage.size();
        case itp::NodeType::META:      return n.index < c.metas.storage.size();
        case itp::NodeType::TYPE_UNIV: return n.index < c.type_univs.storage.size();
        case itp::NodeType::LET:       return n.index < c.lets.storage.size();
        case itp::NodeType::FIX:       return n.index < c.fixes.storage.size();
    }
    return false;
}
inline void require_node(itp::NodeId n, const char* what) {
    if (!node_ok(n)) throw ITPError(std::string("invalid term handle (") + what + "): it does not name a node of this kernel (a handle from another kernel, or one reclaimed by a rewind)");
}
inline void require_scoped(itp::NodeId t, uint32_t nfree, const char* what) {
    require_node(t, what);
    if (!itp::detail::well_scoped(t, nfree))
        throw ITPError(std::string("ill-scoped term (") + what + "): it has an unbound bound variable, or a context variable beyond the " + std::to_string(nfree) + " in scope");
}
inline void require_scoped_telescope(const std::vector<itp::NodeId>& types, const char* what) { for (uint32_t j = 0; j < types.size(); ++j) require_scoped(types[j], j, what); }
// Every id and index that crosses the boundary is validated here: the kernel indexes its tables with them directly, so a bad one must be an exception, never an out-of-range read.
// (These read the active ExecutionContext: call them after `auto g = activate();`.)
inline void require_global(uint32_t gid, const char* what) {
    if (gid >= itp::ctx().globals.defs.size()) throw ITPError(std::string("no such global (") + what + "): id " + std::to_string(gid) + ", there are " + std::to_string(itp::ctx().globals.defs.size()) + " globals");
}
inline void require_meta(uint32_t id, const char* what) {
    if (id >= itp::ctx().meta_ctx.entries.size()) throw ITPError(std::string("unknown metavariable id (") + what + "): " + std::to_string(id));
}
inline void require_small(uint32_t v, const char* what) {
    if (v > itp::kMaxUniverseLevel) throw ITPError(std::string(what) + " is too large (" + std::to_string(v) + "; the limit is " + std::to_string(itp::kMaxUniverseLevel) + ")");
}
// A metavariable is a hole: declarations, statements and final verification must not contain one (the typing rule for ?m is "it has the type it was created with").
inline void require_meta_free(itp::NodeId t, const char* what) {
    if (itp::detail::contains_meta(t)) throw ITPError(std::string("a metavariable occurs in the ") + what + ": declarations and final checks need meta-free terms (finish the proof / instantiate first)");
}
// A global that is only RESERVED (a type, but no body or constructors yet) has no meaning: a final check must not rest on one (see GlobalEnv::declare).
inline void require_complete_globals(itp::NodeId t, const char* what) {
    if (auto bad = itp::detail::unfinished_global_in(itp::ctx().globals, t, itp::kNoGlobal))
        throw ITPError(std::string("the ") + what + " mentions global '" + itp::ctx().globals.defs[*bad].name + "', which is declared but not yet finished");
}
template <class R> void raise_if_error(const R& r) { if (!r) throw ITPError(r.error().message); }

// Wraps itp::ProofState; keeps a raw pointer to its Kernel (start_proof binds nb::keep_alive<0,1> so the kernel outlives it).
struct ProofStateHandle {
    std::unique_ptr<itp::ProofState> ps = std::make_unique<itp::ProofState>();
    KernelWrapper* kernel = nullptr;
    uint32_t root_meta_id = 0;
    itp::ContextGuard activate();
    uint32_t scope_of(std::optional<uint32_t> tm) { auto idx = itp::find_goal_index(*ps, tm); return idx ? ps->goals[*idx].ec.size() : 0; }
    void intro_(const std::string& name, std::optional<uint32_t> tm) { auto g = activate(); raise_if_error(itp::intro(*ps, name, tm)); }
    void apply_(itp::NodeId term, std::optional<uint32_t> tm) { auto g = activate(); require_scoped(term, scope_of(tm), "apply"); raise_if_error(itp::apply(*ps, term, tm)); }
    void cases_(uint32_t hyp_level, std::optional<uint32_t> tm, const std::vector<uint32_t>& gen) { auto g = activate(); raise_if_error(itp::cases(*ps, hyp_level, tm, gen)); }
    void induct_(uint32_t hyp_level, std::optional<uint32_t> tm, const std::vector<uint32_t>& gen, std::optional<uint32_t> eq_gid, std::optional<uint32_t> refl_gid) {
        auto g = activate(); if (eq_gid) require_global(*eq_gid, "induct eq_gid"); if (refl_gid) require_global(*refl_gid, "induct refl_gid");
        raise_if_error(itp::induct(*ps, hyp_level, tm, gen, eq_gid, refl_gid)); }
    void exact_(itp::NodeId term, std::optional<uint32_t> tm) { auto g = activate(); require_scoped(term, scope_of(tm), "exact"); raise_if_error(itp::exact(*ps, term, tm)); }
    void rewrite_(itp::NodeId eq_term, itp::NodeId x, itp::NodeId y, uint32_t transport_gid, uint32_t symm_gid, std::optional<uint32_t> tm) {
        auto g = activate(); require_global(transport_gid, "rewrite transport lemma"); require_global(symm_gid, "rewrite symmetry lemma"); uint32_t sc = scope_of(tm); require_scoped(eq_term, sc, "rewrite eq_term"); require_scoped(x, sc, "rewrite x"); require_scoped(y, sc, "rewrite y"); raise_if_error(itp::rewrite_eq(*ps, eq_term, x, y, transport_gid, symm_gid, tm));
    }
    bool is_complete() { auto g = activate(); return itp::proof_complete(*ps); }
    // The current proof term. While goals are open it contains holes (metavariables); check_open and declare_checked refuse such a term, so an unfinished proof can never be admitted.
    itp::NodeId extract() { auto g = activate(); return itp::extract_proof_term(root_meta_id); }
    std::size_t num_goals() const { return ps->goals.size(); }
    std::size_t num_pending() const { return ps->es.pending.size(); }   // deferred unification constraints
    uint32_t goal_meta_id(std::size_t i) const { return ps->goals.at(i).meta_id; }
    itp::NodeId goal_target(std::size_t i) const { return ps->goals.at(i).target; }
    std::size_t goal_num_hyps(std::size_t i) const { return ps->goals.at(i).hyps.size(); }
    std::string goal_hyp_name(std::size_t i, std::size_t h) const { return ps->goals.at(i).hyps.at(h).name; }
    itp::NodeId goal_hyp_type(std::size_t i, std::size_t h) const { return ps->goals.at(i).hyps.at(h).type; }
};

struct KernelWrapper {
    std::unique_ptr<itp::ExecutionContext> exec = std::make_unique<itp::ExecutionContext>();
    itp::ContextGuard activate() { return itp::ContextGuard(*exec); }

    // ---- S5: the search's probing protocol. A probe is: checkpoint, fork the goal, apply a tactic to the fork, read off, rewind. Rewind reverts the
    // ExecutionContext (node pools, metas, derived globals) but cannot see a ProofState, which is why a probe works on a FORK and discards it.
    itp::ExecutionContext::Checkpoint checkpoint() { auto g = activate(); return itp::ctx().checkpoint(); }
    void rewind(const itp::ExecutionContext::Checkpoint& cp) { auto g = activate(); itp::ctx().rewind(cp); }
    ProofStateHandle fork_single_goal(ProofStateHandle& source, uint32_t target_meta_id) {
        auto g = activate();
        auto idx_opt = itp::find_goal_index(*source.ps, target_meta_id);
        if (!idx_opt) throw ITPError("fork_single_goal: no goal with the given meta_id in the source ProofState");
        ProofStateHandle h;
        h.ps->goals.push_back(source.ps->goals[*idx_opt]);   // a Goal is an ordinary value type
        h.ps->es = itp::ElabState{};                          // deliberately fresh, not shared with the source
        h.root_meta_id = target_meta_id; h.kernel = this;
        return h;
    }
    ProofStateHandle materialize_from_node(itp::NodeId target, const std::vector<itp::NodeId>& hyp_types, const std::vector<std::string>& hyp_names) {
        auto g = activate(); require_scoped_telescope(hyp_types, "hypothesis type"); require_scoped(target, hyp_types.size(), "target");
        if (hyp_types.size() != hyp_names.size()) throw ITPError("materialize_from_node: hyp_types and hyp_names must be the same length");
        itp::ElabCtx ec{}; std::vector<itp::Hypothesis> hyps;
        for (size_t i = 0; i < hyp_types.size(); ++i) { ec = ec.extend(hyp_types[i]); hyps.push_back(itp::Hypothesis{hyp_names[i], hyp_types[i]}); }
        uint32_t meta = itp::ctx().meta_ctx.fresh(itp::close_pi_over_context(target, ec), ec.size());   // closed Pi type, like every goal meta (D1)
        ProofStateHandle h; h.ps->goals.push_back(itp::Goal{meta, ec, std::move(hyps), target}); h.root_meta_id = meta; h.kernel = this;
        return h;
    }
    uint32_t loose_range_of(itp::NodeId t) { auto g = activate(); require_node(t, "loose_range_of"); return itp::loose_range(t); }
    void set_option(const std::string& name, bool value) { auto g = activate(); if (name == "parameters") itp::ctx().param_mode = value; else if (name == "prop_elim") itp::ctx().set_prop_mode(value); else throw ITPError("unknown option: " + name); }
    bool get_option(const std::string& name) { auto g = activate(); if (name == "parameters") return itp::ctx().param_mode; if (name == "prop_elim") return itp::ctx().prop_mode; throw ITPError("unknown option: " + name); }
    std::vector<int> param_slots(uint32_t family_gid) { auto g = activate(); require_global(family_gid, "param_slots"); std::vector<int> out; for (bool b : itp::family_param_slots(family_gid)) out.push_back(b ? 1 : 0); return out; }
    itp::DiscriminationTree index_library() { auto g = activate(); return itp::index_library(); }
    std::vector<itp::NodeId> apply_candidates(ProofStateHandle& psh, const itp::DiscriminationTree& lib, std::optional<uint32_t> target_meta_id) {
        auto g = activate(); return itp::apply_candidates(*psh.ps, lib, target_meta_id);
    }
    ProofStateHandle start_proof(itp::NodeId goal_type) {
        auto g = activate(); require_scoped(goal_type, 0, "goal type"); require_meta_free(goal_type, "goal type"); ProofStateHandle h; h.kernel = this;
        *h.ps = itp::start_proof(goal_type); h.root_meta_id = h.ps->goals[0].meta_id; return h;
    }
    itp::NodeId mk_type(uint32_t l) { auto g = activate(); require_small(l, "universe level"); return itp::mk_type(l); }
    itp::NodeId mk_bvar(uint32_t i) { auto g = activate(); require_small(i, "bound variable index"); return itp::mk_bvar(i); }
    itp::NodeId mk_fvar(uint32_t l) { auto g = activate(); require_small(l, "context variable level"); return itp::mk_fvar(l); }
    itp::NodeId mk_app(itp::NodeId f, itp::NodeId a) { auto g = activate(); require_node(f, "function"); require_node(a, "argument"); return itp::mk_app(f, a); }
    itp::NodeId mk_lambda(itp::NodeId t, itp::NodeId b) { auto g = activate(); require_node(t, "binder type"); require_node(b, "body"); return itp::mk_lambda(t, b); }
    itp::NodeId mk_pi(itp::NodeId d, itp::NodeId c) { auto g = activate(); require_node(d, "domain"); require_node(c, "codomain"); return itp::mk_pi(d, c); }
    itp::NodeId mk_let(itp::NodeId t, itp::NodeId v, itp::NodeId b) { auto g = activate(); require_node(t, "let type"); require_node(v, "let value"); require_node(b, "let body"); return itp::mk_let(t, v, b); }
    itp::NodeId mk_const(uint32_t gid) { auto g = activate(); require_global(gid, "mk_const"); return itp::mk_const(gid); }
    itp::NodeId mk_match(itp::NodeId s, itp::NodeId m, std::vector<itp::NodeId> bs) { auto g = activate(); require_node(s, "scrutinee"); require_node(m, "motive"); for (auto b : bs) require_node(b, "branch"); return itp::mk_match(s, m, std::move(bs)); }
    itp::NodeId mk_fix(itp::NodeId fn_type, uint32_t dec, itp::NodeId body) { auto g = activate(); require_small(dec, "decreasing argument"); require_node(fn_type, "fix type"); require_node(body, "fix body"); return itp::mk_fix(fn_type, dec, body); }

    uint32_t reserve_inductive_type(const std::string& name, itp::NodeId sort) { auto g = activate(); require_scoped(sort, 0, "inductive sort"); require_meta_free(sort, "inductive sort"); return itp::reserve_inductive_type(name, sort); }
    itp::InductiveHandle finish_inductive(uint32_t tid, std::vector<itp::CtorSpec> ctors) { auto g = activate();
        for (auto& c : ctors) { require_scoped_telescope(c.arg_types, "constructor field type"); for (auto a : c.arg_types) require_meta_free(a, "constructor field type"); for (auto r : c.result_indices) { require_scoped(r, static_cast<uint32_t>(c.arg_types.size()), "constructor result index"); require_meta_free(r, "constructor result index"); } }
        return itp::finish_inductive(tid, std::move(ctors)); }
    uint32_t reserve_recursive_function(const std::string& name, itp::NodeId type) { auto g = activate(); require_scoped(type, 0, "function type"); require_meta_free(type, "function type"); return itp::reserve_recursive_function(name, type); }
    void finish_recursive_function(uint32_t gid, uint32_t ind_gid, uint32_t dec, itp::NodeId body) { auto g = activate(); require_scoped(body, 0, "recursive function body"); require_meta_free(body, "recursive function body"); itp::finish_recursive_function(gid, ind_gid, dec, body); }
    uint32_t declare_unchecked(itp::NodeId type, std::optional<itp::NodeId> value, const std::string& name) { auto g = activate(); require_scoped(type, 0, "declared type"); require_meta_free(type, "declared type"); if (value) { require_scoped(*value, 0, "declared value"); require_meta_free(*value, "declared value"); } return itp::declare_unchecked(type, value, name); }

    // ---- S2: checked declarations and the full typing core
    itp::InductiveHandle finish_inductive_unchecked(uint32_t tid, std::vector<itp::CtorSpec> ctors) { auto g = activate();
        for (auto& c : ctors) { require_scoped_telescope(c.arg_types, "constructor field type"); for (auto a : c.arg_types) require_meta_free(a, "constructor field type"); for (auto r : c.result_indices) { require_scoped(r, static_cast<uint32_t>(c.arg_types.size()), "constructor result index"); require_meta_free(r, "constructor result index"); } }
        return itp::finish_inductive_unchecked(tid, std::move(ctors)); }
    uint32_t declare_checked(itp::NodeId type, itp::NodeId value, const std::string& name) {
        auto g = activate(); require_scoped(type, 0, "declared type"); require_scoped(value, 0, "declared value"); auto r = itp::declare_checked(type, value, name);
        if (!r) throw ITPError(r.error().message);
        return *r;
    }
    void finish_recursive_function_checked(uint32_t gid, uint32_t ind_gid, uint32_t dec, itp::NodeId body) {
        auto g = activate(); require_scoped(body, 0, "recursive function body"); auto r = itp::finish_recursive_function_checked(gid, ind_gid, dec, body);
        if (!r) throw ITPError(r.error().message);
    }
    std::pair<bool, nb::object> infer_open(itp::NodeId t, const std::vector<itp::NodeId>& hyp_types) {
        auto g = activate(); require_scoped_telescope(hyp_types, "hypothesis type"); require_scoped(t, static_cast<uint32_t>(hyp_types.size()), "term"); itp::ElabCtx ec; itp::ElabState es;
        for (auto h : hyp_types) ec = ec.extend(h);
        auto r = itp::infer(t, ec, es);
        if (r) return {true, nb::cast(*r)};
        return {false, nb::cast(r.error().message)};
    }
    // check_open is the VERIFICATION entry point (final proofs are re-checked with it): a hole must not pass for a proof, so no metavariable may occur anywhere.
    // check_partial is the elaboration-time check: the same typing, but metavariables are allowed (the rule for ?m is "it has the type it was created with"), which is what
    // "this partial proof, with its open goals as holes, is well typed" means.
    std::pair<bool, std::string> check_open(itp::NodeId t, itp::NodeId expected, const std::vector<itp::NodeId>& hyp_types) { return check_impl(t, expected, hyp_types, false); }
    std::pair<bool, std::string> check_partial(itp::NodeId t, itp::NodeId expected, const std::vector<itp::NodeId>& hyp_types) { return check_impl(t, expected, hyp_types, true); }
    std::pair<bool, std::string> check_impl(itp::NodeId t, itp::NodeId expected, const std::vector<itp::NodeId>& hyp_types, bool allow_holes) {
        auto g = activate(); require_scoped_telescope(hyp_types, "hypothesis type"); require_scoped(t, hyp_types.size(), "term"); require_scoped(expected, hyp_types.size(), "expected type");
        if (!allow_holes) {
            for (auto h : hyp_types) { require_meta_free(h, "hypothesis type"); require_complete_globals(h, "hypothesis type"); }
            require_meta_free(t, "term"); require_complete_globals(t, "term");
            require_meta_free(expected, "expected type"); require_complete_globals(expected, "expected type");
        }
        itp::ElabCtx ec; itp::ElabState es;
        // The statement and its context are CLAIMS, not proofs: a final check proves nothing unless they are well-formed types (`check` takes its expected type on trust, and an ill-typed
        // term that merely evaluates to a type would be "proved"). Elaboration-time checks (holes allowed) work on contexts the tactics built.
        for (auto h : hyp_types) {
            if (!allow_holes) { auto s = itp::sort_level_of(h, ec, es, "a hypothesis type is not a type"); if (!s) return {false, "ill-formed context: " + s.error().message}; }
            ec = ec.extend(h);
        }
        if (!allow_holes) { auto s = itp::sort_level_of(expected, ec, es, "the statement is not a type"); if (!s) return {false, "ill-formed statement: " + s.error().message}; }
        auto r = itp::check(t, expected, ec, es);
        if (r) return {true, "OK"};
        return {false, r.error().message};
    }
    // ---- S3: direct access to metas and unification (used by the unifier unit tests)
    uint32_t fresh_meta(itp::NodeId type, uint32_t scope) { auto g = activate(); require_small(scope, "meta scope"); require_scoped(type, scope, "meta type"); return itp::ctx().meta_ctx.fresh(type, scope); }
    itp::NodeId mk_meta(uint32_t id) { auto g = activate(); require_meta(id, "mk_meta"); return itp::mk_meta(id); }
    bool meta_is_solved(uint32_t id) { auto g = activate(); require_meta(id, "meta_is_solved"); return itp::ctx().meta_ctx.is_solved(id); }
    std::string meta_solution_raw(uint32_t id) {
        auto g = activate(); require_meta(id, "meta_solution_raw"); auto sol = itp::ctx().meta_ctx[id].solution;
        return sol ? dump_raw(*sol) : std::string("");
    }
    std::tuple<bool, std::string, uint32_t> unify_open(itp::NodeId a, itp::NodeId b, const std::vector<itp::NodeId>& hyp_types) {
        auto g = activate(); require_scoped_telescope(hyp_types, "hypothesis type"); require_scoped(a, hyp_types.size(), "lhs"); require_scoped(b, hyp_types.size(), "rhs"); itp::ElabCtx ec; itp::ElabState es;
        for (auto h : hyp_types) ec = ec.extend(h);
        auto r = itp::unify(a, b, ec, es);
        return {static_cast<bool>(r), r ? std::string("OK") : r.error().message, static_cast<uint32_t>(es.pending.size())};
    }
    std::tuple<bool, std::string, uint32_t> unify_drain_open(itp::NodeId a, itp::NodeId b, const std::vector<itp::NodeId>& hyp_types) {
        auto g = activate(); require_scoped_telescope(hyp_types, "hypothesis type"); require_scoped(a, hyp_types.size(), "lhs"); require_scoped(b, hyp_types.size(), "rhs"); itp::ElabCtx ec; itp::ElabState es;           // the production sequence of apply: unify, then drain in the OUTER context
        for (auto h : hyp_types) ec = ec.extend(h);
        auto r = itp::unify(a, b, ec, es);
        if (r) r = itp::drain_pending(ec, es);
        return {static_cast<bool>(r), r ? std::string("OK") : r.error().message, static_cast<uint32_t>(es.pending.size())};
    }
    itp::NodeId close_telescope(const std::string& kind, const std::vector<itp::NodeId>& doms, itp::NodeId body, uint32_t base) {
        auto g = activate(); require_small(base, "telescope base"); for (auto d : doms) require_node(d, "telescope domain"); require_node(body, "telescope body");
        if (kind != "lam" && kind != "pi") throw ITPError("close_telescope: kind must be 'lam' or 'pi', not '" + kind + "'");
        return kind == "lam" ? itp::close_lambda_telescope(doms, body, base) : itp::close_pi_telescope(doms, body, base);
    }
    itp::Env fresh_env(uint32_t n0) { itp::Env env; for (uint32_t i = 0; i < n0; ++i) env = env.extend(itp::fresh_neutral(i)); return env; }
    itp::NodeId normalize_open(itp::NodeId t, uint32_t n0) { auto g = activate(); require_small(n0, "context size"); require_scoped(t, n0, "term"); return itp::normalize(t, fresh_env(n0)); }
    bool conv_open(itp::NodeId a, itp::NodeId b, uint32_t n0) {
        auto g = activate(); require_small(n0, "context size"); require_scoped(a, n0, "lhs"); require_scoped(b, n0, "rhs"); itp::Env env = fresh_env(n0);
        return itp::conv_value(itp::eval(a, env), itp::eval(b, env), n0);
    }
    uint32_t loose_range(itp::NodeId t) { auto g = activate(); require_node(t, "loose_range"); return itp::loose_range(t); }
    // Test hook for the release machinery: builds a chain of `depth` nested values ITERATIVELY (kind 0: one-argument constructors, a unary numeral; 1: an environment with `depth` bindings;
    // 2: closures each capturing an environment that holds the previous closure) and then drops it. With a recursive destructor this overflows any stack long before a million levels;
    // the trampoline in detail::run_release makes it a loop. Returns the depth, so the call cannot be optimised away.
    uint64_t stress_value_release(uint32_t depth, int kind) {
        auto g = activate();
        if (kind == 0) { itp::Value v{itp::VCtor{0, 0, std::vector<itp::Value>{}}}; for (uint32_t i = 0; i < depth; ++i) v = itp::Value{itp::VCtor{0, 1, itp::SharedSeq<itp::Value>::of(std::move(v))}}; }
        else if (kind == 1) { itp::Env e; for (uint32_t i = 0; i < depth; ++i) e = e.extend(itp::Value{itp::VType{0}}); }
        else if (kind == 2) { itp::Value v{itp::VType{0}}; for (uint32_t i = 0; i < depth; ++i) { itp::Env e; e = e.extend(std::move(v)); v = itp::Value{itp::Closure{e, itp::NodeId{}, itp::NodeId{}}}; } }
        else throw ITPError("stress_value_release: kind must be 0, 1 or 2");
        return depth;
    }
    itp::NodeId lift(itp::NodeId t, uint32_t d, uint32_t cutoff) {
        auto g = activate(); require_node(t, "lift"); require_small(d, "lift amount"); require_small(cutoff, "lift cutoff");
        if (static_cast<uint64_t>(itp::loose_range(t)) + d > itp::kMaxUniverseLevel) throw ITPError("lift: the result would have a bound variable index beyond the supported range");
        return itp::lift(t, d, cutoff); }
    itp::NodeId instantiate(itp::NodeId t, const std::vector<itp::NodeId>& vals) { auto g = activate(); require_node(t, "instantiate"); for (auto v : vals) require_node(v, "instantiate value"); return itp::instantiate(t, vals); }
    itp::NodeId abstract_fvars(itp::NodeId t, const std::vector<uint32_t>& fvars) { auto g = activate(); require_node(t, "abstract_fvars"); return itp::abstract_fvars(t, fvars); }
    // Quotients: declares Quot, Quot.mk, Quot.lift, Quot.ind and Quot.sound with kernel-built types (see itp::declare_quot); eq_gid must be the standard equality family.
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> declare_quot(uint32_t eq_gid) {
        auto g = activate(); require_global(eq_gid, "declare_quot"); auto r = itp::declare_quot(eq_gid);
        if (!r) throw ITPError(r.error().message);
        return {r->type, r->mk, r->lift, r->ind, r->sound};
    }
    int quot_role(uint32_t gid) { auto g = activate(); return static_cast<int>(itp::ctx().globals.defs.at(gid).quot_role); }   // 0 none, 1 Quot, 2 mk, 3 lift, 4 ind, 5 sound
    uint32_t num_globals() { auto g = activate(); return static_cast<uint32_t>(itp::ctx().globals.defs.size()); }
    std::vector<std::tuple<uint32_t, std::string, itp::NodeId, bool>> list_globals() {
        auto g = activate(); std::vector<std::tuple<uint32_t, std::string, itp::NodeId, bool>> out;
        const auto& defs = itp::ctx().globals.defs;
        for (uint32_t i = 0; i < defs.size(); ++i) out.emplace_back(i, defs[i].name, defs[i].type, defs[i].value.has_value());
        return out;
    }
    // read-only view of a global for the independent checker: (name, type, value or None, ctor family or -1, ctor index or -1, constructors of this family in order, declared recursive)
    std::tuple<std::string, itp::NodeId, std::optional<itp::NodeId>, int, int, std::vector<uint32_t>, bool> global_info(uint32_t gid) {
        auto g = activate(); const auto& d = itp::ctx().globals.defs.at(gid);
        std::vector<uint32_t> ctors; auto it = itp::ctx().globals.ctors_by_family.find(gid); if (it != itp::ctx().globals.ctors_by_family.end()) ctors = it->second;
        return {d.name, d.type, d.value, d.ctor_info ? static_cast<int>(d.ctor_info->ind_type_global_id) : -1, d.ctor_info ? static_cast<int>(d.ctor_info->ctor_idx) : -1, ctors, d.recursive_info.has_value()};
    }
    // Was this global finished as an inductive family (possibly with no constructors)? Match/cases/induct exist only for those; an opaque type constant (an axiom, Quot) is not one.
    bool is_inductive(uint32_t gid) { auto g = activate(); return itp::ctx().globals.defs.at(gid).is_inductive; }
    // Is the global still being defined? 0 = no (finished, or an ordinary global), 1 = a reserved inductive type, 2 = a reserved recursive function. Nothing finished may mention a reserved global.
    int reserved_kind(uint32_t gid) { auto g = activate(); require_global(gid, "reserved_kind"); return static_cast<int>(itp::ctx().globals.defs[gid].reserved); }
    // The justification recorded when a recursive function was finished: (the inductive type it recurses over, the decreasing argument's position), or None for any other global.
    std::optional<std::pair<uint32_t, uint32_t>> recursive_info(uint32_t gid) {
        auto g = activate(); require_global(gid, "recursive_info"); const auto& ri = itp::ctx().globals.defs[gid].recursive_info;
        if (!ri) return std::nullopt;
        return std::make_pair(ri->ind_type_global_id, ri->decreasing_arg_level);
    }
    // Raw dump with explicit variable kinds: b<i> = bound index, f<l> = context level.
    std::string dump_raw(itp::NodeId n) {
        auto g = activate(); require_node(n, "dump_raw");
        std::string out;   // appended to in place: building each level's string from its children's copied the whole text at every level (quadratic on deep terms)
        std::function<void(itp::NodeId)> go = [&](itp::NodeId t) {
            ITP_STACK_CHECK();
            itp::inspect(t,
                [&](itp::Var v) { out += v.bound ? "b" : "f"; out += std::to_string(v.n); },
                [&](itp::App a) { out += "("; go(a.fn_id); out += " "; go(a.arg_id); out += ")"; },
                [&](itp::Lambda l) { out += "(lam "; go(l.type_id); out += ". "; go(l.body_id); out += ")"; },
                [&](itp::Pi p) { out += "(pi "; go(p.dom_id); out += ". "; go(p.cod_id); out += ")"; },
                [&](itp::Let l) { out += "(let "; go(l.type_id); out += " := "; go(l.value_id); out += " in "; go(l.body_id); out += ")"; },
                [&](itp::Type ty) { out += "T" + std::to_string(ty.universe_level); },
                [&](itp::Const c) { out += "C" + std::to_string(c.global_id); },
                [&](itp::Meta m) { out += "?" + std::to_string(m.meta_id); },
                [&](itp::Match m) {
                    out += "(match "; go(m.scrutinee_id); out += " ["; go(m.motive_id); out += "]";
                    for (auto b : *m.branches) { out += " | "; go(b); }
                    out += ")";
                },
                [&](itp::Fix f) { out += "(fix<" + std::to_string(f.decreasing_arg) + "> "; go(f.fn_type); out += " := "; go(f.body); out += ")"; },
                [&](auto) { out += "<other>"; });
        };
        go(n);
        return out;
    }
    // The structure of ONE node, for Python code that has to read a term (tactics, the ring normaliser, feature extraction). The result is a tuple whose first entry names the kind and whose other
    // entries are children (handles), numbers or a list:  ('type', level)  ('bvar', index)  ('fvar', level)  ('app', f, a)  ('lam', type, body)  ('pi', dom, cod)  ('let', type, value, body)
    // ('const', gid)  ('match', scrutinee, motive, [branches])  ('fix', fn_type, decreasing_arg, body)  ('meta', id)  ('tuniv', index).  O(1) for every kind except match (its branch list).
    // Rebuilding a node from its view with the mk_* functions gives back the same handle (terms are hash-consed): the round trip is the gate's test.
    nb::object view(itp::NodeId n) {
        auto g = activate(); require_node(n, "view");
        return itp::inspect(n,
            [&](itp::Var v) -> nb::object { return nb::make_tuple(v.bound ? "bvar" : "fvar", v.n); },
            [&](itp::App a) -> nb::object { return nb::make_tuple("app", a.fn_id, a.arg_id); },
            [&](itp::Lambda l) -> nb::object { return nb::make_tuple("lam", l.type_id, l.body_id); },
            [&](itp::Pi p) -> nb::object { return nb::make_tuple("pi", p.dom_id, p.cod_id); },
            [&](itp::Let l) -> nb::object { return nb::make_tuple("let", l.type_id, l.value_id, l.body_id); },
            [&](itp::Type t) -> nb::object { return nb::make_tuple("type", t.universe_level); },
            [&](itp::Const c) -> nb::object { return nb::make_tuple("const", c.global_id); },
            [&](itp::Meta m) -> nb::object { return nb::make_tuple("meta", m.meta_id); },
            [&](itp::Match m) -> nb::object { return nb::make_tuple("match", m.scrutinee_id, m.motive_id, nb::cast(std::vector<itp::NodeId>(m.branches->begin(), m.branches->end()))); },
            [&](itp::Fix f) -> nb::object { return nb::make_tuple("fix", f.fn_type, f.decreasing_arg, f.body); },
            [&](itp::TypeUniv t) -> nb::object { return nb::make_tuple("tuniv", t.param_index); });
    }
};

inline itp::ContextGuard ProofStateHandle::activate() { return kernel->activate(); }

// The kernel recurses on the depth of a term. Linux grows the main thread's stack on demand up to its soft limit, which is only 8 MiB by
// default; raise it (as far as the hard limit allows) so deep but legitimate terms -- a unary numeral tens of thousands deep -- have room.
// Threads created from Python get threading.stack_size(); the guard in the kernel turns an overflow into a RuntimeError either way.
static uint64_t raise_stack_limit(uint64_t want) {
    rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) != 0) return 0;
    rlim_t target = static_cast<rlim_t>(want);
    if (rl.rlim_max != RLIM_INFINITY && rl.rlim_max < target) target = rl.rlim_max;
    if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < target) { rl.rlim_cur = target; setrlimit(RLIMIT_STACK, &rl); }
    itp::refresh_stack_floor();
    getrlimit(RLIMIT_STACK, &rl);
    return rl.rlim_cur == RLIM_INFINITY ? ~uint64_t{0} : static_cast<uint64_t>(rl.rlim_cur);
}

NB_MODULE(itp_env_lnl, m) {
    nb::exception<ITPError>(m, "ITPError");
    raise_stack_limit(uint64_t{1} << 30);
    m.def("set_stack_limit", &raise_stack_limit, "bytes"_a, "Raise the soft stack limit of the process (main thread) to at least `bytes` if the hard limit allows; returns the limit now in force.");
    nb::class_<itp::NodeId>(m, "NodeId")
        .def("__eq__", [](const itp::NodeId& a, const itp::NodeId& b) { return a == b; })
        .def("__hash__", [](const itp::NodeId& n) { return std::hash<uint64_t>{}((static_cast<uint64_t>(static_cast<uint32_t>(n.tag)) << 32) | n.index); })
        .def_prop_ro("tag", [](const itp::NodeId& n) { return static_cast<int>(n.tag); })
        .def_prop_ro("index", [](const itp::NodeId& n) { return n.index; });
    nb::class_<itp::CtorSpec>(m, "CtorSpec")
        .def(nb::init<>())
        .def("__init__", [](itp::CtorSpec* self, const std::string& name, std::vector<itp::NodeId> arg_types, std::vector<itp::NodeId> result_indices) {
            new (self) itp::CtorSpec{name, std::move(arg_types), std::move(result_indices)};
        }, "name"_a, "arg_types"_a, "result_indices"_a = std::vector<itp::NodeId>{});
    nb::class_<itp::InductiveHandle>(m, "InductiveHandle")
        .def_ro("type_global_id", &itp::InductiveHandle::type_global_id)
        .def_ro("ctor_global_ids", &itp::InductiveHandle::ctor_global_ids);
    nb::class_<itp::DiscriminationTree>(m, "Library");
    nb::class_<itp::ExecutionContext::Checkpoint>(m, "Checkpoint");
    nb::class_<ProofStateHandle>(m, "ProofState")
        .def("intro", &ProofStateHandle::intro_, "name"_a = "", "target_meta_id"_a = nb::none())
        .def("apply", &ProofStateHandle::apply_, "term"_a, "target_meta_id"_a = nb::none())
        .def("cases", &ProofStateHandle::cases_, "hyp_level"_a, "target_meta_id"_a = nb::none(), "generalizing"_a = std::vector<uint32_t>{})
        .def("induct", &ProofStateHandle::induct_, "hyp_level"_a, "target_meta_id"_a = nb::none(), "generalizing"_a = std::vector<uint32_t>{}, "eq_gid"_a = nb::none(), "refl_gid"_a = nb::none())
        .def("exact", &ProofStateHandle::exact_, "term"_a, "target_meta_id"_a = nb::none())
        .def("rewrite", &ProofStateHandle::rewrite_, "eq_term"_a, "x"_a, "y"_a, "transport_gid"_a, "symm_gid"_a, "target_meta_id"_a = nb::none())
        .def("is_complete", &ProofStateHandle::is_complete)
        .def("extract", &ProofStateHandle::extract)
        .def("num_goals", &ProofStateHandle::num_goals)
        .def("num_pending", &ProofStateHandle::num_pending)
        .def("goal_meta_id", &ProofStateHandle::goal_meta_id, "i"_a)
        .def("goal_target", &ProofStateHandle::goal_target, "i"_a)
        .def("goal_num_hyps", &ProofStateHandle::goal_num_hyps, "i"_a)
        .def("goal_hyp_name", &ProofStateHandle::goal_hyp_name, "i"_a, "h"_a)
        .def("goal_hyp_type", &ProofStateHandle::goal_hyp_type, "i"_a, "h"_a);
    nb::class_<KernelWrapper>(m, "Kernel")
        .def(nb::init<>())
        .def("start_proof", &KernelWrapper::start_proof, "goal_type"_a, nb::keep_alive<0, 1>())
        .def("checkpoint", &KernelWrapper::checkpoint)
        .def("rewind", &KernelWrapper::rewind, "checkpoint"_a)
        .def("fork_single_goal", &KernelWrapper::fork_single_goal, "source"_a, "target_meta_id"_a, nb::keep_alive<0, 1>())
        .def("materialize_from_node", &KernelWrapper::materialize_from_node, "target"_a, "hyp_types"_a, "hyp_names"_a, nb::keep_alive<0, 1>())
        .def("loose_range_of", &KernelWrapper::loose_range_of, "term"_a)
        .def("set_option", &KernelWrapper::set_option, "name"_a, "value"_a)
        .def("get_option", &KernelWrapper::get_option, "name"_a)
        .def("param_slots", &KernelWrapper::param_slots, "family_gid"_a)
        .def("index_library", &KernelWrapper::index_library)
        .def("apply_candidates", &KernelWrapper::apply_candidates, "ps"_a, "library"_a, "target_meta_id"_a = nb::none())
        .def("mk_type", &KernelWrapper::mk_type, "level"_a)
        .def("mk_bvar", &KernelWrapper::mk_bvar, "index"_a)
        .def("mk_fvar", &KernelWrapper::mk_fvar, "level"_a)
        .def("mk_app", &KernelWrapper::mk_app, "f"_a, "a"_a)
        .def("mk_lambda", &KernelWrapper::mk_lambda, "type"_a, "body"_a)
        .def("mk_pi", &KernelWrapper::mk_pi, "dom"_a, "cod"_a)
        .def("mk_let", &KernelWrapper::mk_let, "type"_a, "value"_a, "body"_a)
        .def("mk_const", &KernelWrapper::mk_const, "global_id"_a)
        .def("mk_match", &KernelWrapper::mk_match, "scrutinee"_a, "motive"_a, "branches"_a)
        .def("mk_fix", &KernelWrapper::mk_fix, "fn_type"_a, "decreasing_arg"_a, "body"_a)
        .def("reserve_inductive_type", &KernelWrapper::reserve_inductive_type, "name"_a, "sort"_a)
        .def("finish_inductive", &KernelWrapper::finish_inductive, "type_global_id"_a, "ctors"_a)
        .def("reserve_recursive_function", &KernelWrapper::reserve_recursive_function, "name"_a, "type"_a)
        .def("finish_recursive_function", &KernelWrapper::finish_recursive_function, "gid"_a, "ind_gid"_a, "decreasing_arg_level"_a, "body"_a)
        .def("declare_unchecked", &KernelWrapper::declare_unchecked, "type"_a, "value"_a, "name"_a)
        .def("finish_inductive_unchecked", &KernelWrapper::finish_inductive_unchecked, "type_global_id"_a, "ctors"_a)
        .def("declare_checked", &KernelWrapper::declare_checked, "type"_a, "value"_a, "name"_a)
        .def("finish_recursive_function_checked", &KernelWrapper::finish_recursive_function_checked, "gid"_a, "ind_gid"_a, "decreasing_arg_level"_a, "body"_a)
        .def("infer_open", &KernelWrapper::infer_open, "term"_a, "hyp_types"_a)
        .def("check_open", &KernelWrapper::check_open, "term"_a, "expected"_a, "hyp_types"_a)
        .def("check_partial", &KernelWrapper::check_partial, "term"_a, "expected"_a, "hyp_types"_a)
        .def("close_telescope", &KernelWrapper::close_telescope, "kind"_a, "doms"_a, "body"_a, "base"_a)
        .def("fresh_meta", &KernelWrapper::fresh_meta, "type"_a, "scope"_a)
        .def("mk_meta", &KernelWrapper::mk_meta, "id"_a)
        .def("meta_is_solved", &KernelWrapper::meta_is_solved, "id"_a)
        .def("meta_solution_raw", &KernelWrapper::meta_solution_raw, "id"_a)
        .def("unify_open", &KernelWrapper::unify_open, "a"_a, "b"_a, "hyp_types"_a)
        .def("unify_drain_open", &KernelWrapper::unify_drain_open, "a"_a, "b"_a, "hyp_types"_a)
        .def("normalize_open", &KernelWrapper::normalize_open, "term"_a, "n0"_a)
        .def("conv_open", &KernelWrapper::conv_open, "a"_a, "b"_a, "n0"_a)
        .def("loose_range", &KernelWrapper::loose_range, "term"_a)
        .def("stress_value_release", &KernelWrapper::stress_value_release, "depth"_a, "kind"_a)
        .def("lift", &KernelWrapper::lift, "term"_a, "d"_a, "cutoff"_a = 0)
        .def("instantiate", &KernelWrapper::instantiate, "term"_a, "vals"_a)
        .def("abstract_fvars", &KernelWrapper::abstract_fvars, "term"_a, "levels"_a)
        .def("declare_quot", &KernelWrapper::declare_quot, "eq_gid"_a)
        .def("quot_role", &KernelWrapper::quot_role, "gid"_a)
        .def("is_inductive", &KernelWrapper::is_inductive, "gid"_a)
        .def("reserved_kind", &KernelWrapper::reserved_kind, "gid"_a)
        .def("recursive_info", &KernelWrapper::recursive_info, "gid"_a)
        .def("num_globals", &KernelWrapper::num_globals)
        .def("list_globals", &KernelWrapper::list_globals)
        .def("global_info", &KernelWrapper::global_info, "gid"_a)
        .def("dump_raw", &KernelWrapper::dump_raw, "term"_a)
        .def("view", &KernelWrapper::view, "term"_a);
}
