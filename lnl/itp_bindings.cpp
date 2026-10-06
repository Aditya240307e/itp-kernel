// Bindings for the hybrid-variable kernel (refactor spike S1): terms, NbE, declarations, a small typing fragment and the
// diagnostics the differential harness needs. Tactics, unification and the full typing core are ported in S2-S4.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <functional>
#include "itp_kernel.hpp"

namespace nb = nanobind;
using namespace nb::literals;

struct ITPError : std::runtime_error { using std::runtime_error::runtime_error; };


struct KernelWrapper;
inline void require_scoped(itp::NodeId t, uint32_t nfree, const char* what) {
    if (!itp::detail::well_scoped(t, nfree))
        throw ITPError(std::string("ill-scoped term (") + what + "): it has an unbound bound variable, or a context variable beyond the " + std::to_string(nfree) + " in scope");
}
inline void require_scoped_telescope(const std::vector<itp::NodeId>& types, const char* what) { for (uint32_t j = 0; j < types.size(); ++j) require_scoped(types[j], j, what); }
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
    void induct_(uint32_t hyp_level, std::optional<uint32_t> tm, const std::vector<uint32_t>& gen, std::optional<uint32_t> eq_gid, std::optional<uint32_t> refl_gid) { auto g = activate(); raise_if_error(itp::induct(*ps, hyp_level, tm, gen, eq_gid, refl_gid)); }
    void exact_(itp::NodeId term, std::optional<uint32_t> tm) { auto g = activate(); require_scoped(term, scope_of(tm), "exact"); raise_if_error(itp::exact(*ps, term, tm)); }
    void rewrite_(itp::NodeId eq_term, itp::NodeId x, itp::NodeId y, uint32_t transport_gid, uint32_t symm_gid, std::optional<uint32_t> tm) {
        auto g = activate(); uint32_t sc = scope_of(tm); require_scoped(eq_term, sc, "rewrite eq_term"); require_scoped(x, sc, "rewrite x"); require_scoped(y, sc, "rewrite y"); raise_if_error(itp::rewrite_eq(*ps, eq_term, x, y, transport_gid, symm_gid, tm));
    }
    bool is_complete() { auto g = activate(); return itp::proof_complete(*ps); }
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
    uint32_t loose_range_of(itp::NodeId t) { auto g = activate(); return itp::loose_range(t); }
    void set_option(const std::string& name, bool value) { auto g = activate(); if (name == "parameters") itp::ctx().param_mode = value; else if (name == "prop_elim") itp::ctx().prop_mode = value; else throw ITPError("unknown option: " + name); }
    bool get_option(const std::string& name) { auto g = activate(); if (name == "parameters") return itp::ctx().param_mode; if (name == "prop_elim") return itp::ctx().prop_mode; throw ITPError("unknown option: " + name); }
    std::vector<int> param_slots(uint32_t family_gid) { auto g = activate(); std::vector<int> out; for (bool b : itp::family_param_slots(family_gid)) out.push_back(b ? 1 : 0); return out; }
    itp::DiscriminationTree index_library() { auto g = activate(); return itp::index_library(); }
    std::vector<itp::NodeId> apply_candidates(ProofStateHandle& psh, const itp::DiscriminationTree& lib, std::optional<uint32_t> target_meta_id) {
        auto g = activate(); return itp::apply_candidates(*psh.ps, lib, target_meta_id);
    }
    ProofStateHandle start_proof(itp::NodeId goal_type) {
        auto g = activate(); require_scoped(goal_type, 0, "goal type"); ProofStateHandle h; h.kernel = this;
        *h.ps = itp::start_proof(goal_type); h.root_meta_id = h.ps->goals[0].meta_id; return h;
    }
    itp::NodeId mk_type(uint32_t l) { auto g = activate(); return itp::mk_type(l); }
    itp::NodeId mk_bvar(uint32_t i) { auto g = activate(); return itp::mk_bvar(i); }
    itp::NodeId mk_fvar(uint32_t l) { auto g = activate(); return itp::mk_fvar(l); }
    itp::NodeId mk_app(itp::NodeId f, itp::NodeId a) { auto g = activate(); return itp::mk_app(f, a); }
    itp::NodeId mk_lambda(itp::NodeId t, itp::NodeId b) { auto g = activate(); return itp::mk_lambda(t, b); }
    itp::NodeId mk_pi(itp::NodeId d, itp::NodeId c) { auto g = activate(); return itp::mk_pi(d, c); }
    itp::NodeId mk_let(itp::NodeId t, itp::NodeId v, itp::NodeId b) { auto g = activate(); return itp::mk_let(t, v, b); }
    itp::NodeId mk_const(uint32_t gid) { auto g = activate(); return itp::mk_const(gid); }
    itp::NodeId mk_match(itp::NodeId s, itp::NodeId m, std::vector<itp::NodeId> bs) { auto g = activate(); return itp::mk_match(s, m, std::move(bs)); }
    itp::NodeId mk_fix(itp::NodeId fn_type, uint32_t dec, itp::NodeId body) { auto g = activate(); return itp::mk_fix(fn_type, dec, body); }

    uint32_t reserve_inductive_type(const std::string& name, itp::NodeId sort) { auto g = activate(); require_scoped(sort, 0, "inductive sort"); return itp::reserve_inductive_type(name, sort); }
    itp::InductiveHandle finish_inductive(uint32_t tid, std::vector<itp::CtorSpec> ctors) { auto g = activate();
        for (auto& c : ctors) { require_scoped_telescope(c.arg_types, "constructor field type"); for (auto r : c.result_indices) require_scoped(r, static_cast<uint32_t>(c.arg_types.size()), "constructor result index"); }
        return itp::finish_inductive(tid, std::move(ctors)); }
    uint32_t reserve_recursive_function(const std::string& name, itp::NodeId type) { auto g = activate(); return itp::reserve_recursive_function(name, type); }
    void finish_recursive_function(uint32_t gid, uint32_t ind_gid, uint32_t dec, itp::NodeId body) { auto g = activate(); require_scoped(body, 0, "recursive function body"); itp::finish_recursive_function(gid, ind_gid, dec, body); }
    uint32_t declare_unchecked(itp::NodeId type, std::optional<itp::NodeId> value, const std::string& name) { auto g = activate(); require_scoped(type, 0, "declared type"); if (value) require_scoped(*value, 0, "declared value"); return itp::declare_unchecked(type, value, name); }

    // ---- S2: checked declarations and the full typing core
    itp::InductiveHandle finish_inductive_unchecked(uint32_t tid, std::vector<itp::CtorSpec> ctors) { auto g = activate();
        for (auto& c : ctors) { require_scoped_telescope(c.arg_types, "constructor field type"); for (auto r : c.result_indices) require_scoped(r, static_cast<uint32_t>(c.arg_types.size()), "constructor result index"); }
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
        auto g = activate(); itp::ElabCtx ec; itp::ElabState es;
        for (auto h : hyp_types) ec = ec.extend(h);
        auto r = itp::infer(t, ec, es);
        if (r) return {true, nb::cast(*r)};
        return {false, nb::cast(r.error().message)};
    }
    std::pair<bool, std::string> check_open(itp::NodeId t, itp::NodeId expected, const std::vector<itp::NodeId>& hyp_types) {
        auto g = activate(); require_scoped_telescope(hyp_types, "hypothesis type"); require_scoped(t, hyp_types.size(), "term"); require_scoped(expected, hyp_types.size(), "expected type"); itp::ElabCtx ec; itp::ElabState es;
        for (auto h : hyp_types) ec = ec.extend(h);
        auto r = itp::check(t, expected, ec, es);
        if (r) return {true, "OK"};
        return {false, r.error().message};
    }
    // ---- S3: direct access to metas and unification (used by the unifier unit tests)
    uint32_t fresh_meta(itp::NodeId type, uint32_t scope) { auto g = activate(); return itp::ctx().meta_ctx.fresh(type, scope); }
    itp::NodeId mk_meta(uint32_t id) { auto g = activate(); return itp::mk_meta(id); }
    bool meta_is_solved(uint32_t id) { auto g = activate(); return itp::ctx().meta_ctx.is_solved(id); }
    std::string meta_solution_raw(uint32_t id) {
        auto g = activate(); auto sol = itp::ctx().meta_ctx[id].solution;
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
        auto g = activate();
        return kind == "lam" ? itp::close_lambda_telescope(doms, body, base) : itp::close_pi_telescope(doms, body, base);
    }
    itp::Env fresh_env(uint32_t n0) { itp::Env env; for (uint32_t i = 0; i < n0; ++i) env = env.extend(itp::fresh_neutral(i)); return env; }
    itp::NodeId normalize_open(itp::NodeId t, uint32_t n0) { auto g = activate(); require_scoped(t, n0, "term"); return itp::normalize(t, fresh_env(n0)); }
    bool conv_open(itp::NodeId a, itp::NodeId b, uint32_t n0) {
        auto g = activate(); itp::Env env = fresh_env(n0);
        return itp::conv_value(itp::eval(a, env), itp::eval(b, env), n0);
    }
    std::pair<bool, nb::object> tc_infer_open(itp::NodeId t, const std::vector<itp::NodeId>& hyp_types) {
        auto g = activate(); itp::ElabCtx ec;
        for (auto h : hyp_types) ec = ec.extend(h);
        auto r = itp::tc_infer(t, ec);
        if (r) return {true, nb::cast(*r)};
        return {false, nb::cast(r.error())};
    }
    uint32_t loose_range(itp::NodeId t) { auto g = activate(); return itp::loose_range(t); }
    itp::NodeId lift(itp::NodeId t, uint32_t d, uint32_t cutoff) { auto g = activate(); return itp::lift(t, d, cutoff); }
    itp::NodeId instantiate(itp::NodeId t, const std::vector<itp::NodeId>& vals) { auto g = activate(); return itp::instantiate(t, vals); }
    itp::NodeId abstract_fvars(itp::NodeId t, const std::vector<uint32_t>& fvars) { auto g = activate(); return itp::abstract_fvars(t, fvars); }
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
    // Raw dump with explicit variable kinds: b<i> = bound index, f<l> = context level.
    std::string dump_raw(itp::NodeId n) {
        auto g = activate();
        std::function<std::string(itp::NodeId)> go = [&](itp::NodeId t) -> std::string {
            return itp::inspect(t,
                [&](itp::Var v) { return std::string(v.bound ? "b" : "f") + std::to_string(v.n); },
                [&](itp::App a) { return "(" + go(a.fn_id) + " " + go(a.arg_id) + ")"; },
                [&](itp::Lambda l) { return "(lam " + go(l.type_id) + ". " + go(l.body_id) + ")"; },
                [&](itp::Pi p) { return "(pi " + go(p.dom_id) + ". " + go(p.cod_id) + ")"; },
                [&](itp::Let l) { return "(let " + go(l.type_id) + " := " + go(l.value_id) + " in " + go(l.body_id) + ")"; },
                [&](itp::Type ty) { return "T" + std::to_string(ty.universe_level); },
                [&](itp::Const c) { return "C" + std::to_string(c.global_id); },
                [&](itp::Meta m) { return "?" + std::to_string(m.meta_id); },
                [&](itp::Match m) {
                    std::string r = "(match " + go(m.scrutinee_id) + " [" + go(m.motive_id) + "]";
                    for (auto b : *m.branches) r += " | " + go(b);
                    return r + ")";
                },
                [&](itp::Fix f) { return "(fix<" + std::to_string(f.decreasing_arg) + "> " + go(f.fn_type) + " := " + go(f.body) + ")"; },
                [&](auto) { return std::string("<other>"); });
        };
        return go(n);
    }
};

inline itp::ContextGuard ProofStateHandle::activate() { return kernel->activate(); }

NB_MODULE(itp_env_lnl, m) {
    nb::exception<ITPError>(m, "ITPError");
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
        .def("close_telescope", &KernelWrapper::close_telescope, "kind"_a, "doms"_a, "body"_a, "base"_a)
        .def("fresh_meta", &KernelWrapper::fresh_meta, "type"_a, "scope"_a)
        .def("mk_meta", &KernelWrapper::mk_meta, "id"_a)
        .def("meta_is_solved", &KernelWrapper::meta_is_solved, "id"_a)
        .def("meta_solution_raw", &KernelWrapper::meta_solution_raw, "id"_a)
        .def("unify_open", &KernelWrapper::unify_open, "a"_a, "b"_a, "hyp_types"_a)
        .def("unify_drain_open", &KernelWrapper::unify_drain_open, "a"_a, "b"_a, "hyp_types"_a)
        .def("normalize_open", &KernelWrapper::normalize_open, "term"_a, "n0"_a)
        .def("conv_open", &KernelWrapper::conv_open, "a"_a, "b"_a, "n0"_a)
        .def("tc_infer_open", &KernelWrapper::tc_infer_open, "term"_a, "hyp_types"_a)
        .def("loose_range", &KernelWrapper::loose_range, "term"_a)
        .def("lift", &KernelWrapper::lift, "term"_a, "d"_a, "cutoff"_a = 0)
        .def("instantiate", &KernelWrapper::instantiate, "term"_a, "vals"_a)
        .def("abstract_fvars", &KernelWrapper::abstract_fvars, "term"_a, "levels"_a)
        .def("num_globals", &KernelWrapper::num_globals)
        .def("list_globals", &KernelWrapper::list_globals)
        .def("global_info", &KernelWrapper::global_info, "gid"_a)
        .def("dump_raw", &KernelWrapper::dump_raw, "term"_a);
}
