// itp_bindings.cpp
//
// nanobind surface for the ITP kernel. Built incrementally, tested at
// each step -- this first increment covers just enough to construct
// terms and declare inductive types from Python (the bootstrap-prelude
// primitives), as a foundation to build the ProofState/tactic surface
// on top of next.
//
// LIFETIME DESIGN NOTE: itp::ContextGuard is RAII, designed for strict,
// C++-scope-nested usage (construct on scope entry, destruct on scope
// exit, restoring whatever was active before). A long-lived Python
// object doesn't fit that pattern -- Python's GC order is not
// guaranteed, so if a SECOND Kernel were constructed while a first one
// is still alive, and the first gets garbage-collected first, its
// ContextGuard's destructor would reset active_context to ITS OWN
// "prev" (likely nullptr), breaking the second Kernel's still-active
// operations even though the second Kernel object is still alive and
// in use. Rather than hold one persistent guard for a Kernel's whole
// lifetime, every method that needs ctx() to be correctly set
// constructs a FRESH, call-scoped ContextGuard via activate() at the
// start of that call, letting it destruct naturally at the end of that
// SAME call. This is essentially free (ContextGuard's own constructor/
// destructor are two pointer assignments each) and makes correctness
// independent of how many Kernel objects exist or in what order they
// get destroyed.

#include <functional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
#include "itp_kernel.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// Dense, per-kind visited arrays (indexed directly by NodeId.index) --
// O(1), hash-free, matching how this kernel's own pools are already
// organized. Confirmed faster than a hash-map-based visited tracker by
// direct benchmark several turns ago (a hash-map first attempt was
// actually SLOWER than string pretty-printing; this dense-array version
// is 2.0-2.4x faster). Reset via a generation stamp rather than cleared
// per call, and held as a persistent KernelWrapper member so its own
// backing arrays are reused across observation() calls, not
// reallocated each time.
struct DenseVisited {
    std::array<std::vector<uint32_t>, 14> gen;
    std::array<std::vector<int32_t>, 14> slot;
    uint32_t current_gen = 0;
    void reset() { ++current_gen; }
    int32_t get(itp::NodeId id) {
        auto t = static_cast<size_t>(id.tag);
        if (gen[t].size() <= id.index) { gen[t].resize(id.index + 1, 0); slot[t].resize(id.index + 1, -1); }
        if (gen[t][id.index] != current_gen) return -1;
        return slot[t][id.index];
    }
    void set(itp::NodeId id, int32_t v) {
        auto t = static_cast<size_t>(id.tag);
        gen[t][id.index] = current_gen;
        slot[t][id.index] = v;
    }
};

// Heap-allocated, shared by all four returned ndarrays via one capsule
// owner -- deleted together when the LAST of the four numpy arrays
// referencing it is garbage collected.
struct DagDumpStorage {
    std::vector<uint32_t> tags;
    std::vector<uint32_t> scalar;
    std::vector<int32_t> child_a;
    std::vector<int32_t> child_b;
    std::vector<uint32_t> height; // 0 for a leaf, else 1 + max(children's heights): lets a consumer encode the DAG level by level
    std::vector<int32_t> roots; // populated by callers that dump more than one top-level term into the same storage (e.g. observation()'s own hyps+target) -- see observation()'s own comment
};

// A SyntheticCons node exists only in this dump format -- it has no
// corresponding itp::NodeId at all, and no entry in DenseVisited (there
// is no real NodeId to key on). It exists purely to pack a >2-ary node
// kind's extra fields into the strictly-binary (child_a, child_b) shape
// every other row already uses, rather than changing the array format
// itself. tag=14 is guaranteed not to collide with any real
// itp::NodeId::Tag value, since the kernel's own Term variant has
// exactly 14 members (Var, App, Lambda, Let, Pi, Type, Const, Match,
// Meta, TypeUniv, QuotType, QuotMk, QuotLift, Fix), occupying tags
// 0..13.
constexpr uint32_t kSyntheticConsTag = 14;

int32_t push_synthetic_cons(DagDumpStorage& out, int32_t head, int32_t rest) {
    int32_t my_index = static_cast<int32_t>(out.tags.size());
    out.tags.push_back(kSyntheticConsTag);
    out.scalar.push_back(0);
    out.child_a.push_back(head);
    out.child_b.push_back(rest);
    out.height.push_back(1 + std::max(out.height[head], out.height[rest])); // both already complete
    return my_index;
}

// Builds a right-folded cons chain over `items` (each already a
// dumped index: items[0] is reachable via exactly one cons from the
// returned head, items.back() terminates the chain directly with no
// wrapping cons, since it is itself an ordinary, already-dumped
// node). Requires at least one item.
int32_t build_cons_chain(DagDumpStorage& out, const std::vector<int32_t>& items) {
    int32_t chain = items[items.size() - 1];
    for (std::size_t idx = items.size() - 1; idx > 0; --idx) {
        chain = push_synthetic_cons(out, items[idx - 1], chain);
    }
    return chain;
}

int build_dag_dump(itp::NodeId root, DagDumpStorage& out, DenseVisited& visited) {
    int32_t existing = visited.get(root);
    if (existing >= 0) return existing;
    int32_t my_index = static_cast<int32_t>(out.tags.size());
    visited.set(root, my_index);
    out.tags.push_back(static_cast<uint32_t>(root.tag));
    out.scalar.push_back(0);
    out.child_a.push_back(-1);
    out.child_b.push_back(-1);
    out.height.push_back(0);
    itp::inspect(root,
        [&](itp::Var v) { out.scalar[my_index] = v.level; },
        [&](itp::Const c) { out.scalar[my_index] = c.global_id; },
        [&](itp::App a) {
            int32_t fn = build_dag_dump(a.fn_id, out, visited);
            int32_t arg = build_dag_dump(a.arg_id, out, visited);
            out.child_a[my_index] = fn; out.child_b[my_index] = arg;
        },
        [&](itp::Lambda l) {
            int32_t dom = build_dag_dump(l.type_id, out, visited);
            int32_t body = build_dag_dump(l.body_id, out, visited);
            out.child_a[my_index] = dom; out.child_b[my_index] = body;
        },
        [&](itp::Pi p) {
            int32_t dom = build_dag_dump(p.dom_id, out, visited);
            int32_t cod = build_dag_dump(p.cod_id, out, visited);
            out.child_a[my_index] = dom; out.child_b[my_index] = cod;
        },
        [&](itp::Type t) { out.scalar[my_index] = t.universe_level; },
        // Everything below is new: the previous version of this
        // function left these 8 of 14 kinds as a bare tag with no
        // scalar and no children at all (falling into a `[](auto) {}`
        // catch-all) -- confirmed, by direct design review, to make
        // Match in particular (central to any goal involving a
        // recursive function, since a stuck reduction reconstructs as
        // one) structurally invisible to anything built on this dump.
        [&](itp::Let l) {
            int32_t ty = build_dag_dump(l.type_id, out, visited);
            int32_t val = build_dag_dump(l.value_id, out, visited);
            int32_t body = build_dag_dump(l.body_id, out, visited);
            int32_t chain_head = build_cons_chain(out, {val, body});
            out.child_a[my_index] = ty; out.child_b[my_index] = chain_head;
        },
        [&](itp::Match m) {
            int32_t scrut = build_dag_dump(m.scrutinee_id, out, visited);
            int32_t motive = build_dag_dump(m.motive_id, out, visited);
            std::vector<int32_t> chain_items;
            chain_items.push_back(motive);
            for (itp::NodeId b : *m.branches) chain_items.push_back(build_dag_dump(b, out, visited));
            int32_t chain_head = build_cons_chain(out, chain_items);
            out.child_a[my_index] = scrut; out.child_b[my_index] = chain_head;
        },
        [&](itp::Meta m) { out.scalar[my_index] = m.meta_id; }, // leaf: an unsolved meta has no further syntactic structure to dump
        [&](itp::TypeUniv t) { out.scalar[my_index] = t.param_index; }, // leaf; per eval()'s own comment this should never survive into a finished term, but recorded faithfully if it ever does
        [&](itp::QuotType q) {
            int32_t rel = build_dag_dump(q.relation, out, visited);
            out.child_a[my_index] = rel; // child_b stays -1: QuotType has only one field
        },
        [&](itp::QuotMk q) {
            int32_t rel = build_dag_dump(q.relation, out, visited);
            int32_t elem = build_dag_dump(q.elem, out, visited);
            out.child_a[my_index] = rel; out.child_b[my_index] = elem;
        },
        [&](itp::QuotLift q) {
            int32_t rel = build_dag_dump(q.relation, out, visited);
            int32_t cod = build_dag_dump(q.codomain, out, visited);
            int32_t f = build_dag_dump(q.f, out, visited);
            int32_t resp = build_dag_dump(q.respects_proof, out, visited);
            int32_t qelem = build_dag_dump(q.quotient_elem, out, visited);
            int32_t chain_head = build_cons_chain(out, {cod, f, resp, qelem});
            out.child_a[my_index] = rel; out.child_b[my_index] = chain_head;
        },
        [&](itp::Fix f) {
            int32_t fnty = build_dag_dump(f.fn_type, out, visited);
            int32_t body = build_dag_dump(f.body, out, visited);
            out.scalar[my_index] = f.decreasing_arg;
            out.child_a[my_index] = fnty; out.child_b[my_index] = body;
        }
        // Deliberately no catch-all: itp::Term currently has exactly
        // these 14 alternatives, all listed explicitly above. Omitting
        // a catch-all means a future 15th kind fails to COMPILE here
        // rather than silently falling through with no scalar/children
        // the way the previous 8-kind gap did -- the same "no silent
        // gaps" discipline this session's own bug-hunting has already
        // paid for once.
    );

    // Every child has been fully built by now (this is a DAG: no cycles), so its height is final.
    {
        uint32_t h = 0;
        int32_t ca = out.child_a[my_index], cb = out.child_b[my_index];
        if (ca >= 0) h = std::max(h, out.height[ca] + 1);
        if (cb >= 0) h = std::max(h, out.height[cb] + 1);
        out.height[my_index] = h;
    }
    return my_index;
}


// Any ElabResult/TypeResult failure becomes a Python exception carrying
// the kernel's own TypeError.message -- the natural, Pythonic way to
// signal "this tactic call failed" to a search algorithm (try/except,
// not an error-code return it must remember to check).
struct ITPError : std::runtime_error {
    explicit ITPError(const std::string& msg) : std::runtime_error(msg) {}
};

template <typename Result>
void raise_if_error(const Result& r) {
    if (!r) throw ITPError(r.error().message);
}

// --- Term (de)serialization: the core of cross-process library
// persistence. Key points from the extensive design discussion:
//   - NodeId is a (tag, index) pair into ONE ExecutionContext's own
//     pools; meaningless elsewhere. Real serialization walks the term
//     structure and re-emits it via the same trusted mk_* primitives on
//     the receiving side, never transports raw NodeIds/pool memory.
//   - Only Const needs name-based resolution (global_id isn't portable,
//     a declared name is); Match/Fix reference no global directly.
//   - Meta/TypeUniv should never appear in a finished, checked global's
//     own type/value -- encountering either is treated as a genuine
//     error (a bug elsewhere caught here), not silently handled.
//   - Reconstruction MUST re-derive trust, not bypass it: every
//     deserialized global is re-type-checked (and, for recursive
//     functions, re-guardedness-checked) via the kernel's own existing,
//     trusted declaration paths -- never installed directly into
//     globals without going through them.

nb::object serialize_term(itp::NodeId term, itp::ExecutionContext& ctx) {
    nb::dict d;
    itp::inspect(term,
        [&](itp::Var v) { d["t"] = "var"; d["level"] = v.level; },
        [&](itp::App a) {
            d["t"] = "app";
            d["fn"] = serialize_term(a.fn_id, ctx);
            d["arg"] = serialize_term(a.arg_id, ctx);
        },
        [&](itp::Lambda l) {
            d["t"] = "lambda";
            d["type"] = serialize_term(l.type_id, ctx);
            d["body"] = serialize_term(l.body_id, ctx);
        },
        [&](itp::Let l) {
            d["t"] = "let";
            d["type"] = serialize_term(l.type_id, ctx);
            d["value"] = serialize_term(l.value_id, ctx);
            d["body"] = serialize_term(l.body_id, ctx);
        },
        [&](itp::Pi p) {
            d["t"] = "pi";
            d["dom"] = serialize_term(p.dom_id, ctx);
            d["cod"] = serialize_term(p.cod_id, ctx);
        },
        [&](itp::Type t) { d["t"] = "type"; d["level"] = t.universe_level; },
        [&](itp::Const c) {
            d["t"] = "const";
            d["name"] = ctx.globals[c.global_id].name;
        },
        [&](itp::Match m) {
            d["t"] = "match";
            d["scrutinee"] = serialize_term(m.scrutinee_id, ctx);
            d["motive"] = serialize_term(m.motive_id, ctx);
            nb::list branches;
            for (itp::NodeId b : *m.branches) branches.append(serialize_term(b, ctx));
            d["branches"] = branches;
        },
        [&](itp::Meta) {
            throw ITPError("serialize_term: encountered an unsolved Meta -- a finished, checked global's "
                            "own type/value should never contain one; this indicates a bug elsewhere "
                            "(an incomplete proof being exported), not something to serialize");
        },
        [&](itp::TypeUniv) {
            throw ITPError("serialize_term: encountered an uninstantiated TypeUniv -- universe "
                            "polymorphism is out of scope for this kernel's declared globals; this "
                            "indicates a bug elsewhere, not something to serialize");
        },
        [&](itp::QuotType q) { d["t"] = "quot_type"; d["relation"] = serialize_term(q.relation, ctx); },
        [&](itp::QuotMk q) {
            d["t"] = "quot_mk";
            d["relation"] = serialize_term(q.relation, ctx);
            d["elem"] = serialize_term(q.elem, ctx);
        },
        [&](itp::QuotLift q) {
            d["t"] = "quot_lift";
            d["relation"] = serialize_term(q.relation, ctx);
            d["codomain"] = serialize_term(q.codomain, ctx);
            d["f"] = serialize_term(q.f, ctx);
            d["respects_proof"] = serialize_term(q.respects_proof, ctx);
            d["quotient_elem"] = serialize_term(q.quotient_elem, ctx);
        },
        [&](itp::Fix f) {
            d["t"] = "fix";
            d["fn_type"] = serialize_term(f.fn_type, ctx);
            d["decreasing_arg"] = f.decreasing_arg;
            d["body"] = serialize_term(f.body, ctx);
        }
    );
    return d;
}

itp::NodeId deserialize_term(nb::handle obj, const std::unordered_map<std::string, uint32_t>& name_to_gid) {
    nb::dict d = nb::cast<nb::dict>(obj);
    std::string t = nb::cast<std::string>(d["t"]);
    if (t == "var") return itp::mk_var(nb::cast<uint32_t>(d["level"]));
    if (t == "app") return itp::mk_app(deserialize_term(d["fn"], name_to_gid), deserialize_term(d["arg"], name_to_gid));
    if (t == "lambda") return itp::mk_lambda(deserialize_term(d["type"], name_to_gid), deserialize_term(d["body"], name_to_gid));
    if (t == "let") return itp::mk_let(deserialize_term(d["type"], name_to_gid), deserialize_term(d["value"], name_to_gid), deserialize_term(d["body"], name_to_gid));
    if (t == "pi") return itp::mk_pi(deserialize_term(d["dom"], name_to_gid), deserialize_term(d["cod"], name_to_gid));
    if (t == "type") return itp::mk_type(nb::cast<uint32_t>(d["level"]));
    if (t == "const") {
        std::string name = nb::cast<std::string>(d["name"]);
        auto it = name_to_gid.find(name);
        if (it == name_to_gid.end())
            throw ITPError("deserialize_term: reference to '" + name + "', which has not been reconstructed yet "
                            "(or does not exist in this export) -- declaration events must be replayed in their "
                            "original order, since a later global may reference an earlier one but never the reverse");
        return itp::mk_const(it->second);
    }
    if (t == "match") {
        itp::NodeId scrutinee = deserialize_term(d["scrutinee"], name_to_gid);
        itp::NodeId motive = deserialize_term(d["motive"], name_to_gid);
        std::vector<itp::NodeId> branches;
        for (nb::handle b : nb::cast<nb::list>(d["branches"])) branches.push_back(deserialize_term(b, name_to_gid));
        return itp::mk_match(scrutinee, motive, branches);
    }
    if (t == "quot_type") return itp::mk_quot_type(deserialize_term(d["relation"], name_to_gid));
    if (t == "quot_mk") return itp::mk_quot_mk(deserialize_term(d["relation"], name_to_gid), deserialize_term(d["elem"], name_to_gid));
    if (t == "quot_lift") {
        return itp::mk_quot_lift(deserialize_term(d["relation"], name_to_gid), deserialize_term(d["codomain"], name_to_gid),
                                  deserialize_term(d["f"], name_to_gid), deserialize_term(d["respects_proof"], name_to_gid),
                                  deserialize_term(d["quotient_elem"], name_to_gid));
    }
    if (t == "fix") {
        return itp::mk_fix(deserialize_term(d["fn_type"], name_to_gid), nb::cast<uint32_t>(d["decreasing_arg"]),
                            deserialize_term(d["body"], name_to_gid));
    }
    throw ITPError("deserialize_term: unrecognized node tag '" + t + "' -- corrupted or incompatible export");
}

struct KernelWrapper;

// Wraps itp::ProofState. Holds a RAW pointer back to its owning
// KernelWrapper -- safe specifically because start_proof's own binding
// uses nb::keep_alive<0,1>() to keep the owning Kernel Python object
// (and therefore its underlying KernelWrapper) alive for at least as
// long as any ProofStateHandle derived from it, so this raw pointer
// can never outlive what it points to.
struct ProofStateHandle {
    std::unique_ptr<itp::ProofState> ps = std::make_unique<itp::ProofState>();
    KernelWrapper* kernel = nullptr;
    uint32_t root_meta_id = 0;

    itp::ContextGuard activate();

    void intro_(const std::string& name, std::optional<uint32_t> target_meta_id) {
        auto g = activate();
        raise_if_error(itp::intro(*ps, name, target_meta_id));
    }
    void cases_(uint32_t hyp_level, std::optional<uint32_t> target_meta_id) {
        auto g = activate();
        raise_if_error(itp::cases(*ps, hyp_level, target_meta_id));
    }
    void induct_(uint32_t hyp_level, std::optional<uint32_t> target_meta_id) {
        auto g = activate();
        raise_if_error(itp::induct(*ps, hyp_level, target_meta_id));
    }
    void apply_(itp::NodeId candidate, std::optional<uint32_t> target_meta_id) {
        auto g = activate();
        raise_if_error(itp::apply(*ps, candidate, target_meta_id));
    }
    void rewrite_(itp::NodeId eq_term, itp::NodeId x, itp::NodeId y, uint32_t transport_gid, uint32_t symm_gid, std::optional<uint32_t> target_meta_id) {
        auto g = activate();
        raise_if_error(itp::rewrite_eq(*ps, eq_term, x, y, transport_gid, symm_gid, target_meta_id));
    }

    bool is_complete() {
        auto g = activate();
        return itp::proof_complete(*ps);
    }
    itp::NodeId extract() {
        auto g = activate();
        return itp::extract_proof_term(root_meta_id);
    }

    std::size_t num_goals() const { return ps->goals.size(); }
    uint32_t goal_meta_id(std::size_t i) const { return ps->goals.at(i).meta_id; }
    itp::NodeId goal_target(std::size_t i) const { return ps->goals.at(i).target; }
    std::size_t goal_num_hyps(std::size_t i) const { return ps->goals.at(i).hyps.size(); }
    std::string goal_hyp_name(std::size_t i, std::size_t h) const { return ps->goals.at(i).hyps.at(h).name; }
    itp::NodeId goal_hyp_type(std::size_t i, std::size_t h) const { return ps->goals.at(i).hyps.at(h).type; }
};

struct KernelWrapper {
    std::unique_ptr<itp::ExecutionContext> exec_ctx = std::make_unique<itp::ExecutionContext>();
    DenseVisited dag_visited; // persistent, reused across observation() calls

    itp::ContextGuard activate() { return itp::ContextGuard(*exec_ctx); }

    itp::NodeId mk_type(uint32_t level) { auto g = activate(); return itp::mk_type(level); }
    itp::NodeId mk_prop() { auto g = activate(); return itp::mk_prop(); }
    itp::NodeId mk_pi(itp::NodeId dom, itp::NodeId cod) { auto g = activate(); return itp::mk_pi(dom, cod); }
    itp::NodeId mk_var(uint32_t level) { auto g = activate(); return itp::mk_var(level); }
    itp::NodeId mk_app(itp::NodeId fn, itp::NodeId arg) { auto g = activate(); return itp::mk_app(fn, arg); }
    itp::NodeId mk_lambda(itp::NodeId dom, itp::NodeId body) { auto g = activate(); return itp::mk_lambda(dom, body); }
    itp::NodeId mk_const(uint32_t global_id) { auto g = activate(); return itp::mk_const(global_id); }
    itp::NodeId mk_match(itp::NodeId scrutinee, itp::NodeId motive, std::vector<itp::NodeId> branches) {
        auto g = activate();
        return itp::mk_match(scrutinee, motive, branches);
    }

    uint32_t reserve_inductive_type(const std::string& name, itp::NodeId sort) {
        auto g = activate();
        return itp::reserve_inductive_type(name, sort);
    }
    itp::InductiveHandle finish_inductive(uint32_t type_global_id, std::vector<itp::CtorSpec> ctors,
                                           bool allow_unsafe_skip_positivity = false) {
        auto g = activate();
        return itp::finish_inductive(type_global_id, std::move(ctors), allow_unsafe_skip_positivity);
    }

    uint32_t reserve_recursive_function(const std::string& name, itp::NodeId type) {
        auto g = activate();
        return itp::reserve_recursive_function(name, type);
    }
    void finish_recursive_function_checked(uint32_t fn_gid, uint32_t ind_type_global_id,
                                            uint32_t decreasing_arg_level, itp::NodeId body,
                                            bool allow_unsafe_skip_guard_check = false) {
        auto g = activate();
        auto r = itp::finish_recursive_function_checked(fn_gid, ind_type_global_id, decreasing_arg_level, body,
                                                          allow_unsafe_skip_guard_check);
        raise_if_error(r);
    }

    std::string print_node(itp::NodeId n) {
        auto g = activate();
        return itp::print(n, nullptr);
    }

    // Diagnostic: prints a term with ABSOLUTE de Bruijn levels (#k for Var k), unlike print_node, which names
    // binders by printing depth and so cannot reveal a mis-levelled reference. Permanent: used to root-cause F3.
    std::string dump_raw(itp::NodeId n) {
        auto g = activate();
        return dump_raw_unlocked(n);
    }
    std::string dump_raw_unlocked(itp::NodeId n) {
        std::function<std::string(itp::NodeId)> go = [&](itp::NodeId t) -> std::string {
            return itp::inspect(t,
                [&](itp::Var v) { return "#" + std::to_string(v.level); },
                [&](itp::App a) { return "(" + go(a.fn_id) + " " + go(a.arg_id) + ")"; },
                [&](itp::Lambda l) { return "(lam " + go(l.type_id) + ". " + go(l.body_id) + ")"; },
                [&](itp::Pi p) { return "(pi " + go(p.dom_id) + ". " + go(p.cod_id) + ")"; },
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

    std::optional<itp::NodeId> meta_solution(uint32_t meta_id) {
        auto g = activate();
        const itp::MetaEntry& e = itp::ctx().meta_ctx[meta_id];
        return e.solution;
    }

    ProofStateHandle start_proof(itp::NodeId goal_type) {
        auto g = activate();
        ProofStateHandle h;
        *h.ps = itp::start_proof(goal_type);
        h.root_meta_id = h.ps->goals.at(0).meta_id;
        h.kernel = this;
        return h;
    }

    // Builds a FRESH, independent ProofState containing a COPY of only
    // the single goal matching target_meta_id, with its own fresh
    // ElabState -- NOT sharing es or goals with the source ProofState
    // at all. Exists specifically because a probed tactic call mutates
    // ProofState's own state (goals, es) directly, and rewind() cannot
    // undo that: rewind() only ever reverts ExecutionContext's own
    // state (NodePools, meta_ctx), and ProofState is a genuinely
    // separate object rewind() has no visibility into at all. Probing
    // a candidate tactic against the SAME, shared ProofState -- even
    // wrapped in checkpoint()/rewind() -- was confirmed, by direct
    // test, to permanently corrupt it (intro() replaces ps.goals[idx]
    // with a freshly-minted meta in place, which survives any later
    // rewind() untouched). Forking first, probing the fork, then
    // simply discarding it (letting Python's own GC reclaim it) keeps
    // the original completely untouched regardless of what the probe
    // does to its own copy.
    //
    // The copied goal's own meta_id is reused as-is (not freshly
    // minted): it still refers to the SAME ctx().meta_ctx entry the
    // original ProofState's own goal does, but that is safe precisely
    // because every probe is wrapped in checkpoint()/rewind() at the
    // call site -- any solve() a probe performs on that shared meta_id
    // gets correctly undone by rewind(), regardless of which
    // ProofState object triggered it.
    ProofStateHandle fork_single_goal(ProofStateHandle& source, uint32_t target_meta_id) {
        auto g = activate();
        auto idx_opt = itp::find_goal_index(*source.ps, target_meta_id);
        if (!idx_opt) throw ITPError("fork_single_goal: no goal with the given meta_id in the source ProofState");
        ProofStateHandle h;
        h.ps->goals.push_back(source.ps->goals[*idx_opt]); // copy: Goal itself (NodeIds + ElabCtx) is an ordinary value type
        h.ps->es = itp::ElabState{}; // deliberately fresh, NOT copied from source -- see comment above
        h.root_meta_id = target_meta_id;
        h.kernel = this;
        return h;
    }

    // Rebuilds a FRESH ProofState containing a single, freshly-minted
    // goal matching EXACTLY the given target and hyp_types/hyp_names --
    // not derived from any existing, live goal at all. Exists because
    // materialize() consuming a parent's single live_meta_id made it
    // impossible to later materialize a SECOND HyperEdge from the same
    // node (the underlying kernel goal was already gone, consumed by
    // the first materialization) -- confirmed by direct test to break
    // PUCT's own ability to revisit a parent and try a different
    // tactic, the same way ordinary MCTS can always reconsider a
    // different child. Since a node's own (target, hyp_types,
    // hyp_names) are already fully, immutably captured at discovery
    // time, there is no need to trace back through any parent's live
    // state at all -- each materialization independently rebuilds
    // its own fresh goal from that stored data, mirroring exactly how
    // ElabCtx::extend()/MetaContext::fresh() build a goal's context
    // and meta in start_proof and intro().
    ProofStateHandle materialize_from_node(itp::NodeId target, const std::vector<itp::NodeId>& hyp_types,
                                            const std::vector<std::string>& hyp_names) {
        auto g = activate();
        if (hyp_types.size() != hyp_names.size())
            throw ITPError("materialize_from_node: hyp_types and hyp_names must be the same length");
        itp::ElabCtx ec{};
        std::vector<itp::Hypothesis> hyps;
        for (size_t i = 0; i < hyp_types.size(); ++i) {
            ec = ec.extend(hyp_types[i]);
            hyps.push_back(itp::Hypothesis{hyp_names[i], hyp_types[i]});
        }
        uint32_t meta = itp::ctx().meta_ctx.fresh(target, ec.size());
        ProofStateHandle h;
        h.ps->goals.push_back(itp::Goal{meta, ec, std::move(hyps), target});
        h.root_meta_id = meta;
        h.kernel = this;
        return h;
    }

    itp::DiscriminationTree index_library() {
        auto g = activate();
        return itp::index_library();
    }

    std::vector<itp::NodeId> apply_candidates(ProofStateHandle& psh, const itp::DiscriminationTree& lib,
                                               std::optional<uint32_t> target_meta_id) {
        auto g = activate();
        return itp::apply_candidates(*psh.ps, lib, target_meta_id);
    }

    uint32_t declare_checked(itp::NodeId type, itp::NodeId value, std::string name) {
        auto g = activate();
        auto r = itp::declare_checked(type, value, std::move(name));
        raise_if_error(r);
        return *r;
    }

    // Walks globals.defs in declaration order (== dependency order: a
    // global can only ever reference an EARLIER one, never a later one,
    // by construction), grouping consecutive constructors sharing an
    // ind_type_global_id into one "family" event rather than emitting
    // each as an isolated atom -- reconstruction re-derives CtorInfo
    // (including recursive_mask) via finish_inductive itself, the same
    // trusted path that computed it originally, rather than trusting a
    // serialized copy of a precomputed cache.
    nb::list export_library() {
        auto g = activate();
        auto& defs = exec_ctx->globals.defs;
        nb::list events;
        std::size_t i = 0;
        while (i < defs.size()) {
            const auto& def = defs[i];
            if (def.ctor_info.has_value()) {
                throw ITPError("export_library: orphaned constructor at global " + std::to_string(i) +
                                " (" + def.name + ") -- its own family's type-former was not immediately "
                                "followed by it as expected; export logic assumption violated");
            }
            if (def.recursive_info.has_value()) {
                nb::dict e;
                e["kind"] = "recursive_function";
                e["name"] = def.name;
                e["type"] = serialize_term(def.type, *exec_ctx);
                e["ind_type_name"] = exec_ctx->globals[def.recursive_info->ind_type_global_id].name;
                e["decreasing_arg_level"] = def.recursive_info->decreasing_arg_level;
                e["body"] = serialize_term(*def.value, *exec_ctx);
                e["guard_check_was_skipped"] = def.recursive_info->guard_check_was_skipped;
                events.append(e);
                ++i;
            } else if (def.value.has_value()) {
                nb::dict e;
                e["kind"] = "definition";
                e["name"] = def.name;
                e["type"] = serialize_term(def.type, *exec_ctx);
                e["value"] = serialize_term(*def.value, *exec_ctx);
                events.append(e);
                ++i;
            } else if (i + 1 < defs.size() && defs[i + 1].ctor_info.has_value() &&
                       defs[i + 1].ctor_info->ind_type_global_id == i) {
                nb::dict e;
                e["kind"] = "family";
                e["name"] = def.name;
                e["sort"] = serialize_term(def.type, *exec_ctx);
                nb::list ctors;
                std::size_t j = i + 1;
                while (j < defs.size() && defs[j].ctor_info.has_value() && defs[j].ctor_info->ind_type_global_id == i) {
                    nb::dict c;
                    c["name"] = defs[j].name;
                    // Reconstruct arg_types/result_indices by re-deriving
                    // them from the constructor's own, already-built full
                    // Pi-chain type and CtorInfo's own arity -- simpler
                    // and more robust than trying to have finish_inductive
                    // separately expose its own CtorSpec-shaped inputs
                    // after the fact.
                    itp::NodeId cur = defs[j].type;
                    std::vector<itp::NodeId> arg_types;
                    for (uint32_t k = 0; k < defs[j].ctor_info->arity; ++k) {
                        auto node = itp::realize(cur);
                        auto* pi = std::get_if<itp::Pi>(&node);
                        if (!pi) throw ITPError("export_library: constructor '" + defs[j].name + "' own type has fewer Pi layers than its own recorded arity");
                        arg_types.push_back(pi->dom_id);
                        cur = pi->cod_id;
                    }
                    c["arg_types"] = [&]{ nb::list l; for (auto a : arg_types) l.append(serialize_term(a, *exec_ctx)); return l; }();
                    std::vector<itp::NodeId> result_indices;
                    itp::NodeId result = cur;
                    bool progressed = true;
                    while (progressed) {
                        progressed = false;
                        itp::inspect(result, [&](itp::App a) { result_indices.push_back(a.arg_id); result = a.fn_id; progressed = true; }, [](auto) {});
                    }
                    std::reverse(result_indices.begin(), result_indices.end());
                    c["result_indices"] = [&]{ nb::list l; for (auto a : result_indices) l.append(serialize_term(a, *exec_ctx)); return l; }();
                    ctors.append(c);
                    ++j;
                }
                e["ctors"] = ctors;
                events.append(e);
                i = j;
            } else {
                nb::dict e;
                e["kind"] = "axiom";
                e["name"] = def.name;
                e["type"] = serialize_term(def.type, *exec_ctx);
                events.append(e);
                ++i;
            }
        }
        return events;
    }

    void import_library(nb::list events) {
        auto g = activate();
        std::unordered_map<std::string, uint32_t> name_to_gid;
        for (nb::handle ev_h : events) {
            nb::dict ev = nb::cast<nb::dict>(ev_h);
            std::string kind = nb::cast<std::string>(ev["kind"]);
            std::string name = nb::cast<std::string>(ev["name"]);
            if (kind == "family") {
                itp::NodeId sort = deserialize_term(ev["sort"], name_to_gid);
                uint32_t type_gid = itp::reserve_inductive_type(name, sort);
                name_to_gid[name] = type_gid;
                std::vector<itp::CtorSpec> specs;
                std::vector<std::string> ctor_names;
                for (nb::handle c_h : nb::cast<nb::list>(ev["ctors"])) {
                    nb::dict c = nb::cast<nb::dict>(c_h);
                    std::string cname = nb::cast<std::string>(c["name"]);
                    std::vector<itp::NodeId> arg_types, result_indices;
                    for (nb::handle a : nb::cast<nb::list>(c["arg_types"])) arg_types.push_back(deserialize_term(a, name_to_gid));
                    for (nb::handle a : nb::cast<nb::list>(c["result_indices"])) result_indices.push_back(deserialize_term(a, name_to_gid));
                    specs.push_back(itp::CtorSpec{cname, arg_types, result_indices});
                    ctor_names.push_back(cname);
                }
                itp::InductiveHandle handle = itp::finish_inductive(type_gid, specs);
                for (std::size_t k = 0; k < ctor_names.size(); ++k) name_to_gid[ctor_names[k]] = handle.ctor_global_ids[k];
            } else if (kind == "axiom") {
                itp::NodeId ty = deserialize_term(ev["type"], name_to_gid);
                uint32_t gid = exec_ctx->globals.declare(ty, std::nullopt, name);
                name_to_gid[name] = gid;
            } else if (kind == "definition") {
                itp::NodeId ty = deserialize_term(ev["type"], name_to_gid);
                itp::NodeId val = deserialize_term(ev["value"], name_to_gid);
                auto r = itp::declare_checked(ty, val, name); // RE-verifies type-correctness, does not trust the blob
                raise_if_error(r);
                name_to_gid[name] = *r;
            } else if (kind == "recursive_function") {
                itp::NodeId ty = deserialize_term(ev["type"], name_to_gid);
                uint32_t gid = itp::reserve_recursive_function(name, ty);
                name_to_gid[name] = gid; // MUST be visible before deserializing body: the body self-references it
                std::string ind_name = nb::cast<std::string>(ev["ind_type_name"]);
                auto it = name_to_gid.find(ind_name);
                if (it == name_to_gid.end())
                    throw ITPError("import_library: recursive function '" + name + "' recurses on '" + ind_name +
                                    "', which has not been reconstructed yet");
                uint32_t ind_gid = it->second;
                uint32_t decreasing_arg_level = nb::cast<uint32_t>(ev["decreasing_arg_level"]);
                bool skip_guard = nb::cast<bool>(ev["guard_check_was_skipped"]);
                itp::NodeId body = deserialize_term(ev["body"], name_to_gid);
                // RE-verifies BOTH type-correctness AND termination (unless
                // the original declaration explicitly skipped the guard
                // check, in which case reconstruction honestly preserves
                // that same, already-accepted risk rather than silently
                // tightening or loosening it).
                auto r = itp::finish_recursive_function_checked(gid, ind_gid, decreasing_arg_level, body, skip_guard);
                raise_if_error(r);
            } else {
                throw ITPError("import_library: unrecognized event kind '" + kind + "' -- corrupted or incompatible export");
            }
        }
    }

    // Strips Pi and App layers to find the head Const, then checks
    // whether it names a declared inductive family (has at least one
    // constructor indexed under it) -- lets a search algorithm enumerate
    // which hypotheses are legal cases()/induct() targets directly,
    // rather than discovering it by probing every hyp_level and
    // catching the resulting ITPError on the ones that aren't.
    bool is_inductive_family_type(itp::NodeId ty) {
        auto g = activate();
        itp::NodeId cur = ty;
        bool progressed = true;
        while (progressed) {
            progressed = false;
            itp::inspect(cur, [&](itp::Pi p) { cur = p.cod_id; progressed = true; }, [](auto) {});
        }
        progressed = true;
        while (progressed) {
            progressed = false;
            itp::inspect(cur, [&](itp::App a) { cur = a.fn_id; progressed = true; }, [](auto) {});
        }
        uint32_t head_gid;
        bool is_const = false;
        itp::inspect(cur, [&](itp::Const c) { head_gid = c.global_id; is_const = true; }, [](auto) {});
        if (!is_const) return false;
        return !exec_ctx->globals.ctors_of(head_gid).empty();
    }

    // Whether target genuinely references hyp_level as a free variable
    // -- lets a search algorithm skip offering cases()/induct() on a
    // hypothesis the goal's own target doesn't actually depend on.
    // Necessary, not merely a search-quality nicety: induct()'s own
    // branch_expected/ih_ty construction produces an ill-scoped subgoal
    // for a no-field constructor (e.g. Zero) whenever the discriminee
    // is absent from the target -- confirmed directly by a real crash
    // (Env::operator[] out-of-bounds) -- see references_level's own
    // comment for the full mechanism.
    bool references_level_in_target(itp::NodeId target, uint32_t hyp_level) {
        auto g = activate();
        return itp::references_level(target, hyp_level);
    }

    // General safety check, usable regardless of which specific kernel
    // construction produced a subgoal: whether target is well-formed
    // relative to a context of exactly n_levels entries. Lets a caller
    // verify any newly-produced subgoal is genuinely well-scoped before
    // trusting it, catching this whole class of bug at its point of
    // use -- see is_well_scoped's own comment for why this exists
    // alongside, not instead of, the more targeted references_level_
    // in_target check.
    bool is_target_well_scoped(itp::NodeId target, uint32_t n_levels) {
        auto g = activate();
        return itp::is_well_scoped(target, n_levels);
    }

    // Diagnostics: {pool_name: (entries_interned, hash_table_slots)}.
    nb::dict pool_stats() {
        auto g = activate();
        nb::dict d;
        for (const auto& st : exec_ctx->pool_stats()) d[st.name] = nb::make_tuple(st.size, st.table_capacity);
        return d;
    }

    std::size_t library_size() {
        auto g = activate();
        return exec_ctx->globals.defs.size();
    }

    // Node count of the term reachable from `term` alone -- reuses the
    // same dense-visited DAG-dump machinery observation() uses, so
    // shared subterms are counted once, matching what "proof
    // complexity" should mean for a hash-consed representation.
    std::size_t proof_complexity(itp::NodeId term) {
        auto g = activate();
        DagDumpStorage storage;
        DenseVisited local_visited; // separate from dag_visited: this shouldn't perturb observation()'s own generation counter
        build_dag_dump(term, storage, local_visited);
        return storage.tags.size();
    }

    // Generic term observation, not tied to a ProofState/goal -- for
    // observing an arbitrary term directly (e.g. a candidate's own
    // type, or a partial proof term), same format as observation().
    auto observe_term(itp::NodeId term) {
        auto g = activate();
        auto* storage = new DagDumpStorage();
        dag_visited.reset();
        build_dag_dump(term, *storage, dag_visited);
        nb::capsule owner(storage, [](void* p) noexcept { delete static_cast<DagDumpStorage*>(p); });
        std::size_t n = storage->tags.size();
        return std::make_tuple(
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(storage->tags.data(), {n}, owner),
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(storage->scalar.data(), {n}, owner),
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(storage->child_a.data(), {n}, owner),
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(storage->child_b.data(), {n}, owner)
        );
    }

    // Every currently-declared global: (global_id, name, type, is_axiom
    // -- true iff it has no delta-reducible value). For the qualitative
    // lemma-review experiment (sampling and reviewing agent-discovered
    // lemmas) and general debugging/logging -- there was previously no
    // way to enumerate what the library actually contains from Python.
    std::vector<std::tuple<uint32_t, std::string, itp::NodeId, bool>> list_globals() {
        auto g = activate();
        std::vector<std::tuple<uint32_t, std::string, itp::NodeId, bool>> out;
        out.reserve(exec_ctx->globals.defs.size());
        for (std::size_t i = 0; i < exec_ctx->globals.defs.size(); ++i) {
            const auto& def = exec_ctx->globals.defs[i];
            out.emplace_back(static_cast<uint32_t>(i), def.name, def.type, !def.value.has_value());
        }
        return out;
    }

    itp::ExecutionContext::Checkpoint checkpoint() {
        auto g = activate();
        return exec_ctx->checkpoint();
    }
    void rewind(const itp::ExecutionContext::Checkpoint& cp) {
        auto g = activate();
        exec_ctx->rewind(cp);
    }

    void whole_term_check(itp::NodeId term, itp::NodeId goal_type) {
        auto g = activate();
        itp::ElabState es;
        itp::ElabCtx ec;
        raise_if_error(itp::check(term, goal_type, ec, es));
    }

    // Diagnostic variant: like whole_term_check but on failure returns the kernel's message together with the
    // offending subterm and (if any) the expected type, both printed with absolute levels (dump_raw style).
    std::string whole_term_check_detailed(itp::NodeId term, itp::NodeId goal_type) {
        auto g = activate();
        itp::ElabState es;
        itp::ElabCtx ec;
        auto r = itp::check(term, goal_type, ec, es);
        if (r) return "OK";
        std::string out = r.error().message + "\n  offending: " + dump_raw_unlocked(r.error().offending);
        if (r.error().expected) out += "\n  expected:  " + dump_raw_unlocked(*r.error().expected);
        return out;
    }
    itp::NodeId normalize_node(itp::NodeId term) {
        auto g = activate();
        return itp::normalize(term, {});
    }

    // normalize_node's own {} environment is only sound for a CLOSED
    // term. A goal's own target is routinely NOT closed -- it can
    // reference the goal's own hypotheses via bound Var(level) --  so
    // normalizing it plainly, with an empty environment, is unsound
    // (confirmed directly: it crashes on any goal with hypotheses in
    // scope). This builds the SAME environment ElabCtx::extend would
    // for the given hyp_types (fresh_neutral at each level) and
    // normalizes relative to that instead.
    itp::NodeId normalize_in_context(itp::NodeId term, const std::vector<itp::NodeId>& hyp_types) {
        auto g = activate();
        itp::ElabCtx ec{};
        for (auto ty : hyp_types) ec = ec.extend(ty);
        return itp::normalize(term, ec.values);
    }

    // Exposes itp::generalize directly: abstracts every SYNTACTIC (exact NodeId) occurrence of
    // `term_to_abstract` within `target`, returning the motive `fun w => target[term_to_abstract :=
    // w]` as an ordinary Lambda. This is exactly the same function cases()/induct() already use
    // internally for their own "concrete indices" and scrutinee -- deliberately syntactic, not
    // defeq/conv-based, by this codebase's own established design (see generalize's own doc
    // comment): matching real systems (Coq's own `generalize`, Agda's `with`), which abstract
    // syntactic occurrences too, not convertible-but-not-identical ones. The intended use here is
    // the first building block of a rewrite tactic: an agent can use a small library of explicit,
    // refl-provable "one-step unfolding" lemmas (e.g. mul_succ_right itself: mul n (Succ m) = add
    // (mul n m) n) to bring a goal into a syntactically-matching shape one step at a time, then
    // generalize_over + transport (the standard Eq.subst principle, hand-built once like cong) to
    // actually perform the rewrite -- without needing any conv-aware tree-walk or full
    // normalization, both of which this codebase has already confirmed, directly, cause more harm
    // than good (eager, exhaustive unfolding baking a permanent, stuck-match reconstruction into
    // long-lived goal syntax, exactly the bug intro()'s own history already warns against).
    //
    // hyp_types establishes the ambient context (mirroring normalize_in_context's own convention)
    // so term_to_abstract's own type can be correctly inferred if it references earlier
    // hypotheses; the fresh level the motive's own new binder lands at is exactly hyp_types.size(),
    // matching the "motive built at g.ec.size()" convention this entire codebase already uses.
    // True exactly when term's OWN outermost node is a Pi -- a direct, zero-unfolding check (no
    // eval(), no normalize_in_context). Delegates to the same itp::detail::is_pi_headed every
    // in-kernel caller uses (cases()'s and induct()'s own guards against being called while the
    // goal's own target is still a Pi, and rewrite_eq's own candidate-enumeration restriction in
    // hypergraph.py) rather than duplicating the check -- single source of truth.
    bool is_pi(itp::NodeId term) {
        auto g = activate();
        return itp::detail::is_pi_headed(term);
    }

    // Purely syntactic decomposition of an application chain (head, [arg1,...,argN]) -- walks App
    // nodes directly via inspect(), never calling eval(), so it never triggers the "eval() always
    // unfolds every reachable global Fix" behavior this codebase has already confirmed, twice
    // today, changes a term's own identity even when the printed result looks unchanged. Intended
    // use: given a hypothesis or library constant's own RAW, already-stored, never-evaluated type
    // (e.g. "Eq Nat (mul n' Zero) Zero" exactly as written, not as infer() would reconstruct it),
    // pull out A/x/y directly and safely for the rewrite tactic's own candidate enumeration.
    // Returns [head, arg1, ..., argN] as a single flat list (element 0 is always the head) --
    // avoids needing nanobind's own std::pair support, which this binding setup does not have.
    std::vector<itp::NodeId> decompose_app(itp::NodeId term) {
        auto g = activate();
        itp::detail::AppSpine s = itp::detail::decompose_app_spine(term);
        std::vector<itp::NodeId> out;
        out.reserve(s.args.size() + 1);
        out.push_back(s.head);
        for (auto a : s.args) out.push_back(a);
        return out;
    }

    itp::NodeId generalize_over(itp::NodeId target, itp::NodeId term_to_abstract,
                                 const std::vector<itp::NodeId>& hyp_types) {
        auto g = activate();
        itp::ElabCtx ec{};
        for (auto ty : hyp_types) ec = ec.extend(ty);
        itp::ElabState es{};
        itp::TypeResult ty_r = itp::infer(term_to_abstract, ec, es);
        if (!ty_r) throw std::runtime_error("generalize_over: term_to_abstract failed to infer a type -- " + ty_r.error().message);
        itp::NodeId abstracted_ty = *ty_r;
        itp::NodeId body = itp::detail::generalize(target, {{term_to_abstract, static_cast<uint32_t>(hyp_types.size())}});
        return itp::mk_lambda(abstracted_ty, body);
    }

    // Returns (tags, scalar, child_a, child_b, roots) as five numpy
    // arrays, sharing one heap allocation whose lifetime is tied to
    // all five together via a single capsule -- the arrays stay valid
    // as long as ANY of them is still referenced from Python, and the
    // whole allocation is freed together once none are.
    //
    // roots has length (number of hypotheses + 1): roots[i] for i <
    // num_hyps is hypothesis i's own type's root index into the other
    // four arrays; roots[num_hyps] (the last entry) is the target's
    // own root index. This is necessary, not just convenient: build_
    // dag_dump's own return value (each root's dump index) was
    // previously discarded entirely by this method, leaving no way
    // for a caller to know which index corresponds to which
    // hypothesis or the target -- confirmed directly to matter, not
    // just in principle, since two hypotheses of the identical type
    // (e.g. two Nat hyps) legitimately dump to the SAME index via
    // hash-consing, so position within a hyps loop cannot be
    // recovered from the four main arrays alone.
    // (tag, scalar) of each term's ROOT node, exactly as the first row of observe_term() would report
    // them (Var: level, Const: global id, otherwise the dump's own scalar), in ONE call. This is what a
    // policy needs to embed an APPLY candidate (an atom: a library constant or a hypothesis variable);
    // resolving them here, once per expansion, means no policy needs a kernel handle to do it.
    auto term_features(const std::vector<itp::NodeId>& nodes) {
        auto g = activate();
        struct Features { std::vector<uint32_t> tags, scalars; };
        auto* f = new Features();
        f->tags.reserve(nodes.size()); f->scalars.reserve(nodes.size());
        DagDumpStorage scratch;
        for (const auto& n : nodes) {
            scratch.tags.clear(); scratch.scalar.clear(); scratch.child_a.clear(); scratch.child_b.clear(); scratch.height.clear(); scratch.roots.clear();
            dag_visited.reset();
            build_dag_dump(n, scratch, dag_visited);
            f->tags.push_back(scratch.tags[0]);
            f->scalars.push_back(scratch.scalar[0]);
        }
        nb::capsule owner(f, [](void* p) noexcept { delete static_cast<Features*>(p); });
        std::size_t k = f->tags.size();
        return std::make_tuple(
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(f->tags.data(), {k}, owner),
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(f->scalars.data(), {k}, owner));
    }

    // The DAG dump of a goal given directly as (target, hypothesis types) -- identical arrays to
    // observation() of a ProofState holding that goal (hypotheses first, in order, then the target;
    // `roots` indexes them), but needs no live ProofState, so it can be computed AFTER a probe's
    // ephemeral state has been discarded/compacted, and only for goals that actually need it.
    auto observe_goal(itp::NodeId target, const std::vector<itp::NodeId>& hyp_types) {
        auto g = activate();
        auto* storage = new DagDumpStorage();
        dag_visited.reset();
        for (const auto& h : hyp_types) storage->roots.push_back(build_dag_dump(h, *storage, dag_visited));
        storage->roots.push_back(build_dag_dump(target, *storage, dag_visited));

        nb::capsule owner(storage, [](void* p) noexcept { delete static_cast<DagDumpStorage*>(p); });
        std::size_t n = storage->tags.size();
        std::size_t n_roots = storage->roots.size();
        return std::make_tuple(
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(storage->tags.data(), {n}, owner),
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(storage->scalar.data(), {n}, owner),
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(storage->child_a.data(), {n}, owner),
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(storage->child_b.data(), {n}, owner),
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(storage->roots.data(), {n_roots}, owner),
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(storage->height.data(), {n}, owner)
        );
    }

    // Finishes a SUCCESSFUL probe begun at checkpoint `cp`. If any resulting goal is not well scoped,
    // rewinds everything since `cp` and returns None. Otherwise returns (status, children) where
    // children is a list of (target, hyp_types, hyp_names), and -- if `compact` and it is safe --
    // has first reclaimed all the proof-term scaffolding the probe built (see itp::compact_since);
    // status: 0 compacted, 1 kept (an older meta was solved), 2 kept (a live term mentions a
    // post-checkpoint meta), 3 kept (compaction disabled). AFTER THIS CALL THE PROBE PROOFSTATE MUST
    // NOT BE USED AGAIN: its nodes may no longer exist.
    nb::object finish_probe(const itp::ExecutionContext::Checkpoint& cp, ProofStateHandle& psh, bool compact) {
        auto g = activate();
        auto& goals = psh.ps->goals;
        std::vector<itp::NodeId> roots;
        std::vector<std::vector<std::string>> names;
        int status = 3;
        bool ill_scoped = false;
        {
            // The expensive, purely-C++ part of finishing a probe (the well-scoped check and, the
            // bulk of it, compact_since's own term-graph walk): no Python object is touched
            // anywhere in this block, so it is safe to run with the GIL released. This one
            // binding cannot use a blanket call_guard the way the others in this file do, because
            // the nb::list/nb::make_tuple/nb::cast work below NEEDS the GIL back -- scoping the
            // release to just this block is the surgical alternative. See the GIL-release
            // correctness stress test (test_gil_release.py) for the concurrency evidence this
            // relies on: threads never share an ExecutionContext, so releasing the GIL here
            // never exposes any of this context's own state to a concurrent mutation it wasn't
            // already able to tolerate under normal (GIL-held, cooperatively-scheduled) use.
            nb::gil_scoped_release release;
            for (const auto& goal : goals) {
                if (!itp::is_well_scoped(goal.target, static_cast<uint32_t>(goal.hyps.size()))) {
                    exec_ctx->rewind(cp);
                    ill_scoped = true;
                    break;
                }
            }
            if (!ill_scoped) {
                for (const auto& goal : goals) {
                    std::vector<std::string> nm;
                    for (const auto& h : goal.hyps) { roots.push_back(h.type); nm.push_back(h.name); }
                    roots.push_back(goal.target);
                    names.push_back(std::move(nm));
                }
                if (compact) {
                    itp::CompactResult res = itp::compact_since(cp, roots);
                    status = static_cast<int>(res.status);
                    roots = std::move(res.roots);
                }
            }
        }
        if (ill_scoped) return nb::none();
        nb::list children;
        std::size_t off = 0;
        for (std::size_t gi = 0; gi < names.size(); ++gi) {
            std::size_t nh = names[gi].size();
            std::vector<itp::NodeId> hyps(roots.begin() + off, roots.begin() + off + nh);
            itp::NodeId target = roots[off + nh];
            off += nh + 1;
            children.append(nb::make_tuple(nb::cast(target), nb::cast(hyps), nb::cast(names[gi])));
        }
        return nb::make_tuple(status, children);
    }

    auto observation(ProofStateHandle& psh, std::optional<uint32_t> target_meta_id) {
        auto g = activate();
        auto idx_opt = itp::find_goal_index(*psh.ps, target_meta_id);
        if (!idx_opt) throw ITPError("observation: no goal with the given meta_id (already solved or superseded)");
        const itp::Goal& goal = psh.ps->goals[*idx_opt];

        auto* storage = new DagDumpStorage();
        dag_visited.reset();
        for (const auto& h : goal.hyps) storage->roots.push_back(build_dag_dump(h.type, *storage, dag_visited));
        storage->roots.push_back(build_dag_dump(goal.target, *storage, dag_visited));

        nb::capsule owner(storage, [](void* p) noexcept { delete static_cast<DagDumpStorage*>(p); });
        std::size_t n = storage->tags.size();
        std::size_t n_roots = storage->roots.size();
        return std::make_tuple(
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(storage->tags.data(), {n}, owner),
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(storage->scalar.data(), {n}, owner),
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(storage->child_a.data(), {n}, owner),
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(storage->child_b.data(), {n}, owner),
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(storage->roots.data(), {n_roots}, owner),
            nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(storage->height.data(), {n}, owner)
        );
    }
};

itp::ContextGuard ProofStateHandle::activate() { return kernel->activate(); }

NB_MODULE(itp_env, m) {
    nb::class_<itp::NodeId>(m, "NodeId")
        .def("__eq__", [](const itp::NodeId& a, const itp::NodeId& b) { return a == b; })
        .def("__hash__", [](const itp::NodeId& n) {
            return std::hash<uint64_t>{}((static_cast<uint64_t>(static_cast<uint32_t>(n.tag)) << 32) | n.index);
        })
        .def_prop_ro("tag", [](const itp::NodeId& n) { return static_cast<int>(n.tag); })
        .def_prop_ro("index", [](const itp::NodeId& n) { return n.index; })
        .def("__repr__", [](const itp::NodeId& n) {
            return "NodeId(tag=" + std::to_string(static_cast<int>(n.tag)) + ", index=" + std::to_string(n.index) + ")";
        });

    nb::class_<itp::CtorSpec>(m, "CtorSpec")
        .def(nb::init<>())
        .def_rw("name", &itp::CtorSpec::name)
        .def_rw("arg_types", &itp::CtorSpec::arg_types)
        .def_rw("result_indices", &itp::CtorSpec::result_indices)
        .def("__init__", [](itp::CtorSpec* self, const std::string& name,
                             std::vector<itp::NodeId> arg_types, std::vector<itp::NodeId> result_indices) {
            new (self) itp::CtorSpec{name, std::move(arg_types), std::move(result_indices)};
        }, "name"_a, "arg_types"_a, "result_indices"_a = std::vector<itp::NodeId>{});

    nb::class_<itp::InductiveHandle>(m, "InductiveHandle")
        .def_ro("type_global_id", &itp::InductiveHandle::type_global_id)
        .def_ro("ctor_global_ids", &itp::InductiveHandle::ctor_global_ids);

    nb::class_<itp::DiscriminationTree>(m, "Library");

    nb::class_<itp::ExecutionContext::Checkpoint>(m, "Checkpoint");

    nb::class_<ProofStateHandle>(m, "ProofState")
        .def("intro", &ProofStateHandle::intro_, "name"_a = "", "target_meta_id"_a = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("cases", &ProofStateHandle::cases_, "hyp_level"_a, "target_meta_id"_a = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("induct", &ProofStateHandle::induct_, "hyp_level"_a, "target_meta_id"_a = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("apply", &ProofStateHandle::apply_, "candidate"_a, "target_meta_id"_a = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("rewrite", &ProofStateHandle::rewrite_, "eq_term"_a, "x"_a, "y"_a, "transport_gid"_a, "symm_gid"_a, "target_meta_id"_a = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("is_complete", &ProofStateHandle::is_complete)
        .def("extract", &ProofStateHandle::extract)
        .def_ro("root_meta_id", &ProofStateHandle::root_meta_id)
        .def("num_goals", &ProofStateHandle::num_goals)
        .def("goal_meta_id", &ProofStateHandle::goal_meta_id, "i"_a)
        .def("goal_target", &ProofStateHandle::goal_target, "i"_a)
        .def("goal_num_hyps", &ProofStateHandle::goal_num_hyps, "i"_a)
        .def("goal_hyp_name", &ProofStateHandle::goal_hyp_name, "i"_a, "h"_a)
        .def("goal_hyp_type", &ProofStateHandle::goal_hyp_type, "i"_a, "h"_a);

    nb::class_<KernelWrapper>(m, "Kernel")
        .def(nb::init<>())
        .def("mk_type", &KernelWrapper::mk_type, "level"_a)
        .def("mk_prop", &KernelWrapper::mk_prop)
        .def("mk_pi", &KernelWrapper::mk_pi, "dom"_a, "cod"_a)
        .def("mk_var", &KernelWrapper::mk_var, "level"_a)
        .def("mk_app", &KernelWrapper::mk_app, "fn"_a, "arg"_a)
        .def("mk_lambda", &KernelWrapper::mk_lambda, "dom"_a, "body"_a)
        .def("mk_const", &KernelWrapper::mk_const, "global_id"_a)
        .def("mk_match", &KernelWrapper::mk_match, "scrutinee"_a, "motive"_a, "branches"_a)
        .def("reserve_inductive_type", &KernelWrapper::reserve_inductive_type, "name"_a, "sort"_a)
        .def("finish_inductive", &KernelWrapper::finish_inductive,
             "type_global_id"_a, "ctors"_a, "allow_unsafe_skip_positivity"_a = false)
        .def("reserve_recursive_function", &KernelWrapper::reserve_recursive_function, "name"_a, "type"_a)
        .def("finish_recursive_function_checked", &KernelWrapper::finish_recursive_function_checked,
             "fn_gid"_a, "ind_type_global_id"_a, "decreasing_arg_level"_a, "body"_a,
             "allow_unsafe_skip_guard_check"_a = false)
        .def("print_node", &KernelWrapper::print_node, "n"_a)
        .def("dump_raw", &KernelWrapper::dump_raw, "n"_a)
        .def("whole_term_check_detailed", &KernelWrapper::whole_term_check_detailed, "term"_a, "goal_type"_a)
        .def("meta_solution", &KernelWrapper::meta_solution, "meta_id"_a)
        .def("start_proof", &KernelWrapper::start_proof, "goal_type"_a, nb::keep_alive<0, 1>())
        .def("fork_single_goal", &KernelWrapper::fork_single_goal, "source"_a, "target_meta_id"_a, nb::keep_alive<0, 1>())
        .def("materialize_from_node", &KernelWrapper::materialize_from_node, "target"_a, "hyp_types"_a, "hyp_names"_a, nb::keep_alive<0, 1>(), nb::call_guard<nb::gil_scoped_release>())
        .def("index_library", &KernelWrapper::index_library, nb::call_guard<nb::gil_scoped_release>())
        .def("apply_candidates", &KernelWrapper::apply_candidates,
             "ps"_a, "library"_a, "target_meta_id"_a = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("declare_checked", &KernelWrapper::declare_checked, "type"_a, "value"_a, "name"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("checkpoint", &KernelWrapper::checkpoint, nb::call_guard<nb::gil_scoped_release>())
        .def("rewind", &KernelWrapper::rewind, "checkpoint"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("whole_term_check", &KernelWrapper::whole_term_check, "term"_a, "goal_type"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("normalize_node", &KernelWrapper::normalize_node, "term"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("normalize_in_context", &KernelWrapper::normalize_in_context, "term"_a, "hyp_types"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("generalize_over", &KernelWrapper::generalize_over, "target"_a, "term_to_abstract"_a, "hyp_types"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("decompose_app", &KernelWrapper::decompose_app, "term"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("is_pi", &KernelWrapper::is_pi, "term"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("observation", &KernelWrapper::observation, "ps"_a, "target_meta_id"_a = nb::none())
        .def("observe_goal", &KernelWrapper::observe_goal, "target"_a, "hyp_types"_a)
        .def("term_features", &KernelWrapper::term_features, "nodes"_a)
        .def("finish_probe", &KernelWrapper::finish_probe, "checkpoint"_a, "probe"_a, "compact"_a = true)
        .def("is_inductive_family_type", &KernelWrapper::is_inductive_family_type, "ty"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("references_level_in_target", &KernelWrapper::references_level_in_target, "target"_a, "hyp_level"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("is_target_well_scoped", &KernelWrapper::is_target_well_scoped, "target"_a, "n_levels"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("library_size", &KernelWrapper::library_size, nb::call_guard<nb::gil_scoped_release>())
        .def("pool_stats", &KernelWrapper::pool_stats)
        .def("proof_complexity", &KernelWrapper::proof_complexity, "term"_a)
        .def("observe_term", &KernelWrapper::observe_term, "term"_a)
        .def("list_globals", &KernelWrapper::list_globals)
        .def("export_library", &KernelWrapper::export_library)
        .def("import_library", &KernelWrapper::import_library, "events"_a);

    nb::exception<ITPError>(m, "ITPError");
}
