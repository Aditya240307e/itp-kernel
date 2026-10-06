// itp_kernel.hpp
//
// Infrastructure for building interactive theorem prover (ITP) kernels in
// C++23: hash-consed term representation, an arena allocator, normalization
// by evaluation, and supporting utilities for diagnostics and pattern
// matching.
//
// This library addresses the memory-management and boilerplate overhead
// that C++ typically imposes on this kind of work -- manual AST lifetime
// management, hand-rolled tagged unions, verbose visitor code. It does not
// reduce the size of the type theory itself: normalization, conversion
// checking, elaboration, universe handling, and positivity checking beyond
// the restricted form provided here remain the implementer's
// responsibility, in this or any other language.
//
// Node types are closed at compile time. Adding a new syntactic construct
// means adding a struct, an enum entry, a variant entry, and a NodePool
// member, then recompiling; there is no runtime plugin mechanism.
//
// Requires C++23 (std::expected, std::unreachable). No reflection is used;
// hashing and equality for each node type are written by hand.

#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <cstdio>
#include <vector>

namespace itp {

// ---------------------------------------------------------------------------
// Term representation
// ---------------------------------------------------------------------------
//
// Every syntactic construct is a plain-old-data struct of indices and
// NodeIds. Fields that reference other nodes store a full NodeId (tag plus
// index) rather than a bare index, since a bare index would not indicate
// which pool it refers to.

enum class NodeType : uint32_t { VAR, APP, LAMBDA, PI, TYPE, CONST, MATCH,
                                  META, TYPE_UNIV, QUOT_TYPE, QUOT_MK, QUOT_LIFT, LET, FIX };

/// A handle to an interned term node: a type tag and an index into the
/// corresponding pool. Trivially copyable and intended to be passed by
/// value throughout.
struct NodeId {
    NodeType tag;
    uint32_t index;
    friend bool operator==(NodeId, NodeId) = default;
};
static_assert(std::is_trivially_copyable_v<NodeId>);

/// A de Bruijn *level* reference to a locally bound variable, counted from
/// the bottom of the binder stack. Levels are used rather than indices
/// because they remain stable as the environment is extended, which keeps
/// environment-indexed evaluation free of reindexing.
///
/// Var refers only to locally bound variables. Top-level axioms,
/// definitions, and constructors are referenced via Const.
struct Var {
    uint32_t n;   // bound == true: de Bruijn INDEX (0 = innermost binder); bound == false: context LEVEL
    bool bound;
    friend bool operator==(Var, Var) = default;
};

struct App {
    NodeId fn_id, arg_id;
    friend bool operator==(App, App) = default;
};

/// A lambda abstraction. `type_id` is the annotated domain type of the
/// bound variable.
struct Lambda {
    NodeId type_id;
    NodeId body_id;
    friend bool operator==(Lambda, Lambda) = default;
};

/// A local let-binding, `let x : type_id := value_id in body_id`.
/// Computationally this is exactly a beta redex -- eval(Let) evaluates
/// value_id once and pushes the resulting Value onto the environment, the
/// same sharing App(Lambda(...), value_id) already gets from this being an
/// NbE evaluator over a semantic domain rather than substitution into
/// syntax. The reason to have a dedicated node at all is legibility of
/// unreduced syntax: printing an unnormalized term shows `let x := v in
/// body` rather than an anonymous applied lambda. That benefit lasts only
/// as long as the term is left unevaluated -- eval(Let) reduces it away
/// like any other redex, so a normalized term never shows a Let. There is
/// no selective reduction strategy here (e.g. "reduce everything except
/// let-bindings"); if a let needs to remain visible, don't normalize the
/// term, print the raw syntax instead.
struct Let {
    NodeId type_id;
    NodeId value_id;
    NodeId body_id;
    friend bool operator==(Let, Let) = default;
};

struct Pi {
    NodeId dom_id, cod_id;
    friend bool operator==(Pi, Pi) = default;
};

/// A universe. Carries an explicit level rather than being a single flat
/// sort, since a flat `Type : Type` is inconsistent. Cumulativity and
/// universe unification are not implemented here and are left to the
/// consumer of this library.
struct Type {
    uint32_t universe_level;
    friend bool operator==(Type, Type) = default;
};

/// A reference to a global: an axiom, a definition, or a constructor,
/// depending on how the referenced GlobalDef (see ExecutionContext) is
/// populated. There is no separate constructor node kind -- a constructor
/// is a tagged constant like any other, and partial or full application of
/// it is ordinary App.
struct Const {
    uint32_t global_id;
    friend bool operator==(Const, Const) = default;
};

/// Case analysis on an inductively defined value.
///
/// `branches` has one entry per constructor of the scrutinee's inductive
/// type, in declaration order. Each branch is a curried function of that
/// constructor's arguments: a chain of `arity` nested lambdas, one per
/// field (an arity-0 constructor's branch is the result term directly).
/// This convention means evaluation needs no additional binding machinery:
/// it evaluates the selected branch and applies it to the constructor's
/// stored arguments using the existing closure/application logic.
///
/// `motive_id` is retained for reconstructing and comparing a stuck match
/// (see VNeutral / MatchFrame) but is not inspected during ordinary
/// reduction; fully dependent elimination is left to the consumer.
///
/// `branches` is a shared_ptr rather than a plain vector so that copying a
/// Match (which happens routinely -- every ChunkedPool::push, every
/// pass-by-value through inspect()'s visitors) is a refcount bump instead
/// of a deep copy of the branch list. Equality and hashing compare the
/// pointed-to CONTENT, not the pointer itself, since two structurally
/// identical Match nodes built independently (e.g. by two separate calls
/// to mk_match with equal branch lists) must still hash-cons together;
/// comparing shared_ptr identity would break that. mk_match below always
/// constructs this via make_shared, so branches is never null in a Match
/// obtained through the normal API.
struct Match {
    NodeId scrutinee_id;
    NodeId motive_id;
    std::shared_ptr<const std::vector<NodeId>> branches;
    friend bool operator==(const Match& a, const Match& b) {
        return a.scrutinee_id == b.scrutinee_id && a.motive_id == b.motive_id && *a.branches == *b.branches;
    }
};
// mk_match (below, near the other mk_* constructors) uses
// NodePool<Match>::intern_lazy rather than intern() specifically so the
// std::make_shared allocation for `branches` only happens on a genuine
// cache miss -- NOT on every call regardless of hit/miss, which is what
// intern() would force (it requires a fully-formed Match, including its
// shared_ptr, before it can even check whether one is needed). A flat,
// offset+count arena (all branches from every Match packed into one
// shared, ExecutionContext-owned vector<NodeId>) would remove the
// allocation on the miss path too, but doing so correctly requires
// Match's equality/hash to read from that arena, which requires
// ExecutionContext to be visible at the point Match's hash specialization
// is defined -- and ExecutionContext isn't defined until much later in
// this file. Reordering the file to make that safe is a real, larger
// restructuring with its own risk; deferred rather than attempted under
// time pressure. What IS fixed is the copy cost (every Match value that
// flows through inspect()'s pass-by-value visitors is a refcount bump,
// not a deep copy) AND the allocate-then-immediately-free cost that used
// to hit on every hash-cons HIT, which is the more frequently-hit path in
// practice (e.g. quote() reconstructing an already-seen Match).

/// A reference to a metavariable: a placeholder for a term not yet known,
/// tracked in ExecutionContext::meta_ctx. This is the representation
/// primitive an elaborator needs, not an elaborator itself -- there is no
/// unification or constraint solving here, only a store of metavariables
/// with their expected type and, once assigned, their solution, plus the
/// evaluation/printing/quoting support needed to treat an unsolved meta as
/// an opaque stuck term and a solved one as transparently unfolding, the
/// same way an axiom versus a definition already work for Const.
struct Meta {
    uint32_t meta_id;
    friend bool operator==(Meta, Meta) = default;
};

/// A universe level that refers to one of the enclosing polymorphic
/// declaration's universe parameters, rather than a concrete number. Used
/// only inside the type/value templates passed to declare_polymorphic;
/// instantiate_global substitutes each such reference with a concrete
/// Type node before the result is used anywhere else, so the rest of this
/// library (eval, conv, printing) never needs to know this kind of
/// polymorphism exists except to reject it if it somehow reaches them
/// uninstantiated. This deliberately does not model a general level
/// algebra (max, imax of levels) -- only direct reference to a single
/// parameter -- which is enough for the common case of a declaration
/// whose sort is exactly one of its own universe parameters.
struct TypeUniv {
    uint32_t param_index;
    friend bool operator==(TypeUniv, TypeUniv) = default;
};

/// The quotient type former, `Quot r` for a relation `r`. Modeled as a
/// single generic node kind reused for every quotient a consumer builds,
/// the same way Match is reused for every inductive rather than each
/// getting its own node kind.
struct QuotType {
    NodeId relation;
    friend bool operator==(QuotType, QuotType) = default;
};

/// The quotient introduction form, `Quot.mk r a`.
struct QuotMk {
    NodeId relation;
    NodeId elem;
    friend bool operator==(QuotMk, QuotMk) = default;
};

/// The quotient eliminator, `Quot.lift f h q`: lifts a function `f` that
/// respects `relation` (as witnessed by `respects_proof`) to a function
/// out of the quotient. `codomain` is retained for faithful
/// reconstruction by quote() but, like Match's motive, is not consulted
/// during reduction. The computation rule -- `Quot.lift f h (Quot.mk r a)`
/// reduces to `f a` -- does not need `respects_proof` either; it exists
/// purely for typechecking, which is left to the consumer. `Quot.sound`
/// (the axiom making Quot.mk r a and Quot.mk r b propositionally equal
/// when r a b holds) needs no dedicated node kind at all -- it is an
/// ordinary opaque axiom a consumer declares once, typically via
/// declare_polymorphic below, since it does not compute.
struct QuotLift {
    NodeId relation;
    NodeId codomain;
    NodeId f;
    NodeId respects_proof;
    NodeId quotient_elem;
    friend bool operator==(QuotLift, QuotLift) = default;
};

/// A LOCAL fixpoint: `fix (self : fn_type) := body`, decreasing on
/// argument position `decreasing_arg` of `fn_type`'s own Pi telescope.
/// This is the local, proof-term-scoped counterpart of a global
/// recursive definition's self-reference via Const -- see GlobalEnv's
/// comment on why an ordinary `Const` self-reference only unfolds when
/// guarded by a Lambda, and Phase 2's ROADMAP/PHASE2_PLAN for why
/// routing every in-progress `Induct` attempt through
/// reserve_recursive_function/finish_recursive_function instead would
/// permanently install a GlobalDef per attempt, including abandoned
/// ones -- exactly the linear-scan-pressure concern flagged for
/// GlobalEnv.defs elsewhere in this file.
///
/// `fn_type` is Fix's own Pi type, in the same "closed relative to its
/// binding point" convention every other node's embedded type uses.
/// `body` is evaluated in an environment extended with a reference back
/// to the Fix node's own eventual value at level 0 relative to Fix's
/// own (implicit) binder -- mirroring exactly how a Lambda's own body
/// is evaluated one level further in than its parameter. There is no
/// separate bound name for the self-reference: it is addressed the same
/// way every other local binder is, by level, once `eval`/`infer`
/// extend the ambient context by one for it.
///
/// `decreasing_arg` is not inspected by eval() at all (mirroring
/// `Match::motive_id` and `QuotLift::codomain`, which are retained for
/// faithful reconstruction/printing but never consulted during ordinary
/// reduction) -- it exists purely for is_locally_structurally_decreasing
/// to check once, at Induct-tactic-acceptance time.
struct Fix {
    NodeId fn_type;
    uint32_t decreasing_arg;
    NodeId body;
    friend bool operator==(Fix, Fix) = default;
};

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

inline void hash_combine_into(std::size_t& seed, std::size_t h) {
    seed ^= h + 0x9e3779b9u + (seed << 6) + (seed >> 2);
}

template <typename T>
void hash_combine(std::size_t& seed, const T& val) {
    hash_combine_into(seed, std::hash<T>{}(val));
}

template <typename... Args>
std::size_t hash_fields(const Args&... args) {
    std::size_t seed = 0;
    (hash_combine(seed, args), ...);
    return seed;
}

/// Full-avalanche finalizer (splitmix64), applied by NodePool to every
/// raw hash before it indexes the table. hash_combine above is boost-
/// style over std::hash<uint32_t>, which libstdc++ implements as the
/// IDENTITY function -- so structurally-related nodes (e.g. apps over
/// consecutively-numbered children) get near-consecutive raw hashes,
/// which linear probing turns into huge contiguous clusters that any
/// new key landing inside must walk to the end of. Measured directly:
/// interning 16 fresh nodes cost ~500 us into a 100K-entry pool but
/// ~150 ns into a 400K one, depending only on where the raw hashes
/// happened to fall. The table is indexed by the LOW bits (h & mask),
/// so every input bit must influence them; this guarantees that.
/// Applied inside NodePool rather than by changing hash_combine, so it
/// affects table placement only -- never the iteration order of any
/// std::unordered_* container elsewhere in the kernel, and never index
/// assignment (indices are insertion-ordered).
inline std::size_t mix_hash(std::size_t x) noexcept {
    uint64_t z = static_cast<uint64_t>(x) + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return static_cast<std::size_t>(z ^ (z >> 31));
}

} // namespace itp

template <>
struct std::hash<itp::NodeId> {
    std::size_t operator()(itp::NodeId id) const noexcept {
        return itp::hash_fields(static_cast<uint32_t>(id.tag), id.index);
    }
};
template <>
struct std::hash<itp::Var> {
    std::size_t operator()(itp::Var v) const noexcept { return itp::hash_fields(v.n, v.bound); }
};
template <>
struct std::hash<itp::App> {
    std::size_t operator()(itp::App a) const noexcept { return itp::hash_fields(a.fn_id, a.arg_id); }
};
template <>
struct std::hash<itp::Lambda> {
    std::size_t operator()(itp::Lambda l) const noexcept { return itp::hash_fields(l.type_id, l.body_id); }
};
template <>
struct std::hash<itp::Let> {
    std::size_t operator()(itp::Let l) const noexcept {
        return itp::hash_fields(l.type_id, l.value_id, l.body_id);
    }
};
template <>
struct std::hash<itp::Pi> {
    std::size_t operator()(itp::Pi p) const noexcept { return itp::hash_fields(p.dom_id, p.cod_id); }
};
template <>
struct std::hash<itp::Type> {
    std::size_t operator()(itp::Type t) const noexcept { return itp::hash_fields(t.universe_level); }
};
template <>
struct std::hash<itp::Const> {
    std::size_t operator()(itp::Const c) const noexcept { return itp::hash_fields(c.global_id); }
};
template <>
struct std::hash<itp::Match> {
    std::size_t operator()(const itp::Match& m) const noexcept {
        std::size_t seed = itp::hash_fields(m.scrutinee_id, m.motive_id);
        for (itp::NodeId b : *m.branches) itp::hash_combine(seed, b);
        return seed;
    }
};
template <>
struct std::hash<itp::Meta> {
    std::size_t operator()(itp::Meta m) const noexcept { return itp::hash_fields(m.meta_id); }
};
template <>
struct std::hash<itp::TypeUniv> {
    std::size_t operator()(itp::TypeUniv t) const noexcept { return itp::hash_fields(t.param_index); }
};
template <>
struct std::hash<itp::QuotType> {
    std::size_t operator()(itp::QuotType q) const noexcept { return itp::hash_fields(q.relation); }
};
template <>
struct std::hash<itp::QuotMk> {
    std::size_t operator()(itp::QuotMk q) const noexcept { return itp::hash_fields(q.relation, q.elem); }
};
template <>
struct std::hash<itp::QuotLift> {
    std::size_t operator()(itp::QuotLift q) const noexcept {
        return itp::hash_fields(q.relation, q.codomain, q.f, q.respects_proof, q.quotient_elem);
    }
};
template <>
struct std::hash<itp::Fix> {
    std::size_t operator()(itp::Fix f) const noexcept {
        return itp::hash_fields(f.fn_type, f.decreasing_arg, f.body);
    }
};

namespace itp {

// ---------------------------------------------------------------------------
// Arena allocation
// ---------------------------------------------------------------------------
//
// Nodes are allocated in fixed-size pages of 1024 elements, addressed with
// a shift and a mask rather than division and modulus. Pages are stored
// behind unique_ptr in a vector, so growing the pool never relocates
// existing elements.
//
// The arena is append-only and never reclaims memory until the process
// exits, which is appropriate for a batch typechecking run. A long-running
// server or REPL process would need a different memory strategy.

template <typename T>
class ChunkedPool {
public:
    static constexpr uint32_t kBlockBits = 10;
    static constexpr uint32_t kBlockSize = 1u << kBlockBits;

    T& operator[](uint32_t i) {
        uint32_t block = i >> kBlockBits;
        uint32_t slot  = i & (kBlockSize - 1);
        return (*blocks_[block])[slot];
    }
    const T& operator[](uint32_t i) const {
        uint32_t block = i >> kBlockBits;
        uint32_t slot  = i & (kBlockSize - 1);
        return (*blocks_[block])[slot];
    }

    uint32_t push(T value) {
        uint32_t block = next_index_ >> kBlockBits;
        if (block >= blocks_.size()) {
            blocks_.push_back(std::make_unique<std::array<T, kBlockSize>>());
        }
        uint32_t idx = next_index_++;
        (*this)[idx] = std::move(value);
        return idx;
    }

    uint32_t size() const { return next_index_; }

    // Rolls next_index_ back to new_size, for reclaiming an episode's own
    // ephemeral nodes once nothing keeps a reference to them (see
    // ExecutionContext::rewind's own safety contract). Entries at or
    // above new_size are reset to a default-constructed T IMMEDIATELY,
    // not left as-is until some future push() happens to overwrite that
    // same slot -- for a T holding a resource (e.g. Match's own
    // shared_ptr<vector<NodeId>>), leaving it untouched would keep that
    // resource alive, unreclaimed, for however long it takes a later
    // episode to grow back this far, which may never happen. Blocks
    // themselves are never freed: existing indices below new_size stay
    // validly addressed by the same block/slot arithmetic as before,
    // and a later push() reusing a rolled-back index simply move-
    // assigns over the now-default value, exactly as it would over any
    // other occupied slot.
    void truncate_and_release(uint32_t new_size) {
        for (uint32_t i = new_size; i < next_index_; ++i) (*this)[i] = T{};
        next_index_ = new_size;
    }

private:
    std::vector<std::unique_ptr<std::array<T, kBlockSize>>> blocks_;
    uint32_t next_index_ = 0;
};

// ---------------------------------------------------------------------------
// Hash consing
// ---------------------------------------------------------------------------
//
// Each pool deduplicates by hash first, then confirms with a structural
// equality check on any candidate sharing that hash, before deciding two
// nodes are the same. Relying on the hash alone would let a hash collision
// cause two distinct terms to be treated as identical, which would corrupt
// conversion checking silently. See itp_kernel_test.cpp for a test that
// forces a collision and confirms it is handled correctly.
//
// Deduplication uses a flat, open-addressed table (linear probing, no
// deletions so no tombstones are needed) rather than
// unordered_map<hash, vector<index>>. The earlier chaining design cost
// roughly two heap allocations per newly interned term -- one for the
// map's bucket node, one for the vector's backing array -- confirmed by
// direct measurement, not estimated. The flat table's backing array grows
// geometrically (doubling under a 0.7 load factor), so the amortized
// allocation cost per intern() call is effectively zero; the only
// allocations are the O(log n) table growths over the pool's whole
// lifetime, not one per element. Capacity is kept a power of two so
// wrapping a probe index uses a mask rather than a modulus, consistent
// with the arena's own bitshift/mask indexing above.

template <typename T>
struct NodePool {
    // 8 bytes (was 24: bool + size_t + uint32_t, padded). index_plus1 == 0
    // means empty, so no separate flag is needed; hash32 is the LOW 32 bits
    // of the (mixed) hash. That is all a lookup or a re-placement ever
    // needs: the table is indexed by `hash & mask`, and mask < 2^32
    // (grow() refuses to exceed 2^32 slots), so the low 32 bits determine
    // the bucket exactly as the full hash would. Halves-and-more the
    // cache footprint of every intern, and cuts the table's memory ~3x.
    struct Slot {
        uint32_t index_plus1 = 0;
        uint32_t hash32 = 0;
        bool occupied() const { return index_plus1 != 0; }
        uint32_t index() const { return index_plus1 - 1; }
    };
    static_assert(sizeof(Slot) == 8, "NodePool::Slot must stay 8 bytes: the table's memory and cache footprint depend on it");

    static constexpr std::size_t kInitialCapacity = 16; // must stay a power of two

    ChunkedPool<T> storage;
    std::vector<Slot> table; // empty until the first intern() call
    std::size_t count = 0;

    // slot_of_index[i] is the position in `table` currently holding entry
    // i (always exactly size() long). Together with stable_size below,
    // this is what lets rewind() reclaim entries by clearing just their
    // own slots -- O(entries reclaimed) -- instead of rebuilding the
    // whole table, O(entries ever interned). See rewind()'s own comment
    // for the invariant that makes clearing safe without tombstones.
    std::vector<uint32_t> slot_of_index;

    // High-water mark below which the table's PLACEMENT ORDER may differ
    // from index order: every entry with index >= stable_size was placed
    // by a plain intern() insertion, in index order, after the last
    // grow()/rebuild re-placed everything else. grow() sets it to
    // size() (it re-places entries in arbitrary, old-slot order);
    // rebuild_from_survivors() sets it to 0 (it re-places in index
    // order, so placement order == index order everywhere).
    uint32_t stable_size = 0;

    // rewind() shrinks the table (via a rebuild) only when it has become
    // both large and mostly empty -- see rewind()'s own comment.
    static constexpr std::size_t kShrinkMinCapacity = 1u << 10;
    static constexpr std::size_t kShrinkRatio = 8;

    uint32_t intern(T fields) {
        std::size_t h = mix_hash(std::hash<T>{}(fields));
        const uint32_t h32 = static_cast<uint32_t>(h);
        if (table.empty()) table.assign(kInitialCapacity, Slot{});
        std::size_t mask = table.size() - 1;
        std::size_t i = h & mask;
        while (true) {
            Slot& slot = table[i];
            if (!slot.occupied()) {
                uint32_t idx = storage.push(std::move(fields));
                slot = Slot{idx + 1, h32};
                slot_of_index.push_back(static_cast<uint32_t>(i)); // == idx-th entry's slot; grow() bounds table.size() < 2^32
                ++count;
                if (count * 10 >= table.size() * 7) grow(); // load factor 0.7, integer arithmetic
                return idx;
            }
            if (slot.hash32 == h32 && storage[slot.index()] == fields) {
                return slot.index(); // the check that must not be skipped
            }
            i = (i + 1) & mask; // linear probing; no tombstones: rewind() only ever clears a SUFFIX of the placement order (see rewind)
        }
    }

    /// Like intern(), but for T whose construction is expensive (owns a
    /// heap allocation, e.g. Match's shared_ptr<vector<NodeId>>) and where
    /// that expense should only be paid on a genuine miss, not on every
    /// call regardless of whether the entry already exists. The generic
    /// intern() above cannot do this: it takes a fully-formed T, so the
    /// caller has already paid to construct T before intern() gets a
    /// chance to say "you didn't need to."
    ///
    /// Callers instead supply a lightweight Key (borrowing the caller's
    /// data, no allocation), a hash function over Key, an equality test
    /// comparing an already-stored T against a Key, and a Build thunk that
    /// materializes a T from Key -- called ONLY on the insert path, after
    /// the probe has confirmed no existing entry matches. On a hit, Build
    /// is never invoked and no allocation happens at all.
    template <typename Key, typename HashKey, typename EqAgainstStored, typename Build>
    uint32_t intern_lazy(const Key& key, HashKey&& hash_key, EqAgainstStored&& eq_against_stored, Build&& build) {
        std::size_t h = mix_hash(hash_key(key));
        const uint32_t h32 = static_cast<uint32_t>(h);
        if (table.empty()) table.assign(kInitialCapacity, Slot{});
        std::size_t mask = table.size() - 1;
        std::size_t i = h & mask;
        while (true) {
            Slot& slot = table[i];
            if (!slot.occupied()) {
                uint32_t idx = storage.push(build()); // only path that ever allocates
                slot = Slot{idx + 1, h32};
                slot_of_index.push_back(static_cast<uint32_t>(i));
                ++count;
                if (count * 10 >= table.size() * 7) grow();
                return idx;
            }
            if (slot.hash32 == h32 && eq_against_stored(storage[slot.index()], key)) {
                return slot.index(); // hit: build() never ran, nothing was allocated
            }
            i = (i + 1) & mask;
        }
    }

    void grow() {
        // slot_of_index stores positions as uint32_t; a table this large
        // would need > 4 billion interned entries in ONE pool, far past
        // anything a proof search produces -- but fail loudly rather than
        // silently truncate a position if it ever happens.
        if (table.size() * 2 > (std::size_t{1} << 32))
            throw std::length_error("NodePool: table capacity exceeds 2^32 slots");
        std::vector<Slot> old = std::move(table);
        table.assign(old.size() * 2, Slot{});
        std::size_t mask = table.size() - 1;
        for (const Slot& s : old) {
            if (!s.occupied()) continue;
            std::size_t i = s.hash32 & mask;
            while (table[i].occupied()) i = (i + 1) & mask;
            table[i] = s;
            slot_of_index[s.index()] = static_cast<uint32_t>(i);
        }
        // Every existing entry was just re-placed in old-slot order, not
        // index order -- so nothing below size() can be reclaimed by
        // clearing slots alone until a rebuild restores index order.
        stable_size = storage.size();
    }

    T& operator[](uint32_t i) { return storage[i]; }
    const T& operator[](uint32_t i) const { return storage[i]; }
    uint32_t size() const { return storage.size(); }
    std::size_t capacity() const { return table.size(); } // hash-table slots currently allocated (diagnostics)

    // Reclaims every entry interned since new_size, for use at an
    // episode boundary or after a failed probe, once nothing outside
    // this pool references any of them (see ExecutionContext::rewind's
    // own safety contract -- this method alone does not know whether
    // that holds and trusts its caller).
    //
    // Cost is O(entries reclaimed), NOT O(entries ever interned). The
    // earlier implementation rebuilt the entire hash table on every
    // call (open addressing without tombstones cannot delete an
    // arbitrary slot safely, so it threw the table away and re-placed
    // every survivor) -- correct, and the right trade at an episode
    // boundary where nearly everything is being discarded, but a search
    // rewinds after every FAILED PROBE against a pool that keeps
    // growing (successful probes are committed and never rewound), so
    // each rewind cost O(everything committed so far) for all 14 pools,
    // even pools where nothing had been added: measured directly, one
    // rewind after interning 8 nodes cost 163 ns on an empty pool but
    // 12 us at 1K committed nodes, 4.9 ms at 100K, 16.9 ms at 400K --
    // making total search cost quadratic in tree size.
    //
    // Why clearing just the reclaimed slots is safe WITHOUT tombstones:
    // a linear-probing table's lookups are correct as long as no entry's
    // probe chain contains a hole. Each entry's chain prefix consisted
    // of slots that were already occupied when it was placed, and
    // occupied slots only ever become empty through this very method.
    // So if the entries being removed are a SUFFIX of the table's
    // placement order (everything placed after them stays placed; they
    // were placed after everything that remains), clearing them cannot
    // put a hole in any remaining entry's chain -- the table becomes
    // exactly what it was before they were inserted. Entries with index
    // >= stable_size are placed in index order by construction, so any
    // rewind to new_size >= stable_size removes such a suffix. Below
    // stable_size (a rewind older than the last grow()), placement
    // order is arbitrary and the safe fallback is the old full rebuild
    // -- which this method then does in index order, restoring
    // stable_size to 0. That rebuild costs O(survivors) and is paid at
    // most once per grow() that a rewind reaches back across, itself
    // already an O(capacity) event, so it is amortized against it.
    //
    // The table is also right-sized when a rewind leaves it mostly
    // empty (capacity >= kShrinkRatio x survivors, and non-trivially
    // large): a table left at its old, grown size after most of its
    // entries are discarded keeps the poor cache behavior that
    // prompted the earlier full-rebuild design. That condition can
    // only newly become true through a large rewind (nothing else
    // shrinks the entry count), so the O(survivors) rebuild it
    // triggers is charged to the entries that rewind just reclaimed.
    void rewind(uint32_t new_size) {
        const uint32_t cur = storage.size();
        if (new_size >= cur) return; // nothing interned since the checkpoint: nothing to reclaim
        storage.truncate_and_release(new_size);
        if (new_size == 0) {
            // Leave the table empty; the next intern() lazily re-creates it at kInitialCapacity.
            table.clear();
            slot_of_index.clear();
            count = 0;
            stable_size = 0;
            return;
        }
        if (new_size >= stable_size) {
            for (uint32_t j = new_size; j < cur; ++j) table[slot_of_index[j]] = Slot{};
            slot_of_index.resize(new_size);
            count = new_size;
            if (table.size() >= kShrinkMinCapacity && std::size_t{new_size} * kShrinkRatio < table.size())
                rebuild_from_survivors(new_size);
            return;
        }
        rebuild_from_survivors(new_size);
    }

    // Re-places entries [0, n) into a freshly sized table, IN INDEX
    // ORDER, so afterwards placement order == index order everywhere
    // (stable_size == 0). Precondition: storage has already been
    // truncated to n, and slot_of_index[0..n) still names each
    // survivor's slot in the CURRENT table (reclaimed entries' slots
    // may still be occupied there; they are simply never read).
    void rebuild_from_survivors(uint32_t n) {
        std::vector<Slot> old = std::move(table);
        std::size_t new_cap = kInitialCapacity;
        while (new_cap * 7 <= std::size_t{n} * 10) new_cap *= 2; // restore the load-factor-0.7 invariant intern() relies on
        table.assign(new_cap, Slot{});
        slot_of_index.resize(n);
        std::size_t mask = new_cap - 1;
        for (uint32_t j = 0; j < n; ++j) {
            const Slot s = old[slot_of_index[j]]; // read the OLD position before overwriting it below
            std::size_t i = s.hash32 & mask;
            while (table[i].occupied()) i = (i + 1) & mask;
            table[i] = s;
            slot_of_index[j] = static_cast<uint32_t>(i);
        }
        count = n;
        stable_size = 0;
    }

    /// Structural self-check, for tests: returns an empty string if every
    /// invariant rewind()/grow()/intern() maintain holds, otherwise a
    /// description of the first violation. O(capacity + size); never
    /// called on any hot path.
    std::string check_invariants() const {
        const uint32_t n = storage.size();
        if (count != n) return "count != storage.size()";
        if (slot_of_index.size() != n) return "slot_of_index.size() != storage.size()";
        if (stable_size > n) return "stable_size > storage.size()";
        if (n == 0) return table.empty() || count == 0 ? std::string{} : "non-empty table with zero entries";
        if (table.empty()) return "entries exist but table is empty";
        if ((table.size() & (table.size() - 1)) != 0) return "table capacity is not a power of two";
        if (n * 10 >= table.size() * 7) return "load factor invariant violated";
        std::size_t occupied = 0;
        for (const Slot& sl : table) {
            if (!sl.occupied()) continue;
            ++occupied;
            if (sl.index() >= n) return "occupied slot names an index >= size() (stale entry)";
        }
        if (occupied != n) return "occupied slot count != size()";
        const std::size_t mask = table.size() - 1;
        for (uint32_t j = 0; j < n; ++j) {
            const std::size_t pos = slot_of_index[j];
            if (pos >= table.size() || !table[pos].occupied() || table[pos].index() != j)
                return "slot_of_index[j] does not point at entry j";
            // chain integrity: walking from the ideal slot must reach pos without crossing an empty slot
            std::size_t i = table[pos].hash32 & mask;
            while (i != pos) {
                if (!table[i].occupied()) return "hole in an entry's probe chain";
                i = (i + 1) & mask;
            }
        }
        return {};
    }
};

// ---------------------------------------------------------------------------
// Execution context and global environment
// ---------------------------------------------------------------------------

/// Present on a GlobalDef when that global is a constructor of an
/// inductive type, rather than a plain axiom or definition.
/// Forward-declared here (full definition much later, alongside the
/// structural-decreasing checkers that were its original motivation) so
/// that finish_inductive/finish_mutual_inductive_block, both defined
/// before it in file order, can call it once at declaration time to
/// precompute and cache each constructor's own recursive-field mask --
/// matching this file's existing forward-declaration convention (see
/// print's own two forward declarations above its real definition).
inline std::vector<bool> recursive_field_mask(uint32_t ind_type_global_id, uint32_t ctor_gid);

struct CtorInfo {
    uint32_t ind_type_global_id;
    uint32_t ctor_idx;
    uint32_t arity;
    // Precomputed once, at declaration time (see GlobalEnv::declare_ctor
    // below), rather than recomputed -- and freshly heap-allocated -- on
    // every call to recursive_field_mask from the structural-decreasing
    // checkers' own hot path. A constructor's recursive-field shape is
    // fixed forever once declared, so there is never a reason to
    // recompute it.
    std::vector<bool> recursive_mask;
};

// Mirrors ctor_info's own pattern: recorded once, at declaration time,
// rather than left to be lost. finish_recursive_function and
// finish_recursive_function_checked both take ind_type_global_id and
// decreasing_arg_level purely as transient parameters to run
// is_structurally_decreasing once -- neither previously stored them
// anywhere, so nothing about a global's own termination justification
// survived past that one check. This matters beyond bookkeeping: it
// means a recursive function's termination could never be re-verified
// later (e.g. reconstructing a global from a cross-process export)
// without re-deriving this from scratch -- confirmed by directly
// reading both functions' own implementations, not assumed.
struct RecursiveFnInfo {
    uint32_t ind_type_global_id;
    uint32_t decreasing_arg_level;
    bool guard_check_was_skipped; // true iff declared via allow_unsafe_skip_guard_check=true
};

struct GlobalDef {
    NodeId type;
    std::optional<NodeId> value;       // present for a definition (delta-reducible); absent for an axiom
    std::optional<CtorInfo> ctor_info; // present if this global is a constructor
    std::optional<RecursiveFnInfo> recursive_info; // present if declared via finish_recursive_function(_checked)
    std::string name;
};

struct GlobalEnv {
    std::vector<GlobalDef> defs;
    // Every family's own constructor global_ids, in ctor_idx order --
    // both declaration sites (finish_inductive, finish_mutual_inductive_
    // block) declare a family's constructors via `for (idx = 0; idx <
    // ctors.size(); ++idx)`, so idx increases monotonically and matches
    // ctor_idx exactly; appending at declare_ctor's own call point keeps
    // this vector indexed by ctor_idx for free, with no separate sort
    // ever needed. Turns "find the constructor(s) of family X" -- three
    // separate call sites did this via their own, independent O(all
    // globals) linear scan before this existed, confirmed by direct
    // benchmark to degrade by roughly 37x between a 2,000- and a
    // 12,000-global library -- into a single hash lookup plus, for a
    // SPECIFIC constructor, a direct array index (ctor_idx IS the
    // position), decoupled entirely from total library size. Append-
    // only, like `defs` itself: globals are never deleted in this
    // codebase (see the earlier, explicitly rejected "purge" design),
    // so this index never needs invalidation or removal logic, only
    // growth.
    std::unordered_map<uint32_t, std::vector<uint32_t>> ctors_by_family;

    uint32_t declare(NodeId type, std::optional<NodeId> value, std::string name) {
        defs.push_back(GlobalDef{type, std::move(value), std::nullopt, std::nullopt, std::move(name)});
        return static_cast<uint32_t>(defs.size() - 1);
    }
    const GlobalDef& operator[](uint32_t id) const { return defs.at(id); }

    // Declares a constructor global AND indexes it, in one call -- the
    // single chokepoint both finish_inductive and finish_mutual_
    // inductive_block go through, specifically so "declared a
    // constructor but forgot to index it" is structurally impossible
    // rather than merely unlikely (the two operations can no longer
    // happen independently, unlike the two-line "declare() then
    // manually patch ctor_info" pattern this replaces at both call
    // sites). Computes the recursive-field mask ITSELF, internally,
    // after its own declare() call -- not accepted as a parameter from
    // the caller: recursive_field_mask needs the constructor's own type
    // already stored (which only declare() can do, and only declare()
    // knows the freshly-assigned gid needed to store it against), so
    // there is no way for a caller to hand in a complete mask before
    // this call exists. Callable only because of the forward
    // declaration above (recursive_field_mask's own full definition
    // comes much later in the file, alongside the structural-decreasing
    // checkers that were its original motivation) and because *this*
    // and ctx().globals are the same instance here -- recursive_field_
    // mask's own ctx().globals[ctor_gid].type lookup sees exactly the
    // type declare() just stored, immediately above.
    uint32_t declare_ctor(NodeId type, std::string name, uint32_t ind_type_global_id, uint32_t ctor_idx, uint32_t arity) {
        uint32_t gid = declare(type, std::nullopt, std::move(name));
        std::vector<bool> mask = recursive_field_mask(ind_type_global_id, gid);
        defs[gid].ctor_info = CtorInfo{ind_type_global_id, ctor_idx, arity, std::move(mask)};
        ctors_by_family[ind_type_global_id].push_back(gid);
        return gid;
    }

    // O(1) lookup (hash + direct array index) replacing what used to be
    // an O(all globals) linear scan at every call site that needed it.
    // Returns an empty span-like view (via a static empty vector) for a
    // family with no declared constructors yet, or an unknown id,
    // rather than throwing -- matching ctor_gids_for_type's own,
    // pre-existing "return an empty vector" contract for that case.
    const std::vector<uint32_t>& ctors_of(uint32_t ind_type_global_id) const {
        static const std::vector<uint32_t> empty;
        auto it = ctors_by_family.find(ind_type_global_id);
        return it != ctors_by_family.end() ? it->second : empty;
    }

    // A design decision worth being explicit about here, not just in
    // chat: this index is NOT synchronized. Safe for now specifically
    // because ExecutionContext (and therefore GlobalEnv) is still
    // thread_local -- one implicit context per thread, no concurrent
    // mutation possible. This becomes a genuine concurrent-mutation
    // hazard the moment that thread-locality is removed (a separate,
    // later piece of Phase 3, deliberately not touched by this pass) --
    // flagged here so whoever does that refactor doesn't have to
    // rediscover it from scratch.

    // A definition may reference its own global_id: this is the ordinary
    // way to write a recursive function, and it works correctly as long
    // as the self-reference is guarded by at least one Lambda, which is
    // true of any function body written in the usual style (the recursive
    // call appears inside the function, not at its head). Evaluating a
    // Lambda does not force its body, so the reference only unfolds when
    // the function is actually applied, one call at a time, terminating
    // exactly when the underlying recursion does -- no separate fixpoint
    // construct is needed for this case. A definition whose self-
    // reference is NOT guarded by a Lambda diverges under evaluation, the
    // same way an unguarded self-referential binding would under any
    // strict evaluation strategy.
};

// To declare a set of mutually recursive definitions, reserve a global id
// for each one first (declare with value = std::nullopt), build each body
// referencing the others by their already-known global_ids, then patch
// defs[gid].value in afterward. GlobalDef is not part of any hash-consed
// identity, so mutating it after the fact does not disturb existing
// NodeIds -- the same two-phase pattern used by reserve_inductive_type
// and finish_inductive below.

/// One metavariable: its expected type, its solution once assigned, and
/// the local binder depth in scope when it was created. `scope_depth` is
/// the representation-level hook a real occurs/scope check needs -- a
/// solution referencing a local variable at a level >= scope_depth would
/// escape the context the metavariable was created in, which is exactly
/// the kind of check a unification engine (not provided here) is
/// responsible for performing before calling solve().
struct MetaEntry {
    NodeId type;
    std::optional<NodeId> solution;
    uint32_t scope_depth;
};

/// A store of metavariables, parallel in shape to GlobalEnv but kept
/// separate: metavariable lifetime and bulk-discard-on-backtrack are
/// different enough from permanent global declarations that conflating
/// the two stores would invite mistakes (for example, clearing abandoned
/// metavariables from a failed elaboration attempt should never be able
/// to touch real declarations).
///
/// solve_trail + checkpoint()/rewind() give Phase 2's tactic engine real
/// backtracking: checkpoint() before attempting a tactic, rewind() to
/// unwind every meta solved since if the tactic fails, keep going
/// forward on success. rewind() is correct as a simple "clear the
/// solution field" unwind ONLY because solve() is (and must remain)
/// single-assignment per meta within one checkpoint/rewind bracket -- a
/// meta is never re-solved without being unwound first, so there is
/// never a "previous value" to restore, only "was unsolved before, make
/// it unsolved again."
struct MetaContext {
    std::vector<MetaEntry> entries;
    std::vector<uint32_t> solve_trail; // ids solved since the last checkpoint

    uint32_t fresh(NodeId type, uint32_t scope_depth) {
        entries.push_back(MetaEntry{type, std::nullopt, scope_depth});
        return static_cast<uint32_t>(entries.size() - 1);
    }
    const MetaEntry& operator[](uint32_t id) const { return entries.at(id); }
    bool is_solved(uint32_t id) const { return entries.at(id).solution.has_value(); }
    void solve(uint32_t id, NodeId solution) {
        entries.at(id).solution = solution;
        solve_trail.push_back(id);
    }

    // Mark carries both counters a full rewind needs together: entries
    // at or beyond entries_size are discarded outright (fresh() never
    // handed out an id below entries_size after this checkpoint, so
    // nothing else can reference them), while any id BELOW entries_size
    // that got solved after this checkpoint still exists afterward and
    // needs its solution correctly undone, exactly as the original,
    // solve_trail-only rewind already did. Safe to change unconditionally
    // rather than adding a second, parallel method: confirmed zero
    // existing callers of checkpoint()/rewind() anywhere in this file
    // before this change, so there is no prior semantics to preserve.
    struct Mark { std::size_t solve_trail_size; std::size_t entries_size; };

    Mark checkpoint() const { return Mark{solve_trail.size(), entries.size()}; }
    void rewind(Mark mark) {
        while (solve_trail.size() > mark.solve_trail_size) {
            uint32_t id = solve_trail.back();
            if (id < mark.entries_size) entries.at(id).solution = std::nullopt;
            solve_trail.pop_back();
        }
        entries.resize(mark.entries_size);
    }
};

/// A universe-polymorphic declaration's template, prior to instantiation.
/// See declare_polymorphic / instantiate_global.
struct PolyDecl {
    uint32_t num_univ_params;
    NodeId type_template;
    std::optional<NodeId> value_template;
    std::string name;
};

struct ExecutionContext {
    NodePool<Var>      vars;
    NodePool<App>      apps;
    NodePool<Lambda>   lambdas;
    NodePool<Let>      lets;
    NodePool<Pi>       pis;
    NodePool<Type>     types;
    NodePool<Const>    consts;
    NodePool<Match>    matches;
    NodePool<Meta>     metas;
    NodePool<TypeUniv> type_univs;
    NodePool<QuotType> quot_types;
    NodePool<QuotMk>   quot_mks;
    NodePool<QuotLift> quot_lifts;
    NodePool<Fix>      fixes;
    GlobalEnv          globals;
    MetaContext        meta_ctx;
    std::vector<PolyDecl> poly_decls;
    // Step 3 (refutation exploitation) infrastructure: built lazily, once
    // per ExecutionContext, and reused. unit_type_gid/empty_type_gid are
    // this context's own canonical Unit (one constructor) and Empty (zero
    // constructors) types; same_ctor_gid_by_family maps an inductive
    // family's own type_global_id to a synthesized "same_ctor : F -> F ->
    // Type0" discriminator function for that family (Unit when both sides
    // share a constructor head, Empty when they don't), used to exploit a
    // detected refutation by extending a match's own motive rather than
    // requiring the caller to supply an impossible branch's body by hand.
    std::optional<uint32_t> unit_type_gid, unit_ctor_gid, empty_type_gid;
    std::unordered_map<uint32_t, uint32_t> same_ctor_gid_by_family;
    std::unordered_map<uint64_t, uint32_t> loose_cache; // loose_range memo, keyed by NodeId; cleared on rewind

    // Reclaims every term node and metavariable created since a prior
    // checkpoint(), for use at an episode boundary (an RL step, a
    // rejected or abandoned proof attempt) once nothing outside the pools
    // themselves references any of it. Exists because every one of the 14
    // NodePools above, plus MetaContext's own entries, previously grew
    // monotonically for the ExecutionContext's entire lifetime -- never
    // shrinking, even for a term that becomes logically dead the moment
    // an episode ends. Measured directly, not assumed: a fixed-global-
    // count benchmark of repeated intro()+cases() batches showed per-op
    // latency climbing from ~4.5k to 16-18k ns/op over 20 batches, with
    // spikes to 188k ns/op coinciding with the underlying hash tables'
    // own grow() events -- the working set outgrowing cache, exactly as
    // an unbounded, ever-growing pool would predict.
    //
    // SAFETY CONTRACT: rewind() must only be called when nothing declared
    // into globals since the matching checkpoint() is being kept. A global
    // declared during an episode has its own type/value referencing nodes
    // allocated during that same episode; rewinding past them would not
    // crash -- worse, it would silently leave that global's NodeIds
    // pointing at whatever a LATER episode's own allocations happen to
    // overwrite that same storage slot with, since a NodeId remains a
    // "valid" index regardless of what now lives there. rewind() checks
    // this explicitly (comparing globals.defs.size() against the
    // checkpoint) and throws rather than risk a silent corruption: an
    // episode that successfully proves and keeps something should simply
    // not call rewind() at all, accepting that episode's own scratch
    // nodes staying in the pool permanently -- the same behavior as
    // before this fix, but now bounded by the SUCCESS rate rather than
    // paid by every episode regardless of outcome.
    /// Diagnostics: entries interned and hash-table slots allocated, per
    /// node pool. Used to measure how much state a search leaves
    /// committed (successful probes are deliberately never rewound) and
    /// to verify rewind's cost tracks what it reclaims, not what remains.
    struct PoolStat { const char* name; uint32_t size; std::size_t table_capacity; };
    std::vector<PoolStat> pool_stats() const {
        return {
            {"vars", vars.size(), vars.capacity()},         {"apps", apps.size(), apps.capacity()},
            {"lambdas", lambdas.size(), lambdas.capacity()}, {"lets", lets.size(), lets.capacity()},
            {"pis", pis.size(), pis.capacity()},             {"types", types.size(), types.capacity()},
            {"consts", consts.size(), consts.capacity()},    {"matches", matches.size(), matches.capacity()},
            {"metas", metas.size(), metas.capacity()},       {"type_univs", type_univs.size(), type_univs.capacity()},
            {"quot_types", quot_types.size(), quot_types.capacity()}, {"quot_mks", quot_mks.size(), quot_mks.capacity()},
            {"quot_lifts", quot_lifts.size(), quot_lifts.capacity()}, {"fixes", fixes.size(), fixes.capacity()},
        };
    }

    struct Checkpoint {
        uint32_t vars_n, apps_n, lambdas_n, lets_n, pis_n, types_n, consts_n,
                 matches_n, metas_n, type_univs_n, quot_types_n, quot_mks_n,
                 quot_lifts_n, fixes_n;
        MetaContext::Mark meta_mark;
        std::size_t globals_n; // safety check only; globals itself is never rewound
    };

    Checkpoint checkpoint() const {
        return Checkpoint{
            vars.size(), apps.size(), lambdas.size(), lets.size(), pis.size(), types.size(),
            consts.size(), matches.size(), metas.size(), type_univs.size(), quot_types.size(),
            quot_mks.size(), quot_lifts.size(), fixes.size(),
            meta_ctx.checkpoint(), globals.defs.size()
        };
    }

    // Derived globals are the ones the kernel itself synthesises on demand, deterministically and
    // idempotently, for refutation exploitation (get_or_build_same_ctor / _unit_type / _empty_type):
    // `__same_ctor_<family>`, `__Unit` with its constructor `__tt`, and `__Empty`.
    static bool is_derived_global_name(const std::string& nm) {
        return nm.rfind("__same_ctor_", 0) == 0 || nm == "__Unit" || nm == "__tt" || nm == "__Empty";
    }

    void rewind(const Checkpoint& cp) {
        if (globals.defs.size() != cp.globals_n) {
            // A probe may be the first thing to need a derived global (cases() builds `same_ctor` lazily
            // the first time it exploits a refutation; since cases() can now succeed in more situations,
            // this happens inside probes). Dropping such a global together with the probe is safe: it is
            // reconstructed identically on next use, and anything that referenced it was itself created
            // during the probe and is being rewound. Anything ELSE declared since the checkpoint is a
            // genuine kept result, so the original refusal stands.
            bool all_derived = globals.defs.size() > cp.globals_n;
            for (std::size_t g = cp.globals_n; all_derived && g < globals.defs.size(); ++g)
                all_derived = is_derived_global_name(globals.defs[g].name);
            if (!all_derived) {
                throw std::logic_error("ExecutionContext::rewind: a global was declared since this "
                    "checkpoint -- rewinding would silently orphan the nodes its type/value reference. "
                    "Do not call rewind() for an episode whose result is being kept.");
            }
            const uint32_t keep = static_cast<uint32_t>(cp.globals_n);
            globals.defs.erase(globals.defs.begin() + keep, globals.defs.end());
            for (auto it = globals.ctors_by_family.begin(); it != globals.ctors_by_family.end();)
                it = (it->first >= keep) ? globals.ctors_by_family.erase(it) : std::next(it);
            for (auto it = same_ctor_gid_by_family.begin(); it != same_ctor_gid_by_family.end();)
                it = (it->first >= keep || it->second >= keep) ? same_ctor_gid_by_family.erase(it) : std::next(it);
            if (unit_type_gid && *unit_type_gid >= keep) { unit_type_gid.reset(); unit_ctor_gid.reset(); }
            if (empty_type_gid && *empty_type_gid >= keep) empty_type_gid.reset();
        }
        loose_cache.clear();
        vars.rewind(cp.vars_n);
        apps.rewind(cp.apps_n);
        lambdas.rewind(cp.lambdas_n);
        lets.rewind(cp.lets_n);
        pis.rewind(cp.pis_n);
        types.rewind(cp.types_n);
        consts.rewind(cp.consts_n);
        matches.rewind(cp.matches_n);
        metas.rewind(cp.metas_n);
        type_univs.rewind(cp.type_univs_n);
        quot_types.rewind(cp.quot_types_n);
        quot_mks.rewind(cp.quot_mks_n);
        quot_lifts.rewind(cp.quot_lifts_n);
        fixes.rewind(cp.fixes_n);
        meta_ctx.rewind(cp.meta_mark);
    }
};

inline thread_local ExecutionContext* active_context = nullptr;

/// RAII guard that scopes `active_context` for its lifetime, restoring the
/// previous context on destruction. This allows multiple independent
/// contexts to be used within a single thread (for example, isolated unit
/// tests, or a service checking several files) without requiring every
/// call in this library to take an explicit context parameter.
struct ContextGuard {
    ExecutionContext* prev;
    explicit ContextGuard(ExecutionContext& ctx) : prev(active_context) { active_context = &ctx; }
    ~ContextGuard() { active_context = prev; }
    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;
};

/// Returns the currently active context. Throws if none is set, which
/// indicates a missing ContextGuard rather than a condition callers should
/// need to handle as ordinary control flow.
inline ExecutionContext& ctx() {
    if (!active_context) {
        throw std::logic_error("itp: no active ExecutionContext (missing ContextGuard?)");
    }
    return *active_context;
}

// ---------------------------------------------------------------------------
// Term construction and structural dispatch
// ---------------------------------------------------------------------------

inline NodeId mk_fvar(uint32_t level) { return NodeId{NodeType::VAR, ctx().vars.intern(Var{level, false})}; }
inline NodeId mk_bvar(uint32_t index) { return NodeId{NodeType::VAR, ctx().vars.intern(Var{index, true})}; }
inline NodeId mk_app(NodeId fn, NodeId arg) {
    return NodeId{NodeType::APP, ctx().apps.intern(App{fn, arg})};
}
inline NodeId mk_lambda(NodeId type_id, NodeId body_id) {
    return NodeId{NodeType::LAMBDA, ctx().lambdas.intern(Lambda{type_id, body_id})};
}
inline NodeId mk_let(NodeId type_id, NodeId value_id, NodeId body_id) {
    return NodeId{NodeType::LET, ctx().lets.intern(Let{type_id, value_id, body_id})};
}
inline NodeId mk_pi(NodeId dom_id, NodeId cod_id) {
    return NodeId{NodeType::PI, ctx().pis.intern(Pi{dom_id, cod_id})};
}
inline NodeId mk_type(uint32_t universe_level) {
    return NodeId{NodeType::TYPE, ctx().types.intern(Type{universe_level})};
}

/// Prop, i.e. Sort 0 -- the impredicative, proof-irrelevant-once-
/// conv_at's short-circuit is live, bottom universe. Reserved
/// EXCLUSIVELY for propositions (Eq's own conclusion slot, and any
/// other proof-carrying family's conclusion) -- never for ordinary data
/// types, including non-recursive "enum-shaped" ones like Bool: what
/// puts something in Prop is that it's meant to be reasoned about, not
/// computed on and pattern-inspected, not its syntactic shape.
inline NodeId mk_prop() { return mk_type(0); }

/// Lean's surface-level `Type u` -- Sort(u+1) in this kernel's raw
/// level numbering (Prop = Sort 0 = Type{0} occupies level 0 alone; Type
/// u = Sort(u+1)). THIS -- not a bare mk_type(u) -- is what ordinary
/// data type declarations should be built from:
/// `reserve_inductive_type("Nat", mk_data_sort(0))` puts Nat at Lean's
/// "Type 0" (Sort 1), leaving Sort 0 exclusively for mk_prop(). The gap
/// this closes: every inductive type in this codebase was previously
/// declared via a bare mk_type(0), silently colliding with Prop's own
/// level -- caught only once real motive inference (infer/check) first
/// evaluated a global's declared sort semantically and classified Nat
/// itself as a proposition. mk_type(level) itself is unchanged and
/// still the right tool for genuine raw-Sort-level code (cumulativity
/// checks, universe-polymorphism instantiation, printing) -- these two
/// wrappers are specifically for CtorSpec/reserve_inductive_type/
/// declare_inductive call sites, the population that got this wrong.
inline NodeId mk_data_sort(uint32_t u) { return mk_type(u + 1); }

inline NodeId mk_const(uint32_t global_id) {
    return NodeId{NodeType::CONST, ctx().consts.intern(Const{global_id})};
}
inline NodeId mk_match(NodeId scrutinee_id, NodeId motive_id, std::vector<NodeId> branches) {
    // Key borrows the caller's branches by reference -- no allocation yet.
    // hash_key/eq_against_stored only ever read through that reference;
    // the shared_ptr is built by `build` below, and build is only invoked
    // by intern_lazy on a genuine miss (see intern_lazy's doc comment and
    // the comment above the Match struct for why this matters).
    struct MatchKey {
        NodeId scrutinee_id;
        NodeId motive_id;
        const std::vector<NodeId>& branches;
    };
    MatchKey key{scrutinee_id, motive_id, branches};

    uint32_t idx = ctx().matches.intern_lazy(
        key,
        [](const MatchKey& k) {
            std::size_t seed = hash_fields(k.scrutinee_id, k.motive_id);
            for (NodeId b : k.branches) hash_combine(seed, b);
            return seed;
        },
        [](const Match& stored, const MatchKey& k) {
            return stored.scrutinee_id == k.scrutinee_id && stored.motive_id == k.motive_id &&
                   *stored.branches == k.branches;
        },
        [&]() {
            return Match{scrutinee_id, motive_id, std::make_shared<const std::vector<NodeId>>(std::move(branches))};
        });
    return NodeId{NodeType::MATCH, idx};
}

/// Relocated here (from much later in the file, near CtorSpec) purely so
/// unify_values/infer/check/generalize/etc. can use it -- it has no
/// dependencies beyond mk_app, which is already available at this point.
inline NodeId apply_all(NodeId fn, const std::vector<NodeId>& args) {
    NodeId result = fn;
    for (NodeId a : args) result = mk_app(result, a);
    return result;
}

inline NodeId mk_meta(uint32_t meta_id) {
    return NodeId{NodeType::META, ctx().metas.intern(Meta{meta_id})};
}
inline NodeId mk_type_univ(uint32_t param_index) {
    return NodeId{NodeType::TYPE_UNIV, ctx().type_univs.intern(TypeUniv{param_index})};
}
inline NodeId mk_quot_type(NodeId relation) {
    return NodeId{NodeType::QUOT_TYPE, ctx().quot_types.intern(QuotType{relation})};
}
inline NodeId mk_quot_mk(NodeId relation, NodeId elem) {
    return NodeId{NodeType::QUOT_MK, ctx().quot_mks.intern(QuotMk{relation, elem})};
}
inline NodeId mk_quot_lift(NodeId relation, NodeId codomain, NodeId f, NodeId respects_proof, NodeId quotient_elem) {
    return NodeId{NodeType::QUOT_LIFT,
                  ctx().quot_lifts.intern(QuotLift{relation, codomain, f, respects_proof, quotient_elem})};
}
inline NodeId mk_fix(NodeId fn_type, uint32_t decreasing_arg, NodeId body) {
    return NodeId{NodeType::FIX, ctx().fixes.intern(Fix{fn_type, decreasing_arg, body})};
}

using Term = std::variant<Var, App, Lambda, Let, Pi, Type, Const, Match, Meta, TypeUniv, QuotType, QuotMk, QuotLift, Fix>;

inline Term realize(NodeId id) {
    switch (id.tag) {
        case NodeType::VAR:       return Term{ctx().vars[id.index]};
        case NodeType::APP:       return Term{ctx().apps[id.index]};
        case NodeType::LAMBDA:    return Term{ctx().lambdas[id.index]};
        case NodeType::LET:       return Term{ctx().lets[id.index]};
        case NodeType::PI:        return Term{ctx().pis[id.index]};
        case NodeType::TYPE:      return Term{ctx().types[id.index]};
        case NodeType::CONST:     return Term{ctx().consts[id.index]};
        case NodeType::MATCH:     return Term{ctx().matches[id.index]};
        case NodeType::META:      return Term{ctx().metas[id.index]};
        case NodeType::TYPE_UNIV: return Term{ctx().type_univs[id.index]};
        case NodeType::QUOT_TYPE: return Term{ctx().quot_types[id.index]};
        case NodeType::QUOT_MK:   return Term{ctx().quot_mks[id.index]};
        case NodeType::QUOT_LIFT: return Term{ctx().quot_lifts[id.index]};
        case NodeType::FIX:       return Term{ctx().fixes[id.index]};
    }
    std::unreachable();
}

/// Merges a set of callables into a single object with a combined
/// operator(), for use with std::visit.
template <class... Ts>
struct match_on : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
match_on(Ts...) -> match_on<Ts...>;

/// Dispatches on the syntactic shape of a term. Requires a visitor for
/// every alternative of Term; omitting one is a compile error rather than
/// a silent gap, since std::visit requires the merged visitor to be
/// callable with every alternative of the variant.
template <typename... Visitors>
auto inspect(NodeId id, Visitors... vs) {
    return std::visit(match_on{vs...}, realize(id));
}

/// Two-argument form, for rules that need to match on both operands at
/// once (conversion checking, typing rules comparing two terms).
///
/// std::visit over two N-alternative variants dispatches over the full
/// N*N cross product, so exhaustive coverage would otherwise require N*N
/// visitors. A generic catch-all as the last visitor avoids this:
///
///   inspect(a, b, [](App,App){...}, [](Lambda,Lambda){...},
///                 [](auto,auto){ return false; });
///
/// Overload resolution prefers the specific, non-template matches over the
/// templated catch-all wherever both are viable. See conv_value below for
/// an example on the semantic Value domain.
template <typename... Visitors>
auto inspect(NodeId a, NodeId b, Visitors... vs) {
    return std::visit(match_on{vs...}, realize(a), realize(b));
}

// Universe levels are represented by Type::universe_level above.
// Cumulativity and universe unification are not implemented here.

// ---------------------------------------------------------------------------
// Normalization by evaluation
// ---------------------------------------------------------------------------
//
// Terms are evaluated into a semantic domain rather than reduced by
// substitution on syntax. A lambda evaluates to a closure -- a captured
// environment plus an unevaluated body -- applied lazily. This is where de
// Bruijn levels pay for themselves: a newly bound variable simply takes
// env.size() as its level, with no reindexing of anything already in the
// environment, unlike de Bruijn indices under substitution.
//
// Value and EliminatorFrame are mutually recursive: a VNeutral's spine
// holds EliminatorFrames, and an ApplyFrame holds a Value. This is
// resolved with the usual forward-declare / alias / define-in-dependency-
// order idiom: the alternatives of Value (VType, Closure, VNeutral, VPi,
// VCtor, VQuotType, VQuotMk) must all be fully defined before ApplyFrame
// is defined, since ApplyFrame holds a Value by value and therefore
// requires Value to be a complete type at that point.
//
// VQuotType and VQuotMk need a Value themselves (the relation, and for
// VQuotMk the wrapped element) but, unlike ApplyFrame, they are
// THEMSELVES alternatives of Value -- holding a Value directly would
// require Value complete while Value's own alternative list is still
// being assembled, which is circular. shared_ptr<Value> tolerates the
// incomplete type as a member the same way vector<Value> does elsewhere,
// so that indirection is used for exactly these two structs.

struct Closure; struct VNeutral; struct VPi; struct VType; struct VCtor;
struct VQuotType; struct VQuotMk; struct VFix;
struct ApplyFrame; struct MatchFrame; struct QuotLiftFrame;

using Value = std::variant<VType, Closure, VNeutral, VPi, VCtor, VQuotType, VQuotMk, VFix>;
using EliminatorFrame = std::variant<ApplyFrame, MatchFrame, QuotLiftFrame>;

// ---------------------------------------------------------------------------
// Env: a persistent, structurally shared environment
// ---------------------------------------------------------------------------
//
// A plain std::vector<Value>, copied on every extension, costs O(depth)
// per extension and O(n^2) total across a term of binder-depth n --
// confirmed by direct measurement, not just asymptotic argument: at
// n=16000, building a deeply nested term took roughly 6.7 seconds with a
// vector-based environment. Environments must be persistent (a Closure
// captured earlier must not observe a later extension made from what
// looks like "the same" prefix, since multiple live closures can diverge
// from a shared prefix independently), which is exactly what motivates a
// structurally-shared representation instead of a plain mutable vector.
//
// A naive persistent singly-linked list gets extension down to O(1), but
// pushes lookup cost up to O(depth) -- and that is a real, not
// theoretical, regression: a workload with many lookups of an old
// (deeply-nested) binding measured at over 5 seconds with a linked list,
// worse than the vector-based O(n^2) it was meant to replace. Env is
// instead a persistent vector: a shallow, wide tree (branching factor 32,
// the same technique behind Clojure's PersistentVector), giving both
// extension and lookup O(log32 n) -- in practice a handful of pointer
// hops even at depths in the hundreds of thousands, measured directly
// against both the vector and linked-list alternatives across build-
// heavy, lookup-heavy, and balanced workloads, with Env's worst case
// across all three under 40ms where the alternatives each had a workload
// that took multiple seconds.
//
// Leaf slots are stored as shared_ptr<const Value>, not Value directly.
// This is required by the same mutual-recursion constraint as VQuotType/
// VQuotMk above (EnvNode's structural definition only needs Value
// forward-declared this way, not complete, so it can be fully defined
// here rather than deferred like ApplyFrame/MatchFrame below) -- but it
// is also the right choice on its own terms: it keeps a leaf node's path-
// copy cost at a constant "32 pointer copies" regardless of how large or
// expensive Value itself is to copy, which matters because Value has
// gained alternatives over the life of this header (Meta, TypeUniv, the
// Quot family) and is expected to gain more; an unboxed design would have
// silently gotten more expensive to path-copy every time Value grew.
// Internal (non-leaf) and leaf nodes are kept structurally distinct via a
// variant rather than a struct that always has room for both regardless
// of which kind a given node actually is -- an earlier draft of this
// design did the latter, wasting up to ~2KB per node and destroying cache
// locality for no benefit, caught before it was ever implemented here.

static constexpr uint32_t kEnvBits = 5;
static constexpr uint32_t kEnvBranch = 1u << kEnvBits; // 32

struct EnvNode; // forward-declared for self-reference in EnvInternalSlots
using EnvLeafSlots = std::array<std::shared_ptr<const Value>, kEnvBranch>;
using EnvInternalSlots = std::array<std::shared_ptr<const EnvNode>, kEnvBranch>;

/// A node is EITHER a leaf (holds values) OR internal (holds children),
/// never both -- which kind is determined by comparing the traversal
/// depth against Env::shift, tracked externally, not stored per node.
struct EnvNode {
    std::variant<EnvLeafSlots, EnvInternalSlots> slots;
};

struct Env {
    std::shared_ptr<const EnvNode> root; // nullptr when empty
    uint32_t count = 0;
    uint32_t shift = 0; // 0 when root is a leaf; +kEnvBits per additional tree level

    uint32_t size() const { return count; }

    // Bodies are defined after Value's alternatives (Closure, VPi, ...)
    // are complete, since they call make_shared<const Value> -- the same
    // forward-declare-then-define-later pattern eval()/quote()/
    // conv_value() already use below, applied here for the same reason.
    const Value& operator[](uint32_t level) const;
    Env extend(Value v) const;
};

// ---------------------------------------------------------------------------
// LocalCtx: the elaborator's typing-scope counterpart to Env
// ---------------------------------------------------------------------------
//
// Elaboration (infer/check, Phase 1) needs a structure exactly parallel to
// Env -- indexed by de Bruijn level, extended once per binder crossed --
// but holding the TYPE of each bound variable rather than its (not yet
// evaluated, during elaboration) value. It is kept fully separate from
// Env rather than merged into it: Env is threaded through evaluation,
// where most terms never touch elaboration at all, and LocalCtx is
// threaded through elaboration/proof search, where Phase 2's ProofState
// will branch across many candidate tactics per goal. Tying the two
// together would mean every eval() call anywhere carries the weight of a
// typing context it doesn't need, and every LocalCtx branch during
// tactic search would carry Env's (possibly large) Value payloads it
// doesn't need either.
//
// This reuses Env's actual design principle -- a shallow, wide,
// structurally-shared tree, branching factor 32, so extension and lookup
// are both O(log32 n) and so independent branches (the exact shape
// Phase 2's tactic search produces: many candidate next-goals sharing a
// long common prefix of hypotheses) share structure instead of copying
// it -- rather than reusing Env's actual EnvNode type. Two things differ
// enough to make a separate type the right call, not a shortcut taken to
// avoid genericizing Env: (1) EnvNode's leaves are shared_ptr<const
// Value> specifically because Value is forward-declared at that point in
// the file (mutually recursive with EliminatorFrame/VNeutral) and
// because Value is expensive and still growing (Meta, TypeUniv, the Quot
// family were all added after Env's leaf representation was chosen) --
// neither reason applies to NodeId, which is a complete, 8-byte,
// trivially-copyable POD at the point LocalCtx is defined, so wrapping
// each leaf slot in its own shared_ptr here would be pure overhead with
// no forward-declaration need and no future-growth need to hedge
// against; LocalCtxNode's leaves store NodeId directly. (2) Genericizing
// EnvNode into a template shared by both would touch Env's already-
// measured, already-tested implementation for a benefit (code reuse)
// that doesn't offset the risk of destabilizing it -- the same
// time-pressure tradeoff already made once in this file (see the Match
// struct's comment on why a flat offset+count arena for `branches` was
// designed but not attempted). Only the tree NODES need shared_ptr here
// (for structural sharing between path-copied versions during branching
// search), not the leaf VALUES.
struct LocalCtxNode; // forward-declared for self-reference in LocalCtxInternalSlots
using LocalCtxLeafSlots = std::array<NodeId, kEnvBranch>;
using LocalCtxInternalSlots = std::array<std::shared_ptr<const LocalCtxNode>, kEnvBranch>;

struct LocalCtxNode {
    std::variant<LocalCtxLeafSlots, LocalCtxInternalSlots> slots;
};

struct LocalCtx {
    std::shared_ptr<const LocalCtxNode> root; // nullptr when empty
    uint32_t count = 0;
    uint32_t shift = 0; // 0 when root is a leaf; +kEnvBits per additional tree level

    uint32_t size() const { return count; }

    // Bodies below mirror Env::operator[]/Env::extend exactly (including
    // the fresh-spine-vs-path-copy split, for the identical reason: a
    // brand new subtree that nothing else can reference yet doesn't need
    // copy-on-write, only a genuinely shared one does). NodeId's
    // completeness at this point in the file (unlike Value) is exactly
    // why LocalCtx doesn't need the forward-declare-then-define-later
    // split Env requires -- everything here can live in one place.
    NodeId operator[](uint32_t level) const;
    LocalCtx extend(NodeId ty) const;
};

namespace detail {

inline std::shared_ptr<const LocalCtxNode> localctx_build_fresh_spine(uint32_t sh, uint32_t idx, NodeId ty) {
    auto node = std::make_shared<LocalCtxNode>();
    if (sh == 0) {
        node->slots = LocalCtxLeafSlots{};
        std::get<LocalCtxLeafSlots>(node->slots)[idx & (kEnvBranch - 1)] = ty;
    } else {
        node->slots = LocalCtxInternalSlots{};
        uint32_t slot = (idx >> sh) & (kEnvBranch - 1);
        std::get<LocalCtxInternalSlots>(node->slots)[slot] = localctx_build_fresh_spine(sh - kEnvBits, idx, ty);
    }
    return node;
}

inline std::shared_ptr<const LocalCtxNode> localctx_path_copy_insert(
    const std::shared_ptr<const LocalCtxNode>& node, uint32_t sh, uint32_t idx, NodeId ty) {
    auto copy = std::make_shared<LocalCtxNode>(*node); // duplicate only this level
    if (sh == 0) {
        std::get<LocalCtxLeafSlots>(copy->slots)[idx & (kEnvBranch - 1)] = ty;
    } else {
        uint32_t slot = (idx >> sh) & (kEnvBranch - 1);
        auto& children = std::get<LocalCtxInternalSlots>(copy->slots);
        children[slot] = children[slot]
            ? localctx_path_copy_insert(children[slot], sh - kEnvBits, idx, ty)
            : localctx_build_fresh_spine(sh - kEnvBits, idx, ty);
    }
    return copy;
}

} // namespace detail

inline NodeId LocalCtx::operator[](uint32_t level) const {
    if (level >= count) throw std::logic_error("LocalCtx::operator[]: level out of range");
    const LocalCtxNode* cur = root.get();
    uint32_t sh = shift;
    while (sh > 0) {
        cur = std::get<LocalCtxInternalSlots>(cur->slots)[(level >> sh) & (kEnvBranch - 1)].get();
        sh -= kEnvBits;
    }
    return std::get<LocalCtxLeafSlots>(cur->slots)[level & (kEnvBranch - 1)];
}

inline LocalCtx LocalCtx::extend(NodeId ty) const {
    if (!root) {
        auto leaf = std::make_shared<LocalCtxNode>();
        leaf->slots = LocalCtxLeafSlots{};
        std::get<LocalCtxLeafSlots>(leaf->slots)[0] = ty;
        return LocalCtx{leaf, 1, 0};
    }
    uint32_t capacity = kEnvBranch << shift;
    if (count < capacity) {
        return LocalCtx{detail::localctx_path_copy_insert(root, shift, count, ty), count + 1, shift};
    }
    uint32_t new_shift = shift + kEnvBits;
    auto new_root = std::make_shared<LocalCtxNode>();
    new_root->slots = LocalCtxInternalSlots{};
    std::get<LocalCtxInternalSlots>(new_root->slots)[0] = root;
    uint32_t slot_for_new = (count >> new_shift) & (kEnvBranch - 1);
    std::get<LocalCtxInternalSlots>(new_root->slots)[slot_for_new] =
        detail::localctx_build_fresh_spine(new_shift - kEnvBits, count, ty);
    return LocalCtx{new_root, count + 1, new_shift};
}


/// Identifies what a VNeutral is stuck on: a locally bound variable, an
/// opaque global (an axiom, or an under-saturated constructor), or an
/// unsolved metavariable. These are kept distinct so that, for example, a
/// local variable and a global constant that happen to share a numeric id
/// are never treated as convertible.
struct NeutralHead {
    enum class Kind { LocalVar, GlobalConst, Meta } kind;
    uint32_t id; // level if LocalVar, global_id if GlobalConst, meta_id if Meta
    friend bool operator==(NeutralHead, NeutralHead) = default;
};

struct VType {
    uint32_t universe_level;
};

/// A stuck term: a variable, opaque constant, or unsolved metavariable,
/// with a spine of pending eliminators. The spine holds ApplyFrame for
/// pending applications, MatchFrame for a pending case split, and
/// QuotLiftFrame for a pending quotient lift, since each needs to be
/// reconstructed and compared differently by quote and conv_value.
struct VNeutral {
    NeutralHead head;
    std::vector<EliminatorFrame> spine;
};

struct Closure {
    Env env;
    NodeId dom_id;
    NodeId body_id;
};

struct VPi {
    Env env;
    NodeId dom_id;
    NodeId cod_id;
};

/// A fully saturated constructor application.
///
/// A custom destructor is required here, not the compiler-generated
/// default: `args` can itself contain another VCtor (a long list or a
/// large unary number, once evaluated into actual Values rather than
/// left as syntax, is a genuine chain of VCtors each owning the next by
/// value). The default recursive destructor -- each VCtor's destruction
/// destroying its args vector, which destroys the nested VCtor, which
/// destroys ITS args, and so on -- walks that chain one C++ stack frame
/// per element. Confirmed by direct measurement, not just argument: a
/// 100,000-deep chain survives destruction; a 500,000-deep one segfaults,
/// specifically during destruction and not during construction or
/// evaluation (those were tested and ruled out separately). The fix
/// flattens the chain into an explicit, heap-allocated worklist and
/// destroys it iteratively instead of relying on the C++ call stack.
///
/// This addresses VCtor chains specifically, which is the common,
/// naturally-arising case (any inductively-defined data value more than
/// trivially deep is represented this way). It does not address every
/// theoretically possible route to a similar cascade -- for example, a
/// long chain of VQuotType/VQuotMk each wrapping the next via their own
/// shared_ptr<Value> field -- since that pattern doesn't arise from
/// ordinary evaluation the way VCtor chains do; if a workload is ever
/// found that stresses it, it would need the same treatment, applied
/// there specifically.
struct VCtor {
    uint32_t ind_type_global_id;
    uint32_t ctor_idx;
    std::vector<Value> args;

    // All special members are DECLARED here but DEFINED later in the
    // file, after Value's alternatives (VQuotType/VQuotMk in particular)
    // are complete -- mirroring the same forward-declare-then-define-
    // later pattern already used for Env's own methods below. This is
    // load-bearing, not stylistic: VCtor is itself one of Value's
    // alternatives, defined before VQuotType/VQuotMk in this file, and
    // the destructor's body performs real operations on
    // std::vector<Value> (push_back, pop_back) to iteratively flatten a
    // nested chain rather than let it recurse through the default
    // destructor (see the comment on that definition, further down,
    // for why). Because VCtor's members are ordinary (non-template)
    // inline member functions, their bodies are type-checked immediately
    // at the point they're written -- unlike eval()/quote()/Env's own
    // methods, which are function TEMPLATES-in-spirit only in the sense
    // that inline free functions get the same treatment as long as their
    // declaration and definition are split. A defaulted comparison,
    // copy, or move here would trip the identical problem (checking a
    // defaulted operation's validity touches std::vector<Value>'s own
    // operations, which need Value complete) -- confirmed directly: the
    // very first version of this fix defaulted just the comparison
    // operator and moved on, and hit the same "incomplete type"
    // error, just relocated to the very next special member down. Every
    // member that could plausibly touch Value is deferred here, not just
    // the one that happened to be checked first.
    VCtor() = default; // does not touch args at all; safe to default in place
    VCtor(uint32_t ind, uint32_t ctor, std::vector<Value> a);
    VCtor(const VCtor&);
    VCtor(VCtor&&) noexcept;
    VCtor& operator=(const VCtor&);
    VCtor& operator=(VCtor&&) noexcept;
    ~VCtor(); // no operator==: Value-domain structs are never compared via == anywhere
    // in this codebase -- semantic comparison always goes through conv_value, and
    // Value-domain values are never hash-consed (unlike the syntax/arena node
    // structs above, which need operator== for NodePool::intern's equality
    // check). Adding one here was a reflexive copy of that unrelated pattern;
    // nothing calls it, and completing it correctly would require adding
    // operator== to every other Value alternative too, for no actual benefit.
};

/// The evaluated form of a quotient type former, `Quot r`. `relation` is
/// evaluated eagerly (unlike a Match branch, there is no benefit to
/// deferring it), so conv_value can compare two quotient types by
/// comparing their relations directly.
struct VQuotType {
    std::shared_ptr<Value> relation;
};

/// The evaluated form of `Quot.mk r a`: a canonical, "already reduced"
/// quotient element, analogous to VCtor.
struct VQuotMk {
    std::shared_ptr<Value> relation;
    std::shared_ptr<Value> elem;
};

/// The evaluated form of a Fix node: "the fixpoint, ready to be
/// applied" -- deliberately NOT unfolded here. `fix_id` is the raw
/// syntax of the ORIGINAL Fix node (not its body) and `env` is the
/// ambient environment active at the point the Fix was evaluated,
/// mirroring Closure's own (dom_id/body_id, env) shape exactly. Holding
/// `fix_id` rather than eagerly splitting it into its constituent
/// fn_type/decreasing_arg/body fields keeps quote(VFix) a trivial,
/// already-valid-syntax return (see quote()'s VFix case below) and
/// keeps this struct requiring nothing beyond what Closure already
/// needs -- no self-referential shared_ptr<Value>, no two-step
/// construction. The self-reference is threaded through lazily, once,
/// at the point of application -- see apply()'s VFix case.
struct VFix {
    NodeId fix_id;
    Env env;
};

struct ApplyFrame {
    Value arg;
};

/// A case split that was still pending when its scrutinee got stuck.
/// `captured_env` is the environment active at that point, needed to
/// re-evaluate the branches if the term later becomes unstuck, and to
/// compare two pending matches in conv_value.
struct MatchFrame {
    NodeId motive_id;
    std::vector<NodeId> branches;
    Env captured_env;
};

/// A quotient lift that was still pending when its argument got stuck.
///
/// All four fields are kept as raw, UNEVALUATED syntax (NodeId), not
/// eagerly evaluated Values, mirroring MatchFrame's motive_id/branches
/// exactly. An earlier version of this struct evaluated relation/
/// codomain/f/respects_proof eagerly the moment a QuotLift got stuck --
/// wasted work specifically for respects_proof, which conv_value below
/// never even inspects (proof-irrelevance-in-spirit), and unnecessary for
/// the others too, since quote() only needs to reconstruct them as
/// syntax, not evaluate them, exactly as MatchFrame's motive_id/branches
/// already didn't need evaluating. `captured_env` is still required,
/// for the same soundness reason MatchFrame needs it: comparing the raw
/// NodeIds alone isn't sound if they reference free variables that could
/// resolve differently under different ambient environments.
struct QuotLiftFrame {
    NodeId relation_id;
    NodeId codomain_id;
    NodeId f_id;
    NodeId respects_proof_id;
    Env captured_env;
};

// Env's method bodies, and its two path-copying helpers, are defined here
// -- after Value's alternatives are complete (needed for make_shared<const
// Value>), before eval()/quote()/conv_value() are defined below (which
// need Env's interface usable).

namespace detail {

/// Builds a brand-new (never-shared) subtree, from `sh` down to a leaf,
/// containing exactly one value at `idx`. Every node built here is
/// exclusively owned at construction time (nothing else could reference
/// it yet), so this never needs to copy-on-write -- distinct from
/// path_copy_insert below, which handles the existing, possibly-shared
/// tree. Conflating these two cases (routing a freshly allocated node
/// through the copy-on-write path anyway) was a real bug in an earlier
/// draft of this design: it allocated a node, then immediately allocated
/// a second one to "copy" the first before ever sharing it, discarding
/// the first allocation for no reason.
inline std::shared_ptr<const EnvNode> env_build_fresh_spine(uint32_t sh, uint32_t idx, Value v) {
    auto node = std::make_shared<EnvNode>(); // mutable shared_ptr<EnvNode>; no const_cast needed anywhere here
    if (sh == 0) {
        node->slots = EnvLeafSlots{};
        std::get<EnvLeafSlots>(node->slots)[idx & (kEnvBranch - 1)] = std::make_shared<const Value>(std::move(v));
    } else {
        node->slots = EnvInternalSlots{};
        uint32_t slot = (idx >> sh) & (kEnvBranch - 1);
        std::get<EnvInternalSlots>(node->slots)[slot] = env_build_fresh_spine(sh - kEnvBits, idx, std::move(v));
    }
    return node; // implicit, safe shared_ptr<EnvNode> -> shared_ptr<const EnvNode> conversion
}

/// Path-copies the spine from `node` down to index `idx`, duplicating
/// only the nodes along that one path (everything else is shared by
/// reference with the pre-extension tree, which is what makes multiple
/// live Env values safely coexist).
inline std::shared_ptr<const EnvNode> env_path_copy_insert(
    const std::shared_ptr<const EnvNode>& node, uint32_t sh, uint32_t idx, Value v) {
    auto copy = std::make_shared<EnvNode>(*node); // duplicate only this level
    if (sh == 0) {
        std::get<EnvLeafSlots>(copy->slots)[idx & (kEnvBranch - 1)] = std::make_shared<const Value>(std::move(v));
    } else {
        uint32_t slot = (idx >> sh) & (kEnvBranch - 1);
        auto& children = std::get<EnvInternalSlots>(copy->slots);
        children[slot] = children[slot]
            ? env_path_copy_insert(children[slot], sh - kEnvBits, idx, std::move(v))
            : env_build_fresh_spine(sh - kEnvBits, idx, std::move(v)); // genuinely new territory: no copy needed
    }
    return copy;
}

} // namespace detail

inline const Value& Env::operator[](uint32_t level) const {
    if (level >= count) {
        std::fprintf(stderr, "DBG Env::operator[] OUT OF BOUNDS: level=%u count=%u\n", level, count);
        std::abort();
    }
    const EnvNode* cur = root.get();
    uint32_t sh = shift;
    while (sh > 0) {
        cur = std::get<EnvInternalSlots>(cur->slots)[(level >> sh) & (kEnvBranch - 1)].get();
        sh -= kEnvBits;
    }
    return *std::get<EnvLeafSlots>(cur->slots)[level & (kEnvBranch - 1)];
}

inline Env Env::extend(Value v) const {
    if (!root) {
        auto leaf = std::make_shared<EnvNode>();
        leaf->slots = EnvLeafSlots{};
        std::get<EnvLeafSlots>(leaf->slots)[0] = std::make_shared<const Value>(std::move(v));
        return Env{leaf, 1, 0};
    }
    uint32_t capacity = kEnvBranch << shift;
    if (count < capacity) {
        // Existing tree may be shared by other Env values still
        // referencing the pre-extension version: must copy-on-write.
        return Env{detail::env_path_copy_insert(root, shift, count, std::move(v)), count + 1, shift};
    }
    // Full: grow one level taller. The new root and the entire fresh
    // subtree beneath it (down to the new leaf) are exclusively ours --
    // nothing else can reference any of it yet -- so the whole spine is
    // built via env_build_fresh_spine, not routed through the copy-on-
    // write path meant for the (possibly-shared) existing tree.
    uint32_t new_shift = shift + kEnvBits;
    auto new_root = std::make_shared<EnvNode>();
    new_root->slots = EnvInternalSlots{};
    std::get<EnvInternalSlots>(new_root->slots)[0] = root; // old (possibly-shared) subtree: referenced, not copied
    uint32_t slot_for_new = (count >> new_shift) & (kEnvBranch - 1);
    std::get<EnvInternalSlots>(new_root->slots)[slot_for_new] =
        detail::env_build_fresh_spine(new_shift - kEnvBits, count, std::move(v));
    return Env{new_root, count + 1, new_shift};
}

// VCtor's special members, defined here (not inline in the class body
// above) because their bodies need Value complete -- see the comment on
// the class definition for why. This is the exact same defer-until-
// Value-is-complete pattern Env's own methods use just above.
inline VCtor::VCtor(uint32_t ind, uint32_t ctor, std::vector<Value> a)
    : ind_type_global_id(ind), ctor_idx(ctor), args(std::move(a)) {}
inline VCtor::VCtor(const VCtor&) = default;
inline VCtor::VCtor(VCtor&&) noexcept = default;
inline VCtor& VCtor::operator=(const VCtor&) = default;
inline VCtor& VCtor::operator=(VCtor&&) noexcept = default;
// A custom destructor is required here, not the compiler-generated
// default: `args` can itself contain another VCtor (a long list or a
// large unary number, once evaluated into actual Values rather than left
// as syntax, is a genuine chain of VCtors each owning the next by
// value). The default recursive destructor -- each VCtor's destruction
// destroying its args vector, which destroys the nested VCtor, which
// destroys ITS args, and so on -- walks that chain one C++ stack frame
// per element. Confirmed by direct measurement, not just argument: a
// 100,000-deep chain survives destruction; a 500,000-deep one segfaults,
// specifically during destruction and not during construction or
// evaluation (those were tested and ruled out separately). The fix
// flattens the chain into an explicit, heap-allocated worklist and
// destroys it iteratively instead of relying on the C++ call stack.
//
// This addresses VCtor chains specifically, which is the common,
// naturally-arising case (any inductively-defined data value more than
// trivially deep is represented this way). It does not address every
// theoretically possible route to a similar cascade -- for example, a
// long chain of VQuotType/VQuotMk each wrapping the next via their own
// shared_ptr<Value> field -- since that pattern doesn't arise from
// ordinary evaluation the way VCtor chains do; if a workload is ever
// found that stresses it, it would need the same treatment, applied
// there specifically.
inline VCtor::~VCtor() {
    if (args.empty()) return; // common case: nothing to flatten, avoid the worklist entirely
    std::vector<Value> worklist;
    for (auto& v : args) worklist.push_back(std::move(v));
    args.clear();
    while (!worklist.empty()) {
        Value v = std::move(worklist.back());
        worklist.pop_back();
        if (auto* vc = std::get_if<VCtor>(&v)) {
            for (auto& child : vc->args) worklist.push_back(std::move(child));
            vc->args.clear();
        }
        // v destructs here at end of scope; any nested VCtor content
        // was already flattened into the worklist above, so this is
        // now a cheap, non-recursive destruction regardless of how
        // deep the original chain was.
    }
}

Value eval(NodeId term, const Env& env);
NodeId quote(const Value& v, uint32_t depth);
NodeId requote_captured_syntax(NodeId term, const Env& env, uint32_t depth);
NodeId resolve_metas(NodeId term);
bool conv_value(const Value& a, const Value& b, uint32_t depth);
struct DiagnosticTable;
std::string print(NodeId id, const DiagnosticTable* diag);
namespace detail {
NodeId substitute_var(NodeId g, uint32_t target_level, NodeId replacement);
NodeId substitute_vars(NodeId g, const std::unordered_map<uint32_t, NodeId>& replacements);
NodeId shift_levels_from(NodeId t, uint32_t threshold, uint32_t amount);
NodeId substitute_vars_relevel(NodeId g, const std::unordered_map<uint32_t, NodeId>& replacements, uint32_t base, uint32_t j);
}
// eval, quote, and conv_value recurse without any explicit stack-depth
// protection -- unrelated to the environment representation above, and
// still unaddressed. A trampoline or explicit stack would remove this,
// at the cost of real complexity; a reasonable follow-up once there is a
// concrete workload that needs it.

/// The Value-domain counterpart of inspect, for the same reason: it lets
/// code that pattern-matches on evaluated terms use the same match_on-
/// based dispatch as syntactic code, instead of a chain of std::get_if.
template <typename... Visitors>
auto inspect_value(const Value& v, Visitors... vs) {
    return std::visit(match_on{vs...}, v);
}
template <typename... Visitors>
auto inspect_value(const Value& a, const Value& b, Visitors... vs) {
    return std::visit(match_on{vs...}, a, b);
}

inline Value apply(Value fn, Value arg) {
    return inspect_value(fn,
        [&](Closure clo) -> Value {
            return eval(clo.body_id, clo.env.extend(std::move(arg)));
        },
        [&](VNeutral n) -> Value {
            // A constructor reference that becomes fully saturated by this
            // application is promoted from a neutral to a VCtor.
            n.spine.push_back(EliminatorFrame{ApplyFrame{arg}});
            if (n.head.kind == NeutralHead::Kind::GlobalConst) {
                const GlobalDef& def = ctx().globals[n.head.id];
                if (def.ctor_info) {
                    uint32_t applied = 0;
                    std::vector<Value> ctor_args;
                    for (const auto& frame : n.spine) {
                        if (auto* af = std::get_if<ApplyFrame>(&frame)) {
                            ctor_args.push_back(af->arg);
                            ++applied;
                        }
                    }
                    if (applied == def.ctor_info->arity) {
                        return Value{VCtor{def.ctor_info->ind_type_global_id, def.ctor_info->ctor_idx,
                                           std::move(ctor_args)}};
                    }
                }
            }
            return Value{std::move(n)};
        },
        [&](VFix vf) -> Value {
            // Unfolds exactly ONE layer, the local analogue of Const's
            // own self-reference story (see GlobalEnv's comment): extend
            // the Fix's OWN captured env with a reference back to the
            // Fix's own value (another VFix, wrapping the identical
            // fix_id/env -- not a genuinely self-referential Value,
            // since looking THAT up and applying it again simply
            // reproduces this exact case, one call at a time), evaluate
            // the body in that extended env, then apply the result to
            // `arg`. The self-reference lands at level vf.env.size() --
            // exactly the level a construction site (Induct) must
            // address it at, the same "new binder takes env.size()"
            // convention every other binder in this file already
            // follows.
            Fix f = std::get<Fix>(realize(vf.fix_id));
            Env self_env = vf.env.extend(Value{VFix{vf.fix_id, vf.env}});
            Value body_val = eval(f.body, self_env);
            return apply(std::move(body_val), std::move(arg));
        },
        [&](auto) -> Value {
            // Applying a non-function value indicates the term being
            // evaluated was not well-typed; this should not be reachable
            // for input that has already passed a correct typechecker.
            throw std::logic_error("itp: apply() on a non-function value -- ill-typed term reached eval");
        }
    );
}

/// No-delta evaluation scope (Findings F2a/F4/F5). While a NoDeltaScope is alive on this thread,
/// eval() leaves global constants FOLDED (a neutral GlobalConst) instead of delta-unfolding their
/// bodies, while beta, iota (Match on a constructor) and local Fix behaviour are unchanged.
/// Use ONLY for normalisation whose output is stored or displayed (proof extraction, goal and
/// hypothesis normalisation). Never around conversion checking: definitional equality needs delta.
/// Why it exists: quote(eval(t)) unfolds every definition it can reach, so an extracted proof
/// inlined library theorems such as `transport` (whose bodies carry binders that then no longer
/// line up), and a normalised goal lost its flat shape (see intro()'s own long comment).
inline bool& no_delta_flag() { static thread_local bool f = false; return f; }
struct NoDeltaScope {
    bool prev;
    NoDeltaScope() : prev(no_delta_flag()) { no_delta_flag() = true; }
    ~NoDeltaScope() { no_delta_flag() = prev; }
    NoDeltaScope(const NoDeltaScope&) = delete;
    NoDeltaScope& operator=(const NoDeltaScope&) = delete;
};

inline Value eval(NodeId term, const Env& env) {
    return inspect(term,
        [&](Var v) -> Value {
            return v.bound ? env[env.size() - 1 - v.n] : env[v.n]; // bound: index from the END of env; context: level from its start
        },
        [&](App a) -> Value {
            return apply(eval(a.fn_id, env), eval(a.arg_id, env));
        },
        [&](Lambda l) -> Value {
            return Value{Closure{env, l.type_id, l.body_id}};
        },
        [&](Let l) -> Value {
            return eval(l.body_id, env.extend(eval(l.value_id, env)));
        },
        [&](Pi p) -> Value {
            return Value{VPi{env, p.dom_id, p.cod_id}};
        },
        [&](Type t) -> Value {
            return Value{VType{t.universe_level}};
        },
        [&](Const c) -> Value {
            const GlobalDef& def = ctx().globals[c.global_id];
            if (def.ctor_info && def.ctor_info->arity == 0) {
                // A nullary constructor is already saturated.
                return Value{VCtor{def.ctor_info->ind_type_global_id, def.ctor_info->ctor_idx, {}}};
            }
            if (def.value && !no_delta_flag()) {
                // Delta-reduction: definitions unfold transparently.
                // Definitions are required to be closed, so they are
                // evaluated in an empty environment. (Skipped inside a
                // NoDeltaScope: the constant then stays a folded neutral.)
                return eval(*def.value, Env{});
            }
            return Value{VNeutral{NeutralHead{NeutralHead::Kind::GlobalConst, c.global_id}, {}}};
        },
        [&](Match m) -> Value {
            Value scrutinee = eval(m.scrutinee_id, env);
            return inspect_value(scrutinee,
                [&](const VCtor& vc) -> Value {
                    // Evaluate the selected branch and apply it to the
                    // constructor's stored arguments in turn.
                    NodeId branch = m.branches->at(vc.ctor_idx);
                    Value result = eval(branch, env);
                    for (const Value& a : vc.args) result = apply(result, a);
                    return result;
                },
                [&](const VNeutral& n) -> Value {
                    // The scrutinee is stuck, so the match itself becomes
                    // a new frame on its spine.
                    VNeutral result = n;
                    result.spine.push_back(EliminatorFrame{MatchFrame{m.motive_id, *m.branches, env}});
                    return Value{result};
                },
                [&](const auto&) -> Value {
                    throw std::logic_error(
                        "itp: match on a non-inductive, non-neutral value -- ill-typed term reached eval");
                }
            );
        },
        [&](Meta meta) -> Value {
            const MetaEntry& e = ctx().meta_ctx[meta.meta_id];
            if (e.solution) {
                // As with a solved definition, the solution is expected to
                // be closed with respect to the metavariable's own local
                // context; a real elaborator that wants a solution to
                // depend on local variables is expected to close over
                // them explicitly (applying the metavariable to its
                // dependencies at each occurrence) rather than relying on
                // an ambient environment here.
                return eval(*e.solution, Env{});
            }
            return Value{VNeutral{NeutralHead{NeutralHead::Kind::Meta, meta.meta_id}, {}}};
        },
        [&](TypeUniv) -> Value {
            throw std::logic_error(
                "itp: encountered an uninstantiated universe parameter during evaluation -- "
                "did you forget to call instantiate_global?");
        },
        [&](Fix) -> Value {
            // Lazy: a Fix does NOT unfold its self-reference here, the
            // same way a Lambda's body is not forced by evaluating the
            // Lambda itself. It only reduces one layer when actually
            // applied -- see apply()'s VFix case below, which mirrors
            // exactly how a Const-based recursive definition (see
            // GlobalEnv's comment) only unfolds its self-reference when
            // looked up as part of an application, not at declaration
            // time. `env` here is the AMBIENT environment (whatever
            // hypotheses were already in scope where this Fix occurs),
            // not yet extended with a self-reference -- that extension
            // happens lazily, once, inside apply()'s VFix case, so a
            // Fix that is never applied never pays for it and never
            // risks the circular-construction problem a genuinely
            // self-referential Value would have.
            return Value{VFix{term, env}};
        },
        [&](QuotType q) -> Value {
            return Value{VQuotType{std::make_shared<Value>(eval(q.relation, env))}};
        },
        [&](QuotMk q) -> Value {
            return Value{VQuotMk{std::make_shared<Value>(eval(q.relation, env)),
                                  std::make_shared<Value>(eval(q.elem, env))}};
        },
        [&](QuotLift q) -> Value {
            Value scrutinee = eval(q.quotient_elem, env);
            return inspect_value(scrutinee,
                [&](const VQuotMk& mk) -> Value {
                    // The computation rule: Quot.lift f h (Quot.mk r a)
                    // reduces to f a. respects_proof is not needed here;
                    // it exists purely for typechecking.
                    Value f_val = eval(q.f, env);
                    return apply(f_val, *mk.elem);
                },
                [&](const VNeutral& n) -> Value {
                    // Stored as raw, unevaluated syntax (plus the
                    // ambient env), not eagerly evaluated Values -- see
                    // QuotLiftFrame's comment. This was previously eager
                    // (eval(q.relation, env), eval(q.codomain, env), etc.
                    // computed here every time a QuotLift got stuck),
                    // wasting real work normalizing content -- especially
                    // respects_proof -- that conv_value never inspects.
                    VNeutral result = n;
                    result.spine.push_back(EliminatorFrame{
                        QuotLiftFrame{q.relation, q.codomain, q.f, q.respects_proof, env}});
                    return Value{result};
                },
                [&](const auto&) -> Value {
                    throw std::logic_error(
                        "itp: Quot.lift applied to a non-quotient, non-neutral value -- ill-typed term reached eval");
                }
            );
        }
    );
}

inline Value fresh_neutral(uint32_t level) {
    return Value{VNeutral{NeutralHead{NeutralHead::Kind::LocalVar, level}, {}}};
}

/// Builds an environment representing N ambient hypotheses whose values
/// are not yet known, for typechecking a term under a context of that
/// size. Genuinely free variables should be represented this way rather
/// than with placeholder concrete values.
inline Env open_context(uint32_t n) {
    Env env;
    for (uint32_t i = 0; i < n; ++i) env = env.extend(fresh_neutral(i));
    return env;
}

#if 0 // not yet ported to the hybrid variable representation: shift_levels_down_from
/// Converts a semantic value back into syntax.
///
/// Precondition: `depth` must equal the length of the environment that was
/// active when `v` was produced. Prefer normalize() below, which derives
/// this automatically from the corresponding (term, env) pair and cannot
/// get it wrong; call quote() directly only when no such pair is
/// available.
/// Re-levels a piece of RAW SYNTAX that was captured relative to some
/// (now-inaccessible) evaluation environment `env`, producing syntax
/// valid at the current quoting depth `depth` -- WITHOUT evaluating any
/// of the syntax's own operators (App, Match, ...) along the way. Used
/// exclusively by quote()'s own MatchFrame/QuotLiftFrame-style stuck-
/// eliminator reconstruction (see quote(VNeutral)'s own design note),
/// where `term` is a global recursive function's OWN branch/motive
/// syntax, written relative to THAT function's internal parameter
/// numbering, and `env` is the captured environment active when the
/// enclosing Match got stuck.
///
/// Each bare `Var(k)` is handled one of two ways: if `k < env.size()`,
/// it refers to something ALREADY BOUND when the eliminator got stuck
/// (e.g. a preceding curried argument like `add`'s own `m`) and is
/// replaced by `quote(env[k], depth)` -- the value's own already-
/// computed quoted form, itself either a stable absolute-level Var
/// (for a value that was itself a fresh neutral) or genuinely concrete
/// syntax, needing no further adjustment as more binders are entered
/// during this same walk, since every level this file works with is
/// already absolute, not binder-relative. Otherwise (`k >= env.size()`)
/// it is a genuinely NEW binder introduced within `term` itself (e.g.
/// a recursive branch's own field parameter) and is re-based to land at
/// `depth + (k - env.size())`, matching where that binder will actually
/// sit once `term` is re-embedded starting at `depth`.
///
/// Deliberately NOT eval()+quote(): forcing evaluation of `term`'s own
/// body would, for any branch that calls the same recursive function
/// again, immediately re-trigger that call -- and since its argument is
/// still a fresh neutral (never concrete under an open binder), this
/// produces another stuck Match one level deeper, whose own branches
/// would need the identical treatment, forever. A stuck elimination's
/// correct normal form leaves an embedded recursive call as symbolic,
/// unevaluated syntax; a pure syntactic substitution does exactly that,
/// while eval()+quote() does not -- confirmed by this session's own
/// Induct end-to-end test hanging under the eval()+quote() version and
/// terminating correctly under this one.
/// Recursively replaces every SOLVED metavariable within `term` with its
/// own solution, itself resolved recursively in turn (a solution can
/// embed a further meta reference -- cases()'s own ref_meta always does,
/// since its solution embeds subgoal_meta, the goal the agent actually
/// goes on to solve) -- leaving an UNSOLVED meta, and everything else,
/// completely unchanged.
///
/// A solved meta's own solution is already fully closed, absolute-level
/// syntax (see eval()'s own Meta case doc comment: solve() only ever
/// stores a term with no free variables of its own, produced via
/// close_over_context or the equivalent hand-rolled loop), so splicing
/// it in needs no re-leveling at all -- deliberately NOT the same
/// concern requote_captured_syntax's own Var case handles (a captured
/// Fix/Match-branch's OWN internal, RELATIVE parameter numbering, needing
/// to be re-based against the enclosing env/depth this function knows
/// nothing about and does not replicate).
///
/// Used specifically by requote_captured_syntax's own Meta case, below:
/// a stuck Match's own branches are re-quoted there via a PURE SYNTACTIC
/// walk (never eval()+quote(), to avoid forcing open a recursive Fix
/// call -- see that function's own doc comment), which means a solved
/// meta reference embedded in one of those branches would otherwise be
/// left completely unresolved in the final, extracted proof term --
/// confirmed directly, this session, by whole_term_check() rejecting an
/// otherwise fully tactic-solved trans proof ("mismatched value
/// shapes"), the extracted term showing an unresolved metavariable
/// still literally embedded in its one match branch, despite that
/// meta's own solve() having already, correctly, been called and taken
/// effect (is_solved() true) well before extraction.
/// Shifts every level >= threshold DOWN by `amount` (level -> level -
/// amount), leaving anything below threshold untouched. The mirror
/// image of shift_levels_from (defined later in this file, once
/// NodeType's full variant set is available there); duplicated here,
/// rather than forward-declared and reused, since the two functions
/// are simple, self-contained, and this one is only ever needed by
/// resolve_metas' own callers, immediately below.
///
/// Used specifically when re-anchoring a solved meta's own solution
/// whose own closed context (g.ec, at the point the meta was created)
/// was LARGER than however many binders that solution's own
/// application spine actually corresponds to once beta-reduced at the
/// true-runtime level. Concretely: induct()'s own subgoal_meta is
/// declared over its own, full AGENT-FACING branch_ec, which includes
/// an induction-hypothesis entry -- but the induction hypothesis has
/// no separate Lambda binder at all in the actual, extracted proof
/// term (it is computed inline, as an ordinary subterm, "self applied
/// to the recursive field" -- see induct()'s own comment on this). So
/// a genuinely NEW binder the solution's own body introduces (e.g. a
/// further, nested match embedded within one of the solved
/// arguments), past its own closed context's own size, was originally
/// numbered assuming that "IH slot" as if it were a real, consumed
/// binder -- but once actually beta-reduced against the true-runtime
/// arguments (which never supply that slot as a separate application
/// at all), the new binder's own, correct level is smaller than its
/// own, original numbering assumed, by exactly the count of such
/// agent-facing-only slots. Confirmed directly, this session: a
/// nested match's own, freshly-introduced field ended up referencing
/// an out-of-scope level (one past the true-runtime context's own
/// size) once this was not accounted for, causing "unbound local
/// variable" during whole_term_check's own, independent re-
/// elaboration of an Induct-based proof -- itself found only after an
/// earlier fix (this same function's own sibling, the beta-reduction
/// added to resolve_metas' and requote_captured_syntax's own App
/// cases) resolved the prior, different-looking "different variable/
/// constant heads" symptom this same root cause was also responsible
/// for.
inline NodeId shift_levels_down_from(NodeId t, uint32_t threshold, uint32_t amount) {
    if (amount == 0) return t;
    return inspect(t,
        [&](Var v) -> NodeId { return v.level >= threshold ? mk_var(v.level - amount) : mk_var(v.level); },
        [&](App a) -> NodeId {
            return mk_app(shift_levels_down_from(a.fn_id, threshold, amount), shift_levels_down_from(a.arg_id, threshold, amount));
        },
        [&](Lambda l) -> NodeId {
            return mk_lambda(shift_levels_down_from(l.type_id, threshold, amount), shift_levels_down_from(l.body_id, threshold, amount));
        },
        [&](Let l) -> NodeId {
            return mk_let(shift_levels_down_from(l.type_id, threshold, amount), shift_levels_down_from(l.value_id, threshold, amount),
                           shift_levels_down_from(l.body_id, threshold, amount));
        },
        [&](Pi p) -> NodeId {
            return mk_pi(shift_levels_down_from(p.dom_id, threshold, amount), shift_levels_down_from(p.cod_id, threshold, amount));
        },
        [&](Type ty) { return mk_type(ty.universe_level); },
        [&](Const c) { return mk_const(c.global_id); },
        [&](Match m) -> NodeId {
            std::vector<NodeId> branches;
            branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) branches.push_back(shift_levels_down_from(b, threshold, amount));
            return mk_match(shift_levels_down_from(m.scrutinee_id, threshold, amount),
                             shift_levels_down_from(m.motive_id, threshold, amount), std::move(branches));
        },
        [&](Meta meta) { return mk_meta(meta.meta_id); },
        [&](TypeUniv u) { return mk_type_univ(u.param_index); },
        [&](QuotType q) { return mk_quot_type(shift_levels_down_from(q.relation, threshold, amount)); },
        [&](QuotMk q) {
            return mk_quot_mk(shift_levels_down_from(q.relation, threshold, amount), shift_levels_down_from(q.elem, threshold, amount));
        },
        [&](QuotLift q) {
            return mk_quot_lift(shift_levels_down_from(q.relation, threshold, amount), shift_levels_down_from(q.codomain, threshold, amount),
                                 shift_levels_down_from(q.f, threshold, amount), shift_levels_down_from(q.respects_proof, threshold, amount),
                                 shift_levels_down_from(q.quotient_elem, threshold, amount));
        },
        [&](Fix f) {
            return mk_fix(shift_levels_down_from(f.fn_type, threshold, amount), f.decreasing_arg, shift_levels_down_from(f.body, threshold, amount));
        }
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: resolve_metas
inline NodeId resolve_metas(NodeId term) {
    return inspect(term,
        [&](Var v) { return mk_var(v.level); },
        [&](App a) -> NodeId {
            // Decompose the FULL application spine first (not just one
            // layer at a time, which the naive mk_app(resolve_metas(fn),
            // resolve_metas(arg)) reconstruction below does): if the
            // spine's own head is a solved meta, beta-reduce the WHOLE
            // thing directly via pure substitution, rather than
            // reconstructing an unreduced (Lambda-chain) applied to
            // (args) redex.
            //
            // This matters specifically because a solved meta's own
            // solution (built via close_over_context, a Lambda-chain
            // over the goal's own hypotheses at the time it was solved)
            // can itself embed a FURTHER, nested meta reference whose
            // own solution is just a bare variable -- e.g. "the
            // induction hypothesis itself, referenced directly", the
            // agent's own, correct way to discharge a recursive
            // subgoal that already has a hypothesis of exactly that
            // type in scope. If the OUTER application is left as an
            // unreduced redex, that inner meta's own solution --
            // spliced in here verbatim, since resolve_metas' own Meta
            // case (below) has no way to know it is embedded inside a
            // pending application at all -- keeps referencing its own,
            // original level (relative to the Lambda-chain's own,
            // not-yet-applied parameter list), which no longer means
            // anything correct once this whole structure is spliced in
            // deeper, inside an enclosing match's own branch (which
            // already has its own, unrelated bindings occupying those
            // same numeric levels). Nothing else in this function (or
            // requote_captured_syntax, which invokes it) ever re-
            // anchors a bare variable reference the way infer(Meta)'s
            // own shift_all_levels treatment would if it ever got the
            // chance to run -- it never does, since every meta is
            // always resolved away, via this very function, before
            // infer()/check() ever sees its bare application syntax.
            // Confirmed directly, this session: an induction
            // hypothesis's own value ended up silently referencing the
            // wrong level once embedded this way, passing every tactic-
            // level check at the point it was solved but failing
            // whole_term_check's own, independent re-elaboration with a
            // "different variable/constant heads" mismatch, traced
            // precisely to this gap.
            //
            // A purely syntactic substitution (detail::substitute_var,
            // never eval()) correctly, permanently resolves each
            // reference at the point of splicing, rather than leaving
            // it to be (incorrectly) re-interpreted later relative to
            // wherever this structure ends up embedded -- and carries
            // none of the "might force open a recursive Fix" risk
            // requote_captured_syntax's own, separate, deliberately
            // non-evaluating design exists to avoid, since substitution
            // never evaluates anything, only renames free occurrences
            // of one specific level.
            std::vector<NodeId> args;
            NodeId head = term;
            while (true) {
                bool matched = false;
                inspect(head, [&](App inner) { args.push_back(inner.arg_id); head = inner.fn_id; matched = true; },
                              [&](auto) {});
                if (!matched) break;
            }
            std::reverse(args.begin(), args.end());
            bool head_is_meta = false;
            uint32_t head_meta_id = 0;
            inspect(head, [&](Meta m) { head_is_meta = true; head_meta_id = m.meta_id; }, [&](auto) {});
            if (head_is_meta) {
                const MetaEntry& e = ctx().meta_ctx[head_meta_id];
                if (e.solution) {
                    NodeId result = resolve_metas(*e.solution);
                    // SIMULTANEOUS substitution (peel every Lambda layer first, collecting
                    // each resolved argument against its own position, then ONE
                    // substitute_vars pass) -- NOT the sequential peel-and-substitute loop
                    // this site originally used. Sequential substitution is only correct
                    // when no already-substituted argument contains a free variable whose
                    // level equals a LATER, still-pending position: such a variable is then
                    // wrongly re-substituted as if it had been part of the solution body
                    // all along. Confirmed to actually happen (see substitute_vars's own
                    // doc comment for the identical collision in induct()'s own per-
                    // constructor substitution, and requote_captured_syntax's own matching
                    // fix below, where le_trans's recursive call head -- `self`, re-embedded
                    // at level 9 -- was overwritten by the final argument slot's own value,
                    // position 9, producing a proof variable in function position). It never
                    // surfaced before because an induction hypothesis was always the LAST
                    // argument position, so nothing was substituted after it.
                    std::unordered_map<uint32_t, NodeId> sub;
                    std::size_t i = 0;
                    for (; i < args.size(); ++i) {
                        bool result_is_lambda = false;
                        NodeId lam_body{};
                        inspect(result, [&](Lambda l) { result_is_lambda = true; lam_body = l.body_id; },
                                        [&](auto) {});
                        if (!result_is_lambda) break;
                        sub[static_cast<uint32_t>(i)] = resolve_metas(args[i]);
                        result = lam_body;
                    }
                    result = detail::substitute_vars(result, sub);
                    // Fewer Lambda layers than arguments -- should not happen for a
                    // well-typed solution, but fall back to a plain application rather
                    // than silently dropping an argument.
                    for (; i < args.size(); ++i) result = mk_app(result, resolve_metas(args[i]));
                    return result;
                }
            }
            return mk_app(resolve_metas(a.fn_id), resolve_metas(a.arg_id));
        },
        [&](Lambda l) { return mk_lambda(resolve_metas(l.type_id), resolve_metas(l.body_id)); },
        [&](Let l) { return mk_let(resolve_metas(l.type_id), resolve_metas(l.value_id), resolve_metas(l.body_id)); },
        [&](Pi p) { return mk_pi(resolve_metas(p.dom_id), resolve_metas(p.cod_id)); },
        [&](Type t) { return mk_type(t.universe_level); },
        [&](Const c) { return mk_const(c.global_id); },
        [&](Match m) {
            std::vector<NodeId> branches;
            branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) branches.push_back(resolve_metas(b));
            return mk_match(resolve_metas(m.scrutinee_id), resolve_metas(m.motive_id), std::move(branches));
        },
        [&](Meta meta) -> NodeId {
            const MetaEntry& e = ctx().meta_ctx[meta.meta_id];
            if (e.solution) return resolve_metas(*e.solution);
            return mk_meta(meta.meta_id);
        },
        [&](TypeUniv u) { return mk_type_univ(u.param_index); },
        [&](QuotType q) { return mk_quot_type(resolve_metas(q.relation)); },
        [&](QuotMk q) { return mk_quot_mk(resolve_metas(q.relation), resolve_metas(q.elem)); },
        [&](QuotLift q) {
            return mk_quot_lift(resolve_metas(q.relation), resolve_metas(q.codomain), resolve_metas(q.f),
                                 resolve_metas(q.respects_proof), resolve_metas(q.quotient_elem));
        },
        [&](Fix f) { return mk_fix(resolve_metas(f.fn_type), f.decreasing_arg, resolve_metas(f.body)); }
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: resolve_metas_at
/// Depth-aware metavariable resolution (Finding F3, cause 3). `depth` is the number of binders in
/// scope at `term`'s position in the FINAL numbering. Unlike resolve_metas above, a spliced solution
/// is first re-anchored to the real embedding depth (its internal binders move from the solution's
/// own closed size n to `depth`, via the same two-hop SAFE_OFFSET shift requote_captured_syntax
/// uses) and the arguments are substituted; only THEN are metas nested inside the result resolved,
/// in the final numbering, at their own true depth.
/// Why: resolve_metas resolves a nested solution before the enclosing splice has fixed any anchor,
/// leaving it anchored at its own closed size; the enclosing re-anchoring then shifts it uniformly.
/// That is right only when every nested closed size happens to equal the nested term's local depth.
/// With induct() inside induct() it does at two levels but not at three (agent-only slots such as
/// an induction hypothesis make the closed size exceed the real depth), leaving the innermost fix
/// one level off.
inline NodeId resolve_metas_at(NodeId term, uint32_t depth) {
    return inspect(term,
        [&](App a) -> NodeId {
            std::vector<NodeId> args;
            NodeId head = term;
            while (true) {
                bool matched = false;
                inspect(head, [&](App inner) { args.push_back(inner.arg_id); head = inner.fn_id; matched = true; },
                              [&](auto) {});
                if (!matched) break;
            }
            std::reverse(args.begin(), args.end());
            bool head_is_meta = false;
            uint32_t head_meta_id = 0;
            inspect(head, [&](Meta m) { head_is_meta = true; head_meta_id = m.meta_id; }, [&](auto) {});
            if (head_is_meta) {
                const MetaEntry& e = ctx().meta_ctx[head_meta_id];
                if (e.solution) {
                    NodeId result = *e.solution; // raw: nested metas are resolved AFTER anchoring, below
                    uint32_t n = static_cast<uint32_t>(args.size());
                    constexpr uint32_t SAFE_OFFSET = 1'000'000;
                    result = detail::shift_levels_from(result, n, SAFE_OFFSET);
                    std::unordered_map<uint32_t, NodeId> sub;
                    uint32_t i = 0;
                    for (; i < n; ++i) {
                        bool result_is_lambda = false;
                        NodeId lam_body{};
                        inspect(result, [&](Lambda l) { result_is_lambda = true; lam_body = l.body_id; },
                                        [&](auto) {});
                        if (!result_is_lambda) break;
                        sub[i] = resolve_metas_at(args[i], depth);
                        result = lam_body;
                    }
                    result = detail::substitute_vars_relevel(result, sub, depth, 0);
                    for (; i < n; ++i) result = mk_app(result, resolve_metas_at(args[i], depth));
                    result = shift_levels_down_from(result, n + SAFE_OFFSET, n + SAFE_OFFSET - depth);
                    return resolve_metas_at(result, depth);
                }
            }
            return mk_app(resolve_metas_at(a.fn_id, depth), resolve_metas_at(a.arg_id, depth));
        },
        [&](Lambda l) -> NodeId { return mk_lambda(resolve_metas_at(l.type_id, depth), resolve_metas_at(l.body_id, depth + 1)); },
        [&](Let l) -> NodeId {
            return mk_let(resolve_metas_at(l.type_id, depth), resolve_metas_at(l.value_id, depth), resolve_metas_at(l.body_id, depth + 1));
        },
        [&](Pi p) -> NodeId { return mk_pi(resolve_metas_at(p.dom_id, depth), resolve_metas_at(p.cod_id, depth + 1)); },
        [&](Match m) -> NodeId {
            std::vector<NodeId> branches;
            branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) branches.push_back(resolve_metas_at(b, depth));
            return mk_match(resolve_metas_at(m.scrutinee_id, depth), resolve_metas_at(m.motive_id, depth), std::move(branches));
        },
        [&](Meta meta) -> NodeId {
            const MetaEntry& e = ctx().meta_ctx[meta.meta_id];
            if (e.solution) return resolve_metas(*e.solution); // bare reference: same convention as resolve_metas
            return mk_meta(meta.meta_id);
        },
        [&](QuotType q) -> NodeId { return mk_quot_type(resolve_metas_at(q.relation, depth)); },
        [&](QuotMk q) -> NodeId { return mk_quot_mk(resolve_metas_at(q.relation, depth), resolve_metas_at(q.elem, depth)); },
        [&](QuotLift q) -> NodeId {
            return mk_quot_lift(resolve_metas_at(q.relation, depth), resolve_metas_at(q.codomain, depth), resolve_metas_at(q.f, depth),
                                 resolve_metas_at(q.respects_proof, depth), resolve_metas_at(q.quotient_elem, depth));
        },
        [&](Fix f) -> NodeId { return mk_fix(resolve_metas_at(f.fn_type, depth), f.decreasing_arg, resolve_metas_at(f.body, depth + 1)); },
        [&](auto) -> NodeId { return term; } // Var, Type, Const, TypeUniv
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: requote_captured_syntax
inline NodeId requote_captured_syntax(NodeId term, const Env& env, uint32_t depth) {
    // Two distinct counters, deliberately not conflated (an earlier
    // version of this fix used a single incrementing counter for both
    // roles and got the wrong answer in both directions -- see below):
    //   - `depth` (the outer parameter, held FIXED, never shadowed):
    //     used only in the Var case's "genuinely new binder" branch.
    //     A binder introduced at position k-env.size() WITHIN `term`
    //     itself lands at depth+(k-env.size()) once term is re-embedded
    //     starting at `depth` -- independent of `env`'s own contents
    //     (which may include ordinary VALUES like a prior curried
    //     argument, not just outer-scope neutrals) and independent of
    //     how many of term's OWN binders have been entered on the way
    //     to finding this Var, since the walk's own recursion depth
    //     increments both "the new numbering" and "the old numbering"
    //     by the same amount at each step, leaving their DIFFERENCE
    //     (which this branch does NOT use directly) irrelevant and the
    //     absolute new-binder position exactly depth+(k-env.size()).
    //     (Using env.size()-relative "shift" here instead is the wrong
    //     fix: it silently assumes env's own levels line up positionally
    //     with the outer context, which fails whenever env holds a
    //     captured VALUE like a prior argument rather than a matching
    //     outer-scope neutral -- found by this session's own end-to-end
    //     Induct test, where `add`'s captured `m` is exactly such a
    //     value.)
    //   - `cur_depth` (starts at depth, increments once per binder
    //     entered during this walk): used for quote(env[v.level], ...)
    //     when v.level < env.size() -- THAT value may itself need fresh
    //     binders placed at whatever is "next available" at the CURRENT
    //     embedding position, which does grow as this walk descends
    //     through term's own Lambda/Pi/Match-branch binders.
    std::function<NodeId(NodeId, uint32_t)> go = [&](NodeId t, uint32_t cur_depth) -> NodeId {
        return inspect(t,
            [&](Var v) -> NodeId {
                if (v.level < env.size()) return quote(env[v.level], cur_depth);
                return mk_var(depth + (v.level - static_cast<uint32_t>(env.size())));
            },
            [&](App a) -> NodeId {
                // Decompose the FULL application spine first (mirroring
                // resolve_metas' own, identical fix, for the identical
                // reason -- see its own doc comment for the full
                // explanation): if the spine's own head is a solved
                // meta, beta-reduce the WHOLE thing directly via pure
                // substitution, rather than reconstructing an unreduced
                // (Lambda-chain) applied to (args) redex through the
                // plain mk_app(go(fn),go(arg)) rebuild below.
                //
                // This is the site that actually matters for exactly
                // this failure mode: go's own Meta case (below) already,
                // correctly, resolves a BARE meta reference it reaches
                // directly -- but a captured branch's own content is
                // typically not a bare meta at all, it is that meta
                // APPLIED to the branch's own constructor-specific
                // arguments (e.g. induct()'s own subgoal_meta, applied
                // to this constructor's own fields and any induction-
                // hypothesis value) -- go's own, unfixed App case
                // recurses on fn_id/arg_id independently and only ever
                // reaches the Meta case once it has walked all the way
                // down to the innermost node, by which point the outer
                // application layers have already been rebuilt, uneval-
                // uated, via mk_app -- resolve_metas' own fix, living
                // one level too deep (only reachable via THIS Meta case,
                // never seeing the outer spine at all), never gets a
                // chance to run on it. Confirmed directly, this session:
                // an induction hypothesis's own value, solved correctly
                // and verified sound at the point it was solved, still
                // showed up unreduced -- and consequently wrong, once
                // re-embedded -- in the final extracted term, causing
                // whole_term_check to reject an otherwise fully tactic-
                // solved Induct-based proof.
                //
                // Each argument is processed via go(arg, cur_depth) --
                // NOT resolve_metas(arg) -- since an argument here may
                // still contain env/depth-relative, captured references
                // needing go's own, special treatment; only the solved
                // meta's own solution (already closed, absolute-level
                // syntax, per the Meta case's own comment below) goes
                // through resolve_metas directly.
                std::vector<NodeId> args;
                NodeId head = t;
                while (true) {
                    bool matched = false;
                    inspect(head, [&](App inner) { args.push_back(inner.arg_id); head = inner.fn_id; matched = true; },
                                  [&](auto) {});
                    if (!matched) break;
                }
                std::reverse(args.begin(), args.end());
                bool head_is_meta = false;
                uint32_t head_meta_id = 0;
                inspect(head, [&](Meta m) { head_is_meta = true; head_meta_id = m.meta_id; }, [&](auto) {});
                if (head_is_meta) {
                    const MetaEntry& e = ctx().meta_ctx[head_meta_id];
                    if (e.solution) {
                        NodeId result = *e.solution; // F3 cause 3: raw -- nested metas are resolved after anchoring (resolve_metas_at)
                        uint32_t n = static_cast<uint32_t>(args.size());
                        // Re-anchor any GENUINELY NEW binder the
                        // solution's own body introduces beyond its own
                        // closed context's size (n, matching how many
                        // Pi-layers close_over_context wrapped it in
                        // originally) to land exactly at cur_depth, the
                        // ACTUAL depth at this embedding point -- which
                        // can differ from n in EITHER direction (see
                        // shift_levels_down_from's own doc comment for
                        // why: an agent-facing-only slot, like
                        // induct()'s own induction-hypothesis entry,
                        // makes cur_depth smaller; re-embedding this
                        // exact same solution a second, deeper time
                        // inside a Fix's own preserved body makes it
                        // larger).
                        //
                        // Done in two hops through a large, SAFE
                        // intermediate offset, rather than shifting
                        // straight to cur_depth in one step (an earlier
                        // version of this fix did exactly that, and
                        // regressed): a level that starts at exactly n
                        // is never accidentally safe to move directly,
                        // since either direction of a one-step shift
                        // can land it INSIDE [0, n) -- still-pending
                        // substitution targets, about to be replaced by
                        // the loop below -- confirmed directly, this
                        // session, by a one-step shift-down landing a
                        // nested match's own new field exactly on the
                        // induction hypothesis's own slot, so the
                        // substitution loop below then spliced the
                        // recursive call meant for that slot into the
                        // field's position instead, producing "add
                        // (fix ...) Zero" in place of "add k Zero".
                        // Pushing every level >= n out to
                        // n+SAFE_OFFSET first (guaranteed clear of both
                        // [0, n) and of wherever any of the n argument
                        // values themselves already live, which can be
                        // arbitrarily large) makes the substitution loop
                        // below provably collision-free regardless of
                        // cur_depth, and the second hop then brings
                        // that same, still-relatively-ordered block back
                        // down to its true, final position in one
                        // further shift, only once nothing can collide
                        // with it anymore.
                        constexpr uint32_t SAFE_OFFSET = 1'000'000;
                        result = detail::shift_levels_from(result, n, SAFE_OFFSET);
                        // SIMULTANEOUS substitution of the n arguments (see resolve_metas's own
                        // matching fix, above, for the full derivation). SAFE_OFFSET above
                        // protects the solution BODY's own new-binder levels from the
                        // substitution; it does nothing for the ARGUMENTS' levels, which live
                        // at the actual embedding depth and can numerically equal a later,
                        // still-pending position in [0, n) -- then a sequential loop rewrites
                        // them. Confirmed directly on le_trans: n=10 (branch_ec's size), the
                        // IH argument (position 8) contains `self` at re-embedded level 9, and
                        // the last position's (9) substitution then replaced every Var(9) --
                        // including that `self` head -- with the reverted hyp's value, Var(16).
                        std::unordered_map<uint32_t, NodeId> sub;
                        uint32_t i = 0;
                        for (; i < n; ++i) {
                            bool result_is_lambda = false;
                            NodeId lam_body{};
                            inspect(result, [&](Lambda l) { result_is_lambda = true; lam_body = l.body_id; },
                                            [&](auto) {});
                            if (!result_is_lambda) break;
                            sub[i] = go(args[i], cur_depth);
                            result = lam_body;
                        }
                        result = detail::substitute_vars_relevel(result, sub, cur_depth, 0);
                        for (; i < n; ++i) result = mk_app(result, go(args[i], cur_depth));
                        result = shift_levels_down_from(result, n + SAFE_OFFSET, n + SAFE_OFFSET - cur_depth);
                        return resolve_metas_at(result, cur_depth);
                    }
                }
                return mk_app(go(a.fn_id, cur_depth), go(a.arg_id, cur_depth));
            },
            [&](Lambda l) { return mk_lambda(go(l.type_id, cur_depth), go(l.body_id, cur_depth + 1)); },
            [&](Let l) { return mk_let(go(l.type_id, cur_depth), go(l.value_id, cur_depth), go(l.body_id, cur_depth + 1)); },
            [&](Pi p) { return mk_pi(go(p.dom_id, cur_depth), go(p.cod_id, cur_depth + 1)); },
            [&](Type t) { return mk_type(t.universe_level); },
            [&](Const c) { return mk_const(c.global_id); },
            [&](Match m) {
                std::vector<NodeId> branches;
                branches.reserve(m.branches->size());
                for (NodeId b : *m.branches) branches.push_back(go(b, cur_depth));
                return mk_match(go(m.scrutinee_id, cur_depth), go(m.motive_id, cur_depth), std::move(branches));
            },
            [&](Meta meta) -> NodeId {
                // A solved meta's own solution is closed, absolute-level
                // syntax needing no env/depth-relative treatment at all
                // (see resolve_metas' own doc comment for why this is
                // safe and necessary) -- unlike every other case in this
                // function, which IS about re-basing captured, relative
                // syntax against env/depth.
                const MetaEntry& e = ctx().meta_ctx[meta.meta_id];
                if (e.solution) return resolve_metas(*e.solution);
                return mk_meta(meta.meta_id);
            },
            [&](TypeUniv u) { return mk_type_univ(u.param_index); },
            [&](QuotType q) { return mk_quot_type(go(q.relation, cur_depth)); },
            [&](QuotMk q) { return mk_quot_mk(go(q.relation, cur_depth), go(q.elem, cur_depth)); },
            [&](QuotLift q) {
                return mk_quot_lift(go(q.relation, cur_depth), go(q.codomain, cur_depth), go(q.f, cur_depth),
                                     go(q.respects_proof, cur_depth), go(q.quotient_elem, cur_depth));
            },
            [&](Fix f) { return mk_fix(go(f.fn_type, cur_depth), f.decreasing_arg, go(f.body, cur_depth + 1)); }
        );
    };
    return go(term, depth);
}
#endif


#if 0 // not yet ported to the hybrid variable representation: quote
inline NodeId quote(const Value& v, uint32_t depth) {
    return inspect_value(v,
        [&](const VType& t) -> NodeId {
            return mk_type(t.universe_level);
        },
        [&](const Closure& clo) -> NodeId {
            Value dom_val = eval(clo.dom_id, clo.env);
            NodeId dom_q = quote(dom_val, depth);
            Value body_val = eval(clo.body_id, clo.env.extend(fresh_neutral(depth)));
            NodeId body_q = quote(body_val, depth + 1);
            return mk_lambda(dom_q, body_q);
        },
        [&](const VPi& pi) -> NodeId {
            Value dom_val = eval(pi.dom_id, pi.env);
            NodeId dom_q = quote(dom_val, depth);
            Value cod_val = eval(pi.cod_id, pi.env.extend(fresh_neutral(depth)));
            NodeId cod_q = quote(cod_val, depth + 1);
            return mk_pi(dom_q, cod_q);
        },
        [&](const VCtor& vc) -> NodeId {
            // Recover the constructor's global_id via the ctor_idx-
            // ordered index -- a direct array access, not a scan.
            // Previously a full linear scan over every declared global
            // on every call, on the strength of "the number of declared
            // globals is small relative to term size" -- confirmed
            // directly (not merely suspected) to be the single most
            // costly of the seven O(all globals) scan sites this file
            // used to have: quote() is called constantly throughout
            // check()/infer()/unify() (the standard quote(eval(...),
            // depth) idiom), not just occasionally, so this fired far
            // more often per tactic step than any tactic-specific site.
            const auto& family_ctors = ctx().globals.ctors_of(vc.ind_type_global_id);
            if (vc.ctor_idx >= family_ctors.size())
                throw std::logic_error("itp: quote() found a VCtor with no matching declared constructor");
            uint32_t ctor_global_id = family_ctors[vc.ctor_idx];
            NodeId acc = mk_const(ctor_global_id);
            for (const Value& arg : vc.args) acc = mk_app(acc, quote(arg, depth));
            return acc;
        },
        [&](const VQuotType& q) -> NodeId {
            return mk_quot_type(quote(*q.relation, depth));
        },
        [&](const VQuotMk& q) -> NodeId {
            return mk_quot_mk(quote(*q.relation, depth), quote(*q.elem, depth));
        },
        [&](const VFix& vf) -> NodeId {
            // CORRECTED: a Fix's raw body bakes in ABSOLUTE levels (self
            // at vf.env.size(), its own further binders above that) that
            // are only valid when re-embedded starting at EXACTLY
            // vf.env.size() -- but quote()'s own `depth` parameter
            // reflects how deep THIS quote() call has descended through
            // the OUTER value being quoted, which has no guaranteed
            // relationship to vf.env.size() whenever this VFix is itself
            // a subterm of something quote() reaches at a different
            // depth than the env size active when eval(Fix,...) first
            // produced it (e.g. a second, later quote() pass over an
            // already-extracted proof term, where this same Fix now
            // sits under additional outer binders it did not originally
            // have). Returning vf.fix_id verbatim (this function's
            // original implementation) is exactly the same class of bug
            // already fixed for stuck-Match branches via
            // requote_captured_syntax (see that function and its own
            // Match-branch call sites above) -- and needs the identical
            // fix here: re-level vf.fix_id's own free (>= vf.env.size())
            // occurrences to `depth`, and resolve its bound (< vf.env.size())
            // occurrences against vf.env directly, via the same helper.
            // Found by this session's own end-to-end Induct test: a
            // recursive call's own re-quoted Fix ended up dispatching its
            // Match on its own self-binder instead of the induction
            // variable, once the Fix was re-embedded one level deeper
            // than it was originally evaluated at.
            return requote_captured_syntax(vf.fix_id, vf.env, depth);
        },
        [&](const VNeutral& n) -> NodeId {
            NodeId acc;
            switch (n.head.kind) {
                case NeutralHead::Kind::LocalVar:    acc = mk_var(n.head.id); break;
                case NeutralHead::Kind::GlobalConst: acc = mk_const(n.head.id); break;
                case NeutralHead::Kind::Meta:        acc = mk_meta(n.head.id); break;
            }
            for (const EliminatorFrame& frame : n.spine) {
                if (auto* af = std::get_if<ApplyFrame>(&frame)) {
                    acc = mk_app(acc, quote(af->arg, depth));
                } else if (auto* mf = std::get_if<MatchFrame>(&frame)) {
                    // Branches are stored as RAW SYNTAX relative to
                    // mf->captured_env (see MatchFrame's own doc
                    // comment), not already-valid syntax at this call's
                    // own `depth` -- reusing them verbatim here would
                    // silently keep whatever levels they had relative to
                    // the ORIGINAL function definition they came from
                    // (e.g. a global recursive function's own internal
                    // parameter numbering), which has no relationship to
                    // the ambient depth this stuck Match is being
                    // re-embedded at.
                    //
                    // For BRANCHES specifically, the fix is a PURE
                    // SYNTACTIC re-leveling substitution
                    // (requote_captured_syntax below), not eval()+
                    // quote(): an earlier version of this fix tried
                    // eval(branch, mf->captured_env) followed by quote()
                    // -- which, for any RECURSIVE branch (one whose body
                    // calls the same function again), forces that
                    // recursive call to actually be evaluated one step,
                    // immediately producing ANOTHER stuck Match one
                    // level deeper (since the argument is still a fresh
                    // neutral, never concrete) -- and quoting THAT
                    // repeats the same eval()+quote() step, an unbounded
                    // regress. Found directly, by this session's own
                    // test hanging/segfaulting before this correction --
                    // not a hypothetical risk either. Substituting the
                    // raw syntax's free variables (via captured_env's
                    // own already-computed quoted values) without ever
                    // evaluating the surrounding App/Match structure
                    // leaves an embedded recursive call exactly as
                    // symbolic, un-reduced syntax -- which is the
                    // CORRECT normal form for a stuck elimination in the
                    // first place.
                    std::vector<NodeId> branches_q;
                    branches_q.reserve(mf->branches.size());
                    for (NodeId b : mf->branches) branches_q.push_back(requote_captured_syntax(b, mf->captured_env, depth));
                    // The MOTIVE, in contrast, is reconstructed via
                    // GENUINE eval()+quote() -- the standard NbE
                    // "re-derive a function correct for a different
                    // depth" round-trip, exactly what quote(Closure)
                    // already does for any ordinary function value --
                    // NOT the purely syntactic requote_captured_syntax
                    // used for branches above. This is a genuine,
                    // necessary asymmetry, not an inconsistency: a
                    // branch is meant to be applied to SPECIFIC,
                    // already-fixed constructor field values once the
                    // match actually dispatches (so re-leveling its free
                    // variables in place, without evaluating it, is
                    // exactly right -- see the comment above). The
                    // motive is semantically a GENUINE, REUSABLE
                    // function (Pi idx, (x : T idx) -> Sort), which
                    // check_match_branches later re-applies to entirely
                    // FRESH values (each constructor's own field
                    // variables) that have no relationship at all to
                    // whatever the scrutinee's value happened to be when
                    // this particular Match got stuck. Syntactically
                    // re-leveling motive_id's raw syntax in place (this
                    // function's own earlier version) is unsound for
                    // exactly this reason: motive_id's own parameter is,
                    // by construction elsewhere in this file (see
                    // induct()'s own detailed comment on why its
                    // parameter cannot use the same level scheme as an
                    // ordinary agent-facing binder), deliberately placed
                    // at a level that numerically coincides with a
                    // position already present in mf->captured_env --
                    // and requote_captured_syntax's own generic Var case
                    // cannot distinguish "this is the motive's own
                    // parameter, which must remain a fresh, re-bindable
                    // binder" from "this is a genuine reference to that
                    // captured value, which must be substituted", since
                    // both look identical (the same level number) from
                    // its own purely-syntactic vantage point. The result
                    // was a motive that appeared to still be a function
                    // syntactically (still wrapped in the right number
                    // of Lambdas) but whose own parameter had silently
                    // been substituted away with the ORIGINAL scrutinee
                    // value instead -- a "vacuous" Lambda that could
                    // never actually be reused as intended. Evaluating
                    // it genuinely (eval()+apply(), implicitly, via
                    // quote(Closure)'s own recursive descent through
                    // each curried layer) sidesteps the ambiguity
                    // entirely: apply()'s own env-extension mechanism
                    // does not care what absolute level number the
                    // original syntax used for a parameter -- it simply
                    // extends by one each time, so the motive's own
                    // parameter(s) are always correctly treated as fresh
                    // bindings regardless of what number they happened
                    // to be written with, while any OTHER, genuinely
                    // captured free variable is resolved to its actual
                    // value exactly as normal evaluation would.
                    //
                    // This does not reintroduce the branches' own
                    // regress risk: eval() on a Match never "keeps
                    // unfolding" on its own -- it produces exactly one
                    // MatchFrame per genuine stuck point and stops,
                    // regardless of what recursive structure the motive
                    // happens to embed a reference to (e.g. this
                    // session's own add_zero_right motive, which embeds
                    // a stuck reconstruction of a recursive `add` call);
                    // any FURTHER regress would have to come from
                    // recursively re-processing that embedded match's
                    // own BRANCHES, which -- per the comment above --
                    // remain on the safe, purely syntactic path
                    // regardless of how deeply this recurses. Confirmed
                    // by this session's own end-to-end test: this is the
                    // fix that finally makes a whole-term check() of a
                    // completed, tactic-built recursive proof succeed.
                    // The motive's own idx+scrutinee parameters are, BY
                    // CONSTRUCTION (see induct()'s own detailed comment),
                    // deliberately placed at levels that coincide with
                    // where the scrutinee (and any indices) ACTUALLY sit
                    // within mf->captured_env itself -- necessary so
                    // is_locally_structurally_decreasing's own walk into
                    // motive_id (it does walk it -- see that function's
                    // own Match case) doesn't mistake motive's harmless
                    // internal parameter for an illegal bare escaping
                    // self-reference by numeric coincidence with
                    // self_ref_level. But this same design choice means
                    // eval(mf->motive_id, mf->captured_env) directly
                    // (this function's own immediately-earlier attempt)
                    // is WRONG: captured_env already holds a value at
                    // exactly the position motive_id's own parameter
                    // expects to occupy (the scrutinee's own, ORIGINAL
                    // value, from when this Match got stuck) -- so the
                    // "Closure" produced captures that value directly,
                    // and applying it later lands the NEW argument one
                    // slot further out, never actually reaching the
                    // reference inside the body, which keeps resolving
                    // to the stale, original scrutinee no matter what
                    // motive_id is later applied to. The motive's own
                    // parameter positions must instead be excluded from
                    // captured_env entirely before evaluating, so they
                    // are correctly treated as genuinely NEW bindings
                    // once real application happens (via quote(Closure)'s
                    // own recursive fresh-neutral-per-layer descent),
                    // exactly like any other function value.
                    //
                    // How many trailing positions to exclude is
                    // determined PURELY STRUCTURALLY, without needing
                    // external knowledge of the inductive type's arity:
                    // a motive is, by the standard elimination rule,
                    // always exactly a Lambda-chain of (index count + 1)
                    // layers (Pi idx.. Pi (x : T idx) . Sort) -- so
                    // counting motive_id's own leading Lambda layers
                    // gives that count directly.
                    //
                    // UPDATED (this session's continued debugging,
                    // Option B): the doc comment this replaces claimed
                    // "the motive's own N parameters are guaranteed to
                    // occupy exactly captured_env's own last N
                    // positions" -- true under induct()'s OLD motive_body
                    // scheme (scrutinee at self_level+1+arity, coinciding
                    // exactly with "n"'s own, already-existing level),
                    // false under the current one. induct()'s
                    // motive_body now deliberately places its own
                    // scrutinee parameter at body_base -- ONE level
                    // PAST where "n" itself already sits -- specifically
                    // so that check(Match)'s own explicit-motive
                    // application (`apply_all`+`mk_app`, ordinary
                    // Lambda-application semantics, inserting each
                    // argument at the CURRENT env.size()) lands
                    // correctly rather than colliding with g.target's
                    // own, separate embedded content (see induct()'s own
                    // motive_body construction comment for the full
                    // derivation). That shifts where motive_id's own
                    // parameters sit relative to captured_env by
                    // exactly one: captured_env's own size already
                    // equals body_base for the standard "\idx...
                    // \scrutinee. match scrutinee with ..." construction
                    // pattern (self, any indices, and the scrutinee
                    // itself are ALL already present in captured_env by
                    // the time this Match is reached), so motive_id's
                    // own FIRST parameter needs to land AT
                    // captured_env.size() itself, not one below it --
                    // i.e. only (motive_own_params - 1) levels should be
                    // truncated away, not the full structural count.
                    // Confirmed directly by this session's own bounds-
                    // checked Env::operator[] instrumentation: the old
                    // truncation-by-motive_own_params produced an
                    // out-of-bounds access at exactly the level
                    // motive_body's own new scrutinee reference uses,
                    // the moment induct()'s own construction was updated
                    // to the new scheme -- this truncation amount is the
                    // OTHER half of that same, single change, not an
                    // independent fix.
                    // Reconstruct the motive by evaluating it against
                    // mf->captured_env DIRECTLY, untruncated. This
                    // function's own earlier versions tried to
                    // pre-truncate captured_env by some computed number
                    // of trailing entries -- first "motive_own_params"
                    // (every leading Lambda layer of motive_id), then
                    // "motive_own_params - 1" once that was found to be
                    // off by one for induct()'s OWN motive scheme
                    // specifically. Both were solving the wrong problem:
                    // motive_id's own new parameters need to be bound by
                    // quote(Closure)'s fresh-neutral-per-layer mechanism
                    // starting at exactly captured_env.size() (matching
                    // the ordinary "next available slot" convention every
                    // OTHER absolute-level construction in this file
                    // uses) -- and captured_env, exactly as captured when
                    // the Match got stuck, ALREADY has size equal to that
                    // starting point; no truncation of it was ever
                    // needed. A hand-built motive like cong's own (whose
                    // conclusion, by design, reuses OUTER absolute levels
                    // like n/m directly rather than re-abstracting them --
                    // see itp_prelude_demo.cpp's own cong_motive comment)
                    // needs every one of captured_env's entries resolvable
                    // exactly as captured, which the "-1" truncation broke
                    // (confirmed directly: it zeroed captured_env entirely
                    // for a 4-parameter motive over a 3-entry captured_env,
                    // producing an out-of-bounds access moments later).
                    // induct()'s own motive_body -- despite the elaborate
                    // now-removed comment insisting otherwise -- turns out
                    // to need exactly the same untruncated treatment: its
                    // own idx/scrutinee placeholders are placed (via
                    // body_base) to start precisely at captured_env's own
                    // size already, by construction, not one level below
                    // it. Confirmed by rerunning both the induct()
                    // add_zero_right whole-term check() and this cong
                    // reproduction after this change: both pass.
                    NodeId motive_q = quote(eval(mf->motive_id, mf->captured_env), depth);
                    acc = mk_match(acc, motive_q, std::move(branches_q));
                } else {
                    const auto& qf = std::get<QuotLiftFrame>(frame);
                    // Fields are already syntax (see QuotLiftFrame's
                    // comment) -- no quote() sub-calls needed, exactly
                    // like motive_id/branches above.
                    acc = mk_quot_lift(qf.relation_id, qf.codomain_id, qf.f_id, qf.respects_proof_id, acc);
                }
            }
            return acc;
        }
    );
}
#endif


// ===========================================================================
// HYBRID VARIABLE REPRESENTATION (refactor spike S1)
//   * BOUND variables are de Bruijn INDICES: Var{n = index, bound = true}; 0 is the innermost enclosing binder.
//     A term that contains binders is therefore position independent.
//   * CONTEXT variables (hypotheses) are LEVELS: Var{n = level, bound = false}; valid in any context of size > level,
//     so weakening is free.
//   * Values (NbE) keep using levels for neutral variables, exactly as before.
//   * Fix: fn_type is closed under the ambient context (no self binder); body sits under ONE binder, the self reference.
// Invariant: goal targets, hypothesis types, global types/values and meta solutions have no loose bound variables.
// ===========================================================================

/// Rebuilds `t`, applying f(child, binders_crossed) to every immediate child; binders_crossed is 1 for the body of a
/// Lambda/Let/Pi/Fix and 0 otherwise. Total over node kinds: every binder-aware traversal below is built on this, so none
/// of them can silently skip a node kind (the failure mode of the old hand-written traversals, which all missed Fix).
/// Var, Const, Type, Meta and TypeUniv are returned unchanged; callers that care about Var/Meta test for them first.
template <class F>
inline NodeId map_children(NodeId t, F&& f) {
    return inspect(t,
        [&](App a) -> NodeId { return mk_app(f(a.fn_id, 0), f(a.arg_id, 0)); },
        [&](Lambda l) -> NodeId { return mk_lambda(f(l.type_id, 0), f(l.body_id, 1)); },
        [&](Let l) -> NodeId { return mk_let(f(l.type_id, 0), f(l.value_id, 0), f(l.body_id, 1)); },
        [&](Pi p) -> NodeId { return mk_pi(f(p.dom_id, 0), f(p.cod_id, 1)); },
        [&](Match m) -> NodeId {
            std::vector<NodeId> bs; bs.reserve(m.branches->size());
            for (NodeId b : *m.branches) bs.push_back(f(b, 0));
            return mk_match(f(m.scrutinee_id, 0), f(m.motive_id, 0), std::move(bs));
        },
        [&](Fix x) -> NodeId { return mk_fix(f(x.fn_type, 0), x.decreasing_arg, f(x.body, 1)); },
        [&](QuotType q) -> NodeId { return mk_quot_type(f(q.relation, 0)); },
        [&](QuotMk q) -> NodeId { return mk_quot_mk(f(q.relation, 0), f(q.elem, 0)); },
        [&](QuotLift q) -> NodeId {
            return mk_quot_lift(f(q.relation, 0), f(q.codomain, 0), f(q.f, 0), f(q.respects_proof, 0), f(q.quotient_elem, 0));
        },
        [&](auto) -> NodeId { return t; });
}

inline uint32_t sat_dec(uint32_t x) { return x > 0 ? x - 1 : 0; }

/// 1 + the largest loose bound-variable index in t (0 when t has no loose bound variable). Memoised per node.
inline uint32_t loose_range(NodeId t) {
    auto& cache = ctx().loose_cache;
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(t.tag)) << 32) | t.index;
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    uint32_t r = inspect(t,
        [&](Var v) -> uint32_t { return v.bound ? v.n + 1 : 0; },
        [&](App a) -> uint32_t { return std::max(loose_range(a.fn_id), loose_range(a.arg_id)); },
        [&](Lambda l) -> uint32_t { return std::max(loose_range(l.type_id), sat_dec(loose_range(l.body_id))); },
        [&](Let l) -> uint32_t { return std::max({loose_range(l.type_id), loose_range(l.value_id), sat_dec(loose_range(l.body_id))}); },
        [&](Pi p) -> uint32_t { return std::max(loose_range(p.dom_id), sat_dec(loose_range(p.cod_id))); },
        [&](Match m) -> uint32_t {
            uint32_t r2 = std::max(loose_range(m.scrutinee_id), loose_range(m.motive_id));
            for (NodeId b : *m.branches) r2 = std::max(r2, loose_range(b));
            return r2;
        },
        [&](Fix x) -> uint32_t { return std::max(loose_range(x.fn_type), sat_dec(loose_range(x.body))); },
        [&](QuotType q) -> uint32_t { return loose_range(q.relation); },
        [&](QuotMk q) -> uint32_t { return std::max(loose_range(q.relation), loose_range(q.elem)); },
        [&](QuotLift q) -> uint32_t {
            return std::max({loose_range(q.relation), loose_range(q.codomain), loose_range(q.f), loose_range(q.respects_proof), loose_range(q.quotient_elem)});
        },
        [&](auto) -> uint32_t { return 0; });
    cache[key] = r;
    return r;
}
inline bool has_loose_bvars(NodeId t) { return loose_range(t) > 0; }

/// Raises every loose bound index >= cutoff by d.
inline NodeId lift(NodeId t, uint32_t d, uint32_t cutoff = 0) {
    if (d == 0 || loose_range(t) <= cutoff) return t;
    if (t.tag == NodeType::VAR) {
        Var v = ctx().vars[t.index];
        return (v.bound && v.n >= cutoff) ? mk_bvar(v.n + d) : t;
    }
    return map_children(t, [&](NodeId c, uint32_t extra) { return lift(c, d, cutoff + extra); });
}

/// Replaces loose bound variable (offset + k) by vals[k] (lifted past the `offset` binders crossed) for k < vals.size(), and
/// lowers every higher loose index by vals.size() (those binders are gone). vals[0] replaces the INNERMOST removed binder.
inline NodeId instantiate_at(NodeId t, const std::vector<NodeId>& vals, uint32_t offset) {
    const uint32_t n = static_cast<uint32_t>(vals.size());
    if (n == 0 || loose_range(t) <= offset) return t;
    if (t.tag == NodeType::VAR) {
        Var v = ctx().vars[t.index];
        if (!v.bound || v.n < offset) return t;
        const uint32_t k = v.n - offset;
        return k < n ? lift(vals[k], offset) : mk_bvar(v.n - n);
    }
    return map_children(t, [&](NodeId c, uint32_t extra) { return instantiate_at(c, vals, offset + extra); });
}
inline NodeId instantiate(NodeId t, const std::vector<NodeId>& vals) { return instantiate_at(t, vals, 0); }
inline NodeId instantiate1(NodeId t, NodeId v) { return instantiate_at(t, std::vector<NodeId>{v}, 0); }

/// Closes over context variables: FVar(fvars[i]) becomes the bound variable of the i-th of n NEW outermost binders
/// (fvars.back() is the innermost, index 0 at depth 0). Existing loose bound indices are lifted by n.
inline NodeId abstract_fvars(NodeId t, const std::vector<uint32_t>& fvars, uint32_t depth = 0) {
    const uint32_t n = static_cast<uint32_t>(fvars.size());
    if (n == 0) return t;
    if (t.tag == NodeType::VAR) {
        Var v = ctx().vars[t.index];
        if (v.bound) return v.n >= depth ? mk_bvar(v.n + n) : t;
        for (uint32_t i = 0; i < n; ++i) if (fvars[i] == v.n) return mk_bvar(depth + (n - 1 - i));
        return t;
    }
    return map_children(t, [&](NodeId c, uint32_t extra) { return abstract_fvars(c, fvars, depth + extra); });
}

/// Simultaneously replaces context variables by terms (lifted past the binders crossed if they have loose bound variables).
inline NodeId replace_fvars(NodeId t, const std::unordered_map<uint32_t, NodeId>& m, uint32_t depth = 0) {
    if (m.empty()) return t;
    if (t.tag == NodeType::VAR) {
        Var v = ctx().vars[t.index];
        if (v.bound) return t;
        auto it = m.find(v.n);
        return it == m.end() ? t : lift(it->second, depth);
    }
    return map_children(t, [&](NodeId c, uint32_t extra) { return replace_fvars(c, m, depth + extra); });
}

inline std::optional<NodeId> meta_solution_of(NodeId head) {
    if (head.tag != NodeType::META) return std::nullopt;
    return ctx().meta_ctx[ctx().metas[head.index].meta_id].solution;
}

/// Splices a (closed, already resolved) meta solution applied to `args`: the leading lambdas are instantiated with the
/// processed args by ordinary beta-substitution; no re-anchoring is needed or possible (the solution is closed, args carry
/// their own binders, and instantiate lifts them correctly).
template <class P>
inline NodeId splice_meta(NodeId sol, const std::vector<NodeId>& args, P&& process) {
    std::vector<NodeId> peeled; std::size_t i = 0;
    while (i < args.size() && sol.tag == NodeType::LAMBDA) {
        peeled.push_back(process(args[i]));
        sol = ctx().lambdas[sol.index].body_id;
        ++i;
    }
    std::reverse(peeled.begin(), peeled.end());
    NodeId result = instantiate(sol, peeled);
    for (; i < args.size(); ++i) result = mk_app(result, process(args[i]));
    return result;
}

inline NodeId resolve_metas(NodeId term) {
    if (term.tag == NodeType::META) { auto s = meta_solution_of(term); return s ? resolve_metas(*s) : term; }
    if (term.tag == NodeType::APP) {
        std::vector<NodeId> args; NodeId head = term;
        while (head.tag == NodeType::APP) { App a = ctx().apps[head.index]; args.push_back(a.arg_id); head = a.fn_id; }
        std::reverse(args.begin(), args.end());
        if (auto sol = meta_solution_of(head))
            return splice_meta(resolve_metas(*sol), args, [&](NodeId a) { return resolve_metas(a); });
    }
    return map_children(term, [&](NodeId c, uint32_t) { return resolve_metas(c); });
}

inline NodeId quote_at(const Value& v, uint32_t n0, uint32_t depth);

/// Re-embeds syntax captured with `env` (a stuck match's branches, a Fix, ...) at output depth `depth`: every variable that
/// is free in `term` with respect to its own binders is replaced by the QUOTED env value; variables bound inside `term`
/// are untouched. `bound` counts binders crossed inside `term`. Because bound variables are indices this needs no
/// re-anchoring and no second depth counter (the old requote_captured_syntax needed both).
inline NodeId requote(NodeId term, const Env& env, uint32_t n0, uint32_t depth, uint32_t bound = 0) {
    if (term.tag == NodeType::VAR) {
        Var v = ctx().vars[term.index];
        if (v.bound) {
            if (v.n < bound) return term; // bound inside `term` itself
            const uint32_t j = v.n - bound;
            if (j >= env.size()) throw std::logic_error("itp: requote found a loose bound variable outside the captured environment");
            return quote_at(env[env.size() - 1 - j], n0, depth + bound);
        }
        if (v.n >= env.size()) throw std::logic_error("itp: requote found a context variable outside the captured environment");
        return quote_at(env[v.n], n0, depth + bound);
    }
    if (term.tag == NodeType::META) { auto s = meta_solution_of(term); return s ? resolve_metas(*s) : term; }
    if (term.tag == NodeType::APP) {
        std::vector<NodeId> args; NodeId head = term;
        while (head.tag == NodeType::APP) { App a = ctx().apps[head.index]; args.push_back(a.arg_id); head = a.fn_id; }
        std::reverse(args.begin(), args.end());
        if (auto sol = meta_solution_of(head))
            return splice_meta(resolve_metas(*sol), args, [&](NodeId a) { return requote(a, env, n0, depth, bound); });
    }
    return map_children(term, [&](NodeId c, uint32_t extra) { return requote(c, env, n0, depth, bound + extra); });
}

/// Value -> syntax. `n0` is the size of the ambient context (labels below it are context variables -> free Var by level);
/// `depth` is the number of binders in scope including that context (labels in [n0, depth) are binders opened by this
/// quotation -> bound Var by index).
inline NodeId quote_at(const Value& v, uint32_t n0, uint32_t depth) {
    return inspect_value(v,
        [&](const VType& t) -> NodeId { return mk_type(t.universe_level); },
        [&](const Closure& clo) -> NodeId {
            NodeId dom_q = quote_at(eval(clo.dom_id, clo.env), n0, depth);
            NodeId body_q = quote_at(eval(clo.body_id, clo.env.extend(fresh_neutral(depth))), n0, depth + 1);
            return mk_lambda(dom_q, body_q);
        },
        [&](const VPi& pi) -> NodeId {
            NodeId dom_q = quote_at(eval(pi.dom_id, pi.env), n0, depth);
            NodeId cod_q = quote_at(eval(pi.cod_id, pi.env.extend(fresh_neutral(depth))), n0, depth + 1);
            return mk_pi(dom_q, cod_q);
        },
        [&](const VCtor& vc) -> NodeId {
            const auto& family_ctors = ctx().globals.ctors_of(vc.ind_type_global_id);
            if (vc.ctor_idx >= family_ctors.size()) throw std::logic_error("itp: quote() found a VCtor with no matching declared constructor");
            NodeId acc = mk_const(family_ctors[vc.ctor_idx]);
            for (const Value& arg : vc.args) acc = mk_app(acc, quote_at(arg, n0, depth));
            return acc;
        },
        [&](const VQuotType& q) -> NodeId { return mk_quot_type(quote_at(*q.relation, n0, depth)); },
        [&](const VQuotMk& q) -> NodeId { return mk_quot_mk(quote_at(*q.relation, n0, depth), quote_at(*q.elem, n0, depth)); },
        [&](const VFix& vf) -> NodeId { return requote(vf.fix_id, vf.env, n0, depth); },
        [&](const VNeutral& n) -> NodeId {
            NodeId acc;
            switch (n.head.kind) {
                case NeutralHead::Kind::LocalVar: {
                    const uint32_t label = n.head.id;
                    if (label < n0) acc = mk_fvar(label);
                    else if (label < depth) acc = mk_bvar(depth - 1 - label);
                    else throw std::logic_error("itp: quote() found a neutral variable deeper than the quotation depth");
                    break;
                }
                case NeutralHead::Kind::GlobalConst: acc = mk_const(n.head.id); break;
                case NeutralHead::Kind::Meta:        acc = mk_meta(n.head.id); break;
            }
            for (const EliminatorFrame& frame : n.spine) {
                if (auto* af = std::get_if<ApplyFrame>(&frame)) {
                    acc = mk_app(acc, quote_at(af->arg, n0, depth));
                } else if (auto* mf = std::get_if<MatchFrame>(&frame)) {
                    std::vector<NodeId> branches_q; branches_q.reserve(mf->branches.size());
                    for (NodeId b : mf->branches) branches_q.push_back(requote(b, mf->captured_env, n0, depth));
                    NodeId motive_q = quote_at(eval(mf->motive_id, mf->captured_env), n0, depth);
                    acc = mk_match(acc, motive_q, std::move(branches_q));
                } else {
                    const auto& qf = std::get<QuotLiftFrame>(frame);
                    acc = mk_quot_lift(requote(qf.relation_id, qf.captured_env, n0, depth), requote(qf.codomain_id, qf.captured_env, n0, depth),
                                       requote(qf.f_id, qf.captured_env, n0, depth), requote(qf.respects_proof_id, qf.captured_env, n0, depth), acc);
                }
            }
            return acc;
        });
}
/// Quote a value living in a context of size `depth`.
inline NodeId quote(const Value& v, uint32_t depth) { return quote_at(v, depth, depth); }

/// True iff some immediate child satisfies pred. Total over node kinds (like map_children, read-only).
template <class P>
inline bool any_child(NodeId t, P&& pred) {
    return inspect(t,
        [&](App a) -> bool { return pred(a.fn_id) || pred(a.arg_id); },
        [&](Lambda l) -> bool { return pred(l.type_id) || pred(l.body_id); },
        [&](Let l) -> bool { return pred(l.type_id) || pred(l.value_id) || pred(l.body_id); },
        [&](Pi p) -> bool { return pred(p.dom_id) || pred(p.cod_id); },
        [&](Match m) -> bool {
            if (pred(m.scrutinee_id) || pred(m.motive_id)) return true;
            for (NodeId b : *m.branches) if (pred(b)) return true;
            return false;
        },
        [&](Fix x) -> bool { return pred(x.fn_type) || pred(x.body); },
        [&](QuotType q) -> bool { return pred(q.relation); },
        [&](QuotMk q) -> bool { return pred(q.relation) || pred(q.elem); },
        [&](QuotLift q) -> bool { return pred(q.relation) || pred(q.codomain) || pred(q.f) || pred(q.respects_proof) || pred(q.quotient_elem); },
        [&](auto) -> bool { return false; });
}
namespace detail {
/// Does t mention a context variable of level >= `level`? (bound variables never count)
inline bool references_level_at_or_above(NodeId t, uint32_t level) {
    if (t.tag == NodeType::VAR) { Var v = ctx().vars[t.index]; return !v.bound && v.n >= level; }
    return any_child(t, [&](NodeId c) { return references_level_at_or_above(c, level); });
}
} // namespace detail


/// Evaluates and re-quotes a term, deriving quote's depth argument
/// automatically. Prefer this over calling eval and quote separately.
inline NodeId normalize(NodeId term, const Env& env) {
    return quote(eval(term, env), env.size());
}

/// Compares two Match branches captured under (possibly different)
/// environments for GENUINE semantic equivalence, replacing the old
/// hard-NodeId-equality-on-the-whole-branches-vector check that used to
/// live in conv_value's and unify_values' own MatchFrame cases (both
/// documented there, at the time, as a deliberate "sound but not
/// complete" trade-off). That trade-off turned out to bite in practice,
/// not just in theory: whenever a goal's own target already embeds a
/// recursive function's stuck-Match reconstruction (e.g. from intro()'s
/// eager eval()+quote() normalization -- see quote(VNeutral)'s own
/// MatchFrame handling) and a later proof step re-derives the same
/// computation independently (e.g. a fresh call to the same recursive
/// function), the two resulting stuck values are semantically identical
/// but embed literally different, differently-hash-consed branch
/// syntax -- rejected as "differ" by the old check even though nothing
/// is actually different about what they compute. Found directly by
/// this project's own end-to-end Induct test (see PHASE2_PLAN.md's own
/// milestone and itp_kernel_test.cpp's re-derivation of
/// `add_zero_right`), which needed a normalize()-based workaround
/// specifically because exact()/check() could not see through this.
///
/// A branch is a curried Lambda chain over its own constructor's fields
/// (arity many layers, zero for a nullary constructor) ending in the
/// branch body. This recursively peels ONE Lambda layer off EACH side
/// per step -- extending BOTH sides' environments with the SAME fresh
/// neutral, so a reference to "this field" on either side resolves to
/// the identical placeholder value -- until neither side is a Lambda
/// anymore, then compares the two resulting bodies via conv_value. This
/// is exactly the same eta-style idea conv_value's own Closure-vs-
/// Closure case already uses for a single Lambda layer, generalized to
/// however many layers a constructor's own arity requires.
///
/// If the two sides disagree on how many Lambda layers they have (one
/// peels further than the other), this is treated as "differ" rather
/// than crashing. This should never actually happen: this helper is
/// only ever called on two MatchFrames already reached via two VNeutral
/// spines with equal heads and equal lengths up to this position (per
/// this file's own conv_value/unify_values preconditions), so the two
/// branches being compared here are for the same constructor of the
/// same inductive type and therefore necessarily share the same arity
/// -- but defended against regardless rather than assumed away with an
/// unchecked variant access.
/// Conversion of two stuck Matches descends into their branches under fresh variables. If those branches contain a recursive Fix call on a
/// fresh neutral, every level opens a new stuck Match and the descent never ends (comparing two different-but-equivalent stuck recursive
/// definitions). A nesting bound that FAILS CLOSED -- "not convertible" -- keeps the kernel total: it can only cause a rejection, never an
/// unsound acceptance, and differs from the reference only on inputs that overflow the reference's stack.
inline int& conv_nesting() { static thread_local int n = 0; return n; }
inline constexpr int kMaxConvNesting = 256;
struct ConvNestingGuard { ConvNestingGuard() { ++conv_nesting(); } ~ConvNestingGuard() { --conv_nesting(); } };
inline bool branches_convertible(NodeId ba, const Env& env_a, NodeId bb, const Env& env_b, uint32_t depth) {
    ConvNestingGuard nesting_guard;
    if (conv_nesting() > kMaxConvNesting) return false;
    bool a_is_lambda = ba.tag == NodeType::LAMBDA;
    bool b_is_lambda = bb.tag == NodeType::LAMBDA;
    if (a_is_lambda != b_is_lambda) return false; // arity mismatch: genuinely different shapes
    if (a_is_lambda) {
        Lambda la = std::get<Lambda>(realize(ba));
        Lambda lb = std::get<Lambda>(realize(bb));
        Value fresh = fresh_neutral(depth);
        return branches_convertible(la.body_id, env_a.extend(fresh), lb.body_id, env_b.extend(fresh), depth + 1);
    }
    return conv_value(eval(ba, env_a), eval(bb, env_b), depth);
}

/// Definitional equality on semantic values.
///
/// Precondition: the two values being compared must already be known to
/// have the same type. This has always been implicit for a conversion
/// check embedded in a bidirectional typechecker, but it is now load-
/// bearing: the eta case below (Closure vs. VNeutral) assumes the neutral
/// really does have function type. If that assumption is violated,
/// apply() will throw rather than silently returning an incorrect result.
inline bool conv_value(const Value& a, const Value& b, uint32_t depth) {
    return inspect_value(a, b,
        [&](const VType& ta, const VType& tb) -> bool {
            return ta.universe_level == tb.universe_level;
        },
        [&](const Closure& ca, const Closure& cb) -> bool {
            Value fresh = fresh_neutral(depth);
            return conv_value(eval(ca.body_id, ca.env.extend(fresh)), eval(cb.body_id, cb.env.extend(fresh)), depth + 1);
        },
        [&](const VPi& pa, const VPi& pb) -> bool {
            if (!conv_value(eval(pa.dom_id, pa.env), eval(pb.dom_id, pb.env), depth)) return false;
            Value fresh = fresh_neutral(depth);
            return conv_value(eval(pa.cod_id, pa.env.extend(fresh)), eval(pb.cod_id, pb.env.extend(fresh)), depth + 1);
        },
        [&](const VCtor& ca, const VCtor& cb) -> bool {
            if (ca.ind_type_global_id != cb.ind_type_global_id || ca.ctor_idx != cb.ctor_idx) return false;
            if (ca.args.size() != cb.args.size()) return false;
            for (std::size_t i = 0; i < ca.args.size(); ++i) {
                if (!conv_value(ca.args[i], cb.args[i], depth)) return false;
            }
            return true;
        },
        [&](const VQuotType& qa, const VQuotType& qb) -> bool {
            return conv_value(*qa.relation, *qb.relation, depth);
        },
        [&](const VQuotMk& qa, const VQuotMk& qb) -> bool {
            return conv_value(*qa.relation, *qb.relation, depth) && conv_value(*qa.elem, *qb.elem, depth);
        },
        [&](const VFix& fa, const VFix& fb) -> bool {
            // Sound but not complete, exactly like MatchFrame's own
            // comparison just below: fix_id equality (cheap, exploiting
            // hash-consing -- two Fix nodes built from equal fn_type/
            // decreasing_arg/body already intern to the same fix_id) plus
            // pointwise convertibility of the captured environments. Two
            // Fix values that are semantically equal but reached this
            // point via differently-shaped (not merely differently-
            // valued) syntax will not be recognized as convertible here
            // -- an accepted incompleteness, not a soundness gap, in
            // exactly the same spirit as this file's other captured-
            // env comparisons.
            if (fa.fix_id != fb.fix_id) return false;
            if (fa.env.size() != fb.env.size()) return false;
            for (uint32_t i = 0; i < fa.env.size(); ++i) {
                if (!conv_value(fa.env[i], fb.env[i], depth)) return false;
            }
            return true;
        },
        [&](const VNeutral& na, const VNeutral& nb) -> bool {
            if (na.head != nb.head) return false;
            if (na.spine.size() != nb.spine.size()) return false;
            bool spine_ok = true;
            for (std::size_t i = 0; i < na.spine.size(); ++i) {
                bool ok = std::visit(match_on{
                    [&](const ApplyFrame& fa, const ApplyFrame& fb) {
                        return conv_value(fa.arg, fb.arg, depth);
                    },
                    [&](const MatchFrame& fa, const MatchFrame& fb) -> bool {
                        // Motive is compared by hard NodeId equality
                        // still (motive_id is never consulted during
                        // eval() -- see this file's own "Structural
                        // recursion checking" section header -- so two
                        // differently-derived stuck matches with
                        // genuinely different motives but the SAME
                        // branches/behavior are vanishingly unlikely to
                        // arise and not worth the extra cost of also
                        // eta-comparing the motive).
                        if (fa.motive_id != fb.motive_id) return false;
                        if (fa.branches.size() != fb.branches.size()) return false;
                        // The FAST PATH (branches hard-equal AND
                        // captured_env sizes equal) compares captured_env
                        // pointwise WITHOUT re-evaluating any branch
                        // body -- valid only when both conditions hold,
                        // since it relies on the two sides sharing
                        // exactly the same absolute-level scheme.
                        // captured_env's OWN size is not itself a
                        // meaningful invariant to require equal in
                        // general: it reflects "however much of the
                        // ambient env happened to be in scope when this
                        // particular Match got stuck", which can
                        // legitimately differ between two derivations of
                        // the identical computation -- e.g. a Match
                        // embedded directly inside a hypothesis TYPE
                        // (capturing that hypothesis's own full ambient
                        // context) versus the same computation reached
                        // via a fresh call to the underlying global
                        // function (capturing only that function's own,
                        // isolated closure scope). Requiring equal sizes
                        // here as a precondition (this function's own
                        // earlier version did exactly that) rejects
                        // exactly this legitimate case. branches_convertible
                        // does not have this problem: it evaluates each
                        // side against its OWN captured_env, so mismatched
                        // sizes are perfectly fine there.
                        if (fa.branches == fb.branches && fa.captured_env.size() == fb.captured_env.size()) {
                            for (std::size_t j = 0; j < fa.captured_env.size(); ++j)
                                if (!conv_value(fa.captured_env[j], fb.captured_env[j], depth)) return false;
                            return true;
                        }
                        if (false) {
                            // Equal sizes but a pointwise mismatch does NOT mean the two sides
                            // actually differ -- confirmed directly, not assumed: infer()'s own
                            // reconstruction of an applied type does not substitute the argument
                            // as raw, unevaluated syntax -- it evaluates the argument to a Value
                            // and quotes that VALUE back into the reconstructed type (see check()'s
                            // own infer()+unify() fallback). For an argument whose own computation
                            // is itself stuck on a DEEPER stuck computation (e.g. add's own Match
                            // stuck on mul's own, already-stuck result -- two levels, not one),
                            // this produces a *semantically equal but differently-shaped* term: the
                            // outer Match's scrutinee becomes another Match AST node directly
                            // (quote()'s own, correct rendering of the inner stuck value), rather
                            // than a reference resolved through a genuine Lambda-application step
                            // the way the original mul_c/add_c evaluation built it. Both shapes are
                            // true stuck reconstructions of the identical value (confirmed directly:
                            // normalize_in_context on each independently yields the identical
                            // NodeId) -- but the FIRST one's outer captured_env holds the inner
                            // stuck value as one genuinely bound element, while the SECOND one's
                            // outer captured_env, having no intervening Lambda-application step to
                            // bind it, just carries the same flat ambient env through unchanged.
                            // Pointwise conv_value on captured_env, comparing those two shapes
                            // element-by-element, sees a real structural difference where none is
                            // semantically present. branches_convertible has no such problem (see
                            // its own doc comment): it evaluates each side's branch body against
                            // its OWN captured_env and compares the RESULTING VALUES, so it is
                            // blind to exactly this kind of representational asymmetry. Falling
                            // through to it here, rather than hard-failing, costs nothing in
                            // soundness (it is the same semantic primitive already trusted for the
                            // captured_env-size-mismatch case just below) and does not touch the
                            // segfault-preventing termination guard that case's own comment
                            // documents: fa.branches == fb.branches is still required BEFORE this
                            // whole block is ever entered, exactly as before -- this only changes
                            // what happens once that guard has already passed.
                        }
                        // FALLBACK: branches_convertible's genuine
                        // semantic comparison, reached whenever the fast
                        // path's preconditions do not both hold. Hard
                        // equality on branches is still checked FIRST
                        // (above) as an optimization and, more
                        // importantly, as a termination guard: the
                        // fallback recurses into evaluating branch
                        // bodies, and a branch body for a recursive
                        // function almost always contains ANOTHER call
                        // to that same function -- comparing two stuck
                        // applications of the same recursive function to
                        // a symbolic (never-concrete) argument would
                        // otherwise recurse forever, since each unfold
                        // produces another stuck MatchFrame with the
                        // SAME branches syntax, which the fallback would
                        // try to semantically re-compare all over again.
                        // Checking hard equality first means every
                        // REPEATED unfolding of the same global
                        // definition (branches hash-cons identical every
                        // time, by construction) is accepted immediately
                        // without recursing at all, PROVIDED the
                        // captured_env sizes also match (if they don't,
                        // even identical branch syntax needs
                        // branches_convertible's own per-side evaluation
                        // to be interpreted correctly). Found by this
                        // session's own end-to-end test: the naive
                        // "always eta-compare" version segfaulted (stack
                        // overflow) on the whole-term check() of a
                        // Pi-quantified recursive proof, where the
                        // induction variable is genuinely symbolic
                        // throughout; the naive "require equal
                        // captured_env sizes" version then rejected a
                        // genuinely-equal induction-hypothesis type
                        // check, for the reason explained above.
                        for (std::size_t j = 0; j < fa.branches.size(); ++j)
                            if (!branches_convertible(fa.branches[j], fa.captured_env, fb.branches[j], fb.captured_env, depth))
                                return false;
                        // NO further captured_env check here (this
                        // function's own earlier version had one): once
                        // branches_convertible has semantically compared
                        // every branch (each evaluated against its OWN
                        // captured_env), a trailing pointwise comparison
                        // of captured_env itself is both redundant and
                        // actively wrong -- it rejects two genuinely-equal
                        // stuck reconstructions that just happen to carry
                        // structurally different (but semantically
                        // equivalent) captured environments, e.g. one
                        // with an argument baked into the branches as a
                        // literal constructor and the other keeping it as
                        // a captured, referenced variable. The fast path
                        // above (hard NodeId-equal branches) still checks
                        // captured_env pointwise, correctly: there, equal
                        // branch syntax makes a pointwise env comparison
                        // both meaningful and necessary.
                        return true;
                    },
                    [&](const QuotLiftFrame& fa, const QuotLiftFrame& fb) -> bool {
                        // Same sound-but-not-complete syntactic comparison
                        // as MatchFrame above: relation/codomain/f are
                        // raw syntax now (see QuotLiftFrame's comment), so
                        // comparing them is a cheap NodeId equality check
                        // (exploiting hash-consing) rather than a fresh
                        // evaluate-then-compare, and captured_env is
                        // compared pointwise for the same soundness reason
                        // MatchFrame needs it. respects_proof_id is
                        // deliberately not compared at all: by proof-
                        // irrelevance-in-spirit for this reference
                        // implementation, two lifts of the same function
                        // over the same relation are considered the same
                        // eliminator regardless of which respects-proof
                        // term was supplied, since it has no computational
                        // role. A theory with real Prop-irrelevance would
                        // formalize this generally; this is a narrow,
                        // local application of the same idea.
                        if (fa.relation_id != fb.relation_id) return false;
                        if (fa.codomain_id != fb.codomain_id) return false;
                        if (fa.f_id != fb.f_id) return false;
                        if (fa.captured_env.size() != fb.captured_env.size()) return false;
                        for (std::size_t j = 0; j < fa.captured_env.size(); ++j)
                            if (!conv_value(fa.captured_env[j], fb.captured_env[j], depth)) return false;
                        return true;
                    },
                    [&](const auto&, const auto&) { return false; } // mismatched frame kinds
                }, na.spine[i], nb.spine[i]);
                if (!ok) { spine_ok = false; break; }
            }
            if (spine_ok) return true;
            // The frame-by-frame structural walk found a mismatch somewhere on the spine -- this
            // does NOT necessarily mean na and nb are actually different. See the MatchFrame case's
            // own captured_env comment above for the full derivation (confirmed directly, not
            // assumed): infer()'s own reconstruction of an applied type quotes a substituted
            // argument's computed VALUE back into syntax rather than keeping it raw, so an argument
            // whose own computation is stuck two levels deep can end up represented two different,
            // both-valid ways -- one where an outer stuck Match's captured_env holds the inner
            // stuck value as a genuinely bound element, one where it doesn't. A per-frame structural
            // walk, comparing corresponding spine positions pointwise, can see a difference at some
            // position where none is semantically present, specifically because "which position
            // holds the inner value" itself differs between the two shapes.
            //
            // The fix here is deliberately NOT the same shape as branches_convertible's own
            // fallback (tried first, found unsafe: see this session's own direct A/B test,
            // reproducing a genuine segfault): branches_convertible evaluates a branch body against
            // each side's own captured_env and compares the RESULTING VALUES via conv_value -- but
            // for two na/nb that are genuinely, persistently DIFFERENT stuck applications of the
            // SAME recursive function (an entirely ordinary situation during search, not a rare
            // edge case), every nested branches_convertible->conv_value->(another stuck MatchFrame,
            // same branches)->branches_convertible... step still passes the branches==branches
            // termination guard (it is trivially true at every level, since it is comparing the
            // same global function against itself with different arguments), so the mutual
            // recursion between conv_value and branches_convertible can unwind arbitrarily deep
            // before ever reaching a frame kind that lets it correctly, finally return false.
            // Confirmed directly: this exact shape segfaulted on an ordinary, unremarkable
            // UniformPolicy search step, at a simulation budget of 50, well before any pathologically
            // deep term was plausible -- the recursion risk is real and already small in practice,
            // not a hypothetical worst case.
            //
            // quote() has no such risk: it is a ONE-SIDED, non-mutually-recursive reconstruction --
            // eval() on a Match never keeps unfolding on its own (see quote()'s own MatchFrame case,
            // immediately above in this file, for the full argument), so quoting na and quoting nb
            // are each independently bounded by na/nb's own finite size, with no comparison-driven
            // regress possible between them. Two values are convertible if and only if their
            // canonical (quoted) syntax agrees -- this is the same equivalence conv_value exists to
            // decide in the first place, just computed by fully normalizing BOTH sides at once
            // instead of trying to avoid that cost via a structural shortcut that, here, no longer
            // applies. Strictly more expensive than the structural walk when it is reached (a full
            // quote of each side, rather than a shortcut comparison), but it is only ever reached
            // once per VNeutral-vs-VNeutral call, after the cheap structural path has already been
            // tried and failed -- not nested recursively the way branches_convertible's own fallback
            // was.
            return quote(Value{na}, depth) == quote(Value{nb}, depth);
        },
        // Eta-expansion for functions: a closure and a stuck neutral of
        // (by the precondition above) function type are convertible if
        // applying both to the same fresh variable gives convertible
        // results. This covers function eta only; it does not generalize
        // to unit or record eta without threading an explicit type
        // argument through conv_value, which is not done here.
        //
        // These two cases call eval() internally (evaluating the
        // Closure's body one level further), which means they inherit
        // eval()'s own divergence risk: if `ca`/`cb` is ultimately backed
        // by a genuinely unguarded, self-referential definition (one that
        // bypasses reserve_recursive_function/finish_recursive_function's
        // guardedness check, or is installed directly via GlobalDef
        // manipulation), repeatedly evaluating "one more level" of its
        // body can fail to terminate here, the same way calling eval()
        // on such a definition directly would. This is not a distinct
        // flaw in the eta logic -- the eta cases are strictly asymmetric
        // (Closure vs VNeutral; there is no code path that wraps both
        // sides in fresh binders, and an ordinary finite function
        // compared against an unrelated axiom terminates immediately,
        // confirmed directly) -- it is the SAME underlying risk as any
        // other route to eval() on an unguarded definition, reachable
        // here specifically because this is one more place eval() gets
        // called. The guardedness checker remains the actual mitigation;
        // there is no general fix for this beyond it, since detecting
        // arbitrary non-termination is the halting problem. See
        // GlobalEnv's comment on recursive definitions for the same point
        // made at the point definitions are declared.
        [&](const Closure& ca, const VNeutral& nb) -> bool {
            Value fresh = fresh_neutral(depth);
            Value lhs = eval(ca.body_id, ca.env.extend(fresh));
            Value rhs = apply(Value{nb}, fresh);
            return conv_value(lhs, rhs, depth + 1);
        },
        [&](const VNeutral& na, const Closure& cb) -> bool {
            Value fresh = fresh_neutral(depth);
            Value lhs = apply(Value{na}, fresh);
            Value rhs = eval(cb.body_id, cb.env.extend(fresh));
            return conv_value(lhs, rhs, depth + 1);
        },
        [&](const auto&, const auto&) -> bool {
            return false; // any other mismatched shape is not convertible
        }
    );
}

/// Are two syntactic terms definitionally equal under the given semantic
/// environment? Pass open_context(n) when checking under n ambient
/// hypotheses, or {} for closed top-level terms.
inline bool conv(NodeId a, NodeId b, const Env& env) {
    return conv_value(eval(a, env), eval(b, env), env.size());
}

// ---------------------------------------------------------------------------
// Elaborator error type and per-elaboration state
// ---------------------------------------------------------------------------
//
// Defined here (ahead of ElabCtx/unify_values below, rather than down by
// render_error/DiagnosticTable where it originally lived) because
// unify_values needs TypeError/ElabResult and this is the first point
// that's true; render_error and DiagnosticTable are unrelated to
// elaboration itself (pure error-formatting) and stay where they are.
enum class TypeErrorKind { Mismatch, NotAFunction, UnboundVariable, UniverseError,
                            UnificationFailure, OccursCheckFailure, MotiveInferenceFailure, Other };

struct TypeError {
    TypeErrorKind kind;
    NodeId offending;
    std::optional<NodeId> expected;
    std::string message;
};

using ElabResult = std::expected<void, TypeError>; // check()/unify()'s result: success carries nothing
using TypeResult = std::expected<NodeId, TypeError>; // infer()'s result: success carries the synthesized type

// Relocated here (from its original position much later in the file)
// specifically so infer()/check() below can use it: #define is a pure
// textual preprocessor substitution, unlike ordinary C++ declarations, so
// it must appear before its first use in the file, not merely before it
// in "logical" reading order. See its own comment for what it does.
#define ITP_TRY_CONCAT_(a, b) a##b
#define ITP_TRY_CONCAT(a, b) ITP_TRY_CONCAT_(a, b)
#define ITP_TRY(decl, expr) \
    auto ITP_TRY_CONCAT(itp_try_tmp_, __LINE__) = (expr); \
    if (!ITP_TRY_CONCAT(itp_try_tmp_, __LINE__).has_value()) \
        return std::unexpected(std::move(ITP_TRY_CONCAT(itp_try_tmp_, __LINE__).error())); \
    decl = std::move(ITP_TRY_CONCAT(itp_try_tmp_, __LINE__).value())

// Kept separate from ExecutionContext for the same reason MetaContext is
// kept separate from GlobalEnv: this is per-elaboration-attempt state,
// not permanent kernel state, and conflating the two would risk a failed
// or abandoned elaboration attempt's leftovers touching something that
// should survive it.
//
// A completed elaboration/proof is only accepted once `pending` is
// empty -- this is a hard, non-negotiable gate, not an optimization. A
// deferred constraint that never actually gets rechecked is exactly the
// same SHAPE of soundness hole the proof-irrelevance bug was: a check
// that looks like it ran (something was compared, something returned
// success) but didn't actually verify what it claimed to. See unify()
// below for where constraints get deferred and retried.
struct ElabCtx;
struct PendingConstraint {
    NodeId lhs, rhs;      // to be unified once more progress is possible
    uint32_t local_depth; // LocalCtx::size() at the point this was deferred, for diagnostics
    // The context the constraint was deferred IN. It can be deeper than the context drain_pending is later called with (a constraint met under a
    // binder mentions that binder's variable); re-unifying it in the outer context used to abort with Env::operator[] out of bounds.
    std::shared_ptr<const ElabCtx> ctx_at_deferral;
};

struct ElabState {
    std::vector<PendingConstraint> pending;

    bool all_constraints_resolved() const { return pending.empty(); }

    // Snapshot/restore rather than a trail-based checkpoint (unlike
    // MetaContext::solve_trail): drain_pending can reorder or remove
    // pending entries from the MIDDLE of the vector, not just append to
    // it, so a length-based "truncate back to N" would not correctly
    // restore prior contents. pending is small in practice (a handful
    // of deferred constraints per attempt, not thousands), so a full
    // copy here is cheap and, unlike a trail, correct regardless of how
    // drain_pending mutates the vector internally.
    std::vector<PendingConstraint> checkpoint() const { return pending; }
    void rewind(const std::vector<PendingConstraint>& mark) { pending = mark; }
};

// ---------------------------------------------------------------------------
// Elaborator: ElabCtx, Miller-pattern unification, infer/check
// ---------------------------------------------------------------------------
//
// ElabCtx bundles LocalCtx (types, by de Bruijn level) with the matching
// Env (values -- fresh neutrals for each level, exactly what
// open_context(n) already builds for conv's symbolic-variable tests).
// The two are always extended together, in lockstep: keeping them as two
// independently-threaded parameters at every call site would risk them
// silently drifting out of sync (an easy mistake with real soundness
// consequences -- unify()/quote() calls would use the wrong depth). They
// remain conceptually and representationally distinct types (LocalCtx a
// purpose-built persistent trie over NodeId, Env the pre-existing
// persistent trie over Value) per the earlier decision to keep typing
// scope and evaluation scope separate; ElabCtx is just the pairing
// discipline, not a merge of the two representations.
//
// SelfAwareMarker: a small, persistent cons-list tracking which levels
// were extended via extend_self_aware() rather than the ordinary
// extend(). This exists because infer(Var) needs to know, for any bound
// variable v, how many context entries existed when v's own type was
// ORIGINALLY validated -- the "original size" that re-anchoring a
// deeply-referenced type must evaluate against (see infer(Var)'s own
// comment for the full mechanism). For an ordinary binder (Pi, Lambda,
// plain extend()), the type is always validated BEFORE v occupies its
// own slot, so the original size is always v.level. For Fix's own
// self-reference specifically, infer(Fix) deliberately validates
// fn_type AFTER extending (self's own idx/scrutinee binders are built
// assuming self already has a slot), so the original size is
// v.level + 1.
//
// An earlier version of this fix (this session) tried to distinguish
// these two cases by asking whether v's own type SYNTACTICALLY
// references anything strictly above v.level -- true implied Fix-self,
// false implied ordinary. This is unsound in general, confirmed by a
// real counterexample, not merely suspected: Acc_rect's own F
// parameter has an ordinary (non-self-referential) type with THREE Pi
// layers, whose later layers' own local binders land strictly above
// F's own level purely as a mechanical consequence of being a multi-
// parameter dependent type -- syntactically indistinguishable from
// Fix's own fn_type, which ALSO never literally references its own
// level anywhere, only levels above it (idx params, the scrutinee, by
// design). Both legitimate cases produce identical syntactic evidence;
// no text-only test on the type's own content can separate them. The
// fix instead tracks the true answer explicitly, at the one point it is
// actually known: when the binder is introduced.
struct SelfAwareMarker {
    std::shared_ptr<const SelfAwareMarker> parent;
    uint32_t level;
};

struct ElabCtx {
    LocalCtx types;
    Env values;
    std::shared_ptr<const SelfAwareMarker> self_aware; // nullptr = none yet; see SelfAwareMarker above

    uint32_t size() const { return types.size(); }

    ElabCtx extend(NodeId ty) const {
        return ElabCtx{types.extend(ty), values.extend(fresh_neutral(types.size())), self_aware};
    }

    // Used at exactly one call site: infer(Fix)'s own self-extension.
    // Not a general-purpose alternative to extend() -- see
    // SelfAwareMarker's own comment for why this distinction can't be
    // recovered later from the type's own syntax.
    ElabCtx extend_self_aware(NodeId ty) const {
        auto marker = std::make_shared<SelfAwareMarker>(SelfAwareMarker{self_aware, types.size()});
        return ElabCtx{types.extend(ty), values.extend(fresh_neutral(types.size())), marker};
    }

    bool is_self_aware(uint32_t level) const {
        for (auto p = self_aware; p; p = p->parent) if (p->level == level) return true;
        return false;
    }
};

/// Checks whether `v` is a Miller pattern: a metavariable applied to a
/// spine of zero or more DISTINCT, locally-bound variables (each a bare
/// Var, not itself further applied) and nothing else -- no pending Match
/// or QuotLift frame anywhere in the spine disqualifies it. This is the
/// restricted fragment unify_values below solves directly by assignment;
/// anything else involving a Meta head is deferred to ElabState::pending
/// instead of attempted eagerly (general higher-order unification is
/// undecidable; this restriction is what every production elaborator
/// -- Lean, Coq, Agda -- actually relies on for the common case).
/// On success, returns the meta_id and the ordered list of variable
/// LEVELS in the spine (application order: index 0 was applied first),
/// which is exactly what solve_pattern needs to build the solution.
inline std::optional<std::pair<uint32_t, std::vector<uint32_t>>> is_miller_pattern(const Value& v) {
    auto* neutral = std::get_if<VNeutral>(&v);
    if (!neutral || neutral->head.kind != NeutralHead::Kind::Meta) return std::nullopt;
    std::vector<uint32_t> levels;
    std::unordered_set<uint32_t> seen;
    for (const auto& frame : neutral->spine) {
        auto* af = std::get_if<ApplyFrame>(&frame);
        if (!af) return std::nullopt; // MatchFrame/QuotLiftFrame in the spine: not a pattern
        auto* arg_neutral = std::get_if<VNeutral>(&af->arg);
        if (!arg_neutral || arg_neutral->head.kind != NeutralHead::Kind::LocalVar || !arg_neutral->spine.empty())
            return std::nullopt; // argument isn't a bare local variable
        uint32_t lvl = arg_neutral->head.id;
        if (!seen.insert(lvl).second) return std::nullopt; // repeated variable: not a pattern
        levels.push_back(lvl);
    }
    return std::make_pair(neutral->head.id, std::move(levels));
}


namespace detail {

#if 0 // not yet ported to the hybrid variable representation: occurs_meta
/// Does `t` contain a reference to metavariable `meta_id` anywhere
/// (occurs check -- rejects a solution like `?M := f ?M`, which would
/// make eval() diverge the same way any other unguarded self-reference
/// does; see GlobalEnv's comment on the identical risk for definitions).
inline bool occurs_meta(uint32_t meta_id, NodeId t) {
    return inspect(t,
        [&](Meta m) { return m.meta_id == meta_id; },
        [&](App a) { return occurs_meta(meta_id, a.fn_id) || occurs_meta(meta_id, a.arg_id); },
        [&](Lambda l) { return occurs_meta(meta_id, l.type_id) || occurs_meta(meta_id, l.body_id); },
        [&](Let l) { return occurs_meta(meta_id, l.type_id) || occurs_meta(meta_id, l.value_id) || occurs_meta(meta_id, l.body_id); },
        [&](Pi p) { return occurs_meta(meta_id, p.dom_id) || occurs_meta(meta_id, p.cod_id); },
        [&](Match m) {
            if (occurs_meta(meta_id, m.scrutinee_id) || occurs_meta(meta_id, m.motive_id)) return true;
            for (NodeId b : *m.branches) if (occurs_meta(meta_id, b)) return true;
            return false;
        },
        [&](QuotType q) { return occurs_meta(meta_id, q.relation); },
        [&](QuotMk q) { return occurs_meta(meta_id, q.relation) || occurs_meta(meta_id, q.elem); },
        [&](QuotLift q) {
            return occurs_meta(meta_id, q.relation) || occurs_meta(meta_id, q.codomain) ||
                   occurs_meta(meta_id, q.f) || occurs_meta(meta_id, q.respects_proof) ||
                   occurs_meta(meta_id, q.quotient_elem);
        },
        [&](Fix f) { return occurs_meta(meta_id, f.fn_type) || occurs_meta(meta_id, f.body); },
        [&](auto) { return false; } // Var, Type, Const, TypeUniv: no children, never a Meta
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: collect_free_levels
/// Collects the free variable LEVELS occurring in `t`, where `depth` is
/// the ambient depth `t` itself lives at: a Var{level} is free (and
/// collected) iff level < depth, using this SAME, ORIGINAL depth
/// throughout the whole traversal, never incremented for t's own
/// internal binders. This is not merely a simplification: since levels
/// are absolute/global by this kernel's own convention (a fresh
/// binder always takes the next level counting from the true root, not
/// a locally-reset zero the way a de-Bruijn-INDEX representation
/// would), t's own internal binders always take levels >= depth by
/// construction -- so comparing against the ORIGINAL depth already,
/// correctly excludes them, with no reindexing ever needed as t gains
/// more internal binders, still in a single linear pass.
///
/// An earlier version of this function incremented depth on the way
/// into a Lambda/Pi/Let's own body/codomain -- confirmed directly, by
/// isolated test, to be a genuine bug: it caused the function to
/// report the CLOSED identity function (\x. x) as depending on its own
/// bound parameter's level, and, more generally, to spuriously add ANY
/// internal binder's own level to the collected set whenever the body
/// referenced it directly. references_level_at_or_above's own use
/// (with a huge starting depth) happened not to be affected by this in
/// practice for its one existing caller, since match_index_pattern's
/// own `pat` argument is always a binder-free constructor pattern --
/// but solve_pattern's own use (with an ordinary, small depth = the
/// ambient context size) hit it directly: a proposed meta solution
/// containing so much as a single internal lambda referencing its own
/// parameter (e.g. any stuck Match reconstruction, which always has
/// branch lambdas) was incorrectly rejected as depending on a variable
/// "outside the pattern spine", even though that reference was
/// genuinely, safely bound within the solution itself.
inline void collect_free_levels(NodeId t, uint32_t depth, std::unordered_set<uint32_t>& out) {
    inspect(t,
        [&](Var v) { if (v.level < depth) out.insert(v.level); },
        [&](App a) { collect_free_levels(a.fn_id, depth, out); collect_free_levels(a.arg_id, depth, out); },
        [&](Lambda l) { collect_free_levels(l.type_id, depth, out); collect_free_levels(l.body_id, depth, out); },
        [&](Let l) {
            collect_free_levels(l.type_id, depth, out);
            collect_free_levels(l.value_id, depth, out);
            collect_free_levels(l.body_id, depth, out);
        },
        [&](Pi p) { collect_free_levels(p.dom_id, depth, out); collect_free_levels(p.cod_id, depth, out); },
        [&](Match m) {
            collect_free_levels(m.scrutinee_id, depth, out);
            collect_free_levels(m.motive_id, depth, out);
            for (NodeId b : *m.branches) collect_free_levels(b, depth, out);
        },
        [&](QuotType q) { collect_free_levels(q.relation, depth, out); },
        [&](QuotMk q) { collect_free_levels(q.relation, depth, out); collect_free_levels(q.elem, depth, out); },
        [&](QuotLift q) {
            collect_free_levels(q.relation, depth, out);
            collect_free_levels(q.codomain, depth, out);
            collect_free_levels(q.f, depth, out);
            collect_free_levels(q.respects_proof, depth, out);
            collect_free_levels(q.quotient_elem, depth, out);
        },
        [&](Fix f) { collect_free_levels(f.fn_type, depth, out); collect_free_levels(f.body, depth, out); },
        [&](auto) {} // Var handled above; Type/Const/TypeUniv/Meta: no Var children
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: substitute_levels
/// Rewrites `t` by replacing each free Var{old_level} (old_level a key
/// of `mapping`) with Var{mapping[old_level]}; anything else (t's own
/// internal binders, other node kinds) is reconstructed unchanged. This
/// is the level-renaming step that turns "other_id, referencing the
/// pattern's spine variables by their ORIGINAL ambient levels" into a
/// genuinely closed solution term referencing them by their NEW position
/// in the solution's own fresh lambda nest (position i in the nest is
/// exactly where the i-th spine argument lands once applied, since
/// solve() evaluates solutions in an empty Env -- see eval(Meta)'s
/// comment on why solutions must be closed).
inline NodeId substitute_levels(NodeId t, uint32_t depth, const std::unordered_map<uint32_t, uint32_t>& mapping) {
    return inspect(t,
        [&](Var v) -> NodeId {
            if (v.level < depth) {
                if (auto it = mapping.find(v.level); it != mapping.end()) return mk_var(it->second);
            }
            return t;
        },
        [&](App a) -> NodeId {
            return mk_app(substitute_levels(a.fn_id, depth, mapping), substitute_levels(a.arg_id, depth, mapping));
        },
        [&](Lambda l) -> NodeId {
            return mk_lambda(substitute_levels(l.type_id, depth, mapping), substitute_levels(l.body_id, depth + 1, mapping));
        },
        [&](Let l) -> NodeId {
            return mk_let(substitute_levels(l.type_id, depth, mapping),
                           substitute_levels(l.value_id, depth, mapping),
                           substitute_levels(l.body_id, depth + 1, mapping));
        },
        [&](Pi p) -> NodeId {
            return mk_pi(substitute_levels(p.dom_id, depth, mapping), substitute_levels(p.cod_id, depth + 1, mapping));
        },
        [&](Match m) -> NodeId {
            std::vector<NodeId> new_branches;
            new_branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) new_branches.push_back(substitute_levels(b, depth, mapping));
            return mk_match(substitute_levels(m.scrutinee_id, depth, mapping),
                             substitute_levels(m.motive_id, depth, mapping), std::move(new_branches));
        },
        [&](QuotType q) -> NodeId { return mk_quot_type(substitute_levels(q.relation, depth, mapping)); },
        [&](QuotMk q) -> NodeId {
            return mk_quot_mk(substitute_levels(q.relation, depth, mapping), substitute_levels(q.elem, depth, mapping));
        },
        [&](QuotLift q) -> NodeId {
            return mk_quot_lift(substitute_levels(q.relation, depth, mapping), substitute_levels(q.codomain, depth, mapping),
                                 substitute_levels(q.f, depth, mapping), substitute_levels(q.respects_proof, depth, mapping),
                                 substitute_levels(q.quotient_elem, depth, mapping));
        },
        [&](Fix f) -> NodeId { return mk_fix(substitute_levels(f.fn_type, depth, mapping), f.decreasing_arg, substitute_levels(f.body, depth + 1, mapping)); },
        [&](auto) -> NodeId { return t; } // Type, Const, TypeUniv, Meta: no children, unchanged
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: solve_pattern
/// Solves `meta_id := λ(types of var_levels...). other_id[renamed]`,
/// after checking the two conditions that make this sound: (1) the
/// occurs check (meta_id must not appear in other_id itself), and (2)
/// the scope check -- every free variable other_id depends on, AND every
/// free variable each pattern parameter's OWN type depends on, must be
/// among var_levels. A dependent parameter type mentioning some ambient
/// variable outside the pattern spine cannot be expressed in a solution
/// that has to be valid at every call site of meta_id; failing closed
/// here (returning UnificationFailure / OccursCheckFailure) rather than
/// building an ill-scoped term anyway is the same "absence is safe, a
/// wrong shortcut is not" discipline used to fix the proof-irrelevance
/// bug earlier.
inline ElabResult solve_pattern(uint32_t meta_id, const std::vector<uint32_t>& var_levels, NodeId other_id, ElabCtx& ec) {
    if (occurs_meta(meta_id, other_id)) {
        return std::unexpected(TypeError{TypeErrorKind::OccursCheckFailure, other_id, std::nullopt,
                                          "metavariable occurs in its own proposed solution"});
    }
    uint32_t depth = ec.size();
    std::unordered_set<uint32_t> allowed(var_levels.begin(), var_levels.end());

    std::unordered_set<uint32_t> free_in_solution;
    collect_free_levels(other_id, depth, free_in_solution);
    for (uint32_t lvl : var_levels) collect_free_levels(ec.types[lvl], depth, free_in_solution);
    for (uint32_t lvl : free_in_solution) {
        if (!allowed.count(lvl)) {
            return std::unexpected(TypeError{TypeErrorKind::OccursCheckFailure, other_id, std::nullopt,
                                              "solution depends on a variable outside the metavariable's pattern spine"});
        }
    }

    std::unordered_map<uint32_t, uint32_t> mapping;
    for (uint32_t i = 0; i < var_levels.size(); ++i) mapping[var_levels[i]] = i;

    NodeId body = substitute_levels(other_id, depth, mapping);
    NodeId solution = body;
    for (auto it = var_levels.rbegin(); it != var_levels.rend(); ++it) {
        NodeId param_ty = substitute_levels(ec.types[*it], depth, mapping);
        solution = mk_lambda(param_ty, solution);
    }
    ctx().meta_ctx.solve(meta_id, solution);
    return {};
}
#endif


// ---------------------------------------------------------------------------
// Match motive inference
// ---------------------------------------------------------------------------

/// Result of instantiating a constructor's own type telescope at ambient
/// depth `base_depth`: arg_types[i] is valid once ec has been extended
/// with arg_types[0..i-1] (i.e. exactly the way check()'s Lambda case
/// already extends ElabCtx one binder at a time), and result_indices are
/// valid once ec has been extended with ALL of arg_types.
struct InstantiatedCtor {
    std::vector<NodeId> arg_types;
    std::vector<NodeId> result_indices;
};

#if 0 // not yet ported to the hybrid variable representation: instantiate_ctor_telescope
/// Reads a constructor's own type (a Pi telescope of `arity` argument
/// types, ending in the inductive applied to `result_indices` -- exactly
/// what build_ctor_type constructed at declaration time) and re-levels
/// everything it contains from "written relative to a fresh 0-based
/// scope" (CtorSpec's own convention -- see build_ctor_type and the
/// `refl` example in itp_prelude_demo.cpp, where result_indices
/// reference the constructor's own args via mk_var(0), mk_var(1), ...)
/// to "valid starting at ambient depth base_depth". The mapping is built
/// incrementally exactly the way substitute_levels needs it: level i
/// (the i-th constructor argument) is only added to the mapping AFTER
/// arg_types[i] itself has been shifted, since arg_types[i] can only
/// legitimately reference EARLIER arguments (0..i-1), never itself.
inline InstantiatedCtor instantiate_ctor_telescope(uint32_t ctor_gid, uint32_t base_depth) {
    const CtorInfo& info = *ctx().globals[ctor_gid].ctor_info;
    NodeId cur = ctx().globals[ctor_gid].type;
    InstantiatedCtor out;
    std::unordered_map<uint32_t, uint32_t> mapping;
    for (uint32_t i = 0; i < info.arity; ++i) {
        Pi p = inspect(cur,
            [](Pi pi) { return pi; },
            [](auto) -> Pi { throw std::logic_error("instantiate_ctor_telescope: constructor type has fewer Pi's than its own recorded arity"); });
        out.arg_types.push_back(substitute_levels(p.dom_id, i, mapping));
        mapping[i] = base_depth + i;
        cur = p.cod_id;
    }
    // `cur` is now the inductive applied to result_indices: unwind the
    // application spine to recover them (build_ctor_type's own inverse).
    std::vector<NodeId> raw_indices;
    NodeId head = cur;
    bool progressed = true;
    while (progressed) {
        progressed = false;
        inspect(head, [&](App a) { raw_indices.push_back(a.arg_id); head = a.fn_id; progressed = true; }, [](auto) {});
    }
    std::reverse(raw_indices.begin(), raw_indices.end());
    for (NodeId ri : raw_indices) out.result_indices.push_back(substitute_levels(ri, info.arity, mapping));
    return out;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: instantiate_type_telescope
/// The type-family counterpart of instantiate_ctor_telescope: reads the
/// inductive's own "sort" (its type, as declared via
/// reserve_inductive_type/declare_inductive -- a Pi chain of length
/// `arity`, ending in a bare Type_i, e.g. Eq's `(A:Type0) -> A -> A ->
/// Type0`) and returns just the domain types, re-leveled to be valid
/// starting at ambient depth `base_depth`. Empty for a non-indexed type
/// (Nat, Bool, List as declared in this codebase), matching
/// CtorSpec::result_indices being empty in that same case.
inline std::vector<NodeId> instantiate_type_telescope(uint32_t type_global_id, uint32_t base_depth) {
    NodeId cur = ctx().globals[type_global_id].type;
    std::vector<NodeId> out;
    std::unordered_map<uint32_t, uint32_t> mapping;
    uint32_t i = 0;
    while (true) {
        auto pi = inspect(cur, [](Pi p) -> std::optional<Pi> { return p; }, [](auto) -> std::optional<Pi> { return std::nullopt; });
        if (!pi) break;
        out.push_back(substitute_levels(pi->dom_id, i, mapping));
        mapping[i] = base_depth + i;
        cur = pi->cod_id;
        ++i;
    }
    return out;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: generalize
/// Syntactic generalize: replaces every exact occurrence of a `target`
/// term (compared by hash-consed NodeId equality -- cheap, and exactly
/// what "the scrutinee/an index appears literally in the goal" means in
/// practice) with a bound reference to the given fresh level. This is
/// deliberately NOT a defeq/conv-based search: real implementations
/// (Coq's `generalize`, Agda's `with`) primarily abstract syntactic
/// occurrences too, and doing better would mean searching for
/// convertible-but-not-identical subterms throughout the goal, a much
/// more expensive and rarely-needed operation. When a target doesn't
/// appear at all, generalize simply doesn't touch anything referencing
/// it -- the caller (check()'s Match case) detects that failure mode by
/// type-checking the RESULT, not by inspecting generalize's output
/// directly; see the design note on check(Match, ...) for why that's the
/// correct place to detect it, not here.
inline NodeId generalize(NodeId g, const std::vector<std::pair<NodeId, uint32_t>>& targets) {
    for (const auto& [target, new_level] : targets) {
        if (g == target) return mk_var(new_level);
    }
    return inspect(g,
        [&](App a) { return mk_app(generalize(a.fn_id, targets), generalize(a.arg_id, targets)); },
        [&](Lambda l) { return mk_lambda(generalize(l.type_id, targets), generalize(l.body_id, targets)); },
        [&](Let l) {
            return mk_let(generalize(l.type_id, targets), generalize(l.value_id, targets), generalize(l.body_id, targets));
        },
        [&](Pi p) { return mk_pi(generalize(p.dom_id, targets), generalize(p.cod_id, targets)); },
        [&](Match m) {
            std::vector<NodeId> branches;
            branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) branches.push_back(generalize(b, targets));
            return mk_match(generalize(m.scrutinee_id, targets), generalize(m.motive_id, targets), std::move(branches));
        },
        [&](QuotType q) { return mk_quot_type(generalize(q.relation, targets)); },
        [&](QuotMk q) { return mk_quot_mk(generalize(q.relation, targets), generalize(q.elem, targets)); },
        [&](QuotLift q) {
            return mk_quot_lift(generalize(q.relation, targets), generalize(q.codomain, targets), generalize(q.f, targets),
                                 generalize(q.respects_proof, targets), generalize(q.quotient_elem, targets));
        },
        [&](Fix f) { return mk_fix(generalize(f.fn_type, targets), f.decreasing_arg, generalize(f.body, targets)); },
        [&](auto) { return g; } // Var, Type, Const, Meta, TypeUniv: no children, already checked above
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: substitute_var
/// generalize's dual: replaces every exact occurrence of `mk_var(target_level)`
/// with an ARBITRARY term `replacement`, rather than always abstracting to a
/// fresh Var. Used by induct() to build branch_expected/ih_ty by direct
/// syntactic substitution -- "g.target with the scrutinee replaced by this
/// constructor's own application" or "...replaced by this recursive field" --
/// instead of building a Lambda-abstracted motive and syntactically applying
/// it (mk_app), which does NOT actually substitute anything: a motive's own
/// parameter, once genuinely eval()'d and apply()'d, lands at whatever the
/// ambient env size happens to be at that point, not at whatever level number
/// the motive's own body was written to expect -- see induct()'s own, more
/// detailed design note on why this distinction matters, discovered directly
/// by this session's own end-to-end test failing in several different ways
/// before this direct-substitution approach was used instead.
///
/// No shifting/capture concern here despite descending under Lambda/Pi/Match
/// binders: `replacement` may itself reference other absolute levels, but
/// since this whole file addresses variables by absolute level rather than
/// binder-relative index, embedding it verbatim under MORE binders changes
/// nothing about what those references mean -- the same level-stability
/// property generalize (and everything else in this file) already relies on.
inline NodeId substitute_var(NodeId g, uint32_t target_level, NodeId replacement) {
    return inspect(g,
        [&](Var v) -> NodeId { return v.level == target_level ? replacement : g; },
        [&](App a) { return mk_app(substitute_var(a.fn_id, target_level, replacement), substitute_var(a.arg_id, target_level, replacement)); },
        [&](Lambda l) {
            return mk_lambda(substitute_var(l.type_id, target_level, replacement), substitute_var(l.body_id, target_level, replacement));
        },
        [&](Let l) {
            return mk_let(substitute_var(l.type_id, target_level, replacement), substitute_var(l.value_id, target_level, replacement),
                           substitute_var(l.body_id, target_level, replacement));
        },
        [&](Pi p) { return mk_pi(substitute_var(p.dom_id, target_level, replacement), substitute_var(p.cod_id, target_level, replacement)); },
        [&](Match m) {
            std::vector<NodeId> branches;
            branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) branches.push_back(substitute_var(b, target_level, replacement));
            return mk_match(substitute_var(m.scrutinee_id, target_level, replacement),
                             substitute_var(m.motive_id, target_level, replacement), std::move(branches));
        },
        [&](QuotType q) { return mk_quot_type(substitute_var(q.relation, target_level, replacement)); },
        [&](QuotMk q) {
            return mk_quot_mk(substitute_var(q.relation, target_level, replacement), substitute_var(q.elem, target_level, replacement));
        },
        [&](QuotLift q) {
            return mk_quot_lift(substitute_var(q.relation, target_level, replacement), substitute_var(q.codomain, target_level, replacement),
                                 substitute_var(q.f, target_level, replacement), substitute_var(q.respects_proof, target_level, replacement),
                                 substitute_var(q.quotient_elem, target_level, replacement));
        },
        [&](Fix f) {
            return mk_fix(substitute_var(f.fn_type, target_level, replacement), f.decreasing_arg,
                           substitute_var(f.body, target_level, replacement));
        },
        [&](auto) { return g; } // Var (non-matching level), Type, Const, Meta, TypeUniv: no children
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: substitute_vars
/// substitute_var's multi-target sibling: replaces SEVERAL different
/// levels with their own, independent replacements, all in ONE
/// structural pass -- NOT the same as calling substitute_var repeatedly,
/// one target at a time, which can genuinely corrupt the result whenever
/// one target's own replacement value happens to contain a level that a
/// LATER, still-pending target also matches (the just-inserted value
/// gets wrongly re-substituted, as if it had been part of the original
/// term all along). Confirmed to actually happen, not merely a
/// theoretical concern: induct()'s own per-constructor substitution
/// (generalized_goal's own markers, replaced with a recursive field's
/// recovered index value and its own scrutinee value) hit exactly this
/// collision -- a recovered index level coincided numerically with the
/// scrutinee marker's own target level, so substituting the scrutinee
/// marker second wrongly re-substituted the index value that had just
/// been inserted in its place. A single simultaneous pass has no such
/// intermediate state for a later target to accidentally match against.
inline NodeId substitute_vars(NodeId g, const std::unordered_map<uint32_t, NodeId>& replacements) {
    if (replacements.empty()) return g;
    return inspect(g,
        [&](Var v) -> NodeId { auto it = replacements.find(v.level); return it != replacements.end() ? it->second : g; },
        [&](App a) { return mk_app(substitute_vars(a.fn_id, replacements), substitute_vars(a.arg_id, replacements)); },
        [&](Lambda l) {
            return mk_lambda(substitute_vars(l.type_id, replacements), substitute_vars(l.body_id, replacements));
        },
        [&](Let l) {
            return mk_let(substitute_vars(l.type_id, replacements), substitute_vars(l.value_id, replacements),
                           substitute_vars(l.body_id, replacements));
        },
        [&](Pi p) { return mk_pi(substitute_vars(p.dom_id, replacements), substitute_vars(p.cod_id, replacements)); },
        [&](Match m) {
            std::vector<NodeId> branches;
            branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) branches.push_back(substitute_vars(b, replacements));
            return mk_match(substitute_vars(m.scrutinee_id, replacements),
                             substitute_vars(m.motive_id, replacements), std::move(branches));
        },
        [&](QuotType q) { return mk_quot_type(substitute_vars(q.relation, replacements)); },
        [&](QuotMk q) {
            return mk_quot_mk(substitute_vars(q.relation, replacements), substitute_vars(q.elem, replacements));
        },
        [&](QuotLift q) {
            return mk_quot_lift(substitute_vars(q.relation, replacements), substitute_vars(q.codomain, replacements),
                                 substitute_vars(q.f, replacements), substitute_vars(q.respects_proof, replacements),
                                 substitute_vars(q.quotient_elem, replacements));
        },
        [&](Fix f) {
            return mk_fix(substitute_vars(f.fn_type, replacements), f.decreasing_arg,
                           substitute_vars(f.body, replacements));
        },
        [&](auto) { return g; } // Var (non-matching level), Type, Const, Meta, TypeUniv: no children
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: substitute_vars_relevel
/// substitute_vars for replacements that may themselves CONTAIN BINDERS (Finding F3, cause 4).
/// A replacement was built at depth `base` (free variables < base; its internal binders occupy
/// levels >= base). When it lands at a site that is `j` binders deep inside `g`, those internal
/// binders must move up by j, otherwise they collide with the j enclosing binders of `g` (levels
/// are absolute, so a binder-carrying term is NOT position independent). Plain substitute_vars is
/// correct only for binder-free replacements (variables, constants, applications of them); the
/// meta-solution splice substitutes arguments such as a re-embedded Fix copy (an induction
/// hypothesis's recursive-call value), which do carry binders.
inline NodeId substitute_vars_relevel(NodeId g, const std::unordered_map<uint32_t, NodeId>& replacements, uint32_t base, uint32_t j) {
    if (replacements.empty()) return g;
    auto rec = [&](NodeId t, uint32_t jj) { return substitute_vars_relevel(t, replacements, base, jj); };
    return inspect(g,
        [&](Var v) -> NodeId {
            auto it = replacements.find(v.level);
            if (it == replacements.end()) return g;
            return j == 0 ? it->second : shift_levels_from(it->second, base, j);
        },
        [&](App a) -> NodeId { return mk_app(rec(a.fn_id, j), rec(a.arg_id, j)); },
        [&](Lambda l) -> NodeId { return mk_lambda(rec(l.type_id, j), rec(l.body_id, j + 1)); },
        [&](Let l) -> NodeId { return mk_let(rec(l.type_id, j), rec(l.value_id, j), rec(l.body_id, j + 1)); },
        [&](Pi p) -> NodeId { return mk_pi(rec(p.dom_id, j), rec(p.cod_id, j + 1)); },
        [&](Match m) -> NodeId {
            std::vector<NodeId> branches;
            branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) branches.push_back(rec(b, j));
            return mk_match(rec(m.scrutinee_id, j), rec(m.motive_id, j), std::move(branches));
        },
        [&](QuotType q) -> NodeId { return mk_quot_type(rec(q.relation, j)); },
        [&](QuotMk q) -> NodeId { return mk_quot_mk(rec(q.relation, j), rec(q.elem, j)); },
        [&](QuotLift q) -> NodeId {
            return mk_quot_lift(rec(q.relation, j), rec(q.codomain, j), rec(q.f, j), rec(q.respects_proof, j), rec(q.quotient_elem, j));
        },
        [&](Fix f) -> NodeId { return mk_fix(rec(f.fn_type, j), f.decreasing_arg, rec(f.body, j + 1)); },
        [&](auto) -> NodeId { return g; }
    );
}
#endif


/// Recovers the (ctor_idx-ordered) global ids of every constructor
/// belonging to `type_global_id`, by scanning every declared global's
/// CtorInfo. The SAME O(all globals) scan is_structurally_decreasing's
/// Match case already performs for the identical lookup (see that
/// function) -- not a new cost class introduced here, and not
/// re-engineered here either: a direct type_global_id -> {ctor_gid...}
/// index is exactly the fix already flagged for that call site (Phase 3
/// scope, per the project roadmap), and doing it only here while leaving
/// the other call site as-is would just mean two things to fix instead
/// of one later. Also validates the branch count matches the caller-
/// visible constructor count, catching a non-exhaustive Match.
inline std::vector<uint32_t> ctor_gids_for_type(uint32_t type_global_id) {
    // ctors_by_family (GlobalEnv::ctors_of) is already ctor_idx-ordered
    // by construction (see its own doc comment) -- no scan, no sort.
    return ctx().globals.ctors_of(type_global_id);
}

#if 0 // not yet ported to the hybrid variable representation: shift_all_levels
/// Every global's OWN declared type (Nat's sort, Eq's sort, a
/// constructor's built-via-build_ctor_type curried type, ...) is written
/// as a genuinely CLOSED term, using the "fresh 0-based scope" Var
/// convention documented on instantiate_ctor_telescope/
/// instantiate_type_telescope above -- e.g. Eq's own sort is `(A:Type0)
/// -> A -> A -> Type0`, where the second Pi's domain, `Var{0}`, refers to
/// the FIRST Pi's own bound A, not to anything in whatever ambient
/// context happens to be asking about Eq's type.
///
/// infer(Const c) below returns that raw NodeId. Naively evaluating it
/// later using the CALLER's ambient env (ec.values, which may already be
/// non-empty -- any Const lookup nested inside so much as one Lambda)
/// would look up the WRONG env slots for those internal Var references:
/// eval()'s Pi/Lambda cases always EXTEND whatever env they're given,
/// so a closed type's own first bound variable lands at ec.size() at
/// runtime, not at the literal `0` its syntax says -- an ambient/fresh
/// convention mismatch, not a bug in eval() itself (eval(Const) for a
/// DEFINED global already sidesteps this correctly, by evaluating the
/// global's VALUE in an empty Env{}, exactly matching eval(Meta)'s
/// identical convention for solved metavariables; this is the same fix,
/// applied to the TYPE side, which eval() itself never needs to touch
/// since ordinary reduction never inspects a global's type at all).
///
/// The fix: shift EVERY level in the closed type uniformly by ec.size()
/// before returning it from infer(Const), rebasing its own internal
/// binder numbering from "starts at 0" to "starts at ec.size()" -- after
/// which it composes correctly with eval(..., ec.values) exactly the way
/// every OTHER NodeId infer/check produce already does. This differs
/// from substitute_levels's use elsewhere (a specific, finite renaming,
/// leaving a term's OWN fresh internal binders untouched): here there is
/// no such distinction to make, since EVERY Var in a closed term refers
/// to SOME binder also within that same term, and all of them need the
/// same baseline shift together.
inline NodeId shift_all_levels(NodeId t, uint32_t amount) {
    if (amount == 0) return t;
    return inspect(t,
        [&](Var v) -> NodeId { return mk_var(v.level + amount); },
        [&](App a) -> NodeId { return mk_app(shift_all_levels(a.fn_id, amount), shift_all_levels(a.arg_id, amount)); },
        [&](Lambda l) -> NodeId {
            return mk_lambda(shift_all_levels(l.type_id, amount), shift_all_levels(l.body_id, amount));
        },
        [&](Let l) -> NodeId {
            return mk_let(shift_all_levels(l.type_id, amount), shift_all_levels(l.value_id, amount),
                           shift_all_levels(l.body_id, amount));
        },
        [&](Pi p) -> NodeId { return mk_pi(shift_all_levels(p.dom_id, amount), shift_all_levels(p.cod_id, amount)); },
        [&](Match m) -> NodeId {
            std::vector<NodeId> branches;
            branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) branches.push_back(shift_all_levels(b, amount));
            return mk_match(shift_all_levels(m.scrutinee_id, amount), shift_all_levels(m.motive_id, amount),
                             std::move(branches));
        },
        [&](QuotType q) -> NodeId { return mk_quot_type(shift_all_levels(q.relation, amount)); },
        [&](QuotMk q) -> NodeId {
            return mk_quot_mk(shift_all_levels(q.relation, amount), shift_all_levels(q.elem, amount));
        },
        [&](QuotLift q) -> NodeId {
            return mk_quot_lift(shift_all_levels(q.relation, amount), shift_all_levels(q.codomain, amount),
                                 shift_all_levels(q.f, amount), shift_all_levels(q.respects_proof, amount),
                                 shift_all_levels(q.quotient_elem, amount));
        },
        [&](Fix f) -> NodeId { return mk_fix(shift_all_levels(f.fn_type, amount), f.decreasing_arg, shift_all_levels(f.body, amount)); },
        [&](auto) -> NodeId { return t; } // Type, Const, Meta, TypeUniv: no children
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: shift_levels_from
/// Shifts every Var{level} in `t` with level >= `threshold` by
/// `+amount`, leaving level < threshold untouched entirely. Needed
/// specifically by check_match_branches: a constructed motive's own
/// internal binders (its index/scrutinee abstractions) all start at
/// exactly `ec.size()` at the point the motive was built -- but each
/// branch's expected type embeds that SAME motive underneath `arity`
/// MORE binders (that constructor's own arguments), which independently
/// ALSO start at `ec.size()` (both are extensions of the same base ec).
/// Without shifting, the motive's own binder and the constructor's own
/// first argument collide at the same level, and the motive's reference
/// resolves to the wrong thing once actually evaluated (confirmed by
/// direct trace: a dependent motive's own bound variable was found
/// resolving to the constructor's OWN destructured field instead of the
/// argument the motive was actually applied to). Unlike
/// shift_all_levels (unconditional -- correct only for a term with no
/// free variables below its own starting depth, e.g. a global's closed
/// type), this must NOT touch anything the motive depends on from
/// genuinely outside the whole Match (any ambient hypothesis at a level
/// below `threshold` = ec.size()), since those remain valid at their
/// original level regardless of how many more binders get introduced
/// elsewhere in the term -- the standard level-stability property this
/// entire file relies on, deliberately preserved here rather than
/// blanket-shifted away.
inline NodeId shift_levels_from(NodeId t, uint32_t threshold, uint32_t amount) {
    if (amount == 0) return t;
    return inspect(t,
        [&](Var v) -> NodeId { return v.level >= threshold ? mk_var(v.level + amount) : t; },
        [&](App a) -> NodeId {
            return mk_app(shift_levels_from(a.fn_id, threshold, amount), shift_levels_from(a.arg_id, threshold, amount));
        },
        [&](Lambda l) -> NodeId {
            return mk_lambda(shift_levels_from(l.type_id, threshold, amount), shift_levels_from(l.body_id, threshold, amount));
        },
        [&](Let l) -> NodeId {
            return mk_let(shift_levels_from(l.type_id, threshold, amount), shift_levels_from(l.value_id, threshold, amount),
                           shift_levels_from(l.body_id, threshold, amount));
        },
        [&](Pi p) -> NodeId {
            return mk_pi(shift_levels_from(p.dom_id, threshold, amount), shift_levels_from(p.cod_id, threshold, amount));
        },
        [&](Match m) -> NodeId {
            std::vector<NodeId> branches;
            branches.reserve(m.branches->size());
            for (NodeId b : *m.branches) branches.push_back(shift_levels_from(b, threshold, amount));
            return mk_match(shift_levels_from(m.scrutinee_id, threshold, amount), shift_levels_from(m.motive_id, threshold, amount),
                             std::move(branches));
        },
        [&](QuotType q) -> NodeId { return mk_quot_type(shift_levels_from(q.relation, threshold, amount)); },
        [&](QuotMk q) -> NodeId {
            return mk_quot_mk(shift_levels_from(q.relation, threshold, amount), shift_levels_from(q.elem, threshold, amount));
        },
        [&](QuotLift q) -> NodeId {
            return mk_quot_lift(shift_levels_from(q.relation, threshold, amount), shift_levels_from(q.codomain, threshold, amount),
                                 shift_levels_from(q.f, threshold, amount), shift_levels_from(q.respects_proof, threshold, amount),
                                 shift_levels_from(q.quotient_elem, threshold, amount));
        },
        [&](Fix f) -> NodeId { return mk_fix(shift_levels_from(f.fn_type, threshold, amount), f.decreasing_arg, shift_levels_from(f.body, threshold, amount)); },
        [&](auto) -> NodeId { return t; }
    );
}
#endif


} // namespace detail

/// Core unification on already-evaluated Values. Structural cases mirror
/// conv_value's shape exactly, but recurse via unify_values (not
/// conv_value) wherever a subterm might hide a solvable metavariable, so
/// that e.g. `f ?M` unifying against `f x` solves `?M := x` even though
/// the outermost shape (App/App, i.e. matching VNeutral heads here) was
/// already rigid. Falls back to conv_value-equivalent hard equality for
/// MatchFrame/QuotLiftFrame spine elements specifically -- a deliberate,
/// documented completeness gap (a meta buried under a stuck Match/
/// QuotLift eliminator in this pass won't get solved), not a soundness
/// one: this can only make unify_values FAIL in a case a fuller
/// implementation would have solved, never accept something wrong.
struct DiagnosticTable;
inline std::string print(NodeId id, const DiagnosticTable* diag);

/// If `v` is a VNeutral headed by an ALREADY-SOLVED meta, re-resolves
/// it: evaluates the meta's own solution (matching eval(Meta)'s own,
/// established convention of an empty env, since solutions are
/// expected to be closed) and re-applies the WHOLE of v's own spine to
/// that, one ApplyFrame at a time, via the ordinary Value-level
/// apply(). Returns v unchanged otherwise (not meta-headed, or meta-
/// headed but still genuinely unsolved).
///
/// A non-ApplyFrame spine entry (MatchFrame/QuotLiftFrame -- a meta
/// being matched on directly, before it was solved) is left
/// unresolved rather than handled: no existing call site in this
/// kernel ever mints a meta applied that way (apply()/cases()/
/// induct() only ever apply a fresh meta to its own ambient context's
/// local variables, always via ApplyFrame), so this is deliberately
/// conservative rather than guessed at without a confirmed need --
/// any later comparison against the stale, unresolved form simply
/// fails as an ordinary mismatch, which is safe.
///
/// Exists because a Value is an immutable snapshot, taken at whatever
/// moment eval() originally produced it -- if the SAME meta appears
/// MORE THAN ONCE within a single unification (e.g. refl's own
/// conclusion "Eq A x x", where BOTH occurrences of x become separate
/// Value snapshots of the identical meta, since eval() runs once per
/// occurrence in the source syntax), solving it via the FIRST
/// occurrence does not retroactively update the SECOND occurrence's
/// own, already-captured snapshot. Without this re-resolution,
/// unify_values would treat that second occurrence as still genuinely
/// unsolved, and solve_pattern would silently OVERWRITE the first
/// solution rather than checking the two agree -- MetaContext::solve
/// itself performs no such check by design (each meta is meant to be
/// solved exactly once; this function is what keeps that true once a
/// Value snapshot can go stale mid-unification). Confirmed necessary
/// directly: apply(refl) on a goal like "Eq Nat (add n Zero) n" (NOT
/// definitionally true; it genuinely requires induction) reported
/// SUCCESS -- solving refl's own shared meta to "add n Zero" via the
/// first x-occurrence, then silently re-solving it to "n" via the
/// second, discarding the first solution entirely, with nothing ever
/// checking the two disagree. Accepted at the tactic level, though an
/// independent, from-scratch check() afterward correctly rejected the
/// resulting (unsound) term -- so the kernel's own final soundness
/// boundary held, but the tactic layer's own notion of "solved" was
/// genuinely, importantly wrong.
inline Value resolve_solved_meta_head(Value v) {
    auto* n = std::get_if<VNeutral>(&v);
    if (!n || n->head.kind != NeutralHead::Kind::Meta) return v;
    const MetaEntry& e = ctx().meta_ctx[n->head.id];
    if (!e.solution) return v;
    Value resolved = eval(*e.solution, Env{});
    for (const auto& frame : n->spine) {
        if (auto* af = std::get_if<ApplyFrame>(&frame)) {
            resolved = apply(std::move(resolved), af->arg);
        }
    }
    return resolved;
}



// ===========================================================================
// S3 (early part): helpers the unifier needs. Meta occurrences are  ?m x0 .. x_{n-1}  (applied to every context variable, as
// context variables); solutions are closed lambdas over the context.
// ===========================================================================
inline NodeId close_pi_telescope(const std::vector<NodeId>& opened_doms, NodeId body, uint32_t base);

/// Lambda counterpart of close_pi_telescope.
inline NodeId close_lambda_telescope(const std::vector<NodeId>& opened_doms, NodeId body, uint32_t base) {
    const std::size_t m = opened_doms.size();
    std::vector<uint32_t> fvl(m);
    for (std::size_t i = 0; i < m; ++i) fvl[i] = base + static_cast<uint32_t>(i);
    NodeId result = abstract_fvars(body, fvl);
    for (int j = static_cast<int>(m) - 1; j >= 0; --j)
        result = mk_lambda(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + j)), result);
    return result;
}
inline NodeId close_over_context(NodeId term, const ElabCtx& ec) {
    std::vector<NodeId> doms; for (uint32_t i = 0; i < ec.size(); ++i) doms.push_back(ec.types[i]);
    return close_lambda_telescope(doms, term, 0);
}
inline NodeId close_pi_over_context(NodeId ty, const ElabCtx& ec) {
    std::vector<NodeId> doms; for (uint32_t i = 0; i < ec.size(); ++i) doms.push_back(ec.types[i]);
    return close_pi_telescope(doms, ty, 0);
}

/// Replaces every occurrence of a closed (no loose bound variable) term by its replacement. Both sides are position independent, so
/// no binder bookkeeping is needed.
inline NodeId replace_terms(NodeId t, const std::vector<std::pair<NodeId, NodeId>>& subs) {
    for (auto& [from, to] : subs) if (t == from) return to;
    return map_children(t, [&](NodeId c, uint32_t) { return replace_terms(c, subs); });
}

namespace detail {
inline void collect_fvars(NodeId t, std::unordered_set<uint32_t>& out) {
    if (t.tag == NodeType::VAR) { Var v = ctx().vars[t.index]; if (!v.bound) out.insert(v.n); return; }
    any_child(t, [&](NodeId c) { collect_fvars(c, out); return false; });
}
inline bool occurs_meta(uint32_t meta_id, NodeId t) {
    if (t.tag == NodeType::META) return ctx().metas[t.index].meta_id == meta_id;
    return any_child(t, [&](NodeId c) { return occurs_meta(meta_id, c); });
}
/// ?m x_{l0} .. x_{lk} =?= other  (distinct context variables): the solution is the closed lambda over the spine variables.
/// The spine must be in dependency order (a spine variable's type may mention only EARLIER spine variables); the reference
/// silently assumed this, here it is checked.
inline ElabResult solve_pattern(uint32_t meta_id, const std::vector<uint32_t>& var_levels, NodeId other_id, ElabCtx& ec) {
    if (occurs_meta(meta_id, other_id))
        return std::unexpected(TypeError{TypeErrorKind::OccursCheckFailure, other_id, std::nullopt, "metavariable occurs in its own proposed solution"});
    std::unordered_set<uint32_t> allowed(var_levels.begin(), var_levels.end()), free_in_solution;
    collect_fvars(other_id, free_in_solution);
    for (uint32_t lvl : var_levels) collect_fvars(ec.types[lvl], free_in_solution);
    for (uint32_t lvl : free_in_solution)
        if (!allowed.count(lvl))
            return std::unexpected(TypeError{TypeErrorKind::OccursCheckFailure, other_id, std::nullopt, "solution depends on a variable outside the metavariable's pattern spine"});
    // rename the spine variables to a fresh telescope of context variables B, B+1, ... (beyond the context), then close over it
    const uint32_t B = ec.size();
    std::unordered_map<uint32_t, NodeId> ren;
    for (uint32_t i = 0; i < var_levels.size(); ++i) ren[var_levels[i]] = mk_fvar(B + i);
    std::vector<NodeId> doms;
    for (uint32_t i = 0; i < var_levels.size(); ++i) {
        NodeId d = replace_fvars(ec.types[var_levels[i]], ren);
        if (references_level_at_or_above(d, B + i))
            return std::unexpected(TypeError{TypeErrorKind::Other, other_id, std::nullopt, "metavariable pattern spine is not in dependency order"});
        doms.push_back(d);
    }
    ctx().meta_ctx.solve(meta_id, close_lambda_telescope(doms, replace_fvars(other_id, ren), B));
    return {};
}
} // namespace detail


inline ElabResult unify_values(const Value& a_in, const Value& b_in, ElabCtx& ec, ElabState& es) {
    Value a = resolve_solved_meta_head(a_in);
    Value b = resolve_solved_meta_head(b_in);
    auto is_meta_headed = [](const Value& v) {
        auto* n = std::get_if<VNeutral>(&v);
        return n && n->head.kind == NeutralHead::Kind::Meta;
    };
    if (auto pat = is_miller_pattern(a)) {
        return detail::solve_pattern(pat->first, pat->second, quote(b, ec.size()), ec);
    }
    if (auto pat = is_miller_pattern(b)) {
        return detail::solve_pattern(pat->first, pat->second, quote(a, ec.size()), ec);
    }
    if (is_meta_headed(a) || is_meta_headed(b)) {
        es.pending.push_back(PendingConstraint{quote(a, ec.size()), quote(b, ec.size()), ec.size(), std::make_shared<const ElabCtx>(ec)});
        return {};
    }

    auto mismatch = [&](const char* msg) -> ElabResult {
        return std::unexpected(TypeError{TypeErrorKind::UnificationFailure, NodeId{}, std::nullopt, msg});
    };

    return inspect_value(a, b,
        [&](const VType& ta, const VType& tb) -> ElabResult {
            if (ta.universe_level != tb.universe_level) return mismatch("universe level mismatch");
            return {};
        },
        [&](const Closure& ca, const Closure& cb) -> ElabResult {
            Value dom_a = eval(ca.dom_id, ca.env), dom_b = eval(cb.dom_id, cb.env);
            if (ElabResult r = unify_values(dom_a, dom_b, ec, es); !r) return r;
            NodeId dom_ty = quote(dom_a, ec.size());
            ElabCtx ec2 = ec.extend(dom_ty);
            Value fresh = fresh_neutral(ec.size());
            return unify_values(eval(ca.body_id, ca.env.extend(fresh)), eval(cb.body_id, cb.env.extend(fresh)), ec2, es);
        },
        [&](const VPi& pa, const VPi& pb) -> ElabResult {
            Value dom_a = eval(pa.dom_id, pa.env), dom_b = eval(pb.dom_id, pb.env);
            if (ElabResult r = unify_values(dom_a, dom_b, ec, es); !r) return r;
            NodeId dom_ty = quote(dom_a, ec.size());
            ElabCtx ec2 = ec.extend(dom_ty);
            Value fresh = fresh_neutral(ec.size());
            return unify_values(eval(pa.cod_id, pa.env.extend(fresh)), eval(pb.cod_id, pb.env.extend(fresh)), ec2, es);
        },
        [&](const VCtor& ca, const VCtor& cb) -> ElabResult {
            if (ca.ind_type_global_id != cb.ind_type_global_id || ca.ctor_idx != cb.ctor_idx)
                return mismatch("different constructors");
            if (ca.args.size() != cb.args.size()) return mismatch("constructor arity mismatch");
            for (std::size_t i = 0; i < ca.args.size(); ++i) {
                if (ElabResult r = unify_values(ca.args[i], cb.args[i], ec, es); !r) return r;
            }
            return {};
        },
        [&](const VQuotType& qa, const VQuotType& qb) -> ElabResult {
            return unify_values(*qa.relation, *qb.relation, ec, es);
        },
        [&](const VQuotMk& qa, const VQuotMk& qb) -> ElabResult {
            if (ElabResult r = unify_values(*qa.relation, *qb.relation, ec, es); !r) return r;
            return unify_values(*qa.elem, *qb.elem, ec, es);
        },
        [&](const VFix& fa, const VFix& fb) -> ElabResult {
            // Hard syntactic equality, same documented completeness gap
            // as MatchFrame/QuotLiftFrame just below: fix_id equality
            // plus pointwise conv_value (not recursive unify_values) on
            // the captured environments -- a metavariable buried only
            // inside a captured Fix env won't get solved by this pass,
            // mirroring exactly how a meta buried in a stuck Match's or
            // Quot.lift's captured env already isn't solved here either.
            if (fa.fix_id != fb.fix_id) return mismatch("different Fix nodes");
            if (fa.env.size() != fb.env.size()) return mismatch("Fix captured environments differ in size");
            for (uint32_t i = 0; i < fa.env.size(); ++i) {
                if (!conv_value(fa.env[i], fb.env[i], ec.size()))
                    return mismatch("Fix captured environments differ");
            }
            return {};
        },
        [&](const VNeutral& na, const VNeutral& nb) -> ElabResult {
            if (na.head != nb.head) {
                return mismatch("different variable/constant heads");
            }
            if (na.spine.size() != nb.spine.size()) return mismatch("spine length mismatch");
            bool spine_ok = true;
            for (std::size_t i = 0; i < na.spine.size(); ++i) {
                ElabResult ok = std::visit(match_on{
                    [&](const ApplyFrame& fa, const ApplyFrame& fb) -> ElabResult {
                        return unify_values(fa.arg, fb.arg, ec, es);
                    },
                    [&](const MatchFrame& fa, const MatchFrame& fb) -> ElabResult {
                        // Motive: hard NodeId equality (never consulted
                        // during eval(), see this function's own doc
                        // comment).
                        if (fa.motive_id != fb.motive_id || fa.branches.size() != fb.branches.size()) {
                            return mismatch("stuck Match frames differ");
                        }
                        // See conv_value's own MatchFrame case (mirrored
                        // here) for the full explanation of both the
                        // fast-path/fallback split and why captured_env's
                        // OWN size is not required to match as a
                        // precondition: it can legitimately differ
                        // between two derivations of the identical
                        // computation (e.g. a Match embedded directly in
                        // a hypothesis type, capturing that hypothesis's
                        // full ambient context, versus the same
                        // computation reached via a fresh call to the
                        // underlying global function, capturing only
                        // that function's own isolated closure scope).
                        // captured_env is still compared via conv_value
                        // (not unify_values) pointwise when sizes DO
                        // match, same documented completeness gap as
                        // before: a meta buried only inside a captured
                        // env still won't get solved by this pass.
                        if (fa.branches == fb.branches && fa.captured_env.size() == fb.captured_env.size()) {
                            for (std::size_t j = 0; j < fa.captured_env.size(); ++j)
                                if (!conv_value(fa.captured_env[j], fb.captured_env[j], ec.size()))
                                    return mismatch("stuck Match captured environments differ");
                            return ElabResult{};
                        }
                        if (false) {
                            // Equal sizes but a pointwise mismatch does NOT mean the two sides
                            // actually differ -- see conv_value's own, mirrored case just above for
                            // the full derivation (confirmed directly, not assumed): infer()'s own
                            // reconstruction of an applied type quotes the substituted argument's
                            // computed VALUE back into syntax rather than keeping it raw, so an
                            // argument whose own computation is stuck two levels deep can end up
                            // represented two different, both-valid ways depending on how it was
                            // reached -- one where the outer stuck Match's captured_env holds the
                            // inner stuck value as a genuinely bound element (reached through a real
                            // Lambda-application step), one where it doesn't (the inner stuck value
                            // re-expressed as another Match AST node directly, with no intervening
                            // binding step). Pointwise conv_value here sees a structural difference
                            // where none is semantically present. Falling through to
                            // branches_convertible, exactly as the captured_env-size-mismatch case
                            // below already does, costs nothing in soundness (same trusted semantic
                            // primitive) and leaves the segfault-preventing termination guard that
                            // case's own comment documents completely untouched: fa.branches ==
                            // fb.branches is still required before this whole block is ever
                            // entered, exactly as before.
                        }
                        for (std::size_t j = 0; j < fa.branches.size(); ++j)
                            if (!branches_convertible(fa.branches[j], fa.captured_env, fb.branches[j], fb.captured_env, ec.size()))
                                return mismatch("stuck Match frames differ");
                        // See conv_value's own, mirrored fallback case for
                        // why there is deliberately no further captured_env
                        // comparison here: branches_convertible already
                        // performed a correct, per-side semantic
                        // comparison, and a trailing pointwise check on
                        // captured_env itself would reject two genuinely-
                        // equal stuck reconstructions of the same value.
                        return ElabResult{};
                    },
                    [&](const QuotLiftFrame& fa, const QuotLiftFrame& fb) -> ElabResult {
                        if (fa.relation_id != fb.relation_id || fa.codomain_id != fb.codomain_id ||
                            fa.f_id != fb.f_id || fa.captured_env.size() != fb.captured_env.size())
                            return mismatch("stuck Quot.lift frames differ");
                        for (std::size_t j = 0; j < fa.captured_env.size(); ++j)
                            if (!conv_value(fa.captured_env[j], fb.captured_env[j], ec.size()))
                                return mismatch("stuck Quot.lift captured environments differ");
                        return ElabResult{};
                    },
                    [&](const auto&, const auto&) -> ElabResult { return mismatch("mismatched eliminator frame kinds"); }
                }, na.spine[i], nb.spine[i]);
                if (!ok) { spine_ok = false; break; }
            }
            if (spine_ok) return {};
            // See conv_value's own, mirrored VNeutral case for the full derivation of why this
            // fallback exists and why it takes this specific, non-mutually-recursive shape (quoting
            // each full side once and comparing the resulting NodeId, rather than falling back to
            // branches_convertible the way the captured_env-size-mismatch case still does) -- found
            // unsafe for exactly this trigger (pointwise mismatch, matching sizes) via this
            // session's own direct A/B test, which reproduced a genuine segfault at a simulation
            // budget of 50 during perfectly ordinary UniformPolicy search, well before any
            // pathologically deep term was plausible.
            if (quote(Value{na}, ec.size()) == quote(Value{nb}, ec.size())) return {};
            return mismatch("spine comparison differs even after full normalization");
        },
        [&](const Closure& ca, const VNeutral& nb) -> ElabResult {
            Value fresh = fresh_neutral(ec.size());
            Value lhs = eval(ca.body_id, ca.env.extend(fresh));
            Value rhs = apply(Value{nb}, fresh);
            NodeId dom_ty = quote(eval(ca.dom_id, ca.env), ec.size());
            ElabCtx ec2 = ec.extend(dom_ty);
            return unify_values(lhs, rhs, ec2, es);
        },
        [&](const VNeutral& na, const Closure& cb) -> ElabResult {
            Value fresh = fresh_neutral(ec.size());
            Value lhs = apply(Value{na}, fresh);
            Value rhs = eval(cb.body_id, cb.env.extend(fresh));
            NodeId dom_ty = quote(eval(cb.dom_id, cb.env), ec.size());
            ElabCtx ec2 = ec.extend(dom_ty);
            return unify_values(lhs, rhs, ec2, es);
        },
        [&](const auto&, const auto&) -> ElabResult { return mismatch("mismatched value shapes"); }
    );
}


inline ElabResult unify(NodeId a_id, NodeId b_id, ElabCtx& ec, ElabState& es) {
    return unify_values(eval(a_id, ec.values), eval(b_id, ec.values), ec, es);
}


/// Repeatedly re-attempts every deferred non-pattern constraint until a
/// full pass makes no further progress (either the queue empties, or
/// what's left is genuinely stuck -- left for the caller's acceptance-
/// time all_constraints_resolved() gate to catch, per ElabState's doc
/// comment on why that gate is non-negotiable).
inline ElabResult drain_pending(ElabCtx& ec, ElabState& es) {
    while (!es.pending.empty()) {
        std::vector<PendingConstraint> current;
        current.swap(es.pending);
        std::size_t n_before = current.size();
        for (auto& pc : current) {
            ElabCtx local = pc.ctx_at_deferral ? *pc.ctx_at_deferral : ec;
            if (ElabResult r = unify(pc.lhs, pc.rhs, local, es); !r) return r;
        }
        if (es.pending.size() >= n_before) break; // no net progress this pass
    }
    return {};
}


// Forward declarations: infer and check are mutually recursive (infer's
// App case checks the argument against the domain; check's fallback
// case infers and unifies).
inline TypeResult infer(NodeId t, ElabCtx& ec, ElabState& es);
struct DiagnosticTable;
inline std::string print(NodeId id, const DiagnosticTable* diag);
inline ElabResult check(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es);

/// Phase B (design spec: "Indexed Dependent Pattern Matching"): index
/// unification for dependent pattern matching under UIP.
///
/// Given a constructor's own instantiated telescope (`inst`, already
/// leveled relative to `base_depth`, per instantiate_ctor_telescope's
/// own convention -- its `result_indices[j]` is the constructor's own
/// ABSTRACT pattern for index slot j, e.g. refl's `(A_field, x_field,
/// x_field)`, or a STRUCTURED pattern like Vec's cons `Succ(n_field)`,
/// not merely `n_field` itself) and the family's ACTUAL, concrete
/// indices for this match (`concrete_indices`, e.g. `(Nat, n, m)`
/// recovered from a scrutinee's own inferred type), determines which of
/// the constructor's own fresh field variables are FORCED to equal a
/// concrete value by entering this branch, and returns that forcing as
/// a substitution -- or determines the branch is definitionally
/// UNREACHABLE (refutation: the pattern and the concrete index are
/// different constructors of the same family, which can never be equal
/// -- e.g. matching Nil's own index Zero against a goal's concrete
/// index Succ k).
///
/// This existed as a genuine gap, not a hypothetical one: a branch body
/// that needs to exploit a forced equality (e.g. refl's own field `x`,
/// declared with the family's fully generic index type `A`, being used
/// as if it had the family's ACTUAL, concrete index type once A is known
/// to be `Nat`) could not type-check without it -- confirmed directly by
/// constructing exactly that branch and watching it fail with a genuine,
/// distinct type mismatch (the field's own generic type vs. the concrete
/// one), not a re-anchoring issue.
///
/// Matching is structural and recursive, walking both `pat` and `conc`
/// as ordinary application spines in lockstep (see match_index_pattern's
/// own doc comment for the full discipline -- summarized here):
///  - `pat` is one of this constructor's own fresh field variables,
///    seen for the first time: force it to `conc` (whatever `conc` is
///    at this position -- may itself be a compound term, e.g. `Succ k`).
///  - `pat` is the SAME field variable seen again (either the same index
///    slot's own pattern reuses it, as refl's `(A, x, x)` does, or a
///    LATER slot's pattern reuses a field a prior slot already forced):
///    the already-forced value and the newly-implied one are compared by
///    recursively applying this SAME matching discipline to each other,
///    not a separate check -- trivially/definitionally equal is fine;
///    rigid and provably distinct (e.g. `True` vs `False`, or `Succ(k)`
///    vs `Zero` for any `k`, ground or not) is refutation, not an error;
///    otherwise (neither side rigid enough to conclude anything, e.g.
///    two unrelated free variables) is declined gracefully, exactly like
///    the top-level case below.
///  - `pat` and `conc` decompose to the SAME rigid head (both the same
///    global constant, e.g. both `Succ`, or already syntactically/
///    definitionally identical) with the same number of arguments:
///    recurse pairwise into each argument.
///  - `pat` and `conc` decompose to DIFFERENT constructors of the SAME
///    inductive family (e.g. `Zero` vs `Succ _`): REFUTATION -- this
///    branch can never be reached, by the family's own disjointness of
///    constructors, regardless of what either side's own arguments are.
///    Reported distinctly (CtorIndexForcing::refuted), not as an error:
///    a successful analysis outcome, not a failure of the elaborator.
///  - anything else -- heads don't match and aren't both constructors of
///    the same family, or the term shapes don't line up structurally --
///    is DECLINED, not refused: `conv()` is tried first (handles cases
///    not built from bare application spines, e.g. two different-
///    looking but definitionally-equal closed terms), and if that also
///    fails, nothing is forced and nothing is an error. This is
///    deliberate, not a fallback of last resort: the common shape here
///    is a structured pattern (`Succ(n_field)`) matched against a bare,
///    unrelated free variable -- the ordinary form of a GENERIC,
///    universally-quantified index (Mode 2 induction on an indexed
///    type: `(n:Nat) -> Vec n -> ...`, not a specific concrete one), not
///    a contradiction of any kind. An earlier version of this function
///    treated this as a hard error, which never accepted anything
///    unsound but DID reject perfectly ordinary, well-typed
///    constructions -- found by tracing generic induction through by
///    hand, not by a failing test. `forced` only ever narrows a later
///    field's declared type as a convenience; nothing downstream trusts
///    it as a soundness-relevant premise (the branch body still has to
///    genuinely type-check against whatever type results either way),
///    so declining here can only ever accept a strictly larger set of
///    valid constructions, never a wrong one.
///
/// Refutation is detected, not yet exploited: a refuted branch still
/// goes through ORDINARY type-checking below (its own field forced
/// wherever matching succeeded up to the refutation point, nothing
/// forced for the contradictory slot itself) rather than being silently
/// accepted with an arbitrary body. Deriving False from constructor
/// disjointness and letting a refuted branch's body be anything at all
/// (Coq/Agda's "impossible case" pattern) needs that derivation to exist
/// in the kernel first -- explicitly open work, not implemented here,
/// tracked as its own item rather than silently assumed to fall out of
/// this detection for free.
struct CtorIndexForcing {
    bool ok = true;
    bool refuted = false;
    std::string error;
    std::unordered_map<uint32_t, NodeId> forced; // constructor field level -> its forced concrete value
    // Populated whenever refuted becomes true, via either path: a
    // direct, rigid constructor-head mismatch at the top level (Zero
    // vs Succ _, an ordinary indexed family like Vec) or the
    // repeated-field path (refl's own (A,x,x) forcing a field to two
    // different ground values, e.g. Eq Nat Zero (Succ Zero) -- an
    // uninhabited equality hypothesis). Both are detected by the
    // identical recursive structure in match_index_pattern; the
    // caller (cases()) exploits both uniformly via refuted_pat_value/
    // refuted_conc_value directly -- NOT inst.result_indices
    // [refuted_slot]/concrete_indices[refuted_slot] (which, for the
    // repeated-field path specifically, would be a still-generic,
    // unbound field reference -- see match_index_pattern's own doc
    // comment on why this function recurses through already-forced
    // values rather than gating on groundness). refuted_pat_value/
    // refuted_conc_value are always safe to embed wherever the
    // refuted branch gets built: each is either a direct subterm of
    // concrete_indices[some slot] (always g.ec-level, in scope
    // everywhere) or a previously-forced value recursively derived
    // from one (same property, by induction on the recursion).
    // refuted_family_gid: the inductive family both conflicting values
    // are constructors of. refuted_pat_value / refuted_conc_value: the
    // two values whose constructor heads differed.
    std::optional<uint32_t> refuted_slot;
    std::optional<uint32_t> refuted_family_gid;
    std::optional<NodeId> refuted_pat_value, refuted_conc_value;
};

namespace detail {
/// Unwinds an application spine: t = ((...(head arg0) arg1)...argN) into
/// {head, [arg0, arg1, ..., argN]}. Shared by unify_ctor_pattern_indices;
/// not specific to constructor applications -- head may be anything
/// (Var, Const, Meta, ...), args may be empty (head is already the whole
/// term).
struct AppSpine {
    NodeId head;
    std::vector<NodeId> args;
};
inline AppSpine decompose_app_spine(NodeId t) {
    std::vector<NodeId> rev_args;
    NodeId cur = t;
    while (true) {
        bool matched = false;
        inspect(cur, [&](App a) { rev_args.push_back(a.arg_id); cur = a.fn_id; matched = true; }, [](auto) {});
        if (!matched) break;
    }
    AppSpine s;
    s.head = cur;
    s.args.assign(rev_args.rbegin(), rev_args.rend());
    return s;
}


#if 0 // not yet ported to the hybrid variable representation: has_internal_binder_ref
/// True iff `t` contains a Var whose level is >= `base`. For a term that is well-formed in a
/// context of size `base` (a goal's target: base = g.ec.size(); hypothesis j's type: base = j),
/// every such Var can only be bound by a binder INSIDE the term itself (Lambda/Pi/Let, a Match
/// branch's own lambdas, a Fix's self/params) -- i.e. the term carries embedded bound structure
/// that is actually REFERENCED. An embedded binder whose body never refers to it has nothing to
/// mis-level and is (correctly) not reported.
inline bool has_internal_binder_ref(NodeId t, uint32_t base) {
    return inspect(t,
        [&](Var v) -> bool { return v.level >= base; },
        [&](App a) -> bool { return has_internal_binder_ref(a.fn_id, base) || has_internal_binder_ref(a.arg_id, base); },
        [&](Lambda l) -> bool { return has_internal_binder_ref(l.type_id, base) || has_internal_binder_ref(l.body_id, base); },
        [&](Let l) -> bool {
            return has_internal_binder_ref(l.type_id, base) || has_internal_binder_ref(l.value_id, base) ||
                   has_internal_binder_ref(l.body_id, base);
        },
        [&](Pi p) -> bool { return has_internal_binder_ref(p.dom_id, base) || has_internal_binder_ref(p.cod_id, base); },
        [&](Match m) -> bool {
            if (has_internal_binder_ref(m.scrutinee_id, base) || has_internal_binder_ref(m.motive_id, base)) return true;
            for (NodeId b : *m.branches) if (has_internal_binder_ref(b, base)) return true;
            return false;
        },
        [&](QuotType q) -> bool { return has_internal_binder_ref(q.relation, base); },
        [&](QuotMk q) -> bool { return has_internal_binder_ref(q.relation, base) || has_internal_binder_ref(q.elem, base); },
        [&](QuotLift q) -> bool {
            return has_internal_binder_ref(q.relation, base) || has_internal_binder_ref(q.codomain, base) ||
                   has_internal_binder_ref(q.f, base) || has_internal_binder_ref(q.respects_proof, base) ||
                   has_internal_binder_ref(q.quotient_elem, base);
        },
        [&](Fix f) -> bool { return has_internal_binder_ref(f.fn_type, base) || has_internal_binder_ref(f.body, base); },
        [&](auto) -> bool { return false; }); // Type, Const, Meta, TypeUniv: no variables of their own
}
#endif


#if 0 // not yet ported to the hybrid variable representation: is_pi_headed
/// True exactly when `t`'s own outermost node is a Pi -- a direct, zero-unfolding check (no
/// eval(), no normalize_in_context). Shared by every caller that needs to reject "goal still has a
/// remaining Pi" before doing something that assumes otherwise: cases()/induct()'s own guard
/// (below, in each function) against calling generalize() while a not-yet-introduced variable's own
/// body still references a level generalize()'s own fresh motive variable can collide with (see
/// each guard's own comment for the full derivation), and the identically-motivated guard
/// hypergraph.py's own enumerate_legal_actions already applies before ever offering REWRITE.
inline bool is_pi_headed(NodeId t) {
    bool result = false;
    inspect(t, [&](Pi) { result = true; }, [](auto) {});
    return result;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: is_ground_ctor_term
/// True exactly when `t` is a genuinely CLOSED value built purely from
/// constructor applications -- no Var, no Meta, nothing symbolic
/// anywhere beneath it (e.g. `Succ(Succ(Zero))`, `True`; NOT `Succ(k)`
/// for a free `k`). Not used by match_index_pattern's own repeated-field
/// check below anymore (a uniform recursive call there now subsumes
/// what this was originally added for -- see that function's own doc
/// comment on why "compare heads first, regardless of groundness" is
/// both simpler and more general than "require full groundness before
/// comparing at all"). Kept as a documented, standalone primitive: a
/// genuine "are these provably, unconditionally distinct closed values"
/// check is still a reasonable thing for other constructions (Step 3's
/// own convoy/NoConfusion machinery, not yet built) to want directly,
/// separately from index-pattern matching specifically.
inline bool is_ground_ctor_term(NodeId t) {
    AppSpine s = decompose_app_spine(t);
    bool head_is_ctor = false;
    inspect(s.head, [&](Const c) { head_is_ctor = ctx().globals[c.global_id].ctor_info.has_value(); }, [](auto) {});
    if (!head_is_ctor) return false;
    for (NodeId arg : s.args) if (!is_ground_ctor_term(arg)) return false;
    return true;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: references_level_at_or_above
/// True iff `t` references, anywhere within it (accounting for its own
/// internal binders, exactly like collect_free_levels), any level >=
/// `threshold`. Used to guard a conv() call against being asked to
/// resolve a level that a given env genuinely does not have an entry
/// for -- see match_index_pattern's own fallback case, where `pat` may
/// still contain a reference to one of the CURRENT constructor's own
/// fresh field levels (not yet bound in `env`, which only has entries
/// for what existed before this constructor's fields were introduced)
/// whenever it wasn't recognized as a bare field reference or decomposed
/// via a matching rigid head. Confirmed necessary directly: calling
/// conv() unguarded here crashed (an out-of-bounds Env access, not a
/// graceful rejection) the first time a genuinely generic index
/// (Mode 2 induction) was exercised end-to-end through cases() rather
/// than only through the standalone unit tests that happened not to
/// reach this exact path.
inline bool references_level_at_or_above(NodeId t, uint32_t threshold) {
    // A large-but-safe starting depth, NOT UINT32_MAX itself: collect_
    // free_levels increments its own depth parameter by one for every
    // Pi/Lambda/Let binder it descends through, so starting at the
    // literal maximum uint32_t value means even ONE such binder
    // silently wraps depth to 0, excluding everything nested inside it
    // from collection regardless of what it actually references --
    // confirmed as a genuine, previously-dormant bug via direct,
    // isolated testing, not merely suspected: references_level_at_or_
    // above(Pi(Nat, mk_var(3)), 2) incorrectly returned false, while
    // references_level_at_or_above(mk_var(3), 2) (the same reference,
    // not nested inside a binder) correctly returned true. Never
    // triggered before because every prior caller happened to only
    // ever pass binder-free expressions (a constructor pattern, e.g.
    // "Succ(n_field)", never a full Pi-chain type). A million-binder
    // safety margin is generous by many orders of magnitude for any
    // realistic term in this kernel.
    constexpr uint32_t kSafeStartDepth = UINT32_MAX - 1'000'000;
    std::unordered_set<uint32_t> free_below_huge;
    collect_free_levels(t, kSafeStartDepth, free_below_huge);
    for (uint32_t lvl : free_below_huge) if (lvl >= threshold) return true;
    return false;
}
#endif

} // namespace detail

namespace detail {
#if 0 // not yet ported to the hybrid variable representation: check_well_scoped
/// Walks `t`, tracking `depth` as the level the NEXT internal binder
/// of t would take (starting at n_levels, the ambient context size,
/// and incrementing exactly like collect_free_levels does on the way
/// into Lambda/Pi/Let). Returns false the moment a Var{level} is
/// found with level >= depth at that point -- a reference to
/// something that does not exist yet, neither in the ambient context
/// nor in any of t's own binders passed through so far, and so would
/// crash eval()'s own Env lookup once actually evaluated. This is the
/// DUAL of collect_free_levels' own condition (level < depth there
/// means "free, and fine"): here, level >= depth is the one that
/// signals a problem, not the one that gets filtered out as "already
/// accounted for by an internal binder". Reusing collect_free_levels
/// directly (tried first) cannot distinguish a genuinely malformed,
/// out-of-bounds reference from an ordinary one correctly bound by an
/// internal binder -- both simply fail its own "level < depth" test
/// and get silently excluded either way, which is why an initial
/// attempt using it (with a large starting depth, mirroring
/// references_level_at_or_above's own call pattern) never once
/// flagged the exact malformed subgoal it was built to catch.
inline bool check_well_scoped(NodeId t, uint32_t depth) {
    return inspect(t,
        [&](Var v) { return v.level < depth; },
        [&](App a) { return check_well_scoped(a.fn_id, depth) && check_well_scoped(a.arg_id, depth); },
        [&](Lambda l) { return check_well_scoped(l.type_id, depth) && check_well_scoped(l.body_id, depth + 1); },
        [&](Let l) {
            return check_well_scoped(l.type_id, depth) && check_well_scoped(l.value_id, depth) &&
                   check_well_scoped(l.body_id, depth + 1);
        },
        [&](Pi p) { return check_well_scoped(p.dom_id, depth) && check_well_scoped(p.cod_id, depth + 1); },
        [&](Match m) {
            if (!check_well_scoped(m.scrutinee_id, depth) || !check_well_scoped(m.motive_id, depth)) return false;
            for (NodeId b : *m.branches) if (!check_well_scoped(b, depth)) return false;
            return true;
        },
        [&](QuotType q) { return check_well_scoped(q.relation, depth); },
        [&](QuotMk q) { return check_well_scoped(q.relation, depth) && check_well_scoped(q.elem, depth); },
        [&](QuotLift q) {
            return check_well_scoped(q.relation, depth) && check_well_scoped(q.codomain, depth) &&
                   check_well_scoped(q.f, depth) && check_well_scoped(q.respects_proof, depth) &&
                   check_well_scoped(q.quotient_elem, depth);
        },
        [&](Fix f) { return check_well_scoped(f.fn_type, depth) && check_well_scoped(f.body, depth + 1); },
        [&](auto) { return true; } // Type/Const/TypeUniv/Meta: no Var children, never a problem
    );
}
#endif

} // namespace detail

#if 0 // not yet ported to the hybrid variable representation: is_well_scoped
/// True iff `t` is well-formed relative to a context of exactly
/// `n_levels` entries -- i.e. it references no level >= n_levels that
/// isn't itself bound by one of t's own internal binders (see
/// detail::check_well_scoped's own comment for why this needs a
/// dedicated walk, not a reuse of collect_free_levels). A general
/// safety check, not tied to any one specific cause: rather than
/// chasing every possible triggering condition inside induct()'s own
/// level-shifting logic (already the subject of several rounds of
/// hard-won, individually-documented fixes -- see its own comments),
/// this lets a caller verify ANY newly-produced subgoal (from
/// induct(), cases(), or anything else) is genuinely well-scoped
/// before trusting it, catching this whole class of bug at its point
/// of use rather than needing its root cause fully characterized in
/// advance. Confirmed necessary directly: a subgoal from induct(),
/// applied to a hypothesis that DOES occur in the target (ruling out
/// the specific, already-fixed vestigial-hypothesis case), still
/// produced a target referencing a level one beyond its own hyp count,
/// crashing exactly the same way once evaluated.
inline bool is_well_scoped(NodeId t, uint32_t n_levels) {
    return detail::check_well_scoped(t, n_levels);
}
#endif


// ---------------------------------------------------------------------------
// Compacting commit
// ---------------------------------------------------------------------------
//
// A successful tactic probe leaves behind everything it built, and most of
// that is proof-term scaffolding (motives, matches, Fix bodies, closed-over
// context lambdas) that a hypergraph search never looks at again: only the
// resulting SUBGOALS -- their targets and hypothesis types -- matter.
// Measured directly: a search committed ~207 pool entries per tree node
// while at most ~56 per node were reachable from any goal, i.e. >=73% garbage
// (~30 KB of RSS per tree node), and CASES/INDUCT probes were the bulk of it.
//
// compact_since() reclaims that garbage without touching anything older than
// a checkpoint: it captures a structural recipe for every node REACHABLE
// FROM THE GIVEN ROOTS that was created after the checkpoint, rewinds the
// whole context to the checkpoint, and re-interns just those recipes. Older
// nodes keep their ids; the returned roots are the live terms' new ids.
//
// It refuses (changing NOTHING, reporting why) in exactly the two situations
// where discarding the post-checkpoint region could lose something a caller
// still depends on:
//  * KEPT_OLD_META_SOLVED: some metavariable that existed BEFORE the
//    checkpoint was solved since -- its solution may live in the region that
//    would be discarded, and rewinding would also un-solve it.
//  * KEPT_NEW_META_REACHABLE: a live term still mentions a metavariable
//    created after the checkpoint, which the rewind would drop.
// In both cases the caller simply keeps everything, exactly as if compaction
// did not exist, so this can only ever shrink a context, never corrupt one.

enum class CompactStatus : int { COMPACTED = 0, KEPT_OLD_META_SOLVED = 1, KEPT_NEW_META_REACHABLE = 2 };

struct CompactResult {
    CompactStatus status;
    std::vector<NodeId> roots; // the live terms, re-expressed (unchanged if not COMPACTED)
};

#if 0 // not yet ported to the hybrid variable representation: is_post_checkpoint
/// True iff `id` was interned after `cp` was taken (its index is at or above
/// that pool's size at the checkpoint).
inline bool is_post_checkpoint(const ExecutionContext::Checkpoint& cp, NodeId id) {
    switch (id.tag) {
        case NodeType::VAR:       return id.index >= cp.vars_n;
        case NodeType::APP:       return id.index >= cp.apps_n;
        case NodeType::LAMBDA:    return id.index >= cp.lambdas_n;
        case NodeType::PI:        return id.index >= cp.pis_n;
        case NodeType::TYPE:      return id.index >= cp.types_n;
        case NodeType::CONST:     return id.index >= cp.consts_n;
        case NodeType::MATCH:     return id.index >= cp.matches_n;
        case NodeType::META:      return id.index >= cp.metas_n;
        case NodeType::TYPE_UNIV: return id.index >= cp.type_univs_n;
        case NodeType::QUOT_TYPE: return id.index >= cp.quot_types_n;
        case NodeType::QUOT_MK:   return id.index >= cp.quot_mks_n;
        case NodeType::QUOT_LIFT: return id.index >= cp.quot_lifts_n;
        case NodeType::LET:       return id.index >= cp.lets_n;
        case NodeType::FIX:       return id.index >= cp.fixes_n;
    }
    return false;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: compact_since
inline CompactResult compact_since(const ExecutionContext::Checkpoint& cp, const std::vector<NodeId>& roots) {
    MetaContext& mc = ctx().meta_ctx;

    // (1) An older meta solved since the checkpoint: its solution (and the un-solving a rewind
    //     would perform) is exactly what we must not disturb.
    for (std::size_t i = cp.meta_mark.solve_trail_size; i < mc.solve_trail.size(); ++i)
        if (mc.solve_trail[i] < cp.meta_mark.entries_size) return {CompactStatus::KEPT_OLD_META_SOLVED, roots};

    // (2) Capture a rebuild recipe for every post-checkpoint node reachable from the roots, in
    //     post-order (children before parents), keyed by the node's OLD id.
    using Remap = std::function<NodeId(NodeId)>;
    struct Recipe { NodeId old_id; std::function<NodeId(const Remap&)> rebuild; };
    std::vector<Recipe> order;
    std::unordered_set<uint64_t> seen;
    auto key = [](NodeId n) { return (uint64_t{static_cast<uint32_t>(n.tag)} << 32) | n.index; };
    bool new_meta_reachable = false;

    std::function<void(NodeId)> visit = [&](NodeId n) {
        if (new_meta_reachable || !is_post_checkpoint(cp, n)) return;
        if (!seen.insert(key(n)).second) return;
        inspect(n,
            [&](Var v) { order.push_back({n, [v](const Remap&) { return mk_var(v.level); }}); },
            [&](App a) {
                visit(a.fn_id); visit(a.arg_id);
                order.push_back({n, [a](const Remap& rm) { return mk_app(rm(a.fn_id), rm(a.arg_id)); }});
            },
            [&](Lambda l) {
                visit(l.type_id); visit(l.body_id);
                order.push_back({n, [l](const Remap& rm) { return mk_lambda(rm(l.type_id), rm(l.body_id)); }});
            },
            [&](Let l) {
                visit(l.type_id); visit(l.value_id); visit(l.body_id);
                order.push_back({n, [l](const Remap& rm) { return mk_let(rm(l.type_id), rm(l.value_id), rm(l.body_id)); }});
            },
            [&](Pi p) {
                visit(p.dom_id); visit(p.cod_id);
                order.push_back({n, [p](const Remap& rm) { return mk_pi(rm(p.dom_id), rm(p.cod_id)); }});
            },
            [&](Type t) { order.push_back({n, [t](const Remap&) { return mk_type(t.universe_level); }}); },
            [&](Const c) { order.push_back({n, [c](const Remap&) { return mk_const(c.global_id); }}); },
            [&](Match m) {
                visit(m.scrutinee_id); visit(m.motive_id);
                for (NodeId b : *m.branches) visit(b);
                order.push_back({n, [m](const Remap& rm) {
                    std::vector<NodeId> bs;
                    bs.reserve(m.branches->size());
                    for (NodeId b : *m.branches) bs.push_back(rm(b));
                    return mk_match(rm(m.scrutinee_id), rm(m.motive_id), std::move(bs));
                }});
            },
            [&](Meta m) {
                if (m.meta_id >= cp.meta_mark.entries_size) { new_meta_reachable = true; return; }
                order.push_back({n, [m](const Remap&) { return mk_meta(m.meta_id); }});
            },
            [&](TypeUniv u) { order.push_back({n, [u](const Remap&) { return mk_type_univ(u.param_index); }}); },
            [&](QuotType q) {
                visit(q.relation);
                order.push_back({n, [q](const Remap& rm) { return mk_quot_type(rm(q.relation)); }});
            },
            [&](QuotMk q) {
                visit(q.relation); visit(q.elem);
                order.push_back({n, [q](const Remap& rm) { return mk_quot_mk(rm(q.relation), rm(q.elem)); }});
            },
            [&](QuotLift q) {
                visit(q.relation); visit(q.codomain); visit(q.f); visit(q.respects_proof); visit(q.quotient_elem);
                order.push_back({n, [q](const Remap& rm) {
                    return mk_quot_lift(rm(q.relation), rm(q.codomain), rm(q.f), rm(q.respects_proof), rm(q.quotient_elem));
                }});
            },
            [&](Fix f) {
                visit(f.fn_type); visit(f.body);
                order.push_back({n, [f](const Remap& rm) { return mk_fix(rm(f.fn_type), f.decreasing_arg, rm(f.body)); }});
            });
    };
    for (NodeId r : roots) visit(r);
    if (new_meta_reachable) return {CompactStatus::KEPT_NEW_META_REACHABLE, roots};

    // (3) Discard everything since the checkpoint (pools and metas), then (4) re-intern the live
    //     recipes in post-order, mapping each old id to its fresh one.
    ctx().rewind(cp);
    std::unordered_map<uint64_t, NodeId> remap;
    remap.reserve(order.size() * 2);
    Remap rm = [&](NodeId n) -> NodeId {
        if (!is_post_checkpoint(cp, n)) return n; // older than the checkpoint: id unchanged
        return remap.at(key(n));
    };
    for (Recipe& r : order) remap[key(r.old_id)] = r.rebuild(rm);
    std::vector<NodeId> out;
    out.reserve(roots.size());
    for (NodeId r : roots) out.push_back(rm(r));
    return {CompactStatus::COMPACTED, std::move(out)};
}
#endif


#if 0 // not yet ported to the hybrid variable representation: references_level
/// True iff `t` references level `level` as a genuinely free variable
/// (accounting for t's own internal binders, via the same collect_
/// free_levels/kSafeStartDepth mechanism detail::references_level_at_
/// or_above already uses and this session already confirmed correct).
/// Exposed publicly (unlike references_level_at_or_above) specifically
/// so callers can check, BEFORE invoking cases()/induct() on a given
/// hypothesis, whether that hypothesis's own level actually occurs in
/// the goal's target at all -- induct()'s own branch_expected/ih_ty
/// construction relies on generalize() finding and replacing a marker
/// for the discriminee; when nothing is found (the hypothesis doesn't
/// occur in the target), the collision-avoidance shift baked into
/// generalized_goal specifically to make room for the motive's own
/// extra application layer goes uncompensated, since branch_expected
/// is used directly, standalone -- never actually wrapped in that
/// extra layer. For a constructor whose own branch context doesn't
/// grow enough to make the shifted level valid (e.g. Zero, which adds
/// no new hypotheses at all), this produces an ill-scoped subgoal
/// target -- confirmed directly, by a real crash (Env::operator[]
/// out-of-bounds) once such a subgoal was later evaluated. Skipping
/// cases()/induct() on a hypothesis this returns false for avoids
/// that: inducting on a hypothesis absent from the goal could never
/// help prove it anyway, so this is also a genuine, not merely
/// defensive, search-quality improvement.
inline bool references_level(NodeId t, uint32_t level) {
    constexpr uint32_t kSafeStartDepth = UINT32_MAX - 1'000'000;
    std::unordered_set<uint32_t> free_levels;
    detail::collect_free_levels(t, kSafeStartDepth, free_levels);
    return free_levels.count(level) > 0;
}
#endif


/// Matches a constructor's own index pattern (`pat`, e.g. refl's bare
/// field `x`, or a STRUCTURED pattern like Vec's cons `Succ(n_field)`)
/// against the scrutinee's actual, concrete index (`conc`) at one index
/// slot, threading the result into `out`.
///
/// The central discipline, worth stating explicitly because getting it
/// backwards is exactly the mistake an earlier version of this function
/// made twice (see the design spec's own risk register): this function
/// distinguishes GENUINE, PROVABLE INCONSISTENCY (refutation -- e.g.
/// `Zero` vs `Succ _`, unconditionally distinct constructors of the same
/// family, regardless of what either side's own arguments happen to be)
/// from mere ABSENCE OF INFORMATION (e.g. matching a structured pattern
/// like `Succ(n_field)` against a bare, unrelated free variable `n` --
/// the ordinary shape of a GENERIC, universally-quantified index, not a
/// contradiction of any kind). Only the former is ever a reason to
/// refuse a branch or a whole construction; the latter must always be
/// accepted gracefully, with nothing forced for that slot, because
/// `forced` only ever narrows a later field's declared type as a
/// convenience -- it is never trusted as a soundness-relevant premise by
/// anything downstream (the branch body still has to genuinely
/// type-check against whatever the resulting type is) -- so declining
/// to specialize can only ever make this function accept a STRICTLY
/// LARGER set of valid constructions than requiring more information
/// would, never a wrong one.
inline void match_index_pattern(NodeId pat, NodeId conc, uint32_t base_depth, uint32_t ctor_arity,
                                 const Env& env, CtorIndexForcing& out) {
    if (!out.ok || out.refuted) return; // already settled (failure or refutation) -- stop recursing
    bool pat_is_own_field = false;
    uint32_t pat_level = 0;
    inspect(pat,
        [&](Var v) { if (!v.bound && v.n >= base_depth && v.n < base_depth + ctor_arity) { pat_is_own_field = true; pat_level = v.n; } },
        [](auto) {});
    if (pat_is_own_field) {
        // First occurrence of this field wins for FORCING purposes (see
        // apply_ctor_index_forcing's own doc comment). A REPEATED
        // occurrence of the same field (refl's own (A, x, x) pattern
        // reusing x, or a nested nonlinear pattern) is handled by
        // recursively matching the ALREADY-forced value against this
        // new occurrence's own concrete value -- the SAME function,
        // applied to two already-resolved values instead of a pattern
        // and a concrete term. This is deliberately uniform rather than
        // a separate, groundness-gated special case (an earlier version
        // required BOTH sides to be fully ground before comparing at
        // all, which correctly protected refl's own symbolic (n, m)
        // case but ALSO missed a real refutation whenever one side was
        // merely head-rigid without being fully ground, e.g. `Succ(k)`
        // for a free `k` compared against `Zero` -- exactly the shape
        // the convoy/NoConfusion construction (Step 3) needs to detect,
        // found by tracing that construction through by hand rather
        // than assuming the groundness gate already covered it).
        // Recursing through this same function handles both correctly
        // by construction: comparing rigid heads (Succ vs Zero) refutes
        // regardless of what either side's own arguments are; comparing
        // two genuinely unrelated free variables (n vs m, neither
        // rigid) declines gracefully via the fallback below, exactly as
        // it must.
        auto it = out.forced.find(pat_level);
        if (it != out.forced.end()) {
            match_index_pattern(it->second, conc, base_depth, ctor_arity, env, out);
        } else {
            out.forced.try_emplace(pat_level, conc);
        }
        return;
    }
    if (pat == conc) return; // trivial, syntactically
    detail::AppSpine ps = detail::decompose_app_spine(pat);
    detail::AppSpine cs = detail::decompose_app_spine(conc);
    bool heads_match = false;
    bool distinct_ctors_of_same_family = false;
    inspect(ps.head,
        [&](Const pc) {
            inspect(cs.head,
                [&](Const cc) {
                    if (pc.global_id == cc.global_id) { heads_match = true; return; }
                    const auto& pd = ctx().globals[pc.global_id];
                    const auto& cd = ctx().globals[cc.global_id];
                    if (pd.ctor_info && cd.ctor_info &&
                        pd.ctor_info->ind_type_global_id == cd.ctor_info->ind_type_global_id) {
                        distinct_ctors_of_same_family = true;
                    }
                },
                [](auto) {});
        },
        [](auto) {});
    if (distinct_ctors_of_same_family) {
        // Genuine, provable, unconditional inconsistency -- regardless
        // of what either side's own arguments are (they are never even
        // inspected here). A successful analysis outcome, not an error.
        out.refuted = true;
        out.refuted_pat_value = pat;
        out.refuted_conc_value = conc;
        inspect(cs.head, [&](Const cc) { out.refuted_family_gid = ctx().globals[cc.global_id].ctor_info->ind_type_global_id; }, [](auto) {});
        return;
    }
    if (heads_match && ps.args.size() == cs.args.size()) {
        for (std::size_t i = 0; i < ps.args.size(); ++i) {
            match_index_pattern(ps.args[i], cs.args[i], base_depth, ctor_arity, env, out);
            if (!out.ok || out.refuted) return;
        }
        return;
    }
    // conv() is only safe to attempt when neither side references a
    // level env genuinely has no entry for. This is env.size() itself,
    // not base_depth: an earlier version of this guard used base_depth
    // (reasoning "env only has entries up to base_depth, this
    // constructor's fields haven't been bound in it yet"), which is
    // only a correct proxy for env.size() when a caller happens to pass
    // an env sized to exactly match base_depth -- true for every caller
    // until induct()'s own Mode 1 fix, which deliberately passes
    // g.ec.values (sized to g.ec.size()) against base_depth=body_base (a
    // genuinely LARGER value, spanning self/idx/scrutinee too) to force
    // idx_vars_body -- the Fix's own, always-generic Pi parameters --
    // against ITS OWN fresh fields rather than g.ec's. Using base_depth
    // there crashed outright (Env::operator[] out of bounds, not a
    // graceful decline) the first time an indexed family was inducted
    // on with a genuinely CONCRETE index (Mode 1) rather than a generic
    // one, confirming this by a real failure rather than by inspection
    // alone. env.size() is the actual, direct safety boundary
    // regardless of what base_depth is for in any given call, and is
    // identical to the old behavior for every other existing caller.
    bool pat_unresolved = detail::references_level_at_or_above(pat, static_cast<uint32_t>(env.size()));
    bool conc_unresolved = detail::references_level_at_or_above(conc, static_cast<uint32_t>(env.size()));
    if (!pat_unresolved && !conc_unresolved && conv(pat, conc, env)) return; // definitionally equal
    // Neither provably equal (the conv() check above, when it was even
    // safe to attempt) nor provably inconsistent (rigid, distinct
    // constructor heads, handled above): genuinely insufficient
    // information, not a contradiction -- e.g. pat is a structured
    // pattern (Succ(n_field)) and conc is a bare, unrelated free
    // variable (the ordinary shape of a generic, universally-quantified
    // index -- Mode 2 induction on an indexed type, the common case).
    // Decline gracefully: nothing forced for this slot, `ok` and
    // `refuted` both left unchanged. An earlier version of this
    // function treated this as a hard error instead, which correctly
    // rejected nothing UNSOUND but also incorrectly rejected perfectly
    // ordinary, well-typed constructions -- found by tracing generic
    // (non-concrete-index) induction through by hand, not by any test
    // failing first.
}



inline CtorIndexForcing unify_ctor_pattern_indices(const detail::InstantiatedCtor& inst,
                                                    const std::vector<NodeId>& concrete_indices,
                                                    uint32_t base_depth, uint32_t ctor_arity, const Env& env) {
    CtorIndexForcing out;
    if (inst.result_indices.size() != concrete_indices.size()) {
        out.ok = false;
        out.error = "unify_ctor_pattern_indices: index count mismatch between the constructor's own "
                     "pattern and the scrutinee's concrete indices";
        return out;
    }
    for (std::size_t j = 0; j < inst.result_indices.size(); ++j) {
        match_index_pattern(inst.result_indices[j], concrete_indices[j], base_depth, ctor_arity, env, out);
        if (!out.ok) break;
        if (out.refuted) { out.refuted_slot = static_cast<uint32_t>(j); break; }
    }
    return out;
}


#if 0 // not yet ported to the hybrid variable representation: apply_ctor_index_forcing
/// Applies a CtorIndexForcing's substitution to a constructor's own
/// field types, so a LATER field's declared type correctly reflects an
/// EARLIER field having been forced to a concrete value (e.g. refl's
/// second field `x`, declared with type `A`, becomes declared with type
/// `Nat` once the forcing determines `A := Nat`) -- composing plain
/// substitute_var calls is safe here because a forced value is always
/// either a concrete/ambient term (referencing only levels below this
/// constructor's own fresh fields) or another already-substituted
/// concrete term, never a reference to one of THIS constructor's other,
/// still-generic fields.
inline std::vector<NodeId> apply_ctor_index_forcing(const std::vector<NodeId>& arg_types,
                                                      const CtorIndexForcing& forcing) {
    std::vector<NodeId> out = arg_types;
    for (auto& [level, replacement] : forcing.forced) {
        for (NodeId& ty : out) ty = detail::substitute_var(ty, level, replacement);
    }
    return out;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: apply_ctor_index_forcing_to_result_indices
/// Like apply_ctor_index_forcing, but for a constructor's own
/// result_indices specifically (the values it supplies for the
/// scrutinee's own index positions, e.g. refl's own (A_field, x_field,
/// x_field)) rather than its arg_types. Deliberately EXCLUDES any
/// field whose own level appears more than once, as a bare reference,
/// within result_indices itself: such a field is a REPEATED pattern
/// (refl's own x appearing at both remaining index positions, since
/// its own conclusion is "Eq A x x") and its own, local identity --
/// not whatever single, outer value forcing happens to have resolved
/// it to from just the FIRST occurrence -- is what the motive's own,
/// separate generic parameters at each of those positions must be
/// specialized to: they need to become the SAME repeated field, not
/// two occurrences of one possibly-conflicting outer value squashed
/// together. Concretely: h : Eq Nat n m, cased via refl, forces
/// x_field to n from the first index position alone (the second,
/// x_field =?= m, cannot also hold when n and m are distinct,
/// generic variables, so it is simply never recorded) -- substituting
/// that lone forcing into the SECOND index position too would
/// incorrectly require the branch's own body to prove "Eq Nat n n"
/// instead of "Eq Nat x_field x_field", even though the branch itself,
/// naturally, only ever mentions x_field, never n directly (confirmed
/// directly, this session: forcing.forced held exactly {A_field's own
/// level -> Nat, x_field's own level -> n} for cong's own refl branch,
/// and naively substituting both into result_indices broke cong's own,
/// previously-passing, hand-built proof, which correctly proves the
/// x_field-indexed goal, not the n-indexed one).
///
/// A field occurring exactly once (an ordinary, non-repeated position,
/// like refl's own A) has no such ambiguity: its own, forced, concrete
/// value (not its own, separate field identity) is exactly what every
/// OTHER field's own type is already, consistently substituted to use
/// (via apply_ctor_index_forcing, above) -- leaving IT unforced here
/// instead would create the mirror problem this function exists to
/// avoid, just for that field (confirmed directly, this session, by a
/// "different variable/constant heads" mismatch surviving all the way
/// to whole_term_check for a symm-style proof, before this fix).
inline std::vector<NodeId> apply_ctor_index_forcing_to_result_indices(const std::vector<NodeId>& result_indices,
                                                                       const CtorIndexForcing& forcing) {
    std::unordered_map<uint32_t, uint32_t> occurrence_count;
    for (NodeId n : result_indices) {
        inspect(n, [&](Var v) { occurrence_count[v.level]++; }, [&](auto) {});
    }
    std::vector<NodeId> out = result_indices;
    for (auto& [level, replacement] : forcing.forced) {
        if (occurrence_count[level] > 1) continue; // repeated field: keep its own, local identity
        for (NodeId& n : out) n = detail::substitute_var(n, level, replacement);
    }
    return out;
}
#endif


/// Shared by infer(Match) and check(Match, ...): given the resolved
/// inductive type_global_id and a CONCRETE motive (already constructed
/// or verified by the caller), checks every branch against that motive
/// applied to its constructor's own result indices and the constructor
/// application itself. Mutually recursive with check() (each branch is
/// itself checked, possibly containing further Matches).
///
/// `concrete_indices` is the scrutinee's own ACTUAL indices (e.g. `(Nat,
/// n, m)` for a scrutinee of type `Eq Nat n m`) -- needed for Phase B's
/// index unification (unify_ctor_pattern_indices above); empty for a
/// non-indexed family (Nat/Bool/List), in which case that unification is
/// a no-op and this function's behavior is byte-for-byte unchanged from
/// before Phase B, by construction, not by coincidence.
inline ElabResult check_match_branches(uint32_t type_global_id, NodeId motive, const std::vector<NodeId>& branches,
                                        const std::vector<NodeId>& concrete_indices,
                                        ElabCtx& ec, ElabState& es);
/// Verifies that `motive_id` is well-typed AND function-shaped enough to
/// apply to `arity + 1` arguments (the index count, plus the scrutinee
/// itself) before either infer(Match) or check(Match, ...) trusts a
/// caller-supplied, non-Meta motive_id and applies it. Without this, an
/// arbitrary non-Meta NodeId that was never meant to be inspected as a
/// real motive -- e.g. a placeholder used only because ordinary
/// reduction never looks at motive_id at all, exactly the pattern
/// itp_prelude_demo.cpp's own hand-built Match nodes use -- reaches
/// eval() as an ill-typed application and crashes the process with an
/// uncaught exception rather than failing gracefully. Confirmed
/// empirically, not a hypothetical: this is exactly what happened before
/// this check existed.
inline ElabResult verify_motive_shape(NodeId motive_id, uint32_t arity, ElabCtx& ec, ElabState& es);

// ---- forward declarations for the S2 typing core (defined at the end of this file)
inline TypeResult infer(NodeId t, ElabCtx& ec, ElabState& es);
inline ElabResult check(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es);
inline ElabResult unify(NodeId a, NodeId b, ElabCtx& ec, ElabState& es);
inline NodeId to_levels(NodeId t, uint32_t depth);
#if 0 // not yet ported to the hybrid variable representation: infer
/// Bidirectional type synthesis. Covers every node kind except Match
/// (dependent elimination fundamentally needs a checking-mode target to
/// derive or verify a motive against -- see check()'s Match case, which
/// is the next slice, not yet implemented) and TypeUniv (elaborator-
/// internal to universe-polymorphic template instantiation; not
/// meaningful as a standalone term to infer a type for).
inline TypeResult infer(NodeId t, ElabCtx& ec, ElabState& es) {
    return inspect(t,
        [&](Var v) -> TypeResult {
            if (v.level >= ec.size())
                return std::unexpected(TypeError{TypeErrorKind::UnboundVariable, t, std::nullopt, "unbound local variable"});
            NodeId ty = ec.types[v.level];
            // Re-anchor whenever ec has genuinely grown since v was
            // bound -- NOT specific to infer(App) at all, despite this
            // fix originally living there: confirmed by a second,
            // independent site (check()'s own generic infer()+unify()
            // argument-checking path, comparing an ordinary hypothesis's
            // own declared type against an expected type built at a
            // deeper context) hitting the identical misalignment, via a
            // completely different caller. v's own stored type's
            // internal Pi/Lambda binders are absolute levels that only
            // line up with ordinary application semantics
            // (`env.extend(arg)` inserting at env.size()) when GENUINELY
            // eval()'d against an env of the SAME size ty's own syntax
            // was validated against -- v.level+1 for Fix's own,
            // deliberately self-aware fn_type (whose first real
            // parameter sits at self_level+1), v.level for every
            // ordinary Pi/Lambda/ec.extend() binder (whose own type is
            // validated BEFORE it occupies its own slot).
            //
            // WHICH of these applies is looked up explicitly via
            // ec.is_self_aware(v.level), NOT guessed from ty's own
            // syntax -- an earlier version of this fix tried exactly
            // that (first ">= v.level", then the refined "> v.level"),
            // and both were found genuinely unsound, not just
            // imprecise: Acc_rect's own F parameter has an ORDINARY,
            // non-self-referential type with three Pi layers, whose
            // later layers' own local binders land strictly above F's
            // own level purely as a mechanical consequence of being a
            // multi-parameter dependent type -- syntactically
            // indistinguishable from Fix's fn_type, which also never
            // literally references its own level, only levels above it,
            // by design. No test on ty's own content can separate these
            // two legitimate cases; only recording the true answer at
            // the point the binder was introduced (extend_self_aware(),
            // used at exactly one call site: infer(Fix)'s own self-
            // extension) can.
            if (ec.size() > v.level + 1) {
                uint32_t original_size = ec.is_self_aware(v.level) ? v.level + 1 : v.level;
                Env original_env;
                for (uint32_t i = 0; i < original_size; ++i) original_env = original_env.extend(ec.values[i]);
                ty = quote(eval(ty, original_env), ec.size());
            }
            return ty;
        },
        [&](Const c) -> TypeResult {
            return detail::shift_all_levels(ctx().globals[c.global_id].type, ec.size());
        },
        [&](App a) -> TypeResult {
            ITP_TRY(NodeId fn_ty, infer(a.fn_id, ec, es));
            // The Meta-headed analogue of infer(Var)'s own re-anchoring,
            // closing a genuinely separate gap (found via whole-term
            // check() of an Induct-produced Fix, but not specific to
            // induct() at all): every metavariable in this codebase that
            // is meant to be APPLIED to its own dependencies at each
            // occurrence -- cases()'s ref_meta, apply()'s arg_metas,
            // induct()'s subgoal_meta -- has its DECLARED type built as
            // a genuinely CLOSED, "fresh 0-based scope" Pi-chain, via
            // close_pi_over_context or the equivalent hand-rolled loop
            // (see close_pi_over_context's own doc comment: this is what
            // lets a Miller-pattern metavariable be solved against its
            // full ambient context). That is EXACTLY the same convention
            // shift_all_levels's own doc comment documents for every
            // global's declared type, a few hundred lines above --
            // infer(Const) already re-anchors for it there. infer(Meta)
            // does not: it returns ctx().meta_ctx[m.meta_id].type
            // verbatim, correct for a meta consulted bare (mid-tactic,
            // where ordinary per-goal check()/exact() calls only ever
            // check the SOLUTION supplied for a meta, never this bare
            // application syntax) but wrong here, where a.fn_id is a
            // Meta being applied to arguments: eval()-ing its closed
            // Pi-chain against the CURRENT ec.values (whatever depth
            // this application happens to be re-embedded at -- in
            // practice, the ONLY place this matters, since a Meta-headed
            // application chain elsewhere always gets fully substituted
            // away by eval()+quote() before infer()/check() ever sees it
            // -- the sole exception being syntax nested inside a Match's
            // branches or a Fix's body, which are deliberately never
            // evaluated eagerly, per eval(Match)'s and eval(Fix)'s own
            // "stays stuck/deferred" design) would misplace every one of
            // its own internal Pi-binder positions, for a DIFFERENT
            // structural reason than the Var case (a closed Pi-chain's
            // own binders assume a starting env of size 0, always, not
            // "the size the type happened to be validated at" -- there
            // is no analogous "was it validated before or after its own
            // slot" question for a meta, since a meta's own slot doesn't
            // exist in ec at all). Fix: the identical shift_all_levels
            // treatment infer(Const) already gives a global's declared
            // type, applied here to the meta's own -- confirmed directly
            // against this session's own reproduction (ec.size()=5,
            // headA=LocalVar(4) vs headB=LocalVar(1): the stale,
            // unshifted level baked into subgoal_ty's Pi-chain).
            if (a.fn_id.tag == NodeType::META) {
                fn_ty = detail::shift_all_levels(fn_ty, ec.size());
            }
            Value fn_ty_val = eval(fn_ty, ec.values);
            auto* pi = std::get_if<VPi>(&fn_ty_val);
            if (!pi) return std::unexpected(TypeError{TypeErrorKind::NotAFunction, a.fn_id, std::nullopt,
                                                        "applying a term that is not a function"});
            NodeId dom_ty = quote(eval(pi->dom_id, pi->env), ec.size());
            if (ElabResult r = check(a.arg_id, dom_ty, ec, es); !r) return std::unexpected(r.error());
            Value arg_val = eval(a.arg_id, ec.values);
            Value cod_val = eval(pi->cod_id, pi->env.extend(arg_val));
            return quote(cod_val, ec.size());
        },
        [&](Lambda l) -> TypeResult {
            ElabCtx ec2 = ec.extend(l.type_id);
            ITP_TRY(NodeId body_ty, infer(l.body_id, ec2, es));
            return mk_pi(l.type_id, body_ty); // body_ty's free vars already reference ec2's levels correctly
        },
        [&](Pi p) -> TypeResult {
            ITP_TRY(NodeId dom_sort, infer(p.dom_id, ec, es));
            Value dom_sort_val = eval(dom_sort, ec.values);
            auto* dom_vtype = std::get_if<VType>(&dom_sort_val);
            if (!dom_vtype) return std::unexpected(TypeError{TypeErrorKind::UniverseError, p.dom_id, std::nullopt,
                                                               "Pi domain must be a type"});
            ElabCtx ec2 = ec.extend(p.dom_id);
            ITP_TRY(NodeId cod_sort, infer(p.cod_id, ec2, es));
            Value cod_sort_val = eval(cod_sort, ec2.values);
            auto* cod_vtype = std::get_if<VType>(&cod_sort_val);
            if (!cod_vtype) return std::unexpected(TypeError{TypeErrorKind::UniverseError, p.cod_id, std::nullopt,
                                                               "Pi codomain must be a type"});
            // Impredicative Prop (Open Decision #1, settled): a Prop-sorted
            // (universe 0) codomain makes the WHOLE Pi Prop regardless of
            // the domain's universe -- this is what keeps quantifying over
            // large types inside a proposition weightless (Coq/Lean's
            // actual rule), rather than a strictly predicative max().
            uint32_t result_level = (cod_vtype->universe_level == 0)
                                         ? 0
                                         : std::max(dom_vtype->universe_level, cod_vtype->universe_level);
            return mk_type(result_level);
        },
        [&](Type ty) -> TypeResult { return mk_type(ty.universe_level + 1); },
        [&](Let l) -> TypeResult {
            if (ElabResult r = check(l.value_id, l.type_id, ec, es); !r) return std::unexpected(r.error());
            ElabCtx ec2 = ec.extend(l.type_id);
            return infer(l.body_id, ec2, es);
        },
        [&](Match m) -> TypeResult {
            bool motive_unresolved = false;
            inspect(m.motive_id,
                [&](Meta meta) { if (!ctx().meta_ctx.is_solved(meta.meta_id)) motive_unresolved = true; },
                [&](auto) {});
            if (motive_unresolved) {
                return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, t, std::nullopt,
                    "infer(Match) requires an already-concrete motive; use check() against a goal type "
                    "to have one constructed automatically"});
            }
            ITP_TRY(NodeId scrut_ty, infer(m.scrutinee_id, ec, es));
            Value scrut_ty_val = eval(scrut_ty, ec.values);
            auto* scrut_neutral = std::get_if<VNeutral>(&scrut_ty_val);
            if (!scrut_neutral || scrut_neutral->head.kind != NeutralHead::Kind::GlobalConst) {
                return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, std::nullopt,
                    "scrutinee's type is not an inductive type application"});
            }
            uint32_t type_global_id = scrut_neutral->head.id;
            std::vector<NodeId> concrete_indices;
            for (auto& frame : scrut_neutral->spine) {
                auto* af = std::get_if<ApplyFrame>(&frame);
                if (!af) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, std::nullopt,
                    "scrutinee's type has a non-application eliminator frame; cannot enumerate its indices"});
                concrete_indices.push_back(quote(af->arg, ec.size()));
            }
            if (ElabResult shape_r = verify_motive_shape(m.motive_id, static_cast<uint32_t>(concrete_indices.size()), ec, es);
                !shape_r) {
                return std::unexpected(shape_r.error());
            }
            if (ElabResult r = check_match_branches(type_global_id, m.motive_id, *m.branches, concrete_indices, ec, es); !r)
                return std::unexpected(r.error());
            NodeId result = apply_all(m.motive_id, concrete_indices);
            result = mk_app(result, m.scrutinee_id);
            return quote(eval(result, ec.values), ec.size());
        },
        [&](Meta m) -> TypeResult { return ctx().meta_ctx[m.meta_id].type; },
        [&](TypeUniv) -> TypeResult {
            return std::unexpected(TypeError{TypeErrorKind::Other, t, std::nullopt,
                "TypeUniv is elaborator-internal and not meaningful as a standalone infer target"});
        },
        [&](QuotType q) -> TypeResult {
            ITP_TRY(NodeId rel_ty, infer(q.relation, ec, es));
            Value rel_ty_val = eval(rel_ty, ec.values);
            auto* outer_pi = std::get_if<VPi>(&rel_ty_val);
            if (!outer_pi) return std::unexpected(TypeError{TypeErrorKind::NotAFunction, q.relation, std::nullopt,
                                                              "Quot's relation must be a binary relation A -> A -> Sort"});
            NodeId elem_ty = quote(eval(outer_pi->dom_id, outer_pi->env), ec.size());
            return infer(elem_ty, ec, es); // the quotient lives in the same sort as the underlying type
        },
        [&](QuotMk q) -> TypeResult {
            ITP_TRY(NodeId rel_ty, infer(q.relation, ec, es));
            Value rel_ty_val = eval(rel_ty, ec.values);
            auto* outer_pi = std::get_if<VPi>(&rel_ty_val);
            if (!outer_pi) return std::unexpected(TypeError{TypeErrorKind::NotAFunction, q.relation, std::nullopt,
                                                              "Quot's relation must be a binary relation A -> A -> Sort"});
            NodeId elem_ty = quote(eval(outer_pi->dom_id, outer_pi->env), ec.size());
            if (ElabResult r = check(q.elem, elem_ty, ec, es); !r) return std::unexpected(r.error());
            return mk_quot_type(q.relation);
        },
        [&](QuotLift q) -> TypeResult {
            if (ElabResult r = check(q.quotient_elem, mk_quot_type(q.relation), ec, es); !r) return std::unexpected(r.error());
            return q.codomain;
        },
        [&](Fix f) -> TypeResult {
            // fn_type is now (per induct()'s own updated construction --
            // see its comment) ALREADY built self-aware: its own
            // internal binders (any idx parameters, then the scrutinee)
            // assume self occupies the FIRST new slot at ec.size(),
            // before them, even though fn_type's own content never
            // actually references that slot (self's own type isn't
            // self-referential). So fn_type must be validated against
            // ec2 (self already added), not ec -- infer(Pi)'s own
            // natural per-layer extension then lands fn_type's first
            // REAL binder (idx0, or the scrutinee if non-indexed) at
            // ec2.size() = ec.size()+1, exactly where fn_type's own
            // absolute levels already assume it to be. This replaces
            // the earlier "validate against ec, then re-anchor via
            // quote(eval(...), ec2.size()) before reuse" approach: that
            // re-anchoring was individually correct for fn_type's own
            // new binders, but compounded with a LATER, also
            // individually-correct shift (check_match_branches' own,
            // when this same Fix -- re-embedded deeper via quote(VFix)
            // during proof extraction -- is independently re-validated
            // by a whole-term check() and its own Succ branch is
            // checked), over-shifting g.target's own pre-existing
            // embedded content one time more than intended. Building
            // fn_type self-aware from construction, and validating it
            // directly against the context it already assumes, avoids
            // that compounding entirely rather than trying to correct
            // for it after the fact.
            //
            // extend_self_aware() (not plain extend()): marks this
            // level explicitly so infer(Var) can later, reliably answer
            // "was self's own type validated including its own slot"
            // without guessing from fn_type's own syntax -- no
            // syntactic test can do this soundly, confirmed by a real
            // counterexample; see SelfAwareMarker's own comment.
            ElabCtx ec2 = ec.extend_self_aware(f.fn_type);
            ITP_TRY(NodeId fn_sort, infer(f.fn_type, ec2, es));
            Value fn_sort_val = eval(fn_sort, ec2.values);
            if (!std::get_if<VType>(&fn_sort_val)) {
                return std::unexpected(TypeError{TypeErrorKind::UniverseError, f.fn_type, std::nullopt,
                    "Fix's own fn_type must be a type"});
            }
            // The standard fixpoint typing rule: Gamma, self:fn_type |-
            // body : fn_type. decreasing_arg is deliberately not
            // consulted here at all -- guardedness is a SEPARATE check
            // (is_locally_structurally_decreasing), run once at
            // Induct-tactic-acceptance time, exactly mirroring how
            // finish_recursive_function's structural-decreasing check
            // is independent of (and additional to) ordinary
            // typechecking for an ORDINARY recursive definition.
            //
            // f.fn_type is used DIRECTLY here now, unchanged -- no
            // re-anchoring needed, since it was just validated against
            // this exact ec2 above and is therefore already correct
            // relative to it.
            if (ElabResult r = check(f.body, f.fn_type, ec2, es); !r) return std::unexpected(r.error());
            // The RETURN value, by contrast, must be relative to ec
            // (the caller's own context, without self) -- f.fn_type
            // itself never references self's own slot (ec.size()), so
            // this is a genuine compression, not a lossy truncation:
            // evaluating against ec2's own values then quoting back at
            // ec.size() runs the standard NBE round-trip, which
            // correctly re-derives every one of fn_type's own binders
            // one level lower (matching ec.size() rather than ec2.size()
            // as the new base), while any genuinely free content below
            // ec.size() (ordinary ambient hypotheses) round-trips
            // unchanged, exactly as this same technique already does
            // elsewhere in this file.
            return quote(eval(f.fn_type, ec2.values), ec.size());
        }
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: check
/// Bidirectional type checking against an expected type. Only Lambda
/// (letting an expected Pi drive elaboration of the body, including any
/// holes inside it) and Match (the next slice: motive inference driven
/// by the expected type) need dedicated handling beyond infer+unify;
/// everything else falls through to the generic case.
inline ElabResult check(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {
    return inspect(t,
        [&](Lambda l) -> ElabResult {
            Value expected_val = eval(expected_ty, ec.values);
            auto* pi = std::get_if<VPi>(&expected_val);
            if (!pi) return std::unexpected(TypeError{TypeErrorKind::Mismatch, t, expected_ty,
                                                        "checking a lambda against a non-function expected type"});
            NodeId expected_dom = quote(eval(pi->dom_id, pi->env), ec.size());
            if (ElabResult r = unify(l.type_id, expected_dom, ec, es); !r) return r;
            ElabCtx ec2 = ec.extend(l.type_id);
            // pi->cod_id's own bound variable already, correctly
            // references exactly pi->env.size() -- but unlike intro()
            // (where the goal being peeled is always guaranteed to be a
            // bare, direct Pi node), expected_ty here can be a more
            // complex expression (e.g. the result of an application
            // chain) whose own evaluation genuinely captures a DIFFERENT
            // env than ec.values -- so this is checked explicitly,
            // rather than assumed. When it holds (the common case: a
            // bare Pi, or an application chain that happens to have
            // fully consumed its own arguments back down to ec.size()),
            // pi->cod_id can be used directly, with no eval()+quote()
            // round-trip at all -- avoiding the same unnecessary,
            // eager unfolding (of any global Fix definition reachable
            // from pi->cod_id, e.g. add's own recursive definition)
            // that intro()'s own, now-fixed version used to bake into
            // every goal it touched. When it does not hold, fall back
            // to the original, safe (if occasionally over-eager)
            // re-anchoring -- correctness first, this optimization only
            // where it is actually valid.
            NodeId expected_cod;
            if (pi->env.size() == ec.size()) {
                expected_cod = pi->cod_id;
            } else {
                Value fresh = fresh_neutral(ec.size());
                expected_cod = quote(eval(pi->cod_id, pi->env.extend(fresh)), ec2.size());
            }
            return check(l.body_id, expected_cod, ec2, es);
        },
        [&](Match m) -> ElabResult {
            TypeResult scrut_ty_r = infer(m.scrutinee_id, ec, es);
            if (!scrut_ty_r) return std::unexpected(scrut_ty_r.error());
            Value scrut_ty_val = eval(*scrut_ty_r, ec.values);
            auto* scrut_neutral = std::get_if<VNeutral>(&scrut_ty_val);
            if (!scrut_neutral || scrut_neutral->head.kind != NeutralHead::Kind::GlobalConst) {
                return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, expected_ty,
                    "scrutinee's type is not an inductive type application"});
            }
            uint32_t type_global_id = scrut_neutral->head.id;
            std::vector<NodeId> concrete_indices;
            for (auto& frame : scrut_neutral->spine) {
                auto* af = std::get_if<ApplyFrame>(&frame);
                if (!af) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, expected_ty,
                    "scrutinee's type has a non-application eliminator frame; cannot enumerate its indices"});
                concrete_indices.push_back(quote(af->arg, ec.size()));
            }
            uint32_t arity = static_cast<uint32_t>(concrete_indices.size());

            bool motive_unresolved = false;
            uint32_t hole_meta_id = 0;
            inspect(m.motive_id,
                [&](Meta meta) { if (!ctx().meta_ctx.is_solved(meta.meta_id)) { motive_unresolved = true; hole_meta_id = meta.meta_id; } },
                [&](auto) {});

            NodeId motive;
            if (motive_unresolved) {
                // Auto-construct: generalize the goal over the concrete
                // indices and the scrutinee itself, then TYPE-CHECK the
                // result -- that check succeeding or failing is the
                // actual "can this be abstracted cleanly" detection
                // mechanism (see design note above generalize()), not a
                // separate syntactic heuristic.
                std::vector<std::pair<NodeId, uint32_t>> targets;
                for (uint32_t i = 0; i < arity; ++i) targets.push_back({concrete_indices[i], ec.size() + i});
                targets.push_back({m.scrutinee_id, ec.size() + arity});
                NodeId generalized_goal = detail::generalize(expected_ty, targets);

                std::vector<NodeId> index_types = detail::instantiate_type_telescope(type_global_id, ec.size());
                if (index_types.size() != arity) {
                    return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, t, expected_ty,
                        "index count mismatch between the scrutinee's type application and the "
                        "inductive's own declared arity"});
                }
                std::vector<NodeId> idx_vars;
                for (uint32_t i = 0; i < arity; ++i) idx_vars.push_back(mk_var(ec.size() + i));
                NodeId scrut_binder_ty = apply_all(mk_const(type_global_id), idx_vars);
                NodeId built = mk_lambda(scrut_binder_ty, generalized_goal);
                for (auto it = index_types.rbegin(); it != index_types.rend(); ++it) built = mk_lambda(*it, built);

                TypeResult motive_ty_r = infer(built, ec, es);
                if (!motive_ty_r) {
                    return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, t, expected_ty,
                        "constructed motive failed to type-check -- the goal likely depends on the "
                        "scrutinee in a way that cannot be abstracted cleanly; supply an explicit "
                        "motive_id instead"});
                }
                ctx().meta_ctx.solve(hole_meta_id, built);
                motive = built;
            } else {
                // Caller supplied an explicit motive: verify it's
                // actually well-typed and function-shaped before ever
                // trusting it -- see verify_motive_shape's doc comment
                // for why this is load-bearing, not defensive
                // over-caution.
                if (ElabResult shape_r = verify_motive_shape(m.motive_id, arity, ec, es); !shape_r) return shape_r;
                motive = m.motive_id;
                NodeId applied = apply_all(motive, concrete_indices);
                applied = mk_app(applied, m.scrutinee_id);
                if (ElabResult r = unify(applied, expected_ty, ec, es); !r) return r;
            }
            return check_match_branches(type_global_id, motive, *m.branches, concrete_indices, ec, es);
        },
        [&](auto) -> ElabResult {
            TypeResult inferred = infer(t, ec, es);
            if (!inferred) return std::unexpected(inferred.error());
            return unify(*inferred, expected_ty, ec, es);
        }
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: verify_motive_shape
inline ElabResult verify_motive_shape(NodeId motive_id, uint32_t arity, ElabCtx& ec, ElabState& es) {
    TypeResult motive_ty_r = infer(motive_id, ec, es);
    if (!motive_ty_r) {
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive_id, std::nullopt,
            "explicit motive_id is not well-typed"});
    }
    Value motive_ty_val = eval(*motive_ty_r, ec.values);
    for (uint32_t i = 0; i <= arity; ++i) {
        auto* mpi = std::get_if<VPi>(&motive_ty_val);
        if (!mpi) {
            return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive_id, std::nullopt,
                "explicit motive_id is not shaped like a valid motive -- expected a Pi chain of "
                "(index count + 1) arguments, ending in a sort"});
        }
        // Advance through the Pi chain the same way infer(App) would;
        // only the SHAPE matters here, so a fresh placeholder neutral is
        // enough to open each layer.
        motive_ty_val = eval(mpi->cod_id, mpi->env.extend(fresh_neutral(ec.size())));
    }
    return {};
}
#endif


#if 0 // not yet ported to the hybrid variable representation: check_match_branches
inline ElabResult check_match_branches(uint32_t type_global_id, NodeId motive, const std::vector<NodeId>& branches,
                                        const std::vector<NodeId>& concrete_indices,
                                        ElabCtx& ec, ElabState& es) {
    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(type_global_id);
    if (ctor_gids.size() != branches.size()) {
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive, std::nullopt,
            "Match branch count does not match the scrutinee's inductive type's constructor count "
            "(non-exhaustive or over-long match)"});
    }
    for (std::size_t ci = 0; ci < ctor_gids.size(); ++ci) {
        detail::InstantiatedCtor inst = detail::instantiate_ctor_telescope(ctor_gids[ci], ec.size());
        uint32_t arity = static_cast<uint32_t>(inst.arg_types.size());
        // Phase B: this constructor's own index pattern (inst.result_indices,
        // e.g. refl's (A_field, x_field, x_field)) is unified against the
        // scrutinee's ACTUAL, concrete indices (concrete_indices, e.g.
        // (Nat, n, m)) -- forcing any of this constructor's own fresh
        // fields whose value the pattern determines (A_field := Nat here)
        // to that concrete value THROUGHOUT this constructor's own field
        // types, before they are used below. Without this, a branch body
        // needing to exploit exactly this kind of forced equality (e.g.
        // using refl's own second field, declared with the family's fully
        // generic index type, as if it already had the family's concrete
        // index type) cannot type-check -- confirmed directly, not
        // assumed; see unify_ctor_pattern_indices's own doc comment.
        // concrete_indices is empty for a non-indexed family (Nat/Bool/
        // List), in which case unify_ctor_pattern_indices's own loop
        // never executes and this is a genuine no-op: the byte-for-byte
        // behavior preserved for every existing, already-passing test.
        CtorIndexForcing forcing = unify_ctor_pattern_indices(inst, concrete_indices, ec.size(), arity, ec.values);
        if (!forcing.ok) {
            return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, branches[ci], std::nullopt,
                forcing.error});
        }
        // forcing.refuted (this branch is definitionally unreachable) is
        // not yet exploited to accept an arbitrary body -- see
        // unify_ctor_pattern_indices's own doc comment on why. Whatever
        // forcing succeeded up to the refutation is still applied below;
        // the branch is still checked normally against it.
        std::vector<NodeId> arg_types = apply_ctor_index_forcing(inst.arg_types, forcing);
        // Each branch is a genuine curried Lambda over the constructor's
        // own arguments (matching how eval(Match) actually applies a
        // matched branch to the constructor's captured fields -- see
        // itp_prelude_demo.cpp's hand-built branches for the same
        // convention, e.g. add's Succ branch is mk_lambda(nat_ty, ...),
        // not a bare term checked in an already-extended context). So
        // the expected type here must be the FULL Pi chain over
        // arg_types (the constructor's own field types, WITH Phase B's
        // index forcing already applied), checked against the ORIGINAL
        // (unextended) ec -- check()'s own Lambda case is what descends
        // one binder at a time, the same way it already does for every
        // other Lambda in this elaborator, not a second, parallel
        // extension mechanism here.
        //
        // `motive` must be RE-ANCHORED to land its own next-available
        // binder at exactly ec.size(), not assumed to already be there.
        // shift_levels_from(motive, ec.size(), arity) (this function's
        // own earlier version) implicitly assumed motive's own first
        // binder sits at ec.size() -- true for the common case (a motive
        // built fresh, right here, by check(Match)'s own auto-construct
        // path, or the very first time an original, tactic-built
        // match_term's own motive is checked at exactly the g.ec.size()
        // it was designed for) but false whenever this Match node
        // carries an INHERITED motive re-embedded deeper than its own
        // original design depth -- which is exactly what happens when a
        // Fix's own body (containing this Match) is reconstructed by
        // quote(VFix)/requote_captured_syntax during proof extraction
        // and independently re-validated by a whole-term check(): the
        // motive's own scrutinee parameter is, by induct()'s own
        // construction, numerically IDENTICAL to the Fix's own idx0/
        // scrutinee parameter (by design -- the motive genuinely must
        // depend on that same variable for the tactic-level proof to
        // type-check at all), not a fresh, generic placeholder assumed
        // to sit at whatever ec.size() happens to be once more Lambda
        // layers (this specific branch's own constructor fields, or
        // anything else) have since been entered. shift_levels_from,
        // with threshold=ec.size(), simply misses a binder that sits
        // BELOW that threshold entirely, leaving it unshifted and
        // uninvolved in the substitution apply_all performs below --
        // so the substitution silently misses its target, and the
        // motive's own scrutinee reference resolves to whatever
        // unrelated value already occupies that absolute level instead
        // of the newly-substituted constructor application. Found
        // directly by this session's own instrumentation: print(quote(
        // eval(...))) on both sides of the resulting mismatch showed an
        // extra, genuinely wrong level -- not a checker bug (unify
        // correctly rejected the two, different values) but exactly
        // this construction bug. The fix mirrors the same quote(eval(
        // ...), new_depth) re-anchoring technique used pervasively
        // elsewhere in this file (normalize(), requote_captured_syntax,
        // infer(Fix)'s own construction above) rather than assuming any
        // fixed relationship between a motive's own binder positions and
        // the current ec.size(): evaluate motive ONCE, correctly,
        // against its own actual, current-context values (which are
        // already correct, regardless of where its own binder sits,
        // since only NEW binder positions were ever in question), then
        // quote it back out at ec.size()+arity, which places its own
        // next-available binder at EXACTLY ec.size() (this branch's own
        // new field binders start there) by construction, regardless of
        // whatever absolute level that binder happened to already carry.
        //
        // Confirmed this session that this specific offset is NOT the
        // Phase B gap: removing it broke induct()'s own passing test and
        // did not fix the indexed-match case, which turned out to need
        // the CtorIndexForcing step above instead -- both this offset
        // and Phase B's forcing coexist correctly, independently.
        NodeId shifted_motive = quote(eval(motive, ec.values), ec.size() + arity);
        NodeId ctor_app = mk_const(ctor_gids[ci]);
        for (uint32_t ai = 0; ai < arity; ++ai) ctor_app = mk_app(ctor_app, mk_var(ec.size() + ai));
        // Forced too, exactly like arg_types just above -- see cases()'
        // own, identical fix (same file) for the full reasoning: without
        // this, the motive's own substitution would use a constructor
        // field's own, unforced identity (e.g. A_field) precisely where
        // arg_types' own forcing already, correctly substitutes the
        // concrete value (A) into the OTHER fields' own types -- two
        // syntactically distinct NodeIds meant to be equal, causing a
        // genuine mismatch once this branch is checked against
        // anything built from arg_types' own, forced identities.
        std::vector<NodeId> forced_result_indices = apply_ctor_index_forcing_to_result_indices(inst.result_indices, forcing);
        NodeId branch_expected = apply_all(shifted_motive, forced_result_indices);
        branch_expected = mk_app(branch_expected, ctor_app);
        for (auto it = arg_types.rbegin(); it != arg_types.rend(); ++it) {
            branch_expected = mk_pi(*it, branch_expected);
        }
        if (ElabResult r = check(branches[ci], branch_expected, ec, es); !r) return r;
    }
    return {};
}
#endif


// ---------------------------------------------------------------------------
// Phase 2: ProofState -- goal-directed tactic engine
// ---------------------------------------------------------------------------
//
// Turns the elaborator above into a top-down proof-search interface:
// instead of "elaborate this whole term against this whole type in one
// shot," a small set of tactic primitives incrementally refine a list of
// open goals. An open goal is nothing but an unsolved MetaEntry plus the
// hypothesis telescope in scope for it -- MetaContext was chosen as the
// goal-representation primitive back in Phase 1 specifically so this
// falls out for free, rather than inventing a parallel notion of "goal."
//
// Each Goal owns its OWN ElabCtx (not one shared, mutable context across
// the whole ProofState): tactics that branch (Cases, later Apply) need
// sibling goals with genuinely different, independently-extended
// hypothesis telescopes, and a single shared ElabCtx cannot represent
// that. This costs nothing extra in the common (non-branching, or
// branching-then-extending-independently) case -- ElabCtx's LocalCtx and
// Env are both persistent tries, so sibling goals sharing a hypothesis
// PREFIX share that prefix's structure automatically, the same way any
// other use of these structures already does.
struct Hypothesis {
    std::string name; // display/lookup name; not load-bearing for
                       // correctness (ElabCtx's own levels are) --
                       // purely for human/agent-readable goal printing
                       // and future name-based tactic arguments
    NodeId type;
};

struct Goal {
    uint32_t meta_id;
    ElabCtx ec;
    std::vector<Hypothesis> hyps; // parallel to ec.types, hyps[i] describes level i
    NodeId target;                // this goal's own type; kept alongside meta_id
                                   // for cheap access without a MetaContext lookup
                                   // on every tactic step (must always equal
                                   // ctx().meta_ctx[meta_id].type)
};

struct ProofState {
    std::vector<Goal> goals;
    ElabState es; // SHARED across the whole attempt: a constraint deferred
                  // while solving one goal may only become solvable once a
                  // SIBLING goal's meta is resolved, so this cannot be
                  // scoped to a single Goal
};

/// Locates which entry of ps.goals a tactic should operate on: the goal
/// whose own meta_id matches target_meta_id if one is given, otherwise
/// ps.goals.front()'s own meta_id -- preserving every existing call
/// site's current behavior exactly when target_meta_id is omitted
/// (std::nullopt), since every tactic's own optional target_meta_id
/// parameter defaults to nullopt.
///
/// Exists so a search algorithm exploring multiple pending goals (e.g.
/// an AND-OR/hypergraph search, where a tactic producing several
/// subgoals means all of them need independent attention, not
/// necessarily in the order they were produced) can target any open
/// goal directly, without needing to first fully resolve every earlier
/// goal in ps.goals down to nothing -- which the previous, hardcoded-
/// front()-only convention required.
///
/// Targeting by meta_id rather than by raw vector position is
/// deliberate, not incidental: ps.goals's own positions shift every
/// time any earlier goal is split into multiple subgoals, so a
/// search algorithm's own bookkeeping (e.g. "this MCTS node corresponds
/// to goal index 4") would silently go stale the moment ANY other goal
/// gets expanded. meta_id is stable -- assigned once, by
/// MetaContext::fresh(), and never reassigned -- so a caller can hold a
/// meta_id across any number of intervening tactic calls on OTHER
/// goals and still correctly retarget this one.
///
/// Returns nullopt if ps.goals is empty, or if target_meta_id was given
/// but no longer matches any entry -- an expected, non-error situation
/// for a search algorithm holding a handle to a goal that has since
/// been solved or superseded by a sibling's own tactic application, not
/// something callers should treat as a bug.
inline std::optional<std::size_t> find_goal_index(const ProofState& ps, std::optional<uint32_t> target_meta_id) {
    if (ps.goals.empty()) return std::nullopt;
    if (!target_meta_id) return 0;
    for (std::size_t i = 0; i < ps.goals.size(); ++i) {
        if (ps.goals[i].meta_id == *target_meta_id) return i;
    }
    return std::nullopt;
}


#if 0 // not yet ported to the hybrid variable representation: close_over_context
/// Wraps `term` in one Lambda per entry of `ec` (in order, innermost
/// entry closest to `term`), producing a term with ZERO free variables
/// relative to `ec`'s own ambient levels -- i.e. safe to pass to
/// MetaContext::solve() regardless of what `term` references.
///
/// This is load-bearing, not a convenience wrapper: eval(Meta)'s own doc
/// comment states the exact convention every solution must follow --
/// "the solution is expected to be closed with respect to the
/// metavariable's own local context... a real elaborator that wants a
/// solution to depend on local variables is expected to close over them
/// explicitly" -- because eval(Meta) always evaluates a solved meta's
/// solution in a bare, empty Env{}, never the ambient one. Phase 1's
/// solve_pattern already had to do exactly this for Miller-pattern
/// solutions (built as `lambda(pattern var types...). body`); ProofState
/// needs the identical discipline for every tactic that solves a goal's
/// meta while hypotheses are in scope, which is effectively all of them
/// once a proof has more than one Intro. Skipping this produces a
/// solution that is syntactically fine but crashes eval() the moment
/// anything (extract_proof_term, a later tactic, printing) actually
/// forces it -- confirmed directly during this session's own first
/// end-to-end Intro+Intro+Exact test, not a hypothetical risk.
///
/// Correctness of the level numbering here relies on one property:
/// ec.types[i], for ANY goal anywhere in a ProofState, is always
/// numbered relative to "level 0 = whatever was first introduced since
/// start_proof", the same root every other goal's ec ultimately traces
/// back to (goals only ever grow an ElabCtx via .extend(), never
/// renumber one) -- so re-wrapping ec.types[0..n-1] as closing Lambdas,
/// in the same order, reproduces exactly the level numbering those
/// types already had, with no additional shifting needed.
inline NodeId close_over_context(NodeId term, const ElabCtx& ec) {
    NodeId result = term;
    for (uint32_t i = ec.size(); i-- > 0; ) {
        result = mk_lambda(ec.types[i], result);
    }
    return result;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: close_pi_over_context
/// The Pi-chain counterpart of close_over_context: wraps `ty` in one Pi
/// per entry of `ec`, giving the FUNCTION TYPE a fresh metavariable
/// needs if it's meant to be usable as a Miller pattern applied to ec's
/// own full context (`?m x0 x1 ... x(n-1)`) -- see apply()'s doc comment
/// for why this matters: a bare (unwrapped, unapplied) metavariable can
/// only ever be solved by a fully CLOSED term (its Miller-pattern spine
/// is empty, so solve_pattern's own scope check rejects any solution
/// referencing an ambient hypothesis at all), which is far too
/// restrictive for an Apply argument that a unification against the
/// goal is expected to determine using the goal's own hypotheses.
inline NodeId close_pi_over_context(NodeId ty, const ElabCtx& ec) {
    NodeId result = ty;
    for (uint32_t i = ec.size(); i-- > 0; ) {
        result = mk_pi(ec.types[i], result);
    }
    return result;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: start_proof
inline ProofState start_proof(NodeId goal_type) {
    uint32_t root_meta = ctx().meta_ctx.fresh(goal_type, 0);
    ProofState ps;
    ps.goals.push_back(Goal{root_meta, ElabCtx{}, {}, goal_type});
    return ps;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: proof_complete
inline bool proof_complete(const ProofState& ps) {
    return ps.goals.empty() && ps.es.all_constraints_resolved();
}
#endif


#if 0 // not yet ported to the hybrid variable representation: extract_proof_term
/// Recovers the concrete proof term once a proof is complete (or, for
/// inspection mid-proof, whatever partial term has been committed so
/// far -- solved metas unfold transparently via eval(Meta), unsolved
/// ones remain visible as holes in the result). Correct as written ONLY
/// for the ROOT meta specifically: its own scope_depth is always 0 (see
/// start_proof), so it has no ambient dependencies to worry about --
/// every OTHER meta's solution must be produced via close_over_context
/// before extraction would make sense for it too.
inline NodeId extract_proof_term(uint32_t root_meta_id) {
    NoDeltaScope keep_constants_folded; // F2a: do not inline library theorems/definitions into the proof
    return quote(eval(mk_meta(root_meta_id), {}), 0);
}
#endif


/// GlobalEnv::declare() is, by itself, a TRUSTED registration: it never
/// calls check(value, type, ...) -- confirmed directly this session by
/// reading its own implementation, and confirmed to matter in practice
/// by cong's own internal Match (in itp_prelude_demo.cpp/
/// itp_kernel_test.cpp) going unverified by this elaborator for the
/// entire life of this project until something else (a whole-term
/// check() forced open by proof extraction) happened to force it open.
/// finish_recursive_function has the identical gap: it verifies
/// guardedness/termination only, never that the body actually has the
/// declared type.
///
/// declare_checked closes this for the plain (non-recursive) case: it
/// calls check(value, type, ...) BEFORE registering, in an EMPTY
/// context, matching how every global's own declared type is meant to
/// be interpreted (see shift_all_levels's own doc comment: a global's
/// type is "closed", i.e. elaborated relative to no ambient
/// hypotheses at all) -- so nothing is ever installed into GlobalEnv
/// that this elaborator has not itself independently verified. A value
/// that check() correctly rejects is still correctly rejected here, not
/// silently installed anyway -- confirmed directly and repeatedly
/// across this project's history, most recently this session, using
/// exactly this function (and finish_recursive_function_checked, its
/// sibling for a genuinely self-referential global) to hand-build and
/// independently verify add_zero_right and add_succ_right from
/// scratch, entirely bypassing a since-fixed bug in the tactic-driven
/// extraction path (resolve_metas/requote_captured_syntax's own
/// handling of a solved meta's solution re-embedded across a Fix
/// boundary) -- exactly the kind of independent, from-scratch route
/// around a tactic-layer bug this function's own trust boundary exists
/// to provide.
inline std::expected<uint32_t, TypeError> declare_checked(NodeId type, NodeId value, std::string name) {
    ElabState es;
    ElabCtx empty_ec;
    ElabResult r = check(value, type, empty_ec, es);
    if (!r) return std::unexpected(r.error());
    return ctx().globals.declare(type, std::move(value), std::move(name));
}


#if 0 // not yet ported to the hybrid variable representation: intro
/// Strips one Pi off the current goal, introduces a hypothesis, and
/// commits the OUTGOING goal's proof term to "a lambda around whatever
/// the new goal ends up proving" -- Intro never branches: exactly zero
/// goals result if there was nothing to introduce (an error), otherwise
/// exactly one (the extended goal, replacing the old one in place).
inline ElabResult intro(ProofState& ps, const std::string& name = "", std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) {
        return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt,
            target_meta_id ? "intro: no goal with the given meta_id (already solved or superseded)"
                            : "intro: no open goals"});
    }
    Goal& g = ps.goals[*idx_opt];
    Value target_val = eval(g.target, g.ec.values);
    auto* pi = std::get_if<VPi>(&target_val);
    if (!pi) {
        return std::unexpected(TypeError{TypeErrorKind::Mismatch, g.target, std::nullopt,
            "intro: goal is not a Pi type, nothing to introduce"});
    }
    // pi->env is ALWAYS exactly g.ec.values here (eval(Pi, env) wraps env
    // directly, with no intermediate application at all) -- so pi->dom_id
    // and pi->cod_id are already correct, as-is, relative to g.ec: their
    // own free variables already match g.ec's own levels (0..g.ec.size()
    // -1), and cod_id's own bound variable (the Pi's own parameter)
    // already, correctly references exactly g.ec.size() -- precisely
    // where new_ec's own new hypothesis will land below. No re-leveling
    // is needed at all, so none is done: quote(eval(pi->dom_id, pi->env),
    // g.ec.size()) and quote(eval(pi->cod_id, pi->env.extend(fresh)),
    // g.ec.size()+1) -- this function's own earlier version -- forced a
    // full NBE round-trip for no positional reason, and eval() does not
    // merely re-level: it unfolds every global Fix definition it can
    // reach. For a goal like "Eq Nat (add n Zero) n", this ate the
    // ENTIRE goal, permanently baking add's own one-step-unfolded,
    // stuck-match reconstruction into g.target's own long-lived syntax
    // -- rather than the flat, unreduced "add n Zero" the goal was
    // originally, deliberately written with. That reconstruction's own
    // embedded content (the stuck match's own Succ-branch field) is
    // exactly the "pre-existing embedded content needing correct
    // clearance from new binders" this session spent two full sessions
    // chasing across ih_ty, branch_expected, fn_type, and motive_body
    // -- all of which only ever needed to re-shift it because intro()
    // had already, needlessly, introduced it into the goal in the first
    // place. Using pi->dom_id/pi->cod_id directly keeps the goal exactly
    // as flat as it was written; downstream evaluation (check(Lambda)'s
    // own, transient, per-use eval(), for instance) still, correctly,
    // eagerly unfolds "add n Zero" whenever it is genuinely needed --
    // but freshly, relative to whatever context is actually current at
    // that point, never as a stale snapshot baked in at a DIFFERENT
    // depth by this function, once, long before any of this proof's own
    // later re-embedding could possibly be anticipated.
    NodeId dom_ty = pi->dom_id;
    NodeId new_target = pi->cod_id;

    ElabCtx new_ec = g.ec.extend(dom_ty);
    std::vector<Hypothesis> new_hyps = g.hyps;
    new_hyps.push_back(Hypothesis{name, dom_ty});

    uint32_t new_meta = ctx().meta_ctx.fresh(new_target, new_ec.size());
    // new_meta, referenced applied to its OWN full local context (levels
    // 0..new_ec.size()-1) -- per close_over_context's doc comment, a
    // solution can never reference ambient levels bare.
    NodeId new_meta_ref = mk_meta(new_meta);
    for (uint32_t i = 0; i < new_ec.size(); ++i) new_meta_ref = mk_app(new_meta_ref, mk_var(i));

    // Commit the outgoing goal's proof term BEFORE replacing it -- once
    // g.meta_id is solved, the new Goal below is the only thing that can
    // still be worked on, matching every other tactic's "solve outgoing,
    // replace with incoming" pattern. close_over_context supplies the
    // g.ec.size() outer Lambdas needed to close over g's OWN ambient
    // dependencies; the inner mk_lambda(dom_ty, ...) is the one new
    // binder THIS call itself introduces.
    ctx().meta_ctx.solve(g.meta_id, close_over_context(mk_lambda(dom_ty, new_meta_ref), g.ec));

    ps.goals[*idx_opt] = Goal{new_meta, new_ec, std::move(new_hyps), new_target};
    return {};
}
#endif


#if 0 // not yet ported to the hybrid variable representation: exact
/// Closes the current goal outright by checking a caller-supplied term
/// against it and, on success, solving the goal's meta and removing it
/// from the open list. On failure, ps.goals is left untouched by THIS
/// function -- but check()/unify() may have solved OTHER, unrelated
/// metas along the way before ultimately failing (e.g. partway through
/// a multi-argument unification); callers driving a search loop must
/// checkpoint MetaContext (and ElabState) before calling exact() and
/// rewind on failure, exactly the pattern documented on
/// MetaContext::solve_trail -- exact() itself does not do this
/// automatically, since a caller that already knows the term will
/// succeed (no search involved) shouldn't pay for a checkpoint/rewind
/// pair it will never use.
inline ElabResult exact(ProofState& ps, NodeId term, std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) {
        return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt,
            target_meta_id ? "exact: no goal with the given meta_id (already solved or superseded)"
                            : "exact: no open goals"});
    }
    Goal& g = ps.goals[*idx_opt];
    if (ElabResult r = check(term, g.target, g.ec, ps.es); !r) return r;
    if (!ctx().meta_ctx.is_solved(g.meta_id)) {
        // check() may already have solved g.meta_id transitively (e.g. if
        // term was itself exactly `mk_meta(g.meta_id)` after some prior
        // unification) -- solving again would be harmless (a meta is
        // solved at most once per checkpoint/rewind bracket in practice),
        // but skip it to keep MetaContext::solve_trail's accounting exact.
        // term may reference g's own ambient hypotheses (e.g. `mk_var(0)`
        // for an earlier Intro'd name), so it must be closed the same way
        // intro()'s own solution is -- see close_over_context's doc
        // comment for why this is required, not optional.
        ctx().meta_ctx.solve(g.meta_id, close_over_context(term, g.ec));
    }
    ps.goals.erase(ps.goals.begin() + *idx_opt);
    return {};
}
#endif


// ---------------------------------------------------------------------------
// Apply: Tier 1 (discrimination tree) + Tier 2 (real unification)
// ---------------------------------------------------------------------------

namespace detail {

enum class HeadKind : uint32_t { GlobalConst = 0, LocalVar = 1 };

#if 0 // not yet ported to the hybrid variable representation: pack_head_key
inline uint64_t pack_head_key(HeadKind kind, uint32_t id) {
    return (static_cast<uint64_t>(kind) << 32) | static_cast<uint64_t>(id);
}
#endif


#if 0 // not yet ported to the hybrid variable representation: conclusion_head_key
/// Recovers the head symbol of a (possibly Pi-quantified) conclusion
/// type, by structurally unwinding every Pi layer (no evaluation --
/// these are peeled syntactically, since only the SHAPE matters here)
/// and then the resulting conclusion's own application spine. Returns
/// nullopt if the conclusion isn't headed by a Const or a Var at all
/// (e.g. a bare Type/Meta conclusion), which simply means that
/// candidate is never proposed by Tier 1 -- acceptable, since a
/// conclusion with no discernible head is a poor Apply candidate
/// regardless.
///
/// A GlobalConst-headed key is the common, precise case (Eq, Nat, ...).
/// A LocalVar-headed key is meaningful only when both sides share the
/// same ambient numbering, which holds between a goal and a HYPOTHESIS
/// within the same ProofState, but not in general for a global lemma's
/// own fresh-scope-relative bound variable (e.g. `id : (A:Type) -> A ->
/// A`'s conclusion head is "A", a level meaningless outside id's own
/// declaration) -- a known, accepted imprecision: Tier 1 is a
/// menu-narrowing heuristic, never the soundness authority (Tier 2's
/// real unify() always re-checks whatever Tier 1 proposes).
///
/// OUTER HEAD ONLY, deliberately -- NOT the more discriminating key
/// library_index_key (below) uses. This is used by apply_candidates'
/// own hypothesis-matching loop, which specifically needs the LOOSER
/// "same head symbol" semantics its own doc comment already promises:
/// a hypothesis h : Eq Nat n m must still be proposed against a goal of
/// Eq Nat (Succ n) (Succ m) -- same outer head, structurally different
/// arguments -- confirmed load-bearing by test_apply_bothtiers, which
/// broke when a stricter, argument-profile-aware key was first tried
/// here instead of being confined to library_index_key alone. In-scope
/// hypotheses never grow with library size (a handful per goal, not
/// thousands), so the bucket-growth problem library_index_key exists to
/// fix never applies to this comparison in the first place.
inline std::optional<uint64_t> conclusion_head_key(NodeId ty) {
    NodeId cur = ty;
    while (true) {
        auto next = inspect(cur, [](Pi p) -> std::optional<NodeId> { return p.cod_id; },
                             [](auto) -> std::optional<NodeId> { return std::nullopt; });
        if (!next) break;
        cur = *next;
    }
    NodeId head = cur;
    bool progressed = true;
    while (progressed) {
        progressed = false;
        inspect(head, [&](App a) { head = a.fn_id; progressed = true; }, [](auto) {});
    }
    return inspect(head,
        [](Const c) -> std::optional<uint64_t> { return pack_head_key(HeadKind::GlobalConst, c.global_id); },
        [](Var v) -> std::optional<uint64_t> { return pack_head_key(HeadKind::LocalVar, v.level); },
        [](auto) -> std::optional<uint64_t> { return std::nullopt; });
}
#endif


#if 0 // not yet ported to the hybrid variable representation: library_index_key
/// The DiscriminationTree's OWN indexing key -- conclusion_head_key
/// EXTENDED with a shallow, full-argument profile: the outer head
/// combined with the head of EVERY top-level argument to the outermost
/// application (one level deep into each, not recursively deeper), e.g.
/// for "Eq Nat (add n Zero) n" the key reflects Eq, Nat, add, AND n's
/// own kind, not just Eq alone. Used ONLY by DiscriminationTree::
/// insert()/candidates_for() -- NOT by apply_candidates' own hypothesis
/// loop, which needs conclusion_head_key's looser semantics instead
/// (see its own comment above; confirmed by test failure, not assumed,
/// that using this stricter key there breaks it).
///
/// This exists because conclusion_head_key alone was confirmed, by
/// direct benchmark, to degrade linearly for exactly the dominant shape
/// a Nat/arithmetic-heavy curriculum produces: every Eq-concluding
/// lemma -- most of them -- shared one bucket regardless of what was
/// actually being equated (500 Eq lemmas -> 501 Tier 1 candidates
/// returned per call, 4000 -> 4001, latency climbing from 219ns to
/// 1096ns purely from copying an ever-larger bucket vector). A first
/// attempt indexing only the outer head plus the FIRST argument (giving
/// e.g. just "Eq Nat") was re-benchmarked, not assumed sufficient, and
/// found still insufficient: it still collapsed every "Eq Nat ..."
/// lemma into one bucket, since the type alone doesn't discriminate
/// between arithmetic lemmas about different operations -- the
/// operation itself does (add vs mul vs sub, each giving a different
/// argument head). Indexing the full shallow profile fixed this,
/// confirmed by a third benchmark with genuinely varied lemma shapes:
/// candidates returned grew roughly proportionally to the number of
/// distinct operation/argument profiles actually present, not to total
/// library size, and per-call latency stayed flat (400-560ns) across an
/// 8x growth in declared lemmas.
///
/// Combined via hash-combine, not packed into pack_head_key's own 64
/// bits (no spare room there: 32 for kind, 32 fully used for id) --
/// acceptable because this key is never a soundness authority, only a
/// candidate-menu heuristic Tier 2's real unify() always re-checks
/// afterward, so a rare collision between two distinct argument-
/// profiles costs one spurious suggestion, not a correctness risk.
inline std::optional<uint64_t> library_index_key(NodeId ty) {
    NodeId cur = ty;
    while (true) {
        auto next = inspect(cur, [](Pi p) -> std::optional<NodeId> { return p.cod_id; },
                             [](auto) -> std::optional<NodeId> { return std::nullopt; });
        if (!next) break;
        cur = *next;
    }
    NodeId head = cur;
    std::vector<NodeId> args; // filled in REVERSE application order (last App peeled first)
    bool progressed = true;
    while (progressed) {
        progressed = false;
        inspect(head, [&](App a) { head = a.fn_id; args.push_back(a.arg_id); progressed = true; }, [](auto) {});
    }
    auto shallow_head_key = [](NodeId t) -> std::optional<uint64_t> {
        NodeId h = t;
        bool p = true;
        while (p) {
            p = false;
            inspect(h, [&](App a) { h = a.fn_id; p = true; }, [](auto) {});
        }
        return inspect(h,
            [](Const c) -> std::optional<uint64_t> { return pack_head_key(HeadKind::GlobalConst, c.global_id); },
            [](Var v) -> std::optional<uint64_t> { return pack_head_key(HeadKind::LocalVar, v.level); },
            [](auto) -> std::optional<uint64_t> { return std::nullopt; });
    };
    auto outer = shallow_head_key(head);
    if (!outer) return std::nullopt;
    uint64_t combined = *outer;
    // args is in reverse application order; iterate it as-is -- a fixed,
    // consistent traversal order is all that's needed for two equal
    // conclusions to combine to the same key, regardless of which order
    // that is.
    for (NodeId a : args) {
        auto k = shallow_head_key(a);
        uint64_t part = k.value_or(0xdeadbeefULL); // an un-headed argument (Meta, Type, ...) still contributes a fixed, distinct part rather than being skipped -- skipping would make "Eq Nat (f x) y" and "Eq Nat (f x)" (different arities) collide
        combined ^= part + 0x9e3779b97f4a7c15ULL + (combined << 6) + (combined >> 2); // boost-style hash-combine
    }
    return combined;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: library_index_key_is_generic
/// Whether `ty`'s own conclusion has at least one top-level argument
/// (to the outermost application) whose own shallow head is a local
/// variable, OR has a bare local variable as its own outer head --
/// library_index_key hashes such a position to a LocalVar-kind key
/// specific to ty's own declaration scope, which can never equal any
/// goal's own key at that position (a goal's own arguments are
/// concrete, by construction of ANY actual proof state). Determines
/// which of DiscriminationTree's own two buckets insert() uses; see
/// its own comment for why a second bucket exists at all.
inline bool library_index_key_is_generic(NodeId ty) {
    NodeId cur = ty;
    while (true) {
        auto next = inspect(cur, [](Pi p) -> std::optional<NodeId> { return p.cod_id; },
                             [](auto) -> std::optional<NodeId> { return std::nullopt; });
        if (!next) break;
        cur = *next;
    }
    NodeId head = cur;
    std::vector<NodeId> args;
    bool progressed = true;
    while (progressed) {
        progressed = false;
        inspect(head, [&](App a) { head = a.fn_id; args.push_back(a.arg_id); progressed = true; }, [](auto) {});
    }
    auto is_var_headed = [](NodeId t) -> bool {
        NodeId h = t;
        bool p = true;
        while (p) {
            p = false;
            inspect(h, [&](App a) { h = a.fn_id; p = true; }, [](auto) {});
        }
        bool result = false;
        inspect(h, [&](Var) { result = true; }, [](auto) {});
        return result;
    };
    if (is_var_headed(head)) return true;
    for (NodeId a : args) if (is_var_headed(a)) return true;
    return false;
}
#endif


} // namespace detail

#if 0 // not yet ported: DiscriminationTree (apply machinery, S3)
struct DiscriminationTree {
    std::unordered_map<uint64_t, std::vector<NodeId>> by_head;
    // Entries whose own conclusion has a bound-variable argument at the
    // top level (e.g. refl's "Eq A x x") -- library_index_key can never
    // match these against any concrete goal (see
    // library_index_key_is_generic's own comment), so they are kept
    // here instead, indexed by outer head only. Expected to stay small:
    // confirmed this only catches genuinely structural lemmas (refl and
    // the like), not the bulk of an ordinary library (see the comment
    // on insert()).
    std::unordered_map<uint64_t, std::vector<NodeId>> generic_by_outer_head;

    /// Indexes `candidate_term` (a Const reference for a library lemma,
    /// or a Var for a hypothesis) under its OWN type's library index key
    /// -- library_index_key, not the looser conclusion_head_key: this is
    /// the one place the measured bucket-growth problem lives (a
    /// library's own lemma count, unlike a goal's handful of
    /// hypotheses, genuinely grows with training), so it needs the more
    /// discriminating key. An entry whose conclusion has a bound-
    /// variable top-level argument is indexed into generic_by_outer_head
    /// instead (by outer head alone), since library_index_key's own,
    /// finer key could never match a concrete goal for such an entry --
    /// confirmed directly: refl, indexed only under library_index_key,
    /// was never once returned by candidates_for(), for any goal.
    void insert(NodeId candidate_term, NodeId candidate_type) {
        if (detail::library_index_key_is_generic(candidate_type)) {
            if (auto key = detail::conclusion_head_key(candidate_type)) {
                generic_by_outer_head[*key].push_back(candidate_term);
            }
            return;
        }
        if (auto key = detail::library_index_key(candidate_type)) {
            by_head[*key].push_back(candidate_term);
        }
    }

    std::vector<NodeId> candidates_for(NodeId goal_type) const {
        std::vector<NodeId> result;
        if (auto key = detail::library_index_key(goal_type)) {
            auto it = by_head.find(*key);
            if (it != by_head.end()) result.insert(result.end(), it->second.begin(), it->second.end());
        }
        if (auto key = detail::conclusion_head_key(goal_type)) {
            auto it = generic_by_outer_head.find(*key);
            if (it != generic_by_outer_head.end()) result.insert(result.end(), it->second.begin(), it->second.end());
        }
        return result;
    }
};
#endif

#if 0 // not yet ported to the hybrid variable representation: index_library
/// Seeds a DiscriminationTree from every currently-declared global --
/// meant to be built ONCE (e.g. once per training process, not once per
/// proof attempt) and reused across many ProofStates, matching the
/// roadmap's own concern about repeated O(all globals) scans under
/// self-play-scale search. NOT called automatically by start_proof for
/// exactly that reason -- callers decide when the cost is worth paying.
inline DiscriminationTree index_library() {
    DiscriminationTree tree;
    for (uint32_t gid = 0; gid < ctx().globals.defs.size(); ++gid) {
        tree.insert(mk_const(gid), ctx().globals.defs[gid].type);
    }
    return tree;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: apply_candidates
/// Returns the Tier 1 candidate menu for the TARGETED goal (the goal
/// whose meta_id matches target_meta_id, or ps.goals.front() if
/// omitted) against `library` (globals) merged with every hypothesis
/// currently in scope -- head-symbol matches only, no unification
/// attempted. This is what a policy network / search driver consumes
/// to pick a candidate, and, for a hypergraph-style search exploring
/// several pending goals at once, needs to be queryable for any of
/// them, not only the sequentially-first -- see find_goal_index's own
/// doc comment for why targeting by meta_id specifically, not raw
/// vector position.
inline std::vector<NodeId> apply_candidates(const ProofState& ps, const DiscriminationTree& library,
                                             std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return {};
    const Goal& g = ps.goals[*idx_opt];
    std::vector<NodeId> result = library.candidates_for(g.target);
    for (uint32_t lvl = 0; lvl < g.hyps.size(); ++lvl) {
        auto key = detail::conclusion_head_key(g.target);
        if (key && detail::conclusion_head_key(g.hyps[lvl].type) == key) {
            result.push_back(mk_var(lvl));
        }
    }
    return result;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: rewrite_eq
/// Commits to ONE candidate: infers its type, peels off however many Pi
/// arguments it has -- minting, for EACH one, a "reference" metavariable
/// whose OWN type is Pi-wrapped over the goal's full hypothesis context
/// (see close_pi_over_context) so it is usable as a Miller pattern --
/// then unifies the fully-applied candidate's conclusion against the
/// goal. Real unification (Tier 2) is the only place this pays unify()'s
/// cost; Tier 1 never calls it.
///
/// Any reference metavariable unify() does NOT solve becomes a genuine
/// new open goal: a freshly-minted, UNWRAPPED metavariable is created
/// for it (matching every other Goal's own convention -- its type is
/// the argument's own type directly, not Pi-wrapped) and the reference
/// metavariable is solved with a closed-over pointer to it -- exactly
/// the same "solve outgoing with a reference to a fresh incoming meta"
/// pattern intro() already uses, applied here once per unresolved
/// argument instead of once per Pi.
/// Rewrites the current goal using eq_term : Eq A x y (A, x, y inferred directly from eq_term's
/// own type, not supplied separately -- eq_term is typically a library lemma applied to specific
/// arguments, e.g. mul_succ_right(n, m), or an ordinary hypothesis reference), replacing every
/// SYNTACTIC occurrence of x within g.target with y, producing exactly one new goal (the rewritten
/// target) -- mirroring intro()'s own "solve the outgoing meta with a reference to one fresh,
/// still-unsolved incoming meta" pattern exactly, just with the incoming meta's own type being the
/// REWRITTEN target rather than a Pi's codomain.
///
/// Deliberately reuses two already-declared, already-verified library lemmas rather than
/// re-deriving their logic inline as raw mk_match: transport_gid for the general Eq.subst
/// principle (A:Type)(P:A->Type)(x y:A)->Eq A x y->P x->P y, and symm_gid for Eq's own symmetry --
/// both hand-built in curriculum.py (setup_transport/CURRICULUM respectively) the exact same way
/// cong/cong_add_zero/cong_add_left/cong_add_right already are. Passing their global ids in from
/// the caller (who already knows them, having declared them) is simpler and lower-risk than
/// re-deriving the combined construction as a single new motive from scratch here.
///
/// Uses detail::generalize (the SAME syntactic, exact-NodeId-equality function cases()/induct()
/// already use internally for their own concrete indices and scrutinee) to find and abstract x --
/// deliberately NOT a defeq/conv-based search, by this codebase's own established design (see
/// generalize's own doc comment): if x does not appear in g.target in EXACTLY the syntactic form
/// eq_term's own type states it, rewrite simply does not change anything there (a caller detects
/// this the same way check(Match) does for an analogous case: by the resulting proof failing to
/// type-check, not by this function inspecting its own output). Bringing a goal into a
/// syntactically-matching shape first (e.g. via an explicit one-step unfolding lemma like
/// mul_succ_right itself) is the caller's own responsibility, exactly mirroring how real proof
/// assistants (Coq, Lean) also require `rewrite` to find a literal, syntactic occurrence, relying
/// on a separate `simpl`/`unfold`/`change` step beforehand when one is not already present.
inline ElabResult rewrite_eq(ProofState& ps, NodeId eq_term, NodeId x, NodeId y, uint32_t transport_gid, uint32_t symm_gid,
                              std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) {
        return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt,
            target_meta_id ? "rewrite_eq: no goal with the given meta_id (already solved or superseded)"
                            : "rewrite_eq: no open goals"});
    }
    Goal g = ps.goals[*idx_opt]; // copy: only committed to ps.goals on success

    // x AND y are BOTH supplied directly by the caller, deliberately NOT re-derived from
    // infer(eq_term)'s own computed type: infer() internally evals/quotes to substitute eq_term's
    // own arguments into its head's declared Pi-chain, and eval() -- by this codebase's own
    // established, hard-won design constraint (see intro()'s own history) -- ALWAYS eagerly
    // unfolds every global Fix definition it can reach. Confirmed directly, twice (once for each
    // side): for eq_term = mul_succ_zero_unfold(n'), infer()'s own computed type's "x" AND "y" both
    // came back as fully unfolded "match n' with ..." reconstructions, not the flat terms the
    // caller actually wrote ("mul (Succ n') Zero" and "add (mul n' Zero) Zero" respectively) and
    // that genuinely, syntactically appear in g.target / are needed in the new goal -- the first
    // attempt here fixed only x (generalize() then correctly found and abstracted the occurrence)
    // but still produced a goal full of this same eager reconstruction, since substitute_var was
    // substituting the EAGERLY-UNFOLDED y, not the flat one, into the newly-abstracted position.
    // Only infer()'s own "A" survives being used as-is below, since a TYPE rarely suffers this same
    // eager-unfolding-changes-identity issue the way a computed Nat-valued expression does.
    TypeResult eq_ty_r = infer(eq_term, g.ec, ps.es);
    if (!eq_ty_r) return std::unexpected(eq_ty_r.error());
    Value eq_ty_val = eval(*eq_ty_r, g.ec.values);
    auto* eq_neutral = std::get_if<VNeutral>(&eq_ty_val);
    if (!eq_neutral || eq_neutral->head.kind != NeutralHead::Kind::GlobalConst || eq_neutral->spine.size() != 3) {
        return std::unexpected(TypeError{TypeErrorKind::Mismatch, *eq_ty_r, std::nullopt,
            "rewrite_eq: eq_term's own type is not a 3-argument (A, x, y) relation application -- expected Eq A x y"});
    }
    std::vector<NodeId> parts;
    for (auto& frame : eq_neutral->spine) {
        auto* af = std::get_if<ApplyFrame>(&frame);
        if (!af) {
            return std::unexpected(TypeError{TypeErrorKind::Mismatch, *eq_ty_r, std::nullopt,
                "rewrite_eq: eq_term's own type has a non-application eliminator frame"});
        }
        parts.push_back(quote(af->arg, g.ec.size()));
    }
    NodeId A = parts[0];
    // parts[1]/parts[2] (eq_term's own computed "x"/"y") are deliberately NOT used -- see above.

    NodeId motive_body = detail::generalize(g.target, {{x, g.ec.size()}});
    NodeId motive = mk_lambda(A, motive_body);
    NodeId new_target = detail::substitute_var(motive_body, g.ec.size(), y);

    uint32_t new_meta = ctx().meta_ctx.fresh(new_target, g.ec.size());
    NodeId new_meta_ref = mk_meta(new_meta);
    for (uint32_t lvl = 0; lvl < g.ec.size(); ++lvl) new_meta_ref = mk_app(new_meta_ref, mk_var(lvl));

    NodeId symm_term = mk_app(mk_app(mk_app(mk_app(mk_const(symm_gid), A), x), y), eq_term);
    NodeId wrapped = mk_app(mk_app(mk_app(mk_app(mk_app(mk_app(
        mk_const(transport_gid), A), motive), y), x), symm_term), new_meta_ref);

    ctx().meta_ctx.solve(g.meta_id, close_over_context(wrapped, g.ec));
    ps.goals[*idx_opt] = Goal{new_meta, g.ec, g.hyps, new_target};
    return {};
}
#endif


#if 0 // not yet ported to the hybrid variable representation: apply
inline ElabResult apply(ProofState& ps, NodeId candidate, std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) {
        return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt,
            target_meta_id ? "apply: no goal with the given meta_id (already solved or superseded)"
                            : "apply: no open goals"});
    }
    Goal g = ps.goals[*idx_opt]; // copy: only committed to ps.goals on success

    TypeResult cand_ty_r = infer(candidate, g.ec, ps.es);
    if (!cand_ty_r) return std::unexpected(cand_ty_r.error());

    NodeId applied = candidate;
    Value cur_ty_val = eval(*cand_ty_r, g.ec.values);
    std::vector<std::pair<uint32_t, NodeId>> arg_metas; // (reference meta id, its own unwrapped arg type)
    while (true) {
        auto* pi = std::get_if<VPi>(&cur_ty_val);
        if (!pi) break;
        NodeId arg_ty = quote(eval(pi->dom_id, pi->env), g.ec.size());
        uint32_t m = ctx().meta_ctx.fresh(close_pi_over_context(arg_ty, g.ec), g.ec.size());
        arg_metas.push_back({m, arg_ty});
        NodeId meta_ref = mk_meta(m);
        for (uint32_t i = 0; i < g.ec.size(); ++i) meta_ref = mk_app(meta_ref, mk_var(i));
        applied = mk_app(applied, meta_ref);
        Value arg_val = eval(meta_ref, g.ec.values);
        cur_ty_val = eval(pi->cod_id, pi->env.extend(arg_val));
    }
    NodeId concl_ty = quote(cur_ty_val, g.ec.size());

    if (ElabResult r = unify(concl_ty, g.target, g.ec, ps.es); !r) return r;
    // unify() can defer a sub-constraint to es.pending rather than
    // actually checking it -- concretely, whenever eval() reduces one
    // side to a value headed by a metavariable, which happens not only
    // for a bare, unresolved meta but also for a STUCK recursive-
    // function application whose own discriminee is that meta (e.g.
    // "add META Zero", stuck because add's own match can't proceed
    // without knowing META). unify_values' own "is_meta_headed" check
    // treats both cases identically, deferring rather than decomposing
    // structurally -- so a genuinely mismatched argument buried inside
    // such a stuck application (here, "Zero" against the goal's own
    // "Succ m") can slip through unify() entirely if some OTHER
    // argument position happens to solve the same meta via ordinary
    // pattern unification, since apply()'s own is_solved(m) check below
    // would then see it as solved and never generate a subgoal for it.
    // drain_pending() exists specifically to re-attempt every deferred
    // constraint once other metas may have been solved -- confirmed
    // directly, by finding this exact failure mode via a real
    // curriculum goal, that it was defined but never actually called
    // anywhere in this file. Calling it here, immediately, is far
    // cheaper than the alternative this bug was otherwise relying on
    // (an unrelated, independent whole_term_check catching it much
    // later, after the search had already spent effort treating this
    // node as solved).
    if (ElabResult r = drain_pending(g.ec, ps.es); !r) return r;

    std::vector<Goal> new_goals;
    for (auto& [m, arg_ty] : arg_metas) {
        if (ctx().meta_ctx.is_solved(m)) continue;
        uint32_t subgoal_meta = ctx().meta_ctx.fresh(arg_ty, g.ec.size());
        NodeId subgoal_ref = mk_meta(subgoal_meta);
        for (uint32_t i = 0; i < g.ec.size(); ++i) subgoal_ref = mk_app(subgoal_ref, mk_var(i));
        ctx().meta_ctx.solve(m, close_over_context(subgoal_ref, g.ec));
        new_goals.push_back(Goal{subgoal_meta, g.ec, g.hyps, arg_ty});
    }

    ctx().meta_ctx.solve(g.meta_id, close_over_context(applied, g.ec));

    ps.goals.erase(ps.goals.begin() + *idx_opt);
    ps.goals.insert(ps.goals.begin() + *idx_opt, new_goals.begin(), new_goals.end());
    return {};
}
#endif


// ---------------------------------------------------------------------------
// Cases: the cheap half of the Destruct split
// ---------------------------------------------------------------------------
//
// Assembles EXISTING Phase 1 machinery (detail::generalize,
// instantiate_type_telescope, instantiate_ctor_telescope,
// ctor_gids_for_type -- all reused directly, not rebuilt) into tactic
// shape: builds the motive the same way check(Match)'s auto-construct
// path does, then for EACH constructor mints a reference metavariable
// (Pi-wrapped over the goal's context, per apply()'s own "usable as a
// Miller pattern" discipline) plus a genuine new subgoal -- following
// the exact same "solve outgoing with a closed-over reference to a
// fresh incoming meta" pattern intro()/apply() already use.
//
// Unlike apply()'s subgoals (which stay Pi-typed, expecting the caller
// to intro each argument manually), Cases auto-intros each
// constructor's own fields as named hypotheses directly into the new
// goal's telescope -- matching Lean's own `cases`/`induction`
// convention, and meaning the new subgoal's meta has a LARGER
// scope_depth (g.ec.size() + that constructor's own arity) than the
// reference meta that points to it (g.ec.size() alone).
/// Step 3 (design spec, section on refutation exploitation) forward
/// declarations -- defined after reserve_inductive_type/finish_inductive
/// (which they depend on), below; cases() (this function) is defined
/// earlier in the file than those, so needs these declared ahead of use.
/// See their own doc comments at the definition site for the full
/// rationale.
inline std::pair<uint32_t, uint32_t> get_or_build_unit_type();
inline uint32_t get_or_build_empty_type();
inline uint32_t get_or_build_same_ctor(uint32_t family_gid);

#if 0 // not yet ported to the hybrid variable representation: cases
inline ElabResult cases(ProofState& ps, uint32_t hyp_level, std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) {
        return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt,
            target_meta_id ? "cases: no goal with the given meta_id (already solved or superseded)"
                            : "cases: no open goals"});
    }
    Goal g = ps.goals[*idx_opt]; // copy: only committed to ps.goals on success
    if (hyp_level >= g.hyps.size()) {
        return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt,
            "cases: hypothesis index out of range"});
    }
    if (detail::is_pi_headed(g.target)) {
        // Same guard, same reasoning, as induct()'s own (see that function's own comment for the
        // full derivation): cases() generalizes g.target the identical way, so the identical
        // collision risk applies whenever the goal still has a remaining Pi.
        return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt,
            "cases: goal's own target is still a Pi -- intro every quantified variable before "
            "casing (cases()'s own generalize() step can otherwise collide with a "
            "not-yet-introduced variable's own body-level reference)"});
    }
    NodeId scrut_ty = g.hyps[hyp_level].type;
    Value scrut_ty_val = eval(scrut_ty, g.ec.values);
    auto* scrut_neutral = std::get_if<VNeutral>(&scrut_ty_val);
    if (!scrut_neutral || scrut_neutral->head.kind != NeutralHead::Kind::GlobalConst) {
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, scrut_ty, std::nullopt,
            "cases: hypothesis's type is not an inductive type application"});
    }
    uint32_t type_global_id = scrut_neutral->head.id;
    std::vector<NodeId> concrete_indices;
    for (auto& frame : scrut_neutral->spine) {
        auto* af = std::get_if<ApplyFrame>(&frame);
        if (!af) {
            return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, scrut_ty, std::nullopt,
                "cases: scrutinee's type has a non-application eliminator frame; cannot enumerate its indices"});
        }
        concrete_indices.push_back(quote(af->arg, g.ec.size()));
    }
    uint32_t arity = static_cast<uint32_t>(concrete_indices.size());

    // Build the motive exactly as check(Match)'s auto-construct path
    // does: generalize the CURRENT goal's target over the concrete
    // indices and the hypothesis itself.
    std::vector<std::pair<NodeId, uint32_t>> targets;
    for (uint32_t i = 0; i < arity; ++i) targets.push_back({concrete_indices[i], g.ec.size() + i});
    targets.push_back({mk_var(hyp_level), g.ec.size() + arity});
    // Every hypothesis AFTER hyp_level must be REVERTED (pulled out of
    // the context and back into the target as extra Pi layers) before
    // generalizing -- the standard technique real proof assistants use
    // internally for this exact problem (Coq/Lean's own "revert" then
    // re-intro). Without this, cases() could only ever generalize the
    // bare target, never a sibling hypothesis that also references the
    // same variable being case-split on -- confirmed directly, this
    // session, to make ANY multi-hypothesis dependency (e.g.
    // transitivity of equality: h1:Eq A x y, h2:Eq A y z, both
    // genuinely needing y in scope at once) permanently unprovable,
    // regardless of proof ordering, since the two hypotheses inherently
    // must coexist.
    //
    // Deliberately UNCONDITIONAL for every hyp in the reverted range
    // (not just ones whose own type actually depends on something) --
    // NOT an over-cautious simplification: this kernel's hypotheses
    // have their own absolute level implicit in their own position in
    // g.hyps (hyp at list-index i occupies exactly level i), so
    // selectively reverting only SOME hypotheses in a range while
    // leaving others in place would leave a gap in that correspondence
    // for whatever was left behind -- its own stored type would still
    // reference its old, now-wrong absolute level. Reverting the whole
    // contiguous block and re-introducing it in the same relative
    // order preserves every hypothesis's own relative structure intact;
    // only the absolute level each now occupies shifts (by ctor_arity,
    // the constructor's own new fields, inserted ahead of them), which
    // the eval/quote peeling below handles correctly and directly, the
    // same way shift_levels_from is used elsewhere in this same
    // function for an analogous shift.
    //
    // The block to revert doesn't always start right after hyp_level:
    // an EARLIER hypothesis can also need reverting, if its own type
    // mentions one of the concrete indices being generalized (not
    // hyp_level's own level, which an earlier hypothesis could never
    // reference in the first place, since it didn't exist yet at that
    // point) -- e.g. transitivity's own z:A, introduced before h1:Eq A
    // x y yet whose own type ("A") names the very parameter the motive
    // is about to generalize into a fresh, generic A''. Left alone, z
    // would keep referencing the concrete, ungeneralized A while
    // appearing inside the motive's own generic body -- a genuine
    // structural mismatch, confirmed directly by infer() rejecting the
    // motive with "different variable/constant heads" before this
    // check was added. revert_from is the EARLIEST such hypothesis, if
    // any; everything from there to the end (excluding hyp_level
    // itself, consumed by the match rather than reverted) is swept
    // into the same contiguous, unconditional block.
    uint32_t revert_from = hyp_level + 1;
    for (uint32_t j = 0; j < hyp_level; ++j) {
        if (detail::generalize(g.hyps[j].type, targets) != g.hyps[j].type) {
            revert_from = j;
            break;
        }
    }
    std::vector<uint32_t> reverted_js; // increasing order (original hyp indices)
    for (uint32_t j = revert_from; j < g.hyps.size(); ++j) {
        if (j == hyp_level) continue; // consumed by the match itself, never reverted
        reverted_js.push_back(j);
    }
    NodeId reverted_target = g.target;
    for (auto it = reverted_js.rbegin(); it != reverted_js.rend(); ++it) {
        reverted_target = mk_pi(g.hyps[*it].type, reverted_target);
    }
    // Reverting a hypothesis only wraps its own TYPE as a fresh Pi-
    // domain; it does nothing, by itself, to any EXISTING occurrence of
    // that hypothesis's own level elsewhere (as a value, not just via
    // its type) -- g.target or a later reverted hyp's own type may
    // reference it directly (transitivity's own z, referenced by both
    // h2's type "Eq A y z" and g.target "Eq A x z"). Left unmapped,
    // those occurrences would keep pointing at the OLD, outer level
    // even after being embedded under the new Pi-chain's own binders --
    // a genuine structural mismatch, confirmed directly by infer()
    // rejecting the motive with "different variable/constant heads"
    // before this was added. Adding each reverted hypothesis's own
    // level to the generalization targets, mapped to the fresh level
    // its own new Pi-binder will occupy (continuing right after the
    // motive's own index binders and scrutinee), redirects every such
    // occurrence consistently -- exactly how hyp_level's own level is
    // already mapped, just extended to every OTHER reverted hypothesis
    // too.
    std::vector<std::pair<NodeId, uint32_t>> extended_targets = targets;
    for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
        extended_targets.push_back({mk_var(reverted_js[ri]), g.ec.size() + arity + 1 + static_cast<uint32_t>(ri)});
    }
    // reverted_target's own, deepest portion (g.target itself, before
    // any reverting-wrapping above) may still contain its OWN, further
    // Pi-quantified variables that the caller has not yet introduced
    // (e.g. cases()/induct() invoked before every variable the goal
    // mentions has been intro'd) -- such a variable's own bound
    // reference, within g.target's own body, already, correctly uses
    // g.ec.size() as its level, by the same "a Pi's own bound variable
    // references exactly ec.size()" convention this whole file follows
    // (see intro()'s own comment). But generalize() below is about to
    // insert `arity + 1 + reverted_js.size()` BRAND NEW binders (the
    // scrutinee's own indices, the scrutinee itself, and every reverted
    // hypothesis) starting at that EXACT SAME level -- without first
    // moving g.target's own, pre-existing use of it out of the way, the
    // two collide: an un-introduced variable's own reference and the
    // freshly-generalized discriminee's own reference become
    // indistinguishable, silently merging two genuinely different
    // things into one. Confirmed directly, this session: casing on a
    // Nat hypothesis before introducing a further Nat the goal still
    // quantified over produced a motive where BOTH ended up referencing
    // the same bound variable, extracting a term that used the wrong
    // value in the wrong place and failed whole_term_check with
    // "different variable/constant heads" despite the tactic layer
    // itself reporting the goal solved.
    uint32_t new_binder_count = arity + 1 + static_cast<uint32_t>(reverted_js.size());
    NodeId shifted_reverted_target = detail::shift_levels_from(reverted_target, g.ec.size(), new_binder_count);
    NodeId generalized_goal = detail::generalize(shifted_reverted_target, extended_targets);

    std::vector<NodeId> index_types = detail::instantiate_type_telescope(type_global_id, g.ec.size());
    if (index_types.size() != arity) {
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,
            "cases: index count mismatch between the hypothesis's type application and the "
            "inductive's own declared arity"});
    }
    std::vector<NodeId> idx_vars;
    for (uint32_t i = 0; i < arity; ++i) idx_vars.push_back(mk_var(g.ec.size() + i));
    NodeId scrut_binder_ty = apply_all(mk_const(type_global_id), idx_vars);
    NodeId motive_original = mk_lambda(scrut_binder_ty, generalized_goal);
    for (auto it = index_types.rbegin(); it != index_types.rend(); ++it) motive_original = mk_lambda(*it, motive_original);

    // Type-check the constructed motive -- this IS the "cannot be
    // abstracted cleanly" detection mechanism, same as check(Match)'s
    // own auto-construct path.
    TypeResult motive_ty_r = infer(motive_original, g.ec, ps.es);
    if (!motive_ty_r) {
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,
            "cases: constructed motive failed to type-check -- the goal likely depends on the "
            "hypothesis in a way that cannot be abstracted cleanly"});
    }

    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(type_global_id);

    // Pass 1: index forcing for every constructor (Step 2), and Step 3's
    // own detection pass -- whether any constructor is refuted AND safe
    // to exploit (the refutation must be a direct, top-level constructor-
    // head mismatch on an ENTIRE index slot, not one surfaced through the
    // repeated-field path -- see CtorIndexForcing's own doc comment for
    // exactly why that distinction matters here).
    std::vector<detail::InstantiatedCtor> insts;
    std::vector<CtorIndexForcing> forcings;
    for (uint32_t ctor_gid : ctor_gids) {
        detail::InstantiatedCtor inst = detail::instantiate_ctor_telescope(ctor_gid, g.ec.size());
        uint32_t ctor_arity = static_cast<uint32_t>(inst.arg_types.size());
        CtorIndexForcing forcing = unify_ctor_pattern_indices(inst, concrete_indices, g.ec.size(), ctor_arity, g.ec.values);
        if (!forcing.ok) {
            return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,
                forcing.error});
        }
        insts.push_back(std::move(inst));
        forcings.push_back(std::move(forcing));
    }
    std::optional<std::size_t> exploit_ci;
    for (std::size_t ci = 0; ci < forcings.size(); ++ci) {
        const CtorIndexForcing& f = forcings[ci];
        if (f.refuted && f.refuted_slot && f.refuted_pat_value && f.refuted_conc_value) {
            exploit_ci = ci; // first safely-exploitable refutation found; see the loop below for why only one is handled per call
            break;
        }
    }
    if (!reverted_js.empty()) {
        // Step 3's own synthesized branch value (for the exploited,
        // refuted constructor) is only ever wrapped in arg_types, never
        // in the reverted hyps' own types -- but motive_final's shape is
        // uniform across every branch, so it WOULD expect the reverted
        // hyps there too once any hypothesis was reverted. Rather than
        // adapt Step 3 to handle a combination it was never designed or
        // tested for (and that never actually arises for Eq specifically
        // -- see get_or_build_same_ctor's own doc comment on why this
        // exploitation is already restricted away from Eq), decline the
        // exploitation here and fall back to the ordinary, still fully
        // correct per-branch subgoal path for every constructor.
        exploit_ci.reset();
    }

    // Step 3: if a constructor is safely refuted, extend the motive with
    // one extra Pi layer -- same_ctor(this slot's own index parameter,
    // the known concrete value) -> ORIGINAL GOAL -- built and verified
    // end to end (type-checks AND normalizes correctly at a concrete
    // instance) before being wired in here; see get_or_build_same_ctor's
    // own doc comment for the full mechanism. motive_original is kept
    // separately and used for every AGENT-FACING subgoal computation
    // below, so an ordinary (non-refuted) branch's own goal is entirely
    // unaffected by this -- the agent never sees the exploitation
    // machinery at all, only fewer goals than constructors when it fires.
    //
    // same_ctor_component's own two arguments MUST be idx_vars
    // [refuted_slot] (the motive's own, shared parameter), not a FIXED
    // value (even the correct one, refuted_pat_value/refuted_conc_value
    // -- tried and reverted, see below): the mechanism this depends on
    // (get_or_build_same_ctor's own doc comment covers the full
    // argument) is that the OUTER application, below, supplies a single
    // Unit-typed witness (tt) that is sound at the outer level -- where
    // idx_vars[refuted_slot] is instantiated with concrete_indices
    // [refuted_slot] itself, making this same_ctor(X,X), provably Unit,
    // regardless of family -- while the MOTIVE's own dependent
    // re-application, within the specific (refuted) branch chosen,
    // reduces this SAME textual parameter's type to Empty instead.
    // Hardcoding a fixed value here breaks that: the outer level would
    // then see the ALREADY-Empty type directly, which tt cannot
    // inhabit -- confirmed directly by a real crash (comparing Unit vs
    // Empty during whole-term check()), not assumed.
    //
    // KNOWN, OPEN GAP: for Eq specifically, idx_vars[refuted_slot]'s
    // own declared type is itself one of the motive's OWN, earlier,
    // generic index parameters ("A", Eq's own first index, for the
    // second and third positions -- x:A, y:A) rather than family_gid's
    // own, concrete type (e.g. Nat, recovered from the ACTUAL compared
    // values) -- so this standalone motive validation fails ("different
    // variable/constant heads") for Eq's own repeated-index refutation
    // specifically, regardless of which value is plugged in at this
    // call site: get_or_build_same_ctor(family_gid) is inherently built
    // with its own two parameters concretely typed at family_gid (see
    // its own implementation -- family_ty = mk_const(family_gid)
    // directly), which idx_vars[refuted_slot]:A can never match
    // generically, for any family_gid, since the motive itself must
    // stay well-typed for an ARBITRARY "A", not just this specific
    // scrutinee's own Nat instantiation. A genuine fix needs either a
    // type-generic same_ctor construction (same_ctor parameterized over
    // the type itself, not one fixed family_gid) or a fundamentally
    // different proof strategy for this shape entirely (native
    // unification-failure-driven absurdity, the way a dependent
    // pattern-match compiler ordinarily handles this, rather than
    // threading a witness through the motive at all) -- left for
    // dedicated, unhurried follow-up, not attempted here. This
    // exploitation therefore still only fires for an ORDINARY,
    // non-type-polymorphic single-constructor family with a repeated
    // index (its own index positions concretely typed at one, fixed
    // family, matching every ordinary indexed type this session has
    // exercised) -- not Eq itself.
    NodeId motive_final = motive_original;
    if (exploit_ci) {
        uint32_t refuted_slot = *forcings[*exploit_ci].refuted_slot;
        uint32_t family_gid = *forcings[*exploit_ci].refuted_family_gid;
        uint32_t same_ctor_gid = get_or_build_same_ctor(family_gid);
        NodeId same_ctor_component = apply_all(mk_const(same_ctor_gid), {idx_vars[refuted_slot], concrete_indices[refuted_slot]});
        NodeId extended_inner = mk_pi(same_ctor_component, generalized_goal);
        NodeId m = mk_lambda(scrut_binder_ty, extended_inner);
        for (auto it = index_types.rbegin(); it != index_types.rend(); ++it) m = mk_lambda(*it, m);
        TypeResult mfr = infer(m, g.ec, ps.es);
        if (mfr) {
            motive_final = m;
        } else {
            // Exploitation itself failed to type-check -- fall back to
            // the ORIGINAL, unextended behavior (refutation stays
            // detected, just not exploited) rather than fail the whole
            // tactic call: this should not happen given the validated
            // construction, but declining gracefully here follows the
            // exact same discipline as match_index_pattern's own
            // "insufficient information is not a reason to fail" stance.
            exploit_ci.reset();
        }
    }

    std::vector<NodeId> branches;
    std::vector<Goal> new_goals;
    for (std::size_t ci = 0; ci < ctor_gids.size(); ++ci) {
        uint32_t ctor_gid = ctor_gids[ci];
        const detail::InstantiatedCtor& inst = insts[ci];
        uint32_t ctor_arity = static_cast<uint32_t>(inst.arg_types.size());
        const CtorIndexForcing& forcing = forcings[ci];

        // Index forcing (design spec, Phase B/Step 2): this constructor's
        // own index pattern (inst.result_indices) unified against the
        // hypothesis's ACTUAL, concrete indices (concrete_indices) --
        // exactly mirroring check_match_branches' own treatment. Without
        // this, a constructor whose LATER field's type depends on an
        // EARLIER one (Vec's own cons: t : Vec n) would present the
        // agent with the raw, generic field type (Vec n) even when the
        // index is already known to be concrete (e.g. destructuring a
        // hypothesis of type Vec (Succ k) should tell the agent t has
        // type Vec k, not the unhelpfully generic Vec n). Empty for a
        // non-indexed type (Nat/Bool/List), in which case this is a
        // proven no-op -- identical behavior to every already-passing
        // test, not merely assumed identical.
        std::vector<NodeId> arg_types = apply_ctor_index_forcing(inst.arg_types, forcing);

        // branch_expected is ALWAYS computed via motive_original (the
        // agent-facing goal is never affected by Step 3's own motive
        // extension) -- shift_levels_from's own doc comment (see
        // check_match_branches, which this mirrors exactly) explains why
        // this shift is needed regardless: the motive is being embedded
        // underneath ctor_arity NEW binders (this constructor's own
        // arguments), which independently ALSO start at g.ec.size().
        NodeId shifted_motive_original = detail::shift_levels_from(motive_original, g.ec.size(), ctor_arity);
        NodeId ctor_app = mk_const(ctor_gid);
        for (uint32_t ai = 0; ai < ctor_arity; ++ai) ctor_app = mk_app(ctor_app, mk_var(g.ec.size() + ai));
        // Forced too, exactly like arg_types just above and for the
        // identical reason: inst.result_indices, unforced, would supply
        // the motive's own substitution with a constructor field's own,
        // generic identity (e.g. refl's own A_field) precisely where
        // the scrutinee's own, concrete argument (A) is already known
        // and, critically, already used elsewhere -- x_field's own
        // type (via arg_types' own forcing, above) genuinely,
        // correctly says "A", not "A_field". Leaving result_indices
        // unforced would substitute the motive's own generic "A''"
        // parameter with A_field instead -- a separate, syntactically
        // distinct NodeId from A, even though the constructor's own
        // pattern establishes they must be equal -- so anything
        // downstream comparing a value typed via one against a value
        // typed via the other (reverted_dom_types, below, inherits
        // whichever identity result_indices carries) would see two
        // unrelated variables and fail, exactly as apply()'s own
        // automatic arg_metas construction did here, confirmed
        // directly this session (a "different variable/constant
        // heads" mismatch surviving all the way to whole_term_check
        // for a symm-style proof, traced precisely to this asymmetry).
        std::vector<NodeId> forced_result_indices = apply_ctor_index_forcing_to_result_indices(inst.result_indices, forcing);
        NodeId branch_expected = apply_all(shifted_motive_original, forced_result_indices);
        branch_expected = mk_app(branch_expected, ctor_app);

        std::vector<NodeId> reverted_dom_types; // branch-specific (this constructor's own values already substituted in)
        if (!reverted_js.empty()) {
            Env be_env = g.ec.values;
            for (uint32_t ai = 0; ai < ctor_arity; ++ai) be_env = be_env.extend(fresh_neutral(g.ec.size() + ai));
            Value cur_val = eval(branch_expected, be_env);
            for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
                auto* pi = std::get_if<VPi>(&cur_val);
                if (!pi) {
                    return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,
                        "cases: internal error -- expected a Pi layer for a reverted hypothesis but found none"});
                }
                // target_level: where THIS reverted hypothesis will
                // actually be stored in branch_ec (the FULL, original
                // g.ec, then arg_types, then the reverted hyps in
                // order -- see branch_ec's own construction below,
                // which spans all of g.ec.size(), not just hyp_level+1)
                // -- known in advance, and deliberately used here
                // INSTEAD of pi->env.size() (the level the motive's
                // own, internal syntax happens to place this binder
                // at, which does not match branch_ec at all once any
                // earlier hypothesis also needed reverting -- confirmed
                // directly by a real crash, Env::operator[] out of
                // bounds, before this was corrected).
                // Env::extend is purely positional -- a value's own
                // intrinsic level label, which quote() reads back out,
                // is independent of where it physically sits in the
                // env -- so assigning the fresh neutral its REAL,
                // final level directly means every LATER reference to
                // it (e.g. a subsequent reverted hyp's own domain, like
                // h2's own type mentioning z) gets extracted already
                // expressed in branch_ec's own terms, with no separate
                // re-leveling pass needed afterward.
                uint32_t target_level = g.ec.size() + static_cast<uint32_t>(arg_types.size()) + static_cast<uint32_t>(ri);
                NodeId dom_ty = quote(eval(pi->dom_id, pi->env), target_level);
                reverted_dom_types.push_back(dom_ty);
                Value fresh = fresh_neutral(target_level);
                cur_val = eval(pi->cod_id, pi->env.extend(fresh));
            }
            uint32_t final_depth = g.ec.size() + static_cast<uint32_t>(arg_types.size()) + static_cast<uint32_t>(reverted_js.size());
            branch_expected = quote(cur_val, final_depth);
        } else {
            // F4: with nothing reverted, branch_expected used to be left as the raw motive application
            // `(motive idx...) ctor_app` -- an unreduced beta-redex whose embedded lambda is "embedded bound
            // structure", so a later induct() on a field of this branch was refused by its has_internal_binder_ref
            // guard (cases -> induct failed even though cases -> cases and induct -> induct were fine).
            // Reduce it. NoDeltaScope: beta/iota only; global constants stay folded, so the goal keeps its
            // flat shape (a full NbE round-trip would delta-unfold every definition it can reach -- see
            // intro()'s own long comment on why that is unacceptable for a stored goal).
            NoDeltaScope keep_constants_folded;
            Env be_env = g.ec.values;
            for (uint32_t ai = 0; ai < ctor_arity; ++ai) be_env = be_env.extend(fresh_neutral(g.ec.size() + ai));
            branch_expected = quote(eval(branch_expected, be_env), g.ec.size() + static_cast<uint32_t>(arg_types.size()));
        }

        if (exploit_ci && ci == *exploit_ci) {
            // This constructor is the one Step 3 exploits: synthesize
            // its branch DIRECTLY, no new subgoal at all -- the agent
            // never has to (and, without further kernel support, could
            // not) prove this branch itself.
            uint32_t family_gid = *forcing.refuted_family_gid;
            uint32_t same_ctor_gid = get_or_build_same_ctor(family_gid);
            uint32_t empty_gid = get_or_build_empty_type();
            // refuted_pat_value/refuted_conc_value, NOT
            // inst.result_indices[refuted_slot]/concrete_indices
            // [refuted_slot] directly: for the ordinary refutation path
            // (a direct, rigid mismatch at the top level, e.g. Vec's
            // own Succ(n_field) vs a concrete Zero) these coincide
            // exactly, so this is a no-op there. For the repeated-field
            // path (refl's own (A,x,x), e.g. Eq Nat Zero (Succ Zero))
            // they DIVERGE: inst.result_indices[refuted_slot] would be
            // the bare, still-generic field reference (x), which
            // same_ctor cannot reduce on (neither side head-rigid) --
            // refuted_pat_value is the already-forced, genuinely
            // head-rigid value match_index_pattern actually compared
            // (see this struct's own doc comment for why it is always
            // safe to embed here, both paths alike).
            NodeId absurd_ty = apply_all(mk_const(same_ctor_gid), {*forcing.refuted_pat_value, *forcing.refuted_conc_value});
            uint32_t absurd_level = g.ec.size() + ctor_arity;
            // branch_expected, as raw syntax, still contains
            // motive_original's OWN internal Pi-chain, whose own binder
            // positions were fixed (absolute levels, relative to
            // g.ec.size() at motive-construction time) for embedding
            // directly at THIS constructor's own depth (g.ec.size() +
            // ctor_arity) -- not one level deeper, which is exactly
            // where the inner Empty-match's own motive body places it
            // (below the extra `absurd` binder). Embedding it there
            // unreduced leaves those internal references stale --
            // confirmed directly: this crashed check()'s own comparison
            // (VCtor vs VNeutral) the first time this was tried without
            // the fix below, not merely a theoretical concern. Fully
            // evaluating and re-quoting AT branch_expected's own correct
            // depth genuinely beta-reduces motive_original's application
            // away entirely (it collapses to a plain type, e.g. `Vec
            // Zero`, with no residual Pi-chain of its own left to go
            // stale), which is what makes it safe to embed one level
            // deeper afterward. The env used must still include fresh
            // placeholders for this constructor's own fields (ctor_arity
            // many), since branch_expected may genuinely reference them
            // even though they're not bound in g.ec.values itself.
            Env branch_expected_env = g.ec.values;
            for (uint32_t ai = 0; ai < ctor_arity; ++ai) branch_expected_env = branch_expected_env.extend(fresh_neutral(g.ec.size() + ai));
            NodeId branch_expected_reduced = quote(eval(branch_expected, branch_expected_env), absurd_level);
            NodeId absurd_body = mk_match(mk_var(absurd_level), mk_lambda(mk_const(empty_gid), branch_expected_reduced), {});
            NodeId refuted_branch = mk_lambda(absurd_ty, absurd_body);
            for (auto it = arg_types.rbegin(); it != arg_types.rend(); ++it) refuted_branch = mk_lambda(*it, refuted_branch);
            branches.push_back(refuted_branch);
            continue;
        }

        // Auto-intro: the new goal's own telescope already includes this
        // constructor's fields as named hypotheses -- with index forcing
        // already applied, so the agent sees the concrete type directly.
        //
        // Rebuilt from scratch as the prefix through the FULL, original
        // g.ec.size() -- not just hyp_level+1: the constructor's own new
        // fields (arg_types, below) are positioned by the REST of this
        // function (ctor_app, shifted_motive_original) at g.ec.size() +
        // ai, unconditionally, matching the motive's own construction,
        // which is fixed within the full g.ec and cannot itself shrink
        // to some smaller, per-branch prefix without breaking its own
        // standalone type-check (infer(motive_original, g.ec, ps.es),
        // above). A prefix of only hyp_level+1 is strictly SMALLER than
        // g.ec.size() whenever anything exists after hyp_level at all
        // (transitivity's own h2 does) -- leaving a gap between the
        // prefix's own end and where arg_types actually land, which
        // this kernel's purely-sequential ElabCtx/Env cannot represent:
        // an earlier version of this fix stopped at hyp_level+1 and
        // silently, incorrectly placed arg_types one level too low,
        // confirmed directly by a genuinely wrong absolute-level
        // reference (a reverted hypothesis's own type pointing at a
        // constructor field's own binder instead of that field's own,
        // separately-generic type) surviving all the way to
        // whole_term_check before being caught. So this prefix always
        // spans the full [0, g.ec.size()), regardless of hyp_level or
        // revert_from: any hypothesis reverted from EITHER side of
        // hyp_level -- before it (z) or after it (h2) -- simply remains,
        // stale and unreferenced, at its OLD position here, exactly the
        // same treatment already given to an earlier-reverted hyp,
        // harmless since the generalization already redirected every
        // genuine reference to its OWN, freshly re-introduced position
        // below, and nothing else in the new goal points back at the
        // old one.
        ElabCtx branch_ec = ElabCtx{};
        branch_ec.self_aware = g.ec.self_aware; // extend() alone preserves this going forward; the fresh default would otherwise silently drop it
        for (uint32_t i = 0; i < g.ec.size(); ++i) branch_ec = branch_ec.extend(g.hyps[i].type);
        std::vector<Hypothesis> branch_hyps(g.hyps.begin(), g.hyps.begin() + g.ec.size());
        for (NodeId field_ty : arg_types) {
            branch_ec = branch_ec.extend(field_ty);
            branch_hyps.push_back(Hypothesis{"", field_ty});
        }
        for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
            branch_ec = branch_ec.extend(reverted_dom_types[ri]);
            branch_hyps.push_back(Hypothesis{g.hyps[reverted_js[ri]].name, reverted_dom_types[ri]});
        }

        // Genuine new subgoal: unwrapped, scoped over branch_ec (g's own
        // context PLUS this constructor's fields PLUS the reverted
        // hyps' own, freshly re-introduced ones).
        uint32_t subgoal_meta = ctx().meta_ctx.fresh(branch_expected, branch_ec.size());
        NodeId subgoal_ref = mk_meta(subgoal_meta);
        for (uint32_t i = 0; i < branch_ec.size(); ++i) subgoal_ref = mk_app(subgoal_ref, mk_var(i));

        // This branch's own content: subgoal_ref, wrapped ONLY by the
        // binders this branch itself genuinely, newly introduces
        // (reverted_dom_types, innermost, then arg_types, outermost) --
        // deliberately NOT by branch_ec's own PREFIX (g's own, already-
        // existing hyps), which must instead be referenced directly, at
        // their own, original, ambient absolute levels (subgoal_ref's
        // own mk_var(0..g.ec.size()-1) applications already do exactly
        // this).
        //
        // An earlier version of this fix instead routed every branch
        // through a SEPARATE reference meta, itself Pi-wrapped over the
        // FULL branch_ec (prefix included) and solved to a closure over
        // subgoal_ref -- appealingly symmetric, but unsound: eval(Match)
        // evaluates a branch's own syntax WITHIN THE SAME env as the
        // enclosing match itself (see eval(Match)'s own VCtor case),
        // never a fresh one, so re-wrapping the prefix as brand-new
        // Lambda binders here does not merely duplicate the ambient
        // A,x,y,z,h1,h2 -- it COLLIDES with them: the branch's own,
        // re-introduced "A" binder lands at whatever level env.size()
        // happens to be at that point (already g.ec.size(), since the
        // env there already holds the ambient prefix), the same level
        // arg_types' own fields are separately, correctly assumed to
        // occupy by the rest of this function (ctor_app,
        // shifted_motive_original). A reverted hypothesis's own type
        // (reverted_dom_types, referencing arg_types' own fields by
        // their true, ambient level, e.g. g.ec.size() for the first
        // field) would then resolve against this collision instead,
        // silently producing the WRONG value -- confirmed directly,
        // this session, via whole_term_check rejecting an otherwise
        // fully tactic-solved trans proof, its own extracted term
        // showing a reverted hypothesis's type pointing at the
        // prefix's own re-introduced parameter instead of the
        // constructor's own, separate field. Wrapping only the
        // genuinely new binders here, and leaving the prefix to be
        // referenced directly (exactly how the ORIGINAL, non-reverting
        // cases() already, correctly worked, before this session's own
        // sibling-hypothesis extension existed at all) sidesteps the
        // collision entirely, since nothing re-introduces the prefix as
        // new binders in the first place.
        NodeId branch_body = subgoal_ref;
        for (auto it = reverted_dom_types.rbegin(); it != reverted_dom_types.rend(); ++it) branch_body = mk_lambda(*it, branch_body);
        for (auto it = arg_types.rbegin(); it != arg_types.rend(); ++it) branch_body = mk_lambda(*it, branch_body);

        if (exploit_ci) {
            // Some sibling constructor is refuted, so motive_final
            // carries the extra same_ctor(...) -> layer for EVERY
            // branch, including this one -- wrap branch_body (BEFORE
            // its own arg_types wrapping) with one trivial extra
            // parameter of THIS constructor's own same_ctor(its own
            // pattern, the known value) type (which always reduces to
            // Unit, since it's comparing the SAME constructor's own
            // head to itself -- see get_or_build_same_ctor's own doc
            // comment), so this branch's own type matches what
            // motive_final now requires without changing what the
            // agent is actually asked to prove (still exactly
            // branch_expected, unwrapped).
            //
            // Deliberately uses THIS ci's own inst.result_indices, not
            // forcings[*exploit_ci]'s own refuted_pat_value (the fix
            // applied at the exploited branch itself, above): that
            // value belongs to the EXPLOITED constructor's own pattern,
            // not this (surviving, sibling) one's -- substituting it in
            // here would compare the wrong constructor's own shape
            // entirely. Never exercised by a single-constructor family
            // (Eq included: the loop above always has exactly one ci,
            // which is always *exploit_ci* itself, so this branch of
            // the if/else never runs there) -- genuinely UNEXAMINED for
            // a multi-constructor family where a SURVIVING sibling
            // constructor itself has a repeated-index pattern at
            // refuted_slot (this ci's own inst.result_indices
            // [refuted_slot] would then, by the same reasoning as the
            // exploited branch's own fix, potentially be a bare field
            // reference here too) -- flagged, not assumed safe.
            //
            // exploit_ci is mutually exclusive with reverted_js being
            // non-empty (guarded off above), so reverted_dom_types is
            // always empty here -- branch_body, at this point, is
            // exactly subgoal_ref, not yet wrapped by anything.
            uint32_t refuted_slot = *forcings[*exploit_ci].refuted_slot;
            uint32_t family_gid = *forcings[*exploit_ci].refuted_family_gid;
            uint32_t same_ctor_gid = get_or_build_same_ctor(family_gid);
            NodeId this_wrapper_ty = apply_all(mk_const(same_ctor_gid), {inst.result_indices[refuted_slot], concrete_indices[refuted_slot]});
            NodeId wrapped = mk_lambda(this_wrapper_ty, branch_body);
            for (auto it = arg_types.rbegin(); it != arg_types.rend(); ++it) wrapped = mk_lambda(*it, wrapped);
            branches.push_back(wrapped);
        } else {
            branches.push_back(branch_body);
        }

        new_goals.push_back(Goal{subgoal_meta, branch_ec, std::move(branch_hyps), branch_expected});
    }

    NodeId match_term = mk_match(mk_var(hyp_level), motive_final, branches);
    if (exploit_ci) {
        // Apply to the trivial witness that the whole match now expects
        // as its own extra argument (motive_final's own outermost result
        // is a Pi, not the goal directly -- see the motive extension
        // above): same_ctor(the SCRUTINEE's own actual index, itself)
        // always reduces to Unit, so its own single constructor is a
        // valid witness regardless of which family is involved.
        uint32_t unit_ctor_gid = get_or_build_unit_type().second;
        match_term = mk_app(match_term, mk_const(unit_ctor_gid));
    }
    // Mutually exclusive with exploit_ci (guarded off above whenever
    // reverted_js is non-empty), so no double-application to worry
    // about here: for exactly the same reason exploit_ci's own witness
    // must be supplied, motive_final's own fully-applied result is now
    // the REVERTED HYPS' OWN Pi-CHAIN (generalized_goal's own outermost
    // structure -- confirmed directly, this session, by printing it),
    // not g.target directly -- so match_term itself, exactly like any
    // ordinary function of that Pi-chain's own type, must be applied to
    // each reverted hypothesis's own value to actually produce
    // something of g.target's own type. Applied in reverted_js's own,
    // increasing order, matching the Pi-chain's own outermost-to-
    // innermost nesting (reverted_target was built by wrapping g.target
    // in reverse order, so reverted_js[0]'s own type ends up as the
    // OUTERMOST binder -- see reverted_target's own construction
    // above). Each value is still an ordinary, unchanged, in-scope
    // local variable at this point (mk_var(reverted_js[ri])): reverting
    // only ever affected the MOTIVE's own, separate, generic
    // abstraction of it, never the actual hypothesis itself, which the
    // kept, full-size prefix (g.ec unchanged) still supplies directly.
    for (uint32_t rj : reverted_js) match_term = mk_app(match_term, mk_var(rj));
    ctx().meta_ctx.solve(g.meta_id, close_over_context(match_term, g.ec));

    ps.goals.erase(ps.goals.begin() + *idx_opt);
    ps.goals.insert(ps.goals.begin() + *idx_opt, new_goals.begin(), new_goals.end());
    return {};
}
#endif


#if 0 // not yet ported to the hybrid variable representation: leq_value
/// Universe and Pi-type subtyping.
///
/// `VType i` is a subtype of `VType j` exactly when `i <= j`. `Pi` is
/// cumulative in its codomain: `Pi(A,B) <= Pi(A,B')` when `B <= B'` under
/// a shared fresh variable. The domain is required to be exactly
/// convertible rather than treated contravariantly -- this is a
/// deliberate simplification, matching common practice (Coq's own Pi
/// cumulativity works the same way) rather than the more general
/// contravariant-domain rule some systems use. A consumer that needs full
/// contravariance can extend this case directly.
///
/// Every other pair of shapes falls back to conv_value: subtyping beyond
/// universes and Pi codomains -- cumulative inductive types, for example
/// -- is not implemented here. This also means leq_value(a, a, depth) is
/// always true, since equality is the floor of any subtyping relation and
/// conv_value already establishes it for every shape.
inline bool leq_value(const Value& a, const Value& b, uint32_t depth) {
    return inspect_value(a, b,
        [&](const VType& ta, const VType& tb) -> bool {
            return ta.universe_level <= tb.universe_level;
        },
        [&](const VPi& pa, const VPi& pb) -> bool {
            if (!conv_value(eval(pa.dom_id, pa.env), eval(pb.dom_id, pb.env), depth)) return false;
            Value fresh = fresh_neutral(depth);
            return leq_value(eval(pa.cod_id, pa.env.extend(fresh)), eval(pb.cod_id, pb.env.extend(fresh)), depth + 1);
        },
        [&](const auto& x, const auto& y) -> bool {
            return conv_value(Value{x}, Value{y}, depth);
        }
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: leq
/// Are two syntactic terms related by subtyping (leq_value) under the
/// given semantic environment? Use this where a theory allows universe
/// cumulativity -- for example, checking a term's inferred type against
/// an expected type -- in place of conv() where exact equality would be
/// too strict.
inline bool leq(NodeId a, NodeId b, const Env& env) {
    return leq_value(eval(a, env), eval(b, env), env.size());
}
#endif


// ---------------------------------------------------------------------------
// Proof irrelevance
// ---------------------------------------------------------------------------
//
// conv()/leq() above have no way to short-circuit for propositions: they
// receive only two terms, never the type those terms belong to, so they
// cannot tell "these are two different natural numbers" apart from "these
// are two different proofs of the same proposition, which should be
// treated as interchangeable." That distinction is fundamentally a typed
// one -- a real bidirectional typechecker, which this header does not
// provide, is what would ordinarily have the type in hand already at the
// point it needs to compare two terms.
//
// conv_at/leq_at below take the type as an explicit argument, to be
// supplied by whatever DOES have it (a future elaborator, mid-inference).
// This is deliberately additive: conv()/conv_value() are untouched, and
// nothing here changes their behavior. A consumer that never calls
// conv_at/leq_at gets exactly the same conversion checking as before.
//
// Prop is represented as Type{0} under a naming convention (Sort 0 = Prop,
// the same convention Lean 4 uses), not a new node kind -- this composes
// with universe polymorphism and cumulativity as they already exist,
// with no representation changes anywhere.
//
// *** STATUS: the short-circuit is currently DISABLED. See the note on
// is_prop_sort below for why, and do not re-enable it without fixing the
// condition it is supposed to test. ***
//
// SCOPE, stated plainly: even once corrected, this only helps at
// comparison time, for a caller that already has the type. It does NOT
// make eval() itself skip normalizing a proof passed as an ordinary
// function argument -- eval(App) has no type information flowing through
// it at all, and giving it one would be a much larger redesign (typed
// evaluation). Real systems that erase irrelevant subterms from
// evaluation itself (not just comparison) do it as a separate
// elaboration/compilation pass with full type information available,
// distinct from kernel checking -- out of scope for a header with no
// typechecker. Also left to whoever builds that elaborator: whether Prop
// is impredicative (a proposition may quantify over anything, including
// other propositions, without bumping its own universe level -- Coq's and
// Lean's actual convention). That's a typing-rule decision this
// classification doesn't make or need.

#if 0 // not yet ported to the hybrid variable representation: is_prop_sort
/// Is `ty` the proposition sort? Following Lean 4's Sort 0 = Prop
/// convention: universe level 0 specifically, not any Type whatsoever.
/// This function itself is correct -- "is this Value literally the sort
/// Type{0}" is exactly what it claims to check. The bug that existed here
/// was never in this function; it was in what conv_at/leq_at PASSED to
/// it (see below).
inline bool is_prop_sort(const Value& ty) {
    if (auto* t = std::get_if<VType>(&ty)) return t->universe_level == 0;
    return false;
}
#endif


#if 0 // not yet ported to the hybrid variable representation: conv_at
/// Definitional equality of `a` and `b`, known to both have type `ty_id`.
///
/// The correct condition for "a and b are interchangeable proofs and can
/// be treated as convertible without inspection" is "ty_id ITSELF has
/// type Prop", i.e. `infer(ty_id) == Type{0}` -- ty_id is a proposition,
/// and a/b are two proofs OF it. The bug this replaced checked "ty_id
/// EVALUATES TO Type{0}" instead -- ty_id IS the sort Prop itself, and
/// a/b are two ELEMENTS of Prop (i.e. two propositions such as True and
/// False), not two proofs of one proposition -- an unsound impredicative
/// collapse of all of Prop to a point. See ELABORATOR_PLAN.md and this
/// session's fix history for the full derivation; now that infer()
/// exists, this is exactly the corrected condition it was scoped for.
///
/// `ec` is taken BY VALUE (ElabCtx is two small shared_ptr-rooted
/// structs, cheap to copy) rather than by reference, since conv_at
/// doesn't need to propagate any extension back to its caller -- this
/// also keeps `{}` valid at existing call sites unchanged (ElabCtx{}
/// default-constructs to the same empty state `Env{}` did). A fresh,
/// throwaway ElabState is used for the internal infer(ty_id) call: any
/// constraint it would defer is a sign ty_id itself contains an
/// unresolved hole, an edge case out of scope for a simple query
/// function like this one; if infer() fails outright for any reason,
/// conv_at safely falls through to real comparison rather than treating
/// failure as license to guess.
inline bool conv_at(NodeId ty_id, NodeId a_id, NodeId b_id, ElabCtx ec) {
    ElabState local_es;
    if (TypeResult ty_sort = infer(ty_id, ec, local_es); ty_sort && is_prop_sort(eval(*ty_sort, ec.values))) {
        return true;
    }
    return conv_value(eval(a_id, ec.values), eval(b_id, ec.values), ec.size());
}
#endif


#if 0 // not yet ported to the hybrid variable representation: leq_at
/// The leq_value/leq counterpart of conv_at -- see its doc comment;
/// nothing here differs beyond calling leq_value instead of conv_value.
inline bool leq_at(NodeId ty_id, NodeId a_id, NodeId b_id, ElabCtx ec) {
    ElabState local_es;
    if (TypeResult ty_sort = infer(ty_id, ec, local_es); ty_sort && is_prop_sort(eval(*ty_sort, ec.values))) {
        return true;
    }
    return leq_value(eval(a_id, ec.values), eval(b_id, ec.values), ec.size());
}
#endif


// ---------------------------------------------------------------------------
// Universe polymorphism
// ---------------------------------------------------------------------------
//
// Rather than modeling a general algebra of universe-level expressions
// (max, imax of variables) throughout eval/quote/conv, a polymorphic
// declaration is stored as a template that may contain TypeUniv nodes,
// each referring to one of the declaration's universe parameters.
// Instantiating it at a specific list of concrete levels substitutes each
// TypeUniv away, producing an ordinary term that the rest of this library
// -- eval, quote, conv, leq, printing -- handles exactly as before,
// without needing to know polymorphism exists. This covers the common
// case of a declaration whose sort is directly one of its own parameters
// (as `id.{u} : {A : Sort u} -> A -> A` is); it does not cover a
// declaration whose sort is computed from several parameters via max or
// imax, which would need the fuller level algebra this deliberately avoids.

#if 0 // not yet ported to the hybrid variable representation: subst_univ_params
/// Rebuilds `t`, replacing every TypeUniv{k} with a concrete Type node at
/// level `level_args[k]`. Does not descend into the bodies of globals
/// referenced via Const -- a Const to another polymorphic declaration is
/// expected to be instantiated separately, through its own call to
/// instantiate_global, not silently rewritten by this one.
inline NodeId subst_univ_params(NodeId t, const std::vector<uint32_t>& level_args) {
    return inspect(t,
        [&](Var v) { return mk_var(v.level); },
        [&](App a) {
            return mk_app(subst_univ_params(a.fn_id, level_args), subst_univ_params(a.arg_id, level_args));
        },
        [&](Lambda l) {
            return mk_lambda(subst_univ_params(l.type_id, level_args), subst_univ_params(l.body_id, level_args));
        },
        [&](Let l) {
            return mk_let(subst_univ_params(l.type_id, level_args), subst_univ_params(l.value_id, level_args),
                          subst_univ_params(l.body_id, level_args));
        },
        [&](Pi p) {
            return mk_pi(subst_univ_params(p.dom_id, level_args), subst_univ_params(p.cod_id, level_args));
        },
        [&](Type ty) { return mk_type(ty.universe_level); },
        [&](Const c) { return mk_const(c.global_id); },
        [&](Match m) {
            std::vector<NodeId> branches;
            for (NodeId b : *m.branches) branches.push_back(subst_univ_params(b, level_args));
            return mk_match(subst_univ_params(m.scrutinee_id, level_args),
                             subst_univ_params(m.motive_id, level_args), std::move(branches));
        },
        [&](Meta meta) { return mk_meta(meta.meta_id); },
        [&](TypeUniv tu) {
            if (tu.param_index >= level_args.size()) {
                throw std::logic_error("itp: universe parameter index out of range during instantiation");
            }
            return mk_type(level_args[tu.param_index]);
        },
        [&](QuotType q) { return mk_quot_type(subst_univ_params(q.relation, level_args)); },
        [&](QuotMk q) {
            return mk_quot_mk(subst_univ_params(q.relation, level_args), subst_univ_params(q.elem, level_args));
        },
        [&](QuotLift q) {
            return mk_quot_lift(subst_univ_params(q.relation, level_args), subst_univ_params(q.codomain, level_args),
                                 subst_univ_params(q.f, level_args), subst_univ_params(q.respects_proof, level_args),
                                 subst_univ_params(q.quotient_elem, level_args));
        },
        [&](Fix f) {
            return mk_fix(subst_univ_params(f.fn_type, level_args), f.decreasing_arg,
                           subst_univ_params(f.body, level_args));
        }
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: declare_polymorphic
/// Declares a universe-polymorphic global. `type_template` and, if
/// present, `value_template` may reference TypeUniv{0..num_univ_params-1}.
/// Returns an id used with instantiate_global, not with mk_const directly
/// -- the template itself is never installed as an ordinary GlobalDef.
inline uint32_t declare_polymorphic(std::string name, uint32_t num_univ_params,
                                     NodeId type_template, std::optional<NodeId> value_template) {
    ctx().poly_decls.push_back(PolyDecl{num_univ_params, type_template, std::move(value_template), std::move(name)});
    return static_cast<uint32_t>(ctx().poly_decls.size() - 1);
}
#endif


#if 0 // not yet ported to the hybrid variable representation: instantiate_global
/// Instantiates a polymorphic declaration at a specific list of concrete
/// universe levels, installing the result as an ordinary (monomorphic)
/// global and returning a Const referencing it. Each call creates a fresh
/// global; repeated instantiation at the same levels is not memoized here
/// -- consistent with this library's general stance of not optimizing a
/// path before there is evidence it needs it -- so a consumer calling
/// this often at identical arguments may want to cache the result itself.
inline NodeId instantiate_global(uint32_t poly_id, const std::vector<uint32_t>& univ_args) {
    const PolyDecl& decl = ctx().poly_decls.at(poly_id);
    if (univ_args.size() != decl.num_univ_params) {
        throw std::logic_error(
            "itp: wrong number of universe arguments for polymorphic declaration '" + decl.name + "'");
    }
    NodeId inst_type = subst_univ_params(decl.type_template, univ_args);
    std::optional<NodeId> inst_value;
    if (decl.value_template) inst_value = subst_univ_params(*decl.value_template, univ_args);
    uint32_t fresh_gid = ctx().globals.declare(inst_type, inst_value, decl.name);
    return mk_const(fresh_gid);
}
#endif


// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
//
// Source locations and display names are kept in a table keyed by NodeId,
// separate from the node data itself. Attaching this information directly
// to node fields would break hash consing, since two occurrences of the
// same term at different source locations would then no longer be
// field-equal.
//
// If two structurally identical terms are interned at different times, one
// with attached source info and one without, they share a single NodeId
// and therefore a single diagnostic entry; whichever was attached first is
// kept. Applications that need per-occurrence diagnostics regardless of
// deduplication would need to store a collection of SourceInfo per key
// instead of a single one.

struct SourceInfo {
    uint32_t line = 0, col = 0;
    std::string file;
    std::string display_name;
};

struct DiagnosticTable {
    std::unordered_map<NodeId, SourceInfo> info;
    void attach(NodeId id, SourceInfo s) { info[id] = std::move(s); }
    std::optional<SourceInfo> lookup(NodeId id) const {
        auto it = info.find(id);
        if (it == info.end()) return std::nullopt;
        return it->second;
    }
};

#if 0 // not yet ported to the hybrid variable representation: render_error
inline std::string render_error(const TypeError& err, const DiagnosticTable& diag) {
    std::ostringstream out;
    out << "type error: " << err.message;
    if (auto info = diag.lookup(err.offending)) {
        out << " (at " << info->file << ":" << info->line << ":" << info->col << ")";
    }
    return out.str();
}
#endif


// ---------------------------------------------------------------------------

// Printing
// ---------------------------------------------------------------------------

#if 0 // not yet ported to the hybrid variable representation: print_term
inline std::string print_term(NodeId id, std::vector<std::string>& names,
                               const DiagnosticTable* diag) {
    return inspect(id,
        [&](Var v) -> std::string {
            if (diag) {
                if (auto info = diag->lookup(id); info && !info->display_name.empty())
                    return info->display_name;
            }
            if (v.level < names.size()) return names[v.level];
            return "v" + std::to_string(v.level);
        },
        [&](App a) -> std::string {
            return "(" + print_term(a.fn_id, names, diag) + " " + print_term(a.arg_id, names, diag) + ")";
        },
        [&](Lambda l) -> std::string {
            std::string fresh = "x" + std::to_string(names.size());
            if (diag) {
                if (auto info = diag->lookup(id); info && !info->display_name.empty())
                    fresh = info->display_name;
            }
            std::string dom = print_term(l.type_id, names, diag);
            names.push_back(fresh);
            std::string body = print_term(l.body_id, names, diag);
            names.pop_back();
            return "(\\" + fresh + ":" + dom + ". " + body + ")";
        },
        [&](Let l) -> std::string {
            std::string fresh = "x" + std::to_string(names.size());
            if (diag) {
                if (auto info = diag->lookup(id); info && !info->display_name.empty())
                    fresh = info->display_name;
            }
            std::string ty = print_term(l.type_id, names, diag);
            std::string val = print_term(l.value_id, names, diag);
            names.push_back(fresh);
            std::string body = print_term(l.body_id, names, diag);
            names.pop_back();
            return "(let " + fresh + ":" + ty + " := " + val + " in " + body + ")";
        },
        [&](Pi p) -> std::string {
            std::string fresh = "x" + std::to_string(names.size());
            std::string dom = print_term(p.dom_id, names, diag);
            names.push_back(fresh);
            std::string cod = print_term(p.cod_id, names, diag);
            names.pop_back();
            return "(Pi " + fresh + ":" + dom + ". " + cod + ")";
        },
        [&](Type t) -> std::string {
            return "Type" + std::to_string(t.universe_level);
        },
        [&](Const c) -> std::string {
            if (diag) {
                if (auto info = diag->lookup(id); info && !info->display_name.empty())
                    return info->display_name;
            }
            const std::string& name = ctx().globals[c.global_id].name;
            return name.empty() ? ("c" + std::to_string(c.global_id)) : name;
        },
        [&](Match m) -> std::string {
            std::string out = "(match " + print_term(m.scrutinee_id, names, diag) + " with";
            for (NodeId b : *m.branches) out += " | " + print_term(b, names, diag);
            out += ")";
            return out;
        },
        [&](Meta meta) -> std::string {
            return "?m" + std::to_string(meta.meta_id);
        },
        [&](TypeUniv tu) -> std::string {
            return "u" + std::to_string(tu.param_index);
        },
        [&](QuotType q) -> std::string {
            return "(Quot " + print_term(q.relation, names, diag) + ")";
        },
        [&](QuotMk q) -> std::string {
            return "(Quot.mk " + print_term(q.relation, names, diag) + " " + print_term(q.elem, names, diag) + ")";
        },
        [&](QuotLift q) -> std::string {
            return "(Quot.lift " + print_term(q.f, names, diag) + " " + print_term(q.respects_proof, names, diag) +
                   " " + print_term(q.quotient_elem, names, diag) + ")";
        },
        [&](Fix f) -> std::string {
            std::string fresh = "self" + std::to_string(names.size());
            if (diag) {
                if (auto info = diag->lookup(id); info && !info->display_name.empty())
                    fresh = info->display_name;
            }
            std::string ty = print_term(f.fn_type, names, diag);
            names.push_back(fresh);
            std::string body = print_term(f.body, names, diag);
            names.pop_back();
            return "(fix " + fresh + ":" + ty + " [decreasing_arg=" + std::to_string(f.decreasing_arg) +
                   "] := " + body + ")";
        }
    );
}
#endif


#if 0 // not yet ported to the hybrid variable representation: print
inline std::string print(NodeId id, const DiagnosticTable* diag = nullptr) {
    std::vector<std::string> names;
    return print_term(id, names, diag);
}
#endif


// ---------------------------------------------------------------------------
// Inductive types
// ---------------------------------------------------------------------------
//
// This supports a single, non-mutual inductive type at a time -- Nat,
// Bool, lists of a fixed element type, binary trees, infinitely-branching
// trees (a constructor argument may itself be a function type whose
// codomain mentions the inductive, provided the inductive never appears in
// a domain position; see is_strictly_positive below), and indexed families
// such as Eq a b, where a constructor's `result_indices` targets a specific
// instantiation of the family's indices rather than the family applied to
// nothing.
//
// Not supported:
//   - Mutual inductive blocks, where several types reference each other's
//     not-yet-declared global ids.
//   - Nested inductive occurrences, such as a constructor argument of type
//     `List Bad` where Bad is the type being defined. is_strictly_positive
//     does not detect these; they are treated as acceptable rather than
//     rejected. A general check would need to examine the nested type
//     family's own variance, which is not implemented here.
//
// Reduction of Match does not need to know anything about indices --
// dispatch is purely by constructor tag, exactly as for a non-indexed
// type. What indexing adds is entirely in how a constructor's type is
// built: verifying that a Match's motive and branches are well-formed
// relative to the specific indices each constructor targets, and
// propagating the index equality learned from a successful match (for
// example, knowing y = x after matching refl), is elaboration-time logic
// left to the consumer of this library.

struct CtorSpec {
    std::string name;
    std::vector<NodeId> arg_types; // may reference the inductive's own (already-reserved) global_id,
                                    // and, for a dependent field, earlier entries in this same list
    std::vector<NodeId> result_indices = {}; // which instantiation of the family's indices this
                                              // constructor targets; empty for a non-indexed type,
                                              // in which case the constructor's return type is the
                                              // bare inductive applied to nothing
};

struct InductiveHandle {
    uint32_t type_global_id;
    std::vector<uint32_t> ctor_global_ids;
};
inline NodeId build_ctor_type(uint32_t type_global_id, const CtorSpec& ctor);
inline InductiveHandle finish_inductive_unchecked(uint32_t type_global_id, std::vector<CtorSpec> ctors);
inline InductiveHandle finish_inductive(uint32_t type_global_id, std::vector<CtorSpec> ctors, bool allow_unsafe_skip_positivity = false);

/// Whether `ind_type_global_id` occurs anywhere within `term`, without
/// regard to position. Used as a building block by is_strictly_positive,
/// which is where position (positive vs. negative) is taken into account.
inline bool occurs(uint32_t ind_type_global_id, NodeId term) {
    return inspect(term,
        [&](Var) { return false; },
        [&](App a) { return occurs(ind_type_global_id, a.fn_id) || occurs(ind_type_global_id, a.arg_id); },
        [&](Lambda l) { return occurs(ind_type_global_id, l.type_id) || occurs(ind_type_global_id, l.body_id); },
        [&](Let l) {
            return occurs(ind_type_global_id, l.type_id) || occurs(ind_type_global_id, l.value_id) ||
                   occurs(ind_type_global_id, l.body_id);
        },
        [&](Pi p) { return occurs(ind_type_global_id, p.dom_id) || occurs(ind_type_global_id, p.cod_id); },
        [&](Type) { return false; },
        [&](Const c) { return c.global_id == ind_type_global_id; },
        [&](Match m) {
            if (occurs(ind_type_global_id, m.scrutinee_id) || occurs(ind_type_global_id, m.motive_id)) return true;
            for (NodeId b : *m.branches) if (occurs(ind_type_global_id, b)) return true;
            return false;
        },
        [&](Meta) { return false; },
        [&](TypeUniv) { return false; },
        [&](QuotType q) { return occurs(ind_type_global_id, q.relation); },
        [&](QuotMk q) { return occurs(ind_type_global_id, q.relation) || occurs(ind_type_global_id, q.elem); },
        [&](QuotLift q) {
            return occurs(ind_type_global_id, q.relation) || occurs(ind_type_global_id, q.codomain) ||
                   occurs(ind_type_global_id, q.f) || occurs(ind_type_global_id, q.respects_proof) ||
                   occurs(ind_type_global_id, q.quotient_elem);
        },
        [&](Fix f) {
            // A Fix is not expected to appear inside a constructor's own
            // argument type in ordinary use (Fix is proof-term-scoped,
            // built by Induct, not part of the small vocabulary
            // CtorSpec::arg_types is written in) -- but occurs() is a
            // syntactic search over arbitrary NodeId, not something
            // restricted to "shapes a constructor spec would plausibly
            // contain," so it must still descend correctly rather than
            // silently reporting "doesn't occur" for a shape it simply
            // wasn't taught to look inside.
            return occurs(ind_type_global_id, f.fn_type) || occurs(ind_type_global_id, f.body);
        }
    );
}


/// Restricted strict positivity check for a single, non-mutual inductive.
///
/// `ind_type_global_id` may occur in `arg_type` directly, or in a positive
/// (codomain) position, but never in a negative (domain) position of any
/// arrow within it. This accepts ordinary recursive arguments (Succ :
/// Nat -> Nat, where the argument type is a direct occurrence of Nat) and
/// infinitely-branching arguments (a Branch constructor taking
/// Nat -> Tree, where Tree occurs only in a codomain), while rejecting
/// constructors such as `MkBad : (Bad -> Bool) -> Bad`, where Bad occurs
/// in the domain of the argument type's own arrow. See
/// itp_kernel_test.cpp for tests covering both the accepted and rejected
/// cases, including the infinitely-branching case, which is what rules
/// out a simpler "no direct occurrence" check.
///
/// In the non-Pi case, only a DIRECT (possibly index-applied) reference to
/// the inductive itself is accepted -- an occurrence buried inside some
/// OTHER type's application, for example a constructor argument of type
/// `List Bad`, is rejected rather than silently accepted. Verifying that
/// nested case would require examining List's own variance, which this
/// restricted checker does not do; failing closed here rather than
/// treating "not a Pi" as automatically safe was a real, previously
/// unnoticed gap in an earlier version of this check.
inline bool is_strictly_positive(uint32_t ind_type_global_id, NodeId arg_type) {
    return inspect(arg_type,
        [&](Pi p) {
            if (occurs(ind_type_global_id, p.dom_id)) return false;
            return is_strictly_positive(ind_type_global_id, p.cod_id);
        },
        [&](auto) {
            if (!occurs(ind_type_global_id, arg_type)) return true; // doesn't mention it at all: safe
            // It occurs somewhere. Accept only if arg_type IS, at its
            // application head, a direct reference to the inductive
            // (covering the index-applied case, e.g. `Eq A x x`); reject
            // every other way it could occur. Written without try_match,
            // which is declared later in this header.
            NodeId head = arg_type;
            bool progressed = true;
            while (progressed) {
                progressed = inspect(head,
                    [&](App a) { head = a.fn_id; return true; },
                    [&](auto) { return false; }
                );
            }
            return inspect(head,
                [&](Const c) { return c.global_id == ind_type_global_id; },
                [&](auto) { return false; }
            );
        }
    );
}


/// Reserves a global id for an inductive type before its constructors are
/// declared. This two-phase declaration exists because a recursive
/// constructor's argument type (Succ's `Nat`, for example) must reference
/// the inductive's own global id, which does not exist until the
/// inductive has been at least partially declared.
inline uint32_t reserve_inductive_type(const std::string& name, NodeId sort) {
    return ctx().globals.declare(sort, std::nullopt, name);
}

#if 0 // not yet ported to the hybrid variable representation: build_ctor_type
/// Builds a single constructor's full curried type: its argument types as
/// a Pi chain, ending in the inductive applied to `ctor.result_indices`.
/// Shared between finish_inductive and finish_mutual_inductive_block.
inline NodeId build_ctor_type(uint32_t type_global_id, const CtorSpec& ctor) {
    NodeId ctor_ty = apply_all(mk_const(type_global_id), ctor.result_indices);
    for (auto it = ctor.arg_types.rbegin(); it != ctor.arg_types.rend(); ++it) {
        ctor_ty = mk_pi(*it, ctor_ty);
    }
    return ctor_ty;
}
#endif


/// Admission checks for an inductive declaration (Finding F1). Run by finish_inductive and
/// finish_mutual_inductive_block unless the explicit unsafe flag is passed.
///
/// Before this existed, admission checked strict positivity and NOTHING else: the family's own
/// declared type, every constructor argument type and every result index were taken on trust.
/// That is how a Girard-style paradox gets in: an inductive living in the impredicative universe 0
/// with a constructor field that is itself a large type (e.g. `mk : Type0 -> U`) lets `U` encode
/// "Type0 embeds into a member of Type0".
///
/// Checks, per constructor, in a telescope context that grows one field at a time:
///   (1) the family's declared type is well-typed and is a Pi telescope ending in a sort `Type l`;
///   (2) every constructor argument type is a type (infer yields a sort), well-typed in the context
///       of the earlier fields;
///   (3) the family applied to result_indices is well-typed and FULLY applied (right arity, each
///       index of the right type);
///   (4) universe rule: every field that is not parameter-like must have a type living in a
///       universe <= the family's own universe `l`. A field is parameter-like iff it occurs as a
///       bare variable among the constructor's result indices -- this kernel encodes a parameter
///       as a leading field forced by the result indices (Eq's A, Ex's P), and a parameter is not
///       a constructor argument in Coq/Lean, so it is exempt from the constraint.
/// With this rule the impredicative universe 0 behaves like Coq's impredicative Set: strong
/// elimination is fine, but classical axioms must not be added to a system that conflates Prop and
/// data in universe 0 (see FINDINGS notes).
inline void check_inductive_admission(uint32_t type_gid, const std::vector<CtorSpec>& ctors) {
    ExecutionContext& c = ctx();
    const std::string fam = c.globals.defs[type_gid].name;
    NodeId declared = c.globals.defs[type_gid].type;
    {
        ElabState es0; ElabCtx ec0;
        auto r = infer(declared, ec0, es0);
        bool ok = false;
        if (r) { Value v = eval(*r, ec0.values); ok = std::get_if<VType>(&v) != nullptr; }
        if (!ok) throw std::logic_error("itp: inductive '" + fam + "': its declared type is ill-typed or is not a type");
    }
    NodeId cur = declared;
    for (bool go = true; go;) {
        go = inspect(cur, [&](Pi p) { cur = p.cod_id; return true; }, [&](auto) { return false; });
    }
    std::optional<uint32_t> fam_level = inspect(cur,
        [&](Type t) -> std::optional<uint32_t> { return t.universe_level; },
        [&](auto) -> std::optional<uint32_t> { return std::nullopt; });
    if (!fam_level) {
        throw std::logic_error("itp: inductive '" + fam + "': its declared type must be a Pi telescope ending in a sort");
    }
    for (const auto& ctor : ctors) {
        ElabState es; ElabCtx ec;
        std::vector<uint32_t> field_level;
        for (std::size_t j = 0; j < ctor.arg_types.size(); ++j) {
            auto sr = infer(ctor.arg_types[j], ec, es);
            if (!sr) {
                throw std::logic_error("itp: constructor '" + ctor.name + "' of '" + fam + "': field " +
                                       std::to_string(j) + " is ill-typed (" + sr.error().message + ")");
            }
            Value sv = eval(*sr, ec.values);
            auto* vt = std::get_if<VType>(&sv);
            if (!vt) {
                throw std::logic_error("itp: constructor '" + ctor.name + "' of '" + fam + "': field " +
                                       std::to_string(j) + "'s declared type is not a type");
            }
            field_level.push_back(vt->universe_level);
            ec = ec.extend(ctor.arg_types[j]);
        }
        NodeId applied = apply_all(mk_const(type_gid), ctor.result_indices);
        auto rr = infer(applied, ec, es);
        bool applied_ok = false;
        if (rr) { Value rv = eval(*rr, ec.values); applied_ok = std::get_if<VType>(&rv) != nullptr; }
        if (!applied_ok) {
            throw std::logic_error("itp: constructor '" + ctor.name + "' of '" + fam + "': its result indices are "
                                   "ill-typed, or do not fully apply the family (wrong number of indices)");
        }
        for (std::size_t j = 0; j < field_level.size(); ++j) {
            bool parameter_like = false;
            for (NodeId ri : ctor.result_indices) {
                if (inspect(ri, [&](Var v) { return !v.bound && v.n == j; }, [&](auto) { return false; })) { parameter_like = true; break; }
            }
            if (!parameter_like && field_level[j] > *fam_level) {
                throw std::logic_error("itp: constructor '" + ctor.name + "' of '" + fam + "': field " + std::to_string(j) +
                                       " lives in universe " + std::to_string(field_level[j]) + ", larger than the family's own "
                                       "universe " + std::to_string(*fam_level) + " (a large field in a small inductive is "
                                       "inconsistent with an impredicative universe)");
            }
        }
    }
}


#if 0 // not yet ported to the hybrid variable representation: finish_inductive
/// Declares the constructors of a previously reserved inductive type,
/// after checking each constructor argument for strict positivity (unless
/// explicitly bypassed, which should be reserved for experimentation).
inline InductiveHandle finish_inductive(uint32_t type_global_id, std::vector<CtorSpec> ctors,
                                         bool allow_unsafe_skip_positivity = false) {
    if (!allow_unsafe_skip_positivity) {
        for (const auto& ctor : ctors) {
            for (NodeId arg_ty : ctor.arg_types) {
                if (!is_strictly_positive(type_global_id, arg_ty)) {
                    throw std::logic_error(
                        "itp: constructor '" + ctor.name + "' fails strict positivity -- the inductive "
                        "being defined appears to the left of an arrow in an argument type");
                }
            }
        }
    }
    if (!allow_unsafe_skip_positivity) check_inductive_admission(type_global_id, ctors);
    ExecutionContext& c = ctx();
    InductiveHandle handle;
    handle.type_global_id = type_global_id;
    for (std::size_t idx = 0; idx < ctors.size(); ++idx) {
        NodeId ctor_ty = build_ctor_type(type_global_id, ctors[idx]);
        uint32_t arity = static_cast<uint32_t>(ctors[idx].arg_types.size());
        uint32_t ctor_gid = c.globals.declare_ctor(ctor_ty, ctors[idx].name, type_global_id, static_cast<uint32_t>(idx), arity);
        handle.ctor_global_ids.push_back(ctor_gid);
    }
    return handle;
}
#endif


/// Step 3 (design spec, section on refutation exploitation): lazily
/// builds and caches this context's own canonical Unit type (one
/// constructor, `__tt`) -- used as the "same constructor head" case of
/// same_ctor below. Named with a `__` prefix to avoid colliding with a
/// user- or library-declared "Unit" in the same context.
inline std::pair<uint32_t, uint32_t> get_or_build_unit_type() {
    if (ctx().unit_type_gid && ctx().unit_ctor_gid) return {*ctx().unit_type_gid, *ctx().unit_ctor_gid};
    NodeId type0_v = mk_type(0);
    uint32_t tid = reserve_inductive_type("__Unit", type0_v);
    InductiveHandle h = finish_inductive(tid, {CtorSpec{"__tt", {}}});
    ctx().unit_type_gid = tid;
    ctx().unit_ctor_gid = h.ctor_global_ids[0];
    return {tid, h.ctor_global_ids[0]};
}


/// Lazily builds and caches this context's own canonical Empty type
/// (zero constructors) -- eliminating a value of this type (an ordinary
/// Match with zero branches, already fully supported by
/// check_match_branches' own existing arity-count handling, no kernel
/// change needed for that part) produces any type at all.
inline uint32_t get_or_build_empty_type() {
    if (ctx().empty_type_gid) return *ctx().empty_type_gid;
    NodeId type0_v = mk_type(0);
    uint32_t tid = reserve_inductive_type("__Empty", type0_v);
    finish_inductive(tid, {});
    ctx().empty_type_gid = tid;
    return tid;
}


#if 0 // not yet ported to the hybrid variable representation: get_or_build_same_ctor
/// Lazily builds and caches, PER FAMILY, a discriminator function
/// `same_ctor : F -> F -> Type0` -- Unit when both arguments share a
/// constructor head, Empty when they don't, regardless of what either
/// side's own fields (wildcarded throughout) are. General across any
/// family's own arity and any constructor's own arity, including
/// dependent fields (a later field's type instantiated fresh, at the
/// correct depth, for each nesting point separately -- NOT precomputed
/// once and reused at a mismatched depth, which was tried and found
/// wrong first: dependent fields need their own telescope recomputed at
/// each distinct point they're embedded, exactly the discipline this
/// whole session has repeatedly needed).
///
/// This is the actual mechanism Step 3 exploits: for two constructors of
/// F with DIFFERENT heads, `same_ctor` reduces DIRECTLY to Empty via
/// ordinary evaluation (both scrutinees are head-concrete, so eval(Match)
/// never gets stuck) -- confirmed empirically, not merely derived, that
/// this holds even when one side's own fields are symbolic (e.g.
/// `same_ctor(Zero, Succ k)` reduces to Empty for a free `k`). For the
/// SAME constructor applied to itself with possibly-symbolic fields
/// (`same_ctor(Succ k, Succ m)` for free `k`, `m`), it reduces to Unit --
/// also confirmed empirically. Both sides must be at least
/// HEAD-CONCRETE for this reduction to fire (a genuinely bare,
/// non-constructor-headed variable leaves the match stuck, since
/// eval(Match) has no head to dispatch on) -- but this is never actually
/// needed: the exploitation construction below only ever applies
/// same_ctor to a CONSTRUCTOR'S OWN index pattern (Vec's cons's own
/// `Succ(n_field)`, say), which is always head-concrete by construction,
/// regardless of which constructor it is or what its own fields are.
/// This specific property (not merely "Empty is reachable somehow") is
/// what makes exploitation work without an Eq-based convoy: extending a
/// match's own motive with
/// `same_ctor(this_slot's_own_index_param, the_known_concrete_value) ->`
/// turns the refuted branch's own requirement into a trivial `Empty ->
/// GOAL` (immediate via Empty's own zero-branch eliminator) and every
/// OTHER branch's requirement into an equally trivial `Unit -> GOAL`
/// wrapping around whatever proof it needed anyway -- confirmed by a
/// full, independently type-checked and normalized construction on Vec,
/// not assumed from the reduction behavior alone.
inline uint32_t get_or_build_same_ctor(uint32_t family_gid) {
    auto cached = ctx().same_ctor_gid_by_family.find(family_gid);
    if (cached != ctx().same_ctor_gid_by_family.end()) return cached->second;

    NodeId family_ty = mk_const(family_gid);
    NodeId type0_v = mk_type(0);
    NodeId empty_ty = mk_const(get_or_build_empty_type());
    NodeId unit_ty = mk_const(get_or_build_unit_type().first);
    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(family_gid);
    uint32_t n = static_cast<uint32_t>(ctor_gids.size());
    NodeId const_type0_motive = mk_lambda(family_ty, type0_v);

    // Inner match (on b, level 1) for outer branch `same_i`, given
    // `outer_arity` fields of the outer branch's own constructor already
    // bound before it -- so b's own branch fields (this inner match's
    // own new binders) correctly start at depth 2 (a, b) + outer_arity,
    // recomputed fresh here rather than reused from anywhere else.
    auto build_inner = [&](uint32_t same_i, uint32_t outer_arity) -> NodeId {
        std::vector<NodeId> inner_branches(n);
        for (uint32_t jb = 0; jb < n; ++jb) {
            detail::InstantiatedCtor jinst = detail::instantiate_ctor_telescope(ctor_gids[jb], 2 + outer_arity);
            NodeId branch = (jb == same_i) ? unit_ty : empty_ty;
            for (auto f = jinst.arg_types.rbegin(); f != jinst.arg_types.rend(); ++f) branch = mk_lambda(*f, branch);
            inner_branches[jb] = branch;
        }
        return mk_match(mk_var(1), const_type0_motive, inner_branches);
    };

    std::vector<NodeId> outer_branches(n);
    for (uint32_t i = 0; i < n; ++i) {
        detail::InstantiatedCtor iinst = detail::instantiate_ctor_telescope(ctor_gids[i], 2);
        uint32_t outer_arity = static_cast<uint32_t>(iinst.arg_types.size());
        NodeId branch = build_inner(i, outer_arity);
        for (auto f = iinst.arg_types.rbegin(); f != iinst.arg_types.rend(); ++f) branch = mk_lambda(*f, branch);
        outer_branches[i] = branch;
    }

    NodeId same_ctor_body = mk_lambda(family_ty, mk_lambda(family_ty,
        mk_match(mk_var(0), mk_lambda(family_ty, type0_v), outer_branches)));
    NodeId same_ctor_ty = mk_pi(family_ty, mk_pi(family_ty, type0_v));
    auto declared = declare_checked(same_ctor_ty, same_ctor_body, "__same_ctor_" + std::to_string(family_gid));
    if (!declared) {
        // A bug in THIS synthesis, not in anything the caller supplied --
        // every input here is kernel-derived (constructor telescopes of
        // an already-validated family), not user syntax, so failure here
        // means the synthesis itself is wrong and must be fixed, not
        // handled as an ordinary, recoverable elaboration error.
        throw std::logic_error("itp: internal error -- synthesized same_ctor discriminator for family " +
            std::to_string(family_gid) + " failed to type-check: " + declared.error().message);
    }
    ctx().same_ctor_gid_by_family[family_gid] = *declared;
    return *declared;
}
#endif


/// Convenience wrapper combining reserve_inductive_type and
/// finish_inductive, for inductives whose constructors do not need to
/// reference the type's own global id ahead of time.
inline InductiveHandle declare_inductive(const std::string& name, NodeId sort, std::vector<CtorSpec> ctors,
                                          bool allow_unsafe_skip_positivity = false) {
    uint32_t type_id = reserve_inductive_type(name, sort);
    return finish_inductive(type_id, std::move(ctors), allow_unsafe_skip_positivity);
}


// ---------------------------------------------------------------------------
// Mutual inductive blocks
// ---------------------------------------------------------------------------
//
// Reduction needs no changes to support this, for the same reason indexed
// families needed none: Match dispatches purely by ctor_idx, so mutuality
// is entirely a declaration-time concern. The reserve/finish pattern
// already supports building this by hand (reserve every type in the block
// first, so each type's constructors can freely reference any of the
// others via Const, then finish each one) -- what that does NOT do safely
// on its own is positivity: finish_inductive only checks that ITS OWN
// type doesn't occur negatively in its constructors, saying nothing about
// whether, in a block of Even/Odd, Odd occurs negatively inside Even's
// constructors. A hypothetical `MkEven : (Odd -> Bool) -> Even` would
// pass finish_inductive(even_id, ...) unchallenged, since Even itself
// never appears in that argument type. finish_mutual_inductive_block
// closes this by checking every constructor's argument types against
// EVERY type in the block, not just its own.

struct MutualIndSpec {
    uint32_t type_global_id; // already reserved via reserve_inductive_type
    std::vector<CtorSpec> ctors;
};

inline std::vector<InductiveHandle> finish_mutual_inductive_block(
    const std::vector<uint32_t>& all_type_ids, std::vector<MutualIndSpec> specs,
    bool allow_unsafe_skip_positivity = false) {
    if (!allow_unsafe_skip_positivity) {
        for (const auto& spec : specs) {
            for (const auto& ctor : spec.ctors) {
                for (NodeId arg_ty : ctor.arg_types) {
                    for (uint32_t other_id : all_type_ids) {
                        if (!is_strictly_positive(other_id, arg_ty)) {
                            throw std::logic_error(
                                "itp: constructor '" + ctor.name + "' of a mutual inductive block fails strict "
                                "positivity with respect to one of the types in the block");
                        }
                    }
                }
            }
        }
    }
    if (!allow_unsafe_skip_positivity) {
        for (const auto& spec : specs) check_inductive_admission(spec.type_global_id, spec.ctors);
    }
    ExecutionContext& c = ctx();
    std::vector<InductiveHandle> handles;
    for (const auto& spec : specs) {
        InductiveHandle handle;
        handle.type_global_id = spec.type_global_id;
        for (std::size_t idx = 0; idx < spec.ctors.size(); ++idx) {
            NodeId ctor_ty = build_ctor_type(spec.type_global_id, spec.ctors[idx]);
            uint32_t arity = static_cast<uint32_t>(spec.ctors[idx].arg_types.size());
            uint32_t ctor_gid = c.globals.declare_ctor(ctor_ty, spec.ctors[idx].name, spec.type_global_id, static_cast<uint32_t>(idx), arity);
            handle.ctor_global_ids.push_back(ctor_gid);
        }
        handles.push_back(std::move(handle));
    }
    return handles;
}


// ---------------------------------------------------------------------------
// Pattern-matching and error-handling utilities
// ---------------------------------------------------------------------------
//
// inspect() dispatches on a single level of a term's shape, so a compound
// pattern -- an App whose function is itself a Lambda, for example --
// requires nesting one inspect() inside another. try_match below is a
// small primitive for building named, reusable patterns from such
// compositions once, so that call sites become a single flat check rather
// than repeating the nesting each time.

/// Extracts one alternative of the syntactic Term domain, or nullopt if
/// `id` is not that shape.
template <typename T>
std::optional<T> try_match(NodeId id) {
    return inspect(id,
        [](const T& t) -> std::optional<T> { return t; },
        [](const auto&) -> std::optional<T> { return std::nullopt; }
    );
}

/// Two-argument form, for checking that two terms share a given shape
/// without writing out the full N*N inspect() catch-all.
template <typename T, typename U>
std::optional<std::pair<T, U>> try_match(NodeId a, NodeId b) {
    return inspect(a, b,
        [](const T& ta, const U& tb) -> std::optional<std::pair<T, U>> { return std::make_pair(ta, tb); },
        [](const auto&, const auto&) -> std::optional<std::pair<T, U>> { return std::nullopt; }
    );
}

/// The Value-domain counterpart of try_match.
template <typename T>
std::optional<T> try_match_value(const Value& v) {
    return inspect_value(v,
        [](const T& t) -> std::optional<T> { return t; },
        [](const auto&) -> std::optional<T> { return std::nullopt; }
    );
}

/// A beta redex: an application whose function position is a lambda.
struct BetaRedex {
    Lambda lambda;
    NodeId arg_id;
};

/// Recognizes a beta redex by its literal syntactic shape. Given
/// `id_fun := \x. x` and the term `id_fun y`, this returns nullopt: the
/// function position is a Const, not a Lambda, even though id_fun unfolds
/// to one. This is intentional rather than an oversight -- real kernels
/// commonly distinguish beta-only reduction from beta-plus-delta reduction
/// as separate strategies, since "was this written as a redex" and "would
/// this become one after unfolding definitions" are different questions
/// with different uses. See match_beta_redex_whnf below for the version
/// that unfolds definitions.
inline std::optional<BetaRedex> match_beta_redex(NodeId t) {
    return try_match<App>(t).and_then([](App a) {
        return try_match<Lambda>(a.fn_id).transform([&](Lambda l) {
            return BetaRedex{l, a.arg_id};
        });
    });
}


/// Repeatedly unfolds a Const with a definition (delta-reduction) until
/// reaching a node that is either not a Const, or is a Const with no
/// definition (an axiom or a constructor). This is purely syntactic
/// head-unfolding: it does not evaluate applications, and it does not
/// recurse into subterms other than the head. It exists so that pattern
/// matching can optionally see through a global definition's name without
/// requiring full evaluation of the term.
///
/// This loop only continues while the head stays a Const; it stops as
/// soon as unfolding reaches a Lambda, Pi, or any other non-Const shape.
/// An ordinary recursive function's definition is a Lambda, so this
/// terminates after a single step regardless of what the function's body
/// references. Divergence is possible only for a circular chain of
/// Const-to-Const aliases with no other node in between (`a := b`,
/// `b := a`), which is a degenerate case rather than an ordinary
/// definition; no cycle detection is performed for it.
inline NodeId unfold_head_delta(NodeId t) {
    while (t.tag == NodeType::CONST) {
        const GlobalDef& def = ctx().globals[ctx().consts[t.index].global_id];
        if (!def.value) break;
        t = *def.value;
    }
    return t;
}


/// The delta-aware counterpart of match_beta_redex: unfolds any chain of
/// definitions at the function position before checking for a lambda, so
/// that `id_fun y` is correctly recognized as a redex when
/// `id_fun := \x. x`. Use this for reduction and simplification code,
/// where the effective shape after unfolding is what matters; use
/// match_beta_redex where the literal written shape matters instead.
inline std::optional<BetaRedex> match_beta_redex_whnf(NodeId t) {
    return try_match<App>(t).and_then([](App a) {
        NodeId fn_head = unfold_head_delta(a.fn_id);
        return try_match<Lambda>(fn_head).transform([&](Lambda l) {
            return BetaRedex{l, a.arg_id};
        });
    });
}


/// Converts an optional into an expected, for bridging try_match's
/// optional-based results into std::expected-based error handling: a
/// structural mismatch is usually a real, reportable error rather than
/// something to fail silently on.
template <typename T, typename E>
std::expected<T, E> require(std::optional<T> opt, E err) {
    if (opt) return std::expected<T, E>(std::move(*opt));
    return std::unexpected(std::move(err));
}


// ---------------------------------------------------------------------------
// Structural recursion checking
// ---------------------------------------------------------------------------
//
// A restricted termination check: a recursive definition is accepted if
// every self-call passes, at a fixed argument position, a value obtained
// by destructuring (via Match) something already known to be structurally
// smaller than the original decreasing parameter. This is enough for the
// common case of a function recursing on one argument of an inductive
// type -- addition, list operations, and similar.
//
// Not supported: well-founded recursion beyond direct structural descent
// (typically encoded via an indexed accessibility predicate once that is
// needed), a decreasing argument at a position that varies between calls,
// and recursion through a field whose type is a function returning the
// inductive rather than the inductive directly. is_strictly_positive
// accepts that shape (the infinitely-branching case); this checker does
// not follow it, since doing so needs a different, higher-order recursion
// scheme rather than tracking a single decreasing variable.

/// For a constructor's declared type, a Pi chain, returns which argument
/// positions have a domain whose head -- after stripping any index
/// applications -- is directly `Const{ind_type_global_id}`. These are the
/// fields that are themselves values of the inductive being matched, as
/// opposed to some other type, including a function type that merely
/// returns the inductive.
inline std::vector<bool> recursive_field_mask(uint32_t ind_type_global_id, uint32_t ctor_gid) {
    std::vector<bool> mask;
    NodeId t = ctx().globals[ctor_gid].type;
    while (auto p = try_match<Pi>(t)) {
        NodeId head = p->dom_id;
        while (auto a = try_match<App>(head)) head = a->fn_id;
        auto c = try_match<Const>(head);
        mask.push_back(c.has_value() && c->global_id == ind_type_global_id);
        t = p->cod_id;
    }
    return mask;
}


/// Checks that `body` -- the full definition of `fn_gid`, including its own
/// leading Lambdas -- is structurally decreasing on the parameter at
/// absolute level `decreasing_arg_level`, for a single-clause, non-mutual
/// function recursing over `ind_type_global_id`. Every self-call must
/// supply, at the same argument position as the decreasing parameter, a
/// variable already established as structurally smaller by a Match that
/// destructured the original parameter (or something already smaller).
inline bool is_structurally_decreasing(uint32_t fn_gid, uint32_t ind_type_global_id,
                                        uint32_t decreasing_arg_level, NodeId body) {
    std::unordered_set<uint32_t> smaller;

    std::function<bool(NodeId, uint32_t)> walk = [&](NodeId t, uint32_t depth) -> bool {
        // A self-call is checked by unrolling the application spine and
        // inspecting its head, ahead of the generic structural walk below,
        // since a self-call has a different acceptance condition than an
        // ordinary subterm.
        NodeId head = t;
        std::vector<NodeId> args;
        while (auto a = try_match<App>(head)) { args.push_back(a->arg_id); head = a->fn_id; }
        std::reverse(args.begin(), args.end());
        if (auto c = try_match<Const>(head); c && c->global_id == fn_gid) {
            bool args_ok = true;
            for (NodeId a : args) args_ok = walk(a, depth) && args_ok;
            if (args.size() <= decreasing_arg_level) return false; // not applied far enough to verify
            auto v = try_match<Var>(args[decreasing_arg_level]);
            bool call_ok = v.has_value() && smaller.count(v->n) > 0;
            return call_ok && args_ok;
        }

        return inspect(t,
            [&](Var) { return true; },
            [&](App a) { return walk(a.fn_id, depth) && walk(a.arg_id, depth); },
            [&](Lambda l) { return walk(l.type_id, depth) && walk(l.body_id, depth + 1); },
            [&](Let l) {
                if (!walk(l.type_id, depth) || !walk(l.value_id, depth)) return false;
                // Narrow propagation: if the let-bound value is literally
                // a bare variable already known smaller, the new binding
                // inherits smaller-ness too -- covering the common
                // `let y := n' in ...` readability pattern. Aliasing the
                // ORIGINAL, undestructured decreasing parameter itself
                // does NOT count -- a let-binding destructures nothing,
                // so `let k := n in f k` is exactly as non-decreasing as
                // calling f with n directly. An earlier version of this
                // check incorrectly treated that case as smaller, copied
                // from the Match-scrutinee check below where it IS valid
                // (scrutinizing the original parameter is what licenses
                // treating ITS destructured fields as smaller -- aliasing
                // it via let destructures nothing). That was a real,
                // exploitable soundness bug: it accepted
                // `bad(n) := match n with Zero -> Zero
                //                       | Succ n' -> let k := n in bad k`
                // as valid structural recursion, even though it never
                // actually decreases and diverges for any non-Zero input.
                bool value_is_smaller = false;
                if (auto v = try_match<Var>(l.value_id)) {
                    value_is_smaller = smaller.count(v->n) > 0;
                }
                if (value_is_smaller) smaller.insert(depth);
                bool ok = walk(l.body_id, depth + 1);
                if (value_is_smaller) smaller.erase(depth);
                return ok;
            },
            [&](Pi p) { return walk(p.dom_id, depth) && walk(p.cod_id, depth + 1); },
            [&](Type) { return true; },
            [&](Const c) { return c.global_id != fn_gid; }, // a bare, unapplied self-reference is rejected
            [&](Match m) {
                if (!walk(m.scrutinee_id, depth) || !walk(m.motive_id, depth)) return false;
                bool scrutinee_is_smaller = false;
                if (auto v = try_match<Var>(m.scrutinee_id)) {
                    scrutinee_is_smaller = (v->n == decreasing_arg_level) || smaller.count(v->n) > 0;
                }
                for (std::size_t ci = 0; ci < m.branches->size(); ++ci) {
                    std::vector<bool> mask;
                    if (scrutinee_is_smaller) {
                        // O(1) lookup + direct array index, replacing
                        // what used to be a full O(all globals) scan
                        // PER BRANCH (so O(branches * globals) total for
                        // this one Match) -- and reading the mask
                        // precomputed once at declaration time (see
                        // GlobalEnv::declare_ctor) rather than
                        // recomputing (and heap-allocating) it here via
                        // recursive_field_mask on every call.
                        const auto& family_ctors = ctx().globals.ctors_of(ind_type_global_id);
                        if (ci < family_ctors.size()) mask = ctx().globals[family_ctors[ci]].ctor_info->recursive_mask;
                    }
                    NodeId cur = (*m.branches)[ci];
                    uint32_t cur_depth = depth;
                    std::size_t field = 0;
                    std::vector<uint32_t> added;
                    while (auto l = try_match<Lambda>(cur)) {
                        if (field < mask.size() && mask[field]) {
                            smaller.insert(cur_depth);
                            added.push_back(cur_depth);
                        }
                        cur = l->body_id;
                        ++cur_depth;
                        ++field;
                    }
                    bool ok = walk(cur, cur_depth);
                    for (uint32_t lvl : added) smaller.erase(lvl); // facts learned in this branch don't leak into others
                    if (!ok) return false;
                }
                return true;
            },
            [&](Meta) { return true; }, // a bare unsolved meta reference contains no recursive calls
            [&](TypeUniv) { return true; },
            [&](QuotType q) { return walk(q.relation, depth); },
            [&](QuotMk q) { return walk(q.relation, depth) && walk(q.elem, depth); },
            [&](QuotLift q) {
                return walk(q.relation, depth) && walk(q.codomain, depth) && walk(q.f, depth) &&
                       walk(q.respects_proof, depth) && walk(q.quotient_elem, depth);
            },
            [&](Fix f) {
                // A nested local Fix introduces exactly one new binder
                // (its own self-reference), the same shape as Lambda/Pi
                // above: walk fn_type at the current depth, walk body
                // one level deeper. This checker has no notion of the
                // NESTED Fix's own guardedness (that is exactly what
                // is_locally_structurally_decreasing checks, separately,
                // at Induct-acceptance time for THAT Fix) -- this walk
                // only needs to keep tracking the OUTER function's own
                // self-calls correctly as it descends through the
                // nested binder, the same as it already does through an
                // ordinary Lambda/Pi/Match branch.
                return walk(f.fn_type, depth) && walk(f.body, depth + 1);
            }
        );
    };

    return walk(body, 0);
}


/// The LOCAL counterpart of is_structurally_decreasing, for a Fix's own
/// body rather than a global recursive definition's. A near-verbatim
/// copy with exactly one structural change: the self-call is a bound
/// variable at `self_ref_level` (the Fix's own binder), not a `Const`
/// referencing a global id, since a local Fix has no global identity at
/// all -- see Fix's own doc comment on why this is the deliberate
/// alternative to routing every in-progress Induct attempt through
/// GlobalEnv. Every other rule (the smaller-set walk, the Let-aliasing
/// fix, per-branch smaller-set scoping, the nested-Fix binder shift)
/// carries over completely unchanged from is_structurally_decreasing.
///
/// The two case-level differences this forces, both mechanical
/// consequences of "self is a Var, not a Const": the generic Var case
/// must now reject a bare, unapplied self-reference (self_ref_level's
/// own role, previously played by the generic Const case) instead of
/// unconditionally accepting every Var; and the generic Const case, in
/// turn, unconditionally accepts every Const, since there is no global
/// self-reference for a LOCAL fixpoint to compare against.
inline bool is_locally_structurally_decreasing(uint32_t self_ref_level, uint32_t ind_type_global_id,
                                                uint32_t decreasing_arg, NodeId body) {
    std::unordered_set<uint32_t> smaller;

    std::function<bool(NodeId, uint32_t)> walk = [&](NodeId t, uint32_t depth) -> bool {
        NodeId head = t;
        std::vector<NodeId> args;
        while (auto a = try_match<App>(head)) { args.push_back(a->arg_id); head = a->fn_id; }
        std::reverse(args.begin(), args.end());
        if (auto v0 = try_match<Var>(head); v0 && v0->n == self_ref_level) {
            bool args_ok = true;
            for (NodeId a : args) args_ok = walk(a, depth) && args_ok;
            if (args.size() <= decreasing_arg) {
                return false;
            }
            auto v = try_match<Var>(args[decreasing_arg]);
            bool call_ok = v.has_value() && smaller.count(v->n) > 0;
            return call_ok && args_ok;
        }

        return inspect(t,
            [&](Var v) {
                return v.n != self_ref_level;
            },
            [&](App a) { return walk(a.fn_id, depth) && walk(a.arg_id, depth); },
            [&](Lambda l) { return walk(l.type_id, depth) && walk(l.body_id, depth + 1); },
            [&](Let l) {
                if (!walk(l.type_id, depth) || !walk(l.value_id, depth)) return false;
                bool value_is_smaller = false;
                if (auto v = try_match<Var>(l.value_id)) {
                    value_is_smaller = smaller.count(v->n) > 0;
                }
                if (value_is_smaller) smaller.insert(depth);
                bool ok = walk(l.body_id, depth + 1);
                if (value_is_smaller) smaller.erase(depth);
                return ok;
            },
            [&](Pi p) { return walk(p.dom_id, depth) && walk(p.cod_id, depth + 1); },
            [&](Type) { return true; },
            [&](Const) { return true; }, // no global self-reference concept for a LOCAL fixpoint
            [&](Match m) {
                if (!walk(m.scrutinee_id, depth) || !walk(m.motive_id, depth)) return false;
                bool scrutinee_is_smaller = false;
                if (auto v = try_match<Var>(m.scrutinee_id)) {
                    // NOTE, discovered empirically (not stated in the
                    // original port plan): decreasing_arg here must be
                    // resolved to its TRUE ABSOLUTE LEVEL, not compared
                    // directly, unlike is_structurally_decreasing's
                    // otherwise-identical line. For a GLOBAL function,
                    // decreasing_arg's POSITION in a self-call's
                    // argument list and its ABSOLUTE LEVEL as a
                    // top-level Lambda parameter coincide (nothing
                    // precedes the function's own curried Lambda
                    // chain). For a local Fix, they do NOT: `self`
                    // itself occupies self_ref_level, so the function's
                    // OWN curried parameters start at self_ref_level+1,
                    // meaning the decreasing_arg-th parameter's ABSOLUTE
                    // level is self_ref_level+1+decreasing_arg, not
                    // decreasing_arg itself. The self-call-argument
                    // check just above (`args[decreasing_arg]`) is
                    // correctly left untouched, since THAT usage really
                    // is a positional index, unaffected by self's
                    // extra level.
                    scrutinee_is_smaller = (v->n == self_ref_level + 1 + decreasing_arg) ||
                                           smaller.count(v->n) > 0;
                }
                for (std::size_t ci = 0; ci < m.branches->size(); ++ci) {
                    std::vector<bool> mask;
                    if (scrutinee_is_smaller) {
                        const auto& family_ctors = ctx().globals.ctors_of(ind_type_global_id);
                        if (ci < family_ctors.size()) mask = ctx().globals[family_ctors[ci]].ctor_info->recursive_mask;
                    }
                    NodeId cur = (*m.branches)[ci];
                    uint32_t cur_depth = depth;
                    std::size_t field = 0;
                    std::vector<uint32_t> added;
                    while (auto l = try_match<Lambda>(cur)) {
                        if (field < mask.size() && mask[field]) {
                            smaller.insert(cur_depth);
                            added.push_back(cur_depth);
                        }
                        cur = l->body_id;
                        ++cur_depth;
                        ++field;
                    }
                    bool ok = walk(cur, cur_depth);
                    for (uint32_t lvl : added) smaller.erase(lvl);
                    if (!ok) return false;
                }
                return true;
            },
            [&](Meta) { return true; },
            [&](TypeUniv) { return true; },
            [&](QuotType q) { return walk(q.relation, depth); },
            [&](QuotMk q) { return walk(q.relation, depth) && walk(q.elem, depth); },
            [&](QuotLift q) {
                return walk(q.relation, depth) && walk(q.codomain, depth) && walk(q.f, depth) &&
                       walk(q.respects_proof, depth) && walk(q.quotient_elem, depth);
            },
            [&](Fix f) {
                // A nested Fix (e.g. a further, INNER induction within
                // this one's branches) shifts self_ref_level's own
                // meaning not at all (it is an ABSOLUTE level, stable
                // regardless of how many more binders appear below it)
                // -- only the depth bookkeeping shifts, exactly as in
                // is_structurally_decreasing's identical case.
                return walk(f.fn_type, depth) && walk(f.body, depth + 1);
            }
        );
    };

    // body's own outermost binder (its first curried parameter) sits
    // at ABSOLUTE LEVEL self_ref_level+1, not 0 -- unlike a global
    // function's body, which has NOTHING preceding its own Lambda
    // chain, a local Fix's `self` itself occupies self_ref_level
    // first. depth must start there for its own bookkeeping to track
    // true absolute levels throughout the walk (this is what the
    // scrutinee-level conversion just above relies on).
    return walk(body, self_ref_level + 1);
}


/// One function's identity and decreasing-argument position within a
/// mutually recursive block.
struct MutualFnTarget {
    uint32_t fn_gid;
    uint32_t decreasing_arg_level;
};

#if 0 // not yet ported to the hybrid variable representation: is_structurally_decreasing_mutual
/// The mutual-recursion counterpart of is_structurally_decreasing.
/// `own_decreasing_arg_level` is the level of the CURRENT function's own
/// decreasing parameter (which seeds the smaller-set, the same role
/// `decreasing_arg_level` plays for the single-function version); `body`
/// is that same function's own definition. A call to ANY function in
/// `targets` (including a self-call) is accepted if it supplies, at that
/// callee's own recorded decreasing position, something already
/// established as smaller. All functions in `targets` are required to
/// recurse over the same `ind_type_global_id` -- a block mixing recursion
/// over different inductive types is not supported.
inline bool is_structurally_decreasing_mutual(const std::vector<MutualFnTarget>& targets,
                                                uint32_t ind_type_global_id, uint32_t own_decreasing_arg_level,
                                                NodeId body) {
    std::unordered_set<uint32_t> smaller;

    auto lookup = [&](uint32_t gid) -> std::optional<uint32_t> {
        for (const auto& t : targets) if (t.fn_gid == gid) return t.decreasing_arg_level;
        return std::nullopt;
    };

    std::function<bool(NodeId, uint32_t)> walk = [&](NodeId t, uint32_t depth) -> bool {
        NodeId head = t;
        std::vector<NodeId> args;
        while (auto a = try_match<App>(head)) { args.push_back(a->arg_id); head = a->fn_id; }
        std::reverse(args.begin(), args.end());
        if (auto c = try_match<Const>(head)) {
            if (auto pos = lookup(c->global_id)) {
                bool args_ok = true;
                for (NodeId a : args) args_ok = walk(a, depth) && args_ok;
                if (args.size() <= *pos) return false;
                auto v = try_match<Var>(args[*pos]);
                bool call_ok = v.has_value() && smaller.count(v->n) > 0;
                return call_ok && args_ok;
            }
        }

        return inspect(t,
            [&](Var) { return true; },
            [&](App a) { return walk(a.fn_id, depth) && walk(a.arg_id, depth); },
            [&](Lambda l) { return walk(l.type_id, depth) && walk(l.body_id, depth + 1); },
            [&](Let l) {
                if (!walk(l.type_id, depth) || !walk(l.value_id, depth)) return false;
                // See is_structurally_decreasing's identical comment: an
                // alias of the function's own undestructured decreasing
                // parameter does not count as smaller here either.
                bool value_is_smaller = false;
                if (auto v = try_match<Var>(l.value_id)) {
                    value_is_smaller = smaller.count(v->n) > 0;
                }
                if (value_is_smaller) smaller.insert(depth);
                bool ok = walk(l.body_id, depth + 1);
                if (value_is_smaller) smaller.erase(depth);
                return ok;
            },
            [&](Pi p) { return walk(p.dom_id, depth) && walk(p.cod_id, depth + 1); },
            [&](Type) { return true; },
            [&](Const c) { return !lookup(c.global_id).has_value(); }, // a bare, unapplied reference to any target is rejected
            [&](Match m) {
                if (!walk(m.scrutinee_id, depth) || !walk(m.motive_id, depth)) return false;
                bool scrutinee_is_smaller = false;
                if (auto v = try_match<Var>(m.scrutinee_id)) {
                    scrutinee_is_smaller = (v->n == own_decreasing_arg_level) || smaller.count(v->n) > 0;
                }
                for (std::size_t ci = 0; ci < m.branches->size(); ++ci) {
                    std::vector<bool> mask;
                    if (scrutinee_is_smaller) {
                        const auto& family_ctors = ctx().globals.ctors_of(ind_type_global_id);
                        if (ci < family_ctors.size()) mask = ctx().globals[family_ctors[ci]].ctor_info->recursive_mask;
                    }
                    NodeId cur = (*m.branches)[ci];
                    uint32_t cur_depth = depth;
                    std::size_t field = 0;
                    std::vector<uint32_t> added;
                    while (auto l = try_match<Lambda>(cur)) {
                        if (field < mask.size() && mask[field]) {
                            smaller.insert(cur_depth);
                            added.push_back(cur_depth);
                        }
                        cur = l->body_id;
                        ++cur_depth;
                        ++field;
                    }
                    bool ok = walk(cur, cur_depth);
                    for (uint32_t lvl : added) smaller.erase(lvl);
                    if (!ok) return false;
                }
                return true;
            },
            [&](Meta) { return true; },
            [&](TypeUniv) { return true; },
            [&](QuotType q) { return walk(q.relation, depth); },
            [&](QuotMk q) { return walk(q.relation, depth) && walk(q.elem, depth); },
            [&](QuotLift q) {
                return walk(q.relation, depth) && walk(q.codomain, depth) && walk(q.f, depth) &&
                       walk(q.respects_proof, depth) && walk(q.quotient_elem, depth);
            },
            [&](Fix f) {
                // See the identical case in is_structurally_decreasing.
                return walk(f.fn_type, depth) && walk(f.body, depth + 1);
            }
        );
    };

    return walk(body, 0);
}
#endif


/// Reserves a global id for a recursive function before its body is
/// supplied, mirroring reserve_inductive_type: the body needs to
/// reference the function's own global id, which does not exist until it
/// has been at least partially declared.
inline uint32_t reserve_recursive_function(const std::string& name, NodeId type) {
    return ctx().globals.declare(type, std::nullopt, name);
}

#if 0 // not yet ported to the hybrid variable representation: finish_recursive_function
/// Supplies the body of a previously reserved recursive function, after
/// checking it for structural recursion (unless explicitly bypassed,
/// which should be reserved for experimentation). For mutually recursive
/// functions, see finish_mutual_recursive_functions below.
inline void finish_recursive_function(uint32_t fn_gid, uint32_t ind_type_global_id,
                                       uint32_t decreasing_arg_level, NodeId body,
                                       bool allow_unsafe_skip_guard_check = false) {
    if (!allow_unsafe_skip_guard_check) {
        if (!is_structurally_decreasing(fn_gid, ind_type_global_id, decreasing_arg_level, body)) {
            throw std::logic_error(
                "itp: function '" + ctx().globals[fn_gid].name + "' failed the structural recursion check");
        }
    }
    ctx().globals.defs[fn_gid].value = body;
    ctx().globals.defs[fn_gid].recursive_info =
        RecursiveFnInfo{ind_type_global_id, decreasing_arg_level, allow_unsafe_skip_guard_check};
}
#endif


/// Checked counterpart of finish_recursive_function: guardedness alone
/// only proves termination, never that `body` actually has the type
/// `fn_gid` was reserved with (finish_recursive_function never checks
/// this either -- confirmed by reading its implementation above, same
/// gap as plain declare()). Installs `body` (self-references need it
/// present to resolve during check(), matching how every already-
/// passing whole-term check() in this codebase already relies on
/// values being installed before independent re-verification) only
/// provisionally, and ROLLS BACK to unset if check() rejects it, so a
/// failed verification never leaves a bad definition callable.
inline std::expected<void, TypeError> finish_recursive_function_checked(
        uint32_t fn_gid, uint32_t ind_type_global_id, uint32_t decreasing_arg_level, NodeId body,
        bool allow_unsafe_skip_guard_check = false) {
    if (!allow_unsafe_skip_guard_check) {
        if (!is_structurally_decreasing(fn_gid, ind_type_global_id, decreasing_arg_level, to_levels(body, 0))) {
            return std::unexpected(TypeError{TypeErrorKind::Other, body, std::nullopt,
                "function '" + ctx().globals[fn_gid].name + "' failed the structural recursion check"});
        }
    }
    ctx().globals.defs[fn_gid].value = body;
    ElabState es;
    ElabCtx empty_ec;
    ElabResult r = check(body, ctx().globals[fn_gid].type, empty_ec, es);
    if (!r) {
        ctx().globals.defs[fn_gid].value = std::nullopt; // roll back: never leave a bad definition installed
        return std::unexpected(r.error());
    }
    ctx().globals.defs[fn_gid].recursive_info =
        RecursiveFnInfo{ind_type_global_id, decreasing_arg_level, allow_unsafe_skip_guard_check};
    return {};
}


// ---------------------------------------------------------------------------
// Mutually recursive functions
// ---------------------------------------------------------------------------
//
// All functions checked together are required to recurse over the SAME
// inductive type (a block mixing recursion over, say, Nat and List is not
// supported); each function may recurse on a different argument position
// of its own signature, and may call any other function in the block
// (including itself) as long as, at that callee's own decreasing
// position, it supplies something already established as structurally
// smaller. Reserve every function's global id first, via
// reserve_recursive_function, exactly as for a single recursive function
// -- what changes is only how the bodies are checked and installed.

#if 0 // not yet ported to the hybrid variable representation: finish_mutual_recursive_functions
inline void finish_mutual_recursive_functions(const std::vector<MutualFnTarget>& targets,
                                                uint32_t ind_type_global_id, std::vector<NodeId> bodies,
                                                bool allow_unsafe_skip_guard_check = false) {
    if (targets.size() != bodies.size()) {
        throw std::logic_error("itp: finish_mutual_recursive_functions given mismatched targets and bodies");
    }
    if (!allow_unsafe_skip_guard_check) {
        for (std::size_t i = 0; i < targets.size(); ++i) {
            if (!is_structurally_decreasing_mutual(targets, ind_type_global_id, targets[i].decreasing_arg_level,
                                                    bodies[i])) {
                throw std::logic_error(
                    "itp: function '" + ctx().globals[targets[i].fn_gid].name +
                    "' (in a mutually recursive block) failed the structural recursion check");
            }
        }
    }
    for (std::size_t i = 0; i < targets.size(); ++i) {
        ctx().globals.defs[targets[i].fn_gid].value = bodies[i];
    }
}
#endif


#if 0 // not yet ported to the hybrid variable representation: induct
// ---------------------------------------------------------------------------
// Induct: the expensive half of the Destruct split
// ---------------------------------------------------------------------------
//
// Builds a LOCAL Fix wrapping a Match, the tactic-driven counterpart of
// itp_prelude_demo.cpp's own hand-written add_zero_right: there,
// `azr_gid`/`azr_ref`/reserve_recursive_function/finish_recursive_function
// (a GLOBAL declaration) play the role `mk_fix`/is_locally_structurally_
// decreasing play here, for exactly the reason Fix's own doc comment and
// PHASE2_PLAN.md give: routing every in-progress Induct attempt through
// GlobalEnv would permanently install a GlobalDef per attempt, including
// abandoned ones, which is the scalability landmine already flagged for
// GlobalEnv.defs elsewhere in this file.
//
// Two independent level schemes are in play here, deliberately kept
// separate rather than unified into one:
//
//   - The AGENT-FACING scheme (branch_ec/branch_hyps/subgoal_meta's own
//     type): g.ec.size()-based, built exactly like cases()'s own
//     scheme, completely unchanged. This is what the tactic user
//     actually sees as "the new goal's hypotheses."
//   - The TRUE RUNTIME scheme, for the pieces directly, operationally
//     embedded in the Fix/body/Match term eval() actually walks:
//     self at level g.ec.size(), the curried function's own idx
//     parameters at g.ec.size()+1.., the scrutinee at
//     g.ec.size()+1+arity, and each branch's own constructor fields
//     starting immediately after that (`body_base`).
//
// These two schemes do NOT need to be reconciled via any shift: a
// subgoal_meta's own bare reference, once embedded (unapplied at the
// point of construction, applied only within THIS branch's own,
// directly-built Lambda chain) to g.ec's OWN levels (stable regardless
// of embedding depth, by Env's append-only/prefix-preserving
// extension) plus this constructor's OWN body_base-level field
// variables, resolves correctly purely by argument POSITION once
// eval(Match) supplies the constructor's actual field values -- see
// the design note directly on the branch-construction loop below for
// why the induction hypothesis specifically must be computed INLINE,
// within the branch itself, rather than threaded through the
// close_over_context bridge cases() uses for its own (IH-free)
// branches.
//
// Field types are computed twice per constructor -- once at each
// level scheme -- rather than shifted from one to the other. For
// NON-DEPENDENT constructor fields (true of Nat/Bool/List, this
// session's own target, and any inductive whose fields don't reference
// earlier fields or the motive's own indices) the two calls produce
// the same field types regardless of which base level they start
// from, so this is exact, not approximate, for that common case. A
// constructor with genuinely level-dependent field types would need
// this kept in sync more carefully -- an honestly-scoped limitation,
// matching this file's own precedent of documenting exactly where
// Cases's index handling stops being fully general for indexed types.
inline ElabResult induct(ProofState& ps, uint32_t hyp_level, std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) {
        return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt,
            target_meta_id ? "induct: no goal with the given meta_id (already solved or superseded)"
                            : "induct: no open goals"});
    }
    Goal g = ps.goals[*idx_opt]; // copy: only committed to ps.goals on success
    if (hyp_level >= g.hyps.size()) {
        return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt,
            "induct: hypothesis index out of range"});
    }
    if (detail::is_pi_headed(g.target)) {
        // Defensive guard, not a soundness fix (verified directly, both ways, in this project's
        // own test_induct_pi_ordering.py): calling induct() while the goal's own target is still a
        // Pi -- i.e. before every quantified variable is intro()'d -- never leads to an unsound
        // proof (the resulting divergence between a later hypothesis's own listed index and its
        // genuine level is always caught by ordinary type-checking before whole_term_check is ever
        // reached), but it IS a silent dead end: nothing downstream can close such a goal, and
        // without this guard that dead end is only discovered several tactic steps later, wasting
        // search budget in the meantime -- a real cost for a policy that is free to call INDUCT on
        // any inductive-typed hypothesis already present, regardless of what Pi's remain in the
        // target (see hypergraph.py's own enumerate_legal_actions). Rejecting it immediately here
        // turns that silent, downstream dead end into the same kind of clean, instant tactic
        // failure any other inapplicable action already produces (e.g. CASES on a non-inductive
        // hypothesis) -- a pure efficiency improvement, not a behavior change for any goal this
        // guard does not reject: the established curriculum (mul_succ_right, mul_comm, add_comm)
        // already intro()s every quantified variable before inducting, exactly as this requires.
        return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt,
            "induct: goal's own target is still a Pi -- intro every quantified variable before "
            "inducting (induct()'s own generalize() step can otherwise collide with a "
            "not-yet-introduced variable's own body-level reference)"});
    }
    NodeId scrut_ty = g.hyps[hyp_level].type;
    Value scrut_ty_val = eval(scrut_ty, g.ec.values);
    auto* scrut_neutral = std::get_if<VNeutral>(&scrut_ty_val);
    if (!scrut_neutral || scrut_neutral->head.kind != NeutralHead::Kind::GlobalConst) {
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, scrut_ty, std::nullopt,
            "induct: hypothesis's type is not an inductive type application"});
    }
    uint32_t type_global_id = scrut_neutral->head.id;
    std::vector<NodeId> concrete_indices;
    for (auto& frame : scrut_neutral->spine) {
        auto* af = std::get_if<ApplyFrame>(&frame);
        if (!af) {
            return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, scrut_ty, std::nullopt,
                "induct: scrutinee's type has a non-application eliminator frame; cannot enumerate its indices"});
        }
        concrete_indices.push_back(quote(af->arg, g.ec.size()));
    }
    uint32_t arity = static_cast<uint32_t>(concrete_indices.size());

    // ---- Motive: built g.ec.size()-based, mirroring cases()'s own
    // scheme. Two distinct uses, both load-bearing (the "decorative"
    // description that applies to Match::motive_id's role during
    // ordinary reduction does NOT extend to this variable's other use):
    // the sanity check just below that the goal can be abstracted this
    // way at all, and -- unlike cases(), which has no further use for
    // its own motive beyond that check and the (also decorative)
    // Match field -- this motive's shape also underlies branch_expected
    // (each new subgoal's own AGENT-FACING type, which check()/exact()
    // absolutely will inspect once the agent works on it).
    //
    // g.target itself needs the SAME shift_levels_from treatment as
    // motive_body's own generalized_goal_body above, and for the
    // identical reason: g.target can already contain internal bound
    // variables starting at g.ec.size() (typically from a previously
    // normalized stuck Match -- see quote(VNeutral)'s MatchFrame
    // handling), which numerically collide with whatever NEW binders
    // (arity-many indices, plus the scrutinee) this motive itself is
    // about to introduce starting at that SAME g.ec.size(). Skipping
    // this shift here (while only fixing motive_body) was this
    // session's own next bug after the guardedness-collision one --
    // caught by a segfault in conv() on the Zero case's own target
    // once two structurally different things ended up sharing a level.
    std::vector<std::pair<NodeId, uint32_t>> targets;
    for (uint32_t i = 0; i < arity; ++i) targets.push_back({concrete_indices[i], g.ec.size() + i});
    targets.push_back({mk_var(hyp_level), g.ec.size() + arity});

    // reverted_js: which OTHER hypotheses (besides hyp_level itself) must also be generalized,
    // because they genuinely, transitively depend on hyp_level or one of the scrutinee's own
    // indices. The motivating need is cases()'s own (trans's h1:Eq A x y, h2:Eq A y z, both
    // genuinely needing y in scope at once) -- but the COMPUTATION here is deliberately NOT
    // cases()'s own "sweep the whole contiguous block from the earliest dependent hyp to the
    // end" rule. That rule is sound (anything genuinely dependent is, trivially, inside the
    // contiguous range it sweeps) but a severe OVER-approximation for induct() specifically: this
    // session confirmed directly, by a controlled, side-by-side run at an identical 200-simulation
    // budget, that applying it here explodes add_succ_right's own search tree from the baseline's
    // 749 nodes to 4744 -- and FAILS to find the proof the unpatched kernel finds easily -- because
    // induct()'s own common usage (double/nested Nat induction, e.g. add_succ_right itself)
    // routinely inducts on a hyp with later, genuinely UNRELATED hyps still in scope (a second,
    // independent Nat parameter; an IH from an earlier, separate induction), and sweeping all of
    // them regardless, every time, compounds across repeated calls. cases()'s own typical usage
    // (a small, tightly-related group, like trans's own h1/h2/z) never exposes this cost, which is
    // why the contiguous rule was never a problem there.
    //
    // Computed instead via fixpoint: start from hyp_level's own targets (the scrutinee and its
    // indices), then repeatedly scan every OTHER hyp (both before and after hyp_level) for one
    // whose own type genuinely mentions something already marked -- marking it too -- until a full
    // pass finds nothing new. This is the same genuine-dependency test cases()'s own revert_from
    // scan already uses for the "earlier hypothesis" case (detail::generalize(...) != original),
    // just iterated to a fixpoint over the WHOLE hyp list instead of stopping at the first match
    // and sweeping everything after it. Soundness: every hyp this marks is also inside cases()'s
    // own coarser contiguous range (that range is a superset, by construction, of genuine
    // dependents), so this is strictly more precise, never less safe. A dummy replacement level
    // (kMarkerProbe + j, far outside any real context) is used only to make the equality check
    // detect "mk_var(j) appears somewhere" reliably -- replacing mk_var(j) with itself, the actual
    // target-level scheme would use, is indistinguishable from a no-op and would never register as
    // a change.
    std::vector<bool> needs_revert(g.hyps.size(), false);
    {
        constexpr uint32_t kMarkerProbe = 1'000'000'000u;
        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<std::pair<NodeId, uint32_t>> probe_targets = targets;
            for (uint32_t j = 0; j < g.hyps.size(); ++j)
                if (needs_revert[j]) probe_targets.push_back({mk_var(j), kMarkerProbe + j});
            for (uint32_t j = 0; j < g.hyps.size(); ++j) {
                if (j == hyp_level || needs_revert[j]) continue;
                if (detail::generalize(g.hyps[j].type, probe_targets) != g.hyps[j].type) {
                    needs_revert[j] = true;
                    changed = true;
                }
            }
        }
    }
    std::vector<uint32_t> reverted_js;
    for (uint32_t j = 0; j < g.hyps.size(); ++j) if (needs_revert[j]) reverted_js.push_back(j);

    // Unsupported-input guard -- a PRE-EXISTING limitation, not something this function's revert
    // support introduced: found while mutation-testing it, and reproduced on the UNPATCHED baseline
    // with a plain, non-indexed goal  (n:Nat) -> Eq ((\x:Nat. x) n) n  (induct on n, then apply refl
    // aborts the process: Env::operator[] out of bounds). Cause: child goals are built by SYNTACTIC
    // substitution into generalized_goal, whose embedded bound structure (a lambda/pi/match body that
    // refers to its own binder) was shifted by the MOTIVE's binder count (arity+1+reverted), while
    // the child context has a different number of binders (this constructor's fields, IH, reverted
    // copies) -- so every embedded reference lands on the wrong level. cases() is not affected for the
    // target itself (it re-levels via eval()+quote()). A proper fix needs a per-branch re-leveling pass
    // (embed generalized_goal's bound structure in a collision-free band, substitute, then shift down
    // by this branch's own binder count); until then, reject cleanly. Such a goal is a dead end for
    // search rather than a killed worker -- the same philosophy as the is_pi_headed guard above;
    // soundness was never at stake (whole_term_check re-verifies everything).
    //
    // Precise by construction: a target is well-formed in a context of size g.ec.size() and
    // hypothesis j's type in a context of size j, so any Var at or above that base is bound INSIDE
    // the term. An embedded binder that is never referenced has nothing to mis-level and is allowed.
    //
    // CONSEQUENCE for the shifts below: they only ever move Vars >= g.ec.size() of reverted_target,
    // and this guard guarantees there are none -- so for every input induct() accepts, all three
    // shift_levels_from calls are no-ops. (They are kept, and kept correct, so that relaxing this
    // guard later cannot silently reintroduce an undercount.)
    if (detail::has_internal_binder_ref(g.target, g.ec.size())) {
        return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt,
            "induct: the goal's target contains embedded bound structure that refers to its own binder "
            "(e.g. a lambda-redex); induct()'s syntactic generalization cannot re-level it per branch "
            "-- rejected cleanly (see induct()'s own comment; test_indexed_induction.py)"});
    }
    for (uint32_t j : reverted_js) {
        if (detail::has_internal_binder_ref(g.hyps[j].type, j)) {
            return std::unexpected(TypeError{TypeErrorKind::Other, g.hyps[j].type, std::nullopt,
                "induct: a hypothesis that must be generalized contains embedded bound structure that "
                "refers to its own binder (e.g. a dependent Pi); induct()'s syntactic generalization "
                "cannot re-level it -- rejected cleanly (see induct()'s own comment)"});
        }
    }
    NodeId reverted_target = g.target;
    for (auto it = reverted_js.rbegin(); it != reverted_js.rend(); ++it) {
        reverted_target = mk_pi(g.hyps[*it].type, reverted_target);
    }

    // extended_targets: cases()'s own mechanism, applied to THIS scheme's own numbering --
    // continuing right after this scheme's own scrutinee marker (g.ec.size()+arity), exactly
    // mirroring how cases()'s own extended_targets continues after ITS scrutinee marker.
    std::vector<std::pair<NodeId, uint32_t>> extended_targets = targets;
    for (std::size_t ri = 0; ri < reverted_js.size(); ++ri)
        extended_targets.push_back({mk_var(reverted_js[ri]), g.ec.size() + arity + 1 + static_cast<uint32_t>(ri)});

    NodeId shifted_target = detail::shift_levels_from(reverted_target, g.ec.size(), arity + 1 + static_cast<uint32_t>(reverted_js.size()));
    NodeId generalized_goal = detail::generalize(shifted_target, extended_targets);

    std::vector<NodeId> index_types = detail::instantiate_type_telescope(type_global_id, g.ec.size());
    if (index_types.size() != arity) {
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,
            "induct: index count mismatch between the hypothesis's type application and the "
            "inductive's own declared arity"});
    }
    std::vector<NodeId> idx_vars_g;
    for (uint32_t i = 0; i < arity; ++i) idx_vars_g.push_back(mk_var(g.ec.size() + i));
    NodeId scrut_binder_ty_g = apply_all(mk_const(type_global_id), idx_vars_g);
    NodeId motive = mk_lambda(scrut_binder_ty_g, generalized_goal);
    for (auto it = index_types.rbegin(); it != index_types.rend(); ++it) motive = mk_lambda(*it, motive);

    TypeResult motive_ty_r = infer(motive, g.ec, ps.es);
    if (!motive_ty_r) {
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,
            "induct: constructed motive failed to type-check -- the goal likely depends on the "
            "hypothesis in a way that cannot be abstracted cleanly"});
    }

    // ---- The TRUE runtime levels (see this function's own header
    // comment): self, then idx, then the scrutinee, each consuming
    // exactly one level in application order, the same "new binder
    // takes the next level" discipline every construct in this file
    // follows. self_level MUST equal g.ec.size() exactly, not merely "a
    // level higher than anything already used" -- apply()'s own VFix
    // case extends the ACTUAL runtime env at whatever vf.env.size()
    // truly is, which is g.ec.size() (the env captured when this Fix
    // was built), regardless of what level number this construction
    // code assigns self syntactically. Picking any other number here
    // silently breaks eval() at runtime, once self_level and the
    // runtime env size disagree -- an earlier version of this fix tried
    // exactly that (picking self_level strictly above every level
    // already used in the goal) before this session's own test caught
    // apply()'s env mismatch; the actual collision (see just below) is
    // fixed by shifting the GOAL's own pre-existing content instead of
    // moving self.
    //
    // This block is computed HERE, before fn_type, rather than after
    // motive/fn_type (this function's own earlier draft order) because
    // fn_type's own codomain needs generalized_goal_body -- NOT
    // generalized_goal, the g.ec.size()-based, agent-facing one used for
    // motive's own sanity check just above -- and generalized_goal_body
    // is defined in terms of the levels computed in this block. See
    // fn_type's own comment below for why reusing generalized_goal there
    // (this function's own earlier version) was a genuine bug, not
    // merely a missed simplification.
    uint32_t self_level = g.ec.size();
    std::vector<NodeId> idx_vars_body;
    for (uint32_t i = 0; i < arity; ++i) idx_vars_body.push_back(mk_var(self_level + 1 + i));
    uint32_t scrutinee_level = self_level + 1 + arity;
    NodeId scrut_binder_ty_body = apply_all(mk_const(type_global_id), idx_vars_body);
    uint32_t body_base = scrutinee_level + 1;

    // A SEPARATE copy of generalized_goal/motive, built at the TRUE
    // runtime levels rather than g.ec.size()-based ones, for embedding
    // as match_term's own `motive` field specifically. This is not
    // optional bookkeeping: is_locally_structurally_decreasing DOES
    // walk Match::motive_id (statically scanning for stray self-
    // references anywhere in the syntax, unlike eval(Match), which
    // never inspects it) -- so a g.ec.size()-based motive embedded at
    // this depth can numerically COLLIDE with self_level (both equal
    // g.ec.size() whenever there have been no prior Intros), causing
    // the checker to mistake motive's own harmless internal binder for
    // an illegal bare escaping self-reference.
    //
    // A second, independent collision source (found by this session's
    // own end-to-end test, only after the first was fixed): g.target
    // ITSELF can already contain internal bound variables starting at
    // g.ec.size() -- typically from a previously normalized stuck Match
    // reconstruction (see quote(VNeutral)'s MatchFrame handling and
    // requote_captured_syntax), e.g. `add n Zero` getting stuck on a
    // freshly introduced `n` during intro()'s own normalization, whose
    // reconstructed Succ-branch field lands at exactly g.ec.size(). This
    // has nothing to do with the goal being abstracted here and is not
    // itself a self-reference, but numerically collides with self_level
    // regardless. Shifting g.target's own >= g.ec.size() levels
    // (necessarily all BOUND, since g.target's genuine free variables
    // are by definition < g.ec.size()) clears them out of self/idx/
    // scrutinee's own range before generalize() re-abstracts the
    // (untouched, < g.ec.size()) free occurrences of the hypothesis/
    // indices.
    //
    // The shift amount is `arity + 2`, NOT `1 + arity` (this function's
    // own earlier version, which undercounts by exactly one): the TRUE
    // runtime range that needs clearing is [g.ec.size(), scrutinee_level]
    // inclusive -- self_level occupies g.ec.size() itself, then `arity`
    // index binders, then the scrutinee at self_level+1+arity -- so
    // g.target's own content must land STRICTLY ABOVE scrutinee_level,
    // i.e. shifted by (scrutinee_level+1)-g.ec.size() = arity+2. Using
    // arity+1 instead lands g.target's own internal content (e.g. that
    // reconstructed Succ-branch field above) AT exactly scrutinee_level
    // itself, colliding with the newly-substituted scrutinee reference
    // one line below -- both end up numerically the same level, silently
    // merged into one ambiguous variable. Found directly by this
    // session's own end-to-end test: the WHOLE-TERM check() of the
    // final, tactic-assembled proof (not any single per-goal tactic
    // step, which never exercises this path) compared the motive's own
    // embedded reconstruction against a freshly-evaluated one and found
    // its recursive field silently resolving to the scrutinee itself
    // instead of the field -- the exact same failure mode as the
    // branch_expected/ih_ty undercount bug fixed earlier, just one
    // binder further out (self_level itself, which sits between
    // g.ec.size() and idx/scrutinee only in THIS function's own true-
    // runtime scheme, not in cases()'s, which is where the original
    // `1 + arity` pattern was copied from).
    std::vector<std::pair<NodeId, uint32_t>> targets_body;
    for (uint32_t i = 0; i < arity; ++i) targets_body.push_back({concrete_indices[i], self_level + 1 + i});
    targets_body.push_back({mk_var(hyp_level), scrutinee_level});
    // extended_targets_body: same reverted_js, this scheme's own numbering -- continuing right
    // after scrutinee_level (= body_base - 1, since body_base is defined as scrutinee_level+1
    // just above), i.e. starting exactly at body_base itself.
    std::vector<std::pair<NodeId, uint32_t>> extended_targets_body = targets_body;
    for (std::size_t ri = 0; ri < reverted_js.size(); ++ri)
        extended_targets_body.push_back({mk_var(reverted_js[ri]), body_base + static_cast<uint32_t>(ri)});
    NodeId shifted_target_body = detail::shift_levels_from(reverted_target, g.ec.size(), arity + 2 + static_cast<uint32_t>(reverted_js.size()));
    NodeId generalized_goal_body = detail::generalize(shifted_target_body, extended_targets_body);

    // ---- motive_body's OWN, SEPARATE level scheme (Option B, this
    // session's continued debugging) -- distinct from fn_type's above.
    //
    // fn_type and motive_body/generalized_goal_body used to share ONE
    // construction (self_level+1+arity for the scrutinee). That is
    // correct for fn_type, where "n" genuinely, directly needs to
    // appear at exactly that level (fn_type's own Pi-chain is never
    // itself "applied" -- it's validated standalone, then reused
    // as-is via infer(Fix)'s own re-anchoring). It is WRONG for
    // motive_body specifically, because motive_body, once embedded as
    // a Match node's own `motive` field, IS genuinely applied later --
    // by check(Match)'s own explicit-motive path (`apply_all(motive,
    // indices); mk_app(applied, scrutinee)`), which evaluates via
    // ordinary function-application semantics: `eval(body, env.extend(
    // arg))`, inserting each argument at the CURRENT env.size() at the
    // point of application -- NOT at whatever absolute level the
    // motive's own syntax happens to already use for "n". By the time
    // check(Match) actually runs (after check(Lambda) has peeled
    // fn_type's own full idx+scrutinee Pi-chain, including the layer
    // that binds "n" itself), the ambient context size is exactly
    // body_base (= scrutinee_level+1, already computed above) -- NOT
    // scrutinee_level. Building motive_body with its scrutinee
    // parameter AT scrutinee_level (fn_type's own level, matching "n"
    // directly rather than a genuinely fresh parameter) meant that
    // applying motive_body to the actual scrutinee was a real,
    // ordinary Lambda-application that inserted the scrutinee's value
    // one level ABOVE where "n" already lived (at body_base, not
    // scrutinee_level) -- landing exactly on top of g.target's own,
    // separate, pre-existing embedded content (e.g. an already-stuck
    // Match's own reconstructed Succ-branch field), which the OLD
    // shift (arity+2, clearing only up through scrutinee_level) placed
    // at precisely that same, colliding position. Confirmed directly
    // by this session's own instrumentation: motive's own RAW syntax
    // printed with an unused, vestigial Lambda parameter -- the body
    // referenced "n" as a free variable at its own, already-existing
    // level instead of the Lambda's own, just-introduced one -- and
    // evaluating (genuinely applying) that motive silently overwrote
    // g.target's own, unrelated field reference instead of doing
    // nothing, exactly the observed v4-becomes-v3 collision.
    //
    // The fix: motive_body's OWN scrutinee parameter belongs at
    // body_base (matching EXACTLY where check(Match)'s own first
    // application will actually insert an argument), with idx
    // parameters (if any) sequentially before it, matching the
    // sequential env.size()-then-+1-then-+2... positions ordinary
    // repeated application produces (apply_all applies indices in
    // order, each via its own env.extend(), before the final scrutinee
    // application) -- and g.target's own pre-existing content shifted
    // one level further still (arity+3, not arity+2) to land strictly
    // above THIS scrutinee level rather than fn_type's.
    std::vector<NodeId> idx_vars_motive;
    for (uint32_t i = 0; i < arity; ++i) idx_vars_motive.push_back(mk_var(body_base + i));
    uint32_t scrutinee_level_motive = body_base + arity;
    NodeId scrut_binder_ty_motive = apply_all(mk_const(type_global_id), idx_vars_motive);
    std::vector<std::pair<NodeId, uint32_t>> targets_motive;
    for (uint32_t i = 0; i < arity; ++i) targets_motive.push_back({concrete_indices[i], body_base + i});
    targets_motive.push_back({mk_var(hyp_level), scrutinee_level_motive});
    // extended_targets_motive: continuing right after scrutinee_level_motive, this scheme's own
    // scrutinee marker.
    //
    // Shift amount, DERIVED from where this scheme's binders actually land rather than a closed-form
    // guess: every level from g.ec.size() up to and including scrutinee_level_motive is a binder
    // here -- self, the Fix's idx parameters and its scrutinee (up to body_base), then motive_body's
    // OWN idx lambdas and scrutinee -- so there are scrutinee_level_motive + 1 - g.ec.size() ==
    // 2*arity+3 of them, plus one slot per reverted hyp (the markers appended just below).
    //
    // This was originally the literal `arity + 3`: equal at arity=0 (a no-op for every non-indexed
    // family) but short by `arity` otherwise -- the second arity-sized group (motive_body's own index
    // lambdas) was left out. The arithmetic was simply wrong for arity>0.
    //
    // HONESTY NOTE: an earlier version of this comment claimed le_trans "confirmed" the change was
    // necessary. That was false: le_trans verifies with either value (mutation-tested), and the claim
    // was made while the real causes of its failure were still unfound. What IS true: this shift only
    // moves Vars >= g.ec.size() of reverted_target (bound structure embedded in the target / reverted
    // hyp types), and the has_internal_binder_ref guard above rejects every such input -- so for all
    // inputs induct() accepts it is a no-op, and the mutation `arity+3` is an EQUIVALENT mutant
    // (unobservable by any test, by construction). The correct amount is kept so that relaxing the
    // guard later cannot silently reintroduce the undercount.
    const uint32_t motive_scheme_binders = scrutinee_level_motive + 1 - g.ec.size();
    std::vector<std::pair<NodeId, uint32_t>> extended_targets_motive = targets_motive;
    for (std::size_t ri = 0; ri < reverted_js.size(); ++ri)
        extended_targets_motive.push_back({mk_var(reverted_js[ri]), scrutinee_level_motive + 1 + static_cast<uint32_t>(ri)});
    NodeId shifted_target_motive = detail::shift_levels_from(reverted_target, g.ec.size(), motive_scheme_binders + static_cast<uint32_t>(reverted_js.size()));
    NodeId generalized_goal_motive = detail::generalize(shifted_target_motive, extended_targets_motive);

    // ---- fn_type: the Fix's own flat Pi-chain shape, mirroring
    // add_zero_right's own hand-written azr_ty exactly (there:
    // `Pi n:Nat. Eq Nat (add n Zero) n`).
    //
    // Built from generalized_goal/scrut_binder_ty_g -- the g.ec.size()-
    // based, AGENT-FACING versions, the SAME ones used for motive's own
    // sanity check above -- NOT generalized_goal_body/scrut_binder_ty_body
    // (the self_level-aware, TRUE-runtime versions used for motive_body,
    // embedded within match_term itself). fn_type is validated standalone
    // via infer(f.fn_type, ec, es) with ec UNEXTENDED (size g.ec.size()) --
    // infer(Pi) naturally extends ec by one per binder in fn_type's own
    // Pi chain, landing its scrutinee at g.ec.size()+arity, exactly
    // generalized_goal's own scheme. Using generalized_goal_body here
    // instead (this function's own earlier, now-reverted fix) broke this
    // very validation: generalized_goal_body's own scrutinee reference
    // assumes an EXTRA level of offset (self_level+1+arity) that only
    // exists once self has ALREADY been added to the context -- which
    // hasn't happened yet at the point fn_type is validated standalone.
    //
    // The self-offset this fn_type needs to account for when later used
    // as the expected type for checking body (infer(Fix)'s own
    // `Γ,self:fn_type ⊢ body:fn_type` rule, checked in a context WITH
    // self already added) is handled there instead, correctly and
    // generally, by re-anchoring fn_type via quote(eval(f.fn_type,
    // ec.values), ec2.size()) -- see infer(Fix)'s own, detailed comment
    // for why reusing the SAME raw NodeId across two different-sized
    // contexts (this function's own first fix attempt, and this file's
    // OWN pre-existing infer(Fix) implementation before that) is a
    // genuine, general bug, and why re-deriving the value via the
    // existing quote()/eval() machinery -- rather than trying to pick
    // ONE "correct" absolute level scheme for fn_type that magically
    // works for both its own standalone validation AND its later reuse
    // as an expected type -- is the actual, principled fix. No single
    // level scheme for fn_type CAN be correct for both purposes
    // simultaneously, since the two call sites extend from genuinely
    // different context sizes; the fix belongs in infer(Fix), not here.
    // FIX (this session's continued debugging): fn_type is now built
    // DIRECTLY from generalized_goal_body/scrut_binder_ty_body -- the
    // SAME, already-verified self-aware construction motive_body uses --
    // rather than the separate generalized_goal/scrut_binder_ty_g
    // (g.ec.size()-based) scheme this function used previously. That
    // separate scheme required infer(Fix) to re-anchor fn_type via
    // quote(eval(...), ec2.size()) before reuse as the expected type for
    // checking body -- individually correct in isolation, but it
    // compounds with check_match_branches' OWN later shift (when this
    // same Fix, re-embedded deeper via quote(VFix) during proof
    // extraction, gets independently re-validated by a whole-term
    // check(), and its own Succ branch is checked): both shifts are
    // individually necessary for their own new binder, but together they
    // shift g.target's own pre-existing embedded content (e.g. an
    // already-stuck Match reconstruction within g.target itself) one
    // time more than generalized_goal_body's own, single arity+2 shift
    // does -- producing two DIFFERENT absolute levels for what should be
    // the same reference, confirmed directly by this session's own
    // instrumentation (print(quote(eval(...))) on both sides of the
    // failing comparison showed an extra, genuinely wrong level, not
    // merely a differently-anchored but equivalent one). Building
    // fn_type from the self-aware scheme directly, once, avoids the
    // later re-anchoring step (and its compounding) entirely: fn_type is
    // now correct relative to ec2 (self already included) from
    // construction, matching where it is actually reused for the body
    // check without any further transformation.
    NodeId fn_type = generalized_goal_body;
    fn_type = mk_pi(scrut_binder_ty_body, fn_type);
    for (auto it = index_types.rbegin(); it != index_types.rend(); ++it) fn_type = mk_pi(*it, fn_type);
    uint32_t decreasing_arg = arity;

    NodeId motive_body = mk_lambda(scrut_binder_ty_motive, generalized_goal_motive);
    for (auto it = index_types.rbegin(); it != index_types.rend(); ++it) motive_body = mk_lambda(*it, motive_body);

    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(type_global_id);
    std::vector<NodeId> branches;
    std::vector<Goal> new_goals;

    for (uint32_t ctor_gid : ctor_gids) {
        // The REAL telescope: where this constructor's own fields
        // actually live once the branch is embedded inside the
        // Fix/body/Match nest (see body_base's own definition above).
        detail::InstantiatedCtor inst = detail::instantiate_ctor_telescope(ctor_gid, body_base);
        uint32_t ctor_arity = static_cast<uint32_t>(inst.arg_types.size());
        std::vector<bool> rec_mask = recursive_field_mask(type_global_id, ctor_gid);

        // The AGENT-FACING telescope: g.ec.size()-based, exactly
        // mirroring cases()'s own scheme -- see this function's own
        // header comment for why these are computed independently
        // rather than shifted from one to the other.
        detail::InstantiatedCtor inst_agent = detail::instantiate_ctor_telescope(ctor_gid, g.ec.size());

        // Index forcing (design spec, Phase B/Step 2, now extended to
        // induct()) -- AGENT-FACING side only. Computed against
        // concrete_indices (the scrutinee's own actual index), exactly
        // mirroring cases()'s own Step 2: the agent genuinely should see
        // the forced, specific type for THIS hypothesis's own fields.
        CtorIndexForcing forcing_agent = unify_ctor_pattern_indices(inst_agent, concrete_indices, g.ec.size(), ctor_arity, g.ec.values);
        if (!forcing_agent.ok) {
            return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, forcing_agent.error});
        }
        std::vector<NodeId> arg_types_agent = apply_ctor_index_forcing(inst_agent.arg_types, forcing_agent);

        // The TRUE-RUNTIME (Fix body) side is DELIBERATELY NEVER forced
        // against concrete_indices -- this was a real bug in an earlier
        // version of this function, found by tracing Mode 1 (a genuinely
        // concrete index, e.g. Vec (Succ Zero)) through by hand rather
        // than by a failing test (Mode 2's own bare-variable index made
        // this a silent no-op there, which is exactly why the earlier,
        // wrong version passed every existing test). The Fix built here
        // is a fully generic, reusable function -- Pi-quantified over
        // EVERY possible index, applied to this call's own concrete
        // index only at the very end (see `proof`'s own construction,
        // far below) -- and its own internal branches must stay valid
        // for every index, including whatever smaller index a recursive
        // self-call within a DIFFERENT top-level invocation might use.
        // Forcing this constructor's own field types against the OUTER,
        // top-level concrete_indices would incorrectly specialize (or,
        // for a constructor whose pattern conflicts with that specific
        // outer index -- e.g. VNil when the outer index is Succ Zero --
        // wrongly mark as REFUTED) a branch that is genuinely reachable
        // within the Fix's own generic body; being unreachable from
        // THIS SPECIFIC top-level call is a fact about the caller, not
        // about the Fix's own internal typing, and conflating the two
        // would corrupt the Fix for every index other than this one.
        // Using idx_vars_body (the Fix's own, ALWAYS-generic Pi
        // parameters -- never a concrete value, by construction, for
        // ANY mode) instead of concrete_indices makes this a PROVABLE
        // no-op, not merely one that happens to hold for Mode 2:
        // unify_ctor_pattern_indices always declines gracefully against
        // a bare variable (Gap A's own fix), so forcing_body always
        // succeeds vacuously and arg_types_body always equals
        // inst.arg_types directly, unconditionally.
        CtorIndexForcing forcing_body = unify_ctor_pattern_indices(inst, idx_vars_body, body_base, ctor_arity, g.ec.values);
        if (!forcing_body.ok) {
            return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, forcing_body.error});
        }
        std::vector<NodeId> arg_types_body = apply_ctor_index_forcing(inst.arg_types, forcing_body);
        // The true-runtime scheme's FORCED result indices, computed once and used consistently
        // everywhere this scheme substitutes a constructor's own result pattern (subgoal_applied's
        // index positions, reverted_body_types): exactly the forcing arg_types_body itself was
        // built with, so no site mixes a field's forced identity (e.g. le_S's recursive field
        // `Le idx0 m'`) with that same field's unforced one (`n'`).
        std::vector<NodeId> forced_body_result_indices =
            apply_ctor_index_forcing_to_result_indices(inst.result_indices, forcing_body);

        // branch_ec: g's own hypotheses, PLUS this constructor's own
        // fields, PLUS -- for each field recursive_field_mask marks as
        // a direct recursive occurrence -- an extra induction-hypothesis
        // hypothesis (assuming the recursive field targets the SAME
        // indices as the original scrutinee -- exact for non-indexed
        // recursion, matching this session's own add_zero_right target;
        // see this function's own header comment for the general-
        // indexed-type caveat).
        //
        // ih_ty and branch_expected are both built by substituting
        // generalized_goal's own markers (see this function's own doc
        // comment above generalized_goal's construction) directly --
        // purely syntactic substitute_var, matching the ORIGINAL
        // construction's own mechanism exactly, just applied to
        // generalized_goal (which already, correctly generalizes over a
        // COMPOUND concrete index, e.g. "Succ Zero" appearing literally
        // in g.target -- Mode 1) instead of raw g.target (which never
        // did, only ever replacing the bare scrutinee occurrence -- the
        // actual gap this fix closes). Two earlier attempts, each
        // confirmed wrong by a real regression rather than assumed:
        // raw motive-reapplication (shift_levels_from + bare mk_app,
        // never reduced) left g.ec-relative absolute levels embedded
        // inside subgoal_ty, which infer(App)'s own Meta-headed
        // shift_all_levels treatment (correctly assuming a "fresh
        // 0-based scope") then corrupted -- an Env::operator[]
        // out-of-bounds crash on add_zero_right. Genuinely eval()+
        // quote()-ing the result fixed THAT, but eval() unconditionally
        // delta-reduces every defined global it touches, including
        // "add" itself (embedded in the very goal being generalized
        // here) -- unfolding it into internal Match/Fix structure the
        // agent's own, separately-built proof (which references "add"
        // as a bare, un-unfolded Const) does not share, breaking
        // tactic-level exact() via hard NodeId inequality on two
        // differently-derived stuck reconstructions ("stuck Match
        // frames differ"). Pure substitute_var avoids both: it never
        // evaluates anything, so "add" (or any other global the goal
        // happens to mention) is carried along completely unchanged,
        // exactly as the original, known-good construction already did.
        ElabCtx branch_ec = g.ec;
        std::vector<Hypothesis> branch_hyps = g.hyps;
        NodeId agent_ctor_app = mk_const(ctor_gid);
        for (std::size_t fi = 0; fi < arg_types_agent.size(); ++fi) {
            uint32_t field_level = branch_ec.size();
            branch_ec = branch_ec.extend(arg_types_agent[fi]);
            branch_hyps.push_back(Hypothesis{"", arg_types_agent[fi]});
            agent_ctor_app = mk_app(agent_ctor_app, mk_var(field_level));
            if (fi < rec_mask.size() && rec_mask[fi]) {
                // ih_ty: reuse generalized_goal (computed once, earlier
                // in this function, for motive -- already, correctly
                // generalized over BOTH the scrutinee occurrence and the
                // concrete index, via the SAME generalize() call, marked
                // at levels g.ec.size()+0..+arity-1 for the indices and
                // g.ec.size()+arity for the scrutinee), substituting
                // those markers back with this recursive field's own
                // recovered index arguments and its own value --
                // PURELY SYNTACTIC substitute_var, never genuinely
                // eval()ing anything.
                //
                // Two earlier attempts at this exact fix both failed,
                // for two DIFFERENT reasons, each confirmed by a real
                // regression rather than assumed: (1) raw motive-
                // reapplication (shift_levels_from then bare mk_app,
                // never reduced) left ih_ty still internally referencing
                // g.ec-relative absolute levels, which infer(App)'s own
                // Meta-headed re-anchoring (shift_all_levels) blindly
                // shifts, treating subgoal_ty as a "fresh 0-based scope"
                // it no longer actually was -- confirmed by an
                // Env::operator[] out-of-bounds crash. (2) Genuinely
                // eval()+quote()-ing the result to fix that DID make it
                // 0-based-safe, but eval() unconditionally delta-reduces
                // every defined global it touches -- including "add"
                // itself, embedded in the goal being generalized here --
                // unfolding it into its own internal Match/Fix structure
                // that the AGENT's own, separately-built proof (which
                // references "add" as a bare, un-unfolded Const) does
                // not share, breaking tactic-level exact() via hard
                // NodeId inequality on the two differently-derived stuck
                // reconstructions -- confirmed by "stuck Match frames
                // differ". This construction avoids both: substitute_var
                // never evaluates anything, so "add" (or any other
                // global the goal happens to mention) is carried along
                // completely unchanged, exactly as the original,
                // known-good construction already did -- the only
                // difference is substituting into generalized_goal
                // (which already handles a compound, Mode-1-shaped
                // concrete index) rather than raw g.target (which never
                // did, the actual gap this whole fix targets).
                //
                // Recovered by decomposing the recursive field's own
                // (forced) type -- arg_types_agent[fi] in this scheme --
                // which is guaranteed to have exactly `arity` arguments
                // by recursive_field_mask's own definition. Proven
                // no-op for arity=0 (every already-verified non-indexed
                // case, add_zero_right included): the substitution loop
                // below never executes, and generalized_goal's own
                // scrutinee-marker substitution is identical in effect
                // to the original construction's direct substitute_var
                // into g.target.
                detail::AppSpine field_spine = detail::decompose_app_spine(arg_types_agent[fi]);
                std::vector<NodeId> ih_idx_args(field_spine.args.begin(),
                    field_spine.args.begin() + std::min(field_spine.args.size(), static_cast<std::size_t>(arity)));
                std::unordered_map<uint32_t, NodeId> ih_replacements;
                for (uint32_t j = 0; j < ih_idx_args.size(); ++j) ih_replacements[g.ec.size() + j] = ih_idx_args[j];
                ih_replacements[g.ec.size() + arity] = mk_var(field_level);
                NodeId ih_ty = detail::substitute_vars(generalized_goal, ih_replacements);
                branch_ec = branch_ec.extend(ih_ty);
                branch_hyps.push_back(Hypothesis{"", ih_ty});
            }
        }

        // branch_expected: the SAME mechanism as ih_ty above, substituting
        // generalized_goal's own markers with this constructor's own
        // full pattern (inst_agent.result_indices) and its own
        // application (agent_ctor_app) instead. Simultaneous
        // substitution here too, for the identical reason: this
        // constructor's own recovered pattern values can, in general,
        // numerically coincide with the scrutinee marker's own target
        // level, the same collision ih_ty's own construction hit.
        // Forced result indices, NOT inst_agent.result_indices directly: exactly cases()'s own
        // and check_match_branches's own, already-documented fix for this bug class (see either
        // one's comment: "two syntactically distinct NodeIds meant to be equal"). arg_types_agent
        // above is already forced -- e.g. le_S's own recursive field comes out as (Le b m'), with
        // le_S's first result index (a bare field variable, n') forced to the concrete b -- so the
        // motive's own substitution must use that same, forced identity (b), not the field's own
        // unforced one (n'), or this branch's target / reverted hyps would mention n' exactly where
        // everything else (its own fields, and check_match_branches's own expected type for it)
        // mentions b: two NodeIds meant to be equal, with nothing inside the branch that equates
        // them (n' is a dummy field; the identification n'=b exists ONLY as this syntactic
        // forcing). Never ported to induct() before this session because no induct() target
        // before Le had a constructor whose result index is a bare field variable AND a goal
        // that mentions that index -- found by le_trans itself, confirmed by an unambiguous raw
        // dump of the failing whole_term_check comparison (Le a n' vs Le a b).
        std::vector<NodeId> forced_agent_result_indices =
            apply_ctor_index_forcing_to_result_indices(inst_agent.result_indices, forcing_agent);
        std::unordered_map<uint32_t, NodeId> branch_replacements;
        for (uint32_t j = 0; j < arity; ++j) branch_replacements[g.ec.size() + j] = forced_agent_result_indices[j];
        branch_replacements[g.ec.size() + arity] = agent_ctor_app;
        NodeId branch_expected = detail::substitute_vars(generalized_goal, branch_replacements);

        // reverted_dom_types: NOT cases()'s own eval()/quote()-based peeling -- that mechanism
        // specifically relies on branch_expected being a GENUINE, reduced application of
        // motive_original (apply_all(motive, indices), then mk_app(..., ctor_app)), so eval()
        // against a per-branch env correctly substitutes motive's own Lambda parameters via
        // ordinary, level-agnostic application semantics. induct()'s own branch_expected is
        // built differently: substitute_vars(generalized_goal, branch_replacements), a PURELY
        // SYNTACTIC replacement that leaves every OTHER marker (in particular each reverted
        // hyp's own shared-scheme marker, g.ec.size()+arity+1+ri, used nowhere else but within
        // this one Pi-chain) baked in as a literal mk_var reference -- unrelated to, and not
        // reliably within bounds of, any per-branch env built from arg_types_agent.size() (which
        // can differ from arity+1 for any constructor, confirmed directly this session: an
        // eval()/quote() port of cases()'s own mechanism, tried first, crashed with
        // Env::operator[] out of bounds on ordinary double-Nat-induction search, traced via gdb
        // directly to this exact mismatch, not assumed). The fix: peel PURELY SYNTACTICALLY
        // instead, substitute_var-ing each layer's own shared-scheme marker with this branch's
        // own real, final mk_var(target_level) one layer at a time -- mirroring exactly how
        // ih_ty/branch_expected themselves are already built elsewhere in this same function,
        // rather than cases()'s own, structurally different mechanism. A later reverted hyp's
        // own domain referencing an earlier one's shared-scheme marker is handled correctly by
        // construction: substitute_var on cod_id (the remaining Pi-chain) happens before the
        // NEXT layer's own domain is ever extracted from it, so that marker is already replaced
        // by the time it matters.
        std::vector<NodeId> reverted_dom_types;
        if (!reverted_js.empty()) {
            // base_level: branch_ec's own CURRENT size -- NOT g.ec.size()+arg_types_agent.size()
            // (this function's own earlier version, confirmed wrong by a real crash): branch_ec,
            // by this point, has already been extended through the arg_types_agent loop above,
            // which pushes ONE MORE hyp than arg_types_agent.size() alone accounts for whenever
            // a field is recursive (the separately-pushed IH) -- branch_ec.size() reflects the
            // true count directly, correct for every constructor regardless of how many of its
            // fields are recursive, rather than needing a second, easy-to-forget adjustment.
            uint32_t base_level = branch_ec.size();
            NodeId remaining = branch_expected;
            for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
                auto pi_opt = inspect(remaining, [](Pi p) -> std::optional<Pi> { return p; },
                                                   [](auto) -> std::optional<Pi> { return std::nullopt; });
                if (!pi_opt) {
                    return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,
                        "induct: internal error -- expected a Pi layer for a reverted hypothesis but found none"});
                }
                uint32_t target_level = base_level + static_cast<uint32_t>(ri);
                uint32_t shared_marker = g.ec.size() + arity + 1 + static_cast<uint32_t>(ri);
                reverted_dom_types.push_back(pi_opt->dom_id);
                remaining = detail::substitute_var(pi_opt->cod_id, shared_marker, mk_var(target_level));
            }
            branch_expected = remaining;
            // branch_ec/branch_hyps: already extended through arg_types_agent above (the
            // constructor's own fields and any IH); now append the freshly re-specialized
            // reverted hyps, in order, at their own real, final positions, mirroring exactly how
            // cases()'s own branch_ec construction appends its own reverted_dom_types after its
            // own arg_types.
            for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
                branch_ec = branch_ec.extend(reverted_dom_types[ri]);
                branch_hyps.push_back(Hypothesis{g.hyps[reverted_js[ri]].name, reverted_dom_types[ri]});
            }
        }
        // subgoal_meta's own DECLARED type must be a genuine Pi-chain
        // over branch_ec's own full telescope (mirroring
        // close_over_context's own Lambda-chain construction, but for a
        // TYPE rather than a term), matching how many arguments
        // subgoal_applied actually applies this meta to below -- NOT
        // the bare branch_expected goal type on its own. infer(Meta m)
        // always returns exactly ctx().meta_ctx[m.meta_id].type
        // verbatim, regardless of whether the meta is solved or what
        // shape its eventual solution has -- so infer(App(Meta(...),
        // arg)), reached whenever anything (e.g. a whole-term check() of
        // the extracted proof, as opposed to any single per-goal tactic
        // check, which only ever checks the SOLUTION supplied for a
        // meta, never the meta's own bare application syntax) tries to
        // type-check subgoal_applied's own multi-argument application
        // form, needs the meta's declared type to actually BE a
        // multi-argument function type to peel apart one layer per
        // argument -- exactly as any ordinary function application
        // would require. Leaving it as the bare branch_expected (this
        // function's own earlier version) is not merely imprecise: it
        // is a genuine type mismatch between how the meta is USED
        // (applied to branch_ec.size() arguments) and how it is
        // DECLARED (a zero-argument type), found directly by this
        // session's own whole-term check() test with "applying a term
        // that is not a function" once the earlier bugs blocking that
        // far were fixed.
        //
        // KNOWN, OPEN GAP (indexed families only, e.g. Vec, not Nat/
        // Bool/List): hyp_level's own domain here is
        // branch_ec.types[hyp_level] verbatim -- the scrutinee's own
        // ORIGINAL, generic type (e.g. "Vec (Succ Zero)" for a
        // concrete-index induction, or "Vec n" for a generic one),
        // never reflecting this specific constructor's own concrete
        // result type (e.g. "Vec Zero" for VNil). subgoal_applied
        // (below) supplies true_ctor_app -- this constructor's own
        // concrete value -- at that same position, which for an
        // indexed family is a genuine type mismatch found directly by
        // a real whole-term check() failure ("different constructors":
        // Zero vs Succ Zero) on a concrete-index (Mode 1) induction --
        // the first construction whose own indexed scrutinee has
        // nothing else in the same application chain to
        // coincidentally correct this via ordinary Pi-substitution the
        // way Mode 2's own, separate index position already does. Two
        // fixes were tried this session and both reverted after
        // confirming a real regression, not assumed: (1) moving
        // hyp_level's own slot to the end of this Pi-chain with a
        // directly-computed forced domain broke exact()'s own
        // close_over_context(term, g.ec), which unconditionally wraps a
        // goal's solution as a Lambda-chain over g.ec's own NATURAL,
        // UNREORDERED order (confirmed by reading its implementation) --
        // a runtime crash once the reordered solution got applied by
        // subgoal_applied in the new sequence. (2) Replacing this
        // domain with apply_all(type_global_id, fresh elaboration
        // metas) -- sidestepping the forward-reference problem (a
        // Pi-bound parameter, unlike a meta, cannot depend on a
        // later-bound field like VCons's own n' without moving after
        // it) -- broke on the OPPOSITE constraint: this codebase's own
        // metas are solved via Miller-pattern unification, which
        // rejects a solution referencing a variable outside the meta's
        // own declared pattern spine (VCons's own n' does not qualify),
        // failing with "solution depends on a variable outside the
        // metavariable's pattern spine" -- confirmed to ALSO regress
        // Mode 2 (previously passing), not just fail to fix Mode 1.
        // Left for dedicated, unhurried follow-up: a genuine fix likely
        // needs either a non-pattern (postponed/deferred) unification
        // mechanism this codebase does not yet have, or restructuring
        // how the Fix's own body threads the scrutinee through
        // entirely, not another variation on hyp_level's own domain
        // within this fixed telescope.
        NodeId subgoal_ty = branch_expected;
        for (uint32_t i = branch_ec.size(); i-- > 0;) subgoal_ty = mk_pi(branch_ec.types[i], subgoal_ty);
        uint32_t subgoal_meta = ctx().meta_ctx.fresh(subgoal_ty, branch_ec.size());

        // The actual branch, built DIRECTLY at its true runtime
        // position (body_base-based) rather than via the
        // ref_meta/close_over_context bridge cases() uses for its own
        // (IH-free) branches -- that bridge re-abstracts a branch's
        // solution as a standalone, closed Lambda-chain over
        // branch_ec's own g.ec.size()-based levels, which would
        // require self_level (a level BELOW g.ec.size()+ctor_arity,
        // but semantically a COMPLETELY DIFFERENT binder from
        // whatever branch_ec's own numbering assigns to that same
        // number) to be reachable from inside that re-abstracted
        // frame -- it is not: the bridge's own re-application, by
        // design, only ever threads g.ec's own levels and the
        // constructor's own fields through, never the Fix's self
        // binder, since cases() (which this bridge is copied from) has
        // no Fix at all and so never needed self to survive it. Built
        // directly here instead, `self_level` (g.ec.size(), the Fix's
        // own binder, one level below where this branch's own fields
        // begin) remains reachable, and the induction hypothesis is
        // computed INLINE as `self` applied to the recursive field --
        // an ordinary subterm within the branch's own body, exactly
        // mirroring itp_prelude_demo.cpp's own hand-written
        // succ_branch (`mk_app(azr_ref, mk_var(1))`), not an extra
        // curried branch parameter (Match's branches are curried over
        // EXACTLY the constructor's own field count, per eval(Match)'s
        // fixed-arity application loop -- the induction hypothesis
        // cannot be an additional Lambda layer there).
        NodeId subgoal_applied = mk_meta(subgoal_meta);
        // Apply to g.ec's own hypotheses at their TRUE runtime levels
        // (0..g.ec.size()-1, same as the agent-facing ones, since g.ec
        // itself sits at the very bottom of the Fix/body/Match nest --
        // see body_base's own definition above) -- EXCEPT at hyp_level
        // itself, the hypothesis actually being inducted on: the
        // branch's own solution (closed over branch_ec, in which THAT
        // position holds the constructor application, not the original
        // bare hypothesis -- see branch_expected's own construction,
        // which substitutes hyp_level's occurrence for exactly this
        // reason) expects the concrete constructor application there,
        // not g.ec's own separate copy of the hypothesis. Passing
        // mk_var(hyp_level) here instead (this function's own earlier
        // draft did exactly that) applies the branch's solution to the
        // WRONG value at that position: not this branch's own
        // constructor application, but whatever the Fix's outer wrapper
        // (close_over_context(proof, g.ec) at this function's own
        // end) happened to bind at that same level -- silently valid
        // syntax that computes a nonsensical result, only surfacing as
        // a crash once the recursive call actually dispatches on a
        // wrong-shaped value.
        NodeId true_ctor_app = mk_const(ctor_gid);
        for (uint32_t ai = 0; ai < ctor_arity; ++ai) true_ctor_app = mk_app(true_ctor_app, mk_var(body_base + ai));
        for (uint32_t i = 0; i < g.ec.size(); ++i) {
            // subgoal_ty's own domain at this position (see
            // hyp_level_domain_for_subgoal above for hyp_level itself)
            // genuinely depends on whatever value gets supplied for the
            // index position(s) earlier in this SAME application chain
            // -- Pi-substitution resolves this automatically, correctly,
            // as long as the RIGHT value is supplied at the index's own
            // position, not the generic g.ec variable. Uses
            // inst.result_indices (the body_base scheme) to match the
            // SAME ambient context every other reference in this loop
            // and the ones after it already use -- not inst_agent,
            // which would be inconsistent here (this whole expression
            // lives inside the Fix's own body, not the agent-facing
            // frame). Proven no-op for a non-indexed family (Nat/Bool/
            // List): concrete_indices is empty (arity=0), so this inner
            // search never finds anything to substitute, leaving every
            // position exactly as before.
            NodeId supplied = mk_var(i);
            if (i != hyp_level) {
                bool replaced = false;
                for (uint32_t j = 0; j < arity; ++j) {
                    bool is_bare_var = false; uint32_t idx_level = 0;
                    inspect(concrete_indices[j], [&](Var v) { is_bare_var = true; idx_level = v.level; }, [](auto) {});
                    // FORCED result index (not inst.result_indices directly) -- see
                    // forced_body_result_indices's own comment above.
                    if (is_bare_var && idx_level == i) { supplied = forced_body_result_indices[j]; replaced = true; break; }
                }
                // A REVERTED position (a hyp whose type genuinely depends on the generalized
                // indices/scrutinee) must likewise receive THIS branch's own specialized value --
                // the fresh local parameter bound for it below (revert level
                // body_base+ctor_arity+ri) -- not the outer, original hyp. The outer hyp's type
                // mentions the specific outer index (le_trans: h1 : Le a b), valid only where the
                // Fix's own generic index parameter happens to equal it; this expression lives
                // inside the Fix's GENERIC body, which must stay valid for every index (including
                // the smaller one a recursive call passes). Same principle, and same Pi-
                // substitution mechanism, as the bare-index-variable and scrutinee positions
                // already handled here: supplying the right value at THIS position makes every
                // downstream domain come out correct automatically. Found by le_trans itself: the
                // agent's own (perfectly valid, agent-level) IH(h1_orig) became a recursive call
                // `self idx0 m' p h1_orig` with h1_orig : Le a b where Le a idx0 is required --
                // "spine comparison differs even after full normalization" in whole_term_check.
                if (!replaced) {
                    for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
                        if (reverted_js[ri] == i) { supplied = mk_var(body_base + ctor_arity + static_cast<uint32_t>(ri)); break; }
                    }
                }
            }
            subgoal_applied = mk_app(subgoal_applied, i == hyp_level ? true_ctor_app : supplied);
        }
        for (uint32_t fi = 0; fi < ctor_arity; ++fi) {
            subgoal_applied = mk_app(subgoal_applied, mk_var(body_base + fi));
            if (fi < rec_mask.size() && rec_mask[fi]) {
                // Gap C: fn_type's own Pi-chain curries `arity` index
                // parameters BEFORE the scrutinee, so a correct
                // recursive call needs self applied to those recovered
                // index arguments first, then the field -- self(idx0)
                // ...(idxN)(field), not self(field). Recovered from the
                // SAME field's own (forced) type in the body_base
                // scheme, decomposed the identical way Gap B already
                // does for ih_ty -- reusing the technique, not
                // duplicating a second one. Proven no-op for arity=0:
                // the loop below never executes, leaving self applied
                // to exactly the field, unchanged from before.
                NodeId ih_call = mk_var(self_level);
                detail::AppSpine field_spine_body = detail::decompose_app_spine(arg_types_body[fi]);
                for (uint32_t j = 0; j < arity && j < field_spine_body.args.size(); ++j)
                    ih_call = mk_app(ih_call, field_spine_body.args[j]);
                ih_call = mk_app(ih_call, mk_var(body_base + fi));
                // DELIBERATELY NOT applied to the reverted hyps (an earlier version of this
                // block did, "Gap D", wrongly -- removed after a real whole_term_check failure,
                // "applying a term that is not a function"). self's own fn_type carries one
                // extra Pi-parameter per reverted hyp AFTER idx+scrutinee, so `self idx.. field`
                // -- left exactly as built above -- has type  Pi (reverted hyps). <goal>,
                // which is precisely the agent-facing IH hypothesis's own type (a strengthened
                // IH, quantified over every generalized hyp: le_trans's own IH is
                // Pi (h1 : Le a b). Le a c'). Applying it to the reverted hyps HERE would hand
                // the agent a fully-applied, non-function value where their own hypothesis, and
                // the tactic-level solution built against it (IH(h1)), expects a function.
                // The agent supplies those arguments themselves, whichever proof of them they
                // choose -- exactly like any other universally-quantified IH.
                subgoal_applied = mk_app(subgoal_applied, ih_call);
            }
        }
        // Reverted hyps' own FRESH, LOCAL values for THIS branch -- not the outer, original hyp
        // applied directly (this function's own earlier version, confirmed wrong by a real
        // whole_term_check failure on le_trans: "mismatched value shapes", the base case's own
        // branch -- le_n has zero constructor fields -- left as a bare value instead of the
        // function-of-h1 the motive genuinely requires, since the motive itself was correctly
        // generalized over the reverted hyp for EVERY branch, not just ones with recursive
        // fields). Each reverted hyp needs its own fresh Lambda parameter here, exactly mirroring
        // arg_types_body's own fields, bound at the next available levels (right after this
        // constructor's own fields) -- consumed within subgoal_applied's own application here,
        // and bound further below when branch itself is wrapped.
        for (uint32_t ri = 0; ri < reverted_js.size(); ++ri) {
            subgoal_applied = mk_app(subgoal_applied, mk_var(body_base + ctor_arity + ri));
        }
        NodeId branch = subgoal_applied;
        // reverted_body_types: PORTED, true-runtime analogue of the agent-facing
        // reverted_dom_types peeling above -- same purely-syntactic technique (substitute_var,
        // not eval()/quote(), for the identical structural reason given there), applied to
        // generalized_goal_body instead of branch_expected. generalized_goal_body is first
        // specialized to THIS branch by substituting its own idx markers (self_level+1+j) with
        // inst.result_indices[j] -- the TRUE-RUNTIME, unforced pattern (mirroring exactly how
        // ih_call's own index recovery, above, already uses this same inst, body_base scheme) --
        // before peeling, so each reverted hyp's own domain comes out correctly specialized (e.g.
        // le_n's own repeated field collapsing both indices to the same value, confirmed directly
        // this session by a raw, unambiguous dump of the actual term).
        if (!reverted_js.empty()) {
            // FORCED result indices (forcing_body, the very same forcing arg_types_body above
            // was built with), not inst.result_indices directly -- see the agent-facing
            // branch_replacements's own comment for the full derivation. This is the side
            // whole_term_check actually exercises: check_match_branches builds each branch's
            // expected type from forced_result_indices, so a branch whose reverted-hyp Lambda
            // domain used the unforced field identity instead (this block's own earlier version,
            // le_S: Le a n' where the expected type says Le a b) fails with a hard mismatch.
            std::unordered_map<uint32_t, NodeId> body_idx_replacements;
            for (uint32_t j = 0; j < arity; ++j) body_idx_replacements[self_level + 1 + j] = forced_body_result_indices[j];
            NodeId remaining_body = detail::substitute_vars(generalized_goal_body, body_idx_replacements);
            std::vector<NodeId> reverted_body_types;
            for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
                auto pi_opt = inspect(remaining_body, [](Pi p) -> std::optional<Pi> { return p; },
                                                         [](auto) -> std::optional<Pi> { return std::nullopt; });
                if (!pi_opt) {
                    return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,
                        "induct: internal error -- expected a Pi layer for a reverted hypothesis (true-runtime side) but found none"});
                }
                uint32_t target_level = body_base + ctor_arity + static_cast<uint32_t>(ri);
                uint32_t shared_marker = body_base + static_cast<uint32_t>(ri);
                // F3 (cause 2): a reverted hypothesis may mention the induction variable ITSELF
                // (e.g. h : Eq n n while inducting on n). Its binder domain in this branch must then
                // be specialised to THIS constructor (Eq Zero Zero / Eq (Succ x) (Succ x)) -- the
                // agent-facing side already does this via branch_replacements[scrutinee] =
                // agent_ctor_app, but this true-runtime side only replaced the INDEX markers, so the
                // branch lambda bound h at type (Eq n' n') and whole_term_check then compared a stuck
                // `n'` against the constructor ("spine comparison differs"). Done per domain, AFTER it
                // is peeled (not up front in body_idx_replacements): this scheme's constructor-field
                // levels (body_base+k) are numerically the same range as its reverted-hyp markers
                // (body_base+ri), so substituting first would let the marker replacement below
                // clobber the constructor's own field references.
                reverted_body_types.push_back(detail::substitute_var(pi_opt->dom_id, scrutinee_level, true_ctor_app));
                remaining_body = detail::substitute_var(pi_opt->cod_id, shared_marker, mk_var(target_level));
            }
            for (auto it = reverted_body_types.rbegin(); it != reverted_body_types.rend(); ++it) branch = mk_lambda(*it, branch);
        }
        for (auto it = arg_types_body.rbegin(); it != arg_types_body.rend(); ++it) branch = mk_lambda(*it, branch);
        branches.push_back(branch);

        new_goals.push_back(Goal{subgoal_meta, branch_ec, std::move(branch_hyps), branch_expected});
    }

    NodeId match_term = mk_match(mk_var(scrutinee_level), motive_body, branches);
    NodeId curried_body = mk_lambda(scrut_binder_ty_body, match_term);
    for (auto it = index_types.rbegin(); it != index_types.rend(); ++it) curried_body = mk_lambda(*it, curried_body);

    NodeId fix_term = mk_fix(fn_type, decreasing_arg, curried_body);
    if (!is_locally_structurally_decreasing(self_level, type_global_id, decreasing_arg, curried_body)) {
        return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt,
            "induct: the assembled recursive proof failed the local structural recursion check -- "
            "some recursive call does not recurse on a structurally smaller instance of the "
            "hypothesis's own type"});
    }

    NodeId proof = fix_term;
    for (NodeId idx : concrete_indices) proof = mk_app(proof, idx);
    proof = mk_app(proof, mk_var(hyp_level));
    // Mirrors cases()'s own final match_term application exactly: fn_type's own fully-applied
    // result at this point is now the reverted hyps' own Pi-chain (reverted_target's own
    // outermost structure), not g.target directly, so proof -- like any ordinary function of
    // that Pi-chain's own type -- must be applied to each reverted hyp's own, still-unchanged,
    // in-scope value to actually produce something of g.target's own type. No-op when
    // reverted_js is empty, exactly as every other reverted_js-gated addition in this function.
    for (uint32_t rj : reverted_js) proof = mk_app(proof, mk_var(rj));

    ctx().meta_ctx.solve(g.meta_id, close_over_context(proof, g.ec));

    ps.goals.erase(ps.goals.begin() + *idx_opt);
    ps.goals.insert(ps.goals.begin() + *idx_opt, new_goals.begin(), new_goals.end());
    return {};
}
#endif



// ===========================================================================
// S1: declaration helpers (UNCHECKED -- libraries are loaded from the trusted reference kernel; admission, positivity and
// the guard checker are ported in S2) and a small typing fragment used to measure the cost of opening binders.
// ===========================================================================
namespace detail {
/// Opens a constructor's Pi telescope with context variables base_depth, base_depth+1, ...: field i's type is closed over
/// the earlier fields as FVars (a telescope), and the result indices over all of them.
inline InstantiatedCtor instantiate_ctor_telescope(uint32_t ctor_gid, uint32_t base_depth) {
    const CtorInfo& info = *ctx().globals[ctor_gid].ctor_info;
    NodeId cur = ctx().globals[ctor_gid].type;
    InstantiatedCtor out;
    for (uint32_t i = 0; i < info.arity; ++i) {
        Pi p = inspect(cur, [](Pi pi) { return pi; },
                       [](auto) -> Pi { throw std::logic_error("instantiate_ctor_telescope: constructor type has fewer Pi's than its own recorded arity"); });
        out.arg_types.push_back(p.dom_id);
        cur = instantiate1(p.cod_id, mk_fvar(base_depth + i));
    }
    std::vector<NodeId> raw_indices; NodeId head = cur;
    while (head.tag == NodeType::APP) { App a = ctx().apps[head.index]; raw_indices.push_back(a.arg_id); head = a.fn_id; }
    std::reverse(raw_indices.begin(), raw_indices.end());
    out.result_indices = std::move(raw_indices);
    return out;
}
inline std::vector<NodeId> instantiate_type_telescope(uint32_t type_global_id, uint32_t base_depth) {
    NodeId cur = ctx().globals[type_global_id].type;
    std::vector<NodeId> out; uint32_t i = 0;
    while (cur.tag == NodeType::PI) {
        Pi p = ctx().pis[cur.index];
        out.push_back(p.dom_id);
        cur = instantiate1(p.cod_id, mk_fvar(base_depth + i));
        ++i;
    }
    return out;
}
} // namespace detail

/// CtorSpec in telescope style: arg_types[j] may mention fields i < j as context variables FVar(i); result_indices may
/// mention all fields. The stored constructor type is the closed Pi chain.
inline NodeId build_ctor_type(uint32_t type_global_id, const CtorSpec& ctor) {
    NodeId t = apply_all(mk_const(type_global_id), ctor.result_indices);
    for (int j = static_cast<int>(ctor.arg_types.size()) - 1; j >= 0; --j)
        t = mk_pi(ctor.arg_types[j], abstract_fvars(t, std::vector<uint32_t>{static_cast<uint32_t>(j)}));
    return t;
}
inline InductiveHandle finish_inductive_unchecked(uint32_t type_global_id, std::vector<CtorSpec> ctors) {
    ExecutionContext& c = ctx();
    InductiveHandle handle; handle.type_global_id = type_global_id;
    for (std::size_t idx = 0; idx < ctors.size(); ++idx) {
        NodeId ctor_ty = build_ctor_type(type_global_id, ctors[idx]);
        uint32_t arity = static_cast<uint32_t>(ctors[idx].arg_types.size());
        handle.ctor_global_ids.push_back(c.globals.declare_ctor(ctor_ty, ctors[idx].name, type_global_id, static_cast<uint32_t>(idx), arity));
    }
    return handle;
}
inline void finish_recursive_function(uint32_t fn_gid, uint32_t ind_type_global_id, uint32_t decreasing_arg_level, NodeId body, bool = false) {
    ctx().globals.defs[fn_gid].value = body;
    ctx().globals.defs[fn_gid].recursive_info = RecursiveFnInfo{ind_type_global_id, decreasing_arg_level, true};
}
inline uint32_t declare_unchecked(NodeId type, std::optional<NodeId> value, const std::string& name) {
    return ctx().globals.declare(type, value, name);
}

// ---- S1 typing fragment: Type, Pi, Lambda, App, Let, Const, context variables. Binders are opened with a fresh FVar
// (instantiate), typed, and closed again (abstract): the standard locally-nameless discipline.
using TcResult = std::expected<NodeId, std::string>;
inline TcResult tc_infer(NodeId t, const ElabCtx& ec);
inline std::expected<void, std::string> tc_check(NodeId t, NodeId expected, const ElabCtx& ec) {
    auto got = tc_infer(t, ec);
    if (!got) return std::unexpected(got.error());
    if (!conv_value(eval(*got, ec.values), eval(expected, ec.values), ec.size())) return std::unexpected(std::string("type mismatch"));
    return {};
}
inline std::expected<uint32_t, std::string> tc_sort(NodeId t, const ElabCtx& ec) {
    auto ty = tc_infer(t, ec);
    if (!ty) return std::unexpected(ty.error());
    Value v = eval(*ty, ec.values);
    if (auto* vt = std::get_if<VType>(&v)) return vt->universe_level;
    return std::unexpected(std::string("not a type"));
}
inline TcResult tc_infer(NodeId t, const ElabCtx& ec) {
    switch (t.tag) {
        case NodeType::VAR: {
            Var v = ctx().vars[t.index];
            if (v.bound) return std::unexpected(std::string("loose bound variable"));
            if (v.n >= ec.size()) return std::unexpected(std::string("context variable outside the context"));
            return ec.types[v.n];
        }
        case NodeType::TYPE: return mk_type(ctx().types[t.index].universe_level + 1);
        case NodeType::CONST: return ctx().globals[ctx().consts[t.index].global_id].type;
        case NodeType::PI: {
            // Telescope: open the whole chain of directly nested Pis with fresh context variables in ONE pass (per-binder instantiate
            // would be quadratic in the chain length), type every domain and the final codomain, then fold the universe rule.
            std::vector<NodeId> doms; NodeId cod = t;
            while (cod.tag == NodeType::PI) { Pi p = ctx().pis[cod.index]; doms.push_back(p.dom_id); cod = p.cod_id; }
            const uint32_t m = static_cast<uint32_t>(doms.size()), n = ec.size();
            ElabCtx cur = ec; std::vector<NodeId> fv_rev; std::vector<uint32_t> sorts;
            for (uint32_t j = 0; j < m; ++j) {
                NodeId dj = instantiate(doms[j], fv_rev);
                auto s = tc_sort(dj, cur); if (!s) return std::unexpected(s.error());
                sorts.push_back(*s); cur = cur.extend(dj); fv_rev.insert(fv_rev.begin(), mk_fvar(n + j));
            }
            auto sc = tc_sort(instantiate(cod, fv_rev), cur); if (!sc) return std::unexpected(sc.error());
            uint32_t level = *sc;
            for (int j = static_cast<int>(m) - 1; j >= 0; --j) level = (level == 0) ? 0 : std::max(sorts[j], level);
            return mk_type(level);
        }
        case NodeType::LAMBDA: {
            // Telescope (see PI): open all nested lambdas at once, type the body once, close with one abstract per domain.
            std::vector<NodeId> doms; NodeId body = t;
            while (body.tag == NodeType::LAMBDA) { Lambda l = ctx().lambdas[body.index]; doms.push_back(l.type_id); body = l.body_id; }
            const uint32_t m = static_cast<uint32_t>(doms.size()), n = ec.size();
            ElabCtx cur = ec; std::vector<NodeId> fv_rev, opened; std::vector<uint32_t> fvl;
            for (uint32_t j = 0; j < m; ++j) {
                NodeId dj = instantiate(doms[j], fv_rev);
                auto s = tc_sort(dj, cur); if (!s) return std::unexpected(s.error());
                opened.push_back(dj); cur = cur.extend(dj); fv_rev.insert(fv_rev.begin(), mk_fvar(n + j)); fvl.push_back(n + j);
            }
            auto bt = tc_infer(instantiate(body, fv_rev), cur); if (!bt) return bt;
            NodeId result = abstract_fvars(*bt, fvl);
            for (int j = static_cast<int>(m) - 1; j >= 0; --j)
                result = mk_pi(abstract_fvars(opened[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + j)), result);
            return result;
        }
        case NodeType::APP: {
            App a = ctx().apps[t.index];
            auto tf = tc_infer(a.fn_id, ec); if (!tf) return tf;
            Value fv = eval(*tf, ec.values);
            auto* pi = std::get_if<VPi>(&fv);
            if (!pi) return std::unexpected(std::string("application of a non-function"));
            auto ok = tc_check(a.arg_id, quote(eval(pi->dom_id, pi->env), ec.size()), ec); if (!ok) return std::unexpected(ok.error());
            return quote(eval(pi->cod_id, pi->env.extend(eval(a.arg_id, ec.values))), ec.size());
        }
        case NodeType::LET: {
            Let l = ctx().lets[t.index];
            auto s = tc_sort(l.type_id, ec); if (!s) return std::unexpected(s.error());
            auto ok = tc_check(l.value_id, l.type_id, ec); if (!ok) return std::unexpected(ok.error());
            return tc_infer(instantiate1(l.body_id, l.value_id), ec);
        }
        default: return std::unexpected(std::string("tc_infer: node kind not supported by the S1 typing fragment"));
    }
}


// ===========================================================================
// S2: the typing core on the hybrid representation.
//   * binders are opened with fresh context variables and closed with abstract, ALWAYS as telescopes (one instantiate for a
//     whole chain of lambdas/Pis -- per-binder instantiate was measured quadratic);
//   * infer(Const)/infer(Var) need no shifting or re-anchoring: global types are closed, hypothesis types are valid in any
//     extension of their context;
//   * index forcing is a simultaneous replace_fvars: forced values may carry binders (lambda predicates) and are position
//     independent, which is exactly what the old syntactic substitution could not do;
//   * infer(Fix) runs the structural-recursion guard itself: the kernel does not trust the tactic that built the Fix.
// unify is pure conversion here (no metas); S3 replaces it with real unification.
// ===========================================================================

/// Closes an opened Pi telescope (domains mention earlier fields as context variables base, base+1, ...) around `body`.
inline NodeId close_pi_telescope(const std::vector<NodeId>& opened_doms, NodeId body, uint32_t base) {
    const std::size_t m = opened_doms.size();
    std::vector<uint32_t> fvl(m);
    for (std::size_t i = 0; i < m; ++i) fvl[i] = base + static_cast<uint32_t>(i);
    NodeId result = abstract_fvars(body, fvl);
    for (int j = static_cast<int>(m) - 1; j >= 0; --j)
        result = mk_pi(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + j)), result);
    return result;
}

/// All-levels form of a term, for the legacy structural-recursion analyses: every bound variable becomes the context variable
/// whose level is its absolute binder number (depth counts binders from the start of the context).
inline NodeId to_levels(NodeId t, uint32_t depth) {
    if (t.tag == NodeType::VAR) {
        Var v = ctx().vars[t.index];
        return v.bound ? mk_fvar(depth - 1 - v.n) : t;
    }
    return map_children(t, [&](NodeId c, uint32_t extra) { return to_levels(c, depth + extra); });
}

/// Structural-recursion guard for a Fix node living in a context of size n0 (self = context variable n0 inside the body).
inline bool fix_guard_ok(const Fix& f, uint32_t n0) {
    NodeId cur = f.fn_type;
    for (uint32_t i = 0; i < f.decreasing_arg; ++i) { if (cur.tag != NodeType::PI) return false; cur = ctx().pis[cur.index].cod_id; }
    if (cur.tag != NodeType::PI) return false;
    detail::AppSpine sp = detail::decompose_app_spine(ctx().pis[cur.index].dom_id);
    if (sp.head.tag != NodeType::CONST) return false;
    const uint32_t tid = ctx().consts[sp.head.index].global_id;
    if (ctx().globals.ctors_of(tid).empty()) return false;
    return is_locally_structurally_decreasing(n0, tid, f.decreasing_arg, to_levels(f.body, n0 + 1));
}

inline std::expected<uint32_t, TypeError> sort_level_of(NodeId t, ElabCtx& ec, ElabState& es, const char* what) {
    TypeResult r = infer(t, ec, es);
    if (!r) return std::unexpected(r.error());
    Value v = eval(*r, ec.values);
    if (auto* vt = std::get_if<VType>(&v)) return vt->universe_level;
    return std::unexpected(TypeError{TypeErrorKind::UniverseError, t, std::nullopt, what});
}

inline std::vector<NodeId> reversed(const std::vector<NodeId>& v) { return std::vector<NodeId>(v.rbegin(), v.rend()); }

inline ElabResult verify_motive_shape(NodeId motive_id, uint32_t arity, ElabCtx& ec, ElabState& es) {
    TypeResult motive_ty_r = infer(motive_id, ec, es);
    if (!motive_ty_r) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive_id, std::nullopt, "explicit motive_id is not well-typed"});
    Value motive_ty_val = eval(*motive_ty_r, ec.values);
    for (uint32_t i = 0; i <= arity; ++i) {
        auto* mpi = std::get_if<VPi>(&motive_ty_val);
        if (!mpi) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive_id, std::nullopt,
            "explicit motive_id is not shaped like a valid motive -- expected a Pi chain of (index count + 1) arguments, ending in a sort"});
        Value next = eval(mpi->cod_id, mpi->env.extend(fresh_neutral(ec.size())));
        motive_ty_val = std::move(next);
    }
    return {};
}

/// Forced field values replace the field variables SIMULTANEOUSLY (a forced value may itself carry binders).
inline std::vector<NodeId> apply_ctor_index_forcing(const std::vector<NodeId>& arg_types, const CtorIndexForcing& forcing) {
    std::unordered_map<uint32_t, NodeId> m(forcing.forced.begin(), forcing.forced.end());
    std::vector<NodeId> out = arg_types;
    for (NodeId& ty : out) ty = replace_fvars(ty, m);
    return out;
}
inline std::vector<NodeId> apply_ctor_index_forcing_to_result_indices(const std::vector<NodeId>& result_indices, const CtorIndexForcing& forcing) {
    std::unordered_map<uint32_t, uint32_t> occurrence_count;
    for (NodeId n : result_indices) if (n.tag == NodeType::VAR) { Var v = ctx().vars[n.index]; if (!v.bound) occurrence_count[v.n]++; }
    std::unordered_map<uint32_t, NodeId> m;
    for (auto& [level, replacement] : forcing.forced) if (occurrence_count[level] <= 1) m[level] = replacement; // repeated field keeps its own identity
    std::vector<NodeId> out = result_indices;
    for (NodeId& n : out) n = replace_fvars(n, m);
    return out;
}

/// F9: the constructor application a branch sees for the scrutinee. A field whose forced value was substituted into the result indices
/// (n := a in le_S, occurring once) is applied to that value -- le_S a m h' -- every other field to its own binder (le_n n, refl Nat x).
/// It uses EXACTLY the substitution map of apply_ctor_index_forcing_to_result_indices, so the application's type is the motive's own
/// index/scrutinee-type arguments. (The reference applied the constructor to the field binders, which is ill-typed against the forced field
/// types: le_S n m h' with h' : Le a m. Invisible unless the goal or a reverted hypothesis mentions the scrutinee.)
/// `actual`, when given, maps field position -> the context level of that field's binder (induct interleaves IH hypotheses).
inline NodeId forced_ctor_app(uint32_t ctor_gid, uint32_t base, uint32_t arity, const detail::InstantiatedCtor& inst,
                               const CtorIndexForcing& forcing, const std::vector<uint32_t>* actual = nullptr) {
    std::unordered_map<uint32_t, uint32_t> occurrence_count;
    for (NodeId n : inst.result_indices) if (n.tag == NodeType::VAR) { Var v = ctx().vars[n.index]; if (!v.bound) occurrence_count[v.n]++; }
    NodeId app = mk_const(ctor_gid);
    for (uint32_t ai = 0; ai < arity; ++ai) {
        auto it = forcing.forced.find(base + ai);
        const bool use_forced = it != forcing.forced.end() && occurrence_count[base + ai] <= 1;
        app = mk_app(app, use_forced ? it->second : mk_fvar(actual ? (*actual)[ai] : base + ai));
    }
    return app;
}

inline ElabResult check_match_branches(uint32_t type_global_id, NodeId motive, const std::vector<NodeId>& branches,
                                        const std::vector<NodeId>& concrete_indices, ElabCtx& ec, ElabState& es) {
    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(type_global_id);
    if (ctor_gids.size() != branches.size())
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive, std::nullopt,
            "Match branch count does not match the scrutinee's inductive type's constructor count (non-exhaustive or over-long match)"});
    for (std::size_t ci = 0; ci < ctor_gids.size(); ++ci) {
        detail::InstantiatedCtor inst = detail::instantiate_ctor_telescope(ctor_gids[ci], ec.size());
        const uint32_t arity = static_cast<uint32_t>(inst.arg_types.size());
        CtorIndexForcing forcing = unify_ctor_pattern_indices(inst, concrete_indices, ec.size(), arity, ec.values);
        if (!forcing.ok) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, branches[ci], std::nullopt, forcing.error});
        std::vector<NodeId> arg_types = apply_ctor_index_forcing(inst.arg_types, forcing);
        NodeId ctor_app = forced_ctor_app(ctor_gids[ci], ec.size(), arity, inst, forcing);
        std::vector<NodeId> forced_result_indices = apply_ctor_index_forcing_to_result_indices(inst.result_indices, forcing);
        NodeId branch_expected = mk_app(apply_all(motive, forced_result_indices), ctor_app);
        branch_expected = close_pi_telescope(arg_types, branch_expected, ec.size());
        if (ElabResult r = check(branches[ci], branch_expected, ec, es); !r) return r;
    }
    return {};
}

inline TypeResult infer(NodeId t, ElabCtx& ec, ElabState& es) {
    auto err = [&](TypeErrorKind k, NodeId at, const char* msg) { return std::unexpected(TypeError{k, at, std::nullopt, msg}); };
    switch (t.tag) {
        case NodeType::VAR: {
            Var v = ctx().vars[t.index];
            if (v.bound) return err(TypeErrorKind::UnboundVariable, t, "loose bound variable: a term reached the kernel with an unclosed binder");
            if (v.n >= ec.size()) return err(TypeErrorKind::UnboundVariable, t, "unbound local variable");
            return ec.types[v.n];
        }
        case NodeType::CONST: return ctx().globals[ctx().consts[t.index].global_id].type;
        case NodeType::TYPE: return mk_type(ctx().types[t.index].universe_level + 1);
        case NodeType::META: return ctx().meta_ctx[ctx().metas[t.index].meta_id].type;
        case NodeType::TYPE_UNIV: return err(TypeErrorKind::Other, t, "TypeUniv is elaborator-internal and not meaningful as a standalone infer target");
        case NodeType::APP: {
            App a = ctx().apps[t.index];
            ITP_TRY(NodeId fn_ty, infer(a.fn_id, ec, es));
            Value fn_ty_val = eval(fn_ty, ec.values);
            auto* pi = std::get_if<VPi>(&fn_ty_val);
            if (!pi) return err(TypeErrorKind::NotAFunction, a.fn_id, "applying a term that is not a function");
            NodeId dom_ty = quote(eval(pi->dom_id, pi->env), ec.size());
            if (ElabResult r = check(a.arg_id, dom_ty, ec, es); !r) return std::unexpected(r.error());
            return quote(eval(pi->cod_id, pi->env.extend(eval(a.arg_id, ec.values))), ec.size());
        }
        case NodeType::LAMBDA: {
            std::vector<NodeId> doms; NodeId body = t;
            while (body.tag == NodeType::LAMBDA) { Lambda l = ctx().lambdas[body.index]; doms.push_back(l.type_id); body = l.body_id; }
            const uint32_t m = static_cast<uint32_t>(doms.size()), n = ec.size();
            ElabCtx cur = ec; std::vector<NodeId> fv_rev, opened;
            for (uint32_t j = 0; j < m; ++j) {
                NodeId dj = instantiate(doms[j], fv_rev);
                auto s = sort_level_of(dj, cur, es, "Lambda domain must be a type"); if (!s) return std::unexpected(s.error());
                opened.push_back(dj); cur = cur.extend(dj); fv_rev.insert(fv_rev.begin(), mk_fvar(n + j));
            }
            ITP_TRY(NodeId bt, infer(instantiate(body, fv_rev), cur, es));
            return close_pi_telescope(opened, bt, n);
        }
        case NodeType::PI: {
            std::vector<NodeId> doms; NodeId cod = t;
            while (cod.tag == NodeType::PI) { Pi p = ctx().pis[cod.index]; doms.push_back(p.dom_id); cod = p.cod_id; }
            const uint32_t m = static_cast<uint32_t>(doms.size()), n = ec.size();
            ElabCtx cur = ec; std::vector<NodeId> fv_rev; std::vector<uint32_t> sorts;
            for (uint32_t j = 0; j < m; ++j) {
                NodeId dj = instantiate(doms[j], fv_rev);
                auto s = sort_level_of(dj, cur, es, "Pi domain must be a type"); if (!s) return std::unexpected(s.error());
                sorts.push_back(*s); cur = cur.extend(dj); fv_rev.insert(fv_rev.begin(), mk_fvar(n + j));
            }
            auto sc = sort_level_of(instantiate(cod, fv_rev), cur, es, "Pi codomain must be a type"); if (!sc) return std::unexpected(sc.error());
            uint32_t level = *sc;
            for (int j = static_cast<int>(m) - 1; j >= 0; --j) level = (level == 0) ? 0 : std::max(sorts[j], level);
            return mk_type(level);
        }
        case NodeType::LET: {
            Let l = ctx().lets[t.index];
            auto s = sort_level_of(l.type_id, ec, es, "Let type must be a type"); if (!s) return std::unexpected(s.error());
            if (ElabResult r = check(l.value_id, l.type_id, ec, es); !r) return std::unexpected(r.error());
            return infer(instantiate1(l.body_id, l.value_id), ec, es); // zeta: the body is typed with the value substituted
        }
        case NodeType::MATCH: {
            Match m = ctx().matches[t.index];
            if (m.motive_id.tag == NodeType::META && !ctx().meta_ctx.is_solved(ctx().metas[m.motive_id.index].meta_id))
                return err(TypeErrorKind::MotiveInferenceFailure, t, "infer(Match) requires an already-concrete motive; use check() against a goal type to have one constructed automatically");
            ITP_TRY(NodeId scrut_ty, infer(m.scrutinee_id, ec, es));
            Value scrut_ty_val = eval(scrut_ty, ec.values);
            auto* scrut_neutral = std::get_if<VNeutral>(&scrut_ty_val);
            if (!scrut_neutral || scrut_neutral->head.kind != NeutralHead::Kind::GlobalConst)
                return err(TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, "scrutinee's type is not an inductive type application");
            const uint32_t type_global_id = scrut_neutral->head.id;
            std::vector<NodeId> concrete_indices;
            for (auto& frame : scrut_neutral->spine) {
                auto* af = std::get_if<ApplyFrame>(&frame);
                if (!af) return err(TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, "scrutinee's type has a non-application eliminator frame; cannot enumerate its indices");
                concrete_indices.push_back(quote(af->arg, ec.size()));
            }
            if (ElabResult r = verify_motive_shape(m.motive_id, static_cast<uint32_t>(concrete_indices.size()), ec, es); !r) return std::unexpected(r.error());
            if (ElabResult r = check_match_branches(type_global_id, m.motive_id, *m.branches, concrete_indices, ec, es); !r) return std::unexpected(r.error());
            return quote(eval(mk_app(apply_all(m.motive_id, concrete_indices), m.scrutinee_id), ec.values), ec.size());
        }
        case NodeType::FIX: {
            Fix f = ctx().fixes[t.index];
            auto s = sort_level_of(f.fn_type, ec, es, "Fix's own fn_type must be a type"); if (!s) return std::unexpected(s.error());
            ElabCtx ec2 = ec.extend(f.fn_type); // the self reference: a context variable of type fn_type (valid here by weakening)
            if (ElabResult r = check(instantiate1(f.body, mk_fvar(ec.size())), f.fn_type, ec2, es); !r) return std::unexpected(r.error());
            if (!fix_guard_ok(f, ec.size())) return err(TypeErrorKind::Other, t, "Fix failed the structural recursion check: some recursive call is not on a structurally smaller instance of the decreasing argument");
            return f.fn_type;
        }
        case NodeType::QUOT_TYPE: case NodeType::QUOT_MK: {
            NodeId rel = (t.tag == NodeType::QUOT_TYPE) ? ctx().quot_types[t.index].relation : ctx().quot_mks[t.index].relation;
            ITP_TRY(NodeId rel_ty, infer(rel, ec, es));
            Value rel_ty_val = eval(rel_ty, ec.values);
            auto* outer_pi = std::get_if<VPi>(&rel_ty_val);
            if (!outer_pi) return err(TypeErrorKind::NotAFunction, rel, "Quot's relation must be a binary relation A -> A -> Sort");
            NodeId elem_ty = quote(eval(outer_pi->dom_id, outer_pi->env), ec.size());
            if (t.tag == NodeType::QUOT_TYPE) return infer(elem_ty, ec, es); // the quotient lives in the same sort as the underlying type
            if (ElabResult r = check(ctx().quot_mks[t.index].elem, elem_ty, ec, es); !r) return std::unexpected(r.error());
            return mk_quot_type(rel);
        }
        case NodeType::QUOT_LIFT: {
            QuotLift q = ctx().quot_lifts[t.index];
            if (ElabResult r = check(q.quotient_elem, mk_quot_type(q.relation), ec, es); !r) return std::unexpected(r.error());
            return q.codomain;
        }
    }
    std::unreachable();
}

/// Checks a chain of lambdas against a Pi telescope in ONE pass (one instantiate for the body, not one per binder).
inline ElabResult check_lambda(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {
    Value cur = eval(expected_ty, ec.values);
    ElabCtx cur_ec = ec; std::vector<NodeId> fv_rev; NodeId body = t;
    const uint32_t n = ec.size(); uint32_t j = 0;
    while (body.tag == NodeType::LAMBDA) {
        auto* pi = std::get_if<VPi>(&cur);
        if (!pi) {
            if (j == 0) return std::unexpected(TypeError{TypeErrorKind::Mismatch, t, expected_ty, "checking a lambda against a non-function expected type"});
            break;
        }
        Lambda l = ctx().lambdas[body.index];
        NodeId dom_j = instantiate(l.type_id, fv_rev);
        NodeId expected_dom = quote(eval(pi->dom_id, pi->env), n + j);
        if (ElabResult r = unify(dom_j, expected_dom, cur_ec, es); !r) return r;
        Value next = eval(pi->cod_id, pi->env.extend(fresh_neutral(n + j)));
        cur = std::move(next);
        cur_ec = cur_ec.extend(dom_j); fv_rev.insert(fv_rev.begin(), mk_fvar(n + j)); body = l.body_id; ++j;
    }
    return check(instantiate(body, fv_rev), quote(cur, n + j), cur_ec, es);
}

inline ElabResult check_match(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {
    Match m = ctx().matches[t.index];
    TypeResult scrut_ty_r = infer(m.scrutinee_id, ec, es);
    if (!scrut_ty_r) return std::unexpected(scrut_ty_r.error());
    Value scrut_ty_val = eval(*scrut_ty_r, ec.values);
    auto* scrut_neutral = std::get_if<VNeutral>(&scrut_ty_val);
    if (!scrut_neutral || scrut_neutral->head.kind != NeutralHead::Kind::GlobalConst)
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, expected_ty, "scrutinee's type is not an inductive type application"});
    const uint32_t type_global_id = scrut_neutral->head.id;
    std::vector<NodeId> concrete_indices;
    for (auto& frame : scrut_neutral->spine) {
        auto* af = std::get_if<ApplyFrame>(&frame);
        if (!af) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, expected_ty, "scrutinee's type has a non-application eliminator frame; cannot enumerate its indices"});
        concrete_indices.push_back(quote(af->arg, ec.size()));
    }
    NodeId motive = m.motive_id;
    if (m.motive_id.tag == NodeType::META && !ctx().meta_ctx.is_solved(ctx().metas[m.motive_id.index].meta_id)) {
        // The motive is a hole: build it by abstracting the indices and the scrutinee out of the expected type.
        const uint32_t hole_meta_id = ctx().metas[m.motive_id.index].meta_id, n = ec.size(), arity = static_cast<uint32_t>(concrete_indices.size());
        std::vector<NodeId> index_types = detail::instantiate_type_telescope(type_global_id, n);
        if (index_types.size() != arity)
            return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, t, expected_ty, "index count mismatch between the scrutinee's type application and the inductive's own declared arity"});
        std::vector<std::pair<NodeId, NodeId>> subs; std::vector<NodeId> idx_vars;
        for (uint32_t i = 0; i < arity; ++i) { subs.push_back({concrete_indices[i], mk_fvar(n + i)}); idx_vars.push_back(mk_fvar(n + i)); }
        subs.push_back({m.scrutinee_id, mk_fvar(n + arity)});
        std::vector<NodeId> doms = index_types; doms.push_back(apply_all(mk_const(type_global_id), idx_vars));
        NodeId built = close_lambda_telescope(doms, replace_terms(expected_ty, subs), n);
        if (!infer(built, ec, es))
            return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, t, expected_ty,
                "constructed motive failed to type-check -- the goal likely depends on the scrutinee in a way that cannot be abstracted cleanly; supply an explicit motive_id instead"});
        ctx().meta_ctx.solve(hole_meta_id, built);
        motive = built;
    } else {
        if (ElabResult r = verify_motive_shape(m.motive_id, static_cast<uint32_t>(concrete_indices.size()), ec, es); !r) return r;
        NodeId applied = mk_app(apply_all(motive, concrete_indices), m.scrutinee_id);
        if (ElabResult r = unify(applied, expected_ty, ec, es); !r) return r;
    }
    return check_match_branches(type_global_id, motive, *m.branches, concrete_indices, ec, es);
}

inline ElabResult check(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {
    if (t.tag == NodeType::LAMBDA) return check_lambda(t, expected_ty, ec, es);
    if (t.tag == NodeType::MATCH) return check_match(t, expected_ty, ec, es);
    TypeResult inferred = infer(t, ec, es);
    if (!inferred) return std::unexpected(inferred.error());
    return unify(*inferred, expected_ty, ec, es);
}

/// Checked inductive declaration: strict positivity + admission (typed fields/indices, universe rule), then constructors.
inline InductiveHandle finish_inductive(uint32_t type_global_id, std::vector<CtorSpec> ctors, bool allow_unsafe_skip_positivity) {
    if (!allow_unsafe_skip_positivity) {
        for (const auto& ctor : ctors)
            for (NodeId arg_ty : ctor.arg_types)
                if (!is_strictly_positive(type_global_id, arg_ty))
                    throw std::logic_error("itp: constructor '" + ctor.name + "' fails strict positivity -- the inductive being defined appears to the left of an arrow in an argument type");
        check_inductive_admission(type_global_id, ctors);
    }
    return finish_inductive_unchecked(type_global_id, std::move(ctors));
}


// ===========================================================================
// S3 (tail part): proof states and the first tactics on the hybrid representation.
// ===========================================================================
inline ProofState start_proof(NodeId goal_type) {
    uint32_t root_meta = ctx().meta_ctx.fresh(goal_type, 0);
    ProofState ps;
    ps.goals.push_back(Goal{root_meta, ElabCtx{}, {}, goal_type});
    return ps;
}
inline bool proof_complete(const ProofState& ps) { return ps.goals.empty() && ps.es.all_constraints_resolved(); }
inline NodeId extract_proof_term(uint32_t root_meta_id) {
    NoDeltaScope keep_constants_folded; // F2a: do not inline library theorems/definitions into the proof
    return quote(eval(mk_meta(root_meta_id), {}), 0);
}
inline NodeId meta_applied_to_context(uint32_t meta_id, uint32_t n) {
    NodeId r = mk_meta(meta_id);
    for (uint32_t i = 0; i < n; ++i) r = mk_app(r, mk_fvar(i));
    return r;
}
inline std::unexpected<TypeError> no_goal_error(const char* tactic, std::optional<uint32_t> target_meta_id) {
    return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt,
        std::string(tactic) + (target_meta_id ? ": no goal with the given meta_id (already solved or superseded)" : ": no open goals")});
}

inline ElabResult intro(ProofState& ps, const std::string& name = "", std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return no_goal_error("intro", target_meta_id);
    Goal& g = ps.goals[*idx_opt];
    const uint32_t n = g.ec.size();
    NodeId dom_ty, new_target;
    if (g.target.tag == NodeType::PI) { // the common case stays purely syntactic, so the goal keeps its flat (folded) shape
        Pi p = ctx().pis[g.target.index];
        dom_ty = p.dom_id; new_target = instantiate1(p.cod_id, mk_fvar(n));
    } else {
        Value target_val = eval(g.target, g.ec.values);
        auto* pi = std::get_if<VPi>(&target_val);
        if (!pi) return std::unexpected(TypeError{TypeErrorKind::Mismatch, g.target, std::nullopt, "intro: goal is not a Pi type, nothing to introduce"});
        dom_ty = quote(eval(pi->dom_id, pi->env), n);
        new_target = quote(eval(pi->cod_id, pi->env.extend(fresh_neutral(n))), n + 1);
    }
    ElabCtx new_ec = g.ec.extend(dom_ty);
    std::vector<Hypothesis> new_hyps = g.hyps;
    new_hyps.push_back(Hypothesis{name, dom_ty});
    uint32_t new_meta = ctx().meta_ctx.fresh(close_pi_over_context(new_target, new_ec), new_ec.size());
    NodeId inner = mk_lambda(dom_ty, abstract_fvars(meta_applied_to_context(new_meta, new_ec.size()), std::vector<uint32_t>{n}));
    ctx().meta_ctx.solve(g.meta_id, close_over_context(inner, g.ec));
    ps.goals[*idx_opt] = Goal{new_meta, new_ec, std::move(new_hyps), new_target};
    return {};
}

inline ElabResult exact(ProofState& ps, NodeId term, std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return no_goal_error("exact", target_meta_id);
    Goal& g = ps.goals[*idx_opt];
    if (ElabResult r = check(term, g.target, g.ec, ps.es); !r) return r;
    if (!ctx().meta_ctx.is_solved(g.meta_id)) ctx().meta_ctx.solve(g.meta_id, close_over_context(term, g.ec));
    ps.goals.erase(ps.goals.begin() + *idx_opt);
    return {};
}

inline ElabResult apply(ProofState& ps, NodeId candidate, std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return no_goal_error("apply", target_meta_id);
    Goal g = ps.goals[*idx_opt]; // copy: only committed to ps.goals on success
    TypeResult cand_ty_r = infer(candidate, g.ec, ps.es);
    if (!cand_ty_r) return std::unexpected(cand_ty_r.error());
    NodeId applied = candidate;
    Value cur_ty_val = eval(*cand_ty_r, g.ec.values);
    std::vector<std::pair<uint32_t, NodeId>> arg_metas; // (reference meta id, its own unwrapped arg type)
    while (true) {
        auto* pi = std::get_if<VPi>(&cur_ty_val);
        if (!pi) break;
        NodeId arg_ty = quote(eval(pi->dom_id, pi->env), g.ec.size());
        uint32_t m = ctx().meta_ctx.fresh(close_pi_over_context(arg_ty, g.ec), g.ec.size());
        arg_metas.push_back({m, arg_ty});
        NodeId meta_ref = meta_applied_to_context(m, g.ec.size());
        applied = mk_app(applied, meta_ref);
        Value next = eval(pi->cod_id, pi->env.extend(eval(meta_ref, g.ec.values)));
        cur_ty_val = std::move(next);
    }
    NodeId concl_ty = quote(cur_ty_val, g.ec.size());
    if (ElabResult r = unify(concl_ty, g.target, g.ec, ps.es); !r) return r;
    if (ElabResult r = drain_pending(g.ec, ps.es); !r) return r;
    std::vector<Goal> new_goals;
    for (auto& [m, arg_ty] : arg_metas) {
        if (ctx().meta_ctx.is_solved(m)) continue;
        uint32_t subgoal_meta = ctx().meta_ctx.fresh(close_pi_over_context(arg_ty, g.ec), g.ec.size());
        ctx().meta_ctx.solve(m, close_over_context(meta_applied_to_context(subgoal_meta, g.ec.size()), g.ec));
        new_goals.push_back(Goal{subgoal_meta, g.ec, g.hyps, arg_ty});
    }
    ctx().meta_ctx.solve(g.meta_id, close_over_context(applied, g.ec));
    ps.goals.erase(ps.goals.begin() + *idx_opt);
    ps.goals.insert(ps.goals.begin() + *idx_opt, new_goals.begin(), new_goals.end());
    return {};
}

/// Rewrites occurrences of `x` in the goal by `y`, given eq_term : Eq A x y.  goal[x] <-- transport A (fun z => goal[z]) y x (symm eq) goal[y]
inline ElabResult rewrite_eq(ProofState& ps, NodeId eq_term, NodeId x, NodeId y, uint32_t transport_gid, uint32_t symm_gid,
                              std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return no_goal_error("rewrite_eq", target_meta_id);
    Goal g = ps.goals[*idx_opt]; // copy: only committed to ps.goals on success
    TypeResult eq_ty_r = infer(eq_term, g.ec, ps.es);
    if (!eq_ty_r) return std::unexpected(eq_ty_r.error());
    Value eq_ty_val = eval(*eq_ty_r, g.ec.values);
    auto* eq_neutral = std::get_if<VNeutral>(&eq_ty_val);
    if (!eq_neutral || eq_neutral->head.kind != NeutralHead::Kind::GlobalConst || eq_neutral->spine.size() != 3)
        return std::unexpected(TypeError{TypeErrorKind::Mismatch, *eq_ty_r, std::nullopt,
            "rewrite_eq: eq_term's own type is not a 3-argument (A, x, y) relation application -- expected Eq A x y"});
    std::vector<NodeId> parts;
    for (auto& frame : eq_neutral->spine) {
        auto* af = std::get_if<ApplyFrame>(&frame);
        if (!af) return std::unexpected(TypeError{TypeErrorKind::Mismatch, *eq_ty_r, std::nullopt, "rewrite_eq: eq_term's own type has a non-application eliminator frame"});
        parts.push_back(quote(af->arg, g.ec.size()));
    }
    const uint32_t n = g.ec.size();
    NodeId A = parts[0];
    // F13: rewrite the INSTANTIATED goal. A goal produced by apply mentions solved metas as `?m x0 .. xn`; rewriting a context variable inside
    // those spines (but not in the proofs passed alongside) yields an ill-formed goal, and misses the real occurrence hidden inside the meta.
    x = resolve_metas(x); y = resolve_metas(y);
    NodeId target_z = resolve_metas(g.target);
    NodeId body_open = replace_terms(target_z, {{x, mk_fvar(n)}});           // goal with x abstracted into a context variable z = n
    NodeId motive = mk_lambda(A, abstract_fvars(body_open, std::vector<uint32_t>{n}));
    if (!infer(motive, g.ec, ps.es))
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,
            "rewrite_eq: the motive is not type correct -- some part of the goal depends on the rewritten term in a way that cannot be abstracted"});
    NodeId new_target = replace_fvars(body_open, {{n, y}});                  // goal[x := y]
    uint32_t new_meta = ctx().meta_ctx.fresh(close_pi_over_context(new_target, g.ec), n);
    NodeId new_meta_ref = meta_applied_to_context(new_meta, n);
    NodeId symm_term = mk_app(mk_app(mk_app(mk_app(mk_const(symm_gid), A), x), y), eq_term);
    NodeId wrapped = mk_app(mk_app(mk_app(mk_app(mk_app(mk_app(mk_const(transport_gid), A), motive), y), x), symm_term), new_meta_ref);
    ctx().meta_ctx.solve(g.meta_id, close_over_context(wrapped, g.ec));
    ps.goals[*idx_opt] = Goal{new_meta, g.ec, g.hyps, new_target};
    return {};
}


// ===========================================================================
// S4: cases and induct on the hybrid representation.
//   * ONE construction (not the reference's "agent" and "runtime" numberings): the generalised goal G is built once over fresh
//     context variables n..n+arity (indices, scrutinee); the motive is its lambda closure and the Fix type its Pi closure.
//   * Reverted hypotheses are Pi binders of G (bound variables), peeled one at a time with the next context variable.
//   * No has_internal_binder_ref guards: embedded binders cannot collide with context variables any more.
//   * Field types are mapped from telescope levels to their ACTUAL context levels (induct interleaves an IH after each recursive
//     field; the reference silently assumed consecutive levels).
//   * The exploited (refuted-sibling) branches of cases are built well-typed.
// ===========================================================================
inline NodeId close_binders(bool pi, const std::vector<NodeId>& doms, const std::vector<uint32_t>& fvl, NodeId body) {
    const std::size_t m = doms.size();
    NodeId result = abstract_fvars(body, fvl);
    for (int j = static_cast<int>(m) - 1; j >= 0; --j) {
        NodeId d = abstract_fvars(doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + j));
        result = pi ? mk_pi(d, result) : mk_lambda(d, result);
    }
    return result;
}

/// same_ctor : F -> F -> Type0   (Unit when both arguments share a constructor head, Empty otherwise); non-indexed families only.
inline uint32_t get_or_build_same_ctor(uint32_t family_gid) {
    auto cached = ctx().same_ctor_gid_by_family.find(family_gid);
    if (cached != ctx().same_ctor_gid_by_family.end()) return cached->second;
    NodeId family_ty = mk_const(family_gid), type0_v = mk_type(0);
    NodeId empty_ty = mk_const(get_or_build_empty_type()), unit_ty = mk_const(get_or_build_unit_type().first);
    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(family_gid);
    const uint32_t n = static_cast<uint32_t>(ctor_gids.size());
    NodeId const_type0_motive = mk_lambda(family_ty, type0_v);
    std::vector<NodeId> outer_branches(n);
    for (uint32_t i = 0; i < n; ++i) {
        detail::InstantiatedCtor iinst = detail::instantiate_ctor_telescope(ctor_gids[i], 2);
        const uint32_t oa = static_cast<uint32_t>(iinst.arg_types.size());
        std::vector<NodeId> inner_branches(n);
        for (uint32_t jb = 0; jb < n; ++jb) {
            detail::InstantiatedCtor jinst = detail::instantiate_ctor_telescope(ctor_gids[jb], 2 + oa);
            inner_branches[jb] = close_lambda_telescope(jinst.arg_types, jb == i ? unit_ty : empty_ty, 2 + oa);
        }
        outer_branches[i] = close_lambda_telescope(iinst.arg_types, mk_match(mk_fvar(1), const_type0_motive, inner_branches), 2);
    }
    NodeId body = close_lambda_telescope({family_ty, family_ty}, mk_match(mk_fvar(0), const_type0_motive, outer_branches), 0);
    NodeId ty = mk_pi(family_ty, mk_pi(family_ty, type0_v));
    auto declared = declare_checked(ty, body, "__same_ctor_" + std::to_string(family_gid));
    if (!declared) throw std::logic_error("itp: internal error -- synthesized same_ctor discriminator for family " + std::to_string(family_gid) + " failed to type-check: " + declared.error().message);
    ctx().same_ctor_gid_by_family[family_gid] = *declared;
    return *declared;
}

namespace detail {
struct ScrutInfo { uint32_t type_gid = 0; std::vector<NodeId> indices; };
}
inline std::expected<detail::ScrutInfo, TypeError> scrutinee_info(const char* who, const Goal& g, uint32_t hyp_level) {
    NodeId scrut_ty = g.hyps[hyp_level].type;
    Value v = eval(scrut_ty, g.ec.values);
    auto* nt = std::get_if<VNeutral>(&v);
    if (!nt || nt->head.kind != NeutralHead::Kind::GlobalConst)
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, scrut_ty, std::nullopt, std::string(who) + ": hypothesis's type is not an inductive type application"});
    detail::ScrutInfo si; si.type_gid = nt->head.id;
    for (auto& frame : nt->spine) {
        auto* af = std::get_if<ApplyFrame>(&frame);
        if (!af) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, scrut_ty, std::nullopt, std::string(who) + ": scrutinee's type has a non-application eliminator frame; cannot enumerate its indices"});
        si.indices.push_back(quote(af->arg, g.ec.size()));
    }
    return si;
}

/// G = Pi (reverted hypotheses). target, with the indices and the scrutinee replaced by the context variables n .. n+arity.
inline NodeId generalized_goal(const Goal& g, const std::vector<std::pair<NodeId, NodeId>>& subs, const std::vector<uint32_t>& reverted_js) {
    std::vector<NodeId> doms;
    for (uint32_t j : reverted_js) doms.push_back(replace_terms(g.hyps[j].type, subs));
    return close_binders(true, doms, reverted_js, replace_terms(g.target, subs));
}

inline ElabResult cases(ProofState& ps, uint32_t hyp_level, std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return no_goal_error("cases", target_meta_id);
    Goal g = ps.goals[*idx_opt];
    if (hyp_level >= g.hyps.size()) return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt, "cases: hypothesis index out of range"});
    if (g.target.tag == NodeType::PI)
        return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt, "cases: goal's own target is still a Pi -- intro every quantified variable before casing"});
    auto si = scrutinee_info("cases", g, hyp_level); if (!si) return std::unexpected(si.error());
    const uint32_t n = g.ec.size(), tgid = si->type_gid, arity = static_cast<uint32_t>(si->indices.size());
    const std::vector<NodeId>& concrete_indices = si->indices;
    std::vector<std::pair<NodeId, NodeId>> subs;
    for (uint32_t i = 0; i < arity; ++i) subs.push_back({concrete_indices[i], mk_fvar(n + i)});
    subs.push_back({mk_fvar(hyp_level), mk_fvar(n + arity)});
    // cases reverts a CONTIGUOUS suffix: from the earliest hypothesis that mentions an index or the scrutinee, to the end
    uint32_t revert_from = hyp_level + 1;
    for (uint32_t j = 0; j < hyp_level; ++j) if (replace_terms(g.hyps[j].type, subs) != g.hyps[j].type) { revert_from = j; break; }
    std::vector<uint32_t> reverted_js;
    for (uint32_t j = revert_from; j < g.hyps.size(); ++j) if (j != hyp_level) reverted_js.push_back(j);
    NodeId G = generalized_goal(g, subs, reverted_js);
    std::vector<NodeId> index_types = detail::instantiate_type_telescope(tgid, n);
    if (index_types.size() != arity)
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "cases: index count mismatch between the hypothesis's type application and the inductive's own declared arity"});
    std::vector<NodeId> idx_vars; for (uint32_t i = 0; i < arity; ++i) idx_vars.push_back(mk_fvar(n + i));
    std::vector<NodeId> mdoms = index_types; mdoms.push_back(apply_all(mk_const(tgid), idx_vars));
    NodeId motive_original = close_lambda_telescope(mdoms, G, n);
    if (!infer(motive_original, g.ec, ps.es))
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "cases: constructed motive failed to type-check -- the goal likely depends on the hypothesis in a way that cannot be abstracted cleanly"});
    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(tgid);
    std::vector<detail::InstantiatedCtor> insts; std::vector<CtorIndexForcing> forcings;
    for (uint32_t ctor_gid : ctor_gids) {
        detail::InstantiatedCtor inst = detail::instantiate_ctor_telescope(ctor_gid, n);
        CtorIndexForcing forcing = unify_ctor_pattern_indices(inst, concrete_indices, n, static_cast<uint32_t>(inst.arg_types.size()), g.ec.values);
        if (!forcing.ok) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, forcing.error});
        insts.push_back(std::move(inst)); forcings.push_back(std::move(forcing));
    }
    // refutation exploitation (only one constructor per call, and only when nothing is reverted)
    std::optional<std::size_t> exploit_ci;
    for (std::size_t ci = 0; ci < forcings.size(); ++ci) {
        const CtorIndexForcing& f = forcings[ci];
        if (f.refuted && f.refuted_slot && f.refuted_pat_value && f.refuted_conc_value) { exploit_ci = ci; break; }
    }
    if (!reverted_js.empty()) exploit_ci.reset();
    NodeId motive_final = motive_original;
    if (exploit_ci) {
        const uint32_t slot = *forcings[*exploit_ci].refuted_slot, family_gid = *forcings[*exploit_ci].refuted_family_gid;
        NodeId component = apply_all(mk_const(get_or_build_same_ctor(family_gid)), {idx_vars[slot], concrete_indices[slot]});
        NodeId m = close_lambda_telescope(mdoms, mk_pi(component, G), n);
        if (infer(m, g.ec, ps.es)) motive_final = m; else exploit_ci.reset();
    }
    std::vector<NodeId> branches; std::vector<Goal> new_goals;
    for (std::size_t ci = 0; ci < ctor_gids.size(); ++ci) {
        const uint32_t ctor_gid = ctor_gids[ci];
        const detail::InstantiatedCtor& inst = insts[ci]; const CtorIndexForcing& forcing = forcings[ci];
        const uint32_t ctor_arity = static_cast<uint32_t>(inst.arg_types.size());
        std::vector<NodeId> arg_types = apply_ctor_index_forcing(inst.arg_types, forcing);
        NodeId ctor_app = forced_ctor_app(ctor_gid, n, ctor_arity, inst, forcing);
        std::vector<NodeId> forced_result_indices = apply_ctor_index_forcing_to_result_indices(inst.result_indices, forcing);
        NodeId branch_expected = mk_app(apply_all(motive_original, forced_result_indices), ctor_app);
        Env be_env = g.ec.values;
        for (uint32_t ai = 0; ai < ctor_arity; ++ai) be_env = be_env.extend(fresh_neutral(n + ai));
        std::vector<NodeId> reverted_dom_types;
        if (!reverted_js.empty()) {   // the reverted layers are normalised with FULL delta, exactly as the reference does
            Value cur_val = eval(branch_expected, be_env);
            for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
                auto* pi = std::get_if<VPi>(&cur_val);
                if (!pi) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "cases: internal error -- expected a Pi layer for a reverted hypothesis but found none"});
                const uint32_t target_level = n + ctor_arity + static_cast<uint32_t>(ri);
                reverted_dom_types.push_back(quote(eval(pi->dom_id, pi->env), target_level));
                Value next = eval(pi->cod_id, pi->env.extend(fresh_neutral(target_level)));
                cur_val = std::move(next);
            }
            branch_expected = quote(cur_val, n + ctor_arity + static_cast<uint32_t>(reverted_js.size()));
        } else {                      // nothing reverted: beta/iota only, definitions stay folded (F4)
            NoDeltaScope keep_constants_folded;
            branch_expected = quote(eval(branch_expected, be_env), n + ctor_arity);
        }
        if (exploit_ci && ci == *exploit_ci) {
            const uint32_t family_gid = *forcing.refuted_family_gid, absurd_level = n + ctor_arity;
            NodeId absurd_ty = apply_all(mk_const(get_or_build_same_ctor(family_gid)), {*forcing.refuted_pat_value, *forcing.refuted_conc_value});
            NodeId reduced = quote(eval(branch_expected, be_env), absurd_level);
            NodeId absurd_body = mk_match(mk_fvar(absurd_level), mk_lambda(mk_const(get_or_build_empty_type()), reduced), {});
            std::vector<NodeId> doms = arg_types; doms.push_back(absurd_ty);
            branches.push_back(close_lambda_telescope(doms, absurd_body, n));
            continue;
        }
        ElabCtx branch_ec = g.ec; std::vector<Hypothesis> branch_hyps = g.hyps;
        for (NodeId field_ty : arg_types) { branch_ec = branch_ec.extend(field_ty); branch_hyps.push_back(Hypothesis{"", field_ty}); }
        for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) { branch_ec = branch_ec.extend(reverted_dom_types[ri]); branch_hyps.push_back(Hypothesis{g.hyps[reverted_js[ri]].name, reverted_dom_types[ri]}); }
        uint32_t subgoal_meta = ctx().meta_ctx.fresh(close_pi_over_context(branch_expected, branch_ec), branch_ec.size());
        NodeId subgoal_ref = meta_applied_to_context(subgoal_meta, branch_ec.size());
        std::vector<NodeId> doms = arg_types;
        if (exploit_ci) {   // the extended motive adds a same_ctor premise after the fields; this branch ignores it
            const uint32_t slot = *forcings[*exploit_ci].refuted_slot, family_gid = *forcings[*exploit_ci].refuted_family_gid;
            doms.push_back(apply_all(mk_const(get_or_build_same_ctor(family_gid)), {inst.result_indices[slot], concrete_indices[slot]}));
        } else for (NodeId d : reverted_dom_types) doms.push_back(d);
        branches.push_back(close_lambda_telescope(doms, subgoal_ref, n));
        new_goals.push_back(Goal{subgoal_meta, branch_ec, std::move(branch_hyps), branch_expected});
    }
    NodeId match_term = mk_match(mk_fvar(hyp_level), motive_final, branches);
    if (exploit_ci) match_term = mk_app(match_term, mk_const(get_or_build_unit_type().second));
    for (uint32_t rj : reverted_js) match_term = mk_app(match_term, mk_fvar(rj));
    ctx().meta_ctx.solve(g.meta_id, close_over_context(match_term, g.ec));
    ps.goals.erase(ps.goals.begin() + *idx_opt);
    ps.goals.insert(ps.goals.begin() + *idx_opt, new_goals.begin(), new_goals.end());
    return {};
}

inline ElabResult induct(ProofState& ps, uint32_t hyp_level, std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return no_goal_error("induct", target_meta_id);
    Goal g = ps.goals[*idx_opt];
    if (hyp_level >= g.hyps.size()) return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt, "induct: hypothesis index out of range"});
    if (g.target.tag == NodeType::PI)
        return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt, "induct: goal's own target is still a Pi -- intro every quantified variable before inducting"});
    auto si = scrutinee_info("induct", g, hyp_level); if (!si) return std::unexpected(si.error());
    const uint32_t n = g.ec.size(), tgid = si->type_gid, arity = static_cast<uint32_t>(si->indices.size());
    const std::vector<NodeId>& concrete_indices = si->indices;
    {   // Induction needs something to recurse on. The reference never accepted induct on a non-recursive family (e.g. Eq): its level-based
        // guard tripped over a marker/self-level collision. The principled rule gives the same behaviour with an honest message.
        bool any_recursive = false;
        for (uint32_t cg : detail::ctor_gids_for_type(tgid)) for (bool b : recursive_field_mask(tgid, cg)) any_recursive = any_recursive || b;
        if (!any_recursive)
            return std::unexpected(TypeError{TypeErrorKind::Other, g.hyps[hyp_level].type, std::nullopt, "induct: the hypothesis's inductive type has no recursive constructor field -- use cases"});
    }
    std::vector<std::pair<NodeId, NodeId>> subs;
    for (uint32_t i = 0; i < arity; ++i) subs.push_back({concrete_indices[i], mk_fvar(n + i)});
    subs.push_back({mk_fvar(hyp_level), mk_fvar(n + arity)});
    // induct reverts exactly the hypotheses that (transitively) depend on the scrutinee or its indices: a fixpoint, not a suffix
    std::vector<bool> needs_revert(g.hyps.size(), false);
    for (bool changed = true; changed;) {
        changed = false;
        std::vector<std::pair<NodeId, NodeId>> probe = subs;
        for (uint32_t j = 0; j < g.hyps.size(); ++j) if (needs_revert[j]) probe.push_back({mk_fvar(j), mk_fvar(1'000'000'000u + j)});
        for (uint32_t j = 0; j < g.hyps.size(); ++j) {
            if (j == hyp_level || needs_revert[j]) continue;
            if (replace_terms(g.hyps[j].type, probe) != g.hyps[j].type) { needs_revert[j] = true; changed = true; }
        }
    }
    std::vector<uint32_t> reverted_js;
    for (uint32_t j = 0; j < g.hyps.size(); ++j) if (needs_revert[j]) reverted_js.push_back(j);
    NodeId G = generalized_goal(g, subs, reverted_js);
    std::vector<NodeId> index_types = detail::instantiate_type_telescope(tgid, n);
    if (index_types.size() != arity)
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "induct: index count mismatch between the hypothesis's type application and the inductive's own declared arity"});
    std::vector<NodeId> idx_vars; for (uint32_t i = 0; i < arity; ++i) idx_vars.push_back(mk_fvar(n + i));
    std::vector<NodeId> mdoms = index_types; mdoms.push_back(apply_all(mk_const(tgid), idx_vars));
    NodeId motive = close_lambda_telescope(mdoms, G, n);
    if (!infer(motive, g.ec, ps.es))
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "induct: constructed motive failed to type-check -- the goal likely depends on the hypothesis in a way that cannot be abstracted cleanly"});
    NodeId fn_type = close_pi_telescope(mdoms, G, n);
    // runtime scope of the Fix body:  self = n, index parameters n+1.., scrutinee parameter n+1+arity, constructor fields from body_base
    const uint32_t self_level = n, scrutinee_level = n + 1 + arity, body_base = scrutinee_level + 1;
    std::vector<NodeId> idx_vars_body; for (uint32_t i = 0; i < arity; ++i) idx_vars_body.push_back(mk_fvar(n + 1 + i));
    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(tgid);
    std::vector<NodeId> branches; std::vector<Goal> new_goals;
    for (uint32_t ctor_gid : ctor_gids) {
        // ---- the goal the user sees
        detail::InstantiatedCtor inst_agent = detail::instantiate_ctor_telescope(ctor_gid, n);
        const uint32_t ctor_arity = static_cast<uint32_t>(inst_agent.arg_types.size());
        std::vector<bool> rec_mask = recursive_field_mask(tgid, ctor_gid);
        CtorIndexForcing forcing_agent = unify_ctor_pattern_indices(inst_agent, concrete_indices, n, ctor_arity, g.ec.values);
        if (!forcing_agent.ok) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, forcing_agent.error});
        std::vector<NodeId> arg_types_agent = apply_ctor_index_forcing(inst_agent.arg_types, forcing_agent);
        ElabCtx branch_ec = g.ec; std::vector<Hypothesis> branch_hyps = g.hyps;
        std::vector<uint32_t> actual(ctor_arity);
        auto to_actual = [&](NodeId t, uint32_t upto) { std::unordered_map<uint32_t, NodeId> m; for (uint32_t j = 0; j < upto; ++j) m[n + j] = mk_fvar(actual[j]); return replace_fvars(t, m); };
        for (uint32_t fi = 0; fi < ctor_arity; ++fi) {
            NodeId fty = to_actual(arg_types_agent[fi], fi);
            actual[fi] = branch_ec.size();
            branch_ec = branch_ec.extend(fty); branch_hyps.push_back(Hypothesis{"", fty});
            if (fi < rec_mask.size() && rec_mask[fi]) {   // the induction hypothesis follows its recursive field immediately
                detail::AppSpine sp = detail::decompose_app_spine(fty);
                std::unordered_map<uint32_t, NodeId> m;
                for (uint32_t j = 0; j < arity && j < sp.args.size(); ++j) m[n + j] = sp.args[j];
                m[n + arity] = mk_fvar(actual[fi]);
                NodeId ih_ty = replace_fvars(G, m);
                branch_ec = branch_ec.extend(ih_ty); branch_hyps.push_back(Hypothesis{"", ih_ty});
            }
        }
        NodeId agent_ctor_app = forced_ctor_app(ctor_gid, n, ctor_arity, inst_agent, forcing_agent, &actual);
        std::vector<NodeId> forced_agent_result_indices = apply_ctor_index_forcing_to_result_indices(inst_agent.result_indices, forcing_agent);
        std::unordered_map<uint32_t, NodeId> bm;
        for (uint32_t j = 0; j < arity; ++j) bm[n + j] = to_actual(forced_agent_result_indices[j], ctor_arity);
        bm[n + arity] = agent_ctor_app;
        NodeId branch_expected = replace_fvars(G, bm);
        std::vector<NodeId> reverted_dom_types;
        if (!reverted_js.empty()) {
            const uint32_t base_level = branch_ec.size();
            NodeId remaining = branch_expected;
            for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
                if (remaining.tag != NodeType::PI) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "induct: internal error -- expected a Pi layer for a reverted hypothesis but found none"});
                Pi p = ctx().pis[remaining.index];
                reverted_dom_types.push_back(p.dom_id);
                remaining = instantiate1(p.cod_id, mk_fvar(base_level + static_cast<uint32_t>(ri)));
            }
            branch_expected = remaining;
            for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) { branch_ec = branch_ec.extend(reverted_dom_types[ri]); branch_hyps.push_back(Hypothesis{g.hyps[reverted_js[ri]].name, reverted_dom_types[ri]}); }
        }
        uint32_t subgoal_meta = ctx().meta_ctx.fresh(close_pi_over_context(branch_expected, branch_ec), branch_ec.size());
        // ---- the runtime branch (a lambda over the fields and the reverted hypotheses; no IH binders, the IH is the recursive call)
        detail::InstantiatedCtor inst = detail::instantiate_ctor_telescope(ctor_gid, body_base);
        CtorIndexForcing forcing_body = unify_ctor_pattern_indices(inst, idx_vars_body, body_base, ctor_arity, g.ec.values);
        if (!forcing_body.ok) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, forcing_body.error});
        std::vector<NodeId> arg_types_body = apply_ctor_index_forcing(inst.arg_types, forcing_body);
        std::vector<NodeId> forced_body_result_indices = apply_ctor_index_forcing_to_result_indices(inst.result_indices, forcing_body);
        NodeId true_ctor_app = forced_ctor_app(ctor_gid, body_base, ctor_arity, inst, forcing_body);
        NodeId applied = mk_meta(subgoal_meta);
        for (uint32_t i = 0; i < n; ++i) {
            NodeId supplied = mk_fvar(i);
            if (i == hyp_level) supplied = true_ctor_app;
            else {
                bool replaced = false;
                for (uint32_t j = 0; j < arity && !replaced; ++j) {
                    if (concrete_indices[j].tag == NodeType::VAR) { Var v = ctx().vars[concrete_indices[j].index]; if (!v.bound && v.n == i) { supplied = forced_body_result_indices[j]; replaced = true; } }
                }
                if (!replaced) for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) if (reverted_js[ri] == i) { supplied = mk_fvar(body_base + ctor_arity + static_cast<uint32_t>(ri)); break; }
            }
            applied = mk_app(applied, supplied);
        }
        for (uint32_t fi = 0; fi < ctor_arity; ++fi) {
            applied = mk_app(applied, mk_fvar(body_base + fi));
            if (fi < rec_mask.size() && rec_mask[fi]) {
                NodeId ih_call = mk_fvar(self_level);
                detail::AppSpine sp = detail::decompose_app_spine(arg_types_body[fi]);
                for (uint32_t j = 0; j < arity && j < sp.args.size(); ++j) ih_call = mk_app(ih_call, sp.args[j]);
                applied = mk_app(applied, mk_app(ih_call, mk_fvar(body_base + fi)));
            }
        }
        for (uint32_t ri = 0; ri < reverted_js.size(); ++ri) applied = mk_app(applied, mk_fvar(body_base + ctor_arity + ri));
        std::vector<NodeId> bdoms = arg_types_body;
        if (!reverted_js.empty()) {
            std::unordered_map<uint32_t, NodeId> rm;
            for (uint32_t j = 0; j < arity; ++j) rm[n + j] = forced_body_result_indices[j];
            rm[n + arity] = true_ctor_app;
            NodeId remaining_body = replace_fvars(G, rm);
            for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
                Pi p = ctx().pis[remaining_body.index];
                bdoms.push_back(p.dom_id);
                remaining_body = instantiate1(p.cod_id, mk_fvar(body_base + ctor_arity + static_cast<uint32_t>(ri)));
            }
        }
        branches.push_back(close_lambda_telescope(bdoms, applied, body_base));
        new_goals.push_back(Goal{subgoal_meta, branch_ec, std::move(branch_hyps), branch_expected});
    }
    NodeId match_term = mk_match(mk_fvar(scrutinee_level), motive, branches);
    std::vector<NodeId> body_doms = detail::instantiate_type_telescope(tgid, n + 1);
    std::vector<NodeId> body_idx; for (uint32_t i = 0; i < arity; ++i) body_idx.push_back(mk_fvar(n + 1 + i));
    body_doms.push_back(apply_all(mk_const(tgid), body_idx));
    NodeId fix_body = abstract_fvars(close_lambda_telescope(body_doms, match_term, n + 1), std::vector<uint32_t>{self_level});
    Fix fix_node{fn_type, arity, fix_body};
    if (!fix_guard_ok(fix_node, n))
        return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt,
            "induct: the assembled recursive proof failed the local structural recursion check -- some recursive call does not recurse on a structurally smaller instance of the hypothesis's own type"});
    NodeId proof = mk_fix(fn_type, arity, fix_body);
    for (NodeId idx : concrete_indices) proof = mk_app(proof, idx);
    proof = mk_app(proof, mk_fvar(hyp_level));
    for (uint32_t rj : reverted_js) proof = mk_app(proof, mk_fvar(rj));
    ctx().meta_ctx.solve(g.meta_id, close_over_context(proof, g.ec));
    ps.goals.erase(ps.goals.begin() + *idx_opt);
    ps.goals.insert(ps.goals.begin() + *idx_opt, new_goals.begin(), new_goals.end());
    return {};
}


// ===========================================================================
// S5: the library index (candidate lemmas for `apply`).
// The reference keyed a variable-headed conclusion on the variable's absolute LEVEL. A bound variable of a lemma type had level
// (its binder's position); a context variable has its own level. With de Bruijn indices the same key is recovered as
//     level = base + depth - 1 - index      (bound variable `depth` binders below a context of `base` variables)
// where base = 0 for a lemma type and the context size (or, for a hypothesis, its own position) otherwise. Context variables key on their level.
// ===========================================================================
namespace detail {
inline uint64_t pack_head_key(HeadKind kind, uint32_t id) {
    return (static_cast<uint64_t>(kind) << 32) | static_cast<uint64_t>(id);
}
struct PeeledConclusion { NodeId head; std::vector<NodeId> args; uint32_t depth = 0; };   // args in REVERSE application order, as in the reference
inline PeeledConclusion peel_conclusion(NodeId ty) {
    PeeledConclusion pc; NodeId cur = ty;
    while (true) {
        auto next = inspect(cur, [](Pi p) -> std::optional<NodeId> { return p.cod_id; }, [](auto) -> std::optional<NodeId> { return std::nullopt; });
        if (!next) break;
        cur = *next; ++pc.depth;
    }
    pc.head = cur;
    bool progressed = true;
    while (progressed) {
        progressed = false;
        inspect(pc.head, [&](App a) { pc.head = a.fn_id; pc.args.push_back(a.arg_id); progressed = true; }, [](auto) {});
    }
    return pc;
}
inline std::optional<uint64_t> head_key_at(NodeId h, uint32_t base, uint32_t depth) {
    return inspect(h,
        [](Const c) -> std::optional<uint64_t> { return pack_head_key(HeadKind::GlobalConst, c.global_id); },
        [&](Var v) -> std::optional<uint64_t> {
            if (!v.bound) return pack_head_key(HeadKind::LocalVar, v.n);
            if (v.n >= depth) return std::nullopt;                       // loose bound variable: not a well-formed head here
            return pack_head_key(HeadKind::LocalVar, base + depth - 1 - v.n);
        },
        [](auto) -> std::optional<uint64_t> { return std::nullopt; });
}
inline std::optional<uint64_t> conclusion_head_key(NodeId ty, uint32_t base = 0) {
    PeeledConclusion pc = peel_conclusion(ty);
    return head_key_at(pc.head, base, pc.depth);
}
inline std::optional<uint64_t> library_index_key(NodeId ty, uint32_t base = 0) {
    PeeledConclusion pc = peel_conclusion(ty);
    auto outer = head_key_at(pc.head, base, pc.depth);
    if (!outer) return std::nullopt;
    uint64_t combined = *outer;
    for (NodeId a : pc.args) {
        NodeId h = a; bool p = true;
        while (p) { p = false; inspect(h, [&](App ap) { h = ap.fn_id; p = true; }, [](auto) {}); }
        auto k = head_key_at(h, base, pc.depth);
        uint64_t part = k.value_or(0xdeadbeefULL);   // an un-headed argument (Meta, Type, ...) still contributes a fixed, distinct part
        combined ^= part + 0x9e3779b97f4a7c15ULL + (combined << 6) + (combined >> 2);   // boost-style hash-combine
    }
    return combined;
}
inline bool library_index_key_is_generic(NodeId ty) {
    PeeledConclusion pc = peel_conclusion(ty);
    auto is_var_headed = [](NodeId t) -> bool {
        NodeId h = t; bool p = true;
        while (p) { p = false; inspect(h, [&](App a) { h = a.fn_id; p = true; }, [](auto) {}); }
        bool result = false;
        inspect(h, [&](Var) { result = true; }, [](auto) {});
        return result;
    };
    if (is_var_headed(pc.head)) return true;
    for (NodeId a : pc.args) if (is_var_headed(a)) return true;
    return false;
}
} // namespace detail

struct DiscriminationTree {
    std::unordered_map<uint64_t, std::vector<NodeId>> by_head;
    std::unordered_map<uint64_t, std::vector<NodeId>> generic_by_outer_head;
    void insert(NodeId candidate_term, NodeId candidate_type) {
        if (detail::library_index_key_is_generic(candidate_type)) {
            if (auto key = detail::conclusion_head_key(candidate_type)) generic_by_outer_head[*key].push_back(candidate_term);
            return;
        }
        if (auto key = detail::library_index_key(candidate_type)) by_head[*key].push_back(candidate_term);
    }
    std::vector<NodeId> candidates_for(NodeId goal_type, uint32_t base = 0) const {
        std::vector<NodeId> result;
        if (auto key = detail::library_index_key(goal_type, base)) {
            auto it = by_head.find(*key);
            if (it != by_head.end()) result.insert(result.end(), it->second.begin(), it->second.end());
        }
        if (auto key = detail::conclusion_head_key(goal_type, base)) {
            auto it = generic_by_outer_head.find(*key);
            if (it != generic_by_outer_head.end()) result.insert(result.end(), it->second.begin(), it->second.end());
        }
        return result;
    }
};
inline DiscriminationTree index_library() {
    DiscriminationTree tree;
    for (uint32_t gid = 0; gid < ctx().globals.defs.size(); ++gid) tree.insert(mk_const(gid), ctx().globals.defs[gid].type);
    return tree;
}
inline std::vector<NodeId> apply_candidates(const ProofState& ps, const DiscriminationTree& library, std::optional<uint32_t> target_meta_id = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return {};
    const Goal& g = ps.goals[*idx_opt];
    std::vector<NodeId> result = library.candidates_for(g.target, g.ec.size());
    auto goal_key = detail::conclusion_head_key(g.target, g.ec.size());
    for (uint32_t lvl = 0; lvl < g.hyps.size(); ++lvl)
        if (goal_key && detail::conclusion_head_key(g.hyps[lvl].type, lvl) == goal_key) result.push_back(mk_fvar(lvl));
    return result;
}

} // namespace itp

// -----------------------------------------------------------------------
// ITP_TRY
// -----------------------------------------------------------------------
//
// std::expected already provides and_then/or_else/transform (C++23), which
// gives short-circuiting error propagation without any additional code in
// this library. What that interface does not provide is a flat sequential
// syntax: each and_then callback only sees the immediately preceding
// value, so keeping several earlier results in scope at once requires
// nesting one callback inside another.
//
// ITP_TRY addresses the syntax rather than the mechanics, in the same
// spirit as Abseil's ASSIGN_OR_RETURN: given an expression that evaluates
// to a std::expected, it either binds the contained value to a new
// variable or returns the error from the enclosing function. This is the
// one macro in this header; there is no macro-free way in current C++ to
// express "return from the enclosing function" as part of an expression.
//
// Usage, inside a function returning std::expected<_, E>:
//
//   TypeResult infer_app(NodeId fn, NodeId arg) {
//       ITP_TRY(NodeId fn_ty, infer(fn));
//       ITP_TRY(itp::Pi p, itp::require(itp::try_match<itp::Pi>(fn_ty),
//                   itp::TypeError{itp::TypeErrorKind::NotAFunction, fn, std::nullopt, "not a function"}));
//       ITP_TRY(NodeId arg_ty, infer(arg));
//       if (!itp::conv(arg_ty, p.dom_id, {}))
//           return std::unexpected(itp::TypeError{itp::TypeErrorKind::Mismatch, arg, p.dom_id, "argument type mismatch"});
//       return p.cod_id;
//   }
//
// ITP_TRY expands to multiple statements and declares a variable meant to
// escape into the surrounding scope, so it cannot be used as the unbraced
// body of an if/for/while. Doing so does not compile silently incorrectly;
// it fails to compile, since the declared temporary falls out of scope
// before the macro's later lines reference it.
