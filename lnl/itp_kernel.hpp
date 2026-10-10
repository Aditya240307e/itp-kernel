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
                                  META, TYPE_UNIV, LET, FIX };

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
/// `Match::motive_id`, which is retained for faithful
/// reconstruction/printing but never consulted during ordinary
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

/// Which quotient primitive a global is (see declare_quot). A role is assigned ONLY inside declare_quot, which builds the global's type
/// itself. Mk and Lift drive the one reduction rule (apply); Type, Ind and Sound are recorded so that the independent checker can find
/// the whole family and re-verify every type.
enum class QuotRole : uint8_t { None = 0, Type, Mk, Lift, Ind, Sound };

struct GlobalDef {
    NodeId type;
    std::optional<NodeId> value;       // present for a definition (delta-reducible); absent for an axiom
    std::optional<CtorInfo> ctor_info; // present if this global is a constructor
    std::optional<RecursiveFnInfo> recursive_info; // present if declared via finish_recursive_function(_checked)
    std::string name;
    QuotRole quot_role = QuotRole::None; // set only by declare_quot
    // Set ONLY when the global is finished as an inductive family (finish_inductive_unchecked / finish_mutual_inductive_block), including a
    // family with no constructors. Match, cases and induct are meaningful only on such families. An opaque type constant -- an axiom whose type
    // ends in a sort, or Quot -- is NOT inductive: treating it as one would make a branchless match on any of its inhabitants a proof of
    // False (finding F26).
    bool is_inductive = false;
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
    std::unordered_map<uint64_t, std::vector<bool>> param_cache; // family_param_slots memo, keyed by (family, constructor count); cleared on rewind
    bool param_mode = false; // option "parameters": distinguish a family's PARAMETERS from its indices (family_param_slots); off = the reference's behaviour
    bool prop_mode = false;  // option "prop_elim": level 0 is Prop. An inductive family that lives in Type0 and is not a SYNTACTIC SUBSINGLETON (it may be eliminated into any sort only if it has no constructor, or one whose every
                             // field is a proof or occurs in the result indices) can only be eliminated into Type0. Large fields are then allowed in such families. Required before classical axioms (Berardi, Reynolds).
    std::unordered_set<uint32_t> prop_elim_only;   // the families (global ids) subject to that restriction; filled by finish_inductive in prop_mode

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
            {"fixes", fixes.size(), fixes.capacity()},
        };
    }

    struct Checkpoint {
        uint32_t vars_n, apps_n, lambdas_n, lets_n, pis_n, types_n, consts_n,
                 matches_n, metas_n, type_univs_n, fixes_n;
        MetaContext::Mark meta_mark;
        std::size_t globals_n; // safety check only; globals itself is never rewound
    };

    Checkpoint checkpoint() const {
        return Checkpoint{
            vars.size(), apps.size(), lambdas.size(), lets.size(), pis.size(), types.size(),
            consts.size(), matches.size(), metas.size(), type_univs.size(), fixes.size(),
            meta_ctx.checkpoint(), globals.defs.size()
        };
    }

    // Derived globals are the ones the kernel itself synthesises on demand, deterministically and
    // idempotently, for refutation exploitation (get_or_build_same_ctor / _unit_type / _empty_type):
    // `__same_ctor_<family>`, `__Unit` with its constructor `__tt`, `__Empty`, and the extraction barrier `__barrier_<level>` used by induct's equation path.
    static bool is_derived_global_name(const std::string& nm) {
        return nm.rfind("__same_ctor_", 0) == 0 || nm == "__Unit" || nm == "__tt" || nm == "__Empty" || nm.rfind("__barrier_", 0) == 0;   // __barrier_k: built by name on demand (get_or_build_barrier), so a probe may create and drop it
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
        loose_cache.clear(); param_cache.clear();
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
inline NodeId mk_fix(NodeId fn_type, uint32_t decreasing_arg, NodeId body) {
    return NodeId{NodeType::FIX, ctx().fixes.intern(Fix{fn_type, decreasing_arg, body})};
}

using Term = std::variant<Var, App, Lambda, Let, Pi, Type, Const, Match, Meta, TypeUniv, Fix>;

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
// VCtor, VFix) must all be fully defined before ApplyFrame
// is defined, since ApplyFrame holds a Value by value and therefore
// requires Value to be a complete type at that point.

struct Closure; struct VNeutral; struct VPi; struct VType; struct VCtor;
struct VFix;
struct ApplyFrame; struct MatchFrame;

using Value = std::variant<VType, Closure, VNeutral, VPi, VCtor, VFix>;
using EliminatorFrame = std::variant<ApplyFrame, MatchFrame>;

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
// This is required by the same mutual-recursion constraint as the
// Value alternatives above (EnvNode's structural definition only needs Value
// forward-declared this way, not complete, so it can be fully defined
// here rather than deferred like ApplyFrame/MatchFrame below) -- but it
// is also the right choice on its own terms: it keeps a leaf node's path-
// copy cost at a constant "32 pointer copies" regardless of how large or
// expensive Value itself is to copy, which matters because Value has
// gained alternatives over the life of this header (Meta, TypeUniv, Fix)
// and is expected to gain more; an unboxed design would have
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
// because Value is expensive and still growing (Meta, TypeUniv, Fix
// were all added after Env's leaf representation was chosen) --
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
/// pending applications and MatchFrame for a pending case split, since
/// each needs to be reconstructed and compared differently by quote and
/// conv_value.
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
/// theoretically possible route to a similar cascade; if a workload is
/// ever found that stresses one, it would need the same treatment,
/// applied there specifically.
struct VCtor {
    uint32_t ind_type_global_id;
    uint32_t ctor_idx;
    std::vector<Value> args;

    // All special members are DECLARED here but DEFINED later in the
    // file, after Value's alternatives are complete -- mirroring the
    // same forward-declare-then-define-
    // later pattern already used for Env's own methods below. This is
    // load-bearing, not stylistic: VCtor is itself one of Value's
    // alternatives, and
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
// theoretically possible route to a similar cascade; if a workload is
// ever found that stresses one, it would need the same treatment,
// applied there specifically.
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

/// The quotient computation rule, the one reduction rule that is not beta, delta or iota:
///
///     Quot.lift A r B f h (Quot.mk A' r' a)   ~>   f a
///
/// `n` is a neutral headed by a Quot.lift that has just received its sixth argument. The rule fires only when ALL six frames are plain
/// applications and the sixth is a saturated Quot.mk (a neutral headed by a global with role Mk, with exactly three plain arguments);
/// it returns (f, a). Nothing about h is looked at: that h proves f respects r is enforced where it belongs, by the ordinary
/// application typing rule against Quot.lift's declared type. A and r need not be compared with A' and r' either, because in a
/// well-typed term they are equal (subject reduction), exactly as for every other iota rule.
inline std::optional<std::pair<Value, Value>> quot_lift_redex(const VNeutral& n) {
    if (n.spine.size() != 6) return std::nullopt;
    for (const auto& fr : n.spine) if (!std::holds_alternative<ApplyFrame>(fr)) return std::nullopt;
    const auto* q = std::get_if<VNeutral>(&std::get<ApplyFrame>(n.spine[5]).arg);
    if (!q || q->head.kind != NeutralHead::Kind::GlobalConst || q->spine.size() != 3) return std::nullopt;
    if (ctx().globals[q->head.id].quot_role != QuotRole::Mk) return std::nullopt;
    for (const auto& fr : q->spine) if (!std::holds_alternative<ApplyFrame>(fr)) return std::nullopt;
    return std::make_pair(std::get<ApplyFrame>(n.spine[3]).arg, std::get<ApplyFrame>(q->spine[2]).arg);
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
                if (def.quot_role == QuotRole::Lift) {
                    if (auto redex = quot_lift_redex(n)) return apply(std::move(redex->first), std::move(redex->second));
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
                } else {
                    const auto* mf = &std::get<MatchFrame>(frame);
                    std::vector<NodeId> branches_q; branches_q.reserve(mf->branches.size());
                    for (NodeId b : mf->branches) branches_q.push_back(requote(b, mf->captured_env, n0, depth));
                    NodeId motive_q = quote_at(eval(mf->motive_id, mf->captured_env), n0, depth);
                    acc = mk_match(acc, motive_q, std::move(branches_q));
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
        // An unapplied recursive function is a function value too: (\x. F x) == F by the same eta step. (Without these two cases the kernel
        // was strictly less complete than the independent checker's eta rule, which covers every non-lambda.)
        [&](const Closure& ca, const VFix& fb) -> bool {
            Value fresh = fresh_neutral(depth);
            Value lhs = eval(ca.body_id, ca.env.extend(fresh));
            Value rhs = apply(Value{fb}, fresh);
            return conv_value(lhs, rhs, depth + 1);
        },
        [&](const VFix& fa, const Closure& cb) -> bool {
            Value fresh = fresh_neutral(depth);
            Value lhs = apply(Value{fa}, fresh);
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
/// frame anywhere in the spine disqualifies it. This is the
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
        if (!af) return std::nullopt; // MatchFrame in the spine: not a pattern
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

} // namespace detail

/// Core unification on already-evaluated Values. Structural cases mirror
/// conv_value's shape exactly, but recurse via unify_values (not
/// conv_value) wherever a subterm might hide a solvable metavariable, so
/// that e.g. `f ?M` unifying against `f x` solves `?M := x` even though
/// the outermost shape (App/App, i.e. matching VNeutral heads here) was
/// already rigid. Falls back to conv_value-equivalent hard equality for
/// MatchFrame spine elements specifically -- a deliberate,
/// documented completeness gap (a meta buried under a stuck Match
/// eliminator in this pass won't get solved), not a soundness
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
/// A non-ApplyFrame spine entry (a MatchFrame -- a meta
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
    // the same metavariable applied to the same arguments is trivially equal (the reference rejected ?m =?= ?m: the occurs check fired on itself)
    if (is_meta_headed(a) && is_meta_headed(b) && quote(a, ec.size()) == quote(b, ec.size())) return {};
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
        [&](const VFix& fa, const VFix& fb) -> ElabResult {
            // Hard syntactic equality, same documented completeness gap
            // as MatchFrame just below: fix_id equality
            // plus pointwise conv_value (not recursive unify_values) on
            // the captured environments -- a metavariable buried only
            // inside a captured Fix env won't get solved by this pass,
            // mirroring exactly how a meta buried in a stuck Match's
            // captured env already isn't solved here either.
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
    std::unordered_set<uint32_t> param_vars;     // fields whose value was determined by a PARAMETER slot: substituted everywhere, repeated occurrences included
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


} // namespace detail

namespace detail {

} // namespace detail

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
                                                    uint32_t base_depth, uint32_t ctor_arity, const Env& env,
                                                    const std::vector<bool>* param_slots = nullptr, const std::vector<bool>* skip_slots = nullptr) {
    CtorIndexForcing out;
    if (inst.result_indices.size() != concrete_indices.size()) {
        out.ok = false;
        out.error = "unify_ctor_pattern_indices: index count mismatch between the constructor's own "
                     "pattern and the scrutinee's concrete indices";
        return out;
    }
    for (std::size_t j = 0; j < inst.result_indices.size(); ++j) {
        if (skip_slots && j < skip_slots->size() && (*skip_slots)[j]) continue;      // a slot generalised through an equation gives no forcing information
        match_index_pattern(inst.result_indices[j], concrete_indices[j], base_depth, ctor_arity, env, out);
        if (param_slots && j < param_slots->size() && (*param_slots)[j] && inst.result_indices[j].tag == NodeType::VAR) {
            Var pv = ctx().vars[inst.result_indices[j].index];
            if (!pv.bound && pv.n >= base_depth && pv.n < base_depth + ctor_arity) out.param_vars.insert(pv.n);
        }
        if (!out.ok) break;
        if (out.refuted) { out.refuted_slot = static_cast<uint32_t>(j); break; }
    }
    return out;
}


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
inline ElabResult verify_motive_shape(NodeId motive_id, uint32_t arity, ElabCtx& ec, ElabState& es, uint32_t* sort_level = nullptr);

// ---- forward declarations for the S2 typing core (defined at the end of this file)
inline TypeResult infer(NodeId t, ElabCtx& ec, ElabState& es);
inline ElabResult check(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es);
inline ElabResult unify(NodeId a, NodeId b, ElabCtx& ec, ElabState& es);
inline NodeId to_levels(NodeId t, uint32_t depth);


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


// ---------------------------------------------------------------------------
// Apply: Tier 1 (discrimination tree) + Tier 2 (real unification)
// ---------------------------------------------------------------------------

namespace detail {

enum class HeadKind : uint32_t { GlobalConst = 0, LocalVar = 1 };

} // namespace detail

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

// ---------------------------------------------------------------------------

// Printing
// ---------------------------------------------------------------------------

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
            if (!parameter_like && field_level[j] > *fam_level && !(ctx().prop_mode && *fam_level == 0)) {
                throw std::logic_error("itp: constructor '" + ctor.name + "' of '" + fam + "': field " + std::to_string(j) +
                                       " lives in universe " + std::to_string(field_level[j]) + ", larger than the family's own "
                                       "universe " + std::to_string(*fam_level) + " (a large field in a small inductive is "
                                       "inconsistent with an impredicative universe)");
            }
        }
    }
}


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

/// F23. Proof extraction is quote(eval(..)), and eval's apply(VFix) ALWAYS unfolds one layer, even when the decreasing argument is a neutral variable (a standard fix unfolds only on a
/// constructor). Extraction therefore peels the Fix off an induct proof and leaves a Match on the CONCRETE hypothesis whose branches were typed for the GENERIC Fix parameter. For distinct
/// variable indices the two views coincide; for an index generalised through an equation they do not. `__barrier_k : Pi (A : Type k). A -> A := fun A x => x` is a transparent identity that
/// extraction (NoDeltaScope) leaves folded, so a Fix wrapped in it survives extraction and the kernel keeps checking the generic body. Conversion still sees through it (delta).
inline uint32_t get_or_build_barrier(uint32_t level) {
    const std::string name = "__barrier_" + std::to_string(level);
    for (uint32_t i = 0; i < ctx().globals.defs.size(); ++i) if (ctx().globals.defs[i].name == name) return i;
    NodeId sort = mk_type(level);
    return ctx().globals.declare(mk_pi(sort, mk_pi(mk_bvar(0), mk_bvar(1))), mk_lambda(sort, mk_lambda(mk_bvar(0), mk_bvar(0))), name);
}


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
        c.globals.defs[spec.type_global_id].is_inductive = true;
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

/// Reserves a global id for a recursive function before its body is
/// supplied, mirroring reserve_inductive_type: the body needs to
/// reference the function's own global id, which does not exist until it
/// has been at least partially declared.
inline uint32_t reserve_recursive_function(const std::string& name, NodeId type) {
    return ctx().globals.declare(type, std::nullopt, name);
}

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
/// prop_elim: decides, once the constructors are known, whether a Type0 family can only be eliminated into Type0. Subsingleton (unrestricted): no constructor, or exactly one whose every field is a proof (its
/// type lives in Type0) or occurs as a bare result index. Anything unexpected (an ill-typed field) is treated as restricted. Families above Type0 are data and never restricted.
inline void record_prop_elim_restriction(uint32_t type_gid, const std::vector<CtorSpec>& ctors) {
    if (!ctx().prop_mode) return;
    NodeId cur = ctx().globals.defs[type_gid].type;
    while (cur.tag == NodeType::PI) cur = ctx().pis[cur.index].cod_id;
    if (cur.tag != NodeType::TYPE || ctx().types[cur.index].universe_level != 0) return;
    bool subsingleton = ctors.size() <= 1;
    if (ctors.size() == 1) {
        const CtorSpec& ctor = ctors[0]; ElabState es; ElabCtx ec;
        for (std::size_t j = 0; j < ctor.arg_types.size() && subsingleton; ++j) {
            auto sr = infer(ctor.arg_types[j], ec, es);
            bool proof_field = false;
            if (sr) { Value sv = eval(*sr, ec.values); auto* vt = std::get_if<VType>(&sv); proof_field = vt && vt->universe_level == 0; }
            bool in_indices = false;
            for (NodeId ri : ctor.result_indices) if (inspect(ri, [&](Var v) { return !v.bound && v.n == j; }, [&](auto) { return false; })) { in_indices = true; break; }
            if (!proof_field && !in_indices) subsingleton = false;
            ec = ec.extend(ctor.arg_types[j]);
        }
    }
    if (!subsingleton) ctx().prop_elim_only.insert(type_gid);
}
inline InductiveHandle finish_inductive_unchecked(uint32_t type_global_id, std::vector<CtorSpec> ctors) {
    ExecutionContext& c = ctx();
    InductiveHandle handle; handle.type_global_id = type_global_id;
    for (std::size_t idx = 0; idx < ctors.size(); ++idx) {
        NodeId ctor_ty = build_ctor_type(type_global_id, ctors[idx]);
        uint32_t arity = static_cast<uint32_t>(ctors[idx].arg_types.size());
        handle.ctor_global_ids.push_back(c.globals.declare_ctor(ctor_ty, ctors[idx].name, type_global_id, static_cast<uint32_t>(idx), arity));
    }
    record_prop_elim_restriction(type_global_id, ctors);
    c.globals.defs[type_global_id].is_inductive = true;
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

// ---------------------------------------------------------------------------
// Quotients
// ---------------------------------------------------------------------------
//
// A quotient is not a term former. It is five ordinary global constants -- whose types THIS function builds, so nobody can hand the
// kernel different ones -- and one reduction rule (quot_lift_redex, applied in apply()). With Eq : (A : Type j) -> A -> A -> Prop:
//
//   Quot       : (A : Type j) -> (A -> A -> Prop) -> Type j
//   Quot.mk    : (A : Type j) -> (r : A -> A -> Prop) -> A -> Quot A r
//   Quot.lift  : (A : Type j) -> (r : A -> A -> Prop) -> (B : Type j) -> (f : A -> B)
//                  -> ((a b : A) -> r a b -> Eq B (f a) (f b)) -> Quot A r -> B
//   Quot.ind   : (A : Type j) -> (r : A -> A -> Prop) -> (P : Quot A r -> Prop)
//                  -> ((a : A) -> P (Quot.mk A r a)) -> (q : Quot A r) -> P q
//   Quot.sound : (A : Type j) -> (r : A -> A -> Prop) -> (a b : A) -> r a b -> Eq (Quot A r) (Quot.mk A r a) (Quot.mk A r b)
//
//   computation:   Quot.lift A r B f h (Quot.mk A r a)  ~>  f a
//
// Why constants and not a node kind: the respects-proof h is then checked by the ordinary application rule against the type of
// Quot.lift, so there is no special typing rule that could forget to look at it (the first design in this project had exactly such a
// rule, and did). The standard soundness argument for Lean's Quot applies: h is what makes the computation rule agree with Quot.sound,
// and the motive of Quot.ind is a proposition because a motive in Type would let a function that does not respect r be defined, and
// then Quot.sound would equate its values on related arguments.
//
// Preconditions, all checked: the Prop/Type world (prop_elim), so that Prop is Type0 and h is genuinely a proof; eq_gid is EXACTLY the
// equality family  Eq : (A : Type j) -> A -> A -> Prop  with the single constructor  refl : (A : Type j) -> (a : A) -> Eq A a a  (the
// constants' types mention it, and a different family would give Quot.sound and the hypothesis of Quot.lift another meaning); no
// quotient is declared yet (a second family would give the one reduction rule two meanings). j is read off Eq and must be >= 1.
// Every type built is type-checked before the function returns; on any failure nothing is left declared.
struct QuotHandle { uint32_t type = 0, mk = 0, lift = 0, ind = 0, sound = 0; };

inline std::expected<QuotHandle, TypeError> declare_quot(uint32_t eq_gid) {
    ExecutionContext& c = ctx();
    auto fail = [&](const std::string& msg) { return std::unexpected(TypeError{TypeErrorKind::Other, mk_type(0), std::nullopt, "declare_quot: " + msg}); };
    if (!c.prop_mode) return fail("quotients live in the Prop/Type world (option prop_elim): the hypothesis of Quot.lift must be a proof and the motive of Quot.ind a proposition");
    for (const GlobalDef& d : c.globals.defs) {
        if (d.quot_role != QuotRole::None) return fail("quotients are already declared");
        if (d.name == "Quot" || d.name == "Quot.mk" || d.name == "Quot.lift" || d.name == "Quot.ind" || d.name == "Quot.sound") return fail("a global named " + d.name + " already exists");
    }
    if (eq_gid >= c.globals.defs.size()) return fail("no such global");
    const GlobalDef& eq = c.globals[eq_gid];
    if (eq.value || eq.ctor_info || eq.type.tag != NodeType::PI) return fail("the equality family must be an inductive type of type (A : Type j) -> A -> A -> Prop");
    const NodeId first_dom = c.pis[eq.type.index].dom_id;
    if (first_dom.tag != NodeType::TYPE || c.types[first_dom.index].universe_level == 0) return fail("the first argument of the equality family must be a type in Type j with j >= 1");
    const uint32_t j = c.types[first_dom.index].universe_level;
    const NodeId Tj = mk_type(j), Prop = mk_type(0), EQ = mk_const(eq_gid);
    // Locally nameless: the telescope variables are context variables (names 0.., inner binders 100..); close_pi_telescope binds them.
    const NodeId A = mk_fvar(0), r = mk_fvar(1);
    if (eq.type != close_pi_telescope({Tj, A, A}, Prop, 0)) return fail("the equality family must have the type (A : Type j) -> A -> A -> Prop");
    const auto& eq_ctors = c.globals.ctors_of(eq_gid);
    if (eq_ctors.size() != 1) return fail("the equality family must have exactly one constructor, refl");
    if (c.globals[eq_ctors[0]].type != close_pi_telescope({Tj, A}, apply_all(EQ, {A, r, r}), 0)) return fail("the constructor of the equality family must be refl : (A : Type j) -> (a : A) -> Eq A a a");

    const uint32_t g_type = static_cast<uint32_t>(c.globals.defs.size()), g_mk = g_type + 1, g_lift = g_type + 2, g_ind = g_type + 3, g_sound = g_type + 4;
    const NodeId Q = mk_const(g_type), MK = mk_const(g_mk);
    const NodeId rel = close_pi_telescope({A, A}, Prop, 100);                                  // A -> A -> Prop   (A = context variable 0)
    const NodeId QAr = apply_all(Q, {A, r});                                                    // Quot A r
    const NodeId a2 = mk_fvar(2), b3 = mk_fvar(3);

    const NodeId type_ty = close_pi_telescope({Tj, rel}, Tj, 0);
    const NodeId mk_ty = close_pi_telescope({Tj, rel, A}, QAr, 0);                              // A, r, a  =>  Quot A r
    // lift: A r B f h q  with  f : A -> B ,  h : (a b : A) -> r a b -> Eq B (f a) (f b)
    const NodeId B = mk_fvar(2), f = mk_fvar(3);
    const NodeId f_ty = close_pi_telescope({A}, B, 100);
    const NodeId h_ty = close_pi_telescope({A, A, apply_all(r, {mk_fvar(100), mk_fvar(101)})},
                                           apply_all(EQ, {B, mk_app(f, mk_fvar(100)), mk_app(f, mk_fvar(101))}), 100);
    const NodeId lift_ty = close_pi_telescope({Tj, rel, Tj, f_ty, h_ty, QAr}, B, 0);
    // ind: A r P h q  with  P : Quot A r -> Prop ,  h : (a : A) -> P (Quot.mk A r a)
    const NodeId P = mk_fvar(2);
    const NodeId P_ty = close_pi_telescope({QAr}, Prop, 100);
    const NodeId ih_ty = close_pi_telescope({A}, mk_app(P, apply_all(MK, {A, r, mk_fvar(100)})), 100);
    const NodeId ind_ty = close_pi_telescope({Tj, rel, P_ty, ih_ty, QAr}, mk_app(P, mk_fvar(4)), 0);
    // sound: A r a b h  with  h : r a b
    const NodeId sound_ty = close_pi_telescope({Tj, rel, A, A, apply_all(r, {a2, b3})},
                                               apply_all(EQ, {QAr, apply_all(MK, {A, r, a2}), apply_all(MK, {A, r, b3})}), 0);

    const std::size_t old_n = c.globals.defs.size();
    auto declare = [&](NodeId ty, const char* name, QuotRole role) { uint32_t g = c.globals.declare(ty, std::nullopt, name); c.globals.defs[g].quot_role = role; return g; };
    if (declare(type_ty, "Quot", QuotRole::Type) != g_type || declare(mk_ty, "Quot.mk", QuotRole::Mk) != g_mk || declare(lift_ty, "Quot.lift", QuotRole::Lift) != g_lift ||
        declare(ind_ty, "Quot.ind", QuotRole::Ind) != g_ind || declare(sound_ty, "Quot.sound", QuotRole::Sound) != g_sound)
        throw std::logic_error("declare_quot: global ids are not consecutive");
    for (uint32_t g : {g_type, g_mk, g_lift, g_ind, g_sound}) {
        ElabState es; ElabCtx ec;
        auto s = sort_level_of(c.globals[g].type, ec, es, "the type built for a quotient constant is not a type");
        if (!s) {
            const std::string msg = c.globals[g].name + ": " + s.error().message;
            c.globals.defs.erase(c.globals.defs.begin() + static_cast<std::ptrdiff_t>(old_n), c.globals.defs.end());
            return fail("internal error, the type built for " + msg);
        }
    }
    return QuotHandle{g_type, g_mk, g_lift, g_ind, g_sound};
}

inline ElabResult verify_motive_shape(NodeId motive_id, uint32_t arity, ElabCtx& ec, ElabState& es, uint32_t* sort_level) {
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
    if (sort_level) if (auto* vt = std::get_if<VType>(&motive_ty_val)) *sort_level = vt->universe_level;     // the sort the motive's result type lives in: the ELIMINATION sort
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
    for (auto& [level, replacement] : forcing.forced) if (occurrence_count[level] <= 1 || forcing.param_vars.count(level)) m[level] = replacement; // a repeated field keeps its own identity, unless a PARAMETER determined it
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
        const bool use_forced = it != forcing.forced.end() && (occurrence_count[base + ai] <= 1 || forcing.param_vars.count(base + ai));
        app = mk_app(app, use_forced ? it->second : mk_fvar(actual ? (*actual)[ai] : base + ai));
    }
    return app;
}

// ===========================================================================
// Parameters vs indices (option "parameters").
// A family's leading result-index positions are PARAMETERS when, in EVERY constructor, the position holds a bare field variable that differs from
// the variables of the earlier parameter positions, and EVERY occurrence of the family inside any field type passes exactly that variable there
// (the family is parametric in it). Eq: A, x. Le: n (the second slot repeats it, so it is an index). And/Or/Ex: every slot. Parameters are fixed by a
// match, so a field variable they determine is substituted everywhere (refl Nat a, le_n a): the standard eliminator's minor premise.
// ===========================================================================
inline bool uniform_family_occurrences(NodeId t, uint32_t fam, uint32_t arity, uint32_t pos, NodeId var) {
    if (t.tag == NodeType::APP || t.tag == NodeType::CONST) {
        detail::AppSpine sp = detail::decompose_app_spine(t);
        if (sp.head.tag == NodeType::CONST && ctx().consts[sp.head.index].global_id == fam) {
            if (sp.args.size() != arity) return false;                    // a partial or bare occurrence: not provably uniform
            if (!(sp.args[pos] == var)) return false;
            for (NodeId a : sp.args) if (!uniform_family_occurrences(a, fam, arity, pos, var)) return false;
            return true;
        }
    }
    return !any_child(t, [&](NodeId c) { return !uniform_family_occurrences(c, fam, arity, pos, var); });
}
inline std::vector<bool> family_param_slots_uncached(uint32_t family_gid);
inline std::vector<bool> family_param_slots(uint32_t family_gid) {
    const uint64_t key = (static_cast<uint64_t>(family_gid) << 32) | detail::ctor_gids_for_type(family_gid).size();
    auto it = ctx().param_cache.find(key);
    if (it != ctx().param_cache.end()) return it->second;
    return ctx().param_cache[key] = family_param_slots_uncached(family_gid);
}
inline std::vector<bool> family_param_slots_uncached(uint32_t family_gid) {
    const uint32_t arity = static_cast<uint32_t>(detail::instantiate_type_telescope(family_gid, 0).size());
    std::vector<bool> params(arity, false);
    std::vector<uint32_t> ctors = detail::ctor_gids_for_type(family_gid);
    if (arity == 0 || ctors.empty()) return params;
    std::vector<detail::InstantiatedCtor> insts;
    for (uint32_t c : ctors) insts.push_back(detail::instantiate_ctor_telescope(c, 0));
    for (uint32_t i = 0; i < arity; ++i) {
        bool ok = true;
        for (const auto& inst : insts) {
            const uint32_t ar = static_cast<uint32_t>(inst.arg_types.size());
            if (inst.result_indices.size() != arity) { ok = false; break; }
            NodeId r = inst.result_indices[i];
            if (r.tag != NodeType::VAR) { ok = false; break; }
            Var v = ctx().vars[r.index];
            if (v.bound || v.n >= ar) { ok = false; break; }                 // must be one of this constructor's own fields
            for (uint32_t k = 0; k < i; ++k) if (inst.result_indices[k] == r) { ok = false; break; }   // distinct from the earlier parameters' variables
            if (!ok) break;
            for (NodeId ft : inst.arg_types) if (!uniform_family_occurrences(ft, family_gid, arity, i, r)) { ok = false; break; }
            if (!ok) break;
        }
        if (!ok) break;                                                       // parameters form a PREFIX
        params[i] = true;
    }
    return params;
}
struct MotiveParts { std::vector<bool> params; std::vector<NodeId> doms, idx_vars; uint32_t num_indices = 0; };
/// The motive's binders for a scrutinee whose type is  T c_0 .. c_{k-1}: one binder per slot (so Match's shape is unchanged) plus the scrutinee binder.
/// With parameters, the binder TYPES are specialised to the actual parameter values (y : Nat rather than y : A), which is what lets a premise such as
/// same_ctor y _ type-check; the parameter binders themselves stay unused.
inline MotiveParts motive_binders(uint32_t tgid, uint32_t n, const std::vector<NodeId>& concrete_indices) {
    MotiveParts mp;
    std::vector<NodeId> index_types = detail::instantiate_type_telescope(tgid, n);
    const uint32_t arity = static_cast<uint32_t>(index_types.size());
    mp.params = (ctx().param_mode && concrete_indices.size() == arity) ? family_param_slots(tgid) : std::vector<bool>(arity, false);
    if (mp.params.size() != arity) mp.params.assign(arity, false);
    std::unordered_map<uint32_t, NodeId> pm;
    for (uint32_t i = 0; i < arity; ++i) if (mp.params[i]) pm[n + i] = concrete_indices[i];
    std::vector<NodeId> args;
    for (uint32_t i = 0; i < arity; ++i) {
        mp.doms.push_back(pm.empty() ? index_types[i] : replace_fvars(index_types[i], pm));
        mp.idx_vars.push_back(mk_fvar(n + i));
        args.push_back(mp.params[i] ? concrete_indices[i] : mk_fvar(n + i));
        if (!mp.params[i]) ++mp.num_indices;
    }
    mp.doms.push_back(apply_all(mk_const(tgid), args));
    return mp;
}

inline ElabResult check_match_branches(uint32_t type_global_id, NodeId motive, const std::vector<NodeId>& branches,
                                        const std::vector<NodeId>& concrete_indices, ElabCtx& ec, ElabState& es) {
    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(type_global_id);
    if (ctx().prop_mode && ctx().prop_elim_only.count(type_global_id)) {      // a proposition that is not a syntactic subsingleton: eliminable only into Type0
        uint32_t elim_level = 0;
        if (verify_motive_shape(motive, static_cast<uint32_t>(concrete_indices.size()), ec, es, &elim_level) && elim_level != 0)
            return std::unexpected(TypeError{TypeErrorKind::Other, motive, std::nullopt, "Match: large elimination from the proposition '" + ctx().globals.defs[type_global_id].name +
                "' (an inductive in Type0 that is not a syntactic subsingleton) is only allowed into Type0"});
    }
    std::vector<bool> param_slots_v = ctx().param_mode ? family_param_slots(type_global_id) : std::vector<bool>();
    if (ctor_gids.size() != branches.size())
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive, std::nullopt,
            "Match branch count does not match the scrutinee's inductive type's constructor count (non-exhaustive or over-long match)"});
    for (std::size_t ci = 0; ci < ctor_gids.size(); ++ci) {
        detail::InstantiatedCtor inst = detail::instantiate_ctor_telescope(ctor_gids[ci], ec.size());
        const uint32_t arity = static_cast<uint32_t>(inst.arg_types.size());
        CtorIndexForcing forcing = unify_ctor_pattern_indices(inst, concrete_indices, ec.size(), arity, ec.values, param_slots_v.empty() ? nullptr : &param_slots_v);
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
            if (!ctx().globals[type_global_id].is_inductive)
                return err(TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, "scrutinee's type is not an inductive type application");
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
        // F21: a binder annotation must itself be a type. Unifying it with the expected domain is not enough (an ill-typed annotation can be
        // convertible to a well-typed one), and infer(Lambda) already insists on it. Skipped only when it IS the expected domain.
        if (dom_j != expected_dom) { auto sdom = sort_level_of(dom_j, cur_ec, es, "Lambda domain must be a type"); if (!sdom) return std::unexpected(sdom.error()); }
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
    if (!ctx().globals[type_global_id].is_inductive)
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, expected_ty, "scrutinee's type is not an inductive type application"});
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
    {   // F25: a SOLVED argument meta must have received a solution OF ITS TYPE. Unification solves `?A := goal` for a lemma whose conclusion is a bare variable (choice : Pi (A : Type1), Nonempty A -> A) even when the
        // goal lives in another universe; nothing checked that, so apply produced a state that could never verify (the kernel caught it only at the end). The check runs on a scratch copy and its meta assignments are undone.
        auto mark = ctx().meta_ctx.checkpoint(); ElabState scratch = ps.es; bool solutions_ok = true; std::string why;
        for (auto& [m, arg_ty] : arg_metas) {
            if (!ctx().meta_ctx.is_solved(m)) continue;
            NodeId solution = quote(eval(meta_applied_to_context(m, g.ec.size()), g.ec.values), g.ec.size());
            NodeId wanted = quote(eval(arg_ty, g.ec.values), g.ec.size());
            ElabCtx scratch_ec = g.ec;
            if (ElabResult r = check(solution, wanted, scratch_ec, scratch); !r) { solutions_ok = false; why = r.error().message; break; }
        }
        ctx().meta_ctx.rewind(mark);
        if (!solutions_ok) return std::unexpected(TypeError{TypeErrorKind::Mismatch, g.target, std::nullopt, "apply: the solution unification found for an argument of the lemma does not have that argument's type (" + why + ")"});
    }
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
    if (!ctx().globals[si.type_gid].is_inductive)
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, scrut_ty, std::nullopt, std::string(who) + ": hypothesis's type is not an inductive type application"});
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

inline ElabResult cases(ProofState& ps, uint32_t hyp_level, std::optional<uint32_t> target_meta_id = std::nullopt, const std::vector<uint32_t>& generalizing = {}) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return no_goal_error("cases", target_meta_id);
    Goal g = ps.goals[*idx_opt];
    if (hyp_level >= g.hyps.size()) return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt, "cases: hypothesis index out of range"});
    auto si = scrutinee_info("cases", g, hyp_level); if (!si) return std::unexpected(si.error());
    const uint32_t n = g.ec.size(), tgid = si->type_gid, arity = static_cast<uint32_t>(si->indices.size());
    const std::vector<NodeId>& concrete_indices = si->indices;
    MotiveParts mp = motive_binders(tgid, n, concrete_indices);        // parameters (option) are FIXED: not generalised, not abstracted by the motive or the Fix
    const std::vector<bool>& is_param = mp.params; const uint32_t R = mp.num_indices;
    std::vector<std::pair<NodeId, NodeId>> subs;
    for (uint32_t i = 0; i < arity; ++i) if (!is_param[i]) subs.push_back({concrete_indices[i], mk_fvar(n + i)});
    subs.push_back({mk_fvar(hyp_level), mk_fvar(n + arity)});
    // cases reverts a CONTIGUOUS suffix: from the earliest hypothesis that mentions an index or the scrutinee, to the end
    uint32_t revert_from = hyp_level + 1;
    for (uint32_t j = 0; j < hyp_level; ++j) if (replace_terms(g.hyps[j].type, subs) != g.hyps[j].type) { revert_from = j; break; }
    for (uint32_t j : generalizing) if (j < hyp_level) revert_from = std::min(revert_from, j);   // F6: explicitly generalised hypotheses (those after the scrutinee are reverted anyway)
    std::vector<uint32_t> reverted_js;
    for (uint32_t j = revert_from; j < g.hyps.size(); ++j) if (j != hyp_level) reverted_js.push_back(j);
    NodeId G = generalized_goal(g, subs, reverted_js);
    if (mp.doms.size() != arity + 1)
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "cases: index count mismatch between the hypothesis's type application and the inductive's own declared arity"});
    const std::vector<NodeId>& idx_vars = mp.idx_vars; const std::vector<NodeId>& mdoms = mp.doms;
    NodeId motive_original = close_lambda_telescope(mdoms, G, n);
    if (!infer(motive_original, g.ec, ps.es))
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "cases: constructed motive failed to type-check -- the goal likely depends on the hypothesis in a way that cannot be abstracted cleanly"});
    if (ctx().prop_mode && ctx().prop_elim_only.count(tgid)) {
        uint32_t elim_level = 0;
        if (verify_motive_shape(motive_original, arity, g.ec, ps.es, &elim_level) && elim_level != 0)
            return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt, "cases: the hypothesis's type is a proposition that is not a syntactic subsingleton, so it can only be eliminated into a proposition, but the goal is not one"});
    }
    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(tgid);
    std::vector<detail::InstantiatedCtor> insts; std::vector<CtorIndexForcing> forcings;
    for (uint32_t ctor_gid : ctor_gids) {
        detail::InstantiatedCtor inst = detail::instantiate_ctor_telescope(ctor_gid, n);
        CtorIndexForcing forcing = unify_ctor_pattern_indices(inst, concrete_indices, n, static_cast<uint32_t>(inst.arg_types.size()), g.ec.values, &is_param);
        if (!forcing.ok) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, forcing.error});
        insts.push_back(std::move(inst)); forcings.push_back(std::move(forcing));
    }
    // refutation exploitation (only one constructor per call, and only when nothing is reverted)
    std::optional<std::size_t> exploit_ci;
    for (std::size_t ci = 0; ci < forcings.size(); ++ci) {
        const CtorIndexForcing& f = forcings[ci];
        if (f.refuted && f.refuted_slot && f.refuted_pat_value && f.refuted_conc_value) { exploit_ci = ci; break; }
    }
    // F22: the reference only exploits a refutation when nothing is reverted, so any dependent hypothesis silently disabled discrimination. In parameter mode (which already differs from the reference)
    // the exploit works with reverted hypotheses too: the same_ctor premise is the OUTERMOST layer of the extended motive, so the reverted binders simply sit one level further up in the branch lambdas.
    if (!reverted_js.empty() && !ctx().param_mode) exploit_ci.reset();
    NodeId motive_final = motive_original;
    if (exploit_ci) {
        const uint32_t slot = *forcings[*exploit_ci].refuted_slot, family_gid = *forcings[*exploit_ci].refuted_family_gid;
        NodeId component = apply_all(mk_const(get_or_build_same_ctor(family_gid)), {idx_vars[slot], concrete_indices[slot]});
        NodeId m = close_lambda_telescope(mdoms, mk_pi(component, G), n);
        if (infer(m, g.ec, ps.es)) motive_final = m; else exploit_ci.reset();
    }
    // With parameters EVERY constructor refuted at the same index slot shares the one same_ctor premise (Le (S n) 0 refutes both le_n and le_S);
    // without them only the first is exploited, exactly like the reference.
    std::vector<bool> is_exploited(ctor_gids.size(), false);
    if (exploit_ci) {
        is_exploited[*exploit_ci] = true;
        if (ctx().param_mode)
            for (std::size_t cj = 0; cj < forcings.size(); ++cj) {
                const CtorIndexForcing& f = forcings[cj];
                if (f.refuted && f.refuted_pat_value && f.refuted_conc_value && f.refuted_slot == forcings[*exploit_ci].refuted_slot && f.refuted_family_gid == forcings[*exploit_ci].refuted_family_gid) is_exploited[cj] = true;
            }
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
        const NodeId branch_expected_raw = branch_expected;
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
        // a branch that carries the same_ctor premise has it BEFORE the reverted binders: recompute those layers one level up
        std::vector<NodeId> reverted_dom_types_up; NodeId branch_expected_up = branch_expected;
        if (exploit_ci && !reverted_js.empty()) {
            Value cur_val = eval(branch_expected_raw, be_env);
            for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) {
                auto* pi = std::get_if<VPi>(&cur_val);
                if (!pi) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "cases: internal error -- expected a Pi layer for a reverted hypothesis but found none"});
                const uint32_t target_level = n + ctor_arity + 1 + static_cast<uint32_t>(ri);
                reverted_dom_types_up.push_back(quote(eval(pi->dom_id, pi->env), target_level));
                Value next = eval(pi->cod_id, pi->env.extend(fresh_neutral(target_level)));
                cur_val = std::move(next);
            }
            branch_expected_up = quote(cur_val, n + ctor_arity + 1 + static_cast<uint32_t>(reverted_js.size()));
        }
        if (exploit_ci && is_exploited[ci]) {
            const uint32_t family_gid = *forcing.refuted_family_gid, absurd_level = n + ctor_arity;
            NodeId absurd_ty = apply_all(mk_const(get_or_build_same_ctor(family_gid)), {*forcing.refuted_pat_value, *forcing.refuted_conc_value});
            NodeId reduced = reverted_js.empty() ? quote(eval(branch_expected, be_env), absurd_level) : branch_expected_up;
            NodeId absurd_body = mk_match(mk_fvar(absurd_level), mk_lambda(mk_const(get_or_build_empty_type()), reduced), {});
            std::vector<NodeId> doms = arg_types; doms.push_back(absurd_ty);
            for (NodeId d : reverted_dom_types_up) doms.push_back(d);
            branches.push_back(close_lambda_telescope(doms, absurd_body, n));
            continue;
        }
        ElabCtx branch_ec = g.ec; std::vector<Hypothesis> branch_hyps = g.hyps;
        for (NodeId field_ty : arg_types) { branch_ec = branch_ec.extend(field_ty); branch_hyps.push_back(Hypothesis{"", field_ty}); }
        for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) { branch_ec = branch_ec.extend(reverted_dom_types[ri]); branch_hyps.push_back(Hypothesis{g.hyps[reverted_js[ri]].name, reverted_dom_types[ri]}); }
        uint32_t subgoal_meta = ctx().meta_ctx.fresh(close_pi_over_context(branch_expected, branch_ec), branch_ec.size());
        NodeId subgoal_ref;
        if (exploit_ci && !reverted_js.empty()) {   // the meta is applied to the context, the fields and the reverted binders, which sit one level up (after the premise)
            subgoal_ref = mk_meta(subgoal_meta);
            for (uint32_t l = 0; l < n + ctor_arity; ++l) subgoal_ref = mk_app(subgoal_ref, mk_fvar(l));
            for (uint32_t ri = 0; ri < reverted_js.size(); ++ri) subgoal_ref = mk_app(subgoal_ref, mk_fvar(n + ctor_arity + 1 + ri));
        } else subgoal_ref = meta_applied_to_context(subgoal_meta, branch_ec.size());
        std::vector<NodeId> doms = arg_types;
        if (exploit_ci) {   // the extended motive adds a same_ctor premise after the fields; this branch ignores it
            const uint32_t slot = *forcings[*exploit_ci].refuted_slot, family_gid = *forcings[*exploit_ci].refuted_family_gid;
            doms.push_back(apply_all(mk_const(get_or_build_same_ctor(family_gid)), {forced_result_indices[slot], concrete_indices[slot]}));
            for (NodeId d : reverted_dom_types_up) doms.push_back(d);
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

inline ElabResult induct(ProofState& ps, uint32_t hyp_level, std::optional<uint32_t> target_meta_id = std::nullopt, const std::vector<uint32_t>& generalizing = {},
                          std::optional<uint32_t> eq_gid = std::nullopt, std::optional<uint32_t> refl_gid = std::nullopt) {
    auto idx_opt = find_goal_index(ps, target_meta_id);
    if (!idx_opt) return no_goal_error("induct", target_meta_id);
    Goal g = ps.goals[*idx_opt];
    if (hyp_level >= g.hyps.size()) return std::unexpected(TypeError{TypeErrorKind::Other, NodeId{}, std::nullopt, "induct: hypothesis index out of range"});
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
    MotiveParts mp = motive_binders(tgid, n, concrete_indices);        // parameters (option) are FIXED: not generalised, not abstracted by the motive or the Fix
    const std::vector<bool>& is_param = mp.params; const uint32_t R = mp.num_indices;
    // F12 (index generalisation). An index slot is GOOD when the hypothesis has a distinct variable there that does not occur in the (fixed) parameter values: the Fix is then generic over it and the goal is
    // generalised by replacing that variable with the slot's marker. Every other index slot (a concrete term, a repeated variable, or a variable that also occurs in a parameter) is BAD. With the Eq and refl
    // ids supplied a bad slot is generalised through an EQUATION  marker = c  (dependent induction): the goal is left alone, the Fix takes the equation as one more premise and is applied to refl. Without the
    // ids parameter mode refuses (a partial proof built on a concrete index cannot be completed validly) and legacy mode keeps the reference behaviour.
    std::vector<bool> bad(arity, false); uint32_t n_bad = 0;
    {
        std::unordered_set<uint32_t> pfv, seen_idx_vars;
        for (uint32_t i = 0; i < arity; ++i) if (is_param[i]) detail::collect_fvars(concrete_indices[i], pfv);
        for (uint32_t i = 0; i < arity; ++i) {
            if (is_param[i]) continue;
            bool good = false;
            if (concrete_indices[i].tag == NodeType::VAR) { Var v = ctx().vars[concrete_indices[i].index]; good = !v.bound && !pfv.count(v.n) && seen_idx_vars.insert(v.n).second; }
            if (!good) { bad[i] = true; ++n_bad; }
        }
    }
    const bool use_eq = n_bad > 0 && eq_gid.has_value() && refl_gid.has_value();
    // When any slot needs an equation, EVERY non-parameter slot gets one. No context variable is then substituted by a forced index, so the scrutinee hypothesis can be passed to the branch as the original h
    // (in scope inside the Fix body, and of exactly the type the branch context declares for it). A mixed good/bad treatment would need a cast of the constructor application along the equations.
    if (use_eq) { n_bad = 0; for (uint32_t i = 0; i < arity; ++i) { bad[i] = !is_param[i]; if (bad[i]) ++n_bad; } }
    if (n_bad > 0 && !use_eq && ctx().param_mode) {
        uint32_t first_bad = 0; while (!bad[first_bad]) ++first_bad;
        return std::unexpected(TypeError{TypeErrorKind::Other, g.hyps[hyp_level].type, std::nullopt, "induct: index slot " + std::to_string(first_bad) +
            " of the hypothesis is not a distinct variable (it is a concrete term, repeats another index, or also occurs in a parameter); pass eq_gid and refl_gid to generalise it through an equation, or use cases"});
    }
    if (use_eq && mp.doms.size() == arity + 1) {
        for (uint32_t i = 0; i < arity; ++i) if (bad[i]) {
            std::unordered_set<uint32_t> dep; detail::collect_fvars(mp.doms[i], dep);
            for (uint32_t l : dep) if (l >= n)
                return std::unexpected(TypeError{TypeErrorKind::Other, g.hyps[hyp_level].type, std::nullopt, "induct: index slot " + std::to_string(i) +
                    " needs an equation but its type depends on an earlier index (that would need heterogeneous equality, which is not implemented)"});
        }
    }
    std::vector<std::pair<NodeId, NodeId>> subs;
    for (uint32_t i = 0; i < arity; ++i) if (!is_param[i] && !(use_eq && bad[i])) subs.push_back({concrete_indices[i], mk_fvar(n + i)});
    subs.push_back({mk_fvar(hyp_level), mk_fvar(n + arity)});
    // induct reverts exactly the hypotheses that (transitively) depend on the scrutinee or its indices: a fixpoint, not a suffix
    std::vector<bool> needs_revert(g.hyps.size(), false);
    for (uint32_t j : generalizing) if (j < g.hyps.size() && j != hyp_level) needs_revert[j] = true;   // F6: explicitly generalised hypotheses, then the closure
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
    if (mp.doms.size() != arity + 1)
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "induct: index count mismatch between the hypothesis's type application and the inductive's own declared arity"});
    NodeId G;
    if (!use_eq) G = generalized_goal(g, subs, reverted_js);
    else {
        NodeId gbody = replace_terms(g.target, subs);
        for (uint32_t i = arity; i-- > 0;) if (bad[i]) {
            NodeId eq_ty = mk_app(mk_app(mk_app(mk_const(*eq_gid), mp.doms[i]), mk_fvar(n + i)), replace_terms(concrete_indices[i], subs));      // Eq T_i marker_i c_i
            gbody = mk_pi(eq_ty, gbody);
        }
        std::vector<NodeId> rdoms; for (uint32_t j : reverted_js) rdoms.push_back(replace_terms(g.hyps[j].type, subs));
        G = close_binders(true, rdoms, reverted_js, gbody);
    }
    const std::size_t n_layers = reverted_js.size() + (use_eq ? n_bad : 0);      // the reverted hypotheses, then one equation per generalised slot
    const std::vector<NodeId>& idx_vars = mp.idx_vars; const std::vector<NodeId>& mdoms = mp.doms;
    NodeId motive = close_lambda_telescope(mdoms, G, n);
    if (!infer(motive, g.ec, ps.es))
        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "induct: constructed motive failed to type-check -- the goal likely depends on the hypothesis in a way that cannot be abstracted cleanly"});
    if (ctx().prop_mode && ctx().prop_elim_only.count(tgid)) {
        uint32_t elim_level = 0;
        if (verify_motive_shape(motive, arity, g.ec, ps.es, &elim_level) && elim_level != 0)
            return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt, "induct: the hypothesis's type is a proposition that is not a syntactic subsingleton, so it can only be eliminated into a proposition, but the goal is not one"});
    }
    // the Fix is generic over the INDEX slots only (parameters stay fixed): compact the binders. Without parameters this is the identity.
    std::vector<uint32_t> rank(arity, 0);
    std::unordered_map<uint32_t, NodeId> to_compact;
    { uint32_t r = 0; for (uint32_t i = 0; i < arity; ++i) if (!is_param[i]) { rank[i] = r; to_compact[n + i] = mk_fvar(n + r); ++r; } to_compact[n + arity] = mk_fvar(n + R); }
    std::vector<NodeId> cdoms;
    for (uint32_t i = 0; i < arity; ++i) if (!is_param[i]) cdoms.push_back(replace_fvars(mdoms[i], to_compact));
    cdoms.push_back(replace_fvars(mdoms[arity], to_compact));
    NodeId fn_type = close_pi_telescope(cdoms, replace_fvars(G, to_compact), n);
    // runtime scope of the Fix body:  self = n, index parameters n+1.., scrutinee parameter n+1+arity, constructor fields from body_base
    const uint32_t self_level = n, scrutinee_level = n + 1 + R, body_base = scrutinee_level + 1;
    std::vector<NodeId> idx_vars_body;    // what the Fix body's constructor patterns are unified against: its own index parameters, and the FIXED parameter values
    for (uint32_t i = 0; i < arity; ++i) idx_vars_body.push_back(is_param[i] ? concrete_indices[i] : mk_fvar(n + 1 + rank[i]));
    std::vector<uint32_t> ctor_gids = detail::ctor_gids_for_type(tgid);
    std::vector<NodeId> branches; std::vector<Goal> new_goals;
    for (uint32_t ctor_gid : ctor_gids) {
        // ---- the goal the user sees
        detail::InstantiatedCtor inst_agent = detail::instantiate_ctor_telescope(ctor_gid, n);
        const uint32_t ctor_arity = static_cast<uint32_t>(inst_agent.arg_types.size());
        std::vector<bool> rec_mask = recursive_field_mask(tgid, ctor_gid);
        CtorIndexForcing forcing_agent = unify_ctor_pattern_indices(inst_agent, concrete_indices, n, ctor_arity, g.ec.values, &is_param, use_eq ? &bad : nullptr);
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
                for (uint32_t j = 0; j < arity && j < sp.args.size(); ++j) if (!is_param[j]) m[n + j] = sp.args[j];
                m[n + arity] = mk_fvar(actual[fi]);
                NodeId ih_ty = replace_fvars(G, m);
                branch_ec = branch_ec.extend(ih_ty); branch_hyps.push_back(Hypothesis{"", ih_ty});
            }
        }
        NodeId agent_ctor_app = forced_ctor_app(ctor_gid, n, ctor_arity, inst_agent, forcing_agent, &actual);
        std::vector<NodeId> forced_agent_result_indices = apply_ctor_index_forcing_to_result_indices(inst_agent.result_indices, forcing_agent);
        std::unordered_map<uint32_t, NodeId> bm;
        for (uint32_t j = 0; j < arity; ++j) if (!is_param[j]) bm[n + j] = to_actual(forced_agent_result_indices[j], ctor_arity);
        bm[n + arity] = agent_ctor_app;
        NodeId branch_expected = replace_fvars(G, bm);
        std::vector<NodeId> reverted_dom_types;
        if (n_layers > 0) {
            const uint32_t base_level = branch_ec.size();
            NodeId remaining = branch_expected;
            for (std::size_t ri = 0; ri < n_layers; ++ri) {
                if (remaining.tag != NodeType::PI) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "induct: internal error -- expected a Pi layer for a reverted hypothesis but found none"});
                Pi p = ctx().pis[remaining.index];
                reverted_dom_types.push_back(p.dom_id);
                remaining = instantiate1(p.cod_id, mk_fvar(base_level + static_cast<uint32_t>(ri)));
            }
            branch_expected = remaining;
            for (std::size_t ri = 0; ri < n_layers; ++ri) { branch_ec = branch_ec.extend(reverted_dom_types[ri]); branch_hyps.push_back(Hypothesis{ri < reverted_js.size() ? g.hyps[reverted_js[ri]].name : std::string(), reverted_dom_types[ri]}); }
        }
        uint32_t subgoal_meta = ctx().meta_ctx.fresh(close_pi_over_context(branch_expected, branch_ec), branch_ec.size());
        // ---- the runtime branch (a lambda over the fields and the reverted hypotheses; no IH binders, the IH is the recursive call)
        detail::InstantiatedCtor inst = detail::instantiate_ctor_telescope(ctor_gid, body_base);
        CtorIndexForcing forcing_body = unify_ctor_pattern_indices(inst, idx_vars_body, body_base, ctor_arity, g.ec.values, &is_param);
        if (!forcing_body.ok) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, forcing_body.error});
        std::vector<NodeId> arg_types_body = apply_ctor_index_forcing(inst.arg_types, forcing_body);
        std::vector<NodeId> forced_body_result_indices = apply_ctor_index_forcing_to_result_indices(inst.result_indices, forcing_body);
        NodeId true_ctor_app = forced_ctor_app(ctor_gid, body_base, ctor_arity, inst, forcing_body);
        NodeId applied = mk_meta(subgoal_meta);
        for (uint32_t i = 0; i < n; ++i) {
            NodeId supplied = mk_fvar(i);
            if (i == hyp_level) supplied = use_eq ? mk_fvar(i) : true_ctor_app;
            else {
                bool replaced = false;
                for (uint32_t j = 0; j < arity && !replaced; ++j) {
                    if (!is_param[j] && !(use_eq && bad[j]) && concrete_indices[j].tag == NodeType::VAR) { Var v = ctx().vars[concrete_indices[j].index]; if (!v.bound && v.n == i) { supplied = forced_body_result_indices[j]; replaced = true; } }
                }
                if (!replaced) for (std::size_t ri = 0; ri < reverted_js.size(); ++ri) if (reverted_js[ri] == i) { supplied = mk_fvar(body_base + ctor_arity + static_cast<uint32_t>(ri)); break; }
            }
            applied = mk_app(applied, supplied);
        }
        std::vector<NodeId> agent_ctor_args = detail::decompose_app_spine(agent_ctor_app).args, true_ctor_args = detail::decompose_app_spine(true_ctor_app).args;
        for (uint32_t fi = 0; fi < ctor_arity; ++fi) {
            NodeId field_arg = mk_fvar(body_base + fi);
            // a field the agent left UNFORCED (its slot is generalised through an equation) but that the kernel substitutes against the Fix's generic index: pass the substituted value so the types agree
            if (use_eq && fi < agent_ctor_args.size() && fi < true_ctor_args.size() && agent_ctor_args[fi] == mk_fvar(actual[fi])) field_arg = true_ctor_args[fi];
            applied = mk_app(applied, field_arg);
            if (fi < rec_mask.size() && rec_mask[fi]) {
                NodeId ih_call = mk_fvar(self_level);
                detail::AppSpine sp = detail::decompose_app_spine(arg_types_body[fi]);
                for (uint32_t j = 0; j < arity && j < sp.args.size(); ++j) if (!is_param[j]) ih_call = mk_app(ih_call, sp.args[j]);
                applied = mk_app(applied, mk_app(ih_call, mk_fvar(body_base + fi)));
            }
        }
        for (uint32_t ri = 0; ri < n_layers; ++ri) applied = mk_app(applied, mk_fvar(body_base + ctor_arity + ri));
        std::vector<NodeId> bdoms = arg_types_body;
        if (n_layers > 0) {
            std::unordered_map<uint32_t, NodeId> rm;
            for (uint32_t j = 0; j < arity; ++j) if (!is_param[j]) rm[n + j] = forced_body_result_indices[j];
            rm[n + arity] = true_ctor_app;
            NodeId remaining_body = replace_fvars(G, rm);
            for (std::size_t ri = 0; ri < n_layers; ++ri) {
                Pi p = ctx().pis[remaining_body.index];
                bdoms.push_back(p.dom_id);
                remaining_body = instantiate1(p.cod_id, mk_fvar(body_base + ctor_arity + static_cast<uint32_t>(ri)));
            }
        }
        branches.push_back(close_lambda_telescope(bdoms, applied, body_base));
        new_goals.push_back(Goal{subgoal_meta, branch_ec, std::move(branch_hyps), branch_expected});
    }
    NodeId match_term = mk_match(mk_fvar(scrutinee_level), motive, branches);
    std::vector<NodeId> body_doms;      // the compact binder types, moved from base n to base n+1 (the Fix body sits under the self binder)
    { std::unordered_map<uint32_t, NodeId> up; for (uint32_t r = 0; r <= R; ++r) up[n + r] = mk_fvar(n + 1 + r); for (NodeId d : cdoms) body_doms.push_back(replace_fvars(d, up)); }
    NodeId fix_body = abstract_fvars(close_lambda_telescope(body_doms, match_term, n + 1), std::vector<uint32_t>{self_level});
    Fix fix_node{fn_type, R, fix_body};
    if (!fix_guard_ok(fix_node, n))
        return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt,
            "induct: the assembled recursive proof failed the local structural recursion check -- some recursive call does not recurse on a structurally smaller instance of the hypothesis's own type"});
    NodeId proof = mk_fix(fn_type, R, fix_body);
    if (use_eq || ctx().param_mode) {   // F23: keep the Fix opaque to extraction (see get_or_build_barrier): the generic body is then what the kernel (and the independent checker) verify
        auto fn_sort = infer(fn_type, g.ec, ps.es);
        if (!fn_sort) return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt, "induct: the generalised induction statement failed to type-check"});
        NodeId sty = *fn_sort; if (sty.tag != NodeType::TYPE) sty = quote(eval(sty, g.ec.values), n);
        if (sty.tag == NodeType::TYPE) proof = mk_app(mk_app(mk_const(get_or_build_barrier(ctx().types[sty.index].universe_level)), fn_type), proof);
    }
    for (uint32_t i = 0; i < arity; ++i) if (!is_param[i]) proof = mk_app(proof, concrete_indices[i]);
    proof = mk_app(proof, mk_fvar(hyp_level));
    for (uint32_t rj : reverted_js) proof = mk_app(proof, mk_fvar(rj));
    if (use_eq) for (uint32_t i = 0; i < arity; ++i) if (bad[i]) {      // the equations are discharged by refl at the actual indices
        std::unordered_map<uint32_t, NodeId> actual_idx; for (uint32_t q = 0; q < arity; ++q) if (!is_param[q]) actual_idx[n + q] = concrete_indices[q];
        proof = mk_app(proof, mk_app(mk_app(mk_const(*refl_gid), replace_fvars(mdoms[i], actual_idx)), concrete_indices[i]));
    }
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
    for (uint32_t gid = 0; gid < ctx().globals.defs.size(); ++gid) {
        if (ctx().globals.defs[gid].name.rfind("__barrier_", 0) == 0) continue;      // an identity at every type: it would be a candidate for every goal
        tree.insert(mk_const(gid), ctx().globals.defs[gid].type);
    }
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


// ===========================================================================
// Scope validation for terms built OUTSIDE the kernel (API boundary). A term is well-scoped in a context of `nfree` variables when it has no loose
// bound variable and every context variable has a level below nfree. eval()/Env::operator[] abort on an out-of-range level, so every entry point that
// accepts a user-built term must reject ill-scoped input with an error first. (The reference aborts on such input too: finding F18.)
// ===========================================================================
namespace detail {
inline bool fvars_below(NodeId t, uint32_t nfree, std::unordered_set<uint64_t>& seen) {
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(t.tag)) << 32) | t.index;
    if (!seen.insert(key).second) return true;
    if (t.tag == NodeType::VAR) { Var v = ctx().vars[t.index]; return v.bound || v.n < nfree; }
    return !any_child(t, [&](NodeId c) { return !fvars_below(c, nfree, seen); });
}
inline bool well_scoped(NodeId t, uint32_t nfree) {
    if (loose_range(t) != 0) return false;
    std::unordered_set<uint64_t> seen; return fvars_below(t, nfree, seen);
}
} // namespace detail

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
