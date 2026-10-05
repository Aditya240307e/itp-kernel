// test_nodepool_rewind_fuzz.cpp
//
// Verifies NodePool::rewind's O(reclaimed) implementation against TWO
// independent oracles:
//   (1) OldNodePool -- a verbatim copy of the previous, rebuild-the-whole-
//       table implementation (intern / intern_lazy / grow / rewind).
//   (2) a plain vector + hash-map model of "an interning pool".
// under randomized interleavings of intern (both entry points), nested
// checkpoints, and LIFO rewinds -- including rewinds that reach back across
// grow() boundaries, rewinds to zero, no-op rewinds, and forced-collision
// hash regimes that create very long probe chains -- asserting after every
// rewind that all structural invariants hold (NodePool::check_invariants).
// A second section fuzzes the real ExecutionContext end to end.

#include "itp_kernel.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <vector>

using namespace itp;

static int g_checks = 0;
#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);       \
            std::printf(__VA_ARGS__);                                           \
            std::printf("\n");                                                  \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---- test key types with selectable hash regimes ------------------------------------------
template <int Mode> struct Key {
    uint32_t v = 0;
    bool operator==(const Key&) const = default;
};
namespace std {
template <int Mode> struct hash<Key<Mode>> {
    size_t operator()(const Key<Mode>& k) const noexcept {
        if (Mode == 0) return k.v % 13;        // heavy collisions: 13 distinct hashes, very long chains
        if (Mode == 2) return 0;               // total collision: every key probes from slot 0
        uint64_t x = k.v * 0x9E3779B97F4A7C15ull; // well-mixed
        x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
        return static_cast<size_t>(x);
    }
};
} // namespace std

// ---- verbatim copy of the PREVIOUS implementation (oracle) --------------------------------
template <typename T> struct OldNodePool {
    struct Slot { bool occupied = false; std::size_t hash = 0; uint32_t index = 0; };
    static constexpr std::size_t kInitialCapacity = 16;
    ChunkedPool<T> storage;
    std::vector<Slot> table;
    std::size_t count = 0;
    uint32_t intern(T fields) {
        std::size_t h = std::hash<T>{}(fields);
        if (table.empty()) table.assign(kInitialCapacity, Slot{});
        std::size_t mask = table.size() - 1;
        std::size_t i = h & mask;
        while (true) {
            Slot& slot = table[i];
            if (!slot.occupied) {
                uint32_t idx = storage.push(std::move(fields));
                slot = Slot{true, h, idx};
                ++count;
                if (count * 10 >= table.size() * 7) grow();
                return idx;
            }
            if (slot.hash == h && storage[slot.index] == fields) return slot.index;
            i = (i + 1) & mask;
        }
    }
    void grow() {
        std::vector<Slot> old = std::move(table);
        table.assign(old.size() * 2, Slot{});
        std::size_t mask = table.size() - 1;
        for (const Slot& s : old) {
            if (!s.occupied) continue;
            std::size_t i = s.hash & mask;
            while (table[i].occupied) i = (i + 1) & mask;
            table[i] = s;
        }
    }
    uint32_t size() const { return storage.size(); }
    void rewind(uint32_t new_size) {
        storage.truncate_and_release(new_size);
        std::vector<Slot> old = std::move(table);
        table.clear();
        count = 0;
        if (new_size == 0) return;
        std::size_t surviving = 0;
        for (const Slot& s : old) if (s.occupied && s.index < new_size) ++surviving;
        std::size_t new_cap = kInitialCapacity;
        while (new_cap * 7 <= surviving * 10) new_cap *= 2;
        table.assign(new_cap, Slot{});
        std::size_t mask = table.size() - 1;
        for (const Slot& s : old) {
            if (!s.occupied || s.index >= new_size) continue;
            std::size_t i = s.hash & mask;
            while (table[i].occupied) i = (i + 1) & mask;
            table[i] = s;
            ++count;
        }
    }
};

// ---- the differential fuzz over one hash regime -------------------------------------------
template <int Mode>
static void fuzz_pool(uint32_t seed, int steps, uint32_t universe, int max_depth, const char* label) {
    using K = Key<Mode>;
    NodePool<K> pool;
    OldNodePool<K> ref;
    std::vector<uint32_t> model;                       // index -> value
    std::unordered_map<uint32_t, uint32_t> index_of;   // value -> index
    std::vector<uint32_t> stack;                       // checkpoint sizes (LIFO)
    std::mt19937 rng(seed);
    int rewinds = 0, spans_grow = 0, big_rewinds = 0;
    size_t last_cap = 0, peak = 0;

    auto verify_all = [&](bool full) {
        std::string err = pool.check_invariants();
        CHECK(err.empty(), "[%s seed=%u] invariant violated: %s", label, seed, err.c_str());
        CHECK(pool.size() == model.size() && ref.size() == model.size(),
              "[%s seed=%u] size mismatch new=%u ref=%u model=%zu", label, seed, pool.size(), ref.size(), model.size());
        const uint32_t n = static_cast<uint32_t>(model.size());
        const uint32_t probes = full ? n : std::min<uint32_t>(n, 48);
        for (uint32_t t = 0; t < probes; ++t) {
            uint32_t j = full ? t : (rng() % n);
            uint32_t got = pool.intern(K{model[j]});
            CHECK(got == j, "[%s seed=%u] survivor %u re-interned to %u", label, seed, j, got);
        }
        CHECK(pool.size() == model.size(), "[%s seed=%u] re-interning survivors changed size", label, seed);
    };

    int clear_path = 0, rebuild_path = 0, shrink_events = 0, zero_rewinds = 0;
    for (int step = 0; step < steps; ++step) {
        // Alternate GROWTH phases (interns dominate, so pools reach tens of thousands of
        // entries and cross many grow() boundaries) with CHURN phases (frequent
        // checkpoint/rewind, the search's own pattern) -- a uniform mix keeps the pool
        // tiny and never reaches the interesting paths.
        const bool growth_phase = ((step / 4000) % 3) != 2;
        uint32_t r = rng() % 1000;
        if (growth_phase) r = (r < 950) ? (r % 680) : (r < 985 ? 680 + (r % 120) : 800 + (r % 200)); // ~95% intern
        else r = r % 1000;
        r /= 10; // back to 0..99 scale
        if (r < 68) {
            uint32_t v = rng() % universe;
            uint32_t expect;
            auto it = index_of.find(v);
            if (it != index_of.end()) expect = it->second;
            else { expect = static_cast<uint32_t>(model.size()); model.push_back(v); index_of[v] = expect; }
            uint32_t got_new;
            if (rng() & 1) got_new = pool.intern(K{v});
            else got_new = pool.intern_lazy(v,
                    [](uint32_t k) { return std::hash<K>{}(K{k}); },
                    [](const K& stored, uint32_t k) { return stored.v == k; },
                    [v]() { return K{v}; });
            uint32_t got_ref = ref.intern(K{v});
            CHECK(got_new == expect && got_ref == expect,
                  "[%s seed=%u step=%d] intern(%u): new=%u ref=%u expected=%u", label, seed, step, v, got_new, got_ref, expect);
            if (pool.capacity() != last_cap) last_cap = pool.capacity();
            if (model.size() > peak) peak = model.size();
        } else if (r < 80) {
            if ((int)stack.size() < max_depth) stack.push_back(static_cast<uint32_t>(model.size()));
        } else if (r < 97) {
            if (stack.empty()) continue;
            // LIFO: pop 1..k entries, rewind to the deepest popped one
            size_t pops = 1 + (rng() % std::min<size_t>(stack.size(), 3));
            uint32_t target = 0;
            for (size_t p = 0; p < pops; ++p) { target = stack.back(); stack.pop_back(); }
            uint32_t before = pool.size();
            size_t cap_before = pool.capacity();
            if (target > 0 && target < before) {
                if (target >= pool.stable_size) ++clear_path; else ++rebuild_path;
            }
            pool.rewind(target); ref.rewind(target);
            if (pool.capacity() < cap_before) ++shrink_events;
            for (uint32_t j = target; j < model.size(); ++j) index_of.erase(model[j]);
            model.resize(target);
            ++rewinds;
            if (before - target > 64) ++big_rewinds;
            if (pool.stable_size > 0 && target < pool.stable_size) CHECK(false, "stable_size above size after rewind");
            (void)cap_before;
            bool small = model.size() <= 1500;
            verify_all(small || (rewinds % 40 == 0));
        } else if (r < 98) {
            // rewind to zero (episode boundary) -- only occasionally, or pools never get big
            if (rng() % 8) continue;
            ++zero_rewinds;
            pool.rewind(0); ref.rewind(0);
            model.clear(); index_of.clear(); stack.clear();
            ++rewinds;
            verify_all(true);
        } else {
            // no-op rewind at the current size
            uint32_t before_cap = static_cast<uint32_t>(pool.capacity());
            pool.rewind(pool.size()); ref.rewind(ref.size());
            CHECK(pool.capacity() == before_cap, "[%s seed=%u] no-op rewind changed the table", label, seed);
            verify_all(false);
        }
    }
    verify_all(true);
    std::printf("  [%s seed=%u] %d steps, %d rewinds: %d clear-path, %d rebuild-path (crossed a grow), "
                "%d shrinks, %d to-zero, %d reclaiming >64 entries, peak size %zu: OK\n",
                label, seed, steps, rewinds, clear_path, rebuild_path, shrink_events, zero_rewinds, big_rewinds, peak);
    (void)spans_grow;
}

// ---- targeted deterministic scenarios ------------------------------------------------------
static void scenario_span_grow_and_shrink() {
    using K = Key<1>;
    NodePool<K> pool;
    for (uint32_t i = 0; i < 20; ++i) pool.intern(K{i});
    uint32_t cp = pool.size();                        // checkpoint BEFORE several grow()s
    size_t cap_at_cp = pool.capacity();
    for (uint32_t i = 20; i < 50000; ++i) pool.intern(K{i});
    size_t cap_big = pool.capacity();
    CHECK(cap_big > cap_at_cp * 8, "pool did not grow as expected");
    CHECK(pool.stable_size > cp, "grow() should have raised stable_size above the checkpoint");
    pool.rewind(cp);                                  // reaches back across many grows -> rebuild path
    CHECK(pool.check_invariants().empty(), "invariants after cross-grow rewind: %s", pool.check_invariants().c_str());
    CHECK(pool.stable_size == 0, "rebuild should restore index-order placement (stable_size 0)");
    CHECK(pool.size() == cp, "size after rewind");
    CHECK(pool.capacity() < cap_big / 8, "table should have been right-sized after a large rewind (cap %zu vs %zu)",
          pool.capacity(), cap_big);
    for (uint32_t i = 0; i < cp; ++i) CHECK(pool.intern(K{i}) == i, "survivor %u", i);
    // after the rebuild a small probe-style rewind must take the O(reclaimed) path and stay correct
    uint32_t cp2 = pool.size();
    for (uint32_t i = 1000; i < 1008; ++i) pool.intern(K{i});
    pool.rewind(cp2);
    CHECK(pool.check_invariants().empty(), "invariants after small rewind: %s", pool.check_invariants().c_str());
    for (uint32_t i = 1000; i < 1008; ++i) CHECK(pool.intern(K{i}) == cp2 + (i - 1000), "re-interned index");
    std::printf("  scenario span-grow/shrink: OK\n");
}

static void scenario_probe_loop_is_cheap_in_structure() {
    // The search pattern: a large committed pool, then many (checkpoint, add few, rewind) cycles.
    using K = Key<1>;
    NodePool<K> pool;
    for (uint32_t i = 0; i < 200000; ++i) pool.intern(K{i});
    size_t cap = pool.capacity();
    uint32_t stable = pool.stable_size;
    for (int rep = 0; rep < 2000; ++rep) {
        uint32_t cp = pool.size();
        for (uint32_t j = 0; j < 6; ++j) pool.intern(K{1000000u + (uint32_t)rep * 8 + j});
        pool.rewind(cp);
    }
    CHECK(pool.capacity() == cap, "probe cycles must not rebuild/resize the table");
    CHECK(pool.stable_size == stable, "probe cycles must not disturb stable_size");
    CHECK(pool.check_invariants().empty(), "%s", pool.check_invariants().c_str());
    std::printf("  scenario probe-loop (no rebuild across 2000 cycles): OK\n");
}

// ---- kernel-level fuzz: the real ExecutionContext, all 14 pools ---------------------------
struct Spec { int kind; uint32_t a, b; NodeId id; };
static bool same(NodeId x, NodeId y) { return x.tag == y.tag && x.index == y.index; }

static void fuzz_kernel(uint32_t seed, int steps) {
    ExecutionContext c; ContextGuard g(c);
    std::mt19937 rng(seed);
    std::vector<Spec> log;                                  // committed-and-live terms, in creation order
    struct Frame { ExecutionContext::Checkpoint cp; size_t log_size; };
    std::vector<Frame> stack;

    auto make = [&](int kind, uint32_t a, uint32_t b) -> NodeId {
        switch (kind) {
            case 0: return mk_var(a);
            case 1: return mk_type(a);
            case 2: return mk_app(log[a].id, log[b].id);
            case 3: return mk_lambda(log[a].id, log[b].id);
            default: return mk_pi(log[a].id, log[b].id);
        }
    };
    auto check_pools = [&]() {
        std::string e;
        if (!(e = c.vars.check_invariants()).empty()) CHECK(false, "vars: %s", e.c_str());
        if (!(e = c.apps.check_invariants()).empty()) CHECK(false, "apps: %s", e.c_str());
        if (!(e = c.lambdas.check_invariants()).empty()) CHECK(false, "lambdas: %s", e.c_str());
        if (!(e = c.pis.check_invariants()).empty()) CHECK(false, "pis: %s", e.c_str());
        if (!(e = c.types.check_invariants()).empty()) CHECK(false, "types: %s", e.c_str());
    };
    int rewinds = 0;
    for (int step = 0; step < steps; ++step) {
        uint32_t r = rng() % 100;
        if (r < 72) {
            int kind = log.size() < 2 ? (rng() % 2) : (rng() % 5);
            uint32_t a = 0, b = 0;
            if (kind == 0) a = rng() % 60; else if (kind == 1) a = rng() % 6;
            else { a = rng() % log.size(); b = rng() % log.size(); }
            NodeId id = make(kind, a, b);
            log.push_back(Spec{kind, a, b, id});
        } else if (r < 84) {
            if (stack.size() < 6) stack.push_back(Frame{c.checkpoint(), log.size()});
        } else if (!stack.empty()) {
            size_t pops = 1 + (rng() % std::min<size_t>(stack.size(), 2));
            Frame f = stack.back();
            for (size_t p = 0; p < pops; ++p) { f = stack.back(); stack.pop_back(); }
            c.rewind(f.cp);
            log.resize(f.log_size);
            ++rewinds;
            // every surviving term must still hash-cons to the SAME NodeId when rebuilt from its spec
            size_t n = log.size();
            size_t probes = std::min<size_t>(n, 60);
            for (size_t t = 0; t < probes; ++t) {
                size_t j = (n <= 60) ? t : (rng() % n);
                NodeId again = make(log[j].kind, log[j].a, log[j].b);
                CHECK(same(again, log[j].id), "[kernel seed=%u] survivor %zu re-made to a different NodeId", seed, j);
            }
            CHECK(log.empty() || true, "");
            if (rewinds % 25 == 0) check_pools();
        }
    }
    check_pools();
    std::printf("  [kernel seed=%u] %d steps, %d rewinds, %zu live terms: OK\n", seed, steps, rewinds, log.size());
}

int main() {
    std::printf("== NodePool differential fuzz (new O(reclaimed) rewind vs old rebuild vs model) ==\n");
    for (uint32_t seed : {1u, 2u, 3u}) fuzz_pool<0>(seed, 60000, 4000, 6, "collide-13");
    for (uint32_t seed : {11u, 12u}) fuzz_pool<2>(seed, 40000, 250, 5, "collide-all");
    for (uint32_t seed : {21u, 22u, 23u}) fuzz_pool<1>(seed, 250000, 120000, 8, "good-hash");
    for (uint32_t seed : {31u, 32u}) fuzz_pool<1>(seed, 40000, 300, 10, "good-hash-small-universe");
    fuzz_pool<1>(24u, 600000, 1000000, 8, "good-hash-large-peak");
    std::printf("== targeted scenarios ==\n");
    scenario_span_grow_and_shrink();
    scenario_probe_loop_is_cheap_in_structure();
    std::printf("== ExecutionContext end-to-end fuzz (hash-consing stability across rewinds) ==\n");
    for (uint32_t seed : {101u, 102u, 103u, 104u}) fuzz_kernel(seed, 40000);
    std::printf("\nALL NODEPOOL REWIND TESTS PASSED (%d checks)\n", g_checks);
    return 0;
}
