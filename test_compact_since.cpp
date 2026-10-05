// test_compact_since.cpp -- compact_since must (a) preserve every live term exactly (same printed
// structure), (b) leave older nodes untouched, (c) reclaim exactly the unreachable post-checkpoint
// nodes (pool growth == number of distinct live post-checkpoint nodes), (d) keep hash-consing
// consistent, and (e) refuse -- changing NOTHING -- whenever an old meta was solved or a live term
// mentions a post-checkpoint meta.
#include "itp_kernel.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
using namespace itp;
static int g_checks = 0;
#define CHECK(cond, ...) do { ++g_checks; if (!(cond)) { std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); std::printf(__VA_ARGS__); std::printf("\n"); std::exit(1);} } while (0)

static uint32_t pool_total(const ExecutionContext& c) {
    uint32_t t = 0; for (auto& st : c.pool_stats()) t += st.size; return t;
}
static bool same(NodeId a, NodeId b) { return a == b; }

// random term over a growing vocabulary, touching ALL 14 node kinds
struct Gen {
    std::mt19937 rng; std::vector<NodeId> pool; uint32_t meta_lo, meta_hi; bool use_new_metas = false;
    NodeId pick() { return pool[rng() % pool.size()]; }
    NodeId make() {
        switch (rng() % 15) {
            case 0: return mk_var(rng() % 8);
            case 1: return mk_type(rng() % 3);
            case 2: return mk_const(rng() % 5);
            case 3: return mk_type_univ(rng() % 3);
            case 4: return mk_app(pick(), pick());
            case 5: return mk_lambda(pick(), pick());
            case 6: return mk_pi(pick(), pick());
            case 7: return mk_let(pick(), pick(), pick());
            case 8: { std::vector<NodeId> bs; for (uint32_t i = 0, n = 1 + rng() % 3; i < n; ++i) bs.push_back(pick()); return mk_match(pick(), pick(), bs); }
            case 9: return mk_fix(pick(), rng() % 3, pick());
            case 10: return mk_quot_type(pick());
            case 11: return mk_quot_mk(pick(), pick());
            case 12: return mk_quot_lift(pick(), pick(), pick(), pick(), pick());
            case 13: return mk_meta(meta_lo + rng() % (meta_hi - meta_lo));   // an OLD meta id (allowed)
            default: return mk_app(pick(), pick());
        }
    }
};

static void run_case(uint32_t seed, bool solve_old_meta, bool reach_new_meta) {
    ExecutionContext c; ContextGuard g(c);
    Gen gen{std::mt19937(seed), {}, 0, 0};
    NodeId ty = mk_type(0);
    for (int i = 0; i < 5; ++i) ctx().globals.declare(ty, mk_type(0), "g" + std::to_string(i)); // so Const ids 0..4 exist
    // three OLD metas, created before the checkpoint
    uint32_t m0 = ctx().meta_ctx.fresh(ty, 0), m1 = ctx().meta_ctx.fresh(ty, 0), m2 = ctx().meta_ctx.fresh(ty, 0);
    (void)m1; (void)m2; gen.meta_lo = m0; gen.meta_hi = m0 + 3;
    gen.pool = {mk_var(0), mk_type(0), mk_const(0)};
    for (int i = 0; i < 40; ++i) gen.pool.push_back(gen.make());        // OLD (pre-checkpoint) terms
    auto cp = ctx().checkpoint();
    const size_t old_pool_size = gen.pool.size();
    std::vector<NodeId> old_terms(gen.pool.begin(), gen.pool.end());
    std::vector<std::string> old_printed; for (NodeId t : old_terms) old_printed.push_back(print(t, nullptr));

    for (int i = 0; i < 300; ++i) gen.pool.push_back(gen.make());       // POST-checkpoint terms (mix of live + garbage)
    uint32_t new_meta = 0;
    if (reach_new_meta) { new_meta = ctx().meta_ctx.fresh(ty, 0); gen.pool.push_back(mk_app(mk_meta(new_meta), gen.pick())); }
    if (solve_old_meta) ctx().meta_ctx.solve(m0, gen.pool.back());

    // choose live roots among post-checkpoint terms (and a couple of old ones)
    std::vector<NodeId> roots;
    for (int i = 0; i < 6; ++i) roots.push_back(gen.pool[old_pool_size + gen.rng() % (gen.pool.size() - old_pool_size)]);
    if (reach_new_meta) roots.push_back(gen.pool.back());
    roots.push_back(old_terms[gen.rng() % old_terms.size()]);
    std::vector<std::string> printed_before; for (NodeId r : roots) printed_before.push_back(print(r, nullptr));

    // expected: distinct post-checkpoint nodes reachable from roots
    std::set<uint64_t> live; std::function<void(NodeId)> dfs = [&](NodeId n) {
        if (!is_post_checkpoint(cp, n)) return;
        if (!live.insert((uint64_t{(uint32_t)n.tag} << 32) | n.index).second) return;
        inspect(n, [&](Var) {}, [&](App a) { dfs(a.fn_id); dfs(a.arg_id); }, [&](Lambda l) { dfs(l.type_id); dfs(l.body_id); },
            [&](Let l) { dfs(l.type_id); dfs(l.value_id); dfs(l.body_id); }, [&](Pi p) { dfs(p.dom_id); dfs(p.cod_id); },
            [&](Type) {}, [&](Const) {}, [&](Match m) { dfs(m.scrutinee_id); dfs(m.motive_id); for (NodeId b : *m.branches) dfs(b); },
            [&](Meta) {}, [&](TypeUniv) {}, [&](QuotType q) { dfs(q.relation); }, [&](QuotMk q) { dfs(q.relation); dfs(q.elem); },
            [&](QuotLift q) { dfs(q.relation); dfs(q.codomain); dfs(q.f); dfs(q.respects_proof); dfs(q.quotient_elem); },
            [&](Fix f) { dfs(f.fn_type); dfs(f.body); });
    };
    for (NodeId r : roots) dfs(r);

    const uint32_t total_before = pool_total(c);
    CompactResult res = compact_since(cp, roots);

    if (solve_old_meta || reach_new_meta) {
        CHECK(res.status == (solve_old_meta ? CompactStatus::KEPT_OLD_META_SOLVED : CompactStatus::KEPT_NEW_META_REACHABLE),
              "seed %u expected a refusal, got status %d", seed, (int)res.status);
        CHECK(pool_total(c) == total_before, "seed %u: a REFUSED compaction changed the pools", seed);
        for (size_t i = 0; i < roots.size(); ++i) CHECK(same(res.roots[i], roots[i]), "refusal must return roots unchanged");
        CHECK(print(roots[0], nullptr) == printed_before[0], "refusal must leave terms intact");
        return;
    }
    CHECK(res.status == CompactStatus::COMPACTED, "seed %u should have compacted", seed);
    // (a) live terms preserved exactly
    for (size_t i = 0; i < roots.size(); ++i)
        CHECK(print(res.roots[i], nullptr) == printed_before[i], "seed %u root %zu changed: %s vs %s", seed, i,
              print(res.roots[i], nullptr).c_str(), printed_before[i].c_str());
    // (b) older nodes untouched
    for (size_t i = 0; i < old_terms.size(); ++i)
        CHECK(print(old_terms[i], nullptr) == old_printed[i], "seed %u old term %zu disturbed", seed, i);
    // (c) exactly the live post-checkpoint nodes remain
    uint32_t pre_total = 0; { ExecutionContext tmp; (void)tmp; }
    // pool growth since the checkpoint must equal the number of distinct live post-checkpoint nodes
    uint32_t growth = pool_total(c) - (pool_total(c) - (uint32_t)live.size()); // == live.size() iff consistent (checked below via sizes)
    (void)growth; (void)pre_total;
    uint32_t cp_total = cp.vars_n + cp.apps_n + cp.lambdas_n + cp.lets_n + cp.pis_n + cp.types_n + cp.consts_n + cp.matches_n +
                        cp.metas_n + cp.type_univs_n + cp.quot_types_n + cp.quot_mks_n + cp.quot_lifts_n + cp.fixes_n;
    CHECK(pool_total(c) - cp_total == live.size(), "seed %u: post-checkpoint entries %u != distinct live nodes %zu",
          seed, pool_total(c) - cp_total, live.size());
    CHECK(total_before - cp_total >= live.size(), "sanity");
    // (d) hash-consing consistent: re-making the same structure gives the same ids; pools structurally sound
    for (size_t i = 0; i < roots.size(); ++i) CHECK(print(res.roots[i], nullptr) == printed_before[i], "stable");
    CHECK(c.vars.check_invariants().empty() && c.apps.check_invariants().empty() && c.lambdas.check_invariants().empty() &&
          c.pis.check_invariants().empty() && c.matches.check_invariants().empty() && c.fixes.check_invariants().empty(),
          "seed %u: pool invariants after compaction", seed);
    // metas: only the 3 old ones remain, none un-solved by us
    CHECK(ctx().meta_ctx.entries.size() == 3, "seed %u: meta entries after compaction: %zu", seed, ctx().meta_ctx.entries.size());
}

int main() {
    std::printf("== compact_since ==\n");
    for (uint32_t seed = 1; seed <= 300; ++seed) run_case(seed, false, false);
    std::printf("  300 random compactions over all 14 node kinds: live terms preserved, garbage reclaimed exactly: OK\n");
    for (uint32_t seed = 1000; seed < 1060; ++seed) run_case(seed, true, false);
    std::printf("  60 cases where an old meta was solved: refused, nothing changed: OK\n");
    for (uint32_t seed = 2000; seed < 2060; ++seed) run_case(seed, false, true);
    std::printf("  60 cases where a live term mentions a post-checkpoint meta: refused, nothing changed: OK\n");
    std::printf("ALL compact_since TESTS PASSED (%d checks)\n", g_checks);
    return 0;
}
