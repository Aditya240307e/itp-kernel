"""Mutation testing of the S2 typing core: each mutant re-breaks one rule in an isolated copy; the test suite must FAIL."""
import os, re, shutil, subprocess, sys, tempfile, time
SRC = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.environ.get('ITP_PY_ROOT', os.path.dirname(SRC))
REF_DIR = os.environ.get('ITP_REF_DIR', os.path.join(os.path.dirname(PY_ROOT), 'itp_ref'))
def gate_env(lnl_dir, **extra):   # one place that says where the three source trees live, for the baseline and for every mutant copy
    return dict(os.environ, LNL_DIR=lnl_dir, ITP_PY_ROOT=PY_ROOT, ITP_REF_DIR=REF_DIR, PYTHONPATH=os.pathsep.join([REF_DIR, PY_ROOT, lnl_dir]), **extra)
MUTANTS = {
 'M83 declare_quot ignores a name clash': ('        if (d.name == "Quot" || d.name == "Quot.mk" || d.name == "Quot.lift" || d.name == "Quot.ind" || d.name == "Quot.sound") return fail("a global named " + d.name + " already exists");\n', ''),
 'M100 F27: no function eta in conv_value: Fix vs Closure': ('        [&](const VFix& fa, const Closure& cb) -> bool {\n', '        [&](const VFix& fa, const Closure& cb) -> bool {\n            return false;\n'),
 'M99 F27: no function eta in conv_value: Closure vs Fix': ('        [&](const Closure& ca, const VFix& fb) -> bool {\n', '        [&](const Closure& ca, const VFix& fb) -> bool {\n            return false;\n'),
 'M98 F27: no function eta in conv_value: neutral vs Closure': ('        [&](const VNeutral& na, const Closure& cb) -> bool {\n', '        [&](const VNeutral& na, const Closure& cb) -> bool {\n            return false;\n'),
 'M97 F27: no function eta in conv_value: Closure vs neutral': ('        [&](const Closure& ca, const VNeutral& nb) -> bool {\n', '        [&](const Closure& ca, const VNeutral& nb) -> bool {\n            return false;\n'),
 'M96 F27: no function eta in the unifier (type checking): Fix vs Closure': ('        [&](const VFix& fa, const Closure& cb) -> ElabResult {\n', '        [&](const VFix& fa, const Closure& cb) -> ElabResult {\n            return mismatch("eta disabled");\n'),
 'M95 F27: no function eta in the unifier (type checking): Closure vs Fix': ('        [&](const Closure& ca, const VFix& fb) -> ElabResult {\n', '        [&](const Closure& ca, const VFix& fb) -> ElabResult {\n            return mismatch("eta disabled");\n'),
 'M94 F27: no function eta in the unifier (type checking): neutral vs Closure': ('        [&](const VNeutral& na, const Closure& cb) -> ElabResult {\n', '        [&](const VNeutral& na, const Closure& cb) -> ElabResult {\n            return mismatch("eta disabled");\n'),
 'M93 F27: no function eta in the unifier (type checking): Closure vs neutral': ('        [&](const Closure& ca, const VNeutral& nb) -> ElabResult {\n', '        [&](const Closure& ca, const VNeutral& nb) -> ElabResult {\n            return mismatch("eta disabled");\n'),
 'M92 F26: a finished inductive family is not marked inductive': ('    c.globals.defs[type_global_id].is_inductive = true;\n    c.globals.defs[type_global_id].reserved = GlobalDef::Reserved::None;\n    return handle;', '    c.globals.defs[type_global_id].reserved = GlobalDef::Reserved::None;\n    return handle;'),
 'M91 F26: the quotient constants are declared inductive': ('uint32_t g = c.globals.declare(ty, std::nullopt, name); c.globals.defs[g].quot_role = role; return g; };', 'uint32_t g = c.globals.declare(ty, std::nullopt, name); c.globals.defs[g].quot_role = role; c.globals.defs[g].is_inductive = true; return g; };'),
 'M90 F26: cases and induct accept a hypothesis of an opaque type': ('    if (!ctx().globals[si.type_gid].is_inductive)\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, scrut_ty, std::nullopt, std::string(who) + ": hypothesis\'s type is not an inductive type application"});\n', ''),
 'M89 F26: check(Match) accepts a scrutinee of an opaque type': ('    const uint32_t type_global_id = scrut_neutral->head.id;\n    if (!ctx().globals[type_global_id].is_inductive)\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, expected_ty, "scrutinee\'s type is not an inductive type application"});\n', '    const uint32_t type_global_id = scrut_neutral->head.id;\n'),
 'M88 F26: infer(Match) accepts a scrutinee of an opaque type': ('            const uint32_t type_global_id = scrut_neutral->head.id;\n            if (!ctx().globals[type_global_id].is_inductive)\n                return err(TypeErrorKind::MotiveInferenceFailure, m.scrutinee_id, "scrutinee\'s type is not an inductive type application");\n', '            const uint32_t type_global_id = scrut_neutral->head.id;\n'),
 'M87 Quot.lift is declared without its role': ('declare(lift_ty, "Quot.lift", QuotRole::Lift)', 'declare(lift_ty, "Quot.lift", QuotRole::None)'),
 'M86 Quot.sound: the premise has its arguments swapped (r b a)': ('close_pi_telescope({Tj, rel, A, A, apply_all(r, {a2, b3})},', 'close_pi_telescope({Tj, rel, A, A, apply_all(r, {b3, a2})},'),
 'M85 declare_quot does not check the type of the equality family': ('    if (eq.type != close_pi_telescope({Tj, A, A}, Prop, 0)) return fail(', '    if (false && eq.type != close_pi_telescope({Tj, A, A}, Prop, 0)) return fail('),
 'M84 declare_quot accepts an equality family with several constructors': ('if (eq_ctors.size() != 1) return fail(', 'if (eq_ctors.size() < 1) return fail('),
 'M82 declare_quot may be called twice': ('        if (d.quot_role != QuotRole::None) return fail("quotients are already declared");\n', ''),
 'M81 declare_quot accepts j = 0': ('if (first_dom.tag != NodeType::TYPE || c.types[first_dom.index].universe_level == 0) return fail(', 'if (first_dom.tag != NodeType::TYPE) return fail('),
 "M80 declare_quot does not check the equality family's constructor": ('    if (c.globals[eq_ctors[0]].type != close_pi_telescope({Tj, A}, apply_all(EQ, {A, r, r}), 0)) return fail(', '    if (false && c.globals[eq_ctors[0]].type != close_pi_telescope({Tj, A}, apply_all(EQ, {A, r, r}), 0)) return fail('),
 'M79 declare_quot accepts a world without prop_elim': ('if (!c.prop_mode) return fail("quotients live in the Prop/Type world', 'if (false && !c.prop_mode) return fail("quotients live in the Prop/Type world'),
 'M78 quotient rule is never applied': ('            if (def.quot_role == QuotRole::Lift) {', '            if (false && def.quot_role == QuotRole::Lift) {'),
 'M77 quotient rule fires on ANY global applied to three arguments (the Mk role is not checked)': ('    if (ctx().globals[q->head.id].quot_role != QuotRole::Mk) return std::nullopt;\n', ''),
 'M76 quotient rule: a is taken from the wrong frame': ('return std::make_pair(std::get<ApplyFrame>(n.spine[3]).arg, std::get<ApplyFrame>(q->spine[2]).arg);', 'return std::make_pair(std::get<ApplyFrame>(n.spine[3]).arg, std::get<ApplyFrame>(q->spine[1]).arg);'),
 'M75 quotient rule: f is taken from the wrong frame': ('return std::make_pair(std::get<ApplyFrame>(n.spine[3]).arg, std::get<ApplyFrame>(q->spine[2]).arg);', 'return std::make_pair(std::get<ApplyFrame>(n.spine[2]).arg, std::get<ApplyFrame>(q->spine[2]).arg);'),
 'M74 Quot.ind: the motive may land in Type j': ('const NodeId P_ty = close_pi_telescope({QAr}, Prop, 100);', 'const NodeId P_ty = close_pi_telescope({QAr}, Tj, 100);'),
 'M73 Quot.sound concludes mk a = mk a': ('apply_all(EQ, {QAr, apply_all(MK, {A, r, a2}), apply_all(MK, {A, r, b3})}), 0);', 'apply_all(EQ, {QAr, apply_all(MK, {A, r, a2}), apply_all(MK, {A, r, a2})}), 0);'),
 'M72 Quot.lift: h only asks f a = f a (a vacuous respects-proof)': ('apply_all(EQ, {B, mk_app(f, mk_fvar(100)), mk_app(f, mk_fvar(101))}), 100);', 'apply_all(EQ, {B, mk_app(f, mk_fvar(100)), mk_app(f, mk_fvar(100))}), 100);'),
 'M71 apply does not check that the solutions unification found have their arguments\' types (F25)': ("        if (!solutions_ok) return std::unexpected(TypeError{TypeErrorKind::Mismatch, g.target, std::nullopt, \"apply: the solution unification found for an argument of the lemma does not have that argument's type (\" + why + \")\"});\n", ""),
 'M67 prop_elim: Match does not enforce the elimination restriction': ("    if (ctx().prop_mode && ctx().prop_elim_only.count(type_global_id)) {      // a proposition that is not a syntactic subsingleton: eliminable only into Type0", "    if (false && ctx().prop_mode && ctx().prop_elim_only.count(type_global_id)) {      // a proposition that is not a syntactic subsingleton: eliminable only into Type0"),
 'M68 prop_elim: a family with two constructors counts as a subsingleton': ("    bool subsingleton = ctors.size() <= 1;\n    if (ctors.size() == 1) {", "    bool subsingleton = ctors.size() <= 2;\n    if (ctors.size() == 1) {"),
 'M69 prop_elim: fields that occur in the result indices are not exempt (Eq becomes restricted)': ("            if (!proof_field && !in_indices) subsingleton = false;", "            if (!proof_field) subsingleton = false;"),
 'M70 prop_elim: cases does not refuse a data goal on a restricted proposition': ("        if (verify_motive_shape(motive_original, arity, g.ec, ps.es, &elim_level) && elim_level != 0)\n            return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt, \"cases: the hypothesis's type", "        if (false && verify_motive_shape(motive_original, arity, g.ec, ps.es, &elim_level) && elim_level != 0)\n            return std::unexpected(TypeError{TypeErrorKind::Other, g.target, std::nullopt, \"cases: the hypothesis's type"),
 'M66 parameter-mode induct does not wrap its Fix in the barrier unless equations are used (F23)': ("    if (use_eq || ctx().param_mode) {   // F23: keep the Fix opaque to extraction (see get_or_build_barrier): the generic body is then what the kernel (and the independent checker) verify", "    if (use_eq) {   // F23: keep the Fix opaque to extraction (see get_or_build_barrier)"),
 'M65 rewind does not treat the barrier as a derived global (a probe that creates it cannot be rewound)': (' || nm.rfind("__barrier_", 0) == 0;   // __barrier_k: built by name on demand (get_or_build_barrier), so a probe may create and drop it', ';'),
 'M63 induct with equations does not wrap its Fix in the extraction barrier (F23)': ("        if (sty.tag == NodeType::TYPE) proof = mk_app(mk_app(mk_const(get_or_build_barrier(ctx().types[sty.index].universe_level)), fn_type), proof);\n", ""),
 'M64 the library index offers the barrier as a candidate for every goal': ("        if (ctx().globals.defs[gid].name.rfind(\"__barrier_\", 0) == 0) continue;      // an identity at every type: it would be a candidate for every goal\n", ""),
 'M62 cases: refutation is not exploited when hypotheses are reverted, in parameter mode (F22)': ("    if (!reverted_js.empty() && !ctx().param_mode) exploit_ci.reset();\n", "    if (!reverted_js.empty()) exploit_ci.reset();\n"),
 'M61 induct accepts a concrete index in parameter mode (F12 guard removed)': ('    if (n_bad > 0 && !use_eq && ctx().param_mode) {\n        uint32_t first_bad = 0;', '    if (false && n_bad > 0 && !use_eq && ctx().param_mode) {\n        uint32_t first_bad = 0;'),
 'M60 a lambda checked against a Pi need not have a well-typed binder annotation (F21)': ("        if (dom_j != expected_dom) { auto sdom = sort_level_of(dom_j, cur_ec, es, \"Lambda domain must be a type\"); if (!sdom) return std::unexpected(sdom.error()); }\n", ""),
 'M59 UNSOUND unifier: different constructors unify (the canary must notice)': ("return mismatch(\"different constructors\");", "return ElabResult{};"),
 'M49 parameter analysis ignores uniformity of recursive occurrences': ("            for (NodeId ft : inst.arg_types) if (!uniform_family_occurrences(ft, family_gid, arity, i, r)) { ok = false; break; }", "            for (NodeId ft : inst.arg_types) if (false && !uniform_family_occurrences(ft, family_gid, arity, i, r)) { ok = false; break; }"),
 'M50 parameter analysis ignores distinctness from earlier parameters': ("            for (uint32_t k = 0; k < i; ++k) if (inst.result_indices[k] == r) { ok = false; break; }   // distinct from the earlier parameters' variables", "            for (uint32_t k = 0; k < i; ++k) if (false && inst.result_indices[k] == r) { ok = false; break; }"),
 'M51 parameters need not form a prefix': ("        if (!ok) break;                                                       // parameters form a PREFIX", "        if (!ok) continue;"),
 'M52 field forced by a parameter is not substituted in the constructor application': ("const bool use_forced = it != forcing.forced.end() && (occurrence_count[base + ai] <= 1 || forcing.param_vars.count(base + ai));", "const bool use_forced = it != forcing.forced.end() && occurrence_count[base + ai] <= 1;"),
 'M53 field forced by a parameter is not substituted in the result indices': ("if (occurrence_count[level] <= 1 || forcing.param_vars.count(level)) m[level] = replacement;", "if (occurrence_count[level] <= 1) m[level] = replacement;"),
 'M54 cases still generalises the parameter slots': ("    for (uint32_t i = 0; i < arity; ++i) if (!is_param[i]) subs.push_back({concrete_indices[i], mk_fvar(n + i)});\n    subs.push_back({mk_fvar(hyp_level), mk_fvar(n + arity)});\n    // cases reverts a CONTIGUOUS suffix", "    for (uint32_t i = 0; i < arity; ++i) subs.push_back({concrete_indices[i], mk_fvar(n + i)});\n    subs.push_back({mk_fvar(hyp_level), mk_fvar(n + arity)});\n    // cases reverts a CONTIGUOUS suffix"),
 'M55 motive binder types are not specialised to the parameters': ("mp.doms.push_back(pm.empty() ? index_types[i] : replace_fvars(index_types[i], pm));", "mp.doms.push_back(index_types[i]);"),
 'M56 only the first refuted constructor is exploited in parameter mode': ("        if (ctx().param_mode)\n            for (std::size_t cj = 0;", "        if (false)\n            for (std::size_t cj = 0;"),
 'M57 the Fix is generic over the parameters too': ("idx_vars_body.push_back(is_param[i] ? concrete_indices[i] : mk_fvar(n + 1 + rank[i]));", "idx_vars_body.push_back(mk_fvar(n + 1 + (is_param[i] ? 0u : rank[i])));"),
 'M58 the kernel Match typing ignores parameters': ("param_slots_v.empty() ? nullptr : &param_slots_v);", "nullptr);"),
 'M47 induct ignores the generalizing list (F6)': ("    for (uint32_t j : generalizing) if (j < g.hyps.size() && j != hyp_level) needs_revert[j] = true;   // F6: explicitly generalised hypotheses, then the closure\n", ""),
 'M48 cases ignores the generalizing list (F6)': ("    for (uint32_t j : generalizing) if (j < hyp_level) revert_from = std::min(revert_from, j);   // F6: explicitly generalised hypotheses (those after the scrutinee are reverted anyway)", ""),
 'M42 scope validation accepts everything': ("    if (loose_range(t) != 0) return false;\n    std::unordered_set<uint64_t> seen; return fvars_below(t, nfree, seen);", "    return true;"),
 'M43 scope validation ignores context variables beyond the context': ("return v.bound || v.n < nfree;", "return true;"),
 'M44 scope validation ignores loose bound variables': ("    if (loose_range(t) != 0) return false;\n    std::unordered_set<uint64_t> seen;", "    std::unordered_set<uint64_t> seen;"),
 'M45 lambda binder domain need not be a type (F19)': ("auto s = sort_level_of(dj, cur, es, \"Lambda domain must be a type\"); if (!s) return std::unexpected(s.error());", "auto s = sort_level_of(dj, cur, es, \"Lambda domain must be a type\"); if (!s) s = 0u;"),
 'M46 intro on a non-literal Pi uses the raw ids (F17, the reference behaviour)': ("dom_ty = quote(eval(pi->dom_id, pi->env), n);\n        new_target = quote(eval(pi->cod_id, pi->env.extend(fresh_neutral(n))), n + 1);", "dom_ty = pi->dom_id;\n        new_target = pi->cod_id;"),
 'M41 identical metas no longer unify (the quirk returns)': ("    if (is_meta_headed(a) && is_meta_headed(b) && quote(a, ec.size()) == quote(b, ec.size())) return {};\n", ""),
 'M37 rewind does not clear the loose_range memo': ('        loose_cache.clear(); param_cache.clear();', '        param_cache.clear();'),
 'M39 rewind does not roll back the metas': ("        meta_ctx.rewind(cp.meta_mark);", ""),
 'M40 rewind keeps stale same_ctor cache entries': ("it = (it->first >= keep || it->second >= keep) ? same_ctor_gid_by_family.erase(it) : std::next(it);", "it = std::next(it);"),
 'M32 index keys: a bound variable head ignores the context size': ("return pack_head_key(HeadKind::LocalVar, base + depth - 1 - v.n);", "return pack_head_key(HeadKind::LocalVar, depth - 1 - v.n);"),
 'M33 index: a hypothesis keyed as if it were the first one': ("detail::conclusion_head_key(g.hyps[lvl].type, lvl) == goal_key", "detail::conclusion_head_key(g.hyps[lvl].type, 0) == goal_key"),
 'M34 index: a lemma whose ARGUMENTS are variable-headed is not generic': ("    for (NodeId a : pc.args) if (is_var_headed(a)) return true;\n    return false;", "    return false;"),
 'M35 index: generic lemmas never offered': ("        if (auto key = detail::conclusion_head_key(goal_type, base)) {\n            auto it = generic_by_outer_head.find(*key);", "        if (auto key = detail::conclusion_head_key(goal_type, base); false) {\n            auto it = generic_by_outer_head.find(*key);"),
 'M36 index: hypothesis candidate offered at the wrong level': ("result.push_back(mk_fvar(lvl));\n    return result;", "result.push_back(mk_fvar(lvl + 1));\n    return result;"),
 'M31 drain_pending re-unifies a deferred constraint in the outer context (the abort)': ("ElabCtx local = pc.ctx_at_deferral ? *pc.ctx_at_deferral : ec;", "ElabCtx local = ec;"),
 'M22 cases: scrutinee applied to the field binders (F9 regression)': ("NodeId ctor_app = forced_ctor_app(ctor_gid, n, ctor_arity, inst, forcing);", "NodeId ctor_app = mk_const(ctor_gid); for (uint32_t ai = 0; ai < ctor_arity; ++ai) ctor_app = mk_app(ctor_app, mk_fvar(n + ai));"),
 'M23 induct (goal side): scrutinee applied to the field binders': ("NodeId agent_ctor_app = forced_ctor_app(ctor_gid, n, ctor_arity, inst_agent, forcing_agent, &actual);", "NodeId agent_ctor_app = mk_const(ctor_gid); for (uint32_t fi = 0; fi < ctor_arity; ++fi) agent_ctor_app = mk_app(agent_ctor_app, mk_fvar(actual[fi]));"),
 'M24 induct (runtime side): scrutinee applied to the field binders': ("NodeId true_ctor_app = forced_ctor_app(ctor_gid, body_base, ctor_arity, inst, forcing_body);", "NodeId true_ctor_app = mk_const(ctor_gid); for (uint32_t ai = 0; ai < ctor_arity; ++ai) true_ctor_app = mk_app(true_ctor_app, mk_fvar(body_base + ai));"),
 'M25 kernel Match typing: expected type uses the field binders': ("NodeId ctor_app = forced_ctor_app(ctor_gids[ci], ec.size(), arity, inst, forcing);", "NodeId ctor_app = mk_const(ctor_gids[ci]); for (uint32_t ai = 0; ai < arity; ++ai) ctor_app = mk_app(ctor_app, mk_fvar(ec.size() + ai));"),
 'M26 forced application ignores the repeated-field rule': ('const bool use_forced = it != forcing.forced.end() && (occurrence_count[base + ai] <= 1 || forcing.param_vars.count(base + ai));', 'const bool use_forced = it != forcing.forced.end();'),
 'M27 rewrite: goal not instantiated before abstraction (F13)': ("NodeId target_z = resolve_metas(g.target);", "NodeId target_z = g.target;"),
 'M28 rewrite: motive not type-checked (F13)': ("    if (!infer(motive, g.ec, ps.es))\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,\n            \"rewrite_eq: the motive", "    if (false)\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,\n            \"rewrite_eq: the motive"),
 'M29 conversion nesting bound removed (the segfault)': ("    if (conv_nesting() > kMaxConvNesting) return false;\n", ""),
 'M30 induct accepted on a family without recursive constructors': ("if (!any_recursive)", "if (false)"),
 'M15 cases: no revert of dependent hypotheses': ("for (uint32_t j = 0; j < hyp_level; ++j) if (replace_terms(g.hyps[j].type, subs) != g.hyps[j].type) { revert_from = j; break; }", "for (uint32_t j = 0; j < 0u; ++j) if (replace_terms(g.hyps[j].type, subs) != g.hyps[j].type) { revert_from = j; break; }"),
 'M16 induct: induction hypothesis not added to the context': ("branch_ec = branch_ec.extend(ih_ty); branch_hyps.push_back(Hypothesis{\"\", ih_ty});", ""),
 'M17 induct: field levels assumed consecutive (ignore the IH shift)': ("actual[fi] = branch_ec.size();", "actual[fi] = n + fi;"),
 'M18 cases: delta not suppressed in unreverted branch targets': ("NoDeltaScope keep_constants_folded;\n            branch_expected = quote(eval(branch_expected, be_env), n + ctor_arity);", "branch_expected = quote(eval(branch_expected, be_env), n + ctor_arity);"),
 'M19 induct: revert closure is a single pass, not a fixpoint': ("needs_revert[j] = true; changed = true;", "needs_revert[j] = true;"),
 'M20 cases: refutation exploitation disabled': ('if (!reverted_js.empty() && !ctx().param_mode) exploit_ci.reset();', 'exploit_ci.reset();'),
 'M21 induct: reverted layer opened with the wrong variable': ("remaining = instantiate1(p.cod_id, mk_fvar(base_level + static_cast<uint32_t>(ri)));", "remaining = instantiate1(p.cod_id, mk_fvar(base_level + static_cast<uint32_t>(ri) + 1));"),
 'M9 solve_pattern occurs check disabled': ("    if (occurs_meta(meta_id, other_id))\n        return std::unexpected(TypeError{TypeErrorKind::OccursCheckFailure, other_id, std::nullopt, \"metavariable occurs in its own proposed solution\"});", "    if (false)\n        return std::unexpected(TypeError{TypeErrorKind::OccursCheckFailure, other_id, std::nullopt, \"metavariable occurs in its own proposed solution\"});"),
 'M10 intro does not instantiate the codomain': ("dom_ty = p.dom_id; new_target = instantiate1(p.cod_id, mk_fvar(n));", "dom_ty = p.dom_id; new_target = p.cod_id;"),
 'M11 rewrite substitutes x instead of y': ("NodeId new_target = replace_fvars(body_open, {{n, y}});", "NodeId new_target = replace_fvars(body_open, {{n, x}});"),
 'M12 apply skips drain_pending': ('    if (ElabResult r = drain_pending(g.ec, ps.es); !r) return r;\n    {   // F25:', '    {   // F25:'),
 'M13 close_lambda_telescope domain prefix off by one': ("result = mk_lambda(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + j)), result);", "result = mk_lambda(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + std::min<int>(j + 1, (int)m))), result);"),
 'M14 meta applied to all but the last context variable': ("    for (uint32_t i = 0; i < n; ++i) r = mk_app(r, mk_fvar(i));\n    return r;", "    for (uint32_t i = 0; i + 1 < n; ++i) r = mk_app(r, mk_fvar(i));\n    return r;"),
 'M1 guard disabled (fix_guard_ok always true)': ("    return is_locally_structurally_decreasing(n0, tid, f.decreasing_arg, to_levels(f.body, n0 + 1));", "    (void)tid; return true;"),
 'M2 index forcing is a no-op': ("    std::unordered_map<uint32_t, NodeId> m(forcing.forced.begin(), forcing.forced.end());\n    std::vector<NodeId> out = arg_types;", "    std::unordered_map<uint32_t, NodeId> m;\n    std::vector<NodeId> out = arg_types;"),
 'M3 lambda domain not unified with the expected domain': ("        if (ElabResult r = unify(dom_j, expected_dom, cur_ec, es); !r) return r;\n        Value next", "        (void)expected_dom;\n        Value next"),
 'M4 closing a Pi telescope off by one (wrong fvar prefix)': ("result = mk_pi(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + j))", "result = mk_pi(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + std::min<int>(j + 1, (int)m)))"),
 'M5 application argument not checked': ("            if (ElabResult r = check(a.arg_id, dom_ty, ec, es); !r) return std::unexpected(r.error());\n            // A codomain without loose bound variables", "            (void)dom_ty;\n            // A codomain without loose bound variables"),
 'M6 admission universe rule disabled': ('if (!parameter_like && field_level[j] > *fam_level && !(ctx().prop_mode && *fam_level == 0)) {', 'if (!parameter_like && field_level[j] > *fam_level + 1000 && !(ctx().prop_mode && *fam_level == 0)) {'),
 'M7 strict positivity always true': ("                if (!is_strictly_positive(type_global_id, arg_ty))\n                    throw std::logic_error(\"itp: constructor '\" + ctor.name + \"' fails strict positivity", "                if (false)\n                    throw std::logic_error(\"itp: constructor '\" + ctor.name + \"' fails strict positivity"),
 'M8 Match branch count not checked': ("    if (ctor_gids.size() != branches.size())\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive, std::nullopt,\n            \"Match branch count", "    if (false)\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive, std::nullopt,\n            \"Match branch count"),

 'M101 proof irrelevance is never tried': ("    return proof_irrelevance_fallback(a, b, ec, es, std::move(r), meta_mark, pending_mark);", "    (void)meta_mark; (void)pending_mark; return r;"),
 'M102 proof irrelevance also outside the Prop/Type world': ("    if (!ctx().prop_mode) return unify_structural(a, b, ec, es);\n    const auto meta_mark", "    const auto meta_mark"),
 'M103 irrelevance does not require the type to be a proposition (every type is irrelevant)': ("        if (!s || *s != 0 || !scratch.pending.empty()) return std::nullopt;", "        if (!s || !scratch.pending.empty()) return std::nullopt;"),
 'M104 irrelevance does not compare the two types': ("    if (ElabResult r = unify_values(eval(*ta, ec.values), eval(*tb, ec.values), ec, es); !r) { rollback(); return failure; }", "    (void)tb;"),
 'M105 irrelevance: a type in Type1 counts as a proposition': ("        if (!s || *s != 0 || !scratch.pending.empty()) return std::nullopt;", "        if (!s || *s > 1 || !scratch.pending.empty()) return std::nullopt;"),
 'M106 prefilter: constructor values are never proof candidates': ("            return r.tag == NodeType::TYPE && ctx().types[r.index].universe_level == 0;\n        },\n        [&](const VNeutral& n) {", "            return false && r.tag == NodeType::TYPE && ctx().types[r.index].universe_level == 0;\n        },\n        [&](const VNeutral& n) {"),
 'M107 prefilter: neutrals headed by a global are never proof candidates': ("                case NeutralHead::Kind::GlobalConst: return type_may_be_proposition(ctx().globals[n.head.id].type);", "                case NeutralHead::Kind::GlobalConst: return false;"),
 'M108 prefilter: neutrals headed by a local variable are never proof candidates': ("                case NeutralHead::Kind::LocalVar: return n.head.id < ec.size() && type_may_be_proposition(ec.types[n.head.id]);", "                case NeutralHead::Kind::LocalVar: return false;"),
 'M109 prefilter: a stuck match is never a proof candidate': ("if (!std::holds_alternative<ApplyFrame>(fr)) return true;     // a stuck match: its type is the motive's business, let infer decide", "if (!std::holds_alternative<ApplyFrame>(fr)) return false;"),
 'M110 irrelevance keeps the metavariables the failed structural attempt solved': ("        ctx().meta_ctx.rewind(meta_mark);\n        es.pending.erase(", "        es.pending.erase("),
 'M111 irrelevance keeps the constraints the failed structural attempt deferred': ("        es.pending.erase(es.pending.begin() + static_cast<std::ptrdiff_t>(pending_mark), es.pending.end());\n    };", "    };"),
 'M112 irrelevance nests at most once': ("inline constexpr int kMaxPiNesting = 8;", "inline constexpr int kMaxPiNesting = 1;"),
 'M113 irrelevance has no nesting bound': ("    if (pi_nesting() >= kMaxPiNesting) return failure;\n", ""),
 'C1 checker: proof irrelevance is never tried': ("        return self.prop_elim and ctx is not None and self.irrelevant(a, b, depth, ctx)", "        return False", 'indep_checker.py'),
 'C2 checker: proof irrelevance also without prop_elim': ("        return self.prop_elim and ctx is not None and self.irrelevant(a, b, depth, ctx)", "        return ctx is not None and self.irrelevant(a, b, depth, ctx)", 'indep_checker.py'),
 'C3 checker: irrelevance does not require the type to be a proposition': ("            if self.sort_of(ctx, pa) != 0: return False\n", "", 'indep_checker.py'),
 'C4 checker: irrelevance does not compare the two types': ("            return self.conv(pa, pb, depth + 1, ctx)\n        except (CheckError, Unsupported, KeyError, IndexError): return False", "            return True\n        except (CheckError, Unsupported, KeyError, IndexError): return False", 'indep_checker.py'),
 'C5 checker: the type checker converts without a context (no irrelevance while checking)': ("        if not self.conv(inferred, expected, 0, ctx): raise CheckError('type mismatch')", "        if not self.conv(inferred, expected): raise CheckError('type mismatch')", 'indep_checker.py'),
 'C6 checker: binders do not extend the context of the comparison': ("        ext = lambda z, A: None if ctx is None else {**ctx, z[1]: A}", "        ext = lambda z, A: ctx", 'indep_checker.py'),
 'C7 checker: a type in Type1 counts as a proposition': ("            if self.sort_of(ctx, pa) != 0: return False\n", "            if self.sort_of(ctx, pa) > 1: return False\n", 'indep_checker.py'),
 'C8 checker: eta is not applied (a lambda against a non-lambda)': ("        if (ka == 'lam') != (kb == 'lam'):                  # eta:", "        if False and (ka == 'lam') != (kb == 'lam'):                  # eta:", 'indep_checker.py'),
 'B1 declare_checked accepts metavariables (an unsolved ?m : False proves False)': ('    if (detail::contains_meta(type) || detail::contains_meta(value))\n', '    if (false)\n'),
 'B2 contains_meta never finds a metavariable': ('    if (t.tag == NodeType::META) return true;\n    if (!seen.insert(', '    if (t.tag == NodeType::META) return false;\n    if (!seen.insert('),
 'B3 contains_meta does not look inside binders and applications': ('    return any_child(t, [&](NodeId c) { return contains_meta_rec(c, seen); });', '    return false;'),
 'B4 a finished inductive type can be finished again (more constructors appended)': ('    if (defs[tid].reserved != GlobalDef::Reserved::Inductive)\n', '    if (false)\n'),
 'B5 a recursive function can be finished twice, or an ordinary global finished as one': ('    if (defs[fn_gid].reserved != GlobalDef::Reserved::Function)\n', '    if (false)\n'),
 "B6 the decreasing argument need not be one of the body's parameters": ('    if (dec >= lambda_arity(body)) return', '    if (false) return'),
 'B7 recursion over a global that is not a finished inductive type': ('    if (ind_gid >= defs.size() || !defs[ind_gid].is_inductive) return', '    if (ind_gid >= defs.size()) return'),
 'B8 a function body may contain metavariables (checked finish)': ('    if (contains_meta(body)) return "a function body may not contain metavariables";\n', ''),
 'B9 universe level has a successor even at the maximum (wraparound)': ('            if (ctx().types[t.index].universe_level >= kMaxUniverseLevel) return err(', '            if (false) return err('),
 'B10 finishing an inductive type does not clear its reservation': ('    c.globals.defs[type_global_id].reserved = GlobalDef::Reserved::None;\n    return handle;', '    return handle;'),
 'B11 finishing a checked recursive function does not clear its reservation': ('    ctx().globals.defs[fn_gid].reserved = GlobalDef::Reserved::None;\n    return {};', '    return {};'),
 'B12 finishing an unchecked recursive function does not clear its reservation': ('    ctx().globals.defs[fn_gid].reserved = GlobalDef::Reserved::None;\n    ctx().globals.install_body(fn_gid, body);', '    ctx().globals.install_body(fn_gid, body);'),
 'B13 finish_inductive (checked) looks the id up before validating it': ('    detail::require_reserved_inductive(type_global_id);        // before the admission checks look anything up by this id\n', ''),
 'B14 finish_inductive_unchecked does not validate the id': ('    detail::require_reserved_inductive(type_global_id);\n    detail::require_ctor_specs_complete', '    detail::require_ctor_specs_complete'),
 'B15 checked recursive function: no validation of ids / reservation / decreasing argument': ('    if (auto problem = detail::recursive_function_problem(fn_gid, ind_type_global_id, decreasing_arg_level, body))\n        return std::unexpected(TypeError{TypeErrorKind::Other, body, std::nullopt, *problem});\n', ''),
 'B16 unchecked recursive function: no validation of ids / reservation / decreasing argument': ('    if (auto problem = detail::recursive_function_problem(fn_gid, ind_type_global_id, decreasing_arg_level, body)) throw std::logic_error("itp: finish_recursive_function: " + *problem);\n', ''),
 'B17 the evaluator aborts... no: an out-of-scope level reads the wrong cell instead of throwing': ('    if (!top.p || target > top.p->depth) throw std::logic_error("itp: variable out of scope (level "', '    if (!top.p) throw std::logic_error("itp: variable out of scope (level "'),
 'B20 check_open allows metavariables (a hole passes for a proof)': ('        if (!allow_holes) {\n            for (auto h : hyp_types)', '        if (false) {\n            for (auto h : hyp_types)', 'itp_bindings.cpp'),
 'B21 require_global is off by one (the first id past the end is accepted)': ('    if (gid >= itp::ctx().globals.defs.size()) throw ITPError(', '    if (gid > itp::ctx().globals.defs.size()) throw ITPError(', 'itp_bindings.cpp'),
 'B22 require_meta is off by one': ('    if (id >= itp::ctx().meta_ctx.entries.size()) throw ITPError(', '    if (id > itp::ctx().meta_ctx.entries.size()) throw ITPError(', 'itp_bindings.cpp'),
 'B23 require_small accepts everything (wraparound / absurd sizes)': ('    if (v > itp::kMaxUniverseLevel) throw ITPError(', '    if (false) throw ITPError(', 'itp_bindings.cpp'),
 'B24 lift does not check that the shifted indices stay in range': ('        if (static_cast<uint64_t>(itp::loose_range(t)) + d > itp::kMaxUniverseLevel) throw ITPError("lift: the result would have a bound variable index beyond the supported range");\n', '', 'itp_bindings.cpp'),
 'B25 close_telescope accepts an unknown kind': ('        if (kind != "lam" && kind != "pi") throw ITPError(', '        if (false) throw ITPError(', 'itp_bindings.cpp'),
 'B26 start_proof accepts a goal with a metavariable': ('require_scoped(goal_type, 0, "goal type"); require_meta_free(goal_type, "goal type");', 'require_scoped(goal_type, 0, "goal type");', 'itp_bindings.cpp'),
 'B27 declare_unchecked accepts a metavariable in the type': ('require_scoped(type, 0, "declared type"); require_meta_free(type, "declared type"); if (value) {', 'require_scoped(type, 0, "declared type"); if (value) {', 'itp_bindings.cpp'),
 'B28 declare_unchecked accepts a metavariable in the value': ('require_scoped(*value, 0, "declared value"); require_meta_free(*value, "declared value"); }', 'require_scoped(*value, 0, "declared value"); }', 'itp_bindings.cpp'),
 'B29 reserve_inductive_type accepts a metavariable in the sort': ('require_scoped(sort, 0, "inductive sort"); require_meta_free(sort, "inductive sort");', 'require_scoped(sort, 0, "inductive sort");', 'itp_bindings.cpp'),
 'B30 reserve_recursive_function accepts a metavariable in the type': ('require_scoped(type, 0, "function type"); require_meta_free(type, "function type");', 'require_scoped(type, 0, "function type");', 'itp_bindings.cpp'),
 'B31 finish_inductive accepts metavariables in field types and result indices': ('for (auto a : c.arg_types) require_meta_free(a, "constructor field type"); for (auto r : c.result_indices) { require_scoped(r, static_cast<uint32_t>(c.arg_types.size()), "constructor result index"); require_meta_free(r, "constructor result index"); } }\n        return itp::finish_inductive(tid, std::move(ctors)); }', 'for (auto r : c.result_indices) { require_scoped(r, static_cast<uint32_t>(c.arg_types.size()), "constructor result index"); } }\n        return itp::finish_inductive(tid, std::move(ctors)); }', 'itp_bindings.cpp'),
 'B32 finish_inductive_unchecked accepts metavariables in field types and result indices': ('for (auto a : c.arg_types) require_meta_free(a, "constructor field type"); for (auto r : c.result_indices) { require_scoped(r, static_cast<uint32_t>(c.arg_types.size()), "constructor result index"); require_meta_free(r, "constructor result index"); } }\n        return itp::finish_inductive_unchecked(tid, std::move(ctors)); }', 'for (auto r : c.result_indices) { require_scoped(r, static_cast<uint32_t>(c.arg_types.size()), "constructor result index"); } }\n        return itp::finish_inductive_unchecked(tid, std::move(ctors)); }', 'itp_bindings.cpp'),
 'B33 fresh_meta does not check that the type is in scope': ('require_small(scope, "meta scope"); require_scoped(type, scope, "meta type");', 'require_small(scope, "meta scope");', 'itp_bindings.cpp'),
 'B34 mk_const accepts an unknown global': ('require_global(gid, "mk_const"); return itp::mk_const(gid);', 'return itp::mk_const(gid);', 'itp_bindings.cpp'),
 'B35 mk_meta accepts an unknown metavariable': ('require_meta(id, "mk_meta"); return itp::mk_meta(id);', 'return itp::mk_meta(id);', 'itp_bindings.cpp'),
 'B36 rewrite accepts unknown lemma ids': ('require_global(transport_gid, "rewrite transport lemma"); require_global(symm_gid, "rewrite symmetry lemma"); ', '', 'itp_bindings.cpp'),
 'B37 induct accepts an unknown equality family id': ('if (eq_gid) require_global(*eq_gid, "induct eq_gid"); ', '', 'itp_bindings.cpp'),
 'B38 induct accepts an unknown reflexivity lemma id': ('if (refl_gid) require_global(*refl_gid, "induct refl_gid");', '', 'itp_bindings.cpp'),
 'B39 declare_quot accepts an unknown family id': ('require_global(eq_gid, "declare_quot"); ', '', 'itp_bindings.cpp'),
 'B40 infer_open does not check the hypothesis types are in scope': ('auto g = activate(); require_scoped_telescope(hyp_types, "hypothesis type"); require_scoped(t, static_cast<uint32_t>(hyp_types.size()), "term"); itp::ElabCtx ec; itp::ElabState es;', 'auto g = activate(); require_scoped(t, static_cast<uint32_t>(hyp_types.size()), "term"); itp::ElabCtx ec; itp::ElabState es;', 'itp_bindings.cpp'),
 'B41 normalize_open accepts an absurd context size': ('auto g = activate(); require_small(n0, "context size"); require_scoped(t, n0, "term"); return itp::normalize(t, fresh_env(n0));', 'auto g = activate(); require_scoped(t, n0, "term"); return itp::normalize(t, fresh_env(n0));', 'itp_bindings.cpp'),
 'P1 infer(App): the argument is never evaluated (dependent codomains lose it)': ('Value arg_val = loose_range(pi->cod_id) == 0 ? fresh_neutral(pi->env.size()) : eval(a.arg_id, ec.values);', 'Value arg_val = fresh_neutral(pi->env.size());'),
 'P2 motive_result: a dependent constant-looking motive result is returned as is': ('    return loose_range(cur) == 0 ? cur : mk_app(apply_all(motive, indices), scrutinee);', '    return cur;'),
 'P3 motive_result: a motive that is not a lambda telescope is taken for its own result': ('        if (cur.tag != NodeType::LAMBDA) return mk_app(apply_all(motive, indices), scrutinee);', '        if (cur.tag != NodeType::LAMBDA) return cur;'),
 'P4 a bad checked function stays callable: its cached value is not dropped on rollback': ('void uninstall_body(uint32_t gid) { defs.at(gid).value = std::nullopt; --body_installs; invalidate_value_caches(); }', 'void uninstall_body(uint32_t gid) { defs.at(gid).value = std::nullopt; --body_installs; }'),
 'P5 installing a function body does not drop cached values': ('void install_body(uint32_t gid, NodeId body) { defs.at(gid).value = body; ++body_installs; invalidate_value_caches(); }', 'void install_body(uint32_t gid, NodeId body) { defs.at(gid).value = body; ++body_installs; }'),
 'P7 rewind erases globals but keeps their cached values': ('            globals.defs.erase(globals.defs.begin() + keep, globals.defs.end());\n            globals.invalidate_value_caches();', '            globals.defs.erase(globals.defs.begin() + keep, globals.defs.end());'),
 'P8 Env lookup jumps one level too far': ('        c = (j && j->depth >= target) ? j : c->parent.p;', '        c = (j && j->depth + 1 >= target) ? j : c->parent.p;'),
 'P9 constructor fast path also fires for constructors with more than one field': ('if (ci.arity == 1 && n0->spine.empty()) return Value{VCtor{', 'if (ci.arity >= 1 && n0->spine.empty()) return Value{VCtor{'),
 'P10 Match binds only the first constructor field': ('while (i < nargs && branch.tag == NodeType::LAMBDA) {', 'if (i < nargs && branch.tag == NodeType::LAMBDA) {'),
 'P11 the release trampoline never queues (a deep chain is freed by recursion)': ('inline constexpr int kMaxReleaseDepth = 48;', 'inline constexpr int kMaxReleaseDepth = 1 << 30;'),
 'P12 appending to a spine drops the last old frame': ('for (uint32_t i = 0; i < n; ++i) { new (items(s.b_) + i) T(items(b_)[i]); ++s.b_->n; }', 'for (uint32_t i = 0; i + 1 < n; ++i) { new (items(s.b_) + i) T(items(b_)[i]); ++s.b_->n; }'),
 'G1 stack check removed from apply': ('inline Value apply(Value fn, Value arg) {\n    ITP_STACK_CHECK();\n', 'inline Value apply(Value fn, Value arg) {\n'),
 'G2 stack check removed from eval': ('inline Value eval(NodeId term, const Env& env) {\n    ITP_STACK_CHECK();\n', 'inline Value eval(NodeId term, const Env& env) {\n'),
 'G3 stack check removed from map_children': ('inline NodeId map_children(NodeId t, F&& f) {\n    ITP_STACK_CHECK();\n', 'inline NodeId map_children(NodeId t, F&& f) {\n'),
 'G4 stack check removed from loose_range': ('inline uint32_t loose_range(NodeId t) {\n    ITP_STACK_CHECK();\n', 'inline uint32_t loose_range(NodeId t) {\n'),
 'G5 stack check removed from lift': ('inline NodeId lift(NodeId t, uint32_t d, uint32_t cutoff = 0) {\n    ITP_STACK_CHECK();\n', 'inline NodeId lift(NodeId t, uint32_t d, uint32_t cutoff = 0) {\n'),
 'G6 stack check removed from instantiate_at': ('inline NodeId instantiate_at(NodeId t, const std::vector<NodeId>& vals, uint32_t offset) {\n    ITP_STACK_CHECK();\n', 'inline NodeId instantiate_at(NodeId t, const std::vector<NodeId>& vals, uint32_t offset) {\n'),
 'G7 stack check removed from abstract_fvars': ('inline NodeId abstract_fvars(NodeId t, const std::vector<uint32_t>& fvars, uint32_t depth = 0) {\n    ITP_STACK_CHECK();\n', 'inline NodeId abstract_fvars(NodeId t, const std::vector<uint32_t>& fvars, uint32_t depth = 0) {\n'),
 'G8 stack check removed from replace_fvars': ('inline NodeId replace_fvars(NodeId t, const std::unordered_map<uint32_t, NodeId>& m, uint32_t depth = 0) {\n    ITP_STACK_CHECK();\n', 'inline NodeId replace_fvars(NodeId t, const std::unordered_map<uint32_t, NodeId>& m, uint32_t depth = 0) {\n'),
 'G9 stack check removed from resolve_metas': ('inline NodeId resolve_metas(NodeId term) {\n    ITP_STACK_CHECK();\n', 'inline NodeId resolve_metas(NodeId term) {\n'),
 'G10 stack check removed from requote': ('inline NodeId requote(NodeId term, const Env& env, uint32_t n0, uint32_t depth, uint32_t bound = 0) {\n    ITP_STACK_CHECK();\n', 'inline NodeId requote(NodeId term, const Env& env, uint32_t n0, uint32_t depth, uint32_t bound = 0) {\n'),
 'G11 stack check removed from quote_at': ('inline NodeId quote_at(const Value& v, uint32_t n0, uint32_t depth) {\n    ITP_STACK_CHECK();\n', 'inline NodeId quote_at(const Value& v, uint32_t n0, uint32_t depth) {\n'),
 'G12 stack check removed from any_child': ('inline bool any_child(NodeId t, P&& pred) {\n    ITP_STACK_CHECK();\n', 'inline bool any_child(NodeId t, P&& pred) {\n'),
 'G13 stack check removed from references_level_at_or_above': ('inline bool references_level_at_or_above(NodeId t, uint32_t level) {\n    ITP_STACK_CHECK();\n', 'inline bool references_level_at_or_above(NodeId t, uint32_t level) {\n'),
 'G14 stack check removed from branches_convertible': ('inline bool branches_convertible(NodeId ba, const Env& env_a, NodeId bb, const Env& env_b, uint32_t depth) {\n    ITP_STACK_CHECK();\n', 'inline bool branches_convertible(NodeId ba, const Env& env_a, NodeId bb, const Env& env_b, uint32_t depth) {\n'),
 'G15 stack check removed from conv_value': ('inline bool conv_value(const Value& a, const Value& b, uint32_t depth) {\n    ITP_STACK_CHECK();\n', 'inline bool conv_value(const Value& a, const Value& b, uint32_t depth) {\n'),
 'G16 stack check removed from replace_terms': ('inline NodeId replace_terms(NodeId t, const std::vector<std::pair<NodeId, NodeId>>& subs) {\n    ITP_STACK_CHECK();\n', 'inline NodeId replace_terms(NodeId t, const std::vector<std::pair<NodeId, NodeId>>& subs) {\n'),
 'G17 stack check removed from collect_fvars': ('inline void collect_fvars(NodeId t, std::unordered_set<uint32_t>& out) {\n    ITP_STACK_CHECK();\n', 'inline void collect_fvars(NodeId t, std::unordered_set<uint32_t>& out) {\n'),
 'G18 stack check removed from occurs_meta': ('inline bool occurs_meta(uint32_t meta_id, NodeId t) {\n    ITP_STACK_CHECK();\n', 'inline bool occurs_meta(uint32_t meta_id, NodeId t) {\n'),
 'G19 stack check removed from contains_meta_rec': ('inline bool contains_meta_rec(NodeId t, std::unordered_set<uint64_t>& seen) {\n    ITP_STACK_CHECK();\n', 'inline bool contains_meta_rec(NodeId t, std::unordered_set<uint64_t>& seen) {\n'),
 'G20 stack check removed from unify_structural': ('inline ElabResult unify_structural(const Value& a_in, const Value& b_in, ElabCtx& ec, ElabState& es) {\n    ITP_STACK_CHECK();\n', 'inline ElabResult unify_structural(const Value& a_in, const Value& b_in, ElabCtx& ec, ElabState& es) {\n'),
 'G21 stack check removed from unify_values': ('inline ElabResult unify_values(const Value& a, const Value& b, ElabCtx& ec, ElabState& es) {\n    ITP_STACK_CHECK();\n', 'inline ElabResult unify_values(const Value& a, const Value& b, ElabCtx& ec, ElabState& es) {\n'),
 'G22 stack check removed from unify': ('inline ElabResult unify(NodeId a_id, NodeId b_id, ElabCtx& ec, ElabState& es) {\n    ITP_STACK_CHECK();\n', 'inline ElabResult unify(NodeId a_id, NodeId b_id, ElabCtx& ec, ElabState& es) {\n'),
 'G23 stack check removed from match_index_pattern': ('                                 const Env& env, CtorIndexForcing& out) {\n    ITP_STACK_CHECK();\n', '                                 const Env& env, CtorIndexForcing& out) {\n'),
 'G24 stack check removed from occurs': ('inline bool occurs(uint32_t ind_type_global_id, NodeId term) {\n    ITP_STACK_CHECK();\n', 'inline bool occurs(uint32_t ind_type_global_id, NodeId term) {\n'),
 'G25 stack check removed from is_strictly_positive': ('inline bool is_strictly_positive(uint32_t ind_type_global_id, NodeId arg_type) {\n    ITP_STACK_CHECK();\n', 'inline bool is_strictly_positive(uint32_t ind_type_global_id, NodeId arg_type) {\n'),
 'G26 stack check removed from is_structurally_decreasing / walk (lambda)': ('                                        uint32_t decreasing_arg_level, NodeId body) {\n    std::unordered_set<uint32_t> smaller;\n\n    std::function<bool(NodeId, uint32_t)> walk = [&](NodeId t, uint32_t depth) -> bool {\n        ITP_STACK_CHECK();\n', '                                        uint32_t decreasing_arg_level, NodeId body) {\n    std::unordered_set<uint32_t> smaller;\n\n    std::function<bool(NodeId, uint32_t)> walk = [&](NodeId t, uint32_t depth) -> bool {\n'),
 'G27 stack check removed from is_locally_structurally_decreasing / walk (lambda)': ('                                                uint32_t decreasing_arg, NodeId body) {\n    std::unordered_set<uint32_t> smaller;\n\n    std::function<bool(NodeId, uint32_t)> walk = [&](NodeId t, uint32_t depth) -> bool {\n        ITP_STACK_CHECK();\n', '                                                uint32_t decreasing_arg, NodeId body) {\n    std::unordered_set<uint32_t> smaller;\n\n    std::function<bool(NodeId, uint32_t)> walk = [&](NodeId t, uint32_t depth) -> bool {\n'),
 'G28 stack check removed from to_levels': ('inline NodeId to_levels(NodeId t, uint32_t depth) {\n    ITP_STACK_CHECK();\n', 'inline NodeId to_levels(NodeId t, uint32_t depth) {\n'),
 'G29 stack check removed from uniform_family_occurrences': ('inline bool uniform_family_occurrences(NodeId t, uint32_t fam, uint32_t arity, uint32_t pos, NodeId var) {\n    ITP_STACK_CHECK();\n', 'inline bool uniform_family_occurrences(NodeId t, uint32_t fam, uint32_t arity, uint32_t pos, NodeId var) {\n'),
 'G30 stack check removed from check_match_branches': ('                                        const std::vector<NodeId>& concrete_indices, ElabCtx& ec, ElabState& es) {\n    ITP_STACK_CHECK();\n', '                                        const std::vector<NodeId>& concrete_indices, ElabCtx& ec, ElabState& es) {\n'),
 'G31 stack check removed from infer': ('inline TypeResult infer(NodeId t, ElabCtx& ec, ElabState& es) {\n    ITP_STACK_CHECK();\n', 'inline TypeResult infer(NodeId t, ElabCtx& ec, ElabState& es) {\n'),
 'G32 stack check removed from check_lambda': ('inline ElabResult check_lambda(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {\n    ITP_STACK_CHECK();\n', 'inline ElabResult check_lambda(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {\n'),
 'G33 stack check removed from check_match': ('inline ElabResult check_match(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {\n    ITP_STACK_CHECK();\n', 'inline ElabResult check_match(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {\n'),
 'G34 stack check removed from check': ('inline ElabResult check(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {\n    ITP_STACK_CHECK();\n', 'inline ElabResult check(NodeId t, NodeId expected_ty, ElabCtx& ec, ElabState& es) {\n'),
 'G35 stack check removed from fvars_below': ('inline bool fvars_below(NodeId t, uint32_t nfree, std::unordered_set<uint64_t>& seen) {\n    ITP_STACK_CHECK();\n', 'inline bool fvars_below(NodeId t, uint32_t nfree, std::unordered_set<uint64_t>& seen) {\n'),
 'G36 stack check removed from dump_raw (bindings)': ('        std::function<void(itp::NodeId)> go = [&](itp::NodeId t) {\n            ITP_STACK_CHECK();\n', '        std::function<void(itp::NodeId)> go = [&](itp::NodeId t) {\n', 'itp_bindings.cpp'),
 'R1 declare does not check a TYPE for unfinished globals': ('        require_complete(type);\n        if (value) require_complete(*value);\n', '        if (value) require_complete(*value);\n'),
 'R2 declare does not check a VALUE for unfinished globals': ('        require_complete(type);\n        if (value) require_complete(*value);\n', '        require_complete(type);\n'),
 'R3 a constructor type may not mention its own type (declare_ctor allows no unfinished global)': ('uint32_t gid = declare(type, std::nullopt, std::move(name), ind_type_global_id);', 'uint32_t gid = declare(type, std::nullopt, std::move(name));'),
 'R4 the allowed global is not exempt from the unfinished-global scan': ('if (g != allowed && g < env.defs.size() && env.defs[g].reserved != GlobalDef::Reserved::None) return g;', 'if (g < env.defs.size() && env.defs[g].reserved != GlobalDef::Reserved::None) return g;'),
 'R5 the unfinished-global scan does not bound the global id': ('if (g != allowed && g < env.defs.size() && env.defs[g].reserved != GlobalDef::Reserved::None) return g;', 'if (g != allowed && env.defs[g].reserved != GlobalDef::Reserved::None) return g;'),
 'R6 the scan ignores reserved functions': ('env.defs[g].reserved != GlobalDef::Reserved::None) return g;', 'env.defs[g].reserved == GlobalDef::Reserved::Inductive) return g;'),
 'R7 the scan ignores reserved inductive types': ('env.defs[g].reserved != GlobalDef::Reserved::None) return g;', 'env.defs[g].reserved == GlobalDef::Reserved::Function) return g;'),
 'R8 the unfinished-global scan revisits shared nodes (exponential on a DAG)': ('    if (!seen.insert((static_cast<uint64_t>(static_cast<uint32_t>(t.tag)) << 32) | t.index).second) return std::nullopt;\n    std::optional<uint32_t> found;', '    std::optional<uint32_t> found;'),
 'R9 a function body may mention other reserved globals': ('    if (auto bad = unfinished_global_in(ctx().globals, body, fn_gid))\n        return', '    if (auto bad = unfinished_global_in(ctx().globals, body, fn_gid); bad && false)\n        return'),
 'R10 a function body may not mention the function itself': ('    if (auto bad = unfinished_global_in(ctx().globals, body, fn_gid))\n        return', '    if (auto bad = unfinished_global_in(ctx().globals, body, kNoGlobal))\n        return'),
 'R11 constructor field types may mention other reserved globals': ('        for (NodeId a : c.arg_types) require(a, c, "field type");\n', ''),
 'R12 constructor result indices may mention other reserved globals': ('        for (NodeId r : c.result_indices) require(r, c, "result index");\n', ''),
 'R13 finish_inductive_unchecked does not check the constructor specs': ('    detail::require_ctor_specs_complete(type_global_id, ctors);        // all of them, before the first constructor is declared: a half-finished type would be worse than a refused one\n', ''),
 'R14 declare_checked ignores unfinished globals in the type': ('    for (NodeId part : {type, value})\n        if (auto bad = detail::unfinished_global_in(ctx().globals, part, kNoGlobal))', '    for (NodeId part : {value})\n        if (auto bad = detail::unfinished_global_in(ctx().globals, part, kNoGlobal))'),
 'R15 declare_checked ignores unfinished globals in the value': ('    for (NodeId part : {type, value})\n        if (auto bad = detail::unfinished_global_in(ctx().globals, part, kNoGlobal))', '    for (NodeId part : {type})\n        if (auto bad = detail::unfinished_global_in(ctx().globals, part, kNoGlobal))'),
 'R16 installing a function body is not counted (rewind may cross it)': ('void install_body(uint32_t gid, NodeId body) { defs.at(gid).value = body; ++body_installs; invalidate_value_caches(); }', 'void install_body(uint32_t gid, NodeId body) { defs.at(gid).value = body; invalidate_value_caches(); }'),
 'R17 rolling back a function body is not counted (rewind refused after a refused finish)': ('void uninstall_body(uint32_t gid) { defs.at(gid).value = std::nullopt; --body_installs; invalidate_value_caches(); }', 'void uninstall_body(uint32_t gid) { defs.at(gid).value = std::nullopt; invalidate_value_caches(); }'),
 'R18 a checkpoint forgets how many bodies were installed before it': ('meta_ctx.checkpoint(), globals.defs.size(), globals.body_installs\n', 'meta_ctx.checkpoint(), globals.defs.size(), 0\n'),
 'R19 rewind does not compare the number of installed bodies': ('        if (globals.body_installs != cp.body_installs_n)\n', '        if (false)\n'),
 'R20 check_open accepts a term that mentions a reserved global': ('            require_meta_free(t, "term"); require_complete_globals(t, "term");\n', '            require_meta_free(t, "term");\n', 'itp_bindings.cpp'),
 'R21 check_open accepts an expected type that mentions a reserved global': ('            require_meta_free(expected, "expected type"); require_complete_globals(expected, "expected type");\n', '            require_meta_free(expected, "expected type");\n', 'itp_bindings.cpp'),
 'R22 check_open accepts a hypothesis type that mentions a reserved global': ('{ require_meta_free(h, "hypothesis type"); require_complete_globals(h, "hypothesis type"); }', '{ require_meta_free(h, "hypothesis type"); }', 'itp_bindings.cpp'),
 'R23 recursive_info reports the wrong decreasing argument': ('return std::make_pair(ri->ind_type_global_id, ri->decreasing_arg_level);', 'return std::make_pair(ri->ind_type_global_id, ri->decreasing_arg_level + 1);', 'itp_bindings.cpp'),
 'R24 reserved_kind always says finished': ('return static_cast<int>(itp::ctx().globals.defs[gid].reserved); }', 'return 0; }', 'itp_bindings.cpp'),
 'R25 independent checker: no cycle detection': ('if len(comp) > 1: sccs.append(sorted(comp))', 'if len(comp) > 99: sccs.append(sorted(comp))', 'indep_checker.py'),
 'R26 independent checker: Tarjan lowlink not updated through the stack': ('elif w in on: low[v] = min(low[v], index[w])', 'elif w in on: pass', 'indep_checker.py'),
 'R27 independent checker: mentions of reserved globals are fine': ('elif self.g[h].reserved and h != gid:', 'elif False:', 'indep_checker.py'),
 'R28 independent checker: unrecorded self-reference is fine': ('if gid in d and self.g[gid].recursion is None:', 'if False:', 'indep_checker.py'),
 'R29 independent checker: recursive globals are not guard-checked': ("if not self.guard_ok(d, gl.type, sx, replace_const(gl.value, gid, ('v', sx))):", 'if False:', 'indep_checker.py'),
 'R30 independent checker: recursion over a non-family is fine': ('if fam not in self.g or not self.is_family(fam):', 'if False:', 'indep_checker.py'),
 'R31 independent checker: match branches are not searched for mentions': ("elif k == 'match': stack.append(u[1]); stack.append(u[2]); stack.extend(u[3])", "elif k == 'match': stack.append(u[1]); stack.append(u[2])", 'indep_checker.py'),
 'R32 independent checker: the type of a global is not searched for mentions': ('            d = consts_in(gl.type)\n            if gl.value is not None: d |= consts_in(gl.value)\n            deps[gid] = d', '            d = set()\n            if gl.value is not None: d |= consts_in(gl.value)\n            deps[gid] = d', 'indep_checker.py'),
 'R33 independent checker: a mention of an unknown global is fine': ("if h not in self.g: bad.append((gid, self.g[gid].name, 'mentions the unknown global %d' % h))\n                elif", 'if h not in self.g: pass\n                elif', 'indep_checker.py'),
 'R34 independent checker: the decreasing argument may be of another family': ("if dh != ('c', fam): bad.append(", 'if False: bad.append(', 'indep_checker.py'),
 'R35 prop_elim can be changed after globals exist': ('        if (v != prop_mode && !globals.defs.empty())\n            throw std::logic_error("itp: the option prop_elim cannot', '        if (false)\n            throw std::logic_error("itp: the option prop_elim cannot'),
 'R36 prop_elim cannot even be set to the value it already has': ('        if (v != prop_mode && !globals.defs.empty())', '        if (!globals.defs.empty())'),
 'R37 the prop_elim option is set without the guard': ('else if (name == "prop_elim") itp::ctx().set_prop_mode(value); else throw', 'else if (name == "prop_elim") itp::ctx().prop_mode = value; else throw', 'itp_bindings.cpp'),
 'R38 declare_checked does not check that the declared type is a type': ('    if (auto sort = sort_level_of(type, empty_ec, es, "the declared type is not a type"); !sort) return std::unexpected(sort.error());\n', ''),
 'R39 finishing a recursive function does not check its declared type': ('        if (auto sort = sort_level_of(ctx().globals[fn_gid].type, ec0, es0, "the function\'s declared type is not a type"); !sort) return std::unexpected(sort.error());\n', ''),
 'R40 check_open does not check that the statement is a type': ('        if (!allow_holes) { auto s = itp::sort_level_of(expected, ec, es, "the statement is not a type"); if (!s) return {false, "ill-formed statement: " + s.error().message}; }\n', '', 'itp_bindings.cpp'),
 'R41 check_open does not check that the hypotheses are types': ('            if (!allow_holes) { auto s = itp::sort_level_of(h, ec, es, "a hypothesis type is not a type"); if (!s) return {false, "ill-formed context: " + s.error().message}; }\n', '', 'itp_bindings.cpp'),
}
# Order matters only for speed (a mutant stops at the first gate that fails): fast, high-yield gates first, the 25-second oracle last.
GATE_CMDS = (['s2_tests.py', '1500'], ['s16_quot_tests.py'], ['s17_irrelevance_tests.py'], ['s18_deep_tests.py'], ['s19_boundary_tests.py'], ['s20_reservation_tests.py'], ['harness.py', '1500', '3'], ['s3_tests.py', '600'], ['s4_tests.py', '800'], ['s5_tests.py', '1200', '3'], ['s5_probe_tests.py', '500', '7'], ['s6_logic_tests.py'], ['s6_f6_tests.py'],
             ['s7_param_tests.py', '150', '600'], ['s8_indep_tests.py', '100'], ['s9_strong_tests.py'], ['s10_noconf_tests.py'], ['s11_eqgen_tests.py'], ['s12_probe_eq_tests.py', '150'], ['s13_prop_tests.py'],
             ['s14_prop_world_tests.py', '600'], ['s15_classical_tests.py', '300'], ['oracle_gate.py'])
# a gate has failed when it exits non-zero or prints one of these markers
FAIL_RE = re.compile(r"GATE: FAIL|Traceback")        # NOT words like ACCEPTED/FAILED: healthy gate output contains them ("ACCEPTED 0"); the exit status is the real signal
# mutants that no gate can distinguish from the original BY CONSTRUCTION (equivalent mutants); each needs a reason
KNOWN_EQUIVALENT = {
    'M113 irrelevance has no nesting bound': "the bound is a resource guard against pathological input (inference calls the unifier calls inference ..); every well-typed comparison terminates without reaching it, so no test input can distinguish the two",
    'P4 a bad checked function stays callable: its cached value is not dropped on rollback': "eval consults GlobalDef::cached_value only while the global HAS a value (`if (def.value && !no_delta_flag())`); after a rollback it has none, so a stale cache is unreachable, and install_body (mutant P5) drops every cache before a value can appear again. The invalidation in uninstall_body only frees the cached Value early. s20 (C4) pins the observable behaviour: a refused function does not unfold",
    'P5 installing a function body does not drop cached values': "since the reservation rule (mutants R1-R15, gate s20) nothing can mention a function before it is finished, so no cached value anywhere can depend on the function having had no body; the function's own cache is empty until it has a value. Dropping the caches on install is kept as defence in depth (it costs O(#globals) per finished function)",
    'M12 apply skips drain_pending': "after a successful unify() at the tactic level no deferred constraint is ever pending (every argument meta is applied to the whole context, so its unification is a pattern); drain_pending is exercised by the unifier gate (S3), not by apply",
}

def mutant_parts(spec):
    """a mutant is (old, new) for the kernel header, or (old, new, file) for another source file of this directory (a Python file needs no rebuild)"""
    return spec[0], spec[1], (spec[2] if len(spec) > 2 else 'itp_kernel.hpp')

def check_anchors():
    """every mutant must replace text that occurs EXACTLY ONCE in its file, and must change it; a stale anchor would silently turn a mutant into a no-op (or abort an hour-long run)"""
    texts, bad, same = {}, [], []
    for n, spec in MUTANTS.items():
        old, new, f = mutant_parts(spec)
        if f not in texts: texts[f] = open(os.path.join(SRC, f)).read()
        if texts[f].count(old) != 1: bad.append((n, texts[f].count(old)))
        if old == new: same.append(n)
    return bad, same

def gate_failed(rc, text): return rc != 0 or bool(FAIL_RE.search(text))

def run(names):
    bad, same = check_anchors()
    if bad or same:
        print("STALE MUTANT ANCHORS (fix them before running):"); [print("   %d occurrences: %s" % (c, n)) for n, c in bad]; [print("   no-op mutant: %s" % n) for n in same]; sys.exit(2)
    results = []
    cmds = [c for c in GATE_CMDS if not os.environ.get('MUT_ONLY') or c[0] in os.environ['MUT_ONLY'].split(',')]
    # the UNMUTATED build must be green on every gate, otherwise every mutant would look "caught" and the results would mean nothing
    base_env = gate_env(SRC)
    for cmd in cmds:
        r = subprocess.run([sys.executable, SRC + '/' + cmd[0]] + cmd[1:], cwd=SRC, env=base_env, capture_output=True, text=True, timeout=250)
        if r.returncode != 0:
            print("BASELINE IS RED on %s -- mutation results would be meaningless" % cmd[0]); print((r.stdout + r.stderr)[-500:]); sys.exit(2)
    print("baseline green on all %d gates" % len(cmds), flush=True)
    for nm in names:
        old, new, target = mutant_parts(MUTANTS[nm]); d = os.path.join(tempfile.gettempdir(), 'mut_' + re.sub(r'\W+', '_', nm)[:24])
        rebuild = target.endswith(('.hpp', '.cpp'))
        if os.path.exists(d): shutil.rmtree(d)
        shutil.copytree(SRC, d, ignore=shutil.ignore_patterns('*.so', '__pycache__') if rebuild else shutil.ignore_patterns('__pycache__'))
        h = open(d + '/' + target).read(); assert h.count(old) == 1, (nm, h.count(old))
        open(d + '/' + target, 'w').write(h.replace(old, new))
        env = dict(os.environ, ITP_NB_DOMAIN='lnl'); t0 = time.time()
        if rebuild:
            b = subprocess.run([sys.executable, 'build_kernel.py'], cwd=d, env=env, capture_output=True, text=True)
            if 'Build succeeded' not in b.stdout: results.append((nm, 'BUILD FAILED', b.stdout[-200:] + b.stderr[-200:])); print("  %-62s BUILD FAILED" % nm, flush=True); continue
        env = gate_env(d)
        caught_by = []
        for cmd in cmds:
            try:
                r = subprocess.run([sys.executable, d + '/' + cmd[0]] + cmd[1:], cwd=d, env=env, capture_output=True, text=True, timeout=100)
                rc, text = r.returncode, r.stdout + r.stderr
            except subprocess.TimeoutExpired:
                rc, text = 99, 'Traceback: HANG (timeout) -- the mutant does not terminate'
            if gate_failed(rc, text):
                first = next((l.strip()[:110] for l in text.split('\n') if FAIL_RE.search(l)), '')
                caught_by.append((cmd[0], first))
                if not os.environ.get('MUT_FULL'): break           # MUT_FULL=1 runs every gate to show how many independently catch the mutant
        if caught_by: status = 'CAUGHT'
        elif nm in KNOWN_EQUIVALENT: status = 'SURVIVED (documented equivalent mutant)'
        else: status = 'SURVIVED  <-- test gap'
        results.append((nm, status, caught_by)); print("  %-62s %s  (%.0fs)%s" % (nm, status, time.time() - t0, ('  | ' + ', '.join(c[0] for c in caught_by)) if caught_by and os.environ.get('MUT_VERBOSE') else ''), flush=True)
    return results

if __name__ == '__main__':
    if '--check' in sys.argv:
        bad, same = check_anchors(); print("%d mutants; %d stale anchors; %d no-op mutants" % (len(MUTANTS), len(bad), len(same)))
        for n, c in bad: print("   %d occurrences: %s" % (c, n))
        sys.exit(1 if bad or same else 0)
    prefixes = [a for a in sys.argv[1:] if not a.startswith('--')]
    names = [n for n in MUTANTS if any(n.startswith(p) for p in prefixes)] if prefixes else list(MUTANTS)
    res = run(names)
    caught = sum(1 for r in res if r[1] == 'CAUGHT'); eq = sum(1 for r in res if r[1].startswith('SURVIVED (doc')); gaps = [r[0] for r in res if r[1].startswith('SURVIVED  <--') or r[1] == 'BUILD FAILED']
    print("\n%d mutants: %d caught, %d documented equivalent, %d unexplained%s" % (len(res), caught, eq, len(gaps), ('   <-- ' + '; '.join(gaps)) if gaps else ''))
    sys.exit(1 if gaps else 0)
