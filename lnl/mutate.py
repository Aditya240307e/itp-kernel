"""Mutation testing of the S2 typing core: each mutant re-breaks one rule in an isolated copy; the test suite must FAIL."""
import os, re, shutil, subprocess, sys, time
SRC = '/home/claude/itp_py/lnl'
MUTANTS = {
 'M31 drain_pending re-unifies a deferred constraint in the outer context (the abort)': ("ElabCtx local = pc.ctx_at_deferral ? *pc.ctx_at_deferral : ec;", "ElabCtx local = ec;"),
 'M22 cases: scrutinee applied to the field binders (F9 regression)': ("NodeId ctor_app = forced_ctor_app(ctor_gid, n, ctor_arity, inst, forcing);", "NodeId ctor_app = mk_const(ctor_gid); for (uint32_t ai = 0; ai < ctor_arity; ++ai) ctor_app = mk_app(ctor_app, mk_fvar(n + ai));"),
 'M23 induct (goal side): scrutinee applied to the field binders': ("NodeId agent_ctor_app = forced_ctor_app(ctor_gid, n, ctor_arity, inst_agent, forcing_agent, &actual);", "NodeId agent_ctor_app = mk_const(ctor_gid); for (uint32_t fi = 0; fi < ctor_arity; ++fi) agent_ctor_app = mk_app(agent_ctor_app, mk_fvar(actual[fi]));"),
 'M24 induct (runtime side): scrutinee applied to the field binders': ("NodeId true_ctor_app = forced_ctor_app(ctor_gid, body_base, ctor_arity, inst, forcing_body);", "NodeId true_ctor_app = mk_const(ctor_gid); for (uint32_t ai = 0; ai < ctor_arity; ++ai) true_ctor_app = mk_app(true_ctor_app, mk_fvar(body_base + ai));"),
 'M25 kernel Match typing: expected type uses the field binders': ("NodeId ctor_app = forced_ctor_app(ctor_gids[ci], ec.size(), arity, inst, forcing);", "NodeId ctor_app = mk_const(ctor_gids[ci]); for (uint32_t ai = 0; ai < arity; ++ai) ctor_app = mk_app(ctor_app, mk_fvar(ec.size() + ai));"),
 'M26 forced application ignores the repeated-field rule': ("const bool use_forced = it != forcing.forced.end() && occurrence_count[base + ai] <= 1;", "const bool use_forced = it != forcing.forced.end();"),
 'M27 rewrite: goal not instantiated before abstraction (F13)': ("NodeId target_z = resolve_metas(g.target);", "NodeId target_z = g.target;"),
 'M28 rewrite: motive not type-checked (F13)': ("    if (!infer(motive, g.ec, ps.es))\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,\n            \"rewrite_eq: the motive", "    if (false)\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, g.target, std::nullopt,\n            \"rewrite_eq: the motive"),
 'M29 conversion nesting bound removed (the segfault)': ("    if (conv_nesting() > kMaxConvNesting) return false;\n", ""),
 'M30 induct accepted on a family without recursive constructors': ("if (!any_recursive)", "if (false)"),
 'M15 cases: no revert of dependent hypotheses': ("for (uint32_t j = 0; j < hyp_level; ++j) if (replace_terms(g.hyps[j].type, subs) != g.hyps[j].type) { revert_from = j; break; }", "for (uint32_t j = 0; j < 0u; ++j) if (replace_terms(g.hyps[j].type, subs) != g.hyps[j].type) { revert_from = j; break; }"),
 'M16 induct: induction hypothesis not added to the context': ("branch_ec = branch_ec.extend(ih_ty); branch_hyps.push_back(Hypothesis{\"\", ih_ty});", ""),
 'M17 induct: field levels assumed consecutive (ignore the IH shift)': ("actual[fi] = branch_ec.size();", "actual[fi] = n + fi;"),
 'M18 cases: delta not suppressed in unreverted branch targets': ("NoDeltaScope keep_constants_folded;\n            branch_expected = quote(eval(branch_expected, be_env), n + ctor_arity);", "branch_expected = quote(eval(branch_expected, be_env), n + ctor_arity);"),
 'M19 induct: revert closure is a single pass, not a fixpoint': ("needs_revert[j] = true; changed = true;", "needs_revert[j] = true;"),
 'M20 cases: refutation exploitation disabled': ("if (!reverted_js.empty()) exploit_ci.reset();", "exploit_ci.reset();"),
 'M21 induct: reverted layer opened with the wrong variable': ("remaining = instantiate1(p.cod_id, mk_fvar(base_level + static_cast<uint32_t>(ri)));", "remaining = instantiate1(p.cod_id, mk_fvar(base_level + static_cast<uint32_t>(ri) + 1));"),
 'M9 solve_pattern occurs check disabled': ("    if (occurs_meta(meta_id, other_id))\n        return std::unexpected(TypeError{TypeErrorKind::OccursCheckFailure, other_id, std::nullopt, \"metavariable occurs in its own proposed solution\"});", "    if (false)\n        return std::unexpected(TypeError{TypeErrorKind::OccursCheckFailure, other_id, std::nullopt, \"metavariable occurs in its own proposed solution\"});"),
 'M10 intro does not instantiate the codomain': ("dom_ty = p.dom_id; new_target = instantiate1(p.cod_id, mk_fvar(n));", "dom_ty = p.dom_id; new_target = p.cod_id;"),
 'M11 rewrite substitutes x instead of y': ("NodeId new_target = replace_fvars(body_open, {{n, y}});", "NodeId new_target = replace_fvars(body_open, {{n, x}});"),
 'M12 apply skips drain_pending': ("    if (ElabResult r = drain_pending(g.ec, ps.es); !r) return r;\n    std::vector<Goal> new_goals;", "    std::vector<Goal> new_goals;"),
 'M13 close_lambda_telescope domain prefix off by one': ("result = mk_lambda(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + j)), result);", "result = mk_lambda(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + std::min<int>(j + 1, (int)m))), result);"),
 'M14 meta applied to all but the last context variable': ("    for (uint32_t i = 0; i < n; ++i) r = mk_app(r, mk_fvar(i));\n    return r;", "    for (uint32_t i = 0; i + 1 < n; ++i) r = mk_app(r, mk_fvar(i));\n    return r;"),
 'M1 guard disabled (fix_guard_ok always true)': ("    return is_locally_structurally_decreasing(n0, tid, f.decreasing_arg, to_levels(f.body, n0 + 1));", "    (void)tid; return true;"),
 'M2 index forcing is a no-op': ("    std::unordered_map<uint32_t, NodeId> m(forcing.forced.begin(), forcing.forced.end());\n    std::vector<NodeId> out = arg_types;", "    std::unordered_map<uint32_t, NodeId> m;\n    std::vector<NodeId> out = arg_types;"),
 'M3 lambda domain not unified with the expected domain': ("        if (ElabResult r = unify(dom_j, expected_dom, cur_ec, es); !r) return r;\n        Value next", "        (void)expected_dom;\n        Value next"),
 'M4 closing a Pi telescope off by one (wrong fvar prefix)': ("result = mk_pi(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + j))", "result = mk_pi(abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + std::min<int>(j + 1, (int)m)))"),
 'M5 application argument not checked': ("            if (ElabResult r = check(a.arg_id, dom_ty, ec, es); !r) return std::unexpected(r.error());\n            return quote(eval(pi->cod_id, pi->env.extend(eval(a.arg_id, ec.values))), ec.size());", "            (void)dom_ty;\n            return quote(eval(pi->cod_id, pi->env.extend(eval(a.arg_id, ec.values))), ec.size());"),
 'M6 admission universe rule disabled': ("if (!parameter_like && field_level[j] > *fam_level) {", "if (!parameter_like && field_level[j] > *fam_level + 1000) {"),
 'M7 strict positivity always true': ("                if (!is_strictly_positive(type_global_id, arg_ty))\n                    throw std::logic_error(\"itp: constructor '\" + ctor.name + \"' fails strict positivity", "                if (false)\n                    throw std::logic_error(\"itp: constructor '\" + ctor.name + \"' fails strict positivity"),
 'M8 Match branch count not checked': ("    if (ctor_gids.size() != branches.size())\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive, std::nullopt,\n            \"Match branch count", "    if (false)\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive, std::nullopt,\n            \"Match branch count"),
}
GATE_CMDS = (['s2_tests.py', '1500'], ['harness.py', '1500', '3'], ['s3_tests.py', '600'], ['s4_tests.py', '800'], ['oracle_gate.py'])

def run(names):
    results = []
    # the UNMUTATED build must be green on every gate, otherwise every mutant would look "caught" and the results would mean nothing
    base_env = dict(os.environ, LNL_DIR=SRC, PYTHONPATH='/home/claude/itp_ref:/home/claude/itp_py:' + SRC)
    for cmd in GATE_CMDS:
        r = subprocess.run([sys.executable, SRC + '/' + cmd[0]] + cmd[1:], cwd=SRC, env=base_env, capture_output=True, text=True, timeout=250)
        if r.returncode != 0:
            print("BASELINE IS RED on %s -- mutation results would be meaningless" % cmd[0]); print((r.stdout + r.stderr)[-500:]); sys.exit(2)
    print("baseline green on all gates", flush=True)
    for nm in names:
        old, new = MUTANTS[nm]; d = '/tmp/mut_' + re.sub(r'\W+', '_', nm)[:24]
        if os.path.exists(d): shutil.rmtree(d)
        shutil.copytree(SRC, d, ignore=shutil.ignore_patterns('*.so', '__pycache__'))
        h = open(d + '/itp_kernel.hpp').read(); assert h.count(old) == 1, (nm, h.count(old))
        open(d + '/itp_kernel.hpp', 'w').write(h.replace(old, new))
        env = dict(os.environ, ITP_NB_DOMAIN='lnl'); t0 = time.time()
        b = subprocess.run([sys.executable, 'build_kernel.py'], cwd=d, env=env, capture_output=True, text=True)
        if 'Build succeeded' not in b.stdout: results.append((nm, 'BUILD FAILED', b.stdout[-200:] + b.stderr[-200:])); continue
        env = dict(os.environ, LNL_DIR=d, PYTHONPATH='/home/claude/itp_ref:/home/claude/itp_py:' + d)
        outs = []
        for cmd in GATE_CMDS:
            try:
                r = subprocess.run([sys.executable, d + '/' + cmd[0]] + cmd[1:], cwd=d, env=env, capture_output=True, text=True, timeout=100)
                outs.append((r.returncode, r.stdout + r.stderr))
            except subprocess.TimeoutExpired:
                outs.append((99, 'Traceback: HANG (timeout) -- the mutant does not terminate'))
        text = '\n'.join(o[1] for o in outs)
        caught = any(o[0] != 0 for o in outs) or bool(re.search(r"\bBAD\b|PROBLEM|FAILED|disagreements [1-9]|Traceback|ACCEPTED|S2 GATE: FAIL|S3 GATE: FAIL|S4 GATE: FAIL|ORACLE GATE: FAIL", text))
        first = next((l.strip()[:110] for l in text.split('\n') if re.search(r"\bBAD\b|PROBLEM|FAILED|disagreements [1-9]|Traceback|ACCEPTED|S2 GATE: FAIL|S3 GATE: FAIL|S4 GATE: FAIL|ORACLE GATE: FAIL", l)), '')
        results.append((nm, 'CAUGHT' if caught else 'SURVIVED  <-- test gap', first)); print("  %-62s %s  (%.0fs)" % (nm, results[-1][1], time.time() - t0), flush=True)
    return results
if __name__ == '__main__':
    names = [n for n in MUTANTS if any(n.startswith(p) for p in sys.argv[1:])] if len(sys.argv) > 1 else list(MUTANTS)
    for nm, st, first in run(names): print("     ", nm[:50], "->", st, "|", first)
