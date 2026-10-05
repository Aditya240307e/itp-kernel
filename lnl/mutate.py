"""Mutation testing of the S2 typing core: each mutant re-breaks one rule in an isolated copy; the test suite must FAIL."""
import os, re, shutil, subprocess, sys, time
SRC = '/home/claude/itp_py/lnl'
MUTANTS = {
 'M1 guard disabled (fix_guard_ok always true)': ("    return is_locally_structurally_decreasing(n0, tid, f.decreasing_arg, to_levels(f.body, n0 + 1));", "    (void)tid; return true;"),
 'M2 index forcing is a no-op': ("    std::unordered_map<uint32_t, NodeId> m(forcing.forced.begin(), forcing.forced.end());\n    std::vector<NodeId> out = arg_types;", "    std::unordered_map<uint32_t, NodeId> m;\n    std::vector<NodeId> out = arg_types;"),
 'M3 lambda domain not unified with the expected domain': ("        if (ElabResult r = unify(dom_j, expected_dom, cur_ec, es); !r) return r;\n        Value next", "        (void)expected_dom;\n        Value next"),
 'M4 closing a Pi telescope off by one (wrong fvar prefix)': ("abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + j))", "abstract_fvars(opened_doms[j], std::vector<uint32_t>(fvl.begin(), fvl.begin() + std::min<int>(j + 1, (int)m)))"),
 'M5 application argument not checked': ("            if (ElabResult r = check(a.arg_id, dom_ty, ec, es); !r) return std::unexpected(r.error());\n            return quote(eval(pi->cod_id, pi->env.extend(eval(a.arg_id, ec.values))), ec.size());", "            (void)dom_ty;\n            return quote(eval(pi->cod_id, pi->env.extend(eval(a.arg_id, ec.values))), ec.size());"),
 'M6 admission universe rule disabled': ("if (!parameter_like && field_level[j] > *fam_level) {", "if (!parameter_like && field_level[j] > *fam_level + 1000) {"),
 'M7 strict positivity always true': ("                if (!is_strictly_positive(type_global_id, arg_ty))\n                    throw std::logic_error(\"itp: constructor '\" + ctor.name + \"' fails strict positivity", "                if (false)\n                    throw std::logic_error(\"itp: constructor '\" + ctor.name + \"' fails strict positivity"),
 'M8 Match branch count not checked': ("    if (ctor_gids.size() != branches.size())\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive, std::nullopt,\n            \"Match branch count", "    if (false)\n        return std::unexpected(TypeError{TypeErrorKind::MotiveInferenceFailure, motive, std::nullopt,\n            \"Match branch count"),
}
def run(names):
    results = []
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
        for cmd in (['s2_tests.py', '1500'], ['harness.py', '1500', '3']):
            r = subprocess.run([sys.executable, d + '/' + cmd[0]] + cmd[1:], cwd=d, env=env, capture_output=True, text=True, timeout=120)
            outs.append((r.returncode, r.stdout + r.stderr))
        text = '\n'.join(o[1] for o in outs)
        caught = any(o[0] != 0 for o in outs) or bool(re.search(r"\bBAD\b|PROBLEM|FAILED|disagreements [1-9]|Traceback|ACCEPTED|S2 GATE: FAIL", text))
        first = next((l.strip()[:110] for l in text.split('\n') if re.search(r"\bBAD\b|PROBLEM|FAILED|disagreements [1-9]|Traceback|ACCEPTED|S2 GATE: FAIL", l)), '')
        results.append((nm, 'CAUGHT' if caught else 'SURVIVED  <-- test gap', first)); print("  %-62s %s  (%.0fs)" % (nm, results[-1][1], time.time() - t0), flush=True)
    return results
if __name__ == '__main__':
    names = [n for n in MUTANTS if any(n.startswith(p) for p in sys.argv[1:])] if len(sys.argv) > 1 else list(MUTANTS)
    for nm, st, first in run(names): print("     ", nm[:50], "->", st, "|", first)
