#!/usr/bin/env python3
"""preflight.py -- is THIS machine ready for a run, and how long will the run take here?  The first thing to run on a computer the code has not been on (a Mac, a cloud instance).

    python3 tools/preflight.py                         # about 5-10 minutes: machine, toolchain, kernel build, stack guard, worker processes, speed measurements, projected duration, settings to use
    python3 tools/preflight.py --quick                 # seconds: the machine and the toolchain only (no build, no measurement)
    python3 tools/preflight.py --profile small --seeds 0-2 --set 'archs=()' --set 'widths=()' --set 'scratch=()'      # the run to project (default: the profile `small`, seeds 0-2)

Every line is [ok], [!!] (something that will stop or ruin a run: fix it first) or [..] (worth knowing).  The checks are the ones that went wrong, or could have, when the code was ported from the Linux machine it was written on to
macOS on Apple silicon: the C++ compiler and library (the kernel needs C++23 `std::expected` with its monadic operations: Xcode 16 or newer, or a recent LLVM), the stack of the main thread (macOS gives it at most 64 MB, Linux
1 GiB; the kernel's guard turns an overflow into an error, the checks below show that it does), forked worker processes next to PyTorch, the performance and efficiency cores of Apple silicon (only the fast ones are used), the
battery, Low Power Mode and swap.  Then the measurements: exp_bench on a small set of problems (world build and memory, probes per second of each solver, training steps per second, scaling over worker processes) and the
projection of the chosen profile with THOSE rates (exp_estimate), against the memory and cores the runner will really use.  Everything is written to OUT/preflight.json.

Gates that need Linux (they limit the address space with RLIMIT_AS, which macOS does not enforce): S19, S25, S30, S31, S34, S37, S40, S50.  Run the gates, the soundness folder and the audit on a Linux machine; this machine
runs the experiments.  Exit status: 0 ready (warnings may remain), 1 something must be fixed."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, re, json, time, shutil, platform, argparse, subprocess, tempfile
import ml_meta as MT

ROOT = itp_paths.ROOT
REPORT = []           # (status, name, detail): status True / False / None


def add(status, name, detail=''):
    REPORT.append((status, name, detail))
    print('  [%s] %-30s %s' % ('ok' if status is True else ('!!' if status is False else '..'), name, detail), flush=True)


def section(title): print('\n' + title, flush=True)


def run(argv, timeout=60, **kw):
    """(returncode, stdout + stderr) of a command; (None, why) if it could not be run"""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kw)
        return r.returncode, (r.stdout or '') + (r.stderr or '')
    except subprocess.TimeoutExpired: return None, 'no answer within %d s' % timeout
    except OSError as e: return None, str(e)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ parsers (tested by S51)
def parse_compiler_version(text):
    """(family, major, banner) of the first line of `cxx --version`: family 'apple' (Apple clang, the Xcode compiler), 'clang', 'gcc', or None"""
    line = (text or '').strip().split('\n')[0] if text else ''
    m = re.search(r'Apple clang version (\d+)', line)
    if m: return 'apple', int(m.group(1)), line
    m = re.search(r'clang version (\d+)', line)
    if m: return 'clang', int(m.group(1)), line
    if line and 'clang' not in line.lower():                                                          # g++, gcc, c++ (the GNU compilers print `NAME (distribution) 13.3.0 ...`)
        v = re.findall(r'(\d+)\.\d+(?:\.\d+)?', line)
        if v: return 'gcc', int(v[-1]), line
    return None, 0, line


def compiler_ok(family, major):
    """True / False / None (unknown): can this compiler's library be expected to have what the kernel uses (std::expected with and_then / or_else / transform: libc++ 17, libstdc++ 13)?  Apple clang 16 (Xcode 16) has libc++ 17"""
    if family == 'apple': return major >= 16
    if family == 'clang': return major >= 17
    if family == 'gcc': return major >= 13
    return None


def parse_power_source(text):
    """'ac', 'battery' or None from `pmset -g batt`"""
    if "'AC Power'" in text: return 'ac'
    if "'Battery Power'" in text: return 'battery'
    return None


def parse_low_power_mode(text):
    """True / False / None from `pmset -g` (the line `lowpowermode 1`)"""
    m = re.search(r'lowpowermode\s+(\d)', text or '')
    return None if not m else m.group(1) == '1'


def parse_swapusage(text):
    """{'total_mb', 'used_mb', 'free_mb'} from `sysctl -n vm.swapusage` (`total = 2048.00M  used = 1023.25M  free = 1024.75M  (encrypted)`), {} if it does not parse"""
    out = {}
    for k, v, u in re.findall(r'(total|used|free)\s*=\s*([\d.]+)([MGK])', text or ''):
        out[k + '_mb'] = float(v) * {'K': 1 / 1024.0, 'M': 1.0, 'G': 1024.0}[u]
    return out if len(out) == 3 else {}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the checks
def check_machine(out_dir):
    section('Machine')
    system = platform.system(); cpu = MT.cpu_model(); mem = MT.memory_gb(); n = os.cpu_count() or 1; usable = MT.usable_cores()
    info = {'system': system, 'release': platform.release(), 'machine': platform.machine(), 'cpu': cpu, 'cores': n, 'cores_usable': usable, 'memory_gb': mem}
    add(True, 'system', '%s %s, %s, %s' % (system, platform.mac_ver()[0] if system == 'Darwin' else platform.release(), platform.machine(), cpu))
    add(True if mem else None, 'memory', '%.1f GB' % mem if mem else 'unknown')
    if system == 'Darwin':
        eff = MT._sysctl('hw.perflevel1.logicalcpu')
        add(True, 'cores', '%d logical: %d performance + %s efficiency; the runner counts only the %d performance cores' % (n, usable, eff or '?', usable))
        if MT._sysctl('sysctl.proc_translated') == '1' or (platform.machine() == 'x86_64' and 'Apple' in cpu):
            add(False, 'native Python', 'this Python runs under Rosetta 2 (x86_64 on Apple silicon), about half the speed and too slow for the run: install an arm64 Python (python.org, Homebrew or conda-forge osx-arm64)')
        else: add(True, 'native Python', 'arm64' if platform.machine() == 'arm64' else platform.machine())
        rc, txt = run(['pmset', '-g', 'batt'], 10); src = parse_power_source(txt) if rc == 0 else None
        add(None if src is None else (src == 'ac'), 'power', {'ac': 'AC power', 'battery': 'ON BATTERY: plug in (a laptop on battery throttles, and sleeps when the lid closes)', None: 'unknown'}[src])
        rc, txt = run(['pmset', '-g'], 10); lp = parse_low_power_mode(txt) if rc == 0 else None
        add(None if lp is None else (not lp), 'Low Power Mode', {True: 'ON: turn it off (System Settings > Battery); it halves the speed', False: 'off', None: 'unknown'}[lp])
        sw = parse_swapusage(MT._sysctl('vm.swapusage')); info['swap_mb'] = sw
        if sw: add(sw['used_mb'] < 1024, 'swap in use', '%.0f MB of %.0f MB%s' % (sw['used_mb'], sw['total_mb'], '' if sw['used_mb'] < 1024 else '  (close other applications before a long run: swapping makes a run an order of magnitude slower)'))
        add(None, 'sleep', 'a long run stops when the Mac sleeps: start it as   caffeinate -dimsu ./scripts/run_all.sh ...')
    else: add(True, 'cores', '%d (%d usable by this process)' % (n, usable))
    free = shutil.disk_usage(ROOT).free / 2 ** 30; info['free_disk_gb'] = free
    add(False if free < 10 else (None if free < 30 else True), 'free disk', '%.0f GB  (the profile `small` writes a few GB -- 0.5 GB for a pilot of 40 runs -- and `full` tens of GB: checkpoints, caches, logs)' % free)
    return info


def parse_bash_version(txt):
    """'3.2.57(1)-release' (BASH_VERSION) -> (3, 2); anything else -> None"""
    m = re.match(r'\s*(\d+)\.(\d+)', txt or '')
    return (int(m.group(1)), int(m.group(2))) if m else None


def check_toolchain():
    section('Toolchain')
    import build_kernel as BK
    add(sys.version_info >= (3, 10), 'python', sys.version.split()[0] + ('' if sys.version_info >= (3, 10) else '  (3.10 or newer; 3.13 was used for the paper)'))
    for mod in ('numpy', 'scipy', 'nanobind', 'torch', 'gymnasium', 'matplotlib'):
        try:
            m = __import__(mod); add(True, mod, getattr(m, '__version__', '?'))
        except Exception as e:
            add(False if mod != 'matplotlib' else None, mod, 'missing (pip install -r requirements.txt): %s' % e)
    rc, txt = run(['bash', '-c', 'echo "$BASH_VERSION"'], 10); bv = parse_bash_version(txt) if rc == 0 else None
    if bv is None: add(False, 'bash', 'no `bash` on the PATH: the scripts need it (`brew install bash` on a Mac)')
    else: add(True if bv[0] >= 4 else None, 'bash', txt.strip() + ('' if bv[0] >= 4 else '  (the macOS default: the scripts avoid bash 4 features and were read for it, but not run under it; if one fails with "bad substitution" or "unbound variable", `brew install bash`)'))
    try:
        cxx = BK.find_compiler()
    except RuntimeError as e:
        add(False, 'C++ compiler', str(e)); return None
    rc, txt = run([cxx, '--version'], 20); fam, major, banner = parse_compiler_version(txt)
    ok = compiler_ok(fam, major)
    add(ok if ok is not None else None, 'C++ compiler', '%s   (%s)' % (cxx, banner) + ('' if ok is not False else '  -- too old for the kernel (C++23 std::expected): Xcode 16 or newer, or `brew install llvm` and CXX=$(brew --prefix llvm)/bin/clang++'))
    probe = ('#include <expected>\n#include <utility>\n#include <variant>\n#include <vector>\n'
             'struct A; using V = std::variant<int, std::vector<int>>;\n'
             'int f(int x) { std::expected<int, int> e = x; auto r = e.and_then([](int v) -> std::expected<int, int> { return v + 1; }).transform([](int v) { return v * 2; }); if (!r) std::unreachable(); return *r; }\n'
             'int main() { return f(1) == 4 ? 0 : 1; }\n')
    with tempfile.TemporaryDirectory(prefix='itp_pre_') as td:
        src = os.path.join(td, 'p.cpp'); open(src, 'w').write(probe)
        std = None
        try: std = BK.pick_std_flag(cxx)
        except Exception as e: add(False, 'C++23', 'neither -std=c++23 nor -std=c++2b works: %s' % e)
        if std:
            rc, txt = run([cxx, std, src, '-o', os.path.join(td, 'p')], 120)
            if rc != 0: add(False, 'C++23 library', '%s: std::expected::and_then / std::unreachable are not usable with this compiler\'s standard library:\n%s' % (std, txt[-600:]))
            else:
                rc, _ = run([os.path.join(td, 'p')], 20); add(rc == 0, 'C++23 library', '%s: std::expected (with its monadic operations) and std::unreachable work' % std)
    return cxx


def check_kernel(rebuild):
    section('Kernel')
    t0 = time.time()
    rc, txt = run([sys.executable, os.path.join(ROOT, 'src', 'python', 'build_kernel.py'), 'hybrid'] + (['--force'] if rebuild else []), 1500)
    dt = time.time() - t0
    if rc != 0:
        add(False, 'kernel build', 'FAILED after %.0f s:\n%s' % (dt, txt[-1500:])); return False
    add(True, 'kernel build', ('built in %.0f s' % dt) if 'up to date' not in txt else 'up to date (use --rebuild to time a build)')
    code = ("import itp_paths, itp_env_lnl as k, sys\nprint('LIMIT', k.set_stack_limit(0))\n")
    rc, txt = run([sys.executable, '-c', code], 60, cwd=os.path.join(ROOT, 'src', 'python'))
    m = re.search(r'LIMIT (\d+)', txt)
    if rc != 0 or not m: add(False, 'kernel import', txt[-500:]); return False
    lim = int(m.group(1))
    add(True, 'kernel import', 'the main thread\'s stack limit in force is %s%s' % ('unlimited' if lim >= 2 ** 62 else '%d MiB' % (lim >> 20), '  (Linux raises it to 1 GiB; macOS allows at most 64 MB: deeper terms are refused by the guard, which is an error, never a crash)' if sys.platform == 'darwin' else ''))
    # the stack guard: a term far deeper than a 1 MiB thread stack must end in an error, a legitimate deep term on the main thread must finish (S18's own children)
    s18 = os.path.join(ROOT, 'tests', 'kernel', 's18_deep_tests.py'); bad = []
    for shape, mode in (('numeral', 'small'), ('lambda', 'small'), ('numeral', 'big'), ('lambda', 'big')):
        rc, txt = run([sys.executable, s18, '--child', shape, mode], 300, cwd=os.path.dirname(s18))
        lines = txt.strip().split('\n') if txt.strip() else []
        done = bool(lines) and lines[-1].startswith('DONE')
        stack = [l for l in lines if l.endswith(' stack')]
        if rc != 0 or not done: bad.append('%s/%s: the process died (exit %s)' % (mode, shape, rc))
        elif 'AFTER ok' not in lines: bad.append('%s/%s: the kernel was broken afterwards' % (mode, shape))
        elif mode == 'small' and not stack: bad.append('%s/%s: the guard never fired' % (mode, shape))
        elif mode == 'big' and stack: bad.append('%s/%s: the guard refused a term the stack should hold (%d operations); the main thread\'s stack is too small for it' % (mode, shape, len(stack)))
    add(not bad, 'stack guard', 'deep terms: an error on a small thread, an answer on the main thread' if not bad else '; '.join(bad) + '   (run tests/kernel/s18_deep_tests.py for the whole picture)')
    return not bad or all('too small' in b for b in bad)


FORK_PROBE = r'''
import sys, os, signal
sys.path.insert(0, os.path.join(%(root)r, 'src', 'python')); import itp_paths
signal.alarm(120)
import torch
torch.set_num_threads(1)
a = torch.randn(256, 256); (a @ a).sum().item()                       # (the parent has used torch before it forks, as the training and evaluation commands do)
import ml_pool
def init(): torch.set_num_threads(1); return {}
def h(st, p):
    x = torch.randn(128, 128); (x @ x).sum().item(); return p
pool = ml_pool.Pool(2, init, {'h': h})
print('RESULT', pool.map('h', [1, 2, 3, 4]))
pool.close()
'''


def fork_probe_ok(rc, txt):
    """the probe exited cleanly and every one of its four jobs came back, in order"""
    return rc == 0 and 'RESULT [1, 2, 3, 4]' in txt


def check_workers():
    section('Worker processes')
    rc, txt = run([sys.executable, '-c', FORK_PROBE % {'root': ROOT}], 150)
    ok = fork_probe_ok(rc, txt)
    add(ok, 'forked workers + torch', 'work (WORKERS=1 or 2 can be used)' if ok else 'FAILED (exit %s): use WORKERS=0 and parallelise with JOBS instead\n%s' % (rc, txt[-500:]))
    return ok


def check_speed(out_dir, quick_args):
    section('Speed (exp_bench on a few problems: world, solvers, training, workers)')
    ddir = os.path.join(ROOT, 'data', 'dataset_v2')
    if not os.path.isdir(ddir):
        add(None, 'dataset', 'data/dataset_v2 is not here: the speed is not measured'); return None
    bench = os.path.join(out_dir, 'bench'); t0 = time.time(); quick_args = list(quick_args)
    if '--model' not in quick_args:                                   # (the searches that use a network and the training speed are measured with an untrained one: the same size, the same cost)
        init = os.path.join(out_dir, 'init')
        rc0, txt0 = run([sys.executable, os.path.join(ROOT, 'src', 'python', 'learner', 'ml_init.py'), ddir, '--out', init], 1800)
        if rc0 == 0 and os.path.exists(os.path.join(init, 'best.pt')): quick_args += ['--model', os.path.join(init, 'best.pt')]
        else: add(None, 'untrained network', 'ml_init.py failed (exit %s): the speed of the networks (training, sampling, IDA*) is not measured\n%s' % (rc0, txt0[-300:]))
    argv = [sys.executable, os.path.join(ROOT, 'src', 'python', 'experiments', 'exp_bench.py'), ddir, '--out', bench] + quick_args
    rc, txt = run(argv, 3600)
    path = os.path.join(bench, 'bench.json')
    if rc != 0 or not os.path.exists(path):
        add(False, 'exp_bench', 'FAILED after %.0f s:\n%s' % (time.time() - t0, txt[-1500:])); return None
    b = json.load(open(path)); w = b.get('world') or {}
    add(True, 'exp_bench', '%.0f s' % (time.time() - t0))
    add(None, 'world', 'built in %s s, %s MB resident afterwards' % (round(w['seconds']) if w.get('seconds') is not None else '?', round(w['rss_mb']) if w.get('rss_mb') is not None else '?'))
    for s, v in sorted((b.get('solvers') or {}).items()):
        add(None, 'solver ' + s, '%s probes/s' % (round(v['probes_per_s'], 1) if v.get('probes_per_s') else '?'))
    if w.get('train_steps_per_s'): add(None, 'training', '%.0f steps/s (batch 64, the reference machine does 55)' % w['train_steps_per_s'])
    wk = b.get('workers') or {}
    if wk: add(None, 'worker scaling', json.dumps({k: (round(v.get('speedup', 0), 2) if isinstance(v, dict) else v) for k, v in wk.items()}) if isinstance(wk, dict) else str(wk)[:200])
    return path


def project(profile, seeds, sets, bench_path, cores, mem_gb):
    section('Projection: profile %s, seeds %s%s' % (profile, seeds, (' ' + ' '.join('--set ' + s for s in sets)) if sets else ''))
    import exp_estimate as ES
    argv = [profile, '--seeds', seeds, '--cores', str(cores)] + (['--bench', bench_path] if bench_path else [])
    for s in sets: argv += ['--set', s]
    ES.main(argv)
    import exp_spec as SP, exp_run as RUN
    runs = SP.build(profile, seeds=RUN.parse_seeds(seeds), runs_dir='runs/' + profile, overrides=SP.parse_overrides(sets))
    print('\nruns that fit at once in %d cores and %.0f GB (the runner\'s own packing rule):' % (cores, mem_gb))
    for stage in ('data', 'pre', 'ft', 'rl', 'eval', 'bench'):
        rs = [r for r in runs.values() if r.stage == stage]
        if not rs: continue
        c = max(r.cores for r in rs); m = max(r.mem_gb for r in rs)
        print('  %-6s %3d runs, each up to %d cores and %.1f GB  ->  %d at once' % (stage, len(rs), c, m, max(1, min(int(cores // c), int(mem_gb // m)))))


def settings(info, fork_ok, cores, mem_gb):
    section('Settings to use here (scripts/local.env, or the environment of the call)')
    workers = 1 if fork_ok and mem_gb >= 10 else 0
    print('  JOBS=%d   MEM_GB=%d   WORKERS=%d   DEVICE=cpu' % (max(1, cores - 1) if cores > 4 else cores, int(mem_gb), workers))
    print('  (DEVICE=cpu: the networks are small and the proof search, the larger share of the time, runs on the CPU anyway; Apple\'s GPU (MPS) is not used by this code)')
    if info.get('system') == 'Darwin': print('  start long runs with   caffeinate -dimsu ./scripts/run_all.sh   (the Mac does not sleep while it runs)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split('\n', 2)[2])
    ap.add_argument('--quick', action='store_true', help='the machine and the toolchain only'); ap.add_argument('--rebuild', action='store_true', help='build the kernel again to time it')
    ap.add_argument('--profile', default='small'); ap.add_argument('--seeds', default='0-2'); ap.add_argument('--set', action='append', default=[]); ap.add_argument('--out', default=os.path.join(ROOT, 'runs', 'preflight'))
    ap.add_argument('--bench-args', default='--problems 8 --probes 60 --repeats 1 --workers 1,2 --worker-problems 8 --worker-probes 60 --check-proofs 0 --no-gym', help='the options of exp_bench')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    info = check_machine(a.out)
    cxx = check_toolchain()
    result = {'machine': info, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'profile': a.profile, 'seeds': a.seeds}
    fork_ok = None; bench_path = None
    if not a.quick:
        if cxx and check_kernel(a.rebuild):
            fork_ok = check_workers()
            bench_path = check_speed(a.out, a.bench_args.split())
        else: add(False, 'kernel', 'not built or not importable: nothing below can be measured')
    import exp_run as RUN
    cores, mem_gb = RUN.default_budgets(info.get('memory_gb') or 8.0, info['cores_usable'])
    if not a.quick and not any(s is False for s, _, _ in REPORT):
        project(a.profile, a.seeds, a.set, bench_path, cores, mem_gb)
    settings(info, fork_ok, cores, mem_gb)
    bad = [n for s, n, _ in REPORT if s is False]
    result['checks'] = [{'ok': s, 'name': n, 'detail': d} for s, n, d in REPORT]; result['recommended'] = {'cores': cores, 'mem_gb': mem_gb, 'fork_workers_ok': fork_ok}
    json.dump(result, open(os.path.join(a.out, 'preflight.json'), 'w'), indent=1)
    print('\n%s  (%.0f s; written to %s)' % ('NOT READY: ' + ', '.join(bad) if bad else 'READY', time.time() - t0, os.path.join(a.out, 'preflight.json')))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
