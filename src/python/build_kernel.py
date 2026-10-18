#!/usr/bin/env python3
"""build_kernel.py -- portable build of the two kernels (the C++ header + its nanobind bindings).

    python3 src/python/build_kernel.py [hybrid|reference|all] [--out DIR] [--force] [--verbose]
                                       [--src DIR] [--module NAME] [--domain NAME|none]

    hybrid      src/c++/hybrid     ->  build/itp_env_lnl<suffix>   the kernel this project is about (de Bruijn + context levels, NbE, proof irrelevance, quotients, kernel-checked tactics)
    reference   src/c++/reference  ->  build/itp_env_ref<suffix>   the FROZEN previous generation (absolute levels); the oracle of the lockstep gates (checksums in src/c++/reference/SHA256SUMS)
    all         both (the default)

    --out DIR     where the module is written (default: build/ at the root of the repository, or $ITP_BUILD_DIR)
    --force       build even if the module is newer than its sources
    --src DIR     take the sources from DIR instead of src/c++/<kernel> (the mutation driver builds mutated copies this way)
    --module NAME the name of the Python module (default: itp_env_lnl / itp_env_ref); for `reference` the name inside the bindings, `itp_env`, is replaced in a generated copy, the source is not touched
    --domain NAME the nanobind domain (default: lnl / ref; `none` for no isolation).  Two kernels with different domains can be imported into ONE process, which the lockstep gates do

Environment: CXX picks the compiler; ITP_EXTRA_CXXFLAGS adds flags (tools/sanitize.py builds with "-fsanitize=address,undefined -O1 -g").

Every earlier build command of this project hardcoded Linux paths and called `g++` directly.  This script finds everything through portable introspection: nanobind's own include_dir()/source_dir(), sysconfig for
Python's headers and the extension suffix, and one platform check for the flags that really differ (macOS needs `-bundle -undefined dynamic_lookup` where Linux takes `-shared`).
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# kernel -> (source directory under src/c++, default module name, name of the module inside the bindings source, default nanobind domain)
KERNELS = {
    'hybrid': ('hybrid', 'itp_env_lnl', 'itp_env_lnl', 'lnl'),
    'reference': ('reference', 'itp_env_ref', 'itp_env', 'ref'),
}


def find_compiler() -> str:
    """$CXX if set (respects an explicit user choice, e.g. a newer clang++ installed alongside an older system default); otherwise the first of a portable-name list that actually exists.
    Deliberately never hardcodes 'g++': on macOS that name may not exist at all, or may itself be a thin alias for clang -- 'c++' is the POSIX-conventional, always-present name for whichever C++
    compiler a system considers its default, and is preferred first for exactly that reason."""
    if os.environ.get("CXX"):
        return os.environ["CXX"]
    for name in ("c++", "clang++", "g++"):
        if shutil.which(name):
            return name
    raise RuntimeError("No C++ compiler found (tried $CXX, c++, clang++, g++). "
                       "On macOS: install Xcode Command Line Tools (`xcode-select --install`). "
                       "On Linux: install g++ or clang++ via your package manager.")


def isolation_flags(cxx, domain):
    """Flags that let several independently built kernels be imported into ONE process (the reference kernel and the hybrid kernel side by side).
    NB_DOMAIN gives each module its own nanobind type registry; -fvisibility=hidden keeps nanobind internals private; -fno-gnu-unique (GCC/Linux) stops inline variables such as the thread_local
    active_context from being merged across modules by the dynamic linker even under RTLD_LOCAL (a STB_GNU_UNIQUE symbol is process-global)."""
    if not domain or domain == "none": return []
    flags = [f"-DNB_DOMAIN={domain}", "-fvisibility=hidden"]
    if platform.system() == "Linux":
        try:
            ver = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.lower()
        except Exception:
            ver = ""
        if "clang" not in ver: flags.append("-fno-gnu-unique")
    return flags


def python_include_dir() -> Path:
    """Python's own headers (Python.h and friends). Tries sysconfig first (portable, no shelling out); if Python.h isn't actually there, falls back to `python3-config --includes`, which is generated
    by the actual Python build and sometimes resolves correctly where sysconfig's reported path does not (seen in practice with macOS's Command Line Tools Python). Raises with a specific,
    actionable diagnosis rather than letting a confusing 'Python.h not found' surface from deep inside a real nanobind-including compile."""
    candidates = [Path(sysconfig.get_paths()["include"])]
    py_config = shutil.which("python3-config") or shutil.which(f"python{sys.version_info.major}.{sys.version_info.minor}-config")
    if py_config:
        r = subprocess.run([py_config, "--includes"], capture_output=True, text=True)
        if r.returncode == 0:
            for tok in r.stdout.split():
                if tok.startswith("-I"):
                    candidates.append(Path(tok[2:]))
    for c in candidates:
        if (c / "Python.h").is_file():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise RuntimeError(
        f"Python.h was not found in any of:\n  {tried}\n\n"
        f"This interpreter ({sys.executable}) does not have working development headers. On macOS, "
        f"this almost always means you're running the Command Line Tools' bundled Python "
        f"(/Library/Developer/CommandLineTools/usr/bin/python3) -- it runs Apple's own build "
        f"scripts and is not meant for building C extensions against; its headers are frequently "
        f"missing or misreported by its own sysconfig. The fix is a real, complete Python:\n"
        f"  brew install python@3.12\n"
        f"  /opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate\n"
        f"  pip install -r requirements.txt\n"
        f"  python3 src/python/build_kernel.py\n"
        f"(On Linux: install your distribution's -dev/-devel package, e.g. `apt install "
        f"python3-dev` or `dnf install python3-devel`.)")


def nanobind_paths() -> tuple[Path, Path, Path]:
    """(include_dir, source_dir, robin_map_include_dir) -- all three derived from nanobind's own Python-exposed, package-relative helpers (include_dir()/source_dir()) or a path relative to them
    (robin_map has no dedicated helper, but always lives at ext/robin_map/include next to nanobind's own include dir -- true on every platform, since it is just where pip unpacked the package)."""
    try:
        import nanobind
    except ImportError:
        raise RuntimeError("nanobind is not installed in this Python environment. Run: pip install nanobind") from None
    include_dir = Path(nanobind.include_dir())
    source_dir = Path(nanobind.source_dir())
    robin_map_dir = include_dir.parent / "ext" / "robin_map" / "include"
    if not robin_map_dir.is_dir():
        raise RuntimeError(f"expected robin_map headers at {robin_map_dir}, but that path does not exist -- nanobind's own packaging may have changed; check "
                           f"{include_dir.parent / 'ext'} for the actual location and adjust this script (this is the one place a directory layout is assumed rather than discovered).")
    return include_dir, source_dir, robin_map_dir


def shared_lib_flags() -> list[str]:
    """The one place object-file-to-loadable-extension flags genuinely differ by platform."""
    system = platform.system()
    if system == "Darwin":
        return ["-bundle", "-undefined", "dynamic_lookup"]
    if system == "Linux":
        return ["-shared"]
    if system == "Windows":
        raise RuntimeError("This project has not been built or tested on Windows. WSL2 (a Linux environment) is the supported path there.")
    raise RuntimeError(f"Unrecognized platform.system() = {system!r}; add its shared-library flags here (Linux: -shared; macOS: -bundle -undefined dynamic_lookup) to proceed.")


def try_compile(cxx: str, std_flag: str, extra_include: Path) -> bool:
    """Whether `cxx` accepts `std_flag` at all, checked with a trivial C++23-flavored program (concepts + <expected>, both features this codebase relies on) before committing to a full, slow
    build -- fails fast with a clear reason instead of a wall of real-source errors if the installed compiler is too old."""
    with tempfile.TemporaryDirectory(prefix="itp_probe_") as td:
        probe = Path(td) / "std_probe.cpp"
        probe.write_text(
            "#include <expected>\n#include <concepts>\n"
            "template<class T> concept Addable = requires(T a, T b) { a + b; };\n"
            "std::expected<int,int> f() { return 1; }\nint main() { return *f(); }\n"
        )
        r = subprocess.run([cxx, std_flag, f"-I{extra_include}", "-c", str(probe), "-o", "/dev/null"], capture_output=True, text=True)
        return r.returncode == 0


def pick_std_flag(cxx: str) -> str:
    """Prefers -std=c++23; falls back to -std=c++2b for compilers (some Apple clang versions among them) that only recognize the pre-standardization spelling. Raises with a clear, actionable
    message if neither works, rather than surfacing a wall of unrelated-looking template errors from deep inside the real source files."""
    sysconfig_include = Path(sysconfig.get_paths()["include"])
    for flag in ("-std=c++23", "-std=c++2b"):
        if try_compile(cxx, flag, sysconfig_include):
            return flag
    raise RuntimeError(
        f"{cxx} accepts neither -std=c++23 nor -std=c++2b (both tried, both failed to compile a trivial program using std::expected and concepts). This codebase needs a C++23-capable "
        f"compiler. On macOS: `xcode-select --install`, then `softwareupdate --list` to check for a newer Command Line Tools version -- Xcode 15+ is recommended. On Linux: g++ 12+ or clang++ 16+.")


def build_one(kernel: str, out_dir: Path, src: Path | None, module: str | None, domain: str | None, force: bool, verbose: bool) -> int:
    subdir, default_module, native_module, default_domain = KERNELS[kernel]
    src = Path(src) if src else ROOT / "src" / "c++" / subdir
    module = module or default_module
    domain = domain if domain is not None else os.environ.get("ITP_NB_DOMAIN") or default_domain
    bindings, header = src / "itp_bindings.cpp", src / "itp_kernel.hpp"
    for f in (bindings, header):
        if not f.is_file(): print(f"missing source: {f}", file=sys.stderr); return 2
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("sysconfig could not determine EXT_SUFFIX for this Python interpreter.")
    output = out_dir / f"{module}{ext_suffix}"
    newest = max(f.stat().st_mtime for f in (bindings, header, Path(__file__)))
    if output.exists() and not force and output.stat().st_mtime >= newest:
        print(f"{output.name}: up to date")
        return 0
    cxx = find_compiler()
    include_dir, source_dir, robin_map_dir = nanobind_paths()
    py_include = python_include_dir()
    std_flag = pick_std_flag(cxx)
    print(f"kernel:          {kernel}  ({src})")
    print(f"compiler:        {cxx}")
    print(f"C++ standard:    {std_flag}")
    print(f"platform:        {platform.system()} / {platform.machine()}")
    print(f"nanobind:        {include_dir.parent}")
    print(f"python headers:  {py_include}")
    print(f"nanobind domain: {domain}")
    print(f"output:          {output}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="itp_build_") as td:
        bindings_src = bindings
        if module != native_module:
            # the module name is spelled once, in NB_MODULE(<name>, m); a generated copy carries the requested one (the source file stays exactly as it is)
            text = bindings.read_text()
            needle = f"NB_MODULE({native_module}, m)"
            if text.count(needle) != 1:
                print(f"cannot rename the module: {needle!r} occurs {text.count(needle)} times in {bindings}", file=sys.stderr); return 2
            bindings_src = Path(td) / "itp_bindings.cpp"
            bindings_src.write_text(text.replace(needle, f"NB_MODULE({module}, m)"))
        cmd = [
            cxx, std_flag, "-O2", "-fPIC", "-pthread", *isolation_flags(cxx, domain),
            *os.environ.get("ITP_EXTRA_CXXFLAGS", "").split(),      # e.g. "-fsanitize=address,undefined -O1 -g" for the sanitizer build (tools/sanitize.py)
            f"-I{include_dir}", f"-I{robin_map_dir}", f"-I{py_include}", f"-I{src}",
            *shared_lib_flags(),
            str(bindings_src), str(source_dir / "nb_combined.cpp"),
            "-o", str(output),
        ]
        if verbose:
            print("\n" + " ".join(cmd) + "\n")
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=not verbose, text=True)
        dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"\nBUILD FAILED ({dt:.0f}s):\n", file=sys.stderr)
        if not verbose:
            print(r.stderr, file=sys.stderr)
        return 1
    print(f"\nBuild succeeded in {dt:.0f}s -> {output}")
    return 0


def main(argv) -> int:
    kernels, out, src, module, domain, force, verbose = [], None, None, None, None, False, False
    it = iter(argv)
    for a in it:
        if a in KERNELS or a == "all": kernels.append(a)
        elif a == "--out": out = Path(next(it))
        elif a == "--src": src = Path(next(it))
        elif a == "--module": module = next(it)
        elif a == "--domain": domain = next(it)
        elif a == "--force": force = True
        elif a == "--verbose": verbose = True
        elif a in ("-h", "--help"): print(__doc__); return 0
        else: print(f"unknown argument {a!r}\n\n{__doc__}", file=sys.stderr); return 2
    kernels = kernels or ["all"]
    if "all" in kernels:
        if src or module: print("--src and --module need a single kernel", file=sys.stderr); return 2
        kernels = list(KERNELS)
    out = out or Path(os.environ.get("ITP_BUILD_DIR") or ROOT / "build")
    rc = 0
    for k in kernels:
        rc = build_one(k, out, src, module, domain, force, verbose) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
