# src/c++/reference -- the frozen reference kernel

The previous generation of the kernel: absolute universe levels, variables as levels. It is **frozen** -- `SHA256SUMS` records the checksums of its two files and the layout gate S45 fails if either changes -- because
its job is to be a fixed point of comparison. The gates `s2` ... `s15` and the oracle run the same random scripts through this kernel and through the hybrid kernel (`src/c++/hybrid`) in lockstep and demand the same
verdicts; where the hybrid kernel is deliberately stricter (finding F19) or fixes a reference bug (F-series in `docs/FINDINGS.md`), the gate says so explicitly.

* `itp_kernel.hpp`, `itp_bindings.cpp` -- the sources, byte-for-byte as they were (the bindings name their module `itp_env`; `build_kernel.py reference` builds a generated copy named `itp_env_ref`, so both kernels can be
  imported into one process, and leaves these files untouched).
* Build: `python3 src/python/build_kernel.py reference` (or `all`).
* Do not edit. A bug found in this kernel is recorded in `docs/FINDINGS.md` and fixed in the hybrid kernel only.
