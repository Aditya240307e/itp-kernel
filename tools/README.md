# tools -- programs around the prover

Command-line programs that sit outside the prover: they check a machine, pack a release, run the kernel under the sanitizers, and work on the evidence folders. None of them is part of the trusted base, and none is needed to prove a theorem.

| tool | what it does | `make` target |
|---|---|---|
| [`preflight.py`](preflight.py) | is *this* machine ready: compiler and C++ library, stack size, worker processes, cores | |
| [`sanitize.py`](sanitize.py) | builds the kernel with AddressSanitizer and UndefinedBehaviorSanitizer and runs the gates under it | `make sanitize` |
| [`make_dist.py`](make_dist.py) | packs the repository into a `.tar.gz` and a `.zip` with a checksum manifest | `make dist` |
| [`review_packs.py`](review_packs.py) | splits the statement audit's review sheet into packs for people who read formal statements | |
| [`check_env.py`](check_env.py) | is this machine ready for the optional learning experiments (libraries, GPU, memory, disk) | `make check-env` |
| [`check_numbers.py`](check_numbers.py) | checks a LaTeX source's numbers, tables and figures against the evidence folders; prints the digest of the code the evidence is bound to | |

## `preflight.py` -- a new machine

```bash
python3 tools/preflight.py --quick      # seconds: the machine and the toolchain, no build
python3 tools/preflight.py              # 5-10 minutes: also builds the kernel and measures speeds
```

Every line is `[ok]`, `[!!]` (will stop or ruin a run: fix it first) or `[..]` (worth knowing). The checks are the ones that matter when the code runs on a new kind of machine: the C++ compiler and library (the kernel needs C++23 `std::expected` with its monadic operations: g++ 13 or later, Xcode 16 or later, or a recent LLVM), the size of the main thread's stack (macOS gives at most 64 MB, Linux 1 GiB; the kernel's guard turns an overflow into an error, and the checks show that it does), forked worker processes, the performance and efficiency cores of Apple silicon, battery, Low Power Mode and swap. Exit status 0 means ready (warnings may remain), 1 means something must be fixed.

Eight gates (S19, S25, S30, S31, S34, S37, S40, S50) limit the address space with `RLIMIT_AS`, which macOS does not enforce, so they need Linux.

## `sanitize.py` -- the kernel under the sanitizers

```bash
python3 tools/sanitize.py                              # build (about 80 s) and run the gates that can run under the sanitizers
python3 tools/sanitize.py s2_tests.py s16_quot_tests.py        # only these gates
python3 tools/sanitize.py --keep DIR                   # build into DIR and keep it
```

The evaluator owns its memory by hand (intrusive reference counts, single-allocation sequences, a trampoline that frees deep chains iteratively). A bug there would not give a wrong answer; it would be a use-after-free or a leak that the functional gates cannot see. Under ASan and UBSan such a bug is an immediate, located error. Run it after any change to the value representation, `Env`, `SharedSeq` or the release machinery in `itp_kernel.hpp`. The build happens in a scratch directory, so the repository's `build/` is untouched; the gates that are too slow under ASan are listed with the reason. Leak detection is off for the Python run (the interpreter itself leaks at exit); the evaluator's leaks are checked by a standalone driver compiled with `-fsanitize=address`.

## `make_dist.py` -- a release archive

```bash
python3 tools/make_dist.py             # dist/itp-YYYY-MM-DD-<commit>.tar.gz and .zip
python3 tools/make_dist.py --git       # ... with the history, as a working .git directory
```

The archive holds the files git knows (or, outside a git checkout, every file except build products and caches), `MANIFEST.sha256` (the SHA-256 of every file, in `sha256sum -c` format) and `RELEASE.txt` (which commit it was made from, and whether the tree had uncommitted changes). With `--git` the extracted directory is a working repository at that commit; its configuration is new and minimal, with no remotes and no user name, so nothing about the machine that made it leaks in. `--out DIR`, `--name NAME` and `--add PATH` (a build product git does not hold) are the other options.

## `review_packs.py` -- readers for the statements

A formal statement is only as good as its fit to the problem's words, and the [bank's automatic checks](../src/python/problems/README.md#does-the-statement-mean-what-the-problem-says) are evidence, not proof. `make audit` writes a review sheet with every statement; this tool splits it into packs, one per reader, in which every statement of the chosen split goes to a chosen number of different readers (round-robin over the statements sorted by level, so that each reader gets a similar mix):

```bash
python3 tools/review_packs.py results/audit --readers 4 --copies 2 --split test --out packs
```

Each pack holds the statements as Markdown and as a CSV to fill in, and a copy of the handout [`docs/SECOND_READER.md`](../docs/SECOND_READER.md); `assignment.tsv` lists the digest and the readers of each statement. The filled CSV comes back through `exp_audit.py import-reviews`. The packs are deterministic and nothing in the repository changes.

## `check_env.py` and `check_numbers.py`

These two serve the optional parts of the repository. `check_env.py` reports whether a machine can run the [learning experiments](../scripts/README.md) (Python libraries, that the kernel is built, cores, memory, free disk, GPU, the state of the git tree) and exits with 1 when something required is missing. `check_numbers.py` is for writing about the project's evidence: it reads a LaTeX source and reports a cited number, table or figure that no results folder holds, a folder that is not what its manifest says, and folders made for other code than the code at hand; `python3 tools/check_numbers.py --tree` prints the digest of the code that the [soundness and audit folders](../mutants/README.md) are bound to. `check_numbers.py` is one of the tools whose tests are themselves tested by injecting faults into them (`make tool-mutants`).
