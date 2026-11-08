# preprint/ -- the technical report, and the scripts that produced every number in it

**ITP: A Dependently Typed Proof Assistant with an Integrated Environment for Learning to Prove** (technical report, version 1). The report is `itp_preprint.pdf`.

The report is built from the files in this directory. Every number it quotes is either *measured* (read from `measurements/` by `make_numbers.py`) or *counted* in the repository at the commit that the report names (lines of code, mutants, findings, problems, proofs, digests). Nothing is typed in by hand, so the text cannot drift from the evidence, and the build refuses to mix measurements made at different commits.

The numbers describe commit `f840710`. The commits after it add this directory, list it in the main `README.md`, let the layout gate S45 (`tests/repo/s45_layout_tests.py`) accept it at the top level, add to Appendix A the proof of `False` that the second deviation allows once excluded middle and propositional extensionality are added (`listings/dev_exploit.py`) and its cross-check in Lean (`listings/dev_lean_decl.lean`, `listings/dev_lean_false.lean`), make `build.py` run pdflatex until the cross-references settle, and fill in the address of the repository and the DOI of the archive in the paragraph on availability, remove a note from the index of `docs/` and add the licence of the repository (`LICENSE`); the kernel is not changed by any of them (the digest of its sources is that of the measurements), and neither deviation is repaired.

| path | what it is |
|---|---|
| `itp_preprint.pdf` | the report |
| `itp_preprint.tex` | the report as one self-contained LaTeX file (the numbers are pasted in), written by `build.py` |
| `src/` | the report, one file per section (`00_preamble.tex` ... `99_end.tex`) |
| `references.bib` | the bibliography |
| `generated_numbers.tex` | the macros that hold every number, written by `make_numbers.py` (do not edit) |
| `listings/` | the programs printed in the report (`ex_*.py`, `dev_*.py`, and the two Lean files `dev_lean_*.lean`) and the output of each (`.out`) |
| `measurements/` | the raw measurements: JSON, and the logs they were parsed from (`bench/` the environment benchmark, `gates/` the logs of the gates that `measure_gates.py` reads) |
| `build.py` | numbers, then one flattened `itp_preprint.tex`, then the PDF |
| `make_numbers.py` | `measurements/` and the repository, then `generated_numbers.tex` |
| `measure.py`, `measure_check.py`, `measure_deviations.py`, `measure_assembler.py`, `measure_gates.py` | the measurement scripts |
| `run_listings.py` | runs the listings and checks their output files (the Lean files need a Lean 4 on the `PATH` or named by `$LEAN`; the report used Lean 4.15.0 with the core library only, and without a Lean they are skipped) |

`itp_preprint.tex` and `itp_preprint.pdf` are what `build.py` writes; they are kept here so that the report can be read without building it. The PDF carries no date or time (neither on the title page nor in its metadata), so building it again from the same files with the same TeX Live gives the same file.

## Building the report

    python3 -I preprint/build.py                 numbers, flattened .tex, PDF (pdflatex, bibtex, then pdflatex until the cross-references and the contents stop changing), all in preprint/
    python3 -I preprint/build.py --no-pdf        only the flattened .tex
    python3 -I preprint/build.py --draft         a layout check: numbers of measurements that do not exist yet are printed as ??

It needs a TeX Live with the packages loaded in `src/00_preamble.tex` (among them mathpartir, stmaryrd, mathpazo, algorithm, algpseudocode, tikz, pgfplots, listings, natbib and cleveref). The flattened file is self-contained apart from `references.bib` and the directory `listings/`, which must stay next to it.

## Repeating the measurements

Run these at the commit you want to describe, with the tree below `src/`, `tests/` and `mutants/` clean; each script records the commit and the machine. Counts and digests do not depend on the machine; times do. The times in the report are from one virtual machine with two cores. The scripts that need PyTorch and Gymnasium are the ones that use the optional packages of `requirements.txt`.

    python3 tests/run_gates.py --all --json preprint/measurements/gates_all.jsonl > preprint/measurements/gates_all.log     every gate (about an hour)
    python3 -I preprint/measure.py --out preprint/measurements/libs_run1.json           the library chain (about three minutes); run it twice, the second time to libs_run2.json
    python3 src/python/experiments/exp_bench.py data/dataset_v2 --out preprint/measurements/bench --no-train --problems 8 --probes 200 --time-limit 10 --repeats 2 --workers 1,2 --worker-problems 8 --worker-probes 100 --check-proofs 0
    python3 -I preprint/measure_check.py --out preprint/measurements/check.json         the cost of trusting a sample of reference proofs (about ten minutes, 2 GB)
    python3 -I preprint/measure_deviations.py --out preprint/measurements/deviations.json
    python3 -I preprint/measure_assembler.py --out preprint/measurements/assembler.json
    python3 tools/sanitize.py > LOG ; python3 -I preprint/measure_gates.py --with-env --sanitizers LOG
    python3 -I preprint/run_listings.py            the listings and their outputs (--check compares them with the files that exist)

`measurements/kernel_build.log` and `measurements/gates_quick.log` hold the timed output of `python3 src/python/build_kernel.py all --force` (into an empty directory) and of `make gates-quick`. In the logs the path of the repository is written `<repo>` and a temporary directory `<out>`.

In the run of all the gates (`gates_all.jsonl`, `gates_all.log`) the layout gate S45 objected to this directory, which was then untracked and not yet allowed at the top level. `gates_s45_clean.jsonl` and `gates_s45_clean.log` are a run of S45 alone with the directory moved aside; the report uses that record for S45 and says so in Appendix D. With the directory allowed, a new run of all the gates needs no such workaround.
