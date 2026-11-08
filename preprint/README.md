# preprint/ -- the technical report, and the scripts that produced every number in it

**ITP: A Dependently Typed Proof Assistant with an Integrated Environment for Learning to Prove** (technical report, version 2). The report is `itp_preprint.pdf`.

The report is built from the files in this directory. Every number it quotes is either *measured* (read from `measurements/` by `make_numbers.py`) or *counted* in the repository at the commit that the report names (lines of code, mutants, findings, problems, proofs, digests). Nothing is typed in by hand, so the text cannot drift from the evidence, and the build refuses to mix measurements made at different commits.

The numbers describe commit `2c71442` of the author's working copy; the digest of the code at that commit is `72651a6bbb2c6ecc` (`python3 tools/check_numbers.py --tree`). In this repository that code, together with the report, is the commit tagged `report-v2` (`git checkout report-v2`), and the report was published at that commit. This is version 2 of the report. It describes a kernel in which the positivity condition looks in the indices of a family and the motive of a `match` must fit its scrutinee and its constructors, and a second checker that re-checks the admission of families and applies the same rule to the motive (findings F92 and F93 in `docs/FINDINGS.md`); version 1 (commit `f840710`) described the kernel before that repair, in an appendix that this version does not have, and every number here was measured again on the repaired code. The commit that follows `2c71442` in the working copy changes the report and its measurements only, nothing below `src/`, `tests/` or `mutants/`. The commit after `report-v2` in this repository adds continuous integration (`.github/`) and changes one word below `tests/`: the layout gate (`tests/repo/s45_layout_tests.py`) lists `.github` among the entries it allows at the top level of the repository. The digest of the code at the head of the repository is therefore `4b60c5b890740eb4`; with that one word taken out it is `72651a6bbb2c6ecc` again. A rebuild of the report at the head (`python3 preprint/build.py`) would write the new digest into it, and nothing else would change; the report in this directory is the one that was published, and it is not rebuilt.

| path | what it is |
|---|---|
| `itp_preprint.pdf` | the report |
| `itp_preprint.tex` | the report as one self-contained LaTeX file (the numbers are pasted in), written by `build.py` |
| `src/` | the report, one file per section (`00_preamble.tex` ... `99_end.tex`) |
| `references.bib` | the bibliography |
| `generated_numbers.tex` | the macros that hold every number, written by `make_numbers.py` (do not edit) |
| `listings/` | the programs printed in the report (`ex_*.py`) and the output of each (`.out`) |
| `measurements/` | the raw measurements: JSON, and the logs they were parsed from (`bench/` the environment benchmark, `gates/` the logs of the gates that `measure_gates.py` reads) |
| `build.py` | numbers, then one flattened `itp_preprint.tex`, then the PDF |
| `make_numbers.py` | `measurements/` and the repository, then `generated_numbers.tex` |
| `measure.py`, `measure_check.py`, `measure_assembler.py`, `measure_gates.py` | the measurement scripts |
| `run_listings.py` | runs the listings and checks their output files |

`itp_preprint.tex` and `itp_preprint.pdf` are what `build.py` writes; they are kept here so that the report can be read without building it. The PDF carries no date or time (neither on the title page nor in its metadata), so building it again from the same files with the same TeX Live gives the same file.

## Building the report

    python3 -I preprint/build.py                 numbers, flattened .tex, PDF (pdflatex, bibtex, then pdflatex until the cross-references and the contents stop changing), all in preprint/
    python3 -I preprint/build.py --no-pdf        only the flattened .tex
    python3 -I preprint/build.py --draft         a layout check: numbers of measurements that do not exist yet are printed as ??

It needs a TeX Live with the packages loaded in `src/00_preamble.tex` (among them mathpartir, stmaryrd, mathpazo, algorithm, algpseudocode, tikz, pgfplots, listings, natbib and cleveref). The flattened file is self-contained apart from `references.bib` and the directory `listings/`, which must stay next to it.

## Repeating the measurements

Run these at the commit you want to describe, with the tree below `src/`, `tests/` and `mutants/` clean; each script records the commit and the machine. Counts and digests do not depend on the machine; times do. The times in the report are from one virtual machine with two cores. The scripts that need PyTorch and Gymnasium are the ones that use the optional packages of `requirements.txt`.

    python3 tests/run_gates.py --all --json preprint/measurements/gates_all.jsonl > preprint/measurements/gates_all.log     every gate (about an hour)
    python3 tests/run_gates.py --json preprint/measurements/gates_rerun.jsonl s51_suite_tests.py > preprint/measurements/gates_rerun.log     only if a gate failed in that run and was corrected (see below)
    python3 -I preprint/measure.py --out preprint/measurements/libs_run1.json           the library chain (about three minutes); run it twice, the second time to libs_run2.json
    python3 src/python/experiments/exp_bench.py data/dataset_v2 --out preprint/measurements/bench --no-train --problems 8 --probes 200 --time-limit 10 --repeats 2 --workers 1,2 --worker-problems 8 --worker-probes 100 --check-proofs 0
    python3 -I preprint/measure_check.py --out preprint/measurements/check.json         the cost of trusting a sample of reference proofs (about ten minutes, 2 GB)
    python3 -I preprint/measure_assembler.py --out preprint/measurements/assembler.json
    python3 tools/sanitize.py > LOG ; python3 -I preprint/measure_gates.py --with-env --sanitizers LOG
    python3 -I preprint/run_listings.py            the listings and their outputs (--check compares them with the files that exist)

`measurements/gates_rerun.jsonl` exists because one check of the gate S51 failed in the run of all the gates (it pinned the list of the findings that accepted what they must refuse, and F92 and F93 had been added to it on purpose); the check was corrected in one commit, S51 was run again alone, and `make_numbers.py` keeps the record of the second run for S51 and says so in the report. It checks that the only files below `src/`, `tests/` and `mutants/` that changed between the commit of the first run and the commit of the measurements are the scripts of the gates that were run again.

`measurements/kernel_build.log` and `measurements/gates_quick.log` hold the timed output of `python3 src/python/build_kernel.py all --force` (into an empty directory) and of `make gates-quick`. In the logs the path of the repository is written `<repo>` and a temporary directory `<out>`.
