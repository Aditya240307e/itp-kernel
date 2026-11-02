# results/

`make report` (scripts/07_report.sh) writes the **results folder** of a profile here: `results/<profile>/` (`smoke`, `small`, `full`; set another place with `RESULTS=` or `--results`).
It is a copy of `runs/<profile>/report/` and holds everything a paper is written from, apart from the paper: the tables (md, tex, csv), the figures (pdf, png, and a csv of the values each one draws),
`numbers.json`, the outcome of every attempt on every problem, the statistics (rates, paired comparisons with Holm's correction, the resolution of the benchmark), and the provenance (profile, run graph,
commits, machine, digests of the dataset).  Its `README.md` says what each file is; `python3 src/python/experiments/exp_bundle.py --verify results/<profile>` checks the folder against its manifest.

`results/soundness/` is the other kind of folder: the soundness evidence (`make soundness`, scripts/08_soundness.sh): the gates, the mutation catalogue, the fuzz campaigns, the kernel against the independent checker, the findings; tables for the
paper's soundness section, `numbers.tex`, a README that states what it does and does not show, and the raw records. It is about the code, not about a run; every record carries the digest of the tree it ran on, and the status
(CLEAN / FAILED / INCOMPLETE) is the first thing its README says.  `python3 src/python/experiments/exp_soundness.py --verify results/soundness` checks it against its manifest.

`results/audit/` is the third: the statement audit (`make audit`, scripts/09_audit.sh): for every problem of the bank the checks of its formal statement against the intended meaning, its provenance, the cautions (where the automatic evidence
is weaker) and where it stands with the ledger of second readers (`docs/statement_reviews.tsv`); the tables for the paper's section on the benchmark, `numbers.tex`, and the review sheet (`review_sheet.md` to read, `review_sheet.csv` to fill in,
`exp_audit.py import-reviews` to put the verdicts in the ledger). Same status line, same stamping, `python3 src/python/experiments/exp_audit.py --verify results/audit`.

`python3 tools/check_numbers.py YOUR.tex --results results/full --results results/soundness --results results/audit` reads a LaTeX source typed by hand and says which number macros, tables and figures it cites that no folder holds, which folder was edited after it
was made or is FAILED / INCOMPLETE / made for other code, and which numbers two folders count differently (`docs/PAPER_PLAN.md`, section 8).

Which code made a folder: every folder is bound to the DIGEST of the code (the files below `src/`, `tests/` and `mutants/` and the manifest of the dataset; `python3 tools/check_numbers.py --tree` prints it), not to the commit. A report records the digest at the
start and at the end of every run (`provenance/runs.csv`) and warns of more than one version of the code in one matrix; the soundness and audit folders record the digest they were made for, and their `dirty` flag means that the CODE had changes that were
not committed (documents, the paper, tools and scripts do not count, so typing the paper never makes the evidence stale). The tools that make these folders are tested by fault injection (`make tool-mutants`, `mutants/README.md`).

The folders are derived from `runs/` and are not tracked by git (see `.gitignore`); copy the folder of the run a paper cites, or commit it by hand (`git add -f results/<profile>`).
