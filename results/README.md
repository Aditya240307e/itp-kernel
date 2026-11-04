# results/ -- where the evidence folders are written

Three `make` targets write a **stamped folder** here. The folders are derived, so git ignores them (see `.gitignore`); this file is the only tracked one. To keep a folder in the repository, add it by hand (`git add -f results/soundness`).

| folder | made by | what it holds |
|---|---|---|
| `results/soundness/` | `make soundness` ([`scripts/08_soundness.sh`](../scripts/README.md)) | the evidence about the **kernel**: every gate, every mutant of the catalogue, the falsification campaigns, the kernel against the independent checker, the inventory of the trusted code, the classification of the findings. It is about the code, not about a run |
| `results/audit/` | `make audit` ([`scripts/09_audit.sh`](../scripts/README.md)) | the **statement audit** of the problem bank: for every problem the checks of its formal statement against the intended meaning, its provenance, the cautions (where the automatic evidence is weaker), where it stands with the ledger of second readers ([`docs/statement_reviews.tsv`](../docs/statement_reviews.tsv)), and a review sheet (`review_sheet.md` to read, `review_sheet.csv` to fill in) |
| `results/<profile>/` | `make report` (`scripts/07_report.sh`), optional | a copy of the report of the learning experiments of a profile (`smoke`, `small`, `full`): tables, figures, the outcome of every attempt on every problem, statistics, provenance |

Every folder has the same shape: the tables, a readable `report.md`, `numbers.json` and `numbers.tex` (every number by name), the raw records, a `README.md` that says what the folder shows and what it does not, and a `MANIFEST.sha256` (`sha256sum -c MANIFEST.sha256` inside the folder checks it). The first thing the README states is the **status**: `CLEAN`, `FAILED` (a claim is contradicted) or `INCOMPLETE` (evidence is missing or stale).

## Which code made a folder

A record is bound to the **digest of the code** (the files below `src/`, `tests/` and `mutants/` and the manifest of the dataset; `python3 tools/check_numbers.py --tree` prints it), not to the commit, and counts only for the code it ran on: a record of other code is listed as stale and is run again. The `dirty` flag of a folder means that the code had changes that were not committed; documents, tools and scripts do not count, so editing a document never makes the evidence stale. The tools that make these folders are themselves tested by injecting faults into them (`make tool-mutants`, see [`mutants/README.md`](../mutants/README.md)).

## Checking and using a folder

```bash
python3 src/python/experiments/exp_soundness.py --verify results/soundness    # a file changed after the folder was made?
python3 src/python/experiments/exp_audit.py --verify results/audit
python3 src/python/experiments/exp_bundle.py --verify results/<profile>

python3 src/python/experiments/exp_audit.py import-reviews review_sheet.csv --reviewer NAME    # put a second reader's verdicts into the ledger
python3 tools/check_numbers.py YOUR.tex --results results/soundness --results results/audit   # does a LaTeX source cite only what the folders hold?
```
