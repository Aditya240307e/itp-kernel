# data -- the problem bank as files

[`dataset_v2/`](dataset_v2/DATASET_CARD.md) is the [problem bank](../src/python/problems/README.md) written out as plain JSON, so that the problems and their proofs can be read, loaded or processed without building anything: **268 formal problems**, **187 with a kernel-checked reference proof** written as a sequence of actions of the [proof environment](../src/python/environment/README.md), the split of the problems into train, dev and test, the proofs of the 453 lemmas of the base library, and the theorems that the reference proofs contain.

Nothing here is needed to use the prover. The same problems are built from source by `bank_all.load()`; the files are the form a program in another language, or a data pipeline, can read directly. Every file is **deterministic**: two builds of the same sources give the same bytes, and `manifest.json` holds a digest of each.

## Files

| file | size | what is in it |
|---|---|---|
| `bank.json` | 2.7 MB | the 268 problems. Each has its text, source, year, topics, level, kind (`prove`, `answer`, `exists`), provenance (`origin`, `verified`), the formal statement (printed, as a syntax tree, and as a code of the typed term builder), the premises of both regimes, the digests, its split and family, and the **demonstration**: a kernel-checked sequence of portable actions for the `prefix` regime and, for the 8 proofs that are self-contained, the `strict` regime. `globals` lists the 741 names of the world in id order, `world_digest` identifies the world the ids refer to |
| `splits.json` | 16 KB | the split policy (salt, fractions 70/10/20, family key of each problem) and the resulting splits of the problems (190 / 20 / 58) and of the library's lemmas |
| `library.json` | 0.3 MB | the proofs of the 453 lemmas of the base library as action sequences, split by lemma name; the 6 lemmas that the action language cannot yet say are listed under `unreplayed` with the reason |
| `annotations.json` | 0.9 MB | what each action of each reference proof does: its direction (`CLOSE`, `BACKWARD`, `FORWARD`, `SPLIT`, `CLAIM`, `INTRO`), the depth and width of the proof tree, the claims and the lemmas used |
| `subproblems.json` | 13.5 MB | the theorems *inside* the reference proofs: every closed subtree (5649) and every derived fact (702) is a theorem of its own, with a portable proof that is replayed and checked by the kernel |
| `DATASET_CARD.md` | 15 KB | a human-readable description: composition by level, source, origin and topic; the splits; the regimes; statistics of the proofs |
| `manifest.json` | | the first 16 hex digits of the SHA-256 of each of the files above |

A *portable* action has no kernel handle and no index into a menu. `{"kind": "APPLY", "source": ["c", 477, 0]}` applies the global with id 477 (`Z.exists_prime_dvd`, entry 477 of `globals`; `["v", level, 0]` would name a hypothesis instead), and `{"kind": "USE", "codes": [[["n", -2]]]}` supplies the numeral -2, said in the tokens of the typed term builder. The format is described in [`docs/ACTIONS.md`](../docs/ACTIONS.md).

## Reading it

```python
import json
bank = json.load(open('data/dataset_v2/bank.json'))
p = next(p for p in bank['problems'] if p['name'] == 'imo1959_p1')

p['text']                          # 'Prove that the fraction (21n+4)/(14n+3) is irreducible for every natural number n.'
p['split'], p['level'], p['kind']  # ('train', 2, 'prove')
p['source'], p['origin'], p['verified']        # ('IMO 1959 P1', 'contest', 'answer-checked')
p['demonstration']['length']       # 4
p['demonstration']['steps'][0]     # {'kind': 'APPLY', 'source': ['c', 477, 0]}
```

[`examples/07_read_dataset.py`](../examples/07_read_dataset.py) goes through every file, checks the manifest, and needs no kernel (`python3 examples/07_read_dataset.py`, about a second).

## Checking it

* **The digests**: `manifest.json` against the files (what example 07 does).
* **The proofs**: `make dataset-verify` (or `python3 src/python/problems/dataset.py --verify data/dataset_v2`) reads the files back, builds the world of the bank from source, replays every demonstration and every sub-theorem in a fresh engine, and has the kernel check the term each one builds (about 25 minutes and 3 GB). A file that no longer matches the world it was made from fails here.
* **Rebuilding**: `python3 src/python/problems/dataset.py OUT_DIR` writes all of it again (several minutes).

The statements themselves are checked against the problems' words by the [faithfulness checks of the bank](../src/python/problems/README.md#does-the-statement-mean-what-the-problem-says); the proofs are checked by the kernel, and again by the [independent checker](../src/python/checker/README.md) where they are not too large.

## Other contents

`results/v1/` holds a few output files of early runs of the optional learning pipeline on *dataset v1* (an earlier, 82-problem version of the bank, which `dataset_v2` replaces). They are kept as a historical record, are not about the prover, and are not measurements of `dataset_v2`; see [`results/README.md`](results/README.md). Files named `steps_*.pkl` that the optional learner writes here as a cache are ignored by git.
