# mutants/ -- mutation testing

A test suite that has never failed has not been shown to test anything. Here every mutant **re-breaks one rule of the system** -- the kernel's evaluator, a tactic, the environment, the recorder -- in an isolated copy
of the tree, and the gates must **fail**. A mutant that no gate catches is a gap in the tests, unless it is listed as a documented equivalent mutant (no input can tell it from the original).

```
python3 mutants/mutate.py --check              every mutant must replace text that occurs EXACTLY ONCE in its file, must change it, and no name may be used twice (seconds)
python3 mutants/mutate.py --selftest           the machinery works: an unmutated copy of the tree passes a gate and the same copy without legacy/ fails it (seconds; S45 runs it)
python3 mutants/mutate.py M83 W1 GR            run the mutants whose names start with one of these prefixes (no prefix: all 1647 -- many hours)
MUT_ONLY=s2_tests.py,s16_quot_tests.py python3 mutants/mutate.py M83      run only these gates against the mutants (by file name)
MUT_FULL=1     run EVERY gate on a mutant instead of stopping at the first that fails      MUT_VERBOSE=1   say which gate caught it      MUT_KEEP=1   keep the mutated copy for inspection
```

* `catalogue.py` -- `MUTANTS`: name -> `(old text, new text[, file])`, `KNOWN_EQUIVALENT` (name -> why no input can tell the mutant from the original), `FAIL_RE` (what a failing gate prints). A mutant without a file name
  mutates the kernel header (`src/c++/hybrid/itp_kernel.hpp`); with one, that source file, named by its **base name** (`recorder.py`, `ring.py`, `itp_bindings.cpp`) -- the layout is searched for it.
* `mutate.py` -- the driver. It first runs every selected gate on an UNMUTATED COPY of the tree (it must be green, otherwise every mutant would look caught), then for each mutant makes a self-contained copy
  (`itp_paths.COPIED_DIRS`: `src/`, `tests/`, `legacy/`, `data/`, and a `build/`) in the system temporary directory (`$TMPDIR`), applies the mutation, rebuilds the kernel for a C++ target (about a minute),
  runs the gates in registry order and stops at the first that fails. The baseline must run in the same kind of copy as the mutants: once, the copy lacked `legacy/`, every gate crashed on an import, every mutant
  was "caught", and the original tree -- where the baseline ran -- showed nothing (finding F71). `MUT_VERBOSE=1` prints the line a gate failed with, so a catch can be read, not just counted; a Python mutant
  "caught" in a fraction of the gate's normal run time deserves a look. A mutant gets three times
  the unmutated gate's time before it counts as a hang (a hang is a catch). The gates it runs are `GATE_CMDS` of `tests/gate_registry.py` -- the kernel, tactic, environment and bank gates, not the learner's.

The catalogue is organised by series (the name's letters): `M`, `W`, `X`, `FU`, `AR`, ... for the kernel and its evaluator, `H`, `RC`, `Z`, `RQ`, `LA`, `LC` for `ring` / `linarith` / `linear_combination`, `T`, `TL`, `N`, `EP` for the tactic
and authoring layers, `K`, `C`, `I` for the independent checker, `GA`-`GX` for the environment, the action language, the typed term builder and the recorder. Mutant names are a separate namespace from the findings
(`F31` as a mutant is not finding F31 of `docs/FINDINGS.md`).

Status (2026-11-01): 1647 mutants, `--check` clean. The newest 31 pin the menu-time work budget (finding F87; `GA36`-`GA41`, `GE108`-`GE132`): the normal form of a fact or a goal runs under the engine's budget and not the library's, a spent budget is neither an error nor an empty menu, the caller's budget is given back, the cache of normal forms is keyed by the budget it was made under. Eight older ones (`GA15`, `GA16`, `GE100`, `GE101`, `GQ126`, `GQ127`, `GQ141`, `GQ142`) were re-anchored to the changed code. Each of the 39 was run against the gate that pins it and is CAUGHT. Three of them -- `GA16`, `GQ126`, `GQ141` (a fact that is a hypothesis up to normalisation is proposed, listed or injected) -- were found to SURVIVE S40 on the committed tree as well: they were test gaps older than F87, and section E8 of S40 closes them. Before these, `GE107` re-broke the clean-up of the names a split brings back (finding F81); run against S40 (the library's replay) it SURVIVED -- the proofs that trigger it replay with or without the lost hypothesis -- so S36 (section G) got a direct test, and against S36 it is CAUGHT. Twenty anchors of environment mutants had gone stale in the `sees()` refactor of the environment and were repaired by hand; `--check` passes, but those 20 mutants were not run again after the repair. The catalogue covers the kernel, the authoring
and library layers, the independent checker, the environment (`gym_*`), the action language, the typed term builder, the search and the recorder. The kernel, tactic and environment series were run to completion while
they were written; the series for the newest code (the recorder's `GR`) has been defined and anchored but only partly run -- running them is a completeness task, not a soundness one. **There are no kernel-style mutants for
the problem-set modules (`bank_*`), the learner (`ml_*`: hindsight, the lemma library, the goals, the provers), the generator (`synth_*`) or most of the experiment pipeline (`exp_*`)**: their gates (S44, S47-S51) rely on
fixtures with expectations worked out by hand and on tamper tests inside the gates, and have not been shown to bite by mutation. The tools that produce the paper's EVIDENCE are the exception: they have a catalogue of faults of their own
(`tool_catalogue.py`, below). Known open gap: `GR82` (the recorder does not try a recorded `linarith` /
`nlinarith` without its hints) survives S40 -- nothing exercises that fallback; the other mutants run since the reorganisation (`GR52`, kernel `M83`, `E1`) are caught.


## The faults of the tools that make the evidence (`tool_catalogue.py`, `check_tools.py`)

The kernel's mutants test the kernel's gates. The paper also cites numbers that come from TOOLS -- the soundness folder (`exp_soundness.py`), the statement audit (`exp_audit.py`), the number checker
(`tools/check_numbers.py`), the record of which code a result was made with (`exp_repo`, `ml_meta`, `exp_run`, `exp_report`, `exp_bundle`, the digest of `mutate.py`) -- and a test written together with the code it checks tends to check
what the code does rather than what it must do (finding F88: of the first 172 faults, 21 passed the tests written for the tool, among them stages that reused a result made for other code). So each of these tools has a list of
single-line faults, and the section of a gate that was written for the tool must FAIL for every one of them.

```
python3 mutants/check_tools.py --check                  every fault replaces text that occurs EXACTLY ONCE, changes it, leaves a file that compiles; no name twice in a tool; every gate exists (seconds; S45 O pins it)
python3 mutants/check_tools.py --list [TOOL ...]        the tools, their gate sections and their faults
python3 mutants/check_tools.py [TOOL ...] [--only TEXT ...] [--jobs N] [--json FILE] [--redo]       inject each fault into its own copy of the tree and run the gate section of its tool
make tool-mutants J=8                                   the same, for every tool (252 faults, 47 minutes on two cores)
```

| tool | what is tested | gate section |
|---|---|---|
| `soundness` | the soundness folder: what it reads of the gates, mutants, fuzz and differential results, when a stage is reused | S51 O |
| `numbers` | `tools/check_numbers.py` | S45 N |
| `audit` | the statement audit | S51 P (the small dataset: built once, or `S51_DATA`) |
| `identity` | `dirty`, the digest a run records at its start and end, the warnings of the report, `runs.csv` | S51 Q |
| `digest` | what the digest of the code covers and what changes it | S45 M |
| `runner` | `check_tools.py` itself: when a fault counts as caught, when a baseline is green, which results are reused | S45 P |

* Each fault runs in a copy of a SNAPSHOT of the tree taken when the run starts (what `mutate.py` copies, plus `docs/`, `tools/`, `scripts/`, `paper/`; no `.git`, no `dist/`): editing the tree while a long run goes on cannot reach the faults still to come,
  and a gate that only passes inside a git checkout is found at once (the baseline runs in a copy without one).
* The unmutated copy must be GREEN on the gate section and must have run at least one check (a section letter that selects nothing is green too); otherwise the run stops with exit status 2.
* A fault is caught when the gate exits non-zero or prints a failure. `kind` says how: `assertion` (a check failed -- what a test is for), `crash` (the section raised instead of checking: a weaker catch, and a reason to make the
  check report instead of raise) or `timeout`. The first line the gate complained of is printed: if the same line appears for every fault the faults are being caught for a reason that has nothing to do with them (this happened
  once, see below).
* Exit status: 0 all caught; 1 a fault survived (write the check that fails for it -- or, if no input can tell the fault from the original, remove the line it re-breaks, as was done once); 2 the catalogue is stale or the baseline is red.
* `--json FILE` appends one record per fault (tool, name, digest of the fault, digest of the TREE, status, kind, first failure, seconds) and skips the faults it already holds for this fault and this tree.
* The faults of the runner are checked by S45 **P**, which uses doctored catalogues only; the real catalogue is pinned by S45 **O**. A fault injected into `check_tools.py` makes the real catalogue stale in the copy (its own text is gone),
  so a gate that looked at the real catalogue would "catch" every runner fault for that reason alone -- the first version of the runner's self-test did exactly that, and the same-line symptom above is how it showed.

To add a fault: append `(name, file, old text, new text)` to the tool's list in `tool_catalogue.py` (the old text must occur once; keep it to the line or two that carries the rule), run `--check`, then run the tool. When you change a tool
so that the text a fault replaces moves, S45 O fails until the fault is repaired (keep its intent).

When you change the code a mutant refers to, `--check` will say which anchors went stale; repair the mutant (keep its intent) rather than delete it.
