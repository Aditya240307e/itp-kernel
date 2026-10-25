# mutants/ -- mutation testing

A test suite that has never failed has not been shown to test anything. Here every mutant **re-breaks one rule of the system** -- the kernel's evaluator, a tactic, the environment, the recorder -- in an isolated copy
of the tree, and the gates must **fail**. A mutant that no gate catches is a gap in the tests, unless it is listed as a documented equivalent mutant (no input can tell it from the original).

```
python3 mutants/mutate.py --check              every mutant must replace text that occurs EXACTLY ONCE in its file, must change it, and no name may be used twice (seconds)
python3 mutants/mutate.py --selftest           the machinery works: an unmutated copy of the tree passes a gate and the same copy without legacy/ fails it (seconds; S45 runs it)
python3 mutants/mutate.py M83 W1 GR            run the mutants whose names start with one of these prefixes (no prefix: all 1616 -- many hours)
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

Status (2026-10-25): 1616 mutants, `--check` clean. The newest, `GE107`, re-breaks the clean-up of the names a split brings back (finding F81); run against S40 (the library's replay) it SURVIVED -- the proofs that trigger it replay with or without the lost hypothesis -- so S36 (section G) got a direct test, and against S36 it is CAUGHT. Twenty anchors of environment mutants had gone stale in the `sees()` refactor of the environment and were repaired by hand; `--check` passes, but those 20 mutants were not run again after the repair. The catalogue covers the kernel, the authoring
and library layers, the independent checker, the environment (`gym_*`), the action language, the typed term builder, the search and the recorder. The kernel, tactic and environment series were run to completion while
they were written; the series for the newest code (the recorder's `GR`) has been defined and anchored but only partly run -- running them is a completeness task, not a soundness one. **There are no mutants yet for
the problem-set modules (`bank_*`), the learner (`ml_*`: hindsight, the lemma library, the goals, the provers), the generator (`synth_*`) or the experiment pipeline (`exp_*`)**: their gates (S44, S47-S51) rely on
fixtures with expectations worked out by hand and on tamper tests inside the gates, and have not been shown to bite by mutation. Known open gap: `GR82` (the recorder does not try a recorded `linarith` /
`nlinarith` without its hints) survives S40 -- nothing exercises that fallback; the other mutants run since the reorganisation (`GR52`, kernel `M83`, `E1`) are caught.

When you change the code a mutant refers to, `--check` will say which anchors went stale; repair the mutant (keep its intent) rather than delete it.
