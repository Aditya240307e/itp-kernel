# mutants/ -- mutation testing

A test suite that has never failed has not been shown to test anything. Here every mutant **re-breaks one rule of the system** -- the kernel's evaluator, a tactic, the environment, the recorder -- in an isolated copy
of the tree, and the gates must **fail**. A mutant that no gate catches is a gap in the tests, unless it is listed as a documented equivalent mutant (no input can tell it from the original).

```
python3 mutants/mutate.py --check              every mutant must replace text that occurs EXACTLY ONCE in its file, must change it, and no name may be used twice (seconds)
python3 mutants/mutate.py M83 W1 GR            run the mutants whose names start with one of these prefixes (no prefix: all 1615 -- many hours)
MUT_ONLY=s2_tests.py,s16_quot_tests.py python3 mutants/mutate.py M83      run only these gates against the mutants (by file name)
MUT_FULL=1     run EVERY gate on a mutant instead of stopping at the first that fails      MUT_VERBOSE=1   say which gate caught it      MUT_KEEP=1   keep the mutated copy for inspection
```

* `catalogue.py` -- `MUTANTS`: name -> `(old text, new text[, file])`, `KNOWN_EQUIVALENT` (name -> why no input can tell the mutant from the original), `FAIL_RE` (what a failing gate prints). A mutant without a file name
  mutates the kernel header (`src/c++/hybrid/itp_kernel.hpp`); with one, that source file, named by its **base name** (`recorder.py`, `ring.py`, `itp_bindings.cpp`) -- the layout is searched for it.
* `mutate.py` -- the driver. It first runs every selected gate on the unmutated tree (it must be green, otherwise every mutant would look caught), then for each mutant makes a self-contained copy of `src/`, `tests/`
  and `build/` in the system temporary directory, applies the mutation, rebuilds the kernel for a C++ target (about a minute), runs the gates in registry order and stops at the first that fails. A mutant gets three times
  the unmutated gate's time before it counts as a hang (a hang is a catch). The gates it runs are `GATE_CMDS` of `tests/gate_registry.py` -- the kernel, tactic, environment and bank gates, not the learner's.

The catalogue is organised by series (the name's letters): `M`, `W`, `X`, `FU`, `AR`, ... for the kernel and its evaluator, `H`, `RC`, `Z`, `RQ`, `LA`, `LC` for `ring` / `linarith` / `linear_combination`, `T`, `TL`, `N`, `EP` for the tactic
and authoring layers, `K`, `C`, `I` for the independent checker, `GA`-`GX` for the environment, the action language, the typed term builder and the recorder. Mutant names are a separate namespace from the findings
(`F31` as a mutant is not finding F31 of `docs/FINDINGS.md`).

Status (2026-10-18): 1615 mutants, `--check` clean. The kernel, tactic and environment series were run to completion while they were written; the series for the newest code (the recorder's `GR`, the bank, the
learner) has been defined and anchored but only partly run, because the datasets and the first learners came first -- running them is a completeness task, not a soundness one.

When you change the code a mutant refers to, `--check` will say which anchors went stale; repair the mutant (keep its intent) rather than delete it.
