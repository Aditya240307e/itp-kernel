# ITP

**An interactive theorem prover with a small trusted kernel.**

ITP is a dependently typed proof assistant in which a 7,546-line C++23 kernel decides what counts as a proof. Everything above it -- tactics, decision procedures, libraries, a bank of olympiad problems -- is untrusted Python that can only *propose* terms. A second type checker, written separately from the typing rules, re-checks whole libraries and single proofs. The code base is tested adversarially: 54 gates (test programs), a catalogue of 1726 mutants (deliberate one-rule breakages of the source that the gates must catch) and a tracker of every defect found so far.

```python
# from examples/01_prove_with_tactics.py -- `ring` assembles a proof term; the kernel checks it
w.prove('ex_binomial', forall([Nat, Nat], lambda a, b: Eq((a + b) ** 2, a ** 2 + 2 * a * b + b ** 2)),
        lambda p: (p.intro('a', 'b'), p.ring()))
```

* **Kernel** ([`src/c++/hybrid`](src/c++/hybrid/README.md)) -- dependent types; inductive families with parameters and indices; structural recursion; a `Prop`/`Type` world with definitional proof irrelevance; quotients; a deterministic, fail-closed work budget. Called from Python through [nanobind](https://github.com/wjakob/nanobind).
* **Tactics and decision procedures** ([`authoring`](src/python/authoring/README.md)) -- `intro`, `apply`, `rw`, `cases`, `induct`, `have`, `obtain`, `calc`, `by_contra`, `interval_cases`, and `ring` (unary and binary naturals, integers), `linarith`, `nlinarith`, `linear_combination`. Each one ends in an ordinary proof term that the kernel type-checks.
* **Independent checker** ([`checker`](src/python/checker/README.md)) -- a second implementation of the typing rules (named terms, capture-avoiding substitution, weak-head normalisation) that shares no code with the kernel, and `audit`, which lists the axioms a theorem rests on.
* **Libraries** ([`library`](src/python/library/README.md)) -- Peano arithmetic, binary naturals, the integers as a quotient of pairs of naturals, ring laws, divisibility, division with remainder, Bezout, gcd, primes, Euclid's lemma; all proved by tactic scripts. The naturals, binary naturals and integers use no axiom; the number-theory library adds one, excluded middle.
* **Problem bank** ([`problems`](src/python/problems/README.md)) -- **268 formal problems** (levels 1-5: IMO, ISL, AIME, AMC-style and classical results). Every statement is checked against an independent description of what the problem says. **187 have a kernel-checked reference proof**; the other 81 are statement-only.
* **Proof environment** ([`environment`](src/python/environment/README.md)) -- goals as data, menus of legal actions, actions that *say* their term arguments through a typed term builder, atomic probes with rollback, a recorder that turns authored proofs into action sequences, and a Gymnasium wrapper (`ITPBank-v0`).

ITP is used as a Python library: definitions and tactic proofs are written in Python; there is no separate surface language and no command-line prover. The mathematics it covers is elementary algebra and number theory over the naturals and the integers (see [Limitations](#limitations)).

[Quick start](#quick-start) · [Proving something](#proving-something) · [How it is built](#how-it-is-built) · [Why believe it](#why-believe-it) · [The Python environment](#the-python-environment) · [Limitations](#limitations) · [Repository layout](#repository-layout)

## Quick start

You need Linux with **g++ 13 or newer** (the kernel uses C++23 `std::expected`), `make`, and **Python 3.13** (the version it is developed and tested with). From the root of the repository:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "numpy>=2.5" "scipy>=1.18" "nanobind>=3.1"     # all that the prover, the libraries and the gates need
make kernel                                                 # compiles build/itp_env_lnl*.so and the frozen oracle kernel: about 75 s
python3 examples/01_prove_with_tactics.py                   # a few seconds; ends with EXAMPLE OK
make gates-quick                                            # six kernel gates and the example again: under 10 s
```

`pip install -r requirements.txt` also installs PyTorch, Gymnasium and Matplotlib, which only the optional learner and experiment code uses ([below](#optional-components)); example 06 needs just `gymnasium` and says so when it is missing. The Python scripts find the repository from their own location, so they run from any directory. macOS on Apple silicon has not been run yet: it needs Xcode 16 or newer, and `python3 tools/preflight.py --quick` checks the toolchain of a new machine.

## Proving something

Start Python in the repository root with `PYTHONPATH=src/python python3`. A **world** is a kernel plus a name table, and the libraries build worlds from scratch each time (the natural-number library takes a fraction of a second). A **proof** is a list of goals that tactics refine; `print(p)` shows them.

```python
import itp_paths, nat_lib              # itp_paths makes every module importable by its plain name
from tactics import Proof

w = nat_lib.build_nat()                # 129 globals: Nat, Eq, And, Or, Ex, Le, add, mul, pow ... and their lemmas
Nat, Zero, Succ, Eq, forall = w['Nat'], w['Zero'], w['Succ'], w.Eq, w.forall_

# a function by structural recursion; the kernel checks that the recursion is structural
w.recursive('twice', w.arrow(Nat, Nat),
            lambda f: w.fun([Nat], lambda n: w.match(n, lambda _: Nat, [Zero, lambda k: Succ(Succ(f(k)))])))
twice = w['twice']

# a theorem about it, one tactic at a time
p = Proof(w, forall([Nat], lambda n: Eq(twice(n), n + n)), 'twice_eq_add_self')
p.intro('n')
p.induct('n', names=['k', 'ih'])
print(p)
#   goal 0:
#   n : Nat
#   |- Zero = Zero
#
#   goal 1:
#   n : Nat
#   k : Nat
#   ih : twice k = k + k
#   |- Succ (Succ (twice k)) = Succ (k + Succ k)
p.refl()                               # the base case holds by computation
p.rw('ih')                             # rewrite with the induction hypothesis
print(p)
#   n : Nat
#   k : Nat
#   ih : twice k = k + k
#   |- Succ (Succ (k + k)) = Succ (k + Succ k)
p.rw('add_succ')                       # n + Succ m = Succ (n + m) turns the goal into x = x, which rw closes
print(p.done)                          # True
```

`w.prove(name, statement, script)` runs a whole script, extracts the finished term and hands it to the kernel's checked entry point; only then does the theorem exist in the world.

```python
def by_induction(p):
    p.intro('n')
    p.induct('n', names=['k', 'ih'])
    p.refl()
    p.rw('ih')
    p.rw('add_succ')

w.prove('twice_eq_add_self', forall([Nat], lambda n: Eq(twice(n), n + n)), by_induction)
```

A tactic that cannot do its job says why, and nothing is admitted. Here a false identity is refused, with both normal forms:

```python
w.prove('bad', forall([Nat, Nat], lambda a, b: Eq((a + b) ** 2, a ** 2 + b ** 2)),
        lambda q: (q.intro('a', 'b'), q.ring()))
# author.AuthorError: ring failed: the two sides have different normal forms
#   left:  a^2 + 2*a*b + b^2
#   right: a^2 + b^2
#   (and the goal that was being proved)
```

Finally, the second checker and the auditor:

```python
import recheck, audit
recheck.recheck_world(w)                                         # every global, re-checked by the independent checker
# (131, [])                                                      # 131 globals checked, no problems
audit.dependencies_of_global(w.k, w.gid['twice_eq_add_self'])    # what the theorem rests on: (axioms, quotient primitives)
# (set(), set())
```

[`examples/`](examples/README.md) has eight programs that continue from here: integer arithmetic with `ring`, `linarith` and `nlinarith`; an IMO problem of the bank with its proof replayed and re-checked; the proof environment. [`src/python/authoring`](src/python/authoring/README.md) lists every tactic.

## How it is built

```
  untrusted     library scripts (Nat, N, Z, number theory)      problem bank      proof environment
  Python        tactics: intro apply rw cases induct have ...   ring  linarith  nlinarith
                authoring layer: World, Term, Proof
                                      |
                                      |  terms, through the nanobind module itp_env_lnl
                                      v
  trusted       C++23 kernel -- src/c++/hybrid/itp_kernel.hpp (7,546 lines) + itp_bindings.cpp (788 lines)
                type checker . admission of inductives, recursion, quotients . conversion by NbE . work budget
                                      |
                                      |  exported terms and declarations (Kernel.dump_raw, Kernel.global_info)
                                      v
  cross-check   independent checker -- src/python/checker/indep_checker.py (735 lines)
                named terms . weak-head normalisation . its own universe, guard and match rules
```

**The kernel** is one header. Terms have eleven node kinds: variable, application, lambda, Pi, universe `Type k`, constant, `match`, metavariable, `let`, a local fixpoint, and a placeholder for a universe parameter. Bound variables are de Bruijn *indices* and the hypotheses of the context are *levels*, so a closed term has one canonical form and a hypothesis keeps its identity while the context grows. Terms are hash-consed into a DAG, so syntactic equality is an integer comparison and sharing is never lost. Evaluation is normalisation by evaluation with closures over persistent environments; conversion tries a quick syntactic check first and keeps sharing tables (`ITP_VERIFY_MEMO=1` makes the kernel re-decide every answer it took from a table). Inductive families (parameters and indices) are admitted after a strict-positivity and a universe check; recursion is `Fix` with a structural-decrease guard; the `Prop`/`Type` world (kernel option `prop_elim`) makes `Type 0` impredicative with definitional proof irrelevance, decided by the type checker itself; a quotient is five ordinary constants, whose types the kernel builds, and one reduction rule; `funext` is then a theorem. The tactics the Python layer calls (`intro`, `apply`, `cases`, `induct`, `rewrite`, `exact`, `change`) work on metavariables with Miller-pattern unification and roll back on any failure; the terms they produce are re-checked when the theorem is declared. Every guarded function spends one unit of a deterministic **work budget** and checks the stack: a pathological input ends in `FuelExhausted` or `StackExhausted` -- a failed check, never an accepted one. [More](src/c++/hybrid/README.md).

**The authoring layer** (untrusted) builds terms with higher-order abstract syntax -- `w.pi(A, lambda x: ...)`, Python operators on `Term`s that mean whatever the world's notation says -- and drives the kernel's checked entry points, so a bug there can make a library fail to build and cannot make a false theorem appear. `ring` normalises both sides to a canonical sum of monomials and emits the chain of `ring_*` lemma applications that justifies every step; `linarith` finds non-negative multipliers with an exact-arithmetic simplex and emits the certificate as a proof term; `nlinarith` first adds the squares and pairwise products of the hypotheses. [More](src/python/authoring/README.md).

**The libraries** are Python scripts that declare types and prove lemmas through the authoring layer, in one chain: `lnl_base` (logic, equality, `Nat`), `nat_lib`, `bin_lib` (binary `Pos` and `N`, with a bridge to `Nat`), `int_lib` (`Z` as a quotient of `N x N`), `nt_lib` (number theory), `ext_lib` and `imo_lib`. Every lemma is proved and re-checked each time a world is built; nothing is stored. [More](src/python/library/README.md).

**The independent checker** re-implements the typing rules on named terms with capture-avoiding substitution and weak-head normalisation. It reads the kernel's exported terms and declarations and checks that every global's value has its type, that the structural-recursion guard holds, that each `match` respects its motive and branch types, and that the five quotient constants have exactly the types they should. It is an alarm rather than a judge: independent in code, not in authorship, and it implements the same specification as the kernel. [More](src/python/checker/README.md).

## Why believe it

ITP does not claim to be proved correct; it claims to be hard to fool, and ships the evidence.

* **A small trusted base.** What must be believed for "this proof is correct": the C++ kernel (about 8,300 lines with its bindings), the compiler and the machine. Tactics, decision procedures, libraries, the problem bank and the environment sit outside the boundary by construction; their mistakes cost time, not soundness.
* **Differential testing.** The previous generation of the kernel is kept byte-identical (`src/c++/reference`, checksums in `SHA256SUMS`). The gates run the same random scripts through both kernels in lockstep and demand the same verdicts, except where a documented finding says the new kernel is deliberately stricter.
* **Oracles and canaries.** A self-consistency oracle runs random tactic scripts and re-checks every state and every finished proof; its strict fragment must be perfectly clean. Canaries make random attempts at false statements (`False`, `0 = 1`, `forall P, P`, ...) and demand that none is verified.
* **Mutation testing.** `mutants/` holds 1726 mutants -- each re-breaks one rule of the kernel, a tactic, the checker or the environment in an isolated copy of the tree -- and the gates must fail on each. A mutant that no gate catches is a gap in the tests unless it is listed as an equivalent mutant (22 are); [`mutants/README.md`](mutants/README.md) says what has been run and what survives.
* **Second checker, sanitizers, fuzzing.** Several gates re-check whole libraries with the independent checker; `make sanitize` runs the kernel and bank gates under AddressSanitizer and UBSan; a fuzzer drives the proof environment with random and malformed actions.
* **A findings log.** [`docs/FINDINGS.md`](docs/FINDINGS.md) records F1-F91, each with its status and the test that fails without the fix.
* **Fail-closed behaviour.** A spent work budget, an exhausted stack, a nesting bound and an unknown option are all errors, and a spent budget is never read as "no".

What this is not: a verification of the kernel. Soundness is *tested*, not proved, and the documents say "tested" wherever they mean it. The two checkers implement the same rules, so a flaw in the rules themselves would not show up by comparing them. The statement audit (`make audit`) compares formal statements with their intended meaning on a box of small values -- evidence of faithfulness, not a proof of it.

```bash
make gates-quick       # six kernel gates and an example: under 10 s
make gates             # the 49 default gates: kernel, tactics, libraries, environment, bank, contest proofs (about 45 minutes on two cores)
make gates-all         # all 54 (about an hour); J=8 make gates runs eight at a time
make mutants-check     # every mutant of the catalogue still applies to the code (seconds)
python3 mutants/mutate.py M83 W1     # run the mutants whose names start with these prefixes (no prefix: all of them, many hours)
make dataset-verify    # replay every recorded proof of data/dataset_v2 in a fresh engine (about 25 minutes, 3 GB)
make soundness         # the soundness evidence in one stamped folder, results/soundness (hours)
make audit             # the statement audit of the whole bank, results/audit (minutes)
make sanitize          # kernel and bank gates under ASan + UBSan (long)
```

[`tests/README.md`](tests/README.md) lists every gate with its measured time and what it protects.

## The Python environment

| | |
|---|---|
| Python | 3.13 is the tested version. Modules are imported by plain name (`import gym_env`); the names are unique across the tree, which gate S45 checks |
| Core packages | `numpy`, `scipy`, `nanobind`: the kernel build, tactics, libraries, problem bank, environment and the gates |
| Optional packages | `torch`, `gymnasium`, `matplotlib`: only the Gymnasium wrapper, the learner and the experiment pipeline |
| Import path | every script starts with one line that puts `src/python` on `sys.path` and imports `itp_paths`; a session does the same with `PYTHONPATH=src/python` and `import itp_paths`. The root of the repository is derived from the location of that file, never from the working directory |
| Build directory | `build/`; set `ITP_BUILD_DIR` to use another. The two kernels (`itp_env_lnl`, `itp_env_ref`) are separate extension modules that can be imported into one process |
| Compiler | `CXX=g++-13 make kernel` picks a compiler; `python3 src/python/build_kernel.py --help` lists the build options |
| Memory | the libraries are cheap; building the whole problem bank takes about 80 seconds and 1.3 GB; the bank gate S50 peaks near 4.6 GB, and S19, S25, S30, S31, S34, S37, S40 and S50 limit the address space, so they need Linux |
| Make | `make help` lists the targets; `PY=python3.13` and `J=n` (gates run at once) are the variables |

[`src/python/README.md`](src/python/README.md) explains the layout of the Python code and the import mechanism.

## Limitations

* The type theory has no universe polymorphism or cumulativity, no structures with eta, no mutual or nested inductive types (a nested occurrence is rejected) and no well-founded recursion. Conversion nesting is bounded (256) and fails closed.
* `Nat` is unary (a numeral is a chain of `Succ`); binary `N` and the integers `Z` are libraries. There are no rationals or real numbers, no `omega`, `norm_num` or `simp` set, and no Mathlib. [`docs/ROADMAP.md`](docs/ROADMAP.md) lists what is missing.
* The independent checker rejects terms that contain `let` or metavariables (finished proofs contain neither).
* The proof language is Python: there is no parser for a standalone syntax, no editor integration and no command-line prover.
* Of the 60 IMO, ISL and APMO statements in the bank, 12 have proofs. A statement-only problem is a formal statement with brute-force evidence that it is true, not a theorem.
* Tested on Linux with g++ 13. The macOS port has been syntax-checked against libc++ headers but not built or run on a Mac.

## Repository layout

| | |
|---|---|
| [`src/c++/hybrid/`](src/c++/hybrid/README.md) | the kernel: `itp_kernel.hpp`, `itp_bindings.cpp` |
| [`src/c++/reference/`](src/c++/reference/README.md) | the frozen previous-generation kernel, the oracle of the lockstep gates |
| [`src/python/`](src/python/README.md) | `itp_paths.py` (where everything lives), `build_kernel.py`, and the modules by role: |
| [`  authoring/`](src/python/authoring/README.md) | `World`, `Term`, `Proof`, tactics, `ring`, `linarith`, `linear_combination` |
| [`  checker/`](src/python/checker/README.md) | independent checker, `recheck`, `audit` |
| [`  library/`](src/python/library/README.md) | naturals, binary naturals, integers, ring laws, number theory, olympiad lemmas |
| [`  environment/`](src/python/environment/README.md) | proof environment: engine, action language, term builder, recorder |
| [`  problems/`](src/python/problems/README.md) | the problem bank, faithfulness checks, duplicate screen, splits, dataset builder |
| [`tests/`](tests/README.md) | the gates by role, shared modules, fuzzers, benchmarks, golden files |
| [`mutants/`](mutants/README.md) | the mutation catalogue and its driver |
| [`examples/`](examples/README.md) | eight runnable programs and small reproductions of fixed defects |
| [`docs/`](docs/README.md) | design notes, the findings log, bank and action-language references |
| [`data/`](data/README.md) | the problem bank exported as JSON, with checksums |
| [`preprint/`](preprint/README.md) | the technical report (`itp_preprint.pdf`), its LaTeX sources, and the scripts and raw measurements behind every number in it |
| [`tools/`](tools/README.md), [`scripts/`](scripts/README.md) | environment check, release archive, sanitizer run; shell drivers |

## Optional components

Not needed to use the prover, and listed so that nothing in the tree is a surprise: `src/python/learner` (PyTorch code on top of the environment: policy networks, guided search, and the Gymnasium wrapper `ITPBank-v0` in `bank_env.py`), `src/python/synth` (a generator of synthetic theorems with kernel-checked proofs, [`docs/GENERATOR.md`](docs/GENERATOR.md)), `src/python/experiments` with `scripts/` (a resumable pipeline of runs and reports), `paper/` (a manuscript skeleton) and `legacy/` (the previous generation's modules, which a few differential gates still import).

## Documentation

[`docs/README.md`](docs/README.md) is the index. Start with [`docs/ACTIONS.md`](docs/ACTIONS.md) (the action language), [`docs/BANK.md`](docs/BANK.md) (what makes a formal statement *the* problem, where each text comes from, how duplicates and splits are handled), [`docs/KERNEL_DESIGN.md`](docs/KERNEL_DESIGN.md) (the kernel's design log) and [`docs/FINDINGS.md`](docs/FINDINGS.md).

## Licence

The code, the data and the documentation are released under the [Apache License 2.0](LICENSE). The technical report in [`preprint/`](preprint/README.md) is also archived on Zenodo ([10.5281/zenodo.24047732](https://doi.org/10.5281/zenodo.24047732)) under CC BY 4.0.
