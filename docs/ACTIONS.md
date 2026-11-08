# The action language

How an agent with **no text interface** proves things on the hybrid kernel: goals are data, a step is an
*action* (a kind and its arguments), arguments that are terms are *said* one typed choice at a time, every action
is probed before it is taken, and the finished proof is checked by the kernel (and, in the gates, by an independent
checker). Nothing in this layer is trusted: a bug here can waste work or lose a proof, never produce a false theorem.

Files: `gym_env.py` (engine, goals, menu, probes, first language), `gym_acts.py` (second language, portable
form), `gym_build.py` (typed term builder), `gym_terms.py` (term pool), `gym_search.py` (blind search baseline),
`recorder.py` (authored proofs → action sequences), gate `s40_actions_tests.py` (587 checks) and `s41_bank_tests.py`.

## 1. The model

* A **goal** is `GoalData`: a target and the types/names of the hypotheses (kernel handles, nothing else). Two
  searches that reach the same goal by different routes reach the same object (`GoalData.key`).
* A **step** is an `Action(kind, arg, args, codes, hyps)` applied to the **first** open goal.
* A **probe** (`Engine.probe`) runs an action on a copy of the goal and returns the goals it leaves, then rewinds the
  kernel completely (nodes made during a probe are reclaimed). A refused action leaves the kernel exactly as it was.
  `Episode.step` probes first: an action that does not apply raises `ActionError` and changes nothing.
* After every action the goals it left are **settled** (identically in probes and in replay): a goal closed by a
  hypothesis or by reflexivity is closed, and the quantifiers/implications at the head of a goal are introduced
  (`auto_intro`), so the agent never spends a step on them.
* **Premises**: `Engine.premises` (or `Episode(…, premises=n)`) is the number of globals (by id) the actions may use.
  A theorem is attacked with the library that precedes it. Episodes carry their own premises, so episodes with
  different premises can be interleaved on one engine.
* **Budgets**: every probe runs under a work budget (kernel fuel, per kind of action); running out of work is
  reported as such (`ST_FUEL`), not as a refusal. The same probe gives the same goals (determinism is gated).

```python
eng = Engine(w, kinds=set(Kind))            # all kinds (the first language is the default)
ep  = Episode(eng, statement, premises=n)   # the statement as a goal
m   = ep.menu()                             # Menu: m.actions (a list of Action), m.cands (the lemmas/hypotheses they name)
ep.step(m.actions[i])                       # probe, then do; ActionError if it does not apply
ep.done, ep.finish()                        # finish: the kernel checks the proof term against the statement
```

## 2. The first language (a finite menu)

Kinds `0..11`: the learner picks an action from `Menu.actions`.

| kind | argument | meaning |
|---|---|---|
| `INTRO` | – | introduce the first binder (only when `auto_intro` is off) |
| `CASES`, `INDUCT` | hypothesis position (+ `hyps`: hypotheses to generalise) | case analysis / induction on a hypothesis |
| `APPLY` | index into `cands` | apply a lemma or hypothesis (only when the goal determines its data arguments) |
| `REWRITE` | index into `cands` (direction in `cands`) | rewrite left→right or right→left at the first instance |
| `RING`, `LINARITH`, `NLINARITH` | – | the decision procedures (kernel-checked certificates; `NLINARITH` multiplies hypothesis pairs) |
| `BY_CONTRA` | – | classical: assume the negation, prove `False` |
| `INTERVAL_CASES` | index into `cands` (variable, lo, hi) | split an integer variable bounded on both sides |
| `INSTANTIATE` | pool terms | `have h := H t1 …` with terms from the goal's term pool |
| `USE` | pool term (or `codes`) | the witness of an existential goal |

## 3. The second language (arguments are *said*)

An argument that is an arbitrary well-typed term of the goal's context is a **code**: a tuple of builder tokens (§4).
Kinds `12..20`. A **source** (lemma/hypothesis) is a code of one token: `('c', gid, 0)` for a global, `('v', level, 0)` for
a hypothesis.

| kind | codes / hyps / arg | meaning |
|---|---|---|
| `CLAIM` | `(statement,)` | `have h : statement` — two goals: the statement, then the goal with `h` |
| `REFINE` | `(term,)` | the term has the goal's type; each `('g',)` in it is a proposition left as a goal (`exact`, witnesses, splits, `apply f x ?_`) |
| `APPLY_WITH` | `(source, t1, …)`; arg 1 = let the unifier find the rest | `apply source t1 …` — the `t_i` are the data arguments the goal does not determine |
| `REWRITE_WITH` | `(source, t1, …)`; arg 1 = right→left | `rw source t1 …` — the rest found by matching, first instance |
| `REWRITE_AT` | `(source, t1, …)`, `hyps=(h,)` | the same inside hypothesis `h` |
| `FORWARD` | `(source, s1, …)` | `have h := source s1 …`; a slot is a term code, a hypothesis code (for a premise) or `()` (an argument the others determine) |
| `INDUCT_WITH` | `(principle, e1, …)`, `hyps=(x,)` | induction on `x` by a chosen library principle (`N.induction_on`, `N.induction3`, `Z.induction_from`, …): the motive is read off the goal, the `e_i` are the principle's remaining arguments (starting point) |
| `INJECT` | `arg = h` | from `c a… = c b…` the facts `a = b …` (injectivity lemmas) |
| `LINCOMB` | `hyps=(h1,…)`, `codes=(c1,…)` | `linear_combination (c1,h1) …` closed by `ring` |

The signature of an action (`Menu.signature`) is the whole of it; `Menu.find(signature)` finds it again in any menu,
also one that does not list it (second-language actions mean what they say).

## 4. The typed term builder (`gym_build.py`)

A term is built in prefix order, one **choice** per step, and at every step only the choices that keep the term well
typed are offered:

```python
space = TermSpace(eng, hyp_types, hyp_names)      # the context of a goal
b = space.builder(expected_type)                  # a term of this type is wanted
b.tokens()                                        # the legal choices at the first hole, in a fixed order
b.push(token)                                     # take one; the next hole is the first unfilled one
code = space.encode(term, expected_type)          # the canonical way to say an existing term
term = space.build(code, expected_type)           # a whole code at once (checked by the kernel again)
```

Tokens: `('v', level, k)` variable applied to `k` arguments · `('c', gid, k)` global applied to `k` arguments ·
`('n', value)` numeral · `('lam',)` lambda · `('all',)` ∀ (type, body) · `('imp',)` · `('arr',)` function type ·
`('s', l)` sort · `('p', i)` the i-th pool term · `('g',)` a goal left open (refine).
A head takes the arguments its type needs to reach the hole's type *minus those the expected type determines* (the
implicit carriers), so a statement is said without the noise. A global above the engine's `limit` (the premises) is
refused: `BuildError('no global n')`.

**Known limits.** Terms with `match`/`let` cannot be said (`NotEncodable`; 2 of the library's 400 proofs: `Z.induction_on`,
`Z.mk_inj`); the funext statement is unsayable for the same reason. Numerals have one spelling per type
(`Z.of_N (Npos …)`), so `Z.one` and `lit(1)` are different terms to the matcher.

## 5. The portable form

An action as plain JSON data (strings, ints, lists): `to_portable(eng, menu, action)` / `from_portable(eng, menu, d)` /
`replay_portable(eng, stmt, steps, premises)`. A lemma or hypothesis is named by its global id / level, not by a menu
index; actions relative to the term pool (`INSTANTIATE`, `USE` by position) have none — say the terms by codes.
Malformed data (a missing key, a number where a list is wanted, a global that does not exist) is `ActionError`, never
another exception. The ids are those of the world: a dataset names its world by `bank.world_digest`.

## 6. The recorder: authored proofs as action sequences

`recorder.record_world(build)` builds a library with every authoring call logged; `recorder.replay(eng, rec,
premises=None)` pairs the author's goal with the engine's goal after each call (`align`: a monotone pairing of the
hypotheses by name and type, plus a fallback for variables introduced after a `have`), translates the call into one or
more candidate plans of actions (`TRANSLATORS`), keeps the first plan whose probe leaves the goals the author's call
left, and finally has the kernel verify the term. Coverage of the library: **453 of 459** lemmas (gate S40's library is the whole bank world's library; the six that do not replay are listed with their reasons in BANK.md section 7b: two
terms with a `match` or a proof by computation, two authored inductions whose hypothesis the engine does not have, two `apply`s whose goals the engine settles itself; the gate pins them by name and reason). A result carries `actions`, `portable`, `kinds`, `ops`; `demonstrations(results)` gives
them as JSON. The reference proofs of the problem bank are replayed the same way (BANK.md), also with the lemmas
written for the problem hidden (`premises=`).

## 7. What is gated (S40, 600 checks; S41 for the bank)

The builder says and builds back every term of the library's proofs; random walks over its choices give terms the
kernel accepts and every offered choice is legal; every action kind is accepted/refused as documented, refusals change
nothing; the portable form round-trips and malformed data is `ActionError`; probes leave the kernel as it was; menus are
bounded and deterministic and every action is found again from its signature; the library's proofs replay (also from JSON
in a fresh engine, also checked by the independent checker); random play on true and false statements never finishes a false
one; the assembler's own work budget (a term whose check does not stop is refused, a spent budget is never read as "no proposition": finding F86); the menu's work budget (listing a goal's menu runs its kernel calls -- the normal forms of the candidate facts and of the goal -- under the engine's budget `MENU_FUEL`, not the library's; a fact whose normal form spends it is left out and counted, a goal whose own normal form spends it keeps the goal as it is, the caller's budget is given back, a cached normal form remembers the budget it was made under: finding F87, S40 section E7-E8); the work budget of the tactics that compute in Python before the kernel sees a term (`ring`, `linear_combination`, `linarith`, `nlinarith`: the proof builder and the search count their own work -- units, spent before the memory they stand for is allocated -- and stop at what the probe's kernel budget allows (`fuel // 50` units); a spent budget is a `fuel` error, the probe is `ST_FUEL`, never a refusal; `interval_cases` calls `linarith` up to 1 500 times and each call has the budget, not the action: finding F90, S25 section I, S30 J, S31 H, S40 E9). The mutation driver (`mutate.py`, 1726 mutants of the engine/builder/actions/recorder) measures how much of this
the gates actually pin down.

## 8. Not done yet

* A **slot-by-slot policy interface** (assemble an action argument by argument with legal-kind masks) and numeric
  features of codes for a neural policy; today the agent picks from `Menu.actions` or submits whole codes.
* Numeral-aware matching in the builder; hiding implementation lemmas from menus; trigger-based `INSTANTIATE`.
