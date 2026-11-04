# src/python/environment -- the proof environment

Proving as a sequential decision problem, for an agent that has **no text interface**: a goal is data, a step is an *action* (a kind and its arguments), the legal actions of a goal are listed in a *menu*, an action is *probed* before it is taken, and the finished proof is checked by the kernel. The same layer serves a hand-written search, a script that replays recorded proofs, and a learning agent.

**Nothing here is trusted.** A step is carried out by the tactic layer ([`../authoring`](../authoring/README.md)), which builds ordinary proof terms, and the proof that an episode finishes with is checked by the kernel against the statement (`Engine.verify`, `Episode.finish`). A bug in this directory can waste work or lose a proof; it cannot make a false theorem appear.

| module | what |
|---|---|
| `gym_env.py` | `Engine` (goals, menus, probes, replay for one world), `Episode` (one proof attempt), `GoalData`, `Menu`, `Action` and the 21 `Kind`s; the first, menu-based action language |
| `gym_acts.py` | the second action language, whose arguments are *said* (terms, statements, references); the portable form of an action |
| `gym_build.py` | the typed term builder: any well-typed term of a goal's context, said one typed choice at a time |
| `gym_terms.py` | a deterministic pool of small well-typed terms (variables, numerals, sums, products, ...) for the actions that carry a term |
| `gym_premises.py` | which globals an attempt may use (`Premises`: a prefix of the world minus hidden ranges) |
| `gym_policy.py` | the policy-facing view: an action *assembled* by a sequence of discrete choices (`Assembler`, `Choice`) |
| `gym_search.py` | proof search over goals: iterative deepening, and a cost-ordered variant; `prove_statement` |
| `gym_tree.py` | the proof tree of an episode and the theorems it proved on the way (hindsight), each replayed and kernel-checked |
| `recorder.py` | authored tactic proofs as action sequences: runs a library, logs every tactic call, and replays the proofs through the engine |

## The model

* A **goal** is a `GoalData`: a target and the types and names of the hypotheses (kernel handles, nothing else). Terms are hash-consed, so two searches that reach the same goal by different routes reach the same object (`GoalData.key`), and `observe` gives the goal as arrays (node tags, scalars, child pointers, roots, heights).
* A **step** is an `Action(kind, arg, args, codes, hyps)` applied to the first open goal. After every step the goals it leaves are *settled*: a goal that a hypothesis or reflexivity closes is closed, and the quantifiers and implications at the head of a goal are introduced (`auto_intro`), so no step is spent on them.
* A **probe** runs an action on a one-goal copy of the goal and reads off the goals it leaves, then *rewinds the kernel completely* (checkpoint and rewind; every node made during the probe is reclaimed). An action that fails leaves the kernel exactly as it was. `Episode.step` probes first, so an action that does not apply raises `ActionError` and changes nothing.
* **Budgets.** Every probe runs under a work budget of the kernel's fuel, set per kind of action; running out is reported as a spent budget (`ST_FUEL`), not as a refusal. The same probe gives the same goals: determinism is gated.
* **Premises.** An attempt may use only the globals the engine's `premises` allow, so a theorem is attacked with the library that precedes it, and the problems of a bank do not see each other's solutions.

```python
import itp_paths, nat_lib
import gym_env as G

w = nat_lib.build_nat()
Nat, Eq, forall = w['Nat'], w.Eq, w.forall_
eng = G.Engine(w)                                             # the first action language by default; kinds=set(G.Kind) for all 21
ep = G.Episode(eng, forall([Nat, Nat], lambda a, b: Eq(a + b, b + a)), premises=w.k.num_globals())

print(eng.render(ep.goal()))
# a0 : Nat
# a1 : Nat
# |- a0 + a1 = a1 + a0
menu = ep.menu()                                              # the legal actions of the first goal, and the lemmas they name
print(len(menu.actions))                                      # 27: APPLY 20, CASES 2, INDUCT 2, REWRITE 2, RING 1
try:
    ep.step(G.Action(G.Kind.LINARITH, 0))                     # not applicable: refused, and nothing changes
except G.ActionError as e:
    print(e)                                                  # the action LINARITH(0) does not apply
ep.step(next(a for a in menu.actions if G.Kind(a.kind) == G.Kind.RING))      # returns the number of goals left: 0
print(ep.done)                                                # True
proof = ep.finish()                                           # the kernel checks the finished term against the statement
```

## The action language

Twenty-one kinds. The numbers are part of the data format (recorded proofs store them).

* **The first language** (kinds 0..11) is a finite menu: the agent picks an action from `Menu.actions`. `INTRO`, `CASES`, `INDUCT` (on a hypothesis, optionally generalising others), `APPLY` and `REWRITE` (a lemma or hypothesis from the menu's candidate table; the library index offers only the ones whose conclusion can match the goal), the decision procedures `RING`, `LINARITH`, `NLINARITH`, `BY_CONTRA`, `INTERVAL_CASES`, and the two kinds that carry a term from the pool, `INSTANTIATE` and `USE`.
* **The second language** (kinds 12..20) is for what a list cannot offer, because the argument is an arbitrary well-typed term of the goal's context: `CLAIM` (`have h : statement`), `REFINE` (`exact` with holes that become goals; witnesses, splits), `APPLY_WITH`, `REWRITE_WITH`, `REWRITE_AT`, `FORWARD` (`have h := lemma args`), `INDUCT_WITH` (induction by a library principle), `INJECT` (constructor injectivity) and `LINCOMB` (`linear_combination`). Such an argument is a **code**, a tuple of builder tokens; a lemma or hypothesis is named by a code of one token.
* **The typed term builder** (`gym_build.py`) says a term in prefix order, one choice per step, and at every step offers only the tokens that keep the term well typed: a variable of the context applied to *k* arguments, a global applied to *k* arguments, a numeral of the hole's number type, a lambda, a universal quantifier, an implication. `space.encode(term, type)` is the canonical way to say an existing term, `space.build(code, type)` the inverse. Known limit: terms containing `match` or `let` cannot be said (`NotEncodable`).
* **The portable form** (`gym_acts.to_portable`, `from_portable`, `replay_portable`) writes an action as plain data, with no kernel handle and no index into a menu, so a recorded proof can be stored in a file, sent to another process and replayed in a fresh engine.

The full reference, with the arguments of every kind, is [`docs/ACTIONS.md`](../../../docs/ACTIONS.md).

## Search

```python
import gym_search as S
proof, tree, searcher = S.prove_statement(eng, statement, max_depth=6, max_probes=2000)
```

`prove_statement` searches by depth-bounded iterative deepening (or, with `cost=True`, by iterative deepening on cost, where each action has a cost in steps) over the AND-OR graph of goals: a goal is solved when some action solves all the goals it leaves. It returns a `Tree` of actions and replays it on a live proof, so the term it returns has been checked by the kernel; a wrong search can only fail. The actions of a menu are tried in the order given by `rank`: cheap closing actions first (decision procedures), then hypotheses and lemmas, rewriting, case analysis, induction.

## Recorded proofs and hindsight

* **`recorder.py`** turns the authored proofs of a library into action sequences. The library runs with every tactic call logged as a tree (`RecordingProof`); each call is translated into engine actions, probed, accepted when it leaves as many goals as the author's call did, and carried out; the proof the engine builds is checked by the kernel. A call that cannot be translated is reported with the reason, which is the list of what the action language still lacks. `python3 src/python/environment/recorder.py [--json FILE] [imo_lib.build_imo]` replays a whole library and prints the coverage by part of the library.
* **`gym_tree.py`** keeps the proof tree of an episode (`Episode(..., track=True)`). A failed attempt contains theorems that *were* proved: a case split that closed, a fact an action derived. `tree.examples()` returns them as new standalone problems with proofs, each replayed in a fresh engine and checked by the kernel before it is returned.

## Using it from a learning framework

`src/python/learner/bank_env.py` wraps the engine for the [problem bank](../problems/README.md) as a Gymnasium environment (`ITPBank-v0`, optional: `pip install gymnasium`). One episode is one proof attempt. The observation is the goal as token ids plus integer rows that describe the options of the question being asked; the action is the index of one option; an action is assembled from a short sequence of such questions (`gym_policy.Assembler`); the episode ends when the proof is complete and the kernel has checked it. [`examples/06_gymnasium_env.py`](../../../examples/06_gymnasium_env.py) plays a recorded proof through it.

## Checks

The gates S36 (engine, menus, probes), S38 (the term-carrying actions, the pool, probes with a chosen budget), S40 (the second language, the term builder, the portable form and the recorder), S42 (corner cases of the builder), S47 (hindsight) and S50 (every recorded proof of the bank replayed in a fresh engine and kernel-checked) cover it; `tests/fuzz` drives the engine with random and malformed actions. The examples [`04`](../../../examples/04_environment_episode.py) (an episode on a bank problem) and [`06`](../../../examples/06_gymnasium_env.py) (the Gymnasium wrapper) show it end to end.
