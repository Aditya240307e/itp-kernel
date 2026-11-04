# src/python/authoring -- worlds, terms, tactics and decision procedures

The layer you write mathematics in. **Nothing here is trusted.** Every declaration goes through the kernel's *checked* entry points (`declare_checked`, `finish_inductive`, `finish_recursive_function_checked`), and every tactic ends in a proof term that the kernel type-checks when the theorem is declared. The one unchecked declaration is `World.axiom`, which adds a constant with a type and no value; the libraries declare exactly one, excluded middle (`em`, in `nt_lib`). A bug in this directory can make a library fail to build or a tactic fail; it cannot make a false theorem appear.

| module | what |
|---|---|
| `author.py` | `World` (a kernel plus a name table), `Term` (a kernel term that knows its world), the notation system, declarations, higher-order-abstract-syntax binders, `match`, `prove`, a printer |
| `tactics.py` | `Proof`: named hypotheses, normalisation of goals, rewriting by pattern, `have` / `obtain` / `calc`, and the glue to the decision procedures |
| `ring.py` | `ring`: normal forms of commutative-(semi)ring expressions, with a proof term for every step; instances for `Nat`, binary `N` and `Z`. Also the work meter (`WorkMeter`) |
| `linarith.py` | `linarith` and `nlinarith`: linear arithmetic over `Z` by certificate (an exact simplex) |
| `lincomb.py` | `linear_combination`: close an equation from other equations, given the coefficients |
| `noconf.py` | a generator of constructor-injectivity lemmas (`Succ a = Succ b -> a = b`), exercised by the kernel gate S10; the standard libraries state their own |

All of it is used the same way: build a world (usually with a library, [`../library`](../library/README.md)), state a theorem, give a script.

```python
import itp_paths, nat_lib                       # itp_paths makes every module importable by its plain name
from lnl_base import thm

w = nat_lib.build_nat()
Nat, Zero, Succ, Eq, forall = w['Nat'], w['Zero'], w['Succ'], w.Eq, w.forall_

w.prove('add_zero_again', forall([Nat], lambda n: Eq(n + Zero, n)),
        lambda p: (p.intro('n'), p.ring()))     # the script runs, the term is extracted, the kernel checks it

@thm(w, 'add_comm_again', [Nat, Nat], lambda a, b: Eq(a + b, b + a))    # the same, as a decorator: name, binder types, proposition
def _(p):
    p.intro('a', 'b')
    p.ring()
```

## Declaring types and functions

```python
w.inductive('Tree', w.TYPE, [('leaf', []), ('node', [lambda *_: w['Tree'], lambda *_: w['Tree']])])
Tree, leaf, node = w['Tree'], w['leaf'], w['node']

w.recursive('size', w.arrow(Tree, Nat),                       # structural recursion; the kernel checks that it is
            lambda size: w.fun([Tree], lambda t: w.match(t, lambda _: Nat, [Succ(Zero), lambda l, r: size(l) + size(r)])))
size = w['size']
w.prove('size_node_leaf', Eq(size(node(leaf, leaf)), 2), lambda p: p.refl())      # by computation
```

The kernel refuses what it cannot justify, and the layer passes its reason on:

```python
w.inductive('Bad', w.TYPE, [('mk', [lambda *_: w.arrow(w['Bad'], Nat)])])
# AuthorError: inductive type Bad was rejected: itp: constructor 'mk' fails strict positivity -- the inductive being defined appears to the left of an arrow in an argument type
w.recursive('loop', w.arrow(Nat, Nat), lambda f: w.fun([Nat], lambda n: f(n)))
# AuthorError: recursive function loop was rejected: function 'loop' failed the structural recursion check
```

## `World`

`World(parameters=True, prop_elim=True, fuel=None)` creates a kernel, sets its options (`parameters` separates the uniform parameters of an inductive family from its indices; `prop_elim` is the `Prop`/`Type` world) and the work budget for one call into the kernel (default 2 x 10^9 steps; the environment variable `ITP_FUEL` overrides it, 0 means unlimited). In this world propositions live in `w.PROP` (`Type 0`) and data in `w.TYPE` (`Type 1`).

| | |
|---|---|
| names | `w['name']` is the constant `name` as a `Term`; `'name' in w`; `w.global_type(name)`; `w.global_value(name)` (`None` for axioms and inductive types); `w.constructors(family)` |
| declare | `w.axiom(name, type)`; `w.define(name, type, value)` (the kernel checks the value against the type); `w.inductive(name, type, ctors)`; `w.recursive(name, type, body_fn, on=0, over=None)`; `w.prove(name, statement, script)` |
| build terms | application `f(*args)` or `w.app(f, *args)`; binders by higher-order abstract syntax: `w.pi(A, lambda x: B(x))`, `w.lam(A, lambda x: ...)`, `w.arrow(A, B)`, `w.forall_([A, B, lambda a, b: P(a, b)], lambda a, b, h: ...)` and `w.fun(...)` for telescopes (a later binder type may be a function of the earlier variables); `w.match(scrutinee, motive, branches)` |
| logic | `w.Eq(a, b)`, `w.And`, `w.Or`, `w.Not`, `w.Iff`, `w.Ex(A, fn)`; on `Term`s the operators `&`, `\|`, `~` and `>>` (implication) |
| inspect | `w.show(t)`; `w.infer(t)`, `w.type_of(t)`, `w.check(t, ty)`; `w.normalize(t)`, `w.simp(t)` (the display normal form: computation, definitions left folded), `w.conv(a, b)`; `w.spine(t)` (head and arguments), `w.view(t)`, `w.pi_parts(t)`, `w.eq_parts(prop)` |

* **`inductive`** takes a list of `(constructor, fields, indices)`. Each field type is a `Term` or a function of the fields so far (the family itself is `w[name]`); the optional `indices` is a function of the field variables returning the arguments of the family in the constructor's result. A family's leading arguments are *parameters* when every constructor has, at that position of its result, a bare field variable and every recursive occurrence in a field passes the same one (the kernel's analysis, `param_slots`); the rest are indices. A parameterised family lists its parameters first among the fields: `('conj', [PROP, PROP, lambda a, b: a, lambda a, b, x: b], lambda a, b, x, y: [a, b])` is `And`. Admission runs the universe and strict-positivity checks.
* **`recursive`** declares a function by structural recursion on argument `on` (a position among the Pi arguments of its type; `over` names the inductive type of that argument and is read off the type when omitted). `body_fn(f)` receives the function itself for the recursive calls. The kernel's guard accepts a call only on a structural subterm.
* **`match`** is case analysis for families with parameters and no indices: `motive` is a function of the scrutinee (or a ready-made `Term`; for an indexed family it must be one), `branches` has one entry per constructor in declaration order, each a `Term` (no fields) or a function of the constructor's field variables.
* **Binders** are *locally nameless*: while the body of `w.lam(A, lambda x: body)` is built, `x` is a fresh context variable, abstracted when the binder closes; the Python parameter names become the display names of goals. A `Term` knows its world, so mixing terms of two worlds is an `AuthorError`, not a memory hazard.
* **Notation.** `Term` has `+ * ** - <= < >= >` and unary `-`; what they build depends on the world's notation (`w.nota`). A library registers one per number system (`Nat`, `N`, `Z`) in a `DispatchNotation`: the operator looks at the type of its operands, a Python `int` becomes a numeral of the other operand's type, or of the default one, and `with w.nota.using(Z): ...` changes the default. `a == b` is deliberately *not* the proposition (a `Term` has no truth value); write `Eq(a, b)`, and `a.is_same(b)` for syntactic comparison.
* **Errors.** Everything that goes wrong is an `AuthorError`. Its attribute `fuel` is true when the cause was a spent work budget (a bigger one might have succeeded) and false for a refusal on logical grounds.

## Tactics (`tactics.Proof`)

`Proof(w, statement, name=None, auto_norm=True)` holds a kernel proof state. `print(p)` shows the goals, `p.goal(i)` gives a snapshot (`.hyps`, `.target`), `p.n_goals` and `p.done` say how many are left, `p.term()` is the finished proof term, and `p.trace` records every primitive step. `Proof.from_goal(w, target, hyp_types, hyp_names)` starts from a goal given as data (this is how a search works). Tactics act on the first goal and return the `Proof`, so calls chain. A failing tactic raises `AuthorError` with the reason and the goals. A kernel step that fails leaves the state as it was (the kernel rolls back); a composite tactic such as `have` or `calc` may have completed earlier steps, and the environment's probes use checkpoints to try tactics without side effects. With `auto_norm`, goals are kept in normal form, so `add (Succ m) n` is displayed as `Succ (add m n)`.

| tactic | does |
|---|---|
| `intro(*names)` | introduce `forall` / `->` binders with these names |
| `exact(term)` | close the goal; `term` is a `Term`, a hypothesis or global name, an `int` (a numeral), a tuple `(f, *args)`, or a function of the proof |
| `apply(term, *args)` | apply a lemma or hypothesis; its premises become goals |
| `split()`, `left()`, `right()`, `exfalso()`, `exists(witness)` | `A /\ B`, `A \/ B` and `Ex` goals; `False` elimination |
| `cases(var, names=None, generalizing=())` | case analysis on a hypothesis of an inductive type (indexed families too) |
| `induct(var, names=None, generalizing=(), indexed=False)` | induction. `generalizing` reverts those hypotheses first, so the induction hypothesis holds for all of them; `indexed=True` generalises the indices of an indexed family through equations |
| `induction_on(lemma, var, names=(), extra=())` | induction by a library principle (from a starting point, strong induction, quotient types) |
| `have(name, statement, script)`; `have(name, proof_term)`; `obtain(names, fact, script=None)` | add a fact, proved by `script` or read off a term; `obtain` also destructs a conjunction or an existential |
| `rewrite(eq, rev=False)` | rewrite with an equation given as a term |
| `rw(lemma, *args, rev=False, close=True)` | rewrite by a lemma or hypothesis, finding its quantified variables by matching the first instance in the goal (pre-order; instances that mention a bound variable are not found); closes a goal that becomes `x = x` |
| `rw_at(hyp, lemma, *args, rev=False, new=None)` | the same inside a hypothesis; the rewritten one is added under the same name, shadowing the original |
| `refl()`, `try_refl()`, `assumption()` | close by reflexivity, up to computation, or by a hypothesis (`try_refl` returns a bool) |
| `norm()`, `clear(levels)` | replace the goal by its normal form; drop hypotheses |
| `calc(e0, (justification, e1), ...)` | a chain of equations; a justification is `'ring'`, a proof (either way round), or a function of the proof |
| `by_contra(name='h')` | classical: assume the negation (the world must have the lemma `by_contra`, i.e. excluded middle) |
| `interval_cases(var, lo, hi, name=None)` | split an integer variable into `hi - lo + 1` goals (fewer than 500), each with `var = k`; the bounds are proved by `linarith` |
| `ring()`, `linarith(*hints)`, `nlinarith(*hints)`, `linear_combination((c, h), ...)` | the decision procedures, below |
| `T('a b', lambda a, b: ...)`, `h(name)`, `t(x)`, `focus()` | build terms under the hypotheses of the current goal, for hints and witnesses |

## Decision procedures

Each procedure *finds* a certificate in Python and *builds an ordinary proof term* from it, so a wrong procedure can only fail.

* **`ring`** closes an equation of a commutative semiring (`Nat`, binary `N`) or ring (`Z`). Both sides are normalised to a canonical sum of monomials -- atoms in increasing index order, monomials in decreasing key order, numerals in the number system's own canonical form -- and each merge is one application of a `ring_*` lemma of that number system (proved in [`ring_lib`](../library/ring_lib.py)). Sums, products, powers with a numeral exponent, `succ`, and for `Z` negation and subtraction are understood; every other subterm is an atom, kept in a canonical spelling (so `f (b + 0)` and `f b` are the same atom). Powers use repeated squaring, so `(a + b) ^ 20` takes five multiplications. Numerals of `N` and `Z` are binary: coefficients of competition size cost nothing. When the normal forms differ the error shows both.
* **`linarith`** closes a goal that follows from the hypotheses by linear arithmetic over `Z`. Hypotheses become facts `0 <= p`: `x <= y` is `0 <= y - x`, `x < y` is `0 <= y - (x + 1)`, an equation gives two facts, a negated inequality is read as the opposite inequality, and a monomial that is not linear is an atom. The search looks for integers `m_i >= 0` such that `sum m_i * p_i` is a negative constant as a polynomial -- a linear program, solved by an integer-preserving simplex with Bland's rule in exact arithmetic (for large problems SciPy's HiGHS first guesses the support, which the exact simplex then re-solves, so floating point never decides anything). The goal `x <= y` is proved by contradiction from `y < x`, `x = y` from both inequalities, and any other goal after `False` elimination. The proof is one `exact`, so a failed search changes nothing.
* **`nlinarith`** first adds the squares `t * t` and `t ^ 2` of the subterms that occur in the hypotheses and the products of every pair of facts (at most 600 products), then searches as above: certificates of degree two. Extra facts are passed as hints, which are proofs: `p.nlinarith(p.T('x y', lambda x, y: w['Z.sq_nonneg'](x - y)))` ([`examples/02`](../../../examples/02_ring_linarith.py)).
* **`linear_combination((c1, 'h1'), (c2, 'h2'), ...)`** closes `lhs = rhs` from equations `h_i : l_i = r_i` when `lhs - rhs = sum c_i (l_i - r_i)` as polynomials. A coefficient is a term, a function of the proof, or an `int`. The proof applies the congruence lemmas of the ring to the hypotheses and uses `ring` for the identity; a wrong coefficient makes `ring` fail and shows both normal forms.
* **`noconf.py`** generates, for a family whose every argument is a parameter (`Nat`, trees, `List`, `And`, `Or`, `Ex`), the injectivity lemma `c_inj_q : c a.. = c b.. -> a_q = b_q` for each constructor `c` and payload field `q`, through the checked path (discrimination of different constructors is `cases` itself). Not generated: indexed families (`Eq`, `Le`) and dependent payload fields. Gate S10 checks the lemmas with the kernel and with the independent checker.

**Work budgets.** The Python side of `ring`, `linear_combination`, `linarith` and `nlinarith` computes before the kernel sees a term, so the kernel's budget does not bound it. Each call therefore carries a meter (`ring.WorkMeter`: one unit for every lemma application, monomial and numeral built, and for the cells of the simplex tableau made and rewritten), whose limit is the kernel's budget divided by 50. A spent meter raises `RingWorkExhausted`, an `AuthorError` with `fuel = True`. The caller can tell "ran out of budget" from "refused on logical grounds", and neither is ever an acceptance.

## Adding to a library

A library is a chain of stages (functions of a world that call `w.inductive`, `w.recursive`, `w.define` and `@thm`); every stage runs from scratch each time a world is built, and every theorem in it is proved by a script and re-checked by the kernel. See [`../library`](../library/README.md). To see what a theorem rests on, ask `audit.dependencies_of_global(w.k, w.gid['name'])`, which returns the axioms and quotient primitives it uses (both empty for a theorem about the natural numbers).
