# src/python/library -- the standard library

Mathematics built from scratch on the kernel with the [authoring layer](../authoring/README.md): the logic, unary and binary natural numbers, the integers as a quotient, elementary number theory, finite sums and products, and the first contest theorems.

**Nothing is stored.** A library is a program: each `build_*` function declares every type and definition and re-proves every theorem, by a tactic script, through the kernel's checked path, every time it runs. There is no file of theorems to trust and no `sorry`; the only way in an assumption can enter is `World.axiom`, and the libraries use it once.

```
lnl_base ──► nat_lib ──► bin_lib ──► int_lib ──► nt_lib ──► ext_lib ──► imo_lib
 (logic)     (unary N)   (binary N)  (Z, as a     (number    (sums,       (contest
                                      quotient)   theory)    powers)      problems)
```

Each module builds on the one before it. `ring_lib`, the template of the lemmas that `ring` applies, is instantiated by `nat_lib`, `bin_lib` and `int_lib`; `quot_lib` is used by `int_lib`.

| module | builds | world size | time |
|---|---|---|---|
| [`lnl_base.py`](lnl_base.py) | the logic: `Nat`, `True`, `False`, `Eq`, `And`, `Or`, `Ex`, `Le`, `Not`, `Iff`; symmetry, transitivity, congruence and transport of equality; the first lemmas of the connectives | 40 | 0.02 s |
| [`nat_lib.py`](nat_lib.py) | unary naturals: `add`, `mul`, `pow`, the order; the laws of a commutative semiring; the lemmas `ring` instantiates | 129 | 0.1 s |
| [`bin_lib.py`](bin_lib.py) | binary naturals `Pos`, `N` and their arithmetic, related to the unary ones by theorems; the laws and the order of `N` | 252 | 0.3 s |
| [`quot_lib.py`](quot_lib.py) | the five quotient constants, `funext` derived from them, pairs `Prod`, `Quot.lift2` | -- | -- |
| [`ring_lib.py`](ring_lib.py) | the lemma template of the `ring` normaliser, written once for every commutative semiring with the usual laws | -- | -- |
| [`int_lib.py`](int_lib.py) | the integers `Z` as a quotient of pairs of binary naturals; the ring laws, powers, the order, discreteness, the integral domain | 434 | 0.7 s |
| [`nt_lib.py`](nt_lib.py) | number theory over `Z`: three forms of induction, divisibility, division with remainder, classical logic, Bezout, gcd, primes | 485 | 1 s |
| [`ext_lib.py`](ext_lib.py) | what contest proofs lean on: classical facts of number theory, finite sums and products, factorials, powers, squares, primes | | |
| [`imo_lib.py`](imo_lib.py) | contest problems, and with `ext_lib` the whole library | 555 | 3.5 s |

(The sizes are numbers of globals, counted by the kernel; the times are wall-clock on a two-core Linux machine. `python3 src/python/library/imo_lib.py` prints the size and time of the whole chain, and `nat_lib.py`, `bin_lib.py`, `int_lib.py` and `nt_lib.py` do the same for their part.)

## Using a library

```python
import itp_paths, int_lib
w = int_lib.build_int()                       # 434 globals, under a second
Z = w['Z']
w.prove('square_of_sum', w.forall_([Z, Z], lambda a, b: w.Eq((a + b) ** 2, a ** 2 + 2 * a * b + b ** 2)),
        lambda p: (p.intro('a', 'b'), p.ring()))
```

The builders are `lnl_base.build_base()`, `nat_lib.build_nat()`, `bin_lib.build_bin_defs()`, `int_lib.build_int()`, `nt_lib.build_nt()` and `imo_lib.build_imo()`; each builds the one before it in the chain, and those of `int_lib`, `nt_lib` and `imo_lib` take a `stages` list to build only some of their stages. `ext_lib.build_ext(w)` adds the extension to an existing number-theory world (`build_imo` does so itself). Python operators on terms mean the arithmetic of whichever number system the operands belong to (`+` on `Nat` is `add`, on `Z` is `Z.add`); an `int` literal becomes a numeral of the other operand's type.

**Adding your own results** means writing a stage, a function of a world, and calling it after a builder:

```python
from lnl_base import thm

def my_stage(w):
    Z = w['Z']
    @thm(w, 'neg_neg_again', [Z], lambda a: w.Eq(-(-a), a))      # name, binder types, the proposition over the binder variables
    def _(p):
        p.intro('a')
        p.ring()

my_stage(w)
```

A stage may also declare types and functions (`w.inductive`, `w.recursive`, `w.define`); all of it goes through the checked path.

## What each part contains

* **`lnl_base`** is the logical base, written with the authoring layer like everything after it. The world is the `Prop`/`Type` world: propositions live in `Type 0`, data in `Type 1`. `And`, `Or`, `Ex`, `Eq` and `Le` are inductive families; `Not` and `Iff` are definitions.
* **`nat_lib`** has addition, multiplication and powers by structural recursion, the order, and the semiring laws, all proved by scripts. Unary numerals are the Peano axioms made literal, and a numeral of value *k* is a term of size *k*.
* **`bin_lib`** adds the binary numbers (`Pos ::= XH | XO Pos | XI Pos`, `N ::= N0 | Npos Pos`), whose operations take a few hundred kernel steps for numbers of any size a competition uses. They are defined by plain structural recursion, with no carry flags, and **nothing about them is trusted**: each operation is related to the unary one it implements by a theorem (`Pos.to_nat_add`, ...), `to_nat` is shown injective, and so every equation between binary expressions follows from the unary one (the `transfer` helper). Subtraction, division and gcd are defined as the unary operation read back through the bridge; numeric facts about them are proved by certificates (`a = b*q + r` with `r < b`, checked with the fast operations).
* **`quot_lib`** declares the five quotient constants through the kernel's own entry point (the kernel builds their types itself and checks that the equality family it is given is the right one), derives `funext` from `Quot.sound` and the eta rule, and defines `Prod` and `Quot.lift2`.
* **`ring_lib`** states the lemmas that the `ring` normaliser applies (`ring_add_congr`, `ring_mul_congr`, `ring_pow_split`, ...) once, for any carrier with the laws `add_zero add_assoc add_left_comm mul_zero mul_one mul_assoc mul_left_comm mul_add add_mul pow_one pow_add`, and instantiates them for `Nat`, `N` and `Z`. `ring` only ever applies them, so an error in `ring` can make a proof fail to check, never make a false equation provable.
* **`int_lib`** defines `Z := Quot (Prod N N) Zrel`: the pair `(a, b)` stands for `a - b`. Addition, negation and multiplication are lifted out of the quotient and compute on `Z.mk` by `refl`; every law is proved by quotient induction down to an equation of `N`, which `ring` closes. Effectiveness (`Z.mk a b = Z.mk c d -> a + d = c + b`) comes from a Bool-valued test lifted out of the quotient, because lifting into `Prop` would need propositional extensionality, which the library does not assume. **No axiom is used**; the quotient is the kernel's five primitives.
* **`nt_lib`** is elementary number theory: integer induction from zero in both directions, from a lower bound, and strong induction; divisibility `Z.dvd a b := exists k, b = a * k` and its calculus; division with remainder (existence and uniqueness); Bezout's identity, greatest common divisors (`Z.is_gcd`), coprimality, Euclid's lemma; primes (`Z.prime p := 1 < p and every positive divisor of p is 1 or p`) and the existence of a prime divisor of every `n > 1`. Everything before the stage `nt_classical` is constructive. That stage declares **the one axiom of the libraries, excluded middle (`em`)**, and its everyday consequences (`by_contra`, `not_forall`, `not_exists`, `not_imp`, `not_or`). Gate S34 pins down that exactly the lemmas that need it use it.
* **`ext_lib`** holds the facts the contest problems of the bank lean on: the classical facts of elementary number theory (no multiple of `d` strictly between `dk` and `d(k+1)`, squares modulo 3 and 4, cubes modulo 9, `6 | n^3 - n`, `5 | n^5 - n`, ...), finite sums `Z.sum f n` and products `Z.prod f n`, the factorial, powers (signs, sizes, `x - y | x^n - y^n`, the Bernoulli congruence), squares and primes. No problem of the bank is among its lemmas.
* **`imo_lib`** is where the contest problems start. Its three theorems: `imo1959_p1` (the fraction `(21n+4)/(14n+3)` is irreducible: `forall n >= 0, Z.is_gcd (21n+4) (14n+3) 1`), `sqrt2_irrational` (no integers `p`, `q` with `q /= 0` and `p^2 = 2 q^2`) and `imo2019_p1` (all functions `f : Z -> Z` with `f(2a) + 2 f(b) = f(f(a+b))` are `f = 0` and `f(n) = 2n + c`). Each rests on the five quotient constants and on no axiom.

## Trust

The libraries are untrusted by construction: they are Python programs that drive the kernel, and everything they produce is checked by it when declared. What a theorem rests on is a question the auditor answers (`audit.dependencies_of_global`, see [`../checker`](../checker/README.md)): the natural-number results rest on nothing, the integer results on the five quotient constants, and the classical ones on `em`. The gate S34 checks which lemmas use `em`, and the gates S23, S24 and S25 have the independent checker judge every global of the natural-number, binary and `ring` worlds (examples [`01`](../../../examples/01_prove_with_tactics.py) and [`02`](../../../examples/02_ring_linarith.py) do the same for the natural-number and integer worlds).

The problem bank ([`../problems`](../problems/README.md)) is built on top of this: its world is `imo_lib.build_imo()` followed by the problems of `bank_lib`. The library proper is every global before `imo1959_p1`.
