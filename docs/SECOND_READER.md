# SECOND_READER.md -- reading formal statements against their problems

A handout for a person who has not seen this repository and will not need to. You get a sheet of statements. For each one you decide whether the *formal* statement says what the *problem* says. You prove nothing and run nothing.
Five minutes per statement is typical; stop whenever you like -- every verdict you give is useful on its own.

## Why this is asked of you

The proofs in this project are checked by a small kernel, but a proof of the wrong statement proves nothing about the problem, and the kernel cannot tell the two apart. The authors wrote the statements, and also wrote the
program that checks them against their intended meaning, so a misreading of the English that both share is invisible to every test they have. A reader who did not write the statement is the only protection. An unfaithful
statement that you find is a result, not an embarrassment: it is fixed and the paper says it was found.

## What you have

* `reader_K.md` -- the statements assigned to you: the problem as the authors transcribed it, the formal statement, the authors' own notes and the cautions the checks raised. Read it.
* `reader_K.csv` -- the same statements as a table. Fill in the columns `reviewer` (your name, as you want it recorded), `verdict` and, when the verdict is not plain `faithful`, `note`. Leave every other column alone: the `digest`
  ties your verdict to the exact text you read, so that if the statement changes later your verdict no longer counts for it (which is what you want).

## The notation

Variables are named `x0, x1, ...` in the order they are bound. `Z` is the integers, `N` the naturals `0, 1, 2, ...`; a natural used where an integer is needed is converted silently (it is not written).

| written | means |
|:---|:---|
| `forall x0 x1 : Z, P` / `exists x0 : Z, P` | for all / there are integers; the quantifier reaches as far to the right as it can (parentheses mark where it stops) |
| `~P`, `P /\ Q`, `P \/ Q`, `P -> Q`, `P <-> Q` | not, and, or, implies, if and only if. `~` applies only to the next atom or parenthesised formula (`~(x0 = 2)`, `~prime(n)`). Then, tightest first: `/\`, `\/`, `->` (to the right: `A -> B -> C` is `A -> (B -> C)`), `<->` |
| `a = b`, `a < b`, `a <= b`, `a | b` | the comparisons and divisibility ("a divides b"; `0 | b` holds only for `b = 0`); they bind tighter than the connectives, looser than arithmetic |
| `+`, `-`, `*`, `^` | arithmetic; `^` is the power and binds tightest; `-` on `Z` is subtraction |
| `n!` | factorial |
| `sum(i0 < n, f)`, `prod(i0 < n, f)` | the sum / product of `f` for `i0 = 0, 1, ..., n - 1` (the empty product is 1) |
| `prime(p)` | `1 < p` and every positive divisor of `p` is `1` or `p` |
| `gcd(a, b) = d` | `d` is the (non-negative) greatest common divisor |
| `coprime(a, b)` | `gcd(a, b) = 1` |
| `false` | the false proposition (a statement `... -> false` says that the hypotheses cannot all hold) |

If a formula can be read in two ways, say so in the `note`: the sheet is printed by a program from the statement's tree, and an odd-looking parenthesis is worth a remark.

Under each statement the sheet lists **cautions**. A caution means that the automatic checks are weaker for this statement, so read it with extra care:

| caution | what is weaker |
|:---|:---|
| `adapted` | the problem was changed to fit the vocabulary (an integer variant, one part of several); the notes say how |
| `source_unchecked`, `source_recalled` | the problem text was not compared with a public source, or is from memory: check the transcription yourself |
| `truncated_arithmetic` | the statement uses the predecessor on `N`, which stops at 0 |
| `window_existential` | there is an `exists`; the checks look for witnesses only among small numbers |
| `function_quantifier` | there is a quantifier over a function; the checks use a family of test functions, not all functions |
| `hypotheses_unchecked`, `atoms_unchecked` | the numbers inside an equation of the hypotheses, or of the conclusion, were not compared with the intended ones: read those numbers yourself |
| `exempted_slips` | some deliberate mistakes (a swapped argument, `<` for `<=`) are not tried because the authors declared them equivalent to the statement |
| `thin_box` | few of the test points satisfy the hypotheses, so little was compared |

## What to check, in this order

1. **Read the problem first**, and say to yourself what it asks, before you look at the formal statement. If you know the source (an IMO, an ISL, an AIME), also check the transcription itself.
2. **Domains.** Integers, naturals or positive integers? Does the formal statement have the hypotheses that make the variables positive, distinct, consecutive, coprime, as the problem says?
3. **Quantifiers.** The order of "for all" and "there exists", and what each reaches over. "For every `a` there is a `b`" is not "there is a `b` for every `a`".
4. **Hypotheses.** Each one present, none added, none dropped. An extra hypothesis makes the theorem weaker than the problem; a missing one can make it stronger, even false.
5. **The conclusion.** "Determine all ..." should be an `<->` that gives the full answer set; "find the smallest" needs both that nothing smaller works and that the value works; "prove that there are no solutions" is a negation over the whole domain; "prove that X" is X.
6. **Edge cases.** `<` against `<=`; zero and negative numbers; the empty case; off-by-one in a sum or a product; a power with exponent 0; a subtraction on `N`.
7. **The authors' notes -- last.** They say what the authors meant. They are a claim to test, not a fact. If you agree with the notes but not with the formula, the formula is wrong.

## The verdicts

* `faithful` -- the formal statement says what the problem says.
* `adapted` -- it says what the problem says after the adaptation the notes describe and you accept it (an integer variant of a statement about reals, one part of several, a contrapositive, "find" turned into "prove that the answer is").
* `ambiguous` -- the English admits several readings and the formal statement picks one. Say which reading it picks and which other reading is possible.
* `unfaithful` -- it says something else. Say what is different. (The statement may still be true and proved; that is the point.)

Leave `verdict` empty for a statement you did not read; it counts as not read.

## Three worked examples from the bank

**`aime1989_p4` -- faithful.** *Five positive integers `a < b < c < d < e` are consecutive. The sum `b + c + d` is a perfect square and `a + b + c + d + e` is a perfect cube. What is the smallest possible value of `c`?*
`forall x0 x1 x2 : Z, 2 < x0 -> 3 * x0 = x1^2 -> 5 * x0 = x2^3 -> 675 <= x0 /\ (3 * 675 = 45^2 /\ 5 * 675 = 15^3)`.
Reading: `x0` is `c`; `a = c - 2 > 0` is `2 < x0`; the two sums are `3c` and `5c`; the conclusion says nothing smaller than 675 is possible (`675 <= x0`) and that 675 works (`3 * 675 = 45^2`, `5 * 675 = 15^3`). The
question "what is the smallest" became "no smaller, and this one works". Faithful.

**`imo1986_p1` -- adapted.** *Let `d` be a positive integer different from 2, 5 and 13. Show that one can find two distinct elements `a, b` of `{2, 5, 13, d}` such that `ab - 1` is not a perfect square.*
The formal statement is a contrapositive: for positive `d` not in `{2, 5, 13}`, if `2d - 1`, `5d - 1` and `13d - 1` are all squares, then `false`. The pairs `(2,5)`, `(2,13)`, `(5,13)` already give the squares 9, 25 and 64, so the claim
is that the three expressions with `d` are never all squares. The notes say so; you decide whether you accept the reduction. If so: `adapted`.

**An invented slip -- unfaithful.** *The product of `n` consecutive non-negative integers `m (m+1) ... (m+n-1)` is divisible by `n!`.* The bank's statement is `forall x0 : Z, forall x1 : N, 0 <= x0 -> x1! | prod(i0 < x1, x0 + i0)`
(`x1` factors, `x0, ..., x0 + x1 - 1`). Had it said `prod(i0 < x1 + 1, x0 + i0)` it would be a true theorem with a proof, about `n + 1` factors, and not the problem. A check against the authors' own description may or may not catch this one;
if the description carries the same mistake, nothing will. That is what you are for.

## Sending it back

Return the filled `reader_K.csv`. The author runs `python3 src/python/experiments/exp_audit.py import-reviews reader_K.csv --reviewer "YOUR NAME"` (all or nothing: a wrong row is named and nothing is written), the ledger
`docs/statement_reviews.tsv` gets one line per statement with your name, the date, the verdict and the digest, and `exp_audit.py report` counts the statements that a second reader confirmed. If two readers disagree
the statement is reported as disputed and the authors settle it; an `unfaithful` verdict makes the audit fail until the statement is fixed or the verdict is withdrawn after discussion.

Ask the author how your name is to be used (the ledger holds it; the acknowledgments are the author's choice and yours).
