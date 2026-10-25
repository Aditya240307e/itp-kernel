"""bank_dups.py -- duplicates in the bank: two problems that say the same thing.

Two problems that say the same thing are a leak when they land in different splits (the learner is trained on one and tested on the other) and, in the same split, a double count.  The bank defends itself in
three layers, and gate S49 section E pins all three:

  1. IDENTICAL STATEMENTS.  No two problems have the same formal statement (`bank.statement_digest`).
  2. THE SAME MEANING (`meaning_pairs`).  The statements are decoded (semantic.py) and evaluated on a canonical grid of small values: the truth values of the hypotheses and of the conclusion, and the NUMBERS
     that each relation atom (`l <= r`, `l = r`, `a | b`, ..) is made of.  Two statements with the same parameters (kinds; in any order, under any renaming) and the same tables are the same statement written
     twice, whatever the notation: `a^3 + b^3 >= a^2 b + a b^2` and `a^3 + b^3 >= a b (a + b)` are told apart only by their atoms, which is why the atoms are compared and not the truth values (every true
     inequality has the same truth table).  The statements that the evaluator does not read (functions, large existentials) are not covered; layer 3 is for them.
  3. SIMILAR TEXTS (`text_pairs`).  The words of the problems, with the single-letter variables renamed by order of appearance and the filler words dropped, compared as sequences (difflib ratio): a pair above
     the threshold is REVIEWED by a person: either the problems are one family (`bank_split.FAMILIES`: they are then never separated by a split) or they are different problems and the pair is entered in
     `REVIEWED_DIFFERENT` with the reason.  A new problem that is close to an old one fails the gate until someone has looked.

A pair of the same meaning must be in one family or be removed; there is no list of exceptions for layer 2.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import re, itertools, difflib, collections
import semantic as SEM

TEXT_RATIO = 0.8                                   # the similarity (difflib ratio of the normalised word sequences) from which a pair has to be reviewed

# pairs above the threshold that have been looked at and are different problems: {frozenset({a, b}): why}
REVIEWED_DIFFERENT = {
    frozenset({'fe_additive', 'fe_f_x_plus_f_y'}): 'different functional equations: Cauchy f(x+y) = f(x)+f(y) against f(x+f(y)) = f(x)+y (the second forces f to be a bijection first)',
    frozenset({'fe_additive', 'fe_quadratic'}): 'different functional equations: Cauchy against the quadratic equation f(x+y)+f(x-y) = 2f(x)+2f(y), whose solutions are not additive',
    frozenset({'fe_f_x_plus_f_y', 'fe_quadratic'}): 'different functional equations (f(x+f(y)) = f(x)+y against the quadratic equation)',
    frozenset({'gcd_shift', 'gcd_sum_product'}): 'different statements about coprime integers: gcd(a, b + ka) = 1 against gcd(a + b, ab) = 1',
    frozenset({'ineq_shifted_squares', 'ineq_abc_2abc_1'}): 'different inequalities: a^2 + b^2 + c^2 + 3 >= 2(a + b + c) (squares of a - 1, b - 1, c - 1) against the level-4 a^2 + b^2 + c^2 + 2abc + 1 >= 2(ab + bc + ca) (non-negative variables)',
    frozenset({'amc_fe_03', 'amc_fe_04'}): 'different functional equations: multiplicative f(xy) = f(x) f(y) with f(2) = 3 (find f(8)) against additive f(x + y) = f(x) + f(y) (show f(n) = n f(1))',
    frozenset({'x2_y2_3z2', 'classic_pyth_60'}): 'different Diophantine problems: x^2 + y^2 = 3z^2 has only the trivial solution (descent), against 60 | abc for Pythagorean triples (residues modulo 3, 4, 5)',
    frozenset({'find_all_xy_eq_x_plus_y', 'amc_dio_01'}): 'two Diophantine equations solved by the same factoring trick (xy = x + y against xy + x + y = 20), different equations and different answer sets',
}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ layer 3: the words
_STOP = frozenset('show that for all every any integers integer positive non negative natural numbers number a an the of and if then is are be to in with prove find what determine'.split())
_SHORT = frozenset('as at by do it no or so we how may can let who its too but per yes not has had was one two six ten odd sum mod gcd lcm max min set let any all eve'.split())      # (short words that are not products of variables)


def normalised_words(text):
    """the words of a problem text: lower case, punctuation dropped, variables renamed by order of appearance (a word of up to three letters that is not a word of the language is a product of variables: `2ab`
    is 2, a, b), filler words dropped"""
    toks = re.findall(r"[a-z]+|\d+|[^\sa-z\d]", re.sub(r'[,.;:]', ' ', text.lower()))
    names = {}; out = []

    def var(x):
        if x not in names: names[x] = 'v%d' % len(names)
        return names[x]
    for x in toks:
        if len(x) == 1 and x.isalpha(): out.append(var(x))
        elif x in _STOP: continue
        elif x.isalpha() and len(x) <= 3 and x not in _SHORT: out.extend(var(ch) for ch in x)
        else: out.append(x)
    return out


def text_pairs(problems, ratio=TEXT_RATIO):
    """[(ratio, a, b)] for the pairs of problems whose texts are at least `ratio` alike, most alike first.  `problems`: {name: object with .text}"""
    words = {n: normalised_words(p.text) for n, p in problems.items()}
    out = []
    for a, b in itertools.combinations(list(problems), 2):
        wa, wb = words[a], words[b]
        if not wa or not wb: continue
        if len(set(wa) & set(wb)) < 0.5 * min(len(set(wa)), len(set(wb))): continue                  # (a cheap filter: most pairs share nothing)
        r = difflib.SequenceMatcher(None, wa, wb).ratio()
        if r >= ratio: out.append((round(r, 3), a, b))
    return sorted(out, reverse=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ layer 2: the meaning
def _grid(kinds):
    return list(itertools.product(*[range(-3, 4) if k == 'Z' else range(0, 7) for k in kinds]))


def _tables(sem, ast, ref_kinds, perm):
    """the tables of a statement at the grid of `ref_kinds`, its parameter j taking the value of the reference parameter perm[j]; None if the evaluator cannot read it"""
    pts = [tuple(p[perm[j]] for j in range(len(perm))) for p in _grid(ref_kinds)]
    try:
        _, tabs, atoms = SEM._tables(sem, ast, pts, hyp_atoms=True)
    except (SEM.Unsupported, SEM.ShapeChange, ZeroDivisionError, OverflowError, RecursionError, ValueError):
        return None
    return tabs, [(a.kind, a.binders, a.cols) for a in atoms]


def _same(x, y):
    if x is None or y is None: return False
    (tx, ax), (ty, ay) = x, y
    if tx != ty or len(ax) != len(ay): return False
    for (k1, b1, c1), (k2, b2, c2) in zip(ax, ay):
        if k1 != k2 or b1 != b2: return False
        if c1 != c2 and not (k1 == 'eq' and c1[0] == c2[1] and c1[1] == c2[0]): return False             # (an equation may be written either way round)
    return True


def readable(bk, names=None):
    """{name: parameter kinds} of the statements that this module can compare: universally quantified integers / naturals (1 to 4 of them), read by the evaluator"""
    sem = bk.semantics; out = {}
    for n in (names if names is not None else bk.placed):
        ast = bk.statement_ast(n)
        if ast is None: continue
        try: kinds, _, _ = sem.parts(ast)
        except SEM.Unsupported: continue
        if 1 <= len(kinds) <= 4 and all(k in ('Z', 'N') for k in kinds): out[n] = tuple(kinds)
    return out


def meaning_pairs(bk, names=None):
    """[(a, b, perm)]: the pairs of problems of the loaded bank `bk` whose statements are the same up to the order and the names of their parameters (perm: the parameter j of b is the parameter perm[j] of a)"""
    sem = bk.semantics; kinds = readable(bk, names); groups = collections.defaultdict(list)
    for n, ks in kinds.items(): groups[tuple(sorted(ks))].append(n)
    out = []; cache = {}
    def tab(n, ref, perm):
        key = (n, ref, perm)
        if key not in cache: cache[key] = _tables(sem, bk.statement_ast(n), ref, perm)
        return cache[key]
    for names_ in groups.values():
        for a, b in itertools.combinations(names_, 2):
            ka, kb = kinds[a], kinds[b]; ident = tuple(range(len(ka)))
            ta = tab(a, ka, ident)
            if ta is None or not ta[1]: continue                                                    # (no relation atom to compare: nothing to say)
            for perm in itertools.permutations(range(len(kb))):
                if any(kb[j] != ka[perm[j]] for j in range(len(kb))): continue
                if _same(ta, tab(b, ka, perm)): out.append((a, b, perm)); break
    return out
