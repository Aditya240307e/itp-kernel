"""bank_split.py -- which problems of the bank a learner may train on.

A model that is evaluated on problems it trained on has learned the answers, not the mathematics.  The splits of the bank are therefore

  * by FAMILY, not by problem: the two parts of one olympiad problem, a statement and the same statement in other clothes (`4ab <= (a+b)^2` and `2ab <= a^2+b^2`), an argument and the same argument with
    another constant (the irrationality of sqrt 2 and of sqrt 3) belong together.  `FAMILIES` names those groups; every other problem is its own family.
  * a PURE FUNCTION of the family: the split of a family is a hash of its key under a fixed salt, so adding problems to the bank never moves an existing problem from one split to another (a model
    trained on version 1 of the dataset is still evaluated on problems it has not seen when version 2 comes).  70 % train, 10 % dev, 20 % test, in expectation.
  * CHECKED for coverage: every split has problems of the levels 1..3, the test split has olympiad (IMO) problems and several topics, ... (`coverage`).  The salt (`SALT_NUMBER`) was searched ONCE, as the smallest
    number whose split satisfies those requirements for the 82 problems of dataset version 1, and is frozen: it is part of the dataset, not a tuning knob.  A later version of the bank keeps it (its new
    problems fall where their hashes put them; the requirements are re-checked, and are about the whole bank, not about the new problems).

The library theorems the learner may also imitate (proofs of lemmas written for the library, not for the bank) are split the same way, by their own name (`lemma_split`).

    import bank_split as S
    S.split_of('imo1959_p1')          # 'train' | 'dev' | 'test'
    S.splits(bank_all.names())        # {'train': [..], 'dev': [..], 'test': [..]}
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import hashlib, collections

DATASET_VERSION = 1                   # informational: the splits do not depend on it (they must not move when the bank grows)
SALT_NUMBER = 25                      # FROZEN: found once, for the 82 problems of dataset version 1 (`find_salt`); never changed afterwards
FRACTIONS = (('train', 70), ('dev', 10), ('test', 20))      # percent

FAMILIES = {
    # the two parts of one olympiad problem
    'imo1964_p1a': 'imo1964_p1', 'imo1964_p1b': 'imo1964_p1',
    # the same statement in other clothes
    'ineq_am_gm_two': 'ineq_am_gm', 'ineq_four_ab': 'ineq_am_gm',
    'ineq_square_sum_three': 'ineq_squares_sum', 'ineq_three_squares': 'ineq_squares_sum',
    'ineq_fourth_powers_three': 'ineq_fourth', 'ineq_four_fourth_powers': 'ineq_fourth',
    # one argument with another constant or one more variable
    'sqrt2_irrational': 'irrational_descent', 'sqrt3_irrational': 'irrational_descent',
    'ident_cube_sum': 'ident_cubes', 'ident_cube_diff': 'ident_cubes', 'ident_three_cubes': 'ident_cubes',
    'ineq_cubes_two': 'ineq_cubes', 'ineq_cubes_three': 'ineq_cubes',
    'ineq_cauchy_two': 'ineq_cauchy', 'ineq_cauchy_three': 'ineq_cauchy',
    'n5_minus_n_div30': 'n5_minus_n',                        # (five_dvd_fifth_power, the other member, became a library lemma)
    'five_dvd_six_pow_minus_one': 'dvd_by_induction', 'three_dvd_four_pow_plus_five': 'dvd_by_induction',
    'no_square_between': 'between_squares', 'no_square_pronic': 'between_squares',
}


def family(name): return FAMILIES.get(name, name)


def _bucket(key, salt_number=None):
    h = hashlib.sha256(('itp-bank-split/%d:%s' % (SALT_NUMBER if salt_number is None else salt_number, key)).encode()).hexdigest()
    return int(h[:12], 16) % 100


def split_of(name, salt_number=None):
    """the split ('train', 'dev' or 'test') of a problem of the bank: a function of its family alone"""
    b = _bucket(family(name), salt_number); acc = 0
    for sp, pct in FRACTIONS:
        acc += pct
        if b < acc: return sp
    return FRACTIONS[-1][0]


def lemma_split(name, salt_number=None):
    """the split of a theorem of the library (a lemma the learner may imitate); a function of its name alone"""
    b = _bucket('lemma:' + name, salt_number); acc = 0
    for sp, pct in FRACTIONS:
        acc += pct
        if b < acc: return sp
    return FRACTIONS[-1][0]


def splits(names, salt_number=None):
    out = collections.OrderedDict((sp, []) for sp, _ in FRACTIONS)
    for n in names: out[split_of(n, salt_number)].append(n)
    return out


def straddling(assign):
    """the family keys that occur in more than one split of `assign` ({problem name: split}): a family must not (that is leakage between train and test)"""
    seen = collections.defaultdict(set)
    for n, sp in assign.items(): seen[family(n)].add(sp)
    return sorted(f for f, sps in seen.items() if len(sps) > 1)


def coverage(bank_problems, salt_number=None):
    """the requirements on a split of `bank_problems` ({name: bank.Problem}); the list of what is missing (empty: the split is acceptable)"""
    sp = splits(list(bank_problems), salt_number); missing = []
    byl = {s: collections.Counter(bank_problems[n].level for n in ns) for s, ns in sp.items()}
    top = {s: set(t for n in ns for t in bank_problems[n].topics) for s, ns in sp.items()}
    imo = {s: [n for n in ns if bank_problems[n].source.startswith('IMO')] for s, ns in sp.items()}
    if len(sp['train']) < 0.62 * len(bank_problems): missing.append('train has %d problems' % len(sp['train']))
    if len(sp['dev']) < 6: missing.append('dev has %d problems' % len(sp['dev']))
    if len(sp['test']) < 12: missing.append('test has %d problems' % len(sp['test']))
    for s, lo in (('train', 3), ('dev', 1), ('test', 2)):
        for lv in (1, 2, 3):
            if byl[s][lv] < lo: missing.append('%s has %d problems of level %d' % (s, byl[s][lv], lv))
    if not any(byl['dev'][lv] + byl['test'][lv] for lv in (4, 5)): missing.append('no problem of level 4 or 5 outside train')
    if len(imo['test']) < 2: missing.append('test has %d IMO problems' % len(imo['test']))
    if len(imo['train']) < 2: missing.append('train has %d IMO problems' % len(imo['train']))
    for t in ('number-theory', 'inequality', 'algebra'):
        for s in ('train', 'dev', 'test'):
            if t not in top[s]: missing.append('%s has no problem on %s' % (s, t))
    if 'functional-equation' not in top['train'] or 'functional-equation' not in top['test']: missing.append('functional equations are not in both train and test')
    if len(top['test']) < 6: missing.append('test covers %d topics' % len(top['test']))
    return missing


def find_salt(bank_problems, limit=400):
    """the smallest salt number whose split of the bank satisfies `coverage`"""
    for k in range(1, limit):
        if not coverage(bank_problems, k): return k
    return None


def table(bank_problems, salt_number=None):
    sp = splits(list(bank_problems), salt_number); rows = []
    for s, ns in sp.items():
        lv = collections.Counter(bank_problems[n].level for n in ns)
        rows.append('%-5s %3d problems  levels %s  IMO %d' % (s, len(ns), ' '.join('%d:%d' % (k, lv[k]) for k in sorted(lv)), sum(1 for n in ns if bank_problems[n].source.startswith('IMO'))))
    return '\n'.join(rows)


if __name__ == '__main__':
    import bank_all
    from bank import BANK
    P = BANK.problems
    k = find_salt(P)
    print('smallest salt number satisfying the requirements:', k, ' (the module uses %d)' % SALT_NUMBER)
    print(table(P, SALT_NUMBER)); print('missing with the module salt:', coverage(P, SALT_NUMBER))
    for s, ns in splits(list(P)).items(): print(s, ns)
