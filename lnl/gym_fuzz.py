"""gym_fuzz.py -- a falsification fuzzer for the tactic layer, the unifier and the kernel's final check, driven through the learner's environment (gym_env.py, gym_search.py).

WHY.  Everything the tactic layer builds is checked by the kernel at the end, so a bug in a tactic or in the unifier cannot make a false theorem true -- it makes a proof that is REJECTED. Such a rejection is
a disagreement between two layers and always a bug (S36 found two this way: `cases` on `S (S 0) = S 0` and the unifier's treatment of `?x + 1` after `?x` was solved). They are found by statements that are
FALSE: a sound tactic layer cannot finish a proof of one, so a search that "finishes" one has exposed a step that closed a goal it should not have. This module generates statements about the natural numbers (or the
integers), decides by evaluation which are true, runs the search on every one, and reports
  REJECTED  a proof the search finished that the kernel then refused   (a bug of the tactic layer or of the unifier; the action tree is printed)
  UNSOUND   a proof of a statement that is false at some point        (a bug of the KERNEL: should be impossible)
  ERROR     any other exception out of the engine or the search      (a bug of the environment)
  PROVED / FAILED are the ordinary outcomes (a true statement that was proved / not proved within the budget).

STATEMENTS.  Random polynomial-ish terms (variables, numerals, +, *, successor, powers by a numeral, predecessor), as equations or inequalities; true ones are made by applying equality-preserving rewrites
(commutativity, associativity, distributivity, neutral elements, unfolding the successor) to a random term; false ones by a small perturbation of a true one (a changed numeral, a swapped variable, a dropped
term) -- the near misses that a search is most likely to close wrongly.

THE INTEGERS (--world int): equalities that ring settles (a polynomial identity, or one made false by a small change), polynomial inequalities over a bounded interval (the bounds and the polynomial's true maximum
decide the claim, which is made one too tight half of the time: interval_cases and nlinarith), and linear consequences of linear hypotheses (a non-negative combination of them, or a combination shifted by one: linarith).

    python3 gym_fuzz.py --seed 1 --n 300 [--world nat|int] [--noarith] [--depth 5] [--probes 400] [--seconds 5] [--log fuzz.log]
"""
import argparse, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(200000)
import gym_env as G, gym_search as S
from author import AuthorError

G.STRICT_DEFAULT = True        # a Python error inside a probe is a bug of the environment: it must surface (as an ERROR), not be counted as a refused action

MAX_VALUE = 10 ** 6


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ expressions
# ('var', i) ('num', n) ('add', x, y) ('mul', x, y) ('succ', x) ('pow', x, n) ('pred', x)      (evaluated on the natural numbers)
def evaluate(e, env):
    k = e[0]
    if k == 'var': return env[e[1]]
    if k == 'num': return e[1]
    if k == 'add': return evaluate(e[1], env) + evaluate(e[2], env)
    if k == 'mul': return evaluate(e[1], env) * evaluate(e[2], env)
    if k == 'succ': return evaluate(e[1], env) + 1
    if k == 'pow': return evaluate(e[1], env) ** e[2]
    if k == 'pred': return max(evaluate(e[1], env) - 1, 0)
    raise ValueError(k)


def show(e):
    k = e[0]
    if k == 'var': return 'abcdef'[e[1]]
    if k == 'num': return str(e[1])
    if k == 'add': return '(%s + %s)' % (show(e[1]), show(e[2]))
    if k == 'mul': return '(%s * %s)' % (show(e[1]), show(e[2]))
    if k == 'succ': return 'S %s' % show(e[1])
    if k == 'pow': return '%s^%d' % (show(e[1]), e[2])
    if k == 'pred': return 'pred %s' % show(e[1])


def size(e): return 1 + sum(size(x) for x in e[1:] if isinstance(x, tuple))


def random_expr(rng, depth, nvars, allow=('add', 'mul', 'succ', 'pow', 'pred')):
    if depth == 0 or rng.random() < 0.25:
        return ('var', rng.randrange(nvars)) if rng.random() < 0.7 else ('num', rng.choice([0, 1, 1, 2, 3]))
    k = rng.choice(allow)
    if k in ('add', 'mul'): return (k, random_expr(rng, depth - 1, nvars, allow), random_expr(rng, depth - 1, nvars, allow))
    if k == 'pow': return ('pow', random_expr(rng, depth - 1, nvars, allow), rng.choice([2, 2, 3]))
    return (k, random_expr(rng, depth - 1, nvars, allow))


def positions(e, path=()):
    yield path
    for i, x in enumerate(e[1:], 1):
        if isinstance(x, tuple): yield from positions(x, path + (i,))


def get(e, path):
    for i in path: e = e[i]
    return e


def put(e, path, new):
    if not path: return new
    l = list(e); l[path[0]] = put(e[path[0]], path[1:], new); return tuple(l)


def rewrites(e):
    """the expressions equal to e (as functions of the variables) that one law gives at its root"""
    k = e[0]; out = []
    if k == 'add':
        x, y = e[1], e[2]
        out.append(('add', y, x))
        if y[0] == 'add': out.append(('add', ('add', x, y[1]), y[2]))
        if x[0] == 'add': out.append(('add', x[1], ('add', x[2], y)))
        if y == ('num', 0): out.append(x)
        if x == ('num', 0): out.append(y)
        if y[0] == 'num' and y[1] > 0: out.append(('succ', ('add', x, ('num', y[1] - 1))))
        if y[0] == 'succ': out.append(('succ', ('add', x, y[1])))
        if x[0] == 'succ': out.append(('succ', ('add', x[1], y)))
    elif k == 'mul':
        x, y = e[1], e[2]
        out.append(('mul', y, x))
        if y[0] == 'add': out.append(('add', ('mul', x, y[1]), ('mul', x, y[2])))
        if x[0] == 'add': out.append(('add', ('mul', x[1], y), ('mul', x[2], y)))
        if y == ('num', 1): out.append(x)
        if x == ('num', 1): out.append(y)
        if y == ('num', 0) or x == ('num', 0): out.append(('num', 0))
        if y[0] == 'mul': out.append(('mul', ('mul', x, y[1]), y[2]))
        if x == y: out.append(('pow', x, 2))
    elif k == 'succ':
        out.append(('add', e[1], ('num', 1)))
        out.append(('add', ('num', 1), e[1]))
    elif k == 'pow':
        x, n = e[1], e[2]
        if n == 2: out.append(('mul', x, x))
        if n == 3: out.append(('mul', x, ('mul', x, x)))
    elif k == 'pred':
        if e[1][0] == 'succ': out.append(e[1][1])
    elif k == 'num' and e[1] > 0:
        out.append(('succ', ('num', e[1] - 1)))
    return out


def rewrite_somewhere(rng, e, tries=12):
    ps = list(positions(e))
    for _ in range(tries):
        p = rng.choice(ps); sub = get(e, p); rs = rewrites(sub)
        if rs: return put(e, p, rng.choice(rs))
    return e


def perturb(rng, e, nvars):
    ps = list(positions(e)); p = rng.choice(ps); sub = get(e, p)
    choice = rng.randrange(5)
    if sub[0] == 'num': new = ('num', sub[1] + rng.choice([1, 2, -1]) if sub[1] > 0 else 1)
    elif sub[0] == 'var': new = ('var', (sub[1] + 1) % nvars) if nvars > 1 and choice < 3 else ('succ', sub)
    elif sub[0] in ('add', 'mul') and choice == 0: new = sub[1]
    elif sub[0] in ('add', 'mul') and choice == 1: new = ('mul' if sub[0] == 'add' else 'add', sub[1], sub[2])
    elif sub[0] == 'succ' and choice < 2: new = sub[1]
    elif sub[0] == 'pow': new = ('pow', sub[1], sub[2] + 1)
    else: new = ('succ', sub)
    return put(e, p, new)


def grid(nvars):
    import itertools
    pts = list(itertools.product(range(5), repeat=nvars))
    rng = random.Random(7); pts += [tuple(rng.randrange(5, 60) for _ in range(nvars)) for _ in range(20)]
    return pts


def holds(rel, l, r, nvars):
    for env in grid(nvars):
        try: x, y = evaluate(l, env), evaluate(r, env)
        except (OverflowError, ValueError): return None
        if x > MAX_VALUE or y > MAX_VALUE: continue
        if (x != y) if rel == 'eq' else (x > y): return False
    return True


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ statements of a world
class NatWorld:
    name = 'nat'
    def __init__(self, w=None):
        if w is None:
            import nat_lib
            w = nat_lib.build_nat()
        self.w = w; self.T = w['Nat']
    def build(self, e, vs):
        w = self.w; k = e[0]
        if k == 'var': return vs[e[1]]
        if k == 'num':
            t = w['Zero']
            for _ in range(e[1]): t = w['Succ'](t)
            return t
        if k == 'add': return w['add'](self.build(e[1], vs), self.build(e[2], vs))
        if k == 'mul': return w['mul'](self.build(e[1], vs), self.build(e[2], vs))
        if k == 'succ': return w['Succ'](self.build(e[1], vs))
        if k == 'pow': return w['pow'](self.build(e[1], vs), self.build(('num', e[2]), vs))
        if k == 'pred': return w['pred'](self.build(e[1], vs))
    def statement(self, rel, l, r, nvars):
        w = self.w
        def body(*vs):
            a, b = self.build(l, vs), self.build(r, vs)
            return w.Eq(a, b) if rel == 'eq' else w['Le'](a, b)
        return w.forall_([self.T] * nvars, body)
    def random(self, rng):
        rel, l, r, nvars, truth = random_statement(rng)
        return self.statement(rel, l, r, nvars), '%s %s %s (%d vars)' % (show(l), '=' if rel == 'eq' else '<=', show(r), nvars), truth
    def engine(self, noarith):
        return G.Engine(self.w, kinds=[x for x in G.Kind if x not in (G.Kind.RING, G.Kind.LINARITH, G.Kind.NLINARITH)] if noarith else None)
    def canaries(self):
        """(label, statement, the kind of action that must close it): the decision procedures must work in the harness, or 'nothing false was proved' says nothing about them"""
        a, b = ('var', 0), ('var', 1)
        return [('ring: (a + b)^2 = a^2 + 2ab + b^2', self.statement('eq', ('pow', ('add', a, b), 2), ('add', ('add', ('pow', a, 2), ('mul', ('num', 2), ('mul', a, b))), ('pow', b, 2)), 2), G.Kind.RING)]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the integers
# expressions: ('var', i) ('num', n) ('add', x, y) ('sub', x, y) ('mul', x, y) ('neg', x);  a claim is (hypotheses, goal), each (relation, left, right) with relation in eq le lt ne
def zeval(e, env):
    k = e[0]
    if k == 'var': return env[e[1]]
    if k == 'num': return e[1]
    if k == 'add': return zeval(e[1], env) + zeval(e[2], env)
    if k == 'sub': return zeval(e[1], env) - zeval(e[2], env)
    if k == 'mul': return zeval(e[1], env) * zeval(e[2], env)
    if k == 'neg': return -zeval(e[1], env)
    raise ValueError(k)


def zshow(e):
    k = e[0]
    if k == 'var': return 'xyz'[e[1]]
    if k == 'num': return str(e[1])
    if k in ('add', 'sub', 'mul'): return '(%s %s %s)' % (zshow(e[1]), {'add': '+', 'sub': '-', 'mul': '*'}[k], zshow(e[2]))
    if k == 'neg': return '-%s' % zshow(e[1])


def zrel(rel, a, b): return a == b if rel == 'eq' else a <= b if rel == 'le' else a < b if rel == 'lt' else a != b
RELSHOW = {'eq': '=', 'le': '<=', 'lt': '<', 'ne': '!='}


def zrewrites(e):
    k = e[0]; out = []
    if k == 'add':
        x, y = e[1], e[2]; out += [('add', y, x)]
        if y[0] == 'add': out.append(('add', ('add', x, y[1]), y[2]))
        if x[0] == 'add': out.append(('add', x[1], ('add', x[2], y)))
        if y == ('num', 0): out.append(x)
        if y[0] == 'neg': out.append(('sub', x, y[1]))
        if y[0] == 'mul' and y[1][0] == 'neg': out.append(('sub', x, ('mul', y[1][1], y[2])))
    elif k == 'sub':
        x, y = e[1], e[2]; out += [('add', x, ('neg', y))]
        if y == ('num', 0): out.append(x)
        if x == y: out.append(('num', 0))
    elif k == 'mul':
        x, y = e[1], e[2]; out.append(('mul', y, x))
        if y[0] == 'add': out.append(('add', ('mul', x, y[1]), ('mul', x, y[2])))
        if x[0] == 'add': out.append(('add', ('mul', x[1], y), ('mul', x[2], y)))
        if y[0] == 'sub': out.append(('sub', ('mul', x, y[1]), ('mul', x, y[2])))
        if y == ('num', 1): out.append(x)
        if x == ('num', 1): out.append(y)
        if y[0] == 'neg': out.append(('neg', ('mul', x, y[1])))
        if x[0] == 'neg': out.append(('neg', ('mul', x[1], y)))
    elif k == 'neg':
        if e[1][0] == 'neg': out.append(e[1][1])
        if e[1][0] == 'add': out.append(('add', ('neg', e[1][1]), ('neg', e[1][2])))
        out.append(('sub', ('num', 0), e[1]))
    return out


def zrandom_expr(rng, depth, nvars, ops=('add', 'sub', 'mul', 'neg')):
    if depth == 0 or rng.random() < 0.25:
        return ('var', rng.randrange(nvars)) if rng.random() < 0.7 else ('num', rng.choice([-2, -1, 0, 1, 1, 2, 3]))
    k = rng.choice(ops)
    if k == 'neg': return ('neg', zrandom_expr(rng, depth - 1, nvars, ops))
    return (k, zrandom_expr(rng, depth - 1, nvars, ops), zrandom_expr(rng, depth - 1, nvars, ops))


def zrewrite_somewhere(rng, e, tries=12):
    ps = list(positions(e))
    for _ in range(tries):
        p = rng.choice(ps); sub = get(e, p); rs = zrewrites(sub)
        if rs: return put(e, p, rng.choice(rs))
    return e


def zperturb(rng, e, nvars):
    ps = list(positions(e)); p = rng.choice(ps); sub = get(e, p); c = rng.randrange(4)
    if sub[0] == 'num': new = ('num', sub[1] + rng.choice([1, -1, 2]))
    elif sub[0] == 'var': new = ('var', (sub[1] + 1) % nvars) if nvars > 1 and c < 2 else ('add', sub, ('num', 1))
    elif sub[0] in ('add', 'sub', 'mul') and c == 0: new = (rng.choice(['add', 'sub', 'mul']), sub[1], sub[2])
    elif sub[0] in ('add', 'sub', 'mul') and c == 1: new = sub[1]
    elif sub[0] == 'neg': new = sub[1]
    else: new = ('add', sub, ('num', 1))
    return put(e, p, new)


def zholds(hyps, goal, nvars, lo=-8, hi=8):
    """does the claim hold at every point of the grid [lo, hi]^nvars that satisfies the hypotheses?"""
    import itertools
    for env in itertools.product(range(lo, hi + 1), repeat=nvars):
        if all(zrel(r, zeval(a, env), zeval(b, env)) for r, a, b in hyps) and not zrel(goal[0], zeval(goal[1], env), zeval(goal[2], env)): return False
    return True


def zrandom_claim(rng):
    """(hypotheses, goal, number of variables): one of the three families"""
    fam = rng.choice(['ring', 'ring', 'bounded', 'bounded', 'linear', 'linear'])
    if fam == 'ring':
        nvars = rng.choice([1, 2, 2, 3]); base = zrandom_expr(rng, rng.choice([1, 2, 2, 3]), nvars); r = base
        for _ in range(rng.randrange(1, 4)): r = zrewrite_somewhere(rng, r)
        if rng.random() < 0.5: r = zperturb(rng, r, nvars)
        return [], ('eq', base, r), nvars
    if fam == 'bounded':
        nvars = rng.choice([1, 1, 2]); hyps = []; ranges = []
        for i in range(nvars):
            lo = rng.randint(-3, 2); hi = lo + rng.randint(1, 4); ranges.append((lo, hi))
            hyps += [('le', ('num', lo), ('var', i)), ('le', ('var', i), ('num', hi))]
        import itertools
        poly = zrandom_expr(rng, rng.choice([1, 2, 2]), nvars, ops=('add', 'sub', 'mul', 'mul', 'neg'))
        vals = [zeval(poly, env) for env in itertools.product(*[range(a, b + 1) for a, b in ranges])]
        kind = rng.choice(['le', 'le', 'lt', 'ne'])
        if kind == 'ne': c = rng.choice(vals) if rng.random() < 0.5 else max(vals) + rng.randint(1, 3)
        else: c = max(vals) + (0 if kind == 'le' else 1) + rng.choice([0, 0, -1, 1])
        return hyps, (kind, poly, ('num', c)), nvars
    nvars = rng.choice([2, 2, 3]); hyps = []; comb = None; tot_c = 0
    for _ in range(rng.randint(1, 3)):
        lin = None
        for i in range(nvars):
            co = rng.randint(-2, 2)
            if co: term = ('mul', ('num', co), ('var', i)); lin = term if lin is None else ('add', lin, term)
        if lin is None: lin = ('var', 0)
        c = rng.randint(-4, 6); rel = rng.choice(['le', 'le', 'lt', 'eq'])
        hyps.append((rel, lin, ('num', c)))
    goal_lhs = ('num', 0); goal_c = 0
    for (rel, lin, rhs) in hyps:
        m = rng.choice([0, 1, 1, 2])
        if m: goal_lhs = ('add', goal_lhs, ('mul', ('num', m), lin)); goal_c += m * rhs[1]
    goal_c += rng.choice([0, 0, 1, -1, 2]); kind = rng.choice(['le', 'le', 'lt'])
    return hyps, (kind, goal_lhs, ('num', goal_c)), nvars


class IntFamily:
    name = 'int'
    def __init__(self, w=None):
        if w is None:
            import int_lib
            w = int_lib.build_int()
        self.w = w; self.Z = w['Z']
    def term(self, e, vs):
        w = self.w; k = e[0]
        if k == 'var': return vs[e[1]]
        if k == 'num':
            with w.nota.using(self.Z): return w.nota.lit(e[1])
        if k == 'neg': return w['Z.neg'](self.term(e[1], vs))
        return w[{'add': 'Z.add', 'sub': 'Z.sub', 'mul': 'Z.mul'}[k]](self.term(e[1], vs), self.term(e[2], vs))
    def prop(self, rel, a, b):
        w = self.w
        if rel == 'eq': return w.Eq(a, b)
        if rel == 'le': return w['Z.le'](a, b)
        if rel == 'lt': return w['Z.lt'](a, b)
        return w.Not(w.Eq(a, b))
    def statement(self, hyps, goal, nvars):
        w = self.w; binders = [self.Z] * nvars
        for j, (rel, a, b) in enumerate(hyps):
            binders.append(lambda *prev, rel=rel, a=a, b=b: self.prop(rel, self.term(a, prev[:nvars]), self.term(b, prev[:nvars])))
        return w.forall_(binders, lambda *prev: self.prop(goal[0], self.term(goal[1], prev[:nvars]), self.term(goal[2], prev[:nvars])))
    def random(self, rng):
        while True:
            hyps, goal, nvars = zrandom_claim(rng)
            if sum(size(x) for _, a, b in hyps + [goal] for x in (a, b)) > 60: continue
            truth = zholds(hyps, goal, nvars)
            label = '%s => %s %s %s  (%d vars)' % (' & '.join('%s %s %s' % (zshow(a), RELSHOW[r], zshow(b)) for r, a, b in hyps) or 'true', zshow(goal[1]), RELSHOW[goal[0]], zshow(goal[2]), nvars)
            return self.statement(hyps, goal, nvars), label, truth
    def engine(self, noarith):
        e = G.Engine(self.w)
        for kind in (G.Kind.RING, G.Kind.LINARITH, G.Kind.NLINARITH, G.Kind.INTERVAL_CASES): e.probe_fuel[kind] = min(e.probe_fuel.get(kind, 5_000_000), 20_000_000)
        return e
    def canaries(self):
        x, y = ('var', 0), ('var', 1)
        return [('ring: (x + y)(x - y) = x*x - y*y', self.statement([], ('eq', ('mul', ('add', x, y), ('sub', x, y)), ('sub', ('mul', x, x), ('mul', y, y))), 2), G.Kind.RING),
                ('linarith: 0 <= x <= 3 -> x <= 4', self.statement([('le', ('num', 0), x), ('le', x, ('num', 3))], ('le', x, ('num', 4)), 1), G.Kind.LINARITH),
                ('linarith: -3 <= x <= -2 -> -x <= 3', self.statement([('le', ('num', -3), x), ('le', x, ('num', -2))], ('le', ('neg', x), ('num', 3)), 1), G.Kind.LINARITH),
                ('nlinarith: 1 <= x <= 4 -> x*x <= 16', self.statement([('le', ('num', 1), x), ('le', x, ('num', 4))], ('le', ('mul', x, x), ('num', 16)), 1), G.Kind.NLINARITH)]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the run
def describe(eng, stmt, tree, w):
    """the actions of a tree, replayed on a live proof, one per line with the goal they were taken on"""
    import s36_gym_tests as T
    lines = []; p = eng.start(stmt)
    def go(t, depth):
        g = eng.goal_of(p, 0); menu = eng.menu(g); a = menu.find(t.signature)
        if a is None: lines.append('  ' * depth + '<action not in the menu>'); return False
        lines.append('  ' * depth + T.action_name(w, menu, a) + '    |- ' + eng.render(g).replace('\n', ' ')[:200])
        try: eng.replay(p, a, menu)
        except Exception as ex: lines.append('  ' * depth + '<replay failed: %s>' % str(ex)[:100]); return False
        return all(go(c, depth + 1) for c in t.children)
    go(tree, 0)
    return '\n'.join(lines)


def random_statement(rng):
    """(relation, left, right, number of variables, true at every point of the grid) -- a random equation or inequality of the natural numbers, made by rewriting and perturbing"""
    while True:
        nvars = rng.choice([1, 2, 2, 3])
        base = random_expr(rng, rng.choice([1, 2, 2, 3]), nvars)
        l = base; r = base
        for _ in range(rng.randrange(1, 4)): r = rewrite_somewhere(rng, r)
        if rng.random() < 0.4: l = rewrite_somewhere(rng, l)
        rel = 'eq' if rng.random() < 0.8 else 'le'
        if rng.random() < 0.55: r = perturb(rng, r, nvars)                 # a near miss (usually false)
        if size(l) > 24 or size(r) > 24: continue
        truth = holds(rel, l, r, nvars)
        if truth is None: continue
        return rel, l, r, nvars, truth


def check_canaries(W, eng, noarith):
    """each canary statement must be closed by ONE probe of the decision procedure it is meant for (raises RuntimeError when the harness cannot use it)"""
    for label, stmt, kind in W.canaries():
        if noarith and kind in (G.Kind.RING, G.Kind.LINARITH, G.Kind.NLINARITH): continue
        p = eng.start(stmt)
        if p.done: continue
        g = eng.goal_of(p, 0); menu = eng.menu(g)
        a = next((a for a in menu.actions if a.kind == kind), None)
        if a is None: raise RuntimeError('canary %s: the menu does not offer %s' % (label, kind.name))
        out = eng.probe(g, a, menu)
        if out != []: raise RuntimeError('canary %s: %s did not close the goal (%r; errors %r)' % (label, kind.name, out, dict(eng.errors)))


def run(seed=1, n=200, world='nat', noarith=False, depth=5, probes=400, seconds=5.0, log=None, verbose=False, family=None):
    rng = random.Random(seed)
    W = family if family is not None else NatWorld() if world == 'nat' else IntFamily()
    world = W.name
    w = W.w
    eng = W.engine(noarith)
    check_canaries(W, eng, noarith)
    counts = {'PROVED': 0, 'FAILED': 0, 'REJECTED': 0, 'UNSOUND': 0, 'ERROR': 0}
    n_true = n_false = 0; reports = []; t0 = time.time()
    for i in range(n):
        stmt, label, truth = W.random(rng)
        n_true += truth; n_false += not truth
        outcome = None; detail = ''
        try:
            p = eng.start(stmt)
            if p.done: term = eng.verify(stmt, p); tree = None
            else:
                g = eng.goal_of(p, 0)
                s = S.Searcher(eng, max_probes=probes, time_limit=seconds); tree = s.prove(g, max_depth=depth)
                term = None
                if tree is not None:
                    try: term = S.commit(eng, stmt, tree)
                    except AuthorError as ex:
                        outcome = 'REJECTED'; detail = str(ex)[:300].replace('\n', ' ') + '\n' + describe(eng, stmt, tree, w)
        except Exception as ex:
            import traceback
            outcome = 'ERROR'; detail = '%s: %s\n%s' % (type(ex).__name__, str(ex)[:300], traceback.format_exc(limit=5))
        if outcome is None:
            if term is not None: outcome = 'PROVED' if truth else 'UNSOUND'
            else: outcome = 'FAILED'
        counts[outcome] += 1
        if outcome in ('REJECTED', 'UNSOUND', 'ERROR') or (verbose and (outcome != 'FAILED' or truth)):      # verbose: also the true statements that the search could not prove (a measure of its strength)
            reports.append((outcome, truth, label, detail))
            print('[%s] %s statement  %s\n%s' % (outcome, 'TRUE' if truth else 'FALSE', label, detail), flush=True)
    if eng.bugs:
        counts['ERROR'] += 1; reports.append(('ERROR', True, 'engine bugs', repr(eng.bugs[:3]))); print('[ERROR] the engine recorded bugs: %r' % (eng.bugs[:3],), flush=True)
    summary = 'gym_fuzz seed=%d world=%s noarith=%s: %d statements (%d true, %d false) in %.1f s: %s; probes %s' % (seed, world, noarith, sum(counts.values()), n_true, n_false, time.time() - t0, counts, dict(eng.stats))
    print(summary, flush=True)
    if log:
        with open(log, 'a') as f:
            f.write(summary + '\n')
            for r in reports: f.write('[%s] %s %s\n%s\n' % (r[0], 'TRUE' if r[1] else 'FALSE', r[2], r[3]))
    return counts, reports


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--n', type=int, default=200); ap.add_argument('--world', default='nat')
    ap.add_argument('--noarith', action='store_true'); ap.add_argument('--depth', type=int, default=5); ap.add_argument('--probes', type=int, default=400)
    ap.add_argument('--seconds', type=float, default=5.0); ap.add_argument('--log'); ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    counts, reports = run(a.seed, a.n, a.world, a.noarith, a.depth, a.probes, a.seconds, a.log, a.verbose)
    sys.exit(1 if counts['REJECTED'] or counts['UNSOUND'] or counts['ERROR'] else 0)
