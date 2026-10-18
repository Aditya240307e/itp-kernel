"""gym_search.py -- proof search over the engine's goals (gym_env.py), and the replay of what it finds.

NOTHING HERE IS TRUSTED. The search only decides which actions to try; what it returns is a TREE of actions, and `commit` replays that tree on a live proof (the tactic layer builds the proof term) and has
the kernel check the finished term against the statement. A search that is wrong, or a replay that does not reproduce what the search saw, loses a proof; it cannot produce a false one.

THE SEARCH.  Goals are the nodes (identified by their kernel nodes: `GoalData.key`), an action is an edge to the list of goals it leaves, and a goal is solved when SOME action solves ALL the goals it
leaves (an AND-OR graph). This module is the plain version: depth-bounded depth-first search with iterative deepening, a table of the goals already solved or refuted (and to what depth), cycle
detection along the current path, and a bound on the number of probes. Its actions are tried in an order given by `rank` (cheap closing actions first by default); a learned policy replaces `rank`.
"""
import time
from gym_env import Engine, Kind, Action, ActionError, GoalData, NEEDS_CANDIDATE, ST_OK, ST_REFUSED, ST_FUEL, ST_BUG


class Tree:
    """a solved goal: the action that solved it, as a `Menu.signature` (so that it can be found again in the menu of the same goal), and the trees of the goals the action left, in order"""
    __slots__ = ('signature', 'children')
    def __init__(self, signature, children): self.signature = signature; self.children = children
    def size(self): return 1 + sum(c.size() for c in self.children)
    def depth(self): return 1 + max((c.depth() for c in self.children), default=0)
    def kinds(self):
        out = [self.signature[0]]
        for c in self.children: out += c.kinds()
        return out


class BudgetExhausted(Exception):
    pass


DEFAULT_PRIORITY = {Kind.INTRO: 0, Kind.RING: 1, Kind.LINARITH: 2, Kind.INJECT: 2.5, Kind.APPLY: 3, Kind.USE: 3.2, Kind.APPLY_WITH: 3.4, Kind.REFINE: 3.6, Kind.REWRITE: 4, Kind.REWRITE_WITH: 4.2,
                    Kind.REWRITE_AT: 4.5, Kind.INTERVAL_CASES: 5, Kind.CASES: 6, Kind.FORWARD: 6.4, Kind.INSTANTIATE: 6.5, Kind.INDUCT: 7, Kind.INDUCT_WITH: 7.2, Kind.LINCOMB: 7.6, Kind.NLINARITH: 8,
                    Kind.BY_CONTRA: 9, Kind.CLAIM: 9.5}


def default_rank(eng, g, menu):
    """the actions of a menu in the order to try them: decision procedures, then hypotheses and lemmas, rewriting, case analysis, induction"""
    def key(ia):
        i, a = ia
        pri = DEFAULT_PRIORITY[a.kind]
        if a.kind == Kind.APPLY and menu.cands[a.arg][0] is not None and eng.k.view(menu.cands[a.arg][0])[0] == 'fvar': pri = 2.5       # a hypothesis before a lemma
        return (pri, i)
    return [a for _, a in sorted(enumerate(menu.actions), key=key)]


class _SearcherBase:
    """what the searchers share: the engine, the tables of what is solved / refuted / seen, the menus of goals (made once) and the budgets"""
    def __init__(self, eng, max_probes, rank, time_limit):
        self.eng = eng; self.max_probes = max_probes; self.rank = rank; self.time_limit = time_limit
        self.solved = {}; self.failed = {}; self.menus = {}; self.path = set(); self.probes = 0; self.t0 = None

    def menu_of(self, g):
        m = self.menus.get(g.key)
        if m is None: m = self.menus[g.key] = self.eng.menu(g)
        return m

    def _tick(self):
        self.probes += 1
        if self.probes > self.max_probes: raise BudgetExhausted('probes')
        if self.time_limit is not None and time.time() - self.t0 > self.time_limit: raise BudgetExhausted('time')


class Searcher(_SearcherBase):
    def __init__(self, eng, max_probes=5000, rank=default_rank, time_limit=None):
        _SearcherBase.__init__(self, eng, max_probes, rank, time_limit)

    def solve(self, g, depth):
        """a Tree that solves goal g using at most `depth` more actions along any branch, or None"""
        hit = self.solved.get(g.key)
        if hit is not None: return hit
        if depth <= 0 or self.failed.get(g.key, -1) >= depth or g.key in self.path: return None
        self.path.add(g.key)
        try:
            menu = self.menu_of(g)
            for action in self.rank(self.eng, g, menu):
                self._tick()
                children = self.eng.probe(g, action, menu)
                if children is None: continue
                trees = []
                for c in children:
                    t = self.solve(c, depth - 1)
                    if t is None: break
                    trees.append(t)
                else:
                    tree = Tree(menu.signature(action), trees)
                    self.solved[g.key] = tree
                    return tree
            self.failed[g.key] = max(self.failed.get(g.key, -1), depth)
            return None
        finally:
            self.path.discard(g.key)

    def prove(self, g, max_depth=8, start_depth=1):
        """iterative deepening: the first Tree found, searching to depth start_depth, start_depth + 1, ... max_depth; None if there is none within the budget"""
        self.t0 = time.time()
        try:
            for d in range(start_depth, max_depth + 1):
                t = self.solve(g, d)
                if t is not None: return t
        except BudgetExhausted:
            return None
        return None


KIND_COST = {Kind.INTRO: 1, Kind.RING: 1, Kind.LINARITH: 1, Kind.APPLY: 1, Kind.USE: 1, Kind.REWRITE: 1, Kind.CASES: 1, Kind.INTERVAL_CASES: 2, Kind.INSTANTIATE: 2, Kind.INDUCT: 2, Kind.NLINARITH: 3, Kind.BY_CONTRA: 2,
             Kind.CLAIM: 2, Kind.INDUCT_WITH: 2, Kind.FORWARD: 2, Kind.INJECT: 1, Kind.REWRITE_AT: 1, Kind.REFINE: 1, Kind.APPLY_WITH: 1, Kind.REWRITE_WITH: 1, Kind.LINCOMB: 2}


def default_cost(eng, g, menu, action):
    """what an action costs a search, before the work it needs is counted: a number of steps (>= 1). The terms an action names cost more the later they are in the pool (the pool lists the simplest first)."""
    c = KIND_COST[action.kind]
    for i in action.args: c += (i + 1).bit_length() - 1
    c += (sum(len(code) for code in action.codes) + 3) // 4                  # the terms an action says cost a step for every four tokens (a source is one token)
    if action.kind == Kind.REWRITE:
        term = menu.cands[action.arg][0]
        if eng.k.view(term)[0] == 'const': c += 1                       # a lemma of the library is applied to whatever the goal offers: more ways to go astray than an equation among the hypotheses
        if menu.cand_flags[action.arg] & 2: c += 2                      # a permutative equation walks through the orders of a term
    elif action.kind == Kind.APPLY:
        v = eng.k.view(menu.cands[action.arg][0])
        if v[0] == 'const' and not eng.introduces(v[1]): c += 1         # likewise: a hypothesis, or the introduction rule of a connective, is a step with a purpose; any other lemma is a guess
    return c


class CostSearcher(_SearcherBase):
    """iterative deepening on COST (IDA*): every action has a cost -- `cost(eng, g, menu, action)` steps, plus `fuel_step` steps for each time the work budget of its probe had to be raised -- and a pass with
    budget B explores every proof whose cost along any branch is at most B, B = 1, 2, ... So cheap actions are tried before expensive ones, the terms at the front of the pool before those at the back,
    and an action that needs a big work budget only when nothing cheaper has worked. Everything a probe has told is kept (`cache`: the goals an action left, that it was refused, or the budget it ran
    out at), so a deeper pass repeats only the walk, not the probes; goals are compared up to the order of the hypotheses nothing refers to (`Engine.canon_key`) when remembering a failure. The search
    stops when a pass skipped nothing for lack of budget: the space is exhausted.
    A learned policy replaces `cost` (a negative log-probability) and `rank`."""
    def __init__(self, eng, max_probes=20000, rank=default_rank, cost=default_cost, time_limit=None, fuel_step=1):
        _SearcherBase.__init__(self, eng, max_probes, rank, time_limit)
        self.cost = cost; self.fuel_step = fuel_step; self.cache = {}; self.cuts = 0; self.passes = 0

    def surcharge(self, level):
        """what it costs to have needed the work budget of this rung of the ladder (0 is the smallest)"""
        return self.fuel_step * level

    def attempt(self, g, menu, action, rem):
        """(the goals the action leaves, what it costs) when it applies within the budget `rem`, else None. A cut (something skipped for lack of budget) is counted in `self.cuts`."""
        sig = menu.signature(action); key = (g.key, sig)
        ent = self.cache.get(key)
        base = self.cost(self.eng, g, menu, action)
        if ent is not None:
            if ent[0] == ST_REFUSED: return None
            if ent[0] == ST_OK:
                c = base + self.surcharge(ent[1])
                if c > rem: self.cuts += 1; return None
                return ent[2], c
        level = 0 if ent is None else ent[1] + 1
        budgets = self.eng.fuel_levels(action.kind)
        while level < len(budgets):
            c = base + self.surcharge(level)
            if c > rem: self.cuts += 1; return None
            self._tick()
            children, status = self.eng.probe_status(g, action, menu, budgets[level])
            if status == ST_OK: self.cache[key] = (ST_OK, level, children); return children, c
            if status == ST_FUEL: self.cache[key] = (ST_FUEL, level); level += 1; continue
            self.cache[key] = (ST_REFUSED,); return None
        self.cache[key] = (ST_REFUSED,)                    # not even the largest budget was enough
        return None

    def solve(self, g, rem):
        """a Tree that solves goal g at a cost of at most `rem` along any branch, or None"""
        hit = self.solved.get(g.key)
        if hit is not None: return hit
        ck = self.eng.canon_key(g)
        if rem <= 0: self.cuts += 1; return None
        f = self.failed.get(ck)
        if f is not None and f[0] >= rem:
            if f[1]: self.cuts += 1
            return None
        if ck in self.path: return None
        self.path.add(ck); cuts0 = self.cuts
        try:
            menu = self.menu_of(g)
            for action in self.rank(self.eng, g, menu):
                res = self.attempt(g, menu, action, rem)
                if res is None: continue
                children, c = res
                trees = []
                for ch in children:
                    t = self.solve(ch, rem - c)
                    if t is None: break
                    trees.append(t)
                else:
                    tree = Tree(menu.signature(action), trees)
                    self.solved[g.key] = tree
                    return tree
            if f is None or f[0] < rem: self.failed[ck] = (rem, self.cuts > cuts0)
            return None
        finally:
            self.path.discard(ck)

    def prove(self, g, max_cost=16, start_cost=1):
        """the first Tree found by passes with budgets start_cost, start_cost + 1, ... max_cost; None if there is none within the limits (or the whole space was searched)"""
        self.t0 = time.time()
        try:
            for b in range(start_cost, max_cost + 1):
                self.cuts = 0; self.passes += 1
                t = self.solve(g, b)
                if t is not None: return t
                if self.cuts == 0: return None
        except BudgetExhausted:
            return None
        return None


def commit(eng, stmt, tree):
    """replay a Tree on a live proof of the statement and have the kernel check the result: the proof term. Raises ActionError if the replay does not reproduce what the search saw."""
    p = eng.start(stmt)
    def go(t):
        g = eng.goal_of(p, 0); menu = eng.menu(g)
        a = menu.find(t.signature)
        if a is None: raise ActionError('the action %r is not in the menu of the live goal' % (t.signature[0],))
        n_before = p.n_goals
        left = eng.replay(p, a, menu)
        if left != len(t.children): raise ActionError('the live goal left %d goals, the search saw %d' % (left, len(t.children)))
        for c in t.children: go(c)
    go_roots = 1 if p.n_goals else 0
    if p.n_goals != 1 and tree is not None: raise ActionError('the live proof has %d goals at the start' % p.n_goals)
    go(tree)
    if not (p.done and p.ps.is_complete()): raise ActionError('the replay left goals open')
    return eng.verify(stmt, p)


def prove_statement(eng, stmt, max_depth=8, max_probes=5000, rank=default_rank, time_limit=None, cost=None):
    """search for a proof of a closed statement and commit it: (the proof term, the Tree, the Searcher) or (None, None, the Searcher). With `cost` (a function, or True for `default_cost`) the search is
    a CostSearcher with `max_depth` as the largest cost; otherwise plain depth-bounded iterative deepening."""
    p = eng.start(stmt)
    if p.done: return commit_trivial(eng, stmt, p), None, None
    g = eng.goal_of(p, 0)
    if cost is None:
        s = Searcher(eng, max_probes=max_probes, rank=rank, time_limit=time_limit)
        tree = s.prove(g, max_depth=max_depth)
    else:
        s = CostSearcher(eng, max_probes=max_probes, rank=rank, cost=default_cost if cost is True else cost, time_limit=time_limit)
        tree = s.prove(g, max_cost=max_depth)
    if tree is None: return None, None, s
    return commit(eng, stmt, tree), tree, s


def commit_trivial(eng, stmt, p):
    return eng.verify(stmt, p)
