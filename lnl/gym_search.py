"""gym_search.py -- proof search over the engine's goals (gym_env.py), and the replay of what it finds.

NOTHING HERE IS TRUSTED. The search only decides which actions to try; what it returns is a TREE of actions, and `commit` replays that tree on a live proof (the tactic layer builds the proof term) and has
the kernel check the finished term against the statement. A search that is wrong, or a replay that does not reproduce what the search saw, loses a proof; it cannot produce a false one.

THE SEARCH.  Goals are the nodes (identified by their kernel nodes: `GoalData.key`), an action is an edge to the list of goals it leaves, and a goal is solved when SOME action solves ALL the goals it
leaves (an AND-OR graph). This module is the plain version: depth-bounded depth-first search with iterative deepening, a table of the goals already solved or refuted (and to what depth), cycle
detection along the current path, and a bound on the number of probes. Its actions are tried in an order given by `rank` (cheap closing actions first by default); a learned policy replaces `rank`.
"""
import time
from gym_env import Engine, Kind, Action, ActionError, GoalData, NEEDS_CANDIDATE


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


DEFAULT_PRIORITY = {Kind.INTRO: 0, Kind.RING: 1, Kind.LINARITH: 2, Kind.APPLY: 3, Kind.REWRITE: 4, Kind.INTERVAL_CASES: 5, Kind.CASES: 6, Kind.INDUCT: 7, Kind.NLINARITH: 8, Kind.BY_CONTRA: 9}


def default_rank(eng, g, menu):
    """the actions of a menu in the order to try them: decision procedures, then hypotheses and lemmas, rewriting, case analysis, induction"""
    def key(ia):
        i, a = ia
        pri = DEFAULT_PRIORITY[a.kind]
        if a.kind == Kind.APPLY and menu.cands[a.arg][0] is not None and eng.k.view(menu.cands[a.arg][0])[0] == 'fvar': pri = 2.5       # a hypothesis before a lemma
        return (pri, i)
    return [a for _, a in sorted(enumerate(menu.actions), key=key)]


class Searcher:
    def __init__(self, eng, max_probes=5000, rank=default_rank, time_limit=None):
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


def prove_statement(eng, stmt, max_depth=8, max_probes=5000, rank=default_rank, time_limit=None):
    """search for a proof of a closed statement and commit it: (the proof term, the Tree, the Searcher) or (None, None, the Searcher)"""
    p = eng.start(stmt)
    if p.done: return commit_trivial(eng, stmt, p), None, None
    g = eng.goal_of(p, 0)
    s = Searcher(eng, max_probes=max_probes, rank=rank, time_limit=time_limit)
    tree = s.prove(g, max_depth=max_depth)
    if tree is None: return None, None, s
    return commit(eng, stmt, tree), tree, s


def commit_trivial(eng, stmt, p):
    return eng.verify(stmt, p)
