"""ml_mcts.py -- Monte-Carlo tree search over proof states, guided by the policy network (a PUCT search in the style of AlphaZero, for one agent and a sparse reward).

    searcher = MCTSSearcher(eng, net, ctx, max_probes=4000, seed=0)
    tree = searcher.prove(goal)                       # a gym_search.Tree or None;  searcher.probes, .simulations, .rollouts
    term, tree, s = prove_statement(eng, stmt, premises, net, ctx, max_probes=4000)      # commits the tree: the kernel-checked proof term

THE TREE.  A node is a PROOF STATE: the list of the goals still open (the engine always acts on the first one); an edge is an action on the first goal; the state it leads to is the list of goals the action
left, followed by the goals that were waiting.  States are shared (a transposition table on the exact goals): two orders of closing the same goals meet in one node, which is what makes a search over proof
states a search over an AND-OR graph.  The terminal state is the empty list: the proof is finished.

ONE SIMULATION.  (1) SELECT from the root: at a node, widen it (progressive widening: the number of actions considered grows with the node's visits, `widen_c * n^widen_alpha`; the candidates come from
the policy in decreasing probability, `ml_search.ActionStream`), then take the edge with the best  Q + c_puct * P * sqrt(N) / (1 + n)  where P is the policy's probability of the action renormalised over
the edges considered.  The first time an edge is taken the engine PROBES the action (one counted probe; the fuel ladder of gym_env): a refused action is a dead edge, the others lead to a state.  (2) EVALUATE the
leaf: by a ROLLOUT -- the policy samples actions from the leaf (at most `rollout_depth`; every probe is counted) and the value is 1 when the rollout finishes the proof, else 0 -- or by the size head
(`value='critic'`): exp(-expected remaining actions / `value_scale`).  (3) BACK UP the value along the path.  A finished proof ends the search at once.  Dead ends propagate exactly: a state all of whose edges
are dead (the candidate stream exhausted) is dead, and so is an edge to a dead state; a state on the path being searched is never entered again.

PROBES are the unit of cost: every probe the selection or a rollout makes is counted (`max_probes`), the same unit as the other searchers and the agents that act.  A proof found is replayed on a live proof and
checked by the kernel (`gym_search.commit`); nothing else is trusted.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import time, math
import numpy as np
import gym_env as G, gym_search as S
from gym_env import ST_OK, ST_REFUSED, ST_FUEL
import ml_search as MS, ml_pol as POL


class Edge:
    __slots__ = ('action', 'sig', 'cost', 'prior', 'child', 'n', 'w', 'dead', 'solved', 'n_children')
    def __init__(self, action, sig, cost):
        self.action = action; self.sig = sig; self.cost = cost; self.prior = 0.0; self.child = None; self.n = 0; self.w = 0.0; self.dead = False; self.solved = False; self.n_children = 0


class State:
    __slots__ = ('goals', 'key', 'edges', 'taken', 'n', 'w', 'dead', 'solved', 'trail', 'on_path')
    def __init__(self, goals):
        self.goals = goals; self.key = tuple(g.key for g in goals); self.edges = []; self.taken = 0; self.n = 0; self.w = 0.0
        self.dead = False; self.solved = not goals; self.trail = None; self.on_path = False


def tree_from_preorder(seq):
    """a gym_search.Tree from the actions of a proof in the order the engine takes them: [(signature, number of goals it left)]"""
    it = iter(seq)
    def build():
        sig, k = next(it)
        return S.Tree(sig, [build() for _ in range(k)])
    return build()


class MCTSSearcher(MS.PolicySearcher):
    """PUCT over proof states (see the module documentation). Inherits the probes, caches and candidate streams of ml_search.PolicySearcher; `topk`, `cap`, `step_cost` and `temperature` are its."""
    def __init__(self, eng, net, ctx, max_probes=4000, time_limit=None, c_puct=1.5, widen_c=2.0, widen_alpha=0.5, rollout_depth=12, rollout_temperature=1.0, value='rollout', value_scale=8.0, seed=0,
                 topk=24, cap=9.0, temperature=1.0, fuel_step=1.0, rare=True):
        MS.PolicySearcher.__init__(self, eng, net, ctx, max_probes=max_probes, time_limit=time_limit, step_cost=0.0, topk=topk, cap=cap, temperature=temperature, fuel_step=fuel_step, rare=rare)
        if value not in ('rollout', 'critic'): raise ValueError("value: 'rollout' or 'critic'")
        self.c_puct = c_puct; self.widen_c = widen_c; self.widen_alpha = widen_alpha; self.rollout_depth = rollout_depth; self.rollout_temperature = rollout_temperature
        self.value = value; self.value_scale = value_scale; self.rng = np.random.default_rng(seed)
        self.table = {}; self.simulations = 0; self.rollouts = 0; self.rollout_wins = 0; self.max_depth = 0

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ states and edges
    def state_of(self, goals):
        key = tuple(g.key for g in goals)
        st = self.table.get(key)
        if st is None: st = self.table[key] = State(tuple(goals))
        return st

    def widen(self, st):
        """consider more actions at the state, as many as its visits allow (progressive widening); the priors are renormalised over the edges considered"""
        want = max(1, int(math.ceil(self.widen_c * (st.n + 1) ** self.widen_alpha)))
        if len(st.edges) >= want: return
        g = st.goals[0]; menu = self.menu_of(g); stream = self.stream_of(g, menu)
        while len(st.edges) < want:
            if st.taken < len(stream.out): a, c = stream.out[st.taken]
            else:
                if not stream._advance(math.inf): break
                a, c = stream.out[st.taken]
            st.taken += 1
            st.edges.append(Edge(a, menu.signature(a), c))
        if st.edges:
            m = min(e.cost for e in st.edges); z = sum(math.exp(-(e.cost - m)) for e in st.edges)
            for e in st.edges: e.prior = math.exp(-(e.cost - m)) / z

    def exhausted(self, st):
        """no more actions can be considered at the state (the stream has nothing left)"""
        stream = self.streams.get(st.goals[0].key)
        return stream is not None and st.taken >= len(stream.out) and not stream.heap

    def probe_edge(self, st, e):
        """the first time an edge is taken: probe the action (counted); a state for the goals it leaves, or a dead edge"""
        g = st.goals[0]; menu = self.menu_of(g)
        res = self.attempt_cost(g, menu, e.action, e.cost, math.inf)
        if res is None: e.dead = True; return
        children, _ = res
        e.n_children = len(children)
        e.child = self.state_of(list(children) + list(st.goals[1:]))

    def select(self, st):
        """the edge to follow from a state: the best PUCT score among the live edges (None if there is none)"""
        best = None; bs = -1e18; sq = math.sqrt(st.n + 1)
        for e in st.edges:
            if e.dead: continue
            q = (e.w / e.n) if e.n else 0.0
            s = q + self.c_puct * e.prior * sq / (1 + e.n)
            if e.solved: s = 1e9
            if s > bs: bs = s; best = e
        return best

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ evaluation
    def rollout(self, st):
        """the policy plays from the state for at most `rollout_depth` actions: (finished?, [(signature, goals left) of the actions taken])"""
        goals = list(st.goals); trail = []
        for _ in range(self.rollout_depth):
            if not goals: return True, trail
            g = goals[0]; menu = self.menu_of(g)
            smp = self.runner.sample(g, menu, self.rng, self.rollout_temperature)
            if smp.action is None: continue
            res = self.attempt_cost(g, menu, smp.action, 0.0, math.inf)
            if res is None: continue
            children, _ = res
            trail.append((menu.signature(smp.action), len(children)))
            goals = list(children) + goals[1:]
        return (not goals), trail

    def critic_value(self, st):
        total = 0.0
        for g in st.goals: total += self.runner.decoder(g).size_estimate()
        return math.exp(-total / self.value_scale)

    def evaluate(self, st):
        """(value in [0, 1], trail or None): see the module documentation"""
        if st.solved: return 1.0, None
        if self.value == 'critic': return self.critic_value(st), None
        self.rollouts += 1
        ok, trail = self.rollout(st)
        if ok: self.rollout_wins += 1; return 1.0, trail
        return 0.0, None

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ one simulation
    def simulate(self, root):
        self.simulations += 1
        path = []; st = root; st.on_path = True; on = [root]
        try:
            while True:
                if st.solved or st.dead: leaf = st; break
                self.widen(st)
                e = self.select(st)
                if e is None:
                    if self.exhausted(st) or not st.edges: st.dead = True
                    leaf = st; break
                if e.child is None and not e.dead:
                    self.probe_edge(st, e)
                if e.dead: continue
                child = e.child
                if child.on_path: e.dead = True; continue                  # (a cycle: the goals came back)
                path.append((st, e)); st = child; st.on_path = True; on.append(st)
                if st.n == 0 or st.solved: leaf = st; break
            value, trail = self.evaluate(leaf) if not leaf.dead else (0.0, None)
            if trail is not None: leaf.solved = True; leaf.trail = trail
            self.max_depth = max(self.max_depth, len(path))
            self.backup(path, leaf, value)
        finally:
            for s in on: s.on_path = False

    def backup(self, path, leaf, value):
        leaf.n += 1; leaf.w += value
        for st, e in reversed(path):
            e.n += 1; e.w += value; st.n += 1; st.w += value
            c = e.child
            if c.solved: e.solved = True; st.solved = True
            if c.dead: e.dead = True
            if not st.solved and not st.dead and st.edges and all(x.dead for x in st.edges) and self.exhausted(st): st.dead = True

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the search
    def prove(self, g, max_simulations=10 ** 9):
        """a Tree that solves the goal, or None (the probe budget or the time limit ran out, or the space was exhausted)"""
        self.t0 = time.time()
        root = self.state_of([g])
        try:
            for _ in range(max_simulations):
                if root.solved or root.dead: break
                self.simulate(root)
        except S.BudgetExhausted:
            return None
        if not root.solved: return None
        return self.extract(root)

    def extract(self, root):
        """the proof of a solved root, as a Tree: follow the solved edges to a state that was solved by a rollout (its trail) or is terminal"""
        seq = []; st = root; seen = set()
        while not (st.trail is not None or not st.goals):
            if st.key in seen: return None
            seen.add(st.key)
            e = next((x for x in st.edges if x.solved and x.child is not None and x.child.solved), None)
            if e is None: return None
            seq.append((e.sig, e.n_children)); st = e.child
        if st.trail: seq.extend(st.trail)
        return tree_from_preorder(seq)


def prove_statement(eng, stmt, premises, net, ctx, max_probes=4000, time_limit=None, **kw):
    """search for a proof of a closed statement with MCTS and commit it: (proof term, tree, searcher) or (None, None, searcher)"""
    eng.premises = premises
    p = eng.start(stmt)
    if p.done: return S.commit_trivial(eng, stmt, p), None, None
    g = eng.goal_of(p, 0)
    s = MCTSSearcher(eng, net, ctx, max_probes=max_probes, time_limit=time_limit, **kw)
    tree = s.prove(g)
    if tree is None: return None, None, s
    try: return S.commit(eng, stmt, tree), tree, s
    except G.ActionError: return None, None, s
