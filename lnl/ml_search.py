"""ml_search.py -- proof search guided by the policy network.

The search is gym_search's: an AND-OR search over goals, iterative deepening on COST (IDA*), every probe counted, what it finds replayed on a live proof and checked by the kernel.  What changes is where the
actions come from and what they cost: the candidate actions of a goal are the policy's, enumerated best first (a priority queue over partial decision sequences: `ActionStream`), and an action costs its
negative log-probability under the policy plus a small price per step.  A budget B then admits every proof whose steps are, along every branch, jointly at least e^-B probable -- the policy decides which
actions are tried and in which order; the kernel decides what is true.

    searcher = PolicySearcher(eng, net, ctx, max_probes=4000)
    tree = searcher.prove(g)                           # a gym_search.Tree or None;  searcher.probes, .expansions, .seconds_policy
    term, tree, s = prove_statement(eng, stmt, premises, net, ctx, max_probes=4000)       # commits the tree: the kernel-checked proof term
"""
import os, sys, time, heapq, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import itp_env_lnl as E
import gym_env as G, gym_search as S, gym_policy as P
from gym_env import Kind, ST_OK, ST_REFUSED, ST_FUEL
import ml_features as F, ml_model as M


class ActionStream:
    """the candidate actions of one goal in nondecreasing cost (cost = -log probability of the decisions that assemble the action), computed lazily and remembered.  `upto(rem)` yields (action, cost) with
    cost <= rem; a later call with a larger `rem` continues where the stream stopped."""
    def __init__(self, searcher, g, menu):
        self.s = searcher; self.g = g; self.menu = menu
        self.dec = M.Decoder(searcher.net, searcher.ctx, g, searcher.gtab())
        self.heap = []; self.tie = 0; self.out = []; self.seen = set(); self.exhausted = False
        asm = P.Assembler(searcher.eng, g, menu, rare=searcher.rare)
        ch = asm.start()
        self._push(0.0, (), self.dec.root(), asm, ch)

    def _push(self, cost, trace, carry, asm, ch):
        self.tie += 1
        heapq.heappush(self.heap, (cost, self.tie, trace, carry, asm, ch))

    def has_more(self): return bool(self.heap)

    def _advance(self, rem):
        """produce the next candidate action (appending to self.out); False if the cheapest thing left costs more than rem, or nothing is left"""
        s = self.s
        while self.heap:
            if self.heap[0][0] > rem: return False
            cost, _, trace, carry, asm, ch = heapq.heappop(self.heap)
            if asm is None:                                                    # replay the decisions of the trace from the start
                asm = P.Assembler(s.eng, self.g, self.menu, rare=s.rare); ch = asm.start()
                for i in trace:
                    if ch is None: break
                    ch = asm.choose(i)
            if asm.failed is not None and ch is None and asm.action is None: continue
            if ch is None:
                a = asm.action
                if a is None: continue
                try: sig = self.menu.signature(a)
                except Exception: continue
                if sig in self.seen: continue
                self.seen.add(sig); self.out.append((a, cost)); return True
            t0 = time.time()
            logp, oe, hs = self.dec.scores(carry, ch)
            s.t_policy += time.time() - t0; s.expansions += 1
            n = len(logp)
            k = min(n, s.topk)
            vals, idx = torch.topk(logp, k)
            first = True
            for v, i in zip(vals.tolist(), idx.tolist()):
                c = cost - v / s.temperature
                if c - cost > s.cap: break
                child_carry = self.dec.child(carry, ch, i, oe, hs)
                if first and asm is not None:
                    asm_child = asm; first = False
                    try: ch2 = asm_child.choose(i)
                    except Exception: asm_child = None; ch2 = None
                    if asm_child is not None: self._push(c, trace + (i,), child_carry, asm_child, ch2); continue
                first = False
                self._push(c, trace + (i,), child_carry, None, None)
        return False

    def upto(self, rem):
        i = 0
        while True:
            if i < len(self.out):
                a, c = self.out[i]
                if c > rem: return
                yield a, c; i += 1; continue
            if not self._advance(rem): return


class PolicySearcher(S.CostSearcher):
    """IDA* on the policy's costs (see the module documentation). `step_cost`: a price per action on top of its negative log-probability (it bounds the depth and keeps a certain action from being free);
    `topk`: how many options of one decision are expanded; `cap`: an option less probable than e^-cap is not expanded; `temperature` divides the log-probabilities (a flatter policy for T > 1)."""
    def __init__(self, eng, net, ctx, max_probes=4000, time_limit=None, step_cost=0.2, topk=24, cap=9.0, temperature=1.0, fuel_step=1.0, rare=True):
        S._SearcherBase.__init__(self, eng, max_probes, None, time_limit)
        self.net = net; self.ctx = ctx; self.step_cost = step_cost; self.topk = topk; self.cap = cap; self.temperature = temperature; self.rare = rare
        self.fuel_step = fuel_step; self.cache = {}; self.cuts = 0; self.passes = 0
        self.streams = {}; self._gtab = None; self.expansions = 0; self.t_policy = 0.0

    def gtab(self):
        if self._gtab is None:
            with torch.no_grad(): self._gtab = self.net.global_table()
        return self._gtab

    def stream_of(self, g, menu):
        st = self.streams.get(g.key)
        if st is None: st = self.streams[g.key] = ActionStream(self, g, menu)
        return st

    def attempt_cost(self, g, menu, action, base, rem):
        """like CostSearcher.attempt with the cost of the action given"""
        sig = menu.signature(action); key = (g.key, sig)
        ent = self.cache.get(key)
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
        self.cache[key] = (ST_REFUSED,)
        return None

    def solve(self, g, rem):
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
            stream = self.stream_of(g, menu)
            for action, pc in stream.upto(rem - self.step_cost):
                base = pc + self.step_cost
                res = self.attempt_cost(g, menu, action, base, rem)
                if res is None: continue
                children, c = res
                trees = []
                for ch in children:
                    t = self.solve(ch, rem - c)
                    if t is None: break
                    trees.append(t)
                else:
                    tree = S.Tree(menu.signature(action), trees)
                    self.solved[g.key] = tree
                    return tree
            if stream.has_more(): self.cuts += 1
            if f is None or f[0] < rem: self.failed[ck] = (rem, self.cuts > cuts0)
            return None
        finally:
            self.path.discard(ck)

    def prove(self, g, max_cost=30.0, start_cost=1.5, grow=1.3):
        self.t0 = time.time()
        try:
            b = start_cost
            while b <= max_cost:
                self.cuts = 0; self.passes += 1
                t = self.solve(g, b)
                if t is not None: return t
                if self.cuts == 0: return None
                b = b * grow if b * grow - b > 0.5 else b + 0.5
        except S.BudgetExhausted:
            return None
        return None


def prove_statement(eng, stmt, premises, net, ctx, max_probes=4000, time_limit=None, **kw):
    """search for a proof of a closed statement with the policy and commit it: (proof term, tree, searcher) or (None, None, searcher)"""
    eng.premises = premises
    p = eng.start(stmt)
    if p.done: return S.commit_trivial(eng, stmt, p), None, None
    g = eng.goal_of(p, 0)
    s = PolicySearcher(eng, net, ctx, max_probes=max_probes, time_limit=time_limit, **kw)
    tree = s.prove(g)
    if tree is None: return None, None, s
    return S.commit(eng, stmt, tree), tree, s


def tree_portable(eng, stmt, premises, tree):
    """the actions of a Tree in the order the engine takes them (pre-order: the first goal is always the one acted on), as portable steps (`gym_acts.to_portable`), after replaying them on a live proof; the
    kernel checks the finished term. Raises ActionError if the replay does not reproduce the search."""
    import gym_acts as ACTS
    eng.premises = premises
    p = eng.start(stmt); steps = []
    def go(t):
        g = eng.goal_of(p, 0); menu = eng.menu(g)
        a = menu.find(t.signature)
        if a is None: raise G.ActionError('the action %r is not in the menu of the live goal' % (t.signature[0],))
        steps.append(ACTS.to_portable(eng, menu, a))
        left = eng.replay(p, a, menu)
        if left != len(t.children): raise G.ActionError('the live goal left %d goals, the search saw %d' % (left, len(t.children)))
        for c in t.children: go(c)
    if tree is not None: go(tree)
    if not (p.done and p.ps.is_complete()): raise G.ActionError('the replay left goals open')
    eng.verify(stmt, p)
    return steps
