"""ml_her.py -- hindsight: the theorems an attempt proved on the way, whatever became of the attempt.

An attempt at a theorem that fails still proves things: every goal it closed -- a subtree of its proof tree, a goal a search closed on the way, a claim the worker of the manager-worker agent proved -- is a
theorem of its own (the goal's hypotheses and its target, closed over) with a proof (the actions that closed it).  `Harvest` collects them; every one is replayed in a fresh proof state and checked by the
kernel before it is kept (`gym_tree.replays`), so what comes out is a THEOREM with a proof, not a guess.

    h = Harvest(eng, premises, name, max_len=20, max_n=12)
    ml_act.sample_prove(..., harvest=h.episode)          # tracked episodes (their proof trees)
    h.searcher(s, tree)                                    # a PolicySearcher / MCTSSearcher after `prove` (tree: the proof it found, or None)
    h.claims(res.claims)                                   # the claims the manager-worker agent's worker proved (ml_hier.HierResult.claims)
    found = h.found()                                      # [Found]: one per statement (the shortest proof), longest proofs first, at most max_n

A `Found` is a record that can be sent to another process (`to_dict`: the dict of `gym_tree.Example.to_dict` with the problem it came from).  `steps_of(ctx, eng, found)` turns its proof into policy
training steps (ml_data).  Nothing here is used for evaluation: the learners call it on TRAIN tasks only.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import collections, json
import gym_env as G, gym_tree as GT, gym_premises as PM
import ml_data as D, ml_act as ACT, ml_search as MS

SAFE = ACT.REFUSALS + (IndexError, KeyError, ValueError, TypeError)


class Found:
    """a theorem an attempt proved: the closed statement (`stmt`, a Term of this process's world; None after `to_dict` was sent elsewhere), its printed form, its code, the premises it was proved under, the
    proof (portable actions), where it came from"""
    __slots__ = ('stmt', 'text', 'digest', 'code', 'premises', 'steps', 'origin')
    def __init__(self, stmt, text, code, premises, steps, origin):
        self.stmt = stmt; self.text = text; self.digest = GT.digest(text); self.code = code; self.premises = premises; self.steps = steps; self.origin = origin
    @property
    def length(self): return len(self.steps)
    @property
    def key(self): return (self.digest, json.dumps(PM.to_json(self.premises), sort_keys=True))
    def to_dict(self):
        return {'kind': 'hindsight', 'statement': self.text, 'statement_digest': self.digest, 'statement_code': None if self.code is None else [list(t) for t in self.code],
                'premises': PM.to_json(self.premises), 'steps': self.steps, 'length': self.length, 'origin': self.origin, 'verified': True}


class Harvest:
    def __init__(self, eng, premises, name='', max_len=20, min_len=1, max_n=12, max_candidates=24):
        self.eng = eng; self.premises = premises; self.name = name; self.max_len = max_len; self.min_len = min_len; self.max_n = max_n; self.max_candidates = max_candidates
        self._found = {}; self.stats = collections.Counter()

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ adding
    def _add(self, f):
        if f.length < self.min_len: self.stats['too short'] += 1; return
        cur = self._found.get(f.key)
        if cur is None or f.length < cur.length: self._found[f.key] = f
        else: self.stats['duplicate'] += 1

    def _make(self, stmt, steps, origin):
        """verify (replay in a fresh proof state, the kernel checks the term) and record a candidate theorem"""
        eng = self.eng
        if GT.replays(eng, stmt, steps, self.premises) is None:
            self.stats['settled at once' if GT.closed_at_once(eng, stmt, self.premises) else 'did not replay'] += 1; return None
        self._add(Found(stmt, eng.w.show(stmt), GT.statement_code(eng, stmt, self.premises), self.premises, steps, origin))
        self.stats['kept'] += 1

    def episode(self, ep):
        """the closed subtrees of a TRACKED episode (finished or not), the episode's own root excluded: the largest `max_candidates` of at most `max_len` actions"""
        tree = ep.tree
        if tree is None or tree.broken: self.stats['no tree'] += 1; return
        cands = []
        for n in tree.solved_nodes():
            if n.parent is None: continue
            sub = tree.subtree_steps(n)
            if len(sub) > self.max_len: self.stats['too long'] += 1; continue
            if any(m.portable is None for m in sub): self.stats['not portable'] += 1; continue
            cands.append((len(sub), n, sub))
        cands.sort(key=lambda t: (-t[0], t[1].id))
        for L, n, sub in cands[:self.max_candidates]:
            stmt = GT.closed_statement(self.eng.w, n.goal.hyp_types, n.goal.target)
            self._make(stmt, [m.portable for m in sub], {'problem': self.name, 'kind': G.KIND_NAMES[n.action.kind], 'depth': n.depth, 'via': 'tree'})

    def goals(self, pairs, via='search'):
        """[(GoalData, Tree)]: goals a search proved, each with the tree that proved it"""
        cands = sorted(((t.size(), i, g, t) for i, (g, t) in enumerate(pairs) if t.size() <= self.max_len), key=lambda x: (-x[0], x[1]))
        for size, i, g, t in cands[:self.max_candidates]:
            stmt = GT.closed_statement(self.eng.w, g.hyp_types, g.target)
            try: steps = MS.tree_portable(self.eng, stmt, self.premises, t)
            except SAFE: self.stats['did not replay'] += 1; continue
            self._add(Found(stmt, self.eng.w.show(stmt), GT.statement_code(self.eng, stmt, self.premises), self.premises, steps, {'problem': self.name, 'depth': None, 'via': via}))
            self.stats['kept'] += 1

    def searcher(self, s, root_tree=None):
        """the goals a PolicySearcher (IDA*) or an MCTSSearcher closed on the way, except the root (`root_tree`: the proof the search returned, if any)"""
        if hasattr(s, 'solved_single_goals'): pairs = s.solved_single_goals()
        else: pairs = [(g, s.solved[k]) for k, g in s.solved_goals.items() if k in s.solved and s.solved[k] is not root_tree]
        self.goals(pairs, via='mcts' if hasattr(s, 'solved_single_goals') else 'ida')

    def claims(self, records):
        """the claims the worker of the manager-worker agent proved (dicts of ml_hier.HierAgent._claim_record with `solved`); rebuilt in this world from their code"""
        for r in records:
            if not r.get('solved') or r.get('statement_code') is None or not r.get('steps'): continue
            try: stmt = GT.statement_from_code(self.eng, r['statement_code'], PM.of(r['premises']))
            except Exception: self.stats['claim not sayable'] += 1; continue
            if len(r['steps']) > self.max_len: self.stats['too long'] += 1; continue
            prem0 = self.premises; self.premises = PM.of(r['premises'])
            try: self._make(stmt, r['steps'], {'problem': self.name, 'kind': 'CLAIM', 'via': 'claim'})
            finally: self.premises = prem0

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ result
    def found(self):
        """the theorems found, longest proof first (ties: by statement), at most `max_n`"""
        out = sorted(self._found.values(), key=lambda f: (-f.length, f.digest))
        return out[:self.max_n]


def steps_of(ctx, eng, f, name=None, stats=None):
    """the policy training steps (ml_data.Step) of a found theorem's proof; [] when the proof cannot be decomposed"""
    try: return D.extract(ctx, eng, f.stmt, f.premises, f.steps, name=name or f.digest, stats=stats)
    except Exception: return []
