"""gym_tree.py -- the proof tree of an episode, and the hindsight data it yields.

A failed attempt at a theorem is full of things that WERE proved: the branch of a case split that closed, the claim whose proof was found before the budget ran out, the fact a lemma application derived.
Each of them is a statement in a context, and the actions that produced it are a proof of it. This module keeps the record that makes those proofs recoverable and turns them into new, standalone
problems -- without guessing: every example is replayed in a fresh engine and checked by the kernel before it is returned (`ProofTree.examples(verify=True)`).

    ep = Episode(eng, stmt, name, track=True)            # (or BankEnv(track=True))
    ... ep.step(action) ...                               # the engine always works on the FIRST goal; its action replaces it by the goals it leaves
    tree = ep.tree
    tree.nodes                                            # every goal an action was applied to: its goal, action, parent, children, whether its whole subtree is closed
    tree.examples()                                       # the theorems this episode proved, as Example objects (statement code, premises, portable actions)
    ep.events                                             # what the LAST step achieved: [('solved', node), ('derived', node)]

Two sources of examples (both are theorems the episode PROVED, whether or not it finished its own):

  solved    a node whose whole subtree is closed.  The statement is the goal closed over its hypotheses (`forall h0 .. hn, target`), the proof the actions of the subtree.  (The engine always acts on the
            first goal and an action replaces it in place, so the actions of a subtree are CONTIGUOUS in time, in the order a fresh episode on the closed statement would need them.)
  derived   a node whose action added facts to its goal (FORWARD, INSTANTIATE, INJECT, REWRITE_AT): the fact T, in the context the action was applied in, is closed by that single action ("have h : T := ...;
            exact h").  Only when the context of the successor goal extends the context of the goal (nothing was cleared or re-indexed) and T mentions no new hypothesis.

Nothing here is trusted: an example that does not replay is dropped (and counted, `tree.stats`), and a finished proof is checked by the kernel like every other.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import collections, hashlib
import itp_env_lnl as E
from author import AuthorError
import gym_env as G
import gym_build as GB
import gym_acts as ACTS
import gym_premises as PM
from gym_env import Kind, ActionError

# the "direction" of an action: what it does to the goal (docs/ARCHITECTURE_RL.md, section 2)
DIRECTION = {Kind.RING: 'CLOSE', Kind.LINARITH: 'CLOSE', Kind.NLINARITH: 'CLOSE', Kind.LINCOMB: 'CLOSE',
             Kind.APPLY: 'BACKWARD', Kind.APPLY_WITH: 'BACKWARD', Kind.REFINE: 'BACKWARD', Kind.REWRITE: 'BACKWARD', Kind.REWRITE_WITH: 'BACKWARD', Kind.USE: 'BACKWARD', Kind.BY_CONTRA: 'BACKWARD',
             Kind.FORWARD: 'FORWARD', Kind.INSTANTIATE: 'FORWARD', Kind.REWRITE_AT: 'FORWARD', Kind.INJECT: 'FORWARD',
             Kind.CASES: 'SPLIT', Kind.INDUCT: 'SPLIT', Kind.INDUCT_WITH: 'SPLIT', Kind.INTERVAL_CASES: 'SPLIT',
             Kind.CLAIM: 'CLAIM', Kind.INTRO: 'INTRO'}
DIRECTIONS = ('CLOSE', 'BACKWARD', 'FORWARD', 'SPLIT', 'CLAIM', 'INTRO')
DERIVING = frozenset({Kind.FORWARD, Kind.INSTANTIATE, Kind.INJECT, Kind.REWRITE_AT})        # the kinds that add facts to their goal
REFUSALS = (ActionError, AuthorError, E.ITPError, RuntimeError, GB.BuildError)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ statements
def closed_statement(w, hyp_types, target):
    """`forall (h0 : T0) .. (hn : Tn), target` as a Term of the world `w`: a goal as a closed statement. (A goal's hypotheses are context variables, numbered by level; the telescope binds them in
    order. A goal that was SETTLED has no leading quantifier in its target, so an episode on the closed statement introduces exactly these hypotheses again and is the same goal.)"""
    return w.T(w.k.close_telescope('pi', list(hyp_types), target, 0))


def statement_code(eng, stmt, premises):
    """the canonical code of a closed statement (a tuple of builder tokens, with global ids), checked by building it back; None when the statement cannot be said"""
    eng.premises = premises
    space = GB.TermSpace(eng, [], [], roots=[stmt.n])
    try:
        code = space.encode(stmt.n)
        t = space.build(code, eng.k.mk_type(0))
    except (GB.NotEncodable,) + REFUSALS: return None
    return code if t == stmt.n else None


def statement_from_code(eng, code, premises):
    """the closed statement (a Term) that a code says, in the world of `eng` with `premises` globals visible; GB.BuildError if the code says none"""
    eng.premises = premises
    space = GB.TermSpace(eng, [], [], roots=[])
    return eng.w.T(space.build(tuple(tuple(t) for t in code), eng.k.mk_type(0)))


def digest(text): return hashlib.sha256(text.encode()).hexdigest()[:16]


def closed_at_once(eng, stmt, premises):
    """does the engine close the statement by itself, before any action (its settling: a hypothesis, or a computation, proves it)? Then there is nothing to learn from it, and no action to replay."""
    eng.premises = premises
    try: return eng.start(stmt).n_goals == 0
    except REFUSALS: return False


def replays(eng, stmt, steps, premises):
    """the proof term (a Term) if the portable actions prove the statement in a fresh proof state -- the kernel has checked it against the statement -- else None"""
    try: return ACTS.replay_portable(eng, stmt, steps, premises)
    except REFUSALS: return None


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ examples
class Example:
    """a theorem an episode proved, with its proof: `kind` ('solved' / 'derived'), the closed statement (`stmt`, a Term; `text`, printed; `code`, in the builder's tokens, or None if it cannot be said),
    the number of visible globals (`premises`), the proof as portable actions (`steps`) and, once replayed, as a kernel-checked term (`proof`), and where it came from (`origin`)."""
    __slots__ = ('kind', 'stmt', 'text', 'code', 'premises', 'steps', 'origin', 'verified', 'proof')
    def __init__(self, kind, stmt, text, code, premises, steps, origin, verified, proof=None):
        self.kind = kind; self.stmt = stmt; self.text = text; self.code = code; self.premises = premises; self.steps = steps; self.origin = origin; self.verified = verified; self.proof = proof
    @property
    def length(self): return len(self.steps)
    @property
    def key(self): return (self.stmt.n, self.premises)
    def to_dict(self):
        return {'kind': self.kind, 'statement': self.text, 'statement_digest': digest(self.text), 'statement_code': None if self.code is None else [list(t) for t in self.code],
                'premises': PM.to_json(self.premises), 'steps': self.steps, 'length': self.length, 'origin': self.origin, 'verified': self.verified}
    def __repr__(self): return 'Example(%s, %d actions, %s)' % (self.kind, self.length, self.text[:80])


class Node:
    """one goal: `goal` (GoalData) and `action` once an action was applied to it (`step` is the index of that action in the episode's steps; `portable` its portable form, None if it has none), its `children`
    (the goals the action left in its place, after settling), `post` (for the actions that add facts: the first of those goals), and `closed` once it and everything below it is closed (`last`: the index of
    the last action of its subtree)"""
    __slots__ = ('id', 'parent', 'goal', 'action', 'portable', 'step', 'last', 'children', 'closed', 'depth', 'post')
    def __init__(self, id, parent):
        self.id = id; self.parent = parent; self.depth = 0 if parent is None else parent.depth + 1
        self.goal = None; self.action = None; self.portable = None; self.step = None; self.last = None; self.children = (); self.closed = False; self.post = None
    def __repr__(self): return 'Node(%d, %s%s)' % (self.id, G.KIND_NAMES[self.action.kind] if self.action is not None else 'open', ', closed' if self.closed else '')


class ProofTree:
    def __init__(self, ep):
        self.ep = ep; self.nodes = []; self.by_step = []; self.stats = collections.Counter(); self.broken = False
        self.open = [self._new(None) for _ in range(ep.p.n_goals)]               # one node per open goal, in the order of the proof state's goals
        self.roots = list(self.open)

    def _new(self, parent):
        n = Node(len(self.nodes), parent); self.nodes.append(n); return n

    def record(self, g, menu, action, made, step):
        """called by the episode after the engine carried out `action` (the `step`-th) on its first goal `g`, which `made` goals replaced. Returns the events of the step."""
        if self.broken or not self.open: self.broken = True; return []
        node = self.open.pop(0)
        node.goal = g; node.action = action; node.step = step
        try: node.portable = ACTS.to_portable(self.ep.eng, menu, action)
        except REFUSALS: node.portable = None
        self.by_step.append(node)
        kids = [self._new(node) for _ in range(made)]
        node.children = tuple(kids); self.open[0:0] = kids
        if len(self.open) != self.ep.p.n_goals: self.broken = True; return []                  # (the bookkeeping and the proof state disagree: stop trusting the tree; a test fails on this)
        events = []
        if not kids:
            m = node
            while m is not None:
                m.closed = True; m.last = step; events.append(('solved', m))
                m = m.parent
                if m is None or not all(c.closed for c in m.children): break
        elif action.kind in DERIVING:
            node.post = self.ep.goal()                                                             # (the goal that follows is wanted by the next step anyway: cached by the episode)
            events.append(('derived', node))
        return events

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------- examples
    def subtree_steps(self, n):
        """the nodes of the subtree of a closed node, in the order of their actions"""
        return self.by_step[n.step:n.last + 1]

    def solved_nodes(self):
        return [n for n in self.nodes if n.closed and n.action is not None]

    def examples(self, verify=True, max_len=30, include_root=False, kinds=('solved', 'derived'), dedupe=True):
        """the theorems the episode proved, as Examples (see the module documentation). `verify`: replay each in a fresh proof state and check it with the kernel, and drop those that do not replay (the
        default; set False only to look at the raw candidates). `max_len`: the longest proof to consider (replaying costs a menu per action). `include_root`: also the episode's own problem.
        `dedupe`: one example per (statement, premises), the shortest proof. The reasons candidates were dropped are counted in `self.stats`."""
        ep = self.ep; eng = ep.eng; k = eng.k; out = []; st = self.stats; st.clear()
        if self.broken: st['tree_broken'] += 1; return out
        premises = ep.premises
        def add(kind, stmt, steps, origin):
            code = statement_code(eng, stmt, premises)
            text = eng.w.show(stmt)
            ok = None; proof = None
            if verify:
                proof = replays(eng, stmt, steps, premises)
                if proof is None: st[('settled_at_once_' if closed_at_once(eng, stmt, premises) else 'not_replayed_') + kind] += 1; return
                ok = True
            out.append(Example(kind, stmt, text, code, premises, steps, origin, ok, proof)); st['kept_' + kind] += 1
        if 'solved' in kinds:
            for n in self.solved_nodes():
                if n.parent is None and not include_root: st['root'] += 1; continue
                sub = self.subtree_steps(n)
                if len(sub) > max_len: st['too_long'] += 1; continue
                if any(m.portable is None for m in sub): st['not_portable_solved'] += 1; continue
                stmt = closed_statement(eng.w, n.goal.hyp_types, n.goal.target)
                add('solved', stmt, [m.portable for m in sub], {'problem': ep.name, 'step': n.step, 'node': n.id, 'depth': n.depth, 'kind': G.KIND_NAMES[n.action.kind]})
        if 'derived' in kinds:
            for n in self.nodes:
                if n.post is None: continue
                if n.portable is None: st['not_portable_derived'] += 1; continue
                pre, post = n.goal, n.post; n_pre = pre.n_hyps
                if post.n_hyps <= n_pre or list(post.hyp_types[:n_pre]) != list(pre.hyp_types): st['context_changed'] += 1; continue
                for j in range(n_pre, post.n_hyps):
                    T = post.hyp_types[j]
                    if any(l >= n_pre for l in G.atoms_of(k, T)[1]): st['depends_on_new'] += 1; continue
                    if G.unfolds_to_pi(k, T): st['pi_fact'] += 1; continue
                    stmt = closed_statement(eng.w, pre.hyp_types, T)
                    add('derived', stmt, [n.portable], {'problem': ep.name, 'step': n.step, 'node': n.id, 'depth': n.depth, 'kind': G.KIND_NAMES[n.action.kind], 'fact': j - n_pre})
        if dedupe:
            best = {}
            for ex in out:
                cur = best.get(ex.key)
                if cur is None or ex.length < cur.length: best[ex.key] = ex
            st['duplicates'] += len(out) - len(best)
            out = [ex for ex in out if best[ex.key] is ex]
        return out
