"""recorder.py -- authored tactic proofs as agent actions.

The libraries (nat_lib .. imo_lib) prove their ~400 theorems with tactic SCRIPTS (tactics.Proof): `p.intro('a'); p.rw('add_comm', 'a', 'b'); p.exact('h')`. A learner has no such text: it acts on a GOAL
through the engine (gym_env.py, gym_acts.py): a kind of action and its arguments, chosen from a menu or said as codes of the typed term builder. The RECORDER turns one into the other:

    1. LOG.   The script runs on a `RecordingProof`, a Proof that records every tactic call the script makes (not the calls those make) as a node of a TREE: the goal it acted on, the call with its
              arguments resolved to terms, and the goals it left (its children). Nothing about the proof changes: it is the same kernel proof, and it is admitted as before.
    2. REPLAY. In a settled engine (goals are introduced and closed by assumption / reflexivity by themselves, dead hypotheses are cleared after a split) the tree is walked from the root: for the
              first open goal the author's call is translated into engine actions (`translate`), the actions are PROBED, accepted when they leave as many goals as the author's call left (those
              that settling closes aside), and carried out; the goals they leave are paired with the author's children and their hypotheses with the author's, by NAME and then by TYPE (`align`).
    3. CHECK. When no goal is left the proof the engine built is checked by the kernel against the statement (Engine.verify). So whatever this module believes, what comes out is a kernel-checked
              proof by actions; a translation that is wrong makes the replay FAIL (a gap of the action language or a bug here), never a false theorem appear.

A theorem whose replay fails is reported with the first step that could not be translated and why: that is the list of what the action language still lacks (`report`).
"""
import os, sys, collections, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import itp_env_lnl as E
import author
from author import AuthorError, Term, World
from tactics import Proof
import gym_env as G
import gym_build as GB
import gym_acts as ACTS
from gym_env import Kind, Action, ActionError, GoalData

SKIP_OPS = frozenset({'intro', 'norm', 'clear', 'try_refl'})          # steps the engine does by itself (or never needs): they have one child, the same goal
EMPTY_MENU = G.Menu([], [], np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint8))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the log
class Node:
    """one goal the script acted on: `pre` the goal as data (kernel handles), `op` / `info` the call (info: its arguments resolved to terms), `kids` the goals it left, in order"""
    __slots__ = ('id', 'pre', 'op', 'info', 'kids')
    def __init__(self, nid, pre): self.id = nid; self.pre = pre; self.op = None; self.info = None; self.kids = []
    def __repr__(self): return 'Node(%d %s %d kids)' % (self.id, self.op, len(self.kids))


class RecordingProof(Proof):
    """a Proof that logs the tactic calls of the script (those made from outside the Proof, not the ones the tactics make) as a tree. The scripts of `have` / `obtain` / `calc` are the script's own
    calls and are logged; the tactics that are made of others (rw, rw_at, exists, interval_cases, ...) are logged as one."""
    def __init__(self, w, stmt, name=None, auto_norm=True):
        super().__init__(w, stmt, name, auto_norm)
        self._depth = 0; self._count = 0; self.size = 0
        self.root = self._new_node(self._snap(0)); self._open = [self.root]

    def _new_node(self, pre):
        n = Node(self._count, pre); self._count += 1; self.size += 1; return n

    def _snap(self, i=0):
        ps = self.ps; n = ps.goal_num_hyps(i)
        return GoalData(ps.goal_target(i), [ps.goal_hyp_type(i, h) for h in range(n)], list(self._names[ps.goal_meta_id(i)]))

    def _log(self, op, fn, info=None, after=None):
        """run fn() as the call `op` if it is a call of the script (not nested in another tactic); log it. `after()` (called when the call succeeded, before the goals it left are read) adds to the info"""
        if self._depth > 0 or self.done: return fn()
        if len(self._open) != self.n_goals: raise AuthorError('recorder: the log lost track of the goals (%d logged, %d open)' % (len(self._open), self.n_goals))
        node = self._open[0]
        node.pre = self._snap(0)                                       # (the goal as it is NOW: a metavariable that an earlier call assigned is replaced by its value)
        node.info = info() if info is not None else {}
        node.op = op
        before = self.n_goals
        self._depth += 1
        try:
            res = fn()
            if after is not None: node.info.update(after())
        except BaseException:
            node.op = None; node.info = None; raise                    # (a script may try a tactic and go on when it fails: nothing is logged for a call that failed)
        finally: self._depth -= 1
        self._finish(node, before)
        return res

    def _solution(self, meta, n):
        """the proof term that the goal with this meta (n hypotheses) has now, in terms of the hypotheses (fvars of levels 0..n-1): what the call that was just made put in its place"""
        k = self.k; t = k.resolve_metas(k.mk_meta(meta))
        for lvl in range(n):
            v = k.view(t)
            if v[0] != 'lam': return None
            t = k.instantiate(v[2], [k.mk_fvar(lvl)])
        return t

    def _finish(self, node, before):
        made = self.n_goals - before + 1
        if made < 0: raise AuthorError('recorder: a tactic closed goals other than the first')
        kids = [self._new_node(self._snap(i)) for i in range(made)]
        node.kids = kids; self._open[0:1] = kids

    # --- the tactics
    def intro(self, *names): return self._log('intro', lambda: Proof.intro(self, *names))
    def apply(self, term, *args):
        if self._depth > 0 or self.done: return Proof.apply(self, term, *args)
        meta = self.ps.goal_meta_id(0); n = self.ps.goal_num_hyps(0)
        return self._log('apply', lambda: Proof.apply(self, term, *args), lambda: {'head': self.t(term), 'args': [self.t(a) for a in args]}, lambda: {'sol': self._solution(meta, n)})
    def exists(self, witness):
        return self._log('exists', lambda: Proof.exists(self, witness), lambda: {'witness': self.t(witness)})
    def exact(self, term):
        return self._log('exact', lambda: Proof.exact(self, term), lambda: {'term': self.t(term)})
    def cases(self, var, names=None, generalizing=()):
        return self._log('cases', lambda: Proof.cases(self, var, names, generalizing),
                         lambda: {'lvl': self._level(var), 'gen': [self._level(g) for g in generalizing], 'names': names})
    def induct(self, var, names=None, generalizing=(), indexed=False):
        return self._log('induct', lambda: Proof.induct(self, var, names, generalizing, indexed),
                         lambda: {'lvl': self._level(var), 'gen': [self._level(g) for g in generalizing], 'names': names, 'indexed': indexed})
    def clear(self, levels, i=0):
        if i != 0: return Proof.clear(self, levels, i)
        return self._log('clear', lambda: Proof.clear(self, levels, i))
    def norm(self, i=0):
        if i != 0: return Proof.norm(self, i)
        return self._log('norm', lambda: Proof.norm(self, i))
    def refl(self): return self._log('refl', lambda: Proof.refl(self))
    def rewrite(self, eq, rev=False):
        return self._log('rewrite', lambda: Proof.rewrite(self, eq, rev), lambda: {'eq': self.t(eq), 'rev': bool(rev)})
    def rw(self, lemma, *args, rev=False, close=True):
        return self._log('rw', lambda: Proof.rw(self, lemma, *args, rev=rev, close=close),
                         lambda: {'lemma': self.t(lemma), 'args': [self.t(a) for a in args], 'rev': bool(rev), 'close': bool(close)})
    def rw_at(self, hyp, lemma, *args, rev=False, new=None):
        return self._log('rw_at', lambda: Proof.rw_at(self, hyp, lemma, *args, rev=rev, new=new),
                         lambda: {'hyp': self._level(hyp), 'lemma': self.t(lemma), 'args': [self.t(a) for a in args], 'rev': bool(rev), 'new': new})
    def try_refl(self): return self._log('try_refl', lambda: Proof.try_refl(self))
    def ring(self): return self._log('ring', lambda: Proof.ring(self))
    def linear_combination(self, *pairs):
        def info():
            out = []
            for c, h in pairs:
                out.append((c if isinstance(c, int) and not isinstance(c, bool) else self.t(c), self._level(self.t(h))))
            return {'pairs': out}
        return self._log('linear_combination', lambda: Proof.linear_combination(self, *pairs), info)
    def linarith(self, *extra):
        return self._log('linarith', lambda: Proof.linarith(self, *extra), lambda: {'extra': [self.t(e) for e in extra]})
    def nlinarith(self, *extra):
        return self._log('nlinarith', lambda: Proof.nlinarith(self, *extra), lambda: {'extra': [self.t(e) for e in extra]})
    def by_contra(self, name='h'):
        return self._log('by_contra', lambda: Proof.by_contra(self, name), lambda: {'name': name})
    def interval_cases(self, var, lo, hi, name=None):
        return self._log('interval_cases', lambda: Proof.interval_cases(self, var, lo, hi, name), lambda: {'lvl': self._level(var), 'lo': lo, 'hi': hi, 'name': name})
    def induction_on(self, lemma, var, names=()):
        return self._log('induction_on', lambda: Proof.induction_on(self, lemma, var, names), lambda: {'lemma': self.t(lemma), 'lvl': self._level(var), 'names': tuple(names)})
    def assumption(self): return self._log('assumption', lambda: Proof.assumption(self))

    def have(self, name, fact, script=None):
        """`have` is logged as the step that adds the claim (the kernel's `apply cut P G`), then the script's calls on the claim, then the `intro` of the new hypothesis"""
        if self._depth > 0 or self.done: return Proof.have(self, name, fact, script)
        if len(self._open) != self.n_goals: raise AuthorError('recorder: the log lost track of the goals (%d logged, %d open)' % (len(self._open), self.n_goals))
        w = self.w
        with self.focus():
            G0 = self.goal().target
            if script is None: proof = self.t(fact); P = w.simp(w.type_of(proof))
            else: P = self.t(fact); proof = None
        node = self._open[0]; node.pre = self._snap(0); node.op = 'have'; node.info = {'name': name, 'P': P, 'proof': proof, 'script': script is not None}
        before = self.n_goals
        self._depth += 1
        try: self.apply('cut', P, G0, *([proof] if proof is not None else []))
        except BaseException: node.op = None; node.info = None; raise
        finally: self._depth -= 1
        self._finish(node, before)
        if proof is None:
            script(self)
            if self.n_goals != before: raise AuthorError('have %s: the script did not close exactly the statement it was asked to prove\n%s' % (name, self))
        return self.intro(name)


class Recording:
    """a theorem with the log of its proof: `stmt` (Term), `gid` (its id: the lemmas before it are the premises), `root`, `size`"""
    def __init__(self, world, name, gid, stmt, proof):
        self.w = world; self.name = name; self.gid = gid; self.stmt = stmt; self.root = proof.root; self.size = proof.size
        self.steps = collections.Counter(); self._count(proof.root)
    def _count(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            if n.op: self.steps[n.op] += 1
            stack.extend(n.kids)


def record_world(build, *args, **kw):
    """run a library builder (`imo_lib.build_imo`, ...) with every theorem proved on a RecordingProof: (the world, {name: Recording} in the order proved)"""
    recs = {}
    def collect(w, name, gid, stmt, proof): recs[name] = Recording(w, name, gid, stmt, proof)
    old = (World.proof_class, World.on_proved)
    World.proof_class = RecordingProof; World.on_proved = staticmethod(collect)
    try: w = build(*args, **kw)
    finally: World.proof_class, World.on_proved = old
    return w, recs


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ gaps
RETIRED = '~'                                              # a hypothesis the engine introduced that belongs to no author's goal below this point is renamed to start with this: it takes no part in the pairing
TAG = 'e_'                                                 # the engine's names for the hypotheses it introduces start with this while a recording is replayed: no author's name clashes with them


class Gap(Exception):
    """a step that could not be translated or carried out: `reason` is a short code (the report counts them), `detail` the message"""
    def __init__(self, reason, detail=''):
        super().__init__('%s: %s' % (reason, detail)); self.reason = reason; self.detail = detail


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ aligning goals
def subst_levels(k, node, lm, mentioned):
    """`node` with the author's context levels replaced by the engine's (`lm`); None when it mentions a level that has no counterpart"""
    if not mentioned: return node
    repl = []
    for f in mentioned:
        j = lm.get(f)
        if j is None: return None
        if j != f: repl.append((f, k.mk_fvar(j)))
    return k.replace_fvars(node, repl) if repl else node


def conv_bounded(eng, a, b, n):
    """are two terms of a context of n variables convertible, within a small budget?"""
    k = eng.k; saved = k.get_fuel(); k.set_fuel(G.SETTLE_FUEL)
    try: return bool(k.conv_open(a, b, n))
    except (E.ITPError, RuntimeError): return False
    finally: k.set_fuel(saved)


def head_syms(k, t):
    """the symbols a type can be said to be headed by, looking through the definitions at its head: ('c', global id), 'pi', ('t', level); None stands for anything (a variable or a computation at the head)"""
    out = set()
    for v in GB.variants(k, t):
        vv = k.view(v)
        if vv[0] == 'pi': out.add('pi')
        elif vv[0] == 'type': out.add(('t', vv[1]))
        else:
            h, _ = GB.spine(k, v); hv = k.view(h)
            out.add(('c', hv[1]) if hv[0] == 'const' else None)
    return out


def align(eng, raw, S):
    """pair the hypotheses of the author's goal `raw` with those of the engine's goal S. Returns (lm: author level -> engine level, renames: engine level -> the author's name, target_ok).

    The contexts are two versions of one: the engine's has cleared what the author's still has (the variable a split used up, copies the kernel makes), has introduced the binders the author's target still
    has, and has taken names of its own for what it introduced; the ORDER of what both have is the same. So the pairing is MONOTONE:
      1. the live hypotheses the two contexts name alike (the latest of a name on each side) are ANCHORS, those of them that agree in order (a longest increasing subsequence, live ones first);
      2. what lies between two anchors is aligned by TYPE, in order, to pair as many live hypotheses as possible (a search over the few ways; dead variables -- nothing mentions them -- come last and
         never take the engine's last hypotheses, the ones it introduced from the quantifiers the author's target still has);
    a pairing is accepted only if the types agree under the pairing made so far. The hypotheses of the engine that stay unpaired are the ones the author introduces later."""
    k = eng.k; nR, nS = raw.n_hyps, S.n_hyps; PROP = k.mk_type(0)
    men = [G.atoms_of(k, t)[1] for t in raw.hyp_types]
    tgt_atoms = G.atoms_of(k, raw.target)[1]
    def is_prop(types, i):
        ok, sort = k.infer_open(types[i], types[:i]); return bool(ok) and sort == PROP
    isprop = [is_prop(raw.hyp_types, l) for l in range(nR)]
    # the author's hypotheses that are DEAD: variables that nothing the script can still use mentions (a hypothesis whose name a later one has taken is out of reach, so it mentions nothing)
    shadowed = {l for l in range(nR) if raw.hyp_names[l] in raw.hyp_names[l + 1:]}
    mentioned = set(tgt_atoms)
    for l in range(nR):
        if l not in shadowed: mentioned |= men[l]
    dead = {l for l in range(nR) if l not in mentioned and not isprop[l]}
    weight = [1 if (l in dead or l in shadowed) else 1000 for l in range(nR)]
    try: ext = len(G.open_binders(k, raw.target, nR, G.MAX_INTROS)[0])
    except (E.ITPError, RuntimeError): ext = 0
    reserved = set(range(max(0, nS - ext), nS))
    retired = {j for j in range(nS) if S.hyp_names[j][:1] == RETIRED}
    symA = [head_syms(k, raw.hyp_types[l]) if isprop[l] else None for l in range(nR)]
    symS = {}
    memo = {}
    def fits(l, j, lm):
        key = (l, j) + tuple(lm.get(f) for f in sorted(men[l]))
        r = memo.get(key)
        if r is not None: return r
        r = False
        t = subst_levels(k, raw.hyp_types[l], lm, men[l])
        if t is not None:
            if t == S.hyp_types[j]: r = True
            elif isprop[l]:
                sa = symA[l]; sb = symS.get(j)
                if sb is None: sb = symS[j] = head_syms(k, S.hyp_types[j])
                if None in sa or None in sb or (sa & sb):
                    try:
                        r = (k.simp_normalize(t, j) == k.simp_normalize(S.hyp_types[j], j)            # (a premise the author introduced from a normalised goal, a hypothesis the engine introduced before the goal was normalised)
                             or conv_bounded(eng, t, S.hyp_types[j], j))                              # (`Not P` and `P -> False`)
                    except (E.ITPError, RuntimeError): r = False
        memo[key] = r
        return r
    # 1. anchors
    byname = collections.defaultdict(list)
    for j in range(nS - 1, -1, -1):
        if j not in retired: byname[S.hyp_names[j]].append(j)
    cands = []; used = set()
    for l in range(nR - 1, -1, -1):
        if l in dead or l in shadowed: continue
        for j in byname.get(raw.hyp_names[l], ()):
            if j not in used: cands.append((l, j)); used.add(j); break
    cands.sort()
    best = []                                                          # longest increasing subsequence in j, weighted
    for i, (l, j) in enumerate(cands):
        sc = weight[l]; prev = None
        for i2 in range(i):
            if cands[i2][1] < j and best[i2][0] + weight[l] > sc: sc = best[i2][0] + weight[l]; prev = i2
        best.append((sc, prev))
    anchors = []
    if cands:
        i = max(range(len(cands)), key=lambda x: best[x][0])
        while i is not None: anchors.append(cands[i]); i = best[i][1]
        anchors.reverse()
    # 2. walk: gaps between anchors
    lm = {}; renames = {}; taken = set()
    def align_gap(As, Es):
        As = [l for l in As]; Es = [j for j in Es if j not in taken and j not in retired]
        if not As or not Es: return
        n = len(As); suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1): suffix[i] = suffix[i + 1] + weight[As[i]]
        top = [0, []]; budget = [4000]; cur = dict(lm); pairs = []
        def rec(ia, ie, score):
            if score + suffix[ia] <= top[0]: return
            if ia == n: top[0] = score; top[1] = list(pairs); return
            budget[0] -= 1
            if budget[0] < 0: return
            l = As[ia]
            for je in range(ie, len(Es)):
                j = Es[je]
                if l in dead and j in reserved: continue
                if fits(l, j, cur):
                    cur[l] = j; pairs.append((l, j))
                    rec(ia + 1, je + 1, score + weight[l])
                    pairs.pop(); del cur[l]
            rec(ia + 1, ie, score)
        rec(0, 0, 0)
        for l, j in top[1]: pair(l, j)
    def pair(l, j):
        lm[l] = j; taken.add(j)
        if S.hyp_names[j] != raw.hyp_names[l]: renames[j] = raw.hyp_names[l]
    curA = 0; curE = 0; poolA = []; poolE = []
    anchored_a = {l for l, _ in anchors}
    for (l, j) in anchors + [(nR, nS)]:
        poolA += [a for a in range(curA, l) if a not in anchored_a]
        poolE += [e for e in range(curE, j)]
        align_gap(poolA, poolE); poolA = []; poolE = []
        if l < nR:
            if fits(l, j, lm): pair(l, j)
            else: poolA = [l]; poolE = [j]
        curA = l + 1; curE = j + 1
    for l in range(nR):                                  # a variable the script introduced AFTER a `have` (the engine has the binders of the goal already and the claim behind them): not in the engine's order, paired by type
        if l in lm or isprop[l] or l in dead or l in shadowed: continue
        for j in range(nS):
            if j in taken or j in retired: continue
            if fits(l, j, lm): pair(l, j); break
    for l in range(nR):                                  # a proposition the author has twice (`have h : P` when P was a hypothesis): the same fact
        if l in lm or not isprop[l]: continue
        for j in range(nS):
            if j not in retired and fits(l, j, lm): lm[l] = j; break
    tgt = subst_levels(k, raw.target, lm, tgt_atoms)
    ok = tgt is not None and (tgt == S.target or k.simp_normalize(tgt, nS) == k.simp_normalize(S.target, nS) or conv_bounded(eng, tgt, S.target, nS))
    return lm, renames, ok


def rename_hyps(p, i, renames):
    """give the hypotheses of goal i of the live proof state `p` these names"""
    ps = p.ps; names = p._names[ps.goal_meta_id(i)]
    for j, nm in renames.items():
        if names[j] != nm: names[j] = nm; ps.set_goal_hyp_name(i, j, nm)


class Pair:
    """the author's goal (a log node: the call to translate next) paired with the first engine goal: `lm` maps the author's context levels to the engine's"""
    __slots__ = ('node', 'lm')
    def __init__(self, node, lm): self.node = node; self.lm = lm


def follow(node):
    """the node of the next call that has to be translated: past the calls the engine makes by itself"""
    while node.op in SKIP_OPS and len(node.kids) == 1: node = node.kids[0]
    return node


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ probing
def run_probe(eng, g, fn):
    """run fn(p) on a fresh one-goal proof state made from g, then rewind the kernel: (the goals left as GoalData | None when they are not independent, None) or (None, why) when fn is refused"""
    k = eng.k; w = eng.w
    cp = k.checkpoint(); old_fuel = k.get_fuel(); w.probing += 1
    why = None; children = None
    try:
        p = Proof.from_goal(w, g.target, g.hyp_types, g.hyp_names)
        try: fn(p)
        except G.REFUSALS as e: why = '%s: %s' % (type(e).__name__, str(e).split('\n')[0][:200])
        if why is None: children = k.finish_probe(cp, p.ps)
    except BaseException:
        k.rewind(cp); raise
    finally: w.probing -= 1; k.set_fuel(old_fuel)
    if why is not None: k.rewind(cp); return None, why
    if children is None: return None, 'the goals are not independent'
    return [GoalData(t, hs, ns) for (t, hs, ns) in children], None


def probe_plan(eng, g, plan, menu):
    """the goals the actions of `plan` leave, applied one after the other to the first goal, starting from g alone (see run_probe)"""
    k = eng.k
    def go(p):
        for idx, act in enumerate(plan):
            if p.n_goals == 0: raise ActionError('no goal for the action %r' % (act,))
            k.set_fuel(eng.fuel_of(act))
            eng.step_in(p, act, menu if len(plan) == 1 else EMPTY_MENU)
    return run_probe(eng, g, go)


def settle_probe(eng, g, intros=0):
    """the goals left by settling g (what the engine does to every goal it makes): [] when it is closed. With `intros`, that many leading binders are introduced first: the engine's goals have their
    binders introduced before an action, the author's may not (`exact f` closes `a -> b` when `f : a -> b` is a hypothesis; the engine's goal `b`, with `x : a`, is not closed by `f`)."""
    def go(p):
        eng.k.set_fuel(eng.default_fuel)
        for _ in range(intros):
            if not eng._intro_at(p, 0): break
        eng.settle(p)
    kids, why = run_probe(eng, g, go)
    return kids if kids is not None else [g]


def is_data_goal(eng, g):
    """is the goal a type to be given a VALUE (a data argument that `apply` left as a goal), not a proposition?"""
    k = eng.k; ok, sort = k.infer_open(g.target, g.hyp_types)
    return bool(ok) and sort != k.mk_type(0)


def effective(eng, node, cache, extras=0):
    """the goals the author's call left that the engine has to prove too: those that settling does not close, each as the node of its next call that has to be translated. Two lists: with the goals
    settled as they are, and (when the engine has `extras` introduced binders the author has not) with those binders introduced first."""
    plain = []; intro = []
    for c in node.kids:
        if is_data_goal(eng, c.pre): continue                          # (a value the script gave later: the engine's action says it)
        key = (c.id, 0); closed = cache.get(key)
        if closed is None: closed = cache[key] = (len(settle_probe(eng, c.pre)) == 0)
        if not closed: plain.append(follow(c))
        if extras:
            key = (c.id, extras); closed = cache.get(key)
            if closed is None: closed = cache[key] = (len(settle_probe(eng, c.pre, extras)) == 0)
        if not closed: intro.append(follow(c))
    return plain, (intro if extras else plain)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ translating terms
class Ctx:
    """what the translation of one call needs: the engine's goal `g`, the author's goal `raw`, the pairing of their hypotheses `lm`, the term space and (when asked for) the menu"""
    def __init__(self, eng, g, raw, lm):
        self.eng = eng; self.k = eng.k; self.w = eng.w; self.g = g; self.raw = raw; self.lm = lm
        self.space = eng.space(g); self._menu = None
        paired = set(lm.values()); self.extras = [j for j in range(g.n_hyps) if j not in paired]
    @property
    def menu(self):
        if self._menu is None: self._menu = self.eng.menu(self.g)
        return self._menu
    def hyp(self, raw_level):
        j = self.lm.get(raw_level)
        if j is None: raise Gap('stale_ref', 'the author uses the hypothesis %s, which the engine does not have' % self.raw.hyp_names[raw_level])
        return j
    def node(self, term, extra=None):
        """the author's term as a term of the engine's goal (its hypotheses renumbered; `extra`: {level: engine level} for variables the author's term has that are not in its goal's context)"""
        n = term.n if isinstance(term, Term) else term
        repl = []
        for f in G.atoms_of(self.k, n)[1]:
            j = extra[f] if extra and f in extra else self.hyp(f)
            if j != f: repl.append((f, self.k.mk_fvar(j)))
        return self.k.replace_fvars(n, repl) if repl else n
    def code_node(self, n, expected=None):
        try: return self.space.encode(n, expected)
        except GB.BuildError as e: raise Gap('not_encodable', str(e).split('\n')[0][:200]) from None
    def code(self, term, expected=None): return self.code_node(self.node(term), expected)
    def source(self, n):
        """(token, Head) of a term that is a global or a hypothesis"""
        k = self.k; v = k.view(n)
        if v[0] == 'const':
            if v[1] >= self.eng.limit: raise Gap('premise', 'the global %d is not available' % v[1])
            h = self.space.table.heads.get(v[1])
            if h is None: raise Gap('head', 'the global %d cannot be used in a term' % v[1])
            return ('c', v[1], 0), h
        if v[0] == 'fvar': return ('v', v[1], 0), self.space.local_head(v[1])
        raise Gap('head', 'the head of the term is neither a global nor a hypothesis')
    def arg_codes(self, Hd, args):
        """the codes of explicit leading arguments of a source, each against its binder's type (the earlier arguments substituted)"""
        k = self.k; out = []; binding = {}
        for i, a in enumerate(args):
            if i >= Hd.arity: raise Gap('too_many_args', '%d arguments for a source with %d binders' % (len(args), Hd.arity))
            dom = GB.inst_beta(k, Hd.doms[i], binding)
            out.append(self.code_node(a, dom)); binding[GB.HOLE + i] = a
        return out


def data_solutions(ctx, node):
    """the values the script gave to the data goals that `node` (an `apply`) left, in order, as terms of the engine's goal: the goals are closed by `exact`"""
    sols = []
    for c in node.kids:
        if not is_data_goal(ctx.eng, c.pre): continue
        n = follow(c)
        if n.op != 'exact': raise Gap('data_goal', 'a data goal is closed by %s, not by exact' % n.op)
        sols.append(ctx.node(n.info['term']))
    return sols


def apply_codes(ctx, tok, Hd, args, sols=()):
    """the code of `source args ?_ ..` (the premises the author did not give are goals) for the goal of ctx, and the codes of its data arguments when the author gave no proof: (code, data codes | None).
    `sols`: the values the author gave to the data goals that were left (see data_solutions), for the data arguments that neither the author's arguments nor the goal determine"""
    k = ctx.k; g = ctx.g; sols = iter(sols)
    hd, _ = GB.open_pis(k, g.target); kk = Hd.arity - len(hd)
    if kk < 0: raise Gap('apply_shape', 'the goal has more binders than the source')
    b = ctx.space.builder(g.target); b.allow_goals = True
    try: b.push((tok[0], tok[1], kk))
    except GB.BuildError as e: raise Gap('apply_shape', str(e).split('\n')[0][:200]) from None
    data = []; pure = True
    while not b.done:
        f = b.stack[-1]; j = f.j; H = b.hole_type
        if j < len(args) and args[j] is not None:
            code = ctx.code_node(args[j], H)
            if b.is_prop(H): pure = False
            else: data.append(code)
            try:
                for t in code: b.push(t)
            except GB.BuildError as e: raise Gap('apply_arg', str(e).split('\n')[0][:200]) from None
        elif b.is_prop(H):
            try: b.push(('g',))
            except GB.BuildError as e: raise Gap('apply_goal', str(e).split('\n')[0][:200]) from None
        else:
            t = next(sols, None)
            if t is None: raise Gap('undetermined', 'the argument %d (%s) is determined by nothing the author or the goal said' % (j, b._show(H)))
            code = ctx.code_node(t, H); data.append(code)
            try:
                for tk in code: b.push(tk)
            except GB.BuildError as e: raise Gap('apply_arg', str(e).split('\n')[0][:200]) from None
    return tuple(b.code), (data if pure else None)


def forward_plans(ctx, proof):
    """the plans that add the fact `proof` (an engine term: a source applied to arguments) to the hypotheses: FORWARD in its compact form (the arguments that the others determine left out), then with
    every argument said"""
    k = ctx.k; g = ctx.g; space = ctx.space
    head, args = GB.spine(k, proof)
    tok, Hd = ctx.source(head)
    n = len(args)
    if n > Hd.arity: raise Gap('forward_args', '%d arguments for a source with %d binders' % (n, Hd.arity))
    ctxt = [] if tok[0] == 'c' else list(g.hyp_types)
    try: real, _ = G.open_binders(k, Hd.ty, len(ctxt))
    except (E.ITPError, RuntimeError) as e: raise Gap('forward_shape', str(e)[:200]) from None
    prem = []
    for j in range(n):
        ok, sort = k.infer_open(real[j], ctxt + real[:j])
        prem.append(bool(ok) and sort == space.prop)
    slots = []; binding = {}
    for i, a in enumerate(args):
        dom = GB.inst_beta(k, Hd.doms[i], binding)
        av = k.view(a)
        if prem[i] and av[0] == 'fvar' and av[1] < g.n_hyps: slots.append((('v', av[1], 0),))
        else: slots.append(ctx.code_node(a, dom))
        binding[GB.HOLE + i] = a
    plans = []
    compact = list(slots)
    for i in range(n):
        if prem[i]: continue
        trial = compact[:i] + [()] + compact[i + 1:]
        try: vals = ACTS.resolve_slots(ctx.eng, space, Hd.ty, trial)
        except (ActionError, GB.BuildError, E.ITPError, RuntimeError): continue
        if vals[i] == args[i]: compact = trial
    if compact != slots: plans.append([Action(Kind.FORWARD, 0, (), ((tok,),) + tuple(compact))])
    plans.append([Action(Kind.FORWARD, 0, (), ((tok,),) + tuple(slots))])
    return plans


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ translators
def has_meta(k, node):
    tags = k.observe_term(node)[0]
    return bool((tags == E.OBS_TAG_NAMES.index('META')).any())


def solution_call(ctx, node):
    """(head, arguments) of the call that the kernel made for an `apply`, as terms of the engine's goal; an argument the kernel left open (a premise that became a goal, a value that a later call gave) is None.
    None when the log has no solution. The solution has the unifier's findings in it: the values of the arguments that only unification could find."""
    sol = node.info.get('sol')
    if sol is None: return None
    k = ctx.k; n0 = node.pre.n_hyps; lam = []
    while k.view(sol)[0] == 'lam':                                   # (the binders of a goal that is still a Pi: the engine has introduced them)
        sol = k.instantiate(k.view(sol)[2], [k.mk_fvar(n0 + len(lam))]); lam.append(n0 + len(lam))
    if len(lam) > len(ctx.extras): return None
    extra = {lvl: ctx.extras[i] for i, lvl in enumerate(lam)}
    hd, args = GB.spine(k, sol)
    if k.view(hd)[0] not in ('const', 'fvar'): return None
    out = []
    for a in args:
        if has_meta(k, a): out.append(None)
        else:
            try: out.append(ctx.node(a, extra))
            except Gap: out.append(None)
    return ctx.node(hd, extra), out


def tr_apply(ctx, node):
    info = node.info; k = ctx.k
    call = solution_call(ctx, node)
    if call is not None: hd, args = call
    else:
        hd, hargs = GB.spine(k, ctx.node(info['head']))
        args = hargs + [ctx.node(a) for a in info['args']]
    tok, Hd = ctx.source(hd)
    plans = []; err = None
    author_hd, author_hargs = GB.spine(k, ctx.node(info['head']))
    if not info['args'] and not author_hargs:                           # the author named the lemma and nothing else: the menu's APPLY is that call
        for a in ctx.menu.actions:
            if a.kind == Kind.APPLY and ctx.menu.cands[a.arg][0] == author_hd: plans.append([a])
    try:
        code, data = apply_codes(ctx, tok, Hd, args, data_solutions(ctx, node))
        if data is not None: plans.append([Action(Kind.APPLY_WITH, 0, (), ((tok,),) + tuple(data))])
        plans.append([Action(Kind.REFINE, 0, (), (code,))])
    except Gap as e: err = e
    lead = []
    for a in args:
        if a is None: break
        lead.append(a)
    try: plans.append([Action(Kind.APPLY_WITH, 1, (), ((tok,),) + tuple(ctx.arg_codes(Hd, lead)))])           # the leading arguments that are known: the unifier finds the other arguments
    except Gap as e: err = err or e
    if not plans: raise err
    return plans


def tr_exact(ctx, node):
    k = ctx.k; t = ctx.node(node.info['term']); cands = [t]
    if ctx.extras:                                                  # the author's goal is still a Pi: the engine has introduced its binders
        ext = t
        for j in ctx.extras: ext = k.mk_app(ext, k.mk_fvar(j))
        cands.append(GB.beta_head(k, ext))
    plans = []; err = None
    for c in cands:
        try: plans.append([Action(Kind.REFINE, 0, (), (ctx.code_node(c, ctx.g.target),))])
        except Gap as e: err = err or e
    if not plans: raise err
    return plans


def tr_have(ctx, node):
    info = node.info; plans = []; k = ctx.k; g = ctx.g
    P = ctx.node(info['P'])
    if P in g.hyp_types or k.simp_normalize(P, g.n_hyps) in {k.simp_normalize(t, g.n_hyps) for t in g.hyp_types}: return [[]]          # the author states a hypothesis again: the engine has the fact already
    if info['proof'] is not None:
        pf = ctx.node(info['proof'])
        try: plans += forward_plans(ctx, pf)
        except Gap: pass
        plans.append([Action(Kind.CLAIM, 0, (), (ctx.code_node(P, ctx.space.prop),)), Action(Kind.REFINE, 0, (), (ctx.code_node(pf, P),))])
    else:
        plans.append([Action(Kind.CLAIM, 0, (), (ctx.code_node(P, ctx.space.prop),))])
    return plans


def tr_cases(ctx, node):
    info = node.info; lvl = ctx.hyp(info['lvl'])
    return [[Action(Kind.CASES, lvl, (), (), tuple(sorted({ctx.hyp(l) for l in info['gen']})))]]


def tr_induct(ctx, node):
    """induction on a hypothesis. The author's goal may still have its quantifiers (`forall m, P n m`), the engine's has introduced them: the induction hypothesis is then about the one `m`; so the
    hypotheses that the author's goal has not introduced yet are generalised"""
    info = node.info; lvl = ctx.hyp(info['lvl'])
    gen = {ctx.hyp(l) for l in info['gen']} | {j for j in ctx.extras if j > lvl}
    return [[Action(Kind.INDUCT, lvl, (), (), tuple(sorted(gen)))], [Action(Kind.INDUCT, lvl, (), (), ())]]


def rw_plans(ctx, lemma, extra, rev, kind, hyps=()):
    k = ctx.k
    hd, hargs = GB.spine(k, ctx.node(lemma))
    args = hargs + [ctx.node(a) for a in extra]
    tok, Hd = ctx.source(hd)
    plans = []
    if kind == Kind.REWRITE_WITH and not args:
        for a in ctx.menu.actions:
            if a.kind == Kind.REWRITE and ctx.menu.cands[a.arg][0] == hd and bool(ctx.menu.cands[a.arg][1]) == bool(rev): plans.append([a])
    codes = ctx.arg_codes(Hd, args)
    plans.append([Action(kind, int(bool(rev)), (), ((tok,),) + tuple(codes), tuple(hyps))])
    if kind == Kind.REWRITE_WITH:                                       # the author's goal is still `A -> B` (a negation: `A -> False`) and the engine has introduced `A`: rewriting the goal is rewriting there
        for j in reversed(ctx.extras): plans.append([Action(Kind.REWRITE_AT, int(bool(rev)), (), ((tok,),) + tuple(codes), (j,))])
    return plans


def tr_rw(ctx, node): return rw_plans(ctx, node.info['lemma'], node.info['args'], node.info['rev'], Kind.REWRITE_WITH)
def tr_rewrite(ctx, node): return rw_plans(ctx, node.info['eq'], (), node.info['rev'], Kind.REWRITE_WITH)
def tr_rw_at(ctx, node): return rw_plans(ctx, node.info['lemma'], node.info['args'], node.info['rev'], Kind.REWRITE_AT, (ctx.hyp(node.info['hyp']),))


def tr_exists(ctx, node):
    A = ctx.eng._exists_type(ctx.g.target)
    if A is None: raise Gap('exists_shape', 'the goal is not an existential statement')
    return [[Action(Kind.USE, 0, (), (ctx.code(node.info['witness'], A),))]]


def tr_simple(kind):
    def tr(ctx, node): return [[Action(kind, 0)]]
    return tr


def tr_arith(kind):
    def tr(ctx, node):
        pre = []
        for e in node.info['extra']:
            pre.append(forward_plans(ctx, ctx.node(e))[0][0])
        plans = [pre + [Action(kind, 0)]]
        if pre: plans.append([Action(kind, 0)])
        return plans
    return tr


def tr_lincomb(ctx, node):
    pairs = node.info['pairs']
    with ctx.eng.in_context(ctx.g.hyp_types, ctx.g.hyp_names):
        try: A, _, _ = ctx.w.eq_parts(ctx.w.T(ctx.g.target))
        except AuthorError: raise Gap('lincomb_shape', 'the goal is not an equation') from None
    codes = []; hyps = []
    for c, l in pairs:
        codes.append((('n', c),) if isinstance(c, int) else ctx.code(c, A.n)); hyps.append(ctx.hyp(l))
    return [[Action(Kind.LINCOMB, 0, (), tuple(codes), tuple(hyps))]]


def tr_induction_on(ctx, node):
    info = node.info; n = ctx.node(info['lemma']); v = ctx.k.view(n)
    if v[0] != 'const': raise Gap('head', 'the principle is not a global')
    return [[Action(Kind.INDUCT_WITH, 0, (), (((('c', v[1], 0),)),), (ctx.hyp(info['lvl']),))]]


def tr_interval_cases(ctx, node):
    info = node.info; h = ctx.hyp(info['lvl']); plans = []
    for a in ctx.menu.actions:
        if a.kind == Kind.INTERVAL_CASES and ctx.menu.cands[a.arg][2] == (h, info['lo'], info['hi']): plans.append([a])
    if not plans: raise Gap('interval_bounds', 'the engine offers no split of hypothesis %d into %d..%d' % (h, info['lo'], info['hi']))
    return plans


def tr_closer(ctx, node): raise Gap('closer', '%s is a step the engine does by settling the goal; it did not close this one' % node.op)


TRANSLATORS = {'apply': tr_apply, 'exact': tr_exact, 'have': tr_have, 'cases': tr_cases, 'induct': tr_induct, 'rw': tr_rw, 'rewrite': tr_rewrite, 'rw_at': tr_rw_at, 'exists': tr_exists,
               'ring': tr_simple(Kind.RING), 'linarith': tr_arith(Kind.LINARITH), 'nlinarith': tr_arith(Kind.NLINARITH), 'linear_combination': tr_lincomb, 'by_contra': tr_simple(Kind.BY_CONTRA),
               'interval_cases': tr_interval_cases, 'induction_on': tr_induction_on, 'refl': tr_closer, 'assumption': tr_closer, 'try_refl': tr_closer}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ replaying
class Result:
    """the outcome of replaying one recording: `ok` (the proof by actions was checked by the kernel), `gap` (a Gap, the first step that failed), `actions` (the signatures of the actions taken, in
    order), `ops` (the author's calls translated, by name), `kinds` (the actions by kind)"""
    def __init__(self, rec):
        self.rec = rec; self.ok = False; self.gap = None; self.actions = []; self.ops = collections.Counter(); self.kinds = collections.Counter()
        self.soft = collections.Counter(); self.where = None; self.seconds = 0.0; self.view = None; self.portable = []
    def record(self, act, menu, eng=None):
        self.kinds[Kind(act.kind).name] += 1
        self.actions.append((act, menu.signature(act)))
        self.portable.append(ACTS.to_portable(eng, menu, act) if eng is not None else None)


def make_pair(eng, p, i, node, res, retire=()):
    if retire:                                                         # (the engine's extras that the goal's own proof does not introduce)
        names = p._names[p.ps.goal_meta_id(i)]
        rename_hyps(p, i, {j: RETIRED + names[j] for j in retire if names[j][:1] != RETIRED})
    g = eng.goal_of(p, i)
    lm, renames, ok = align(eng, node.pre, g)
    if renames: rename_hyps(p, i, renames)
    if not ok and not G.unfolds_to_pi(eng.k, node.pre.target): res.soft['target_differs'] += 1
    return Pair(node, lm)


def show_info(w, info):
    """a readable line for the arguments of a logged call"""
    out = []
    for key, val in (info or {}).items():
        def one(x):
            if isinstance(x, Term):
                try: return w.show(x)
                except Exception: return '<term>'
            if isinstance(x, (list, tuple)): return '[' + ', '.join(one(y) for y in x) + ']'
            return repr(x)
        out.append('%s=%s' % (key, one(val)))
    return ' '.join(out)


def view_of(eng, rec, pairs, p):
    """(the engine's first goal, the author's goal and call that failed to translate) as text, for the report"""
    try:
        node = pairs[0].node; g = eng.goal_of(p, 0)
        return {'engine': eng.render(g), 'author': eng.render(node.pre), 'call': '%s %s' % (node.op, show_info(eng.w, node.info))}
    except Exception as e: return {'engine': '?', 'author': '?', 'call': 'view failed: %s' % (e,)}


def replay(eng, rec, trace=None, premises=None):
    """replay a recording in the engine (see the module documentation): a Result. `trace`: a list that receives, for every step, the author's call, both goals and the actions taken;
    `premises`: the number of globals the engine offers (default: every global before the theorem; fewer hides the lemmas written for it -- the proof must not need them)"""
    import traceback
    res = Result(rec); t0 = time.time(); cache = {}
    eng.premises = rec.gid if premises is None else premises; old_tag = eng.name_tag; eng.name_tag = TAG
    try:
        p = eng.start(rec.stmt)
        pairs = [make_pair(eng, p, 0, follow(rec.root), res)] if p.n_goals else []
        while p.n_goals:
            pr = pairs[0]; node = pr.node; g = eng.goal_of(p, 0)
            res.where = (node.op, node.id)
            fn = TRANSLATORS.get(node.op)
            if fn is None: raise Gap('op', 'no translation for the call %r' % (node.op,))
            ctx = Ctx(eng, g, node.pre, pr.lm)
            kidsA, kidsB = effective(eng, node, cache, len(ctx.extras))
            plans = fn(ctx, node)
            chosen = None; failures = []
            for plan in plans:
                if not plan:                                                   # nothing to do: the author's call changes nothing the engine has not got (a `have` of a hypothesis)
                    if len(kidsA) == 1 and kidsA[0].pre is not node.pre: chosen = (plan, EMPTY_MENU, kidsA); break
                    failures.append('nothing to do, but the author\'s call left %d goals' % len(kidsA)); continue
                menu = ctx.menu if len(plan) == 1 else EMPTY_MENU
                out, why = probe_plan(eng, g, plan, menu)
                if out is None: failures.append(why); continue
                kids = kidsA if len(out) == len(kidsA) else kidsB if len(out) == len(kidsB) else None
                if kids is None: failures.append('%s leaves %d goals, the author\'s call %d' % ([Kind(a.kind).name for a in plan], len(out), len(kidsA))); continue
                if any(c.key == g.key for c in out): failures.append('no progress'); continue
                chosen = (plan, menu, kids); break
            if chosen is None: raise Gap('probe_%s' % node.op, '; '.join(failures)[:400])
            plan, menu, kids = chosen; n0 = p.n_goals
            if trace is not None: trace.append({'call': '%s %s' % (node.op, show_info(eng.w, node.info)), 'author': eng.render(node.pre), 'engine': eng.render(g), 'plan': [repr(a) for a in plan]})
            for act in plan:
                eng.replay(p, act, menu if len(plan) == 1 else EMPTY_MENU)
                res.record(act, menu if len(plan) == 1 else EMPTY_MENU, eng)
            made = p.n_goals - n0 + 1
            if made != len(kids): raise Gap('desync', 'the engine made %d goals, expected %d' % (made, len(kids)))
            retire = ctx.extras if (len(plan) == 1 and plan[0].kind == Kind.CLAIM) else ()         # the claim is proved without the binders that the engine has introduced from the goal it is part of
            pairs[0:1] = [make_pair(eng, p, i, kids[i], res, retire if i == 0 else ()) for i in range(made)]
            res.ops[node.op] += 1
        eng.verify(rec.stmt, p)
        res.ok = True
    except Gap as e: res.gap = e; res.view = view_of(eng, rec, pairs, p)
    except Exception as e: res.gap = Gap('bug', '%s: %s\n%s' % (type(e).__name__, str(e).split('\n')[0][:200], traceback.format_exc(limit=6)))
    finally: eng.name_tag = old_tag
    res.seconds = time.time() - t0
    return res


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the report
def replay_all(recs, eng=None, names=None):
    """replay recordings (all by default): {name: Result}"""
    if eng is None:
        w = next(iter(recs.values())).w; eng = G.Engine(w)
    return {nm: replay(eng, rec) for nm, rec in recs.items() if names is None or nm in names}


def group_of(name):
    """the part of the library a theorem belongs to, by its name: `Z.add_comm` -> Z, `le_trans` -> (core)"""
    return name.split('.')[0] if '.' in name else '(core)'


def report(results, out=None):
    """print what a replay of many recordings achieved: coverage by part of the library, the kinds of action used, the steps of the author's scripts that were translated, and the gaps (the first step
    that failed, by reason, with examples). Returns the numbers as a dict."""
    out = out or sys.stdout
    n = len(results); ok = sum(1 for r in results.values() if r.ok)
    groups = collections.defaultdict(lambda: [0, 0])
    for nm, r in results.items():
        g = groups[group_of(nm)]; g[0] += 1; g[1] += r.ok
    kinds = collections.Counter(); ops = collections.Counter(); steps = 0
    for r in results.values():
        if r.ok: kinds.update(r.kinds); ops.update(r.ops); steps += len(r.actions)
    reasons = collections.Counter(); ex = collections.defaultdict(list)
    for nm, r in results.items():
        if not r.ok: reasons[r.gap.reason] += 1; ex[r.gap.reason].append((nm, r.where[0] if r.where else '-', r.gap.detail[:140].replace('\n', ' ')))
    secs = sum(r.seconds for r in results.values())
    p = lambda *a: print(*a, file=out)
    p('replayed as kernel-checked action sequences: %d of %d (%.1f%%), %d actions, %.1f s' % (ok, n, 100.0 * ok / max(n, 1), steps, secs))
    p('by part of the library: ' + ', '.join('%s %d/%d' % (g, v[1], v[0]) for g, v in sorted(groups.items())))
    p('actions by kind: ' + ', '.join('%s %d' % kv for kv in kinds.most_common()))
    p('author calls translated: ' + ', '.join('%s %d' % kv for kv in ops.most_common()))
    if reasons:
        p('gaps:')
        for rs, c in reasons.most_common():
            p('  %3d  %s' % (c, rs))
            for e in ex[rs][:3]: p('         %s' % (e,))
    else: p('no gaps')
    return {'n': n, 'ok': ok, 'actions': steps, 'groups': {g: tuple(v) for g, v in groups.items()}, 'kinds': dict(kinds), 'gaps': dict(reasons)}


def demonstrations(results):
    """the replayed proofs as plain data, ready for JSON: [{'name', 'gid', 'steps': [portable actions]}] for those that were checked (see gym_acts.to_portable / replay_portable)"""
    return [{'name': nm, 'gid': r.rec.gid, 'steps': r.portable} for nm, r in results.items() if r.ok]


def main(argv):
    """python3 recorder.py [--json FILE] [LIBRARY]    replay a library's proofs (default imo_lib.build_imo) and report"""
    import importlib
    path = None; lib = 'imo_lib.build_imo'; it = iter(argv)
    for a in it:
        if a == '--json': path = next(it)
        else: lib = a
    mod, fn = lib.rsplit('.', 1)
    build = getattr(importlib.import_module(mod), fn)
    t0 = time.time(); w, recs = record_world(build)
    print('recorded %d proofs (%d logged calls) in %.1f s' % (len(recs), sum(r.size for r in recs.values()), time.time() - t0))
    results = replay_all(recs, G.Engine(w))
    stats = report(results)
    if path:
        with open(path, 'w') as f: json.dump({'library': lib, 'globals': w.k.num_globals(), 'demonstrations': demonstrations(results)}, f)
        print('wrote %d demonstrations to %s' % (stats['ok'], path))
    return 0 if stats['ok'] == stats['n'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
