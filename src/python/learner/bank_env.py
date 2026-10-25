"""bank_env.py -- the problem bank as a Gymnasium environment, at the level of the policy's DECISIONS.

    env = BankEnv(bank, names=splits['train'], regime='prefix')
    obs, info = env.reset(options={'problem': 'imo1959_p1'})
    while True:
        a = int(np.argmax(np.where(info['action_mask'], scores, -1e9)))      # an option of the current decision
        obs, reward, terminated, truncated, info = env.step(a)

One episode is one proof attempt.  The state is a proof of the problem's statement; the agent works on its first open goal.  To act, it answers the engine's questions (`gym_policy.Assembler`): which kind of
action, which lemma, which hypothesis, then -- for the actions that carry terms -- the next token of the term it says, each among the OPTIONS the engine offers at that point.  When the last question is answered
the action is probed and, if it applies, taken; a proof that closes every goal is checked by the kernel and ends the episode with `solve_reward`.  There is no text interface anywhere: the observation is the goal
as token sequences and the options as integer rows.

GOALS AND HINDSIGHT.  `reset(options={'goal': {...}})` starts an episode on ANY closed statement given as portable data (`statement_code`, `premises` -- the dict of a `gym_tree.Example` is such a goal),
`options={'statement': term, 'premises': n}` on a statement in this world, `options={'budget': b}` limits this episode to b actions (taken or refused; the default is `max_actions`).  With `track=True` the episode
keeps its proof tree (gym_tree.py): `info['achieved']` lists what each action achieved (a subtree closed, a fact derived) and `env.hindsight()` returns the theorems the episode proved, finished or not, each
replayed and kernel-checked -- new problems, with proofs, from an attempt that may have failed.

OBSERVATION (a Dict of arrays, padded; `n_options` and `action_mask` say how much is real):
    tokens, seg, hyp_of            the goal: `[CLS] hyp_1 [SEP] ... [TGT] target` as token ids, the segment of each token, the hypothesis it belongs to (-1: none)
    n_hyps, goals_left             number of hypotheses of the goal, number of open goals
    budget                         actions (taken or refused) left before the episode is truncated
    overlap                        (globals, 2): the fraction of each global's symbols (type, conclusion) that the goal mentions
    choice                         the kind of decision (`ml_features.CHOICE_NAMES`), the kind of action it belongs to (-1: not yet known), the position inside a term
    hole                           the tokens of the type of the term being said (padding 0)
    rows, n_options, mask          the options: (otype, a, b, flags) rows, how many, which are legal
ACTION   Discrete(max_options): the index of the option (`action_masks()` says which are legal; an illegal index raises ValueError, or with `illegal='penalize'` loses the decision and pays `fail_penalty`).   REWARD   `solve_reward` (default 1) when the proof is finished and kernel-checked; `-action_penalty` for every action taken, `-fail_penalty`
for every action that did not apply.   TRUNCATION   after `max_actions` actions (taken or refused) or `max_decisions` decisions.
`info` has the problem, the number of actions taken and refused (= probes), the budget left (`remaining`), `achieved` (with `track`) and -- when `rich` -- the raw `Choice` and `GoalFeat` that the learners of this repository use to recompute log-probabilities.
Nothing the agent does can make a false theorem provable: the kernel checks every finished proof.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itp_env_lnl as E
import gym_env as G, gym_policy as P
from gym_env import ActionError
import ml_features as F, ml_data as D
from author import AuthorError


class BankEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, bank, names=None, regime='prefix', ctx=None, engine=None, max_actions=60, max_decisions=600, max_options=1024, max_tokens=512,
                 solve_reward=1.0, action_penalty=0.0, fail_penalty=0.0, seed=0, rich=True, lemmas=None, illegal='raise', track=False, library=None):
        super().__init__()
        self.bank = bank; self.regime = regime; self.track = track
        self.library = library                          # ml_lemmas.LemmaLibrary: the lemmas the agent proved, which the problems and goals of this environment may use besides what their premises say
        self.lemmas = dict(lemmas or {})                # name -> (statement, premises): library lemmas to prove, besides the problems of the bank
        self.names = list(names if names is not None else bank.placed)
        self.eng = engine or G.Engine(bank.w, kinds=set(G.Kind), strict=False)
        self.ctx = ctx or D.DataContext(bank.w)
        self.max_actions = max_actions; self.max_decisions = max_decisions; self.max_options = max_options; self.max_tokens = max_tokens
        self.solve_reward = solve_reward; self.action_penalty = action_penalty; self.fail_penalty = fail_penalty; self.rich = rich
        if illegal not in ('raise', 'penalize'): raise ValueError("illegal is 'raise' or 'penalize'")
        self.illegal = illegal                         # an option index outside the current decision's options: an error, or a decision lost with the penalty of a refused action
        self.rng = random.Random(seed)
        n_glob = F.GMAX if library is not None else self.ctx.bags.n          # (a library grows: the observation has room for every global the policy can represent)
        self.n_globals = n_glob
        self.observation_space = spaces.Dict({
            'tokens': spaces.Box(0, F.VOCAB, (max_tokens,), np.int32), 'seg': spaces.Box(0, 2, (max_tokens,), np.int8), 'hyp_of': spaces.Box(-1, 4096, (max_tokens,), np.int16),
            'n_hyps': spaces.Box(0, 4096, (), np.int32), 'goals_left': spaces.Box(0, 4096, (), np.int32), 'budget': spaces.Box(0, 1 << 20, (), np.int32),
            'overlap': spaces.Box(0.0, 1.0, (n_glob, 2), np.float32),
            'choice': spaces.Box(-1, 1 << 20, (3,), np.int32),
            'hole': spaces.Box(0, F.VOCAB, (F.MAX_HOLE_TOKENS,), np.int32),
            'rows': spaces.Box(-(1 << 31), (1 << 31) - 1, (max_options, 4), np.int64), 'n_options': spaces.Box(0, max_options, (), np.int32),
            'mask': spaces.Box(0, 1, (max_options,), np.int8)})
        self.action_space = spaces.Discrete(max_options)
        self.ep = None; self.name = None; self.asm = None; self.choice = None; self.gf = None; self.ov = None
        self.actions = 0; self.fails = 0; self.decisions = 0; self.solved = False; self.term = None; self.budget = max_actions; self._achieved = []
        self.trajectory = []                           # per action: (GoalFeat, overlap, [(Choice, index)], taken) -- what a learner needs to recompute the log-probabilities; taken is True (the engine took
                                                       # the action), False (it refused it) or None (the episode ended before the action was complete)
        self.meta = []                                 # per trajectory entry: (actions of budget left, open goals) when its assembly began -- what a critic needs besides the goal
        self._cur = []; self._cur_meta = (max_actions, 1)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ episodes
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.rng = random.Random(seed)
        o = options or {}
        if 'goal' in o or 'statement' in o: name, stmt, prem = self._given_goal(o)
        else:
            name = o.get('problem') or self.rng.choice(self.names)
            regime = o.get('regime', self.regime)
            if name in self.lemmas: stmt, prem = self.lemmas[name]
            else:
                pl = self.bank.placed[name]; stmt, prem = pl.stmt, self._widen(pl.premises(regime))
        budget = o.get('budget')
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1): raise ValueError('budget: a positive number of actions, not %r' % (budget,))
        ep = G.Episode(self.eng, stmt, name, premises=prem, track=self.track)
        if ep.done: raise ValueError('%s: the engine closes this statement by itself, before any action: there is nothing to prove' % (name,))
        self.budget = self.max_actions if budget is None else int(budget)
        self.name = name; self.ep = ep
        self.actions = 0; self.fails = 0; self.decisions = 0; self.solved = False; self.term = None; self.trajectory = []; self.meta = []; self._cur = []; self._achieved = []
        self._begin_goal()
        return self._obs(), self._info()

    def _given_goal(self, o):
        """(name, statement, premises) of a goal that `reset` was given: `goal` = portable data with `statement_code` and `premises` (a gym_tree.Example's dict is one), or `statement` = a Term of this
        world with `premises`. A code that says no closed statement is a ValueError (the data is the caller's)."""
        import gym_tree as GTR, gym_build as GB
        if 'goal' in o:
            g = o['goal']
            if not isinstance(g, dict) or g.get('statement_code') is None: raise ValueError("goal: a dict with a 'statement_code' (a list of tokens) and 'premises'")
            prem = self._widen(g.get('premises', None))
            try: stmt = GTR.statement_from_code(self.eng, g['statement_code'], prem)
            except (GB.BuildError, TypeError, ValueError) as e: raise ValueError('goal: the statement code says no statement: %s' % (e,)) from None
            name = o.get('name') or g.get('name') or 'goal:%s' % (g.get('statement_digest') or GTR.digest(self.eng.w.show(stmt)))
        else:
            stmt, prem = o['statement'], self._widen(o.get('premises', None)); name = o.get('name') or 'statement'
        return name, stmt, prem

    def _widen(self, prem):
        """the premises of a task, widened by the lemmas of the library (when the environment has one)"""
        return prem if self.library is None else self.library.premises(prem)

    def _begin_goal(self):
        ep = self.ep
        if ep.done: self.choice = None; return
        g = ep.goal(); m = ep.menu()
        self.gf = F.goal_features(self.ctx.ser, g); self.ov = self.ctx.bags.overlap(self.gf.symbols)
        self.asm = P.Assembler(self.eng, g, m, rare=True); self._cur = []; self._cur_meta = (self._remaining(), ep.n_goals)
        self.choice = self.asm.start()
        if self.choice is None: self._refused(None)

    def _refused(self, action):
        """the assembled action cannot be taken: the same goal again, a new assembly"""
        self.fails += 1
        if self.rich and self._cur: self.trajectory.append((self.gf, self.ov, self._cur, False)); self.meta.append(self._cur_meta)
        self._cur = []
        if self.actions + self.fails >= self.budget: self.choice = None; return
        ep = self.ep; self._cur_meta = (self._remaining(), ep.n_goals)
        self.asm = P.Assembler(self.eng, ep.goal(), ep.menu(), rare=True); self.choice = self.asm.start()
        if self.choice is None: self.choice = None

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ step
    def step(self, a):
        a = int(a)
        if self.choice is None: raise RuntimeError('the episode is over')
        ch = self.choice
        reward = 0.0; terminated = False
        if not 0 <= a < len(ch.options):
            if self.illegal == 'raise': raise ValueError('option %d of %d' % (a, len(ch.options)))
            self.decisions += 1; reward -= self.fail_penalty
            truncated = self.decisions >= self.max_decisions
            if truncated: self.choice = None
            return (self._obs() if self.choice is not None else self._empty_obs()), reward, False, truncated, self._info()
        self.decisions += 1; self._achieved = []
        if self.rich: self._cur.append((ch, a))
        nxt = self.asm.choose(a)
        if nxt is not None:
            self.choice = nxt
        else:                                                                    # the action is complete
            action = self.asm.action
            taken = False
            if action is not None:
                try:
                    self.ep.step(action); taken = True
                except (ActionError, AuthorError, E.ITPError, RuntimeError) as e:
                    if isinstance(e, E.FuelExhausted): pass
            if taken:
                self.actions += 1; reward -= self.action_penalty
                if self.track: self._achieved = [{'kind': kind, 'node': n.id, 'step': n.step, 'depth': n.depth, 'length': (n.last - n.step + 1) if n.closed else 1} for kind, n in self.ep.events]
                if self.rich: self.trajectory.append((self.gf, self.ov, self._cur, True)); self.meta.append(self._cur_meta)
                self._cur = []
                if self.ep.done:
                    try:
                        self.term = self.ep.finish(); self.solved = True; reward += self.solve_reward; terminated = True
                    except Exception: terminated = True
                    self.choice = None
                elif self.actions + self.fails >= self.budget: self.choice = None
                else: self._begin_goal()
            else:
                reward -= self.fail_penalty
                self._refused(action)
        truncated = (not terminated) and (self.choice is None or self.decisions >= self.max_decisions)
        if self.decisions >= self.max_decisions: self.choice = None if not terminated else self.choice
        if self.choice is None and self._cur and self.rich:                         # the episode ended in the middle of an action: its decisions were made, the action never was
            self.trajectory.append((self.gf, self.ov, self._cur, None)); self.meta.append(self._cur_meta); self._cur = []
        obs = self._obs() if self.choice is not None else self._empty_obs()
        return obs, reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ observations
    def _info(self):
        info = {'problem': self.name, 'actions': self.actions, 'refused': self.fails, 'decisions': self.decisions, 'solved': self.solved, 'remaining': self._remaining()}
        if self.track: info['achieved'] = list(self._achieved)
        if self.choice is not None:
            n = len(self.choice.options); m = np.zeros(self.max_options, dtype=np.int8); m[:min(n, self.max_options)] = 1
            info['action_mask'] = m
            if self.rich: info['choice'] = self.choice; info['goal_feat'] = self.gf; info['overlap'] = self.ov; info['in_action'] = len(self._cur)
        return info

    def _remaining(self): return max(0, self.budget - self.actions - self.fails)

    def hindsight(self, **kw):
        """the theorems this episode proved so far (finished or not), as gym_tree.Example objects: closed subtrees and one-step derived facts, each replayed and kernel-checked (keywords: those of
        `ProofTree.examples`). Needs `track=True`."""
        if not self.track or self.ep is None or self.ep.tree is None: raise RuntimeError('hindsight needs an environment made with track=True and an episode that has begun')
        return self.ep.tree.examples(**kw)

    def episode_record(self):
        """the episode as plain data (JSON-ready): the problem, the outcome, the counts, and -- with `track` -- the portable form of every action taken (None for one that has none)"""
        d = {'problem': self.name, 'solved': self.solved, 'actions': self.actions, 'refused': self.fails, 'decisions': self.decisions, 'budget': self.budget}
        if self.track and self.ep is not None and self.ep.tree is not None: d['steps'] = [n.portable for n in self.ep.tree.by_step]
        return d

    def action_masks(self):
        m = np.zeros(self.max_options, dtype=np.int8)
        if self.choice is not None: m[:min(len(self.choice.options), self.max_options)] = 1
        return m

    def _empty_obs(self):
        return {'tokens': np.zeros(self.max_tokens, np.int32), 'seg': np.zeros(self.max_tokens, np.int8), 'hyp_of': np.full(self.max_tokens, -1, np.int16), 'n_hyps': np.int32(0), 'goals_left': np.int32(0), 'budget': np.int32(self._remaining()),
                'overlap': np.zeros((self.n_globals, 2), np.float32), 'choice': np.array([-1, -1, 0], np.int32), 'hole': np.zeros(F.MAX_HOLE_TOKENS, np.int32),
                'rows': np.zeros((self.max_options, 4), np.int64), 'n_options': np.int32(0), 'mask': np.zeros(self.max_options, np.int8)}

    def _obs(self):
        gf = self.gf; ch = self.choice
        L = min(len(gf.tokens), self.max_tokens)
        tokens = np.zeros(self.max_tokens, np.int32); tokens[:L] = gf.tokens[:L]
        seg = np.zeros(self.max_tokens, np.int8); seg[:L] = gf.seg[:L]
        hyp_of = np.full(self.max_tokens, -1, np.int16)
        for h, (a, b) in enumerate(gf.spans):
            if a < L: hyp_of[a:min(b, L)] = h
        ov = np.zeros((self.n_globals, 2), np.float32); k = min(len(self.ov), self.n_globals); ov[:k] = self.ov[:k]
        rows = np.zeros((self.max_options, 4), np.int64); r = F.rows_array(ch.options)[:self.max_options]; rows[:len(r)] = r
        hole = np.zeros(F.MAX_HOLE_TOKENS, np.int32); h_ = F.hole_tokens(self.ctx.ser, ch.hole)[:F.MAX_HOLE_TOKENS]; hole[:len(h_)] = h_
        m = np.zeros(self.max_options, np.int8); m[:len(r)] = 1
        return {'tokens': tokens, 'seg': seg, 'hyp_of': hyp_of, 'n_hyps': np.int32(gf.n_hyps), 'goals_left': np.int32(self.ep.n_goals), 'budget': np.int32(self._remaining()), 'overlap': ov,
                'choice': np.array([F.CHOICE_ID[ch.name], -1 if ch.kind is None else int(ch.kind), min(ch.pos, 31)], np.int32), 'hole': hole, 'rows': rows, 'n_options': np.int32(len(r)), 'mask': m}


def library_lemmas(bank, ddir, split=None, max_len=None):
    """{name: (statement, premises)} for the library lemmas of a dataset directory (library.json), the ones whose statement is not already a bank problem; `split`: only those of the given split; `max_len`:
    only those whose recorded proof has at most that many actions"""
    import json
    lib = json.load(open(os.path.join(ddir, 'library.json')))
    out = {}
    for l in lib['lemmas']:
        if l['bank_duplicate'] or (split is not None and l['split'] != split) or (max_len is not None and l['length'] > max_len): continue
        gid = l['gid']; out[l['name']] = (bank.w.T(bank.k.global_info(gid)[1]), gid)
    return out


def make(split=None, names=None, **kw):
    """a BankEnv over the whole bank (`bank_all.load()`: the world is built, about 30 s), restricted to a split of `bank_split` ('train', 'dev', 'test') or to `names`; the other arguments are BankEnv's"""
    import bank_all, bank_split as BS
    bk = bank_all.load()
    if names is None and split is not None: names = [n for n in bk.placed if BS.split_of(n) == split]
    return BankEnv(bk, names=names, **kw)


try: gym.register(id='ITPBank-v0', entry_point='bank_env:make')                   # gym.make('ITPBank-v0', split='train')
except Exception: pass
