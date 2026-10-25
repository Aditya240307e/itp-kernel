"""ml_act.py -- an agent that ACTS: episodes of a policy on a closed statement, every probe counted.

    runner = ml_pol.PolicyRunner(eng, net, ctx)
    out = sample_prove(eng, stmt, premises, policy_chooser(runner, rng), probes=2000, max_actions=40)
    out.solved, out.probes, out.steps            # `steps`: the portable actions of the proof, which the kernel has checked

An EPISODE is one proof attempt: the agent chooses an action for the first open goal, the engine probes it (one counted probe, whether the engine takes the action or refuses it), and so on until the
proof is finished (the kernel checks it) or the episode has used `max_actions` probes.  `sample_prove` plays episode after episode until a proof is finished, the probes run out or the time limit is
reached; the probes of all the episodes are added up, so a curve "solved within B probes" can be read off a run with the largest budget (ml_protocol).

A `chooser(g, menu) -> (action or None, number of decisions)` is the agent's mind: `policy_chooser` samples the network (the decisions of the assembler, ml_pol), `uniform_chooser` answers every question
of the assembler at random (the weakest baseline).  The same loop is the top level of the manager-worker agent (ml_hier).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import time
import numpy as np
import itp_env_lnl as E
import gym_env as G, gym_acts as ACTS, gym_policy as P
from gym_env import ActionError
from author import AuthorError
import ml_pol as POL

REFUSALS = (ActionError, AuthorError, E.ITPError, RuntimeError)          # what the engine says when it does not take an action


class Outcome:
    """what an attempt came to: `solved` (a proof was finished and the kernel checked it), `probes` (spent; the budget that was spent when unsolved), `decisions` (questions put to the policy), `seconds`, `steps`
    (the portable actions of the proof, or None when unsolved or when an action has no portable form), `episodes`, `actions` (taken), `refused`"""
    __slots__ = ('solved', 'probes', 'decisions', 'seconds', 'steps', 'episodes', 'actions', 'refused', 'error')
    def __init__(self, solved=False, probes=0, decisions=0, seconds=0.0, steps=None, episodes=0, actions=0, refused=0, error=None):
        self.solved = solved; self.probes = probes; self.decisions = decisions; self.seconds = seconds; self.steps = steps; self.episodes = episodes; self.actions = actions; self.refused = refused
        self.error = error
    def row(self):
        return {'solved': bool(self.solved), 'probes': int(self.probes), 'decisions': int(self.decisions), 'seconds': round(self.seconds, 3), 'steps': None if self.steps is None else len(self.steps),
                'episodes': self.episodes, 'actions': self.actions, 'refused': self.refused, **({'error': self.error} if self.error else {}), **({'proof': self.steps} if self.steps is not None else {})}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ choosers
def policy_chooser(runner, rng, temperature=1.0, greedy=False, kmask=POL.ALL_KINDS):
    """the network's choice for a goal: samples the decisions of the assembler (ml_pol.PolicyRunner.sample)"""
    def choose(g, menu):
        smp = runner.sample(g, menu, rng, temperature, greedy=greedy, kmask=kmask)
        return smp.action, len(smp.decisions)
    return choose


def uniform_chooser(eng, rng):
    """every question of the assembler answered by a uniformly random option"""
    def choose(g, menu):
        asm = P.Assembler(eng, g, menu, rare=True); ch = asm.start(); n = 0
        while ch is not None:
            ch = asm.choose(int(rng.integers(len(ch.options)))); n += 1
        return asm.action, n
    return choose


def portable_of(eng, menu, action):
    """the portable form of an action in the menu of its goal (None when it has none)"""
    try: return ACTS.to_portable(eng, menu, action)
    except Exception: return None


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the loop
def play_episode(eng, ep, choose, cap, deadline=None):
    """one episode on a started `Episode`: at most `cap` probes.  Returns (probes spent, decisions, portable steps taken -- None entries for actions without a portable form --, actions taken, refused)"""
    spent = decisions = taken = refused = 0; portable = []
    while not ep.done and spent < cap:
        if deadline is not None and time.time() > deadline: break
        g = ep.goal(); menu = ep.menu()
        action, nd = choose(g, menu); decisions += nd; spent += 1
        if action is None: refused += 1; continue
        d = portable_of(eng, menu, action)
        try: ep.step(action)
        except REFUSALS: refused += 1; continue
        portable.append(d); taken += 1
    return spent, decisions, portable, taken, refused


def sample_prove(eng, stmt, premises, choose, probes, max_actions=40, time_limit=None, name='', max_episodes=None, harvest=None):
    """play episodes on the closed statement `stmt` (with `premises`) until a proof is finished and kernel-checked, `probes` are spent, the time is up or `max_episodes` episodes were played: an `Outcome`.
    `harvest(ep)`: called with every episode when it ends (finished or not; the episodes are TRACKED then, `ep.tree` is the proof tree) -- what hindsight learning takes the proved sub-goals from (ml_her)"""
    t0 = time.time(); deadline = None if time_limit is None else t0 + time_limit
    out = Outcome()
    while out.probes < probes and (deadline is None or time.time() < deadline) and (max_episodes is None or out.episodes < max_episodes):
        ep = G.Episode(eng, stmt, name, premises=premises, track=harvest is not None)
        if ep.done:                                                                   # (the engine settles the statement by itself: nothing to do)
            ep.finish(); out.solved = True; out.steps = []; break
        out.episodes += 1
        spent, nd, portable, taken, refused = play_episode(eng, ep, choose, int(min(max_actions, probes - out.probes)), deadline)
        out.probes += spent; out.decisions += nd; out.actions += taken; out.refused += refused
        if harvest is not None: harvest(ep)
        if ep.done:
            try: ep.finish()
            except Exception as e: out.error = '%s: %s' % (type(e).__name__, str(e)[:120]); continue             # (the kernel refused: not a proof)
            out.solved = True; out.steps = None if any(d is None for d in portable) else portable
            break
    out.seconds = time.time() - t0
    return out
