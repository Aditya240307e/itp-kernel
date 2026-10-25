"""ml_pol.py -- the policy at the level of one GOAL: assemble an action for a goal by sampling the network's decisions, with their log-probabilities.

    runner = PolicyRunner(eng, net, ctx)
    s = runner.sample(g, menu, rng, temperature=1.0)       # g: a gym_env.GoalData, menu = eng.menu(g)
    s.action            # the assembled gym_env.Action, or None when the decisions led nowhere (s.failed says why)
    s.decisions         # [(Choice, index)]  -- what ml_data.make_step turns into a Step
    s.logps             # the log-probability of each decision under the policy at temperature 1 (after the mask of `kmask`)

Everything that works on goals and not on a Gymnasium episode (the search of ml_mcts, the manager-worker agent of ml_hier, the evaluation by sampling) goes through this class, so that the policy is sampled in
exactly one place.  The class holds one decoder per goal (the transformer runs once per goal, however many actions are sampled in it) and the table of embeddings of the globals (once per set of weights:
`refresh()` after the weights changed).

`kmask` is a set of action KINDS as a bit set (bit k: gym_env.Kind k is allowed; -1 all): at the decision of the kind the others get no probability, the rest is renormalised -- the same mask that the network
applies in training (ml_model: Step.kmask), so that a policy gradient is taken of the distribution that was sampled from.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import collections, time
import numpy as np
import torch
import gym_policy as P
import ml_features as F, ml_model as M
from gym_env import Kind


def kind_mask(*kinds):
    """the bit set of the given action kinds (gym_env.Kind members)"""
    m = 0
    for k in kinds: m |= 1 << int(k)
    return m


ALL_KINDS = -1
NO_CLAIM = ~kind_mask(Kind.CLAIM) & ((1 << 62) - 1)           # every kind but CLAIM (what the worker of the manager-worker agent may choose)
ONLY_CLAIM = kind_mask(Kind.CLAIM)


class Sampled:
    """the result of sampling one action: see the module documentation"""
    __slots__ = ('action', 'decisions', 'logps', 'failed', 'gf', 'ov')
    def __init__(self, action, decisions, logps, failed, gf, ov):
        self.action = action; self.decisions = decisions; self.logps = logps; self.failed = failed; self.gf = gf; self.ov = ov
    @property
    def logp(self): return float(sum(self.logps))


def _allowed(choice, kmask):
    """a bool vector over the options of a `kind` decision: may the policy choose the option's kind"""
    return torch.tensor([bool((kmask >> int(r[1])) & 1) if r[0] == P.OT_KIND else True for r in choice.options])


class PolicyRunner:
    def __init__(self, eng, net, ctx, rare=True, cache=512):
        self.eng = eng; self.net = net; self.ctx = ctx; self.rare = rare; self.cache = cache
        self._gtab = None; self._dec = collections.OrderedDict()
        self.t_policy = 0.0; self.n_decisions = 0; self.n_encodes = 0

    def refresh(self):
        """the weights of the network changed: drop what was computed from them"""
        self._gtab = None; self._dec.clear()

    def gtab(self):
        if self._gtab is None:
            with torch.no_grad(): self._gtab = self.net.global_table()
        return self._gtab

    def decoder(self, g, gf=None, ov=None):
        """the (cached) decoder of a goal"""
        key = getattr(g, 'key', None)
        dec = self._dec.get(key) if key is not None else None
        if dec is None:
            dec = M.Decoder(self.net, self.ctx, g, self.gtab(), gf=gf, ov=ov); self.n_encodes += 1
            if key is not None:
                self._dec[key] = dec
                if len(self._dec) > self.cache: self._dec.popitem(last=False)
        else:
            self._dec.move_to_end(key)
        return dec

    def sample(self, g, menu, rng, temperature=1.0, greedy=False, kmask=ALL_KINDS, gf=None, ov=None):
        """assemble one action for goal `g` (see the module documentation).  `rng`: a numpy Generator.  At a temperature other than 1 the decisions are drawn from softmax(logp / T) while `logps` stay the
        log-probabilities at temperature 1 (what a policy gradient or the search's cost needs)."""
        net = self.net
        dec = self.decoder(g, gf, ov)
        asm = P.Assembler(self.eng, g, menu, rare=self.rare)
        ch = asm.start(); carry = dec.root(); decisions = []; logps = []
        while ch is not None:
            t0 = time.time()
            logp, oe, hs = dec.scores(carry, ch)
            if ch.name == 'kind' and kmask != ALL_KINDS:
                ok = _allowed(ch, kmask)
                if not bool(ok.any()): asm.failed = 'no allowed kind'; ch = None; break
                logp = torch.log_softmax(logp.masked_fill(~ok, -1e9), dim=-1)
            self.t_policy += time.time() - t0; self.n_decisions += 1
            if len(logp) == 1: i = 0
            elif greedy: i = int(torch.argmax(logp))
            else:
                pr = torch.softmax(logp / temperature, dim=-1).numpy().astype(np.float64); pr /= pr.sum()
                i = int(rng.choice(len(pr), p=pr))
            decisions.append((ch, i)); logps.append(float(logp[i]))
            carry = dec.child(carry, ch, i, oe, hs)
            ch = asm.choose(i)
        return Sampled(asm.action, decisions, logps, asm.failed, dec.goal, dec.ov)

    def action_logp(self, g, menu, action, kmask=ALL_KINDS):
        """(the log-probability that the policy assembles exactly `action` for the goal -- temperature 1, masked --, the decisions that say it); (-inf, []) when the assembler cannot say the action
        (gym_policy.decompose raises)"""
        dec = self.decoder(g)
        try: ds = P.decompose(self.eng, g, menu, action)
        except Exception: return float('-inf'), []
        carry = dec.root(); total = 0.0
        for ch, i in ds:
            logp, oe, hs = dec.scores(carry, ch)
            if ch.name == 'kind' and kmask != ALL_KINDS:
                logp = torch.log_softmax(logp.masked_fill(~_allowed(ch, kmask), -1e9), dim=-1)
            total += float(logp[i]); carry = dec.child(carry, ch, i, oe, hs)
        return total, ds
