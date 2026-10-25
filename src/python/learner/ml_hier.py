"""ml_hier.py -- the manager-worker prover.

A proof is a sequence of actions; a CLAIM action is a different kind of step from the others: it says what to prove next (`have h : statement`), where the others only transform the goal.  The agent here
splits the two jobs.  One network plays both; what differs is the set of action kinds each may choose (ml_pol.kind_mask) and the head that decides who acts.

  MANAGER   at a goal, the manager head (`PolicyNet.mgr`: P(the next action is a CLAIM) from the goal, the fraction of the episode's probes left and the number of open goals) decides whether to propose
            a claim; if so the policy is sampled with ONLY the kind CLAIM allowed and says the statement of the claim (the assembler's decisions, token by token).
  WORKER    a claim is proved IN ISOLATION: the claim goal (its hypotheses and the claim) becomes a closed statement `forall hyps, claim` (gym_tree.closed_statement) and the worker -- the same policy with
            CLAIM forbidden, sampling episodes (`worker='sample'`) or searching (`worker='ida'`, ml_search.PolicySearcher) -- has `worker_probes` probes to prove it.  A claim that is not proved is dropped and
            the goal is unchanged; the manager does not propose it again at that goal.
  TRANSFER  a proved claim is added to the real proof by replaying, in a fresh episode, the actions taken so far, the CLAIM and the worker's actions (portable form): the engine probes every one of them
            again and the kernel checks the finished proof; nothing the worker or the manager says is trusted.
  OTHER     when the manager does not claim, the worker (CLAIM forbidden) chooses the action as in a flat agent.

All the probes of all the parts count against one budget (the claim's probe, the worker's attempts including those on claims that were not proved, the actions of the top level).  The episodes of the top
level restart after `max_actions` probes, as in ml_act.sample_prove.

TRAINING DATA.  `mw_steps(steps)` gives the steps of a proof the masks and labels this agent samples under: a CLAIM step is a manager step (`kmask` = only CLAIM, `mgr` = 1), every other step a worker step
(`kmask` = no CLAIM, `mgr` = 0), so that the likelihood a learner maximises is the probability the agent draws from.  The claims the worker proved -- theorems with kernel-checked proofs, whether or not the
episode they were proposed in was finished -- come out as hindsight examples (`HierResult.claims`).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import copy, math, time
import numpy as np
import gym_env as G, gym_acts as ACTS, gym_tree as GT, gym_premises as PM
from gym_env import ActionError
import ml_pol as POL, ml_act as ACT, ml_search as MS


def _sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class HierResult:
    """the outcome of one run of the agent: the fields of `ml_act.Outcome` and `claims`, a list of dicts (one per claim proposed and probed: whether the worker proved it, with what probes, the closed statement's
    code and the worker's portable proof when it did), plus counters"""
    def __init__(self):
        self.out = ACT.Outcome(); self.claims = []; self.proposed = 0; self.proved = 0; self.dropped = 0; self.transfer_failed = 0; self.worker_probes = 0
    def row(self):
        r = self.out.row()
        r.update({'claims_proposed': self.proposed, 'claims_proved': self.proved, 'claims_dropped': self.dropped, 'transfer_failed': self.transfer_failed, 'worker_probes': self.worker_probes})
        return r


def replay_into(eng, ep, steps):
    """take the portable `steps` in an Episode, one after the other (each is probed by the engine: ActionError if one does not apply)"""
    for d in steps:
        menu = ep.menu()
        act = ACTS.from_portable(eng, menu if d.get('kind') in ACTS.MENU_KINDS else ACTS.NO_MENU, d)
        ep.step(act)


class HierAgent:
    def __init__(self, eng, net, ctx, worker='sample', worker_probes=100, worker_actions=24, max_claims=3, max_actions=40, gate='head', gate_p=0.15, temperature=1.0, claim_temperature=1.0,
                 runner=None, search_kw=None):
        self.eng = eng; self.net = net; self.ctx = ctx
        if worker not in ('sample', 'ida'): raise ValueError("worker: 'sample' or 'ida'")
        if gate not in ('head', 'fixed', 'never'): raise ValueError("gate: 'head' (the manager head), 'fixed' (probability gate_p) or 'never' (no claims: the flat agent)")
        self.worker = worker; self.worker_probes = int(worker_probes); self.worker_actions = int(worker_actions); self.max_claims = int(max_claims); self.max_actions = int(max_actions)
        self.gate = gate; self.gate_p = float(gate_p); self.temperature = float(temperature); self.claim_temperature = float(claim_temperature)
        self.runner = runner or POL.PolicyRunner(eng, net, ctx); self.search_kw = dict(search_kw or {})

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the worker
    def prove_claim(self, goal, premises, probes, rng, deadline):
        """prove a goal (a GoalData: the claim in its context) in isolation: (solved, probes spent, decisions, the worker's portable steps or None, the closed statement)"""
        closed = GT.closed_statement(self.eng.w, goal.hyp_types, goal.target)
        if self.worker == 'sample':
            out = ACT.sample_prove(self.eng, closed, premises, ACT.policy_chooser(self.runner, rng, self.temperature, kmask=POL.NO_CLAIM), probes, max_actions=self.worker_actions,
                                   time_limit=None if deadline is None else max(0.0, deadline - time.time()))
            return out.solved and out.steps is not None, out.probes, out.decisions, out.steps, closed
        tl = None if deadline is None else max(0.0, deadline - time.time())
        searcher = MS.PolicySearcher(self.eng, self.net, self.ctx, max_probes=probes, time_limit=tl, kmask=POL.NO_CLAIM, **self.search_kw)
        self.eng.premises = premises
        try:
            p = self.eng.start(closed)
            if p.done: return True, 0, 0, [], closed
            tree = searcher.prove(self.eng.goal_of(p, 0))
            steps = None
            if tree is not None:
                try: steps = MS.tree_portable(self.eng, closed, premises, tree)
                except (G.ActionError, ACT.REFUSALS): steps = None
        except ACT.REFUSALS:
            return False, searcher.probes, searcher.runner.n_decisions + searcher.expansions, None, closed
        return steps is not None, searcher.probes, searcher.runner.n_decisions + searcher.expansions, steps, closed

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the top level
    def claim_now(self, dec, rng, left, goals):
        """does the manager propose a claim at this goal?"""
        if self.gate == 'never': return False
        if self.gate == 'fixed': return bool(rng.random() < self.gate_p)
        return bool(rng.random() < _sigmoid(dec.manager_logit(left, goals)))

    def prove(self, stmt, premises, probes, rng, time_limit=None, name='', want_claims=True, harvest=None):
        """attack the closed statement `stmt` under `premises` with `probes` probes: a HierResult.  `harvest(ep)`: called with every top-level episode when it ends (they are tracked then: ml_act.sample_prove)"""
        eng = self.eng; res = HierResult(); out = res.out; t0 = time.time(); deadline = None if time_limit is None else t0 + time_limit
        tried = set(); track = harvest is not None
        while out.probes < probes and (deadline is None or time.time() < deadline):
            ep = G.Episode(eng, stmt, name, premises=premises, track=track)
            if ep.done: ep.finish(); out.solved = True; out.steps = []; break
            out.episodes += 1; portable = []; ep_probes = 0; claims_used = 0; cap = int(min(self.max_actions, probes - out.probes))
            while not ep.done and ep_probes < cap and out.probes + ep_probes < probes:
                if deadline is not None and time.time() > deadline: break
                g = ep.goal(); menu = ep.menu(); dec = self.runner.decoder(g)
                left = (cap - ep_probes) / max(cap, 1)
                if claims_used < self.max_claims and self.claim_now(dec, rng, left, ep.n_goals):
                    # --- the manager proposes a claim
                    smp = self.runner.sample(g, menu, rng, self.claim_temperature, kmask=POL.ONLY_CLAIM)
                    out.decisions += len(smp.decisions); ep_probes += 1
                    if smp.action is None: continue
                    key = (g.key, menu.signature(smp.action))
                    if key in tried: continue
                    tried.add(key); res.proposed += 1
                    kids = eng.probe(g, smp.action, menu)
                    if kids is None or len(kids) not in (1, 2): res.dropped += 1; continue
                    d_claim = ACT.portable_of(eng, menu, smp.action)
                    if d_claim is None: res.dropped += 1; continue
                    if len(kids) == 1: ok, wprobes, wdec, wsteps, closed = True, 0, 0, [], None          # (the engine settled the claim by itself)
                    else:
                        budget = int(min(self.worker_probes, probes - out.probes - ep_probes))
                        if budget < 1: break
                        ok, wprobes, wdec, wsteps, closed = self.prove_claim(kids[0], premises, budget, rng, deadline)
                    out.probes += wprobes; out.decisions += wdec; res.worker_probes += wprobes
                    if want_claims and closed is not None:
                        res.claims.append(self._claim_record(closed, premises, ok, wprobes, wsteps, name))
                    if not ok: res.dropped += 1; continue
                    # --- transfer: replay what was taken so far, the claim and the worker's actions in a fresh episode
                    ep2 = G.Episode(eng, stmt, name, premises=premises, track=track)
                    try: replay_into(eng, ep2, portable + [d_claim] + list(wsteps))
                    except (ActionError, ACT.REFUSALS, KeyError, ValueError): res.transfer_failed += 1; continue
                    ep = ep2; portable = portable + [d_claim] + list(wsteps); claims_used += 1; res.proved += 1
                    continue
                # --- the worker acts
                smp = self.runner.sample(g, menu, rng, self.temperature, kmask=POL.NO_CLAIM if self.gate != 'never' else POL.ALL_KINDS)
                out.decisions += len(smp.decisions); ep_probes += 1
                if smp.action is None: out.refused += 1; continue
                d = ACT.portable_of(eng, menu, smp.action)
                try: ep.step(smp.action)
                except ACT.REFUSALS: out.refused += 1; continue
                portable.append(d); out.actions += 1
            out.probes += ep_probes
            if harvest is not None: harvest(ep)
            if ep.done:
                try: ep.finish()
                except Exception as e: out.error = '%s: %s' % (type(e).__name__, str(e)[:120]); continue
                out.solved = True; out.steps = None if any(d is None for d in portable) else portable
                break
        out.seconds = time.time() - t0
        return res

    def _claim_record(self, closed, premises, solved, probes, steps, name):
        """a claim the worker was asked to prove, as a hindsight example if it was proved (the dict of `gym_tree.Example.to_dict`, with the problem it came from)"""
        eng = self.eng; text = eng.w.show(closed)
        rec = {'kind': 'claim', 'statement': text, 'statement_digest': GT.digest(text), 'solved': bool(solved), 'probes': int(probes), 'problem': name}
        if solved:
            code = GT.statement_code(eng, closed, premises)
            rec.update({'statement_code': None if code is None else [list(t) for t in code], 'premises': PM.to_json(PM.of(premises)), 'steps': steps, 'length': len(steps),
                        'origin': {'kind': 'CLAIM', 'problem': name}, 'verified': True})
        return rec


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ training data
def mw_steps(steps, label_mgr=True):
    """the steps of a proof (ml_data.Step) with the masks and labels the manager-worker agent samples under: a CLAIM step has `kmask` = only CLAIM and `mgr` = 1, every other step `kmask` = no CLAIM and
    `mgr` = 0 (or -1 -- no label -- when `label_mgr` is False: the synthetic theorems, which have no claims, must not teach the manager that claims never happen).  New Step objects: the arguments are untouched."""
    from gym_env import Kind
    out = []
    for s in steps:
        t = copy.copy(s)
        is_claim = int(s.kind) == int(Kind.CLAIM)
        t.kmask = POL.ONLY_CLAIM if is_claim else POL.NO_CLAIM
        t.mgr = (1 if is_claim else 0) if label_mgr else -1
        out.append(t)
    return out
