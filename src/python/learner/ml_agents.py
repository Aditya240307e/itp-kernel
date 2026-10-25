"""ml_agents.py -- the provers the experiments compare, behind one interface.

    solver = make_solver('mcts:value=critic,c_puct=2', eng, net=net, ctx=ctx)         # 'name:key=value,...' (the defaults are in DEFAULTS)
    row = solver.solve(stmt, premises, probes=2000, seed=0, name='imo1959_p1', regime='prefix', time_limit=120)
    row['solved']         # a proof was found and the KERNEL checked it
    row['probes']         # the probes spent when it was found (when not found: the probes spent)
    row['proof']          # the portable actions of the proof (when solved)

A PROBE is one try of an action on a goal by the engine, whoever asks (ml_protocol).  Every solver is measured in probes, and every proof it reports was replayed on a live proof and checked by the
kernel; nothing a solver says is trusted.

    blind     iterative deepening on the engine's own action costs (gym_search.CostSearcher): no learning, the baseline every learned prover is compared with
    uniform   episodes of actions chosen uniformly at random from the options the engine offers (ml_act.uniform_chooser): the weakest baseline
    sample    the network samples its decisions, episode after episode (temperature `T`), until the probes are spent: the "pass@k" agent
    greedy    the network's most probable decision at every question: one episode
    ida       IDA* on the negative log-probabilities of the policy (ml_search.PolicySearcher): the policy decides which actions are tried and in which order
    mcts      PUCT over proof states (ml_mcts.MCTSSearcher), by rollouts or by the critic's value
    mw        the manager-worker prover (ml_hier.HierAgent)

The deterministic searchers (blind, ida) are run once at the largest budget and the curve "solved within B probes" is read off their rows; the others are seeded.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import time
import numpy as np
import gym_env as G, gym_search as S
import ml_pol as POL, ml_act as ACT, ml_search as MS, ml_mcts as MM, ml_protocol as PR

SOLVERS = ('blind', 'uniform', 'sample', 'greedy', 'ida', 'mcts', 'mw')
DETERMINISTIC = ('blind', 'ida', 'greedy')
NEEDS_NET = ('sample', 'greedy', 'ida', 'mcts', 'mw')
DEFAULTS = {
    'blind': {'max_depth': 12, 'all_kinds': False},
    'uniform': {'max_actions': 40},
    'sample': {'T': 1.0, 'max_actions': 40},
    'greedy': {'max_actions': 40},
    'ida': {'step_cost': 0.2, 'topk': 24, 'cap': 9.0, 'T': 1.0},
    'mcts': {'c_puct': 1.5, 'widen_c': 2.0, 'widen_alpha': 0.5, 'rollout_depth': 12, 'value': 'rollout', 'value_scale': 8.0, 'T': 1.0, 'rollout_T': 1.0, 'topk': 24, 'cap': 9.0},
    'mw': {'worker': 'sample', 'worker_probes': 100, 'worker_actions': 24, 'max_claims': 3, 'max_actions': 40, 'gate': 'head', 'gate_p': 0.15, 'T': 1.0, 'claim_T': 1.0},
}


def _value(text):
    for conv in (int, float):
        try: return conv(text)
        except ValueError: pass
    return {'true': True, 'false': False}.get(text.lower(), text)


def parse_spec(spec):
    """'mcts:value=critic,c_puct=2' -> ('mcts', {'value': 'critic', 'c_puct': 2}); the options not given keep their defaults (DEFAULTS); an unknown solver or option is a ValueError"""
    name, _, rest = spec.partition(':')
    if name not in SOLVERS: raise ValueError('%r: not a solver (%s)' % (name, ', '.join(SOLVERS)))
    opts = dict(DEFAULTS[name])
    for item in filter(None, rest.split(',')):
        k, _, v = item.partition('=')
        if k not in opts: raise ValueError('%s: no option %r (it has %s)' % (name, k, ', '.join(opts)))
        opts[k] = _value(v)
    return name, opts


def canonical(spec):
    """the spec with every option written out, in a fixed order (a name for rows and files)"""
    name, opts = parse_spec(spec)
    base = DEFAULTS[name]
    changed = ','.join('%s=%s' % (k, opts[k]) for k in base if opts[k] != base[k])
    return name + (':' + changed if changed else '')


class Solver:
    def __init__(self, spec, eng, net=None, ctx=None):
        self.spec = canonical(spec); self.name, self.opts = parse_spec(spec); self.eng = eng; self.net = net; self.ctx = ctx
        if self.name in NEEDS_NET and (net is None or ctx is None): raise ValueError('the solver %r needs a network and a data context' % self.name)
        self.runner = POL.PolicyRunner(eng, net, ctx) if self.name in ('sample', 'greedy', 'mw') else None
        self._blind = None; self._agent = None
        if net is not None: net.eval()

    @property
    def deterministic(self): return self.name in DETERMINISTIC

    def refresh(self):
        """the world grew or the weights changed: drop what was computed from them"""
        if self.runner is not None: self.runner.refresh()
        self._blind = None; self._agent = None

    def solve(self, stmt, premises, probes, seed=0, name='', regime='prefix', time_limit=None, harvest=None):
        """attack `stmt` under `premises` with `probes` probes: a row (dict) -- solved, probes, decisions, seconds, steps, proof.  `harvest`: an ml_her.Harvest that collects the theorems the attempt proved on
        the way (not for `blind`)"""
        t0 = time.time()
        try: row = getattr(self, '_solve_' + self.name)(stmt, premises, probes, seed, name, regime, time_limit, harvest)
        except (MemoryError, KeyboardInterrupt): raise
        except Exception as e:
            row = {'solved': False, 'probes': 0, 'decisions': 0, 'seconds': round(time.time() - t0, 3), 'steps': None, 'error': '%s: %s' % (type(e).__name__, str(e)[:160])}
        row['name'] = name; row['regime'] = regime; row['solver'] = self.spec
        return row

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the solvers
    def _rng(self, kind, seed, name, regime): return np.random.default_rng(PR.stable_seed(kind, seed, name, regime))

    def _solve_blind(self, stmt, premises, probes, seed, name, regime, time_limit, harvest=None):
        o = self.opts
        if self._blind is None or self._blind[1] != self.eng.k.num_globals():
            self._blind = (G.Engine(self.eng.w, kinds=set(G.Kind), strict=False) if o['all_kinds'] else G.Engine(self.eng.w), self.eng.k.num_globals())
        eng = self._blind[0]; eng.premises = premises; t0 = time.time()
        term, tree, s = S.prove_statement(eng, stmt, max_depth=int(o['max_depth']), max_probes=probes, time_limit=time_limit, cost=True)
        return self._search_row(eng, stmt, premises, term, tree, s, t0, portable=True)

    def _search_row(self, eng, stmt, premises, term, tree, s, t0, portable=True):
        row = {'solved': term is not None, 'probes': s.probes if s is not None else 0, 'seconds': round(time.time() - t0, 3), 'steps': (tree.size() if tree is not None else (0 if term is not None else None))}
        dec = 0
        if s is not None and hasattr(s, 'runner'): dec = s.runner.n_decisions + s.expansions
        row['decisions'] = dec
        if term is not None and tree is not None and portable:
            try: row['proof'] = MS.tree_portable(eng, stmt, premises, tree)
            except (G.ActionError, ACT.REFUSALS): row['proof_error'] = 'the tree did not replay'
        return row

    def _solve_uniform(self, stmt, premises, probes, seed, name, regime, time_limit, harvest=None):
        rng = self._rng('uniform', seed, name, regime)
        return ACT.sample_prove(self.eng, stmt, premises, ACT.uniform_chooser(self.eng, rng), probes, max_actions=int(self.opts['max_actions']), time_limit=time_limit, name=name,
                                harvest=None if harvest is None else harvest.episode).row()

    def _solve_sample(self, stmt, premises, probes, seed, name, regime, time_limit, harvest=None):
        rng = self._rng('sample', seed, name, regime); o = self.opts
        return ACT.sample_prove(self.eng, stmt, premises, ACT.policy_chooser(self.runner, rng, float(o['T'])), probes, max_actions=int(o['max_actions']), time_limit=time_limit, name=name,
                                harvest=None if harvest is None else harvest.episode).row()

    def _solve_greedy(self, stmt, premises, probes, seed, name, regime, time_limit, harvest=None):
        rng = self._rng('greedy', seed, name, regime)
        return ACT.sample_prove(self.eng, stmt, premises, ACT.policy_chooser(self.runner, rng, greedy=True), probes, max_actions=int(self.opts['max_actions']), time_limit=time_limit, name=name, max_episodes=1,
                                harvest=None if harvest is None else harvest.episode).row()

    def _solve_ida(self, stmt, premises, probes, seed, name, regime, time_limit, harvest=None):
        o = self.opts; t0 = time.time()
        term, tree, s = MS.prove_statement(self.eng, stmt, premises, self.net, self.ctx, max_probes=probes, time_limit=time_limit, step_cost=float(o['step_cost']), topk=int(o['topk']), cap=float(o['cap']),
                                           temperature=float(o['T']))
        row = self._search_row(self.eng, stmt, premises, term, tree, s, t0)
        if harvest is not None and s is not None: harvest.searcher(s, tree)
        return row

    def _solve_mcts(self, stmt, premises, probes, seed, name, regime, time_limit, harvest=None):
        o = self.opts; t0 = time.time()
        term, tree, s = MM.prove_statement(self.eng, stmt, premises, self.net, self.ctx, max_probes=probes, time_limit=time_limit, c_puct=float(o['c_puct']), widen_c=float(o['widen_c']),
                                           widen_alpha=float(o['widen_alpha']), rollout_depth=int(o['rollout_depth']), rollout_temperature=float(o['rollout_T']), value=str(o['value']),
                                           value_scale=float(o['value_scale']), seed=PR.stable_seed('mcts', seed, name, regime), topk=int(o['topk']), cap=float(o['cap']), temperature=float(o['T']))
        row = self._search_row(self.eng, stmt, premises, term, tree, s, t0)
        if s is not None: row.update({'simulations': s.simulations, 'rollouts': s.rollouts})
        if harvest is not None and s is not None: harvest.searcher(s, tree)
        return row

    def agent(self):
        if self._agent is None:
            import ml_hier as H
            o = self.opts
            self._agent = H.HierAgent(self.eng, self.net, self.ctx, worker=str(o['worker']), worker_probes=int(o['worker_probes']), worker_actions=int(o['worker_actions']), max_claims=int(o['max_claims']),
                                      max_actions=int(o['max_actions']), gate=str(o['gate']), gate_p=float(o['gate_p']), temperature=float(o['T']), claim_temperature=float(o['claim_T']), runner=self.runner)
        return self._agent

    def _solve_mw(self, stmt, premises, probes, seed, name, regime, time_limit, harvest=None):
        rng = self._rng('mw', seed, name, regime)
        res = self.agent().prove(stmt, premises, probes, rng, time_limit=time_limit, name=name, want_claims=harvest is not None, harvest=None if harvest is None else harvest.episode)
        if harvest is not None: harvest.claims(res.claims)
        return res.row()


def make_solver(spec, eng, net=None, ctx=None):
    return Solver(spec, eng, net, ctx)
