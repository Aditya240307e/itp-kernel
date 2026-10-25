"""ml_goals.py -- where the training episodes of the reinforcement learners start.

    bank, subs = train_goals(bk, ddir)                          # the train problems [(name, reference length or None)] and the sub-theorem records of the train split
    goals = Goals(bk, bank, subs, synth=records, p_sub=0.25, p_synth=0.1)
    task = goals.sample(rng)                                     # {'name', 'source': 'bank' | 'sub' | 'synth', 'options': {...}}   -- plain data, so it can be sent to a worker process
    name, stmt, premises = resolve(bk, eng, lib, task)           # the statement of a task in a world, with the premises it is attacked under (widened by the lemma library, if there is one)

Nothing from the dev or test split is read here: the problems are those of the train split of the bank, the sub-theorems are those of train problems' reference proofs (the hindsight dataset) and the synthetic
theorems are the generator's train records.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, json, math
import numpy as np
import gym_tree as GT
import ml_train as T, ml_protocol as PR


class Goals:
    """where the training episodes start.  `bank`: [(name, reference length or None)] of train problems; `subs`: sub-theorem records (subproblems.json, train split); `synth`: synthetic records (code,
    steps).  `sample(rng)` -> a task dict {'name', 'options', 'source'} that a worker can run; reproducible from the rng."""
    def __init__(self, bk, bank, subs=(), synth=(), p_sub=0.25, p_synth=0.1, p_hard=0.1, max_ref_len=30, max_actions=40, budget_mult=3.0, synth_premises=None):
        self.bk = bk; self.p_sub = p_sub if subs else 0.0; self.p_synth = p_synth if synth else 0.0; self.max_actions = max_actions; self.budget_mult = budget_mult
        self.easy = [n for n, L in bank if L is not None and L <= max_ref_len]
        self.hard = [n for n, L in bank if L is None or L > max_ref_len]
        self.p_hard = p_hard if self.hard else 0.0
        if not self.easy and not self.hard: raise ValueError('no train problems')
        if not self.easy: self.easy, self.hard, self.p_hard = self.hard, [], 0.0
        self.subs = list(subs); self.synth = list(synth); self.synth_premises = synth_premises

    def _budget(self, length):
        return int(min(self.max_actions, max(8, math.ceil(self.budget_mult * length) + 4)))

    def sample(self, rng):
        u = rng.random()
        if u < self.p_sub:
            ex = self.subs[int(rng.integers(len(self.subs)))]
            return {'name': ex['id'], 'source': 'sub', 'options': {'goal': {'statement_code': ex['statement_code'], 'premises': ex['premises']}, 'name': ex['id'], 'budget': self._budget(ex['length'])}}
        if u < self.p_sub + self.p_synth:
            r = self.synth[int(rng.integers(len(self.synth)))]
            return {'name': r['id'], 'source': 'synth', 'options': {'goal': {'statement_code': r['code'], 'premises': self.synth_premises}, 'name': r['id'], 'budget': self._budget(r['n_steps'])}}
        pool = self.hard if (self.p_hard and rng.random() < self.p_hard) else self.easy
        name = pool[int(rng.integers(len(pool)))]
        return {'name': name, 'source': 'bank', 'options': {'problem': name, 'regime': regime_for(self.bk, name, rng)}}


def regime_for(bk, name, rng):
    pl = bk.placed[name]
    if not pl.problem.strict or pl.premises('strict') == pl.premises('prefix'): return 'prefix'
    return 'prefix' if rng.random() < 0.5 else 'strict'


def train_goals(bk, ddir, split='train', max_len=30):
    """(bank, subs): [(name, reference length or None)] of the problems of the split that are in the world, and the sub-theorem records of the split with proofs of at most `max_len` actions"""
    bank_json, _ = T.load_dataset(ddir)
    ref = {r['name']: (r['demonstration']['length'] if 'demonstration' in r else None) for r in bank_json['problems'] if r['split'] == split and r['name'] in bk.placed}
    bank = sorted(ref.items())
    subs = []
    path = os.path.join(ddir, 'subproblems.json')
    if os.path.exists(path):
        for ex in json.load(open(path))['examples']:
            if ex['split'] == split and ex['length'] <= max_len and ex.get('statement_code') is not None: subs.append(ex)
    return bank, subs


def resolve(bk, eng, lib, task):
    """(name, statement, premises) of a task (see the module documentation).  A statement given as a code is built in the world of `eng` under the task's premises."""
    o = task['options']
    widen = (lambda p: p) if lib is None else lib.premises
    if 'goal' in o:
        g = o['goal']; prem = widen(g.get('premises', None))
        return o.get('name') or task['name'], GT.statement_from_code(eng, g['statement_code'], prem), prem
    name = o['problem']; pl = bk.placed[name]
    return name, pl.stmt, widen(pl.premises(o.get('regime', 'prefix')))


def chunks(xs, n):
    """`xs` in at most `n` consecutive parts of nearly equal size"""
    xs = list(xs); k = max(1, math.ceil(len(xs) / max(n, 1)))
    return [xs[i:i + k] for i in range(0, len(xs), k)]


def dev_items(bk, names, regimes=('prefix',)):
    """[(name, regime)]: the tasks of the problems `names` (a regime that offers the same task as another is dropped: ml_protocol.regimes_of)"""
    return [(n, r) for n in names for r, same in PR.regimes_of(bk.placed[n], regimes) if same is None]
