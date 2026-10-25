"""ml_protocol.py -- how a prover is measured: kernel-checked proofs within a budget of PROBES.

The unit of cost is the probe, one try of an action on a goal by the engine (a search counts every probe it makes; an agent that acts counts every action it takes and every action the engine refuses).  A
result is a ROW per (problem, regime):

    name, regime, level, split, method, seed,
    solved     a proof was found and the kernel checked it
    probes     the probes spent when it was found (when not found: the budget that was spent)
    seconds    wall-clock time          steps   the actions of the proof found (None when unsolved)

A method that is deterministic (a search) is run once with the largest budget and its curve "solved within B probes" is read off the rows (a problem is solved within B iff `solved and probes <= B`).  A method
that samples (a policy acting) is run with a seed: independent episodes on the problem until one finishes the proof or the budget is spent, every episode's probes added up, so the same curve can be read off
a run with the largest budget.  `ml_report.py` and `ml_aggregate.py` turn rows into tables.

    row = sample_row(env, net, ctx, gtab, 'imo1959_p1', 'prefix', probe_budget=4000, max_actions=40, seed=0)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import time, hashlib
import numpy as np
import torch
import ml_model as M

BUDGETS = (125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def stable_seed(*parts):
    """a 63-bit seed from any strings and numbers, the same in every process (Python's hash() is not)"""
    h = hashlib.sha256('/'.join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], 'little') >> 1


def solved_within(rows, budget):
    return sum(1 for r in rows if r['solved'] and r['probes'] <= budget)


def curve(rows, budgets=BUDGETS):
    """{budget: number of rows solved within it}"""
    return {b: solved_within(rows, b) for b in budgets}


def play(env, net, ctx, gtab, rng, options, temperature=1.0, greedy=False):
    """one episode of the policy in a BankEnv: (info, logps) -- `logps` the log-probability (temperature 1) of every decision in order.  The episode's record is in `env` (trajectory, meta, solved, term)."""
    obs, info = env.reset(options=options)
    dec = None; last_gf = None; carry = None; logps = []
    while True:
        ch = info.get('choice')
        if ch is None: break
        gf = info['goal_feat']
        if gf is not last_gf:
            dec = M.Decoder(net, ctx, None, gtab, gf=gf, ov=info['overlap']); last_gf = gf
        if info['in_action'] == 0: carry = dec.root()
        logp, oe, hs = dec.scores(carry, ch)
        if len(logp) == 1: i = 0
        elif greedy: i = int(torch.argmax(logp))
        else:
            pr = torch.softmax(logp / temperature, dim=-1).numpy().astype(np.float64); pr /= pr.sum()
            i = int(rng.choice(len(pr), p=pr))
        logps.append(float(logp[i]))
        carry = dec.child(carry, ch, i, oe, hs)
        obs, r, term, trunc, info = env.step(i)
        if term or trunc: break
    return info, logps


def sample_row(env, net, ctx, gtab, name, regime, probe_budget, max_actions, seed, temperature=1.0, extra=None):
    """the row of a policy that SAMPLES: episodes on the problem (each at most `max_actions` probes, and at most what is left of `probe_budget`) until one is kernel-checked"""
    rng = np.random.default_rng(stable_seed('sample', seed, name, regime))
    spent = 0; episodes = 0; t0 = time.time(); steps = None; solved = False
    while spent < probe_budget:
        b = int(min(max_actions, probe_budget - spent))
        info, _ = play(env, net, ctx, gtab, rng, {'problem': name, 'regime': regime, 'budget': b}, temperature)
        spent += max(1, env.actions + env.fails); episodes += 1
        if env.solved: solved = True; steps = env.actions; break
    row = {'name': name, 'regime': regime, 'solved': solved, 'probes': int(spent), 'seconds': round(time.time() - t0, 2), 'steps': steps, 'episodes': episodes}
    if extra: row.update(extra)
    return row


def annotate(row, bk, method, seed=None):
    """level, split, method, whether the problem has a reference proof (`proved`), its origin, source and family on a row"""
    import bank_split as BS
    pl = bk.placed[row['name']]; pr = pl.problem
    row['level'] = pr.level; row['split'] = BS.split_of(row['name']); row['method'] = method
    row['proved'] = bool(pl.proved); row['origin'] = pr.origin; row['source'] = pr.source; row['family'] = pr.family or row['name']          # (a statement-only problem has no reference proof: nobody has proved it)
    if seed is not None: row['seed'] = seed
    return row


def regimes_of(pl, regimes=('prefix', 'strict')):
    """the (regime, premises) pairs to run for a placed problem: strict only when it is a different task (the problem has a self-contained proof) -- the rule of ml_eval.evaluate"""
    out = []; seen = {}
    for regime in regimes:
        prem = pl.premises(regime)
        if regime == 'strict' and prem != pl.premises('prefix') and not pl.problem.strict: continue
        if prem in seen: out.append((regime, seen[prem])); continue
        seen[prem] = regime; out.append((regime, None))
    return out
