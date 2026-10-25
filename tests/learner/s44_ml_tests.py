"""S44 gate: the learner's machinery -- features, the policy network, the search it guides, the Gymnasium environment and the policy-gradient bookkeeping.

NOTHING HERE IS TRUSTED BY THE KERNEL: whatever the network says, an action is probed and a finished proof is checked.  What these tests protect is the SCIENCE: a learner whose probabilities are computed
differently when it samples and when it learns, or whose environment hands out a reward for something else than a kernel-checked proof, would train without error and learn nothing (or the wrong thing).

  A. FEATURES AND DATA.  A goal's tokens are determined, framed and typed (`[CLS] hyp [SEP] .. [TGT] target`); different statements get different tokens; the symbol overlap of a global with its own type is 1;
     the demonstrations of six bank problems decompose into decisions without a gap and batch into arrays of the documented shapes.
  B. THE NETWORK.  The INCREMENTAL decoder (one decision at a time, the thing that searches and rollouts use) computes the same log-probabilities as the BATCHED forward pass (the thing that trains), padding
     included; weights multiply the loss; a positive weight raises the probability of the decisions it is given to and a negative weight lowers it; a saved model loads to the same function.
  C. THE SEARCH.  The stream of candidate actions of a goal is in nondecreasing cost, without repeats, and its cost for a given action is the negative log-probability of the decisions that assemble it; a policy
     search (an untrained network: any order of attempts is the same to the kernel) proves one-step problems, and what it returns is a kernel-checked term whose actions replay from their portable form.
  D. THE ENVIRONMENT.  Gymnasium's API (observations lie in the space, a step returns five things, an illegal option is an error or a penalty, a finished episode refuses steps), the action mask, the rewards
     (`solve_reward` only for a kernel-checked proof, `action_penalty`, `fail_penalty`), truncation, reproducible seeds; a SCRIPTED run that answers the engine's questions with the decisions of a recorded
     proof solves the problem in each of the six cases; a library lemma can be played like a problem.
  E. THE RL LOOP.  The log-probabilities recorded while a rollout samples are the ones the batched network gives to the same decisions (so a policy-gradient step differentiates what was sampled from); a
     greedy rollout is deterministic.
  F. GOALS, BUDGET, HINDSIGHT.  The environment starts on ANY closed statement given as data (`reset(options={'goal': ...})`: the dict of a hindsight example is one) and the episode is solved by the example's
     own proof; the step budget is observed and enforced; a tracked episode reports what each action achieved and returns the theorems it proved, finished or not, kernel-checked; bad data is refused.

    python3 s44_ml_tests.py          (S44_ONLY=A,C to select sections; S44_TRACE=1 for timings)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, time, math, copy, itertools, collections, tempfile, traceback
sys.setrecursionlimit(200000)

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S44_TRACE'))
def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return bool(cond)
def raises(what, fn, exc):
    try: fn()
    except exc: return expect(True, what)
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:100]))
    return expect(False, what + ' (it raised nothing)')
def close(a, b, tol=2e-4): return abs(float(a) - float(b)) <= tol * (1.0 + abs(float(b)))

try:
    import numpy as np
    import torch
    import gymnasium as gym
except ImportError as e:
    print('S44 needs numpy, torch and gymnasium: %s' % (e,)); print('S44 ML GATE: FAIL'); sys.exit(1)
import itp_env_lnl as E
import bank_all
import gym_env as G, gym_policy as P, gym_acts as ACTS
from gym_env import Kind
import ml_features as F, ml_data as D, ml_model as M, ml_search as MS, ml_train as T, ml_rl as RL, bank_env as BE

DEMOS = ('ident_diff_squares', 'ineq_am_gm_two', 'imo1959_p1', 'ineq_cauchy_three', 'coprime_consecutive', 'pow_sub_dvd')
ONE_STEP = ('ident_diff_squares', 'ident_cube_sum')           # proved by one `ring`: any policy finds them within a few attempts
STRICT = ('imo1969_p1',)                                        # a problem with a lemma of its own: another proof in the strict regime
CFG = {'d': 32, 'layers': 1, 'heads': 2, 'ff': 64, 'dropout': 0.0}


def tiny_net(ctx, seed=0):
    torch.manual_seed(seed)
    net = T.make_net(ctx.bags, CFG); net.eval()
    return net


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ fixture
class Fixture:
    def __init__(self):
        torch.set_num_threads(1)
        t0 = time.time()
        self.bk = bank_all.load(); self.w = self.bk.w; self.k = self.w.k
        self.eng = G.Engine(self.w, kinds=set(G.Kind), strict=False)
        self.ctx = D.DataContext(self.w)
        trace('world %.1f s (%d globals), context %.1f s' % (time.time() - t0, self.k.num_globals(), time.time() - t0))
        t0 = time.time()
        w2, _ = self.bk.recordings()
        eng2 = G.Engine(w2, kinds=set(G.Kind), strict=True)
        self.demos = {}; self.strict = {}
        for n in DEMOS + ONE_STEP + STRICT:
            res = self.bk.demonstrate(n, engine=eng2)
            expect(res.ok, 'fixture: the reference proof of %s replays (%s)' % (n, res.gap))
            self.demos[n] = res
            pl = self.bk.placed[n]
            if pl.problem.strict and pl.premises('strict') != pl.premises('prefix'):
                res = self.bk.demonstrate(n, engine=eng2, regime='strict')
                expect(res.ok, 'fixture: the reference proof of %s replays with its lemmas hidden (%s)' % (n, res.gap))
                self.strict[n] = res
        trace('recordings and demonstrations %.1f s' % (time.time() - t0))
        self._traces = {}

    def trace_of(self, name):
        """per action of the demonstration: (goal, [(Choice, index)], Step)"""
        if name in self._traces: return self._traces[name]
        pl = self.bk.placed[name]; eng = self.eng; eng.premises = pl.premises('prefix')
        p = eng.start(pl.stmt); out = []
        for st in self.demos[name].portable:
            g = eng.goal_of(p, 0); m = eng.menu(g)
            menu = m if st.get('kind') in ACTS.MENU_KINDS else ACTS.NO_MENU
            a = ACTS.from_portable(eng, menu, st)
            ds = P.decompose(eng, g, m, a)
            out.append((g, ds, D.make_step(self.ctx, name, len(out), 0, g, ds, a.kind)))
            eng.replay(p, a, m)
        expect(p.done and p.ps.is_complete(), 'fixture: the decomposed actions of %s finish the proof' % name)
        self._traces[name] = out
        return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. features and data
def sec_A(c):
    fx = c['fx']; ctx = fx.ctx; eng = fx.eng
    gfs = {}
    for n in DEMOS:
        pl = fx.bk.placed[n]; eng.premises = pl.premises('prefix'); p = eng.start(pl.stmt); g = eng.goal_of(p, 0)
        gf = F.goal_features(ctx.ser, g); gf2 = F.goal_features(ctx.ser, g); gfs[n] = gf
        expect(int(gf.tokens[0]) == F.CLS and int(gf.seg[0]) == F.SEG_SPECIAL, 'A1 %s: a goal begins with [CLS]' % n)
        expect(np.array_equal(gf.tokens, gf2.tokens) and gf.spans == gf2.spans, 'A1 %s: the features of a goal are determined' % n)
        a, b = gf.tgt
        expect(int(gf.tokens[a - 1]) == F.TGT and b == len(gf.tokens) and a < b, 'A1 %s: the target follows [TGT] and ends the sequence' % n)
        expect(len(gf.spans) == g.n_hyps == gf.n_hyps, 'A1 %s: a span per hypothesis' % n)
        ok = all(0 < s <= e <= len(gf.tokens) and (e == s or int(gf.tokens[e]) == F.SEP) for s, e in gf.spans)
        expect(ok, 'A1 %s: every hypothesis is followed by [SEP]' % n)
        expect(int(gf.tokens.max()) < F.VOCAB and int(gf.tokens.min()) >= 0, 'A1 %s: the tokens are in the vocabulary' % n)
        expect(len(gf.tokens) <= F.MAX_TOTAL, "A1 %s: a goal fits the network's window" % n)
        expect(len(gf.tokens) <= 2 + F.MAX_HYPS * (F.MAX_HYP_TOKENS + 1) + F.MAX_TGT_TOKENS, 'A1 %s: the length is bounded' % n)
        expect(gf.symbols == frozenset(int(x) for x in gf.tokens if x >= F.CONST_BASE), 'A1 %s: the symbols are the constants' % n)
        ov = ctx.bags.overlap(gf.symbols)
        expect(ov.shape == (ctx.bags.n, 2) and float(ov.min()) >= 0.0 and float(ov.max()) <= 1.0, 'A2 %s: the overlap is a fraction for every global' % n)
    expect(len({tuple(gfs[n].tokens.tolist()) for n in DEMOS}) == len(DEMOS), 'A1 different statements have different tokens')
    gid = fx.w.gid['Z.mul_comm']; ty = fx.k.global_info(gid)[1]
    syms = frozenset(x for x in ctx.ser.tokens(ty, 160) if x >= F.CONST_BASE)
    ov = ctx.bags.overlap(syms)
    expect(close(ov[gid, 0], 1.0), 'A2 the symbols of a global\'s own type overlap it completely (Z.mul_comm: %s)' % float(ov[gid, 0]))
    expect(float(ctx.bags.overlap(frozenset())[gid, 0]) == 0.0, 'A2 an empty symbol set overlaps nothing')
    # data
    total = 0
    for n in DEMOS:
        tr = fx.trace_of(n); st = [t[2] for t in tr]; total += len(st)
        pl = fx.bk.placed[n]; stats = collections.Counter()
        ex = D.extract(ctx, fx.eng, pl.stmt, pl.premises('prefix'), fx.demos[n].portable, name=n, stats=stats)
        expect(len(ex) == len(st) == len(fx.demos[n].portable) and stats['ok'] == len(st) and not stats['missing'] and not stats['error'], 'A3 %s: every action of the demonstration decomposes (%s)' % (n, dict(stats)))
        for s in ex:
            ok = s.n_decisions >= 1 and all(0 <= int(w_) < len(r) and r.shape[1] == 4 for w_, r in zip(s.want, s.rows)) and len(s.rows) == len(s.holes) == s.n_decisions
            expect(ok, 'A3 %s: a decision wants one of its options' % n)
        b = D.collate(ex)
        S_, L = b['tokens'].shape
        expect(S_ == len(ex) and b['seg'].shape == (S_, L) and b['hyp_of'].shape == (S_, L) and np.array_equal(b['mask'], b['tokens'] != 0), 'A4 %s: the batch arrays have the shape of the goals' % n)
        expect(b['overlap'].shape == (S_, ctx.bags.n, 2) and b['weights'] is None and b['remaining'].shape == (S_,), 'A4 %s: overlap, weights and remaining' % n)
        ok = all(int(t['counts'].sum()) == len(t['rows']) and len(t['act']) == len(t['want']) == len(t['counts']) for t in b['time'])
        expect(ok and len(b['time']) == max(s.n_decisions for s in ex), 'A4 %s: the options of a time step are the rows of its decisions' % n)
        expect(int(b['hyp_of'].max()) < int(b['n_hyps'].max()) if b['n_hyps'].max() else True, 'A4 %s: hyp_of names a hypothesis of the goal' % n)
        bw = D.collate(ex, weights=[0.5] * len(ex))
        expect(bw['weights'].dtype == np.float32 and bw['weights'].shape == (len(ex),), 'A4 %s: weights ride along' % n)
    expect(total >= 18, 'A3 the six demonstrations have at least 18 actions (%d)' % total)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. the network
def sum_logp(net, step):
    out = net(D.collate([step], max_dec=10 ** 4))
    return -out['policy_loss'].item() * out['n'], out['n']


def sec_B(c):
    fx = c['fx']; ctx = fx.ctx
    net = tiny_net(ctx)
    traces = [(n, t) for n in DEMOS for t in fx.trace_of(n)]
    steps = [t[2] for n, t in traces]
    batch = D.collate(steps, max_dec=10 ** 4)
    out = net(batch, want_logits=True)
    expect(math.isfinite(out['loss'].item()) and out['n'] > 0 and out['n'] <= sum(s.n_decisions for s in steps), 'B1 the loss is finite and counts the decisions with a real choice (%d)' % out['n'])
    expect(out['value'].shape == (len(steps),), 'B1 one value per step')
    # the batched log-probabilities of the wanted options
    bl = {}
    for t, (act, logp) in enumerate(out['logits']):
        want = batch['time'][t]['want']
        for j, i in enumerate(act.tolist()): bl[(i, t)] = float(logp[j, int(want[j])])
    with torch.no_grad(): gtab = net.global_table()
    worst = 0.0; n_cmp = 0; first_scores = {}
    for si, (n, (g, ds, step)) in enumerate(traces):
        dec = M.Decoder(net, ctx, g, gtab); carry = dec.root()
        for t, (ch, i) in enumerate(ds):
            logp, oe, hs = dec.scores(carry, ch)
            lp = float(logp[i]); carry = dec.child(carry, ch, i, oe, hs)
            if t == 0: first_scores[si] = logp
            worst = max(worst, abs(lp - bl[(si, t)])); n_cmp += 1
    expect(n_cmp == sum(s.n_decisions for s in steps) and worst < 2e-4, 'B2 the incremental decoder and the batched network give the same log-probabilities (%d decisions, worst difference %.2e)' % (n_cmp, worst))
    # a decoder given the features gives those of a decoder that computes them
    n0, (g0, ds0, st0) = traces[0]
    dec_a = M.Decoder(net, ctx, g0, gtab)
    gf0 = F.goal_features(ctx.ser, g0)
    dec_b = M.Decoder(net, ctx, None, gtab, gf=gf0, ov=ctx.bags.overlap(gf0.symbols))
    la = dec_a.scores(dec_a.root(), ds0[0][0])[0]; lb = dec_b.scores(dec_b.root(), ds0[0][0])[0]
    expect(torch.allclose(la, lb, atol=1e-5), 'B2 a decoder given the goal\'s features is the decoder that computes them')
    # one step alone against the batch (padding does not matter)
    k = max(range(len(steps)), key=lambda i: len(steps[i].goal.tokens))
    one = D.collate([steps[k]], max_dec=10 ** 4)
    alone = net(one, want_logits=True)
    ok = all(close(float(lp[0, int(one['time'][t]['want'][0])]), bl[(k, t)], 2e-4) for t, (act, lp) in enumerate(alone['logits']))
    expect(ok and len(alone['logits']) == steps[k].n_decisions, 'B2 the longest goal alone and in the batch (padding)')
    # weights
    l1 = net(D.collate(steps, max_dec=10 ** 4))['policy_loss'].item()
    l2 = net(D.collate(steps, max_dec=10 ** 4, weights=[2.0] * len(steps)))['policy_loss'].item()
    l3 = net(D.collate(steps, max_dec=10 ** 4, weights=[-1.0] * len(steps)))['policy_loss'].item()
    l0 = net(D.collate(steps, max_dec=10 ** 4, weights=[0.0] * len(steps)))['policy_loss'].item()
    expect(close(l2, 2 * l1) and close(l3, -l1) and l0 == 0.0, 'B3 weights multiply the loss (%.4f, x2 %.4f, x-1 %.4f, x0 %.4f)' % (l1, l2, l3, l0))
    # direction of a policy-gradient step
    cand = [s for s in steps if sum_logp(net, s)[1] >= 1]
    expect(len(cand) >= 5, 'B4 there are steps with a real choice (%d)' % len(cand))
    moved = 0
    for s in cand[:4]:
        before, n_ = sum_logp(net, s)
        for sign in (+1.0, -1.0):
            n2 = copy.deepcopy(net); n2.train()
            opt = torch.optim.SGD(n2.parameters(), lr=0.05)
            res = n2(D.collate([s], weights=[sign], max_dec=10 ** 4)); opt.zero_grad(); res['policy_loss'].backward()
            torch.nn.utils.clip_grad_norm_(n2.parameters(), 1.0); opt.step(); n2.eval()
            after, _ = sum_logp(n2, s)
            moved += int((after - before) * sign > 0)
    expect(moved == 2 * len(cand[:4]), 'B4 a positive weight raises the probability of the decisions it is given to, a negative one lowers it (%d of %d)' % (moved, 2 * len(cand[:4])))
    # save and load
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'm.pt'); T.save(net, CFG, path, {'tag': 's44'})
        net2, cfg2, extra = T.load(path, ctx.bags)
        a_ = net(D.collate(steps[:6], max_dec=10 ** 4))['policy_loss'].item(); b_ = net2(D.collate(steps[:6], max_dec=10 ** 4))['policy_loss'].item()
        expect(cfg2 == CFG and extra.get('tag') == 's44' and a_ == b_, 'B5 a saved model loads to the same function')
    # a different seed is a different function
    other = tiny_net(ctx, seed=1)
    expect(abs(other(D.collate(steps[:6], max_dec=10 ** 4))['policy_loss'].item() - a_) > 1e-6, 'B5 two seeds are two functions')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. the search
def sec_C(c):
    fx = c['fx']; ctx = fx.ctx; eng = fx.eng
    net = tiny_net(ctx)
    n = 'ident_diff_squares'; pl = fx.bk.placed[n]
    eng.premises = pl.premises('prefix'); p = eng.start(pl.stmt); g = eng.goal_of(p, 0)
    s = MS.PolicySearcher(eng, net, ctx, topk=10 ** 6, cap=1e9)
    menu = s.menu_of(g)
    stream = s.stream_of(g, menu)
    first = list(itertools.islice(stream.upto(1e9), 60))
    costs = [cost for a, cost in first]
    expect(len(first) == 60 and all(costs[i] <= costs[i + 1] + 1e-9 for i in range(59)) and costs[0] >= 0.0, 'C1 the candidate actions come in nondecreasing cost')
    sigs = [menu.signature(a) for a, cost in first]
    expect(len(set(sigs)) == len(sigs), 'C1 ... without repeats')
    again = list(itertools.islice(MS.ActionStream(s, g, menu).upto(1e9), 60))
    expect([menu.signature(a) for a, cost in again] == sigs and all(close(x[1], y) for x, y in zip(again, costs)), 'C1 ... in the same order every time')
    cheap = list(stream.upto(costs[9] + 1e-9)); wide = list(stream.upto(costs[30] + 1e-9))
    expect([menu.signature(a) for a, cost in cheap] == sigs[:len(cheap)] and len(wide) >= 31, 'C1 a larger bound continues the stream where the smaller stopped')
    # the stream's cost of the demonstration's first action is the negative log-probability of its decisions
    tr = fx.trace_of(n); g0, ds0, st0 = tr[0]
    with torch.no_grad(): gtab = net.global_table()
    dec = M.Decoder(net, ctx, g0, gtab); carry = dec.root(); cost = 0.0
    for ch, i in ds0:
        logp, oe, hs = dec.scores(carry, ch); cost -= float(logp[i]); carry = dec.child(carry, ch, i, oe, hs)
    a_demo = ACTS.from_portable(eng, menu, fx.demos[n].portable[0])
    sig = menu.signature(a_demo)
    full = list(stream.upto(cost + 1e-6))
    found = [cc for a, cc in full if menu.signature(a) == sig]
    expect(len(found) == 1 and close(found[0], cost, 1e-3), 'C2 the demonstration\'s action is in the stream at the cost of its decisions (stream %s, decisions %.4f)' % (found, cost))
    # the search proves one-step problems; the term is the kernel's
    for name in ONE_STEP:
        pl = fx.bk.placed[name]
        term, tree, srch = MS.prove_statement(eng, pl.stmt, pl.premises('prefix'), net, ctx, max_probes=3000)
        expect(term is not None and tree is not None, 'C3 %s: the policy search proves it (%s probes)' % (name, None if srch is None else srch.probes))
        if term is None: continue
        eng.premises = pl.premises('prefix')
        steps = MS.tree_portable(eng, pl.stmt, pl.premises('prefix'), tree)
        expect(len(steps) == tree.size() and len(steps) >= 1, 'C3 %s: the tree has the actions of the proof (%d)' % (name, len(steps)))
        t2 = ACTS.replay_portable(eng, pl.stmt, steps, pl.premises('prefix'))
        expect(t2 is not None, 'C3 %s: its portable actions finish the proof in a fresh episode' % name)
        expect(srch.probes <= 3000 and srch.probes >= 1, 'C3 %s: the probes are counted (%d)' % (name, srch.probes))
    # a budget of nothing proves nothing, and says so
    pl = fx.bk.placed['coprime_consecutive']
    term, tree, srch = MS.prove_statement(eng, pl.stmt, pl.premises('prefix'), net, ctx, max_probes=3)
    expect(term is None and tree is None, 'C4 three probes do not prove coprime_consecutive')
    # the same search twice is the same
    pl = fx.bk.placed['ident_cube_sum']
    r1 = MS.prove_statement(eng, pl.stmt, pl.premises('prefix'), net, ctx, max_probes=3000)
    r2 = MS.prove_statement(eng, pl.stmt, pl.premises('prefix'), net, ctx, max_probes=3000)
    expect(r1[2].probes == r2[2].probes and r1[1].signature == r2[1].signature, 'C5 the search is deterministic (%d probes both times)' % r1[2].probes)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. the environment
def scripted(fx, env, name, regime='prefix', problem=None):
    """play the recorded proof of `name` through the environment, answering each question with the option the decomposition wants. Returns (total reward, steps, last info, terminated)"""
    pl = fx.bk.placed[name]; eng = fx.eng
    obs, info = env.reset(options={'problem': problem or name, 'regime': regime})
    total = 0.0; term = trunc = False; n_actions = 0
    demo = fx.strict[name] if regime == 'strict' else fx.demos[name]
    for st in demo.portable:
        eng.premises = pl.premises(regime)
        g = env.ep.goal(); m = env.ep.menu()
        menu = m if st.get('kind') in ACTS.MENU_KINDS else ACTS.NO_MENU
        a = ACTS.from_portable(eng, menu, st)
        ds = P.decompose(eng, g, m, a); n_actions += 1
        for ch, i in ds:
            if info.get('choice') is None:
                return total, n_actions, info, term
            if info['choice'].name != ch.name or len(info['choice'].options) != len(ch.options):
                return total, n_actions, dict(info, mismatch=(info['choice'].name, ch.name)), term
            obs, r, term, trunc, info = env.step(i); total += r
    return total, n_actions, info, term


def sec_D(c):
    fx = c['fx']; ctx = fx.ctx; eng = fx.eng; bk = fx.bk
    env = BE.BankEnv(bk, names=list(DEMOS), ctx=ctx, engine=eng, max_actions=40, max_decisions=400)
    obs, info = env.reset(options={'problem': 'coprime_consecutive'})
    expect(env.observation_space.contains(obs), 'D1 the first observation lies in the observation space')
    expect(isinstance(info, dict) and info['problem'] == 'coprime_consecutive' and info['actions'] == 0 and info['refused'] == 0 and not info['solved'], 'D1 reset returns (observation, info)')
    expect(int(info['action_mask'].sum()) == len(info['choice'].options) == int(obs['n_options']) == int(obs['mask'].sum()) and np.array_equal(info['action_mask'], env.action_masks()), 'D1 the mask marks exactly the options')
    expect(env.action_space.n == 1024, 'D1 the action space is the option index')
    n0 = len(info['choice'].options)
    raises('D2 an option beyond the decision\'s options is an error', lambda: env.step(n0), ValueError)
    raises('D2 ... and so is a negative one', lambda: env.step(-1), ValueError)
    # the scripted runs
    for n in DEMOS:
        env = BE.BankEnv(bk, names=list(DEMOS), ctx=ctx, engine=eng, max_actions=40, max_decisions=400, solve_reward=2.0, action_penalty=0.1)
        total, na, info, term = scripted(fx, env, n)
        expect(term and env.solved and 'mismatch' not in info and not info.get('refused'), 'D3 %s: the recorded proof, said decision by decision, solves it (%s)' % (n, {k: v for k, v in info.items() if k in ('actions', 'refused', 'decisions', 'mismatch')}))
        expect(term and close(total, 2.0 - 0.1 * na), 'D3 %s: the reward is solve_reward less the price of the actions (%.3f for %d actions)' % (n, total, na))
        if term:
            t = env.term; w = fx.w
            try: w.check(t, w.term(bk.placed[n].stmt, 'statement')); ok = True
            except Exception as e: ok = False
            expect(ok, 'D3 %s: the environment\'s term is a proof of the statement' % n)
            raises('D3 %s: a finished episode takes no more steps' % n, lambda: env.step(0), RuntimeError)
            expect(info['actions'] == na and 'action_mask' not in info and 'choice' not in info, 'D3 %s: the final info has the counts and no decision' % n)
    # the strict regime hides the lemmas written for the problem: another proof, and the same environment
    for n in STRICT:
        env = BE.BankEnv(bk, names=[n], ctx=ctx, engine=eng, max_actions=80, max_decisions=2000)
        for regime in ('prefix', 'strict'):
            total, na, info, term = scripted(fx, env, n, regime=regime)
            expect(term and env.solved, 'D3 %s: the recorded proof solves it in the %s regime (%s)' % (n, regime, {k: v for k, v in info.items() if k in ('actions', 'refused', 'decisions', 'mismatch')}))
    expect(set(fx.strict) == set(STRICT), 'D3 the fixture has the strict proofs it plays (%s)' % (sorted(fx.strict),))
    # a library lemma plays like a problem
    pl = bk.placed['ineq_cauchy_three']
    env = BE.BankEnv(bk, names=[], ctx=ctx, engine=eng, lemmas={'alias': (pl.stmt, pl.premises('prefix'))})
    total, na, info, term = scripted(fx, env, 'ineq_cauchy_three', problem='alias')
    expect(term and env.solved and env.name == 'alias' and info['problem'] == 'alias', 'D4 a statement with its premises, given as a lemma, is played like a problem')
    # random play: the counts add up, the mask is right, nothing raises
    rng = np.random.default_rng(7)
    for n in ('coprime_consecutive', 'pow_sub_dvd', 'ineq_am_gm_two'):
        env = BE.BankEnv(bk, names=[n], ctx=ctx, engine=eng, max_actions=12, max_decisions=150)
        obs, info = env.reset(options={'problem': n}); steps = 0; ok = True; cur = None
        while True:
            ch = info.get('choice')
            if ch is None: break
            ok &= int(info['action_mask'].sum()) == len(ch.options) and env.observation_space.contains(obs)
            obs, r, term, trunc, info = env.step(int(rng.integers(len(ch.options)))); steps += 1
            if term or trunc: break
            if steps > 400: break
        taken = sum(1 for t in env.trajectory if t[3]); refused = sum(1 for t in env.trajectory if t[3] is False)
        expect(steps <= 400 and (term or trunc), 'D5 %s: a random episode ends (%d decisions)' % (n, steps))
        expect(ok and taken == info['actions'] and refused <= info['refused'] and info['decisions'] == steps, 'D5 %s: the counts add up (actions %d/%d, refused %d/%d)' % (n, taken, info['actions'], refused, info['refused']))
        expect(info['actions'] + info['refused'] <= 12 + 1 and (info['actions'] + info['refused'] >= 12 or term or info['decisions'] >= 150), 'D5 %s: an episode ends by a proof or by a limit' % n)
    # fail penalty and truncation
    env = BE.BankEnv(bk, names=['coprime_consecutive'], ctx=ctx, engine=eng, max_actions=2, max_decisions=300, fail_penalty=0.5, action_penalty=0.0)
    obs, info = env.reset(options={'problem': 'coprime_consecutive'}); rewards = []; end = None
    for _ in range(300):
        ch = info.get('choice')
        if ch is None: break
        obs, r, term, trunc, info = env.step(0); rewards.append(r)
        if term or trunc: end = (term, trunc); break
    expect(end is not None and (end[0] or end[1]), 'D6 an episode of at most 2 actions ends')
    expect(close(sum(rewards), -0.5 * info['refused'] + (1.0 if env.solved else 0.0)), 'D6 a refused action costs fail_penalty and nothing else does (%s, %d refused)' % (rewards, info['refused']))
    expect(info['actions'] + info['refused'] <= 2 and (env.solved or info['actions'] + info['refused'] == 2 or info['decisions'] >= 300), 'D6 the limit counts the actions taken and refused')
    # illegal options: the penalising mode loses the decision and nothing else
    env = BE.BankEnv(bk, names=['coprime_consecutive'], ctx=ctx, engine=eng, fail_penalty=0.25, illegal='penalize')
    obs, info = env.reset(options={'problem': 'coprime_consecutive'}); ch0 = info['choice']
    obs2, r, term, trunc, info2 = env.step(10 ** 6)
    expect(r == -0.25 and not term and info2['choice'] is ch0 and info2['decisions'] == 1 and info2['actions'] == 0, 'D7 illegal=penalize: the decision is asked again and costs fail_penalty')
    raises('D7 illegal must be raise or penalize', lambda: BE.BankEnv(bk, names=[], ctx=ctx, engine=eng, illegal='ignore'), ValueError)
    # seeds
    env = BE.BankEnv(bk, names=list(DEMOS), ctx=ctx, engine=eng)
    a1 = [env.reset(seed=5)[1]['problem'] for _ in range(1)] + [env.reset()[1]['problem'] for _ in range(4)]
    a2 = [env.reset(seed=5)[1]['problem'] for _ in range(1)] + [env.reset()[1]['problem'] for _ in range(4)]
    expect(a1 == a2 and len(set(a1)) > 1, 'D8 the same seed gives the same problems (%s)' % (a1,))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. the RL loop
def sec_E(c):
    fx = c['fx']; ctx = fx.ctx; eng = fx.eng; bk = fx.bk
    net = tiny_net(ctx, seed=3)
    with torch.no_grad(): gtab = net.global_table()
    env = BE.BankEnv(bk, names=list(DEMOS), ctx=ctx, engine=eng, max_actions=10, max_decisions=120)
    worst = 0.0; n_roll = 0; n_dec = 0
    for seed, name in enumerate(('coprime_consecutive', 'ident_diff_squares', 'ineq_am_gm_two', 'pow_sub_dvd', 'imo1959_p1')):
        rng = np.random.default_rng(seed); rec = []
        solved, steps, info = RL.rollout(env, net, ctx, gtab, name, 'prefix', rng, 1.0, False, rec)
        expect(solved == env.solved and info['decisions'] == len(rec), 'E1 %s: a rollout records one log-probability per decision (%d)' % (name, len(rec)))
        if not steps: continue
        b = D.collate([s for s, taken in steps], max_dec=10 ** 4)
        out = net(b)
        mine = -out['policy_loss'].item() * out['n']
        worst = max(worst, abs(mine - sum(rec))); n_roll += 1; n_dec += len(rec)
        expect(sum(1 for s, taken in steps if taken is True) == info['actions'] and sum(1 for s, taken in steps if taken is False) <= info['refused'] and sum(1 for s, taken in steps if taken is None) <= 1, 'E1 %s: the steps are the attempted actions: taken, refused, or the one the episode ended in' % name)
        expect(out['value_loss'].item() == 0.0, 'E1 %s: a rollout\'s steps carry no value target' % name)
    expect(n_roll >= 4 and worst < 1e-2, 'E2 the log-probabilities recorded while sampling are the ones the batched network gives the same decisions (%d rollouts, %d decisions, worst difference %.2e)' % (n_roll, n_dec, worst))
    # greedy rollouts are deterministic
    r1 = RL.rollout(env, net, ctx, gtab, 'coprime_consecutive', 'prefix', np.random.default_rng(0), 1.0, True)
    r2 = RL.rollout(env, net, ctx, gtab, 'coprime_consecutive', 'prefix', np.random.default_rng(99), 1.0, True)
    w1 = [list(map(int, s.want)) for s, t in r1[1]]; w2 = [list(map(int, s.want)) for s, t in r2[1]]
    expect(w1 == w2 and len(w1) >= 1, 'E3 a greedy rollout does not depend on the random generator')
    # sampling at temperature 1 follows the policy: over many first decisions, the frequencies are the probabilities
    rng = np.random.default_rng(11); counts = None
    obs, info = env.reset(options={'problem': 'coprime_consecutive'})
    dec = M.Decoder(net, ctx, None, gtab, gf=info['goal_feat'], ov=info['overlap'])
    logp, oe, hs = dec.scores(dec.root(), info['choice']); pr = np.exp(logp.numpy().astype(np.float64)); pr /= pr.sum()
    draws = np.bincount([int(rng.choice(len(pr), p=pr)) for _ in range(4000)], minlength=len(pr)) / 4000.0
    expect(float(np.abs(draws - pr).max()) < 0.04, 'E3 the policy\'s probabilities are a distribution one can sample (max deviation %.3f)' % float(np.abs(draws - pr).max()))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. goals, budget, hindsight
def decide(env, info, steps):
    """answer the engine's questions with the decisions that say the portable `steps`, one action after the other. Returns (total reward, decisions, last observation, last info, terminated, truncated)"""
    eng = env.eng; total = 0.0; nd = 0; term = trunc = False; obs = None
    for st in steps:
        if env.choice is None: break
        g = env.ep.goal(); m = env.ep.menu()
        a = ACTS.from_portable(eng, m if st['kind'] in ACTS.MENU_KINDS else ACTS.NO_MENU, st)
        for ch, i in P.decompose(eng, g, m, a):
            obs, r, term, trunc, info = env.step(i); total += r; nd += 1
    return total, nd, obs, info, term, trunc


def sec_F(c):
    import json
    import gym_tree as GTR
    fx = c['fx']; ctx = fx.ctx; eng = fx.eng; bk = fx.bk; w = fx.w
    data = {p['name']: p for p in json.load(open(os.path.join(__import__('itp_paths').ROOT, 'data', 'dataset_v2', 'bank.json')))['problems']}
    mk = lambda **kw: BE.BankEnv(bk, names=['imo1959_p1', 'quad_form_zero'], ctx=ctx, engine=eng, max_actions=60, max_decisions=900, **kw)
    steps4 = data['imo1959_p1']['demonstration']['steps']; steps13 = data['quad_form_zero']['demonstration']['steps']
    expect(len(steps4) == 4 and len(steps13) == 13, 'F0 (the two proofs the section plays: %d and %d actions)' % (len(steps4), len(steps13)))
    # ---- F1. the budget is observed and counts the actions
    env = mk()
    obs, info = env.reset(options={'problem': 'imo1959_p1'})
    expect(env.observation_space.contains(obs) and int(obs['budget']) == 60 == info['remaining'], 'F1 the default budget is max_actions, and it is in the observation (%s)' % (obs['budget'],))
    obs, info = env.reset(options={'problem': 'imo1959_p1', 'budget': 5})
    expect(env.observation_space.contains(obs) and int(obs['budget']) == 5 == info['remaining'] == env.budget, 'F1 an episode can have a budget of its own')
    total, nd, obs, info, term, trunc = decide(env, info, steps4[:2])
    expect(info['actions'] == 2 and info['remaining'] == 3 and int(obs['budget']) == 3 and not term and not trunc, 'F1 two actions taken: three left (%s)' % ({k: info[k] for k in ('actions', 'refused', 'remaining')},))
    total, nd, obs, info, term, trunc = decide(env, info, steps4[2:])
    expect(term and env.solved and info['remaining'] == 1 and total == 1.0, 'F1 the proof of four actions ends the episode with the reward and one action of the budget left (%s)' % (info['remaining'],))
    # ---- F2. the budget is enforced
    obs, info = env.reset(options={'problem': 'imo1959_p1', 'budget': 2})
    total, nd, obs, info, term, trunc = decide(env, info, steps4)
    expect(trunc and not term and not env.solved and info['remaining'] == 0 and info['actions'] == 2 and int(obs['budget']) == 0, 'F2 a budget of 2 truncates the episode after two actions (%s)' % ({k: info[k] for k in ('actions', 'refused', 'remaining')},))
    raises('F2 a truncated episode takes no more steps', lambda: env.step(0), RuntimeError)
    # refused actions count too: always answering with option 0 either proves something or is refused, and the budget is spent
    env0 = BE.BankEnv(bk, names=['coprime_consecutive'], ctx=ctx, engine=eng, max_actions=60, max_decisions=400)
    obs, info = env0.reset(options={'problem': 'coprime_consecutive', 'budget': 3}); end = None
    for _ in range(400):
        if info.get('choice') is None: break
        obs, r, term, trunc, info = env0.step(0)
        if term or trunc: break
    expect(info['actions'] + info['refused'] <= 3 and info['remaining'] == max(0, 3 - info['actions'] - info['refused']) and (env0.solved or info['remaining'] == 0 or info['decisions'] >= 400), 'F2 the budget counts the actions taken and refused (%s)' % ({k: info[k] for k in ('actions', 'refused', 'remaining')},))
    # ---- F3. bad budgets
    for b in (0, -1, 2.5, True, '3', [1]):
        raises('F3 a budget of %r is refused' % (b,), lambda b=b: env.reset(options={'problem': 'imo1959_p1', 'budget': b}), ValueError)
    obs, info = env.reset(options={'problem': 'imo1959_p1', 'budget': np.int64(7)}); expect(info['remaining'] == 7, 'F3 a numpy integer is a budget')
    # ---- F4. without tracking
    env = mk()
    obs, info = env.reset(options={'problem': 'imo1959_p1'})
    expect('achieved' not in info and env.ep.tree is None, 'F4 an environment made without track has no `achieved` and no tree')
    raises('F4 ... and no hindsight', lambda: env.hindsight(), RuntimeError)
    decide(env, info, steps4)
    rec = env.episode_record()
    expect(json.loads(json.dumps(rec)) == rec and rec['solved'] and rec['actions'] == 4 and 'steps' not in rec and rec['problem'] == 'imo1959_p1', 'F4 the episode record is plain data (%s)' % (sorted(rec),))
    raises('F4 hindsight before any episode', lambda: mk(track=True).hindsight(), RuntimeError)
    # ---- F5. tracked: what each action achieved
    env = mk(track=True)
    obs, info = env.reset(options={'problem': 'quad_form_zero'})
    expect(info['achieved'] == [], 'F5 nothing is achieved before the first action')
    solved = []; derived = []; per_step = []
    for st in steps13:
        total, nd, obs, info, term, trunc = decide(env, info, [st])
        per_step.append(info['achieved'])
        for a in info['achieved']:
            (solved if a['kind'] == 'solved' else derived).append(a)
    expect(len(solved) == 13 and len(derived) == 6 and {a['node'] for a in solved} == set(range(13)), 'F5 over the proof: every node is solved once (%d), six facts are derived (%d)' % (len(solved), len(derived)))
    expect(all(set(a) == {'kind', 'node', 'step', 'depth', 'length'} for a in solved + derived), 'F5 an achievement has kind, node, step, depth, length')
    last = per_step[-1]
    expect(last and last[-1]['node'] == 0 and last[-1]['depth'] == 0 and last[-1]['length'] == 13 and last[-1]['step'] == 0, 'F5 the last action completes the root: the whole proof, 13 actions (%s)' % (last[-1:],))
    expect(all(a['length'] == 1 for a in derived), 'F5 a derived fact is one action')
    expect(term and env.solved, 'F5 (the proof is finished)')
    # ---- F6. hindsight of an unfinished episode
    obs, info = env.reset(options={'problem': 'quad_form_zero'})
    total, nd, obs, info, term, trunc = decide(env, info, steps13[:7])
    expect(not term and not env.solved and info['actions'] == 7, 'F6 seven of thirteen actions')
    part = env.hindsight(); keys_part = {e.key for e in part}
    expect(len(part) > 0 and all(e.verified and e.proof is not None for e in part), 'F6 an unfinished episode already proved %d theorems, each verified' % len(part))
    short = env.hindsight(max_len=1)
    expect(all(e.length <= 1 for e in short) and len(short) <= len(part), 'F6 the keywords of `examples` pass through (max_len=1: %d)' % len(short))
    rec = env.episode_record()
    expect(json.loads(json.dumps(rec)) == rec and rec['actions'] == 7 and len(rec['steps']) == 7 and rec['steps'] == steps13[:7] and not rec['solved'], 'F6 the tracked record has the portable steps taken')
    obs, info = env.reset(options={'problem': 'quad_form_zero'})
    total, nd, obs, info, term, trunc = decide(env, info, steps13)
    full = env.hindsight(); keys_full = {e.key for e in full}
    expect(term and keys_part <= keys_full and len(full) == 18, 'F6 ... and they are among the %d theorems of the whole proof' % len(full))
    # ---- F7. goals given as data: every example is solved from its own data
    n_ok = 0; bad = []
    for i, ex in enumerate(full):
        d = json.loads(json.dumps(ex.to_dict()))
        obs, info = env.reset(options={'goal': d, 'budget': 20})
        expect_ok = env.observation_space.contains(obs) and info['problem'].startswith('goal:') and int(obs['budget']) == 20
        total, nd, obs, info, term, trunc = decide(env, info, d['steps'])
        if not (expect_ok and term and env.solved and total == 1.0 and info['actions'] == len(d['steps'])): bad.append((i, ex.kind, info.get('problem'), total)); continue
        try: w.check(env.term, env.ep.stmt)
        except Exception as e: bad.append((i, 'term refused by the kernel', str(e)[:60])); continue
        n_ok += 1
    expect(n_ok == len(full) and not bad, 'F7 all %d hindsight examples of quad_form_zero are solved through the environment from their own data, with a term the kernel accepts (%s)' % (len(full), bad[:3]))
    d = full[0].to_dict()
    obs, info = env.reset(options={'goal': d, 'name': 'mine'}); expect(env.name == 'mine' and info['problem'] == 'mine', 'F7 the goal can be given a name')
    obs, info = env.reset(options={'goal': dict(d, name='theirs')}); expect(info['problem'] == 'theirs', 'F7 ... or brings its own')
    obs, info = env.reset(seed=3)
    expect(info['problem'] in ('imo1959_p1', 'quad_form_zero') and env.ep.tree is not None and len(env.ep.tree.nodes) == 1 and info['actions'] == 0, 'F7 reset without options is back to the bank, with a fresh tree')
    expect(not env.ep.tree.by_step and info['achieved'] == [], 'F7 ... which has recorded nothing')
    # a statement of this world, with its premises
    pl = bk.placed['imo1959_p1']
    obs, info = env.reset(options={'statement': pl.stmt, 'premises': pl.premises('prefix')})
    total, nd, obs, info, term, trunc = decide(env, info, steps4)
    expect(term and env.solved and info['problem'] == 'statement', 'F7 a statement given as a term is played like a problem')
    # ---- F8. bad data
    for what, goal in (('not a dict', 5), ('no code', {'statement_code': None, 'premises': 0}), ('no code key', {'premises': 0}), ('an unknown token', {'statement_code': [['zzz']], 'premises': 0}),
                       ('a code that says no closed statement', {'statement_code': [['n', 1]], 'premises': 0}), ('an empty code', {'statement_code': [], 'premises': 0})):
        raises('F8 a goal that is %s is refused' % what, lambda goal=goal: env.reset(options={'goal': goal}), ValueError)
    # the engine closes `forall n, n = n` before the agent can act
    with w.nota.using(w['Z']):
        triv = w.forall_([w['Z']], lambda n: w.Eq(n, n))
    raises('F8 a statement the engine closes at once is refused (nothing to prove)', lambda: env.reset(options={'statement': triv, 'premises': None}), ValueError)
    obs, info = env.reset(options={'problem': 'imo1959_p1'}); expect(info['problem'] == 'imo1959_p1', 'F8 ... and the environment is still usable after a refused reset')


SECTIONS = [('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E), ('F', sec_F)]


def main():
    t_start = time.time()
    c = {'fx': Fixture()}
    only = os.environ.get('S44_ONLY')
    for name, fn in SECTIONS:
        if only and name not in only.split(','): continue
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:300])); traceback.print_exc(limit=8)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
    print('S44: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:60]: print('  FAIL:', f)
    print('S44 ML GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
