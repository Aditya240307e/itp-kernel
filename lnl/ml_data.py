"""ml_data.py -- behaviour-cloning data: a demonstration (a proof as a sequence of actions) decomposed into the DECISIONS of the policy assembler, with the features of the goal each action was taken in.

    ctx = DataContext(world)                                      # tokeniser, symbol bags of the world's globals
    steps = extract(ctx, eng, stmt, premises, portable_steps, name='imo1959_p1')       # [Step, ..]
    batch = collate(steps_subset)                                   # numpy arrays for the network (see ml_model)

A `Step` is one action taken in one goal: the goal's tokens, the symbol overlap with every global, and the list of decisions the assembler asked for (choice name, option rows, hole tokens, the index of the
option the demonstration chose).  A decision with one option is forced and is kept (the network's recurrent state needs it) but not trained on.
"""
import os, sys, time, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itp_env_lnl as E
import gym_env as G, gym_acts as ACTS, gym_policy as P
from gym_env import Kind
import ml_features as F


class DataContext:
    def __init__(self, w):
        self.w = w; self.k = w.k
        self.ser = F.Serializer(self.k)
        self.bags = F.GlobalBags(self.k, self.ser)


class Step:
    __slots__ = ('name', 'idx', 'remaining', 'goal', 'overlap', 'kind', 'ch_name', 'ch_kind', 'ch_pos', 'rows', 'holes', 'want')
    def __init__(self, name, idx, remaining, goal, overlap, kind, ch_name, ch_kind, ch_pos, rows, holes, want):
        self.name = name; self.idx = idx; self.remaining = remaining; self.goal = goal; self.overlap = overlap; self.kind = kind
        self.ch_name = ch_name; self.ch_kind = ch_kind; self.ch_pos = ch_pos; self.rows = rows; self.holes = holes; self.want = want
    @property
    def n_decisions(self): return len(self.ch_name)


def make_step(ctx, name, idx, remaining, g, decisions, kind, gf=None, ov=None):
    """a Step from a goal and the decisions of an action: `decisions` is a list of (Choice, index); `gf`, `ov`: the goal's features when the caller has them already (g may then be None)"""
    if gf is None: gf = F.goal_features(ctx.ser, g)
    ov = (ctx.bags.overlap(gf.symbols) if ov is None else ov).astype(np.float16)
    names = []; kinds = []; poss = []; rows = []; holes = []; want = []
    for ch, i in decisions:
        names.append(F.CHOICE_ID[ch.name]); kinds.append(-1 if ch.kind is None else int(ch.kind)); poss.append(min(ch.pos, 31))
        rows.append(F.rows_array(ch.options)); holes.append(F.hole_tokens(ctx.ser, ch.hole)); want.append(i)
    return Step(name, idx, remaining, gf, ov, int(kind), np.asarray(names, dtype=np.int8), np.asarray(kinds, dtype=np.int16), np.asarray(poss, dtype=np.int8), rows, holes, np.asarray(want, dtype=np.int32))


def extract(ctx, eng, stmt, premises, steps, name='', stats=None, skip_missing=True):
    """replay the portable `steps` of a proof of `stmt` in `eng` and decompose every action into decisions. Returns [Step]; an action that cannot be decomposed (the assembler does not offer what the
    demonstration says: `Missing`) ends the extraction there (later goals would be wrong), counted in `stats`."""
    stats = stats if stats is not None else collections.Counter()
    eng.premises = premises
    p = eng.start(stmt); out = []; total = len(steps)
    for si, st in enumerate(steps):
        if not p.n_goals: break
        g = eng.goal_of(p, 0); m = eng.menu(g)
        name_k = st.get('kind')
        menu = m if name_k in ACTS.MENU_KINDS else ACTS.NO_MENU
        a = ACTS.from_portable(eng, menu, st)
        try:
            ds = P.decompose(eng, g, m, a)
            out.append(make_step(ctx, name, si, total - si, g, ds, a.kind)); stats['ok'] += 1
        except P.Missing as e:
            stats['missing'] += 1; stats['missing ' + Kind(a.kind).name] += 1
            if not skip_missing: raise
        except Exception as e:
            stats['error'] += 1; stats['error ' + type(e).__name__] += 1
            if not skip_missing: raise
        eng.replay(p, a, m)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ batching
def collate(steps, max_tokens=512, max_dec=128, weights=None):
    """numpy arrays for a batch of Steps:

    tokens (S, L) int64, seg (S, L), mask (S, L) bool; hyp_of (S, L) int16: the hypothesis a token belongs to (-1 for none); n_hyps (S);
    overlap (S, G, 2) float32 (G = the largest global id + 1 over the batch's worlds: all steps share one world);
    T = the longest decision list (capped by max_dec); for each time t: `act[t]` the indices of the steps that have a t-th decision; name[t], kind[t], pos[t], want[t] for those; holes[t] (A, Hh) tokens;
    rows[t] (R, 4) the option rows of all of them, concatenated; counts[t] (A,) number of options of each."""
    S = len(steps); L = min(max_tokens, max(len(s.goal.tokens) for s in steps))
    tokens = np.zeros((S, L), dtype=np.int64); seg = np.zeros((S, L), dtype=np.int64); hyp_of = np.full((S, L), -1, dtype=np.int64)
    n_hyps = np.zeros(S, dtype=np.int64)
    for i, s in enumerate(steps):
        gf = s.goal; n = min(len(gf.tokens), L)
        tokens[i, :n] = gf.tokens[:n]; seg[i, :n] = gf.seg[:n]
        for h, (a, b) in enumerate(gf.spans):
            if a < L: hyp_of[i, a:min(b, L)] = h
        n_hyps[i] = gf.n_hyps
    mask = tokens != 0
    G_ = max(s.overlap.shape[0] for s in steps)
    overlap = np.zeros((S, G_, 2), dtype=np.float32)
    for i, s in enumerate(steps): overlap[i, :s.overlap.shape[0]] = s.overlap
    T = min(max_dec, max(s.n_decisions for s in steps))
    time_ = []
    for t in range(T):
        act = [i for i, s in enumerate(steps) if s.n_decisions > t]
        if not act: break
        rows = [steps[i].rows[t] for i in act]
        counts = np.asarray([len(r) for r in rows], dtype=np.int64)
        hh = max(1, max(len(steps[i].holes[t]) for i in act))
        holes = np.zeros((len(act), hh), dtype=np.int64)
        for j, i in enumerate(act):
            h = steps[i].holes[t]; holes[j, :len(h)] = h
        time_.append({'act': np.asarray(act, dtype=np.int64), 'name': np.asarray([steps[i].ch_name[t] for i in act], dtype=np.int64), 'kind': np.asarray([steps[i].ch_kind[t] for i in act], dtype=np.int64),
                      'pos': np.asarray([steps[i].ch_pos[t] for i in act], dtype=np.int64), 'want': np.asarray([steps[i].want[t] for i in act], dtype=np.int64), 'holes': holes,
                      'rows': np.concatenate(rows, axis=0), 'counts': counts})
    return {'tokens': tokens, 'seg': seg, 'mask': mask, 'hyp_of': hyp_of, 'n_hyps': n_hyps, 'overlap': overlap, 'time': time_, 'S': S, 'weights': None if weights is None else np.asarray(weights, dtype=np.float32),
            'remaining': np.asarray([s.remaining for s in steps], dtype=np.float32), 'action_kind': np.asarray([s.kind for s in steps], dtype=np.int64)}


def save_steps(steps, path):
    with open(path, 'wb') as f: pickle.dump(steps, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_steps(path):
    with open(path, 'rb') as f: return pickle.load(f)
