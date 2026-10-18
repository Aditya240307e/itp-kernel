"""ml_features.py -- from the engine's objects to integers a network can read.

A learner with no text interface sees a GOAL (hypothesis types and a target, kernel terms) and, at each decision of `gym_policy.Assembler`, a list of OPTIONS (rows `(otype, a, b, flags)`).  This module turns
both into arrays:

  * a goal is a TOKEN SEQUENCE: `[CLS] hyp_1 [SEP] hyp_2 [SEP] ... [TGT] target`, every term in prefix order with applications flattened (`APP_n head arg_1 .. arg_n`), constants by global id, context
    variables by level, bound variables by de Bruijn index, binders `PI` / `LAM`.  `GoalFeat` remembers where each hypothesis lies in the sequence (the network pools over those spans).
  * a global (a lemma, a constructor, a definition) is also described by the SYMBOLS of its type: the constants it mentions and those of its conclusion (`GlobalBags`); the learner can score a lemma it never
    saw in a proof by what the lemma talks about.  `overlap` is the fraction of those symbols that the goal mentions (a number the network gets as a feature).
  * an option row is the row itself; `choice_features` also gives the tokens of the HOLE (the type of the term being said).

Nothing here is trusted by the kernel: features only make good actions likely; every action is probed and every proof is kernel-checked.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_build as GB

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ the vocabulary
PAD, CLS, SEP, TGT, TRUNC, OTHER = range(6)
APP_BASE = 6                        # APP_BASE + min(n, 8) - 1 : an application with n arguments
PI, LAM, LET, MATCH, FIX = 14, 15, 16, 17, 18
TYPE_BASE = 19                      # universe levels 0..3
FVAR_BASE = 23                      # context variables, level 0..31
BVAR_BASE = 55                      # bound variables, de Bruijn index 0..15
CONST_BASE = 71
GMAX = 1100                         # global ids below this get their own token
VOCAB = CONST_BASE + GMAX
SEG_HYP, SEG_TGT, SEG_SPECIAL = 0, 1, 2

MAX_HYP_TOKENS = 64
MAX_TGT_TOKENS = 200
MAX_HYPS = 24
MAX_HOLE_TOKENS = 24


def tok_const(gid): return CONST_BASE + gid if gid < GMAX else OTHER


class Serializer:
    """terms of one kernel as token lists (memoised per node)"""
    def __init__(self, k):
        self.k = k; self.memo = {}
    def tokens(self, node, limit):
        out = []
        self._go(node, out, limit)
        if len(out) > limit: out = out[:limit - 1] + [TRUNC]
        return out
    def _go(self, node, out, limit):
        stack = [node]; k = self.k; memo = self.memo
        while stack and len(out) <= limit:
            n = stack.pop()
            if isinstance(n, int):                     # a token to emit (pushed in order)
                out.append(n); continue
            t = memo.get(n)
            if t is not None and len(t) + len(out) <= limit + 8:
                out.extend(t); continue
            v = k.view(n); kind = v[0]
            if kind == 'app':
                head, args = GB.spine(k, n)
                out.append(APP_BASE + min(len(args), 8) - 1)
                for a in reversed(args): stack.append(a)
                stack.append(head)
            elif kind == 'const': out.append(tok_const(v[1]))
            elif kind == 'fvar': out.append(FVAR_BASE + min(v[1], 31))
            elif kind == 'bvar': out.append(BVAR_BASE + min(v[1], 15))
            elif kind in ('pi', 'lam'):
                out.append(PI if kind == 'pi' else LAM)
                stack.append(v[2]); stack.append(v[1])
            elif kind == 'type': out.append(TYPE_BASE + min(v[1], 3))
            elif kind == 'let': out.append(LET); stack.append(v[3]); stack.append(v[2]); stack.append(v[1])
            elif kind == 'match': out.append(MATCH)
            elif kind == 'fix': out.append(FIX)
            else: out.append(OTHER)
    def small(self, node, limit):
        """tokens of a term, cached when short"""
        t = self.memo.get(node)
        if t is None:
            t = self.tokens(node, limit)
            if len(t) < limit: self.memo[node] = t
        return t


class GoalFeat:
    """a goal as arrays: `tokens` (int16), `seg` (int8), `spans` (hypothesis i occupies tokens[spans[i][0]:spans[i][1]]), `tgt` (the span of the target), `n_hyps`, `symbols` (the set of constant tokens)"""
    __slots__ = ('tokens', 'seg', 'spans', 'tgt', 'n_hyps', 'symbols', 'key')
    def __init__(self, tokens, seg, spans, tgt, n_hyps, symbols, key=None):
        self.tokens = tokens; self.seg = seg; self.spans = spans; self.tgt = tgt; self.n_hyps = n_hyps; self.symbols = symbols; self.key = key
    def __len__(self): return len(self.tokens)


MAX_TOTAL = 512                     # the window of the network (positions): a goal is never longer


def goal_features(ser, g):
    """the GoalFeat of an engine goal (gym_env.GoalData). The latest hypotheses are the most relevant: when the hypotheses and the target together do not fit the window (`MAX_TOTAL` tokens, and at most
    MAX_HYPS hypotheses), the OLDEST hypotheses are left out, never the target; their positions stay: `spans` has an entry per hypothesis, empty for the ones left out."""
    n = g.n_hyps; first = max(0, n - MAX_HYPS)
    tgt = ser.small(g.target, MAX_TGT_TOKENS)
    hts = {i: ser.small(g.hyp_types[i], MAX_HYP_TOKENS) for i in range(first, n)}
    budget = MAX_TOTAL - (2 + len(tgt)); used = 0; keep = n                      # ([CLS] .. [TGT] target)
    for i in range(n - 1, first - 1, -1):
        c = len(hts[i]) + 1                                                      # (the tokens and a [SEP])
        if used + c > budget: break
        used += c; keep = i
    toks = [CLS]; seg = [SEG_SPECIAL]; spans = []
    for i in range(n):
        if i < keep: spans.append((len(toks), len(toks))); continue
        t = hts[i]
        s = len(toks); toks.extend(t); seg.extend([SEG_HYP] * len(t)); spans.append((s, len(toks)))
        toks.append(SEP); seg.append(SEG_SPECIAL)
    toks.append(TGT); seg.append(SEG_SPECIAL)
    s = len(toks); toks.extend(tgt); seg.extend([SEG_TGT] * len(tgt))
    syms = frozenset(x for x in toks if x >= CONST_BASE)
    return GoalFeat(np.asarray(toks, dtype=np.int16), np.asarray(seg, dtype=np.int8), spans, (s, len(toks)), n, syms, g.key if hasattr(g, 'key') else None)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ globals
class GlobalBags:
    """the symbols of every global's type: `all_` (the constants it mentions), `concl` (the constants of its conclusion), `arity` (its leading binders), for the globals of a kernel up to `limit`.
    Stored as index arrays for an embedding bag, and as a sparse 0/1 matrix to compute the overlap with a goal's symbols."""
    def __init__(self, k, ser=None, limit=None):
        ser = ser or Serializer(k)
        n = k.num_globals() if limit is None else min(limit, k.num_globals())
        self.n = n
        all_idx, all_off, con_idx, con_off = [], [0], [], [0]
        arity = np.zeros(n, dtype=np.int16)
        rows_all, cols_all = [], []
        sizes_all = np.zeros(n, dtype=np.float32)
        rows_con, cols_con = [], []
        sizes_con = np.zeros(n, dtype=np.float32)
        for gid in range(n):
            ty = k.global_info(gid)[1]
            doms, concl = GB.open_pis(k, ty)
            arity[gid] = min(len(doms), 15)
            allt = sorted({x for x in ser.tokens(ty, 160) if x >= CONST_BASE})[:48]
            cont = sorted({x for x in ser.tokens(concl, 96) if x >= CONST_BASE})[:24]
            all_idx.extend(allt); all_off.append(len(all_idx)); con_idx.extend(cont); con_off.append(len(con_idx))
            sizes_all[gid] = max(len(allt), 1); sizes_con[gid] = max(len(cont), 1)
            rows_all.extend([gid] * len(allt)); cols_all.extend(allt)
            rows_con.extend([gid] * len(cont)); cols_con.extend(cont)
        self.all_idx = np.asarray(all_idx, dtype=np.int64); self.all_off = np.asarray(all_off, dtype=np.int64)
        self.con_idx = np.asarray(con_idx, dtype=np.int64); self.con_off = np.asarray(con_off, dtype=np.int64)
        self.arity = arity
        import scipy.sparse as sp
        self.M_all = sp.csr_matrix((np.ones(len(rows_all), dtype=np.float32), (rows_all, cols_all)), shape=(n, VOCAB))
        self.M_con = sp.csr_matrix((np.ones(len(rows_con), dtype=np.float32), (rows_con, cols_con)), shape=(n, VOCAB))
        self.sizes_all = sizes_all; self.sizes_con = sizes_con
    def overlap(self, symbols):
        """(n, 2) float32: for every global, the fraction of its symbols (all, conclusion) that the goal's symbol set contains"""
        v = np.zeros(VOCAB, dtype=np.float32)
        if symbols: v[np.fromiter(symbols, dtype=np.int64)] = 1.0
        return np.stack([(self.M_all @ v) / self.sizes_all, (self.M_con @ v) / self.sizes_con], axis=1).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ choices
CHOICE_NAMES = ('kind', 'cand', 'hyp', 'source', 'token', 'more', 'slot', 'gen', 'dir', 'lin', 'range')
CHOICE_ID = {n: i for i, n in enumerate(CHOICE_NAMES)}


def rows_array(options):
    return np.asarray(options, dtype=np.int64).reshape(-1, 4)


def hole_tokens(ser, hole):
    if hole is None: return np.zeros(0, dtype=np.int16)
    try: return np.asarray(ser.small(hole, MAX_HOLE_TOKENS), dtype=np.int16)
    except Exception: return np.zeros(0, dtype=np.int16)
