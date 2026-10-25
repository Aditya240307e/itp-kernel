"""ml_model.py -- the policy network: a goal (token sequence) and a decision (the options the assembler offers) in, a score for every option out.

    net = PolicyNet(bags)                     # bags: ml_features.GlobalBags of the world (index arrays of every global's symbols)
    out = net(batch)                          # batch: ml_data.collate(..); out['loss'], out['logits'] per time step, out['value']

THE ENCODER.  A transformer over the goal's tokens (hypotheses, target) gives a vector for the whole goal (the `[CLS]` position) and one for every hypothesis (the mean over its tokens).
THE DECODER.  An action is assembled by a sequence of decisions (gym_policy): a recurrent cell carries what has been decided so far (the kind of action, the lemma, the tokens of a term said so far).  At each
decision the network forms a query from the goal, the recurrent state, the decision's name, the kind of action and the expected type of the term being said (the HOLE: the mean of the embeddings of its tokens),
and scores each OPTION by a dot product with the option's embedding.
THE OPTIONS.  Rows `(otype, a, b, flags)`:  a global (lemma, constructor, definition) by its symbol embedding, the symbols of its type and the overlap of those with the goal; a hypothesis or variable by the
embedding of its type in the goal; a number by its value; a kind of action; a structural token; a range of values for interval cases; stop / yes / direction by category.  Nothing about the options is
learned per problem: a lemma that was never used in a training proof is scored by what it talks about.
A VALUE HEAD predicts log(1 + number of actions still needed) from the goal (trained on demonstrations; the search may use it).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import os, sys, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
import ml_features as F
import gym_policy as P

N_KIND = 40
N_NAMES = len(F.CHOICE_NAMES)
N_OT = P.N_OT
NUM_BUCKETS = 65 + 82


def num_index(a):
    """the embedding index of an integer: -32..32 each their own, the others by sign and bit length"""
    a = int(a)
    if -32 <= a <= 32: return a + 32
    return 65 + min(abs(a).bit_length(), 40) + (41 if a < 0 else 0)


class PolicyNet(nn.Module):
    def __init__(self, bags, d=128, layers=3, heads=4, ff=384, dropout=0.1, max_len=512):
        super().__init__()
        self.d = d
        # (what the world says about its globals -- the symbols of their types -- is not learned: these buffers are not part of a checkpoint, `set_globals` puts the current world's in, so that one set of weights
        # serves the library alone, the whole bank and a library that has grown)
        for name in ('all_idx', 'all_off', 'con_idx', 'con_off', 'arity'): self.register_buffer(name, torch.zeros(0, dtype=torch.long), persistent=False)
        self.n_globals = 0
        self.set_globals(bags)
        self.tok = nn.Embedding(F.VOCAB, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.segE = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        # globals
        self.bag_all = nn.EmbeddingBag(F.VOCAB, d, mode='mean'); self.bag_con = nn.EmbeddingBag(F.VOCAB, d, mode='mean')
        self.arity_emb = nn.Embedding(16, d)
        self.glob = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d))
        self.ov = nn.Linear(2, d)
        # options
        self.ot_emb = nn.Embedding(N_OT, d)
        self.kind_emb = nn.Embedding(N_KIND, d); self.mode_emb = nn.Embedding(4, d)
        self.num_emb = nn.Embedding(NUM_BUCKETS, d)
        self.small_emb = nn.Embedding(N_OT * 16, d)
        self.flag_emb = nn.Embedding(8, d)
        self.local_emb = nn.Embedding(16, d)
        self.hyp_proj = nn.Linear(d, d)
        self.b_emb = nn.Embedding(16, d)
        self.opt_ln = nn.LayerNorm(d)
        self.unary = nn.Linear(d, 1)
        # decoder
        self.name_emb = nn.Embedding(N_NAMES, d); self.ckind_emb = nn.Embedding(N_KIND + 1, d); self.cpos_emb = nn.Embedding(32, d)
        self.h0 = nn.Linear(d, d)
        self.gru = nn.GRUCell(2 * d, d)
        self.query = nn.Sequential(nn.Linear(5 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.drop = nn.Dropout(dropout)
        # heads of the reinforcement learners (created last, so that every parameter above keeps the initial value it has had): the critic V(goal, budget left, open goals), the manager's P(the next action is a
        # CLAIM) and the size head, log(1 + the actions the goal's subtree needs)
        self.critic = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, 1))
        self.mgr = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, 1))
        self.size = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ encoder
    def encode(self, tokens, seg, mask, hyp_of, n_hyps):
        """cls (S, d); hyp (S, H, d) the mean over each hypothesis' tokens"""
        S, L = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(L, device=tokens.device))[None] + self.segE(seg)
        x = self.drop(x)
        x = self.enc(x, src_key_padding_mask=~mask)
        x = self.ln(x)
        cls = x[:, 0]
        H = int(n_hyps.max().item()) if S else 0
        H = max(H, 1)
        idx = hyp_of.clamp(min=0) + torch.arange(S, device=tokens.device)[:, None] * H            # (S, L)
        valid = (hyp_of >= 0).float()
        flat = torch.zeros(S * H, self.d, device=tokens.device).index_add_(0, idx.reshape(-1), (x * valid[..., None]).reshape(-1, self.d))
        cnt = torch.zeros(S * H, device=tokens.device).index_add_(0, idx.reshape(-1), valid.reshape(-1)).clamp(min=1.0)
        hyp = (flat / cnt[:, None]).reshape(S, H, self.d)
        return cls, hyp

    def set_globals(self, bags):
        """take the symbols of the globals of a world (ml_features.GlobalBags) -- also of a world whose library has grown since the network was made"""
        dev = self.all_idx.device
        for name, arr in (('all_idx', bags.all_idx), ('all_off', bags.all_off[:-1]), ('con_idx', bags.con_idx), ('con_off', bags.con_off[:-1]), ('arity', bags.arity.astype(np.int64))):
            setattr(self, name, torch.as_tensor(np.asarray(arr, dtype=np.int64)).to(dev))
        self.n_globals = bags.n

    def global_table(self, sym_tok=None):
        """(G, d): an embedding for every global: its own symbol, the symbols of its type, of its conclusion, its arity"""
        G_ = self.n_globals
        ids = torch.arange(G_, device=self.tok.weight.device)
        own = self.tok(F.CONST_BASE + ids.clamp(max=F.GMAX - 1))
        a = self.bag_all(self.all_idx, self.all_off); c = self.bag_con(self.con_idx, self.con_off)
        return self.glob(torch.cat([own, a, c, self.arity_emb(self.arity)], dim=-1))

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ options
    def option_embeddings(self, rows, owner, gtab, hyp, overlap, n_hyps):
        """rows (R, 4) int64 option rows; owner (R,) the step index each belongs to; -> (R, d)"""
        d = self.d; dev = rows.device
        ot = rows[:, 0]; a = rows[:, 1]; b = rows[:, 2]; fl = rows[:, 3]
        out = self.ot_emb(ot)
        # global: otype OT_GLOBAL
        m = ot == P.OT_GLOBAL
        if m.any():
            gid = a[m].clamp(0, self.n_globals - 1)
            e = gtab[gid] + self.ov(overlap[owner[m], gid]) + self.flag_emb((fl[m] & 7)) + self.b_emb(b[m].clamp(0, 15))
            out = out.index_put((m.nonzero().squeeze(-1),), out[m] + e, accumulate=False)
        # hypothesis / variable
        for otv in (P.OT_HYP, P.OT_VAR, P.OT_RANGE):
            m = ot == otv
            if not m.any(): continue
            h = a[m]; o = owner[m]; nh = n_hyps[o]
            inside = h < nh
            hv = hyp[o, h.clamp(0, hyp.shape[1] - 1)]
            e = self.hyp_proj(hv) * inside[:, None].float() + self.local_emb((h - nh).clamp(0, 15)) * (~inside)[:, None].float()
            if otv == P.OT_VAR: e = e + self.flag_emb((fl[m] & 7)) + self.b_emb(b[m].clamp(0, 15))
            if otv == P.OT_RANGE: e = e + self.num_emb(self._num_idx(b[m])) + self.num_emb(self._num_idx(fl[m]))
            out = out.index_put((m.nonzero().squeeze(-1),), out[m] + e, accumulate=False)
        m = ot == P.OT_NUM
        if m.any():
            out = out.index_put((m.nonzero().squeeze(-1),), out[m] + self.num_emb(self._num_idx(a[m])), accumulate=False)
        m = ot == P.OT_KIND
        if m.any():
            out = out.index_put((m.nonzero().squeeze(-1),), out[m] + self.kind_emb(a[m].clamp(0, N_KIND - 1)) + self.mode_emb(b[m].clamp(0, 3)), accumulate=False)
        for otv in (P.OT_STRUCT, P.OT_STOP, P.OT_SLOT, P.OT_DIR, P.OT_YES):
            m = ot == otv
            if m.any(): out = out.index_put((m.nonzero().squeeze(-1),), out[m] + self.small_emb(otv * 16 + a[m].clamp(0, 15)), accumulate=False)
        return self.opt_ln(out)

    @staticmethod
    def _num_idx(a):
        a = a.clone()
        small = (a >= -32) & (a <= 32)
        mag = a.abs().clamp(min=1).float()
        bl = (torch.log2(mag).floor().long() + 1).clamp(max=40)
        big = 65 + bl + (a < 0).long() * 41
        return torch.where(small, a + 32, big)

    @torch.no_grad()
    def goal_heads(self, batch):
        """the heads that look at the goal alone, for the Steps of a batch (ml_data.collate; no decision is needed): (critic V, manager logit, size estimate), each a float array with one entry per step"""
        dev = self.tok.weight.device
        T = lambda x: torch.as_tensor(x).to(dev)
        cls, _ = self.encode(T(batch['tokens']), T(batch['seg']), T(batch['mask']), T(batch['hyp_of']), T(batch['n_hyps']))
        cf = T(batch['cfeat']); x = torch.cat([cls, cf], -1)
        return self.critic(x).squeeze(-1).cpu().numpy(), self.mgr(x).squeeze(-1).cpu().numpy(), torch.expm1(self.size(cls).squeeze(-1)).clamp(min=0).cpu().numpy()

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ forward
    def forward(self, batch, train=True, want_logits=False, ppo=None, aux=None):
        """the loss of a batch of Steps (ml_data.collate).  By default behaviour cloning: `weights` (if the batch has them) multiply the log-likelihood of each step's decisions (0 / negative weights: policy
        gradient).  `ppo={'clip': 0.2}`: the clipped surrogate of PPO instead, the weights being the advantages of the steps and the batch carrying `old_logp`.  `aux={'size': c, 'mgr': c, 'critic': c}` adds
        the size head, the manager head and the critic to 'loss' (the parts are always in the result).  A step's `kmask` forbids the action kinds outside it at the decision of the kind."""
        aux = aux or {}
        dev = self.tok.weight.device
        T = lambda x: torch.as_tensor(x).to(dev)
        tokens = T(batch['tokens']); seg = T(batch['seg']); mask = T(batch['mask']); hyp_of = T(batch['hyp_of']); n_hyps = T(batch['n_hyps']); overlap = T(batch['overlap'])
        cls, hyp = self.encode(tokens, seg, mask, hyp_of, n_hyps)
        gtab = self.global_table()
        S = cls.shape[0]
        wts = None if batch.get('weights') is None else T(batch['weights'])
        kmask = T(batch['kmask']) if batch.get('has_kmask') else None
        if ppo is not None and (wts is None or not batch.get('has_old')): raise ValueError('ppo needs the advantages (weights) and the old log-probabilities of the decisions (Step.old_logp)')
        ent_sum = 0.0; clipped = 0; kl_sum = 0.0
        h = torch.tanh(self.h0(cls))
        prev_name = torch.zeros(S, dtype=torch.long, device=dev); prev_opt = torch.zeros(S, self.d, device=dev); started = torch.zeros(S, dtype=torch.bool, device=dev)
        loss_sum = 0.0; n_dec = 0; correct = 0; per_name = {}; logits_out = []
        all_ok = torch.ones(S, dtype=torch.bool, device=dev)
        for step in batch['time']:
            act = T(step['act']); A_ = len(act)
            name = T(step['name']); kind = T(step['kind']); pos = T(step['pos']); want = T(step['want']); holes = T(step['holes'])
            rows = T(step['rows']); counts = T(step['counts'])
            # recurrent state: update with the previous decision of the actions that have one
            hs = h[act]
            has_prev = started[act]
            inp = torch.cat([self.name_emb(prev_name[act]), prev_opt[act]], dim=-1)
            hn = self.gru(inp, hs)
            hs = torch.where(has_prev[:, None], hn, hs)
            hole_e = self.tok(holes); hm = (holes != 0).float()
            hole_v = (hole_e * hm[..., None]).sum(1) / hm.sum(1, keepdim=True).clamp(min=1.0)
            q = self.query(torch.cat([cls[act], hs, self.name_emb(name), self.ckind_emb((kind + 1).clamp(min=0)) + self.cpos_emb(pos), hole_v], dim=-1))
            owner = torch.repeat_interleave(act, counts)
            oe = self.option_embeddings(rows, owner, gtab, hyp, overlap, n_hyps)
            row_q = torch.repeat_interleave(q, counts, dim=0)
            score = (row_q * oe).sum(-1) / math.sqrt(self.d) + self.unary(oe).squeeze(-1)
            if kmask is not None:                                          # the action kinds this step may not choose get no probability
                dec_of_row = torch.repeat_interleave(torch.arange(A_, device=dev), counts)
                km = kmask[owner]
                forbidden = (name[dec_of_row] == F.CHOICE_ID['kind']) & (rows[:, 0] == P.OT_KIND) & (km != -1) & (((km >> rows[:, 1].clamp(0, 62)) & 1) == 0)
                score = score.masked_fill(forbidden, -1e9)
            # per-decision softmax over its options
            maxn = int(counts.max().item())
            starts = torch.cumsum(counts, 0) - counts
            ar = torch.arange(maxn, device=dev)[None, :]
            valid = ar < counts[:, None]
            gidx = (starts[:, None] + ar).clamp(max=len(score) - 1)
            logits = torch.where(valid, score[gidx], torch.full((A_, maxn), -1e9, device=dev))
            logp = torch.log_softmax(logits, dim=-1)
            train_mask = (counts > 1) & (want >= 0)
            if train_mask.any():
                lp = logp[torch.arange(A_, device=dev), want.clamp(min=0)]
                if ppo is not None:
                    old_lp = T(step['old_logp']); ratio = torch.exp(lp - old_lp); adv = wts[act]; tm = train_mask.float()
                    surr = torch.min(ratio * adv, ratio.clamp(1.0 - ppo['clip'], 1.0 + ppo['clip']) * adv)
                    loss_sum = loss_sum - (surr * tm).sum()
                    ent_sum = ent_sum + ((-(torch.exp(logp) * logp.clamp(min=-1e4)).sum(-1)) * tm).sum()
                    clipped += int((((ratio - 1.0).abs() > ppo['clip']) & train_mask).sum().item()); kl_sum = kl_sum + ((old_lp - lp) * tm).sum().item()
                else:
                    wt = train_mask.float() if wts is None else train_mask.float() * wts[act]
                    loss_sum = loss_sum - (lp * wt).sum()
                n_dec += int(train_mask.sum().item())
                pred = logits.argmax(-1)
                ok = (pred == want) & train_mask
                correct += int(ok.sum().item())
                bad = train_mask & ~ok
                if bad.any(): all_ok = all_ok.index_put((act[bad],), torch.zeros(int(bad.sum().item()), dtype=torch.bool, device=dev))
                for nm in torch.unique(name[train_mask]).tolist():
                    mk = train_mask & (name == nm)
                    c, t_ = per_name.get(nm, (0, 0)); per_name[nm] = (c + int((ok & mk).sum().item()), t_ + int(mk.sum().item()))
            if want_logits: logits_out.append((act, logp))
            # remember the decision for the next step: the name and the chosen option's embedding
            chosen = starts + want.clamp(min=0)
            prev_name = prev_name.index_put((act,), name)
            prev_opt = prev_opt.index_put((act,), oe[chosen.clamp(max=len(oe) - 1)])
            started = started.index_put((act,), torch.ones_like(act, dtype=torch.bool))
            h = h.index_put((act,), hs)
        value = self.value(cls).squeeze(-1)
        rem = T(batch['remaining']); vmask = rem >= 0
        vloss = Fn.smooth_l1_loss(value[vmask], torch.log1p(rem[vmask])) if vmask.any() else torch.zeros((), device=dev)
        pol = loss_sum / max(n_dec, 1) if n_dec else torch.zeros((), device=dev)
        zero = torch.zeros((), device=dev); loss = pol + 0.2 * vloss
        cf = T(batch['cfeat']) if batch.get('cfeat') is not None else torch.cat([torch.ones(S, 1), torch.zeros(S, 1)], -1).to(dev)
        size_loss = mgr_loss = critic_loss = zero; critic_v = None
        sub = T(batch['sub']) if batch.get('sub') is not None else None
        if sub is not None and (sub > 0).any():
            sm = sub > 0; size_loss = Fn.smooth_l1_loss(self.size(cls).squeeze(-1)[sm], torch.log1p(sub[sm]))
        mg = T(batch['mgr']) if batch.get('mgr') is not None else None
        if mg is not None and (mg >= 0).any():
            mm = mg >= 0; mgr_loss = Fn.binary_cross_entropy_with_logits(self.mgr(torch.cat([cls, cf], -1)).squeeze(-1)[mm], mg[mm].float())
        rt = T(batch['ret']) if batch.get('ret') is not None else None
        if rt is not None and (~torch.isnan(rt)).any():
            rm = ~torch.isnan(rt); critic_v = self.critic(torch.cat([cls, cf], -1)).squeeze(-1); critic_loss = Fn.mse_loss(critic_v[rm], rt[rm])
        loss = loss + aux.get('size', 0.0) * size_loss + aux.get('mgr', 0.0) * mgr_loss + aux.get('critic', 0.0) * critic_loss
        res = {'loss': loss, 'policy_loss': pol, 'value_loss': vloss, 'size_loss': size_loss, 'mgr_loss': mgr_loss, 'critic_loss': critic_loss, 'n': n_dec, 'correct': correct, 'per_name': per_name,
               'logits': logits_out, 'action_ok': all_ok, 'value': value.detach()}
        if ppo is not None:
            nd = max(n_dec, 1); res.update({'entropy': (ent_sum / nd) if n_dec else zero, 'clip_fraction': clipped / nd, 'approx_kl': kl_sum / nd})
        return res


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ inference
class Carry:
    """what the decoder remembers of the decisions made so far in one action: the recurrent state before the previous decision was consumed, and that decision (its name and the embedding of the option chosen)"""
    __slots__ = ('h', 'name', 'opt')
    def __init__(self, h, name=None, opt=None): self.h = h; self.name = name; self.opt = opt


class Decoder:
    """the network used on ONE goal, one decision at a time, without gradients (search, evaluation, rollouts).  `scores(carry, choice)` -> (log-probabilities over the choice's options, the option embeddings, the
    new carry's recurrent state); `child(carry, choice, i)` the carry after option i was chosen."""
    def __init__(self, net, ctx, g, gtab=None, gf=None, ov=None):
        import ml_data as D
        self.net = net; self.ctx = ctx
        if gf is None: gf = F.goal_features(ctx.ser, g)
        if ov is None: ov = ctx.bags.overlap(gf.symbols)
        step = D.Step('', 0, 1, gf, ov.astype(np.float16), 0, np.zeros(0, np.int8), np.zeros(0, np.int16), np.zeros(0, np.int8), [], [], np.zeros(0, np.int32))
        batch = D.collate([step])
        with torch.no_grad():
            T = torch.as_tensor
            self.cls, self.hyp = net.encode(T(batch['tokens']), T(batch['seg']), T(batch['mask']), T(batch['hyp_of']), T(batch['n_hyps']))
            self.gtab = gtab if gtab is not None else net.global_table()
            self.overlap = T(batch['overlap']); self.n_hyps = T(batch['n_hyps'])
            self.h0 = torch.tanh(net.h0(self.cls))
        self.goal = gf; self.ov = ov
    def root(self): return Carry(self.h0)
    def critic(self, budget_left=1.0, goals_left=1):
        """V(goal, budget left as a fraction, open goals): the critic's estimate of the return"""
        with torch.no_grad():
            cf = torch.tensor([[float(budget_left), min(int(goals_left), 16) / 8.0]])
            return float(self.net.critic(torch.cat([self.cls, cf], -1)).squeeze())
    def manager_logit(self, budget_left=1.0, goals_left=1):
        """the manager head's logit for "the next action is a CLAIM"""
        with torch.no_grad():
            cf = torch.tensor([[float(budget_left), min(int(goals_left), 16) / 8.0]])
            return float(self.net.mgr(torch.cat([self.cls, cf], -1)).squeeze())
    def size_estimate(self):
        """the size head: the number of actions the goal is expected to need, expm1 of the prediction (>= 0)"""
        with torch.no_grad():
            return max(0.0, float(torch.expm1(self.net.size(self.cls).squeeze())))
    def scores(self, carry, choice):
        net = self.net
        with torch.no_grad():
            hs = carry.h if carry.name is None else net.gru(torch.cat([net.name_emb(torch.tensor([carry.name])), carry.opt[None]], dim=-1), carry.h)
            holes = torch.as_tensor(F.hole_tokens(self.ctx.ser, choice.hole).astype(np.int64))[None]
            if holes.shape[1] == 0: holes = torch.zeros(1, 1, dtype=torch.long)
            hole_e = net.tok(holes); hm = (holes != 0).float()
            hole_v = (hole_e * hm[..., None]).sum(1) / hm.sum(1, keepdim=True).clamp(min=1.0)
            name = F.CHOICE_ID[choice.name]; kind = -1 if choice.kind is None else int(choice.kind)
            q = net.query(torch.cat([self.cls, hs, net.name_emb(torch.tensor([name])), net.ckind_emb(torch.tensor([max(kind + 1, 0)])) + net.cpos_emb(torch.tensor([min(choice.pos, 31)])), hole_v], dim=-1))
            rows = torch.as_tensor(F.rows_array(choice.options))
            owner = torch.zeros(len(rows), dtype=torch.long)
            oe = net.option_embeddings(rows, owner, self.gtab, self.hyp, self.overlap, self.n_hyps)
            score = (q * oe).sum(-1) / math.sqrt(net.d) + net.unary(oe).squeeze(-1)
            logp = torch.log_softmax(score, dim=-1)
        return logp, oe, hs
    def child(self, carry, choice, i, oe, hs):
        return Carry(hs, F.CHOICE_ID[choice.name], oe[i])
