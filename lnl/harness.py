"""Differential harness: reference kernel (absolute levels) vs hybrid kernel (bound = index, context = level)."""
import re, sys, time, random
import os
LNL_DIR = os.environ.get('LNL_DIR', '/home/claude/itp_py/lnl')   # which build of the hybrid kernel to test (mutation runs use copies)
sys.path[:0] = ['/home/claude/itp_ref', LNL_DIR, '/home/claude/itp_py']
import itp_env_ref as R
import itp_env_lnl as H
sys.modules['itp_env'] = R            # curriculum builds its kernels through `itp_env`: alias it to the REFERENCE module
import curriculum as cur

# ------------------------------------------------------------------ trees
# level tree (reference):  ('var',l) ('app',f,a) ('lam',T,B) ('pi',D,C) ('type',l) ('const',gid) ('match',S,M,(b..)) ('fix',dec,fn_type,body)
# hybrid tree (new):       ('bvar',i) ('fvar',l) + the same others; ('let',T,V,B); Fix.fn_type closed, body under the self binder
RAW_REF = re.compile(r'\s*(\(fix<\d+>|\(lam|\(pi|\(match|#\d+|T\d+|C\d+|\?\d+|<other>|\(|\)|\[|\]|\||:=|\.)')
RAW_NEW = re.compile(r'\s*(\(fix<\d+>|\(lam|\(pi|\(match|\(let|\bin\b|b\d+|f\d+|T\d+|C\d+|\?\d+|<other>|\(|\)|\[|\]|\||:=|\.)')

def parse_raw(s, new):
    toks = [m.group(1) for m in (RAW_NEW if new else RAW_REF).finditer(s)]; pos = [0]
    def eat(): t = toks[pos[0]]; pos[0] += 1; return t
    def peek(): return toks[pos[0]]
    def term():
        t = eat()
        if t[0] == '#': return ('var', int(t[1:]))
        if new and t[0] == 'b' and t[1:].isdigit(): return ('bvar', int(t[1:]))
        if new and t[0] == 'f' and t[1:].isdigit(): return ('fvar', int(t[1:]))
        if t[0] == 'T': return ('type', int(t[1:]))
        if t[0] == 'C': return ('const', int(t[1:]))
        if t[0] == '?': return ('meta', int(t[1:]))
        if t == '<other>': return ('other',)
        if t in ('(lam', '(pi'):
            a = term(); assert eat() == '.'; b = term(); assert eat() == ')'; return (t[1:], a, b)
        if t == '(let':
            ty = term(); assert eat() == ':='; v = term(); assert eat() == 'in'; b = term(); assert eat() == ')'; return ('let', ty, v, b)
        if t.startswith('(fix<'):
            d = int(t[5:-1]); ty = term(); assert eat() == ':='; b = term(); assert eat() == ')'; return ('fix', d, ty, b)
        if t == '(match':
            s_ = term(); assert eat() == '['; m = term(); assert eat() == ']'; br = []
            while peek() == '|': eat(); br.append(term())
            assert eat() == ')'; return ('match', s_, m, tuple(br))
        if t == '(':
            f = term(); a = term(); assert eat() == ')'; return ('app', f, a)
        raise ValueError(t)
    r = term(); assert pos[0] == len(toks), (s[:80], pos[0], len(toks)); return r

def l2h(t, n0, d=None, ph=(), lenient=False):
    """level tree -> hybrid tree. n0 = context size; d = ref depth; ph = ref slots that do not exist in hybrid numbering
    (the self slot of an enclosing Fix, while converting that Fix's fn_type: the old convention anchors fn_type one above)."""
    if d is None: d = n0
    k = t[0]
    if k == 'var':
        L = t[1]
        assert L not in ph, "reference to a phantom slot"
        hl = L - sum(1 for p in ph if p < L); dh = d - len(ph)
        if hl < n0: return ('fvar', hl)
        if hl >= dh and lenient: return ('fvar', hl)
        assert hl < dh, ("dangling level", L, d, ph)
        return ('bvar', dh - 1 - hl)
    if k in ('type', 'const', 'meta', 'other'): return t
    if k == 'app': return ('app', l2h(t[1], n0, d, ph, lenient), l2h(t[2], n0, d, ph, lenient))
    if k in ('lam', 'pi'): return (k, l2h(t[1], n0, d, ph, lenient), l2h(t[2], n0, d + 1, ph, lenient))
    if k == 'match': return ('match', l2h(t[1], n0, d, ph, lenient), l2h(t[2], n0, d, ph, lenient), tuple(l2h(b, n0, d, ph, lenient) for b in t[3]))
    if k == 'fix': return ('fix', t[1], l2h(t[2], n0, d + 1, ph + (d,), lenient), l2h(t[3], n0, d + 1, ph, lenient))
    raise ValueError(k)

def build_new(k, t):
    g = t[0]
    if g == 'bvar': return k.mk_bvar(t[1])
    if g == 'fvar': return k.mk_fvar(t[1])
    if g == 'type': return k.mk_type(t[1])
    if g == 'const': return k.mk_const(t[1])
    if g == 'app': return k.mk_app(build_new(k, t[1]), build_new(k, t[2]))
    if g == 'lam': return k.mk_lambda(build_new(k, t[1]), build_new(k, t[2]))
    if g == 'pi': return k.mk_pi(build_new(k, t[1]), build_new(k, t[2]))
    if g == 'let': return k.mk_let(build_new(k, t[1]), build_new(k, t[2]), build_new(k, t[3]))
    if g == 'match': return k.mk_match(build_new(k, t[1]), build_new(k, t[2]), [build_new(k, b) for b in t[3]])
    if g == 'fix': return k.mk_fix(build_new(k, t[2]), t[1], build_new(k, t[3]))
    if g == 'meta': return k.mk_meta(t[1])
    raise ValueError(g)

def build_ref(k, t):
    g = t[0]
    if g == 'var': return k.mk_var(t[1])
    if g == 'type': return k.mk_type(t[1])
    if g == 'const': return k.mk_const(t[1])
    if g == 'app': return k.mk_app(build_ref(k, t[1]), build_ref(k, t[2]))
    if g == 'lam': return k.mk_lambda(build_ref(k, t[1]), build_ref(k, t[2]))
    if g == 'pi': return k.mk_pi(build_ref(k, t[1]), build_ref(k, t[2]))
    if g == 'match': return k.mk_match(build_ref(k, t[1]), build_ref(k, t[2]), [build_ref(k, b) for b in t[3]])
    if g == 'fix': return k.mk_fix(build_ref(k, t[2]), t[1], build_ref(k, t[3]))
    raise ValueError(g)

def dict_to_tree(d, gid_of):
    t = d['t']
    if t == 'var': return ('var', d['level'])
    if t == 'app': return ('app', dict_to_tree(d['fn'], gid_of), dict_to_tree(d['arg'], gid_of))
    if t == 'lambda': return ('lam', dict_to_tree(d['type'], gid_of), dict_to_tree(d['body'], gid_of))
    if t == 'pi': return ('pi', dict_to_tree(d['dom'], gid_of), dict_to_tree(d['cod'], gid_of))
    if t == 'type': return ('type', d['level'])
    if t == 'const': return ('const', gid_of[d['name']])
    if t == 'match': return ('match', dict_to_tree(d['scrutinee'], gid_of), dict_to_tree(d['motive'], gid_of), tuple(dict_to_tree(b, gid_of) for b in d['branches']))
    if t == 'fix': return ('fix', d['decreasing_arg'], dict_to_tree(d['fn_type'], gid_of), dict_to_tree(d['body'], gid_of))
    raise ValueError("unsupported node in library export: %s" % t)

def load_library(kref, knew):
    """Replays the reference kernel's library into the new kernel (unchecked), converting every term."""
    gid_of = {name: gid for gid, name, _, _ in kref.list_globals()}
    events = kref.export_library(); kinds = {}
    for e in events:
        kind = e['kind']; kinds[kind] = kinds.get(kind, 0) + 1
        H_ = lambda d, n0: build_new(knew, l2h(dict_to_tree(d, gid_of), n0))
        if kind == 'family':
            tid = knew.reserve_inductive_type(e['name'], H_(e['sort'], 0)); specs = []
            for c in e['ctors']:
                ats = [H_(a, j) for j, a in enumerate(c['arg_types'])]
                ris = [H_(r, len(c['arg_types'])) for r in c['result_indices']]
                specs.append(H.CtorSpec(c['name'], ats, ris))
            knew.finish_inductive(tid, specs)
        elif kind == 'recursive_function':
            gid = knew.reserve_recursive_function(e['name'], H_(e['type'], 0))
            ind = gid_of[e['ind_type_name']]
            knew.finish_recursive_function(gid, ind, e['decreasing_arg_level'], H_(e['body'], 0))
        elif kind == 'definition':
            knew.declare_unchecked(H_(e['type'], 0), H_(e['value'], 0), e['name'])
        elif kind == 'axiom':
            knew.declare_unchecked(H_(e['type'], 0), None, e['name'])
        else: raise ValueError(kind)
    return kinds, events

# ------------------------------------------------------------------ typed random terms (level trees, ctx index == level)
class Gen:
    def __init__(self, g, rng, zero_first=True):
        self.g, self.rng, self.zf = g, rng, zero_first
        self.allow_match = True; self.allow_fix = False   # the reference API has no mk_fix: Fix terms come from the proof corpus below
        self.NAT, self.BOOL = ('const', g['Nat']), ('const', g['Bool'])
        self.ZERO, self.SUCC, self.ADD = ('const', g['Zero']), ('const', g['Succ']), ('const', g['add'])
    def ty(self, T):
        if T == 'nat': return self.NAT
        if T == 'bool': return self.BOOL
        return ('pi', self.ty(T[1]), self.ty(T[2]))
    def rtype(self, d=1):
        r = self.rng.random()
        if d <= 0 or r < 0.45: return 'nat' if self.rng.random() < 0.7 else 'bool'
        return ('fn', self.rtype(d - 1), self.rtype(d - 1))
    def match_nat(self, scrut, motive_body, zero_b, succ_b):
        br = (zero_b, succ_b) if self.zf else (succ_b, zero_b)
        return ('match', scrut, ('lam', self.NAT, motive_body), br)
    def gen(self, T, ctx, d):
        r, rnd = self.rng, self.rng.random()
        vs = [i for i, t in enumerate(ctx) if t == T]
        if vs and (d <= 0 or rnd < 0.18): return ('var', r.choice(vs))
        if T == 'nat':
            if d <= 0: return self.ZERO
            c = r.randrange(6)
            if c == 4 and not self.allow_match: c = 0
            if c == 0: return ('app', self.SUCC, self.gen('nat', ctx, d - 1))
            if c == 1: return ('app', ('app', self.ADD, self.gen('nat', ctx, d - 1)), self.gen('nat', ctx, d - 1))
            if c == 2:
                A = r.choice(['nat', 'bool', ('fn', 'nat', 'nat')]); return ('app', ('lam', self.ty(A), self.gen('nat', ctx + [A], d - 1)), self.gen(A, ctx, d - 1))
            if c == 3:
                A = r.choice(['nat', 'bool']); return ('app', self.gen(('fn', A, 'nat'), ctx, d - 1), self.gen(A, ctx, d - 1))
            if c == 4: return self.match_nat(self.gen('nat', ctx, d - 1), self.NAT, self.gen('nat', ctx, d - 1), ('lam', self.NAT, self.gen('nat', ctx + ['nat'], d - 1)))
            if self.allow_fix and r.random() < 0.5: return self.fix_app(ctx, d)
            return self.ZERO
        if T == 'bool':
            if d <= 0: return ('const', self.g['True' if r.random() < 0.5 else 'False'])
            c = r.randrange(4)
            if c == 2 and not self.allow_match: c = 0
            if c == 0:
                A = r.choice(['nat', 'bool']); return ('app', self.gen(('fn', A, 'bool'), ctx, d - 1), self.gen(A, ctx, d - 1))
            if c == 1: return ('app', ('lam', self.NAT, self.gen('bool', ctx + ['nat'], d - 1)), self.gen('nat', ctx, d - 1))
            if c == 2: return self.match_nat(self.gen('nat', ctx, d - 1), self.BOOL, self.gen('bool', ctx, d - 1), ('lam', self.NAT, self.gen('bool', ctx + ['nat'], d - 1)))
            return ('const', self.g['True'])
        A, B = T[1], T[2]
        if d > 0 and r.random() < 0.15:
            C = r.choice(['nat', 'bool']); return ('app', self.gen(('fn', C, T), ctx, d - 1), self.gen(C, ctx, d - 1))
        if T == ('fn', 'nat', 'nat') and r.random() < 0.3: return self.SUCC
        if T == ('fn', 'nat', ('fn', 'nat', 'nat')) and r.random() < 0.4: return self.ADD
        return ('lam', self.ty(A), self.gen(B, ctx + [A], d - 1))
    def fix_app(self, ctx, d):
        """(fix f. fun n. match n with Zero -> z | Succ k -> s) applied to a Nat; reference levels: f = n0, n = n0+1, k = n0+2.
        z lives under (f, n) and s under (f, n, k): the generation context is padded with opaque placeholders so that binders
        created inside z / s get the right absolute levels (k is a usable Nat variable inside s)."""
        n = len(ctx); f, nn_, k = ('var', n), ('var', n + 1), ('var', n + 2)
        dd = max(d - 1, 0); r = self.rng.randrange(3); rec = ('app', f, k)
        zero_b = self.gen('nat', ctx + ['_f', '_n'], dd)
        if r == 0: succ_b = ('app', self.SUCC, rec)
        elif r == 1: succ_b = ('app', ('app', self.ADD, rec), ('var', n + 2))
        else: succ_b = ('app', ('app', self.ADD, self.gen('nat', ctx + ['_f', '_n', 'nat'], dd)), rec)
        body = ('lam', self.NAT, self.match_nat(nn_, self.NAT, zero_b, ('lam', self.NAT, succ_b)))
        return ('app', ('fix', 0, ('pi', self.NAT, self.NAT), body), self.gen('nat', ctx, dd))
    def bad(self, ctx, d):
        c = self.rng.randrange(3)
        if c == 0: return ('app', self.gen(('fn', 'nat', 'nat'), ctx, d), self.gen('bool', ctx, d))           # argument of the wrong type
        if c == 1: return ('app', self.gen('nat', ctx, d), self.gen('nat', ctx, d))                           # applying a non-function
        return ('lam', self.NAT, ('app', ('var', len(ctx)), self.ZERO))                                      # applying a Nat variable

# ------------------------------------------------------------------ independent textbook implementations on hybrid trees
BODY = {'lam': (2,), 'pi': (2,), 'let': (3,), 'fix': (3,)}
def _map(t, f):
    k = t[0]
    if k == 'app': return ('app', f(t[1], 0), f(t[2], 0))
    if k in ('lam', 'pi'): return (k, f(t[1], 0), f(t[2], 1))
    if k == 'let': return ('let', f(t[1], 0), f(t[2], 0), f(t[3], 1))
    if k == 'match': return ('match', f(t[1], 0), f(t[2], 0), tuple(f(b, 0) for b in t[3]))
    if k == 'fix': return ('fix', t[1], f(t[2], 0), f(t[3], 1))
    return t
def py_lift(t, d, c=0):
    if t[0] == 'bvar': return ('bvar', t[1] + d) if t[1] >= c else t
    return _map(t, lambda x, e: py_lift(x, d, c + e))
def py_inst(t, vals, off=0):
    if t[0] == 'bvar':
        i = t[1]
        if i < off: return t
        k = i - off
        return py_lift(vals[k], off) if k < len(vals) else ('bvar', i - len(vals))
    return _map(t, lambda x, e: py_inst(x, vals, off + e))
def py_abs(t, fvars, depth=0):
    n = len(fvars)
    if t[0] == 'bvar': return ('bvar', t[1] + n) if t[1] >= depth else t
    if t[0] == 'fvar':
        for i, l in enumerate(fvars):
            if l == t[1]: return ('bvar', depth + (n - 1 - i))
        return t
    return _map(t, lambda x, e: py_abs(x, fvars, depth + e))
def py_loose(t):
    k = t[0]
    if k == 'bvar': return t[1] + 1
    if k == 'app': return max(py_loose(t[1]), py_loose(t[2]))
    if k in ('lam', 'pi'): return max(py_loose(t[1]), max(py_loose(t[2]) - 1, 0))
    if k == 'let': return max(py_loose(t[1]), py_loose(t[2]), max(py_loose(t[3]) - 1, 0))
    if k == 'match': return max([py_loose(t[1]), py_loose(t[2])] + [py_loose(b) for b in t[3]])
    if k == 'fix': return max(py_loose(t[2]), max(py_loose(t[3]) - 1, 0))
    return 0

def rand_h(rng, d, nb, nf, nconst=14):
    """random hybrid tree; bvar indices < nb (loose ones allowed up to nb), fvar levels < nf"""
    if d <= 0 or rng.random() < 0.2:
        c = rng.randrange(4)
        if c == 0 and nb > 0: return ('bvar', rng.randrange(nb))
        if c == 1 and nf > 0: return ('fvar', rng.randrange(nf))
        if c == 2: return ('type', rng.randrange(3))
        return ('const', rng.randrange(nconst))
    c = rng.randrange(8)
    s = lambda b=0: rand_h(rng, d - 1, nb + b, nf)
    if c in (0, 1): return ('app', s(), s())
    if c == 2: return ('lam', s(), s(1))
    if c == 3: return ('pi', s(), s(1))
    if c == 4: return ('let', s(), s(), s(1))
    if c == 5: return ('match', s(), s(), tuple(s() for _ in range(rng.randrange(1, 3))))
    if c == 6: return ('fix', rng.randrange(2), s(), s(1))
    return ('app', s(), s())

def test_properties(knew, n, seed):
    rng = random.Random(seed); bad = []
    for i in range(n):
        nb = rng.randrange(0, 4); t = rand_h(rng, rng.randrange(1, 6), nb, 3)
        tn = build_new(knew, t); get = lambda x: parse_raw(knew.dump_raw(x), True)
        checks = []
        checks.append(('loose_range', knew.loose_range(tn), py_loose(t)))
        d = rng.randrange(0, 3); c = rng.randrange(0, 3)
        checks.append(('lift', get(knew.lift(tn, d, c)), py_lift(t, d, c)))
        vals = [rand_h(rng, 2, 0, 3) for _ in range(rng.randrange(1, 3))]
        checks.append(('instantiate', get(knew.instantiate(tn, [build_new(knew, v) for v in vals])), py_inst(t, vals)))
        fv = rng.sample(range(0, 5), rng.randrange(1, 3))
        checks.append(('abstract', get(knew.abstract_fvars(tn, fv)), py_abs(t, fv)))
        t0 = t if py_loose(t) <= 1 else None
        if t0 is not None:                       # law: abstract (instantiate t [fvar k]) [k] == t   for fresh k
            k = 50 + rng.randrange(5); inst = knew.instantiate(tn, [knew.mk_fvar(k)])
            checks.append(('law abstract∘instantiate', get(knew.abstract_fvars(inst, [k])), t0))
        if py_loose(t) == 0:                     # law: instantiate (abstract t [k]) [fvar k] == t
            k = 7; checks.append(('law instantiate∘abstract', get(knew.instantiate(knew.abstract_fvars(tn, [k]), [knew.mk_fvar(k)])), t))
        for name, got, want in checks:
            if got != want: bad.append((name, t, got, want))
    return bad

# ------------------------------------------------------------------ differential tests
R_refs = {}
def setup():
    kref, refs = cur.build_base_kernel(); refs.update(cur.setup_nat_order(kref, refs)); R_refs.update(refs)
    knew = H.Kernel(); load_library(kref, knew)
    g = {name: gid for gid, name, _, _ in kref.list_globals()}
    probe = ('match', ('const', g['Zero']), ('lam', ('const', g['Nat']), ('const', g['Nat'])),
             (('const', g['Zero']), ('lam', ('const', g['Nat']), ('app', ('const', g['Succ']), ('var', 0)))))
    nf = parse_raw(kref.dump_raw(kref.normalize_node(build_ref(kref, probe))), False)
    return kref, knew, g, nf == ('const', g['Zero'])

def count_kinds(t, acc):
    acc[t[0]] = acc.get(t[0], 0) + 1
    for c in t[1:]:
        if isinstance(c, tuple) and c and isinstance(c[0], str): count_kinds(c, acc)
        elif isinstance(c, tuple):
            for x in c:
                if isinstance(x, tuple) and x and isinstance(x[0], str): count_kinds(x, acc)

def differential(N, seed, depth=4):
    kref, knew, g, zf = setup(); rng = random.Random(seed); G = Gen(g, rng, zf); G2 = Gen(g, rng, zf); G2.allow_match = False; G2.allow_fix = False
    cov = {}; res = dict(norm=[0, []], norm_open=[0, []], conv=[0, []], typing=[0, []], typing_open=[0, []], reject=[0, []])
    t_ref = t_new = tc_ref = tc_new = 0.0
    def ref_eq_check(A, x_ref, y_ref):
        """ground truth for conversion: the REFERENCE kernel's own check of  refl A x : Eq A x y  (succeeds iff x == y, eta included)"""
        eq = cur.eq_app(kref, {'eq_ty': R_refs['eq_ty']}, A, x_ref, y_ref) if False else kref.mk_app(kref.mk_app(kref.mk_app(R_refs['eq_ty'], A), x_ref), y_ref)
        pf = kref.mk_app(kref.mk_app(R_refs['refl_c'], A), x_ref)
        try: kref.whole_term_check(pf, eq); return True
        except Exception: return False
    for i in range(N):
        T = G.rtype(); term = G.gen(T, [], depth); count_kinds(term, cov)
        tr, tn = build_ref(kref, term), build_new(knew, l2h(term, 0))
        try:   # closed normalisation
            a = time.perf_counter(); nr = kref.normalize_node(tr); b = time.perf_counter(); nn = knew.normalize_open(tn, 0); c = time.perf_counter()
            t_ref += b - a; t_new += c - b
            want = l2h(parse_raw(kref.dump_raw(nr), False), 0); got = parse_raw(knew.dump_raw(nn), True); res['norm'][0] += 1
            if want != got or knew.loose_range(nn) != 0: res['norm'][1].append((term, want, got, 'loose=%d' % knew.loose_range(nn)))
        except Exception as e: res['norm'][1].append((term, 'EXC', repr(e)[:120]))
        # typing: Match-free closed terms must be accepted at their type by both
        tt = G.gen(T, [], depth); ttr, ttn = build_ref(kref, tt), build_new(knew, l2h(tt, 0))
        try:
            exp_new = build_new(knew, l2h(G.ty(T), 0)); a = time.perf_counter()
            kref.whole_term_check(ttr, build_ref(kref, G.ty(T))); b = time.perf_counter(); ok, ty = knew.infer_open(ttn, []); c = time.perf_counter()
            tc_ref += b - a; tc_new += c - b; res['typing'][0] += 1
            if not ok or not knew.conv_open(ty, exp_new, 0): res['typing'][1].append((tt, T, ok, ty if not ok else knew.dump_raw(ty)))
        except Exception as e: res['typing'][1].append((tt, T, 'EXC', repr(e)[:120]))
        # conversion against the reference kernel's own check (refl)
        t2 = G.gen(T, [], depth); t2r, t2n = build_ref(kref, t2), build_new(knew, l2h(t2, 0)); Aref = build_ref(kref, G.ty(T))
        try:
            nf1 = kref.normalize_node(tr)
            for (x_new, y_new, y_ref) in ((tn, t2n, t2r), (tn, build_new(knew, l2h(parse_raw(kref.dump_raw(nf1), False), 0)), nf1)):
                truth = ref_eq_check(Aref, tr, y_ref); res['conv'][0] += 1
                if knew.conv_open(x_new, y_new, 0) != truth: res['conv'][1].append((term, t2, 'ref says', truth))
        except Exception as e: res['conv'][1].append((term, t2, 'EXC ' + repr(e)[:100]))
        # open terms (1-3 context variables): normalisation (with Match) and typing (without)
        nf_ = rng.randrange(1, 4); ctx = [G.rtype(0) if rng.random() < 0.7 else G.rtype(1) for _ in range(nf_)]
        hr = [build_ref(kref, G.ty(c)) for c in ctx]; hn_ = [build_new(knew, l2h(G.ty(c), 0)) for c in ctx]
        T2 = G.rtype(); ot = G.gen(T2, ctx, depth); otr, otn = build_ref(kref, ot), build_new(knew, l2h(ot, nf_))
        try:
            nr = kref.normalize_in_context(otr, hr); nn = knew.normalize_open(otn, nf_)
            want = l2h(parse_raw(kref.dump_raw(nr), False), nf_); got = parse_raw(knew.dump_raw(nn), True); res['norm_open'][0] += 1
            if want != got or knew.loose_range(nn) != 0: res['norm_open'][1].append((ot, ctx, want, got, 'loose=%d' % knew.loose_range(nn)))
            cov['stuck_match_in_nf'] = cov.get('stuck_match_in_nf', 0) + (1 if 'match' in str(got) else 0)
        except Exception as e: res['norm_open'][1].append((ot, ctx, 'EXC', repr(e)[:120]))
        ot2 = G.gen(T2, ctx, depth)
        try:
            ok, ty = knew.infer_open(build_new(knew, l2h(ot2, nf_)), hn_); res['typing_open'][0] += 1
            if not ok or not knew.conv_open(ty, build_new(knew, l2h(G.ty(T2), 0)), nf_): res['typing_open'][1].append((ot2, ctx, T2, ok, ty if not ok else knew.dump_raw(ty)))
        except Exception as e: res['typing_open'][1].append((ot2, ctx, T2, 'EXC', repr(e)[:120]))
        if i % 3 == 0:   # ill-typed terms must be rejected by both
            bt = G.bad([], 2)
            try:
                try: kref.whole_term_check(build_ref(kref, bt), build_ref(kref, G.ty(T))); ref_rejects = False
                except Exception: ref_rejects = True
                ok, ty = knew.infer_open(build_new(knew, l2h(bt, 0)), [])
                new_rejects = (not ok) or (not knew.conv_open(ty, build_new(knew, l2h(G.ty(T), 0)), 0)); res['reject'][0] += 1
                if not (ref_rejects and new_rejects): res['reject'][1].append((bt, T, ref_rejects, new_rejects))
            except Exception as e: res['reject'][1].append((bt, T, 'EXC', repr(e)[:100]))
    return res, cov, dict(norm_ref_ms=t_ref * 1e3, norm_new_ms=t_new * 1e3, tc_ref_ms=tc_ref * 1e3, tc_new_ms=tc_new * 1e3), zf


# ------------------------------------------------------------------ proof corpus: REAL Fix/Match terms from the reference kernel's tactics
def proof_corpus_test(seed=0):
    """Extract proofs of  forall v0..v_{n-1}. v0 = v0  built with nested induct/cases in the reference kernel (their Fix nodes carry
    fn_types that mention outer binders), then compare normalisation of the proof as is, applied to numerals, and applied to free
    variables (stuck Fix/Match -> exercises requote)."""
    kref, knew, g, zf = setup(); rng = random.Random(seed)
    kk, refs = cur.build_base_kernel(); nat = refs['nat_ty']; V = kk.mk_var
    specs = [(['I:0'], 1), (['C:0'], 1), (['I:0', 'I:1'], 2), (['I:1', 'I:0'], 2), (['I:0', 'I:last'], 2), (['C:0', 'I:last'], 2),
             (['I:0', 'I:1', 'I:2'], 3), (['I:2', 'I:1', 'I:0'], 3), (['I:1', 'I:2', 'I:0'], 3), (['C:0', 'C:1', 'I:2'], 3),
             (['I:0', 'I:1', 'I:2', 'I:3'], 4), (['I:3', 'I:2', 'I:1', 'I:0'], 4), (['I:2', 'I:0', 'I:3', 'I:1'], 4)]
    out = dict(n=0, bad=[], fix_nodes=0, stuck=0)
    NATC = ('const', g['Nat']); ZERO = ('const', g['Zero']); SUCC = ('const', g['Succ'])
    def numeral(n):
        t = ZERO
        for _ in range(n): t = ('app', SUCC, t)
        return t
    for spec, nv in specs:
        ty = cur.eq_app(kk, refs, nat, V(0), V(0))
        for _ in range(nv): ty = kk.mk_pi(nat, ty)
        ps = kk.start_proof(ty)
        for i in range(nv): ps.intro('v%d' % i, target_meta_id=ps.goal_meta_id(0))
        cm = ps.goal_meta_id(0); gi = lambda m: next(i for i in range(ps.num_goals()) if ps.goal_meta_id(i) == m)
        def last_nat(gg): return max(j for j in range(ps.goal_num_hyps(gg)) if kk.print_node(ps.goal_hyp_type(gg, j)) == 'Nat')
        for step in spec:
            kind, which = step.split(':'); gg = gi(cm); h = last_nat(gg) if which == 'last' else int(which)
            before = {ps.goal_meta_id(i) for i in range(ps.num_goals())}
            (ps.induct if kind == 'I' else ps.cases)(h, target_meta_id=cm)
            new = [ps.goal_meta_id(i) for i in range(ps.num_goals()) if ps.goal_meta_id(i) not in before]
            for m in new[:-1]: ps.apply(refs['refl_c'], target_meta_id=m)
            cm = new[-1]
        ps.apply(refs['refl_c'], target_meta_id=cm); proof = ps.extract()
        ptree = parse_raw(kk.dump_raw(proof), False); out['fix_nodes'] += str(ptree).count("'fix'")
        # (1) the proof itself, (2) applied to numerals, (3) applied to free variables.  The reference side keeps the ORIGINAL proof node
        # (it has no mk_fix, so it cannot be rebuilt) and applies arguments to it.
        for mode in ('closed', 'numerals', 'free'):
            args = []
            if mode == 'numerals': args = [numeral(rng.randrange(0, 4)) for _ in range(nv)]
            if mode == 'free': args = [('var', i) for i in range(nv)]
            ctxn = nv if mode == 'free' else 0
            t = ptree
            for a in args: t = ('app', t, a)
            hyps_ref = [kk.mk_const(g['Nat'])] * nv if mode == 'free' else []
            try:
                tr = proof
                for a in args: tr = kk.mk_app(tr, build_ref(kk, a))
                tn = build_new(knew, l2h(t, ctxn))
                nr = kk.normalize_in_context(tr, hyps_ref) if mode == 'free' else kk.normalize_node(tr)
                nn = knew.normalize_open(tn, ctxn)
                want = l2h(parse_raw(kk.dump_raw(nr), False), ctxn); got = parse_raw(knew.dump_raw(nn), True)
                out['n'] += 1; out['stuck'] += ('match' in str(got) or 'fix' in str(got))
                if want != got or knew.loose_range(nn) != 0: out['bad'].append((spec, mode, str(want)[:160], str(got)[:160]))
            except Exception as e: out['bad'].append((spec, mode, 'EXC', repr(e)[:160]))
    return out

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    t0 = time.time(); res, cov, tm, zf = differential(N, seed, depth)
    print("seed %d, N=%d terms (Zero is ctor 0: %s) | %.1fs" % (seed, N, zf, time.time() - t0))
    print("coverage (node kinds in generated closed terms):", {k: cov[k] for k in sorted(cov)})
    for name, (n, bad) in res.items(): print("  %-10s checked %6d | disagreements %d" % (name, n, len(bad)))
    for name, (n, bad) in res.items():
        for ex in bad[:2]: print("  FIRST FAILURES [%s]: %s" % (name, str(ex)[:330]))
    pc = proof_corpus_test(seed)
    print("proof corpus (real Fix/Match terms from reference tactics): %d normalisations, %d Fix nodes in the proofs, %d with a stuck Fix/Match | disagreements %d" % (pc['n'], pc['fix_nodes'], pc['stuck'], len(pc['bad'])))
    for b in pc['bad'][:3]: print('  FIRST FAILURES [corpus]:', b)
    print("timing  normalise: ref %.0f ms vs hybrid %.0f ms (ratio %.2fx) | typing: ref check %.0f ms vs hybrid infer %.0f ms (ratio %.2fx)" %
          (tm['norm_ref_ms'], tm['norm_new_ms'], tm['norm_new_ms'] / max(tm['norm_ref_ms'], 1e-9), tm['tc_ref_ms'], tm['tc_new_ms'], tm['tc_new_ms'] / max(tm['tc_ref_ms'], 1e-9)))
