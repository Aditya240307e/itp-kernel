"""synth_pool.py -- the lemmas the synthetic theorem generator may build on, and what it needs to know about each.

    pool = Pool(w, limit)               # w: a world (the library), limit: the number of globals the generator may use (the library before the problem bank)
    pool.lemmas                         # the curated lemmas, in global-id order: PoolLemma(gid, name, area, domain, ...)
    pool.by_gid[gid]; pool.by_name[name]; pool.summary()

WHAT IS CURATED.  A synthetic theorem is only as natural as the lemmas it is built from. The library holds what its proofs needed, and a good part of that is plumbing -- the quotient's well-definedness
lemmas, the representation of the integers as pairs of naturals, the induction principles, the lemmas about the decision procedures -- which no mathematician would apply to a goal. A lemma is in the
pool when (a) it is a theorem of the library BEFORE the problem bank (so that nothing in a synthetic theorem depends on a problem), (b) every constant its statement mentions belongs to the vocabulary of
one DOMAIN (`DOMAINS`: propositional logic, the integers, the binary naturals, the unary naturals), (c) its binders are numbers or propositions (no higher-order lemma: no motive, no function, no
type), and (d) it is not on the list of lemmas that are the library's own machinery (`NOT_A_BUILDING_BLOCK`). The test is on the STRUCTURE of the statement, not on its name, so that a library
that grows (sums, factorials, remainders, ...) feeds the pool by extending a vocabulary, not by listing lemmas by hand.

AREA is the topic of a lemma, read off the head of its conclusion (order, equality, divisibility, gcd, prime, logic, existence); it is what the generator balances.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import collections
import gym_env as G

NUMERAL_CONSTS = ('Z.of_N', 'Z.neg', 'Z.zero', 'Z.one', 'Npos', 'N0', 'XH', 'XO', 'XI')
LOGIC_CONSTS = ('Eq', 'And', 'Or', 'Not', 'Iff', 'False', 'True', 'Ex')
# the constants a statement of a domain may mention, in the order the domains are tried (a statement that mentions no number type at all is `logic`)
DOMAINS = {
    'logic': dict(types=(), consts=LOGIC_CONSTS),
    'Z': dict(types=('Z',), consts=('Z', 'Z.le', 'Z.lt', 'Z.add', 'Z.mul', 'Z.sub', 'Z.neg', 'Z.zero', 'Z.one', 'Z.dvd', 'Z.coprime', 'Z.is_gcd', 'Z.prime') + LOGIC_CONSTS + NUMERAL_CONSTS),
    'N': dict(types=('N',), consts=('N', 'N.le', 'N.lt', 'N.add', 'N.mul', 'N.succ', 'N0', 'Npos', 'XH', 'XO', 'XI') + LOGIC_CONSTS),
    'Nat': dict(types=('Nat',), consts=('Nat', 'Zero', 'Succ', 'add', 'mul', 'pow', 'Le', 'Lt', 'pred') + LOGIC_CONSTS),
}
# lemmas that are plumbing of the library's own construction (or need a motive, a function, a type), however their statements look
NOT_A_BUILDING_BLOCK = frozenset('''
    Z.mk_eq Z.shift Z.test_wd Z.neg_wd Z.add_wd Z.mul_wd1 Z.mul_wd2 Z.mul_wd Z.neg_mk Z.add_mk Z.mul_mk Z.sub_def Z.sqr_def Z.ppow_succ Z.of_N_def Z.le_of_mk Z.mk_le
    Z.int_induction Z.induction_from Z.strong_induction Z.bezout_aux Z.sqrt2_aux Z.not_nonneg_neg Z.of_N_inj Z.nonneg_exists
    transport f_equal cong_succ true_intro false_elim cut ex_elim succ_inj succ_ne_zero zero_ne_succ
'''.split())
AREA_OF_HEAD = {'Z.le': 'order', 'Z.lt': 'order', 'N.le': 'order', 'N.lt': 'order', 'Le': 'order', 'Lt': 'order', 'Eq': 'equality', 'Z.dvd': 'divisibility',
                'Z.coprime': 'gcd', 'Z.is_gcd': 'gcd', 'Z.prime': 'prime', 'And': 'logic', 'Or': 'logic', 'Not': 'logic', 'Iff': 'logic', 'False': 'logic', 'True': 'logic', 'Ex': 'existence'}
PLUMBING_PREFIXES = ('ring_', 'Quot.', 'Bool.', 'Pos.', 'of_N_')         # (tested on the name and on its last component: Z.ring_add_congr, Z.of_N_zero)


class PoolLemma:
    """one lemma of the pool: `data` / `premises` are the positions of its binders that are numbers (or proposition variables) and proofs of propositions"""
    __slots__ = ('gid', 'name', 'area', 'domain', 'n_binders', 'data', 'premises', 'head', 'statement', 'negated')
    def __init__(self, gid, name, area, domain, n_binders, data, premises, head, statement, negated=False):
        self.gid = gid; self.name = name; self.area = area; self.domain = domain; self.n_binders = n_binders
        self.data = data; self.premises = premises; self.head = head; self.statement = statement
        self.negated = negated                      # the conclusion, as written, is a negation (`Not P` opens to `P -> False`: the engine counts P among the binders)
    def __repr__(self): return 'PoolLemma(%s, %s/%s, %d premises)' % (self.name, self.domain, self.area, len(self.premises))


def consts_in(k, node, acc):
    """the global ids a kernel term mentions (added to the set `acc`)"""
    stack = [node]; seen = set()
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        v = k.view(n); t = v[0]
        if t == 'const': acc.add(v[1])
        elif t == 'app' or t == 'lam' or t == 'pi': stack.append(v[1]); stack.append(v[2])
        elif t == 'let': stack += [v[1], v[2], v[3]]
        elif t == 'match': stack += [v[1], v[2]] + list(v[3])
        elif t == 'fix': stack += [v[1], v[3]]
    return acc


class Pool:
    def __init__(self, w, limit, domains=('logic', 'Z'), extra_excluded=()):
        self.w = w; self.k = w.k; self.limit = min(limit, w.k.num_globals())
        self.lemmas = []; self.by_gid = {}; self.by_name = {}; self.rejected = collections.Counter(); self.domains = tuple(domains)
        k = self.k
        gname = {g: n for n, g in w.gid.items()}
        self._spec = {}
        for dom in self.domains:
            spec = DOMAINS[dom]
            self._spec[dom] = ({w.gid[c] for c in spec['consts'] if c in w.gid}, [k.mk_const(w.gid[c]) for c in spec['types'] if c in w.gid])
        for gid in range(self.limit):
            name = gname.get(gid)
            if name is None: continue
            lem = self._consider(gid, name, set(extra_excluded))
            if lem is not None: self.lemmas.append(lem); self.by_gid[gid] = lem; self.by_name[name] = lem

    def _no(self, why):
        self.rejected[why] += 1
        return None

    def _consider(self, gid, name, extra):
        k = self.k; w = self.w
        _, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        if val is None or k.is_inductive(gid): return self._no('not a proved lemma')
        ok, sort = k.infer_open(ty, [])
        if not ok or sort != k.mk_type(0): return self._no('not a theorem')
        if name in NOT_A_BUILDING_BLOCK or name in extra or name.startswith(PLUMBING_PREFIXES) or name.split('.')[-1].startswith(PLUMBING_PREFIXES): return self._no('plumbing')
        mentioned = consts_in(k, ty, set())
        dom = next((d for d in self.domains if mentioned <= self._spec[d][0]), None)
        if dom is None: return self._no('mentions a constant outside the vocabularies')
        numbers = self._spec[dom][1]
        try: doms, concl = G.open_binders(k, ty, 0)
        except Exception: return self._no('unreadable')
        data = []; premises = []
        for j, d in enumerate(doms):
            okd, sortd = k.infer_open(d, doms[:j])
            if okd and sortd == w.PROP.n: premises.append(j)
            elif d == w.PROP.n or d in numbers: data.append(j)
            else: return self._no('a binder that is neither a number nor a proposition')
        head = self._head_name(concl)
        area = AREA_OF_HEAD.get(head)
        if area is None: return self._no('the conclusion has no area (%s)' % head)
        written = 0; cur = ty
        while k.view(cur)[0] == 'pi': cur = k.view(cur)[2]; written += 1
        return PoolLemma(gid, name, area, dom, len(doms), tuple(data), tuple(premises), head, w.show(w.global_type(name)), negated=len(doms) > written)

    def _head_name(self, node):
        k = self.k
        while True:
            v = k.view(node)
            if v[0] == 'app': node = v[1]; continue
            break
        return self.w.gname.get(v[1], '?') if v[0] == 'const' else '?'

    def summary(self):
        by = collections.Counter((l.domain, l.area) for l in self.lemmas)
        return dict(sorted(by.items()))
