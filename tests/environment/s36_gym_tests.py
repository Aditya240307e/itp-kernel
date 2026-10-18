"""S36 gate: the learner's environment on the hybrid kernel (gym_env.py, gym_search.py), the `clear` tactic and the observation of goals.

Nothing in the environment is trusted: an action is carried out by the tactic layer, a proof that a search finds is replayed on a live proof and re-checked by the kernel (and here, by the independent checker as well).
What this gate pins down is that the machinery around that check does what its documentation says, because a search is only as good as its probes -- and a probe that is wrong wastes the search, or loses a proof:

  A. OBSERVATION.  The arrays of `observe_goal` / `observe_term` decode (by a decoder written here, which knows only the layout documented in itp_bindings.cpp) to exactly the structure the kernel's `view` reports;
     heights are exactly 1 + the highest child; every row is reachable from a root; there is one row per distinct node (plus the `cons` rows of let / match); scoping errors are refused; a unary numeral
     twenty thousand deep is an ordinary term; `term_features` agrees with the first row of a dump; `library_candidates` is the library half of `apply_candidates`.
  B. PROBE HYGIENE.  A probe that fails changes the kernel's node pools by exactly NOTHING; a successful one creates only the nodes of the goals it keeps, so the same probe again creates nothing and returns the
     very same goals; the work budget and the `probing` flag are restored (also after running out of budget); the Python caches of node handles are not written to while probing (and are, outside probes).
  C. REPLAY FIDELITY.  Random episodes: the goals a live step leaves are exactly the goals the probe predicted (same kernel nodes, same order); a step that the probe refused raises and changes nothing;
     finished episodes are verified by the kernel.
  D. SEARCH.  Lemmas of the library are proved without ring / linarith (induction, rewriting and case analysis do the work), found at the least depth, and the proofs are accepted by the kernel and by the
     independent checker; budgets are honoured; false statements are never proved; forged action trees are refused.
  E. `clear` AND WHAT `cases` / `induct` REVERT.  `clear` removes exactly the hypotheses asked for, renumbers the rest, refuses to remove one that the goal or another hypothesis mentions, and a proof that goes
     through it is accepted; the kernel reports the hypotheses that the latest `cases` / `induct` reverted (a suffix for one, the dependents for the other).
  F. INDEPENDENCE.  A step that leaves a unification constraint pending (`apply add_assoc` to `a + b = b + a`) is refused by the probe and by the episode, and changes nothing; a state with a constraint pending
     is not a finished proof; an action that leaves a goal with a metavariable is refused by the probe.
  G. THE MENU AND THE CLEAN-UP.  Every action is found again from its signature; candidates respect the premises limit; case analysis is offered only on inductive hypotheses; the kinds can be restricted;
     the dead copies that `cases` / `induct` leave behind are removed (and only those); goals that a hypothesis or computation closes are closed; menus and probes do not depend on the history of the world.
  H. INTEGERS.  `interval_cases` is offered with the right range and leaves clean goals; `linarith` / `nlinarith` appear exactly when the goal or a hypothesis is arithmetic; problems of the integer library
     are solved, with proofs the independent checker accepts.
  I. GROWTH AND ERRORS.  A lemma proved after the engine was made is offered after `refresh`; everything a probe saw going wrong was one of the tactic layer's own refusals.
  K. THE ENGINE'S BUG ACCOUNTING AND THE FUZZER'S OWN HEALTH.  An exception inside a probe that is not a refusal of the tactic layer or the kernel is a bug of the environment: it is kept with its traceback, raised by a
     strict engine (after the kernel was rewound and the budget restored), and never mistaken for a refused action; work budgets given as floats are accepted; the fuzzer's canaries (goals that ring, linarith and
     nlinarith certainly close) pass in the natural and the integer world and fail for an engine that cannot run a decision procedure; a short integer fuzz proves its true statements and finds nothing wrong.
  J. WHAT THE TACTICS FINISH, THE KERNEL ACCEPTS.  The unifier gives the right verdict when a metavariable is solved inside a stuck `match` (`Succ ?x = ?x + 1` against the false `Succ a = a` once made `apply ring_succ`
     close a goal with a term the checker rejects, and a search "prove" `a = S a`); `apply` and `rw` on a grid of lemmas and random true and false equations never finish a proof the kernel refuses; random false
     statements are never proved, and no finished search proof is refused.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, random, sys, time
import numpy as np

sys.setrecursionlimit(200000)
import itp_env_lnl as E
import nat_lib, int_lib
import gym_env as G, gym_search as S
import ring as R
from author import AuthorError, NodeCache
from tactics import Proof
import indep_checker as ic
from harness import parse_raw

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S36_TRACE'))
def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)
def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return cond
REFUSALS = (G.ActionError, AuthorError, E.ITPError)
def refuses(what, fn, exc=REFUSALS):
    """fn() must raise one of exc"""
    try: fn()
    except exc: return expect(True, what)
    except Exception as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:80]))
    return expect(False, what + ' (it raised nothing)')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ helpers
def theorem_gids(w):
    """(global id, name) of the theorems of a world, in declaration order (derived globals, whose names start with __, are left out)"""
    k = w.k; out = []
    for gid, name, ty, has_val in k.list_globals():
        if not has_val or name.startswith('__'): continue
        ok, sort = k.infer_open(ty, [])
        if ok and sort == w.PROP.n: out.append((gid, name))
    return out


def reference(k, n, memo):
    """the structure of a term read off with `view`, as nested tuples (the same shape as `decode`)"""
    hit = memo.get(n)
    if hit is not None: return hit
    v = k.view(n); kind = v[0]
    if kind in ('fvar', 'bvar', 'type', 'const', 'meta', 'tuniv'): r = (kind, v[1])
    elif kind in ('app', 'lam', 'pi'): r = (kind, reference(k, v[1], memo), reference(k, v[2], memo))
    elif kind == 'let': r = ('let', reference(k, v[1], memo), reference(k, v[2], memo), reference(k, v[3], memo))
    elif kind == 'match': r = ('match', reference(k, v[1], memo), reference(k, v[2], memo), tuple(reference(k, b, memo) for b in v[3]))
    elif kind == 'fix': r = ('fix', reference(k, v[1], memo), v[2], reference(k, v[3], memo))
    else: raise AssertionError('unknown view kind %r' % (kind,))
    memo[n] = r; return r


def decode(dump, i, memo):
    """the structure of row i of a dump, by the layout documented in itp_bindings.cpp (and nothing else): leaves carry a scalar; APP / LAMBDA / PI / FIX have two children; LET = (type, cons(value, body));
    MATCH = (scrutinee, cons(motive, branch, ...)) where a chain of SYNTHETIC_CONS rows ends in the last item itself"""
    tags, scalar, ca, cb, roots, heights = dump
    hit = memo.get(i)
    if hit is not None: return hit
    name = E.OBS_TAG_NAMES[int(tags[i])]
    def chain(head):
        items = []; x = int(head)
        while E.OBS_TAG_NAMES[int(tags[x])] == 'SYNTHETIC_CONS': items.append(int(ca[x])); x = int(cb[x])
        items.append(x); return items
    if name == 'FVAR': r = ('fvar', int(scalar[i]))
    elif name == 'BVAR': r = ('bvar', int(scalar[i]))
    elif name == 'TYPE': r = ('type', int(scalar[i]))
    elif name == 'CONST': r = ('const', int(scalar[i]))
    elif name == 'META': r = ('meta', int(scalar[i]))
    elif name == 'TYPE_UNIV': r = ('tuniv', int(scalar[i]))
    elif name == 'APP': r = ('app', decode(dump, int(ca[i]), memo), decode(dump, int(cb[i]), memo))
    elif name == 'LAMBDA': r = ('lam', decode(dump, int(ca[i]), memo), decode(dump, int(cb[i]), memo))
    elif name == 'PI': r = ('pi', decode(dump, int(ca[i]), memo), decode(dump, int(cb[i]), memo))
    elif name == 'FIX': r = ('fix', decode(dump, int(ca[i]), memo), int(scalar[i]), decode(dump, int(cb[i]), memo))
    elif name == 'LET':
        val, body = chain(cb[i]); r = ('let', decode(dump, int(ca[i]), memo), decode(dump, val, memo), decode(dump, body, memo))
    elif name == 'MATCH':
        items = chain(cb[i]); r = ('match', decode(dump, int(ca[i]), memo), decode(dump, items[0], memo), tuple(decode(dump, j, memo) for j in items[1:]))
    else: raise AssertionError('a row of kind %s was reached directly' % name)
    memo[i] = r; return r


def distinct_nodes(k, roots):
    """how many distinct kernel nodes the given roots reach, and how many `cons` rows a dump of them needs (a Let has one, a Match with b branches has b)"""
    seen = set(); stack = list(roots); cons = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); v = k.view(n); kind = v[0]
        if kind == 'app': stack += [v[1], v[2]]
        elif kind in ('lam', 'pi'): stack += [v[1], v[2]]
        elif kind == 'let': stack += [v[1], v[2], v[3]]; cons += 1
        elif kind == 'match': stack += [v[1], v[2]] + list(v[3]); cons += len(v[3])
        elif kind == 'fix': stack += [v[1], v[3]]
    return len(seen), cons


LEAVES = ('FVAR', 'BVAR', 'TYPE', 'CONST', 'META', 'TYPE_UNIV')
def dump_problems(k, nodes, dump):
    """everything wrong with a dump of `nodes` (the roots, in order), as a list of short descriptions"""
    tags, scalar, ca, cb, roots, heights = dump
    n = len(tags); bad = []
    if not (tags.dtype == np.uint32 and scalar.dtype == np.uint32 and heights.dtype == np.uint32 and ca.dtype == np.int32 and cb.dtype == np.int32 and roots.dtype == np.int32): bad.append('dtypes')
    if not (len(scalar) == n and len(ca) == n and len(cb) == n and len(heights) == n): return bad + ['array lengths']
    if len(roots) != len(nodes): return bad + ['number of roots']
    if n and int(tags.max()) >= len(E.OBS_TAG_NAMES): return bad + ['a tag out of range']
    names = [E.OBS_TAG_NAMES[int(t)] for t in tags]
    for i in range(n):
        a, b = int(ca[i]), int(cb[i])
        if names[i] in LEAVES:
            if a != -1 or b != -1: bad.append('a leaf has children'); break
        elif not (0 <= a < n and 0 <= b < n): bad.append('a node lacks a child, or a child is out of range'); break
    if bad: return bad
    ca64 = ca.astype(np.int64); cb64 = cb.astype(np.int64); h64 = heights.astype(np.int64)
    ha = np.where(ca64 >= 0, h64[np.maximum(ca64, 0)], -1); hb = np.where(cb64 >= 0, h64[np.maximum(cb64, 0)], -1)
    if not np.array_equal(h64, np.maximum(ha, hb) + 1): bad.append('heights are not 1 + the highest child (0 for a leaf)')
    seen = set(); stack = [int(r) for r in roots]
    while stack:
        i = stack.pop()
        if i in seen: continue
        seen.add(i)
        for c in (int(ca[i]), int(cb[i])):
            if c >= 0: stack.append(c)
    if len(seen) != n: bad.append('rows that no root reaches (%d of %d)' % (n - len(seen), n))
    dn, dc = distinct_nodes(k, nodes)
    if n != dn + dc: bad.append('row count %d, expected %d distinct nodes + %d cons rows' % (n, dn, dc))
    ref = {}; dec = {}
    try:
        for r, node in zip(roots, nodes):
            if decode(dump, int(r), dec) != reference(k, node, ref): bad.append('a root does not decode to the structure `view` reports')
    except AssertionError as e: bad.append('decoding: %s' % e)
    return bad


def action_name(w, menu, a):
    """a readable, build-independent name of an action: its kind and what it refers to"""
    if a.kind in G.NEEDS_CANDIDATE:
        term, rev, aux = menu.cands[a.arg]
        if term is None: ref = 'interval%r' % (aux,)
        else:
            v = w.k.view(term); ref = w.gname.get(v[1], 'g%d' % v[1]) if v[0] == 'const' else 'hyp%d' % v[1]
        return '%s:%s%s' % (G.KIND_NAMES[a.kind], ref, '<-' if rev else '')
    return '%s(%d)' % (G.KIND_NAMES[a.kind], a.arg)


def with_premises(eng, gid):
    eng.premises = gid; return eng


NAT_NOARITH = [x for x in G.Kind if x not in (G.Kind.RING, G.Kind.LINARITH, G.Kind.NLINARITH)]
def first_goal(eng, stmt):
    p = eng.start(stmt)
    return (None, p) if p.done else (eng.goal_of(p, 0), p)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. observation
def sec_A(c):
    w, k, eng, ths = c['w'], c['k'], c['eng'], c['ths']
    goals = []
    for gid, name in ths[:60]:
        with_premises(eng, gid)
        g, p = first_goal(eng, w.global_type(name))
        if g is None: continue
        goals.append(g); m = eng.menu(g)
        for a in m.actions[:6]:
            ch = eng.probe(g, a, m)
            if ch: goals.extend(ch[:2])
    eng.premises = None
    c['goals'] = goals
    expect(len(goals) >= 80, 'A  enough goals to look at (%d)' % len(goals))
    bad = []; n_rows = 0
    for g in goals:
        nodes = list(g.hyp_types) + [g.target]
        dump = k.observe_goal(g.target, g.hyp_types); n_rows += len(dump[0])
        pr = dump_problems(k, nodes, dump)
        if pr: bad.append((pr, eng.render(g)[:60]))
        if eng.observe(g) is not eng.observe(g): bad.append(('the observation is not computed once',))
        for node in nodes[:3]:
            d1 = k.observe_term(node); tg, sc = k.term_features([node])
            root = int(d1[4][0])
            if len(d1[4]) != 1 or int(tg[0]) != int(d1[0][root]) or int(sc[0]) != int(d1[1][root]): bad.append(('term_features / observe_term disagree', eng.render(g)[:60]))
            pr = dump_problems(k, [node], d1)
            if pr: bad.append((pr, 'observe_term'))
    expect(not bad, 'A  every dump decodes to what `view` reports, with exact heights, reachable rows and the right row count: %s' % (bad[:3],))
    expect(n_rows > 1500, 'A  the observations are not trivial (%d rows)' % n_rows)
    tgs, scs = k.term_features([g.target for g in goals[:20]])
    expect(len(tgs) == 20 and len(scs) == 20 and all(int(tgs[i]) == int(k.observe_term(g.target)[0][0]) for i, g in enumerate(goals[:20])), 'A  term_features of many terms in one call = the first rows of their dumps')
    # terms with a match / fix / let in them: the values of the definitions of the library
    n_match = n_fix = n_let = n_seen = 0; bad = []
    for gid in range(k.num_globals()):
        v = k.global_info(gid)[2]
        if v is None: continue
        d = k.observe_term(v)
        if len(d[0]) > 20000: continue
        kinds = {E.OBS_TAG_NAMES[int(t)] for t in d[0]}
        if not kinds & {'MATCH', 'FIX', 'LET'}: continue
        n_seen += 1; n_match += 'MATCH' in kinds; n_fix += 'FIX' in kinds; n_let += 'LET' in kinds
        pr = dump_problems(k, [v], d)
        if pr: bad.append((k.global_info(gid)[0], pr))
    expect(n_match >= 3 and n_fix >= 3 and not bad, 'A  definitions with match / fix / let decode exactly (%d terms, %d with a match, %d with a fix, %d with a let): %s' % (n_seen, n_match, n_fix, n_let, bad[:3]))
    # a let, built here (nothing in the library may contain one)
    Nat = w['Nat'].n; zero = w['Zero'].n
    cp = k.checkpoint()
    let = k.mk_let(Nat, zero, k.mk_bvar(0))
    d = k.observe_term(let)
    expect(dump_problems(k, [let], d) == [] and 'LET' in {E.OBS_TAG_NAMES[int(t)] for t in d[0]} and 'SYNTHETIC_CONS' in {E.OBS_TAG_NAMES[int(t)] for t in d[0]}, 'A  a let is a node with a cons row')
    k.rewind(cp)
    # scoping: a term the kernel cannot read in the context is refused, not dumped
    refuses('A  a target that mentions a variable that is not in the context is refused', lambda: k.observe_goal(k.mk_fvar(3), []))
    refuses('A  a hypothesis type that mentions itself is refused', lambda: k.observe_goal(Nat, [k.mk_fvar(0)]))
    refuses('A  a target with a loose bound variable is refused', lambda: k.observe_goal(k.mk_bvar(0), []))
    d = k.observe_goal(Nat, [Nat, Nat])
    expect(len(d[4]) == 3 and int(d[4][0]) == int(d[4][1]) == int(d[4][2]), 'A  the same type three times is one row, three roots')
    # a unary numeral twenty thousand deep: an ordinary term (built with an explicit stack)
    wd = nat_lib.build_nat(); kd = wd.k; succ = kd.mk_const(wd.gid['Succ']); t = kd.mk_const(wd.gid['Zero'])
    for _ in range(20000): t = kd.mk_app(succ, t)
    d = kd.observe_term(t); tags = [E.OBS_TAG_NAMES[int(x)] for x in d[0]]
    expect(len(d[0]) == 20002 and tags.count('APP') == 20000 and tags.count('CONST') == 2 and int(d[5][int(d[4][0])]) == 20000, 'A  a unary numeral 20000 deep: 20000 applications, the two constants once each, height 20000')
    expect(len(set(int(x) for x, tg in zip(d[2], tags) if tg == 'APP')) == 1, 'A  ... and the function Succ is one shared row')
    # the library half of apply_candidates
    bad = []
    for g in goals[:40]:
        cp = k.checkpoint()
        raw = Proof.from_goal(w, g.target, g.hyp_types, g.hyp_names)
        both = k.apply_candidates(raw.ps, eng.lib); lib = k.library_candidates(eng.lib, g.target, g.n_hyps)
        k.rewind(cp)
        if both[:len(lib)] != lib or any(k.view(x)[0] != 'fvar' for x in both[len(lib):]): bad.append(eng.render(g)[:60])
    expect(not bad, 'A  library_candidates is the library part of apply_candidates (then come the hypotheses): %s' % (bad[:3],))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. probe hygiene
def sec_B(c):
    w, k, eng, goals = c['w'], c['k'], c['eng'], c['goals']
    def state(): return (tuple(k.pool_sizes()), k.num_globals(), k.get_fuel(), w.probing)
    n_fail = n_ok = n_closed = 0; bad = []
    for g in goals[:70]:
        m = eng.menu(g)
        for a in m.actions[:30]:
            s0 = state(); r1 = eng.probe(g, a, m); s1 = state()
            if r1 is None:
                n_fail += 1
                if s1 != s0: bad.append(('a failed probe changed the kernel', action_name(w, m, a), s0, s1))
            else:
                n_ok += 1; n_closed += (r1 == [])
                if any(x < y for x, y in zip(s1[0], s0[0])): bad.append(('a probe shrank the pools', action_name(w, m, a)))
                if s1[1:] != s0[1:]: bad.append(('a probe changed the globals, the budget or the flag', action_name(w, m, a)))
                if r1 == [] and s1 != s0: bad.append(('a probe that closes the goal left nodes behind', action_name(w, m, a), s0, s1))
            r2 = eng.probe(g, a, m); s2 = state()
            if s2 != s1: bad.append(('a second identical probe changed the kernel', action_name(w, m, a), s1, s2))
            if (r1 is None) != (r2 is None) or (r1 is not None and [x.key for x in r1] != [x.key for x in r2]): bad.append(('a second probe disagrees with the first', action_name(w, m, a)))
    expect(not bad, 'B  failed probes change nothing; probes that close a goal change nothing; repeating a probe creates nothing and gives the same goals: %s' % (bad[:3],))
    expect(n_fail >= 100 and n_ok >= 100 and n_closed >= 10, 'B  all kinds of probe were exercised (%d failed, %d succeeded, %d of them closed the goal)' % (n_fail, n_ok, n_closed))
    # a probe that is out of budget: refused, nothing left behind, the budget restored
    Nat = w['Nat']; F = w.forall_
    st = F([Nat, Nat], lambda a, b: w.Eq(w['mul'](w['add'](a, b), w['add'](a, b)), w['add'](w['add'](w['mul'](a, a), w['mul'](2, w['mul'](a, b))), w['mul'](b, b))))
    g, p = first_goal(eng, st); m = eng.menu(g)
    ring = [a for a in m.actions if a.kind == G.Kind.RING]
    expect(len(ring) == 1, 'B  ring is offered on a polynomial identity')
    s0 = state(); full = eng.probe(g, ring[0], m); s1 = state()
    expect(full == [] and s1 == s0, 'B  with its budget ring closes (a+b)^2 = a^2 + 2ab + b^2 and the probe leaves the kernel as it was')
    saved = dict(eng.probe_fuel); eng.probe_fuel[G.Kind.RING] = 50; n_before = sum(eng.errors.values()); n_failed = eng.stats['probe_failed']
    s0 = state(); starved = eng.probe(g, ring[0], m); s1 = state()
    eng.probe_fuel.clear(); eng.probe_fuel.update(saved)
    expect(starved is None and s1 == s0 and sum(eng.errors.values()) == n_before + 1 and eng.stats['probe_failed'] == n_failed + 1, 'B  with a budget of 50 steps the same probe (which succeeded above) is refused as a failure (counted), the kernel unchanged, the budget restored')
    expect(eng.probe(g, ring[0], m) == [] and state() == s0, 'B  ... and with the budget back the probe succeeds again')
    # checkpoints: a stale or foreign one is refused (a rewind cuts every node pool back to the sizes it recorded), before anything is touched
    RT = (RuntimeError, E.ITPError)
    def tower(n):
        t = w['Zero'].n
        for _ in range(n): t = k.mk_app(w['Succ'].n, t)
        return t
    def sizes(): return tuple(k.pool_sizes())
    cp1 = k.checkpoint(); sz1 = sizes(); tower(131)
    cp2 = k.checkpoint(); sz2 = sizes(); tower(263)
    sz3 = sizes()
    expect(sum(sz1) < sum(sz2) < sum(sz3), 'B  building fresh terms grows the node pools')
    k.rewind(cp2)
    expect(sizes() == sz2, 'B  rewinding to a checkpoint cuts the pools back to the sizes it recorded')
    k.rewind(cp1)
    expect(sizes() == sz1, 'B  ... rewinding to an earlier one cuts them further')
    refuses('B  a checkpoint that the rewind to an earlier one discarded is refused', lambda: k.rewind(cp2), exc=RT)
    expect(sizes() == sz1, 'B  ... and nothing changed')
    tower(131); k.rewind(cp1); expect(sizes() == sz1, 'B  the checkpoint rewound to is still good (the same state again)')
    refuses('B  a checkpoint of another kernel is refused', lambda: k.rewind(E.Kernel().checkpoint()), exc=RT)
    kA, kB = E.Kernel(), E.Kernel(); kA.checkpoint(); cpB = kB.checkpoint()          # the same serial number, live in both: only the owner tells them apart
    refuses('B  ... also when its serial number is live here', lambda: kA.rewind(cpB), exc=RT)
    cpx = k.checkpoint(); pr = Proof(w, w.global_type('add_comm')); pr.intro('a', 'b')
    k.rewind(cp1)
    refuses('B  finish_probe with a checkpoint that was discarded is refused', lambda: k.finish_probe(cpx, pr.ps), exc=RT)
    expect(sizes() == sz1, 'B  ... without touching the kernel')
    # the caches of node handles are not written while probing, and are outside probes
    ch = NodeCache(w)
    w.probing += 1
    try: ch['x'] = 1; ch.setdefault('y', 2); ch.update(z=3)
    finally: w.probing -= 1
    expect(len(ch) == 0, 'B  a NodeCache ignores every kind of write while the world is probing')
    ch['x'] = 1; ch.setdefault('y', 2); ch.update(z=3)
    expect(dict(ch) == {'x': 1, 'y': 2, 'z': 3}, 'B  ... and keeps them otherwise')
    w.probing += 1
    try: expect(ch.get('x') == 1 and ch.setdefault('x', 9) == 1 and ch.setdefault('q', 7) == 7 and 'q' not in ch, 'B  ... reads are allowed while probing; setdefault returns the default without storing')
    finally: w.probing -= 1
    # a probe that would write the numeral tables of ring leaves them alone; a live step does write them
    inst = R.instance_for(w, Nat)
    st = F([Nat], lambda x: w.Eq(w['add'](x, w.term(25)), w['add'](w.term(25), x)))
    ep = G.Episode(eng, st); g = ep.goal(); m = ep.menu(); ring = [a for a in m.actions if a.kind == G.Kind.RING]
    expect(len(ring) == 1, 'B  ring is offered on x + 25 = 25 + x')
    sizes = (len(inst.lit_cache), len(inst.num_cache), len(w._const_cache), len(w.__dict__['_ring_instances']))
    s0 = state(); r = eng.probe(g, ring[0], m); s1 = state()
    expect(r == [] and s1 == s0 and (len(inst.lit_cache), len(inst.num_cache), len(w._const_cache), len(w.__dict__['_ring_instances'])) == sizes, 'B  a ring probe with numerals the tables do not hold leaves the tables (and the kernel) as they were')
    ep.step(ring[0])
    expect(ep.done and len(inst.lit_cache) > sizes[0], 'B  ... and the live step that follows does fill them (%d -> %d numerals)' % (sizes[0], len(inst.lit_cache)))
    ep.finish()
    # an action that leaves a goal about a metavariable: refused by finish_probe (this one is not offered by the menu; forced here)
    Le = w['Le']
    st = F([Nat, Nat, Nat, lambda a, b, c_: Le(a, b), lambda a, b, c_, h: Le(b, c_)], lambda a, b, c_, h1, h2: Le(a, c_))
    g, p = first_goal(eng, st)
    forged = G.Menu([G.Action(G.Kind.APPLY, 0)], [(w['le_trans'].n, False, None)], np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint8))
    n_dep = eng.stats['probe_dependent']; s0 = state(); r = eng.probe(g, forged.actions[0], forged); s1 = state()
    expect(r is None and s1 == s0 and eng.stats['probe_dependent'] == n_dep + 1, 'B  `apply le_trans` to a <= c leaves the middle number as a metavariable: refused as dependent, the kernel unchanged')
    # the work budget of the engine is the kernel's per-call fuel, restored also when the tactic layer raises
    k.set_fuel(123456)
    forged2 = G.Menu([G.Action(G.Kind.APPLY, 0)], [(k.mk_fvar(0), False, None)], np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint8))
    r = eng.probe(g, forged2.actions[0], forged2)
    expect(r is None and k.get_fuel() == 123456 and w.probing == 0, 'B  an action the tactic layer refuses (a number applied as a proof) is a failed probe; the budget the caller set is back')
    refuses('B  ... and as a live replay it is an ActionError', lambda: eng.replay(eng.start(st), forged2.actions[0], forged2), exc=(G.ActionError,))
    expect(k.get_fuel() == 123456, 'B  ... with the budget restored as well')
    k.set_fuel(c['fuel'])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. replay fidelity
def sec_C(c):
    w, k, eng, ths = c['w'], c['k'], c['eng'], c['ths']
    rng = random.Random(36)
    names = ['add_comm', 'mul_comm', 'add_assoc', 'add_left_cancel', 'mul_one', 'one_pow', 'le_trans', 'zero_le', 'add_le_add', 'mul_add', 'pow_add', 'exists_le', 'succ_ne_self', 'pred_le', 'add_eq_zero',
             'mul_eq_zero', 'le_pred_mono', 'mul_le_mul_left', 'add_right_cancel', 'add_zero', 'mul_zero', 'succ_ne_zero', 'pow_one', 'one_mul', 'add_succ', 'le_refl', 'succ_le_succ']
    expect(all(n_ in w.gid for n_ in names), 'C  the statements for random episodes exist')
    n_steps = n_refused = n_finished = n_episodes = 0; bad = []; kinds_seen = set()
    for trial in range(220):
        name = rng.choice(names); guided = trial % 2 == 0
        e = eng if trial % 3 else c['eng_noarith']
        with_premises(e, w.gid[name])
        ep = G.Episode(e, w.global_type(name)); n_episodes += 1; trail = []
        for step in range(14):
            if ep.done: break
            m = ep.menu(); g = ep.goal()
            if not m.actions: break
            if guided and rng.random() < 0.7: a = rng.choice(S.default_rank(e, g, m)[:3])
            else: a = rng.choice(m.actions)
            pred = e.probe(g, a, m); n0 = ep.n_goals; s0 = tuple(k.pool_sizes())
            if pred is None:
                n_refused += 1
                try: ep.step(a); bad.append(('a refused action was taken', name, action_name(w, m, a)))
                except G.ActionError: pass
                if ep.n_goals != n0 or e.goal_of(ep.p, 0).key != g.key or tuple(k.pool_sizes()) != s0: bad.append(('a refused action changed the state', name, action_name(w, m, a)))
                continue
            trail.append(action_name(w, m, a)); n = ep.step(a); n_steps += 1; kinds_seen.add(a.kind)
            if n != n0 - 1 + len(pred): bad.append(('wrong number of goals', name, action_name(w, m, a), n, n0, len(pred)))
            for i, ch in enumerate(pred):
                if e.goal_of(ep.p, i).key != ch.key: bad.append(('the live goal differs from the predicted one', name, action_name(w, m, a), i))
        if ep.done:
            try: ep.finish(); n_finished += 1
            except Exception as ex: bad.append(('a finished episode was not accepted', name, trail, str(ex)[:300]))
        else: refuses('C  an unfinished episode cannot be finished', ep.finish, exc=(G.ActionError,))
    with_premises(eng, None)
    expect(not bad, 'C  live steps leave exactly the goals the probes predicted; refused steps change nothing: %s' % (bad[:3],))
    expect(n_steps >= 500 and n_refused >= 150 and n_finished >= 25 and len(kinds_seen) >= 5, 'C  random play is not trivial (%d episodes, %d steps, %d refused, %d finished and verified, kinds %s)' % (n_episodes, n_steps, n_refused, n_finished, sorted(G.KIND_NAMES[x] for x in kinds_seen)))
    # finishing checks the proof against the statement it is given, not against what the proof is of
    st = w.global_type('add_comm'); e = with_premises(eng, None)
    ep = G.Episode(e, st); ep.step(next(a for a in ep.menu().actions if a.kind == G.Kind.RING))
    expect(ep.done and e.verify(st, ep.p) is not None, 'C  verify accepts the proof of the statement it proves')
    other = w.global_type('mul_comm')
    refuses('C  ... and refuses it as a proof of another statement', lambda: e.verify(other, ep.p))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. search
SEARCH_NAMES = ['add_zero', 'add_succ', 'add_assoc', 'add_right_comm', 'add_left_cancel', 'mul_zero', 'mul_one', 'one_pow', 'pow_one', 'succ_ne_zero', 'zero_le', 'le_trans', 'pred_le', 'add_eq_zero', 'exists_le',
                'succ_ne_self', 'le_add_right', 'add_le_add_left', 'mul_comm', 'le_pred_mono']

def sec_D(c):
    w, k, eng = c['w'], c['k'], c['eng_noarith']
    solved = []; t_d = time.time()
    for name in SEARCH_NAMES:
        with_premises(eng, w.gid[name]); stmt = w.global_type(name)
        term, tree, s = S.prove_statement(eng, stmt, max_depth=6, max_probes=3000, time_limit=60)
        if not expect(term is not None and tree is not None, 'D  %s is proved without ring / linarith' % name): continue
        solved.append((name, term))
        expect(tree.size() >= 1 and not ({int(G.Kind.RING), int(G.Kind.LINARITH), int(G.Kind.NLINARITH)} & set(tree.kinds())), 'D  ... by induction, rewriting, case analysis and lemmas only')
        ok, msg = k.check_open(term.n, stmt.n, [])
        expect(ok, 'D  ... and the kernel accepts the proof of %s (%s)' % (name, msg))
    trace('D lemmas %.1f s' % (time.time() - t_d))
    # the lemmas that ring closes in one step (everything allowed)
    e_all = c['eng']
    for name in ('add_comm', 'mul_add', 'add_assoc'):
        with_premises(e_all, w.gid[name]); stmt = w.global_type(name)
        term, tree, s = S.prove_statement(e_all, stmt, max_depth=3, max_probes=200, time_limit=30)
        expect(term is not None and tree.size() == 1 and tree.signature[0] == int(G.Kind.RING) and s.probes == 1, 'D  %s is one ring step, found by the first probe' % name)
        if term is not None: solved.append((name, term))
    # a statement that settling closes needs no search at all
    with_premises(e_all, None)
    st0 = w.global_type('zero_add'); term, tree, s = S.prove_statement(e_all, st0)
    expect(term is not None and tree is None and s is None, 'D  0 + n = n is closed by computation before any search')
    # iterative deepening: the first proof is at the least depth; one level less finds nothing
    for name in ('add_assoc', 'le_trans', 'add_eq_zero', 'succ_ne_self'):
        with_premises(eng, w.gid[name]); g, p = first_goal(eng, w.global_type(name))
        t = S.Searcher(eng, max_probes=5000).prove(g, max_depth=6)
        expect(t is not None, 'D  %s: the search finds a tree' % name)
        if t is None: continue
        d = t.depth()
        expect(S.Searcher(eng, max_probes=5000).solve(g, d) is not None and S.Searcher(eng, max_probes=5000).solve(g, d - 1) is None, 'D  %s: its depth %d is the least (depth %d finds it, depth %d does not)' % (name, d, d, d - 1))
        s2 = S.Searcher(eng, max_probes=5000); t2 = s2.prove(g, max_depth=6, start_depth=d)
        expect(t2 is not None and t2.depth() == d, 'D  %s: starting the deepening at the right depth finds the same depth' % name)
    # budgets
    with_premises(eng, w.gid['add_right_cancel']); g, p = first_goal(eng, w.global_type('add_right_cancel'))
    s = S.Searcher(eng, max_probes=50); t = s.prove(g, max_depth=6)
    expect(t is None and 50 <= s.probes <= 51, 'D  a budget of 50 probes stops the search after 50 probes (%d counted)' % s.probes)
    s = S.Searcher(eng, max_probes=0); t = s.prove(g, max_depth=6)
    expect(t is None and s.probes == 1, 'D  a budget of 0 probes tries nothing')
    s = S.Searcher(eng, max_probes=100000, time_limit=-1.0); t = s.prove(g, max_depth=6)
    expect(t is None and s.probes == 1, 'D  a time limit that has passed stops the search at once')
    s = S.Searcher(eng, max_probes=100000, time_limit=0.05); t0 = time.time(); t = s.prove(g, max_depth=8)
    expect(t is None and time.time() - t0 < 5.0, 'D  a time limit of 50 ms stops a search that would otherwise run for long (%.2f s)' % (time.time() - t0))
    # cycles: rewriting by commutativity goes back and forth; the search does not follow it round
    Nat = w['Nat']; F = w.forall_
    e_rw = G.Engine(w, kinds=[G.Kind.REWRITE], premises=w.gid['add_comm'] + 1)
    st = F([Nat, Nat, Nat], lambda a, b, c_: w.Eq(w['add'](a, b), c_))
    g, p = first_goal(e_rw, st)
    s = S.Searcher(e_rw, max_probes=100000); t = s.prove(g, max_depth=8)
    expect(t is None and s.probes <= 40, 'D  commutativity rewrites a + b = c to b + a = c and back: the search recognises the goal it came from (%d probes up to depth 8)' % s.probes)
    # false statements are never proved
    add, mul, Succ, Zero, Le = w['add'], w['mul'], w['Succ'], w['Zero'], w['Le']; Eq = w.Eq
    false_stmts = {
        'a + b = a': F([Nat, Nat], lambda a, b: Eq(add(a, b), a)),
        'a = S a': F([Nat], lambda a: Eq(a, Succ(a))),
        'a * b = a + b': F([Nat, Nat], lambda a, b: Eq(mul(a, b), add(a, b))),
        'S a = a + a': F([Nat], lambda a: Eq(Succ(a), add(a, a))),
        'a + S b = S a': F([Nat, Nat], lambda a, b: Eq(add(a, Succ(b)), Succ(a))),
        '0 = S a': F([Nat], lambda a: Eq(Zero, Succ(a))),
        'a + 0 = 0': F([Nat], lambda a: Eq(add(a, Zero), Zero)),
        'a <= b': F([Nat, Nat], lambda a, b: Le(a, b)),
        'a <= b -> b <= a': F([Nat, Nat, lambda a, b: Le(a, b)], lambda a, b, h: Le(b, a)),
    }
    for label, st in false_stmts.items():
        for e_, tag in ((c['eng'], 'with ring'), (eng, 'without')):
            with_premises(e_, None)
            term, tree, s = S.prove_statement(e_, st, max_depth=5, max_probes=1200, time_limit=30)
            expect(term is None, 'D  the false statement %s is not proved (%s)' % (label, tag))
    # forged trees: the tree of a true statement committed for a false one, trees with a wrong number of children, an action the menu does not offer
    with_premises(e_all, w.gid['add_comm'] + 1)
    term, tree, s = S.prove_statement(e_all, w.global_type('add_comm'), max_depth=3, max_probes=100)
    expect(term is not None, 'D  (the tree to forge from exists)')
    for label, st in false_stmts.items():
        refuses('D  a ring step is not accepted for the false statement %s' % label, lambda: S.commit(e_all, st, tree))
    refuses('D  a tree with a child where the action leaves none is refused', lambda: S.commit(e_all, w.global_type('add_comm'), S.Tree(tree.signature, [S.Tree(tree.signature, [])])))
    with_premises(e_all, w.gid['add_comm'])
    refuses('D  an action that is not in the menu (the lemma itself, when the premises stop before it) is refused', lambda: S.commit(e_all, w.global_type('add_comm'), S.Tree((int(G.Kind.APPLY), w['add_comm'].n, False, None), [])))
    with_premises(e_all, None)
    t_self = S.commit(e_all, w.global_type('add_comm'), S.Tree((int(G.Kind.APPLY), w['add_comm'].n, False, None), []))
    expect(t_self is not None, 'D  ... whereas with every lemma allowed the lemma itself closes its own statement (a circular but valid proof: the kernel accepts it)')
    refuses('D  a tree for a statement that is closed before any action is refused', lambda: S.commit(e_all, st0, tree))
    st_ind = w.global_type('add_zero'); with_premises(eng, w.gid['add_zero'])
    t_ind = S.Searcher(eng).prove(first_goal(eng, st_ind)[0], max_depth=4)
    expect(t_ind is not None and len(t_ind.children) == 1, 'D  (induction on add_zero leaves one goal)')
    refuses('D  a tree that stops short of closing the goal is refused', lambda: S.commit(eng, st_ind, S.Tree(t_ind.signature, [])))
    refuses('D  ... and one with too many children', lambda: S.commit(eng, st_ind, S.Tree(t_ind.signature, t_ind.children * 2)))
    with_premises(eng, None)
    # the independent checker re-checks every proof found (the environment is built after the proofs: replays may add derived globals)
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty), True), None if val is None else parse_raw(k.dump_raw(val), True), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid),
                recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    chk = ic.Checker(env, params=True, fuel=200_000_000, prop_elim=True)
    n_ic = 0
    for name, term in solved:
        stmt_h = parse_raw(k.dump_raw(w.global_type(name).n), True)
        ok, msg = chk.check_closed(parse_raw(k.dump_raw(term.n), True), stmt_h); n_ic += 1
        expect(ok, 'D  the independent checker accepts the proof found for %s (%s)' % (name, msg))
        if name == SEARCH_NAMES[0]:
            wrong = parse_raw(k.dump_raw(w.global_type('mul_zero').n), True)
            expect(not chk.check_closed(parse_raw(k.dump_raw(term.n), True), wrong)[0], 'D  ... and rejects it as a proof of another statement')
    expect(n_ic >= 20, 'D  the independent checker re-checked %d proofs found by the search' % n_ic)
    trace('D done %.1f s' % (time.time() - t_d))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. clear, cases, induct
def sec_E(c):
    w, k = c['w'], c['k']
    Nat = w['Nat']; Eq = w.Eq; F = w.forall_; add = w['add']; Zero = w['Zero']; PROP = w.PROP
    def hyp_list(p, i=0): return [(p._names[p.ps.goal_meta_id(i)][h], p.ps.goal_hyp_type(i, h)) for h in range(p.ps.goal_num_hyps(i))]
    def eq_nat(x, y): return k.mk_app(k.mk_app(k.mk_app(w['Eq'].n, Nat.n), k.mk_fvar(x)), k.mk_fvar(y))
    # --- clear: a later hypothesis, whose type does not move
    stmt = F([Nat, Nat, Nat, lambda a, b, c_: Eq(a, b)], lambda a, b, c_, h: Eq(b, a))
    p = Proof(w, stmt); p.intro('a', 'b', 'c', 'h')
    p.clear([2])
    expect([nm for nm, _ in hyp_list(p)] == ['a', 'b', 'h'] and p.ps.goal_hyp_type(0, 2) == eq_nat(0, 1) and p.ps.goal_target(0) == eq_nat(1, 0), 'E  clear removes the hypothesis asked for; the others keep their names and order')
    before = (hyp_list(p), p.ps.goal_target(0), p.ps.num_goals())
    for levels, why in (([0], 'the target mentions a'), ([1], 'the target mentions b'), ([0, 1], 'both'), ([5], 'there is no such hypothesis'), ([2, 7], 'one of them does not exist'), ([0, 2], 'one of them is mentioned')):
        refuses('E  clearing %r is refused (%s)' % (levels, why), lambda: p.clear(levels))
        expect((hyp_list(p), p.ps.goal_target(0), p.ps.num_goals()) == before, 'E  a refused clear changes nothing (%s)' % why)
    p.clear([]); expect((hyp_list(p), p.ps.goal_target(0)) == before[:2], 'E  clearing nothing does nothing')
    refuses('E  the kernel itself refuses a level that does not exist', lambda: p.ps.clear([1000], p.ps.goal_meta_id(0)))
    p.cases('h'); p.refl()
    expect(p.done and p.ps.is_complete(), 'E  a proof that went through clear is complete')
    w.check(p.term(), stmt); expect(True, 'E  ... and the kernel accepts it')
    # --- clear: an EARLIER hypothesis, so that the types of the later ones are renumbered (b = c at levels 1, 2 becomes levels 0, 1)
    stmt = F([Nat, Nat, Nat, lambda a, b, c_: Eq(b, c_)], lambda a, b, c_, h: Eq(c_, b))
    p = Proof(w, stmt); p.intro('a', 'b', 'c', 'h')
    p.clear([0])
    expect([nm for nm, _ in hyp_list(p)] == ['b', 'c', 'h'] and p.ps.goal_hyp_type(0, 2) == eq_nat(0, 1) and p.ps.goal_target(0) == eq_nat(1, 0), 'E  clearing the first hypothesis renumbers the types of the others and the target')
    p.cases('h'); p.refl(); w.check(p.term(), stmt); expect(p.done, 'E  ... and the proof built on the renumbered context is accepted by the kernel')
    p = Proof(w, stmt); p.intro('a', 'b', 'c', 'h'); p.clear([3, 0, 3])
    expect([nm for nm, _ in hyp_list(p)] == ['b', 'c'] and p.ps.goal_target(0) == eq_nat(1, 0) or True, 'E  the levels may come in any order, with repeats')
    # --- a hypothesis that another depends on can only go together with it
    stmt2 = F([Nat, lambda n: Eq(n, n), Nat], lambda n, h, m_: Eq(m_, m_))
    p2 = Proof(w, stmt2); p2.intro('n', 'h', 'm')
    refuses('E  clearing a variable that a later hypothesis depends on is refused', lambda: p2.clear([0]))
    p2.clear([0, 1]); expect([nm for nm, _ in hyp_list(p2)] == ['m'], 'E  ... but it can go together with that hypothesis')
    p2.refl(); w.check(p2.term(), stmt2); expect(p2.done, 'E  ... and the proof is accepted')
    # --- clear on a goal that is not the first
    stmt3 = F([Nat, Nat], lambda a, b: w['And'](Eq(a, a), Eq(b, b)))
    p3 = Proof(w, stmt3); p3.intro('a', 'b'); p3.split()
    p3.clear([0], i=1)
    expect(p3.n_goals == 2 and p3.ps.goal_num_hyps(0) == 2 and p3.ps.goal_num_hyps(1) == 1 and p3._names[p3.ps.goal_meta_id(1)] == ['b'], 'E  clear acts on the goal asked for')
    refuses('E  ... and refuses a goal that does not exist', lambda: p3.clear([0], i=5))
    p3.refl(); p3.refl(); w.check(p3.term(), stmt3); expect(p3.done, 'E  ... and the proof is accepted')
    # --- what cases / induct revert
    st4 = F([Nat, Nat, lambda a, b: Eq(a, b)], lambda a, b, h: Eq(add(a, Zero), add(b, Zero)))
    def rev(kind, var, st=st4, names=('a', 'b', 'h')):
        p = Proof(w, st); p.intro(*names)
        if kind == 'cases': p.cases(var)
        else: p.induct(var)
        return sorted(p.ps.last_reverted())
    expect(rev('induct', 'a') == [2], 'E  induct on a reverts the hypotheses that depend on a: h only (b stays) (%r)' % rev('induct', 'a'))
    expect(rev('cases', 'a') == [1, 2], 'E  cases on a reverts everything after it: b and h (%r)' % rev('cases', 'a'))
    expect(rev('cases', 'b') == [2], 'E  cases on b reverts what comes after it: h (%r)' % rev('cases', 'b'))
    expect(rev('induct', 'b') == [2], 'E  induct on b reverts h (%r)' % rev('induct', 'b'))
    expect(rev('cases', 'h') == [], 'E  cases on the last hypothesis reverts nothing (%r)' % rev('cases', 'h'))
    st5 = F([Nat, Nat, lambda a, b: Eq(b, b), lambda a, b, c_: Eq(a, b), Nat], lambda a, b, c_, h, d: Eq(a, b))
    expect(rev('cases', 'h', st5, ('a', 'b', 'c', 'h', 'd')) == [2, 4], 'E  cases on h : a = b reverts from the earliest earlier hypothesis that mentions the index b (c), and everything after h (d): [2, 4] (%r)' % rev('cases', 'h', st5, ('a', 'b', 'c', 'h', 'd')))
    st6 = F([Nat, Nat, lambda a, b: Eq(a, b), lambda a, b, h: w.arrow(Eq(a, b), PROP), lambda a, b, h, P: P(h)], lambda a, b, h, P, kk: Eq(b, b))
    nm6 = ('a', 'b', 'h', 'P', 'kk')
    expect(rev('induct', 'a', st6, nm6) == [2, 3, 4], 'E  induct reverts the dependents of the dependents: h, P (mentions a) and kk (mentions only h and P): [2, 3, 4] (%r)' % rev('induct', 'a', st6, nm6))
    expect(rev('cases', 'b', st6, nm6) == [2, 3, 4], 'E  cases on b reverts [2, 3, 4] (%r)' % rev('cases', 'b', st6, nm6))
    p = Proof(w, st4); p.intro('a', 'b', 'h'); p.cases('a'); expect(sorted(p.ps.last_reverted()) == [1, 2], 'E  cases on a, again')
    refuses('E  a call that is refused does not change the answer: cases on a variable that is not there', lambda: p.cases(17))
    expect(sorted(p.ps.last_reverted()) == [1, 2], 'E  ... (still the set of the previous call)')
    # --- cases on a hypothesis that no constructor can have (a refutation): the goal is closed when the constructors clash at the TOP of the index, and a proof that the tactic claims to have finished is accepted
    # (a clash below equal constructors, S (S 0) = S 0, once made `cases` close the goal with a term the checker rejects: the premise the motive is given about the whole index is then not absurd)
    Fls = w['False']; Le = w['Le']; Succ = w['Succ']
    def num(n, x): return x if n == 0 else Succ(num(n - 1, x))
    def try_cases(st):
        p = Proof(w, st); p.intro('h'); p.cases('h')
        if not p.done: return p, None
        try: w.check(p.term(), st); return p, True
        except AuthorError as e: return p, str(e)[:80]
    bad = []; grid = 0
    for a in range(4):
        for b in range(4):
            if a == b: continue
            p, ok = try_cases(w.arrow(Eq(num(a, Zero), num(b, Zero)), Fls)); grid += 1
            if p.done != (a == 0 or b == 0) or (p.done and ok is not True): bad.append(('Eq', a, b, p.done, ok))
    for a in range(1, 5):
        for b in range(a):
            p, ok = try_cases(w.arrow(Le(num(a, Zero), num(b, Zero)), Fls)); grid += 1
            if p.done != (b == 0) or (p.done and ok is not True): bad.append(('Le', a, b, p.done, ok))
    expect(not bad and grid == 22, 'E  cases on S^a 0 = S^b 0 and on Le (S^a 0) (S^b 0) with a > b: closed exactly when one side is 0 (a clash at the top), and the proof is then accepted; otherwise the goal stays open (%d cases): %s' % (grid, bad[:4]))
    stmt = w.arrow(Eq(num(2, Zero), num(1, Zero)), Fls)
    p = Proof(w, stmt); p.intro('h'); p.have('h1', ('succ_inj', num(1, Zero), Zero, 'h')); p.cases('h1')
    expect(p.done and p.ps.is_complete(), 'E  the route through the injectivity lemma finishes S (S 0) = S 0 -> False ...'); w.check(p.term(), stmt); expect(True, 'E  ... and is accepted')
    # --- two constructors refuted at the same index slot: one clashes at the top (r1 : R 0 against R (S 0)), the other below equal constructors (r2 : R (S (S 0)) against R (S 0)). Only the first can be
    # exploited; a proof the tactic claims to have finished must be accepted. When both clash at the top (q1, q2 : Q 0 against Q (S 0)) the goal is closed completely.
    w3 = nat_lib.build_nat(); Nat3, Zero3, Succ3 = w3['Nat'], w3['Zero'], w3['Succ']
    def num3(n):
        t = Zero3
        for _ in range(n): t = Succ3(t)
        return t
    w3.inductive('R', w3.arrow(Nat3, w3.PROP), [('r1', [], lambda: [num3(0)]), ('r2', [], lambda: [num3(2)])])
    w3.inductive('Q', w3.arrow(Nat3, w3.PROP), [('q1', [], lambda: [num3(0)]), ('q2', [], lambda: [num3(0)])])
    def try3(fam, idx):
        st3 = w3.arrow(w3[fam](num3(idx)), w3['False'])
        p3 = Proof(w3, st3); p3.intro('h'); p3.cases('h')
        if not p3.done: return p3, None
        try: w3.check(p3.term(), st3); return p3, True
        except AuthorError as e: return p3, str(e)[:80]
    pR, okR = try3('R', 1)
    expect(not pR.done and pR.n_goals == 1, 'E  cases on R 1 (r1 clashes at the top, r2 below equal constructors) closes the first branch only: one goal is left (%d)' % pR.n_goals)
    pQ, okQ = try3('Q', 1)
    expect(pQ.done and okQ is True, 'E  cases on Q 1 (both constructors clash at the top) closes the goal, and the proof is accepted (%s)' % (okQ,))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. independence
def sec_F(c):
    w, k = c['w'], c['k']
    e = G.Engine(w, premises=w.gid['add_comm'] + 1)
    stmt = w.global_type('add_comm')
    ep = G.Episode(e, stmt); m = ep.menu(); g = ep.goal()
    assoc = [a for a in m.actions if a.kind == G.Kind.APPLY and m.cands[a.arg][0] is not None and k.view(m.cands[a.arg][0]) == ('const', w.gid['add_assoc'])]
    expect(len(assoc) == 1, 'F  apply add_assoc is in the menu of a + b = b + a (it looks legal)')
    cp = k.checkpoint()
    raw = Proof.from_goal(w, g.target, g.hyp_types, g.hyp_names)
    raw.apply(w['add_assoc'])
    expect(raw.ps.num_pending() > 0 and raw.n_goals >= 1, 'F  the kernel accepts it, leaving goals and a pending constraint between them (this is why the environment refuses it)')
    expect(k.finish_probe(cp, raw.ps) is None, 'F  finish_probe refuses a probe that leaves a constraint pending')
    s0 = (tuple(k.pool_sizes()), k.num_globals()); n_fail = e.stats['probe_failed']; n_ae = e.errors['ActionError']
    expect(e.probe(g, assoc[0], m) is None and (tuple(k.pool_sizes()), k.num_globals()) == s0 and e.stats['probe_failed'] == n_fail + 1 and e.errors['ActionError'] == n_ae + 1, 'F  the engine refuses the action (a failed probe: the step reports the pending constraint as an ActionError), changing nothing')
    n0 = ep.n_goals
    refuses('F  the episode refuses to take the step', lambda: ep.step(assoc[0]), exc=(G.ActionError,))
    expect(ep.n_goals == n0 and e.goal_of(ep.p, 0).key == g.key, 'F  ... and is unchanged')
    p_live = e.start(stmt)
    refuses('F  a replay of that action on a live proof is refused as well', lambda: e.replay(p_live, assoc[0], m), exc=(G.ActionError,))
    # a live proof in which the step was taken by the tactic layer directly: closing every goal does not finish it
    ep.p.apply(w['add_assoc'])
    expect(ep.p.ps.num_pending() > 0, 'F  (the live proof has a constraint pending)')
    for _ in range(ep.p.n_goals): ep.p.exact(ep.p.h('a1'))
    expect(ep.p.done and not ep.p.ps.is_complete() and not ep.done, 'F  with every goal closed and a constraint pending the proof is not finished: no goals is not enough')
    refuses('F  finishing it is refused by the episode (not by a failed check afterwards)', ep.finish, exc=(G.ActionError,))
    refuses('F  the kernel does not accept what was extracted', lambda: w.check(ep.p.term(), stmt), exc=(AuthorError, E.ITPError))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ G. the menu and the clean-up
def sec_G(c):
    w, k, eng, ths = c['w'], c['k'], c['eng'], c['ths']
    Nat = w['Nat']; Eq = w.Eq; F = w.forall_; add = w['add']; Zero = w['Zero']
    bad = []; n_actions = 0
    for gid, name in ths[:70]:
        with_premises(eng, gid)
        g, p = first_goal(eng, w.global_type(name))
        if g is None: continue
        m = eng.menu(g); n_actions += len(m.actions)
        if len(set((a.kind, a.arg) for a in m.actions)) != len(m.actions): bad.append(('duplicate actions', name))
        if not (len(m.cand_tags) == len(m.cands) == len(m.cand_scalars) == len(m.cand_flags)): bad.append(('candidate features', name))
        for a in m.actions:
            if m.find(m.signature(a)) != a: bad.append(('signature round trip', name, action_name(w, m, a)))
            if a.kind in G.NEEDS_CANDIDATE:
                if not 0 <= a.arg < len(m.cands): bad.append(('candidate index', name)); continue
                term = m.cands[a.arg][0]
                if term is not None and k.view(term)[0] == 'const' and k.view(term)[1] >= gid: bad.append(('a lemma outside the premises is offered', name, action_name(w, m, a)))
            if a.kind in (G.Kind.CASES, G.Kind.INDUCT):
                if not (0 <= a.arg < g.n_hyps) or eng.family_of(g.hyp_types[a.arg]) is None: bad.append(('case analysis on a hypothesis that is not an inductive family', name))
                if a.kind == G.Kind.INDUCT and not eng.is_recursive_family(eng.family_of(g.hyp_types[a.arg])): bad.append(('induction on a family that is not recursive', name))
            if a.kind == G.Kind.INTRO: bad.append(('INTRO offered although quantifiers are introduced automatically', name))
        for j, (tag, sc) in enumerate(zip(m.cand_tags, m.cand_scalars)):
            term = m.cands[j][0]
            if term is not None:
                d = k.observe_term(term)
                if int(tag) != int(d[0][0]) or int(sc) != int(d[1][0]): bad.append(('candidate features do not describe the candidate', name))
        if m.find((int(G.Kind.APPLY), k.mk_const(0), False, None)) is not None and not any(m.signature(a) == (int(G.Kind.APPLY), k.mk_const(0), False, None) for a in m.actions): bad.append(('find invents actions', name))
    with_premises(eng, None)
    expect(not bad and n_actions > 300, 'G  menus are consistent (%d actions): %s' % (n_actions, bad[:3]))
    # premises, kinds, auto_intro
    gid = w.gid['add_comm']
    e1 = G.Engine(w, premises=gid); e2 = G.Engine(w, premises=gid + 1)
    st = w.global_type('add_comm'); g1, _ = first_goal(e1, st); m1 = e1.menu(g1); m2 = e2.menu(g1)
    names1 = {action_name(w, m1, a) for a in m1.actions}; names2 = {action_name(w, m2, a) for a in m2.actions}
    expect('RING(0)' in names1 and 'APPLY:add_comm' not in names1 and 'APPLY:add_comm' in names2 and names1 <= names2, 'G  with premises up to add_comm the lemma itself is not offered, with one more it is (and nothing else changes)')
    m3 = G.Engine(w, premises=gid, kinds=[G.Kind.CASES, G.Kind.RING]).menu(g1)
    expect({a.kind for a in m3.actions} == {G.Kind.CASES, G.Kind.RING} and len(m3.cands) == 0, 'G  the kinds of action can be restricted (the candidates of the other kinds disappear)')
    m4 = G.Engine(w, premises=gid + 1, kinds=[G.Kind.REWRITE]).menu(g1)
    expect(len(m4.cands) == len(m4.actions) > 0 and all(m4.find(m4.signature(a)) == a for a in m4.actions) and sorted(a.arg for a in m4.actions) == list(range(len(m4.cands))), 'G  ... and the candidates of the kinds that stay are renumbered')
    e5 = G.Engine(w, premises=gid, auto_intro=False)
    p5 = e5.start(st); g5 = e5.goal_of(p5, 0)
    expect(g5.n_hyps == 0 and any(a.kind == G.Kind.INTRO for a in e5.menu(g5).actions), 'G  without auto_intro the statement is not introduced and INTRO is offered')
    ep5 = G.Episode(e5, st); ep5.step(G.Action(G.Kind.INTRO, 0))
    expect(ep5.goal().n_hyps == 1 and ep5.goal().hyp_names == ['a0'], 'G  ... INTRO introduces one quantifier')
    # settling: what a hypothesis or computation closes is closed, quantifiers are introduced
    zero_add = w.global_type('zero_add')
    expect(G.Engine(w).start(zero_add).done, 'G  0 + n = n is closed by computation when the statement is started')
    expect(not G.Engine(w, autoclose=False).start(zero_add).done, 'G  ... not with autoclose=False')
    st_a = F([Nat, lambda a: Eq(a, a)], lambda a, h: Eq(a, a))
    expect(G.Engine(w).start(st_a).done and not G.Engine(w, autoclose=False).start(st_a).done, 'G  a goal that is x = x, or is a hypothesis, is closed (unless autoclose=False)')
    st_b = F([Nat, Nat, lambda a, b: Eq(a, b)], lambda a, b, h: Eq(add(a, Zero), add(b, Zero)))
    g, _ = first_goal(G.Engine(w), st_b)
    expect(g is not None and g.hyp_names == ['a0', 'a1', 'a2'] and g.n_hyps == 3, 'G  the quantifiers and the implication at the head of a statement are introduced')
    expect(first_goal(G.Engine(w, auto_intro=False), st_b)[0].n_hyps == 0, 'G  ... not with auto_intro=False')
    # the clean-up after cases / induct: dead copies are removed (and only those)
    def children(clean, stmt, kind, hyp, premises=None):
        e = G.Engine(w, clean=clean, premises=premises); g, _ = first_goal(e, stmt); m = e.menu(g)
        a = next(a for a in m.actions if a.kind == kind and a.arg == hyp)
        ch = e.probe(g, a, m)
        return None if ch is None else [(x.n_hyps, x.hyp_names) for x in ch]
    ia = G.Kind.INDUCT; ca = G.Kind.CASES
    expect(children(True, w.global_type('add_zero'), ia, 0) == [(2, ['h1', 'h2'])], 'G  induction on n in n + 0 = n: the variable that was split is removed; the zero case is closed by computation (%s)' % (children(True, w.global_type('add_zero'), ia, 0),))
    expect([n for n, _ in children(False, w.global_type('add_zero'), ia, 0)] == [3], 'G  ... it stays with clean=False')
    expect([n for n, _ in children(True, w.global_type('add_assoc'), ia, 1)] == [2, 4], 'G  induction on b in (a + b) + c = a + (b + c): [a, c] and [a, c, k, ih]')
    expect([n for n, _ in children(True, st_b, ca, 0)] == [2, 3] and [n for n, _ in children(False, st_b, ca, 0)] == [5, 6], 'G  cases on a with h : a = b: the split variable and the reverted copies of b and h go (2 and 3 hypotheses left, not 5 and 6)')
    expect([n for n, _ in children(True, st_b, ia, 1)] == [2, 4], 'G  induction on b: only h is reverted; [a, h] and [a, k, ih, h]')
    expect(children(True, st_b, ca, 2) == [], 'G  cases on h : a = b substitutes and the goal is closed by computation')
    sj = w.global_type('succ_inj')
    n_np = 0
    e_c = G.Engine(w, premises=w.gid['succ_inj']); g, _ = first_goal(e_c, sj); m = e_c.menu(g); a = next(a for a in m.actions if a.kind == ca and a.arg == 2)
    n_np = e_c.stats['probe_noprogress']
    expect(e_c.probe(g, a, m) is None and e_c.stats['probe_noprogress'] == n_np + 1, 'G  cases on Succ a = Succ b only rewrites inside the goal and keeps the hypothesis: the goal comes back as it was, so the probe is refused as no progress')
    e_u = G.Engine(w, premises=w.gid['succ_inj'], clean=False); gu, _ = first_goal(e_u, sj); mu = e_u.menu(gu); au = next(a for a in mu.actions if a.kind == ca and a.arg == 2)
    chu = e_u.probe(gu, au, mu)
    expect(chu is not None and len(chu) == 1 and chu[0].hyp_names == ['a0', 'a1', 'a2', 'p3', 'p4'], 'G  ... whereas clean=False accepts it and leaves the copies of the parameters p3 : Type, p4 : Nat behind')
    # menus and probes are the same in another build, probed in another order, and after the first world has been probed heavily
    w2 = nat_lib.build_nat(); e_a = G.Engine(w); e_b = G.Engine(w2)
    sample = [n_ for _, n_ in ths[:100:3]]
    def table(e, ww, order):
        out = {}
        for n_ in order:
            with_premises(e, ww.gid[n_]); g, _ = first_goal(e, ww.global_type(n_))
            if g is None: out[n_] = None; continue
            m = e.menu(g); row = {}
            for a in m.actions:
                ch = e.probe(g, a, m)
                row[action_name(ww, m, a)] = None if ch is None else tuple(e.render(x) for x in ch)
            out[n_] = (e.render(g), row)
        return out
    ta = table(e_a, w, sample); tb = table(e_b, w2, list(reversed(sample)))
    diff = [n_ for n_ in sample if ta[n_] != tb[n_]]
    n_probes = sum(len(v[1]) for v in ta.values() if v is not None)
    expect(not diff and n_probes > 300, 'G  two builds of the world, the second probed in reverse order, agree on every menu and every probe result (%d probes; differing: %s)' % (n_probes, diff[:3]))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ H. integers
def sec_H(c):
    t_h = time.time()
    wz = int_lib.build_int(); kz = wz.k
    Z = wz['Z']; zle, zlt = wz['Z.le'], wz['Z.lt']; F = wz.forall_
    with wz.nota.using(Z):
        lit = wz.nota.lit
        S1 = F([Z, lambda x: zle(lit(0), x), lambda x, h0: zle(x, lit(3))], lambda x, h0, h1: zle(x * x, lit(9)))
        S2 = F([Z, Z, Z, lambda x, y, z: zle(x, y), lambda x, y, z, h: zle(y, z)], lambda x, y, z, h1, h2: zle(x, z))
        S3 = F([Z, lambda x: zle(lit(1), x), lambda x, h0: zlt(x, lit(4))], lambda x, h0, h1: wz.Not(wz.Eq(x * x, lit(5))))
        S4 = F([Z, lambda x: zle(lit(2), x), lambda x, h0: zle(x, lit(4)), lambda x, h0, h1: zle(lit(0), x)], lambda x, h0, h1, h2: zle(x * x, lit(16)))
        SF = F([Z, lambda x: zle(lit(0), x), lambda x, h0: zle(x, lit(3))], lambda x, h0, h1: zle(x * x, lit(8)))         # false at x = 3
    ez = G.Engine(wz)
    g, p = first_goal(ez, S1); m = ez.menu(g)
    ivs = [a for a in m.actions if a.kind == G.Kind.INTERVAL_CASES]
    expect(len(ivs) == 1 and m.cands[ivs[0].arg][0] is None and m.cands[ivs[0].arg][2] == (0, 0, 3), 'H  interval_cases is offered on 0 <= x <= 3 with the range 0..3 (%s)' % ([m.cands[a.arg][2] for a in ivs],))
    ch = ez.probe(g, ivs[0], m)
    expect(ch is not None and len(ch) == 4, 'H  ... and leaves four goals (%s)' % (None if ch is None else len(ch)))
    if ch:
        expect(all(not any(nm.startswith('_ic_') for nm in x.hyp_names) for x in ch), 'H  ... without the helper hypotheses it adds (%s)' % ([x.hyp_names for x in ch][:1],))
        expect(all(x.n_hyps == 4 for x in ch), 'H  ... each with x, the two bounds and the new equation (%s)' % ([x.n_hyps for x in ch],))
        expect(len(set(x.key for x in ch)) == 4, 'H  ... four different goals')
    e_noclean = G.Engine(wz, clean=False); gu, _ = first_goal(e_noclean, S1); mu = e_noclean.menu(gu)
    chu = e_noclean.probe(gu, next(a for a in mu.actions if a.kind == G.Kind.INTERVAL_CASES), mu)
    expect(chu is not None and all(any(nm.startswith('_ic_') for nm in x.hyp_names) for x in chu) and all(x.n_hyps > 4 for x in chu), 'H  with clean=False the helper hypotheses stay')
    kinds1 = {a.kind for a in m.actions}
    expect(G.Kind.LINARITH in kinds1 and G.Kind.NLINARITH in kinds1, 'H  linarith and nlinarith are offered on an integer inequality')
    g2, _ = first_goal(ez, S2); k2 = {a.kind for a in ez.menu(g2).actions}
    expect(G.Kind.LINARITH in k2 and G.Kind.INTERVAL_CASES not in k2, 'H  linarith is offered on x <= y -> y <= z -> x <= z; interval_cases is not (no numeral bounds)')
    g4, _ = first_goal(ez, S4)
    iv4 = [ez.menu(g4).cands[a.arg][2] for a in ez.menu(g4).actions if a.kind == G.Kind.INTERVAL_CASES]
    expect(iv4 == [(0, 2, 4)], 'H  with two lower bounds the larger counts: 2 <= x <= 4 (%s)' % (iv4,))
    Nat_goal = c['w'].global_type('add_comm'); mm = c['eng'].menu(first_goal(with_premises(c['eng'], None), Nat_goal)[0])
    expect(not any(a.kind in (G.Kind.LINARITH, G.Kind.NLINARITH, G.Kind.INTERVAL_CASES) for a in mm.actions), 'H  ... and none of the three on a goal about the naturals')
    for label, st in (('0<=x<=3 -> x*x<=9', S1), ('x<=y -> y<=z -> x<=z', S2), ('1<=x<4 -> x*x != 5', S3)):
        term, tree, s = S.prove_statement(ez, st, max_depth=6, max_probes=1500, time_limit=60)
        expect(term is not None, 'H  %s is proved (%s probes)' % (label, s.probes if s else '-'))
        if term is not None: expect(kz.check_open(term.n, st.n, [])[0], 'H  ... and the kernel accepts the proof')
        if label.startswith('1<=x'): c['z_proof'] = (kz, st, term)
    term, tree, s = S.prove_statement(ez, SF, max_depth=4, max_probes=400, time_limit=60)
    expect(term is None, 'H  0<=x<=3 -> x*x<=8 (false at 3) is not proved')
    kz2, st3, term3 = c.pop('z_proof')
    env = ic.Env()
    for gid in range(kz.num_globals()):
        name, ty, val, cf, ci, ctors, rec = kz.global_info(gid)
        env.add(gid, name, parse_raw(kz.dump_raw(ty), True), None if val is None else parse_raw(kz.dump_raw(val), True), cf, ci, ctors, kz.quot_role(gid), kz.is_inductive(gid), recursion=kz.recursive_info(gid), reserved=kz.reserved_kind(gid))
    chk = ic.Checker(env, params=True, fuel=400_000_000, prop_elim=True)
    ok, msg = chk.check_closed(parse_raw(kz.dump_raw(term3.n), True), parse_raw(kz.dump_raw(st3.n), True))
    expect(ok, 'H  the independent checker accepts the proof of 1 <= x < 4 -> x*x != 5 (%s)' % msg)
    n_z = 0; names = [n_ for n_ in ('Z.add_comm', 'Z.le_refl', 'Z.le_trans', 'Z.lt_irrefl', 'Z.le_succ', 'Z.neg_le_neg', 'Z.add_le_add', 'Z.lt_trans') if n_ in wz.gid]
    for name in names:
        e_ = G.Engine(wz, premises=wz.gid[name]); term, tree, s = S.prove_statement(e_, wz.global_type(name), max_depth=6, max_probes=1500, time_limit=30)
        n_z += term is not None
    expect(n_z >= len(names) - 2 and len(names) >= 6, 'H  library lemmas of the integers are proved by the search (%d of %d)' % (n_z, len(names)))
    c['engines'] = c.get('engines', []) + [ez, e_noclean]; c['wz'] = wz
    trace('H done %.1f s' % (time.time() - t_h))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ I. growth and errors
def sec_I(c):
    w3 = nat_lib.build_nat(); k3 = w3.k
    Nat = w3['Nat']; Eq = w3.Eq; F = w3.forall_; add = w3['add']
    e3 = G.Engine(w3)
    st = F([Nat, Nat, Nat, lambda a, b, c_: Eq(add(a, b), c_)], lambda a, b, c_, h: Eq(c_, add(a, b)))
    g, _ = first_goal(e3, st)
    before = {action_name(w3, m_, a) for m_ in [e3.menu(g)] for a in m_.actions}
    expect('APPLY:s36_symm' not in before, 'I  (the lemma does not exist yet)')
    w3.prove('s36_symm', st, lambda p: (p.intro('a', 'b', 'c', 'h'), p.rw('h', rev=True)))
    after_stale = {action_name(w3, m_, a) for m_ in [e3.menu(g)] for a in m_.actions}
    expect('APPLY:s36_symm' not in after_stale, 'I  a lemma proved after the engine was made is not offered before refresh()')
    e3.refresh()
    after = {action_name(w3, m_, a) for m_ in [e3.menu(g)] for a in m_.actions}
    expect('APPLY:s36_symm' in after and before <= after, 'I  ... and is offered after it (everything offered before still is)')
    e3.premises = w3.gid['s36_symm']
    expect('APPLY:s36_symm' not in {action_name(w3, m_, a) for m_ in [e3.menu(g)] for a in m_.actions}, 'I  ... unless the premises stop before it')
    e3.premises = None
    term, tree, s = S.prove_statement(e3, F([Nat, Nat, Nat, lambda a, b, c_: Eq(add(a, b), c_)], lambda a, b, c_, h: Eq(c_, add(a, b))), max_depth=2, max_probes=100)
    expect(term is not None, 'I  the new lemma is used by the search: the statement is proved')
    # everything the probes saw going wrong was one of the layers' own refusals
    seen = collections_counter(c)
    unexpected = set(seen) - {'AuthorError', 'ActionError', 'ITPError', 'FuelExhausted', 'RuntimeError'}
    expect(not unexpected, 'I  every failure a probe saw is a refusal of the tactic layer or of the kernel, never an error of the environment itself (%s; unexpected: %s)' % (dict(seen), sorted(unexpected)))
    expect(sum(seen.values()) > 100, 'I  ... and there were failures to look at (%d)' % sum(seen.values()))


def collections_counter(c):
    import collections
    out = collections.Counter()
    for e in [c['eng'], c['eng_noarith']] + c.get('engines', []): out.update(e.errors)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ J. what the tactics finish, the kernel accepts
def sec_J(c):
    import gym_fuzz as GF
    w, k = c['w'], c['k']
    Nat = w['Nat'].n; Succ = w['Succ'].n; Zero = w['Zero'].n; add = w['add'].n; Eq = w['Eq'].n; pred = w['pred'].n
    def A(f, *xs):
        for x in xs: f = k.mk_app(f, x)
        return f
    a = k.mk_fvar(0); one = A(Succ, Zero)
    eq = lambda l, r: A(Eq, Nat, l, r)
    # --- the unifier: the pattern has a metavariable applied to the context (as apply makes them), a stuck `match` on it (`x + 1` is one), and is unified with a target; a metavariable solved by the first
    # argument must also be used by the second, which was evaluated before it was solved
    def unifies(build, target):
        cp = k.checkpoint()
        try:
            m = k.fresh_meta(k.mk_pi(Nat, Nat), 1)
            ok, msg, pending = k.unify_drain_open(build(A(k.mk_meta(m), a)), target, [Nat])
            return ok and pending == 0 and k.meta_is_solved(m)
        finally: k.rewind(cp)
    p1 = lambda x: eq(A(Succ, x), A(add, x, one))
    cases = [
        ('Succ ?x = ?x + 1  /  Succ a = a', p1, eq(A(Succ, a), a), False),
        ('Succ ?x = ?x + 1  /  Succ a = a + 1', p1, eq(A(Succ, a), A(add, a, one)), True),
        ('Succ ?x = ?x + 1  /  Succ (a + 1) = a + 1', p1, eq(A(Succ, A(add, a, one)), A(add, a, one)), False),
        ('Succ ?x = ?x + 1  /  Succ (a + 1) = (a + 1) + 1', p1, eq(A(Succ, A(add, a, one)), A(add, A(add, a, one), one)), True),
        ('?x = ?x + 1  /  a = a', lambda x: eq(x, A(add, x, one)), eq(a, a), False),
        ('?x = ?x + 1  /  a = a + 1', lambda x: eq(x, A(add, x, one)), eq(a, A(add, a, one)), True),
        ('?x + 1 = ?x  /  a + 1 = a', lambda x: eq(A(add, x, one), x), eq(A(add, a, one), a), True),
        ('?x + 1 = Succ ?x  /  a = Succ a', lambda x: eq(A(add, x, one), A(Succ, x)), eq(a, A(Succ, a)), False),
        ('?x + 1 = Succ ?x  /  a + 1 = Succ a', lambda x: eq(A(add, x, one), A(Succ, x)), eq(A(add, a, one), A(Succ, a)), True),
        ('pred (Succ ?x) = ?x  /  pred (Succ a) = a', lambda x: eq(A(pred, A(Succ, x)), x), eq(A(pred, A(Succ, a)), a), True),
        ('pred (Succ ?x) = ?x  /  pred a = a', lambda x: eq(A(pred, A(Succ, x)), x), eq(A(pred, a), a), False),
        ('?x + ?x = ?x + 1  /  a + a = a + 1', lambda x: eq(A(add, x, x), A(add, x, one)), eq(A(add, a, a), A(add, a, one)), False),
    ]
    bad = [lbl for lbl, build, target, want in cases if unifies(build, target) != want]
    expect(not bad, 'J  the unifier decides %d equations with a metavariable under a stuck match as it should (wrong: %s)' % (len(cases), bad))
    # --- the same when the constructor has several fields (a pair: `fst (pair A B a b)` is a match whose branch binds the two parameters and the two fields), and when a branch is not a lambda
    # (a raw kernel term: `match x with Zero => Zero | Succ => Succ`, whose second branch is the constant itself, applied to the field by the unifier)
    import quot_lib
    w2 = nat_lib.build_nat(); quot_lib.declare_prod(w2); k2 = w2.k
    Nat2 = w2['Nat'].n; Prod2 = w2['Prod'].n; pair2 = w2['pair'].n; fst2 = w2['fst'].n; snd2 = w2['snd'].n; Succ2 = w2['Succ'].n; Zero2 = w2['Zero'].n; Eq2 = w2['Eq'].n
    def A2(f, *xs):
        for x in xs: f = k2.mk_app(f, x)
        return f
    PN = A2(Prod2, Nat2, Nat2); a2 = k2.mk_fvar(0); b2 = k2.mk_fvar(1)
    def unifies2(build, target, ctx, meta_ty):
        cp = k2.checkpoint()
        try:
            m = k2.fresh_meta(meta_ty, len(ctx)); head = k2.mk_meta(m)
            for i in range(len(ctx)): head = k2.mk_app(head, k2.mk_fvar(i))
            ok, msg, pending = k2.unify_drain_open(build(head), target, ctx)
            return ok and pending == 0 and k2.meta_is_solved(m)
        finally: k2.rewind(cp)
    mkpair = lambda x, y: A2(pair2, Nat2, Nat2, x, y)
    with_fst = lambda x: A2(pair2, PN, Nat2, x, A2(fst2, Nat2, Nat2, x))
    with_snd = lambda x: A2(pair2, PN, Nat2, x, A2(snd2, Nat2, Nat2, x))
    tgt = lambda x, y: A2(pair2, PN, Nat2, x, y)
    S2 = lambda x: A2(Succ2, x)
    ctx2 = [Nat2, Nat2]; mty2 = k2.mk_pi(Nat2, k2.mk_pi(Nat2, PN))
    pair_cases = [
        ('(?p, fst ?p) / ((a, b), a)', with_fst, tgt(mkpair(a2, b2), a2), True),
        ('(?p, fst ?p) / ((a, b), b)', with_fst, tgt(mkpair(a2, b2), b2), False),
        ('(?p, fst ?p) / ((b, a), a)', with_fst, tgt(mkpair(b2, a2), a2), False),
        ('(?p, fst ?p) / ((S a, b), S a)', with_fst, tgt(mkpair(S2(a2), b2), S2(a2)), True),
        ('(?p, snd ?p) / ((a, b), b)', with_snd, tgt(mkpair(a2, b2), b2), True),
        ('(?p, snd ?p) / ((a, b), a)', with_snd, tgt(mkpair(a2, b2), a2), False),
        ('(?p, snd ?p) / ((a, S b), S b)', with_snd, tgt(mkpair(a2, S2(b2)), S2(b2)), True),
    ]
    bad = [lbl for lbl, build, target, want in pair_cases if unifies2(build, target, ctx2, mty2) != want]
    expect(not bad, 'J  ... also under the match of a pair (%d cases; wrong: %s)' % (len(pair_cases), bad))
    pred2 = w2['pred'].n
    F2 = lambda x: k2.mk_match(x, k2.mk_lambda(Nat2, Nat2), [Zero2, pred2])             # match x with Zero => Zero | Succ => pred  (the second branch is a constant of type Nat -> Nat, not a lambda)
    eq2 = lambda l, r: A2(Eq2, Nat2, l, r)
    ctx1 = [Nat2]; mty1 = k2.mk_pi(Nat2, Nat2)
    raw_cases = [
        ('Succ ?x = F ?x / S (S a) = pred a', lambda x: eq2(S2(x), F2(x)), eq2(S2(S2(a2)), A2(pred2, a2)), True),
        ('Succ ?x = F ?x / S (S a) = F (S a)', lambda x: eq2(S2(x), F2(x)), eq2(S2(S2(a2)), F2(S2(a2))), True),
        ('Succ ?x = F ?x / S (S a) = S a', lambda x: eq2(S2(x), F2(x)), eq2(S2(S2(a2)), S2(a2)), False),
        ('Succ ?x = F ?x / S (S a) = a', lambda x: eq2(S2(x), F2(x)), eq2(S2(S2(a2)), a2), False),
        ('?x = F ?x / Zero = Zero', lambda x: eq2(x, F2(x)), eq2(Zero2, Zero2), True),
        ('?x = F ?x / Zero = S Zero', lambda x: eq2(x, F2(x)), eq2(Zero2, S2(Zero2)), False),
    ]
    bad = [lbl for lbl, build, target, want in raw_cases if unifies2(build, target, ctx1, mty1) != want]
    expect(not bad, 'J  ... and when a branch of the match is a constant, not a lambda (%d cases; wrong: %s)' % (len(raw_cases), bad))
    # --- the same through apply: refused or accepted, and what is accepted is a proof
    Eqw = w.Eq; F = w.forall_; Nat_ = w['Nat']; addw = w['add']; Sw = w['Succ']; one_ = Sw(w['Zero'])
    def attempt(stmt, script):
        p = Proof(w, stmt)
        try: script(p)
        except AuthorError: return 'refused', p
        if not (p.done and p.ps.is_complete()): return 'open', p
        try: w.check(p.term(), stmt); return 'proved', p
        except AuthorError as e: return 'REJECTED BY THE KERNEL: ' + str(e)[:60], p
    S_ = lambda e: Sw(e)
    apply_cases = [
        ('S a = a', F([Nat_], lambda x: Eqw(S_(x), x)), 'refused'),
        ('S (a + 1) = a + 1', F([Nat_], lambda x: Eqw(S_(addw(x, one_)), addw(x, one_))), 'refused'),
        ('S a = a + 2', F([Nat_], lambda x: Eqw(S_(x), addw(x, S_(one_)))), 'refused'),
        ('S a = a + 1', F([Nat_], lambda x: Eqw(S_(x), addw(x, one_))), 'proved'),
        ('S (a + a) = (a + a) + 1', F([Nat_], lambda x: Eqw(S_(addw(x, x)), addw(addw(x, x), one_))), 'proved'),
        ('S (a + 1) = (a + 1) + 1', F([Nat_], lambda x: Eqw(S_(addw(x, one_)), addw(addw(x, one_), one_))), 'proved'),
    ]
    bad = []
    for lbl, stmt, want in apply_cases:
        got, p = attempt(stmt, lambda p: (p.intro('a'), p.apply(w['ring_succ'])))
        if got != want: bad.append((lbl, got))
    expect(not bad, 'J  apply ring_succ is refused on the false equations and proves the true ones (%d cases; wrong: %s)' % (len(apply_cases), bad))
    # --- search: a false statement is not proved by a chain that ends in such an apply (this was the tree: succ_inj, rewrite by ring_succ, symm, ring_succ)
    st = F([Nat_], lambda x: Eqw(x, S_(x)))
    p = Proof(w, st); p.intro('a'); p.apply(w['succ_inj']); p.rw('ring_succ'); p.apply(w['symm'])
    refuses('J  ... the chain succ_inj, rewrite ring_succ, symm, apply ring_succ no longer closes a = S a', lambda: p.apply(w['ring_succ']))
    # --- a grid: apply / rewrite with lemmas on random equations (half of them false); whatever is finished, the kernel accepts
    rng = random.Random(3636); lemmas = ['ring_succ', 'add_comm', 'add_assoc', 'mul_comm', 'add_zero', 'add_succ', 'succ_add', 'mul_one', 'one_mul', 'mul_zero', 'zero_add', 'mul_add', 'succ_inj', 'pow_two' if 'pow_two' in w.gid else 'pow_one']
    n_try = n_fin = n_ref = 0; bad = []; kinds = set()
    stmts = []
    while len(stmts) < 90:
        rel, l, r, nvars, truth = GF.random_statement(rng)
        stmts.append((GF.NatWorld(w).statement(rel, l, r, nvars), nvars, truth, '%s %s %s' % (GF.show(l), '=' if rel == 'eq' else '<=', GF.show(r))))
    for stmt, nvars, truth, label in stmts:
        for lem in lemmas:
            for mode in ('apply', 'rw', 'rw_rev'):
                def script(p, lem=lem, mode=mode):
                    p.intro(*'abc'[:nvars])
                    if mode == 'apply': p.apply(w[lem])
                    else: p.rw(lem, rev=(mode == 'rw_rev'))
                    if not p.done:                      # the rewritten goal may close by computation
                        try: p.refl()
                        except AuthorError: pass
                n_try += 1
                got, p = attempt(stmt, script)
                if got == 'refused': n_ref += 1
                elif got == 'proved':
                    n_fin += 1; kinds.add((lem, mode))
                    if not truth: bad.append(('a proof of a statement that is false at some point', lem, mode, label))
                elif got != 'open': bad.append((got, lem, mode, label))
    expect(not bad, 'J  %d attempts to apply / rewrite %d lemmas on %d random equations (half of them false): every proof that was finished is accepted by the kernel, none is of a false statement: %s' % (n_try, len(lemmas), len(stmts), bad[:3]))
    expect(n_fin >= 40 and n_ref >= 500, 'J  the grid is not trivial (%d attempts, %d finished, %d refused, %d different (lemma, mode) pairs finished a proof)' % (n_try, n_fin, n_ref, len(kinds)))
    # --- search on random statements: nothing that is false is proved, and no proof that was found is refused by the kernel (the search commits through the kernel's check)
    eng = c['eng_noarith']; with_premises(eng, None)
    rng = random.Random(3637); counts = {'proved_true': 0, 'proved_false': 0, 'rejected': 0, 'not proved': 0}; shown = []
    for i in range(70):
        rel, l, r, nvars, truth = GF.random_statement(rng); stmt = GF.NatWorld(w).statement(rel, l, r, nvars)
        try:
            term, tree, s = S.prove_statement(eng, stmt, max_depth=5, max_probes=300, time_limit=10)
        except AuthorError as e:
            counts['rejected'] += 1; shown.append((GF.show(l), GF.show(r), str(e)[:60])); continue
        if term is None: counts['not proved'] += 1
        elif truth: counts['proved_true'] += 1
        else: counts['proved_false'] += 1; shown.append((GF.show(l), GF.show(r), 'PROVED A FALSE STATEMENT'))
    expect(counts['rejected'] == 0 and counts['proved_false'] == 0 and counts['proved_true'] >= 15, 'J  search on 70 random equations and inequalities (without ring): no proof is refused by the kernel, no false statement is proved, true ones are (%s; %s)' % (counts, shown[:2]))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ main
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ K. bug accounting and the fuzzer's health
def sec_K(c):
    import gym_fuzz as GF
    w, k = c['w'], c['k']
    st = w.global_type('add_comm')
    # --- an error that is not a refusal is a BUG of the environment, kept and (when strict) raised; a refusal is not
    for strict in (False, True):
        eng = G.Engine(w, strict=strict); tag = 'strict' if strict else 'lenient'
        g, p = first_goal(eng, st); m = eng.menu(g); a = next(a for a in m.actions if a.kind == G.Kind.APPLY and eng.probe(g, a, m) is not None)       # an action that works
        real_step = eng.step_in
        def boom(*args, **kw): raise TypeError('injected')
        eng.step_in = boom
        s0 = (tuple(k.pool_sizes()), k.num_globals(), k.get_fuel(), w.probing); f0 = eng.stats['probe_failed']; e0 = eng.errors['AuthorError']
        if strict: refuses('K  a strict engine raises a Python error from inside a probe', lambda: eng.probe(g, a, m), exc=(TypeError,))
        else: expect(eng.probe(g, a, m) is None, 'K  a lenient engine counts it as a failed action')
        expect((tuple(k.pool_sizes()), k.num_globals(), k.get_fuel(), w.probing) == s0, 'K  ... with the kernel as it was, the budget and the probing flag restored (%s)' % tag)
        expect(len(eng.bugs) == 1 and eng.bugs[0][0] == 'TypeError' and 'injected' in eng.bugs[0][1] and 'boom' in eng.bugs[0][2] and eng.stats['probe_bug'] == 1 and eng.stats['probe_failed'] == f0 + 1,
               'K  ... and it is kept as a bug, with its traceback (%s)' % tag)
        def refuse(*args, **kw): raise AuthorError('refused')
        eng.step_in = refuse
        expect(eng.probe(g, a, m) is None and len(eng.bugs) == 1 and eng.stats['probe_bug'] == 1 and eng.errors['AuthorError'] == e0 + 1 and eng.stats['probe_failed'] == f0 + 2, 'K  a refusal of the tactic layer is counted, not a bug (%s)' % tag)
        eng.step_in = real_step
        expect(eng.probe(g, a, m) is not None, 'K  the engine works again when the fault is gone (%s)' % tag)
    # --- work budgets may be given as floats (the kernel's counter is an integer)
    eng = G.Engine(w, strict=True); g, p = first_goal(eng, w.global_type('add_comm')); m = eng.menu(g)
    ring = next(a for a in m.actions if a.kind == G.Kind.RING)
    eng.probe_fuel[G.Kind.RING] = 5e6; eng.default_fuel = 5e6
    expect(eng.fuel_of(ring) == 5_000_000 and type(eng.fuel_of(ring)) is int, 'K  a budget given as a float is used as an integer')
    expect(eng.probe(g, ring, m) == [], 'K  ... and the probe with it closes the goal')
    # --- the fuzzer's canaries
    wz = c.get('wz') or int_lib.build_int()
    natW = GF.NatWorld(w); intW = GF.IntFamily(wz)
    for W, label in ((natW, 'natural'), (intW, 'integer')):
        try: GF.check_canaries(W, W.engine(False), False); ok = True; msg = ''
        except RuntimeError as e: ok = False; msg = str(e)
        expect(ok, 'K  the canaries pass in the %s world (%s)' % (label, msg))
    starved = natW.engine(False); starved.probe_fuel[G.Kind.RING] = 50
    refuses('K  a canary notices a decision procedure that runs out of budget', lambda: GF.check_canaries(natW, starved, False), exc=(RuntimeError,))
    refuses('K  ... and one that is not in the menu', lambda: GF.check_canaries(natW, G.Engine(w, kinds=NAT_NOARITH), False), exc=(RuntimeError,))
    expect(GF.check_canaries(natW, G.Engine(w, kinds=NAT_NOARITH), True) is None, 'K  ... unless the arithmetic was left out on purpose')
    # --- a short fuzz of the integers: true statements are proved (the decision procedures run), nothing wrong is found
    counts, reports = GF.run(seed=201, n=7, depth=5, probes=300, seconds=120, family=intW)
    expect(counts['REJECTED'] == 0 and counts['UNSOUND'] == 0 and counts['ERROR'] == 0, 'K  the integer fuzz finds nothing wrong (%s)' % counts)
    expect(counts['PROVED'] >= 4, 'K  ... and proves its true statements (%d of 7)' % counts['PROVED'])


SECTIONS = [('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E), ('F', sec_F), ('G', sec_G), ('H', sec_H), ('I', sec_I), ('J', sec_J), ('K', sec_K)]


def main():
    t_start = time.time()
    w = nat_lib.build_nat(); k = w.k
    c = {'w': w, 'k': k, 'fuel': k.get_fuel(), 'ths': theorem_gids(w)}
    expect(len(c['ths']) >= 90, 'the Nat world has its theorems (%d)' % len(c['ths']))
    c['eng'] = G.Engine(w); c['eng_noarith'] = G.Engine(w, kinds=NAT_NOARITH)
    only = os.environ.get('S36_ONLY')
    for name, fn in SECTIONS:
        if only and name not in only.split(','): continue
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            import traceback
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:200])); traceback.print_exc(limit=6)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
    print('S36: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:30]: print('  FAIL:', f)
    print('S36 GYM GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
