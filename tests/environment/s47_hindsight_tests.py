"""S47 gate: HINDSIGHT -- the proof tree of an episode (gym_tree.py, `Episode(track=True)`) and the theorems it proved.

A failed attempt at a theorem is full of things that WERE proved: the branch that closed, the claim whose proof was found, the fact a lemma derived.  gym_tree.py keeps a record of the episode (one node per
goal an action was applied to) and turns the closed subtrees and the derived facts into new problems with proofs.  The hierarchical learner of docs/ARCHITECTURE_RL.md relabels its failures with them,
so what this gate protects is that the relabelling is EXACT: an example is a theorem, its statement says what its goal said, its steps are a proof of it that a fresh engine replays, the kernel and the
independent checker accept that proof -- and nothing the record gets wrong can make an example that is not.

  A. DIRECTIONS.   Every kind of action has a direction; the kinds that add facts are the FORWARD ones.
  B. THE TREE.     All the reference proofs of the bank (its proved tier; the statement-only problems have none) are replayed with `track=True`.  After every step the record agrees with the proof state (one open node per open goal); at the end there is one node per
                   action, all closed; the actions of a closed subtree are CONTIGUOUS and in the order of a depth-first walk (so they are a proof of the subtree's goal on their own); the events of a step
                   are what the step achieved; each goal is said by a closed statement (`forall hyps, target`) that a fresh episode turns into the same goal.
  C. EXAMPLES.     Pinned yields for named problems (and for the whole bank, in the tree).  Every example is verified, has a statement code that builds the statement back, is plain data (JSON), and its
                   steps replay in a fresh engine of another configuration.  Candidates that are dropped are counted by reason (too long, closed at once by the engine, context changed, ...), not lost.
  D. TAMPERING.    A record that is wrong yields NO example that is false: a corrupted action drops exactly the examples that contain it; bad data (steps dropped or added, premises too few, a code that says
                   nothing) is refused.
  E. UNFINISHED.   The examples of the first n actions of a proof are examples of the whole; an action without a portable form (INSTANTIATE) is counted and excluded, not guessed; a record that disagrees with
                   the proof state is given up (`broken`), and gives no examples; the filters (`kinds`, `max_len`, `include_root`, `dedupe`) do what they say.
  Z. SECOND OPINION.  Proofs of the examples above are accepted by the independently written checker (indep_checker.py).

Environment: S47_TRACE=1 prints timings, S47_ONLY=A,B selects sections.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, json, time, traceback, collections
sys.setrecursionlimit(200000)
import itp_env_lnl as E
import bank_all
import gym_env as G, gym_acts as A, gym_build as GB, gym_tree as T
import recheck
from gym_env import Kind, Action, ActionError
from author import AuthorError

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S47_TRACE'))
ONLY = set(os.environ['S47_ONLY'].split(',')) if os.environ.get('S47_ONLY') else None


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


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ pins
# problem -> (actions, solved examples, derived examples, and the counted reasons for the candidates that were NOT kept)
PINS = {
    'imo1959_p1':        (4, 3, 0, {}),
    'imo2019_p1':        (3, 2, 0, {}),
    'ident_diff_squares': (1, 0, 0, {}),
    'ineq_am_gm_two':    (2, 1, 1, {}),
    'two_pow_gt_n':      (7, 6, 1, {}),
    'quad_form_zero':  (13, 12, 6, {}),
    'sqrt2_irrational':  (14, 13, 2, {}),
    'frobenius_3_5':     (37, 33, 2, {'too_long': 2, 'duplicates': 1}),
    'n4_n2_1_composite': (7, 6, 0, {'context_changed': 1, 'pi_fact': 1}),
    'fe_f_x_plus_f_y':   (50, 46, 17, {'too_long': 3, 'context_changed': 1, 'settled_at_once_derived': 3}),
}
# the whole bank, by the tree alone: actions (= nodes = solved events), derived events, and the actions by direction
BANK_ACTIONS = 1816
BANK_DERIVED_EVENTS = 247
BANK_BY_DIRECTION = {'BACKWARD': 453, 'CLAIM': 177, 'CLOSE': 604, 'FORWARD': 252, 'SPLIT': 330}
SMALL = ('imo1959_p1', 'imo2019_p1', 'ineq_am_gm_two', 'two_pow_gt_n', 'quad_form_zero', 'sqrt2_irrational')      # every example of these is judged by the second checker
JUDGE = []                                  # (what, proof, statement): handed to the independent checker at the end


class Fixture:
    def __init__(self):
        t0 = time.time()
        self.bk = bank_all.load(); self.w = self.bk.w; self.k = self.w.k
        self.eng = G.Engine(self.w, kinds=set(Kind), strict=True)
        self.eng_plain = G.Engine(self.w)                                   # another configuration (the bank's own: the first language's kinds, not strict): examples are data and replay in either
        self.data = {p['name']: p for p in json.load(open(os.path.join(itp_paths.ROOT, 'data', 'dataset_v2', 'bank.json')))['problems']}
        expect(set(self.data) == set(self.bk.placed), 'fixture: the dataset has the problems of the bank')
        self.proved = [n for n, pl in self.bk.placed.items() if pl.proved]
        expect(set(self.proved) == {n for n, p in self.data.items() if 'demonstration' in p}, 'fixture: exactly the proved problems have a demonstration in the dataset (%d)' % len(self.proved))
        trace('world %.1f s (%d globals)' % (time.time() - t0, self.k.num_globals()))
        self.runs = {}

    def play(self, name, upto=None, eng=None):
        """the reference proof of `name` (its first `upto` actions) played through a tracking episode: (episode, events per step, list of what went wrong)"""
        eng = eng or self.eng
        pl = self.bk.placed[name]; steps = self.data[name]['demonstration']['steps']
        ep = G.Episode(eng, pl.stmt, name, premises=pl.premises('prefix'), track=True)
        bad = []; events = []
        for i, st in enumerate(steps[:upto]):
            menu = ep.menu()
            ep.step(A.from_portable(eng, menu if st['kind'] in A.MENU_KINDS else A.NO_MENU, st))
            events.append(list(ep.events))
            if ep.tree.broken: bad.append('the tree is broken after step %d' % i)
            if len(ep.tree.open) != ep.p.n_goals: bad.append('step %d: %d open nodes, %d open goals' % (i, len(ep.tree.open), ep.p.n_goals))
        return ep, events, bad

    def run(self, name):
        if name not in self.runs: self.runs[name] = self.play(name)
        return self.runs[name]


def descendants(n):
    """the nodes of the subtree of n in depth-first preorder, children in order"""
    out = [n]
    for c in n.children: out += descendants(c)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. directions
def sec_A(c):
    D = T.DIRECTION
    expect(set(D) == set(Kind), 'A1 every kind of action has a direction (missing %s, unknown %s)' % (sorted(set(Kind) - set(D)), sorted(set(D) - set(Kind))))
    expect(set(D.values()) <= set(T.DIRECTIONS) and set(D.values()) == set(T.DIRECTIONS), 'A2 the directions are exactly %s (%s)' % (T.DIRECTIONS, sorted(set(D.values()))))
    expect(T.DERIVING == frozenset(k for k, d in D.items() if d == 'FORWARD'), 'A3 the actions that add facts to their goal are the FORWARD ones (%s)' % sorted(G.KIND_NAMES[k] for k in T.DERIVING))
    expect(all(D[k] == 'CLOSE' for k in (Kind.RING, Kind.LINARITH, Kind.NLINARITH)) and D[Kind.CLAIM] == 'CLAIM' and D[Kind.CASES] == 'SPLIT' and D[Kind.APPLY] == 'BACKWARD', 'A4 the arithmetic closes, claim claims, cases splits, apply goes backward')
    expect(all(isinstance(v, str) for v in D.values()) and len(set(D)) == len(D), 'A5 directions are plain names')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. the tree
def sec_B(c):
    fx = c['fx']; w = fx.w
    bad = collections.defaultdict(list); tot = collections.Counter(); by_dir = collections.Counter(); t0 = time.time()
    for name in fx.proved:
        ep, events, errs = fx.run(name); tree = ep.tree; steps = fx.data[name]['demonstration']['steps']
        if errs: bad['B1 record vs proof state'].append((name, errs[:2]))
        if not (ep.done and not tree.broken and not tree.open): bad['B1 finished, record complete'].append(name)
        nodes = tree.nodes
        if not (len(nodes) == len(steps) == len(tree.by_step) == len(ep.steps) and [n.step for n in tree.by_step] == list(range(len(steps)))): bad['B2 one node per action, in order of the actions'].append(name)
        if not all(n.action is not None and n.closed and n.goal is not None for n in nodes): bad['B2 every node was acted on and is closed'].append(name)
        root = nodes[0]
        if not (tree.roots == [root] and root.parent is None and root.step == 0 and root.last == len(steps) - 1 and root.depth == 0): bad['B3 the root'].append(name)
        for n in nodes:
            if not all(ch.parent is n and ch.depth == n.depth + 1 for ch in n.children): bad['B4 parents, children, depth'].append((name, n.id))
            if n.closed != all(ch.closed for ch in n.children): bad['B4 closed iff all children are'].append((name, n.id))
            want_last = n.step if not n.children else max(ch.last for ch in n.children)
            if n.last != want_last: bad['B4 the last action of a subtree'].append((name, n.id))
            if n.portable != steps[n.step]: bad['B4 the recorded portable action is the dataset\'s'].append((name, n.id))
            sub = tree.subtree_steps(n); pre = descendants(n)
            if [m.id for m in sub] != [m.id for m in pre]: bad['B5 the actions of a subtree are contiguous and in depth-first order'].append((name, n.id))
            if (n.post is not None) != (bool(n.children) and n.action.kind in T.DERIVING): bad['B4 `post` is set for exactly the fact-adding actions that leave goals'].append((name, n.id))
            by_dir[T.DIRECTION[n.action.kind]] += 1
        # the events
        n_solved = n_derived = 0
        for i, evs in enumerate(events):
            node = tree.by_step[i]; kids = bool(node.children)
            solved = [n for kind, n in evs if kind == 'solved']; derived = [n for kind, n in evs if kind == 'derived']
            n_solved += len(solved); n_derived += len(derived)
            if kids:
                if solved: bad['B6 an action that leaves goals closes nothing'].append((name, i))
                if (len(derived) == 1 and derived[0] is node) != (node.action.kind in T.DERIVING) or len(derived) > 1: bad['B6 a fact-adding action that leaves goals derives exactly one thing'].append((name, i))
            else:
                chain = []; m = node
                while m is not None and m.closed and m.last == i: chain.append(m); m = m.parent
                if derived or solved != chain or not solved or solved[0] is not node: bad['B6 a closing action solves the node and the ancestors it completes, in that order'].append((name, i))
            if any(kind not in ('solved', 'derived') for kind, n in evs): bad['B6 event kinds'].append((name, i))
        if n_solved != len(nodes): bad['B6 every node is solved exactly once'].append((name, n_solved, len(nodes)))
        tot['actions'] += len(steps); tot['derived'] += n_derived; tot['nodes'] += len(nodes)
    trace('B: %d problems %.1f s' % (len(fx.proved), time.time() - t0))
    for what, lst in sorted(bad.items()): expect(False, '%s (%d: %s)' % (what, len(lst), lst[:3]))
    for what in ('B1 record vs proof state', 'B1 finished, record complete', 'B2 one node per action, in order of the actions', 'B2 every node was acted on and is closed', 'B3 the root', 'B4 parents, children, depth',
                 'B4 closed iff all children are', 'B4 the last action of a subtree', 'B4 the recorded portable action is the dataset\'s', 'B5 the actions of a subtree are contiguous and in depth-first order',
                 'B4 `post` is set for exactly the fact-adding actions that leave goals', 'B6 an action that leaves goals closes nothing', 'B6 a fact-adding action that leaves goals derives exactly one thing',
                 'B6 a closing action solves the node and the ancestors it completes, in that order', 'B6 every node is solved exactly once'):
        if what not in bad: expect(True, what)
    expect(tot['actions'] == tot['nodes'] == BANK_ACTIONS, 'B7 the bank\'s reference proofs: %d actions, one node each (pinned %d)' % (tot['actions'], BANK_ACTIONS))
    expect(tot['derived'] == BANK_DERIVED_EVENTS, 'B7 ... and %d facts derived by one action (pinned %d)' % (tot['derived'], BANK_DERIVED_EVENTS))
    if BANK_BY_DIRECTION: expect(dict(by_dir) == BANK_BY_DIRECTION, 'B7 ... by direction %s (pinned %s)' % (dict(by_dir), BANK_BY_DIRECTION))
    else: print('   [pin: BANK_BY_DIRECTION = %s]' % dict(sorted(by_dir.items())))
    c['by_dir'] = by_dir


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. examples
def check_example(fx, ex, tag, judge):
    """an example is a theorem with a proof: verified, a code that builds the statement, plain data, replayed by a fresh engine of another configuration"""
    w = fx.w; ok = True
    ok &= expect(ex.verified is True and ex.proof is not None, '%s: it was verified and has its proof term' % tag)
    ok &= expect(ex.kind in ('solved', 'derived') and ex.length == len(ex.steps) >= 1 and ex.premises is not None, '%s: kind, length, premises' % tag)
    ok &= expect(ex.code is not None, '%s: its statement can be said in the builder\'s tokens' % tag)
    d = ex.to_dict(); d2 = json.loads(json.dumps(d))
    ok &= expect(d2 == d, '%s: it is plain data (JSON round trip)' % tag)
    ok &= expect(d['statement_digest'] == T.digest(ex.text) and d['length'] == ex.length and d['verified'] is True, '%s: digest, length, verified in the dict' % tag)
    if ex.code is not None:
        eng2 = fx.eng_plain
        try:
            st2 = T.statement_from_code(eng2, d2['statement_code'], d2['premises'])
            ok &= expect(st2.n == ex.stmt.n, '%s: the code builds the statement back (the same kernel term)' % tag)
            pf2 = A.replay_portable(eng2, st2, d2['steps'], d2['premises'])
            ok &= True
            if judge: JUDGE.append((tag + ' (replayed in a fresh engine)', pf2, st2))
        except Exception as e:
            ok &= expect(False, '%s: the data replays in a fresh engine (%s: %s)' % (tag, type(e).__name__, str(e)[:120]))
    if judge: JUDGE.append((tag, ex.proof, ex.stmt))
    return ok


def sec_C(c):
    fx = c['fx']; t0 = time.time(); c['examples'] = {}
    for name, (n_act, n_solved, n_derived, reasons) in PINS.items():
        ep, events, errs = fx.run(name); tree = ep.tree
        exs = tree.examples(); st = dict(tree.stats); kinds = collections.Counter(e.kind for e in exs)
        c['examples'][name] = exs
        expect(len(tree.nodes) == n_act, 'C1 %s: %d actions' % (name, n_act))
        expect(kinds['solved'] == n_solved and kinds['derived'] == n_derived, 'C1 %s: %d solved + %d derived examples (got %d + %d; %s)' % (name, n_solved, n_derived, kinds['solved'], kinds['derived'], st))
        got = {kk: v for kk, v in st.items() if not kk.startswith('kept') and v and kk != 'root'}
        expect(got == reasons, 'C2 %s: the candidates that were not kept, by reason (got %s, pinned %s)' % (name, got, reasons))
        expect(st['root'] == 1 and st.get('not_replayed_solved', 0) == 0 and st.get('not_replayed_derived', 0) == 0, 'C2 %s: the root is not offered and every candidate that was not dropped for a stated reason replays' % name)
        expect(len({e.key for e in exs}) == len(exs), 'C3 %s: one example per statement (with its premises)' % name)
        every = name in SMALL
        for i, ex in enumerate(exs):
            if every or i % 6 == 0: check_example(fx, ex, 'C4 %s #%d (%s, %d actions)' % (name, i, ex.kind, ex.length), True)
            else: expect(ex.verified is True, 'C4 %s #%d verified' % (name, i))
        # the origin says where the example came from
        o_ok = all(e.origin['problem'] == name and 0 <= e.origin['step'] < n_act and e.origin['kind'] == G.KIND_NAMES[tree.by_step[e.origin['step']].action.kind] for e in exs)
        expect(o_ok, 'C5 %s: each example names its problem, the step and the kind of action it came from' % name)
        expect(all(e.origin['node'] == tree.by_step[e.origin['step']].id for e in exs), 'C5 %s: ... and the node' % name)
        trace('C %s %.1f s' % (name, time.time() - t0))
    # a solved example is the proof of a subtree: its actions are the subtree's, in order
    ep = fx.run('quad_form_zero')[0]; tree = ep.tree
    sol = [e for e in c['examples']['quad_form_zero'] if e.kind == 'solved']
    expect(all(e.steps == [m.portable for m in tree.subtree_steps(tree.by_step[e.origin['step']])] for e in sol), 'C6 a solved example\'s steps are the actions of its subtree')
    der = [e for e in c['examples']['quad_form_zero'] if e.kind == 'derived']
    expect(all(e.length == 1 and e.steps == [tree.by_step[e.origin['step']].portable] and tree.by_step[e.origin['step']].action.kind in T.DERIVING for e in der), 'C6 a derived example is ONE fact-adding action')
    # the statement of a derived example is the fact, in the context the action was applied in
    ok = True
    for e in der:
        n = tree.by_step[e.origin['step']]; pre, post = n.goal, n.post
        want = T.closed_statement(fx.w, pre.hyp_types, post.hyp_types[pre.n_hyps + e.origin['fact']])
        ok &= want.n == e.stmt.n
    expect(ok, 'C7 the statement of a derived example is forall (the context of the action), (the new fact)')
    # the example dict is the data BankEnv.reset(options={'goal': ...}) takes
    e0 = c['examples']['quad_form_zero'][0].to_dict()
    expect(set(e0) == {'kind', 'statement', 'statement_digest', 'statement_code', 'premises', 'steps', 'length', 'origin', 'verified'}, 'C8 the keys of an example\'s dict')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. goals
def sec_F(c):
    """a goal is said by `forall hyps, target`, and a fresh episode on that statement is on the same goal: the identification the whole relabelling rests on"""
    fx = c['fx']; w = fx.w; eng = fx.eng; n_nodes = 0; bad = []
    for name in ('quad_form_zero', 'sqrt2_irrational', 'imo1959_p1', 'two_pow_gt_n', 'ineq_am_gm_two', 'imo2019_p1', 'n4_n2_1_composite', 'ident_sophie_germain'):
        ep = fx.run(name)[0]
        for n in ep.tree.nodes:
            stmt = T.closed_statement(w, n.goal.hyp_types, n.goal.target)
            eng.premises = ep.premises; p = eng.start(stmt)
            n_nodes += 1
            if p.n_goals != 1: bad.append((name, n.id, 'goals %d' % p.n_goals)); continue
            g2 = eng.goal_of(p, 0)
            if list(g2.hyp_types) != list(n.goal.hyp_types) or g2.target != n.goal.target: bad.append((name, n.id, 'another goal'))
    expect(not bad, 'F1 %d goals: a fresh episode on the closed statement of a goal is on that goal (%s)' % (n_nodes, bad[:3]))
    # the settled form is not always the statement as it was written (`Not P` is `P -> False` once the engine has introduced the hypothesis): it is what the agent sees, and convertible to the problem
    pl = fx.bk.placed['sqrt2_irrational']; ep = fx.run('sqrt2_irrational')[0]; root = ep.tree.nodes[0]
    cs = T.closed_statement(w, root.goal.hyp_types, root.goal.target)
    expect(cs.n != pl.stmt.n, 'F2 the settled form of sqrt2_irrational is not the statement as written')
    ex = ep.tree.examples(include_root=True, kinds=('solved',), max_len=100)
    r = [e for e in ex if e.origin['node'] == root.id]
    expect(len(r) == 1 and r[0].steps == fx.data['sqrt2_irrational']['demonstration']['steps'] and r[0].length == 14, 'F3 include_root: the root\'s example is the whole reference proof')
    # (the proof of the settled form is checked against the settled form: a different statement from the problem, the same proof)
    expect(r and r[0].verified and r[0].stmt.n == cs.n, 'F3 ... of the settled statement, verified')
    pl_ = fx.bk.placed['imo1959_p1']; ep_ = fx.run('imo1959_p1')[0]
    root_ = ep_.tree.nodes[0]
    expect(T.closed_statement(w, root_.goal.hyp_types, root_.goal.target).n == pl_.stmt.n, 'F4 when settling changes nothing, the closed statement of the root IS the problem\'s statement')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. tampering
def sec_D(c):
    fx = c['fx']; eng = fx.eng_plain; w = fx.w
    exs = c['examples']['quad_form_zero']
    ex = max((e for e in exs if e.kind == 'solved'), key=lambda e: e.length)
    expect(ex.length >= 3, 'D0 (the longest solved example of quad_form_zero has %d actions)' % ex.length)
    d = json.loads(json.dumps(ex.to_dict())); pr = d['premises']
    stmt = T.statement_from_code(eng, d['statement_code'], pr)
    A.replay_portable(eng, stmt, d['steps'], pr)
    raises('D1 an action fewer: the proof is not finished', lambda: A.replay_portable(eng, stmt, d['steps'][:-1], pr), ActionError)
    raises('D1 an action too many: the proof is already finished', lambda: A.replay_portable(eng, stmt, d['steps'] + [d['steps'][-1]], pr), ActionError)
    raises('D1 no actions at all', lambda: A.replay_portable(eng, stmt, [], pr), ActionError)
    # the premises are part of the example: the lemmas an example uses must be visible
    def used(steps):
        out = set()
        for s in steps:
            if isinstance(s.get('source'), list) and s['source'][:1] == ['c']: out.add(s['source'][1])
            for code in s.get('codes', ()):
                out |= {t[1] for t in code if t[0] == 'c'}
        return out
    gids = sorted(set().union(*[used(e.steps) for e in exs]))
    user = next((e for e in exs if gids and gids[0] in used(e.steps)), None)
    if expect(user is not None, 'D2 (an example that uses a library lemma: %s)' % (gids[:3],)):
        du = json.loads(json.dumps(user.to_dict()))
        stmt_u = T.statement_from_code(eng, du['statement_code'], du['premises'])
        A.replay_portable(eng, stmt_u, du['steps'], du['premises'])
        raises('D2 the same data with the premises cut below the lemma it uses is refused', lambda: A.replay_portable(eng, stmt_u, du['steps'], gids[0]), (ActionError, AuthorError, E.ITPError))
    # a code that says nothing
    for what, code in (('an empty code', []), ('an unknown token', [['zzz']]), ('a token with the wrong shape', [['c']]), ('a constant that does not exist', [['c', 10 ** 6, 0]]), ('not a list of tokens', 5)):
        raises('D3 %s builds no statement' % what, lambda code=code: T.statement_from_code(eng, code, pr), (GB.BuildError, TypeError, ValueError))
    # a false statement is not made true by a record: the kernel is the judge at the end
    with w.nota.using(w['Z']):
        lit = w.nota.lit; false = w.Eq(lit(1), lit(0))
    expect(T.replays(eng, false, [{'kind': 'RING'}], None) is None and T.replays(eng, false, [{'kind': 'NLINARITH'}], None) is None, 'D4 1 = 0 is not proved by ring or nlinarith')
    # a wrong RECORD: one leaf's action is replaced by `ring` (which proves no inequality): the examples that contain that action are dropped, counted, and no other
    ep, events, errs = fx.play('quad_form_zero'); tree = ep.tree
    leaf = next(n for n in tree.nodes if not n.children and n.action.kind == Kind.NLINARITH)
    inside = []; m = leaf
    while m is not None: inside.append(m); m = m.parent
    base = {e.key: e for e in c['examples']['quad_form_zero']}
    leaf.portable = {'kind': 'RING'}
    got = tree.examples(); st = dict(tree.stats)
    want_dropped = len(inside) - 1                       # (the root is not offered)
    expect(st.get('not_replayed_solved', 0) == want_dropped and st.get('kept_solved', 0) == 12 - want_dropped, 'D5 a wrong action in the record drops the %d solved examples that contain it, and no other (%s)' % (want_dropped, st))
    expect(st.get('kept_derived', 0) == 6 and all(e.verified for e in got), 'D5 ... the derived ones are untouched, and all that is returned is verified')
    expect({e.key for e in got} <= set(base) and not any(e.origin['node'] in {x.id for x in inside} and e.kind == 'solved' for e in got), 'D5 ... and nothing new appears')
    # the record of an action that cannot be written down is absent, not invented
    leaf.portable = None
    got = tree.examples(); st = dict(tree.stats)
    expect(st.get('not_portable_solved', 0) == want_dropped and st.get('kept_solved', 0) == 12 - want_dropped, 'D6 an action without a portable form excludes the %d examples that contain it (%s)' % (want_dropped, st))
    leaf.portable = fx.data['quad_form_zero']['demonstration']['steps'][leaf.step]
    expect(len(tree.examples()) == len(base), 'D7 (the record put right gives the examples again)')
    # the checks that guard what a derived fact is a fact ABOUT (no action of today's language can trip them: the record is made to)
    n = next(x for x in tree.nodes if x.post is not None); post = n.post; n_pre = n.goal.n_hyps; k = fx.k
    good = tree.examples(kinds=('derived',)); n_good = len(good)
    n.post = G.GoalData(post.target, [post.hyp_types[-1]] + list(post.hyp_types[1:]), post.hyp_names)               # the context the action found is not the one the record says
    got = tree.examples(kinds=('derived',)); st = dict(tree.stats)
    expect(st.get('context_changed') == 1 and len(got) == n_good - 1 and not st.get('not_replayed_derived'), 'D8 a derived fact whose context is not the context of the action is counted, not guessed (%s)' % (st,))
    n.post = G.GoalData(post.target, list(post.hyp_types[:-1]) + [k.mk_fvar(n_pre)], post.hyp_names)               # a fact that mentions a variable the action brought
    got = tree.examples(kinds=('derived',)); st = dict(tree.stats)
    expect(st.get('depends_on_new') == 1 and len(got) == n_good - 1 and not st.get('not_replayed_derived'), 'D9 a derived fact about a hypothesis that is new in the context is counted, not guessed (%s)' % (st,))
    n.post = post
    expect(len(tree.examples(kinds=('derived',))) == n_good, 'D9 (the record put right gives the examples again)')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. unfinished, odd
def lin_statement(w):
    Z = w['Z']; F = w.forall_; Eq = w.Eq; ZZ = w.arrow(Z, Z)
    with w.nota.using(Z):
        lit = w.nota.lit
        return F([ZZ, lambda f: F([Z], lambda n: Eq(f(n), 2 * n + 1))], lambda f, h: Eq(f(lit(3)) + f(lit(3)), lit(14)))          # (forall n, f n = 2 n + 1) -> f 3 + f 3 = 14


def sec_E(c):
    fx = c['fx']; w = fx.w; k = fx.k; eng = fx.eng
    full = {e.key for e in c['examples']['quad_form_zero']}
    # E1. the first n actions of a proof: their examples are examples of the whole, and the closed subtrees only grow
    prev_closed = -1; n_by_len = []
    for L in range(0, 14):
        ep, events, errs = fx.play('quad_form_zero', upto=L); tree = ep.tree
        expect(not errs and not tree.broken, 'E1 n=%d: the record agrees with the proof state' % L)
        exs = tree.examples(); keys = {e.key for e in exs}
        expect(keys <= full, 'E1 n=%d: %d examples, all of them examples of the whole proof' % (L, len(exs)))
        expect(all(e.verified for e in exs), 'E1 n=%d: ... all verified' % L)
        closed = len(tree.solved_nodes())
        expect(closed >= prev_closed, 'E1 n=%d: closed subtrees only grow (%d)' % (L, closed)); prev_closed = closed
        expect(len(tree.open) == ep.p.n_goals and (L == 13) == ep.done, 'E1 n=%d: open nodes are open goals; finished only at the end' % L)
        n_by_len.append(len(exs))
    expect(keys == full, 'E1 the whole proof gives all the examples')
    expect(n_by_len[0] == 0 and n_by_len[1] == 0 and n_by_len[-1] == 18 and n_by_len[-1] == max(n_by_len), 'E1 the yield by prefix length %s' % n_by_len)
    expect(any(0 < x < 18 for x in n_by_len), 'E1 an unfinished attempt already yields examples (%s)' % n_by_len)
    # E2. an action with no portable form (INSTANTIATE: relative to the menu's pool of terms) is counted, never guessed
    lin = lin_statement(w)
    ep = G.Episode(eng, lin, 'lin', premises=None, track=True); m = ep.menu()
    src1 = k.mk_fvar(1)
    inst = next((a for a in m.actions if a.kind == Kind.INSTANTIATE and m.cands[a.arg][0] == src1 and tuple(m.pool.labels[i] for i in a.args) == ('3',)), None)
    if expect(inst is not None, 'E2 (the menu offers the instance of the hypothesis at 3)'):
        ep.step(inst)
        expect(ep.events and ep.events[0][0] == 'derived' and ep.events[0][1].portable is None, 'E2 the step derived a fact; the action has no portable form')
        lins = [a for a in ep.menu().actions if a.kind == Kind.LINARITH]
        ep.step(lins[0]); expect(ep.done, 'E2 linarith closes the goal with the new fact')
        tree = ep.tree; exs = tree.examples(); st = dict(tree.stats)
        expect(st.get('not_portable_derived') == 1 and st.get('root') == 1 and len(exs) == 1 and exs[0].kind == 'solved' and exs[0].length == 1, 'E2 only the linarith step is an example; the instance is counted, not guessed (%s)' % st)
        exs_r = tree.examples(include_root=True); st = dict(tree.stats)
        expect(st.get('not_portable_solved') == 1 and len(exs_r) == 1, 'E2 include_root: the whole proof has an action without a portable form: not an example (%s)' % st)
        judge_ex = exs[0]; JUDGE.append(('E2 the step after an INSTANTIATE', judge_ex.proof, judge_ex.stmt))
        try: json.dumps(ep.tree.by_step[0].portable)
        except Exception as e: expect(False, 'E2 (json) %s' % e)
        expect(exs[0].to_dict()['origin']['kind'] == 'LINARITH', 'E2 the example came from the LINARITH step')
    # E3. a record that disagrees with the proof state is given up
    for what, tamper in (('an open node is missing', lambda t: t.open.clear()), ('an open node too many', lambda t: t.open.append(t._new(None))),):
        ep = G.Episode(eng, fx.bk.placed['imo1959_p1'].stmt, 'imo1959_p1', premises=fx.bk.placed['imo1959_p1'].premises('prefix'), track=True)
        steps = fx.data['imo1959_p1']['demonstration']['steps']
        ep.step(A.from_portable(eng, ep.menu() if steps[0]['kind'] in A.MENU_KINDS else A.NO_MENU, steps[0]))
        tamper(ep.tree)
        ok = True
        for st_ in steps[1:]:
            try: ep.step(A.from_portable(eng, ep.menu() if st_['kind'] in A.MENU_KINDS else A.NO_MENU, st_))
            except Exception as e: ok = False; expect(False, 'E3 %s: the episode itself goes on (%s)' % (what, e)); break
        expect(ok and ep.done and ep.tree.broken and ep.events == [], 'E3 %s: the episode finishes, the record is given up and reports no events' % what)
        out = ep.tree.examples()
        expect(out == [] and ep.tree.stats.get('tree_broken') == 1, 'E3 %s: a broken record gives no examples' % what)
    # E4. the filters
    ep = fx.run('frobenius_3_5')[0]; tree = ep.tree
    a = tree.examples(); b = tree.examples(dedupe=False)
    expect(len(b) == len(a) + 1, 'E4 dedupe=False keeps the one duplicate frobenius_3_5 has (%d vs %d)' % (len(b), len(a)))
    groups = collections.defaultdict(list)
    for e in b: groups[e.key].append(e)
    dup = [g for g in groups.values() if len(g) > 1]
    if expect(len(dup) == 1 and len(dup[0]) == 2, 'E4 ... two proofs of one statement'):
        kept = next(e for e in a if e.key == dup[0][0].key)
        expect(kept.length == min(e.length for e in dup[0]), 'E4 ... dedupe keeps the shorter (%d of %s)' % (kept.length, [e.length for e in dup[0]]))
    s_only = tree.examples(kinds=('solved',)); d_only = tree.examples(kinds=('derived',))
    expect(all(e.kind == 'solved' for e in s_only) and all(e.kind == 'derived' for e in d_only) and len(s_only) == 33 and len(d_only) == 2, 'E4 kinds selects solved or derived (%d, %d)' % (len(s_only), len(d_only)))
    short = tree.examples(max_len=3); st = dict(tree.stats)
    expect(short and all(e.length <= 3 for e in short) and st.get('too_long', 0) > 2, 'E4 max_len=3 keeps proofs of at most 3 actions (%d examples, %d too long)' % (len(short), st.get('too_long', 0)))
    # ... exactly those: the limit itself is kept, one more is not
    every = tree.examples(verify=False, max_len=100); lengths = sorted({e.length for e in every})
    limits = sorted({lengths[0], lengths[1], lengths[len(lengths) // 2], lengths[-2], lengths[-1]})
    expect(len(limits) >= 4, 'E4 (frobenius_3_5 has examples of many lengths: %s)' % lengths)
    for L in limits:                                                                  # (each is the length of some example, so that the limit itself is in play)
        got = {e.key for e in tree.examples(verify=False, max_len=L)}; want = {e.key for e in every if e.length <= L}
        expect(got == want and any(e.length == L for e in every), 'E4 max_len=%d keeps exactly the %d examples of at most %d actions (got %d)' % (L, len(want), L, len(got)))
    # two goals with one statement and proofs of different lengths: dedupe keeps the shorter, whichever comes first
    Z = w['Z']; F_ = w.forall_; zle = w['Z.le']; lemma = {'kind': 'APPLY', 'source': ['c', w.gid['Z.le_of_sub_nonneg'], 0]}
    with w.nota.using(Z):
        lit = w.nota.lit
        dup = F_([Z, Z, lambda a, b: w['Or'](zle(a, b), zle(a, b))], lambda a, b, h: zle(a, b + lit(1)))              # a <= b \/ a <= b -> a <= b + 1: cases leaves the same goal twice
    for first_short in (True, False):
        ep2 = G.Episode(eng, dup, 'dup', premises=None, track=True)
        ep2.step(next(a for a in ep2.menu().actions if a.kind == Kind.CASES))
        def one(): ep2.step(next(a for a in ep2.menu().actions if a.kind == Kind.LINARITH))
        def two(): ep2.step(A.from_portable(eng, ep2.menu(), lemma)); one()
        (one if first_short else two)(); (two if first_short else one)()
        if not expect(ep2.done, 'E4 (the two branches of a cases on `P or P` are proved by one action and by two; short first: %s)' % first_short): continue
        t2 = ep2.tree; both = t2.examples(dedupe=False, kinds=('solved',)); one_each = t2.examples(kinds=('solved',))
        same = [e for e in both if e.length in (1, 2) and e.stmt.n == both[0].stmt.n] if both else []
        groups = collections.defaultdict(list)
        for e in both: groups[e.key].append(e)
        dups = [g_ for g_ in groups.values() if len(g_) == 2]
        if expect(len(both) == 3 and len(dups) == 1 and sorted(e.length for e in dups[0]) == [1, 2], 'E4 the branches give two examples of one statement, 1 and 2 actions long (short first: %s; %s)' % (first_short, [(e.length, e.origin['node']) for e in both])):
            kept = next(e for e in one_each if e.key == dups[0][0].key)
            expect(len(one_each) == 2 and kept.length == 1 and t2.stats['duplicates'] == 1, 'E4 dedupe keeps the one-action proof (short first: %s; kept %d actions, %s)' % (first_short, kept.length, dict(t2.stats)))
    raw = tree.examples(verify=False, kinds=('solved',))
    expect(all(e.verified is None and e.proof is None for e in raw) and len(raw) >= len(s_only), 'E4 verify=False returns candidates that say they are not verified (%d)' % len(raw))
    # E5. the engine closes some facts at once: no action to learn from
    ep = fx.run('fe_f_x_plus_f_y')[0]
    ex = ep.tree.examples(kinds=('derived',)); st = dict(ep.tree.stats)
    expect(st.get('settled_at_once_derived') == 3 and len(ex) == 17 and st.get('not_replayed_derived', 0) == 0, 'E5 a fact the engine\'s settling closes (a hypothesis proves it) is counted as such, not as a failure (%s)' % st)
    # E6. the record of a problem that is NOT tracked
    ep = G.Episode(eng, fx.bk.placed['imo1959_p1'].stmt, 'imo1959_p1', premises=fx.bk.placed['imo1959_p1'].premises('prefix'))
    expect(ep.tree is None and ep.events == [] and ep.name == 'imo1959_p1', 'E6 an episode made without track keeps no tree')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ Z. second opinion
def small_world_equivalence():
    """`SecondOpinion(k, needed_by=...)` gives the verdict of the full environment, on a small world where the full one is cheap: the same proofs accepted, the same wrong ones refused; it holds fewer globals; and a
    term it was not made for is refused with a reason (never a crash)"""
    import int_lib
    w = int_lib.build_int(); k = w.k; Z = w['Z']; F = w.forall_; Eq = w.Eq; zle = w['Z.le']
    eng = G.Engine(w, kinds=set(Kind), strict=True)
    comm = F([Z, Z], lambda a, b: Eq(a + b, b + a)); wrong = F([Z, Z], lambda a, b: Eq(a + b, b + b))
    with w.nota.using(Z):
        lit = w.nota.lit
        succ = F([Z, Z, lambda a, b: zle(a, b)], lambda a, b, h: zle(a, b + lit(1)))
    p1 = A.replay_portable(eng, comm, [{'kind': 'RING'}], None)
    p2 = A.replay_portable(eng, succ, [{'kind': 'LINARITH'}], None)
    full = recheck.SecondOpinion(k)
    part = recheck.SecondOpinion(k, needed_by=[p1, comm])
    both = recheck.SecondOpinion(k, needed_by=[p1, comm, p2, succ])
    ok = lambda r: bool(r[0]) if isinstance(r, tuple) else bool(r)
    expect(ok(full.accepts(p1, comm)) and ok(part.accepts(p1, comm)) and ok(both.accepts(p1, comm)), 'Z4 a proof of commutativity is accepted by the full environment and by the ones made for it')
    expect(ok(full.accepts(p2, succ)) and ok(both.accepts(p2, succ)), 'Z4 so is a proof by linarith')
    expect(not ok(full.accepts(p1, wrong)) and not ok(part.accepts(p1, wrong)), 'Z4 a proof of another statement made of the same globals is refused by both')
    expect(not ok(full.accepts(p1, succ)) and not ok(both.accepts(p1, succ)), 'Z4 ... also a proof of an inequality offered for an equation')
    needed = recheck.needed_globals(k, [p1, comm])
    expect(0 < len(part.gids) < len(both.gids) < k.num_globals() and part.gids == needed and len(full.checker.g) == k.num_globals(), 'Z4 the environment made for one proof is smaller than the one for two, and than the world (%d, %d of %d)' % (len(part.gids), len(both.gids), k.num_globals()))
    expect(set(part.checker.g) == part.gids and all(part.checker.g[g].name == full.checker.g[g].name for g in part.gids), 'Z4 it holds exactly those globals, as the full one has them')
    closed = True; n_ind = n_ctor = n_rec = 0
    for gid in part.gids:
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid); ri = k.recursive_info(gid)
        closed &= set(ctors) <= part.gids and (cf is None or cf < 0 or cf in part.gids) and (ri is None or ri[0] in part.gids)
        n_ind += bool(ctors); n_ctor += (cf is not None and cf >= 0); n_rec += ri is not None
    expect(closed and n_ind and n_ctor and n_rec, 'Z4 the globals it holds are closed under constructors, families and the family a recursive function recurses on (%d inductive types, %d constructors, %d recursive functions in it)' % (n_ind, n_ctor, n_rec))
    r = part.accepts(p2, succ)
    expect(isinstance(r, tuple) and r[0] is False and 'not made for' in r[1], 'Z4 a term that mentions globals the environment was not made for is refused, with the reason (%s)' % (r,))
    expect(part.problems() == [] and both.problems() == [], 'Z4 the globals the environments hold pass the independent check of their definitions')


def constructors_by_type():
    """the one link that a scan of the terms cannot find: a proof that matches on a type and names none of its constructors.  The environment made for it must hold the constructors all the same, or the
    independent checker cannot read the `match`"""
    import author
    w = author.World(); k = w.k
    w.inductive('True', w.PROP, [('tt', [])])
    w.inductive('Two', w.TYPE, [('first', []), ('second', [])])
    Two, tt, truth = w['Two'], w['tt'], w['True']
    stmt = w.forall_([Two], lambda x: truth)
    proof = w.fun([Two], lambda x: w.match(x, lambda _: truth, [tt, tt]))
    w.define('all_two', stmt, proof)                                              # the kernel's own verdict comes first
    named = recheck.consts_of(k.dump_raw(proof.n)) | recheck.consts_of(k.dump_raw(stmt.n))
    ctors = set(k.global_info(w.gid['Two'])[5])
    expect(len(ctors) == 2 and not (named & ctors) and w.gid['Two'] in named, 'Z5 the proof names the type and neither of its two constructors')
    needed = recheck.needed_globals(k, [proof, stmt])
    expect(ctors <= needed, 'Z5 yet the environment made for it holds both constructors')
    try: r = recheck.SecondOpinion(k, needed_by=[proof, stmt]).accepts(proof, stmt)
    except Exception as e: r = (False, '%s: %s' % (type(e).__name__, str(e)[:80]))
    expect(isinstance(r, tuple) and bool(r[0]), 'Z5 and the independent checker accepts the proof there (%s)' % (r,))


def sec_Z(c):
    fx = c['fx']; t0 = time.time()
    small_world_equivalence()
    trace('Z4: %.1f s' % (time.time() - t0))
    constructors_by_type()
    trace('Z5: %.1f s' % (time.time() - t0))
    roots = [x for _, p, s in JUDGE for x in (p, s)]
    t0 = time.time(); second = recheck.SecondOpinion(fx.k, needed_by=roots)
    trace('Z: the independent checker\'s environment %.1f s (%d of %d globals), %d proofs to judge' % (time.time() - t0, len(second.gids), fx.k.num_globals(), len(JUDGE)))
    bad = []
    for what, proof, stmt in JUDGE:
        try: r = second.accepts(proof, stmt)
        except Exception as e: r = (False, '%s: %s' % (type(e).__name__, str(e)[:80]))
        ok = r if isinstance(r, bool) else bool(r[0]) if isinstance(r, tuple) else bool(r)
        if not ok: bad.append((what, r))
    expect(len(JUDGE) >= 60, 'Z1 (%d proofs judged)' % len(JUDGE))
    expect(not bad, 'Z2 the independent checker accepts every proof (%d judged; refused: %s)' % (len(JUDGE), bad[:3]))
    # and it is not a rubber stamp: the same proof against another statement
    what, proof, stmt = JUDGE[0]
    other = next((s for w_, p_, s in JUDGE if s.n != stmt.n), None)
    if expect(other is not None, 'Z3 (another statement)'):
        r = second.accepts(proof, other); ok = r if isinstance(r, bool) else bool(r[0]) if isinstance(r, tuple) else bool(r)
        expect(not ok, 'Z3 ... and it refuses a proof of another statement')
    trace('Z: %.1f s' % (time.time() - t0))


SECTIONS = (('A', sec_A), ('B', sec_B), ('C', sec_C), ('F', sec_F), ('D', sec_D), ('E', sec_E), ('Z', sec_Z))


def main():
    t_start = time.time()
    c = {'fx': Fixture()}
    for name, fn in SECTIONS:
        if ONLY is not None and name not in ONLY: continue
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:300])); traceback.print_exc(limit=8)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
    print('S47: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:60]: print('  FAIL:', f)
    print('S47 HINDSIGHT GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
