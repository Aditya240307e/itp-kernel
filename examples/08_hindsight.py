"""08 -- hindsight: what an attempt proved, even when it did not finish.

    python3 examples/08_hindsight.py [problem_name [n_actions]]       (about 40 s; default: quad_form_zero, stopped after 7 of its 13 actions)

A proof attempt that fails is full of things that WERE proved: a case that closed, a fact that a lemma derived.  With `Episode(..., track=True)` the engine keeps a record of the attempt (src/python/environment/
gym_tree.py): one node per goal an action was applied to.  From it come

  * what each action ACHIEVED (a subtree closed -- 'solved' -- or a fact added to the goal -- 'derived');
  * new PROBLEMS with proofs: the closed subtrees and the derived facts, each as a closed statement `forall hypotheses, target` and the actions that proved it.

Nothing is guessed: before an example is returned it is replayed in a fresh proof state and the kernel checks the term.  An example is plain data (statement code, premises, portable actions): the Gymnasium
environment starts an episode on it (`BankEnv.reset(options={'goal': example.to_dict()})`, example 06 shows the environment), and a hierarchical agent can use it as a goal for a worker.
Here: the attempt, what it achieved, its examples, and one example proved again from nothing but its data, then judged by the independent checker.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import json, os, sys
import bank_all
import gym_env as G, gym_acts as A, gym_tree as T
import recheck

NAME = sys.argv[1] if len(sys.argv) > 1 else 'quad_form_zero'
bank = bank_all.load()
w = bank.w
if NAME not in bank.placed: raise SystemExit('no such problem; the bank has %s ...' % ', '.join(list(bank.placed)[:6]))
record = next(p for p in json.load(open(os.path.join(itp_paths.DATA, 'dataset_v2', 'bank.json')))['problems'] if p['name'] == NAME)
steps = record['demonstration']['steps']
N = int(sys.argv[2]) if len(sys.argv) > 2 else (7 if NAME == 'quad_form_zero' else max(1, len(steps) // 2))
N = min(N, len(steps))
engine = G.Engine(w, kinds=set(G.Kind), strict=True)
placed = bank.placed[NAME]
print('problem %s: %s' % (NAME, placed.problem.text))


def attempt(n):
    """the first n actions of the recorded proof, played on a tracking episode"""
    ep = G.Episode(engine, placed.stmt, NAME, premises=placed.premises('prefix'), track=True)
    log = []
    for i, st in enumerate(steps[:n]):
        menu = ep.menu()
        action = A.from_portable(engine, menu if st['kind'] in A.MENU_KINDS else A.NO_MENU, st)
        ep.step(action)
        log.append((G.KIND_NAMES[action.kind], T.DIRECTION[action.kind], [(kind, node.step, node.last) for kind, node in ep.events], ep.n_goals))
    return ep, log


# ---------------------------------------------------------------------------------------------------------------------------------------- an attempt that stops
ep, log = attempt(N)
print('\nan attempt that stops after %d of the %d recorded actions (%d goals still open):' % (N, len(steps), ep.n_goals))
for i, (kind, direction, events, left) in enumerate(log):
    achieved = ', '.join('%s (%s)' % (k, ('actions %d..%d' % (a, b)) if k == 'solved' else 'a fact') for k, a, b in events)
    print('  %2d. %-9s %-8s %2d open   %s' % (i + 1, kind, direction, left, achieved))

# ---------------------------------------------------------------------------------------------------------------------------------------- what it proved
tree = ep.tree
exs = tree.examples()
print('\n%d theorems proved by the attempt (%d closed subtrees, %d derived facts); candidates not kept: %s'
      % (len(exs), sum(e.kind == 'solved' for e in exs), sum(e.kind == 'derived' for e in exs), {k: v for k, v in tree.stats.items() if not k.startswith('kept')}))
for e in exs[:6]:
    print('  %-8s %2d action(s)   %s' % (e.kind, e.length, e.text[:100] + (' ...' if len(e.text) > 100 else '')))
if len(exs) > 6: print('  ... and %d more' % (len(exs) - 6))
assert exs and all(e.verified for e in exs)
whole = ep if N == len(steps) else attempt(len(steps))[0]
full = whole.tree.examples()
print('the same proof finished gives %d theorems; every one of the %d above is among them: %s' % (len(full), len(exs), {e.key for e in exs} <= {e.key for e in full}))
assert {e.key for e in exs} <= {e.key for e in full}

# ---------------------------------------------------------------------------------------------------------------------------------------- an example is data
ex = max((e for e in exs if e.kind == 'solved'), key=lambda e: e.length, default=exs[0])
d = json.loads(json.dumps(ex.to_dict()))                           # plain data: this is what a dataset file or a replay buffer holds
print('\nthe longest solved example, as data:')
print('  statement  ', d['statement'][:120] + (' ...' if len(d['statement']) > 120 else ''))
print('  code       ', '%d tokens of the typed term builder, %d premises' % (len(d['statement_code']), d['premises']))
print('  proof      ', ', '.join(s['kind'] for s in d['steps']))
fresh = G.Engine(w)                                                # another engine, configured as the bank's own
statement = T.statement_from_code(fresh, d['statement_code'], d['premises'])
proof = A.replay_portable(fresh, statement, d['steps'], d['premises'])
print('proved again from the data alone in a fresh engine; the kernel accepted the term')
ok, message = recheck.SecondOpinion(bank.k, needed_by=[proof, statement]).accepts(proof, statement)
print('independent checker:', 'accepts' if ok else 'REJECTS (%s)' % message)
assert ok

# ---------------------------------------------------------------------------------------------------------------------------------------- the bank, in numbers
print('\nthe directions of the actions (what an action does to its goal): %s' % ', '.join('%s %d' % (k, sum(1 for e in log if e[1] == k)) for k in T.DIRECTIONS))
print('EXAMPLE OK')
