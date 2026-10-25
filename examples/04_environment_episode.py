"""04 -- one proof attempt as a game: the environment's goals, menus and atomic steps, with no text anywhere.

    python3 examples/04_environment_episode.py [problem_name]        (about 40 s; default problem: two_pow_gt_n, "n < 2^n")

The engine (src/python/environment/gym_env.py) turns proving into a sequential decision problem.  The state is a proof; the agent acts on its first open goal.  For that goal the engine offers

  * an OBSERVATION: the goal as a hash-consed DAG of arrays (node tags, scalars, child pointers, roots, heights) -- what the featurizer of the learner reads;
  * a MENU of legal actions: kinds (INTRO, CASES, INDUCT, APPLY, REWRITE, RING, LINARITH, ...), each with its argument (a hypothesis, a lemma from the library index, ...);
  * atomic STEPS: an action is probed first, so one that fails changes nothing.

Actions whose arguments are terms (a witness, a lemma's arguments, a claim) are SAID with the typed term builder instead of being listed; the dataset stores every action in a portable form
(`gym_acts.to_portable` / `from_portable`), which is what this example replays.  Whatever the agent does, the finished proof is re-checked by the kernel (`Episode.finish`).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import collections, json, os, sys
import bank_all
import gym_env as G, gym_acts as A

NAME = sys.argv[1] if len(sys.argv) > 1 else 'two_pow_gt_n'
bank = bank_all.load()
engine = bank.engine
episode = bank.episode(NAME, regime='prefix')                  # an Episode: the statement of the problem, with the premises of the prefix regime
print('problem %s:  %s' % (NAME, bank.placed[NAME].problem.text))

goal = episode.goal()
print('\nthe first goal (hypotheses above the turnstile):')
print('   ' + engine.render(goal).replace('\n', '\n   ')[:400])

names = ('tags', 'scalar', 'child_a', 'child_b', 'roots', 'heights')
obs = episode.observe()
print('\nobservation: ' + ', '.join('%s%s' % (n, tuple(a.shape)) for n, a in zip(names, obs)))
menu = episode.menu()
count = collections.Counter(G.Kind(a.kind).name for a in menu.actions)
print('menu: %d legal actions  (%s)' % (len(menu.actions), ', '.join('%s %d' % kv for kv in sorted(count.items()))))

# an action that does not apply is refused, and nothing changes
try:
    episode.step(G.Action(G.Kind.RING, 0))
    raise SystemExit('RING closed this goal?!')
except G.ActionError as e:
    print('RING on this goal: refused (%s); goals open: %d, steps taken: %d' % (e, episode.n_goals, len(episode.steps)))

# play the recorded proof, action by action, as an agent would
record = next(p for p in json.load(open(os.path.join(itp_paths.DATA, 'dataset_v2', 'bank.json')))['problems'] if p['name'] == NAME)
print('\nplaying the recorded proof (%d actions):' % len(record['demonstration']['steps']))
for step in record['demonstration']['steps']:
    menu = episode.menu()                                       # the menu of the current first goal
    action = A.from_portable(engine, menu, step)               # the portable form -> an action of this menu (or an equivalent that says its arguments)
    left = episode.step(action)
    print('  %-12s -> %d goal(s) open' % (G.Kind(action.kind).name, left))
assert episode.done
proof = episode.finish()                                        # the kernel checks the finished term against the statement
print('\nthe episode is done; the kernel accepted the proof (reward 1 in the Gymnasium wrapper, see example 06)')
print('EXAMPLE OK')
