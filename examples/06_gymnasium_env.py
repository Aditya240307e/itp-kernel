"""06 -- the problem bank as a Gymnasium environment, at the level of the policy's DECISIONS.

    python3 examples/06_gymnasium_env.py [problem_name]       (about a minute; needs gymnasium, numpy; default problem: imo1959_p1)

`BankEnv` (src/python/learner/bank_env.py) wraps the engine of example 04 in the Gymnasium interface.  There is no text: the observation is the goal as arrays of token ids plus integer rows that
describe the OPTIONS of the question currently asked, and the action is the index of one option.  An agent does not pick "an action" at once; it answers a short sequence of questions that assemble
one -- which KIND of action, then which lemma / hypothesis, then (for an action that carries a term) the next token of the term it says, each among the options the engine offers at that point
(`gym_policy.Assembler`).  The episode ends when the proof is complete and checked by the kernel (reward `solve_reward`), or on truncation.

Below, the recorded proof of a problem is played through the environment decision by decision: the "agent" answers each question with the option that the recorded action implies.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import json, os, sys, warnings
import numpy as np
import gymnasium
warnings.filterwarnings('ignore', message='.*Casting input x to numpy array.*')          # (Space.contains on an observation of mixed dtypes)
import bank_all, bank_env as BE
import gym_acts as ACTS, gym_policy as P

NAME = sys.argv[1] if len(sys.argv) > 1 else 'imo1959_p1'
bank = bank_all.load()
env = BE.BankEnv(bank, names=[NAME], max_actions=40, max_decisions=400, solve_reward=1.0)
print('registered with Gymnasium as ITPBank-v0:', 'ITPBank-v0' in gymnasium.registry)

obs, info = env.reset(options={'problem': NAME})
print('\nobservation space:  ' + ', '.join('%s%s' % (k, tuple(v.shape)) for k, v in env.observation_space.spaces.items()))
print('action space:       Discrete(%d)  (the index of an option of the current question)' % env.action_space.n)
print('first question:     "%s", %d options, %d of them legal (mask)' % (info['choice'].name, len(info['choice'].options), int(info['action_mask'].sum())))
assert env.observation_space.contains(obs)

# ---------------------------------------------------------------------------------------------------------------------------------------- play the recorded proof, decision by decision
record = next(p for p in json.load(open(os.path.join(itp_paths.DATA, 'dataset_v1', 'bank.json')))['problems'] if p['name'] == NAME)
engine, placed = env.eng, bank.placed[NAME]
total, trace, n_decisions = 0.0, [], 0
for step in record['demonstration']['steps']:
    engine.premises = placed.premises('prefix')
    goal, menu = env.ep.goal(), env.ep.menu()
    action = ACTS.from_portable(engine, menu if step['kind'] in ACTS.MENU_KINDS else ACTS.NO_MENU, step)
    questions = []
    for choice, wanted in P.decompose(engine, goal, menu, action):              # the questions that assemble this action, and the option that answers each
        assert info['choice'].name == choice.name and len(info['choice'].options) == len(choice.options)
        questions.append('%s[%d]=%d' % (choice.name, len(choice.options), wanted))
        obs, reward, terminated, truncated, info = env.step(wanted)
        total += reward; n_decisions += 1
    trace.append('%-9s %s' % (step['kind'], ' '.join(questions)))
print('\nthe recorded proof as decisions  (question[number of options]=chosen option):')
for t in trace: print('   ' + t[:150])
print('\n%d actions took %d decisions; terminated=%s, solved=%s, total reward %.1f' % (len(trace), n_decisions, terminated, env.solved, total))
assert terminated and env.solved and total == 1.0
print('EXAMPLE OK')
