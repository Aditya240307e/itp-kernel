import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths
import random
import numpy as np
import bank_all, bank_env

bank = bank_all.load()                                     # the world of the bank: libraries, 268 statements, reference proofs
env = bank_env.BankEnv(bank, names=['two_pow_gt_n'], max_actions=40, max_decisions=400)
obs, info = env.reset(options={'problem': 'two_pow_gt_n'})
print({k: tuple(v.shape) for k, v in env.observation_space.spaces.items() if k in ('tokens', 'rows', 'mask')})
print('first question:', info['choice'].name, '|', int(info['action_mask'].sum()), 'legal options')

rng = random.Random(0)
decisions = 0
while decisions < 50:
    legal = np.flatnonzero(info['action_mask'])            # the options of the current question that are legal
    obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
    decisions += 1
    if terminated or truncated: break
print('decisions:', info['decisions'], '| actions taken:', info['actions'], '| refused:', info['refused'], '| solved:', info['solved'])
