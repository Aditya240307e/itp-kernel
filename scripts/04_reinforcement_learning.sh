#!/usr/bin/env bash
# 04_reinforcement_learning.sh -- the learners that continue from the behaviour-cloned network of the same seed:
#     rl/ppo  rl/ppo_nosub  rl/ei  rl/ei_her  rl/ei_her_boot  rl/ei_mw  rl/ei_mcts  rl/ei_ida
# One variant:  TARGET='rl/ei_her/*' ./scripts/04_reinforcement_learning.sh
. "$(dirname "$0")/env.sh"
stage_script 04_reinforcement_learning rl "4. reinforcement learning (PPO, expert iteration, hindsight, bootstrapped library)"
