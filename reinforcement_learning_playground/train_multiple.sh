#!/bin/bash

# Run in series
# python3 train.py --rl_alg PPO
# python3 train.py --rl_alg DDPG
# python3 train.py --rl_alg TD3

# Run in parallel
python3 train.py --rl_alg PPO &
python3 train.py --rl_alg DDPG &
python3 train.py --rl_alg TD3 &
wait