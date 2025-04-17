#!/bin/bash

# Run in series
# python3 train.py --rl_alg PPO
# python3 train.py --rl_alg DDPG
# python3 train.py --rl_alg TD3

# Run in parallel
# python3 train.py --rl_alg PPO &
# python3 train.py --rl_alg SAC &
# python3 train.py --rl_alg DDPG &
# python3 train.py --rl_alg TD3 &
# wait

# Run different gravities
python3 train.py --rl_alg DDPG --alter_plot_name normal_gravity &
python3 train.py --rl_alg DDPG --alter_gravity 10 --alter_plot_name 10x_gravity &
python3 train.py --rl_alg DDPG --alter_gravity 100 --alter_plot_name 100x_gravity &
python3 train.py --rl_alg DDPG --alter_gravity 1000 --alter_plot_name 1000x_gravity &
python3 train.py --rl_alg DDPG --alter_gravity 10000 --alter_plot_name 10000x_gravity &
wait
