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

# Train different env parameters
python3 train.py --rl_alg DDPG --alter_gravity 1 --alter_friction 1 --alter_plot_name normal_gravity_normal_friction &
python3 train.py --rl_alg DDPG --alter_gravity 10 --alter_friction 1 --alter_plot_name 10x_gravity_normal_friction &
python3 train.py --rl_alg DDPG --alter_gravity 100 --alter_friction 1 --alter_plot_name 100x_gravity_normal_friction &
python3 train.py --rl_alg DDPG --alter_gravity 1 --alter_friction 10 --alter_plot_name normal_gravity_10x_friction &
python3 train.py --rl_alg DDPG --alter_gravity 10 --alter_friction 10 --alter_plot_name 10x_gravity_10x_friction &
python3 train.py --rl_alg DDPG --alter_gravity 100 --alter_friction 10 --alter_plot_name 100x_gravity_10x_friction &
python3 train.py --rl_alg DDPG --alter_gravity 1 --alter_friction 100 --alter_plot_name normal_gravity_100x_friction &
python3 train.py --rl_alg DDPG --alter_gravity 10 --alter_friction 100 --alter_plot_name 10x_gravity_100x_friction &
python3 train.py --rl_alg DDPG --alter_gravity 100 --alter_friction 100 --alter_plot_name 100x_gravity_100x_friction &
wait

# Test different policies against eachother
# python3 test.py --rl_alg DDPG --alter_plot_name normal_gravity_trained --model Ant-v5_DDPG_-2.08534_2025-04-17_19-31_normal_gravity.pth &
# python3 test.py --rl_alg DDPG --alter_plot_name 10x_gravity_trained --model Ant-v5_DDPG_-2.1239_2025-04-17_19-31_10x_gravity.pth &
# python3 test.py --rl_alg DDPG --alter_plot_name 100x_gravity_trained --model Ant-v5_DDPG_-1.54667_2025-04-17_19-31_100x_gravity.pth &
# python3 test.py --rl_alg DDPG --alter_plot_name 1000x_gravity_trained --model Ant-v5_DDPG_-1.54805_2025-04-17_19-31_1000x_gravity.pth &
# python3 test.py --rl_alg DDPG --alter_plot_name 10000x_gravity_trained --model &
# wait