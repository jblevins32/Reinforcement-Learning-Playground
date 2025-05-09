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
# python3 train.py --rl_alg DDPG --alter_gravity 1 --alter_friction 1 --alter_plot_name normal_gravity_normal_friction &
# python3 train.py --rl_alg DDPG --alter_gravity 10 --alter_friction 1 --alter_plot_name 10x_gravity_normal_friction &
# python3 train.py --rl_alg DDPG --alter_gravity 1 --alter_friction 10 --alter_plot_name normal_gravity_10x_friction &
# python3 train.py --rl_alg DDPG --alter_gravity 10 --alter_friction 10 --alter_plot_name 10x_gravity_10x_friction &
# python3 train.py --rl_alg DDPG --alter_gravity 1 --alter_friction 1 --disturb_limit 0.5 --disturb_rate 0.05 --alter_plot_name normal_gravity_normal_friction_0.5_disturb &
# python3 train.py --rl_alg DDPG --alter_gravity 10 --alter_friction 1 --disturb_limit 0.5 --disturb_rate 0.05 --alter_plot_name 10x_gravity_normal_friction_0.5_disturb &
# python3 train.py --rl_alg DDPG --alter_gravity 1 --alter_friction 10 --disturb_limit 0.5 --disturb_rate 0.05 --alter_plot_name normal_gravity_10x_friction_0.5_disturb &
# python3 train.py --rl_alg DDPG --alter_gravity 10 --alter_friction 10 --disturb_limit 0.5 --disturb_rate 0.05 --alter_plot_name 10x_gravity_10x_friction_0.5_disturb &
# wait

# Test different policies against eachother
python3 test.py --rl_alg DDPG --alter_plot_name normal_gravity_normal_friction_trained --model Ant-v5_DDPG_1.38703_2025-04-18_01-17_normal_gravity_normal_friction.pth &
python3 test.py --rl_alg DDPG --alter_plot_name 10x_gravity_normal_friction_trained --model Ant-v5_DDPG_2.94184_2025-04-18_03-13_10x_gravity_normal_friction.pth &
python3 test.py --rl_alg DDPG --alter_plot_name normal_gravity_10x_friction_trained --model Ant-v5_DDPG_0.86775_2025-04-17_21-50_normal_gravity_10x_friction.pth &
python3 test.py --rl_alg DDPG --alter_plot_name 10x_gravity_10x_friction_trained --model Ant-v5_DDPG_0.63077_2025-04-17_21-49_10x_gravity_10x_friction.pth &
python3 test.py --rl_alg DDPG --alter_plot_name normal_gravity_normal_friction_trained --model Ant-v5_DDPG_1.01706_2025-04-18_23-00_normal_gravity_normal_friction_0.5_disturb.pth &
python3 test.py --rl_alg DDPG --alter_plot_name 10x_gravity_normal_friction_trained --model Ant-v5_DDPG_2.57938_2025-04-19_03-17_10x_gravity_normal_friction_0.5_disturb.pth &
python3 test.py --rl_alg DDPG --alter_plot_name normal_gravity_10x_friction_trained --model Ant-v5_DDPG_0.85009_2025-04-18_21-36_normal_gravity_10x_friction_0.5_disturb.pth &
python3 test.py --rl_alg DDPG --alter_plot_name 10x_gravity_10x_friction_trained --model Ant-v5_DDPG_0.63178_2025-04-18_22-32_10x_gravity_10x_friction_0.5_disturb.pth &
wait


