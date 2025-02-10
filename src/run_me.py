from agent import Agent
import multiprocessing
import time
from globals import root_directory
import os
import yaml
import mujoco
from mujoco import viewer
import gymnasium as gym

# 2D Simulation imports
from my_simulation.sim import *
from my_simulation.animation import *

############################# Import args from config.yaml #############################
config_path = os.path.join(root_directory, "config.yaml")
with open(config_path, "r") as read_file:
    config = yaml.safe_load(read_file)

# Arguments to all sim environments
args = (config['rl_alg'],config['operation'],config['num_environments'],config['epochs'],config['t_steps'],config['discount'],config['epsilon'],config['lr'],config['live_sim'])

############################# Load the chosen environment #############################
# MuJoCo
if config['env'] == "mujoco":
    model = mujoco.MjModel.from_xml_path(config['muj_model_dir'])
    data = mujoco.MjData(model)

# Personal Simulation
elif config['env'] == "2dsim":
    # Generate costmap and make obstacles
    env = Sim(grid_size=100)
    env.set_start(10, 10)
    env.set_goal(90, 90)
    env.make_obstacles(num_obstacles=75,cost_map=env.cost_map, obstacle_size=7,build_wall=False)

    n_obs = 2
    n_actions = 1 # This needs to change for continuous
    args = args[:5] + (env,n_obs,n_actions) + args[5:]

    # Initialize and show the animation
    anim = animate(env.cost_map, env.start, env.goal) 

    # Start Training
    processes = [multiprocessing.Process(target=Agent,args=args) for _ in range(config['num_environments'])]

    # For tracking sim times
    time_tracker = {}

    # Start all parallel processes
    for idx, process in enumerate(processes):
        time_tracker[idx] = time.time()
        process.start()

    # Wait for all parallel processes to finish
    for idx, process in enumerate(processes):
        process.join()
        print(f"Environment {idx+1} took {time.time()-time_tracker[idx]} seconds.")

# Gym
elif config['env'] == "gym":
    env = gym.vector.SyncVectorEnv([lambda: gym.make(config['gym_model'], render_mode="rgb_array") for _ in range(config['num_environments'])])
    
    # Define the observation space and action space sizes
    n_actions = int(env.action_space.nvec[0])
    n_obs = env._observations[0].shape[0]

    # Add new variables to args
    args = args[:5] + (env,n_obs,n_actions) + args[5:]

    # Start Training
    Agent(*args)

print("Done!")