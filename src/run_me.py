from agent import Agent
import multiprocessing
import time
import os
import yaml
import mujoco
from mujoco import viewer
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from globals import root_dir
from get_params import GetParams

# 2D Simulation imports
from my_simulation.sim import *
from my_simulation.animation import *

############################# Import args from config.yaml #############################
config = GetParams()

# Arguments to all sim environments
args = (config['rl_alg'],config['operation'],config['num_environments'],config['epochs'],config['t_steps'],config['discount'],config['epsilon'],config['lr'],config['live_sim'],config['save_every'],config['gym_model'])

############################# Load the chosen environment and start training or inference #############################
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

    # Only use one env for recording
    if config['record']:
        env = gym.make(config['gym_model'], render_mode="rgb_array")
        n_actions = env.action_space.shape[0] #int(env.action_space.n)
        n_obs = env.observation_space.shape[0]

        # Recording video parameters
        num_training_episodes = config['epochs']  # total number of training episodes
        video_dir = os.path.join(root_dir, "videos", config['gym_model'], config['rl_alg'])
        env = RecordVideo(env, video_folder=video_dir, name_prefix="training",
                            episode_trigger=lambda x: x % config['record_period'] == 0)
        env = RecordEpisodeStatistics(env)

    else:
        env = gym.vector.SyncVectorEnv([lambda: gym.make(config['gym_model'], render_mode="rgb_array") for _ in range(config['num_environments'])])
        n_actions = env.action_space.shape[1] #int(env.action_space.nvec[0])
        n_obs = env.observation_space.shape[1] # env._observations[0].shape[0]

    # Add new variables to args
    args = args[:5] + (env,n_obs,n_actions) + args[5:]

    # Start Training
    agent = Agent(*args)
    agent.train()

    env.close()

elif config['env'] == "gym_adv":
    env = gym.vector.SyncVectorEnv([lambda: gym.make(config['gym_model'], render_mode="rgb_array") for _ in range(config['num_environments'])])
    
    # Define the observation space and action space sizes
    n_actions = int(env.action_space.nvec[0])
    n_obs = env._observations[0].shape[0]

    # Add new variables to args
    args = args[:5] + (env,n_obs,n_actions) + args[5:]

    args1 = args
    args = list(args)
    args[0] = config['rl_alg_adv']
    args2 = tuple(args)

    # Start Training
    for _ in range(config['adv_iter']):
        agent1 = Agent(*args1)
        agent2 = Agent(*args2)

        agent1.train_adv(adversary = agent2.rl_alg)
        agent2.train_adv(adversary = agent1.rl_alg)

print("Training Complete!")