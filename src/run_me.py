from agent import Agent
import os
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from globals import root_dir
from get_params import GetParams

# 2D Simulation imports
from my_sim.gym_simulation import *

############################# Import args from config.yaml #############################
config = GetParams()

# Arguments to all sim environments
args = (config['rl_alg'],config['operation'],config['num_environments'],config['epochs'],config['t_steps'],config['discount'],config['epsilon'],config['lr'],config['live_sim'],config['save_every'],config['gym_model'])

############################# Load the chosen environment and start training or inference #############################

# Personal Simulation... this is just testing a P controller right now.
if config['env'] == "2dsim":

    # Create and check the environment if changes are made
    env = gym.make('MRPP_Env', **config)
    # check_env(env.unwrapped)

    # Reset the env
    obs, _ = env.reset()
    
    done = False
    while not done:
        # random actions
        # action = np.random.uniform(-1, 1, (config['num_agents'],2))
        action = obs[:,2:] - obs[:,0:2]
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    env.close()

# Gym
elif config['env'] == "gym":

    # Only use one env for recording
    if config['record']:
        env = gym.make(config['gym_model'], render_mode="rgb_array", **config)
        n_actions = env.action_space.shape[0]*env.action_space.shape[1] #int(env.action_space.n)
        n_obs = env.observation_space.shape[0]*env.observation_space.shape[1]

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