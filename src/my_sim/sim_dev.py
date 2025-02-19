import gymnasium as gym
from get_params import GetParams
from my_sim.gym_simulation import *
from gymnasium.utils.env_checker import check_env

'''For testing my env!'''
config = GetParams()

# Create and check the environment if changes are made
env = gym.make('MRPP_Env', **config)
check_env(env.unwrapped)

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