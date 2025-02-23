from agent import Agent
from get_params import GetParams
from my_sim.gym_simulation import *
from create_env import CreateEnv
from tensorboard_setup import SetupBoard

# Import args from config.yaml
config = GetParams()

# Tensor board setup
writer = SetupBoard(config['rl_alg'])

# Create environment
env,n_actions,n_obs = CreateEnv(operation="train")

# Arguments to all sim environments
args = (config['rl_alg'],config['num_environments'],config['epochs'],config['t_steps'],env,n_obs,n_actions,config['discount'],config['epsilon'],config['lr'],config['save_every'],config['gym_model'],config['num_agents'], config['space'], writer)

# Start Training
agent = Agent(*args)
agent.train()

env.close()

print("Training Complete!")