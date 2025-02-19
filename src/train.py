from agent import Agent
from get_params import GetParams
from my_sim.gym_simulation import *
from create_env import CreateEnv

############################# Import args from config.yaml #############################
config = GetParams()

# Arguments to all sim environments
args = (config['rl_alg'],config['num_environments'],config['epochs'],config['t_steps'],config['discount'],config['epsilon'],config['lr'],config['save_every'],config['gym_model'],config['num_agents'], config['space'])

############################# Load the chosen environment and start training or inference #############################

# Create environment
env,n_actions,n_obs = CreateEnv(operation="train")

# Add new variables to args
args = args[:4] + (env,n_obs,n_actions) + args[4:]

# Start Training
agent = Agent(*args)
agent.train()

env.close()

# elif config['env'] == "gym_adv":
#     env = gym.vector.SyncVectorEnv([lambda: gym.make(config['gym_model'], render_mode="rgb_array") for _ in range(config['num_environments'])])
    
#     # Define the observation space and action space sizes
#     n_actions = int(env.action_space.nvec[0])
#     n_obs = env._observations[0].shape[0]

#     # Add new variables to args
#     args = args[:5] + (env,n_obs,n_actions) + args[5:]

#     args1 = args
#     args = list(args)
#     args[0] = config['rl_alg_adv']
#     args2 = tuple(args)

#     # Start Training
#     for _ in range(config['adv_iter']):
#         agent1 = Agent(*args1)
#         agent2 = Agent(*args2)

#         agent1.train_adv(adversary = agent2.rl_alg)
#         agent2.train_adv(adversary = agent1.rl_alg)

print("Training Complete!")