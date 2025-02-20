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
agent1 = Agent(*args)
agent2 = Agent(*args)

agent1_reward = 0
agent2_reward = 0
adv_iters = config['adv_iter']

for adv_iter in range(adv_iters):
    print(f'Adversarial Episode {adv_iter}/{adv_iters}')

    print('Protagonist Turn')
    agent1.train_adv(adversary = agent2.rl_alg, player = "protagonist")
    print('Adversary Turn')
    agent2.train_adv(adversary = agent1.rl_alg, player = "adversary")

    # Update tensorboard and terminal
    writer.add_scalars("", {
        "Protagonist": agent1.total_reward,
        "Adversary": agent2.total_reward
    }, adv_iter)
    writer.flush()

env.close()

print("Training Complete!")