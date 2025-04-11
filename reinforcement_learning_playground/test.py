import torch
import os
from global_dir import root_dir
from get_params import GetParams
from RL_algorithms.reinforce import *
from RL_algorithms.vpg import *
from RL_algorithms.ppo_disc import *
from RL_algorithms.ppo import *
from RL_algorithms.sac import *
from RL_algorithms.ddpg import *
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from my_sim.gym_simulation import *
from create_env import CreateEnv
from agent import Agent

config = GetParams()
device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define model directory and parse info from it
model_dir = os.path.join(root_dir,"models",config['test_model']) 
gym_model = config['test_model'].split('_')[0]
rl_alg_name = config['test_model'].split('_')[1]

env,n_obs,n_actions,writer,config = CreateEnv(operation="test", open_local=False)

# Recording video parameters
num_training_episodes = config['episodes']  # total number of training episodes
video_dir = os.path.join(root_dir, "videos", gym_model, rl_alg_name)
env = RecordVideo(env, video_folder=video_dir, name_prefix=f"{config['test_model']}_{config['test_steps']}_steps",
                    episode_trigger=lambda x: x % 1 == 0)
env = RecordEpisodeStatistics(env)

# THIS HERE IS FOR MY ENV
# Run inference and record video
# obs = env.reset()
# obs = obs[0]
# done = False
# count_steps = 0
# while (count_steps < config['test_steps']):
# # while (done is False) and (count_steps < config['test_steps']):
#     count_steps += 1
#     with torch.no_grad():
#         mean = rl_alg.policy(torch.Tensor(obs.flatten()))
#         std = torch.exp(rl_alg.log_std)
#         dist = torch.distributions.Normal(mean, std)
#         action = dist.sample()

#     obs, reward, done, truncated, _ = env.step(action.numpy())
#     env.render()
#     print(f'step taken {count_steps}')
#     done = done or truncated

# Setup agent
agent = Agent(env, n_obs, n_actions, writer, **config)

# Load the agent model parameters
agent.rl_alg.load_state_dict(torch.load(model_dir,map_location=device))

obs = env.reset()
obs = obs[0]
done = False
count_steps = 0
total_reward = 0
while (count_steps < config['test_steps']):
# while (done is False) and (count_steps < config['test_steps']):
    count_steps += 1
    with torch.no_grad():
        action,_,_ = agent.GetAction(torch.tensor(obs,device=device).to(torch.float), target=False,grad=False)

    # Attack the action
    if config['test_attack']:
        action += torch.randn_like(action) * 1

    obs, reward, done, truncated, _ = env.step(action.cpu().numpy())
    total_reward += reward
    env.render()
    print(f'step taken {count_steps}')
    done = done or truncated

print(f'Total Reward: {total_reward}, Video saved to {video_dir}')

env.close()