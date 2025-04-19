import torch
import os
from global_dir import root_dir
from get_params_args import *
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
from get_action import GetAction

config = GetParams()
args = GetArgs()

# Overwrite model file from args else use params from config
if args.model is not None:
    model_file = args.model
else:
    model_file = config['test_model']

device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define model directory and parse info from it
model_dir = os.path.join(root_dir,"models",model_file) 
# gym_model = model_file.split('_')[0]
gym_model = config['gym_model_test']
rl_alg_name = model_file.split('_')[1]

env,n_obs,n_actions,writer,config = CreateEnv(operation="test")

# Recording video parameters
num_training_episodes = config['episodes']  # total number of training episodes
video_dir = os.path.join(root_dir, "videos", gym_model, rl_alg_name)
env = RecordVideo(env, video_folder=video_dir, name_prefix=f"{model_file}_{config['test_steps']}_steps",
                    episode_trigger=lambda x: x % config['record_period'] == 0)
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

reward_runs = [] # Vector of rewards for each run/episode
for test in range(config['num_tests']):
    obs = env.reset()
    obs = obs[0]
    done = False
    count_steps = 0
    total_reward_ep = 0
    while (count_steps < config['test_steps']):
    # while (done is False) and (count_steps < config['test_steps']):
        count_steps += 1
        with torch.no_grad():
            action,_,_ = GetAction(agent.rl_alg, torch.tensor(obs,device=device).to(torch.float), target=False,grad=False)
            # print(action)
        # Attack the action
        # if config['test_attack']:
        #     action += torch.randn_like(action) * 1

        obs, reward, done, truncated, _ = env.step(action.cpu().numpy())
        total_reward_ep += reward
        env.render()
        # print(f'step taken {count_steps}')
        done = done or truncated

    print(f'Test episode {test} total reward {total_reward_ep}')
    reward_runs.append(total_reward_ep)
    env.close()

print(f'Avg Reward: {np.mean(reward_runs)}, std: {np.std(reward_runs)}, Model: {model_file}, Video saved to {video_dir}')

