import torch
import os
from global_dir import root_dir
from get_params import GetParams
from RL_algorithms.reinforce import *
from RL_algorithms.vpg import *
from RL_algorithms.ppo_disc import *
from RL_algorithms.ppo_cont import *
from RL_algorithms.sac import *
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from my_sim.gym_simulation import *
from create_env import CreateEnv
from get_action import GetAction

config = GetParams()

env,n_obs,n_actions,writer,config = CreateEnv(operation="test")

# Recording video parameters
num_training_episodes = config['epochs']  # total number of training episodes
video_dir = os.path.join(root_dir, "videos", config['gym_model_test'], config['rl_alg'])
env = RecordVideo(env, video_folder=video_dir, name_prefix=f"testing_{config['test_model_reward']}_reward_{config['test_steps']}_steps",
                    episode_trigger=lambda x: x % config['record_period'] == 0)
env = RecordEpisodeStatistics(env)

# Choose RL algorithm
if config['rl_alg'] == "PPO":
    rl_alg = PPO_DISC(input_dim=n_obs, output_dim=n_actions, epsilon=config['epsilon'])
elif config['rl_alg'] == "REINFORCE":
    rl_alg = REINFORCE(input_dim=n_obs, output_dim=n_actions)
elif config['rl_alg'] == "VPG":
    rl_alg = VPG(input_dim=n_obs, output_dim=n_actions)
elif config['rl_alg'] =="PPO_CONT":
    rl_alg = PPO_CONT(input_dim=n_obs, output_dim=n_actions, lr=config['lr'])
elif config['rl_alg'] =="SAC":
    rl_alg = SAC(input_dim=n_obs, output_dim=n_actions, lr=config['lr'])

# Load the model parameters
model_dir = os.path.join(root_dir,"models",f"{config['gym_model_test']}_{config['rl_alg']}_{config['test_model_reward']}.pth")
rl_alg.load_state_dict(torch.load(model_dir))

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

# Run inference and record video

obs = env.reset()
obs = obs[0]
done = False
count_steps = 0
while (count_steps < config['test_steps']):
# while (done is False) and (count_steps < config['test_steps']):
    count_steps += 1
    with torch.no_grad():
        action,_,_ = GetAction(rl_alg, torch.Tensor(obs), target=False,grad=False)

    obs, reward, done, truncated, _ = env.step(action.numpy())
    env.render()
    print(f'step taken {count_steps}')
    done = done or truncated

env.close()