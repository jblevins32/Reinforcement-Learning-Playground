import torch
import gymnasium as gym
import os
from gym.wrappers import RecordVideo
from globals import root_dir
from get_params import GetParams
from RL_algorithms.reinforce import *
from RL_algorithms.vpg import *
from RL_algorithms.ppo import *
from RL_algorithms.ppo_adv import *
from RL_algorithms.ppo_cont import *
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

config = GetParams()

env = gym.make(config['gym_model'], render_mode="rgb_array")

# Recording video parameters
num_training_episodes = config['epochs']  # total number of training episodes
video_dir = os.path.join(root_dir, "videos", config['gym_model'], config['rl_alg'])
env = RecordVideo(env, video_folder=video_dir, name_prefix="testing",
                    episode_trigger=lambda x: x % config['record_period'] == 0)
env = RecordEpisodeStatistics(env)

n_actions = env.action_space.shape[0] #int(env.action_space.n)
n_obs = env.observation_space.shape[0]

# Choose RL algorithm
if config['rl_alg'] == "PPO":
    rl_alg = PPO(input_dim=n_obs, output_dim=n_actions, epsilon=config['epsilon'])
elif config['rl_alg'] == "REINFORCE":
    rl_alg = REINFORCE(input_dim=n_obs, output_dim=n_actions)
elif config['rl_alg'] == "VPG":
    rl_alg = VPG(input_dim=n_obs, output_dim=n_actions)
elif config['rl_alg'] =="PPO_ADV":
    rl_alg = PPO_ADV(input_dim=n_obs, output_dim=n_actions, epsilon=config['epsilon'])
elif config['rl_alg'] =="PPO_CONT":
    rl_alg = PPO_CONT(input_dim=n_obs, output_dim=n_actions, epsilon=config['epsilon'])

# Load the model parameters
model_dir = os.path.join(root_dir,"models",f"{config['gym_model']}_{config['rl_alg']}_2.12313.pth")
rl_alg.load_state_dict(torch.load(model_dir))

# Run inference and record video
obs = env.reset()
obs = obs[0]
done = False
while not done:
    with torch.no_grad():
        mean = rl_alg.policy(torch.Tensor(obs))
        std = torch.exp(rl_alg.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().numpy()

    obs, reward, done, truncated, _ = env.step(action)
    print('step taken')
    done = done or truncated

env.close()