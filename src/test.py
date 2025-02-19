import torch
import os
from globals import root_dir
from get_params import GetParams
from RL_algorithms.reinforce import *
from RL_algorithms.vpg import *
from RL_algorithms.ppo import *
from RL_algorithms.ppo_adv import *
from RL_algorithms.ppo_cont import *
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from my_sim.gym_simulation import *
from create_env import CreateEnv

config = GetParams()

env,n_actions,n_obs = CreateEnv(operation="test")

# Recording video parameters
num_training_episodes = config['epochs']  # total number of training episodes
video_dir = os.path.join(root_dir, "videos", config['gym_model'], config['rl_alg'])
env = RecordVideo(env, video_folder=video_dir, name_prefix=f"testing_{config['test_model_reward']}_reward_{config['test_steps']}_steps",
                    episode_trigger=lambda x: x % config['record_period'] == 0)
env = RecordEpisodeStatistics(env)

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
model_dir = os.path.join(root_dir,"models",f"{config['gym_model']}_{config['rl_alg']}_{config['test_model_reward']}.pth")
rl_alg.load_state_dict(torch.load(model_dir))

# Run inference and record video
obs = env.reset()
obs = obs[0]
done = False
count_steps = 0
while (done is False) and (count_steps < config['test_steps']):
    count_steps += 1
    with torch.no_grad():
        mean = rl_alg.policy(torch.Tensor(obs.flatten()))
        std = torch.exp(rl_alg.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()

    obs, reward, done, truncated, _ = env.step(action.numpy())
    print(f'step taken {count_steps}')
    done = done or truncated

env.close()