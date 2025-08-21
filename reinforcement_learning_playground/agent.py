from traj_data import *
from replay_buffer import ReplayBuffer
from global_dir import root_dir
import os
import time
from datetime import datetime
import numpy as np
from torch.distributions import Categorical
from gymnasium.spaces import Discrete, Box
from get_action import GetAction
from get_params_args import *
from domain_rand import Randomdisturbs
from timer import units
from tensorboard_setup import get_log_dir

from RL_algorithms.reinforce import *
from RL_algorithms.vpg import *
from RL_algorithms.ppo_disc import *
from RL_algorithms.ppo import *
from RL_algorithms.sac import *
from RL_algorithms.ddpg import *
from RL_algorithms.td3 import *

from stable_baselines3 import DDPG as SB3_DDPG

############################################################################################################

# Load and run the agent


class Agent():
    def __init__(self, env, n_obs, n_actions, writer, **kwargs):

        self.env = env
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.writer = writer

        self.episodes = kwargs.get('episodes', 1000)
        self.discount = kwargs.get('discount', 0.99)
        self.t_steps = kwargs.get('t_steps', 256)
        self.save_every = kwargs.get('save_every', 250)
        self.gym_model = kwargs.get('gym_model', 'Ant-v5')
        self.num_environments = kwargs.get('num_environments', 64)
        self.num_agents = kwargs.get('num_agents', 1)
        self.rl_alg_name = kwargs.get('rl_alg_name', 'PPO')
        self.epsilon = kwargs.get('epsilon', 0.2)
        self.lr = kwargs.get('lr', 1e-3)
        self.gamma = kwargs.get('gamma', 0.99)
        self.load_dict = kwargs.get('load_dict', False)
        self.load_path = kwargs.get('load_path', "0")
        self.adv_iters = kwargs.get('adv_iter', 10)
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Domain Randomization parameters
        self.alter_plot_name = kwargs.get('alter_plot_name', 'no-mods')
        self.disturb_limit = kwargs.get('disturb_limit', 0)
        self.disturb_rate = kwargs.get('disturb_rate', 0)

        # Define the space as cont of disc
        if isinstance(self.env.action_space, Box):
            self.space = "cont"
        else: self.space = "disc"

        # rl_alg,num_environments,episodes,t_steps,env,n_obs,n_actions,discount, epsilon, lr, save_every, gym_model, num_agents, space, writer
        # Initialize plot variables
        self.episode_vec = []
        self.reward_vec = []
        self.frames = []

        # Choose RL algorithm
        if self.rl_alg_name == "PPO_DISC":
            self.rl_alg = PPO_DISC(
                input_dim=n_obs, output_dim=n_actions, epsilon=self.epsilon)
        elif self.rl_alg_name == "REINFORCE":
            self.rl_alg = REINFORCE(input_dim=n_obs, output_dim=n_actions)
        elif self.rl_alg_name == "VPG":
            self.rl_alg = VPG(input_dim=n_obs, output_dim=n_actions, lr=self.lr)
        elif self.rl_alg_name == "SAC":
            self.rl_alg = SAC(input_dim=n_obs, output_dim=n_actions, lr=self.lr)
        elif self.rl_alg_name == "DDPG":
            self.rl_alg = DDPG(input_dim=n_obs, output_dim=n_actions, lr=self.lr)
        elif self.rl_alg_name =="PPO":
            self.rl_alg = PPO(input_dim=n_obs, output_dim=n_actions, lr=self.lr)
        elif self.rl_alg_name =="TD3":
            self.rl_alg = TD3(input_dim=n_obs, output_dim=n_actions, lr=self.lr)
        elif self.rl_alg_name == "SB3_DDPG":
            self.rl_alg = SB3_DDPG("MlpPolicy", env, verbose=1, learning_rate=self.lr, buffer_size=1000000, batch_size=256, tau=0.1, gamma=self.gamma, tensorboard_log=get_log_dir(self.gym_model, self.rl_alg_name, self.alter_plot_name))

        # Load state dictionary if continuing training
        if self.load_dict:
            model_dir = os.path.join(root_dir,"models",self.load_path) # If starting training is a pretrained model
            self.rl_alg.load_state_dict(torch.load(model_dir,map_location=self.device))

        # Create traj data or buffer
        if self.rl_alg.on_off_policy == "off":
            self.buffer = ReplayBuffer()
        elif self.rl_alg.on_off_policy == "on":
            self.traj_data = TrajData(n_steps=self.t_steps, n_envs=self.num_environments,
                                      n_obs=n_obs, n_actions=n_actions, space=self.space)

    def train(self):
        total_steps = self.episodes * self.t_steps * self.num_environments

        # SB3 training
        if self.rl_alg_name[0:3] == "SB3":
            self.rl_alg.learn(total_timesteps=total_steps, tb_log_name=get_log_dir(self.gym_model, self.rl_alg_name, self.alter_plot_name))  # Initialize the model without training

        # My training
        else:
            time_start_train = time.time()
            print(f"Training for {total_steps} steps")

            # Running for n episodes
            for episode in range(self.episodes):

                time_start_episode = time.time()

                # Reset the environment at beginning of each episode
                obs, _ = self.env.reset()
                obs = torch.tensor(obs, device=self.device).to(torch.float)

                # Rollout
                if self.space == "cont":
                    avg_reward = self.rollout_cont(obs)
                elif self.space == "disc":
                    self.rollout_disc(obs)

                # Update parameters
                loss_policy, loss_critic = self.update(episode)
                
                reward_to_log = round(avg_reward,5)
                loss_to_log_policy = round(loss_policy.item(),5)
                loss_to_log_critic = round(loss_critic.item(),5)

                # Update tensorboard and terminal
                total_steps_curr = (episode+1)*self.t_steps*self.num_environments
                self.writer.add_scalar("episodic reward", reward_to_log, total_steps_curr)
                self.writer.add_scalar("loss/policy", loss_to_log_policy, total_steps_curr)
                self.writer.add_scalar("loss/critic", loss_to_log_critic, total_steps_curr)
                self.writer.add_scalar("exploration rate", self.rl_alg.exploration_rate, total_steps_curr)
                self.writer.flush()

                episode_runtime = time.time()-time_start_episode
                total_runtime = time.time()-time_start_train
                episode_runtime_avg = total_runtime/(episode+1)
                if episode > 4:
                    total_runtime_estimate = np.round(self.episodes*episode_runtime_avg,3)
                else:
                    total_runtime_estimate = 0
                    print(f"{5-episode} more episodes to estimate total runtime")
                print(f"Episode {episode + 1}, Step {total_steps_curr}: Total runtime {units(total_runtime)}/{units(total_runtime_estimate)}, {np.round(100*total_runtime/(self.episodes*episode_runtime_avg),4)}% done, episode runtime {np.round(episode_runtime,3)} sec, Reward: {reward_to_log}, Policy Loss: {loss_to_log_policy}, Critic Loss: {loss_to_log_critic}")

                # Save the model iteratively, naming based on final reward
                if ((episode + 1) % self.save_every == 0) and episode != 0:
                    model_dir = os.path.join(
                        root_dir, "models", f"{self.gym_model}_{self.rl_alg.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{self.alter_plot_name}_reward_{reward_to_log}.pth")
                    os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)
                    torch.save(self.rl_alg.state_dict(), model_dir)
                    print('Policy saved at', model_dir)

    def rollout_disc(self, obs):
        # Rollout for t timesteps
        for t in range(self.t_steps):

            # Step 1: forward pass on the actor and critic to get action and value
            with torch.no_grad() if self.rl_alg.need_grad == False else torch.enable_grad():
                logits = self.rl_alg.policy(obs)

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            probs = Categorical(logits=logits)
            actions = probs.sample()
            log_probs = probs.log_prob(actions)

            # Step 3: take the action in the environment, using the action as a control command to the robot model.
            obs_new, reward, done, truncated, infos = self.env.step(
                actions.numpy())
            done = done | truncated  # Change done if the episode is truncated

            # Step 4: store data in traj_data
            self.traj_data.store(t, obs, actions, reward, log_probs, done)
            obs = torch.tensor(obs_new, device=self.device).to(torch.float)

    def rollout_cont(self, obs):
        total_reward = 0

        # If disturb rate is set, create a random vector of disturb times to perturb the agent in the rollout
        t_disturbed = np.random.uniform(0,1,size=self.t_steps) < self.disturb_rate

        # Rollout for t timesteps
        for t in range(self.t_steps):

            # Get an action from the policy based on the current observation
            actions, log_probs, _ = GetAction(self.rl_alg, obs, target=False, grad = self.rl_alg.need_grad, noisy = self.rl_alg.need_noisy)

            # Domain randomization for disturb forces at specific intervals and random magnitudes. Will not happen unless flag is used
            if t_disturbed[t]:
                self.env = Randomdisturbs(self.env, self.disturb_limit)

            # Take the action in the environment
            obs_new, reward, done, truncated, infos = self.env.step(
                actions.cpu().numpy())
            done = done | truncated  # Change done if the episode is truncated

            # Store data in traj_data or buffer

            # Use traj_data for on-policy
            if self.rl_alg.on_off_policy == "on":

                # traj data needs numpy not tensors
                self.traj_data.store(t, obs, actions, reward*0.1, log_probs, done)
                reward = torch.tensor(
                    reward, device=self.device).to(torch.float)                
                obs = torch.tensor(obs_new, device=self.device).to(torch.float)
            
            # Use a buffer for off policy
            elif self.rl_alg.on_off_policy == "off": 

                # Buffer needs tensors not numpy
                obs_new = torch.tensor(
                    obs_new, device=self.device).to(torch.float)   
                reward = torch.tensor(
                    reward, device=self.device).to(torch.float)              
                self.buffer.store(obs, actions, reward, obs_new, done)
                obs = obs_new
            
            total_reward += reward

        # Average reward per step for this rollout
        avg_reward = total_reward.mean().item() / self.t_steps
        return avg_reward
    
    def update(self, episode):

        # Updates for on policy
        if self.rl_alg.on_off_policy == "on":
            # Get total expected return from the rollout in this episode
            self.traj_data.calc_returns()

            # Update networks
            update_epochs = 10 if (self.rl_alg.name == "PPO") else 1

            for _ in range(update_epochs):
                loss_policy, loss_critic = self.rl_alg.loss_func(self.traj_data)
                loss_total = loss_policy + loss_critic # This is for PPO and VPG
                self.rl_alg.policy_optimizer.zero_grad()
                loss_total.backward()
                self.rl_alg.policy_optimizer.step()

            # Dump the traj data for this rollout
            self.traj_data.detach()

        # Updates for off policy
        elif self.rl_alg.on_off_policy == "off":

            epochs_critic = 64
            epochs_policy = 4

            # Critic update

            for _ in range(epochs_critic):
                loss_critic = self.rl_alg.loss_func_critic(*self.buffer.sample())
                self.rl_alg.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.rl_alg.critic_optimizer.step()

            # Policy update with a delay if called for (only TD3 thus far)
            loss_policy = torch.tensor(0)
            if episode % self.rl_alg.policy_update_delay == 0:
                for _ in range(epochs_policy):
                    loss_policy = self.rl_alg.loss_func_policy(*self.buffer.sample())
                    self.rl_alg.policy_optimizer.zero_grad()
                    loss_policy.backward()
                    self.rl_alg.policy_optimizer.step()
        else:
            raise Exception('On or off policy not defined for this algorithm')

        # Target updates
        if self.rl_alg.target_updates:
            if self.rl_alg.name == 'SAC' or self.rl_alg.name == 'TD3':
                if episode % self.rl_alg.critic_update_delay == 0: # Update the critic at some interval for TD3
                    for target_param, param in zip(self.rl_alg.critic_1_target.parameters(), self.rl_alg.critic_1.parameters()):
                        target_param.data.copy_(
                            self.rl_alg.tau * param.data + (1 - self.rl_alg.tau) * target_param.data)

                    for target_param, param in zip(self.rl_alg.critic_2_target.parameters(), self.rl_alg.critic_2.parameters()):
                        target_param.data.copy_(
                            self.rl_alg.tau * param.data + (1 - self.rl_alg.tau) * target_param.data)
                
                if episode % self.rl_alg.policy_update_delay == 0:
                    for target_param, param in zip(self.rl_alg.policy_target.parameters(), self.rl_alg.policy.parameters()):
                        target_param.data.copy_(
                            self.rl_alg.tau * param.data + (1 - self.rl_alg.tau) * target_param.data)
            
            elif self.rl_alg_name == 'DDPG':
                if episode % self.rl_alg.critic_update_delay == 0:
                    for target_param, param in zip(self.rl_alg.critic_target.parameters(), self.rl_alg.critic.parameters()):
                        target_param.data.copy_(
                            self.rl_alg.tau * param.data + (1 - self.rl_alg.tau) * target_param.data)

                if episode % self.rl_alg.policy_update_delay == 0:
                    for target_param, param in zip(self.rl_alg.policy_target.parameters(), self.rl_alg.policy.parameters()):
                        target_param.data.copy_(
                            self.rl_alg.tau * param.data + (1 - self.rl_alg.tau) * target_param.data)
        
        # Update exploration rate, keeping it above 0.05
        if self.rl_alg.explore is True:
            self.rl_alg.exploration_rate  = max(self.rl_alg.exploration_rate  * 0.99, 0.05)

        return loss_policy, loss_critic

############################# ADVERSARIAL #####################################
    def train_adv(self, adversary, player_identifier):

        time_start_train = time.time()
        self.total_reward = 0  # Track total reward for this training run

        # Running for n episodes
        for episode in range(self.episodes):

            time_start_episode = time.time()

            # Reset the environment at beginning of each episode
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, device=self.device).to(torch.float)

            # Rollout
            if self.space == "cont":
                avg_reward = self.rollout_cont_adv(obs, adversary, player_identifier)
            elif self.space == "disc":  # Not implemented for discrete
                self.rollout_disc(obs)

            # Update parameters
            policy_loss, critic_loss = self.update()

            # Store reward for this episode: for adervarial plotting
            self.total_reward += avg_reward
            reward_to_log = round(avg_reward,5)
            loss_to_log_policy = -round(policy_loss.item(),5)
            loss_to_log_critic = -round(critic_loss.item(),5)

            episode_runtime = time.time()-time_start_episode
            total_runtime = time.time()-time_start_train
            episode_runtime_avg = total_runtime/(episode+1)
            print(f"Completed episode {episode + 1}: Estimated overall runtime {np.round(self.adv_iters*self.episodes*episode_runtime_avg/60,3)}, Episode runtime {np.round(total_runtime/60,3)}/{np.round(self.episodes*episode_runtime_avg/60,3)} min, {np.round(100*total_runtime/(self.episodes*episode_runtime_avg),4)}% done, episode runtime {np.round(episode_runtime,3)} sec, Reward: {reward_to_log}, Policy Loss: {loss_to_log_policy}")

    def rollout_cont_adv(self, obs, adversary, player_identifier):
        total_reward = 0

        # Rollout for t timesteps
        for t in range(self.t_steps):

            # PLAYER (Whichever protagonist or adversary is rolling out) Step 1: forward pass on the actor and critic to get action and value
            actions, log_probs, _ = GetAction(obs, target=False, grad=False)

            # Off player Step 1: forward pass on the actor and critic to get action and value NEED TO SOMEHOW DEFINE THAT THIS IS THE ANTAGONIST
            actions_adv, log_probs_adv, _ = GetAction(obs, target=False, grad=False)

            # Step 2: combine actions
            actions_combined = actions + actions_adv

            # Step 3: take the action in the environment
            obs_new, reward, done, truncated, infos = self.env.step(
                actions_combined.cpu().numpy())
            done = done | truncated  # Change done if the episode is truncated

            # Make adversary have opposite reward
            if player_identifier != "adversary":
                reward = -reward

             # Store data in traj_data or buffer
            if self.rl_alg.on_off_policy == "on":
                self.traj_data.store(t, obs, actions, reward, log_probs, done)
                reward = torch.tensor(
                    reward, device=self.device).to(torch.float)                
                obs = torch.tensor(obs_new, device=self.device).to(torch.float)

            elif self.rl_alg.on_off_policy == "off": # Use a buffer for off policy
                obs_new = torch.tensor(
                    obs_new, device=self.device).to(torch.float)   
                reward = torch.tensor(
                    reward, device=self.device).to(torch.float)              
                self.buffer.store(obs, actions, reward*0.01, obs_new, done)
                obs = obs_new
            
            total_reward += reward

        # Average reward for this rollout
        avg_reward = total_reward.mean().item() / self.t_steps
        return avg_reward
