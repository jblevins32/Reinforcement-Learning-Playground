import matplotlib.pyplot as plt
from traj_data import *
from replay_buffer import ReplayBuffer
from global_dir import root_dir
import os
import time
import numpy as np
from torch.distributions import Normal
from torch.distributions import Categorical
from get_action import GetAction
from RL_algorithms.reinforce import *
from RL_algorithms.vpg import *
from RL_algorithms.ppo_disc import *
from RL_algorithms.ppo_cont import *
from RL_algorithms.sac import *
from RL_algorithms.ddpg import *

############################################################################################################

# Load and run the agent


class Agent():
    def __init__(self, env, n_obs, n_actions, writer, **kwargs):

        self.env = env
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.writer = writer

        self.epochs = kwargs.get('epochs', 1000)
        self.discount = kwargs.get('discount', 0.99)
        self.t_steps = kwargs.get('t_steps', 256)
        self.save_every = kwargs.get('save_every', 250)
        self.gym_model = kwargs.get('gym_model_train', 'Ant-v5')
        self.num_environments = kwargs.get('num_environments', 64)
        self.num_agents = kwargs.get('num_agents', 1)
        self.space = kwargs.get('space', 'CONT')
        self.rl_alg = kwargs.get('rl_alg', 'PPO_CONT')
        self.epsilon = kwargs.get('epsilon', 0.2)
        self.lr = kwargs.get('lr', 1e-3)
        self.gamma = kwargs.get('gamma', 0.99)
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        # rl_alg,num_environments,epochs,t_steps,env,n_obs,n_actions,discount, epsilon, lr, save_every, gym_model, num_agents, space, writer
        # Initialize plot variables
        self.epoch_vec = []
        self.reward_vec = []
        self.frames = []

        # Choose RL algorithm
        if self.rl_alg == "PPO_DISC":
            self.rl_alg = PPO_DISC(
                input_dim=n_obs, output_dim=n_actions, epsilon=self.epsilon)
        elif self.rl_alg == "REINFORCE":
            self.rl_alg = REINFORCE(input_dim=n_obs, output_dim=n_actions)
        elif self.rl_alg == "VPG":
            self.rl_alg = VPG(input_dim=n_obs, output_dim=n_actions)
        elif self.rl_alg == "SAC":
            self.rl_alg = SAC(
                input_dim=n_obs, output_dim=n_actions, lr=self.lr)
        elif self.rl_alg == "DDPG":
            self.rl_alg = DDPG(input_dim=n_obs, output_dim=n_actions, lr=self.lr)
        elif self.rl_alg =="PPO_CONT":
            self.rl_alg = PPO_CONT(input_dim=n_obs, output_dim=n_actions, lr=self.lr)

        # Create traj data or buffer
        if self.rl_alg.on_off_policy == "off":
            self.buffer = ReplayBuffer()
        elif self.rl_alg.on_off_policy == "on":
            self.traj_data = TrajData(n_steps=self.t_steps, n_envs=self.num_environments,
                                      n_obs=n_obs, n_actions=n_actions, space=self.space)

    def train(self):

        time_start_train = time.time()

        # Running for n epochs
        for epoch in range(self.epochs):

            time_start_epoch = time.time()

            # Reset the environment at beginning of each epoch
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, device=self.device).to(torch.float)

            # Rollout
            if self.space == "cont":
                avg_reward = self.rollout_cont(obs)
            elif self.space == "disc":
                self.rollout_disc(obs)

            # Update parameters FOR ON POLICY
            policy_loss, critic_loss = self.update()

            # if self.rl_alg.on_off_policy == "on":
                # reward_to_log = round(float(self.traj_data.rewards.mean()),5)

            reward_to_log = round(avg_reward,5)
            loss_to_log_policy = -round(policy_loss.item(),5)
            loss_to_log_critic = -round(critic_loss.item(),5)

            # Update tensorboard and terminal
            self.writer.add_scalars(
                "reward", {self.rl_alg.name: reward_to_log}, epoch)
            self.writer.add_scalars(
                "loss/policy", {self.rl_alg.name: loss_to_log_policy}, epoch)
            self.writer.add_scalars(
                "loss/critic", {self.rl_alg.name: loss_to_log_critic}, epoch)
            self.writer.flush()

            epoch_runtime = time.time()-time_start_epoch
            total_runtime = time.time()-time_start_train
            epoch_runtime_avg = total_runtime/(epoch+1)
            print(f"Completed epoch {epoch + 1}: Total runtime {np.round(total_runtime/60,3)}/{np.round(self.epochs*epoch_runtime_avg/60,3)} min, Epoch runtime {np.round(epoch_runtime,3)} sec, Reward: {reward_to_log}, Policy Loss: {loss_to_log_policy}")

            # Save the model iteratively, naming based on final reward
            if ((epoch + 1) % self.save_every == 0) and epoch != 0:
                model_dir = os.path.join(
                    root_dir, "models", f"{self.gym_model}_{self.rl_alg.name}_{reward_to_log}.pth")
                os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)
                torch.save(self.rl_alg.state_dict(), model_dir)
                print('Policy saved at', model_dir)

    def update(self):

        # Updates for on policy
        if self.rl_alg.on_off_policy == "on":
            # Get total expected return from the rollout in this epoch
            self.traj_data.calc_returns()

            # Update networks
            update_epochs = 10 if (self.rl_alg.name == "PPO_CONT") else 1

            for _ in range(update_epochs):
                policy_loss, critic_loss = self.rl_alg.loss_func(self.traj_data)
                self.rl_alg.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.rl_alg.policy_optimizer.step()

            self.traj_data.detach()

        # Updates for off policy
        elif self.rl_alg.on_off_policy == "off":

            # Critic update
            for _ in range(64):
                critic_loss, _ = self.rl_alg.loss_func(*self.buffer.sample())
                self.rl_alg.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.rl_alg.critic_optimizer.step()

            # Policy update
            for _ in range(4):
                _, policy_loss = self.rl_alg.loss_func(*self.buffer.sample())
                self.rl_alg.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.rl_alg.policy_optimizer.step()

        # Target updates
        if self.rl_alg.target_updates:
            for target_param, param in zip(self.rl_alg.critic_1_target.parameters(), self.rl_alg.critic_1.parameters()):
                target_param.data.copy_(
                    self.rl_alg.tau * param.data + (1 - self.rl_alg.tau) * target_param.data)

            for target_param, param in zip(self.rl_alg.critic_2_target.parameters(), self.rl_alg.critic_2.parameters()):
                target_param.data.copy_(
                    self.rl_alg.tau * param.data + (1 - self.rl_alg.tau) * target_param.data)

            for target_param, param in zip(self.rl_alg.policy_target.parameters(), self.rl_alg.policy.parameters()):
                target_param.data.copy_(
                    self.rl_alg.tau * param.data + (1 - self.rl_alg.tau) * target_param.data)

        return policy_loss, critic_loss

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
        # Rollout for t timesteps
        for t in range(self.t_steps):

            actions, log_probs, dist = GetAction(
                self.rl_alg, obs, target=False, grad=False)

            # Step 3: take the action in the environment, using the action as a control command to the robot model.
            obs_new, reward, done, truncated, infos = self.env.step(
                actions.cpu().numpy())
            done = done | truncated  # Change done if the episode is truncated

            # Step 4: store data in traj_data
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

        avg_reward = total_reward.mean().item() / self.t_steps
        return avg_reward

    def plot_reward(self, epoch):
        if self.rl_alg.name == "PPO_ADV":
            color = "red"
        else:
            color = "blue"

        self.epoch_vec.append(epoch)
        self.reward_vec.append(self.traj_data.rewards.mean())
        plt.figure('Reward Plot')
        plt.plot(self.epoch_vec, self.reward_vec, color)
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.pause(.000001)

############################# ADVERSARIAL #####################################
    def train_adv(self, adversary, player):

        time_start_train = time.time()
        self.total_reward = 0  # Track total reward for this training run

        # Running for n epochs
        for epoch in range(self.epochs):

            time_start_epoch = time.time()

            # Reset the environment at beginning of each epoch
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, device=self.device).to(torch.float)

            # Rollout
            if self.space == "cont":
                self.rollout_cont_adv(obs, adversary, player)
            elif self.space == "disc":  # Not implemented for discrete
                self.rollout_disc(obs)

            # Update parameters
            self.update()

            # Store reward for this epoch: for adervarial plotting
            self.total_reward += self.traj_data.rewards.mean()

            print(f"Completed epoch {epoch + 1}: Total runtime {np.round((time.time()-time_start_train)/60,5)} min, Epoch runtime {np.round(time.time()-time_start_epoch,5)} sec, Reward: {np.round(self.buffer.rewards.mean(),5)}")

    def rollout_cont_adv(self, obs, adversary, player):
        # Rollout for t timesteps
        for t in range(self.t_steps):

            # DEFENDER Step 1: forward pass on the actor and critic to get action and value
            with torch.no_grad() if self.rl_alg.need_grad == False else torch.enable_grad():
                mean = self.rl_alg.policy(obs.reshape(
                    self.num_environments, self.n_obs))
                std = torch.exp(self.rl_alg.log_std)

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            dist = Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)

            # ADVERSARY Step 1: forward pass on the actor and critic to get action and value
            with torch.no_grad() if adversary.need_grad == False else torch.enable_grad():
                mean_adv = adversary.policy(obs.reshape(
                    self.num_environments, self.n_obs))
                std_adv = torch.exp(adversary.log_std)

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            dist_adv = Normal(mean_adv, std_adv)
            actions_adv = dist_adv.sample()
            log_probs_adv = dist_adv.log_prob(actions_adv).sum(dim=-1)

            # Step 3: take the action in the environment, using the action as a control command to the robot model.
            obs_new, reward, done, truncated, infos = self.env.step(
                actions.numpy())
            obs_new, reward, done, truncated, infos = self.env.step(
                actions_adv.numpy())
            done = done | truncated  # Change done if the episode is truncated

            # Make adversary have opposite reward
            if player == "adversary":
                reward = -reward

            # Step 4: store data in traj_data
            self.traj_data.store(t, obs, actions, reward, log_probs, done)
            obs = torch.tensor(obs_new, device=self.device).to(torch.float)
