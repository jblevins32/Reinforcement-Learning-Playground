from torch.optim import Adam
import matplotlib.pyplot as plt
from buffer import *
from globals import root_dir
import os
import time
from torch.distributions import Normal
from torch.distributions import Categorical
from RL_algorithms.reinforce import *
from RL_algorithms.vpg import *
from RL_algorithms.ppo import *
from RL_algorithms.ppo_adv import *
from RL_algorithms.ppo_cont import *

############################################################################################################

# Load and run the agent
class Agent():
    def __init__(self, rl_alg,num_environments,epochs,t_steps,env,n_obs,n_actions,discount, epsilon, lr, save_every, gym_model, num_agents, space, writer):

        # Initialize plot variables
        self.epoch_vec = []
        self.reward_vec = []
        self.frames = []

        self.epochs = epochs
        self.env = env
        self.discount = discount
        self.t_steps = t_steps
        self.save_every = save_every
        self.gym_model = gym_model
        self.num_environments = num_environments
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.num_agents = num_agents
        self.space = space
        self.writer = writer

        # Choose RL algorithm
        if rl_alg == "PPO":
            self.rl_alg = PPO(input_dim=n_obs, output_dim=n_actions, epsilon=epsilon)
        elif rl_alg == "REINFORCE":
            self.rl_alg = REINFORCE(input_dim=n_obs, output_dim=n_actions)
        elif rl_alg == "VPG":
            self.rl_alg = VPG(input_dim=n_obs, output_dim=n_actions)
        elif rl_alg =="PPO_ADV":
            self.rl_alg = PPO_ADV(input_dim=n_obs, output_dim=n_actions, epsilon=epsilon)
        elif rl_alg =="PPO_CONT":
            self.rl_alg = PPO_CONT(input_dim=n_obs, output_dim=n_actions, epsilon=epsilon)

        # Choose optimizer
        self.optimizer = Adam(params=self.rl_alg.parameters(), lr=lr)

        # Create buffer
        self.buffer = Buffer(n_steps=self.t_steps, n_envs=num_environments, n_obs=n_obs, n_actions=n_actions, space=space)
    
    def train(self):
        
        time_start_train = time.time()

        # Running for n epochs
        for epoch in range(self.epochs):
            
            time_start_epoch = time.time()

            # Reset the environment at beginning of each epoch
            obs, _ = self.env.reset()
            obs = torch.Tensor(obs)

            # Rollout 
            if self.space == "cont":
                self.rollout_cont(obs)
            elif self.space == "disc":
                self.rollout_disc(obs)

            # Update parameters
            self.update()

            # Update tensorboard and terminal
            self.writer.add_scalars("", {self.rl_alg.name: self.buffer.rewards.mean()}, epoch)
            self.writer.flush()

            print(f"Completed epoch {epoch + 1}: Total runtime {np.round((time.time()-time_start_train)/60,5)} min, Epoch runtime {np.round(time.time()-time_start_epoch,5)} sec, Reward: {np.round(self.buffer.rewards.mean(),5)}")
            
            # Save the model iteratively
            if ((epoch + 1) % self.save_every == 0) and epoch != 0:
                final_reward = round(float(self.buffer.rewards.mean()),5)
                model_dir = os.path.join(root_dir,"models",f"{self.gym_model}_{self.rl_alg.name}_{final_reward}.pth")
                os.makedirs(os.path.join(root_dir,"models"), exist_ok=True)
                torch.save(self.rl_alg.state_dict(),model_dir)

    def update(self):
        # Get total expected return from the rollout in this epoch
        self.buffer.calc_returns()

        # Update networks
        update_epochs = 10 if ((self.rl_alg.name == "PPO") | (self.rl_alg.name == "PPO_ADV")) else 1

        for _ in range(update_epochs):
            loss = self.rl_alg.loss_func(self.buffer)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   

        self.buffer.detach()

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
            obs_new, reward, done, truncated, infos = self.env.step(actions.numpy())
            done = done | truncated # Change done if the episode is truncated

            # Step 4: store data in buffer
            self.buffer.store(t, obs, actions, reward, log_probs, done)
            obs = torch.Tensor(obs_new)

    def rollout_cont(self, obs):
        # Rollout for t timesteps
        for t in range(self.t_steps):

            # Step 1: forward pass on the actor and critic to get action and value
            with torch.no_grad() if self.rl_alg.need_grad == False else torch.enable_grad():
                mean = self.rl_alg.policy(obs.reshape(self.num_environments,self.n_obs))
                std = torch.exp(self.rl_alg.log_std)

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            dist = Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)

            # Step 3: take the action in the environment, using the action as a control command to the robot model. 
            obs_new, reward, done, truncated, infos = self.env.step(actions.numpy())
            done = done | truncated # Change done if the episode is truncated

            # Step 4: store data in buffer
            self.buffer.store(t, obs, actions, reward, log_probs, done)
            obs = torch.Tensor(obs_new)

    def plot_reward(self, epoch):
        if self.rl_alg.name == "PPO_ADV":
            color = "red"
        else:
            color = "blue"

        self.epoch_vec.append(epoch)
        self.reward_vec.append(self.buffer.rewards.mean())
        plt.figure('Reward Plot')
        plt.plot(self.epoch_vec, self.reward_vec, color)
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.pause(.000001)

############################# ADVERSARIAL #####################################
    def train_adv(self, adversary, player):
        
        time_start_train = time.time()
        self.total_reward = 0 # Track total reward for this training run

        # Running for n epochs
        for epoch in range(self.epochs):
            
            time_start_epoch = time.time()

            # Reset the environment at beginning of each epoch
            obs, _ = self.env.reset()
            obs = torch.Tensor(obs)

            # Rollout 
            if self.space == "cont":
                self.rollout_cont_adv(obs, adversary, player)
            elif self.space == "disc": # Not implemented for discrete
                self.rollout_disc(obs)

            # Update parameters
            self.update()

            # Store reward for this epoch: for adervarial plotting
            self.total_reward += self.buffer.rewards.mean()

            print(f"Completed epoch {epoch + 1}: Total runtime {np.round((time.time()-time_start_train)/60,5)} min, Epoch runtime {np.round(time.time()-time_start_epoch,5)} sec, Reward: {np.round(self.buffer.rewards.mean(),5)}")

    def rollout_cont_adv(self, obs, adversary, player):
        # Rollout for t timesteps
        for t in range(self.t_steps):
            
            # DEFENDER Step 1: forward pass on the actor and critic to get action and value
            with torch.no_grad() if self.rl_alg.need_grad == False else torch.enable_grad():
                mean = self.rl_alg.policy(obs.reshape(self.num_environments,self.n_obs))
                std = torch.exp(self.rl_alg.log_std)

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            dist = Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)

            # ADVERSARY Step 1: forward pass on the actor and critic to get action and value
            with torch.no_grad() if adversary.need_grad == False else torch.enable_grad():
                mean_adv = adversary.policy(obs.reshape(self.num_environments,self.n_obs))
                std_adv = torch.exp(adversary.log_std)

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            dist_adv = Normal(mean_adv, std_adv)
            actions_adv = dist_adv.sample()
            log_probs_adv = dist_adv.log_prob(actions_adv).sum(dim=-1)

            # Step 3: take the action in the environment, using the action as a control command to the robot model. 
            obs_new, reward, done, truncated, infos = self.env.step(actions.numpy())
            obs_new, reward, done, truncated, infos = self.env.step(actions_adv.numpy())
            done = done | truncated # Change done if the episode is truncated

            # Make adversary have opposite reward
            if player == "adversary":
                reward = -reward

            # Step 4: store data in buffer
            self.buffer.store(t, obs, actions, reward, log_probs, done)
            obs = torch.Tensor(obs_new)