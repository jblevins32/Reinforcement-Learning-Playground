# Mujoco imports
import mujoco
from mujoco import viewer
import gymnasium as gym
from test import *
from torch.optim import Adam
import matplotlib.pyplot as plt
from buffer import *
from torch.utils.tensorboard import SummaryWriter
import webbrowser
from globals import root_dir
import os
import subprocess

from RL_algorithms.reinforce import *
from RL_algorithms.vpg import *
from RL_algorithms.ppo import *
from RL_algorithms.ppo_adv import *




############################################################################################################

# Load and run the agent
class Agent():
    def __init__(self, rl_alg,operation,num_environments,epochs,t_steps,env,n_obs,n_actions,discount, epsilon, lr, live_sim):

        # Initialize plot variables
        self.epoch_vec = []
        self.reward_vec = []
        self.frames = []

        self.operation = operation
        self.epochs = epochs
        self.env = env
        self.discount = discount
        self.live_sim = live_sim
        self.t_steps = t_steps

        # Choose RL algorithm
        if rl_alg == "PPO":
            self.rl_alg = PPO(input_dim=n_obs, output_dim=n_actions, epsilon=epsilon)
        elif rl_alg == "REINFORCE":
            self.rl_alg = REINFORCE(input_dim=n_obs, output_dim=n_actions)
        elif rl_alg == "VPG":
            self.rl_alg = VPG(input_dim=n_obs, output_dim=n_actions)
        elif rl_alg =="PPO_ADV":
            self.rl_alg = PPO_ADV(input_dim=n_obs, output_dim=n_actions, epsilon=epsilon)

        # Tensor board setup
        log_dir=os.path.join(root_dir,"tensorboard",self.rl_alg.name)

        # Start the tensorboard
        tensorboard_cmd = f"tensorboard --logdir={log_dir} --port=6006 --bind_all"
        subprocess.Popen(tensorboard_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Create the writer
        self.writer = SummaryWriter(log_dir=log_dir, comment=f"_{self.rl_alg.name}")
        webbrowser.open("http://localhost:6006")

        # Choose optimizer
        self.optimizer = Adam(params=self.rl_alg.parameters(), lr=lr)

        # Create buffer
        self.buffer = Buffer(n_steps=self.t_steps, n_envs=num_environments, n_obs=n_obs, n_actions=n_actions)
    
    def train(self):
        # Running for n epochs
        for epoch in range(self.epochs):
            print(f"Beginning Epoch {epoch+1}")
            
            # Reset the environment at beginning of each epoch
            obs, _ = self.env.reset()
            obs = torch.Tensor(obs)

            # Rollout 
            self.rollout(obs)

            # Update parameters
            self.update()

            # Update Tensorboard
            self.writer.add_scalar("Reward", self.buffer.rewards.mean(), epoch)
            self.writer.flush()
            
            # anim.move(env.cost_map, env.position)

            self.plot_reward(epoch)

            # if self.live_sim == True:
            #     plt.figure('Environment')
            #     plt.imshow(frames[-1])
            #     plt.axis('off')
            #     plt.pause(.000001)

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

    def rollout(self, obs):
        # Rollout for t timesteps
        for t in range(self.t_steps):

            # Step 1: forward pass on the actor and critic to get action and value
            with torch.no_grad() if self.rl_alg.name == 'PPO' else torch.enable_grad():
                logits = self.rl_alg.policy(obs)

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            probs = categorical.Categorical(logits=logits)
            actions = probs.sample()
            log_probs = probs.log_prob(actions)

            # Step 3: take the action in the environment, using the action as a control command to the robot model. 
            obs_new, reward, done, truncated, infos = self.env.step(actions.numpy())
            done = done | truncated # Change done if the episode is truncated

            # Step 4: store data in buffer
            self.buffer.store(t, obs, actions, reward, log_probs, done)
            obs = torch.Tensor(obs_new)


            # For visualization
            # frames.append(env.render()[0])

    # Train or test without the Mujoco simulation
    # else:
        # with viewer.launch_passive(model, data) as v:\
            
        #     # Running for t timesteps
        #     for _ in range(t_steps):

        #         # Train or Test the agent
        #         if operation == "1":
        #             Train(data)
        #         else:
        #             Test()

        #         # Step Mujoco
        #         mujoco.mj_step(model, data)
                #   v.step()

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
    def train_adv(self, adversary):
        # Running for n epochs
        for epoch in range(self.epochs):
            print(f"Beginning Epoch {epoch+1}")
            
            # Reset the environment at beginning of each epoch
            obs, _ = self.env.reset()
            obs = torch.Tensor(obs)

            # Rollout 
            self.rollout_adv(obs, adversary)

            # Update parameters
            self.update()

            # Update Tensorboard
            self.writer.add_scalar("Reward", self.buffer.rewards.mean(), epoch)
            self.writer.flush()

            # anim.move(env.cost_map, env.position)

            self.plot_reward(epoch)

            # if self.live_sim == True:
            #     plt.figure('Environment')
            #     plt.imshow(frames[-1])
            #     plt.axis('off')
            #     plt.pause(.000001)

    def rollout_adv(self, obs, adversary):
        # Rollout for t timesteps
        for t in range(self.t_steps):

            with torch.no_grad() if ((self.rl_alg.name == 'PPO') | (self.rl_alg.name == 'PPO_ADV')) else torch.enable_grad():
                logits = self.rl_alg.policy(obs)

            probs = categorical.Categorical(logits=logits)
            actions = probs.sample()
            log_probs = probs.log_prob(actions)

            with torch.no_grad() if ((adversary.name == 'PPO') | (adversary.name == 'PPO_ADV')) else torch.enable_grad():
                logits_adv = adversary.policy(obs)

            probs_adv = categorical.Categorical(logits=logits_adv)
            actions_adv = probs_adv.sample()

            obs_new, reward, done, truncated, infos = self.env.step(actions.numpy())
            obs_new, reward, done, truncated, infos = self.env.step(actions_adv.numpy())
            done = done | truncated # Change done if the episode is truncated

            self.buffer.store(t, obs, actions, reward, log_probs, done)

            obs = torch.Tensor(obs_new)

            # For visualization
            # frames.append(env.render()[0])