import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import categorical

class PPO(nn.Module):
    '''
    The goal of PPO is to improve training stability of a policy by limiting the changes that can be made to a policy.
        - smaller updates are more likely to converge to an optimal solutions
        - large jumps can fall off of a cliff
    '''
    def __init__(self, input_dim, output_dim, epsilon):
        super(PPO, self).__init__()
        
        self.epsilon = epsilon
        self.name = "PPO"
        
        # Learns the mean
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        # Learns the value
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Parameters for a continuous model
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
    
    def loss_func(self, buffer):

        logits, value = self.forward(buffer.states)

        adv = buffer.returns - value.squeeze(-1)
        loss_value = torch.mean(adv**2)

        # importance sampling. logs turned this into a subtraction
        # First, get probability distribution of the updated policy. Second, get the log probabilities of the same actions taken in the old policy.
        probs = categorical.Categorical(logits=logits)
        new_log_probs = probs.log_prob(buffer.actions)

        r = torch.exp(new_log_probs - buffer.log_probs)

        loss_policy = -torch.mean(torch.min(r*adv, torch.clamp(r,1-self.epsilon, 1+self.epsilon)*adv))

        loss = loss_value + loss_policy

        return loss

    def train(self, t, env, obs, buffer):

        # Step 1: forward pass on the actor and critic to get action and value
        with torch.no_grad():
            logits, _ = self.forward(obs)

        # Step 2: create a distribution from the logits (raw outputs) and sample from it
        probs = categorical.Categorical(logits=logits)
        actions = probs.sample()
        log_probs = probs.log_prob(actions)

        # Step 3: take the action in the environment, using the action as a control command to the robot model. 
        obs_new, reward, done, truncated, infos = env.step(actions.numpy())
        done = done | truncated # Change done if the episode is truncated

        # Step 4: store data in buffer
        buffer.store(t, obs, actions, reward, log_probs, done)

        return env, torch.Tensor(obs_new), buffer