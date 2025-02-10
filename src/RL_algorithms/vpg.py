import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import categorical

class VPG(nn.Module):
    '''
    The goal of PPO is to improve training stability of a policy by limiting the changes that can be made to a policy.
        - smaller updates are more likely to converge to an optimal solutions
        - large jumps can fall off of a cliff
    '''
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        
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
    
    def loss_func(self, t, obs_new, reward, discount, value, log_probs, epsilon, buffer):

        # Step 4: calculate the advantage
        _, value_new = self.forward(torch.Tensor(obs_new))
        Adv = reward + discount * value_new - value

        # Step 5: get importance sampling ratio. Check if there is anything in buffer, if not, use current log prob as old log prob
        if np.all(np.array(buffer.states[0])) == 0:
            log_probs_old = log_probs
        else:
            log_probs_old = buffer.log_probs[t]
        r = torch.exp(log_probs - log_probs_old)

        # Step 6: calculate the surrogate loss
        loss = -torch.min(r*Adv, torch.clamp(r,1-epsilon,1+epsilon)*Adv)

        # Step 7: calcualte the critic loss
        critic_loss = F.mse_loss(value, reward + discount * value_new.detach())

        # Step 8: Total loss
        return loss + critic_loss

    def train(self, t, env, obs, discount, epsilon, buffer):

        # Step 1: forward pass on the actor and critic to get action and value
        with torch.enable_grad():
            logits, value = self.forward(obs)

        # Step 2: create a distribution from the logits (raw outputs) and sample from it
        # probs = torch.distributions.Normal(mean, std)
        # probs = torch.distributions.MultivariateNormal(mean, covariance_matrix=)
        probs = categorical.Categorical(logits=logits)
        action = probs.sample()
        log_probs = probs.log_prob(action)

        # Step 3: take the action in the environment, using the action as a control command to the robot model. 
        obs_new, reward, done, truncated, infos = env.step(action.numpy())
        
        # old stuff from my env that needs to be updated
        # reward = env.reward(state_new,state,action)
        # done = env.reached_goal(state_new) # Deternime if the goal is reached

        # Step 9 store data in buffer
        buffer.store(t, obs, action, reward, log_probs, done)

        return env, torch.Tensor(obs_new), buffer