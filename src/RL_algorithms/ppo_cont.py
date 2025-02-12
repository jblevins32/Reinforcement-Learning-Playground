import torch
import torch.nn as nn
from torch.distributions import Normal

class PPO_CONT(nn.Module):
    '''
    The goal of PPO is to improve training stability of a policy by limiting the changes that can be made to a policy.
        - smaller updates are more likely to converge to an optimal solutions
        - large jumps can fall off of a cliff
    '''
    def __init__(self, input_dim, output_dim, epsilon):
        super(PPO_CONT, self).__init__()
        
        self.epsilon = epsilon
        self.name = "PPO_CONT"

        # Learns the mean
        self.policy = nn.Sequential(
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

        self.log_std = nn.Parameter(torch.zeros(output_dim))
    
    def loss_func(self, buffer):

        mean = self.policy(buffer.states)
        value = self.critic(buffer.states)
        std = torch.exp(self.log_std)

        # Create the distribution
        dist = Normal(mean, std)
        
        adv = buffer.returns - value.squeeze(-1)
        loss_value = torch.mean(adv**2)

        new_log_probs = dist.log_prob(buffer.actions).sum(dim=-1)

        r = torch.exp(new_log_probs - buffer.log_probs)

        loss_policy = -torch.mean(torch.min(r*adv, torch.clamp(r,1-self.epsilon, 1+self.epsilon)*adv))

        loss = loss_value + loss_policy

        return loss