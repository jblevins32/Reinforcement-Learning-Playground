import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PPO(nn.Module):
    '''
    The goal of PPO is to imprrove training stability of a polciy by limiting the changes that can be made to a policy.
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

        # Learns the std deviation
        self.log_std = nn.Parameter(torch.zeros(output_dim))

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Parameters for a continuous model
        mean = self.actor(x)
        std = torch.exp(self.log_std) # Convert the log_std to std
        value = self.critic(x)
        return mean, std, value