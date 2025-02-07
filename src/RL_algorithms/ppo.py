import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from my_simulation.robot_model import RobotModel

class PPO(nn.Module):
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

        # Learns the std deviation
        self.log_std = nn.Parameter(torch.zeros(output_dim))

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
        mean = self.actor(x)
        std = torch.exp(self.log_std) # Convert the log_std to std
        value = self.critic(x)
        return mean, std, value
    
    def train(self, env, discount, epsilon, buffer):
        # Step 0: get current state of environment
        state = torch.tensor(env.position, dtype=torch.float32).unsqueeze(0)

        # Step 1: forward pass on the actor and critic to get action and value
        mean, std, value = self.forward(state)

        # Step 2: create a distribution from the mean and std
        gaussian = torch.distributions.Normal(mean, std)
        action = gaussian.sample()

        # Step 3: take the action in the environment, using the action as a control command to the robot model. 
        state_new = RobotModel(state,action)
        reward = env.reward(state_new,state,action)
        done = env.reached_goal(state_new) # Deternime if the goal is reached

        # Step 4: calculate the advantage
        _,_, value_new = self.forward(torch.tensor(state_new, dtype=torch.float32))
        Adv = torch.tensor(reward) + discount * value_new - value

        # Step 5: caluclate probability ratio of taking an action under the current policy vs the old policy (estimate divergence in the two policies)
        log_prob = gaussian.log_prob(action).sum(dim=-1)

        # Check if there is anything in buffer, if not, use current log prob as old log prob
        if np.all(buffer.states[0]) == 0:
            log_prob_old = log_prob
        else:
            log_prob_old = buffer.log_probs[buffer.ptr]
        r = torch.exp(log_prob - log_prob_old)

        # Step 6: calculate the surrogate loss
        loss = -torch.min(r*Adv, torch.clamp(r,1-epsilon,1+epsilon)*Adv)

        # Step 7: calcualte the critic loss
        critic_loss = F.mse_loss(value, reward + discount * value_new.detach())

        # Step 8: Total loss
        total_loss = loss + critic_loss

        # Step 9: Update buffer with this experience
        buffer.store(state,action,reward,state_new,done,log_prob,value)

        return total_loss, state_new, reward