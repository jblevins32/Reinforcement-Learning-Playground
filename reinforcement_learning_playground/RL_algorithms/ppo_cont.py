import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
from get_action import GetAction


class PPO_CONT(nn.Module):
    '''
    The goal of PPO is to improve training stability of a policy by limiting the changes that can be made to a policy.
        - smaller updates are more likely to converge to an optimal solutions
        - large jumps can fall off of a cliff
    '''
<<<<<<< HEAD:src/RL_algorithms/ppo_cont.py
    def __init__(self, input_dim, output_dim, lr):
=======

    def __init__(self, input_dim, output_dim, epsilon, lr):
>>>>>>> improve:reinforcement_learning_playground/RL_algorithms/ppo_cont.py
        super(PPO_CONT, self).__init__()

        self.name = "PPO_CONT"
        self.type = "stochastic"
        self.on_off_policy = "on"
        self.target_updates = False
        self.need_grad = False
        self.epsilon = 0.2
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Learns the mean
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ).to(self.device)

        # Learns the value
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.log_std = nn.Parameter(torch.zeros(output_dim)).to(self.device)

        self.policy_optimizer = Adam(self.parameters(), lr=lr)

    def loss_func(self, traj_data):

        _, _, dist = GetAction(self, traj_data.states, target=False, grad=True)

        value = self.critic(traj_data.states)

        adv = traj_data.returns - value.squeeze(-1)
        loss_value = torch.mean(adv**2)

        new_log_probs = dist.log_prob(traj_data.actions).sum(dim=-1)

        r = torch.exp(new_log_probs - traj_data.log_probs)

        loss_policy = - \
            torch.mean(
                torch.min(r*adv, torch.clamp(r, 1-self.epsilon, 1+self.epsilon)*adv))

        loss = loss_value + loss_policy

        return loss
