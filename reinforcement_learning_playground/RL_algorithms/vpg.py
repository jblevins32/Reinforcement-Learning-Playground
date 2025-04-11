import torch
import torch.nn as nn
from torch.optim import Adam

class VPG(nn.Module):

    def __init__(self, input_dim, output_dim, lr):
        super(VPG, self).__init__()

        self.name = "VPG"
        self.type = "stochastic"
        self.on_off_policy = "on"
        self.target_updates = False
        self.need_grad = True
        self.need_noisy = False
        self.policy_update_delay = 1 # This is no delay, update every episode
        self.explore = False
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Learns the mean
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim*2)
        ).to(self.device)

        # Learns the value
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.policy_optimizer = Adam(self.parameters(), lr=lr)

    def loss_func(self, traj_data, GetAction):

        value = self.critic(traj_data.states)

        # Get advantage
        adv = traj_data.returns - value.squeeze(-1)

        # Losses for each network
        loss_value = torch.mean(adv**2)
        loss_policy = -torch.mean(traj_data.log_probs*adv)

        loss = loss_value + loss_policy

        return loss_policy, loss_value
