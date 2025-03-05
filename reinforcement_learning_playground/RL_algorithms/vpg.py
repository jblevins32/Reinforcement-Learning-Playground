import torch
import torch.nn as nn
from torch.distributions import categorical


class VPG(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(VPG, self).__init__()

        self.need_grad = True
        self.name = "VPG"
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)

    def loss_func(self, buffer):

        value = self.critic(buffer.states)

        # Get advantage
        adv = buffer.returns - value.squeeze(-1)

        # Losses for each network
        loss_value = torch.mean(adv**2)
        loss_policy = -torch.mean(buffer.log_probs*adv)

        loss = loss_value + loss_policy

        return loss
