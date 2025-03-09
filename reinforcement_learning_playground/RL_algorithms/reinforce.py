import torch
import torch.nn as nn
from torch.distributions import categorical


class REINFORCE(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(REINFORCE, self).__init__()

        self.name = "REINFORCE"
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        ).to(self.device)

    def loss_func(self, buffer):

        loss = -torch.sum(buffer.log_probs * buffer.returns) / (buffer.n_steps*buffer.returns.shape[1])

        return loss
