import torch
import torch.nn as nn
from torch.distributions import categorical

class VPG(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(VPG, self).__init__()
        
        self.name = "REINFORCE"

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # Parameters for a continuous model
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
    
    def loss_func(self, buffer):

        _, value = self.forward(buffer.states)

        # Get advantage
        adv = buffer.returns - value.squeeze(-1)

        # Losses for each network
        loss_value = torch.mean(adv**2)
        loss_policy = -torch.mean(buffer.log_probs*adv)

        loss = loss_value + loss_policy

        return loss

    def train(self, t, env, obs, buffer):

        # Step 1: forward pass on the actor and critic to get action and value
        with torch.enable_grad():
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