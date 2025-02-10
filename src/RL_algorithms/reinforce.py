import torch
import torch.nn as nn
from torch.distributions import categorical

class REINFORCE(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(REINFORCE, self).__init__()
        
        self.name = "REINFORCE"

        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):

        logits = self.policy(x)

        return logits
    
    def loss_func(self, buffer):

        loss = -torch.sum(buffer.log_probs * buffer.returns)/(buffer.n_steps*buffer.returns.shape[1])

        return loss

    def train(self, t, env, obs, buffer):

        # Step 1: forward pass on the actor and critic to get action and value
        with torch.enable_grad():
            logits = self.forward(obs)

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