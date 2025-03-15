import torch.nn as nn
import torch
import copy

class DDPG(nn.Module):
    def __init__(self, input_dim, output_dim, lr):
        super(DDPG, self).__init__()

        self.name = "DDPG"
        self.type = "deterministic"
        self.on_off_policy = "off"
        self.target_updates = True
        self.tau = 0.1  # update rate for target policies
        self.need_grad = False
        self.noisy = True
        self.exploration_rate = 1
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.critic = nn.Sequential(
            nn.Linear(input_dim+output_dim,64),
            nn.ReLU(),
            nn.Linear(64,1)
        ).to(self.device)

        self.policy = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.policy_target = copy.deepcopy(self.policy).to(self.device)

        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(),lr=lr)
        self.policy_optimizer = torch.optim.Adam(params=self.policy.parameters(),lr=lr)
        
        self.criterion = nn.MSELoss()

    def loss_func(self, states, actions, rewards, next_states, not_dones, gamma=.99):

        with torch.no_grad():

            # 1) Get next actions from target policy
            next_actions = self.policy_target(next_states)

            # 2) Get target value from next states and next actions
            next_state_action_vec = torch.cat(
                (next_states, next_actions), dim=-1)
            q_next = self.critic_target(next_state_action_vec)

        # 3) Get value from current states and current actions
        state_action_vec = torch.cat((states, actions), dim=-1)
        q = self.critic(state_action_vec)

        # 4) Get target q
        q_target = rewards.unsqueeze(-1) + gamma*q_next*not_dones.unsqueeze(-1)

        # 5) Get q loss
        critic_loss = self.criterion(q, q_target)

        # Policy loss
        actions = self.policy(states)
        state_action_vec = torch.cat((states, actions), dim=-1)
        policy_loss = -torch.mean(self.critic(state_action_vec))

        return critic_loss, policy_loss
