import torch.nn as nn
import torch
import copy
import torch.optim.adam as Adam


class DDPG(nn.Module):
    def __init__(self, n_obs, n_actions, lr):
        super(DDPG, self).__init__()

        self.name = "DDPG"
        self.type = "deterministic"
        self.on_off_policy = "off"
        self.target_updates = True
        self.tau = 0.1  # update rate for target policies
        self.need_grad = False
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.policy = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU,
            nn.linear(64, n_actions)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(n_obs+n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)
        self.policy_target = copy.deepcopy(self.policy)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

        self.criterion = nn.MSELoss()

    def loss_func(self, states, actions, rewards, next_states, not_dones, gamma):

        with torch.no_grad():

            # 1) Get next actions from target policy
            next_actions = self.policy_target_net(next_states)

            # 2) Get target value from next states and next actions
            next_state_action_vec = torch.cat(
                (next_states, next_actions), dim=-1)
            q_next = self.q_target_net(next_state_action_vec)

        # 3) Get value from current states and current actions
        state_action_vec = torch.cat((states, actions), dim=-1)
        q = self.q_net(state_action_vec)

        # 4) Get target q
        q_target = rewards.unsqueeze(-1) + gamma*q_next*not_dones.unsqueeze(-1)

        # 5) Get q loss
        critic_loss = self.criterion(q, q_target)

        # Policy loss
        actions = self.policy(states)
        state_action_vec = torch.cat((states, actions), dim=-1)
        policy_loss = -torch.mean(self.q_net(state_action_vec))

        return critic_loss, policy_loss
