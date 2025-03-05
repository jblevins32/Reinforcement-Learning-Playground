import torch.nn as nn
import torch
import copy
from torch.optim import Adam
from get_action import GetAction
import torch.distributions.normal as Normal


class SAC(nn.Module):
    '''
    SAC: Soft Actor Critic

    Training Process. For each epoch:
        1) Trajectory rollout:
            1) Get mean and std from policy, one for each action (stochastic policy)
            2) Sample actions from distributions created by the generated means and stds (stochastic policy)
            3) Send action to env and get obs and reward
            4) Save the transition in a buffer
        2) For some amount of updates
            1) Sample from the buffer
            2) Get critic loss
                1) Get next actions from target policy network using the next actions 
                2) Get value of next state, next action pair from q1 target network
                3) Get value of next state, next action pair from q2 target network
                4) Get value of current state, current action pair from q1 network
                5) Get value of current state, current action pair from q2 network
                6) Get mean and std from policy, one for each one for each action from current states
                7) Get entropy of each mean and std
                8) Take the minimum values from 2) and 3) (double q learning idea)
                9) Get target value
                10) Sum MSE loss of (value of current state from q1 and target value) and (value of current state from q2 and target value)  (double q learning idea)
            3) Update both critic networks
        3) For some amount of updates
            1) Get policy loss
                1) Sample from the buffer
                2) Get - mean of the value of the next state, next action pair from q1 network, incorporationg entropy
            2) Update policy network
        4) Update target networks


    '''

    def __init__(self, input_dim, output_dim, lr):
        super(SAC, self).__init__()

        self.name = "SAC"
        self.type = "stochastic"
        self.on_off_policy = "off"
        self.target_updates = True
        self.need_grad = False
        self.tau = 0.1  # update rate for target policies
        self.alpha = 0.002
        self.gamma = 0.99
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim*2)
        ).to(self.device)

        self.critic_1 = nn.Sequential(
            nn.Linear(input_dim+output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.critic_2 = nn.Sequential(
            nn.Linear(input_dim+output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.policy_target = copy.deepcopy(self.policy)

        self.critic_optimizer = Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

        self.criterion = nn.MSELoss()

    def loss_func(self, states, actions, rewards, next_states, not_dones):

        with torch.no_grad():

            # 1) Get next actions from target policy
            next_actions, _, _ = GetAction(
                self, next_states, target=True, grad=False)

            # next_actions = self.get_target_action(next_states, noisy=True)

            # 2) Get target value from next states and next actions
            next_state_action_vec = torch.cat(
                (next_states, next_actions), dim=-1)
            q1_next = self.critic_1_target(next_state_action_vec)
            q2_next = self.critic_2_target(next_state_action_vec)

        # 3) Get value from current states and current actions
        state_action_vec = torch.cat((states, actions), dim=-1)
        q1 = self.critic_1(state_action_vec)
        q2 = self.critic_2(state_action_vec)

        # 4) Get target q, starting by getting  entropy H
        _, _, dist = GetAction(self, next_states, target=False, grad=False)
        H = dist.entropy()

        q_next = torch.min(q1_next, q2_next)
        q_target = rewards.unsqueeze(-1) + (self.gamma *
                                            q_next + self.alpha*H)*not_dones.unsqueeze(-1)

        # 5) Get q loss
        criterion = nn.MSELoss()
        critic_loss = criterion(q1, q_target) + criterion(q2, q_target)

        # Get policy loss
        actions, _, dist = GetAction(self, states, target=False, grad=True)
        H = dist.entropy()

        state_action_vec = torch.cat((states, actions), dim=-1)

        policy_loss = - \
            torch.mean(self.critic_1(state_action_vec) + self.alpha*H)

        return critic_loss, policy_loss
