import torch.nn as nn
import torch
import copy

class TD3(nn.Module):
    def __init__(self, input_dim, output_dim, lr):
        super(TD3, self).__init__()

        self.name = "TD3"
        self.type = "deterministic"
        self.on_off_policy = "off"
        self.target_updates = True
        self.tau = 0.1  # update rate for target policies
        self.gamma=0.99
        self.need_grad = False
        self.need_noisy = True
        self.critic_update_delay = 50
        self.policy_update_delay = 1
        self.explore = True
        self.exploration_rate = 1
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.critic_1 = nn.Sequential(
            nn.Linear(input_dim+output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.critic_2 = nn.Sequential(
            nn.Linear(input_dim+output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.policy = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        ).to(self.device)

        self.critic_1_target = copy.deepcopy(self.critic_1).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2).to(self.device)
        self.policy_target = copy.deepcopy(self.policy).to(self.device)

        self.critic_optimizer = torch.optim.Adam(params=list(self.critic_1.parameters()) + list(self.critic_2.parameters()),lr=lr)
        self.policy_optimizer = torch.optim.Adam(params=self.policy.parameters(),lr=lr)
        
        self.criterion = nn.MSELoss()

    def loss_func_critic(self, states, actions, rewards, next_states, not_dones, GetAction):

        # Get the target critic loss based on the next state and action for the bellman eqn
        with torch.no_grad():

            # 1) Get next actions from target policy
            next_actions,_,_ = GetAction(next_states, target=True,grad=False)

            # Unique here to TD3 next_actions smoothing so Q does not overestimate
            policy_noise = 0.2
            noise_clip = 0.5
            noise = (torch.randn_like(next_actions) * policy_noise).clamp(-noise_clip, noise_clip)
            next_actions = (next_actions + noise)

            # 2) Get target value from next states and next actions
            next_state_action_vec = torch.cat(
                (next_states, next_actions), dim=-1)
            q1 = self.critic_1_target(next_state_action_vec)
            q2 = self.critic_2_target(next_state_action_vec)
            
        q_next = torch.min(q1,q2)

        # 3) Get value from current states and current actions
        state_action_vec = torch.cat((states, actions), dim=-1)
        q1 = self.critic_1(state_action_vec)
        q2 = self.critic_2(state_action_vec)

        # 4) Formulate target q aka the bellman eqn
        q_target = rewards.unsqueeze(-1) + self.gamma*q_next*not_dones.unsqueeze(-1)

        # 5) Get critic loss
        loss_critic = self.criterion(q1,q_target) + self.criterion(q2,q_target)

        return loss_critic

    def loss_func_policy(self, states, actions, rewards, next_states, not_dones, GetAction):
        # 6) Get policy loss
        actions,_,_ = GetAction(states, target=False,grad=True)
        state_action_vec = torch.cat((states, actions), dim=-1)
        loss_policy = -torch.mean(self.critic_1(state_action_vec))

        return loss_policy
