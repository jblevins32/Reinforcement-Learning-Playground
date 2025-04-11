import torch

def GetAction(self, obs, target, grad):
    # obs = obs.reshape(self.num_environments,self.n_obs)

    with torch.no_grad() if grad == False else torch.enable_grad():

        # Actions are based on deterministic vs stochastic policy

        if self.rl_alg.type == "stochastic":
            # Step 1: forward pass on the actor and critic to get action and value
            if self.rl_alg.name == "SAC" or self.rl_alg.name == "VPG":
                if target:
                    mean, log_std = self.rl_alg.policy_target(obs).chunk(2, dim=-1)
                else:
                    mean, log_std = self.rl_alg.policy(obs).chunk(2, dim=-1)
                std = torch.exp(log_std)  # Use clamp?
            elif self.rl_alg.name == "PPO":
                if target:
                    mean = self.rl_alg.policy_target(obs)
                else: 
                    mean = self.rl_alg.policy(obs)
                std = torch.exp(self.rl_alg.log_std)

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).clamp(1e-3,10).sum(dim=-1)

        elif self.rl_alg.type == "deterministic":

            # Target policy or regular policy
            if target:
                actions = self.rl_alg.policy_target(obs)
            else:
                actions = self.rl_alg.policy(obs)

            # Noise for exploration
            if self.rl_alg.noisy:
                actions += torch.normal(0, self.rl_alg.exploration_rate,
                                        size=actions.shape)

            # Log probs and dist are 0 because this is deterministic
            log_probs = []
            dist = []

    # Clip actions to the action space of the env
    # actions = actions.clamp(self.env)

    return actions, log_probs, dist
