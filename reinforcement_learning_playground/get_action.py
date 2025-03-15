import torch


def GetAction(rl_alg, obs, target, grad):
    # obs = obs.reshape(self.num_environments,self.n_obs)

    with torch.no_grad() if grad == False else torch.enable_grad():
        if rl_alg.type == "stochastic":
            # Step 1: forward pass on the actor and critic to get action and value
            if rl_alg.name == "SAC":
                if target:
                    mean, log_std = rl_alg.policy_target(obs).chunk(2, dim=-1)
                else:
                    mean, log_std = rl_alg.policy(obs).chunk(2, dim=-1)
                std = torch.exp(log_std).clamp(0.1,0.4)  # Use clamp?
            elif rl_alg.name == "PPO_CONT":
                if target:
                    mean = rl_alg.policy_target(obs)
                else: 
                    mean = rl_alg.policy(obs)
                std = torch.exp(rl_alg.log_std).clamp(0.1,0.4) # Use clamp?

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            dist = torch.distributions.Normal(mean, std)
            actions = dist.rsample()
            actions = actions
            log_probs = dist.log_prob(actions).sum(dim=-1)

        elif rl_alg.type == "deterministic":
            if target:
                actions = rl_alg.policy_target(obs)
            else:
                actions = rl_alg.policy(obs)

            if rl_alg.noisy:
                actions += torch.normal(0, rl_alg.exploration_rate,
                                        size=actions.shape)

            log_probs = []
            dist = []

    return actions, log_probs, dist
