import torch


def GetAction(rl_alg, obs, target, grad, noisy=False):
    # obs = obs.reshape(self.num_environments,self.n_obs)

    with torch.no_grad() if grad == False else torch.enable_grad():

        # Actions are based on deterministic vs stochastic policy

        if rl_alg.type == "stochastic":
            # Step 1: forward pass on the actor and critic to get action and value
            if rl_alg.name == "SAC" or rl_alg.name == "VPG":
                if target:
                    mean, log_std = rl_alg.policy_target(obs).chunk(2, dim=-1)
                else:
                    mean, log_std = rl_alg.policy(obs).chunk(2, dim=-1)
                std = torch.exp(log_std)  # Use clamp?
            elif rl_alg.name == "PPO":
                if target:
                    mean = rl_alg.policy_target(obs)
                else: 
                    mean = rl_alg.policy(obs)
                std = torch.exp(rl_alg.log_std)

            # Step 2: create a distribution from the logits (raw outputs) and sample from it
            dist = torch.distributions.Normal(mean, std)
            actions = dist.rsample()
            log_probs = dist.log_prob(actions).sum(dim=-1)

        elif rl_alg.type == "deterministic":

            # Target policy or regular policy
            if target:
                actions = rl_alg.policy_target(obs)
            else:
                actions = rl_alg.policy(obs)

            # Log probs and dist are 0 because this is deterministic
            log_probs = []
            dist = []

    # Noise for exploration, assigned to specific algorithms now due to misunderstanding
    if noisy and rl_alg.name == "DDPG":
        actions += torch.normal(0, rl_alg.exploration_rate,
                                        size=actions.shape).to(rl_alg.device)
    # elif not noisy and rl_alg.name == "SAC":
    #     actions = mean

    # Clip actions to the action space of the env
    # actions = actions.clamp(np.min(rl_alg.env.action_space.low),np.max(env.action_space.high))

    return actions, log_probs, dist