## Bugs
- MRPP_Env pops up 64 windows at start of training
- MRPP_Env not learning
    - update obs to include obstacles
    - update reward weights
- Tensorboard not updating with discrete envs

## RARL
- They assume transition function has errors instead of perfect environment transitions
- Create plotting for attacked case

## General
- Make continuous algs (DDPG)
- IMPORTANT Make RARL
- Fix environment.yaml to work in PACE

## Papers
- MADDPG - [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275)
- RARL - [Robust Adversarial Reinforcement Learning](https://arxiv.org/pdf/1703.02702)