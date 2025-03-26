## Notes for getting MARL to work
- std is clamped in get_action right now. Double check these bounds are fine.
- agent targets and obstacle info are all static ... need this in my observation if it isn't changing?
- Check to be sure rewards are balanced

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

## Papers
- MADDPG - [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275)
- RARL - [Robust Adversarial Reinforcement Learning](https://arxiv.org/pdf/1703.02702)