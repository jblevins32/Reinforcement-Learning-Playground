# Main training logic
import numpy as np
import torch
import torch.nn as nn
from my_simulation.robot_model import RobotModel

def Train(env, rl_alg, buffer, optimizer, discount=0.99, epsilon=0.2):
    # Step 0: get current state of environment
    state = torch.tensor(env.position, dtype=torch.float32).unsqueeze(0)

    # Step 1: forward pass on the actor and critic to get action and value
    mean, std, value = rl_alg.forward(state)

    # Step 2: create a distribution from the mean and std
    gaussian = torch.distributions.Normal(mean, std)
    action = gaussian.sample()

    # Step 3: take the action in the environment, using the action as a control command to the robot model. 
    state_new = RobotModel(state,action)
    reward = env.reward(state_new,state,action)
    done = env.reached_goal(state_new) # Deternime if the goal is reached

    # Step 4: calculate the advantage
    _, value_new = rl_alg.forward(torch.tensor(state_new, dtype=torch.float32).unsqueeze(0))
    Adv = reward + discount * value_new - value

    # Step 5: caluclate probability ratio of taking an action under the current policy vs the old policy (estimate divergence in the two policies)
    log_prob = torch.log(action_prob[action])
    log_prob_old = buffer.log_prob
    r = torch.exp(log_prob - log_prob_old)

    # Step 6: calculate the surrogate loss
    loss = -torch.min(r*Adv, torch.clamp(r,1-epsilon,1+epsilon)*Adv)

    # Step 7: calcualte the critic loss
    critic_loss = nn.MSELoss(value, reward)

    # Step 8: Total loss
    total_loss = loss + critic_loss

    # Step 9: Update networks
    total_loss.backward()
    optimizer.zero_grad()
    optimizer.step()

    # Step 10: Update buffer with this experience
    
    return env