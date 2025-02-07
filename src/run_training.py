# Main training logic
import numpy as np
import torch
import torch.nn as nn
from my_simulation.robot_model import RobotModel
from RL_algorithms.ppo import *

def RunTraining(env, rl_alg, buffer, optimizer, discount=0.99, epsilon=0.2):

    loss, state_new, reward = rl_alg.train(env, discount, epsilon, buffer)

    # Update networks
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()   
    
    # Update environment
    env.position = state_new 

    return env, rl_alg, reward