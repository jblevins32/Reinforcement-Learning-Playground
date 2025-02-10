# Mujoco imports
import mujoco
from mujoco import viewer
import gymnasium as gym
from test import *
from torch.optim import Adam
import matplotlib.pyplot as plt
from buffer import *

from RL_algorithms.reinforce import *
from RL_algorithms.vpg import *
from RL_algorithms.ppo import *

############################################################################################################

# Load and run the agent
def Agent(rl_alg,operation,num_environments,epochs,t_steps,env,n_obs,n_actions,discount, epsilon, lr, live_sim):

    # Initialize plot variables
    epoch_vec = []
    reward_vec = []
    frames = []

    # Choose RL algorithm
    if rl_alg == "PPO":
        rl_alg = PPO(input_dim=n_obs, output_dim=n_actions, discount=discount, epsilon=epsilon)
    elif rl_alg == "REINFORCE":
        rl_alg = REINFORCE(input_dim=n_obs, output_dim=n_actions)
    elif rl_alg == "VPG":
        rl_alg = VPG(input_dim=n_obs, output_dim=n_actions)

    # Choose optimizer
    optimizer = Adam(params=rl_alg.parameters(), lr=lr)

    # Create buffer
    buffer = Buffer(n_steps=t_steps, n_envs=num_environments, n_obs=4, n_actions=1)
    
    # Running for n epochs
    for epoch in range(epochs):
        print(f"Beginning Epoch {epoch+1}")
        
        # Reset the environment at beginning of each epoch
        obs, _ = env.reset()
        obs = torch.Tensor(obs)

        # Rollout for t timesteps
        for t in range(t_steps):
            # Train or Test the agent
            if operation == "train":
                env, obs, buffer = rl_alg.train(t, env, obs, buffer)

                # For visualization
                frames.append(env.render()[0])

        # Get total expected return from the rollout in this epoch
        buffer.calc_returns()
        loss = rl_alg.loss_func(buffer)

        # Update networks
        update_epochs = 10 if rl_alg.name == "PPO" else 1

        for _ in range(update_epochs):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

        buffer.detach()

        # anim.move(env.cost_map, env.position)

        # Plot 
        epoch_vec.append(epoch)
        reward_vec.append(buffer.rewards.mean())
        plt.figure('Reward Plot')
        plt.plot(epoch_vec, reward_vec, color="blue")
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.pause(.000001)

        if live_sim == True:
            plt.figure('Environment')
            plt.imshow(frames[-1])
            plt.axis('off')
            plt.pause(.000001)

    print("Training Complete!")

    # Train or test without the Mujoco simulation
    # else:
        # with viewer.launch_passive(model, data) as v:\
            
        #     # Running for t timesteps
        #     for _ in range(t_steps):

        #         # Train or Test the agent
        #         if operation == "1":
        #             Train(data)
        #         else:
        #             Test()

        #         # Step Mujoco
        #         mujoco.mj_step(model, data)
                #   v.step()
