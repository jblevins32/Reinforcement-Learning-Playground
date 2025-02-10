# Mujoco imports
import mujoco
from mujoco import viewer
import gymnasium as gym
from test import *
from torch.optim import Adam
from RL_algorithms.ppo import *
from RL_algorithms.ppo_buffer import *
import matplotlib.pyplot as plt
from traj_data import *
# import future agent types here


# 2D Simulation imports
from my_simulation.sim import *
from my_simulation.animation import *

############################################################################################################

# Load and run the agent
def RunAgent(operation,num_environments,epochs,t_steps,model_dir,env,live_sim, discount, epsilon, lr):

    epoch_vec = []
    reward_vec = []
    frames = []

    # Load the chosen simulator environment
    # MuJoCo
    if env == "mujoco":
        model = mujoco.MjModel.from_xml_path(model_dir)
        data = mujoco.MjData(model)

    # Personal Simulation
    elif env == "2dsim":
        # Generate costmap and make obstacles
        env = Sim(grid_size=100)
        env.set_start(10, 10)
        env.set_goal(90, 90)
        env.make_obstacles(num_obstacles=75,cost_map=env.cost_map, obstacle_size=7,build_wall=False)

        # Initialize and show the animation
        anim = animate(env.cost_map, env.start, env.goal) 

    # Gym
    elif env == "gym":
        env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1", render_mode="rgb_array") for _ in range(num_environments)])

    # Choose RL algorithm
    rl_alg = PPO(input_dim=4, output_dim=2)

    # Choose optimizer
    optimizer = Adam(params=rl_alg.parameters(), lr=lr)

    # Create buffer
    buffer = TrajData(n_steps=t_steps, n_envs=num_environments, n_obs=4, n_actions=1)
    
    # Running for n epochs
    for epoch in range(epochs):
        print(f"Beginning Epoch {epoch+1}")
        
        # Reset the environment at beginning of each epoch
        obs, _ = env.reset()
        obs = torch.Tensor(obs)

        # Running each epoch for t timesteps
        # This is a rollout
        for t in range(t_steps):
            # Train or Test the agent
            if operation == "train":
                env, obs, buffer = rl_alg.train(t, env, obs, discount, epsilon, buffer)

                # For visualization
                frames.append(env.render()[0])

            else:
                Test()

        # loss = rl_alg.loss_func(t, obs_new, torch.Tensor(reward), discount, value, log_probs, epsilon, buffer)

        # Get total expected return from this epoch
        buffer.calc_returns()
        loss = -torch.sum(buffer.log_probs * buffer.returns)/(buffer.n_steps*buffer.returns.shape[1])

        # Update networks
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

        plt.figure('Environment')
        plt.imshow(frames[-1])
        plt.axis('off')
        plt.pause(.000001)

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
