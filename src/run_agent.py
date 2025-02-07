# Mujoco imports
import mujoco
from mujoco import viewer
from run_training import *
from test import *
from torch.optim import Adam
from RL_algorithms.ppo import *
from RL_algorithms.ppo_buffer import *
import matplotlib.pyplot as plt
# import future agent types here


# 2D Simulation imports
from my_simulation.sim import *
from my_simulation.animation import *

############################################################################################################

# Load and run the agent
def RunAgent(operation,epochs,t_steps,model_dir,simulator,live_sim, discount, epsilon, lr):

    epoch_vec = []
    reward_vec = []

    # Load the chosen simulator
    if simulator == "mujoco":
        model = mujoco.MjModel.from_xml_path(model_dir)
        data = mujoco.MjData(model)
    elif simulator == "2dsim":
        # Generate costmap and make obstacles
        env = Sim(grid_size=100)
        env.set_start(10, 10)
        env.set_goal(90, 90)
        env.make_obstacles(num_obstacles=75,cost_map=env.cost_map, obstacle_size=7,build_wall=False)

        # Initialize and show the animation
        anim = animate(env.cost_map, env.start, env.goal) 

    # Choose RL algorithm. See robot model for input dim (state) and output dim (action)
    rl_alg = PPO(input_dim=2, output_dim=2)
    optimizer = Adam(params=rl_alg.parameters(), lr=lr)

    # Train or test with the Mujoco simulation
    if live_sim == False:
            
        # Running for n epochs
        for epoch in range(epochs):
            print(f"Beginning Epoch {epoch+1}")
            
            # Reset the environment at beginning of each epoch
            env.reset()
            buffer = PPOBuffer(state_dim=2,action_dim=2)
            total_reward = 0

            # Running each epoch for t timesteps
            for _ in range(t_steps):

                # Train or Test the agent
                if operation == "train":
                    env, rl_alg, reward = RunTraining(env, rl_alg, buffer, optimizer, discount, epsilon)
                    total_reward+=reward
                    # anim.move(env.cost_map, env.position)
                else:
                    Test()

            epoch_vec.append(epoch)
            reward_vec.append(total_reward)
            plt.figure("reward plot")
            plt.plot(epoch_vec, reward_vec, color="blue")
            plt.xlabel('epoch')
            plt.ylabel('reward')
            plt.pause(.0001)

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
