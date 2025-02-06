# Mujoco imports
import mujoco
from mujoco import viewer
from train import *
from test import *
from RL_algorithms.ppo import *
# import future agent types here


# 2D Simulation imports
from my_simulation.sim import *
from my_simulation.animation import *

# Load and run the agent
def RunAgent(operation,epochs,t_steps,model_dir,simulator,live_sim, discount, epsilon):

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

    # Choose RL algorithm. See robot model for input dim (state) and output dim (control)
    rl_alg = PPO(input_dim=2, output_dim=2)

    # Train or test with the Mujoco simulation
    if live_sim == True:
        # with viewer.launch_passive(model, data) as v:\
            
        # Running for n epochs
        for epoch in range(epochs):
            print(f"Beginning Epoch {epoch+1}")
            
            # Reset the environment at beginning of each epoch
            env.reset()
            buffer = []
            total_reward = 0

            # Running each epoch for t timesteps
            for _ in range(t_steps):

                # Train or Test the agent
                if operation == "1":
                    env = Train(env, rl_alg, buffer, discount, epsilon)
                    anim.move(env.cost_map, env.position)
                else:
                    Test()

    # Train or test without the Mujoco simulation
    # else:
    #     # Running for t timesteps
    #     for _ in range(t_steps):

    #         # Train or Test the agent
    #         if operation == "1":
    #             Train(data)
    #         else:
    #             Test()

    #         # Step Mujoco
    #         mujoco.mj_step(model, data)
