from agent import Agent
from my_sim.gym_simulation import *
from create_env import CreateEnv

if __name__ == "__main__":
    # Create environment
    env, n_obs, n_actions, writer, config = CreateEnv(operation="train")

    # Start Training
    agent = Agent(env, n_obs, n_actions, writer, **config)
    agent.train()

    env.close()

    print("Training Complete!")
