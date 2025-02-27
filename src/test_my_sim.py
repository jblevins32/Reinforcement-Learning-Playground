import gymnasium as gym
from get_params import GetParams
from my_sim.gym_simulation import *
from gymnasium.utils.env_checker import check_env
from create_env import CreateEnv
from tensorboard_setup import SetupBoard

def TestMySim():
    '''For testing my env!'''
    config = GetParams()

    # Tensor board setup
    # writer = SetupBoard('My Sim Test')

    # Create environment
    env,n_actions,n_obs = CreateEnv(operation="test")
    # check_env(env.unwrapped)

    # Reset the env
    obs, _ = env.reset()

    done = False
    while not done:

        # get action
        action = Obstacle_avoid_control(obs, env)

        # test control attack
        if config['test_attack']:
            action = action_attack(action)
            obs_prev = obs.copy()

        # Take step
        obs, reward, done, truncated, info = env.step(action)

        # test obs attack
        if config['test_attack']:
            obs = obs_attack(obs, obs_prev)
            env.env.env.env.map.robot_positions[:,1:] = obs[:,0:2]

        env.render()

    while done:
        env.render()

    env.close()

def P_control(obs):
    kp = 1
    action = obs[:,2:] - obs[:,0:2]
    return kp*action

def Obstacle_avoid_control(obs, env):
    # Control = distance + agent collisions + obstacle collisions
    root = env.env.env.env
    map = root.map
    num_agents = root.num_agents
    robot_positions = obs
    env.render()

    kp_collisions_agents = 20
    kp_dist = 2
    kp_collisions_obstacles = 10

    # Distance control: correct for error of agents away from their goals
    distance_control = robot_positions[:,2:] - robot_positions[:,0:2]

    # Collision control: correct for agents being near or on obstacles
    dist_obstacles_radius_r, dir_obstacles = map.check_obstacle_distances("agent")
    inv_law = 1/(dist_obstacles_radius_r**2)
    collision_control_obstacles = np.sum(np.expand_dims(inv_law,axis=-1) * dir_obstacles,axis=0)

    # Collision control: correct for agents being near eachother
    if num_agents > 1:
        dist_agents_radius_r, dir_agents = map.check_agent_distances()

        inv_law = 1/(dist_agents_radius_r**2)

        # Set the diagonals to ~0 because they represent agents' directions to themselves 
        diagonal_mask = np.arange(dir_agents.shape[0])
        inv_law[diagonal_mask, diagonal_mask] = 0 # Set the diagonals to 0 because they represent agents' distances to themselves. Doing this here so we don't divide by 0 on the inv_law
        
        collision_control_agents = np.sum(np.expand_dims(inv_law,axis=-1) * dir_agents,axis=0)
    else:
        collision_control_agents = 0

    return kp_dist*distance_control + kp_collisions_obstacles*collision_control_obstacles + kp_collisions_agents*collision_control_agents # Returns num_agents x 2 matrix of x and y control commands

def action_attack(action):
    Su = 0.1
    return Su * action

def obs_attack(obs, obs_prev):
    Sx = 1/0.1
    dx = obs_prev - Sx*obs_prev
    return Sx*obs + dx

if __name__ == "__main__":
    TestMySim()