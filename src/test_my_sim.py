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
        action = Obstacle_avoid_control(obs, env)
        action = action_attack(action)
        obs_prev = obs.copy()
        obs, reward, done, truncated, info = env.step(action)
        obs = obs_attack(obs, obs_prev)
        env.env.env.env.map.robot_positions[:,1:] = obs[:,0:2]
        print(obs)
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
    map = env.env.env.env.map
    num_agents = env.env.env.env.num_agents
    agent_radius = env.env.env.env.agent_radius
    obstacle_positions = map.obstacle_positions[:,np.newaxis,0:-1]
    obstacle_radii = map.obstacle_positions[:,-1]
    robot_positions = obs
    env.render()

    kp_collisions_agents = 50
    kp_dist = 1
    kp_collisions_obstacles = 10

    # Distance control: correct for error of agents away from their goals
    distance_control = robot_positions[:,2:] - robot_positions[:,0:2]

    # Collision control: correct for agents being near or on obstacles

    # Get each agent's x and y center distances from each obstacle center: num_obstacles x num_agents x 2
    dist_obstacles_centers_xy = robot_positions[np.newaxis,:,:-2] - obstacle_positions

    # Get straight line distance from each agent center to obstacle center: num_obstacles x num_agents
    dist_obstacles_centers_r = np.linalg.norm(dist_obstacles_centers_xy,axis=2)

    # Get straight line distance from each agent border to obstacle border: num_obstacles x num_agents
    dist_obstacles_radius_r = dist_obstacles_centers_r - np.expand_dims(obstacle_radii,axis=-1) - agent_radius

    # Get directions of agents from each obstacle: num_obstacles x num_agents x 2
    dist_obstacles_centers_r = np.expand_dims(dist_obstacles_centers_r, axis=-1)
    dir_obstacles = dist_obstacles_centers_xy/dist_obstacles_centers_r

    inv_law = 1/(dist_obstacles_radius_r**2)
    collision_control_obstacles = np.sum(np.expand_dims(inv_law,axis=-1) * dir_obstacles,axis=0)

    # Collision control: correct for agents being near eachother
    if num_agents > 1:
        # Get each agent's x and y center distances from each other's center: num_agents x num_agents x 2
        dist_agents_centers_xy = robot_positions[np.newaxis,:,:-2] - robot_positions[:,np.newaxis,:-2]

        # Get straight line distance from each agent center to obstacle center: num_agents x num_agents
        dist_agents_centers_r = np.linalg.norm(dist_agents_centers_xy,axis=2)

        # Get straight line distance from each agent border to obstacle border: num_agents x num_agents
        dist_agents_radius_r = dist_agents_centers_r - agent_radius*2

        # Get directions of agents from each obstacle: num_agents x num_agents x 2
        dist_agents_centers_r = np.expand_dims(dist_agents_centers_r, axis=-1)
        dir_agents = dist_agents_centers_xy/dist_agents_centers_r

        # Set the diagonals to 0 because they represent agents' directions to themselves 
        diagonal_mask = np.arange(dir_agents.shape[0])
        dir_agents[diagonal_mask, diagonal_mask] = np.array([1e-10,1e-10])

        inv_law = 1/(dist_agents_radius_r**2)
        inv_law[diagonal_mask, diagonal_mask] = 1e-10 # Set the diagonals to 0 because they represent agents' distances to themselves. Doing this here so we don't divide by 0 on the inv_law
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