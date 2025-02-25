import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt

register(id='MRPP_Env',
         entry_point='my_sim.gym_simulation:MRPP_Env')

class MRPP_Env(gym.Env):

    # Render the environment
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self,render_mode,**kwargs):
        
        self.num_agents = kwargs.get('num_agents', 5)
        self.map_size = kwargs.get('map_size', (10, 10))
        self.num_obstacles = kwargs.get('num_obstacles', 5)
        self.obstacle_radius_max = kwargs.get('obstacle_radius_max', 1)
        self.obstacle_cost = kwargs.get('obstacle_cost', 1)
        self.dynamics = kwargs.get('dynamics', None)
        self.max_episode_steps = kwargs.get('max_episode_steps', 100)
        self.communication_range = kwargs.get('communication_range', 5)
        self.observation_range = kwargs.get('observation_range', 5)
        self.centralized = kwargs.get('centralized', False)
        self.dt = kwargs.get('dt', 0.1)
        self.done_threshold = kwargs.get('done_threshold',5)
        self.w_dist = kwargs.get('w_dist',1)
        self.w_coll = kwargs.get('w_coll',1)
        self.w_dir = kwargs.get('w_dir',1)
        self.w_goal = kwargs.get('w_goal',1)
        self.seed_value = kwargs.get("seed_value", None)
        self.num_environments = kwargs.get('num_environments',1)
        self.n_actions = 2
        self.n_obs = 4
        self.agent_radius = kwargs.get('agent_radius',1.5)

        self.render_mode = render_mode

        # Action space: x and y velocities for each agent
        self.action_space = spaces.Box(low=-1,high=1, shape=(self.num_agents,self.n_actions), dtype=float)

        # Create observation space: x and y position for each agent and same for each target
        self.observation_space = spaces.Box(low=0,high=np.max(self.map_size), shape=(self.num_agents,self.n_obs), dtype=float)

        # Create the map
        self.map = EnvMap(self.num_agents, self.map_size, self.num_obstacles, self.obstacle_radius_max, self.obstacle_cost, self.dt, self.done_threshold,self.w_dist,self.w_coll,self.w_dir,self.w_goal, self.num_environments, self.n_actions, self.agent_radius)

    def reset(self, seed=None, options=None):
        super().reset(seed=self.seed_value)

        # Seed the overarching gym env
        self.seed(self.seed_value)

        # Reset the env
        self.map.reset(seed=self.seed_value)

        # Get initial observation
        obs = self.map.get_obs()

        # self.render()

        info = {}

        return obs, info
    
    def seed(self, seed=None):
        """ Seed the environment and sub-components """
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        # Format the actions
        action = action.reshape(self.num_agents,self.n_actions)

        # Take the action
        terminated = self.map.perform_action(action)

        # Get reward and observation
        reward = self.map.get_reward(action)
        obs = self.map.get_obs() # observation done after the action

        info = {}

        # If we want to cut it off after a certain amount of steps
        truncated = False

        # self.render()

        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.map.plot_env()

class EnvMap():
    def __init__(self, num_agents, map_size, num_obstacles, obstacle_radius_max, obstacle_cost, dt, done_threshold, w_dist, w_coll, w_dir, w_goal, num_environments, n_actions,agent_radius):
        self.num_agents = num_agents
        self.map_width = map_size[0]
        self.map_height = map_size[1]
        self.num_obstacles = num_obstacles
        self.obstacle_radius_max = obstacle_radius_max
        self.obstacle_cost = obstacle_cost
        self.dt = dt
        self.num_environments = num_environments
        self.n_actions = n_actions
        self.agent_radius = agent_radius

        self.done_threshold = done_threshold

        # # reward weights
        self.w_dist = w_dist
        self.w_coll = w_coll
        self.w_dir = w_dir
        self.w_goal = w_goal

        self.done = False

        # Store agent paths
        self.agent_paths = [[] for _ in range(self.num_agents)]

        # Store observations for plotting error
        self.obs_vec = []

        # Initialize Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1,figsize=(self.map_width/10, self.map_height/10))

    def reset(self, seed=None):
        
        self.seed(seed)

        # Reset paths for each agent
        self.agent_paths = [[] for _ in range(self.num_agents)]

        self.generate_locations()
        self.generate_obstacles()

    def seed(self, seed=None):
        """ Properly seed the environment for reproducibility """
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def generate_locations(self):
        # Generate random starting positions for each agent in the map
        robot_index = np.linspace(1,self.num_agents,self.num_agents,dtype=int)
        robot_positions_x = self.np_random.random(self.num_agents) * self.map_width
        robot_positions_y = self.np_random.random(self.num_agents) * self.map_height
        self.robot_positions_start = np.concatenate((robot_index.reshape(-1,1),robot_positions_x.reshape(-1,1),robot_positions_y.reshape(-1,1)),1)
        self.robot_positions = self.robot_positions_start

        # Generate random target positions for each agent in the map

        robot_targets_x = self.np_random.random(self.num_agents) * self.map_width
        robot_targets_y = self.np_random.random(self.num_agents) * self.map_height
        self.robot_targets_start = np.concatenate((robot_index.reshape(-1,1),robot_targets_x.reshape(-1,1),robot_targets_y.reshape(-1,1)),1)
        self.robot_targets = self.robot_targets_start

    def generate_obstacles(self):
        
        obstacle_positions_x = self.np_random.random(self.num_obstacles) * self.map_width
        obstacle_positions_y = self.np_random.random(self.num_obstacles) * self.map_height
        obstacle_radii = self.np_random.random(self.num_obstacles) * self.obstacle_radius_max
        self.obstacle_positions = np.concatenate((obstacle_positions_x.reshape(-1,1),obstacle_positions_y.reshape(-1,1),obstacle_radii.reshape(-1,1)),1)
        
    def check_obstacle_collisions(self):

        # Distance formula broadcasting with robots and obstacles
        dist_mask = np.sqrt((self.obstacle_positions[:,0].reshape(-1,1) - self.robot_positions[:,1].reshape(1,-1))**2 + (self.obstacle_positions[:,1].reshape(-1,1) - self.robot_positions[:,2].reshape(1,-1))**2) < self.obstacle_positions[:,2].reshape(-1,1)
        collisions = np.sum(dist_mask)

        return collisions

    def get_obs(self):
        return np.concatenate((self.robot_positions[:, 1:], self.robot_targets[:, 1:]), axis=1)  # Shape: (num_agents, 4)
    
    def perform_action(self, actions):

        # Assume velocity commands
        self.robot_positions[:,1:] += actions*self.dt

        # Check if robots have reached targets
        if np.all(np.sqrt(np.sum((self.robot_positions[:,1:] - self.robot_targets[:,1:])**2,axis=1)) < self.done_threshold):
            self.done = True

        # Update paths for each agent
        for i in range(self.num_agents):
            self.agent_paths[i].append((self.robot_positions[i, 1], self.robot_positions[i, 2]))
            
        return self.done
    
    def get_reward(self, actions):

        # Distance to goals reward. Want less
        dist_rmse = np.sqrt(np.mean((self.robot_targets[:,1:] - self.robot_positions[:,1:])**2))

        # Collisions. Want less.
        num_collisions = self.check_obstacle_collisions()

        # Direction of travel. Want more
        dir_accuracy = 0

        for idx,action in enumerate(actions):
            control_norm_dir = action/np.linalg.norm(action)
            desired_norm_dir = (self.robot_targets[idx,1:] - self.robot_positions[idx,1:])/np.linalg.norm((self.robot_targets[idx,1:] - self.robot_positions[idx,1:]))
            dir_accuracy += np.dot(desired_norm_dir, control_norm_dir)

        # Reaching goal. Want more.

        reward = (- self.w_dist * dist_rmse) + (- self.w_coll * num_collisions) + (self.w_dir * dir_accuracy) + (self.w_goal * self.done)

        return reward
    
    def plot_env(self):
        # Return the rgb array for the video and also plot the env
        self.ax1.clear()  # Clear previous plot

        self.ax1.set_xlim(0, self.map_width)
        self.ax1.set_ylim(0, self.map_height)

        # Draw environment background
        rectangle = plt.Rectangle((0,0), self.map_width, self.map_height, color='green', alpha=0.25)
        self.ax1.add_patch(rectangle)

        # Plot robots and targets
        self.ax1.scatter(self.robot_positions[:,1], self.robot_positions[:,2], c='blue', linewidths=self.agent_radius*6.66)
        self.ax1.scatter(self.robot_targets[:,1], self.robot_targets[:,2], c='red', linewidths=self.agent_radius*6.66)

        # Plot robot paths
        for path in self.agent_paths:
            if len(path) > 1:
                path_x, path_y = zip(*path)
                self.ax1.plot(path_x, path_y, 'k--', alpha=0.6)  # Dotted black line for paths

        # Label robots
        for robot in self.robot_positions:
            self.ax1.text(robot[1], robot[2], str(int(robot[0])), color='white', ha='center', va='center')

        # Label targets
        for target in self.robot_targets:
            self.ax1.text(target[1], target[2], str(int(target[0])), color='white', ha='center', va='center')

        # Plot obstacles
        for obstacle in self.obstacle_positions:
            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='grey', alpha=0.5)
            self.ax1.add_patch(circle)

        # Error plot:
        obs = self.get_obs()
        self.obs_vec = np.append(self.obs_vec, np.linalg.norm(obs[:,:2] - obs[:,2:]))
        self.ax2.plot(self.obs_vec)

        # self.fig.legend()
        self.fig.show()
        plt.pause(0.01)

        # Save rgb array
        # self.fig.canvas.draw()
        # img_array = np.array(self.fig.canvas.renderer.buffer_rgba())

        # return img_array