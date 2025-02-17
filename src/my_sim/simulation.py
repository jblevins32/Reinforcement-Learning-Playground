import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gym import spaces
from typing import Dict, Tuple, List, Optional

class MapGenerator:
    """
    Generates either:
      - A random 2D obstacle map (binary: free vs. obstacle)
      - Or a 'mountainous' map with varying costs.

    Attributes:
        map_size (Tuple[int, int]): Size of the map (width, height).
        num_obstacles (int): Number of obstacles (if using obstacle-based map).
        mountainous (bool): Whether to generate a cost map with random 'mountains'.
        obstacle_cost (float): Cost for being on an obstacle tile.
        mountainous_scale (float): Scale factor for mountainous generation.
    """

    def __init__(self,
                 map_size: Tuple[int, int],
                 num_obstacles: int = 5,
                 mountainous: bool = False,
                 obstacle_cost: float = 5.0,
                 mountainous_scale: float = 1.0):
        self.map_size = map_size
        self.num_obstacles = num_obstacles
        self.mountainous = mountainous
        self.obstacle_cost = obstacle_cost
        self.mountainous_scale = mountainous_scale
        # Internal representation:
        #   cost_map: 2D numpy array of costs (0 -> minimal cost, higher -> more cost)
        #   obstacle_map: 2D array of booleans (True = obstacle)
        self.cost_map = np.zeros(map_size, dtype=np.float32)
        self.obstacle_map = np.zeros(map_size, dtype=bool)

        self._generate_map()

    def _generate_map(self):
        """
        Generate either random obstacles or a mountainous cost map (or both).
        """
        width, height = self.map_size
        if self.mountainous:
            # Simple mountainous cost: Perlin noise or random Gaussian lumps
            # For simplicity, just create some random bumps
            x_coords = np.linspace(0, 2 * np.pi, width)
            y_coords = np.linspace(0, 2 * np.pi, height)
            xv, yv = np.meshgrid(x_coords, y_coords)
            noise = (np.sin(xv) + np.cos(yv) + np.random.randn(height, width)) * self.mountainous_scale
            self.cost_map = np.clip(1.0 + noise, 0.0, None)

        # Place obstacles randomly (only if not mountainous or user wants obstacles too)
        # Here, let's do both if mountainous == True and num_obstacles > 0
        for _ in range(self.num_obstacles):
            ox = np.random.randint(0, width)
            oy = np.random.randint(0, height)
            # random obstacle radius
            r = np.random.randint(1, min(width, height)//8)
            # mark a circular region as obstacle
            for i in range(max(0, ox-r), min(width, ox+r)):
                for j in range(max(0, oy-r), min(height, oy+r)):
                    if (i - ox)**2 + (j - oy)**2 <= r**2:
                        self.obstacle_map[j, i] = True
                        self.cost_map[j, i] += self.obstacle_cost

    def get_cost(self, x: float, y: float) -> float:
        """
        Get the cost at a continuous location (x, y).
        We'll floor or round to nearest cell for simplicity.
        """
        ix = int(np.clip(np.floor(x), 0, self.map_size[0]-1))
        iy = int(np.clip(np.floor(y), 0, self.map_size[1]-1))
        return self.cost_map[iy, ix]

    def is_obstacle(self, x: float, y: float) -> bool:
        """
        Check if the given continuous point is in an obstacle cell.
        """
        ix = int(np.clip(np.floor(x), 0, self.map_size[0]-1))
        iy = int(np.clip(np.floor(y), 0, self.map_size[1]-1))
        return self.obstacle_map[iy, ix]


class MultiRobot2DEnv(gym.Env):
    """
    A multi-robot 2D environment with optional centralized or decentralized observations.

    Observation:
      - Centralized: single global observation = positions, velocities, and goals of all robots
      - Decentralized: each agent's observation includes only:
            - Its own position, velocity, and goal
            - Positions (and velocities) of other agents if within comm range
            - Possibly partial map info? (For simplicity, omit or add if needed)

    Action:
      - Continuous. The shape depends on the chosen dynamics:
          single_integrator -> 2D velocity
          unicycle -> (v, omega)
          double_integrator -> 2D acceleration
      - We pass a single numpy array that is a concatenation of actions for each agent.

    Reward:
      - +ve for moving in direction of goal
      - -ve for large control input
      - -ve for being on high-cost terrain
      - No termination on collisions, but robots bounce off each other/obstacles

    Episode ends:
      - If time limit is reached
      - Or if all agents have reached their goals

    Rendering:
      - Matplotlib, updated after every step
      - Option to save .mp4 at the end of an episode
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self,
                 num_agents: int = 2,
                 map_size: Tuple[int, int] = (50, 50),
                 num_obstacles: int = 5,
                 obstacle_cost: float = 5.0,
                 mountainous: bool = False,
                 mountainous_scale: float = 1.0,
                 robot_dynamics: str = "single_integrator",
                 max_episode_steps: int = 200,
                 communication_range: float = 10.0,
                 observation_range: float = 10.0,
                 centralized: bool = True,
                 save_video: bool = True,
                 video_filename: str = "multi_robot_env_demo.mp4",
                 render_mode: str = "human"):
        """
        Args:
            num_agents (int): Number of robots/agents.
            map_size (Tuple[int, int]): (width, height) of the map in continuous space.
            num_obstacles (int): Number of obstacles to generate (circular).
            obstacle_cost (float): Cost for an obstacle area.
            mountainous (bool): Use mountainous cost map or not.
            mountainous_scale (float): Scale factor for mountainous cost generation.
            robot_dynamics (str): 'single_integrator', 'unicycle', or 'double_integrator'.
            max_episode_steps (int): Time limit for an episode.
            communication_range (float): Comm range if decentralized.
            observation_range (float): Observation range if decentralized.
            centralized (bool): If True, we give a single global observation. If False, partial obs.
            save_video (bool): If True, save an .mp4 at the end.
            video_filename (str): Name of output video file.
            render_mode (str): "human" or None for no render.
        """
        super(MultiRobot2DEnv, self).__init__()

        self.num_agents = num_agents
        self.map_size = map_size
        self.width, self.height = map_size
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # Map generation
        self.map_gen = MapGenerator(
            map_size=map_size,
            num_obstacles=num_obstacles,
            mountainous=mountainous,
            obstacle_cost=obstacle_cost,
            mountainous_scale=mountainous_scale
        )

        # Robot dynamics choice
        if robot_dynamics not in ["single_integrator", "unicycle", "double_integrator"]:
            raise ValueError("robot_dynamics must be one of "
                             "['single_integrator', 'unicycle', 'double_integrator']")
        self.robot_dynamics = robot_dynamics

        # Multi-agent config
        self.centralized = centralized
        self.comm_range = communication_range
        self.obs_range = observation_range

        # Positions, velocities, goals
        # agent_states: shape (num_agents, 4) => (x, y, vx, vy) for single_integrator/double_integrator
        # for unicycle, might store (x, y, theta, v) or similar,
        # but for simplicity let's store (x, y, vx, vy) & interpret differently.
        self.agent_states = np.zeros((self.num_agents, 4), dtype=np.float32)
        self.agent_goals = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agents_reached_goal = np.array([False]*self.num_agents)

        # Episode termination if all goals reached or time limit
        self.done = False

        # Action space
        # single_integrator: (vx, vy)
        # unicycle: (v, omega)
        # double_integrator: (ax, ay)
        if self.robot_dynamics == "single_integrator":
            self.action_dim = 2
        elif self.robot_dynamics == "unicycle":
            self.action_dim = 2
        else:  # double_integrator
            self.action_dim = 2

        # The total action space is for all agents: shape = (num_agents * action_dim,)
        # We'll clip actions to, say, [-1, 1] in each dimension (you can adjust).
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_agents * self.action_dim,), dtype=np.float32
        )

        # Observation space
        # If centralized: we observe all agents' positions & velocities + all goals => shape?
        #   shape = (num_agents * 4 + num_agents * 2) = num_agents*6
        # If decentralized: each agent gets a partial view => but for Gym's standard single obs,
        #   we usually flatten. We'll still define a single observation_space that can hold
        #   the largest possible dimension (centralized).
        # For multi-agent training frameworks, you might return a list of obs. But let's keep it simple.
        if self.centralized:
            obs_dim = self.num_agents * 6  # (x,y,vx,vy, gx,gy) repeated
        else:
            # We'll define a max dimension if every agent sees every other agent in range
            # The dimension for each agent = 6 (self pos, vel, goal) + 4*(others), up to (num_agents-1).
            # This is a naive bound. In practice, we might have fewer if they are out of range.
            max_others = self.num_agents - 1
            obs_dim_per_agent = 6 + 4 * max_others  # (self pos+vel+goal) + (pos+vel of others)
            obs_dim = obs_dim_per_agent * self.num_agents
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Rendering
        self.render_mode = render_mode
        self.save_video = save_video
        self.video_filename = video_filename
        self.fig = None
        self.ax = None
        self.ims = []  # for animation frames

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment state:
          - Randomize agent positions (not on obstacles).
          - Randomize goals (not on obstacles).
          - Reset step_count, done.
        """
        super().reset(seed=seed)
        self.done = False
        self.step_count = 0
        self.agents_reached_goal = np.array([False]*self.num_agents)

        for i in range(self.num_agents):
            self.agent_states[i, :] = self._get_random_free_state()
            self.agent_goals[i, :] = self._get_random_free_pos()

        if self.render_mode == "human":
            self._init_render()

        return self._get_observation(), {}

    def _get_random_free_state(self) -> np.ndarray:
        """
        Sample a random free (x, y) that's not in obstacle and random velocity (0,0).
        """
        while True:
            x = np.random.uniform(0, self.width - 1)
            y = np.random.uniform(0, self.height - 1)
            if not self.map_gen.is_obstacle(x, y):
                return np.array([x, y, 0.0, 0.0], dtype=np.float32)

    def _get_random_free_pos(self) -> np.ndarray:
        """
        Sample a random free (x, y) for a goal or start.
        """
        while True:
            x = np.random.uniform(0, self.width - 1)
            y = np.random.uniform(0, self.height - 1)
            if not self.map_gen.is_obstacle(x, y):
                return np.array([x, y], dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        Gym step:
          - Parse each agent's action from the concatenated array.
          - Update each agent's state according to the chosen dynamics.
          - Check for collisions, bounce off if needed.
          - Compute reward, check done/truncated, return obs, reward, done, truncated, info.
        """
        if self.done:
            # If we're already done, call reset or handle gracefully
            return self._get_observation(), 0.0, True, True, {}

        actions = action.reshape(self.num_agents, self.action_dim)

        # Update each agent's state
        for i in range(self.num_agents):
            if not self.agents_reached_goal[i]:
                self._apply_action(i, actions[i])

        # Check collisions between agents & obstacles => bounce
        self._handle_collisions()

        # Check if each agent reached its goal
        for i in range(self.num_agents):
            if not self.agents_reached_goal[i]:
                dist_to_goal = np.linalg.norm(self.agent_states[i, :2] - self.agent_goals[i])
                if dist_to_goal < 1.0:  # threshold
                    self.agents_reached_goal[i] = True

        # Compute reward
        rewards = self._compute_rewards(actions)

        # Check if all goals reached or time limit
        self.step_count += 1
        all_reached = np.all(self.agents_reached_goal)
        truncated = (self.step_count >= self.max_episode_steps)
        done = all_reached or truncated
        self.done = done

        # Info
        info = {"step_count": self.step_count}

        # Render
        if self.render_mode == "human":
            self.render()

        # If done and saving video, generate .mp4
        if done and self.save_video and self.render_mode == "human":
            self._save_video()

        # Return (obs, reward, done, truncated, info)
        # If following new gym API: (obs, reward, done, truncated, info)
        # We'll follow the signature you requested.
        return self._get_observation(), rewards, done, truncated, info

    def _apply_action(self, agent_idx: int, act: np.ndarray):
        """
        Update agent state using the chosen dynamics.
        """
        x, y, vx, vy = self.agent_states[agent_idx]

        if self.robot_dynamics == "single_integrator":
            # act = (vx_cmd, vy_cmd)
            # Update position directly
            vx_cmd, vy_cmd = act
            # let's do a small time step
            dt = 1.0
            new_x = x + vx_cmd * dt
            new_y = y + vy_cmd * dt
            new_vx = vx_cmd
            new_vy = vy_cmd
            # clip to map bounds
            new_x = np.clip(new_x, 0, self.width - 1)
            new_y = np.clip(new_y, 0, self.height - 1)
            self.agent_states[agent_idx] = [new_x, new_y, new_vx, new_vy]

        elif self.robot_dynamics == "unicycle":
            # act = (v, omega)
            # We'll interpret (vx, vy) in the state as (theta, v) for convenience, but let's do simpler:
            # let's store orientation in vx and vy for now
            theta = np.arctan2(vy, vx)  # orientation from velocity
            speed = np.linalg.norm([vx, vy])
            v, omega = act
            dt = 1.0
            # new orientation
            new_theta = theta + omega * dt
            new_speed = speed + (v - speed) * 0.5  # or some partial update
            # new position
            new_x = x + new_speed * np.cos(new_theta) * dt
            new_y = y + new_speed * np.sin(new_theta) * dt
            # clip
            new_x = np.clip(new_x, 0, self.width - 1)
            new_y = np.clip(new_y, 0, self.height - 1)
            new_vx = new_speed * np.cos(new_theta)
            new_vy = new_speed * np.sin(new_theta)
            self.agent_states[agent_idx] = [new_x, new_y, new_vx, new_vy]

        else:  # double_integrator
            # act = (ax, ay), acceleration
            ax, ay = act
            dt = 1.0
            # new velocity
            new_vx = vx + ax * dt
            new_vy = vy + ay * dt
            # new position
            new_x = x + vx * dt + 0.5 * ax * (dt**2)
            new_y = y + vy * dt + 0.5 * ay * (dt**2)
            # clip
            new_x = np.clip(new_x, 0, self.width - 1)
            new_y = np.clip(new_y, 0, self.height - 1)
            self.agent_states[agent_idx] = [new_x, new_y, new_vx, new_vy]

    def _handle_collisions(self):
        """
        If agents overlap or agent is in obstacle, bounce them off.
        We'll do a simplistic approach: if agent is in an obstacle or
        collides with another agent, just invert velocity.
        """
        for i in range(self.num_agents):
            if self.agents_reached_goal[i]:
                continue
            x_i, y_i, vx_i, vy_i = self.agent_states[i]
            # check obstacle
            if self.map_gen.is_obstacle(x_i, y_i):
                # bounce
                self.agent_states[i, 2] = -vx_i
                self.agent_states[i, 3] = -vy_i

            # check collisions with other agents
            for j in range(i+1, self.num_agents):
                x_j, y_j, vx_j, vy_j = self.agent_states[j]
                dist = np.linalg.norm([x_i - x_j, y_i - y_j])
                if dist < 1.0:  # collision threshold
                    # bounce them
                    # simplest approach: swap velocities or invert them
                    self.agent_states[i, 2] = -vx_i
                    self.agent_states[i, 3] = -vy_i
                    self.agent_states[j, 2] = -vx_j
                    self.agent_states[j, 3] = -vy_j

    def _compute_rewards(self, actions: np.ndarray) -> float:
        """
        Compute a scalar reward for the entire system. Or you can do
        per-agent rewards if you prefer. For now, do a single float reward.

        Terms:
          - + for moving in direction of goal
          - - for large action magnitude
          - - for being on high-cost terrain
        """
        reward = 0.0
        # parse actions
        actions = actions.reshape(self.num_agents, self.action_dim)

        for i in range(self.num_agents):
            if self.agents_reached_goal[i]:
                # no reward penalty if already at goal
                continue
            x, y, vx, vy = self.agent_states[i]
            gx, gy = self.agent_goals[i]
            # direction to goal
            to_goal = np.array([gx - x, gy - y])
            if np.linalg.norm(to_goal) > 1e-6:
                to_goal_norm = to_goal / np.linalg.norm(to_goal)
            else:
                to_goal_norm = np.zeros_like(to_goal)

            # actual velocity or commanded direction
            if self.robot_dynamics == "single_integrator":
                commanded_dir = actions[i] / (np.linalg.norm(actions[i]) + 1e-6)
            elif self.robot_dynamics == "unicycle":
                # direction of motion is from (v, omega)? Let's approximate with velocity
                v, omega = actions[i]
                # approximate velocity vector from unicycle dynamics
                commanded_dir = np.array([np.cos(omega), np.sin(omega)])  # rough
            else:
                # double integrator
                # we consider acceleration direction
                commanded_dir = actions[i] / (np.linalg.norm(actions[i]) + 1e-6)

            # alignment reward
            alignment = np.dot(to_goal_norm, commanded_dir)
            reward += 1.0 * alignment  # weight

            # cost for large action
            action_mag = np.linalg.norm(actions[i])
            reward -= 0.1 * action_mag**2

            # terrain cost
            terrain_cost = self.map_gen.get_cost(x, y)
            reward -= 0.05 * terrain_cost

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Return either a centralized or decentralized observation.
        For a single-agent RL interface, we can flatten everything
        into one array. For multi-agent frameworks, you'd typically
        return a list of agent-specific observations. We'll flatten
        for simplicity.
        """
        if self.centralized:
            # shape = (num_agents * 6,)
            # each agent: (x, y, vx, vy, gx, gy)
            obs_list = []
            for i in range(self.num_agents):
                x, y, vx, vy = self.agent_states[i]
                gx, gy = self.agent_goals[i]
                obs_list.extend([x, y, vx, vy, gx, gy])
            return np.array(obs_list, dtype=np.float32)
        else:
            # Decentralized. Build each agent's partial obs and then flatten all.
            # Each agent sees [x, y, vx, vy, gx, gy] + other agents' (x, y, vx, vy) if within comm range.
            all_obs = []
            for i in range(self.num_agents):
                # self info
                x_i, y_i, vx_i, vy_i = self.agent_states[i]
                gx_i, gy_i = self.agent_goals[i]
                agent_obs = [x_i, y_i, vx_i, vy_i, gx_i, gy_i]

                # others if in comm range
                for j in range(self.num_agents):
                    if j == i:
                        continue
                    x_j, y_j, vx_j, vy_j = self.agent_states[j]
                    dist = np.linalg.norm([x_i - x_j, y_i - y_j])
                    if dist <= self.obs_range:
                        agent_obs.extend([x_j, y_j, vx_j, vy_j])
                    else:
                        # Could pad with something or just skip
                        pass

                # Pad to maximum possible dimension for a single agent
                # Max is (self) 6 + (num_agents-1)*4
                # We'll do that for a uniform shape
                max_len = 6 + (self.num_agents - 1) * 4
                while len(agent_obs) < max_len:
                    agent_obs.append(0.0)

                all_obs.extend(agent_obs)

            return np.array(all_obs, dtype=np.float32)

    def render(self):
        """
        Render the environment in 2D with matplotlib.
        We'll plot the obstacle map, the agents, the trails,
        and goals. Also label collisions if needed.
        """
        if self.fig is None or self.ax is None:
            self._init_render()

        self.ax.clear()
        cost_map = self.map_gen.cost_map
        self.ax.imshow(cost_map, origin='lower', cmap='gray',
                    extent=[0, self.width, 0, self.height])

        for i in range(self.num_agents):
            x, y, vx, vy = self.agent_states[i]
            gx, gy = self.agent_goals[i]
            self.ax.plot(x, y, 'bo', markersize=5)
            self.ax.plot(gx, gy, 'rx', markersize=7)
            self.ax.arrow(x, y, vx*0.5, vy*0.5, head_width=0.2, color='blue')
            if self.agents_reached_goal[i]:
                self.ax.text(x, y+0.5, f"Agent {i} reached", color='green')

        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_title(f"Step: {self.step_count}")
        self.ax.set_aspect('equal', 'box')

        self.fig.canvas.draw()  # ensure it's fully rendered
        w, h = self.fig.canvas.get_width_height()
        buf = self.fig.canvas.buffer_rgba()       # <--- CHANGED
        image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

        # If you only need RGB:
        # image = image[..., :3]

        # Store image for creating video later
        self.ims.append([plt.imshow(image, animated=True)])

        plt.pause(0.001)


    def _init_render(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ims = []

    def _save_video(self):
        """
        Save the recorded frames as an .mp4 using matplotlib.animation.
        """
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=100, blit=True)
        writer = animation.FFMpegWriter(fps=self.metadata["render_fps"])
        ani.save(self.video_filename, writer=writer)
        print(f"Video saved to {self.video_filename}")

    def close(self):
        """
        Close the environment.
        """
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
