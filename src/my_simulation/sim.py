import numpy as np
import matplotlib.pyplot as plt
from my_simulation.animation import animate
from my_simulation.obstacle_hit import *
import torch

class Sim():
    def __init__(self, grid_size):
        self.cost_map = np.zeros([grid_size,grid_size])
        self.grid_size = grid_size
        self.distance = 0
        
        # Variables to modify
        self.obstacle_cost = 2
        self.direction_weight = 1
        self.distance_weight = 1
        self.collision_weight = 1

    def reset(self):
        self.position = self.start
        self.path = [self.position]

    def make_obstacles(self, num_obstacles,cost_map,obstacle_size,build_wall):
        
        # add obstacles until num_obstacles are added
        while num_obstacles != 0:
            
            # Randomly select coordinates for obstacles
            rand_idx = np.random.choice(cost_map.shape[0]**2)
            obs_center_x,obs_center_y = np.unravel_index(rand_idx, cost_map.shape)
            
            x_min = int(np.ceil(obs_center_x-obstacle_size/2))
            x_max = int(np.floor(obs_center_x+obstacle_size/2))
            y_min = int(np.ceil(obs_center_y-obstacle_size/2))
            y_max = int(np.floor(obs_center_y+obstacle_size/2))
            
            # Ensure obstacles are not going over sides
            if x_min < 0 or x_max > cost_map.shape[0] or y_min < 0 or y_max > cost_map.shape[0]:
                continue
            
            # Ensure the obstacles aren't placed on the start or end goal. Again, the axes are flipped and idk why
            if (self.start[0] < x_max and self.start[0] > x_min and self.start[1] < y_max and self.start[1] > y_min) or (self.goal[0] < x_max and self.goal[0] > x_min and self.goal[1] < y_max and self.goal[1] > y_min):
                continue
            
            # Assign extra cost to obstacle area
            cost_map[x_min:x_max,y_min:y_max] = self.obstacle_cost
            num_obstacles -= 1
            
        # Build a heavy wall in the middle of the path
        if build_wall == True:
            cost_map[round(self.grid_size/2)-3:round(self.grid_size/2)+3,round(self.grid_size/10):self.grid_size - round(self.grid_size/10)] = self.obstacle_cost
            
        self.cost_map = cost_map
        
    def draw_path(self,path):
        plt.plot(path[:,0],path[:,1],color='red')
        plt.xlim([0,self.grid_size])
        plt.ylim([0,self.grid_size])

    def set_goal(self, x, y):
        # Check if the goal is on the map
        assert (0 <= x < self.grid_size) and (0 <= y < self.grid_size), 'Goal is not on the map.'
        
        # Assign goal
        self.goal = (x, y)
        
    def set_start(self, x, y):
        # Check if the start is on the map
        assert (0 <= x < self.grid_size) and (0 <= y < self.grid_size), 'Start is not on the map.'
        
        # Assign the goal
        self.start = (x, y)

    def reward(self, position, former_position, control):
        
        # For some reason the costmap is flipped. Spent 2 hours trying to figure out the plotting.
        # position = position[[1,0]]
        # former_position = np.array(former_position)[[1,0]]

        # Convert to numpy array from tensor
        position = np.array(position)
        control = np.array(control)

        x = position[0][1]
        y = position[0][0]
        
        # Get distance cost
        self.distance_cost = np.linalg.norm(position - self.goal)  # linalg.norm is the euclidean distance
        
        # Collision cost case 2 precaclulations
        coords = np.round(np.linspace(former_position, position)).astype(int)
        x_coords = coords[:,0,1]
        y_coords = coords[:,0,0]
        
        # Case 1 where robot is out of bounds, punished more than normal
        if x < 0 or x >= self.cost_map.shape[0] or y < 0 or y >= self.cost_map.shape[1]:
            collision_cost = self.obstacle_cost * 10
            
        # Case 2 where an obstacle is in the path from one point to another, but not explicitly hit. This case will make the agent take corners and not jump over obstacles
        elif np.any(self.cost_map[x_coords,y_coords] != 0):
            collision_cost = self.obstacle_cost
            
        # Case 3 where an obstacle is hit
        elif ObstacleHit(self.cost_map, position):
            collision_cost = self.obstacle_cost
            
        # Case 4 where no obstacle is hit
        else:
            collision_cost = 0
        
        num = (self.goal - position) / np.linalg.norm(position - self.goal)
        den = control / np.linalg.norm(control)
        direction_cost = -np.dot(num, den.T)

        weighted_direction_cost = self.direction_weight * direction_cost

        weighted_distance_cost = self.distance_weight * self.distance_cost
        
        weighted_collision_cost = self.collision_weight * collision_cost
        
        return torch.tensor(weighted_distance_cost + weighted_collision_cost + weighted_direction_cost, dtype=torch.float32)

    def reached_goal(self, position):
        if abs(int(position[0,0]) - self.goal[0]) > 2 or abs(int(position[0,1]) - self.goal[1]) > 2:
            return False
        else:
            return True

    def generate_path(self):
        # MPPI implementation
        position = self.start
        path = [position]
        control_sequence = np.zeros((self.monte_carlo_iters, self.prediction_horizon, 2)) # 2 is one control for each direction
        
        # Initialize annimation
        ann = animate(self.cost_map, self.start, self.goal)
        
        # While the robot is not within some tolerance of the goal position
        while abs(int(position[0]) - self.goal[0]) > 2 or abs(int(position[1]) - self.goal[1]) > 2:
            
            # Simulate the newest state of the robot
            ann.move(self.cost_map, position)
            
            # Random simulated noise generated before the loop
            noise = np.random.normal(0, 1, (self.monte_carlo_iters, self.prediction_horizon, 2))
            controls = control_sequence + noise # 5 monte carlos for 30 steps into the future
            
            # Build cost vector
            cost = np.array([0 for _ in range(self.monte_carlo_iters)])
            
            # Build a number of monte_carlo estimations
            for i in range(self.monte_carlo_iters):
                simulated_position = position
                ann.reset(position) # Reset the variables in the annimation

                # Over some prediction horizon
                for j in range(self.prediction_horizon):
                    
                    # Sample possible controls with noise
                    control = controls[i,j] 
                    
                    # Apply control to model to find simulated movement
                    former_simulated_position = simulated_position
                    simulated_position = self.robot_model(simulated_position, control)
                    
                    # Accumulate rollout cost over the prediction horizon for this monte carlo
                    cost[i] += self.cost_function(simulated_position, former_simulated_position, control)
                    
                    # Plot predictions
                    if self.draw_preds == True:
                        ann.predict(simulated_position, cost)
                    
            # weights cannot be too small thus add np.min(cost)
            weights = np.exp(((-cost)+np.min(cost)) / self.lambda_)
            total_weight = np.sum(weights)

            if total_weight == 0:  # Handle division by zero
                weights = np.ones_like(weights) / len(weights)  # Assign uniform weights
            else:
                weights /= total_weight
            # for i in range(prediction_horizon):
            control_sequence = np.sum(weights[:, None, None] * controls, axis=0)
                
            # Apply first control in the sequence
            position = self.robot_model(position, control_sequence[0])
            # self.check_stuck(position)
            path.append(position)
            control_sequence = np.roll(control_sequence, -1, axis=0)
            control_sequence[-1] = (0, 0)
    
        # Final simulated movement
        ann.move(self.cost_map, position)
        
        return np.array(path)