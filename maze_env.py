import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple
import random

class MazeEnv(gym.Env):
    def __init__(self, maze_size: int = 10, render_mode: Optional[str] = None):
        super(MazeEnv, self).__init__()
        
        self.maze_size = maze_size
        self.render_mode = render_mode
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: (rat_row, rat_col, cheese_row, cheese_col)
        self.observation_space = spaces.Box(
            low=0, high=maze_size-1, shape=(4,), dtype=np.int32
        )
        
        # Action mappings
        self.action_to_direction = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }
        
        self.maze = None
        self.rat_pos = None
        self.cheese_pos = None
        self.steps = 0
        self.max_steps = maze_size * maze_size * 2
        
    def _generate_maze(self):
        # Create a maze with walls (1) and free spaces (0)
        maze = np.zeros((self.maze_size, self.maze_size), dtype=np.int32)
        
        # Add some random walls (about 20% of the maze)
        wall_probability = 0.2
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if random.random() < wall_probability:
                    maze[i, j] = 1
        
        # Ensure start and end positions are free
        maze[0, 0] = 0  # Starting position
        maze[self.maze_size-1, self.maze_size-1] = 0  # Cheese position
        
        # Create a simple path to ensure solvability
        for i in range(self.maze_size):
            maze[i, min(i, self.maze_size-1)] = 0
            
        return maze
    
    def _get_valid_positions(self):
        valid_positions = []
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if self.maze[i, j] == 0:  # Free space
                    valid_positions.append((i, j))
        return valid_positions
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Generate new maze
        self.maze = self._generate_maze()
        
        # Get valid positions
        valid_positions = self._get_valid_positions()
        
        # Place rat and cheese randomly on valid positions
        positions = random.sample(valid_positions, 2)
        self.rat_pos = positions[0]
        self.cheese_pos = positions[1]
        
        # Ensure they're not in the same position
        while self.rat_pos == self.cheese_pos:
            positions = random.sample(valid_positions, 2)
            self.rat_pos = positions[0]
            self.cheese_pos = positions[1]
        
        self.steps = 0
        
        observation = np.array([
            self.rat_pos[0], self.rat_pos[1],
            self.cheese_pos[0], self.cheese_pos[1]
        ], dtype=np.int32)
        
        info = {"steps": self.steps}
        
        return observation, info
    
    def step(self, action: int):
        self.steps += 1
        
        # Get movement direction
        direction = self.action_to_direction[action]
        new_row = self.rat_pos[0] + direction[0]
        new_col = self.rat_pos[1] + direction[1]
        
        # Check if move is valid (within bounds and not a wall)
        if (0 <= new_row < self.maze_size and 
            0 <= new_col < self.maze_size and 
            self.maze[new_row, new_col] == 0):
            self.rat_pos = (new_row, new_col)
        
        # Calculate reward
        reward = 0
        terminated = False
        
        # Check if rat found the cheese
        if self.rat_pos == self.cheese_pos:
            reward = 100
            terminated = True
        else:
            # Small negative reward for each step to encourage efficiency
            reward = -1
            
            # Additional reward based on distance to cheese (to guide the agent)
            distance = abs(self.rat_pos[0] - self.cheese_pos[0]) + abs(self.rat_pos[1] - self.cheese_pos[1])
            reward += -0.1 * distance
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            terminated = True
            reward = -50  # Penalty for not finding cheese in time
        
        observation = np.array([
            self.rat_pos[0], self.rat_pos[1],
            self.cheese_pos[0], self.cheese_pos[1]
        ], dtype=np.int32)
        
        info = {
            "steps": self.steps,
            "distance_to_cheese": abs(self.rat_pos[0] - self.cheese_pos[0]) + abs(self.rat_pos[1] - self.cheese_pos[1])
        }
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "human":
            # Create a visual representation of the maze
            visual_maze = np.copy(self.maze).astype(str)
            visual_maze = np.where(visual_maze == '0', '.', visual_maze)  # Free space
            visual_maze = np.where(visual_maze == '1', '#', visual_maze)  # Wall
            
            # Place rat and cheese
            visual_maze[self.rat_pos[0], self.rat_pos[1]] = 'R'
            visual_maze[self.cheese_pos[0], self.cheese_pos[1]] = 'C'
            
            print(f"\nStep {self.steps}:")
            for row in visual_maze:
                print(' '.join(row))
            print()
            
    def get_state(self):
        return {
            "maze": self.maze.copy(),
            "rat_pos": self.rat_pos,
            "cheese_pos": self.cheese_pos,
            "steps": self.steps
        }