import numpy as np
import random
from typing import Tuple, List, Dict
import pickle

class QLearningAgent:
    def __init__(
        self, 
        maze_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.maze_size = maze_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action values
        # State is represented as (rat_row, rat_col, cheese_row, cheese_col)
        self.q_table = {}
        self.action_space_size = 4  # up, right, down, left
        
        # For tracking learning progress
        self.episode_rewards = []
        self.episode_steps = []
        
    def _get_state_key(self, state: np.ndarray) -> Tuple[int, int, int, int]:
        return tuple(state.astype(int))
    
    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        return self.q_table[state_key]
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_space_size - 1)
        else:
            # Exploitation: best action according to Q-table
            q_values = self._get_q_values(state)
            return np.argmax(q_values)
    
    def update_q_table(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        terminated: bool
    ):
        current_q_values = self._get_q_values(state)
        
        if terminated:
            target = reward
        else:
            next_q_values = self._get_q_values(next_state)
            target = reward + self.discount_factor * np.max(next_q_values)
        
        # Q-learning update rule
        current_q_values[action] += self.learning_rate * (target - current_q_values[action])
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = self.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            self.update_q_table(state, action, reward, next_state, terminated)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        self.decay_epsilon()
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        
        return total_reward, steps
    
    def evaluate_episode(self, env, max_steps: int = 1000) -> Tuple[float, int, List[Tuple[int, int]]]:
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        path = [(env.rat_pos[0], env.rat_pos[1])]  # Track the path taken
        
        for step in range(max_steps):
            action = self.choose_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            path.append((env.rat_pos[0], env.rat_pos[1]))
            total_reward += reward
            steps += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        return total_reward, steps, path
    
    def get_policy(self, state: np.ndarray) -> Dict[int, float]:
        q_values = self._get_q_values(state)
        # Convert Q-values to policy probabilities (softmax-like)
        exp_q = np.exp(q_values - np.max(q_values))  # Numerical stability
        probabilities = exp_q / np.sum(exp_q)
        
        return {action: prob for action, prob in enumerate(probabilities)}
    
    def save_model(self, filepath: str):
        model_data = {
            'q_table': self.q_table,
            'maze_size': self.maze_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.maze_size = model_data['maze_size']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.epsilon = model_data['epsilon']
        self.epsilon_decay = model_data['epsilon_decay']
        self.epsilon_min = model_data['epsilon_min']
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.episode_steps = model_data.get('episode_steps', [])
    
    def get_stats(self) -> Dict:
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        recent_steps = self.episode_steps[-100:] if len(self.episode_steps) >= 100 else self.episode_steps
        
        return {
            'total_episodes': len(self.episode_rewards),
            'average_reward_last_100': np.mean(recent_rewards),
            'average_steps_last_100': np.mean(recent_steps),
            'best_reward': np.max(self.episode_rewards),
            'current_epsilon': self.epsilon,
            'q_table_size': len(self.q_table)
        }