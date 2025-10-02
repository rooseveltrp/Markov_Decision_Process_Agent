import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional

class MazeVisualizer:
    def __init__(self, maze_size: int = 10):
        self.maze_size = maze_size
        self.cell_size = 50  # Size of each cell in pixels
        
        # Load and resize images
        self.rat_img = self._load_and_resize_image("Rat.png", (40, 40))
        self.cheese_img = self._load_and_resize_image("Cheese.png", (40, 40))
        
        # Colors for visualization
        self.colors = {
            'wall': '#8B4513',      # Brown
            'free': '#F5F5DC',      # Beige
            'path': '#90EE90',      # Light green
            'visited': '#FFE4E1',   # Misty rose
            'grid': '#D3D3D3'       # Light gray
        }
    
    def _load_and_resize_image(self, filepath: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
        try:
            if os.path.exists(filepath):
                img = Image.open(filepath).convert('RGBA')
                img = img.resize(size, Image.Resampling.LANCZOS)
                return np.array(img)
            else:
                print(f"Warning: {filepath} not found")
                return None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def create_maze_figure(self, maze: np.ndarray, figsize: Tuple[int, int] = (12, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw maze cells
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                color = self.colors['wall'] if maze[i, j] == 1 else self.colors['free']
                rect = Rectangle((j, self.maze_size - 1 - i), 1, 1, 
                               facecolor=color, edgecolor=self.colors['grid'], linewidth=1)
                ax.add_patch(rect)
        
        # Set up the plot
        ax.set_xlim(0, self.maze_size)
        ax.set_ylim(0, self.maze_size)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.maze_size + 1))
        ax.set_yticks(range(self.maze_size + 1))
        ax.grid(True, color=self.colors['grid'], alpha=0.3)
        
        return fig, ax
    
    def add_entities(self, ax, rat_pos: Tuple[int, int], cheese_pos: Tuple[int, int]):
        # Add rat
        if self.rat_img is not None:
            rat_x = rat_pos[1] + 0.1
            rat_y = self.maze_size - 1 - rat_pos[0] + 0.1
            ax.imshow(self.rat_img, extent=[rat_x, rat_x + 0.8, rat_y, rat_y + 0.8], zorder=10)
        else:
            # Fallback to text if image not available
            ax.text(rat_pos[1] + 0.5, self.maze_size - 1 - rat_pos[0] + 0.5, 'R', 
                   fontsize=20, ha='center', va='center', color='red', weight='bold')
        
        # Add cheese
        if self.cheese_img is not None:
            cheese_x = cheese_pos[1] + 0.1
            cheese_y = self.maze_size - 1 - cheese_pos[0] + 0.1
            ax.imshow(self.cheese_img, extent=[cheese_x, cheese_x + 0.8, cheese_y, cheese_y + 0.8], zorder=10)
        else:
            # Fallback to text if image not available
            ax.text(cheese_pos[1] + 0.5, self.maze_size - 1 - cheese_pos[0] + 0.5, 'C', 
                   fontsize=20, ha='center', va='center', color='orange', weight='bold')
    
    def add_path(self, ax, path: List[Tuple[int, int]], alpha: float = 0.7):
        for i, pos in enumerate(path):
            # Color intensity based on position in path (earlier = lighter)
            intensity = 0.3 + 0.7 * (i / len(path)) if len(path) > 1 else 1.0
            color_alpha = alpha * intensity
            
            rect = Rectangle((pos[1], self.maze_size - 1 - pos[0]), 1, 1, 
                           facecolor=self.colors['path'], alpha=color_alpha, zorder=5)
            ax.add_patch(rect)
        
        # Draw path as connected line
        if len(path) > 1:
            path_x = [pos[1] + 0.5 for pos in path]
            path_y = [self.maze_size - 1 - pos[0] + 0.5 for pos in path]
            ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, zorder=6)
            ax.plot(path_x, path_y, 'bo', markersize=6, alpha=0.8, zorder=7)
    
    def visualize_episode(self, env_state: Dict, path: List[Tuple[int, int]], 
                         title: str = "Maze Solution", save_path: Optional[str] = None):
        fig, ax = self.create_maze_figure(env_state['maze'])
        
        # Add path first (so it's behind entities)
        if path:
            self.add_path(ax, path)
        
        # Add rat and cheese
        self.add_entities(ax, env_state['rat_pos'], env_state['cheese_pos'])
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def create_training_progress_plot(self, rewards: List[float], steps: List[int], 
                                    window: int = 100, save_path: Optional[str] = None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        episodes = range(1, len(rewards) + 1)
        
        # Plot rewards
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        # Moving average
        if len(rewards) >= window:
            moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
            ax1.plot(episodes, moving_avg, color='red', linewidth=2, label=f'Moving Average ({window})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress - Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot steps
        ax2.plot(episodes, steps, alpha=0.3, color='green', label='Episode Steps')
        
        # Moving average for steps
        if len(steps) >= window:
            moving_avg_steps = [np.mean(steps[max(0, i-window):i+1]) for i in range(len(steps))]
            ax2.plot(episodes, moving_avg_steps, color='orange', linewidth=2, label=f'Moving Average ({window})')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Training Progress - Steps to Completion')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def create_journey_timeline(self, path: List[Tuple[int, int]], maze: np.ndarray, 
                               rat_pos: Tuple[int, int], cheese_pos: Tuple[int, int],
                               steps_per_frame: int = 3, save_path: Optional[str] = None):
        num_frames = min(8, max(1, len(path) // steps_per_frame))
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for frame in range(num_frames):
            ax = axes[frame]
            
            # Calculate which part of the path to show
            end_step = min(len(path), (frame + 1) * steps_per_frame)
            current_path = path[:end_step]
            
            # Draw maze
            for i in range(self.maze_size):
                for j in range(self.maze_size):
                    color = self.colors['wall'] if maze[i, j] == 1 else self.colors['free']
                    rect = Rectangle((j, self.maze_size - 1 - i), 1, 1, 
                                   facecolor=color, edgecolor=self.colors['grid'], linewidth=0.5)
                    ax.add_patch(rect)
            
            # Add path up to current frame
            if current_path:
                self.add_path(ax, current_path, alpha=0.6)
                
                # Highlight current position
                if current_path:
                    current_pos = current_path[-1]
                    highlight = Rectangle((current_pos[1], self.maze_size - 1 - current_pos[0]), 1, 1, 
                                        facecolor='red', alpha=0.5, zorder=8)
                    ax.add_patch(highlight)
            
            # Add entities
            self.add_entities(ax, rat_pos if not current_path else current_path[-1], cheese_pos)
            
            ax.set_xlim(0, self.maze_size)
            ax.set_ylim(0, self.maze_size)
            ax.set_aspect('equal')
            ax.set_title(f'Step {end_step}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(num_frames, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle("Rat's Journey Through the Maze", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, axes
    
    def create_summary_visualization(self, env_state: Dict, path: List[Tuple[int, int]], 
                                   agent_stats: Dict, save_path: Optional[str] = None):
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.2)
        
        # Main maze visualization
        ax_maze = fig.add_subplot(gs[0, :])
        
        # Draw maze
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                color = self.colors['wall'] if env_state['maze'][i, j] == 1 else self.colors['free']
                rect = Rectangle((j, self.maze_size - 1 - i), 1, 1, 
                               facecolor=color, edgecolor=self.colors['grid'], linewidth=1)
                ax_maze.add_patch(rect)
        
        # Add path and entities
        if path:
            self.add_path(ax_maze, path)
        self.add_entities(ax_maze, env_state['rat_pos'], env_state['cheese_pos'])
        
        ax_maze.set_xlim(0, self.maze_size)
        ax_maze.set_ylim(0, self.maze_size)
        ax_maze.set_aspect('equal')
        ax_maze.set_title('Final Solution Path', fontsize=16, fontweight='bold')
        ax_maze.set_xlabel('Column')
        ax_maze.set_ylabel('Row')
        
        # Statistics text
        ax_stats = fig.add_subplot(gs[1, 0])
        ax_stats.axis('off')
        
        stats_text = f"""
        Training Statistics:
        • Total Episodes: {agent_stats.get('total_episodes', 'N/A')}
        • Average Reward (Last 100): {agent_stats.get('average_reward_last_100', 0):.2f}
        • Average Steps (Last 100): {agent_stats.get('average_steps_last_100', 0):.1f}
        • Best Reward: {agent_stats.get('best_reward', 'N/A')}
        • Q-Table Size: {agent_stats.get('q_table_size', 'N/A')} states
        • Final Epsilon: {agent_stats.get('current_epsilon', 0):.3f}
        """
        
        ax_stats.text(0.1, 0.9, stats_text, fontsize=12, verticalalignment='top', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        # Solution stats
        ax_solution = fig.add_subplot(gs[1, 1])
        ax_solution.axis('off')
        
        path_length = len(path) if path else 0
        manhattan_distance = abs(env_state['rat_pos'][0] - env_state['cheese_pos'][0]) + \
                           abs(env_state['rat_pos'][1] - env_state['cheese_pos'][1])
        efficiency = manhattan_distance / path_length if path_length > 0 else 0
        
        solution_text = f"""
        Solution Statistics:
        • Path Length: {path_length} steps
        • Manhattan Distance: {manhattan_distance}
        • Path Efficiency: {efficiency:.2f}
        • Success: {'Yes' if path and path[-1] == env_state['cheese_pos'] else 'No'}
        • Start: {env_state['rat_pos']}
        • Goal: {env_state['cheese_pos']}
        """
        
        ax_solution.text(0.1, 0.9, solution_text, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # Journey timeline (mini version)
        ax_timeline = fig.add_subplot(gs[2, :])
        if path and len(path) > 1:
            # Show key positions in the journey
            key_positions = [0, len(path)//4, len(path)//2, 3*len(path)//4, len(path)-1]
            key_positions = [pos for pos in key_positions if pos < len(path)]
            
            for i, pos_idx in enumerate(key_positions):
                x_offset = i * (self.maze_size + 1)
                pos = path[pos_idx]
                
                # Mini maze for this position
                for row in range(self.maze_size):
                    for col in range(self.maze_size):
                        color = self.colors['wall'] if env_state['maze'][row, col] == 1 else self.colors['free']
                        if (row, col) == pos:
                            color = self.colors['path']
                        
                        rect = Rectangle((x_offset + col*0.8, (self.maze_size-1-row)*0.8), 0.8, 0.8, 
                                       facecolor=color, edgecolor='black', linewidth=0.3)
                        ax_timeline.add_patch(rect)
                
                ax_timeline.text(x_offset + self.maze_size*0.4, -1, f'Step {pos_idx+1}', 
                               ha='center', fontsize=10)
        
        ax_timeline.set_xlim(-1, (len(key_positions) if 'key_positions' in locals() else 1) * (self.maze_size + 1))
        ax_timeline.set_ylim(-2, self.maze_size * 0.8)
        ax_timeline.set_aspect('equal')
        ax_timeline.set_title('Journey Progress', fontsize=14)
        ax_timeline.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig