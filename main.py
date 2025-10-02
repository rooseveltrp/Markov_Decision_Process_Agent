#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from q_learning_agent import QLearningAgent
from visualizer import MazeVisualizer
import argparse
import time

def train_agent(maze_size=10, episodes=1000, save_model=True):
    print(f"Training Q-Learning Agent on {maze_size}x{maze_size} maze for {episodes} episodes...")
    
    # Create environment and agent
    env = MazeEnv(maze_size=maze_size, render_mode=None)
    agent = QLearningAgent(
        maze_size=maze_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Training loop
    start_time = time.time()
    best_reward = float('-inf')
    
    for episode in range(episodes):
        reward, steps = agent.train_episode(env)
        
        if reward > best_reward:
            best_reward = reward
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            stats = agent.get_stats()
            print(f"Episode {episode + 1}/{episodes} - "
                  f"Avg Reward: {stats['average_reward_last_100']:.2f} - "
                  f"Avg Steps: {stats['average_steps_last_100']:.1f} - "
                  f"Epsilon: {stats['current_epsilon']:.3f} - "
                  f"Best: {best_reward:.1f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    if save_model:
        model_path = f"maze_agent_{maze_size}x{maze_size}.pkl"
        agent.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    return env, agent

def evaluate_agent(env, agent, num_episodes=10):
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    total_rewards = []
    total_steps = []
    successful_episodes = 0
    best_path = None
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        reward, steps, path = agent.evaluate_episode(env)
        total_rewards.append(reward)
        total_steps.append(steps)
        
        if reward > 0:  # Successful episode (found cheese)
            successful_episodes += 1
            
        if reward > best_reward:
            best_reward = reward
            best_path = path
            best_env_state = env.get_state()
    
    success_rate = successful_episodes / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    
    print(f"Evaluation Results:")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"  Best Reward: {best_reward:.1f}")
    
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'best_reward': best_reward,
        'best_path': best_path,
        'best_env_state': best_env_state
    }

def create_visualizations(env, agent, eval_results):
    print("Creating visualizations...")
    
    visualizer = MazeVisualizer(maze_size=env.maze_size)
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. Training progress plot
    print("  - Training progress visualization...")
    fig_progress, _ = visualizer.create_training_progress_plot(
        agent.episode_rewards, 
        agent.episode_steps,
        save_path="visualizations/training_progress.png"
    )
    plt.close(fig_progress)
    
    # 2. Best solution path
    if eval_results['best_path']:
        print("  - Best solution path visualization...")
        fig_solution, _ = visualizer.visualize_episode(
            eval_results['best_env_state'],
            eval_results['best_path'],
            title=f"Best Solution - Reward: {eval_results['best_reward']:.1f}",
            save_path="visualizations/best_solution.png"
        )
        plt.close(fig_solution)
        
        # 3. Journey timeline
        print("  - Journey timeline visualization...")
        fig_timeline, _ = visualizer.create_journey_timeline(
            eval_results['best_path'],
            eval_results['best_env_state']['maze'],
            eval_results['best_env_state']['rat_pos'],
            eval_results['best_env_state']['cheese_pos'],
            save_path="visualizations/journey_timeline.png"
        )
        plt.close(fig_timeline)
        
        # 4. Summary visualization
        print("  - Summary visualization...")
        agent_stats = agent.get_stats()
        fig_summary = visualizer.create_summary_visualization(
            eval_results['best_env_state'],
            eval_results['best_path'],
            agent_stats,
            save_path="visualizations/summary.png"
        )
        plt.close(fig_summary)
    
    print("All visualizations saved to 'visualizations/' directory")

def main():
    parser = argparse.ArgumentParser(description="Maze Q-Learning Agent")
    parser.add_argument("--maze-size", type=int, default=10, help="Size of the maze (default: 10)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes (default: 1000)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--load-model", type=str, help="Path to load a pre-trained model")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (requires --load-model)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MAZE Q-LEARNING AGENT")
    print("=" * 60)
    
    # Create environment
    env = MazeEnv(maze_size=args.maze_size, render_mode=None)
    
    if args.load_model:
        # Load pre-trained agent
        print(f"Loading pre-trained model from {args.load_model}")
        agent = QLearningAgent(maze_size=args.maze_size)
        agent.load_model(args.load_model)
        print("Model loaded successfully")
    else:
        agent = None
    
    if not args.skip_training:
        # Train the agent
        env, agent = train_agent(
            maze_size=args.maze_size,
            episodes=args.episodes,
            save_model=True
        )
        print()
    elif not agent:
        print("Error: --skip-training requires --load-model")
        return
    
    # Evaluate the agent
    eval_results = evaluate_agent(env, agent, num_episodes=args.eval_episodes)
    print()
    
    # Create visualizations
    if not args.no_viz:
        create_visualizations(env, agent, eval_results)
        print()
    
    # Show final statistics
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    agent_stats = agent.get_stats()
    print(f"Training Episodes: {agent_stats['total_episodes']}")
    print(f"Final Success Rate: {eval_results['success_rate']:.1%}")
    print(f"Average Reward: {eval_results['avg_reward']:.2f}")
    print(f"Best Reward Achieved: {eval_results['best_reward']:.1f}")
    print(f"Q-Table States: {agent_stats['q_table_size']}")
    
    if eval_results['best_path']:
        print(f"Best Path Length: {len(eval_results['best_path'])} steps")
        rat_pos = eval_results['best_env_state']['rat_pos']
        cheese_pos = eval_results['best_env_state']['cheese_pos']
        manhattan_distance = abs(rat_pos[0] - cheese_pos[0]) + abs(rat_pos[1] - cheese_pos[1])
        efficiency = manhattan_distance / len(eval_results['best_path'])
        print(f"Path Efficiency: {efficiency:.2f} (1.0 = optimal)")
    
    print()
    print("Check the 'visualizations/' directory for generated images!")
    print("=" * 60)

if __name__ == "__main__":
    main()