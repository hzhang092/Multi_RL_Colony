# experiments/ppo_eval.py
"""
PPO Agent Evaluation and Colony Growth Visualization Script

This script loads a trained PPO agent from a checkpoint file and runs it in the
colony environment to visualize how the trained policy performs. It provides
real-time visualization of colony growth dynamics.

Features:
- Load trained PPO agent from .pt checkpoint files
- Real-time colony growth visualization
- Configurable evaluation parameters
- Performance metrics tracking
- Option for deterministic or stochastic evaluation
- Support for saving visualization frames

Usage:
    python experiments/ppo_eval.py --checkpoint path/to/model.pt --render --max_steps 500
    
    # Or run with latest checkpoint automatically:
    python experiments/ppo_eval.py

Requirements:
    - torch
    - numpy
    - matplotlib
    - Colony environment (envs.colony_env)
    - PPO agent (agents.ppo_agent)

Author: Multi-Agent RL Colony Project
Date: 2024
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import time

import sys
# Add the project root to the Python path to allow for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from envs.colony_env import ColonyEnv
from agents.ppo_agent import PPOAgent, make_action_dicts


def load_agent(checkpoint_path: str, obs_dim: int, device: torch.device) -> PPOAgent:
    """
    Load a trained PPO agent from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the .pt checkpoint file
        obs_dim (int): Observation dimension of the environment
        device (torch.device): Device to load the model on
    
    Returns:
        PPOAgent: Loaded and initialized PPO agent
    """
    agent = PPOAgent(obs_dim=obs_dim, device=device)
    
    # Handle both full checkpoint and state_dict only files
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
        # Full checkpoint with metadata
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"✅ Loaded full checkpoint from: {checkpoint_path}")
        if 'update' in checkpoint:
            print(f"   - Training update: {checkpoint['update']}")
        if 'total_steps' in checkpoint:
            print(f"   - Total training steps: {checkpoint['total_steps']:,}")
        if 'best_reward' in checkpoint:
            print(f"   - Best reward achieved: {checkpoint['best_reward']:.3f}")
    else:
        # State dict only
        agent.policy.load_state_dict(checkpoint)
        print(f"✅ Loaded policy state dict from: {checkpoint_path}")
    
    agent.policy.eval()
    return agent


def run_evaluation(checkpoint_path: str, 
                  render: bool = True, 
                  max_steps: int = 300,
                  deterministic: bool = True,
                  save_frames: bool = False,
                  frame_interval: int = 5,
                  env_seed: int = 686,
                  figsize: tuple = (8, 8)):
    """
    Run a single rollout of the trained PPO agent with visualization.
    
    Args:
        checkpoint_path (str): Path to the trained model checkpoint
        render (bool): Whether to display real-time visualization
        max_steps (int): Maximum number of environment steps
        deterministic (bool): Use deterministic (greedy) actions vs stochastic sampling
        save_frames (bool): Whether to save visualization frames to disk
        frame_interval (int): Interval between rendered frames
        env_seed (int): Random seed for environment initialization
        figsize (tuple): Figure size for visualization
    
    Returns:
        dict: Evaluation results and metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # Initialize environment
    env = ColonyEnv(seed=env_seed)
    obs, _ = env.reset()
    obs_dim = obs.shape[1]
    print(f"Environment initialized with observation dimension: {obs_dim}")
    
    # Load trained agent
    agent = load_agent(checkpoint_path, obs_dim, device)
    
    # Tracking variables
    total_reward = 0.0
    step_rewards = []
    num_cells_history = []
    step = 0  # Initialize step counter
    terminated = False  # Initialize termination flags
    truncated = False
    ax = None  # Initialize ax variable
    
    if render:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=figsize)
        plt.title("PPO Agent - Colony Growth Visualization")
    
    frames_dir = None  # Initialize frames_dir
    if save_frames:
        frames_dir = Path("evaluation_frames")
        frames_dir.mkdir(exist_ok=True)
        print(f"Saving frames to: {frames_dir}")
    
    print(f"Starting evaluation rollout (max {max_steps} steps)...")
    print("=" * 60)
    
    for step in range(max_steps):
        # Prepare observation tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        
        # Sample actions from the policy
        with torch.no_grad():
            if deterministic:
                # Use deterministic (greedy) actions
                logits, param_mean, param_std, values = agent.policy(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                sampled_type = torch.argmax(probs, dim=-1)
                sampled_params = param_mean  # Use mean instead of sampling
            else:
                # Use stochastic sampling (as during training)
                sampled_type, sampled_params, logp, values = agent.act(obs_tensor)
        
        # Convert to environment-compatible actions
        actions = make_action_dicts(sampled_type, sampled_params)
        
        # Step the environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Track metrics
        step_reward = np.mean(rewards) if len(rewards) > 0 else 0.0
        total_reward += step_reward
        step_rewards.append(step_reward)
        num_cells_history.append(len(obs))
        
        # Update observation
        obs = next_obs
        
        # Render visualization
        if render and step % frame_interval == 0 and ax is not None:
            try:
                # Use the fixed env.render() method
                ax.clear()
                
                # Get RGB array from environment
                rgb_array = env.render(mode="rgb_array")
                
                if rgb_array is not None:
                    # Display the rendered image
                    ax.imshow(rgb_array, origin='upper')
                    ax.set_title(f"Colony Growth - Step {step} | Cells: {len(obs)} | Reward: {step_reward:.3f}")
                    ax.axis('off')  # Turn off axis for cleaner look
                else:
                    # Fallback to manual drawing if render returns None
                    ax.set_xlim(0, env.world_size[0])
                    ax.set_ylim(0, env.world_size[1])
                    ax.set_aspect('equal', 'box')
                    ax.set_title(f"Colony Growth - Step {step} | Cells: {len(obs)} | Reward: {step_reward:.3f}")
                
                # Save frame if requested
                if save_frames and frames_dir is not None:
                    frame_path = frames_dir / f"frame_{step:04d}.png"
                    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                
                plt.draw()
                plt.pause(0.1)  # Pause for animation effect
                
            except Exception as render_error:
                print(f"Warning: Rendering failed at step {step}: {render_error}")
                # Continue without rendering for this step
        
        # Print progress
        if step % 20 == 0 or step < 10:
            print(f"Step {step:3d} | Cells: {len(obs):2d} | "
                  f"Step Reward: {step_reward:6.3f} | "
                  f"Total Reward: {total_reward:6.3f}")
        
        # Check if episode ended
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            if terminated:
                print("Reason: Environment terminated (goal reached or failure)")
            if truncated:
                print("Reason: Episode truncated (time limit or other constraint)")
            break
    
    # Final statistics
    print("=" * 60)
    print("📊 EVALUATION RESULTS:")
    print(f"   Total steps: {step + 1}")
    print(f"   Total reward: {total_reward:.3f}")
    print(f"   Average reward per step: {total_reward/(step+1):.3f}")
    print(f"   Final number of cells: {len(obs)}")
    print(f"   Max cells reached: {max(num_cells_history)}")
    print("=" * 60)
    
    if render and ax is not None:
        try:
            # Show final state using the fixed env.render() method
            rgb_array = env.render(mode="rgb_array")
            
            if rgb_array is not None:
                ax.clear()
                ax.imshow(rgb_array, origin='upper')
                ax.set_title(f"Final Colony State - {len(obs)} cells, Total Reward: {total_reward:.3f}")
                ax.axis('off')
            else:
                # Fallback if render returns None
                ax.clear()
                ax.text(0.5, 0.5, f"Final Colony: {len(obs)} cells\nTotal Reward: {total_reward:.3f}", 
                       transform=ax.transAxes, ha='center', va='center', fontsize=14)
                ax.set_title("Colony Evaluation Complete")
            
            plt.ioff()
            plt.show()
        except Exception as e:
            print(f"Warning: Final visualization failed: {e}")
            print("Colony evaluation completed successfully, but visualization had issues.")
    
    # Return evaluation results
    results = {
        'total_reward': total_reward,
        'num_steps': step + 1,
        'avg_reward_per_step': total_reward / (step + 1),
        'final_num_cells': len(obs),
        'max_cells': max(num_cells_history),
        'step_rewards': step_rewards,
        'num_cells_history': num_cells_history,
        'terminated': terminated,
        'truncated': truncated
    }
    
    return results


def find_latest_checkpoint(checkpoint_dir: str = "ppo_checkpoints") -> str:
    """
    Find the latest checkpoint file in the specified directory.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint files
    
    Returns:
        str: Path to the latest checkpoint file
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for checkpoint files
    ckpt_patterns = ["ppo_colony_*.pt", "ppo_ckpt_*.pt", "*.pt"]
    checkpoint_files = []
    
    for pattern in ckpt_patterns:
        checkpoint_files.extend(list(checkpoint_path.glob(pattern)))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by modification time and return the latest
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return str(latest_checkpoint)


def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO agent and visualize colony growth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to checkpoint file. If not provided, uses latest from ppo_checkpoints/')
    parser.add_argument('--max_steps', '-s', type=int, default=300,
                        help='Maximum number of evaluation steps')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable real-time visualization')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions instead of deterministic')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save visualization frames to disk')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='Interval between rendered frames')
    parser.add_argument('--seed', type=int, default=686,
                        help='Random seed for environment')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint is None:
        try:
            checkpoint_path = find_latest_checkpoint("ppo_checkpoints_original")
            print(f"🔍 Using latest checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            return
    else:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
            return
    
    # Run evaluation
    try:
        results = run_evaluation(
            checkpoint_path=checkpoint_path,
            render=not args.no_render,
            max_steps=args.max_steps,
            deterministic=not args.stochastic,
            save_frames=args.save_frames,
            frame_interval=args.frame_interval,
            env_seed=args.seed
        )
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
