# experiments/ppo_train.py
"""
PPO Training Script for Multi-Agent Colony Environment

This script implements a complete training pipeline for Proximal Policy Optimization (PPO)
with the multi-agent ColonyEnv. It uses a shared policy approach where all agents in the
environment share the same neural network parameters.

Companion Files:
- agents/ppo_agent.py: Core PPO agent implementation used by this script
- experiments/ppo_shared_policy.py: Original monolithic implementation (deprecated)

Training Pipeline:
1. Environment initialization with fixed seed for reproducibility
2. PPO agent initialization with shared actor-critic network
3. Rollout collection: agents interact with environment collecting experience
4. PPO updates: policy and value function training using collected experience
5. Periodic checkpointing for model persistence

Key Features:
- Shared policy multi-agent training
- Experience collection with proper multi-agent handling
- GAE-based advantage estimation
- PPO clipped objective with entropy regularization
- Automatic model checkpointing
- GPU acceleration support

Hyperparameters:
- NUM_UPDATES: Total number of policy updates (2000)
- STEPS_PER_UPDATE: Environment steps per update (512)
- PPO_EPOCHS: Training epochs per update (4)
- MINIBATCH_SIZE: Batch size for PPO updates (4096)
- GAMMA: Discount factor (0.99)
- LAM: GAE lambda parameter (0.95)

Usage:
    python experiments/ppo_train.py

Requirements:
    - torch
    - numpy
    - Colony environment (envs.colony_env)
    - PPO agent (agents.ppo_agent)

Output:
    - Console logging of training progress
    - Model checkpoints saved to ppo_checkpoints/
    - Final trained model

"""

# Fix for OpenMP duplicate library warning (common with PyTorch + conda)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import numpy as np
import torch
from pathlib import Path

import sys
# Add the project root to the Python path to allow for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from envs.colony_env import ColonyEnv
from agents.ppo_agent import PPOAgent, RolloutBuffer, make_action_dicts

# =====================================================
# Training Hyperparameters and Configuration
# =====================================================

# Device configuration - automatically uses GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment configuration
ENV_SEED = 686                 # Fixed seed for reproducible experiments

# Training schedule
NUM_UPDATES = 500             # Total number of policy updates to perform
STEPS_PER_UPDATE = 100         # Environment steps collected per policy update
                               # Total training steps = NUM_UPDATES * STEPS_PER_UPDATE

# PPO algorithm parameters
PPO_EPOCHS = 4                 # Number of optimization epochs per collected batch
MINIBATCH_SIZE = 4096          # Batch size for gradient updates
                               # Should be <= total transitions per update

# Advantage estimation parameters
GAMMA = 0.99                   # Discount factor for future rewards
LAM = 0.95                     # GAE lambda parameter (bias-variance tradeoff)

# Checkpointing configuration
SAVE_DIR = Path("ppo_checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

# ---------------- Training Loop ----------------
def main():
    """
    Main training function implementing the complete PPO training loop.
    
    Training Process:
    1. Initialize environment and agent
    2. For each update iteration:
       a. Collect experience through environment interaction
       b. Compute advantages using GAE
       c. Perform PPO policy updates
       d. Log progress and save checkpoints
    3. Save final trained model
    
    The training uses a shared policy approach where all agents in the multi-agent
    environment share the same neural network parameters. This enables efficient
    learning in homogeneous multi-agent settings.
    
    Returns:
        None
    
    Raises:
        RuntimeError: If environment or agent initialization fails
        KeyboardInterrupt: Training can be safely interrupted and resumed from checkpoints
    """
    print(f"Starting PPO training on device: {DEVICE}")
    print(f"Training configuration:")
    print(f"  - Updates: {NUM_UPDATES}")
    print(f"  - Steps per update: {STEPS_PER_UPDATE}")
    print(f"  - Total training steps: {NUM_UPDATES * STEPS_PER_UPDATE:,}")
    print(f"  - PPO epochs: {PPO_EPOCHS}")
    print(f"  - Minibatch size: {MINIBATCH_SIZE}")
    print(f"  - Discount factor (gamma): {GAMMA}")
    print(f"  - GAE lambda: {LAM}")
    print("" + "="*50)
    
    # =====================================================
    # Environment and Agent Initialization
    # =====================================================
    env = ColonyEnv(seed=ENV_SEED)
    obs, _ = env.reset()
    obs_dim = obs.shape[1]  # Observation dimension per agent
    print(f"Environment initialized with observation dimension: {obs_dim}")
    
    # Initialize PPO agent with shared policy
    agent = PPOAgent(obs_dim=obs_dim, device=DEVICE)
    print(f"PPO agent initialized with {sum(p.numel() for p in agent.policy.parameters()):,} parameters")
    
    # Initialize experience buffer
    buffer = RolloutBuffer()
    
    # Training tracking variables
    total_steps = 0
    start_time = time.time()
    best_reward = float('-inf')
    
    print("Starting training loop...")
    print("" + "="*50)

    # =====================================================
    # Main Training Loop
    # =====================================================
    for update in range(1, NUM_UPDATES + 1):
        # =====================================================
        # Experience Collection Phase
        # =====================================================
        collected_steps = 0
        episode_rewards = []  # Track rewards for logging
        num_cells = []
        
        while collected_steps < STEPS_PER_UPDATE:
            # Convert observations to tensors for neural network processing
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            
            # Sample actions from current policy
            # Returns: action types, continuous parameters, log-probs, value estimates
            sampled_type, sampled_params, logp, values = agent.act(obs_tensor)
            
            # Convert neural network outputs to environment-compatible action format
            actions = make_action_dicts(sampled_type, sampled_params)
            
            # Execute actions in the environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done_flag = bool(terminated or truncated)
            
            # Process rewards and prepare for storage
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            episode_rewards.extend(rewards)  # Collect for logging
            num_cells.append(info["n_cells"])
            
            # Compute next state values for GAE bootstrap (if episode continues)
            if len(next_obs) > 0 and not done_flag:
                with torch.no_grad():
                    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)
                    _, _, _, next_vals_tensor = agent.policy(next_obs_t)
                    next_values_np = next_vals_tensor.cpu().numpy()
            else:
                next_values_np = np.array([], dtype=np.float32)
            
            # Store transitions in experience buffer
            # Each agent's experience is stored as a separate transition
            # Note: Number of agents can change between timesteps due to reproduction/death
            N_agents_current = len(values)  # Number of agents that took actions
            for i in range(N_agents_current):
                # Get next value for this agent (0.0 if episode done or agent index out of bounds)
                if done_flag:
                    next_value = 0.0
                else:
                    next_value = float(next_values_np[i]) if i < len(next_values_np) else 0.0
                    
                buffer.add(
                    obs[i].astype(np.float32),                    # Current observation
                    int(sampled_type[i].cpu().numpy()),           # Discrete action type
                    sampled_params[i].cpu().numpy().astype(np.float32),  # Continuous params
                    float(rewards_arr[i]) if i < len(rewards_arr) else 0.0,  # Reward
                    float(values[i].cpu().numpy()),               # Value estimate
                    float(logp[i].cpu().numpy()),                 # Action log-probability
                    done_flag,                                     # Episode termination
                    next_value                                     # Next state value
                )
            
            # Update state and counters
            obs = next_obs
            collected_steps += 1
            total_steps += 1
            
            # Reset environment if episode ended
            if done_flag:
                obs, _ = env.reset()
        
        # =====================================================
        # Policy Update Phase
        # =====================================================
        # Perform PPO updates using collected experience
        agent.ppo_update(buffer, PPO_EPOCHS, MINIBATCH_SIZE, GAMMA, LAM)
        
        # Clear buffer for next rollout
        buffer.clear()
        
        # =====================================================
        # Logging and Checkpointing
        # =====================================================
        if update % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            avg_num_cells = int(np.mean(num_cells)) if num_cells else 0
            max_num_cells = int(np.max(num_cells)) if num_cells else 0
            
            print(f"Update {update:4d}/{NUM_UPDATES} | "
                  f"Steps: {total_steps:6d} | "
                  f"Time: {elapsed_time:6.1f}s | "
                  f"Avg Reward: {avg_reward:6.3f} | "
                  f"Avg Num Cells: {avg_num_cells:6d} | "
                  f"Max Num Cells: {max_num_cells:6d}")
            
            # Track best performance
            if avg_reward > best_reward:
                best_reward = avg_reward
        
        # Save model checkpoints periodically
        if update % 100 == 0:
            ckpt_path = SAVE_DIR / f"ppo_colony_{update}.pt"
            checkpoint = {
                'policy_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'update': update,
                'total_steps': total_steps,
                'best_reward': best_reward,
                'hyperparameters': {
                    'obs_dim': obs_dim,
                    'gamma': GAMMA,
                    'lam': LAM,
                    'ppo_epochs': PPO_EPOCHS,
                    'minibatch_size': MINIBATCH_SIZE
                }
            }
            torch.save(checkpoint, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # =====================================================
    # Training Completion
    # =====================================================
    final_time = time.time() - start_time
    print("" + "="*50)
    print("Training completed successfully!")
    print(f"Total training time: {final_time:.1f} seconds ({final_time/3600:.2f} hours)")
    print(f"Total environment steps: {total_steps:,}")
    print(f"Best average reward achieved: {best_reward:.3f}")
    
    # Save final model
    final_ckpt = SAVE_DIR / "ppo_colony_final.pt"
    final_checkpoint = {
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'update': NUM_UPDATES,
        'total_steps': total_steps,
        'best_reward': best_reward,
        'training_time': final_time,
        'hyperparameters': {
            'obs_dim': obs_dim,
            'gamma': GAMMA,
            'lam': LAM,
            'ppo_epochs': PPO_EPOCHS,
            'minibatch_size': MINIBATCH_SIZE
        }
    }
    torch.save(final_checkpoint, final_ckpt)
    print(f"Final model saved: {final_ckpt}")
    print("" + "="*50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Checkpoints are saved in ppo_checkpoints/")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Check logs and configuration. Partial checkpoints may be available.")
        raise
