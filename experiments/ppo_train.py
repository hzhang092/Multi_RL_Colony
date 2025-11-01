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
import logging
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
STEPS_PER_UPDATE = 150         # Environment steps collected per policy update
                               # Total training steps = NUM_UPDATES * STEPS_PER_UPDATE

# PPO algorithm parameters
PPO_EPOCHS = 4                 # Number of optimization epochs per collected batch
MINIBATCH_SIZE = 1024          # Batch size for gradient updates
                               # Should be <= total transitions per update

# Optimizer and loss weighting
LR = 3e-4                      # Learning rate for PPO optimizer
VALUE_COEF = 0.5               # Critic loss coefficient (higher → stronger value fitting)

# Advantage estimation parameters
GAMMA = 0.99                   # Discount factor for future rewards
LAM = 0.95                     # GAE lambda parameter (bias-variance tradeoff)

# Checkpointing configuration
SAVE_DIR = Path("ppo_checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

# Logging configuration (writes to console and a timestamped .txt file)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"ppo_train_{time.strftime('%Y%m%d-%H%M%S')}.txt"

logger = logging.getLogger("ppo_train")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter('%(asctime)s | %(message)s')
_fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger.handlers.clear()
logger.addHandler(_fh)
logger.addHandler(_sh)

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
    logger.info(f"Starting PPO training on device: {DEVICE}")
    logger.info("Training configuration:")
    logger.info(f"  - Updates: {NUM_UPDATES}")
    logger.info(f"  - Steps per update: {STEPS_PER_UPDATE}")
    logger.info(f"  - Total training steps: {NUM_UPDATES * STEPS_PER_UPDATE:,}")
    logger.info(f"  - PPO epochs: {PPO_EPOCHS}")
    logger.info(f"  - Minibatch size: {MINIBATCH_SIZE}")
    logger.info(f"  - Discount factor (gamma): {GAMMA}")
    logger.info(f"  - GAE lambda: {LAM}")
    logger.info(f"  - Learning rate: {LR}")
    logger.info(f"  - Value coefficient: {VALUE_COEF}")
    logger.info(f"  - Log file: {LOG_FILE}")
    logger.info("" + "="*50)
    
    # =====================================================
    # Environment and Agent Initialization
    # =====================================================
    env = ColonyEnv(seed=ENV_SEED)
    obs, _ = env.reset()
    obs_dim = obs.shape[1]  # Observation dimension per agent
    logger.info(f"Environment initialized with observation dimension: {obs_dim}")
    
    # Initialize PPO agent with shared policy
    agent = PPOAgent(obs_dim=obs_dim, device=DEVICE, lr=LR, value_coef=VALUE_COEF)
    logger.info(f"PPO agent initialized with {sum(p.numel() for p in agent.policy.parameters()):,} parameters")
    
    # Initialize experience buffer
    buffer = RolloutBuffer()
    
    # Training tracking variables
    total_steps = 0
    start_time = time.time()
    best_reward = float('-inf')
    
    logger.info("Starting training loop...")
    logger.info("" + "="*50)

    # =====================================================
    # Main Training Loop
    # =====================================================
    for update in range(1, NUM_UPDATES + 1):
        # Reset environment at the start of each update
        obs, _ = env.reset()
        terminated, truncated = False, False
        num_survivors = 0  # Track number of surviving agents
        
        # =====================================================
        # Experience Collection Phase
        # =====================================================
        collected_steps = 0
        episode_rewards = []  # Track rewards for logging
        num_cells = []
        action_history = []
        
        while collected_steps < STEPS_PER_UPDATE:
            # Convert observations to tensors for neural network processing
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            
            # Sample actions from current policy
            # Returns: action types, log-probs, value estimates
            sampled_type, logp, values = agent.act(obs_tensor)
            
            # Convert neural network outputs to environment-compatible action format
            actions = make_action_dicts(sampled_type)
            
            # Execute actions in the environment
            # now len(rewards) == number of agents that acted == len(actions)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done_flag = bool(terminated or truncated)

            # Store action history
            action_history.extend(sampled_type.cpu().numpy().tolist())
            
            # Process rewards and prepare for storage
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            episode_rewards.extend(rewards)  # Collect for logging
            num_cells.append(info["n_cells"])
            
            num_survivors = 0
            survivor_indices = []
            next_values_np = np.array([], dtype=np.float32)
            # Compute next state values for GAE bootstrap (if episode continues)
            if len(next_obs) > 0 and not done_flag:
                with torch.no_grad():
                    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)
                    _, next_vals_tensor = agent.policy(next_obs_t)
                    next_values_np = next_vals_tensor.cpu().numpy()

                # save survivor indices (cells that did not divide) for mapping next values
                survivor_indices = info.get("survivor_indices", [])
                num_survivors = len(survivor_indices)

            # Build next_values for GAE:
            # - Start with baseline = parent's current value (neutral w.r.t dividing)
            # - For survivors: use actual next-state values from next_obs
            # - For dividing parents: blend avg child value with parent value to encourage correct division
            N_agents_acted = len(obs)  # Number of agents that took actions
            parent_values_np = values.cpu().numpy().flatten().astype(np.float32)
            next_values_for_acting_agents = parent_values_np.copy()

            if len(next_obs) > 0 and not done_flag and next_values_np.size > 0:
                # Assign survivors' next values (next_obs ordering: [survivors..., children...])
                if num_survivors > 0:
                    next_values_for_acting_agents[survivor_indices] = next_values_np[:num_survivors].astype(np.float32)

                # For dividing parents, use a blended estimate based on average child value
                child_values_np = next_values_np[num_survivors:] if next_values_np.size > num_survivors else np.array([], dtype=np.float32)
                dividing_parents = [i for i in range(N_agents_acted) if i not in survivor_indices]

                if len(dividing_parents) > 0:
                    if child_values_np.size > 0:
                        avg_child_value = float(np.mean(child_values_np))
                    else:
                        # Fallback: average over all next values if children slice is empty
                        avg_child_value = float(np.mean(next_values_np)) if next_values_np.size > 0 else 0.0

                    blend = 0.8  # weight for child value vs parent value (tunable 0.7–0.9)
                    for idx_parent in dividing_parents:
                        parent_val = next_values_for_acting_agents[idx_parent]
                        next_values_for_acting_agents[idx_parent] = (
                            blend * avg_child_value + (1.0 - blend) * float(parent_val)
                        )
            
            # Store transitions in experience buffer
            # Each agent's experience is stored as a separate transition
            obs_batch = [o.astype(np.float32) for o in obs]
            action_type_batch = sampled_type.cpu().numpy().astype(int).tolist()
            rewards_batch = rewards_arr.tolist()
            values_batch = values.cpu().numpy().flatten().tolist()
            logp_batch = logp.cpu().numpy().tolist()
            done_flag_batch = [done_flag] * len(obs_batch)

            buffer.add_batch(obs_batch,
                             action_type_batch,
                             rewards_batch,
                             values_batch, 
                             logp_batch, 
                             done_flag_batch,
                             next_values_for_acting_agents.tolist())
            
            
            # Update state and counters
            obs = next_obs
            collected_steps += 1
            total_steps += 1
            
            # Reset environment if episode ended
            if done_flag:
                obs, _ = env.reset()
                num_survivors = 0  # Reset survivor count for next episode
                
        
        # =====================================================
        # Policy Update Phase
        # =====================================================
        # Get number of transitions before clearing buffer
        num_transitions = len(buffer.obs)
        
        # Perform PPO updates using collected experience
        train_stats = agent.ppo_update(buffer, PPO_EPOCHS, MINIBATCH_SIZE, GAMMA, LAM)
        
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
            action_tuple = (np.sum(np.array(action_history) == 0), np.sum(np.array(action_history) == 1), np.sum(np.array(action_history) == 2))
            
            logger.info(
                f"Update {update:4d}/{NUM_UPDATES} | "
                f"Steps: {total_steps:6d} | "
                f"Transitions: {num_transitions:5d} | "
                f"Time: {elapsed_time:6.1f}s | "
                f"AVG Reward: {avg_reward:6.3f} | "
                f"AVG Num Cells: {avg_num_cells:4d} | "
                f"P_Loss: {train_stats['policy_loss']:.3f} | "
                f"V_Loss: {train_stats['value_loss']:.3f} | "
                f"Entropy: {train_stats['entropy']:.3f} | "
                f"Avg_Return: {train_stats['avg_return']:.3f} | "
                f"avg actions: {action_tuple}"
            )

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
                    'minibatch_size': MINIBATCH_SIZE,
                    'lr': LR,
                    'value_coef': VALUE_COEF
                }
            }
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # =====================================================
    # Training Completion
    # =====================================================
    final_time = time.time() - start_time
    logger.info("" + "="*50)
    logger.info("Training completed successfully!")
    logger.info(f"Total training time: {final_time:.1f} seconds ({final_time/3600:.2f} hours)")
    logger.info(f"Total environment steps: {total_steps:,}")
    logger.info(f"Best average reward achieved: {best_reward:.3f}")
    
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
            'minibatch_size': MINIBATCH_SIZE,
            'lr': LR,
            'value_coef': VALUE_COEF
        }
    }
    torch.save(final_checkpoint, final_ckpt)
    logger.info(f"Final model saved: {final_ckpt}")
    logger.info("" + "="*50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Checkpoints are saved in ppo_checkpoints/")
    except Exception as e:
        logger.exception(f"\nTraining failed with error: {e}")
        logger.info("Check logs and configuration. Partial checkpoints may be available.")
        raise
