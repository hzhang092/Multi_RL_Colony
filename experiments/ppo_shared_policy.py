## deprecated, split into agents/ppo_agent.py and experiments/ppo_train.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# ppo_full_trainer.py
# Full PPO trainer for shared-policy ColonyEnv.
# Usage: python ppo_full_trainer.py
# Requirements: torch, numpy, pandas (pip install torch numpy pandas)

"""
This script implements a full Proximal Policy Optimization (PPO) trainer pipeline for
the multi-agent ColonyEnv, where all agents share a single policy.

Key Features:
- Shared Actor-Critic Network: A single neural network provides both the policy
  (actor) and the value function (critic) for all agents.
- Hybrid Action Space: Handles the environment's `Dict` action space by
  predicting a discrete action 'type' and continuous parameters 'grow_frac'
  and 'torque'.
- Per-Agent Transition Collection: Correctly unnests and flattens transitions
  from multiple agents at each timestep into a single buffer.
- GAE Advantage Estimation: Uses Generalized Advantage Estimation (GAE) for
  a stable and effective advantage calculation.
- PPO Clipped Objective: Implements the core PPO objective function, including
  value loss and an entropy bonus to encourage exploration.
- Minibatch Updates: Iterates over the collected data for multiple epochs using
  randomized minibatches for stable and efficient learning.
- Checkpointing and Logging: Saves model checkpoints periodically and logs
  training progress to the console.
- Gymnasium Compatibility: Correctly handles the `terminated` and `truncated`
  flags from the environment's `step` method.

Approximation Note on Continuous Actions:
The policy models the continuous actions (`grow_frac`, `torque`) as samples
from a Gaussian distribution. These are then squashed into the required ranges
(0,1) and (-1,1) using sigmoid and tanh functions, respectively. For simplicity
and stability, this implementation uses the log-probability of the raw,
pre-squashed Gaussian sample in the PPO objective. This is a common and
practical approximation that avoids the complexity of applying the change of
variables formula (log-Jacobian correction).
"""

import math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

import sys
# Add the project root to the Python path to allow for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from envs.colony_env import ColonyEnv

# --- Hyperparameters ---
ENV_SEED = 686
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_UPDATES = 500             # Total number of policy updates to perform.
STEPS_PER_UPDATE = 200         # Number of environment steps to run for each policy update.
PPO_EPOCHS = 4                 # Number of epochs to train on the collected data per update.
MINIBATCH_SIZE = 4096          # Size of minibatches for training.
GAMMA = 0.99                   # Discount factor for future rewards.
LAM = 0.95                     # Lambda parameter for GAE.
CLIP_EPS = 0.2                 # Clipping parameter for the PPO surrogate objective.
LR = 3e-4                      # Learning rate for the Adam optimizer.
VALUE_COEF = 0.5               # Coefficient for the value function loss.
ENT_COEF = 0.01                # Coefficient for the entropy bonus.
MAX_GRAD_NORM = 0.5            # Maximum norm for gradient clipping.

SAVE_DIR = Path("ppo_checkpoints_original")
SAVE_DIR.mkdir(exist_ok=True)

# --- Shared Actor-Critic Network ---
class SharedActorCritic(nn.Module):
    """
    A shared actor-critic network that processes an observation and outputs:
    - Logits for the discrete action 'type'.
    - Mean values for the continuous action parameters ('grow_frac', 'torque').
    - A state-value estimate.
    """
    def __init__(self, obs_dim: int, n_types: int = 3, hidden: int = 128):
        """
        Args:
            obs_dim (int): The dimension of the observation space.
            n_types (int): The number of discrete action types.
            hidden (int): The size of the hidden layers.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # Actor head for the discrete action 'type'
        self.type_head = nn.Linear(hidden, n_types)
        # Actor head for the continuous parameters (predicts the mean)
        self.param_mean = nn.Linear(hidden, 2)
        # A single learnable log standard deviation for the continuous parameters.
        # This is shared across all states for simplicity.
        self.param_logstd = nn.Parameter(torch.ones(2) * -0.5)
        # Critic head for the value function
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.FloatTensor):
        """
        Performs a forward pass through the network.

        Args:
            obs (torch.FloatTensor): A batch of observations.

        Returns:
            Tuple containing:
            - logits (torch.FloatTensor): Logits for the discrete action distribution.
            - param_mean (torch.FloatTensor): Mean values for the continuous parameter distributions.
            - param_std (torch.FloatTensor): Standard deviation for the continuous parameters.
            - value (torch.FloatTensor): The estimated state-value.
        """
        h = self.encoder(obs)
        logits = self.type_head(h)
        param_mean = self.param_mean(h)
        param_std = torch.exp(self.param_logstd)  # Ensure std is positive
        value = self.value_head(h).squeeze(-1)
        return logits, param_mean, param_std, value

# --- Helper Utilities ---
def make_action_dicts(type_tensor: torch.Tensor, param_tensor: torch.Tensor) -> list:
    """
    Converts batched action tensors into a list of action dictionaries
    compatible with the ColonyEnv's `step` method.

    It applies squashing functions to map the raw network outputs to the
    required action ranges:
    - `grow_frac`: sigmoid(raw_param_0) -> (0, 1)
    - `torque`: tanh(raw_param_1) -> (-1, 1)

    Args:
        type_tensor (torch.Tensor): A tensor of sampled discrete action types.
        param_tensor (torch.Tensor): A tensor of raw sampled continuous parameters.

    Returns:
        list: A list of action dictionaries.
    """
    types = type_tensor.cpu().numpy().astype(int)
    params = param_tensor.cpu().numpy()
    action_list = []
    for i in range(len(types)):
        grow_raw, torque_raw = params[i, 0], params[i, 1]
        grow_frac = float(1.0 / (1.0 + math.exp(-grow_raw)))     # Sigmoid
        torque = float(math.tanh(torque_raw))                   # Tanh
        action_list.append({
            "type": int(types[i]),
            "grow_frac": np.array([grow_frac], dtype=np.float32),
            "torque": np.array([torque], dtype=np.float32)
        })
    return action_list

class RolloutBuffer:
    """
    A buffer to store and flatten transitions from multiple agents over multiple steps.
    """
    def __init__(self):
        self.obs = []
        self.actions_type = []
        self.actions_params = []  # Raw (pre-squash) parameters
        self.rewards = []
        self.values = []
        self.logp = []
        self.dones = []           # True if the episode ended after this transition
        self.next_values = []     # Value of the next state for GAE bootstrapping

    def add(self, obs, a_type, a_params, reward, value, logp, done, next_value):
        """Adds a single agent's transition to the buffer."""
        self.obs.append(obs)
        self.actions_type.append(a_type)
        self.actions_params.append(a_params)
        self.rewards.append(reward)
        self.values.append(value)
        self.logp.append(logp)
        self.dones.append(done)
        self.next_values.append(next_value)

    def size(self) -> int:
        """Returns the number of transitions stored in the buffer."""
        return len(self.obs)

    def clear(self):
        """Clears the buffer."""
        self.__init__()

def compute_gae_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) and returns.

    Args:
        rewards (list): List of rewards for each transition.
        values (list): List of state-value estimates for each transition.
        next_values (list): List of next state-value estimates.
        dones (list): List of done flags.
        gamma (float): Discount factor.
        lam (float): GAE lambda parameter.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - adv: The computed advantages.
            - returns: The computed returns (advantages + values).
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns

# --- Main training loop ---
def train():
    """
    The main training function.
    """
    # --- Initialization ---
    env = ColonyEnv(seed=ENV_SEED)
    obs, _ = env.reset()
    obs_dim = obs.shape[1]
    policy = SharedActorCritic(obs_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    buffer = RolloutBuffer()
    total_steps = 0
    start_time = time.time()

    # --- Outer Training Loop ---
    for update in range(1, NUM_UPDATES + 1):
        # --- Data Collection Phase ---
        collected_steps = 0
        while collected_steps < STEPS_PER_UPDATE:
            obs_batch = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

            # Sample actions from the current policy
            with torch.no_grad():
                logits, param_mean, param_std, values = policy(obs_batch)
                # Sample discrete action type
                probs = torch.softmax(logits, dim=-1)
                dist_type = Categorical(probs)
                sampled_type = dist_type.sample()
                logp_type = dist_type.log_prob(sampled_type)
                # Sample continuous parameters from a Gaussian distribution
                dist_params = Normal(param_mean, param_std)
                sampled_params = dist_params.sample()  # Raw (pre-squash) samples
                logp_params = dist_params.log_prob(sampled_params).sum(dim=-1)
                # Total log probability is the sum of log-probs from each distribution
                logp_total = logp_type + logp_params

            # Convert tensors to action dicts and step the environment
            actions_for_env = make_action_dicts(sampled_type, sampled_params)
            step_out = env.step(actions_for_env)

            # Handle Gymnasium's 5-element tuple (obs, reward, terminated, truncated, info)
            if len(step_out) == 5:
                next_obs, rewards_raw, terminated, truncated, info = step_out
                done_flag = bool(terminated or truncated)
            else: # Handle older 4-element tuple for backward compatibility
                next_obs, rewards_raw, done_flag, info = step_out

            rewards_arr = np.asarray(rewards_raw, dtype=np.float32)
            next_obs_batch = next_obs

            # Get the value of the next state for GAE bootstrapping
            if len(next_obs_batch) > 0:
                with torch.no_grad():
                    next_obs_tensor = torch.tensor(next_obs_batch, dtype=torch.float32, device=DEVICE)
                    _, _, _, next_values_tensor = policy(next_obs_tensor)
                    next_values = next_values_tensor.cpu().numpy()
            else:
                next_values = np.array([], dtype=np.float32)

            # Store all per-agent transitions in the buffer
            N_agents_prev = len(values) # Number of agents that took an action
            for i in range(N_agents_prev):
                o = obs[i].astype(np.float32)
                a_type = int(sampled_type[i].cpu().numpy())
                a_param = sampled_params[i].cpu().numpy().astype(np.float32)
                # The reward array might be longer if agents divided.
                # We assign the reward to the parent agent that took the action.
                r = float(rewards_arr[i]) if i < len(rewards_arr) else 0.0
                v = float(values[i].cpu().numpy())
                lp = float(logp_total[i].cpu().numpy())
                # If the episode is done, the next state's value is 0.
                nv = 0.0 if done_flag else (float(next_values[i]) if i < len(next_values) else 0.0)
                buffer.add(o, a_type, a_param, r, v, lp, done_flag, nv)

            # Prepare for the next step
            obs = next_obs
            collected_steps += 1
            total_steps += 1

            # If the episode ended, reset the environment to continue collecting data
            if done_flag:
                obs, _ = env.reset()

        # --- PPO Update Phase ---
        # Compute advantages and returns from the collected transitions
        advs, returns = compute_gae_advantages(buffer.rewards, buffer.values, buffer.next_values, buffer.dones, GAMMA, LAM)
        # Normalize advantages for training stability
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Convert collected data into tensors for training
        obs_tensor = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=DEVICE)
        types_tensor = torch.tensor(np.array(buffer.actions_type), dtype=torch.int64, device=DEVICE)
        params_tensor = torch.tensor(np.array(buffer.actions_params), dtype=torch.float32, device=DEVICE)
        old_logp_tensor = torch.tensor(np.array(buffer.logp), dtype=torch.float32, device=DEVICE)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        advs_tensor = torch.tensor(advs, dtype=torch.float32, device=DEVICE)
        values_tensor = torch.tensor(np.array(buffer.values), dtype=torch.float32, device=DEVICE)

        dataset_size = obs_tensor.shape[0]
        batch_size = MINIBATCH_SIZE if MINIBATCH_SIZE < dataset_size else dataset_size

        # Train for multiple epochs on the collected data
        for epoch in range(PPO_EPOCHS):
            perm = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                idx = perm[start:start+batch_size]
                obs_b, types_b, params_b = obs_tensor[idx], types_tensor[idx], params_tensor[idx]
                old_logp_b, returns_b, adv_b, vals_b = old_logp_tensor[idx], returns_tensor[idx], advs_tensor[idx], values_tensor[idx]

                # Re-evaluate the policy on the minibatch to get current values and log-probs
                logits, param_mean, param_std, values_pred = policy(obs_b)
                
                # Calculate new log probabilities
                dist_type = Categorical(torch.softmax(logits, dim=-1))
                logp_type = dist_type.log_prob(types_b)
                dist_params = Normal(param_mean, param_std)
                logp_params = dist_params.log_prob(params_b).sum(dim=-1)
                logp = logp_type + logp_params

                # --- PPO Loss Calculation ---
                # 1. Policy Loss (Clipped Surrogate Objective)
                ratio = torch.exp(logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # 2. Value Loss (Clipped Value Function)
                value_clipped = vals_b + (values_pred - vals_b).clamp(-CLIP_EPS, CLIP_EPS)
                value_loss1 = (values_pred - returns_b).pow(2)
                value_loss2 = (value_clipped - returns_b).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))

                # 3. Entropy Bonus (to encourage exploration)
                ent_type = dist_type.entropy().mean()
                ent_params = dist_params.entropy().sum(dim=-1).mean()
                entropy = ent_type + ent_params

                # Total Loss
                loss = policy_loss + VALUE_COEF * value_loss - ENT_COEF * entropy

                # --- Optimization Step ---
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # --- Logging & Checkpointing ---
        if update % 10 == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(buffer.rewards) if buffer.size() > 0 else 0.0
            print(f"[Update {update}/{NUM_UPDATES}] Steps={total_steps} Elapsed={int(elapsed)}s AvgReward={avg_reward:.4f} Transitions={dataset_size}")
            
            # Save a checkpoint
            ckpt_path = SAVE_DIR / f"ppo_ckpt_update_{update}.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
                "total_steps": total_steps
            }, ckpt_path)

        # Clear the buffer for the next round of data collection
        buffer.clear()

    print("Training finished.")

if __name__ == "__main__":
    train()
