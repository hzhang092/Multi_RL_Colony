# agents/ppo_agent.py
"""
PPO Agent Implementation for Multi-Agent Colony Environment

This module contains the core components for a Proximal Policy Optimization (PPO) agent
designed for the multi-agent ColonyEnv. The implementation features a shared policy
approach where all agents use the same neural        device (str or torch.device, optional): Device for computation ('cpu' or 'cuda'). Defaults to \"cpu\".network policy.

Key Components:
- SharedActorCritic: Neural network that outputs both policy and value estimates
- PPOAgent: Wrapper class that handles action sampling, evaluation, and policy updates
- RolloutBuffer: Storage for experience collection and replay
- Utility functions for action processing and advantage estimation

Features:
- Hybrid action space support (discrete + continuous actions)
- Generalized Advantage Estimation (GAE)
- PPO clipped objective with value function loss and entropy bonus
- Action squashing for continuous parameters (sigmoid/tanh)

Usage:
    from agents.ppo_agent import PPOAgent, RolloutBuffer
    
    agent = PPOAgent(obs_dim=observation_dimension)
    buffer = RolloutBuffer()
    
    # During rollout:
    actions_type, actions_params, logprobs, values = agent.act(observations)
    
    # During training:
    agent.ppo_update(buffer, epochs=4, batch_size=256, gamma=0.99, lam=0.95)

Author: Multi-Agent RL Colony Project
Date: 2024
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Union


# =====================================================
# Shared Actor–Critic Network
# =====================================================
class SharedActorCritic(nn.Module):
    """
    Shared Actor-Critic Neural Network for Multi-Agent Colony Environment.
    
    This network serves as both the policy (actor) and value function (critic) for all agents
    in the colony environment. It handles the hybrid action space consisting of:
    - Discrete action type selection (3 types: dormant, grow, reproduce)
    - Continuous parameters grow_frac
    
    Architecture:
    - Shared encoder: Maps observations to hidden representations
    - Type head: Outputs logits for discrete action type distribution
    - Parameter head: Outputs mean values for continuous parameter distributions
    - Value head: Outputs state-value estimates for the critic
    
    Args:
        obs_dim (int): Dimension of the observation space from the environment
        n_types (int, optional): Number of discrete action types. Defaults to 3.
        hidden (int, optional): Size of hidden layers in the network. Defaults to 128.
    
    Returns:
        When called with forward():
        - logits (torch.Tensor): Raw logits for discrete action type (shape: [batch, n_types])
        - param_mean (torch.Tensor): Mean values for continuous parameters (shape: [batch, 2])
        - param_std (torch.Tensor): Standard deviations for continuous parameters (shape: [2])
        - value (torch.Tensor): State-value estimates (shape: [batch])
    
    Example:
        >>> network = SharedActorCritic(obs_dim=10, n_types=3, hidden=128)
        >>> obs = torch.randn(32, 10)  # batch of 32 observations
        >>> logits, means, stds, values = network(obs)
        >>> logits.shape  # torch.Size([32, 3])
        >>> means.shape   # torch.Size([32, 2])
        >>> values.shape  # torch.Size([32])
    """
    def __init__(self, obs_dim: int, n_types: int = 3, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        #self.type_head = nn.Linear(hidden, n_types)
        #self.value_head = nn.Linear(hidden, 1)
        
        # Actor head: small nonlinear projection before logits
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_types)
        )

        # Critic head: small nonlinear projection before scalar value
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, obs: torch.FloatTensor):
        """
        Forward pass through the shared actor-critic network.
        
        Args:
            obs (torch.FloatTensor): Batch of observations from the environment.
                                   Shape: [batch_size, obs_dim]
        
        Returns:
            tuple: A 4-tuple containing:
                - logits (torch.FloatTensor): Raw logits for discrete action types.
                                            Shape: [batch_size, n_types]
                - (Deprecated) param_mean (torch.FloatTensor): Mean values for continuous parameters.
                                                Shape: [batch_size, 2]
                - (Deprecated) param_std (torch.FloatTensor): Standard deviations for continuous parameters.
                                               Shape: [2] (shared across batch)
                - value (torch.FloatTensor): State-value estimates.
                                           Shape: [batch_size]
        
        Note:
            The continuous parameters are modeled as Gaussian distributions with
            learned means and shared standard deviations. The standard deviations
            are constrained to be positive through exponentiation.
        """
        h = self.encoder(obs)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return logits, value


# =====================================================
# Utilities
# =====================================================
def make_action_dicts(type_tensor: torch.Tensor):
    """
    Convert raw action tensors to environment-compatible action format.
    
    This function transforms the raw network outputs into the specific format
    required by the ColonyEnv. Since growth is now constant, we only need
    to handle discrete action types.
    
    Args:
        type_tensor (torch.Tensor): Sampled discrete action types.
                                  Shape: [batch_size]
    
    Returns:
        list: List of integers representing discrete action types compatible 
              with ColonyEnv.step().
    
    Example:
        >>> types = torch.tensor([0, 1, 2])
        >>> actions = make_action_dicts(types)
        >>> actions  # [0, 1, 2]
    """
    types = type_tensor.cpu().numpy().astype(int)
    return types.tolist()


# =====================================================
# Rollout Buffer
# =====================================================
class RolloutBuffer:
    """
    Experience Replay Buffer for PPO Training.
    
    Stores transitions collected during environment rollouts for batch training.
    Each transition represents a single agent's experience at one timestep.
    The buffer flattens multi-agent experiences into a single sequence for
    shared policy training.
    
    Attributes:
        obs (list): Observations from each transition
        actions_type (list): Discrete action types taken
        actions_params (list): Raw continuous parameters (pre-squashing)
        rewards (list): Rewards received for each transition
        values (list): Value function estimates at each state
        logp (list): Log-probabilities of the actions taken
        dones (list): Episode termination flags
        next_values (list): Value estimates of next states (for GAE)
    
    Usage:
        >>> buffer = RolloutBuffer()
        >>> buffer.add(obs, action_type, params, reward, value, logp, done, next_val)
        >>> print(f"Buffer contains {len(buffer.obs)} transitions")
        >>> buffer.clear()  # Reset for next rollout
    """
    def __init__(self):
        self.obs, self.actions_type = [], []
        self.rewards, self.values, self.logp = [], [], []
        self.dones, self.next_values = [], []

    def add(self, obs, a_type, reward, value, logp, done, next_value):
        """
        Add a single transition to the buffer.
        
        Args:
            obs (np.ndarray): Observation at current timestep
            a_type (int): Discrete action type taken
            reward (float): Reward received for this transition
            value (float): Value function estimate for current state
            logp (float): Log-probability of the action taken
            done (bool): Whether episode terminated after this transition
            next_value (float): Value estimate for next state (0 if done)
        """
        self.obs.append(obs)
        self.actions_type.append(a_type)
        self.rewards.append(reward)
        self.values.append(value)
        self.logp.append(logp)
        self.dones.append(done)
        self.next_values.append(next_value)
        
    def add_batch(self, obs_batch, a_type_batch, reward_batch, value_batch, logp_batch, done_batch, next_value_batch):
        """
        Add a batch of transitions.
        Each input should be an iterable with the same length.
        """
        self.obs.extend(obs_batch)
        self.actions_type.extend(a_type_batch)
        self.rewards.extend(reward_batch)
        self.values.extend(value_batch)
        self.logp.extend(logp_batch)
        self.dones.extend(done_batch)
        self.next_values.extend(next_value_batch)

    def clear(self):
        #self.__init__()
        self.obs.clear()
        self.actions_type.clear()
        self.rewards.clear()
        self.values.clear()
        self.logp.clear()
        self.dones.clear()
        self.next_values.clear()


def compute_gae_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) for PPO training.
    
    GAE provides a bias-variance tradeoff for advantage estimation by combining
    multiple n-step returns with exponentially decaying weights. This leads to
    more stable policy gradient estimates compared to simple advantage estimation.
    
    The GAE formula:
    A_t^GAE = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    
    where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the temporal difference error.
    
    Args:
        rewards (list): Rewards for each transition
        values (list): Value function estimates for each state
        next_values (list): Value estimates for next states
        dones (list): Episode termination flags
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        lam (float, optional): GAE lambda parameter controlling bias-variance
                             tradeoff. Higher values increase variance but reduce bias.
                             Defaults to 0.95.
    
    Returns:
        tuple: A 2-tuple containing:
            - adv (np.ndarray): Computed advantages (shape: [T])
            - returns (np.ndarray): Target returns for value function training (shape: [T])
    
    Note:
        Returns are computed as advantages + values, which gives the GAE-based
        estimate of the true value function targets.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


# =====================================================
# PPO Agent Wrapper
# =====================================================
class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for Multi-Agent Colony Environment.
    
    This class wraps the SharedActorCritic network and implements the complete PPO
    algorithm including action sampling, policy evaluation, and parameter updates.
    The agent uses a shared policy approach where all agents in the environment
    share the same neural network parameters.
    
    Key Features:
    - Clipped surrogate objective for stable policy updates
    - Value function learning with clipped value loss
    - Entropy regularization for exploration
    - Gradient clipping for training stability
    - Minibatch training with multiple epochs per update
    
    Args:
        obs_dim (int): Dimension of the observation space
        lr (float, optional): Learning rate for Adam optimizer. Defaults to 3e-4.
        clip_eps (float, optional): PPO clipping parameter. Defaults to 0.2.
        value_coef (float, optional): Coefficient for value function loss. Defaults to 0.5.
        ent_coef (float, optional): Coefficient for entropy bonus. Defaults to 0.01.
        max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 0.5.
        device (str, optional): Device for computation ('cpu' or 'cuda'). Defaults to "cpu".
    
    Example:
        >>> agent = PPOAgent(obs_dim=20, lr=1e-3, device='cuda')
        >>> obs_tensor = torch.randn(16, 20)  # batch of observations
        >>> types, params, logprobs, values = agent.act(obs_tensor)
        >>> 
        >>> # After collecting experience in buffer:
        >>> agent.ppo_update(buffer, epochs=4, batch_size=256, gamma=0.99, lam=0.95)
    """
    def __init__(self, obs_dim: int, lr: float = 3e-4, clip_eps: float = 0.2, 
                 value_coef: float = 0.5, ent_coef: float = 0.01, 
                 max_grad_norm: float = 0.5, device: Union[str, torch.device] = "cpu"):
        self.device = device
        self.policy = SharedActorCritic(obs_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def act(self, obs_tensor):
        """
        Sample actions from the current policy given observations.
        
        This method performs forward pass through the policy network and samples
        actions from the resulting distributions. It handles both discrete and
        continuous action components.
        
        Args:
            obs_tensor (torch.Tensor): Batch of observations from the environment.
                                     Shape: [batch_size, obs_dim]
        
        Returns:
            tuple: A 4-tuple containing:
                - sampled_type (torch.Tensor): Sampled discrete action types.
                                              Shape: [batch_size]
                - logp (torch.Tensor): Log-probabilities of sampled actions.
                                     Shape: [batch_size]
                - values (torch.Tensor): Value function estimates.
                                        Shape: [batch_size]
        
        Note:
            This method is used during environment rollouts. All computations
            are performed with torch.no_grad() for efficiency.
        """
        with torch.no_grad():
            logits, values = self.policy(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist_type = Categorical(probs)

            sampled_type = dist_type.sample()
            logp = dist_type.log_prob(sampled_type)

        return sampled_type, logp, values

    def evaluate_actions(self, obs, types, with_grad: bool = True):
        """
        Evaluate actions under the current policy for PPO updates.
        
        This method re-evaluates previously taken actions under the current policy
        to compute updated log-probabilities, value estimates, and entropy for
        the PPO objective function.
        
        Args:
            obs (torch.Tensor): Batch of observations. Shape: [batch_size, obs_dim]
            types (torch.Tensor): Discrete action types to evaluate. Shape: [batch_size]
            with_grad (bool, optional): Whether to compute gradients.
                usage:  Training step / PPO update (backprop): with_grad=True
                        Logging or rollout evaluation (no backprop): with_grad=False
        
        Returns:
            tuple: A 3-tuple containing:
                - logp (torch.Tensor): Log-probabilities of the actions.
                                     Shape: [batch_size]
                - values_pred (torch.Tensor): Current value function estimates.
                                             Shape: [batch_size]
                - entropy (torch.Tensor): Policy entropy (scalar)
        
        Note:
            This method is used during PPO updates to compute the ratio between
            current and old policy probabilities for the clipped objective.
        """
        if not with_grad:
            with torch.no_grad():
                logits, values_pred = self.policy(obs)
                dist_type = torch.distributions.Categorical(logits=logits)
                logp = dist_type.log_prob(types)
                entropy = dist_type.entropy().mean()
                return logp, values_pred, entropy
        
        
        logits, values_pred = self.policy(obs)
        dist_type = Categorical(torch.softmax(logits, dim=-1))

        logp = dist_type.log_prob(types)
        entropy = dist_type.entropy().mean()

        return logp, values_pred, entropy

    def ppo_update(self, buffer, epochs, batch_size, gamma, lam):
        """
        Perform PPO policy and value function updates.
        
        This method implements the core PPO algorithm including:
        1. GAE advantage computation
        2. Advantage normalization
        3. Minibatch training over multiple epochs
        4. Clipped surrogate objective
        5. Value function loss with clipping
        6. Entropy regularization
        
        Args:
            buffer (RolloutBuffer): Experience buffer containing collected transitions
            epochs (int): Number of optimization epochs per update
            batch_size (int): Size of minibatches for training
            gamma (float): Discount factor for GAE computation
            lam (float): Lambda parameter for GAE
        
        Returns:
            dict: A dictionary containing training statistics for logging.
        
        Note:
            The method modifies the policy network parameters in-place through
            gradient-based optimization. It uses normalized advantages for stability
            and clips both policy and value updates to prevent large parameter changes.
        
        PPO Objective:
            L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
            where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio.
        """
        advs, returns = compute_gae_advantages(
            buffer.rewards, buffer.values, buffer.next_values, buffer.dones, gamma, lam)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        obs_tensor = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=self.device)
        types_tensor = torch.tensor(np.array(buffer.actions_type), dtype=torch.int64, device=self.device)
        old_logp_tensor = torch.tensor(np.array(buffer.logp), dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advs_tensor = torch.tensor(advs, dtype=torch.float32, device=self.device)
        values_tensor = torch.tensor(np.array(buffer.values), dtype=torch.float32, device=self.device)

        n = obs_tensor.shape[0]
        
        # For logging
        all_policy_loss, all_value_loss, all_entropy = [], [], []

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                obs_b, types_b = obs_tensor[batch_idx], types_tensor[batch_idx]
                old_logp_b, returns_b, adv_b, vals_b = old_logp_tensor[batch_idx], returns_tensor[batch_idx], advs_tensor[batch_idx], values_tensor[batch_idx]

                logp, values_pred, entropy = self.evaluate_actions(obs_b, types_b)

                ratio = torch.exp(logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                value_clipped = vals_b + (values_pred - vals_b).clamp(-self.clip_eps, self.clip_eps)
                value_loss1 = (values_pred - returns_b).pow(2)
                value_loss2 = (value_clipped - returns_b).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))

                loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Log stats
                all_policy_loss.append(policy_loss.item())
                all_value_loss.append(value_loss.item())
                all_entropy.append(entropy.item())
        
        return {
            "policy_loss": np.mean(all_policy_loss),
            "value_loss": np.mean(all_value_loss),
            "entropy": np.mean(all_entropy),
            "avg_return": np.mean(returns)
        }


"""
Policy Loss (policy_loss): This measures how much the policy is changing. It should ideally decrease over time, indicating that the policy is stabilizing around an optimum. If it's flat, the agent isn't learning. If it's noisy or increasing, the learning rate might be too high.
Value Loss (value_loss): This measures how well the critic is predicting the expected returns. It should also decrease, showing that the agent is getting better at estimating the value of states.
Entropy (entropy): This measures the randomness or "exploratoriness" of the policy. It should gradually decrease as the agent becomes more confident in its actions. If it drops too quickly, the agent might be getting stuck in a suboptimal policy. If it stays high, the agent isn't learning a clear strategy.
"""