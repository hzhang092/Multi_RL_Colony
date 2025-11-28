# ColonyEnv Project Guide

A high-signal, practical guide to understand the full project and start contributing quickly. It covers the environment, PPO setup, data/shape contracts, survivor mapping, reward shaping, and important engineering considerations.

---

## Overview

This project simulates a growing 2D bacterial colony where each rod-shaped cell (a "StickCell") is an agent controlled by a shared neural policy trained with PPO. The environment models growth, division (with next-step replacement and jitter), simple physics (iterative overlap relaxation), and renders colony morphology. The main research question is how simple local policies can lead to emergent, colony-scale structure (e.g., anisotropy, shape stability, density control).

- Shared-policy, multi-agent RL (all cells use the same policy).
- Variable population size (agents divide; parents can be replaced by children).
- Observation per agent: 6 normalized features capturing internal state + local context.
- Discrete actions: Dormant (0), Grow (1), Divide (2).
- Reward: implemented growth-based global reward; optional morphology-based reward (commented in code) with AR/D/Fourier descriptors.

---

## Repository map (what to read first)

- `envs/colony_env.py` — Core environment `ColonyEnv` (Gymnasium API). Handles cell state, physics, observations, rewards, rendering.
- `agents/ppo_agent.py` — PPO agent: `SharedActorCritic` network, action sampling/evaluation, GAE, PPO update, `RolloutBuffer`.
- `experiments/ppo_train.py` — Training loop: rollout collection, value bootstrapping with dynamic populations, PPO updates, checkpointing.
- `evaluations/` — Scripts for rollouts, anisotropy experiments, manual viewers.
- `notebooks/` — Interactive demos/analysis.
- `ppo_checkpoints/` & `saved_checkpoints/` — Periodic/final model checkpoints.
- `GEMINI.md` and `README.md` — Research context and quick start.

---

## Environment at a glance (`envs/colony_env.py`)

### StickCell
- Fields: `pos ∈ R^2`, `theta` (radians), `length`, `age`, flags: `pending_divide`, `just_divided`.
- `endpoints()` returns segment endpoints for hull/morphology and rendering.

### Observation (per agent, 6D)
1. `rel_length` — normalized length vs. `L_divide` (clipped to [0, 1.25]).
2. `rel_age` — age / `max_steps` (0..1).
3. `orientation_sin` — sin(theta) for wrap-around stability.
4. `orientation_cos` — cos(theta).
5. `local_density` — Gaussian-kernel density estimate over neighbor distances.
6. `pressure_proxy` — mean inverse distance to neighbors within a cutoff.

Observation space: `Box(low=[0,0,-1,-1,0,0], high=[1.25,1,1,1,1,1], shape=(6,))`.

### Action (discrete)
- 0: Dormant
- 1: Grow (adds fixed `growth_rate` to length)
- 2: Divide (valid iff `length ≥ L_divide`; division applied next step with jitter)

### Reward (implemented default)
- Global colony-size reward: `min(N / max_cells, 1.0) * r_colony_size` added to each acting agent.
- Per-step existence penalty `p_living` (small negative).
- Division-related terms: `r_div_len`, `r_div_success`, penalties for invalid divide/growing too long (`p_invalid_div`, `p_too_long`).

### Reward (optional morphology-based, commented in code)
- Computes convex hull of all endpoints; PCA aspect ratio (AR), density (sum cell areas over hull area), Fourier boundary descriptors (F). Reward increases as metrics approach target `M_target = {AR, D, F}`.

### Episode termination
- `max_steps` reached (default 150), or
- `N ≥ max_cells`, or
- Empty population (unlikely, but treated as terminal).

### Rendering
- `render(mode="rgb_array", figsize=(w,h))` draws capsule cells, colony boundary; useful for diagnostics/visualization.

---

## PPO components (`agents/ppo_agent.py`)

### SharedActorCritic
- Encoder: MLP (hidden=128), ReLU + dropout; shared trunk.
- Heads:
  - Actor head: logits over 3 discrete actions.
  - Critic head: scalar value V(s).
- Forward returns `(logits, value)`.

### Action interface
- `agent.act(obs_tensor) -> (sampled_type, logp, values)` where `sampled_type` are int IDs [0..2].
- `make_action_dicts(type_tensor)` converts sampled types to Python `list[int]` compatible with `ColonyEnv.step()`.

### Experience buffer `RolloutBuffer`
- Flattens per-agent transitions across timesteps:
  - `obs`, `actions_type`, `rewards`, `values`, `logp`, `dones`, `next_values`.
- `compute_gae_advantages` computes GAE using `rewards`, `values`, `next_values`, `dones`.

### PPO update
- Clipped objective, value loss with coefficient, entropy bonus, gradient clipping, minibatch/epochs.

---

## Training loop (`experiments/ppo_train.py`)

- Defaults: `NUM_UPDATES=500`, `STEPS_PER_UPDATE=150`, `PPO_EPOCHS=4`, `MINIBATCH_SIZE=1024`, `LR=3e-4`, `GAMMA=0.99`, `LAM=0.95`.
- Auto-selects CUDA if available.
- Per update:
  1. Reset env; collect transitions until `STEPS_PER_UPDATE`.
  2. Build `next_values_for_acting_agents` using survivor mapping (see deep dive) and value blend for dividing parents.
  3. Push batches to buffer and run PPO update.
  4. Log stats every 10 updates; save checkpoints every 100 updates and at the end.

---

## Deep dive: survivor mapping and dynamic populations

Population size changes due to division, so `len(rewards)` (for the agents that acted) can differ from `len(next_obs)` (after division). To keep value targets consistent for GAE, the training loop maps parent indices to next-state values carefully.

Key points:
- `env.step(actions)` returns:
  - `rewards`: length = number of agents that acted (parents before division).
  - `next_obs`: shape `[M, 6]` after transition (survivors + children).
  - `info["survivor_indices"]`: indices (w.r.t. the previous timestep’s parent ordering) of agents that did NOT divide.
  - Convention: `next_obs` is ordered with survivors first, then newly created children.

Value bootstrap construction in training loop:
1. Compute baseline parent values: `parent_values = values` from the critic on the pre-step observations.
2. Initialize `next_values_for_acting_agents = parent_values.copy()` as a neutral fallback.
3. If episode continues (not done) and you have `next_obs` values:
   - Run critic on `next_obs` to get `next_values_np`.
   - Let `num_survivors = len(survivor_indices)`.
   - Assign survivors’ next values directly:
     - For each `i` in `survivor_indices`, set `next_values_for_acting_agents[i] = next_values_np[survivors_order_index]`.
     - Because survivors come first in `next_obs`, the slice `next_values_np[:num_survivors]` aligns to survivors.
   - For dividing parents (those not in `survivor_indices`):
     - Children are located after survivors in `next_obs`, i.e., `next_values_np[num_survivors:]`.
     - Compute `avg_child_value` as the mean of that slice (fallback to mean of all `next_values_np` if empty).
     - Blend with parent value to encourage value continuity across reproduction:
       - `next_values_for_acting_agents[parent_idx] = blend * avg_child_value + (1 - blend) * parent_values[parent_idx]`, with `blend ≈ 0.8` (tunable).

Example:
- Parents P0, P1, P2 act. P1 divides; P0, P2 survive.
- `survivor_indices = [0, 2]`; `num_survivors = 2`.
- `next_obs` order: [S(P0), S(P2), C1(P1), C2(P1)].
- `next_values_np[:2]` align to P0, P2; children are `next_values_np[2:]`.
- Assign P0, P2 directly; blend for P1 using average child value.

Edge cases:
- If episode terminates (`done=True`), keep `next_values_for_acting_agents = parent_values` or zeros depending on your GAE convention.
- If `child_values` slice is empty (shouldn’t happen in normal division), fallback to mean of all `next_values_np` or 0.0.
- Always ensure the length of `next_values_for_acting_agents` equals the number of acting parents.

Why this matters:
- Prevents value target discontinuities when parents disappear and are replaced by children.
- Encourages the policy to learn sensible division timing because parent returns reflect children’s future prospects.

---

## Deep dive: reward shaping variants

Implemented (simple, default):
- `global_reward = min(N / max_cells, 1.0) * r_colony_size` added to each agent’s step reward.
- `p_living` small negative per step.
- Division-specific terms: `r_div_len` (reward for reaching division length), `r_div_success` (reward when division occurs), penalties `p_invalid_div`, `p_too_long`.

Morphology-based (commented in code):
- Metrics from colony geometry:
  - Aspect Ratio (AR) via PCA on endpoints.
  - Density (D) = sum(cell areas) / convex hull area.
  - Fourier descriptors (F) of hull boundary (length `fourier_K`).
- Error to target: `err_AR`, `err_D`, `err_F`; global reward roughly `1 / (1 + w_AR*err_AR + w_D*err_D + w_F*err_F)`.
- To enable: replace the simple size-based section in `_compute_rewards` with the morphology block and tune weights.

Additional shaping ideas (optional):
- Local density moderation: small negative reward for high `local_density` beyond a threshold.
- Pressure penalty: penalize high `pressure_proxy` to reduce overcrowding.
- Division spacing bonus: bonus if division happens when local density is low.
- Orientation alignment: encourage neighbors to align (anisotropy) by rewarding cos(Δθ) with nearest neighbors (use carefully to avoid collapse).
- Curriculum: start with size-based reward, gradually mix in morphology terms over updates.

Shaping pitfalls and tips:
- Scale rewards to similar magnitudes to avoid dominance (e.g., standardize by moving averages).
- Beware reward hacking (e.g., oscillating divide/grow without structural improvement).
- Keep components smooth and differentiable with respect to agent’s influence where possible.

---

## Other important aspects and best practices

- Variable population handling:
  - Always supply `len(actions) == len(current_cells)` to `env.step()`.
  - Expect `len(next_obs)` to differ from `len(rewards)` after division.
  - Use `info["survivor_indices"]` for value alignment.

- GAE and bootstrapping:
  - Use the survivor mapping and child blend as described to compute `next_values`.
  - If `done=True`, set `next_values=0` or use `parent_values` per your convention (consistent handling is key).

- Physics relaxation:
  - `_relax_positions(max_iters=12)` pushes overlapping capsules apart; increasing iterations improves realism but costs compute.
  - Consider capping growth if persistent overlaps remain to avoid instability.

- Observation normalization:
  - Keep the observation space bounds consistent with features. If you add features, update `observation_space` accordingly.

- Action extensions:
  - If introducing continuous controls (e.g., growth fraction, steering), add heads to the actor, squash with `tanh`/`sigmoid`, and update env/action parsing.

- Checkpointing and resume:
  - Periodic checkpoints at `ppo_checkpoints/ppo_colony_{update}.pt`; final at `ppo_checkpoints/ppo_colony_final.pt`.
  - Store optimizer state for true resumption.

- Reproducibility:
  - Seed env (`ENV_SEED`), PyTorch, and NumPy consistently for experiments.

- Performance:
  - Batch critic evaluations (already done); keep tensors on device.
  - Profile render-heavy workflows; skip rendering during training for speed.

- Troubleshooting:
  - Mismatch lengths (actions vs. cells): ensure action list length equals current number of cells.
  - Stalled learning (flat entropy, no growth): lower `value_coef`, adjust LR, increase `entropy` bonus slightly, or warm-start with simpler rewards.
  - Value/advantage instability: verify survivor mapping indices and child blending; print shapes and indices when debugging.
  - Exploding gradients: enable gradient clipping (already on) and consider smaller LR.

---

## Minimal contracts (for integration/testing)

- Step input: `actions: list[int]` with length equal to current number of cells.
- Step output:
  - `next_obs: float32 ndarray [M, 6]`
  - `rewards: float list length N` (pre-division agents)
  - `terminated: bool`, `truncated: bool`
  - `info: { "n_cells": int, "survivor_indices": list[int] }`

Success criteria:
- Under size-based reward: increasing population size toward `max_cells`; stable PPO metrics (policy/value loss trending down, entropy tapering gradually).
- With morphology reward: metrics (AR/D/F) approach target values; colony shapes reflect desired morphology.

---

## Quick start

Training (from repo root):

```bat
python experiments\ppo_train.py
```

- Uses CUDA if available.
- Saves checkpoints in `ppo_checkpoints/`.
- Logs every 10 updates (rewards, cells, losses, entropy).

---

## Extending the project

- Add morphology reward: enable/comment-in the block in `_compute_rewards` and tune weights.
- New observation features: implement in `_obs_for_cell` and revise `observation_space` bounds.
- Continuous actions: add parameter heads in `SharedActorCritic`, squash outputs, and update `make_action_dicts` + env step parsing.
- Multi-colony batching: wrap multiple `ColonyEnv` instances and concatenate batches for throughput.

---

## File references

- Environment: `envs/colony_env.py` (`ColonyEnv`, `_gather_obs`, `_obs_for_cell`, `_compute_rewards`, `_relax_positions`, `render`)
- Agent: `agents/ppo_agent.py` (`SharedActorCritic`, `PPOAgent`, `RolloutBuffer`, `compute_gae_advantages`, `make_action_dicts`)
- Training: `experiments/ppo_train.py` (`main` training loop and survivor mapping logic)
- Context: `GEMINI.md`, `README.md`

---

If you’d like, I can add a short link in `README.md` pointing to this guide.
