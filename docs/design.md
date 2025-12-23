# ColonyEnv: Environment Design Document

This document provides a detailed description of `ColonyEnv`, a multi-agent Gymnasium environment designed for simulating the growth of a bacterial colony.

## 1. Overview

`ColonyEnv` simulates a colony of capsule-shaped bacterial cells in a continuous 2D space. Each cell is an independent agent that makes decisions to grow or divide. The collective goal is to guide the colony's growth to maximize population size and achieve specific morphological properties (e.g., high aspect ratio). The environment is designed to be compatible with standard reinforcement learning libraries and uses a shared policy approach where all agents (cells) are controlled by the same policy network.

Key features include:
- **Continuous Space**: Cells exist and move in a continuous 2D world.
- **Dynamic Agent Population**: The number of agents changes as cells divide.
- **Physics-based Interaction**: A simple iterative relaxation model resolves physical overlaps between cells, simulating mechanical pressure and pushing forces.
- **Aggregated Observation Space**: Each agent observes its own state and aggregated statistics of its local neighborhood (density, pressure).
- **Discrete Action Space**: Agents take simple discrete actions (dormant, grow, divide).
- **Hybrid Reward System**: The reward function combines individual incentives (growth, division) with global colony-level objectives (size, aspect ratio).

## 2. Environment Setup

### 2.1. World and Cell Properties

The environment is initialized with several key parameters that define the physical properties of the world and the cells within it:

- `world_size`: A tuple `(width, height)` defining the boundaries of the 2D world (default: `(64.0, 64.0)`).
- `L_init`: The initial length of the first cell when the environment is reset (default: `1.0`).
- `L_divide`: The length threshold. A cell must be at least this long to be able to divide (default: `2.0`).
- `max_cells`: The maximum number of cells allowed in the colony. If this number is reached, the episode terminates (default: `80`).
- `K_nn`: The number of nearest neighbors considered for local density calculations (default: `6`).

### 2.2. Initial State

An episode begins by calling the `reset()` method. This initializes the environment with a single `StickCell` placed at the center of the world. The cell has an initial length of `L_init` and a default orientation.

## 3. Markov Decision Process (MDP) Formulation

The environment is modeled as a multi-agent MDP. Here are its components:

### 3.1. Observation Space

The observation space is defined for a single agent and is represented by a `spaces.Box` of shape `(6,)`. It contains normalized features representing the agent's internal state and its local context.

**Features (6 dimensions):**
1.  `rel_length`: The cell's current length, normalized by `L_divide` and clipped to `[0, 1.25]`.
2.  `rel_age`: The cell's age, normalized by `max_steps` (150).
3.  `orientation_sin`: Sine of the cell's orientation angle $\theta$.
4.  `orientation_cos`: Cosine of the cell's orientation angle $\theta$.
5.  `local_density`: A measure of local crowding, computed using a Gaussian kernel over distances to `K_nn` nearest neighbors. Normalized to `[0, 1]`.
6.  `pressure_proxy`: The mean inverse distance to neighbors within a cutoff radius ($2 \times L_{divide}$). Normalized to `[0, 1]`.

### 3.2. Action Space

The action space is `spaces.Discrete(3)`, representing three mutually exclusive behaviors:

- `0`: **Dormant**: The cell does nothing.
- `1`: **Grow**: The cell increases its length by a fixed amount (`growth_rate * dt`).
- `2`: **Divide**: The cell attempts to divide. This is valid only if `cell.length >= L_divide`.

### 3.3. Reward Function

The reward function is designed to encourage a healthy, growing colony with specific morphological characteristics. It is a sum of individual and global components.

**Individual Components:**
- **Growth Reward (`r_grow`)**: Small positive reward (+0.005) for choosing the Grow action.
- **Readiness Bonus (`r_div_len`)**: Bonus (+0.2) when a cell grows past `L_divide`.
- **Division Success (`r_div_success`)**: Large bonus (+3.0) for successfully initiating a division.
- **Invalid Division Penalty (`p_invalid_div`)**: Penalty (-0.001) for attempting to divide when too short.
- **Living Penalty (`p_living`)**: Small negative reward (-0.0001) per step to encourage efficiency.
- **Overgrowth Penalty (`p_too_long`)**: Penalty (-0.02) if a cell grows significantly beyond `L_divide` (e.g., > 1.2x), discouraging "giant" cells.

**Global Components:**
- **Colony Size Reward**: A shared reward proportional to the total population size, scaled by `r_colony_size` (2.0).
- **Morphology Reward (Aspect Ratio)**: A shared reward based on the change in the colony's Aspect Ratio (AR).
    - `delta = |prev_AR - target_AR| - |current_AR - target_AR|`
    - If `delta > 0` (colony shape improved towards target), a reward of `AR_delta_coef * delta` is added.

### 3.4. Episode Termination

An episode ends under one of two conditions:
- **Terminated**: The number of cells in the colony reaches `max_cells`.
- **Truncated**: The number of timesteps `t` reaches `max_steps` (150).

## 4. Simulation Dynamics (The `step` function)

A single timestep in the environment involves the following sequence:

1.  **Action Application**:
    - **Grow**: Cell length increases by `growth_rate * dt`.
    - **Divide**: If valid, the cell is marked `pending_divide`.
    - **Dormant**: No change.
    - Age is incremented for all cells.

2.  **Cell Division**:
    - Cells marked `pending_divide` are removed.
    - Two daughter cells are created at the parent's position, offset slightly along the parent's axis.
    - Daughter cells inherit half the parent's length.
    - A random "jitter" (Gaussian noise) is added to the daughter cells' orientations to break symmetry.

3.  **Physics Relaxation (`_relax_positions`)**:
    - The environment resolves physical overlaps iteratively (default 12 iterations).
    - Overlapping cells push each other apart linearly.
    - Overlaps also induce rotation (torque) on the cells, simulating physical contact forces.
    - This step is crucial for the emergent arrangement of the colony.

4.  **State Update**:
    - New observations are gathered.
    - Global metrics (Aspect Ratio) are re-calculated.
    - Rewards are computed.

## 5. Rendering

The environment supports visualization via `matplotlib`:
- `mode="human"`: Opens an interactive window showing the colony in real-time.
- `mode="rgb_array"`: Returns a NumPy array of the rendered frame.

**Visual Encoding:**
- **Blue**: Normal cells.
- **Red**: Cells pending division (will divide next step).
- **Green**: Cells that just divided (newly created).
- **Lines**: Represent the cell bodies.
- **Dots**: Represent cell centers.
