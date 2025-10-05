# ColonyEnv: Environment Design Document

This document provides a detailed description of `ColonyEnv`, a multi-agent Gymnasium environment designed for simulating the growth of a bacterial colony.

## 1. Overview

`ColonyEnv` simulates a colony of capsule-shaped bacterial cells in a continuous 2D space. Each cell is an independent agent that makes decisions to grow, rotate, and divide. The collective goal is to guide the colony's growth to match a predefined target morphology. The environment is designed to be compatible with standard reinforcement learning libraries and uses a shared policy approach where all agents (cells) are controlled by the same policy network.

Key features include:
- **Continuous Space**: Cells exist and move in a continuous 2D world.
- **Dynamic Agent Population**: The number of agents changes as cells divide.
- **Physics-based Interaction**: A simple iterative relaxation model resolves physical overlaps between cells.
- **Complex Observation Space**: Each agent observes its own state and the relative states of its nearest neighbors.
- **Hybrid Action Space**: Agents can take discrete actions (grow, divide) and apply continuous modulation (torque, growth amount).
- **Morphology-based Rewards**: The reward function is based on global properties of the colony's shape.

## 2. Environment Setup

### 2.1. World and Cell Properties

The environment is initialized with several key parameters that define the physical properties of the world and the cells within it:

- `world_size`: A tuple `(width, height)` defining the boundaries of the 2D world.
- `r`: The radius of the capsule cells. This is constant for all cells.
- `L_init`: The initial length of the first cell when the environment is reset.
- `L_divide`: The length threshold. A cell must be at least this long to be able to divide.
- `max_cells`: The maximum number of cells allowed in the colony. If this number is reached, the episode terminates.

### 2.2. Initial State

An episode begins by calling the `reset()` method. This initializes the environment with a single `CapsuleCell` placed at the center of the world. The cell has an initial length of `L_init` and a default orientation.


## 3. Markov Decision Process (MDP) Formulation

The environment is modeled as a multi-agent MDP. Here are its components:

### 3.1. Observation Space

The observation space is defined for a single agent and is represented by a `spaces.Box`. It is a flat vector concatenating features about the agent itself and its `K_nn` nearest neighbors.

- **Dimension**: `5 + K_nn * 5`
- **Structure**: `[self_features, neighbor_1_feature1, ..., neighbor_K_feature5]`

**Self Features (5 dimensions):**
1.  `L_norm`: The cell's current length, normalized by `L_divide`.
2.  `sin(theta)`: Sine of the cell's orientation angle.
3.  `cos(theta)`: Cosine of the cell's orientation angle.
4.  `age_norm`: The cell's age (in timesteps), normalized by a constant factor.
5.  `local_density`: A measure of how many other cells are within a certain radius, normalized.

**Neighbor Features (5 dimensions per neighbor):**
For each of the `K_nn` nearest neighbors:
1.  `rel_x_norm`: The x-component of the normalized direction vector from the agent to the neighbor.
2.  `rel_y_norm`: The y-component of the normalized direction vector.
3.  `dist_norm`: The distance to the neighbor, normalized by the world size.
4.  `sin(theta_neighbor)`: Sine of the neighbor's orientation.
5.  `cos(theta_neighbor)`: Cosine of the neighbor's orientation.

If a cell has fewer than `K_nn` neighbors, the feature vectors for the missing neighbors are padded with zeros.

### 3.2. Action Space

The action space is a `spaces.Dict` that allows for a hybrid of discrete and continuous actions.

- **`type` (`spaces.Discrete(3)`):** The primary discrete action.
    - `0`: **No-op**: The cell does nothing.
    - `1`: **Grow**: The cell increases its length.
    - `2`: **Divide**: The cell signals its intent to divide. The division will only occur if `cell.L >= L_divide`.

- **`grow_frac` (`spaces.Box(0.0, 1.0)`):** A continuous parameter that modulates the amount of growth if the `type` is `1`. The actual length increase is `max_growth * grow_frac`.

- **`torque` (`spaces.Box(-1.0, 1.0)`):** A continuous parameter that applies rotational force to the cell. The actual angle change is `max_rot * torque`.

### 3.3. Reward Function

The reward function is designed to encourage the colony to adopt a target morphology. It is composed of a global, shared reward and small, individual penalties.

**Global Reward (`R_morph`):**
This reward is shared among all agents. It is calculated based on the negative distance between the current colony's morphology and a target morphology (`M_target`). The morphology is quantified by three metrics:

| Metric | Description | Target Value (chosen randomly) | Notes|
| :--- | :--- | :--- |:---|
| **Aspect Ratio (AR)** | Calculated via PCA on the cell centers, this measures the colony's elongation. | `1.0` | |
| **Density (D)** | The number of cells divided by the area of the colony's convex hull. | `0.02` | |
| **Fourier Descriptors (F)** | A set of `fourier_K` values that describe the shape of the colony's boundary in the frequency domain. | An array of `fourier_K` (default 8) zeros. | weigth set to 0 for now |


The global reward is `R_morph = -d_morph`, where `d_morph` is the sum of normalized differences between the current and target metrics.

**Per-Agent Penalties:**
Each agent receives small individual penalties to guide its local behavior:
- **Length Penalty**: A penalty proportional to how much the cell's length deviates from the ideal division length (`L_divide`). This encourages cells to grow to the right size and then divide.
- **Age Penalty**: A small penalty for age, encouraging division to reset the age counter.

**Total Reward:**
The final reward for each agent is the sum of its individual penalties and a fraction of the global morphology reward.

### 3.4. Episode Termination

An episode ends under one of two conditions:
- **Terminated**: The number of cells in the colony reaches `max_cells`.
- **Truncated**: The number of timesteps `t` reaches a predefined limit (e.g., 1000).

## 4. Simulation Dynamics (The `step` function)

A single timestep in the environment involves a sequence of deterministic and stochastic updates:

1.  **Action Application**: The actions provided by the policy for each cell are applied.
    - The cell's orientation `theta` is updated based on the `torque` value.
    - If the action `type` is `1` (Grow), the cell's length `L` is increased based on `grow_frac`.
    - If the action `type` is `2` (Divide) and the cell is long enough, its `pending_divide` flag is set to `True`.
    - The `age` of every cell is incremented.

2.  **Physics Relaxation (`_relax_positions`)**: The environment resolves physical overlaps between cells. This is done iteratively:
    - For every pair of cells, the algorithm checks if their capsule bodies overlap (i.e., if the distance between their central axes is less than `2 * r`).
    - If an overlap is detected, the two cells are pushed apart along the vector connecting their closest points. Each cell moves by half the overlap distance.
    - This process is repeated for a fixed number of iterations (`max_iters`) or until no overlaps are detected in a full pass.

3.  **Cell Division**: The environment processes all cells with `pending_divide = True`.
    - The parent cell is removed from the simulation.
    - Two new daughter cells are created. They are positioned at opposite ends of where the parent cell was, with a small separation.
    - A small amount of random "jitter" is added to the orientation of the daughter cells to introduce stochasticity and break perfect symmetry.

4.  **Final Relaxation**: After new cells are added, the relaxation process is run again to resolve any new overlaps created by the daughter cells.

5.  **State Update**: After all dynamics are resolved, a new observation is gathered for each agent, and rewards are computed based on the final state of the colony for that timestep.

## 5. Rendering

The environment can be visualized using the `render()` method, which supports two modes:
- `mode="human"`: Displays a plot of the colony using `matplotlib`. Cell orientations are indicated by small arrows.
- `mode="rgb_array"`: Returns a NumPy array representing an RGB image of the current environment state, which can be used for creating videos or for pixel-based policies.
