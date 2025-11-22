# GEMINI.md: Project Overview of ColonyEnv

## 🧠 Project Overview: “ColonyEnv: Multi-Agent RL with Stick-Cell Colonies”

**Purpose:** This project simulates the collective growth of a bacterial colony in a continuous 2D space. Each bacterium, modeled as a `StickCell`, acts as an autonomous reinforcement learning agent. All agents share a common neural policy trained with PPO. The environment is designed to study how simple, local rules can give rise to emergent, colony-scale structures such as anisotropy, shape stability, and coordinated growth.

**Research Goal:** The core objective is to explore how microscopic decision-making (simple RL rules at the cell level) can yield macroscopic order, creating a computational analog to biological morphogenesis. The experiment aims to answer questions such as:
- Do shared-policy cells spontaneously form anisotropic structures?
- Can global colony morphology emerge without explicit coordination?
- How does mechanical feedback (pressure, density) influence optimal policies?

---

## ⚙️ Core Components

This project is primarily composed of three Python modules:

1.  **`envs/colony_env.py`**: Implements the main `ColonyEnv` using the Gymnasium API. It manages the simulation state, cell dynamics (growth, division, physics), observation gathering, and reward computation.
2.  **`agents/ppo_agent.py`**: Defines the PPO agent, including the `SharedActorCritic` neural network, the `PPOAgent` class that handles updates, and the `RolloutBuffer` for experience collection.
3.  **`experiments/ppo_train.py`**: The main training script that orchestrates the entire process. It initializes the environment and the agent, runs the training loop, collects experience, performs PPO updates, and saves model checkpoints.

---

## 🌿 Biological & Conceptual Basis

-   **Agents (`StickCell`)**: Each agent represents a single rod-shaped bacterial cell with a defined position, orientation, and length in a continuous 2D plane.
-   **Shared Policy**: All cells act based on a single, shared policy network (`SharedActorCritic`), promoting the learning of cooperative or competitive behaviors from local observations.
-   **Dynamics**: The simulation proceeds in discrete timesteps. Growth and division cause mechanical pushes between neighboring cells, which are resolved through an iterative physics relaxation step (`_relax_positions`).
-   **Emergent Goal**: The primary experiment is to observe whether these shared-policy agents can collectively produce realistic colony morphologies (e.g., circular, branching, or anisotropic growth) purely from learned local behaviors.

---

## 👁️ Observation Space (Per-Agent)

Each agent's observation is a 6-dimensional vector containing normalized, continuous features that capture its local context:

1.  **`rel_length`**: The cell's current length, normalized relative to the division length (`L_divide`).
2.  **`rel_age`**: The cell's current age, normalized by the maximum episode steps.
3.  **`orientation_sin`**: The sine of the cell's orientation angle (`theta`).
4.  **`orientation_cos`**: The cosine of the cell's orientation angle (`theta`).
5.  **`local_density`**: A measure of local crowding, computed using a Gaussian kernel over distances to neighboring cells.
6.  **`pressure_proxy`**: The mean inverse distance to nearby neighbors, serving as a proxy for mechanical pressure.

---

## 🕹️ Action Space

At each timestep, every cell chooses one of three discrete actions:

-   **`0: Dormant`**: Do nothing.
-   **`1: Grow`**: Increase in length by a fixed `growth_rate`.
-   **`2: Divide`**: If the cell's length exceeds the `L_divide` threshold, mark it for division. The cell is replaced by two new daughter cells in the next timestep, with slight random jitter in their orientation.

---

## 💎 Reward Function

The reward system is designed to guide the colony's growth.

-   **Implemented Reward**: The current implementation in `colony_env.py` uses a simple, effective reward structure:
    -   **Global Reward**: A shared reward proportional to the total number of cells (`min(N / max_cells, 1.0)`), encouraging population growth.
    -   **Existence Penalty**: A small penalty (`-0.01`) applied to all cells at each step to encourage efficient growth and division.
    -   **Division Bonus**: A bonus (`+0.5`) given to cells that have just been created through division.

-   **Planned Morphological Reward (Commented Out)**: The code also contains a more complex, commented-out reward system based on global morphology metrics, which could be used for future experiments:
    -   **Aspect Ratio (AR)**: Measures the elongation of the colony.
    -   **Density (D)**: The ratio of total cell area to the area of the colony's convex hull.
    -   **Fourier Descriptors (F)**: Captures the shape of the colony's boundary.

---

## 🤖 RL Setup

-   **Algorithm**: Proximal Policy Optimization (PPO), implemented in `agents/ppo_agent.py`. The implementation includes:
    -   Generalized Advantage Estimation (GAE).
    -   Clipped surrogate objective for policy updates.
    -   Clipped value loss to stabilize critic training.
    -   Entropy bonus for exploration.
-   **Network**: A `SharedActorCritic` model with a simple MLP architecture (64-unit hidden layers) serves as the policy and value function for all agents.
-   **Framework**: The environment is built with `Gymnasium` for compatibility with standard RL libraries.
-   **Dynamic Population**: The environment and training loop are designed to handle a variable number of agents, as cells divide and the population grows.

---

## 🚀 How to Run

The training process is managed by the `ppo_train.py` script.

**To start training:**
```bash
python experiments/ppo_train.py
```

-   The script will automatically use a CUDA-enabled GPU if available.
-   Training progress, including losses, rewards, and cell counts, will be printed to the console.
-   Model checkpoints are saved periodically to the `ppo_checkpoints/` directory.

### Key Hyperparameters (`ppo_train.py`)
-   **`NUM_UPDATES`**: 500 (Total number of policy updates)
-   **`STEPS_PER_UPDATE`**: 150 (Environment steps collected per update)
-   **`PPO_EPOCHS`**: 4 (Number of optimization epochs per batch)
-   **`MINIBATCH_SIZE`**: 1024
-   **`GAMMA`**: 0.9 (Discount factor)
-   **`LAM`**: 0.95 (GAE lambda parameter)
