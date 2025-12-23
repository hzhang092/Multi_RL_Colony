# Multi_RL_Colony

Multi-agent reinforcement learning for simulated bacterial colonies, built around a custom Gymnasium environment **ColonyEnv**. Each rod-shaped cell is an RL agent that decides when to grow or divide based only on local observations; global colony morphology then emerges from these decentralized policies.

This repository accompanies the CS686 project report *"Multi-Agent RL with Authentic Colony Features"*.

## Overview

- **Environment**: 2D continuous world with a dynamic population of rod-shaped agents.
- **Agents**: Each bacterium is an agent with position, orientation, length, and age.
- **Actions**: Discrete action space \(A = \{0: \text{Dormant}, 1: \text{Grow}, 2: \text{Divide}\}\).
- **Physics**: Simple overlap relaxation with repulsive forces and limited torque (no friction or adhesion).
- **Algorithm**: Proximal Policy Optimization (PPO) with a **shared policy** across all agents.
- **Goal**: Study how local RL policies give rise to global colony properties (e.g., aspect ratio, anisotropy) and how reward shaping interacts with physical constraints.

High-level findings from the project:

- Agents learn **biologically plausible division timing** (divide near a length threshold \(L_{divide}\)).
- Pure morphology rewards can be **exploited** (agents grow into "giant cells" if length is unconstrained).
- Enforcing realistic division (length penalties) prevents exploitation but makes colony aspect ratio stay near a baseline circular morphology, highlighting the importance of the underlying physics (lack of friction/adhesion).

## Key Components

- `envs/colony_env.py` – Implementation of **ColonyEnv**, including:
	- Agent representation (rod-shaped capsules).
	- Physics / collision resolution by iterative overlap relaxation.
	- Observation construction (6D feature vector per agent).
	- Reward computation (local + global terms).
- `envs/utilities/geo_helpers.py` – Geometric helpers for capsule/segment operations.
- `agents/ppo_agent.py` – PPO implementation with a shared policy network (actor–critic MLP).
- `experiments/ppo_train.py` – Main training script for PPO on ColonyEnv.
- `experiments/ppo_shared_policy.py` – Additional shared-policy experiments / configurations.
- `experiments/visualize_training.py` – Utilities for visualizing training curves.
- `evaluations/` – Analysis scripts:
	- `policy_action_inspector.py` – Plots action distributions vs. observation features (e.g., relative length).
	- `morphology_runs.py` – Runs trained policies to measure colony aspect ratio under different reward settings.
	- `rollout_view.py`, `manual_env_viewer.py` – Visual inspection and debugging tools for rollouts.

Saved models and experiment outputs:

- `ppo_checkpoints/` – Checkpoints from baseline PPO training (e.g., `ppo_colony_final.pt`).
- `saved_checkpoints/` – Checkpoints for individual experiment runs (with timestamp/ID suffixes).
- `evaluations/results/` – CSV summaries of morphology experiments.
- `evaluations/plots/` – Generated plots (policy inspector, morphology curves, training curves, etc.).

The full project report (LaTeX source) is in `reports/` (e.g., `project_report.tex`, `report3.tex`).

## ColonyEnv Details (Summary)

**Observation (per agent, 6D)**

- Relative length: \(L_i / L_{divide}\), clipped to \([0, 1.25]\).
- Relative age: \(t_i / T_{max}\), clipped to \([0, 1]\).
- Orientation: \(\sin \theta_i, \cos \theta_i\).
- Local density: Gaussian-weighted sum over nearest neighbors.
- Pressure proxy: Inverse-distance-based average over nearest neighbors.

**Reward structure**

- Local reward \(R_{local}^i\):
	- Positive reward for valid **Grow** and **Divide** actions.
	- Small existence penalty to encourage efficient life cycles.
	- Strong penalty if a cell becomes **too long** (prevents giant-cell exploit).
- Global reward \(R_{global}\):
	- Encourages colony size up to a maximum cell count.
	- Morphology term that rewards aspect ratios close to a target \(AR_{target}\).
	- Delta-improvement term that rewards steps which move the aspect ratio closer to the target.

Global terms are shared among all acting agents in a step to stabilize credit assignment in a variable-population setting.

## PPO Training Setup

- Shared-policy PPO (homogeneous multi-agent): one network used by all cells.
- Two-layer MLP encoder (128 hidden units) with separate actor and critic heads.
- Advantage estimation: GAE with \(\gamma = 0.99, \lambda = 0.95\).
- Training schedule used in the report (typical configuration):
	- 500 PPO updates, ~300 environment steps per update (~150k steps total).
	- Mini-batch PPO epochs over collected rollouts.
	- Max steps per episode ~150, max cells ~80.

For more complete experimental details (including hyperparameters, ablations, and plots), see `reports/report3.tex`.

## Getting Started

### Prerequisites

- Python 3.9+ recommended.
- PyTorch (for PPO implementation).
- `gymnasium` and common scientific Python packages (NumPy, Matplotlib, etc.).

A minimal installation might look like:

```bash
pip install torch gymnasium numpy matplotlib
```

You may need additional packages depending on your environment and plotting backend.

### Running Training

From the repo root:

```bash
cd experiments
python ppo_train.py
```

This will train a PPO agent on ColonyEnv and save checkpoints under `ppo_checkpoints/` and logs under `logs/`.

### Inspecting Policies and Morphology

After training, you can reproduce key analyses from the report:

- Action–feature analysis (division timing, policy structure):

```bash
cd evaluations
python policy_action_inspector.py
```

- Colony morphology runs (aspect ratio under different reward settings):

```bash
cd evaluations
python morphology_runs.py
```

- Visual rollouts / manual environment viewer:

```bash
cd evaluations
python manual_env_viewer.py
```

Adjust script arguments or hard-coded paths to point to the desired checkpoint in `saved_checkpoints/` or `ppo_checkpoints/`.

## Reproducing Report Figures

The LaTeX report includes several plots and tables that can be regenerated from this codebase:

- **Policy action distributions / decision maps** ("policy inspector" figure):
	- Edit the CONFIG block in `evaluations/policy_action_inspector.py` (especially `CHECKPOINT_NAME`, `CHECKPOINT_PATH`, and `PLOTS_DIR`).
	- From the repo root, run:

		```bash
		python evaluations/policy_action_inspector.py
		```

	- Plots (feature histograms and 2D decision maps) will be saved under `evaluations/plots/<CHECKPOINT_NAME>_policy_inspector/`.

- **Colony aspect ratio / anisotropy statistics** (tables summarizing AR ± SD):
	- Edit the CONFIG block in `evaluations/morphology_runs.py` (or the anisotropy/morphology script you are using) to select the checkpoint and number of runs.
	- Run:

		```bash
		python evaluations/morphology_runs.py
		```

	- CSV summaries (including mean aspect ratio and anisotropy) are written to `evaluations/results/`, and can be used to recreate the tables in the report.

- **Simulation frames / colony snapshots**:
	- To obtain single frames like those shown in the report, edit `CHECKPOINT_PATH` and other CONFIG options in `evaluations/rollout_view.py`.
	- Run:

		```bash
		python evaluations/rollout_view.py
		```

	- PNG frames will be saved under `evaluations/frames/`; you can select representative frames to include in documents.

- **Training curves**:
	- Set `FILENAME` in `experiments/visualize_training.py` to one of the log files in `logs/` (e.g., the run used in the report).
	- Run:

		```bash
		python experiments/visualize_training.py
		```

	- This script parses the log and displays training metrics (average reward, returns, losses, etc.); you can save the resulting Matplotlib figure as `training_plot.png` to match the report.

## Limitations and Future Work (from the Report)

- Reward shaping alone cannot reliably produce high-aspect-ratio colonies when physics is purely repulsive; the colony tends to relax toward circular shapes.
- Achieving filamentous or strongly anisotropic colonies likely requires:
	- Friction and cell-to-cell adhesion.
	- Elastic bending or other mechanical constraints.
	- Possibly richer observations (e.g., local anisotropy or global axis information).

Future directions include curriculum learning, multi-objective optimization over colony features, and representation learning for richer per-cell embeddings.

## References and Resources

- Project report: see `reports/` for LaTeX sources and bibliography.
- Code for ColonyEnv and training scripts (this repository) is also linked at:
	- https://github.com/hzhang092/Multi_RL_ColonyEnv
- Gymnasium tutorial used as an initial reference:
	- [Collection of Python code that solves the Gymnasium Reinforcement Learning environments, along with YouTube tutorials](https://github.com/johnnycode8/gym_solutions?tab=readme-ov-file#beginner-reinforcement-learning-tutorials)
