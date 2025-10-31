"""
Run ColonyEnv with random actions for multiple episodes, stop when target cell
count is reached (default 80), and record the mean local anisotropy.

Output:
- Prints per-run mean anisotropy and a final summary (mean ± std)
- Saves results to CSV (optional)

How to use (no argparse):
1) Edit the CONFIG section below.
2) Run:
       python evaluations\\anisotropy_random_runs.py
"""

import os
import sys
from pathlib import Path
from typing import Optional, Any

import numpy as np
import csv
import torch

# Add the project root to the Python path to allow for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.colony_env import ColonyEnv
from envs.utilities.geo_helpers import get_local_anisotropy
try:
    from agents.ppo_agent import PPOAgent, make_action_dicts
except Exception:
    PPOAgent = None
    make_action_dicts = None


# ======================
# CONFIG (edit these)
# ======================
SEED: int = 686
NUM_RUNS: int = 20
TARGET_CELLS: int = 80            # Stop an episode when this many cells exist
MAX_STEPS_PER_RUN: int = 200     # Safety cap on steps per run

# Neighbourhood range used in local anisotropy (distance threshold)
# A reasonable default is tied to division length; final value computed at runtime.
NEIGHBORHOOD_MULTIPLIER: float = 3.0   # neighbourhood_range = multiplier * env.L_divide

# Save CSV results
SAVE_CSV: bool = False
CSV_PATH: str = "evaluations/anisotropy_random_runs.csv"

# Policy config (set USE_TRAINED=True to use a checkpoint)
USE_TRAINED: bool = True
CHECKPOINT_PATH: str = "saved_checkpoints/ppo_colony_final-1013-3.pt"  # used only if USE_TRAINED=True
DETERMINISTIC: bool = False   # True: argmax over logits; False: stochastic sampling
DEVICE: Optional[str] = None # None auto-selects; set to 'cpu' to force CPU


def _fallback_make_action_dicts(action_types: np.ndarray):
    return action_types.astype(int).tolist()


def load_agent(checkpoint_path: str, obs_dim: int) -> Optional[Any]:
    if not USE_TRAINED:
        return None
    if PPOAgent is None:
        raise RuntimeError("PPOAgent not available. Check agents/ppo_agent.py import.")
    device = torch.device(DEVICE) if isinstance(DEVICE, str) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(obs_dim=obs_dim, device=device)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'policy_state_dict' in ckpt:
            agent.policy.load_state_dict(ckpt['policy_state_dict'])
        else:
            agent.policy.load_state_dict(ckpt)
    agent.policy.eval()
    return agent


def _select_actions(obs: np.ndarray, agent: Optional[Any]) -> list:
    if agent is None:
        action_types = np.random.randint(0, 3, size=(len(obs),), dtype=np.int64)
        if make_action_dicts is not None:
            at_t = torch.tensor(action_types, dtype=torch.int64)
            return make_action_dicts(at_t)
        return _fallback_make_action_dicts(action_types)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=getattr(agent, 'device', None))
    with torch.no_grad():
        if DETERMINISTIC:
            logits, _ = agent.policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            sampled_type = torch.argmax(probs, dim=-1)
        else:
            sampled_type, _, _ = agent.act(obs_t)
    if make_action_dicts is not None:
        return make_action_dicts(sampled_type)
    return sampled_type.detach().cpu().numpy().astype(int).tolist()


def run_single_episode(env: ColonyEnv, target_cells: int, max_steps: int, neighbourhood_range: float, agent: Optional[Any]) -> float:
    """Run one random-policy rollout until target cell count or step cap.

    Returns the mean local anisotropy over all cells at the stopping point.
    """
    obs, _ = env.reset()
    steps = 0
    while steps < max_steps:
        # Select actions (trained or random)
        if len(obs) == 0:
            # no agents -> reset state and continue
            obs, _ = env.reset()
            steps = 0
            continue
        actions = _select_actions(obs, agent)
        obs, rewards, terminated, truncated, info = env.step(actions)
        steps += 1

        if len(env.cells) >= target_cells or terminated or truncated:
            # Compute mean anisotropy
            centers = np.array([c.pos for c in env.cells], dtype=float)
            thetas = np.array([c.theta for c in env.cells], dtype=float)
            la = get_local_anisotropy(centers, thetas, neighbourhood_range)
            return float(np.mean(la)) if la.size > 0 else 0.0
    # If we exit due to step cap, compute anyway
    centers = np.array([c.pos for c in env.cells], dtype=float)
    thetas = np.array([c.theta for c in env.cells], dtype=float)
    if centers.size == 0:
        return 0.0
    la = get_local_anisotropy(centers, thetas, neighbourhood_range)
    return float(np.mean(la)) if la.size > 0 else 0.0


def main():
    env = ColonyEnv(seed=SEED)
    # Tie neighbourhood range to environment division length
    neighbourhood_range = NEIGHBORHOOD_MULTIPLIER * getattr(env, 'L_divide', 2.0)

    # Prepare agent if using trained policy
    # Need obs_dim to build the policy; do a quick reset to get obs
    agent = None
    if USE_TRAINED:
        obs0, _ = env.reset()
        obs_dim = obs0.shape[1] if hasattr(obs0, 'shape') and len(obs0.shape) == 2 else 6
        agent = load_agent(CHECKPOINT_PATH, obs_dim)

    results = []
    for i in range(NUM_RUNS):
        mean_aniso = run_single_episode(env, TARGET_CELLS, MAX_STEPS_PER_RUN, neighbourhood_range, agent)
        results.append(mean_aniso)
        policy_desc = "trained" if agent is not None else "random"
        det_desc = "det" if DETERMINISTIC and agent is not None else ("stoch" if (agent is not None and not DETERMINISTIC) else "uniform")
        print(f"Run {i+1:02d}/{NUM_RUNS} | policy={policy_desc}/{det_desc} | mean anisotropy = {mean_aniso:.4f}")

    results_np = np.array(results, dtype=float)
    overall_mean = float(np.mean(results_np)) if results_np.size else 0.0
    overall_std = float(np.std(results_np)) if results_np.size else 0.0

    print("" + "="*50)
    print(f"Completed {NUM_RUNS} runs")
    print(f"Target cells: {TARGET_CELLS}")
    print(f"Neighbourhood range: {neighbourhood_range:.3f}")
    print(f"Mean anisotropy across runs: {overall_mean:.4f} ± {overall_std:.4f}")

    if SAVE_CSV:
        csv_path = Path(CSV_PATH)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["run", "mean_anisotropy", "target_cells", "neighbourhood_range", "policy", "deterministic", "checkpoint"])
            for idx, val in enumerate(results, start=1):
                writer.writerow([
                    idx,
                    f"{val:.6f}",
                    TARGET_CELLS,
                    f"{neighbourhood_range:.6f}",
                    "trained" if agent is not None else "random",
                    bool(DETERMINISTIC) if agent is not None else False,
                    CHECKPOINT_PATH if agent is not None else "",
                ])
        print(f"Saved CSV: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
