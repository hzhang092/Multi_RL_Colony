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
from typing import Optional, Any, Tuple
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import csv
import torch

# Add the project root to the Python path to allow for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.colony_env import ColonyEnv
from envs.utilities.geo_helpers import get_local_anisotropy, pca_aspect_ratio
try:
    from agents.ppo_agent import PPOAgent, make_action_dicts
except Exception:
    PPOAgent = None
    make_action_dicts = None


# ======================
# CONFIG (edit these)
# ======================
SEED: int = 686
NUM_RUNS: int = 50
TARGET_CELLS: int = 80            # Stop an episode when this many cells exist
MAX_STEPS_PER_RUN: int = 150     # Safety cap on steps per run

# Neighbourhood range used in local anisotropy (distance threshold)
# A reasonable default is tied to division length; final value computed at runtime.
NEIGHBORHOOD_MULTIPLIER: float = 3.0   # neighbourhood_range = multiplier * env.L_divide

# Policy config (set USE_TRAINED=True to use a checkpoint)
USE_TRAINED: bool = True
CHECKPOINT_FOLDER: str = "saved_checkpoints"
CHECKPOINT_NAME: str = "ppo_colony_final-1129-4.pt"
CHECKPOINT_PATH: str = f"{CHECKPOINT_FOLDER}/{CHECKPOINT_NAME}"  # used only if USE_TRAINED=True
DETERMINISTIC: bool = True   # True: argmax over logits; False: stochastic sampling
DEVICE: Optional[str] = None # None auto-selects; set to 'cpu' to force CPU

# Save CSV results
SAVE_CSV: bool = True
CSV_PATH: str = "evaluations/results/" + CHECKPOINT_NAME.replace(".pt", ".csv")


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


def run_single_episode(env: ColonyEnv, target_cells: int, max_steps: int, neighbourhood_range: float, agent: Optional[Any]) -> Tuple[float, float, float]:
    """Run one random-policy rollout until target cell count or step cap.

    Returns (mean local anisotropy, colony aspect ratio) at the stopping point.
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
            # Compute metrics
            centers = np.array([c.pos for c in env.cells], dtype=float)
            thetas = np.array([c.theta for c in env.cells], dtype=float)
            
            la = get_local_anisotropy(centers, thetas, neighbourhood_range)
            mean_la = float(np.mean(la)) if la.size > 0 else 0.0
            
            ar = pca_aspect_ratio(centers)
            print(f"Reached target cells or termination at step {steps}.")
            return mean_la, ar, len(env.cells)

    # If we exit due to step cap, compute anyway
    centers = np.array([c.pos for c in env.cells], dtype=float)
    thetas = np.array([c.theta for c in env.cells], dtype=float)
    
    la = get_local_anisotropy(centers, thetas, neighbourhood_range)
    mean_la = float(np.mean(la)) if la.size > 0 else 0.0
    
    ar = pca_aspect_ratio(centers) if centers.size > 0 else 1.0
    return mean_la, ar, len(env.cells)


def main():
    # Use defaults to match training environment (torque_rate=0.5, division_jitter=0.1)
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

    results_aniso = []
    results_ar = []
    
    for i in range(NUM_RUNS):
        mean_aniso, ar, num_cells = run_single_episode(env, TARGET_CELLS, MAX_STEPS_PER_RUN, neighbourhood_range, agent)
        results_aniso.append(mean_aniso)
        results_ar.append(ar)
        
        policy_desc = "trained" if agent is not None else "random"
        det_desc = "det" if DETERMINISTIC and agent is not None else ("stoch" if (agent is not None and not DETERMINISTIC) else "uniform")
        print(f"Run {i+1:02d}/{NUM_RUNS} | policy={policy_desc}/{det_desc} | Anisotropy={mean_aniso:.4f} | AR={ar:.4f} | Cells={num_cells}")

    results_aniso_np = np.array(results_aniso, dtype=float)
    mean_aniso_all = float(np.mean(results_aniso_np)) if results_aniso_np.size else 0.0
    std_aniso_all = float(np.std(results_aniso_np)) if results_aniso_np.size else 0.0

    results_ar_np = np.array(results_ar, dtype=float)
    mean_ar_all = float(np.mean(results_ar_np)) if results_ar_np.size else 0.0
    std_ar_all = float(np.std(results_ar_np)) if results_ar_np.size else 0.0

    print("" + "="*50)
    print(f"Completed {NUM_RUNS} runs")
    print(f"Target cells: {TARGET_CELLS}")
    print(f"Neighbourhood range: {neighbourhood_range:.3f}")
    print(f"Mean Anisotropy: {mean_aniso_all:.4f} ± {std_aniso_all:.4f}")
    print(f"Mean Aspect Ratio: {mean_ar_all:.4f} ± {std_ar_all:.4f}")

    if SAVE_CSV:
        csv_path = Path(CSV_PATH)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["run", "mean_anisotropy", "aspect_ratio", "target_cells", "neighbourhood_range", "policy", "deterministic", "checkpoint"])
            for idx, (val_a, val_ar) in enumerate(zip(results_aniso, results_ar), start=1):
                writer.writerow([
                    idx,
                    f"{val_a:.6f}",
                    f"{val_ar:.6f}",
                    TARGET_CELLS,
                    f"{neighbourhood_range:.6f}",
                    "trained" if agent is not None else "random",
                    bool(DETERMINISTIC) if agent is not None else False,
                    CHECKPOINT_PATH if agent is not None else "",
                ])
            writer.writerow([f"Mean Anisotropy: {mean_aniso_all:.4f} +- {std_aniso_all:.4f}"])
            writer.writerow([f"Mean Aspect Ratio: {mean_ar_all:.4f} +- {std_ar_all:.4f}"])
        print(f"Saved CSV: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
