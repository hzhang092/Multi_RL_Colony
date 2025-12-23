"""
Policy Action Inspector: analyze when a trained policy chooses each action.

What it does:
- Runs a short rollout in `ColonyEnv` using a trained PPO policy.
- Records observations, chosen actions, and action probabilities.
- Produces plots to visualize "circumstances" for each action:
  - Per-feature histograms split by action (0=dormant, 1=grow, 2=divide).
  - Decision maps for selected 2D feature pairs (argmax action across a grid).

Usage:
  1) Adjust CONFIG below (checkpoint path, steps, etc.).
  2) Run:
       python evaluations\\policy_action_inspector.py

Outputs are saved under `evaluations/plots/`.
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.colony_env import ColonyEnv
from agents.ppo_agent import PPOAgent


# =====================
# CONFIG (edit these)
# =====================
SEED: int = 686
ROLLOUT_STEPS: int = 200        # number of env steps to sample data
MAX_EPISODES: int = 3           # cap episodes if they end early
DETERMINISTIC: bool = True      # argmax vs sampling during rollout

# Model checkpoint to load (required when USE_TRAINED=True)
USE_TRAINED: bool = True
CHECKPOINT_PREFIX = "final"
CHECKPOINT_NAME: str = "1129-3"  #!!!!!
CHECKPOINT_PATH: str = f"saved_checkpoints/ppo_colony_{CHECKPOINT_PREFIX}-{CHECKPOINT_NAME}.pt"  # or saved_checkpoints/... if preferred

# Device
DEVICE: Optional[str] = None    # None auto-selects, or 'cpu'/'cuda'

# Output
PLOTS_DIR = Path(f"evaluations/plots/{CHECKPOINT_NAME}_policy_inspector") #!!!!!
SAVE_FIG_DPI = 140


# Feature meta (order from env._obs_for_cell)
FEATURES = [
    (0, "rel_length"),
    (1, "rel_age"),
    (2, "orientation_sin"),
    (3, "orientation_cos"),
    (4, "local_density"),
    (5, "pressure_proxy"),
    #(6, "anisotropy"),
]

PAIRWISE_TO_PLOT = [
    (0, 4),  # rel_length vs local_density
    (0, 5),  # rel_length vs pressure_proxy
    (1, 4),  # rel_age vs local_density
    (1, 5),  # rel_age vs pressure_proxy
    #(0, 6),  # rel_length vs anisotropy
    #(1, 6),  # rel_age vs anisotropy
]


def make_agent(obs_dim: int) -> Optional[PPOAgent]:
    if not USE_TRAINED:
        return None
    device = torch.device(DEVICE) if isinstance(DEVICE, str) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(obs_dim=obs_dim, device=device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'policy_state_dict' in ckpt:
        agent.policy.load_state_dict(ckpt['policy_state_dict'])
    else:
        agent.policy.load_state_dict(ckpt)
    agent.policy.eval()
    return agent


@torch.no_grad()
def policy_probs(agent: PPOAgent, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return action probabilities [N,3] and chosen actions [N]."""
    obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    logits, _ = agent.policy(obs_t)
    probs = torch.softmax(logits, dim=-1)
    if DETERMINISTIC:
        acts = torch.argmax(probs, dim=-1)
    else:
        dist = torch.distributions.Categorical(probs=probs)
        acts = dist.sample()
    return probs.detach().cpu().numpy(), acts.detach().cpu().numpy()


def collect_rollout(env: ColonyEnv, agent: Optional[PPOAgent], steps: int, max_episodes: int) -> Dict[str, np.ndarray]:
    """Run rollout and collect observations, actions, and probabilities.

    Returns dict with keys: 'obs' [M,6], 'act' [M], 'probs' [M,3]
    (M is total per-cell decisions across timesteps; each row is one cell's decision).
    """
    rng = np.random.default_rng(SEED)
    data_obs = []
    data_act = []
    data_probs = []

    obs, _ = env.reset()
    terminated = truncated = False
    ep_count = 0
    step_count = 0

    while step_count < steps and ep_count < max_episodes:
        if agent is None:
            # random policy
            acts = rng.integers(0, 3, size=(len(obs),), dtype=np.int64)
            probs = np.full((len(obs), 3), 1/3.0, dtype=np.float32)
        else:
            probs, acts = policy_probs(agent, obs)

        # record per-cell decisions for current step
        data_obs.append(obs.astype(np.float32))
        data_act.append(acts.astype(np.int64))
        data_probs.append(probs.astype(np.float32))

        # step env
        next_obs, rewards, terminated, truncated, info = env.step(acts.tolist())
        step_count += 1

        if terminated or truncated:
            ep_count += 1
            obs, _ = env.reset()
            terminated = truncated = False
        else:
            obs = next_obs

    if not data_obs:
        return {"obs": np.zeros((0, 6), dtype=np.float32), "act": np.zeros((0,), dtype=np.int64), "probs": np.zeros((0, 3), dtype=np.float32)}

    obs_all = np.concatenate(data_obs, axis=0)
    act_all = np.concatenate(data_act, axis=0)
    probs_all = np.concatenate(data_probs, axis=0)
    return {"obs": obs_all, "act": act_all, "probs": probs_all}


def plot_feature_histograms(obs: np.ndarray, acts: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = int(np.ceil(len(FEATURES) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4.0, nrows * 3.2))
    axes = np.array(axes).reshape(nrows, ncols)

    colors = {0: "#455A64", 1: "#1E88E5", 2: "#E53935"}
    labels = {0: "0 dormant", 1: "1 grow", 2: "2 divide"}

    for k, (idx, name) in enumerate(FEATURES):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        for a in [0, 1, 2]:
            vals = obs[acts == a, idx]
            if vals.size == 0:
                continue
            ax.hist(vals, bins=40, alpha=0.55, color=colors[a], label=labels[a], density=True)
        ax.set_title(name)
        ax.set_ylabel("density")
        ax.set_xlabel("value")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

    # Hide any unused axes
    total_axes = nrows * ncols
    for k in range(len(FEATURES), total_axes):
        r, c = divmod(k, ncols)
        axes[r, c].axis('off')

    fig.suptitle("Per-feature distributions by chosen action")
    fig.tight_layout()
    out_path = out_dir / "feature_histograms.png"
    fig.savefig(out_path.as_posix(), dpi=SAVE_FIG_DPI)
    plt.close(fig)
    return out_path


@torch.no_grad()
def decision_map(agent: PPOAgent, base_vec: np.ndarray, i: int, j: int, grid: int, bounds_i: Tuple[float, float], bounds_j: Tuple[float, float]) -> np.ndarray:
    """Compute argmax action over a 2D grid varying features i and j."""
    vmin, vmax = bounds_i
    wmin, wmax = bounds_j
    xs = np.linspace(vmin, vmax, grid)
    ys = np.linspace(wmin, wmax, grid)
    grid_points = []
    for y in ys:
        for x in xs:
            vec = base_vec.copy()
            vec[i] = x
            vec[j] = y
            grid_points.append(vec)
    X = torch.tensor(np.stack(grid_points, axis=0), dtype=torch.float32, device=agent.device)
    logits, _ = agent.policy(X)
    probs = torch.softmax(logits, dim=-1)
    acts = torch.argmax(probs, dim=-1).detach().cpu().numpy()
    return acts.reshape(grid, grid)


def plot_pairwise_maps(agent: PPOAgent, obs: np.ndarray, out_dir: Path, grid: int = 80):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Base vector: use median of observed states so the slice is realistic
    base = np.median(obs, axis=0)

    # Bounds from observation space or observed min/max as fallback
    env_low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    env_high = np.array([1.25, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    obs_min = np.minimum(np.min(obs, axis=0), env_low)
    obs_max = np.maximum(np.max(obs, axis=0), env_high)

    ncols = 3
    nrows = int(np.ceil(len(PAIRWISE_TO_PLOT) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4.4, nrows * 4.0))
    axes = np.array(axes).reshape(nrows, ncols)

    # color map for 3 actions
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#455A64", "#1E88E5", "#E53935"])  # 0,1,2

    for k, (i, j) in enumerate(PAIRWISE_TO_PLOT):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        Z = decision_map(
            agent,
            base_vec=base,
            i=i,
            j=j,
            grid=grid,
            bounds_i=(env_low[i], env_high[i]),
            bounds_j=(env_low[j], env_high[j]),
        )
        extent = [env_low[i], env_high[i], env_low[j], env_high[j]]
        im = ax.imshow(Z.T, origin='lower', extent=extent, cmap=cmap, interpolation='nearest', aspect='auto')
        xi = FEATURES[i][1]
        xj = FEATURES[j][1]
        ax.set_xlabel(xi)
        ax.set_ylabel(xj)
        ax.set_title(f"Argmax action over {xi} vs {xj}")
        # overlay sampled points colored by real chosen action for reference
        # Use a thin sample to avoid overplotting
        if obs.shape[0] > 0:
            # Estimate chosen actions at observed states for color reference
            # Use the agent's deterministic policy on observed data
            probs_obs, acts_obs = policy_probs(agent, obs)
            # Subsample
            idx = np.linspace(0, obs.shape[0] - 1, num=min(1200, obs.shape[0]), dtype=int)
            ax.scatter(obs[idx, i], obs[idx, j], c=acts_obs[idx], s=6, cmap=cmap, alpha=0.4, edgecolors='none')

    # Hide any unused axes
    total_axes = nrows * ncols
    for k in range(len(PAIRWISE_TO_PLOT), total_axes):
        r, c = divmod(k, ncols)
        axes[r, c].axis('off')

    # Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color="#455A64", label="0 dormant"),
        mpatches.Patch(color="#1E88E5", label="1 grow"),
        mpatches.Patch(color="#E53935", label="2 divide"),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Policy decision maps (argmax action)")
    fig.tight_layout()
    out_path = out_dir / "pairwise_decision_maps.png"
    fig.savefig(out_path.as_posix(), dpi=SAVE_FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    # Prepare output dir
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Env and agent
    env = ColonyEnv(seed=SEED)
    obs0, _ = env.reset()
    obs_dim = obs0.shape[1]
    agent = make_agent(obs_dim) if USE_TRAINED else None
    if USE_TRAINED and agent is None:
        raise RuntimeError("Failed to create agent; check checkpoint path and imports.")

    # Collect rollout data
    data = collect_rollout(env, agent, steps=ROLLOUT_STEPS, max_episodes=MAX_EPISODES)
    obs = data['obs']
    acts = data['act']

    if obs.shape[0] == 0:
        print("No data collected. Is the environment producing observations?")
        return

    # Plot 1: per-feature histograms
    hist_path = plot_feature_histograms(obs, acts, PLOTS_DIR)

    # Plot 2: pairwise decision maps (requires trained agent)
    map_path = None
    if agent is not None:
        map_path = plot_pairwise_maps(agent, obs, PLOTS_DIR)

    print("Saved:")
    print(f" - {hist_path}")
    if map_path is not None:
        print(f" - {map_path}")


if __name__ == "__main__":
    main()
