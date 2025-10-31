r"""
Interactive ColonyEnv viewer with optional trained policy.

Features:
- Renders each timestep in a Matplotlib window.
- Only advances when you decide (press a key/Enter).
- Choose between random (untrained) actions or a loaded PPO policy.
- Optional: reset, save frames, or quit via keyboard.

How to use (no argparse):
1) Edit the CONFIG section below (seed, max steps, use trained, checkpoint path, etc.).
2) Run:
       python evaluations\\manual_env_viewer.py

Controls in the plot window or console:
- Enter/Space/n: advance one step
- r: reset environment
- s: save current frame (if SAVE_DIR provided)
- q: quit
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Any

# Add the project root to the Python path to allow for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.colony_env import ColonyEnv
try:
    # Preferred: use shared helper to format actions for the env
    from agents.ppo_agent import PPOAgent, make_action_dicts
except Exception:
    PPOAgent = None
    make_action_dicts = None  # Fallback defined later if needed


# ======================
# CONFIG (edit these)
# ======================
SEED: int = 686
MAX_STEPS: int = 120
SAVE_DIR: str = ""       # e.g., "frames" to enable saving by pressing 's'; empty string disables saving

# Use a trained PPO policy instead of random actions
USE_TRAINED: bool = False
CHECKPOINT_PATH: str = "saved_checkpoints/ppo_colony_final-1013-1.pt"  # e.g., "saved_checkpoints/ppo_colony_final-1013-1.pt"
DETERMINISTIC: bool = False  # If True, argmax over action probs; else sample stochastically (still biased by what the policy learned; not uniform random.)

# Device selection for policy (if used)
DEVICE = None  # auto-select (cuda if available); set to "cpu" to force CPU


def _fallback_make_action_dicts(action_types: np.ndarray):
    """
    Fallback action mapping if agents.ppo_agent.make_action_dicts isn't available.
    Maps integers (0,1,2) to simple action dicts expected by the env.

    This is a best-effort guess mirroring PPO usage where "type" selects:
      0: dormant, 1: grow, 2: divide
    """
    actions = []
    for a in action_types:
        actions.append({"type": int(a)})
    return actions


class ViewerState:
    def __init__(self):
        self.im = None
        self.last_key: Optional[str] = None


def render_frame(env: ColonyEnv, ax, state: ViewerState, title_prefix: str = ""):
    """Render the environment and update the matplotlib Axes image."""
    img = env.render(mode="rgb_array")
    if state.im is None:
        ax.clear()
        state.im = ax.imshow(img)
        ax.axis("off")
    else:
        state.im.set_data(img)
    ax.set_title(f"{title_prefix}")
    plt.pause(0.001)


def interactive_loop(env: ColonyEnv, max_steps: int, save_dir: Optional[Path], agent: Optional[Any] = None, deterministic: bool = True):
    obs, _ = env.reset()

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.ion()
    fig.show()
    step_idx = 0
    saved_count = 0
    state = ViewerState()

    def on_key(event):
        # Store last key pressed in local state for polling
        state.last_key = event.key

    fig.canvas.mpl_connect('key_press_event', on_key)
    state.last_key = None

    while step_idx < max_steps:
        n_cells = len(obs)
        render_frame(env, ax, state, title_prefix=f"t={step_idx} | cells={n_cells}")

        # Save current frame on demand
        if save_dir is not None and isinstance(state.last_key, str) and state.last_key.lower() == 's':
            img = env.render(mode="rgb_array")
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"frame_{step_idx:05d}.png"
            plt.imsave(out_path.as_posix(), img)
            print(f"Saved frame: {out_path}")
            state.last_key = None

        # Wait for user input to advance
        print("[n/Enter/Space] next | r reset | s save | q quit > ", end="", flush=True)
        try:
            # Prefer plot window key if present, else fallback to console input
            waiting = True
            while waiting:
                plt.pause(0.05)
                key = state.last_key
                if isinstance(key, str):
                    k = key.lower()
                    state.last_key = None
                else:
                    # Non-blocking console check isn't trivial; use a quick blocking input
                    # if no key pressed in the plot window.
                    # We allow an immediate Enter press in the console to advance.
                    k = input().strip().lower() if key is None else None

                if k in (None, ""):
                    # Empty Enter -> advance
                    cmd = "n"
                else:
                    cmd = k

                if cmd in ("n", " ", "enter"):
                    waiting = False
                elif cmd == "r":
                    obs, _ = env.reset()
                    step_idx = 0
                    print("\nEnvironment reset.")
                    # Immediately re-render after reset
                    render_frame(env, ax, state, title_prefix=f"t={step_idx} | cells={len(obs)}")
                elif cmd == "s":
                    if save_dir is None:
                        print("\nNo --save-dir specified. Skipping save.")
                    else:
                        img = env.render(mode="rgb_array")
                        save_dir.mkdir(parents=True, exist_ok=True)
                        out_path = save_dir / f"frame_{step_idx:05d}.png"
                        plt.imsave(out_path.as_posix(), img)
                        saved_count += 1
                        print(f"\nSaved frame: {out_path}")
                elif cmd == "q":
                    print("\nQuitting.")
                    return
                else:
                    # Unknown key -> ignore and continue waiting
                    pass

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            return

        # Sample actions (trained policy or random)
        if len(obs) == 0:
            # If no agents, reset automatically
            obs, _ = env.reset()
            step_idx = 0
            continue

        if agent is not None and PPOAgent is not None:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device if hasattr(agent, 'device') else None)
            with torch.no_grad():
                if deterministic:
                    logits, _ = agent.policy(obs_t)
                    probs = torch.softmax(logits, dim=-1)
                    sampled_type = torch.argmax(probs, dim=-1)
                else:
                    sampled_type, _, _ = agent.act(obs_t)
            if make_action_dicts is not None:
                actions = make_action_dicts(sampled_type)
            else:
                # fall back to raw ints
                action_types = sampled_type.detach().cpu().numpy().astype(np.int64)
                actions = action_types.tolist()
        else:
            # random policy
            action_types = np.random.randint(0, 3, size=(len(obs),), dtype=np.int64)
            if make_action_dicts is not None:
                at_t = torch.tensor(action_types, dtype=torch.int64)
                actions = make_action_dicts(at_t)
            else:
                actions = _fallback_make_action_dicts(action_types)

        # Step environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        obs = next_obs
        step_idx += 1

        if terminated or truncated:
            print("Episode finished (terminated or truncated). Press 'r' to reset or 'q' to quit.")

    print("Reached max steps. Exiting.")


def load_agent(checkpoint_path: str, obs_dim: int):
    if PPOAgent is None:
        raise RuntimeError("PPOAgent not available. Check agents/ppo_agent.py import.")
    device = torch.device(DEVICE) if isinstance(DEVICE, str) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(obs_dim=obs_dim, device=device)
    if not checkpoint_path:
        return agent  # uninitialized (random weights)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'policy_state_dict' in ckpt:
        agent.policy.load_state_dict(ckpt['policy_state_dict'])
    else:
        agent.policy.load_state_dict(ckpt)
    agent.policy.eval()
    return agent


def main():
    save_dir = Path(SAVE_DIR) if SAVE_DIR else None
    env = ColonyEnv(seed=SEED)
    agent = None
    if USE_TRAINED:
        # Need obs_dim to build the policy; do a quick reset to get obs
        obs, _ = env.reset()
        obs_dim = obs.shape[1] if hasattr(obs, 'shape') and len(obs.shape) == 2 else 6
        agent = load_agent(CHECKPOINT_PATH, obs_dim=obs_dim)
        #DETERMINISTIC = True
    try:
        interactive_loop(env, max_steps=MAX_STEPS, save_dir=save_dir, agent=agent, deterministic=DETERMINISTIC)
    finally:
        env.close()
        plt.ioff()
        plt.close("all")


if __name__ == "__main__":
    main()
