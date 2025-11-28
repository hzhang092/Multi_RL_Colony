r"""
Rollout viewer: run a trained or untrained policy and visualize every N steps.

No interactions, just a simple playback that updates the frame periodically.

How to use (no argparse):
1) Edit the CONFIG section below.
2) Run:
       python evaluations\\rollout_view.py
"""

import os
import sys
from pathlib import Path
from typing import Optional, Any
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the project root to the Python path to allow for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.colony_env import ColonyEnv
try:
    from agents.ppo_agent import PPOAgent, make_action_dicts
except Exception:
    PPOAgent = None
    make_action_dicts = None


# ======================
# CONFIG (edit these)
# ======================
SEED: int = 686
MAX_STEPS: int = 300
RENDER_INTERVAL: int = 2      # show frame every N steps
FIGSIZE = (7, 7)

# Output frames
SAVE_FRAMES: bool = False
FRAMES_DIR: str = "eval_frames"  # used only if SAVE_FRAMES=True

# Final frame display options
SHOW_FINAL: bool = True        # ensure the last frame is displayed after loop ends
FINAL_BLOCK: bool = True       # block on plt.show() so the window stays until closed
FINAL_PAUSE_SEC: float = 2.0   # used if FINAL_BLOCK=False

# Policy selection
USE_TRAINED: bool = True  # False: random policy
CHECKPOINT_PATH: str = "saved_checkpoints/ppo_colony_final-1124-1.pt" #"saved_checkpoints/ppo_colony_400-1123-2300.pt" #"saved_checkpoints/ppo_colony_final-1013-3.pt"  # ignored if USE_TRAINED=False
DETERMINISTIC: bool = True   # True: argmax; False: stochastic sampling

# Device for policy
DEVICE: Optional[str] = None  # None auto-selects; set to 'cpu' to force CPU


def _fallback_make_action_dicts(action_types: np.ndarray):
    # Env accepts list of ints [0,1,2]
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


def select_actions(obs: np.ndarray, agent: Optional[Any]):
    if agent is None:
        # uniform random over 3 discrete actions per agent
        action_types = np.random.randint(0, 3, size=(len(obs),), dtype=np.int64)
        if make_action_dicts is not None:
            at_t = torch.tensor(action_types, dtype=torch.int64)
            return make_action_dicts(at_t)
        return _fallback_make_action_dicts(action_types)

    # Use trained policy
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


def main():
    env = ColonyEnv(seed=SEED)
    obs, _ = env.reset()
    obs_dim = obs.shape[1] if hasattr(obs, 'shape') and len(obs.shape) == 2 else 6
    agent = load_agent(CHECKPOINT_PATH, obs_dim) if USE_TRAINED else None

    # Setup matplotlib
    plt.ion()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = None

    # Prepare frame saving
    frames_dir = None
    if SAVE_FRAMES:
        frames_dir = Path(FRAMES_DIR)
        frames_dir.mkdir(parents=True, exist_ok=True)

    terminated = False
    truncated = False
    last_step = -1
    for step in range(MAX_STEPS):
        actions = select_actions(obs, agent)
        obs, rewards, terminated, truncated, info = env.step(actions)
        last_step = step
        
        print(info.get('colony_aspect_ratio', -1.0))

        if step % RENDER_INTERVAL == 0 or terminated or truncated or step == 0:
            img = env.render(mode="rgb_array")
            if im is None:
                ax.clear()
                im = ax.imshow(img)
                ax.axis('off')
            else:
                im.set_data(img)
            ax.set_title(f"Step {step+1} | Cells: {len(obs)} | Reward: {np.sum(rewards):.3f}")
            plt.pause(0.01)

            if SAVE_FRAMES and frames_dir is not None:
                out_path = frames_dir / f"frame_{step:05d}.png"
                plt.imsave(out_path.as_posix(), img)

        if terminated or truncated:
            print(f"Episode ended at step {step}.")
            break

    # Always render and display the final frame
    if SHOW_FINAL:
        img = env.render(mode="rgb_array")
        if im is None:
            ax.clear()
            im = ax.imshow(img)
            ax.axis('off')
        else:
            im.set_data(img)
        status = "terminated" if terminated else ("truncated" if truncated else "finished")
        ax.set_title(f"Final ({status}) | Step {last_step} | Cells: {len(obs)}")
        plt.draw()
        if SAVE_FRAMES and frames_dir is not None:
            out_path = frames_dir / f"frame_{last_step:05d}_final.png"
            plt.imsave(out_path.as_posix(), img)

        if FINAL_BLOCK:
            plt.ioff()
            plt.show()
        else:
            plt.pause(FINAL_PAUSE_SEC)
    env.close()


if __name__ == "__main__":
    main()
