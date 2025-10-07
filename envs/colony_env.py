# colony_env.py
# Gymnasium-compatible ColonyEnv: capsule (pill) cells, continuous positions,
# growth at poles, division next-step with jitter, iterative overlap relaxation,
# morphology metrics: Aspect Ratio, Density, Fourier descriptors.

"""
This module implements a Gymnasium-compatible environment for simulating the
growth of a bacterial colony. The key features are:

- **Cell Representation**: Cells are modeled as capsules (or pills) with a
  fixed radius and variable length, existing in a continuous 2D space.
- **Growth Dynamics**: Cells can grow in length at their poles.
- **Division**: When a cell reaches a threshold length, it can be marked for
  division. The division occurs in the next timestep, creating two daughter
  cells with some positional and angular jitter.
- **Physics**: A simple, iterative relaxation method is used to resolve
  overlaps between cells, simulating physical pushing forces.
- **Observation Space**: Each agent (cell) observes its own state (length,
  orientation, age) and the relative states of its K-nearest neighbors.
- **Action Space**: Each agent can choose to do nothing, grow, or divide.
  Growth and rotation can be modulated.
- **Reward System**: Rewards are based on how closely the colony's overall
  morphology matches a target shape, defined by metrics like aspect ratio,
  density, and Fourier descriptors.
"""

import math, numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

# ---------- Geometry helpers (self-contained) ----------
def seg_seg_closest_points(a0, a1, b0, b1):
    """
    Calculates the closest points between two line segments in 2D or 3D.

    This is a standard algorithm to find the minimum distance between two finite
    lines. It's used here to detect the distance between the central axes of
    two capsule cells for collision detection.

    Args:
        a0 (np.ndarray): Start point of segment A.
        a1 (np.ndarray): End point of segment A.
        b0 (np.ndarray): Start point of segment B.
        b1 (np.ndarray): End point of segment B.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - Pa: The point on segment A closest to segment B.
            - Pb: The point on segment B closest to segment A.
            - distance: The Euclidean distance between Pa and Pb.
    """
    A = a1 - a0
    B = b1 - b0
    C = a0 - b0
    a = np.dot(A, A)
    b = np.dot(A, B)
    c = np.dot(B, B)
    d = np.dot(A, C)
    e = np.dot(B, C)
    denom = a*c - b*b
    s = 0.0
    t = 0.0
    eps = 1e-9
    if denom > eps:
        s = (b*e - c*d) / denom
        s = np.clip(s, 0.0, 1.0)
    else:
        s = 0.0
    t = (b*s + e) / c if c > eps else 0.0
    if t < 0.0:
        t = 0.0
        s = np.clip(-d / a if a>eps else 0.0, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - d) / a if a>eps else 0.0, 0.0, 1.0)
    Pa = a0 + A * s
    Pb = b0 + B * t
    diff = Pa - Pb
    dist2 = np.dot(diff, diff)
    return Pa, Pb, math.sqrt(max(dist2, 0.0))

def unit_vector(v):
    """
    Computes the unit vector of a given vector.

    Args:
        v (np.ndarray): The input vector.

    Returns:
        np.ndarray: The unit vector. Returns a default vector [1, 0] if the
                    input vector has a near-zero norm.
    """
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0.0])
    return v / n

def monotone_chain_convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Computes the convex hull of a set of 2D points using Andrew's monotone
    chain algorithm.

    This is used to find the outer boundary of the entire colony.

    Args:
        points (np.ndarray): An array of shape (N, 2) of 2D points.

    Returns:
        np.ndarray: An array of points representing the convex hull, ordered
                    counter-clockwise.
    """
    pts = sorted([tuple(p) for p in points])
    if len(pts) <= 1:
        return np.array(pts)
    lower = []
    for p in pts:
        while len(lower) >= 2:
            q1 = np.array(lower[-2]); q2 = np.array(lower[-1]); q3 = np.array(p)
            if np.cross(q2 - q1, q3 - q2) <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2:
            q1 = np.array(upper[-2]); q2 = np.array(upper[-1]); q3 = np.array(p)
            if np.cross(q2 - q1, q3 - q2) <= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull)

def polygon_area(points: np.ndarray) -> float:
    """
    Calculates the area of a polygon using the shoelace formula.

    Used to compute the area of the colony's convex hull.

    Args:
        points (np.ndarray): An array of shape (N, 2) of polygon vertices,
                             ordered clockwise or counter-clockwise.

    Returns:
        float: The area of the polygon.
    """
    if len(points) < 3:
        return 0.0
    x = points[:,0]; y = points[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def pca_aspect_ratio(points: np.ndarray) -> float:
    """
    Calculates the aspect ratio of a point cloud via Principal Component Analysis.

    The aspect ratio is the ratio of the largest eigenvalue to the second-largest
    eigenvalue of the covariance matrix of the points. This measures the
    elongation of the colony.

    Args:
        points (np.ndarray): An array of shape (N, 2) of points.

    Returns:
        float: The aspect ratio (>= 1.0).
    """
    if len(points) < 2:
        return 1.0
    mean = points.mean(axis=0)
    X = points - mean
    cov = np.cov(X.T)
    vals = np.linalg.eigvalsh(cov)
    vals = np.flip(np.sort(vals))
    if vals[1] <= 1e-9:
        return float(vals[0] / (vals[1] + 1e-9))
    return float(max(vals[0] / max(vals[1], 1e-9), 1.0))

def fourier_descriptor_from_boundary(boundary_pts: np.ndarray, K=8):
    """
    Computes scale-invariant Fourier descriptors from a set of boundary points.

    These descriptors capture the shape of the colony's boundary in the
    frequency domain, providing a quantitative measure of its morphology.

    Args:
        boundary_pts (np.ndarray): An array of shape (N, 2) of ordered points
                                   on the boundary.
        K (int): The number of Fourier descriptors to return.

    Returns:
        np.ndarray: A vector of K Fourier descriptors.
    """
    if len(boundary_pts) == 0:
        return np.zeros(K)
    # Convert 2D points to a 1D complex signal
    z = boundary_pts[:,0] + 1j * boundary_pts[:,1]
    # Compute the Fast Fourier Transform
    Z = np.fft.fft(z)
    mags = np.abs(Z)
    # Normalize by the magnitude of the DC component (Z[0]) to achieve
    # scale invariance.
    mags0 = mags[0] if mags[0] > 1e-9 else 1.0
    descriptor = []
    for k in range(1, K+1):
        descriptor.append((mags[k] / mags0) if k < len(mags) else 0.0)
    return np.array(descriptor)

# ---------- Stick cell ----------
@dataclass
class StickCell:
    """
    A data class representing a single stick-shaped bacterial cell.

    Attributes:
        pos (np.ndarray): The (x, y) coordinates of the cell's geometric center.
        theta (float): The orientation angle in radians.
        length (float): The total length of the cell's central axis.
        age (float): The number of timesteps the cell has existed.
        pending_divide (bool): A flag set to True when the cell is ready to
                               divide in the next timestep.
        just_divided (bool): A flag set to True for cells created from division
                             in the current timestep, used for reward calculation.
    """
    pos: np.ndarray  # shape (2,)
    theta: float # orientation angle (radians)
    length: float  # total length
    age: float = 0.0
    pending_divide: bool = False
    just_divided: bool = False

    def endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the two endpoints of the cell's central axis.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the two endpoint
                                           coordinate vectors.
        """
        ux = np.array([math.cos(self.theta), math.sin(self.theta)])
        half = 0.5 * self.length
        return (self.pos + ux * half, self.pos - ux * half)


# ---------- Gymnasium environment ----------

class ColonyEnv(gym.Env):
    """
    A Gymnasium environment for simulating bacterial colony growth.

    In this environment, each cell is an independent agent that decides when to
    grow and divide. The goal is to collectively form a colony with specific
    morphological properties (e.g., high aspect ratio, low density).

    The simulation proceeds in discrete timesteps. In each step:
    1. Agents (cells) provide actions (grow, divide, rotate).
    2. Cell states are updated based on actions.
    3. Physical overlaps between cells are resolved via iterative relaxation.
    4. Cells marked for division are replaced by two new daughter cells.
    5. A new observation is gathered for each agent.
    6. A reward is computed based on the colony's global morphology.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 world_size=(64.0, 64.0),
                 L_init=1.0,
                 L_divide=2.0,
                 max_cells=80,
                 K_nn=6, # number of nearest neighbors in observation
                 fourier_K=8,
                 seed: Optional[int]=None):
        """
        Initializes the Colony environment.

        Args:
            world_size (Tuple[float, float]): The (width, height) of the 2D world.
            L_init (float): The initial length of the first cell.
            L_divide (float): The length at which cells can divide.
            max_cells (int): The maximum number of cells before the episode ends.
            K_nn (int): The number of nearest neighbors each cell observes.
            fourier_K (int): The number of Fourier descriptors for shape analysis.
            seed (Optional[int]): A seed for the random number generator.
        """
        super().__init__()
        self.world_size = np.array(world_size, dtype=float)
        self.L_init = L_init # initial length of the first cell
        self.L_divide = L_divide
        self.K_nn = K_nn
        self.max_cells = max_cells
        self.fourier_K = fourier_K
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0

        # The action and observation spaces are defined for a single agent.
        # An external policy manager is expected to handle the multi-agent setup.
        self.action_space = spaces.Discrete(3)  # 0: dormant, 1: grow, 2: divide
        # Observation per cell (4 features):
        #  - rel_length: current length / division length
        #  - local_density: smoothed local crowding via Gaussian kernel
        #  - pressure_proxy: averaged inverse distance to neighbors
        #  - orientation: cell angle (radians)
        obs_dim = 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Define the target morphology for the reward function.
        self.M_target = {"AR": 0.7, "D": 0.9, "F": np.zeros(self.fourier_K)}  # target morphology metrics
        self.reset()

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        Resets the environment to its initial state.

        A new colony is started with a single cell at the center of the world.

        Returns:
            Tuple[np.ndarray, dict]:
                - The initial observation for the single agent.
                - An empty info dictionary.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        cx, cy = 0.5 * self.world_size # Start in the center
        first = StickCell(pos=np.array([cx, cy], dtype=float), theta=0.0, length=self.L_init)
        self.cells: List[StickCell] = [first]
        self._recent_divisions = {}  # Track divisions for reward calculation
        obs = self._gather_obs()
        return obs, {} # empty info

    def step(self, action):
        """
        Advances the environment by one timestep.

        Args:
            action (list): A list of action dictionaries, one for
                                      each cell in `self.cells`.

        Returns:
            Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
                - obs: The new observations for each agent.
                - rewards: The rewards for each agent.
                - terminated: Whether the episode has ended (e.g. goal reached).
                - truncated: Whether the episode was cut short (e.g. time limit).
                - info: A dictionary with auxiliary information.
        """
        actions_per_agent = action
        # actions_per_agent must be a list aligned with self.cells
        if len(actions_per_agent) != len(self.cells):
            raise ValueError("Number of actions must match number of cells.")
        # apply actions (rotation, growth, mark division)
        for cell, a in zip(self.cells, actions_per_agent):
            cell.age += self.dt
            if a == 1: # grow
                cell.length += 0.1 * self.dt
            elif a == 2: # divide
                if cell.length >= self.L_divide:
                    cell.pending_divide = True
        # relax overlaps
        self._relax_positions(max_iters=12)
        # handle divisions
        new_cells = []
        remaining_cells = []
        for cell in self.cells:
            if not cell.pending_divide:
                remaining_cells.append(cell)
                continue
            # Divide the cell
            jitter = self.rng.normal(0, 0.1)
            L_new = cell.length / 2.0
            offset_dist = L_new / 2.0
            offset = offset_dist * np.array([math.cos(cell.theta), math.sin(cell.theta)])
            
            c1 = StickCell(pos=cell.pos + offset, theta=cell.theta + jitter, length=L_new, just_divided=True)
            c2 = StickCell(pos=cell.pos - offset, theta=cell.theta - jitter, length=L_new, just_divided=True)
            new_cells.extend([c1, c2])
            self._recent_divisions[id(c1)] = self.t
            self._recent_divisions[id(c2)] = self.t
        self.cells = remaining_cells + new_cells
        # time marches on
        self.t += 1
        # gather observations, rewards, and check for termination
        obs = self._gather_obs()
        rewards = self._compute_rewards()
        terminated, truncated = self._check_done()
        info = {"n_cells": len(self.cells)}
        return obs, rewards, terminated, truncated, info

    def _gather_obs(self):
        """
        Gathers observations for all cells in the colony.

        Returns:
            np.ndarray: An array of observations, one row per cell.
        """
        if not self.cells:
            return np.array([])
        centers = np.array([c.pos for c in self.cells])
        return np.array([self._obs_for_cell(i, centers) for i in range(len(self.cells))])

    def _obs_for_cell(self, idx, centers):
        """
        Compute observation for a single cell with 4 features:
        - rel_length: current length / division length
        - local_density: smoothed local crowding using a Gaussian kernel
        - pressure_proxy: averaged inverse distance to neighbors
        - orientation: cell angle in radians
        """
        cell = self.cells[idx]
        # Distances to all cells
        diffs = centers - cell.pos
        dists = np.linalg.norm(diffs, axis=1)

        # Exclude self (distance ~ 0) for neighbor-based metrics
        mask = np.ones(len(dists), dtype=bool)
        mask[idx] = False
        neighbor_dists = dists[mask]

        # 1) Relative length
        rel_length = float(cell.length / max(self.L_divide, 1e-9))

        # 2) Local density (Gaussian kernel smoothing)
        # Use sigma proportional to division length for locality
        if neighbor_dists.size > 0:
            sigma = max(0.25 * self.L_divide, 1e-6)
            weights = np.exp(-0.5 * (neighbor_dists / sigma) ** 2)
            local_density = float(np.sum(weights))
        else:
            local_density = 0.0

        # 3) Pressure proxy: mean inverse distance to neighbors within a cutoff
        if neighbor_dists.size > 0:
            cutoff = 2.0 * self.L_divide
            near = neighbor_dists[neighbor_dists <= cutoff] if cutoff > 0 else neighbor_dists
            if near.size > 0:
                pressure_proxy = float(np.mean(1.0 / (near + 1e-6)))
            else:
                pressure_proxy = 0.0
        else:
            pressure_proxy = 0.0

        # 4) Orientation (raw angle)
        orientation = float(cell.theta)

        return np.array([rel_length, local_density, pressure_proxy, orientation], dtype=np.float32)

    def _relax_positions(self, max_iters=12):
        """
        Iteratively resolves overlaps between cells.

        This is a simple physics simulation where overlapping cells push each
        other apart. The process is repeated for a fixed number of iterations
        to allow forces to propagate through the colony.

        Args:
            max_iters (int): The number of relaxation iterations to perform.
        """
        for _ in range(max_iters):
            for i in range(len(self.cells)):
                for j in range(i + 1, len(self.cells)):
                    c1, c2 = self.cells[i], self.cells[j]
                    p1a, p1b = c1.endpoints()
                    p2a, p2b = c2.endpoints()
                    
                    # Use a fixed radius for collision detection, as sticks have no radius
                    effective_radius = 0.5 

                    _, _, dist = seg_seg_closest_points(p1a, p1b, p2a, p2b)
                    overlap = (2 * effective_radius) - dist
                    
                    if overlap > 0:
                        # Simple linear push-apart
                        direction = c2.pos - c1.pos
                        if np.linalg.norm(direction) < 1e-9:
                            direction = self.rng.random(2) - 0.5
                        
                        direction = unit_vector(direction)
                        push = 0.5 * overlap * direction
                        
                        c1.pos -= push
                        c2.pos += push

    def _compute_rewards(self):
        """
        Computes rewards for all cells based on the colony's morphology.

        The reward has several components:
        - A global reward based on how well the colony's shape matches target
          morphological metrics (aspect ratio, density, Fourier descriptors).
        - A small penalty for existing to encourage growth and division.
        - A bonus for cells that have just divided.

        Returns:
            np.ndarray: An array of rewards, one for each cell.
        """
        N = len(self.cells)
        if N < 2:
            return np.zeros(N)
        
        # --- Global morphology calculation ---
        all_endpoints = np.vstack([c.endpoints() for c in self.cells])
        hull_pts = monotone_chain_convex_hull(all_endpoints)
        
        # Aspect Ratio (AR)
        AR = pca_aspect_ratio(all_endpoints)
        
        # Density (D)
        colony_area = polygon_area(hull_pts)
        
        # Use a fixed radius for density calculation
        effective_radius = 0.5
        cell_area = sum(c.length * 2 * effective_radius for c in self.cells)
        D = cell_area / colony_area if colony_area > 1e-9 else 0.0
        
        # Fourier Descriptors (F)
        F = fourier_descriptor_from_boundary(hull_pts, K=self.fourier_K)
        
        # --- Reward calculation ---
        # Compare current morphology to target
        err_AR = (AR - self.M_target["AR"])**2
        err_D = (D - self.M_target["D"])**2
        err_F = np.mean((F - self.M_target["F"])**2)
        
        # Global reward is inverse of error (higher is better)
        w_AR, w_D, w_F = 1.0, 1.0, 0.5
        global_reward = 1.0 / (1.0 + w_AR*err_AR + w_D*err_D + w_F*err_F)
        
        # --- Per-agent rewards ---
        rewards = np.full(N, global_reward)
        
        # Small penalty for existing (encourages faster growth/division)
        rewards -= 0.01
        
        # Bonus for recent divisions
        for i, cell in enumerate(self.cells):
            if cell.just_divided:
                rewards[i] += 0.5
                cell.just_divided = False # Reset flag
                
        return rewards

    def _check_done(self):
        """
        Checks if the episode should terminate.

        Termination occurs if the number of cells exceeds `max_cells`.
        Truncation is not currently implemented but the hook is here.

        Returns:
            Tuple[bool, bool]: A tuple of (terminated, truncated).
        """
        terminated = len(self.cells) >= self.max_cells
        truncated = False # No time limit for now
        return terminated, truncated

    def render(self, mode="rgb_array", figsize=(6, 6)):
        """
        Render the current colony state with Matplotlib.

        - human: draws to an interactive window and pauses briefly
        - rgb_array: returns an RGB numpy array of the current frame

        The first call initializes a persistent figure/axes; subsequent calls
        reuse them for better performance.
        """
        if mode not in ("human", "rgb_array"):
            raise ValueError("mode must be 'human' or 'rgb_array'")

        # Initialize persistent figure/axes once
        if not hasattr(self, "fig") or self.fig is None or not hasattr(self, "ax"):
            self.fig, self.ax = plt.subplots(figsize=figsize)
            if mode == "human":
                try:
                    plt.ion()
                    mgr = getattr(self.fig.canvas, "manager", None)
                    if mgr is not None and hasattr(mgr, "set_window_title"):
                        mgr.set_window_title("ColonyEnv")
                except Exception:
                    # Backend might not support window title; safe to ignore
                    pass

        ax = self.ax
        if ax is None:
            # Create an axes if missing (defensive guard for atypical backends)
            ax = self.fig.add_subplot(111)
            self.ax = ax
        ax.clear()

        # World bounds and styling
        width, height = self.world_size
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        ax.grid(False)
        ax.set_title(f"Colony Simulation — {len(self.cells)} cells | t={self.t}")

        # Draw convex hull of all endpoints (gives a sense of colony outline)
        if len(self.cells) >= 2:
            try:
                all_endpoints = np.vstack([c.endpoints() for c in self.cells])
                hull_pts = monotone_chain_convex_hull(all_endpoints)
                if len(hull_pts) >= 3:
                    poly = Polygon(hull_pts, closed=True, facecolor="#90caf955", edgecolor="#2196f3", linewidth=1.5)
                    ax.add_patch(poly)
            except Exception:
                # Hull calculation can fail in degenerate cases; ignore
                pass

        # Prepare segments for efficient drawing with LineCollection
        segments = []
        colors = []
        widths = []
        centers_x = []
        centers_y = []
        center_colors = []

        for cell in self.cells:
            p1, p2 = cell.endpoints()
            segments.append([p1, p2])

            if cell.pending_divide:
                c = "#e53935"  # red
                lw = 3.0
            elif cell.just_divided:
                c = "#43a047"  # green
                lw = 2.5
            else:
                c = "#1e88e5"  # blue
                lw = 2.0

            colors.append(c)
            widths.append(lw)
            centers_x.append(cell.pos[0])
            centers_y.append(cell.pos[1])
            center_colors.append(c)

        if segments:
            lc = LineCollection(segments, colors=colors, linewidths=widths, capstyle="round", joinstyle="round")
            ax.add_collection(lc)
            # Small markers at centers
            ax.scatter(centers_x, centers_y, s=10, c=center_colors, alpha=0.6, edgecolors="none")

        # Overlay info box
        info_text = f"time: {self.t}\ncells: {len(self.cells)}"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#777777", alpha=0.85),
        )

        if mode == "human":
            # Draw and briefly pause to update the interactive window
            self.fig.canvas.draw_idle()
            plt.pause(1.0 / max(self.metadata.get("render_fps", 4), 1))
            return None

        # rgb_array: return the pixel buffer
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        # Attempt RGB via getattr to appease static analyzers and support multiple backends
        to_rgb = getattr(self.fig.canvas, "tostring_rgb", None)
        if callable(to_rgb):
            try:
                rgb_obj = to_rgb()
                if isinstance(rgb_obj, (bytes, bytearray, memoryview)):
                    rgb_bytes = bytes(rgb_obj)
                    buf = np.frombuffer(rgb_bytes, dtype=np.uint8)
                    expected = w * h * 3
                    if buf.size != expected and w > 0 and h > 0:
                        # Adjust for DPI scaling by inferring scale factor
                        scale = (buf.size / 3) / (w * h)
                        s = max(scale, 1.0) ** 0.5
                        w2 = int(round(w * s))
                        h2 = int(round(h * s))
                        if w2 * h2 * 3 == buf.size:
                            return buf.reshape(h2, w2, 3)
                    return buf.reshape(h, w, 3)
            except Exception:
                pass
        # Try RGBA buffer and strip alpha
        rgba_bytes = None
        buffer_rgba = getattr(self.fig.canvas, "buffer_rgba", None)
        if callable(buffer_rgba):
            try:
                rgba_obj = buffer_rgba()
                if isinstance(rgba_obj, (bytes, bytearray, memoryview)):
                    rgba_bytes = bytes(rgba_obj)
            except Exception:
                rgba_bytes = None
        if rgba_bytes is None:
            # Try renderer-based access as a fallback (Agg backends)
            try:
                renderer = getattr(self.fig.canvas, "get_renderer", None)
                if callable(renderer):
                    r = renderer()
                    rb = getattr(r, "buffer_rgba", None)
                    if callable(rb):
                        rgba_obj2 = rb()
                        if isinstance(rgba_obj2, (bytes, bytearray, memoryview)):
                            rgba_bytes = bytes(rgba_obj2)
            except Exception:
                rgba_bytes = None
        if rgba_bytes is not None:
            buf_rgba = np.frombuffer(rgba_bytes, dtype=np.uint8)
            expected = w * h * 4
            if buf_rgba.size != expected and w > 0 and h > 0:
                # Adjust for DPI scaling by inferring scale factor
                scale = (buf_rgba.size / 4) / (w * h)
                s = max(scale, 1.0) ** 0.5
                w2 = int(round(w * s))
                h2 = int(round(h * s))
                if w2 * h2 * 4 == buf_rgba.size:
                    rgba = buf_rgba.reshape(h2, w2, 4)
                    return rgba[..., :3]
            rgba = buf_rgba.reshape(h, w, 4)
            return rgba[..., :3]
        # Last resort: return an empty image with correct shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    def close(self):
        """Clean up matplotlib figures."""
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None