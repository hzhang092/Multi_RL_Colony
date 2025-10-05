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

# ---------- Capsule cell ----------

@dataclass
class CapsuleCell:
    """
    A data class representing a single capsule-shaped bacterial cell.

    Attributes:
        center (np.ndarray): The (x, y) coordinates of the cell's geometric center.
        theta (float): The orientation angle in radians.
        L (float): The total length of the cell's central axis.
        r (float): The radius of the hemispherical caps.
        age (float): The number of timesteps the cell has existed.
        pending_divide (bool): A flag set to True when the cell is ready to
                               divide in the next timestep.
        just_divided (bool): A flag set to True for cells created from division
                             in the current timestep, used for reward calculation.
    """
    center: np.ndarray  # shape (2,)
    theta: float # orientation angle (radians)
    L: float  # total length (center-to-pole distance = L/2 in each direction)
    r: float  # radius(constant across cells)
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
        half = 0.5 * self.L
        return (self.center + ux * half, self.center - ux * half)

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
                 r=0.5,
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
            r (float): The radius of the capsule cells.
            L_init (float): The initial length of the first cell.
            L_divide (float): The length at which cells can divide.
            max_cells (int): The maximum number of cells before the episode ends.
            K_nn (int): The number of nearest neighbors each cell observes.
            fourier_K (int): The number of Fourier descriptors for shape analysis.
            seed (Optional[int]): A seed for the random number generator.
        """
        super().__init__()
        self.world_size = np.array(world_size, dtype=float)
        self.r = r
        self.L_init = L_init # initial length of the first cell
        self.L_divide = L_divide
        self.K_nn = K_nn
        self.max_cells = max_cells
        self.fourier_K = fourier_K
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0

        # The action and observation spaces are defined for a single agent.
        # An external policy manager is expected to handle the multi-agent setup.
        self.action_space = spaces.Dict({
            "type": spaces.Discrete(3),  # 0: dormant, 1: grow, 2: divide
            "grow_frac": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        })
        # Observation: [self_features, neighbor_1_features, ..., neighbor_K_features]
        # Self features (5): L, sin(theta), cos(theta), age, local_density
        # Neighbor features (5 per neighbor): rel_x, rel_y, dist, sin(theta), cos(theta)
        obs_dim = 5 + self.K_nn * 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) 

        # Define the target morphology for the reward function.
        self.M_target = {"AR": 0.7, "D": 0.9, "F": np.zeros(self.fourier_K)} # target morphology metrics
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
        first = CapsuleCell(center=np.array([cx, cy], dtype=float), theta=0.0, L=self.L_init, r=self.r)
        self.cells: List[CapsuleCell] = [first]
        self._recent_divisions = {}  # Track divisions for reward calculation
        return self._gather_obs(), {} # empty info

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
            raise ValueError("Length of actions must equal number of cells.")
        # apply actions (rotation, growth, mark division)
        for cell, a in zip(self.cells, actions_per_agent):
            atype = int(a["type"])
            grow_frac = float(a["grow_frac"])
            max_growth = 0.8
            if atype == 1: # Grow action
                dL = max_growth * grow_frac
                cell.L += dL
            if atype == 2 and cell.L >= self.L_divide: # Divide action
                cell.pending_divide = True
            cell.age += 1.0
        # relax overlaps
        self._relax_positions(max_iters=12)
        # perform pending divisions (children placed + jitter)
        divided_cell_rewards = {}  # Track which cells successfully divided for reward
        if any(c.pending_divide for c in self.cells):
            new_cells = []
            parents_to_remove = []
            for i, cell in enumerate(self.cells):
                if cell.pending_divide:
                    cell.pending_divide = False
                    parents_to_remove.append(i)
                    divided_cell_rewards[i] = True  # Mark this cell as having divided
                    # Daughter cells are slightly shorter than half the parent's length.
                    Lc = max(0.48 * cell.L, 0.5 * self.L_divide * 0.9)
                    ux = np.array([math.cos(cell.theta), math.sin(cell.theta)])
                    offset = (Lc/2.0 + 0.01) * ux
                    # Add some random jitter to the orientation of daughter cells.
                    jitter = float(self.rng.normal(scale=0.08))
                    c1 = CapsuleCell(center=cell.center + offset, theta=cell.theta + jitter, L=Lc, r=self.r, just_divided=True)
                    c2 = CapsuleCell(center=cell.center - offset, theta=cell.theta - jitter, L=Lc, r=self.r, just_divided=True)
                    new_cells.extend([c1, c2])
            # Remove parent cells and add new daughter cells
            for idx in sorted(parents_to_remove, reverse=True):
                self.cells.pop(idx)
            self.cells.extend(new_cells)
            # A final relaxation step to accommodate the new cells
            self._relax_positions(max_iters=20)
        
        # Store division info for reward calculation
        self._recent_divisions = divided_cell_rewards
        obs = self._gather_obs()
        rewards = self._compute_rewards()
        terminated, truncated = self._check_done()
        info = {"t": self.t, "n_cells": len(self.cells)}
        self.t += 1
        return obs, rewards, terminated, truncated, info

    def _gather_obs(self):
        """
        Gathers observations for all cells in the colony.

        Returns:
            np.ndarray: A stacked array of observation vectors, one per cell.
        """
        centers = np.array([c.center for c in self.cells]) # cell centers, shape (N, 2)
        obs_list = []
        for i, c in enumerate(self.cells):
            obs_list.append(self._obs_for_cell(i, centers))
        return np.array(obs_list, dtype=np.float32)

    def _obs_for_cell(self, idx, centers):
        """
        Computes the observation vector for a single cell.

        The observation includes the cell's own state (length, orientation, age,
        local density) and the relative state of its K-nearest neighbors.

        Args:
            idx (int): The index of the cell in `self.cells`.
            centers (np.ndarray): Pre-computed array of all cell centers.

        Returns:
            np.ndarray: The observation vector for the specified cell.
        """
        c = self.cells[idx] # c: capsule cell
        # --- Self Features ---
        L_norm = c.L / self.L_divide
        sin_t, cos_t = math.sin(c.theta), math.cos(c.theta)
        age_norm = c.age / 100.0
        # --- Neighbor Features ---
        dists = np.linalg.norm(centers - c.center, axis=1)
        order = np.argsort(dists)
        feats = []
        count = 0
        # Find K-nearest neighbors (excluding self, which is at index 0)
        for j in order[1:self.K_nn+1]:
            rel = centers[j] - c.center
            dist = np.linalg.norm(rel)
            if dist > 1e-8:
                dir_norm = rel / (dist + 1e-9)
            else:
                dir_norm = np.array([0.0, 0.0])
            sin2, cos2 = math.sin(self.cells[j].theta), math.cos(self.cells[j].theta)
            # Relative direction, distance, and neighbor's orientation
            feats.extend([dir_norm[0], dir_norm[1], dist / max(1.0, self.world_size[0]), sin2, cos2])
            count += 1
        # Pad with zeros if there are fewer than K neighbors
        while count < self.K_nn:
            feats.extend([0.0]*5)
            count += 1
        # --- Local Density ---
        local_radius = max(2.0 * c.L, 4.0*c.r)
        local_count = np.sum(dists < local_radius) - 1 # -1 for self
        density = local_count / 8.0 # Normalize density
        # Concatenate all features into a single vector
        vec = np.concatenate([[L_norm, sin_t, cos_t, age_norm, density], feats])
        return vec

    def _relax_positions(self, max_iters=12):
        """
        Resolves physical overlaps between cells using iterative relaxation.

        In each iteration, it checks every pair of cells for overlap. If two
        cells overlap, they are pushed apart along the vector connecting their
        closest points. This process is repeated for a fixed number of
        iterations or until no more overlaps are detected.

        Args:
            max_iters (int): The maximum number of relaxation iterations.
        """
        N = len(self.cells)
        if N <= 1:
            return
        for it in range(max_iters):
            moved = 0
            for i in range(N):
                ci = self.cells[i]
                a0, a1 = ci.endpoints()
                for j in range(i+1, N):
                    cj = self.cells[j]
                    b0, b1 = cj.endpoints()
                    # Find closest points on the central axes of the two cells
                    Pa, Pb, d = seg_seg_closest_points(a0, a1, b0, b1)
                    # The gap is the distance between surfaces
                    gap = d - (ci.r + cj.r)
                    if gap < 0:
                        moved += 1
                        nvec = Pa - Pb
                        nv = unit_vector(nvec)
                        # Push each cell by half of the overlap distance
                        disp = -gap * 0.5 * nv
                        ci.center += disp
                        cj.center -= disp
            if moved == 0:
                break

    def _compute_rewards(self):
        """
        Computes the reward for the current state of the colony.

        The reward has two components:
        1. A global, shared reward based on how well the colony's morphology
           (aspect ratio, density, Fourier descriptors) matches a target.
        2. A small, individual penalty for each cell to discourage undesirable
           states (e.g., being too long without dividing).

        Returns:
            np.ndarray: An array of reward values, one for each cell.
        """
        # --- Global Morphology Metrics ---
        """
        centers = np.array([c.center for c in self.cells])
        AR = pca_aspect_ratio(centers) if len(centers)>=2 else 1.0 # aspect ratio
        # Compute convex hull of all cell endpoints
        endpoints = []
        for c in self.cells:
            p1, p2 = c.endpoints()
            endpoints.append(tuple(p1)); endpoints.append(tuple(p2))
        endpoints = np.array(endpoints)
        hull = monotone_chain_convex_hull(endpoints) if len(endpoints)>0 else endpoints
        A_hull = polygon_area(hull) if len(hull)>=3 else (self.world_size[0]*self.world_size[1])
        D = len(self.cells) / max(1e-6, A_hull) # density
        #F = fourier_descriptor_from_boundary(hull if len(hull)>0 else centers, K=self.fourier_K) # Fourier descriptors

        # --- Compute Global Reward ---
        # The reward is the negative distance to the target morphology.
        ar_t, d_t, f_t = self.M_target["AR"], self.M_target["D"], self.M_target["F"]
        d_morph = 0.0
        if ar_t > 0:
            d_morph += abs(AR - ar_t) / (ar_t + 1e-9)
        if d_t > 0:
            d_morph += abs(D - d_t) / (d_t + 1e-9)
        #if f_t is not None and len(f_t)>0:
            #d_morph += np.linalg.norm(F - f_t)
        R_morph = -1.0 * d_morph
        """# Temporarily disable global morphology reward for simplicity
        R_morph = 0.0
        # Bonus for colony size (up to max_cells)
        colony_size = len(self.cells)
        if colony_size >= self.max_cells:
            R_morph += 0.0  # No global reward for very large colonies
        else:
            R_morph += 0.1 * colony_size  # Simple reward scaling with colony size

        # --- Compute Per-Agent Rewards ---
        per_agent = []
        for c in self.cells:
            # Penalty for being too far from the ideal division length
            L_norm = c.L / (self.L_divide)
            r_len = -0.2 * abs(L_norm - 1.0)
            # Small penalty for age to encourage division
            r_age = -0.05 * min(abs(c.age)/10.0, 1.0)
            # Reward for successful division (for newly created daughter cells)
            r_divide = 1.0 if c.just_divided else 0.0
            # Each agent gets its individual penalties/rewards plus a share of the global reward
            #print(f"Length Reward: {r_len}, Age Reward: {r_age}, Divide Reward: {r_divide}, Morphology Reward: {(R_morph / max(1, len(self.cells)))*0.2}")
            per_agent.append(r_len + r_age + r_divide + (R_morph / max(1, len(self.cells)))*0.2)
        
        # Reset the just_divided flags after computing rewards
        for c in self.cells:
            c.just_divided = False
            
        return np.array(per_agent, dtype=np.float32)

    def _check_done(self):
        """
        Checks if the episode should terminate.

        Termination occurs if the cell count reaches the maximum or if the
        time limit is exceeded.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        terminated = len(self.cells) >= self.max_cells
        truncated = self.t >= 1000
        return terminated, truncated

    def render(self, mode="rgb_array", figsize=(6,6)):
        """
        Renders the current state of the environment.

        Args:
            mode (str): The rendering mode. "rgb_array" returns an image as a
                        numpy array. "human" displays the image using matplotlib.
            figsize (Tuple[int, int]): The size of the figure for rendering.

        Returns:
            Union[np.ndarray, None]: An RGB array if mode is "rgb_array", otherwise None.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, self.world_size[0]); ax.set_ylim(0, self.world_size[1])
        ax.set_aspect('equal', 'box')
        for c in self.cells:
            p1, p2 = c.endpoints()
            xs = [p1[0], p2[0]]; ys=[p1[1], p2[1]]
            # The 'linewidth' is scaled to represent the cell's radius.
            ax.plot(xs, ys, color='black', linewidth=2*c.r)
            mid = c.center
            # Draw an arrow to indicate the cell's orientation.
            ax.arrow(mid[0], mid[1], 0.5*math.cos(c.theta), 0.5*math.sin(c.theta),
                     head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=0.8)
        ax.set_title(f"t={self.t}, cells={len(self.cells)}")
        ax.set_xticks([]); ax.set_yticks([])
        
        if mode == "rgb_array":
            # Use the most reliable method - save to memory and read back
            try:
                import io
                # Force draw the figure
                fig.canvas.draw()
                
                # Save figure to in-memory buffer as PNG
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                           facecolor='white', edgecolor='none', pad_inches=0.1)
                buf.seek(0)
                
                # Read the PNG image back as RGB array
                import matplotlib.image as mpimg
                img = mpimg.imread(buf, format='png')
                
                # Convert to RGB if needed
                if img.shape[-1] == 4:  # RGBA format
                    if img.max() <= 1.0:  # Normalized values
                        rgb_array = (img[:, :, :3] * 255).astype(np.uint8)
                    else:  # Already in 0-255 range
                        rgb_array = img[:, :, :3].astype(np.uint8)
                elif img.shape[-1] == 3:  # RGB format
                    if img.max() <= 1.0:  # Normalized values
                        rgb_array = (img * 255).astype(np.uint8)
                    else:  # Already in 0-255 range
                        rgb_array = img.astype(np.uint8)
                else:
                    # Grayscale - convert to RGB
                    if img.max() <= 1.0:
                        gray = (img * 255).astype(np.uint8)
                    else:
                        gray = img.astype(np.uint8)
                    rgb_array = np.stack([gray, gray, gray], axis=-1)
                
                buf.close()
                plt.close(fig)
                return rgb_array
                
            except Exception as e:
                print(f"Warning: Render failed: {e}")
                plt.close(fig)
                # Return a simple fallback image
                return np.full((400, 400, 3), 200, dtype=np.uint8)  # Light gray image
                
        elif mode == "human":
            plt.show()
            plt.close(fig)
            return None
