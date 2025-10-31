# ---------- Geometry helpers (self-contained) ----------
import math
import numpy as np


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


 # main idea: https://github.com/ingallslab/bsim-related/blob/main/bsim_related/data_processing/cell_data_processing.py#L184

def get_local_anisotropy(
    cell_centers: np.ndarray,
    cell_orientations: np.ndarray,
    neighbourhood_range: float,
) -> np.ndarray:
    """
    Compute local anisotropy per cell by averaging projection matrices of
    neighbouring cells' orientation vectors within a distance threshold.

    This mirrors the logic in `envs/utilities/cell_data_processing.py::get_local_anisotropies`
    but provides a NumPy-first API.

    Args:
        cell_centers: Array of shape (N, 2) with (x, y) positions for each cell.
        cell_orientations: Array of shape (N,) with orientation angles (radians).
        neighbourhood_range: Scalar distance threshold; neighbours with center-to-center
                              distance <= neighbourhood_range are included (including self).

    Returns:
        np.ndarray: Array of shape (N,) with the local anisotropy (max real eigenvalue
                    of the mean projection matrix) for each cell.

    Notes:
        - If a cell has zero valid neighbours (including itself), its projection matrix
          remains zero and the anisotropy returned will be 0.0.
        - NaN orientations are ignored and do not contribute to the mean.
    """
    # Validate and coerce inputs
    cell_centers = np.asarray(cell_centers, dtype=float)
    cell_orientations = np.asarray(cell_orientations, dtype=float)

    if cell_centers.ndim != 2 or cell_centers.shape[1] != 2:
        raise ValueError("cell_centers must have shape (N, 2)")
    if cell_orientations.ndim != 1 or cell_orientations.shape[0] != cell_centers.shape[0]:
        raise ValueError("cell_orientations must have shape (N,)")
    if neighbourhood_range < 0:
        raise ValueError("neighbourhood_range must be non-negative")

    N = cell_centers.shape[0]
    anisotropy = np.zeros(N, dtype=float)

    # Pre-compute cos and sin of orientations; mask invalid
    valid_mask = ~np.isnan(cell_orientations)
    cos_t = np.zeros(N, dtype=float)
    sin_t = np.zeros(N, dtype=float)
    cos_t[valid_mask] = np.cos(cell_orientations[valid_mask])
    sin_t[valid_mask] = np.sin(cell_orientations[valid_mask])

    # Brute-force O(N^2) neighbourhood scan, sufficient for small colonies
    for i in range(N):
        # Accumulate projection matrices over neighbours
        proj_sum = np.zeros((2, 2), dtype=float)
        valid_neighbours = 0

        bi = cell_centers[i]

        for j in range(N):
            # Distance check
            bj = cell_centers[j]
            dist = math.hypot(bi[0] - bj[0], bi[1] - bj[1])
            if dist > neighbourhood_range:
                continue

            # Orientation validity
            if not valid_mask[j]:
                continue

            c = cos_t[j]
            s = sin_t[j]
            # Outer product of orientation unit vector with itself
            # [[c^2, c*s], [c*s, s^2]]
            proj_sum[0, 0] += c * c
            proj_sum[0, 1] += c * s
            proj_sum[1, 0] += c * s
            proj_sum[1, 1] += s * s
            valid_neighbours += 1

        if valid_neighbours > 0:
            proj_mean = proj_sum / float(valid_neighbours)
            # Eigenvalues of 2x2 symmetric matrix are real; take max
            evals = np.linalg.eigvals(proj_mean).real
            anisotropy[i] = float(np.max(evals))
        else:
            anisotropy[i] = 0.0

    return anisotropy