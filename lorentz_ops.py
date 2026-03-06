"""
lorentz_ops.py — Lorentz (hyperboloid) manifold primitives.

All operations use the Lorentz model with signature (-,+,...,+).
Points x on the hyperboloid satisfy <x,x>_L = -1/c where c is the curvature.

Convention: x[..., 0] is the temporal coordinate, x[..., 1:] are spatial.
"""

import torch
import torch.nn.functional as F

EPS_NORM = 1e-7


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product: <x,y>_L = -x_0*y_0 + sum(x_i*y_i, i>0)."""
    time_part = -x[..., 0:1] * y[..., 0:1]
    space_part = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return (time_part + space_part).squeeze(-1)


def lorentz_norm_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared Lorentz norm: <x,x>_L."""
    return lorentz_inner(x, x)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Geodesic distance on the hyperboloid.

    d(x,y) = (1/sqrt(c)) * arccosh(-c * <x,y>_L)
    """
    inner = lorentz_inner(x, y)
    # On the hyperboloid, -c*<x,y>_L >= 1. Clamp for numerical safety.
    return torch.acosh((-c * inner).clamp(min=1.0 + EPS_NORM)) / (c ** 0.5)


def lorentz_distance_batch(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Pairwise Lorentz distance: x (B, d+1), y (N, d+1) -> (B, N).

    Uses matrix multiply for the spatial part and outer product for temporal.
    """
    # <x,y>_L = -x_0*y_0 + x_spatial @ y_spatial^T
    time_part = -x[:, 0:1] @ y[:, 0:1].T  # (B, N)
    space_part = x[:, 1:] @ y[:, 1:].T     # (B, N)
    inner = time_part + space_part          # (B, N)
    return torch.acosh((-c * inner).clamp(min=1.0 + EPS_NORM)) / (c ** 0.5)


def exp_map_origin(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map at the origin of the hyperboloid.

    Maps a tangent vector v in R^d to a point on the hyperboloid in R^{d+1}.

    exp_0(v) = (cosh(sqrt(c)*||v||), sinh(sqrt(c)*||v||) * v / (sqrt(c)*||v||))

    Args:
        v: Tangent vectors in R^d, shape (..., d).
        c: Curvature parameter (positive).

    Returns:
        Points on hyperboloid, shape (..., d+1).
    """
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS_NORM)  # (..., 1)
    scaled_norm = sqrt_c * v_norm

    time_coord = torch.cosh(scaled_norm)           # (..., 1)
    space_factor = torch.sinh(scaled_norm) / (sqrt_c * v_norm)  # (..., 1)
    space_coords = space_factor * v                 # (..., d)

    return torch.cat([time_coord, space_coords], dim=-1)  # (..., d+1)


def log_map_origin(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Logarithmic map at the origin: hyperboloid -> tangent space at origin.

    log_0(x) = arccosh(sqrt(c)*x_0) * x_spatial / (sqrt(c) * ||x_spatial||)

    But more precisely:
    log_0(x) = d(o, x) * x_spatial / ||x_spatial||_2

    where d(o, x) = arccosh(x_0) for c=1.

    Args:
        x: Points on hyperboloid, shape (..., d+1).
        c: Curvature parameter.

    Returns:
        Tangent vectors at origin, shape (..., d).
    """
    sqrt_c = c ** 0.5
    x_time = x[..., 0:1]   # (..., 1)
    x_space = x[..., 1:]   # (..., d)
    x_space_norm = x_space.norm(dim=-1, keepdim=True).clamp(min=EPS_NORM)

    # Distance from origin: arccosh(sqrt(c) * x_0 * sqrt(c)) / sqrt(c)
    # For c=1: arccosh(x_0)
    dist = torch.acosh((c * x_time).clamp(min=1.0 + EPS_NORM)) / sqrt_c  # (..., 1)

    return dist * x_space / x_space_norm


def project_to_hyperboloid(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Project a point to the hyperboloid by fixing the temporal coordinate.

    Given spatial coordinates x[..., 1:], set x[..., 0] = sqrt(1/c + ||x_spatial||^2).
    This ensures <x,x>_L = -1/c.
    """
    space = x[..., 1:]
    space_sq = (space * space).sum(dim=-1, keepdim=True)
    time_coord = torch.sqrt(1.0 / c + space_sq)
    return torch.cat([time_coord, space], dim=-1)


def frechet_mean_tangent(points: torch.Tensor, weights: torch.Tensor = None,
                          c: float = 1.0, max_iter: int = 5) -> torch.Tensor:
    """Approximate Fréchet mean via tangent-space averaging at origin.

    1. Log-map all points to tangent space at origin.
    2. Weighted Euclidean mean in tangent space.
    3. Exp-map result back to hyperboloid.

    This is a first-order approximation that works well when points
    are clustered (which prototypes within a class typically are).
    """
    if weights is None:
        weights = torch.ones(points.shape[0], device=points.device)
    weights = weights / weights.sum()

    tangent_vecs = log_map_origin(points, c)  # (N, d)
    mean_tangent = (weights.unsqueeze(-1) * tangent_vecs).sum(dim=0)  # (d,)
    return exp_map_origin(mean_tangent.unsqueeze(0), c).squeeze(0)


def sign_flip(x: torch.Tensor) -> torch.Tensor:
    """Negate temporal coordinate for FAISS IP compatibility.

    After sign-flip, standard inner product <x', y>_E = <x, y>_L.
    Maximizing Euclidean IP = maximizing Lorentz IP = minimizing distance.
    """
    out = x.clone()
    out[..., 0] = -out[..., 0]
    return out


def lorentz_similarity(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Similarity proxy: negative Lorentz distance.

    Higher = more similar (analogous to cosine similarity).
    """
    return -lorentz_distance(x, y, c)


def lorentz_similarity_batch(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Batch similarity: x (B, d+1), y (N, d+1) -> (B, N). Higher = closer."""
    return -lorentz_distance_batch(x, y, c)
