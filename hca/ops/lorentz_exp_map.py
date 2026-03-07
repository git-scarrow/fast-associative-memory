"""LorentzExpMap: exponential map at origin on Lorentz hyperboloid.

Uses autograd-compatible forward ops (H1/H5 mitigations via clamping).
PyTorch autograd handles the backward pass automatically.
"""

import torch

EPS_NORM = 1e-7
EPS_SINH = 1e-6
SINH_CLAMP = 85.0


def lorentz_exp_map(v: torch.Tensor, c: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponential map at origin: project tangent vectors onto hyperboloid.

    Args:
        v: tangent vectors at origin [..., d], fp32/fp64
        c: curvature (positive scalar or 0-d tensor)

    Returns:
        x_sp: spatial coordinates on hyperboloid [..., d]
        x0: temporal coordinate [...]
    """
    sqrt_c = c ** 0.5 if isinstance(c, float) else c.sqrt()

    v_norm = torch.norm(v, dim=-1, keepdim=True)

    # H1: clamp to prevent sinh/cosh overflow
    max_norm = SINH_CLAMP / sqrt_c
    # Differentiable clamping: scale down v if norm exceeds max
    safe_norm = v_norm.clamp(min=EPS_SINH)
    scale = torch.where(safe_norm > max_norm, max_norm / safe_norm, torch.ones_like(safe_norm))
    v_eff = v * scale
    v_eff_norm = safe_norm * scale  # = min(safe_norm, max_norm)

    sc = sqrt_c * v_eff_norm  # [..., 1]

    # H5: when ||v|| ~ 0, sinh(sc)/sc -> 1 via Taylor: sinh(x)/x = 1 + x^2/6 + ...
    # Use the stable form: for small sc, spatial_scale = 1 + sc^2/6
    # For large sc, spatial_scale = sinh(sc) / sc
    small = (sc < 1e-4)
    sinh_sc = torch.sinh(sc)
    cosh_sc = torch.cosh(sc)

    # sinh(sc) / (sqrt_c * v_eff_norm) = sinh(sc) / sc
    sinc_h = torch.where(small, 1.0 + sc * sc / 6.0, sinh_sc / sc.clamp(min=EPS_SINH))

    x_sp = sinc_h * v_eff  # [..., d]
    x0 = (cosh_sc / sqrt_c).squeeze(-1)  # [...]

    return x_sp, x0


# Keep class interface for backward compatibility with tests
class LorentzExpMap:
    """Wrapper class preserving the .apply() interface expected by tests."""

    @staticmethod
    def apply(v, c):
        return lorentz_exp_map(v, c)
