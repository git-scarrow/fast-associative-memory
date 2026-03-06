"""LorentzLogMap: custom autograd.Function for logarithmic map on Lorentz hyperboloid."""

import torch
from torch.autograd import Function

EPS_NORM = 1e-7
EPS_Z = 1e-8
GRAD_CLAMP = 1e5


class LorentzLogMap(Function):
    """Logarithmic map at origin: project hyperboloid points to tangent space.

    Input: x_sp (spatial coordinates on hyperboloid) [B, ..., d]
    Output: v (tangent vectors at origin) [B, ..., d]

    Temporal coordinate x_0 is derived from x_sp (H7 mitigation).
    """

    @staticmethod
    def forward(ctx, x_sp, c):
        sqrt_c = c ** 0.5

        # Derive x0 (H7)
        inv_c = 1.0 / c
        x0_sq = inv_c + (x_sp * x_sp).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(x0_sq.clamp(min=1e-14))  # [B,...,1]

        # arg to arcosh: sqrt(c) * x0
        arg = (sqrt_c * x0).clamp(min=1.0 + EPS_Z)  # H4: clamp to prevent gradient explosion

        # Distance from origin
        d_origin = torch.acosh(arg)  # [B,...,1]

        # Spatial norm
        x_sp_norm = torch.norm(x_sp, dim=-1, keepdim=True).clamp(min=EPS_NORM)

        # Tangent vector: scale * x_sp where scale = d_origin / (sqrt_c * x_sp_norm)
        scale = d_origin / (sqrt_c * x_sp_norm)
        v = scale * x_sp

        ctx.save_for_backward(x_sp, x0, arg, d_origin, x_sp_norm, scale)
        ctx.c = c
        return v

    @staticmethod
    def backward(ctx, grad_v):
        x_sp, x0, arg, d_origin, x_sp_norm, scale = ctx.saved_tensors
        c = ctx.c
        sqrt_c = c ** 0.5

        # v = scale * x_sp, where scale = d_origin / (sqrt_c * x_sp_norm)
        # d_origin = acosh(sqrt_c * x0), x0 = sqrt(1/c + ||x_sp||^2)

        # d(v_i)/d(x_sp_j) = d(scale)/d(x_sp_j) * x_sp_i + scale * delta_ij

        # d(scale)/d(x_sp_j) = [d(d_origin)/d(x_sp_j) * x_sp_norm - d_origin * d(x_sp_norm)/d(x_sp_j)] / (sqrt_c * x_sp_norm^2)

        # d(x_sp_norm)/d(x_sp_j) = x_sp_j / x_sp_norm
        # d(d_origin)/d(x_sp_j) = d(acosh(arg))/d(arg) * d(arg)/d(x_sp_j)
        # d(acosh(z))/dz = 1/sqrt(z^2 - 1), clamped (H4)
        # d(arg)/d(x_sp_j) = sqrt_c * d(x0)/d(x_sp_j) = sqrt_c * x_sp_j / x0

        arcosh_grad = 1.0 / torch.sqrt((arg * arg - 1.0).clamp(min=EPS_Z))
        arcosh_grad = arcosh_grad.clamp(max=GRAD_CLAMP)  # H4: clamp gradient

        darg_dxsp = sqrt_c * x_sp / x0.clamp(min=1e-14)  # [B,...,d]
        dd_dxsp = arcosh_grad * darg_dxsp  # [B,...,d]

        dnorm_dxsp = x_sp / x_sp_norm.clamp(min=EPS_NORM)  # [B,...,d]

        # d(scale)/d(x_sp_j) — a vector
        dscale_dxsp = (dd_dxsp * x_sp_norm - d_origin * dnorm_dxsp) / (sqrt_c * x_sp_norm * x_sp_norm).clamp(min=1e-14)

        # d(v_i)/d(x_sp_j) = dscale_j * x_sp_i + scale * delta_ij
        # grad_x_sp_j = sum_i grad_v_i * d(v_i)/d(x_sp_j)
        #             = sum_i grad_v_i * dscale_j * x_sp_i + grad_v_j * scale
        #             = dscale_j * (grad_v . x_sp) + scale * grad_v_j

        gv_dot_xsp = (grad_v * x_sp).sum(dim=-1, keepdim=True)
        grad_x_sp = dscale_dxsp * gv_dot_xsp + scale * grad_v

        return grad_x_sp, None


def lorentz_log_map(x_sp, c):
    """Functional interface: log map at origin."""
    return LorentzLogMap.apply(x_sp, c)
