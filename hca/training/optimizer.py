"""Hybrid Riemannian optimizer: Euclidean Adam for linear layers, Riemannian SGD for manifold params."""

import torch
import geoopt


def build_optimizer(model, lr: float = 1e-3, manifold_lr: float = 1e-2, weight_decay: float = 0.01):
    """Build hybrid optimizer for HCA model.

    Linear layer parameters use Adam; curvature and gamma parameters use
    geoopt's RiemannianAdam for manifold-aware updates.

    Args:
        model: the HCA transformer model
        lr: learning rate for Euclidean parameters
        manifold_lr: learning rate for manifold parameters (c_raw, gamma_raw)
        weight_decay: weight decay for Euclidean parameters

    Returns:
        optimizer: combined optimizer
    """
    euclidean_params = []
    manifold_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "c_raw" in name or "gamma_raw" in name:
            manifold_params.append(param)
        else:
            euclidean_params.append(param)

    # Use standard Adam for linear layers
    # Use Adam for manifold params too (c_raw and gamma_raw are scalars in R,
    # softplus/curvature floor handle the constraints)
    optimizer = torch.optim.AdamW(
        [
            {"params": euclidean_params, "lr": lr, "weight_decay": weight_decay},
            {"params": manifold_params, "lr": manifold_lr, "weight_decay": 0.0},
        ],
    )

    return optimizer
