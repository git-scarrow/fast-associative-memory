#!/usr/bin/env python3
"""Export DINOv2 (S/B/L 14) from torch.hub to ONNX, optionally at reduced resolution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    p.add_argument("--output", required=True, help="Output .onnx path")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--input-size", type=int, default=224, help="Must be divisible by 14")
    p.add_argument("--dynamic-batch", action="store_true")
    return p.parse_args()


def maybe_resize_pos_embed(model: nn.Module, new_size: int) -> None:
    """Resize positional embeddings for patch grid changes (DINOv2 ViT/14)."""
    if new_size == 224:
        return
    if new_size % 14 != 0:
        raise ValueError("input-size must be divisible by 14 for DINOv2 ViT/14.")
    if not hasattr(model, "pos_embed"):
        raise RuntimeError("Model does not expose pos_embed; cannot resize position embeddings.")

    pos_embed = model.pos_embed  # (1, 1+N, C)
    cls_tok = pos_embed[:, :1, :]
    pos_tok = pos_embed[:, 1:, :]

    n_old = pos_tok.shape[1]
    grid_old = int(n_old ** 0.5)
    grid_new = new_size // 14
    if grid_old * grid_old != n_old:
        raise RuntimeError(f"Unexpected pos_embed token count: {n_old}")

    c = pos_tok.shape[-1]
    pos_tok = pos_tok.reshape(1, grid_old, grid_old, c).permute(0, 3, 1, 2)
    pos_tok = F.interpolate(pos_tok, size=(grid_new, grid_new), mode="bicubic", align_corners=False)
    pos_tok = pos_tok.permute(0, 2, 3, 1).reshape(1, grid_new * grid_new, c)
    model.pos_embed = nn.Parameter(torch.cat([cls_tok, pos_tok], dim=1))


def main() -> None:
    args = parse_args()
    # DINOv2 will prefer xFormers if installed, but xFormers has no CPU path for export tracing.
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} from torch.hub...")
    model = torch.hub.load("facebookresearch/dinov2", args.model, pretrained=True).eval()
    maybe_resize_pos_embed(model, args.input_size)

    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"image": {0: "batch"}, "embedding": {0: "batch"}}

    print(f"Exporting to {out_path} (opset={args.opset}, input={args.input_size})...")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
