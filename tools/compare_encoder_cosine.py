#!/usr/bin/env python3
"""
Compare PyTorch DINOv2 vs ONNX Runtime encoder outputs on real images.

Reports per-image cosine similarity and summary stats to validate ONNX/INT8 drift.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shutter_deck.preprocess import preprocess_batch as np_preprocess_batch


def _load_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".arw":
        import rawpy
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        return Image.fromarray(rgb)
    return Image.open(path).convert("RGB")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="Image paths (.ARW/.JPG)")
    p.add_argument("--torch-model", default="dinov2_vits14")
    p.add_argument("--onnx-model", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--input-size", type=int, default=224)
    return p.parse_args()


def _load_onnx_session(path: str, threads: int):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, threads)
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.images]
    os.environ.setdefault("XFORMERS_DISABLED", "1")

    torch_model = torch.hub.load("facebookresearch/dinov2", args.torch_model, pretrained=True).eval()
    sess, input_name, output_name = _load_onnx_session(args.onnx_model, args.threads)

    scores = []
    for path in paths:
        img = _load_image(path)
        batch_np = np_preprocess_batch([img], size=args.input_size)
        with torch.inference_mode():
            ref = torch_model(torch.from_numpy(batch_np))
            ref = F.normalize(ref.float(), dim=-1).cpu().numpy()[0]

        out = sess.run([output_name], {input_name: batch_np})[0]
        out = np.asarray(out, dtype=np.float32)
        out = out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12, None)
        test = out[0]

        cos = float(np.dot(ref, test) / (np.linalg.norm(ref) * np.linalg.norm(test)))
        scores.append(cos)
        print(f"{path}: cosine={cos:.6f}")

    print(
        "summary:",
        f"n={len(scores)}",
        f"median={statistics.median(scores):.6f}",
        f"mean={statistics.mean(scores):.6f}",
        f"min={min(scores):.6f}",
    )


if __name__ == "__main__":
    main()
