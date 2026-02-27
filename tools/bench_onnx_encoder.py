#!/usr/bin/env python3
"""Benchmark an ONNX DINOv2 encoder on CPU with ONNX Runtime."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="Path to .onnx model")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, args.threads)
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(args.model, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    dummy = np.random.randn(1, 3, args.height, args.width).astype(np.float32)
    for _ in range(args.warmup):
        _ = sess.run([output_name], {input_name: dummy})

    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        out = sess.run([output_name], {input_name: dummy})[0]
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"run={i+1:02d} sec={dt:.4f} shape={tuple(out.shape)}")

    med = statistics.median(times)
    mean = statistics.mean(times)
    p95 = max(times) if len(times) < 20 else statistics.quantiles(times, n=20)[18]
    print(f"median={med:.4f}s mean={mean:.4f}s p95_or_max={p95:.4f}s")


if __name__ == "__main__":
    main()

