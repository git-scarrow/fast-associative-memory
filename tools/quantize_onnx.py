#!/usr/bin/env python3
"""Dynamic INT8 quantization for ONNX models via ONNX Runtime."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input_model", help="FP32 ONNX model path")
    p.add_argument("output_model", help="INT8 ONNX model path")
    p.add_argument(
        "--op-types",
        nargs="*",
        default=["MatMul", "Gemm"],
        help="ONNX op types to quantize (default: MatMul Gemm)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=args.input_model,
        model_output=args.output_model,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=args.op_types,
    )
    print(f"Wrote quantized model: {args.output_model}")


if __name__ == "__main__":
    main()
