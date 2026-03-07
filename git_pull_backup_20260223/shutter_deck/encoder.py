"""
shutter_deck/encoder.py — DINOv2 encoder with ONNX Runtime backend (CPU).

Drop-in replacement for the previous torch.hub-based encoder:
- Keeps the DINOv2Encoder methods used by the ingest/query pipeline
- Uses Pillow + NumPy preprocessing (no torchvision dependency)
- Runs inference via ONNX Runtime on CPU for better ARM performance

The pipeline still returns torch.Tensor embeddings because the CAM/retrieval
stack is torch-based, but the encoder forward pass itself does not require
PyTorch model execution when backend="onnxruntime".
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from shutter_deck.preprocess import preprocess_batch as np_preprocess_batch

logger = logging.getLogger(__name__)

MODEL_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


def _resolve_model_dim(model_name: str, output_shape: list[int | str] | tuple[Any, ...] | None) -> int:
    if output_shape and len(output_shape) >= 2 and isinstance(output_shape[-1], int):
        return int(output_shape[-1])
    for key, dim in MODEL_DIMS.items():
        if key in model_name:
            return dim
    raise ValueError(f"Unable to infer embedding dim for model '{model_name}'.")


def _default_onnx_path(model_name: str) -> Path | None:
    name = model_name.lower()
    candidates: list[str] = []
    if name.endswith(".onnx"):
        candidates.append(model_name)
    elif name in MODEL_DIMS:
        # Prefer quantized if present, then fp32 export.
        candidates.extend([
            f"models/{name}_int8.onnx",
            f"models/{name}.onnx",
            f"{name}_int8.onnx",
            f"{name}.onnx",
        ])
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return p
    return Path(candidates[0]) if candidates else None


class ShutterDeckEncoder:
    """DINOv2 encoder supporting ONNX Runtime (default) and optional torch backend."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: str = "cpu",
        *,
        model_path: str | None = None,
        backend: str = "onnxruntime",
        num_threads: int = 4,
        input_size: int = 224,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.num_threads = int(num_threads)
        self.input_size = int(input_size)
        self._dim: int | None = None
        self._torch_model = None
        self._ort_session = None
        self._ort_input_name = "image"
        self._ort_output_name = "embedding"
        self._ort_fixed_batch: int | None = None

        if backend == "onnxruntime":
            self._init_onnxruntime(model_path=model_path)
        elif backend == "torch":
            self._init_torch()
        else:
            raise ValueError(f"Unsupported encoder backend: {backend}")

        logger.info(
            "Encoder ready: backend=%s model=%s dim=%d input=%dx%d",
            self.backend, self.model_name, self.dim, self.input_size, self.input_size,
        )

    def _init_onnxruntime(self, model_path: str | None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is required for backend='onnxruntime'. "
                "Install it in the venv (e.g. pip install onnxruntime)."
            ) from e

        path = Path(model_path) if model_path else _default_onnx_path(self.model_name)
        if path is None:
            raise RuntimeError(
                f"No ONNX model path configured for model_name={self.model_name!r}. "
                "Set [encoder].model_path in config.toml."
            )
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {path}. Export/quantize the model first and update [encoder].model_path."
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, self.num_threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        logger.info("Loading ONNX model %s (threads=%d)", path, self.num_threads)
        sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._ort_session = sess

        # Infer IO names and embedding dim from graph metadata when available.
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        if inputs:
            self._ort_input_name = inputs[0].name
            in_shape = inputs[0].shape
            if in_shape and isinstance(in_shape[0], int):
                self._ort_fixed_batch = int(in_shape[0])
        if outputs:
            self._ort_output_name = outputs[0].name
            self._dim = _resolve_model_dim(self.model_name, outputs[0].shape)
        else:
            self._dim = MODEL_DIMS.get(self.model_name)

    def _init_torch(self) -> None:
        logger.info("Loading %s on %s via torch.hub", self.model_name, self.device)
        model = torch.hub.load("facebookresearch/dinov2", self.model_name, pretrained=True)
        model = model.to(torch.device(self.device)).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._torch_model = model
        self._dim = MODEL_DIMS.get(self.model_name)
        if self._dim is None:
            with torch.inference_mode():
                dummy = torch.randn(1, 3, self.input_size, self.input_size, device=self.device)
                out = model(dummy)
            self._dim = int(out.shape[-1])

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Encoder dimension not initialized.")
        return self._dim

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL Image -> (1, 3, H, W) float32 torch tensor."""
        arr = np_preprocess_batch([image], size=self.input_size)  # (1,3,H,W)
        return torch.from_numpy(arr)

    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """List[PIL] -> (B, 3, H, W) float32 torch tensor."""
        arr = np_preprocess_batch(images, size=self.input_size)
        return torch.from_numpy(arr)

    def _run_onnx(self, batch_np: np.ndarray) -> np.ndarray:
        if self._ort_session is None:
            raise RuntimeError("ONNX Runtime session not initialized.")
        if batch_np.dtype != np.float32:
            batch_np = batch_np.astype(np.float32, copy=False)
        batch_np = np.ascontiguousarray(batch_np)
        # Some exported models (current PyTorch 2.10 path) are fixed batch=1.
        # Fall back to micro-batching so the ingest pipeline can keep batch_size>1.
        if self._ort_fixed_batch is not None and batch_np.shape[0] != self._ort_fixed_batch:
            if self._ort_fixed_batch <= 0:
                raise RuntimeError(f"Invalid fixed ONNX batch size: {self._ort_fixed_batch}")
            outs = []
            for i in range(0, batch_np.shape[0], self._ort_fixed_batch):
                chunk = batch_np[i:i + self._ort_fixed_batch]
                if chunk.shape[0] != self._ort_fixed_batch:
                    # For batch=1 exports this path always matches exactly.
                    # For other fixed sizes, pad by repeating the last sample and trim.
                    pad = self._ort_fixed_batch - chunk.shape[0]
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)], axis=0)
                    chunk_out = self._ort_session.run([self._ort_output_name], {self._ort_input_name: chunk})[0]
                    outs.append(np.asarray(chunk_out[: chunk.shape[0] - pad], dtype=np.float32))
                else:
                    chunk_out = self._ort_session.run([self._ort_output_name], {self._ort_input_name: chunk})[0]
                    outs.append(np.asarray(chunk_out, dtype=np.float32))
            out = np.concatenate(outs, axis=0)
        else:
            out = self._ort_session.run([self._ort_output_name], {self._ort_input_name: batch_np})[0]
        out = np.asarray(out, dtype=np.float32)
        # L2 normalize in NumPy to avoid extra torch compute in encoder path.
        norms = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.clip(norms, 1e-12, None)
        return out.astype(np.float32, copy=False)

    def warmup(self) -> None:
        """Pay first-run ONNX Runtime initialization cost at startup."""
        if self.backend != "onnxruntime":
            return
        t0 = time.perf_counter()
        _ = self._run_onnx(np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32))
        dt = time.perf_counter() - t0
        logger.info("ONNX warmup complete in %.2fs", dt)

    @torch.no_grad()
    def embed(self, tensor: torch.Tensor | np.ndarray) -> torch.Tensor:
        """(B, 3, H, W) float32 -> (B, D) L2-normalized embeddings."""
        if isinstance(tensor, torch.Tensor):
            batch_np = tensor.detach().cpu().numpy()
        else:
            batch_np = np.asarray(tensor, dtype=np.float32)

        if batch_np.ndim == 3:
            batch_np = batch_np[None, ...]
        if batch_np.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape {batch_np.shape}")

        if self.backend == "onnxruntime":
            out_np = self._run_onnx(batch_np)
            return torch.from_numpy(np.ascontiguousarray(out_np))

        if self._torch_model is None:
            raise RuntimeError("Torch backend selected but model is not initialized.")
        features = self._torch_model(torch.from_numpy(np.ascontiguousarray(batch_np)).to(self.device))
        return F.normalize(features.float().cpu(), dim=-1)

    def encode(self, image_tensor: np.ndarray) -> np.ndarray:
        """User-facing NumPy API: (1,3,H,W) or (3,H,W) -> normalized embedding ndarray."""
        out = self.embed(image_tensor).numpy()
        return out[0] if out.ndim == 2 and out.shape[0] == 1 else out

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        """List[PIL] -> (B, D) L2-normalized embeddings."""
        tensor = self.preprocess_batch(images)
        return self.embed(tensor)

    def embed_file(self, path: Path) -> torch.Tensor:
        """Single image path -> (1, D) normalized embedding."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".arw":
            import rawpy
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            image = Image.fromarray(rgb)
        else:
            image = Image.open(path)
        return self.embed_images([image])


# Preserve existing import sites (`from shutter_deck.encoder import DINOv2Encoder`)
DINOv2Encoder = ShutterDeckEncoder
