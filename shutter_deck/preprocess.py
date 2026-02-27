"""
shutter_deck/preprocess.py — Pillow + NumPy image preprocessing for DINOv2.

This mirrors the previous torchvision transform:
    Resize(224, bicubic) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def _resize_shorter_side(image: Image.Image, target: int) -> Image.Image:
    """Match torchvision Resize(int): scale shorter side to target, preserve aspect."""
    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image with zero dimension.")
    if min(w, h) == target:
        return image
    if w < h:
        new_w = target
        new_h = int(round(h * (target / w)))
    else:
        new_h = target
        new_w = int(round(w * (target / h)))
    return image.resize((new_w, new_h), resample=_BICUBIC)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    if w < size or h < size:
        # Safety net: if resize rounding undershot, force exact dimensions.
        image = image.resize((max(w, size), max(h, size)), resample=_BICUBIC)
        w, h = image.size
    left = (w - size) // 2
    top = (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess_pil(image: Image.Image, size: int = 224) -> np.ndarray:
    """PIL Image -> (3, size, size) float32 normalized tensor-like array."""
    img = image.convert("RGB")
    img = _resize_shorter_side(img, size)
    img = _center_crop(img, size)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
    return np.ascontiguousarray(arr, dtype=np.float32)


def preprocess_batch(images: Iterable[Image.Image], size: int = 224) -> np.ndarray:
    """Iterable[PIL] -> (B, 3, size, size) float32."""
    batch = [preprocess_pil(img, size=size) for img in images]
    if not batch:
        raise ValueError("preprocess_batch() received no images.")
    return np.ascontiguousarray(np.stack(batch, axis=0), dtype=np.float32)
