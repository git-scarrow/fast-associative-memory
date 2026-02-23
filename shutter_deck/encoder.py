"""
shutter_deck/encoder.py — Frozen DINOv2 ViT-L/14 encoder for CPU inference.

Loads the model once, runs all forward passes under torch.no_grad() in float32.
Returns L2-normalized CLS token embeddings (1024-d).
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLS_DIM = 1024  # DINOv2 ViT-L/14 output dimension


def build_transform() -> T.Compose:
    """ImageNet normalization pipeline for 224×224 input."""
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DINOv2Encoder:
    """Frozen DINOv2 ViT-L/14 on CPU.

    Parameters
    ----------
    model_name : str
        torch.hub model identifier (default: ``dinov2_vitl14``).
    device : str
        Torch device string (default: ``cpu``).
    """

    def __init__(self, model_name: str = "dinov2_vitl14", device: str = "cpu"):
        logger.info("Loading %s on %s …", model_name, device)
        self.device = torch.device(device)
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True,
        )
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.transform = build_transform()
        logger.info("Encoder ready (%.1f M params).",
                     sum(p.numel() for p in self.model.parameters()) / 1e6)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → (1, 3, 224, 224) float32 tensor."""
        return self.transform(image.convert("RGB")).unsqueeze(0)

    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """List of PIL Images → (B, 3, 224, 224) float32 tensor."""
        return torch.stack([self.transform(img.convert("RGB")) for img in images])

    @torch.no_grad()
    def embed(self, tensor: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, 1024) L2-normalized CLS embeddings."""
        features = self.model(tensor.to(self.device))  # (B, 1024)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Convenience: list of PIL Images → (B, 1024) normalized embeddings."""
        tensor = self.preprocess_batch(images)
        return self.embed(tensor)

    def embed_file(self, path: Path) -> torch.Tensor:
        """Single file → (1, 1024) normalized embedding."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".arw", ".dng", ".cr2", ".nef"):
            import rawpy
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            if rgb.ndim == 3 and rgb.shape[2] == 1:
                rgb = rgb.squeeze(2)
            image = Image.fromarray(rgb).convert("RGB")
        else:
            image = Image.open(path)
        return self.embed_images([image])
