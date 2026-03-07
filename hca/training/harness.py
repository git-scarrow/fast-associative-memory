"""HCA training harness: minimal Transformer with HCAAttention on WikiText-103."""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.hca_attention import HCAAttention
from ..modules.radial_tracker import RadialPositionTracker
from ..modules.qos_hub import GeometryQoSHub
from .optimizer import build_optimizer

logger = logging.getLogger(__name__)


class HCATransformerLayer(nn.Module):
    """Single Transformer encoder layer with HCAAttention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attn = HCAAttention(d_model, n_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Pre-norm Transformer
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x


class HCATransformerLM(nn.Module):
    """Small Transformer language model with HCA attention."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            HCATransformerLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, targets=None):
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) * math.sqrt(self.d_model) + self.pos_embed(positions)

        # Causal mask
        causal_mask = torch.triu(
            torch.full((N, N), float("-inf"), device=input_ids.device),
            diagonal=1,
        )

        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)

        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    def get_alpha_s_per_layer(self):
        """Return alpha_s from each layer's HCAAttention."""
        return [layer.self_attn.get_alpha_s() for layer in self.layers]

    def get_radial_reg_loss(self):
        """Compute radial regularization loss across all layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            attn = layer.self_attn
            if attn._cached_x0 is not None:
                c = attn.curvature(None)
                x0 = attn._cached_x0
                x0_target = math.cosh(8.0) / (c.item() ** 0.5)
                total = total + 0.01 * ((x0 - x0_target) ** 2).mean()
        return total


def train_hca(
    steps: int = 100,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    seq_len: int = 128,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_wikitext: bool = True,
    qos_cadence: int = 100,
):
    """Run HCA training micro-benchmark.

    Args:
        steps: number of training steps
        d_model: model dimension
        n_heads: number of attention heads
        n_layers: number of layers
        seq_len: sequence length
        batch_size: batch size
        lr: learning rate
        device: torch device
        use_wikitext: if True, use WikiText-103; else use random data
        qos_cadence: QoS update cadence (k)

    Returns:
        dict with training metrics
    """
    logger.info(f"Training HCA: steps={steps}, d_model={d_model}, n_heads={n_heads}, "
                f"n_layers={n_layers}, seq_len={seq_len}, device={device}")

    # Load data
    if use_wikitext:
        try:
            train_data = _load_wikitext(seq_len, device)
        except Exception as e:
            logger.warning(f"WikiText-103 load failed ({e}), falling back to random data")
            use_wikitext = False

    if not use_wikitext:
        vocab_size = 10000
        train_data = torch.randint(0, vocab_size, (1000, seq_len + 1), device=device)
    else:
        vocab_size = train_data.max().item() + 1

    model = HCATransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=seq_len,
    ).to(device)

    optimizer = build_optimizer(model, lr=lr)
    qos_hub = GeometryQoSHub(n_layers=n_layers)
    radial_tracker = RadialPositionTracker(n_layers, n_heads, seq_len)

    metrics = {"loss": [], "alpha_s": [], "c_eff": [], "gamma": [], "qos_state": []}

    for step in range(1, steps + 1):
        # Sample batch
        idx = torch.randint(0, len(train_data), (batch_size,))
        batch = train_data[idx]
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        # Forward
        _, loss = model(input_ids, targets)

        # Radial regularization
        rad_loss = model.get_radial_reg_loss()
        total_loss = loss + rad_loss

        # Check for NaN/Inf
        if not torch.isfinite(total_loss):
            logger.error(f"Step {step}: loss is {total_loss.item()}, aborting")
            metrics["kill_reason"] = "nan_loss"
            break

        # Backward
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Collect metrics
        alpha_s_per_layer = model.get_alpha_s_per_layer()
        c_eff_vals = [layer.self_attn.get_c_eff() for layer in model.layers]
        gamma_vals = [layer.self_attn.get_gamma() for layer in model.layers]

        metrics["loss"].append(loss.item())
        metrics["alpha_s"].append(alpha_s_per_layer)
        metrics["c_eff"].append(c_eff_vals)
        metrics["gamma"].append(gamma_vals)

        # QoS + Hebbian pull at cadence k
        if step % qos_cadence == 0:
            qos_hub.update(step, alpha_s_per_layer, model)
            metrics["qos_state"].append(qos_hub.get_states())
            logger.info(
                f"Step {step}: loss={loss.item():.4f} "
                f"alpha_s={[f'{a:.3f}' for a in alpha_s_per_layer]} "
                f"qos={qos_hub.get_states()}"
            )
        elif step % 10 == 0:
            logger.info(f"Step {step}: loss={loss.item():.4f}")

    # Summary
    if metrics["loss"]:
        metrics["final_loss"] = metrics["loss"][-1]
        metrics["loss_decreased"] = metrics["loss"][-1] < metrics["loss"][0] if len(metrics["loss"]) > 1 else False
        metrics["final_alpha_s"] = metrics["alpha_s"][-1] if metrics["alpha_s"] else None

    return metrics


def _load_wikitext(seq_len: int, device: str):
    """Load WikiText-103 via datasets library."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Simple character-level tokenization for prototype
    # (full tokenizer is out of scope for DS-11)
    text = " ".join([x for x in ds["text"] if len(x.strip()) > 0][:50000])

    # Build vocab from most common chars
    from collections import Counter
    char_counts = Counter(text)
    vocab = ["<pad>", "<unk>"] + [c for c, _ in char_counts.most_common(9998)]
    char2idx = {c: i for i, c in enumerate(vocab)}

    # Encode
    encoded = [char2idx.get(c, 1) for c in text]
    data = torch.tensor(encoded, dtype=torch.long, device=device)

    # Chunk into sequences
    n_seqs = len(data) // (seq_len + 1)
    data = data[:n_seqs * (seq_len + 1)].view(n_seqs, seq_len + 1)

    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = train_hca(steps=100, use_wikitext=False)
    print(f"\nFinal loss: {metrics.get('final_loss', 'N/A')}")
    print(f"Loss decreased: {metrics.get('loss_decreased', 'N/A')}")
    print(f"Final alpha_s: {metrics.get('final_alpha_s', 'N/A')}")
