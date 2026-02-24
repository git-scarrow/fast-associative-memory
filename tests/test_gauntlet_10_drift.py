"""
tests/test_gauntlet_10_drift.py — G10 Drift Benchmark with Full Stack.

Confirms that the full-stack combination (Quantized Storage + NSTP + Adapter)
preserves the non-destructive slot allocation property via synthetic embeddings.

Success criterion (from issue spec):
  Backward Transfer (BWT) > -5%  (No Catastrophic Forgetting)

Kill condition (G20):
  BWT < -5% on the Drift Benchmark with full stack enabled.
"""

import sys
import os

import pytest
import torch
import torch.nn.functional as F

# Allow import from repo root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_associative_memory import FastAssociativeMemory
from adapter import MetricAdapter
from nstp import NSTPController

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM = 64          # embedding dimension for synthetic data
ADAPTER_DIM = 32        # projected dimension after MetricAdapter
N_CLASSES = 5           # number of classes in the drift test
N_SAMPLES_PER_CLASS = 40  # samples per class per phase
BATCH_SIZE = 20         # batch size for learn_local calls
BWT_THRESHOLD = -5.0    # G20 kill condition: BWT must stay above this


def _eval_embeds(fam, embeds, labels, batch_size=128):
    """Evaluate FAM accuracy on pre-extracted embeddings, return accuracy (%)."""
    fam.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(embeds), batch_size):
            preds = fam(embeds[i:i + batch_size]).argmax(1)
            correct += (preds == labels[i:i + batch_size]).sum().item()
    return 100.0 * correct / len(labels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_class_embeddings(n_classes, n_samples, dim, seed):
    """Generate well-separated class embeddings on the unit sphere.

    Each class has a tight cluster around a random unit-vector mean, so
    cosine similarity between classes is close to 0.  Two calls with
    different seeds produce statistically independent distributions,
    simulating a concept drift.
    """
    torch.manual_seed(seed)
    class_means = F.normalize(torch.randn(n_classes, dim), dim=-1)
    embeds, labels = [], []
    for c in range(n_classes):
        samples = class_means[c] + 0.02 * torch.randn(n_samples, dim)
        embeds.append(F.normalize(samples, dim=-1))
        labels.extend([c] * n_samples)
    return torch.cat(embeds), torch.tensor(labels)


def _build_full_stack_fam():
    """Construct FastAssociativeMemory with Quantized Storage + NSTP + Adapter."""
    adapter = MetricAdapter(input_dim=INPUT_DIM, output_dim=ADAPTER_DIM)
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
    return FastAssociativeMemory(
        input_dim=INPUT_DIM,
        value_dim=N_CLASSES,
        core_entries=500,
        core_vigilance=0.7,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=10,
        inference_temp=0.05,
        use_bfloat16=True,   # Quantized Storage
        use_lfu=True,
        adapter=adapter,      # Adapter
        nstp=nstp,            # NSTP
    )


# ---------------------------------------------------------------------------
# G10 Drift tests
# ---------------------------------------------------------------------------

class TestG10DriftFullStack:
    """G10 Concept Drift benchmark: full stack must not cause catastrophic forgetting."""

    def test_bwt_above_kill_threshold(self):
        """BWT > -5%: Quantized Storage + NSTP + Adapter must not erase clean knowledge."""
        fam = _build_full_stack_fam()
        clean_embeds, clean_labels = _make_class_embeddings(
            N_CLASSES, N_SAMPLES_PER_CLASS, INPUT_DIM, seed=0
        )
        drift_embeds, drift_labels = _make_class_embeddings(
            N_CLASSES, N_SAMPLES_PER_CLASS, INPUT_DIM, seed=1
        )

        # Phase 1: learn clean domain
        for i in range(0, len(clean_embeds), BATCH_SIZE):
            fam.learn_local(clean_embeds[i:i + BATCH_SIZE],
                            clean_labels[i:i + BATCH_SIZE])
        acc_t1 = _eval_embeds(fam, clean_embeds, clean_labels)

        # Phase 2: learn drift domain — new slots should be allocated non-destructively
        for i in range(0, len(drift_embeds), BATCH_SIZE):
            fam.learn_local(drift_embeds[i:i + BATCH_SIZE],
                            drift_labels[i:i + BATCH_SIZE])
        acc_t2 = _eval_embeds(fam, clean_embeds, clean_labels)

        bwt = acc_t2 - acc_t1
        assert bwt > BWT_THRESHOLD, (
            f"G20 KILL CONDITION MET (Full Stack): BWT={bwt:+.2f}% ≤ {BWT_THRESHOLD}% "
            f"(T1={acc_t1:.1f}%, T2={acc_t2:.1f}%)"
        )

    def test_new_slots_allocated_for_drift_domain(self):
        """Non-destructive slot allocation: drift learning must add new prototypes."""
        fam = _build_full_stack_fam()
        clean_embeds, clean_labels = _make_class_embeddings(
            N_CLASSES, N_SAMPLES_PER_CLASS, INPUT_DIM, seed=0
        )
        drift_embeds, drift_labels = _make_class_embeddings(
            N_CLASSES, N_SAMPLES_PER_CLASS, INPUT_DIM, seed=1
        )

        # Phase 1
        for i in range(0, len(clean_embeds), BATCH_SIZE):
            fam.learn_local(clean_embeds[i:i + BATCH_SIZE],
                            clean_labels[i:i + BATCH_SIZE])
        slots_after_clean = int(fam.core_cam.occupied.sum().item())

        # Phase 2 — drift embeddings are orthogonal to clean, so vigilance misses
        for i in range(0, len(drift_embeds), BATCH_SIZE):
            fam.learn_local(drift_embeds[i:i + BATCH_SIZE],
                            drift_labels[i:i + BATCH_SIZE])
        slots_after_drift = int(fam.core_cam.occupied.sum().item())

        assert slots_after_drift > slots_after_clean, (
            f"No new slots allocated for drift domain: "
            f"{slots_after_clean} → {slots_after_drift} (quantization may have "
            f"collapsed vigilance gate)"
        )

    def test_full_stack_forward_output_shape(self):
        """FAM with full stack must return (B, n_classes) logits."""
        fam = _build_full_stack_fam()
        x = F.normalize(torch.randn(8, INPUT_DIM), dim=-1)
        ids = torch.randint(0, N_CLASSES, (8,))
        fam.learn_local(x, ids)
        out = fam(x)
        assert out.shape == (8, N_CLASSES)

    def test_nstp_does_not_corrupt_forward(self):
        """NSTP must not raise errors or produce NaN/Inf in forward with bfloat16."""
        fam = _build_full_stack_fam()
        x = F.normalize(torch.randn(16, INPUT_DIM), dim=-1)
        ids = torch.randint(0, N_CLASSES, (16,))
        fam.learn_local(x, ids)
        out = fam(x)
        assert not out.isnan().any(), "NaN in FAM output with NSTP + bfloat16"
        assert not out.isinf().any(), "Inf in FAM output with NSTP + bfloat16"

    def test_quantized_keys_stored_in_bfloat16(self):
        """Quantized Storage: keys must be stored as bfloat16."""
        fam = _build_full_stack_fam()
        x = F.normalize(torch.randn(4, INPUT_DIM), dim=-1)
        ids = torch.randint(0, N_CLASSES, (4,))
        fam.learn_local(x, ids)
        assert fam.core_cam.keys.dtype == torch.bfloat16, (
            f"Expected bfloat16 keys, got {fam.core_cam.keys.dtype}"
        )

    def test_adapter_projects_to_reduced_dim(self):
        """Adapter must project INPUT_DIM → ADAPTER_DIM before CAM storage."""
        fam = _build_full_stack_fam()
        assert fam.core_cam.key_dim == ADAPTER_DIM, (
            f"CAM key_dim {fam.core_cam.key_dim} != adapter output_dim {ADAPTER_DIM}"
        )


# ---------------------------------------------------------------------------
# _eval_embeds helper tests
# ---------------------------------------------------------------------------

class TestEvalEmbeds:
    """Unit tests for the _eval_embeds benchmark helper."""

    def test_perfect_accuracy(self):
        """FAM trained and evaluated on the same data should achieve high accuracy."""
        fam = _build_full_stack_fam()
        embeds, labels = _make_class_embeddings(N_CLASSES, 20, INPUT_DIM, seed=99)
        for i in range(0, len(embeds), BATCH_SIZE):
            fam.learn_local(embeds[i:i + BATCH_SIZE], labels[i:i + BATCH_SIZE])
        acc = _eval_embeds(fam, embeds, labels)
        assert acc > 90.0, f"Expected >90% on training data, got {acc:.1f}%"

    def test_returns_float(self):
        fam = _build_full_stack_fam()
        embeds, labels = _make_class_embeddings(N_CLASSES, 10, INPUT_DIM, seed=42)
        fam.learn_local(embeds, labels)
        acc = _eval_embeds(fam, embeds, labels)
        assert isinstance(acc, float)

    def test_range_zero_to_hundred(self):
        fam = _build_full_stack_fam()
        embeds, labels = _make_class_embeddings(N_CLASSES, 10, INPUT_DIM, seed=7)
        fam.learn_local(embeds, labels)
        acc = _eval_embeds(fam, embeds, labels)
        assert 0.0 <= acc <= 100.0
